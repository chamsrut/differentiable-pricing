#include "dp/finite_difference_pde.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dp {
namespace {

void require_finite(const double value, const char* const description) {
    if (!std::isfinite(value)) {
        throw std::overflow_error(description);
    }
}

[[nodiscard]] double payoff(const OptionType option_type, const double spot, const double strike) {
    if (option_type == OptionType::call) {
        return std::max(spot - strike, 0.0);
    }
    return std::max(strike - spot, 0.0);
}

[[nodiscard]] double time_tolerance_for(const double expiry_time) {
    return pde_time_tolerance * std::max(1.0, std::abs(expiry_time));
}

// Diagnostic message formatting only; never used for a returned numeric result.
[[nodiscard]] std::string format_scientific(const double value) {
    std::ostringstream stream;
    stream << std::scientific << std::setprecision(6) << value;
    return stream.str();
}

// Fritsch-Carlson monotone cubic Hermite slopes on a uniform grid. The limiter
// is what makes the dividend jump monotone: a plain cubic through option values
// can overshoot near the payoff kink and manufacture a local arbitrage that no
// later time step removes.
void monotone_slopes(const std::vector<double>& values, const double step,
                     std::vector<double>& slopes) {
    const std::size_t nodes = values.size();
    slopes.assign(nodes, 0.0);
    if (nodes < 2U) {
        return;
    }

    for (std::size_t index = 1U; index + 1U < nodes; ++index) {
        slopes[index] = (values[index + 1U] - values[index - 1U]) / (2.0 * step);
    }
    slopes[0] = (values[1] - values[0]) / step;
    slopes[nodes - 1U] = (values[nodes - 1U] - values[nodes - 2U]) / step;

    for (std::size_t index = 0U; index + 1U < nodes; ++index) {
        const double secant = (values[index + 1U] - values[index]) / step;
        if (secant == 0.0) {
            slopes[index] = 0.0;
            slopes[index + 1U] = 0.0;
            continue;
        }
        double left = slopes[index] / secant;
        double right = slopes[index + 1U] / secant;
        if (left < 0.0) {
            left = 0.0;
        }
        if (right < 0.0) {
            right = 0.0;
        }
        const double magnitude = (left * left) + (right * right);
        if (magnitude > 9.0) {
            const double scale = 3.0 / std::sqrt(magnitude);
            left *= scale;
            right *= scale;
        }
        slopes[index] = left * secant;
        slopes[index + 1U] = right * secant;
    }
}

// Cubic Hermite evaluation on the uniform grid the slopes were built for.
// `position` must lie inside [0, step * (values.size() - 1)]; every caller
// clamps beforehand, so this never extrapolates.
[[nodiscard]] double monotone_evaluate(const std::vector<double>& values,
                                       const std::vector<double>& slopes, const double step,
                                       const double position) {
    const std::size_t intervals = values.size() - 1U;
    const double raw_cell = std::floor(position / step);
    std::size_t cell = 0U;
    if (raw_cell > 0.0) {
        cell = static_cast<std::size_t>(raw_cell);
    }
    if (cell >= intervals) {
        cell = intervals - 1U;
    }

    const double local = (position - (static_cast<double>(cell) * step)) / step;
    const double local_squared = local * local;
    const double local_cubed = local_squared * local;
    const double basis_value_left = (2.0 * local_cubed) - (3.0 * local_squared) + 1.0;
    const double basis_slope_left = local_cubed - (2.0 * local_squared) + local;
    const double basis_value_right = (-2.0 * local_cubed) + (3.0 * local_squared);
    const double basis_slope_right = local_cubed - local_squared;

    return (basis_value_left * values[cell]) + (basis_slope_left * step * slopes[cell]) +
           (basis_value_right * values[cell + 1U]) +
           (basis_slope_right * step * slopes[cell + 1U]);
}

struct SpotDiscretization {
    std::size_t intervals;
    double step;
    double maximum;
    std::size_t strike_index;
};

// The strike is placed exactly on a node. The terminal payoff kink is the
// dominant source of error for a vanilla contract, and letting it fall at an
// arbitrary point inside a cell makes the observed convergence order oscillate
// with the grid. `spot_intervals` and `spot_maximum` are therefore targets, and
// the values actually used are reported back.
[[nodiscard]] SpotDiscretization build_spot_grid(const PdeContract& contract,
                                                 const PdeGrid& grid) {
    const double target_step = grid.spot_maximum / static_cast<double>(grid.spot_intervals);
    const double raw_strike_intervals = contract.strike / target_step;
    auto strike_index = static_cast<std::size_t>(std::llround(raw_strike_intervals));
    if (strike_index == 0U) {
        strike_index = 1U;
    }
    const double step = contract.strike / static_cast<double>(strike_index);
    require_finite(step, "spot step became non-finite");

    auto intervals = static_cast<std::size_t>(std::ceil((grid.spot_maximum / step) - 1.0e-9));
    if (intervals <= strike_index) {
        intervals = strike_index + 1U;
    }
    if (intervals > maximum_pde_spot_intervals) {
        throw std::invalid_argument(
            "strike-aligned spot grid exceeds the finite-difference node limit");
    }

    const double maximum = static_cast<double>(intervals) * step;
    require_finite(maximum, "spot domain maximum became non-finite");
    if (contract.spot >= maximum) {
        throw std::invalid_argument("spot must lie strictly inside the truncated spot domain");
    }
    return {intervals, step, maximum, strike_index};
}

struct TimeDiscretization {
    std::vector<double> aligned;
    std::vector<std::size_t> steps;
    std::size_t total_steps;
};

// Every dividend ex-time and every interior curve knot becomes an endpoint of a
// uniformly stepped segment, so no time step ever straddles a rate change or a
// jump.
[[nodiscard]] TimeDiscretization build_time_grid(const PdeContract& contract,
                                                 const PdeGrid& grid) {
    const double tolerance = time_tolerance_for(contract.expiry_time);
    std::vector<double> candidates;
    candidates.push_back(contract.valuation_time);
    for (const double knot : contract.discount_curve.times) {
        if (knot > contract.valuation_time + tolerance &&
            knot < contract.expiry_time - tolerance) {
            candidates.push_back(knot);
        }
    }
    for (const CashDividend& dividend : contract.dividends.dividends()) {
        candidates.push_back(dividend.ex_time);
    }
    candidates.push_back(contract.expiry_time);
    std::sort(candidates.begin(), candidates.end());

    std::vector<double> aligned;
    aligned.push_back(candidates.front());
    for (const double candidate : candidates) {
        if (candidate > aligned.back() + tolerance) {
            aligned.push_back(candidate);
        }
    }

    // Defence in depth behind PdeContract::validate: a grid with no segment to
    // step over must never reach the marching loop, where it would silently
    // return the terminal payoff.
    if (aligned.size() < 2U) {
        throw std::invalid_argument(
            "valuation and expiry times collapse to a single aligned instant");
    }
    const std::size_t segments = aligned.size() - 1U;
    if (grid.time_steps < segments) {
        throw std::invalid_argument(
            "requested time steps cannot align every dividend ex-time and curve knot");
    }

    const double span = contract.expiry_time - contract.valuation_time;
    std::vector<std::size_t> steps(segments, 1U);
    std::size_t total_steps = 0U;
    for (std::size_t segment = 0U; segment < segments; ++segment) {
        const double length = aligned[segment + 1U] - aligned[segment];
        const double share = static_cast<double>(grid.time_steps) * length / span;
        auto allocated = static_cast<std::size_t>(std::llround(share));
        if (allocated == 0U) {
            allocated = 1U;
        }
        steps[segment] = allocated;
        total_steps += allocated;
    }
    if (total_steps > maximum_pde_time_steps) {
        throw std::invalid_argument("aligned time grid exceeds the finite-difference step limit");
    }
    return {std::move(aligned), std::move(steps), total_steps};
}

struct SolveOutcome {
    std::size_t iterations;
    double absolute_residual;
    double relative_residual;
};

// Natural LCP residual max_i |min(x_i - obstacle_i, (A x - b)_i)|, normalised by
// max(1, ||b||_inf). Without an obstacle it degenerates to ||A x - b||_inf, so
// the same measurement covers the European direct solve.
[[nodiscard]] SolveOutcome measure_residual(const std::vector<double>& lower,
                                            const std::vector<double>& diagonal,
                                            const std::vector<double>& upper,
                                            const std::vector<double>& right_hand_side,
                                            const std::vector<double>& obstacle,
                                            const std::vector<double>& solution,
                                            const bool has_obstacle) {
    const std::size_t nodes = solution.size();
    double absolute = 0.0;
    double scale = 1.0;
    for (std::size_t index = 0U; index < nodes; ++index) {
        scale = std::max(scale, std::abs(right_hand_side[index]));
    }
    for (std::size_t index = 0U; index < nodes; ++index) {
        require_finite(solution[index], "finite-difference solution left double precision");
        const double below = (index == 0U) ? 0.0 : solution[index - 1U];
        const double above = (index + 1U == nodes) ? 0.0 : solution[index + 1U];
        const double linear = (lower[index] * below) + (diagonal[index] * solution[index]) +
                              (upper[index] * above) - right_hand_side[index];
        const double natural =
            has_obstacle ? std::min(solution[index] - obstacle[index], linear) : linear;
        absolute = std::max(absolute, std::abs(natural));
    }
    return {0U, absolute, absolute / scale};
}

// Thomas algorithm for the European step. The system is diagonally dominant by
// construction (the caller rejects a step size that would break it), so no
// pivoting is required.
void solve_tridiagonal(const std::vector<double>& lower, const std::vector<double>& diagonal,
                       const std::vector<double>& upper,
                       const std::vector<double>& right_hand_side,
                       std::vector<double>& upper_scratch, std::vector<double>& rhs_scratch,
                       std::vector<double>& solution) {
    const std::size_t nodes = diagonal.size();
    upper_scratch[0] = upper[0] / diagonal[0];
    rhs_scratch[0] = right_hand_side[0] / diagonal[0];
    for (std::size_t index = 1U; index < nodes; ++index) {
        const double pivot = diagonal[index] - (lower[index] * upper_scratch[index - 1U]);
        if (pivot == 0.0) {
            throw std::overflow_error("finite-difference tridiagonal pivot vanished");
        }
        upper_scratch[index] = upper[index] / pivot;
        rhs_scratch[index] =
            (right_hand_side[index] - (lower[index] * rhs_scratch[index - 1U])) / pivot;
    }
    solution[nodes - 1U] = rhs_scratch[nodes - 1U];
    for (std::size_t index = nodes - 1U; index-- > 0U;) {
        solution[index] = rhs_scratch[index] - (upper_scratch[index] * solution[index + 1U]);
    }
}

// Projected successive over-relaxation for the American obstacle. The sweep
// order is fixed and no reduction is reassociated, so repeated runs are
// bitwise identical.
[[nodiscard]] SolveOutcome solve_psor(const std::vector<double>& lower,
                                      const std::vector<double>& diagonal,
                                      const std::vector<double>& upper,
                                      const std::vector<double>& right_hand_side,
                                      const std::vector<double>& obstacle, const PdeGrid& grid,
                                      const std::size_t aligned_segment,
                                      std::vector<double>& solution) {
    const std::size_t nodes = diagonal.size();
    for (std::size_t index = 0U; index < nodes; ++index) {
        solution[index] = std::max(solution[index], obstacle[index]);
    }

    for (std::size_t iteration = 1U; iteration <= grid.psor_maximum_iterations; ++iteration) {
        for (std::size_t index = 0U; index < nodes; ++index) {
            const double below = (index == 0U) ? 0.0 : solution[index - 1U];
            const double above = (index + 1U == nodes) ? 0.0 : solution[index + 1U];
            const double defect = right_hand_side[index] - (lower[index] * below) -
                                  (diagonal[index] * solution[index]) - (upper[index] * above);
            const double candidate =
                solution[index] + (grid.psor_relaxation * defect / diagonal[index]);
            solution[index] = std::max(candidate, obstacle[index]);
        }
        const SolveOutcome measured =
            measure_residual(lower, diagonal, upper, right_hand_side, obstacle, solution, true);
        if (measured.relative_residual <= grid.psor_tolerance) {
            return {iteration, measured.absolute_residual, measured.relative_residual};
        }
        if (iteration == grid.psor_maximum_iterations) {
            // The diagnostics are interpolated into the message, not only into
            // the getters: pybind11 forwards what() and nothing else, so a
            // Python caller would otherwise see a bare static string.
            throw PdePsorFailure(
                "PSOR did not reach the declared LCP residual within the iteration limit: "
                "aligned segment " +
                    std::to_string(aligned_segment) + ", " + std::to_string(iteration) +
                    " iterations, relative residual " +
                    format_scientific(measured.relative_residual) + " above tolerance " +
                    format_scientific(grid.psor_tolerance),
                aligned_segment, iteration, measured.relative_residual, grid.psor_tolerance);
        }
    }
    throw std::logic_error("PSOR loop exited without a verdict");
}

}  // namespace

// --------------------------------------------------------------------------
// Conventions and validation
// --------------------------------------------------------------------------

SettlementConvention parse_settlement_convention(const std::string_view value) {
    if (value == "cash") {
        return SettlementConvention::cash;
    }
    if (value == "physical") {
        return SettlementConvention::physical;
    }
    throw std::invalid_argument("settlement convention must be either 'cash' or 'physical', got '" +
                                std::string(value) + "'");
}

void PiecewiseLogDiscountCurve::validate() const {
    if (times.size() != log_discounts.size()) {
        throw std::invalid_argument("discount curve times and log discounts must have equal length");
    }
    if (times.size() < 2U) {
        throw std::invalid_argument("discount curve must declare at least two knots");
    }
    if (times.size() > maximum_pde_curve_nodes) {
        throw std::invalid_argument("discount curve exceeds the declared knot limit");
    }
    if (times.front() != 0.0 || log_discounts.front() != 0.0) {
        throw std::invalid_argument("discount curve must begin at (0, 0)");
    }
    for (std::size_t index = 0U; index < times.size(); ++index) {
        if (!std::isfinite(times[index]) || !std::isfinite(log_discounts[index])) {
            throw std::invalid_argument("discount curve values must be finite");
        }
        if (index != 0U && times[index] <= times[index - 1U] + pde_time_tolerance) {
            throw std::invalid_argument("discount curve times must be strictly increasing");
        }
    }
}

std::size_t PiecewiseLogDiscountCurve::segment_index_for(const double time) const {
    validate();
    const double tolerance = time_tolerance_for(times.back());
    if (time < times.front() - tolerance || time > times.back() + tolerance) {
        throw std::invalid_argument(
            "discount curve does not bracket the requested time; extrapolation is refused");
    }
    const auto upper = std::upper_bound(times.begin(), times.end(), time);
    if (upper == times.begin()) {
        return 0U;
    }
    const auto index = static_cast<std::size_t>(std::distance(times.begin(), upper)) - 1U;
    if (index + 1U >= times.size()) {
        return times.size() - 2U;
    }
    return index;
}

double PiecewiseLogDiscountCurve::log_discount(const double time) const {
    const std::size_t segment = segment_index_for(time);
    const double left = times[segment];
    const double right = times[segment + 1U];
    const double weight = (time - left) / (right - left);
    const double rise = log_discounts[segment + 1U] - log_discounts[segment];
    return log_discounts[segment] + (weight * rise);
}

double PiecewiseLogDiscountCurve::discount_factor(const double from_time,
                                                  const double to_time) const {
    const double factor = std::exp(log_discount(to_time) - log_discount(from_time));
    require_finite(factor, "discount factor left double precision");
    return factor;
}

double PiecewiseLogDiscountCurve::segment_rate(const std::size_t segment_index) const {
    validate();
    if (segment_index + 1U >= times.size()) {
        throw std::invalid_argument("discount curve segment index is out of range");
    }
    const double rate = -(log_discounts[segment_index + 1U] - log_discounts[segment_index]) /
                        (times[segment_index + 1U] - times[segment_index]);
    require_finite(rate, "discount curve segment rate left double precision");
    return rate;
}

CashDividendSchedule::CashDividendSchedule(const bool declared, std::vector<CashDividend> dividends)
    : declared_(declared), dividends_(std::move(dividends)) {}

CashDividendSchedule CashDividendSchedule::declared_none() {
    return CashDividendSchedule(true, {});
}

CashDividendSchedule CashDividendSchedule::declared(std::vector<CashDividend> dividends) {
    return CashDividendSchedule(true, std::move(dividends));
}

void CashDividendSchedule::validate(const double valuation_time, const double expiry_time) const {
    if (!declared_) {
        throw std::invalid_argument(
            "cash dividend schedule must be declared explicitly; an empty schedule is a statement, "
            "not a default");
    }
    if (dividends_.size() > maximum_pde_cash_dividends) {
        throw std::invalid_argument("cash dividend schedule exceeds the declared event limit");
    }
    const double tolerance = time_tolerance_for(expiry_time);
    for (std::size_t index = 0U; index < dividends_.size(); ++index) {
        const CashDividend& dividend = dividends_[index];
        if (!std::isfinite(dividend.amount) || dividend.amount <= 0.0) {
            throw std::invalid_argument("cash dividend amount must be finite and positive");
        }
        if (!std::isfinite(dividend.ex_time)) {
            throw std::invalid_argument("cash dividend ex-time must be finite");
        }
        if (dividend.ex_time <= valuation_time + tolerance ||
            dividend.ex_time >= expiry_time - tolerance) {
            throw std::invalid_argument(
                "cash dividend ex-time must lie strictly inside the valuation-expiry interval");
        }
        if (index == 0U) {
            continue;
        }
        const double previous = dividends_[index - 1U].ex_time;
        if (dividend.ex_time < previous) {
            throw std::invalid_argument("cash dividend schedule must be sorted by ex-time");
        }
        if (dividend.ex_time <= previous + tolerance) {
            throw std::invalid_argument("cash dividend schedule must not repeat an ex-time");
        }
    }
}

void PdeContract::validate() const {
    if (option_type != OptionType::call && option_type != OptionType::put) {
        throw std::invalid_argument("unsupported option type");
    }
    if (exercise_style != ExerciseStyle::european && exercise_style != ExerciseStyle::american) {
        throw std::invalid_argument("unsupported exercise style");
    }
    if (!std::isfinite(spot) || spot <= 0.0) {
        throw std::invalid_argument("spot must be finite and positive");
    }
    if (!std::isfinite(strike) || strike <= 0.0) {
        throw std::invalid_argument("strike must be finite and positive");
    }
    if (!std::isfinite(valuation_time) || valuation_time < 0.0) {
        throw std::invalid_argument("valuation time must be finite and non-negative");
    }
    // The margin is not cosmetic. Time-grid construction merges declared times
    // that agree to within this tolerance, so a span below it collapses to a
    // single aligned instant, leaves no segment to step over, and would return
    // the undiscounted terminal payoff while reporting discrete-system success.
    if (!std::isfinite(expiry_time) ||
        expiry_time <= valuation_time + time_tolerance_for(expiry_time)) {
        throw std::invalid_argument(
            "expiry time must be finite and after the valuation time by more than the "
            "time-alignment tolerance");
    }
    if (!std::isfinite(volatility) || volatility <= 0.0) {
        throw std::invalid_argument("volatility must be finite and positive");
    }
    if (!continuous_carry.has_value()) {
        throw std::invalid_argument(
            "continuous carry must be stated explicitly; it has no default");
    }
    if (!std::isfinite(*continuous_carry)) {
        throw std::invalid_argument("continuous carry must be finite");
    }
    discount_curve.validate();
    const double tolerance = time_tolerance_for(expiry_time);
    if (discount_curve.times.front() > valuation_time + tolerance ||
        discount_curve.times.back() < expiry_time - tolerance) {
        throw std::invalid_argument(
            "discount curve must bracket the valuation-expiry interval; extrapolation is refused");
    }
    dividends.validate(valuation_time, expiry_time);
    if (settlement != SettlementConvention::cash && settlement != SettlementConvention::physical) {
        throw std::invalid_argument("unsupported settlement convention");
    }
    if (!std::isfinite(contract_multiplier) || contract_multiplier <= 0.0) {
        throw std::invalid_argument("contract multiplier must be finite and positive");
    }
}

void PdeGrid::validate() const {
    if (spot_intervals < 4U) {
        throw std::invalid_argument("spot grid must declare at least four intervals");
    }
    if (spot_intervals > maximum_pde_spot_intervals) {
        throw std::invalid_argument("spot grid exceeds the finite-difference node limit");
    }
    if (time_steps == 0U) {
        throw std::invalid_argument("time steps must be positive");
    }
    if (time_steps > maximum_pde_time_steps) {
        throw std::invalid_argument("time steps exceed the finite-difference step limit");
    }
    if (!std::isfinite(spot_maximum) || spot_maximum <= 0.0) {
        throw std::invalid_argument("spot domain maximum must be finite and positive");
    }
    if (rannacher_steps > maximum_pde_rannacher_steps) {
        throw std::invalid_argument("Rannacher damping exceeds the declared step limit");
    }
    if (!std::isfinite(psor_tolerance) || psor_tolerance <= 0.0) {
        throw std::invalid_argument("PSOR tolerance must be finite and positive");
    }
    if (!std::isfinite(psor_relaxation) || psor_relaxation <= 0.0 || psor_relaxation >= 2.0) {
        throw std::invalid_argument("PSOR relaxation must lie strictly inside (0, 2)");
    }
    if (psor_maximum_iterations == 0U) {
        throw std::invalid_argument("PSOR iteration limit must be positive");
    }
}

PdePsorFailure::PdePsorFailure(const std::string& message, const std::size_t aligned_segment,
                               const std::size_t iterations, const double residual,
                               const double tolerance)
    : std::runtime_error(message), aligned_segment_(aligned_segment), iterations_(iterations),
      residual_(residual), tolerance_(tolerance) {}

double post_dividend_spot(const double spot, const double amount) {
    return std::max(spot - amount, 0.0);
}

// --------------------------------------------------------------------------
// Solver
// --------------------------------------------------------------------------

PdeResult finite_difference_price(const PdeContract& contract, const PdeGrid& grid) {
    contract.validate();
    grid.validate();
    if (grid.spot_maximum <= std::max(contract.spot, contract.strike)) {
        throw std::invalid_argument(
            "spot domain maximum must exceed both the spot and the strike");
    }

    const SpotDiscretization spot_grid = build_spot_grid(contract, grid);
    const TimeDiscretization time_grid = build_time_grid(contract, grid);
    const bool is_american = contract.exercise_style == ExerciseStyle::american;
    const bool is_call = contract.option_type == OptionType::call;
    const double carry = *contract.continuous_carry;
    const double expiry = contract.expiry_time;
    const std::vector<CashDividend>& dividends = contract.dividends.dividends();
    const double alignment_tolerance = time_tolerance_for(expiry);

    // Constant part of the deep-in-the-money call boundary: the value each
    // remaining dividend removes from the terminal forward, carried back to
    // expiry. The log discounts of the fixed times are evaluated once, so no
    // time step searches the curve.
    std::vector<double> dividend_forward_weights(dividends.size(), 0.0);
    std::vector<double> dividend_log_discounts(dividends.size(), 0.0);
    for (std::size_t index = 0U; index < dividends.size(); ++index) {
        dividend_forward_weights[index] =
            dividends[index].amount * std::exp(-carry * (expiry - dividends[index].ex_time));
        require_finite(dividend_forward_weights[index], "dividend forward weight overflowed");
        dividend_log_discounts[index] =
            contract.discount_curve.log_discount(dividends[index].ex_time);
    }
    const double expiry_log_discount = contract.discount_curve.log_discount(expiry);

    const std::size_t nodes = spot_grid.intervals + 1U;
    std::vector<double> values(nodes, 0.0);
    std::vector<double> obstacle(nodes, 0.0);
    std::vector<double> shifted(nodes, 0.0);
    std::vector<double> slopes(nodes, 0.0);
    std::vector<double> lower(nodes, 0.0);
    std::vector<double> diagonal(nodes, 0.0);
    std::vector<double> upper(nodes, 0.0);
    std::vector<double> right_hand_side(nodes, 0.0);
    std::vector<double> upper_scratch(nodes, 0.0);
    std::vector<double> rhs_scratch(nodes, 0.0);
    std::vector<double> solution(nodes, 0.0);
    std::vector<double> coefficient_lower(nodes, 0.0);
    std::vector<double> coefficient_diagonal(nodes, 0.0);
    std::vector<double> coefficient_upper(nodes, 0.0);

    for (std::size_t index = 0U; index < nodes; ++index) {
        const double node_spot = static_cast<double>(index) * spot_grid.step;
        obstacle[index] = payoff(contract.option_type, node_spot, contract.strike);
        values[index] = obstacle[index];
    }

    // Dividends still ahead of the current time, as a suffix of the schedule.
    std::size_t first_ahead = dividends.size();

    // Deep in the money a call is worth the discounted forward intrinsic, which
    // the discrete dividends reduce by their carried-back present value; a deep
    // out-of-the-money put is worth zero. Both are Dirichlet conditions on the
    // truncated boundary and both are exact only in the limit S_max -> infinity,
    // so domain truncation is a reported sensitivity, not a free choice.
    const auto boundary_value = [&](const double time, const double log_discount_now,
                                    const std::size_t ahead) {
        if (!is_call) {
            return 0.0;
        }
        double value = spot_grid.maximum * std::exp(-carry * (expiry - time));
        for (std::size_t index = ahead; index < dividends.size(); ++index) {
            value -= dividend_forward_weights[index] *
                     std::exp(dividend_log_discounts[index] - log_discount_now);
        }
        value -= contract.strike * std::exp(expiry_log_discount - log_discount_now);
        value = std::max(value, 0.0);
        if (is_american) {
            value = std::max(value, spot_grid.maximum - contract.strike);
        }
        require_finite(value, "upper boundary value left double precision");
        return value;
    };

    PdeResult result{};
    result.solver_status = PdeSolverStatus::discrete_system_converged;
    result.discretization_accuracy = PdeDiscretizationAccuracy::not_assessed;
    result.spot_intervals = spot_grid.intervals;
    result.spot_maximum = spot_grid.maximum;
    result.spot_step = spot_grid.step;
    result.strike_node_index = spot_grid.strike_index;
    result.time_steps = time_grid.total_steps;
    result.rannacher_steps = grid.rannacher_steps;
    result.psor_tolerance = grid.psor_tolerance;
    result.psor_relaxation = grid.psor_relaxation;
    result.aligned_times = time_grid.aligned;

    std::size_t damping_remaining = grid.rannacher_steps;

    // Refreshed once per aligned segment. The log discount is linear in time on
    // a segment because every curve knot is a segment endpoint, so the time
    // stepping never searches the curve.
    double segment_start_time = contract.valuation_time;
    double segment_start_log_discount = 0.0;
    double segment_short_rate = 0.0;

    const auto advance = [&](const double target_time, const double step_size, const double theta,
                             const std::size_t aligned_segment) {
        const double log_discount_now =
            segment_start_log_discount - (segment_short_rate * (target_time - segment_start_time));
        const double implicit_weight = theta * step_size;
        const double explicit_weight = (1.0 - theta) * step_size;
        for (std::size_t index = 0U; index < nodes - 1U; ++index) {
            lower[index] = -implicit_weight * coefficient_lower[index];
            diagonal[index] = 1.0 - (implicit_weight * coefficient_diagonal[index]);
            upper[index] = -implicit_weight * coefficient_upper[index];
            const double below = (index == 0U) ? 0.0 : values[index - 1U];
            const double above = values[index + 1U];
            right_hand_side[index] =
                values[index] + (explicit_weight * ((coefficient_lower[index] * below) +
                                                    (coefficient_diagonal[index] * values[index]) +
                                                    (coefficient_upper[index] * above)));
        }
        lower[nodes - 1U] = 0.0;
        diagonal[nodes - 1U] = 1.0;
        upper[nodes - 1U] = 0.0;
        right_hand_side[nodes - 1U] = boundary_value(target_time, log_discount_now, first_ahead);

        SolveOutcome measured{};
        if (is_american) {
            solution = values;
            measured = solve_psor(lower, diagonal, upper, right_hand_side, obstacle, grid,
                                  aligned_segment, solution);
            ++result.psor_solves;
            result.psor_total_iterations += measured.iterations;
            result.psor_maximum_iterations_used =
                std::max(result.psor_maximum_iterations_used, measured.iterations);
        } else {
            solve_tridiagonal(lower, diagonal, upper, right_hand_side, upper_scratch, rhs_scratch,
                              solution);
            ++result.linear_solves;
            measured = measure_residual(lower, diagonal, upper, right_hand_side, obstacle, solution,
                                        false);
        }
        result.maximum_lcp_residual =
            std::max(result.maximum_lcp_residual, measured.absolute_residual);
        result.maximum_relative_lcp_residual =
            std::max(result.maximum_relative_lcp_residual, measured.relative_residual);
        values.swap(solution);
    };

    for (std::size_t segment = time_grid.steps.size(); segment-- > 0U;) {
        const double segment_start = time_grid.aligned[segment];
        const double segment_end = time_grid.aligned[segment + 1U];
        const std::size_t segment_steps = time_grid.steps[segment];
        const double step_size = (segment_end - segment_start) / static_cast<double>(segment_steps);
        const double rate = contract.discount_curve.segment_rate(
            contract.discount_curve.segment_index_for(0.5 * (segment_start + segment_end)));
        segment_start_time = segment_start;
        segment_start_log_discount = contract.discount_curve.log_discount(segment_start);
        segment_short_rate = rate;

        std::size_t upwinded_rows = 0U;
        for (std::size_t index = 1U; index + 1U < nodes; ++index) {
            const double node = static_cast<double>(index);
            const double diffusion = contract.volatility * contract.volatility * node * node;
            const double convection = (rate - carry) * node;
            double coefficient_below = (0.5 * diffusion) - (0.5 * convection);
            double coefficient_above = (0.5 * diffusion) + (0.5 * convection);
            if (coefficient_below < 0.0 || coefficient_above < 0.0) {
                ++upwinded_rows;
                if (convection >= 0.0) {
                    coefficient_below = 0.5 * diffusion;
                    coefficient_above = (0.5 * diffusion) + convection;
                } else {
                    coefficient_below = (0.5 * diffusion) - convection;
                    coefficient_above = 0.5 * diffusion;
                }
            }
            coefficient_lower[index] = coefficient_below;
            coefficient_upper[index] = coefficient_above;
            coefficient_diagonal[index] = -(coefficient_below + coefficient_above) - rate;
        }
        coefficient_lower[0] = 0.0;
        coefficient_upper[0] = 0.0;
        coefficient_diagonal[0] = -rate;
        result.upwinded_rows = std::max(result.upwinded_rows, upwinded_rows);

        // Every off-diagonal is non-negative after upwinding, so the row sum of
        // the implicit operator collapses to 1 + theta * dt * r. Both step
        // flavours use theta * dt = step_size / 2 (Crank-Nicolson at full size,
        // fully implicit at half size), so one margin covers both.
        if (1.0 + (step_size * 0.5 * rate) <= 0.0) {
            throw std::invalid_argument(
                "time step is too coarse for the declared rate to keep the implicit operator "
                "diagonally dominant");
        }

        for (std::size_t step = segment_steps; step-- > 0U;) {
            const double target_time =
                segment_start + (static_cast<double>(step) * step_size);
            if (damping_remaining > 0U) {
                const double half = 0.5 * step_size;
                advance(target_time + half, half, 1.0, segment);
                advance(target_time, half, 1.0, segment);
                --damping_remaining;
                result.damped_half_steps += 2U;
            } else {
                advance(target_time, step_size, 0.5, segment);
                ++result.crank_nicolson_steps;
            }
        }

        if (segment == 0U) {
            continue;
        }
        const double boundary_time = time_grid.aligned[segment];
        if (first_ahead == 0U) {
            continue;
        }
        const CashDividend& candidate = dividends[first_ahead - 1U];
        if (std::abs(candidate.ex_time - boundary_time) > alignment_tolerance) {
            continue;
        }

        monotone_slopes(values, spot_grid.step, slopes);
        for (std::size_t index = 0U; index < nodes; ++index) {
            const double node_spot = static_cast<double>(index) * spot_grid.step;
            const double after = post_dividend_spot(node_spot, candidate.amount);
            double carried = monotone_evaluate(values, slopes, spot_grid.step, after);
            if (is_american) {
                carried = std::max(carried, obstacle[index]);
            }
            require_finite(carried, "dividend jump value left double precision");
            shifted[index] = carried;
        }
        values.swap(shifted);
        --first_ahead;
        result.dividend_events.push_back({candidate.ex_time, candidate.amount, segment});
        damping_remaining = grid.rannacher_steps;
    }

    std::reverse(result.dividend_events.begin(), result.dividend_events.end());

    monotone_slopes(values, spot_grid.step, slopes);
    result.price = monotone_evaluate(values, slopes, spot_grid.step, contract.spot);
    require_finite(result.price, "finite-difference price left double precision");
    return result;
}

}  // namespace dp
