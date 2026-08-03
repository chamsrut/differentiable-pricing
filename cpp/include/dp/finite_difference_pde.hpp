#pragma once

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "dp/option.hpp"

namespace dp {

// Operational ceilings. They exist so an accidental request fails immediately
// instead of allocating for minutes. None of them is a numerical claim.
inline constexpr std::size_t maximum_pde_spot_intervals = 200'000U;
inline constexpr std::size_t maximum_pde_time_steps = 400'000U;
inline constexpr std::size_t maximum_pde_cash_dividends = 512U;
inline constexpr std::size_t maximum_pde_curve_nodes = 4'096U;
inline constexpr std::size_t maximum_pde_rannacher_steps = 64U;

// Times are ACT/365F year fractions measured from the discount curve's origin.
// One second is about 3.17e-8 years, so this tolerance is far below any
// economically distinguishable instant. It is used only to decide whether two
// declared times are the same knot, never to repair an ordering.
inline constexpr double pde_time_tolerance = 1.0e-12;

// Settlement is carried because the task 9B input contract lists it. Task 9C-A
// prices per share and reads neither this nor the contract multiplier in any
// arithmetic; both are reporting metadata that must still be stated.
enum class SettlementConvention {
    cash,
    physical,
};

[[nodiscard]] SettlementConvention parse_settlement_convention(std::string_view value);

// Times and log discount factors of a piecewise-linear log-discount curve.
//
// The curve is carried in log-discount space so positivity holds by
// construction and a zero rate is a zero slope rather than a quotient. Linear
// interpolation of the log discount makes the implied instantaneous rate
// piecewise constant, equal on each segment to the negative slope. The curve
// must begin at (0, 0) and must bracket the whole valuation interval:
// evaluation outside [times.front(), times.back()] is rejected, never
// extrapolated.
struct PiecewiseLogDiscountCurve {
    std::vector<double> times;
    std::vector<double> log_discounts;

    void validate() const;

    // Linear interpolation in log-discount space. Throws std::invalid_argument
    // outside the declared knot range.
    [[nodiscard]] double log_discount(double time) const;

    // exp(log_discount(to_time) - log_discount(from_time)).
    [[nodiscard]] double discount_factor(double from_time, double to_time) const;

    // Piecewise-constant instantaneous rate on segment [times[i], times[i+1]].
    [[nodiscard]] double segment_rate(std::size_t segment_index) const;

    // Index of the segment containing `time`; the last segment is used at the
    // final knot. Throws std::invalid_argument outside the knot range.
    [[nodiscard]] std::size_t segment_index_for(double time) const;
};

struct CashDividend {
    double ex_time;
    double amount;
};

// A dividend schedule is a declaration, not a default.
//
// "This contract pays no cash dividend" and "the caller never said" are
// different statements, and a default-constructed vector cannot distinguish
// them. A default-constructed schedule is therefore undeclared and every entry
// point rejects it; callers reach the empty case only through
// `declared_none()`.
class CashDividendSchedule {
public:
    CashDividendSchedule() = default;

    [[nodiscard]] static CashDividendSchedule declared_none();
    [[nodiscard]] static CashDividendSchedule declared(std::vector<CashDividend> dividends);

    [[nodiscard]] bool is_declared() const noexcept {
        return declared_;
    }
    [[nodiscard]] const std::vector<CashDividend>& dividends() const noexcept {
        return dividends_;
    }

    // Rejects an undeclared schedule, a nonpositive or non-finite amount, an
    // ex-time outside the open interval (valuation_time, expiry_time), an
    // unsorted schedule, and duplicated ex-times.
    void validate(double valuation_time, double expiry_time) const;

private:
    CashDividendSchedule(bool declared, std::vector<CashDividend> dividends);

    bool declared_ = false;
    std::vector<CashDividend> dividends_;
};

// The contract priced by the finite-difference oracle, in the field order of
// the task 9B proposed input contract. Nothing here has a default: a missing
// rate, carry, dividend list or convention is an error, not a zero.
struct PdeContract {
    OptionType option_type;
    ExerciseStyle exercise_style;
    double spot;
    double strike;
    double valuation_time;
    double expiry_time;
    // Constant for task 9C-A.
    double volatility;
    // The continuous carry c subtracted from the risk-neutral drift, so the
    // drift is (r(t) - c). std::nullopt means the caller has no carry
    // information; it is rejected rather than silently read as zero, because a
    // zero that reads as an observation is exactly the failure mode task 9B
    // wrote this field to prevent.
    std::optional<double> continuous_carry;
    PiecewiseLogDiscountCurve discount_curve;
    CashDividendSchedule dividends;
    SettlementConvention settlement;
    // Reporting metadata only. Prices are per share and this value never enters
    // the pricing arithmetic.
    double contract_multiplier;

    void validate() const;
};

// Discretization settings. `spot_intervals` and `spot_maximum` are targets: the
// solver places a node exactly on the strike, which perturbs both slightly. The
// values actually used are reported in PdeResult and are what a refinement
// ladder must quote.
struct PdeGrid {
    std::size_t spot_intervals;
    std::size_t time_steps;
    double spot_maximum;
    // Number of time steps after the terminal payoff and after each dividend
    // jump that are replaced by two fully implicit half steps (Rannacher
    // damping). Zero disables damping and is permitted so its effect can be
    // measured.
    std::size_t rannacher_steps;
    double psor_tolerance;
    double psor_relaxation;
    std::size_t psor_maximum_iterations;

    void validate() const;
};

// A dividend as it was actually applied, with the index of the aligned time in
// PdeResult::aligned_times. This is the evidence that alignment happened.
struct PdeDividendEvent {
    double ex_time;
    double amount;
    std::size_t aligned_time_index;
};

// A returned result always carries `discrete_system_converged`: the implicit
// tridiagonal or PSOR system reached its declared residual contract. This says
// nothing about continuum discretization accuracy, which is reported
// separately.
enum class PdeSolverStatus {
    discrete_system_converged,
    psor_iteration_limit_exceeded,
};

enum class PdeDiscretizationAccuracy {
    not_assessed,
};

struct PdeResult {
    double price;
    PdeSolverStatus solver_status;
    PdeDiscretizationAccuracy discretization_accuracy;

    // Grid and time settings actually used.
    std::size_t spot_intervals;
    double spot_maximum;
    double spot_step;
    std::size_t strike_node_index;
    std::size_t time_steps;
    std::size_t rannacher_steps;
    std::size_t damped_half_steps;
    std::size_t crank_nicolson_steps;
    // Interior rows where central differencing would have produced a negative
    // off-diagonal and one-sided upwinding was used instead. Nonzero means the
    // convection term dominates diffusion near S = 0 on this grid.
    std::size_t upwinded_rows;

    // Solver diagnostics.
    std::size_t linear_solves;
    std::size_t psor_solves;
    std::size_t psor_total_iterations;
    std::size_t psor_maximum_iterations_used;
    double psor_tolerance;
    double psor_relaxation;
    // max_i |min(x_i - obstacle_i, (A x - b)_i)| over every solve, absolute and
    // normalised by max(1, ||b||_inf). The normalised value is the one compared
    // against psor_tolerance. For a European step the obstacle is absent and
    // this degenerates to the linear residual ||A x - b||_inf of the direct
    // tridiagonal solve.
    double maximum_lcp_residual;
    double maximum_relative_lcp_residual;

    // Times every dividend and every interior curve knot was aligned to,
    // including both interval endpoints. Small by construction.
    std::vector<double> aligned_times;
    std::vector<PdeDividendEvent> dividend_events;
};

class PdePsorFailure : public std::runtime_error {
public:
    PdePsorFailure(const std::string& message, std::size_t aligned_segment,
                   std::size_t iterations, double residual, double tolerance);

    [[nodiscard]] PdeSolverStatus solver_status() const noexcept {
        return PdeSolverStatus::psor_iteration_limit_exceeded;
    }
    [[nodiscard]] std::size_t aligned_segment() const noexcept {
        return aligned_segment_;
    }
    [[nodiscard]] std::size_t iterations() const noexcept {
        return iterations_;
    }
    [[nodiscard]] double residual() const noexcept {
        return residual_;
    }
    [[nodiscard]] double tolerance() const noexcept {
        return tolerance_;
    }

private:
    std::size_t aligned_segment_;
    std::size_t iterations_;
    double residual_;
    double tolerance_;
};

// The spot immediately after a cash dividend: max(spot - amount, 0). Exposed
// because it is the whole content of the dividend jump in spot space and is
// worth testing on its own, including the spot < amount branch.
[[nodiscard]] double post_dividend_spot(double spot, double amount);

// Deterministic one-dimensional finite-difference price of a European or
// American vanilla option with explicit discrete cash dividends.
//
// Crank-Nicolson time stepping with Rannacher damping after the terminal payoff
// and after each dividend jump; the American obstacle is a linear
// complementarity problem solved by projected successive over-relaxation.
// Working memory is O(spot_intervals) plus the small aligned-time and
// dividend-event vectors: no space-time array is ever held.
//
// Throws std::invalid_argument for an invalid contract or grid, std::overflow_error
// when a price-domain quantity leaves double precision, and PdePsorFailure when
// PSOR does not reach the declared LCP residual within the iteration limit.
[[nodiscard]] PdeResult finite_difference_price(const PdeContract& contract, const PdeGrid& grid);

}  // namespace dp
