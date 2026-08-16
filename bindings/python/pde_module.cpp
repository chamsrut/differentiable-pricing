// The finite-difference oracle is bound from its own translation unit rather
// than from bindings/python/module.cpp.
//
// CMakeLists derives ``__crr_implementation_sha256__`` and
// ``__lsm_implementation_sha256__`` from the binding source together with the
// engine headers and sources, and the frozen American LSM cross-check snapshot
// records the digest the study actually ran against. Editing module.cpp would
// invalidate that recorded provenance without any LSM behaviour having changed,
// and the snapshot cannot be re-derived without rerunning the study. A separate
// extension keeps the existing engines' identity byte-stable.

#include "dp/finite_difference_pde.hpp"
#include "dp/option.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

[[nodiscard]] dp::PiecewiseLogDiscountCurve build_curve(std::vector<double> times,
                                                        std::vector<double> log_discounts) {
    dp::PiecewiseLogDiscountCurve curve{std::move(times), std::move(log_discounts)};
    curve.validate();
    return curve;
}

// The dividend list is a required argument with no default, so an empty list at
// this boundary is the caller stating "no cash dividends" rather than a
// default-constructed vector arriving by accident.
[[nodiscard]] dp::CashDividendSchedule build_schedule(
    const std::vector<std::pair<double, double>>& dividends) {
    std::vector<dp::CashDividend> parsed;
    parsed.reserve(dividends.size());
    for (const auto& [ex_time, amount] : dividends) {
        parsed.push_back({ex_time, amount});
    }
    return dp::CashDividendSchedule::declared(std::move(parsed));
}

[[nodiscard]] py::list dividend_event_rows(const std::vector<dp::PdeDividendEvent>& events) {
    py::list rows;
    for (const dp::PdeDividendEvent& event : events) {
        py::dict row;
        row["ex_time"] = event.ex_time;
        row["amount"] = event.amount;
        row["aligned_time_index"] = event.aligned_time_index;
        rows.append(std::move(row));
    }
    return rows;
}

// The solve metadata is reported under exactly the key names `pde_price` uses,
// so a caller reads one solve the same way whichever entry point produced it.
void add_solve_diagnostics(py::dict& output, const dp::PdeSolveDiagnostics& diagnostics) {
    output["solver_status"] =
        diagnostics.solver_status == dp::PdeSolverStatus::discrete_system_converged
            ? "discrete_system_converged"
            : "psor_iteration_limit_exceeded";
    output["discretization_accuracy"] =
        diagnostics.discretization_accuracy == dp::PdeDiscretizationAccuracy::not_assessed
            ? "not_assessed"
            : "unknown";
    output["spot_intervals"] = diagnostics.spot_intervals;
    output["spot_maximum"] = diagnostics.spot_maximum;
    output["spot_step"] = diagnostics.spot_step;
    output["strike_node_index"] = diagnostics.strike_node_index;
    output["time_steps"] = diagnostics.time_steps;
    output["rannacher_steps"] = diagnostics.rannacher_steps;
    output["damped_half_steps"] = diagnostics.damped_half_steps;
    output["crank_nicolson_steps"] = diagnostics.crank_nicolson_steps;
    output["upwinded_rows"] = diagnostics.upwinded_rows;
    output["linear_solves"] = diagnostics.linear_solves;
    output["psor_solves"] = diagnostics.psor_solves;
    output["psor_total_iterations"] = diagnostics.psor_total_iterations;
    output["psor_maximum_iterations_used"] = diagnostics.psor_maximum_iterations_used;
    output["psor_tolerance"] = diagnostics.psor_tolerance;
    output["psor_relaxation"] = diagnostics.psor_relaxation;
    output["maximum_lcp_residual"] = diagnostics.maximum_lcp_residual;
    output["maximum_relative_lcp_residual"] = diagnostics.maximum_relative_lcp_residual;
    output["aligned_times"] = diagnostics.aligned_times;
    output["dividend_events"] = dividend_event_rows(diagnostics.dividend_events);
}

// Nodes are transferred columnar: one std::vector per field, converted in bulk
// by pybind11's stl caster. Greeks that do not exist carry None rather than a
// plausible-looking substitute.
void add_surface_nodes(py::dict& output, const dp::PdeValuationSurface& surface) {
    const std::size_t count = surface.nodes.size();
    std::vector<double> spots(count, 0.0);
    std::vector<double> values(count, 0.0);
    std::vector<double> obstacles(count, 0.0);
    std::vector<double> obstacle_slacks(count, 0.0);
    std::vector<double> residuals(count, 0.0);
    std::vector<std::optional<double>> deltas(count, std::nullopt);
    std::vector<std::optional<double>> gammas(count, std::nullopt);
    std::vector<bool> eligible(count, false);
    std::vector<std::string> states;
    std::vector<std::string> reasons;
    states.reserve(count);
    reasons.reserve(count);
    for (std::size_t index = 0U; index < count; ++index) {
        const dp::PdeSurfaceNode& node = surface.nodes[index];
        spots[index] = node.spot;
        values[index] = node.value;
        obstacles[index] = node.obstacle;
        obstacle_slacks[index] = node.obstacle_slack;
        residuals[index] = node.lcp_linear_residual;
        deltas[index] = node.delta;
        gammas[index] = node.gamma;
        eligible[index] = node.greek_eligible;
        states.emplace_back(dp::pde_exercise_state_name(node.exercise_state));
        reasons.emplace_back(dp::pde_greek_eligibility_reason_name(node.greek_eligibility_reason));
    }
    output["spot_nodes"] = std::move(spots);
    output["values"] = std::move(values);
    output["obstacles"] = std::move(obstacles);
    output["obstacle_slacks"] = std::move(obstacle_slacks);
    output["lcp_linear_residuals"] = std::move(residuals);
    output["deltas"] = std::move(deltas);
    output["gammas"] = std::move(gammas);
    output["greek_eligible"] = std::move(eligible);
    output["exercise_states"] = std::move(states);
    output["greek_eligibility_reasons"] = std::move(reasons);
}

[[nodiscard]] py::list query_rows(const std::vector<dp::PdeSurfaceQuery>& queries) {
    py::list rows;
    for (const dp::PdeSurfaceQuery& query : queries) {
        py::dict row;
        row["query_index"] = query.query_index;
        row["first_occurrence_index"] = query.first_occurrence_index;
        row["spot"] = query.spot;
        row["value"] = query.value;
        row["left_node_index"] = query.left_node_index;
        row["right_node_index"] = query.right_node_index;
        if (query.exercise_state.has_value()) {
            row["exercise_state"] = std::string(dp::pde_exercise_state_name(*query.exercise_state));
        } else {
            row["exercise_state"] = py::none();
        }
        row["delta"] = query.delta;
        row["gamma"] = query.gamma;
        row["greek_eligible"] = query.greek_eligible;
        row["greek_eligibility_reason"] =
            std::string(dp::pde_greek_eligibility_reason_name(query.greek_eligibility_reason));
        rows.append(std::move(row));
    }
    return rows;
}

}  // namespace

PYBIND11_MODULE(_pde, module) {
    module.doc() = "Deterministic finite-difference option-price oracle";
    module.attr("__version__") = "0.1.0";
    module.attr("__build_configuration__") = DP_BUILD_CONFIGURATION;
    module.attr("__cxx_compiler__") = DP_CXX_COMPILER;
    module.attr("__pde_header_sha256__") = DP_PDE_HEADER_SHA256;
    module.attr("__pde_source_sha256__") = DP_PDE_SOURCE_SHA256;
    module.attr("__pde_implementation_sha256__") = DP_PDE_IMPLEMENTATION_SHA256;
    module.attr("maximum_pde_spot_intervals") = dp::maximum_pde_spot_intervals;
    module.attr("maximum_pde_time_steps") = dp::maximum_pde_time_steps;
    module.attr("maximum_pde_cash_dividends") = dp::maximum_pde_cash_dividends;
    module.attr("maximum_pde_curve_nodes") = dp::maximum_pde_curve_nodes;
    module.attr("maximum_pde_rannacher_steps") = dp::maximum_pde_rannacher_steps;
    module.attr("pde_regime_stencil_radius") = dp::pde_regime_stencil_radius;

    // Non-convergence is an explicit, separately catchable failure. It is never
    // a returned price.
    py::register_exception<dp::PdePsorFailure>(module, "PsorConvergenceError", PyExc_RuntimeError);

    module.def(
        "pde_post_dividend_spot",
        &dp::post_dividend_spot,
        py::arg("spot"),
        py::arg("amount"),
        "Return max(spot - amount, 0), the spot immediately after a cash dividend."
    );

    module.def(
        "pde_log_discount",
        [](std::vector<double> times, std::vector<double> log_discounts, const double time) {
            return build_curve(std::move(times), std::move(log_discounts)).log_discount(time);
        },
        py::arg("times"),
        py::arg("log_discounts"),
        py::arg("time"),
        "Interpolate the piecewise-linear log discount; refuse to extrapolate."
    );

    module.def(
        "pde_discount_factor",
        [](std::vector<double> times, std::vector<double> log_discounts, const double from_time,
           const double to_time) {
            return build_curve(std::move(times), std::move(log_discounts))
                .discount_factor(from_time, to_time);
        },
        py::arg("times"),
        py::arg("log_discounts"),
        py::arg("from_time"),
        py::arg("to_time"),
        "Return exp(log_discount(to_time) - log_discount(from_time))."
    );

    module.def(
        "pde_segment_rate",
        [](std::vector<double> times, std::vector<double> log_discounts,
           const std::size_t segment_index) {
            return build_curve(std::move(times), std::move(log_discounts))
                .segment_rate(segment_index);
        },
        py::arg("times"),
        py::arg("log_discounts"),
        py::arg("segment_index"),
        "Return the piecewise-constant instantaneous rate implied by one curve segment."
    );

    module.def(
        "pde_price",
        [](const std::string& option_type,
           const std::string& exercise_style,
           const double spot,
           const double strike,
           const double valuation_time,
           const double expiry_time,
           const double volatility,
           const std::optional<double> continuous_carry,
           std::vector<double> curve_times,
           std::vector<double> curve_log_discounts,
           const std::vector<std::pair<double, double>>& dividends,
           const std::string& settlement,
           const double contract_multiplier,
           const std::size_t spot_intervals,
           const std::size_t time_steps,
           const double spot_maximum,
           const std::size_t rannacher_steps,
           const double psor_tolerance,
           const double psor_relaxation,
           const std::size_t psor_maximum_iterations) {
            const dp::PdeContract contract{
                dp::parse_option_type(option_type),
                dp::parse_exercise_style(exercise_style),
                spot,
                strike,
                valuation_time,
                expiry_time,
                volatility,
                continuous_carry,
                dp::PiecewiseLogDiscountCurve{
                    std::move(curve_times),
                    std::move(curve_log_discounts),
                },
                build_schedule(dividends),
                dp::parse_settlement_convention(settlement),
                contract_multiplier,
            };
            const dp::PdeGrid grid{
                spot_intervals,
                time_steps,
                spot_maximum,
                rannacher_steps,
                psor_tolerance,
                psor_relaxation,
                psor_maximum_iterations,
            };
            const dp::PdeResult result = [&]() {
                py::gil_scoped_release release;
                return dp::finite_difference_price(contract, grid);
            }();

            py::list events;
            for (const dp::PdeDividendEvent& event : result.dividend_events) {
                py::dict row;
                row["ex_time"] = event.ex_time;
                row["amount"] = event.amount;
                row["aligned_time_index"] = event.aligned_time_index;
                events.append(std::move(row));
            }

            py::dict output;
            output["price"] = result.price;
            output["solver_status"] =
                result.solver_status == dp::PdeSolverStatus::discrete_system_converged
                    ? "discrete_system_converged"
                    : "psor_iteration_limit_exceeded";
            output["discretization_accuracy"] = result.discretization_accuracy ==
                                                        dp::PdeDiscretizationAccuracy::not_assessed
                                                    ? "not_assessed"
                                                    : "unknown";
            output["spot_intervals"] = result.spot_intervals;
            output["spot_maximum"] = result.spot_maximum;
            output["spot_step"] = result.spot_step;
            output["strike_node_index"] = result.strike_node_index;
            output["time_steps"] = result.time_steps;
            output["rannacher_steps"] = result.rannacher_steps;
            output["damped_half_steps"] = result.damped_half_steps;
            output["crank_nicolson_steps"] = result.crank_nicolson_steps;
            output["upwinded_rows"] = result.upwinded_rows;
            output["linear_solves"] = result.linear_solves;
            output["psor_solves"] = result.psor_solves;
            output["psor_total_iterations"] = result.psor_total_iterations;
            output["psor_maximum_iterations_used"] = result.psor_maximum_iterations_used;
            output["psor_tolerance"] = result.psor_tolerance;
            output["psor_relaxation"] = result.psor_relaxation;
            output["maximum_lcp_residual"] = result.maximum_lcp_residual;
            output["maximum_relative_lcp_residual"] = result.maximum_relative_lcp_residual;
            output["aligned_times"] = result.aligned_times;
            output["dividend_events"] = std::move(events);
            return output;
        },
        py::arg("option_type"),
        py::arg("exercise_style"),
        py::arg("spot"),
        py::arg("strike"),
        py::arg("valuation_time"),
        py::arg("expiry_time"),
        py::arg("volatility"),
        py::arg("continuous_carry"),
        py::arg("curve_times"),
        py::arg("curve_log_discounts"),
        py::arg("dividends"),
        py::arg("settlement"),
        py::arg("contract_multiplier"),
        py::arg("spot_intervals"),
        py::arg("time_steps"),
        py::arg("spot_maximum"),
        py::arg("rannacher_steps"),
        py::arg("psor_tolerance"),
        py::arg("psor_relaxation"),
        py::arg("psor_maximum_iterations"),
        "Price one European or American vanilla option with explicit discrete cash "
        "dividends by Crank-Nicolson finite differences with Rannacher damping and a "
        "PSOR obstacle solve. Every argument is required: no rate, carry, dividend "
        "list or convention has a default."
    );

    module.def(
        "pde_valuation_surface",
        [](const std::string& option_type,
           const std::string& exercise_style,
           const double strike,
           const double valuation_time,
           const double expiry_time,
           const double volatility,
           const std::optional<double> continuous_carry,
           std::vector<double> curve_times,
           std::vector<double> curve_log_discounts,
           const std::vector<std::pair<double, double>>& dividends,
           const std::string& settlement,
           const double contract_multiplier,
           const std::size_t spot_intervals,
           const std::size_t time_steps,
           const double spot_maximum,
           const std::size_t rannacher_steps,
           const double psor_tolerance,
           const double psor_relaxation,
           const std::size_t psor_maximum_iterations,
           const std::size_t boundary_exclusion_nodes,
           const std::vector<double>& query_spots) {
            const dp::PdeSurfaceContract contract{
                dp::parse_option_type(option_type),
                dp::parse_exercise_style(exercise_style),
                strike,
                valuation_time,
                expiry_time,
                volatility,
                continuous_carry,
                dp::PiecewiseLogDiscountCurve{
                    std::move(curve_times),
                    std::move(curve_log_discounts),
                },
                build_schedule(dividends),
                dp::parse_settlement_convention(settlement),
                contract_multiplier,
            };
            const dp::PdeGrid grid{
                spot_intervals,
                time_steps,
                spot_maximum,
                rannacher_steps,
                psor_tolerance,
                psor_relaxation,
                psor_maximum_iterations,
            };
            const dp::PdeSurfaceSettings settings{boundary_exclusion_nodes};
            const dp::PdeSurfaceEvaluation evaluation = [&]() {
                py::gil_scoped_release release;
                return dp::finite_difference_valuation_surface_at(contract, grid, settings,
                                                                  query_spots);
            }();

            py::dict output;
            add_solve_diagnostics(output, evaluation.surface.diagnostics);
            output["backward_inductions"] = evaluation.surface.backward_inductions;
            output["valuation_time"] = evaluation.surface.valuation_time;
            output["exercise_classification_scale"] =
                evaluation.surface.exercise_classification_scale;
            output["boundary_exclusion_nodes"] = evaluation.surface.boundary_exclusion_nodes;
            output["regime_stencil_radius"] = evaluation.surface.regime_stencil_radius;
            add_surface_nodes(output, evaluation.surface);
            output["query_count"] = evaluation.queries.size();
            output["queries"] = query_rows(evaluation.queries);
            return output;
        },
        py::arg("option_type"),
        py::arg("exercise_style"),
        py::arg("strike"),
        py::arg("valuation_time"),
        py::arg("expiry_time"),
        py::arg("volatility"),
        py::arg("continuous_carry"),
        py::arg("curve_times"),
        py::arg("curve_log_discounts"),
        py::arg("dividends"),
        py::arg("settlement"),
        py::arg("contract_multiplier"),
        py::arg("spot_intervals"),
        py::arg("time_steps"),
        py::arg("spot_maximum"),
        py::arg("rannacher_steps"),
        py::arg("psor_tolerance"),
        py::arg("psor_relaxation"),
        py::arg("psor_maximum_iterations"),
        py::arg("boundary_exclusion_nodes"),
        py::arg("query_spots"),
        "Run one backward induction and return the valuation-time spot slice with "
        "nodewise price, delta, gamma, exercise classification and Greek eligibility, "
        "together with every requested spot evaluated against that single solve. There "
        "is no spot argument: the solved domain depends only on the strike and the grid. "
        "Query order is preserved, duplicates are kept, and a spot outside the solved "
        "domain is rejected rather than extrapolated. A query's delta and gamma are "
        "interpolated nodewise discrete Greeks, not derivatives of the query price "
        "interpolant; on an exact grid node they are bitwise that node's own."
    );
}
