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
}
