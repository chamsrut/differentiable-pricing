#include "dp/binomial_tree.hpp"
#include "dp/black_scholes.hpp"
#include "dp/smooth_mlp.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

template <typename T>
void require_column_size(const std::vector<T>& column, const std::size_t expected,
                         const std::string_view name) {
    if (column.size() != expected) {
        throw std::invalid_argument(std::string(name) +
                                    " must have the same length as option_types");
    }
}

}  // namespace

PYBIND11_MODULE(_core, module) {
    module.doc() = "C++ pricing and differentiable-inference core";
    module.attr("__version__") = "0.1.0";
    module.attr("__build_configuration__") = DP_BUILD_CONFIGURATION;
    module.attr("__cxx_compiler__") = DP_CXX_COMPILER;
    module.attr("__crr_header_sha256__") = DP_CRR_HEADER_SHA256;
    module.attr("__crr_implementation_sha256__") = DP_CRR_IMPLEMENTATION_SHA256;
    module.attr("__crr_source_sha256__") = DP_CRR_SOURCE_SHA256;
    module.attr("maximum_crr_steps") = dp::maximum_crr_steps;
    module.attr("maximum_crr_batch_threads") = dp::maximum_crr_batch_threads;

    module.def(
        "black_scholes",
        [](const std::string& option_type,
           const double spot,
           const double strike,
           const double maturity,
           const double rate,
           const double dividend_yield,
           const double volatility) {
            const auto result = dp::black_scholes(
                dp::parse_option_type(option_type),
                dp::BlackScholesInput{
                    spot,
                    strike,
                    maturity,
                    rate,
                    dividend_yield,
                    volatility,
                }
            );
            py::dict output;
            output["price"] = result.price;
            output["delta"] = result.delta;
            output["gamma"] = result.gamma;
            output["vega"] = result.vega;
            output["theta"] = result.theta;
            output["rho"] = result.rho;
            return output;
        },
        py::arg("option_type"),
        py::arg("spot"),
        py::arg("strike"),
        py::arg("maturity"),
        py::arg("rate"),
        py::arg("dividend_yield"),
        py::arg("volatility"),
        "Return the Black-Scholes price and analytic Greeks."
    );

    module.def(
        "crr_lattice_parameters",
        [](const double spot, const double strike, const double maturity, const double rate,
           const double dividend_yield, const double volatility, const std::size_t steps) {
            const auto result = dp::crr_lattice_parameters(
                dp::VanillaOptionInput{
                    spot,
                    strike,
                    maturity,
                    rate,
                    dividend_yield,
                    volatility,
                },
                steps
            );
            py::dict output;
            output["time_step"] = result.time_step;
            output["up_factor"] = result.up_factor;
            output["down_factor"] = result.down_factor;
            output["risk_neutral_probability"] = result.risk_neutral_probability;
            output["discount_factor"] = result.discount_factor;
            return output;
        },
        py::arg("spot"),
        py::arg("strike"),
        py::arg("maturity"),
        py::arg("rate"),
        py::arg("dividend_yield"),
        py::arg("volatility"),
        py::arg("steps"),
        "Return validated CRR lattice parameters without traversing the tree."
    );

    module.def(
        "crr_price",
        [](const std::string& option_type, const std::string& exercise_style, const double spot,
           const double strike, const double maturity, const double rate,
           const double dividend_yield, const double volatility, const std::size_t steps) {
            const dp::CrrPriceRequest request{
                dp::parse_option_type(option_type),
                dp::parse_exercise_style(exercise_style),
                {
                    spot,
                    strike,
                    maturity,
                    rate,
                    dividend_yield,
                    volatility,
                },
                steps,
            };
            const dp::CrrPriceOnlyResult result = [&request]() {
                py::gil_scoped_release release;
                return dp::crr_price_only(request);
            }();
            py::dict output;
            output["price"] = result.price;
            output["steps"] = result.steps;
            output["risk_neutral_probability"] = result.risk_neutral_probability;
            return output;
        },
        py::arg("option_type"),
        py::arg("exercise_style"),
        py::arg("spot"),
        py::arg("strike"),
        py::arg("maturity"),
        py::arg("rate"),
        py::arg("dividend_yield"),
        py::arg("volatility"),
        py::arg("steps"),
        "Return a CRR price without retaining exercise-boundary diagnostics."
    );

    module.def(
        "crr_diagnostics",
        [](const std::string& option_type, const std::string& exercise_style, const double spot,
           const double strike, const double maturity, const double rate,
           const double dividend_yield, const double volatility, const std::size_t steps) {
            const dp::OptionType parsed_option_type = dp::parse_option_type(option_type);
            const dp::ExerciseStyle parsed_exercise_style =
                dp::parse_exercise_style(exercise_style);
            const dp::VanillaOptionInput input{
                spot,
                strike,
                maturity,
                rate,
                dividend_yield,
                volatility,
            };
            const dp::CrrResult result = [&]() {
                py::gil_scoped_release release;
                return dp::crr_binomial(parsed_option_type, parsed_exercise_style, input, steps);
            }();
            py::dict output;
            output["price"] = result.price;
            output["steps"] = result.steps;
            output["risk_neutral_probability"] = result.risk_neutral_probability;
            output["early_exercise_nodes"] = result.early_exercise_nodes;
            output["earliest_exercise_step"] = result.earliest_exercise_step;
            output["exercise_boundary_by_step"] = result.exercise_boundary_by_step;
            return output;
        },
        py::arg("option_type"),
        py::arg("exercise_style"),
        py::arg("spot"),
        py::arg("strike"),
        py::arg("maturity"),
        py::arg("rate"),
        py::arg("dividend_yield"),
        py::arg("volatility"),
        py::arg("steps"),
        "Return a scalar CRR price with exercise-region diagnostics."
    );

    module.def(
        "crr_price_batch",
        [](const std::vector<std::string>& option_types,
           const std::vector<std::string>& exercise_styles, const std::vector<double>& spots,
           const std::vector<double>& strikes, const std::vector<double>& maturities,
           const std::vector<double>& rates, const std::vector<double>& dividend_yields,
           const std::vector<double>& volatilities, const std::vector<std::size_t>& steps,
           const std::size_t thread_count) {
            const std::size_t size = option_types.size();
            require_column_size(exercise_styles, size, "exercise_styles");
            require_column_size(spots, size, "spots");
            require_column_size(strikes, size, "strikes");
            require_column_size(maturities, size, "maturities");
            require_column_size(rates, size, "rates");
            require_column_size(dividend_yields, size, "dividend_yields");
            require_column_size(volatilities, size, "volatilities");
            require_column_size(steps, size, "steps");

            std::vector<dp::CrrPriceRequest> requests;
            requests.reserve(size);
            for (std::size_t index = 0U; index < size; ++index) {
                try {
                    requests.push_back({
                        dp::parse_option_type(option_types[index]),
                        dp::parse_exercise_style(exercise_styles[index]),
                        {
                            spots[index],
                            strikes[index],
                            maturities[index],
                            rates[index],
                            dividend_yields[index],
                            volatilities[index],
                        },
                        steps[index],
                    });
                } catch (const std::exception& error) {
                    throw std::invalid_argument("CRR batch request at index " +
                                                std::to_string(index) +
                                                " is invalid: " + error.what());
                }
            }

            std::vector<dp::CrrPriceOnlyResult> results;
            {
                py::gil_scoped_release release;
                results = dp::crr_price_batch(requests, thread_count);
            }

            std::vector<double> prices;
            std::vector<double> probabilities;
            std::vector<std::size_t> result_steps;
            prices.reserve(size);
            probabilities.reserve(size);
            result_steps.reserve(size);
            for (const dp::CrrPriceOnlyResult& result : results) {
                prices.push_back(result.price);
                probabilities.push_back(result.risk_neutral_probability);
                result_steps.push_back(result.steps);
            }

            py::dict output;
            output["price"] = std::move(prices);
            output["steps"] = std::move(result_steps);
            output["risk_neutral_probability"] = std::move(probabilities);
            return output;
        },
        py::arg("option_types"),
        py::arg("exercise_styles"),
        py::arg("spots"),
        py::arg("strikes"),
        py::arg("maturities"),
        py::arg("rates"),
        py::arg("dividend_yields"),
        py::arg("volatilities"),
        py::arg("steps"),
        py::arg("thread_count") = 1U,
        "Price independent CRR requests in deterministic input order."
    );

    py::class_<dp::DenseLayer>(module, "DenseLayer")
        .def(
            py::init(
                [](const std::size_t input_size,
                   const std::size_t output_size,
                   std::vector<double> weights,
                   std::vector<double> biases) {
                    return dp::DenseLayer{
                        input_size,
                        output_size,
                        std::move(weights),
                        std::move(biases),
                    };
                }
            ),
            py::arg("input_size"),
            py::arg("output_size"),
            py::arg("weights"),
            py::arg("biases")
        );

    py::class_<dp::SmoothMlp>(module, "SmoothMlp")
        .def(py::init<std::vector<dp::DenseLayer>>(), py::arg("layers"))
        .def_property_readonly("input_size", &dp::SmoothMlp::input_size)
        .def(
            "forward_with_input_gradient",
            [](const dp::SmoothMlp& model, const std::vector<double>& input) {
                const auto result = model.forward_with_input_gradient(input);
                return py::make_tuple(result.value, result.input_gradient);
            },
            py::arg("input"),
            "Return the scalar output and exact input gradient of the C++ MLP."
        );
}
