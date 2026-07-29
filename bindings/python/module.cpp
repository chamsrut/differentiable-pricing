#include "dp/black_scholes.hpp"
#include "dp/smooth_mlp.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(_core, module) {
    module.doc() = "C++ pricing and differentiable-inference core";
    module.attr("__version__") = "0.1.0";

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
