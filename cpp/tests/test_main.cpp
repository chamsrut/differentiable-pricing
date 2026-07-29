#include "dp/black_scholes.hpp"
#include "dp/smooth_mlp.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void expect_near(
    const double actual,
    const double expected,
    const double tolerance,
    const std::string& message
) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": actual=" + std::to_string(actual) +
            ", expected=" + std::to_string(expected)
        );
    }
}

void test_known_black_scholes_values() {
    const dp::BlackScholesInput input{100.0, 100.0, 1.0, 0.05, 0.0, 0.2};
    const auto call = dp::black_scholes(dp::OptionType::call, input);
    const auto put = dp::black_scholes(dp::OptionType::put, input);
    expect_near(call.price, 10.450583572185565, 1.0e-12, "call price");
    expect_near(put.price, 5.573526022256971, 1.0e-12, "put price");
}

void test_put_call_parity() {
    const dp::BlackScholesInput input{95.0, 100.0, 2.0, 0.03, 0.01, 0.25};
    const auto call = dp::black_scholes(dp::OptionType::call, input);
    const auto put = dp::black_scholes(dp::OptionType::put, input);
    const double parity =
        input.spot * std::exp(-input.dividend_yield * input.maturity) -
        input.strike * std::exp(-input.rate * input.maturity);
    expect_near(call.price - put.price, parity, 1.0e-12, "put-call parity");
}

void test_analytic_delta_against_central_difference() {
    const dp::BlackScholesInput input{100.0, 105.0, 0.75, 0.02, 0.01, 0.3};
    const auto result = dp::black_scholes(dp::OptionType::call, input);
    constexpr double bump = 1.0e-4;
    auto up = input;
    auto down = input;
    up.spot += bump;
    down.spot -= bump;
    const double finite_difference =
        (dp::black_scholes(dp::OptionType::call, up).price -
         dp::black_scholes(dp::OptionType::call, down).price) /
        (2.0 * bump);
    expect_near(result.delta, finite_difference, 1.0e-8, "analytic delta");
}

void test_mlp_reverse_gradient_against_central_difference() {
    dp::SmoothMlp model({
        dp::DenseLayer{
            2U,
            2U,
            {0.4, -0.2, 0.1, 0.3},
            {0.05, -0.1},
        },
        dp::DenseLayer{
            2U,
            1U,
            {0.7, -0.5},
            {0.2},
        },
    });

    const std::vector<double> input{0.25, -0.4};
    const auto result = model.forward_with_input_gradient(input);
    constexpr double bump = 1.0e-6;
    for (std::size_t feature = 0; feature < input.size(); ++feature) {
        auto up = input;
        auto down = input;
        up[feature] += bump;
        down[feature] -= bump;
        const double finite_difference =
            (model.forward_with_input_gradient(up).value -
             model.forward_with_input_gradient(down).value) /
            (2.0 * bump);
        expect_near(
            result.input_gradient[feature],
            finite_difference,
            1.0e-9,
            "MLP input gradient"
        );
    }
}

void test_invalid_input_is_rejected() {
    bool threw = false;
    try {
        static_cast<void>(dp::black_scholes(
            dp::OptionType::call,
            dp::BlackScholesInput{-1.0, 100.0, 1.0, 0.03, 0.0, 0.2}
        ));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    if (!threw) {
        throw std::runtime_error("negative spot was not rejected");
    }
}

}  // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests{
        {"known Black-Scholes values", test_known_black_scholes_values},
        {"put-call parity", test_put_call_parity},
        {"analytic delta", test_analytic_delta_against_central_difference},
        {"MLP reverse gradient", test_mlp_reverse_gradient_against_central_difference},
        {"invalid input", test_invalid_input_is_rejected},
    };

    int failures = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "PASS: " << name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL: " << name << ": " << error.what() << '\n';
        }
    }

    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << tests.size() << " test(s) passed\n";
    return 0;
}

