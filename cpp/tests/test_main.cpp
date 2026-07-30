#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dp/binomial_tree.hpp"
#include "dp/black_scholes.hpp"
#include "dp/smooth_mlp.hpp"

namespace {

void expect_near(const double actual, const double expected, const double tolerance,
                 const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message + ": actual=" + std::to_string(actual) +
                                 ", expected=" + std::to_string(expected));
    }
}

void expect_true(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expect_invalid_argument(const std::function<void()>& action, const std::string& message) {
    bool threw = false;
    try {
        action();
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    expect_true(threw, message);
}

void expect_overflow_error(const std::function<void()>& action, const std::string& message) {
    bool threw = false;
    try {
        action();
    } catch (const std::overflow_error&) {
        threw = true;
    }
    expect_true(threw, message);
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
    const double parity = input.spot * std::exp(-input.dividend_yield * input.maturity) -
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
    const double finite_difference = (dp::black_scholes(dp::OptionType::call, up).price -
                                      dp::black_scholes(dp::OptionType::call, down).price) /
                                     (2.0 * bump);
    expect_near(result.delta, finite_difference, 1.0e-8, "analytic delta");
}

void test_european_crr_converges_to_black_scholes() {
    const dp::BlackScholesInput input{100.0, 100.0, 1.0, 0.05, 0.0, 0.2};
    for (const dp::OptionType option_type : {dp::OptionType::call, dp::OptionType::put}) {
        const auto analytic = dp::black_scholes(option_type, input);
        const auto tree =
            dp::crr_adjacent_step_estimate(option_type, dp::ExerciseStyle::european, input, 4096U);
        expect_near(tree.average_price, analytic.price, 1.0e-4,
                    "European CRR versus Black-Scholes");
        expect_true(tree.at_steps.early_exercise_nodes == 0U &&
                        !tree.at_steps.earliest_exercise_step.has_value() &&
                        tree.at_steps.exercise_boundary_by_step.empty(),
                    "European tree reported early-exercise diagnostics");
    }
}

void test_no_dividend_american_call_matches_european_tree() {
    const dp::BlackScholesInput input{100.0, 100.0, 1.0, 0.05, 0.0, 0.2};
    const auto european =
        dp::crr_binomial(dp::OptionType::call, dp::ExerciseStyle::european, input, 2048U);
    const auto american =
        dp::crr_binomial(dp::OptionType::call, dp::ExerciseStyle::american, input, 2048U);
    expect_near(american.price, european.price, 1.0e-12, "no-dividend American call");
    expect_true(american.early_exercise_nodes == 0U &&
                    !american.earliest_exercise_step.has_value() &&
                    std::none_of(
                        american.exercise_boundary_by_step.begin(),
                        american.exercise_boundary_by_step.end(),
                        [](const std::optional<double>& boundary) { return boundary.has_value(); }),
                "no-dividend American call exercised early");
}

void test_american_put_has_exercise_premium() {
    const dp::BlackScholesInput input{100.0, 100.0, 1.0, 0.05, 0.0, 0.2};
    const auto european =
        dp::crr_binomial(dp::OptionType::put, dp::ExerciseStyle::european, input, 2048U);
    const auto american =
        dp::crr_binomial(dp::OptionType::put, dp::ExerciseStyle::american, input, 2048U);
    expect_true(american.price > european.price,
                "American put did not have a positive exercise premium");
    expect_true(american.early_exercise_nodes > 0U && american.earliest_exercise_step.has_value(),
                "American put did not report an exercise region");
    const auto midpoint_boundary = american.exercise_boundary_by_step.at(american.steps / 2U);
    const auto near_expiry_boundary = american.exercise_boundary_by_step.at(american.steps - 1U);
    expect_true(midpoint_boundary.has_value() && near_expiry_boundary.has_value() &&
                    *midpoint_boundary > 0.0 && (*midpoint_boundary < input.strike) &&
                    (*near_expiry_boundary > *midpoint_boundary) &&
                    (*near_expiry_boundary < input.strike),
                "American put exercise boundary had an invalid economic shape");
}

void test_exercise_diagnostics_ignore_numerical_indifference() {
    const dp::BlackScholesInput input{100.0, 100.0, 1.0, 0.0, 0.0, 0.2};
    for (const std::size_t steps : {64U, 256U, 1024U, 2048U, 4096U}) {
        for (const dp::OptionType option_type : {dp::OptionType::call, dp::OptionType::put}) {
            const auto european =
                dp::crr_binomial(option_type, dp::ExerciseStyle::european, input, steps);
            const auto american =
                dp::crr_binomial(option_type, dp::ExerciseStyle::american, input, steps);
            expect_near(american.price, european.price, 1.0e-11,
                        "zero-rate American versus European price");
            expect_true(american.early_exercise_nodes == 0U &&
                            !american.earliest_exercise_step.has_value() &&
                            std::none_of(american.exercise_boundary_by_step.begin(),
                                         american.exercise_boundary_by_step.end(),
                                         [](const std::optional<double>& boundary) {
                                             return boundary.has_value();
                                         }),
                        "numerical indifference was reported as material exercise");
        }
    }
}

void test_dividend_call_can_have_exercise_premium() {
    const dp::BlackScholesInput input{100.0, 100.0, 1.0, 0.01, 0.10, 0.25};
    const auto european =
        dp::crr_binomial(dp::OptionType::call, dp::ExerciseStyle::european, input, 2048U);
    const auto american =
        dp::crr_binomial(dp::OptionType::call, dp::ExerciseStyle::american, input, 2048U);
    expect_true(american.price > european.price,
                "dividend-paying American call did not have an exercise premium");
    expect_true(american.early_exercise_nodes > 0U,
                "dividend-paying American call did not report an exercise region");
}

void test_american_prices_respect_bounds_and_monotonicity() {
    const dp::BlackScholesInput low_spot{
        90.0, 100.0, 1.25, 0.04, 0.02, 0.3,
    };
    auto high_spot = low_spot;
    high_spot.spot = 110.0;

    const double low_call =
        dp::crr_binomial(dp::OptionType::call, dp::ExerciseStyle::american, low_spot, 1024U).price;
    const double high_call =
        dp::crr_binomial(dp::OptionType::call, dp::ExerciseStyle::american, high_spot, 1024U).price;
    const double low_put =
        dp::crr_binomial(dp::OptionType::put, dp::ExerciseStyle::american, low_spot, 1024U).price;
    const double high_put =
        dp::crr_binomial(dp::OptionType::put, dp::ExerciseStyle::american, high_spot, 1024U).price;

    expect_true(high_call > low_call, "American call was not spot-monotone");
    expect_true(low_put > high_put, "American put was not spot-monotone");
    expect_true(
        low_call >= std::max(low_spot.spot - low_spot.strike, 0.0) && low_call <= low_spot.spot,
        "American call violated intrinsic or spot bounds");
    expect_true(high_put >= std::max(high_spot.strike - high_spot.spot, 0.0) &&
                    high_put <= high_spot.strike,
                "American put violated intrinsic or strike bounds");
}

void test_american_crr_step_convergence() {
    const dp::BlackScholesInput input{100.0, 100.0, 1.0, 0.05, 0.0, 0.2};
    const double coarse =
        dp::crr_adjacent_step_estimate(dp::OptionType::put, dp::ExerciseStyle::american, input, 64U)
            .average_price;
    const double fine = dp::crr_adjacent_step_estimate(dp::OptionType::put,
                                                       dp::ExerciseStyle::american, input, 512U)
                            .average_price;
    const double reference = dp::crr_adjacent_step_estimate(
                                 dp::OptionType::put, dp::ExerciseStyle::american, input, 2048U)
                                 .average_price;
    expect_true(std::abs(fine - reference) < std::abs(coarse - reference),
                "American CRR refinement did not improve price stability");
}

void test_invalid_crr_requests_are_rejected() {
    const dp::BlackScholesInput input{100.0, 100.0, 1.0, 0.05, 0.0, 0.2};
    expect_invalid_argument(
        [&input]() {
            static_cast<void>(
                dp::crr_binomial(dp::OptionType::put, dp::ExerciseStyle::american, input, 0U));
        },
        "zero CRR steps were not rejected");
    expect_invalid_argument(
        [&input]() {
            static_cast<void>(dp::crr_binomial(dp::OptionType::put, dp::ExerciseStyle::american,
                                               input, dp::maximum_crr_steps + 1U));
        },
        "excessive CRR steps were not rejected");
    expect_invalid_argument(
        [&input]() {
            static_cast<void>(dp::crr_adjacent_step_estimate(
                dp::OptionType::put, dp::ExerciseStyle::american, input, dp::maximum_crr_steps));
        },
        "invalid adjacent-step request was not rejected");
    expect_invalid_argument(
        []() {
            static_cast<void>(dp::crr_binomial(dp::OptionType::call, dp::ExerciseStyle::american,
                                               dp::BlackScholesInput{
                                                   100.0,
                                                   100.0,
                                                   1.0,
                                                   1.0,
                                                   0.0,
                                                   0.01,
                                               },
                                               1U));
        },
        "invalid CRR probability was not rejected");
    expect_invalid_argument([]() { static_cast<void>(dp::parse_exercise_style("bermudan")); },
                            "invalid exercise style was not rejected");
    expect_overflow_error(
        []() {
            static_cast<void>(dp::crr_binomial(dp::OptionType::call, dp::ExerciseStyle::american,
                                               dp::BlackScholesInput{
                                                   100.0,
                                                   100.0,
                                                   1.0,
                                                   0.0,
                                                   0.0,
                                                   100.0,
                                               },
                                               100U));
        },
        "underflowed terminal lattice was not rejected");
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
        const double finite_difference = (model.forward_with_input_gradient(up).value -
                                          model.forward_with_input_gradient(down).value) /
                                         (2.0 * bump);
        expect_near(result.input_gradient[feature], finite_difference, 1.0e-9,
                    "MLP input gradient");
    }
}

void test_invalid_input_is_rejected() {
    bool threw = false;
    try {
        static_cast<void>(dp::black_scholes(
            dp::OptionType::call, dp::BlackScholesInput{-1.0, 100.0, 1.0, 0.03, 0.0, 0.2}));
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
        {"European CRR convergence", test_european_crr_converges_to_black_scholes},
        {"no-dividend American call", test_no_dividend_american_call_matches_european_tree},
        {"American put exercise premium", test_american_put_has_exercise_premium},
        {"exercise diagnostic indifference",
         test_exercise_diagnostics_ignore_numerical_indifference},
        {"dividend call exercise premium", test_dividend_call_can_have_exercise_premium},
        {"American bounds and monotonicity", test_american_prices_respect_bounds_and_monotonicity},
        {"American CRR step convergence", test_american_crr_step_convergence},
        {"invalid CRR requests", test_invalid_crr_requests_are_rejected},
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
