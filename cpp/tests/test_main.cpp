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
#include "dp/least_squares_monte_carlo.hpp"
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

void expect_invalid_argument_contains(const std::function<void()>& action,
                                      const std::string& expected_text,
                                      const std::string& message) {
    try {
        action();
    } catch (const std::invalid_argument& error) {
        expect_true(std::string(error.what()).find(expected_text) != std::string::npos,
                    message + ": unexpected message=" + error.what());
        return;
    }
    throw std::runtime_error(message + ": no invalid_argument was thrown");
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

void test_negative_rate_call_can_have_exercise_premium() {
    const dp::BlackScholesInput input{100.0, 100.0, 1.0, -0.01, 0.0, 0.2};
    const auto european =
        dp::crr_binomial(dp::OptionType::call, dp::ExerciseStyle::european, input, 2048U);
    const auto american =
        dp::crr_binomial(dp::OptionType::call, dp::ExerciseStyle::american, input, 2048U);
    expect_true(american.price > european.price + 0.01,
                "negative-rate American call did not have a material exercise premium");
    expect_true(american.early_exercise_nodes > 0U,
                "negative-rate American call did not report an exercise region");
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

void test_crr_price_only_matches_diagnostic_engine() {
    const std::vector<dp::CrrPriceRequest> requests{
        {
            dp::OptionType::put,
            dp::ExerciseStyle::american,
            {100.0, 100.0, 1.0, 0.05, 0.0, 0.2},
            257U,
        },
        {
            dp::OptionType::call,
            dp::ExerciseStyle::american,
            {90.0, 100.0, 0.25, 0.01, 0.08, 0.35},
            384U,
        },
        {
            dp::OptionType::call,
            dp::ExerciseStyle::european,
            {120.0, 100.0, 2.0, -0.01, 0.02, 0.5},
            513U,
        },
    };

    for (const dp::CrrPriceRequest& request : requests) {
        const dp::CrrResult diagnostic =
            dp::crr_binomial(request.option_type, request.exercise_style, request.input,
                             request.steps);
        const dp::CrrPriceOnlyResult price_only = dp::crr_price_only(request);
        expect_true(price_only.price == diagnostic.price,
                    "price-only CRR changed the scalar price recursion");
        expect_true(price_only.steps == diagnostic.steps,
                    "price-only CRR changed the reported step count");
        expect_true(price_only.risk_neutral_probability ==
                        diagnostic.risk_neutral_probability,
                    "price-only CRR changed the lattice probability");
    }
}

void test_crr_batch_is_ordered_and_thread_deterministic() {
    const std::vector<dp::CrrPriceRequest> requests{
        {
            dp::OptionType::put,
            dp::ExerciseStyle::american,
            {70.0, 100.0, 0.1, 0.03, 0.0, 0.15},
            127U,
        },
        {
            dp::OptionType::call,
            dp::ExerciseStyle::american,
            {90.0, 100.0, 0.5, 0.01, 0.08, 0.35},
            256U,
        },
        {
            dp::OptionType::put,
            dp::ExerciseStyle::american,
            {100.0, 100.0, 1.0, 0.05, 0.0, 0.2},
            513U,
        },
        {
            dp::OptionType::call,
            dp::ExerciseStyle::european,
            {130.0, 100.0, 2.0, -0.01, 0.02, 0.5},
            384U,
        },
        {
            dp::OptionType::put,
            dp::ExerciseStyle::american,
            {110.0, 100.0, 3.0, 0.08, 0.01, 0.6},
            193U,
        },
    };

    const auto serial = dp::crr_price_batch(requests, 1U);
    const auto parallel = dp::crr_price_batch(requests, 4U);
    expect_true(serial.size() == requests.size() && parallel.size() == requests.size(),
                "CRR batch changed the number of rows");

    for (std::size_t index = 0U; index < requests.size(); ++index) {
        const auto scalar = dp::crr_price_only(requests[index]);
        expect_true(serial[index].price == scalar.price &&
                        parallel[index].price == scalar.price,
                    "CRR batch changed a scalar result or output order");
        expect_true(serial[index].steps == requests[index].steps &&
                        parallel[index].steps == requests[index].steps,
                    "CRR batch changed a request step count or output order");
        expect_true(serial[index].risk_neutral_probability ==
                            scalar.risk_neutral_probability &&
                        parallel[index].risk_neutral_probability ==
                            scalar.risk_neutral_probability,
                    "CRR batch changed a lattice probability");
    }
}

void test_crr_batch_error_contract() {
    const dp::CrrPriceRequest valid{
        dp::OptionType::put,
        dp::ExerciseStyle::american,
        {100.0, 100.0, 1.0, 0.05, 0.0, 0.2},
        64U,
    };
    expect_true(dp::crr_price_batch({}, 1U).empty(), "empty CRR batch was not empty");
    expect_invalid_argument(
        [&valid]() {
            static_cast<void>(
                dp::crr_price_batch(std::span<const dp::CrrPriceRequest>(&valid, 1U), 0U));
        },
        "zero CRR batch threads were not rejected");
    expect_invalid_argument(
        [&valid]() {
            static_cast<void>(dp::crr_price_batch(
                std::span<const dp::CrrPriceRequest>(&valid, 1U),
                dp::maximum_crr_batch_threads + 1U));
        },
        "excessive CRR batch threads were not rejected");

    auto invalid = valid;
    invalid.steps = 0U;
    const std::vector<dp::CrrPriceRequest> requests{valid, invalid, valid};
    expect_invalid_argument_contains(
        [&requests]() { static_cast<void>(dp::crr_price_batch(requests, 2U)); },
        "index 1", "CRR batch did not identify the first invalid row");

    auto invalid_option = valid;
    invalid_option.option_type = static_cast<dp::OptionType>(999);
    expect_invalid_argument_contains(
        [&invalid_option]() { static_cast<void>(dp::crr_price_only(invalid_option)); },
        "option type", "price-only CRR accepted an invalid option enum");
    expect_invalid_argument_contains(
        [&valid]() {
            static_cast<void>(
                dp::crr_binomial(static_cast<dp::OptionType>(999), valid.exercise_style,
                                 valid.input, valid.steps));
        },
        "option type", "diagnostic CRR accepted an invalid option enum");

    auto invalid_style = valid;
    invalid_style.exercise_style = static_cast<dp::ExerciseStyle>(999);
    const std::vector<dp::CrrPriceRequest> invalid_batch{valid, invalid_style};
    expect_invalid_argument_contains(
        [&invalid_batch]() { static_cast<void>(dp::crr_price_batch(invalid_batch, 2U)); },
        "index 1", "CRR batch accepted an invalid exercise-style enum");
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

[[nodiscard]] dp::LsmConfig small_lsm_config() {
    return {
        32U,
        4'096U,
        8'192U,
        2U,
        0x123456789abcdef0ULL,
        0x0fedcba987654321ULL,
        8U * 1'024U * 1'024U,
    };
}

void test_lsm_is_deterministic_and_pair_aware() {
    const dp::BlackScholesInput input{100.0, 100.0, 1.0, 0.05, 0.0, 0.2};
    const dp::LsmConfig config = small_lsm_config();
    const dp::LsmResult first =
        dp::least_squares_monte_carlo(dp::OptionType::put, input, config);
    const dp::LsmResult second =
        dp::least_squares_monte_carlo(dp::OptionType::put, input, config);
    expect_true(first.price == second.price &&
                    first.standard_error == second.standard_error &&
                    first.raw_price == second.raw_price,
                "LSM repeated run was not bit deterministic");
    expect_true(first.independent_valuation_pairs == config.valuation_paths / 2U,
                "LSM did not use antithetic pair averages as independent samples");
    expect_true(first.estimated_training_working_set_bytes ==
                    dp::lsm_training_memory_bytes(config),
                "LSM reported the wrong training working-set estimate");
    expect_true(first.regressions.size() == config.exercise_steps - 1U,
                "LSM did not report every fitted exercise step");
    expect_true(first.confidence_interval_lower <= first.price &&
                    first.price <= first.confidence_interval_upper &&
                    first.standard_error > 0.0,
                "LSM confidence interval is invalid");
}

void test_lsm_no_dividend_call_control_is_exact() {
    const dp::BlackScholesInput input{100.0, 100.0, 1.0, 0.05, 0.0, 0.2};
    const dp::LsmResult result =
        dp::least_squares_monte_carlo(
            dp::OptionType::call, input, small_lsm_config()
        );
    const double analytic = dp::black_scholes(dp::OptionType::call, input).price;
    expect_near(result.price, analytic, 1.0e-12,
                "LSM European control did not recover a no-dividend call");
    expect_near(result.standard_error, 0.0, 1.0e-14,
                "LSM no-dividend call retained adjusted sampling error");
    expect_true(result.valuation_early_exercise_paths == 0U,
                "LSM no-dividend call exercised early");
}

void test_lsm_negative_rate_put_control_is_exact() {
    const dp::BlackScholesInput input{100.0, 100.0, 1.0, -0.01, 0.0, 0.2};
    const dp::LsmResult result =
        dp::least_squares_monte_carlo(
            dp::OptionType::put, input, small_lsm_config()
        );
    const double analytic = dp::black_scholes(dp::OptionType::put, input).price;
    expect_near(result.price, analytic, 1.0e-12,
                "LSM European control did not recover a negative-rate put");
    expect_near(result.standard_error, 0.0, 1.0e-14,
                "LSM negative-rate put retained adjusted sampling error");
    expect_true(result.valuation_early_exercise_paths == 0U,
                "LSM negative-rate put exercised early");
}

void test_lsm_american_put_cross_checks_crr() {
    const dp::BlackScholesInput input{100.0, 100.0, 1.0, 0.05, 0.0, 0.2};
    dp::LsmConfig config = small_lsm_config();
    config.exercise_steps = 64U;
    config.training_paths = 16'384U;
    config.valuation_paths = 32'768U;
    config.maximum_training_memory_bytes = 16U * 1'024U * 1'024U;
    const dp::LsmResult lsm =
        dp::least_squares_monte_carlo(dp::OptionType::put, input, config);
    const double crr =
        dp::crr_adjacent_step_estimate(
            dp::OptionType::put, dp::ExerciseStyle::american, input, 2048U
        )
            .average_price;
    expect_true(lsm.price <= crr + 4.0 * lsm.standard_error + 0.02,
                "LSM put materially exceeded the high-step CRR cross-check");
    expect_true(crr - lsm.price < 0.15,
                "LSM put was too far below the high-step CRR cross-check");
    expect_true(lsm.valuation_early_exercise_paths > 0U,
                "LSM American put never exercised early");
}

void test_lsm_can_choose_immediate_exercise() {
    const dp::BlackScholesInput input{50.0, 100.0, 1.0, 0.20, 0.0, 0.05};
    const dp::LsmResult result =
        dp::least_squares_monte_carlo(
            dp::OptionType::put, input, small_lsm_config()
        );
    expect_true(result.exercise_at_zero, "LSM did not choose immediate exercise");
    expect_near(result.price, 50.0, 0.0, "LSM immediate-exercise value");
    expect_near(result.standard_error, 0.0, 0.0,
                "LSM immediate exercise retained Monte Carlo error");

    // Immediate exercise is deterministic. Both the raw and adjusted variances
    // are zero, so the variance-reduction ratio is the undefined form 0/0 and
    // must be flagged inapplicable rather than carrying a placeholder that a
    // downstream minimum could mistake for a measurement.
    expect_true(!result.variance_reduction_applicable,
                "LSM marked an undefined 0/0 variance reduction as applicable");
    expect_near(result.variance_reduction_ratio, 0.0, 0.0,
                "LSM immediate exercise reported a variance-reduction value");

    // No valuation simulation runs on this branch, so no European Monte Carlo
    // observation exists. The analytic price must still be reported.
    expect_true(!result.european_monte_carlo_sampled,
                "LSM claimed a European Monte Carlo sample without simulating");
    expect_near(result.european_monte_carlo_price, 0.0, 0.0,
                "LSM reported an unsampled European Monte Carlo price");
    expect_near(result.european_standard_error, 0.0, 0.0,
                "LSM reported an unsampled European standard error");
    const double analytic =
        dp::black_scholes(dp::OptionType::put, input).price;
    expect_near(result.european_analytic_price, analytic, 1.0e-12,
                "LSM dropped the analytic European price on immediate exercise");
    expect_true(result.european_analytic_price != result.european_monte_carlo_price,
                "LSM presented the analytic price as a Monte Carlo observation");
}

void test_lsm_marks_stochastic_variance_reduction_applicable() {
    const dp::BlackScholesInput input{100.0, 100.0, 1.0, 0.05, 0.0, 0.2};
    const dp::LsmResult result =
        dp::least_squares_monte_carlo(
            dp::OptionType::put, input, small_lsm_config()
        );
    expect_true(!result.exercise_at_zero, "LSM exercised a fair put immediately");
    expect_true(result.variance_reduction_applicable,
                "LSM marked a genuinely stochastic case inapplicable");
    expect_true(result.european_monte_carlo_sampled,
                "LSM did not record a sampled European Monte Carlo estimate");
    expect_true(result.standard_error > 0.0,
                "LSM reported no valuation sampling error");
    expect_true(result.variance_reduction_ratio > 0.0,
                "LSM reported a non-positive applicable variance reduction");

    // Structural suppression makes the adjusted estimator exact while the raw
    // estimator still varies. That is a measurement of complete variance
    // removal, so it stays applicable and infinite.
    const dp::BlackScholesInput suppressed{100.0, 100.0, 1.0, 0.05, 0.0, 0.2};
    const dp::LsmResult exact =
        dp::least_squares_monte_carlo(
            dp::OptionType::call, suppressed, small_lsm_config()
        );
    expect_true(exact.raw_standard_error > 0.0,
                "structural suppression left no raw sampling variation");
    expect_true(exact.variance_reduction_applicable,
                "LSM marked a complete variance removal inapplicable");
    expect_true(std::isinf(exact.variance_reduction_ratio),
                "LSM did not report complete variance removal as infinite");
}

// A contract so far out of the money that every valuation pair pays exactly
// zero leaves nothing to reduce: raw and adjusted variances are both zero, so
// the ratio is the same undefined 0/0 as immediate exercise and must not be
// reported as complete variance removal.
void test_lsm_deterministic_raw_estimator_has_no_variance_reduction() {
    const dp::BlackScholesInput input{100.0, 1.0e-6, 1.0, 0.05, 0.0, 0.2};
    const dp::LsmResult result =
        dp::least_squares_monte_carlo(
            dp::OptionType::put, input, small_lsm_config()
        );
    expect_true(!result.exercise_at_zero,
                "deep out-of-the-money put exercised immediately");
    expect_near(result.raw_standard_error, 0.0, 0.0,
                "deep out-of-the-money put retained raw sampling variation");
    expect_near(result.standard_error, 0.0, 0.0,
                "deep out-of-the-money put retained adjusted sampling error");
    expect_true(!result.variance_reduction_applicable,
                "LSM reported an undefined 0/0 ratio as applicable");
    expect_true(!std::isinf(result.variance_reduction_ratio),
                "LSM claimed variance removal where there was no variance");
}

// The shared vanilla validator admits every finite rate, dividend yield and
// volatility. A finite log spot does not imply a representable spot, so each of
// these once returned a silent NaN or a corrupted statistic.
void test_lsm_rejects_non_finite_price_domain_quantities() {
    const dp::LsmConfig config = small_lsm_config();

    // exp(log_spot) overflows while the log spot itself stays finite, and the
    // matching discount factor underflows to zero: the product was inf * 0.
    expect_overflow_error(
        [&config]() {
            static_cast<void>(dp::least_squares_monte_carlo(
                dp::OptionType::call,
                dp::BlackScholesInput{100.0, 100.0, 1.0, 750.0, 0.0, 0.2},
                config));
        },
        "LSM accepted a rate that overflows the simulated spot");

    // A large negative dividend yield drives the same linear-spot overflow
    // through the drift term instead of the rate.
    expect_overflow_error(
        [&config]() {
            static_cast<void>(dp::least_squares_monte_carlo(
                dp::OptionType::call,
                dp::BlackScholesInput{100.0, 100.0, 1.0, 0.05, -1000.0, 0.2},
                config));
        },
        "LSM accepted a dividend yield that overflows the simulated spot");

    // A finite but extreme volatility makes the drift itself non-finite.
    expect_overflow_error(
        [&config]() {
            static_cast<void>(dp::least_squares_monte_carlo(
                dp::OptionType::put,
                dp::BlackScholesInput{100.0, 100.0, 1.0, 0.05, 0.0, 1.0e160},
                config));
        },
        "LSM accepted a volatility that makes the drift non-finite");

    // A large negative rate overflows the discount factor rather than the spot.
    expect_overflow_error(
        [&config]() {
            static_cast<void>(dp::least_squares_monte_carlo(
                dp::OptionType::put,
                dp::BlackScholesInput{100.0, 100.0, 1.0, -750.0, 0.0, 0.2},
                config));
        },
        "LSM accepted a rate that overflows the discount factor");
}

// Extreme but representable inputs must still produce a completely finite
// result rather than a silently corrupted one.
void test_lsm_extreme_but_representable_inputs_stay_finite() {
    const dp::BlackScholesInput input{100.0, 100.0, 1.0, 0.05, -50.0, 0.2};
    const dp::LsmResult result =
        dp::least_squares_monte_carlo(
            dp::OptionType::call, input, small_lsm_config()
        );
    expect_true(std::isfinite(result.price) && result.price > 0.0,
                "LSM produced a non-finite price for a representable input");
    expect_true(std::isfinite(result.standard_error) &&
                    result.standard_error >= 0.0,
                "LSM produced a non-finite standard error");
    expect_true(std::isfinite(result.confidence_interval_lower) &&
                    std::isfinite(result.confidence_interval_upper),
                "LSM produced a non-finite confidence interval");
    expect_true(std::isfinite(result.raw_price) &&
                    std::isfinite(result.european_analytic_price),
                "LSM produced a non-finite reported component");
}

// A deep out-of-the-money contract has no in-the-money training paths at any
// exercise date, so every regression takes the zero-observation fallback.
void test_lsm_reports_zero_observation_constant_fallback() {
    const dp::BlackScholesInput input{100.0, 40.0, 1.0, 0.05, 0.0, 0.2};
    const dp::LsmResult result =
        dp::least_squares_monte_carlo(
            dp::OptionType::put, input, small_lsm_config()
        );
    expect_true(result.regressions.size() == small_lsm_config().exercise_steps - 1U,
                "LSM did not report one diagnostic per interior exercise date");
    std::size_t fallbacks = 0U;
    for (const dp::LsmRegressionDiagnostic& row : result.regressions) {
        expect_true(row.in_the_money_paths == 0U,
                    "deep out-of-the-money put had in-the-money training paths");
        expect_true(row.used_constant_fallback,
                    "LSM did not fall back without any observation");
        expect_true(row.regression_rank == 0U,
                    "zero-observation fallback reported a non-zero rank");
        expect_true(row.coefficients.size() == 1U,
                    "constant fallback did not collapse to one coefficient");
        // The mean of an empty response set is exactly zero.
        expect_near(row.coefficients[0], 0.0, 0.0,
                    "zero-observation fallback coefficient was not zero");
        expect_near(row.minimum_relative_r_diagonal, 0.0, 0.0,
                    "fallback reported a fitted conditioning diagnostic");
        ++fallbacks;
    }
    expect_true(fallbacks > 0U, "no regression fallback was exercised");
}

// A strike just inside the simulated range leaves a handful of in-the-money
// paths at late dates: fewer observations than basis columns, which is the
// rank-deficient fallback branch rather than the empty-sample branch.
void test_lsm_reports_rank_deficient_constant_fallback() {
    const dp::BlackScholesInput input{100.0, 60.0, 1.0, 0.05, 0.0, 0.2};
    const dp::LsmConfig config = small_lsm_config();
    const dp::LsmResult result =
        dp::least_squares_monte_carlo(dp::OptionType::put, input, config);
    const double time_step =
        input.maturity / static_cast<double>(config.exercise_steps);
    std::size_t observed_fallbacks = 0U;
    for (const dp::LsmRegressionDiagnostic& row : result.regressions) {
        if (row.in_the_money_paths == 0U || !row.used_constant_fallback) {
            continue;
        }
        ++observed_fallbacks;
        expect_true(row.in_the_money_paths < config.polynomial_degree + 1U,
                    "a full-rank sample took the constant fallback");
        expect_true(row.regression_rank == 1U,
                    "observed constant fallback did not report unit rank");
        expect_true(row.coefficients.size() == 1U,
                    "constant fallback did not collapse to one coefficient");
        expect_true(std::isfinite(row.coefficients[0]),
                    "constant fallback coefficient was not finite");
        // The coefficient is the mean discounted continuation response. Every
        // response is a put payoff, bounded by the strike, discounted over at
        // least one exercise interval, so an undiscounted or mis-indexed cash
        // flow would breach this bound.
        const double maximum_discounted_response =
            input.strike * std::exp(-input.rate * time_step);
        expect_true(row.coefficients[0] >= 0.0 &&
                        row.coefficients[0] <= maximum_discounted_response,
                    "constant fallback coefficient was not a mean discounted "
                    "response");
    }
    expect_true(observed_fallbacks > 0U,
                "no rank-deficient regression fallback was exercised");

    // A fitted, non-degenerate regression must report a strictly positive
    // relative diagonal; only fallbacks report zero.
    std::size_t fitted = 0U;
    for (const dp::LsmRegressionDiagnostic& row : result.regressions) {
        if (row.used_constant_fallback) {
            continue;
        }
        ++fitted;
        expect_true(row.minimum_relative_r_diagonal > 0.0 &&
                        row.minimum_relative_r_diagonal <= 1.0,
                    "fitted regression reported an invalid relative diagonal");
        expect_true(row.regression_rank == config.polynomial_degree + 1U,
                    "fitted regression did not report full rank");
    }
    expect_true(fitted > 0U, "no regression was fitted at full rank");
}

// The early-exercise premium is a far stronger regression signal than the price
// level: the large European component cancels on both sides, so a discount-time
// or stopping-index error shows up first order instead of being buried under
// diffusion noise.
void test_lsm_american_put_premium_matches_crr_premium() {
    const dp::BlackScholesInput input{100.0, 100.0, 1.0, 0.05, 0.0, 0.2};
    dp::LsmConfig config = small_lsm_config();
    config.exercise_steps = 64U;
    config.training_paths = 16'384U;
    config.valuation_paths = 32'768U;
    config.maximum_training_memory_bytes = 16U * 1'024U * 1'024U;
    const dp::LsmResult lsm =
        dp::least_squares_monte_carlo(dp::OptionType::put, input, config);
    const double american =
        dp::crr_adjacent_step_estimate(
            dp::OptionType::put, dp::ExerciseStyle::american, input, 2048U
        )
            .average_price;
    const double european =
        dp::crr_adjacent_step_estimate(
            dp::OptionType::put, dp::ExerciseStyle::european, input, 2048U
        )
            .average_price;
    const double lsm_premium = lsm.price - lsm.european_analytic_price;
    const double crr_premium = american - european;

    expect_true(lsm_premium > 0.0,
                "LSM American put showed no early-exercise premium");
    // A learned policy on a coarser grid cannot materially exceed the
    // near-continuous reference premium.
    expect_true(lsm_premium <= crr_premium + 4.0 * lsm.standard_error,
                "LSM early-exercise premium materially exceeded CRR");
    // For this configuration the observed gap is 0.0619 with a standard error
    // of 0.0207, so the bound keeps about 1.6x headroom. Cross-platform libm
    // differences act at the ULP level on a mean over 16,384 pairs, orders of
    // magnitude below the remaining 0.038 margin, while a mis-discounted or
    // mis-indexed exercise cash flow moves the premium far more than that.
    expect_true(crr_premium - lsm_premium < 0.10,
                "LSM early-exercise premium fell far below CRR");
}

// The suppression proof for puts requires r <= 0 and q >= 0. Check the strict
// interior of that region, not only the r < 0, q = 0 boundary.
void test_lsm_negative_rate_positive_dividend_put_control_is_exact() {
    const dp::BlackScholesInput input{100.0, 100.0, 1.0, -0.02, 0.03, 0.25};
    const dp::LsmResult result =
        dp::least_squares_monte_carlo(
            dp::OptionType::put, input, small_lsm_config()
        );
    const double analytic = dp::black_scholes(dp::OptionType::put, input).price;
    expect_near(result.price, analytic, 1.0e-12,
                "LSM did not suppress early exercise for r<0 and q>0");
    expect_near(result.standard_error, 0.0, 1.0e-14,
                "LSM retained sampling error under structural suppression");
    expect_true(result.valuation_early_exercise_paths == 0U,
                "LSM exercised early where continuation strictly dominates");
}

void test_invalid_lsm_requests_are_rejected() {
    const dp::BlackScholesInput input{100.0, 100.0, 1.0, 0.05, 0.0, 0.2};
    dp::LsmConfig config = small_lsm_config();
    config.training_seed = config.valuation_seed;
    expect_invalid_argument_contains(
        [&input, &config]() {
            static_cast<void>(
                dp::least_squares_monte_carlo(dp::OptionType::put, input, config)
            );
        },
        "seeds", "LSM accepted identical training and valuation seeds");

    config = small_lsm_config();
    config.training_paths = 4'098U;
    config.maximum_training_memory_bytes = 1U;
    expect_invalid_argument_contains(
        [&input, &config]() {
            static_cast<void>(
                dp::least_squares_monte_carlo(dp::OptionType::put, input, config)
            );
        },
        "exceeding", "LSM ignored its training-memory guard");

    config = small_lsm_config();
    config.polynomial_degree = 4U;
    expect_invalid_argument_contains(
        [&input, &config]() {
            static_cast<void>(
                dp::least_squares_monte_carlo(dp::OptionType::put, input, config)
            );
        },
        "degree", "LSM accepted an unsupported basis degree");
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
        {"negative-rate call exercise premium",
         test_negative_rate_call_can_have_exercise_premium},
        {"American bounds and monotonicity", test_american_prices_respect_bounds_and_monotonicity},
        {"American CRR step convergence", test_american_crr_step_convergence},
        {"CRR price-only parity", test_crr_price_only_matches_diagnostic_engine},
        {"CRR batch determinism", test_crr_batch_is_ordered_and_thread_deterministic},
        {"CRR batch error contract", test_crr_batch_error_contract},
        {"invalid CRR requests", test_invalid_crr_requests_are_rejected},
        {"LSM deterministic pair-aware estimate",
         test_lsm_is_deterministic_and_pair_aware},
        {"LSM no-dividend call control", test_lsm_no_dividend_call_control_is_exact},
        {"LSM negative-rate put control",
         test_lsm_negative_rate_put_control_is_exact},
        {"LSM negative-rate positive-dividend put control",
         test_lsm_negative_rate_positive_dividend_put_control_is_exact},
        {"LSM American put CRR cross-check", test_lsm_american_put_cross_checks_crr},
        {"LSM American put premium cross-check",
         test_lsm_american_put_premium_matches_crr_premium},
        {"LSM immediate exercise", test_lsm_can_choose_immediate_exercise},
        {"LSM stochastic variance reduction applicability",
         test_lsm_marks_stochastic_variance_reduction_applicable},
        {"LSM deterministic raw estimator variance reduction",
         test_lsm_deterministic_raw_estimator_has_no_variance_reduction},
        {"LSM non-finite price-domain rejection",
         test_lsm_rejects_non_finite_price_domain_quantities},
        {"LSM extreme representable inputs stay finite",
         test_lsm_extreme_but_representable_inputs_stay_finite},
        {"LSM zero-observation constant fallback",
         test_lsm_reports_zero_observation_constant_fallback},
        {"LSM rank-deficient constant fallback",
         test_lsm_reports_rank_deficient_constant_fallback},
        {"invalid LSM requests", test_invalid_lsm_requests_are_rejected},
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
