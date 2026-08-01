#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "dp/option.hpp"

namespace dp {

inline constexpr std::size_t maximum_lsm_exercise_steps = 1'024U;
inline constexpr std::size_t maximum_lsm_polynomial_degree = 3U;
inline constexpr std::size_t maximum_lsm_paths = 16'777'216U;
inline constexpr std::size_t maximum_lsm_training_memory_bytes =
    std::size_t{8U} * 1'024U * 1'024U * 1'024U;
inline constexpr double lsm_confidence_level = 0.95;
inline constexpr double lsm_normal_confidence_quantile = 1.959963984540054;

struct LsmConfig {
    std::size_t exercise_steps;
    std::size_t training_paths;
    std::size_t valuation_paths;
    std::size_t polynomial_degree;
    std::uint64_t training_seed;
    std::uint64_t valuation_seed;
    std::size_t maximum_training_memory_bytes;
};

struct LsmRegressionDiagnostic {
    std::size_t exercise_step;
    std::size_t in_the_money_paths;
    std::size_t regression_rank;
    bool used_constant_fallback;
    double state_mean;
    double state_scale;
    double minimum_relative_r_diagonal;
    std::vector<double> coefficients;
};

struct LsmResult {
    // The primary estimate uses a European control-variate coefficient fitted
    // on the training stream and frozen before valuation. The raw fixed-policy
    // estimate is reported separately.
    //
    // confidence_interval_lower/upper describe valuation sampling error for the
    // already-frozen policy only. They are not a calibration interval for the
    // same-grid optimal value, for a continuously exercisable American value,
    // or for any tree reference. When the policy exercises at time zero, or
    // when the control variate reproduces the payoff exactly, standard_error is
    // zero and the interval degenerates to a point.
    double price;
    double standard_error;
    double confidence_interval_lower;
    double confidence_interval_upper;
    double raw_price;
    double raw_standard_error;
    // european_monte_carlo_price and european_standard_error are meaningful
    // only when european_monte_carlo_sampled is true. An immediate-exercise
    // policy runs no valuation simulation, so no European Monte Carlo
    // observation exists; both fields are then zero and must not be reported as
    // sampled quantities. european_analytic_price is always the closed-form
    // Black--Scholes value and is always meaningful.
    bool european_monte_carlo_sampled;
    double european_monte_carlo_price;
    double european_analytic_price;
    double european_standard_error;
    double control_variate_coefficient;
    // variance_reduction_ratio is meaningful only when
    // variance_reduction_applicable is true. An immediate-exercise policy has
    // zero raw and zero adjusted variance, so the ratio is the undefined form
    // 0/0; the field is then zero and must be ignored. An applicable control
    // variate that removes all variance reports a positive infinity, which
    // callers serializing to JSON must map explicitly rather than silently.
    bool variance_reduction_applicable;
    double variance_reduction_ratio;
    double training_continuation_value_at_zero;
    bool exercise_at_zero;
    std::size_t exercise_steps;
    std::size_t training_paths;
    std::size_t valuation_paths;
    std::size_t independent_valuation_pairs;
    std::size_t estimated_training_working_set_bytes;
    std::size_t valuation_early_exercise_paths;
    std::vector<LsmRegressionDiagnostic> regressions;
};

// Estimate a discretely exercisable American-option value with the
// Longstaff--Schwartz least-squares Monte Carlo algorithm.
//
// The stopping policy is learned on training_paths and evaluated on an
// independent valuation stream. Both samples use antithetic path pairs.
// Standard errors treat each pair average as one independent observation.
// The valuation stream is O(exercise_steps) memory. The explicit memory guard
// covers the deterministic estimate of all bulk policy-training allocations,
// including the path matrix and maximum-size regression workspaces. It is
// evaluated once, before any bulk allocation.
//
// Throws std::invalid_argument for an invalid option input or configuration.
//
// Throws std::overflow_error when any simulated or derived price-domain
// quantity leaves double precision. The shared VanillaOptionInput validator
// admits every finite rate, dividend yield and volatility, so a finite log spot
// alone is not sufficient: exp(log_spot) can overflow while its logarithm stays
// finite, and an extreme rate can drive a discount factor out of range. Every
// exponentiated spot, discount factor, intrinsic value, discounted cash flow,
// regression response, control-variate observation, pair average and reported
// summary statistic is checked before it can propagate. This function never
// returns a non-finite price, standard error or interval bound.
[[nodiscard]] LsmResult least_squares_monte_carlo(
    OptionType option_type,
    const VanillaOptionInput& input,
    const LsmConfig& config
);

// Deterministic upper estimate of bulk policy-training allocations. Allocator
// metadata and implementation-specific std::vector object overhead are not
// included.
[[nodiscard]] std::size_t lsm_training_memory_bytes(const LsmConfig& config);

}  // namespace dp
