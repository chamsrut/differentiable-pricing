#pragma once

#include <cstddef>
#include <optional>
#include <vector>

#include "dp/option.hpp"

namespace dp {

// The tree is O(steps^2) in work. This operational ceiling prevents accidental
// requests whose run time is incompatible with the scalar reference API.
inline constexpr std::size_t maximum_crr_steps = 16'384U;
// Exercise metadata ignores advantages at or below this scale-relative
// threshold. The price recursion still uses the exact nodewise maximum.
inline constexpr double crr_exercise_diagnostic_relative_tolerance = 1.0e-12;

struct CrrResult {
    double price;
    std::size_t steps;
    double up_factor;
    double down_factor;
    double risk_neutral_probability;
    double discount_factor;
    // These diagnostics count only materially preferable early exercise.
    std::size_t early_exercise_nodes;
    std::optional<std::size_t> earliest_exercise_step;
    // One entry per pre-expiry time layer. For puts this is the greatest
    // exercised spot; for calls it is the smallest exercised spot.
    std::vector<std::optional<double>> exercise_boundary_by_step;
};

struct CrrAdjacentStepEstimate {
    CrrResult at_steps;
    CrrResult at_steps_plus_one;
    double average_price;
    double absolute_pair_gap;
};

// Cox-Ross-Rubinstein discretization of the same constant-parameter lognormal
// dynamics used by Black-Scholes. The American branch compares continuation
// and intrinsic value at every tree node.
[[nodiscard]] CrrResult crr_binomial(OptionType option_type, ExerciseStyle exercise_style,
                                     const VanillaOptionInput& input, std::size_t steps);

// Adjacent step counts often land on opposite sides of CRR's strike-alignment
// oscillation. Their average is a useful convergence diagnostic, not a formal
// numerical error bound.
[[nodiscard]] CrrAdjacentStepEstimate crr_adjacent_step_estimate(OptionType option_type,
                                                                 ExerciseStyle exercise_style,
                                                                 const VanillaOptionInput& input,
                                                                 std::size_t steps);

}  // namespace dp
