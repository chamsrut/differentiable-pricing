#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#include "dp/option.hpp"

namespace dp {

// The tree is O(steps^2) in work. This operational ceiling prevents accidental
// requests whose run time is incompatible with the scalar reference API.
inline constexpr std::size_t maximum_crr_steps = 16'384U;
// A defensive operational ceiling. Batch callers choose the worker count
// explicitly; the implementation never silently substitutes hardware
// concurrency or creates an unbounded number of threads.
inline constexpr std::size_t maximum_crr_batch_threads = 256U;
// Exercise metadata ignores advantages at or below this scale-relative
// threshold. The price recursion still uses the exact nodewise maximum.
inline constexpr double crr_exercise_diagnostic_relative_tolerance = 1.0e-12;

struct CrrLatticeParameters {
    double time_step;
    double up_factor;
    double down_factor;
    double risk_neutral_probability;
    double discount_factor;
};

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

struct CrrPriceRequest {
    OptionType option_type;
    ExerciseStyle exercise_style;
    VanillaOptionInput input;
    std::size_t steps;
};

struct CrrPriceOnlyResult {
    double price;
    std::size_t steps;
    double risk_neutral_probability;
};

// Construct and validate the CRR time-step parameters without traversing the
// tree. This is the authoritative feasibility check used by convergence
// diagnostics; callers must not reimplement the probability formula.
[[nodiscard]] CrrLatticeParameters crr_lattice_parameters(const VanillaOptionInput& input,
                                                          std::size_t steps);

// Cox-Ross-Rubinstein discretization of the same constant-parameter lognormal
// dynamics used by Black-Scholes. The American branch compares continuation
// and intrinsic value at every tree node.
[[nodiscard]] CrrResult crr_binomial(OptionType option_type, ExerciseStyle exercise_style,
                                     const VanillaOptionInput& input, std::size_t steps);

// Run the identical price recursion without retaining exercise-boundary
// diagnostics. This remains O(steps^2) work and O(steps) memory.
[[nodiscard]] CrrPriceOnlyResult crr_price_only(const CrrPriceRequest& request);

// Price independent contracts in deterministic input order. Every request is
// validated before worker threads start. Each worker reuses its own O(steps)
// workspace, and each individual tree remains serial so changing thread_count
// cannot change floating-point operation order within a price.
[[nodiscard]] std::vector<CrrPriceOnlyResult> crr_price_batch(
    std::span<const CrrPriceRequest> requests, std::size_t thread_count);

// Adjacent step counts change terminal-lattice alignment, time discretization,
// and the American exercise frontier together. Their average and discrepancy
// are useful refinement diagnostics, not formal numerical error bounds or pure
// parity measurements.
[[nodiscard]] CrrAdjacentStepEstimate crr_adjacent_step_estimate(OptionType option_type,
                                                                 ExerciseStyle exercise_style,
                                                                 const VanillaOptionInput& input,
                                                                 std::size_t steps);

}  // namespace dp
