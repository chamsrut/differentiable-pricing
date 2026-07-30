#include "dp/binomial_tree.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace dp {
namespace {

[[nodiscard]] double payoff(const OptionType option_type, const double spot, const double strike) {
    if (option_type == OptionType::call) {
        return std::max(spot - strike, 0.0);
    }
    return std::max(strike - spot, 0.0);
}

void validate_steps(const std::size_t steps) {
    if (steps == 0U) {
        throw std::invalid_argument("CRR steps must be positive");
    }
    if (steps > maximum_crr_steps) {
        throw std::invalid_argument("CRR steps exceed the scalar reference-engine limit");
    }
}

void require_finite(const double value, const char* const description) {
    if (!std::isfinite(value)) {
        throw std::overflow_error(description);
    }
}

void require_positive_finite(const double value, const char* const description) {
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::overflow_error(description);
    }
}

}  // namespace

CrrResult crr_binomial(const OptionType option_type, const ExerciseStyle exercise_style,
                       const VanillaOptionInput& input, const std::size_t steps) {
    input.validate();
    validate_steps(steps);

    const double time_step = input.maturity / static_cast<double>(steps);
    const double up_factor = std::exp(input.volatility * std::sqrt(time_step));
    const double down_factor = 1.0 / up_factor;
    const double growth_factor = std::exp((input.rate - input.dividend_yield) * time_step);
    const double discount_factor = std::exp(-input.rate * time_step);
    require_finite(up_factor, "CRR up factor overflowed");
    require_finite(growth_factor, "CRR growth factor overflowed");
    require_finite(discount_factor, "CRR discount factor overflowed");

    const double denominator = up_factor - down_factor;
    const double risk_neutral_probability = (growth_factor - down_factor) / denominator;
    if (!std::isfinite(risk_neutral_probability) || risk_neutral_probability <= 0.0 ||
        risk_neutral_probability >= 1.0) {
        throw std::invalid_argument(
            "CRR risk-neutral probability must lie strictly between zero "
            "and one; increase the step count or revise the model inputs");
    }

    std::vector<double> option_values(steps + 1U);
    std::vector<double> node_spots;
    if (exercise_style == ExerciseStyle::american) {
        node_spots.resize(steps + 1U);
    }
    double terminal_spot = input.spot * std::pow(down_factor, static_cast<double>(steps));
    require_positive_finite(terminal_spot,
                            "CRR lowest terminal spot became non-positive or non-finite");
    const double adjacent_spot_ratio = up_factor / down_factor;
    require_finite(adjacent_spot_ratio, "CRR adjacent spot ratio overflowed");

    for (std::size_t node = 0U; node <= steps; ++node) {
        if (node != 0U) {
            terminal_spot *= adjacent_spot_ratio;
            require_finite(terminal_spot, "CRR terminal spot overflowed");
        }
        if (exercise_style == ExerciseStyle::american) {
            node_spots[node] = terminal_spot;
        }
        option_values[node] = payoff(option_type, terminal_spot, input.strike);
    }

    std::size_t early_exercise_nodes = 0U;
    std::optional<std::size_t> earliest_exercise_step;
    std::vector<std::optional<double>> exercise_boundary_by_step;
    if (exercise_style == ExerciseStyle::american) {
        exercise_boundary_by_step.resize(steps);
    }
    for (std::size_t level = steps; level-- > 0U;) {
        for (std::size_t node = 0U; node <= level; ++node) {
            const double continuation_value =
                discount_factor * ((1.0 - risk_neutral_probability) * option_values[node] +
                                   risk_neutral_probability * option_values[node + 1U]);
            require_finite(continuation_value, "CRR continuation value overflowed");

            if (exercise_style == ExerciseStyle::european) {
                option_values[node] = continuation_value;
                continue;
            }

            node_spots[node] /= down_factor;
            require_finite(node_spots[node], "CRR node spot overflowed");
            const double exercise_value = payoff(option_type, node_spots[node], input.strike);
            option_values[node] = std::max(exercise_value, continuation_value);
            const double comparison_scale =
                std::max({1.0, std::abs(exercise_value), std::abs(continuation_value)});
            const double exercise_advantage = exercise_value - continuation_value;
            if (exercise_advantage >
                crr_exercise_diagnostic_relative_tolerance * comparison_scale) {
                ++early_exercise_nodes;
                std::optional<double>& boundary = exercise_boundary_by_step[level];
                if (!boundary.has_value() ||
                    (option_type == OptionType::call && node_spots[node] < *boundary) ||
                    (option_type == OptionType::put && node_spots[node] > *boundary)) {
                    boundary = node_spots[node];
                }
                if (!earliest_exercise_step.has_value() || level < *earliest_exercise_step) {
                    earliest_exercise_step = level;
                }
            }
        }
    }

    return {
        option_values[0],
        steps,
        up_factor,
        down_factor,
        risk_neutral_probability,
        discount_factor,
        early_exercise_nodes,
        earliest_exercise_step,
        std::move(exercise_boundary_by_step),
    };
}

CrrAdjacentStepEstimate crr_adjacent_step_estimate(const OptionType option_type,
                                                   const ExerciseStyle exercise_style,
                                                   const VanillaOptionInput& input,
                                                   const std::size_t steps) {
    if (steps >= maximum_crr_steps) {
        throw std::invalid_argument(
            "adjacent-step CRR estimate requires steps below the scalar "
            "reference-engine limit");
    }
    CrrResult at_steps = crr_binomial(option_type, exercise_style, input, steps);
    CrrResult at_steps_plus_one = crr_binomial(option_type, exercise_style, input, steps + 1U);
    const double average_price = 0.5 * (at_steps.price + at_steps_plus_one.price);
    const double absolute_pair_gap = std::abs(at_steps.price - at_steps_plus_one.price);
    return {
        std::move(at_steps),
        std::move(at_steps_plus_one),
        average_price,
        absolute_pair_gap,
    };
}

}  // namespace dp
