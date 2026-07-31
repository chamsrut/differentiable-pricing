#include "dp/binomial_tree.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace dp {
namespace {

struct CrrWorkspace {
    std::vector<double> option_values;
    std::vector<double> node_spots;
};

struct PreparedCrr {
    CrrLatticeParameters lattice;
    double lowest_terminal_spot;
    double adjacent_spot_ratio;
};

[[nodiscard]] double payoff(const OptionType option_type, const double spot, const double strike) {
    if (option_type == OptionType::call) {
        return std::max(spot - strike, 0.0);
    }
    return std::max(strike - spot, 0.0);
}

void validate_option_type(const OptionType option_type) {
    if (option_type != OptionType::call && option_type != OptionType::put) {
        throw std::invalid_argument("unsupported option type");
    }
}

void validate_exercise_style(const ExerciseStyle exercise_style) {
    if (exercise_style != ExerciseStyle::european &&
        exercise_style != ExerciseStyle::american) {
        throw std::invalid_argument("unsupported exercise style");
    }
}

void validate_contract_enums(const OptionType option_type,
                             const ExerciseStyle exercise_style) {
    validate_option_type(option_type);
    validate_exercise_style(exercise_style);
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

[[nodiscard]] PreparedCrr prepare_crr(const VanillaOptionInput& input, const std::size_t steps) {
    const CrrLatticeParameters lattice = crr_lattice_parameters(input, steps);
    const double lowest_terminal_spot =
        input.spot * std::pow(lattice.down_factor, static_cast<double>(steps));
    require_positive_finite(lowest_terminal_spot,
                            "CRR lowest terminal spot became non-positive or non-finite");
    const double adjacent_spot_ratio = lattice.up_factor / lattice.down_factor;
    require_finite(adjacent_spot_ratio, "CRR adjacent spot ratio overflowed");
    return {lattice, lowest_terminal_spot, adjacent_spot_ratio};
}

[[nodiscard]] CrrResult run_prepared_crr(
    const OptionType option_type, const ExerciseStyle exercise_style,
    const VanillaOptionInput& input, const std::size_t steps, const PreparedCrr& prepared,
    CrrWorkspace& workspace, const bool collect_exercise_diagnostics) {
    workspace.option_values.resize(steps + 1U);
    if (exercise_style == ExerciseStyle::american) {
        workspace.node_spots.resize(steps + 1U);
    } else {
        workspace.node_spots.clear();
    }

    double terminal_spot = prepared.lowest_terminal_spot;
    for (std::size_t node = 0U; node <= steps; ++node) {
        if (node != 0U) {
            terminal_spot *= prepared.adjacent_spot_ratio;
            require_finite(terminal_spot, "CRR terminal spot overflowed");
        }
        if (exercise_style == ExerciseStyle::american) {
            workspace.node_spots[node] = terminal_spot;
        }
        workspace.option_values[node] = payoff(option_type, terminal_spot, input.strike);
    }

    std::size_t early_exercise_nodes = 0U;
    std::optional<std::size_t> earliest_exercise_step;
    std::vector<std::optional<double>> exercise_boundary_by_step;
    if (exercise_style == ExerciseStyle::american && collect_exercise_diagnostics) {
        exercise_boundary_by_step.resize(steps);
    }
    for (std::size_t level = steps; level-- > 0U;) {
        for (std::size_t node = 0U; node <= level; ++node) {
            const double continuation_value =
                prepared.lattice.discount_factor *
                ((1.0 - prepared.lattice.risk_neutral_probability) *
                     workspace.option_values[node] +
                 prepared.lattice.risk_neutral_probability * workspace.option_values[node + 1U]);
            require_finite(continuation_value, "CRR continuation value overflowed");

            if (exercise_style == ExerciseStyle::european) {
                workspace.option_values[node] = continuation_value;
                continue;
            }

            workspace.node_spots[node] /= prepared.lattice.down_factor;
            require_finite(workspace.node_spots[node], "CRR node spot overflowed");
            const double exercise_value =
                payoff(option_type, workspace.node_spots[node], input.strike);
            workspace.option_values[node] = std::max(exercise_value, continuation_value);
            if (!collect_exercise_diagnostics) {
                continue;
            }
            const double comparison_scale =
                std::max({1.0, std::abs(exercise_value), std::abs(continuation_value)});
            const double exercise_advantage = exercise_value - continuation_value;
            if (exercise_advantage >
                crr_exercise_diagnostic_relative_tolerance * comparison_scale) {
                ++early_exercise_nodes;
                std::optional<double>& boundary = exercise_boundary_by_step[level];
                if (!boundary.has_value() ||
                    (option_type == OptionType::call && workspace.node_spots[node] < *boundary) ||
                    (option_type == OptionType::put && workspace.node_spots[node] > *boundary)) {
                    boundary = workspace.node_spots[node];
                }
                if (!earliest_exercise_step.has_value() || level < *earliest_exercise_step) {
                    earliest_exercise_step = level;
                }
            }
        }
    }

    return {
        workspace.option_values[0],
        steps,
        prepared.lattice.up_factor,
        prepared.lattice.down_factor,
        prepared.lattice.risk_neutral_probability,
        prepared.lattice.discount_factor,
        early_exercise_nodes,
        earliest_exercise_step,
        std::move(exercise_boundary_by_step),
    };
}

[[nodiscard]] CrrPriceOnlyResult run_price_only(const CrrPriceRequest& request,
                                                const PreparedCrr& prepared,
                                                CrrWorkspace& workspace) {
    const CrrResult result =
        run_prepared_crr(request.option_type, request.exercise_style, request.input, request.steps,
                         prepared, workspace, false);
    return {result.price, result.steps, result.risk_neutral_probability};
}

}  // namespace

CrrLatticeParameters crr_lattice_parameters(const VanillaOptionInput& input,
                                            const std::size_t steps) {
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
    return {
        time_step,
        up_factor,
        down_factor,
        risk_neutral_probability,
        discount_factor,
    };
}

CrrResult crr_binomial(const OptionType option_type, const ExerciseStyle exercise_style,
                       const VanillaOptionInput& input, const std::size_t steps) {
    validate_contract_enums(option_type, exercise_style);
    const PreparedCrr prepared = prepare_crr(input, steps);
    CrrWorkspace workspace;
    return run_prepared_crr(option_type, exercise_style, input, steps, prepared, workspace, true);
}

CrrPriceOnlyResult crr_price_only(const CrrPriceRequest& request) {
    validate_contract_enums(request.option_type, request.exercise_style);
    const PreparedCrr prepared = prepare_crr(request.input, request.steps);
    CrrWorkspace workspace;
    return run_price_only(request, prepared, workspace);
}

std::vector<CrrPriceOnlyResult> crr_price_batch(
    const std::span<const CrrPriceRequest> requests, const std::size_t thread_count) {
    if (thread_count == 0U || thread_count > maximum_crr_batch_threads) {
        throw std::invalid_argument("CRR batch thread count must lie between 1 and 256");
    }
    if (requests.empty()) {
        return {};
    }

    std::vector<PreparedCrr> prepared;
    prepared.reserve(requests.size());
    for (std::size_t index = 0U; index < requests.size(); ++index) {
        try {
            validate_contract_enums(requests[index].option_type,
                                    requests[index].exercise_style);
            prepared.push_back(prepare_crr(requests[index].input, requests[index].steps));
        } catch (const std::exception& error) {
            throw std::invalid_argument("CRR batch request at index " + std::to_string(index) +
                                        " is invalid: " + error.what());
        }
    }

    std::vector<CrrPriceOnlyResult> results(requests.size());
    std::vector<std::exception_ptr> errors(requests.size());
    const std::size_t worker_count = std::min(thread_count, requests.size());
    std::atomic<std::size_t> next_index{0U};

    const auto worker = [&]() {
        CrrWorkspace workspace;
        while (true) {
            const std::size_t index = next_index.fetch_add(1U, std::memory_order_relaxed);
            if (index >= requests.size()) {
                return;
            }
            try {
                results[index] = run_price_only(requests[index], prepared[index], workspace);
            } catch (...) {
                errors[index] = std::current_exception();
            }
        }
    };

    if (worker_count == 1U) {
        worker();
    } else {
        std::vector<std::jthread> workers;
        workers.reserve(worker_count);
        for (std::size_t worker_index = 0U; worker_index < worker_count; ++worker_index) {
            workers.emplace_back(worker);
        }
    }

    for (std::size_t index = 0U; index < errors.size(); ++index) {
        if (errors[index] == nullptr) {
            continue;
        }
        try {
            std::rethrow_exception(errors[index]);
        } catch (const std::exception& error) {
            throw std::runtime_error("CRR batch request at index " + std::to_string(index) +
                                     " failed during pricing: " + error.what());
        }
    }
    return results;
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
