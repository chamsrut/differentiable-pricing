#include "dp/least_squares_monte_carlo.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dp/black_scholes.hpp"

namespace dp {
namespace {

constexpr double regression_rank_relative_tolerance = 1.0e-11;

// The shared vanilla-input validator admits every finite rate, dividend yield
// and volatility. Price-domain quantities must therefore be checked where they
// are produced: a finite log spot does not imply a representable spot, and an
// extreme rate can push a discount factor out of double precision. Each helper
// throws before the offending value can reach a regression, an online moment,
// or a reported field.
[[noreturn]] void report_overflow(const char* const operation) {
    throw std::overflow_error(
        std::string("LSM ") + operation + " left double precision"
    );
}

void check_finite(const double value, const char* const operation) {
    if (!std::isfinite(value)) {
        report_overflow(operation);
    }
}

[[nodiscard]] double require_finite(
    const double value,
    const char* const operation
) {
    check_finite(value, operation);
    return value;
}

// exp(x) is representable only strictly below this bound. Checking the log spot
// against it rejects an overflowing spot before the exponential is taken, and
// costs nothing in the simulation inner loop.
const double maximum_representable_log_spot =
    std::log(std::numeric_limits<double>::max());

void require_representable_log_spot(
    const double log_spot,
    const char* const operation
) {
    if (!std::isfinite(log_spot) || log_spot >= maximum_representable_log_spot) {
        report_overflow(operation);
    }
}

[[nodiscard]] double spot_from_log(const double log_spot) {
    const double spot = std::exp(log_spot);
    if (!std::isfinite(spot)) {
        report_overflow("simulated spot");
    }
    return spot;
}

// Validate an exponential without altering how its argument was formed. Callers
// pass the exponent expression unchanged so discounting stays bit-identical.
[[nodiscard]] double checked_exp(
    const double exponent,
    const char* const operation
) {
    const double value = std::exp(exponent);
    if (!std::isfinite(value)) {
        report_overflow(operation);
    }
    return value;
}

struct SplitMix64 {
    std::uint64_t state;

    [[nodiscard]] std::uint64_t next() {
        state += 0x9e3779b97f4a7c15ULL;
        std::uint64_t value = state;
        value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31U);
    }
};

class BoxMullerNormal {
public:
    explicit BoxMullerNormal(const std::uint64_t seed) : engine_{seed} {}

    [[nodiscard]] double next() {
        if (has_spare_) {
            has_spare_ = false;
            return spare_;
        }
        const double first = uniform_open();
        const double second = uniform_open();
        const double radius = std::sqrt(-2.0 * std::log(first));
        const double angle = 2.0 * std::numbers::pi * second;
        spare_ = radius * std::sin(angle);
        has_spare_ = true;
        return radius * std::cos(angle);
    }

private:
    [[nodiscard]] double uniform_open() {
        // Use the leading 53 bits to construct a value strictly in (0, 1).
        constexpr double inverse_two_to_53 = 1.0 / 9'007'199'254'740'992.0;
        const std::uint64_t mantissa = engine_.next() >> 11U;
        return (static_cast<double>(mantissa) + 0.5) * inverse_two_to_53;
    }

    SplitMix64 engine_;
    bool has_spare_{false};
    double spare_{0.0};
};

struct RegressionModel {
    std::size_t step{0U};
    std::size_t observations{0U};
    std::size_t rank{0U};
    bool fallback{true};
    double mean{0.0};
    double scale{1.0};
    double minimum_relative_r_diagonal{0.0};
    std::vector<double> coefficients{0.0};
};

struct Policy {
    bool exercise_at_zero{false};
    bool suppress_early_exercise{false};
    double continuation_at_zero{0.0};
    double control_variate_coefficient{0.0};
    std::vector<RegressionModel> by_step;
};

struct OnlineMoments {
    std::size_t count{0U};
    double mean{0.0};
    double sum_squared_deviation{0.0};

    void add(const double value) {
        check_finite(value, "online moment observation");
        ++count;
        const double difference = value - mean;
        mean += difference / static_cast<double>(count);
        const double updated_difference = value - mean;
        sum_squared_deviation += difference * updated_difference;
        check_finite(mean, "online moment mean");
        check_finite(sum_squared_deviation, "online moment dispersion");
    }

    [[nodiscard]] double sample_variance() const {
        if (count < 2U) {
            return 0.0;
        }
        // Welford's accumulation is well conditioned but not guaranteed
        // non-negative under floating-point cancellation. Floor it so ordinary
        // rounding noise cannot produce sqrt of a negative and be reported as a
        // price-domain overflow.
        return std::max(
            0.0, sum_squared_deviation / static_cast<double>(count - 1U)
        );
    }

    [[nodiscard]] double standard_error() const {
        if (count == 0U) {
            throw std::logic_error("cannot compute a standard error with no samples");
        }
        return std::sqrt(sample_variance() / static_cast<double>(count));
    }
};

struct OnlineCovariance {
    std::size_t count{0U};
    double mean_left{0.0};
    double mean_right{0.0};
    double co_moment{0.0};
    double right_sum_squared_deviation{0.0};

    void add(const double left, const double right) {
        check_finite(left, "control-variate observation");
        check_finite(right, "control-variate observation");
        ++count;
        const double left_difference = left - mean_left;
        const double right_difference = right - mean_right;
        mean_left += left_difference / static_cast<double>(count);
        mean_right += right_difference / static_cast<double>(count);
        co_moment += left_difference * (right - mean_right);
        right_sum_squared_deviation +=
            right_difference * (right - mean_right);
        check_finite(co_moment, "control-variate co-moment");
        check_finite(
            right_sum_squared_deviation, "control-variate dispersion"
        );
    }

    [[nodiscard]] double optimal_coefficient() const {
        if (count < 2U ||
            right_sum_squared_deviation <=
                std::numeric_limits<double>::epsilon()) {
            return 0.0;
        }
        return require_finite(
            co_moment / right_sum_squared_deviation,
            "control-variate coefficient"
        );
    }
};

[[nodiscard]] double payoff(
    const OptionType option_type,
    const double spot,
    const double strike
) {
    switch (option_type) {
        case OptionType::call:
            return std::max(spot - strike, 0.0);
        case OptionType::put:
            return std::max(strike - spot, 0.0);
    }
    throw std::invalid_argument("option type is invalid");
}

void validate_config(const LsmConfig& config) {
    if (config.exercise_steps < 2U ||
        config.exercise_steps > maximum_lsm_exercise_steps) {
        throw std::invalid_argument(
            "LSM exercise steps must lie in [2, " +
            std::to_string(maximum_lsm_exercise_steps) + "]"
        );
    }
    for (const auto& [name, paths] :
         std::vector<std::pair<std::string, std::size_t>>{
             {"training paths", config.training_paths},
             {"valuation paths", config.valuation_paths},
         }) {
        if (paths < 4U || paths > maximum_lsm_paths || paths % 2U != 0U) {
            throw std::invalid_argument(
                "LSM " + name + " must be even and lie in [4, " +
                std::to_string(maximum_lsm_paths) + "]"
            );
        }
    }
    if (config.polynomial_degree < 1U ||
        config.polynomial_degree > maximum_lsm_polynomial_degree) {
        throw std::invalid_argument(
            "LSM polynomial degree must lie in [1, " +
            std::to_string(maximum_lsm_polynomial_degree) + "]"
        );
    }
    if (config.training_seed == config.valuation_seed) {
        throw std::invalid_argument(
            "LSM training and valuation seeds must be distinct"
        );
    }
    if (config.maximum_training_memory_bytes == 0U ||
        config.maximum_training_memory_bytes >
            maximum_lsm_training_memory_bytes) {
        throw std::invalid_argument(
            "LSM maximum training memory bytes must lie in [1, " +
            std::to_string(maximum_lsm_training_memory_bytes) + "]"
        );
    }
}

[[nodiscard]] std::size_t checked_training_memory_bytes(const LsmConfig& config) {
    // Peak policy-training storage comprises the log-spot matrix, cashflows,
    // stopping indices, maximum-size regression state/response vectors, all QR
    // columns, one working column, and the accumulated small policy models.
    const std::size_t double_columns =
        config.exercise_steps + config.polynomial_degree + 6U;
    if (double_columns >
        std::numeric_limits<std::size_t>::max() / sizeof(double)) {
        throw std::overflow_error("LSM training byte multiplier overflows size_t");
    }
    const std::size_t bytes_per_path =
        double_columns * sizeof(double) + sizeof(std::size_t);
    if (config.training_paths >
        std::numeric_limits<std::size_t>::max() / bytes_per_path) {
        throw std::overflow_error("LSM training working set overflows size_t");
    }
    const std::size_t path_storage = config.training_paths * bytes_per_path;
    const std::size_t model_storage =
        (config.exercise_steps + 1U) * sizeof(RegressionModel) +
        config.exercise_steps * (config.polynomial_degree + 1U) * sizeof(double);
    if (path_storage >
        std::numeric_limits<std::size_t>::max() - model_storage) {
        throw std::overflow_error("LSM training working set overflows size_t");
    }
    return path_storage + model_storage;
}

// Enforce the training-memory limit. The estimate is computed once at the
// public entry point and passed in, so no bulk allocation happens before this
// check and the identical value is reported to callers.
void require_training_memory_within_limit(
    const LsmConfig& config,
    const std::size_t memory_bytes
) {
    if (memory_bytes > config.maximum_training_memory_bytes) {
        throw std::invalid_argument(
            "LSM training working set requires " + std::to_string(memory_bytes) +
            " bytes, exceeding the configured limit of " +
            std::to_string(config.maximum_training_memory_bytes)
        );
    }
}

[[nodiscard]] std::vector<double> simulate_training_log_spots(
    const VanillaOptionInput& input,
    const LsmConfig& config
) {
    const std::size_t columns = config.exercise_steps + 1U;
    std::vector<double> log_spots(config.training_paths * columns);
    const double time_step = require_finite(
        input.maturity / static_cast<double>(config.exercise_steps),
        "time step"
    );
    const double drift = require_finite(
        (input.rate - input.dividend_yield -
         0.5 * input.volatility * input.volatility) *
            time_step,
        "log-price drift"
    );
    const double diffusion = require_finite(
        input.volatility * std::sqrt(time_step), "log-price diffusion"
    );
    const double initial_log_spot =
        require_finite(std::log(input.spot), "initial log spot");
    BoxMullerNormal normal(config.training_seed);

    for (std::size_t first_path = 0U; first_path < config.training_paths;
         first_path += 2U) {
        double positive = initial_log_spot;
        double negative = initial_log_spot;
        log_spots[first_path * columns] = positive;
        log_spots[(first_path + 1U) * columns] = negative;
        for (std::size_t step = 1U; step <= config.exercise_steps; ++step) {
            const double shock = normal.next();
            positive += drift + diffusion * shock;
            negative += drift - diffusion * shock;
            require_representable_log_spot(positive, "training path simulation");
            require_representable_log_spot(negative, "training path simulation");
            log_spots[first_path * columns + step] = positive;
            log_spots[(first_path + 1U) * columns + step] = negative;
        }
    }
    return log_spots;
}

[[nodiscard]] double dot(
    const std::vector<double>& left,
    const std::vector<double>& right
) {
    double result = 0.0;
    for (std::size_t index = 0U; index < left.size(); ++index) {
        result += left[index] * right[index];
    }
    return result;
}

[[nodiscard]] RegressionModel constant_model(
    const std::size_t step,
    const std::size_t observations,
    const double mean,
    const double scale,
    const std::vector<double>& response
) {
    double response_mean = 0.0;
    for (const double value : response) {
        response_mean += value;
    }
    if (!response.empty()) {
        response_mean /= static_cast<double>(response.size());
    }
    check_finite(response_mean, "constant continuation fallback");
    return {
        step,
        observations,
        observations == 0U ? 0U : 1U,
        true,
        mean,
        scale,
        0.0,
        {response_mean},
    };
}

[[nodiscard]] RegressionModel fit_continuation(
    const std::size_t step,
    const std::size_t degree,
    const std::vector<double>& state,
    const std::vector<double>& response
) {
    if (state.size() != response.size()) {
        throw std::logic_error("LSM regression state and response sizes differ");
    }
    if (state.empty()) {
        return constant_model(step, 0U, 0.0, 1.0, response);
    }

    double mean = 0.0;
    for (const double value : state) {
        mean += value;
    }
    mean /= static_cast<double>(state.size());
    double squared_scale = 0.0;
    for (const double value : state) {
        const double centered = value - mean;
        squared_scale += centered * centered;
    }
    const double scale =
        std::sqrt(squared_scale / static_cast<double>(state.size()));
    const std::size_t columns = degree + 1U;
    if (!std::isfinite(scale) ||
        scale <= std::numeric_limits<double>::epsilon() ||
        state.size() < columns) {
        return constant_model(step, state.size(), mean, 1.0, response);
    }

    std::vector<std::vector<double>> q(
        columns, std::vector<double>(state.size(), 0.0)
    );
    std::vector<double> upper(columns * columns, 0.0);
    std::vector<double> projected_response(columns, 0.0);
    double first_diagonal = 0.0;
    double minimum_relative_diagonal = 1.0;

    for (std::size_t column = 0U; column < columns; ++column) {
        std::vector<double> values(state.size(), 1.0);
        if (column > 0U) {
            for (std::size_t row = 0U; row < state.size(); ++row) {
                values[row] =
                    std::pow((state[row] - mean) / scale, static_cast<int>(column));
            }
        }

        // Modified Gram--Schmidt with one reorthogonalization pass. The basis
        // is deliberately capped at four columns.
        for (std::size_t pass = 0U; pass < 2U; ++pass) {
            for (std::size_t previous = 0U; previous < column; ++previous) {
                const double projection = dot(q[previous], values);
                upper[previous * columns + column] += projection;
                for (std::size_t row = 0U; row < state.size(); ++row) {
                    values[row] -= projection * q[previous][row];
                }
            }
        }

        const double diagonal = std::sqrt(dot(values, values));
        if (column == 0U) {
            first_diagonal = diagonal;
        }
        const double relative_diagonal =
            first_diagonal > 0.0 ? diagonal / first_diagonal : 0.0;
        minimum_relative_diagonal =
            std::min(minimum_relative_diagonal, relative_diagonal);
        if (!std::isfinite(diagonal) ||
            relative_diagonal <= regression_rank_relative_tolerance) {
            return constant_model(step, state.size(), mean, scale, response);
        }
        upper[column * columns + column] = diagonal;
        for (std::size_t row = 0U; row < state.size(); ++row) {
            q[column][row] = values[row] / diagonal;
        }
        projected_response[column] = dot(q[column], response);
    }

    std::vector<double> coefficients(columns, 0.0);
    for (std::size_t reverse = columns; reverse-- > 0U;) {
        double residual = projected_response[reverse];
        for (std::size_t column = reverse + 1U; column < columns; ++column) {
            residual -= upper[reverse * columns + column] * coefficients[column];
        }
        coefficients[reverse] =
            residual / upper[reverse * columns + reverse];
    }
    if (std::any_of(
            coefficients.begin(),
            coefficients.end(),
            [](const double value) { return !std::isfinite(value); }
        )) {
        return constant_model(step, state.size(), mean, scale, response);
    }
    return {
        step,
        state.size(),
        columns,
        false,
        mean,
        scale,
        minimum_relative_diagonal,
        std::move(coefficients),
    };
}

[[nodiscard]] double continuation_value(
    const RegressionModel& model,
    const double log_moneyness
) {
    const double normalized = require_finite(
        (log_moneyness - model.mean) / model.scale, "regression coordinate"
    );
    double value = 0.0;
    for (auto iterator = model.coefficients.rbegin();
         iterator != model.coefficients.rend();
         ++iterator) {
        value = value * normalized + *iterator;
    }
    return std::max(require_finite(value, "fitted continuation value"), 0.0);
}

[[nodiscard]] Policy train_policy(
    const OptionType option_type,
    const VanillaOptionInput& input,
    const LsmConfig& config,
    const std::vector<double>& log_spots
) {
    const bool suppress_early_exercise =
        (option_type == OptionType::call && input.rate >= 0.0 &&
         input.dividend_yield <= 0.0) ||
        (option_type == OptionType::put && input.rate <= 0.0 &&
         input.dividend_yield >= 0.0);
    const std::size_t columns = config.exercise_steps + 1U;
    const double log_strike =
        require_finite(std::log(input.strike), "log strike");
    const double time_step = require_finite(
        input.maturity / static_cast<double>(config.exercise_steps),
        "time step"
    );
    std::vector<double> cashflow(config.training_paths, 0.0);
    std::vector<std::size_t> stopping_step(
        config.training_paths, config.exercise_steps
    );
    for (std::size_t path = 0U; path < config.training_paths; ++path) {
        cashflow[path] = require_finite(
            payoff(
                option_type,
                spot_from_log(log_spots[path * columns + config.exercise_steps]),
                input.strike
            ),
            "training terminal intrinsic value"
        );
    }

    std::vector<RegressionModel> models(config.exercise_steps + 1U);
    for (std::size_t step = config.exercise_steps; step-- > 1U;) {
        std::vector<double> state;
        std::vector<double> response;
        state.reserve(config.training_paths);
        response.reserve(config.training_paths);
        for (std::size_t path = 0U; path < config.training_paths; ++path) {
            const double log_spot = log_spots[path * columns + step];
            const double intrinsic =
                payoff(option_type, spot_from_log(log_spot), input.strike);
            if (intrinsic > 0.0) {
                state.push_back(
                    require_finite(log_spot - log_strike, "log moneyness")
                );
                // Discount this path's current future stopping cash flow back
                // to the regression date, not merely one interval.
                const double discount = checked_exp(
                    -input.rate * time_step *
                        static_cast<double>(stopping_step[path] - step),
                    "discount factor"
                );
                response.push_back(require_finite(
                    cashflow[path] * discount, "discounted regression response"
                ));
            }
        }
        models[step] = fit_continuation(
            step, config.polynomial_degree, state, response
        );

        for (std::size_t path = 0U; path < config.training_paths; ++path) {
            const double log_spot = log_spots[path * columns + step];
            const double intrinsic =
                payoff(option_type, spot_from_log(log_spot), input.strike);
            if (!suppress_early_exercise && intrinsic > 0.0 &&
                intrinsic >
                    continuation_value(
                        models[step], log_spot - log_strike
                    )) {
                cashflow[path] = intrinsic;
                stopping_step[path] = step;
            }
        }
    }

    double continuation_at_zero = 0.0;
    for (std::size_t path = 0U; path < config.training_paths; ++path) {
        continuation_at_zero += require_finite(
            cashflow[path] *
                checked_exp(
                    -input.rate * time_step *
                        static_cast<double>(stopping_step[path]),
                    "discount factor"
                ),
            "training discounted cash flow"
        );
    }
    continuation_at_zero = require_finite(
        continuation_at_zero / static_cast<double>(config.training_paths),
        "training continuation value at zero"
    );
    const double intrinsic_at_zero =
        payoff(option_type, input.spot, input.strike);
    const bool exercise_at_zero =
        !suppress_early_exercise && intrinsic_at_zero > continuation_at_zero;
    OnlineCovariance control_fit;
    if (!exercise_at_zero) {
        const double terminal_discount = checked_exp(
            -input.rate * input.maturity, "terminal discount factor"
        );
        for (std::size_t first_path = 0U;
             first_path < config.training_paths;
             first_path += 2U) {
            double american_pair = 0.0;
            double european_pair = 0.0;
            for (std::size_t offset = 0U; offset < 2U; ++offset) {
                const std::size_t path = first_path + offset;
                american_pair +=
                    0.5 * cashflow[path] *
                    checked_exp(
                        -input.rate * time_step *
                            static_cast<double>(stopping_step[path]),
                        "discount factor"
                    );
                european_pair +=
                    0.5 *
                    payoff(
                        option_type,
                        spot_from_log(
                            log_spots[path * columns + config.exercise_steps]
                        ),
                        input.strike
                    ) *
                    terminal_discount;
            }
            control_fit.add(american_pair, european_pair);
        }
    }
    return {
        exercise_at_zero,
        suppress_early_exercise,
        continuation_at_zero,
        control_fit.optimal_coefficient(),
        std::move(models),
    };
}

struct DiscountedPathPayoffs {
    double american;
    double european;
    bool exercised_early;
};

[[nodiscard]] DiscountedPathPayoffs finish_path(
    const OptionType option_type,
    const VanillaOptionInput& input,
    const LsmConfig& config,
    const Policy& policy,
    const std::vector<double>& shocks,
    const double shock_sign
) {
    const double time_step = require_finite(
        input.maturity / static_cast<double>(config.exercise_steps),
        "time step"
    );
    const double drift = require_finite(
        (input.rate - input.dividend_yield -
         0.5 * input.volatility * input.volatility) *
            time_step,
        "log-price drift"
    );
    const double diffusion = require_finite(
        input.volatility * std::sqrt(time_step), "log-price diffusion"
    );
    const double log_strike =
        require_finite(std::log(input.strike), "log strike");
    double log_spot = require_finite(std::log(input.spot), "initial log spot");
    double discounted_american = 0.0;
    bool exercised = false;
    bool exercised_early = false;

    for (std::size_t step = 1U; step <= config.exercise_steps; ++step) {
        log_spot += drift + shock_sign * diffusion * shocks[step - 1U];
        require_representable_log_spot(log_spot, "valuation path simulation");
        if (!policy.suppress_early_exercise && !exercised &&
            step < config.exercise_steps) {
            const double intrinsic =
                payoff(option_type, spot_from_log(log_spot), input.strike);
            if (intrinsic > 0.0 &&
                intrinsic >
                    continuation_value(
                        policy.by_step[step],
                        require_finite(log_spot - log_strike, "log moneyness")
                    )) {
                discounted_american = require_finite(
                    intrinsic *
                        checked_exp(
                            -input.rate * time_step *
                                static_cast<double>(step),
                            "discount factor"
                        ),
                    "valuation discounted exercise cash flow"
                );
                exercised = true;
                exercised_early = true;
            }
        }
    }

    const double terminal_spot = spot_from_log(log_spot);
    const double terminal_discount = checked_exp(
        -input.rate * input.maturity, "terminal discount factor"
    );
    const double discounted_european = require_finite(
        payoff(option_type, terminal_spot, input.strike) * terminal_discount,
        "valuation discounted European cash flow"
    );
    if (!exercised) {
        discounted_american = discounted_european;
    }
    return {
        discounted_american,
        discounted_european,
        exercised_early,
    };
}

[[nodiscard]] LsmResult evaluate_policy(
    const OptionType option_type,
    const VanillaOptionInput& input,
    const LsmConfig& config,
    const Policy& policy,
    const std::size_t memory_bytes
) {
    const double analytic_european = require_finite(
        black_scholes(option_type, input).price, "analytic European price"
    );
    if (policy.exercise_at_zero) {
        // Immediate exercise is deterministic: no valuation paths are drawn, so
        // there is no European Monte Carlo observation and no sampling variance
        // on either the raw or the adjusted estimator. The variance-reduction
        // ratio would be the undefined form 0/0, so it is marked inapplicable
        // rather than given a placeholder value that downstream minima could
        // mistake for a measurement.
        const double intrinsic = payoff(option_type, input.spot, input.strike);
        return {
            intrinsic,
            0.0,
            intrinsic,
            intrinsic,
            intrinsic,
            0.0,
            false,
            0.0,
            analytic_european,
            0.0,
            0.0,
            false,
            0.0,
            policy.continuation_at_zero,
            true,
            config.exercise_steps,
            config.training_paths,
            config.valuation_paths,
            config.valuation_paths / 2U,
            memory_bytes,
            config.valuation_paths,
            {},
        };
    }

    BoxMullerNormal normal(config.valuation_seed);
    std::vector<double> shocks(config.exercise_steps);
    OnlineMoments raw;
    OnlineMoments european;
    OnlineMoments adjusted;
    std::size_t early_exercise_paths = 0U;
    for (std::size_t pair = 0U; pair < config.valuation_paths / 2U; ++pair) {
        for (double& shock : shocks) {
            shock = normal.next();
        }
        const DiscountedPathPayoffs positive =
            finish_path(option_type, input, config, policy, shocks, 1.0);
        const DiscountedPathPayoffs negative =
            finish_path(option_type, input, config, policy, shocks, -1.0);
        const double raw_pair =
            0.5 * (positive.american + negative.american);
        const double european_pair =
            0.5 * (positive.european + negative.european);
        const double adjusted_pair = require_finite(
            raw_pair -
                policy.control_variate_coefficient *
                    (european_pair - analytic_european),
            "control-variate adjusted pair average"
        );
        raw.add(raw_pair);
        european.add(european_pair);
        adjusted.add(adjusted_pair);
        early_exercise_paths +=
            static_cast<std::size_t>(positive.exercised_early) +
            static_cast<std::size_t>(negative.exercised_early);
    }

    const double adjusted_standard_error = require_finite(
        adjusted.standard_error(), "adjusted standard error"
    );
    const double raw_variance = raw.sample_variance();
    const double adjusted_variance = adjusted.sample_variance();
    // The ratio measures something only when there was variance to reduce. A
    // raw estimator that is already deterministic across every valuation pair
    // -- for instance a contract so far out of the money that every path pays
    // exactly zero -- makes both variances zero and the ratio the undefined
    // form 0/0, exactly as immediate exercise does. Reporting infinity there
    // would claim the control variate removed variance that never existed.
    const bool variance_reduction_applicable = raw_variance > 0.0;
    const double variance_reduction =
        !variance_reduction_applicable
            ? 0.0
            // A control variate that removes all remaining variance reports a
            // positive infinity; that is a measurement, not an undefined form.
            : adjusted_variance > 0.0
                ? raw_variance / adjusted_variance
                : std::numeric_limits<double>::infinity();
    return {
        require_finite(adjusted.mean, "adjusted price"),
        adjusted_standard_error,
        require_finite(
            adjusted.mean -
                lsm_normal_confidence_quantile * adjusted_standard_error,
            "confidence interval bound"
        ),
        require_finite(
            adjusted.mean +
                lsm_normal_confidence_quantile * adjusted_standard_error,
            "confidence interval bound"
        ),
        require_finite(raw.mean, "raw price"),
        require_finite(raw.standard_error(), "raw standard error"),
        true,
        require_finite(european.mean, "European Monte Carlo price"),
        analytic_european,
        require_finite(european.standard_error(), "European standard error"),
        policy.control_variate_coefficient,
        variance_reduction_applicable,
        variance_reduction,
        policy.continuation_at_zero,
        false,
        config.exercise_steps,
        config.training_paths,
        config.valuation_paths,
        adjusted.count,
        memory_bytes,
        early_exercise_paths,
        {},
    };
}

}  // namespace

std::size_t lsm_training_memory_bytes(const LsmConfig& config) {
    validate_config(config);
    return checked_training_memory_bytes(config);
}

LsmResult least_squares_monte_carlo(
    const OptionType option_type,
    const VanillaOptionInput& input,
    const LsmConfig& config
) {
    input.validate();
    static_cast<void>(payoff(option_type, input.spot, input.strike));
    validate_config(config);
    // Estimated once, enforced before any bulk allocation, and reported.
    const std::size_t memory_bytes = checked_training_memory_bytes(config);
    require_training_memory_within_limit(config, memory_bytes);
    Policy policy;
    {
        const std::vector<double> log_spots =
            simulate_training_log_spots(input, config);
        policy = train_policy(option_type, input, config, log_spots);
    }
    LsmResult result =
        evaluate_policy(option_type, input, config, policy, memory_bytes);
    result.regressions.reserve(config.exercise_steps - 1U);
    for (std::size_t step = 1U; step < config.exercise_steps; ++step) {
        const RegressionModel& model = policy.by_step[step];
        result.regressions.push_back({
            model.step,
            model.observations,
            model.rank,
            model.fallback,
            model.mean,
            model.scale,
            model.minimum_relative_r_diagonal,
            model.coefficients,
        });
    }
    return result;
}

}  // namespace dp
