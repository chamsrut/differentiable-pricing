#include <algorithm>
#include <charconv>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

#include "dp/binomial_tree.hpp"

namespace {

void usage(const std::string_view program) {
    std::cerr << "Usage: " << program
              << " <european|american> <call|put> <spot> <strike> <maturity> "
                 "<rate> <dividend-yield> <volatility> <steps>\n";
}

[[nodiscard]] std::string_view without_optional_plus(std::string_view value) {
    if (!value.empty() && value.front() == '+') {
        value.remove_prefix(1U);
    }
    return value;
}

[[nodiscard]] std::size_t parse_steps(const std::string_view value) {
    const std::string_view parsed_value = without_optional_plus(value);
    std::size_t steps = 0U;
    const char* const begin = parsed_value.data();
    const char* const end = begin + parsed_value.size();
    const auto [parsed_end, error] = std::from_chars(begin, end, steps);
    if (parsed_value.empty() || error != std::errc{} || parsed_end != end) {
        throw std::invalid_argument("steps must be a positive base-10 integer");
    }
    return steps;
}

[[nodiscard]] double parse_number(const std::string_view value, const std::string_view name) {
    const std::string_view parsed_value = without_optional_plus(value);
    double number = 0.0;
    const char* const begin = parsed_value.data();
    const char* const end = begin + parsed_value.size();
    const auto [parsed_end, error] = std::from_chars(begin, end, number);
    if (parsed_value.empty() || error != std::errc{} || parsed_end != end) {
        throw std::invalid_argument(std::string(name) + " must be a base-10 number");
    }
    return number;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 10) {
        usage(argv[0]);
        return 2;
    }

    try {
        const dp::ExerciseStyle exercise_style = dp::parse_exercise_style(argv[1]);
        const dp::OptionType option_type = dp::parse_option_type(argv[2]);
        const dp::VanillaOptionInput input{
            parse_number(argv[3], "spot"),           parse_number(argv[4], "strike"),
            parse_number(argv[5], "maturity"),       parse_number(argv[6], "rate"),
            parse_number(argv[7], "dividend yield"), parse_number(argv[8], "volatility"),
        };
        const std::size_t steps = parse_steps(argv[9]);
        const dp::CrrAdjacentStepEstimate estimate =
            dp::crr_adjacent_step_estimate(option_type, exercise_style, input, steps);
        const dp::CrrResult& result = estimate.at_steps;
        const auto exercise_boundary_points = static_cast<std::size_t>(std::count_if(
            result.exercise_boundary_by_step.begin(), result.exercise_boundary_by_step.end(),
            [](const std::optional<double>& boundary) { return boundary.has_value(); }));

        std::cout << std::setprecision(15) << "{\n"
                  << "  \"price\": " << result.price << ",\n"
                  << "  \"steps\": " << result.steps << ",\n"
                  << "  \"adjacent_step_price\": " << estimate.at_steps_plus_one.price << ",\n"
                  << "  \"adjacent_step_average\": " << estimate.average_price << ",\n"
                  << "  \"adjacent_step_gap\": " << estimate.absolute_pair_gap << ",\n"
                  << "  \"risk_neutral_probability\": " << result.risk_neutral_probability << ",\n"
                  << "  \"early_exercise_nodes\": " << result.early_exercise_nodes << ",\n"
                  << "  \"exercise_boundary_points\": " << exercise_boundary_points << ",\n"
                  << "  \"earliest_exercise_step\": ";
        if (result.earliest_exercise_step.has_value()) {
            std::cout << *result.earliest_exercise_step;
        } else {
            std::cout << "null";
        }
        std::cout << "\n}\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
