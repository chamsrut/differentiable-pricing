#pragma once

#include <string_view>

namespace dp {

enum class OptionType {
    call,
    put,
};

enum class ExerciseStyle {
    european,
    american,
};

struct VanillaOptionInput {
    double spot;
    double strike;
    double maturity;
    double rate;
    double dividend_yield;
    double volatility;

    void validate() const;
};

[[nodiscard]] OptionType parse_option_type(std::string_view value);
[[nodiscard]] ExerciseStyle parse_exercise_style(std::string_view value);

}  // namespace dp
