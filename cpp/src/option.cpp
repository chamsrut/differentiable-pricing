#include "dp/option.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace dp {

void VanillaOptionInput::validate() const {
    if (!std::isfinite(spot) || spot <= 0.0) {
        throw std::invalid_argument("spot must be finite and positive");
    }
    if (!std::isfinite(strike) || strike <= 0.0) {
        throw std::invalid_argument("strike must be finite and positive");
    }
    if (!std::isfinite(maturity) || maturity <= 0.0) {
        throw std::invalid_argument("maturity must be finite and positive");
    }
    if (!std::isfinite(rate)) {
        throw std::invalid_argument("rate must be finite");
    }
    if (!std::isfinite(dividend_yield)) {
        throw std::invalid_argument("dividend yield must be finite");
    }
    if (!std::isfinite(volatility) || volatility <= 0.0) {
        throw std::invalid_argument("volatility must be finite and positive");
    }
}

OptionType parse_option_type(const std::string_view value) {
    if (value == "call") {
        return OptionType::call;
    }
    if (value == "put") {
        return OptionType::put;
    }
    throw std::invalid_argument("option type must be either 'call' or 'put', got '" +
                                std::string(value) + "'");
}

ExerciseStyle parse_exercise_style(const std::string_view value) {
    if (value == "european") {
        return ExerciseStyle::european;
    }
    if (value == "american") {
        return ExerciseStyle::american;
    }
    throw std::invalid_argument("exercise style must be either 'european' or 'american', got '" +
                                std::string(value) + "'");
}

}  // namespace dp
