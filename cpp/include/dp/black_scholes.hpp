#pragma once

#include <string_view>

namespace dp {

enum class OptionType {
    call,
    put,
};

[[nodiscard]] OptionType parse_option_type(std::string_view value);

struct BlackScholesInput {
    double spot;
    double strike;
    double maturity;
    double rate;
    double dividend_yield;
    double volatility;

    void validate() const;
};

struct BlackScholesResult {
    double price;
    double delta;
    double gamma;
    double vega;
    double theta;
    double rho;
};

[[nodiscard]] BlackScholesResult black_scholes(
    OptionType option_type,
    const BlackScholesInput& input
);

}  // namespace dp

