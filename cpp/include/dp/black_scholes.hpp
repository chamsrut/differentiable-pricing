#pragma once

#include "dp/option.hpp"

namespace dp {

// Backward-compatible name for the shared constant-parameter vanilla input.
using BlackScholesInput = VanillaOptionInput;

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
