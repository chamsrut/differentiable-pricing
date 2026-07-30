#include "dp/black_scholes.hpp"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace dp {
namespace {

[[nodiscard]] double normal_cdf(const double value) {
    return 0.5 * std::erfc(-value / std::numbers::sqrt2);
}

[[nodiscard]] double normal_pdf(const double value) {
    const double inverse_sqrt_two_pi =
        1.0 / std::sqrt(2.0 * std::numbers::pi);
    return inverse_sqrt_two_pi * std::exp(-0.5 * value * value);
}

}  // namespace

BlackScholesResult black_scholes(
    const OptionType option_type,
    const BlackScholesInput& input
) {
    input.validate();

    const double sqrt_maturity = std::sqrt(input.maturity);
    const double volatility_time = input.volatility * sqrt_maturity;
    const double d1 =
        (std::log(input.spot / input.strike) +
         (input.rate - input.dividend_yield +
          0.5 * input.volatility * input.volatility) *
             input.maturity) /
        volatility_time;
    const double d2 = d1 - volatility_time;
    const double spot_discount = std::exp(-input.dividend_yield * input.maturity);
    const double strike_discount = std::exp(-input.rate * input.maturity);
    const double density = normal_pdf(d1);

    const double gamma =
        spot_discount * density /
        (input.spot * input.volatility * sqrt_maturity);
    const double vega =
        input.spot * spot_discount * density * sqrt_maturity;
    const double theta_common =
        -input.spot * spot_discount * density * input.volatility /
        (2.0 * sqrt_maturity);

    if (option_type == OptionType::call) {
        const double price =
            input.spot * spot_discount * normal_cdf(d1) -
            input.strike * strike_discount * normal_cdf(d2);
        const double delta = spot_discount * normal_cdf(d1);
        const double theta =
            theta_common -
            input.rate * input.strike * strike_discount * normal_cdf(d2) +
            input.dividend_yield * input.spot * spot_discount * normal_cdf(d1);
        const double rho =
            input.strike * input.maturity * strike_discount * normal_cdf(d2);
        return {price, delta, gamma, vega, theta, rho};
    }

    const double price =
        input.strike * strike_discount * normal_cdf(-d2) -
        input.spot * spot_discount * normal_cdf(-d1);
    const double delta = spot_discount * (normal_cdf(d1) - 1.0);
    const double theta =
        theta_common +
        input.rate * input.strike * strike_discount * normal_cdf(-d2) -
        input.dividend_yield * input.spot * spot_discount * normal_cdf(-d1);
    const double rho =
        -input.strike * input.maturity * strike_discount * normal_cdf(-d2);
    return {price, delta, gamma, vega, theta, rho};
}

}  // namespace dp
