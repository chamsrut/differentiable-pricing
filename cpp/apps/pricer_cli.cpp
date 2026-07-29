#include "dp/black_scholes.hpp"

#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <string_view>

namespace {

void usage(const std::string_view program) {
    std::cerr
        << "Usage: " << program
        << " <call|put> <spot> <strike> <maturity> <rate> "
           "<dividend-yield> <volatility>\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 8) {
        usage(argv[0]);
        return 2;
    }

    try {
        const dp::OptionType option_type = dp::parse_option_type(argv[1]);
        const dp::BlackScholesInput input{
            std::stod(argv[2]),
            std::stod(argv[3]),
            std::stod(argv[4]),
            std::stod(argv[5]),
            std::stod(argv[6]),
            std::stod(argv[7]),
        };
        const dp::BlackScholesResult result =
            dp::black_scholes(option_type, input);

        std::cout << std::setprecision(15)
                  << "{\n"
                  << "  \"price\": " << result.price << ",\n"
                  << "  \"delta\": " << result.delta << ",\n"
                  << "  \"gamma\": " << result.gamma << ",\n"
                  << "  \"vega\": " << result.vega << ",\n"
                  << "  \"theta\": " << result.theta << ",\n"
                  << "  \"rho\": " << result.rho << "\n"
                  << "}\n";
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
