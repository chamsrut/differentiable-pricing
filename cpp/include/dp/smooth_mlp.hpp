#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace dp {

struct DenseLayer {
    std::size_t input_size;
    std::size_t output_size;
    // Row-major: weights[output_index * input_size + input_index].
    std::vector<double> weights;
    std::vector<double> biases;

    void validate() const;
};

struct SurrogateResult {
    double value;
    std::vector<double> input_gradient;
};

// A scalar-output MLP with tanh hidden activations and a linear output layer.
// It deliberately implements its own reverse pass so C++ inference can return
// exact derivatives of the surrogate without finite differences.
class SmoothMlp {
public:
    explicit SmoothMlp(std::vector<DenseLayer> layers);

    [[nodiscard]] std::size_t input_size() const noexcept;
    [[nodiscard]] SurrogateResult forward_with_input_gradient(
        std::span<const double> input
    ) const;

private:
    std::vector<DenseLayer> layers_;
};

}  // namespace dp

