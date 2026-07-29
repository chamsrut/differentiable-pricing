#include "dp/smooth_mlp.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace dp {

void DenseLayer::validate() const {
    if (input_size == 0U || output_size == 0U) {
        throw std::invalid_argument("dense-layer dimensions must be positive");
    }
    if (weights.size() != input_size * output_size) {
        throw std::invalid_argument("dense-layer weight count does not match dimensions");
    }
    if (biases.size() != output_size) {
        throw std::invalid_argument("dense-layer bias count does not match dimensions");
    }
    for (const double value : weights) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("dense-layer weights must be finite");
        }
    }
    for (const double value : biases) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("dense-layer biases must be finite");
        }
    }
}

SmoothMlp::SmoothMlp(std::vector<DenseLayer> layers)
    : layers_(std::move(layers)) {
    if (layers_.empty()) {
        throw std::invalid_argument("MLP must contain at least one layer");
    }
    for (std::size_t index = 0; index < layers_.size(); ++index) {
        layers_[index].validate();
        if (index > 0U &&
            layers_[index - 1U].output_size != layers_[index].input_size) {
            throw std::invalid_argument("adjacent MLP layer dimensions do not match");
        }
    }
    if (layers_.back().output_size != 1U) {
        throw std::invalid_argument("MLP output layer must have size one");
    }
}

std::size_t SmoothMlp::input_size() const noexcept {
    return layers_.front().input_size;
}

SurrogateResult SmoothMlp::forward_with_input_gradient(
    const std::span<const double> input
) const {
    if (input.size() != input_size()) {
        throw std::invalid_argument("MLP input size does not match the first layer");
    }
    for (const double value : input) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("MLP inputs must be finite");
        }
    }

    std::vector<std::vector<double>> activations;
    activations.reserve(layers_.size() + 1U);
    activations.emplace_back(input.begin(), input.end());

    for (std::size_t layer_index = 0; layer_index < layers_.size(); ++layer_index) {
        const DenseLayer& layer = layers_[layer_index];
        const std::vector<double>& previous = activations.back();
        std::vector<double> current(layer.output_size, 0.0);
        const bool output_layer = layer_index + 1U == layers_.size();

        for (std::size_t output = 0; output < layer.output_size; ++output) {
            double value = layer.biases[output];
            for (std::size_t feature = 0; feature < layer.input_size; ++feature) {
                value +=
                    layer.weights[output * layer.input_size + feature] *
                    previous[feature];
            }
            current[output] = output_layer ? value : std::tanh(value);
        }
        activations.push_back(std::move(current));
    }

    std::vector<double> gradient{1.0};
    for (std::size_t reverse = layers_.size(); reverse-- > 0U;) {
        const DenseLayer& layer = layers_[reverse];
        const bool output_layer = reverse + 1U == layers_.size();
        std::vector<double> gradient_z(layer.output_size, 0.0);
        for (std::size_t output = 0; output < layer.output_size; ++output) {
            const double activation_derivative =
                output_layer
                    ? 1.0
                    : 1.0 -
                          activations[reverse + 1U][output] *
                              activations[reverse + 1U][output];
            gradient_z[output] = gradient[output] * activation_derivative;
        }

        std::vector<double> previous_gradient(layer.input_size, 0.0);
        for (std::size_t output = 0; output < layer.output_size; ++output) {
            for (std::size_t feature = 0; feature < layer.input_size; ++feature) {
                previous_gradient[feature] +=
                    layer.weights[output * layer.input_size + feature] *
                    gradient_z[output];
            }
        }
        gradient = std::move(previous_gradient);
    }

    return {activations.back().front(), std::move(gradient)};
}

}  // namespace dp

