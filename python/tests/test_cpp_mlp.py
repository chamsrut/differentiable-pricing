import math

import pytest

from differentiable_pricing import DenseLayer, SmoothMlp


def test_cpp_mlp_returns_input_gradient() -> None:
    model = SmoothMlp(
        [
            DenseLayer(
                input_size=2,
                output_size=2,
                weights=[0.4, -0.2, 0.1, 0.3],
                biases=[0.05, -0.1],
            ),
            DenseLayer(
                input_size=2,
                output_size=1,
                weights=[0.7, -0.5],
                biases=[0.2],
            ),
        ]
    )
    value, gradient = model.forward_with_input_gradient([0.25, -0.4])
    assert math.isfinite(value)
    assert len(gradient) == 2

    bump = 1e-6
    for feature in range(2):
        up = [0.25, -0.4]
        down = up.copy()
        up[feature] += bump
        down[feature] -= bump
        up_value, _ = model.forward_with_input_gradient(up)
        down_value, _ = model.forward_with_input_gradient(down)
        finite_difference = (up_value - down_value) / (2.0 * bump)
        assert gradient[feature] == pytest.approx(finite_difference, abs=1e-9)
