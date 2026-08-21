"""Task 9H model dispatch, representation and physical reconstruction.

Small synthetic fixtures only. No dataset partition is opened and no numerical
study runs.

The properties pinned here are the ones an attempt silently depends on: that the
five representation coordinates are the Task 9G ones, that the NumPy path used
to fit scaling is the same construction the model consumes, that
``V = A * u`` really is what ``forward`` returns, and that the premium head's
European anchor holds.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from differentiable_pricing import black_scholes
from differentiable_pricing.ml.american_dev.models import (
    ArchitectureError,
    DenseNetwork,
    ResidualNetwork,
    build_network,
    parameter_count,
)
from differentiable_pricing.ml.american_dev.representation import (
    BASE_FEATURE_ORDER,
    REPRESENTATION,
    AmericanDevPriceModel,
    base_features,
    conditioning_feature,
    discounted_spot,
    european_price,
    feature_order,
    intrinsic_value,
    network_features,
    representation_arrays,
)
from differentiable_pricing.ml.model import PhysicalInputError, Scaling

# option_type, spot, strike, maturity, rate, dividend_yield, volatility
PHYSICAL = np.asarray(
    [
        [1.0, 100.0, 100.0, 1.0, 0.05, 0.02, 0.25],
        [-1.0, 80.0, 100.0, 0.25, 0.03, 0.0, 0.4],
        [1.0, 130.0, 110.0, 2.5, 0.02, 0.08, 0.2],
        [-1.0, 120.0, 100.0, 0.05, 0.07, 0.01, 0.12],
    ],
    dtype=np.float64,
)


def _scaling(dimension: int) -> Scaling:
    return Scaling(
        np.zeros(dimension, dtype=np.float64),
        np.ones(dimension, dtype=np.float64),
        0.0,
        1.0,
    )


def _model(conditioning: tuple[str, ...] = (), head: str = "direct") -> AmericanDevPriceModel:
    torch.manual_seed(20260820)
    dimension = len(feature_order(conditioning))
    network = DenseNetwork(dimension, (16, 16), "tanh")
    return AmericanDevPriceModel(
        network, _scaling(dimension), head=head, conditioning=conditioning
    ).eval()


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------


def test_the_representation_is_the_task_9g_five_coordinates() -> None:
    assert REPRESENTATION == "american_forward_carry_v1"
    assert BASE_FEATURE_ORDER == (
        "option_type",
        "log_forward_moneyness",
        "total_volatility",
        "rate_time",
        "yield_time",
    )
    features = base_features(torch.as_tensor(PHYSICAL, dtype=torch.float64)).numpy()
    for row, physical in zip(features, PHYSICAL, strict=True):
        _, spot, strike, maturity, rate, dividend_yield, volatility = physical
        assert row[0] == physical[0]
        assert row[1] == pytest.approx(
            math.log(spot / strike) + (rate - dividend_yield) * maturity, abs=1e-14
        )
        assert row[2] == pytest.approx(volatility * math.sqrt(maturity), abs=1e-14)
        assert row[3] == pytest.approx(rate * maturity, abs=1e-14)
        assert row[4] == pytest.approx(dividend_yield * maturity, abs=1e-14)


def test_the_european_anchor_matches_the_analytic_reference() -> None:
    computed = european_price(torch.as_tensor(PHYSICAL, dtype=torch.float64)).numpy()
    for value, physical in zip(computed, PHYSICAL, strict=True):
        option_type = "call" if physical[0] > 0.0 else "put"
        expected = black_scholes(option_type, *(float(item) for item in physical[1:]))["price"]
        assert value == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_intrinsic_value_is_the_payoff_at_the_current_spot() -> None:
    computed = intrinsic_value(torch.as_tensor(PHYSICAL, dtype=torch.float64)).numpy()
    for value, physical in zip(computed, PHYSICAL, strict=True):
        omega, spot, strike = physical[0], physical[1], physical[2]
        assert value == pytest.approx(max(omega * (spot - strike), 0.0), abs=0.0)


def test_conditioning_features_are_deterministic_functions_of_the_inputs() -> None:
    """They change conditioning, not information."""
    tensor = torch.as_tensor(PHYSICAL, dtype=torch.float64)
    scale = discounted_spot(tensor).numpy()
    european = european_price(tensor).numpy()
    intrinsic = intrinsic_value(tensor).numpy()
    assert conditioning_feature("european_price_ratio", tensor).numpy() == pytest.approx(
        european / scale
    )
    assert conditioning_feature("intrinsic_ratio", tensor).numpy() == pytest.approx(
        intrinsic / scale
    )
    assert conditioning_feature("european_gap", tensor).numpy() == pytest.approx(
        (european - intrinsic) / scale
    )
    with pytest.raises(ValueError, match="unknown conditioning feature"):
        conditioning_feature("not_a_feature", tensor)


def test_conditioning_widens_the_input_in_the_declared_order() -> None:
    assert feature_order(()) == BASE_FEATURE_ORDER
    assert feature_order(("european_price_ratio", "intrinsic_ratio"))[-2:] == (
        "european_price_ratio",
        "intrinsic_ratio",
    )
    model = _model(("european_price_ratio", "intrinsic_ratio"))
    assert model.feature_mean.shape == (7,)


def test_the_numpy_and_torch_feature_paths_agree_exactly() -> None:
    """The NumPy path calls the torch path, so scaling cannot drift from it."""
    prices = np.asarray([9.0, 21.0, 25.0, 1.5], dtype=np.float64)
    for conditioning in ((), ("european_price_ratio", "intrinsic_ratio")):
        features, targets = representation_arrays(PHYSICAL, prices, conditioning)
        tensor = torch.as_tensor(PHYSICAL, dtype=torch.float64)
        with torch.no_grad():
            expected = network_features(tensor, conditioning).numpy()
            scale = discounted_spot(tensor).numpy()
        assert features.shape == (PHYSICAL.shape[0], len(feature_order(conditioning)))
        assert np.array_equal(features, expected)
        assert np.array_equal(targets, prices / scale)


# ---------------------------------------------------------------------------
# Reconstruction and heads
# ---------------------------------------------------------------------------


def test_physical_reconstruction_is_the_declared_identity() -> None:
    model = _model()
    tensor = torch.as_tensor(PHYSICAL, dtype=torch.float64)
    with torch.no_grad():
        price = model(tensor).numpy()
        normalized = model.normalized_target(tensor).numpy()
        scale = discounted_spot(tensor).numpy()
    assert np.array_equal(price, scale * normalized)


def test_the_premium_head_holds_its_european_anchor() -> None:
    model = _model(head="premium_over_european")
    tensor = torch.as_tensor(PHYSICAL, dtype=torch.float64)
    with torch.no_grad():
        price = model(tensor).numpy()
        anchor = european_price(tensor).numpy()
    assert bool((price >= anchor).all())
    assert bool((price > anchor).any())


def test_the_premium_anchor_survives_an_extreme_network_output() -> None:
    """softplus underflows to zero, collapsing onto the anchor, never below it."""
    model = _model(head="premium_over_european")
    with torch.no_grad():
        for parameter in model.network.parameters():
            parameter.fill_(0.0)
        final = [layer for layer in model.network.layers if isinstance(layer, torch.nn.Linear)][-1]
        final.bias.fill_(-500.0)
    tensor = torch.as_tensor(PHYSICAL, dtype=torch.float64)
    with torch.no_grad():
        price = model(tensor).numpy()
        anchor = european_price(tensor).numpy()
    assert bool((price >= anchor).all())
    assert bool(np.isfinite(price).all())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"head": "clamped"}, "unknown head"),
        ({"conditioning": ("not_a_feature",)}, "unknown conditioning feature"),
        ({"conditioning": ("intrinsic_ratio", "intrinsic_ratio")}, "distinct"),
    ],
)
def test_the_model_rejects_unknown_or_duplicated_declarations(
    kwargs: dict[str, object], match: str
) -> None:
    network = DenseNetwork(5, (4,), "tanh")
    with pytest.raises(ValueError, match=match):
        AmericanDevPriceModel(network, _scaling(5), **kwargs)


def test_the_model_rejects_a_scaling_of_the_wrong_width() -> None:
    network = DenseNetwork(6, (4,), "tanh")
    with pytest.raises(ValueError, match=r"feature coordinates"):
        AmericanDevPriceModel(network, _scaling(5), conditioning=("european_price_ratio",))


def test_domain_violations_raise_a_named_error_rather_than_a_quiet_nan() -> None:
    model = _model()
    bad = PHYSICAL.copy()
    bad[0, 1] = -1.0
    with pytest.raises(PhysicalInputError):
        model(torch.as_tensor(bad, dtype=torch.float64))


def test_float32_inputs_are_refused() -> None:
    model = _model()
    with pytest.raises(PhysicalInputError, match="float64 CPU tensor"):
        model(torch.as_tensor(PHYSICAL, dtype=torch.float32))


# ---------------------------------------------------------------------------
# Architecture dispatch
# ---------------------------------------------------------------------------


def test_dispatch_builds_each_declared_architecture() -> None:
    dense = build_network(
        {"name": "smooth_mlp", "hidden_dimensions": [8, 8], "activation": "tanh"}, 5
    )
    residual = build_network(
        {"name": "smooth_residual", "width": 8, "blocks": 3, "activation": "tanh"}, 5
    )
    assert isinstance(dense, DenseNetwork)
    assert isinstance(residual, ResidualNetwork)
    for network in (dense, residual):
        output = network(torch.zeros((4, 5), dtype=torch.float64))
        assert output.shape == (4,)
        assert output.dtype == torch.float64
        assert parameter_count(network) > 0


def test_residual_blocks_are_identity_shortcuts() -> None:
    """A zeroed block passes its input through unchanged."""
    network = ResidualNetwork(5, 8, 2, "tanh")
    with torch.no_grad():
        for block in network.blocks:
            block.outer.weight.zero_()
            block.outer.bias.zero_()
        hidden = network.projection(torch.ones((2, 5), dtype=torch.float64))
        for block in network.blocks:
            assert torch.equal(block(hidden), hidden)


@pytest.mark.parametrize(
    ("section", "match"),
    [
        ({"name": "transformer", "activation": "tanh"}, "architecture.name"),
        (
            {"name": "smooth_mlp", "hidden_dimensions": [8], "activation": "swish"},
            "architecture.activation",
        ),
        ({"name": "smooth_mlp", "activation": "tanh"}, "hidden_dimensions"),
        ({"name": "smooth_residual", "activation": "tanh"}, "integer width and blocks"),
        (
            {"name": "smooth_mlp", "hidden_dimensions": [0], "activation": "tanh"},
            "positive integers",
        ),
        ({"name": "smooth_mlp", "hidden_dimensions": [99999], "activation": "tanh"}, "exceeds"),
    ],
)
def test_dispatch_rejects_malformed_architecture_sections(
    section: dict[str, object], match: str
) -> None:
    with pytest.raises(ArchitectureError, match=match):
        build_network(section, 5)


def test_an_absurd_declaration_is_refused_before_anything_is_allocated() -> None:
    with pytest.raises(ArchitectureError, match="declares"):
        build_network(
            {"name": "smooth_residual", "width": 4096, "blocks": 64, "activation": "tanh"}, 5
        )
