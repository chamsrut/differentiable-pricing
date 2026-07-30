"""Physical pricing inputs are rejected at the boundary, never silently NaN.

The contract is checked on both representations and on both the unconstrained
and bounded (projected) models, because a projection can mask a NaN by clamping
it to a bound.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from differentiable_pricing.ml.config import (
    FEATURE_ORDER,
    FORWARD_NORMALIZED_REPRESENTATION,
    RAW_PHYSICAL_REPRESENTATION,
)
from differentiable_pricing.ml.model import (
    EUROPEAN_BOUNDS_CONSTRAINT,
    NO_OUTPUT_CONSTRAINT,
    PhysicalInputError,
    PhysicalPriceModel,
    PricingMlp,
    Scaling,
    representation_arrays,
    validate_physical_features,
)

INDEX = {name: FEATURE_ORDER.index(name) for name in FEATURE_ORDER}
# option_type, spot, strike, maturity, rate, dividend_yield, volatility
VALID_ROW = [1.0, 100.0, 100.0, 1.0, 0.03, 0.01, 0.2]

REPRESENTATIONS = (RAW_PHYSICAL_REPRESENTATION, FORWARD_NORMALIZED_REPRESENTATION)
CONSTRAINTS = (NO_OUTPUT_CONSTRAINT, EUROPEAN_BOUNDS_CONSTRAINT)


def build_model(
    representation: str = FORWARD_NORMALIZED_REPRESENTATION,
    constraint: str = NO_OUTPUT_CONSTRAINT,
) -> PhysicalPriceModel:
    dimension = 7 if representation == RAW_PHYSICAL_REPRESENTATION else 3
    scaling = Scaling(
        feature_mean=np.zeros(dimension, dtype=np.float64),
        feature_scale=np.ones(dimension, dtype=np.float64),
        price_mean=0.0,
        price_scale=1.0,
    )
    torch.manual_seed(11)
    return PhysicalPriceModel(
        PricingMlp(dimension, (4,)),
        scaling,
        representation=representation,
        output_constraint=constraint,
    ).eval()


def row(**overrides: float) -> torch.Tensor:
    values = list(VALID_ROW)
    for name, value in overrides.items():
        values[INDEX[name]] = value
    return torch.tensor([values], dtype=torch.float64)


@pytest.mark.parametrize("representation", REPRESENTATIONS)
@pytest.mark.parametrize("constraint", CONSTRAINTS)
def test_valid_row_prices_finitely(representation: str, constraint: str) -> None:
    model = build_model(representation, constraint)
    price = model(row())
    assert price.shape == (1,)
    assert bool(torch.isfinite(price).all())


@pytest.mark.parametrize("field", ["spot", "strike", "maturity", "volatility"])
@pytest.mark.parametrize("bad", [0.0, -1.0])
@pytest.mark.parametrize("representation", REPRESENTATIONS)
@pytest.mark.parametrize("constraint", CONSTRAINTS)
def test_non_positive_inputs_are_rejected(
    field: str,
    bad: float,
    representation: str,
    constraint: str,
) -> None:
    model = build_model(representation, constraint)
    with pytest.raises(PhysicalInputError, match=f"'{field}' must be strictly positive"):
        model(row(**{field: bad}))


@pytest.mark.parametrize("field", list(FEATURE_ORDER))
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_non_finite_inputs_are_rejected(
    field: str,
    bad: float,
    representation: str,
) -> None:
    model = build_model(representation)
    with pytest.raises(PhysicalInputError, match=f"'{field}' contains NaN or infinite"):
        model(row(**{field: bad}))


@pytest.mark.parametrize("bad", [0.0, 0.5, 2.0, -2.0])
def test_invalid_option_type_encoding_is_rejected(bad: float) -> None:
    model = build_model()
    with pytest.raises(PhysicalInputError, match="'option_type' must be encoded"):
        model(row(option_type=bad))


@pytest.mark.parametrize("option_type", [1.0, -1.0])
def test_both_option_types_are_accepted(option_type: float) -> None:
    model = build_model()
    assert bool(torch.isfinite(model(row(option_type=option_type))).all())


def test_wrong_column_count_is_rejected() -> None:
    model = build_model()
    with pytest.raises(PhysicalInputError, match="must have 7 columns"):
        model(torch.tensor([[1.0, 100.0, 100.0]], dtype=torch.float64))


def test_wrong_rank_is_rejected() -> None:
    model = build_model()
    with pytest.raises(PhysicalInputError, match="must be a 2-D"):
        model(torch.tensor(VALID_ROW, dtype=torch.float64))


def test_empty_batch_is_rejected() -> None:
    model = build_model()
    with pytest.raises(PhysicalInputError, match="at least one row"):
        model(torch.empty((0, len(FEATURE_ORDER)), dtype=torch.float64))


def test_bad_row_is_caught_even_when_other_rows_are_valid() -> None:
    """A single bad row in a batch must fail, not be averaged away."""
    model = build_model()
    batch = torch.tensor([VALID_ROW, VALID_ROW, VALID_ROW], dtype=torch.float64)
    batch[1, INDEX["strike"]] = 0.0
    with pytest.raises(PhysicalInputError, match="'strike' must be strictly positive"):
        model(batch)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("maturity", 1.0e-8),
        ("volatility", 1.0e-8),
        ("spot", 1.0e-6),
        ("strike", 1.0e-6),
        ("rate", -0.05),
        ("dividend_yield", 0.0),
    ],
)
def test_boundary_adjacent_inputs_are_accepted(field: str, value: float) -> None:
    """Tiny-but-positive, negative rates, and zero dividends stay in domain."""
    model = build_model()
    price = model(row(**{field: value}))
    assert bool(torch.isfinite(price).all())


def test_numpy_representation_path_shares_the_contract() -> None:
    features = np.array([VALID_ROW], dtype=np.float64)
    prices = np.array([10.0], dtype=np.float64)
    # Valid input works on both representations.
    for representation in REPRESENTATIONS:
        network_features, targets = representation_arrays(
            features, prices, representation
        )
        assert bool(np.isfinite(network_features).all())
        assert bool(np.isfinite(targets).all())

    bad = features.copy()
    bad[0, INDEX["strike"]] = -1.0
    with pytest.raises(PhysicalInputError, match="'strike' must be strictly positive"):
        representation_arrays(bad, prices, FORWARD_NORMALIZED_REPRESENTATION)
    # The raw path validates identically, so the two cannot drift apart.
    with pytest.raises(PhysicalInputError, match="'strike' must be strictly positive"):
        representation_arrays(bad, prices, RAW_PHYSICAL_REPRESENTATION)


def test_validator_accepts_numpy_and_torch_alike() -> None:
    values = [VALID_ROW]
    validate_physical_features(np.array(values, dtype=np.float64))
    validate_physical_features(torch.tensor(values, dtype=torch.float64))

    bad = [list(VALID_ROW)]
    bad[0][INDEX["spot"]] = float("nan")
    with pytest.raises(PhysicalInputError, match="'spot' contains NaN"):
        validate_physical_features(np.array(bad, dtype=np.float64))
    with pytest.raises(PhysicalInputError, match="'spot' contains NaN"):
        validate_physical_features(torch.tensor(bad, dtype=torch.float64))


def test_validation_precedes_the_projection() -> None:
    """A bounded model must not clamp a NaN into a plausible-looking price."""
    model = build_model(constraint=EUROPEAN_BOUNDS_CONSTRAINT)
    with pytest.raises(PhysicalInputError):
        model(row(spot=float("nan")))
