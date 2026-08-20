"""Small tests for the exact Task 9G representation and transfer lift."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from differentiable_pricing.ml.american import (
    AmericanPriceModel,
    american_representation_arrays,
    fit_price_model,
    lift_european_network,
    scaling_from_arrays,
    select_rows_by_hash,
    verify_exact_lift,
)
from differentiable_pricing.ml.config import FORWARD_NORMALIZED_REPRESENTATION
from differentiable_pricing.ml.model import (
    PhysicalInputError,
    PhysicalPriceModel,
    PricingMlp,
    Scaling,
)


def _features() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 100.0, 95.0, 0.25, -0.01, 0.0, 0.15],
            [-1.0, 120.0, 130.0, 2.5, 0.08, 0.12, 0.55],
        ],
        dtype=np.float64,
    )


def test_american_forward_carry_order_and_target() -> None:
    physical = _features()
    prices = np.asarray([7.0, 18.0], dtype=np.float64)
    features, targets = american_representation_arrays(physical, prices)
    expected = np.column_stack(
        (
            physical[:, 0],
            np.log(physical[:, 1] / physical[:, 2])
            + (physical[:, 4] - physical[:, 5]) * physical[:, 3],
            physical[:, 6] * np.sqrt(physical[:, 3]),
            physical[:, 4] * physical[:, 3],
            physical[:, 5] * physical[:, 3],
        )
    )
    np.testing.assert_allclose(features, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        targets,
        prices / (physical[:, 1] * np.exp(-physical[:, 5] * physical[:, 3])),
        rtol=0.0,
        atol=0.0,
    )


def test_american_reconstruction_stays_inside_autograd_graph() -> None:
    network = PricingMlp(5, ())
    with torch.no_grad():
        network.layers[0].weight.zero_()
        network.layers[0].bias.zero_()
    model = AmericanPriceModel(
        network,
        Scaling(np.zeros(5), np.ones(5), 0.2, 1.0),
    )
    physical = torch.tensor(_features()[:1], dtype=torch.float64, requires_grad=True)
    price = model(physical)
    delta = torch.autograd.grad(price.sum(), physical)[0][0, 1]
    expected_price = 0.2 * 100.0
    assert float(price[0].detach()) == pytest.approx(expected_price)
    assert float(delta) == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        (0, 0.0, "option_type"),
        (1, 0.0, "spot"),
        (2, -1.0, "strike"),
        (3, 0.0, "maturity"),
        (6, float("nan"), "volatility"),
    ],
)
def test_american_representation_rejects_invalid_domain(
    column: int, value: float, message: str
) -> None:
    physical = _features()[:1].copy()
    physical[0, column] = value
    with pytest.raises(PhysicalInputError, match=message):
        american_representation_arrays(physical, np.asarray([1.0]))


def test_numpy_and_torch_paths_reject_transformed_domain_overflow() -> None:
    physical = _features()[:1].copy()
    physical[0, 3] = 1.0e308
    physical[0, 4] = 10.0
    with pytest.raises(PhysicalInputError, match="outside its domain"):
        american_representation_arrays(physical, np.asarray([1.0]))
    model = AmericanPriceModel(PricingMlp(5, ()), Scaling(np.zeros(5), np.ones(5), 0.0, 1.0))
    with pytest.raises(PhysicalInputError, match="outside its domain"):
        model(torch.as_tensor(physical, dtype=torch.float64))


def _source_model() -> PhysicalPriceModel:
    torch.manual_seed(9)
    return PhysicalPriceModel(
        PricingMlp(3, (64, 64, 64)),
        Scaling(
            np.asarray([0.1, -0.2, 0.35]),
            np.asarray([0.8, 0.4, 0.25]),
            0.12,
            0.07,
        ),
        FORWARD_NORMALIZED_REPRESENTATION,
    ).eval()


def test_exact_lift_rebases_scaling_and_zeroes_new_columns() -> None:
    source = _source_model()
    target_scaling = Scaling(
        np.asarray([-0.05, 0.3, 0.4, 0.02, 0.01]),
        np.asarray([1.0, 0.9, 0.5, 0.08, 0.06]),
        0.3,
        0.2,
    )
    lifted = lift_european_network(source, target_scaling)
    probes = np.vstack(
        (
            _features(),
            np.asarray(
                [
                    [1.0, 70.0, 120.0, 0.05, 0.1, 0.0, 0.7],
                    [-1.0, 150.0, 80.0, 3.0, -0.02, 0.1, 0.08],
                ]
            ),
        )
    )
    result = verify_exact_lift(source, lifted, probes)
    assert result["probes"] == 4
    first = lifted.network.layers[0]
    assert isinstance(first, torch.nn.Linear)
    assert torch.equal(first.weight[:, 3:], torch.zeros_like(first.weight[:, 3:]))


def test_exact_lift_refuses_a_projected_source() -> None:
    source = _source_model()
    source.output_constraint = "european_bounds_v1"
    with pytest.raises(ValueError, match="unconstrained"):
        lift_european_network(
            source,
            Scaling(np.zeros(5), np.ones(5), 0.0, 1.0),
        )


def test_hash_row_selection_is_order_independent() -> None:
    sample_ids = np.asarray(["c", "a", "d", "b"], dtype=object)
    chosen = {sample_ids[index] for index in select_rows_by_hash(sample_ids, 2, "salt")}
    reversed_ids = sample_ids[::-1]
    reversed_chosen = {
        reversed_ids[index] for index in select_rows_by_hash(reversed_ids, 2, "salt")
    }
    assert chosen == reversed_chosen


def test_training_history_records_learning_rate_used_by_each_epoch() -> None:
    features = _features()
    prices = np.asarray([7.0, 18.0], dtype=np.float64)
    result = fit_price_model(
        features,
        prices,
        features,
        prices,
        scaling_from_arrays(features, prices),
        seed=11,
        batch_size=2,
        epochs=2,
        learning_rate=0.001,
        weight_decay=0.0,
        beta1=0.9,
        beta2=0.999,
        epsilon=1.0e-8,
        amsgrad=False,
        maximize=False,
        foreach=False,
        capturable=False,
        differentiable=False,
        fused=False,
        minimum_learning_rate=0.0,
        schedule_period_epochs=2,
        schedule_last_epoch=-1,
        num_threads=1,
    )

    assert [record["learning_rate"] for record in result.history] == pytest.approx([0.001, 0.0005])
