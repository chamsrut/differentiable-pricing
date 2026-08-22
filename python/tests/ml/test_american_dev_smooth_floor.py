"""Task 9H E2: the smooth one-sided lower-floor output head.

Synthetic fixtures throughout. No dataset partition is opened, no attempt is
run and no training happens: the head is exercised directly, which is where its
guarantees live.

The properties pinned here are the ones the transformation exists to provide and
the ones it must not quietly break: that the network still predicts the **direct**
normalized price rather than a premium, that the output never falls below either
deployment-computable floor, that it reproduces the direct output away from the
floor, that its first and second derivatives stay finite, that no exponential
overflows or underflows into a non-finite value anywhere on the admitted domain,
and that nothing at inference needs a CRR price.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from differentiable_pricing.ml.american_dev.attempts import (
    HEAD_TEMPERATURES,
    HEADS,
    SMOOTH_FLOOR_TEMPERATURE,
    AttemptError,
    FinalPartitionAccessError,
    assert_temperature_consistent,
    head_temperature,
    load_toml,
    validate_attempt_config,
)
from differentiable_pricing.ml.american_dev.representation import (
    AmericanDevPriceModel,
    discounted_spot,
    european_price,
    intrinsic_value,
    normalized_lower_floor,
    representation_arrays,
    smooth_lower_bound,
    smooth_maximum,
)
from differentiable_pricing.ml.american_dev.workbench import build_model
from differentiable_pricing.ml.model import fit_scaling

PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPRESENTATION_MODULE = (
    PROJECT_ROOT / "python/src/differentiable_pricing/ml/american_dev/representation.py"
)
PARENT_CONFIG = Path("configs/american_dev_attempt_scratch_residual_architecture_v1.toml")
FLOOR_CONFIG = Path("configs/american_dev_attempt_scratch_residual_smooth_floor_v1.toml")
ACCEPTANCE_CONFIG = Path("configs/american_neural_pilot_acceptance_v1.toml")
TAU = SMOOTH_FLOOR_TEMPERATURE


def _physical(count: int = 512) -> np.ndarray:
    """Rows spanning the admitted diagnostic domain, both option types."""
    rng = np.random.default_rng(20260822)
    strike = np.full(count, 100.0)
    # log_moneyness_domain is [-0.7, 0.7] and spot_domain is [50, 150].
    spot = np.clip(strike * np.exp(rng.uniform(-0.7, 0.7, count)), 50.0, 150.0)
    return np.column_stack(
        (
            rng.choice([1.0, -1.0], size=count),
            spot,
            strike,
            rng.uniform(0.05, 3.0, size=count),
            rng.uniform(-0.02, 0.10, size=count),
            rng.uniform(0.0, 0.08, size=count),
            rng.uniform(0.05, 0.8, size=count),
        )
    )


def _extreme_physical() -> np.ndarray:
    """The corners of the admitted domain, where an exponential would blow up."""
    rows = []
    for option_type in (1.0, -1.0):
        for spot in (50.0, 150.0):
            for maturity in (0.05, 3.0):
                for volatility in (0.05, 0.8):
                    for rate in (-0.02, 0.10):
                        for dividend_yield in (0.0, 0.08):
                            rows.append(
                                [
                                    option_type,
                                    spot,
                                    100.0,
                                    maturity,
                                    rate,
                                    dividend_yield,
                                    volatility,
                                ]
                            )
    return np.asarray(rows, dtype=np.float64)


def _model(head: str, physical: np.ndarray) -> AmericanDevPriceModel:
    config = {
        "architecture": {"name": "smooth_mlp", "hidden_dimensions": [8, 8], "activation": "tanh"},
        "head": head,
        "conditioning_features": [],
        "seeds": {"initialization_label": "unit-test/init", "shuffle_label": "unit-test/shuffle"},
    }
    prices = np.linspace(1.0, 30.0, physical.shape[0])
    features, targets = representation_arrays(physical, prices, ())
    return build_model(config, fit_scaling(features, targets))


# ---------------------------------------------------------------------------
# The predeclared temperature, and the units it is stated in
# ---------------------------------------------------------------------------


def test_the_floor_heads_share_exactly_one_predeclared_temperature() -> None:
    """Every floor head applies the same projection, so all carry the same tau."""
    assert "smooth_lower_floor" in HEADS
    assert head_temperature("smooth_lower_floor") == TAU == 1.0e-4
    assert head_temperature("smooth_lower_floor_raw_loss") == TAU
    assert head_temperature("smooth_lower_floor_margin_raw_loss") == TAU
    assert head_temperature("direct") is None
    assert head_temperature("premium_over_european") is None
    assert set(HEAD_TEMPERATURES) == {
        "smooth_lower_floor",
        "smooth_lower_floor_raw_loss",
        "smooth_lower_floor_margin_raw_loss",
    }
    assert len(set(HEAD_TEMPERATURES.values())) == 1
    with pytest.raises(AttemptError, match="unknown head"):
        head_temperature("floor")


def test_the_temperature_is_consistent_with_the_repositorys_normalized_units() -> None:
    """1e-6 < tau < 3e-3, both bounds read from the digest-pinned acceptance file."""
    acceptance = load_toml(ACCEPTANCE_CONFIG)
    assert assert_temperature_consistent("smooth_lower_floor", acceptance) == TAU
    tolerance = acceptance["diagnostics"]["material_normalized_tolerance"]
    accuracy = acceptance["validation_final_entry"]["normalized_rmse_max"]
    assert tolerance < TAU < accuracy
    assert assert_temperature_consistent("direct", acceptance) is None


@pytest.mark.parametrize(
    ("tolerance", "accuracy"), [(1.0e-4, 3.0e-3), (1.0e-6, 1.0e-4), (1.0e-3, 3.0e-3)]
)
def test_an_inconsistent_unit_scale_is_reported_not_silently_resolved(
    tolerance: float, accuracy: float
) -> None:
    """The failure mode is a raised error, never a substituted temperature."""
    acceptance = {
        "diagnostics": {"material_normalized_tolerance": tolerance},
        "validation_final_entry": {"normalized_rmse_max": accuracy},
    }
    with pytest.raises(AttemptError, match="contradict the predeclared"):
        assert_temperature_consistent("smooth_lower_floor", acceptance)


# ---------------------------------------------------------------------------
# The smooth maximum
# ---------------------------------------------------------------------------


def test_the_smooth_maximum_never_falls_below_the_hard_maximum() -> None:
    """Bitwise, in float64: a lower floor may not round downwards."""
    rng = np.random.default_rng(7)
    left = torch.as_tensor(rng.normal(0.0, 2.0, 50_000), dtype=torch.float64)
    right = torch.as_tensor(rng.normal(0.0, 2.0, 50_000), dtype=torch.float64)
    smooth = smooth_maximum(left, right, TAU)
    assert bool((smooth >= torch.maximum(left, right)).all())


def test_the_smooth_maximum_is_bounded_above_by_tau_log_two() -> None:
    rng = np.random.default_rng(8)
    left = torch.as_tensor(rng.normal(0.0, 1.0e-4, 50_000), dtype=torch.float64)
    right = torch.as_tensor(rng.normal(0.0, 1.0e-4, 50_000), dtype=torch.float64)
    excess = smooth_maximum(left, right, TAU) - torch.maximum(left, right)
    assert bool((excess <= TAU * math.log(2.0) * (1.0 + 1.0e-12)).all())
    equal = torch.zeros(4, dtype=torch.float64)
    assert smooth_maximum(equal, equal, TAU)[0].item() == pytest.approx(TAU * math.log(2.0))


def test_the_smooth_maximum_matches_log_sum_exp_where_that_is_computable() -> None:
    values = torch.tensor([0.3, 0.30001, -0.2, 1.5], dtype=torch.float64)
    other = torch.tensor([0.1, 0.30002, 0.7, 1.5], dtype=torch.float64)
    reference = TAU * torch.logsumexp(torch.stack((values, other), dim=-1) / TAU, dim=-1)
    assert torch.allclose(smooth_maximum(values, other, TAU), reference, rtol=0.0, atol=1e-15)


def test_the_smooth_maximum_gradient_is_the_log_sum_exp_softmax() -> None:
    left = torch.tensor([1.0, -1.0, 1.0e-5, 0.0], dtype=torch.float64, requires_grad=True)
    right = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    gradient, = torch.autograd.grad(smooth_maximum(left, right, TAU).sum(), left)
    expected = torch.sigmoid((left.detach() - right) / TAU)
    assert torch.allclose(gradient, expected, rtol=1e-12, atol=1e-15)
    # At an exact tie the two branches split evenly, which is the log-sum-exp value.
    assert gradient[-1].item() == pytest.approx(0.5)


def test_the_smooth_maximum_survives_a_separation_no_exponential_could() -> None:
    """exp is only ever evaluated at a non-positive argument, so it cannot overflow."""
    left = torch.tensor([1.0e6, -1.0e6, 0.0], dtype=torch.float64)
    right = torch.tensor([-1.0e6, 1.0e6, 0.0], dtype=torch.float64)
    smooth = smooth_maximum(left, right, TAU)
    assert bool(torch.isfinite(smooth).all())
    # The correction underflows to exactly zero, which returns the hard maximum.
    assert smooth[0].item() == 1.0e6
    assert smooth[1].item() == 1.0e6


@pytest.mark.parametrize("temperature", [0.0, -1.0e-4, float("inf"), float("nan")])
def test_a_non_positive_or_non_finite_temperature_is_refused(temperature: float) -> None:
    values = torch.zeros(3, dtype=torch.float64)
    with pytest.raises(ValueError, match="finite positive"):
        smooth_maximum(values, values, temperature)
    with pytest.raises(ValueError, match="finite positive"):
        smooth_lower_bound(values, values, temperature)


# ---------------------------------------------------------------------------
# The one-sided projection
# ---------------------------------------------------------------------------


def test_the_projection_never_falls_below_the_floor() -> None:
    rng = np.random.default_rng(9)
    floor = torch.as_tensor(rng.uniform(0.0, 2.0, 50_000), dtype=torch.float64)
    value = floor + torch.as_tensor(rng.normal(0.0, 0.5, 50_000), dtype=torch.float64)
    assert bool((smooth_lower_bound(value, floor, TAU) >= floor).all())


def test_the_projection_reproduces_the_direct_value_far_above_the_floor() -> None:
    """Softplus's linear branch: the transformation is inert where it should be."""
    floor = torch.as_tensor(np.linspace(0.0, 1.0, 256), dtype=torch.float64)
    for separation in (1.0e-2, 1.0e-1, 1.0):
        value = floor + separation
        projected = smooth_lower_bound(value, floor, TAU)
        assert torch.allclose(projected, value, rtol=1.0e-12, atol=0.0)


def test_the_projection_bias_near_the_floor_is_order_tau() -> None:
    floor = torch.as_tensor(np.linspace(0.0, 1.0, 256), dtype=torch.float64)
    at_floor = smooth_lower_bound(floor, floor, TAU) - floor
    assert bool(torch.allclose(at_floor, torch.full_like(at_floor, TAU * math.log(2.0))))
    # One temperature above the floor the bias has already shrunk substantially.
    above = smooth_lower_bound(floor + TAU, floor, TAU) - (floor + TAU)
    assert bool((above < TAU * math.log(2.0)).all())
    assert bool((above > 0.0).all())


def test_the_projection_collapses_onto_the_floor_far_below_it() -> None:
    floor = torch.as_tensor(np.linspace(0.0, 1.0, 64), dtype=torch.float64)
    deep = floor - 1.0e3
    projected = smooth_lower_bound(deep, floor, TAU)
    assert bool(torch.isfinite(projected).all())
    assert bool((projected == floor).all())


def test_the_projection_has_finite_first_and_second_derivatives() -> None:
    """Including at the floor and at softplus's linear-branch switch."""
    floor = torch.zeros(6, dtype=torch.float64)
    value = torch.tensor(
        [-1.0, -TAU, 0.0, TAU, 20.0 * TAU, 25.0 * TAU], dtype=torch.float64, requires_grad=True
    )
    projected = smooth_lower_bound(value, floor, TAU)
    gradient, = torch.autograd.grad(projected.sum(), value, create_graph=True)
    curvature, = torch.autograd.grad(gradient.sum(), value)
    assert bool(torch.isfinite(gradient).all())
    assert bool(torch.isfinite(curvature).all())
    assert bool(((gradient >= 0.0) & (gradient <= 1.0)).all())
    assert gradient[2].item() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# The floor is deployment-computable, and it is the right floor
# ---------------------------------------------------------------------------


def test_the_floor_dominates_both_deployment_computable_bounds() -> None:
    physical = torch.as_tensor(_physical(), dtype=torch.float64)
    scale = discounted_spot(physical)
    floor = normalized_lower_floor(physical, TAU)
    assert bool((floor >= european_price(physical) / scale).all())
    assert bool((floor >= intrinsic_value(physical) / scale).all())
    assert bool(torch.isfinite(floor).all())


def test_the_floor_needs_no_lattice_and_no_stored_column() -> None:
    """No CRR price is computed, read or required at inference."""
    physical = torch.as_tensor(_physical(16), dtype=torch.float64)
    assert bool(torch.isfinite(normalized_lower_floor(physical, TAU)).all())
    source = REPRESENTATION_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(REPRESENTATION_MODULE))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    code_strings = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]
    assert not [text for text in code_strings if "crr" in text.lower()]
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not [name for name in imported if "binomial" in name or "crr" in name]


# ---------------------------------------------------------------------------
# The model: a direct target with a transformed output
# ---------------------------------------------------------------------------


def test_the_head_transforms_the_direct_output_and_predicts_no_premium() -> None:
    """The floor head's own network output is the direct head's, unchanged."""
    physical_array = _physical(128)
    physical = torch.as_tensor(physical_array, dtype=torch.float64)
    direct_model = _model("direct", physical_array)
    floor_model = _model("smooth_lower_floor", physical_array)
    # Same seed label and same architecture, so the two networks are identical.
    floor_model.network.load_state_dict(direct_model.network.state_dict(), strict=True)
    with torch.no_grad():
        direct = direct_model.normalized_target(physical)
        projected = floor_model.normalized_target(physical)
        floor = normalized_lower_floor(physical, TAU)
    assert floor_model.temperature == TAU
    assert direct_model.temperature is None
    # Where the direct output already clears the floor comfortably, the head is inert.
    clear = direct > floor + 40.0 * TAU
    assert bool(clear.any())
    assert torch.allclose(projected[clear], direct[clear], rtol=1.0e-12, atol=0.0)
    # Everywhere, the output is at or above the direct value and at or above the floor.
    assert bool((projected >= direct - 1.0e-12).all())
    assert bool((projected >= floor).all())


def test_the_reconstructed_price_respects_both_floors_to_float64_tolerance() -> None:
    physical_array = _physical(256)
    physical = torch.as_tensor(physical_array, dtype=torch.float64)
    model = _model("smooth_lower_floor", physical_array)
    with torch.no_grad():
        price = model(physical)
        analytic = european_price(physical)
        intrinsic = intrinsic_value(physical)
    # The A * (u) reconstruction round-trip can cost one unit in the last place,
    # which is what this tolerance allows and nothing more. It is far below the
    # 1e-6 * A material tolerance the diagnostics apply.
    slack = 8.0 * np.finfo(np.float64).eps * torch.maximum(analytic.abs(), intrinsic.abs())
    assert bool((price >= analytic - slack).all())
    assert bool((price >= intrinsic - slack).all())
    material = 1.0e-6 * discounted_spot(physical)
    assert bool(((analytic - price) <= material).all())
    assert bool(((intrinsic - price) <= material).all())


def test_the_head_is_finite_on_the_corners_of_the_admitted_domain() -> None:
    physical_array = _extreme_physical()
    physical = torch.as_tensor(physical_array, dtype=torch.float64)
    model = _model("smooth_lower_floor", physical_array)
    with torch.no_grad():
        floor = normalized_lower_floor(physical, TAU)
        target = model.normalized_target(physical)
        price = model(physical)
    for name, tensor in (("floor", floor), ("target", target), ("price", price)):
        assert bool(torch.isfinite(tensor).all()), name
    assert bool((target >= floor).all())


def test_the_head_keeps_finite_gradients_through_the_physical_inputs() -> None:
    physical_array = _physical(64)
    model = _model("smooth_lower_floor", physical_array)
    physical = torch.as_tensor(physical_array, dtype=torch.float64).requires_grad_(True)
    target = model.normalized_target(physical).sum()
    gradient, = torch.autograd.grad(target, physical, create_graph=True)
    curvature, = torch.autograd.grad(gradient.sum(), physical)
    assert bool(torch.isfinite(gradient).all())
    assert bool(torch.isfinite(curvature).all())


def test_the_head_is_reproducible_for_a_fixed_seed_label() -> None:
    physical_array = _physical(32)
    physical = torch.as_tensor(physical_array, dtype=torch.float64)
    with torch.no_grad():
        first = _model("smooth_lower_floor", physical_array)(physical)
        second = _model("smooth_lower_floor", physical_array)(physical)
    assert torch.equal(first, second)


# ---------------------------------------------------------------------------
# The immutable E2 configuration
# ---------------------------------------------------------------------------


def _flatten(section: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in section.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def test_e2_changes_only_the_head_and_its_derived_seed_from_the_direct_parent() -> None:
    parent = _flatten(load_toml(PARENT_CONFIG))
    child = _flatten(load_toml(FLOOR_CONFIG))
    differing = {key for key in parent | child if parent.get(key) != child.get(key)}
    assert differing == {
        "attempt_id",
        "parent_attempt",
        "hypothesis",
        "head",
        "seeds.initialization_label",
        "paths.output_directory",
        "paths.ledger",
    }
    assert parent["head"] == "direct"
    assert child["head"] == "smooth_lower_floor"
    assert child["parent_attempt"] == "scratch_residual_architecture_v1"
    assert child["seeds.shuffle_label"] == parent["seeds.shuffle_label"]


def test_e2_preserves_the_direct_normalized_price_target() -> None:
    """The declared target and reconstruction are the parent's, not a premium."""
    config = validate_attempt_config(load_toml(FLOOR_CONFIG))
    assert config["target"] == "u = V / (S * exp(-q * T))"
    assert config["physical_reconstruction"] == "V = S * exp(-q * T) * u"
    assert config["representation"] == "american_forward_carry_v1"
    assert config["conditioning_features"] == []
    assert config["head"] != "premium_over_european"


def test_e2_keeps_the_residual_backbone_and_the_full_budget() -> None:
    config = validate_attempt_config(load_toml(FLOOR_CONFIG))
    assert config["architecture"] == {
        "name": "smooth_residual",
        "width": 128,
        "blocks": 6,
        "activation": "tanh",
    }
    assert config["training"]["epochs"] == 120
    assert config["training"]["batch_size"] == 2048
    assert config["row_selection"]["row_budget"] == 32768
    assert config["dtype"] == "float64"
    assert config["device"] == "cpu"


@pytest.mark.parametrize("key", ["dataset", "dataset_manifest", "output_directory", "ledger"])
def test_e2_cannot_be_pointed_at_a_final_partition(key: str) -> None:
    config = validate_attempt_config(load_toml(FLOOR_CONFIG))
    config["paths"] = {**config["paths"], key: "data/american-option-v1/interpolation_test"}
    with pytest.raises(FinalPartitionAccessError):
        validate_attempt_config(config)


@pytest.mark.parametrize("section", ["row_selection", "checkpoint"])
def test_e2_cannot_select_on_a_final_partition(section: str) -> None:
    config = validate_attempt_config(load_toml(FLOOR_CONFIG))
    key = "partition" if section == "row_selection" else "selection_partition"
    config[section] = {**config[section], key: "interpolation_test"}
    with pytest.raises(FinalPartitionAccessError):
        validate_attempt_config(config)


def test_e2_uses_the_pinned_criterion_unchanged() -> None:
    config = validate_attempt_config(load_toml(FLOOR_CONFIG))
    assert config["paths"]["acceptance_config"] == str(ACCEPTANCE_CONFIG)
    criterion = load_toml(ACCEPTANCE_CONFIG)["validation_final_entry"]
    assert criterion["normalized_rmse_max"] == 0.003
    assert criterion["maximum_material_bound_violations"] == 0
    assert criterion["maximum_material_shape_violations"] == 0
