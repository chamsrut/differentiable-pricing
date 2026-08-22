"""Task 9H E2c: the additive margin on the European leg of the smooth floor.

Synthetic fixtures throughout. Nothing is trained on a partition and no attempt
is executed; what is pinned is the arithmetic of the transformation and the
composition of the configuration that declares it.

Five properties carry the change:

* the margin is a **code constant with a derived value**, checked against the
  units the acceptance criterion is stated in;
* it is added to the **European leg only**, before the existing smooth maximum;
* at zero margin the floor is **bitwise** what it was, so no earlier head moves;
* the deployed output is bitwise at or above ``E_analytic / A + delta`` wherever
  the European leg dominates, and never below the intrinsic leg;
* the **training loss is unchanged**: E2c and E2b differentiate the same
  pre-projection value, so raising the floor cannot reintroduce E2's saturation.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from differentiable_pricing.ml.american_dev.attempts import (
    EUROPEAN_FLOOR_MARGIN,
    HEAD_EUROPEAN_MARGINS,
    HEAD_TEMPERATURES,
    HEADS,
    RAW_LOSS_HEADS,
    SMOOTH_FLOOR_TEMPERATURE,
    AttemptError,
    FinalPartitionAccessError,
    assert_margin_consistent,
    attempt_seeds,
    head_european_margin,
    load_toml,
    validate_attempt_config,
)
from differentiable_pricing.ml.american_dev.domain import derive_margin
from differentiable_pricing.ml.american_dev.representation import (
    discounted_spot,
    european_price,
    intrinsic_value,
    normalized_lower_floor,
    representation_arrays,
    smooth_maximum,
)
from differentiable_pricing.ml.american_dev.workbench import build_model
from differentiable_pricing.ml.model import fit_scaling

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ACCEPTANCE_CONFIG = Path("configs/american_neural_pilot_acceptance_v1.toml")
E2B_CONFIG = Path("configs/american_dev_attempt_scratch_residual_smooth_floor_raw_loss_v1.toml")
E2C_CONFIG = Path("configs/american_dev_attempt_scratch_residual_smooth_floor_margin_v1.toml")
TAU = SMOOTH_FLOOR_TEMPERATURE
DELTA = EUROPEAN_FLOOR_MARGIN
RAW_LOSS_HEAD = "smooth_lower_floor_raw_loss"
MARGIN_HEAD = "smooth_lower_floor_margin_raw_loss"

#: The supremum the label-free domain characterization measured for
#: ``(E_CRR - E_BS) / A``. Recorded here so the derivation of the margin can be
#: re-run against the rule rather than asserted as a bare number.
MEASURED_DOMAIN_SUPREMUM = 4.192769575172157e-05


def _physical(count: int = 64) -> np.ndarray:
    rng = np.random.default_rng(20260824)
    strike = np.full(count, 100.0)
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


def _config(head: str) -> dict[str, Any]:
    return {
        "architecture": {"name": "smooth_mlp", "hidden_dimensions": [8, 8], "activation": "tanh"},
        "head": head,
        "conditioning_features": [],
        "seeds": {"initialization_label": "unit-test/init", "shuffle_label": "unit-test/shuffle"},
    }


def _fitted(head: str, physical: np.ndarray) -> Any:
    prices = np.linspace(2.0, 40.0, physical.shape[0])
    features, targets = representation_arrays(physical, prices, ())
    return build_model(_config(head), fit_scaling(features, targets))


def _paired_models(physical: np.ndarray) -> tuple[Any, Any]:
    """E2b and E2c sharing one network, so only the head can explain a difference."""
    raw_loss = _fitted(RAW_LOSS_HEAD, physical)
    margin = _fitted(MARGIN_HEAD, physical)
    margin.network.load_state_dict(raw_loss.network.state_dict(), strict=True)
    return raw_loss, margin


def _push_far_below_the_floor(model: Any, physical: torch.Tensor, offset: float = 0.05) -> None:
    """Start every prediction well below the floor, where E2 saturated."""
    layers = [layer for layer in model.network.layers if isinstance(layer, torch.nn.Linear)]
    with torch.no_grad():
        final = layers[-1]
        final.weight.zero_()
        floor = normalized_lower_floor(physical, TAU, model.european_margin)
        final.bias.fill_(float(((floor.min() - offset) - model.price_mean) / model.price_scale))


# ---------------------------------------------------------------------------
# The margin is a code constant with a derived value
# ---------------------------------------------------------------------------


def test_the_margin_is_the_value_the_predeclared_rule_returns() -> None:
    """1e-4, reproduced from the measured supremum through the rule itself."""
    derived = derive_margin(MEASURED_DOMAIN_SUPREMUM)
    assert derived["candidate"] == pytest.approx(9.0e-5)
    assert derived["delta"] == pytest.approx(EUROPEAN_FLOOR_MARGIN)
    assert derived["binding_term"] == "minimum"
    assert derived["realized_safety_factor"] == pytest.approx(2.385, abs=5.0e-4)
    assert EUROPEAN_FLOOR_MARGIN == 1.0e-4


def test_the_margin_is_declared_per_head_and_defaults_to_zero() -> None:
    assert MARGIN_HEAD in HEADS
    assert HEAD_EUROPEAN_MARGINS == {MARGIN_HEAD: EUROPEAN_FLOOR_MARGIN}
    assert head_european_margin(MARGIN_HEAD) == EUROPEAN_FLOOR_MARGIN
    for head in HEADS:
        if head != MARGIN_HEAD:
            assert head_european_margin(head) == 0.0
    with pytest.raises(AttemptError, match="unknown head"):
        head_european_margin("margin")


def test_the_margin_head_keeps_the_temperature_and_the_raw_training_loss() -> None:
    """Only the floor moves: tau and the loss prediction are E2b's."""
    assert HEAD_TEMPERATURES[MARGIN_HEAD] == HEAD_TEMPERATURES[RAW_LOSS_HEAD] == TAU
    assert MARGIN_HEAD in RAW_LOSS_HEADS


def test_the_margin_is_consistent_with_the_repositorys_normalized_units() -> None:
    """1e-6 < delta < 3e-3, both bounds read from the digest-pinned acceptance file."""
    acceptance = load_toml(ACCEPTANCE_CONFIG)
    assert assert_margin_consistent(MARGIN_HEAD, acceptance) == DELTA
    assert float(acceptance["diagnostics"]["material_normalized_tolerance"]) < DELTA
    assert float(acceptance["validation_final_entry"]["normalized_rmse_max"]) > DELTA


def test_a_head_without_a_margin_is_not_checked_against_the_units() -> None:
    """Zero is the absence of the transformation, not a value of it."""
    assert assert_margin_consistent(RAW_LOSS_HEAD, {}) == 0.0
    assert assert_margin_consistent("direct", {}) == 0.0


@pytest.mark.parametrize(
    "acceptance",
    [
        {},
        {"diagnostics": {"material_normalized_tolerance": 1.0e-4}, "validation_final_entry": {}},
        {
            "diagnostics": {"material_normalized_tolerance": 1.0e-3},
            "validation_final_entry": {"normalized_rmse_max": 3.0e-3},
        },
        {
            "diagnostics": {"material_normalized_tolerance": 1.0e-6},
            "validation_final_entry": {"normalized_rmse_max": 1.0e-5},
        },
    ],
)
def test_units_that_contradict_the_margin_are_reported_not_resolved(
    acceptance: dict[str, Any],
) -> None:
    with pytest.raises(AttemptError):
        assert_margin_consistent(MARGIN_HEAD, acceptance)


# ---------------------------------------------------------------------------
# The floor: European leg only, and bitwise unchanged at zero margin
# ---------------------------------------------------------------------------


def test_a_zero_margin_floor_is_bitwise_the_floor_that_already_ran() -> None:
    """No earlier head moves: the addition is skipped, not performed with a zero."""
    physical = torch.as_tensor(_physical(), dtype=torch.float64)
    scale = discounted_spot(physical)
    reference = smooth_maximum(
        european_price(physical) / scale, intrinsic_value(physical) / scale, TAU
    )
    assert torch.equal(normalized_lower_floor(physical, TAU), reference)
    assert torch.equal(normalized_lower_floor(physical, TAU, 0.0), reference)


def test_the_margin_is_added_to_the_european_leg_before_the_smooth_maximum() -> None:
    physical = torch.as_tensor(_physical(), dtype=torch.float64)
    scale = discounted_spot(physical)
    expected = smooth_maximum(
        european_price(physical) / scale + DELTA, intrinsic_value(physical) / scale, TAU
    )
    assert torch.equal(normalized_lower_floor(physical, TAU, DELTA), expected)


def test_the_raised_floor_dominates_the_margined_european_leg_bitwise() -> None:
    physical = torch.as_tensor(_physical(), dtype=torch.float64)
    scale = discounted_spot(physical)
    floor = normalized_lower_floor(physical, TAU, DELTA)
    assert bool((floor >= european_price(physical) / scale + DELTA).all())
    assert bool((floor >= intrinsic_value(physical) / scale).all())


def test_the_floor_rises_by_at_most_the_margin_and_never_falls() -> None:
    """The intrinsic leg is untouched, so the lift is bounded by delta."""
    physical = torch.as_tensor(_physical(256), dtype=torch.float64)
    lift = normalized_lower_floor(physical, TAU, DELTA) - normalized_lower_floor(physical, TAU)
    assert bool((lift >= 0.0).all())
    assert bool((lift <= DELTA + 1.0e-15).all())


def test_the_lift_is_the_full_margin_where_the_european_leg_dominates() -> None:
    physical = torch.as_tensor(
        [[1.0, 100.0, 105.0, 0.5, 0.03, 0.01, 0.25]], dtype=torch.float64
    )
    scale = discounted_spot(physical)
    assert float(european_price(physical) / scale) > float(intrinsic_value(physical) / scale)
    lift = normalized_lower_floor(physical, TAU, DELTA) - normalized_lower_floor(physical, TAU)
    assert float(lift) == pytest.approx(DELTA, rel=1.0e-9)


def test_a_deep_in_the_money_state_keeps_its_intrinsic_floor() -> None:
    """Where intrinsic dominates by many temperatures, the margin buys nothing."""
    physical = torch.as_tensor(
        [[-1.0, 60.0, 100.0, 0.05, 0.02, 0.0, 0.10]], dtype=torch.float64
    )
    scale = discounted_spot(physical)
    assert float(intrinsic_value(physical) / scale) > float(european_price(physical) / scale)
    lift = normalized_lower_floor(physical, TAU, DELTA) - normalized_lower_floor(physical, TAU)
    assert 0.0 <= float(lift) < TAU


@pytest.mark.parametrize("margin", [-1.0e-6, float("nan"), float("inf")])
def test_a_degenerate_margin_fails_closed(margin: float) -> None:
    physical = torch.as_tensor(_physical(8), dtype=torch.float64)
    with pytest.raises(ValueError, match="margin"):
        normalized_lower_floor(physical, TAU, margin)


# ---------------------------------------------------------------------------
# The model: one changed floor, one unchanged loss
# ---------------------------------------------------------------------------


def test_the_model_reads_the_margin_from_the_single_definition() -> None:
    physical = _physical()
    raw_loss, margin = _paired_models(physical)
    assert raw_loss.european_margin == 0.0
    assert margin.european_margin == DELTA
    assert margin.temperature == raw_loss.temperature == TAU


def test_the_two_heads_differentiate_the_same_prediction() -> None:
    """E2c's training loss is E2b's, bit for bit: only deployment changes."""
    physical_array = _physical()
    physical = torch.as_tensor(physical_array, dtype=torch.float64)
    raw_loss, margin = _paired_models(physical_array)
    with torch.no_grad():
        assert torch.equal(
            raw_loss.training_target(physical), margin.training_target(physical)
        )
        assert not torch.equal(
            raw_loss.normalized_target(physical), margin.normalized_target(physical)
        )


def test_the_raised_floor_does_not_reintroduce_the_saturated_gradient() -> None:
    """Far below the floor the loss still reaches the network, as in E2b."""
    physical_array = _physical()
    physical = torch.as_tensor(physical_array, dtype=torch.float64)
    prices = np.linspace(2.0, 40.0, physical_array.shape[0])
    _, targets_array = representation_arrays(physical_array, prices, ())
    targets = torch.as_tensor(targets_array, dtype=torch.float64)
    model = _fitted(MARGIN_HEAD, physical_array)
    _push_far_below_the_floor(model, physical)
    model.zero_grad(set_to_none=True)
    predicted = model.training_target(physical)
    torch.mean(torch.square((predicted - targets) / model.price_scale)).backward()
    norm = float(
        torch.sqrt(
            sum(
                torch.sum(torch.square(parameter.grad))
                for parameter in model.network.parameters()
                if parameter.grad is not None
            )
        )
    )
    assert norm > 1.0e-3


def test_the_deployed_price_stays_at_or_above_the_margined_european_value() -> None:
    """The physical bound is a near-bound: one ulp of the ``A * (E / A)`` round-trip."""
    physical_array = _physical(128)
    physical = torch.as_tensor(physical_array, dtype=torch.float64)
    model = _fitted(MARGIN_HEAD, physical_array)
    _push_far_below_the_floor(model, physical)
    with torch.no_grad():
        price = model(physical)
        scale = discounted_spot(physical)
        analytic = european_price(physical)
        intrinsic = intrinsic_value(physical)
    slack = 8.0 * torch.finfo(torch.float64).eps * torch.abs(price)
    assert bool((price >= analytic + DELTA * scale - slack).all())
    assert bool((price >= intrinsic - slack).all())


def test_the_deployed_output_is_the_projection_onto_the_raised_floor() -> None:
    physical_array = _physical()
    physical = torch.as_tensor(physical_array, dtype=torch.float64)
    model = _fitted(MARGIN_HEAD, physical_array)
    with torch.no_grad():
        floor = normalized_lower_floor(physical, TAU, DELTA)
        deployed = model.normalized_target(physical)
    assert bool((deployed >= floor).all())


# ---------------------------------------------------------------------------
# The E2c configuration
# ---------------------------------------------------------------------------


def _flatten(section: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in section.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


def test_e2c_changes_only_the_head_and_its_derived_identity_from_e2b() -> None:
    parent = _flatten(load_toml(E2B_CONFIG))
    child = _flatten(load_toml(E2C_CONFIG))
    assert set(parent) == set(child)
    differing = {key for key in parent if parent[key] != child[key]}
    assert differing == {
        "attempt_id",
        "parent_attempt",
        "hypothesis",
        "head",
        "seeds.initialization_label",
        "paths.output_directory",
        "paths.ledger",
    }
    assert child["head"] == MARGIN_HEAD
    assert child["parent_attempt"] == "scratch_residual_smooth_floor_raw_loss_v1"


def test_e2c_keeps_the_backbone_target_budget_and_shuffle_seed() -> None:
    parent = validate_attempt_config(load_toml(E2B_CONFIG))
    child = validate_attempt_config(load_toml(E2C_CONFIG))
    assert child["architecture"] == parent["architecture"] == {
        "name": "smooth_residual",
        "width": 128,
        "blocks": 6,
        "activation": "tanh",
    }
    assert child["conditioning_features"] == []
    assert child["target"] == parent["target"]
    assert child["row_selection"] == parent["row_selection"]
    assert child["training"] == parent["training"]
    assert child["optimizer"] == parent["optimizer"]
    assert child["checkpoint"] == parent["checkpoint"]
    assert child["evaluation"] == parent["evaluation"]
    assert attempt_seeds(child)["shuffle"] == attempt_seeds(parent)["shuffle"]
    assert attempt_seeds(child)["initialization"] != attempt_seeds(parent)["initialization"]


def test_e2c_declares_no_margin_field_of_its_own() -> None:
    """The margin is pinned in source_digests, not in an editable configuration."""
    text = E2C_CONFIG.read_text(encoding="utf-8")
    body = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
    assert "margin =" not in body
    assert "delta =" not in body
    assert "temperature" not in body


@pytest.mark.parametrize("key", ["dataset", "dataset_manifest", "output_directory", "ledger"])
def test_e2c_cannot_be_pointed_at_a_final_partition(key: str) -> None:
    config = validate_attempt_config(load_toml(E2C_CONFIG))
    config["paths"] = {**config["paths"], key: "data/interpolation_test.parquet"}
    with pytest.raises(FinalPartitionAccessError):
        validate_attempt_config(config)


def test_e2c_adds_a_configuration_and_edits_none() -> None:
    """A used configuration is immutable."""
    tracked = subprocess.run(
        ["git", "ls-files", "configs/american_dev_attempt_*.toml"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    earlier = [name for name in tracked if Path(name).name != E2C_CONFIG.name]
    assert len(earlier) == 8
    modified = subprocess.run(
        ["git", "diff", "--name-only", "--", *earlier],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert modified == []
