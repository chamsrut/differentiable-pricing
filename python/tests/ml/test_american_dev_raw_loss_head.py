"""Task 9H E2b: a floored deployment output with the loss on the raw prediction.

Synthetic fixtures throughout. No dataset partition is opened, no attempt is run
and no numerical study happens: :func:`train_model` is exercised only on a
handful of in-memory rows for a few epochs, exactly as the existing workbench
tests do.

The property under test is a **measured** one. ``scratch_residual_smooth_floor_v1``
(E2) computed its loss through the smooth-floor projection and stopped at best
epoch 1. The projection's derivative is ``sigmoid((raw_u - floor) / tau)``, of
order ``1e-44`` at ``tau = 1e-4`` once a prediction sits a hundredth of a
discounted spot below the floor -- which is where a scratch network starts.
These tests pin both halves: that the projected-loss path really is saturated in
that regime, and that the raw-loss path really does carry a material gradient
through the same weights and the same rows.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from differentiable_pricing.ml.american_dev.attempts import (
    HEAD_TEMPERATURES,
    HEADS,
    RAW_LOSS_HEADS,
    SMOOTH_FLOOR_TEMPERATURE,
    FinalPartitionAccessError,
    load_toml,
    sha256_file,
    validate_attempt_config,
)
from differentiable_pricing.ml.american_dev.representation import (
    discounted_spot,
    european_price,
    intrinsic_value,
    normalized_lower_floor,
    representation_arrays,
)
from differentiable_pricing.ml.american_dev.workbench import build_model, train_model
from differentiable_pricing.ml.model import fit_scaling

PROJECT_ROOT = Path(__file__).resolve().parents[3]
E2_CONFIG = Path("configs/american_dev_attempt_scratch_residual_smooth_floor_v1.toml")
E2B_CONFIG = Path("configs/american_dev_attempt_scratch_residual_smooth_floor_raw_loss_v1.toml")
E2C_CONFIG = Path("configs/american_dev_attempt_scratch_residual_smooth_floor_margin_v1.toml")
PARENT_CONFIG = Path("configs/american_dev_attempt_scratch_residual_architecture_v1.toml")
TAU = SMOOTH_FLOOR_TEMPERATURE
PROJECTED_HEAD = "smooth_lower_floor"
RAW_LOSS_HEAD = "smooth_lower_floor_raw_loss"


def _physical(count: int = 48) -> np.ndarray:
    rng = np.random.default_rng(20260823)
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
        "training": {
            "batch_size": 16,
            "epochs": 4,
            "num_threads": 1,
            "deterministic_algorithms": True,
        },
        "optimizer": {"learning_rate": 0.05, "schedule_period_epochs": 4},
    }


def _fitted(head: str, physical: np.ndarray) -> tuple[Any, np.ndarray]:
    prices = np.linspace(2.0, 40.0, physical.shape[0])
    features, targets = representation_arrays(physical, prices, ())
    model = build_model(_config(head), fit_scaling(features, targets))
    return model, targets


def _push_far_below_the_floor(model: Any, physical: torch.Tensor, margin: float = 0.05) -> None:
    """Set the output bias so every prediction starts well below the floor.

    ``margin`` is in normalized units, so ``0.05`` is 500 temperatures below the
    floor -- squarely in the regime a scratch network occupies at epoch 0 and the
    regime E2 never escaped.
    """
    layers = [layer for layer in model.network.layers if isinstance(layer, torch.nn.Linear)]
    with torch.no_grad():
        final = layers[-1]
        final.weight.zero_()
        floor = normalized_lower_floor(physical, TAU)
        lowest = float(((floor.min() - margin) - model.price_mean) / model.price_scale)
        final.bias.fill_(lowest)


def _loss(model: Any, physical: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """The workbench's batch loss, spelled the same way."""
    predicted = model.training_target(physical)
    return torch.mean(torch.square((predicted - targets) / model.price_scale))


def _gradient_norm(model: Any, physical: torch.Tensor, targets: torch.Tensor) -> float:
    model.zero_grad(set_to_none=True)
    _loss(model, physical, targets).backward()
    return float(
        torch.sqrt(
            sum(
                torch.sum(torch.square(parameter.grad))
                for parameter in model.network.parameters()
                if parameter.grad is not None
            )
        )
    )


# ---------------------------------------------------------------------------
# The saturation E2 hit, and the gradient E2b restores
# ---------------------------------------------------------------------------


def test_the_projected_loss_is_saturated_far_below_the_floor() -> None:
    """E2's path: the loss is large and the gradient is numerically nothing."""
    physical_array = _physical()
    physical = torch.as_tensor(physical_array, dtype=torch.float64)
    model, targets = _fitted(PROJECTED_HEAD, physical_array)
    _push_far_below_the_floor(model, physical)
    target_tensor = torch.as_tensor(targets, dtype=torch.float64)
    loss = float(_loss(model, physical, target_tensor).detach())
    norm = _gradient_norm(model, physical, target_tensor)
    # The model is badly wrong ...
    assert loss > 1.0e-3
    # ... and yet essentially no gradient reaches the network.
    assert norm < 1.0e-20


def test_the_raw_loss_has_a_material_gradient_in_the_same_case() -> None:
    """E2b's path: identical weights, identical rows, a usable gradient."""
    physical_array = _physical()
    physical = torch.as_tensor(physical_array, dtype=torch.float64)
    projected, targets = _fitted(PROJECTED_HEAD, physical_array)
    raw_loss, _ = _fitted(RAW_LOSS_HEAD, physical_array)
    raw_loss.network.load_state_dict(projected.network.state_dict(), strict=True)
    _push_far_below_the_floor(projected, physical)
    raw_loss.network.load_state_dict(projected.network.state_dict(), strict=True)
    target_tensor = torch.as_tensor(targets, dtype=torch.float64)

    saturated = _gradient_norm(projected, physical, target_tensor)
    material = _gradient_norm(raw_loss, physical, target_tensor)
    assert np.isfinite(material)
    assert material > 1.0e-2
    # Not a marginal improvement: many orders of magnitude.
    assert material > saturated * 1.0e15


def test_the_saturation_factor_is_the_projection_derivative() -> None:
    """The mechanism, stated as arithmetic rather than as a story."""
    below = torch.tensor([-0.05, -0.01, -1.0e-3, 0.0], dtype=torch.float64)
    derivative = torch.sigmoid(below / TAU)
    assert derivative[0].item() < 1.0e-200
    assert derivative[1].item() < 1.0e-40
    assert derivative[2].item() < 1.0e-4
    assert derivative[3].item() == pytest.approx(0.5)


def test_training_moves_the_raw_loss_model_and_leaves_the_projected_one_pinned() -> None:
    """The same failure, end to end through the workbench's own training loop."""
    physical_array = _physical()
    physical = torch.as_tensor(physical_array, dtype=torch.float64)
    deltas: dict[str, float] = {}
    records: dict[str, dict[str, Any]] = {}
    for head in (PROJECTED_HEAD, RAW_LOSS_HEAD):
        model, targets = _fitted(head, physical_array)
        _push_far_below_the_floor(model, physical)
        before = [parameter.detach().clone() for parameter in model.network.parameters()]
        records[head] = train_model(
            model, _config(head), physical_array, targets, physical_array, targets
        )
        deltas[head] = float(
            torch.sqrt(
                sum(
                    torch.sum(torch.square(after.detach() - start))
                    for after, start in zip(model.network.parameters(), before, strict=True)
                )
            )
        )
    assert deltas[PROJECTED_HEAD] == pytest.approx(0.0, abs=1.0e-12)
    assert deltas[RAW_LOSS_HEAD] > 1.0e-3
    assert records[PROJECTED_HEAD]["best_epoch"] == 1


# ---------------------------------------------------------------------------
# Which prediction is trained, and which is selected and evaluated
# ---------------------------------------------------------------------------


def test_the_raw_loss_is_bit_identical_to_the_direct_models_loss() -> None:
    """The latent problem is exactly the one the residual architecture solved.

    Two independent statements of the same number: the ``direct`` head's batch
    loss under identical weights, and ``MSE(raw_standardized,
    standardized_direct_price_target)`` written out longhand. Both must equal the
    raw-loss head's batch loss bit for bit, or the head is optimizing something
    other than what it claims to.
    """
    physical_array = _physical()
    physical = torch.as_tensor(physical_array, dtype=torch.float64)
    direct, targets = _fitted("direct", physical_array)
    raw_loss, _ = _fitted(RAW_LOSS_HEAD, physical_array)
    raw_loss.network.load_state_dict(direct.network.state_dict(), strict=True)
    target_tensor = torch.as_tensor(targets, dtype=torch.float64)
    with torch.no_grad():
        direct_loss = _loss(direct, physical, target_tensor)
        head_loss = _loss(raw_loss, physical, target_tensor)
        standardized_raw = raw_loss._network_output(physical)
        standardized_target = (target_tensor - raw_loss.price_mean) / raw_loss.price_scale
        longhand = torch.mean(torch.square(standardized_raw - standardized_target))
    assert torch.equal(head_loss, direct_loss)
    assert torch.equal(head_loss, longhand)
    # And it is emphatically not the projected loss on the same weights.
    projected, _ = _fitted(PROJECTED_HEAD, physical_array)
    projected.network.load_state_dict(direct.network.state_dict(), strict=True)
    _push_far_below_the_floor(projected, physical)
    raw_loss.network.load_state_dict(projected.network.state_dict(), strict=True)
    with torch.no_grad():
        assert not torch.equal(
            _loss(raw_loss, physical, target_tensor),
            _loss(projected, physical, target_tensor),
        )


def test_the_raw_loss_head_trains_on_the_pre_projection_prediction() -> None:
    physical_array = _physical()
    physical = torch.as_tensor(physical_array, dtype=torch.float64)
    model, _ = _fitted(RAW_LOSS_HEAD, physical_array)
    _push_far_below_the_floor(model, physical)
    with torch.no_grad():
        training = model.training_target(physical)
        deployed = model.normalized_target(physical)
        floor = normalized_lower_floor(physical, TAU)
    # The training prediction is below the floor; the deployed one is not.
    assert bool((training < floor).all())
    assert bool((deployed >= floor).all())
    assert not torch.allclose(training, deployed)


@pytest.mark.parametrize("head", ["direct", "premium_over_european", PROJECTED_HEAD])
def test_every_other_head_trains_on_exactly_what_it_deploys(head: str) -> None:
    """The distinction is one head wide, and it is not a general framework."""
    physical_array = _physical()
    physical = torch.as_tensor(physical_array, dtype=torch.float64)
    model, _ = _fitted(head, physical_array)
    with torch.no_grad():
        assert torch.equal(model.training_target(physical), model.normalized_target(physical))
    assert head not in RAW_LOSS_HEADS
    # Every raw-loss head is a floor head: the distinction is which prediction
    # the loss reads, never a general per-attempt objective axis.
    assert all(name.startswith("smooth_lower_floor") for name in RAW_LOSS_HEADS)


def test_the_two_floor_heads_deploy_bit_identical_outputs() -> None:
    """Only the loss changes: the deployed transformation is E2's, unchanged."""
    physical_array = _physical()
    physical = torch.as_tensor(physical_array, dtype=torch.float64)
    projected, _ = _fitted(PROJECTED_HEAD, physical_array)
    raw_loss, _ = _fitted(RAW_LOSS_HEAD, physical_array)
    raw_loss.network.load_state_dict(projected.network.state_dict(), strict=True)
    with torch.no_grad():
        assert torch.equal(
            projected.normalized_target(physical), raw_loss.normalized_target(physical)
        )
        assert torch.equal(projected(physical), raw_loss(physical))
    assert projected.temperature == raw_loss.temperature == TAU


def test_the_training_record_names_both_predictions() -> None:
    physical_array = _physical()
    for head, expected in (
        (RAW_LOSS_HEAD, "pre-projection direct normalized price"),
        (PROJECTED_HEAD, "the deployed normalized_target output"),
        ("direct", "the deployed normalized_target output"),
    ):
        model, targets = _fitted(head, physical_array)
        record = train_model(
            model, _config(head), physical_array, targets, physical_array, targets
        )
        assert record["loss_prediction"] == expected
        assert record["selection_prediction"] == "the deployed normalized_target output"


def test_checkpoint_selection_reads_the_deployed_prediction() -> None:
    """The recorded metric is the projected one, not the latent one."""
    physical_array = _physical()
    model, targets = _fitted(RAW_LOSS_HEAD, physical_array)
    record = train_model(
        model, _config(RAW_LOSS_HEAD), physical_array, targets, physical_array, targets
    )
    physical = torch.as_tensor(physical_array, dtype=torch.float64)
    target_tensor = torch.as_tensor(targets, dtype=torch.float64)
    with torch.no_grad():
        deployed = model.normalized_target(physical)
        latent = model.training_target(physical)
    projected_metric = float(
        torch.mean(torch.square((deployed - target_tensor) / model.price_scale))
    )
    latent_metric = float(torch.mean(torch.square((latent - target_tensor) / model.price_scale)))
    best = record["best_validation_standardized_target_mse"]
    assert best == pytest.approx(projected_metric, rel=1e-12)
    assert best != pytest.approx(latent_metric, rel=1e-12)


# ---------------------------------------------------------------------------
# The floors still hold
# ---------------------------------------------------------------------------


def test_the_deployed_output_still_enforces_both_floors() -> None:
    physical_array = _physical(256)
    physical = torch.as_tensor(physical_array, dtype=torch.float64)
    model, _ = _fitted(RAW_LOSS_HEAD, physical_array)
    _push_far_below_the_floor(model, physical)
    with torch.no_grad():
        normalized = model.normalized_target(physical)
        price = model(physical)
        analytic = european_price(physical)
        intrinsic = intrinsic_value(physical)
        scale = discounted_spot(physical)
    assert bool((normalized >= analytic / scale).all())
    assert bool((normalized >= intrinsic / scale).all())
    material = 1.0e-6 * scale
    assert bool(((analytic - price) <= material).all())
    assert bool(((intrinsic - price) <= material).all())


def test_the_raw_loss_head_carries_the_same_predeclared_temperature() -> None:
    assert RAW_LOSS_HEAD in HEADS
    assert HEAD_TEMPERATURES[RAW_LOSS_HEAD] == HEAD_TEMPERATURES[PROJECTED_HEAD] == 1.0e-4
    assert {PROJECTED_HEAD, RAW_LOSS_HEAD} <= set(HEAD_TEMPERATURES)
    assert len(set(HEAD_TEMPERATURES.values())) == 1


# ---------------------------------------------------------------------------
# The E2b configuration, and the untouched ones before it
# ---------------------------------------------------------------------------


def _flatten(section: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in section.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def test_e2b_changes_only_the_head_and_its_derived_seed_from_e2() -> None:
    parent = _flatten(load_toml(E2_CONFIG))
    child = _flatten(load_toml(E2B_CONFIG))
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
    assert parent["head"] == PROJECTED_HEAD
    assert child["head"] == RAW_LOSS_HEAD
    assert child["parent_attempt"] == "scratch_residual_smooth_floor_v1"


def test_e2b_keeps_the_backbone_features_target_budget_and_shuffle_seed() -> None:
    child = validate_attempt_config(load_toml(E2B_CONFIG))
    grandparent = load_toml(PARENT_CONFIG)
    assert child["architecture"] == {
        "name": "smooth_residual",
        "width": 128,
        "blocks": 6,
        "activation": "tanh",
    }
    assert child["conditioning_features"] == []
    assert child["target"] == "u = V / (S * exp(-q * T))"
    assert child["physical_reconstruction"] == "V = S * exp(-q * T) * u"
    assert child["representation"] == "american_forward_carry_v1"
    assert child["training"] == grandparent["training"]
    assert child["optimizer"] == grandparent["optimizer"]
    assert child["checkpoint"] == grandparent["checkpoint"]
    assert child["row_selection"] == grandparent["row_selection"]
    assert child["seeds"]["shuffle_label"] == grandparent["seeds"]["shuffle_label"]
    assert child["seeds"]["initialization_label"] != grandparent["seeds"]["initialization_label"]
    assert child["dtype"] == "float64"
    assert child["device"] == "cpu"


@pytest.mark.parametrize("key", ["dataset", "dataset_manifest", "output_directory", "ledger"])
def test_e2b_cannot_be_pointed_at_a_final_partition(key: str) -> None:
    config = validate_attempt_config(load_toml(E2B_CONFIG))
    config["paths"] = {**config["paths"], key: "data/american-option-v1/interpolation_test"}
    with pytest.raises(FinalPartitionAccessError):
        validate_attempt_config(config)


@pytest.mark.parametrize("section", ["row_selection", "checkpoint"])
def test_e2b_cannot_select_on_a_final_partition(section: str) -> None:
    config = validate_attempt_config(load_toml(E2B_CONFIG))
    key = "partition" if section == "row_selection" else "selection_partition"
    config[section] = {**config[section], key: "interpolation_test"}
    with pytest.raises(FinalPartitionAccessError):
        validate_attempt_config(config)


def test_every_earlier_attempt_configuration_is_byte_identical_to_head() -> None:
    """A used configuration is immutable; E2b adds a file, it edits none."""
    tracked = subprocess.run(
        ["git", "ls-files", "configs/american_dev_attempt_*.toml"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    later = {E2B_CONFIG.name, E2C_CONFIG.name}
    earlier = [name for name in tracked if Path(name).name not in later]
    assert len(earlier) == 7
    modified = subprocess.run(
        ["git", "diff", "--name-only", "--", *earlier],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    assert modified == []


def test_the_e2_evidence_that_survived_is_untouched() -> None:
    """E2's summary still matches the hash its immutable ledger recorded."""
    ledger = PROJECT_ROOT / "runs/task-9h/scratch_residual_smooth_floor_v1/attempt.json"
    summary = PROJECT_ROOT / "artifacts/task-9h/scratch_residual_smooth_floor_v1/summary.json"
    if not ledger.is_file() or not summary.is_file():
        pytest.skip("E2's ignored run artifacts are not present in this clone")
    import json

    recorded = json.loads(ledger.read_text(encoding="utf-8"))
    assert recorded["summary"]["sha256"] == sha256_file(summary)
    assert recorded["status"] == "complete"
