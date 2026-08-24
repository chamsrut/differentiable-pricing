"""Task 9H frozen checkpoints: identity, and the deployed head's decomposition.

Every test here runs against a synthetic attempt tree or a freshly constructed
model. None of them opens a dataset partition, trains anything, or reads the
real checkpoints -- which live under the ignored ``artifacts/`` tree and are not
present in a fresh clone.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Final

import numpy as np
import pytest
import torch
from differentiable_pricing.ml.american_dev.attempts import (
    ATTEMPT_LOG_PATH,
    AttemptError,
    FinalPartitionAccessError,
    load_toml,
    sha256_file,
)
from differentiable_pricing.ml.american_dev.frozen import (
    ATTEMPT_REPORT_NAME,
    CHECKPOINT_NAME,
    DEPLOYMENT_FUNCTION,
    FLOOR_HEADS,
    FROZEN_ATTEMPTS,
    FrozenCheckpointError,
    assert_floor_head,
    assert_frozen,
    deployed_floor,
    direct_normalized,
    floor_leg_weight,
    load_frozen_model,
    normalized_with_margin,
    price_with_margin,
    projection_weight,
    raw_price,
    recorded_scaling,
)
from differentiable_pricing.ml.american_dev.models import build_network
from differentiable_pricing.ml.american_dev.representation import (
    AmericanDevPriceModel,
    discounted_spot,
)
from differentiable_pricing.ml.american_dev.workbench import build_model
from differentiable_pricing.ml.model import Scaling

PROJECT_ROOT: Final = Path(__file__).resolve().parents[3]
E2B: Final = "scratch_residual_smooth_floor_raw_loss_v1"
E2C: Final = "scratch_residual_smooth_floor_margin_v1"
E2C_CONFIG: Final = "configs/american_dev_attempt_scratch_residual_smooth_floor_margin_v1.toml"


def _states() -> torch.Tensor:
    """A handful of contracts spanning both types, moneyness and maturity."""
    return torch.tensor(
        [
            [-1.0, 100.0, 100.0, 1.0, 0.05, 0.02, 0.20],
            [1.0, 120.0, 90.0, 0.10, 0.03, 0.00, 0.40],
            [-1.0, 60.0, 140.0, 2.50, -0.01, 0.08, 0.70],
            [1.0, 100.0, 100.0, 0.019178082191780823, 0.05, 0.00, 0.05],
        ],
        dtype=torch.float64,
    )


def _model(head: str, *, width: int = 16, blocks: int = 2) -> AmericanDevPriceModel:
    torch.manual_seed(11)
    network = build_network(
        {"name": "smooth_residual", "width": width, "blocks": blocks, "activation": "tanh"}, 5
    )
    scaling = Scaling(
        feature_mean=np.zeros(5, dtype=np.float64),
        feature_scale=np.ones(5, dtype=np.float64),
        price_mean=0.2,
        price_scale=0.15,
    )
    model = AmericanDevPriceModel(network, scaling, head=head)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_the_frozen_set_is_exactly_e2b_and_e2c() -> None:
    assert set(FROZEN_ATTEMPTS) == {E2B, E2C}
    assert FROZEN_ATTEMPTS[E2B]["label"] == "E2b"
    assert FROZEN_ATTEMPTS[E2C]["label"] == "E2c"


def test_an_unregistered_attempt_is_not_freezable() -> None:
    with pytest.raises(FrozenCheckpointError):
        assert_frozen("scratch_direct_control_v1")


def test_the_deployment_function_is_written_out_not_described() -> None:
    """Every later report cites this string beside the numbers it produced."""
    for line in ("u_dir", "floor", "softplus", "log1p", "delta", "tau"):
        assert line in DEPLOYMENT_FUNCTION


# ---------------------------------------------------------------------------
# The margin override reproduces the native head exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head", FLOOR_HEADS)
def test_the_margin_override_reproduces_the_native_head_bitwise(head: str) -> None:
    """B.0's whole premise: the margin is a deployment transform, not a weight."""
    model = _model(head)
    states = _states()
    with torch.no_grad():
        native = model.normalized_target(states)
        override = normalized_with_margin(model, states)
        explicit = normalized_with_margin(model, states, model.european_margin)
    assert torch.equal(native, override)
    assert torch.equal(native, explicit)


def test_a_zero_margin_weight_set_evaluated_at_delta_matches_a_margin_head() -> None:
    """The off-diagonal cells of the 2x2 are exact, not approximate.

    E2b's weights under ``delta = 1e-4`` must be bitwise what a margin head
    would compute from the same weights, or the row difference in B.0 would
    conflate the margin with an implementation difference.
    """
    torch.manual_seed(11)
    network = build_network(
        {"name": "smooth_residual", "width": 16, "blocks": 2, "activation": "tanh"}, 5
    )
    scaling = Scaling(
        feature_mean=np.zeros(5, dtype=np.float64),
        feature_scale=np.ones(5, dtype=np.float64),
        price_mean=0.2,
        price_scale=0.15,
    )
    zero_margin = AmericanDevPriceModel(network, scaling, head="smooth_lower_floor_raw_loss")
    margined = AmericanDevPriceModel(
        network, scaling, head="smooth_lower_floor_margin_raw_loss"
    )
    zero_margin.eval()
    margined.eval()
    states = _states()
    with torch.no_grad():
        crossed = normalized_with_margin(zero_margin, states, margined.european_margin)
        native = margined.normalized_target(states)
    assert torch.equal(crossed, native)


def test_the_override_never_mutates_the_model() -> None:
    model = _model("smooth_lower_floor_margin_raw_loss")
    before = model.european_margin
    with torch.no_grad():
        normalized_with_margin(model, _states(), 0.0)
    assert model.european_margin == before


def test_a_head_without_a_floor_refuses_every_decomposition() -> None:
    model = _model("direct")
    states = _states()
    with pytest.raises(FrozenCheckpointError):
        assert_floor_head(model)
    for call in (projection_weight, floor_leg_weight, deployed_floor, normalized_with_margin):
        with pytest.raises(FrozenCheckpointError):
            call(model, states)


# ---------------------------------------------------------------------------
# The projection weight really is the projection's derivative
# ---------------------------------------------------------------------------


def test_the_projection_weight_is_the_autograd_derivative_of_the_deployed_output() -> None:
    """``s`` needs no invented threshold because it is an exact derivative.

    Differentiating the deployed normalized output with respect to the direct
    normalized price has to return ``sigmoid((u_dir - floor) / tau)``. This
    asserts that against autograd rather than trusting the algebra.
    """
    model = _model("smooth_lower_floor_margin_raw_loss")
    states = _states()
    floor = deployed_floor(model, states).detach()
    direct = direct_normalized(model, states).detach().requires_grad_(True)
    deployed = floor + model.temperature * torch.nn.functional.softplus(
        (direct - floor) / model.temperature
    )
    (gradient,) = torch.autograd.grad(deployed.sum(), direct)
    with torch.no_grad():
        weight = projection_weight(model, states)
    assert torch.allclose(gradient, weight, rtol=0.0, atol=1e-15)


def test_the_projection_weight_lies_in_the_unit_interval() -> None:
    model = _model("smooth_lower_floor_raw_loss")
    with torch.no_grad():
        weight = projection_weight(model, _states())
    assert bool(((weight >= 0.0) & (weight <= 1.0)).all())


def test_the_floor_leg_weight_is_the_smooth_maximums_derivative() -> None:
    """``w`` separates European-leg from intrinsic-leg dominance exactly."""
    model = _model("smooth_lower_floor_margin_raw_loss")
    states = _states()
    with torch.no_grad():
        weight = floor_leg_weight(model, states)
    assert bool(((weight >= 0.0) & (weight <= 1.0)).all())
    # Deep in the money on a call with zero yield the intrinsic leg dominates,
    # so the European leg's weight must be strictly below one there.
    deep = torch.tensor([[1.0, 150.0, 55.0, 0.02, 0.0, 0.0, 0.05]], dtype=torch.float64)
    with torch.no_grad():
        assert float(floor_leg_weight(model, deep)) < 1.0


def test_the_raw_and_deployed_prices_reconstruct_through_the_same_scale() -> None:
    """Section F contrasts these two; they must differ only by the projection."""
    model = _model("smooth_lower_floor_margin_raw_loss")
    states = _states()
    with torch.no_grad():
        scale = discounted_spot(states)
        assert torch.equal(raw_price(model, states), scale * direct_normalized(model, states))
        assert torch.equal(
            price_with_margin(model, states), scale * normalized_with_margin(model, states)
        )
        assert torch.equal(price_with_margin(model, states), model(states))


def test_the_deployed_output_never_falls_below_its_floor() -> None:
    model = _model("smooth_lower_floor_margin_raw_loss")
    states = _states()
    with torch.no_grad():
        assert bool(
            (normalized_with_margin(model, states) >= deployed_floor(model, states)).all()
        )


# ---------------------------------------------------------------------------
# Loading, and what it refuses
# ---------------------------------------------------------------------------


def _synthetic_tree(root: Path, *, attempt_id: str = E2C, empty_report: bool = False) -> Path:
    (root / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PROJECT_ROOT / E2C_CONFIG, root / E2C_CONFIG)
    digest = sha256_file(root / E2C_CONFIG)
    directory = root / FROZEN_ATTEMPTS[E2C]["directory"]
    directory.mkdir(parents=True, exist_ok=True)
    config = load_toml(root / E2C_CONFIG)
    scaling = Scaling(
        feature_mean=np.zeros(5, dtype=np.float64),
        feature_scale=np.ones(5, dtype=np.float64),
        price_mean=0.2,
        price_scale=0.15,
    )
    model = build_model(config, scaling)
    torch.save(model.network.state_dict(), directory / CHECKPOINT_NAME)
    parameters = int(sum(int(tensor.numel()) for tensor in model.network.parameters()))
    report: dict[str, Any] = {
        "attempt_id": attempt_id,
        "config_path": E2C_CONFIG,
        "config_sha256": digest,
        "architecture": {"parameters": parameters},
        "training": {"best_epoch": 89},
        "repository": {"commit": "0" * 40},
        "seeds": {"initialization": 259930797, "shuffle": 3544330082},
        "scaling": {
            "feature_mean": [0.0] * 5,
            "feature_scale": [1.0] * 5,
            "target_mean": 0.2,
            "target_scale": 0.15,
        },
    }
    (directory / ATTEMPT_REPORT_NAME).write_text(
        "" if empty_report else json.dumps(report), encoding="utf-8"
    )
    log = root / ATTEMPT_LOG_PATH
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        json.dumps({"record": "attempt", "attempt_id": attempt_id, "config_sha256": digest})
        + "\n",
        encoding="utf-8",
    )
    return directory


def test_a_frozen_model_rebuilds_with_its_full_provenance(tmp_path: Path) -> None:
    _synthetic_tree(tmp_path)
    model, provenance = load_frozen_model(tmp_path, E2C)
    assert provenance["attempt_id"] == E2C
    assert provenance["label"] == "E2c"
    assert provenance["head"] == "smooth_lower_floor_margin_raw_loss"
    assert provenance["head_temperature"] == pytest.approx(1.0e-4)
    assert provenance["head_european_margin"] == pytest.approx(1.0e-4)
    assert provenance["best_epoch"] == 89
    assert provenance["deployment_function"] == DEPLOYMENT_FUNCTION
    assert len(provenance["checkpoint_sha256"]) == 64
    assert "no run-time-recorded digest in the historical ledger" in (
        provenance["checkpoint_digest_provenance"]
    )
    # The strongest identity evidence is functional, and the manifest says so
    # rather than resting on a digest that cannot be compared with anything.
    evidence = provenance["identity_evidence"]
    assert "reproduces the diagonal cell" in evidence["functional"]
    assert evidence["structural_state_dict_strict"] is True
    assert evidence["structural_parameter_count"] == provenance["parameters"]
    assert not model.training


def test_an_empty_attempt_report_is_refused_rather_than_parsed(tmp_path: Path) -> None:
    """The exact failure mode that destroyed E2's evidence."""
    _synthetic_tree(tmp_path, empty_report=True)
    with pytest.raises(FrozenCheckpointError, match="did not survive"):
        load_frozen_model(tmp_path, E2C)


def test_an_unrecorded_checkpoint_is_not_frozen(tmp_path: Path) -> None:
    _synthetic_tree(tmp_path)
    (tmp_path / ATTEMPT_LOG_PATH).write_text("", encoding="utf-8")
    with pytest.raises(FrozenCheckpointError, match="not recorded exactly once"):
        load_frozen_model(tmp_path, E2C)


def test_an_edited_configuration_is_refused(tmp_path: Path) -> None:
    _synthetic_tree(tmp_path)
    path = tmp_path / E2C_CONFIG
    path.write_text(path.read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    with pytest.raises(FrozenCheckpointError, match="immutable"):
        load_frozen_model(tmp_path, E2C)


def test_a_directory_recording_another_attempt_is_refused(tmp_path: Path) -> None:
    _synthetic_tree(tmp_path, attempt_id=E2B)
    with pytest.raises(FrozenCheckpointError, match="records attempt"):
        load_frozen_model(tmp_path, E2C)


@pytest.mark.parametrize(
    ("directory", "error"),
    [
        ("data/american-option-v1", FrozenCheckpointError),
        ("artifacts/../configs", AttemptError),
        ("artifacts/task-9h/interpolation_test", FinalPartitionAccessError),
    ],
)
def test_a_checkpoint_outside_the_ignored_artifact_tree_is_refused(
    tmp_path: Path, directory: str, error: type[Exception]
) -> None:
    with pytest.raises(error):
        load_frozen_model(tmp_path, E2C, directory)


@pytest.mark.parametrize(
    "section",
    [
        None,
        {"feature_mean": [0.0] * 5, "feature_scale": [0.0] * 5, "target_mean": 0.0,
         "target_scale": 1.0},
        {"feature_mean": [0.0] * 5, "feature_scale": [1.0] * 5, "target_mean": 0.0,
         "target_scale": 0.0},
    ],
)
def test_an_unusable_recorded_scaling_fails_closed(section: Any) -> None:
    with pytest.raises(FrozenCheckpointError):
        recorded_scaling({} if section is None else {"scaling": section})
