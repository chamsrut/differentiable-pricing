"""Task 9H workbench: partition guard, identity, immutability, metrics, logging.

Synthetic fixtures throughout. No dataset partition is opened and no numerical
study runs; :func:`execute_attempt` is exercised only along its refusal paths,
which is where its guarantees live.

The properties pinned here are the ones an adaptive loop can quietly lose: that
only ``train`` and ``validation`` are reachable, that an attempt is written once,
that a used configuration cannot be edited afterwards, that two attempts sharing
a salt see identical rows, and that the "works" criterion is Task 9G's rather
than a fresh number.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from differentiable_pricing.ml.american_dev.attempts import (
    ACCEPTANCE_CONFIG_PATH,
    ATTEMPT_LOG_SCHEMA,
    ATTEMPT_SCHEMA,
    OUTCOMES,
    AttemptError,
    FinalPartitionAccessError,
    append_attempt,
    assert_canonical_log_path,
    assert_path_allowed,
    assert_report_is_recordable,
    assert_split_allowed,
    attempt_log_entries,
    attempt_seeds,
    check_configuration_immutability,
    derive_seed,
    load_toml,
    repository_identity,
    select_rows_by_hash,
    source_digests,
    validate_acceptance_config,
    validate_attempt_config,
    validate_attempt_log,
    verify_committed_source,
)
from differentiable_pricing.ml.american_dev.models import parameter_count
from differentiable_pricing.ml.american_dev.representation import (
    AmericanDevPriceModel,
    discounted_spot,
    european_price,
    feature_order,
    representation_arrays,
)
from differentiable_pricing.ml.american_dev.workbench import (
    CRITERION_SECTION,
    WorkbenchError,
    attempt_status,
    build_model,
    compact_summary,
    evaluate_attempt,
    execute_attempt,
    locked_dataset_identity,
    locked_dataset_paths,
    preflight,
    restricted_manifest,
    train_model,
    verify_dataset_identity,
)
from differentiable_pricing.ml.model import fit_scaling

CONTROL_CONFIG = Path("configs/american_dev_attempt_scratch_direct_control_v1.toml")
RESIDUAL_CONFIG = Path("configs/american_dev_attempt_scratch_residual_architecture_v1.toml")
PREMIUM_CONFIG = Path("configs/american_dev_attempt_scratch_american_premium_v1.toml")
RESIDUAL_PREMIUM_CONFIG = Path("configs/american_dev_attempt_scratch_residual_premium_v1.toml")
ACCEPTANCE_CONFIG = Path("configs/american_neural_pilot_acceptance_v1.toml")


def _physical(count: int = 24) -> np.ndarray:
    rng = np.random.default_rng(20260820)
    return np.column_stack(
        (
            rng.choice([1.0, -1.0], size=count),
            rng.uniform(80.0, 120.0, size=count),
            np.full(count, 100.0),
            rng.uniform(0.1, 2.0, size=count),
            np.full(count, 0.05),
            np.full(count, 0.02),
            rng.uniform(0.15, 0.5, size=count),
        )
    )


def _training_config() -> dict[str, Any]:
    return {
        "architecture": {"name": "smooth_mlp", "hidden_dimensions": [8, 8], "activation": "tanh"},
        "head": "direct",
        "conditioning_features": [],
        "seeds": {"initialization_label": "unit-test/init", "shuffle_label": "unit-test/shuffle"},
        "training": {
            "batch_size": 8,
            "epochs": 3,
            "num_threads": 1,
            "deterministic_algorithms": True,
        },
        "optimizer": {"learning_rate": 0.01, "schedule_period_epochs": 3},
    }


def _fitted_model(config: dict[str, Any], physical: np.ndarray) -> tuple[Any, np.ndarray]:
    prices = np.linspace(4.0, 16.0, physical.shape[0])
    features, targets = representation_arrays(physical, prices, ())
    scaling = fit_scaling(features, targets)
    return build_model(config, scaling), targets


def _columns(count: int = 8) -> dict[str, np.ndarray]:
    physical = _physical(count)
    option_type = np.asarray(
        ["call" if value > 0.0 else "put" for value in physical[:, 0]], dtype=object
    )
    spot, strike = physical[:, 1], physical[:, 2]
    intrinsic = np.maximum(
        np.where(physical[:, 0] > 0.0, spot - strike, strike - spot), 0.0
    )
    american = intrinsic + 3.0
    return {
        "sample_id": np.asarray(
            [f"validation-{index:06d}" for index in range(count)], dtype=object
        ),
        "stratum": np.asarray(["core"] * count, dtype=object),
        "option_type": option_type,
        "spot": physical[:, 1],
        "strike": physical[:, 2],
        "maturity": physical[:, 3],
        "rate": physical[:, 4],
        "dividend_yield": physical[:, 5],
        "volatility": physical[:, 6],
        "american_price": american,
        "european_crr_price": american - 0.25,
        "early_exercise_premium": np.full(count, 0.25),
        "intrinsic_value": intrinsic,
        "earliest_exercise_step": np.full(count, -1, dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# Final-partition rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "split",
    ["interpolation_test", "INTERPOLATION_TEST", "boundary_test", "final", "holdout", "test"],
)
def test_only_train_and_validation_are_reachable(split: str) -> None:
    with pytest.raises(FinalPartitionAccessError):
        assert_split_allowed(split)


def test_train_and_validation_are_accepted_and_normalized() -> None:
    assert assert_split_allowed("train") == "train"
    assert assert_split_allowed("Validation") == "validation"


@pytest.mark.parametrize(
    "path",
    [
        "data/american-option-v1/interpolation_test.parquet",
        "artifacts/task-9h/final/summary.json",
        "runs/holdout/attempt.json",
    ],
)
def test_paths_naming_a_final_partition_fail_closed(path: str) -> None:
    with pytest.raises(FinalPartitionAccessError, match="final or held-out partition"):
        assert_path_allowed(path, where="test")


def test_restricted_manifest_drops_every_other_partition() -> None:
    """The final partition's declared identity is never read into a variable."""
    manifest = {
        "schema_version": "american-option-dataset/1",
        "files": [
            {"split": "train", "file": "train.parquet", "sha256": "a" * 64, "rows": 10},
            {"split": "validation", "file": "validation.parquet", "sha256": "b" * 64, "rows": 5},
            {
                "split": "interpolation_test",
                "file": "interpolation_test.parquet",
                "sha256": "c" * 64,
                "rows": 5,
            },
        ],
    }
    restricted = restricted_manifest(manifest)
    assert {entry["split"] for entry in restricted["files"]} == {"train", "validation"}
    assert "c" * 64 not in json.dumps(restricted)


def test_restricted_manifest_requires_both_reachable_partitions() -> None:
    manifest = {"files": [{"split": "train", "file": "t.parquet", "sha256": "a" * 64, "rows": 1}]}
    with pytest.raises(WorkbenchError, match="validation"):
        restricted_manifest(manifest)


def test_the_workbench_exposes_no_final_evaluation_entry_point() -> None:
    import differentiable_pricing.ml.american_dev.workbench as module

    callables = {
        name
        for name in dir(module)
        if not name.startswith("_")
        and callable(getattr(module, name))
        and getattr(getattr(module, name), "__module__", None) == module.__name__
    }
    assert not [name for name in callables if "final" in name.lower()]
    assert "execute_attempt" in callables


# ---------------------------------------------------------------------------
# Deterministic paired row and seed selection
# ---------------------------------------------------------------------------


def test_row_selection_is_paired_across_attempts_sharing_a_salt() -> None:
    ids = [f"train-{index:06d}" for index in range(500)]
    first = select_rows_by_hash(ids, 64, "task-9h-train-subset-v1")
    second = select_rows_by_hash(list(reversed(ids)), 64, "task-9h-train-subset-v1")
    assert sorted(ids[index] for index in first) == sorted(
        list(reversed(ids))[index] for index in second
    )


def test_row_selection_is_nested_in_the_budget_and_changes_with_the_salt() -> None:
    ids = [f"train-{index:06d}" for index in range(500)]
    small = {ids[index] for index in select_rows_by_hash(ids, 16, "salt-a")}
    large = {ids[index] for index in select_rows_by_hash(ids, 64, "salt-a")}
    assert small <= large
    assert {ids[index] for index in select_rows_by_hash(ids, 16, "salt-b")} != small


@pytest.mark.parametrize(("budget", "salt"), [(0, "s"), (501, "s"), (16, "")])
def test_row_selection_refuses_an_impossible_request(budget: int, salt: str) -> None:
    ids = [f"train-{index:06d}" for index in range(500)]
    with pytest.raises(AttemptError):
        select_rows_by_hash(ids, budget, salt)


def test_seeds_are_derived_from_labels_not_written_as_magic_numbers() -> None:
    assert derive_seed("a") == derive_seed("a")
    assert derive_seed("a") != derive_seed("b")
    assert 0 <= derive_seed("a") < 2**32
    with pytest.raises(AttemptError):
        derive_seed("")
    seeds = attempt_seeds(validate_attempt_config(load_toml(CONTROL_CONFIG)))
    assert set(seeds) == {"initialization", "shuffle"}


def test_every_candidate_shares_the_rows_the_shuffle_and_the_budget() -> None:
    """Paired where technically meaningful; only initialization differs."""
    configs = [
        validate_attempt_config(load_toml(path))
        for path in sorted(Path("configs").glob("american_dev_attempt_*.toml"))
    ]
    assert len(configs) == 6
    assert len({config["row_selection"]["salt"] for config in configs}) == 1
    assert len({config["row_selection"]["row_budget"] for config in configs}) == 1
    assert len({config["seeds"]["shuffle_label"] for config in configs}) == 1
    assert len({config["training"]["epochs"] for config in configs}) == 1
    assert len({config["checkpoint"]["rule"] for config in configs}) == 1
    assert len({config["seeds"]["initialization_label"] for config in configs}) == 6


# ---------------------------------------------------------------------------
# Attempt configuration
# ---------------------------------------------------------------------------


def test_the_tracked_control_configuration_validates() -> None:
    config = validate_attempt_config(load_toml(CONTROL_CONFIG))
    assert config["schema_version"] == ATTEMPT_SCHEMA
    assert config["attempt_id"] == "scratch_direct_control_v1"
    assert config["objective"] == "price_only"
    assert config["architecture"]["hidden_dimensions"] == [64, 64, 64]


def test_every_attempt_is_price_only() -> None:
    """Greeks are a separate follow-up stage, not a flag on an attempt."""
    config = validate_attempt_config(load_toml(CONTROL_CONFIG))
    config["objective"] = "derivative_aware"
    with pytest.raises(AttemptError, match="price_only"):
        validate_attempt_config(config)


def test_checkpoints_may_only_be_selected_on_validation() -> None:
    config = validate_attempt_config(load_toml(CONTROL_CONFIG))
    config["checkpoint"] = {**config["checkpoint"], "selection_partition": "train"}
    with pytest.raises(AttemptError, match="selected on validation"):
        validate_attempt_config(config)


def test_outputs_must_live_beneath_the_ignored_trees() -> None:
    config = validate_attempt_config(load_toml(CONTROL_CONFIG))
    config["paths"] = {**config["paths"], "output_directory": "docs/results/x"}
    with pytest.raises(AttemptError, match="ignored artifacts/ tree"):
        validate_attempt_config(config)


def test_every_attempt_must_state_its_hypothesis() -> None:
    config = validate_attempt_config(load_toml(CONTROL_CONFIG))
    config["hypothesis"] = "   "
    with pytest.raises(AttemptError, match="hypothesis"):
        validate_attempt_config(config)


def test_an_unknown_conditioning_feature_is_refused() -> None:
    config = validate_attempt_config(load_toml(CONTROL_CONFIG))
    config["conditioning_features"] = ["implied_vol_ratio"]
    with pytest.raises(AttemptError, match="unknown conditioning feature"):
        validate_attempt_config(config)


# ---------------------------------------------------------------------------
# The residual + premium diagnostic composes existing components only
# ---------------------------------------------------------------------------


def _flatten(section: Any, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in section.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def test_the_residual_premium_diagnostic_changes_only_the_head_and_its_seed() -> None:
    """Against its parent, exactly one behavioral field moves: the head.

    ``attempt_id``, ``parent_attempt``, ``hypothesis`` and the two output paths
    are identity, not behavior, and the initialization label is derived from the
    new attempt ID by the existing rule -- the one confound this composition
    necessarily carries. Everything that determines the training run itself is
    byte-identical to the parent's.
    """
    parent = _flatten(load_toml(RESIDUAL_CONFIG))
    child = _flatten(load_toml(RESIDUAL_PREMIUM_CONFIG))
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
    assert child["head"] == "premium_over_european"
    assert child["parent_attempt"] == "scratch_residual_architecture_v1"
    assert child["seeds.initialization_label"] == "scratch_residual_premium_v1/init"


def test_the_residual_premium_diagnostic_changes_only_the_backbone_from_the_premium_control(
) -> None:
    """Against the small premium arm, only the backbone and identity move."""
    control = _flatten(load_toml(PREMIUM_CONFIG))
    child = _flatten(load_toml(RESIDUAL_PREMIUM_CONFIG))
    behavioral = {
        key
        for key in control | child
        if control.get(key) != child.get(key)
        and not key.startswith(("attempt_id", "parent_attempt", "hypothesis", "paths.", "seeds."))
    }
    assert behavioral == {
        "architecture.name",
        "architecture.width",
        "architecture.blocks",
        "architecture.hidden_dimensions",
    }
    assert child["architecture.name"] == "smooth_residual"
    assert child["architecture.width"] == 128
    assert child["architecture.blocks"] == 6
    assert child["architecture.activation"] == "tanh"


def test_the_residual_premium_diagnostic_uses_the_five_base_features_only() -> None:
    config = validate_attempt_config(load_toml(RESIDUAL_PREMIUM_CONFIG))
    assert config["conditioning_features"] == []
    assert feature_order(()) == (
        "option_type",
        "log_forward_moneyness",
        "total_volatility",
        "rate_time",
        "yield_time",
    )


def test_the_residual_premium_diagnostic_declares_no_normalization_or_dropout() -> None:
    """Neither exists in the architectures; the section may not declare one."""
    config = validate_attempt_config(load_toml(RESIDUAL_PREMIUM_CONFIG))
    assert set(config["architecture"]) == {"name", "width", "blocks", "activation"}
    for unsupported in ("normalization", "dropout"):
        broken = {**config, "architecture": {**config["architecture"], unsupported: "none"}}
        with pytest.raises(AttemptError, match="unsupported key"):
            validate_attempt_config(broken)


def test_the_residual_premium_diagnostic_builds_at_the_parent_capacity() -> None:
    """Composition only: both components already exist and are dispatched."""
    config = validate_attempt_config(load_toml(RESIDUAL_PREMIUM_CONFIG))
    parent = validate_attempt_config(load_toml(RESIDUAL_CONFIG))
    physical = _physical()
    prices = np.linspace(4.0, 16.0, physical.shape[0])
    features, targets = representation_arrays(physical, prices, ())
    scaling = fit_scaling(features, targets)
    model = build_model(config, scaling)
    assert model.head == "premium_over_european"
    assert parameter_count(model.network) == parameter_count(
        build_model(parent, scaling).network
    )
    with torch.no_grad():
        predicted = model(torch.as_tensor(physical, dtype=torch.float64))
        anchor = european_price(torch.as_tensor(physical, dtype=torch.float64))
    # The head's guarantee is non-strict and not bitwise: the A * (E / A)
    # round-trip can leave the price one unit in the last place below the
    # anchor, which is what the tolerance below allows and nothing more.
    assert bool((predicted >= anchor - 8.0 * np.finfo(np.float64).eps * anchor.abs()).all())


@pytest.mark.parametrize(
    "key", ["dataset", "dataset_manifest", "output_directory", "ledger"]
)
def test_the_residual_premium_diagnostic_cannot_be_pointed_at_a_final_partition(
    key: str,
) -> None:
    config = validate_attempt_config(load_toml(RESIDUAL_PREMIUM_CONFIG))
    config["paths"] = {**config["paths"], key: "data/american-option-v1/interpolation_test"}
    with pytest.raises(FinalPartitionAccessError):
        validate_attempt_config(config)


@pytest.mark.parametrize("section", ["row_selection", "checkpoint"])
def test_the_residual_premium_diagnostic_cannot_select_on_a_final_partition(
    section: str,
) -> None:
    config = validate_attempt_config(load_toml(RESIDUAL_PREMIUM_CONFIG))
    key = "partition" if section == "row_selection" else "selection_partition"
    config[section] = {**config[section], key: "interpolation_test"}
    with pytest.raises(FinalPartitionAccessError):
        validate_attempt_config(config)


def test_the_residual_premium_diagnostic_is_not_yet_recorded_as_an_attempt() -> None:
    """A predeclaration is not evidence: no run has happened and none is faked."""
    logged = {
        record["attempt_id"]
        for record in attempt_log_entries(Path("docs/attempts/task-9h-attempt-log.jsonl"))
    }
    assert "scratch_residual_architecture_v1" in logged
    assert "scratch_residual_premium_v1" not in logged


# ---------------------------------------------------------------------------
# The criterion is Task 9G's, reused rather than restated
# ---------------------------------------------------------------------------


def test_every_attempt_points_at_the_task_9g_acceptance_configuration() -> None:
    for path in sorted(Path("configs").glob("american_dev_attempt_*.toml")):
        config = validate_attempt_config(load_toml(path))
        assert config["paths"]["acceptance_config"] == str(ACCEPTANCE_CONFIG)


def test_the_criterion_section_carries_the_task_9g_thresholds() -> None:
    """Fixed before the first attempt, and not revised because one fails."""
    acceptance = load_toml(ACCEPTANCE_CONFIG)
    assert CRITERION_SECTION == "validation_final_entry"
    criterion = acceptance[CRITERION_SECTION]
    assert criterion["normalized_rmse_max"] == 0.003
    assert criterion["normalized_p99_absolute_error_max"] == 0.015
    assert criterion["normalized_maximum_absolute_error_max"] == 0.08
    assert criterion["maximum_material_bound_violations"] == 0
    assert criterion["maximum_material_shape_violations"] == 0


def test_evaluation_reports_price_metrics_bounds_shape_and_a_verdict() -> None:
    config = _training_config()
    physical = _physical()
    model, _ = _fitted_model(config, physical)
    acceptance = load_toml(ACCEPTANCE_CONFIG)
    metrics = evaluate_attempt(model, _columns(), acceptance, batch_size=8)
    overall = metrics["slices"]["overall"]
    assert overall["rows"] == 8
    assert set(overall["normalized"]) >= {
        "mae",
        "rmse",
        "p95_absolute_error",
        "p99_absolute_error",
        "maximum_absolute_error",
    }
    diagnostics = metrics["diagnostics"]
    assert "material_bound_violations" in diagnostics
    assert "material_shape_violations" in diagnostics
    assert set(diagnostics["checks"]) >= {
        "intrinsic_lower_bound",
        "european_comparator_lower_bound",
        "spot_monotonicity",
        "spot_convexity",
        "volatility_monotonicity",
    }
    assert set(metrics["gate"]["checks"]) == {
        "normalized_rmse",
        "normalized_p99_absolute_error",
        "normalized_maximum_absolute_error",
        "material_bound_violations",
        "material_shape_violations",
    }
    assert isinstance(metrics["gate"]["passed"], bool)
    assert metrics["european_crr_baseline"]["normalized"]["rmse"] > 0.0


def test_a_wildly_wrong_model_fails_the_criterion() -> None:
    """The verdict discriminates; it is not vacuously true."""
    config = _training_config()
    physical = _physical()
    model, _ = _fitted_model(config, physical)
    with torch.no_grad():
        for parameter in model.network.parameters():
            parameter.fill_(0.0)
        final = [layer for layer in model.network.layers if isinstance(layer, torch.nn.Linear)][-1]
        final.bias.fill_(5.0)
    acceptance = load_toml(ACCEPTANCE_CONFIG)
    metrics = evaluate_attempt(model, _columns(), acceptance, batch_size=8)
    assert metrics["gate"]["passed"] is False


# ---------------------------------------------------------------------------
# Clean tree and attempt lifecycle
# ---------------------------------------------------------------------------


def _git(repository: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=repository, check=True, capture_output=True)


def _repository(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "config", "user.email", "test@example.invalid")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "tracked.txt").write_text("one\n", encoding="utf-8")
    _git(tmp_path, "add", "tracked.txt")
    _git(tmp_path, "commit", "--quiet", "-m", "initial")
    return tmp_path


def test_repository_identity_requires_a_clean_tracked_worktree(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    identity = repository_identity(repository)
    assert identity["tracked_worktree_clean"] is True
    assert len(identity["commit"]) == 40

    (repository / "tracked.txt").write_text("two\n", encoding="utf-8")
    with pytest.raises(AttemptError, match="clean tracked worktree"):
        repository_identity(repository)


def test_an_untracked_file_does_not_block_an_attempt(tmp_path: Path) -> None:
    """Only *tracked* modifications matter: ignored outputs are expected."""
    repository = _repository(tmp_path)
    (repository / "scratch.log").write_text("noise\n", encoding="utf-8")
    assert repository_identity(repository)["tracked_worktree_clean"] is True


def test_an_existing_attempt_is_never_overwritten(tmp_path: Path) -> None:
    config_path = tmp_path / "attempt.toml"
    config_path.write_text(CONTROL_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "artifacts/task-9h/scratch_direct_control_v1").mkdir(parents=True)
    with pytest.raises(WorkbenchError, match="never overwritten"):
        execute_attempt(config_path, tmp_path)


def test_a_failed_attempt_may_not_be_silently_rerun_under_the_same_id(
    tmp_path: Path,
) -> None:
    """Deleting the outputs is not the remedy; a retry gets a new attempt ID.

    The ledger alone is enough to refuse, so removing the artifacts directory
    after a crash still does not free the ID.
    """
    config_path = tmp_path / "attempt.toml"
    config_path.write_text(CONTROL_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    ledger = tmp_path / "runs/task-9h/scratch_direct_control_v1/attempt.json"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps({"attempt_id": "scratch_direct_control_v1", "status": "failed"}),
        encoding="utf-8",
    )
    assert not (tmp_path / "artifacts").exists()
    with pytest.raises(WorkbenchError, match="never overwritten"):
        execute_attempt(config_path, tmp_path)
    assert attempt_status(config_path, tmp_path)["status"] == "failed"


def test_status_of_an_unstarted_attempt_opens_nothing(tmp_path: Path) -> None:
    config_path = tmp_path / "attempt.toml"
    config_path.write_text(CONTROL_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    assert attempt_status(config_path, tmp_path) == {
        "attempt_id": "scratch_direct_control_v1",
        "status": "not_started",
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def test_training_selects_a_checkpoint_on_validation_and_is_reproducible() -> None:
    """Two runs of the same attempt agree; a different seed label does not.

    The second half matters as much as the first: a reproducibility assertion
    that would also pass for two *different* seeds is measuring nothing.
    """
    config = _training_config()
    physical = _physical()
    first_model, targets = _fitted_model(config, physical)
    first = train_model(first_model, config, physical, targets, physical, targets)
    assert 1 <= first["best_epoch"] <= first["epochs"]
    assert len(first["history"]) == first["epochs"]

    second_model, _ = _fitted_model(config, physical)
    second = train_model(second_model, config, physical, targets, physical, targets)
    assert first["best_validation_standardized_target_mse"] == pytest.approx(
        second["best_validation_standardized_target_mse"], rel=1e-12, abs=1e-15
    )

    other = _training_config()
    other["seeds"] = {**other["seeds"], "initialization_label": "unit-test/other"}
    other_model, _ = _fitted_model(other, physical)
    different = train_model(other_model, other, physical, targets, physical, targets)
    assert different["best_validation_standardized_target_mse"] != pytest.approx(
        first["best_validation_standardized_target_mse"], rel=1e-12, abs=1e-15
    )


# ---------------------------------------------------------------------------
# Append-only attempt log
# ---------------------------------------------------------------------------


def _record(
    attempt_id: str, parent: str = "", outcome: str = "criterion_not_met"
) -> dict[str, Any]:
    return {
        "attempt_id": attempt_id,
        "parent_attempt": parent,
        "hypothesis": "a hypothesis",
        "git_commit": "0" * 40,
        "config_path": "configs/american_dev_attempt_scratch_direct_control_v1.toml",
        "config_sha256": "1" * 64,
        "source_digests": {},
        "architecture": {},
        "features": [],
        "target_and_reconstruction": {},
        "seeds": {},
        "selected_rows": {},
        "optimizer_and_budget": {},
        "price_metrics": {},
        "bound_and_shape_diagnostics": {},
        "criterion": {},
        "outcome": outcome,
        "interpretation": "prices did not reach the criterion",
        "next_action": "continue with more capacity",
    }


def test_the_log_is_created_with_its_schema_header(tmp_path: Path) -> None:
    log = tmp_path / "attempts.jsonl"
    state = append_attempt(log, _record("first"))
    assert state["attempts"] == 1
    assert state["header"]["schema_version"] == ATTEMPT_LOG_SCHEMA
    assert state["header"]["append_only"] is True


def test_an_existing_attempt_entry_is_never_rewritten(tmp_path: Path) -> None:
    log = tmp_path / "attempts.jsonl"
    append_attempt(log, _record("first"))
    with pytest.raises(AttemptError, match="already recorded"):
        append_attempt(log, _record("first"))


def test_a_failed_attempt_is_recorded_like_any_other(tmp_path: Path) -> None:
    """The recorded search is the deliverable, so dead ends are first class."""
    log = tmp_path / "attempts.jsonl"
    for outcome in OUTCOMES:
        append_attempt(log, _record(f"attempt_{outcome}", outcome=outcome))
    assert validate_attempt_log(log)["attempts"] == len(OUTCOMES)


def test_an_unknown_outcome_is_refused(tmp_path: Path) -> None:
    log = tmp_path / "attempts.jsonl"
    with pytest.raises(AttemptError, match="outcome must be one of"):
        append_attempt(log, _record("first", outcome="looked_promising"))


def test_a_child_attempt_must_name_an_earlier_parent(tmp_path: Path) -> None:
    log = tmp_path / "attempts.jsonl"
    append_attempt(log, _record("first"))
    append_attempt(log, _record("second", parent="first"))
    with pytest.raises(AttemptError, match="not an earlier entry"):
        append_attempt(log, _record("third", parent="never_ran"))


def test_an_incomplete_record_is_refused(tmp_path: Path) -> None:
    log = tmp_path / "attempts.jsonl"
    record = _record("first")
    del record["interpretation"]
    with pytest.raises(AttemptError, match="missing field"):
        append_attempt(log, record)


def test_earlier_entries_survive_verbatim_when_a_later_one_is_added(tmp_path: Path) -> None:
    log = tmp_path / "attempts.jsonl"
    append_attempt(log, _record("failed_one"))
    first_line = log.read_text(encoding="utf-8").splitlines()[1]
    append_attempt(log, _record("second", parent="failed_one"))
    assert log.read_text(encoding="utf-8").splitlines()[1] == first_line


def test_a_log_without_its_header_is_invalid(tmp_path: Path) -> None:
    log = tmp_path / "attempts.jsonl"
    log.write_text(json.dumps({"record": "attempt"}) + "\n", encoding="utf-8")
    with pytest.raises(AttemptError, match="schema header"):
        validate_attempt_log(log)


def test_the_tracked_attempt_log_is_valid_and_keeps_every_recorded_attempt() -> None:
    state = validate_attempt_log(Path("docs/attempts/task-9h-attempt-log.jsonl"))
    assert state["header"]["task"] == "task-9h-american-pricer-development"
    assert "none is a project result" in state["header"]["selection_bias"]
    assert "scratch_direct_control_v1" in state["attempt_ids"]
    assert state["attempts"] == len(state["attempt_ids"])


# ---------------------------------------------------------------------------
# Configuration immutability
# ---------------------------------------------------------------------------


def test_a_used_configuration_may_not_be_edited_afterwards(tmp_path: Path) -> None:
    import hashlib

    configuration = tmp_path / "configs/american_dev_attempt_x.toml"
    configuration.parent.mkdir(parents=True)
    configuration.write_text('attempt_id = "x"\n', encoding="utf-8")
    digest = hashlib.sha256(configuration.read_bytes()).hexdigest()
    log = tmp_path / "attempts.jsonl"
    record = _record("x")
    record["config_path"] = "configs/american_dev_attempt_x.toml"
    record["config_sha256"] = digest
    append_attempt(log, record)
    assert check_configuration_immutability(tmp_path, log) == []

    configuration.write_text('attempt_id = "x"\nedited = true\n', encoding="utf-8")
    failures = check_configuration_immutability(tmp_path, log)
    assert failures and "immutable after use" in failures[0]


def test_a_deleted_used_configuration_is_reported(tmp_path: Path) -> None:
    log = tmp_path / "attempts.jsonl"
    record = _record("x")
    record["config_path"] = "configs/gone.toml"
    append_attempt(log, record)
    failures = check_configuration_immutability(tmp_path, log)
    assert failures and "no longer exists" in failures[0]


# ---------------------------------------------------------------------------
# The compact, agent-readable summary
# ---------------------------------------------------------------------------


def test_the_compact_summary_carries_a_verdict_and_the_worst_slices() -> None:
    report = {
        "attempt_id": "unit_test_attempt",
        "parent_attempt": "",
        "hypothesis": "a hypothesis",
        "repository": {"commit": "0" * 40},
        "config_path": "configs/american_dev_attempt_scratch_direct_control_v1.toml",
        "config_sha256": "1" * 64,
        "architecture": {"name": "smooth_mlp"},
        "seeds": {"initialization": 1, "shuffle": 2},
        "selected_rows": {"count": 32768},
        "training": {"best_epoch": 42},
        "criterion": {"section": "validation_final_entry"},
        "price_metrics": {
            "slices": {
                "overall": {
                    "rows": 100,
                    "normalized": {"rmse": 0.01, "maximum_absolute_error": 0.2},
                },
                "option_type:put": {
                    "rows": 40,
                    "normalized": {"rmse": 0.05, "maximum_absolute_error": 0.3},
                },
                "expiry:empty": {"rows": 0, "normalized": None},
            },
            "diagnostics": {"material_bound_violations": 3, "material_shape_violations": 0},
            "gate": {"checks": {"normalized_rmse": False}, "passed": False},
            "european_crr_baseline": {"normalized": {"rmse": 0.02}},
        },
    }
    summary = compact_summary(report)
    assert summary["attempt_id"] == "unit_test_attempt"
    assert summary["gate"]["passed"] is False
    assert summary["material_bound_violations"] == 3
    assert summary["worst_slices"][0]["slice"] == "option_type:put"
    assert all(entry["rows"] for entry in summary["worst_slices"])
    assert "not a project result" in summary["selection_bias"]
    assert len(json.dumps(summary)) < 8192


# ---------------------------------------------------------------------------
# Committed-source verification
# ---------------------------------------------------------------------------


def test_committed_source_accepts_a_tracked_unmodified_file(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    identities = verify_committed_source(repository, ["tracked.txt"])
    assert set(identities) == {"tracked.txt"}
    assert len(identities["tracked.txt"]) == 40


def test_committed_source_refuses_a_file_that_is_not_tracked_at_head(tmp_path: Path) -> None:
    """A brand-new file passes `git status --untracked-files=no` and must not."""
    repository = _repository(tmp_path)
    (repository / "new_module.py").write_text("x = 1\n", encoding="utf-8")
    assert repository_identity(repository)["tracked_worktree_clean"] is True
    with pytest.raises(AttemptError, match="not tracked at HEAD"):
        verify_committed_source(repository, ["new_module.py"])


def test_committed_source_refuses_a_staged_but_uncommitted_file(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    (repository / "staged.py").write_text("x = 1\n", encoding="utf-8")
    _git(repository, "add", "staged.py")
    with pytest.raises(AttemptError, match="not tracked at HEAD"):
        verify_committed_source(repository, ["staged.py"])


def test_committed_source_refuses_a_tracked_file_that_differs_from_head(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    (repository / "tracked.txt").write_text("edited\n", encoding="utf-8")
    with pytest.raises(AttemptError, match="differs from its HEAD blob"):
        verify_committed_source(repository, ["tracked.txt"])


def test_committed_source_ignores_the_rest_of_the_workspace(tmp_path: Path) -> None:
    """Ignored artifacts and runs are expected; only the named files are pinned."""
    repository = _repository(tmp_path)
    (repository / "artifacts").mkdir()
    (repository / "artifacts/checkpoint.pt").write_text("weights\n", encoding="utf-8")
    (repository / "runs").mkdir()
    (repository / "runs/attempt.json").write_text("{}\n", encoding="utf-8")
    assert verify_committed_source(repository, ["tracked.txt"])


def test_committed_source_refuses_a_name_that_escapes_the_repository(tmp_path: Path) -> None:
    repository = _repository(tmp_path)
    with pytest.raises(AttemptError, match="relative path inside the repository"):
        verify_committed_source(repository, ["../outside.py"])


def test_source_digests_pin_the_criterion_and_the_locked_protocol() -> None:
    """Files that change what an attempt *means* are pinned, not just its code."""
    digests = source_digests(Path("."))
    assert ACCEPTANCE_CONFIG_PATH in digests
    assert "configs/american_neural_pilot_protocol_v1.toml" in digests
    assert "python/src/differentiable_pricing/ml/american_dev/workbench.py" in digests
    assert all(len(value) == 64 for value in digests.values())


def test_every_pinned_source_file_is_committed_and_unmodified() -> None:
    """The repository itself satisfies the rule an attempt would be held to.

    Conditional by construction: mid-change the tracked tree is dirty and an
    attempt could not start either, which is the point. It is asserted wherever
    the tree is clean — CI, and any checkout an attempt would actually run from.
    """
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        pytest.skip("tracked worktree is modified; an attempt could not start here either")
    digests = source_digests(Path("."))
    assert set(verify_committed_source(Path("."), digests)) == set(digests)


# ---------------------------------------------------------------------------
# Partition identity and manifest path enforcement
# ---------------------------------------------------------------------------


def _manifest(**overrides: Any) -> dict[str, Any]:
    manifest = {
        "schema_version": "american-option-dataset/1",
        "files": [
            {"split": "train", "file": "train.parquet", "sha256": "a" * 64, "rows": 10},
            {"split": "validation", "file": "validation.parquet", "sha256": "b" * 64, "rows": 5},
        ],
    }
    manifest.update(overrides)
    return manifest


@pytest.mark.parametrize(
    "name",
    [
        "../interpolation_test.parquet",
        "subdir/../../holdout.parquet",
        "/absolute/train.parquet",
    ],
    ids=["parent-escape", "escape-through-subdir", "absolute"],
)
def test_a_manifest_entry_may_not_escape_the_dataset_directory(name: str) -> None:
    """The manifest is data: its file name is guarded before anything opens it."""
    manifest = _manifest(
        files=[
            {"split": "train", "file": name, "sha256": "a" * 64, "rows": 10},
            {"split": "validation", "file": "validation.parquet", "sha256": "b" * 64, "rows": 5},
        ]
    )
    with pytest.raises(AttemptError):
        restricted_manifest(manifest)


def test_a_retained_manifest_entry_naming_a_final_partition_fails_closed() -> None:
    """A `train` entry pointing at the final partition is refused, not opened."""
    manifest = _manifest(
        files=[
            {"split": "train", "file": "interpolation_test.parquet", "sha256": "a" * 64, "rows": 1},
            {"split": "validation", "file": "validation.parquet", "sha256": "b" * 64, "rows": 5},
        ]
    )
    with pytest.raises(FinalPartitionAccessError, match="final or held-out partition"):
        restricted_manifest(manifest)


@pytest.mark.parametrize(
    "entry",
    [
        {"split": "train", "rows": 10, "sha256": "a" * 64},
        {"split": "train", "file": "   ", "sha256": "a" * 64, "rows": 10},
        {"split": "train", "file": "train.parquet", "rows": 10},
        {"split": "train", "file": "train.parquet", "sha256": "short", "rows": 10},
    ],
    ids=["no-file", "blank-file", "no-digest", "malformed-digest"],
)
def test_a_retained_manifest_entry_must_declare_a_name_and_a_digest(entry: dict) -> None:
    manifest = _manifest(
        files=[
            entry,
            {"split": "validation", "file": "validation.parquet", "sha256": "b" * 64, "rows": 5},
        ]
    )
    with pytest.raises(WorkbenchError):
        restricted_manifest(manifest)


def test_a_manifest_declaring_a_partition_twice_is_refused() -> None:
    manifest = _manifest(
        files=[
            {"split": "train", "file": "train.parquet", "sha256": "a" * 64, "rows": 10},
            {"split": "train", "file": "other.parquet", "sha256": "c" * 64, "rows": 10},
            {"split": "validation", "file": "validation.parquet", "sha256": "b" * 64, "rows": 5},
        ]
    )
    with pytest.raises(WorkbenchError, match="twice"):
        restricted_manifest(manifest)


def test_the_locked_identity_is_read_from_the_task_9g_protocol() -> None:
    protocol = load_toml(Path("configs/american_neural_pilot_protocol_v1.toml"))
    identity = locked_dataset_identity(protocol)
    assert set(identity) == {"manifest_sha256", "train_sha256", "validation_sha256"}
    assert all(len(value) == 64 for value in identity.values())
    dataset, manifest = locked_dataset_paths(protocol)
    assert dataset == "data/american-option-v1"
    assert manifest == "data/american-option-v1/manifest.json"


def test_the_locked_identity_never_carries_another_partitions_digest() -> None:
    """Whitelisted by key: no other partition's declared identity is read out."""
    protocol = load_toml(Path("configs/american_neural_pilot_protocol_v1.toml"))
    identity = locked_dataset_identity(protocol)
    other = {
        str(value)
        for key, value in protocol["dataset"].items()
        if key not in identity and isinstance(value, str)
    }
    assert other
    assert not other & set(identity.values())


@pytest.mark.parametrize("missing", ["manifest_sha256", "train_sha256", "validation_sha256"])
def test_a_protocol_without_a_locked_identity_is_refused(missing: str) -> None:
    protocol = load_toml(Path("configs/american_neural_pilot_protocol_v1.toml"))
    dataset = {key: value for key, value in protocol["dataset"].items() if key != missing}
    with pytest.raises(WorkbenchError, match=missing):
        locked_dataset_identity({**protocol, "dataset": dataset})


def _dataset_on_disk(tmp_path: Path) -> tuple[Path, Path, dict[str, Any], dict[str, str]]:
    dataset = tmp_path / "data/american-option-v1"
    dataset.mkdir(parents=True)
    digests = {}
    for split in ("train", "validation"):
        path = dataset / f"{split}.parquet"
        path.write_text(f"{split} rows\n", encoding="utf-8")
        digests[split] = _sha256(path)
    manifest = {
        "schema_version": "american-option-dataset/1",
        "files": [
            {"split": split, "file": f"{split}.parquet", "sha256": digests[split], "rows": 1}
            for split in ("train", "validation")
        ],
    }
    manifest_path = dataset / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    locked = {
        "manifest_sha256": _sha256(manifest_path),
        "train_sha256": digests["train"],
        "validation_sha256": digests["validation"],
    }
    return dataset, manifest_path, restricted_manifest(manifest), locked


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dataset_identity_matching_the_locked_one_is_accepted(tmp_path: Path) -> None:
    dataset, manifest_path, manifest, locked = _dataset_on_disk(tmp_path)
    digests = verify_dataset_identity(dataset, manifest_path, manifest, locked)
    assert set(digests) == {"manifest", "train", "validation"}
    assert digests["train"] == locked["train_sha256"]


@pytest.mark.parametrize("key", ["manifest_sha256", "train_sha256", "validation_sha256"])
def test_a_dataset_that_is_not_the_locked_one_fails_closed(tmp_path: Path, key: str) -> None:
    """Otherwise an attempt could be compared against a control it never shared."""
    dataset, manifest_path, manifest, locked = _dataset_on_disk(tmp_path)
    with pytest.raises(WorkbenchError, match="Task 9G locked"):
        verify_dataset_identity(dataset, manifest_path, manifest, {**locked, key: "d" * 64})


def test_a_partition_that_contradicts_its_own_manifest_entry_fails_closed(
    tmp_path: Path,
) -> None:
    dataset, manifest_path, manifest, locked = _dataset_on_disk(tmp_path)
    (dataset / "train.parquet").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(WorkbenchError, match="manifest entry declares"):
        verify_dataset_identity(dataset, manifest_path, manifest, locked)


def test_the_workbench_reuses_the_task_9g_row_level_policy_verification() -> None:
    """N5: the admitted label policy is checked per row, by Task 9G's own code."""
    import ast
    import inspect

    import differentiable_pricing.ml.american_dev.workbench as module

    source = Path(inspect.getfile(module)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "american_pilot" and node.level == 2
        for alias in node.names
    }
    assert "verify_partition_policy" in imported
    executor = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "execute_attempt"
    )
    called = {
        node.func.id
        for node in ast.walk(executor)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "verify_partition_policy" in called
    assert "verify_dataset_identity" in called


# ---------------------------------------------------------------------------
# Pre-flight: everything checkable before an attempt is reserved
# ---------------------------------------------------------------------------


def test_preflight_refuses_a_configuration_outside_the_repository(tmp_path: Path) -> None:
    config_path = tmp_path / "attempt.toml"
    config_path.write_text(CONTROL_CONFIG.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(WorkbenchError, match="outside the repository root"):
        preflight(config_path, tmp_path / "elsewhere")


def test_preflight_reserves_nothing_when_it_refuses(tmp_path: Path) -> None:
    """A bad attempt leaves its ID unused: no directory, no ledger, no reuse."""
    config_path = tmp_path / "attempt.toml"
    broken = CONTROL_CONFIG.read_text(encoding="utf-8").replace(
        'rule = "lowest SHA-256(salt + NUL + sample_id), sample_id tie-break"',
        'rule = "first 32768 rows"',
    )
    config_path.write_text(broken, encoding="utf-8")
    with pytest.raises(AttemptError, match=r"row_selection\.rule"):
        execute_attempt(config_path, tmp_path)
    assert not (tmp_path / "artifacts").exists()
    assert not (tmp_path / "runs").exists()


def test_the_acceptance_configuration_must_carry_the_reused_criterion() -> None:
    acceptance = load_toml(ACCEPTANCE_CONFIG)
    section = validate_acceptance_config(acceptance)
    assert section["normalized_rmse_max"] == 0.003

    with pytest.raises(AttemptError, match="schema"):
        validate_acceptance_config({**acceptance, "schema_version": "something-else/1"})
    without_section = {key: value for key, value in acceptance.items() if key != CRITERION_SECTION}
    with pytest.raises(AttemptError, match=CRITERION_SECTION):
        validate_acceptance_config(without_section)
    thinned = {
        key: value
        for key, value in acceptance[CRITERION_SECTION].items()
        if key != "normalized_rmse_max"
    }
    with pytest.raises(AttemptError, match="missing threshold"):
        validate_acceptance_config({**acceptance, CRITERION_SECTION: thinned})


def test_an_attempt_may_not_point_at_another_acceptance_configuration() -> None:
    config = validate_attempt_config(load_toml(CONTROL_CONFIG))
    config["paths"] = {**config["paths"], "acceptance_config": "configs/looser_criterion.toml"}
    with pytest.raises(AttemptError, match=r"paths\.acceptance_config must be"):
        validate_attempt_config(config)


# ---------------------------------------------------------------------------
# Declared fields are validated against implemented behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        (None, "representation", "american_raw_v1"),
        (None, "target", "V"),
        (None, "physical_reconstruction", "V = u"),
        ("optimizer", "name", "sgd"),
        ("optimizer", "schedule", "step"),
        ("checkpoint", "metric", "physical_price_rmse"),
        ("checkpoint", "rule", "last epoch"),
        ("training", "shuffle", "numpy default_rng once per epoch"),
        ("row_selection", "rule", "first N rows"),
        ("seeds", "derivation", "a magic number"),
    ],
    ids=lambda value: str(value),
)
def test_an_unsupported_declared_value_is_refused(
    section: str | None, key: str, value: str
) -> None:
    """Declared but not dispatched is the failure mode: the log would lie."""
    config = validate_attempt_config(load_toml(CONTROL_CONFIG))
    if section is None:
        config[key] = value
    else:
        config[section] = {**config[section], key: value}
    with pytest.raises(AttemptError, match=key):
        validate_attempt_config(config)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        (None, "gradient_weight"),
        ("optimizer", "momentum"),
        ("training", "early_stopping_patience"),
        ("checkpoint", "smoothing"),
        ("evaluation", "greeks"),
        ("paths", "interpolation_manifest"),
        ("seeds", "dropout_label"),
        ("row_selection", "stratify_by"),
        ("architecture", "dropout"),
    ],
    ids=lambda value: str(value),
)
def test_an_unknown_declared_key_is_refused(section: str | None, key: str) -> None:
    """An ignored declaration would still be recorded in the attempt log."""
    config = validate_attempt_config(load_toml(CONTROL_CONFIG))
    if section is None:
        config[key] = "something"
    else:
        config[section] = {**config[section], key: "something"}
    with pytest.raises(AttemptError, match="unsupported key"):
        validate_attempt_config(config)


def test_a_missing_declared_key_is_refused() -> None:
    config = validate_attempt_config(load_toml(CONTROL_CONFIG))
    config["optimizer"] = {
        key: value for key, value in config["optimizer"].items() if key != "weight_decay"
    }
    with pytest.raises(AttemptError, match="missing key"):
        validate_attempt_config(config)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("learning_rate", 0.0),
        ("epsilon", 0.0),
        ("beta1", 1.0),
        ("beta2", -0.1),
        ("weight_decay", -1.0),
        ("schedule_period_epochs", 0),
    ],
    ids=lambda value: str(value),
)
def test_an_out_of_range_optimizer_value_is_refused(key: str, value: float) -> None:
    config = validate_attempt_config(load_toml(CONTROL_CONFIG))
    config["optimizer"] = {**config["optimizer"], key: value}
    with pytest.raises(AttemptError, match=key):
        validate_attempt_config(config)


def test_every_tracked_configuration_declares_only_dispatched_behavior() -> None:
    """Every immutable attempt survives the stricter rules unchanged."""
    paths = sorted(Path("configs").glob("american_dev_attempt_*.toml"))
    assert len(paths) == 6
    for path in paths:
        config = validate_attempt_config(load_toml(path))
        assert config["optimizer"]["name"] == "adamw"
        assert config["optimizer"]["schedule"] == "cosine_annealing"
        assert config["checkpoint"]["metric"] == "standardized_target_mse"
        assert config["paths"]["acceptance_config"] == ACCEPTANCE_CONFIG_PATH


# ---------------------------------------------------------------------------
# The canonical log and the recordable report
# ---------------------------------------------------------------------------


def test_the_attempt_log_path_is_canonical(tmp_path: Path) -> None:
    canonical = tmp_path / "docs/attempts/task-9h-attempt-log.jsonl"
    assert assert_canonical_log_path(canonical, tmp_path) == canonical.resolve()
    with pytest.raises(AttemptError, match="append-only file"):
        assert_canonical_log_path(tmp_path / "other.jsonl", tmp_path)


def test_a_report_is_recordable_only_with_its_schema_and_criterion() -> None:
    report = {
        "schema_version": "american-dev-attempt-report/1",
        "criterion": {"source": ACCEPTANCE_CONFIG_PATH, "section": CRITERION_SECTION},
    }
    assert assert_report_is_recordable(report) is None
    with pytest.raises(AttemptError, match="report schema"):
        assert_report_is_recordable({**report, "schema_version": "american-dev-attempt-report/2"})
    with pytest.raises(AttemptError, match="criterion must come from"):
        assert_report_is_recordable({**report, "criterion": {"source": "configs/other.toml"}})
    with pytest.raises(AttemptError, match="section"):
        assert_report_is_recordable(
            {
                **report,
                "criterion": {"source": ACCEPTANCE_CONFIG_PATH, "section": "final_accuracy"},
            }
        )


# ---------------------------------------------------------------------------
# The premium head's guarantee, as it actually holds
# ---------------------------------------------------------------------------


def test_the_premium_head_is_bounded_below_by_the_european_anchor_up_to_rounding() -> None:
    """Non-strict, and *not* bitwise: the A*(E/A) round-trip costs about an ulp.

    The claim under test is the corrected one. A price may sit one unit in the
    last place below its analytic European anchor after reconstruction, which is
    immaterial against a 3e-3 normalized criterion but is not the exact bound the
    documentation used to assert.
    """
    physical = _physical(256)
    prices = np.linspace(4.0, 16.0, physical.shape[0])
    features, targets = representation_arrays(physical, prices, ())
    scaling = fit_scaling(features, targets)
    config = {**_training_config(), "head": "premium_over_european"}
    network = build_model(config, scaling).network
    model = AmericanDevPriceModel(network, scaling, head="premium_over_european")
    tensor = torch.as_tensor(physical, dtype=torch.float64)
    with torch.no_grad():
        # Drive softplus into underflow, the case the guarantee is weakest in.
        for parameter in model.network.parameters():
            parameter.fill_(0.0)
        last = [layer for layer in model.network.layers if isinstance(layer, torch.nn.Linear)][-1]
        last.bias.fill_(-1000.0)
        price = model(tensor)
        anchor = european_price(tensor)
        scale = discounted_spot(tensor)
    shortfall = (anchor - price).numpy()
    tolerance = 4.0 * np.spacing(np.abs(anchor.numpy()))
    assert bool((shortfall <= tolerance).all())
    assert bool((shortfall / scale.numpy() <= 1.0e-12).all())
