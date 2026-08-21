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
    ATTEMPT_LOG_SCHEMA,
    ATTEMPT_SCHEMA,
    OUTCOMES,
    AttemptError,
    FinalPartitionAccessError,
    append_attempt,
    assert_path_allowed,
    assert_split_allowed,
    attempt_seeds,
    check_configuration_immutability,
    derive_seed,
    load_toml,
    repository_identity,
    select_rows_by_hash,
    validate_attempt_config,
    validate_attempt_log,
)
from differentiable_pricing.ml.american_dev.representation import representation_arrays
from differentiable_pricing.ml.american_dev.workbench import (
    CRITERION_SECTION,
    WorkbenchError,
    attempt_status,
    build_model,
    compact_summary,
    evaluate_attempt,
    execute_attempt,
    restricted_manifest,
    train_model,
)
from differentiable_pricing.ml.model import fit_scaling

CONTROL_CONFIG = Path("configs/american_dev_attempt_scratch_direct_control_v1.toml")
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
    assert len(configs) == 5
    assert len({config["row_selection"]["salt"] for config in configs}) == 1
    assert len({config["row_selection"]["row_budget"] for config in configs}) == 1
    assert len({config["seeds"]["shuffle_label"] for config in configs}) == 1
    assert len({config["training"]["epochs"] for config in configs}) == 1
    assert len({config["checkpoint"]["rule"] for config in configs}) == 1
    assert len({config["seeds"]["initialization_label"] for config in configs}) == 5


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


def test_the_tracked_attempt_log_is_valid_and_starts_empty() -> None:
    state = validate_attempt_log(Path("docs/attempts/task-9h-attempt-log.jsonl"))
    assert state["header"]["task"] == "task-9h-american-pricer-development"
    assert "none is a project result" in state["header"]["selection_bias"]
    assert state["attempts"] == 0


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
