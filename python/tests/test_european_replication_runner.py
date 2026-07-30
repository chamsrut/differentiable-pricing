"""Lifecycle and gate tests for the one-shot European replication runner."""

from __future__ import annotations

import importlib.util
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "run_european_replication.py"
ACCEPTANCE = PROJECT_ROOT / "configs" / "european_neural_acceptance_v1.toml"


def load_runner() -> Any:
    spec = importlib.util.spec_from_file_location("dp_run_european_replication", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = load_runner()


@pytest.fixture
def acceptance() -> dict[str, Any]:
    with ACCEPTANCE.open("rb") as stream:
        return tomllib.load(stream)


@pytest.fixture
def passing_report(acceptance: dict[str, Any]) -> dict[str, Any]:
    overall = acceptance["overall"]
    tail = acceptance["tail"]
    return {
        "slices": {
            "overall": {
                "metrics": {
                    "price_over_spot": {
                        "rmse": overall["price_over_spot_rmse_max"],
                        "p99_absolute_error": overall[
                            "price_over_spot_p99_absolute_error_max"
                        ],
                    },
                    "delta": {"rmse": overall["delta_rmse_max"]},
                    "gamma": {"rmse": overall["gamma_rmse_max"]},
                    "vega": {"rmse": overall["vega_rmse_max"]},
                    "theta": {"rmse": overall["theta_rmse_max"]},
                    "rho": {"rmse": overall["rho_rmse_max"]},
                }
            },
            "moneyness_band": {
                "tail": {
                    "metrics": {
                        "price_over_spot": {
                            "rmse": tail["price_over_spot_rmse_max"]
                        },
                        "delta": {"rmse": tail["delta_rmse_max"]},
                        "gamma": {"rmse": tail["gamma_rmse_max"]},
                    }
                }
            },
        },
        "learned_price_no_arbitrage": {
            "material_relative_tolerance": acceptance["no_arbitrage"][
                "material_relative_tolerance"
            ],
            "material_violations": acceptance["no_arbitrage"][
                "material_violations_max"
            ],
        },
    }


def nested_set(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    current = payload
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_every_frozen_gate_passes_at_its_limit(
    passing_report: dict[str, Any],
    acceptance: dict[str, Any],
) -> None:
    result = runner.assess_acceptance(passing_report, acceptance)

    assert result["passed"] is True
    assert len(result["checks"]) == 11
    assert all(check["passed"] for check in result["checks"])


@pytest.mark.parametrize(
    ("report_path", "gate_path"),
    [
        (
            ("slices", "overall", "metrics", "price_over_spot", "rmse"),
            ("overall", "price_over_spot_rmse_max"),
        ),
        (
            (
                "slices",
                "overall",
                "metrics",
                "price_over_spot",
                "p99_absolute_error",
            ),
            ("overall", "price_over_spot_p99_absolute_error_max"),
        ),
        (
            ("slices", "overall", "metrics", "delta", "rmse"),
            ("overall", "delta_rmse_max"),
        ),
        (
            ("slices", "overall", "metrics", "gamma", "rmse"),
            ("overall", "gamma_rmse_max"),
        ),
        (
            ("slices", "overall", "metrics", "vega", "rmse"),
            ("overall", "vega_rmse_max"),
        ),
        (
            ("slices", "overall", "metrics", "theta", "rmse"),
            ("overall", "theta_rmse_max"),
        ),
        (
            ("slices", "overall", "metrics", "rho", "rmse"),
            ("overall", "rho_rmse_max"),
        ),
        (
            (
                "slices",
                "moneyness_band",
                "tail",
                "metrics",
                "price_over_spot",
                "rmse",
            ),
            ("tail", "price_over_spot_rmse_max"),
        ),
        (
            (
                "slices",
                "moneyness_band",
                "tail",
                "metrics",
                "delta",
                "rmse",
            ),
            ("tail", "delta_rmse_max"),
        ),
        (
            (
                "slices",
                "moneyness_band",
                "tail",
                "metrics",
                "gamma",
                "rmse",
            ),
            ("tail", "gamma_rmse_max"),
        ),
    ],
)
def test_each_metric_gate_can_fail_independently(
    passing_report: dict[str, Any],
    acceptance: dict[str, Any],
    report_path: tuple[str, ...],
    gate_path: tuple[str, ...],
) -> None:
    limit = runner._nested(acceptance, gate_path)
    nested_set(passing_report, report_path, limit * 1.01)

    result = runner.assess_acceptance(passing_report, acceptance)

    assert result["passed"] is False
    failed = [check for check in result["checks"] if not check["passed"]]
    assert len(failed) == 1


def test_material_arbitrage_gate_can_fail_independently(
    passing_report: dict[str, Any],
    acceptance: dict[str, Any],
) -> None:
    passing_report["learned_price_no_arbitrage"]["material_violations"] = 1

    result = runner.assess_acceptance(passing_report, acceptance)

    assert result["passed"] is False
    assert result["checks"][-1]["name"] == "no_arbitrage.material_violations"
    assert result["checks"][-1]["passed"] is False


def test_material_tolerance_must_match_evaluator(
    passing_report: dict[str, Any],
    acceptance: dict[str, Any],
) -> None:
    passing_report["learned_price_no_arbitrage"][
        "material_relative_tolerance"
    ] = 1e-5

    with pytest.raises(runner.ReplicationError, match="tolerances differ"):
        runner.assess_acceptance(passing_report, acceptance)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("dataset", "wrong dataset manifest"),
        ("artifact", "wrong artifact manifest"),
        ("weights", "wrong weight bytes"),
        ("constraint", "wrong output constraint"),
    ],
)
def test_report_must_reference_exact_evaluation_inputs(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    dataset = tmp_path / "data"
    bounded = tmp_path / "artifact"
    dataset.mkdir()
    bounded.mkdir()
    dataset_manifest = dataset / runner.DATASET_MANIFEST
    artifact_manifest = bounded / runner.ARTIFACT_MANIFEST
    weights = bounded / runner.WEIGHTS_FILE
    dataset_manifest.write_bytes(b"dataset")
    artifact_manifest.write_bytes(b"artifact")
    weights.write_bytes(b"weights")
    resolved = {"dataset": dataset, "bounded_artifact": bounded}
    protocol = {"output_constraint": "european_bounds_v1"}
    report = {
        "artifact": {
            "manifest_sha256": runner._sha256(artifact_manifest),
            "output_constraint": "european_bounds_v1",
            "weights_sha256": runner._sha256(weights),
        },
        "dataset": {"manifest_sha256": runner._sha256(dataset_manifest)},
        "partition": "validation",
        "partition_role": "model_selection",
    }
    if field == "dataset":
        report["dataset"]["manifest_sha256"] = "0" * 64
    elif field == "artifact":
        report["artifact"]["manifest_sha256"] = "0" * 64
    elif field == "weights":
        report["artifact"]["weights_sha256"] = "0" * 64
    else:
        report["artifact"]["output_constraint"] = "none"

    with pytest.raises(runner.ReplicationError, match=message):
        runner._validate_report_identity(
            report,
            resolved,
            protocol,
            partition="validation",
            role="model_selection",
        )


def test_validation_outputs_capture_lineage_and_every_dataset_split(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "data"
    source = tmp_path / "artifacts" / "source"
    bounded = tmp_path / "artifacts" / "bounded"
    dataset.mkdir()
    source.mkdir(parents=True)
    bounded.mkdir()
    files = []
    for split in ("train", "validation", "interpolation_test"):
        path = dataset / f"{split}.parquet"
        path.write_bytes(split.encode())
        files.append(
            {
                "file": path.name,
                "sha256": runner._sha256(path),
                "split": split,
            }
        )
    dataset_manifest = dataset / runner.DATASET_MANIFEST
    write_json(dataset_manifest, {"files": files})

    weights = b"identical weights"
    (source / runner.WEIGHTS_FILE).write_bytes(weights)
    (bounded / runner.WEIGHTS_FILE).write_bytes(weights)
    weights_sha256 = runner._sha256(source / runner.WEIGHTS_FILE)
    source_manifest = source / runner.ARTIFACT_MANIFEST
    write_json(source_manifest, {"weights": {"sha256": weights_sha256}})
    write_json(
        bounded / runner.ARTIFACT_MANIFEST,
        {
            "output_constraint": {
                "name": "european_bounds_v1",
                "source_artifact": {
                    "manifest_sha256": runner._sha256(source_manifest),
                    "weights_sha256": weights_sha256,
                },
            },
            "weights": {"sha256": weights_sha256},
        },
    )
    diagnostics = dataset / "diagnostics.json"
    write_json(diagnostics, {"findings": [], "status": "ok"})
    validation = bounded / "validation.json"
    write_json(
        validation,
        {
            "artifact": {
                "manifest_sha256": runner._sha256(
                    bounded / runner.ARTIFACT_MANIFEST
                ),
                "output_constraint": "european_bounds_v1",
                "weights_sha256": weights_sha256,
            },
            "dataset": {
                "manifest_sha256": runner._sha256(dataset_manifest),
            },
            "partition": "validation",
            "partition_role": "model_selection",
        },
    )
    resolved = {
        "bounded_artifact": bounded,
        "dataset": dataset,
        "diagnostics": diagnostics,
        "unconstrained_artifact": source,
        "validation_report": validation,
    }
    protocol = {
        "execution": {"checkpoint_selection_partition": "validation"},
        "output_constraint": "european_bounds_v1",
    }

    outputs = runner._collect_validation_outputs(resolved, protocol)

    assert {
        "dataset_split_train",
        "dataset_split_validation",
        "dataset_split_interpolation_test",
    } <= outputs.keys()
    runner._verify_recorded_outputs({"outputs": outputs}, tmp_path)
    (dataset / "interpolation_test.parquet").write_bytes(b"tampered")
    with pytest.raises(runner.ReplicationError, match="dataset_split_interpolation_test"):
        runner._verify_recorded_outputs({"outputs": outputs}, tmp_path)
    with pytest.raises(runner.ReplicationError, match="manifest digest"):
        runner._collect_validation_outputs(resolved, protocol)


@pytest.mark.parametrize(
    "output_key",
    [
        "dataset",
        "diagnostics",
        "unconstrained_artifact",
        "bounded_artifact",
        "validation_report",
        "final_report",
    ],
)
def test_preflight_refuses_every_existing_output(
    tmp_path: Path,
    output_key: str,
) -> None:
    resolved = {
        key: tmp_path / key
        for key in (
            "dataset",
            "diagnostics",
            "unconstrained_artifact",
            "bounded_artifact",
            "validation_report",
            "final_report",
        )
    }
    resolved[output_key].mkdir(parents=True)

    with pytest.raises(runner.ReplicationError, match="must not exist"):
        runner._refuse_existing_outputs(
            resolved,
            tmp_path / "runs" / "replication" / "execution.json",
        )


def test_preflight_refuses_an_existing_run_directory(tmp_path: Path) -> None:
    resolved = {
        key: tmp_path / key
        for key in (
            "dataset",
            "diagnostics",
            "unconstrained_artifact",
            "bounded_artifact",
            "validation_report",
            "final_report",
        )
    }
    state_path = tmp_path / "runs" / "replication" / "execution.json"
    state_path.parent.mkdir(parents=True)

    with pytest.raises(runner.ReplicationError, match="must not exist"):
        runner._refuse_existing_outputs(resolved, state_path)


@pytest.mark.parametrize("name", ["../escape", ".", "..", "nested/name"])
def test_protocol_name_cannot_escape_runs_directory(
    tmp_path: Path,
    name: str,
) -> None:
    with pytest.raises(runner.ReplicationError, match="not safe"):
        runner._state_path({"name": name}, tmp_path)


def minimal_protocol(root: Path) -> Path:
    protocol = root / "protocol.toml"
    protocol.write_text(
        """
name = "test-replication"
output_constraint = "european_bounds_v1"

[paths]
acceptance_config = "acceptance.toml"
replication_dataset_config = "dataset.toml"
replication_training_config = "training.toml"

[outputs]
dataset = "data"
diagnostics = "data/diagnostics.json"
unconstrained_artifact = "artifacts/source"
bounded_artifact = "artifacts/bounded"
validation_report = "artifacts/bounded/validation.json"
final_report = "artifacts/bounded/final.json"

[execution]
checkpoint_selection_partition = "validation"
final_partition = "interpolation_test"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (root / "acceptance.toml").write_text(
        ACCEPTANCE.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return protocol


def ready_state(root: Path, protocol: Path) -> Path:
    payload = tomllib.loads(protocol.read_text(encoding="utf-8"))
    state_path = runner._state_path(payload, root)
    state = {
        "commands": [],
        "final_evaluation_attempts": 0,
        "outputs": {"placeholder": {"path": "placeholder", "sha256": "0" * 64}},
        "repository": {
            "protocol_commit": "a" * 40,
            "source_commit": "b" * 40,
        },
        "schema_version": runner.RUN_SCHEMA_VERSION,
        "status": "ready_for_final",
        "validation_acceptance": {"passed": True},
    }
    runner._write_state(state_path, state)
    return state_path


def test_run_to_validation_stops_before_the_final_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    passing_report: dict[str, Any],
) -> None:
    protocol = minimal_protocol(tmp_path)
    monkeypatch.setattr(
        runner,
        "_validate_protocol",
        lambda *args: {"name": "test-replication"},
    )
    monkeypatch.setattr(
        runner,
        "_repository_evidence",
        lambda *args: {
            "branch": "main",
            "clean_worktree": True,
            "protocol_commit": "a" * 40,
            "remote_main": "b" * 40,
            "source_commit": "b" * 40,
        },
    )
    monkeypatch.setattr(runner, "_runtime_evidence", lambda *args: {})
    monkeypatch.setattr(
        runner,
        "_collect_validation_outputs",
        lambda *args: {
            "placeholder": {
                "bytes": 0,
                "path": "placeholder",
                "sha256": "0" * 64,
            }
        },
    )

    def complete_stage(
        ledger: Path,
        state: dict[str, Any],
        stage: str,
        arguments: list[str],
        project_root: Path,
    ) -> None:
        state["commands"].append({"arguments": arguments, "stage": stage})
        state["status"] = f"{stage}_completed"
        runner._write_state(ledger, state)
        if stage == "evaluate_validation":
            write_json(
                project_root / "artifacts" / "bounded" / "validation.json",
                passing_report,
            )

    monkeypatch.setattr(runner, "_record_command", complete_stage)

    state = runner.run_to_validation(protocol, tmp_path)

    assert state["status"] == "ready_for_final"
    assert state["validation_acceptance"]["passed"] is True
    assert [command["stage"] for command in state["commands"]] == [
        "generate_dataset",
        "diagnose_dataset",
        "train_model",
        "constrain_artifact",
        "evaluate_validation",
    ]
    arguments = [
        argument
        for command in state["commands"]
        for argument in command["arguments"]
    ]
    assert "interpolation_test" not in arguments
    assert not (tmp_path / "artifacts" / "bounded" / "final.json").exists()


def test_final_attempt_is_persisted_before_evaluator_starts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = minimal_protocol(tmp_path)
    state_path = ready_state(tmp_path, protocol)
    monkeypatch.setattr(runner, "_validate_protocol", lambda *args: {})
    monkeypatch.setattr(runner, "_verify_repository_unchanged", lambda *args: None)
    monkeypatch.setattr(runner, "_verify_recorded_outputs", lambda *args: None)

    def stop_before_evaluation(
        ledger: Path,
        state: dict[str, Any],
        stage: str,
        arguments: list[str],
        project_root: Path,
    ) -> None:
        del state, arguments, project_root
        persisted = json.loads(ledger.read_text(encoding="utf-8"))
        assert stage == "evaluate_locked_final"
        assert persisted["final_evaluation_attempts"] == 1
        assert persisted["status"] == "final_evaluation_reserved"
        raise runner.ReplicationError("simulated interruption")

    monkeypatch.setattr(runner, "_record_command", stop_before_evaluation)

    with pytest.raises(runner.ReplicationError, match="simulated interruption"):
        runner.final_evaluate(protocol, tmp_path, confirmed=True)

    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["final_evaluation_attempts"] == 1
    assert persisted["status"] == "final_evaluation_failed"


def test_successful_final_evaluation_is_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    passing_report: dict[str, Any],
) -> None:
    protocol = minimal_protocol(tmp_path)
    ready_state(tmp_path, protocol)
    monkeypatch.setattr(runner, "_validate_protocol", lambda *args: {})
    monkeypatch.setattr(runner, "_verify_repository_unchanged", lambda *args: None)
    monkeypatch.setattr(runner, "_verify_recorded_outputs", lambda *args: None)
    monkeypatch.setattr(runner, "_validate_report_identity", lambda *args, **kwargs: None)

    def complete_final(
        ledger: Path,
        state: dict[str, Any],
        stage: str,
        arguments: list[str],
        project_root: Path,
    ) -> None:
        del ledger, state, arguments
        assert stage == "evaluate_locked_final"
        write_json(
            project_root / "artifacts" / "bounded" / "final.json",
            passing_report,
        )

    monkeypatch.setattr(runner, "_record_command", complete_final)

    state = runner.final_evaluate(protocol, tmp_path, confirmed=True)

    assert state["final_evaluation_attempts"] == 1
    assert state["final_acceptance"]["passed"] is True
    assert state["status"] == "replication_passed"
    with pytest.raises(runner.ReplicationError, match="ready_for_final"):
        runner.final_evaluate(protocol, tmp_path, confirmed=True)


def test_final_evaluation_requires_explicit_confirmation(tmp_path: Path) -> None:
    protocol = minimal_protocol(tmp_path)

    with pytest.raises(runner.ReplicationError, match="requires --confirm"):
        runner.final_evaluate(protocol, tmp_path, confirmed=False)


def test_consumed_final_attempt_cannot_run_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = minimal_protocol(tmp_path)
    state_path = ready_state(tmp_path, protocol)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["final_evaluation_attempts"] = 1
    runner._write_state(state_path, state)
    monkeypatch.setattr(runner, "_validate_protocol", lambda *args: {})

    with pytest.raises(runner.ReplicationError, match="already consumed"):
        runner.final_evaluate(protocol, tmp_path, confirmed=True)


def test_command_plan_never_evaluates_the_final_partition(tmp_path: Path) -> None:
    protocol = {
        "execution": {
            "checkpoint_selection_partition": "validation",
            "final_partition": "interpolation_test",
        }
    }
    resolved = {
        "replication_dataset_config": tmp_path / "dataset.toml",
        "replication_training_config": tmp_path / "training.toml",
        "dataset": tmp_path / "data",
        "diagnostics": tmp_path / "data" / "diagnostics.json",
        "unconstrained_artifact": tmp_path / "artifacts" / "source",
        "bounded_artifact": tmp_path / "artifacts" / "bounded",
        "validation_report": tmp_path / "artifacts" / "bounded" / "validation.json",
    }

    plan = runner._command_plan(resolved, protocol)
    flattened = [argument for _, arguments in plan for argument in arguments]

    assert "differentiable_pricing.ml.constrain" in flattened
    assert "validation" in flattened
    assert "interpolation_test" not in flattened


def test_validation_gate_failure_returns_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        runner,
        "run_to_validation",
        lambda *args: {
            "final_evaluation_attempts": 0,
            "status": "validation_gates_failed",
            "validation_acceptance": {"passed": False},
        },
    )
    monkeypatch.setattr(
        runner,
        "_load_toml",
        lambda *args: {"name": "test-replication"},
    )
    monkeypatch.setattr(
        runner,
        "_state_path",
        lambda *args: Path("runs/test-replication/execution.json"),
    )

    assert runner.main(["run-to-validation"]) == 3
    assert json.loads(capsys.readouterr().out)["validation_passed"] is False


def test_atomic_state_write_never_leaves_temporary_files(tmp_path: Path) -> None:
    state_path = tmp_path / "run" / "execution.json"

    runner._write_state(state_path, {"status": "first"})
    runner._write_state(state_path, {"status": "second"})

    assert json.loads(state_path.read_text(encoding="utf-8"))["status"] == "second"
    assert list(state_path.parent.iterdir()) == [state_path]
