"""Task 9G partition-lifecycle tests; every operation is mocked or fixture-only."""

from __future__ import annotations

import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _module():
    path = PROJECT_ROOT / "scripts/run_american_neural_pilot.py"
    spec = importlib.util.spec_from_file_location("task9g_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _protocol(root: Path) -> Path:
    path = root / "protocol.toml"
    path.write_text(
        """
[paths]
execution_ledger = "runs/execution.json"
dataset = "data/dataset"
dataset_manifest = "data/dataset/manifest.json"
source_artifact = "artifacts/source"
output_directory = "artifacts/output"
training_config = "training.toml"
acceptance_config = "acceptance.toml"
latency_config = "latency.toml"
iv_config = "iv.toml"
protocol = "protocol.toml"
raw_report = "artifacts/output/validation-report.json"
final_report = "artifacts/output/final-report.json"
final_failure_report = "artifacts/output/final-attempt-failure-report.json"
result_snapshot = "docs/results/result.json"
""",
        encoding="utf-8",
    )
    return path


def _write_ledger(root: Path, status: str) -> None:
    runner = _module()
    path = root / "runs/execution.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "american-neural-pilot-execution/1",
                "status": status,
                "protocol_sha256": runner.sha256_file(root / "protocol.toml"),
                "repository": {
                    "commit": "b" * 40,
                    "branch": "fixture",
                    "tracked_worktree_clean": True,
                },
                "final_evaluation_attempts": 0,
                "final_partition_consumed": False,
                "validation_report": None,
                "final_report": None,
                "failure": None,
            }
        )
    )


def _guard_final_path(monkeypatch, final_path: Path) -> list[str]:
    accesses: list[str] = []
    original_open = Path.open
    original_stat = Path.stat

    def guarded_open(path: Path, *args, **kwargs):
        if path == final_path:
            accesses.append("open")
            raise AssertionError("pre-final path opened the locked-final file")
        return original_open(path, *args, **kwargs)

    def guarded_stat(path: Path, *args, **kwargs):
        if path == final_path:
            accesses.append("stat")
            raise AssertionError("pre-final path inspected the locked-final file")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    return accesses


def test_validation_executor_has_no_locked_final_partition_reference() -> None:
    runner = _module()
    source = inspect.getsource(runner.execute_validation_run)
    assert "interpolation_test" not in source
    assert "locked_final" not in source


def test_status_does_not_touch_any_dataset_partition(tmp_path, monkeypatch) -> None:
    runner = _module()
    protocol = _protocol(tmp_path)
    _write_ledger(tmp_path, "ready_for_final")
    final_path = tmp_path / "data/dataset/interpolation_test.parquet"
    final_path.parent.mkdir(parents=True)
    final_path.write_bytes(b"locked-final-sentinel")
    accesses = _guard_final_path(monkeypatch, final_path)
    assert runner.status(protocol, tmp_path)["status"] == "ready_for_final"
    assert accesses == []


def test_mocked_validation_failure_leaves_real_final_sentinel_untouched(
    tmp_path, monkeypatch
) -> None:
    runner = _module()
    protocol_path = _protocol(tmp_path)
    final_path = tmp_path / "data/dataset/interpolation_test.parquet"
    final_path.parent.mkdir(parents=True)
    final_path.write_bytes(b"locked-final-sentinel")
    accesses = _guard_final_path(monkeypatch, final_path)
    monkeypatch.setattr(
        runner,
        "_verify_pre_final_identities",
        lambda *args: (
            {"files": []},
            {"commit": "b" * 40, "branch": "fixture", "tracked_worktree_clean": True},
        ),
    )

    def fail(**kwargs):
        raise runner.PilotError("mocked validation failure")

    monkeypatch.setattr(runner, "execute_validation_run", fail)
    with pytest.raises(runner.PilotError, match="validation"):
        runner.run_to_validation(protocol_path, tmp_path)
    assert accesses == []


def test_mocked_successful_validation_leaves_real_final_sentinel_untouched(
    tmp_path, monkeypatch
) -> None:
    runner = _module()
    protocol_path = _protocol(tmp_path)
    final_path = tmp_path / "data/dataset/interpolation_test.parquet"
    final_path.parent.mkdir(parents=True)
    final_path.write_bytes(b"locked-final-sentinel")
    accesses = _guard_final_path(monkeypatch, final_path)
    monkeypatch.setattr(
        runner,
        "_verify_pre_final_identities",
        lambda *args: (
            {"files": []},
            {"commit": "b" * 40, "branch": "fixture", "tracked_worktree_clean": True},
        ),
    )

    def succeed(**kwargs):
        output = kwargs["output_directory"]
        output.mkdir(parents=True)
        (output / "validation-report.json").write_text("{}\n", encoding="utf-8")
        return {"validation_final_entry_passed": False}

    monkeypatch.setattr(runner, "execute_validation_run", succeed)
    ledger = runner.run_to_validation(protocol_path, tmp_path)
    assert ledger["status"] == "validation_gates_failed"
    assert accesses == []


def test_run_to_validation_identity_failure_never_calls_executor(tmp_path, monkeypatch) -> None:
    runner = _module()
    protocol = _protocol(tmp_path)
    called = False

    def fail_identity(*args, **kwargs):
        raise runner.RunnerError("identity failure")

    def execute(**kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(runner, "_verify_pre_final_identities", fail_identity)
    monkeypatch.setattr(runner, "execute_validation_run", execute)
    with pytest.raises(runner.RunnerError, match="identity failure"):
        runner.run_to_validation(protocol, tmp_path)
    assert not called
    assert not (tmp_path / "data/dataset/interpolation_test.parquet").exists()


def test_final_confirmation_and_validation_gate_precede_final_access(tmp_path, monkeypatch) -> None:
    runner = _module()
    protocol = _protocol(tmp_path)
    with pytest.raises(runner.RunnerError, match="requires"):
        runner.final_evaluate(protocol, tmp_path, confirmed=False)
    _write_ledger(tmp_path, "validation_gates_failed")
    monkeypatch.setattr(runner, "_validate_protocol", lambda *args: {})
    with pytest.raises(runner.RunnerError, match="final-entry"):
        runner.final_evaluate(protocol, tmp_path, confirmed=True)
    assert not (tmp_path / "data/dataset/interpolation_test.parquet").exists()


def test_atomic_attempt_marker_is_one_shot(tmp_path) -> None:
    runner = _module()
    marker = tmp_path / "attempt.consumed"
    runner._reserve_final_attempt(marker)
    assert marker.read_text() == "consumed\n"
    with pytest.raises(runner.RunnerError, match="already consumed"):
        runner._reserve_final_attempt(marker)


@pytest.mark.parametrize("interrupt_after_reservation", [False, True])
def test_first_locked_final_file_access_occurs_after_atomic_reservation(
    tmp_path, monkeypatch, interrupt_after_reservation
) -> None:
    runner = _module()
    protocol_path = tmp_path / "protocol.toml"
    protocol_path.write_text("schema_version = 'fixture'\n", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    dataset = tmp_path / "data/dataset"
    dataset.mkdir(parents=True)
    final_path = dataset / "interpolation_test.parquet"
    final_path.write_bytes(b"locked-final-sentinel")
    final_digest = runner.sha256_file(final_path)
    manifest_path = dataset / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "split": "interpolation_test",
                        "file": final_path.name,
                        "sha256": final_digest,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    artifact_digests = {}
    for arm in ("scratch", "transfer"):
        directory = output / arm
        directory.mkdir()
        manifest_file = directory / "artifact.json"
        manifest_file.write_text(f'{{"arm": "{arm}"}}\n', encoding="utf-8")
        artifact_digests[arm] = runner.sha256_file(manifest_file)
    protocol = {
        "schema_version": "fixture",
        "paths": {
            "protocol": "protocol.toml",
            "execution_ledger": "execution.json",
            "dataset": "data/dataset",
            "dataset_manifest": "data/dataset/manifest.json",
            "output_directory": "output",
            "final_failure_report": "output/final-attempt-failure-report.json",
            "acceptance_config": "acceptance.toml",
            "latency_config": "latency.toml",
            "iv_config": "iv.toml",
            "training_config": "training.toml",
        },
        "dataset": {
            "manifest_sha256": "a" * 64,
            "train_sha256": "b" * 64,
            "validation_sha256": "c" * 64,
            "locked_final_sha256": final_digest,
        },
    }
    protocol_digest = runner.sha256_file(protocol_path)
    validation_path = output / "validation-report.json"
    validation_path.write_text(
        json.dumps(
            {
                "artifacts": {
                    arm: {
                        "manifest_sha256": artifact_digests[arm],
                        "weights_sha256": "d" * 64,
                    }
                    for arm in ("scratch", "transfer")
                },
                "validation": {},
                "latency": {},
                "implied_volatility": {},
                "validation_evidence_complete": {"passed": True},
            }
        ),
        encoding="utf-8",
    )
    ledger = {
        "schema_version": "american-neural-pilot-execution/1",
        "status": "ready_for_final",
        "protocol_sha256": protocol_digest,
        "repository": {
            "commit": "e" * 40,
            "branch": "fixture",
            "tracked_worktree_clean": True,
        },
        "final_evaluation_attempts": 0,
        "final_partition_consumed": False,
        "validation_report": {
            "path": "output/validation-report.json",
            "sha256": runner.sha256_file(validation_path),
        },
        "final_report": None,
        "failure": None,
    }
    (tmp_path / "execution.json").write_text(json.dumps(ledger), encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "load_toml",
        lambda path: (
            protocol
            if path == protocol_path
            else (
                {"training": {"batch_size": 1, "num_threads": 1}}
                if path.name == "training.toml"
                else ({"torch_interop_threads": 1} if path.name == "latency.toml" else {})
            )
        ),
    )
    monkeypatch.setattr(runner, "_validate_protocol", lambda *args: {})
    monkeypatch.setattr(
        runner,
        "_load_freezer",
        lambda root: SimpleNamespace(validate_validation_report=lambda *args: True),
    )
    monkeypatch.setattr(runner, "_repository_identity", lambda root: ledger["repository"])
    configured_threads = []
    monkeypatch.setattr(
        runner.torch,
        "set_num_interop_threads",
        lambda value: configured_threads.append(("interop", value)),
    )
    monkeypatch.setattr(
        runner.torch,
        "set_num_threads",
        lambda value: configured_threads.append(("intraop", value)),
    )
    monkeypatch.setattr(runner, "_runtime_metadata", lambda: {})
    payload = {
        "dataset": {
            "manifest_sha256": "a" * 64,
            "train_sha256": "b" * 64,
            "validation_sha256": "c" * 64,
            "selected_train_rows": 32768,
        },
        "protocol": {"sha256": protocol_digest, "schema_version": "fixture"},
        "weights": {"sha256": "d" * 64},
    }
    monkeypatch.setattr(runner, "load_american_artifact", lambda path: (object(), payload))
    marker_created = False
    original_reserve = runner._reserve_final_attempt

    def reserve(marker):
        nonlocal marker_created
        original_reserve(marker)
        marker_created = True
        if interrupt_after_reservation:
            raise KeyboardInterrupt

    monkeypatch.setattr(runner, "_reserve_final_attempt", reserve)
    original_open = Path.open
    original_stat = Path.stat

    def guarded_open(path: Path, *args, **kwargs):
        if path == final_path:
            assert marker_created
        return original_open(path, *args, **kwargs)

    def guarded_stat(path: Path, *args, **kwargs):
        if path == final_path:
            assert marker_created
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "stat", guarded_stat)
    monkeypatch.setattr(
        runner,
        "read_partition_columns",
        lambda *args, **kwargs: {"sample_id": []} if marker_created else pytest.fail(),
    )
    monkeypatch.setattr(
        runner,
        "evaluate_models",
        lambda *args, **kwargs: {
            "scratch": {"gate": {"passed": True}},
            "transfer": {"gate": {"passed": True}},
            "european_crr_baseline": {},
        },
    )
    monkeypatch.setattr(runner, "classify_outcome", lambda *args, **kwargs: {"outcome": "x"})
    if interrupt_after_reservation:
        with pytest.raises(KeyboardInterrupt):
            runner.final_evaluate(protocol_path, tmp_path, confirmed=True)
        consumed = json.loads(
            (output / "final-attempt-failure-report.json").read_text(encoding="utf-8")
        )
        assert consumed["lifecycle"] == {
            "final_partition_consumed": True,
            "second_attempt_allowed": False,
        }
        assert consumed["failure_stage"] == "reservation"
        assert consumed["expected_partition_sha256"] == final_digest
        assert consumed["observed_partition_sha256"] is None
        ledger_after = json.loads((tmp_path / "execution.json").read_text(encoding="utf-8"))
        assert ledger_after["status"] == "final_evaluation_failed_consumed"
        assert marker_created
        return
    result = runner.final_evaluate(protocol_path, tmp_path, confirmed=True)
    assert marker_created
    assert configured_threads == [("interop", 1), ("intraop", 1)]
    assert result["final_partition_consumed"] is True
