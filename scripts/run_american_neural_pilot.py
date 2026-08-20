#!/usr/bin/env python3
"""Task 9G runner: validation, status, and one-shot locked-final evaluation.

The numerical cross-check, training, latency, and IV operations are manual,
terminal-invoked work.  This module is importable for fixture tests, but no test
or repository check calls the expensive execution function.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Final

import torch
from differentiable_pricing.ml.american_artifact import load_american_artifact
from differentiable_pricing.ml.american_pilot import (
    EVALUATION_COLUMNS,
    PilotError,
    _runtime_metadata,
    classify_outcome,
    evaluate_models,
    execute_validation_run,
    load_toml,
    read_partition_columns,
    sha256_file,
)
from differentiable_pricing.ml.artifact import load_physical_model, write_json_atomic

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL: Final = PROJECT_ROOT / "configs/american_neural_pilot_protocol_v1.toml"
LEDGER_SCHEMA: Final = "american-neural-pilot-execution/1"
LEDGER_KEYS: Final = frozenset(
    {
        "schema_version",
        "status",
        "protocol_sha256",
        "repository",
        "final_evaluation_attempts",
        "final_partition_consumed",
        "validation_report",
        "final_report",
        "failure",
    }
)


class RunnerError(RuntimeError):
    """Raised on any fail-closed identity or lifecycle condition."""


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RunnerError(f"cannot load {description} '{path}': {error}") from error
    if not isinstance(value, dict):
        raise RunnerError(f"{description} must be a JSON object")
    return value


def _load_validator(project_root: Path) -> Any:
    path = project_root / "scripts/check_american_neural_pilot_protocol.py"
    spec = importlib.util.spec_from_file_location("task9g_protocol_validator", path)
    if spec is None or spec.loader is None:
        raise RunnerError("cannot load Task 9G protocol validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_freezer(project_root: Path) -> Any:
    path = project_root / "scripts/freeze_american_neural_pilot_results.py"
    spec = importlib.util.spec_from_file_location("task9g_raw_validator", path)
    if spec is None or spec.loader is None:
        raise RunnerError("cannot load Task 9G raw-report validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_ledger(ledger: dict[str, Any]) -> None:
    missing = sorted(LEDGER_KEYS - set(ledger))
    unknown = sorted(set(ledger) - LEDGER_KEYS)
    if missing or unknown:
        raise RunnerError(f"execution ledger keys differ: missing={missing}, unknown={unknown}")
    if ledger["schema_version"] != LEDGER_SCHEMA:
        raise RunnerError("execution ledger schema differs")
    repository = ledger["repository"]
    if not isinstance(repository, dict) or set(repository) != {
        "commit",
        "branch",
        "tracked_worktree_clean",
    }:
        raise RunnerError("execution ledger repository identity is malformed")


def _validate_protocol(protocol_path: Path, project_root: Path) -> dict[str, Any]:
    try:
        return _load_validator(project_root).validate_protocol(protocol_path, project_root)
    except Exception as error:
        raise RunnerError(f"protocol validation failed: {error}") from error


def _paths(protocol: dict[str, Any], project_root: Path) -> dict[str, Path]:
    return {name: project_root / value for name, value in protocol["paths"].items()}


def _ledger_path(protocol: dict[str, Any], project_root: Path) -> Path:
    return project_root / protocol["paths"]["execution_ledger"]


def _repository_identity(project_root: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        result = subprocess.run(
            ["git", *arguments],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RunnerError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
        return result.stdout.strip()

    if git("status", "--porcelain", "--untracked-files=no"):
        raise RunnerError("locked execution requires no tracked worktree changes")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "tracked_worktree_clean": True,
    }


def _verify_pre_final_identities(
    protocol_path: Path,
    protocol: dict[str, Any],
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify everything except the locked-final file, which stays untouched."""
    _validate_protocol(protocol_path, project_root)
    paths = _paths(protocol, project_root)
    manifest = _load_json(paths["dataset_manifest"], "American dataset manifest")
    dataset = protocol["dataset"]
    if sha256_file(paths["dataset_manifest"]) != dataset["manifest_sha256"]:
        raise RunnerError("dataset manifest SHA-256 mismatch")
    if manifest.get("schema_version") != dataset["schema_version"]:
        raise RunnerError("dataset schema identity mismatch")
    if manifest.get("generator_version") != dataset["generator_version"]:
        raise RunnerError("dataset generator identity mismatch")
    if manifest.get("config", {}).get("sha256") != dataset["config_sha256"]:
        raise RunnerError("dataset configuration identity mismatch")
    for split, key in (("train", "train_sha256"), ("validation", "validation_sha256")):
        entry = [item for item in manifest["files"] if item["split"] == split]
        if len(entry) != 1 or entry[0]["sha256"] != dataset[key]:
            raise RunnerError(f"manifest {split} identity mismatch")
        path = paths["dataset"] / entry[0]["file"]
        if sha256_file(path) != dataset[key]:
            raise RunnerError(f"{split} file identity mismatch")
    # Do not resolve, stat, hash, or open the final partition here.
    source = protocol["source_artifact"]
    if sha256_file(paths["source_artifact"] / "artifact.json") != source["manifest_sha256"]:
        raise RunnerError("European source manifest identity mismatch")
    if sha256_file(paths["source_artifact"] / "weights.npz") != source["weights_sha256"]:
        raise RunnerError("European source weights identity mismatch")
    _, source_payload = load_physical_model(paths["source_artifact"])
    if source_payload["representation"]["name"] != "forward_normalized_v1":
        raise RunnerError("European source representation is incompatible")
    return manifest, _repository_identity(project_root)


def run_to_validation(
    protocol_path: Path = DEFAULT_PROTOCOL,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    protocol = load_toml(protocol_path)
    paths = _paths(protocol, project_root)
    ledger_path = _ledger_path(protocol, project_root)
    output = paths["output_directory"]
    if ledger_path.exists() or output.exists():
        raise RunnerError("pilot output or execution ledger already exists")
    manifest, repository = _verify_pre_final_identities(protocol_path, protocol, project_root)
    ledger_path.parent.mkdir(parents=True, exist_ok=False)
    ledger = {
        "schema_version": LEDGER_SCHEMA,
        "status": "validation_reserved",
        "protocol_sha256": sha256_file(protocol_path),
        "repository": repository,
        "final_evaluation_attempts": 0,
        "final_partition_consumed": False,
        "validation_report": None,
        "final_report": None,
        "failure": None,
    }
    write_json_atomic(ledger_path, ledger)
    try:
        report = execute_validation_run(
            project_root=project_root,
            protocol=protocol,
            manifest=manifest,
            output_directory=output,
        )
        validation_path = output / "validation-report.json"
        ledger["validation_report"] = {
            "path": str(validation_path.relative_to(project_root)),
            "sha256": sha256_file(validation_path),
        }
        ledger["status"] = (
            "ready_for_final"
            if report["validation_final_entry_passed"]
            else "validation_gates_failed"
        )
        write_json_atomic(ledger_path, ledger, overwrite=True)
    except BaseException as error:
        entry_failure = output / "entry-failure-report.json"
        if entry_failure.exists():
            ledger["status"] = "entry_gate_failed"
            ledger["validation_report"] = {
                "path": str(entry_failure.relative_to(project_root)),
                "sha256": sha256_file(entry_failure),
            }
        else:
            ledger["status"] = "validation_failed"
        ledger["failure"] = f"{type(error).__name__}: {error}"
        write_json_atomic(ledger_path, ledger, overwrite=True)
        raise
    return ledger


def status(
    protocol_path: Path = DEFAULT_PROTOCOL,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Read only protocol and ledger; never inspect any dataset partition."""
    protocol = load_toml(protocol_path)
    ledger = _load_json(_ledger_path(protocol, project_root), "execution ledger")
    _validate_ledger(ledger)
    return {
        "status": ledger.get("status"),
        "final_evaluation_attempts": ledger.get("final_evaluation_attempts"),
        "final_partition_consumed": ledger.get("final_partition_consumed"),
        "validation_report": ledger.get("validation_report"),
    }


def _reserve_final_attempt(marker: Path) -> None:
    marker.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RunnerError("the one-shot locked-final attempt is already consumed") from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write("consumed\n")
        stream.flush()
        os.fsync(stream.fileno())


def final_evaluate(
    protocol_path: Path = DEFAULT_PROTOCOL,
    project_root: Path = PROJECT_ROOT,
    *,
    confirmed: bool,
) -> dict[str, Any]:
    if not confirmed:
        raise RunnerError("final evaluation requires --confirm-locked-final-evaluation")
    protocol = load_toml(protocol_path)
    _validate_protocol(protocol_path, project_root)
    paths = _paths(protocol, project_root)
    ledger_path = _ledger_path(protocol, project_root)
    ledger = _load_json(ledger_path, "execution ledger")
    _validate_ledger(ledger)
    protocol_digest = sha256_file(protocol_path)
    if ledger["protocol_sha256"] != protocol_digest:
        raise RunnerError("execution ledger protocol identity differs")
    if ledger.get("schema_version") != LEDGER_SCHEMA or ledger.get("status") != "ready_for_final":
        raise RunnerError("final evaluation requires the validation final-entry gate")
    if ledger.get("final_evaluation_attempts") != 0:
        raise RunnerError("the one-shot locked-final attempt is already consumed")
    validation_fingerprint = ledger.get("validation_report")
    if not isinstance(validation_fingerprint, dict):
        raise RunnerError("ledger does not pin the validation report")
    validation_path = project_root / validation_fingerprint["path"]
    if sha256_file(validation_path) != validation_fingerprint["sha256"]:
        raise RunnerError("validation report changed before final evaluation")
    validation_report = _load_json(validation_path, "validation report")
    acceptance = load_toml(paths["acceptance_config"])
    latency = load_toml(paths["latency_config"])
    iv = load_toml(paths["iv_config"])
    training = load_toml(paths["training_config"])
    try:
        torch.set_num_interop_threads(int(latency["torch_interop_threads"]))
    except RuntimeError:
        if torch.get_num_interop_threads() != int(latency["torch_interop_threads"]):
            raise RunnerError("Torch inter-op thread budget was fixed incompatibly") from None
    torch.set_num_threads(int(training["training"]["num_threads"]))
    try:
        validation_passed = _load_freezer(project_root).validate_validation_report(
            validation_report,
            protocol,
            protocol_digest,
            acceptance,
            latency,
            iv,
        )
    except Exception as error:
        raise RunnerError(f"validation report failed strict validation: {error}") from error
    if not validation_passed:
        raise RunnerError("validation report does not pass the final-entry gate")
    repository = _repository_identity(project_root)
    if repository["commit"] != ledger["repository"]["commit"]:
        raise RunnerError("repository commit changed after validation")
    output = paths["output_directory"]
    models = {}
    for arm in ("scratch", "transfer"):
        artifact_manifest = output / arm / "artifact.json"
        if sha256_file(artifact_manifest) != validation_report["artifacts"][arm]["manifest_sha256"]:
            raise RunnerError(f"{arm} artifact changed after validation")
        row_budget = int(training["row_selection"]["row_budget"])
        fit_partition = str(training["standardization"]["fit_partition"])
        model, payload = load_american_artifact(
            output / arm,
            expected_row_budget=row_budget,
            expected_fit_partition=fit_partition,
        )
        expected_dataset = {
            "manifest_sha256": protocol["dataset"]["manifest_sha256"],
            "train_sha256": protocol["dataset"]["train_sha256"],
            "validation_sha256": protocol["dataset"]["validation_sha256"],
            "selected_train_rows": row_budget,
        }
        if (
            payload["dataset"] != expected_dataset
            or payload["protocol"]
            != {"sha256": protocol_digest, "schema_version": protocol["schema_version"]}
            or payload["weights"]["sha256"] != validation_report["artifacts"][arm]["weights_sha256"]
        ):
            raise RunnerError(f"{arm} artifact provenance differs from validation")
        models[arm] = model

    # The atomic marker is created before any final-partition path is resolved,
    # hashed, opened, counted, or read. Interruption after this line consumes it.
    failure_runtime = _runtime_metadata()
    failure_stage = "reservation"
    observed_partition_sha256 = None

    def record_consumed_failure(error: BaseException) -> None:
        failure_path = paths["final_failure_report"]
        failure_report = {
            "schema_version": "american-neural-pilot-final-attempt-failure-report/1",
            "protocol_sha256": protocol_digest,
            "attempt": 1,
            "partition": "interpolation_test",
            "expected_partition_sha256": protocol["dataset"]["locked_final_sha256"],
            "observed_partition_sha256": observed_partition_sha256,
            "failure_stage": failure_stage,
            "error": {
                "type": type(error).__name__,
                "message": str(error) or type(error).__name__,
            },
            "runtime": failure_runtime,
            "lifecycle": {
                "final_partition_consumed": True,
                "second_attempt_allowed": False,
            },
            "interpretation": (
                "terminal consumed final-attempt failure; no retry under this protocol"
            ),
        }
        write_json_atomic(failure_path, failure_report)
        ledger["final_evaluation_attempts"] = 1
        ledger["final_partition_consumed"] = True
        ledger["status"] = "final_evaluation_failed_consumed"
        ledger["final_report"] = {
            "path": str(failure_path.relative_to(project_root)),
            "sha256": sha256_file(failure_path),
        }
        ledger["failure"] = f"{type(error).__name__}: {error}"
        write_json_atomic(ledger_path, ledger, overwrite=True)

    attempt_marker = output / "locked-final-attempt.consumed"
    try:
        _reserve_final_attempt(attempt_marker)
    except RunnerError:
        raise
    except BaseException as error:
        if not attempt_marker.exists():
            raise
        record_consumed_failure(error)
        raise
    try:
        failure_stage = "ledger_reservation"
        ledger["final_evaluation_attempts"] = 1
        ledger["final_partition_consumed"] = True
        ledger["status"] = "final_evaluation_reserved"
        write_json_atomic(ledger_path, ledger, overwrite=True)
        failure_stage = "manifest_identity"
        manifest = _load_json(paths["dataset_manifest"], "American dataset manifest")
        final_entry = [item for item in manifest["files"] if item["split"] == "interpolation_test"]
        if len(final_entry) != 1:
            raise RunnerError("manifest final-partition declaration is malformed")
        final_path = paths["dataset"] / final_entry[0]["file"]
        if final_entry[0]["sha256"] != protocol["dataset"]["locked_final_sha256"]:
            raise RunnerError("manifest locked-final identity mismatch")
        failure_stage = "partition_identity"
        observed_partition_sha256 = sha256_file(final_path)
        if observed_partition_sha256 != protocol["dataset"]["locked_final_sha256"]:
            raise RunnerError("locked-final file identity mismatch")
        failure_stage = "partition_read"
        columns = read_partition_columns(
            paths["dataset"],
            manifest,
            "interpolation_test",
            EVALUATION_COLUMNS,
            verify_digest=False,
        )
        failure_stage = "model_evaluation"
        evidence = evaluate_models(
            models,
            columns,
            acceptance,
            batch_size=training["training"]["batch_size"],
            gate_section="final_accuracy",
        )
        final_report = {
            "schema_version": "american-neural-pilot-final-report/1",
            "protocol_sha256": protocol_digest,
            "attempt": 1,
            "partition": "interpolation_test",
            "partition_sha256": protocol["dataset"]["locked_final_sha256"],
            "models_and_baseline": evidence,
            "all_arms_passed": all(
                evidence[arm]["gate"]["passed"] for arm in ("scratch", "transfer")
            ),
            "outcome": classify_outcome(
                evidence,
                validation_report["validation"],
                validation_report["latency"],
                validation_report["implied_volatility"],
                acceptance,
                phase="final",
                evidence_complete=validation_report["validation_evidence_complete"]["passed"],
            ),
            "runtime": _runtime_metadata(),
            "lifecycle": {
                "final_partition_consumed": True,
                "second_attempt_allowed": False,
            },
        }
        failure_stage = "report_write"
        final_path_out = output / "final-report.json"
        write_json_atomic(final_path_out, final_report)
        ledger["status"] = (
            "pilot_final_complete"
            if final_report["all_arms_passed"]
            else "pilot_final_gates_failed"
        )
        ledger["final_report"] = {
            "path": str(final_path_out.relative_to(project_root)),
            "sha256": sha256_file(final_path_out),
        }
        write_json_atomic(ledger_path, ledger, overwrite=True)
    except BaseException as error:
        record_consumed_failure(error)
        raise
    return ledger


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run-to-validation")
    commands.add_parser("status")
    final = commands.add_parser("final-evaluate")
    final.add_argument("--confirm-locked-final-evaluation", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.command == "run-to-validation":
            payload = run_to_validation(arguments.protocol.resolve(), PROJECT_ROOT)
        elif arguments.command == "status":
            payload = status(arguments.protocol.resolve(), PROJECT_ROOT)
        else:
            payload = final_evaluate(
                arguments.protocol.resolve(),
                PROJECT_ROOT,
                confirmed=arguments.confirm_locked_final_evaluation,
            )
    except (PilotError, RunnerError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
