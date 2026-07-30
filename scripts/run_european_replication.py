#!/usr/bin/env python3
"""Execute the frozen European neural-pricing replication with an audit ledger.

``run-to-validation`` performs the fresh dataset generation, diagnostics,
training, bounds projection, and validation evaluation. It never reads the
locked final partition through model evaluation. ``final-evaluate`` is a
separate, explicit, one-shot transition that is enabled only after every
validation gate passes.

The ledger is stored under ``runs/`` (which is gitignored). Every transition is
written atomically. In particular, the final-evaluation attempt is recorded
before the evaluator starts, so a crash cannot turn one attempt into two.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL: Final = (
    PROJECT_ROOT / "configs/european_neural_replication_protocol_v1.toml"
)
PROTOCOL_VALIDATOR: Final = (
    PROJECT_ROOT / "scripts/check_european_replication_protocol.py"
)
RUN_SCHEMA_VERSION: Final = "european-neural-replication-run/v1"
STATE_FILE: Final = "execution.json"
ARTIFACT_MANIFEST: Final = "artifact.json"
WEIGHTS_FILE: Final = "weights.npz"
DATASET_MANIFEST: Final = "manifest.json"
EXPECTED_BRANCH: Final = "main"
PACKAGE_NAMES: Final = (
    "differentiable-pricing",
    "numpy",
    "pyarrow",
    "torch",
)
METRIC_GATES: Final = (
    (
        "overall.price_over_spot_rmse",
        ("slices", "overall", "metrics", "price_over_spot", "rmse"),
        ("overall", "price_over_spot_rmse_max"),
    ),
    (
        "overall.price_over_spot_p99_absolute_error",
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
        "overall.delta_rmse",
        ("slices", "overall", "metrics", "delta", "rmse"),
        ("overall", "delta_rmse_max"),
    ),
    (
        "overall.gamma_rmse",
        ("slices", "overall", "metrics", "gamma", "rmse"),
        ("overall", "gamma_rmse_max"),
    ),
    (
        "overall.vega_rmse",
        ("slices", "overall", "metrics", "vega", "rmse"),
        ("overall", "vega_rmse_max"),
    ),
    (
        "overall.theta_rmse",
        ("slices", "overall", "metrics", "theta", "rmse"),
        ("overall", "theta_rmse_max"),
    ),
    (
        "overall.rho_rmse",
        ("slices", "overall", "metrics", "rho", "rmse"),
        ("overall", "rho_rmse_max"),
    ),
    (
        "tail.price_over_spot_rmse",
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
        "tail.delta_rmse",
        ("slices", "moneyness_band", "tail", "metrics", "delta", "rmse"),
        ("tail", "delta_rmse_max"),
    ),
    (
        "tail.gamma_rmse",
        ("slices", "moneyness_band", "tail", "metrics", "gamma", "rmse"),
        ("tail", "gamma_rmse_max"),
    ),
)


class ReplicationError(RuntimeError):
    """Raised when the frozen replication cannot advance safely."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReplicationError(f"cannot load {description} '{path}': {error}") from error
    if not isinstance(payload, dict):
        raise ReplicationError(f"{description} '{path}' must be a JSON object")
    return payload


def _load_toml(path: Path, description: str) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            payload = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ReplicationError(f"cannot load {description} '{path}': {error}") from error
    if not isinstance(payload, dict):
        raise ReplicationError(f"{description} '{path}' must be a TOML table")
    return payload


def _load_protocol_validator() -> Any:
    spec = importlib.util.spec_from_file_location(
        "dp_check_european_replication_protocol",
        PROTOCOL_VALIDATOR,
    )
    if spec is None or spec.loader is None:
        raise ReplicationError("cannot load the replication protocol validator")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _validate_protocol(protocol_path: Path, project_root: Path) -> dict[str, Any]:
    validator = _load_protocol_validator()
    try:
        return validator.validate_protocol(protocol_path, project_root=project_root)
    except validator.ProtocolError as error:
        raise ReplicationError(f"invalid replication protocol: {error}") from error


def _run_capture(
    arguments: list[str],
    *,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            arguments,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise ReplicationError(
            f"cannot execute {arguments[0]!r}: {error}"
        ) from error


def _git(project_root: Path, *arguments: str) -> str:
    completed = _run_capture(["git", *arguments], cwd=project_root)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ReplicationError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout.strip()


def _repository_evidence(project_root: Path, protocol_path: Path) -> dict[str, Any]:
    if _git(project_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ReplicationError("working tree must be clean before replication execution")
    branch = _git(project_root, "branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise ReplicationError(
            f"replication must run from branch {EXPECTED_BRANCH!r}, got {branch!r}"
        )
    head = _git(project_root, "rev-parse", "HEAD")
    remote_main = _git(project_root, "rev-parse", "origin/main")
    if head != remote_main:
        raise ReplicationError(
            "HEAD must equal origin/main before replication execution"
        )
    relative_protocol = _relative_to_root(
        protocol_path,
        project_root,
        "replication protocol",
    )
    protocol_commit = _git(
        project_root,
        "log",
        "-1",
        "--format=%H",
        "--",
        relative_protocol.as_posix(),
    )
    if not protocol_commit:
        raise ReplicationError("replication protocol is not committed")
    ancestor = _run_capture(
        ["git", "merge-base", "--is-ancestor", protocol_commit, head],
        cwd=project_root,
    )
    if ancestor.returncode != 0:
        raise ReplicationError(
            "replication protocol commit is not an ancestor of the source commit"
        )
    return {
        "branch": branch,
        "clean_worktree": True,
        "protocol_commit": protocol_commit,
        "remote_main": remote_main,
        "source_commit": head,
    }


def _runtime_evidence(project_root: Path) -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    tools: dict[str, str] = {}
    for name, arguments in (
        ("git", ["git", "--version"]),
        ("cmake", ["cmake", "--version"]),
        ("cxx", ["c++", "--version"]),
    ):
        completed = _run_capture(arguments, cwd=project_root)
        if completed.returncode == 0:
            tools[name] = completed.stdout.splitlines()[0]
        else:
            tools[name] = "unavailable"
    extension = _run_capture(
        [
            sys.executable,
            "-c",
            (
                "import differentiable_pricing._core as core; "
                "print(core.__file__)"
            ),
        ],
        cwd=project_root,
    )
    if extension.returncode != 0:
        detail = extension.stderr.strip() or extension.stdout.strip()
        raise ReplicationError(
            f"cannot import the compiled pricing extension: {detail}"
        )
    extension_path = Path(extension.stdout.strip()).resolve()
    if not extension_path.is_file():
        raise ReplicationError(
            f"compiled pricing extension is not a file: {extension_path}"
        )
    return {
        "compiled_pricing_extension": _fingerprint(extension_path),
        "executable": sys.executable,
        "packages": packages,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "tools": tools,
    }


def _resolve_paths(
    protocol: dict[str, Any],
    project_root: Path,
) -> dict[str, Path]:
    outputs = protocol.get("outputs")
    paths = protocol.get("paths")
    if not isinstance(outputs, dict) or not isinstance(paths, dict):
        raise ReplicationError("validated protocol is missing paths or outputs")
    resolved: dict[str, Path] = {}
    for key, value in {**paths, **outputs}.items():
        if not isinstance(value, str):
            raise ReplicationError(f"protocol path {key!r} must be a string")
        resolved[key] = project_root / value
    return resolved


def _relative_to_root(path: Path, project_root: Path, description: str) -> Path:
    try:
        return path.resolve().relative_to(project_root.resolve())
    except ValueError as error:
        raise ReplicationError(
            f"{description} must stay inside the repository: {path}"
        ) from error


def _state_path(protocol: dict[str, Any], project_root: Path) -> Path:
    name = protocol.get("name")
    if not isinstance(name, str) or not name:
        raise ReplicationError("validated protocol has no name")
    if Path(name).name != name or name in {".", ".."}:
        raise ReplicationError("protocol name is not safe for a run directory")
    return project_root / "runs" / name / STATE_FILE


def _refuse_existing_outputs(
    resolved: dict[str, Path],
    state_path: Path,
) -> None:
    output_keys = (
        "dataset",
        "diagnostics",
        "unconstrained_artifact",
        "bounded_artifact",
        "validation_report",
        "final_report",
    )
    existing = sorted(
        str(resolved[key])
        for key in output_keys
        if resolved[key].exists()
    )
    if state_path.parent.exists():
        existing.append(str(state_path.parent))
    if existing:
        raise ReplicationError(
            "replication outputs must not exist before execution: "
            + ", ".join(existing)
        )


def _write_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_utc"] = _utc_now()
    text = json.dumps(
        state,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _record_command(
    state_path: Path,
    state: dict[str, Any],
    stage: str,
    arguments: list[str],
    project_root: Path,
) -> None:
    command = {
        "arguments": arguments,
        "finished_utc": None,
        "returncode": None,
        "stage": stage,
        "started_utc": _utc_now(),
        "stderr": None,
        "stdout": None,
    }
    state["commands"].append(command)
    state["status"] = f"{stage}_started"
    _write_state(state_path, state)
    print(f"[{stage}] {' '.join(arguments)}", flush=True)
    completed = _run_capture(arguments, cwd=project_root)
    command.update(
        {
            "finished_utc": _utc_now(),
            "returncode": completed.returncode,
            "stderr": completed.stderr,
            "stdout": completed.stdout,
        }
    )
    if completed.returncode != 0:
        state["failed_stage"] = stage
        state["status"] = "command_failed"
        _write_state(state_path, state)
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ReplicationError(
            f"stage {stage!r} failed with exit {completed.returncode}: {detail}"
        )
    state["status"] = f"{stage}_completed"
    _write_state(state_path, state)
    if completed.stdout:
        print(completed.stdout.rstrip())


def _number(payload: Any, path: str) -> float:
    if isinstance(payload, bool) or not isinstance(payload, int | float):
        raise ReplicationError(f"{path} must be numeric")
    value = float(payload)
    if not math.isfinite(value):
        raise ReplicationError(f"{path} must be finite")
    return value


def _nonnegative_integer(payload: Any, path: str) -> int:
    if isinstance(payload, bool) or not isinstance(payload, int) or payload < 0:
        raise ReplicationError(f"{path} must be a non-negative integer")
    return payload


def _nested(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    traversed: list[str] = []
    for key in path:
        traversed.append(key)
        if not isinstance(current, dict) or key not in current:
            raise ReplicationError(
                f"evaluation report is missing {'.'.join(traversed)}"
            )
        current = current[key]
    return current


def assess_acceptance(
    report: dict[str, Any],
    acceptance: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate every frozen acceptance gate against one evaluation report."""
    checks: list[dict[str, Any]] = []
    for name, report_path, gate_path in METRIC_GATES:
        value = _number(_nested(report, report_path), f"report {name}")
        limit = _number(
            _nested(acceptance, gate_path),
            f"acceptance {'.'.join(gate_path)}",
        )
        checks.append(
            {
                "limit": limit,
                "name": name,
                "passed": value <= limit,
                "value": value,
            }
        )

    no_arbitrage = _nested(report, ("learned_price_no_arbitrage",))
    if not isinstance(no_arbitrage, dict):
        raise ReplicationError("report learned_price_no_arbitrage must be an object")
    configured = _nested(acceptance, ("no_arbitrage",))
    if not isinstance(configured, dict):
        raise ReplicationError("acceptance no_arbitrage must be a table")
    report_tolerance = _number(
        no_arbitrage.get("material_relative_tolerance"),
        "report material_relative_tolerance",
    )
    configured_tolerance = _number(
        configured.get("material_relative_tolerance"),
        "acceptance material_relative_tolerance",
    )
    if report_tolerance != configured_tolerance:
        raise ReplicationError(
            "evaluation and acceptance material-relative tolerances differ"
        )
    violations = _nonnegative_integer(
        no_arbitrage.get("material_violations"),
        "report material_violations",
    )
    maximum_violations = _nonnegative_integer(
        configured.get("material_violations_max"),
        "acceptance material_violations_max",
    )
    checks.append(
        {
            "limit": maximum_violations,
            "name": "no_arbitrage.material_violations",
            "passed": violations <= maximum_violations,
            "value": violations,
        }
    )
    return {
        "checks": checks,
        "passed": all(bool(check["passed"]) for check in checks),
    }


def _fingerprint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReplicationError(f"expected file does not exist: {path}")
    return {
        "bytes": path.stat().st_size,
        "path": path.as_posix(),
        "sha256": _sha256(path),
    }


def _validate_report_identity(
    report: dict[str, Any],
    resolved: dict[str, Path],
    protocol: dict[str, Any],
    *,
    partition: str,
    role: str,
) -> None:
    if report.get("partition") != partition or report.get("partition_role") != role:
        raise ReplicationError(
            f"{role} report has the wrong partition or partition role"
        )
    dataset = report.get("dataset")
    artifact = report.get("artifact")
    if not isinstance(dataset, dict) or not isinstance(artifact, dict):
        raise ReplicationError("evaluation report lacks dataset or artifact provenance")
    dataset_digest = _sha256(resolved["dataset"] / DATASET_MANIFEST)
    artifact_digest = _sha256(
        resolved["bounded_artifact"] / ARTIFACT_MANIFEST
    )
    weights_digest = _sha256(resolved["bounded_artifact"] / WEIGHTS_FILE)
    if dataset.get("manifest_sha256") != dataset_digest:
        raise ReplicationError("evaluation report references the wrong dataset manifest")
    if artifact.get("manifest_sha256") != artifact_digest:
        raise ReplicationError("evaluation report references the wrong artifact manifest")
    if artifact.get("weights_sha256") != weights_digest:
        raise ReplicationError("evaluation report references the wrong weight bytes")
    if artifact.get("output_constraint") != protocol["output_constraint"]:
        raise ReplicationError("evaluation report uses the wrong output constraint")


def _collect_validation_outputs(
    resolved: dict[str, Path],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    diagnostics = _load_json(resolved["diagnostics"], "dataset diagnostics")
    if diagnostics.get("status") != "ok" or diagnostics.get("findings") != []:
        raise ReplicationError("dataset diagnostics must have status 'ok' and no findings")
    source = _load_json(
        resolved["unconstrained_artifact"] / ARTIFACT_MANIFEST,
        "unconstrained artifact manifest",
    )
    bounded = _load_json(
        resolved["bounded_artifact"] / ARTIFACT_MANIFEST,
        "bounded artifact manifest",
    )
    source_weights = _sha256(resolved["unconstrained_artifact"] / WEIGHTS_FILE)
    bounded_weights = _sha256(resolved["bounded_artifact"] / WEIGHTS_FILE)
    if source_weights != bounded_weights:
        raise ReplicationError("bounded artifact does not reuse identical weights")
    source_weight_block = source.get("weights")
    bounded_weight_block = bounded.get("weights")
    if (
        not isinstance(source_weight_block, dict)
        or source_weight_block.get("sha256") != source_weights
        or not isinstance(bounded_weight_block, dict)
        or bounded_weight_block.get("sha256") != bounded_weights
    ):
        raise ReplicationError("artifact manifests do not match their weight bytes")
    constraint = bounded.get("output_constraint")
    if not isinstance(constraint, dict):
        raise ReplicationError("bounded artifact has no output_constraint object")
    if constraint.get("name") != protocol["output_constraint"]:
        raise ReplicationError("bounded artifact constraint does not match the protocol")
    lineage = constraint.get("source_artifact")
    if not isinstance(lineage, dict):
        raise ReplicationError("bounded artifact has no source-artifact lineage")
    source_manifest_path = resolved["unconstrained_artifact"] / ARTIFACT_MANIFEST
    if (
        lineage.get("manifest_sha256") != _sha256(source_manifest_path)
        or lineage.get("weights_sha256") != source_weights
    ):
        raise ReplicationError("bounded artifact source lineage is inconsistent")
    report = _load_json(resolved["validation_report"], "validation report")
    _validate_report_identity(
        report,
        resolved,
        protocol,
        partition=protocol["execution"]["checkpoint_selection_partition"],
        role="model_selection",
    )
    dataset_manifest_path = resolved["dataset"] / DATASET_MANIFEST
    dataset_manifest = _load_json(dataset_manifest_path, "dataset manifest")
    files = dataset_manifest.get("files")
    if not isinstance(files, list):
        raise ReplicationError("dataset manifest files must be an array")
    outputs = {
        "bounded_artifact_manifest": _fingerprint(
            resolved["bounded_artifact"] / ARTIFACT_MANIFEST
        ),
        "bounded_weights": _fingerprint(
            resolved["bounded_artifact"] / WEIGHTS_FILE
        ),
        "dataset_diagnostics": _fingerprint(resolved["diagnostics"]),
        "dataset_manifest": _fingerprint(dataset_manifest_path),
        "unconstrained_artifact_manifest": _fingerprint(source_manifest_path),
        "unconstrained_weights": _fingerprint(
            resolved["unconstrained_artifact"] / WEIGHTS_FILE
        ),
        "validation_report": _fingerprint(resolved["validation_report"]),
    }
    for entry in files:
        if not isinstance(entry, dict):
            raise ReplicationError("dataset manifest file entry must be an object")
        split = entry.get("split")
        filename = entry.get("file")
        declared_sha256 = entry.get("sha256")
        if (
            not isinstance(split, str)
            or not isinstance(filename, str)
            or not isinstance(declared_sha256, str)
        ):
            raise ReplicationError("dataset manifest file entry is malformed")
        key = f"dataset_split_{split}"
        if key in outputs:
            raise ReplicationError(f"duplicate dataset split in manifest: {split}")
        split_path = resolved["dataset"] / filename
        _relative_to_root(split_path, resolved["dataset"], "dataset split")
        fingerprint = _fingerprint(split_path)
        if fingerprint["sha256"] != declared_sha256:
            raise ReplicationError(
                f"dataset split {split!r} does not match its manifest digest"
            )
        outputs[key] = fingerprint
    return outputs


def _command_plan(
    resolved: dict[str, Path],
    protocol: dict[str, Any],
) -> list[tuple[str, list[str]]]:
    python = sys.executable
    return [
        (
            "generate_dataset",
            [
                python,
                "-m",
                "differentiable_pricing.data.generate",
                "--config",
                str(resolved["replication_dataset_config"]),
                "--output",
                str(resolved["dataset"]),
            ],
        ),
        (
            "diagnose_dataset",
            [
                python,
                "-m",
                "differentiable_pricing.data.diagnose",
                "--dataset",
                str(resolved["dataset"]),
                "--output",
                str(resolved["diagnostics"]),
            ],
        ),
        (
            "train_model",
            [
                python,
                "-m",
                "differentiable_pricing.ml.train",
                "--dataset",
                str(resolved["dataset"]),
                "--config",
                str(resolved["replication_training_config"]),
                "--output",
                str(resolved["unconstrained_artifact"]),
            ],
        ),
        (
            "constrain_artifact",
            [
                python,
                "-m",
                "differentiable_pricing.ml.constrain",
                "--artifact",
                str(resolved["unconstrained_artifact"]),
                "--output",
                str(resolved["bounded_artifact"]),
            ],
        ),
        (
            "evaluate_validation",
            [
                python,
                "-m",
                "differentiable_pricing.ml.evaluate",
                "--dataset",
                str(resolved["dataset"]),
                "--artifact",
                str(resolved["bounded_artifact"]),
                "--partition",
                str(protocol["execution"]["checkpoint_selection_partition"]),
                "--output",
                str(resolved["validation_report"]),
            ],
        ),
    ]


def run_to_validation(protocol_path: Path, project_root: Path) -> dict[str, Any]:
    """Execute every locked stage up to, but not including, final evaluation."""
    summary = _validate_protocol(protocol_path, project_root)
    protocol = _load_toml(protocol_path, "replication protocol")
    resolved = _resolve_paths(protocol, project_root)
    state_path = _state_path(protocol, project_root)
    _refuse_existing_outputs(resolved, state_path)
    repository = _repository_evidence(project_root, protocol_path)
    state = {
        "commands": [],
        "created_utc": _utc_now(),
        "final_evaluation_attempts": 0,
        "outputs": {},
        "protocol": {
            "name": summary["name"],
            "path": _relative_to_root(
                protocol_path,
                project_root,
                "replication protocol",
            ).as_posix(),
            "sha256": _sha256(protocol_path),
        },
        "repository": repository,
        "runtime": _runtime_evidence(project_root),
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "prepared",
        "validation_acceptance": None,
    }
    try:
        state_path.parent.mkdir(parents=True, exist_ok=False)
    except OSError as error:
        raise ReplicationError(
            f"cannot reserve replication run directory '{state_path.parent}': {error}"
        ) from error
    _write_state(state_path, state)
    try:
        for stage, arguments in _command_plan(resolved, protocol):
            _record_command(state_path, state, stage, arguments, project_root)
        outputs = _collect_validation_outputs(resolved, protocol)
        validation = _load_json(resolved["validation_report"], "validation report")
        acceptance = _load_toml(
            resolved["acceptance_config"],
            "acceptance configuration",
        )
        gate_result = assess_acceptance(validation, acceptance)
        state["outputs"] = outputs
        state["validation_acceptance"] = gate_result
        state["status"] = (
            "ready_for_final" if gate_result["passed"] else "validation_gates_failed"
        )
        _write_state(state_path, state)
    except BaseException as error:
        if state.get("status") not in {
            "command_failed",
            "validation_gates_failed",
        }:
            state["failed_stage"] = state.get("status")
            state["failure"] = f"{type(error).__name__}: {error}"
            state["status"] = "execution_failed"
            _write_state(state_path, state)
        raise
    return state


def _verify_repository_unchanged(
    state: dict[str, Any],
    project_root: Path,
    protocol_path: Path,
) -> None:
    current = _repository_evidence(project_root, protocol_path)
    recorded = state.get("repository")
    if not isinstance(recorded, dict):
        raise ReplicationError("execution ledger has no repository evidence")
    if current["source_commit"] != recorded.get("source_commit"):
        raise ReplicationError("source commit changed after validation")
    if current["protocol_commit"] != recorded.get("protocol_commit"):
        raise ReplicationError("protocol commit changed after validation")


def _verify_recorded_outputs(
    state: dict[str, Any],
    project_root: Path,
) -> None:
    outputs = state.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        raise ReplicationError("execution ledger has no output fingerprints")
    for name, fingerprint in outputs.items():
        if not isinstance(fingerprint, dict):
            raise ReplicationError(f"invalid fingerprint for {name}")
        relative = fingerprint.get("path")
        expected = fingerprint.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ReplicationError(f"invalid fingerprint for {name}")
        path = Path(relative)
        if not path.is_absolute():
            path = project_root / path
        if not path.is_file() or _sha256(path) != expected:
            raise ReplicationError(f"recorded output changed after validation: {name}")


def final_evaluate(
    protocol_path: Path,
    project_root: Path,
    *,
    confirmed: bool,
) -> dict[str, Any]:
    """Consume the locked final partition exactly once."""
    if not confirmed:
        raise ReplicationError(
            "final evaluation requires --confirm-locked-final-evaluation"
        )
    _validate_protocol(protocol_path, project_root)
    protocol = _load_toml(protocol_path, "replication protocol")
    resolved = _resolve_paths(protocol, project_root)
    state_path = _state_path(protocol, project_root)
    state = _load_json(state_path, "replication execution ledger")
    if state.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ReplicationError("execution ledger schema version is unsupported")
    if state.get("status") != "ready_for_final":
        raise ReplicationError(
            "final evaluation requires a validation-passing ready_for_final ledger"
        )
    attempts = state.get("final_evaluation_attempts")
    if attempts != 0:
        raise ReplicationError("the locked final-evaluation attempt is already consumed")
    if resolved["final_report"].exists():
        raise ReplicationError("final report already exists; refusing another evaluation")
    _verify_repository_unchanged(state, project_root, protocol_path)
    _verify_recorded_outputs(state, project_root)

    state["final_evaluation_attempts"] = 1
    state["status"] = "final_evaluation_reserved"
    _write_state(state_path, state)
    arguments = [
        sys.executable,
        "-m",
        "differentiable_pricing.ml.evaluate",
        "--dataset",
        str(resolved["dataset"]),
        "--artifact",
        str(resolved["bounded_artifact"]),
        "--partition",
        str(protocol["execution"]["final_partition"]),
        "--output",
        str(resolved["final_report"]),
    ]
    try:
        _record_command(
            state_path,
            state,
            "evaluate_locked_final",
            arguments,
            project_root,
        )
        report = _load_json(resolved["final_report"], "locked final report")
        _validate_report_identity(
            report,
            resolved,
            protocol,
            partition=protocol["execution"]["final_partition"],
            role="locked_final_evaluation",
        )
        acceptance = _load_toml(
            resolved["acceptance_config"],
            "acceptance configuration",
        )
        gate_result = assess_acceptance(report, acceptance)
        state["final_acceptance"] = gate_result
        state["outputs"]["final_report"] = _fingerprint(resolved["final_report"])
        state["status"] = (
            "replication_passed" if gate_result["passed"] else "final_gates_failed"
        )
        _write_state(state_path, state)
    except BaseException as error:
        if state.get("status") != "command_failed":
            state["failed_stage"] = state.get("status")
            state["failure"] = f"{type(error).__name__}: {error}"
            state["status"] = "final_evaluation_failed"
            _write_state(state_path, state)
        raise
    return state


def status(protocol_path: Path, project_root: Path) -> dict[str, Any]:
    protocol = _load_toml(protocol_path, "replication protocol")
    state = _load_json(
        _state_path(protocol, project_root),
        "replication execution ledger",
    )
    protocol_block = state.get("protocol")
    repository = state.get("repository")
    validation = state.get("validation_acceptance")
    final = state.get("final_acceptance")
    return {
        "final_evaluation_attempts": state.get("final_evaluation_attempts"),
        "protocol": (
            protocol_block.get("name")
            if isinstance(protocol_block, dict)
            else None
        ),
        "source_commit": (
            repository.get("source_commit")
            if isinstance(repository, dict)
            else None
        ),
        "status": state.get("status"),
        "final_passed": (
            final.get("passed")
            if isinstance(final, dict)
            else None
        ),
        "validation_passed": (
            validation.get("passed")
            if isinstance(validation, dict)
            else None
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "run-to-validation",
        help="generate, train, constrain, and evaluate validation only",
    )
    final = subparsers.add_parser(
        "final-evaluate",
        help="consume the locked final partition once",
    )
    final.add_argument(
        "--confirm-locked-final-evaluation",
        action="store_true",
        help="confirm that this one-way transition intentionally consumes the final split",
    )
    subparsers.add_parser("status", help="print a concise run-ledger summary")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    protocol_path = arguments.protocol.resolve()
    try:
        if arguments.command == "run-to-validation":
            state = run_to_validation(protocol_path, PROJECT_ROOT)
            exit_code = 0 if state["validation_acceptance"]["passed"] else 3
            payload = {
                "final_evaluation_attempts": state["final_evaluation_attempts"],
                "ledger": str(
                    _state_path(
                        _load_toml(protocol_path, "protocol"),
                        PROJECT_ROOT,
                    )
                ),
                "status": state["status"],
                "validation_passed": state["validation_acceptance"]["passed"],
            }
        elif arguments.command == "final-evaluate":
            state = final_evaluate(
                protocol_path,
                PROJECT_ROOT,
                confirmed=arguments.confirm_locked_final_evaluation,
            )
            exit_code = 0 if state["final_acceptance"]["passed"] else 3
            payload = {
                "final_evaluation_attempts": state["final_evaluation_attempts"],
                "ledger": str(
                    _state_path(
                        _load_toml(protocol_path, "protocol"),
                        PROJECT_ROOT,
                    )
                ),
                "status": state["status"],
                "final_passed": state["final_acceptance"]["passed"],
            }
        else:
            payload = status(protocol_path, PROJECT_ROOT)
            exit_code = 0
    except ReplicationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
