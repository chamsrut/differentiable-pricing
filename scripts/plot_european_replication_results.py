#!/usr/bin/env python3
"""Validate the frozen European replication snapshot and render deterministic SVGs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final
from xml.sax.saxutils import escape

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS: Final = (
    PROJECT_ROOT / "docs/results/european_replication_results_v1.json"
)
DEFAULT_OUTPUT_DIRECTORY: Final = PROJECT_ROOT / "docs/figures"
EXPECTED_PROTOCOL: Final = (
    PROJECT_ROOT / "configs/european_neural_replication_protocol_v1.toml"
)
EXPECTED_DATASET_CONFIG: Final = (
    PROJECT_ROOT / "configs/european_option_dataset_replication_v1.toml"
)
EXPECTED_TRAINING_CONFIG: Final = (
    PROJECT_ROOT / "configs/european_neural_replication_v1.toml"
)
EXPECTED_ACCEPTANCE_CONFIG: Final = (
    PROJECT_ROOT / "configs/european_neural_acceptance_v1.toml"
)
SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
COMMIT_PATTERN: Final = re.compile(r"^[0-9a-f]{40}$")
METRICS: Final = (
    "price_rmse",
    "price_over_spot_rmse",
    "price_over_spot_p99_absolute_error",
    "delta_rmse",
    "gamma_rmse",
    "vega_rmse",
    "theta_rmse",
    "rho_rmse",
    "tail_price_over_spot_rmse",
    "tail_delta_rmse",
    "tail_gamma_rmse",
)
CHECKS: Final = (
    (
        "overall.price_over_spot_rmse",
        "price_over_spot_rmse",
        "overall",
        "price_over_spot_rmse_max",
        "Price / spot RMSE",
    ),
    (
        "overall.price_over_spot_p99_absolute_error",
        "price_over_spot_p99_absolute_error",
        "overall",
        "price_over_spot_p99_absolute_error_max",
        "Price / spot P99",
    ),
    ("overall.delta_rmse", "delta_rmse", "overall", "delta_rmse_max", "Delta RMSE"),
    ("overall.gamma_rmse", "gamma_rmse", "overall", "gamma_rmse_max", "Gamma RMSE"),
    ("overall.vega_rmse", "vega_rmse", "overall", "vega_rmse_max", "Vega RMSE"),
    (
        "overall.theta_rmse",
        "theta_rmse",
        "overall",
        "theta_rmse_max",
        "Theta RMSE",
    ),
    ("overall.rho_rmse", "rho_rmse", "overall", "rho_rmse_max", "Rho RMSE"),
    (
        "tail.price_over_spot_rmse",
        "tail_price_over_spot_rmse",
        "tail",
        "price_over_spot_rmse_max",
        "Tail price / spot",
    ),
    (
        "tail.delta_rmse",
        "tail_delta_rmse",
        "tail",
        "delta_rmse_max",
        "Tail delta RMSE",
    ),
    (
        "tail.gamma_rmse",
        "tail_gamma_rmse",
        "tail",
        "gamma_rmse_max",
        "Tail gamma RMSE",
    ),
    (
        "no_arbitrage.material_violations",
        "material_violations",
        "no_arbitrage",
        "material_violations_max",
        "Material violations",
    ),
)


PROJECTION_TICK_STEP: Final = 300
PROJECTION_LABEL_RESERVE: Final = 23.0


class PlotError(RuntimeError):
    """Raised when the result snapshot or generated figures are inconsistent."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PlotError(f"cannot load result snapshot '{path}': {error}") from error
    if not isinstance(payload, dict):
        raise PlotError("result snapshot must be a JSON object")
    return payload


def _load_toml(path: Path, description: str) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            payload = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise PlotError(f"cannot load {description} '{path}': {error}") from error
    if not isinstance(payload, dict):
        raise PlotError(f"{description} must be a TOML table")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise PlotError(f"cannot hash '{path}': {error}") from error
    return digest.hexdigest()


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PlotError(f"{where} must be an object")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise PlotError(f"{where} must be a non-empty string")
    return value


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PlotError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise PlotError(f"{where} must be finite and non-negative")
    return result


def _integer(value: Any, where: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise PlotError(f"{where} must be an integer >= {minimum}")
    return value


def _gate_limit(value: Any, where: str, *, zero: bool) -> float:
    """Validate a gate limit before it is compared against or divided into.

    Only the material-violation gate may be zero, and it must be: every other
    gate is rendered as a fraction of its limit, so a zero would turn a stale
    or future snapshot into an uncaught ``ZeroDivisionError``.
    """
    limit = _number(value, where)
    if zero:
        if limit != 0.0:
            raise PlotError(f"{where} must remain the frozen zero limit")
    elif limit <= 0.0:
        raise PlotError(f"{where} must be a strictly positive limit")
    return limit


def _digest(value: Any, where: str) -> str:
    result = _string(value, where)
    if SHA256_PATTERN.fullmatch(result) is None:
        raise PlotError(f"{where} must be a lowercase SHA-256 digest")
    return result


def _commit(value: Any, where: str) -> str:
    result = _string(value, where)
    if COMMIT_PATTERN.fullmatch(result) is None:
        raise PlotError(f"{where} must be a full lowercase Git commit")
    return result


def _repo_path(value: Any, where: str, expected: Path) -> Path:
    result = _string(value, where)
    pure = PurePosixPath(result)
    if pure.is_absolute() or ".." in pure.parts:
        raise PlotError(f"{where} must be a repository-relative path")
    resolved = PROJECT_ROOT / Path(*pure.parts)
    if resolved != expected:
        raise PlotError(f"{where} must be '{expected.relative_to(PROJECT_ROOT)}'")
    return resolved


def _assert_file_digest(path: Path, declared: Any, where: str) -> str:
    digest = _digest(declared, where)
    actual = _sha256_file(path)
    if digest != actual:
        raise PlotError(f"{where} does not match the checked-in file: {digest} != {actual}")
    return digest


def _validate_partition(
    result: Mapping[str, Any],
    *,
    name: str,
    partition: str,
    role: str,
    rows: int,
) -> Mapping[str, Any]:
    if result.get("partition") != partition:
        raise PlotError(f"results.{name}.partition must be '{partition}'")
    if result.get("partition_role") != role:
        raise PlotError(f"results.{name}.partition_role must be '{role}'")
    if result.get("rows") != rows:
        raise PlotError(f"results.{name}.rows must be {rows}")
    _digest(result.get("report_sha256"), f"results.{name}.report_sha256")

    metrics = _mapping(result.get("metrics"), f"results.{name}.metrics")
    if set(metrics) != set(METRICS):
        raise PlotError(f"results.{name}.metrics must contain exactly {list(METRICS)}")
    for metric in METRICS:
        _number(metrics.get(metric), f"results.{name}.metrics.{metric}")

    no_arbitrage = _mapping(
        result.get("no_arbitrage"),
        f"results.{name}.no_arbitrage",
    )
    material = _integer(
        no_arbitrage.get("material_violations"),
        f"results.{name}.no_arbitrage.material_violations",
    )
    total = _integer(
        no_arbitrage.get("total_violations"),
        f"results.{name}.no_arbitrage.total_violations",
    )
    if material > total or total > rows:
        raise PlotError(f"results.{name}.no_arbitrage counts are inconsistent")

    projection = _mapping(result.get("projection"), f"results.{name}.projection")
    active = _integer(
        projection.get("active_rows"),
        f"results.{name}.projection.active_rows",
    )
    lower = _integer(
        projection.get("lower_bound_activations"),
        f"results.{name}.projection.lower_bound_activations",
    )
    upper = _integer(
        projection.get("upper_bound_activations"),
        f"results.{name}.projection.upper_bound_activations",
    )
    unconstrained = _integer(
        projection.get("unconstrained_material_violations"),
        f"results.{name}.projection.unconstrained_material_violations",
    )
    _number(
        projection.get("maximum_absolute_adjustment"),
        f"results.{name}.projection.maximum_absolute_adjustment",
    )
    if active != lower + upper or active > rows or unconstrained > active:
        raise PlotError(f"results.{name}.projection counts are inconsistent")
    if active == 0 or unconstrained == 0:
        raise PlotError(
            f"results.{name} must disclose the non-zero projection intervention"
        )
    return result


def validate_snapshot(results: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the frozen result and reconcile it with every checked-in lock."""
    if results.get("schema_version") != "european-neural-replication-result/v1":
        raise PlotError("result snapshot schema is unsupported")

    protocol = _mapping(results.get("protocol"), "protocol")
    protocol_path = _repo_path(
        protocol.get("path"),
        "protocol.path",
        EXPECTED_PROTOCOL,
    )
    protocol_digest = _assert_file_digest(
        protocol_path,
        protocol.get("sha256"),
        "protocol.sha256",
    )
    protocol_config = _load_toml(protocol_path, "replication protocol")
    if protocol.get("name") != protocol_config.get("name"):
        raise PlotError("protocol.name does not match the checked-in protocol")

    protocol_hashes = _mapping(protocol_config.get("sha256"), "protocol [sha256]")
    dataset = _mapping(results.get("dataset"), "dataset")
    dataset_config_path = _repo_path(
        dataset.get("config"),
        "dataset.config",
        EXPECTED_DATASET_CONFIG,
    )
    dataset_config_digest = _assert_file_digest(
        dataset_config_path,
        dataset.get("config_sha256"),
        "dataset.config_sha256",
    )
    if dataset_config_digest != protocol_hashes.get("replication_dataset_config"):
        raise PlotError("dataset config digest does not match the protocol lock")
    dataset_config = _load_toml(dataset_config_path, "replication dataset config")
    dataset_table = _mapping(dataset_config.get("dataset"), "dataset config [dataset]")
    row_table = _mapping(dataset_table.get("rows"), "dataset config [dataset.rows]")
    if dataset.get("name") != dataset_table.get("name"):
        raise PlotError("dataset.name does not match the replication dataset config")
    if dataset.get("base_seed") != dataset_table.get("base_seed"):
        raise PlotError("dataset.base_seed does not match the replication dataset config")

    rows = _mapping(dataset.get("rows"), "dataset.rows")
    for split in ("train", "validation", "interpolation_test"):
        expected_rows = _integer(row_table.get(split), f"dataset config rows.{split}")
        if rows.get(split) != expected_rows:
            raise PlotError(f"dataset.rows.{split} does not match its config")
    _digest(dataset.get("manifest_sha256"), "dataset.manifest_sha256")
    diagnostics = _mapping(dataset.get("diagnostics"), "dataset.diagnostics")
    _digest(diagnostics.get("sha256"), "dataset.diagnostics.sha256")
    if diagnostics.get("status") != "ok" or diagnostics.get("findings") != 0:
        raise PlotError("dataset diagnostics must be ok with zero findings")
    splits = _mapping(dataset.get("splits"), "dataset.splits")
    if set(splits) != {"train", "validation", "interpolation_test"}:
        raise PlotError("dataset.splits must name train, validation, and interpolation_test")
    for split, details_value in splits.items():
        details = _mapping(details_value, f"dataset.splits.{split}")
        _digest(details.get("sha256"), f"dataset.splits.{split}.sha256")
        _integer(details.get("bytes"), f"dataset.splits.{split}.bytes", minimum=1)

    model = _mapping(results.get("model"), "model")
    training_path = _repo_path(
        model.get("training_config"),
        "model.training_config",
        EXPECTED_TRAINING_CONFIG,
    )
    training_digest = _assert_file_digest(
        training_path,
        model.get("training_config_sha256"),
        "model.training_config_sha256",
    )
    if training_digest != protocol_hashes.get("replication_training_config"):
        raise PlotError("training config digest does not match the protocol lock")
    training_config = _load_toml(training_path, "replication training config")
    if model.get("representation") != training_config.get("representation"):
        raise PlotError("model representation does not match the training config")
    objective = _mapping(training_config.get("objective"), "training config [objective]")
    if model.get("objective") != objective.get("name"):
        raise PlotError("model objective does not match the training config")
    architecture = _mapping(model.get("architecture"), "model.architecture")
    model_config = _mapping(training_config.get("model"), "training config [model]")
    if architecture.get("hidden_dimensions") != model_config.get("hidden_dimensions"):
        raise PlotError("model hidden dimensions do not match the training config")
    if architecture.get("activation") != model_config.get("activation"):
        raise PlotError("model activation does not match the training config")
    training = _mapping(model.get("training"), "model.training")
    training_config_table = _mapping(
        training_config.get("training"),
        "training config [training]",
    )
    if training.get("seed") != training_config_table.get("seed"):
        raise PlotError("model training seed does not match the training config")
    if training.get("completed_epochs") != training_config_table.get("max_epochs"):
        raise PlotError("model completed epochs do not match the frozen budget")
    best_epoch = _integer(training.get("best_epoch"), "model.training.best_epoch", minimum=1)
    if best_epoch > training["completed_epochs"]:
        raise PlotError("model best epoch is outside the completed training history")
    _number(
        training.get("best_validation_normalized_price_mse"),
        "model.training.best_validation_normalized_price_mse",
    )
    _number(
        training.get("best_validation_objective"),
        "model.training.best_validation_objective",
    )
    if model.get("output_constraint") != protocol_config.get("output_constraint"):
        raise PlotError("model output constraint does not match the protocol")

    unconstrained = _mapping(
        model.get("unconstrained_artifact"),
        "model.unconstrained_artifact",
    )
    bounded = _mapping(model.get("bounded_artifact"), "model.bounded_artifact")
    protocol_outputs = _mapping(protocol_config.get("outputs"), "protocol [outputs]")
    if unconstrained.get("path") != protocol_outputs.get("unconstrained_artifact"):
        raise PlotError("unconstrained artifact path does not match the protocol")
    if bounded.get("path") != protocol_outputs.get("bounded_artifact"):
        raise PlotError("bounded artifact path does not match the protocol")
    unconstrained_manifest = _digest(
        unconstrained.get("manifest_sha256"),
        "model.unconstrained_artifact.manifest_sha256",
    )
    unconstrained_weights = _digest(
        unconstrained.get("weights_sha256"),
        "model.unconstrained_artifact.weights_sha256",
    )
    bounded_manifest = _digest(
        bounded.get("manifest_sha256"),
        "model.bounded_artifact.manifest_sha256",
    )
    bounded_weights = _digest(
        bounded.get("weights_sha256"),
        "model.bounded_artifact.weights_sha256",
    )
    if bounded.get("source_manifest_sha256") != unconstrained_manifest:
        raise PlotError("bounded artifact source manifest does not match unconstrained")
    if bounded_weights != unconstrained_weights:
        raise PlotError("bounded artifact did not preserve the trained weight bytes")
    if bounded_manifest == unconstrained_manifest:
        raise PlotError("bounded and unconstrained artifacts must have distinct manifests")

    acceptance = _mapping(results.get("acceptance"), "acceptance")
    acceptance_path = _repo_path(
        acceptance.get("config"),
        "acceptance.config",
        EXPECTED_ACCEPTANCE_CONFIG,
    )
    acceptance_digest = _assert_file_digest(
        acceptance_path,
        acceptance.get("config_sha256"),
        "acceptance.config_sha256",
    )
    if acceptance_digest != protocol_hashes.get("acceptance_config"):
        raise PlotError("acceptance config digest does not match the protocol lock")
    gates = _load_toml(acceptance_path, "acceptance config")

    result_tables = _mapping(results.get("results"), "results")
    validation = _validate_partition(
        _mapping(result_tables.get("validation"), "results.validation"),
        name="validation",
        partition="validation",
        role="model_selection",
        rows=_integer(rows.get("validation"), "dataset.rows.validation"),
    )
    final = _validate_partition(
        _mapping(result_tables.get("locked_final"), "results.locked_final"),
        name="locked_final",
        partition="interpolation_test",
        role="locked_final_evaluation",
        rows=_integer(rows.get("interpolation_test"), "dataset.rows.interpolation_test"),
    )

    checks = acceptance.get("checks")
    if not isinstance(checks, list) or not all(
        isinstance(check, dict) for check in checks
    ):
        raise PlotError("acceptance.checks must be a list of objects")
    if [check.get("name") for check in checks] != [check[0] for check in CHECKS]:
        raise PlotError("acceptance check order or names do not match the frozen contract")
    for check, (name, metric, section, key, _) in zip(checks, CHECKS, strict=True):
        gate_table = _mapping(gates.get(section), f"acceptance config [{section}]")
        zero_gate = metric == "material_violations"
        limit = _gate_limit(
            gate_table.get(key),
            f"acceptance config {section}.{key}",
            zero=zero_gate,
        )
        _gate_limit(check.get("limit"), f"acceptance check {name} limit", zero=zero_gate)
        if check.get("limit") != gate_table.get(key):
            raise PlotError(f"acceptance check {name} changed its frozen limit")
        if metric == "material_violations":
            validation_value = validation["no_arbitrage"][metric]
            final_value = final["no_arbitrage"][metric]
        else:
            validation_value = validation["metrics"][metric]
            final_value = final["metrics"][metric]
        if check.get("validation_value") != validation_value:
            raise PlotError(f"acceptance check {name} disagrees with validation metrics")
        if check.get("final_value") != final_value:
            raise PlotError(f"acceptance check {name} disagrees with final metrics")
        if _number(validation_value, f"{name}.validation") > limit:
            raise PlotError(f"validation failed {name}")
        if _number(final_value, f"{name}.final") > limit:
            raise PlotError(f"locked final failed {name}")
    if acceptance.get("validation_passed") is not True:
        raise PlotError("acceptance.validation_passed must be true")
    if acceptance.get("final_passed") is not True:
        raise PlotError("acceptance.final_passed must be true")

    evidence = _mapping(results.get("evidence"), "evidence")
    archive = _mapping(evidence.get("archive"), "evidence.archive")
    if archive.get("file") != "european-replication-evidence-v1.tar.gz":
        raise PlotError("evidence.archive.file has an unexpected identity")
    _digest(archive.get("sha256"), "evidence.archive.sha256")
    if archive.get("tracked") is not False:
        raise PlotError("the external evidence archive must not be claimed as tracked")
    ledger = _mapping(evidence.get("execution_ledger"), "evidence.execution_ledger")
    if ledger.get("path") != "runs/european-neural-replication-v1/execution.json":
        raise PlotError("evidence.execution_ledger.path has an unexpected identity")
    _digest(ledger.get("sha256"), "evidence.execution_ledger.sha256")

    execution = _mapping(results.get("execution"), "execution")
    if execution.get("status") != "replication_passed":
        raise PlotError("execution.status must be replication_passed")
    if execution.get("final_evaluation_attempts") != 1:
        raise PlotError("execution must record exactly one final evaluation attempt")
    repository = _mapping(execution.get("repository"), "execution.repository")
    source_commit = _commit(
        repository.get("source_commit"),
        "execution.repository.source_commit",
    )
    remote_main = _commit(
        repository.get("remote_main"),
        "execution.repository.remote_main",
    )
    protocol_commit = _commit(
        repository.get("protocol_commit"),
        "execution.repository.protocol_commit",
    )
    if source_commit != remote_main:
        raise PlotError("source commit must equal the recorded remote main")
    if source_commit == protocol_commit:
        raise PlotError("protocol commit must precede the runner source commit")
    if repository.get("branch") != "main" or repository.get("clean_worktree") is not True:
        raise PlotError("replication must record a clean main worktree")
    runtime = _mapping(execution.get("runtime"), "execution.runtime")
    _digest(
        runtime.get("compiled_pricing_extension_sha256"),
        "execution.runtime.compiled_pricing_extension_sha256",
    )
    for field in ("python", "numpy", "pyarrow", "torch", "platform"):
        _string(runtime.get(field), f"execution.runtime.{field}")

    interpretation = _mapping(results.get("interpretation"), "interpretation")
    if interpretation.get("locked_final_evaluation_consumed") is not True:
        raise PlotError("interpretation must mark the locked final partition consumed")
    limitations = interpretation.get("limitations")
    if not isinstance(limitations, list) or len(limitations) < 5 or not all(
        isinstance(item, str) and item for item in limitations
    ):
        raise PlotError("interpretation must preserve the scoped non-claims")

    # Keep the variable live: the snapshot must bind exactly the checked-in
    # protocol bytes, not merely the configs named by that protocol.
    if protocol_digest != _sha256_file(EXPECTED_PROTOCOL):
        raise PlotError("protocol digest reconciliation failed")
    return results


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 14,
    weight: int = 400,
    anchor: str = "start",
    fill: str = "#172033",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">'
        f"{escape(value)}</text>"
    )


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str,
    width: float = 1.0,
    dash: str | None = None,
) -> str:
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width:.1f}"{dashed}/>'
    )


def _document(width: int, height: int, title: str, body: Sequence[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        f"<title>{escape(title)}</title>\n"
        "<style>text { font-family: Inter, ui-sans-serif, system-ui, sans-serif; }</style>\n"
        f'<rect width="{width}" height="{height}" rx="18" fill="#ffffff"/>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def _gate_margins(results: Mapping[str, Any]) -> str:
    width, height = 1100, 690
    left, right, top = 250.0, 100.0, 128.0
    chart_width = width - left - right
    row_height = 47.0
    checks = results["acceptance"]["checks"][:-1]
    body = [
        _text(36, 42, "Fresh-seed replication gate margins", size=25, weight=700),
        _text(
            36,
            70,
            "Observed error divided by the frozen limit; lower is better. "
            "Both partitions passed all gates.",
            size=14,
            fill="#536078",
        ),
        _line(720, 94, 746, 94, stroke="#2563eb", width=8),
        _text(754, 99, "Validation", size=12, fill="#354158"),
        _line(850, 94, 876, 94, stroke="#d97706", width=8),
        _text(884, 99, "Locked final", size=12, fill="#354158"),
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + chart_width * tick
        body.append(
            _line(
                x,
                top - 14,
                x,
                top + row_height * len(checks) - 8,
                stroke="#dc2626" if tick == 1.0 else "#dce2ea",
                width=1.5 if tick == 1.0 else 1.0,
                dash="7 6" if tick == 1.0 else None,
            )
        )
        body.append(
            _text(
                x,
                top - 24,
                f"{tick:.0%}",
                size=12,
                anchor="middle",
                fill="#b42318" if tick == 1.0 else "#657189",
            )
        )
    body.append(
        _text(left + chart_width - 8, top - 43, "Gate", size=12, anchor="end", fill="#b42318")
    )

    labels = {name: label for name, _, _, _, label in CHECKS}
    for index, check in enumerate(checks):
        y = top + index * row_height
        name = str(check.get("name"))
        limit = _gate_limit(check.get("limit"), f"acceptance check {name} limit", zero=False)
        validation_ratio = float(check["validation_value"]) / limit
        final_ratio = float(check["final_value"]) / limit
        body.append(
            _text(
                left - 16,
                y + 18,
                labels[name],
                size=13,
                weight=600 if name == "overall.gamma_rmse" else 400,
                anchor="end",
            )
        )
        body.append(
            f'<rect x="{left:.1f}" y="{y + 3:.1f}" '
            f'width="{chart_width * validation_ratio:.1f}" height="8" rx="4" '
            'fill="#2563eb"/>'
        )
        body.append(
            f'<rect x="{left:.1f}" y="{y + 16:.1f}" '
            f'width="{chart_width * final_ratio:.1f}" height="8" rx="4" '
            'fill="#d97706"/>'
        )
        body.append(
            _text(
                left + chart_width * final_ratio + 8,
                y + 25,
                f"{final_ratio:.1%}",
                size=11,
                fill="#9a5b06" if final_ratio < 0.9 else "#b42318",
            )
        )
    body.append(
        _text(
            36,
            height - 32,
            "Material European-bound violations: validation 0, locked final 0 "
            "(frozen limit: 0).",
            size=13,
            fill="#354158",
        )
    )
    return _document(width, height, "Fresh-seed replication gate margins", body)


def _projection_axis_maximum(peak: int, chart_height: float) -> int:
    """Smallest tick multiple that keeps the tallest bar and its label in the plot.

    The ceiling is derived from the displayed counts rather than hard-coded, and
    always leaves at least one tick of headroom so a bar can never fill the plot
    area and push its value label into the header.
    """
    if peak < 0:
        raise PlotError("projection counts must be non-negative")
    maximum = PROJECTION_TICK_STEP * (peak // PROJECTION_TICK_STEP + 1)
    while chart_height * peak / maximum > chart_height - PROJECTION_LABEL_RESERVE:
        maximum += PROJECTION_TICK_STEP
    return maximum


def _projection_intervention(results: Mapping[str, Any]) -> str:
    width, height = 1000, 570
    left, right, top, bottom = 95.0, 45.0, 115.0, 120.0
    chart_width = width - left - right
    chart_height = height - top - bottom
    groups = (
        ("Validation", results["results"]["validation"]),
        ("Locked final", results["results"]["locked_final"]),
    )
    series = (
        ("Unconstrained material violations", "unconstrained_material_violations", "#dc2626"),
        ("Rows projected to a bound", "active_rows", "#d97706"),
        ("Constrained material violations", None, "#059669"),
    )
    counts = {
        label: (
            int(result["projection"]["unconstrained_material_violations"]),
            int(result["projection"]["active_rows"]),
            int(result["no_arbitrage"]["material_violations"]),
        )
        for label, result in groups
    }
    maximum = _projection_axis_maximum(
        max(max(values) for values in counts.values()),
        chart_height,
    )
    body = [
        _text(36, 42, "European-bounds projection intervention", size=25, weight=700),
        _text(
            36,
            70,
            "The projection removes lower-bound violations without changing the "
            "trained weight bytes.",
            size=14,
            fill="#536078",
        ),
    ]
    legend_x = 155.0
    for index, (label, _, color) in enumerate(series):
        x = legend_x + index * 280.0
        body.append(f'<rect x="{x:.1f}" y="90" width="14" height="14" rx="3" fill="{color}"/>')
        body.append(_text(x + 21, 102, label, size=11, fill="#354158"))
    for tick in range(0, maximum + 1, PROJECTION_TICK_STEP):
        y = top + chart_height * (1.0 - tick / maximum)
        body.append(_line(left, y, width - right, y, stroke="#dce2ea"))
        body.append(
            _text(
                left - 12,
                y + 5,
                f"{tick:,}",
                size=12,
                anchor="end",
                fill="#657189",
            )
        )
    group_width = chart_width / len(groups)
    bar_width = 72.0
    gap = 22.0
    for group_index, (group_label, result) in enumerate(groups):
        projection = result["projection"]
        values = counts[group_label]
        total_width = len(values) * bar_width + (len(values) - 1) * gap
        group_left = left + group_width * group_index + (group_width - total_width) / 2
        for series_index, ((_, _, color), value) in enumerate(
            zip(series, values, strict=True)
        ):
            x = group_left + series_index * (bar_width + gap)
            raw_height = chart_height * value / maximum
            visible_height = max(raw_height, 3.0)
            y = top + chart_height - visible_height
            body.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{visible_height:.1f}" rx="7" fill="{color}"/>'
            )
            body.append(
                _text(
                    x + bar_width / 2,
                    y - 10,
                    f"{value:,}",
                    size=13,
                    weight=700,
                    anchor="middle",
                    fill=color,
                )
            )
        body.append(
            _text(
                left + group_width * (group_index + 0.5),
                top + chart_height + 35,
                group_label,
                size=14,
                weight=700,
                anchor="middle",
            )
        )
        active = int(projection["active_rows"])
        rows = int(result["rows"])
        body.append(
            _text(
                left + group_width * (group_index + 0.5),
                top + chart_height + 59,
                f"{active / rows:.2%} of rows projected",
                size=12,
                anchor="middle",
                fill="#657189",
            )
        )
    body.append(
        _text(
            width / 2,
            height - 24,
            "All activations were at the discounted European lower bound; "
            "upper-bound activations were zero.",
            size=12,
            anchor="middle",
            fill="#657189",
        )
    )
    return _document(width, height, "European-bounds projection intervention", body)


def render_figures(results_path: Path) -> dict[str, str]:
    """Validate the snapshot and return deterministic figure contents."""
    results = _load_json(results_path)
    validated = validate_snapshot(results)
    return {
        "european_replication_gate_margins.svg": _gate_margins(validated),
        "european_replication_projection.svg": _projection_intervention(validated),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if checked-in figures differ from a fresh deterministic render.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        figures = render_figures(arguments.results)
        if arguments.check:
            for name, content in figures.items():
                path = arguments.output_directory / name
                try:
                    current = path.read_text(encoding="utf-8")
                except OSError as error:
                    raise PlotError(f"cannot read checked-in figure '{path}': {error}") from error
                if current != content:
                    raise PlotError(
                        f"checked-in figure '{path}' is stale; rerun {Path(__file__).name}"
                    )
        else:
            arguments.output_directory.mkdir(parents=True, exist_ok=True)
            for name, content in figures.items():
                (arguments.output_directory / name).write_text(
                    content,
                    encoding="utf-8",
                )
    except PlotError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
