#!/usr/bin/env python3
"""Strictly validate Task 9G reports and freeze/check tracked evidence.

Neither mode imports a pricing engine or performs pricing, training, timing, or
IV inversion. Freeze reads reviewed raw JSON; check reads only tracked files.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL: Final = PROJECT_ROOT / "configs/american_neural_pilot_protocol_v1.toml"
DEFAULT_SNAPSHOT: Final = PROJECT_ROOT / "docs/results/american_neural_pilot_results_v1.json"
SNAPSHOT_SCHEMA: Final = "american-neural-pilot-result/1"
ARMS: Final = ("scratch", "transfer")
MODEL_NAMES: Final = frozenset({*ARMS, "european_crr_baseline"})
METRIC_KEYS: Final = frozenset(
    {
        "rows",
        "mae",
        "rmse",
        "p95_absolute_error",
        "p99_absolute_error",
        "maximum_absolute_error",
    }
)
GATE_CHECKS: Final = frozenset(
    {
        "normalized_rmse",
        "normalized_p99_absolute_error",
        "normalized_maximum_absolute_error",
        "material_bound_violations",
        "material_shape_violations",
    }
)
DIAGNOSTIC_CHECKS: Final = frozenset(
    {
        "intrinsic_lower_bound",
        "european_comparator_lower_bound",
        "american_call_upper_bound",
        "american_put_rate_aware_upper_bound",
        "spot_monotonicity",
        "spot_convexity",
        "volatility_monotonicity",
    }
)
VALIDATION_KEYS: Final = frozenset(
    {
        "schema_version",
        "lifecycle",
        "protocol_sha256",
        "dataset",
        "source_artifact",
        "exact_lift",
        "independent_pde_check",
        "selected_train_rows",
        "artifacts",
        "validation",
        "validation_evidence_complete",
        "latency",
        "implied_volatility",
        "validation_final_entry_passed",
        "runtime",
        "interpretation",
    }
)
ENTRY_KEYS: Final = frozenset(
    {
        "schema_version",
        "lifecycle",
        "protocol_sha256",
        "dataset",
        "source_artifact",
        "failed_gate",
        "evidence",
        "runtime",
        "interpretation",
    }
)
FINAL_KEYS: Final = frozenset(
    {
        "schema_version",
        "protocol_sha256",
        "attempt",
        "partition",
        "partition_sha256",
        "models_and_baseline",
        "all_arms_passed",
        "outcome",
        "runtime",
        "lifecycle",
    }
)
FINAL_FAILURE_KEYS: Final = frozenset(
    {
        "schema_version",
        "protocol_sha256",
        "attempt",
        "partition",
        "expected_partition_sha256",
        "observed_partition_sha256",
        "failure_stage",
        "error",
        "runtime",
        "lifecycle",
        "interpretation",
    }
)
SNAPSHOT_KEYS: Final = frozenset(
    {
        "schema_version",
        "protocol",
        "source_reports",
        "dataset",
        "source_artifact",
        "lifecycle",
        "audit",
        "entry_failure",
        "exact_lift",
        "independent_pde_check",
        "arms",
        "baseline",
        "latency",
        "implied_volatility",
        "outcome",
        "interpretation",
    }
)
RUNTIME_KEYS: Final = frozenset(
    {
        "python",
        "numpy",
        "torch",
        "platform",
        "machine",
        "processor",
        "logical_cpu_count",
        "cpu_affinity",
        "core_version",
        "build_configuration",
        "cxx_compiler",
    }
)


class FreezeError(RuntimeError):
    """Raised on schema, provenance, lifecycle, or recomputation mismatch."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], where: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise FreezeError(f"{where} keys differ: missing={missing}, unknown={unknown}")


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FreezeError(f"{where} must be an object")
    return value


def _sequence(value: Any, where: str) -> Sequence[Any]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise FreezeError(f"{where} must be an array")
    return value


def _digest(value: Any, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FreezeError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _finite_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise FreezeError(f"{where} must be finite numeric")
    return float(value)


def _finite_tree(value: Any, where: str = "report") -> None:
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FreezeError(f"{where} contains a non-finite value")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise FreezeError(f"{where} contains a non-string key")
            _finite_tree(item, f"{where}.{key}")
        return
    if isinstance(value, Sequence):
        for index, item in enumerate(value):
            _finite_tree(item, f"{where}[{index}]")
        return
    raise FreezeError(f"{where} contains unsupported type {type(value).__name__}")


def _load_json(path: Path, where: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FreezeError(f"cannot load {where} '{path}': {error}") from error
    if not isinstance(value, dict):
        raise FreezeError(f"{where} must be an object")
    return value


def _load_toml(path: Path, where: str) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            value = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise FreezeError(f"cannot load {where} '{path}': {error}") from error
    return value


def _validate_protocol(protocol_path: Path, root: Path) -> dict[str, Any]:
    validator_path = root / "scripts/check_american_neural_pilot_protocol.py"
    spec = importlib.util.spec_from_file_location("task9g_freeze_protocol", validator_path)
    if spec is None or spec.loader is None:
        raise FreezeError("cannot load Task 9G protocol validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        module.validate_protocol(protocol_path, root)
    except Exception as error:
        raise FreezeError(f"protocol validation failed: {error}") from error
    return _load_toml(protocol_path, "protocol")


def _leaves(protocol: Mapping[str, Any], root: Path) -> tuple[dict, dict, dict]:
    paths = protocol["paths"]
    return (
        _load_toml(root / paths["acceptance_config"], "acceptance config"),
        _load_toml(root / paths["latency_config"], "latency config"),
        _load_toml(root / paths["iv_config"], "IV config"),
    )


def _slice_names(acceptance: Mapping[str, Any]) -> frozenset[str]:
    names = {"overall", "option_type:call", "option_type:put"}
    for family, config_key in (
        ("expiry", "expiry_edges_years"),
        ("moneyness", "log_moneyness_edges"),
        ("volatility", "volatility_edges"),
    ):
        edges = [-math.inf, *acceptance["bins"][config_key], math.inf]
        names.update(f"{family}:{low:g}:{high:g}" for low, high in pairwise(edges))
    names.update(
        {
            "premium_status:zero",
            "premium_status:positive",
            "exercise_status:no_exercise",
            "exercise_status:exercise_observed",
        }
    )
    return frozenset(names)


def _metric(value: Any, where: str, rows: int) -> None:
    metric = _mapping(value, where)
    _exact_keys(metric, METRIC_KEYS, where)
    if metric["rows"] != rows:
        raise FreezeError(f"{where}.rows differs from its slice")
    for name in METRIC_KEYS - {"rows"}:
        if _finite_number(metric[name], f"{where}.{name}") < 0.0:
            raise FreezeError(f"{where}.{name} must be non-negative")


def _slices(value: Any, acceptance: Mapping[str, Any], where: str) -> None:
    slices = _mapping(value, where)
    _exact_keys(slices, _slice_names(acceptance), where)
    for name, raw in slices.items():
        entry = _mapping(raw, f"{where}.{name}")
        _exact_keys(entry, frozenset({"rows", "physical", "normalized"}), f"{where}.{name}")
        rows = entry["rows"]
        if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
            raise FreezeError(f"{where}.{name}.rows is invalid")
        if rows == 0:
            if entry["physical"] is not None or entry["normalized"] is not None:
                raise FreezeError(f"{where}.{name} empty slice must carry null metrics")
        else:
            _metric(entry["physical"], f"{where}.{name}.physical", rows)
            _metric(entry["normalized"], f"{where}.{name}.normalized", rows)


AUDIT_ROW_KEYS: Final = frozenset(
    {
        "sample_id",
        "option_type",
        "spot",
        "strike",
        "maturity",
        "dividend_yield",
        "volatility",
        "early_exercise_premium",
        "earliest_exercise_step",
        "reference_price",
        "prediction",
        "diagnostic_tolerance",
        "diagnostic_violations",
    }
)


def _linear_quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _statistics(reference: list[float], prediction: list[float]) -> dict[str, Any]:
    errors = [abs(left - right) for left, right in zip(reference, prediction, strict=True)]
    return {
        "rows": len(errors),
        "mae": sum(errors) / len(errors),
        "rmse": math.sqrt(sum(value * value for value in errors) / len(errors)),
        "p95_absolute_error": _linear_quantile(errors, 0.95),
        "p99_absolute_error": _linear_quantile(errors, 0.99),
        "maximum_absolute_error": max(errors),
    }


def _close_statistics(actual: Mapping[str, Any], expected: Mapping[str, Any], where: str) -> None:
    if actual["rows"] != expected["rows"]:
        raise FreezeError(f"{where} row-count recomputation mismatch")
    for name in METRIC_KEYS - {"rows"}:
        if not math.isclose(actual[name], expected[name], rel_tol=2.0e-14, abs_tol=1.0e-15):
            raise FreezeError(f"{where}.{name} recomputation mismatch")


def _row_in_slice(row: Mapping[str, Any], name: str, acceptance: Mapping[str, Any]) -> bool:
    if name == "overall":
        return True
    family, label = name.split(":", 1)
    if family == "option_type":
        return row["option_type"] == label
    if family == "premium_status":
        return (
            (row["early_exercise_premium"] == 0.0)
            if label == "zero"
            else (row["early_exercise_premium"] > 0.0)
        )
    if family == "exercise_status":
        return (
            (row["earliest_exercise_step"] < 0)
            if label == "no_exercise"
            else (row["earliest_exercise_step"] >= 0)
        )
    low_text, high_text = label.split(":", 1)
    low, high = float(low_text), float(high_text)
    if family == "expiry":
        value = row["maturity"]
    elif family == "moneyness":
        value = math.log(row["spot"] / row["strike"])
    else:
        value = row["volatility"]
    return low <= value < high


def _audit_rows(value: Any, where: str, *, baseline: bool) -> list[Mapping[str, Any]]:
    rows = list(_sequence(value, where))
    if not rows:
        raise FreezeError(f"{where} must contain underlying rows")
    seen = set()
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"{where}[{index}]")
        _exact_keys(row, AUDIT_ROW_KEYS, f"{where}[{index}]")
        if not isinstance(row["sample_id"], str) or row["sample_id"] in seen:
            raise FreezeError(f"{where} sample IDs must be unique strings")
        seen.add(row["sample_id"])
        if row["option_type"] not in {"call", "put"}:
            raise FreezeError(f"{where} option type differs")
        for name in AUDIT_ROW_KEYS - {
            "sample_id",
            "option_type",
            "earliest_exercise_step",
            "diagnostic_violations",
        }:
            if name == "diagnostic_tolerance" and baseline:
                if row[name] is not None:
                    raise FreezeError(f"{where} baseline tolerance must be null")
            else:
                _finite_number(row[name], f"{where}.{name}")
        if isinstance(row["earliest_exercise_step"], bool) or not isinstance(
            row["earliest_exercise_step"], int
        ):
            raise FreezeError(f"{where} exercise step differs")
        if baseline:
            if row["diagnostic_violations"] is not None:
                raise FreezeError(f"{where} baseline diagnostic audit must be null")
        else:
            violations = _mapping(row["diagnostic_violations"], f"{where}.violations")
            _exact_keys(violations, DIAGNOSTIC_CHECKS, f"{where}.violations")
            for violation in violations.values():
                if violation is not None:
                    _finite_number(violation, f"{where}.violation")
    return rows


def _recompute_slices(
    reported: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    acceptance: Mapping[str, Any],
    where: str,
) -> None:
    for name in _slice_names(acceptance):
        selected = [row for row in rows if _row_in_slice(row, name, acceptance)]
        actual = reported[name]
        if actual["rows"] != len(selected):
            raise FreezeError(f"{where}.{name} membership recomputation mismatch")
        if not selected:
            continue
        reference = [row["reference_price"] for row in selected]
        prediction = [row["prediction"] for row in selected]
        _close_statistics(
            actual["physical"], _statistics(reference, prediction), f"{where}.{name}.physical"
        )
        normalization = [
            row["spot"] * math.exp(-row["dividend_yield"] * row["maturity"]) for row in selected
        ]
        _close_statistics(
            actual["normalized"],
            _statistics(
                [value / scale for value, scale in zip(reference, normalization, strict=True)],
                [value / scale for value, scale in zip(prediction, normalization, strict=True)],
            ),
            f"{where}.{name}.normalized",
        )


def _diagnostics(value: Any, audit_rows: list[Mapping[str, Any]], where: str) -> tuple[int, int]:
    diagnostics = _mapping(value, where)
    _exact_keys(
        diagnostics,
        frozenset(
            {
                "checks",
                "material_bound_violations",
                "material_shape_violations",
                "american_greek_accuracy_claim",
            }
        ),
        where,
    )
    checks = _mapping(diagnostics["checks"], f"{where}.checks")
    _exact_keys(checks, DIAGNOSTIC_CHECKS, f"{where}.checks")
    counts = {}
    for name, raw in checks.items():
        detail = _mapping(raw, f"{where}.checks.{name}")
        _exact_keys(
            detail,
            frozenset({"material_violations", "maximum_violation", "eligible_rows"}),
            f"{where}.checks.{name}",
        )
        for key in ("material_violations", "eligible_rows"):
            if isinstance(detail[key], bool) or not isinstance(detail[key], int) or detail[key] < 0:
                raise FreezeError(f"{where}.checks.{name}.{key} is invalid")
        if detail["material_violations"] > detail["eligible_rows"]:
            raise FreezeError(f"{where}.checks.{name} violation count exceeds eligible rows")
        if _finite_number(detail["maximum_violation"], where) < 0.0:
            raise FreezeError(f"{where}.checks.{name}.maximum_violation is negative")
        raw_values = [
            (row["diagnostic_violations"][name], row["diagnostic_tolerance"])
            for row in audit_rows
            if row["diagnostic_violations"][name] is not None
        ]
        expected_count = sum(value > tolerance for value, tolerance in raw_values)
        expected_maximum = max((max(value, 0.0) for value, _ in raw_values), default=0.0)
        if (
            detail["eligible_rows"] != len(raw_values)
            or detail["material_violations"] != expected_count
            or not math.isclose(
                detail["maximum_violation"], expected_maximum, rel_tol=2.0e-14, abs_tol=1.0e-15
            )
        ):
            raise FreezeError(f"{where}.checks.{name} audit recomputation mismatch")
        counts[name] = expected_count
    bound = sum(
        counts[name]
        for name in (
            "intrinsic_lower_bound",
            "european_comparator_lower_bound",
            "american_call_upper_bound",
            "american_put_rate_aware_upper_bound",
        )
    )
    shape = sum(
        counts[name] for name in ("spot_monotonicity", "spot_convexity", "volatility_monotonicity")
    )
    if diagnostics["material_bound_violations"] != bound:
        raise FreezeError(f"{where} bound-count recomputation mismatch")
    if diagnostics["material_shape_violations"] != shape:
        raise FreezeError(f"{where} shape-count recomputation mismatch")
    if diagnostics["american_greek_accuracy_claim"] is not False:
        raise FreezeError(f"{where} must not claim American Greek accuracy")
    return bound, shape


def _model_entry(
    value: Any,
    acceptance: Mapping[str, Any],
    section: str,
    where: str,
    *,
    baseline: bool = False,
) -> bool | None:
    entry = _mapping(value, where)
    expected = (
        {"slices", "diagnostics", "audit_rows"}
        if baseline
        else {"slices", "diagnostics", "gate", "audit_rows"}
    )
    _exact_keys(entry, frozenset(expected), where)
    _slices(entry["slices"], acceptance, f"{where}.slices")
    audit = _audit_rows(entry["audit_rows"], f"{where}.audit_rows", baseline=baseline)
    _recompute_slices(entry["slices"], audit, acceptance, f"{where}.slices")
    if baseline:
        diagnostics = _mapping(entry["diagnostics"], f"{where}.diagnostics")
        _exact_keys(
            diagnostics,
            frozenset({"not_applicable", "stored_error_equals_early_exercise_premium"}),
            f"{where}.diagnostics",
        )
        if diagnostics["stored_error_equals_early_exercise_premium"] is not True:
            raise FreezeError(f"{where} baseline identity check failed")
        if not all(
            row["reference_price"] - row["prediction"] == row["early_exercise_premium"]
            for row in audit
        ):
            raise FreezeError(f"{where} baseline row identity recomputation mismatch")
        return None
    bound, shape = _diagnostics(entry["diagnostics"], audit, f"{where}.diagnostics")
    gate = _mapping(entry["gate"], f"{where}.gate")
    _exact_keys(gate, frozenset({"checks", "passed"}), f"{where}.gate")
    checks = _mapping(gate["checks"], f"{where}.gate.checks")
    _exact_keys(checks, GATE_CHECKS, f"{where}.gate.checks")
    overall = entry["slices"]["overall"]["normalized"]
    limits = acceptance[section]
    recomputed = {
        "normalized_rmse": overall["rmse"] <= limits["normalized_rmse_max"],
        "normalized_p99_absolute_error": overall["p99_absolute_error"]
        <= limits["normalized_p99_absolute_error_max"],
        "normalized_maximum_absolute_error": overall["maximum_absolute_error"]
        <= limits["normalized_maximum_absolute_error_max"],
        "material_bound_violations": bound <= limits["maximum_material_bound_violations"],
        "material_shape_violations": shape <= limits["maximum_material_shape_violations"],
    }
    if dict(checks) != recomputed or gate["passed"] is not all(recomputed.values()):
        raise FreezeError(f"{where} gate recomputation mismatch")
    return bool(gate["passed"])


def _models(value: Any, acceptance: Mapping[str, Any], section: str, where: str) -> dict[str, bool]:
    models = _mapping(value, where)
    _exact_keys(models, MODEL_NAMES, where)
    passed = {
        arm: bool(_model_entry(models[arm], acceptance, section, f"{where}.{arm}")) for arm in ARMS
    }
    _model_entry(
        models["european_crr_baseline"],
        acceptance,
        section,
        f"{where}.european_crr_baseline",
        baseline=True,
    )
    common_keys = AUDIT_ROW_KEYS - {
        "prediction",
        "diagnostic_tolerance",
        "diagnostic_violations",
    }
    scratch_rows = list(models["scratch"]["audit_rows"])
    scratch_common = [{key: row[key] for key in common_keys} for row in scratch_rows]
    for name in ("transfer", "european_crr_baseline"):
        rows = list(models[name]["audit_rows"])
        if (
            len(rows) != len(scratch_rows)
            or [{key: row[key] for key in common_keys} for row in rows] != scratch_common
        ):
            raise FreezeError(f"{where} model audit rows differ across arms")
    return passed


def _runtime(value: Any, where: str) -> None:
    runtime = _mapping(value, where)
    _exact_keys(runtime, RUNTIME_KEYS, where)


def _exact_lift(value: Any, where: str) -> bool:
    evidence = _mapping(value, where)
    _exact_keys(
        evidence,
        frozenset(
            {
                "passed",
                "probes",
                "probe_prices",
                "maximum_absolute_difference",
                "maximum_relative_difference",
                "relative_tolerance",
                "absolute_tolerance",
            }
        ),
        where,
    )
    probes = _sequence(evidence["probe_prices"], f"{where}.probe_prices")
    if evidence["probes"] != 8 or len(probes) != 8:
        raise FreezeError(f"{where} probe count differs")
    differences = []
    relatives = []
    comparisons = []
    for index, raw in enumerate(probes):
        probe = _mapping(raw, f"{where}.probe_prices[{index}]")
        _exact_keys(
            probe,
            frozenset({"source", "lifted"}),
            f"{where}.probe_prices[{index}]",
        )
        source = _finite_number(probe["source"], f"{where}.source")
        lifted = _finite_number(probe["lifted"], f"{where}.lifted")
        difference = abs(source - lifted)
        differences.append(difference)
        relatives.append(difference / max(abs(source), sys.float_info.min))
        comparisons.append(
            difference
            <= evidence["absolute_tolerance"] + evidence["relative_tolerance"] * abs(source)
        )
    if not math.isclose(
        evidence["maximum_absolute_difference"],
        max(differences),
        rel_tol=1e-15,
        abs_tol=0.0,
    ) or not math.isclose(
        evidence["maximum_relative_difference"],
        max(relatives),
        rel_tol=1e-15,
        abs_tol=0.0,
    ):
        raise FreezeError(f"{where} maxima recomputation mismatch")
    passed = (
        evidence["relative_tolerance"] <= 1.0e-12
        and evidence["absolute_tolerance"] <= 1.0e-12
        and all(comparisons)
    )
    if evidence["passed"] is not passed:
        raise FreezeError(f"{where} recomputation mismatch")
    return passed


PDE_RESULT_KEYS: Final = frozenset(
    {
        "price",
        "solver_status",
        "discretization_accuracy",
        "spot_intervals",
        "spot_maximum",
        "spot_step",
        "strike_node_index",
        "time_steps",
        "rannacher_steps",
        "damped_half_steps",
        "crank_nicolson_steps",
        "upwinded_rows",
        "linear_solves",
        "psor_solves",
        "psor_total_iterations",
        "psor_maximum_iterations_used",
        "psor_tolerance",
        "psor_relaxation",
        "maximum_lcp_residual",
        "maximum_relative_lcp_residual",
        "aligned_times",
        "dividend_events",
    }
)
PDE_INPUT_KEYS: Final = frozenset(
    {
        "option_type",
        "exercise_style",
        "spot",
        "strike",
        "valuation_time",
        "expiry_time",
        "volatility",
        "continuous_carry",
        "curve_times",
        "curve_log_discounts",
        "dividends",
        "settlement",
        "contract_multiplier",
        "spot_intervals",
        "time_steps",
        "spot_maximum",
        "rannacher_steps",
        "psor_tolerance",
        "psor_relaxation",
        "psor_maximum_iterations",
    }
)


def _pde(value: Any, protocol: Mapping[str, Any], where: str) -> bool:
    evidence = _mapping(value, where)
    _exact_keys(evidence, frozenset({"rows", "passed", "interpretation"}), where)
    rows = _sequence(evidence["rows"], f"{where}.rows")
    spec = protocol["independent_pde_check"]
    if evidence["interpretation"] != spec["interpretation"]:
        raise FreezeError(f"{where} interpretation differs")
    cases = {case["sample_id"]: case for case in spec["cases"]}
    if len(rows) != len(cases):
        raise FreezeError(f"{where} case count differs")
    row_passes = []
    seen_samples = set()
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"{where}.rows[{index}]")
        _exact_keys(
            row,
            frozenset(
                {
                    "sample_id",
                    "coarse_price",
                    "fine_price",
                    "crr_label",
                    "solves",
                    "normalized_refinement_difference",
                    "normalized_pde_crr_difference",
                    "passed",
                }
            ),
            f"{where}.rows[{index}]",
        )
        case = cases.get(row["sample_id"])
        if case is None or row["sample_id"] in seen_samples:
            raise FreezeError(f"{where} contains an unpinned sample")
        seen_samples.add(row["sample_id"])
        solves = _sequence(row["solves"], f"{where}.rows[{index}].solves")
        if len(solves) != 2:
            raise FreezeError(f"{where} must retain both PDE grids")
        prices = []
        for solve_index, (raw_solve, grid) in enumerate(zip(solves, spec["grids"], strict=True)):
            solve = _mapping(raw_solve, f"{where}.rows[{index}].solves[{solve_index}]")
            _exact_keys(solve, frozenset({"grid", "requested_input", "result"}), where)
            if solve["grid"] != grid["name"]:
                raise FreezeError(f"{where} PDE grid order differs")
            requested = _mapping(solve["requested_input"], f"{where}.requested_input")
            _exact_keys(requested, PDE_INPUT_KEYS, f"{where}.requested_input")
            expected = {
                "option_type": case["option_type"],
                "exercise_style": "american",
                "spot": case["spot"],
                "strike": case["strike"],
                "valuation_time": spec["valuation_time"],
                "expiry_time": case["maturity"],
                "volatility": case["volatility"],
                "continuous_carry": case["dividend_yield"],
                "curve_times": [0.0, case["maturity"]],
                "curve_log_discounts": [0.0, -case["rate"] * case["maturity"]],
                "dividends": [],
                "settlement": spec["settlement"],
                "contract_multiplier": spec["contract_multiplier"],
                "spot_intervals": grid["spot_intervals"],
                "time_steps": grid["time_steps"],
                "spot_maximum": grid["spot_maximum_factor"] * max(case["spot"], case["strike"]),
                "rannacher_steps": spec["rannacher_steps"],
                "psor_tolerance": spec["psor_tolerance"],
                "psor_relaxation": spec["psor_relaxation"],
                "psor_maximum_iterations": spec["psor_maximum_iterations"],
            }
            if dict(requested) != expected:
                raise FreezeError(f"{where} PDE input echo drift")
            result = _mapping(solve["result"], f"{where}.result")
            _exact_keys(result, PDE_RESULT_KEYS, f"{where}.result")
            if (
                result["solver_status"] != "discrete_system_converged"
                or result["discretization_accuracy"] != "not_assessed"
                or result["time_steps"] != requested["time_steps"]
                or result["rannacher_steps"] != requested["rannacher_steps"]
                or result["psor_tolerance"] != requested["psor_tolerance"]
                or result["psor_relaxation"] != requested["psor_relaxation"]
                or result["aligned_times"] != [0.0, case["maturity"]]
                or result["dividend_events"] != []
            ):
                raise FreezeError(f"{where} PDE effective diagnostics drift")
            if not math.isclose(
                result["spot_intervals"] * result["spot_step"],
                result["spot_maximum"],
                rel_tol=1.0e-14,
                abs_tol=1.0e-14,
            ) or not math.isclose(
                result["strike_node_index"] * result["spot_step"],
                case["strike"],
                rel_tol=1.0e-14,
                abs_tol=1.0e-14,
            ):
                raise FreezeError(f"{where} PDE actual grid is malformed")
            target_step = requested["spot_maximum"] / requested["spot_intervals"]
            expected_strike_index = max(1, math.floor(case["strike"] / target_step + 0.5))
            expected_step = case["strike"] / expected_strike_index
            expected_intervals = max(
                expected_strike_index + 1,
                math.ceil((requested["spot_maximum"] / expected_step) - 1.0e-9),
            )
            if (
                result["spot_intervals"] <= 1
                or result["strike_node_index"] <= 0
                or result["strike_node_index"] >= result["spot_intervals"]
                or result["spot_step"] <= 0.0
                or result["spot_maximum"] <= max(case["spot"], case["strike"])
                or result["strike_node_index"] != expected_strike_index
                or result["spot_intervals"] != expected_intervals
                or result["spot_step"] != expected_step
                or result["spot_maximum"] != expected_intervals * expected_step
            ):
                raise FreezeError(f"{where} PDE strike-aligned grid transformation differs")
            count_names = (
                "damped_half_steps",
                "crank_nicolson_steps",
                "upwinded_rows",
                "linear_solves",
                "psor_solves",
                "psor_total_iterations",
                "psor_maximum_iterations_used",
            )
            if any(
                isinstance(result[name], bool)
                or not isinstance(result[name], int)
                or result[name] < 0
                for name in count_names
            ):
                raise FreezeError(f"{where} PDE solve counts are malformed")
            if (
                result["linear_solves"] != 0
                or result["psor_solves"]
                != result["damped_half_steps"] + result["crank_nicolson_steps"]
                or result["psor_total_iterations"] < result["psor_solves"]
                or result["psor_maximum_iterations_used"] > requested["psor_maximum_iterations"]
            ):
                raise FreezeError(f"{where} PDE solve accounting differs")
            for residual_name in ("maximum_lcp_residual", "maximum_relative_lcp_residual"):
                if _finite_number(result[residual_name], f"{where}.{residual_name}") < 0.0:
                    raise FreezeError(f"{where} PDE residual is negative")
            if result["maximum_relative_lcp_residual"] > requested["psor_tolerance"]:
                raise FreezeError(f"{where} converged PDE residual exceeds tolerance")
            price = _finite_number(result["price"], f"{where}.price")
            if price < 0.0:
                raise FreezeError(f"{where} PDE price is negative")
            prices.append(price)
        if row["coarse_price"] != prices[0] or row["fine_price"] != prices[1]:
            raise FreezeError(f"{where} PDE price echo mismatch")
        scale = case["spot"] * math.exp(-case["dividend_yield"] * case["maturity"])
        refinement = abs(prices[1] - prices[0]) / scale
        agreement = abs(prices[1] - row["crr_label"]) / scale
        if not math.isclose(row["normalized_refinement_difference"], refinement, rel_tol=1e-15):
            raise FreezeError(f"{where} PDE refinement recomputation mismatch")
        if not math.isclose(row["normalized_pde_crr_difference"], agreement, rel_tol=1e-15):
            raise FreezeError(f"{where} PDE agreement recomputation mismatch")
        passed = (
            refinement <= spec["maximum_normalized_refinement_difference"]
            and agreement <= spec["maximum_normalized_pde_crr_difference"]
        )
        if row["passed"] is not passed:
            raise FreezeError(f"{where} PDE row gate recomputation mismatch")
        row_passes.append(passed)
    passed = bool(row_passes) and all(row_passes)
    if seen_samples != set(cases):
        raise FreezeError(f"{where} pinned sample set differs")
    if evidence["passed"] is not passed:
        raise FreezeError(f"{where} PDE aggregate gate recomputation mismatch")
    return passed


LATENCY_ROW_KEYS: Final = frozenset(
    {
        "shape",
        "batch_size",
        "thread_budget",
        "crr_depth",
        "economic_requests",
        "observed_measurement_orders",
        "crr_adjacent_average_raw_ns",
        "neural_end_to_end_raw_ns",
        "median_speedup",
        "median_speedup_confidence_interval",
        "median_speedup_confidence_level_at_least",
        "median_speedup_interval_actual_coverage",
    }
)


def _latency(value: Any, config: Mapping[str, Any], where: str) -> bool:
    evidence = _mapping(value, where)
    _exact_keys(evidence, frozenset({"runtime", "measurements"}), where)
    _runtime(evidence["runtime"], f"{where}.runtime")
    rows = _sequence(evidence["measurements"], f"{where}.measurements")
    expected_cells = {
        (shape, batch, threads, depth)
        for shape, batch, threads in zip(
            config["request_shapes"], config["batch_sizes"], config["thread_budgets"], strict=True
        )
        for depth in config["crr_depths"]
    }
    seen = set()
    operations = ["crr", *ARMS]
    expected_orders = []
    for repetition in range(config["warmups"], config["warmups"] + config["repetitions"]):
        rotation = repetition % len(operations)
        expected_orders.append(operations[rotation:] + operations[:rotation])
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"{where}.measurements[{index}]")
        _exact_keys(row, LATENCY_ROW_KEYS, f"{where}.measurements[{index}]")
        cell = (row["shape"], row["batch_size"], row["thread_budget"], row["crr_depth"])
        if cell not in expected_cells or cell in seen:
            raise FreezeError(f"{where} latency cell set differs")
        seen.add(cell)
        cases_by_name = {case["name"]: case for case in config["cases"]}
        expected_requests = [
            cases_by_name[name] for name in config["request_case_names"][row["shape"]]
        ]
        if row["economic_requests"] != expected_requests:
            raise FreezeError(f"{where} economic request echo differs")
        if row["observed_measurement_orders"] != expected_orders:
            raise FreezeError(f"{where} cyclic measurement order differs")
        crr = _sequence(row["crr_adjacent_average_raw_ns"], f"{where}.crr_raw")
        neural = _mapping(row["neural_end_to_end_raw_ns"], f"{where}.neural_raw")
        _exact_keys(neural, frozenset(ARMS), f"{where}.neural_raw")
        if len(crr) != config["repetitions"]:
            raise FreezeError(f"{where} CRR repetition count differs")
        medians = _mapping(row["median_speedup"], f"{where}.median_speedup")
        intervals = _mapping(row["median_speedup_confidence_interval"], f"{where}.median_interval")
        _exact_keys(medians, frozenset(ARMS), f"{where}.median_speedup")
        _exact_keys(intervals, frozenset(ARMS), f"{where}.median_interval")
        for arm in ARMS:
            values = _sequence(neural[arm], f"{where}.neural_raw.{arm}")
            if len(values) != config["repetitions"]:
                raise FreezeError(f"{where} neural repetition count differs")
            if any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in [*crr, *values]
            ):
                raise FreezeError(f"{where} timings must be positive integer nanoseconds")
            ratios = [left / right for left, right in zip(crr, values, strict=True)]
            if not math.isclose(medians[arm], statistics.median(ratios), rel_tol=1e-15):
                raise FreezeError(f"{where} median speedup recomputation mismatch")
            if list(intervals[arm]) != [min(ratios), max(ratios)]:
                raise FreezeError(f"{where} median interval recomputation mismatch")
        if (
            row["median_speedup_confidence_level_at_least"] != 0.95
            or row["median_speedup_interval_actual_coverage"] != 0.984375
        ):
            raise FreezeError(f"{where} median uncertainty contract differs")
    return seen == expected_cells


def _iv(value: Any, config: Mapping[str, Any], where: str) -> tuple[bool, int]:
    evidence = _mapping(value, where)
    _exact_keys(evidence, frozenset({"terminology", "rows", "failures"}), where)
    if evidence["terminology"] != config["terminology"]:
        raise FreezeError(f"{where} terminology differs")
    rows = _sequence(evidence["rows"], f"{where}.rows")
    expected_names = [case["name"] for case in config["cases"]]
    if [row.get("case") for row in rows if isinstance(row, Mapping)] != expected_names:
        raise FreezeError(f"{where} case set/order differs")
    failures = 0
    for index, raw in enumerate(rows):
        row = _mapping(raw, f"{where}.rows[{index}]")
        _exact_keys(row, frozenset({"case", "prices", "inversions"}), f"{where}.rows[{index}]")
        prices = _mapping(row["prices"], f"{where}.prices")
        inversions = _mapping(row["inversions"], f"{where}.inversions")
        expected = frozenset({"label", *ARMS})
        _exact_keys(prices, expected, f"{where}.prices")
        _exact_keys(inversions, expected, f"{where}.inversions")
        for name, price in prices.items():
            _finite_number(price, f"{where}.prices.{name}")
        label_iv = inversions["label"].get("volatility")
        for name, raw_inversion in inversions.items():
            inversion = _mapping(raw_inversion, f"{where}.inversions.{name}")
            _exact_keys(
                inversion,
                frozenset({"status", "volatility", "iterations", "absolute_error_vs_label_iv"}),
                f"{where}.inversions.{name}",
            )
            if inversion["status"] not in {
                "converged",
                "unbracketed",
                "iteration_exhausted",
                "non_finite_price",
            }:
                raise FreezeError(f"{where} IV status differs")
            iterations = inversion["iterations"]
            if (
                isinstance(iterations, bool)
                or not isinstance(iterations, int)
                or not 0 <= iterations <= config["maximum_iterations"]
                or (
                    inversion["status"] == "iteration_exhausted"
                    and iterations != config["maximum_iterations"]
                )
                or (inversion["status"] == "unbracketed" and iterations != 0)
            ):
                raise FreezeError(f"{where} IV iteration evidence differs")
            converged = inversion["status"] == "converged"
            if converged:
                volatility = _finite_number(inversion["volatility"], f"{where}.volatility")
                if (
                    not config["volatility_bracket"][0]
                    <= volatility
                    <= config["volatility_bracket"][1]
                ):
                    raise FreezeError(f"{where} IV result is outside bracket")
            elif inversion["volatility"] is not None:
                raise FreezeError(f"{where} failed IV must not report volatility")
            if not converged:
                failures += 1
            expected_error = (
                None
                if label_iv is None or inversion["volatility"] is None
                else abs(inversion["volatility"] - label_iv)
            )
            if inversion["absolute_error_vs_label_iv"] != expected_error:
                raise FreezeError(f"{where} IV error recomputation mismatch")
    if evidence["failures"] != failures:
        raise FreezeError(f"{where} IV failure recomputation mismatch")
    return len(rows) == len(config["cases"]), failures


OUTCOME_KEYS: Final = frozenset(
    {
        "phase",
        "outcome",
        "arm_accuracy_passed",
        "reference_depth",
        "reference_median_speedups",
        "all_reference_speedups_passed",
        "all_iv_errors_passed",
        "transfer_validation_rmse_strictly_better",
    }
)


def _outcome(
    accuracy: Mapping[str, Any],
    validation: Mapping[str, Any],
    latency: Mapping[str, Any],
    iv: Mapping[str, Any],
    acceptance: Mapping[str, Any],
    phase: str,
    evidence_complete: bool,
) -> dict[str, Any]:
    arm_passed = {arm: bool(accuracy[arm]["gate"]["passed"]) for arm in ARMS}
    depth = acceptance["latency"]["reference_depth_for_interpretation"]
    speedups = {
        f"{row['shape']}:{arm}": row["median_speedup"][arm]
        for row in latency["measurements"]
        if row["crr_depth"] == depth
        for arm in ARMS
    }
    speedup_passed = bool(speedups) and all(
        value >= acceptance["latency"]["minimum_median_end_to_end_speedup"]
        for value in speedups.values()
    )
    iv_passed = all(
        iv["rows"]
        and all(
            row["inversions"][arm]["absolute_error_vs_label_iv"] is not None for row in iv["rows"]
        )
        and max(row["inversions"][arm]["absolute_error_vs_label_iv"] for row in iv["rows"])
        <= acceptance["implied_volatility"]["maximum_absolute_volatility_error"]
        for arm in ARMS
    )
    scratch_rmse = validation["scratch"]["slices"]["overall"]["normalized"]["rmse"]
    transfer_rmse = validation["transfer"]["slices"]["overall"]["normalized"]["rmse"]
    transfer_better = transfer_rmse < scratch_rmse
    passed_count = sum(arm_passed.values())
    if passed_count == 0:
        name = "failure_to_learn"
    elif passed_count == 1:
        name = "mixed_accuracy_outcome"
    elif not evidence_complete:
        name = "required_evidence_failed"
    elif phase == "validation":
        name = "eligible_for_final_evaluation"
    elif not transfer_better:
        name = "negative_transfer"
    elif speedup_passed and iv_passed:
        name = "promising"
    else:
        name = "accuracy_passed_other_feasibility_gate_failed"
    return {
        "phase": phase,
        "outcome": name,
        "arm_accuracy_passed": arm_passed,
        "reference_depth": depth,
        "reference_median_speedups": speedups,
        "all_reference_speedups_passed": speedup_passed,
        "all_iv_errors_passed": iv_passed,
        "transfer_validation_rmse_strictly_better": transfer_better,
    }


def _check_outcome(value: Any, expected: Mapping[str, Any], where: str) -> None:
    outcome = _mapping(value, where)
    _exact_keys(outcome, OUTCOME_KEYS, where)
    if dict(outcome) != dict(expected):
        raise FreezeError(f"{where} recomputation mismatch")


def _identity(report: Mapping[str, Any], protocol: Mapping[str, Any], digest: str) -> None:
    if report["protocol_sha256"] != digest:
        raise FreezeError("raw report protocol digest drift")
    if (
        report["dataset"] != protocol["dataset"]
        or report["source_artifact"] != protocol["source_artifact"]
    ):
        raise FreezeError("raw report input identity drift")


def _validate_entry_failure_evidence(
    gate: Any, evidence_value: Any, protocol: Mapping[str, Any]
) -> None:
    if gate not in {"admission_policy_invariants", "independent_pde_check", "exact_lift"}:
        raise FreezeError("unknown entry gate")
    evidence = _mapping(evidence_value, "entry-failure.evidence")
    if gate == "exact_lift":
        if _exact_lift(evidence, "entry-failure.exact_lift"):
            raise FreezeError("entry-failure lift evidence unexpectedly passes")
    elif gate == "independent_pde_check" and set(evidence) == {
        "rows",
        "passed",
        "interpretation",
    }:
        if _pde(evidence, protocol, "entry-failure.pde"):
            raise FreezeError("entry-failure PDE evidence unexpectedly passes")
    else:
        _exact_keys(
            evidence,
            frozenset({"passed", "error_type", "message"}),
            "entry-failure.evidence",
        )
        if evidence["passed"] is not False:
            raise FreezeError("entry-failure evidence unexpectedly passes")


def validate_entry_failure_report(
    report: Mapping[str, Any], protocol: Mapping[str, Any], protocol_sha256: str
) -> None:
    _exact_keys(report, ENTRY_KEYS, "entry-failure report")
    _finite_tree(report, "entry-failure report")
    if report["schema_version"] != "american-neural-pilot-entry-failure-report/1":
        raise FreezeError("entry-failure report schema differs")
    _identity(report, protocol, protocol_sha256)
    if report["lifecycle"] != {
        "phase": "entry_gate_failed",
        "final_evaluation_attempts": 0,
        "final_partition_consumed": False,
    }:
        raise FreezeError("entry-failure lifecycle differs")
    if report["interpretation"] != (
        "terminal entry failure; no optimization or final evaluation permitted"
    ):
        raise FreezeError("entry-failure interpretation differs")
    _validate_entry_failure_evidence(report["failed_gate"], report["evidence"], protocol)
    _runtime(report["runtime"], "entry-failure.runtime")


def validate_validation_report(
    report: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    acceptance: Mapping[str, Any],
    latency_config: Mapping[str, Any],
    iv_config: Mapping[str, Any],
) -> bool:
    _exact_keys(report, VALIDATION_KEYS, "validation report")
    _finite_tree(report, "validation report")
    if report["schema_version"] != "american-neural-pilot-raw-report/1":
        raise FreezeError("validation report schema differs")
    _identity(report, protocol, protocol_sha256)
    if report["lifecycle"] != {
        "phase": "validation_complete",
        "final_evaluation_attempts": 0,
        "final_partition_consumed": False,
    }:
        raise FreezeError("validation lifecycle differs")
    if report["selected_train_rows"] != 32768:
        raise FreezeError("selected row budget drift")
    if not _exact_lift(report["exact_lift"], "validation.exact_lift"):
        raise FreezeError("training report contains a failed exact lift")
    pde_passed = _pde(report["independent_pde_check"], protocol, "validation.pde")
    artifacts = _mapping(report["artifacts"], "validation.artifacts")
    _exact_keys(artifacts, frozenset(ARMS), "validation.artifacts")
    for arm in ARMS:
        artifact = _mapping(artifacts[arm], f"validation.artifacts.{arm}")
        _exact_keys(
            artifact,
            frozenset({"path", "manifest_sha256", "weights_sha256"}),
            f"validation.artifacts.{arm}",
        )
        _digest(artifact["manifest_sha256"], f"validation.artifacts.{arm}.manifest_sha256")
        _digest(artifact["weights_sha256"], f"validation.artifacts.{arm}.weights_sha256")
    arm_passes = _models(
        report["validation"], acceptance, "validation_final_entry", "validation.models"
    )
    latency_complete = _latency(report["latency"], latency_config, "validation.latency")
    iv_complete, iv_failures = _iv(report["implied_volatility"], iv_config, "validation.iv")
    evidence = _mapping(report["validation_evidence_complete"], "validation.evidence_complete")
    _exact_keys(
        evidence,
        frozenset(
            {
                "passed",
                "independent_pde_check",
                "matched_latency",
                "synthetic_implied_volatility",
            }
        ),
        "validation.evidence_complete",
    )
    expected_evidence = {
        "independent_pde_check": pde_passed,
        "matched_latency": latency_complete,
        "synthetic_implied_volatility": iv_complete
        and iv_failures <= acceptance["implied_volatility"]["failures_max"],
    }
    expected_evidence["passed"] = all(expected_evidence.values())
    if dict(evidence) != expected_evidence:
        raise FreezeError("validation evidence-completeness recomputation mismatch")
    passed = expected_evidence["passed"] and all(arm_passes.values())
    if report["validation_final_entry_passed"] is not passed:
        raise FreezeError("validation final-entry gate recomputation mismatch")
    interpretation = _mapping(report["interpretation"], "validation.interpretation")
    _exact_keys(
        interpretation,
        frozenset({"claim_scope", "locked_experiments_ran", "outcome"}),
        "validation.interpretation",
    )
    if (
        interpretation["claim_scope"] != "one-seed one-budget mapping-only feasibility pilot"
        or interpretation["locked_experiments_ran"] is not True
    ):
        raise FreezeError("validation interpretation differs")
    _check_outcome(
        interpretation["outcome"],
        _outcome(
            report["validation"],
            report["validation"],
            report["latency"],
            report["implied_volatility"],
            acceptance,
            "validation",
            expected_evidence["passed"],
        ),
        "validation.interpretation.outcome",
    )
    _runtime(report["runtime"], "validation.runtime")
    return passed


def validate_final_report(
    report: Mapping[str, Any],
    validation: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    acceptance: Mapping[str, Any],
) -> None:
    _exact_keys(report, FINAL_KEYS, "final report")
    _finite_tree(report, "final report")
    if report["schema_version"] != "american-neural-pilot-final-report/1":
        raise FreezeError("final report schema differs")
    if report["protocol_sha256"] != protocol_sha256:
        raise FreezeError("final report protocol digest drift")
    if report["attempt"] != 1 or report["partition"] != "interpolation_test":
        raise FreezeError("final report is not the first locked-final attempt")
    if report["partition_sha256"] != protocol["dataset"]["locked_final_sha256"]:
        raise FreezeError("final partition digest drift")
    if report["lifecycle"] != {
        "final_partition_consumed": True,
        "second_attempt_allowed": False,
    }:
        raise FreezeError("final lifecycle differs")
    arm_passes = _models(
        report["models_and_baseline"], acceptance, "final_accuracy", "final.models"
    )
    passed = all(arm_passes.values())
    if report["all_arms_passed"] is not passed:
        raise FreezeError("final all-arms gate recomputation mismatch")
    expected = _outcome(
        report["models_and_baseline"],
        validation["validation"],
        validation["latency"],
        validation["implied_volatility"],
        acceptance,
        "final",
        True,
    )
    _check_outcome(report["outcome"], expected, "final.outcome")
    _runtime(report["runtime"], "final.runtime")


def _validate_final_failure_identity(expected: Any, observed: Any, stage: Any) -> None:
    expected_digest = _digest(expected, "final-attempt-failure.expected_partition_sha256")
    if observed is not None:
        observed = _digest(observed, "final-attempt-failure.observed_partition_sha256")
    early_stages = {"reservation", "ledger_reservation", "manifest_identity"}
    late_stages = {"partition_read", "model_evaluation", "report_write"}
    if stage not in {
        "reservation",
        "ledger_reservation",
        "manifest_identity",
        "partition_identity",
        "partition_read",
        "model_evaluation",
        "report_write",
    }:
        raise FreezeError("final-attempt-failure stage differs")
    if (stage in early_stages and observed is not None) or (
        stage in late_stages and observed != expected_digest
    ):
        raise FreezeError("final-attempt-failure stage/identity relationship differs")


def validate_final_failure_report(
    report: Mapping[str, Any],
    protocol: Mapping[str, Any],
    protocol_sha256: str,
) -> None:
    _exact_keys(report, FINAL_FAILURE_KEYS, "final-attempt-failure report")
    _finite_tree(report, "final-attempt-failure report")
    if report["schema_version"] != "american-neural-pilot-final-attempt-failure-report/1":
        raise FreezeError("final-attempt-failure schema differs")
    if report["protocol_sha256"] != protocol_sha256:
        raise FreezeError("final-attempt-failure protocol digest drift")
    if (
        report["attempt"] != 1
        or report["partition"] != "interpolation_test"
        or report["expected_partition_sha256"] != protocol["dataset"]["locked_final_sha256"]
    ):
        raise FreezeError("final-attempt-failure partition identity differs")
    _validate_final_failure_identity(
        report["expected_partition_sha256"],
        report["observed_partition_sha256"],
        report["failure_stage"],
    )
    error = _mapping(report["error"], "final-attempt-failure.error")
    _exact_keys(error, frozenset({"type", "message"}), "final-attempt-failure.error")
    if not all(isinstance(error[name], str) and error[name] for name in ("type", "message")):
        raise FreezeError("final-attempt-failure error is malformed")
    if report["lifecycle"] != {
        "final_partition_consumed": True,
        "second_attempt_allowed": False,
    }:
        raise FreezeError("final-attempt-failure lifecycle differs")
    if report["interpretation"] != (
        "terminal consumed final-attempt failure; no retry under this protocol"
    ):
        raise FreezeError("final-attempt-failure interpretation differs")
    _runtime(report["runtime"], "final-attempt-failure.runtime")


def validate_raw_reports(
    validation: Mapping[str, Any],
    final: Mapping[str, Any] | None,
    protocol: Mapping[str, Any],
    protocol_sha256: str,
    acceptance: Mapping[str, Any],
    latency_config: Mapping[str, Any],
    iv_config: Mapping[str, Any],
) -> str:
    if validation.get("schema_version") == "american-neural-pilot-entry-failure-report/1":
        validate_entry_failure_report(validation, protocol, protocol_sha256)
        if final is not None:
            raise FreezeError("entry-gate failure must not have a final report")
        return "entry_failure"
    passed = validate_validation_report(
        validation,
        protocol,
        protocol_sha256,
        acceptance,
        latency_config,
        iv_config,
    )
    if not passed:
        if final is not None:
            raise FreezeError("failed validation entry gate must not have a final report")
        return "validation_terminal"
    if final is None:
        raise FreezeError("passed validation entry gate requires the one-shot final report")
    if final.get("schema_version") == "american-neural-pilot-final-attempt-failure-report/1":
        validate_final_failure_report(final, protocol, protocol_sha256)
        return "final_failed_consumed"
    validate_final_report(final, validation, protocol, protocol_sha256, acceptance)
    return "final_complete"


def _compact_partition_audit(
    models: Mapping[str, Any], acceptance: Mapping[str, Any]
) -> dict[str, Any]:
    source_rows = {
        name: list(_sequence(models[name]["audit_rows"], f"{name}.audit_rows"))
        for name in MODEL_NAMES
    }
    scratch_rows = source_rows["scratch"]
    raw_common_keys = AUDIT_ROW_KEYS - {
        "prediction",
        "diagnostic_tolerance",
        "diagnostic_violations",
    }
    for name, rows in source_rows.items():
        if len(rows) != len(scratch_rows):
            raise FreezeError(f"{name} audit row count differs while compacting")
        for index, row in enumerate(rows):
            if {key: row[key] for key in raw_common_keys} != {
                key: scratch_rows[index][key] for key in raw_common_keys
            }:
                raise FreezeError(f"{name} common audit row differs while compacting")
    return {
        "common_rows": [
            {
                "slice_memberships": sorted(
                    name
                    for name in _slice_names(acceptance)
                    if _row_in_slice(row, name, acceptance)
                ),
                "normalization_scale": row["spot"]
                * math.exp(-row["dividend_yield"] * row["maturity"]),
                "early_exercise_premium": row["early_exercise_premium"],
            }
            for row in scratch_rows
        ],
        "physical_errors": {
            name: [row["prediction"] - row["reference_price"] for row in source_rows[name]]
            for name in MODEL_NAMES
        },
        "diagnostics": {
            arm: [
                {
                    "diagnostic_tolerance": row["diagnostic_tolerance"],
                    "diagnostic_violations": row["diagnostic_violations"],
                }
                for row in source_rows[arm]
            ]
            for arm in ARMS
        },
    }


def _without_audit(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in entry.items() if key != "audit_rows"}


def _validate_compact_partition_models(
    compact_models: Mapping[str, Any],
    audit_value: Any,
    where: str,
    acceptance: Mapping[str, Any],
    section: str,
) -> dict[str, bool]:
    audit = _mapping(audit_value, f"{where}.audit")
    _exact_keys(
        audit,
        frozenset({"common_rows", "physical_errors", "diagnostics"}),
        f"{where}.audit",
    )
    common = list(_sequence(audit["common_rows"], f"{where}.audit.common_rows"))
    if not common:
        raise FreezeError(f"{where}.audit.common_rows must not be empty")
    slice_names = _slice_names(acceptance)
    for index, raw in enumerate(common):
        row = _mapping(raw, f"{where}.audit.common_rows[{index}]")
        _exact_keys(
            row,
            frozenset({"slice_memberships", "normalization_scale", "early_exercise_premium"}),
            f"{where}.audit.common_rows[{index}]",
        )
        memberships = list(
            _sequence(row["slice_memberships"], f"{where}.audit.common_rows[{index}].slices")
        )
        if (
            len(memberships) != 7
            or len(set(memberships)) != 7
            or not set(memberships) <= slice_names
            or "overall" not in memberships
            or any(
                sum(name.startswith(f"{family}:") for name in memberships) != 1
                for family in (
                    "option_type",
                    "expiry",
                    "moneyness",
                    "volatility",
                    "premium_status",
                    "exercise_status",
                )
            )
        ):
            raise FreezeError(f"{where} compact slice membership is malformed")
        if _finite_number(row["normalization_scale"], where) <= 0.0:
            raise FreezeError(f"{where} compact normalization scale is invalid")
        premium = _finite_number(row["early_exercise_premium"], where)
        if premium < 0.0:
            raise FreezeError(f"{where} compact premium is invalid")
        expected_premium_slice = (
            "premium_status:zero" if premium == 0.0 else "premium_status:positive"
        )
        if expected_premium_slice not in memberships:
            raise FreezeError(f"{where} compact premium membership differs")
    errors = _mapping(audit["physical_errors"], f"{where}.audit.physical_errors")
    diagnostics = _mapping(audit["diagnostics"], f"{where}.audit.diagnostics")
    _exact_keys(errors, MODEL_NAMES, f"{where}.audit.physical_errors")
    _exact_keys(diagnostics, frozenset(ARMS), f"{where}.audit.diagnostics")
    passed: dict[str, bool] = {}
    for name in MODEL_NAMES:
        entry = dict(_mapping(compact_models[name], f"{where}.{name}"))
        _exact_keys(
            entry,
            frozenset(
                {"slices", "diagnostics"} if name not in ARMS else {"slices", "diagnostics", "gate"}
            ),
            f"{where}.{name}",
        )
        _slices(entry["slices"], acceptance, f"{where}.{name}.slices")
        values = list(_sequence(errors[name], f"{where}.audit.physical_errors.{name}"))
        if len(values) != len(common):
            raise FreezeError(f"{where}.{name} error count differs")
        values = [_finite_number(value, f"{where}.{name}.error") for value in values]
        for slice_name in slice_names:
            selected = [
                index for index, row in enumerate(common) if slice_name in row["slice_memberships"]
            ]
            reported = entry["slices"][slice_name]
            if reported["rows"] != len(selected):
                raise FreezeError(f"{where}.{name}.{slice_name} membership differs")
            if not selected:
                continue
            physical = [abs(values[index]) for index in selected]
            normalized = [
                abs(values[index]) / common[index]["normalization_scale"] for index in selected
            ]
            for label, expected_values in (
                ("physical", physical),
                ("normalized", normalized),
            ):
                expected = {
                    "rows": len(expected_values),
                    "mae": sum(expected_values) / len(expected_values),
                    "rmse": math.sqrt(
                        sum(value * value for value in expected_values) / len(expected_values)
                    ),
                    "p95_absolute_error": _linear_quantile(expected_values, 0.95),
                    "p99_absolute_error": _linear_quantile(expected_values, 0.99),
                    "maximum_absolute_error": max(expected_values),
                }
                _close_statistics(reported[label], expected, f"{where}.{name}.{slice_name}.{label}")
        details = None
        if name in ARMS:
            details = list(_sequence(diagnostics[name], f"{where}.audit.diagnostics.{name}"))
            if len(details) != len(common):
                raise FreezeError(f"{where}.{name} diagnostic count differs")
        diagnostic_rows = []
        for index in range(len(common)):
            detail = None if details is None else _mapping(details[index], f"{where}.detail")
            if detail is not None:
                _exact_keys(
                    detail,
                    frozenset({"diagnostic_tolerance", "diagnostic_violations"}),
                    f"{where}.detail",
                )
            if detail is not None:
                diagnostic_rows.append(detail)
        if name not in ARMS:
            baseline = _mapping(entry["diagnostics"], f"{where}.{name}.diagnostics")
            _exact_keys(
                baseline,
                frozenset({"not_applicable", "stored_error_equals_early_exercise_premium"}),
                f"{where}.{name}.diagnostics",
            )
            if baseline["stored_error_equals_early_exercise_premium"] is not True or not all(
                -error == row["early_exercise_premium"]
                for error, row in zip(values, common, strict=True)
            ):
                raise FreezeError(f"{where} baseline premium identity differs")
            continue
        bound, shape = _diagnostics(
            entry["diagnostics"], diagnostic_rows, f"{where}.{name}.diagnostics"
        )
        gate = _mapping(entry["gate"], f"{where}.{name}.gate")
        _exact_keys(gate, frozenset({"checks", "passed"}), f"{where}.{name}.gate")
        checks = _mapping(gate["checks"], f"{where}.{name}.gate.checks")
        _exact_keys(checks, GATE_CHECKS, f"{where}.{name}.gate.checks")
        overall = entry["slices"]["overall"]["normalized"]
        limits = acceptance[section]
        recomputed = {
            "normalized_rmse": overall["rmse"] <= limits["normalized_rmse_max"],
            "normalized_p99_absolute_error": overall["p99_absolute_error"]
            <= limits["normalized_p99_absolute_error_max"],
            "normalized_maximum_absolute_error": overall["maximum_absolute_error"]
            <= limits["normalized_maximum_absolute_error_max"],
            "material_bound_violations": bound <= limits["maximum_material_bound_violations"],
            "material_shape_violations": shape <= limits["maximum_material_shape_violations"],
        }
        if dict(checks) != recomputed or gate["passed"] is not all(recomputed.values()):
            raise FreezeError(f"{where}.{name} gate recomputation differs")
        passed[name] = bool(gate["passed"])
    return passed


def extract_snapshot(
    validation: Mapping[str, Any],
    final: Mapping[str, Any] | None,
    *,
    state: str,
    protocol_path: Path,
    validation_path: Path,
    final_path: Path | None,
    acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    entry = state == "entry_failure"
    validation_terminal = state == "validation_terminal"
    final_failed = state == "final_failed_consumed"
    final_evidence = final is not None and not final_failed
    return {
        "schema_version": SNAPSHOT_SCHEMA,
        "protocol": {
            "path": str(protocol_path.relative_to(PROJECT_ROOT)),
            "sha256": _sha256(protocol_path),
        },
        "source_reports": {
            "validation": {"file": validation_path.name, "sha256": _sha256(validation_path)},
            "final": (
                None
                if final_path is None
                else {"file": final_path.name, "sha256": _sha256(final_path)}
            ),
        },
        "dataset": dict(validation["dataset"]),
        "source_artifact": dict(validation["source_artifact"]),
        "lifecycle": {
            "state": state,
            "validation_final_entry_passed": (
                False if entry else validation["validation_final_entry_passed"]
            ),
            "final_evaluation_attempts": 0 if final is None else 1,
            "final_partition_consumed": final is not None,
            "second_attempt_allowed": False,
        },
        "audit": (
            None
            if entry
            else {
                "validation": _compact_partition_audit(validation["validation"], acceptance),
                "final": (
                    None
                    if not final_evidence
                    else _compact_partition_audit(final["models_and_baseline"], acceptance)
                ),
            }
        ),
        "entry_failure": (
            {"failed_gate": validation["failed_gate"], "evidence": validation["evidence"]}
            if entry
            else None
        ),
        "exact_lift": None if entry else validation["exact_lift"],
        "independent_pde_check": None if entry else validation["independent_pde_check"],
        "arms": (
            None
            if entry
            else {
                arm: {
                    "artifact": validation["artifacts"][arm],
                    "validation": _without_audit(validation["validation"][arm]),
                    "final": (
                        None
                        if not final_evidence
                        else _without_audit(final["models_and_baseline"][arm])
                    ),
                }
                for arm in ARMS
            }
        ),
        "baseline": (
            None
            if entry
            else {
                "validation": _without_audit(validation["validation"]["european_crr_baseline"]),
                "final": (
                    None
                    if not final_evidence
                    else _without_audit(final["models_and_baseline"]["european_crr_baseline"])
                ),
            }
        ),
        "latency": None if entry else validation["latency"],
        "implied_volatility": None if entry else validation["implied_volatility"],
        "outcome": (
            {
                "phase": "entry",
                "outcome": "entry_gate_failed",
                "failed_gate": validation["failed_gate"],
            }
            if entry
            else (
                validation["interpretation"]["outcome"]
                if validation_terminal
                else (
                    {
                        "phase": "final",
                        "outcome": "final_attempt_failed_consumed",
                        "expected_partition_sha256": final[
                            "expected_partition_sha256"
                        ],
                        "observed_partition_sha256": final[
                            "observed_partition_sha256"
                        ],
                        "failure_stage": final["failure_stage"],
                        "error": dict(final["error"]),
                    }
                    if final_failed
                    else final["outcome"]
                )
            )
        ),
        "interpretation": {
            "claim_scope": "one-seed one-budget mapping-only feasibility pilot; not H2",
            "no_american_greek_claim": True,
            "no_market_or_bid_ask_claim": True,
        },
    }


def validate_snapshot(
    snapshot: Mapping[str, Any],
    protocol_path: Path,
    protocol: Mapping[str, Any],
    acceptance: Mapping[str, Any],
    latency_config: Mapping[str, Any],
    iv_config: Mapping[str, Any],
) -> None:
    _exact_keys(snapshot, SNAPSHOT_KEYS, "snapshot")
    _finite_tree(snapshot, "snapshot")
    if snapshot["schema_version"] != SNAPSHOT_SCHEMA:
        raise FreezeError("snapshot schema differs")
    protocol_block = _mapping(snapshot["protocol"], "snapshot.protocol")
    _exact_keys(protocol_block, frozenset({"path", "sha256"}), "snapshot.protocol")
    if protocol_block["sha256"] != _sha256(protocol_path):
        raise FreezeError("snapshot protocol digest drift")
    if (
        snapshot["dataset"] != protocol["dataset"]
        or snapshot["source_artifact"] != protocol["source_artifact"]
    ):
        raise FreezeError("snapshot input identity drift")
    source_reports = _mapping(snapshot["source_reports"], "snapshot.source_reports")
    _exact_keys(source_reports, frozenset({"validation", "final"}), "snapshot.source_reports")
    validation_source = _mapping(source_reports["validation"], "snapshot.source_reports.validation")
    _exact_keys(
        validation_source,
        frozenset({"file", "sha256"}),
        "snapshot.source_reports.validation",
    )
    _digest(validation_source["sha256"], "snapshot.source_reports.validation.sha256")
    if source_reports["final"] is not None:
        final_source = _mapping(source_reports["final"], "snapshot.source_reports.final")
        _exact_keys(final_source, frozenset({"file", "sha256"}), "snapshot.source_reports.final")
        _digest(final_source["sha256"], "snapshot.source_reports.final.sha256")
    interpretation = _mapping(snapshot["interpretation"], "snapshot.interpretation")
    if interpretation != {
        "claim_scope": "one-seed one-budget mapping-only feasibility pilot; not H2",
        "no_american_greek_claim": True,
        "no_market_or_bid_ask_claim": True,
    }:
        raise FreezeError("snapshot interpretation differs")
    lifecycle = _mapping(snapshot["lifecycle"], "snapshot.lifecycle")
    _exact_keys(
        lifecycle,
        frozenset(
            {
                "state",
                "validation_final_entry_passed",
                "final_evaluation_attempts",
                "final_partition_consumed",
                "second_attempt_allowed",
            }
        ),
        "snapshot.lifecycle",
    )
    state = lifecycle["state"]
    if state == "entry_failure":
        entry_failure = _mapping(snapshot["entry_failure"], "snapshot.entry_failure")
        _exact_keys(
            entry_failure,
            frozenset({"failed_gate", "evidence"}),
            "snapshot.entry_failure",
        )
        _validate_entry_failure_evidence(
            entry_failure["failed_gate"], entry_failure["evidence"], protocol
        )
        entry_outcome = _mapping(snapshot["outcome"], "snapshot.outcome")
        if entry_outcome != {
            "phase": "entry",
            "outcome": "entry_gate_failed",
            "failed_gate": entry_failure["failed_gate"],
        }:
            raise FreezeError("entry-failure snapshot outcome differs")
        if (
            snapshot["entry_failure"] is None
            or source_reports["final"] is not None
            or any(
                snapshot[name] is not None
                for name in (
                    "exact_lift",
                    "independent_pde_check",
                    "audit",
                    "arms",
                    "baseline",
                    "latency",
                    "implied_volatility",
                )
            )
            or lifecycle
            != {
                "state": "entry_failure",
                "validation_final_entry_passed": False,
                "final_evaluation_attempts": 0,
                "final_partition_consumed": False,
                "second_attempt_allowed": False,
            }
        ):
            raise FreezeError("entry-failure snapshot lifecycle differs")
        return
    if state not in {"validation_terminal", "final_failed_consumed", "final_complete"}:
        raise FreezeError("snapshot lifecycle state differs")
    consumed = state in {"final_failed_consumed", "final_complete"}
    expected_lifecycle = {
        "state": state,
        "validation_final_entry_passed": state != "validation_terminal",
        "final_evaluation_attempts": 1 if consumed else 0,
        "final_partition_consumed": consumed,
        "second_attempt_allowed": False,
    }
    if dict(lifecycle) != expected_lifecycle:
        raise FreezeError("snapshot lifecycle differs")
    if (source_reports["final"] is not None) is not consumed:
        raise FreezeError("snapshot final source-report lifecycle differs")
    if snapshot["entry_failure"] is not None:
        raise FreezeError("non-entry snapshot contains entry-failure evidence")
    if not _exact_lift(snapshot["exact_lift"], "snapshot.exact_lift"):
        raise FreezeError("snapshot contains a failed exact lift")
    if not _pde(snapshot["independent_pde_check"], protocol, "snapshot.pde"):
        raise FreezeError("snapshot contains a failed PDE entry check")
    _latency(snapshot["latency"], latency_config, "snapshot.latency")
    _iv(snapshot["implied_volatility"], iv_config, "snapshot.iv")
    arms = _mapping(snapshot["arms"], "snapshot.arms")
    _exact_keys(arms, frozenset(ARMS), "snapshot.arms")
    validation_models_compact = {}
    final_models_compact = {}
    baseline = _mapping(snapshot["baseline"], "snapshot.baseline")
    _exact_keys(baseline, frozenset({"validation", "final"}), "snapshot.baseline")
    for arm in ARMS:
        block = _mapping(arms[arm], f"snapshot.arms.{arm}")
        _exact_keys(
            block,
            frozenset({"artifact", "validation", "final"}),
            f"snapshot.arms.{arm}",
        )
        artifact = _mapping(block["artifact"], f"snapshot.arms.{arm}.artifact")
        _exact_keys(
            artifact,
            frozenset({"path", "manifest_sha256", "weights_sha256"}),
            f"snapshot.arms.{arm}.artifact",
        )
        _digest(artifact["manifest_sha256"], f"snapshot.arms.{arm}.artifact.manifest_sha256")
        _digest(artifact["weights_sha256"], f"snapshot.arms.{arm}.artifact.weights_sha256")
        validation_models_compact[arm] = block["validation"]
        if block["final"] is not None:
            final_models_compact[arm] = block["final"]
    validation_models_compact["european_crr_baseline"] = baseline["validation"]
    snapshot_audit = _mapping(snapshot["audit"], "snapshot.audit")
    _exact_keys(snapshot_audit, frozenset({"validation", "final"}), "snapshot.audit")
    validation_passes = _validate_compact_partition_models(
        validation_models_compact,
        snapshot_audit["validation"],
        "snapshot.validation",
        acceptance,
        "validation_final_entry",
    )
    if state in {"validation_terminal", "final_failed_consumed"}:
        if (
            final_models_compact
            or baseline["final"] is not None
            or snapshot_audit["final"] is not None
        ):
            raise FreezeError("terminal snapshot contains final model evidence")
        if state == "final_failed_consumed":
            failure_outcome = _mapping(snapshot["outcome"], "snapshot.outcome")
            _exact_keys(
                failure_outcome,
                frozenset(
                    {
                        "phase",
                        "outcome",
                        "expected_partition_sha256",
                        "observed_partition_sha256",
                        "failure_stage",
                        "error",
                    }
                ),
                "snapshot.outcome",
            )
            error = _mapping(failure_outcome["error"], "snapshot.outcome.error")
            _exact_keys(error, frozenset({"type", "message"}), "snapshot.outcome.error")
            if (
                failure_outcome["phase"] != "final"
                or failure_outcome["outcome"] != "final_attempt_failed_consumed"
                or failure_outcome["expected_partition_sha256"]
                != protocol["dataset"]["locked_final_sha256"]
                or not all(
                    isinstance(error[name], str) and error[name] for name in ("type", "message")
                )
            ):
                raise FreezeError("consumed-failure snapshot outcome differs")
            _validate_final_failure_identity(
                failure_outcome["expected_partition_sha256"],
                failure_outcome["observed_partition_sha256"],
                failure_outcome["failure_stage"],
            )
            return
        expected = _outcome(
            {arm: {"gate": {"passed": validation_passes[arm]}} for arm in ARMS},
            validation_models_compact,
            snapshot["latency"],
            snapshot["implied_volatility"],
            acceptance,
            "validation",
            False,
        )
    else:
        if len(final_models_compact) != 2 or baseline["final"] is None:
            raise FreezeError("final-complete snapshot is missing final evidence")
        final_models_compact["european_crr_baseline"] = baseline["final"]
        final_passes = _validate_compact_partition_models(
            final_models_compact,
            snapshot_audit["final"],
            "snapshot.final",
            acceptance,
            "final_accuracy",
        )
        expected = _outcome(
            {arm: {"gate": {"passed": final_passes[arm]}} for arm in ARMS},
            validation_models_compact,
            snapshot["latency"],
            snapshot["implied_volatility"],
            acceptance,
            "final",
            True,
        )
    _check_outcome(snapshot["outcome"], expected, "snapshot.outcome")


def serialise(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_atomic(path: Path, text: str, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--final-report", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--update", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        protocol_path = arguments.protocol.resolve()
        protocol = _validate_protocol(protocol_path, PROJECT_ROOT)
        acceptance, latency_config, iv_config = _leaves(protocol, PROJECT_ROOT)
        if arguments.check:
            if arguments.validation_report or arguments.final_report or arguments.update:
                raise FreezeError("--check does not accept raw reports or --update")
            snapshot = _load_json(arguments.output, "Task 9G snapshot")
            validate_snapshot(
                snapshot,
                protocol_path,
                protocol,
                acceptance,
                latency_config,
                iv_config,
            )
            if arguments.output.read_text(encoding="utf-8") != serialise(snapshot):
                raise FreezeError("snapshot is not canonical JSON")
            print(json.dumps({"checked": str(arguments.output)}, sort_keys=True))
            return 0
        if arguments.validation_report is None:
            raise FreezeError("freeze mode requires --validation-report")
        validation = _load_json(arguments.validation_report, "validation report")
        final = (
            None
            if arguments.final_report is None
            else _load_json(arguments.final_report, "final report")
        )
        state = validate_raw_reports(
            validation,
            final,
            protocol,
            _sha256(protocol_path),
            acceptance,
            latency_config,
            iv_config,
        )
        snapshot = extract_snapshot(
            validation,
            final,
            state=state,
            protocol_path=protocol_path,
            validation_path=arguments.validation_report,
            final_path=arguments.final_report,
            acceptance=acceptance,
        )
        validate_snapshot(
            snapshot,
            protocol_path,
            protocol,
            acceptance,
            latency_config,
            iv_config,
        )
        write_atomic(arguments.output, serialise(snapshot), overwrite=arguments.update)
    except (FreezeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"snapshot": str(arguments.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
