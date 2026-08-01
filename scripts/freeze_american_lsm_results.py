#!/usr/bin/env python3
"""Freeze the reviewed American CRR/LSM cross-check report as a compact snapshot.

The raw cross-check report is expensive to produce and is deliberately ignored
by git (``artifacts/``). This tool distils it into a small, strictly versioned,
canonically serialised snapshot that *is* checked in, so the repository carries
the reviewed numerical evidence without carrying the multi-hundred-kilobyte
report.

Two modes:

``--report R --output S``
    Extract a fresh snapshot from the reviewed raw report. Every numeric field
    is computed from the report; nothing is hand-copied. Refuses to overwrite an
    existing ``S`` unless ``--update`` is given.

``--check`` (CI-safe)
    Validate the checked-in snapshot on its own and reconcile its recorded
    configuration and C++ provenance digests against the current repository
    files. This mode never reads the ignored raw report, so continuous
    integration can enforce the snapshot without the artifact being present. If
    ``--report`` is supplied as well, the snapshot is additionally required to
    be a byte-exact re-extraction of that report (staleness detection).

Exit codes: ``0`` success, ``2`` any validation, provenance, or I/O failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT: Final = (
    PROJECT_ROOT / "docs/results/american_lsm_crosscheck_results_v1.json"
)
DEFAULT_REPORT: Final = (
    PROJECT_ROOT / "artifacts/american-lsm-crosscheck-review-fixed-v1.json"
)

SNAPSHOT_SCHEMA_VERSION: Final = "american-lsm-crosscheck-results/1"
REPORT_SCHEMA_VERSION: Final = "american-lsm-crosscheck-report/1"
SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")

# Exact key sets. Extraction rejects a report whose shape differs at all, in
# either direction: an unknown key means the generator emitted something this
# snapshot schema does not understand, and a missing key means a field the
# snapshot depends on silently disappeared. Both must fail loudly rather than
# produce a quietly incomplete frozen result.
REPORT_KEYS: Final = frozenset(
    {"schema_version", "source", "study", "experiments", "interpretation"}
)
REPORT_SOURCE_KEYS: Final = frozenset(
    {
        "build_configuration",
        "config_file",
        "config_sha256",
        "crr_implementation_sha256",
        "cxx_compiler",
        "lsm_header_sha256",
        "lsm_implementation_sha256",
        "lsm_source_sha256",
        "oracle_version",
        "rng",
    }
)
REPORT_STUDY_KEYS: Final = frozenset(
    {
        "base_seed",
        "case_names",
        "case_source",
        "case_source_sha256",
        "crr_reference_definition",
        "crr_reference_steps",
        "engine",
        "maximum_training_memory_bytes",
        "name",
    }
)
REPORT_INTERPRETATION_KEYS: Final = frozenset(
    {
        "control_variate",
        "crr_reference",
        "estimand",
        "lower_bound",
        "non_finite_rejection",
        "standard_error",
        "valuation_interval",
        "variance_reduction",
    }
)
REPORT_EXPERIMENT_ENTRY_KEYS: Final = frozenset({"cases", "experiment", "summary"})
REPORT_EXPERIMENT_KEYS: Final = frozenset(
    {
        "exercise_steps",
        "name",
        "polynomial_degree",
        "role",
        "training_paths",
        "valuation_paths",
    }
)
REPORT_SUMMARY_KEYS: Final = frozenset(
    {
        "cases",
        "crr_reference_inside_stochastic_valuation_interval_cases",
        "deterministic_maximum_absolute_crr_difference",
        "deterministic_zero_width_valuation_cases",
        "maximum_absolute_crr_reference_minus_lsm",
        "mean_crr_reference_minus_lsm",
        "mean_standard_error",
        "mean_stochastic_standard_error",
        "minimum_variance_reduction_ratio",
        "notes",
        "stochastic_valuation_cases",
        "total_constant_fallback_dates",
        "variance_reduction_applicable_cases",
        "variance_reduction_infinite_cases",
    }
)
REPORT_CASE_ENTRY_KEYS: Final = frozenset({"case", "crr_reference", "lsm", "seeds"})
REPORT_CASE_KEYS: Final = frozenset(
    {
        "dividend_yield",
        "exercise_expectation",
        "maturity",
        "name",
        "option_type",
        "rate",
        "spot",
        "strike",
        "volatility",
    }
)
REPORT_CRR_KEYS: Final = frozenset(
    {"absolute_pair_gap", "at_steps", "at_steps_plus_one", "price", "steps"}
)
REPORT_LSM_KEYS: Final = frozenset(
    {
        "confidence_interval_lower",
        "confidence_interval_upper",
        "confidence_level",
        "control_variate_coefficient",
        "crr_reference_inside_lsm_valuation_interval",
        "crr_reference_minus_lsm",
        "estimated_training_working_set_bytes",
        "european_analytic_price",
        "european_monte_carlo_price",
        "european_monte_carlo_sampled",
        "european_standard_error",
        "exercise_at_zero",
        "exercise_steps",
        "independent_valuation_pairs",
        "price",
        "raw_price",
        "raw_standard_error",
        "regression_summary",
        "standard_error",
        "standardized_crr_gap",
        "training_continuation_value_at_zero",
        "training_paths",
        "valuation_early_exercise_fraction",
        "valuation_early_exercise_paths",
        "valuation_interval_is_zero_width",
        "valuation_paths",
        "variance_reduction_applicable",
        "variance_reduction_is_infinite",
        "variance_reduction_ratio",
    }
)
REPORT_REGRESSION_KEYS: Final = frozenset(
    {
        "constant_fallback_dates",
        "dates_with_in_the_money_paths",
        "exercise_dates_fitted",
        "minimum_in_the_money_paths",
        "minimum_relative_r_diagonal",
    }
)

SNAPSHOT_KEYS: Final = frozenset(
    {
        "schema_version",
        "source",
        "study",
        "experiments",
        "primary_cases",
        "report_semantics",
        "limitations",
    }
)
SNAPSHOT_SOURCE_KEYS: Final = frozenset(
    {
        "build_configuration",
        "case_source",
        "case_source_sha256",
        "config_file",
        "config_sha256",
        "crr_implementation_sha256",
        "cxx_compiler",
        "lsm_header_sha256",
        "lsm_implementation_sha256",
        "lsm_source_sha256",
        "oracle_version",
        "report_filename",
        "report_schema_version",
        "report_sha256",
        "rng",
    }
)
SNAPSHOT_STUDY_KEYS: Final = frozenset(
    {
        "base_seed",
        "case_count",
        "case_names",
        "case_rows",
        "crr_reference_definition",
        "crr_reference_steps",
        "engine",
        "experiment_count",
        "maximum_training_memory_bytes",
        "name",
        "primary_experiment",
    }
)
SNAPSHOT_EXPERIMENT_KEYS: Final = frozenset(
    {
        "cases",
        "deterministic_maximum_absolute_crr_difference",
        "deterministic_zero_width_valuation_cases",
        "exercise_steps",
        "maximum_absolute_crr_reference_minus_lsm",
        "mean_crr_reference_minus_lsm",
        "mean_standard_error",
        "mean_stochastic_valuation_standard_error",
        "minimum_applicable_finite_variance_reduction_ratio",
        "name",
        "polynomial_degree",
        "role",
        "stochastic_valuation_cases",
        "stochastic_valuation_interval_containment_cases",
        "total_constant_fallback_dates",
        "training_paths",
        "valuation_paths",
        "variance_reduction_applicable_cases",
        "variance_reduction_infinite_cases",
    }
)
SNAPSHOT_PRIMARY_CASE_KEYS: Final = frozenset(
    {
        "constant_fallback_dates",
        "control_variate_coefficient",
        "crr_reference_minus_lsm",
        "crr_reference_price",
        "dividend_yield",
        "exercise_expectation",
        "lsm_price",
        "maturity",
        "name",
        "option_type",
        "rate",
        "spot",
        "standardized_crr_gap",
        "strike",
        "valuation_early_exercise_fraction",
        "valuation_interval_is_zero_width",
        "valuation_only_standard_error",
        "variance_reduction_applicable",
        "variance_reduction_is_infinite",
        "variance_reduction_ratio",
        "volatility",
    }
)
SNAPSHOT_SEMANTICS_KEYS: Final = frozenset(
    {
        "control_variate",
        "crr_reference",
        "deterministic_cases",
        "estimand",
        "lower_estimate",
        "standard_error",
        "valuation_interval",
        "variance_reduction",
    }
)

# Engine files whose digests the report records verbatim. Verified in --check
# mode without needing the raw report or a compiled binding.
DIRECT_PROVENANCE_PATHS: Final = {
    "lsm_header_sha256": "cpp/include/dp/least_squares_monte_carlo.hpp",
    "lsm_source_sha256": "cpp/src/least_squares_monte_carlo.cpp",
}

# The binding exposes ``__crr_implementation_sha256__`` and
# ``__lsm_implementation_sha256__`` as composite identities built by CMakeLists
# from the binding source plus the headers and sources each engine is compiled
# from. Recomputing them here from the same files keeps --check independent of
# both the ignored artifact and a built extension module.
COMPOSITE_PROVENANCE: Final = {
    "crr_implementation_sha256": (
        "bindings/python/module.cpp",
        "cpp/include/dp/binomial_tree.hpp",
        "cpp/include/dp/black_scholes.hpp",
        "cpp/include/dp/option.hpp",
        "cpp/src/binomial_tree.cpp",
        "cpp/src/black_scholes.cpp",
        "cpp/src/option.cpp",
    ),
    "lsm_implementation_sha256": (
        "bindings/python/module.cpp",
        "cpp/include/dp/least_squares_monte_carlo.hpp",
        "cpp/include/dp/black_scholes.hpp",
        "cpp/include/dp/option.hpp",
        "cpp/src/least_squares_monte_carlo.cpp",
        "cpp/src/black_scholes.cpp",
        "cpp/src/option.cpp",
    ),
}

REPORT_SEMANTICS: Final = {
    "estimand": (
        "Each LSM price is the discounted value of a learned Bermudan stopping "
        "policy evaluated on valuation paths independent of the paths the policy "
        "was fitted on, over the declared discrete exercise grid."
    ),
    "lower_estimate": (
        "For a fixed learned policy the expectation is no greater than the "
        "same-grid optimal stopping value, so LSM sitting below the finer-step "
        "CRR reference is the expected ordering rather than a defect."
    ),
    "valuation_interval": (
        "The reported interval and standard error cover valuation sampling "
        "error only. They quantify noise around an already-frozen policy."
    ),
    "standard_error": (
        "Antithetic pair averages are the independent observations; individual "
        "paths are not counted as independent."
    ),
    "control_variate": (
        "A coefficient estimated on policy-training pairs is frozen before "
        "valuation; the estimate subtracts that coefficient times the same-path "
        "European deviation from its analytic Black-Scholes expectation."
    ),
    "variance_reduction": (
        "Reported only where mathematically defined. Deterministic "
        "immediate-exercise cases have an undefined 0/0 ratio and are excluded "
        "from every variance-reduction summary, so they can neither set nor "
        "lower a reported minimum."
    ),
    "deterministic_cases": (
        "Cases whose valuation estimator has zero variance produce a "
        "single-point interval, so containment of any finite-step tree value is "
        "arithmetically impossible regardless of agreement. They are summarised "
        "separately from stochastic interval comparisons."
    ),
    "crr_reference": (
        "The high-step CRR adjacent average is an internal model cross-check. "
        "It is neither exact American truth nor market truth, and no count "
        "derived from it is an acceptance gate."
    ),
}

LIMITATIONS: Final = (
    "Scope is one-factor constant-parameter geometric Brownian motion with "
    "constant rate, dividend yield, and volatility.",
    "All evidence is synthetic. No market data, quoted price, or calibration "
    "target enters this study.",
    "The valuation interval covers valuation sampling error only. It does not "
    "cover policy-fitting error or discrete-exercise (grid) bias, both of which "
    "are systematic and do not shrink with valuation paths.",
    "Stochastic interval containment counts are diagnostic, not acceptance "
    "gates, and are expected to fall as valuation paths increase.",
    "The CRR reference is an adjacent-step average at a finite step count, not "
    "the continuously exercisable American value.",
    "These results select no production label policy. Task 8E separately "
    "predeclares and runs the CRR label-policy calibration study.",
)


class FreezeError(RuntimeError):
    """Raised when the report, the snapshot, or repository provenance is invalid."""


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FreezeError(f"{where} must be an object")
    return value


def _sequence(value: Any, where: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise FreezeError(f"{where} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], where: str) -> None:
    present = set(value)
    missing = sorted(expected - present)
    unknown = sorted(present - expected)
    if missing:
        raise FreezeError(f"{where} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise FreezeError(f"{where} has unknown fields: {', '.join(unknown)}")


def _string(container: Mapping[str, Any], key: str, where: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value:
        raise FreezeError(f"{where}.{key} must be a non-empty string")
    return value


def _digest(container: Mapping[str, Any], key: str, where: str) -> str:
    value = _string(container, key, where)
    if not SHA256_PATTERN.fullmatch(value):
        raise FreezeError(f"{where}.{key} must be a lowercase hex SHA-256 digest")
    return value


def _bool(container: Mapping[str, Any], key: str, where: str) -> bool:
    value = container.get(key)
    if not isinstance(value, bool):
        raise FreezeError(f"{where}.{key} must be a boolean")
    return value


def _integer(container: Mapping[str, Any], key: str, where: str, *, minimum: int) -> int:
    value = container.get(key)
    # bool is a subclass of int; an accidental boolean must not pass as a count.
    if not isinstance(value, int) or isinstance(value, bool):
        raise FreezeError(f"{where}.{key} must be an integer")
    if value < minimum:
        raise FreezeError(f"{where}.{key} must be at least {minimum}")
    return value


def _finite(container: Mapping[str, Any], key: str, where: str) -> float:
    value = container.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FreezeError(f"{where}.{key} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise FreezeError(f"{where}.{key} must be finite, got {value!r}")
    return number


def _optional_finite(
    container: Mapping[str, Any], key: str, where: str
) -> float | None:
    if container.get(key) is None:
        return None
    return _finite(container, key, where)


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise FreezeError(f"cannot hash '{path}': {error}") from error


def _composite_digest(relative_paths: Sequence[str]) -> str:
    joined = ":".join(_sha256_file(PROJECT_ROOT / name) for name in relative_paths)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Raw report validation
# --------------------------------------------------------------------------


def validate_report(report: Any) -> Mapping[str, Any]:
    """Strictly validate the raw cross-check report and return it unchanged."""
    payload = _mapping(report, "report")
    _exact_keys(payload, REPORT_KEYS, "report")
    schema = _string(payload, "schema_version", "report")
    if schema != REPORT_SCHEMA_VERSION:
        raise FreezeError(
            f"report schema_version must be '{REPORT_SCHEMA_VERSION}', got '{schema}'"
        )

    source = _mapping(payload["source"], "report.source")
    _exact_keys(source, REPORT_SOURCE_KEYS, "report.source")
    for key in (
        "config_sha256",
        "crr_implementation_sha256",
        "lsm_header_sha256",
        "lsm_implementation_sha256",
        "lsm_source_sha256",
    ):
        _digest(source, key, "report.source")
    for key in ("build_configuration", "config_file", "cxx_compiler", "oracle_version", "rng"):
        _string(source, key, "report.source")

    study = _mapping(payload["study"], "report.study")
    _exact_keys(study, REPORT_STUDY_KEYS, "report.study")
    _digest(study, "case_source_sha256", "report.study")
    for key in ("case_source", "crr_reference_definition", "engine", "name"):
        _string(study, key, "report.study")
    _integer(study, "base_seed", "report.study", minimum=0)
    _integer(study, "crr_reference_steps", "report.study", minimum=1)
    _integer(study, "maximum_training_memory_bytes", "report.study", minimum=1)
    case_names = _sequence(study["case_names"], "report.study.case_names")
    if not case_names or any(not isinstance(name, str) or not name for name in case_names):
        raise FreezeError("report.study.case_names must be a non-empty array of strings")
    if len(set(case_names)) != len(case_names):
        raise FreezeError("report.study.case_names contains duplicates")

    interpretation = _mapping(payload["interpretation"], "report.interpretation")
    _exact_keys(interpretation, REPORT_INTERPRETATION_KEYS, "report.interpretation")
    for key in sorted(REPORT_INTERPRETATION_KEYS):
        _string(interpretation, key, "report.interpretation")

    experiments = _sequence(payload["experiments"], "report.experiments")
    if not experiments:
        raise FreezeError("report.experiments must not be empty")

    seen_names: set[str] = set()
    primary_names: list[str] = []
    for index, entry in enumerate(experiments):
        where = f"report.experiments[{index}]"
        block = _mapping(entry, where)
        _exact_keys(block, REPORT_EXPERIMENT_ENTRY_KEYS, where)
        experiment = _mapping(block["experiment"], f"{where}.experiment")
        _exact_keys(experiment, REPORT_EXPERIMENT_KEYS, f"{where}.experiment")
        name = _string(experiment, "name", f"{where}.experiment")
        if name in seen_names:
            raise FreezeError(f"duplicate experiment name '{name}'")
        seen_names.add(name)
        role = _string(experiment, "role", f"{where}.experiment")
        if role == "primary":
            primary_names.append(name)
        _integer(experiment, "exercise_steps", f"{where}.experiment", minimum=1)
        _integer(experiment, "polynomial_degree", f"{where}.experiment", minimum=1)
        _integer(experiment, "training_paths", f"{where}.experiment", minimum=1)
        _integer(experiment, "valuation_paths", f"{where}.experiment", minimum=1)
        _validate_report_cases(block, list(case_names), where)
        _validate_report_summary(block, where)

    if len(primary_names) != 1:
        raise FreezeError(
            "report must declare exactly one experiment with role 'primary', "
            f"found {len(primary_names)}"
        )
    return payload


def _validate_report_cases(
    block: Mapping[str, Any], case_names: list[str], where: str
) -> None:
    cases = _sequence(block["cases"], f"{where}.cases")
    if [_case_name(case, f"{where}.cases[{i}]") for i, case in enumerate(cases)] != case_names:
        raise FreezeError(
            f"{where}.cases must list exactly the study case names, in order"
        )
    for index, entry in enumerate(cases):
        case_where = f"{where}.cases[{index}]"
        case_block = _mapping(entry, case_where)
        _exact_keys(case_block, REPORT_CASE_ENTRY_KEYS, case_where)

        case = _mapping(case_block["case"], f"{case_where}.case")
        _exact_keys(case, REPORT_CASE_KEYS, f"{case_where}.case")
        option_type = _string(case, "option_type", f"{case_where}.case")
        if option_type not in {"call", "put"}:
            raise FreezeError(f"{case_where}.case.option_type must be 'call' or 'put'")
        _string(case, "exercise_expectation", f"{case_where}.case")
        for key in ("dividend_yield", "maturity", "rate", "spot", "strike", "volatility"):
            _finite(case, key, f"{case_where}.case")

        crr = _mapping(case_block["crr_reference"], f"{case_where}.crr_reference")
        _exact_keys(crr, REPORT_CRR_KEYS, f"{case_where}.crr_reference")
        _integer(crr, "steps", f"{case_where}.crr_reference", minimum=1)
        for key in ("absolute_pair_gap", "at_steps", "at_steps_plus_one", "price"):
            _finite(crr, key, f"{case_where}.crr_reference")

        _validate_report_lsm(case_block["lsm"], f"{case_where}.lsm")

        seeds = _mapping(case_block["seeds"], f"{case_where}.seeds")
        if set(seeds) != {"derivation", "training", "valuation"}:
            raise FreezeError(f"{case_where}.seeds has an unexpected shape")


def _case_name(entry: Any, where: str) -> str:
    block = _mapping(entry, where)
    case = _mapping(block.get("case"), f"{where}.case")
    return _string(case, "name", f"{where}.case")


def _validate_report_lsm(value: Any, where: str) -> None:
    lsm = _mapping(value, where)
    _exact_keys(lsm, REPORT_LSM_KEYS, where)
    for key in (
        "crr_reference_inside_lsm_valuation_interval",
        "european_monte_carlo_sampled",
        "exercise_at_zero",
        "valuation_interval_is_zero_width",
        "variance_reduction_applicable",
        "variance_reduction_is_infinite",
    ):
        _bool(lsm, key, where)
    for key in (
        "estimated_training_working_set_bytes",
        "exercise_steps",
        "independent_valuation_pairs",
        "training_paths",
        "valuation_early_exercise_paths",
        "valuation_paths",
    ):
        _integer(lsm, key, where, minimum=0)
    for key in (
        "confidence_interval_lower",
        "confidence_interval_upper",
        "confidence_level",
        "control_variate_coefficient",
        "crr_reference_minus_lsm",
        "european_analytic_price",
        "price",
        "raw_price",
        "raw_standard_error",
        "standard_error",
        "training_continuation_value_at_zero",
        "valuation_early_exercise_fraction",
    ):
        _finite(lsm, key, where)
    # Undefined by construction on deterministic or non-applicable cases, so
    # null is meaningful here and must stay distinguishable from zero.
    _optional_finite(lsm, "standardized_crr_gap", where)
    _optional_finite(lsm, "variance_reduction_ratio", where)

    # A case that exercises immediately never samples the European control, so
    # both control quantities are null exactly when it was not sampled.
    sampled = bool(lsm["european_monte_carlo_sampled"])
    for key in ("european_monte_carlo_price", "european_standard_error"):
        value = _optional_finite(lsm, key, where)
        if sampled and value is None:
            raise FreezeError(f"{where}.{key} must be present when the control was sampled")
        if not sampled and value is not None:
            raise FreezeError(f"{where}.{key} must be null when the control was not sampled")
    if not sampled and lsm["variance_reduction_applicable"]:
        raise FreezeError(
            f"{where}: variance reduction cannot be applicable without a sampled control"
        )

    zero_width = bool(lsm["valuation_interval_is_zero_width"])
    if zero_width and float(lsm["standard_error"]) != 0.0:
        raise FreezeError(f"{where}: zero-width interval with non-zero standard error")
    if not zero_width and float(lsm["standard_error"]) <= 0.0:
        raise FreezeError(f"{where}: stochastic case with non-positive standard error")
    if zero_width and lsm["standardized_crr_gap"] is not None:
        raise FreezeError(
            f"{where}: deterministic case must not report a standardized gap"
        )
    if lsm["variance_reduction_ratio"] is not None and not (
        lsm["variance_reduction_applicable"] and not lsm["variance_reduction_is_infinite"]
    ):
        raise FreezeError(
            f"{where}: variance_reduction_ratio present on a case where it is undefined"
        )

    regression = _mapping(lsm["regression_summary"], f"{where}.regression_summary")
    _exact_keys(regression, REPORT_REGRESSION_KEYS, f"{where}.regression_summary")
    for key in (
        "constant_fallback_dates",
        "dates_with_in_the_money_paths",
        "exercise_dates_fitted",
        "minimum_in_the_money_paths",
    ):
        _integer(regression, key, f"{where}.regression_summary", minimum=0)
    _finite(regression, "minimum_relative_r_diagonal", f"{where}.regression_summary")


def _validate_report_summary(block: Mapping[str, Any], where: str) -> None:
    """Recompute every summary statistic from the case rows it claims to cover.

    This is the semantic-consistency gate. It is what makes the frozen snapshot
    trustworthy without re-running the engine: a report whose summaries were
    edited, or whose deterministic cases leaked into a stochastic statistic,
    fails here instead of being frozen.
    """
    summary = _mapping(block["summary"], f"{where}.summary")
    _exact_keys(summary, REPORT_SUMMARY_KEYS, f"{where}.summary")
    notes = _mapping(summary["notes"], f"{where}.summary.notes")
    for key in sorted(notes):
        _string(notes, key, f"{where}.summary.notes")

    cases = [_mapping(entry["lsm"], f"{where}.lsm") for entry in block["cases"]]
    deterministic = [c for c in cases if c["valuation_interval_is_zero_width"]]
    stochastic = [c for c in cases if not c["valuation_interval_is_zero_width"]]
    if not stochastic:
        raise FreezeError(f"{where}: no stochastic cases to summarise")

    gaps = [float(c["crr_reference_minus_lsm"]) for c in cases]
    expected: dict[str, Any] = {
        "cases": len(cases),
        "deterministic_zero_width_valuation_cases": len(deterministic),
        "stochastic_valuation_cases": len(stochastic),
        "mean_crr_reference_minus_lsm": sum(gaps) / len(gaps),
        "maximum_absolute_crr_reference_minus_lsm": max(abs(g) for g in gaps),
        "mean_standard_error": sum(float(c["standard_error"]) for c in cases) / len(cases),
        "mean_stochastic_standard_error": (
            sum(float(c["standard_error"]) for c in stochastic) / len(stochastic)
        ),
        "crr_reference_inside_stochastic_valuation_interval_cases": sum(
            1 for c in stochastic if c["crr_reference_inside_lsm_valuation_interval"]
        ),
        "variance_reduction_applicable_cases": sum(
            1 for c in cases if c["variance_reduction_applicable"]
        ),
        "variance_reduction_infinite_cases": sum(
            1 for c in cases if c["variance_reduction_is_infinite"]
        ),
        "total_constant_fallback_dates": sum(
            int(c["regression_summary"]["constant_fallback_dates"]) for c in cases
        ),
        "deterministic_maximum_absolute_crr_difference": (
            max(abs(float(c["crr_reference_minus_lsm"])) for c in deterministic)
            if deterministic
            else 0.0
        ),
    }
    # The reported minimum must come from applicable, finite, stochastic cases
    # only: deterministic immediate-exercise cases have an undefined 0/0 ratio
    # and must never be able to set or lower it.
    applicable = [
        float(c["variance_reduction_ratio"])
        for c in stochastic
        if c["variance_reduction_applicable"]
        and not c["variance_reduction_is_infinite"]
        and c["variance_reduction_ratio"] is not None
    ]
    if not applicable:
        raise FreezeError(f"{where}: no applicable finite variance-reduction ratio")
    expected["minimum_variance_reduction_ratio"] = min(applicable)

    for key, want in expected.items():
        got = summary[key]
        if isinstance(want, int) and not isinstance(want, bool):
            if got != want:
                raise FreezeError(
                    f"{where}.summary.{key} is {got!r}, recomputed {want!r}"
                )
        elif not math.isclose(float(got), float(want), rel_tol=1e-12, abs_tol=1e-15):
            raise FreezeError(f"{where}.summary.{key} is {got!r}, recomputed {want!r}")

    for case in deterministic:
        if case["crr_reference_inside_lsm_valuation_interval"]:
            raise FreezeError(
                f"{where}: a deterministic zero-width case claims interval containment"
            )


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def extract_snapshot(report: Any, *, report_filename: str, report_sha256: str) -> dict[str, Any]:
    """Build the frozen snapshot from a validated raw report."""
    if not SHA256_PATTERN.fullmatch(report_sha256):
        raise FreezeError("report_sha256 must be a lowercase hex SHA-256 digest")
    payload = validate_report(report)
    source = payload["source"]
    study = payload["study"]
    experiments = payload["experiments"]
    primary = next(e for e in experiments if e["experiment"]["role"] == "primary")

    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "source": {
            "build_configuration": source["build_configuration"],
            "case_source": study["case_source"],
            "case_source_sha256": study["case_source_sha256"],
            "config_file": source["config_file"],
            "config_sha256": source["config_sha256"],
            "crr_implementation_sha256": source["crr_implementation_sha256"],
            "cxx_compiler": source["cxx_compiler"],
            "lsm_header_sha256": source["lsm_header_sha256"],
            "lsm_implementation_sha256": source["lsm_implementation_sha256"],
            "lsm_source_sha256": source["lsm_source_sha256"],
            "oracle_version": source["oracle_version"],
            "report_filename": report_filename,
            "report_schema_version": payload["schema_version"],
            "report_sha256": report_sha256,
            "rng": source["rng"],
        },
        "study": {
            "base_seed": study["base_seed"],
            "case_count": len(study["case_names"]),
            "case_names": list(study["case_names"]),
            "case_rows": sum(len(e["cases"]) for e in experiments),
            "crr_reference_definition": study["crr_reference_definition"],
            "crr_reference_steps": study["crr_reference_steps"],
            "engine": study["engine"],
            "experiment_count": len(experiments),
            "maximum_training_memory_bytes": study["maximum_training_memory_bytes"],
            "name": study["name"],
            "primary_experiment": primary["experiment"]["name"],
        },
        "experiments": [_extract_experiment(entry) for entry in experiments],
        "primary_cases": [_extract_case(entry) for entry in primary["cases"]],
        "report_semantics": dict(REPORT_SEMANTICS),
        "limitations": list(LIMITATIONS),
    }
    validate_snapshot(snapshot)
    return snapshot


def _extract_experiment(entry: Mapping[str, Any]) -> dict[str, Any]:
    experiment = entry["experiment"]
    summary = entry["summary"]
    return {
        "cases": summary["cases"],
        "deterministic_maximum_absolute_crr_difference": summary[
            "deterministic_maximum_absolute_crr_difference"
        ],
        "deterministic_zero_width_valuation_cases": summary[
            "deterministic_zero_width_valuation_cases"
        ],
        "exercise_steps": experiment["exercise_steps"],
        "maximum_absolute_crr_reference_minus_lsm": summary[
            "maximum_absolute_crr_reference_minus_lsm"
        ],
        "mean_crr_reference_minus_lsm": summary["mean_crr_reference_minus_lsm"],
        "mean_standard_error": summary["mean_standard_error"],
        "mean_stochastic_valuation_standard_error": summary[
            "mean_stochastic_standard_error"
        ],
        "minimum_applicable_finite_variance_reduction_ratio": summary[
            "minimum_variance_reduction_ratio"
        ],
        "name": experiment["name"],
        "polynomial_degree": experiment["polynomial_degree"],
        "role": experiment["role"],
        "stochastic_valuation_cases": summary["stochastic_valuation_cases"],
        "stochastic_valuation_interval_containment_cases": summary[
            "crr_reference_inside_stochastic_valuation_interval_cases"
        ],
        "total_constant_fallback_dates": summary["total_constant_fallback_dates"],
        "training_paths": experiment["training_paths"],
        "valuation_paths": experiment["valuation_paths"],
        "variance_reduction_applicable_cases": summary["variance_reduction_applicable_cases"],
        "variance_reduction_infinite_cases": summary["variance_reduction_infinite_cases"],
    }


def _extract_case(entry: Mapping[str, Any]) -> dict[str, Any]:
    case = entry["case"]
    lsm = entry["lsm"]
    return {
        "constant_fallback_dates": lsm["regression_summary"]["constant_fallback_dates"],
        "control_variate_coefficient": lsm["control_variate_coefficient"],
        "crr_reference_minus_lsm": lsm["crr_reference_minus_lsm"],
        "crr_reference_price": entry["crr_reference"]["price"],
        "dividend_yield": case["dividend_yield"],
        "exercise_expectation": case["exercise_expectation"],
        "lsm_price": lsm["price"],
        "maturity": case["maturity"],
        "name": case["name"],
        "option_type": case["option_type"],
        "rate": case["rate"],
        "spot": case["spot"],
        "standardized_crr_gap": lsm["standardized_crr_gap"],
        "strike": case["strike"],
        "valuation_early_exercise_fraction": lsm["valuation_early_exercise_fraction"],
        "valuation_interval_is_zero_width": lsm["valuation_interval_is_zero_width"],
        "valuation_only_standard_error": lsm["standard_error"],
        "variance_reduction_applicable": lsm["variance_reduction_applicable"],
        "variance_reduction_is_infinite": lsm["variance_reduction_is_infinite"],
        "variance_reduction_ratio": lsm["variance_reduction_ratio"],
        "volatility": case["volatility"],
    }


# --------------------------------------------------------------------------
# Snapshot validation
# --------------------------------------------------------------------------


def validate_snapshot(snapshot: Any) -> Mapping[str, Any]:
    """Strictly validate a frozen snapshot on its own terms and return it."""
    payload = _mapping(snapshot, "snapshot")
    _exact_keys(payload, SNAPSHOT_KEYS, "snapshot")
    schema = _string(payload, "schema_version", "snapshot")
    if schema != SNAPSHOT_SCHEMA_VERSION:
        raise FreezeError(
            f"snapshot schema_version must be '{SNAPSHOT_SCHEMA_VERSION}', got '{schema}'"
        )

    source = _mapping(payload["source"], "snapshot.source")
    _exact_keys(source, SNAPSHOT_SOURCE_KEYS, "snapshot.source")
    for key in (
        "case_source_sha256",
        "config_sha256",
        "crr_implementation_sha256",
        "lsm_header_sha256",
        "lsm_implementation_sha256",
        "lsm_source_sha256",
        "report_sha256",
    ):
        _digest(source, key, "snapshot.source")
    for key in (
        "build_configuration",
        "case_source",
        "config_file",
        "cxx_compiler",
        "oracle_version",
        "report_filename",
        "rng",
    ):
        _string(source, key, "snapshot.source")
    report_schema = _string(source, "report_schema_version", "snapshot.source")
    if report_schema != REPORT_SCHEMA_VERSION:
        raise FreezeError(
            f"snapshot.source.report_schema_version must be '{REPORT_SCHEMA_VERSION}'"
        )
    if "/" in source["report_filename"] or "\\" in source["report_filename"]:
        raise FreezeError("snapshot.source.report_filename must be a bare filename")

    study = _mapping(payload["study"], "snapshot.study")
    _exact_keys(study, SNAPSHOT_STUDY_KEYS, "snapshot.study")
    for key in ("crr_reference_definition", "engine", "name", "primary_experiment"):
        _string(study, key, "snapshot.study")
    _integer(study, "base_seed", "snapshot.study", minimum=0)
    _integer(study, "crr_reference_steps", "snapshot.study", minimum=1)
    _integer(study, "maximum_training_memory_bytes", "snapshot.study", minimum=1)
    case_count = _integer(study, "case_count", "snapshot.study", minimum=1)
    experiment_count = _integer(study, "experiment_count", "snapshot.study", minimum=1)
    case_rows = _integer(study, "case_rows", "snapshot.study", minimum=1)
    case_names = _sequence(study["case_names"], "snapshot.study.case_names")
    if len(case_names) != case_count:
        raise FreezeError("snapshot.study.case_count disagrees with case_names")
    if len(set(case_names)) != len(case_names):
        raise FreezeError("snapshot.study.case_names contains duplicates")
    if case_rows != case_count * experiment_count:
        raise FreezeError(
            "snapshot.study.case_rows must equal case_count * experiment_count"
        )

    experiments = _sequence(payload["experiments"], "snapshot.experiments")
    if len(experiments) != experiment_count:
        raise FreezeError("snapshot.experiments length disagrees with experiment_count")
    primary_seen = 0
    names: set[str] = set()
    for index, entry in enumerate(experiments):
        where = f"snapshot.experiments[{index}]"
        block = _mapping(entry, where)
        _exact_keys(block, SNAPSHOT_EXPERIMENT_KEYS, where)
        name = _string(block, "name", where)
        if name in names:
            raise FreezeError(f"{where}.name is duplicated")
        names.add(name)
        role = _string(block, "role", where)
        if role == "primary":
            primary_seen += 1
            if name != study["primary_experiment"]:
                raise FreezeError(f"{where} is primary but is not study.primary_experiment")
        for key in (
            "exercise_steps",
            "polynomial_degree",
            "training_paths",
            "valuation_paths",
        ):
            _integer(block, key, where, minimum=1)
        for key in (
            "cases",
            "deterministic_zero_width_valuation_cases",
            "stochastic_valuation_cases",
            "stochastic_valuation_interval_containment_cases",
            "total_constant_fallback_dates",
            "variance_reduction_applicable_cases",
            "variance_reduction_infinite_cases",
        ):
            _integer(block, key, where, minimum=0)
        for key in (
            "deterministic_maximum_absolute_crr_difference",
            "maximum_absolute_crr_reference_minus_lsm",
            "mean_crr_reference_minus_lsm",
            "mean_standard_error",
            "mean_stochastic_valuation_standard_error",
            "minimum_applicable_finite_variance_reduction_ratio",
        ):
            _finite(block, key, where)
        if block["cases"] != case_count:
            raise FreezeError(f"{where}.cases must equal study.case_count")
        if (
            block["deterministic_zero_width_valuation_cases"]
            + block["stochastic_valuation_cases"]
            != block["cases"]
        ):
            raise FreezeError(
                f"{where}: deterministic and stochastic counts must partition the cases"
            )
        if (
            block["stochastic_valuation_interval_containment_cases"]
            > block["stochastic_valuation_cases"]
        ):
            raise FreezeError(
                f"{where}: containment count exceeds the stochastic case count"
            )
        if block["maximum_absolute_crr_reference_minus_lsm"] < abs(
            block["mean_crr_reference_minus_lsm"]
        ):
            raise FreezeError(f"{where}: maximum absolute gap is below the mean gap")
    if primary_seen != 1:
        raise FreezeError("snapshot must contain exactly one primary experiment")

    _validate_snapshot_primary_cases(payload, experiments, list(case_names))

    semantics = _mapping(payload["report_semantics"], "snapshot.report_semantics")
    _exact_keys(semantics, SNAPSHOT_SEMANTICS_KEYS, "snapshot.report_semantics")
    for key in sorted(SNAPSHOT_SEMANTICS_KEYS):
        _string(semantics, key, "snapshot.report_semantics")

    limitations = _sequence(payload["limitations"], "snapshot.limitations")
    if not limitations or any(not isinstance(item, str) or not item for item in limitations):
        raise FreezeError("snapshot.limitations must be a non-empty array of strings")
    return payload


def _validate_snapshot_primary_cases(
    payload: Mapping[str, Any],
    experiments: Sequence[Any],
    case_names: list[str],
) -> None:
    primary_name = payload["study"]["primary_experiment"]
    primary = next(e for e in experiments if e["name"] == primary_name)
    cases = _sequence(payload["primary_cases"], "snapshot.primary_cases")
    if [_string(_mapping(c, "snapshot.primary_cases[]"), "name", "case") for c in cases] != (
        case_names
    ):
        raise FreezeError(
            "snapshot.primary_cases must list exactly the study case names, in order"
        )

    deterministic = 0
    fallbacks = 0
    gaps: list[float] = []
    ratios: list[float] = []
    for index, entry in enumerate(cases):
        where = f"snapshot.primary_cases[{index}]"
        case = _mapping(entry, where)
        _exact_keys(case, SNAPSHOT_PRIMARY_CASE_KEYS, where)
        _string(case, "exercise_expectation", where)
        option_type = _string(case, "option_type", where)
        if option_type not in {"call", "put"}:
            raise FreezeError(f"{where}.option_type must be 'call' or 'put'")
        for key in (
            "control_variate_coefficient",
            "crr_reference_minus_lsm",
            "crr_reference_price",
            "dividend_yield",
            "lsm_price",
            "maturity",
            "rate",
            "spot",
            "strike",
            "valuation_early_exercise_fraction",
            "valuation_only_standard_error",
            "volatility",
        ):
            _finite(case, key, where)
        _integer(case, "constant_fallback_dates", where, minimum=0)
        zero_width = _bool(case, "valuation_interval_is_zero_width", where)
        applicable = _bool(case, "variance_reduction_applicable", where)
        infinite = _bool(case, "variance_reduction_is_infinite", where)
        standardized = _optional_finite(case, "standardized_crr_gap", where)
        ratio = _optional_finite(case, "variance_reduction_ratio", where)

        error = _finite(case, "valuation_only_standard_error", where)
        if zero_width:
            deterministic += 1
            if error != 0.0:
                raise FreezeError(f"{where}: zero-width case with non-zero standard error")
            if standardized is not None:
                raise FreezeError(
                    f"{where}: deterministic case must not report a standardized gap"
                )
        else:
            if error <= 0.0:
                raise FreezeError(f"{where}: stochastic case with non-positive error")
            if standardized is None:
                raise FreezeError(f"{where}: stochastic case must report a standardized gap")
            if not math.isclose(
                standardized,
                _finite(case, "crr_reference_minus_lsm", where) / error,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise FreezeError(f"{where}: standardized gap disagrees with gap / error")
        if ratio is None:
            if applicable and not infinite and not zero_width:
                raise FreezeError(f"{where}: applicable finite case is missing its ratio")
        else:
            if not applicable or infinite:
                raise FreezeError(f"{where}: ratio present where it is undefined")
            if not zero_width:
                ratios.append(ratio)
        fraction = _finite(case, "valuation_early_exercise_fraction", where)
        if not 0.0 <= fraction <= 1.0:
            raise FreezeError(f"{where}.valuation_early_exercise_fraction must be in [0, 1]")
        gaps.append(_finite(case, "crr_reference_minus_lsm", where))
        fallbacks += int(case["constant_fallback_dates"])

    # The primary experiment's frozen summary must be exactly what its own
    # frozen case rows imply, so the two halves of the snapshot cannot drift.
    checks = (
        ("cases", len(cases), primary["cases"]),
        (
            "deterministic_zero_width_valuation_cases",
            deterministic,
            primary["deterministic_zero_width_valuation_cases"],
        ),
        (
            "stochastic_valuation_cases",
            len(cases) - deterministic,
            primary["stochastic_valuation_cases"],
        ),
        ("total_constant_fallback_dates", fallbacks, primary["total_constant_fallback_dates"]),
    )
    for label, got, want in checks:
        if got != want:
            raise FreezeError(
                f"snapshot.primary_cases imply {label}={got}, "
                f"the primary experiment records {want}"
            )
    numeric = (
        ("mean_crr_reference_minus_lsm", sum(gaps) / len(gaps)),
        ("maximum_absolute_crr_reference_minus_lsm", max(abs(g) for g in gaps)),
        ("minimum_applicable_finite_variance_reduction_ratio", min(ratios)),
    )
    for label, got in numeric:
        if not math.isclose(got, float(primary[label]), rel_tol=1e-12, abs_tol=1e-15):
            raise FreezeError(
                f"snapshot.primary_cases imply {label}={got!r}, "
                f"the primary experiment records {primary[label]!r}"
            )


def verify_provenance(snapshot: Mapping[str, Any]) -> dict[str, str]:
    """Reconcile the snapshot's recorded digests against current repository files.

    Uses only checked-in files: no ignored artifact and no compiled binding.
    """
    source = snapshot["source"]
    verified: dict[str, str] = {}

    config_file = str(source["config_file"])
    if "/" in config_file or "\\" in config_file:
        raise FreezeError("snapshot.source.config_file must be a bare filename")
    paths = {
        "config_sha256": PROJECT_ROOT / "configs" / config_file,
        "case_source_sha256": PROJECT_ROOT / str(source["case_source"]),
        "lsm_header_sha256": PROJECT_ROOT
        / DIRECT_PROVENANCE_PATHS["lsm_header_sha256"],
        "lsm_source_sha256": PROJECT_ROOT
        / DIRECT_PROVENANCE_PATHS["lsm_source_sha256"],
    }
    case_source = paths["case_source_sha256"].resolve()
    if not case_source.is_relative_to(PROJECT_ROOT.resolve()):
        raise FreezeError("snapshot.source.case_source escapes the repository")
    for field, path in paths.items():
        if not path.is_file():
            raise FreezeError(f"{field} refers to missing file '{path}'")
        actual = _sha256_file(path)
        if actual != source[field]:
            raise FreezeError(
                f"{field} does not match {path.relative_to(PROJECT_ROOT)}: "
                f"snapshot records {source[field]}, file hashes to {actual}"
            )
        verified[field] = actual

    for field, members in COMPOSITE_PROVENANCE.items():
        actual = _composite_digest(members)
        if actual != source[field]:
            raise FreezeError(
                f"{field} does not match the current C++ sources: "
                f"snapshot records {source[field]}, sources hash to {actual}"
            )
        verified[field] = actual
    return verified


# --------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------


def serialise(snapshot: Mapping[str, Any]) -> str:
    """Return the canonical text form: sorted keys, two-space indent, trailing newline.

    Deterministic by construction and free of wall-clock or environment fields,
    so regenerating from the same report reproduces identical bytes.
    """
    return json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically via a same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_json(path: Path, description: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise FreezeError(f"cannot read {description} '{path}': {error}") from error
    except json.JSONDecodeError as error:
        raise FreezeError(f"{description} '{path}' is not valid JSON: {error}") from error


def snapshot_from_report(report_path: Path) -> dict[str, Any]:
    """Load, hash, validate, and extract a snapshot from a raw report file."""
    report = load_json(report_path, "report")
    return extract_snapshot(
        report,
        report_filename=report_path.name,
        report_sha256=_sha256_file(report_path),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "Reviewed raw cross-check report. Required to generate; optional with "
            f"--check to also detect staleness. Default when generating: {DEFAULT_REPORT}"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SNAPSHOT,
        help="Frozen snapshot path to write, or to validate under --check.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Validate the checked-in snapshot and its repository provenance without "
            "writing. Does not require the ignored raw report."
        ),
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Permit overwriting an existing snapshot when generating.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        if arguments.check:
            if arguments.update:
                raise FreezeError("--update cannot be combined with --check")
            snapshot = validate_snapshot(load_json(arguments.output, "snapshot"))
            verify_provenance(snapshot)
            current = arguments.output.read_text(encoding="utf-8")
            if current != serialise(snapshot):
                raise FreezeError(
                    f"snapshot '{arguments.output}' is not canonically serialised; "
                    f"regenerate it with {Path(__file__).name}"
                )
            if arguments.report is not None:
                fresh = snapshot_from_report(arguments.report)
                if serialise(fresh) != current:
                    raise FreezeError(
                        f"snapshot '{arguments.output}' is stale with respect to "
                        f"'{arguments.report}'; regenerate it with --update"
                    )
            print(f"snapshot ok: {arguments.output}")
            return 0

        report_path = arguments.report if arguments.report is not None else DEFAULT_REPORT
        if arguments.output.exists() and not arguments.update:
            raise FreezeError(
                f"refusing to overwrite existing snapshot '{arguments.output}'; "
                "pass --update to replace it deliberately"
            )
        snapshot = snapshot_from_report(report_path)
        verify_provenance(snapshot)
        write_atomic(arguments.output, serialise(snapshot))
        print(f"wrote {arguments.output}")
    except FreezeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        # The documented contract is exit 2 for I/O failure too, so an
        # unwritable directory or a full disk must not escape as a traceback.
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
