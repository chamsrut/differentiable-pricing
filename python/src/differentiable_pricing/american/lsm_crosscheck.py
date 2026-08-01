"""Run a deterministic LSM/CRR American-option cross-check.

The C++ engine owns simulation, regression, stopping, uncertainty, the
European control variate, and CRR pricing. This module validates the versioned
study, derives distinct explicit seeds, orchestrates named cases, and writes a
deterministic provenance-rich report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from differentiable_pricing import _core
from differentiable_pricing.american.convergence import (
    ConvergenceCase,
    load_convergence_config,
)

SCHEMA_VERSION: Final = "american-lsm-crosscheck/1"
REPORT_SCHEMA_VERSION: Final = "american-lsm-crosscheck-report/1"
ENGINE_NAME: Final = "dp::least_squares_monte_carlo/v1"
PROJECT_ROOT: Final = Path(__file__).resolve().parents[4]
EXPERIMENT_ROLES: Final = frozenset(
    {
        "path_convergence",
        "exercise_grid_convergence",
        "reference",
        "basis_sensitivity",
        "primary",
    }
)
_TOP_LEVEL_KEYS: Final = frozenset({"schema_version", "study", "experiments"})
_STUDY_KEYS: Final = frozenset(
    {
        "name",
        "engine",
        "case_source",
        "case_source_sha256",
        "case_names",
        "crr_reference_steps",
        "crr_batch_threads",
        "base_seed",
        "maximum_training_memory_bytes",
    }
)
_EXPERIMENT_KEYS: Final = frozenset(
    {
        "name",
        "role",
        "exercise_steps",
        "training_paths",
        "valuation_paths",
        "polynomial_degree",
    }
)
# Emitted beside the counts they qualify, so the caveat travels with the number
# into any consumer that reads only this report.
_SUMMARY_NOTES: Final = {
    "crr_reference_inside_stochastic_valuation_interval_cases": (
        "Counts stochastic cases only. The interval describes valuation "
        "sampling error for an already-frozen policy; it excludes "
        "policy-fitting error and the discrete-exercise gap, which are "
        "systematic and do not shrink with valuation paths. This is NOT a "
        "coverage or calibration statistic: it has no nominal rate, a low or "
        "zero count is expected once the interval is narrower than those "
        "biases, and it must not be used as an acceptance gate."
    ),
    "deterministic_zero_width_valuation_cases": (
        "Cases whose valuation estimator has zero variance, from immediate "
        "exercise at time zero or from a control variate that reproduces the "
        "payoff exactly. Their interval is a single point, so exact "
        "containment of any finite-step tree value is arithmetically "
        "impossible regardless of agreement. Their CRR differences are "
        "reported separately instead."
    ),
    "minimum_variance_reduction_ratio": (
        "Minimum over applicable finite stochastic cases only. Deterministic "
        "immediate-exercise cases have zero raw and zero adjusted variance, "
        "an undefined 0/0 ratio, and are excluded. Applicable cases whose "
        "control variate removes all variance are counted in "
        "variance_reduction_infinite_cases and excluded from this minimum."
    ),
}


class LsmCrosscheckError(RuntimeError):
    """Raised when the cross-check contract or execution is invalid."""


@dataclass(frozen=True, slots=True)
class LsmExperiment:
    """One declared LSM numerical experiment."""

    name: str
    role: str
    exercise_steps: int
    training_paths: int
    valuation_paths: int
    polynomial_degree: int


@dataclass(frozen=True, slots=True)
class LsmCrosscheckConfig:
    """Validated cross-check configuration and referenced CRR cases."""

    name: str
    engine: str
    case_source: str
    case_source_sha256: str
    case_names: tuple[str, ...]
    crr_reference_steps: int
    crr_batch_threads: int
    base_seed: int
    maximum_training_memory_bytes: int
    experiments: tuple[LsmExperiment, ...]
    cases: tuple[ConvergenceCase, ...]
    source_name: str
    source_sha256: str


def load_lsm_crosscheck_config(path: Path | str) -> LsmCrosscheckConfig:
    """Load and strictly validate a versioned LSM cross-check."""
    config_path = Path(path)
    try:
        raw = config_path.read_bytes()
    except OSError as error:
        raise LsmCrosscheckError(
            f"cannot read configuration '{config_path}': {error}"
        ) from error
    try:
        document = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise LsmCrosscheckError(
            f"configuration '{config_path}' is not valid UTF-8 TOML: {error}"
        ) from error
    return parse_lsm_crosscheck_config(
        document,
        source_name=config_path.name,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        project_root=_find_project_root(config_path),
    )


def parse_lsm_crosscheck_config(
    document: Mapping[str, Any],
    *,
    source_name: str,
    source_sha256: str,
    project_root: Path = PROJECT_ROOT,
) -> LsmCrosscheckConfig:
    """Validate an already-parsed cross-check configuration."""
    _reject_unknown(document, _TOP_LEVEL_KEYS, "top-level")
    schema_version = _require_string(document, "schema_version", "top-level")
    if schema_version != SCHEMA_VERSION:
        raise LsmCrosscheckError(
            f"unsupported schema_version '{schema_version}'; "
            f"expected '{SCHEMA_VERSION}'"
        )

    study = _require_table(document, "study", "top-level")
    _reject_unknown(study, _STUDY_KEYS, "[study]")
    engine = _require_string(study, "engine", "[study]")
    if engine != ENGINE_NAME:
        raise LsmCrosscheckError(f"[study].engine must be '{ENGINE_NAME}'")
    case_source = _require_string(study, "case_source", "[study]")
    source_path = Path(case_source)
    if source_path.is_absolute() or ".." in source_path.parts:
        raise LsmCrosscheckError(
            "[study].case_source must be a repository-relative path"
        )
    declared_case_hash = _require_sha256(
        study, "case_source_sha256", "[study]"
    )
    resolved_case_path = (project_root / source_path).resolve()
    try:
        resolved_case_path.relative_to(project_root.resolve())
    except ValueError as error:
        raise LsmCrosscheckError(
            "[study].case_source escapes the repository"
        ) from error
    try:
        actual_case_hash = hashlib.sha256(resolved_case_path.read_bytes()).hexdigest()
    except OSError as error:
        raise LsmCrosscheckError(
            f"cannot read declared case source '{case_source}': {error}"
        ) from error
    if actual_case_hash != declared_case_hash:
        raise LsmCrosscheckError(
            "[study].case_source_sha256 does not match the referenced file"
        )
    crr_config = load_convergence_config(resolved_case_path)

    case_names = _require_string_array(study, "case_names", "[study]")
    source_cases = {case.name: case for case in crr_config.cases}
    unknown_cases = sorted(set(case_names) - set(source_cases))
    if unknown_cases:
        raise LsmCrosscheckError(
            f"[study].case_names contains unknown cases: {', '.join(unknown_cases)}"
        )
    cases = tuple(source_cases[name] for name in case_names)

    reference_steps = _require_int(study, "crr_reference_steps", "[study]")
    if reference_steps >= int(_core.maximum_crr_steps):
        raise LsmCrosscheckError(
            "[study].crr_reference_steps must leave room for the adjacent N+1 tree"
        )
    threads = _require_int(study, "crr_batch_threads", "[study]")
    if threads > int(_core.maximum_crr_batch_threads):
        raise LsmCrosscheckError(
            "[study].crr_batch_threads exceeds the engine limit"
        )
    base_seed = _require_int(study, "base_seed", "[study]")
    if base_seed > 2**64 - 1:
        raise LsmCrosscheckError("[study].base_seed exceeds uint64")
    memory_limit = _require_int(
        study, "maximum_training_memory_bytes", "[study]"
    )
    if memory_limit > int(_core.maximum_lsm_training_memory_bytes):
        raise LsmCrosscheckError(
            "[study].maximum_training_memory_bytes exceeds the engine limit"
        )

    raw_experiments = document.get("experiments")
    if not isinstance(raw_experiments, list) or not raw_experiments:
        raise LsmCrosscheckError(
            "top-level experiments must be a non-empty array of tables"
        )
    experiments = tuple(
        _parse_experiment(value, index, memory_limit)
        for index, value in enumerate(raw_experiments)
    )
    names = [experiment.name for experiment in experiments]
    if len(set(names)) != len(names):
        raise LsmCrosscheckError("experiment names must be unique")
    roles = [experiment.role for experiment in experiments]
    if roles.count("primary") != 1:
        raise LsmCrosscheckError(
            "exactly one experiment must have role = 'primary'"
        )
    if "reference" not in roles:
        raise LsmCrosscheckError(
            "at least one experiment must have role = 'reference'"
        )

    return LsmCrosscheckConfig(
        name=_require_string(study, "name", "[study]"),
        engine=engine,
        case_source=case_source,
        case_source_sha256=declared_case_hash,
        case_names=case_names,
        crr_reference_steps=reference_steps,
        crr_batch_threads=threads,
        base_seed=base_seed,
        maximum_training_memory_bytes=memory_limit,
        experiments=experiments,
        cases=cases,
        source_name=source_name,
        source_sha256=source_sha256,
    )


def run_lsm_crosscheck(config: LsmCrosscheckConfig) -> dict[str, Any]:
    """Execute the declared LSM experiments and high-step CRR cross-check."""
    references = _crr_references(config)
    experiments: list[dict[str, Any]] = []
    for experiment in config.experiments:
        case_results = []
        for case in config.cases:
            training_seed = _derive_seed(
                config.base_seed, experiment.name, case.name, "policy_training"
            )
            valuation_seed = _derive_seed(
                config.base_seed, experiment.name, case.name, "policy_valuation"
            )
            try:
                result = _core.lsm_price(
                    case.option_type,
                    case.spot,
                    case.strike,
                    case.maturity,
                    case.rate,
                    case.dividend_yield,
                    case.volatility,
                    experiment.exercise_steps,
                    experiment.training_paths,
                    experiment.valuation_paths,
                    experiment.polynomial_degree,
                    training_seed,
                    valuation_seed,
                    config.maximum_training_memory_bytes,
                )
            except (OverflowError, ValueError) as error:
                raise LsmCrosscheckError(
                    f"LSM pricing failed for experiment "
                    f"'{experiment.name}' case '{case.name}': {error}"
                ) from error
            reference = references[case.name]
            standard_error = float(result["standard_error"])
            difference = reference["price"] - float(result["price"])
            regressions = result.pop("regressions")
            _apply_variance_reduction_semantics(result)
            _apply_european_sampling_semantics(result)
            result["regression_summary"] = _regression_summary(regressions)
            result["valuation_early_exercise_fraction"] = (
                int(result["valuation_early_exercise_paths"])
                / experiment.valuation_paths
            )
            result["crr_reference_minus_lsm"] = difference
            result["standardized_crr_gap"] = (
                difference / standard_error if standard_error > 0.0 else None
            )
            # A deterministic estimator has a zero-width interval, so exact
            # containment of any finite-step tree value is arithmetically
            # impossible. Those cases are labelled and counted separately
            # rather than being folded into a containment statistic.
            result["valuation_interval_is_zero_width"] = standard_error <= 0.0
            result["crr_reference_inside_lsm_valuation_interval"] = (
                float(result["confidence_interval_lower"])
                <= reference["price"]
                <= float(result["confidence_interval_upper"])
            )
            case_results.append(
                {
                    "case": asdict(case),
                    "seeds": {
                        "derivation": (
                            "sha256(base_seed|experiment|case|stream), "
                            "first 64 bits"
                        ),
                        "training": training_seed,
                        "valuation": valuation_seed,
                    },
                    "crr_reference": reference,
                    "lsm": result,
                }
            )
        experiments.append(
            {
                "experiment": asdict(experiment),
                "cases": case_results,
                "summary": _experiment_summary(case_results),
            }
        )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "study": {
            "name": config.name,
            "engine": config.engine,
            "case_source": config.case_source,
            "case_source_sha256": config.case_source_sha256,
            "case_names": list(config.case_names),
            "crr_reference_steps": config.crr_reference_steps,
            "crr_reference_definition": "adjacent average at N and N+1",
            "base_seed": config.base_seed,
            "maximum_training_memory_bytes": (
                config.maximum_training_memory_bytes
            ),
        },
        "experiments": experiments,
        "source": {
            "config_file": config.source_name,
            "config_sha256": config.source_sha256,
            "oracle_version": str(_core.__version__),
            "build_configuration": str(_core.__build_configuration__),
            "cxx_compiler": str(_core.__cxx_compiler__),
            "lsm_header_sha256": str(_core.__lsm_header_sha256__),
            "lsm_source_sha256": str(_core.__lsm_source_sha256__),
            "lsm_implementation_sha256": str(
                _core.__lsm_implementation_sha256__
            ),
            "crr_implementation_sha256": str(
                _core.__crr_implementation_sha256__
            ),
            "rng": "splitmix64_box_muller_antithetic_v1",
        },
        "interpretation": {
            "estimand": (
                "The discounted value of an independently evaluated learned "
                "Bermudan stopping policy on the declared exercise grid."
            ),
            "lower_bound": (
                "For a fixed learned policy, the raw expectation is no greater "
                "than the same-grid optimal stopping value. Finite-sample "
                "confidence intervals describe valuation sampling error, not "
                "policy or exercise-grid bias. LSM therefore sitting below the "
                "finer-grid CRR reference is expected behaviour, not a defect."
            ),
            "valuation_interval": (
                "The reported interval is valuation-only: it quantifies "
                "sampling noise around the value of an already-frozen policy. "
                "It has no nominal coverage rate for the same-grid optimal "
                "value, the continuously exercisable American value, or any "
                "tree reference, and no count derived from it is an acceptance "
                "gate. As valuation paths increase the interval narrows around "
                "a fixed policy and grid bias, so the containment count is "
                "expected to fall rather than approach any target fraction."
            ),
            "variance_reduction": (
                "Reported only where mathematically defined. Deterministic "
                "immediate-exercise cases have an undefined 0/0 ratio and are "
                "excluded from every variance-reduction summary; they cannot "
                "set or lower a reported minimum."
            ),
            "non_finite_rejection": (
                "The engine validates every price-domain quantity it derives "
                "and raises rather than returning a non-finite or corrupted "
                "price, so every value in this report is finite by "
                "construction."
            ),
            "control_variate": (
                "A coefficient estimated on policy-training pairs is frozen "
                "before valuation. The primary estimate subtracts that "
                "coefficient times the same-path European deviation from its "
                "analytic Black--Scholes expectation."
            ),
            "standard_error": (
                "Antithetic pair averages are the independent observations; "
                "individual paths are not counted as independent."
            ),
            "crr_reference": (
                "The high-step CRR adjacent average is an internal model "
                "cross-check, not market truth and not a formal error bound."
            ),
        },
    }


def write_report(report: Mapping[str, Any], output: Path | str) -> None:
    """Atomically create a report and refuse to overwrite an existing file."""
    output_path = Path(output)
    if output_path.exists():
        raise LsmCrosscheckError(
            f"refusing to overwrite existing output '{output_path}'"
        )
    try:
        payload = (
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode()
    except (TypeError, ValueError) as error:
        raise LsmCrosscheckError(
            f"report for '{output_path}' is not finite canonical JSON: {error}"
        ) from error
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary_path, output_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    except FileExistsError as error:
        raise LsmCrosscheckError(
            f"refusing to overwrite existing output '{output_path}'"
        ) from error
    except OSError as error:
        raise LsmCrosscheckError(
            f"cannot write output '{output_path}': {error}"
        ) from error


def _parse_experiment(
    value: Any, index: int, memory_limit: int
) -> LsmExperiment:
    where = f"[[experiments]] at index {index}"
    if not isinstance(value, dict):
        raise LsmCrosscheckError(f"{where} must be a table")
    _reject_unknown(value, _EXPERIMENT_KEYS, where)
    role = _require_string(value, "role", where)
    if role not in EXPERIMENT_ROLES:
        raise LsmCrosscheckError(f"{where}.role is unsupported")
    experiment = LsmExperiment(
        name=_require_string(value, "name", where),
        role=role,
        exercise_steps=_require_int(value, "exercise_steps", where),
        training_paths=_require_even_paths(value, "training_paths", where),
        valuation_paths=_require_even_paths(value, "valuation_paths", where),
        polynomial_degree=_require_int(value, "polynomial_degree", where),
    )
    if experiment.exercise_steps > int(_core.maximum_lsm_exercise_steps):
        raise LsmCrosscheckError(
            f"{where}.exercise_steps exceeds the engine limit"
        )
    if experiment.polynomial_degree > int(
        _core.maximum_lsm_polynomial_degree
    ):
        raise LsmCrosscheckError(
            f"{where}.polynomial_degree exceeds the engine limit"
        )
    try:
        estimated_bytes = int(
            _core.lsm_training_memory_bytes(
                experiment.exercise_steps,
                experiment.training_paths,
                experiment.valuation_paths,
                experiment.polynomial_degree,
                1,
                2,
                memory_limit,
            )
        )
    except (OverflowError, ValueError) as error:
        raise LsmCrosscheckError(f"{where} is invalid: {error}") from error
    if estimated_bytes > memory_limit:
        raise LsmCrosscheckError(
            f"{where} requires {estimated_bytes} training bytes, "
            "exceeding the study limit"
        )
    return experiment


def _crr_references(
    config: LsmCrosscheckConfig,
) -> dict[str, dict[str, Any]]:
    columns: dict[str, list[Any]] = {
        "option_types": [],
        "exercise_styles": [],
        "spots": [],
        "strikes": [],
        "maturities": [],
        "rates": [],
        "dividend_yields": [],
        "volatilities": [],
        "steps": [],
    }
    for steps in (
        config.crr_reference_steps,
        config.crr_reference_steps + 1,
    ):
        for case in config.cases:
            columns["option_types"].append(case.option_type)
            columns["exercise_styles"].append("american")
            columns["spots"].append(case.spot)
            columns["strikes"].append(case.strike)
            columns["maturities"].append(case.maturity)
            columns["rates"].append(case.rate)
            columns["dividend_yields"].append(case.dividend_yield)
            columns["volatilities"].append(case.volatility)
            columns["steps"].append(steps)
    result = _core.crr_price_batch(
        columns["option_types"],
        columns["exercise_styles"],
        columns["spots"],
        columns["strikes"],
        columns["maturities"],
        columns["rates"],
        columns["dividend_yields"],
        columns["volatilities"],
        columns["steps"],
        config.crr_batch_threads,
    )
    case_count = len(config.cases)
    references: dict[str, dict[str, Any]] = {}
    for index, case in enumerate(config.cases):
        at_steps = float(result["price"][index])
        at_next = float(result["price"][case_count + index])
        references[case.name] = {
            "price": 0.5 * (at_steps + at_next),
            "at_steps": at_steps,
            "at_steps_plus_one": at_next,
            "absolute_pair_gap": abs(at_steps - at_next),
            "steps": config.crr_reference_steps,
        }
    return references


def _derive_seed(
    base_seed: int, experiment: str, case: str, stream: str
) -> int:
    payload = f"{base_seed}|{experiment}|{case}|{stream}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _apply_variance_reduction_semantics(result: dict[str, Any]) -> None:
    """Represent variance reduction only where it is mathematically defined.

    An immediate-exercise policy has zero raw and zero adjusted variance, so the
    ratio is the undefined form 0/0 and is not a measurement. An applicable
    control variate that removes all variance is a measurement, reported as an
    explicit infinite flag with a JSON-safe ``None`` ratio.
    """
    applicable = bool(result["variance_reduction_applicable"])
    ratio = float(result["variance_reduction_ratio"])
    result["variance_reduction_is_infinite"] = applicable and math.isinf(ratio)
    if not applicable or not math.isfinite(ratio):
        result["variance_reduction_ratio"] = None


def _apply_european_sampling_semantics(result: dict[str, Any]) -> None:
    """Suppress European Monte Carlo fields that were never sampled.

    An immediate-exercise policy runs no valuation simulation. Reporting the
    analytic Black--Scholes value in a field named for a Monte Carlo estimate
    would present a closed form as an observation, so both sampled fields become
    ``None``. The separate analytic price is always retained.
    """
    if not bool(result["european_monte_carlo_sampled"]):
        result["european_monte_carlo_price"] = None
        result["european_standard_error"] = None


def _regression_summary(
    regressions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    observed = [
        row for row in regressions if int(row["in_the_money_paths"]) > 0
    ]
    return {
        "exercise_dates_fitted": len(regressions),
        "dates_with_in_the_money_paths": len(observed),
        "constant_fallback_dates": sum(
            bool(row["used_constant_fallback"]) for row in regressions
        ),
        "minimum_in_the_money_paths": min(
            (int(row["in_the_money_paths"]) for row in observed),
            default=0,
        ),
        "minimum_relative_r_diagonal": min(
            (
                float(row["minimum_relative_r_diagonal"])
                for row in observed
                if not bool(row["used_constant_fallback"])
            ),
            default=None,
        ),
    }


def _experiment_summary(
    case_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    lsm_rows = [row["lsm"] for row in case_results]
    gaps = [float(row["crr_reference_minus_lsm"]) for row in lsm_rows]
    stochastic = [
        row for row in lsm_rows if not row["valuation_interval_is_zero_width"]
    ]
    deterministic = [
        row for row in lsm_rows if row["valuation_interval_is_zero_width"]
    ]
    deterministic_gaps = [
        abs(float(row["crr_reference_minus_lsm"])) for row in deterministic
    ]
    # Only applicable, finite ratios are measurements. Deterministic
    # immediate-exercise cases are excluded entirely, so they can neither set
    # nor lower this minimum; applicable infinite ratios are counted separately.
    applicable = [
        row for row in lsm_rows if bool(row["variance_reduction_applicable"])
    ]
    finite_variance_reductions = [
        float(row["variance_reduction_ratio"])
        for row in applicable
        if row["variance_reduction_ratio"] is not None
    ]
    return {
        "cases": len(case_results),
        "mean_crr_reference_minus_lsm": sum(gaps) / len(gaps),
        "maximum_absolute_crr_reference_minus_lsm": max(map(abs, gaps)),
        "stochastic_valuation_cases": len(stochastic),
        "deterministic_zero_width_valuation_cases": len(deterministic),
        "crr_reference_inside_stochastic_valuation_interval_cases": sum(
            bool(row["crr_reference_inside_lsm_valuation_interval"])
            for row in stochastic
        ),
        "deterministic_maximum_absolute_crr_difference": max(
            deterministic_gaps, default=None
        ),
        "mean_standard_error": (
            sum(float(row["standard_error"]) for row in lsm_rows)
            / len(lsm_rows)
        ),
        "mean_stochastic_standard_error": (
            sum(float(row["standard_error"]) for row in stochastic)
            / len(stochastic)
            if stochastic
            else None
        ),
        "variance_reduction_applicable_cases": len(applicable),
        "variance_reduction_infinite_cases": sum(
            bool(row["variance_reduction_is_infinite"]) for row in applicable
        ),
        "minimum_variance_reduction_ratio": min(
            finite_variance_reductions, default=None
        ),
        "total_constant_fallback_dates": sum(
            int(row["regression_summary"]["constant_fallback_dates"])
            for row in lsm_rows
        ),
        "notes": _SUMMARY_NOTES,
    }


def _reject_unknown(
    table: Mapping[str, Any], allowed: frozenset[str], where: str
) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise LsmCrosscheckError(
            f"{where} contains unknown keys: {', '.join(unknown)}"
        )


def _require_table(
    table: Mapping[str, Any], key: str, where: str
) -> Mapping[str, Any]:
    value = table.get(key)
    if not isinstance(value, dict):
        raise LsmCrosscheckError(f"{where}.{key} must be a table")
    return value


def _require_string(table: Mapping[str, Any], key: str, where: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise LsmCrosscheckError(
            f"{where}.{key} must be a non-empty string"
        )
    return value


def _require_sha256(table: Mapping[str, Any], key: str, where: str) -> str:
    value = _require_string(table, key, where)
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise LsmCrosscheckError(
            f"{where}.{key} must be a lowercase SHA-256 digest"
        )
    return value


def _require_int(table: Mapping[str, Any], key: str, where: str) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LsmCrosscheckError(
            f"{where}.{key} must be a positive integer"
        )
    return value


def _require_even_paths(
    table: Mapping[str, Any], key: str, where: str
) -> int:
    value = _require_int(table, key, where)
    if value < 4 or value % 2:
        raise LsmCrosscheckError(
            f"{where}.{key} must be even and at least four"
        )
    if value > int(_core.maximum_lsm_paths):
        raise LsmCrosscheckError(f"{where}.{key} exceeds the engine limit")
    return value


def _require_string_array(
    table: Mapping[str, Any], key: str, where: str
) -> tuple[str, ...]:
    value = table.get(key)
    if not isinstance(value, list) or not value:
        raise LsmCrosscheckError(
            f"{where}.{key} must be a non-empty string array"
        )
    if any(not isinstance(item, str) or not item for item in value):
        raise LsmCrosscheckError(
            f"{where}.{key} must contain non-empty strings"
        )
    if len(set(value)) != len(value):
        raise LsmCrosscheckError(
            f"{where}.{key} must not contain duplicates"
        )
    return tuple(value)


def _find_project_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    for candidate in resolved.parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return PROJECT_ROOT


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the cross-check and return a stable CLI status."""
    try:
        arguments = build_parser().parse_args(argv)
        if arguments.output.exists():
            raise LsmCrosscheckError(
                f"refusing to overwrite existing output '{arguments.output}'"
            )
        config = load_lsm_crosscheck_config(arguments.config)
        report = run_lsm_crosscheck(config)
        write_report(report, arguments.output)
    except (LsmCrosscheckError, OverflowError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "cases": len(config.cases),
                "experiments": len(config.experiments),
                "output": str(arguments.output),
                "primary": next(
                    experiment.name
                    for experiment in config.experiments
                    if experiment.role == "primary"
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
