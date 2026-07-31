"""Deterministic CRR convergence and feasibility diagnostics.

The report compares lower-step adjacent averages with a declared high-step
adjacent average from the same CRR implementation. That comparison measures
internal refinement stability, not independent truth. Analytic European
Black--Scholes controls are included to catch implementation and discretization
errors where closed-form truth is available.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
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

SCHEMA_VERSION: Final = "american-crr-convergence/1"
REPORT_SCHEMA_VERSION: Final = "american-crr-convergence-report/1"
ENGINE_NAME: Final = "dp::crr_binomial/v1"
OPTION_TYPES: Final = frozenset({"call", "put"})
EXERCISE_STYLE: Final = "american"
EXERCISE_EXPECTATIONS: Final = frozenset({"material", "none", "unconstrained"})
BOUNDARY_SAMPLE_TIME_FRACTIONS: Final = (0.25, 0.50, 0.75, 0.90)

_TOP_LEVEL_KEYS: Final = frozenset({"schema_version", "study", "cases", "feasibility"})
_STUDY_KEYS: Final = frozenset(
    {
        "name",
        "engine",
        "exercise_style",
        "step_ladder",
        "reference_steps",
        "diagnostic_step_ladder",
        "batch_threads",
    }
)
_CASE_KEYS: Final = frozenset(
    {
        "name",
        "option_type",
        "spot",
        "strike",
        "maturity",
        "rate",
        "dividend_yield",
        "volatility",
        "exercise_expectation",
    }
)
_FEASIBILITY_KEYS: Final = frozenset(
    {
        "maturities",
        "rates",
        "dividend_yields",
        "volatilities",
        "step_candidates",
    }
)


class ConvergenceError(RuntimeError):
    """Raised when the convergence contract or execution is invalid."""


@dataclass(frozen=True, slots=True)
class ConvergenceCase:
    """One named American-option state in the refinement study."""

    name: str
    option_type: str
    spot: float
    strike: float
    maturity: float
    rate: float
    dividend_yield: float
    volatility: float
    exercise_expectation: str


@dataclass(frozen=True, slots=True)
class FeasibilityGrid:
    """Axes used to map the strict CRR probability condition."""

    maturities: tuple[float, ...]
    rates: tuple[float, ...]
    dividend_yields: tuple[float, ...]
    volatilities: tuple[float, ...]
    step_candidates: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ConvergenceConfig:
    """Fully validated convergence study and source provenance."""

    name: str
    engine: str
    exercise_style: str
    step_ladder: tuple[int, ...]
    reference_steps: int
    diagnostic_step_ladder: tuple[int, ...]
    batch_threads: int
    cases: tuple[ConvergenceCase, ...]
    feasibility: FeasibilityGrid
    source_name: str
    source_sha256: str


@dataclass(frozen=True, slots=True)
class _PriceKey:
    case_index: int
    exercise_style: str
    steps: int


def load_convergence_config(path: Path | str) -> ConvergenceConfig:
    """Load and strictly validate a versioned convergence configuration."""
    config_path = Path(path)
    try:
        raw = config_path.read_bytes()
    except OSError as error:
        raise ConvergenceError(f"cannot read configuration '{config_path}': {error}") from error
    try:
        document = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ConvergenceError(
            f"configuration '{config_path}' is not valid UTF-8 TOML: {error}"
        ) from error
    return parse_convergence_config(
        document,
        source_name=config_path.name,
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


def parse_convergence_config(
    document: Mapping[str, Any],
    *,
    source_name: str,
    source_sha256: str,
) -> ConvergenceConfig:
    """Validate an already-parsed convergence configuration."""
    _reject_unknown(document, _TOP_LEVEL_KEYS, "top-level")
    schema_version = _require_string(document, "schema_version", "top-level")
    if schema_version != SCHEMA_VERSION:
        raise ConvergenceError(
            f"unsupported schema_version '{schema_version}'; expected '{SCHEMA_VERSION}'"
        )

    study = _require_table(document, "study", "top-level")
    _reject_unknown(study, _STUDY_KEYS, "[study]")
    name = _require_string(study, "name", "[study]")
    engine = _require_string(study, "engine", "[study]")
    if engine != ENGINE_NAME:
        raise ConvergenceError(f"[study].engine must be '{ENGINE_NAME}', got '{engine}'")
    exercise_style = _require_string(study, "exercise_style", "[study]")
    if exercise_style != EXERCISE_STYLE:
        raise ConvergenceError(
            f"[study].exercise_style must be '{EXERCISE_STYLE}', got '{exercise_style}'"
        )
    step_ladder = _require_int_array(study, "step_ladder", "[study]")
    _require_strictly_increasing(step_ladder, "[study].step_ladder")
    reference_steps = _require_int(study, "reference_steps", "[study]")
    if reference_steps <= step_ladder[-1]:
        raise ConvergenceError("[study].reference_steps must exceed every step_ladder entry")
    if reference_steps >= int(_core.maximum_crr_steps):
        raise ConvergenceError(
            "[study].reference_steps must be below maximum_crr_steps because the "
            "adjacent reference also prices reference_steps + 1"
        )
    diagnostic_step_ladder = _require_int_array(
        study, "diagnostic_step_ladder", "[study]"
    )
    _require_strictly_increasing(
        diagnostic_step_ladder, "[study].diagnostic_step_ladder"
    )
    if not set(diagnostic_step_ladder).issubset(step_ladder):
        raise ConvergenceError(
            "[study].diagnostic_step_ladder must be a subset of step_ladder"
        )
    batch_threads = _require_int(study, "batch_threads", "[study]")
    if not 1 <= batch_threads <= int(_core.maximum_crr_batch_threads):
        raise ConvergenceError(
            f"[study].batch_threads must lie in [1, {_core.maximum_crr_batch_threads}]"
        )

    raw_cases = document.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ConvergenceError("top-level cases must be a non-empty array of tables")
    cases = tuple(_parse_case(value, index) for index, value in enumerate(raw_cases))
    names = [case.name for case in cases]
    if len(set(names)) != len(names):
        raise ConvergenceError("case names must be unique")

    feasibility_table = _require_table(document, "feasibility", "top-level")
    _reject_unknown(feasibility_table, _FEASIBILITY_KEYS, "[feasibility]")
    feasibility = FeasibilityGrid(
        maturities=_require_float_array(
            feasibility_table, "maturities", "[feasibility]", positive=True
        ),
        rates=_require_float_array(feasibility_table, "rates", "[feasibility]"),
        dividend_yields=_require_float_array(
            feasibility_table, "dividend_yields", "[feasibility]"
        ),
        volatilities=_require_float_array(
            feasibility_table, "volatilities", "[feasibility]", positive=True
        ),
        step_candidates=_require_int_array(
            feasibility_table, "step_candidates", "[feasibility]"
        ),
    )
    _require_strictly_increasing(
        feasibility.step_candidates, "[feasibility].step_candidates"
    )
    if feasibility.step_candidates[-1] > int(_core.maximum_crr_steps):
        raise ConvergenceError("[feasibility].step_candidates exceed maximum_crr_steps")

    return ConvergenceConfig(
        name=name,
        engine=engine,
        exercise_style=exercise_style,
        step_ladder=step_ladder,
        reference_steps=reference_steps,
        diagnostic_step_ladder=diagnostic_step_ladder,
        batch_threads=batch_threads,
        cases=cases,
        feasibility=feasibility,
        source_name=source_name,
        source_sha256=source_sha256,
    )


def run_convergence(config: ConvergenceConfig) -> dict[str, Any]:
    """Run the configured study through the compiled C++ pricing boundary."""
    keys = _price_keys(config)
    requests = [_request_columns(config.cases[key.case_index], key) for key in keys]
    columns = {
        name: [request[name] for request in requests]
        for name in (
            "option_type",
            "exercise_style",
            "spot",
            "strike",
            "maturity",
            "rate",
            "dividend_yield",
            "volatility",
            "steps",
        )
    }
    try:
        priced = _core.crr_price_batch(
            columns["option_type"],
            columns["exercise_style"],
            columns["spot"],
            columns["strike"],
            columns["maturity"],
            columns["rate"],
            columns["dividend_yield"],
            columns["volatility"],
            columns["steps"],
            config.batch_threads,
        )
    except (RuntimeError, ValueError, OverflowError) as error:
        raise ConvergenceError(f"C++ CRR batch failed: {error}") from error

    prices = [float(value) for value in priced["price"]]
    probabilities = [float(value) for value in priced["risk_neutral_probability"]]
    if len(prices) != len(keys) or len(probabilities) != len(keys):
        raise ConvergenceError("C++ CRR batch returned a different row count")
    lookup = {
        key: {"price": prices[index], "risk_neutral_probability": probabilities[index]}
        for index, key in enumerate(keys)
    }

    case_reports = [
        _case_report(config, case_index, case, lookup)
        for case_index, case in enumerate(config.cases)
    ]
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "study": {
            "name": config.name,
            "engine": config.engine,
            "exercise_style": config.exercise_style,
            "step_ladder": list(config.step_ladder),
            "reference_steps": config.reference_steps,
            "diagnostic_step_ladder": list(config.diagnostic_step_ladder),
            "batch_threads": config.batch_threads,
        },
        "source": {
            "config_file": config.source_name,
            "config_sha256": config.source_sha256,
            "oracle_version": str(_core.__version__),
            "build_configuration": str(_core.__build_configuration__),
            "cxx_compiler": str(_core.__cxx_compiler__),
            "crr_header_sha256": str(_core.__crr_header_sha256__),
            "crr_implementation_sha256": str(
                _core.__crr_implementation_sha256__
            ),
            "crr_source_sha256": str(_core.__crr_source_sha256__),
            "execution_api": "differentiable_pricing._core.crr_price_batch",
        },
        "interpretation": {
            "reference": (
                "The high-step adjacent CRR average is an internal refinement reference, "
                "not independent truth or a certified error bound."
            ),
            "adjacent_refinement_discrepancy": (
                "The absolute N versus N+1 discrepancy combines terminal-lattice "
                "alignment, time discretization, and exercise-frontier movement; it "
                "is neither a pure parity diagnostic nor an error bar."
            ),
            "timing": (
                "No timings are stored in this deterministic numerical report. Use the "
                "separate benchmark command with machine metadata."
            ),
        },
        "cases": case_reports,
        "summary_by_steps": _summarize_steps(config, case_reports),
        "control_summary": _control_summary(case_reports),
        "feasibility": _feasibility_report(config.feasibility),
    }


def write_report(report: Mapping[str, Any], output: Path | str) -> None:
    """Atomically create ``output`` without overwriting prior evidence."""
    output_path = Path(output)
    if output_path.exists():
        raise ConvergenceError(f"refusing to overwrite existing report '{output_path}'")
    descriptor: int | None = None
    temporary: Path | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        descriptor, temporary_name = tempfile.mkstemp(
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = None
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, output_path)
        finally:
            if descriptor is not None:
                os.close(descriptor)
            temporary.unlink(missing_ok=True)
    except OSError as error:
        raise ConvergenceError(f"cannot write report '{output_path}': {error}") from error


def _price_keys(config: ConvergenceConfig) -> list[_PriceKey]:
    keys = []
    for case_index in range(len(config.cases)):
        for steps in (*config.step_ladder, config.reference_steps):
            keys.extend(
                (
                    _PriceKey(case_index, "american", steps),
                    _PriceKey(case_index, "american", steps + 1),
                )
            )
        keys.extend(
            (
                _PriceKey(case_index, "european", config.reference_steps),
                _PriceKey(case_index, "european", config.reference_steps + 1),
            )
        )
    return keys


def _request_columns(case: ConvergenceCase, key: _PriceKey) -> dict[str, Any]:
    return {
        "option_type": case.option_type,
        "exercise_style": key.exercise_style,
        "spot": case.spot,
        "strike": case.strike,
        "maturity": case.maturity,
        "rate": case.rate,
        "dividend_yield": case.dividend_yield,
        "volatility": case.volatility,
        "steps": key.steps,
    }


def _exercise_diagnostic_report(
    config: ConvergenceConfig,
    case_index: int,
    case: ConvergenceCase,
    steps: int,
    lookup: Mapping[_PriceKey, Mapping[str, float]],
) -> dict[str, Any]:
    try:
        diagnostics = _core.crr_diagnostics(
            case.option_type,
            config.exercise_style,
            case.spot,
            case.strike,
            case.maturity,
            case.rate,
            case.dividend_yield,
            case.volatility,
            steps,
        )
    except (RuntimeError, ValueError, OverflowError) as error:
        raise ConvergenceError(
            f"case '{case.name}' exercise diagnostics failed at {steps} steps: {error}"
        ) from error
    price = float(diagnostics["price"])
    expected_price = float(
        lookup[_PriceKey(case_index, "american", steps)]["price"]
    )
    if price != expected_price:
        raise ConvergenceError(
            f"case '{case.name}' diagnostic and price-only engines disagree "
            f"at {steps} steps"
        )

    boundaries = list(diagnostics["exercise_boundary_by_step"])
    if len(boundaries) != steps:
        raise ConvergenceError(
            f"case '{case.name}' returned an invalid exercise-boundary length"
        )
    boundary_points = [
        (step, float(value))
        for step, value in enumerate(boundaries)
        if value is not None
    ]
    samples = []
    for requested_fraction in BOUNDARY_SAMPLE_TIME_FRACTIONS:
        step = min(steps - 1, round(requested_fraction * steps))
        value = boundaries[step]
        samples.append(
            {
                "requested_time_fraction": requested_fraction,
                "step": step,
                "time_fraction": step / steps,
                "spot": None if value is None else float(value),
            }
        )
    early_exercise_nodes = int(diagnostics["early_exercise_nodes"])
    possible_pre_expiry_nodes = steps * (steps + 1) // 2
    earliest_step = diagnostics["earliest_exercise_step"]
    return {
        "steps": steps,
        "price": price,
        "early_exercise_nodes": early_exercise_nodes,
        "exercise_activity_fraction": (
            early_exercise_nodes / possible_pre_expiry_nodes
        ),
        "earliest_exercise_step": earliest_step,
        "earliest_exercise_time_fraction": (
            None if earliest_step is None else int(earliest_step) / steps
        ),
        "boundary_point_count": len(boundary_points),
        "first_boundary_point": (
            {
                "step": boundary_points[0][0],
                "time_fraction": boundary_points[0][0] / steps,
                "spot": boundary_points[0][1],
            }
            if boundary_points
            else None
        ),
        "last_boundary_point": (
            {
                "step": boundary_points[-1][0],
                "time_fraction": boundary_points[-1][0] / steps,
                "spot": boundary_points[-1][1],
            }
            if boundary_points
            else None
        ),
        "boundary_samples": samples,
    }


def _case_report(
    config: ConvergenceConfig,
    case_index: int,
    case: ConvergenceCase,
    lookup: Mapping[_PriceKey, Mapping[str, float]],
) -> dict[str, Any]:
    reference = _pair(lookup, case_index, "american", config.reference_steps)
    european_reference = _pair(lookup, case_index, "european", config.reference_steps)
    analytic = _core.black_scholes(
        case.option_type,
        case.spot,
        case.strike,
        case.maturity,
        case.rate,
        case.dividend_yield,
        case.volatility,
    )
    analytic_price = float(analytic["price"])
    diagnostic_reports = [
        _exercise_diagnostic_report(
            config,
            case_index,
            case,
            steps,
            lookup,
        )
        for steps in config.diagnostic_step_ladder
    ]
    exercise_premium = (
        reference["adjacent_average"] - european_reference["adjacent_average"]
    )
    comparison_scale = max(
        1.0,
        abs(reference["adjacent_average"]),
        abs(european_reference["adjacent_average"]),
    )
    if exercise_premium < -1.0e-12 * comparison_scale:
        raise ConvergenceError(
            f"case '{case.name}' produced an American price below its European tree"
        )
    if (
        case.option_type == "call"
        and case.dividend_yield == 0.0
        and case.rate >= 0.0
        and abs(exercise_premium) > 1.0e-12 * comparison_scale
    ):
        raise ConvergenceError(
            f"case '{case.name}' violated the no-dividend American-call control"
        )
    material_threshold = 1.0e-10 * comparison_scale
    final_diagnostics = diagnostic_reports[-1]
    if case.exercise_expectation == "none" and (
        int(final_diagnostics["early_exercise_nodes"]) != 0
        or exercise_premium > material_threshold
    ):
        raise ConvergenceError(
            f"case '{case.name}' violated its no-material-exercise expectation"
        )
    if case.exercise_expectation == "material" and (
        int(final_diagnostics["early_exercise_nodes"]) == 0
        or exercise_premium <= material_threshold
    ):
        raise ConvergenceError(
            f"case '{case.name}' violated its material-exercise expectation"
        )
    levels = []
    for steps in config.step_ladder:
        pair = _pair(lookup, case_index, "american", steps)
        pair["absolute_difference_to_reference"] = abs(
            pair["adjacent_average"] - reference["adjacent_average"]
        )
        levels.append(pair)
    return {
        "name": case.name,
        "option_type": case.option_type,
        "exercise_expectation": case.exercise_expectation,
        "input": {
            key: value
            for key, value in asdict(case).items()
            if key not in {"name", "option_type", "exercise_expectation"}
        },
        "reference": reference,
        "levels": levels,
        "exercise_diagnostics": diagnostic_reports,
        "european_control": {
            "black_scholes_price": analytic_price,
            "crr_reference": european_reference,
            "absolute_error": abs(
                european_reference["adjacent_average"] - analytic_price
            ),
            "american_minus_black_scholes": (
                reference["adjacent_average"] - analytic_price
            ),
            "american_exercise_premium_over_european_crr": exercise_premium,
        },
    }


def _pair(
    lookup: Mapping[_PriceKey, Mapping[str, float]],
    case_index: int,
    exercise_style: str,
    steps: int,
) -> dict[str, Any]:
    left = lookup[_PriceKey(case_index, exercise_style, steps)]
    right = lookup[_PriceKey(case_index, exercise_style, steps + 1)]
    price = float(left["price"])
    adjacent_price = float(right["price"])
    return {
        "steps": steps,
        "price": price,
        "adjacent_step_price": adjacent_price,
        "adjacent_average": 0.5 * (price + adjacent_price),
        "absolute_pair_gap": abs(price - adjacent_price),
        "risk_neutral_probability": float(left["risk_neutral_probability"]),
        "adjacent_risk_neutral_probability": float(
            right["risk_neutral_probability"]
        ),
    }


def _summarize_steps(
    config: ConvergenceConfig,
    case_reports: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    summaries = []
    for index, steps in enumerate(config.step_ladder):
        reference_errors = [
            float(case["levels"][index]["absolute_difference_to_reference"])
            for case in case_reports
        ]
        pair_gaps = [
            float(case["levels"][index]["absolute_pair_gap"]) for case in case_reports
        ]
        summaries.append(
            {
                "steps": steps,
                "cases": len(case_reports),
                "mean_absolute_difference_to_reference": math.fsum(reference_errors)
                / len(reference_errors),
                "p95_absolute_difference_to_reference": _quantile(
                    reference_errors, 0.95
                ),
                "maximum_absolute_difference_to_reference": max(reference_errors),
                "maximum_absolute_pair_gap": max(pair_gaps),
            }
        )
    return summaries


def _control_summary(case_reports: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    european_errors = [
        float(case["european_control"]["absolute_error"]) for case in case_reports
    ]
    theorem_control_calls = [
        case
        for case in case_reports
        if case["option_type"] == "call"
        and float(case["input"]["dividend_yield"]) == 0.0
        and float(case["input"]["rate"]) >= 0.0
    ]
    negative_rate_no_dividend_calls = [
        case
        for case in case_reports
        if case["option_type"] == "call"
        and float(case["input"]["dividend_yield"]) == 0.0
        and float(case["input"]["rate"]) < 0.0
    ]
    return {
        "maximum_european_crr_absolute_error_vs_black_scholes": max(
            european_errors
        ),
        "nonnegative_rate_no_dividend_call_control_cases": len(
            theorem_control_calls
        ),
        "maximum_nonnegative_rate_no_dividend_american_call_absolute_error_vs_black_scholes": max(
            (
                abs(
                    float(
                        case["european_control"][
                            "american_minus_black_scholes"
                        ]
                    )
                )
                for case in theorem_control_calls
            ),
            default=None,
        ),
        "negative_rate_no_dividend_call_cases": len(
            negative_rate_no_dividend_calls
        ),
        "minimum_negative_rate_no_dividend_call_exercise_premium": min(
            (
                float(
                    case["european_control"][
                        "american_exercise_premium_over_european_crr"
                    ]
                )
                for case in negative_rate_no_dividend_calls
            ),
            default=None,
        ),
    }


def _feasibility_report(grid: FeasibilityGrid) -> dict[str, Any]:
    states = list(
        itertools.product(
            grid.maturities,
            grid.rates,
            grid.dividend_yields,
            grid.volatilities,
        )
    )
    histogram: dict[str, int] = {}
    unsupported: list[dict[str, float]] = []
    for maturity, rate, dividend_yield, volatility in states:
        minimum_steps: int | None = None
        for steps in grid.step_candidates:
            try:
                _core.crr_lattice_parameters(
                    100.0,
                    100.0,
                    maturity,
                    rate,
                    dividend_yield,
                    volatility,
                    steps,
                )
            except (OverflowError, ValueError):
                continue
            minimum_steps = steps
            break
        if minimum_steps is None:
            unsupported.append(
                {
                    "maturity": maturity,
                    "rate": rate,
                    "dividend_yield": dividend_yield,
                    "volatility": volatility,
                }
            )
        else:
            key = str(minimum_steps)
            histogram[key] = histogram.get(key, 0) + 1
    return {
        "probability_condition": "strict 0 < p < 1; no clipping",
        "axes": {
            "maturities": list(grid.maturities),
            "rates": list(grid.rates),
            "dividend_yields": list(grid.dividend_yields),
            "volatilities": list(grid.volatilities),
            "step_candidates": list(grid.step_candidates),
        },
        "total_states": len(states),
        "feasible_states": len(states) - len(unsupported),
        "unsupported_states": unsupported,
        "minimum_feasible_candidate_steps_histogram": dict(
            sorted(histogram.items(), key=lambda item: int(item[0]))
        ),
    }


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _parse_case(value: Any, index: int) -> ConvergenceCase:
    where = f"[[cases]] at index {index}"
    if not isinstance(value, dict):
        raise ConvergenceError(f"{where} must be a table")
    _reject_unknown(value, _CASE_KEYS, where)
    option_type = _require_string(value, "option_type", where)
    if option_type not in OPTION_TYPES:
        raise ConvergenceError(f"{where}.option_type must be 'call' or 'put'")
    exercise_expectation = value.get("exercise_expectation", "unconstrained")
    if (
        not isinstance(exercise_expectation, str)
        or exercise_expectation not in EXERCISE_EXPECTATIONS
    ):
        raise ConvergenceError(
            f"{where}.exercise_expectation must be 'material', 'none', or "
            "'unconstrained'"
        )
    return ConvergenceCase(
        name=_require_string(value, "name", where),
        option_type=option_type,
        spot=_require_number(value, "spot", where, positive=True),
        strike=_require_number(value, "strike", where, positive=True),
        maturity=_require_number(value, "maturity", where, positive=True),
        rate=_require_number(value, "rate", where),
        dividend_yield=_require_number(value, "dividend_yield", where),
        volatility=_require_number(value, "volatility", where, positive=True),
        exercise_expectation=str(exercise_expectation),
    )


def _reject_unknown(table: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ConvergenceError(f"{where} contains unknown keys: {', '.join(unknown)}")


def _require_table(table: Mapping[str, Any], key: str, where: str) -> Mapping[str, Any]:
    value = table.get(key)
    if not isinstance(value, dict):
        raise ConvergenceError(f"{where}.{key} must be a table")
    return value


def _require_string(table: Mapping[str, Any], key: str, where: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise ConvergenceError(f"{where}.{key} must be a non-empty string")
    return value


def _require_int(table: Mapping[str, Any], key: str, where: str) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConvergenceError(f"{where}.{key} must be a positive integer")
    return value


def _require_number(
    table: Mapping[str, Any],
    key: str,
    where: str,
    *,
    positive: bool = False,
) -> float:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConvergenceError(f"{where}.{key} must be a number")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ConvergenceError(f"{where}.{key} must be {qualifier}")
    return number


def _require_int_array(
    table: Mapping[str, Any], key: str, where: str
) -> tuple[int, ...]:
    value = table.get(key)
    if not isinstance(value, list) or not value:
        raise ConvergenceError(f"{where}.{key} must be a non-empty integer array")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise ConvergenceError(
                f"{where}.{key}[{index}] must be a positive integer"
            )
        result.append(item)
    return tuple(result)


def _require_float_array(
    table: Mapping[str, Any],
    key: str,
    where: str,
    *,
    positive: bool = False,
) -> tuple[float, ...]:
    value = table.get(key)
    if not isinstance(value, list) or not value:
        raise ConvergenceError(f"{where}.{key} must be a non-empty numeric array")
    result = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ConvergenceError(f"{where}.{key}[{index}] must be a number")
        number = float(item)
        if not math.isfinite(number) or (positive and number <= 0.0):
            qualifier = "finite and positive" if positive else "finite"
            raise ConvergenceError(
                f"{where}.{key}[{index}] must be {qualifier}"
            )
        result.append(number)
    if len(set(result)) != len(result):
        raise ConvergenceError(f"{where}.{key} must not contain duplicates")
    return tuple(result)


def _require_strictly_increasing(values: Sequence[int], where: str) -> None:
    if any(left >= right for left, right in itertools.pairwise(values)):
        raise ConvergenceError(f"{where} must be strictly increasing")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the convergence study and return a stable CLI status."""
    try:
        arguments = build_parser().parse_args(argv)
        if arguments.output.exists():
            raise ConvergenceError(
                f"refusing to overwrite existing report '{arguments.output}'"
            )
        config = load_convergence_config(arguments.config)
        report = run_convergence(config)
        write_report(report, arguments.output)
    except ConvergenceError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "cases": len(report["cases"]),
                "feasibility_states": report["feasibility"]["total_states"],
                "output": str(arguments.output),
                "reference_steps": config.reference_steps,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
