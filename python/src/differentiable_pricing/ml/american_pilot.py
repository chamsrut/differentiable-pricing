"""Locked Task 9G experiment operations (manual invocation only).

Tests replace the expensive pricing/training operations with tiny fixtures or
mocks.  No test or repository check calls :func:`execute_validation_run`.
"""

from __future__ import annotations

import hashlib
import math
import os
import platform
import statistics
import time
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pyarrow.parquet as pq
import torch

from differentiable_pricing import _core, pde_price

from ..data.american_admission import AdmissionGateError, check_label_policy
from ..data.american_schema import TABLE_SCHEMA
from .american import (
    AmericanPriceModel,
    ExactLiftError,
    fit_price_model,
    lift_european_network,
    scaling_from_arrays,
    select_rows_by_hash,
    verify_exact_lift,
)
from .american_artifact import save_american_artifact
from .artifact import load_physical_model, write_json_atomic
from .config import FEATURE_ORDER

SELECTION_COLUMNS: Final = (
    "sample_id",
    "stratum",
    "option_type",
    "spot",
    "strike",
    "maturity",
    "rate",
    "dividend_yield",
    "volatility",
)
EVALUATION_COLUMNS: Final = (
    *SELECTION_COLUMNS,
    "american_price",
    "european_crr_price",
    "early_exercise_premium",
    "intrinsic_value",
    "earliest_exercise_step",
)
POLICY_COLUMNS: Final = ("label_policy", "label_steps")


class PilotError(RuntimeError):
    """Raised when the locked pilot cannot continue safely."""


def load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            payload = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise PilotError(f"cannot load locked TOML '{path}': {error}") from error
    if not isinstance(payload, dict):
        raise PilotError(f"locked TOML '{path}' must contain a table")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_entry(manifest: Mapping[str, Any], split: str) -> Mapping[str, Any]:
    entries = [entry for entry in manifest["files"] if entry["split"] == split]
    if len(entries) != 1:
        raise PilotError(f"manifest must declare exactly one {split!r} file")
    return entries[0]


def read_partition_columns(
    dataset: Path,
    manifest: Mapping[str, Any],
    split: str,
    columns: Sequence[str],
    *,
    verify_digest: bool = True,
) -> dict[str, np.ndarray]:
    """Open one explicitly named partition and no other partition."""
    entry = _manifest_entry(manifest, split)
    path = dataset / str(entry["file"])
    if verify_digest:
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            raise PilotError(f"{split} SHA-256 mismatch")
    file_schema = pq.read_schema(path)
    if not file_schema.equals(TABLE_SCHEMA, check_metadata=False):
        raise PilotError(f"{split} schema differs from american-option-dataset/1")
    table = pq.read_table(path, columns=list(columns))
    if table.num_rows != entry["rows"]:
        raise PilotError(f"{split} row count differs from manifest")
    arrays: dict[str, np.ndarray] = {}
    for name in columns:
        if name in {"sample_id", "stratum", "option_type", "label_policy"}:
            arrays[name] = np.asarray(table[name].to_pylist(), dtype=object)
        else:
            arrays[name] = np.asarray(table[name])
            if np.issubdtype(arrays[name].dtype, np.floating) and not bool(
                np.isfinite(arrays[name]).all()
            ):
                raise PilotError(f"{split}.{name} contains non-finite values")
    return arrays


def verify_partition_policy(dataset: Path, manifest: Mapping[str, Any], split: str) -> None:
    entry = _manifest_entry(manifest, split)
    path = dataset / str(entry["file"])
    table = pq.read_table(path, columns=list(POLICY_COLUMNS))
    try:
        check_label_policy(table, manifest, where=f"split '{split}'")
    except AdmissionGateError as error:
        raise PilotError(str(error)) from error


def physical_features(columns: Mapping[str, np.ndarray]) -> np.ndarray:
    encoding = np.where(columns["option_type"] == "call", 1.0, -1.0)
    unknown = ~np.isin(columns["option_type"], ("call", "put"))
    if bool(unknown.any()):
        raise PilotError("partition contains unknown option_type")
    values = {
        "option_type": encoding,
        **{name: np.asarray(columns[name], dtype=np.float64) for name in FEATURE_ORDER[1:]},
    }
    return np.ascontiguousarray(np.column_stack([values[name] for name in FEATURE_ORDER]))


def predict_prices(model: AmericanPriceModel, physical: np.ndarray, batch_size: int) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, physical.shape[0], batch_size):
            tensor = torch.as_tensor(physical[start : start + batch_size], dtype=torch.float64)
            outputs.append(model(tensor).cpu().numpy())
    result = np.concatenate(outputs)
    if not bool(np.isfinite(result).all()):
        raise PilotError("model produced a non-finite price")
    return result


def error_statistics(
    reference: np.ndarray, prediction: np.ndarray, *, quantile_method: str
) -> dict[str, float | int]:
    if reference.shape != prediction.shape or reference.ndim != 1 or reference.size == 0:
        raise PilotError("error statistics require non-empty matching vectors")
    error = np.abs(prediction - reference)
    return {
        "rows": int(error.size),
        "mae": float(error.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "p95_absolute_error": float(np.quantile(error, 0.95, method=quantile_method)),
        "p99_absolute_error": float(np.quantile(error, 0.99, method=quantile_method)),
        "maximum_absolute_error": float(error.max()),
    }


def _slice_masks(
    columns: Mapping[str, np.ndarray], acceptance: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    bins = acceptance["bins"]
    maturity = np.asarray(columns["maturity"], dtype=np.float64)
    log_moneyness = np.log(
        np.asarray(columns["spot"], dtype=np.float64)
        / np.asarray(columns["strike"], dtype=np.float64)
    )
    volatility = np.asarray(columns["volatility"], dtype=np.float64)
    premium = np.asarray(columns["early_exercise_premium"], dtype=np.float64)
    earliest = np.asarray(columns["earliest_exercise_step"], dtype=np.int64)
    masks: dict[str, np.ndarray] = {"overall": np.ones(maturity.shape, dtype=bool)}
    for name in ("call", "put"):
        masks[f"option_type:{name}"] = columns["option_type"] == name
    for label, low, high in _intervals(bins["expiry_edges_years"]):
        masks[f"expiry:{label}"] = (maturity >= low) & (maturity < high)
    for label, low, high in _intervals(bins["log_moneyness_edges"]):
        masks[f"moneyness:{label}"] = (log_moneyness >= low) & (log_moneyness < high)
    for label, low, high in _intervals(bins["volatility_edges"]):
        masks[f"volatility:{label}"] = (volatility >= low) & (volatility < high)
    masks["premium_status:zero"] = premium == 0.0
    masks["premium_status:positive"] = premium > 0.0
    masks["exercise_status:no_exercise"] = earliest < 0
    masks["exercise_status:exercise_observed"] = earliest >= 0
    return masks


def _intervals(edges: Sequence[float]) -> list[tuple[str, float, float]]:
    values = [-math.inf, *(float(value) for value in edges), math.inf]
    return [
        (f"{values[index]:g}:{values[index + 1]:g}", values[index], values[index + 1])
        for index in range(len(values) - 1)
    ]


def sliced_metrics(
    columns: Mapping[str, np.ndarray],
    prediction: np.ndarray,
    acceptance: Mapping[str, Any],
) -> dict[str, Any]:
    reference = np.asarray(columns["american_price"], dtype=np.float64)
    discounted_spot = np.asarray(columns["spot"], dtype=np.float64) * np.exp(
        -np.asarray(columns["dividend_yield"], dtype=np.float64)
        * np.asarray(columns["maturity"], dtype=np.float64)
    )
    output: dict[str, Any] = {}
    for name, mask in _slice_masks(columns, acceptance).items():
        if not bool(mask.any()):
            output[name] = {"rows": 0, "physical": None, "normalized": None}
            continue
        output[name] = {
            "rows": int(mask.sum()),
            "physical": error_statistics(
                reference[mask],
                prediction[mask],
                quantile_method=acceptance["quantile_method"],
            ),
            "normalized": error_statistics(
                reference[mask] / discounted_spot[mask],
                prediction[mask] / discounted_spot[mask],
                quantile_method=acceptance["quantile_method"],
            ),
        }
    return output


def shape_diagnostics(
    model: AmericanPriceModel,
    columns: Mapping[str, np.ndarray],
    acceptance: Mapping[str, Any],
    *,
    batch_size: int,
    center_prediction: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, list[float | None]]]:
    """Evaluate fixed American bounds and local price-shape properties."""
    physical = physical_features(columns)
    price = (
        predict_prices(model, physical, batch_size)
        if center_prediction is None
        else center_prediction
    )
    index = {name: FEATURE_ORDER.index(name) for name in FEATURE_ORDER}
    spot = physical[:, index["spot"]]
    strike = physical[:, index["strike"]]
    maturity = physical[:, index["maturity"]]
    rate = physical[:, index["rate"]]
    calls = physical[:, index["option_type"]] > 0.0
    intrinsic = np.where(calls, np.maximum(spot - strike, 0.0), np.maximum(strike - spot, 0.0))
    comparator = np.asarray(columns["european_crr_price"], dtype=np.float64)
    upper = np.where(calls, spot, strike * np.exp(np.maximum(-rate * maturity, 0.0)))
    discounted_spot = spot * np.exp(-physical[:, index["dividend_yield"]] * maturity)
    tolerance = float(acceptance["diagnostics"]["material_normalized_tolerance"]) * discounted_spot
    spot_fraction = float(acceptance["diagnostics"]["spot_bump_fraction"])
    volatility_bump = float(acceptance["diagnostics"]["volatility_bump"])
    spot_low, spot_high = (float(value) for value in acceptance["diagnostics"]["spot_domain"])
    volatility_low, volatility_high = (
        float(value) for value in acceptance["diagnostics"]["volatility_domain"]
    )
    moneyness_low, moneyness_high = (
        float(value) for value in acceptance["diagnostics"]["log_moneyness_domain"]
    )
    desired_minus_spot = spot * (1.0 - spot_fraction)
    desired_plus_spot = spot * (1.0 + spot_fraction)
    spot_eligible = (
        (desired_minus_spot >= spot_low)
        & (desired_plus_spot <= spot_high)
        & (np.log(desired_minus_spot / strike) >= moneyness_low)
        & (np.log(desired_plus_spot / strike) <= moneyness_high)
    )
    volatility = physical[:, index["volatility"]]
    volatility_eligible = (volatility - volatility_bump >= volatility_low) & (
        volatility + volatility_bump <= volatility_high
    )
    minus_spot = physical.copy()
    plus_spot = physical.copy()
    minus_spot[:, index["spot"]] = np.maximum(
        desired_minus_spot, np.maximum(spot_low, strike * np.exp(moneyness_low))
    )
    plus_spot[:, index["spot"]] = np.minimum(
        desired_plus_spot, np.minimum(spot_high, strike * np.exp(moneyness_high))
    )
    minus_vol = physical.copy()
    plus_vol = physical.copy()
    minus_vol[:, index["volatility"]] = np.maximum(
        volatility_low,
        minus_vol[:, index["volatility"]] - volatility_bump,
    )
    plus_vol[:, index["volatility"]] = np.minimum(
        plus_vol[:, index["volatility"]] + volatility_bump,
        volatility_high,
    )
    p_minus = predict_prices(model, minus_spot, batch_size)
    p_plus = predict_prices(model, plus_spot, batch_size)
    v_minus = predict_prices(model, minus_vol, batch_size)
    v_plus = predict_prices(model, plus_vol, batch_size)
    spot_monotonicity = np.where(
        calls,
        np.maximum(p_minus - price, price - p_plus),
        np.maximum(price - p_minus, p_plus - price),
    )
    violations = {
        "intrinsic_lower_bound": intrinsic - price,
        "european_comparator_lower_bound": comparator - price,
        "american_call_upper_bound": np.where(calls, price - upper, -np.inf),
        "american_put_rate_aware_upper_bound": np.where(~calls, price - upper, -np.inf),
        "spot_monotonicity": np.where(spot_eligible, spot_monotonicity, -np.inf),
        "spot_convexity": np.where(spot_eligible, 2.0 * price - p_minus - p_plus, -np.inf),
        "volatility_monotonicity": np.where(
            volatility_eligible,
            np.maximum(v_minus - price, price - v_plus),
            -np.inf,
        ),
    }
    details = {
        name: {
            "material_violations": int((value > tolerance).sum()),
            "maximum_violation": float(np.maximum(value, 0.0).max(initial=0.0)),
            "eligible_rows": (
                int(spot_eligible.sum())
                if name in {"spot_monotonicity", "spot_convexity"}
                else (
                    int(volatility_eligible.sum())
                    if name == "volatility_monotonicity"
                    else (
                        int(calls.sum())
                        if name == "american_call_upper_bound"
                        else (
                            int((~calls).sum())
                            if name == "american_put_rate_aware_upper_bound"
                            else int(price.size)
                        )
                    )
                )
            ),
        }
        for name, value in violations.items()
    }
    summary = {
        "checks": details,
        "material_bound_violations": sum(
            details[name]["material_violations"]
            for name in (
                "intrinsic_lower_bound",
                "european_comparator_lower_bound",
                "american_call_upper_bound",
                "american_put_rate_aware_upper_bound",
            )
        ),
        "material_shape_violations": sum(
            details[name]["material_violations"]
            for name in (
                "spot_monotonicity",
                "spot_convexity",
                "volatility_monotonicity",
            )
        ),
        "american_greek_accuracy_claim": False,
    }
    eligible = {
        "intrinsic_lower_bound": np.ones(price.shape, dtype=bool),
        "european_comparator_lower_bound": np.ones(price.shape, dtype=bool),
        "american_call_upper_bound": calls,
        "american_put_rate_aware_upper_bound": ~calls,
        "spot_monotonicity": spot_eligible,
        "spot_convexity": spot_eligible,
        "volatility_monotonicity": volatility_eligible,
    }
    audit = {
        name: [
            float(raw) if is_eligible else None
            for raw, is_eligible in zip(value, eligible[name], strict=True)
        ]
        for name, value in violations.items()
    }
    return summary, audit


def assess_arm(
    metrics: Mapping[str, Any], acceptance: Mapping[str, Any], section: str
) -> dict[str, Any]:
    gates = acceptance[section]
    normalized = metrics["slices"]["overall"]["normalized"]
    checks = {
        "normalized_rmse": normalized["rmse"] <= gates["normalized_rmse_max"],
        "normalized_p99_absolute_error": normalized["p99_absolute_error"]
        <= gates["normalized_p99_absolute_error_max"],
        "normalized_maximum_absolute_error": normalized["maximum_absolute_error"]
        <= gates["normalized_maximum_absolute_error_max"],
        "material_bound_violations": metrics["diagnostics"]["material_bound_violations"]
        <= gates["maximum_material_bound_violations"],
        "material_shape_violations": metrics["diagnostics"]["material_shape_violations"]
        <= gates["maximum_material_shape_violations"],
    }
    return {"checks": checks, "passed": all(checks.values())}


def evaluate_models(
    models: Mapping[str, AmericanPriceModel],
    columns: Mapping[str, np.ndarray],
    acceptance: Mapping[str, Any],
    *,
    batch_size: int,
    gate_section: str,
) -> dict[str, Any]:
    physical = physical_features(columns)
    output: dict[str, Any] = {}
    predictions = {
        name: predict_prices(model, physical, batch_size) for name, model in models.items()
    }
    predictions["european_crr_baseline"] = np.asarray(
        columns["european_crr_price"], dtype=np.float64
    )
    for name, prediction in predictions.items():
        diagnostic_audit = None
        if name in models:
            diagnostic_summary, diagnostic_audit = shape_diagnostics(
                models[name],
                columns,
                acceptance,
                batch_size=batch_size,
                center_prediction=prediction,
            )
        entry = {
            "slices": sliced_metrics(columns, prediction, acceptance),
            "diagnostics": (
                diagnostic_summary
                if name in models
                else {
                    "not_applicable": "stored paired no-learning baseline",
                    "stored_error_equals_early_exercise_premium": bool(
                        np.array_equal(
                            np.asarray(columns["american_price"])
                            - np.asarray(columns["european_crr_price"]),
                            np.asarray(columns["early_exercise_premium"]),
                        )
                    ),
                }
            ),
            "audit_rows": [
                {
                    "sample_id": str(columns["sample_id"][index]),
                    "option_type": str(columns["option_type"][index]),
                    "spot": float(columns["spot"][index]),
                    "strike": float(columns["strike"][index]),
                    "maturity": float(columns["maturity"][index]),
                    "dividend_yield": float(columns["dividend_yield"][index]),
                    "volatility": float(columns["volatility"][index]),
                    "early_exercise_premium": float(columns["early_exercise_premium"][index]),
                    "earliest_exercise_step": int(columns["earliest_exercise_step"][index]),
                    "reference_price": float(columns["american_price"][index]),
                    "prediction": float(prediction[index]),
                    "diagnostic_tolerance": (
                        None
                        if name not in models
                        else float(
                            acceptance["diagnostics"]["material_normalized_tolerance"]
                            * columns["spot"][index]
                            * math.exp(
                                -columns["dividend_yield"][index] * columns["maturity"][index]
                            )
                        )
                    ),
                    "diagnostic_violations": (
                        None
                        if diagnostic_audit is None
                        else {check: values[index] for check, values in diagnostic_audit.items()}
                    ),
                }
                for index in range(prediction.size)
            ],
        }
        if name in models:
            entry["gate"] = assess_arm(entry, acceptance, gate_section)
        output[name] = entry
    return output


def _selected_validation_rows(
    columns: Mapping[str, np.ndarray], salt: str
) -> dict[tuple[str, str], int]:
    selected: dict[tuple[str, str], tuple[bytes, str, int]] = {}
    for index, sample_id in enumerate(columns["sample_id"]):
        key = (str(columns["stratum"][index]), str(columns["option_type"][index]))
        rank = hashlib.sha256(f"{salt}\0{sample_id}".encode()).digest()
        candidate = (rank, str(sample_id), index)
        if key not in selected or candidate[:2] < selected[key][:2]:
            selected[key] = candidate
    return {key: candidate[2] for key, candidate in selected.items()}


def run_independent_pde_check(
    dataset: Path,
    manifest: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the predeclared validation-only PDE mapping check before optimization."""
    specification = protocol["independent_pde_check"]
    inputs = read_partition_columns(
        dataset, manifest, "validation", SELECTION_COLUMNS, verify_digest=False
    )
    selected = _selected_validation_rows(inputs, specification["selection_salt"])
    pinned = specification["cases"]
    if len(pinned) != len(selected):
        raise PilotError("pinned PDE cases do not cover every (stratum, option_type)")
    pinned_ids = {case["sample_id"] for case in pinned}
    actual_ids = {str(inputs["sample_id"][index]) for index in selected.values()}
    if pinned_ids != actual_ids:
        raise PilotError("validation-only PDE selection differs from pinned sample IDs")
    index_by_id = {str(inputs["sample_id"][index]): index for index in selected.values()}
    for case in pinned:
        index = index_by_id[case["sample_id"]]
        for name in ("stratum", "option_type"):
            if str(inputs[name][index]) != case[name]:
                raise PilotError(f"pinned PDE case {case['sample_id']} differs in {name}")
        for name in (
            "spot",
            "strike",
            "maturity",
            "rate",
            "dividend_yield",
            "volatility",
        ):
            if float(inputs[name][index]) != float(case[name]):
                raise PilotError(f"pinned PDE case {case['sample_id']} differs in {name}")
    labels = read_partition_columns(
        dataset,
        manifest,
        "validation",
        ("sample_id", "american_price"),
        verify_digest=False,
    )
    label_by_id = dict(zip(labels["sample_id"], labels["american_price"], strict=True))
    rows = []
    for case in pinned:
        prices = []
        solves = []
        for grid in specification["grids"]:
            requested = {
                "option_type": case["option_type"],
                "exercise_style": "american",
                "spot": case["spot"],
                "strike": case["strike"],
                "valuation_time": specification["valuation_time"],
                "expiry_time": case["maturity"],
                "volatility": case["volatility"],
                "continuous_carry": case["dividend_yield"],
                "curve_times": [0.0, case["maturity"]],
                "curve_log_discounts": [0.0, -case["rate"] * case["maturity"]],
                "dividends": [],
                "settlement": specification["settlement"],
                "contract_multiplier": specification["contract_multiplier"],
                "spot_intervals": grid["spot_intervals"],
                "time_steps": grid["time_steps"],
                "spot_maximum": grid["spot_maximum_factor"] * max(case["spot"], case["strike"]),
                "rannacher_steps": specification["rannacher_steps"],
                "psor_tolerance": specification["psor_tolerance"],
                "psor_relaxation": specification["psor_relaxation"],
                "psor_maximum_iterations": specification["psor_maximum_iterations"],
            }
            result = dict(pde_price(**requested))
            if result.get("solver_status") != "discrete_system_converged":
                raise PilotError(f"PDE solve {case['sample_id']}/{grid['name']} did not converge")
            if result.get("discretization_accuracy") != "not_assessed":
                raise PilotError("PDE engine discretization status differs from its contract")
            for name in ("time_steps", "rannacher_steps", "psor_tolerance", "psor_relaxation"):
                if result.get(name) != requested[name]:
                    raise PilotError(f"PDE effective {name} differs from the requested value")
            actual_intervals = result.get("spot_intervals")
            actual_step = result.get("spot_step")
            actual_maximum = result.get("spot_maximum")
            strike_index = result.get("strike_node_index")
            if (
                not isinstance(actual_intervals, int)
                or not isinstance(strike_index, int)
                or actual_intervals <= 1
                or strike_index <= 0
                or strike_index >= actual_intervals
                or not isinstance(actual_step, float)
                or not isinstance(actual_maximum, float)
                or actual_step <= 0.0
                or actual_maximum <= max(case["spot"], case["strike"])
                or not math.isclose(
                    actual_intervals * actual_step, actual_maximum, rel_tol=1.0e-14, abs_tol=1.0e-14
                )
                or not math.isclose(
                    strike_index * actual_step,
                    case["strike"],
                    rel_tol=1.0e-14,
                    abs_tol=1.0e-14,
                )
            ):
                raise PilotError("PDE effective spatial grid is malformed or not strike-aligned")
            target_step = requested["spot_maximum"] / requested["spot_intervals"]
            expected_strike_index = max(1, math.floor(case["strike"] / target_step + 0.5))
            expected_step = case["strike"] / expected_strike_index
            expected_intervals = max(
                expected_strike_index + 1,
                math.ceil(requested["spot_maximum"] / expected_step - 1.0e-9),
            )
            if (
                strike_index != expected_strike_index
                or actual_intervals != expected_intervals
                or actual_step != expected_step
                or actual_maximum != expected_intervals * expected_step
            ):
                raise PilotError(
                    "PDE effective spatial grid differs from the locked transformation"
                )
            if result.get("dividend_events") != [] or result.get("aligned_times") != [
                0.0,
                case["maturity"],
            ]:
                raise PilotError("PDE effective time grid or dividend schedule differs")
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
                isinstance(result.get(name), bool)
                or not isinstance(result.get(name), int)
                or result[name] < 0
                for name in count_names
            ):
                raise PilotError("PDE effective solve counts are malformed")
            if (
                result["linear_solves"] != 0
                or result["psor_solves"]
                != result["damped_half_steps"] + result["crank_nicolson_steps"]
                or result["psor_total_iterations"] < result["psor_solves"]
                or result["psor_maximum_iterations_used"] > specification["psor_maximum_iterations"]
            ):
                raise PilotError("PDE effective solve accounting differs")
            residuals = (
                result.get("maximum_lcp_residual"),
                result.get("maximum_relative_lcp_residual"),
            )
            if any(
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or float(value) < 0.0
                for value in residuals
            ) or float(result["maximum_relative_lcp_residual"]) > float(
                specification["psor_tolerance"]
            ):
                raise PilotError("PDE effective complementarity residual differs")
            price = result.get("price")
            if (
                isinstance(price, bool)
                or not isinstance(price, int | float)
                or not math.isfinite(float(price))
                or float(price) < 0.0
            ):
                raise PilotError("PDE returned a non-finite or negative price")
            prices.append(float(price))
            solves.append(
                {
                    "grid": grid["name"],
                    "requested_input": requested,
                    "result": result,
                }
            )
        scale = case["spot"] * math.exp(-case["dividend_yield"] * case["maturity"])
        label = float(label_by_id[case["sample_id"]])
        refinement = abs(prices[1] - prices[0]) / scale
        agreement = abs(prices[1] - label) / scale
        rows.append(
            {
                "sample_id": case["sample_id"],
                "coarse_price": prices[0],
                "fine_price": prices[1],
                "crr_label": label,
                "solves": solves,
                "normalized_refinement_difference": refinement,
                "normalized_pde_crr_difference": agreement,
                "passed": refinement <= specification["maximum_normalized_refinement_difference"]
                and agreement <= specification["maximum_normalized_pde_crr_difference"],
            }
        )
    return {
        "rows": rows,
        "passed": all(row["passed"] for row in rows),
        "interpretation": specification["interpretation"],
    }


def crr_adjacent_average(
    case: Mapping[str, Any], steps: int, volatility: float | None = None
) -> float:
    sigma = float(case["volatility"] if volatility is None else volatility)
    arguments = (
        case["option_type"],
        "american",
        float(case["spot"]),
        float(case["strike"]),
        float(case["maturity"]),
        float(case["rate"]),
        float(case["dividend_yield"]),
        sigma,
    )
    return 0.5 * (
        float(_core.crr_price(*arguments, steps)["price"])
        + float(_core.crr_price(*arguments, steps + 1)["price"])
    )


def _runtime_metadata() -> dict[str, Any]:
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "core_version": str(_core.__version__),
        "build_configuration": str(_core.__build_configuration__),
        "cxx_compiler": str(_core.__cxx_compiler__),
    }


def run_latency(
    models: Mapping[str, AmericanPriceModel], specification: Mapping[str, Any]
) -> dict[str, Any]:
    """Measure matched economic requests; CRR always computes N and N+1."""
    if torch.get_num_interop_threads() != int(specification["torch_interop_threads"]):
        raise PilotError("Torch inter-op thread budget differs during latency measurement")
    cases = specification["cases"]
    case_by_name = {case["name"]: case for case in cases}
    measurements = []
    for shape, batch_size, threads in zip(
        specification["request_shapes"],
        specification["batch_sizes"],
        specification["thread_budgets"],
        strict=True,
    ):
        torch.set_num_threads(threads)
        requests = [case_by_name[name] for name in specification["request_case_names"][shape]]
        if len(requests) != batch_size:
            raise PilotError(f"latency request set {shape!r} differs from its batch size")
        columns = {
            name: [row[name] for row in requests]
            for name in (
                "option_type",
                "spot",
                "strike",
                "maturity",
                "rate",
                "dividend_yield",
                "volatility",
            )
        }
        for steps in specification["crr_depths"]:
            crr_times = []
            neural_times = {name: [] for name in models}
            operation_names = ["crr", *models]
            observed_orders = []
            for repetition in range(specification["warmups"] + specification["repetitions"]):
                rotation = repetition % len(operation_names)
                order = operation_names[rotation:] + operation_names[:rotation]
                if repetition >= specification["warmups"]:
                    observed_orders.append(order)
                for operation in order:
                    if operation == "crr":
                        started = time.perf_counter_ns()
                        prices_n = _core.crr_price_batch(
                            columns["option_type"],
                            ["american"] * batch_size,
                            columns["spot"],
                            columns["strike"],
                            columns["maturity"],
                            columns["rate"],
                            columns["dividend_yield"],
                            columns["volatility"],
                            [steps] * batch_size,
                            threads,
                        )["price"]
                        prices_n1 = _core.crr_price_batch(
                            columns["option_type"],
                            ["american"] * batch_size,
                            columns["spot"],
                            columns["strike"],
                            columns["maturity"],
                            columns["rate"],
                            columns["dividend_yield"],
                            columns["volatility"],
                            [steps + 1] * batch_size,
                            threads,
                        )["price"]
                        averaged = 0.5 * (np.asarray(prices_n) + np.asarray(prices_n1))
                        elapsed = time.perf_counter_ns() - started
                        if not bool(np.isfinite(averaged).all()):
                            raise PilotError("CRR latency request returned non-finite price")
                        if repetition >= specification["warmups"]:
                            crr_times.append(elapsed)
                    else:
                        started = time.perf_counter_ns()
                        physical_columns = {
                            name: np.asarray(
                                value,
                                dtype=object if name == "option_type" else np.float64,
                            )
                            for name, value in columns.items()
                        }
                        physical = physical_features(physical_columns)
                        prediction = predict_prices(models[operation], physical, batch_size)
                        elapsed = time.perf_counter_ns() - started
                        if not bool(np.isfinite(prediction).all()):
                            raise PilotError("neural latency request returned non-finite price")
                        if repetition >= specification["warmups"]:
                            neural_times[operation].append(elapsed)
            paired_speedups = {
                name: np.asarray(crr_times, dtype=np.float64) / np.asarray(values, dtype=np.float64)
                for name, values in neural_times.items()
            }
            measurements.append(
                {
                    "shape": shape,
                    "batch_size": batch_size,
                    "thread_budget": threads,
                    "crr_depth": steps,
                    "economic_requests": [dict(request) for request in requests],
                    "observed_measurement_orders": observed_orders,
                    "crr_adjacent_average_raw_ns": crr_times,
                    "neural_end_to_end_raw_ns": neural_times,
                    "median_speedup": {
                        name: statistics.median(values) for name, values in paired_speedups.items()
                    },
                    "median_speedup_confidence_interval": {
                        name: [float(np.min(values)), float(np.max(values))]
                        for name, values in paired_speedups.items()
                    },
                    "median_speedup_confidence_level_at_least": 0.95,
                    "median_speedup_interval_actual_coverage": 0.984375,
                }
            )
    return {"runtime": _runtime_metadata(), "measurements": measurements}


def _bisect_iv(
    case: Mapping[str, Any], target: float, specification: Mapping[str, Any]
) -> dict[str, Any]:
    low, high = (float(value) for value in specification["volatility_bracket"])
    steps = int(specification["crr_steps"])
    if not math.isfinite(target):
        return {"status": "non_finite_price", "volatility": None, "iterations": 0}
    f_low = crr_adjacent_average(case, steps, low) - target
    f_high = crr_adjacent_average(case, steps, high) - target
    if not math.isfinite(f_low) or not math.isfinite(f_high):
        return {"status": "non_finite_price", "volatility": None, "iterations": 0}
    if f_low * f_high > 0.0:
        return {"status": "unbracketed", "volatility": None, "iterations": 0}
    price_tolerance = specification["price_absolute_tolerance_over_spot"] * case["spot"]
    if abs(f_low) <= price_tolerance:
        return {"status": "converged", "volatility": low, "iterations": 0}
    if abs(f_high) <= price_tolerance:
        return {"status": "converged", "volatility": high, "iterations": 0}
    for iteration in range(1, specification["maximum_iterations"] + 1):
        middle = 0.5 * (low + high)
        f_middle = crr_adjacent_average(case, steps, middle) - target
        if not math.isfinite(f_middle):
            return {
                "status": "non_finite_price",
                "volatility": None,
                "iterations": iteration,
            }
        if (
            abs(f_middle) <= price_tolerance
            or 0.5 * (high - low) <= specification["volatility_absolute_tolerance"]
        ):
            return {"status": "converged", "volatility": middle, "iterations": iteration}
        if f_low * f_middle <= 0.0:
            high = middle
        else:
            low = middle
            f_low = f_middle
    return {
        "status": "iteration_exhausted",
        "volatility": None,
        "iterations": specification["maximum_iterations"],
    }


def run_implied_volatility(
    models: Mapping[str, AmericanPriceModel], specification: Mapping[str, Any]
) -> dict[str, Any]:
    rows = []
    for raw_case in specification["cases"]:
        case = dict(raw_case)
        case["volatility"] = case.pop("true_volatility")
        physical_columns = {
            name: np.asarray([case[name]], dtype=object if name == "option_type" else np.float64)
            for name in FEATURE_ORDER
        }
        physical = physical_features(physical_columns)
        targets = {"label": crr_adjacent_average(case, specification["crr_steps"])}
        targets.update(
            {name: float(predict_prices(model, physical, 1)[0]) for name, model in models.items()}
        )
        inversions = {
            name: _bisect_iv(case, target, specification) for name, target in targets.items()
        }
        label_iv = inversions["label"]["volatility"]
        for inversion in inversions.values():
            inversion["absolute_error_vs_label_iv"] = (
                None
                if label_iv is None or inversion["volatility"] is None
                else abs(inversion["volatility"] - label_iv)
            )
        rows.append({"case": raw_case["name"], "prices": targets, "inversions": inversions})
    failures = sum(
        inversion["status"] != "converged"
        for row in rows
        for inversion in row["inversions"].values()
    )
    return {"terminology": specification["terminology"], "rows": rows, "failures": failures}


def classify_outcome(
    accuracy: Mapping[str, Any],
    validation: Mapping[str, Any],
    latency: Mapping[str, Any],
    implied_volatility: Mapping[str, Any],
    acceptance: Mapping[str, Any],
    *,
    phase: str,
    evidence_complete: bool,
) -> dict[str, Any]:
    """Compute the exhaustive locked pilot interpretation from reported evidence."""
    arm_passed = {arm: bool(accuracy[arm]["gate"]["passed"]) for arm in ("scratch", "transfer")}
    reference_depth = int(acceptance["latency"]["reference_depth_for_interpretation"])
    reference_speedups = {
        f"{row['shape']}:{arm}": float(row["median_speedup"][arm])
        for row in latency["measurements"]
        if row["crr_depth"] == reference_depth
        for arm in ("scratch", "transfer")
    }
    speedup_passed = bool(reference_speedups) and all(
        value >= acceptance["latency"]["minimum_median_end_to_end_speedup"]
        for value in reference_speedups.values()
    )
    iv_errors = {
        arm: [
            row["inversions"][arm]["absolute_error_vs_label_iv"]
            for row in implied_volatility["rows"]
        ]
        for arm in ("scratch", "transfer")
    }
    iv_passed = all(
        errors
        and all(error is not None for error in errors)
        and max(float(error) for error in errors if error is not None)
        <= acceptance["implied_volatility"]["maximum_absolute_volatility_error"]
        for errors in iv_errors.values()
    )
    scratch_rmse = validation["scratch"]["slices"]["overall"]["normalized"]["rmse"]
    transfer_rmse = validation["transfer"]["slices"]["overall"]["normalized"]["rmse"]
    transfer_strictly_better = transfer_rmse < scratch_rmse
    passed_count = sum(arm_passed.values())
    if passed_count == 0:
        outcome = "failure_to_learn"
    elif passed_count == 1:
        outcome = "mixed_accuracy_outcome"
    elif not evidence_complete:
        outcome = "required_evidence_failed"
    elif phase == "validation":
        outcome = "eligible_for_final_evaluation"
    elif not transfer_strictly_better:
        outcome = "negative_transfer"
    elif speedup_passed and iv_passed:
        outcome = "promising"
    else:
        outcome = "accuracy_passed_other_feasibility_gate_failed"
    return {
        "phase": phase,
        "outcome": outcome,
        "arm_accuracy_passed": arm_passed,
        "reference_depth": reference_depth,
        "reference_median_speedups": reference_speedups,
        "all_reference_speedups_passed": speedup_passed,
        "all_iv_errors_passed": iv_passed,
        "transfer_validation_rmse_strictly_better": transfer_strictly_better,
    }


def execute_validation_run(
    *,
    project_root: Path,
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    output_directory: Path,
) -> dict[str, Any]:
    """Execute the reviewed manual run through validation, latency, and IV."""
    paths = protocol["paths"]
    dataset = project_root / paths["dataset"]
    training = load_toml(project_root / paths["training_config"])
    acceptance = load_toml(project_root / paths["acceptance_config"])
    latency = load_toml(project_root / paths["latency_config"])
    iv = load_toml(project_root / paths["iv_config"])
    try:
        torch.set_num_interop_threads(int(latency["torch_interop_threads"]))
    except RuntimeError:
        if torch.get_num_interop_threads() != int(latency["torch_interop_threads"]):
            raise PilotError("Torch inter-op thread budget was fixed incompatibly") from None
    torch.set_num_threads(int(training["training"]["num_threads"]))
    output_directory.mkdir(parents=True, exist_ok=False)
    protocol_digest = sha256_file(project_root / paths["protocol"])

    def record_entry_failure(gate: str, evidence: Mapping[str, Any]) -> None:
        report = {
            "schema_version": "american-neural-pilot-entry-failure-report/1",
            "lifecycle": {
                "phase": "entry_gate_failed",
                "final_evaluation_attempts": 0,
                "final_partition_consumed": False,
            },
            "protocol_sha256": protocol_digest,
            "dataset": protocol["dataset"],
            "source_artifact": protocol["source_artifact"],
            "failed_gate": gate,
            "evidence": dict(evidence),
            "runtime": _runtime_metadata(),
            "interpretation": (
                "terminal entry failure; no optimization or final evaluation permitted"
            ),
        }
        write_json_atomic(output_directory / "entry-failure-report.json", report)

    try:
        verify_partition_policy(dataset, manifest, "train")
        verify_partition_policy(dataset, manifest, "validation")
    except PilotError as error:
        record_entry_failure(
            "admission_policy_invariants",
            {"passed": False, "error_type": type(error).__name__, "message": str(error)},
        )
        raise
    try:
        pde_check = run_independent_pde_check(dataset, manifest, protocol)
    except Exception as error:
        record_entry_failure(
            "independent_pde_check",
            {"passed": False, "error_type": type(error).__name__, "message": str(error)},
        )
        raise PilotError(f"independent PDE check failed: {error}") from error
    if not pde_check["passed"]:
        record_entry_failure("independent_pde_check", pde_check)
        raise PilotError("independent PDE mapping-consistency gate failed")

    train_columns = read_partition_columns(
        dataset, manifest, "train", (*EVALUATION_COLUMNS, *POLICY_COLUMNS), verify_digest=False
    )
    validation_columns = read_partition_columns(
        dataset, manifest, "validation", (*EVALUATION_COLUMNS, *POLICY_COLUMNS), verify_digest=False
    )
    selected = select_rows_by_hash(
        train_columns["sample_id"],
        int(training["row_selection"]["row_budget"]),
        str(training["row_selection"]["salt"]),
    )
    train_physical_all = physical_features(train_columns)
    train_physical = train_physical_all[selected]
    train_prices = np.asarray(train_columns["american_price"], dtype=np.float64)[selected]
    validation_physical = physical_features(validation_columns)
    validation_prices = np.asarray(validation_columns["american_price"], dtype=np.float64)
    scaling = scaling_from_arrays(train_physical, train_prices)

    source_path = project_root / paths["source_artifact"]
    source_model, _ = load_physical_model(source_path)
    lifted = lift_european_network(source_model, scaling)
    probes = np.asarray(
        [
            [
                1.0 if probe["option_type"] == "call" else -1.0,
                *(probe[name] for name in FEATURE_ORDER[1:]),
            ]
            for probe in training["exact_lift"]["probes"]
        ],
        dtype=np.float64,
    )
    try:
        lift_result = verify_exact_lift(
            source_model,
            lifted,
            probes,
            rtol=training["exact_lift"]["relative_tolerance"],
            atol=training["exact_lift"]["absolute_tolerance"],
        )
    except ExactLiftError as error:
        record_entry_failure("exact_lift", error.evidence)
        raise PilotError(str(error)) from error
    common = {
        "batch_size": training["training"]["batch_size"],
        "epochs": training["training"]["epochs"],
        "learning_rate": training["optimizer"]["learning_rate"],
        "weight_decay": training["optimizer"]["weight_decay"],
        "beta1": training["optimizer"]["beta1"],
        "beta2": training["optimizer"]["beta2"],
        "epsilon": training["optimizer"]["epsilon"],
        "amsgrad": training["optimizer"]["amsgrad"],
        "maximize": training["optimizer"]["maximize"],
        "foreach": training["optimizer"]["foreach"],
        "capturable": training["optimizer"]["capturable"],
        "differentiable": training["optimizer"]["differentiable"],
        "fused": training["optimizer"]["fused"],
        "minimum_learning_rate": training["optimizer"]["minimum_learning_rate"],
        "schedule_period_epochs": training["optimizer"]["schedule_period_epochs"],
        "schedule_last_epoch": training["optimizer"]["schedule_last_epoch"],
        "num_threads": training["training"]["num_threads"],
    }
    scratch = fit_price_model(
        train_physical,
        train_prices,
        validation_physical,
        validation_prices,
        scaling,
        seed=training["arms"]["scratch_seed"],
        **common,
    )
    transfer = fit_price_model(
        train_physical,
        train_prices,
        validation_physical,
        validation_prices,
        scaling,
        seed=training["arms"]["transfer_seed"],
        initial_model=lifted,
        **common,
    )
    models = {"scratch": scratch.model, "transfer": transfer.model}
    validation_evidence = evaluate_models(
        models,
        validation_columns,
        acceptance,
        batch_size=training["training"]["batch_size"],
        gate_section="validation_final_entry",
    )
    latency_evidence = run_latency(models, latency)
    iv_evidence = run_implied_volatility(models, iv)
    artifact_metadata = {
        "experiment_name": training["experiment_name"],
        "dataset": {
            "manifest_sha256": protocol["dataset"]["manifest_sha256"],
            "train_sha256": protocol["dataset"]["train_sha256"],
            "validation_sha256": protocol["dataset"]["validation_sha256"],
            "selected_train_rows": len(selected),
        },
        "protocol": {"sha256": protocol_digest, "schema_version": protocol["schema_version"]},
        "code": {"tracked_inputs": protocol["tracked_inputs"]},
    }
    artifacts = {}
    for name, result in (("scratch", scratch), ("transfer", transfer)):
        directory = output_directory / name
        payload = save_american_artifact(
            directory,
            result.model,
            arm=name,
            training={
                "seed": training["arms"][f"{name}_seed"],
                "best_epoch": result.best_epoch,
                "best_validation_mse": result.best_validation_mse,
                "history": list(result.history),
            },
            source_lineage=(
                None
                if name == "scratch"
                else {
                    "manifest_sha256": protocol["source_artifact"]["manifest_sha256"],
                    "weights_sha256": protocol["source_artifact"]["weights_sha256"],
                    "lift": "task-9g-exact-european-to-american-v1",
                }
            ),
            **artifact_metadata,
        )
        artifacts[name] = {
            "path": str(directory.relative_to(project_root)),
            "manifest_sha256": sha256_file(directory / "artifact.json"),
            "weights_sha256": payload["weights"]["sha256"],
        }
    expected_latency_rows = len(latency["request_shapes"]) * len(latency["crr_depths"])
    latency_complete = len(latency_evidence["measurements"]) == expected_latency_rows and all(
        len(row["crr_adjacent_average_raw_ns"]) == latency["repetitions"]
        and all(
            len(values) == latency["repetitions"]
            for values in row["neural_end_to_end_raw_ns"].values()
        )
        for row in latency_evidence["measurements"]
    )
    iv_complete = (
        len(iv_evidence["rows"]) == len(iv["cases"])
        and iv_evidence["failures"] <= acceptance["implied_volatility"]["failures_max"]
    )
    validation_evidence_complete = pde_check["passed"] and latency_complete and iv_complete
    validation_passed = validation_evidence_complete and all(
        validation_evidence[name]["gate"]["passed"] for name in models
    )
    outcome = classify_outcome(
        validation_evidence,
        validation_evidence,
        latency_evidence,
        iv_evidence,
        acceptance,
        phase="validation",
        evidence_complete=validation_evidence_complete,
    )
    report = {
        "schema_version": "american-neural-pilot-raw-report/1",
        "lifecycle": {
            "phase": "validation_complete",
            "final_evaluation_attempts": 0,
            "final_partition_consumed": False,
        },
        "protocol_sha256": protocol_digest,
        "dataset": protocol["dataset"],
        "source_artifact": protocol["source_artifact"],
        "exact_lift": lift_result,
        "independent_pde_check": pde_check,
        "selected_train_rows": len(selected),
        "artifacts": artifacts,
        "validation": validation_evidence,
        "validation_evidence_complete": {
            "passed": validation_evidence_complete,
            "independent_pde_check": pde_check["passed"],
            "matched_latency": latency_complete,
            "synthetic_implied_volatility": iv_complete,
        },
        "latency": latency_evidence,
        "implied_volatility": iv_evidence,
        "validation_final_entry_passed": validation_passed,
        "runtime": _runtime_metadata(),
        "interpretation": {
            "claim_scope": "one-seed one-budget mapping-only feasibility pilot",
            "locked_experiments_ran": True,
            "outcome": outcome,
        },
    }
    write_json_atomic(output_directory / "validation-report.json", report)
    return report
