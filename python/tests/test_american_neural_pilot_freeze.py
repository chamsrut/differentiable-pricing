"""Synthetic-fixture tests for strict Task 9G report freezing."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
from itertools import product
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DIGEST = "a" * 64


def _module():
    path = PROJECT_ROOT / "scripts/freeze_american_neural_pilot_results.py"
    spec = importlib.util.spec_from_file_location("task9g_freeze", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configs() -> tuple[dict, dict, dict]:
    acceptance = {
        "quantile_method": "linear",
        "bins": {
            "expiry_edges_years": [0.25, 1.0, 2.0],
            "log_moneyness_edges": [-0.2, 0.2],
            "volatility_edges": [0.15, 0.5],
        },
        "validation_final_entry": {
            "normalized_rmse_max": 0.003,
            "normalized_p99_absolute_error_max": 0.015,
            "normalized_maximum_absolute_error_max": 0.08,
            "maximum_material_bound_violations": 0,
            "maximum_material_shape_violations": 0,
        },
        "final_accuracy": {
            "normalized_rmse_max": 0.003,
            "normalized_p99_absolute_error_max": 0.015,
            "normalized_maximum_absolute_error_max": 0.08,
            "maximum_material_bound_violations": 0,
            "maximum_material_shape_violations": 0,
        },
        "latency": {
            "reference_depth_for_interpretation": 1024,
            "minimum_median_end_to_end_speedup": 10.0,
        },
        "implied_volatility": {
            "maximum_absolute_volatility_error": 0.01,
            "failures_max": 0,
        },
    }
    latency = {
        "request_shapes": ["single", "batch8"],
        "batch_sizes": [1, 8],
        "thread_budgets": [1, 4],
        "crr_depths": [256, 512, 1024, 2048, 4096],
        "warmups": 2,
        "repetitions": 7,
        "cases": [
            {
                "name": f"case-{index}",
                "option_type": "call" if index % 2 == 0 else "put",
                "spot": 100.0,
                "strike": 100.0 + index,
                "maturity": 1.0,
                "rate": 0.03,
                "dividend_yield": 0.02,
                "volatility": 0.2,
            }
            for index in range(8)
        ],
        "request_case_names": {
            "single": ["case-0"],
            "batch8": [f"case-{index}" for index in range(8)],
        },
    }
    iv = {
        "terminology": "surface",
        "volatility_bracket": [0.05, 0.8],
        "maximum_iterations": 80,
        "cases": [{"name": "a"}, {"name": "b"}],
    }
    return acceptance, latency, iv


def _runtime() -> dict:
    return {
        "python": "x",
        "numpy": "x",
        "torch": "x",
        "platform": "x",
        "machine": "x",
        "processor": "x",
        "logical_cpu_count": 1,
        "cpu_affinity": [0],
        "core_version": "x",
        "build_configuration": "x",
        "cxx_compiler": "x",
    }


def _audit_row(*, prediction: float = 10.0, baseline: bool = False) -> dict:
    diagnostic_violations = None
    if not baseline:
        diagnostic_violations = {name: -1.0 for name in _module().DIAGNOSTIC_CHECKS}
        diagnostic_violations["american_put_rate_aware_upper_bound"] = None
    return {
        "sample_id": "sample",
        "option_type": "call",
        "spot": 100.0,
        "strike": 100.0,
        "maturity": 1.0,
        "dividend_yield": 0.0,
        "volatility": 0.2,
        "early_exercise_premium": 0.0,
        "earliest_exercise_step": -1,
        "reference_price": 10.0,
        "prediction": prediction,
        "diagnostic_tolerance": None if baseline else 0.0001,
        "diagnostic_violations": diagnostic_violations,
    }


def _slices(row: dict, acceptance: dict) -> dict:
    module = _module()
    output = {}
    for name in module._slice_names(acceptance):
        selected = module._row_in_slice(row, name, acceptance)
        if not selected:
            output[name] = {"rows": 0, "physical": None, "normalized": None}
            continue
        physical = module._statistics([row["reference_price"]], [row["prediction"]])
        scale = row["spot"] * math.exp(-row["dividend_yield"] * row["maturity"])
        normalized = module._statistics(
            [row["reference_price"] / scale], [row["prediction"] / scale]
        )
        output[name] = {"rows": 1, "physical": physical, "normalized": normalized}
    return output


def _model(acceptance: dict, *, prediction: float = 10.0, baseline: bool = False) -> dict:
    row = _audit_row(prediction=prediction, baseline=baseline)
    value = {
        "slices": _slices(row, acceptance),
        "diagnostics": (
            {
                "not_applicable": "stored paired no-learning baseline",
                "stored_error_equals_early_exercise_premium": True,
            }
            if baseline
            else {
                "checks": {
                    name: {
                        "material_violations": 0,
                        "maximum_violation": 0.0,
                        "eligible_rows": int(row["diagnostic_violations"][name] is not None),
                    }
                    for name in _module().DIAGNOSTIC_CHECKS
                },
                "material_bound_violations": 0,
                "material_shape_violations": 0,
                "american_greek_accuracy_claim": False,
            }
        ),
        "audit_rows": [row],
    }
    if not baseline:
        overall = value["slices"]["overall"]["normalized"]
        checks = {
            "normalized_rmse": overall["rmse"] <= 0.003,
            "normalized_p99_absolute_error": overall["p99_absolute_error"] <= 0.015,
            "normalized_maximum_absolute_error": overall["maximum_absolute_error"] <= 0.08,
            "material_bound_violations": True,
            "material_shape_violations": True,
        }
        value["gate"] = {"checks": checks, "passed": all(checks.values())}
    return value


def _model_from_rows(
    acceptance: dict,
    rows: list[dict],
    *,
    baseline: bool = False,
    section: str = "validation_final_entry",
) -> dict:
    module = _module()
    slices = {}
    for slice_name in module._slice_names(acceptance):
        selected = [row for row in rows if module._row_in_slice(row, slice_name, acceptance)]
        if not selected:
            slices[slice_name] = {"rows": 0, "physical": None, "normalized": None}
            continue
        references = [row["reference_price"] for row in selected]
        predictions = [row["prediction"] for row in selected]
        scales = [
            row["spot"] * math.exp(-row["dividend_yield"] * row["maturity"]) for row in selected
        ]
        slices[slice_name] = {
            "rows": len(selected),
            "physical": module._statistics(references, predictions),
            "normalized": module._statistics(
                [value / scale for value, scale in zip(references, scales, strict=True)],
                [value / scale for value, scale in zip(predictions, scales, strict=True)],
            ),
        }
    if baseline:
        return {
            "slices": slices,
            "diagnostics": {
                "not_applicable": "stored paired no-learning baseline",
                "stored_error_equals_early_exercise_premium": True,
            },
            "audit_rows": rows,
        }
    checks = {}
    for check_name in module.DIAGNOSTIC_CHECKS:
        eligible = [
            (row["diagnostic_violations"][check_name], row["diagnostic_tolerance"])
            for row in rows
            if row["diagnostic_violations"][check_name] is not None
        ]
        checks[check_name] = {
            "material_violations": sum(value > tolerance for value, tolerance in eligible),
            "maximum_violation": max((max(value, 0.0) for value, _ in eligible), default=0.0),
            "eligible_rows": len(eligible),
        }
    bound = sum(
        checks[name]["material_violations"]
        for name in (
            "intrinsic_lower_bound",
            "european_comparator_lower_bound",
            "american_call_upper_bound",
            "american_put_rate_aware_upper_bound",
        )
    )
    shape = sum(
        checks[name]["material_violations"]
        for name in ("spot_monotonicity", "spot_convexity", "volatility_monotonicity")
    )
    overall = slices["overall"]["normalized"]
    limits = acceptance[section]
    gate_checks = {
        "normalized_rmse": overall["rmse"] <= limits["normalized_rmse_max"],
        "normalized_p99_absolute_error": overall["p99_absolute_error"]
        <= limits["normalized_p99_absolute_error_max"],
        "normalized_maximum_absolute_error": overall["maximum_absolute_error"]
        <= limits["normalized_maximum_absolute_error_max"],
        "material_bound_violations": bound <= limits["maximum_material_bound_violations"],
        "material_shape_violations": shape <= limits["maximum_material_shape_violations"],
    }
    return {
        "slices": slices,
        "diagnostics": {
            "checks": checks,
            "material_bound_violations": bound,
            "material_shape_violations": shape,
            "american_greek_accuracy_claim": False,
        },
        "gate": {"checks": gate_checks, "passed": all(gate_checks.values())},
        "audit_rows": rows,
    }


def _varied_models(acceptance: dict, row_count: int) -> dict:
    states = list(
        product(
            ("call", "put"),
            (0.1, 0.5, 1.5, 2.5),
            (-0.3, 0.0, 0.3),
            (0.1, 0.3, 0.6),
            (0.0, 0.25),
            (-1, 3),
        )
    )
    scratch_rows = []
    transfer_rows = []
    baseline_rows = []
    for index in range(row_count):
        option_type, maturity, log_moneyness, volatility, premium, exercise_step = states[
            (index * 17) % len(states)
        ]
        spot = 100.0 * math.exp(log_moneyness)
        reference_price = 10.0 + premium + 0.01 * (index % 11)
        common = {
            "sample_id": f"sample-{index:05d}",
            "option_type": option_type,
            "spot": spot,
            "strike": 100.0,
            "maturity": maturity,
            "dividend_yield": 0.01,
            "volatility": volatility,
            "early_exercise_premium": premium,
            "earliest_exercise_step": exercise_step,
            "reference_price": reference_price,
        }
        scratch_error = 0.01 + 0.001 * (index % 17)
        transfer_error = 0.008 + 0.0008 * (index % 17)
        scratch = _audit_row(prediction=reference_price + scratch_error)
        scratch.update(common)
        transfer = _audit_row(prediction=reference_price + transfer_error)
        transfer.update(common)
        if option_type == "put":
            for row in (scratch, transfer):
                row["diagnostic_violations"]["american_call_upper_bound"] = None
                row["diagnostic_violations"]["american_put_rate_aware_upper_bound"] = -1.0
        baseline = _audit_row(baseline=True)
        baseline.update(common)
        baseline["prediction"] = reference_price - premium
        scratch_rows.append(scratch)
        transfer_rows.append(transfer)
        baseline_rows.append(baseline)
    return {
        "scratch": _model_from_rows(acceptance, scratch_rows),
        "transfer": _model_from_rows(acceptance, transfer_rows),
        "european_crr_baseline": _model_from_rows(acceptance, baseline_rows, baseline=True),
    }


def _install_varied_models(validation: dict, final: dict, acceptance: dict, row_count: int) -> None:
    module = _module()
    models = _varied_models(acceptance, row_count)
    validation["validation"] = models
    validation["interpretation"]["outcome"] = module._outcome(
        models,
        models,
        validation["latency"],
        validation["implied_volatility"],
        acceptance,
        "validation",
        True,
    )
    final_models = copy.deepcopy(models)
    final["models_and_baseline"] = final_models
    final["outcome"] = module._outcome(
        final_models,
        models,
        validation["latency"],
        validation["implied_volatility"],
        acceptance,
        "final",
        True,
    )


def _pde(protocol: dict) -> dict:
    case = protocol["independent_pde_check"]["cases"][0]
    solves = []
    for grid in protocol["independent_pde_check"]["grids"]:
        requested_maximum = grid["spot_maximum_factor"] * max(case["spot"], case["strike"])
        target_step = requested_maximum / grid["spot_intervals"]
        strike_index = max(1, math.floor(case["strike"] / target_step + 0.5))
        step = case["strike"] / strike_index
        actual_intervals = max(
            strike_index + 1,
            math.ceil(requested_maximum / step - 1.0e-9),
        )
        requested = {
            "option_type": case["option_type"],
            "exercise_style": "american",
            "spot": case["spot"],
            "strike": case["strike"],
            "valuation_time": 0.0,
            "expiry_time": case["maturity"],
            "volatility": case["volatility"],
            "continuous_carry": case["dividend_yield"],
            "curve_times": [0.0, case["maturity"]],
            "curve_log_discounts": [0.0, -case["rate"] * case["maturity"]],
            "dividends": [],
            "settlement": "cash",
            "contract_multiplier": 1.0,
            "spot_intervals": grid["spot_intervals"],
            "time_steps": grid["time_steps"],
            "spot_maximum": requested_maximum,
            "rannacher_steps": 2,
            "psor_tolerance": 1.0e-11,
            "psor_relaxation": 1.2,
            "psor_maximum_iterations": 50000,
        }
        result = {
            "price": 10.0,
            "solver_status": "discrete_system_converged",
            "discretization_accuracy": "not_assessed",
            "spot_intervals": actual_intervals,
            "spot_maximum": actual_intervals * step,
            "spot_step": step,
            "strike_node_index": strike_index,
            "time_steps": grid["time_steps"],
            "rannacher_steps": 2,
            "damped_half_steps": 4,
            "crank_nicolson_steps": grid["time_steps"] - 2,
            "upwinded_rows": 0,
            "linear_solves": 0,
            "psor_solves": grid["time_steps"] + 2,
            "psor_total_iterations": grid["time_steps"] + 2,
            "psor_maximum_iterations_used": 1,
            "psor_tolerance": 1.0e-11,
            "psor_relaxation": 1.2,
            "maximum_lcp_residual": 1.0e-12,
            "maximum_relative_lcp_residual": 1.0e-12,
            "aligned_times": [0.0, case["maturity"]],
            "dividend_events": [],
        }
        solves.append({"grid": grid["name"], "requested_input": requested, "result": result})
    return {
        "rows": [
            {
                "sample_id": case["sample_id"],
                "coarse_price": 10.0,
                "fine_price": 10.0,
                "crr_label": 10.0,
                "solves": solves,
                "normalized_refinement_difference": 0.0,
                "normalized_pde_crr_difference": 0.0,
                "passed": True,
            }
        ],
        "passed": True,
        "interpretation": protocol["independent_pde_check"]["interpretation"],
    }


def _latency(config: dict) -> dict:
    operations = ["crr", "scratch", "transfer"]
    orders = []
    for repetition in range(config["warmups"], config["warmups"] + config["repetitions"]):
        offset = repetition % 3
        orders.append(operations[offset:] + operations[:offset])
    rows = []
    for shape, batch, threads in zip(
        config["request_shapes"], config["batch_sizes"], config["thread_budgets"], strict=True
    ):
        for depth in config["crr_depths"]:
            crr = [1000 + index for index in range(7)]
            neural = {arm: [50 + index for index in range(7)] for arm in ("scratch", "transfer")}
            ratios = {
                arm: [left / right for left, right in zip(crr, values, strict=True)]
                for arm, values in neural.items()
            }
            rows.append(
                {
                    "shape": shape,
                    "batch_size": batch,
                    "thread_budget": threads,
                    "crr_depth": depth,
                    "economic_requests": [
                        next(case for case in config["cases"] if case["name"] == name)
                        for name in config["request_case_names"][shape]
                    ],
                    "observed_measurement_orders": orders,
                    "crr_adjacent_average_raw_ns": crr,
                    "neural_end_to_end_raw_ns": neural,
                    "median_speedup": {arm: sorted(values)[3] for arm, values in ratios.items()},
                    "median_speedup_confidence_interval": {
                        arm: [min(values), max(values)] for arm, values in ratios.items()
                    },
                    "median_speedup_confidence_level_at_least": 0.95,
                    "median_speedup_interval_actual_coverage": 0.984375,
                }
            )
    return {"runtime": _runtime(), "measurements": rows}


def _iv(config: dict) -> dict:
    rows = []
    for case in config["cases"]:
        inversion = {
            "status": "converged",
            "volatility": 0.2,
            "iterations": 3,
            "absolute_error_vs_label_iv": 0.0,
        }
        rows.append(
            {
                "case": case["name"],
                "prices": {"label": 10.0, "scratch": 10.0, "transfer": 10.0},
                "inversions": {
                    "label": copy.deepcopy(inversion),
                    "scratch": copy.deepcopy(inversion),
                    "transfer": copy.deepcopy(inversion),
                },
            }
        )
    return {"terminology": "surface", "rows": rows, "failures": 0}


def _reports() -> tuple[dict, dict, dict, dict, dict, dict]:
    module = _module()
    acceptance, latency_config, iv_config = _configs()
    dataset = {"locked_final_sha256": "b" * 64}
    source = {"weights_sha256": "c" * 64}
    protocol = {
        "dataset": dataset,
        "source_artifact": source,
        "limitations": dict(module.PILOT_LIMITATIONS),
        "independent_pde_check": {
            "interpretation": "mapping-consistency only",
            "maximum_normalized_refinement_difference": 0.0005,
            "maximum_normalized_pde_crr_difference": 0.001,
            "valuation_time": 0.0,
            "settlement": "cash",
            "contract_multiplier": 1.0,
            "rannacher_steps": 2,
            "psor_tolerance": 1.0e-11,
            "psor_relaxation": 1.2,
            "psor_maximum_iterations": 50000,
            "grids": [
                {
                    "name": "coarse",
                    "spot_intervals": 400,
                    "time_steps": 200,
                    "spot_maximum_factor": 4.0,
                },
                {
                    "name": "fine",
                    "spot_intervals": 800,
                    "time_steps": 400,
                    "spot_maximum_factor": 4.0,
                },
            ],
            "cases": [
                {
                    "sample_id": "pde",
                    "option_type": "call",
                    "spot": 100.0,
                    "strike": 100.0,
                    "maturity": 1.0,
                    "rate": 0.03,
                    "dividend_yield": 0.02,
                    "volatility": 0.2,
                }
            ],
        },
    }
    models = {
        "scratch": _model(acceptance),
        "transfer": _model(acceptance),
        "european_crr_baseline": _model(acceptance, baseline=True),
    }
    latency = _latency(latency_config)
    iv = _iv(iv_config)
    validation_outcome = module._outcome(
        models, models, latency, iv, acceptance, "validation", True
    )
    exact_lift = {
        "passed": True,
        "probes": 8,
        "probe_prices": [{"source": 10.0, "lifted": 10.0} for _ in range(8)],
        "maximum_absolute_difference": 0.0,
        "maximum_relative_difference": 0.0,
        "relative_tolerance": 1.0e-12,
        "absolute_tolerance": 1.0e-12,
    }
    validation = {
        "schema_version": "american-neural-pilot-raw-report/1",
        "lifecycle": {
            "phase": "validation_complete",
            "final_evaluation_attempts": 0,
            "final_partition_consumed": False,
        },
        "protocol_sha256": DIGEST,
        "dataset": dataset,
        "source_artifact": source,
        "exact_lift": exact_lift,
        "independent_pde_check": _pde(protocol),
        "selected_train_rows": 32768,
        "artifacts": {
            arm: {"path": arm, "manifest_sha256": DIGEST, "weights_sha256": DIGEST}
            for arm in ("scratch", "transfer")
        },
        "validation": models,
        "validation_evidence_complete": {
            "passed": True,
            "independent_pde_check": True,
            "matched_latency": True,
            "synthetic_implied_volatility": True,
        },
        "latency": latency,
        "implied_volatility": iv,
        "validation_final_entry_passed": True,
        "runtime": _runtime(),
        "interpretation": {
            "claim_scope": "one-seed one-budget mapping-only feasibility pilot",
            "locked_experiments_ran": True,
            "limitations": dict(module.PILOT_LIMITATIONS),
            "outcome": validation_outcome,
        },
    }
    final_models = copy.deepcopy(models)
    final = {
        "schema_version": "american-neural-pilot-final-report/1",
        "protocol_sha256": DIGEST,
        "attempt": 1,
        "partition": "interpolation_test",
        "partition_sha256": "b" * 64,
        "models_and_baseline": final_models,
        "all_arms_passed": True,
        "outcome": module._outcome(final_models, models, latency, iv, acceptance, "final", True),
        "runtime": _runtime(),
        "lifecycle": {"final_partition_consumed": True, "second_attempt_allowed": False},
    }
    return validation, final, protocol, acceptance, latency_config, iv_config


def _validate(validation: dict, final: dict | None = None) -> str:
    protocol_values = _reports()[2:]
    protocol, acceptance, latency, iv = protocol_values
    return _module().validate_raw_reports(
        validation, final, protocol, DIGEST, acceptance, latency, iv
    )


def test_raw_report_validator_accepts_consistent_fixture() -> None:
    validation, final, protocol, acceptance, latency, iv = _reports()
    assert (
        _module().validate_raw_reports(validation, final, protocol, DIGEST, acceptance, latency, iv)
        == "final_complete"
    )


@pytest.mark.parametrize("model", ["transfer", "european_crr_baseline"])
def test_raw_report_requires_identical_rows_across_models(model) -> None:
    module = _module()
    validation, final, protocol, acceptance, latency, iv = _reports()
    validation["validation"][model]["audit_rows"][0]["sample_id"] = "different-sample"
    with pytest.raises(module.FreezeError, match="differ across arms"):
        module.validate_raw_reports(validation, final, protocol, DIGEST, acceptance, latency, iv)


def test_final_report_requires_exact_runtime_metadata() -> None:
    module = _module()
    validation, final, protocol, acceptance, latency, iv = _reports()
    del final["runtime"]["python"]
    final["runtime"]["unknown"] = "x"
    with pytest.raises(module.FreezeError, match="runtime"):
        module.validate_raw_reports(validation, final, protocol, DIGEST, acceptance, latency, iv)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda validation: validation["validation"]["scratch"].update({"unknown": True}),
            "unknown",
        ),
        (
            lambda validation: validation["validation"]["scratch"]["slices"]["overall"][
                "normalized"
            ].update({"rmse": 0.2}),
            "primitive|recomputation",
        ),
        (lambda validation: validation["latency"].update({"measurements": []}), "evidence"),
        (lambda validation: validation["implied_volatility"].update({"failures": 1}), "failure"),
        (
            lambda validation: validation["exact_lift"]["probe_prices"][0].update({"lifted": 10.1}),
            "maxima",
        ),
        (
            lambda validation: validation["independent_pde_check"].update(
                {"interpretation": "invented"}
            ),
            "interpretation",
        ),
    ],
)
def test_raw_report_validator_rejects_nested_or_recomputed_drift(mutation, match) -> None:
    module = _module()
    validation, final, protocol, acceptance, latency, iv = _reports()
    mutation(validation)
    with pytest.raises(module.FreezeError, match=match):
        module.validate_raw_reports(validation, final, protocol, DIGEST, acceptance, latency, iv)


def test_raw_report_validator_rejects_pde_threshold_drift_marked_passed() -> None:
    module = _module()
    validation, final, protocol, acceptance, latency, iv = _reports()
    row = validation["independent_pde_check"]["rows"][0]
    row["solves"][1]["result"]["price"] = 20.0
    row["fine_price"] = 20.0
    with pytest.raises(module.FreezeError, match="refinement recomputation"):
        module.validate_raw_reports(validation, final, protocol, DIGEST, acceptance, latency, iv)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("psor_total_iterations", -1, "counts"),
        ("maximum_relative_lcp_residual", 1.0e-6, "residual"),
        ("strike_node_index", 999999, "grid"),
    ],
)
def test_raw_report_validator_rejects_impossible_pde_diagnostics(field, value, match) -> None:
    module = _module()
    validation, final, protocol, acceptance, latency, iv = _reports()
    validation["independent_pde_check"]["rows"][0]["solves"][0]["result"][field] = value
    with pytest.raises(module.FreezeError, match=match):
        module.validate_raw_reports(validation, final, protocol, DIGEST, acceptance, latency, iv)


def test_entry_failure_freezes_without_final_report() -> None:
    validation, _, protocol, acceptance, latency, iv = _reports()
    entry = {
        "schema_version": "american-neural-pilot-entry-failure-report/1",
        "lifecycle": {
            "phase": "entry_gate_failed",
            "final_evaluation_attempts": 0,
            "final_partition_consumed": False,
        },
        "protocol_sha256": DIGEST,
        "dataset": validation["dataset"],
        "source_artifact": validation["source_artifact"],
        "failed_gate": "admission_policy_invariants",
        "evidence": {"passed": False, "error_type": "PilotError", "message": "mismatch"},
        "runtime": _runtime(),
        "interpretation": "terminal entry failure; no optimization or final evaluation permitted",
    }
    assert (
        _module().validate_raw_reports(entry, None, protocol, DIGEST, acceptance, latency, iv)
        == "entry_failure"
    )


def test_entry_failure_snapshot_revalidates_gate_and_forbids_final_source(tmp_path) -> None:
    module = _module()
    validation, _, protocol, acceptance, latency, iv = _reports()
    entry = {
        "schema_version": "american-neural-pilot-entry-failure-report/1",
        "lifecycle": {
            "phase": "entry_gate_failed",
            "final_evaluation_attempts": 0,
            "final_partition_consumed": False,
        },
        "protocol_sha256": DIGEST,
        "dataset": validation["dataset"],
        "source_artifact": validation["source_artifact"],
        "failed_gate": "admission_policy_invariants",
        "evidence": {"passed": False, "error_type": "PilotError", "message": "mismatch"},
        "runtime": _runtime(),
        "interpretation": "terminal entry failure; no optimization or final evaluation permitted",
    }
    entry_path = tmp_path / "entry.json"
    entry_path.write_text(json.dumps(entry), encoding="utf-8")
    snapshot = module.extract_snapshot(
        entry,
        None,
        state="entry_failure",
        protocol_path=PROJECT_ROOT / "configs/american_neural_pilot_protocol_v1.toml",
        validation_path=entry_path,
        final_path=None,
        acceptance=acceptance,
    )
    module.validate_snapshot(
        snapshot,
        PROJECT_ROOT / "configs/american_neural_pilot_protocol_v1.toml",
        protocol,
        acceptance,
        latency,
        iv,
    )
    invented = copy.deepcopy(snapshot)
    invented["entry_failure"]["failed_gate"] = "invented"
    invented["outcome"]["failed_gate"] = "invented"
    with pytest.raises(module.FreezeError, match="unknown entry gate"):
        module.validate_snapshot(
            invented,
            PROJECT_ROOT / "configs/american_neural_pilot_protocol_v1.toml",
            protocol,
            acceptance,
            latency,
            iv,
        )
    with_final_source = copy.deepcopy(snapshot)
    with_final_source["source_reports"]["final"] = {"file": "final.json", "sha256": DIGEST}
    with pytest.raises(module.FreezeError, match="lifecycle"):
        module.validate_snapshot(
            with_final_source,
            PROJECT_ROOT / "configs/american_neural_pilot_protocol_v1.toml",
            protocol,
            acceptance,
            latency,
            iv,
        )


@pytest.mark.parametrize("iterations", [-1, True, 81])
def test_raw_report_rejects_invalid_iv_iteration_evidence(iterations) -> None:
    module = _module()
    validation, final, protocol, acceptance, latency, iv = _reports()
    validation["implied_volatility"]["rows"][0]["inversions"]["scratch"]["iterations"] = iterations
    with pytest.raises(module.FreezeError, match="iteration evidence"):
        module.validate_raw_reports(validation, final, protocol, DIGEST, acceptance, latency, iv)


def test_validation_failure_freezes_without_final_report() -> None:
    validation, _, protocol, acceptance, latency, iv = _reports()
    for arm in ("scratch", "transfer"):
        validation["validation"][arm] = _model(acceptance, prediction=20.0)
    validation["validation_final_entry_passed"] = False
    validation["interpretation"]["outcome"] = _module()._outcome(
        validation["validation"],
        validation["validation"],
        validation["latency"],
        validation["implied_volatility"],
        acceptance,
        "validation",
        True,
    )
    assert (
        _module().validate_raw_reports(validation, None, protocol, DIGEST, acceptance, latency, iv)
        == "validation_terminal"
    )


def _final_failure_report(protocol: dict, *, stage: str, observed: str | None) -> dict:
    return {
        "schema_version": "american-neural-pilot-final-attempt-failure-report/1",
        "protocol_sha256": DIGEST,
        "attempt": 1,
        "partition": "interpolation_test",
        "expected_partition_sha256": protocol["dataset"]["locked_final_sha256"],
        "observed_partition_sha256": observed,
        "failure_stage": stage,
        "error": {"type": "KeyboardInterrupt", "message": "KeyboardInterrupt"},
        "runtime": _runtime(),
        "lifecycle": {
            "final_partition_consumed": True,
            "second_attempt_allowed": False,
        },
        "interpretation": ("terminal consumed final-attempt failure; no retry under this protocol"),
    }


@pytest.mark.parametrize(
    ("stage", "observed"),
    [
        ("reservation", DIGEST),
        ("ledger_reservation", DIGEST),
        ("manifest_identity", DIGEST),
        ("partition_read", None),
        ("model_evaluation", DIGEST),
        ("report_write", None),
    ],
)
def test_consumed_final_failure_rejects_stage_identity_drift(
    stage: str, observed: str | None
) -> None:
    module = _module()
    validation, _, protocol, acceptance, latency, iv = _reports()
    failure = _final_failure_report(protocol, stage=stage, observed=observed)
    with pytest.raises(module.FreezeError, match="stage/identity relationship"):
        module.validate_raw_reports(validation, failure, protocol, DIGEST, acceptance, latency, iv)


def test_consumed_final_failure_freezes_without_final_metrics(tmp_path) -> None:
    module = _module()
    validation, _, protocol, acceptance, latency, iv = _reports()
    failure = _final_failure_report(protocol, stage="reservation", observed=None)
    assert (
        module.validate_raw_reports(validation, failure, protocol, DIGEST, acceptance, latency, iv)
        == "final_failed_consumed"
    )
    validation_path = tmp_path / "validation.json"
    failure_path = tmp_path / "failure.json"
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    failure_path.write_text(json.dumps(failure), encoding="utf-8")
    snapshot = module.extract_snapshot(
        validation,
        failure,
        state="final_failed_consumed",
        protocol_path=PROJECT_ROOT / "configs/american_neural_pilot_protocol_v1.toml",
        validation_path=validation_path,
        final_path=failure_path,
        acceptance=acceptance,
    )
    assert snapshot["lifecycle"]["final_partition_consumed"] is True
    assert snapshot["audit"]["final"] is None
    assert snapshot["arms"]["scratch"]["final"] is None
    assert snapshot["outcome"]["failure_stage"] == "reservation"
    assert snapshot["outcome"]["expected_partition_sha256"] == "b" * 64
    assert snapshot["outcome"]["observed_partition_sha256"] is None
    module.validate_snapshot(
        snapshot,
        PROJECT_ROOT / "configs/american_neural_pilot_protocol_v1.toml",
        protocol,
        acceptance,
        latency,
        iv,
    )
    invalid_snapshot = copy.deepcopy(snapshot)
    invalid_snapshot["outcome"]["observed_partition_sha256"] = DIGEST
    with pytest.raises(module.FreezeError, match="stage/identity relationship"):
        module.validate_snapshot(
            invalid_snapshot,
            PROJECT_ROOT / "configs/american_neural_pilot_protocol_v1.toml",
            protocol,
            acceptance,
            latency,
            iv,
        )


def test_snapshot_retains_all_slices_and_latency_depths(tmp_path) -> None:
    validation, final, _, acceptance, latency, _ = _reports()
    validation_path = tmp_path / "validation.json"
    final_path = tmp_path / "final.json"
    validation_path.write_text(json.dumps(validation))
    final_path.write_text(json.dumps(final))
    snapshot = _module().extract_snapshot(
        validation,
        final,
        state="final_complete",
        protocol_path=PROJECT_ROOT / "configs/american_neural_pilot_protocol_v1.toml",
        validation_path=validation_path,
        final_path=final_path,
        acceptance=acceptance,
    )
    assert set(snapshot["arms"]["scratch"]["validation"]["slices"]) == _module()._slice_names(
        acceptance
    )
    assert {row["crr_depth"] for row in snapshot["latency"]["measurements"]} == set(
        latency["crr_depths"]
    )
    assert snapshot["audit"]["validation"]["row_count"] == 1
    assert "audit_rows" not in snapshot["arms"]["scratch"]["validation"]
    assert "audit_rows" not in snapshot["arms"]["transfer"]["validation"]
    assert "audit_rows" not in snapshot["baseline"]["validation"]
    assert set(snapshot["audit"]["validation"]["models"]) == {
        "scratch",
        "transfer",
        "european_crr_baseline",
    }
    overall = snapshot["audit"]["validation"]["models"]["scratch"]["slices"]["overall"][
        "normalized"
    ]
    assert overall["count"] == 1
    assert set(overall["quantiles"]) == {"p95", "p99"}


def test_snapshot_round_trip_recomputes_published_metrics_and_verdict(tmp_path) -> None:
    module = _module()
    validation, final, protocol, acceptance, latency, iv = _reports()
    validation_path = tmp_path / "validation.json"
    final_path = tmp_path / "final.json"
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    final_path.write_text(json.dumps(final), encoding="utf-8")
    snapshot = module.extract_snapshot(
        validation,
        final,
        state="final_complete",
        protocol_path=PROJECT_ROOT / "configs/american_neural_pilot_protocol_v1.toml",
        validation_path=validation_path,
        final_path=final_path,
        acceptance=acceptance,
    )
    module.validate_snapshot(
        snapshot,
        PROJECT_ROOT / "configs/american_neural_pilot_protocol_v1.toml",
        protocol,
        acceptance,
        latency,
        iv,
    )
    metric_drift = copy.deepcopy(snapshot)
    metric_drift["arms"]["scratch"]["validation"]["slices"]["overall"]["normalized"]["rmse"] = 0.5
    with pytest.raises(module.FreezeError, match="recomputation"):
        module.validate_snapshot(
            metric_drift,
            PROJECT_ROOT / "configs/american_neural_pilot_protocol_v1.toml",
            protocol,
            acceptance,
            latency,
            iv,
        )
    verdict_drift = copy.deepcopy(snapshot)
    verdict_drift["outcome"]["outcome"] = "promising"
    with pytest.raises(module.FreezeError, match="outcome"):
        module.validate_snapshot(
            verdict_drift,
            PROJECT_ROOT / "configs/american_neural_pilot_protocol_v1.toml",
            protocol,
            acceptance,
            latency,
            iv,
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda audit: audit["models"]["scratch"]["slices"]["overall"]["normalized"].update(
                {"sum_absolute_error": 0.25}
            ),
            "primitive|recomputation",
        ),
        (
            lambda audit: audit["models"]["scratch"]["slices"]["overall"]["normalized"][
                "quantiles"
            ]["p99"].update({"lower_value": 0.01, "upper_value": 0.01}),
            "primitive|recomputation",
        ),
        (
            lambda audit: audit["models"]["scratch"]["slices"]["overall"]["normalized"].update(
                {
                    "sum_squared_error": audit["models"]["scratch"]["slices"]["overall"][
                        "normalized"
                    ]["sum_squared_error"]
                    * 1.1
                }
            ),
            "primitive|recomputation",
        ),
        (
            lambda audit: audit["models"]["scratch"]["slices"]["overall"]["normalized"].update(
                {
                    "maximum_absolute_error": audit["models"]["scratch"]["slices"]["overall"][
                        "normalized"
                    ]["maximum_absolute_error"]
                    * 1.1
                }
            ),
            "primitive|recomputation|maximum-rank",
        ),
        (
            lambda audit: audit["models"]["scratch"]["slices"]["overall"]["normalized"][
                "quantiles"
            ]["p95"].update({"interpolation_fraction": 0.123}),
            "rank primitive",
        ),
        (
            lambda audit: audit["models"]["scratch"]["slices"]["overall"]["normalized"][
                "quantiles"
            ]["p95"].update(
                {
                    "lower_rank": audit["models"]["scratch"]["slices"]["overall"]["normalized"][
                        "quantiles"
                    ]["p95"]["lower_rank"]
                    - 1
                }
            ),
            "rank primitive",
        ),
        (lambda audit: audit.update({"row_count": 2}), "overall count"),
        (
            lambda audit: audit["models"]["scratch"]["diagnostics"]["checks"][
                "spot_monotonicity"
            ].update({"material_violations": 1}),
            "diagnostic.*(primitive|relationship)",
        ),
    ],
)
def test_compact_snapshot_rejects_sufficient_statistic_drift(tmp_path, mutation, match) -> None:
    module = _module()
    validation, final, _, acceptance, _, _ = _reports()
    _install_varied_models(validation, final, acceptance, 12)
    fixture_rows = validation["validation"]["scratch"]["audit_rows"]
    fixture_errors = {abs(row["prediction"] - row["reference_price"]) for row in fixture_rows}
    assert len(fixture_rows) == 12
    assert len(fixture_errors) > 1
    assert min(fixture_errors) > 0.0
    validation_path = tmp_path / "validation.json"
    final_path = tmp_path / "final.json"
    validation_path.write_text(json.dumps(validation))
    final_path.write_text(json.dumps(final))
    snapshot = module.extract_snapshot(
        validation,
        final,
        state="final_complete",
        protocol_path=PROJECT_ROOT / "configs/american_neural_pilot_protocol_v1.toml",
        validation_path=validation_path,
        final_path=final_path,
        acceptance=acceptance,
    )
    overall = snapshot["audit"]["validation"]["models"]["scratch"]["slices"]["overall"][
        "normalized"
    ]
    assert overall["sum_absolute_error"] > 0.0
    assert all(
        primitive["normalized"] is not None
        for primitive in snapshot["audit"]["validation"]["models"]["scratch"]["slices"].values()
    )
    mutation(snapshot["audit"]["validation"])
    compact = {arm: snapshot["arms"][arm]["validation"] for arm in ("scratch", "transfer")}
    compact["european_crr_baseline"] = snapshot["baseline"]["validation"]
    with pytest.raises(module.FreezeError, match=match):
        module._validate_compact_partition_models(
            compact,
            snapshot["audit"]["validation"],
            "snapshot.validation",
            acceptance,
            "validation_final_entry",
        )


def test_quantile_primitives_reject_coincident_and_crossed_order_statistics() -> None:
    module = _module()
    coincident = module._metric_primitive([1.0])
    coincident["quantiles"]["p95"]["lower_value"] = 0.5
    with pytest.raises(module.FreezeError, match=r"coincident|maximum-rank"):
        module._statistics_from_primitive(coincident, "coincident")

    crossed = module._metric_primitive([float(value) for value in range(100)])
    crossed["quantiles"]["p95"]["upper_value"] = 99.0
    with pytest.raises(module.FreezeError, match="cross-quantile"):
        module._statistics_from_primitive(crossed, "crossed")

    contradictory = module._metric_primitive([0.0] * 99 + [1.0])
    for label in ("p95", "p99"):
        contradictory["quantiles"][label]["lower_value"] = 1.0
        contradictory["quantiles"][label]["upper_value"] = 1.0
    with pytest.raises(module.FreezeError, match="stored moments"):
        module._statistics_from_primitive(contradictory, "contradictory")


def test_large_constant_metric_primitive_remains_numerically_consistent() -> None:
    module = _module()
    values = [0.123456789] * 25_000
    primitive = module._metric_primitive(values)

    statistics = module._statistics_from_primitive(primitive, "constant")

    assert statistics["rows"] == 25_000
    assert statistics["mae"] == pytest.approx(0.123456789)
    assert statistics["rmse"] == pytest.approx(0.123456789)
    assert statistics["p95_absolute_error"] == pytest.approx(0.123456789)
    assert statistics["p99_absolute_error"] == pytest.approx(0.123456789)


@pytest.mark.parametrize(
    ("check_name", "related_check"),
    [
        ("american_call_upper_bound", None),
        ("intrinsic_lower_bound", None),
        ("spot_monotonicity", "spot_convexity"),
    ],
)
def test_compact_snapshot_rejects_diagnostic_eligibility_drift(
    tmp_path, check_name, related_check
) -> None:
    module = _module()
    validation, final, _, acceptance, _, _ = _reports()
    validation_path = tmp_path / "validation.json"
    final_path = tmp_path / "final.json"
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    final_path.write_text(json.dumps(final), encoding="utf-8")
    snapshot = module.extract_snapshot(
        validation,
        final,
        state="final_complete",
        protocol_path=PROJECT_ROOT / "configs/american_neural_pilot_protocol_v1.toml",
        validation_path=validation_path,
        final_path=final_path,
        acceptance=acceptance,
    )
    primitive = snapshot["audit"]["validation"]["models"]["scratch"]["diagnostics"]["checks"][
        check_name
    ]
    published = snapshot["arms"]["scratch"]["validation"]["diagnostics"]["checks"][check_name]
    primitive["eligible_rows"] = 0
    published["eligible_rows"] = 0
    if related_check is not None:
        assert (
            snapshot["audit"]["validation"]["models"]["scratch"]["diagnostics"]["checks"][
                related_check
            ]["eligible_rows"]
            == 1
        )
    compact = {arm: snapshot["arms"][arm]["validation"] for arm in ("scratch", "transfer")}
    compact["european_crr_baseline"] = snapshot["baseline"]["validation"]
    with pytest.raises(module.FreezeError, match="eligibility"):
        module._validate_compact_partition_models(
            compact,
            snapshot["audit"]["validation"],
            "snapshot.validation",
            acceptance,
            "validation_final_entry",
        )


def test_validation_passed_can_freeze_without_advancing_to_final(tmp_path) -> None:
    module = _module()
    validation, _, protocol, acceptance, latency, iv = _reports()
    assert (
        module.validate_raw_reports(validation, None, protocol, DIGEST, acceptance, latency, iv)
        == "validation_passed_not_advanced"
    )
    validation_path = tmp_path / "validation.json"
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    snapshot = module.extract_snapshot(
        validation,
        None,
        state="validation_passed_not_advanced",
        protocol_path=PROJECT_ROOT / "configs/american_neural_pilot_protocol_v1.toml",
        validation_path=validation_path,
        final_path=None,
        acceptance=acceptance,
    )
    module.validate_snapshot(
        snapshot,
        PROJECT_ROOT / "configs/american_neural_pilot_protocol_v1.toml",
        protocol,
        acceptance,
        latency,
        iv,
    )
    assert snapshot["lifecycle"]["final_partition_consumed"] is False
    assert snapshot["outcome"] == {
        "phase": "validation",
        "outcome": "validation_passed_not_advanced",
        "final_eligibility_approved": False,
        "neural_result_approved": False,
    }
    inconsistent_snapshots = []
    consumed = copy.deepcopy(snapshot)
    consumed["lifecycle"]["final_partition_consumed"] = True
    inconsistent_snapshots.append(consumed)
    attempted = copy.deepcopy(snapshot)
    attempted["lifecycle"]["final_evaluation_attempts"] = 1
    inconsistent_snapshots.append(attempted)
    final_source = copy.deepcopy(snapshot)
    final_source["source_reports"]["final"] = {"file": "final.json", "sha256": DIGEST}
    inconsistent_snapshots.append(final_source)
    for inconsistent in inconsistent_snapshots:
        with pytest.raises(module.FreezeError, match="lifecycle"):
            module.validate_snapshot(
                inconsistent,
                PROJECT_ROOT / "configs/american_neural_pilot_protocol_v1.toml",
                protocol,
                acceptance,
                latency,
                iv,
            )


@pytest.mark.parametrize("state", ["validation_passed_not_advanced", "final_complete"])
@pytest.mark.parametrize("evidence", ["latency", "iv"])
def test_passing_snapshot_recomputes_required_evidence(tmp_path, state, evidence) -> None:
    module = _module()
    validation, final, protocol, acceptance, latency, iv = _reports()
    validation_path = tmp_path / "validation.json"
    final_path = tmp_path / "final.json"
    validation_path.write_text(json.dumps(validation), encoding="utf-8")
    final_path.write_text(json.dumps(final), encoding="utf-8")
    include_final = state == "final_complete"
    snapshot = module.extract_snapshot(
        validation,
        final if include_final else None,
        state=state,
        protocol_path=PROJECT_ROOT / "configs/american_neural_pilot_protocol_v1.toml",
        validation_path=validation_path,
        final_path=final_path if include_final else None,
        acceptance=acceptance,
    )
    if evidence == "latency":
        snapshot["latency"]["measurements"].pop()
    else:
        inversion = snapshot["implied_volatility"]["rows"][0]["inversions"]["scratch"]
        inversion.update(
            {
                "status": "unbracketed",
                "volatility": None,
                "iterations": 0,
                "absolute_error_vs_label_iv": None,
            }
        )
        snapshot["implied_volatility"]["failures"] = 1
    with pytest.raises(module.FreezeError, match="final-entry gate"):
        module.validate_snapshot(
            snapshot,
            PROJECT_ROOT / "configs/american_neural_pilot_protocol_v1.toml",
            protocol,
            acceptance,
            latency,
            iv,
        )


def test_realistic_partition_snapshot_stays_below_two_mib(tmp_path) -> None:
    module = _module()
    validation, final, protocol, acceptance, latency, iv = _reports()
    row_count = 25_000
    _install_varied_models(validation, final, acceptance, row_count)
    validation_path = tmp_path / "validation.json"
    final_path = tmp_path / "final.json"
    validation_path.write_text("fixture", encoding="utf-8")
    final_path.write_text("fixture", encoding="utf-8")
    snapshot = module.extract_snapshot(
        validation,
        final,
        state="final_complete",
        protocol_path=PROJECT_ROOT / "configs/american_neural_pilot_protocol_v1.toml",
        validation_path=validation_path,
        final_path=final_path,
        acceptance=acceptance,
    )
    module.validate_snapshot(
        snapshot,
        PROJECT_ROOT / "configs/american_neural_pilot_protocol_v1.toml",
        protocol,
        acceptance,
        latency,
        iv,
    )
    expected_slices = module._slice_names(acceptance)
    for partition in ("validation", "final"):
        for model in module.MODEL_NAMES:
            slices = snapshot["audit"][partition]["models"][model]["slices"]
            assert set(slices) == expected_slices
            for primitive in slices.values():
                assert primitive["physical"] is not None
                assert primitive["normalized"] is not None
    snapshot_size = len(module.serialise(snapshot).encode("utf-8"))
    assert snapshot_size < 2 * 1024 * 1024
