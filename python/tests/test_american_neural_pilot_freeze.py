"""Synthetic-fixture tests for strict Task 9G report freezing."""

from __future__ import annotations

import copy
import importlib.util
import json
import math
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
        "diagnostic_violations": (
            None if baseline else {name: -1.0 for name in _module().DIAGNOSTIC_CHECKS}
        ),
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
                        "eligible_rows": 1,
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
            "recomputation",
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
        "interpretation": (
            "terminal consumed final-attempt failure; no retry under this protocol"
        ),
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
        module.validate_raw_reports(
            validation, failure, protocol, DIGEST, acceptance, latency, iv
        )


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
    assert len(snapshot["audit"]["validation"]["common_rows"]) == 1
    assert "audit_rows" not in snapshot["arms"]["scratch"]["validation"]
    assert "audit_rows" not in snapshot["arms"]["transfer"]["validation"]
    assert "audit_rows" not in snapshot["baseline"]["validation"]
    assert set(snapshot["audit"]["validation"]["physical_errors"]) == {
        "scratch",
        "transfer",
        "european_crr_baseline",
    }
    common = snapshot["audit"]["validation"]["common_rows"][0]
    assert set(common) == {
        "slice_memberships",
        "normalization_scale",
        "early_exercise_premium",
    }


def test_compact_snapshot_rejects_premium_membership_drift(tmp_path) -> None:
    module = _module()
    validation, final, _, acceptance, _, _ = _reports()
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
    row = snapshot["audit"]["validation"]["common_rows"][0]
    row["slice_memberships"].remove("premium_status:zero")
    row["slice_memberships"].append("premium_status:positive")
    compact = {arm: snapshot["arms"][arm]["validation"] for arm in ("scratch", "transfer")}
    compact["european_crr_baseline"] = snapshot["baseline"]["validation"]
    with pytest.raises(module.FreezeError, match="premium membership"):
        module._validate_compact_partition_models(
            compact,
            snapshot["audit"]["validation"],
            "snapshot.validation",
            acceptance,
            "validation_final_entry",
        )
