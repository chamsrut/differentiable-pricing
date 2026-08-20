"""Cheap mocked numerical-contract regressions for Task 9G."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from differentiable_pricing.ml import american_pilot as pilot


def _case(**overrides) -> dict:
    value = {
        "name": "case",
        "option_type": "call",
        "spot": 100.0,
        "strike": 100.0,
        "maturity": 1.0,
        "rate": 0.03,
        "dividend_yield": 0.02,
        "volatility": 0.2,
    }
    value.update(overrides)
    return value


def _iv_config() -> dict:
    return {
        "volatility_bracket": [0.05, 0.8],
        "crr_steps": 1024,
        "price_absolute_tolerance_over_spot": 1.0e-10,
        "volatility_absolute_tolerance": 1.0e-8,
        "maximum_iterations": 80,
        "terminology": "surface",
        "cases": [{**_case(), "true_volatility": 0.2}],
    }


def test_crr_adjacent_average_extracts_prices_at_n_and_n_plus_one(monkeypatch) -> None:
    calls = []

    def price(*arguments):
        calls.append(arguments[-1])
        return {"price": float(arguments[-1])}

    monkeypatch.setattr(pilot._core, "crr_price", price)
    assert pilot.crr_adjacent_average(_case(), 256) == 256.5
    assert calls == [256, 257]


def test_iv_bisection_reports_nonfinite_unbracketed_and_exhaustion(monkeypatch) -> None:
    config = _iv_config()
    monkeypatch.setattr(
        pilot,
        "crr_adjacent_average",
        lambda case, steps, volatility=None: 100.0 * float(volatility),
    )
    assert pilot._bisect_iv(_case(), 20.0, config)["status"] == "converged"
    assert pilot._bisect_iv(_case(), 1000.0, config)["status"] == "unbracketed"
    monkeypatch.setattr(pilot, "crr_adjacent_average", lambda *args, **kwargs: float("nan"))
    assert pilot._bisect_iv(_case(), 20.0, config)["status"] == "non_finite_price"
    calls = 0

    def late_nan(case, steps, volatility=None):
        nonlocal calls
        calls += 1
        return float("nan") if calls == 3 else 100.0 * float(volatility)

    monkeypatch.setattr(pilot, "crr_adjacent_average", late_nan)
    assert pilot._bisect_iv(_case(), 20.0, config)["status"] == "non_finite_price"
    monkeypatch.setattr(
        pilot,
        "crr_adjacent_average",
        lambda case, steps, volatility=None: 100.0 * float(volatility),
    )
    exhausted = {**config, "maximum_iterations": 1, "volatility_absolute_tolerance": 0.0}
    assert pilot._bisect_iv(_case(), 20.123456, exhausted)["status"] == "iteration_exhausted"


def test_run_implied_volatility_uses_one_solver_for_label_and_models(monkeypatch) -> None:
    config = _iv_config()
    monkeypatch.setattr(
        pilot,
        "crr_adjacent_average",
        lambda case, steps, volatility=None: (
            100.0 * float(case["volatility"] if volatility is None else volatility)
        ),
    )
    monkeypatch.setattr(pilot, "predict_prices", lambda model, physical, batch: np.asarray([20.0]))
    result = pilot.run_implied_volatility({"scratch": object(), "transfer": object()}, config)
    assert result["failures"] == 0
    assert set(result["rows"][0]["inversions"]) == {"label", "scratch", "transfer"}


def test_linear_quantile_method_is_explicit_at_noninteger_rank() -> None:
    result = pilot.error_statistics(
        np.zeros(4), np.asarray([0.0, 1.0, 2.0, 10.0]), quantile_method="linear"
    )
    assert result["p95_absolute_error"] == pytest.approx(8.8)


class _ConstantModel(torch.nn.Module):
    def forward(self, physical: torch.Tensor) -> torch.Tensor:
        return torch.full((physical.shape[0],), 200.0, dtype=torch.float64)


def test_shape_diagnostics_separates_bounds_and_excludes_moneyness_boundary() -> None:
    strike_boundary = 100.0 / math.exp(0.699)
    columns = {
        "sample_id": np.asarray(["call", "put", "boundary"], dtype=object),
        "option_type": np.asarray(["call", "put", "call"], dtype=object),
        "spot": np.asarray([100.0, 100.0, 100.0]),
        "strike": np.asarray([100.0, 100.0, strike_boundary]),
        "maturity": np.asarray([1.0, 1.0, 1.0]),
        "rate": np.asarray([0.03, -0.02, 0.03]),
        "dividend_yield": np.asarray([0.0, 0.0, 0.0]),
        "volatility": np.asarray([0.2, 0.2, 0.2]),
        "european_crr_price": np.asarray([10.0, 10.0, 10.0]),
    }
    acceptance = {
        "diagnostics": {
            "material_normalized_tolerance": 1.0e-6,
            "spot_bump_fraction": 0.01,
            "volatility_bump": 0.01,
            "spot_domain": [50.0, 150.0],
            "volatility_domain": [0.05, 0.8],
            "log_moneyness_domain": [-0.7, 0.7],
        }
    }
    summary, audit = pilot.shape_diagnostics(_ConstantModel(), columns, acceptance, batch_size=3)
    assert summary["checks"]["american_call_upper_bound"]["eligible_rows"] == 2
    assert summary["checks"]["american_put_rate_aware_upper_bound"]["eligible_rows"] == 1
    assert audit["spot_monotonicity"][2] is None


def _pde_result(request: dict) -> dict:
    target_step = request["spot_maximum"] / request["spot_intervals"]
    strike_index = max(1, math.floor(request["strike"] / target_step + 0.5))
    step = request["strike"] / strike_index
    actual_intervals = max(
        strike_index + 1,
        math.ceil(request["spot_maximum"] / step - 1.0e-9),
    )
    return {
        "price": 10.0,
        "solver_status": "discrete_system_converged",
        "discretization_accuracy": "not_assessed",
        "spot_intervals": actual_intervals,
        "spot_maximum": actual_intervals * step,
        "spot_step": step,
        "strike_node_index": strike_index,
        "time_steps": request["time_steps"],
        "rannacher_steps": request["rannacher_steps"],
        "damped_half_steps": 4,
        "crank_nicolson_steps": request["time_steps"] - 2,
        "upwinded_rows": 0,
        "linear_solves": 0,
        "psor_solves": request["time_steps"] + 2,
        "psor_total_iterations": request["time_steps"] + 2,
        "psor_maximum_iterations_used": 1,
        "psor_tolerance": request["psor_tolerance"],
        "psor_relaxation": request["psor_relaxation"],
        "maximum_lcp_residual": 0.0,
        "maximum_relative_lcp_residual": 0.0,
        "aligned_times": [0.0, request["expiry_time"]],
        "dividend_events": [],
    }


def test_pde_check_records_effective_diagnostics_and_rejects_status(monkeypatch, tmp_path) -> None:
    inputs = {
        "sample_id": np.asarray(["id"], dtype=object),
        "stratum": np.asarray(["core"], dtype=object),
        "option_type": np.asarray(["call"], dtype=object),
        "spot": np.asarray([100.0]),
        "strike": np.asarray([100.0]),
        "maturity": np.asarray([1.0]),
        "rate": np.asarray([0.03]),
        "dividend_yield": np.asarray([0.02]),
        "volatility": np.asarray([0.2]),
    }
    monkeypatch.setattr(
        pilot,
        "read_partition_columns",
        lambda dataset, manifest, split, columns, verify_digest=False: (
            {"sample_id": inputs["sample_id"], "american_price": np.asarray([10.0])}
            if "american_price" in columns
            else inputs
        ),
    )
    protocol = {
        "independent_pde_check": {
            "selection_salt": "salt",
            "cases": [
                {
                    "sample_id": "id",
                    "stratum": "core",
                    "option_type": "call",
                    "spot": 100.0,
                    "strike": 100.0,
                    "maturity": 1.0,
                    "rate": 0.03,
                    "dividend_yield": 0.02,
                    "volatility": 0.2,
                }
            ],
            "grids": [
                {
                    "name": "coarse",
                    "spot_intervals": 10,
                    "time_steps": 5,
                    "spot_maximum_factor": 4.0,
                },
                {
                    "name": "fine",
                    "spot_intervals": 20,
                    "time_steps": 10,
                    "spot_maximum_factor": 4.0,
                },
            ],
            "valuation_time": 0.0,
            "settlement": "cash",
            "contract_multiplier": 1.0,
            "rannacher_steps": 2,
            "psor_tolerance": 1.0e-11,
            "psor_relaxation": 1.2,
            "psor_maximum_iterations": 100,
            "maximum_normalized_refinement_difference": 0.001,
            "maximum_normalized_pde_crr_difference": 0.001,
            "interpretation": "mapping-consistency only",
        }
    }
    monkeypatch.setattr(pilot, "pde_price", lambda **request: _pde_result(request))
    result = pilot.run_independent_pde_check(tmp_path, {"files": []}, protocol)
    assert result["passed"] and len(result["rows"][0]["solves"]) == 2

    def failed(**request):
        value = _pde_result(request)
        value["solver_status"] = "psor_iteration_limit_exceeded"
        return value

    monkeypatch.setattr(pilot, "pde_price", failed)
    with pytest.raises(pilot.PilotError, match="did not converge"):
        pilot.run_independent_pde_check(tmp_path, {"files": []}, protocol)

    def impossible_grid(**request):
        value = _pde_result(request)
        value["strike_node_index"] = value["spot_intervals"]
        value["spot_step"] = value["strike"] if "strike" in value else request["strike"]
        value["spot_maximum"] = value["spot_intervals"] * value["spot_step"]
        return value

    monkeypatch.setattr(pilot, "pde_price", impossible_grid)
    with pytest.raises(pilot.PilotError, match="spatial grid"):
        pilot.run_independent_pde_check(tmp_path, {"files": []}, protocol)

    def nonfinite_price(**request):
        value = _pde_result(request)
        value["price"] = float("nan")
        return value

    monkeypatch.setattr(pilot, "pde_price", nonfinite_price)
    with pytest.raises(pilot.PilotError, match="non-finite or negative price"):
        pilot.run_independent_pde_check(tmp_path, {"files": []}, protocol)


def test_native_pde_exception_records_entry_failure_before_optimization(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "protocol.toml").write_text("schema_version = 'fixture'\n", encoding="utf-8")
    output = tmp_path / "output"
    protocol = {
        "schema_version": "fixture",
        "paths": {
            "dataset": "dataset",
            "training_config": "training.toml",
            "acceptance_config": "acceptance.toml",
            "latency_config": "latency.toml",
            "iv_config": "iv.toml",
            "protocol": "protocol.toml",
            "source_artifact": "source",
        },
        "dataset": {"identity": "fixture"},
        "source_artifact": {"identity": "fixture"},
    }
    training = {"training": {"num_threads": 1}}
    latency = {"torch_interop_threads": 1}
    monkeypatch.setattr(
        pilot,
        "load_toml",
        lambda path: latency if path.name == "latency.toml" else training,
    )
    monkeypatch.setattr(pilot.torch, "set_num_interop_threads", lambda value: None)
    monkeypatch.setattr(pilot.torch, "set_num_threads", lambda value: None)
    monkeypatch.setattr(pilot, "verify_partition_policy", lambda *args: None)
    monkeypatch.setattr(
        pilot,
        "run_independent_pde_check",
        lambda *args: (_ for _ in ()).throw(RuntimeError("native PSOR failure")),
    )
    optimized = False

    def forbidden_fit(*args, **kwargs):
        nonlocal optimized
        optimized = True

    monkeypatch.setattr(pilot, "fit_price_model", forbidden_fit)
    monkeypatch.setattr(pilot, "_runtime_metadata", lambda: _runtime_stub())
    with pytest.raises(pilot.PilotError, match="native PSOR failure"):
        pilot.execute_validation_run(
            project_root=tmp_path,
            protocol=protocol,
            manifest={},
            output_directory=output,
        )
    assert not optimized
    report = json.loads((output / "entry-failure-report.json").read_text(encoding="utf-8"))
    assert report["failed_gate"] == "independent_pde_check"
    assert report["evidence"] == {
        "passed": False,
        "error_type": "RuntimeError",
        "message": "native PSOR failure",
    }


def test_latency_uses_pinned_identical_requests_and_rotates(monkeypatch) -> None:
    cases = [_case(name="a"), _case(name="b", option_type="put")]
    config = {
        "torch_interop_threads": 1,
        "cases": cases,
        "request_shapes": ["single", "batch2"],
        "batch_sizes": [1, 2],
        "thread_budgets": [1, 2],
        "request_case_names": {"single": ["a"], "batch2": ["a", "b"]},
        "crr_depths": [256],
        "warmups": 0,
        "repetitions": 3,
    }
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)
    monkeypatch.setattr(torch, "set_num_threads", lambda value: None)
    crr_calls = []

    def batch(*arguments):
        crr_calls.append((list(arguments[0]), list(arguments[8])))
        return {"price": [1.0] * len(arguments[0])}

    monkeypatch.setattr(pilot._core, "crr_price_batch", batch)
    monkeypatch.setattr(pilot, "predict_prices", lambda model, physical, batch: np.ones(batch))
    ticks = iter(range(10_000))
    monkeypatch.setattr(pilot.time, "perf_counter_ns", lambda: next(ticks))
    monkeypatch.setattr(pilot, "_runtime_metadata", _runtime_stub)
    result = pilot.run_latency({"scratch": object(), "transfer": object()}, config)
    assert result["measurements"][0]["observed_measurement_orders"] == [
        ["crr", "scratch", "transfer"],
        ["scratch", "transfer", "crr"],
        ["transfer", "crr", "scratch"],
    ]
    assert result["measurements"][1]["economic_requests"] == cases
    assert (["call", "put"], [256, 256]) in crr_calls
    assert (["call", "put"], [257, 257]) in crr_calls


def _runtime_stub() -> dict:
    return {}


def test_outcome_precedence_mixed_tie_and_required_evidence() -> None:
    def model(passed: bool, rmse: float) -> dict:
        return {
            "gate": {"passed": passed},
            "slices": {"overall": {"normalized": {"rmse": rmse}}},
        }

    latency = {
        "measurements": [
            {
                "shape": "single",
                "crr_depth": 1024,
                "median_speedup": {"scratch": 20.0, "transfer": 20.0},
            }
        ]
    }
    iv = {
        "rows": [
            {
                "inversions": {
                    arm: {"absolute_error_vs_label_iv": 0.0} for arm in ("scratch", "transfer")
                }
            }
        ]
    }
    acceptance = {
        "latency": {
            "reference_depth_for_interpretation": 1024,
            "minimum_median_end_to_end_speedup": 10.0,
        },
        "implied_volatility": {"maximum_absolute_volatility_error": 0.01},
    }
    mixed = {"scratch": model(True, 0.001), "transfer": model(False, 0.002)}
    assert (
        pilot.classify_outcome(
            mixed, mixed, latency, iv, acceptance, phase="final", evidence_complete=True
        )["outcome"]
        == "mixed_accuracy_outcome"
    )
    tied = {"scratch": model(True, 0.001), "transfer": model(True, 0.001)}
    assert (
        pilot.classify_outcome(
            tied, tied, latency, iv, acceptance, phase="final", evidence_complete=True
        )["outcome"]
        == "negative_transfer"
    )
    assert (
        pilot.classify_outcome(
            tied, tied, latency, iv, acceptance, phase="validation", evidence_complete=False
        )["outcome"]
        == "required_evidence_failed"
    )
