"""Lightweight task 9C-B contract tests; the expensive pilot is never run here."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import tomllib
from pathlib import Path

import pytest
from differentiable_pricing.american import pde_label_policy as pilot_module
from differentiable_pricing.american.pde_label_policy import (
    BumpRequest,
    BumpSettings,
    Grid,
    LabelPolicyError,
    PilotCase,
    SolverSettings,
    _canonical_json,
    _policy_outcomes,
    _price_one,
    _runtime_csv,
    centered_delta,
    centered_gamma,
    centered_vega,
    labels_from_prices,
    load_pilot_config,
    main,
    observed_order,
    parse_pilot_config,
    richardson_extrapolate,
    select_accuracy_policy,
    threshold_pass,
    vega_per_volatility_point,
    verify_byte_identity,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs" / "pde_label_policy_pilot_v1.toml"


def _document() -> dict[str, object]:
    with CONFIG.open("rb") as stream:
        return tomllib.load(stream)


def test_checked_in_pilot_design_is_valid_versioned_and_separated() -> None:
    config = load_pilot_config(CONFIG)
    assert config.name == "pde-label-policy-pilot-v1"
    assert config.source_sha256 == hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    assert len(config.cases) == 28
    assert sum(case.classification == "regular" for case in config.cases) == 22
    assert sum(case.classification == "stress" for case in config.cases) == 6
    assert {case.option_type for case in config.cases} == {"call", "put"}
    assert {case.exercise_style for case in config.cases} == {"european", "american"}
    assert min(case.spot / case.strike for case in config.cases) == pytest.approx(0.6)
    assert max(case.spot / case.strike for case in config.cases) == pytest.approx(1.4)
    assert tuple((grid.spot_intervals, grid.time_steps) for grid in config.grids) == (
        (800, 400),
        (1600, 800),
        (3200, 1600),
        (6400, 3200),
    )
    assert config.criteria.price_absolute_error == 5.0e-4
    assert config.criteria.delta_absolute_error == 1.0e-3
    assert config.criteria.gamma_absolute_error == 2.0e-4
    assert config.criteria.vega_absolute_error_per_unit_volatility == 5.0e-2


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda document: document.update({"unknown": 1}), "unknown keys"),
        (
            lambda document: document["criteria"].update(
                {"price_absolute_error": 5.1e-4}
            ),
            "thresholds changed",
        ),
        (
            lambda document: document["richardson"].update(
                {"assumed_order": 1.0}
            ),
            "second-order",
        ),
        (
            lambda document: document["cases"][0].update(
                {"classification": "descriptive"}
            ),
            "regular or stress",
        ),
        (
            lambda document: document["bumps"].update(
                {"primary_spot": 0.25}
            ),
            "must be members",
        ),
        (
            lambda document: document["cases"][0].update({"volatility": 0.01}),
            "crosses sigma <= 0",
        ),
    ],
)
def test_mutated_protocol_is_rejected(mutation, message: str) -> None:
    document = copy.deepcopy(_document())
    mutation(document)
    with pytest.raises(LabelPolicyError, match=message):
        parse_pilot_config(document, source_name="mutated.toml", source_sha256="0" * 64)


def test_centered_formulas_have_the_declared_signs_denominators_and_units() -> None:
    # V(S) = 2 + 3S + 4S^2 at S=5 has delta 43 and gamma 8.
    center = 117.0
    up = 164.0
    down = 78.0
    assert centered_delta(up, down, 1.0) == 43.0
    assert centered_delta(down, up, 1.0) == -43.0
    assert centered_gamma(up, center, down, 1.0) == 8.0
    assert centered_gamma(up, center + 1.0, down, 1.0) == 6.0
    assert centered_vega(10.8, 9.2, 0.02) == pytest.approx(40.0)
    assert vega_per_volatility_point(40.0) == pytest.approx(0.4)


def test_richardson_sign_order_and_observed_order_are_mutation_sensitive() -> None:
    # Errors 4, 1 and 1/4 have observed order two. The correction must extend
    # from coarse toward fine, not reverse its sign or exchange the pair.
    assert observed_order(14.0, 11.0, 10.25) == pytest.approx(2.0)
    assert richardson_extrapolate(14.0, 11.0) == pytest.approx(10.0)
    assert richardson_extrapolate(11.0, 14.0) == pytest.approx(15.0)
    assert observed_order(10.0, 10.0, 10.0) is None


def test_label_ladders_share_price_bumps_and_report_both_vega_units() -> None:
    bumps = BumpSettings(
        spot=(2.0, 1.0),
        primary_spot=1.0,
        volatility=(0.02, 0.01),
        primary_volatility=0.01,
    )
    prices = {
        "center": 10.0,
        "spot_2_down": 8.4,
        "spot_2_up": 12.4,
        "spot_1_down": 9.1,
        "spot_1_up": 11.1,
        "volatility_0.02_down": 9.2,
        "volatility_0.02_up": 10.8,
        "volatility_0.01_down": 9.6,
        "volatility_0.01_up": 10.4,
    }
    labels = labels_from_prices(prices, bumps)
    assert labels["delta_by_spot_bump"] == {"2": 1.0, "1": 1.0}
    assert labels["gamma_by_spot_bump"]["1"] == pytest.approx(0.2)
    assert labels["vega_per_unit_by_volatility_bump"]["0.01"] == pytest.approx(40.0)
    assert labels["vega_per_point_by_volatility_bump"]["0.01"] == pytest.approx(0.4)


def test_thresholds_are_inclusive_and_no_policy_is_manufactured() -> None:
    assert threshold_pass(5.0e-4, 5.0e-4)
    assert not threshold_pass(math.nextafter(5.0e-4, math.inf), 5.0e-4)
    outcomes = {
        "coarse": {"all_regular_cases_pass": False},
        "fine": {"all_regular_cases_pass": True},
    }
    assert select_accuracy_policy(("coarse", "fine"), outcomes) == "fine"
    outcomes["fine"]["all_regular_cases_pass"] = False
    assert select_accuracy_policy(("coarse", "fine"), outcomes) == "no_policy_selected"


def test_stress_failures_are_retained_but_cannot_reject_a_regular_policy() -> None:
    config = load_pilot_config(CONFIG)
    cases = [
        {
            "state": {"name": "regular", "classification": "regular"},
            "candidate_outcomes": {
                policy: {"passes": True} for policy in config.selection_order
            },
        },
        {
            "state": {"name": "stress", "classification": "stress"},
            "candidate_outcomes": {
                policy: {"passes": False} for policy in config.selection_order
            },
        },
    ]
    outcomes = _policy_outcomes(cases, config)
    for outcome in outcomes.values():
        assert outcome["all_regular_cases_pass"] is True
        assert outcome["regular_failures"] == []
        assert outcome["stress_failures_descriptive_only"] == ["stress"]


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_small_synthetic_pde_case_produces_finite_centered_labels(option_type: str) -> None:
    case = PilotCase(
        name=f"small_{option_type}",
        classification="regular",
        description="CI-sized synthetic control",
        option_type=option_type,
        exercise_style="european",
        spot=100.0,
        strike=100.0,
        expiry_time=0.5,
        rate=0.03,
        continuous_carry=0.0,
        volatility=0.2,
        dividends=(),
    )
    grid = Grid("ci_80x40", 80, 40, "candidate")
    solver = SolverSettings(400.0, 2, 1.0e-11, 1.2, 10_000, "cash", 100.0)
    bumps = BumpSettings((1.0,), 1.0, (0.01,), 0.01)
    requests = (
        BumpRequest("center", 100.0, 0.2, "center", None, 0),
        BumpRequest("spot_1_down", 99.0, 0.2, "spot", 1.0, -1),
        BumpRequest("spot_1_up", 101.0, 0.2, "spot", 1.0, 1),
        BumpRequest("volatility_0.01_down", 100.0, 0.19, "volatility", 0.01, -1),
        BumpRequest("volatility_0.01_up", 100.0, 0.21, "volatility", 0.01, 1),
    )
    prices = {
        request.name: float(_price_one(case, request, grid, solver, "european")[0]["price"])
        for request in requests
    }
    labels = labels_from_prices(prices, bumps)
    assert all(
        math.isfinite(value)
        for value in (
            labels["price"],
            labels["delta_by_spot_bump"]["1"],
            labels["gamma_by_spot_bump"]["1"],
            labels["vega_per_unit_by_volatility_bump"]["0.01"],
        )
    )
    assert labels["gamma_by_spot_bump"]["1"] >= 0.0
    assert labels["vega_per_unit_by_volatility_bump"]["0.01"] > 0.0


def test_canonical_regeneration_and_identity_check_detect_a_mutation(tmp_path: Path) -> None:
    payload = {"schema_version": "test/1", "regular": [1.0], "stress": [2.0]}
    assert _canonical_json(payload) == _canonical_json(copy.deepcopy(payload))
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    names = ("report.json", "labels.csv", "sensitivity.csv", "failures.csv", "runtime.csv")
    for name in names:
        content = _canonical_json(payload) if name == "report.json" else "a,b\n1,2\n"
        (first / name).write_text(content, encoding="utf-8")
        shutil.copyfile(first / name, second / name)
    verify_byte_identity(first, second)
    (second / "labels.csv").write_text("a,b\n1,3\n", encoding="utf-8")
    with pytest.raises(LabelPolicyError, match=r"labels\.csv"):
        verify_byte_identity(first, second)


def test_runtime_csv_uses_selection_order_after_json_key_sorting() -> None:
    policies = ["grid_800x400", "grid_1600x800", "richardson_800x400_1600x800"]
    projection = {
        "measured_mean_seconds_per_four-label_state": 1.0,
        "serial_hours_for_250000_labels": 2.0,
        "idealized_8_worker_hours": 3.0,
        "idealized_16_worker_hours": 4.0,
    }
    report = {
        "recommendation": {
            "selected_accuracy_policy": "no_policy_selected",
            "selection_order": policies,
        },
        # Deliberately use the order produced by sorted-key JSON loading.
        "performance": {
            "projections": {
                "grid_1600x800": projection,
                "grid_800x400": projection,
                "richardson_800x400_1600x800": projection,
            }
        },
    }
    rows = _runtime_csv(report).splitlines()
    assert [row.split(",", 1)[0] for row in rows[1:]] == policies


def test_cli_preflights_nonempty_output_without_running_expensive_pilot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")

    def unexpected_run(_config):
        raise AssertionError("expensive pilot must not run in CI preflight")

    monkeypatch.setattr(pilot_module, "run_pilot", unexpected_run)
    assert main(["--config", str(CONFIG), "--output-directory", str(output)]) == 2
    assert json.loads(json.dumps({"unchanged": True})) == {"unchanged": True}
