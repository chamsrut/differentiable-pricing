"""Tests for the deterministic LSM/CRR cross-check protocol."""

from __future__ import annotations

import copy
import hashlib
import json
import tomllib
from dataclasses import replace
from pathlib import Path

import differentiable_pricing.american.lsm_crosscheck as lsm_crosscheck_module
import pytest
from differentiable_pricing.american.lsm_crosscheck import (
    LsmCrosscheckError,
    LsmExperiment,
    load_lsm_crosscheck_config,
    main,
    parse_lsm_crosscheck_config,
    run_lsm_crosscheck,
    write_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs" / "american_lsm_crosscheck_v1.toml"
CASE_SOURCE = PROJECT_ROOT / "configs" / "american_crr_convergence_v1.toml"


def _document() -> dict[str, object]:
    with CONFIG.open("rb") as stream:
        return tomllib.load(stream)


def _small_config():
    config = load_lsm_crosscheck_config(CONFIG)
    experiment = LsmExperiment(
        name="small-primary-v1",
        role="primary",
        exercise_steps=8,
        training_paths=1024,
        valuation_paths=2048,
        polynomial_degree=2,
    )
    reference = replace(experiment, name="small-reference-v1", role="reference")
    return replace(
        config,
        crr_reference_steps=32,
        crr_batch_threads=2,
        experiments=(reference, experiment),
        cases=config.cases[:2],
        case_names=config.case_names[:2],
        maximum_training_memory_bytes=2 * 1024 * 1024,
        source_name="small.toml",
        source_sha256="0" * 64,
    )


def test_checked_in_lsm_crosscheck_is_valid_and_pinned() -> None:
    config = load_lsm_crosscheck_config(CONFIG)
    assert config.name == "american-lsm-crosscheck-v1"
    assert config.crr_reference_steps == 8192
    assert len(config.cases) == 9
    assert len(config.experiments) == 7
    assert sum(experiment.role == "primary" for experiment in config.experiments) == 1
    assert config.case_source_sha256 == hashlib.sha256(CASE_SOURCE.read_bytes()).hexdigest()
    assert config.source_sha256 == hashlib.sha256(CONFIG.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda document: document.update({"unknown": 1}), "unknown keys"),
        (
            lambda document: document["study"].update({"engine": "python/lsm"}),
            "engine",
        ),
        (
            lambda document: document["study"].update(
                {"case_source_sha256": "0" * 64}
            ),
            "does not match",
        ),
        (
            lambda document: document["study"].update(
                {"case_names": ["not-a-case"]}
            ),
            "unknown cases",
        ),
        (
            lambda document: document["experiments"][0].update(
                {"training_paths": 3}
            ),
            "even and at least four",
        ),
        (
            lambda document: document["experiments"][0].update(
                {"polynomial_degree": 4}
            ),
            "degree",
        ),
        (
            lambda document: [
                experiment.update({"role": "reference"})
                for experiment in document["experiments"]
                if experiment["role"] == "primary"
            ],
            "exactly one",
        ),
        (
            lambda document: document["study"].update(
                {
                    # This exceeds the primary path matrix alone but remains
                    # below the C++ engine's complete bulk-storage estimate.
                    "maximum_training_memory_bytes": 35 * 1024 * 1024
                }
            ),
            "exceeding",
        ),
    ],
)
def test_invalid_lsm_crosscheck_configurations_are_rejected(
    mutation, message: str
) -> None:
    document = copy.deepcopy(_document())
    mutation(document)
    with pytest.raises(LsmCrosscheckError, match=message):
        parse_lsm_crosscheck_config(
            document,
            source_name="bad.toml",
            source_sha256="0" * 64,
            project_root=PROJECT_ROOT,
        )


def test_small_crosscheck_is_complete_and_deterministic() -> None:
    config = _small_config()
    first = run_lsm_crosscheck(config)
    second = run_lsm_crosscheck(config)
    assert first == second
    assert first["schema_version"] == "american-lsm-crosscheck-report/1"
    assert len(first["experiments"]) == 2
    for experiment in first["experiments"]:
        assert len(experiment["cases"]) == 2
        assert experiment["summary"]["cases"] == 2
        for row in experiment["cases"]:
            assert row["seeds"]["training"] != row["seeds"]["valuation"]
            assert row["lsm"]["independent_valuation_pairs"] in {512, 1024}
            assert row["lsm"]["regression_summary"]["exercise_dates_fitted"] == 7
            assert row["crr_reference"]["steps"] == 32
            assert row["lsm"]["standard_error"] >= 0.0
    assert first["source"]["rng"] == "splitmix64_box_muller_antithetic_v1"
    assert len(first["source"]["lsm_implementation_sha256"]) == 64
    assert "not policy or exercise-grid bias" in first["interpretation"]["lower_bound"]


def test_immediate_exercise_cannot_contaminate_summaries() -> None:
    """A deterministic case must not set or lower the reported minimum."""
    config = _small_config()
    report = run_lsm_crosscheck(config)
    for experiment in report["experiments"]:
        rows = [row["lsm"] for row in experiment["cases"]]
        deterministic = [
            row for row in rows if not row["variance_reduction_applicable"]
        ]
        assert deterministic, "the small study must contain an immediate-exercise case"
        summary = experiment["summary"]

        for row in deterministic:
            # Undefined 0/0 is never surfaced as a number.
            assert row["variance_reduction_ratio"] is None
            assert row["variance_reduction_is_infinite"] is False
            # No valuation simulation ran, so no Monte Carlo value is claimed.
            assert row["european_monte_carlo_sampled"] is False
            assert row["european_monte_carlo_price"] is None
            assert row["european_standard_error"] is None
            assert row["european_analytic_price"] is not None
            assert row["valuation_interval_is_zero_width"] is True

        applicable_finite = [
            float(row["variance_reduction_ratio"])
            for row in rows
            if row["variance_reduction_applicable"]
            and row["variance_reduction_ratio"] is not None
        ]
        assert summary["minimum_variance_reduction_ratio"] == (
            min(applicable_finite) if applicable_finite else None
        )
        assert summary["variance_reduction_applicable_cases"] == len(rows) - len(
            deterministic
        )
        # The deterministic placeholder was exactly 1.0; the reported minimum
        # must now come from a genuinely stochastic case.
        if applicable_finite:
            assert summary["minimum_variance_reduction_ratio"] == pytest.approx(
                min(applicable_finite)
            )

        # Zero-width cases are excluded from the stochastic containment count
        # and reported separately instead.
        stochastic = [
            row for row in rows if not row["valuation_interval_is_zero_width"]
        ]
        assert summary["stochastic_valuation_cases"] == len(stochastic)
        assert summary["deterministic_zero_width_valuation_cases"] == len(
            rows
        ) - len(stochastic)
        assert summary[
            "crr_reference_inside_stochastic_valuation_interval_cases"
        ] <= len(stochastic)
        assert summary[
            "crr_reference_inside_stochastic_valuation_interval_cases"
        ] == sum(
            bool(row["crr_reference_inside_lsm_valuation_interval"])
            for row in stochastic
        )


def test_valuation_interval_semantics_are_documented_in_the_report() -> None:
    """`0/N` must never be readable as a coverage or acceptance statistic."""
    report = run_lsm_crosscheck(_small_config())
    interpretation = report["interpretation"]
    assert "valuation-only" in interpretation["valuation_interval"]
    assert "acceptance" in interpretation["valuation_interval"]
    assert "no nominal coverage rate" in interpretation["valuation_interval"]
    assert "expected behaviour, not a defect" in interpretation["lower_bound"]
    assert "undefined 0/0" in interpretation["variance_reduction"]
    assert "finite by" in interpretation["non_finite_rejection"]

    for experiment in report["experiments"]:
        notes = experiment["summary"]["notes"]
        # The caveat must travel with the number, not only live in the docs.
        assert set(notes) == {
            "crr_reference_inside_stochastic_valuation_interval_cases",
            "deterministic_zero_width_valuation_cases",
            "minimum_variance_reduction_ratio",
        }
        containment = notes[
            "crr_reference_inside_stochastic_valuation_interval_cases"
        ]
        assert "NOT a coverage or calibration statistic" in containment
        assert "must not be used as an acceptance gate" in containment
        assert "arithmetically" in notes["deterministic_zero_width_valuation_cases"]
        assert "excluded" in notes["minimum_variance_reduction_ratio"]
        # The old ambiguous field names must not reappear.
        assert (
            "crr_reference_inside_confidence_interval_cases"
            not in experiment["summary"]
        )
        for row in experiment["cases"]:
            assert (
                "crr_reference_inside_lsm_confidence_interval" not in row["lsm"]
            )


def test_pricing_failures_identify_the_failing_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A production study must report exactly which row failed.

    The engine's price-domain overflow guard is exercised directly in
    ``test_american_lsm_binding``. Every input extreme enough to trip it is also
    rejected by the CRR reference batch that runs first, so the orchestration
    contract is checked by raising the same error from the pricing boundary.
    """
    config = _small_config()

    def failing_lsm_price(*_args, **_kwargs):
        raise OverflowError("LSM simulated spot left double precision")

    monkeypatch.setattr(
        lsm_crosscheck_module._core, "lsm_price", failing_lsm_price
    )
    with pytest.raises(LsmCrosscheckError) as error:
        run_lsm_crosscheck(config)
    message = str(error.value)
    assert "small-reference-v1" in message
    assert config.cases[0].name in message
    assert "left double precision" in message


def test_report_write_is_atomic_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "report.json"
    report = {"schema_version": "test/1", "value": 1.25}
    write_report(report, output)
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert not list(output.parent.glob(".*.tmp"))
    with pytest.raises(LsmCrosscheckError, match="refusing to overwrite"):
        write_report(report, output)
    invalid_output = tmp_path / "nonfinite.json"
    with pytest.raises(LsmCrosscheckError, match="not finite canonical JSON"):
        write_report({"value": float("inf")}, invalid_output)
    assert not invalid_output.exists()


def test_cli_preflights_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing.json"
    output.write_text("{}\n", encoding="utf-8")

    def unexpected_load(_path):
        raise AssertionError("configuration must not be loaded")

    monkeypatch.setattr(
        lsm_crosscheck_module, "load_lsm_crosscheck_config", unexpected_load
    )
    assert main(["--config", str(CONFIG), "--output", str(output)]) == 2
