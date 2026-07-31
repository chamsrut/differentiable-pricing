"""Tests for the deterministic CRR convergence and benchmark protocols."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tomllib
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import differentiable_pricing.american.convergence as convergence_module
import pytest
from differentiable_pricing.american.convergence import (
    ConvergenceError,
    load_convergence_config,
    main,
    parse_convergence_config,
    run_convergence,
    write_report,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "configs" / "american_crr_convergence_v1.toml"
BENCHMARK_SCRIPT = PROJECT_ROOT / "scripts" / "benchmark_american_crr.py"


def _load_benchmark_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("dp_benchmark_american_crr", BENCHMARK_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _document() -> dict[str, object]:
    with CONFIG.open("rb") as stream:
        return tomllib.load(stream)


def _small_config():
    document = copy.deepcopy(_document())
    document["study"]["step_ladder"] = [8, 16]
    document["study"]["reference_steps"] = 32
    document["study"]["diagnostic_step_ladder"] = [8, 16]
    document["study"]["batch_threads"] = 2
    document["cases"] = document["cases"][:2]
    document["feasibility"] = {
        "maturities": [0.05, 1.0],
        "rates": [-0.02, 0.15],
        "dividend_yields": [0.0, 0.10],
        "volatilities": [0.05, 0.20],
        "step_candidates": [1, 2, 4, 8, 16, 32],
    }
    return parse_convergence_config(
        document,
        source_name="small.toml",
        source_sha256="0" * 64,
    )


def test_checked_in_config_is_valid_and_declares_high_step_reference() -> None:
    config = load_convergence_config(CONFIG)
    assert config.name == "american-crr-convergence-v1"
    assert config.reference_steps == 8192
    assert config.diagnostic_step_ladder == (256, 512, 1024, 2048)
    assert config.step_ladder == (64, 128, 256, 512, 1024, 2048, 4096)
    assert len(config.cases) == 13
    assert next(
        case for case in config.cases if case.name == "negative_rate_call"
    ).exercise_expectation == "material"
    assert config.source_sha256 == hashlib.sha256(CONFIG.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda document: document.update({"unknown": 1}), "unknown keys"),
        (
            lambda document: document["study"].update({"reference_steps": 4096}),
            "must exceed",
        ),
        (
            lambda document: document["study"].update({"step_ladder": [64, 32]}),
            "strictly increasing",
        ),
        (
            lambda document: document["study"].update(
                {"diagnostic_step_ladder": [63]}
            ),
            "must be a subset",
        ),
        (
            lambda document: document["cases"][1].update(
                {"name": document["cases"][0]["name"]}
            ),
            "unique",
        ),
        (
            lambda document: document["cases"][0].update({"volatility": 0.0}),
            "finite and positive",
        ),
        (
            lambda document: document["study"].update({"batch_threads": 0}),
            "positive integer",
        ),
        (
            lambda document: document["cases"][0].update(
                {"exercise_expectation": "sometimes"}
            ),
            "exercise_expectation",
        ),
        (
            lambda document: document["cases"][0].update(
                {"exercise_expectation": ["material"]}
            ),
            "exercise_expectation",
        ),
    ],
)
def test_invalid_configurations_are_rejected(mutation, message: str) -> None:
    document = _document()
    mutation(document)
    with pytest.raises(ConvergenceError, match=message):
        parse_convergence_config(document, source_name="bad.toml", source_sha256="0" * 64)


def test_small_convergence_report_is_complete_and_deterministic() -> None:
    config = _small_config()
    serial = run_convergence(replace(config, batch_threads=1))
    parallel = run_convergence(replace(config, batch_threads=2))

    assert serial["cases"] == parallel["cases"]
    assert serial["summary_by_steps"] == parallel["summary_by_steps"]
    assert serial["feasibility"] == parallel["feasibility"]
    assert len(serial["cases"]) == 2
    assert [row["steps"] for row in serial["summary_by_steps"]] == [8, 16]
    assert serial["feasibility"]["total_states"] == 16
    assert serial["feasibility"]["feasible_states"] <= 16
    assert (
        serial["control_summary"][
            "maximum_european_crr_absolute_error_vs_black_scholes"
        ]
        >= 0.0
    )
    for case in serial["cases"]:
        assert len(case["levels"]) == 2
        assert case["reference"]["steps"] == 32
        assert [row["steps"] for row in case["exercise_diagnostics"]] == [8, 16]
        assert all(
            len(row["boundary_samples"]) == 4
            for row in case["exercise_diagnostics"]
        )
        assert case["european_control"]["absolute_error"] >= 0.0
        assert all(
            level["absolute_difference_to_reference"] >= 0.0
            for level in case["levels"]
        )
    assert serial["source"]["build_configuration"]
    assert serial["source"]["cxx_compiler"]
    assert len(serial["source"]["crr_implementation_sha256"]) == 64
    assert (
        serial["interpretation"]["adjacent_refinement_discrepancy"]
        .startswith("The absolute N versus N+1 discrepancy")
    )


def test_report_write_is_atomic_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "report.json"
    report = {"schema_version": "test/1", "value": 1.25}
    write_report(report, output)
    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert not list(output.parent.glob(".*.tmp"))
    with pytest.raises(ConvergenceError, match="refusing to overwrite"):
        write_report(report, output)


def test_cli_writes_small_report(tmp_path: Path) -> None:
    document = _document()
    document["study"]["step_ladder"] = [8]
    document["study"]["reference_steps"] = 16
    document["study"]["batch_threads"] = 2
    document["cases"] = document["cases"][:1]
    document["feasibility"]["step_candidates"] = [1, 2, 4, 8, 16]
    config_path = tmp_path / "config.toml"
    # Keep this CLI test focused on execution; serialize the reduced document
    # through a small explicit fixture rather than adding a TOML writer.
    case = document["cases"][0]
    config_path.write_text(
        f"""
schema_version = "american-crr-convergence/1"
[study]
name = "cli-test"
engine = "dp::crr_binomial/v1"
exercise_style = "american"
step_ladder = [8]
reference_steps = 16
diagnostic_step_ladder = [8]
batch_threads = 2
[[cases]]
name = "{case['name']}"
option_type = "{case['option_type']}"
spot = {case['spot']}
strike = {case['strike']}
maturity = {case['maturity']}
rate = {case['rate']}
dividend_yield = {case['dividend_yield']}
volatility = {case['volatility']}
[feasibility]
maturities = [0.05]
rates = [0.0]
dividend_yields = [0.0]
volatilities = [0.2]
step_candidates = [1, 2, 4, 8, 16]
""".lstrip(),
        encoding="utf-8",
    )
    output = tmp_path / "report.json"
    assert main(["--config", str(config_path), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["study"]["name"] == "cli-test"
    assert main(["--config", str(config_path), "--output", str(output)]) == 2


def test_convergence_cli_preflights_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing.json"
    output.write_text("{}\n", encoding="utf-8")

    def unexpected_run(_config):
        raise AssertionError("expensive convergence must not run")

    monkeypatch.setattr(convergence_module, "run_convergence", unexpected_run)
    assert main(["--config", str(CONFIG), "--output", str(output)]) == 2


def test_benchmark_modes_preserve_prices_and_record_protocol() -> None:
    module = _load_benchmark_module()
    report = module.run_benchmark(
        config_path=CONFIG,
        steps=8,
        batch_sizes=(1, 3),
        thread_counts=(1, 2),
        warmups=0,
        repetitions=2,
    )
    assert report["schema_version"] == "american-crr-benchmark/1"
    assert report["protocol"]["steps"] == 8
    assert report["protocol"]["untimed_validation_prepasses_per_batch_size"] == 1
    assert len(report["measurements"]) == 6
    assert "cyclic rotation" in report["protocol"]["measurement_order"]
    for batch_size in (1, 3):
        digests = {
            measurement["output_sha256"]
            for measurement in report["measurements"]
            if measurement["batch_size"] == batch_size
        }
        assert len(digests) == 1
    assert all(
        len(measurement["durations_seconds"]) == 2
        and measurement["median_seconds"] > 0.0
        for measurement in report["measurements"]
    )
    one_row_four_requested = module.run_benchmark(
        config_path=CONFIG,
        steps=8,
        batch_sizes=(1,),
        thread_counts=(4,),
        warmups=0,
        repetitions=1,
    )
    batch = next(
        row for row in one_row_four_requested["measurements"]
        if row["mode"] == "batch"
    )
    assert batch["requested_thread_count"] == 4
    assert batch["effective_worker_count"] == 1


def test_benchmark_cli_preflights_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_benchmark_module()
    output = tmp_path / "existing.json"
    output.write_text("{}\n", encoding="utf-8")

    def unexpected_run(**_kwargs):
        raise AssertionError("expensive benchmark must not run")

    monkeypatch.setattr(module, "run_benchmark", unexpected_run)
    assert module.main(
        [
            "--config",
            str(CONFIG),
            "--output",
            str(output),
            "--steps",
            "8",
        ]
    ) == 2
