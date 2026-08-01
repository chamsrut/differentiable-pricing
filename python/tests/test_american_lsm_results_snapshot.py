"""The frozen American LSM cross-check result must stay strict, honest, and reproducible.

These tests deliberately never read ``artifacts/``. The raw cross-check report is
git-ignored, so everything CI enforces has to hold using only checked-in files:
the snapshot, the repository sources it claims provenance over, the figures, and
the README table. Extraction itself is exercised against a small synthetic
fixture built in this module.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
import stat
import sys
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FREEZE_SCRIPT = PROJECT_ROOT / "scripts" / "freeze_american_lsm_results.py"
PLOT_SCRIPT = PROJECT_ROOT / "scripts" / "plot_american_lsm_results.py"
SNAPSHOT_PATH = (
    PROJECT_ROOT / "docs" / "results" / "american_lsm_crosscheck_results_v1.json"
)
FIGURES = PROJECT_ROOT / "docs" / "figures"
README = PROJECT_ROOT / "README.md"


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


freeze = _load("dp_freeze_american_lsm", FREEZE_SCRIPT)
plot = _load("dp_plot_american_lsm", PLOT_SCRIPT)


@pytest.fixture
def snapshot() -> dict[str, Any]:
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Synthetic raw-report fixture
# ---------------------------------------------------------------------------


def _regression(fallbacks: int = 0) -> dict[str, Any]:
    return {
        "constant_fallback_dates": fallbacks,
        "dates_with_in_the_money_paths": 31,
        "exercise_dates_fitted": 31,
        "minimum_in_the_money_paths": 900,
        "minimum_relative_r_diagonal": 0.9999,
    }


def _stochastic_case() -> dict[str, Any]:
    return {
        "case": {
            "dividend_yield": 0.0,
            "exercise_expectation": "material",
            "maturity": 1.0,
            "name": "fixture_put",
            "option_type": "put",
            "rate": 0.05,
            "spot": 100.0,
            "strike": 100.0,
            "volatility": 0.2,
        },
        "crr_reference": {
            "absolute_pair_gap": 0.0002,
            "at_steps": 6.04,
            "at_steps_plus_one": 6.06,
            "price": 6.05,
            "steps": 8192,
        },
        "lsm": {
            "confidence_interval_lower": 5.96,
            "confidence_interval_upper": 6.04,
            "confidence_level": 0.95,
            "control_variate_coefficient": 0.35,
            "crr_reference_inside_lsm_valuation_interval": False,
            "crr_reference_minus_lsm": 0.05,
            "estimated_training_working_set_bytes": 1024,
            "european_analytic_price": 5.57,
            "european_monte_carlo_price": 5.55,
            "european_monte_carlo_sampled": True,
            "european_standard_error": 0.03,
            "exercise_at_zero": False,
            "exercise_steps": 32,
            "independent_valuation_pairs": 512,
            "price": 6.0,
            "raw_price": 5.99,
            "raw_standard_error": 0.03,
            "regression_summary": _regression(),
            "standard_error": 0.02,
            "standardized_crr_gap": 2.5,
            "training_continuation_value_at_zero": 6.04,
            "training_paths": 1024,
            "valuation_early_exercise_fraction": 0.4,
            "valuation_early_exercise_paths": 409,
            "valuation_interval_is_zero_width": False,
            "valuation_paths": 1024,
            "variance_reduction_applicable": True,
            "variance_reduction_is_infinite": False,
            "variance_reduction_ratio": 1.5,
        },
        "seeds": {"derivation": "fixture", "training": 1, "valuation": 2},
    }


def _deterministic_case() -> dict[str, Any]:
    return {
        "case": {
            "dividend_yield": 0.0,
            "exercise_expectation": "immediate",
            "maturity": 1.0,
            "name": "fixture_deep_itm_put",
            "option_type": "put",
            "rate": 0.05,
            "spot": 70.0,
            "strike": 100.0,
            "volatility": 0.25,
        },
        "crr_reference": {
            "absolute_pair_gap": 0.0,
            "at_steps": 30.0,
            "at_steps_plus_one": 30.0,
            "price": 30.0,
            "steps": 8192,
        },
        "lsm": {
            "confidence_interval_lower": 30.0,
            "confidence_interval_upper": 30.0,
            "confidence_level": 0.95,
            "control_variate_coefficient": 0.0,
            "crr_reference_inside_lsm_valuation_interval": False,
            "crr_reference_minus_lsm": 1e-09,
            "estimated_training_working_set_bytes": 1024,
            "european_analytic_price": 27.0,
            "european_monte_carlo_price": None,
            "european_monte_carlo_sampled": False,
            "european_standard_error": None,
            "exercise_at_zero": True,
            "exercise_steps": 32,
            "independent_valuation_pairs": 512,
            "price": 30.0,
            "raw_price": 30.0,
            "raw_standard_error": 0.0,
            "regression_summary": _regression(),
            "standard_error": 0.0,
            "standardized_crr_gap": None,
            "training_continuation_value_at_zero": 29.0,
            "training_paths": 1024,
            "valuation_early_exercise_fraction": 1.0,
            "valuation_early_exercise_paths": 1024,
            "valuation_interval_is_zero_width": True,
            "valuation_paths": 1024,
            "variance_reduction_applicable": False,
            "variance_reduction_is_infinite": False,
            "variance_reduction_ratio": None,
        },
        "seeds": {"derivation": "fixture", "training": 3, "valuation": 4},
    }


def fixture_report() -> dict[str, Any]:
    """A minimal report with one stochastic and one deterministic case."""
    cases = [_stochastic_case(), _deterministic_case()]
    source = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))["source"]
    return {
        "schema_version": freeze.REPORT_SCHEMA_VERSION,
        "source": {
            "build_configuration": "Release",
            "config_file": "american_lsm_crosscheck_v1.toml",
            "config_sha256": source["config_sha256"],
            "crr_implementation_sha256": source["crr_implementation_sha256"],
            "cxx_compiler": "GNU 13.3.0",
            "lsm_header_sha256": source["lsm_header_sha256"],
            "lsm_implementation_sha256": source["lsm_implementation_sha256"],
            "lsm_source_sha256": source["lsm_source_sha256"],
            "oracle_version": "0.1.0",
            "rng": "splitmix64_box_muller_antithetic_v1",
        },
        "study": {
            "base_seed": 1,
            "case_names": ["fixture_put", "fixture_deep_itm_put"],
            "case_source": "configs/american_crr_convergence_v1.toml",
            "case_source_sha256": source["case_source_sha256"],
            "crr_reference_definition": "adjacent average at N and N+1",
            "crr_reference_steps": 8192,
            "engine": "dp::least_squares_monte_carlo/v1",
            "maximum_training_memory_bytes": 1048576,
            "name": "fixture-study",
        },
        "experiments": [
            {
                "cases": cases,
                "experiment": {
                    "exercise_steps": 32,
                    "name": "fixture-primary-v1",
                    "polynomial_degree": 2,
                    "role": "primary",
                    "training_paths": 1024,
                    "valuation_paths": 1024,
                },
                "summary": {
                    "cases": 2,
                    "crr_reference_inside_stochastic_valuation_interval_cases": 0,
                    "deterministic_maximum_absolute_crr_difference": 1e-09,
                    "deterministic_zero_width_valuation_cases": 1,
                    "maximum_absolute_crr_reference_minus_lsm": 0.05,
                    "mean_crr_reference_minus_lsm": 0.0250000005,
                    "mean_standard_error": 0.01,
                    "mean_stochastic_standard_error": 0.02,
                    "minimum_variance_reduction_ratio": 1.5,
                    "notes": {"deterministic_zero_width_valuation_cases": "fixture note"},
                    "stochastic_valuation_cases": 1,
                    "total_constant_fallback_dates": 0,
                    "variance_reduction_applicable_cases": 1,
                    "variance_reduction_infinite_cases": 0,
                },
            }
        ],
        "interpretation": {key: "fixture text" for key in freeze.REPORT_INTERPRETATION_KEYS},
    }


DIGEST = "0" * 64


def extract(report: dict[str, Any]) -> dict[str, Any]:
    return freeze.extract_snapshot(report, report_filename="fixture.json", report_sha256=DIGEST)


# ---------------------------------------------------------------------------
# Checked-in snapshot
# ---------------------------------------------------------------------------


def test_checked_in_snapshot_validates(snapshot: dict[str, Any]) -> None:
    assert freeze.validate_snapshot(snapshot) is snapshot


def test_checked_in_snapshot_provenance_matches_repository(
    snapshot: dict[str, Any],
) -> None:
    verified = freeze.verify_provenance(snapshot)
    assert set(verified) == {
        "config_sha256",
        "case_source_sha256",
        "lsm_header_sha256",
        "lsm_source_sha256",
        "crr_implementation_sha256",
        "lsm_implementation_sha256",
    }
    for field, digest in verified.items():
        assert snapshot["source"][field] == digest


def test_checked_in_snapshot_is_canonically_serialised(snapshot: dict[str, Any]) -> None:
    assert SNAPSHOT_PATH.read_text(encoding="utf-8") == freeze.serialise(snapshot)


def test_check_mode_passes_without_the_ignored_raw_report() -> None:
    """CI must be able to enforce the snapshot with artifacts/ absent."""
    assert freeze.main(["--check", "--output", str(SNAPSHOT_PATH)]) == 0


def test_check_mode_never_touches_the_artifacts_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[str] = []
    original = Path.read_bytes

    def spy(self: Path) -> bytes:
        opened.append(str(self))
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", spy)
    assert freeze.main(["--check", "--output", str(SNAPSHOT_PATH)]) == 0
    assert not [name for name in opened if "artifacts" in name]


def test_snapshot_records_the_reviewed_report_identity(snapshot: dict[str, Any]) -> None:
    source = snapshot["source"]
    assert source["report_filename"] == "american-lsm-crosscheck-review-fixed-v1.json"
    assert re.fullmatch(r"[0-9a-f]{64}", source["report_sha256"])
    assert source["report_schema_version"] == freeze.REPORT_SCHEMA_VERSION


def test_snapshot_covers_every_case_row(snapshot: dict[str, Any]) -> None:
    study = snapshot["study"]
    assert study["experiment_count"] == len(snapshot["experiments"]) == 7
    assert study["case_count"] == len(study["case_names"]) == 9
    assert study["case_rows"] == 63
    assert len(snapshot["primary_cases"]) == 9


# ---------------------------------------------------------------------------
# Extraction from the fixture
# ---------------------------------------------------------------------------


def test_extraction_derives_every_field_from_the_report() -> None:
    result = extract(fixture_report())
    assert result["schema_version"] == freeze.SNAPSHOT_SCHEMA_VERSION
    assert result["source"]["report_sha256"] == DIGEST
    assert result["study"]["primary_experiment"] == "fixture-primary-v1"
    assert result["study"]["case_rows"] == 2
    experiment = result["experiments"][0]
    assert experiment["mean_crr_reference_minus_lsm"] == 0.0250000005
    assert experiment["maximum_absolute_crr_reference_minus_lsm"] == 0.05
    assert experiment["mean_stochastic_valuation_standard_error"] == 0.02
    assert experiment["minimum_applicable_finite_variance_reduction_ratio"] == 1.5
    assert experiment["deterministic_zero_width_valuation_cases"] == 1
    assert experiment["stochastic_valuation_cases"] == 1
    assert experiment["stochastic_valuation_interval_containment_cases"] == 0
    stochastic, deterministic = result["primary_cases"]
    assert stochastic["lsm_price"] == 6.0
    assert stochastic["crr_reference_price"] == 6.05
    assert stochastic["standardized_crr_gap"] == 2.5
    assert stochastic["variance_reduction_ratio"] == 1.5
    assert deterministic["valuation_interval_is_zero_width"] is True
    assert deterministic["standardized_crr_gap"] is None
    assert deterministic["variance_reduction_ratio"] is None


def test_extraction_is_deterministic() -> None:
    first = freeze.serialise(extract(fixture_report()))
    second = freeze.serialise(extract(fixture_report()))
    assert first == second
    assert first.endswith("\n")


def test_serialisation_carries_no_wall_clock_field() -> None:
    text = freeze.serialise(extract(fixture_report())).lower()
    for banned in ("timestamp", "generated_at", "created_at", "hostname", "elapsed"):
        assert banned not in text


# ---------------------------------------------------------------------------
# Rejection cases
# ---------------------------------------------------------------------------


def test_unknown_report_field_is_rejected() -> None:
    report = fixture_report()
    report["experiments"][0]["experiment"]["unexpected"] = 1
    with pytest.raises(freeze.FreezeError, match="unknown fields"):
        extract(report)


def test_missing_report_field_is_rejected() -> None:
    report = fixture_report()
    del report["experiments"][0]["summary"]["mean_standard_error"]
    with pytest.raises(freeze.FreezeError, match="missing required fields"):
        extract(report)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_price_is_rejected(bad: float) -> None:
    report = fixture_report()
    report["experiments"][0]["cases"][0]["lsm"]["price"] = bad
    with pytest.raises(freeze.FreezeError, match="must be finite"):
        extract(report)


def test_wrong_report_schema_version_is_rejected() -> None:
    report = fixture_report()
    report["schema_version"] = "american-lsm-crosscheck-report/2"
    with pytest.raises(freeze.FreezeError, match="schema_version"):
        extract(report)


def test_edited_summary_mean_is_rejected() -> None:
    """A hand-edited summary must not survive extraction."""
    report = fixture_report()
    report["experiments"][0]["summary"]["mean_crr_reference_minus_lsm"] = 0.001
    with pytest.raises(freeze.FreezeError, match="recomputed"):
        extract(report)


def test_deterministic_case_leaking_into_stochastic_mean_is_rejected() -> None:
    """Averaging the zero-width case into the stochastic SE must fail."""
    report = fixture_report()
    report["experiments"][0]["summary"]["mean_stochastic_standard_error"] = 0.01
    with pytest.raises(freeze.FreezeError, match="mean_stochastic_standard_error"):
        extract(report)


def test_deterministic_case_counted_as_stochastic_is_rejected() -> None:
    report = fixture_report()
    summary = report["experiments"][0]["summary"]
    summary["deterministic_zero_width_valuation_cases"] = 0
    summary["stochastic_valuation_cases"] = 2
    with pytest.raises(freeze.FreezeError, match="deterministic_zero_width"):
        extract(report)


def test_deterministic_case_claiming_containment_is_rejected() -> None:
    report = fixture_report()
    case = report["experiments"][0]["cases"][1]["lsm"]
    case["crr_reference_inside_lsm_valuation_interval"] = True
    with pytest.raises(freeze.FreezeError, match=r"containment|deterministic"):
        extract(report)


def test_deterministic_case_lowering_the_variance_minimum_is_rejected() -> None:
    """A zero-width case must not be able to set the reported minimum ratio."""
    report = fixture_report()
    deterministic = report["experiments"][0]["cases"][1]["lsm"]
    # Give the zero-width case a sampled control so it clears the earlier
    # applicability rule and the variance-minimum rule is what actually fires.
    deterministic["european_monte_carlo_sampled"] = True
    deterministic["european_monte_carlo_price"] = 27.0
    deterministic["european_standard_error"] = 0.0
    deterministic["variance_reduction_applicable"] = True
    deterministic["variance_reduction_ratio"] = 0.2
    report["experiments"][0]["summary"]["minimum_variance_reduction_ratio"] = 0.2
    report["experiments"][0]["summary"]["variance_reduction_applicable_cases"] = 2
    with pytest.raises(freeze.FreezeError, match="minimum_variance_reduction_ratio"):
        extract(report)


def test_zero_width_case_with_non_zero_error_is_rejected() -> None:
    report = fixture_report()
    report["experiments"][0]["cases"][1]["lsm"]["standard_error"] = 0.01
    with pytest.raises(freeze.FreezeError, match="zero-width interval"):
        extract(report)


def test_deterministic_case_reporting_a_standardized_gap_is_rejected() -> None:
    report = fixture_report()
    report["experiments"][0]["cases"][1]["lsm"]["standardized_crr_gap"] = 1.0
    with pytest.raises(freeze.FreezeError, match="standardized gap"):
        extract(report)


def test_report_without_exactly_one_primary_experiment_is_rejected() -> None:
    report = fixture_report()
    report["experiments"][0]["experiment"]["role"] = "reference"
    with pytest.raises(freeze.FreezeError, match="exactly one experiment"):
        extract(report)


def test_malformed_json_is_reported_as_an_error(tmp_path: Path) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert freeze.main(["--report", str(broken), "--output", str(tmp_path / "out.json")]) == 2


def test_snapshot_with_unknown_field_is_rejected(snapshot: dict[str, Any]) -> None:
    snapshot["unexpected"] = True
    with pytest.raises(freeze.FreezeError, match="unknown fields"):
        freeze.validate_snapshot(snapshot)


def test_snapshot_provenance_mismatch_is_rejected(snapshot: dict[str, Any]) -> None:
    snapshot["source"]["lsm_source_sha256"] = "a" * 64
    with pytest.raises(freeze.FreezeError, match="lsm_source_sha256"):
        freeze.verify_provenance(snapshot)


def test_snapshot_primary_case_disagreeing_with_its_summary_is_rejected(
    snapshot: dict[str, Any],
) -> None:
    edited = copy.deepcopy(snapshot)
    # Perturb a deterministic row, which carries no standardized gap, so the
    # summary-reconciliation check is the one that fires rather than the
    # per-case standardized-gap check.
    deterministic = next(
        index
        for index, case in enumerate(edited["primary_cases"])
        if case["valuation_interval_is_zero_width"]
    )
    edited["primary_cases"][deterministic]["crr_reference_minus_lsm"] += 1.0
    with pytest.raises(freeze.FreezeError, match="primary_cases imply"):
        freeze.validate_snapshot(edited)


# ---------------------------------------------------------------------------
# Write behaviour
# ---------------------------------------------------------------------------


def test_generation_refuses_silent_overwrite(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps(fixture_report()), encoding="utf-8")
    output = tmp_path / "snapshot.json"
    assert freeze.main(["--report", str(report), "--output", str(output)]) == 0
    original = output.read_text(encoding="utf-8")
    assert freeze.main(["--report", str(report), "--output", str(output)]) == 2
    assert output.read_text(encoding="utf-8") == original
    assert freeze.main(["--report", str(report), "--output", str(output), "--update"]) == 0


def test_update_cannot_be_combined_with_check(tmp_path: Path) -> None:
    assert freeze.main(["--check", "--update", "--output", str(SNAPSHOT_PATH)]) == 2


def test_write_is_atomic_and_leaves_no_temporary_behind(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "snapshot.json"
    freeze.write_atomic(target, "payload\n")
    assert target.read_text(encoding="utf-8") == "payload\n"
    assert sorted(p.name for p in target.parent.iterdir()) == ["snapshot.json"]


def test_failed_write_preserves_the_previous_file(tmp_path: Path) -> None:
    """A write that dies mid-flight must not truncate the existing snapshot."""
    target = tmp_path / "snapshot.json"
    target.write_text("original\n", encoding="utf-8")

    real_replace = os.replace

    def failing_replace(source: Any, destination: Any) -> None:
        raise OSError("disk full")

    os.replace = failing_replace
    try:
        with pytest.raises(OSError):
            freeze.write_atomic(target, "replacement\n")
    finally:
        os.replace = real_replace
    assert target.read_text(encoding="utf-8") == "original\n"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["snapshot.json"]


def test_unwritable_directory_returns_the_documented_exit_code(tmp_path: Path) -> None:
    """An I/O failure must honour the exit-2 contract, not escape as a traceback."""
    report = tmp_path / "report.json"
    report.write_text(json.dumps(fixture_report()), encoding="utf-8")
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(stat.S_IREAD | stat.S_IEXEC)
    try:
        if os.access(locked, os.W_OK):  # pragma: no cover - running as root
            pytest.skip("directory permissions are not enforced for this user")
        assert freeze.main(["--report", str(report), "--output", str(locked / "s.json")]) == 2
    finally:
        locked.chmod(stat.S_IRWXU)


def test_plot_unwritable_directory_returns_the_documented_exit_code(
    tmp_path: Path,
) -> None:
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(stat.S_IREAD | stat.S_IEXEC)
    try:
        if os.access(locked, os.W_OK):  # pragma: no cover - running as root
            pytest.skip("directory permissions are not enforced for this user")
        assert plot.main(["--output-directory", str(locked)]) == 2
    finally:
        locked.chmod(stat.S_IRWXU)


def test_plot_write_mode_produces_the_checked_in_figures(tmp_path: Path) -> None:
    """The non---check write path is exercised, not just the comparison path."""
    assert plot.main(["--output-directory", str(tmp_path)]) == 0
    written = sorted(p.name for p in tmp_path.iterdir())
    assert written == [
        "american_lsm_experiment_comparison.svg",
        "american_lsm_primary_cases.svg",
        "american_lsm_sensitivity.svg",
    ]
    for name in written:
        assert (tmp_path / name).read_text(encoding="utf-8") == (
            FIGURES / name
        ).read_text(encoding="utf-8")


def test_failed_figure_write_preserves_the_previous_figure(tmp_path: Path) -> None:
    """A figure write that dies mid-flight must not truncate the existing SVG."""
    target = tmp_path / "american_lsm_sensitivity.svg"
    target.write_text("original\n", encoding="utf-8")
    real_replace = os.replace

    def failing_replace(source: Any, destination: Any) -> None:
        raise OSError("disk full")

    os.replace = failing_replace
    try:
        assert plot.main(["--output-directory", str(tmp_path)]) == 2
    finally:
        os.replace = real_replace
    assert target.read_text(encoding="utf-8") == "original\n"
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


# ---------------------------------------------------------------------------
# Semantics preserved in the frozen snapshot
# ---------------------------------------------------------------------------


def test_every_experiment_separates_deterministic_from_stochastic_cases(
    snapshot: dict[str, Any],
) -> None:
    for experiment in snapshot["experiments"]:
        deterministic = experiment["deterministic_zero_width_valuation_cases"]
        stochastic = experiment["stochastic_valuation_cases"]
        assert deterministic == 3
        assert stochastic == 6
        assert deterministic + stochastic == experiment["cases"] == 9
        assert experiment["stochastic_valuation_interval_containment_cases"] <= stochastic


def test_primary_deterministic_cases_are_exactly_the_zero_width_ones(
    snapshot: dict[str, Any],
) -> None:
    deterministic = [
        case
        for case in snapshot["primary_cases"]
        if case["valuation_interval_is_zero_width"]
    ]
    assert {case["name"] for case in deterministic} == {
        "deep_itm_put",
        "negative_rate_put",
        "non_dividend_call_control",
    }
    for case in deterministic:
        assert case["valuation_only_standard_error"] == 0.0
        assert case["standardized_crr_gap"] is None
        assert case["variance_reduction_ratio"] is None


def test_every_applicable_finite_variance_ratio_is_at_least_about_one(
    snapshot: dict[str, Any],
) -> None:
    for experiment in snapshot["experiments"]:
        ratio = experiment["minimum_applicable_finite_variance_reduction_ratio"]
        assert ratio >= 1.0 - 1e-3, experiment["name"]


def test_basis_degree_rows_are_not_relabelled(snapshot: dict[str, Any]) -> None:
    """degree-one/two/three arms must keep their declared degree and controls."""
    by_name = {e["name"]: e for e in snapshot["experiments"]}
    expected = {"degree-one-v1": 1, "reference-v1": 2, "degree-three-v1": 3}
    for name, degree in expected.items():
        assert by_name[name]["polynomial_degree"] == degree, name
        # The basis comparison is only valid if every other factor is held fixed.
        assert by_name[name]["exercise_steps"] == 64
        assert by_name[name]["training_paths"] == 32768
        assert by_name[name]["valuation_paths"] == 65536
    assert by_name["degree-one-v1"]["role"] == "basis_sensitivity"
    assert by_name["degree-three-v1"]["role"] == "basis_sensitivity"
    # The headline finding: degree one is an order of magnitude worse, while
    # quadratic and cubic are comparable at this path budget.
    one = by_name["degree-one-v1"]["mean_crr_reference_minus_lsm"]
    two = by_name["reference-v1"]["mean_crr_reference_minus_lsm"]
    three = by_name["degree-three-v1"]["mean_crr_reference_minus_lsm"]
    assert one > 10.0 * two
    assert abs(three - two) < 0.5 * two


def test_exercise_grid_arms_hold_paths_fixed(snapshot: dict[str, Any]) -> None:
    by_name = {e["name"]: e for e in snapshot["experiments"]}
    for name, steps in {"steps-low-v1": 32, "reference-v1": 64, "steps-high-v1": 128}.items():
        assert by_name[name]["exercise_steps"] == steps
        assert by_name[name]["training_paths"] == 32768
        assert by_name[name]["polynomial_degree"] == 2
    # Refining the exercise grid at a fixed path budget did not reduce the gap.
    assert by_name["steps-high-v1"]["mean_crr_reference_minus_lsm"] > (
        by_name["reference-v1"]["mean_crr_reference_minus_lsm"]
    )


def test_path_ladder_reduces_valuation_noise(snapshot: dict[str, Any]) -> None:
    by_name = {e["name"]: e for e in snapshot["experiments"]}
    ladder = ["paths-low-v1", "reference-v1", "paths-high-primary-v1"]
    for name in ladder:
        assert by_name[name]["exercise_steps"] == 64
        assert by_name[name]["polynomial_degree"] == 2
    paths = [by_name[name]["training_paths"] for name in ladder]
    errors = [by_name[name]["mean_stochastic_valuation_standard_error"] for name in ladder]
    assert paths == sorted(paths)
    assert errors == sorted(errors, reverse=True)

    # The README quotes these ratios and their deviation from the sqrt(2) that
    # n^-1/2 implies for a doubling. Pin them so regenerating the snapshot
    # cannot silently invalidate that prose.
    ratios = [before / after for before, after in pairwise(errors)]
    assert ratios[0] == pytest.approx(1.3946, abs=5e-5)
    assert ratios[1] == pytest.approx(1.4244, abs=5e-5)
    for ratio in ratios:
        assert abs(ratio / 2.0**0.5 - 1.0) < 0.02


def test_path_ladder_mean_gap_is_not_monotone(snapshot: dict[str, Any]) -> None:
    """The README must not be allowed to claim a clean monotone improvement."""
    by_name = {e["name"]: e for e in snapshot["experiments"]}
    gaps = [
        by_name[name]["mean_crr_reference_minus_lsm"]
        for name in ("paths-low-v1", "reference-v1", "paths-high-primary-v1")
    ]
    assert gaps != sorted(gaps, reverse=True), "gap ladder is monotone; README says it is not"
    assert gaps[-1] < gaps[0], "end-to-end improvement claimed by the README is absent"


def test_report_semantics_state_valuation_only_coverage(snapshot: dict[str, Any]) -> None:
    semantics = snapshot["report_semantics"]
    assert "valuation sampling error only" in semantics["valuation_interval"]
    crr = semantics["crr_reference"].lower()
    assert "internal model cross-check" in crr
    assert "neither exact american truth nor market truth" in crr
    assert "acceptance gate" in crr
    assert "excluded" in semantics["variance_reduction"]
    joined = " ".join(snapshot["limitations"]).lower()
    assert "policy-fitting" in joined
    assert "grid" in joined
    assert "no market data" in joined
    assert "task 8e" in joined


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def test_checked_in_figures_are_current() -> None:
    assert plot.main(["--check"]) == 0


def test_every_rendered_figure_is_checked_in() -> None:
    figures = plot.render_figures(SNAPSHOT_PATH)
    assert set(figures) == {
        "american_lsm_experiment_comparison.svg",
        "american_lsm_primary_cases.svg",
        "american_lsm_sensitivity.svg",
    }
    for name, content in figures.items():
        assert (FIGURES / name).read_text(encoding="utf-8") == content


def test_figure_rendering_is_deterministic() -> None:
    assert plot.render_figures(SNAPSHOT_PATH) == plot.render_figures(SNAPSHOT_PATH)


def test_figures_carry_no_environment_dependent_metadata() -> None:
    for content in plot.render_figures(SNAPSHOT_PATH).values():
        lowered = content.lower()
        for banned in ("timestamp", "generated", str(PROJECT_ROOT).lower(), "id="):
            assert banned not in lowered


def test_stale_figure_is_detected(tmp_path: Path) -> None:
    edited = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    edited["experiments"][0]["mean_crr_reference_minus_lsm"] = 0.5
    edited["experiments"][0]["maximum_absolute_crr_reference_minus_lsm"] = 0.9
    path = tmp_path / "snapshot.json"
    path.write_text(freeze.serialise(edited), encoding="utf-8")
    assert (
        plot.main(["--snapshot", str(path), "--check", "--output-directory", str(FIGURES)]) == 2
    )


def test_plotting_rejects_an_invalid_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.json"
    path.write_text('{"schema_version": "nope"}', encoding="utf-8")
    assert plot.main(["--snapshot", str(path), "--check"]) == 2


def test_figures_label_uncertainty_as_valuation_only() -> None:
    figures = plot.render_figures(SNAPSHOT_PATH)
    comparison = figures["american_lsm_experiment_comparison.svg"]
    assert "valuation-only SE" in comparison
    assert "NOT a total-error bar" in comparison
    cases = figures["american_lsm_primary_cases.svg"]
    assert "valuation-only standard error" in cases
    assert "Deterministic case (zero-width interval)" in cases
    assert "not an acceptance gate" in cases


# ---------------------------------------------------------------------------
# README agreement
# ---------------------------------------------------------------------------

README_ROW = re.compile(r"^\|\s*`?(?P<name>[a-z0-9-]+-v1)`?\s*\|(?P<rest>.*)\|\s*$", re.MULTILINE)


def _readme_rows() -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for match in README_ROW.finditer(README.read_text(encoding="utf-8")):
        cells = [cell.strip().strip("`") for cell in match.group("rest").split("|")]
        rows[match.group("name")] = cells
    return rows


def test_readme_table_matches_the_snapshot(snapshot: dict[str, Any]) -> None:
    """Every number in the README table is read back and compared to the snapshot."""
    rows = _readme_rows()
    experiments = {e["name"]: e for e in snapshot["experiments"]}
    assert set(rows) == set(experiments), "README table and snapshot list different experiments"
    for name, cells in rows.items():
        experiment = experiments[name]
        assert len(cells) == 12, f"{name} row has {len(cells)} cells"
        (
            role,
            steps,
            degree,
            train,
            valuation,
            mean_gap,
            max_gap,
            mean_error,
            stochastic,
            containment,
            deterministic,
            minimum_ratio,
        ) = cells
        assert role == experiment["role"].replace("_", " ")
        assert int(steps) == experiment["exercise_steps"]
        assert int(degree) == experiment["polynomial_degree"]
        assert int(train.replace(",", "")) == experiment["training_paths"]
        assert int(valuation.replace(",", "")) == experiment["valuation_paths"]
        assert float(mean_gap) == pytest.approx(
            experiment["mean_crr_reference_minus_lsm"], abs=5e-5
        )
        assert float(max_gap) == pytest.approx(
            experiment["maximum_absolute_crr_reference_minus_lsm"], abs=5e-5
        )
        assert float(mean_error) == pytest.approx(
            experiment["mean_stochastic_valuation_standard_error"], abs=5e-5
        )
        assert int(stochastic) == experiment["stochastic_valuation_cases"]
        assert int(containment) == experiment["stochastic_valuation_interval_containment_cases"]
        assert int(deterministic) == experiment["deterministic_zero_width_valuation_cases"]
        assert float(minimum_ratio) == pytest.approx(
            experiment["minimum_applicable_finite_variance_reduction_ratio"], abs=5e-5
        )


def _readme_prose() -> str:
    """README text with markdown emphasis and line wrapping normalised away.

    Substring assertions over the raw file are vacuous: a line break inside a
    phrase, or a bold marker in the middle of it, silently defeats the match.
    Collapsing whitespace and stripping emphasis makes these guards real.
    """
    text = README.read_text(encoding="utf-8").lower().replace("*", "").replace("`", "")
    return re.sub(r"\s+", " ", text)


def test_readme_does_not_claim_total_error_coverage() -> None:
    prose = _readme_prose()
    for forbidden in (
        "crr is exact",
        "reference is exact american truth",
        "validates market",
        "market-calibrated",
        "acceptance gate for containment",
    ):
        assert forbidden not in prose, forbidden
    # The disclaimers must be present as complete phrases, not merely implied.
    for required in (
        "not exact american truth",
        "not market truth",
        "not an acceptance gate",
        "valuation-only",
        "policy-fitting",
        "exercise-grid",
        "no market data",
        "select no production label policy",
        "task 8e",
    ):
        assert required in prose, required


def test_readme_references_the_generated_figures() -> None:
    text = README.read_text(encoding="utf-8")
    for name in (
        "american_lsm_experiment_comparison.svg",
        "american_lsm_primary_cases.svg",
        "american_lsm_sensitivity.svg",
    ):
        assert name in text
