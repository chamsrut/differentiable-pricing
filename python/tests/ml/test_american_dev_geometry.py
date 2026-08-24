"""Task 9H validation-set geometry: definitions, guards and immutability.

Synthetic fixtures throughout. No dataset partition is opened, no model is
trained and :func:`analyze_validation_geometry` is exercised only along its
refusal paths and its pure helpers, which is where its guarantees live.

The properties pinned here are the ones an exploratory analysis can quietly
lose: that it reads exactly one partition and that the partition is fixed in the
source rather than passed in, that it recomputes no financial quantity of its
own, that its thresholds come from the recorded attempt log rather than from a
typed-in number, that it can neither reach nor name a final partition, and that
running it leaves every byte of existing attempt evidence untouched.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from differentiable_pricing.ml.american_dev.attempts import (
    ATTEMPT_LOG_PATH,
    AttemptError,
    FinalPartitionAccessError,
    load_toml,
    sha256_file,
)
from differentiable_pricing.ml.american_dev.geometry import (
    ABSOLUTE_THRESHOLDS,
    ANALYSIS_PARTITION,
    COMPARATOR_DISCREPANCY,
    DEFAULT_OUTPUT,
    DISCREPANCY_QUANTILES,
    DISCREPANCY_THRESHOLDS,
    EXPLORATORY_LABEL,
    GEOMETRY_SCHEMA,
    MEASURED_QUANTITIES,
    QUANTILES,
    REFERENCE_ATTEMPTS,
    GeometryError,
    _guarded_output,
    acceptance_view,
    analyze_validation_geometry,
    attempt_rmse_thresholds,
    column_consistency,
    compact_geometry,
    comparator_discrepancy,
    comparator_discrepancy_values,
    effectively_zero_premium_mask,
    exceedance_statistics,
    geometry_populations,
    normalized_slacks,
    population_masks,
    slack_statistics,
    thresholds,
)
from differentiable_pricing.ml.american_dev.representation import (
    european_price,
    european_price_array,
)
from differentiable_pricing.ml.american_pilot import physical_features

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GEOMETRY_MODULE = PROJECT_ROOT / "python/src/differentiable_pricing/ml/american_dev/geometry.py"
GEOMETRY_SCRIPT = PROJECT_ROOT / "scripts/analyze_american_dev_geometry.py"
REAL_LOG = PROJECT_ROOT / ATTEMPT_LOG_PATH
GEOMETRY_V1_REPORT = PROJECT_ROOT / "artifacts/task-9h/geometry/validation-geometry-v1.json"
ACCEPTANCE_CONFIG = PROJECT_ROOT / "configs/american_neural_pilot_acceptance_v1.toml"
BINS = {
    "expiry_edges_years": [0.25, 1.0, 2.0],
    "log_moneyness_edges": [-0.2, 0.2],
    "volatility_edges": [0.15, 0.5],
}


def _columns(count: int = 12) -> dict[str, np.ndarray]:
    """Rows whose slacks are known exactly by construction."""
    rng = np.random.default_rng(20260821)
    spot = np.linspace(90.0, 110.0, count)
    strike = np.full(count, 100.0)
    maturity = np.linspace(0.1, 2.5, count)
    dividend_yield = np.full(count, 0.02)
    option_type = np.asarray(["call" if index % 2 else "put" for index in range(count)])
    intrinsic = np.maximum(np.where(option_type == "call", spot - strike, strike - spot), 0.0)
    discounted_spot = spot * np.exp(-dividend_yield * maturity)
    # A deliberately wide spread of premia, including exact zeros, so the
    # at-or-below counts have something to discriminate.
    premium = discounted_spot * np.concatenate(
        (
            np.zeros(count // 3),
            np.full(count // 3, 1.0e-5),
            np.full(count - 2 * (count // 3), 2.0e-3),
        )
    )
    european = intrinsic + 1.0
    return {
        "sample_id": np.asarray([f"validation-{index:06d}" for index in range(count)]),
        "stratum": np.asarray(["core"] * count),
        "option_type": option_type,
        "spot": spot,
        "strike": strike,
        "maturity": maturity,
        "rate": np.full(count, 0.03),
        "dividend_yield": dividend_yield,
        "volatility": rng.uniform(0.1, 0.6, size=count),
        "american_price": european + premium,
        "european_crr_price": european,
        "early_exercise_premium": premium,
        "intrinsic_value": intrinsic,
        "earliest_exercise_step": np.where(premium > 0.0, 3, -1).astype(np.int64),
    }


def _file_state(path: Path) -> str | None:
    """The digest of ``path``, or ``None`` when it is absent.

    Files under the gitignored ``artifacts/`` tree exist in a working clone that
    has run an exploratory analysis and are absent in a clean checkout, so an
    immutability check has to pin both cases: an existing file keeps its bytes,
    and an absent one stays absent. Hashing unconditionally would instead make
    the invariant untestable exactly where nothing has run yet.
    """
    return sha256_file(path) if path.is_file() else None


# ---------------------------------------------------------------------------
# One partition, fixed in the source
# ---------------------------------------------------------------------------


def test_the_v2_schema_writes_beside_the_v1_report_rather_than_over_it() -> None:
    """An earlier published geometry report is evidence and is never overwritten."""
    assert GEOMETRY_SCHEMA == "american-dev-validation-geometry/2"
    assert DEFAULT_OUTPUT == "artifacts/task-9h/geometry/validation-geometry-v2.json"
    assert str(GEOMETRY_V1_REPORT.relative_to(PROJECT_ROOT)) != DEFAULT_OUTPUT
    # The schema-1 report this clone once held was truncated to zero bytes by the
    # same local incident that destroyed E2's report, so its content can no
    # longer be asserted here and is deliberately not reconstructed. The
    # structural claim above is what this test exists to pin; the content claim
    # runs only where a readable schema-1 report is actually present.
    text = GEOMETRY_V1_REPORT.read_text(encoding="utf-8") if GEOMETRY_V1_REPORT.is_file() else ""
    if text.strip():
        published = json.loads(text)
        assert published["schema_version"] == "american-dev-validation-geometry/1"
        assert "comparator_discrepancy" not in published


def test_the_analysis_partition_is_a_constant_naming_validation() -> None:
    """Not an argument, a flag or a configuration key: a module constant."""
    assert ANALYSIS_PARTITION == "validation"
    tree = ast.parse(GEOMETRY_MODULE.read_text(encoding="utf-8"), filename=str(GEOMETRY_MODULE))
    assigned = [
        node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "ANALYSIS_PARTITION"
        and isinstance(node.value, ast.Constant)
    ]
    assert assigned == ["validation"]


def test_the_analysis_signature_accepts_no_partition_argument() -> None:
    import inspect

    parameters = set(inspect.signature(analyze_validation_geometry).parameters)
    assert parameters == {"project_root", "output", "overwrite"}


def test_the_geometry_surface_exposes_no_final_evaluation_entry_point() -> None:
    import differentiable_pricing.ml.american_dev.geometry as module

    names = {
        name
        for name in dir(module)
        if not name.startswith("__")
        and getattr(getattr(module, name), "__module__", None) == module.__name__
    }
    assert not [name for name in names if "final" in name.lower()]
    assert "analyze_validation_geometry" in names


@pytest.mark.parametrize(
    "output",
    [
        "artifacts/task-9h/interpolation_test/geometry.json",
        "artifacts/task-9h/final/geometry.json",
        "artifacts/holdout.json",
    ],
)
def test_a_final_partition_output_path_fails_closed(tmp_path: Path, output: str) -> None:
    with pytest.raises(FinalPartitionAccessError):
        _guarded_output(tmp_path, output)


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("docs/results/geometry.json", GeometryError),
        ("/etc/geometry.json", AttemptError),
        ("artifacts/../docs/geometry.json", AttemptError),
    ],
)
def test_the_report_must_live_beneath_the_ignored_artifacts_tree(
    tmp_path: Path, output: str, expected: type[Exception]
) -> None:
    """A development observation never lands in the frozen-evidence tree."""
    assert DEFAULT_OUTPUT.startswith("artifacts/")
    with pytest.raises(expected):
        _guarded_output(tmp_path, output)


def test_the_analysis_refuses_to_run_from_a_tree_that_is_not_this_repository(
    tmp_path: Path,
) -> None:
    """It fails on provenance, before anything could open or write a partition."""
    with pytest.raises(AttemptError):
        analyze_validation_geometry(tmp_path)
    assert not (tmp_path / "artifacts").exists()


# ---------------------------------------------------------------------------
# The repository's own definitions, reused rather than restated
# ---------------------------------------------------------------------------


def test_the_slacks_are_exactly_the_repository_definitions() -> None:
    columns = _columns()
    discounted_spot = columns["spot"] * np.exp(-columns["dividend_yield"] * columns["maturity"])
    slacks = normalized_slacks(columns)
    assert set(slacks) == set(MEASURED_QUANTITIES)
    np.testing.assert_allclose(
        slacks["european_slack_normalized"],
        (columns["american_price"] - columns["european_crr_price"]) / discounted_spot,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        slacks["intrinsic_slack_normalized"],
        (columns["american_price"] - columns["intrinsic_value"]) / discounted_spot,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        slacks["early_exercise_premium_normalized"],
        columns["early_exercise_premium"] / discounted_spot,
        rtol=0.0,
        atol=0.0,
    )


def test_a_degenerate_discounted_spot_fails_closed() -> None:
    columns = _columns()
    columns["spot"] = columns["spot"].copy()
    columns["spot"][0] = 0.0
    with pytest.raises(GeometryError, match="A = S"):
        normalized_slacks(columns)


def test_the_populations_are_the_repository_slices() -> None:
    """The bin edges live in one place; this analysis does not restate them."""
    masks = population_masks(_columns(), BINS)
    assert {"overall", "option_type:call", "option_type:put"} <= set(masks)
    assert {"premium_status:zero", "premium_status:positive"} <= set(masks)
    assert {"exercise_status:no_exercise", "exercise_status:exercise_observed"} <= set(masks)
    assert any(name.startswith("expiry:") for name in masks)
    assert any(name.startswith("moneyness:") for name in masks)
    assert any(name.startswith("volatility:") for name in masks)
    assert masks["overall"].all()


def test_the_premium_column_is_checked_against_the_european_slack() -> None:
    """Agreement is asserted at double-rounding level, not by exact equality."""
    consistency = column_consistency(_columns())
    assert consistency["rows"] == 12
    assert consistency["maximum_absolute_difference"] < 1.0e-12
    assert consistency["rows_beyond_double_rounding"] == 0


def test_a_disagreeing_premium_column_is_reported_not_hidden() -> None:
    columns = _columns()
    columns["early_exercise_premium"] = columns["early_exercise_premium"] + 0.5
    consistency = column_consistency(columns)
    assert consistency["maximum_absolute_difference"] == pytest.approx(0.5)
    assert consistency["rows_beyond_double_rounding"] == columns["spot"].size
    assert consistency["maximum_normalized_difference"] > 0.0


def test_the_acceptance_file_is_read_through_a_three_key_whitelist() -> None:
    """The acceptance configuration also names Task 9G's final partition."""
    acceptance = {
        "bins": BINS,
        "quantile_method": "linear",
        "diagnostics": {"material_normalized_tolerance": 1.0e-6, "output_projection": "none"},
        "final_partition": "interpolation_test",
        "validation_final_entry": {"normalized_rmse_max": 0.003},
    }
    view = acceptance_view(acceptance)
    assert sorted(view) == ["bins", "material_normalized_tolerance", "quantile_method"]
    assert view["material_normalized_tolerance"] == 1.0e-6
    assert "interpolation_test" not in json.dumps(view)


@pytest.mark.parametrize(
    "acceptance",
    [
        {"quantile_method": "linear", "diagnostics": {"material_normalized_tolerance": 1e-6}},
        {"bins": BINS, "diagnostics": {"material_normalized_tolerance": 1e-6}},
        {
            "bins": BINS,
            "quantile_method": "",
            "diagnostics": {"material_normalized_tolerance": 1e-6},
        },
        {"bins": BINS, "quantile_method": "linear"},
        {"bins": BINS, "quantile_method": "linear", "diagnostics": {}},
        {
            "bins": BINS,
            "quantile_method": "linear",
            "diagnostics": {"material_normalized_tolerance": 0.0},
        },
    ],
)
def test_an_incomplete_acceptance_file_fails_closed(acceptance: dict[str, Any]) -> None:
    with pytest.raises(GeometryError):
        acceptance_view(acceptance)


def test_the_real_acceptance_file_supplies_the_whitelisted_keys() -> None:
    view = acceptance_view(load_toml(ACCEPTANCE_CONFIG))
    assert view["quantile_method"] == "linear"
    assert view["material_normalized_tolerance"] == 1.0e-6


# ---------------------------------------------------------------------------
# The CRR versus analytic-European comparator discrepancy
# ---------------------------------------------------------------------------


def test_the_comparator_discrepancy_is_crr_minus_analytic_over_A() -> None:
    """Sign convention fixed and checkable: positive means CRR sits above analytic."""
    columns = _columns()
    values = comparator_discrepancy_values(columns)
    analytic = european_price_array(physical_features(columns))
    expected = (
        np.asarray(columns["european_crr_price"], dtype=np.float64) - analytic
    ) / (columns["spot"] * np.exp(-columns["dividend_yield"] * columns["maturity"]))
    np.testing.assert_allclose(values, expected, rtol=0.0, atol=0.0)
    # Raising the stored CRR leg raises the discrepancy by exactly that amount.
    lifted = dict(columns)
    lifted["european_crr_price"] = columns["european_crr_price"] + 1.0
    scale = columns["spot"] * np.exp(-columns["dividend_yield"] * columns["maturity"])
    np.testing.assert_allclose(
        comparator_discrepancy_values(lifted) - values, 1.0 / scale, rtol=1e-12, atol=0.0
    )


def test_the_analytic_leg_is_the_one_the_head_enforces() -> None:
    """Measured and enforced by the same function, so they cannot drift apart."""
    columns = _columns()
    physical = physical_features(columns)
    array = european_price_array(physical)
    with torch.no_grad():
        tensor = european_price(torch.as_tensor(physical, dtype=torch.float64)).numpy()
    np.testing.assert_allclose(array, tensor, rtol=0.0, atol=0.0)


def test_exceedance_counts_are_strictly_above_and_monotone() -> None:
    values = np.asarray([-1.0e-3, 0.0, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3])
    statistics = exceedance_statistics(values, quantile_method="linear")
    counts = statistics["count_above"]
    assert set(counts) == {f"above_{value:g}" for value in DISCREPANCY_THRESHOLDS}
    # Strictly above: the row exactly on the threshold does not count.
    assert counts["above_0"] == 5
    assert counts["above_1e-06"] == 3
    assert counts["above_0.0001"] == 1
    assert list(counts.values()) == sorted(counts.values(), reverse=True)
    assert statistics["fraction_above"]["above_0"] == pytest.approx(5 / 7)
    assert set(statistics["quantiles"]) == {f"{q:g}" for q in DISCREPANCY_QUANTILES}
    assert statistics["quantiles"]["1"] == pytest.approx(1.0e-3)
    assert statistics["rows"] == 7


def test_an_empty_or_non_finite_discrepancy_population_is_handled() -> None:
    empty = exceedance_statistics(np.asarray([]), quantile_method="linear")
    assert empty == {"rows": 0, "quantiles": None, "count_above": None, "fraction_above": None}
    with pytest.raises(GeometryError, match="non-finite"):
        exceedance_statistics(np.asarray([0.0, np.nan]), quantile_method="linear")


def test_effectively_zero_premium_uses_the_acceptance_material_tolerance() -> None:
    """Not a fresh number: the tolerance the violation counts already use."""
    columns = _columns()
    scale = columns["spot"] * np.exp(-columns["dividend_yield"] * columns["maturity"])
    normalized = columns["early_exercise_premium"] / scale
    mask = effectively_zero_premium_mask(columns, 1.0e-6)
    np.testing.assert_array_equal(mask, normalized <= 1.0e-6)
    assert int(mask.sum()) == 4
    # The exact-zero repository slice is a subset of it, never the other way round.
    exact = population_masks(columns, BINS)["premium_status:zero"]
    assert bool((~exact | mask).all())


@pytest.mark.parametrize("tolerance", [0.0, -1.0e-6, float("nan")])
def test_an_invalid_material_tolerance_fails_closed(tolerance: float) -> None:
    with pytest.raises(GeometryError, match="material normalized tolerance"):
        effectively_zero_premium_mask(_columns(), tolerance)


def test_the_discrepancy_block_reports_every_slice_and_the_binding_population() -> None:
    columns = _columns()
    block = comparator_discrepancy(
        columns, BINS, quantile_method="linear", material_tolerance=1.0e-6
    )
    assert block["definition"] == COMPARATOR_DISCREPANCY
    assert "ABOVE" in block["sign_convention"]
    assert "european_price_array" in block["analytic_source"]
    assert block["effectively_zero_premium"]["material_normalized_tolerance"] == 1.0e-6
    assert block["effectively_zero_premium"]["rows"] == 4
    populations = block["populations"]
    assert {"overall", "option_type:call", "option_type:put"} <= set(populations)
    assert "effectively_zero_premium" in populations
    assert populations["overall"]["rows"] == 12
    assert populations["effectively_zero_premium"]["rows"] == 4
    assert set(populations["overall"]["fraction_above"]) == {
        f"above_{value:g}" for value in DISCREPANCY_THRESHOLDS
    }


def test_the_discrepancy_measurement_opens_no_partition_and_writes_nothing(
    tmp_path: Path,
) -> None:
    monitored = [REAL_LOG, GEOMETRY_V1_REPORT]
    before = {path: _file_state(path) for path in monitored}
    comparator_discrepancy(_columns(), BINS, quantile_method="linear", material_tolerance=1e-6)
    # Unchanged where the file was already there, and still not created where it
    # was not: the measurement is a pure read either way.
    assert {path: _file_state(path) for path in monitored} == before
    assert not list(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# Distribution statistics
# ---------------------------------------------------------------------------


def _threshold_set() -> list[dict[str, Any]]:
    return [
        {"name": f"absolute_{value:g}", "value": float(value), "source": "test"}
        for value in ABSOLUTE_THRESHOLDS
    ]


def test_at_or_below_counts_are_inclusive_monotone_and_complete() -> None:
    values = np.asarray([0.0, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2])
    statistics = slack_statistics(values, _threshold_set(), quantile_method="linear")
    counts = statistics["count_at_or_below"]
    assert counts["absolute_1e-06"] == 2  # inclusive: the row exactly on the threshold counts
    assert counts["absolute_0.0001"] == 4
    assert counts["absolute_0.001"] == 5
    assert list(counts.values()) == sorted(counts.values())
    assert statistics["fraction_at_or_below"]["absolute_0.001"] == pytest.approx(5 / 6)
    assert statistics["rows"] == 6
    assert statistics["negative_rows"] == 0
    assert statistics["exactly_zero_rows"] == 1


def test_quantiles_use_the_supplied_method_and_cover_the_range() -> None:
    values = np.linspace(0.0, 1.0, 101)
    statistics = slack_statistics(values, _threshold_set(), quantile_method="linear")
    assert set(statistics["quantiles"]) == {f"{quantile:g}" for quantile in QUANTILES}
    assert statistics["quantiles"]["0.5"] == pytest.approx(0.5)
    assert statistics["quantiles"]["0"] == pytest.approx(statistics["minimum"])
    assert statistics["quantiles"]["1"] == pytest.approx(statistics["maximum"])
    lower = slack_statistics(values, _threshold_set(), quantile_method="lower")
    assert lower["quantiles"]["0.999"] <= statistics["quantiles"]["0.999"]


def test_a_negative_slack_is_counted_not_clipped() -> None:
    """A label-side bound violation would be evidence, so it is never hidden."""
    statistics = slack_statistics(
        np.asarray([-1.0e-9, 0.0, 1.0]), _threshold_set(), quantile_method="linear"
    )
    assert statistics["negative_rows"] == 1
    assert statistics["minimum"] < 0.0


def test_an_empty_population_reports_zero_rows_rather_than_failing() -> None:
    statistics = slack_statistics(np.asarray([]), _threshold_set(), quantile_method="linear")
    assert statistics == {"rows": 0, "quantiles": None, "count_at_or_below": None}


def test_a_non_finite_slack_fails_closed() -> None:
    with pytest.raises(GeometryError, match="non-finite"):
        slack_statistics(
            np.asarray([0.0, np.inf]), _threshold_set(), quantile_method="linear"
        )


def test_every_population_carries_every_measured_quantity() -> None:
    populations = geometry_populations(
        _columns(), BINS, _threshold_set(), quantile_method="linear"
    )
    assert populations["overall"]["rows"] == 12
    for entry in populations.values():
        assert set(entry) == {"rows", *MEASURED_QUANTITIES}
    positive = populations["premium_status:positive"]
    assert positive["rows"] == 8
    assert positive["early_exercise_premium_normalized"]["minimum"] > 0.0
    assert populations["premium_status:zero"]["european_slack_normalized"][
        "exactly_zero_rows"
    ] == 4


# ---------------------------------------------------------------------------
# Thresholds come from the recorded attempts, not from a typed-in number
# ---------------------------------------------------------------------------


def test_the_reference_rmses_are_read_out_of_the_tracked_attempt_log() -> None:
    entries = attempt_rmse_thresholds(REAL_LOG)
    assert [entry["name"] for entry in entries] == [name for name, _ in REFERENCE_ATTEMPTS]
    recorded = {
        json.loads(line)["attempt_id"]: json.loads(line)
        for line in REAL_LOG.read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("record") == "attempt"
    }
    for (name, attempt_id), entry in zip(REFERENCE_ATTEMPTS, entries, strict=True):
        assert entry["name"] == name
        assert entry["value"] == recorded[attempt_id]["price_metrics"]["overall_normalized"]["rmse"]
        assert attempt_id in entry["source"]


def test_the_reference_rmses_appear_nowhere_as_a_literal_in_the_module() -> None:
    """A threshold typed in by hand would drift from the attempt it represents."""
    text = GEOMETRY_MODULE.read_text(encoding="utf-8")
    for entry in attempt_rmse_thresholds(REAL_LOG):
        assert repr(entry["value"]) not in text


def test_a_missing_reference_attempt_fails_closed(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    log.write_text(
        json.dumps({"record": "header", "schema_version": "american-dev-attempt-log/1"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(GeometryError, match="not recorded"):
        attempt_rmse_thresholds(log)


def test_a_reference_attempt_without_an_rmse_fails_closed(tmp_path: Path) -> None:
    log = tmp_path / "log.jsonl"
    lines = [json.dumps({"record": "header", "schema_version": "american-dev-attempt-log/1"})]
    for _, attempt_id in REFERENCE_ATTEMPTS:
        lines.append(json.dumps({"record": "attempt", "attempt_id": attempt_id}))
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(GeometryError, match="records no overall normalized RMSE"):
        attempt_rmse_thresholds(log)


def test_the_threshold_set_is_the_three_scales_then_the_three_attempts() -> None:
    entries = thresholds(REAL_LOG)
    assert len(entries) == len(ABSOLUTE_THRESHOLDS) + len(REFERENCE_ATTEMPTS)
    assert [entry["value"] for entry in entries[: len(ABSOLUTE_THRESHOLDS)]] == list(
        ABSOLUTE_THRESHOLDS
    )
    assert all(entry["source"] for entry in entries)


# ---------------------------------------------------------------------------
# It trains nothing and mutates no attempt evidence
# ---------------------------------------------------------------------------


def test_the_analysis_module_contains_no_training_machinery() -> None:
    tree = ast.parse(GEOMETRY_MODULE.read_text(encoding="utf-8"), filename=str(GEOMETRY_MODULE))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert "torch" not in {name.split(".")[0] for name in imported}
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert not called & {"train_model", "build_model", "execute_attempt", "build_network"}


def test_the_analysis_module_never_writes_the_attempt_log() -> None:
    """It reads the log for thresholds; it is not a second writer of it."""
    tree = ast.parse(GEOMETRY_MODULE.read_text(encoding="utf-8"), filename=str(GEOMETRY_MODULE))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "append_attempt" not in called
    assert "attempt_log_entries" in called


def test_reading_the_thresholds_leaves_every_byte_of_evidence_untouched() -> None:
    tracked = [REAL_LOG, *sorted(PROJECT_ROOT.glob("configs/american_dev_attempt_*.toml"))]
    before = {path: sha256_file(path) for path in tracked}
    attempt_rmse_thresholds(REAL_LOG)
    thresholds(REAL_LOG)
    assert {path: sha256_file(path) for path in tracked} == before


# ---------------------------------------------------------------------------
# The report says what it is
# ---------------------------------------------------------------------------


def test_the_exploratory_label_refuses_the_word_result() -> None:
    assert "not a project result" in EXPLORATORY_LABEL
    assert "validation" in EXPLORATORY_LABEL


def test_the_compact_view_reports_only_numbers_the_report_carries() -> None:
    columns = _columns()
    threshold_set = _threshold_set()
    report = {
        "schema_version": GEOMETRY_SCHEMA,
        "partition_analyzed": ANALYSIS_PARTITION,
        "repository": {"commit": "0" * 40},
        "thresholds": threshold_set,
        "populations": geometry_populations(
            columns, BINS, threshold_set, quantile_method="linear"
        ),
        "comparator_discrepancy": comparator_discrepancy(
            columns, BINS, quantile_method="linear", material_tolerance=1.0e-6
        ),
    }
    compact = compact_geometry(report)
    assert compact["partition_analyzed"] == "validation"
    assert compact["rows"] == 12
    assert set(compact["overall"]) == set(MEASURED_QUANTITIES)
    assert compact["premium_status:positive"]["rows"] == 8
    assert compact["selection_bias"] == EXPLORATORY_LABEL
    assert compact["comparator_discrepancy"]["definition"] == COMPARATOR_DISCREPANCY
    assert compact["comparator_discrepancy"]["effectively_zero_premium"]["rows"] == 4
    assert compact["overall"]["european_slack_normalized"]["median"] == (
        report["populations"]["overall"]["european_slack_normalized"]["quantiles"]["0.5"]
    )


def test_the_script_declares_exactly_analyze_and_show() -> None:
    tree = ast.parse(GEOMETRY_SCRIPT.read_text(encoding="utf-8"), filename=str(GEOMETRY_SCRIPT))
    declared: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "COMMANDS"
            and node.value is not None
        ):
            declared = {
                element.value
                for element in getattr(node.value, "elts", [])
                if isinstance(element, ast.Constant)
            }
    assert declared == {"analyze", "show"}
