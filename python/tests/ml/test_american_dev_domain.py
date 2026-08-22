"""Task 9H European CRR-versus-Black-Scholes domain characterization.

Synthetic fixtures and tracked configuration throughout. **No dataset partition
is opened, no model is trained, and the analysis entry point is exercised only
along its refusal paths and its pure helpers**, which is where its guarantees
live. The full measurement is a manual, human-invoked run of hundreds of
thousands of lattices and is never invoked from a test.

The properties pinned here are the ones a margin derivation can quietly lose:
that the margin rule is predeclared in code and has no free parameter, that the
sample is deterministic and comes from the declared domain rather than from any
partition, that the CRR comparator reproduces the label policy's own semantics
at the label policy's own resolution, that the analytic leg is the one the
deployed floor enforces, and that nothing here can reach or name a partition.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest
from differentiable_pricing import _core
from differentiable_pricing.ml.american_dev.attempts import (
    AttemptError,
    FinalPartitionAccessError,
    load_toml,
    sha256_file,
)
from differentiable_pricing.ml.american_dev.domain import (
    DATASET_CONFIG_PATH,
    DEFAULT_OUTPUT,
    DIFFICULT_NEAR_MONEY_COUNT,
    DOMAIN_COORDINATES,
    DOMAIN_SCHEMA,
    EXCEEDANCE_THRESHOLDS,
    EXPECTED_LABEL_POLICY,
    HALTON_BASES,
    HALTON_SKIP,
    MARGIN_QUANTUM,
    MARGIN_RULE,
    MARGIN_SAFETY_FACTOR,
    MINIMUM_MARGIN,
    OPTION_TYPES,
    SAMPLE_POINTS,
    DomainAnalysisError,
    _guarded_output,
    assert_domain_consistent,
    ceil_to_quantum,
    characterize_european_comparator,
    corner_coordinates,
    declared_domain,
    derive_margin,
    difficult_coordinates,
    domain_states,
    european_crr_adjacent_average,
    face_coordinates,
    gap_statistics,
    halton_points,
    label_policy,
    locked_tracked_input_digest,
    normalized_comparator_gap,
    radical_inverse,
    scale_to_domain,
)
from differentiable_pricing.ml.american_dev.representation import european_price_array
from differentiable_pricing.ml.config import FEATURE_ORDER

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DOMAIN_MODULE = (
    PROJECT_ROOT / "python/src/differentiable_pricing/ml/american_dev/domain.py"
)
DOMAIN_SCRIPT = PROJECT_ROOT / "scripts/analyze_american_dev_domain.py"
PROTOCOL = PROJECT_ROOT / "configs/american_neural_pilot_protocol_v1.toml"
ACCEPTANCE = PROJECT_ROOT / "configs/american_neural_pilot_acceptance_v1.toml"


@pytest.fixture(scope="module")
def bounds() -> dict[str, tuple[float, float]]:
    return declared_domain(load_toml(PROJECT_ROOT / DATASET_CONFIG_PATH))


# ---------------------------------------------------------------------------
# The predeclared margin rule
# ---------------------------------------------------------------------------


def test_the_margin_rule_is_three_constants_and_one_implementation() -> None:
    """Predeclared in code, before the run, with nothing left to tune."""
    assert MARGIN_SAFETY_FACTOR == 2.0
    assert MARGIN_QUANTUM == 1.0e-5
    assert MINIMUM_MARGIN == 1.0e-4
    assert "ceil_to_1e-5(2 * domain_supremum)" in MARGIN_RULE
    assert "max(1e-4, candidate)" in MARGIN_RULE
    tree = ast.parse(DOMAIN_MODULE.read_text(encoding="utf-8"), filename=str(DOMAIN_MODULE))
    functions = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "derive_margin"
    ]
    assert functions == ["derive_margin"]


@pytest.mark.parametrize(
    ("supremum", "candidate", "delta", "binding"),
    [
        (0.0, 0.0, 1.0e-4, "minimum"),
        (1.0e-6, 1.0e-5, 1.0e-4, "minimum"),
        (3.75e-5, 8.0e-5, 1.0e-4, "minimum"),
        (5.0e-5, 1.0e-4, 1.0e-4, "candidate"),
        (1.0e-3, 2.0e-3, 2.0e-3, "candidate"),
    ],
)
def test_the_rule_applies_exactly_as_written(
    supremum: float, candidate: float, delta: float, binding: str
) -> None:
    margin = derive_margin(supremum)
    assert margin["domain_supremum"] == pytest.approx(supremum)
    assert margin["candidate"] == pytest.approx(candidate)
    assert margin["delta"] == pytest.approx(delta)
    assert margin["binding_term"] == binding
    assert margin["rule"] == MARGIN_RULE


def test_a_domain_with_no_positive_gap_still_gets_the_declared_minimum() -> None:
    """``domain_supremum`` is the maximum *positive* gap, so it never goes negative."""
    margin = derive_margin(-4.0e-5)
    assert margin["measured_maximum_gap"] == -4.0e-5
    assert margin["domain_supremum"] == 0.0
    assert margin["delta"] == MINIMUM_MARGIN
    assert margin["realized_safety_factor"] is None


def test_the_realized_safety_factor_is_delta_over_the_supremum() -> None:
    margin = derive_margin(2.0e-5)
    assert margin["realized_safety_factor"] == pytest.approx(
        margin["delta"] / margin["domain_supremum"]
    )
    assert margin["realized_safety_factor"] >= MARGIN_SAFETY_FACTOR


@pytest.mark.parametrize("value", [1.0e-5, 2.0e-5, 1.0e-4])
def test_the_ceiling_leaves_exact_multiples_alone(value: float) -> None:
    assert ceil_to_quantum(value) == pytest.approx(value)


def test_the_ceiling_rounds_up_and_clamps_at_zero() -> None:
    assert ceil_to_quantum(1.0e-6) == pytest.approx(1.0e-5)
    assert ceil_to_quantum(1.1e-5) == pytest.approx(2.0e-5)
    assert ceil_to_quantum(0.0) == 0.0
    assert ceil_to_quantum(-1.0) == 0.0


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_a_non_finite_measurement_fails_closed(value: float) -> None:
    with pytest.raises(DomainAnalysisError):
        derive_margin(value)


# ---------------------------------------------------------------------------
# The declared domain, read through a whitelist
# ---------------------------------------------------------------------------


def test_the_declared_domain_is_the_one_the_locked_protocol_pins() -> None:
    protocol = load_toml(PROTOCOL)
    pinned = locked_tracked_input_digest(protocol, DATASET_CONFIG_PATH)
    assert sha256_file(PROJECT_ROOT / DATASET_CONFIG_PATH) == pinned


def test_an_unpinned_tracked_input_fails_closed() -> None:
    with pytest.raises(DomainAnalysisError):
        locked_tracked_input_digest(load_toml(PROTOCOL), "configs/not_a_tracked_input.toml")


def test_the_domain_is_read_as_six_finite_increasing_intervals(
    bounds: dict[str, tuple[float, float]],
) -> None:
    assert tuple(bounds) == DOMAIN_COORDINATES
    for low, high in bounds.values():
        assert math.isfinite(low) and math.isfinite(high) and high > low
    assert bounds["spot"] == (50.0, 150.0)
    assert bounds["log_moneyness"] == (-0.7, 0.7)


def test_the_per_partition_row_table_is_never_read() -> None:
    """``[domain]`` and nothing else: the row counts are not resolved."""
    config = load_toml(PROJECT_ROOT / DATASET_CONFIG_PATH)
    assert "rows" in config["dataset"]
    reduced = {"domain": config["domain"]}
    assert declared_domain(reduced) == declared_domain(config)


@pytest.mark.parametrize(
    "section",
    [
        {},
        {"spot": [50.0]},
        {"spot": [150.0, 50.0]},
        {"spot": "50 to 150"},
    ],
)
def test_a_malformed_domain_fails_closed(section: dict[str, object]) -> None:
    with pytest.raises(DomainAnalysisError):
        declared_domain({"domain": section})


def test_the_label_policy_and_its_resolution_come_from_the_configuration() -> None:
    policy, steps = label_policy(load_toml(PROJECT_ROOT / DATASET_CONFIG_PATH))
    assert policy == EXPECTED_LABEL_POLICY
    assert steps == 1024


def test_another_label_policy_is_refused() -> None:
    with pytest.raises(DomainAnalysisError):
        label_policy({"label": {"policy": "some-other-policy/1", "steps": 1024}})


def test_the_acceptance_diagnostics_domain_must_agree(
    bounds: dict[str, tuple[float, float]],
) -> None:
    checked = assert_domain_consistent(bounds, load_toml(ACCEPTANCE))
    assert checked["spot_domain"] == [50.0, 150.0]
    assert checked["volatility_domain"] == [0.05, 0.8]
    assert checked["log_moneyness_domain"] == [-0.7, 0.7]


def test_a_drifted_acceptance_domain_is_refused(
    bounds: dict[str, tuple[float, float]],
) -> None:
    drifted = {
        "diagnostics": {
            "spot_domain": [50.0, 140.0],
            "volatility_domain": [0.05, 0.8],
            "log_moneyness_domain": [-0.7, 0.7],
        }
    }
    with pytest.raises(DomainAnalysisError):
        assert_domain_consistent(bounds, drifted)


# ---------------------------------------------------------------------------
# The deterministic sample
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("index", "base", "expected"),
    [(1, 2, 0.5), (2, 2, 0.25), (3, 2, 0.75), (4, 2, 0.125), (1, 3, 1.0 / 3.0), (0, 5, 0.0)],
)
def test_the_radical_inverse_is_the_textbook_definition(
    index: int, base: int, expected: float
) -> None:
    assert radical_inverse(np.asarray([index]), base)[0] == pytest.approx(expected)


def test_a_negative_index_or_degenerate_base_fails_closed() -> None:
    with pytest.raises(DomainAnalysisError):
        radical_inverse(np.asarray([-1]), 2)
    with pytest.raises(DomainAnalysisError):
        radical_inverse(np.asarray([1]), 1)


def test_the_sample_is_deterministic_and_inside_the_unit_cube() -> None:
    first = halton_points(256)
    second = halton_points(256)
    assert np.array_equal(first, second)
    assert first.shape == (256, len(HALTON_BASES))
    assert bool((first >= 0.0).all()) and bool((first < 1.0).all())


def test_the_sample_skips_the_correlated_leading_points() -> None:
    """The first returned point is the one at index ``HALTON_SKIP``."""
    expected = [radical_inverse(np.asarray([HALTON_SKIP]), base)[0] for base in HALTON_BASES]
    assert halton_points(1)[0].tolist() == pytest.approx(expected)


def test_the_declared_sample_size_is_at_least_the_predeclared_budget() -> None:
    assert SAMPLE_POINTS == 131_072
    assert SAMPLE_POINTS * len(OPTION_TYPES) >= 262_144


def test_unit_points_map_onto_the_declared_endpoints(
    bounds: dict[str, tuple[float, float]],
) -> None:
    unit = np.asarray([[0.0] * len(DOMAIN_COORDINATES), [1.0] * len(DOMAIN_COORDINATES)])
    scaled = scale_to_domain(unit, bounds)
    for name in DOMAIN_COORDINATES:
        assert scaled[name][0] == pytest.approx(bounds[name][0])
        assert scaled[name][1] == pytest.approx(bounds[name][1])


def test_the_corner_set_is_every_vertex_of_the_declared_box(
    bounds: dict[str, tuple[float, float]],
) -> None:
    corners = corner_coordinates(bounds)
    rows = np.column_stack([corners[name] for name in DOMAIN_COORDINATES])
    assert rows.shape == (2 ** len(DOMAIN_COORDINATES), len(DOMAIN_COORDINATES))
    assert len({tuple(row) for row in rows.tolist()}) == rows.shape[0]
    for index, name in enumerate(DOMAIN_COORDINATES):
        assert set(np.unique(rows[:, index]).tolist()) == set(bounds[name])


def test_each_face_pins_exactly_one_coordinate_to_an_endpoint(
    bounds: dict[str, tuple[float, float]],
) -> None:
    faces = face_coordinates(bounds)
    rows = np.column_stack([faces[name] for name in DOMAIN_COORDINATES])
    assert rows.shape == (2 * len(DOMAIN_COORDINATES), len(DOMAIN_COORDINATES))
    for row in rows.tolist():
        pinned = [
            index
            for index, name in enumerate(DOMAIN_COORDINATES)
            if row[index] in bounds[name]
        ]
        assert pinned


def test_the_difficult_sweep_stays_inside_the_domain_and_reaches_the_money(
    bounds: dict[str, tuple[float, float]],
) -> None:
    sweep = difficult_coordinates(bounds)
    for name in DOMAIN_COORDINATES:
        low, high = bounds[name]
        assert float(sweep[name].min()) >= low
        assert float(sweep[name].max()) <= high
    moneyness = np.unique(sweep["log_moneyness"])
    assert 0.0 in moneyness.tolist()
    assert bounds["log_moneyness"][0] in moneyness.tolist()
    assert bounds["log_moneyness"][1] in moneyness.tolist()
    assert (np.abs(moneyness) <= 0.02).sum() >= DIFFICULT_NEAR_MONEY_COUNT
    assert float(np.unique(sweep["maturity"]).min()) == pytest.approx(bounds["maturity"][0])


def test_every_construction_is_evaluated_at_both_option_types(
    bounds: dict[str, tuple[float, float]],
) -> None:
    states = domain_states(bounds, sample_points=32)
    assert set(states["counts"]) == {
        "low_discrepancy",
        "difficult_regions",
        "corners",
        "faces",
    }
    assert states["counts"]["low_discrepancy"] == 32 * len(OPTION_TYPES)
    assert states["counts"]["corners"] == 2 ** len(DOMAIN_COORDINATES) * len(OPTION_TYPES)
    assert states["total"] == sum(states["counts"].values())
    assert set(np.unique(states["option_type"]).tolist()) == set(OPTION_TYPES)


def test_the_states_are_physical_features_in_the_repository_order(
    bounds: dict[str, tuple[float, float]],
) -> None:
    states = domain_states(bounds, sample_points=16)
    physical = states["physical"]
    assert physical.shape[1] == len(FEATURE_ORDER)
    encoded = np.where(states["option_type"] == "call", 1.0, -1.0)
    assert np.array_equal(physical[:, FEATURE_ORDER.index("option_type")], encoded)
    spot = physical[:, FEATURE_ORDER.index("spot")]
    strike = physical[:, FEATURE_ORDER.index("strike")]
    assert bool((spot > 0.0).all()) and bool((strike > 0.0).all())
    # strike = spot * exp(-log_moneyness), so log(spot/strike) recovers it.
    assert bool(np.isfinite(np.log(spot / strike)).all())
    # The analytic leg the deployed floor enforces accepts every constructed state.
    assert bool(np.isfinite(european_price_array(physical)).all())


# ---------------------------------------------------------------------------
# The two European legs
# ---------------------------------------------------------------------------


def _small_states() -> dict[str, np.ndarray]:
    physical = np.asarray(
        [
            [1.0, 100.0, 105.0, 0.5, 0.03, 0.01, 0.25],
            [-1.0, 95.0, 100.0, 1.0, 0.04, 0.02, 0.20],
            [1.0, 120.0, 100.0, 2.0, -0.01, 0.05, 0.35],
        ],
        dtype=np.float64,
    )
    return {
        "physical": physical,
        "option_type": np.asarray(["call", "put", "call"], dtype=object),
    }


def test_the_comparator_is_the_label_policy_adjacent_average() -> None:
    """``0.5 * (E_CRR(N) + E_CRR(N+1))``, against the scalar engine directly."""
    states = _small_states()
    steps = 128
    measured = european_crr_adjacent_average(
        states["physical"], states["option_type"], steps, thread_count=1
    )
    for index, row in enumerate(states["physical"]):
        arguments = (
            str(states["option_type"][index]),
            "european",
            float(row[1]),
            float(row[2]),
            float(row[3]),
            float(row[4]),
            float(row[5]),
            float(row[6]),
        )
        expected = 0.5 * (
            float(_core.crr_price(*arguments, steps)["price"])
            + float(_core.crr_price(*arguments, steps + 1)["price"])
        )
        assert measured[index] == pytest.approx(expected, rel=0.0, abs=0.0)


def test_the_comparator_is_independent_of_threads_and_chunking() -> None:
    """Both are throughput settings; neither may move a value."""
    states = _small_states()
    baseline = european_crr_adjacent_average(
        states["physical"], states["option_type"], 96, thread_count=1, chunk_rows=1
    )
    other = european_crr_adjacent_average(
        states["physical"], states["option_type"], 96, thread_count=4, chunk_rows=8192
    )
    assert np.array_equal(baseline, other)


@pytest.mark.parametrize("steps", [0, -1, 2.5])
def test_a_degenerate_crr_depth_fails_closed(steps: object) -> None:
    states = _small_states()
    with pytest.raises(DomainAnalysisError):
        european_crr_adjacent_average(states["physical"], states["option_type"], steps)  # type: ignore[arg-type]


def test_the_gap_is_crr_minus_analytic_over_the_discounted_spot() -> None:
    states = _small_states()
    physical = states["physical"]
    crr = european_crr_adjacent_average(physical, states["option_type"], 64, thread_count=1)
    analytic = european_price_array(physical)
    gap = normalized_comparator_gap(physical, crr, analytic)
    scale = physical[:, 1] * np.exp(-physical[:, 5] * physical[:, 3])
    assert np.array_equal(gap, (crr - analytic) / scale)


def test_the_normalized_gap_is_invariant_under_a_common_scale() -> None:
    """Both legs are homogeneous of degree one, so ``gap / A`` is scale-free."""
    states = _small_states()
    physical = states["physical"]
    scaled = physical.copy()
    scaled[:, 1] *= 1.5
    scaled[:, 2] *= 1.5
    values = []
    for candidate in (physical, scaled):
        crr = european_crr_adjacent_average(
            candidate, states["option_type"], 64, thread_count=1
        )
        values.append(normalized_comparator_gap(candidate, crr, european_price_array(candidate)))
    assert values[0] == pytest.approx(values[1], rel=1.0e-12, abs=1.0e-15)


def test_a_degenerate_discounted_spot_fails_closed() -> None:
    physical = np.asarray([[1.0, 0.0, 100.0, 1.0, 0.03, 0.0, 0.25]], dtype=np.float64)
    with pytest.raises(DomainAnalysisError):
        normalized_comparator_gap(physical, np.asarray([1.0]), np.asarray([1.0]))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def test_exceedance_counts_are_strictly_above_and_monotone() -> None:
    values = np.asarray([-1.0e-5, 0.0, 5.0e-7, 5.0e-6, 5.0e-5, 5.0e-4])
    statistics = gap_statistics(values, quantile_method="linear")
    counts = statistics["count_above"]
    assert counts["above_0"] == 4
    assert counts["above_1e-06"] == 3
    assert counts["above_1e-05"] == 2
    assert counts["above_0.0001"] == 1
    ordered = [counts[f"above_{threshold:g}"] for threshold in EXCEEDANCE_THRESHOLDS]
    assert ordered == sorted(ordered, reverse=True)
    assert statistics["maximum"] == pytest.approx(5.0e-4)
    assert statistics["quantiles"]["1"] == pytest.approx(5.0e-4)
    assert statistics["fraction_above"]["above_0"] == pytest.approx(4 / 6)


@pytest.mark.parametrize("values", [np.zeros(0), np.zeros((2, 2))])
def test_a_degenerate_gap_vector_fails_closed(values: np.ndarray) -> None:
    with pytest.raises(DomainAnalysisError):
        gap_statistics(values, quantile_method="linear")


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "output",
    [
        "artifacts/task-9h/interpolation_test/domain.json",
        "artifacts/task-9h/final/domain.json",
        "artifacts/holdout.json",
    ],
)
def test_a_final_partition_output_path_fails_closed(tmp_path: Path, output: str) -> None:
    with pytest.raises(FinalPartitionAccessError):
        _guarded_output(tmp_path, output)


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("docs/results/domain.json", DomainAnalysisError),
        ("/etc/domain.json", AttemptError),
        ("artifacts/../docs/domain.json", AttemptError),
    ],
)
def test_the_report_must_live_beneath_the_ignored_artifacts_tree(
    tmp_path: Path, output: str, expected: type[Exception]
) -> None:
    assert DEFAULT_OUTPUT.startswith("artifacts/")
    with pytest.raises(expected):
        _guarded_output(tmp_path, output)


def test_the_analysis_refuses_to_run_from_a_tree_that_is_not_this_repository(
    tmp_path: Path,
) -> None:
    """It fails on provenance, before anything could price a lattice or write."""
    with pytest.raises(AttemptError):
        characterize_european_comparator(tmp_path)
    assert not (tmp_path / "artifacts").exists()


def test_the_analysis_refuses_a_degenerate_thread_count(tmp_path: Path) -> None:
    with pytest.raises((AttemptError, DomainAnalysisError)):
        characterize_european_comparator(tmp_path, thread_count=0)


def test_the_domain_surface_exposes_no_final_evaluation_entry_point() -> None:
    import differentiable_pricing.ml.american_dev.domain as module

    names = {
        name
        for name in dir(module)
        if not name.startswith("__")
        and getattr(getattr(module, name), "__module__", None) == module.__name__
    }
    assert not [name for name in names if "final" in name.lower()]
    assert "characterize_european_comparator" in names
    assert DOMAIN_SCHEMA == "american-dev-european-comparator-domain/1"


def test_the_analysis_imports_no_partition_machinery() -> None:
    """It reads no partition, so it does not even reach the loader that could."""
    text = DOMAIN_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(DOMAIN_MODULE))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imported.update(alias.name for alias in node.names)
    assert "load_partition" not in imported
    assert "verify_partition_policy" not in imported
    assert "read_partition_columns" not in imported


def test_the_script_declares_exactly_two_commands_and_no_evaluation() -> None:
    text = DOMAIN_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(DOMAIN_SCRIPT))
    declared: set[str] = set()
    for node in ast.walk(tree):
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target = node.targets[0].id
        if target == "COMMANDS" and node.value is not None:
            declared = {
                element.value
                for element in getattr(node.value, "elts", [])
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            }
    assert declared == {"analyze", "show"}
    for forbidden in ("final-evaluate", "final_evaluate"):
        assert f'"{forbidden}"' not in text and f"'{forbidden}'" not in text
