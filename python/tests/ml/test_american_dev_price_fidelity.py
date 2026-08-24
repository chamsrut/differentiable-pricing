"""Task 9H price fidelity: the 2x2 decomposition and the B.3 characterization.

Runs against a small synthetic row set whose labels are genuine adjacent-averaged
CRR prices, so the reused slice, bound and shape machinery is exercised for real.
No dataset partition is opened, nothing is trained and no checkpoint is read.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import differentiable_pricing as dp
import numpy as np
import pytest
import torch
from differentiable_pricing.ml.american_dev.attempts import (
    EUROPEAN_FLOOR_MARGIN,
    load_toml,
)
from differentiable_pricing.ml.american_dev.eligibility import assess_rows
from differentiable_pricing.ml.american_dev.frozen import (
    FrozenCheckpointError,
    MarginOverrideModel,
    RawNetworkModel,
    declaration_state,
)
from differentiable_pricing.ml.american_dev.models import build_network
from differentiable_pricing.ml.american_dev.price_fidelity import (
    BOUND_FAMILIES,
    DELTA_EFFECT_PREDICTION,
    FLOOR_DOMINATED_MAXIMUM,
    MARGIN_CELLS,
    NETWORK_DOMINATED_MINIMUM,
    SHAPE_FAMILIES,
    cell_metrics,
    characterize,
    decompose,
    diagnostic_breakdown,
    projection_partition,
    signed_bias,
)
from differentiable_pricing.ml.american_dev.representation import AmericanDevPriceModel
from differentiable_pricing.ml.american_dev.tolerance import GATED_SLICE_FAMILY
from differentiable_pricing.ml.american_pilot import (
    physical_features,
    predict_prices,
    shape_diagnostics,
)
from differentiable_pricing.ml.model import Scaling

PROJECT_ROOT: Final = Path(__file__).resolve().parents[3]
ACCEPTANCE: Final = PROJECT_ROOT / "configs/american_neural_pilot_acceptance_v1.toml"
#: Small enough to stay fast, wide enough that every gated slice is non-empty.
ROWS: Final = 240
#: The label policy's own resolution, priced as the adjacent average.
DEPTH: Final = 256


@pytest.fixture(scope="module")
def acceptance() -> dict[str, Any]:
    return load_toml(ACCEPTANCE)


@pytest.fixture(scope="module")
def columns() -> dict[str, np.ndarray]:
    """Synthetic contracts with genuine adjacent-averaged CRR labels.

    A shallower depth than the dataset's is used deliberately: this exercises
    the analysis, not the label policy, and the analysis is indifferent to the
    depth its reference came from.
    """
    generator = np.random.default_rng(20260823)
    spot = generator.uniform(60.0, 140.0, ROWS)
    strike = spot * np.exp(-generator.uniform(-0.6, 0.6, ROWS))
    maturity = generator.uniform(0.05, 2.5, ROWS)
    rate = generator.uniform(-0.01, 0.10, ROWS)
    dividend_yield = generator.uniform(0.0, 0.10, ROWS)
    volatility = generator.uniform(0.08, 0.70, ROWS)
    option_type = generator.choice(["call", "put"], ROWS)

    def priced(style: str, steps: int) -> np.ndarray:
        return np.asarray(
            dp.crr_price_batch(
                list(option_type),
                [style] * ROWS,
                list(spot),
                list(strike),
                list(maturity),
                list(rate),
                list(dividend_yield),
                list(volatility),
                [steps] * ROWS,
                4,
            )["price"],
            dtype=np.float64,
        )

    american = 0.5 * (priced("american", DEPTH) + priced("american", DEPTH + 1))
    european = 0.5 * (priced("european", DEPTH) + priced("european", DEPTH + 1))
    omega = np.where(option_type == "call", 1.0, -1.0)
    premium = american - european
    return {
        "sample_id": np.array([f"row-{index:05d}" for index in range(ROWS)], dtype=object),
        "stratum": np.array(["synthetic"] * ROWS, dtype=object),
        "option_type": option_type,
        "spot": spot,
        "strike": strike,
        "maturity": maturity,
        "rate": rate,
        "dividend_yield": dividend_yield,
        "volatility": volatility,
        "american_price": american,
        "european_crr_price": european,
        "early_exercise_premium": premium,
        "intrinsic_value": np.maximum(omega * (spot - strike), 0.0),
        "earliest_exercise_step": np.where(premium > 1.0e-8, 10, -1).astype(np.int64),
    }


def _model(head: str, seed: int) -> AmericanDevPriceModel:
    torch.manual_seed(seed)
    network = build_network(
        {"name": "smooth_residual", "width": 32, "blocks": 2, "activation": "tanh"}, 5
    )
    scaling = Scaling(
        feature_mean=np.zeros(5, dtype=np.float64),
        feature_scale=np.ones(5, dtype=np.float64),
        price_mean=0.05,
        price_scale=0.05,
    )
    model = AmericanDevPriceModel(network, scaling, head=head)
    model.eval()
    return model


@pytest.fixture(scope="module")
def eligibility_record(columns: dict[str, np.ndarray]) -> dict[str, Any]:
    """Assessability for these rows, computed before any prediction exists.

    Built from ``assess_rows`` plus the current declaration state rather than
    from ``assess``, because the strict source-state attestation demands a clean
    worktree and a unit test must not have to commit to run.
    """
    return {
        **assess_rows("validation", columns),
        "provenance": declaration_state(PROJECT_ROOT),
    }


@pytest.fixture(scope="module")
def models() -> dict[str, AmericanDevPriceModel]:
    return {
        "E2b": _model("smooth_lower_floor_raw_loss", 1),
        "E2c": _model("smooth_lower_floor_margin_raw_loss", 2),
    }


# ---------------------------------------------------------------------------
# The views drive the reused diagnostics, bumped states included
# ---------------------------------------------------------------------------


def test_a_native_margin_view_reproduces_the_model_it_wraps(
    columns: dict[str, np.ndarray], models: dict[str, AmericanDevPriceModel]
) -> None:
    model = models["E2c"]
    physical = physical_features(columns)
    native = predict_prices(model, physical, 128)
    through_view = predict_prices(MarginOverrideModel(model, model.european_margin), physical, 128)
    assert np.array_equal(native, through_view)


def test_the_shape_diagnostic_bumps_go_through_the_overridden_margin(
    columns: dict[str, np.ndarray],
    models: dict[str, AmericanDevPriceModel],
    acceptance: dict[str, Any],
) -> None:
    """The reason a view is used instead of recomputing only the centre price.

    A shortcut that overrode the margin for the centre prediction alone would
    leave the bumped states on the model's native floor, and the shape counts
    would silently describe a floor nobody deployed.
    """
    model = models["E2c"]
    native, _ = shape_diagnostics(
        MarginOverrideModel(model, EUROPEAN_FLOOR_MARGIN), columns, acceptance, batch_size=128
    )
    zero, _ = shape_diagnostics(
        MarginOverrideModel(model, 0.0), columns, acceptance, batch_size=128
    )
    assert native != zero


def test_a_non_floor_head_cannot_be_given_a_margin_view() -> None:
    with pytest.raises(FrozenCheckpointError):
        MarginOverrideModel(_model("direct", 3), 0.0)


@pytest.mark.parametrize("margin", [-1.0e-9, float("nan"), float("inf")])
def test_an_unusable_margin_is_refused(margin: float) -> None:
    with pytest.raises(FrozenCheckpointError):
        MarginOverrideModel(_model("smooth_lower_floor_raw_loss", 4), margin)


def test_the_raw_view_is_the_unprojected_function(
    columns: dict[str, np.ndarray], models: dict[str, AmericanDevPriceModel]
) -> None:
    """The raw price carries no floor, so it may sit below the deployed one."""
    model = models["E2c"]
    physical = physical_features(columns)
    raw = predict_prices(RawNetworkModel(model), physical, 128)
    deployed = predict_prices(MarginOverrideModel(model, model.european_margin), physical, 128)
    assert bool((deployed >= raw - 1.0e-12).all())
    assert bool((deployed > raw + 1.0e-9).any())


# ---------------------------------------------------------------------------
# B.0
# ---------------------------------------------------------------------------


def test_signed_bias_keeps_its_sign() -> None:
    assert signed_bias(np.array([1.0, 2.0]), np.array([1.5, 2.5])) == pytest.approx(0.5)
    assert signed_bias(np.array([1.0, 2.0]), np.array([0.5, 1.5])) == pytest.approx(-0.5)


def test_the_decomposition_produces_all_four_cells(
    columns: dict[str, np.ndarray],
    models: dict[str, AmericanDevPriceModel],
    acceptance: dict[str, Any],
) -> None:
    report = decompose(models, columns, acceptance, 128)
    assert sorted(report["cells"]) == [
        "E2b@delta=0",
        "E2b@delta=0.0001",
        "E2c@delta=0",
        "E2c@delta=0.0001",
    ]
    native = [name for name, cell in report["cells"].items() if cell["is_native_floor"]]
    assert sorted(native) == ["E2b@delta=0", "E2c@delta=0.0001"]


def test_a_diagonal_cell_reproduces_the_models_own_deployed_metrics(
    columns: dict[str, np.ndarray],
    models: dict[str, AmericanDevPriceModel],
    acceptance: dict[str, Any],
) -> None:
    """The check that makes the off-diagonal cells mean what they claim."""
    model = models["E2c"]
    report = decompose({"E2c": model}, columns, acceptance, 128)
    diagonal = report["cells"]["E2c@delta=0.0001"]
    direct, _ = cell_metrics(
        MarginOverrideModel(model, model.european_margin), columns, acceptance, 128
    )
    assert diagonal["slices"]["overall"] == direct["slices"]["overall"]
    assert diagonal["diagnostics"] == direct["diagnostics"]


def test_the_margin_effect_is_measured_at_fixed_weights(
    columns: dict[str, np.ndarray],
    models: dict[str, AmericanDevPriceModel],
    acceptance: dict[str, Any],
) -> None:
    report = decompose(models, columns, acceptance, 128)
    for label in ("E2b", "E2c"):
        effect = report["margin_effect"][label]["normalized_rmse"]
        assert effect["at_zero_margin"] > 0.0
        assert effect["at_declared_margin"] > 0.0
        assert effect["relative_change"] == pytest.approx(
            (effect["at_declared_margin"] - effect["at_zero_margin"]) / effect["at_zero_margin"]
        )


def test_the_pre_registered_prediction_is_recorded_and_judged(
    columns: dict[str, np.ndarray],
    models: dict[str, AmericanDevPriceModel],
    acceptance: dict[str, Any],
) -> None:
    """A prediction that is not checked mechanically is not pre-registered."""
    report = decompose(models, columns, acceptance, 128)
    assert report["prediction"] == DELTA_EFFECT_PREDICTION
    for label, check in report["prediction_check"].items():
        assert set(check) == {
            "measured_relative_rmse_change",
            "predicted_interval",
            "within_prediction",
            "falsified",
        }, label
        assert check["falsified"] is not check["within_prediction"]
        inside = (
            DELTA_EFFECT_PREDICTION["lower"]
            <= check["measured_relative_rmse_change"]
            <= DELTA_EFFECT_PREDICTION["upper"]
        )
        assert check["within_prediction"] is inside


def test_the_column_difference_is_never_called_seed_variance(
    columns: dict[str, np.ndarray],
    models: dict[str, AmericanDevPriceModel],
    acceptance: dict[str, Any],
) -> None:
    report = decompose(models, columns, acceptance, 128)
    assert report["attempt_variation_label"] == "attempt-to-attempt variation at fixed design"
    assert "not initialization-seed variance" in report["attempt_variation_caveat"]
    assert "n = 2" in report["attempt_variation_caveat"]
    assert sorted(report["attempt_variation"]) == ["delta=0", "delta=0.0001"]
    assert "none of it becomes a gate" in report["no_selection_gate"]


def test_the_diagnostic_breakdown_names_all_seven_families(
    columns: dict[str, np.ndarray],
    models: dict[str, AmericanDevPriceModel],
    acceptance: dict[str, Any],
) -> None:
    diagnostics, _ = shape_diagnostics(
        MarginOverrideModel(models["E2c"], 0.0), columns, acceptance, batch_size=128
    )
    breakdown = diagnostic_breakdown(diagnostics)
    assert sorted(breakdown["bound_families"]) == sorted(BOUND_FAMILIES)
    assert sorted(breakdown["shape_families"]) == sorted(SHAPE_FAMILIES)
    assert breakdown["material_shape_violations"] == sum(
        entry["material_violations"] for entry in breakdown["shape_families"].values()
    )
    assert breakdown["material_bound_violations"] == sum(
        entry["material_violations"] for entry in breakdown["bound_families"].values()
    )


# ---------------------------------------------------------------------------
# The deployment-behaviour partition
# ---------------------------------------------------------------------------


def test_the_projection_partition_is_a_partition(
    columns: dict[str, np.ndarray], models: dict[str, AmericanDevPriceModel]
) -> None:
    partition = projection_partition(models["E2c"], columns)
    masks = [
        partition[name]
        for name in (
            "projection:floor_dominated",
            "projection:crossover",
            "projection:network_dominated",
        )
    ]
    stacked = np.vstack(masks)
    assert bool((stacked.sum(axis=0) == 1).all())
    weight = partition["_projection_weight"]
    assert bool(((weight >= 0.0) & (weight <= 1.0)).all())
    assert np.array_equal(masks[0], weight < FLOOR_DOMINATED_MAXIMUM)
    assert np.array_equal(masks[2], weight > NETWORK_DOMINATED_MINIMUM)


def test_leg_dominance_is_reported_only_where_the_floor_binds(
    columns: dict[str, np.ndarray], models: dict[str, AmericanDevPriceModel]
) -> None:
    partition = projection_partition(models["E2c"], columns)
    binding = ~partition["projection:network_dominated"]
    legs = (
        partition["floor_leg:european_dominant"]
        | partition["floor_leg:intrinsic_dominant"]
        | partition["floor_leg:mixed"]
    )
    assert np.array_equal(legs, binding)


def test_the_partition_follows_the_margin_it_is_asked_about(
    columns: dict[str, np.ndarray], models: dict[str, AmericanDevPriceModel]
) -> None:
    """Raising the floor can only move rows toward floor dominance."""
    model = models["E2c"]
    low = projection_partition(model, columns, 0.0)
    high = projection_partition(model, columns, EUROPEAN_FLOOR_MARGIN)
    assert bool((high["_projection_weight"] <= low["_projection_weight"] + 1.0e-15).all())


# ---------------------------------------------------------------------------
# B.3
# ---------------------------------------------------------------------------


def test_the_characterization_measures_raw_and_deployed_separately(
    columns: dict[str, np.ndarray],
    models: dict[str, AmericanDevPriceModel],
    acceptance: dict[str, Any],
    eligibility_record: dict[str, Any],
) -> None:
    report = characterize(
        models["E2c"],
        columns,
        acceptance,
        128,
        project_root=PROJECT_ROOT,
        row_set="validation",
        eligibility_record=eligibility_record,
    )
    assert sorted(report["predictions"]) == ["deployed", "raw_network"]
    deployed = report["predictions"]["deployed"]["slices"]["overall"]
    raw = report["predictions"]["raw_network"]["slices"]["overall"]
    assert deployed["rows"] == raw["rows"] == ROWS
    assert deployed["normalized"] != raw["normalized"]


def test_every_gated_slice_is_produced(
    columns: dict[str, np.ndarray],
    models: dict[str, AmericanDevPriceModel],
    acceptance: dict[str, Any],
    eligibility_record: dict[str, Any],
) -> None:
    report = characterize(
        models["E2c"],
        columns,
        acceptance,
        128,
        project_root=PROJECT_ROOT,
        row_set="validation",
        eligibility_record=eligibility_record,
    )
    produced = set(report["predictions"]["deployed"]["slices"])
    assert set(GATED_SLICE_FAMILY) <= produced
    assert "projection:crossover" in produced
    assert "floor_leg:intrinsic_dominant" in produced


def test_a_pass_rate_always_travels_with_its_excluded_count(
    columns: dict[str, np.ndarray],
    models: dict[str, AmericanDevPriceModel],
    acceptance: dict[str, Any],
    eligibility_record: dict[str, Any],
) -> None:
    """So "99% of eligible rows" can never be read as "99% of rows"."""
    report = characterize(
        models["E2c"],
        columns,
        acceptance,
        128,
        project_root=PROJECT_ROOT,
        row_set="validation",
        eligibility_record=eligibility_record,
    )
    overall = report["predictions"]["deployed"]["slices"]["overall"]["tolerance"]
    assert overall["rows"] == ROWS
    assert overall["iv_in_scope_rows"] + overall["iv_out_of_scope_rows"] == ROWS
    assert 0.0 <= overall["iv_out_of_scope_fraction"] <= 1.0
    assert 0.0 <= overall["secondary_pass_rate"] <= 1.0
    note = overall["primary"]["tail_moments_are_not_the_declared_statistic"]
    assert "the pass rate is what the 99% bar is stated on" in note
    # The declared pass rate is a number only when the predeclared assessability
    # rule says it may be one. Otherwise it is None under the NOT ASSESSABLE
    # verdict, and the conditional rate is reported separately and marked as a
    # different statistic -- so nothing can be quoted as the declared one.
    assessable = report["eligibility"]["artifact"]["assessability"][
        "primary_statistic_assessable"
    ]
    if assessable:
        assert overall["primary"]["verdict"] == "assessable"
        assert 0.0 <= overall["primary"]["pass_rate"] <= 1.0
        assert "conditional_on_non_degenerate_eligible" not in overall
    else:
        assert overall["primary"]["verdict"] == "NOT ASSESSABLE"
        assert overall["primary"]["pass_rate"] is None
        assert overall["primary"]["meets_bar"] is None
        conditional = overall["conditional_on_non_degenerate_eligible"]
        assert conditional["is_not_the_declared_statistic"] is True
        assert 0.0 <= conditional["pass_rate"] <= 1.0


def test_the_eligibility_map_reports_both_comparison_floors(
    columns: dict[str, np.ndarray],
    models: dict[str, AmericanDevPriceModel],
    acceptance: dict[str, Any],
    eligibility_record: dict[str, Any],
) -> None:
    report = characterize(
        models["E2c"],
        columns,
        acceptance,
        128,
        project_root=PROJECT_ROOT,
        row_set="validation",
        eligibility_record=eligibility_record,
    )
    floors = report["eligibility"]["comparison_floors"]
    assert sorted(floors) == ["one_sided", "two_sided"]
    assert floors["two_sided"]["in_scope_fraction"] <= floors["one_sided"]["in_scope_fraction"]
    assert "select nothing" in report["eligibility"]["comparison_note"]
    assert report["eligibility"]["label_free"] is False


def test_the_calibration_gate_covers_only_the_frozen_family(
    columns: dict[str, np.ndarray],
    models: dict[str, AmericanDevPriceModel],
    acceptance: dict[str, Any],
    eligibility_record: dict[str, Any],
) -> None:
    report = characterize(
        models["E2c"],
        columns,
        acceptance,
        128,
        project_root=PROJECT_ROOT,
        row_set="validation",
        eligibility_record=eligibility_record,
    )
    calibration = report["predictions"]["deployed"]["calibration"]
    for name, entry in calibration.items():
        assert entry["gated"] is (name in GATED_SLICE_FAMILY)
    for name in report["predictions"]["deployed"]["calibration_gate"]["failing_gated_slices"]:
        assert name in GATED_SLICE_FAMILY
    for family in ("moneyness", "expiry"):
        spread = report["predictions"]["deployed"]["bias_spread"][family]
        assert spread["family"] == family


def test_the_characterization_declares_the_tolerance_it_applied(
    columns: dict[str, np.ndarray],
    models: dict[str, AmericanDevPriceModel],
    acceptance: dict[str, Any],
    eligibility_record: dict[str, Any],
) -> None:
    report = characterize(
        models["E2c"],
        columns,
        acceptance,
        128,
        project_root=PROJECT_ROOT,
        row_set="validation",
        eligibility_record=eligibility_record,
    )
    assert report["tolerance"]["declared_before_analysis"] is True
    assert report["european_margin"] == pytest.approx(EUROPEAN_FLOOR_MARGIN)
    assert report["rows"] == ROWS
    assert MARGIN_CELLS == (0.0, EUROPEAN_FLOOR_MARGIN)
