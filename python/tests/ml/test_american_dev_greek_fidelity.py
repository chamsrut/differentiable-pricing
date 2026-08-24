"""Task 9H Greek fidelity: autograd correctness, the mandatory split, the grids.

Autograd Greeks are validated against closed form on an analytic-only pricer,
where the exact answer is known. Nothing here opens a dataset partition or reads
a checkpoint.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import pytest
import torch
from differentiable_pricing.ml.american_dev.frozen import FrozenCheckpointError
from differentiable_pricing.ml.american_dev.greek_fidelity import (
    CROSSOVER_SITES,
    DEGENERATE_SENSITIVITY,
    GREEKS,
    GRID2_PRIMARY_STATISTIC,
    PREDICTIONS,
    RELATIVE_ERROR_UNCERTAINTY_MULTIPLE,
    SPOT_SCAN_HALF_WIDTH,
    SPOT_SCAN_SPACING,
    GreekFidelityError,
    autograd_greeks,
    autograd_sign_violations,
    both_greeks,
    compare_greek,
    crossover_signals,
    declared_contracts,
    degeneracy,
    locate_crossings,
    scan_crossover,
    scope_declaration,
    sign_comparison,
)
from differentiable_pricing.ml.american_dev.models import build_network
from differentiable_pricing.ml.american_dev.representation import (
    AmericanDevPriceModel,
    european_price,
)
from differentiable_pricing.ml.model import Scaling

ATM: Final = {
    "option_type": 1.0,
    "spot": 100.0,
    "strike": 100.0,
    "maturity": 1.0,
    "rate": 0.03,
    "dividend_yield": 0.0,
    "volatility": 0.25,
}


def _model(head: str = "smooth_lower_floor_margin_raw_loss") -> AmericanDevPriceModel:
    torch.manual_seed(4)
    network = build_network(
        {"name": "smooth_residual", "width": 32, "blocks": 2, "activation": "tanh"}, 5
    )
    scaling = Scaling(
        feature_mean=np.zeros(5, dtype=np.float64),
        feature_scale=np.ones(5, dtype=np.float64),
        price_mean=0.15,
        price_scale=0.05,
    )
    model = AmericanDevPriceModel(network, scaling, head=head)
    model.eval()
    return model


def _states(rows: int = 6) -> np.ndarray:
    generator = np.random.default_rng(17)
    spot = generator.uniform(85.0, 115.0, rows)
    return np.column_stack(
        [
            np.where(np.arange(rows) % 2 == 0, 1.0, -1.0),
            spot,
            spot * np.exp(-generator.uniform(-0.3, 0.3, rows)),
            generator.uniform(0.2, 2.0, rows),
            generator.uniform(0.0, 0.06, rows),
            generator.uniform(0.0, 0.04, rows),
            generator.uniform(0.15, 0.5, rows),
        ]
    )


# ---------------------------------------------------------------------------
# Autograd correctness, where the exact answer is known
# ---------------------------------------------------------------------------


def test_autograd_recovers_black_scholes_greeks_exactly() -> None:
    """An analytic-only pricer, so the closed form is the truth, not a proxy."""
    greeks = autograd_greeks(
        european_price, np.array([[1.0, 100.0, 100.0, 1.0, 0.03, 0.0, 0.25]])
    )
    spot, strike, maturity, rate, volatility = 100.0, 100.0, 1.0, 0.03, 0.25
    d1 = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / (
        volatility * math.sqrt(maturity)
    )
    normal = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    density = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    assert float(greeks["delta"][0]) == pytest.approx(normal, rel=1e-10)
    assert float(greeks["gamma"][0]) == pytest.approx(
        density / (spot * volatility * math.sqrt(maturity)), rel=1e-10
    )
    assert float(greeks["vega"][0]) == pytest.approx(
        spot * density * math.sqrt(maturity), rel=1e-10
    )


# ---------------------------------------------------------------------------
# The mandatory split
# ---------------------------------------------------------------------------


def test_both_predictions_are_always_computed() -> None:
    greeks = both_greeks(_model(), _states())
    assert sorted(greeks) == sorted(PREDICTIONS)
    for entry in greeks.values():
        for greek in GREEKS:
            assert entry[greek].shape == (6,)


def test_the_split_really_differs_where_the_floor_binds() -> None:
    """If raw and deployed agreed everywhere the split would decide nothing."""
    deep = np.array([[1.0, 145.0, 55.0, 0.05, 0.0, 0.0, 0.08]])
    greeks = both_greeks(_model(), deep)
    assert float(greeks["deployed"]["vega"][0]) == pytest.approx(0.0, abs=1e-12)
    assert abs(float(greeks["raw_network"]["vega"][0])) > 1e-6


def test_a_head_without_a_floor_has_no_split() -> None:
    with pytest.raises(FrozenCheckpointError):
        both_greeks(_model("direct"), _states())


# ---------------------------------------------------------------------------
# Error conventions
# ---------------------------------------------------------------------------


def test_delta_carries_no_relative_error() -> None:
    entry = compare_greek("delta", np.array([0.5, -0.5]), np.array([0.4, -0.6]), 1e-3)
    assert entry["relative"] is None
    assert "dimensionless" in entry["convention"]
    assert entry["absolute"]["rows"] == 2


def test_the_relative_denominator_floor_is_derived_from_the_uncertainty() -> None:
    uncertainty = 1e-4
    entry = compare_greek(
        "gamma",
        np.array([0.02, 0.0002]),
        np.array([0.021, 0.0003]),
        uncertainty,
    )
    assert entry["relative_denominator_floor"] == pytest.approx(
        RELATIVE_ERROR_UNCERTAINTY_MULTIPLE * uncertainty
    )
    assert entry["relative_rows"] == 1
    assert entry["relative_excluded_fraction"] == pytest.approx(0.5)


def test_every_comparison_carries_the_reference_uncertainty() -> None:
    entry = compare_greek("vega", np.array([30.0, 1.0]), np.array([31.0, 1.0]), 0.5)
    assert entry["reference_uncertainty"] == 0.5
    assert entry["conclusions_only_where_disagreement_exceeds_uncertainty"] is True
    assert entry["rows_exceeding_reference_uncertainty"] == 1
    assert entry["sign_disagreement_rate"] == pytest.approx(0.0)


def test_an_unknown_greek_is_refused() -> None:
    with pytest.raises(GreekFidelityError):
        compare_greek("theta", np.array([1.0]), np.array([1.0]), 1e-3)


# ---------------------------------------------------------------------------
# Degeneracy
# ---------------------------------------------------------------------------


def test_the_excess_is_what_is_reported() -> None:
    surrogate = np.array([0.0, 0.0, 5.0])
    reference = np.array([0.0, 8.0, 5.0])
    result = degeneracy(surrogate, reference)
    assert result["threshold"] == DEGENERATE_SENSITIVITY
    assert result["surrogate_degenerate_rows"] == 2
    assert result["reference_degenerate_rows"] == 1
    assert result["excess_rows"] == 1
    assert "manufactured failure region" in result["excess_meaning"]


# ---------------------------------------------------------------------------
# Grid 2 and 2b
# ---------------------------------------------------------------------------


def test_both_curvature_sites_are_located_separately() -> None:
    model = _model()
    found = {}
    for site in CROSSOVER_SITES:
        located = locate_crossings(model, ATM, site, axis="spot", low=60.0, high=140.0)
        assert located["site"] == site
        assert "crossing_count" in located
        found[site] = located
    assert set(found) == set(CROSSOVER_SITES)


def test_a_contract_without_a_crossing_is_reported_not_dropped() -> None:
    model = _model()
    located = locate_crossings(
        model, ATM, "raw_minus_floor", axis="spot", low=99.999, high=100.0, points=5
    )
    assert located["crossing_exists"] is False
    assert "substitute another" in located["note"]


def test_multiple_crossings_are_counted_not_silently_reduced() -> None:
    model = _model()
    located = locate_crossings(model, ATM, "raw_minus_floor", axis="spot", low=55.0, high=145.0)
    assert located["crossing_count"] == len(located["crossings"])
    assert located["multiple_crossings"] is (located["crossing_count"] > 1)


def test_the_primary_statistic_is_the_autograd_projection_difference() -> None:
    """Not a bumped CRR Gamma: this quantity has no finite-difference noise."""
    model = _model()
    located = locate_crossings(model, ATM, "raw_minus_floor", axis="spot", low=60.0, high=140.0)
    assert located["crossing_exists"]
    scan = scan_crossover(
        model,
        ATM,
        located["crossings"][0],
        axis="spot",
        half_width=SPOT_SCAN_HALF_WIDTH,
        spacing=SPOT_SCAN_SPACING,
    )
    primary = scan["gamma_projection_difference"]
    assert primary["carries_no_finite_difference_uncertainty"] is True
    assert scan["primary_statistic"] == GRID2_PRIMARY_STATISTIC
    assert primary["maximum_absolute"] >= 0.0
    assert "slope difference" in scan["delta_projection_difference"]["note"]
    assert len(scan["grid"]) == scan["points"]
    for name in PREDICTIONS:
        assert sorted(scan[name if name == "deployed" else "raw_network"]) == sorted(GREEKS)


def test_the_crossover_signals_are_label_free() -> None:
    model = _model()
    states = torch.as_tensor(_states(), dtype=torch.float64)
    signals = crossover_signals(model, states)
    assert set(signals) >= set(CROSSOVER_SITES)
    weight = signals["projection_weight"]
    assert bool(((weight >= 0.0) & (weight <= 1.0)).all())


def test_an_unknown_axis_or_site_is_refused() -> None:
    model = _model()
    with pytest.raises(GreekFidelityError):
        locate_crossings(model, ATM, "raw_minus_floor", axis="rate", low=0.0, high=1.0)
    with pytest.raises(GreekFidelityError):
        locate_crossings(model, ATM, "invented_site", axis="spot", low=60.0, high=140.0)


def test_the_declared_contract_set_is_deterministic_and_spans_both_types() -> None:
    contracts = declared_contracts()
    assert 20 <= len(contracts) <= 50
    assert {entry["option_type"] for entry in contracts} == {1.0, -1.0}
    assert declared_contracts()[0] == contracts[0]


# ---------------------------------------------------------------------------
# Section G
# ---------------------------------------------------------------------------


def test_pointwise_signs_are_reported_as_a_different_functional() -> None:
    model = _model()
    states = _states()
    greeks = both_greeks(model, states)["deployed"]
    option_type = ["call" if row[0] > 0 else "put" for row in states]
    pointwise = autograd_sign_violations(greeks, option_type, np.ones(states.shape[0]))
    assert pointwise["rows"] == states.shape[0]
    assert pointwise["functional"] == "pointwise autograd derivative"
    comparison = sign_comparison(
        {"material_shape_violations": 551, "shape_families": {}}, pointwise
    )
    assert comparison["finite_bump_battery"]["material_shape_violations"] == 551
    assert "different functionals" in comparison["caveat"]
    assert "not a Greek-accuracy metric" in comparison["caveat"]


def test_the_scope_declaration_states_the_split_and_the_limitation() -> None:
    declaration = scope_declaration()
    assert declaration["predictions"] == list(PREDICTIONS)
    assert "costing no neural attempt" in declaration["mandatory_split"]
    assert "finite-difference uncertainty" in declaration["grid1_limitation"]
    assert declaration["crossover_sites"] == list(CROSSOVER_SITES)
    assert declaration["grid2_primary_statistic"] == GRID2_PRIMARY_STATISTIC
