"""Cross-language tests for the task 9C-C1 valuation-time PDE surface.

These exercise the `pde_valuation_surface` boundary of the `_pde` extension.
The same requirements are covered independently in `cpp/tests/test_main.cpp`,
and neither suite substitutes a Python reimplementation of the backward
induction for the engine.

One surface produces many rows. Those rows are correlated numerical outputs of
a single solve, and nothing here treats them as independent observations.
"""

from __future__ import annotations

import itertools
import math

import pytest
from differentiable_pricing import black_scholes, pde_price, pde_valuation_surface

# The PSOR tolerance bounds the residual of one implicit solve, so the solver's
# contribution accumulates over the time steps. Comparisons that are exact in
# exact arithmetic are given this band.
SOLVER_BAND = 1.0e-6

# Conservative regression bands for nodewise Greeks against Black-Scholes on the
# grids used below, measured on smooth interior nodes only. They are cross-check
# bands, not accuracy claims and not acceptance gates: task 9C-B remains
# `no_policy_selected` and no grid here is label-grade.
FINE_PRICE_BAND = 5.0e-4
FINE_DELTA_BAND = 1.0e-4
FINE_GAMMA_BAND = 1.0e-5
COARSE_PRICE_BAND = 2.0e-3
COARSE_DELTA_BAND = 2.0e-4
COARSE_GAMMA_BAND = 2.0e-5


def _surface(
    *,
    option_type: str = "call",
    exercise_style: str = "european",
    strike: float = 100.0,
    valuation_time: float = 0.0,
    expiry_time: float = 1.0,
    volatility: float = 0.2,
    continuous_carry: float | None = 0.0,
    rate: float = 0.05,
    curve_times: list[float] | None = None,
    curve_log_discounts: list[float] | None = None,
    dividends: list[tuple[float, float]] | None = None,
    settlement: str = "cash",
    contract_multiplier: float = 100.0,
    spot_intervals: int = 800,
    time_steps: int = 400,
    spot_maximum: float = 400.0,
    rannacher_steps: int = 2,
    psor_tolerance: float = 1.0e-11,
    psor_relaxation: float = 1.2,
    psor_maximum_iterations: int = 50_000,
    boundary_exclusion_nodes: int = 4,
    query_spots: list[float] | None = None,
) -> dict[str, object]:
    """Solve one surface on a flat curve unless explicit knots are supplied."""
    if curve_times is None:
        curve_times = [0.0, expiry_time]
        curve_log_discounts = [0.0, -rate * expiry_time]
    assert curve_log_discounts is not None
    return pde_valuation_surface(
        option_type=option_type,
        exercise_style=exercise_style,
        strike=strike,
        valuation_time=valuation_time,
        expiry_time=expiry_time,
        volatility=volatility,
        continuous_carry=continuous_carry,
        curve_times=curve_times,
        curve_log_discounts=curve_log_discounts,
        dividends=[] if dividends is None else dividends,
        settlement=settlement,
        contract_multiplier=contract_multiplier,
        spot_intervals=spot_intervals,
        time_steps=time_steps,
        spot_maximum=spot_maximum,
        rannacher_steps=rannacher_steps,
        psor_tolerance=psor_tolerance,
        psor_relaxation=psor_relaxation,
        psor_maximum_iterations=psor_maximum_iterations,
        boundary_exclusion_nodes=boundary_exclusion_nodes,
        query_spots=[] if query_spots is None else query_spots,
    )


def _scalar_price(*, spot: float, **kwargs: object) -> dict[str, object]:
    """Price the same contract through the untouched scalar entry point."""
    rate = float(kwargs.pop("rate", 0.05))
    expiry_time = float(kwargs.pop("expiry_time", 1.0))
    curve_times = kwargs.pop("curve_times", None)
    curve_log_discounts = kwargs.pop("curve_log_discounts", None)
    if curve_times is None:
        curve_times = [0.0, expiry_time]
        curve_log_discounts = [0.0, -rate * expiry_time]
    defaults: dict[str, object] = {
        "option_type": "call",
        "exercise_style": "european",
        "strike": 100.0,
        "valuation_time": 0.0,
        "volatility": 0.2,
        "continuous_carry": 0.0,
        "dividends": [],
        "settlement": "cash",
        "contract_multiplier": 100.0,
        "spot_intervals": 800,
        "time_steps": 400,
        "spot_maximum": 400.0,
        "rannacher_steps": 2,
        "psor_tolerance": 1.0e-11,
        "psor_relaxation": 1.2,
        "psor_maximum_iterations": 50_000,
    }
    defaults.update(kwargs)
    return pde_price(
        spot=spot,
        expiry_time=expiry_time,
        curve_times=curve_times,
        curve_log_discounts=curve_log_discounts,
        **defaults,  # type: ignore[arg-type]
    )


@pytest.fixture(scope="module")
def american_put() -> dict[str, object]:
    """One American put surface, reused so the suite pays for one solve."""
    return _surface(option_type="put", exercise_style="american")


@pytest.fixture(scope="module")
def european_call() -> dict[str, object]:
    return _surface(spot_intervals=1600, time_steps=800)


# --------------------------------------------------------------------------
# 1-3. Structure, grid metadata, and agreement with the scalar API
# --------------------------------------------------------------------------


def test_surface_vectors_are_aligned_ordered_and_finite(european_call: dict[str, object]) -> None:
    surface = european_call
    count = int(surface["spot_intervals"]) + 1
    columns = (
        "spot_nodes",
        "values",
        "obstacles",
        "obstacle_slacks",
        "lcp_linear_residuals",
        "deltas",
        "gammas",
        "greek_eligible",
        "exercise_states",
        "greek_eligibility_reasons",
    )
    for column in columns:
        assert len(surface[column]) == count, column

    step = float(surface["spot_step"])
    nodes = surface["spot_nodes"]
    assert nodes[0] == 0.0
    assert all(later > earlier for earlier, later in itertools.pairwise(nodes))
    for index, node in enumerate(nodes):
        assert node == pytest.approx(index * step, abs=1.0e-12)
    assert nodes[-1] == pytest.approx(float(surface["spot_maximum"]), abs=1.0e-12)
    assert all(math.isfinite(value) for value in surface["values"])
    assert float(surface["spot_step"]) * int(surface["strike_node_index"]) == pytest.approx(
        100.0, abs=1.0e-12
    )
    assert surface["backward_inductions"] == 1
    assert surface["valuation_time"] == 0.0
    assert surface["regime_stencil_radius"] == 2
    assert surface["boundary_exclusion_nodes"] == 4
    assert surface["solver_status"] == "discrete_system_converged"
    assert surface["discretization_accuracy"] == "not_assessed"
    assert float(surface["exercise_classification_scale"]) > 0.0


@pytest.mark.parametrize(
    ("option_type", "exercise_style", "spot", "dividends"),
    [
        ("call", "european", 100.0, []),
        ("put", "american", 103.0, []),
        ("call", "european", 97.5, [(0.4, 3.0)]),
        ("put", "american", 88.0, [(0.4, 3.0)]),
    ],
)
def test_a_surface_query_reproduces_the_scalar_price_exactly(
    option_type: str, exercise_style: str, spot: float, dividends: list[tuple[float, float]]
) -> None:
    # Same interpolation implementation on the same solved slice, so this is a
    # bitwise identity rather than a tolerance.
    surface = _surface(
        option_type=option_type,
        exercise_style=exercise_style,
        dividends=dividends,
        query_spots=[spot],
    )
    scalar = _scalar_price(
        spot=spot,
        option_type=option_type,
        exercise_style=exercise_style,
        dividends=dividends,
    )
    assert surface["queries"][0]["value"] == scalar["price"]
    assert surface["spot_intervals"] == scalar["spot_intervals"]
    assert surface["spot_step"] == scalar["spot_step"]
    assert surface["time_steps"] == scalar["time_steps"]
    assert surface["psor_total_iterations"] == scalar["psor_total_iterations"]
    assert surface["maximum_lcp_residual"] == scalar["maximum_lcp_residual"]
    assert surface["solver_status"] == scalar["solver_status"]
    assert surface["dividend_events"] == scalar["dividend_events"]


# --------------------------------------------------------------------------
# 4-5. Analytic European Greeks, monotonicity and convexity
# --------------------------------------------------------------------------


def test_european_call_nodes_match_black_scholes_price_delta_and_gamma(
    european_call: dict[str, object],
) -> None:
    surface = european_call
    checked = 0
    for index, spot in enumerate(surface["spot_nodes"]):
        if not (60.0 <= spot <= 160.0) or not surface["greek_eligible"][index]:
            continue
        reference = black_scholes("call", spot, 100.0, 1.0, 0.05, 0.0, 0.2)
        assert surface["values"][index] == pytest.approx(
            reference["price"], abs=FINE_PRICE_BAND
        ), spot
        assert surface["deltas"][index] == pytest.approx(
            reference["delta"], abs=FINE_DELTA_BAND
        ), spot
        assert surface["gammas"][index] == pytest.approx(
            reference["gamma"], abs=FINE_GAMMA_BAND
        ), spot
        checked += 1
    assert checked >= 100


def test_european_put_nodes_match_black_scholes_on_a_coarser_grid() -> None:
    surface = _surface(option_type="put")
    checked = 0
    for index, spot in enumerate(surface["spot_nodes"]):
        if not (60.0 <= spot <= 160.0) or not surface["greek_eligible"][index]:
            continue
        reference = black_scholes("put", spot, 100.0, 1.0, 0.05, 0.0, 0.2)
        assert surface["values"][index] == pytest.approx(
            reference["price"], abs=COARSE_PRICE_BAND
        ), spot
        assert surface["deltas"][index] == pytest.approx(
            reference["delta"], abs=COARSE_DELTA_BAND
        ), spot
        assert surface["gammas"][index] == pytest.approx(
            reference["gamma"], abs=COARSE_GAMMA_BAND
        ), spot
        checked += 1
    assert checked >= 100


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("exercise_style", ["european", "american"])
def test_surface_values_are_monotone_and_convex_in_the_spot(
    option_type: str, exercise_style: str
) -> None:
    surface = _surface(option_type=option_type, exercise_style=exercise_style)
    values = surface["values"]
    step = float(surface["spot_step"])
    for earlier, later in itertools.pairwise(values):
        if option_type == "call":
            assert later >= earlier - SOLVER_BAND
        else:
            assert later <= earlier + SOLVER_BAND
    for index in range(1, len(values) - 1):
        second_difference = values[index + 1] - 2.0 * values[index] + values[index - 1]
        assert second_difference >= -SOLVER_BAND * step
    for index, gamma in enumerate(surface["gammas"]):
        if surface["greek_eligible"][index]:
            assert gamma >= -SOLVER_BAND


# --------------------------------------------------------------------------
# 6-9. Obstacle dominance, exercise classification and Greek eligibility
# --------------------------------------------------------------------------


def test_american_nodes_dominate_the_discrete_obstacle(american_put: dict[str, object]) -> None:
    for value, obstacle, slack in zip(
        american_put["values"],
        american_put["obstacles"],
        american_put["obstacle_slacks"],
        strict=True,
    ):
        assert value >= obstacle
        assert slack == value - obstacle


def test_pure_exercise_region_put_nodes_are_intrinsic(american_put: dict[str, object]) -> None:
    scale = float(american_put["exercise_classification_scale"])
    checked = 0
    for index, spot in enumerate(american_put["spot_nodes"]):
        if not (20.0 <= spot <= 60.0):
            continue
        assert american_put["exercise_states"][index] == "exercise", spot
        assert american_put["values"][index] == pytest.approx(100.0 - spot, abs=scale), spot
        assert american_put["greek_eligible"][index] is True, spot
        # Intrinsic derivatives are exact for a stencil entirely inside one
        # exercise regime, so these rows are correct but nearly information-free.
        assert american_put["deltas"][index] == pytest.approx(-1.0, abs=1.0e-12), spot
        assert american_put["gammas"][index] == pytest.approx(0.0, abs=1.0e-12), spot
        checked += 1
    assert checked >= 20


def test_continuation_region_put_nodes_are_classified_and_above_intrinsic(
    american_put: dict[str, object],
) -> None:
    scale = float(american_put["exercise_classification_scale"])
    checked = 0
    for index, spot in enumerate(american_put["spot_nodes"]):
        if not (120.0 <= spot <= 180.0):
            continue
        assert american_put["exercise_states"][index] == "continuation", spot
        assert american_put["obstacle_slacks"][index] > scale, spot
        assert american_put["greek_eligible"][index] is True, spot
        assert american_put["deltas"][index] < 0.0
        checked += 1
    assert checked >= 20


def test_european_nodes_report_no_obstacle(european_call: dict[str, object]) -> None:
    assert set(european_call["exercise_states"]) == {"no_obstacle"}


def test_nodes_whose_stencil_crosses_the_free_boundary_are_ineligible(
    american_put: dict[str, object],
) -> None:
    states = american_put["exercise_states"]
    transition = next(index for index, state in enumerate(states) if state != "exercise")
    radius = int(american_put["regime_stencil_radius"])
    assert states[transition - 1] == "exercise"
    # A node is ineligible exactly when its window spans the last exercise node
    # and the first node beyond it, which is the band [t - radius, t + radius - 1].
    for index in range(transition - radius, transition + radius):
        assert american_put["greek_eligible"][index] is False, index
        assert american_put["greek_eligibility_reasons"][index] == "regime_stencil_not_uniform"
    # The rule is local: the free boundary must not sterilise the whole regime.
    assert american_put["greek_eligible"][transition - radius - 1] is True
    assert american_put["greek_eligible"][transition + radius] is True


def test_domain_boundary_nodes_are_ineligible(american_put: dict[str, object]) -> None:
    intervals = int(american_put["spot_intervals"])
    buffer = int(american_put["boundary_exclusion_nodes"])
    reasons = american_put["greek_eligibility_reasons"]
    assert reasons[0] == "centered_stencil_unavailable"
    assert reasons[intervals] == "centered_stencil_unavailable"
    assert american_put["deltas"][0] is None
    assert american_put["gammas"][0] is None
    assert american_put["deltas"][intervals] is None
    assert american_put["gammas"][intervals] is None
    for index in list(range(1, buffer)) + list(range(intervals - buffer + 1, intervals)):
        assert american_put["greek_eligible"][index] is False, index
        assert reasons[index] == "inside_domain_boundary_buffer", index
        # The stencil exists here; only its eligibility is denied.
        assert american_put["deltas"][index] is not None


def test_numerically_indifferent_nodes_are_never_greek_eligible(
    american_put: dict[str, object],
) -> None:
    # The regime is uncertified at the solver's own residual scale, so which
    # quantity the difference quotient estimates - the derivative of an
    # obstacle-clamped payoff or of a free PDE solution - is uncertified with
    # it. The refusal is structural and must hold wherever an indifferent band
    # appears, including one wide enough to have a regime-uniform interior at
    # the free boundary.
    seen = 0
    for index, state in enumerate(american_put["exercise_states"]):
        if state != "numerically_indifferent":
            continue
        assert american_put["greek_eligible"][index] is False, index
        assert american_put["greek_eligibility_reasons"][index] in {
            "unresolved_exercise_state",
            "centered_stencil_unavailable",
            "inside_domain_boundary_buffer",
        }, index
        seen += 1
    assert seen >= 100
    # The interior of the band carries the specific reason, not a boundary one.
    interior = [
        index
        for index, state in enumerate(american_put["exercise_states"])
        if state == "numerically_indifferent"
        and 4 <= index <= int(american_put["spot_intervals"]) - 4
    ]
    assert interior
    for index in interior:
        assert (
            american_put["greek_eligibility_reasons"][index] == "unresolved_exercise_state"
        ), index
        # The stencil still exists; only its interpretation is refused.
        assert american_put["deltas"][index] is not None


def test_the_exercise_classification_uses_the_declared_solver_scale(
    american_put: dict[str, object],
) -> None:
    scale = float(american_put["exercise_classification_scale"])
    for index, state in enumerate(american_put["exercise_states"]):
        slack = american_put["obstacle_slacks"][index]
        residual = american_put["lcp_linear_residuals"][index]
        if state == "continuation":
            assert slack > scale
        elif state == "exercise":
            assert slack <= scale and residual > scale
        else:
            assert state == "numerically_indifferent"
            assert slack <= scale and residual <= scale


# --------------------------------------------------------------------------
# 10-12. Dividends, negative rates and carry
# --------------------------------------------------------------------------


def test_a_discrete_dividend_surface_solves_and_stays_above_intrinsic() -> None:
    surface = _surface(
        option_type="put",
        exercise_style="american",
        continuous_carry=0.01,
        dividends=[(0.35, 2.0), (0.8, 1.5)],
    )
    assert [event["ex_time"] for event in surface["dividend_events"]] == [0.35, 0.8]
    for value, obstacle in zip(surface["values"], surface["obstacles"], strict=True):
        assert value >= obstacle
    # No analytic comparator applies: the cash-dividend jump invalidates
    # Black-Scholes, so this asserts structure and the scalar identity only.
    scalar = _scalar_price(
        spot=100.0,
        option_type="put",
        exercise_style="american",
        continuous_carry=0.01,
        dividends=[(0.35, 2.0), (0.8, 1.5)],
    )
    queried = _surface(
        option_type="put",
        exercise_style="american",
        continuous_carry=0.01,
        dividends=[(0.35, 2.0), (0.8, 1.5)],
        query_spots=[100.0],
    )
    assert queried["queries"][0]["value"] == scalar["price"]


def test_a_negative_rate_european_call_surface_matches_black_scholes() -> None:
    surface = _surface(rate=-0.01, continuous_carry=0.03, spot_intervals=1600, time_steps=800)
    checked = 0
    for index, spot in enumerate(surface["spot_nodes"]):
        if not (70.0 <= spot <= 140.0) or not surface["greek_eligible"][index]:
            continue
        reference = black_scholes("call", spot, 100.0, 1.0, -0.01, 0.03, 0.2)
        assert surface["values"][index] == pytest.approx(reference["price"], abs=FINE_PRICE_BAND)
        assert surface["deltas"][index] == pytest.approx(reference["delta"], abs=FINE_DELTA_BAND)
        assert surface["gammas"][index] == pytest.approx(reference["gamma"], abs=FINE_GAMMA_BAND)
        checked += 1
    assert checked >= 100


def test_a_negative_rate_american_put_surface_has_no_exercise_region() -> None:
    # The early-exercise incentive of a put is the interest earned on the strike.
    # At a negative rate there is none and a positive carry only adds a reason to
    # wait, so the classification must report no exercise node anywhere and the
    # American surface must collapse onto the European one.
    settings = {"rate": -0.01, "continuous_carry": 0.02, "option_type": "put"}
    american = _surface(exercise_style="american", **settings)  # type: ignore[arg-type]
    european = _surface(exercise_style="european", **settings)  # type: ignore[arg-type]
    assert "exercise" not in american["exercise_states"]
    for value, obstacle in zip(american["values"], american["obstacles"], strict=True):
        assert value >= obstacle
    for value, reference in zip(american["values"], european["values"], strict=True):
        assert value == pytest.approx(reference, abs=SOLVER_BAND)


def test_a_negative_rate_carrying_american_call_surface_has_an_exercise_region() -> None:
    surface = _surface(
        option_type="call", exercise_style="american", rate=-0.01, continuous_carry=0.06
    )
    states = surface["exercise_states"]
    assert "exercise" in states
    assert "continuation" in states
    # A call exercises high, so the exercise region is the upper spot range.
    exercise_spots = [
        surface["spot_nodes"][index] for index, state in enumerate(states) if state == "exercise"
    ]
    continuation_spots = [
        surface["spot_nodes"][index]
        for index, state in enumerate(states)
        if state == "continuation"
    ]
    assert min(exercise_spots) > max(continuation_spots)


# --------------------------------------------------------------------------
# 13-17. Multi-spot evaluation, determinism and working memory
# --------------------------------------------------------------------------


def test_thirty_two_queries_cost_exactly_one_solve() -> None:
    spots = [60.0 + 2.5 * index for index in range(32)]
    many = _surface(option_type="put", exercise_style="american", query_spots=spots)
    one = _surface(option_type="put", exercise_style="american", query_spots=[spots[0]])
    assert many["backward_inductions"] == 1
    assert many["query_count"] == 32
    assert len({query["spot"] for query in many["queries"]}) == 32
    # The solve is identical whatever was asked of it afterwards. Thirty-two
    # separate solves could not reproduce one shared iteration total.
    assert many["psor_total_iterations"] == one["psor_total_iterations"]
    assert many["psor_solves"] == one["psor_solves"]
    assert many["maximum_lcp_residual"] == one["maximum_lcp_residual"]
    assert many["queries"][0]["value"] == one["queries"][0]["value"]
    eligible = [query for query in many["queries"] if query["greek_eligible"]]
    assert len(eligible) >= 16
    for query in eligible:
        assert query["delta"] is not None
        assert query["gamma"] is not None
        assert query["exercise_state"] in {"exercise", "continuation"}
    for query in many["queries"]:
        if not query["greek_eligible"]:
            assert query["delta"] is None
            assert query["gamma"] is None
            assert query["greek_eligibility_reason"] != "eligible"


def test_query_order_is_preserved_and_duplicates_are_kept() -> None:
    spots = [130.0, 70.0, 100.0, 70.0, 100.0]
    surface = _surface(option_type="put", exercise_style="american", query_spots=spots)
    queries = surface["queries"]
    assert [query["spot"] for query in queries] == spots
    assert [query["query_index"] for query in queries] == [0, 1, 2, 3, 4]
    assert [query["first_occurrence_index"] for query in queries] == [0, 1, 2, 1, 2]
    assert queries[3]["value"] == queries[1]["value"]
    assert queries[4]["value"] == queries[2]["value"]
    assert queries[3]["delta"] == queries[1]["delta"]
    assert queries[4]["gamma"] == queries[2]["gamma"]


def test_a_query_reports_the_cell_that_produced_its_price() -> None:
    surface = _surface(query_spots=[100.3])
    query = surface["queries"][0]
    step = float(surface["spot_step"])
    assert query["left_node_index"] == int(100.3 // step)
    assert query["right_node_index"] == query["left_node_index"] + 1
    left = surface["spot_nodes"][query["left_node_index"]]
    right = surface["spot_nodes"][query["right_node_index"]]
    assert left <= query["spot"] <= right


def test_query_greeks_are_interpolated_nodewise_greeks_not_price_derivatives() -> None:
    # Documented semantics, pinned so they cannot drift into an implied claim
    # that the query Greeks differentiate the query price interpolant.
    surface = _surface(query_spots=[100.25])
    query = surface["queries"][0]
    left = query["left_node_index"]
    right = query["right_node_index"]
    weight = (query["spot"] - surface["spot_nodes"][left]) / (
        surface["spot_nodes"][right] - surface["spot_nodes"][left]
    )
    expected_delta = surface["deltas"][left] + weight * (
        surface["deltas"][right] - surface["deltas"][left]
    )
    expected_gamma = surface["gammas"][left] + weight * (
        surface["gammas"][right] - surface["gammas"][left]
    )
    assert query["delta"] == expected_delta
    assert query["gamma"] == expected_gamma

    # Differentiating the price interpolant is a different number. This asserts
    # the disagreement exists rather than pretending the two coincide.
    step = 1.0e-4
    bumped = _surface(query_spots=[100.25 - step, 100.25 + step])
    slope = (bumped["queries"][1]["value"] - bumped["queries"][0]["value"]) / (2.0 * step)
    assert slope != query["delta"]
    assert abs(slope - query["delta"]) < 1.0e-3


def test_a_query_on_a_grid_node_returns_that_node_exactly() -> None:
    # The consistent case: on a node the price interpolant and the nodewise
    # Greeks agree bitwise, with no blending at all.
    surface = _surface(option_type="put", exercise_style="american", query_spots=[100.0, 60.0])
    for query in surface["queries"]:
        index = query["left_node_index"]
        assert surface["spot_nodes"][index] == query["spot"]
        assert query["value"] == surface["values"][index]
        assert query["delta"] == surface["deltas"][index]
        assert query["gamma"] == surface["gammas"][index]


@pytest.mark.parametrize("spot", [0.0, -5.0, 400.0, 1.0e6, math.inf, math.nan])
def test_out_of_domain_queries_are_rejected(spot: float) -> None:
    with pytest.raises(ValueError, match="surface query spot 1"):
        _surface(query_spots=[100.0, spot])


def test_repeated_surface_solves_are_bitwise_identical() -> None:
    kwargs = {
        "option_type": "put",
        "exercise_style": "american",
        "continuous_carry": 0.01,
        "volatility": 0.28,
        "rate": 0.04,
        "dividends": [(0.35, 1.25)],
        "query_spots": [88.0, 100.0, 117.5],
    }
    first = _surface(**kwargs)  # type: ignore[arg-type]
    again = _surface(**kwargs)  # type: ignore[arg-type]
    for column in ("spot_nodes", "values", "deltas", "gammas", "exercise_states"):
        assert again[column] == first[column]
    assert again["queries"] == first["queries"]
    assert again["psor_total_iterations"] == first["psor_total_iterations"]
    assert again["exercise_classification_scale"] == first["exercise_classification_scale"]


def test_the_returned_surface_is_one_time_slice_and_scales_only_with_the_spot_grid() -> None:
    # A structural regression test on the O(N_S) design, not a measurement: what
    # comes back is the valuation-time slice, so its size is set by the spot
    # grid and is unmoved by taking four times as many time steps.
    coarse = _surface(spot_intervals=200, time_steps=100)
    fine_in_time = _surface(spot_intervals=200, time_steps=400)
    assert len(coarse["values"]) == len(fine_in_time["values"]) == 201
    assert fine_in_time["time_steps"] == 4 * coarse["time_steps"]
    assert len(fine_in_time["aligned_times"]) == len(coarse["aligned_times"]) == 2
    assert "value_history" not in fine_in_time
    assert "time_slices" not in fine_in_time


# --------------------------------------------------------------------------
# 18. Rejected surface requests
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"boundary_exclusion_nodes": 1}, "at least the regime stencil radius"),
        ({"boundary_exclusion_nodes": 600}, "leaves no Greek-eligible interior node"),
        ({"spot_maximum": 100.0}, "must exceed the strike"),
        # A malformed grid is reported before any query is examined.
        ({"spot_maximum": 100.0, "query_spots": [50.0]}, "must exceed the strike"),
        ({"strike": 0.0}, "strike must be finite and positive"),
        ({"continuous_carry": None}, "stated explicitly"),
        ({"volatility": 0.0}, "volatility must be finite and positive"),
        ({"spot_intervals": 3}, "at least four intervals"),
        ({"expiry_time": 0.0}, "expiry time must be finite and after"),
    ],
)
def test_invalid_surface_requests_are_rejected(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _surface(**overrides)  # type: ignore[arg-type]


def test_the_surface_has_no_spot_argument() -> None:
    # The solved domain must not depend on a requested spot, so the entry point
    # does not accept one at all.
    with pytest.raises(TypeError):
        pde_valuation_surface(
            option_type="call",
            exercise_style="european",
            spot=100.0,
            strike=100.0,
            valuation_time=0.0,
            expiry_time=1.0,
            volatility=0.2,
            continuous_carry=0.0,
            curve_times=[0.0, 1.0],
            curve_log_discounts=[0.0, -0.05],
            dividends=[],
            settlement="cash",
            contract_multiplier=100.0,
            spot_intervals=200,
            time_steps=100,
            spot_maximum=400.0,
            rannacher_steps=2,
            psor_tolerance=1.0e-11,
            psor_relaxation=1.2,
            psor_maximum_iterations=1000,
            boundary_exclusion_nodes=4,
            query_spots=[],
        )


# ---------------------------------------------------------------------------
# The mandatory normalized input echo
# ---------------------------------------------------------------------------

_ECHO_IDENTITY_FIELDS = (
    "option_type",
    "exercise_style",
    "strike",
    "valuation_time",
    "expiry_time",
    "volatility",
    "continuous_carry",
    "curve_times",
    "curve_log_discounts",
    "dividends",
    "settlement",
    "spot_intervals",
    "time_steps",
    "spot_maximum",
    "rannacher_steps",
    "psor_tolerance",
    "psor_relaxation",
    "psor_maximum_iterations",
    "boundary_exclusion_nodes",
)
_ECHO_METADATA_FIELDS = ("contract_multiplier", "dividends_declared")


def test_every_surface_result_carries_the_mandatory_input_echo() -> None:
    # The echo is mandatory, so a consumer never has to treat it as optional and
    # never has an "absent, therefore fine" success path.
    for style in ("european", "american"):
        for option_type in ("call", "put"):
            surface = _surface(option_type=option_type, exercise_style=style)
            assert "surface_input" in surface
            echo = surface["surface_input"]
            assert set(echo) == set(_ECHO_IDENTITY_FIELDS) | set(_ECHO_METADATA_FIELDS)


def test_the_input_echo_reports_the_normalized_inputs_the_solver_accepted() -> None:
    dividends = [(0.25, 1.5), (0.75, 2.0)]
    surface = _surface(
        option_type="put",
        exercise_style="american",
        strike=120.0,
        valuation_time=0.1,
        expiry_time=1.6,
        volatility=0.27,
        continuous_carry=0.03,
        curve_times=[0.0, 0.8, 1.6],
        curve_log_discounts=[0.0, -0.032, -0.064],
        dividends=dividends,
        settlement="physical",
        contract_multiplier=50.0,
        spot_intervals=400,
        time_steps=200,
        spot_maximum=480.0,
        rannacher_steps=3,
        psor_tolerance=1.0e-10,
        psor_relaxation=1.1,
        psor_maximum_iterations=40_000,
        boundary_exclusion_nodes=5,
    )
    echo = surface["surface_input"]
    assert echo["option_type"] == "put"
    assert echo["exercise_style"] == "american"
    assert echo["strike"] == 120.0
    assert echo["valuation_time"] == 0.1
    assert echo["expiry_time"] == 1.6
    assert echo["volatility"] == 0.27
    assert echo["continuous_carry"] == 0.03
    assert echo["curve_times"] == [0.0, 0.8, 1.6]
    assert echo["curve_log_discounts"] == [0.0, -0.032, -0.064]
    assert echo["dividends"] == [[0.25, 1.5], [0.75, 2.0]]
    assert echo["dividends_declared"] is True
    assert echo["settlement"] == "physical"
    assert echo["contract_multiplier"] == 50.0
    assert echo["spot_intervals"] == 400
    assert echo["time_steps"] == 200
    assert echo["spot_maximum"] == 480.0
    assert echo["rannacher_steps"] == 3
    assert echo["psor_tolerance"] == 1.0e-10
    assert echo["psor_relaxation"] == 1.1
    assert echo["psor_maximum_iterations"] == 40_000
    assert echo["boundary_exclusion_nodes"] == 5


def test_the_echo_reports_requested_grid_targets_not_the_adjusted_grid() -> None:
    # `spot_intervals` and `spot_maximum` are targets: the solver moves a node
    # onto the strike. The echo reports what was *accepted*, the diagnostics
    # report what was *used*, and the two are deliberately distinguishable.
    surface = _surface(strike=99.0, spot_intervals=300, spot_maximum=400.0)
    echo = surface["surface_input"]
    assert echo["spot_intervals"] == 300
    assert echo["spot_maximum"] == 400.0
    assert surface["spot_intervals"] != 300 or surface["spot_maximum"] != 400.0
    # The node lands on the strike to rounding, which is the placement rule; the
    # exact float is the accumulated product of the adjusted step.
    assert surface["spot_nodes"][surface["strike_node_index"]] == pytest.approx(99.0, abs=1e-9)


def test_an_empty_declared_dividend_schedule_still_reports_declared() -> None:
    echo = _surface(dividends=[])["surface_input"]
    assert echo["dividends"] == []
    assert echo["dividends_declared"] is True


def test_the_echo_is_bitwise_stable_across_repeated_solves() -> None:
    first = _surface(option_type="put", exercise_style="american")["surface_input"]
    second = _surface(option_type="put", exercise_style="american")["surface_input"]
    assert first == second


def test_the_echo_distinguishes_every_identity_bearing_input() -> None:
    base = _surface(option_type="put", exercise_style="american")["surface_input"]
    variations = {
        "option_type": _surface(option_type="call", exercise_style="american"),
        "exercise_style": _surface(option_type="put", exercise_style="european"),
        "volatility": _surface(option_type="put", exercise_style="american", volatility=0.25),
        "settlement": _surface(
            option_type="put", exercise_style="american", settlement="physical"
        ),
        "curve_log_discounts": _surface(
            option_type="put", exercise_style="american", rate=0.02
        ),
        "boundary_exclusion_nodes": _surface(
            option_type="put", exercise_style="american", boundary_exclusion_nodes=6
        ),
    }
    for field, surface in variations.items():
        assert surface["surface_input"][field] != base[field], field


def test_the_multiplier_is_echoed_but_changes_no_solved_value() -> None:
    # Reporting metadata: present in the echo, absent from the arithmetic.
    first = _surface(option_type="put", exercise_style="american", contract_multiplier=1.0)
    second = _surface(option_type="put", exercise_style="american", contract_multiplier=250.0)
    assert first["surface_input"]["contract_multiplier"] == 1.0
    assert second["surface_input"]["contract_multiplier"] == 250.0
    assert first["values"] == second["values"]
    assert first["deltas"] == second["deltas"]
    assert first["exercise_states"] == second["exercise_states"]


def test_the_scalar_api_is_unchanged_and_carries_no_echo() -> None:
    # The echo is additive to the surface result only. The scalar entry point is
    # untouched, including its exact key set.
    scalar = pde_price(
        option_type="put",
        exercise_style="american",
        spot=95.0,
        strike=100.0,
        valuation_time=0.0,
        expiry_time=1.0,
        volatility=0.2,
        continuous_carry=0.0,
        curve_times=[0.0, 1.0],
        curve_log_discounts=[0.0, -0.05],
        dividends=[],
        settlement="cash",
        contract_multiplier=100.0,
        spot_intervals=800,
        time_steps=400,
        spot_maximum=400.0,
        rannacher_steps=2,
        psor_tolerance=1.0e-11,
        psor_relaxation=1.2,
        psor_maximum_iterations=50_000,
    )
    assert "surface_input" not in scalar
    surface = _surface(option_type="put", exercise_style="american", query_spots=[95.0])
    assert scalar["price"] == surface["queries"][0]["value"]
