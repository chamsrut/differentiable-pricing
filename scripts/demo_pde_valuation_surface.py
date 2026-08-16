#!/usr/bin/env python3
"""Task 9C-C1 demonstration: many spot labels from one PDE solve.

Two small cases are run, each asking for 32 interior spots from a single
backward induction:

- a smooth European call, checked against analytic Black-Scholes price, delta
  and gamma;
- an American put, whose price is checked against the independent CRR engine
  and whose delta and gamma are compared against the same PDE engine on a
  doubled grid.

This is a correctness demonstration, not a throughput measurement. The elapsed
times below are wall clock on one core of whatever machine ran them; they are
not a production-feasibility claim and must not be extrapolated to a label
budget. The rows one surface produces are correlated outputs of one solve, and
the American delta and gamma differences are a refinement gap between two grids
of the same engine, not an error against truth.

    python scripts/demo_pde_valuation_surface.py
"""

from __future__ import annotations

import time
from collections import Counter

from differentiable_pricing import black_scholes, crr_price, pde_valuation_surface

STRIKE = 100.0
EXPIRY = 1.0
RATE = 0.05
VOLATILITY = 0.2
SPOT_MAXIMUM = 400.0
BOUNDARY_EXCLUSION_NODES = 4
QUERY_SPOTS = [70.0 + 2.0 * index for index in range(32)]
CRR_REFERENCE_STEPS = 4096


def _surface(
    *,
    option_type: str,
    exercise_style: str,
    continuous_carry: float,
    spot_intervals: int,
    time_steps: int,
    query_spots: list[float],
) -> dict[str, object]:
    return pde_valuation_surface(
        option_type=option_type,
        exercise_style=exercise_style,
        strike=STRIKE,
        valuation_time=0.0,
        expiry_time=EXPIRY,
        volatility=VOLATILITY,
        continuous_carry=continuous_carry,
        curve_times=[0.0, EXPIRY],
        curve_log_discounts=[0.0, -RATE * EXPIRY],
        dividends=[],
        settlement="cash",
        contract_multiplier=100.0,
        spot_intervals=spot_intervals,
        time_steps=time_steps,
        spot_maximum=SPOT_MAXIMUM,
        rannacher_steps=2,
        psor_tolerance=1.0e-11,
        psor_relaxation=1.2,
        psor_maximum_iterations=50_000,
        boundary_exclusion_nodes=BOUNDARY_EXCLUSION_NODES,
        query_spots=query_spots,
    )


def _report(title: str, surface: dict[str, object], elapsed: float) -> None:
    queries = surface["queries"]
    eligible = [query for query in queries if query["greek_eligible"]]
    node_states = Counter(surface["exercise_states"])
    query_states = Counter(
        "unavailable" if query["exercise_state"] is None else query["exercise_state"]
        for query in queries
    )
    print(f"{title}")
    print(f"  grid                     {surface['spot_intervals']} x {surface['time_steps']}")
    print(f"  spot step / domain       {surface['spot_step']} / {surface['spot_maximum']}")
    print(f"  queries                  {surface['query_count']}")
    print(f"  backward inductions      {surface['backward_inductions']}")
    print(f"  implicit linear solves   {surface['linear_solves']}")
    print(f"  PSOR solves / iterations {surface['psor_solves']} / "
          f"{surface['psor_total_iterations']}")
    print(f"  max relative LCP residual {surface['maximum_relative_lcp_residual']:.3e} "
          f"(tolerance {surface['psor_tolerance']:.1e})")
    print(f"  classification scale     {surface['exercise_classification_scale']:.3e}")
    print(f"  Greek-eligible queries   {len(eligible)} eligible, "
          f"{len(queries) - len(eligible)} ineligible")
    print(f"  query exercise states    {dict(sorted(query_states.items()))}")
    print(f"  node exercise states     {dict(sorted(node_states.items()))}")
    print(f"  eligible nodes           {sum(surface['greek_eligible'])} of "
          f"{len(surface['spot_nodes'])}")
    print(f"  elapsed                  {elapsed:.2f} s")


def demonstrate_european_call() -> None:
    started = time.perf_counter()
    surface = _surface(
        option_type="call",
        exercise_style="european",
        continuous_carry=0.0,
        spot_intervals=800,
        time_steps=400,
        query_spots=QUERY_SPOTS,
    )
    elapsed = time.perf_counter() - started

    worst = {"price": 0.0, "delta": 0.0, "gamma": 0.0}
    for query in surface["queries"]:
        reference = black_scholes(
            "call", query["spot"], STRIKE, EXPIRY, RATE, 0.0, VOLATILITY
        )
        worst["price"] = max(worst["price"], abs(query["value"] - reference["price"]))
        if not query["greek_eligible"]:
            continue
        worst["delta"] = max(worst["delta"], abs(query["delta"] - reference["delta"]))
        worst["gamma"] = max(worst["gamma"], abs(query["gamma"] - reference["gamma"]))

    _report("European call, analytic Black-Scholes reference", surface, elapsed)
    print(f"  max |price - BS|         {worst['price']:.3e}")
    print(f"  max |delta - BS|         {worst['delta']:.3e}")
    print(f"  max |gamma - BS|         {worst['gamma']:.3e}")
    print()


def demonstrate_american_put() -> None:
    started = time.perf_counter()
    surface = _surface(
        option_type="put",
        exercise_style="american",
        continuous_carry=0.0,
        spot_intervals=800,
        time_steps=400,
        query_spots=QUERY_SPOTS,
    )
    elapsed = time.perf_counter() - started

    # The same engine on a doubled grid. This is a refinement gap between two
    # discretizations, not an error against truth: no closed form and no second
    # engine in this repository prices an American Greek.
    refined = _surface(
        option_type="put",
        exercise_style="american",
        continuous_carry=0.0,
        spot_intervals=1600,
        time_steps=800,
        query_spots=QUERY_SPOTS,
    )

    worst = {"price": 0.0, "delta": 0.0, "gamma": 0.0, "intrinsic_delta": 0.0}
    for query, control in zip(surface["queries"], refined["queries"], strict=True):
        reference = crr_price(
            "put",
            "american",
            query["spot"],
            STRIKE,
            EXPIRY,
            RATE,
            0.0,
            VOLATILITY,
            CRR_REFERENCE_STEPS,
        )
        worst["price"] = max(worst["price"], abs(query["value"] - reference["price"]))
        if not (query["greek_eligible"] and control["greek_eligible"]):
            continue
        worst["delta"] = max(worst["delta"], abs(query["delta"] - control["delta"]))
        worst["gamma"] = max(worst["gamma"], abs(query["gamma"] - control["gamma"]))
        if query["exercise_state"] == "exercise":
            worst["intrinsic_delta"] = max(
                worst["intrinsic_delta"], abs(query["delta"] + 1.0)
            )

    _report("American put, CRR price reference and refined-grid Greek control", surface, elapsed)
    print(f"  max |price - CRR({CRR_REFERENCE_STEPS})|  {worst['price']:.3e}")
    print(f"  max |delta - 1600x800|   {worst['delta']:.3e}")
    print(f"  max |gamma - 1600x800|   {worst['gamma']:.3e}")
    print(f"  max |delta + 1| on exercise rows {worst['intrinsic_delta']:.3e}")
    print()


def main() -> None:
    print(__doc__.splitlines()[0])
    print()
    demonstrate_european_call()
    demonstrate_american_put()
    print("One surface produces many rows. Those rows are correlated outputs of one solve")
    print("and are not independent observations; grouped partitioning is task 9C-C2.")


if __name__ == "__main__":
    main()
