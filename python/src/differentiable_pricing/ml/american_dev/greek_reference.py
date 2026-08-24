"""Task 9H CRR Greek reference contract (section E): define it before comparing.

Which comparison this study makes
---------------------------------
Three objects are easy to conflate, and every report must say which one it means:

1. ``d(surrogate)/dx`` versus ``d(L_1024)/dx`` -- did the surrogate learn the
   mapping's derivative? **This is what this study measures**, and it is the
   right question for the immediate problem.
2. ``d(L_1024)/dx`` versus ``d(true American)/dx`` -- is the label operator's own
   derivative converged? A property of the label set. The depth check below is
   the only evidence produced here that bears on it.
3. the composition -- surrogate versus true American Greeks. What matters
   downstream, and **not established by this study**.

The reference operator
----------------------
The model was trained against ``L(x) = 0.5 * [CRR_N(x) + CRR_{N+1}(x)]``, so the
reference Greeks differentiate **that same operator**. Differencing ``CRR_N``
alone would differentiate a different function and inject the lattice's even/odd
oscillation straight into the reference.

Tree-internal Greeks are unavailable, and cannot be added
---------------------------------------------------------
``crr_diagnostics`` returns a price and exercise-region diagnostics only -- no
root Delta or Gamma. Adding them would mean editing ``cpp/src/binomial_tree.cpp``,
``cpp/include/dp/binomial_tree.hpp``, ``bindings/python/module.cpp`` and
``CMakeLists.txt``, **all four of which are digest-pinned** by
``configs/american_neural_pilot_protocol_v1.toml``; editing any of them breaks
the historical Task 9G protocol check that ``scripts/check.sh`` and CI run.

So every reference Greek here is a **central difference** of the adjacent-
averaged operator. That makes the bump-plateau study load-bearing rather than a
cross-check, and it makes Gamma's reference uncertainty -- lattice noise
``O(eps/h^2)`` -- the binding constraint on what section F can conclude.

Separate bumps for Delta and Gamma
----------------------------------
Central-difference error is truncation ``O(h^2)`` plus lattice noise ``O(eps/h)``
for Delta and ``O(eps/h^2)`` for Gamma, so the usable bump for Gamma is
generally **larger**. They are selected separately and are never reused across
Greeks.

Scope
-----
Delta, Gamma and Vega. **Theta and rho are explicitly out of scope** and are
recorded as declared non-claims rather than omitted silently. Theta is the
notable gap: the domain reaches 0.0192 years, and near-expiry ``dV/dT`` is where
American surrogates are hardest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import pairwise
from typing import Any, Final

import numpy as np

import differentiable_pricing as dp

GREEK_REFERENCE_SCHEMA: Final = "american-dev-greek-reference/1"

DEFAULT_OUTPUT: Final = "artifacts/task-9h/greeks/reference-contract-v1.json"

#: The label policy's own resolution. The reference differentiates the adjacent
#: average of this depth and the next, exactly as the training target was built.
REFERENCE_DEPTH: Final = 1024
#: The depth the convergence check re-derives the reference at.
DEEPER_DEPTH: Final = 2048

REFERENCE_OPERATOR: Final = "L(x) = 0.5 * [CRR_N(x) + CRR_{N+1}(x)], N = 1024"

#: Predeclared bump ladders. Selection happens by plateau, not by preference.
SPOT_BUMP_FRACTIONS: Final = (0.02, 0.01, 0.005, 0.0025, 0.001)
VOLATILITY_BUMPS: Final = (0.02, 0.01, 0.005, 0.0025)

#: The declared domain, matching the acceptance file's diagnostics section. A
#: bumped state that leaves it is excluded, which is the existing shape-battery
#: convention rather than a new rule.
SPOT_DOMAIN: Final = (50.0, 150.0)
VOLATILITY_DOMAIN: Final = (0.05, 0.8)
LOG_MONEYNESS_DOMAIN: Final = (-0.7, 0.7)

GREEKS: Final = ("delta", "gamma", "vega")

OUT_OF_SCOPE: Final = {
    "theta": (
        "not established. The declared domain reaches 0.0192 years and near-expiry dV/dT is "
        "where American surrogates are hardest; this phase makes no theta claim."
    ),
    "rho": "not established; this phase makes no rho claim.",
}

TREE_INTERNAL_UNAVAILABLE: Final = (
    "the CRR implementation exposes no tree-internal root Delta or Gamma, and adding them "
    "would require editing cpp/src/binomial_tree.cpp, cpp/include/dp/binomial_tree.hpp, "
    "bindings/python/module.cpp and CMakeLists.txt -- all four digest-pinned by the Task 9G "
    "protocol. Every reference Greek here is therefore a central difference, and its "
    "finite-difference uncertainty is reported beside every comparison."
)

WHICH_COMPARISON: Final = (
    "d(surrogate)/dx versus d(L_1024)/dx: whether the surrogate learned the label operator's "
    "derivative. NOT whether the label operator's own derivative is converged (the depth "
    "check is the only evidence here bearing on that), and NOT the composition against true "
    "American Greeks, which this study does not establish."
)


class GreekReferenceError(RuntimeError):
    """Raised when a reference Greek cannot be formed as declared."""


# ---------------------------------------------------------------------------
# The reference operator
# ---------------------------------------------------------------------------


def adjacent_average(
    option_type: Sequence[str],
    spot: np.ndarray,
    strike: np.ndarray,
    maturity: np.ndarray,
    rate: np.ndarray,
    dividend_yield: np.ndarray,
    volatility: np.ndarray,
    *,
    depth: int = REFERENCE_DEPTH,
    thread_count: int = 4,
) -> np.ndarray:
    """``L(x) = 0.5 * [CRR_depth(x) + CRR_{depth+1}(x)]``, the trained-against target.

    Both legs are priced, always. Differencing a single depth would
    differentiate a different function and put the lattice's even/odd
    oscillation directly into the reference.
    """
    rows = len(spot)
    if rows == 0:
        raise GreekReferenceError("the reference operator needs at least one state")

    def priced(steps: int) -> np.ndarray:
        return np.asarray(
            dp.crr_price_batch(
                list(option_type),
                ["american"] * rows,
                list(np.asarray(spot, dtype=np.float64)),
                list(np.asarray(strike, dtype=np.float64)),
                list(np.asarray(maturity, dtype=np.float64)),
                list(np.asarray(rate, dtype=np.float64)),
                list(np.asarray(dividend_yield, dtype=np.float64)),
                list(np.asarray(volatility, dtype=np.float64)),
                [steps] * rows,
                int(thread_count),
            )["price"],
            dtype=np.float64,
        )

    return 0.5 * (priced(int(depth)) + priced(int(depth) + 1))


def _state_arrays(states: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        "option_type": list(states["option_type"]),
        "spot": np.asarray(states["spot"], dtype=np.float64),
        "strike": np.asarray(states["strike"], dtype=np.float64),
        "maturity": np.asarray(states["maturity"], dtype=np.float64),
        "rate": np.asarray(states["rate"], dtype=np.float64),
        "dividend_yield": np.asarray(states["dividend_yield"], dtype=np.float64),
        "volatility": np.asarray(states["volatility"], dtype=np.float64),
    }


def _priced_at(
    arrays: Mapping[str, Any],
    *,
    spot: np.ndarray | None = None,
    volatility: np.ndarray | None = None,
    depth: int,
    thread_count: int,
) -> np.ndarray:
    return adjacent_average(
        arrays["option_type"],
        arrays["spot"] if spot is None else spot,
        arrays["strike"],
        arrays["maturity"],
        arrays["rate"],
        arrays["dividend_yield"],
        arrays["volatility"] if volatility is None else volatility,
        depth=depth,
        thread_count=thread_count,
    )


# ---------------------------------------------------------------------------
# Domain eligibility
# ---------------------------------------------------------------------------


def spot_eligible(
    arrays: Mapping[str, Any], fraction: float
) -> np.ndarray:
    """States whose spot bump stays inside the declared domain.

    The existing shape-battery convention, reused rather than restated:
    consistency with the battery beats a new rule.
    """
    spot = arrays["spot"]
    strike = arrays["strike"]
    low = spot * (1.0 - fraction)
    high = spot * (1.0 + fraction)
    return (
        (low >= SPOT_DOMAIN[0])
        & (high <= SPOT_DOMAIN[1])
        & (np.log(low / strike) >= LOG_MONEYNESS_DOMAIN[0])
        & (np.log(high / strike) <= LOG_MONEYNESS_DOMAIN[1])
    )


def volatility_eligible(arrays: Mapping[str, Any], bump: float) -> np.ndarray:
    volatility = arrays["volatility"]
    return (volatility - bump >= VOLATILITY_DOMAIN[0]) & (
        volatility + bump <= VOLATILITY_DOMAIN[1]
    )


# ---------------------------------------------------------------------------
# Central differences of the reference operator
# ---------------------------------------------------------------------------


def reference_delta(
    arrays: Mapping[str, Any], fraction: float, *, depth: int, thread_count: int
) -> np.ndarray:
    """``dL/dS`` by central difference at ``h = fraction * S``."""
    step = arrays["spot"] * float(fraction)
    up = _priced_at(
        arrays, spot=arrays["spot"] + step, depth=depth, thread_count=thread_count
    )
    down = _priced_at(
        arrays, spot=arrays["spot"] - step, depth=depth, thread_count=thread_count
    )
    return (up - down) / (2.0 * step)


def reference_gamma(
    arrays: Mapping[str, Any], fraction: float, *, depth: int, thread_count: int
) -> np.ndarray:
    """``d2L/dS2`` by the three-point central stencil at ``h = fraction * S``."""
    step = arrays["spot"] * float(fraction)
    up = _priced_at(
        arrays, spot=arrays["spot"] + step, depth=depth, thread_count=thread_count
    )
    centre = _priced_at(arrays, depth=depth, thread_count=thread_count)
    down = _priced_at(
        arrays, spot=arrays["spot"] - step, depth=depth, thread_count=thread_count
    )
    return (up - 2.0 * centre + down) / np.square(step)


def reference_vega(
    arrays: Mapping[str, Any], bump: float, *, depth: int, thread_count: int
) -> np.ndarray:
    """``dL/dsigma`` by central difference. Vega has no tree-internal shortcut."""
    up = _priced_at(
        arrays,
        volatility=arrays["volatility"] + float(bump),
        depth=depth,
        thread_count=thread_count,
    )
    down = _priced_at(
        arrays,
        volatility=arrays["volatility"] - float(bump),
        depth=depth,
        thread_count=thread_count,
    )
    return (up - down) / (2.0 * float(bump))


# ---------------------------------------------------------------------------
# Bump selection by plateau
# ---------------------------------------------------------------------------


def _plateau(
    values: Mapping[float, np.ndarray], eligible: Mapping[float, np.ndarray]
) -> dict[str, Any]:
    """Select the bump whose neighbours agree most closely, and quantify it.

    The plateau is where truncation error has fallen but lattice noise has not
    yet risen. The selected bump is the one minimizing the median absolute
    change to the next finer bump; the reference uncertainty is that change, and
    the plateau width is the range of bumps whose values stay inside it.

    Comparisons are made only over states eligible at **both** bumps, so a
    plateau is never read off a changing population.
    """
    ladder = sorted(values, reverse=True)
    if len(ladder) < 2:
        raise GreekReferenceError("a plateau study needs at least two bumps")
    changes: dict[float, float] = {}
    for coarse, fine in pairwise(ladder):
        shared = eligible[coarse] & eligible[fine]
        if not bool(shared.any()):
            continue
        difference = np.abs(values[coarse][shared] - values[fine][shared])
        finite = difference[np.isfinite(difference)]
        if finite.size:
            changes[fine] = float(np.median(finite))
    if not changes:
        raise GreekReferenceError("no bump pair shares an eligible state")
    selected = min(changes, key=lambda bump: changes[bump])
    uncertainty = changes[selected]
    inside = sorted(bump for bump, change in changes.items() if change <= 2.0 * uncertainty)
    finest = min(ladder)
    reached = selected != finest
    return {
        "ladder": [float(bump) for bump in ladder],
        "selected_bump": float(selected),
        "reference_uncertainty": uncertainty,
        "plateau_reached": bool(reached),
        "plateau_status": (
            "a plateau was reached: successive changes stopped falling before the ladder ran "
            "out, so the selected bump sits between truncation error and lattice noise"
            if reached
            else (
                "NO PLATEAU: the finest bump in the declared ladder was selected, so successive "
                "changes were still falling when the ladder ended. The ladder is truncation-"
                "dominated at its fine end, the reference uncertainty below is an UPPER BOUND "
                "rather than a plateau reading, and a finer ladder would be needed to resolve "
                "it. Reported rather than presented as a plateau."
            )
        ),
        "uncertainty_definition": (
            "median absolute change to the next finer bump over states eligible at both; the "
            "scale below which a disagreement is indistinguishable from finite-difference noise"
        ),
        "successive_changes": {f"{bump:g}": change for bump, change in sorted(changes.items())},
        "plateau_bumps": [float(bump) for bump in inside],
        "plateau_width": (
            float(max(inside) - min(inside)) if len(inside) > 1 else 0.0
        ),
        "eligible_rows_at_selected": int(eligible[selected].sum()),
    }


def bump_convergence(
    states: Mapping[str, np.ndarray],
    *,
    depth: int = REFERENCE_DEPTH,
    thread_count: int = 4,
) -> dict[str, Any]:
    """Run the declared bump ladders and select a bump per Greek.

    Delta and Gamma are selected **separately**: their noise scalings differ, so
    one bump cannot be optimal for both, and reusing one would silently accept a
    worse reference for whichever Greek lost.
    """
    arrays = _state_arrays(states)
    delta_values: dict[float, np.ndarray] = {}
    gamma_values: dict[float, np.ndarray] = {}
    spot_masks: dict[float, np.ndarray] = {}
    for fraction in SPOT_BUMP_FRACTIONS:
        spot_masks[fraction] = spot_eligible(arrays, fraction)
        delta_values[fraction] = reference_delta(
            arrays, fraction, depth=depth, thread_count=thread_count
        )
        gamma_values[fraction] = reference_gamma(
            arrays, fraction, depth=depth, thread_count=thread_count
        )
    vega_values: dict[float, np.ndarray] = {}
    vega_masks: dict[float, np.ndarray] = {}
    for bump in VOLATILITY_BUMPS:
        vega_masks[bump] = volatility_eligible(arrays, bump)
        vega_values[bump] = reference_vega(
            arrays, bump, depth=depth, thread_count=thread_count
        )
    return {
        "states": int(arrays["spot"].size),
        "depth": int(depth),
        "operator": REFERENCE_OPERATOR,
        "delta": _plateau(delta_values, spot_masks),
        "gamma": _plateau(gamma_values, spot_masks),
        "vega": _plateau(vega_values, vega_masks),
        "separate_bumps": (
            "Delta and Gamma are selected independently: central-difference noise scales as "
            "eps/h for Delta and eps/h^2 for Gamma, so the usable Gamma bump is generally "
            "larger and one bump cannot serve both"
        ),
    }


# ---------------------------------------------------------------------------
# Depth convergence
# ---------------------------------------------------------------------------


def depth_convergence(
    states: Mapping[str, np.ndarray],
    selection: Mapping[str, Any],
    *,
    thread_count: int = 4,
) -> dict[str, Any]:
    """Recompute price and Greeks at ``N = 2048/2049`` and compare with ``1024/1025``.

    **Prices as well as Greeks.** The price difference is a direct estimate of
    the American label's own discretization error, which is the quantity the
    tolerance's European-leg proxy stands in for; it validates that safety
    factor retrospectively and **does not adjust it**.

    For the Greeks, a difference exceeding the bump plateau width means the
    reference Greek is not depth-converged and every downstream comparison
    inherits that uncertainty. This matters more for derivatives than for
    prices: the label surface carries a state-dependent discretization error,
    and differentiating amplifies its variation.
    """
    arrays = _state_arrays(states)
    shallow_price = _priced_at(arrays, depth=REFERENCE_DEPTH, thread_count=thread_count)
    deep_price = _priced_at(arrays, depth=DEEPER_DEPTH, thread_count=thread_count)
    scale = arrays["spot"] * np.exp(-arrays["dividend_yield"] * arrays["maturity"])
    price_gap = np.abs(deep_price - shallow_price)

    report: dict[str, Any] = {
        "states": int(arrays["spot"].size),
        "shallow_depth": REFERENCE_DEPTH,
        "deep_depth": DEEPER_DEPTH,
        "price": {
            "maximum_absolute_difference": float(price_gap.max()),
            "median_absolute_difference": float(np.median(price_gap)),
            "maximum_normalized_difference": float((price_gap / scale).max()),
            "median_normalized_difference": float(np.median(price_gap / scale)),
            "purpose": (
                "a direct estimate of the American label's own discretization error, which "
                "the tolerance's European-leg proxy stands in for. It validates the declared "
                "safety factor retrospectively and does not adjust it."
            ),
        },
    }
    for greek in GREEKS:
        bump = float(selection[greek]["selected_bump"])
        uncertainty = float(selection[greek]["reference_uncertainty"])
        if greek == "delta":
            shallow = reference_delta(
                arrays, bump, depth=REFERENCE_DEPTH, thread_count=thread_count
            )
            deep = reference_delta(arrays, bump, depth=DEEPER_DEPTH, thread_count=thread_count)
            mask = spot_eligible(arrays, bump)
        elif greek == "gamma":
            shallow = reference_gamma(
                arrays, bump, depth=REFERENCE_DEPTH, thread_count=thread_count
            )
            deep = reference_gamma(arrays, bump, depth=DEEPER_DEPTH, thread_count=thread_count)
            mask = spot_eligible(arrays, bump)
        else:
            shallow = reference_vega(
                arrays, bump, depth=REFERENCE_DEPTH, thread_count=thread_count
            )
            deep = reference_vega(arrays, bump, depth=DEEPER_DEPTH, thread_count=thread_count)
            mask = volatility_eligible(arrays, bump)
        difference = np.abs(deep[mask] - shallow[mask])
        finite = difference[np.isfinite(difference)]
        maximum = float(finite.max()) if finite.size else None
        report[greek] = {
            "selected_bump": bump,
            "eligible_states": int(mask.sum()),
            "maximum_absolute_difference": maximum,
            "median_absolute_difference": float(np.median(finite)) if finite.size else None,
            "reference_uncertainty": uncertainty,
            "depth_converged": (
                None if maximum is None else bool(maximum <= uncertainty)
            ),
            "interpretation": (
                "a depth difference larger than the bump-plateau uncertainty means the "
                "reference Greek is not depth-converged and every downstream comparison "
                "inherits that uncertainty"
            ),
        }
    return report


def contract(
    selection: Mapping[str, Any], convergence: Mapping[str, Any]
) -> dict[str, Any]:
    """The reference contract every later Greek comparison cites."""
    return {
        "schema_version": GREEK_REFERENCE_SCHEMA,
        "operator": REFERENCE_OPERATOR,
        "which_comparison": WHICH_COMPARISON,
        "tree_internal_greeks": TREE_INTERNAL_UNAVAILABLE,
        "scope": list(GREEKS),
        "out_of_scope": OUT_OF_SCOPE,
        "bump_selection": selection,
        "depth_convergence": convergence,
        "domain_eligibility": (
            "bumped states leaving the declared domain are excluded, reusing the existing "
            "shape-battery convention rather than introducing a new rule"
        ),
        "reporting_rule": (
            "every Greek error is reported alongside the reference's own uncertainty; without "
            "it the question 'is this CRR finite-difference noise?' is unanswerable by "
            "construction, and a conclusion may be drawn only where disagreement exceeds it"
        ),
    }


# ---------------------------------------------------------------------------
# The declared state sets
# ---------------------------------------------------------------------------

#: The bump-plateau study's fixed contract set, and the depth check's subset.
#: Declared as counts plus a deterministic construction rather than as a literal
#: table: the construction is reproducible and the table would not be checkable.
PLATEAU_STATES: Final = 200
DEPTH_CHECK_STATES: Final = 100
STATE_SEED: Final = 20260823

STATE_CONSTRUCTION: Final = (
    "a fixed NumPy PCG64 draw at seed 20260823 over the declared domain interior, with both "
    "option types in equal proportion. Label-free and partition-free: no dataset row and no "
    "partition-derived sampling location is used, so the reference contract is not tuned to "
    "the rows it will later be applied to."
)


def declared_states(count: int = PLATEAU_STATES) -> dict[str, np.ndarray]:
    """The declared contract set, drawn deterministically from the domain interior.

    Kept inside a margin of the declared boundaries so that the coarsest bump in
    each ladder still has eligible states; a set pressed against the boundary
    would make the plateau study a study of the eligibility rule instead.
    """
    generator = np.random.default_rng(STATE_SEED)
    rows = int(count)
    spot = generator.uniform(SPOT_DOMAIN[0] * 1.10, SPOT_DOMAIN[1] * 0.90, rows)
    moneyness = generator.uniform(
        LOG_MONEYNESS_DOMAIN[0] * 0.80, LOG_MONEYNESS_DOMAIN[1] * 0.80, rows
    )
    volatility = generator.uniform(
        VOLATILITY_DOMAIN[0] + 0.03, VOLATILITY_DOMAIN[1] - 0.03, rows
    )
    option_type = np.where(np.arange(rows) % 2 == 0, "call", "put")
    return {
        "option_type": option_type,
        "spot": spot,
        "strike": spot * np.exp(-moneyness),
        "maturity": generator.uniform(0.05, 2.90, rows),
        "rate": generator.uniform(-0.015, 0.115, rows),
        "dividend_yield": generator.uniform(0.0, 0.115, rows),
        "volatility": volatility,
    }


def analyze(
    project_root: Any,
    *,
    output: str = DEFAULT_OUTPUT,
    overwrite: bool = False,
    thread_count: int = 4,
    plateau_states: int = PLATEAU_STATES,
    depth_states: int = DEPTH_CHECK_STATES,
) -> dict[str, Any]:
    """Run the bump ladders and the depth check, and publish the contract."""
    from pathlib import Path

    from ..artifact import write_json_atomic
    from .frozen import _guarded_artifact_path, diagnostic_provenance

    project_root = Path(project_root)
    path = _guarded_artifact_path(project_root, output, where="greek reference path")
    if path.exists() and not overwrite:
        raise GreekReferenceError(f"'{output}' already exists; pass overwrite to replace it")
    states = declared_states(plateau_states)
    selection = bump_convergence(states, thread_count=thread_count)
    subset = {key: value[:depth_states] for key, value in states.items()}
    convergence = depth_convergence(subset, selection, thread_count=thread_count)
    report = {
        **contract(selection, convergence),
        "states": {
            "plateau_states": int(plateau_states),
            "depth_check_states": int(depth_states),
            "seed": STATE_SEED,
            "construction": STATE_CONSTRUCTION,
        },
        "provenance": diagnostic_provenance(project_root),
    }
    write_json_atomic(path, report, overwrite=overwrite)
    return report


def compact_reference(report: Mapping[str, Any]) -> dict[str, Any]:
    """The small, agent-readable view: selected bumps and their uncertainties."""
    return {
        "protocol_commit": report.get("provenance", {}).get("protocol_commit"),
        "operator": report["operator"],
        "bumps": {
            greek: {
                "selected_bump": report["bump_selection"][greek]["selected_bump"],
                "reference_uncertainty": report["bump_selection"][greek][
                    "reference_uncertainty"
                ],
                "plateau_reached": report["bump_selection"][greek]["plateau_reached"],
                "depth_converged": report["depth_convergence"][greek]["depth_converged"],
            }
            for greek in GREEKS
        },
        "label_price_discretization": {
            "maximum_normalized_difference": report["depth_convergence"]["price"][
                "maximum_normalized_difference"
            ],
            "median_normalized_difference": report["depth_convergence"]["price"][
                "median_normalized_difference"
            ],
        },
        "out_of_scope": sorted(report["out_of_scope"]),
    }
