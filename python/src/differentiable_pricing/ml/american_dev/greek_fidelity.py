"""Task 9H Greek fidelity (sections F and G): raw versus deployed, on three grids.

The mandatory split
-------------------
Every grid computes **both** raw-network Greeks and deployed-E2c Greeks, by
autograd, at identical states. The split is not optional -- it is the decisive
decision input:

* raw good, deployed bad -> the problem is the projection, not the learned
  function; the next intervention is a **deployment-head change**, costing no
  neural attempt;
* raw bad, deployed bad -> the learned function itself has a derivative-fidelity
  problem, and a learning intervention is indicated.

Those lead to different section H branches and cannot be separated without it.

Grid 1 -- validation and H1
---------------------------
Absolute surrogate-versus-label-operator fidelity. **Genuinely limited by CRR
finite-difference uncertainty**, because no tree-internal Greeks exist (see
:mod:`.greek_reference`). A conclusion may be drawn only where the disagreement
exceeds that uncertainty, and every metric is reported beside it.

Error conventions, predeclared:

* **Delta: absolute error only.** It is dimensionless in ``[-1, 1]``, so a
  relative measure adds nothing.
* **Gamma and Vega: relative error only where the reference magnitude exceeds
  ``10x`` the reference uncertainty.** That makes the denominator floor derived
  rather than asserted. The excluded fraction is reported.

Grid 2 -- dense spot crossover
------------------------------
**The primary statistic here is not a bumped CRR Gamma.** It is

    Gamma_deployed_autograd - Gamma_raw_autograd

at identical physical states. Both are exact derivatives of the frozen surrogate
implementation, so the difference isolates the deployment transformation
directly and carries **no finite-difference uncertainty at all**. Making a
high-resolution bumped CRR Gamma the decisive quantity would put the reference's
own noise -- ``O(eps/h^2)`` -- into precisely the narrow, high-curvature band the
study exists to resolve. CRR Gamma is used here only as an **ambient-scale
reference**, reported with its uncertainty.

Two curvature sites, located and reported separately:

**A. raw versus deployed floor.** For ``u_dep = floor + tau*softplus(z)`` with
``z = (u_dir - floor)/tau``, writing ``s = sigmoid(z)``::

    d2u_dep/dS2 = (1-s)*floor'' + s*u_dir'' + [s(1-s)/tau] * (u_dir' - floor')^2

The third term is the predicted spike: of order ``1e4`` at ``tau = 1e-4``.

**B. European+delta versus intrinsic, inside the floor.** The floor is itself a
smooth maximum at the same ``tau``, so ``floor''`` carries its own spike where
its two legs cross. It enters with weight ``(1-s)`` -- suppressed where the
network dominates, **not** suppressed in the crossover band this study scans.
This site is not covered by the existing shape battery either.

Grid 2b -- dense volatility crossover, and Vega degeneracy
-----------------------------------------------------------
Where the intrinsic leg dominates the floor, the deployed function is
independent of ``sigma``, so deployed Vega and Gamma are **exactly zero by
construction**. Deep in the exercise region that is correct. But the floor binds
wherever the network under-predicts, not only where truth equals intrinsic, so
any **excess** zero-Vega region over the reference is a manufactured failure
region: an implied-volatility Newton solve landing there has a zero Jacobian.

Section G
---------
The finite-bump shape battery and pointwise autograd signs are **different
functionals**. A model can pass one and fail the other, so they are computed
separately and reported side by side to determine whether the battery is
actually predictive of pointwise Greek quality.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

import numpy as np
import torch

from ..config import FEATURE_ORDER
from .frozen import (
    MarginOverrideModel,
    RawNetworkModel,
    assert_floor_head,
    deployed_floor,
    direct_normalized,
    floor_leg_weight,
    projection_weight,
)
from .greek_reference import GREEKS, REFERENCE_OPERATOR, WHICH_COMPARISON
from .representation import AmericanDevPriceModel, discounted_spot

GREEK_FIDELITY_SCHEMA: Final = "american-dev-greek-fidelity/1"

DEFAULT_GRID1_OUTPUT: Final = "artifacts/task-9h/greeks/grid1-fidelity-v1.json"
DEFAULT_GRID2_OUTPUT: Final = "artifacts/task-9h/greeks/grid2-spot-crossover-v1.json"
DEFAULT_GRID2B_OUTPUT: Final = "artifacts/task-9h/greeks/grid2b-vol-crossover-v1.json"

SPOT_INDEX: Final = FEATURE_ORDER.index("spot")
VOLATILITY_INDEX: Final = FEATURE_ORDER.index("volatility")

#: The two predictions every grid measures. Never one without the other.
PREDICTIONS: Final = ("raw_network", "deployed")

#: Gamma and Vega get a relative error only where the reference magnitude clears
#: this multiple of the reference's own uncertainty. Derived, not asserted.
RELATIVE_ERROR_UNCERTAINTY_MULTIPLE: Final = 10.0

#: Declared small threshold for calling a sensitivity degenerate. It is the
#: normalized-vega degeneracy scale a Newton solve would already struggle with.
DEGENERATE_SENSITIVITY: Final = 1.0e-8

#: Dense scan resolutions, predeclared.
SPOT_SCAN_SPACING: Final = 0.01
VOLATILITY_SCAN_SPACING: Final = 0.001
SPOT_SCAN_HALF_WIDTH: Final = 0.50
VOLATILITY_SCAN_HALF_WIDTH: Final = 0.05

#: The two curvature sites Grid 2 locates separately.
CROSSOVER_SITES: Final = ("raw_minus_floor", "european_plus_delta_minus_intrinsic")

GRID2_PRIMARY_STATISTIC: Final = (
    "Gamma_deployed_autograd - Gamma_raw_autograd at identical physical states. Both are "
    "exact derivatives of the frozen surrogate implementation, so the difference isolates the "
    "deployment transformation and carries no finite-difference uncertainty. CRR Gamma is "
    "used on this grid only as an ambient-scale reference, reported with its uncertainty."
)

DELTA_CONVENTION: Final = (
    "absolute error only: Delta is dimensionless in [-1, 1], so a relative measure adds nothing"
)
RELATIVE_CONVENTION: Final = (
    "relative error only where the reference magnitude exceeds 10x the reference uncertainty "
    "from the Greek reference contract, so the denominator floor is derived rather than "
    "asserted; the excluded fraction is reported"
)


class GreekFidelityError(RuntimeError):
    """Raised when a Greek comparison cannot be made as declared."""


# ---------------------------------------------------------------------------
# Autograd Greeks of the frozen surrogate
# ---------------------------------------------------------------------------


def autograd_greeks(
    pricer: Callable[[torch.Tensor], torch.Tensor], physical: np.ndarray
) -> dict[str, np.ndarray]:
    """Delta, Gamma and Vega of one surrogate view, by automatic differentiation.

    Exact derivatives of the implementation, not finite differences of it: the
    whole point of section F's raw-versus-deployed split is that both sides are
    the same kind of object, so any difference between them is the projection
    and not two different discretizations.

    Gamma needs a second derivative, so the first backward pass builds a graph.
    """
    inputs = torch.as_tensor(np.ascontiguousarray(physical), dtype=torch.float64)
    inputs.requires_grad_(True)
    price = pricer(inputs)
    (first,) = torch.autograd.grad(price.sum(), inputs, create_graph=True)
    delta = first[:, SPOT_INDEX]
    vega = first[:, VOLATILITY_INDEX]
    (second,) = torch.autograd.grad(delta.sum(), inputs, retain_graph=False)
    gamma = second[:, SPOT_INDEX]
    return {
        "price": price.detach().numpy(),
        "delta": delta.detach().numpy(),
        "gamma": gamma.detach().numpy(),
        "vega": vega.detach().numpy(),
    }


def surrogate_views(
    model: AmericanDevPriceModel, european_margin: float | None = None
) -> dict[str, Callable[[torch.Tensor], torch.Tensor]]:
    """The raw and deployed pricers, as differentiable callables."""
    assert_floor_head(model)
    margin = model.european_margin if european_margin is None else float(european_margin)
    return {
        "raw_network": RawNetworkModel(model),
        "deployed": MarginOverrideModel(model, margin),
    }


def both_greeks(
    model: AmericanDevPriceModel,
    physical: np.ndarray,
    european_margin: float | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """The mandatory raw/deployed split, computed together at identical states."""
    return {
        name: autograd_greeks(view, physical)
        for name, view in surrogate_views(model, european_margin).items()
    }


# ---------------------------------------------------------------------------
# Error conventions
# ---------------------------------------------------------------------------


def _statistics(error: np.ndarray) -> dict[str, Any]:
    finite = error[np.isfinite(error)]
    if finite.size == 0:
        return {"rows": 0}
    absolute = np.abs(finite)
    return {
        "rows": int(finite.size),
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(finite)))),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "p99_absolute_error": float(np.quantile(absolute, 0.99)),
        "maximum_absolute_error": float(absolute.max()),
        "signed_bias": float(finite.mean()),
    }


def compare_greek(
    greek: str,
    surrogate: np.ndarray,
    reference: np.ndarray,
    uncertainty: float,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """One Greek's error, under the predeclared convention for that Greek.

    Delta is absolute only. Gamma and Vega additionally carry a relative error,
    computed **only** where the reference magnitude exceeds ten times the
    reference's own uncertainty -- so the denominator floor comes from the
    measured finite-difference noise rather than from a chosen epsilon.

    Every entry carries the reference uncertainty and the fraction of rows where
    the disagreement exceeds it, because a disagreement smaller than the
    reference's own noise supports no conclusion.
    """
    if greek not in GREEKS:
        raise GreekFidelityError(f"unknown Greek {greek!r}; the scope is {list(GREEKS)}")
    selected = np.ones(surrogate.shape, dtype=bool) if mask is None else np.asarray(mask, bool)
    error = np.asarray(surrogate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    error = error[selected]
    magnitude = np.abs(np.asarray(reference, dtype=np.float64)[selected])
    resolved = np.abs(error) > uncertainty
    entry: dict[str, Any] = {
        "absolute": _statistics(error),
        "reference_uncertainty": float(uncertainty),
        "rows_exceeding_reference_uncertainty": int(resolved.sum()),
        "fraction_exceeding_reference_uncertainty": (
            float(resolved.mean()) if resolved.size else None
        ),
        "conclusions_only_where_disagreement_exceeds_uncertainty": True,
        "sign_disagreement_rate": (
            float(
                np.mean(
                    np.sign(np.asarray(surrogate, dtype=np.float64)[selected])
                    != np.sign(np.asarray(reference, dtype=np.float64)[selected])
                )
            )
            if error.size
            else None
        ),
    }
    if greek == "delta":
        entry["convention"] = DELTA_CONVENTION
        entry["relative"] = None
        return entry
    entry["convention"] = RELATIVE_CONVENTION
    floor = RELATIVE_ERROR_UNCERTAINTY_MULTIPLE * float(uncertainty)
    usable = magnitude > floor
    entry["relative_denominator_floor"] = floor
    entry["relative_rows"] = int(usable.sum())
    entry["relative_excluded_fraction"] = (
        float((~usable).mean()) if usable.size else None
    )
    entry["relative"] = (
        _statistics(error[usable] / magnitude[usable]) if bool(usable.any()) else None
    )
    return entry


# ---------------------------------------------------------------------------
# Degeneracy (Grid 1 and Grid 2b)
# ---------------------------------------------------------------------------


def degeneracy(
    surrogate: np.ndarray,
    reference: np.ndarray,
    *,
    threshold: float = DEGENERATE_SENSITIVITY,
) -> dict[str, Any]:
    """Degenerate-sensitivity fractions, and **the excess** over the reference.

    The excess is the quantity that matters: rows where the deployed sensitivity
    is degenerate but the reference's is not are a manufactured failure region,
    not a property of the option. An implied-volatility Newton solve landing
    there has a zero Jacobian and must fall back to bracketing.
    """
    surrogate_degenerate = np.abs(np.asarray(surrogate, dtype=np.float64)) <= threshold
    reference_degenerate = np.abs(np.asarray(reference, dtype=np.float64)) <= threshold
    excess = surrogate_degenerate & ~reference_degenerate
    return {
        "threshold": float(threshold),
        "surrogate_degenerate_rows": int(surrogate_degenerate.sum()),
        "surrogate_degenerate_fraction": float(surrogate_degenerate.mean()),
        "reference_degenerate_rows": int(reference_degenerate.sum()),
        "reference_degenerate_fraction": float(reference_degenerate.mean()),
        "excess_rows": int(excess.sum()),
        "excess_fraction": float(excess.mean()),
        "excess_meaning": (
            "rows where the surrogate's sensitivity is degenerate but the reference's is not: "
            "a manufactured failure region for an implied-volatility solve, not a property of "
            "the contract"
        ),
    }


# ---------------------------------------------------------------------------
# Grid 2 and 2b: locating the crossover
# ---------------------------------------------------------------------------


def crossover_signals(
    model: AmericanDevPriceModel,
    physical: torch.Tensor,
    european_margin: float | None = None,
) -> dict[str, np.ndarray]:
    """The two quantities whose sign changes define the curvature sites.

    Both are **label-free**: computable from the frozen model and the contract
    inputs alone, so locating a crossing costs no lattice.
    """
    assert_floor_head(model)
    with torch.no_grad():
        floor = deployed_floor(model, physical, european_margin)
        direct = direct_normalized(model, physical)
        leg = floor_leg_weight(model, physical, european_margin)
        weight = projection_weight(model, physical, european_margin)
    return {
        "raw_minus_floor": (direct - floor).numpy(),
        # w = sigmoid((e_leg - i_leg)/tau) crosses 0.5 exactly where the legs do,
        # so w - 0.5 is a sign-preserving stand-in that never overflows.
        "european_plus_delta_minus_intrinsic": (leg - 0.5).numpy(),
        "projection_weight": weight.numpy(),
        "floor_leg_weight": leg.numpy(),
    }


def _sign_changes(values: np.ndarray) -> list[int]:
    signs = np.sign(values)
    return [
        index
        for index in range(1, signs.size)
        if signs[index - 1] != 0 and signs[index] != 0 and signs[index - 1] != signs[index]
    ]


def locate_crossings(
    model: AmericanDevPriceModel,
    contract: Mapping[str, float],
    site: str,
    *,
    axis: str,
    low: float,
    high: float,
    points: int = 400,
    european_margin: float | None = None,
) -> dict[str, Any]:
    """Existence check first: does this contract cross inside the declared domain?

    Reports the crossing count rather than silently taking the first, and
    reports contracts with **no** crossing so they can be substituted rather
    than quietly dropped.
    """
    if site not in CROSSOVER_SITES:
        raise GreekFidelityError(f"unknown crossover site {site!r}")
    if axis not in ("spot", "volatility"):
        raise GreekFidelityError(f"unknown scan axis {axis!r}")
    grid = np.linspace(float(low), float(high), int(points))
    states = _sweep(contract, axis, grid)
    signals = crossover_signals(
        model, torch.as_tensor(states, dtype=torch.float64), european_margin
    )
    values = signals[site]
    indices = _sign_changes(values)
    crossings = [float(0.5 * (grid[index - 1] + grid[index])) for index in indices]
    return {
        "site": site,
        "axis": axis,
        "search_range": [float(low), float(high)],
        "search_points": int(points),
        "crossing_exists": bool(crossings),
        "crossing_count": len(crossings),
        "crossings": crossings,
        "multiple_crossings": len(crossings) > 1,
        "note": (
            "no sign change inside the declared domain for this contract; substitute another"
            if not crossings
            else "crossings reported in full rather than taking the first silently"
        ),
    }


def _sweep(contract: Mapping[str, float], axis: str, grid: np.ndarray) -> np.ndarray:
    """One contract swept along one axis, as physical feature rows."""
    rows = grid.size
    states = np.empty((rows, len(FEATURE_ORDER)), dtype=np.float64)
    for position, name in enumerate(FEATURE_ORDER):
        states[:, position] = float(contract[name])
    states[:, SPOT_INDEX if axis == "spot" else VOLATILITY_INDEX] = grid
    return states


def scan_crossover(
    model: AmericanDevPriceModel,
    contract: Mapping[str, float],
    centre: float,
    *,
    axis: str,
    half_width: float,
    spacing: float,
    european_margin: float | None = None,
) -> dict[str, Any]:
    """Dense scan across one crossing, with the primary projection statistic.

    The decisive quantity is ``Gamma_deployed - Gamma_raw`` from autograd: both
    are exact derivatives of the frozen implementation, so their difference is
    the projection's own contribution and carries no finite-difference
    uncertainty. A high-resolution bumped CRR Gamma would put the reference's
    ``O(eps/h^2)`` noise into exactly the narrow band being resolved.
    """
    points = round(2.0 * half_width / spacing) + 1
    grid = np.linspace(centre - half_width, centre + half_width, points)
    states = _sweep(contract, axis, grid)
    greeks = both_greeks(model, states, european_margin)
    signals = crossover_signals(
        model, torch.as_tensor(states, dtype=torch.float64), european_margin
    )
    gamma_projection = greeks["deployed"]["gamma"] - greeks["raw_network"]["gamma"]
    delta_projection = greeks["deployed"]["delta"] - greeks["raw_network"]["delta"]
    vega_projection = greeks["deployed"]["vega"] - greeks["raw_network"]["vega"]
    return {
        "axis": axis,
        "centre": float(centre),
        "half_width": float(half_width),
        "spacing": float(spacing),
        "points": points,
        "grid": [float(value) for value in grid],
        "primary_statistic": GRID2_PRIMARY_STATISTIC,
        "gamma_projection_difference": {
            "maximum_absolute": float(np.abs(gamma_projection).max()),
            "maximum_signed": float(gamma_projection.max()),
            "minimum_signed": float(gamma_projection.min()),
            "at_grid_value": float(grid[int(np.argmax(np.abs(gamma_projection)))]),
            "carries_no_finite_difference_uncertainty": True,
        },
        "delta_projection_difference": {
            "maximum_absolute": float(np.abs(delta_projection).max()),
            "note": (
                "the slope difference the curvature prediction rests on; if no spike appears "
                "this is the quantity that explains why"
            ),
        },
        "vega_projection_difference": {
            "maximum_absolute": float(np.abs(vega_projection).max()),
        },
        "deployed": {
            greek: [float(value) for value in greeks["deployed"][greek]] for greek in GREEKS
        },
        "raw_network": {
            greek: [float(value) for value in greeks["raw_network"][greek]] for greek in GREEKS
        },
        "signals": {
            name: [float(value) for value in signals[name]]
            for name in ("raw_minus_floor", "projection_weight", "floor_leg_weight")
        },
        "transition_band": _transition_band(grid, signals["projection_weight"]),
    }


def _transition_band(grid: np.ndarray, weight: np.ndarray) -> dict[str, Any]:
    """Width of the band where the projection weight moves through its range."""
    inside = (weight > 0.01) & (weight < 0.99)
    if not bool(inside.any()):
        return {"points": 0, "width": None, "note": "no crossover band inside this scan"}
    values = grid[inside]
    return {
        "points": int(inside.sum()),
        "low": float(values.min()),
        "high": float(values.max()),
        "width": float(values.max() - values.min()),
    }


def vega_step(scan: Mapping[str, Any]) -> dict[str, Any]:
    """Size of the deployed-Vega step across a volatility crossover.

    Reported absolute and relative, with the transition width in volatility
    points: a band narrow enough for a Newton iterate to cross in one step means
    its Jacobian changes discontinuously from the solver's point of view.
    """
    vega = np.asarray(scan["deployed"]["vega"], dtype=np.float64)
    band = scan["transition_band"]
    low, high = float(vega[0]), float(vega[-1])
    step = high - low
    scale = max(abs(low), abs(high))
    return {
        "vega_before": low,
        "vega_after": high,
        "absolute_step": step,
        "relative_step": (step / scale) if scale > 0 else None,
        "transition_width_in_volatility_points": (
            band["width"] * 100.0 if band.get("width") is not None else None
        ),
        "note": (
            "a transition narrow enough for a Newton iterate to cross in one step presents a "
            "discontinuous Jacobian to the solver"
        ),
    }


# ---------------------------------------------------------------------------
# Section G: finite-bump battery versus pointwise autograd signs
# ---------------------------------------------------------------------------


def autograd_sign_violations(
    greeks: Mapping[str, np.ndarray], option_type: Sequence[str], scale: np.ndarray
) -> dict[str, Any]:
    """Pointwise sign violations, the autograd analogue of the shape battery.

    Deliberately **not** compared against the battery's counts as if they were
    the same measurement. The battery uses finite bumps of +/-1% in spot and
    +/-0.01 in volatility; these are pointwise derivatives. They are different
    functionals, and reporting them side by side is how section G determines
    whether the battery predicts pointwise quality at all.
    """
    calls = np.asarray([str(value) == "call" for value in option_type], dtype=bool)
    delta = np.asarray(greeks["delta"], dtype=np.float64)
    gamma = np.asarray(greeks["gamma"], dtype=np.float64)
    vega = np.asarray(greeks["vega"], dtype=np.float64)
    monotonic = np.where(calls, delta < 0.0, delta > 0.0)
    convex = gamma < 0.0
    volatility = vega < 0.0
    return {
        "rows": int(delta.size),
        "delta_sign_violations": int(monotonic.sum()),
        "gamma_sign_violations": int(convex.sum()),
        "vega_sign_violations": int(volatility.sum()),
        "delta_sign_violation_fraction": float(monotonic.mean()),
        "gamma_sign_violation_fraction": float(convex.mean()),
        "vega_sign_violation_fraction": float(volatility.mean()),
        "normalized_scale_available": bool(scale.size == delta.size),
        "functional": "pointwise autograd derivative",
    }


def sign_comparison(
    finite_bump: Mapping[str, Any], pointwise: Mapping[str, Any]
) -> dict[str, Any]:
    """The side-by-side section G reports, with the caveat attached."""
    return {
        "finite_bump_battery": {
            "material_shape_violations": finite_bump.get("material_shape_violations"),
            "families": finite_bump.get("shape_families"),
            "functional": "finite bumps of +/-1% in spot and +/-0.01 in volatility",
        },
        "pointwise_autograd": dict(pointwise),
        "caveat": (
            "these are different functionals and a model can pass one while failing the "
            "other; the 551 finite-bump shape violations are not a Greek-accuracy metric. "
            "The purpose of reporting them together is to determine whether the structural "
            "battery is predictive of pointwise Greek quality, not to reconcile the counts."
        ),
    }


def scope_declaration() -> dict[str, Any]:
    """What this study measures, and what it explicitly does not."""
    return {
        "schema_version": GREEK_FIDELITY_SCHEMA,
        "reference_operator": REFERENCE_OPERATOR,
        "which_comparison": WHICH_COMPARISON,
        "predictions": list(PREDICTIONS),
        "mandatory_split": (
            "raw-network and deployed Greeks are computed at identical states on every grid; "
            "raw good with deployed bad indicates a deployment-head change costing no neural "
            "attempt, while both bad indicates a learning intervention"
        ),
        "grid1_limitation": (
            "absolute surrogate-versus-label-operator fidelity on Grid 1 is genuinely limited "
            "by CRR finite-difference uncertainty; conclusions may be drawn only where "
            "disagreement exceeds it"
        ),
        "grid2_primary_statistic": GRID2_PRIMARY_STATISTIC,
        "crossover_sites": list(CROSSOVER_SITES),
        "conventions": {"delta": DELTA_CONVENTION, "gamma_and_vega": RELATIVE_CONVENTION},
        "degenerate_sensitivity_threshold": DEGENERATE_SENSITIVITY,
    }


def normalized_scale(physical: np.ndarray) -> np.ndarray:
    """``A = S * exp(-q * T)``, for reporting Greeks in normalized units."""
    with torch.no_grad():
        return discounted_spot(
            torch.as_tensor(np.ascontiguousarray(physical), dtype=torch.float64)
        ).numpy()


# ---------------------------------------------------------------------------
# The declared crossover contract set
# ---------------------------------------------------------------------------

#: Contracts scanned in Grid 2 and Grid 2b. Declared as a deterministic
#: construction spanning both option types, maturity, volatility and
#: representative moneyness, at a count inside the 20-50 the protocol declares.
CROSSOVER_CONTRACTS: Final = 32
CROSSOVER_SEED: Final = 20260824


def declared_contracts(count: int = CROSSOVER_CONTRACTS) -> list[dict[str, float]]:
    """The declared crossover contract set, as physical feature mappings."""
    generator = np.random.default_rng(CROSSOVER_SEED)
    rows = int(count)
    spot = generator.uniform(70.0, 130.0, rows)
    moneyness = generator.uniform(-0.45, 0.45, rows)
    return [
        {
            "option_type": 1.0 if index % 2 == 0 else -1.0,
            "spot": float(spot[index]),
            "strike": float(spot[index] * np.exp(-moneyness[index])),
            "maturity": float(generator.uniform(0.08, 2.5)),
            "rate": float(generator.uniform(0.0, 0.10)),
            "dividend_yield": float(generator.uniform(0.0, 0.08)),
            "volatility": float(generator.uniform(0.10, 0.65)),
        }
        for index in range(rows)
    ]


def scan_all(
    model: AmericanDevPriceModel,
    *,
    axis: str,
    contracts: Sequence[Mapping[str, float]] | None = None,
    european_margin: float | None = None,
) -> dict[str, Any]:
    """Locate and scan both curvature sites, on every declared contract.

    Contracts with no crossing inside the declared domain are reported as such
    rather than dropped, and contracts with several crossings report the count
    rather than silently taking the first.
    """
    if axis == "spot":
        low, high, half_width, spacing = 50.0, 150.0, SPOT_SCAN_HALF_WIDTH, SPOT_SCAN_SPACING
    elif axis == "volatility":
        low, high = 0.05, 0.8
        half_width, spacing = VOLATILITY_SCAN_HALF_WIDTH, VOLATILITY_SCAN_SPACING
    else:
        raise GreekFidelityError(f"unknown scan axis {axis!r}")
    declared = list(contracts) if contracts is not None else declared_contracts()
    results: dict[str, Any] = {"axis": axis, "contracts": len(declared), "sites": {}}
    for site in CROSSOVER_SITES:
        located: list[dict[str, Any]] = []
        scans: list[dict[str, Any]] = []
        for index, contract in enumerate(declared):
            location = locate_crossings(
                model,
                contract,
                site,
                axis=axis,
                low=low,
                high=high,
                european_margin=european_margin,
            )
            location["contract_index"] = index
            located.append(location)
            if location["crossing_exists"]:
                scan = scan_crossover(
                    model,
                    contract,
                    location["crossings"][0],
                    axis=axis,
                    half_width=half_width,
                    spacing=spacing,
                    european_margin=european_margin,
                )
                scan["contract_index"] = index
                scans.append(scan)
        maxima = [
            entry["gamma_projection_difference"]["maximum_absolute"] for entry in scans
        ]
        results["sites"][site] = {
            "existence": {
                "contracts_with_a_crossing": sum(
                    1 for entry in located if entry["crossing_exists"]
                ),
                "contracts_without_a_crossing": sorted(
                    entry["contract_index"] for entry in located if not entry["crossing_exists"]
                ),
                "contracts_with_multiple_crossings": sorted(
                    entry["contract_index"] for entry in located if entry["multiple_crossings"]
                ),
            },
            "maximum_gamma_projection_difference": max(maxima) if maxima else None,
            "maximum_delta_projection_difference": (
                max(
                    entry["delta_projection_difference"]["maximum_absolute"] for entry in scans
                )
                if scans
                else None
            ),
            "scans": scans,
        }
        if axis == "volatility" and scans:
            results["sites"][site]["vega_steps"] = [vega_step(entry) for entry in scans]
    results["primary_statistic"] = GRID2_PRIMARY_STATISTIC
    return results


def compact_scan(report: Mapping[str, Any]) -> dict[str, Any]:
    """The small, agent-readable view of a crossover study."""
    return {
        "axis": report["axis"],
        "contracts": report["contracts"],
        "primary_statistic": report["primary_statistic"],
        "sites": {
            site: {
                "contracts_with_a_crossing": entry["existence"]["contracts_with_a_crossing"],
                "contracts_without_a_crossing": len(
                    entry["existence"]["contracts_without_a_crossing"]
                ),
                "contracts_with_multiple_crossings": len(
                    entry["existence"]["contracts_with_multiple_crossings"]
                ),
                "maximum_gamma_projection_difference": entry[
                    "maximum_gamma_projection_difference"
                ],
                "maximum_delta_projection_difference": entry[
                    "maximum_delta_projection_difference"
                ],
            }
            for site, entry in report["sites"].items()
        },
    }


# ---------------------------------------------------------------------------
# Grid 1, and the human-invoked driver
# ---------------------------------------------------------------------------

DEFAULT_GRID1_TEMPLATE: Final = "artifacts/task-9h/greeks/grid1-fidelity-{row_set}-v1.json"


def grid_one(
    model: AmericanDevPriceModel,
    columns: Mapping[str, np.ndarray],
    reference_contract: Mapping[str, Any],
    acceptance: Mapping[str, Any],
    *,
    batch_size: int = 4096,
    thread_count: int = 8,
    european_margin: float | None = None,
) -> dict[str, Any]:
    """Greek fidelity on one row set, raw and deployed, against ``L_1024``.

    The bumps and their uncertainties come from the **reference contract
    artifact**, not from a fresh choice here: selecting a bump alongside the
    comparison it feeds would be choosing the reference's noise level with the
    answer in view.
    """
    from ..american_pilot import physical_features, shape_diagnostics
    from . import greek_reference as reference
    from .price_fidelity import diagnostic_breakdown, projection_partition

    physical = physical_features(columns)
    arrays = reference._state_arrays(columns)
    selection = reference_contract["bump_selection"]
    greeks = both_greeks(model, physical, european_margin)

    references: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    for greek in GREEKS:
        bump = float(selection[greek]["selected_bump"])
        if greek == "delta":
            references[greek] = reference.reference_delta(
                arrays, bump, depth=reference.REFERENCE_DEPTH, thread_count=thread_count
            )
            masks[greek] = reference.spot_eligible(arrays, bump)
        elif greek == "gamma":
            references[greek] = reference.reference_gamma(
                arrays, bump, depth=reference.REFERENCE_DEPTH, thread_count=thread_count
            )
            masks[greek] = reference.spot_eligible(arrays, bump)
        else:
            references[greek] = reference.reference_vega(
                arrays, bump, depth=reference.REFERENCE_DEPTH, thread_count=thread_count
            )
            masks[greek] = reference.volatility_eligible(arrays, bump)

    slices = dict(_slice_masks_for(columns, acceptance))
    partition = projection_partition(model, columns, european_margin)
    partition.pop("_projection_weight", None)
    partition.pop("_floor_leg_weight", None)
    slices.update(partition)

    per_prediction: dict[str, Any] = {}
    for name in PREDICTIONS:
        computed = greeks[name]
        entry: dict[str, Any] = {"global": {}, "slices": {}, "degeneracy": {}}
        for greek in GREEKS:
            uncertainty = float(selection[greek]["reference_uncertainty"])
            entry["global"][greek] = compare_greek(
                greek, computed[greek], references[greek], uncertainty, masks[greek]
            )
            entry["slices"][greek] = {
                slice_name: compare_greek(
                    greek, computed[greek], references[greek], uncertainty, mask & masks[greek]
                )
                for slice_name, mask in slices.items()
                if bool((mask & masks[greek]).any())
            }
        for greek in ("vega", "gamma"):
            entry["degeneracy"][greek] = degeneracy(computed[greek], references[greek])
        entry["pointwise_signs"] = autograd_sign_violations(
            computed, columns["option_type"], normalized_scale(physical)
        )
        per_prediction[name] = entry

    battery, _ = shape_diagnostics(
        MarginOverrideModel(
            model, model.european_margin if european_margin is None else european_margin
        ),
        columns,
        acceptance,
        batch_size=batch_size,
    )
    return {
        **scope_declaration(),
        "rows": int(physical.shape[0]),
        "reference_bumps": {
            greek: {
                "selected_bump": float(selection[greek]["selected_bump"]),
                "reference_uncertainty": float(selection[greek]["reference_uncertainty"]),
                "plateau_reached": selection[greek]["plateau_reached"],
                "eligible_rows": int(masks[greek].sum()),
            }
            for greek in GREEKS
        },
        "predictions": per_prediction,
        "section_g": sign_comparison(
            diagnostic_breakdown(battery), per_prediction["deployed"]["pointwise_signs"]
        ),
    }


def _slice_masks_for(
    columns: Mapping[str, np.ndarray], acceptance: Mapping[str, Any]
) -> Mapping[str, np.ndarray]:
    """The acceptance file's own slice masks, imported rather than restated."""
    from ..american_pilot import _slice_masks

    return _slice_masks(columns, acceptance)


GRIDS: Final = ("1", "2", "2b")
DEFAULT_OUTPUT_TEMPLATE: Final = "artifacts/task-9h/greeks/grid{grid}{suffix}-v1.json"

#: The frozen checkpoint the Greek study measures.
STUDIED_ATTEMPT: Final = "scratch_residual_smooth_floor_margin_v1"


def analyze(
    project_root: Any,
    grid: str,
    *,
    row_set: str | None = None,
    output: str | None = None,
    overwrite: bool = False,
    thread_count: int = 8,
    batch_size: int = 4096,
) -> dict[str, Any]:
    """One grid of the Greek study, published beneath the ignored artifacts tree.

    Grid 1 needs a row set and the reference contract. Grids 2 and 2b are
    label-free crossover scans over the declared contract set and need neither.
    """
    from pathlib import Path

    from ..artifact import write_json_atomic
    from .attempts import ACCEPTANCE_CONFIG_PATH, load_toml
    from .frozen import _guarded_artifact_path, diagnostic_provenance, load_frozen_model
    from .greek_reference import DEFAULT_OUTPUT as REFERENCE_OUTPUT

    if grid not in GRIDS:
        raise GreekFidelityError(f"unknown grid {grid!r}; the grids are {list(GRIDS)}")
    project_root = Path(project_root)
    suffix = f"-{row_set}" if grid == "1" else ""
    relative = output or DEFAULT_OUTPUT_TEMPLATE.format(grid=grid, suffix=suffix)
    path = _guarded_artifact_path(project_root, relative, where="greek fidelity path")
    if path.exists() and not overwrite:
        raise GreekFidelityError(f"'{relative}' already exists; pass overwrite to replace it")

    model, identity = load_frozen_model(project_root, STUDIED_ATTEMPT)
    if grid == "1":
        if not row_set:
            raise GreekFidelityError("grid 1 needs a row set")
        from .eligibility import load_row_set

        reference_path = project_root / REFERENCE_OUTPUT
        if not reference_path.is_file():
            raise GreekFidelityError(
                f"'{REFERENCE_OUTPUT}' does not exist; the Greek reference contract defines "
                "the bumps and their uncertainties and must be produced first"
            )
        contract = json.loads(reference_path.read_text(encoding="utf-8"))
        columns = load_row_set(project_root, row_set)
        acceptance = load_toml(project_root / ACCEPTANCE_CONFIG_PATH)
        body = {
            "grid": "1",
            "row_set": row_set,
            "reference_contract_sha256": _digest(reference_path),
            **grid_one(
                model,
                columns,
                contract,
                acceptance,
                batch_size=batch_size,
                thread_count=thread_count,
            ),
        }
    else:
        axis = "spot" if grid == "2" else "volatility"
        body = {
            "grid": grid,
            "axis": axis,
            **scope_declaration(),
            "scan": scan_all(model, axis=axis),
        }
    report = {**body, "checkpoint": identity, "provenance": diagnostic_provenance(project_root)}
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, report, overwrite=overwrite)
    return report


def _digest(path: Any) -> str:
    from .attempts import sha256_file

    return sha256_file(path)


def compact_grid(report: Mapping[str, Any]) -> dict[str, Any]:
    """The small, agent-readable view of one grid."""
    common = {
        "grid": report["grid"],
        "protocol_commit": report.get("provenance", {}).get("protocol_commit"),
        "checkpoint": report["checkpoint"]["attempt_id"],
    }
    if report["grid"] != "1":
        return {**common, **compact_scan(report["scan"])}
    summary: dict[str, Any] = {**common, "row_set": report["row_set"], "rows": report["rows"]}
    for name in PREDICTIONS:
        entry = report["predictions"][name]
        summary[name] = {
            greek: {
                "mae": entry["global"][greek]["absolute"].get("mae"),
                "reference_uncertainty": entry["global"][greek]["reference_uncertainty"],
                "fraction_exceeding_uncertainty": entry["global"][greek][
                    "fraction_exceeding_reference_uncertainty"
                ],
                "sign_disagreement_rate": entry["global"][greek]["sign_disagreement_rate"],
            }
            for greek in GREEKS
        }
        summary[name]["vega_degeneracy_excess"] = entry["degeneracy"]["vega"]["excess_fraction"]
        summary[name]["gamma_degeneracy_excess"] = entry["degeneracy"]["gamma"][
            "excess_fraction"
        ]
    summary["section_g"] = {
        "finite_bump_shape_violations": report["section_g"]["finite_bump_battery"][
            "material_shape_violations"
        ],
        "pointwise_delta_sign_violations": report["section_g"]["pointwise_autograd"][
            "delta_sign_violations"
        ],
        "pointwise_gamma_sign_violations": report["section_g"]["pointwise_autograd"][
            "gamma_sign_violations"
        ],
        "pointwise_vega_sign_violations": report["section_g"]["pointwise_autograd"][
            "vega_sign_violations"
        ],
    }
    return summary
