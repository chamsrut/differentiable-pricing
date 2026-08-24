"""Task 9H practical pricing tolerance: **predeclared**, before any slice existed.

This module is the declaration itself. It was written and committed before the
price-fidelity analysis ran and before H1 was evaluated, and it is not revised
because a result came out badly. It is **additional** to the Task 9H development
criterion, which stays exactly as
``configs/american_neural_pilot_acceptance_v1.toml`` fixes it and is neither
replaced nor loosened here.

The intended pricing claim
--------------------------
The project's downstream objective is implied-volatility inversion and
volatility-surface construction at materially lower latency. A price tolerance
for that claim therefore has to be stated in the units the claim is about, which
are **volatility units, not price units**.

Primary tolerance
-----------------
``|V_hat - V| / vega_BS <= 0.01`` -- one volatility point. ``vega_BS`` is the
analytic Black--Scholes vega at the row's own state, obtained by differentiating
:func:`.representation.european_price` rather than by reimplementing it, so the
denominator cannot drift from the ``d1`` construction the deployed floor's
European leg already uses. It is label-free and model-independent, so the raw
network and the deployed output are scored against the same denominator and are
directly comparable.

Reported as a per-row pass rate, globally and by maturity bucket, separately for
the raw network and the deployed output. The bar is **99% of eligible rows**.

Eligibility: per-row resolvability
----------------------------------
``0.01 * vega_BS(x) > k * g(x)`` with ``g(x) = |E_CRR(x) - E_BS(x)|`` and
``k = 3``. A row is resolvable at one volatility point only if a one-point move
in volatility changes the price by more than the label's own discretization
error. Evaluated **per row**, because vega and ``g`` shrink together and a
global-maximum floor is far over-strict at a typical row -- measured at 134x on
the domain sample, whose maximum is 4.19e-5 against a median of 3.13e-7.

``k = 3`` because ``g`` is the **European-leg proxy** for the American label's
discretization error, which is plausibly larger because of free-boundary
resolution. The depth-convergence check extended to prices gives a direct
American estimate that validates this choice **retrospectively and does not
adjust it**.

Rows below the floor are recorded IV-out-of-scope, excluded from the pass rate,
and reported with their absolute price errors. **The excluded fraction and its
composition are a headline beside every pass rate**, so "99% of eligible rows"
is never readable as "99% of rows".

Secondary screen
----------------
``|u_hat - u| <= 5.5e-4`` in normalized units, flat. This is the
shortest-maturity at-the-money transmission of the same 0.01 volatility
standard: the minimum at-the-money normalized vega over the declared domain is
``phi(d1) * sqrt(T) ~ 0.0552`` at ``T = 0.019178``, so ``5.5e-4 / 0.0552 =
0.0099 <= 0.01``. It is therefore **sufficient** for the volatility standard at
the money at every maturity, and **not** sufficient away from the money where
vega is smaller -- which is why it is a screen and not the primary.

Both pass rates are reported. The gap between them localizes the errors:
primary-pass-but-screen-fail rows are large price errors in high-vega states,
cheap in volatility terms; screen-pass-but-primary-fail rows are small price
errors in low-vega states, expensive in volatility terms and exactly the ones
the downstream objective depends on.

Assessability of the primary statistic
--------------------------------------
The eligibility rule anchors on the **label's** discretization error, not on the
**model's** error. Where the lattice is well converged ``g`` is tiny, so rows
whose vega is very small are admitted as formally resolvable: the reference is
sharp enough to answer the question even though answering it demands extreme
price accuracy.

A row is therefore called **vega-degenerate** when

    0.01 * vega_BS(x) < 5.5e-4 * A(x)

-- equivalently ``normalized vega_BS < 0.055`` -- because meeting the primary
tolerance there would demand price accuracy *finer than this project's own
declared secondary price screen*. The threshold is **derived** from the two
already-declared tolerances rather than asserted as a third number:
``SECONDARY_NORMALIZED_TOLERANCE / VOLATILITY_TOLERANCE``.

**Predeclared interpretation.** If more than 1% of eligible rows are
vega-degenerate, the ``>= 99%`` primary pass rate is reported as **NOT
ASSESSABLE**. The conditional pass rate on the non-degenerate eligible subset is
then also reported, and it **does not silently substitute** for the original
declaration.

These eligible-set vega diagnostics are **model-independent** and are computed
**before** any model output is scored. That ordering is enforced structurally by
a separate human-invoked eligibility step, not by intention.

A **secondary eligibility view** is reported alongside, and is descriptive only:

    0.01 * vega_BS > 3 * g   AND   normalized vega_BS > 1e-4

It never replaces the declared eligibility rule.

**Disclosure.** This secondary diagnostic was added after a synthetic smoke
exercise revealed that the mathematical statistic can have extreme tails when
vega approaches zero. The smoke measurements used a synthetic model on synthetic
states to exercise the statistic; they revealed a property of the statistic and
are **not** an observation of any E2c result.

Calibration objective
---------------------
``|bias| / RMSE <= max(0.25, z / sqrt(n))`` within every gated slice, on the
**deployed** output, in **normalized** units. Normalized because the network's
five inputs exclude spot, so normalized error is spot-independent by
construction while ``A`` is spot-dominated -- a physical-unit gate would weight
by a variable the model cannot see.

``z`` comes from a two-sided Bonferroni correction at 5% family-wise error over
the frozen slice count, so no slice fails on sampling noise and the bar
documents each slice's power. The noise term binds below ``n ~ 144`` and the
substantive 0.25 above it.

**This is a declared proxy for the calibration objective, not a
calibration-readiness test.** No calibration machinery exists in this project
and the price-space objective's weighting scheme is undeclared, so the criterion
cannot be validated against the thing it protects.

Deliberately not used
---------------------
No quoting-increment anchor: this project holds no market data, and a tolerance
set as a fraction of the underlying would scale an option's tick by the
underlying's level, which is not how quoting increments work. No absolute price
floor either: ``A = S * exp(-q * T)`` is bounded below by ``50 * exp(-0.36) =
34.9`` over the declared domain, so any absolute floor below about 0.0175 would
never bind and would only look like a safeguard.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from statistics import NormalDist
from typing import Any, Final

import numpy as np
import torch

from ..config import FEATURE_ORDER
from .representation import discounted_spot, european_price

TOLERANCE_SCHEMA: Final = "american-dev-price-tolerance/1"

#: Declared before the price-fidelity analysis was run, and before H1 was
#: evaluated. A constant, not an argument: a tolerance that can be passed in is
#: a tolerance that can be chosen after seeing the answer.
DECLARED_BEFORE_ANALYSIS: Final = True

#: One volatility point. The primary tolerance, in the units the downstream
#: claim is about.
VOLATILITY_TOLERANCE: Final = 0.01

#: Safety factor on the European-leg proxy for the American label's own
#: discretization error. Predeclared; the depth check validates it
#: retrospectively and never adjusts it.
RESOLVABILITY_SAFETY_FACTOR: Final = 3.0

#: The flat normalized secondary screen.
SECONDARY_NORMALIZED_TOLERANCE: Final = 5.5e-4

#: Fraction of **eligible** rows that must meet the primary tolerance.
PASS_RATE_BAR: Final = 0.99

#: Normalized vega below which a row is **vega-degenerate**: meeting the primary
#: tolerance there would demand price accuracy finer than this project's own
#: declared secondary price screen. Derived from the two tolerances already
#: declared above rather than asserted as a third independent number, so it
#: cannot drift away from them.
#:
#: ``0.01 * vega_BS < 5.5e-4 * A``  <=>  ``vega_BS / A < 5.5e-4 / 0.01 = 0.055``.
VEGA_DEGENERACY_NORMALIZED: Final = SECONDARY_NORMALIZED_TOLERANCE / VOLATILITY_TOLERANCE

#: If more than this fraction of eligible rows is vega-degenerate, the primary
#: pass-rate statistic is reported NOT ASSESSABLE rather than reported as a
#: number that a degenerate denominator has already decided.
ASSESSABILITY_DEGENERATE_LIMIT: Final = 0.01

#: The reporting threshold for the **descriptive** secondary eligibility view.
#: Not a criterion, and never a substitute for the declared eligibility rule.
SECONDARY_VIEW_NORMALIZED_VEGA_MINIMUM: Final = 1.0e-4

#: Where the secondary view came from, worded so it cannot be misread as a
#: result. Recorded in every artifact that reports the view.
SECONDARY_VIEW_DISCLOSURE: Final = (
    "This secondary diagnostic was added after a synthetic smoke exercise revealed that "
    "the mathematical statistic can have extreme tails when vega approaches zero. The "
    "smoke measurements used a synthetic model on synthetic states to exercise the "
    "statistic; they revealed a property of the statistic, not the performance of any "
    "trained checkpoint, and no E2c result was observed."
)

NOT_ASSESSABLE: Final = "NOT ASSESSABLE"

#: Substantive calibration bound, and the family-wise error rate the noise-aware
#: term is derived at.
CALIBRATION_RATIO: Final = 0.25
CALIBRATION_FAMILY_WISE_ALPHA: Final = 0.05

#: The measured domain statistics of ``(E_CRR - E_BS) / A``, from
#: ``artifacts/task-9h/domain/european-comparator-domain-v1.json`` over 275,496
#: states. Recorded here so the two comparison floors below are traceable to a
#: measurement rather than to a remembered number.
DOMAIN_GAP_MAXIMUM_POSITIVE: Final = 4.192769575172157e-05
DOMAIN_GAP_MINIMUM_SIGNED: Final = -2.806600831049622e-04
DOMAIN_GAP_MEDIAN: Final = 3.1267736913126274e-07

#: The frozen, **gated** slice family. Enumerated explicitly so ``m`` is well
#: defined at declaration time, and reconciled against the acceptance file's own
#: bins by :func:`assert_slice_family_matches` so it cannot silently drift.
#:
#: The ``s`` partition and its leg-dominance sub-split are deliberately absent:
#: they are endogenous to the head, and floor-dominated rows carry a
#: structurally determined bias sign because the projection only ever moves a
#: prediction toward the floor. A failure there would test the head's design,
#: not calibration-readiness, so those slices are reported and never gated.
GATED_SLICE_FAMILY: Final = (
    "overall",
    "option_type:call",
    "option_type:put",
    "expiry:-inf:0.25",
    "expiry:0.25:1",
    "expiry:1:2",
    "expiry:2:inf",
    "moneyness:-inf:-0.2",
    "moneyness:-0.2:0.2",
    "moneyness:0.2:inf",
    "volatility:-inf:0.15",
    "volatility:0.15:0.5",
    "volatility:0.5:inf",
    "premium_status:zero",
    "premium_status:positive",
    "exercise_status:no_exercise",
    "exercise_status:exercise_observed",
    "iv_scope:in_scope",
    "iv_scope:out_of_scope",
)

#: Slices reported with the calibration ratio but never gated on it.
UNGATED_SLICE_FAMILIES: Final = ("projection:", "floor_leg:")

#: The statistic the calibration objective actually cares about: bias that
#: varies across the surface distorts skew and term structure, while uniform
#: bias mostly shifts the volatility level.
BIAS_SPREAD_FAMILIES: Final = ("moneyness", "expiry")

CALIBRATION_IS_A_PROXY: Final = (
    "a declared proxy for the calibration objective, not a calibration-readiness test: no "
    "calibration machinery exists in this project and the price-space objective's weighting "
    "scheme is undeclared, so this criterion cannot be validated against what it protects"
)


class ToleranceError(RuntimeError):
    """Raised when the declared tolerance cannot be applied as declared."""


_VOLATILITY_INDEX: Final = FEATURE_ORDER.index("volatility")


# ---------------------------------------------------------------------------
# The primary tolerance
# ---------------------------------------------------------------------------


def black_scholes_vega(physical_features: torch.Tensor) -> torch.Tensor:
    """``dE_BS/dsigma`` at each row, by differentiating the deployed European leg.

    Differentiated rather than reimplemented on purpose: the floor's European
    leg and this denominator must be two views of one function, or a row could
    be judged against a vega that belongs to a slightly different Black--Scholes
    parameterization than the one the model is actually floored by.

    Returned detached. Vega here is a fixed scale for an error, not a quantity
    anything differentiates through.
    """
    inputs = physical_features.detach().clone().requires_grad_(True)
    (gradient,) = torch.autograd.grad(european_price(inputs).sum(), inputs)
    vega = gradient[:, _VOLATILITY_INDEX].detach()
    if not bool(torch.isfinite(vega).all()):
        raise ToleranceError("the analytic Black-Scholes vega is outside its domain")
    return vega


def volatility_equivalent_error(
    prediction: np.ndarray, reference: np.ndarray, vega: np.ndarray
) -> np.ndarray:
    """``|V_hat - V| / vega_BS``: the price error expressed in volatility units.

    **The quotient may be infinite.** Black--Scholes vega underflows toward
    denormal float64 values -- of order 1e-313 -- for deep in- or out-of-the-money
    short-dated contracts, and the declared eligibility rule admits such a row
    whenever the label's own discretization gap is smaller still. Dividing a
    normal-sized price error by a denormal overflows.

    That is left as ``inf`` rather than clamped. ``inf <= 0.01`` is ``False``, so
    the **pass rate stays exact** and such a row correctly counts as a failure;
    clamping would fabricate a finite number for a quantity that genuinely has
    none. Callers summarizing the distribution must handle non-finite entries
    explicitly rather than averaging them.
    """
    vega = np.asarray(vega, dtype=np.float64)
    if bool((vega <= 0.0).any()):
        raise ToleranceError(
            "a non-positive vega cannot scale a price error; screen with iv_in_scope first"
        )
    difference = np.asarray(prediction, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    with np.errstate(over="ignore", divide="ignore"):
        return np.abs(difference) / vega


def label_discretization_proxy(
    physical_features: torch.Tensor, european_crr_price: np.ndarray
) -> np.ndarray:
    """``g(x) = |E_CRR(x) - E_BS(x)|``, in **price** units.

    The stored CRR European column against the analytic value. This is the only
    quantity in the tolerance that reads a dataset column, so the eligibility
    map is model-free but **not** label-free: it needs a European lattice per
    state and cannot be evaluated at an arbitrary domain point for free.
    """
    with torch.no_grad():
        analytic = european_price(physical_features).numpy()
    return np.abs(np.asarray(european_crr_price, dtype=np.float64) - analytic)


def iv_in_scope(
    physical_features: torch.Tensor,
    european_crr_price: np.ndarray,
    *,
    safety_factor: float = RESOLVABILITY_SAFETY_FACTOR,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Per-row resolvability: ``0.01 * vega_BS > k * g``.

    Returns the boolean eligibility mask and the terms it was built from, so a
    report can show why a row was excluded rather than only that it was.
    """
    vega = black_scholes_vega(physical_features).numpy()
    gap = label_discretization_proxy(physical_features, european_crr_price)
    mask = (VOLATILITY_TOLERANCE * vega) > (float(safety_factor) * gap)
    return mask, {
        "rule": "0.01 * vega_BS(x) > k * |E_CRR(x) - E_BS(x)|",
        "safety_factor": float(safety_factor),
        "volatility_tolerance": VOLATILITY_TOLERANCE,
        "vega": vega,
        "gap": gap,
        "model_free": True,
        "label_free": False,
        "reads": "european_crr_price",
    }


def comparison_floors(physical_features: torch.Tensor) -> dict[str, np.ndarray]:
    """The two global-maximum floors, for **reporting comparison only**.

    Neither selects the eligibility rule. They exist because "how much less
    strict is a per-row floor than a global one?" is only answerable against a
    stated global one, and there are two defensible statements of it:

    * ``one_sided`` uses the positive supremum 4.19e-5, the quantity the
      ``delta`` derivation used, because only ``E_CRR > E_BS`` threatens the
      floor;
    * ``two_sided`` uses ``|.|``'s maximum 2.81e-4, which is what the
      eligibility rule's own ``g = |E_CRR - E_BS|`` implies and is 6.7x larger.

    Reporting one without the other would silently make the comparison look
    whichever way was chosen.
    """
    with torch.no_grad():
        scale = discounted_spot(physical_features).numpy()
    return {
        "one_sided": (DOMAIN_GAP_MAXIMUM_POSITIVE / VOLATILITY_TOLERANCE) * scale,
        "two_sided": (abs(DOMAIN_GAP_MINIMUM_SIGNED) / VOLATILITY_TOLERANCE) * scale,
    }


def normalized_vega(physical_features: torch.Tensor) -> np.ndarray:
    """``vega_BS / A``: vega in the units the tolerances are stated in."""
    with torch.no_grad():
        scale = discounted_spot(physical_features).numpy()
    return black_scholes_vega(physical_features).numpy() / scale


def eligible_vega_diagnostics(
    physical_features: torch.Tensor, european_crr_price: np.ndarray
) -> dict[str, Any]:
    """Model-independent vega structure of the eligible set.

    **Computed before any model output is scored.** Nothing here reads a
    prediction, so whether the primary statistic is assessable is settled by the
    contracts and the label operator alone -- not by how well a checkpoint
    happened to do.

    A row is *vega-degenerate* when a one-volatility-point move changes its price
    by less than the declared secondary price screen. Meeting the primary
    tolerance there would demand accuracy finer than this project's own declared
    price resolution, so such a row is arithmetically certain to fail whenever
    the model merely meets the declared screen.
    """
    eligible, terms = iv_in_scope(physical_features, european_crr_price)
    scaled = normalized_vega(physical_features)
    rows = int(scaled.size)
    degenerate = scaled < VEGA_DEGENERACY_NORMALIZED
    tiny = scaled < SECONDARY_VIEW_NORMALIZED_VEGA_MINIMUM
    within = eligible & ~degenerate
    eligible_rows = int(eligible.sum())
    quantiles = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)
    return {
        "rows": rows,
        "eligible_rows": eligible_rows,
        "eligible_fraction": float(eligible.mean()),
        "eligibility_rule": terms["rule"],
        "model_independent": True,
        "computed_before_any_model_output_was_scored": True,
        "normalized_vega_quantiles_among_eligible": (
            {
                f"{quantile:g}": float(np.quantile(scaled[eligible], quantile))
                for quantile in quantiles
            }
            if eligible_rows
            else None
        ),
        "degeneracy": {
            "threshold_normalized_vega": VEGA_DEGENERACY_NORMALIZED,
            "threshold_derivation": (
                "SECONDARY_NORMALIZED_TOLERANCE / VOLATILITY_TOLERANCE: a one-volatility-"
                "point move smaller than the declared secondary price screen"
            ),
            "degenerate_eligible_rows": int((eligible & degenerate).sum()),
            "degenerate_eligible_fraction": (
                float((eligible & degenerate).sum() / eligible_rows) if eligible_rows else None
            ),
            "eligible_rows_below_1e-4_normalized_vega": int((eligible & tiny).sum()),
            "eligible_fraction_below_1e-4_normalized_vega": (
                float((eligible & tiny).sum() / eligible_rows) if eligible_rows else None
            ),
        },
        "secondary_view": {
            "rule": "0.01 * vega_BS > 3 * |E_CRR - E_BS| AND normalized vega_BS > 1e-4",
            "rows": int((eligible & ~tiny).sum()),
            "fraction_of_all_rows": float((eligible & ~tiny).mean()),
            "descriptive_only": True,
            "never_replaces_the_declared_rule": True,
            "disclosure": SECONDARY_VIEW_DISCLOSURE,
        },
        "non_degenerate_eligible_rows": int(within.sum()),
        "_eligible": eligible,
        "_degenerate": degenerate,
        "_normalized_vega": scaled,
    }


def assessability(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    """Whether the ``>= 99%`` primary statistic may be reported as a number.

    The rule is predeclared and mechanical: above
    :data:`ASSESSABILITY_DEGENERATE_LIMIT` of the eligible set being
    vega-degenerate, the primary pass rate is reported ``NOT ASSESSABLE``.

    The conditional pass rate on the non-degenerate eligible subset is still
    reported in that case, but it is a **different statistic** and the report
    must not present it as the declared one.
    """
    fraction = diagnostics["degeneracy"]["degenerate_eligible_fraction"]
    assessable = fraction is not None and fraction <= ASSESSABILITY_DEGENERATE_LIMIT
    return {
        "primary_statistic_assessable": bool(assessable),
        "verdict": "assessable" if assessable else NOT_ASSESSABLE,
        "degenerate_eligible_fraction": fraction,
        "limit": ASSESSABILITY_DEGENERATE_LIMIT,
        "rule": (
            "if more than 1% of eligible rows are vega-degenerate -- a one-volatility-point "
            "move smaller than the declared 5.5e-4 normalized secondary price screen -- the "
            ">= 99% primary pass rate is reported NOT ASSESSABLE"
        ),
        "if_not_assessable": (
            "the conditional pass rate on the non-degenerate eligible subset is also "
            "reported, and does not substitute for the original declaration"
        ),
        "predeclared": True,
    }


def meets_primary(
    prediction: np.ndarray, reference: np.ndarray, vega: np.ndarray
) -> np.ndarray:
    """Per-row pass under the primary tolerance."""
    return volatility_equivalent_error(prediction, reference, vega) <= VOLATILITY_TOLERANCE


def meets_secondary(normalized_error: np.ndarray) -> np.ndarray:
    """Per-row pass under the flat normalized secondary screen."""
    return np.abs(np.asarray(normalized_error, dtype=np.float64)) <= (
        SECONDARY_NORMALIZED_TOLERANCE
    )


# ---------------------------------------------------------------------------
# The calibration objective
# ---------------------------------------------------------------------------


def bonferroni_z(slices: int, alpha: float = CALIBRATION_FAMILY_WISE_ALPHA) -> float:
    """Two-sided Bonferroni critical value over a frozen slice family."""
    if slices < 1 or not 0.0 < alpha < 1.0:
        raise ToleranceError("a Bonferroni correction needs at least one slice and alpha in (0,1)")
    return float(NormalDist().inv_cdf(1.0 - alpha / (2.0 * slices)))


def calibration_bar(rows: int, slices: int = len(GATED_SLICE_FAMILY)) -> float:
    """``max(0.25, z / sqrt(n))``: substantive above ~144 rows, noise-aware below."""
    if rows < 1:
        raise ToleranceError("a calibration bar needs at least one row")
    return max(CALIBRATION_RATIO, bonferroni_z(slices) / float(np.sqrt(rows)))


def calibration_statistics(
    normalized_error: np.ndarray, slices: int = len(GATED_SLICE_FAMILY)
) -> dict[str, Any]:
    """Signed bias, RMSE, their ratio, the bar, and the slice's own noise floor.

    The absolute bias and the noise floor are reported beside the ratio because
    the ratio form tightens the bound exactly where the model is most accurate:
    a slice can pass while carrying more absolute systematic error than one it
    fails.
    """
    errors = np.asarray(normalized_error, dtype=np.float64)
    rows = int(errors.size)
    if rows == 0:
        raise ToleranceError("a calibration statistic needs at least one row")
    bias = float(errors.mean())
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    bar = calibration_bar(rows, slices)
    ratio = float(abs(bias) / rmse) if rmse > 0.0 else 0.0
    return {
        "rows": rows,
        "signed_bias": bias,
        "absolute_bias": abs(bias),
        "rmse": rmse,
        "bias_to_rmse": ratio,
        "bar": bar,
        "noise_floor": float(1.0 / np.sqrt(rows)),
        "bar_is_noise_limited": bar > CALIBRATION_RATIO,
        "passes": ratio <= bar,
    }


def bias_spread(slice_statistics: Mapping[str, Mapping[str, Any]], family: str) -> dict[str, Any]:
    """``max - min`` of signed bias across one slice family.

    The primary calibration-relevant statistic: bias that varies across the
    surface distorts skew and term structure, while uniform bias mostly shifts
    the volatility level.
    """
    members = {
        name: entry
        for name, entry in slice_statistics.items()
        if name.startswith(f"{family}:") and entry.get("rows")
    }
    if not members:
        return {"family": family, "members": 0, "spread": None}
    values = {name: float(entry["signed_bias"]) for name, entry in members.items()}
    high = max(values, key=lambda name: values[name])
    low = min(values, key=lambda name: values[name])
    return {
        "family": family,
        "members": len(values),
        "spread": values[high] - values[low],
        "maximum": {"slice": high, "signed_bias": values[high]},
        "minimum": {"slice": low, "signed_bias": values[low]},
        "interpretation": (
            "bias varying across this family distorts the surface's shape; uniform bias "
            "mostly shifts its level"
        ),
    }


# ---------------------------------------------------------------------------
# Reconciliation and the declaration record
# ---------------------------------------------------------------------------


def assert_slice_family_matches(available: Sequence[str]) -> None:
    """Require the frozen gated family to be exactly the slices actually produced.

    ``m`` has to be fixed at declaration time for the Bonferroni correction to
    mean anything, and it has to describe the slices that really exist or the
    correction is applied to a family nobody computed.
    """
    missing = sorted(set(GATED_SLICE_FAMILY) - set(available))
    if missing:
        raise ToleranceError(
            f"the frozen gated slice family names slice(s) the analysis did not produce: "
            f"{missing}"
        )


def declaration() -> dict[str, Any]:
    """The tolerance, as a record every downstream report embeds verbatim."""
    return {
        "schema_version": TOLERANCE_SCHEMA,
        "declared_before_analysis": DECLARED_BEFORE_ANALYSIS,
        "declared_before_h1_was_evaluated": True,
        "primary": {
            "statement": "|V_hat - V| / vega_BS <= 0.01",
            "units": "volatility points",
            "kind": "relative to analytic Black-Scholes vega",
            "vega_source": (
                "autograd through representation.european_price, so the denominator cannot "
                "drift from the d1 construction the deployed floor's European leg uses"
            ),
            "pass_rate_bar": PASS_RATE_BAR,
            "reported": "globally and by maturity bucket, raw network and deployed output",
        },
        "eligibility": {
            "statement": "0.01 * vega_BS(x) > 3 * |E_CRR(x) - E_BS(x)|",
            "safety_factor": RESOLVABILITY_SAFETY_FACTOR,
            "rationale": (
                "a row is resolvable at one volatility point only if a one-point move "
                "changes the price by more than the label's own discretization error; "
                "evaluated per row because vega and the gap shrink together"
            ),
            "safety_factor_rationale": (
                "g is the European-leg proxy for the American label's discretization error, "
                "plausibly larger because of free-boundary resolution; the depth check "
                "validates this retrospectively and does not adjust it"
            ),
            "excluded_rows": (
                "recorded IV-out-of-scope with absolute price errors and excluded from the "
                "pass rate; the excluded fraction and its composition are a headline beside "
                "every pass rate"
            ),
            "model_free": True,
            "label_free": False,
        },
        "assessability": {
            "degeneracy_threshold_normalized_vega": VEGA_DEGENERACY_NORMALIZED,
            "degeneracy_threshold_derivation": (
                "SECONDARY_NORMALIZED_TOLERANCE / VOLATILITY_TOLERANCE, so the threshold is "
                "derived from the two tolerances already declared here rather than asserted "
                "as a third independent number"
            ),
            "degenerate_eligible_limit": ASSESSABILITY_DEGENERATE_LIMIT,
            "rule": (
                "if more than 1% of eligible rows are vega-degenerate the >= 99% primary "
                "pass rate is reported NOT ASSESSABLE; the conditional pass rate on the "
                "non-degenerate eligible subset is also reported and does not substitute "
                "for the original declaration"
            ),
            "ordering": (
                "the eligible-set vega diagnostics are model-independent and are computed "
                "and persisted before any model output is scored; the ordering is enforced "
                "structurally by a separate human-invoked eligibility step"
            ),
            "secondary_view": {
                "rule": "0.01 * vega_BS > 3 * |E_CRR - E_BS| AND normalized vega_BS > 1e-4",
                "descriptive_only": True,
                "never_replaces_the_declared_rule": True,
                "disclosure": SECONDARY_VIEW_DISCLOSURE,
            },
        },
        "secondary_screen": {
            "statement": "|u_hat - u| <= 5.5e-4",
            "units": "normalized, u = V / (S * exp(-q * T))",
            "kind": "flat absolute",
            "derivation": (
                "shortest-maturity at-the-money transmission of the same 0.01 volatility "
                "standard: minimum ATM normalized vega over the declared domain is about "
                "0.0552 at T = 0.019178, and 5.5e-4 / 0.0552 = 0.0099 <= 0.01"
            ),
            "sufficiency": (
                "sufficient for the volatility standard at the money at every maturity; not "
                "sufficient away from the money, where vega is smaller"
            ),
        },
        "low_price_contracts": (
            "the tolerance is not relative to the option price. A price-relative rule would "
            "demand sub-penny accuracy on a deep out-of-the-money contract, below the "
            "resolution at which the label itself is meaningful. The fraction of rows whose "
            "reference price is below their own tolerance is reported separately, because "
            "'meets tolerance' is close to vacuous there."
        ),
        "calibration": {
            "statement": "|bias| / RMSE <= max(0.25, z / sqrt(n)) within every gated slice",
            "prediction": "the deployed output",
            "units": "normalized",
            "units_rationale": (
                "the network's five inputs exclude spot, so normalized error is "
                "spot-independent by construction while A is spot-dominated; a physical-unit "
                "gate would weight by a variable the model cannot see"
            ),
            "family_wise_alpha": CALIBRATION_FAMILY_WISE_ALPHA,
            "gated_slices": len(GATED_SLICE_FAMILY),
            "bonferroni_z": bonferroni_z(len(GATED_SLICE_FAMILY)),
            "noise_term_binds_below_rows": int(
                np.ceil((bonferroni_z(len(GATED_SLICE_FAMILY)) / CALIBRATION_RATIO) ** 2)
            ),
            "ungated": (
                "the s partition and its leg-dominance sub-split are reported and never "
                "gated: floor-dominated rows carry a structurally determined bias sign, so a "
                "failure there tests the head's design rather than calibration-readiness"
            ),
            "primary_statistic": (
                "the spread (max - min) of signed bias across the moneyness family and "
                "across the maturity family"
            ),
            "status": CALIBRATION_IS_A_PROXY,
        },
        "relation_to_existing_gates": (
            "additional, not a replacement: the Task 9H development criterion in "
            "configs/american_neural_pilot_acceptance_v1.toml is unchanged and unrevised"
        ),
        "deliberately_not_used": (
            "no quoting-increment anchor -- this project holds no market data, and a "
            "fraction-of-underlying tolerance would scale an option's tick by the "
            "underlying's level; and no absolute price floor, because A >= 34.9 over the "
            "declared domain makes any such floor non-binding"
        ),
        "domain_gap_statistics": {
            "source": "artifacts/task-9h/domain/european-comparator-domain-v1.json",
            "states": 275496,
            "maximum_positive": DOMAIN_GAP_MAXIMUM_POSITIVE,
            "minimum_signed": DOMAIN_GAP_MINIMUM_SIGNED,
            "median": DOMAIN_GAP_MEDIAN,
            "maximum_over_median": DOMAIN_GAP_MAXIMUM_POSITIVE / DOMAIN_GAP_MEDIAN,
            "note": (
                "over the Halton and structured domain sample, not over partition rows; the "
                "per-row medians on validation and H1 are measured, not assumed"
            ),
        },
    }
