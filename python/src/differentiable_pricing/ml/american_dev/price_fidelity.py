"""Task 9H price fidelity: the margin/variation decomposition and the B.3 slices.

**No training, no weight change, no configuration change.** Every number here
comes from two frozen checkpoints evaluated under stated deployment floors.

What B.0 measures
-----------------
The European-leg margin is a *deployment-time* transformation rather than a
weight, so both frozen weight sets can be evaluated under both floor
definitions::

                     delta = 0        delta = 1e-4
    E2b weights      existing         new
    E2c weights      new              existing

* **Row difference** -- fixed weights, varying delta -- is the **pure margin
  effect**.
* **Column difference** -- fixed delta, varying weights -- is
  **attempt-to-attempt variation at fixed design**.

The column difference is deliberately *not* called initialization-seed variance.
E2b's checkpoint was selected under ``delta = 0`` and E2c's under
``delta = 1e-4``, so in general it combines the initialization seed with the
selected epoch. This project's measured case is narrower than that -- both runs
selected epoch 89, the loss of a raw-loss head never sees the margin, and the
shuffle seed is shared -- and that measurement is reported beside the label
rather than replacing it. Either way it is a paired observation with ``n = 2``,
not a variance estimate, and **no selection gate is derived from it**.

What B.3 measures
-----------------
Global and sliced price fidelity against the adjacent-depth-averaged CRR target,
for the **raw network** and the **deployed** output separately, on ``validation``
and on H1, under the predeclared tolerance in :mod:`.tolerance`.

Rows are additionally partitioned by deployment behaviour using the projection
weight ``s`` -- an exact derivative, so no threshold is invented -- and, where
the floor binds, by which leg of the floor dominates. The two legs matter
because the intrinsic leg is independent of volatility, so wherever it dominates
the deployed volatility sensitivity is zero by construction.

Reuse rather than reimplementation
----------------------------------
Slices, price metrics, bound diagnostics and shape diagnostics are Task 9G's
own, driven over an evaluation-only view of a frozen model. A view is an
``nn.Module`` returning a physical price, which is the entire interface those
functions need -- so their **bumped-state** predictions go through the
overridden margin too, which a "recompute only the centre prediction" shortcut
would silently get wrong.

``_slice_masks`` is imported privately on purpose. Restating its binning here
would let the slice definitions drift from the digest-pinned acceptance file,
and ``american_pilot`` is itself digest-pinned by the Task 9G protocol, so a
public alias cannot be added to it.

Nothing here creates a model-selection gate.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from ..american_pilot import (
    _slice_masks,
    assess_arm,
    error_statistics,
    physical_features,
    predict_prices,
    shape_diagnostics,
    sliced_metrics,
)
from ..artifact import write_json_atomic
from .attempts import ACCEPTANCE_CONFIG_PATH, EUROPEAN_FLOOR_MARGIN, load_toml
from .eligibility import assert_matches as assert_eligibility_matches
from .eligibility import conditional_subset
from .frozen import (
    MarginOverrideModel,
    RawNetworkModel,
    _guarded_artifact_path,
    diagnostic_provenance,
    floor_leg_weight,
    projection_weight,
)
from .representation import AmericanDevPriceModel, discounted_spot
from .tolerance import (
    GATED_SLICE_FAMILY,
    NOT_ASSESSABLE,
    PASS_RATE_BAR,
    RESOLVABILITY_SAFETY_FACTOR,
    SECONDARY_NORMALIZED_TOLERANCE,
    assert_slice_family_matches,
    bias_spread,
    black_scholes_vega,
    calibration_statistics,
    comparison_floors,
    declaration,
    eligible_vega_diagnostics,
    meets_primary,
    meets_secondary,
    volatility_equivalent_error,
)

PRICE_FIDELITY_SCHEMA: Final = "american-dev-price-fidelity/1"

#: The two deployment floors every cell of the 2x2 is evaluated under.
MARGIN_CELLS: Final = (0.0, EUROPEAN_FLOOR_MARGIN)

#: The three families the reused shape diagnostic reports, kept here so the
#: breakdown is reported by name rather than as a single total.
SHAPE_FAMILIES: Final = ("spot_monotonicity", "spot_convexity", "volatility_monotonicity")
BOUND_FAMILIES: Final = (
    "intrinsic_lower_bound",
    "european_comparator_lower_bound",
    "american_call_upper_bound",
    "american_put_rate_aware_upper_bound",
)

#: The deployment-behaviour partition. The cut points are on ``s``, which is a
#: probability-like exact derivative in [0, 1], so they are readings of the
#: function rather than invented thresholds.
FLOOR_DOMINATED_MAXIMUM: Final = 0.01
NETWORK_DOMINATED_MINIMUM: Final = 0.99

#: The pre-registered arithmetic bound on the margin's own effect, recorded
#: before it was measured. The upper bound is the worst case in which the full
#: margin becomes added error on every floored row; the lower bound is the best
#: case in which every floored row is corrected upward by the full margin.
DELTA_EFFECT_PREDICTION: Final = {
    "quantity": "relative change in overall normalized RMSE at fixed weights, delta 0 -> 1e-4",
    "lower": -0.030,
    "upper": 0.001,
    "derivation": (
        "upper bound: the full margin becomes added error on every floored row. lower "
        "bound: every floored row is corrected upward by the full margin."
    ),
    "motivating_observation": (
        "the observed E2b -> E2c change was -5.68% normalized RMSE and +3.57% material "
        "shape violations, outside both bounds, which is why an attempt-to-attempt "
        "component of roughly 3-5% was expected"
    ),
    "if_outside": (
        "record the prediction as falsified and re-derive the arithmetic rather than "
        "explaining the discrepancy away"
    ),
}

COLUMN_DIFFERENCE_LABEL: Final = "attempt-to-attempt variation at fixed design"
COLUMN_DIFFERENCE_CAVEAT: Final = (
    "not initialization-seed variance: in general the column difference combines the "
    "initialization seed with the selected epoch, because E2b's checkpoint was selected "
    "under delta = 0 and E2c's under delta = 1e-4. It is a paired observation with n = 2, "
    "not a seed-variance distribution, and no selection gate is derived from it."
)

NO_SELECTION_GATE: Final = (
    "every number here is a development measurement over a validation partition this "
    "loop has selected against across nine attempts, plus reserved train rows evaluated "
    "once; none of it is a project result and none of it becomes a gate"
)


class PriceFidelityError(RuntimeError):
    """Raised when a price-fidelity measurement cannot be made as declared."""


# ---------------------------------------------------------------------------
# Per-cell metrics
# ---------------------------------------------------------------------------


def signed_bias(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Mean **signed** error. The reused statistics report absolute error only."""
    difference = np.asarray(prediction, dtype=np.float64) - np.asarray(
        reference, dtype=np.float64
    )
    return float(np.mean(difference))


def _normalized(columns: Mapping[str, np.ndarray]) -> np.ndarray:
    """``A = S * exp(-q * T)``, the scale the normalized target divides by."""
    with torch.no_grad():
        return discounted_spot(
            torch.as_tensor(physical_features(columns), dtype=torch.float64)
        ).numpy()


def diagnostic_breakdown(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    """Bound and shape violations as named families, not as two totals."""
    checks = diagnostics["checks"]
    return {
        "material_bound_violations": diagnostics["material_bound_violations"],
        "material_shape_violations": diagnostics["material_shape_violations"],
        "bound_families": {
            name: {
                "material_violations": checks[name]["material_violations"],
                "maximum_violation": checks[name]["maximum_violation"],
                "eligible_rows": checks[name]["eligible_rows"],
            }
            for name in BOUND_FAMILIES
        },
        "shape_families": {
            name: {
                "material_violations": checks[name]["material_violations"],
                "maximum_violation": checks[name]["maximum_violation"],
                "eligible_rows": checks[name]["eligible_rows"],
            }
            for name in SHAPE_FAMILIES
        },
    }


def cell_metrics(
    view: torch.nn.Module,
    columns: Mapping[str, np.ndarray],
    acceptance: Mapping[str, Any],
    batch_size: int,
    *,
    with_diagnostics: bool = True,
) -> dict[str, Any]:
    """One ``(weights, margin)`` cell: price metrics, bias and diagnostics.

    Everything is Task 9G's own implementation applied to a frozen view, so a
    cell is measured by the code that measured the attempt it came from.
    """
    physical = physical_features(columns)
    prediction = predict_prices(view, physical, batch_size)
    reference = np.asarray(columns["american_price"], dtype=np.float64)
    scale = _normalized(columns)
    metrics: dict[str, Any] = {
        "slices": sliced_metrics(columns, prediction, acceptance),
        "signed_bias": {
            "normalized": signed_bias(reference / scale, prediction / scale),
            "physical": signed_bias(reference, prediction),
        },
    }
    if with_diagnostics:
        diagnostics, _ = shape_diagnostics(
            view, columns, acceptance, batch_size=batch_size, center_prediction=prediction
        )
        metrics["diagnostics"] = diagnostic_breakdown(diagnostics)
        metrics["gate"] = assess_arm(
            {"slices": metrics["slices"], "diagnostics": diagnostics},
            acceptance,
            "validation_final_entry",
        )
    return metrics, prediction


# ---------------------------------------------------------------------------
# B.0 -- the 2x2
# ---------------------------------------------------------------------------


def _relative(new: float, old: float) -> float | None:
    return None if old == 0.0 else float((new - old) / old)


def decompose(
    models: Mapping[str, AmericanDevPriceModel],
    columns: Mapping[str, np.ndarray],
    acceptance: Mapping[str, Any],
    batch_size: int,
) -> dict[str, Any]:
    """The fixed-weight margin / attempt-variation decomposition on one partition.

    ``models`` maps a label (``"E2b"``, ``"E2c"``) to a frozen model. Every cell
    is evaluated identically; the diagonal cells reproduce what those attempts
    already recorded, which is the check that the off-diagonal cells mean what
    they claim.
    """
    labels = sorted(models)
    cells: dict[str, dict[str, Any]] = {}
    for label in labels:
        for margin in MARGIN_CELLS:
            view = MarginOverrideModel(models[label], margin)
            metrics, _ = cell_metrics(view, columns, acceptance, batch_size)
            cells[f"{label}@delta={margin:g}"] = {
                "weights": label,
                "european_margin": float(margin),
                "is_native_floor": float(margin) == float(models[label].european_margin),
                **metrics,
            }

    def overall(label: str, margin: float, statistic: str) -> float:
        return float(
            cells[f"{label}@delta={margin:g}"]["slices"]["overall"]["normalized"][statistic]
        )

    def shape(label: str, margin: float) -> int:
        return int(
            cells[f"{label}@delta={margin:g}"]["diagnostics"]["material_shape_violations"]
        )

    low, high = MARGIN_CELLS
    margin_effect = {
        label: {
            "normalized_rmse": {
                "at_zero_margin": overall(label, low, "rmse"),
                "at_declared_margin": overall(label, high, "rmse"),
                "relative_change": _relative(
                    overall(label, high, "rmse"), overall(label, low, "rmse")
                ),
            },
            "material_shape_violations": {
                "at_zero_margin": shape(label, low),
                "at_declared_margin": shape(label, high),
                "change": shape(label, high) - shape(label, low),
            },
            "material_bound_violations": {
                "at_zero_margin": int(
                    cells[f"{label}@delta={low:g}"]["diagnostics"]["material_bound_violations"]
                ),
                "at_declared_margin": int(
                    cells[f"{label}@delta={high:g}"]["diagnostics"]["material_bound_violations"]
                ),
            },
        }
        for label in labels
    }
    prediction_check = {
        label: {
            "measured_relative_rmse_change": entry["normalized_rmse"]["relative_change"],
            "predicted_interval": [
                DELTA_EFFECT_PREDICTION["lower"],
                DELTA_EFFECT_PREDICTION["upper"],
            ],
            "within_prediction": (
                entry["normalized_rmse"]["relative_change"] is not None
                and DELTA_EFFECT_PREDICTION["lower"]
                <= entry["normalized_rmse"]["relative_change"]
                <= DELTA_EFFECT_PREDICTION["upper"]
            ),
        }
        for label, entry in margin_effect.items()
    }
    for entry in prediction_check.values():
        entry["falsified"] = not entry["within_prediction"]

    attempt_variation = {}
    if len(labels) == 2:
        first, second = labels
        for margin in MARGIN_CELLS:
            attempt_variation[f"delta={margin:g}"] = {
                "normalized_rmse": {
                    first: overall(first, margin, "rmse"),
                    second: overall(second, margin, "rmse"),
                    "relative_difference": _relative(
                        overall(second, margin, "rmse"), overall(first, margin, "rmse")
                    ),
                },
                "material_shape_violations": {
                    first: shape(first, margin),
                    second: shape(second, margin),
                    "difference": shape(second, margin) - shape(first, margin),
                },
            }
    return {
        "cells": cells,
        "margin_effect": margin_effect,
        "prediction": DELTA_EFFECT_PREDICTION,
        "prediction_check": prediction_check,
        "attempt_variation": attempt_variation,
        "attempt_variation_label": COLUMN_DIFFERENCE_LABEL,
        "attempt_variation_caveat": COLUMN_DIFFERENCE_CAVEAT,
        "no_selection_gate": NO_SELECTION_GATE,
    }


# ---------------------------------------------------------------------------
# The deployment-behaviour partition
# ---------------------------------------------------------------------------


def projection_partition(
    model: AmericanDevPriceModel,
    columns: Mapping[str, np.ndarray],
    european_margin: float | None = None,
) -> dict[str, np.ndarray]:
    """Row masks for floor-dominated, crossover and network-dominated states.

    Built from ``s``, the exact derivative of the deployed output with respect
    to the direct normalized price, and from ``w``, the exact derivative of the
    floor's smooth maximum with respect to its European leg. Both live in
    ``[0, 1]``, so the partition reads the function rather than asserting a
    threshold on an arbitrary scale.
    """
    tensor = torch.as_tensor(physical_features(columns), dtype=torch.float64)
    with torch.no_grad():
        weight = projection_weight(model, tensor, european_margin).numpy()
        leg = floor_leg_weight(model, tensor, european_margin).numpy()
    floor_dominated = weight < FLOOR_DOMINATED_MAXIMUM
    network_dominated = weight > NETWORK_DOMINATED_MINIMUM
    crossover = ~(floor_dominated | network_dominated)
    binding = ~network_dominated
    return {
        "projection:floor_dominated": floor_dominated,
        "projection:crossover": crossover,
        "projection:network_dominated": network_dominated,
        "floor_leg:european_dominant": binding & (leg > NETWORK_DOMINATED_MINIMUM),
        "floor_leg:intrinsic_dominant": binding & (leg < FLOOR_DOMINATED_MAXIMUM),
        "floor_leg:mixed": binding
        & (leg >= FLOOR_DOMINATED_MAXIMUM)
        & (leg <= NETWORK_DOMINATED_MINIMUM),
        "_projection_weight": weight,
        "_floor_leg_weight": leg,
    }


# ---------------------------------------------------------------------------
# B.3 -- price fidelity under the predeclared tolerance
# ---------------------------------------------------------------------------


def _masked_statistics(
    reference: np.ndarray,
    prediction: np.ndarray,
    scale: np.ndarray,
    mask: np.ndarray,
    quantile_method: str,
) -> dict[str, Any]:
    if not bool(mask.any()):
        return {"rows": 0, "normalized": None, "physical": None, "signed_bias": None}
    return {
        "rows": int(mask.sum()),
        "fraction": float(mask.mean()),
        "normalized": error_statistics(
            reference[mask] / scale[mask],
            prediction[mask] / scale[mask],
            quantile_method=quantile_method,
        ),
        "physical": error_statistics(
            reference[mask], prediction[mask], quantile_method=quantile_method
        ),
        "signed_bias": {
            "normalized": signed_bias(
                reference[mask] / scale[mask], prediction[mask] / scale[mask]
            ),
            "physical": signed_bias(reference[mask], prediction[mask]),
        },
    }


def _finite_summary(values: np.ndarray) -> dict[str, float | None]:
    """Distribution summary that survives an infinite volatility-equivalent error.

    Vega underflows to denormals deep in and out of the money, so the quotient
    can overflow. Those rows still count as failures in the pass rate -- the
    statistic the bar is stated on -- but they have no finite magnitude to
    average or to quantile. They are excluded from the moments and counted
    separately rather than clamped to an invented ceiling, and an all-infinite
    set yields ``None`` rather than a fabricated number.

    This is not cosmetic: ``write_json_atomic`` refuses a payload containing an
    infinity, so an unguarded quantile would discard a completed analysis at the
    moment it tried to persist it.
    """
    if values.size == 0:
        return {
            "median_volatility_error": None,
            "p99_volatility_error": None,
            "maximum_volatility_error": None,
        }
    return {
        "median_volatility_error": float(np.median(values)),
        "p99_volatility_error": float(np.quantile(values, 0.99)),
        "maximum_volatility_error": float(values.max()),
    }


def tolerance_rates(
    reference: np.ndarray,
    prediction: np.ndarray,
    scale: np.ndarray,
    vega: np.ndarray,
    eligible: np.ndarray,
    non_degenerate: np.ndarray,
    assessable: bool,
    mask: np.ndarray,
) -> dict[str, Any]:
    """Primary and secondary pass rates on one row set.

    The primary rate is over **eligible** rows only, and the excluded count
    travels with it so a rate can never be read as a fraction of all rows.

    When the predeclared assessability rule says the primary statistic is NOT
    ASSESSABLE -- more than 1% of eligible rows vega-degenerate -- the declared
    pass rate is reported as ``None`` under that verdict rather than as a number
    a degenerate denominator already decided. The conditional pass rate on the
    non-degenerate eligible subset is reported separately and is explicitly
    marked as **not** the declared statistic, so it cannot be quoted in its
    place.
    """
    rows = int(mask.sum())
    if rows == 0:
        return {"rows": 0}
    scoped = mask & eligible
    excluded = mask & ~eligible
    secondary = meets_secondary(
        (prediction[mask] - reference[mask]) / scale[mask]
    )
    result: dict[str, Any] = {
        "rows": rows,
        "iv_in_scope_rows": int(scoped.sum()),
        "iv_out_of_scope_rows": int(excluded.sum()),
        "iv_out_of_scope_fraction": float(excluded.sum() / rows),
        "secondary_pass_rate": float(secondary.mean()),
    }
    if bool(scoped.any()):
        volatility_error = volatility_equivalent_error(
            prediction[scoped], reference[scoped], vega[scoped]
        )
        primary = meets_primary(prediction[scoped], reference[scoped], vega[scoped])
        finite = np.isfinite(volatility_error)
        result["primary"] = {
            "verdict": "assessable" if assessable else NOT_ASSESSABLE,
            "pass_rate": float(primary.mean()) if assessable else None,
            "meets_bar": bool(primary.mean() >= PASS_RATE_BAR) if assessable else None,
            "not_assessable_reason": (
                None
                if assessable
                else (
                    "more than the declared 1% of eligible rows are vega-degenerate, so the "
                    ">= 99% pass rate is decided by the denominator before the model is "
                    "consulted; see the eligibility artifact"
                )
            ),
            "non_finite_volatility_error_rows": int((~finite).sum()),
            **_finite_summary(volatility_error[finite]),
            "minimum_in_scope_vega": float(vega[scoped].min()),
            "tail_moments_are_not_the_declared_statistic": (
                "the pass rate is what the 99% bar is stated on. The eligibility rule "
                "anchors on the label's own discretization error, not on the model's, so a "
                "row can be formally resolvable -- the reference is sharp enough to answer "
                "the question -- while the surrogate's error there is enormous in volatility "
                "terms. The upper quantiles and the maximum are therefore dominated by the "
                "smallest admitted vega and are reported for shape, not as a criterion."
            ),
        }
    else:
        result["primary"] = None
    conditional = mask & non_degenerate
    if not assessable and bool(conditional.any()):
        conditional_primary = meets_primary(
            prediction[conditional], reference[conditional], vega[conditional]
        )
        result["conditional_on_non_degenerate_eligible"] = {
            "rows": int(conditional.sum()),
            "fraction_of_eligible": (
                float(conditional.sum() / scoped.sum()) if bool(scoped.any()) else None
            ),
            "pass_rate": float(conditional_primary.mean()),
            "is_not_the_declared_statistic": True,
            "note": (
                "a different statistic on a different row set; reported because the declared "
                "one is NOT ASSESSABLE, and never as a substitute for it"
            ),
        }
    if bool(excluded.any()):
        result["out_of_scope_absolute_price_error"] = error_statistics(
            reference[excluded], prediction[excluded], quantile_method="linear"
        )
    return result


def characterize(
    model: AmericanDevPriceModel,
    columns: Mapping[str, np.ndarray],
    acceptance: Mapping[str, Any],
    batch_size: int,
    *,
    project_root: Path,
    row_set: str,
    eligibility_record: Mapping[str, Any],
    european_margin: float | None = None,
) -> dict[str, Any]:
    """Full B.3 characterization of one frozen model on one row set.

    Measures the **raw network** and the **deployed** output separately, because
    "is the learned function right?" and "is the shipped function right?" are
    different questions and the section H branches turn on which one fails.

    **An eligibility record is required, not optional.** Assessability is settled
    before any prediction is computed, by a separate human-invoked step whose
    artifact pins the row identity, the declaration digests and the protocol
    commit. Passing it in as a mandatory argument -- and re-verifying it against
    the rows actually being scored -- is what makes "eligibility before scoring"
    a property of the code rather than of the operator's discipline.
    """
    record = assert_eligibility_matches(eligibility_record, project_root, row_set, columns)
    assessable = bool(record["assessability"]["primary_statistic_assessable"])
    quantile_method = str(acceptance["quantile_method"])
    physical = physical_features(columns)
    tensor = torch.as_tensor(physical, dtype=torch.float64)
    reference = np.asarray(columns["american_price"], dtype=np.float64)
    scale = _normalized(columns)
    vega = black_scholes_vega(tensor).numpy()
    diagnostics = eligible_vega_diagnostics(
        tensor, np.asarray(columns["european_crr_price"], dtype=np.float64)
    )
    eligible = np.asarray(diagnostics["_eligible"], dtype=bool)
    non_degenerate = conditional_subset(diagnostics)
    if int(eligible.sum()) != int(record["vega_diagnostics"]["eligible_rows"]):
        raise PriceFidelityError(
            f"the eligibility artifact recorded "
            f"{int(record['vega_diagnostics']['eligible_rows'])} eligible rows but these rows "
            f"give {int(eligible.sum())}; the artifact does not describe what is being scored"
        )
    eligibility_terms = {"rule": diagnostics["eligibility_rule"]}
    floors = comparison_floors(tensor)

    masks = dict(_slice_masks(columns, acceptance))
    masks["iv_scope:in_scope"] = eligible
    masks["iv_scope:out_of_scope"] = ~eligible
    partition = projection_partition(model, columns, european_margin)
    weight = partition.pop("_projection_weight")
    leg = partition.pop("_floor_leg_weight")
    masks.update(partition)
    assert_slice_family_matches(list(masks))

    views = {
        "deployed": MarginOverrideModel(
            model, model.european_margin if european_margin is None else european_margin
        ),
        "raw_network": RawNetworkModel(model),
    }
    predictions = {
        name: predict_prices(view, physical, batch_size) for name, view in views.items()
    }

    per_prediction: dict[str, Any] = {}
    for name, prediction in predictions.items():
        slices = {
            slice_name: {
                **_masked_statistics(reference, prediction, scale, mask, quantile_method),
                "tolerance": tolerance_rates(
                    reference,
                    prediction,
                    scale,
                    vega,
                    eligible,
                    non_degenerate,
                    assessable,
                    mask,
                ),
            }
            for slice_name, mask in masks.items()
        }
        calibration = {
            slice_name: calibration_statistics(
                (prediction[mask] - reference[mask]) / scale[mask]
            )
            for slice_name, mask in masks.items()
            if bool(mask.any())
        }
        for slice_name, entry in calibration.items():
            entry["gated"] = slice_name in GATED_SLICE_FAMILY
        per_prediction[name] = {
            "slices": slices,
            "calibration": calibration,
            "calibration_gate": {
                "failing_gated_slices": sorted(
                    slice_name
                    for slice_name, entry in calibration.items()
                    if entry["gated"] and not entry["passes"]
                ),
            },
            "bias_spread": {
                family: bias_spread(calibration, family) for family in ("moneyness", "expiry")
            },
        }

    return {
        "rows": int(reference.size),
        "european_margin": float(
            model.european_margin if european_margin is None else european_margin
        ),
        "tolerance": declaration(),
        "eligibility": {
            "rule": eligibility_terms["rule"],
            "safety_factor": RESOLVABILITY_SAFETY_FACTOR,
            "artifact": {
                "row_set": record["row_set"],
                "protocol_commit": record["provenance"]["protocol_commit"],
                "assessability": record["assessability"],
                "computed_before_scoring": True,
            },
            "in_scope_rows": int(eligible.sum()),
            "in_scope_fraction": float(eligible.mean()),
            "out_of_scope_rows": int((~eligible).sum()),
            "out_of_scope_fraction": float((~eligible).mean()),
            "model_free": True,
            "label_free": False,
            "comparison_floors": {
                name: {
                    "in_scope_rows": int((vega > floor).sum()),
                    "in_scope_fraction": float((vega > floor).mean()),
                }
                for name, floor in floors.items()
            },
            "comparison_note": (
                "the two global-maximum floors are reported for comparison only and select "
                "nothing; one_sided uses the positive supremum the margin derivation used, "
                "two_sided uses the maximum of |E_CRR - E_BS| that the eligibility rule implies"
            ),
        },
        "reference_price_below_own_tolerance": {
            "rows": int((reference < SECONDARY_NORMALIZED_TOLERANCE * scale).sum()),
            "fraction": float((reference < SECONDARY_NORMALIZED_TOLERANCE * scale).mean()),
            "note": (
                "rows where meeting the tolerance is close to vacuous, reported separately "
                "rather than folded into a pass rate"
            ),
        },
        "projection_weight_summary": {
            "median": float(np.median(weight)),
            "floor_dominated_fraction": float((weight < FLOOR_DOMINATED_MAXIMUM).mean()),
            "network_dominated_fraction": float((weight > NETWORK_DOMINATED_MINIMUM).mean()),
            "floor_leg_median": float(np.median(leg)),
        },
        "predictions": per_prediction,
        "no_selection_gate": NO_SELECTION_GATE,
    }


def write_report(
    project_root: Path,
    relative: str,
    report: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Attach provenance and publish one diagnostic report atomically.

    Provenance is attached **here**, at the artifact boundary, rather than
    inside the measurement functions. Two consequences, both wanted: the strict
    clean-worktree attestation applies to everything that becomes a durable
    record, and a unit test can exercise the arithmetic without committing.
    """
    path = _guarded_artifact_path(project_root, relative, where="price fidelity report path")
    payload = {
        "schema_version": PRICE_FIDELITY_SCHEMA,
        **dict(report),
        "provenance": diagnostic_provenance(project_root),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, payload, overwrite=overwrite)
    return path


# ---------------------------------------------------------------------------
# The human-invoked driver
# ---------------------------------------------------------------------------

DEFAULT_OUTPUT_TEMPLATE: Final = "artifacts/task-9h/price/price-fidelity-{row_set}-v1.json"


def analyze(
    project_root: Path,
    row_set: str,
    *,
    output: str | None = None,
    overwrite: bool = False,
    batch_size: int = 4096,
) -> dict[str, Any]:
    """B.0 and B.3 on one row set, for both frozen checkpoints.

    Refuses to start without the matching eligibility artifact: assessability is
    settled before any prediction exists, and this is where that is enforced.
    """
    from .eligibility import load as load_eligibility
    from .eligibility import load_row_set
    from .frozen import FROZEN_ATTEMPTS, load_frozen_model

    project_root = Path(project_root)
    record = load_eligibility(project_root, row_set)
    columns = load_row_set(project_root, row_set)
    acceptance = load_toml(project_root / ACCEPTANCE_CONFIG_PATH)
    models: dict[str, AmericanDevPriceModel] = {}
    provenance: dict[str, Any] = {}
    for attempt_id, entry in FROZEN_ATTEMPTS.items():
        model, identity = load_frozen_model(project_root, attempt_id)
        models[str(entry["label"])] = model
        provenance[str(entry["label"])] = identity
    report = {
        "row_set": record["row_set"],
        "checkpoints": provenance,
        "decomposition": decompose(models, columns, acceptance, batch_size),
        "characterization": {
            label: characterize(
                model,
                columns,
                acceptance,
                batch_size,
                project_root=project_root,
                row_set=row_set,
                eligibility_record=record,
            )
            for label, model in models.items()
        },
    }
    relative = output or DEFAULT_OUTPUT_TEMPLATE.format(row_set=row_set)
    write_report(project_root, relative, report, overwrite=overwrite)
    return report


def compact_fidelity(report: Mapping[str, Any]) -> dict[str, Any]:
    """The small, agent-readable view: the 2x2, the verdicts and the branches."""
    decomposition = report["decomposition"]
    summary: dict[str, Any] = {
        "row_set": report["row_set"]["row_set"],
        "rows": report["row_set"]["rows"],
        "protocol_commit": report.get("provenance", {}).get("protocol_commit"),
        "margin_effect": {
            label: entry["normalized_rmse"]["relative_change"]
            for label, entry in decomposition["margin_effect"].items()
        },
        "delta_prediction_falsified": {
            label: entry["falsified"]
            for label, entry in decomposition["prediction_check"].items()
        },
        "attempt_variation": {
            key: entry["normalized_rmse"].get("relative_difference")
            for key, entry in decomposition["attempt_variation"].items()
        },
        "attempt_variation_label": decomposition["attempt_variation_label"],
    }
    for label, entry in report["characterization"].items():
        overall = entry["predictions"]["deployed"]["slices"]["overall"]
        tolerance = overall["tolerance"]
        summary[label] = {
            "normalized_rmse": overall["normalized"]["rmse"],
            "assessability": entry["eligibility"]["artifact"]["assessability"]["verdict"],
            "primary_pass_rate": (tolerance.get("primary") or {}).get("pass_rate"),
            "secondary_pass_rate": tolerance.get("secondary_pass_rate"),
            "conditional_pass_rate": (
                tolerance.get("conditional_on_non_degenerate_eligible", {}).get("pass_rate")
            ),
            "iv_out_of_scope_fraction": tolerance.get("iv_out_of_scope_fraction"),
            "failing_gated_calibration_slices": entry["predictions"]["deployed"][
                "calibration_gate"
            ]["failing_gated_slices"],
        }
    return summary
