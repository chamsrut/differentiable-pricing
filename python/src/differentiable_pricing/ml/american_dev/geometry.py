"""Task 9H validation-set geometry of the binding constraints (exploratory).

Every recorded Task 9H attempt has failed the same way: the price gates are
reachable, the structural gates are not. Five attempts report thousands of
``european_comparator_lower_bound`` and ``intrinsic_lower_bound`` violations at
a material tolerance of ``1e-6`` in normalized units. That count is a property
of the model **and** of the data: a row whose American price sits a hair above
its European comparator is one an approximation of any accuracy will cross.

This module measures the second half — how much slack the validation rows
actually leave above each binding constraint — so that a later decision about
architectural floors is made against the geometry rather than against intuition.
It **trains nothing**, reserves no attempt, writes no attempt evidence and
touches no attempt log.

What it measures, in the repository's own definitions and no others:

* ``A = S * exp(-q * T)``, the reconstruction scale of the Task 9G/9H
  normalized target, taken from :func:`..american_pilot.sliced_metrics`'s
  definition rather than restated;
* the **normalized European slack** ``(V_American - V_European_CRR) / A``, using
  the dataset's stored CRR European leg — the same comparator the
  ``european_comparator_lower_bound`` diagnostic uses;
* the **normalized intrinsic slack** ``(V_American - intrinsic) / A``, using the
  dataset's stored intrinsic value;
* the **normalized early-exercise premium** ``premium / A`` from the dataset's
  stored column, reported for the positive-premium population separately;
* the **normalized comparator discrepancy**
  ``(V_European_CRR - V_European_BS) / A`` between the dataset's stored CRR
  European leg and the analytic Black--Scholes value a model can compute at
  inference from its own inputs. This is the quantity that decides how much of
  the ``european_comparator_lower_bound`` gate an analytic-European floor can
  reach at all: where the CRR leg sits *above* the analytic value by more than
  the material tolerance, a model pinned exactly to the analytic floor is still
  counted as violating the stored comparator. The analytic value comes from
  :func:`..american_dev.representation.european_price_array`, which calls the
  same function the ``smooth_lower_floor`` head enforces, so the measurement and
  the enforcement cannot drift apart;
* for each of those, the fraction of rows at or below a set of thresholds: the
  three predeclared absolute scales, and the normalized RMSE of the control,
  capacity and residual attempts, read out of the tracked append-only attempt
  log rather than typed in here.

**Scope and honesty.**

* The analysis partition is ``validation`` and nothing else. It is a module
  constant, checked offline by ``scripts/american_dev_attempts.py check``.
* The dataset identity check is Task 9H's hardened one, reused unchanged: it
  pins the manifest and **both reachable** partitions to the digests Task 9G
  locked. So ``train`` is hashed for identity while only ``validation`` is read
  as data, and the report says exactly that. No final or held-out partition is
  opened, hashed, stat-ed, imported, counted or inspected.
* The acceptance configuration is read through a **whitelist** of the three keys
  this analysis needs. The key naming Task 9G's final partition is never
  resolved, read into a variable, or written to the report.
* The output is **exploratory and validation-selected**. Task 9H selects against
  ``validation`` repeatedly; a geometry measured on the same partition the loop
  selects on is a development observation, never a project result, and the
  report carries that label in its own payload.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from ..american_pilot import _slice_masks, physical_features, verify_partition_policy
from ..artifact import write_json_atomic
from .attempts import (
    ACCEPTANCE_CONFIG_PATH,
    ALLOWED_SPLITS,
    ATTEMPT_LOG_PATH,
    PROTOCOL_CONFIG_PATH,
    assert_contained_relative_path,
    assert_path_allowed,
    assert_split_allowed,
    attempt_log_entries,
    load_toml,
    repository_identity,
    sha256_file,
    source_digests,
    validate_acceptance_config,
    verify_committed_source,
)
from .representation import european_price_array
from .workbench import (
    load_partition,
    locked_dataset_identity,
    locked_dataset_paths,
    restricted_manifest,
    verify_dataset_identity,
)

GEOMETRY_SCHEMA: Final = "american-dev-validation-geometry/2"

#: The one partition this analysis reads as data. A module constant rather than
#: an argument: there is no flag, subcommand or configuration key that can point
#: the geometry analysis at another partition, and
#: ``scripts/american_dev_attempts.py check`` re-verifies that offline.
ANALYSIS_PARTITION: Final = "validation"

#: Where the exploratory report is written. Beneath the ignored ``artifacts/``
#: tree, like every other Task 9H output: this is a development observation, not
#: frozen evidence, and it is never committed.
#: Schema ``/2`` adds the comparator discrepancy and writes to its own file, so
#: the ``/1`` report an earlier run already published is left byte-for-byte
#: intact rather than overwritten with a differently shaped payload.
DEFAULT_OUTPUT: Final = "artifacts/task-9h/geometry/validation-geometry-v2.json"

#: Predeclared absolute scales, in normalized units, spanning "numerically
#: indistinguishable from the bound" (``1e-6``, the acceptance file's own
#: material tolerance) up to a thousandth of the discounted spot.
ABSOLUTE_THRESHOLDS: Final = (1.0e-6, 1.0e-4, 1.0e-3)

#: The recorded attempts whose normalized RMSE is used as an error-scale
#: threshold, and the name each one carries in the report. Their values are read
#: from the tracked attempt log, never restated here, so a threshold cannot
#: drift from the measurement it is supposed to represent.
REFERENCE_ATTEMPTS: Final = (
    ("control_rmse", "scratch_direct_control_v1"),
    ("capacity_rmse", "scratch_capacity_v1"),
    ("residual_rmse", "scratch_residual_architecture_v1"),
)

#: Reported quantiles of each slack distribution.
QUANTILES: Final = (
    0.0,
    0.001,
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    0.999,
    1.0,
)

#: The measured quantities, and the exact definition each report entry carries.
MEASURED_QUANTITIES: Final = {
    "european_slack_normalized": "(american_price - european_crr_price) / A",
    "intrinsic_slack_normalized": "(american_price - intrinsic_value) / A",
    "early_exercise_premium_normalized": "early_exercise_premium / A",
}

#: The comparator discrepancy, measured and reported separately from the slacks
#: because it is a property of the two European valuations alone -- it does not
#: involve the American price at all, and a positive value is not a defect of
#: any model.
COMPARATOR_DISCREPANCY: Final = "(european_crr_price - european_black_scholes_price) / A"

#: Exceedance thresholds for the comparator discrepancy, in normalized units.
#: ``0`` answers "how often does the stored CRR leg sit above the analytic value
#: at all"; ``1e-6`` is the acceptance file's own material tolerance, so it
#: answers "how often is it above by enough to be counted as a violation"; and
#: ``1e-4`` is the predeclared smoothing temperature's order of magnitude.
DISCREPANCY_THRESHOLDS: Final = (0.0, 1.0e-6, 1.0e-4)

#: Quantiles reported for the comparator discrepancy.
DISCREPANCY_QUANTILES: Final = (0.50, 0.90, 0.95, 0.99, 1.0)

#: The falsifier :func:`column_consistency` applies, and the slack it allows.
DIFFERENCE_DEFINITION: Final = (
    "|early_exercise_premium - (american_price - european_crr_price)|"
)
ROUNDING_TOLERANCE_ULPS: Final = 8.0

EXPLORATORY_LABEL: Final = (
    "exploratory validation-set geometry; Task 9H selects against validation "
    "repeatedly, so this is a development observation and not a project result"
)


class GeometryError(RuntimeError):
    """Raised when the validation-geometry analysis cannot proceed safely."""


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------


def attempt_rmse_thresholds(log_path: Path) -> list[dict[str, Any]]:
    """The recorded normalized RMSE of each reference attempt, read from the log.

    Offline: it parses the tracked append-only attempt log and opens no dataset.
    A missing reference attempt is a hard failure rather than a silently dropped
    threshold, because a report whose threshold set depends on which attempts
    happened to parse is not comparable with the next one.
    """
    recorded = {
        str(record.get("attempt_id")): record for record in attempt_log_entries(Path(log_path))
    }
    thresholds: list[dict[str, Any]] = []
    for name, attempt_id in REFERENCE_ATTEMPTS:
        record = recorded.get(attempt_id)
        if record is None:
            raise GeometryError(
                f"attempt {attempt_id!r} is not recorded in '{ATTEMPT_LOG_PATH}'; its normalized "
                "RMSE is a threshold of this analysis and is never typed in by hand"
            )
        try:
            value = float(record["price_metrics"]["overall_normalized"]["rmse"])
        except (KeyError, TypeError, ValueError) as error:
            raise GeometryError(
                f"attempt {attempt_id!r} records no overall normalized RMSE: {error}"
            ) from error
        if not np.isfinite(value) or value <= 0.0:
            raise GeometryError(f"attempt {attempt_id!r} records a non-positive RMSE")
        thresholds.append(
            {
                "name": name,
                "value": value,
                "source": (
                    f"{ATTEMPT_LOG_PATH} :: {attempt_id} :: "
                    "price_metrics.overall_normalized.rmse"
                ),
            }
        )
    return thresholds


def thresholds(log_path: Path) -> list[dict[str, Any]]:
    """The full ordered threshold set: three absolute scales, then three RMSEs."""
    absolute = [
        {
            "name": f"absolute_{value:g}",
            "value": float(value),
            "source": "predeclared absolute scale in normalized units",
        }
        for value in ABSOLUTE_THRESHOLDS
    ]
    return [*absolute, *attempt_rmse_thresholds(log_path)]


# ---------------------------------------------------------------------------
# Distribution statistics
# ---------------------------------------------------------------------------


def slack_statistics(
    values: np.ndarray,
    threshold_set: Sequence[Mapping[str, Any]],
    *,
    quantile_method: str,
) -> dict[str, Any]:
    """Quantiles, counts and at-or-below fractions of one normalized slack vector.

    ``count_at_or_below`` uses ``<=``, deliberately: the question a floor asks is
    how many rows sit *within* a given distance of the bound, and a row exactly
    on the bound is the extreme case of that, not an exception to it.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise GeometryError("a slack vector must be one-dimensional")
    if array.size == 0:
        return {"rows": 0, "quantiles": None, "count_at_or_below": None}
    if not bool(np.isfinite(array).all()):
        raise GeometryError("a slack vector contains non-finite values")
    rows = int(array.size)
    counts = {
        str(entry["name"]): int((array <= float(entry["value"])).sum())
        for entry in threshold_set
    }
    return {
        "rows": rows,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=0)),
        "negative_rows": int((array < 0.0).sum()),
        "exactly_zero_rows": int((array == 0.0).sum()),
        "quantiles": {
            f"{quantile:g}": float(np.quantile(array, quantile, method=quantile_method))
            for quantile in QUANTILES
        },
        "count_at_or_below": counts,
        "fraction_at_or_below": {name: count / rows for name, count in counts.items()},
    }


def discounted_spot(columns: Mapping[str, np.ndarray]) -> np.ndarray:
    """``A = S * exp(-q * T)``.

    Spelled exactly as :func:`..american_pilot.sliced_metrics` spells it: the
    reconstruction scale of the Task 9G/9H normalized target, not a new one.
    """
    spot = np.asarray(columns["spot"], dtype=np.float64)
    maturity = np.asarray(columns["maturity"], dtype=np.float64)
    dividend_yield = np.asarray(columns["dividend_yield"], dtype=np.float64)
    scale = spot * np.exp(-dividend_yield * maturity)
    if bool((scale <= 0.0).any()) or not bool(np.isfinite(scale).all()):
        raise GeometryError("the discounted spot A = S*exp(-q*T) is outside its domain")
    return scale


def normalized_slacks(columns: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The three measured quantities, normalized by ``A = S * exp(-q * T)``.

    The American price, the CRR European leg, the intrinsic value and the
    early-exercise premium are all stored dataset columns: this function
    introduces no financial semantics of its own and recomputes none of them.
    """
    scale = discounted_spot(columns)
    american = np.asarray(columns["american_price"], dtype=np.float64)
    return {
        "european_slack_normalized": (
            american - np.asarray(columns["european_crr_price"], dtype=np.float64)
        )
        / scale,
        "intrinsic_slack_normalized": (
            american - np.asarray(columns["intrinsic_value"], dtype=np.float64)
        )
        / scale,
        "early_exercise_premium_normalized": (
            np.asarray(columns["early_exercise_premium"], dtype=np.float64) / scale
        ),
    }


def population_masks(
    columns: Mapping[str, np.ndarray], bins: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    """The repository's own validation slices, reused rather than reimplemented.

    :func:`..american_pilot._slice_masks` is the single definition of what
    ``overall``, ``option_type:*``, ``expiry:*``, ``moneyness:*``,
    ``volatility:*``, ``premium_status:*`` and ``exercise_status:*`` mean in this
    project. Restating the bin edges here would let the geometry report and the
    attempt reports disagree about which rows a slice contains.
    """
    return _slice_masks(columns, {"bins": bins})


def geometry_populations(
    columns: Mapping[str, np.ndarray],
    bins: Mapping[str, Any],
    threshold_set: Sequence[Mapping[str, Any]],
    *,
    quantile_method: str,
) -> dict[str, Any]:
    """Every measured quantity, for every repository slice of the partition."""
    slacks = normalized_slacks(columns)
    output: dict[str, Any] = {}
    for name, mask in sorted(population_masks(columns, bins).items()):
        rows = int(mask.sum())
        entry: dict[str, Any] = {"rows": rows}
        for quantity, values in slacks.items():
            entry[quantity] = (
                slack_statistics(values[mask], threshold_set, quantile_method=quantile_method)
                if rows
                else {"rows": 0, "quantiles": None, "count_at_or_below": None}
            )
        output[name] = entry
    return output


def comparator_discrepancy_values(columns: Mapping[str, np.ndarray]) -> np.ndarray:
    """``(european_crr_price - european_black_scholes_price) / A``.

    The CRR leg is the dataset's stored column -- the one the
    ``european_comparator_lower_bound`` diagnostic compares against. The
    Black--Scholes leg is :func:`..representation.european_price_array`, the
    analytic continuous-yield value computed from the seven physical contract
    inputs, which is what a deployed model can evaluate at inference and what the
    ``smooth_lower_floor`` head enforces. Neither is recomputed here.

    A **positive** value is the case that matters: the stored comparator sits
    above the analytic value, so a model held exactly at the analytic floor is
    still below the stored one. Sign convention is fixed here and stated in the
    report so it cannot be read backwards.
    """
    crr = np.asarray(columns["european_crr_price"], dtype=np.float64)
    analytic = european_price_array(physical_features(columns))
    return (crr - analytic) / discounted_spot(columns)


def exceedance_statistics(
    values: np.ndarray,
    *,
    quantile_method: str,
    thresholds_above: Sequence[float] = DISCREPANCY_THRESHOLDS,
    quantiles: Sequence[float] = DISCREPANCY_QUANTILES,
) -> dict[str, Any]:
    """Counts, fractions **strictly above** each threshold, and quantiles.

    Deliberately the mirror image of :func:`slack_statistics`, which counts rows
    at or below a threshold. A slack is a budget and the question is how little
    of it a row has left; a discrepancy is an obstruction and the question is how
    often it exceeds a scale. Using ``>`` here matches
    ``ml.american_pilot.shape_diagnostics``, which counts a violation as
    ``value > tolerance``.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise GeometryError("a discrepancy vector must be one-dimensional")
    if array.size == 0:
        return {"rows": 0, "quantiles": None, "count_above": None, "fraction_above": None}
    if not bool(np.isfinite(array).all()):
        raise GeometryError("a discrepancy vector contains non-finite values")
    rows = int(array.size)
    counts = {
        f"above_{threshold:g}": int((array > float(threshold)).sum())
        for threshold in thresholds_above
    }
    return {
        "rows": rows,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=0)),
        "count_above": counts,
        "fraction_above": {name: count / rows for name, count in counts.items()},
        "quantiles": {
            f"{quantile:g}": float(np.quantile(array, quantile, method=quantile_method))
            for quantile in quantiles
        },
    }


def effectively_zero_premium_mask(
    columns: Mapping[str, np.ndarray], material_tolerance: float
) -> np.ndarray:
    """Rows whose normalized early-exercise premium is within the material tolerance.

    "Effectively zero" is the acceptance file's own
    ``material_normalized_tolerance``, not a fresh number: these are exactly the
    rows on which a model cannot be more than a tolerance above the European
    comparator without also being above the American price, so they are where a
    European floor is simultaneously most binding and most useful. Reported
    beside the repository's exact-zero ``premium_status:zero`` slice rather than
    instead of it.
    """
    if not material_tolerance > 0.0 or not np.isfinite(material_tolerance):
        raise GeometryError("the material normalized tolerance must be finite and positive")
    premium = np.asarray(columns["early_exercise_premium"], dtype=np.float64)
    return (premium / discounted_spot(columns)) <= float(material_tolerance)


def comparator_discrepancy(
    columns: Mapping[str, np.ndarray],
    bins: Mapping[str, Any],
    *,
    quantile_method: str,
    material_tolerance: float,
) -> dict[str, Any]:
    """The comparator discrepancy overall, on the repository slices, and where it binds."""
    values = comparator_discrepancy_values(columns)
    zero_premium = effectively_zero_premium_mask(columns, material_tolerance)
    populations: dict[str, Any] = {}
    for name, mask in sorted(population_masks(columns, bins).items()):
        populations[name] = (
            exceedance_statistics(values[mask], quantile_method=quantile_method)
            if bool(mask.any())
            else {"rows": 0, "quantiles": None, "count_above": None, "fraction_above": None}
        )
    populations["effectively_zero_premium"] = (
        exceedance_statistics(values[zero_premium], quantile_method=quantile_method)
        if bool(zero_premium.any())
        else {"rows": 0, "quantiles": None, "count_above": None, "fraction_above": None}
    )
    return {
        "definition": COMPARATOR_DISCREPANCY,
        "sign_convention": (
            "positive means the stored CRR European leg sits ABOVE the analytic "
            "Black-Scholes value, so an analytic-European floor does not reach the stored "
            "comparator on that row"
        ),
        "analytic_source": (
            "differentiable_pricing.ml.american_dev.representation.european_price_array, the "
            "same analytic function the smooth_lower_floor head enforces at inference"
        ),
        "effectively_zero_premium": {
            "definition": "early_exercise_premium / A <= material_normalized_tolerance",
            "material_normalized_tolerance": float(material_tolerance),
            "rows": int(zero_premium.sum()),
        },
        "thresholds_above": [float(value) for value in DISCREPANCY_THRESHOLDS],
        "quantiles": [float(value) for value in DISCREPANCY_QUANTILES],
        "populations": populations,
    }


def column_consistency(columns: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """How far the stored premium column is from the European slack it should equal.

    A cheap falsifier for the assumption this analysis rests on: if the stored
    ``early_exercise_premium`` is not ``american_price - european_crr_price``,
    the two are measuring different things and the report should say so instead
    of quietly presenting them as interchangeable.
    """
    american = np.asarray(columns["american_price"], dtype=np.float64)
    european = np.asarray(columns["european_crr_price"], dtype=np.float64)
    premium = np.asarray(columns["early_exercise_premium"], dtype=np.float64)
    difference = np.abs(premium - (american - european))
    if difference.size == 0:
        return {
            "definition": DIFFERENCE_DEFINITION,
            "rows": 0,
            "maximum_absolute_difference": 0.0,
            "maximum_normalized_difference": 0.0,
            "rows_beyond_double_rounding": 0,
        }
    # Exact equality is the wrong test: the three columns are stored
    # independently, so agreement can only be at the level of double rounding.
    # A row beyond that bound is a real disagreement and is counted as one.
    tolerance = ROUNDING_TOLERANCE_ULPS * float(np.finfo(np.float64).eps) * np.maximum(
        np.abs(american), np.abs(european)
    )
    return {
        "definition": DIFFERENCE_DEFINITION,
        "rows": int(difference.size),
        "maximum_absolute_difference": float(difference.max()),
        "maximum_normalized_difference": float((difference / discounted_spot(columns)).max()),
        "rows_beyond_double_rounding": int((difference > tolerance).sum()),
    }


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------


def _guarded_output(project_root: Path, relative: str) -> Path:
    where = "geometry output path"
    assert_path_allowed(relative, where=where)
    assert_contained_relative_path(relative, where=where)
    if not str(relative).replace("\\", "/").startswith("artifacts/"):
        raise GeometryError(
            "the exploratory geometry report lives beneath the ignored artifacts/ tree; it is a "
            "development observation, not frozen evidence"
        )
    return Path(project_root) / relative


def acceptance_view(acceptance: Mapping[str, Any]) -> dict[str, Any]:
    """The three acceptance keys this analysis reads, and no others.

    Whitelisted by key for the same reason :func:`..workbench.locked_dataset_identity`
    is: the acceptance configuration also names Task 9G's final partition, and
    that key is never resolved, read into a variable, or written to the report.
    """
    bins = acceptance.get("bins")
    quantile_method = acceptance.get("quantile_method")
    diagnostics = acceptance.get("diagnostics")
    if not isinstance(bins, Mapping):
        raise GeometryError(f"'{ACCEPTANCE_CONFIG_PATH}' declares no [bins] table")
    if not isinstance(quantile_method, str) or not quantile_method:
        raise GeometryError(f"'{ACCEPTANCE_CONFIG_PATH}' declares no quantile_method")
    tolerance = (
        diagnostics.get("material_normalized_tolerance")
        if isinstance(diagnostics, Mapping)
        else None
    )
    if not isinstance(tolerance, int | float) or not float(tolerance) > 0.0:
        raise GeometryError(
            f"'{ACCEPTANCE_CONFIG_PATH}' declares no positive "
            "[diagnostics].material_normalized_tolerance"
        )
    return {
        "bins": dict(bins),
        "quantile_method": quantile_method,
        "material_normalized_tolerance": float(tolerance),
    }


def analyze_validation_geometry(
    project_root: Path,
    *,
    output: str = DEFAULT_OUTPUT,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Measure the validation-set geometry once and publish the report.

    **Manual, human-invoked analysis.** It opens a dataset partition, so no test,
    hook, CI job or repository check calls it. It trains nothing, reserves no
    attempt directory or ledger, writes no attempt-log entry, and mutates no
    existing attempt evidence.
    """
    project_root = Path(project_root)
    split = assert_split_allowed(ANALYSIS_PARTITION)
    output_path = _guarded_output(project_root, output)

    # Provenance first, and on the same terms an attempt gets: a geometry
    # measured from an uncommitted tree cannot be reproduced from its own report.
    digests = source_digests(project_root)
    committed = verify_committed_source(project_root, digests)
    repository = repository_identity(project_root)

    acceptance_path = project_root / ACCEPTANCE_CONFIG_PATH
    acceptance = load_toml(acceptance_path)
    validate_acceptance_config(acceptance)
    view = acceptance_view(acceptance)

    # The thresholds come from the recorded attempts, and a missing or malformed
    # reference attempt is a hard failure. Resolved here, before anything opens a
    # partition, so that failure costs nothing.
    log_path = project_root / ATTEMPT_LOG_PATH
    threshold_set = thresholds(log_path)

    protocol = load_toml(project_root / PROTOCOL_CONFIG_PATH)
    locked_identity = locked_dataset_identity(protocol)
    locked_dataset, locked_manifest = locked_dataset_paths(protocol)
    dataset = project_root / locked_dataset
    manifest_path = project_root / locked_manifest
    manifest = restricted_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
    dataset_digests = verify_dataset_identity(dataset, manifest_path, manifest, locked_identity)
    verify_partition_policy(dataset, manifest, split)
    columns = load_partition(dataset, manifest, split)

    populations = geometry_populations(
        columns, view["bins"], threshold_set, quantile_method=view["quantile_method"]
    )
    discrepancy = comparator_discrepancy(
        columns,
        view["bins"],
        quantile_method=view["quantile_method"],
        material_tolerance=view["material_normalized_tolerance"],
    )

    report = {
        "schema_version": GEOMETRY_SCHEMA,
        "task": "task-9h-american-pricer-development",
        "analysis": "validation-set geometry of the binding constraints",
        "status": {
            "confirmatory": False,
            "exploratory": True,
            "selection_bias": EXPLORATORY_LABEL,
            "trains_a_model": False,
            "reserves_an_attempt": False,
            "writes_attempt_evidence": False,
        },
        "partition_analyzed": split,
        "partitions_read_as_data": [split],
        "partitions_hashed_for_identity": list(ALLOWED_SPLITS),
        "row_level_policy_verified": [split],
        "final_partition_touched": False,
        "repository": repository,
        "source_digests": digests,
        "committed_source": committed,
        "dataset": {
            "manifest_sha256": dataset_digests["manifest"],
            "partition_sha256": {name: dataset_digests[name] for name in ALLOWED_SPLITS},
            "identity_locked_by": PROTOCOL_CONFIG_PATH,
        },
        "acceptance_reference": {
            "source": ACCEPTANCE_CONFIG_PATH,
            "sha256": sha256_file(acceptance_path),
            "keys_read": sorted(view),
            "note": (
                "read through a two-key whitelist for the slice bins and the quantile method; "
                "no threshold of the fixed criterion is revised, reinterpreted or restated here"
            ),
        },
        "definitions": {
            "normalization": "A = spot * exp(-dividend_yield * maturity)",
            **MEASURED_QUANTITIES,
            "comparator_discrepancy_normalized": COMPARATOR_DISCREPANCY,
            "slices": (
                "differentiable_pricing.ml.american_pilot._slice_masks, the repository's single "
                "definition of the validation slices, reused rather than reimplemented"
            ),
            "at_or_below": "count of rows whose value is <= the threshold",
        },
        "thresholds": threshold_set,
        "quantiles": [float(value) for value in QUANTILES],
        "quantile_method": view["quantile_method"],
        "column_consistency": column_consistency(columns),
        "populations": populations,
        "comparator_discrepancy": discrepancy,
        "limitations": {
            "selection_bias": EXPLORATORY_LABEL,
            "descriptive_only": (
                "this measures the geometry of the labels, not any model's behavior; it "
                "predicts no violation count and admits no candidate"
            ),
            "analytic_european_is_not_the_comparator": (
                "the european_comparator_lower_bound gate compares against the stored CRR "
                "European leg, not against the analytic Black-Scholes value a model can "
                "compute at inference; enforcing the analytic floor therefore enforces the "
                "analytic bound and makes no claim about that gate, which is exactly what the "
                "comparator discrepancy measures"
            ),
            "label_dependence": (
                "every quantity is a stored dataset column under Task 9E's conditional, "
                "mapping-only admission; the CRR European leg carries the lattice's own "
                "discretization error and is not a converged European truth"
            ),
            "final_partition": (
                "no final or held-out partition was opened, hashed, stat-ed, imported, counted "
                "or inspected; the geometry of any such partition is unknown and stays unknown"
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_json_atomic(output_path, report, overwrite=overwrite)
    except FileExistsError as error:
        raise GeometryError(
            f"'{output}' already exists; pass --overwrite to replace the exploratory geometry "
            "report, or choose another artifacts/ path"
        ) from error
    return report


def compact_geometry(report: Mapping[str, Any]) -> dict[str, Any]:
    """The small view of one geometry report: overall and positive-premium rows."""
    populations = report["populations"]
    return {
        "schema_version": report["schema_version"],
        "partition_analyzed": report["partition_analyzed"],
        "commit": report["repository"]["commit"],
        "rows": populations["overall"]["rows"],
        "thresholds": {entry["name"]: entry["value"] for entry in report["thresholds"]},
        "overall": {
            quantity: {
                "median": populations["overall"][quantity]["quantiles"]["0.5"],
                "fraction_at_or_below": populations["overall"][quantity][
                    "fraction_at_or_below"
                ],
            }
            for quantity in MEASURED_QUANTITIES
        },
        "premium_status:positive": {
            "rows": populations["premium_status:positive"]["rows"],
            "fraction_at_or_below": populations["premium_status:positive"][
                "european_slack_normalized"
            ]["fraction_at_or_below"],
        },
        "comparator_discrepancy": {
            "definition": report["comparator_discrepancy"]["definition"],
            "overall": {
                "fraction_above": report["comparator_discrepancy"]["populations"]["overall"][
                    "fraction_above"
                ],
                "quantiles": report["comparator_discrepancy"]["populations"]["overall"][
                    "quantiles"
                ],
            },
            "effectively_zero_premium": {
                "rows": report["comparator_discrepancy"]["populations"][
                    "effectively_zero_premium"
                ]["rows"],
                "fraction_above": report["comparator_discrepancy"]["populations"][
                    "effectively_zero_premium"
                ]["fraction_above"],
            },
        },
        "selection_bias": EXPLORATORY_LABEL,
    }


__all__ = [
    "ABSOLUTE_THRESHOLDS",
    "ANALYSIS_PARTITION",
    "COMPARATOR_DISCREPANCY",
    "DEFAULT_OUTPUT",
    "DISCREPANCY_QUANTILES",
    "DISCREPANCY_THRESHOLDS",
    "GEOMETRY_SCHEMA",
    "GeometryError",
    "acceptance_view",
    "analyze_validation_geometry",
    "attempt_rmse_thresholds",
    "column_consistency",
    "compact_geometry",
    "comparator_discrepancy",
    "comparator_discrepancy_values",
    "discounted_spot",
    "effectively_zero_premium_mask",
    "exceedance_statistics",
    "geometry_populations",
    "normalized_slacks",
    "population_masks",
    "slack_statistics",
    "thresholds",
]
