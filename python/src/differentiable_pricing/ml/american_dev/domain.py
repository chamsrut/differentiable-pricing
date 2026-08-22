"""Task 9H European CRR-versus-Black-Scholes domain characterization.

**Label-free.** It measures a difference between two *European* valuations of
the same contract state:

``(E_CRR - E_BS) / A``, with ``A = spot * exp(-dividend_yield * maturity)``

Both legs are computable from the seven physical contract inputs alone, so this
analysis needs **no American label and no dataset partition**. It opens none:
nothing here reads, hashes, stats, counts or imports ``train``, ``validation`` or
any other partition, and the module imports no partition machinery at all.

Why the quantity matters. Task 9H's ``european_comparator_lower_bound``
diagnostic compares a model's price against the dataset's **stored CRR**
European leg, while the deployed ``smooth_lower_floor`` head can only enforce
the **analytic** Black-Scholes value it can compute at inference. Where the CRR
leg sits above the analytic value by more than the material tolerance, a model
resting exactly on the analytic floor is still counted as violating the stored
comparator. That gap is the lattice's own discretization error, and this
analysis characterizes its supremum over the **declared domain** rather than
over the partition Task 9H selects against.

**The margin rule is predeclared here, in code, before the run.** It has no free
parameter left to tune once the measurement lands:

.. code-block:: text

    domain_supremum = maximum positive normalized CRR-minus-BS gap
    candidate       = ceil_to_1e-5(2 * domain_supremum)
    delta           = max(1e-4, candidate)

:data:`MARGIN_SAFETY_FACTOR`, :data:`MARGIN_QUANTUM` and :data:`MINIMUM_MARGIN`
are the three constants that rule is written from, and
:func:`derive_margin` is its single implementation.

**What this analysis does not do.** It derives a candidate margin and writes one
report beneath the ignored ``artifacts/`` tree. It **modifies no configuration
and no source file**, implements no attempt, reserves no attempt directory or
ledger, appends nothing to the attempt log, trains nothing, and admits nothing.
Applying the derived margin would be a separate, separately predeclared attempt.

**Honesty about independence.** The derivation is independent of ``validation``
in construction: the sample comes from the declared domain, and no sampling
location is derived from any partition. It is *not* independent of the project's
history — the validation-set supremum of the same quantity has already been
observed by the exploratory geometry analysis, and that cannot be un-seen. The
report records that fact rather than arguing it away.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

from differentiable_pricing import _core

from ..artifact import write_json_atomic
from ..config import FEATURE_ORDER
from .attempts import (
    ACCEPTANCE_CONFIG_PATH,
    PROTOCOL_CONFIG_PATH,
    assert_contained_relative_path,
    assert_path_allowed,
    load_toml,
    repository_identity,
    sha256_file,
    source_digests,
    validate_acceptance_config,
    verify_committed_source,
)
from .representation import european_price_array

DOMAIN_SCHEMA: Final = "american-dev-european-comparator-domain/1"

#: The configuration that declares the sampling domain and the label policy.
#: It is a Task 9G tracked input, so its digest is re-verified here against the
#: value the locked protocol pinned rather than trusted or typed in.
DATASET_CONFIG_PATH: Final = "configs/american_option_dataset_v1.toml"

#: The label policy whose European leg this analysis reproduces. A dataset
#: configuration declaring any other policy fails closed, because the CRR
#: comparator semantics below would then describe something the dataset does not
#: contain.
EXPECTED_LABEL_POLICY: Final = "american-crr-adjacent-average/1"

#: Where the report is written: beneath the ignored ``artifacts/`` tree, like
#: every other Task 9H output. This is a development measurement, not frozen
#: evidence, and it is never committed.
DEFAULT_OUTPUT: Final = "artifacts/task-9h/domain/european-comparator-domain-v1.json"

#: The six continuous coordinates the declared domain is stated in. ``strike``
#: is not one of them: the declared domain fixes ``log(spot/strike)``, and the
#: strike follows as ``spot * exp(-log_moneyness)``.
DOMAIN_COORDINATES: Final = (
    "spot",
    "log_moneyness",
    "maturity",
    "rate",
    "dividend_yield",
    "volatility",
)

#: Both option types are evaluated at every sampled coordinate, so the option
#: type costs no sampling dimension and neither type is under-covered.
OPTION_TYPES: Final = ("call", "put")

#: Points in the deterministic low-discrepancy sample, before the option type is
#: applied. Each one becomes two domain states, so the sample alone supplies
#: 262,144 states.
SAMPLE_POINTS: Final = 131_072

#: The first six primes, one per declared coordinate: a Halton sequence in six
#: dimensions. Halton is chosen over a scrambled Sobol' sequence because it is a
#: dozen deterministic lines with no direction-number table to get wrong, and at
#: six dimensions with bases this small its correlation defects are immaterial
#: for a supremum characterization.
HALTON_BASES: Final = (2, 3, 5, 7, 11, 13)

#: Leading Halton points are discarded. The first few points of every radical
#: inverse cluster near the origin of each coordinate simultaneously, which is a
#: correlated corner rather than a representative sample; the explicit corner and
#: boundary sets below cover the domain's extremes deliberately instead.
HALTON_SKIP: Final = 1_024

#: The structured sweep over the regions where a binomial lattice is least
#: accurate: the shortest and longest maturities, the lowest and highest
#: volatilities, a fine sweep of near-the-money log moneyness where the CRR
#: European error oscillates most strongly, and both extremes of the rate and of
#: the dividend yield. Values that are not domain endpoints are stated here;
#: endpoints are taken from the declared domain so the two cannot drift apart.
DIFFICULT_INTERIOR_MATURITIES: Final = (0.05, 0.25, 1.0)
DIFFICULT_INTERIOR_VOLATILITIES: Final = (0.10, 0.20, 0.40)
DIFFICULT_NEAR_MONEY_COUNT: Final = 21
DIFFICULT_NEAR_MONEY_HALF_WIDTH: Final = 0.02
DIFFICULT_INTERIOR_LOG_MONEYNESS: Final = (
    -0.5,
    -0.35,
    -0.2,
    -0.1,
    -0.05,
    0.05,
    0.1,
    0.2,
    0.35,
    0.5,
)

#: Reported quantiles of the normalized gap distribution.
QUANTILES: Final = (0.0, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0)

#: Exceedance thresholds, counted **strictly above** each value so the counts
#: match ``shape_diagnostics``, which counts ``value > tolerance``. ``1e-6`` is
#: the acceptance file's own material tolerance; ``1e-5`` is the margin rule's
#: rounding quantum; ``1e-4`` is the floor's predeclared temperature scale.
EXCEEDANCE_THRESHOLDS: Final = (0.0, 1.0e-6, 1.0e-5, 1.0e-4)

# --- the predeclared margin rule -------------------------------------------

#: Multiplier applied to the measured domain supremum before rounding.
MARGIN_SAFETY_FACTOR: Final = 2.0
#: The margin is rounded up to a whole multiple of this quantum.
MARGIN_QUANTUM: Final = 1.0e-5
#: The margin never falls below this, whatever the measurement returns.
MINIMUM_MARGIN: Final = 1.0e-4
#: The rule, written once, in the words the report carries.
MARGIN_RULE: Final = (
    "domain_supremum = maximum positive normalized CRR-minus-BS gap; "
    "candidate = ceil_to_1e-5(2 * domain_supremum); delta = max(1e-4, candidate)"
)

#: Rows priced per call into the batch pricing boundary. Chunking changes
#: throughput only: the engine prices independent requests in deterministic
#: input order, so the values do not depend on the chunk size or on the thread
#: count.
PRICING_CHUNK_ROWS: Final = 8_192
#: Default worker threads for the batch pricing boundary. Throughput only.
DEFAULT_THREAD_COUNT: Final = 4

EXPLORATORY_LABEL: Final = (
    "exploratory domain characterization; it derives a candidate margin from the "
    "declared domain and admits nothing, and no partition was opened"
)


class DomainAnalysisError(RuntimeError):
    """Raised when the domain characterization cannot proceed safely."""


# ---------------------------------------------------------------------------
# Output guard
# ---------------------------------------------------------------------------


def _guarded_output(project_root: Path, relative: str) -> Path:
    where = "domain characterization output path"
    assert_path_allowed(relative, where=where)
    assert_contained_relative_path(relative, where=where)
    if not str(relative).replace("\\", "/").startswith("artifacts/"):
        raise DomainAnalysisError(
            "the domain characterization report lives beneath the ignored artifacts/ tree; "
            "it is a development measurement, not frozen evidence"
        )
    return Path(project_root) / relative


# ---------------------------------------------------------------------------
# The declared domain
# ---------------------------------------------------------------------------


def locked_tracked_input_digest(protocol: Mapping[str, Any], relative: str) -> str:
    """The digest the locked Task 9G protocol pinned for one tracked input.

    Whitelisted by path, in the same spirit as
    :func:`..american_dev.workbench.locked_dataset_identity`: exactly one entry
    is resolved, and no other tracked input's digest is read into a variable or
    written to the report.
    """
    entries = protocol.get("tracked_inputs")
    if not isinstance(entries, Sequence):
        raise DomainAnalysisError(f"'{PROTOCOL_CONFIG_PATH}' declares no tracked inputs")
    for entry in entries:
        if isinstance(entry, Mapping) and entry.get("path") == relative:
            digest = entry.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                raise DomainAnalysisError(
                    f"'{PROTOCOL_CONFIG_PATH}' pins no usable digest for '{relative}'"
                )
            return digest
    raise DomainAnalysisError(f"'{PROTOCOL_CONFIG_PATH}' does not pin '{relative}'")


def declared_domain(dataset_config: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    """The declared input domain, read through a whitelist of its coordinates.

    The dataset configuration also declares per-partition row counts. That table
    is never resolved, read into a variable or written to the report: this
    function reads ``[domain]`` and nothing else.
    """
    section = dataset_config.get("domain")
    if not isinstance(section, Mapping):
        raise DomainAnalysisError(f"'{DATASET_CONFIG_PATH}' declares no [domain] table")
    bounds: dict[str, tuple[float, float]] = {}
    for name in DOMAIN_COORDINATES:
        interval = section.get(name)
        if (
            not isinstance(interval, Sequence)
            or isinstance(interval, str)
            or len(interval) != 2
        ):
            raise DomainAnalysisError(f"[domain].{name} must be a two-element interval")
        low, high = (float(value) for value in interval)
        if not math.isfinite(low) or not math.isfinite(high) or not high > low:
            raise DomainAnalysisError(f"[domain].{name} must be a finite increasing interval")
        bounds[name] = (low, high)
    if bounds["spot"][0] <= 0.0 or bounds["volatility"][0] <= 0.0 or bounds["maturity"][0] <= 0.0:
        raise DomainAnalysisError(
            "the declared domain must keep spot, maturity and volatility strictly positive"
        )
    return bounds


def label_policy(dataset_config: Mapping[str, Any]) -> tuple[str, int]:
    """The declared label policy name and its step count."""
    section = dataset_config.get("label")
    if not isinstance(section, Mapping):
        raise DomainAnalysisError(f"'{DATASET_CONFIG_PATH}' declares no [label] table")
    policy = section.get("policy")
    steps = section.get("steps")
    if policy != EXPECTED_LABEL_POLICY:
        raise DomainAnalysisError(
            f"'{DATASET_CONFIG_PATH}' declares label policy {policy!r}; this analysis "
            f"reproduces the European leg of {EXPECTED_LABEL_POLICY!r} and no other"
        )
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise DomainAnalysisError(f"'{DATASET_CONFIG_PATH}' declares no positive [label].steps")
    return str(policy), int(steps)


def assert_domain_consistent(
    bounds: Mapping[str, tuple[float, float]], acceptance: Mapping[str, Any]
) -> dict[str, list[float]]:
    """Cross-check the three coordinates the acceptance file also states.

    ``[diagnostics].spot_domain``, ``volatility_domain`` and
    ``log_moneyness_domain`` are the shape diagnostics' own bump domain. They are
    read here through a three-key whitelist and required to agree with the
    dataset configuration's declared domain, so a drift between the two files
    surfaces as a refusal rather than as a silently different sample.
    """
    diagnostics = acceptance.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise DomainAnalysisError(f"'{ACCEPTANCE_CONFIG_PATH}' declares no [diagnostics] table")
    checked: dict[str, list[float]] = {}
    for coordinate, key in (
        ("spot", "spot_domain"),
        ("volatility", "volatility_domain"),
        ("log_moneyness", "log_moneyness_domain"),
    ):
        stated = diagnostics.get(key)
        if (
            not isinstance(stated, Sequence)
            or isinstance(stated, str)
            or len(stated) != 2
        ):
            raise DomainAnalysisError(f"'{ACCEPTANCE_CONFIG_PATH}' declares no [diagnostics].{key}")
        pair = [float(value) for value in stated]
        if pair != [bounds[coordinate][0], bounds[coordinate][1]]:
            raise DomainAnalysisError(
                f"[diagnostics].{key} is {pair} but the declared domain states "
                f"{list(bounds[coordinate])}; the two must agree before a margin is derived"
            )
        checked[key] = pair
    return checked


# ---------------------------------------------------------------------------
# Deterministic low-discrepancy sample
# ---------------------------------------------------------------------------


def radical_inverse(indices: np.ndarray, base: int) -> np.ndarray:
    """Van der Corput radical inverse of every index, in ``base``.

    Digits are accumulated least-significant first, which is the ordinary
    definition and makes the result a pure function of the index and the base:
    the same array of indices always produces the same float64 values.
    """
    if base < 2:
        raise DomainAnalysisError("a radical-inverse base must be at least 2")
    remaining = np.asarray(indices, dtype=np.int64).copy()
    if bool((remaining < 0).any()):
        raise DomainAnalysisError("radical-inverse indices must be non-negative")
    result = np.zeros(remaining.shape, dtype=np.float64)
    factor = 1.0 / float(base)
    while bool((remaining > 0).any()):
        result += (remaining % base).astype(np.float64) * factor
        remaining //= base
        factor /= float(base)
    return result


def halton_points(
    count: int = SAMPLE_POINTS,
    bases: Sequence[int] = HALTON_BASES,
    skip: int = HALTON_SKIP,
) -> np.ndarray:
    """``count`` deterministic Halton points in ``[0, 1)^len(bases)``."""
    if count < 1:
        raise DomainAnalysisError("the low-discrepancy sample must be non-empty")
    indices = np.arange(skip, skip + int(count), dtype=np.int64)
    columns = [radical_inverse(indices, int(base)) for base in bases]
    points = np.column_stack(columns)
    if not bool(np.isfinite(points).all()) or bool(((points < 0.0) | (points >= 1.0)).any()):
        raise DomainAnalysisError("the low-discrepancy sample left the unit cube")
    return np.ascontiguousarray(points)


def scale_to_domain(
    unit: np.ndarray, bounds: Mapping[str, tuple[float, float]]
) -> dict[str, np.ndarray]:
    """Map unit-cube points onto the declared domain, coordinate by coordinate."""
    unit = np.asarray(unit, dtype=np.float64)
    if unit.ndim != 2 or unit.shape[1] != len(DOMAIN_COORDINATES):
        raise DomainAnalysisError(
            f"unit points must be [rows, {len(DOMAIN_COORDINATES)}] in the order "
            f"{list(DOMAIN_COORDINATES)}"
        )
    return {
        name: bounds[name][0] + unit[:, index] * (bounds[name][1] - bounds[name][0])
        for index, name in enumerate(DOMAIN_COORDINATES)
    }


def corner_coordinates(bounds: Mapping[str, tuple[float, float]]) -> dict[str, np.ndarray]:
    """Every vertex of the declared box: ``2 ** 6`` points."""
    dimension = len(DOMAIN_COORDINATES)
    pattern = np.arange(1 << dimension, dtype=np.int64)
    columns = {}
    for index, name in enumerate(DOMAIN_COORDINATES):
        upper = (pattern >> index) & 1
        low, high = bounds[name]
        columns[name] = np.where(upper == 1, high, low).astype(np.float64)
    return columns


def face_coordinates(bounds: Mapping[str, tuple[float, float]]) -> dict[str, np.ndarray]:
    """One point on each face: a coordinate pinned to an endpoint, the rest centred."""
    centre = {name: 0.5 * (low + high) for name, (low, high) in bounds.items()}
    rows: list[dict[str, float]] = []
    for name in DOMAIN_COORDINATES:
        for endpoint in bounds[name]:
            row = dict(centre)
            row[name] = float(endpoint)
            rows.append(row)
    return {
        name: np.asarray([row[name] for row in rows], dtype=np.float64)
        for name in DOMAIN_COORDINATES
    }


def difficult_coordinates(bounds: Mapping[str, tuple[float, float]]) -> dict[str, np.ndarray]:
    """The structured sweep of the numerically hardest regions of the domain.

    A binomial lattice's European error is largest where the terminal
    distribution is resolved worst relative to the strike: the shortest
    maturities and the lowest volatilities, and near the money, where the error
    oscillates with the position of the strike between adjacent terminal nodes.
    The sweep therefore takes both maturity endpoints, both volatility endpoints,
    both rate endpoints, both dividend-yield endpoints and both spot endpoints —
    the spot pair also probes the exact scale invariance of the normalized gap —
    crossed with a fine near-the-money log-moneyness sweep and both moneyness
    endpoints.
    """
    maturities = (
        bounds["maturity"][0],
        *DIFFICULT_INTERIOR_MATURITIES,
        bounds["maturity"][1],
    )
    volatilities = (
        bounds["volatility"][0],
        *DIFFICULT_INTERIOR_VOLATILITIES,
        bounds["volatility"][1],
    )
    near_money = np.linspace(
        -DIFFICULT_NEAR_MONEY_HALF_WIDTH,
        DIFFICULT_NEAR_MONEY_HALF_WIDTH,
        DIFFICULT_NEAR_MONEY_COUNT,
    )
    moneyness = (
        bounds["log_moneyness"][0],
        *DIFFICULT_INTERIOR_LOG_MONEYNESS,
        *(float(value) for value in near_money),
        bounds["log_moneyness"][1],
    )
    axes = {
        "spot": bounds["spot"],
        "log_moneyness": tuple(moneyness),
        "maturity": tuple(maturities),
        "rate": bounds["rate"],
        "dividend_yield": bounds["dividend_yield"],
        "volatility": tuple(volatilities),
    }
    for name, values in axes.items():
        low, high = bounds[name]
        if any(not low <= float(value) <= high for value in values):
            raise DomainAnalysisError(f"the difficult sweep left the declared domain in {name!r}")
    grids = np.meshgrid(
        *(np.asarray(axes[name], dtype=np.float64) for name in DOMAIN_COORDINATES),
        indexing="ij",
    )
    return {
        name: np.ascontiguousarray(grid.reshape(-1))
        for name, grid in zip(DOMAIN_COORDINATES, grids, strict=True)
    }


def domain_states(
    bounds: Mapping[str, tuple[float, float]],
    *,
    sample_points: int = SAMPLE_POINTS,
) -> dict[str, Any]:
    """Every evaluated contract state, its construction label and its inputs.

    Four deterministic constructions are concatenated in a fixed order — the
    low-discrepancy sample, the structured difficult sweep, every box corner and
    one point per face — and each is then evaluated at **both** option types.
    """
    parts = {
        "low_discrepancy": scale_to_domain(halton_points(sample_points), bounds),
        "difficult_regions": difficult_coordinates(bounds),
        "corners": corner_coordinates(bounds),
        "faces": face_coordinates(bounds),
    }
    sources: list[str] = []
    columns: dict[str, list[np.ndarray]] = {name: [] for name in DOMAIN_COORDINATES}
    option_types: list[np.ndarray] = []
    counts: dict[str, int] = {}
    for source, coordinates in parts.items():
        rows = int(coordinates["spot"].size)
        counts[source] = rows * len(OPTION_TYPES)
        for option_type in OPTION_TYPES:
            for name in DOMAIN_COORDINATES:
                columns[name].append(coordinates[name])
            option_types.append(np.full(rows, option_type, dtype=object))
            sources.extend([source] * rows)
    stacked = {name: np.concatenate(values) for name, values in columns.items()}
    strike = stacked["spot"] * np.exp(-stacked["log_moneyness"])
    physical = np.ascontiguousarray(
        np.column_stack(
            [
                np.where(np.concatenate(option_types) == "call", 1.0, -1.0),
                stacked["spot"],
                strike,
                stacked["maturity"],
                stacked["rate"],
                stacked["dividend_yield"],
                stacked["volatility"],
            ]
        )
    )
    if physical.shape[1] != len(FEATURE_ORDER):
        raise DomainAnalysisError("the constructed states do not match the physical feature order")
    return {
        "physical": physical,
        "option_type": np.concatenate(option_types),
        "source": np.asarray(sources, dtype=object),
        "counts": counts,
        "total": int(physical.shape[0]),
    }


# ---------------------------------------------------------------------------
# The two European legs
# ---------------------------------------------------------------------------


def european_crr_adjacent_average(
    physical: np.ndarray,
    option_type: np.ndarray,
    steps: int,
    *,
    thread_count: int = DEFAULT_THREAD_COUNT,
    chunk_rows: int = PRICING_CHUNK_ROWS,
) -> np.ndarray:
    """``0.5 * (E_CRR(N) + E_CRR(N+1))``: the stored comparator's own semantics.

    The dataset's ``european_crr_price`` column is the European CRR price formed
    the same way, on the same lattice, as the American label: the average of the
    ``N``-step and ``(N+1)``-step prices at ``N = [label].steps``. This function
    reproduces exactly that, through the compiled engine's batch boundary — the
    recursion is never reimplemented in Python.

    ``thread_count`` and ``chunk_rows`` change throughput only. The engine prices
    independent requests in deterministic input order, so neither changes a value.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise DomainAnalysisError("the CRR depth must be a positive integer")
    rows = int(physical.shape[0])
    prices = np.empty(rows, dtype=np.float64)
    for start in range(0, rows, chunk_rows):
        stop = min(start + chunk_rows, rows)
        width = stop - start
        types = [str(value) for value in option_type[start:stop]]
        styles = ["european"] * width
        arguments = (
            types,
            styles,
            physical[start:stop, 1].tolist(),
            physical[start:stop, 2].tolist(),
            physical[start:stop, 3].tolist(),
            physical[start:stop, 4].tolist(),
            physical[start:stop, 5].tolist(),
            physical[start:stop, 6].tolist(),
        )
        at_n = np.asarray(
            _core.crr_price_batch(*arguments, [steps] * width, thread_count)["price"],
            dtype=np.float64,
        )
        at_n1 = np.asarray(
            _core.crr_price_batch(*arguments, [steps + 1] * width, thread_count)["price"],
            dtype=np.float64,
        )
        prices[start:stop] = 0.5 * (at_n + at_n1)
    if not bool(np.isfinite(prices).all()):
        raise DomainAnalysisError("the European CRR comparator returned a non-finite price")
    return prices


def normalized_comparator_gap(
    physical: np.ndarray, crr: np.ndarray, analytic: np.ndarray
) -> np.ndarray:
    """``(E_CRR - E_BS) / A`` with ``A = spot * exp(-dividend_yield * maturity)``.

    ``A`` is spelled exactly as the Task 9G/9H normalized target's reconstruction
    scale, so the gap is in the units the criterion and the floor are stated in.
    """
    scale = physical[:, 1] * np.exp(-physical[:, 5] * physical[:, 3])
    if bool((scale <= 0.0).any()) or not bool(np.isfinite(scale).all()):
        raise DomainAnalysisError("the discounted spot A = S*exp(-q*T) is outside its domain")
    gap = (np.asarray(crr, dtype=np.float64) - np.asarray(analytic, dtype=np.float64)) / scale
    if not bool(np.isfinite(gap).all()):
        raise DomainAnalysisError("the normalized comparator gap is not finite")
    return gap


# ---------------------------------------------------------------------------
# Statistics and the predeclared margin
# ---------------------------------------------------------------------------


def gap_statistics(values: np.ndarray, *, quantile_method: str) -> dict[str, Any]:
    """Quantiles, extremes and strictly-above exceedance counts of one gap vector."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise DomainAnalysisError("a gap vector must be one-dimensional and non-empty")
    rows = int(array.size)
    counts = {
        f"above_{threshold:g}": int((array > float(threshold)).sum())
        for threshold in EXCEEDANCE_THRESHOLDS
    }
    return {
        "rows": rows,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std(ddof=0)),
        "quantiles": {
            f"{quantile:g}": float(np.quantile(array, quantile, method=quantile_method))
            for quantile in QUANTILES
        },
        "count_above": counts,
        "fraction_above": {name: count / rows for name, count in counts.items()},
    }


def ceil_to_quantum(value: float, quantum: float = MARGIN_QUANTUM) -> float:
    """Round ``value`` up to a whole multiple of ``quantum``; zero stays zero."""
    if not math.isfinite(value) or not math.isfinite(quantum) or quantum <= 0.0:
        raise DomainAnalysisError("the margin quantum must be finite and positive")
    if value <= 0.0:
        return 0.0
    return math.ceil(value / quantum) * quantum


def derive_margin(maximum_gap: float) -> dict[str, Any]:
    """Apply the predeclared margin rule. **The single implementation of it.**

    ``domain_supremum`` is the maximum *positive* gap, so a domain over which the
    CRR leg never exceeds the analytic value yields a supremum of zero and the
    rule falls back to its declared minimum rather than to no margin at all.
    """
    if not math.isfinite(maximum_gap):
        raise DomainAnalysisError("the measured maximum gap is not finite")
    supremum = max(0.0, float(maximum_gap))
    candidate = ceil_to_quantum(MARGIN_SAFETY_FACTOR * supremum, MARGIN_QUANTUM)
    delta = max(MINIMUM_MARGIN, candidate)
    return {
        "rule": MARGIN_RULE,
        "predeclared_safety_factor": MARGIN_SAFETY_FACTOR,
        "quantum": MARGIN_QUANTUM,
        "minimum": MINIMUM_MARGIN,
        "measured_maximum_gap": float(maximum_gap),
        "domain_supremum": supremum,
        "candidate": candidate,
        "delta": delta,
        "binding_term": "candidate" if candidate >= MINIMUM_MARGIN else "minimum",
        "realized_safety_factor": (delta / supremum) if supremum > 0.0 else None,
    }


# ---------------------------------------------------------------------------
# The analysis
# ---------------------------------------------------------------------------


def characterize_european_comparator(
    project_root: Path,
    *,
    output: str = DEFAULT_OUTPUT,
    overwrite: bool = False,
    thread_count: int = DEFAULT_THREAD_COUNT,
    sample_points: int = SAMPLE_POINTS,
) -> dict[str, Any]:
    """Measure the gap over the declared domain once and publish the report.

    **Manual, human-invoked analysis.** It prices hundreds of thousands of
    lattices, so no test, hook, CI job or repository check calls it. It opens no
    dataset partition, uses no American label, trains nothing, reserves no
    attempt directory or ledger, appends nothing to the attempt log, and
    **modifies no configuration or source file**: it derives a candidate margin
    and writes one report beneath the ignored ``artifacts/`` tree.
    """
    project_root = Path(project_root)
    output_path = _guarded_output(project_root, output)
    if isinstance(thread_count, bool) or not isinstance(thread_count, int) or thread_count < 1:
        raise DomainAnalysisError("the worker thread count must be a positive integer")

    # Provenance first, and on the same terms an attempt gets: a margin derived
    # from an uncommitted tree cannot be reproduced from its own report.
    digests = source_digests(project_root)
    committed = verify_committed_source(project_root, digests)
    repository = repository_identity(project_root)

    acceptance_path = project_root / ACCEPTANCE_CONFIG_PATH
    acceptance = load_toml(acceptance_path)
    validate_acceptance_config(acceptance)
    quantile_method = acceptance.get("quantile_method")
    if not isinstance(quantile_method, str) or not quantile_method:
        raise DomainAnalysisError(f"'{ACCEPTANCE_CONFIG_PATH}' declares no quantile_method")

    protocol = load_toml(project_root / PROTOCOL_CONFIG_PATH)
    locked_digest = locked_tracked_input_digest(protocol, DATASET_CONFIG_PATH)
    dataset_config_path = project_root / DATASET_CONFIG_PATH
    dataset_digest = sha256_file(dataset_config_path)
    if dataset_digest != locked_digest:
        raise DomainAnalysisError(
            f"'{DATASET_CONFIG_PATH}' hashes to {dataset_digest[:12]}… but the locked protocol "
            f"pins {locked_digest[:12]}…; the declared domain is not the one Task 9G locked"
        )
    dataset_config = load_toml(dataset_config_path)
    bounds = declared_domain(dataset_config)
    policy, steps = label_policy(dataset_config)
    cross_checked = assert_domain_consistent(bounds, acceptance)

    states = domain_states(bounds, sample_points=sample_points)
    physical = states["physical"]
    analytic = european_price_array(physical)
    crr = european_crr_adjacent_average(
        physical, states["option_type"], steps, thread_count=thread_count
    )
    gap = normalized_comparator_gap(physical, crr, analytic)

    overall = gap_statistics(gap, quantile_method=quantile_method)
    by_source = {
        source: gap_statistics(gap[states["source"] == source], quantile_method=quantile_method)
        for source in states["counts"]
    }
    by_option_type = {
        option_type: gap_statistics(
            gap[states["option_type"] == option_type], quantile_method=quantile_method
        )
        for option_type in OPTION_TYPES
    }
    margin = derive_margin(overall["maximum"])
    argmax = int(np.argmax(gap))

    report = {
        "schema_version": DOMAIN_SCHEMA,
        "task": "task-9h-american-pricer-development",
        "analysis": "European CRR-versus-Black-Scholes discrepancy over the declared domain",
        "status": {
            "confirmatory": False,
            "exploratory": True,
            "selection_bias": EXPLORATORY_LABEL,
            "uses_american_labels": False,
            "opens_a_dataset_partition": False,
            "trains_a_model": False,
            "reserves_an_attempt": False,
            "writes_attempt_evidence": False,
            "modifies_configuration_or_source": False,
        },
        "partitions_opened": [],
        "repository": repository,
        "source_digests": digests,
        "committed_source": committed,
        "declared_domain": {
            "source": DATASET_CONFIG_PATH,
            "sha256": dataset_digest,
            "identity_locked_by": PROTOCOL_CONFIG_PATH,
            "section": "domain",
            "coordinates": {name: list(value) for name, value in bounds.items()},
            "derived_strike": "strike = spot * exp(-log_moneyness)",
            "cross_checked_against_acceptance": cross_checked,
        },
        "crr_comparator_semantics": {
            "label_policy": policy,
            "label_steps": steps,
            "depths": [steps, steps + 1],
            "exercise_style": "european",
            "operation": "0.5 * (E_CRR(N) + E_CRR(N+1))",
            "engine": "dp::crr_binomial through the compiled batch boundary",
            "note": (
                "the dataset's stored european_crr_price is formed this way, on the same "
                "lattice as the American label; this analysis reproduces it and reads no "
                "stored column"
            ),
        },
        "analytic_semantics": {
            "function": (
                "differentiable_pricing.ml.american_dev.representation.european_price_array"
            ),
            "note": (
                "the same analytic continuous-yield Black-Scholes function the "
                "smooth_lower_floor head enforces at inference, so the measurement and the "
                "enforcement cannot drift apart"
            ),
        },
        "sample": {
            "construction": [
                "low_discrepancy: Halton points in the six declared coordinates",
                "difficult_regions: structured sweep of the hardest lattice regions",
                "corners: every vertex of the declared box",
                "faces: one centred point per box face",
            ],
            "low_discrepancy_points": int(sample_points),
            "halton_bases": list(HALTON_BASES),
            "halton_skip": HALTON_SKIP,
            "coordinate_order": list(DOMAIN_COORDINATES),
            "option_types": list(OPTION_TYPES),
            "states_by_construction": dict(states["counts"]),
            "total_states": states["total"],
            "derived_from_a_partition": False,
        },
        "measurement": {
            "definition": "(european_crr_adjacent_average - european_black_scholes) / A",
            "normalization": "A = spot * exp(-dividend_yield * maturity)",
            "sign_convention": (
                "positive means the CRR comparator sits ABOVE the analytic value, which is the "
                "direction that leaves an analytic floor short of the stored comparator"
            ),
            "exceedance_convention": "counted strictly above each threshold",
            "quantile_method": quantile_method,
            "overall": overall,
            "by_construction": by_source,
            "by_option_type": by_option_type,
            "maximum_at": {
                "construction": str(states["source"][argmax]),
                "option_type": str(states["option_type"][argmax]),
                **{
                    name: float(physical[argmax, index])
                    for index, name in enumerate(FEATURE_ORDER)
                    if name != "option_type"
                },
            },
        },
        "margin": margin,
        "limitations": {
            "candidate_only": (
                "this derives a candidate margin and nothing else; it modifies no "
                "configuration or source file, implements no attempt and admits nothing"
            ),
            "prior_exposure": (
                "the validation-set supremum of this same quantity was already observed by the "
                "exploratory geometry analysis and cannot be un-seen; the rule above is "
                "predeclared in code and has no free parameter left, and the derivation uses no "
                "partition row and no sampling location derived from one"
            ),
            "sampling_is_not_a_proof": (
                "a finite deterministic sample bounds the gap where it looks; it is not a proof "
                "of a supremum over the continuum, which is why the rule applies a predeclared "
                "safety factor and a floor rather than the measured maximum itself"
            ),
            "declared_domain_is_a_box": (
                "the declared domain is the bounding box of the sampling strata, so it is a "
                "superset of the region the dataset actually populates; a supremum over it is "
                "conservative for the dataset and says nothing about states outside it"
            ),
            "crr_is_not_european_truth": (
                "both legs are model values; the CRR leg carries the lattice's own "
                "discretization error and the analytic leg is exact only for the "
                "continuous-yield Black-Scholes model, so the gap is a discretization "
                "measurement, not a pricing error"
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_json_atomic(output_path, report, overwrite=overwrite)
    except FileExistsError as error:
        raise DomainAnalysisError(
            f"'{output}' already exists; pass --overwrite to replace the exploratory domain "
            "report, or choose another artifacts/ path"
        ) from error
    return report


def compact_domain(report: Mapping[str, Any]) -> dict[str, Any]:
    """The small view of one domain report: the supremum and the derived margin."""
    measurement = report["measurement"]
    return {
        "schema_version": report["schema_version"],
        "commit": report["repository"]["commit"],
        "declared_domain": report["declared_domain"]["coordinates"],
        "crr_depths": report["crr_comparator_semantics"]["depths"],
        "total_states": report["sample"]["total_states"],
        "states_by_construction": report["sample"]["states_by_construction"],
        "definition": measurement["definition"],
        "maximum": measurement["overall"]["maximum"],
        "quantiles": measurement["overall"]["quantiles"],
        "fraction_above": measurement["overall"]["fraction_above"],
        "maximum_at": measurement["maximum_at"],
        "margin": report["margin"],
        "selection_bias": report["status"]["selection_bias"],
    }


__all__ = [
    "DATASET_CONFIG_PATH",
    "DEFAULT_OUTPUT",
    "DEFAULT_THREAD_COUNT",
    "DOMAIN_COORDINATES",
    "DOMAIN_SCHEMA",
    "EXCEEDANCE_THRESHOLDS",
    "EXPECTED_LABEL_POLICY",
    "HALTON_BASES",
    "HALTON_SKIP",
    "MARGIN_QUANTUM",
    "MARGIN_RULE",
    "MARGIN_SAFETY_FACTOR",
    "MINIMUM_MARGIN",
    "OPTION_TYPES",
    "QUANTILES",
    "SAMPLE_POINTS",
    "DomainAnalysisError",
    "assert_domain_consistent",
    "ceil_to_quantum",
    "characterize_european_comparator",
    "compact_domain",
    "corner_coordinates",
    "declared_domain",
    "derive_margin",
    "difficult_coordinates",
    "domain_states",
    "european_crr_adjacent_average",
    "face_coordinates",
    "gap_statistics",
    "halton_points",
    "label_policy",
    "locked_tracked_input_digest",
    "normalized_comparator_gap",
    "radical_inverse",
    "scale_to_domain",
]
