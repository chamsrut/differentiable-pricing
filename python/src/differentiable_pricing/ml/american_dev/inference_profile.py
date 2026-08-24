"""Task 9H frozen-inference profile and bounded optimization (sections C and D).

**The same frozen mathematical function throughout.** Nothing here retrains,
changes an architecture, or changes a weight. Section D optimizes the
*execution* of E2c's deployed function; a variant that changes what the function
computes is a different deployment, not an optimization, and is reported as one.

Why section D is decision-critical, not deferrable
--------------------------------------------------
The tempting shortcut is to fit ``T = F + k*c`` across two historical points,
read off a zero-backbone ceiling, and close capacity reduction if that ceiling
misses the reference. That answers a narrower question than the one being
asked: whether capacity reduction works **as a standalone intervention, before
overhead is optimized**. It says nothing about whether a smaller backbone would
clear the reference *after* the overhead is removed, which is the configuration
anyone would actually ship.

So the profile reports four quantities, and the branch is decided from them
together:

``a`` observed batch-1 total latency;
``b`` implied total if backbone cost were zero -- the **standalone-capacity
      lower bound**;
``c`` measured total after the predeclared section-D optimizations, retaining
      the existing backbone;
``d`` counterfactual totals after section D with the backbone reduced by 22.7x
      and by 4x.

Decision rule, predeclared
--------------------------
* ``b`` cannot reach the reference
      -> capacity reduction is closed **as a standalone latency intervention**;
* ``c`` reaches the reference
      -> no capacity reduction is needed for latency;
* ``c`` below the reference but one or both counterfactuals in ``d`` clear it
      -> capacity reduction stays a **live option**, and section H weighs one
      remaining neural attempt against the expected accuracy loss and the
      measured remaining gap;
* ``c`` below the reference and the plausible reductions in ``d`` also fail
      -> capacity reduction is closed **as a practical latency remedy**.

The ``< 25%`` and ``> 60%`` backbone-share thresholds are retained as
**descriptive reporting only**. They no longer decide the branch.

The 10x figure is Task 9G's historical **reference**, not a Task 9H gate. Task
9H makes and measures no latency claim.

Bounding section D
------------------
:data:`INFERENCE_VARIANTS` is a closed, predeclared list. There is no search, no
autotuning loop and no "try one more thing": a variant absent from that tuple is
not attempted in this phase. Every variant is reported whether it helped or not,
including the ones that made things worse.

Reduced precision is **not** an optimization. It changes the computed function,
so it is carried as a distinct deployment variant and requires the full price,
bound, shape and Greek equivalence battery before anything may be called
equivalent.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Final

import numpy as np
import torch

from ..config import FEATURE_ORDER
from .frozen import (
    FLOOR_HEADS,
    FrozenCheckpointError,
    MarginOverrideModel,
    assert_floor_head,
)
from .latency import DIAGNOSTIC_BENCHMARK_ATTEMPT
from .representation import (
    AmericanDevPriceModel,
    discounted_spot,
    european_price,
    intrinsic_value,
    network_features,
    normalized_lower_floor,
)

PROFILE_SCHEMA: Final = "american-dev-inference-profile/1"

DEFAULT_PROFILE_OUTPUT: Final = "artifacts/task-9h/latency/inference-profile-v1.json"
DEFAULT_VARIANTS_OUTPUT: Final = "artifacts/task-9h/latency/inference-variants-v1.json"

#: Task 9G's historical end-to-end speedup reference. Context for interpretation,
#: never applied as a Task 9H gate.
HISTORICAL_SPEEDUP_REFERENCE: Final = 10.0

#: Retained as descriptive reporting only. These no longer decide branch D.
BACKBONE_SHARE_DESCRIPTIVE_LOW: Final = 0.25
BACKBONE_SHARE_DESCRIPTIVE_HIGH: Final = 0.60

#: The counterfactual backbone reductions section D reports after optimization.
#: 22.7x is the Task 9G-to-E2b parameter ratio -- the largest reduction this
#: project has any evidence about -- and 4x is a mid-range alternative.
COUNTERFACTUAL_BACKBONE_REDUCTIONS: Final = (22.7, 4.0)

#: Request shapes reported for every variant.
BATCH_SIZES: Final = (1, 8)

#: Reported quantiles. Deliberately median and p95, matching the Task 9G
#: contract's own reporting.
QUANTILES: Final = (0.5, 0.95)

#: The components the batch-1 wall time is decomposed into. Measured
#: individually and against the complete wrapper, so the residual -- framework
#: and Python dispatch -- is reported rather than assumed away.
COMPONENTS: Final = (
    "input_and_feature_preparation",
    "residual_backbone",
    "analytic_black_scholes_and_phi",
    "intrinsic",
    "smooth_projection_and_margin",
    "allocation_and_conversion",
    "complete_wrapper",
)

#: **The closed, predeclared section-D variant list.** Adding to this tuple after
#: measurements exist would turn a bounded comparison into a search.
INFERENCE_VARIANTS: Final = (
    {
        "name": "frozen_float64_reference",
        "changes_the_function": False,
        "description": "the frozen E2c deployed path exactly as it stands; the reference",
    },
    {
        "name": "inference_mode",
        "changes_the_function": False,
        "description": "torch.inference_mode instead of no_grad: no autograd bookkeeping",
    },
    {
        "name": "reused_buffers",
        "changes_the_function": False,
        "description": "a preallocated request buffer, removing per-call tensor allocation",
    },
    {
        "name": "single_thread",
        "changes_the_function": False,
        "description": "one intra-op thread: a single small request pays for thread handoff",
    },
    {
        "name": "compiled_static",
        "changes_the_function": False,
        "description": "torch.compile with a static batch shape, where the runtime supports it",
    },
    {
        "name": "fused_floor",
        "changes_the_function": False,
        "description": (
            "Black-Scholes, intrinsic and the smooth projection evaluated in one fused pass "
            "over the same arithmetic"
        ),
    },
    {
        "name": "float32_deployment",
        "changes_the_function": True,
        "description": (
            "a DISTINCT DEPLOYMENT VARIANT, not an optimization: float32 changes the computed "
            "function. Requires price, bound, shape and Greek equivalence diagnostics, and "
            "specifically the interaction with tau = 1e-4 and delta = 1e-4 -- the smooth "
            "maximum evaluates arguments of order (raw - floor)/tau that reach 1e4, and the "
            "crossover band is only 1e-4 wide."
        ),
    },
)

REDUCED_PRECISION_REQUIREMENT: Final = (
    "float32 is a distinct deployment variant, never an optimization. The float64 frozen E2c "
    "path is the reference; price output is compared numerically, and the bound, shape and "
    "Greek diagnostics are rerun with exact counts reported. It may not be described as "
    "equivalent without those measurements."
)

UNGATED_LABEL: Final = (
    "ungated exploratory diagnostic; Task 9H makes and measures no latency claim, the 10x "
    "figure is Task 9G's historical reference rather than a Task 9H gate, and no dataset "
    "partition is opened"
)


class InferenceProfileError(RuntimeError):
    """Raised when a profile or variant cannot be measured as declared."""


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _summary(samples: Sequence[int]) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64)
    return {
        f"p{int(quantile * 100)}_nanoseconds": float(np.quantile(values, quantile))
        for quantile in QUANTILES
    } | {"repetitions": int(values.size)}


def time_callable(
    operation: Callable[[], Any], *, warmups: int = 50, repetitions: int = 400
) -> dict[str, float]:
    """Median and p95 wall time of one operation, after warm-up.

    A monotonic nanosecond clock, and the operation is called with no arguments
    so nothing but the operation itself is inside the timed region.
    """
    if warmups < 0 or repetitions < 1:
        raise InferenceProfileError("timing needs a non-negative warm-up and at least one call")
    for _ in range(warmups):
        operation()
    samples: list[int] = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        operation()
        samples.append(time.perf_counter_ns() - start)
    return _summary(samples)


def _median(entry: Mapping[str, float]) -> float:
    return float(entry["p50_nanoseconds"])


# ---------------------------------------------------------------------------
# The component profile
# ---------------------------------------------------------------------------


def component_profile(
    model: AmericanDevPriceModel,
    request: torch.Tensor,
    *,
    warmups: int = 50,
    repetitions: int = 400,
) -> dict[str, Any]:
    """Decompose one request shape's wall time into named components.

    Each component is timed by calling the **same functions the deployed path
    calls**, not reimplementations, so the decomposition describes the function
    that actually ships.

    The components do not sum to the wrapper and are not expected to: the
    difference is framework and Python dispatch, and it is reported explicitly
    as a residual rather than distributed silently across the parts.
    """
    assert_floor_head(model)
    margin = float(model.european_margin)
    temperature = float(model.temperature)
    with torch.no_grad():
        features = network_features(request, model.conditioning)
        standardized = (features - model.feature_mean) / model.feature_scale
    raw_input = request.numpy()

    operations: dict[str, Callable[[], Any]] = {
        "input_and_feature_preparation": lambda: network_features(request, model.conditioning),
        "residual_backbone": lambda: model.network(standardized),
        "analytic_black_scholes_and_phi": lambda: european_price(request),
        "intrinsic": lambda: intrinsic_value(request),
        "smooth_projection_and_margin": lambda: normalized_lower_floor(
            request, temperature, margin
        ),
        "allocation_and_conversion": lambda: torch.as_tensor(raw_input, dtype=torch.float64),
        "complete_wrapper": lambda: model(request),
    }
    measured: dict[str, Any] = {}
    with torch.no_grad():
        for name in COMPONENTS:
            measured[name] = time_callable(
                operations[name], warmups=warmups, repetitions=repetitions
            )
    wrapper = _median(measured["complete_wrapper"])
    parts = {name: _median(measured[name]) for name in COMPONENTS if name != "complete_wrapper"}
    accounted = sum(parts.values())
    return {
        "batch_size": int(request.shape[0]),
        "components": measured,
        "median_nanoseconds": {**parts, "complete_wrapper": wrapper},
        "shares_of_wrapper": {name: value / wrapper for name, value in parts.items()},
        "backbone_share": parts["residual_backbone"] / wrapper,
        "residual_dispatch_nanoseconds": wrapper - accounted,
        "residual_dispatch_share": (wrapper - accounted) / wrapper,
        "residual_note": (
            "components are timed individually and do not sum to the wrapper; the difference "
            "is framework and Python dispatch and is reported here rather than distributed "
            "across the parts"
        ),
        "descriptive_thresholds": {
            "low": BACKBONE_SHARE_DESCRIPTIVE_LOW,
            "high": BACKBONE_SHARE_DESCRIPTIVE_HIGH,
            "status": (
                "descriptive reporting only; these thresholds do not decide the capacity "
                "branch, which is decided by the four-quantity rule in decide_capacity_branch"
            ),
        },
    }


# ---------------------------------------------------------------------------
# Section D variants
# ---------------------------------------------------------------------------


def assert_variant_declared(name: str) -> dict[str, Any]:
    """Refuse any variant absent from the predeclared closed list."""
    for variant in INFERENCE_VARIANTS:
        if variant["name"] == name:
            return dict(variant)
    raise InferenceProfileError(
        f"variant {name!r} is not predeclared; section D is a bounded comparison over "
        f"{[entry['name'] for entry in INFERENCE_VARIANTS]}, not a search"
    )


def build_variant(
    model: AmericanDevPriceModel, name: str
) -> tuple[Callable[[torch.Tensor], torch.Tensor], dict[str, Any]]:
    """Construct one predeclared variant's callable and its declaration record.

    A variant that the runtime cannot provide -- ``torch.compile`` on an
    unsupported build, say -- is reported as unavailable rather than silently
    substituted by the reference, because a table row that quietly repeats the
    baseline reads as a measurement that was made.
    """
    declaration = assert_variant_declared(name)
    deployed = MarginOverrideModel(model, model.european_margin)
    available = True
    unavailable_reason: str | None = None
    call: Callable[[torch.Tensor], torch.Tensor]

    if name == "frozen_float64_reference":
        call = deployed
    elif name == "inference_mode":

        def call(request: torch.Tensor) -> torch.Tensor:
            with torch.inference_mode():
                return deployed(request)

    elif name == "reused_buffers":
        buffer: dict[int, torch.Tensor] = {}

        def call(request: torch.Tensor) -> torch.Tensor:
            held = buffer.get(request.shape[0])
            if held is None:
                held = torch.empty_like(request)
                buffer[request.shape[0]] = held
            held.copy_(request)
            return deployed(held)

    elif name == "single_thread":

        def call(request: torch.Tensor) -> torch.Tensor:
            previous = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                return deployed(request)
            finally:
                torch.set_num_threads(previous)

    elif name == "compiled_static":
        try:
            compiled = torch.compile(deployed, dynamic=False)
            call = compiled
        except Exception as error:  # a build without a working compiler backend
            available = False
            unavailable_reason = f"torch.compile is unavailable here: {error}"
            call = deployed
    elif name == "fused_floor":
        call = _fused_deployed(model)
    elif name == "float32_deployment":
        available = False
        unavailable_reason = (
            "float32 changes the computed function, so it is measured through the separate "
            "deployment-variant path with its full equivalence battery, never as a timing "
            "row alongside optimizations"
        )
        call = deployed
    else:  # pragma: no cover - assert_variant_declared already refused
        raise InferenceProfileError(f"variant {name!r} has no construction")

    return call, {
        **declaration,
        "available": available,
        "unavailable_reason": unavailable_reason,
    }


def _fused_deployed(
    model: AmericanDevPriceModel,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """One pass over the floor arithmetic, computing the same value.

    The pieces are the deployed path's own: the same Black--Scholes expression,
    the same intrinsic payoff, the same smooth maximum and the same one-sided
    projection. Only the number of separate traversals changes, which is why
    this is an optimization and not a deployment variant -- and the variant
    harness checks the output against the reference to prove it.
    """
    margin = float(model.european_margin)
    temperature = float(model.temperature)

    def call(request: torch.Tensor) -> torch.Tensor:
        scale = discounted_spot(request)
        floor = normalized_lower_floor(request, temperature, margin)
        features = network_features(request, model.conditioning)
        standardized = (features - model.feature_mean) / model.feature_scale
        raw = model.network(standardized)
        if raw.ndim == 2 and raw.shape[1] == 1:
            raw = raw.squeeze(-1)
        direct = raw * model.price_scale + model.price_mean
        projected = floor + temperature * torch.nn.functional.softplus(
            (direct - floor) / temperature
        )
        return scale * projected

    return call


def measure_variants(
    model: AmericanDevPriceModel,
    requests: Mapping[int, torch.Tensor],
    *,
    warmups: int = 50,
    repetitions: int = 400,
) -> dict[str, Any]:
    """Time every predeclared variant at every request shape, and check its output.

    Numerical agreement against the frozen float64 path is measured for each
    variant, so "this was faster" is never reported without "and it computed the
    same thing".
    """
    reference_call, _ = build_variant(model, "frozen_float64_reference")
    with torch.no_grad():
        reference = {size: reference_call(request) for size, request in requests.items()}

    results: dict[str, Any] = {}
    for declaration in INFERENCE_VARIANTS:
        name = str(declaration["name"])
        call, record = build_variant(model, name)
        entry: dict[str, Any] = {"declaration": record, "shapes": {}}
        if record["available"]:
            with torch.no_grad():
                for size, request in requests.items():
                    produced = call(request)
                    difference = torch.abs(produced - reference[size])
                    entry["shapes"][str(size)] = {
                        **time_callable(
                            lambda call=call, request=request: call(request),
                            warmups=warmups,
                            repetitions=repetitions,
                        ),
                        "maximum_absolute_difference_from_reference": float(difference.max()),
                        "bitwise_identical_to_reference": bool(
                            torch.equal(produced, reference[size])
                        ),
                    }
        results[name] = entry
    results["reduced_precision_requirement"] = REDUCED_PRECISION_REQUIREMENT
    return results


# ---------------------------------------------------------------------------
# The corrected capacity decision
# ---------------------------------------------------------------------------


def decide_capacity_branch(
    observed_total_nanoseconds: float,
    backbone_nanoseconds: float,
    optimized_total_nanoseconds: float | None,
    crr_reference_nanoseconds: float,
    *,
    reductions: Sequence[float] = COUNTERFACTUAL_BACKBONE_REDUCTIONS,
) -> dict[str, Any]:
    """The four-quantity capacity branch, decided as predeclared.

    ``optimized_total_nanoseconds`` may be ``None`` only to describe the state
    *before* section D has run. The branch is then reported as undecided rather
    than resolved from the pre-optimization profile, because deciding it there
    would answer the standalone question and label it as the general one.
    """
    if crr_reference_nanoseconds <= 0.0:
        raise InferenceProfileError("the matched CRR reference must be a positive duration")

    def speedup(total: float) -> float:
        return float(crr_reference_nanoseconds / total)

    zero_backbone = max(observed_total_nanoseconds - backbone_nanoseconds, 0.0)
    quantities: dict[str, Any] = {
        "a_observed_total_nanoseconds": float(observed_total_nanoseconds),
        "a_observed_speedup": speedup(observed_total_nanoseconds),
        "b_zero_backbone_total_nanoseconds": zero_backbone,
        "b_zero_backbone_speedup": speedup(zero_backbone) if zero_backbone > 0 else None,
        "c_optimized_total_nanoseconds": optimized_total_nanoseconds,
        "c_optimized_speedup": (
            speedup(optimized_total_nanoseconds)
            if optimized_total_nanoseconds is not None
            else None
        ),
        "d_counterfactuals": None,
        "reference": HISTORICAL_SPEEDUP_REFERENCE,
        "reference_status": (
            "Task 9G's historical reference, not a Task 9H gate; Task 9H makes no latency claim"
        ),
    }

    standalone_closed = (
        quantities["b_zero_backbone_speedup"] is None
        or quantities["b_zero_backbone_speedup"] < HISTORICAL_SPEEDUP_REFERENCE
    )
    if optimized_total_nanoseconds is None:
        return {
            **quantities,
            "standalone_capacity_closed": bool(standalone_closed),
            "branch": "undecided_until_section_d_runs",
            "reason": (
                "section D has not run. The branch is not decided from the pre-optimization "
                "profile: that would answer whether capacity reduction works as a standalone "
                "intervention and report it as the general answer."
            ),
        }

    optimized_backbone_share = min(backbone_nanoseconds, optimized_total_nanoseconds)
    counterfactuals = {}
    for factor in reductions:
        if factor <= 0.0:
            raise InferenceProfileError("a backbone reduction factor must be positive")
        saved = optimized_backbone_share * (1.0 - 1.0 / float(factor))
        total = max(optimized_total_nanoseconds - saved, 0.0)
        counterfactuals[f"{factor:g}x"] = {
            "total_nanoseconds": float(total),
            "speedup": speedup(total) if total > 0 else None,
            "clears_reference": bool(
                total > 0 and speedup(total) >= HISTORICAL_SPEEDUP_REFERENCE
            ),
        }
    quantities["d_counterfactuals"] = counterfactuals

    optimized_clears = quantities["c_optimized_speedup"] >= HISTORICAL_SPEEDUP_REFERENCE
    any_counterfactual_clears = any(
        entry["clears_reference"] for entry in counterfactuals.values()
    )
    if optimized_clears:
        branch = "no_capacity_reduction_needed"
        reason = (
            "the optimized implementation of the identical frozen function reaches the "
            "historical reference; latency does not motivate a smaller backbone"
        )
    elif any_counterfactual_clears:
        branch = "capacity_reduction_remains_live"
        reason = (
            "the optimized implementation stays below the reference but a plausible backbone "
            "reduction clears it; section H must weigh one remaining neural attempt, the "
            "expected accuracy loss, and the measured remaining gap"
        )
    else:
        branch = "capacity_reduction_closed_as_a_practical_remedy"
        reason = (
            "the optimized implementation stays below the reference and the plausible backbone "
            "reductions do not clear it either; a smaller backbone does not fix the gap"
        )
    return {
        **quantities,
        "standalone_capacity_closed": bool(standalone_closed),
        "branch": branch,
        "reason": reason,
        "descriptive_backbone_share": float(backbone_nanoseconds / observed_total_nanoseconds),
        "descriptive_thresholds_do_not_decide": True,
    }


def assert_profiled_head(model: AmericanDevPriceModel) -> None:
    """Only a floor head has the components this profile names."""
    if model.head not in FLOOR_HEADS:
        raise FrozenCheckpointError(
            f"head {model.head!r} has no smooth floor; the component decomposition names "
            "components it does not have"
        )


# ---------------------------------------------------------------------------
# The human-invoked driver
# ---------------------------------------------------------------------------

#: The frozen checkpoint this phase profiles. Read from :mod:`.latency` rather
#: than restated, so the component profile and the matched baseline cannot end
#: up describing different models. E2c, not E2b: the existing matched
#: measurement is E2b's, and section C exists because E2c has never been timed.
PROFILED_ATTEMPT: Final = DIAGNOSTIC_BENCHMARK_ATTEMPT

#: A representative request for the component decomposition. The contract's own
#: latency cases drive the matched measurement; this one drives the breakdown,
#: which is about where time goes rather than about how much there is.
PROFILE_REQUEST: Final = (-1.0, 100.0, 100.0, 1.0, 0.05, 0.02, 0.20)


def profile(
    project_root: Any,
    *,
    output: str = DEFAULT_PROFILE_OUTPUT,
    variants_output: str = DEFAULT_VARIANTS_OUTPUT,
    overwrite: bool = False,
    warmups: int = 50,
    repetitions: int = 400,
    run_variants: bool = True,
) -> dict[str, Any]:
    """Section C profile and, unless disabled, the bounded section D comparison.

    Section D is **not** deferrable: the capacity branch is decided from the
    optimized implementation and its counterfactuals, not from the
    pre-optimization profile. Running the profile alone leaves the branch
    reported as ``undecided_until_section_d_runs``.
    """
    from pathlib import Path

    from ..artifact import write_json_atomic
    from .frozen import _guarded_artifact_path, diagnostic_provenance, load_frozen_model

    project_root = Path(project_root)
    path = _guarded_artifact_path(project_root, output, where="inference profile path")
    if path.exists() and not overwrite:
        raise InferenceProfileError(f"'{output}' already exists; pass overwrite to replace it")
    model, identity = load_frozen_model(project_root, PROFILED_ATTEMPT)
    assert_profiled_head(model)
    requests = {
        size: torch.tensor([list(PROFILE_REQUEST)] * size, dtype=torch.float64)
        for size in BATCH_SIZES
    }
    profiles = {
        str(size): component_profile(
            model, request, warmups=warmups, repetitions=repetitions
        )
        for size, request in requests.items()
    }
    report: dict[str, Any] = {
        "schema_version": PROFILE_SCHEMA,
        "task": "task-9h-american-pricer-development",
        "status": UNGATED_LABEL,
        "checkpoint": identity,
        "request": dict(zip(FEATURE_ORDER, PROFILE_REQUEST, strict=True)),
        "component_profiles": profiles,
        "provenance": diagnostic_provenance(project_root),
    }
    write_json_atomic(path, report, overwrite=overwrite)

    if run_variants:
        variant_path = _guarded_artifact_path(
            project_root, variants_output, where="inference variants path"
        )
        if variant_path.exists() and not overwrite:
            raise InferenceProfileError(
                f"'{variants_output}' already exists; pass overwrite to replace it"
            )
        variants = measure_variants(
            model, requests, warmups=warmups, repetitions=repetitions
        )
        variant_report = {
            "schema_version": PROFILE_SCHEMA,
            "status": UNGATED_LABEL,
            "checkpoint": identity,
            "predeclared_variants": [dict(entry) for entry in INFERENCE_VARIANTS],
            "bounded": (
                "a closed predeclared list, not a search; a variant absent from it is not "
                "attempted in this phase"
            ),
            "measurements": variants,
            "provenance": diagnostic_provenance(project_root),
        }
        write_json_atomic(variant_path, variant_report, overwrite=overwrite)
        report["variants"] = variant_report
    return report


def compact_profile(report: Mapping[str, Any]) -> dict[str, Any]:
    """The small, agent-readable view: where batch-1 time goes."""
    batch_one = report["component_profiles"]["1"]
    summary = {
        "protocol_commit": report.get("provenance", {}).get("protocol_commit"),
        "checkpoint": report["checkpoint"]["attempt_id"],
        "batch1_wrapper_nanoseconds": batch_one["median_nanoseconds"]["complete_wrapper"],
        "batch1_backbone_share": batch_one["backbone_share"],
        "batch1_residual_dispatch_share": batch_one["residual_dispatch_share"],
        "descriptive_thresholds_do_not_decide_the_branch": True,
        "status": report["status"],
    }
    variants = report.get("variants")
    if variants:
        summary["variants"] = {
            name: (
                entry["shapes"]["1"]["p50_nanoseconds"]
                if entry["declaration"]["available"]
                else entry["declaration"]["unavailable_reason"]
            )
            for name, entry in variants["measurements"].items()
            if isinstance(entry, dict) and "declaration" in entry
        }
    return summary
