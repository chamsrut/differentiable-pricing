"""Task 9H matched latency diagnostic for one surviving development checkpoint.

**Ungated exploratory diagnostic.** Task 9H's scope is price only; it makes and
measures no latency claim. This module measures one number that the task's own
non-claims exclude from every verdict, because the decision about what to build
next depends on it: how long the deployed neural path takes against the matched
adjacent-average CRR comparator it is meant to replace. **No gate is applied.**
Task 9G's latency bar is recorded in the report as context and is explicitly
**not** evaluated here — a Task 9H measurement cannot pass or fail a Task 9G
gate, and this one is not offered as evidence for or against any hypothesis.

**No validation exposure.** It opens no dataset partition. The requests are the
fixed synthetic latency cases the Task 9G contract already declares, and the
model comes from an ignored checkpoint a completed attempt already wrote.

**The contract is Task 9G's, reused rather than restated.** The measurement is
performed by :func:`..american_pilot.run_latency` itself — the same
implementation that produced Task 9G's latency evidence — driven by
``configs/american_neural_pilot_latency_cases_v1.toml``, whose digest is
re-verified against the value the locked Task 9G protocol pinned. The clock, the
warm-up count, the repetition count, the deterministic cyclic measurement
rotation, the per-shape thread budgets, the inter-op thread budget, the timed
neural region (feature construction, standardization, inference, inverse target
transform and physical reconstruction) and the CRR operation
(``0.5 * (CRR(N) + CRR(N+1))``, priced through the compiled batch boundary) are
all the configuration's and the reused implementation's.

Two deviations from the Task 9G run exist and are recorded in the report rather
than glossed:

* **the CRR depth ladder is restricted to the matched depth** :data:`MATCHED_CRR_DEPTH`,
  the label policy's own resolution and the depth Task 9G's acceptance file
  names for interpretation. Every other depth of the Task 9G ladder is simply
  not measured; no semantic changes;
* **one model is timed instead of two arms**, so the deterministic rotation
  alternates between two operations rather than three. That is a consequence of
  benchmarking a single checkpoint, and it means the interleaving is not
  byte-identical to Task 9G's.

**Artifact loading is excluded from the timed region**, exactly as Task 9G
excludes it: the model is constructed and its weights are loaded before
:func:`..american_pilot.run_latency` is entered.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from ..american_pilot import run_latency
from ..artifact import write_json_atomic
from ..model import Scaling
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
from .domain import locked_tracked_input_digest
from .frozen import FROZEN_ATTEMPTS, FrozenCheckpointError, load_frozen_model
from .frozen import recorded_scaling as frozen_recorded_scaling
from .representation import AmericanDevPriceModel

LATENCY_SCHEMA: Final = "american-dev-matched-latency/1"

#: The Task 9G latency contract, reused verbatim. Its digest is re-verified
#: against the value the locked Task 9G protocol pinned, so the cases, the
#: clock, the warm-ups, the repetitions and the thread budgets cannot drift.
LATENCY_CONFIG_PATH: Final = "configs/american_neural_pilot_latency_cases_v1.toml"

#: The one CRR depth this diagnostic measures: the label policy's own resolution,
#: priced as the adjacent average of ``N`` and ``N + 1``. It must appear in the
#: contract's own depth ladder, and it is required to be the depth the acceptance
#: file names for interpretation.
MATCHED_CRR_DEPTH: Final = 1024

#: The **historical** benchmarked attempt. E2b was the first checkpoint timed on
#: this contract, and its result is the number the project has quoted since. It
#: stays the default of :func:`load_benchmarked_model` so nothing that already
#: reads E2b's measurement changes meaning.
#:
#: It is deliberately **not** the default of the command line. The diagnostic
#: phase measures E2c, and a benchmark whose subject depends on which argument
#: was omitted is exactly the failure this registry exists to prevent.
BENCHMARKED_ATTEMPT: Final = "scratch_residual_smooth_floor_raw_loss_v1"
DEFAULT_ATTEMPT_DIRECTORY: Final = (
    "artifacts/task-9h/scratch_residual_smooth_floor_raw_loss_v1"
)
ATTEMPT_REPORT_NAME: Final = "attempt-report.json"
CHECKPOINT_NAME: Final = "checkpoint.pt"

#: E2b's historical artifact. **Never reused for another attempt**: overwriting
#: or reinterpreting it would destroy the only record of the measurement the
#: project has been quoting.
DEFAULT_OUTPUT: Final = "artifacts/task-9h/latency/matched-latency-v1.json"

#: Every attempt this diagnostic may benchmark, and where each one's result
#: lives. Each entry's directory and output are **derived from the attempt**, so
#: selecting an attempt cannot leave a path pointing at a different one.
#:
#: The keys are required to be a subset of the frozen registry, checked at
#: import: a checkpoint that is not frozen has no immutable identity to record
#: alongside a timing, and timing an unidentified model produces a number
#: nothing can be said about.
BENCHMARK_TARGETS: Final = {
    "scratch_residual_smooth_floor_raw_loss_v1": {
        "label": "E2b",
        "directory": "artifacts/task-9h/scratch_residual_smooth_floor_raw_loss_v1",
        "output": DEFAULT_OUTPUT,
        "historical": True,
    },
    "scratch_residual_smooth_floor_margin_v1": {
        "label": "E2c",
        "directory": "artifacts/task-9h/scratch_residual_smooth_floor_margin_v1",
        "output": "artifacts/task-9h/latency/matched-latency-e2c-v1.json",
        "historical": False,
    },
}

#: The attempt the diagnostic phase's official matched baseline measures. The
#: existing measurement is E2b's; section C exists because E2c has never been
#: timed.
DIAGNOSTIC_BENCHMARK_ATTEMPT: Final = "scratch_residual_smooth_floor_margin_v1"

#: Reported latency quantiles. With the contract's seven repetitions the p95 is
#: an order statistic of a very small sample and is reported as such.
LATENCY_QUANTILES: Final = (0.5, 0.95)

UNGATED_LABEL: Final = (
    "ungated exploratory diagnostic; Task 9H makes no latency claim, no gate is "
    "applied to this measurement, and no dataset partition was opened"
)


class LatencyDiagnosticError(RuntimeError):
    """Raised when the matched latency diagnostic cannot proceed safely."""


# ---------------------------------------------------------------------------
# Output and input guards
# ---------------------------------------------------------------------------


def assert_benchmark_target(attempt_id: str) -> dict[str, Any]:
    """Resolve one attempt against the closed benchmark registry.

    Refuses anything unregistered, and anything registered here but absent from
    the frozen registry -- a timing without an immutable identity beside it is a
    number nothing can be said about.
    """
    name = str(attempt_id)
    target = BENCHMARK_TARGETS.get(name)
    if target is None:
        raise LatencyDiagnosticError(
            f"attempt {attempt_id!r} is not a benchmarkable Task 9H checkpoint; the registry "
            f"is {sorted(BENCHMARK_TARGETS)}"
        )
    if name not in FROZEN_ATTEMPTS:
        raise LatencyDiagnosticError(
            f"attempt {attempt_id!r} is registered for benchmarking but is not a frozen "
            "checkpoint; a timing is only meaningful beside an immutable identity"
        )
    return dict(target)


def resolve_benchmark_paths(
    attempt_id: str, output: str | None = None, attempt_directory: str | None = None
) -> dict[str, Any]:
    """Derive this attempt's directory and output, and refuse a mismatched pair.

    Both default **from the attempt**, so choosing an attempt cannot leave a
    path aimed at another one. Two overrides are refused outright rather than
    honoured:

    * writing any attempt but E2b to E2b's historical artifact, which would
      overwrite the only record of the measurement the project quotes;
    * writing E2b anywhere but its historical artifact, which would silently
      fork that record into two.
    """
    target = assert_benchmark_target(attempt_id)
    resolved_output = output or str(target["output"])
    resolved_directory = attempt_directory or str(target["directory"])
    historical = str(BENCHMARK_TARGETS[BENCHMARKED_ATTEMPT]["output"])
    if resolved_output == historical and attempt_id != BENCHMARKED_ATTEMPT:
        raise LatencyDiagnosticError(
            f"'{historical}' is {BENCHMARKED_ATTEMPT!r}'s historical latency artifact and is "
            f"never reused; {attempt_id!r} writes to '{target['output']}'"
        )
    if attempt_id == BENCHMARKED_ATTEMPT and resolved_output != historical:
        raise LatencyDiagnosticError(
            f"{BENCHMARKED_ATTEMPT!r}'s measurement lives at '{historical}'; writing it "
            "elsewhere would fork the record the project quotes"
        )
    return {
        "attempt_id": attempt_id,
        "label": target["label"],
        "output": resolved_output,
        "attempt_directory": resolved_directory,
        "historical": bool(target["historical"]),
    }


def _guarded_output(project_root: Path, relative: str) -> Path:
    where = "matched latency output path"
    assert_path_allowed(relative, where=where)
    assert_contained_relative_path(relative, where=where)
    if not str(relative).replace("\\", "/").startswith("artifacts/"):
        raise LatencyDiagnosticError(
            "the matched latency report lives beneath the ignored artifacts/ tree; it is a "
            "development diagnostic, not frozen evidence"
        )
    return Path(project_root) / relative


def _guarded_attempt_directory(project_root: Path, relative: str) -> Path:
    where = "benchmarked attempt directory"
    assert_path_allowed(relative, where=where)
    assert_contained_relative_path(relative, where=where)
    if not str(relative).replace("\\", "/").startswith("artifacts/"):
        raise LatencyDiagnosticError(
            "a Task 9H checkpoint lives beneath the ignored artifacts/ tree"
        )
    return Path(project_root) / relative


# ---------------------------------------------------------------------------
# The benchmarked model, rebuilt from committed configuration plus its checkpoint
# ---------------------------------------------------------------------------


def recorded_scaling(report: Mapping[str, Any]) -> Scaling:
    """Delegate to :func:`.frozen.recorded_scaling`, preserving this module's error type.

    One definition, in :mod:`.frozen`, so the model this diagnostic times and
    the model the price and Greek diagnostics evaluate cannot be rebuilt two
    slightly different ways.
    """
    try:
        return frozen_recorded_scaling(report)
    except FrozenCheckpointError as error:
        raise LatencyDiagnosticError(str(error)) from error


def load_benchmarked_model(
    project_root: Path,
    directory: str = DEFAULT_ATTEMPT_DIRECTORY,
    attempt_id: str = BENCHMARKED_ATTEMPT,
) -> tuple[AmericanDevPriceModel, dict[str, Any]]:
    """Rebuild the benchmarked attempt's deployed model, with its provenance.

    Delegates to :func:`.frozen.load_frozen_model`, which owns the single
    definition of "rebuild a completed Task 9H attempt": tracked immutable
    configuration for the architecture, the recorded scaling for the affine
    transforms, a configuration digest that must still match, and a run that
    must appear exactly once in the append-only attempt log. This module adds
    only the one fact that belongs to a timing measurement -- that artifact
    loading happens here, outside the timed region.
    """
    try:
        model, provenance = load_frozen_model(project_root, attempt_id, directory)
    except FrozenCheckpointError as error:
        raise LatencyDiagnosticError(str(error)) from error
    provenance["artifact_loading_excluded_from_timing"] = True
    return model, provenance


# ---------------------------------------------------------------------------
# The matched specification
# ---------------------------------------------------------------------------


def matched_specification(latency_config: Mapping[str, Any]) -> dict[str, Any]:
    """The Task 9G latency contract with its depth ladder restricted to one depth."""
    specification = copy.deepcopy(dict(latency_config))
    depths = specification.get("crr_depths")
    if not isinstance(depths, list) or MATCHED_CRR_DEPTH not in depths:
        raise LatencyDiagnosticError(
            f"'{LATENCY_CONFIG_PATH}' does not declare CRR depth {MATCHED_CRR_DEPTH}"
        )
    for key in ("request_shapes", "batch_sizes", "thread_budgets", "cases", "request_case_names"):
        if key not in specification:
            raise LatencyDiagnosticError(f"'{LATENCY_CONFIG_PATH}' declares no {key}")
    specification["crr_depths"] = [MATCHED_CRR_DEPTH]
    return specification


def acceptance_latency_reference(acceptance: Mapping[str, Any]) -> dict[str, Any]:
    """The Task 9G latency bar, read through a two-key whitelist, **not applied**.

    Whitelisted by key for the same reason the geometry analysis whitelists its
    reads: the acceptance configuration also names Task 9G's final partition, and
    that key is never resolved, read into a variable, or written to the report.
    """
    section = acceptance.get("latency")
    if not isinstance(section, Mapping):
        raise LatencyDiagnosticError(f"'{ACCEPTANCE_CONFIG_PATH}' declares no [latency] table")
    depth = section.get("reference_depth_for_interpretation")
    bar = section.get("minimum_median_end_to_end_speedup")
    if isinstance(depth, bool) or not isinstance(depth, int) or depth != MATCHED_CRR_DEPTH:
        raise LatencyDiagnosticError(
            f"'{ACCEPTANCE_CONFIG_PATH}' names reference depth {depth!r}; this diagnostic "
            f"measures {MATCHED_CRR_DEPTH} and refuses to report against another"
        )
    if not isinstance(bar, int | float) or not float(bar) > 0.0:
        raise LatencyDiagnosticError(
            f"'{ACCEPTANCE_CONFIG_PATH}' declares no positive minimum median speedup"
        )
    return {
        "source": ACCEPTANCE_CONFIG_PATH,
        "reference_depth": int(depth),
        "task_9g_minimum_median_end_to_end_speedup": float(bar),
        "applied": False,
        "note": (
            "recorded as context only; Task 9H applies no latency gate and this measurement "
            "neither passes nor fails a Task 9G gate"
        ),
    }


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _summary(values: list[int] | list[float], *, quantile_method: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise LatencyDiagnosticError("a latency sample must be one-dimensional and non-empty")
    summary: dict[str, Any] = {
        "repetitions": int(array.size),
        "minimum_ns": float(array.min()),
        "maximum_ns": float(array.max()),
    }
    for quantile in LATENCY_QUANTILES:
        summary[f"q{quantile:g}_ns"] = float(
            np.quantile(array, quantile, method=quantile_method)
        )
    return summary


def measurement_statistics(
    measurement: Mapping[str, Any], model_name: str, *, quantile_method: str
) -> dict[str, Any]:
    """Median and p95 nanoseconds for both operations, and the paired speedup."""
    crr = _summary(
        list(measurement["crr_adjacent_average_raw_ns"]), quantile_method=quantile_method
    )
    neural = _summary(
        list(measurement["neural_end_to_end_raw_ns"][model_name]), quantile_method=quantile_method
    )
    paired = np.asarray(
        measurement["crr_adjacent_average_raw_ns"], dtype=np.float64
    ) / np.asarray(measurement["neural_end_to_end_raw_ns"][model_name], dtype=np.float64)
    return {
        "shape": measurement["shape"],
        "batch_size": measurement["batch_size"],
        "thread_budget": measurement["thread_budget"],
        "crr_depth": measurement["crr_depth"],
        "crr_adjacent_average": crr,
        "neural_end_to_end": neural,
        "paired_speedup": {
            "median": float(measurement["median_speedup"][model_name]),
            "interval": [
                float(value)
                for value in measurement["median_speedup_confidence_interval"][model_name]
            ],
            "confidence_level_at_least": measurement["median_speedup_confidence_level_at_least"],
            "interval_actual_coverage": measurement["median_speedup_interval_actual_coverage"],
            "minimum": float(paired.min()),
            "maximum": float(paired.max()),
        },
    }


# ---------------------------------------------------------------------------
# The diagnostic
# ---------------------------------------------------------------------------


def benchmark_matched_latency(
    project_root: Path,
    attempt_id: str,
    *,
    output: str | None = None,
    attempt_directory: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Measure the matched latency of one registered checkpoint once, and publish it.

    ``attempt_id`` is **positional and required**. The paths are derived from it
    rather than defaulted independently, so there is no argument whose omission
    silently changes which model was timed.

    **Manual, human-invoked diagnostic.** It is a timing measurement, so its
    numbers depend on the machine being otherwise quiet; no test, hook, CI job or
    repository check calls it. It opens no dataset partition, trains nothing,
    reserves no attempt directory or ledger, appends nothing to the attempt log,
    and mutates no existing attempt evidence.
    """
    project_root = Path(project_root)
    resolved = resolve_benchmark_paths(attempt_id, output, attempt_directory)
    output = str(resolved["output"])
    attempt_directory = str(resolved["attempt_directory"])
    output_path = _guarded_output(project_root, output)

    digests = source_digests(project_root)
    committed = verify_committed_source(project_root, digests)
    repository = repository_identity(project_root)

    acceptance_path = project_root / ACCEPTANCE_CONFIG_PATH
    acceptance = load_toml(acceptance_path)
    validate_acceptance_config(acceptance)
    quantile_method = acceptance.get("quantile_method")
    if not isinstance(quantile_method, str) or not quantile_method:
        raise LatencyDiagnosticError(f"'{ACCEPTANCE_CONFIG_PATH}' declares no quantile_method")
    reference = acceptance_latency_reference(acceptance)

    protocol = load_toml(project_root / PROTOCOL_CONFIG_PATH)
    locked_digest = locked_tracked_input_digest(protocol, LATENCY_CONFIG_PATH)
    latency_path = project_root / LATENCY_CONFIG_PATH
    latency_digest = sha256_file(latency_path)
    if latency_digest != locked_digest:
        raise LatencyDiagnosticError(
            f"'{LATENCY_CONFIG_PATH}' hashes to {latency_digest[:12]}… but the locked protocol "
            f"pins {locked_digest[:12]}…; the latency contract is not the one Task 9G locked"
        )
    specification = matched_specification(load_toml(latency_path))

    # The inter-op budget is a process-wide setting the reused implementation
    # refuses to run without, and PyTorch accepts it only before its inter-op
    # pool exists — so it is fixed here, on the contract's own terms, before any
    # tensor is constructed.
    interop = int(specification["torch_interop_threads"])
    if torch.get_num_interop_threads() != interop:
        try:
            torch.set_num_interop_threads(interop)
        except RuntimeError as error:
            raise LatencyDiagnosticError(
                f"the Torch inter-op thread budget could not be fixed at {interop}; start a "
                "fresh process before running this diagnostic"
            ) from error

    # Artifact loading happens here, before the timed region is entered, exactly
    # as Task 9G excludes it.
    model, provenance = load_benchmarked_model(project_root, attempt_directory, attempt_id)
    model_name = str(provenance["attempt_id"])

    evidence = run_latency({model_name: model}, specification)

    rows = [
        measurement_statistics(measurement, model_name, quantile_method=quantile_method)
        for measurement in evidence["measurements"]
    ]
    expected_rows = len(specification["request_shapes"])
    if len(rows) != expected_rows:
        raise LatencyDiagnosticError(
            f"expected {expected_rows} matched measurement(s), got {len(rows)}"
        )

    report = {
        "schema_version": LATENCY_SCHEMA,
        "task": "task-9h-american-pricer-development",
        "analysis": "matched adjacent-average CRR versus deployed neural latency",
        "status": {
            "confirmatory": False,
            "exploratory": True,
            "gate_applied": False,
            "selection_bias": UNGATED_LABEL,
            "opens_a_dataset_partition": False,
            "validation_exposure": False,
            "trains_a_model": False,
            "reserves_an_attempt": False,
            "writes_attempt_evidence": False,
        },
        "partitions_opened": [],
        "protocol_commit": repository["commit"],
        "repository": repository,
        "source_digests": digests,
        "committed_source": committed,
        "benchmarked_attempt": {
            "attempt_id": resolved["attempt_id"],
            "label": resolved["label"],
            "artifact": output,
            "is_the_historical_e2b_measurement": resolved["historical"],
            "selection": (
                "chosen explicitly; the command line has no default attempt, and the "
                "directory and artifact are derived from the attempt rather than defaulted "
                "independently"
            ),
        },
        "benchmarked_model": provenance,
        "matched_crr_depth": MATCHED_CRR_DEPTH,
        "contract": {
            "source": LATENCY_CONFIG_PATH,
            "sha256": latency_digest,
            "identity_locked_by": PROTOCOL_CONFIG_PATH,
            "implementation": "differentiable_pricing.ml.american_pilot.run_latency",
            "clock": specification["clock"],
            "warmups": specification["warmups"],
            "repetitions": specification["repetitions"],
            "measurement_order": specification["measurement_order"],
            "request_shapes": list(specification["request_shapes"]),
            "batch_sizes": list(specification["batch_sizes"]),
            "thread_budgets": list(specification["thread_budgets"]),
            "torch_interop_threads": interop,
            "crr_operation": specification["crr_operation"],
            "neural_end_to_end": specification["neural_end_to_end"],
            "crr_depths_measured": list(specification["crr_depths"]),
            "deviations_from_the_task_9g_run": [
                "the CRR depth ladder is restricted to the matched depth "
                f"{MATCHED_CRR_DEPTH}; no other semantic changes",
                "one model is timed instead of two arms, so the deterministic rotation "
                "alternates between two operations rather than three",
            ],
        },
        "acceptance_reference": reference,
        "runtime": evidence["runtime"],
        "measurements": rows,
        "raw": evidence["measurements"],
        "limitations": {
            "ungated": UNGATED_LABEL,
            "single_machine_single_run": (
                "one process on one machine, seven repetitions per operation after two "
                "warm-ups; it characterizes this machine under these conditions and is not a "
                "portable performance claim"
            ),
            "not_a_task_9h_result": (
                "Task 9H's scope is price only and it makes no latency claim; this diagnostic "
                "informs what to build next and admits nothing"
            ),
            "development_checkpoint": (
                "the benchmarked checkpoint was selected against validation by a development "
                "loop, so it carries that selection bias; its latency does not depend on the "
                "selection, but the choice of which model to time does"
            ),
            "synthetic_requests": (
                "the requests are the Task 9G contract's fixed synthetic cases, not a market "
                "workload, and batch-1 and batch-8 are the only shapes measured"
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_json_atomic(output_path, report, overwrite=overwrite)
    except FileExistsError as error:
        raise LatencyDiagnosticError(
            f"'{output}' already exists; pass --overwrite to replace the exploratory latency "
            "report, or choose another artifacts/ path"
        ) from error
    return report


def compact_latency(report: Mapping[str, Any]) -> dict[str, Any]:
    """The small view of one latency report: per-shape medians, p95 and speedup."""
    return {
        "schema_version": report["schema_version"],
        "protocol_commit": report["repository"]["commit"],
        "commit": report["repository"]["commit"],
        "attempt_id": report["benchmarked_model"]["attempt_id"],
        "label": report.get("benchmarked_attempt", {}).get("label"),
        "checkpoint_sha256": report["benchmarked_model"]["checkpoint_sha256"][:16],
        "parameters": report["benchmarked_model"]["parameters"],
        "matched_crr_depth": report.get("matched_crr_depth"),
        "crr_depths_measured": report["contract"]["crr_depths_measured"],
        "batch_sizes": report["contract"]["batch_sizes"],
        "thread_budgets": report["contract"]["thread_budgets"],
        "torch_interop_threads": report["contract"]["torch_interop_threads"],
        "gate_applied": report["status"]["gate_applied"],
        "task_9g_reference_bar": report["acceptance_reference"][
            "task_9g_minimum_median_end_to_end_speedup"
        ],
        "measurements": [
            {
                "shape": row["shape"],
                "batch_size": row["batch_size"],
                "thread_budget": row["thread_budget"],
                "crr_median_ns": row["crr_adjacent_average"]["q0.5_ns"],
                "crr_p95_ns": row["crr_adjacent_average"]["q0.95_ns"],
                "neural_median_ns": row["neural_end_to_end"]["q0.5_ns"],
                "neural_p95_ns": row["neural_end_to_end"]["q0.95_ns"],
                "median_speedup": row["paired_speedup"]["median"],
                "speedup_interval": row["paired_speedup"]["interval"],
            }
            for row in report["measurements"]
        ],
        "selection_bias": report["status"]["selection_bias"],
    }


__all__ = [
    "ATTEMPT_REPORT_NAME",
    "BENCHMARKED_ATTEMPT",
    "BENCHMARK_TARGETS",
    "CHECKPOINT_NAME",
    "DEFAULT_ATTEMPT_DIRECTORY",
    "DEFAULT_OUTPUT",
    "DIAGNOSTIC_BENCHMARK_ATTEMPT",
    "LATENCY_CONFIG_PATH",
    "LATENCY_QUANTILES",
    "LATENCY_SCHEMA",
    "MATCHED_CRR_DEPTH",
    "UNGATED_LABEL",
    "LatencyDiagnosticError",
    "acceptance_latency_reference",
    "assert_benchmark_target",
    "benchmark_matched_latency",
    "compact_latency",
    "load_benchmarked_model",
    "matched_specification",
    "measurement_statistics",
    "recorded_scaling",
    "resolve_benchmark_paths",
]
