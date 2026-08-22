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
import json
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
    ATTEMPT_LOG_PATH,
    PROTOCOL_CONFIG_PATH,
    assert_contained_relative_path,
    assert_path_allowed,
    attempt_log_entries,
    load_toml,
    repository_identity,
    sha256_file,
    source_digests,
    validate_acceptance_config,
    validate_attempt_config,
    verify_committed_source,
)
from .domain import locked_tracked_input_digest
from .representation import AmericanDevPriceModel
from .workbench import build_model

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

#: The attempt whose surviving checkpoint is benchmarked, and the ignored
#: directory a completed run wrote it to.
BENCHMARKED_ATTEMPT: Final = "scratch_residual_smooth_floor_raw_loss_v1"
DEFAULT_ATTEMPT_DIRECTORY: Final = (
    "artifacts/task-9h/scratch_residual_smooth_floor_raw_loss_v1"
)
ATTEMPT_REPORT_NAME: Final = "attempt-report.json"
CHECKPOINT_NAME: Final = "checkpoint.pt"

#: Where the report is written: beneath the ignored ``artifacts/`` tree. A
#: development diagnostic, never frozen evidence, never committed.
DEFAULT_OUTPUT: Final = "artifacts/task-9h/latency/matched-latency-v1.json"

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
    """The train-fitted scaling the attempt recorded, rebuilt exactly.

    The checkpoint stores the network's weights only, so the affine input and
    target transforms come from the attempt report. They are read, never
    refitted: refitting would require opening a partition, and a scaling that
    differs from the one training used is a different model.
    """
    section = report.get("scaling")
    if not isinstance(section, Mapping):
        raise LatencyDiagnosticError("the attempt report records no scaling")
    try:
        scaling = Scaling(
            feature_mean=np.asarray(section["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(section["feature_scale"], dtype=np.float64),
            price_mean=float(section["target_mean"]),
            price_scale=float(section["target_scale"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LatencyDiagnosticError(f"the recorded scaling is unusable: {error}") from error
    if (
        not bool(np.isfinite(scaling.feature_mean).all())
        or not bool(np.isfinite(scaling.feature_scale).all())
        or bool((scaling.feature_scale <= 0.0).any())
        or not np.isfinite(scaling.price_scale)
        or scaling.price_scale <= 0.0
    ):
        raise LatencyDiagnosticError("the recorded scaling is not finite with positive scales")
    return scaling


def load_benchmarked_model(
    project_root: Path, directory: str = DEFAULT_ATTEMPT_DIRECTORY
) -> tuple[AmericanDevPriceModel, dict[str, Any]]:
    """Rebuild the deployed model of one recorded attempt, with its provenance.

    The architecture, head and conditioning features come from the **tracked,
    immutable attempt configuration**, not from the report, and the report's
    recorded configuration digest is required to match the tracked file's — the
    same immutability property ``scripts/american_dev_attempts.py check``
    enforces for the log. The attempt is also required to be recorded in the
    append-only attempt log under that same digest, so a checkpoint from an
    unrecorded run cannot be benchmarked as if it were part of the search.
    """
    project_root = Path(project_root)
    attempt_directory = _guarded_attempt_directory(project_root, directory)
    report_path = attempt_directory / ATTEMPT_REPORT_NAME
    checkpoint_path = attempt_directory / CHECKPOINT_NAME
    for path in (report_path, checkpoint_path):
        if not path.is_file():
            raise LatencyDiagnosticError(
                f"'{path}' does not exist; the benchmarked checkpoint comes from a completed "
                "attempt and this diagnostic never trains one"
            )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    attempt_id = str(report.get("attempt_id"))
    if attempt_id != BENCHMARKED_ATTEMPT:
        raise LatencyDiagnosticError(
            f"'{directory}' records attempt {attempt_id!r}; this diagnostic benchmarks "
            f"{BENCHMARKED_ATTEMPT!r}"
        )
    relative_config = str(report.get("config_path"))
    assert_path_allowed(relative_config, where="benchmarked attempt configuration")
    assert_contained_relative_path(relative_config, where="benchmarked attempt configuration")
    config_path = project_root / relative_config
    if not config_path.is_file():
        raise LatencyDiagnosticError(f"'{relative_config}' does not exist")
    config_digest = sha256_file(config_path)
    if config_digest != str(report.get("config_sha256")):
        raise LatencyDiagnosticError(
            f"'{relative_config}' hashes to {config_digest[:12]}… but the attempt recorded "
            f"{str(report.get('config_sha256'))[:12]}…; a used configuration is immutable"
        )
    logged = [
        entry
        for entry in attempt_log_entries(project_root / ATTEMPT_LOG_PATH)
        if str(entry.get("attempt_id")) == attempt_id
    ]
    if len(logged) != 1 or str(logged[0].get("config_sha256")) != config_digest:
        raise LatencyDiagnosticError(
            f"attempt {attempt_id!r} is not recorded exactly once in '{ATTEMPT_LOG_PATH}' under "
            "the configuration digest it ran; an unrecorded checkpoint is not benchmarked"
        )

    config = load_toml(config_path)
    validate_attempt_config(config)
    model = build_model(config, recorded_scaling(report))
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state, strict=True)
    model.eval()
    parameters = int(sum(int(tensor.numel()) for tensor in model.network.parameters()))
    recorded_parameters = report.get("architecture", {}).get("parameters")
    if recorded_parameters is not None and int(recorded_parameters) != parameters:
        raise LatencyDiagnosticError(
            f"the rebuilt network has {parameters} parameters but the attempt recorded "
            f"{int(recorded_parameters)}; the checkpoint does not match its configuration"
        )
    provenance = {
        "attempt_id": attempt_id,
        "attempt_directory": directory,
        "config_path": relative_config,
        "config_sha256": config_digest,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "attempt_report_sha256": sha256_file(report_path),
        "recorded_in": ATTEMPT_LOG_PATH,
        "head": str(config["head"]),
        "head_temperature": model.temperature,
        "head_european_margin": model.european_margin,
        "conditioning_features": list(config.get("conditioning_features", ())),
        "architecture": dict(config["architecture"]),
        "parameters": parameters,
        "best_epoch": report.get("training", {}).get("best_epoch"),
        "scaling_source": f"{directory}/{ATTEMPT_REPORT_NAME} :: scaling",
        "artifact_loading_excluded_from_timing": True,
    }
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
    *,
    output: str = DEFAULT_OUTPUT,
    attempt_directory: str = DEFAULT_ATTEMPT_DIRECTORY,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Measure the matched latency of one recorded checkpoint once, and publish it.

    **Manual, human-invoked diagnostic.** It is a timing measurement, so its
    numbers depend on the machine being otherwise quiet; no test, hook, CI job or
    repository check calls it. It opens no dataset partition, trains nothing,
    reserves no attempt directory or ledger, appends nothing to the attempt log,
    and mutates no existing attempt evidence.
    """
    project_root = Path(project_root)
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
    model, provenance = load_benchmarked_model(project_root, attempt_directory)
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
        "repository": repository,
        "source_digests": digests,
        "committed_source": committed,
        "benchmarked_model": provenance,
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
        "commit": report["repository"]["commit"],
        "attempt_id": report["benchmarked_model"]["attempt_id"],
        "parameters": report["benchmarked_model"]["parameters"],
        "crr_depths_measured": report["contract"]["crr_depths_measured"],
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
    "CHECKPOINT_NAME",
    "DEFAULT_ATTEMPT_DIRECTORY",
    "DEFAULT_OUTPUT",
    "LATENCY_CONFIG_PATH",
    "LATENCY_QUANTILES",
    "LATENCY_SCHEMA",
    "MATCHED_CRR_DEPTH",
    "UNGATED_LABEL",
    "LatencyDiagnosticError",
    "acceptance_latency_reference",
    "benchmark_matched_latency",
    "compact_latency",
    "load_benchmarked_model",
    "matched_specification",
    "measurement_statistics",
    "recorded_scaling",
]
