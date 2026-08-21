"""Task 9H attempt orchestration (manual invocation only).

One tracked immutable attempt configuration in, one recorded attempt out. The
workbench:

* exposes ``train`` and ``validation`` and nothing else, and fails closed on any
  split or path naming a final or held-out partition;
* strips the dataset manifest to those two splits before anything else sees it,
  so the final partition's declared identity is never read into a variable;
* requires a clean tracked worktree and records the exact commit;
* records source, configuration and dataset digests and the selected row
  identities;
* refuses to overwrite an existing attempt;
* writes the checkpoint and reports beneath ignored paths, and emits a compact
  summary an agent can read;
* exposes **no** final-evaluation entry point.

**The development criterion is Task 9G's, reused as code rather than restated.**
Price metrics, slices, bound and shape diagnostics and the pass/fail verdict all
come from :mod:`differentiable_pricing.ml.american_pilot`, evaluated against
``configs/american_neural_pilot_acceptance_v1.toml``. That file is digest-pinned
by the Task 9G protocol and reconciled in ``scripts/check.sh`` and CI, so the
criterion cannot be quietly loosened after an attempt fails.

:func:`execute_attempt` trains a model. It is a manual, terminal-invoked human
command; no test, hook, CI job or repository check calls it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from ..american_pilot import (
    assess_arm,
    physical_features,
    read_partition_columns,
    shape_diagnostics,
    sliced_metrics,
)
from ..artifact import write_json_atomic
from ..model import fit_scaling
from .attempts import (
    ALLOWED_SPLITS,
    assert_path_allowed,
    assert_split_allowed,
    attempt_seeds,
    load_toml,
    repository_identity,
    select_rows_by_hash,
    sha256_file,
    source_digests,
    validate_attempt_config,
)
from .models import build_network, parameter_count
from .representation import (
    NORMALIZED_TARGET,
    PHYSICAL_RECONSTRUCTION,
    REPRESENTATION,
    AmericanDevPriceModel,
    feature_order,
    representation_arrays,
)

ATTEMPT_REPORT_SCHEMA: Final = "american-dev-attempt-report/1"
ATTEMPT_SUMMARY_SCHEMA: Final = "american-dev-attempt-summary/1"
LEDGER_SCHEMA: Final = "american-dev-attempt-ledger/1"

#: The Task 9G acceptance section Task 9H reuses verbatim as its "works" rule.
CRITERION_SECTION: Final = "validation_final_entry"

EVALUATION_COLUMNS: Final = (
    "sample_id",
    "stratum",
    "option_type",
    "spot",
    "strike",
    "maturity",
    "rate",
    "dividend_yield",
    "volatility",
    "american_price",
    "european_crr_price",
    "early_exercise_premium",
    "intrinsic_value",
    "earliest_exercise_step",
)


class WorkbenchError(RuntimeError):
    """Raised when a Task 9H attempt cannot proceed safely."""


# ---------------------------------------------------------------------------
# Dataset access, restricted to train and validation
# ---------------------------------------------------------------------------


def restricted_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the manifest with only ``train`` and ``validation`` reachable.

    Every other declared file — including the final partition — is dropped here,
    before any consumer sees the mapping. Its path is never resolved, its digest
    is never read, and its row count is never counted.
    """
    files = [entry for entry in manifest.get("files", []) if entry.get("split") in ALLOWED_SPLITS]
    present = {str(entry["split"]) for entry in files}
    missing = sorted(set(ALLOWED_SPLITS) - present)
    if missing:
        raise WorkbenchError(f"dataset manifest does not declare partition(s): {missing}")
    restricted = {key: value for key, value in manifest.items() if key != "files"}
    restricted["files"] = files
    return restricted


def load_partition(dataset: Path, manifest: Mapping[str, Any], split: str) -> dict[str, np.ndarray]:
    return read_partition_columns(
        dataset, manifest, assert_split_allowed(split), EVALUATION_COLUMNS, verify_digest=True
    )


def _guarded(project_root: Path, relative: str, *, where: str) -> Path:
    """Guard the *declared* relative path, then resolve it under the root.

    The guard runs on the configured value rather than on the absolute path, so
    a checkout directory that happens to contain a guarded word is not mistaken
    for an attempt to reach a final partition.
    """
    return project_root / assert_path_allowed(relative, where=where)


# ---------------------------------------------------------------------------
# Model construction and training
# ---------------------------------------------------------------------------


def build_model(config: Mapping[str, Any], scaling: Any) -> AmericanDevPriceModel:
    """Construct the attempt's model under its own declared initialization seed.

    Seeding happens here, not at the start of training: a network's weights are
    drawn when its layers are constructed, so seeding afterwards would leave the
    initialization at the mercy of whatever consumed the global generator first
    and two runs of the same attempt would not match.
    """
    torch.manual_seed(attempt_seeds(config)["initialization"])
    conditioning = tuple(config.get("conditioning_features", ()))
    network = build_network(dict(config["architecture"]), len(feature_order(conditioning)))
    return AmericanDevPriceModel(
        network, scaling, head=str(config["head"]), conditioning=conditioning
    )


def train_model(
    model: AmericanDevPriceModel,
    config: Mapping[str, Any],
    train_physical: np.ndarray,
    train_targets: np.ndarray,
    validation_physical: np.ndarray,
    validation_targets: np.ndarray,
) -> dict[str, Any]:
    """Fit one price-only attempt with checkpoint selection on validation.

    The objective is the standardized-target mean squared error in
    ``u = V / (S*exp(-q*T))`` — the Task 9G control's objective, unchanged.
    """
    training = config["training"]
    optimizer_section = config["optimizer"]
    torch.use_deterministic_algorithms(bool(training.get("deterministic_algorithms", True)))
    torch.set_num_threads(int(training["num_threads"]))
    x_train = torch.as_tensor(train_physical, dtype=torch.float64)
    y_train = torch.as_tensor(train_targets, dtype=torch.float64)
    x_validation = torch.as_tensor(validation_physical, dtype=torch.float64)
    y_validation = torch.as_tensor(validation_targets, dtype=torch.float64)
    optimizer = torch.optim.AdamW(
        model.network.parameters(),
        lr=float(optimizer_section["learning_rate"]),
        betas=(
            float(optimizer_section.get("beta1", 0.9)),
            float(optimizer_section.get("beta2", 0.999)),
        ),
        eps=float(optimizer_section.get("epsilon", 1.0e-8)),
        weight_decay=float(optimizer_section.get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(optimizer_section.get("schedule_period_epochs", training["epochs"])),
        eta_min=float(optimizer_section.get("minimum_learning_rate", 0.0)),
    )
    generator = torch.Generator(device="cpu").manual_seed(attempt_seeds(config)["shuffle"])
    batch_size = int(training["batch_size"])
    best_metric = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(training["epochs"]) + 1):
        learning_rate = float(optimizer.param_groups[0]["lr"])
        model.train()
        order = torch.randperm(x_train.shape[0], generator=generator)
        for start in range(0, x_train.shape[0], batch_size):
            index = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            predicted = model.normalized_target(x_train[index])
            loss = torch.mean(torch.square((predicted - y_train[index]) / model.price_scale))
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            predicted = model.normalized_target(x_validation)
            metric = float(torch.mean(torch.square((predicted - y_validation) / model.price_scale)))
        if not np.isfinite(metric):
            raise WorkbenchError("training produced a non-finite validation metric")
        history.append(
            {
                "epoch": epoch,
                "validation_standardized_target_mse": metric,
                "learning_rate": learning_rate,
            }
        )
        if metric < best_metric:
            best_metric, best_epoch = metric, epoch
            best_state = {
                key: value.detach().clone() for key, value in model.network.state_dict().items()
            }
    if best_state is None:
        raise WorkbenchError("training produced no checkpoint")
    model.network.load_state_dict(best_state, strict=True)
    model.eval()
    return {
        "objective": "standardized target mean squared error",
        "best_epoch": best_epoch,
        "best_validation_standardized_target_mse": best_metric,
        "epochs": int(training["epochs"]),
        "history": history,
        "checkpoint_rule": "minimum validation standardized-target MSE; earliest epoch wins ties",
    }


# ---------------------------------------------------------------------------
# Evaluation against the reused Task 9G criterion
# ---------------------------------------------------------------------------


def predict_prices(
    model: AmericanDevPriceModel, physical: np.ndarray, batch_size: int
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, physical.shape[0], batch_size):
            block = torch.as_tensor(physical[start : start + batch_size], dtype=torch.float64)
            outputs.append(model(block).numpy())
    prediction = np.concatenate(outputs)
    if not bool(np.isfinite(prediction).all()):
        raise WorkbenchError("the model produced a non-finite price")
    return prediction


def evaluate_attempt(
    model: AmericanDevPriceModel,
    columns: Mapping[str, np.ndarray],
    acceptance: Mapping[str, Any],
    batch_size: int,
) -> dict[str, Any]:
    """Price metrics, Task 9G bound/shape diagnostics, and the reused verdict."""
    physical = physical_features(columns)
    prediction = predict_prices(model, physical, batch_size)
    diagnostics, _ = shape_diagnostics(
        model, columns, acceptance, batch_size=batch_size, center_prediction=prediction
    )
    metrics = {
        "slices": sliced_metrics(columns, prediction, acceptance),
        "diagnostics": diagnostics,
    }
    metrics["gate"] = assess_arm(metrics, acceptance, CRITERION_SECTION)
    baseline = np.asarray(columns["european_crr_price"], dtype=np.float64)
    metrics["european_crr_baseline"] = sliced_metrics(columns, baseline, acceptance)["overall"]
    return metrics


# ---------------------------------------------------------------------------
# Attempt execution
# ---------------------------------------------------------------------------


def _ledger(config: Mapping[str, Any], project_root: Path) -> Path:
    return _guarded(project_root, str(config["paths"]["ledger"]), where="paths.ledger")


def _output(config: Mapping[str, Any], project_root: Path) -> Path:
    return _guarded(
        project_root, str(config["paths"]["output_directory"]), where="paths.output_directory"
    )


def attempt_status(config_path: Path, project_root: Path) -> dict[str, Any]:
    """Read the ledger only. Opens no dataset partition and runs nothing."""
    config = validate_attempt_config(load_toml(config_path))
    ledger_path = _ledger(config, project_root)
    if not ledger_path.is_file():
        return {"attempt_id": config["attempt_id"], "status": "not_started"}
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    return {
        "attempt_id": ledger.get("attempt_id"),
        "status": ledger.get("status"),
        "report": ledger.get("report"),
        "summary": ledger.get("summary"),
    }


def compact_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    """The small, agent-readable view of one attempt.

    A verdict and the worst slices, not a per-row dump: an agent choosing the
    next attempt needs to see where the model is wrong, not every prediction.
    """
    slices = report["price_metrics"]["slices"]
    worst = sorted(
        (
            {
                "slice": name,
                "rows": entry["rows"],
                "normalized_rmse": entry["normalized"]["rmse"],
                "normalized_maximum_absolute_error": entry["normalized"][
                    "maximum_absolute_error"
                ],
            }
            for name, entry in slices.items()
            if entry["rows"] and entry["normalized"] is not None
        ),
        key=lambda item: -item["normalized_rmse"],
    )
    return {
        "schema_version": ATTEMPT_SUMMARY_SCHEMA,
        "attempt_id": report["attempt_id"],
        "parent_attempt": report["parent_attempt"],
        "hypothesis": report["hypothesis"],
        "git_commit": report["repository"]["commit"],
        "config_path": report["config_path"],
        "config_sha256": report["config_sha256"],
        "architecture": report["architecture"],
        "seeds": report["seeds"],
        "selected_rows": report["selected_rows"]["count"],
        "best_epoch": report["training"]["best_epoch"],
        "overall_normalized": slices["overall"]["normalized"],
        "european_crr_baseline_normalized": report["price_metrics"]["european_crr_baseline"][
            "normalized"
        ],
        "material_bound_violations": report["price_metrics"]["diagnostics"][
            "material_bound_violations"
        ],
        "material_shape_violations": report["price_metrics"]["diagnostics"][
            "material_shape_violations"
        ],
        "criterion": report["criterion"],
        "gate": report["price_metrics"]["gate"],
        "worst_slices": worst[:10],
        "selection_bias": (
            "measured against validation, which this loop selects on repeatedly; a "
            "development measurement, not a project result"
        ),
    }


def execute_attempt(config_path: Path, project_root: Path) -> dict[str, Any]:
    """Run one Task 9H price attempt end to end. **Manual human command only.**"""
    config = validate_attempt_config(load_toml(config_path))
    paths = config["paths"]
    output = _output(config, project_root)
    ledger_path = _ledger(config, project_root)
    if output.exists() or ledger_path.exists():
        raise WorkbenchError(
            f"attempt {config['attempt_id']!r} already has outputs; an attempt is never "
            "overwritten — assign a new attempt ID"
        )
    repository = repository_identity(project_root)
    config_digest = sha256_file(config_path)
    output.mkdir(parents=True, exist_ok=False)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = {
        "schema_version": LEDGER_SCHEMA,
        "attempt_id": config["attempt_id"],
        "status": "reserved",
        "config_sha256": config_digest,
        "repository": repository,
        "report": None,
        "summary": None,
        "failure": None,
    }
    write_json_atomic(ledger_path, ledger)

    try:
        dataset = _guarded(project_root, str(paths["dataset"]), where="paths.dataset")
        manifest_path = _guarded(
            project_root, str(paths["dataset_manifest"]), where="paths.dataset_manifest"
        )
        acceptance_path = _guarded(
            project_root, str(paths["acceptance_config"]), where="paths.acceptance_config"
        )
        acceptance = load_toml(acceptance_path)
        manifest = restricted_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
        train_columns = load_partition(dataset, manifest, "train")
        validation_columns = load_partition(dataset, manifest, "validation")

        selection = config["row_selection"]
        selected = select_rows_by_hash(
            train_columns["sample_id"], int(selection["row_budget"]), str(selection["salt"])
        )
        selected_ids = [str(train_columns["sample_id"][index]) for index in selected.tolist()]
        train_physical = physical_features(train_columns)[selected]
        train_prices = np.asarray(train_columns["american_price"], dtype=np.float64)[selected]
        validation_physical = physical_features(validation_columns)
        validation_prices = np.asarray(validation_columns["american_price"], dtype=np.float64)

        conditioning = tuple(config.get("conditioning_features", ()))
        train_features, train_targets = representation_arrays(
            train_physical, train_prices, conditioning
        )
        _, validation_targets = representation_arrays(
            validation_physical, validation_prices, conditioning
        )
        scaling = fit_scaling(train_features, train_targets)
        model = build_model(config, scaling)
        training_record = train_model(
            model,
            config,
            train_physical,
            train_targets,
            validation_physical,
            validation_targets,
        )
        price_metrics = evaluate_attempt(
            model, validation_columns, acceptance, int(config["evaluation"]["batch_size"])
        )
        torch.save(model.network.state_dict(), output / "checkpoint.pt")

        report = {
            "schema_version": ATTEMPT_REPORT_SCHEMA,
            "attempt_id": config["attempt_id"],
            "parent_attempt": str(config.get("parent_attempt", "")),
            "hypothesis": str(config["hypothesis"]),
            "config_path": str(Path(config_path).relative_to(project_root)),
            "config_sha256": config_digest,
            "repository": repository,
            "source_digests": source_digests(project_root),
            "dataset": {
                "manifest_sha256": sha256_file(manifest_path),
                "partitions_opened": list(ALLOWED_SPLITS),
                "final_partition_touched": False,
            },
            "criterion": {
                "source": str(Path(acceptance_path).relative_to(project_root)),
                "sha256": sha256_file(acceptance_path),
                "section": CRITERION_SECTION,
                "thresholds": dict(acceptance[CRITERION_SECTION]),
                "note": (
                    "Task 9G's criterion, reused as code and as a digest-pinned file; it is "
                    "not revised because an attempt failed"
                ),
            },
            "architecture": {
                **dict(config["architecture"]),
                "parameters": parameter_count(model.network),
                "head": str(config["head"]),
                "conditioning_features": list(conditioning),
            },
            "features": list(feature_order(conditioning)),
            "target_and_reconstruction": {
                "representation": REPRESENTATION,
                "target": NORMALIZED_TARGET,
                "reconstruction": PHYSICAL_RECONSTRUCTION,
            },
            "scaling": {
                "fit_partition": "selected train rows only",
                "population_standard_deviation_ddof": 0,
                "feature_mean": [float(value) for value in scaling.feature_mean],
                "feature_scale": [float(value) for value in scaling.feature_scale],
                "target_mean": float(scaling.price_mean),
                "target_scale": float(scaling.price_scale),
            },
            "seeds": attempt_seeds(config),
            "selected_rows": {
                "partition": str(selection["partition"]),
                "salt": str(selection["salt"]),
                "rule": str(selection["rule"]),
                "count": len(selected_ids),
                "sample_ids": selected_ids,
            },
            "optimizer_and_budget": {**dict(config["optimizer"]), **dict(config["training"])},
            "training": training_record,
            "price_metrics": price_metrics,
            "limitations": {
                "selection_bias": (
                    "validation is used repeatedly for architecture and training selection, "
                    "so nothing here is an unbiased result"
                ),
                "price_only": (
                    "no Greek, latency or implied-volatility claim is made or measured; those "
                    "are separate follow-up stages"
                ),
                "final_partition": (
                    "interpolation_test and every other final partition were not opened, "
                    "hashed, stat-ed, imported, counted or inspected"
                ),
            },
        }
        write_json_atomic(output / "attempt-report.json", report)
        summary = compact_summary(report)
        write_json_atomic(output / "summary.json", summary)
        ledger["status"] = "complete"
        ledger["report"] = {
            "path": str((output / "attempt-report.json").relative_to(project_root)),
            "sha256": sha256_file(output / "attempt-report.json"),
        }
        ledger["summary"] = {
            "path": str((output / "summary.json").relative_to(project_root)),
            "sha256": sha256_file(output / "summary.json"),
        }
        write_json_atomic(ledger_path, ledger, overwrite=True)
        return report
    except BaseException as error:
        ledger["status"] = "failed"
        ledger["failure"] = f"{type(error).__name__}: {error}"
        write_json_atomic(ledger_path, ledger, overwrite=True)
        raise
