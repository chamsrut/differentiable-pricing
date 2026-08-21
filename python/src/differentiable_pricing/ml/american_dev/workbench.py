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

**A genuine infrastructure failure does not become a silent retry.** Everything
that can be checked is checked *before* the output directory is created, so a
bad configuration, an uncommitted source file or a dataset whose identity does
not match Task 9G's locked one fails with nothing reserved and the same attempt
ID still available. Once the directory exists the attempt ID is spent: a crash
part-way through leaves ``status="failed"`` and the recorded failure in the
ledger, the directory stays, and :func:`execute_attempt` refuses to run that ID
again. The human records the dead attempt with
``--outcome infrastructure_failure`` and the retry gets a **new** attempt ID and
a new configuration file. Deleting the directory to reuse the ID would erase the
evidence that the first run happened, so it is never the remedy.

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
    verify_partition_policy,
)
from ..artifact import write_json_atomic
from ..model import fit_scaling
from .attempts import (
    ACCEPTANCE_CONFIG_PATH,
    ALLOWED_SPLITS,
    ATTEMPT_REPORT_SCHEMA,
    CRITERION_SECTION,
    PROTOCOL_CONFIG_PATH,
    assert_contained_relative_path,
    assert_path_allowed,
    assert_split_allowed,
    attempt_seeds,
    load_toml,
    repository_identity,
    select_rows_by_hash,
    sha256_file,
    source_digests,
    validate_acceptance_config,
    validate_attempt_config,
    verify_committed_source,
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

ATTEMPT_SUMMARY_SCHEMA: Final = "american-dev-attempt-summary/1"
LEDGER_SCHEMA: Final = "american-dev-attempt-ledger/1"

#: ``ATTEMPT_REPORT_SCHEMA`` and ``CRITERION_SECTION`` are imported from
#: :mod:`attempts`, which owns the single definition: the PyTorch-free recorder
#: validates a report against exactly the values this module writes.
#:
#: The dataset identities Task 9H may know are the ones Task 9G already locked.
#: Only the manifest, ``train`` and ``validation`` digests are ever read out of
#: the protocol; the key naming any other partition's digest is not in the
#: whitelist below and is never resolved, opened or compared.
LOCKED_IDENTITY_KEYS: Final = ("manifest_sha256", "train_sha256", "validation_sha256")

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

    The two retained entries are not trusted either. The manifest is data, not
    configuration: its ``file`` value is a name this process is about to join to
    the dataset directory and open. So each retained name goes through the same
    final-partition guard and containment check as a configured path *before*
    anything opens, hashes, stats or counts it — a ``train`` entry pointing at
    ``../interpolation_test.parquet`` fails closed here rather than being read.
    """
    files: list[dict[str, Any]] = []
    for entry in manifest.get("files", []):
        if not isinstance(entry, Mapping) or entry.get("split") not in ALLOWED_SPLITS:
            continue
        split = assert_split_allowed(str(entry["split"]))
        name = entry.get("file")
        if not isinstance(name, str) or not name.strip():
            raise WorkbenchError(f"dataset manifest entry for {split!r} declares no file name")
        # Both guards raise ``AttemptError`` subclasses, which the runner already
        # reports; a final-partition name stays a ``FinalPartitionAccessError``.
        where = f"dataset manifest entry for {split!r}"
        assert_path_allowed(name, where=where)
        assert_contained_relative_path(name, where=where)
        digest = entry.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise WorkbenchError(f"{where} declares no SHA-256")
        files.append(dict(entry))
    present = {str(entry["split"]) for entry in files}
    missing = sorted(set(ALLOWED_SPLITS) - present)
    if missing:
        raise WorkbenchError(f"dataset manifest does not declare partition(s): {missing}")
    duplicated = sorted(name for name in present if sum(e["split"] == name for e in files) > 1)
    if duplicated:
        raise WorkbenchError(f"dataset manifest declares partition(s) twice: {duplicated}")
    restricted = {key: value for key, value in manifest.items() if key != "files"}
    restricted["files"] = files
    return restricted


def locked_dataset_identity(protocol: Mapping[str, Any]) -> dict[str, str]:
    """The manifest, ``train`` and ``validation`` digests Task 9G already locked.

    Whitelisted by key, so no other partition's declared digest is read into a
    variable, compared, or written to a report — Task 9H learns the identity of
    the two partitions it may reach and nothing about any other.
    """
    dataset = protocol.get("dataset")
    if not isinstance(dataset, Mapping):
        raise WorkbenchError(f"'{PROTOCOL_CONFIG_PATH}' declares no [dataset] table")
    identity: dict[str, str] = {}
    for key in LOCKED_IDENTITY_KEYS:
        value = dataset.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise WorkbenchError(f"'{PROTOCOL_CONFIG_PATH}' declares no [dataset].{key}")
        identity[key] = value
    return identity


def locked_dataset_paths(protocol: Mapping[str, Any]) -> tuple[str, str]:
    """The dataset directory and manifest Task 9G locked, guarded before use."""
    paths = protocol.get("paths")
    if not isinstance(paths, Mapping):
        raise WorkbenchError(f"'{PROTOCOL_CONFIG_PATH}' declares no [paths] table")
    resolved: list[str] = []
    for key in ("dataset", "dataset_manifest"):
        value = paths.get(key)
        if not isinstance(value, str) or not value.strip():
            raise WorkbenchError(f"'{PROTOCOL_CONFIG_PATH}' declares no paths.{key}")
        where = f"locked protocol paths.{key}"
        assert_path_allowed(value, where=where)
        assert_contained_relative_path(value, where=where)
        resolved.append(value)
    return resolved[0], resolved[1]


def verify_dataset_identity(
    dataset: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    locked: Mapping[str, str],
) -> dict[str, str]:
    """Pin the manifest and both reachable partitions to Task 9G's identities.

    Fails closed on any mismatch. Without this an attempt could train on a
    regenerated or silently edited dataset and still be compared against the
    Task 9G control, which would make every paired comparison in the attempt log
    meaningless.
    """
    actual_manifest = sha256_file(manifest_path)
    if actual_manifest != locked["manifest_sha256"]:
        raise WorkbenchError(
            f"dataset manifest '{manifest_path}' hashes to {actual_manifest[:12]}…, but Task 9G "
            f"locked {locked['manifest_sha256'][:12]}…; Task 9H trains on the locked dataset"
        )
    digests: dict[str, str] = {"manifest": actual_manifest}
    for entry in manifest["files"]:
        split = assert_split_allowed(str(entry["split"]))
        path = dataset / str(entry["file"])
        actual = sha256_file(path)
        if actual != str(entry["sha256"]):
            raise WorkbenchError(
                f"{split} partition hashes to {actual[:12]}…, but its manifest entry declares "
                f"{str(entry['sha256'])[:12]}…"
            )
        expected = locked[f"{split}_sha256"]
        if actual != expected:
            raise WorkbenchError(
                f"{split} partition hashes to {actual[:12]}…, but Task 9G locked "
                f"{expected[:12]}…; Task 9H trains on the locked dataset"
            )
        digests[split] = actual
    return digests


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


def _repository_relative(path: Path, project_root: Path) -> str:
    """The repository-relative name of a file that must live inside the tree."""
    try:
        relative = Path(path).resolve().relative_to(Path(project_root).resolve())
    except ValueError as error:
        raise WorkbenchError(
            f"'{path}' is outside the repository root '{project_root}'; a Task 9H attempt runs "
            "only from committed source inside this repository"
        ) from error
    return str(relative).replace("\\", "/")


def preflight(config_path: Path, project_root: Path) -> dict[str, Any]:
    """Everything checkable before a single byte of an attempt is reserved.

    Ordered deliberately: a configuration that is invalid, not committed, not
    inside the repository, or pointed at a dataset whose identity is not Task
    9G's fails here — with no output directory created, no ledger written and
    the attempt ID still unused. Only once every one of these holds does
    :func:`execute_attempt` reserve the attempt, after which the ID is spent
    whatever happens next.
    """
    config = validate_attempt_config(load_toml(config_path))
    relative_config = _repository_relative(config_path, project_root)

    acceptance_path = _guarded(
        project_root, str(config["paths"]["acceptance_config"]), where="paths.acceptance_config"
    )
    acceptance = load_toml(acceptance_path)
    validate_acceptance_config(acceptance)

    protocol_path = project_root / PROTOCOL_CONFIG_PATH
    protocol = load_toml(protocol_path)
    locked_identity = locked_dataset_identity(protocol)
    locked_dataset, locked_manifest = locked_dataset_paths(protocol)
    for key, locked in (("dataset", locked_dataset), ("dataset_manifest", locked_manifest)):
        declared = str(config["paths"][key])
        if declared != locked:
            raise WorkbenchError(
                f"paths.{key} is '{declared}', but Task 9G locked '{locked}'; Task 9H trains on "
                "the locked dataset and no other"
            )

    digests = source_digests(project_root)
    committed = verify_committed_source(project_root, (relative_config, *digests))
    return {
        "config": config,
        "relative_config": relative_config,
        "acceptance": acceptance,
        "acceptance_path": acceptance_path,
        "locked_identity": locked_identity,
        "source_digests": digests,
        "committed_source": committed,
    }


def execute_attempt(config_path: Path, project_root: Path) -> dict[str, Any]:
    """Run one Task 9H price attempt end to end. **Manual human command only.**"""
    # The reuse refusal comes first, before anything else is examined: an
    # attempt ID that has already produced outputs is spent, whatever the state
    # of the tree or the dataset.
    config = validate_attempt_config(load_toml(config_path))
    output = _output(config, project_root)
    ledger_path = _ledger(config, project_root)
    if output.exists() or ledger_path.exists():
        raise WorkbenchError(
            f"attempt {config['attempt_id']!r} already has outputs; an attempt is never "
            "overwritten — assign a new attempt ID"
        )
    checked = preflight(config_path, project_root)
    config = checked["config"]
    paths = config["paths"]
    acceptance_path = checked["acceptance_path"]
    acceptance = checked["acceptance"]
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
        "committed_source": checked["committed_source"],
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
        manifest = restricted_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
        dataset_digests = verify_dataset_identity(
            dataset, manifest_path, manifest, checked["locked_identity"]
        )
        # Task 9G's own row-level admission check, reused rather than
        # reimplemented: every row of both reachable partitions must carry the
        # admitted label policy and step count. A dataset that matches the
        # locked digests but was admitted under a different policy would train a
        # model against labels the project never accepted.
        for split in ALLOWED_SPLITS:
            verify_partition_policy(dataset, manifest, split)
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
            "config_path": checked["relative_config"],
            "config_sha256": config_digest,
            "repository": repository,
            "source_digests": checked["source_digests"],
            "committed_source": checked["committed_source"],
            "dataset": {
                "manifest_sha256": dataset_digests["manifest"],
                "partition_sha256": {
                    split: dataset_digests[split] for split in ALLOWED_SPLITS
                },
                "identity_locked_by": PROTOCOL_CONFIG_PATH,
                "row_level_policy_verified": list(ALLOWED_SPLITS),
                "partitions_opened": list(ALLOWED_SPLITS),
                "final_partition_touched": False,
            },
            "criterion": {
                "source": ACCEPTANCE_CONFIG_PATH,
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
