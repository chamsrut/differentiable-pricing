#!/usr/bin/env python3
"""Validate the locked European neural-pricing replication protocol.

The protocol binds exact source-study, dataset, training, and acceptance files
by SHA-256. It also proves mechanically that the replication changes only the
two experiment identities and the two independently derived seeds. The script
does not generate data, train a model, or inspect a final result.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any, Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL: Final = (
    PROJECT_ROOT / "configs" / "european_neural_replication_protocol_v1.toml"
)
PROTOCOL_SCHEMA_VERSION: Final = "european-neural-replication-protocol/v1"
EXPECTED_TOP_LEVEL_KEYS: Final = frozenset(
    {
        "schema_version",
        "name",
        "selected_development_experiment",
        "output_constraint",
        "paths",
        "sha256",
        "seed_policy",
        "execution",
        "outputs",
    }
)
REFERENCE_KEYS: Final = frozenset(
    {
        "development_results",
        "development_dataset_config",
        "development_training_config",
        "replication_dataset_config",
        "replication_training_config",
        "acceptance_config",
    }
)
SEED_POLICY_KEYS: Final = frozenset(
    {
        "algorithm",
        "dataset_label",
        "dataset_seed",
        "training_label",
        "training_seed",
    }
)
EXECUTION_KEYS: Final = frozenset(
    {
        "diagnose_dataset_before_training",
        "checkpoint_selection_partition",
        "final_partition",
        "maximum_final_evaluations",
        "allow_tuning_after_final_evaluation",
        "failure_action",
    }
)
OUTPUT_KEYS: Final = frozenset(
    {
        "dataset",
        "diagnostics",
        "unconstrained_artifact",
        "bounded_artifact",
        "validation_report",
        "final_report",
    }
)
SHA256_PATTERN: Final = re.compile(r"[0-9a-f]{64}")


class ProtocolError(ValueError):
    """Raised when the replication protocol is incomplete or inconsistent."""


def _load_toml(path: Path, description: str) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            payload = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ProtocolError(f"cannot load {description} '{path}': {error}") from error
    if not isinstance(payload, dict):
        raise ProtocolError(f"{description} '{path}' must contain a TOML table")
    return payload


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ProtocolError(f"cannot load {description} '{path}': {error}") from error
    if not isinstance(payload, dict):
        raise ProtocolError(f"{description} '{path}' must contain a JSON object")
    return payload


def _table(payload: dict[str, Any], key: str, expected: frozenset[str]) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ProtocolError(f"protocol section [{key}] must be a table")
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        raise ProtocolError(f"protocol section [{key}] is missing: {', '.join(missing)}")
    if unknown:
        raise ProtocolError(
            f"protocol section [{key}] has unknown keys: {', '.join(unknown)}"
        )
    return value


def _relative_path(root: Path, value: Any, where: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{where} must be a non-empty repository-relative path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ProtocolError(f"{where} must not be absolute or contain '.' or '..': {value!r}")
    root_resolved = root.resolve()
    candidate = root.joinpath(*pure.parts).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as error:
        raise ProtocolError(f"{where} escapes the repository root: {value!r}") from error
    return candidate


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ProtocolError(f"cannot hash protocol input '{path}': {error}") from error
    return digest.hexdigest()


def _require_string(payload: dict[str, Any], key: str, where: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{where}.{key} must be a non-empty string")
    return value


def _require_integer(payload: dict[str, Any], key: str, where: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProtocolError(f"{where}.{key} must be a non-negative integer")
    return value


def _derived_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:4], "big")


def _validate_hashes(
    root: Path,
    paths: dict[str, Any],
    declared_hashes: dict[str, Any],
) -> dict[str, Path]:
    resolved: dict[str, Path] = {}
    for key in sorted(REFERENCE_KEYS):
        path = _relative_path(root, paths[key], f"paths.{key}")
        digest = declared_hashes[key]
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise ProtocolError(f"sha256.{key} must be a lowercase 64-character SHA-256")
        actual = _sha256_file(path)
        if actual != digest:
            raise ProtocolError(
                f"sha256.{key} does not match '{paths[key]}': "
                f"declared {digest}, actual {actual}"
            )
        resolved[key] = path
    return resolved


def _validate_seed_policy(
    seed_policy: dict[str, Any],
    replication_dataset: dict[str, Any],
    replication_training: dict[str, Any],
) -> tuple[int, int]:
    if seed_policy["algorithm"] != "sha256_first_u32_big_endian":
        raise ProtocolError(
            "seed_policy.algorithm must be 'sha256_first_u32_big_endian'"
        )
    dataset_label = _require_string(seed_policy, "dataset_label", "seed_policy")
    training_label = _require_string(seed_policy, "training_label", "seed_policy")
    dataset_seed = _require_integer(seed_policy, "dataset_seed", "seed_policy")
    training_seed = _require_integer(seed_policy, "training_seed", "seed_policy")
    for label, declared, name in (
        (dataset_label, dataset_seed, "dataset_seed"),
        (training_label, training_seed, "training_seed"),
    ):
        derived = _derived_seed(label)
        if declared != derived:
            raise ProtocolError(
                f"seed_policy.{name} is {declared}, but label {label!r} derives {derived}"
            )

    dataset_table = replication_dataset.get("dataset")
    training_table = replication_training.get("training")
    if not isinstance(dataset_table, dict) or not isinstance(training_table, dict):
        raise ProtocolError("replication configurations are missing seed-bearing tables")
    if dataset_table.get("base_seed") != dataset_seed:
        raise ProtocolError(
            "replication dataset base_seed does not match seed_policy.dataset_seed"
        )
    if training_table.get("seed") != training_seed:
        raise ProtocolError(
            "replication training seed does not match seed_policy.training_seed"
        )
    return dataset_seed, training_seed


def _validate_controlled_changes(
    development_dataset: dict[str, Any],
    replication_dataset: dict[str, Any],
    development_training: dict[str, Any],
    replication_training: dict[str, Any],
) -> None:
    development_dataset_table = development_dataset.get("dataset")
    replication_dataset_table = replication_dataset.get("dataset")
    if not isinstance(development_dataset_table, dict) or not isinstance(
        replication_dataset_table, dict
    ):
        raise ProtocolError("dataset configurations must contain [dataset]")
    development_training_table = development_training.get("training")
    replication_training_table = replication_training.get("training")
    if not isinstance(development_training_table, dict) or not isinstance(
        replication_training_table, dict
    ):
        raise ProtocolError("training configurations must contain [training]")

    if development_dataset_table.get("base_seed") == replication_dataset_table.get(
        "base_seed"
    ):
        raise ProtocolError("replication dataset must use a fresh base seed")
    if development_dataset_table.get("name") == replication_dataset_table.get("name"):
        raise ProtocolError("replication dataset must use a fresh dataset identity")
    if development_training_table.get("seed") == replication_training_table.get("seed"):
        raise ProtocolError("replication training must use a fresh training seed")
    if development_training.get("experiment_name") == replication_training.get(
        "experiment_name"
    ):
        raise ProtocolError("replication training must use a fresh experiment identity")

    normalized_dataset = copy.deepcopy(replication_dataset)
    normalized_dataset["dataset"]["name"] = development_dataset_table.get("name")
    normalized_dataset["dataset"]["base_seed"] = development_dataset_table.get(
        "base_seed"
    )
    if normalized_dataset != development_dataset:
        raise ProtocolError(
            "replication dataset configuration may differ from development only "
            "in dataset.name and dataset.base_seed"
        )

    normalized_training = copy.deepcopy(replication_training)
    normalized_training["experiment_name"] = development_training.get("experiment_name")
    normalized_training["training"]["seed"] = development_training_table.get("seed")
    if normalized_training != development_training:
        raise ProtocolError(
            "replication training configuration may differ from development only "
            "in experiment_name and training.seed"
        )


def _validate_execution(
    protocol: dict[str, Any],
    paths: dict[str, Any],
    acceptance: dict[str, Any],
    development_results: dict[str, Any],
    development_training: dict[str, Any],
    execution: dict[str, Any],
) -> None:
    if execution["diagnose_dataset_before_training"] is not True:
        raise ProtocolError("execution.diagnose_dataset_before_training must be true")
    if execution["checkpoint_selection_partition"] != "validation":
        raise ProtocolError(
            "execution.checkpoint_selection_partition must be 'validation'"
        )
    if execution["final_partition"] != "interpolation_test":
        raise ProtocolError("execution.final_partition must be 'interpolation_test'")
    if (
        _require_integer(
            execution,
            "maximum_final_evaluations",
            "execution",
        )
        != 1
    ):
        raise ProtocolError("execution.maximum_final_evaluations must be 1")
    if execution["allow_tuning_after_final_evaluation"] is not False:
        raise ProtocolError(
            "execution.allow_tuning_after_final_evaluation must be false"
        )
    if (
        execution["failure_action"]
        != "record_failure_and_version_a_new_protocol"
    ):
        raise ProtocolError(
            "execution.failure_action must require a new versioned protocol"
        )
    if acceptance.get("selection_partition") != execution[
        "checkpoint_selection_partition"
    ]:
        raise ProtocolError(
            "acceptance selection_partition does not match the replication protocol"
        )
    if acceptance.get("final_partition") != execution["final_partition"]:
        raise ProtocolError(
            "acceptance final_partition does not match the replication protocol"
        )
    if development_results.get("acceptance_config") != paths["acceptance_config"]:
        raise ProtocolError(
            "development results do not reference the pinned acceptance configuration"
        )
    if development_results.get("partition") != "validation":
        raise ProtocolError("development results must describe the validation partition")

    selected = _require_string(
        protocol, "selected_development_experiment", "protocol"
    )
    experiments = development_results.get("experiments")
    if not isinstance(experiments, list):
        raise ProtocolError("development results must contain an experiments list")
    matches = [
        experiment
        for experiment in experiments
        if isinstance(experiment, dict) and experiment.get("id") == selected
    ]
    if len(matches) != 1:
        raise ProtocolError(
            f"selected development experiment {selected!r} must identify exactly one result"
        )
    if matches[0].get("output_constraint") != protocol.get("output_constraint"):
        raise ProtocolError(
            "selected development experiment does not use the protocol output constraint"
        )
    development_model = development_training.get("model")
    development_objective = development_training.get("objective")
    if not isinstance(development_model, dict) or not isinstance(
        development_objective, dict
    ):
        raise ProtocolError(
            "development training configuration is missing model or objective metadata"
        )
    selected_result = matches[0]
    expected_result_fields = {
        "hidden_dimensions": development_model.get("hidden_dimensions"),
        "objective": development_objective.get("name"),
        "representation": development_training.get("representation"),
    }
    for field, expected in expected_result_fields.items():
        if selected_result.get(field) != expected:
            raise ProtocolError(
                f"selected development experiment {field} does not match "
                "the pinned development training configuration"
            )
    interpretation = development_results.get("interpretation")
    if not isinstance(interpretation, dict):
        raise ProtocolError("development results must contain an interpretation object")
    if interpretation.get("final_claim") is not False:
        raise ProtocolError("development results must not claim a final result")
    if (
        interpretation.get("interpolation_test_status")
        != "consumed_by_earlier_model_development"
    ):
        raise ProtocolError(
            "development results must mark their interpolation test as consumed"
        )


def _validate_outputs(
    root: Path,
    outputs: dict[str, Any],
    replication_dataset: dict[str, Any],
    replication_training: dict[str, Any],
) -> None:
    resolved = {
        key: _relative_path(root, value, f"outputs.{key}")
        for key, value in outputs.items()
    }
    dataset_table = replication_dataset["dataset"]
    experiment_name = replication_training["experiment_name"]
    if resolved["dataset"].parent != root.resolve() / "data":
        raise ProtocolError("outputs.dataset must be a direct child of data/")
    if resolved["dataset"].name != dataset_table["name"]:
        raise ProtocolError("outputs.dataset must match the replication dataset name")
    if resolved["diagnostics"].parent != resolved["dataset"]:
        raise ProtocolError("outputs.diagnostics must live inside outputs.dataset")
    if resolved["diagnostics"].name != "diagnostics.json":
        raise ProtocolError("outputs.diagnostics must be named diagnostics.json")
    if resolved["unconstrained_artifact"].parent != root.resolve() / "artifacts":
        raise ProtocolError(
            "outputs.unconstrained_artifact must be a direct child of artifacts/"
        )
    if resolved["unconstrained_artifact"].name != experiment_name:
        raise ProtocolError(
            "outputs.unconstrained_artifact must match the replication experiment name"
        )
    if resolved["bounded_artifact"].parent != root.resolve() / "artifacts":
        raise ProtocolError(
            "outputs.bounded_artifact must be a direct child of artifacts/"
        )
    if resolved["bounded_artifact"] == resolved["unconstrained_artifact"]:
        raise ProtocolError("bounded and unconstrained artifact outputs must differ")
    expected_bounded_name = f"{experiment_name.removesuffix('-v1')}-bounded-v1"
    if resolved["bounded_artifact"].name != expected_bounded_name:
        raise ProtocolError(
            "outputs.bounded_artifact must be the replication experiment name "
            "with the '-bounded-v1' suffix"
        )
    for key, expected_name in (
        ("validation_report", "validation-evaluation.json"),
        ("final_report", "interpolation-test-evaluation.json"),
    ):
        if resolved[key].parent != resolved["bounded_artifact"]:
            raise ProtocolError(f"outputs.{key} must live inside the bounded artifact")
        if resolved[key].name != expected_name:
            raise ProtocolError(f"outputs.{key} must be named {expected_name}")


def validate_protocol(
    protocol_path: Path = DEFAULT_PROTOCOL,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Validate all protocol references and return a deterministic summary."""
    root = project_root.resolve()
    protocol = _load_toml(protocol_path, "replication protocol")
    actual_keys = set(protocol)
    missing = sorted(EXPECTED_TOP_LEVEL_KEYS - actual_keys)
    unknown = sorted(actual_keys - EXPECTED_TOP_LEVEL_KEYS)
    if missing:
        raise ProtocolError(f"replication protocol is missing: {', '.join(missing)}")
    if unknown:
        raise ProtocolError(
            f"replication protocol has unknown keys: {', '.join(unknown)}"
        )
    if protocol["schema_version"] != PROTOCOL_SCHEMA_VERSION:
        raise ProtocolError(
            f"schema_version must be {PROTOCOL_SCHEMA_VERSION!r}, "
            f"got {protocol['schema_version']!r}"
        )
    _require_string(protocol, "name", "protocol")
    if protocol["output_constraint"] != "european_bounds_v1":
        raise ProtocolError("protocol.output_constraint must be 'european_bounds_v1'")

    paths = _table(protocol, "paths", REFERENCE_KEYS)
    declared_hashes = _table(protocol, "sha256", REFERENCE_KEYS)
    seed_policy = _table(protocol, "seed_policy", SEED_POLICY_KEYS)
    execution = _table(protocol, "execution", EXECUTION_KEYS)
    outputs = _table(protocol, "outputs", OUTPUT_KEYS)
    resolved = _validate_hashes(root, paths, declared_hashes)

    development_results = _load_json(
        resolved["development_results"], "development results"
    )
    development_dataset = _load_toml(
        resolved["development_dataset_config"], "development dataset configuration"
    )
    development_training = _load_toml(
        resolved["development_training_config"], "development training configuration"
    )
    replication_dataset = _load_toml(
        resolved["replication_dataset_config"], "replication dataset configuration"
    )
    replication_training = _load_toml(
        resolved["replication_training_config"], "replication training configuration"
    )
    acceptance = _load_toml(
        resolved["acceptance_config"], "acceptance configuration"
    )
    _validate_controlled_changes(
        development_dataset,
        replication_dataset,
        development_training,
        replication_training,
    )
    dataset_seed, training_seed = _validate_seed_policy(
        seed_policy, replication_dataset, replication_training
    )
    _validate_execution(
        protocol,
        paths,
        acceptance,
        development_results,
        development_training,
        execution,
    )
    _validate_outputs(root, outputs, replication_dataset, replication_training)

    return {
        "dataset_seed": dataset_seed,
        "final_partition": execution["final_partition"],
        "maximum_final_evaluations": execution["maximum_final_evaluations"],
        "name": protocol["name"],
        "output_constraint": protocol["output_constraint"],
        "status": "valid",
        "training_seed": training_seed,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        summary = validate_protocol(arguments.protocol)
    except ProtocolError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
