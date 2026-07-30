"""The fresh-seed replication protocol must stay locked and self-consistent."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "check_european_replication_protocol.py"
PROTOCOL = PROJECT_ROOT / "configs" / "european_neural_replication_protocol_v1.toml"


def load_protocol_module() -> Any:
    spec = importlib.util.spec_from_file_location("dp_check_replication_protocol", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check = load_protocol_module()


@pytest.fixture
def replication_root(tmp_path: Path) -> Path:
    with PROTOCOL.open("rb") as stream:
        protocol = tomllib.load(stream)
    relative_paths = [
        Path("configs/european_neural_replication_protocol_v1.toml"),
        *(Path(value) for value in protocol["paths"].values()),
    ]
    for relative in relative_paths:
        source = PROJECT_ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return tmp_path


def protocol_path(root: Path) -> Path:
    return root / "configs" / "european_neural_replication_protocol_v1.toml"


def replace(path: Path, old: str, new: str) -> None:
    contents = path.read_text(encoding="utf-8")
    assert old in contents
    path.write_text(contents.replace(old, new, 1), encoding="utf-8")


def refresh_digest(root: Path, key: str) -> None:
    path = protocol_path(root)
    with path.open("rb") as stream:
        protocol = tomllib.load(stream)
    referenced = root / protocol["paths"][key]
    old = protocol["sha256"][key]
    new = hashlib.sha256(referenced.read_bytes()).hexdigest()
    replace(path, f'{key} = "{old}"', f'{key} = "{new}"')


def validate(root: Path) -> dict[str, Any]:
    return check.validate_protocol(protocol_path(root), project_root=root)


def test_checked_in_replication_protocol_is_valid() -> None:
    summary = check.validate_protocol()
    assert summary == {
        "dataset_seed": 709748617,
        "final_partition": "interpolation_test",
        "maximum_final_evaluations": 1,
        "name": "european-neural-replication-v1",
        "output_constraint": "european_bounds_v1",
        "status": "valid",
        "training_seed": 2240113204,
    }


def test_changed_reference_without_digest_update_is_rejected(
    replication_root: Path,
) -> None:
    training = (
        replication_root / "configs" / "european_neural_replication_v1.toml"
    )
    replace(training, "batch_size = 4096", "batch_size = 2048")
    with pytest.raises(check.ProtocolError, match="does not match"):
        validate(replication_root)


def test_dataset_may_change_only_identity_and_seed(replication_root: Path) -> None:
    dataset = (
        replication_root
        / "configs"
        / "european_option_dataset_replication_v1.toml"
    )
    replace(dataset, "validation = 25000", "validation = 24000")
    refresh_digest(replication_root, "replication_dataset_config")
    with pytest.raises(check.ProtocolError, match="dataset configuration may differ"):
        validate(replication_root)


def test_dataset_identity_must_be_fresh(replication_root: Path) -> None:
    dataset = (
        replication_root
        / "configs"
        / "european_option_dataset_replication_v1.toml"
    )
    replace(
        dataset,
        'name = "european-option-replication-v1"',
        'name = "european-option-v1"',
    )
    refresh_digest(replication_root, "replication_dataset_config")
    with pytest.raises(check.ProtocolError, match="fresh dataset identity"):
        validate(replication_root)


def test_training_may_change_only_identity_and_seed(replication_root: Path) -> None:
    training = (
        replication_root / "configs" / "european_neural_replication_v1.toml"
    )
    replace(training, "learning_rate = 0.001", "learning_rate = 0.0005")
    refresh_digest(replication_root, "replication_training_config")
    with pytest.raises(check.ProtocolError, match="training configuration may differ"):
        validate(replication_root)


def test_experiment_identity_must_be_fresh(replication_root: Path) -> None:
    training = (
        replication_root / "configs" / "european_neural_replication_v1.toml"
    )
    replace(
        training,
        'experiment_name = "european-neural-forward-differential-replication-v1"',
        'experiment_name = "european-neural-forward-differential-v1"',
    )
    refresh_digest(replication_root, "replication_training_config")
    with pytest.raises(check.ProtocolError, match="fresh experiment identity"):
        validate(replication_root)


def test_seed_must_follow_public_derivation(replication_root: Path) -> None:
    replace(
        protocol_path(replication_root),
        "dataset_seed = 709748617",
        "dataset_seed = 709748618",
    )
    with pytest.raises(check.ProtocolError, match="derives 709748617"):
        validate(replication_root)


def test_acceptance_partitions_must_match_execution(replication_root: Path) -> None:
    acceptance = (
        replication_root / "configs" / "european_neural_acceptance_v1.toml"
    )
    replace(
        acceptance,
        'selection_partition = "validation"',
        'selection_partition = "interpolation_test"',
    )
    refresh_digest(replication_root, "acceptance_config")
    with pytest.raises(check.ProtocolError, match="selection_partition does not match"):
        validate(replication_root)


def test_development_results_must_reference_pinned_acceptance(
    replication_root: Path,
) -> None:
    results = (
        replication_root / "docs" / "results" / "european_validation_results_v1.json"
    )
    payload = json.loads(results.read_text(encoding="utf-8"))
    payload["acceptance_config"] = "configs/other.toml"
    results.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    refresh_digest(replication_root, "development_results")
    with pytest.raises(check.ProtocolError, match="pinned acceptance"):
        validate(replication_root)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("hidden_dimensions", [32, 32, 32]),
        ("objective", "price_only_v1"),
        ("representation", "raw_physical_v1"),
    ],
)
def test_selected_result_must_match_pinned_training_design(
    replication_root: Path,
    field: str,
    replacement: object,
) -> None:
    results = (
        replication_root / "docs" / "results" / "european_validation_results_v1.json"
    )
    payload = json.loads(results.read_text(encoding="utf-8"))
    selected = next(
        experiment
        for experiment in payload["experiments"]
        if experiment["id"] == "bounded_differential"
    )
    selected[field] = replacement
    results.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    refresh_digest(replication_root, "development_results")
    with pytest.raises(check.ProtocolError, match=rf"{field} does not match"):
        validate(replication_root)


def test_selected_development_experiment_must_exist(replication_root: Path) -> None:
    replace(
        protocol_path(replication_root),
        'selected_development_experiment = "bounded_differential"',
        'selected_development_experiment = "missing"',
    )
    with pytest.raises(check.ProtocolError, match="identify exactly one"):
        validate(replication_root)


def test_output_paths_cannot_escape_the_repository(replication_root: Path) -> None:
    replace(
        protocol_path(replication_root),
        'dataset = "data/european-option-replication-v1"',
        'dataset = "../data/european-option-replication-v1"',
    )
    with pytest.raises(check.ProtocolError, match="must not be absolute"):
        validate(replication_root)


def test_unconstrained_artifact_name_must_match_replication_identity(
    replication_root: Path,
) -> None:
    replace(
        protocol_path(replication_root),
        'unconstrained_artifact = '
        '"artifacts/european-neural-forward-differential-replication-v1"',
        'unconstrained_artifact = '
        '"artifacts/european-neural-forward-differential-v1"',
    )
    with pytest.raises(
        check.ProtocolError,
        match="must match the replication experiment name",
    ):
        validate(replication_root)


def test_bounded_artifact_must_differ_from_unconstrained(
    replication_root: Path,
) -> None:
    replace(
        protocol_path(replication_root),
        'bounded_artifact = '
        '"artifacts/european-neural-forward-differential-replication-bounded-v1"',
        'bounded_artifact = '
        '"artifacts/european-neural-forward-differential-replication-v1"',
    )
    with pytest.raises(
        check.ProtocolError,
        match="bounded and unconstrained artifact outputs must differ",
    ):
        validate(replication_root)


def test_bounded_artifact_name_must_derive_from_replication_identity(
    replication_root: Path,
) -> None:
    replace(
        protocol_path(replication_root),
        'bounded_artifact = '
        '"artifacts/european-neural-forward-differential-replication-bounded-v1"',
        'bounded_artifact = '
        '"artifacts/european-neural-forward-differential-bounded-v1"',
    )
    with pytest.raises(check.ProtocolError, match="with the '-bounded-v1' suffix"):
        validate(replication_root)


def test_only_one_final_evaluation_is_allowed(replication_root: Path) -> None:
    replace(
        protocol_path(replication_root),
        "maximum_final_evaluations = 1",
        "maximum_final_evaluations = 2",
    )
    with pytest.raises(check.ProtocolError, match="maximum_final_evaluations must be 1"):
        validate(replication_root)


def test_cli_reports_valid_protocol_as_json() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert completed.stderr == ""
    assert json.loads(completed.stdout)["status"] == "valid"


def test_cli_returns_two_for_invalid_protocol(tmp_path: Path) -> None:
    broken = tmp_path / "broken.toml"
    contents = PROTOCOL.read_text(encoding="utf-8").replace(
        'schema_version = "european-neural-replication-protocol/v1"',
        'schema_version = "european-neural-replication-protocol/invalid"',
        1,
    )
    broken.write_text(contents, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--protocol", str(broken)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert completed.stderr.startswith("error: schema_version must be")
