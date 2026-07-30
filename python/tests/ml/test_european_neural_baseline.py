"""Focused tests for stage-1 price training and physical-unit autograd Greeks."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch
from differentiable_pricing.data.config import (
    GENERATOR_VERSION,
    SCHEMA_VERSION,
    load_dataset_config,
)
from differentiable_pricing.data.generate import generate_dataset
from differentiable_pricing.ml import dataset as dataset_module
from differentiable_pricing.ml.artifact import (
    ARTIFACT_MANIFEST,
    CHECKPOINT_WARNING,
    ArtifactError,
    load_physical_model,
)
from differentiable_pricing.ml.config import (
    ARTIFACT_SCHEMA_VERSION,
    CONFIG_SCHEMA_VERSION,
    FEATURE_ORDER,
    TrainingConfigError,
    load_training_config,
)
from differentiable_pricing.ml.dataset import DatasetLoadError, SplitData
from differentiable_pricing.ml.evaluate import (
    EVALUATION_SCHEMA_VERSION,
    MATERIAL_ARBITRAGE_RELATIVE_TOLERANCE,
    EvaluationError,
    error_metrics,
    evaluate_artifact,
    moneyness_masks,
    no_arbitrage_violations,
    predict_with_greeks,
)
from differentiable_pricing.ml.evaluate import (
    main as evaluate_main,
)
from differentiable_pricing.ml.model import PhysicalPriceModel, PricingMlp, Scaling
from differentiable_pricing.ml.train import fit_model, train_to_artifact
from differentiable_pricing.ml.train import main as train_main

DATASET_CONFIG = f"""
schema_version = "{SCHEMA_VERSION}"
generator_version = "{GENERATOR_VERSION}"

[dataset]
name = "european-neural-test"
oracle = "dp::black_scholes"
base_seed = 13579

[dataset.rows]
train = 96
validation = 32
interpolation_test = 32

[domain]
spot = [50.0, 150.0]
log_forward_moneyness = [-0.4, 0.4]
maturity = [0.019178082191780823, 2.0]
rate = [-0.01, 0.08]
dividend_yield = [0.0, 0.05]
volatility = [0.05, 0.8]
"""

TRAINING_CONFIG = f"""
schema_version = "{CONFIG_SCHEMA_VERSION}"
experiment_name = "european-neural-test"
features = {json.dumps(list(FEATURE_ORDER))}
dtype = "float64"
device = "cpu"

[model]
hidden_dimensions = [8, 8]
activation = "tanh"
output_dimension = 1

[optimizer]
name = "adamw"
learning_rate = 0.01
weight_decay = 0.0

[training]
seed = 24680
batch_size = 32
max_epochs = 6
early_stopping_patience = 3
early_stopping_min_delta = 0.0
num_threads = 1

[artifact]
schema_version = "{ARTIFACT_SCHEMA_VERSION}"

[evaluation]
batch_size = 16
core_max_absolute_z = 4.0
tail_max_absolute_z = 8.0
"""

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def write_training_config(root: Path, text: str = TRAINING_CONFIG) -> Path:
    path = root / "training.toml"
    path.write_text(text, encoding="utf-8")
    return path


def build_dataset(root: Path) -> Path:
    config_path = root / "dataset.toml"
    config_path.write_text(DATASET_CONFIG, encoding="utf-8")
    output = root / "dataset"
    generate_dataset(load_dataset_config(config_path), output)
    return output


def synthetic_split(name: str, shift: float = 0.0) -> SplitData:
    rows = 24
    index = np.arange(rows, dtype=np.float64)
    option_types = np.where(index.astype(int) % 2 == 0, "call", "put")
    features = np.column_stack(
        [
            np.where(option_types == "call", 1.0, -1.0),
            80.0 + index + shift,
            90.0 + 0.5 * index,
            0.2 + index / 30.0,
            0.01 + index / 1000.0,
            0.005 + index / 2000.0,
            0.1 + index / 100.0,
        ]
    )
    price = (
        0.1 * features[:, 1]
        - 0.05 * features[:, 2]
        + 2.0 * features[:, 3]
        + features[:, 0]
    )
    return SplitData(
        split=name,
        features=features,
        option_types=option_types,
        columns={"price": price},
    )


def test_pinned_training_configuration_loads(tmp_path: Path) -> None:
    config = load_training_config(write_training_config(tmp_path))
    assert config.features == FEATURE_ORDER
    assert config.dtype == "float64"
    assert config.model.hidden_dimensions == (8, 8)
    assert len(config.sha256) == 64


def test_long_and_wide_configs_are_controlled_experiments() -> None:
    baseline = load_training_config(
        PROJECT_ROOT / "configs/european_neural_baseline_v1.toml"
    )
    long = load_training_config(
        PROJECT_ROOT / "configs/european_neural_long_v1.toml"
    )
    wide = load_training_config(
        PROJECT_ROOT / "configs/european_neural_wide_v2.toml"
    )

    assert long.model == baseline.model
    assert long.optimizer == baseline.optimizer == wide.optimizer
    assert wide.training == long.training
    assert long.training.seed == baseline.training.seed == wide.training.seed
    assert long.training.batch_size == baseline.training.batch_size == wide.training.batch_size
    assert (
        long.training.early_stopping_min_delta
        == baseline.training.early_stopping_min_delta
    )
    assert long.training.num_threads == baseline.training.num_threads
    assert long.training.max_epochs == wide.training.max_epochs == 300
    assert long.training.early_stopping_patience == wide.training.early_stopping_patience == 25
    assert wide.model.hidden_dimensions == (128, 128, 128, 128)


def test_acceptance_gates_are_versioned_and_match_evaluation() -> None:
    path = PROJECT_ROOT / "configs/european_neural_acceptance_v1.toml"
    with path.open("rb") as stream:
        gates = tomllib.load(stream)

    assert gates["selection_partition"] == "validation"
    assert gates["final_partition"] == "interpolation_test"
    assert (
        gates["no_arbitrage"]["material_relative_tolerance"]
        == MATERIAL_ARBITRAGE_RELATIVE_TOLERANCE
    )
    assert gates["no_arbitrage"]["material_violations_max"] == 0


def test_unknown_or_incompatible_configuration_is_rejected(tmp_path: Path) -> None:
    unknown = TRAINING_CONFIG.replace('device = "cpu"', 'device = "cpu"\nsurprise = true')
    with pytest.raises(TrainingConfigError, match="unknown key"):
        load_training_config(write_training_config(tmp_path, unknown))

    relu = TRAINING_CONFIG.replace('activation = "tanh"', 'activation = "relu"')
    with pytest.raises(TrainingConfigError, match="tanh"):
        load_training_config(write_training_config(tmp_path, relu))


def test_transforms_are_fit_from_training_only(tmp_path: Path) -> None:
    config = load_training_config(write_training_config(tmp_path))
    train = synthetic_split("train")
    validation = synthetic_split("validation", shift=10_000.0)
    result = fit_model(train, validation, config)

    np.testing.assert_array_equal(result.scaling.feature_mean, train.features.mean(axis=0))
    assert result.scaling.price_mean == pytest.approx(train.columns["price"].mean())
    assert result.scaling.feature_mean[1] != pytest.approx(validation.features[:, 1].mean())


def test_tiny_training_is_deterministic(tmp_path: Path) -> None:
    config = load_training_config(write_training_config(tmp_path))
    train = synthetic_split("train")
    validation = synthetic_split("validation", shift=1.0)
    first = fit_model(train, validation, config)
    second = fit_model(train, validation, config)

    assert first.history == second.history
    assert first.best_epoch == second.best_epoch
    for name, value in first.network.state_dict().items():
        torch.testing.assert_close(value, second.network.state_dict()[name], rtol=0.0, atol=0.0)


def test_training_never_hashes_or_opens_interpolation_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = build_dataset(tmp_path)
    config = write_training_config(tmp_path)
    visited: list[str] = []
    original = dataset_module.sha256_file

    def recording_hash(path: Path) -> str:
        visited.append(path.name)
        if path.name == "interpolation_test.parquet":
            raise AssertionError("training touched the held-out partition")
        return original(path)

    monkeypatch.setattr(dataset_module, "sha256_file", recording_hash)
    payload = train_to_artifact(dataset, config, tmp_path / "artifact")

    assert visited == ["train.parquet", "validation.parquet"]
    assert payload["training"]["partitions_read"] == ["train", "validation"]
    assert payload["training"]["held_out_partition"] == "interpolation_test"


def test_training_rejects_a_hash_mismatched_partition(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path)
    with (dataset / "train.parquet").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(DatasetLoadError, match="sha256 mismatch"):
        train_to_artifact(
            dataset,
            write_training_config(tmp_path),
            tmp_path / "artifact",
        )


def linear_physical_model() -> tuple[PhysicalPriceModel, Scaling, np.ndarray]:
    scaling = Scaling(
        feature_mean=np.array([0.0, 100.0, 100.0, 1.0, 0.02, 0.01, 0.2]),
        feature_scale=np.array([1.0, 20.0, 25.0, 0.5, 0.03, 0.02, 0.1]),
        price_mean=10.0,
        price_scale=4.0,
    )
    network = PricingMlp(len(FEATURE_ORDER), ())
    weights = np.arange(1.0, len(FEATURE_ORDER) + 1.0, dtype=np.float64)
    with torch.no_grad():
        layer = network.layers[0]
        assert isinstance(layer, torch.nn.Linear)
        layer.weight.copy_(torch.as_tensor(weights.reshape(1, -1)))
        layer.bias.fill_(0.5)
    features = np.array(
        [[1.0, 105.0, 98.0, 0.8, 0.03, 0.015, 0.25]],
        dtype=np.float64,
    )
    return PhysicalPriceModel(network, scaling), scaling, features


def test_physical_chain_rule_and_theta_sign() -> None:
    model, scaling, features = linear_physical_model()
    result = predict_with_greeks(model, features, batch_size=1)
    weights = np.arange(1.0, len(FEATURE_ORDER) + 1.0)

    assert result["delta"][0] == pytest.approx(
        scaling.price_scale * weights[1] / scaling.feature_scale[1]
    )
    assert result["vega"][0] == pytest.approx(
        scaling.price_scale * weights[6] / scaling.feature_scale[6]
    )
    assert result["rho"][0] == pytest.approx(
        scaling.price_scale * weights[4] / scaling.feature_scale[4]
    )
    assert result["theta"][0] == pytest.approx(
        -scaling.price_scale * weights[3] / scaling.feature_scale[3]
    )
    assert result["gamma"][0] == pytest.approx(0.0, abs=1e-14)


def test_autograd_delta_and_gamma_match_network_finite_differences() -> None:
    torch.manual_seed(7)
    scaling = Scaling(np.zeros(7), np.ones(7), 2.0, 3.0)
    model = PhysicalPriceModel(PricingMlp(7, (4,)), scaling)
    # Small but financially valid magnitudes: the identity scaling above feeds
    # these to the network unchanged, and large values would saturate tanh and
    # blunt the finite-difference comparison.
    features = np.array([[1.0, 1.2, 1.1, 0.7, 0.03, 0.01, 0.25]])
    result = predict_with_greeks(model, features, batch_size=1)
    spot_index = FEATURE_ORDER.index("spot")
    step = 1.0e-4
    plus = features.copy()
    minus = features.copy()
    plus[0, spot_index] += step
    minus[0, spot_index] -= step
    plus_result = predict_with_greeks(model, plus, batch_size=1)
    minus_result = predict_with_greeks(model, minus, batch_size=1)

    finite_delta = (plus_result["price"][0] - minus_result["price"][0]) / (2.0 * step)
    finite_gamma = (plus_result["delta"][0] - minus_result["delta"][0]) / (2.0 * step)
    assert result["delta"][0] == pytest.approx(finite_delta, rel=1e-8, abs=1e-10)
    assert result["gamma"][0] == pytest.approx(finite_gamma, rel=1e-7, abs=1e-9)


def metric_split() -> SplitData:
    features = np.array(
        [
            [1.0, 100.0, 100.0, 1.0, 0.0, 0.0, 1.0],
            [-1.0, 100.0, 100.0, 1.0, 0.0, 0.0, 1.0],
            [1.0, 100.0, 100.0, 1.0, 0.0, 0.0, 1.0],
        ]
    )
    columns = {
        "spot": features[:, 1],
        "strike": features[:, 2],
        "maturity": features[:, 3],
        "rate": features[:, 4],
        "dividend_yield": features[:, 5],
        "volatility": features[:, 6],
        "log_forward_moneyness": np.array([1.0, 5.0, 9.0]),
        "price": np.array([10.0, 10.0, 10.0]),
    }
    return SplitData(
        split="interpolation_test",
        features=features,
        option_types=np.array(["call", "put", "call"]),
        columns=columns,
    )


def test_metrics_and_moneyness_stratification() -> None:
    split = metric_split()
    masks = moneyness_masks(split, core_max=4.0, tail_max=8.0)
    assert {name: int(mask.sum()) for name, mask in masks.items()} == {
        "core": 1,
        "tail": 1,
        "extreme": 1,
    }
    metrics = error_metrics(np.array([1.0, 2.0]), np.array([2.0, 4.0]))
    assert metrics["mae"] == pytest.approx(1.5)
    assert metrics["rmse"] == pytest.approx(np.sqrt(2.5))
    assert metrics["maximum_absolute_error"] == 2.0


def test_learned_price_bound_violations_are_counted() -> None:
    split = metric_split()
    result = no_arbitrage_violations(split, np.array([-1.0, 101.0, 10.0]))
    assert result["below_lower_bound"] == 1
    assert result["above_upper_bound"] == 1
    assert result["total_violations"] == 2
    assert result["material_violations"] == 2


def test_artifact_integrity_and_safe_weight_format(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path)
    artifact = tmp_path / "artifact"
    config = write_training_config(tmp_path)
    train_to_artifact(dataset, config, artifact)
    repeated = tmp_path / "artifact-repeated"
    train_to_artifact(dataset, config, repeated)
    assert (artifact / "weights.npz").read_bytes() == (
        repeated / "weights.npz"
    ).read_bytes()
    assert (artifact / ARTIFACT_MANIFEST).read_bytes() == (
        repeated / ARTIFACT_MANIFEST
    ).read_bytes()
    payload = json.loads((artifact / ARTIFACT_MANIFEST).read_text(encoding="utf-8"))
    assert payload["checkpoint_security"] == CHECKPOINT_WARNING
    assert payload["weights"]["format"].endswith("allow_pickle=False")
    load_physical_model(artifact)

    payload["weights"]["file"] = "untrusted.pt"
    (artifact / ARTIFACT_MANIFEST).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ArtifactError, match="weights file"):
        load_physical_model(artifact)

    payload["weights"]["file"] = "weights.npz"
    (artifact / ARTIFACT_MANIFEST).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (artifact / "weights.npz").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ArtifactError, match="sha256 mismatch"):
        load_physical_model(artifact)


def test_evaluation_is_separate_and_links_dataset_and_artifact(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path)
    artifact = tmp_path / "artifact"
    train_to_artifact(dataset, write_training_config(tmp_path), artifact)
    output = artifact / "evaluation.json"
    report = evaluate_artifact(
        dataset,
        artifact,
        output,
        partition="interpolation_test",
    )

    assert report["schema_version"] == EVALUATION_SCHEMA_VERSION
    assert report["partition"] == "interpolation_test"
    assert report["partition_role"] == "locked_final_evaluation"
    assert report["slices"]["overall"]["rows"] == 32
    assert report["artifact"]["weights_sha256"]
    assert report["dataset"]["manifest_sha256"]
    assert report["definitions"]["theta"].startswith("-d learned price")
    assert json.loads(output.read_text(encoding="utf-8")) == report
    second_output = artifact / "evaluation-second.json"
    evaluate_artifact(
        dataset,
        artifact,
        second_output,
        partition="interpolation_test",
    )
    assert output.read_bytes() == second_output.read_bytes()


def test_evaluation_requires_a_declared_partition(tmp_path: Path) -> None:
    with pytest.raises(EvaluationError, match="partition must be one of"):
        evaluate_artifact(
            tmp_path / "dataset",
            tmp_path / "artifact",
            tmp_path / "evaluation.json",
            partition="train",
        )


def test_validation_evaluation_does_not_touch_interpolation_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = build_dataset(tmp_path)
    artifact = tmp_path / "artifact"
    train_to_artifact(dataset, write_training_config(tmp_path), artifact)
    visited: list[str] = []
    original = dataset_module.sha256_file

    def recording_hash(path: Path) -> str:
        visited.append(path.name)
        if path.name == "interpolation_test.parquet":
            raise AssertionError("validation evaluation touched the locked test")
        return original(path)

    monkeypatch.setattr(dataset_module, "sha256_file", recording_hash)
    report = evaluate_artifact(
        dataset,
        artifact,
        artifact / "validation-evaluation.json",
        partition="validation",
    )

    assert visited == ["validation.parquet"]
    assert report["partition"] == "validation"
    assert report["partition_role"] == "model_selection"
    assert report["slices"]["overall"]["rows"] == 32


def test_evaluation_rejects_the_wrong_dataset(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    first_root.mkdir()
    first = build_dataset(first_root)
    artifact = tmp_path / "artifact"
    train_to_artifact(first, write_training_config(tmp_path), artifact)

    second_root = tmp_path / "second"
    second_root.mkdir()
    altered = DATASET_CONFIG.replace("base_seed = 13579", "base_seed = 13580")
    config_path = second_root / "dataset.toml"
    config_path.write_text(altered, encoding="utf-8")
    second = second_root / "dataset"
    generate_dataset(load_dataset_config(config_path), second)
    with pytest.raises(EvaluationError, match="manifest SHA-256"):
        evaluate_artifact(
            second,
            artifact,
            tmp_path / "evaluation.json",
            partition="interpolation_test",
        )


def test_command_lines_report_contract_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    training_status = train_main(
        [
            "--dataset",
            str(tmp_path / "missing-dataset"),
            "--config",
            str(tmp_path / "missing-config"),
            "--output",
            str(tmp_path / "artifact"),
        ]
    )
    assert training_status == 2
    assert "error:" in capsys.readouterr().err

    evaluation_status = evaluate_main(
        [
            "--dataset",
            str(tmp_path / "missing-dataset"),
            "--artifact",
            str(tmp_path / "missing-artifact"),
            "--partition",
            "validation",
            "--output",
            str(tmp_path / "evaluation.json"),
        ]
    )
    assert evaluation_status == 2
    assert "error:" in capsys.readouterr().err
