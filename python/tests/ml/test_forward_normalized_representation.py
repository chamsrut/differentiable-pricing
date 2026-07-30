"""Focused tests for the forward-normalized price representation."""

from __future__ import annotations

import json
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
from differentiable_pricing.ml.artifact import (
    ARTIFACT_MANIFEST,
    ArtifactError,
    load_physical_model,
)
from differentiable_pricing.ml.config import (
    FEATURE_ORDER,
    FORWARD_NORMALIZED_REPRESENTATION,
    RAW_PHYSICAL_REPRESENTATION,
    TrainingConfigError,
    load_training_config,
)
from differentiable_pricing.ml.evaluate import evaluate_artifact, predict_with_greeks
from differentiable_pricing.ml.model import (
    PhysicalPriceModel,
    PricingMlp,
    Scaling,
    representation_arrays,
)
from differentiable_pricing.ml.train import train_to_artifact

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATASET_CONFIG = f"""
schema_version = "{SCHEMA_VERSION}"
generator_version = "{GENERATOR_VERSION}"

[dataset]
name = "forward-representation-test"
oracle = "dp::black_scholes"
base_seed = 24680

[dataset.rows]
train = 48
validation = 16
interpolation_test = 16

[domain]
spot = [50.0, 150.0]
log_forward_moneyness = [-0.4, 0.4]
maturity = [0.019178082191780823, 2.0]
rate = [-0.01, 0.08]
dividend_yield = [0.0, 0.05]
volatility = [0.05, 0.8]
"""


def training_config(representation: str | None) -> str:
    representation_line = (
        "" if representation is None else f'representation = "{representation}"\n'
    )
    return f"""
schema_version = "european-neural-baseline-config/v1"
experiment_name = "forward-representation-test"
{representation_line}features = [
  "option_type",
  "spot",
  "strike",
  "maturity",
  "rate",
  "dividend_yield",
  "volatility",
]
dtype = "float64"
device = "cpu"

[model]
hidden_dimensions = [8, 8]
activation = "tanh"
output_dimension = 1

[optimizer]
name = "adamw"
learning_rate = 0.001
weight_decay = 0.000001

[training]
seed = 20260730
batch_size = 16
max_epochs = 5
early_stopping_patience = 3
early_stopping_min_delta = 0.0
num_threads = 1

[artifact]
schema_version = "european-neural-artifact/v1"

[evaluation]
batch_size = 16
core_max_absolute_z = 4.0
tail_max_absolute_z = 8.0
"""


def write_config(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.write_text(text, encoding="utf-8")
    return path


def build_dataset(root: Path) -> Path:
    config = write_config(root, "dataset.toml", DATASET_CONFIG)
    output = root / "dataset"
    generate_dataset(load_dataset_config(config), output)
    return output


def test_forward_config_changes_only_the_financial_representation() -> None:
    long = load_training_config(
        PROJECT_ROOT / "configs/european_neural_long_v1.toml"
    )
    forward = load_training_config(
        PROJECT_ROOT / "configs/european_neural_forward_normalized_v1.toml"
    )

    assert long.representation == RAW_PHYSICAL_REPRESENTATION
    assert forward.representation == FORWARD_NORMALIZED_REPRESENTATION
    assert forward.model == long.model
    assert forward.optimizer == long.optimizer
    assert forward.training == long.training
    assert forward.evaluation == long.evaluation


def test_unknown_representation_is_rejected(tmp_path: Path) -> None:
    path = write_config(
        tmp_path,
        "unknown.toml",
        training_config("unknown"),
    )
    with pytest.raises(TrainingConfigError, match="representation must be one of"):
        load_training_config(path)


def test_forward_representation_collapses_homogeneous_states() -> None:
    features = np.array(
        [
            [1.0, 100.0, 90.0, 1.2, 0.04, 0.01, 0.25],
            [1.0, 250.0, 225.0, 1.2, 0.04, 0.01, 0.25],
        ],
        dtype=np.float64,
    )
    prices = np.array([12.0, 30.0], dtype=np.float64)

    network_features, targets = representation_arrays(
        features,
        prices,
        FORWARD_NORMALIZED_REPRESENTATION,
    )

    np.testing.assert_allclose(network_features[0], network_features[1])
    np.testing.assert_allclose(targets[0], targets[1])


def test_forward_physical_reconstruction_has_exact_chain_rule() -> None:
    scaling = Scaling(np.zeros(3), np.ones(3), 0.1, 1.0)
    network = PricingMlp(3, ())
    with torch.no_grad():
        layer = network.layers[0]
        assert isinstance(layer, torch.nn.Linear)
        layer.weight.zero_()
        layer.bias.zero_()
    model = PhysicalPriceModel(
        network,
        scaling,
        FORWARD_NORMALIZED_REPRESENTATION,
    )
    features = np.array(
        [[1.0, 100.0, 95.0, 2.0, 0.03, 0.01, 0.2]],
        dtype=np.float64,
    )

    result = predict_with_greeks(model, features, batch_size=1)
    discounted_spot = 100.0 * np.exp(-0.01 * 2.0)
    assert result["price"][0] == pytest.approx(0.1 * discounted_spot)
    assert result["delta"][0] == pytest.approx(0.1 * np.exp(-0.01 * 2.0))
    assert result["gamma"][0] == pytest.approx(0.0, abs=1e-14)
    assert result["vega"][0] == pytest.approx(0.0, abs=1e-14)
    assert result["rho"][0] == pytest.approx(0.0, abs=1e-14)
    assert result["theta"][0] == pytest.approx(0.01 * 0.1 * discounted_spot)


def test_forward_representation_greeks_match_finite_differences() -> None:
    torch.manual_seed(11)
    scaling = Scaling(np.zeros(3), np.ones(3), 0.2, 0.4)
    model = PhysicalPriceModel(
        PricingMlp(3, (5,)),
        scaling,
        FORWARD_NORMALIZED_REPRESENTATION,
    )
    features = np.array(
        [[1.0, 105.0, 98.0, 0.8, 0.03, 0.015, 0.25]],
        dtype=np.float64,
    )
    result = predict_with_greeks(model, features, batch_size=1)
    comparisons = {
        "delta": ("spot", 1.0e-4, 1.0),
        "vega": ("volatility", 1.0e-5, 1.0),
        "rho": ("rate", 1.0e-6, 1.0),
        "theta": ("maturity", 1.0e-6, -1.0),
    }
    for greek, (feature, step, sign) in comparisons.items():
        index = FEATURE_ORDER.index(feature)
        plus = features.copy()
        minus = features.copy()
        plus[0, index] += step
        minus[0, index] -= step
        plus_price = predict_with_greeks(model, plus, batch_size=1)["price"][0]
        minus_price = predict_with_greeks(model, minus, batch_size=1)["price"][0]
        finite_difference = sign * (plus_price - minus_price) / (2.0 * step)
        assert result[greek][0] == pytest.approx(
            finite_difference,
            rel=2e-7,
            abs=2e-9,
        )

    spot_step = 1.0e-3
    plus = features.copy()
    minus = features.copy()
    spot_index = FEATURE_ORDER.index("spot")
    plus[0, spot_index] += spot_step
    minus[0, spot_index] -= spot_step
    plus_delta = predict_with_greeks(model, plus, batch_size=1)["delta"][0]
    minus_delta = predict_with_greeks(model, minus, batch_size=1)["delta"][0]
    finite_gamma = (plus_delta - minus_delta) / (2.0 * spot_step)
    assert result["gamma"][0] == pytest.approx(
        finite_gamma,
        rel=2e-6,
        abs=2e-8,
    )


def test_forward_artifact_round_trip_and_legacy_raw_loading(tmp_path: Path) -> None:
    dataset = build_dataset(tmp_path)
    forward_artifact = tmp_path / "forward-artifact"
    payload = train_to_artifact(
        dataset,
        write_config(
            tmp_path,
            "forward.toml",
            training_config(FORWARD_NORMALIZED_REPRESENTATION),
        ),
        forward_artifact,
    )

    assert payload["representation"]["name"] == FORWARD_NORMALIZED_REPRESENTATION
    assert payload["architecture"]["input_dimension"] == 3
    model, loaded = load_physical_model(forward_artifact)
    assert model.representation == FORWARD_NORMALIZED_REPRESENTATION
    assert loaded == payload
    report = evaluate_artifact(
        dataset,
        forward_artifact,
        forward_artifact / "validation-evaluation.json",
        partition="validation",
    )
    assert report["partition_role"] == "model_selection"
    assert np.isfinite(
        report["slices"]["overall"]["metrics"]["price"]["rmse"]
    )

    raw_artifact = tmp_path / "raw-artifact"
    train_to_artifact(
        dataset,
        write_config(tmp_path, "raw.toml", training_config(None)),
        raw_artifact,
    )
    legacy_payload = json.loads(
        (raw_artifact / ARTIFACT_MANIFEST).read_text(encoding="utf-8")
    )
    legacy_payload.pop("representation")
    legacy_payload["architecture"].pop("input_dimension")
    (raw_artifact / ARTIFACT_MANIFEST).write_text(
        json.dumps(legacy_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    legacy_model, _ = load_physical_model(raw_artifact)
    assert legacy_model.representation == RAW_PHYSICAL_REPRESENTATION

    payload["representation"]["network_feature_order"] = ["wrong"]
    (forward_artifact / ARTIFACT_MANIFEST).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ArtifactError, match="network_feature_order"):
        load_physical_model(forward_artifact)
