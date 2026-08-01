"""Tests for first-derivative supervision in economic coordinates."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from differentiable_pricing.data.config import (
    GENERATOR_VERSION,
    SCHEMA_VERSION,
    load_dataset_config,
)
from differentiable_pricing.data.generate import generate_dataset, sha256_file
from differentiable_pricing.ml import dataset as dataset_module
from differentiable_pricing.ml.artifact import (
    ARTIFACT_MANIFEST,
    ArtifactError,
    load_physical_model,
)
from differentiable_pricing.ml.config import (
    FORWARD_NORMALIZED_REPRESENTATION,
    TrainingConfigError,
    load_training_config,
)
from differentiable_pricing.ml.constrain import (
    ConstraintError,
    derive_bounded_artifact,
)
from differentiable_pricing.ml.constrain import main as constrain_main
from differentiable_pricing.ml.evaluate import (
    EvaluationError,
    evaluate_artifact,
    predict_with_greeks,
)
from differentiable_pricing.ml.model import (
    EUROPEAN_BOUNDS_CONSTRAINT,
    NO_OUTPUT_CONSTRAINT,
    PhysicalPriceModel,
    PricingMlp,
    Scaling,
    forward_normalized_derivative_targets,
)
from differentiable_pricing.ml.train import train_to_artifact

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FLOAT64_EPS = np.finfo(np.float64).eps
# Cross-library bit identity is not required and must not be asserted. NumPy and
# PyTorch evaluate exp() through different vectorized libm kernels, so the
# discounted operands S*exp(-qT) and K*exp(-rT) may disagree in their last bit
# across platforms and library versions. A one-ULP operand disagreement survives
# at full absolute size in the intrinsic-value lower bound, which is a cancelling
# difference of those operands and is several times smaller than either of them,
# so tolerances must be derived from the operand scale rather than from the
# expected value. Budget: 2 ULP of library-to-library exp disagreement, 1 ULP for
# the two roundings of the multiply by spot/strike, and 1 ULP for the rounding of
# the combining subtraction; doubled for headroom.
BOUND_ULP_BUDGET = 8.0


def bound_tolerance(*operands: np.ndarray) -> np.ndarray:
    """Return the per-row float64 tolerance for a bound built from ``operands``.

    ``ulp(x) <= FLOAT64_EPS * |x|``, so ``BOUND_ULP_BUDGET`` ULP of the largest
    operand bounds the accumulated evaluation-order difference. At the scales
    used here this is ~2e-13, roughly twelve orders of magnitude below the error
    a materially wrong projection (undiscounted strike, wrong sign, missing
    floor at zero) would produce.
    """
    scale = np.max(np.abs(np.stack(operands)), axis=0)
    return BOUND_ULP_BUDGET * FLOAT64_EPS * scale


def assert_matches_bound(
    actual: np.ndarray,
    expected: np.ndarray,
    tolerance: np.ndarray,
    label: str,
) -> None:
    """Assert ``actual`` sits on ``expected`` within a per-row ULP budget.

    ``np.testing.assert_allclose`` cannot report an array-valued ``atol``, so the
    per-row comparison is spelled out here to keep the tolerance scale-aware.
    """
    difference = np.abs(actual - expected)
    assert np.all(difference <= tolerance), (
        f"{label} projection off its discounted bound:\n"
        f"  actual:    {actual!r}\n"
        f"  expected:  {expected!r}\n"
        f"  |diff|:    {difference!r}\n"
        f"  tolerance: {tolerance!r} ({BOUND_ULP_BUDGET} ULP of the operand scale)"
    )

DATASET_CONFIG = f"""
schema_version = "{SCHEMA_VERSION}"
generator_version = "{GENERATOR_VERSION}"

[dataset]
name = "differential-training-test"
oracle = "dp::black_scholes"
base_seed = 97531

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

TRAINING_CONFIG = """
schema_version = "european-neural-baseline-config/v1"
experiment_name = "differential-training-test"
representation = "forward_normalized_v1"
features = [
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

[objective]
name = "price_and_first_derivatives_v1"
price_weight = 1.0
gradient_weight = 1.0

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


def write_file(root: Path, name: str, text: str) -> Path:
    path = root / name
    path.write_text(text, encoding="utf-8")
    return path


def build_dataset(root: Path) -> Path:
    config = write_file(root, "dataset.toml", DATASET_CONFIG)
    output = root / "dataset"
    generate_dataset(load_dataset_config(config), output)
    return output


def constant_physical_model(
    value: float,
    *,
    constraint: str,
) -> PhysicalPriceModel:
    scaling = Scaling(
        feature_mean=np.zeros(7, dtype=np.float64),
        feature_scale=np.ones(7, dtype=np.float64),
        price_mean=0.0,
        price_scale=1.0,
    )
    network = PricingMlp(7, ())
    layer = network.layers[0]
    assert isinstance(layer, torch.nn.Linear)
    with torch.no_grad():
        layer.weight.zero_()
        layer.bias.fill_(value)
    return PhysicalPriceModel(
        network,
        scaling,
        output_constraint=constraint,
    ).eval()


def test_differential_config_changes_only_the_objective() -> None:
    price_only = load_training_config(
        PROJECT_ROOT / "configs/european_neural_forward_normalized_v1.toml"
    )
    differential = load_training_config(
        PROJECT_ROOT / "configs/european_neural_forward_differential_v1.toml"
    )

    assert differential.representation == price_only.representation
    assert differential.representation == FORWARD_NORMALIZED_REPRESENTATION
    assert differential.model == price_only.model
    assert differential.optimizer == price_only.optimizer
    assert differential.training == price_only.training
    assert differential.evaluation == price_only.evaluation
    assert price_only.objective.name == "price_only_v1"
    assert differential.objective.name == "price_and_first_derivatives_v1"
    assert differential.objective.price_weight == 1.0
    assert differential.objective.gradient_weight == 1.0


def test_differential_objective_requires_forward_representation(
    tmp_path: Path,
) -> None:
    raw = TRAINING_CONFIG.replace(
        'representation = "forward_normalized_v1"',
        'representation = "raw_physical_v1"',
    )
    path = write_file(tmp_path, "raw-differential.toml", raw)
    with pytest.raises(TrainingConfigError, match="requires representation"):
        load_training_config(path)


def test_derivative_label_transform_matches_chain_rule() -> None:
    features = np.array(
        [[1.0, 100.0, 95.0, 2.0, 0.03, 0.01, 0.2]],
        dtype=np.float64,
    )
    spot = features[0, 1]
    strike = features[0, 2]
    maturity = features[0, 3]
    rate = features[0, 4]
    dividend_yield = features[0, 5]
    volatility = features[0, 6]
    x = np.log(spot / strike) + (rate - dividend_yield) * maturity
    v = volatility * np.sqrt(maturity)
    normalized_price = 2.0 + 3.0 * x + 4.0 * v
    discounted_spot = spot * np.exp(-dividend_yield * maturity)
    prices = np.array([discounted_spot * normalized_price])
    deltas = np.array(
        [np.exp(-dividend_yield * maturity) * (normalized_price + 3.0)]
    )
    vegas = np.array([discounted_spot * 4.0 * np.sqrt(maturity)])
    scaling = Scaling(
        feature_mean=np.zeros(3),
        feature_scale=np.array([1.0, 2.0, 5.0]),
        price_mean=0.0,
        price_scale=10.0,
    )

    result = forward_normalized_derivative_targets(
        features,
        prices,
        deltas,
        vegas,
        scaling,
    )

    np.testing.assert_allclose(result, np.array([[0.6, 2.0]]), rtol=1e-14)


def test_european_projection_enforces_discounted_price_bounds() -> None:
    features = np.array(
        [
            [1.0, 120.0, 100.0, 1.0, 0.03, 0.01, 0.2],
            [-1.0, 80.0, 100.0, 1.0, 0.03, 0.01, 0.2],
        ],
        dtype=np.float64,
    )
    discounted_spot = features[:, 1] * np.exp(-features[:, 5] * features[:, 3])
    discounted_strike = features[:, 2] * np.exp(-features[:, 4] * features[:, 3])
    lower = np.array(
        [
            discounted_spot[0] - discounted_strike[0],
            discounted_strike[1] - discounted_spot[1],
        ]
    )
    upper = np.array([discounted_spot[0], discounted_strike[1]])

    low_raw = constant_physical_model(
        -100.0,
        constraint=NO_OUTPUT_CONSTRAINT,
    )
    low_bounded = constant_physical_model(
        -100.0,
        constraint=EUROPEAN_BOUNDS_CONSTRAINT,
    )
    high_raw = constant_physical_model(
        1_000.0,
        constraint=NO_OUTPUT_CONSTRAINT,
    )
    high_bounded = constant_physical_model(
        1_000.0,
        constraint=EUROPEAN_BOUNDS_CONSTRAINT,
    )
    tensor = torch.as_tensor(features, dtype=torch.float64)
    low_price = low_bounded(tensor).detach().numpy()
    high_price = high_bounded(tensor).detach().numpy()

    # Both bounds are compared against the same operand-scaled tolerance: the
    # upper bound is a single discounted operand, the lower bound a difference
    # of the two, and both inherit absolute error at the operand scale.
    tolerance = bound_tolerance(discounted_spot, discounted_strike)
    assert_matches_bound(low_price, lower, tolerance, "lower")
    assert_matches_bound(high_price, upper, tolerance, "upper")
    target_inside_bounds = (lower + upper) / 2.0
    assert np.all(
        np.abs(low_price - target_inside_bounds)
        <= np.abs(low_raw(tensor).detach().numpy() - target_inside_bounds)
    )
    assert np.all(
        np.abs(high_price - target_inside_bounds)
        <= np.abs(high_raw(tensor).detach().numpy() - target_inside_bounds)
    )


def test_active_bound_greeks_are_physical_bound_derivatives() -> None:
    features = np.array(
        [
            [1.0, 120.0, 100.0, 1.0, 0.03, 0.01, 0.2],
            [-1.0, 80.0, 100.0, 1.0, 0.03, 0.01, 0.2],
        ],
        dtype=np.float64,
    )
    predictions = predict_with_greeks(
        constant_physical_model(
            -100.0,
            constraint=EUROPEAN_BOUNDS_CONSTRAINT,
        ),
        features,
        batch_size=2,
    )
    discounted_spot = features[:, 1] * np.exp(-features[:, 5] * features[:, 3])
    discounted_strike = features[:, 2] * np.exp(-features[:, 4] * features[:, 3])
    expected = {
        "price": np.array(
            [
                discounted_spot[0] - discounted_strike[0],
                discounted_strike[1] - discounted_spot[1],
            ]
        ),
        "delta": np.array(
            [
                np.exp(-features[0, 5] * features[0, 3]),
                -np.exp(-features[1, 5] * features[1, 3]),
            ]
        ),
        "gamma": np.zeros(2),
        "vega": np.zeros(2),
        "theta": np.array(
            [
                features[0, 5] * discounted_spot[0]
                - features[0, 4] * discounted_strike[0],
                features[1, 4] * discounted_strike[1]
                - features[1, 5] * discounted_spot[1],
            ]
        ),
        "rho": np.array(
            [
                features[0, 3] * discounted_strike[0],
                -features[1, 3] * discounted_strike[1],
            ]
        ),
    }
    for name, reference in expected.items():
        np.testing.assert_allclose(
            predictions[name],
            reference,
            rtol=1.0e-13,
            atol=1.0e-13,
        )


def test_differential_artifact_is_deterministic_and_evaluable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = build_dataset(tmp_path)
    config = write_file(tmp_path, "training.toml", TRAINING_CONFIG)
    visited: list[str] = []
    original_hash = dataset_module.sha256_file

    def recording_hash(path: Path) -> str:
        visited.append(path.name)
        if path.name == "interpolation_test.parquet":
            raise AssertionError("differential training touched the locked test")
        return original_hash(path)

    monkeypatch.setattr(dataset_module, "sha256_file", recording_hash)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_payload = train_to_artifact(dataset, config, first)
    second_payload = train_to_artifact(dataset, config, second)

    assert (first / "weights.npz").read_bytes() == (
        second / "weights.npz"
    ).read_bytes()
    assert first_payload == second_payload
    assert visited == [
        "train.parquet",
        "validation.parquet",
        "train.parquet",
        "validation.parquet",
    ]
    assert first_payload["objective"]["greek_labels_used_for_training"] == [
        "delta",
        "vega",
    ]
    assert first_payload["objective"]["differential_coordinates"] == [
        "log_forward_moneyness",
        "total_volatility",
    ]
    assert first_payload["training"]["selection_metric"] == "validation_objective"
    assert np.isfinite(first_payload["training"]["best_validation_objective"])
    history = first_payload["training"]["history"]
    assert history
    assert all(
        np.isfinite(row["validation_normalized_first_derivative_mse"])
        for row in history
    )

    model, loaded = load_physical_model(first)
    assert model.representation == FORWARD_NORMALIZED_REPRESENTATION
    assert loaded == json.loads(
        (first / ARTIFACT_MANIFEST).read_text(encoding="utf-8")
    )
    report = evaluate_artifact(
        dataset,
        first,
        first / "validation-evaluation.json",
        partition="validation",
    )
    assert report["partition_role"] == "model_selection"
    assert np.isfinite(
        report["slices"]["overall"]["metrics"]["gamma"]["rmse"]
    )

    bounded_first = tmp_path / "bounded-first"
    bounded_second = tmp_path / "bounded-second"
    bounded_payload = derive_bounded_artifact(first, bounded_first)
    second_bounded_payload = derive_bounded_artifact(second, bounded_second)
    assert bounded_payload == second_bounded_payload
    assert (bounded_first / ARTIFACT_MANIFEST).read_bytes() == (
        bounded_second / ARTIFACT_MANIFEST
    ).read_bytes()
    assert (bounded_first / "weights.npz").read_bytes() == (
        first / "weights.npz"
    ).read_bytes()
    assert bounded_payload["weights"]["sha256"] == first_payload["weights"]["sha256"]
    assert bounded_payload["output_constraint"]["name"] == EUROPEAN_BOUNDS_CONSTRAINT
    assert bounded_payload["output_constraint"]["source_artifact"] == {
        "manifest_sha256": sha256_file(first / ARTIFACT_MANIFEST),
        "weights_sha256": first_payload["weights"]["sha256"],
    }

    bounded_model, loaded_bounded = load_physical_model(bounded_first)
    assert bounded_model.output_constraint == EUROPEAN_BOUNDS_CONSTRAINT
    assert loaded_bounded == bounded_payload
    bounded_report = evaluate_artifact(
        dataset,
        bounded_first,
        bounded_first / "validation-evaluation.json",
        partition="validation",
    )
    assert (
        bounded_report["artifact"]["output_constraint"]
        == EUROPEAN_BOUNDS_CONSTRAINT
    )
    constraint_diagnostics = bounded_report["output_constraint_diagnostics"]
    assert constraint_diagnostics["active_rows"] == (
        constraint_diagnostics["lower_bound_activations"]
        + constraint_diagnostics["upper_bound_activations"]
    )
    assert (
        constraint_diagnostics["unconstrained_price_no_arbitrage"]
        == report["learned_price_no_arbitrage"]
    )
    assert (
        constraint_diagnostics["unconstrained_price_metrics"]
        == report["slices"]["overall"]["metrics"]["price"]
    )
    assert bounded_report["learned_price_no_arbitrage"]["material_violations"] == 0
    unconstrained_metrics = report["slices"]["overall"]["metrics"]["price"]
    bounded_metrics = bounded_report["slices"]["overall"]["metrics"]["price"]
    assert bounded_metrics["mae"] <= unconstrained_metrics["mae"]
    assert bounded_metrics["rmse"] <= unconstrained_metrics["rmse"]

    # Reports are published exclusively: a second evaluation must refuse rather
    # than overwrite, leaving the first report byte-identical.
    existing_report = first / "validation-evaluation.json"
    before = existing_report.read_bytes()
    with pytest.raises(EvaluationError, match="already exists"):
        evaluate_artifact(dataset, first, existing_report, partition="validation")
    assert existing_report.read_bytes() == before

    # A failed publication leaves no partial report and no temporaries, and the
    # retry afterwards succeeds.
    retry_report = first / "retry-evaluation.json"

    def failing_fsync(descriptor: int) -> None:
        raise OSError("simulated disk failure")

    monkeypatch.setattr(os, "fsync", failing_fsync)
    with pytest.raises(EvaluationError, match="cannot write evaluation report"):
        evaluate_artifact(dataset, first, retry_report, partition="validation")
    assert not retry_report.exists()
    assert not [path for path in first.iterdir() if path.name.endswith(".tmp")]

    monkeypatch.undo()
    monkeypatch.setattr(dataset_module, "sha256_file", recording_hash)
    retried = evaluate_artifact(dataset, first, retry_report, partition="validation")
    assert json.loads(retry_report.read_text(encoding="utf-8")) == retried

    with pytest.raises(ConstraintError, match="already exists"):
        derive_bounded_artifact(first, bounded_first)
    with pytest.raises(ConstraintError, match="already has"):
        derive_bounded_artifact(bounded_first, tmp_path / "bounded-again")

    tampered = json.loads(
        (bounded_first / ARTIFACT_MANIFEST).read_text(encoding="utf-8")
    )
    tampered["output_constraint"]["source_artifact"]["weights_sha256"] = "0" * 64
    (bounded_first / ARTIFACT_MANIFEST).write_text(
        json.dumps(tampered, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ArtifactError, match="weights do not match its declared source"):
        load_physical_model(bounded_first)


def test_constraint_command_reports_contract_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = constrain_main(
        [
            "--artifact",
            str(tmp_path / "missing-artifact"),
            "--output",
            str(tmp_path / "bounded-artifact"),
        ]
    )
    assert status == 2
    assert "error:" in capsys.readouterr().err
