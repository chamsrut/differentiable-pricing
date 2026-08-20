"""Strict American artifact round-trip and rejection tests."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from differentiable_pricing.ml.american import AmericanPriceModel
from differentiable_pricing.ml.american_artifact import (
    AmericanArtifactError,
    load_american_artifact,
    save_american_artifact,
)
from differentiable_pricing.ml.model import PricingMlp, Scaling


def _model() -> AmericanPriceModel:
    torch.manual_seed(17)
    return AmericanPriceModel(
        PricingMlp(5, (64, 64, 64)),
        Scaling(np.zeros(5), np.ones(5), 0.2, 0.3),
    ).eval()


def _metadata(arm: str = "scratch") -> dict:
    return {
        "experiment_name": "fixture",
        "arm": arm,
        "dataset": {
            "manifest_sha256": "0" * 64,
            "train_sha256": "1" * 64,
            "validation_sha256": "2" * 64,
            "selected_train_rows": 32768,
        },
        "protocol": {
            "sha256": "3" * 64,
            "schema_version": "american-neural-pilot-protocol/1",
        },
        "code": {"tracked_inputs": [{"path": "fixture.py", "sha256": "4" * 64}]},
        "training": {
            "seed": 3,
            "best_epoch": 1,
            "best_validation_mse": 0.1,
            "history": [
                {
                    "epoch": 1,
                    "validation_standardized_target_mse": 0.1,
                    "learning_rate": 0.001,
                }
            ],
        },
        "source_lineage": (
            None
            if arm == "scratch"
            else {
                "manifest_sha256": "5" * 64,
                "weights_sha256": "6" * 64,
                "lift": "task-9g-exact-european-to-american-v1",
            }
        ),
    }


def test_american_artifact_round_trip_is_deterministic(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    save_american_artifact(first, _model(), **_metadata())
    save_american_artifact(second, _model(), **_metadata())
    assert (first / "weights.npz").read_bytes() == (second / "weights.npz").read_bytes()
    loaded, payload = load_american_artifact(first)
    features = torch.tensor([[1.0, 100.0, 95.0, 1.0, 0.03, 0.01, 0.2]], dtype=torch.float64)
    with torch.no_grad():
        assert torch.equal(loaded(features), _model()(features))
    assert payload["output_constraint"] == {"name": "none_v1", "projection": None}


def test_american_artifact_rejects_unknown_manifest_key(tmp_path) -> None:
    directory = tmp_path / "artifact"
    save_american_artifact(directory, _model(), **_metadata())
    path = directory / "artifact.json"
    payload = json.loads(path.read_text())
    payload["unknown"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(AmericanArtifactError, match="unknown"):
        load_american_artifact(directory)


def test_transfer_artifact_requires_source_lineage(tmp_path) -> None:
    directory = tmp_path / "artifact"
    metadata = _metadata("transfer")
    metadata["source_lineage"] = None
    save_american_artifact(directory, _model(), **metadata)
    with pytest.raises(AmericanArtifactError, match="source_lineage"):
        load_american_artifact(directory)


@pytest.mark.parametrize(
    ("block", "mutation", "message"),
    [
        ("weights", lambda value: value.update({"unknown": True}), "unknown"),
        ("dataset", lambda value: value.pop("train_sha256"), "missing"),
        ("protocol", lambda value: value.update({"sha256": "bad"}), "SHA-256"),
        ("code", lambda value: value.update({"unknown": True}), "unknown"),
        ("training", lambda value: value.pop("history"), "missing"),
    ],
)
def test_american_artifact_rejects_nested_identity_drift(
    tmp_path, block, mutation, message
) -> None:
    directory = tmp_path / "artifact"
    save_american_artifact(directory, _model(), **_metadata())
    path = directory / "artifact.json"
    payload = json.loads(path.read_text())
    mutation(payload[block])
    path.write_text(json.dumps(payload))
    with pytest.raises(AmericanArtifactError, match=message):
        load_american_artifact(directory)


def test_transfer_artifact_rejects_wrong_lift_identity(tmp_path) -> None:
    directory = tmp_path / "artifact"
    save_american_artifact(directory, _model(), **_metadata("transfer"))
    path = directory / "artifact.json"
    payload = json.loads(path.read_text())
    payload["source_lineage"]["lift"] = "invented"
    path.write_text(json.dumps(payload))
    with pytest.raises(AmericanArtifactError, match="lift identity"):
        load_american_artifact(directory)
