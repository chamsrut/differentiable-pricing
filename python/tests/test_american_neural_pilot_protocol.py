"""The checked-in Task 9G protocol is complete and internally pinned."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _module():
    path = PROJECT_ROOT / "scripts/check_american_neural_pilot_protocol.py"
    spec = importlib.util.spec_from_file_location("task9g_protocol", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_locked_task9g_protocol_is_complete() -> None:
    summary = _module().validate_protocol(
        PROJECT_ROOT / "configs/american_neural_pilot_protocol_v1.toml",
        PROJECT_ROOT,
    )
    assert summary == {
        "name": "task-9g-american-neural-pilot-v1",
        "tracked_inputs": 35,
        "pde_cases": 21,
        "scratch_seed": 2909056561,
        "transfer_seed": 2009073353,
    }


@pytest.mark.parametrize(
    ("filename", "old", "new"),
    [
        (
            "american_neural_pilot_training_v1.toml",
            'salt = "task-9g-train-subset-v1"',
            'salt = "mutable"',
        ),
        (
            "american_neural_pilot_acceptance_v1.toml",
            'quantile_method = "linear"',
            'quantile_method = "lower"',
        ),
        (
            "american_neural_pilot_latency_cases_v1.toml",
            'batch_sizes = [1, 8]',
            'batch_sizes = [1, 4]',
        ),
        (
            "american_neural_pilot_iv_cases_v1.toml",
            "volatility_bracket = [0.05, 0.8]",
            "volatility_bracket = [0.1, 0.8]",
        ),
    ],
)
def test_leaf_semantics_remain_locked_when_digest_could_be_refreshed(
    tmp_path: Path, filename: str, old: str, new: str
) -> None:
    module = _module()
    for source in (
        "american_neural_pilot_training_v1.toml",
        "american_neural_pilot_acceptance_v1.toml",
        "american_neural_pilot_latency_cases_v1.toml",
        "american_neural_pilot_iv_cases_v1.toml",
    ):
        shutil.copyfile(PROJECT_ROOT / "configs" / source, tmp_path / source)
    target = tmp_path / filename
    text = target.read_text(encoding="utf-8")
    assert old in text
    target.write_text(text.replace(old, new, 1), encoding="utf-8")
    protocol = {
        "paths": {
            "training_config": "american_neural_pilot_training_v1.toml",
            "acceptance_config": "american_neural_pilot_acceptance_v1.toml",
            "latency_config": "american_neural_pilot_latency_cases_v1.toml",
            "iv_config": "american_neural_pilot_iv_cases_v1.toml",
        }
    }
    with pytest.raises(module.ProtocolError):
        module._validate_leaf_configs(protocol, tmp_path)
