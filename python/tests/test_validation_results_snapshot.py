"""The versioned result snapshot must carry complete, consistent provenance.

The snapshot is the only record of the controlled study that is checked into
the repository: the datasets and artifacts it describes are deliberately
gitignored. Its digests are therefore the sole link between a reported number
and the artifact that produced it, and the validator has to reject an
incomplete or self-inconsistent link rather than plot it.

The plot script is stdlib-only, so these run in the lightweight CI job.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "plot_european_validation_results.py"
RESULTS = PROJECT_ROOT / "docs" / "results" / "european_validation_results_v1.json"
ACCEPTANCE = PROJECT_ROOT / "configs" / "european_neural_acceptance_v1.toml"

DIGEST_A = "a" * 64
DIGEST_B = "b" * 64


def load_plot_module() -> Any:
    spec = importlib.util.spec_from_file_location("dp_plot_results", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plot = load_plot_module()


@pytest.fixture
def results() -> dict[str, Any]:
    return json.loads(RESULTS.read_text(encoding="utf-8"))


@pytest.fixture
def gates() -> dict[str, Any]:
    with ACCEPTANCE.open("rb") as stream:
        return tomllib.load(stream)


def test_checked_in_snapshot_validates(
    results: dict[str, Any],
    gates: dict[str, Any],
) -> None:
    experiments = plot._validate(results, gates)
    assert len(experiments) == len(plot.EXPECTED_EXPERIMENT_IDS)


def test_every_experiment_carries_complete_provenance(
    results: dict[str, Any],
) -> None:
    """Guards against a row silently reverting to a weights-digest-only record."""
    for experiment in results["experiments"]:
        assert plot.SHA256_PATTERN.fullmatch(experiment["weights_sha256"])
        assert plot.SHA256_PATTERN.fullmatch(experiment["artifact_manifest_sha256"])
        assert experiment["artifact"].startswith("artifacts/")
    assert plot.SHA256_PATTERN.fullmatch(results["dataset_manifest_sha256"])


@pytest.mark.parametrize("field", ["weights_sha256", "artifact_manifest_sha256"])
@pytest.mark.parametrize("index", range(6))
def test_missing_experiment_digest_is_rejected(
    results: dict[str, Any],
    gates: dict[str, Any],
    field: str,
    index: int,
) -> None:
    broken = copy.deepcopy(results)
    del broken["experiments"][index][field]
    with pytest.raises(plot.PlotError, match=f"missing {field}"):
        plot._validate(broken, gates)


@pytest.mark.parametrize("field", ["weights_sha256", "artifact_manifest_sha256"])
@pytest.mark.parametrize("bad", ["", "not-a-digest", "ABC" * 21 + "D", DIGEST_A[:63]])
def test_malformed_experiment_digest_is_rejected(
    results: dict[str, Any],
    gates: dict[str, Any],
    field: str,
    bad: str,
) -> None:
    broken = copy.deepcopy(results)
    broken["experiments"][0][field] = bad
    with pytest.raises(plot.PlotError, match=f"invalid {field}"):
        plot._validate(broken, gates)


def test_missing_dataset_manifest_digest_is_rejected(
    results: dict[str, Any],
    gates: dict[str, Any],
) -> None:
    broken = copy.deepcopy(results)
    del broken["dataset_manifest_sha256"]
    with pytest.raises(plot.PlotError, match="dataset_manifest_sha256"):
        plot._validate(broken, gates)


def test_malformed_dataset_manifest_digest_is_rejected(
    results: dict[str, Any],
    gates: dict[str, Any],
) -> None:
    broken = copy.deepcopy(results)
    broken["dataset_manifest_sha256"] = "0x" + "0" * 62
    with pytest.raises(plot.PlotError, match="dataset_manifest_sha256"):
        plot._validate(broken, gates)


def test_missing_artifact_directory_is_rejected(
    results: dict[str, Any],
    gates: dict[str, Any],
) -> None:
    broken = copy.deepcopy(results)
    del broken["experiments"][2]["artifact"]
    with pytest.raises(plot.PlotError, match="must name its artifact directory"):
        plot._validate(broken, gates)


def test_artifact_outside_the_artifacts_directory_is_rejected(
    results: dict[str, Any],
    gates: dict[str, Any],
) -> None:
    broken = copy.deepcopy(results)
    broken["experiments"][2]["artifact"] = "/etc/passwd"
    with pytest.raises(plot.PlotError, match="must name its artifact directory"):
        plot._validate(broken, gates)


def test_duplicate_artifact_directory_is_rejected(
    results: dict[str, Any],
    gates: dict[str, Any],
) -> None:
    broken = copy.deepcopy(results)
    broken["experiments"][1]["artifact"] = broken["experiments"][0]["artifact"]
    with pytest.raises(plot.PlotError, match="distinct artifact directory"):
        plot._validate(broken, gates)


def test_duplicate_artifact_manifest_digest_is_rejected(
    results: dict[str, Any],
    gates: dict[str, Any],
) -> None:
    """Two rows sharing a manifest digest are the same artifact reported twice."""
    broken = copy.deepcopy(results)
    broken["experiments"][1]["artifact_manifest_sha256"] = broken["experiments"][0][
        "artifact_manifest_sha256"
    ]
    with pytest.raises(plot.PlotError, match="distinct artifact manifest digest"):
        plot._validate(broken, gates)


def test_bounded_row_must_reuse_the_differential_weights(
    results: dict[str, Any],
    gates: dict[str, Any],
) -> None:
    broken = copy.deepcopy(results)
    broken["experiments"][-1]["weights_sha256"] = DIGEST_A
    broken["experiments"][-1]["source_weights_sha256"] = DIGEST_A
    with pytest.raises(plot.PlotError, match="source digest does not match"):
        plot._validate(broken, gates)


def test_bounded_row_must_preserve_its_source_weights(
    results: dict[str, Any],
    gates: dict[str, Any],
) -> None:
    broken = copy.deepcopy(results)
    broken["experiments"][-1]["weights_sha256"] = DIGEST_B
    with pytest.raises(plot.PlotError, match="did not preserve its source weights"):
        plot._validate(broken, gates)


def test_bounded_row_source_manifest_must_match_the_differential(
    results: dict[str, Any],
    gates: dict[str, Any],
) -> None:
    broken = copy.deepcopy(results)
    broken["experiments"][-1]["source_manifest_sha256"] = DIGEST_A
    with pytest.raises(plot.PlotError, match="source manifest does not match"):
        plot._validate(broken, gates)


def test_bounded_row_must_have_its_own_manifest(
    results: dict[str, Any],
    gates: dict[str, Any],
) -> None:
    """Identical weights, but the constraint block changes the manifest."""
    broken = copy.deepcopy(results)
    shared = broken["experiments"][-2]["artifact_manifest_sha256"]
    broken["experiments"][-1]["artifact_manifest_sha256"] = shared
    # Duplicate-manifest detection fires first, which is itself the right
    # rejection; drop to a single check by making the directories differ only.
    with pytest.raises(plot.PlotError, match="distinct artifact manifest digest"):
        plot._validate(broken, gates)


def test_snapshot_digests_match_the_recorded_bounded_lineage(
    results: dict[str, Any],
) -> None:
    """The bounded row's declared lineage is internally consistent.

    This is traceability metadata: it links the bounded row to the differential
    row's manifest. It does not authenticate the source artifact, which is not
    in the repository.
    """
    differential, bounded = results["experiments"][-2], results["experiments"][-1]
    assert bounded["source_weights_sha256"] == differential["weights_sha256"]
    assert bounded["weights_sha256"] == differential["weights_sha256"]
    assert bounded["source_manifest_sha256"] == differential["artifact_manifest_sha256"]
    assert bounded["artifact_manifest_sha256"] != differential["artifact_manifest_sha256"]
