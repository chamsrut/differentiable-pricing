"""The frozen replication result must remain complete, scoped, and reproducible."""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "plot_european_replication_results.py"
RESULTS = PROJECT_ROOT / "docs" / "results" / "european_replication_results_v1.json"
FIGURES = PROJECT_ROOT / "docs" / "figures"


def load_plot_module() -> Any:
    spec = importlib.util.spec_from_file_location("dp_plot_replication", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plot = load_plot_module()


@pytest.fixture
def results() -> dict[str, Any]:
    return json.loads(RESULTS.read_text(encoding="utf-8"))


def test_checked_in_replication_snapshot_validates(
    results: dict[str, Any],
) -> None:
    assert plot.validate_snapshot(results) is results


def test_snapshot_records_the_single_terminal_attempt(
    results: dict[str, Any],
) -> None:
    assert results["execution"]["status"] == "replication_passed"
    assert results["execution"]["final_evaluation_attempts"] == 1
    assert results["interpretation"]["locked_final_evaluation_consumed"] is True
    assert results["acceptance"]["validation_passed"] is True
    assert results["acceptance"]["final_passed"] is True


def test_protocol_precedes_the_clean_runner_commit(
    results: dict[str, Any],
) -> None:
    repository = results["execution"]["repository"]
    assert repository["clean_worktree"] is True
    assert repository["branch"] == "main"
    assert repository["source_commit"] == repository["remote_main"]
    assert repository["protocol_commit"] != repository["source_commit"]


def test_bounded_artifact_reuses_exact_weight_bytes(
    results: dict[str, Any],
) -> None:
    unconstrained = results["model"]["unconstrained_artifact"]
    bounded = results["model"]["bounded_artifact"]
    assert bounded["weights_sha256"] == unconstrained["weights_sha256"]
    assert bounded["source_manifest_sha256"] == unconstrained["manifest_sha256"]
    assert bounded["manifest_sha256"] != unconstrained["manifest_sha256"]


def test_final_partition_role_and_report_are_locked(
    results: dict[str, Any],
) -> None:
    final = results["results"]["locked_final"]
    assert final["partition"] == "interpolation_test"
    assert final["partition_role"] == "locked_final_evaluation"
    assert final["rows"] == 25_000
    assert plot.SHA256_PATTERN.fullmatch(final["report_sha256"])


def test_all_frozen_gates_pass_and_gamma_is_the_tightest(
    results: dict[str, Any],
) -> None:
    checks = results["acceptance"]["checks"]
    positive_limits = [check for check in checks if check["limit"] > 0]
    for check in positive_limits:
        assert check["validation_value"] <= check["limit"]
        assert check["final_value"] <= check["limit"]
    zero_gate = checks[-1]
    assert zero_gate == {
        "final_value": 0,
        "limit": 0,
        "name": "no_arbitrage.material_violations",
        "validation_value": 0,
    }
    tightest = max(positive_limits, key=lambda check: check["final_value"] / check["limit"])
    assert tightest["name"] == "overall.gamma_rmse"
    assert tightest["final_value"] / tightest["limit"] == pytest.approx(
        0.968269608200436,
    )


@pytest.mark.parametrize("partition", ["validation", "locked_final"])
def test_projection_intervention_is_disclosed(
    results: dict[str, Any],
    partition: str,
) -> None:
    result = results["results"][partition]
    projection = result["projection"]
    assert projection["active_rows"] > 0
    assert projection["active_rows"] == projection["lower_bound_activations"]
    assert projection["upper_bound_activations"] == 0
    assert 0 < projection["unconstrained_material_violations"] <= projection["active_rows"]
    assert result["no_arbitrage"]["material_violations"] == 0


def test_evidence_archive_is_traceability_not_a_tracked_attestation(
    results: dict[str, Any],
) -> None:
    archive = results["evidence"]["archive"]
    assert archive["tracked"] is False
    assert plot.SHA256_PATTERN.fullmatch(archive["sha256"])
    assert "not a cryptographic attestation" in results["evidence"]["note"]


def test_snapshot_preserves_scoped_non_claims(
    results: dict[str, Any],
) -> None:
    limitations = " ".join(results["interpretation"]["limitations"])
    assert "Synthetic Black-Scholes" in limitations
    assert "96.8%" in limitations
    assert "1586 of 25000" in limitations
    assert "latency" in limitations
    assert "attestation" in limitations


def test_checked_in_figures_are_current() -> None:
    for name, expected in plot.render_figures(RESULTS).items():
        assert (FIGURES / name).read_text(encoding="utf-8") == expected


def test_wrong_protocol_digest_is_rejected(
    results: dict[str, Any],
) -> None:
    broken = copy.deepcopy(results)
    broken["protocol"]["sha256"] = "0" * 64
    with pytest.raises(plot.PlotError, match="does not match the checked-in file"):
        plot.validate_snapshot(broken)


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("dataset", "config_sha256"),
        ("model", "training_config_sha256"),
        ("acceptance", "config_sha256"),
    ],
)
def test_wrong_pinned_config_digest_is_rejected(
    results: dict[str, Any],
    section: str,
    field: str,
) -> None:
    broken = copy.deepcopy(results)
    broken[section][field] = "0" * 64
    with pytest.raises(plot.PlotError, match="does not match the checked-in file"):
        plot.validate_snapshot(broken)


def test_second_final_attempt_is_rejected(
    results: dict[str, Any],
) -> None:
    broken = copy.deepcopy(results)
    broken["execution"]["final_evaluation_attempts"] = 2
    with pytest.raises(plot.PlotError, match="exactly one final evaluation attempt"):
        plot.validate_snapshot(broken)


def test_dirty_worktree_claim_is_rejected(
    results: dict[str, Any],
) -> None:
    broken = copy.deepcopy(results)
    broken["execution"]["repository"]["clean_worktree"] = False
    with pytest.raises(plot.PlotError, match="clean main worktree"):
        plot.validate_snapshot(broken)


def test_final_partition_cannot_be_relabelled(
    results: dict[str, Any],
) -> None:
    broken = copy.deepcopy(results)
    broken["results"]["locked_final"]["partition_role"] = "model_selection"
    with pytest.raises(plot.PlotError, match="locked_final_evaluation"):
        plot.validate_snapshot(broken)


def test_metric_cannot_diverge_from_its_gate_record(
    results: dict[str, Any],
) -> None:
    broken = copy.deepcopy(results)
    broken["results"]["locked_final"]["metrics"]["gamma_rmse"] = 0.001
    with pytest.raises(plot.PlotError, match="disagrees with final metrics"):
        plot.validate_snapshot(broken)


def test_frozen_gate_cannot_be_relaxed(
    results: dict[str, Any],
) -> None:
    broken = copy.deepcopy(results)
    broken["acceptance"]["checks"][3]["limit"] = 0.003
    with pytest.raises(plot.PlotError, match="changed its frozen limit"):
        plot.validate_snapshot(broken)


def test_bounded_weights_cannot_diverge(
    results: dict[str, Any],
) -> None:
    broken = copy.deepcopy(results)
    broken["model"]["bounded_artifact"]["weights_sha256"] = "0" * 64
    with pytest.raises(plot.PlotError, match="did not preserve"):
        plot.validate_snapshot(broken)


def test_bounded_source_manifest_cannot_diverge(
    results: dict[str, Any],
) -> None:
    broken = copy.deepcopy(results)
    broken["model"]["bounded_artifact"]["source_manifest_sha256"] = "0" * 64
    with pytest.raises(plot.PlotError, match="source manifest"):
        plot.validate_snapshot(broken)


@pytest.mark.parametrize(
    ("artifact", "replacement"),
    [
        ("unconstrained_artifact", "artifacts/wrong-unconstrained"),
        ("bounded_artifact", "artifacts/wrong-bounded"),
    ],
)
def test_artifact_paths_must_match_the_protocol(
    results: dict[str, Any],
    artifact: str,
    replacement: str,
) -> None:
    broken = copy.deepcopy(results)
    broken["model"][artifact]["path"] = replacement
    with pytest.raises(plot.PlotError, match="artifact path does not match"):
        plot.validate_snapshot(broken)


def test_dataset_findings_cannot_be_hidden(
    results: dict[str, Any],
) -> None:
    broken = copy.deepcopy(results)
    broken["dataset"]["diagnostics"]["findings"] = 1
    with pytest.raises(plot.PlotError, match="zero findings"):
        plot.validate_snapshot(broken)


def test_projection_intervention_cannot_be_omitted(
    results: dict[str, Any],
) -> None:
    broken = copy.deepcopy(results)
    projection = broken["results"]["locked_final"]["projection"]
    projection["active_rows"] = 0
    projection["lower_bound_activations"] = 0
    projection["unconstrained_material_violations"] = 0
    with pytest.raises(plot.PlotError, match="non-zero projection intervention"):
        plot.validate_snapshot(broken)


def test_material_violation_cannot_pass_the_zero_gate(
    results: dict[str, Any],
) -> None:
    broken = copy.deepcopy(results)
    broken["results"]["locked_final"]["no_arbitrage"]["material_violations"] = 1
    broken["results"]["locked_final"]["no_arbitrage"]["total_violations"] = 1
    broken["acceptance"]["checks"][-1]["final_value"] = 1
    with pytest.raises(plot.PlotError, match="locked final failed"):
        plot.validate_snapshot(broken)


DELETE = object()


def _mutate(payload: Any, path: str, value: Any) -> None:
    """Apply one mutation addressed by a dotted path; integer parts index lists."""
    node = payload
    parts = path.split(".")
    for part in parts[:-1]:
        node = node[int(part)] if part.isdigit() else node[part]
    key: Any = int(parts[-1]) if parts[-1].isdigit() else parts[-1]
    if value is DELETE:
        del node[key]
    else:
        node[key] = value


# One mutation per validator branch, with the substring the branch must report.
# Removing any guarded branch from validate_snapshot makes its row pass, which
# fails the corresponding test.
MUTATIONS: list[tuple[str, Any, str]] = [
    # Repository-relative path rejection: traversal and absolute paths.
    ("protocol.path", "../../etc/passwd", "repository-relative path"),
    ("protocol.path", "/etc/passwd", "repository-relative path"),
    ("dataset.config", "/tmp/dataset.toml", "repository-relative path"),
    ("acceptance.config", "configs/../../escape.toml", "repository-relative path"),
    # The runner must have executed the recorded remote main.
    ("execution.repository.remote_main", "1" * 40, "must equal the recorded remote main"),
    # Malformed SHA-256 digests: uppercase, short, non-hex, wrong type.
    ("dataset.manifest_sha256", "A" * 64, "lowercase SHA-256 digest"),
    ("evidence.archive.sha256", "abc123", "lowercase SHA-256 digest"),
    ("results.locked_final.report_sha256", "z" * 64, "lowercase SHA-256 digest"),
    ("evidence.execution_ledger.sha256", 0, "non-empty string"),
    # Malformed Git commits: short, non-hex, uppercase.
    ("execution.repository.source_commit", "abc123", "full lowercase Git commit"),
    ("execution.repository.source_commit", "g" * 40, "full lowercase Git commit"),
    ("execution.repository.protocol_commit", "A" * 40, "full lowercase Git commit"),
    # Metric domain: NaN, infinity, negative, boolean, wrong type.
    ("results.locked_final.metrics.theta_rmse", float("nan"), "finite and non-negative"),
    ("results.locked_final.metrics.price_rmse", float("inf"), "finite and non-negative"),
    ("results.locked_final.metrics.theta_rmse", -1e-9, "finite and non-negative"),
    ("results.validation.metrics.rho_rmse", True, "must be numeric"),
    ("results.validation.metrics.rho_rmse", "0.001", "must be numeric"),
    # Metric key set: extra and missing keys.
    ("results.validation.metrics.bogus_rmse", 0.0, "must contain exactly"),
    ("results.locked_final.metrics.rho_rmse", DELETE, "must contain exactly"),
    # No-arbitrage consistency: material <= total <= rows.
    ("results.locked_final.no_arbitrage.material_violations", 5, "counts are inconsistent"),
    ("results.validation.no_arbitrage.total_violations", 999_999, "counts are inconsistent"),
    # Acceptance check names and ordering.
    ("acceptance.checks.0.name", "overall.bogus_rmse", "order or names"),
    ("acceptance.checks.0.name", "overall.delta_rmse", "order or names"),
    # Gate limits: only the material-violation gate may be — and must be — zero.
    ("acceptance.checks.3.limit", 0.0, "strictly positive limit"),
    ("acceptance.checks.10.limit", 1, "frozen zero limit"),
    # Evidence identity and tracking status.
    ("evidence.archive.file", "european-replication-evidence-v2.tar.gz", "unexpected identity"),
    ("evidence.archive.tracked", True, "must not be claimed as tracked"),
    ("evidence.execution_ledger.path", "runs/other/execution.json", "unexpected identity"),
    # Scoped non-claims: at least five non-empty strings.
    ("interpretation.limitations", ["a", "b", "c", "d"], "scoped non-claims"),
    ("interpretation.limitations.0", "", "scoped non-claims"),
    ("interpretation.limitations.1", 5, "scoped non-claims"),
    # Dataset splits: exact membership, digest format, positive byte counts.
    ("dataset.splits.extra", {"bytes": 1, "sha256": "0" * 64}, "train, validation, and"),
    ("dataset.splits.validation", DELETE, "train, validation, and"),
    ("dataset.splits.train.sha256", "not-a-digest", "lowercase SHA-256 digest"),
    ("dataset.splits.train.bytes", 0, "must be an integer >= 1"),
    ("dataset.splits.train.bytes", "12", "must be an integer >= 1"),
]


@pytest.mark.parametrize(("path", "value", "message"), MUTATIONS)
def test_snapshot_mutation_is_rejected(
    results: dict[str, Any],
    path: str,
    value: Any,
    message: str,
) -> None:
    broken = copy.deepcopy(results)
    _mutate(broken, path, value)
    with pytest.raises(plot.PlotError, match=re.escape(message)):
        plot.validate_snapshot(broken)


def test_zero_limit_gate_cannot_reach_the_renderer(
    results: dict[str, Any],
) -> None:
    """A zero RMSE limit must raise PlotError, never an uncaught ZeroDivisionError."""
    broken = copy.deepcopy(results)
    broken["acceptance"]["checks"][3]["limit"] = 0.0
    with pytest.raises(plot.PlotError, match="strictly positive limit"):
        plot._gate_margins(broken)


@pytest.mark.parametrize(
    ("peak", "expected"),
    [(0, 300), (300, 600), (1_615, 1_800), (1_800, 2_100), (2_400, 2_700), (25_000, 27_000)],
)
def test_projection_axis_ceiling_is_derived_with_headroom(peak: int, expected: int) -> None:
    # 335.0 is the projection chart's plot height (570 - 115 top - 120 bottom).
    assert plot._projection_axis_maximum(peak, 335.0) == expected


def test_projection_chart_rescales_above_the_former_fixed_ceiling(
    results: dict[str, Any],
) -> None:
    broken = copy.deepcopy(results)
    for partition, active in (("validation", 2_400), ("locked_final", 2_100)):
        projection = broken["results"][partition]["projection"]
        projection["active_rows"] = active
        projection["lower_bound_activations"] = active
        projection["unconstrained_material_violations"] = active - 10
    svg = plot._projection_intervention(broken)

    assert ">2,700<" in svg
    bars = re.findall(
        r'<rect x="[\d.]+" y="([\d.]+)" width="72\.0" height="([\d.]+)"',
        svg,
    )
    assert len(bars) == 6
    top, chart_height = 115.0, 335.0
    for y_text, height_text in bars:
        y, height = float(y_text), float(height_text)
        assert y + height <= top + chart_height + 1e-9
        assert y - plot.PROJECTION_LABEL_RESERVE >= top - 1e-9


def test_rho_theta_stays_scoped_to_the_unconstrained_branch(
    results: dict[str, Any],
) -> None:
    evidence = results["interpretation"]["greek_evidence"]
    rho_and_theta = evidence["rho_and_theta"]
    assert "unconstrained branch" in rho_and_theta
    assert "European-bounds projection is active" in rho_and_theta
    assert "not unique at the projection kinks" in rho_and_theta
    assert "structural part of the model" in rho_and_theta
    # The gamma mechanism stays an explanation, not a measured ablation result.
    assert "plausibly" in evidence["gamma"]
    assert "unablated" in evidence["gamma"]
