"""Task 9H matched latency diagnostic: contract reuse, provenance and guards.

Synthetic fixtures throughout. **No dataset partition is opened, no model is
trained, and no timing is measured here**: the benchmark is a manual,
human-invoked run whose numbers depend on a quiet machine, and a test that ran it
would be measuring the test runner. What the tests pin is everything around the
measurement — that the contract is Task 9G's own file and implementation, that
the only declared deviation is the restricted depth ladder, that the benchmarked
checkpoint must come from a recorded attempt under its immutable configuration,
that artifact loading stays outside the timed region, and that the result is
labelled ungated with no validation exposure.
"""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from differentiable_pricing.ml.american_dev.attempts import (
    ATTEMPT_LOG_PATH,
    AttemptError,
    FinalPartitionAccessError,
    load_toml,
    sha256_file,
)
from differentiable_pricing.ml.american_dev.latency import (
    ATTEMPT_REPORT_NAME,
    BENCHMARK_TARGETS,
    BENCHMARKED_ATTEMPT,
    CHECKPOINT_NAME,
    DEFAULT_ATTEMPT_DIRECTORY,
    DEFAULT_OUTPUT,
    DIAGNOSTIC_BENCHMARK_ATTEMPT,
    LATENCY_CONFIG_PATH,
    LATENCY_QUANTILES,
    LATENCY_SCHEMA,
    MATCHED_CRR_DEPTH,
    UNGATED_LABEL,
    LatencyDiagnosticError,
    _guarded_output,
    acceptance_latency_reference,
    assert_benchmark_target,
    benchmark_matched_latency,
    compact_latency,
    load_benchmarked_model,
    matched_specification,
    measurement_statistics,
    recorded_scaling,
    resolve_benchmark_paths,
)
from differentiable_pricing.ml.american_dev.workbench import build_model
from differentiable_pricing.ml.model import Scaling

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LATENCY_MODULE = (
    PROJECT_ROOT / "python/src/differentiable_pricing/ml/american_dev/latency.py"
)
LATENCY_SCRIPT = PROJECT_ROOT / "scripts/benchmark_american_dev_latency.py"
PROTOCOL = PROJECT_ROOT / "configs/american_neural_pilot_protocol_v1.toml"
ACCEPTANCE = PROJECT_ROOT / "configs/american_neural_pilot_acceptance_v1.toml"
ATTEMPT_CONFIG = (
    "configs/american_dev_attempt_scratch_residual_smooth_floor_raw_loss_v1.toml"
)


# ---------------------------------------------------------------------------
# The Task 9G contract, reused rather than restated
# ---------------------------------------------------------------------------


def test_the_contract_is_the_file_the_locked_protocol_pins() -> None:
    from differentiable_pricing.ml.american_dev.domain import locked_tracked_input_digest

    pinned = locked_tracked_input_digest(load_toml(PROTOCOL), LATENCY_CONFIG_PATH)
    assert sha256_file(PROJECT_ROOT / LATENCY_CONFIG_PATH) == pinned


def test_the_matched_depth_is_the_one_the_acceptance_file_names() -> None:
    acceptance = load_toml(ACCEPTANCE)
    assert acceptance["latency"]["reference_depth_for_interpretation"] == MATCHED_CRR_DEPTH
    assert MATCHED_CRR_DEPTH in load_toml(PROJECT_ROOT / LATENCY_CONFIG_PATH)["crr_depths"]


def test_only_the_depth_ladder_is_restricted() -> None:
    """Every other semantic of the contract survives the restriction untouched."""
    contract = load_toml(PROJECT_ROOT / LATENCY_CONFIG_PATH)
    specification = matched_specification(contract)
    assert specification["crr_depths"] == [MATCHED_CRR_DEPTH]
    for key, value in contract.items():
        if key == "crr_depths":
            continue
        assert specification[key] == value
    assert specification["warmups"] == contract["warmups"]
    assert specification["repetitions"] == contract["repetitions"]
    assert specification["request_shapes"] == ["single", "batch8"]
    assert specification["batch_sizes"] == [1, 8]


def test_restricting_the_ladder_does_not_mutate_the_loaded_contract() -> None:
    contract = load_toml(PROJECT_ROOT / LATENCY_CONFIG_PATH)
    before = list(contract["crr_depths"])
    matched_specification(contract)
    assert contract["crr_depths"] == before


@pytest.mark.parametrize(
    "contract",
    [
        {"crr_depths": [256, 512]},
        {"crr_depths": 1024},
        {},
    ],
)
def test_a_contract_without_the_matched_depth_fails_closed(contract: dict[str, Any]) -> None:
    with pytest.raises(LatencyDiagnosticError):
        matched_specification(contract)


def test_a_contract_missing_a_required_section_fails_closed() -> None:
    with pytest.raises(LatencyDiagnosticError):
        matched_specification({"crr_depths": [MATCHED_CRR_DEPTH]})


# ---------------------------------------------------------------------------
# The Task 9G bar is context, never a gate
# ---------------------------------------------------------------------------


def test_the_acceptance_reference_is_read_through_a_two_key_whitelist() -> None:
    reference = acceptance_latency_reference(load_toml(ACCEPTANCE))
    assert reference["applied"] is False
    assert reference["reference_depth"] == MATCHED_CRR_DEPTH
    assert reference["task_9g_minimum_median_end_to_end_speedup"] == 10.0
    assert "final_partition" not in json.dumps(reference)


@pytest.mark.parametrize(
    "acceptance",
    [
        {},
        {"latency": {"reference_depth_for_interpretation": 2048}},
        {"latency": {"reference_depth_for_interpretation": MATCHED_CRR_DEPTH}},
        {
            "latency": {
                "reference_depth_for_interpretation": MATCHED_CRR_DEPTH,
                "minimum_median_end_to_end_speedup": 0.0,
            }
        },
    ],
)
def test_a_missing_or_mismatched_latency_reference_fails_closed(
    acceptance: dict[str, Any],
) -> None:
    with pytest.raises(LatencyDiagnosticError):
        acceptance_latency_reference(acceptance)


# ---------------------------------------------------------------------------
# The benchmarked model
# ---------------------------------------------------------------------------


def _scaling() -> Scaling:
    return Scaling(
        feature_mean=np.zeros(5, dtype=np.float64),
        feature_scale=np.ones(5, dtype=np.float64),
        price_mean=0.2,
        price_scale=0.15,
    )


def _synthetic_attempt_tree(root: Path, *, attempt_id: str = BENCHMARKED_ATTEMPT) -> Path:
    """A minimal tree holding one recorded attempt's configuration and checkpoint."""
    (root / "configs").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PROJECT_ROOT / ATTEMPT_CONFIG, root / ATTEMPT_CONFIG)
    digest = sha256_file(root / ATTEMPT_CONFIG)

    directory = root / DEFAULT_ATTEMPT_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    config = load_toml(root / ATTEMPT_CONFIG)
    model = build_model(config, _scaling())
    torch.save(model.network.state_dict(), directory / CHECKPOINT_NAME)
    parameters = int(sum(int(tensor.numel()) for tensor in model.network.parameters()))
    report = {
        "attempt_id": attempt_id,
        "config_path": ATTEMPT_CONFIG,
        "config_sha256": digest,
        "architecture": {"parameters": parameters},
        "training": {"best_epoch": 89},
        "scaling": {
            "feature_mean": [0.0] * 5,
            "feature_scale": [1.0] * 5,
            "target_mean": 0.2,
            "target_scale": 0.15,
        },
    }
    (directory / ATTEMPT_REPORT_NAME).write_text(json.dumps(report), encoding="utf-8")

    log = root / ATTEMPT_LOG_PATH
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        json.dumps({"record": "attempt", "attempt_id": attempt_id, "config_sha256": digest})
        + "\n",
        encoding="utf-8",
    )
    return directory


def test_the_recorded_scaling_is_read_and_never_refitted() -> None:
    scaling = recorded_scaling(
        {
            "scaling": {
                "feature_mean": [0.0, 1.0, 2.0, 3.0, 4.0],
                "feature_scale": [1.0, 2.0, 3.0, 4.0, 5.0],
                "target_mean": 0.17,
                "target_scale": 0.18,
            }
        }
    )
    assert scaling.price_mean == pytest.approx(0.17)
    assert scaling.price_scale == pytest.approx(0.18)
    assert scaling.feature_scale.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


@pytest.mark.parametrize(
    "section",
    [
        None,
        {"feature_mean": [0.0], "feature_scale": [0.0], "target_mean": 0.0, "target_scale": 1.0},
        {"feature_mean": [0.0], "feature_scale": [1.0], "target_mean": 0.0, "target_scale": 0.0},
        {"feature_mean": [0.0]},
    ],
)
def test_an_unusable_recorded_scaling_fails_closed(section: Any) -> None:
    with pytest.raises(LatencyDiagnosticError):
        recorded_scaling({} if section is None else {"scaling": section})


def test_the_model_is_rebuilt_from_the_tracked_configuration_and_its_checkpoint(
    tmp_path: Path,
) -> None:
    _synthetic_attempt_tree(tmp_path)
    model, provenance = load_benchmarked_model(tmp_path)
    assert provenance["attempt_id"] == BENCHMARKED_ATTEMPT
    assert provenance["head"] == "smooth_lower_floor_raw_loss"
    assert provenance["head_temperature"] == pytest.approx(1.0e-4)
    assert provenance["parameters"] == 199_041
    assert provenance["best_epoch"] == 89
    assert provenance["artifact_loading_excluded_from_timing"] is True
    assert provenance["config_path"] == ATTEMPT_CONFIG
    assert provenance["recorded_in"] == ATTEMPT_LOG_PATH
    physical = torch.tensor(
        [[1.0, 100.0, 105.0, 0.5, 0.03, 0.01, 0.25], [-1.0, 95.0, 100.0, 1.0, 0.04, 0.02, 0.2]],
        dtype=torch.float64,
    )
    with torch.no_grad():
        prices = model(physical)
    assert bool(torch.isfinite(prices).all())


def test_a_checkpoint_whose_configuration_changed_is_refused(tmp_path: Path) -> None:
    """A used configuration is immutable; a rewritten one breaks the digest."""
    _synthetic_attempt_tree(tmp_path)
    path = tmp_path / ATTEMPT_CONFIG
    path.write_text(path.read_text(encoding="utf-8") + "\n# edited after the run\n", "utf-8")
    with pytest.raises(LatencyDiagnosticError):
        load_benchmarked_model(tmp_path)


def test_a_checkpoint_from_an_unrecorded_attempt_is_refused(tmp_path: Path) -> None:
    _synthetic_attempt_tree(tmp_path)
    (tmp_path / ATTEMPT_LOG_PATH).write_text("", encoding="utf-8")
    with pytest.raises(LatencyDiagnosticError):
        load_benchmarked_model(tmp_path)


def test_a_directory_recording_another_attempt_is_refused(tmp_path: Path) -> None:
    directory = _synthetic_attempt_tree(tmp_path, attempt_id="some_other_attempt_v1")
    report = json.loads((directory / ATTEMPT_REPORT_NAME).read_text(encoding="utf-8"))
    assert report["attempt_id"] == "some_other_attempt_v1"
    with pytest.raises(LatencyDiagnosticError):
        load_benchmarked_model(tmp_path)


def test_a_mismatched_parameter_count_is_refused(tmp_path: Path) -> None:
    directory = _synthetic_attempt_tree(tmp_path)
    path = directory / ATTEMPT_REPORT_NAME
    report = json.loads(path.read_text(encoding="utf-8"))
    report["architecture"]["parameters"] = 1
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(LatencyDiagnosticError):
        load_benchmarked_model(tmp_path)


def test_a_missing_checkpoint_is_refused_rather_than_trained(tmp_path: Path) -> None:
    directory = _synthetic_attempt_tree(tmp_path)
    (directory / CHECKPOINT_NAME).unlink()
    with pytest.raises(LatencyDiagnosticError):
        load_benchmarked_model(tmp_path)


@pytest.mark.parametrize(
    ("directory", "expected"),
    [
        ("artifacts/task-9h/interpolation_test", FinalPartitionAccessError),
        ("artifacts/holdout", FinalPartitionAccessError),
        ("data/american-option-v1", LatencyDiagnosticError),
        ("/etc", AttemptError),
    ],
)
def test_the_benchmarked_directory_is_guarded(
    tmp_path: Path, directory: str, expected: type[Exception]
) -> None:
    with pytest.raises(expected):
        load_benchmarked_model(tmp_path, directory)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _measurement() -> dict[str, Any]:
    crr = [1000, 1100, 1200, 1300, 1400, 1500, 1600]
    neural = [100, 110, 120, 130, 140, 150, 160]
    paired = [a / b for a, b in zip(crr, neural, strict=True)]
    return {
        "shape": "single",
        "batch_size": 1,
        "thread_budget": 1,
        "crr_depth": MATCHED_CRR_DEPTH,
        "crr_adjacent_average_raw_ns": crr,
        "neural_end_to_end_raw_ns": {BENCHMARKED_ATTEMPT: neural},
        "median_speedup": {BENCHMARKED_ATTEMPT: sorted(paired)[3]},
        "median_speedup_confidence_interval": {
            BENCHMARKED_ATTEMPT: [min(paired), max(paired)]
        },
        "median_speedup_confidence_level_at_least": 0.95,
        "median_speedup_interval_actual_coverage": 0.984375,
    }


def test_the_reported_statistics_are_the_median_p95_and_paired_speedup() -> None:
    row = measurement_statistics(_measurement(), BENCHMARKED_ATTEMPT, quantile_method="linear")
    assert LATENCY_QUANTILES == (0.5, 0.95)
    assert row["crr_adjacent_average"]["q0.5_ns"] == pytest.approx(1300.0)
    assert row["neural_end_to_end"]["q0.5_ns"] == pytest.approx(130.0)
    assert row["crr_adjacent_average"]["q0.95_ns"] == pytest.approx(
        float(np.quantile(np.asarray([1000, 1100, 1200, 1300, 1400, 1500, 1600.0]), 0.95))
    )
    assert row["crr_adjacent_average"]["repetitions"] == 7
    assert row["paired_speedup"]["median"] == pytest.approx(10.0)
    assert row["paired_speedup"]["minimum"] == pytest.approx(10.0)
    assert row["crr_depth"] == MATCHED_CRR_DEPTH


def test_an_empty_latency_sample_fails_closed() -> None:
    measurement = _measurement()
    measurement["crr_adjacent_average_raw_ns"] = []
    with pytest.raises(LatencyDiagnosticError):
        measurement_statistics(measurement, BENCHMARKED_ATTEMPT, quantile_method="linear")


def test_the_compact_view_labels_the_result_ungated() -> None:
    report = {
        "schema_version": LATENCY_SCHEMA,
        "repository": {"commit": "0" * 40},
        "benchmarked_model": {
            "attempt_id": BENCHMARKED_ATTEMPT,
            "parameters": 199_041,
            "checkpoint_sha256": "a" * 64,
        },
        "benchmarked_attempt": {"label": "E2b"},
        "matched_crr_depth": MATCHED_CRR_DEPTH,
        "contract": {
            "crr_depths_measured": [MATCHED_CRR_DEPTH],
            "batch_sizes": [1, 8],
            "thread_budgets": [1, 1],
            "torch_interop_threads": 1,
        },
        "status": {"gate_applied": False, "selection_bias": UNGATED_LABEL},
        "acceptance_reference": {"task_9g_minimum_median_end_to_end_speedup": 10.0},
        "measurements": [
            measurement_statistics(_measurement(), BENCHMARKED_ATTEMPT, quantile_method="linear")
        ],
    }
    compact = compact_latency(report)
    assert compact["gate_applied"] is False
    assert "ungated" in compact["selection_bias"]
    assert compact["measurements"][0]["median_speedup"] == pytest.approx(10.0)
    assert compact["measurements"][0]["crr_median_ns"] == pytest.approx(1300.0)


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "output",
    [
        "artifacts/task-9h/interpolation_test/latency.json",
        "artifacts/task-9h/final/latency.json",
        "artifacts/holdout.json",
    ],
)
def test_a_final_partition_output_path_fails_closed(tmp_path: Path, output: str) -> None:
    with pytest.raises(FinalPartitionAccessError):
        _guarded_output(tmp_path, output)


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("docs/results/latency.json", LatencyDiagnosticError),
        ("/etc/latency.json", AttemptError),
        ("artifacts/../docs/latency.json", AttemptError),
    ],
)
def test_the_report_must_live_beneath_the_ignored_artifacts_tree(
    tmp_path: Path, output: str, expected: type[Exception]
) -> None:
    assert DEFAULT_OUTPUT.startswith("artifacts/")
    with pytest.raises(expected):
        _guarded_output(tmp_path, output)


def test_the_diagnostic_refuses_to_run_from_a_tree_that_is_not_this_repository(
    tmp_path: Path,
) -> None:
    """It fails on provenance, before anything could load a model or time it."""
    with pytest.raises(AttemptError):
        benchmark_matched_latency(tmp_path, DIAGNOSTIC_BENCHMARK_ATTEMPT)
    assert not (tmp_path / "artifacts").exists()


def test_the_latency_surface_exposes_no_final_evaluation_entry_point() -> None:
    import differentiable_pricing.ml.american_dev.latency as module

    names = {
        name
        for name in dir(module)
        if not name.startswith("__")
        and getattr(getattr(module, name), "__module__", None) == module.__name__
    }
    assert not [name for name in names if "final" in name.lower()]
    assert "benchmark_matched_latency" in names
    assert LATENCY_SCHEMA == "american-dev-matched-latency/1"


def test_the_measurement_is_task_9g_s_implementation_not_a_new_one() -> None:
    """``run_latency`` is imported and called; no second timing loop exists here."""
    text = LATENCY_MODULE.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(LATENCY_MODULE))
    imported = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "run_latency" in imported
    assert "perf_counter_ns" not in text
    assert "import time" not in text


def test_the_diagnostic_imports_no_partition_machinery() -> None:
    tree = ast.parse(LATENCY_MODULE.read_text(encoding="utf-8"), filename=str(LATENCY_MODULE))
    imported = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "load_partition" not in imported
    assert "verify_partition_policy" not in imported
    assert "read_partition_columns" not in imported


def test_the_script_declares_exactly_two_commands_and_no_evaluation() -> None:
    text = LATENCY_SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(LATENCY_SCRIPT))
    declared: set[str] = set()
    for node in ast.walk(tree):
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target = node.targets[0].id
        if target == "COMMANDS" and node.value is not None:
            declared = {
                element.value
                for element in getattr(node.value, "elts", [])
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            }
    assert declared == {"benchmark", "show"}
    for forbidden in ("final-evaluate", "final_evaluate"):
        assert f'"{forbidden}"' not in text and f"'{forbidden}'" not in text


# ---------------------------------------------------------------------------
# The benchmark target is explicit, and E2c cannot fall back to E2b
# ---------------------------------------------------------------------------


def test_the_registry_is_a_subset_of_the_frozen_checkpoints() -> None:
    """A timing without an immutable identity beside it says nothing."""
    from differentiable_pricing.ml.american_dev.frozen import FROZEN_ATTEMPTS

    assert set(BENCHMARK_TARGETS) <= set(FROZEN_ATTEMPTS)
    assert set(BENCHMARK_TARGETS) == {BENCHMARKED_ATTEMPT, DIAGNOSTIC_BENCHMARK_ATTEMPT}


def test_the_diagnostic_phase_benchmarks_e2c() -> None:
    assert DIAGNOSTIC_BENCHMARK_ATTEMPT == "scratch_residual_smooth_floor_margin_v1"
    assert BENCHMARK_TARGETS[DIAGNOSTIC_BENCHMARK_ATTEMPT]["label"] == "E2c"
    assert BENCHMARK_TARGETS[DIAGNOSTIC_BENCHMARK_ATTEMPT]["historical"] is False


def test_e2b_remains_the_historical_measurement_at_its_own_artifact() -> None:
    """The number the project quotes keeps its path and its meaning."""
    historical = BENCHMARK_TARGETS[BENCHMARKED_ATTEMPT]
    assert historical["label"] == "E2b"
    assert historical["historical"] is True
    assert historical["output"] == DEFAULT_OUTPUT
    assert historical["directory"] == DEFAULT_ATTEMPT_DIRECTORY


def test_each_attempt_resolves_to_its_own_directory_and_artifact() -> None:
    """Paths are derived from the attempt, never defaulted independently."""
    for attempt_id, target in BENCHMARK_TARGETS.items():
        resolved = resolve_benchmark_paths(attempt_id)
        assert resolved["attempt_id"] == attempt_id
        assert resolved["output"] == target["output"]
        assert resolved["attempt_directory"] == target["directory"]
    e2b = resolve_benchmark_paths(BENCHMARKED_ATTEMPT)
    e2c = resolve_benchmark_paths(DIAGNOSTIC_BENCHMARK_ATTEMPT)
    assert e2b["output"] != e2c["output"]
    assert e2b["attempt_directory"] != e2c["attempt_directory"]


def test_e2c_cannot_be_written_to_e2bs_historical_artifact() -> None:
    """The explicit anti-fallback guard: no reuse of the historical record."""
    with pytest.raises(LatencyDiagnosticError, match="never reused"):
        resolve_benchmark_paths(DIAGNOSTIC_BENCHMARK_ATTEMPT, DEFAULT_OUTPUT)


def test_e2b_cannot_be_forked_away_from_its_historical_artifact() -> None:
    with pytest.raises(LatencyDiagnosticError, match="fork the record"):
        resolve_benchmark_paths(
            BENCHMARKED_ATTEMPT, "artifacts/task-9h/latency/somewhere-else.json"
        )


@pytest.mark.parametrize(
    "attempt_id",
    ["scratch_direct_control_v1", "scratch_residual_smooth_floor_v1", "", "E2c"],
)
def test_an_unregistered_or_non_frozen_attempt_is_rejected(attempt_id: str) -> None:
    with pytest.raises(LatencyDiagnosticError):
        assert_benchmark_target(attempt_id)
    with pytest.raises(LatencyDiagnosticError):
        resolve_benchmark_paths(attempt_id)


def test_e2c_pointed_at_e2bs_directory_is_refused_by_the_loader(tmp_path: Path) -> None:
    """The last line of defence: the checkpoint's own report must agree.

    Even if a caller forced both paths, the loader reads the attempt report in
    that directory and refuses when its ``attempt_id`` is not the one asked for.
    So E2c can never be silently satisfied by E2b's checkpoint.
    """
    _synthetic_attempt_tree(tmp_path, attempt_id=BENCHMARKED_ATTEMPT)
    with pytest.raises(LatencyDiagnosticError, match="records attempt"):
        load_benchmarked_model(
            tmp_path, DEFAULT_ATTEMPT_DIRECTORY, DIAGNOSTIC_BENCHMARK_ATTEMPT
        )


def test_e2b_is_still_loadable_and_reproducible(tmp_path: Path) -> None:
    """The historical path keeps working exactly as it did."""
    _synthetic_attempt_tree(tmp_path, attempt_id=BENCHMARKED_ATTEMPT)
    model, provenance = load_benchmarked_model(tmp_path)
    assert provenance["attempt_id"] == BENCHMARKED_ATTEMPT
    assert provenance["artifact_loading_excluded_from_timing"] is True
    again, repeated = load_benchmarked_model(
        tmp_path, DEFAULT_ATTEMPT_DIRECTORY, BENCHMARKED_ATTEMPT
    )
    assert repeated["checkpoint_sha256"] == provenance["checkpoint_sha256"]
    assert repeated["parameters"] == provenance["parameters"]
    del model, again


def test_the_command_line_has_no_default_attempt() -> None:
    """Omitting the subject must be an error, not a silent choice of E2b."""
    import importlib.util

    specification = importlib.util.spec_from_file_location(
        "task9h_latency_cli", PROJECT_ROOT / "scripts/benchmark_american_dev_latency.py"
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    for argv in (["benchmark"], ["show"]):
        with pytest.raises(SystemExit) as raised:
            module.main(argv)
        assert raised.value.code == 2
