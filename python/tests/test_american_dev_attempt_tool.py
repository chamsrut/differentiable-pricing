"""The Task 9H offline attempt-log tool, run the way `check.sh` and CI run it.

Deliberately outside `python/tests/ml/`: the tool imports nothing from the
project package and needs neither PyTorch nor a compiled pricing extension, so
it belongs to the lightweight CI job — and that is exactly the property being
asserted, since a check that only runs where PyTorch happens to be installed is
not much of a check.

It runs no pricing, no training and no dataset access.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from typing import Any, Final

import pytest

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
TOOL: Final = PROJECT_ROOT / "scripts/american_dev_attempts.py"
PACKAGE: Final = PROJECT_ROOT / "python/src/differentiable_pricing/ml/american_dev"
RUNNER: Final = PROJECT_ROOT / "scripts/run_american_dev_attempt.py"
ATTEMPT_LOG: Final = PROJECT_ROOT / "docs/attempts/task-9h-attempt-log.jsonl"
MANUAL_ONLY: Final = ("run_american_dev_attempt.py", "run_american_neural_pilot.py")


@pytest.fixture(scope="module")
def tool() -> Any:
    specification = importlib.util.spec_from_file_location("task9h_attempt_tool", TOOL)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_the_repository_passes_its_own_attempt_checks(tool: Any) -> None:
    """The same invocation `scripts/check.sh` and the CI python job perform."""
    assert tool.main(["check"]) == 0


def test_the_tool_needs_no_torch_and_no_compiled_extension() -> None:
    tree = ast.parse(TOOL.read_text(encoding="utf-8"), filename=str(TOOL))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not [name for name in imported if name.split(".")[0] in {"torch", "numpy"}]
    assert not [name for name in imported if name.startswith("differentiable_pricing")]


def test_no_repository_check_or_ci_job_launches_a_training_run() -> None:
    """Training stays a manual, terminal-invoked human job."""
    for relative in ("scripts/check.sh", ".github/workflows/ci.yml"):
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        for command in MANUAL_ONLY:
            assert command not in text
        assert "american_dev_attempts.py check" in text


def test_no_task_9h_source_names_a_final_partition_as_data(tool: Any) -> None:
    assert tool.check_no_final_partition_references() == []


def test_the_partition_scan_distinguishes_a_path_from_prose(tool: Any) -> None:
    """Prose has to be able to state the prohibition; a path may not name it."""
    assert tool.names_a_partition("interpolation_test") is True
    assert tool.names_a_partition("data/american-option-v1/interpolation_test.parquet") is True
    assert tool.names_a_partition("holdout") is True
    assert (
        tool.names_a_partition("interpolation_test was never opened, hashed or counted") is False
    )
    assert tool.names_a_partition("train") is False
    assert tool.names_a_partition("validation") is False


def test_the_runner_exposes_no_final_evaluation_command(tool: Any) -> None:
    assert tool.check_runner_has_no_final_evaluation() == []
    text = RUNNER.read_text(encoding="utf-8")
    for forbidden in ("final-evaluate", "final_evaluate"):
        assert f'"{forbidden}"' not in text
        assert f"'{forbidden}'" not in text


def test_the_runner_declares_exactly_the_two_supported_subcommands(tool: Any) -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"), filename=str(RUNNER))
    declared: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "COMMANDS"
            and node.value is not None
        ):
            declared = {
                element.value
                for element in getattr(node.value, "elts", [])
                if isinstance(element, ast.Constant)
            }
    assert declared == tool.EXPECTED_SUBCOMMANDS == {"run", "status"}


def test_the_check_would_catch_a_runner_that_added_a_final_evaluation_command(
    tool: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A negative control: the check fails when the property it guards is broken."""
    offending = tmp_path / "runner.py"
    offending.write_text('COMMANDS = ("run", "status", "final-evaluate")\n', encoding="utf-8")
    monkeypatch.setattr(tool, "RUNNER", offending)
    failures = tool.check_runner_has_no_final_evaluation()
    assert failures
    assert any("no final-evaluation command" in failure for failure in failures)


def test_every_tracked_attempt_configuration_is_valid_and_marked_immutable(tool: Any) -> None:
    rules = tool._load_attempt_rules()
    assert tool.check_attempt_configurations(rules) == []
    configurations = sorted(PROJECT_ROOT.glob(tool.ATTEMPT_CONFIG_GLOB))
    identifiers = set()
    for path in configurations:
        assert "IMMUTABLE AFTER USE" in path.read_text(encoding="utf-8")
        identifiers.add(path.stem.removeprefix("american_dev_attempt_"))
    assert identifiers == {
        "scratch_direct_control_v1",
        "scratch_capacity_v1",
        "scratch_residual_architecture_v1",
        "scratch_american_premium_v1",
        "scratch_conditioning_v1",
    }


def test_the_check_would_catch_an_edited_used_configuration(
    tool: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The negative control for "immutable after use"."""
    import hashlib

    configuration = tmp_path / "configs/american_dev_attempt_x.toml"
    configuration.parent.mkdir(parents=True)
    configuration.write_text('attempt_id = "x"\n', encoding="utf-8")
    digest = hashlib.sha256(configuration.read_bytes()).hexdigest()
    log = tmp_path / "log.jsonl"
    log.write_text(
        "\n".join(
            [
                json.dumps({"record": "header", "schema_version": "american-dev-attempt-log/1"}),
                json.dumps(
                    {
                        "record": "attempt",
                        "attempt_id": "x",
                        "config_path": "configs/american_dev_attempt_x.toml",
                        "config_sha256": digest,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rules = tool._load_attempt_rules()
    assert rules.check_configuration_immutability(tmp_path, log) == []

    configuration.write_text('attempt_id = "x"\nedited = true\n', encoding="utf-8")
    failures = rules.check_configuration_immutability(tmp_path, log)
    assert failures and "immutable after use" in failures[0]


def test_the_tracked_attempt_log_starts_with_its_header_and_no_attempts() -> None:
    lines = [line for line in ATTEMPT_LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 1
    header = json.loads(lines[0])
    assert header["record"] == "header"
    assert header["append_only"] is True
    assert header["task"] == "task-9h-american-pricer-development"


def test_the_task_9h_package_is_exactly_the_five_retained_modules(tool: Any) -> None:
    """Nothing is kept "for later": Greeks, latency, IV and transfer are absent."""
    assert {path.name for path in PACKAGE.glob("*.py")} == {
        "__init__.py",
        "attempts.py",
        "models.py",
        "representation.py",
        "workbench.py",
    }
    assert {path.name for path in tool._task_9h_sources()} >= {
        "attempts.py",
        "workbench.py",
        "run_american_dev_attempt.py",
    }


def test_recording_an_attempt_appends_it_to_the_log(tool: Any, tmp_path: Path) -> None:
    """The recorder builds its record from the report a completed run wrote."""
    report = {
        "attempt_id": "unit_test_attempt",
        "parent_attempt": "",
        "hypothesis": "a hypothesis",
        "repository": {"commit": "0" * 40},
        "config_path": "configs/american_dev_attempt_scratch_direct_control_v1.toml",
        "config_sha256": "1" * 64,
        "source_digests": {},
        "architecture": {"name": "smooth_mlp"},
        "features": [],
        "target_and_reconstruction": {},
        "seeds": {"initialization": 1, "shuffle": 2},
        "selected_rows": {"partition": "train", "salt": "s", "rule": "r", "count": 8},
        "optimizer_and_budget": {},
        "training": {"best_epoch": 3},
        "criterion": {"section": "validation_final_entry"},
        "price_metrics": {
            "slices": {"overall": {"rows": 4, "normalized": {"rmse": 0.5}}},
            "diagnostics": {
                "material_bound_violations": 1,
                "material_shape_violations": 0,
                "checks": {},
            },
            "gate": {"checks": {}, "passed": False},
            "european_crr_baseline": {"normalized": {"rmse": 0.6}},
        },
    }
    report_path = tmp_path / "attempt-report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    log = tmp_path / "attempts.jsonl"
    assert (
        tool.main(
            [
                "record",
                "--report",
                str(report_path),
                "--log",
                str(log),
                "--outcome",
                "criterion_not_met",
                "--interpretation",
                "prices did not reach the criterion",
                "--next-action",
                "continue with more capacity",
            ]
        )
        == 0
    )
    rules = tool._load_attempt_rules()
    entries = rules.attempt_log_entries(log)
    assert len(entries) == 1
    assert entries[0]["outcome"] == "criterion_not_met"
    assert entries[0]["price_metrics"]["gate"]["passed"] is False
    assert entries[0]["bound_and_shape_diagnostics"]["material_bound_violations"] == 1


def test_recording_refuses_an_unknown_outcome_or_next_action(tool: Any, tmp_path: Path) -> None:
    report_path = tmp_path / "attempt-report.json"
    report_path.write_text(json.dumps({"attempt_id": "x"}), encoding="utf-8")
    common = ["record", "--report", str(report_path), "--log", str(tmp_path / "a.jsonl")]
    bad_outcome = [*common, "--outcome", "looked_good", "--interpretation", "x"]
    assert tool.main([*bad_outcome, "--next-action", "stop"]) == 2
    assert (
        tool.main(
            [*common, "--outcome", "abandoned", "--interpretation", "x", "--next-action", "maybe"]
        )
        == 2
    )
