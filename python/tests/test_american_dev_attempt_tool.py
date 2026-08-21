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


def test_the_tracked_attempt_log_is_a_header_then_append_only_attempts() -> None:
    """The recorded search grows; the header and earlier entries do not change."""
    lines = [line for line in ATTEMPT_LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    header = json.loads(lines[0])
    assert header["record"] == "header"
    assert header["append_only"] is True
    assert header["task"] == "task-9h-american-pricer-development"
    identifiers = [json.loads(line)["attempt_id"] for line in lines[1:]]
    assert all(json.loads(line)["record"] == "attempt" for line in lines[1:])
    assert len(identifiers) == len(set(identifiers))
    assert "scratch_direct_control_v1" in identifiers


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


def _report(**overrides: Any) -> dict[str, Any]:
    report = {
        "schema_version": "american-dev-attempt-report/1",
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
        "criterion": {
            "source": "configs/american_neural_pilot_acceptance_v1.toml",
            "section": "validation_final_entry",
        },
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
    report.update(overrides)
    return report


def _canonical_log(tool: Any, root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the tool at a temporary root and return its canonical log path."""
    monkeypatch.setattr(tool, "PROJECT_ROOT", root)
    return root / "docs/attempts/task-9h-attempt-log.jsonl"


def _record_arguments(report_path: Path, log: Path) -> list[str]:
    return [
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


def test_recording_an_attempt_appends_it_to_the_log(
    tool: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The recorder builds its record from the report a completed run wrote."""
    report_path = tmp_path / "attempt-report.json"
    report_path.write_text(json.dumps(_report()), encoding="utf-8")
    log = _canonical_log(tool, tmp_path, monkeypatch)
    assert tool.main(_record_arguments(report_path, log)) == 0
    rules = tool._load_attempt_rules()
    entries = rules.attempt_log_entries(log)
    assert len(entries) == 1
    assert entries[0]["outcome"] == "criterion_not_met"
    assert entries[0]["price_metrics"]["gate"]["passed"] is False
    assert entries[0]["bound_and_shape_diagnostics"]["material_bound_violations"] == 1


def test_recording_refuses_any_log_but_the_canonical_tracked_one(
    tool: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second log would record an attempt where no checker ever reads it."""
    report_path = tmp_path / "attempt-report.json"
    report_path.write_text(json.dumps(_report()), encoding="utf-8")
    _canonical_log(tool, tmp_path, monkeypatch)
    elsewhere = tmp_path / "attempts.jsonl"
    assert tool.main(_record_arguments(report_path, elsewhere)) == 2
    assert not elsewhere.exists()


def test_recording_defaults_to_the_canonical_log_when_none_is_given(
    tool: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path = tmp_path / "attempt-report.json"
    report_path.write_text(json.dumps(_report()), encoding="utf-8")
    log = _canonical_log(tool, tmp_path, monkeypatch)
    arguments = _record_arguments(report_path, log)
    del arguments[arguments.index("--log") : arguments.index("--log") + 2]
    assert tool.main(arguments) == 0
    assert log.is_file()


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"schema_version": "american-dev-attempt-report/2"}, "report schema"),
        ({"schema_version": None}, "report schema"),
        (
            {"criterion": {"source": "configs/looser.toml", "section": "validation_final_entry"}},
            "criterion source",
        ),
        (
            {
                "criterion": {
                    "source": "configs/american_neural_pilot_acceptance_v1.toml",
                    "section": "final_accuracy",
                }
            },
            "criterion section",
        ),
    ],
    ids=["wrong-schema", "missing-schema", "other-criterion-file", "other-criterion-section"],
)
def test_recording_refuses_an_unrecognized_report_or_criterion(
    tool: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, Any],
    reason: str,
) -> None:
    """A report this workbench did not write is not evidence of an attempt."""
    report_path = tmp_path / "attempt-report.json"
    report_path.write_text(json.dumps(_report(**overrides)), encoding="utf-8")
    log = _canonical_log(tool, tmp_path, monkeypatch)
    assert tool.main(_record_arguments(report_path, log)) == 2
    assert not log.exists()


def test_the_static_scan_shares_the_runtime_forbidden_partition_tokens(tool: Any) -> None:
    """One definition: a token added for the guard cannot be missed by the scan."""
    rules = tool._load_attempt_rules()
    assert tool.FORBIDDEN_PARTITIONS == rules.FORBIDDEN_PARTITION_TOKENS
    assert tool.FORBIDDEN_PARTITIONS is tool.RULES.FORBIDDEN_PARTITION_TOKENS
    # Derived, not copied: the tool may not carry its own literal token list.
    tree = ast.parse(TOOL.read_text(encoding="utf-8"), filename=str(TOOL))
    assignments = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "FORBIDDEN_PARTITIONS"
    ]
    assert len(assignments) == 1
    assert isinstance(assignments[0], ast.Attribute)
    for token in ("interpolation_test", "boundary_test", "holdout", "final"):
        assert token in tool.FORBIDDEN_PARTITIONS
        assert tool.names_a_partition(token) is True


def test_the_check_reports_an_attempt_that_used_another_criterion(
    tool: Any, tmp_path: Path
) -> None:
    """The negative control for "the criterion is fixed, not revised"."""
    rules = tool._load_attempt_rules()
    acceptance = tmp_path / rules.ACCEPTANCE_CONFIG_PATH
    acceptance.parent.mkdir(parents=True)
    acceptance.write_text('schema_version = "x"\n', encoding="utf-8")
    digest = rules.sha256_file(acceptance)
    log = tmp_path / "log.jsonl"

    def _entry(**criterion: Any) -> str:
        return json.dumps(
            {"record": "attempt", "attempt_id": "x", "criterion": criterion},
        )

    header = json.dumps({"record": "header", "schema_version": "american-dev-attempt-log/1"})
    canonical = {
        "source": rules.ACCEPTANCE_CONFIG_PATH,
        "section": rules.CRITERION_SECTION,
        "sha256": digest,
    }
    log.write_text("\n".join([header, _entry(**canonical)]) + "\n", encoding="utf-8")
    assert rules.check_recorded_criterion(tmp_path, log) == []

    log.write_text(
        "\n".join([header, _entry(**{**canonical, "sha256": "0" * 64})]) + "\n", encoding="utf-8"
    )
    assert any("the criterion is fixed" in failure for failure in
               rules.check_recorded_criterion(tmp_path, log))

    log.write_text(
        "\n".join([header, _entry(**{**canonical, "section": "final_accuracy"})]) + "\n",
        encoding="utf-8",
    )
    assert any("criterion section" in failure for failure in
               rules.check_recorded_criterion(tmp_path, log))


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
