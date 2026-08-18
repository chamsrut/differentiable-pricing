"""Fixture and mutation tests for the task 9C-C3 v2 snapshot freeze tool.

Every report here is a tiny synthetic fixture built from the checked-in v2
configuration and hand-written solve records. **No PDE solve runs, and no v2
snapshot is created under ``docs/results/``**: the tool writes only into
``tmp_path``.

The mutation tests are the point of the file. Each one edits a published
report or a published snapshot, re-serialises it canonically so no file-level
hash check can catch it, and requires the tool to reject it anyway — because
the tool recomputes every verdict, count and lifecycle field from the raw
per-solve numbers rather than reading them back.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from differentiable_pricing.american import pde_label_policy_v2 as v2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts" / "freeze_pde_label_policy_v2_results.py"
CONFIG_V2 = PROJECT_ROOT / "configs" / "pde_label_policy_pilot_v2.toml"


def _load_script() -> Any:
    specification = importlib.util.spec_from_file_location("freeze_v2", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


freeze = _load_script()


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

_PRICE = 60.0
_RESIDUAL = 1.0e-12
_STENCIL_ERROR = 1.0e-3
_VEGA_SLOPE = 12.0
_CONVEXITY = 0.005
_SCALE_BY_GRID = {
    v2.ORDER_PROBE_GRID: 1.0,
    v2.CANDIDATE_GRID: 0.25,
    v2.REFERENCE_GRID: 0.0625,
    v2.ANCHOR_GRID: 0.015625,
}


@pytest.fixture(scope="module")
def config() -> v2.PolicyConfigV2:
    return v2.load_policy_config_v2(CONFIG_V2)


def _solve(
    case: v2.CaseV2, config: v2.PolicyConfigV2, grid_name: str, role: str
) -> v2.SurfaceSolve:
    grid = config.grid(grid_name)
    style, volatility = v2.planned_role_state(case, role)
    price = _PRICE + _VEGA_SLOPE * (volatility - case.volatility)
    if role == v2.DOMINANCE_CONTROL_ROLE:
        price = _PRICE - 1.0
    true_delta = 0.5 if case.option_type == "call" else -0.5
    bumped: dict[str, float] = {}
    if role == v2.BASE_ROLE and grid_name in {v2.CANDIDATE_GRID, v2.REFERENCE_GRID}:
        for bump in config.bumps.spot:
            for direction, sign in (("down", -1.0), ("up", 1.0)):
                bumped[v2.spot_bump_key(bump, direction)] = (
                    _PRICE + sign * true_delta * bump + _CONVEXITY * bump * bump
                )
    step, intervals = v2.expected_spot_grid(
        strike=case.strike,
        spot_maximum=config.solver.spot_maximum,
        spot_intervals=grid.spot_intervals,
    )
    solve = v2.SurfaceSolve(
        grid_name=grid_name,
        role=role,
        exercise_style=style,
        volatility=volatility,
        solver_status="discrete_system_converged",
        time_steps=grid.time_steps,
        maximum_lcp_residual=_RESIDUAL,
        spot_step=step,
        spot_intervals=intervals,
        pricing_grid_identity="",
        centre_node_index=round(case.spot / step),
        centre_spot_is_exact_node=True,
        centre_price=price,
        centre_delta=true_delta + _STENCIL_ERROR * _SCALE_BY_GRID[grid_name],
        centre_gamma=0.02,
        centre_greek_eligible=True,
        centre_greek_eligibility_reason="eligible",
        centre_exercise_state="no_obstacle" if style == "european" else "continuation",
        backward_inductions=1,
        linear_solves=grid.time_steps,
        psor_solves=0 if style == "european" else grid.time_steps,
        psor_total_iterations=0 if style == "european" else 7 * grid.time_steps,
        bumped_prices=bumped,
        bumped_spots_are_exact_nodes=True,
    )
    return dataclasses.replace(
        solve, pricing_grid_identity=v2.expected_pricing_grid_identity(case, config, solve)
    )


def _drop(case: v2.CaseV2, solves: dict, key: tuple[str, str]) -> tuple[dict, list]:
    remaining = {other: solve for other, solve in solves.items() if other != key}
    failure = v2.SolveFailure(
        case=case.name,
        grid=key[0],
        role=key[1],
        option_type=case.option_type,
        exercise_style=v2.planned_role_state(case, key[1])[0],
        criticality=v2.solve_criticality(case, key[0], key[1]),
        exception_class="PsorConvergenceError",
        message="injected",
    )
    return remaining, [failure]


def _report(
    config: v2.PolicyConfigV2, *, stage: str = "remediation", fail: bool = True
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name in config.stage_case_names(stage):
        case = config.case(name)
        solves = {
            key: _solve(case, config, key[0], key[1])
            for key in v2.required_solve_keys(case, config)
        }
        failures: list[v2.SolveFailure] = []
        if fail and name == config.gate_cases[0]:
            solves, failures = _drop(case, solves, (v2.CANDIDATE_GRID, v2.BASE_ROLE))
        rows.append(v2.evaluate_case(case, config, solves, failures))
    outcome = v2.stage_outcome(rows, config)
    accounting = v2.solve_accounting(rows, config, stage)
    lifecycle = v2.lifecycle_block(
        stage=stage,
        all_gate_eligible_cases_pass=bool(outcome["all_gate_eligible_cases_pass"]),
        config=config,
    )
    report = v2.build_report(
        config=config,
        stage=stage,
        cases=rows,
        outcome=outcome,
        accounting=accounting,
        lifecycle=lifecycle,
        performance={
            "wall_seconds": 0.0,
            "timed_solve_attempts": 0,
            "timed_solve_seconds": 0.0,
            "peak_resident_memory_bytes": 0,
            "peak_resident_memory_increase_bytes": 0,
            "projection_caveat": "fixture",
            "host": {"system": "fixture", "machine": "fixture", "python": "3"},
        },
    )
    return json.loads(v2.canonical_json(report))


def _write(path: Path, report: Mapping) -> Path:
    path.write_text(v2.canonical_json(report), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Finding 12: CLI and documentation consistency
# ---------------------------------------------------------------------------


def test_a_mode_flag_is_always_required(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        freeze.main(["--output", str(tmp_path / "s.json")])


def test_check_fails_when_the_expected_snapshot_is_absent(tmp_path: Path) -> None:
    assert freeze.main(["--check", "--output", str(tmp_path / "absent.json")]) == 2


def test_the_v2_snapshot_is_checked_in_and_valid() -> None:
    """Task 9C-C3 is terminal, so the snapshot this tool enforces must be present.

    This assertion is the inverse of the pre-run one it replaces
    (``test_no_v2_snapshot_is_checked_in_yet``), which asserted the snapshot was
    absent and ``--check`` therefore failed by design. That predicate expired
    when the confirmation stage was frozen. ``--check`` still refuses a *missing*
    snapshot: ``test_check_fails_when_the_expected_snapshot_is_absent`` above
    covers that on a temporary path, so the refusal stays under test.
    """
    assert freeze.DEFAULT_SNAPSHOT.is_file()
    assert freeze.main(["--check"]) == 0


def test_the_canonical_snapshot_filename_is_consistent() -> None:
    name = "american_pde_label_policy_v2_results_v1.json"
    assert freeze.DEFAULT_SNAPSHOT.name == name
    assert freeze.DEFAULT_SNAPSHOT.parent == PROJECT_ROOT / "docs/results"
    for document in (
        PROJECT_ROOT / "docs/pde-numerical-contract.md",
        PROJECT_ROOT / "CONTRIBUTING.md",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "docs/project-state.md",
    ):
        assert name in document.read_text(), document


def test_the_documented_commands_carry_the_required_mode_flag() -> None:
    contract = (PROJECT_ROOT / "docs/pde-numerical-contract.md").read_text()
    assert "freeze_pde_label_policy_v2_results.py \\\n  --extract" in contract
    assert "freeze_pde_label_policy_v2_results.py --check" in contract
    assert "pde-label-policy-v2-remediation/report.json" in contract
    assert "pde-label-policy-v2-confirmation/report.json" in contract


def test_the_freeze_tool_is_wired_into_the_gate_and_ci_exactly_once() -> None:
    """The inverse of the pre-run assertion, now that a snapshot exists to enforce.

    Wiring was deliberately deferred until the change that first commits a v2
    snapshot, because ``--check`` fails when the snapshot is absent. That change
    has happened, so the check must now run in the local gate and in CI — and in
    each of them exactly once, so no job pays for it twice.
    """
    invocation = "scripts/freeze_pde_label_policy_v2_results.py --check"

    assert (PROJECT_ROOT / "scripts/check.sh").read_text().count(invocation) == 1

    workflows = sorted((PROJECT_ROOT / ".github/workflows").glob("*.yml"))
    assert workflows, "no CI workflow found"
    total = sum(workflow.read_text().count(invocation) for workflow in workflows)
    assert total == 1, "the CI check must appear in exactly one job, exactly once"


def test_check_rejects_update(tmp_path: Path) -> None:
    assert freeze.main(["--check", "--update", "--output", str(tmp_path / "x.json")]) == 2


def test_extract_requires_a_report(tmp_path: Path) -> None:
    assert freeze.main(["--extract", "--output", str(tmp_path / "s.json")]) == 2


# ---------------------------------------------------------------------------
# Extract, then check
# ---------------------------------------------------------------------------


def test_a_remediation_failure_report_freezes_and_checks(
    config: v2.PolicyConfigV2, tmp_path: Path
) -> None:
    report = _write(tmp_path / "report.json", _report(config))
    snapshot = tmp_path / "snapshot.json"
    assert freeze.main(["--extract", "--report", str(report), "--output", str(snapshot)]) == 0
    payload = json.loads(snapshot.read_text())
    assert payload["schema_version"] == freeze.SNAPSHOT_SCHEMA_VERSION
    assert payload["lifecycle"]["remediation_status"] == "failed"
    assert payload["lifecycle"]["confirmation_status"] == "not_run"
    assert payload["lifecycle"]["confirmation_not_run_reason"] == "remediation_failed"
    assert payload["lifecycle"]["selected_accuracy_policy"] == v2.NO_POLICY_SELECTED
    assert payload["outcome"]["gate_eligible_failure_count"] == 1
    assert payload["solve_accounting"]["attempted_surface_solves"] == 116
    assert payload["solve_accounting"]["solver_exceptions"] == 1
    failing = next(case for case in payload["cases"] if not case["passes"])
    assert failing["solve_failures"][0]["criticality"] == v2.PRICE_CRITICAL
    assert freeze.main(["--check", "--output", str(snapshot)]) == 0
    assert (
        freeze.main(["--check", "--output", str(snapshot), "--report", str(report)]) == 0
    )


def test_a_confirmation_selection_freezes_pending_fresh_approval(
    config: v2.PolicyConfigV2, tmp_path: Path
) -> None:
    report = _write(
        tmp_path / "report.json", _report(config, stage="confirmation", fail=False)
    )
    snapshot = tmp_path / "snapshot.json"
    assert freeze.main(["--extract", "--report", str(report), "--output", str(snapshot)]) == 0
    payload = json.loads(snapshot.read_text())
    assert payload["lifecycle"]["confirmation_status"] == "run"
    assert payload["lifecycle"]["selected_accuracy_policy"] == v2.CANDIDATE_GRID
    assert payload["lifecycle"]["selection_pending_fresh_top_level_approval"] is True
    assert payload["lifecycle"]["authorizes_dataset_generation"] is False
    assert payload["study"]["case_count"] == v2.CONFIRMATION_CASE_COUNT
    assert payload["solve_accounting"]["attempted_surface_solves"] == 323
    assert payload["outcome"]["gamma_eligible_cases"] == []
    assert all(case["gamma_label_eligible"] is False for case in payload["cases"])
    assert freeze.main(["--check", "--output", str(snapshot)]) == 0


def test_extraction_refuses_to_overwrite_without_update(
    config: v2.PolicyConfigV2, tmp_path: Path
) -> None:
    report = _write(tmp_path / "report.json", _report(config))
    snapshot = tmp_path / "snapshot.json"
    arguments = ["--extract", "--report", str(report), "--output", str(snapshot)]
    assert freeze.main(arguments) == 0
    assert freeze.main(arguments) == 2
    assert freeze.main([*arguments, "--update"]) == 0


def test_regeneration_is_byte_identical(config: v2.PolicyConfigV2, tmp_path: Path) -> None:
    report = _write(tmp_path / "report.json", _report(config))
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    for output in (first, second):
        assert (
            freeze.main(["--extract", "--report", str(report), "--output", str(output)]) == 0
        )
    assert first.read_bytes() == second.read_bytes()


def test_a_nonterminal_passing_remediation_may_not_be_frozen(
    config: v2.PolicyConfigV2, tmp_path: Path
) -> None:
    report = _report(config, fail=False)
    assert report["lifecycle"]["freezable"] is False
    assert "selected_accuracy_policy" not in report["lifecycle"]
    path = _write(tmp_path / "report.json", report)
    assert (
        freeze.main(["--extract", "--report", str(path), "--output", str(tmp_path / "s.json")])
        == 2
    )
    with pytest.raises(freeze.FreezeError, match=r"nonterminal state"):
        freeze.validate_report(report, config)


# ---------------------------------------------------------------------------
# Finding 6: report-level mutations, with the file hash regenerated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        pytest.param(
            lambda r: r["cases"][0]["price"].update({"absolute_error": 999}),
            "disagrees with the decisions",
            id="price_absolute_error_999",
        ),
        pytest.param(
            lambda r: r["cases"][0].update({"invented_block": {}}),
            r"unknown fields|object key set is wrong",
            id="unknown_case_key",
        ),
        pytest.param(
            lambda r: r["criteria_block"]["criteria"].update({"shape_absolute_floor": 1.0e-6}),
            "criteria_block does not match",
            id="changed_shape_absolute_floor",
        ),
        pytest.param(
            lambda r: r["criteria_block"]["criteria"].update(
                {"minimum_supported_bump_order": 1.2}
            ),
            "criteria_block does not match",
            id="changed_order_bound",
        ),
        pytest.param(
            lambda r: r["criteria_block"].update({"supported_order_band": [1.2, 2.5]}),
            "criteria_block does not match",
            id="changed_order_band",
        ),
        pytest.param(
            lambda r: next(
                case for case in r["cases"] if not case["passes"]
            )["checks"].update({"price_critical_solves_returned_and_converged": True}),
            # Now caught at admission by the exact failed_checks_by_name
            # reconciliation; the recomputation gate still stands behind it.
            r"disagrees with the decisions|failed_checks_by_name",
            id="changed_gate_boolean",
        ),
        pytest.param(
            lambda r: r["cases"].pop(4),
            r"missing=|attempted_by_case_classification value is wrong",
            id="removed_case",
        ),
        pytest.param(
            lambda r: r["cases"].append(copy.deepcopy(r["cases"][0])),
            r"duplicated=|differing=\['case_count'\]"
            r"|attempted_by_case_classification value is wrong",
            id="duplicated_case",
        ),
        pytest.param(
            lambda r: r["lifecycle"].update({"remediation_status": "passed"}),
            "disagrees with the value recomputed",
            id="altered_lifecycle",
        ),
        pytest.param(
            lambda r: r["source"].update({"pde_implementation_sha256": "0" * 64}),
            "does not match the repository",
            id="altered_source_digest",
        ),
        pytest.param(
            lambda r: r["source"].update({"v2_runner_module_sha256": "0" * 64}),
            "does not match the repository",
            id="altered_runner_digest",
        ),
        pytest.param(
            lambda r: r["source"].update({"canonical_payload_module_sha256": "0" * 64}),
            "does not match the repository",
            id="altered_shared_source_digest",
        ),
        pytest.param(
            lambda r: r["source"]["executable_source_inventory"].pop("pde_binding_source"),
            r"not the declared inventory|object key set is wrong",
            id="shrunken_inventory",
        ),
        pytest.param(
            lambda r: r["cases"][0]["delta_eligibility"].update({"eligible": True}),
            "disagrees with the decisions",
            id="fabricated_delta_eligibility",
        ),
        pytest.param(
            lambda r: r["cases"][0]["vega_eligibility"].update({"eligible": True}),
            "disagrees with the decisions",
            id="fabricated_vega_eligibility",
        ),
        pytest.param(
            lambda r: r["cases"][1]["gamma_evaluation_only"].update(
                {"supervision_eligible": True}
            ),
            "disagrees with the decisions",
            id="fabricated_gamma_eligibility",
        ),
        pytest.param(
            lambda r: r["stage_outcome"].update({"all_gate_eligible_cases_pass": True}),
            "stage_outcome disagrees",
            id="fabricated_aggregate",
        ),
        pytest.param(
            lambda r: r["solve_accounting"].update({"solver_exceptions": 0}),
            "solve_accounting disagrees",
            id="fabricated_accounting",
        ),
        pytest.param(
            lambda r: r["cases"][1]["solves"][1].update({"centre_greek_eligible": False}),
            "disagrees with the decisions",
            id="edited_raw_structural_flag",
        ),
        pytest.param(
            lambda r: r["cases"][1]["solves"][1].update({"solver_status": "boom"}),
            r"disagrees with the decisions|expected one of",
            id="edited_raw_solver_status",
        ),
    ],
)
def test_a_mutated_report_is_refused_even_with_a_regenerated_hash(
    config: v2.PolicyConfigV2, tmp_path: Path, mutate, match: str
) -> None:
    report = _report(config)
    mutate(report)
    # Re-serialised canonically and re-hashed at extraction time: no file-level
    # digest can catch this. The recomputation must.
    path = _write(tmp_path / "report.json", report)
    with pytest.raises(freeze.FreezeError, match=match):
        freeze.snapshot_from_report(path, config)
    assert (
        freeze.main(["--extract", "--report", str(path), "--output", str(tmp_path / "s.json")])
        == 2
    )
    assert not (tmp_path / "s.json").exists()


# ---------------------------------------------------------------------------
# Finding 6: snapshot-level mutations, canonically re-serialised
# ---------------------------------------------------------------------------


def _frozen(config: v2.PolicyConfigV2, tmp_path: Path, **report_kwargs: Any) -> Path:
    report = _write(tmp_path / "report.json", _report(config, **report_kwargs))
    snapshot = tmp_path / "snapshot.json"
    assert freeze.main(["--extract", "--report", str(report), "--output", str(snapshot)]) == 0
    return snapshot


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda s: s["cases"][0].update({"price_absolute_error": 999}),
            id="price_absolute_error_999",
        ),
        pytest.param(
            lambda s: s["cases"][0].update({"invented_key": 1}), id="unknown_case_key"
        ),
        pytest.param(
            lambda s: s["predeclared_criteria"].update({"shape_absolute_floor": 1.0e-6}),
            id="changed_shape_absolute_floor",
        ),
        pytest.param(
            lambda s: s["predeclared_criteria"].update(
                {"maximum_supported_stencil_order": 3.0}
            ),
            id="changed_order_bound",
        ),
        pytest.param(
            lambda s: next(
                case for case in s["cases"] if not case["passes"]
            ).update({"passes": True, "failed_checks": []}),
            id="changed_gate_boolean",
        ),
        pytest.param(lambda s: s["cases"].pop(3), id="removed_case"),
        pytest.param(
            lambda s: s["cases"].append(copy.deepcopy(s["cases"][0])), id="duplicated_case"
        ),
        pytest.param(
            lambda s: s["lifecycle"].update({"selected_accuracy_policy": v2.CANDIDATE_GRID}),
            id="altered_lifecycle",
        ),
        pytest.param(
            lambda s: s["source"].update({"pde_header_sha256": "0" * 64}),
            id="altered_source_digest",
        ),
        pytest.param(
            lambda s: s["source"].update({"v1_label_policy_module_sha256": "0" * 64}),
            id="altered_shared_source_digest",
        ),
        pytest.param(
            lambda s: s.update({"criteria_digest": "crit-" + "0" * 64}),
            id="altered_criteria_digest",
        ),
        pytest.param(
            lambda s: s["study"].update({"gate_cases": s["study"]["gate_cases"][:8]}),
            id="shrunken_gate_set",
        ),
        pytest.param(
            lambda s: s["study"].update({"remediation_cases": v2.REMEDIATION_CASES[:9]}),
            id="shrunken_remediation_set",
        ),
        pytest.param(
            lambda s: s["outcome"].update({"delta_eligible_cases": []}),
            id="fabricated_eligibility_aggregate",
        ),
        pytest.param(
            lambda s: s["cases"][0].update({"gamma_label_eligible": True}),
            id="gamma_becomes_eligible",
        ),
        pytest.param(lambda s: s["non_claims"].pop(), id="dropped_non_claim"),
        pytest.param(lambda s: s["limitations"].pop(0), id="dropped_limitation"),
    ],
)
def test_a_mutated_snapshot_fails_check_even_when_re_serialised(
    config: v2.PolicyConfigV2, tmp_path: Path, mutate
) -> None:
    snapshot = _frozen(config, tmp_path)
    payload = json.loads(snapshot.read_text())
    mutate(payload)
    # Canonically re-serialised, so the "is it canonical?" check cannot be
    # what rejects it.
    snapshot.write_text(freeze.serialise(payload), encoding="utf-8")
    assert freeze.main(["--check", "--output", str(snapshot)]) == 2


def test_a_non_canonically_serialised_snapshot_fails_check(
    config: v2.PolicyConfigV2, tmp_path: Path
) -> None:
    snapshot = _frozen(config, tmp_path)
    payload = json.loads(snapshot.read_text())
    snapshot.write_text(json.dumps(payload), encoding="utf-8")
    assert freeze.main(["--check", "--output", str(snapshot)]) == 2


def test_a_stale_snapshot_is_detected_against_its_report(
    config: v2.PolicyConfigV2, tmp_path: Path
) -> None:
    snapshot = _frozen(config, tmp_path)
    report_path = tmp_path / "report.json"
    _write(report_path, _report(config, stage="confirmation", fail=False))
    assert (
        freeze.main(["--check", "--output", str(snapshot), "--report", str(report_path)]) == 2
    )


# ---------------------------------------------------------------------------
# Provenance and stated limits
# ---------------------------------------------------------------------------


def test_the_snapshot_records_and_reconciles_every_executable_source(
    config: v2.PolicyConfigV2, tmp_path: Path
) -> None:
    payload = json.loads(_frozen(config, tmp_path).read_text())
    source = payload["source"]
    assert source["executable_source_inventory"] == dict(v2.EXECUTABLE_SOURCE_INVENTORY)
    digests = v2.executable_source_digests()
    for name in v2.EXECUTABLE_SOURCE_INVENTORY:
        assert source[f"{name}_sha256"] == digests[f"{name}_sha256"]
    assert source["executable_source_composite_digest"] == (
        v2.executable_source_composite_digest(digests)
    )
    assert source["source_digest_limitation"] == v2.SOURCE_DIGEST_LIMITATION


def test_a_foreign_configuration_is_refused(
    config: v2.PolicyConfigV2, tmp_path: Path
) -> None:
    snapshot = _frozen(config, tmp_path)
    payload = json.loads(snapshot.read_text())
    payload["source"]["raw_config_sha256"] = "0" * 64
    payload["lifecycle"]["raw_config_sha256"] = "0" * 64
    with pytest.raises(freeze.FreezeError, match="does not match the checked-in configuration"):
        freeze.validate_snapshot(payload, config)


def test_the_snapshot_states_the_coordinated_rewrite_limitation(
    config: v2.PolicyConfigV2, tmp_path: Path
) -> None:
    payload = json.loads(_frozen(config, tmp_path).read_text())
    text = " ".join(payload["limitations"])
    assert "cannot authenticate a fully coordinated fabricated numerical report" in text
    assert "does not re-solve" in text
    assert "do not prove the loaded extension binary was built from them" in text
    assert payload["non_claims"] == list(v2.NON_CLAIMS)


def test_the_snapshot_validates_the_exact_ten_and_nine_case_sets(
    config: v2.PolicyConfigV2, tmp_path: Path
) -> None:
    payload = json.loads(_frozen(config, tmp_path).read_text())
    assert payload["study"]["remediation_cases"] == list(v2.REMEDIATION_CASES)
    assert len(payload["study"]["remediation_cases"]) == 10
    assert payload["study"]["gate_cases"] == list(v2.REMEDIATION_GATE_CASES)
    assert len(payload["study"]["gate_cases"]) == 9
    assert payload["study"]["gate_eligible_case_count"] == 9
    assert payload["study"]["anchor_cases"] == list(v2.ANCHOR_CASES)


# ---------------------------------------------------------------------------
# The frozen price-reference rule, at snapshot level
# ---------------------------------------------------------------------------


def test_the_snapshot_pins_the_price_reference_method(
    config: v2.PolicyConfigV2, tmp_path: Path
) -> None:
    payload = json.loads(_frozen(config, tmp_path).read_text())
    assert payload["study"]["price_reference_method"] == "raw_grid_3200x1600_center"
    assert payload["study"]["name"] == v2.STUDY_NAME
    text = " ".join(payload["limitations"])
    assert "never Richardson-extrapolated" in text
    assert "selectively Richardson-extrapolated" not in text


@pytest.mark.parametrize("key", ["name", "price_reference_method"])
def test_a_mutated_snapshot_protocol_string_fails_check(
    config: v2.PolicyConfigV2, tmp_path: Path, key: str
) -> None:
    snapshot = _frozen(config, tmp_path)
    payload = json.loads(snapshot.read_text())
    payload["study"][key] = "something_else"
    snapshot.write_text(freeze.serialise(payload), encoding="utf-8")
    assert freeze.main(["--check", "--output", str(snapshot)]) == 2


@pytest.mark.parametrize(
    "key", ["name", "delta_method", "vega_method", "price_reference_method"]
)
def test_a_mutated_report_protocol_string_is_refused(
    config: v2.PolicyConfigV2, tmp_path: Path, key: str
) -> None:
    report = _report(config)
    if key == "name":
        report["study"]["name"] = "other-study"
    else:
        report["conventions"][key] = "other_method"
    path = _write(tmp_path / "report.json", report)
    with pytest.raises(freeze.FreezeError):
        freeze.snapshot_from_report(path, config)


# ---------------------------------------------------------------------------
# Nested schemas, through the freeze path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "section",
    [
        "study",
        "conventions",
        "validation_rules",
        "eligibility_contract",
        "richardson_contract",
        "solver",
        "performance",
    ],
)
@pytest.mark.parametrize("kind", ["unknown", "removed"])
def test_freezing_rejects_a_nested_schema_mutation(
    config: v2.PolicyConfigV2, tmp_path: Path, section: str, kind: str
) -> None:
    report = _report(config)
    if kind == "unknown":
        report[section]["injected_field"] = 1
    else:
        report[section].pop(sorted(report[section])[0])
    path = _write(tmp_path / "report.json", report)
    with pytest.raises(freeze.FreezeError):
        freeze.snapshot_from_report(path, config)
    assert (
        freeze.main(["--extract", "--report", str(path), "--output", str(tmp_path / "s.json")])
        == 2
    )
    assert not (tmp_path / "s.json").exists()


def test_freezing_rejects_a_mutated_grid_record(
    config: v2.PolicyConfigV2, tmp_path: Path
) -> None:
    report = _report(config)
    report["grids"][0]["injected_field"] = 1
    path = _write(tmp_path / "report.json", report)
    with pytest.raises(freeze.FreezeError):
        freeze.snapshot_from_report(path, config)


# ---------------------------------------------------------------------------
# Every snapshot aggregate is recomputed from the case rows
# ---------------------------------------------------------------------------


def _mutated_value(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return 999 if value != 999 else 0
    if isinstance(value, float):
        return 999.0 if value != 999.0 else 0.0
    if isinstance(value, str):
        return f"{value}_injected"
    if isinstance(value, list):
        return [*value, "injected"]
    if isinstance(value, dict):
        return {**value, "injected": 1}
    return "injected"


@pytest.mark.parametrize("block", ["outcome", "solve_accounting"])
def test_every_snapshot_aggregate_is_recomputed(
    config: v2.PolicyConfigV2, tmp_path: Path, block: str
) -> None:
    baseline = json.loads(_frozen(config, tmp_path).read_text())
    accepted: list[str] = []
    for key in sorted(baseline[block]):
        payload = copy.deepcopy(baseline)
        payload[block][key] = _mutated_value(payload[block][key])
        target = tmp_path / f"mutated_{block}_{key}.json"
        target.write_text(freeze.serialise(payload), encoding="utf-8")
        if freeze.main(["--check", "--output", str(target)]) == 0:
            accepted.append(key)
    assert accepted == [], f"snapshot {block} aggregates accepted after mutation: {accepted}"


@pytest.mark.parametrize(
    ("block", "key", "value"),
    [
        ("solve_accounting", "planned_surface_solves", 999),
        ("solve_accounting", "psor_total_iterations", 999),
        ("outcome", "descriptive_failure_count", 999),
        (
            "outcome",
            "failed_checks_by_name",
            {"price_absolute_error": ["regular_euro_atm_put"]},
        ),
    ],
)
def test_the_named_snapshot_aggregate_mutations_are_rejected(
    config: v2.PolicyConfigV2, tmp_path: Path, block: str, key: str, value: Any
) -> None:
    snapshot = _frozen(config, tmp_path)
    payload = json.loads(snapshot.read_text())
    payload[block][key] = value
    snapshot.write_text(freeze.serialise(payload), encoding="utf-8")
    assert freeze.main(["--check", "--output", str(snapshot)]) == 2


@pytest.mark.parametrize(
    "key", ["linear_solves", "psor_solves", "psor_total_iterations"]
)
def test_a_mutated_per_case_engine_total_breaks_the_accounting(
    config: v2.PolicyConfigV2, tmp_path: Path, key: str
) -> None:
    snapshot = _frozen(config, tmp_path)
    payload = json.loads(snapshot.read_text())
    payload["cases"][0][key] = int(payload["cases"][0][key]) + 1
    snapshot.write_text(freeze.serialise(payload), encoding="utf-8")
    assert freeze.main(["--check", "--output", str(snapshot)]) == 2


def test_the_snapshot_accounting_is_derivable_from_its_own_rows(
    config: v2.PolicyConfigV2, tmp_path: Path
) -> None:
    payload = json.loads(_frozen(config, tmp_path).read_text())
    accounting = payload["solve_accounting"]
    assert accounting["planned_surface_solves"] == 116
    assert accounting["attempted_surface_solves"] == 116
    exceptions = sum(len(case["solve_failures"]) for case in payload["cases"])
    assert accounting["solver_exceptions"] == exceptions == 1
    assert accounting["completed_surface_solves"] == 116 - exceptions
    assert accounting["backward_inductions"] == 116 - exceptions
    for key in ("linear_solves", "psor_solves", "psor_total_iterations"):
        assert accounting[key] == sum(int(case[key]) for case in payload["cases"])


# ---------------------------------------------------------------------------
# Provenance for the newly tracked option translation unit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["option_header", "option_implementation"])
def test_a_mutated_option_digest_fails_extraction_and_check(
    config: v2.PolicyConfigV2, tmp_path: Path, name: str
) -> None:
    report = _report(config)
    report["source"][f"{name}_sha256"] = "0" * 64
    path = _write(tmp_path / "report.json", report)
    with pytest.raises(freeze.FreezeError, match=f"{name}_sha256 does not match"):
        freeze.snapshot_from_report(path, config)

    snapshot = _frozen(config, tmp_path)
    payload = json.loads(snapshot.read_text())
    payload["source"][f"{name}_sha256"] = "0" * 64
    snapshot.write_text(freeze.serialise(payload), encoding="utf-8")
    assert freeze.main(["--check", "--output", str(snapshot)]) == 2


def test_the_snapshot_inventory_includes_the_option_translation_unit(
    config: v2.PolicyConfigV2, tmp_path: Path
) -> None:
    payload = json.loads(_frozen(config, tmp_path).read_text())
    inventory = payload["source"]["executable_source_inventory"]
    assert inventory["option_header"] == "cpp/include/dp/option.hpp"
    assert inventory["option_implementation"] == "cpp/src/option.cpp"
    for name in ("option_header", "option_implementation"):
        assert payload["source"][f"{name}_sha256"] == v2.executable_source_digests()[
            f"{name}_sha256"
        ]


# ---------------------------------------------------------------------------
# Strict JSON on the freeze path
# ---------------------------------------------------------------------------


def _inject_duplicate(text: str, anchor: str, line: str) -> str:
    start = text.index(anchor)
    brace = text.index("{", start)
    return f"{text[: brace + 1]}\n{line}{text[brace + 1 :]}"


@pytest.mark.parametrize(
    ("anchor", "line", "expected_path"),
    [
        pytest.param("{", '  "schema_version": "duplicated",', "$.schema_version",
                     id="duplicate_top_level_report_key"),
        pytest.param('"study": ', '    "name": "duplicated",', ".study.name",
                     id="duplicate_nested_study_name"),
        pytest.param('"validation_rules": ', '    "gating_checks": [],',
                     ".validation_rules.gating_checks", id="duplicate_in_validation_rules"),
        pytest.param('"solves": [', '        "role": "duplicated",',
                     ".cases[0].solves[0].role", id="duplicate_inside_an_array_object"),
    ],
)
def test_extraction_refuses_a_duplicate_key_before_writing_anything(
    config: v2.PolicyConfigV2, tmp_path: Path, anchor: str, line: str, expected_path: str
) -> None:
    text = v2.canonical_json(_report(config))
    report = tmp_path / "report.json"
    report.write_text(_inject_duplicate(text, anchor, line), encoding="utf-8")
    snapshot = tmp_path / "snapshot.json"

    with pytest.raises(freeze.FreezeError, match="repeats object keys") as raised:
        freeze.snapshot_from_report(report, config)
    assert expected_path in str(raised.value)
    assert freeze.main(["--extract", "--report", str(report), "--output", str(snapshot)]) == 2
    assert not snapshot.exists()


def test_check_refuses_a_duplicate_key_in_the_snapshot(
    config: v2.PolicyConfigV2, tmp_path: Path
) -> None:
    snapshot = _frozen(config, tmp_path)
    text = snapshot.read_text()
    snapshot.write_text(
        _inject_duplicate(text, '"study": ', '    "candidate": "duplicated",'),
        encoding="utf-8",
    )
    assert freeze.main(["--check", "--output", str(snapshot)]) == 2
    with pytest.raises(freeze.FreezeError, match="repeats object keys") as raised:
        freeze.load_json(snapshot, "snapshot")
    assert ".study.candidate" in str(raised.value)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_the_freeze_path_refuses_non_standard_constants(
    config: v2.PolicyConfigV2, tmp_path: Path, constant: str
) -> None:
    text = v2.canonical_json(_report(config))
    report = tmp_path / "report.json"
    report.write_text(text.replace('"wall_seconds": 0.0', f'"wall_seconds": {constant}'), "utf-8")
    snapshot = tmp_path / "snapshot.json"
    with pytest.raises(freeze.FreezeError, match="non-standard JSON constant"):
        freeze.snapshot_from_report(report, config)
    assert freeze.main(["--extract", "--report", str(report), "--output", str(snapshot)]) == 2
    assert not snapshot.exists()

    good = _frozen(config, tmp_path)
    good.write_text(
        good.read_text().replace('"case_count": 10', f'"case_count": {constant}'), "utf-8"
    )
    assert freeze.main(["--check", "--output", str(good)]) == 2


# ---------------------------------------------------------------------------
# Exhaustive recursive type validation of the snapshot
# ---------------------------------------------------------------------------


def _incompatible(value: Any) -> Any:
    if type(value) is bool:
        return 1
    if type(value) is int:
        return True
    if type(value) is float:
        return "not a number"
    if type(value) is str:
        return 7
    if type(value) in (list, dict):
        return 7
    return {"injected_incompatible": True}


def _nodes(document: Any, path: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    found = [path]
    if isinstance(document, dict):
        for key in document:
            found.extend(_nodes(document[key], (*path, key)))
    elif isinstance(document, list):
        for index, item in enumerate(document):
            found.extend(_nodes(item, (*path, index)))
    return found


def _distinct_positions(document: Any) -> dict[str, tuple[Any, ...]]:
    positions: dict[str, tuple[Any, ...]] = {}
    for path in _nodes(document):
        if not path:
            continue
        name = "".join("[]" if isinstance(x, int) else f".{x}" for x in path)
        positions.setdefault(name, path)
    return positions


def _read_at(document: Any, path: tuple[Any, ...]) -> Any:
    target = document
    for part in path:
        target = target[part]
    return target


def _set_at(document: Any, path: tuple[Any, ...], value: Any) -> None:
    target = document
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value


def test_every_serialized_snapshot_position_is_type_validated(
    config: v2.PolicyConfigV2, tmp_path: Path
) -> None:
    baseline = json.loads(_frozen(config, tmp_path).read_text())
    v2.validate_against_schema(freeze.SNAPSHOT_SCHEMA, baseline, where="snapshot")
    positions = _distinct_positions(baseline)
    assert len(positions) > 60, len(positions)
    accepted: list[str] = []
    for name, path in sorted(positions.items()):
        document = copy.deepcopy(baseline)
        _set_at(document, path, _incompatible(_read_at(baseline, path)))
        try:
            v2.validate_against_schema(freeze.SNAPSHOT_SCHEMA, document, where="snapshot")
        except v2.LabelPolicyV2Error:
            continue
        accepted.append(name)
    assert accepted == [], f"snapshot positions accepted an incompatible type: {accepted}"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        pytest.param(("cases", 0, "gate_eligible"), 1, id="gate_eligible_true_to_1"),
        pytest.param(("cases", 0, "gamma_label_eligible"), 1, id="gamma_flag_to_1"),
        pytest.param(("cases", 0, "anchor_evidence_complete"), 1, id="anchor_flag_to_1"),
        pytest.param(("cases", 0, "fixed_bump_delta_branch"), 7, id="delta_branch_null_to_7"),
        pytest.param(("cases", 0, "solve_failures", 0, "exception_class"), 7,
                     id="exception_class_to_7"),
        pytest.param(("cases", 0, "delta_ineligibility_reasons"), 7, id="reasons_to_7"),
        pytest.param(("cases", 0, "psor_total_iterations"), True, id="iterations_to_true"),
        pytest.param(("solve_accounting", "planned_surface_solves"), True,
                     id="planned_to_true"),
    ],
)
def test_the_named_adversarial_snapshot_type_mutations_are_rejected(
    config: v2.PolicyConfigV2, tmp_path: Path, path: tuple[Any, ...], value: Any
) -> None:
    snapshot = _frozen(config, tmp_path)
    payload = json.loads(snapshot.read_text())
    _set_at(payload, path, value)
    snapshot.write_text(freeze.serialise(payload), encoding="utf-8")
    assert freeze.main(["--check", "--output", str(snapshot)]) == 2
    with pytest.raises(freeze.FreezeError, match="expected"):
        freeze.validate_snapshot(payload, config)


def test_snapshot_type_errors_name_the_path_and_the_observation(
    config: v2.PolicyConfigV2, tmp_path: Path
) -> None:
    payload = json.loads(_frozen(config, tmp_path).read_text())
    _set_at(payload, ("cases", 0, "gate_eligible"), 1)
    with pytest.raises(freeze.FreezeError) as raised:
        freeze.validate_snapshot(payload, config)
    message = str(raised.value)
    assert "snapshot.cases[0].gate_eligible" in message
    assert "expected boolean" in message
    assert "observed int 1" in message


# ---------------------------------------------------------------------------
# Snapshot digest and identity fields
# ---------------------------------------------------------------------------

_DIGEST_LIKE_SUFFIXES = ("_sha256", "_digest")


def _digest_spec_of(spec: Any) -> Any | None:
    inner = spec.inner if isinstance(spec, v2.NullableSpec) else spec
    if isinstance(inner, v2.StrSpec) and (inner.hex_digest or inner.digest_prefix):
        return inner
    return None


def _schema_digest_fields(schema: Any) -> dict[str, tuple[Any, bool]]:
    found: dict[str, tuple[Any, bool]] = {}
    for path, spec in v2.iter_schema_specs(schema):
        field = path.rsplit(".", 1)[-1].removesuffix("[]").removesuffix("{}")
        if not field.endswith(_DIGEST_LIKE_SUFFIXES):
            continue
        if isinstance(spec, (v2.ObjectSpec, v2.ArraySpec, v2.MapSpec)):
            continue
        found.setdefault(path, (spec, isinstance(spec, v2.NullableSpec)))
    return found


def test_every_digest_like_snapshot_field_is_typed_as_a_digest() -> None:
    plain = [
        path
        for path, (spec, _) in _schema_digest_fields(freeze.SNAPSHOT_SCHEMA).items()
        if _digest_spec_of(spec) is None
    ]
    assert plain == [], f"digest-like snapshot fields typed as plain strings: {plain}"
    fields = _schema_digest_fields(freeze.SNAPSHOT_SCHEMA)
    for name in (
        "reported_binding_pde_header_sha256",
        "reported_binding_pde_source_sha256",
        "reported_binding_pde_implementation_sha256",
    ):
        assert _digest_spec_of(fields[f".source.{name}"][0]).hex_digest is True, name
    assert len(fields) >= 18, len(fields)


def _bad_digests(spec: Any) -> list[tuple[str, Any]]:
    prefix = f"{spec.digest_prefix}-" if spec.digest_prefix else ""
    cases = [
        ("not_a_digest", "not-a-digest"),
        ("sixty_three_hex", prefix + "0" * 63),
        ("sixty_five_hex", prefix + "0" * 65),
        ("uppercase_hex", prefix + "0" * 63 + "A"),
        ("non_hex_characters", prefix + "0" * 63 + "z"),
        ("integer", 7),
        ("null", None),
    ]
    if spec.digest_prefix:
        cases.append(("wrong_prefix", "other-" + "0" * 64))
    return cases


def _snapshot_digest_params() -> list[Any]:
    params: list[Any] = []
    for path, (spec, _) in sorted(_schema_digest_fields(freeze.SNAPSHOT_SCHEMA).items()):
        inner = _digest_spec_of(spec)
        assert inner is not None, path
        params.extend(
            pytest.param(path, value, id=f"{path}-{label}")
            for label, value in _bad_digests(inner)
        )
    return params


def _positions_for(document: Any, normalised: str) -> list[tuple[Any, ...]]:
    return [
        path
        for path in _nodes(document)
        if path
        and "".join("[]" if isinstance(x, int) else f".{x}" for x in path) == normalised
    ]


@pytest.mark.parametrize(("path", "value"), _snapshot_digest_params())
def test_every_snapshot_digest_field_rejects_a_malformed_value(
    config: v2.PolicyConfigV2, tmp_path: Path, path: str, value: Any
) -> None:
    snapshot = _frozen(config, tmp_path)
    payload = json.loads(snapshot.read_text())
    concrete = _positions_for(payload, path)
    assert concrete, path
    _set_at(payload, concrete[0], value)
    with pytest.raises(freeze.FreezeError, match="expected"):
        freeze.validate_snapshot(payload, config)
    snapshot.write_text(freeze.serialise(payload), encoding="utf-8")
    assert freeze.main(["--check", "--output", str(snapshot)]) == 2


# ---------------------------------------------------------------------------
# Snapshot MapSpec key-closure audit
# ---------------------------------------------------------------------------


def _map_specs(spec: Any) -> list[tuple[str, Any]]:
    return [
        (path, node)
        for path, node in v2.iter_schema_specs(spec)
        if isinstance(node, v2.MapSpec)
    ]


def test_every_snapshot_map_is_key_constrained_or_contextually_validated() -> None:
    unbounded = [
        path
        for path, node in _map_specs(freeze.SNAPSHOT_SCHEMA)
        if not node.has_key_constraint and path not in freeze.CONTEXTUAL_MAP_PATHS
    ]
    assert unbounded == [], f"snapshot MapSpecs with arbitrary keys: {unbounded}"
    present = {path for path, _ in _map_specs(freeze.SNAPSHOT_SCHEMA)}
    assert present >= freeze.CONTEXTUAL_MAP_PATHS


def test_the_snapshot_map_classification_is_recorded() -> None:
    classification = {
        ".outcome.failed_checks_by_name": "enum-keyed subset",
        ".solve_accounting.attempted_by_case_classification": "enum-keyed subset",
        ".solve_accounting.attempted_by_criticality": "enum-keyed subset",
        ".solve_accounting.attempted_by_solve_role": "context-derived",
    }
    assert {path for path, _ in _map_specs(freeze.SNAPSHOT_SCHEMA)} == set(classification)


def test_the_snapshot_role_map_key_set_is_exact(
    config: v2.PolicyConfigV2, tmp_path: Path
) -> None:
    snapshot = _frozen(config, tmp_path)
    payload = json.loads(snapshot.read_text())
    payload["solve_accounting"]["attempted_by_solve_role"]["grid_1600x800/invented"] = 1
    with pytest.raises(freeze.FreezeError, match="key set is wrong"):
        freeze.validate_snapshot(payload, config)
    payload = json.loads(snapshot.read_text())
    payload["solve_accounting"]["attempted_by_solve_role"].pop("grid_800x400/base")
    with pytest.raises(freeze.FreezeError, match="key set is wrong"):
        freeze.validate_snapshot(payload, config)
    payload = json.loads(snapshot.read_text())
    payload["solve_accounting"]["attempted_by_solve_role"]["not a role key"] = 1
    with pytest.raises(freeze.FreezeError, match="key matching"):
        freeze.validate_snapshot(payload, config)


# ---------------------------------------------------------------------------
# Systematic object-key closure over every terminal snapshot variant
# ---------------------------------------------------------------------------


def _object_positions(document: Any) -> dict[str, tuple[Any, ...]]:
    positions: dict[str, tuple[Any, ...]] = {}

    def discriminator(container: str, record: Any) -> str:
        if isinstance(record, dict):
            if container == "cases":
                shape = (
                    record["classification"],
                    record["gate_eligible"],
                    record["passes"],
                    bool(record["solve_failures"]),
                )
                return f"[{'/'.join(map(str, shape))}]"
            if container == "solve_failures":
                return f"[{record['criticality']}]"
        return "[]"

    def walk(value: Any, path: tuple[Any, ...], key: str, container: str) -> None:
        if isinstance(value, dict):
            positions.setdefault(key, path)
            for field in value:
                walk(value[field], (*path, field), f"{key}.{field}", field)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, (*path, index), key + discriminator(container, item), container)

    walk(document, (), "", "")
    return positions


def _snapshot_variants(config: v2.PolicyConfigV2, tmp_path: Path) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for name, kwargs in (
        ("remediation_failed", {"stage": "remediation", "fail": True}),
        ("confirmation_passed", {"stage": "confirmation", "fail": False}),
        ("confirmation_failed", {"stage": "confirmation", "fail": True}),
    ):
        directory = tmp_path / name
        directory.mkdir()
        report = _write(directory / "report.json", _report(config, **kwargs))
        snapshot = directory / "snapshot.json"
        assert (
            freeze.main(["--extract", "--report", str(report), "--output", str(snapshot)]) == 0
        )
        variants[name] = json.loads(snapshot.read_text())
    return variants


def _snapshot_rejected(payload: Any, config: v2.PolicyConfigV2) -> bool:
    try:
        v2.validate_against_schema(freeze.SNAPSHOT_SCHEMA, payload, where="snapshot")
        freeze.validate_snapshot_context(payload, config)
    except (v2.LabelPolicyV2Error, FreezeErrorAlias):
        return True
    try:
        freeze.validate_snapshot(payload, config)
    except FreezeErrorAlias:
        return True
    return False


FreezeErrorAlias = freeze.FreezeError


@pytest.mark.parametrize(
    "variant", ["remediation_failed", "confirmation_passed", "confirmation_failed"]
)
def test_object_key_closure_for_every_terminal_snapshot_variant(
    config: v2.PolicyConfigV2, tmp_path: Path, variant: str
) -> None:
    baseline = _snapshot_variants(config, tmp_path)[variant]
    freeze.validate_snapshot(baseline, config)
    positions = _object_positions(baseline)
    assert len(positions) > 10, (variant, len(positions))
    survivors: list[str] = []
    for name, path in sorted(positions.items()):
        target = _read_at(baseline, path)
        target["unexpected_serialized_key"] = 1
        if not _snapshot_rejected(baseline, config):
            survivors.append(f"{name}: added key")
        del target["unexpected_serialized_key"]
        for key in sorted(target):
            removed = target.pop(key)
            if not _snapshot_rejected(baseline, config):
                survivors.append(f"{name}: removed {key}")
            target[key] = removed
    assert survivors == [], f"{variant} accepted key mutations: {survivors}"
