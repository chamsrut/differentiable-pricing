"""Preservation tests for the frozen task 9C-C3 (label-policy v2) snapshot.

`docs/results/american_pde_label_policy_v2_results_v1.json` is the terminal
evidence of task 9C-C3: the confirmation stage ran once, against criteria fixed
before execution, and selected `grid_1600x800` for price. This module pins that
file so a later change cannot quietly reinterpret, regenerate, or loosen it.

Three properties keep this module CI-cheap and honest:

* it reads only checked-in files — never `artifacts/`, never the ignored raw
  confirmation report, never a compiled extension;
* it re-runs the designated `--check` validator in-process, so the semantic
  recomputation the freeze tool performs is exercised here too, not merely the
  byte digest;
* it never edits data to satisfy an assertion. Every expected value below was
  read out of the reviewed snapshot, and the snapshot's own SHA-256 is pinned
  first, so a mutation fails here before any derived assertion is reached.

**What 18/22 delta eligibility means.** It is *conservative observed-order
measurability*, not evidence that four deltas are inaccurate. Each of the four
excluded regular cases records `grid_stencil_delta_validation_passed` — the
stencil delta was inside its cap — and was excluded solely by
`grid_stencil_observed_order_unsupported`. Two of them have an exactly flat
stencil delta (absolute error 0.0; the raw report records the stencil delta as
exactly -1 and exactly +1), which leaves the factor-two order undefined; the
other two measured an order outside the predeclared `[1.5, 2.5]` band while
their errors stayed far below the 1e-3 delta cap. The band is not widened here
and eligibility is not changed: any reconsideration requires a separately
versioned task.
"""

from __future__ import annotations

import importlib.util
import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import pytest

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
SNAPSHOT: Final = (
    PROJECT_ROOT / "docs/results/american_pde_label_policy_v2_results_v1.json"
)
SNAPSHOT_V1: Final = PROJECT_ROOT / "docs/results/american_pde_label_policy_results_v1.json"
CONFIG_V2: Final = PROJECT_ROOT / "configs/pde_label_policy_pilot_v2.toml"
CONFIG_V1: Final = PROJECT_ROOT / "configs/pde_label_policy_pilot_v1.toml"
FREEZE_SCRIPT: Final = PROJECT_ROOT / "scripts/freeze_pde_label_policy_v2_results.py"

# The reviewed, externally approved snapshot and the raw confirmation report it
# was distilled from. Neither may ever be regenerated to make a test pass.
EXPECTED_SNAPSHOT_SHA256: Final = (
    "75d9402f071323065f8398ccd2cf427e2fb6fa1e663aef90ec7cbcf9c46a1186"
)
EXPECTED_REPORT_SHA256: Final = (
    "f9bf3f8fd636498b09fae20df8e42e976c68d2b70b28fdff20ab93752a7e130e"
)
EXPECTED_RAW_CONFIG_SHA256: Final = (
    "01b310a48036d5c5215ca2b10fd37bbcb210b69e5766b1ac34ca65af0097e138"
)
EXPECTED_CRITERIA_DIGEST: Final = (
    "crit-6d7918d8ba4cd961b388047bcb8abb70bebb10d86ce12b3aec22c0d7357ebe38"
)
EXPECTED_CASE_DESIGN_DIGEST: Final = (
    "cases-c2d4ca8210cebd2c951c87052b22f519daf5d14bd1875e56802a9d04fcc3d91d"
)
EXPECTED_SOURCE_COMPOSITE_DIGEST: Final = (
    "src-8b2585e31491cc093b7577b3637c3d0ebf4749d9b5f15c73be89697e14f8aa58"
)

# Task 9C-B's frozen negative result. It coexists with the v2 snapshot and is
# never reinterpreted by it.
EXPECTED_V1_SNAPSHOT_SHA256: Final = (
    "7871e52e52149171c16fb665f2114ccf300ac4e33babee2242d99e35488736c6"
)

SELECTED_POLICY: Final = "grid_1600x800"

# The four v1 absolute-error caps, carried into v2 unchanged. v2's question was
# whether a revised *stability and shape* rule can pass; holding the accuracy
# caps fixed is what makes the two studies comparable.
EXPECTED_CRITERIA: Final = {
    "delta_absolute_error": 0.001,
    "delta_flat_epsilon_rule": "small=3E/(2h), large=3E/(4h), E=M_base*R_base",
    "fixed_bump_charge_scheme": "one_over_one_minus_two_to_minus_three_halves/1",
    "gamma_absolute_error": 0.0002,
    "gamma_is_evaluation_only": True,
    "grid_stencil_bump_bias_charge_applied": False,
    "grid_stencil_order_rule": "observed_order_from_order_probe_candidate_and_reference",
    "grid_stencil_reference_rule": "grid_richardson_of_candidate_and_fine_stencil_deltas",
    "maximum_supported_bump_order": 2.5,
    "maximum_supported_stencil_order": 2.5,
    "minimum_supported_bump_order": 1.5,
    "minimum_supported_stencil_order": 1.5,
    "price_absolute_error": 0.0005,
    "shape_absolute_floor": 1e-08,
    "vega_absolute_error_per_unit_volatility": 0.05,
    "vega_flat_epsilon_rule": (
        "small=err(h)+err(2h), large=err(2h)+err(4h), "
        "err(b)=(M_down_b*R_down_b+M_up_b*R_up_b)/(2b)"
    ),
}

# Conservative observed-order measurability, case by case. The value is the
# observed factor-two stencil order; `None` means the stencil delta was exactly
# flat, so no order is defined.
DELTA_INELIGIBLE_REGULAR_CASES: Final = {
    "regular_american_put_high_rate": None,
    "regular_american_high_carry_call": None,
    "regular_euro_deep_otm_call_short_low_vol": 23.74653099571015,
    "regular_american_one_dividend_call": 3.0274095602490747,
}

# Every one of the four was excluded on order alone, with its stencil-delta
# validation recorded as passed.
DELTA_ORDER_ONLY_EXCLUSION_REASONS: Final = [
    "grid_stencil_delta_validation_passed",
    "grid_stencil_observed_order_unsupported",
]

WORST_REGULAR_PRICE_ABSOLUTE_ERROR: Final = 2.686655623431733e-04
DESCRIPTIVE_STRESS_FAILURE: Final = "stress_euro_short_low_vol_atm"
DESCRIPTIVE_STRESS_PRICE_ABSOLUTE_ERROR: Final = 2.9942556865550918e-03


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def snapshot() -> dict[str, Any]:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def regular_cases(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    return [case for case in snapshot["cases"] if case["classification"] == "regular"]


@pytest.fixture(scope="module")
def freeze_tool() -> Any:
    specification = importlib.util.spec_from_file_location("freeze_v2", FREEZE_SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Identity of the frozen file
# ---------------------------------------------------------------------------


def test_snapshot_digest_is_frozen() -> None:
    """The reviewed, externally approved bytes. Never regenerate to fix a test."""
    assert _digest(SNAPSHOT) == EXPECTED_SNAPSHOT_SHA256


def test_snapshot_and_report_schema_versions_are_pinned(snapshot: dict[str, Any]) -> None:
    assert snapshot["schema_version"] == "american-pde-label-policy-v2-results/1"
    assert snapshot["source"]["report_schema_version"] == "pde-label-policy-v2-report/1"
    assert snapshot["source"]["report_filename"] == "report.json"
    assert snapshot["source"]["config_file"] == "pde_label_policy_pilot_v2.toml"


def test_snapshot_records_the_reviewed_confirmation_report_digest(
    snapshot: dict[str, Any],
) -> None:
    """The raw report is git-ignored; only its content digest is preserved here."""
    assert snapshot["source"]["report_sha256"] == EXPECTED_REPORT_SHA256


def test_snapshot_pins_the_versioned_configuration_and_criteria_digests(
    snapshot: dict[str, Any],
) -> None:
    source = snapshot["source"]
    lifecycle = snapshot["lifecycle"]

    assert source["raw_config_sha256"] == EXPECTED_RAW_CONFIG_SHA256
    assert source["raw_config_sha256"] == _digest(CONFIG_V2)
    assert lifecycle["raw_config_sha256"] == source["raw_config_sha256"]

    assert snapshot["criteria_digest"] == EXPECTED_CRITERIA_DIGEST
    assert lifecycle["criteria_digest"] == EXPECTED_CRITERIA_DIGEST
    assert lifecycle["case_design_digest"] == EXPECTED_CASE_DESIGN_DIGEST


def test_predeclared_criteria_are_exactly_the_frozen_set(snapshot: dict[str, Any]) -> None:
    """Including the four v1 accuracy caps, carried over unchanged."""
    assert snapshot["predeclared_criteria"] == EXPECTED_CRITERIA


# ---------------------------------------------------------------------------
# Terminal lifecycle and selection
# ---------------------------------------------------------------------------


def test_lifecycle_is_the_terminal_confirmation_state(snapshot: dict[str, Any]) -> None:
    """Lifecycle state 3 of the three supported states: terminal and freezable."""
    lifecycle = snapshot["lifecycle"]

    assert lifecycle["stage"] == "confirmation"
    assert lifecycle["remediation_status"] == "passed"
    assert lifecycle["confirmation_status"] == "run"
    assert lifecycle["confirmation_not_run_reason"] is None
    assert lifecycle["terminal"] is True
    assert lifecycle["freezable"] is True
    assert lifecycle["protocol"] == "one_shot_predeclared"
    assert lifecycle["one_shot_enforcement"] == "procedural_and_provenance_backed"
    assert snapshot["study"]["stage"] == "confirmation"


def test_selected_accuracy_policy_is_the_single_candidate(snapshot: dict[str, Any]) -> None:
    assert snapshot["lifecycle"]["selected_accuracy_policy"] == SELECTED_POLICY
    assert snapshot["outcome"]["candidate"] == SELECTED_POLICY
    assert snapshot["study"]["candidate"] == SELECTED_POLICY
    assert snapshot["outcome"]["selection_order"] == [SELECTED_POLICY]


def test_criteria_were_not_loosened(snapshot: dict[str, Any]) -> None:
    assert snapshot["lifecycle"]["criteria_were_not_loosened"] is True
    assert snapshot["outcome"]["criteria_were_not_loosened"] is True


def test_pending_approval_field_is_historical_and_stays_true(
    snapshot: dict[str, Any],
) -> None:
    """The report predates the external approval, so `true` is correct and frozen.

    Completion of the fresh top-level approval is recorded in the decision log,
    not by editing this file.
    """
    assert snapshot["lifecycle"]["selection_pending_fresh_top_level_approval"] is True


def test_snapshot_authorizes_no_dataset_and_no_training_input(
    snapshot: dict[str, Any],
) -> None:
    lifecycle = snapshot["lifecycle"]

    assert lifecycle["authorizes_dataset_generation"] is False
    assert lifecycle["authorizes_training_input"] is False
    assert any(
        "No dataset and no training input is authorized by task 9C-C3" in claim
        for claim in snapshot["non_claims"]
    )


# ---------------------------------------------------------------------------
# Regular-only selection and the exact eligibility conclusions
# ---------------------------------------------------------------------------


def test_only_regular_cases_decide_selection(
    snapshot: dict[str, Any], regular_cases: list[dict[str, Any]]
) -> None:
    outcome = snapshot["outcome"]
    stress_cases = [case for case in snapshot["cases"] if case["classification"] == "stress"]

    assert outcome["regular_cases_only_decide_selection"] is True
    assert outcome["stress_case_is_descriptive_anchor_evidence"] is True
    assert snapshot["study"]["case_count"] == 28
    assert snapshot["study"]["confirmation_case_count"] == 28
    assert len(regular_cases) == 22
    assert len(stress_cases) == 6
    # No stress case is gate-eligible, so none can enter the price gate.
    assert not any(case["gate_eligible"] for case in stress_cases)


def test_price_policy_is_twenty_two_of_twenty_two_regular_cases(
    snapshot: dict[str, Any], regular_cases: list[dict[str, Any]]
) -> None:
    outcome = snapshot["outcome"]

    assert outcome["gate_eligible_case_count"] == 22
    assert outcome["gate_eligible_failure_count"] == 0
    assert outcome["gate_eligible_failures"] == []
    assert outcome["all_gate_eligible_cases_pass"] is True
    assert sum(case["gate_eligible"] for case in regular_cases) == 22
    assert sum(case["passes"] for case in regular_cases) == 22
    # Numerically valid: every regular case returned a result, none failed a solve.
    assert not any(case["solve_failures"] for case in regular_cases)
    assert not any(case["failed_checks"] for case in regular_cases)


def test_worst_regular_price_error_sits_under_the_unchanged_cap(
    snapshot: dict[str, Any], regular_cases: list[dict[str, Any]]
) -> None:
    worst = max(case["price_absolute_error"] for case in regular_cases)

    assert worst == pytest.approx(WORST_REGULAR_PRICE_ABSOLUTE_ERROR, rel=1e-12)
    assert worst < snapshot["predeclared_criteria"]["price_absolute_error"] == 5e-4


def test_delta_supervision_is_eligible_on_eighteen_of_twenty_two_regular_cases(
    regular_cases: list[dict[str, Any]],
) -> None:
    eligible = [case for case in regular_cases if case["delta_label_eligible"]]

    assert len(eligible) == 18
    assert len(regular_cases) - len(eligible) == 4


def test_the_four_delta_exclusions_are_order_measurability_not_inaccuracy(
    snapshot: dict[str, Any], regular_cases: list[dict[str, Any]]
) -> None:
    """18/22 is conservative observed-order measurability, not four bad deltas."""
    delta_cap = snapshot["predeclared_criteria"]["delta_absolute_error"]
    minimum = snapshot["predeclared_criteria"]["minimum_supported_stencil_order"]
    maximum = snapshot["predeclared_criteria"]["maximum_supported_stencil_order"]
    excluded = {
        case["name"]: case for case in regular_cases if not case["delta_label_eligible"]
    }

    assert set(excluded) == set(DELTA_INELIGIBLE_REGULAR_CASES)

    for name, expected_order in DELTA_INELIGIBLE_REGULAR_CASES.items():
        case = excluded[name]
        # Excluded on order alone; the stencil-delta validation itself passed.
        assert case["delta_ineligibility_reasons"] == DELTA_ORDER_ONLY_EXCLUSION_REASONS
        # And the delta error is far inside its unchanged cap in every one.
        assert case["grid_stencil_delta_absolute_error"] < delta_cap
        observed = case["grid_stencil_observed_order"]
        if expected_order is None:
            # An exactly flat stencil delta leaves the factor-two order undefined.
            assert observed is None
            assert case["grid_stencil_delta_absolute_error"] == 0.0
        else:
            assert observed == pytest.approx(expected_order, rel=1e-12)
            assert not minimum <= observed <= maximum

    # The two measured orders, stated explicitly so a silent drift is caught.
    assert excluded["regular_euro_deep_otm_call_short_low_vol"][
        "grid_stencil_observed_order"
    ] == pytest.approx(23.74653099571015, rel=1e-12)
    one_dividend = excluded["regular_american_one_dividend_call"]
    assert one_dividend["grid_stencil_observed_order"] == pytest.approx(
        3.0274095602490747, rel=1e-12
    )
    assert one_dividend["grid_stencil_delta_absolute_error"] == pytest.approx(
        9.711173257764827e-07, rel=1e-12
    )
    assert one_dividend["grid_stencil_delta_absolute_error"] < 1e-3


def test_vega_supervision_is_eligible_on_all_twenty_two_regular_cases(
    regular_cases: list[dict[str, Any]],
) -> None:
    assert sum(case["vega_label_eligible"] for case in regular_cases) == 22
    assert not any(case["vega_ineligibility_reasons"] for case in regular_cases)


def test_gamma_is_evaluation_only_and_never_supervision_eligible(
    snapshot: dict[str, Any], regular_cases: list[dict[str, Any]]
) -> None:
    assert snapshot["predeclared_criteria"]["gamma_is_evaluation_only"] is True
    assert snapshot["outcome"]["gamma_eligible_cases"] == []
    assert sum(case["gamma_label_eligible"] for case in regular_cases) == 0
    assert not any(case["gamma_label_eligible"] for case in snapshot["cases"])


def test_published_eligible_case_lists_match_the_per_case_records(
    snapshot: dict[str, Any], regular_cases: list[dict[str, Any]]
) -> None:
    """The published lists span all 28 cases; the 18/22 and 22/22 are the regular part."""
    outcome = snapshot["outcome"]
    regular_names = {case["name"] for case in regular_cases}

    assert set(outcome["delta_eligible_cases"]) == {
        case["name"] for case in snapshot["cases"] if case["delta_label_eligible"]
    }
    assert set(outcome["vega_eligible_cases"]) == {
        case["name"] for case in snapshot["cases"] if case["vega_label_eligible"]
    }
    assert len(set(outcome["delta_eligible_cases"]) & regular_names) == 18
    assert len(set(outcome["vega_eligible_cases"]) & regular_names) == 22


# ---------------------------------------------------------------------------
# Descriptive stress evidence and solve accounting
# ---------------------------------------------------------------------------


def test_the_single_descriptive_failure_is_a_stress_case_and_decides_nothing(
    snapshot: dict[str, Any],
) -> None:
    outcome = snapshot["outcome"]
    failure = next(
        case for case in snapshot["cases"] if case["name"] == DESCRIPTIVE_STRESS_FAILURE
    )

    assert outcome["descriptive_failure_count"] == 1
    assert outcome["descriptive_failures"] == [DESCRIPTIVE_STRESS_FAILURE]
    assert failure["classification"] == "stress"
    assert failure["gate_eligible"] is False
    assert failure["passes"] is False
    assert failure["failed_checks"] == ["price_absolute_error"]
    assert failure["price_absolute_error"] == pytest.approx(
        DESCRIPTIVE_STRESS_PRICE_ABSOLUTE_ERROR, rel=1e-12
    )
    # Its gamma error is published and, being evaluation-only, gates nothing.
    assert failure["gamma_label_eligible"] is False
    assert outcome["anchor_evidence_incomplete_cases"] == []


def test_solve_accounting_is_complete_and_exception_free(snapshot: dict[str, Any]) -> None:
    accounting = snapshot["solve_accounting"]

    assert accounting["planned_surface_solves"] == 323
    assert accounting["attempted_surface_solves"] == 323
    assert accounting["completed_surface_solves"] == 323
    assert accounting["backward_inductions"] == 323
    assert accounting["solver_exceptions"] == 0
    assert accounting["linear_solves"] == 176130
    assert accounting["psor_solves"] == 144086
    assert accounting["psor_total_iterations"] == 34888292
    assert accounting["reconciliation"] == {
        "attempted_equals_completed_plus_exceptions": True,
        "backward_inductions_equal_completed_solves": True,
        "planned_equals_attempted": True,
    }


def test_the_residual_scale_allowance_is_published_as_non_rigorous(
    snapshot: dict[str, Any],
) -> None:
    """The negative-rate dominance gap passes through an operational allowance."""
    control = next(
        case
        for case in snapshot["cases"]
        if case["name"] == "regular_american_put_negative_rate_control"
    )

    assert control["classification"] == "regular"
    assert control["gate_eligible"] is True
    assert control["passes"] is True
    assert control["failed_checks"] == []
    assert any(
        "not certified bounds" in claim and "operational price-error scale" in claim
        for claim in snapshot["non_claims"]
    )
    assert any(
        "not certified bounds" in limitation for limitation in snapshot["limitations"]
    )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_executable_source_digests_reconcile_against_current_repository_files(
    snapshot: dict[str, Any],
) -> None:
    """The v2 snapshot contract requires a source checkout that still matches.

    Unlike the v1 snapshot — whose PDE digests are deliberately historical — the
    v2 freeze tool reconciles its whole inventory against the repository, so this
    test reconciles all of it too.
    """
    source = snapshot["source"]
    inventory = source["executable_source_inventory"]

    assert set(inventory) == {
        "canonical_payload_module",
        "option_header",
        "option_implementation",
        "package_init_module",
        "pde_binding_source",
        "pde_header",
        "pde_implementation",
        "v1_label_policy_module",
        "v2_freeze_script",
        "v2_runner_module",
    }
    for name, relative in inventory.items():
        path = PROJECT_ROOT / relative
        assert path.is_file(), f"inventory entry '{name}' is missing from the repository"
        assert source[f"{name}_sha256"] == _digest(path), (
            f"inventory entry '{name}' ({relative}) no longer matches the digest the "
            f"confirmation stage recorded"
        )

    assert source["executable_source_composite_digest"] == EXPECTED_SOURCE_COMPOSITE_DIGEST
    assert source["private_market_data_used"] is False
    assert source["data_provenance"] == "fixed_synthetic_public_configuration_only"
    assert "do not prove the loaded _pde extension binary" in source[
        "source_digest_limitation"
    ]


def test_designated_check_mode_validates_the_checked_in_snapshot(freeze_tool: Any) -> None:
    """Runs the same validator `scripts/check.sh` and CI run, in-process.

    It recomputes every verdict, count and lifecycle field from the snapshot's
    own numbers against the checked-in configuration, and reads no `artifacts/`.
    """
    assert freeze_tool.main(["--check"]) == 0


# ---------------------------------------------------------------------------
# Coexistence with the immutable v1 evidence
# ---------------------------------------------------------------------------


def test_v1_and_v2_snapshots_coexist_and_v1_is_untouched() -> None:
    """v2 answers a new, separately predeclared question. It never reinterprets v1."""
    assert SNAPSHOT_V1.is_file()
    assert SNAPSHOT.is_file()
    assert SNAPSHOT_V1 != SNAPSHOT
    assert CONFIG_V1.is_file() and CONFIG_V2.is_file()
    assert _digest(SNAPSHOT_V1) == EXPECTED_V1_SNAPSHOT_SHA256
    assert _digest(SNAPSHOT) == EXPECTED_SNAPSHOT_SHA256

    v1 = json.loads(SNAPSHOT_V1.read_text(encoding="utf-8"))
    assert v1["schema_version"] == "pde-label-policy-report/1"
    assert v1["recommendation"]["selected_accuracy_policy"] == "no_policy_selected"
