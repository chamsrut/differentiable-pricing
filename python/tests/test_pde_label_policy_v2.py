"""Task 9C-C3 label-policy v2 contract tests.

Every test here is cheap. The predeclared criteria are pure functions of
already-measured numbers, so the whole v2 decision surface is exercised with
tiny synthetic solve records, a throwing fake solver, and closed-form
Black-Scholes ladders whose expected values are written out from the
predeclared formulas rather than obtained from the functions under test.

**No PDE solve, no remediation run and no confirmation run happens here.**
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from differentiable_pricing import black_scholes
from differentiable_pricing.american import pde_label_policy_v2 as v2
from differentiable_pricing.american.pde_label_policy_v2 import (
    ANCHOR_GRID,
    CANDIDATE_GRID,
    ORDER_PROBE_GRID,
    REFERENCE_GRID,
    CaseV2,
    LabelPolicyV2Error,
    PolicyConfigV2,
    SolveFailure,
    SurfaceSolve,
    build_report,
    canonical_json,
    collect_case_solves,
    conventions_block,
    criteria_digest,
    delta_flat_epsilons,
    delta_residual_scale,
    evaluate_case,
    evaluate_fixed_bump_validation,
    evaluate_grid_stencil_delta,
    expected_pricing_grid_identity,
    lifecycle_block,
    load_policy_config_v2,
    parse_policy_config_v2,
    planned_solve_count,
    pricing_grid_identity,
    require_confirmation_entry,
    richardson_contract_block,
    select_accuracy_policy,
    shape_allowance,
    solve_accounting,
    stage_outcome,
    vega_flat_epsilons,
    vega_residual_scale,
    verify_stage_report,
    write_outputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_V2 = PROJECT_ROOT / "configs" / "pde_label_policy_pilot_v2.toml"
CONFIG_V1 = PROJECT_ROOT / "configs" / "pde_label_policy_pilot_v1.toml"
RESULT_V1 = PROJECT_ROOT / "docs" / "results" / "american_pde_label_policy_results_v1.json"

AMERICAN_CASE = "regular_american_put_early_exercise"
EUROPEAN_ANCHOR_CASE = "regular_euro_atm_call"
STRESS_CASE = v2.NON_GATING_REMEDIATION_CASE


@pytest.fixture(scope="module")
def config() -> PolicyConfigV2:
    return load_policy_config_v2(CONFIG_V2)


def _document() -> dict[str, Any]:
    with CONFIG_V2.open("rb") as stream:
        return tomllib.load(stream)


def _parse(document: dict[str, Any]) -> PolicyConfigV2:
    return parse_policy_config_v2(document, source_name="test.toml", raw_config_sha256="0" * 64)


# ---------------------------------------------------------------------------
# Finding 7: the whole predeclaration is locked in the parser
# ---------------------------------------------------------------------------


def test_checked_in_v2_design_is_valid_and_versioned(config: PolicyConfigV2) -> None:
    assert config.name == "pde-label-policy-pilot-v2"
    assert len(config.cases) == v2.CONFIRMATION_CASE_COUNT == 28
    assert config.selection_order == (CANDIDATE_GRID,)
    assert config.order_probe_grids == (ORDER_PROBE_GRID,)
    assert config.remediation_cases == v2.REMEDIATION_CASES
    assert config.gate_cases == v2.REMEDIATION_GATE_CASES
    assert len(config.gate_cases) == 9
    assert config.anchor_cases == v2.ANCHOR_CASES
    assert config.case_design_digest == v2.CASE_DESIGN_DIGEST


def test_v2_case_states_are_the_unchanged_v1_case_states() -> None:
    with CONFIG_V1.open("rb") as stream:
        v1_document = tomllib.load(stream)
    assert _document()["cases"] == v1_document["cases"]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        pytest.param(
            lambda d: d["cases"].append({**copy.deepcopy(d["cases"][0]), "name": "extra"}),
            "exactly 28 cases",
            id="extra_case",
        ),
        pytest.param(
            lambda d: d["cases"].pop(3), "exactly 28 cases", id="removed_case"
        ),
        pytest.param(
            lambda d: d["cases"].insert(0, d["cases"].pop(5)),
            "order changed",
            id="reordered_cases",
        ),
        pytest.param(
            lambda d: d["cases"][0].__setitem__("volatility", 0.21),
            "order changed",
            id="edited_case_state",
        ),
        pytest.param(
            lambda d: d["study"].__setitem__(
                "anchor_cases", ["regular_euro_atm_put", STRESS_CASE]
            ),
            "anchor_cases must be exactly",
            id="replaced_anchor",
        ),
        pytest.param(
            lambda d: d["study"].__setitem__(
                "remediation_cases",
                [
                    n if n != "regular_euro_high_rate_call" else "regular_american_put_high_rate"
                    for n in d["study"]["remediation_cases"]
                ],
            ),
            "remediation_cases must be exactly",
            id="swapped_v1_failure",
        ),
        pytest.param(
            lambda d: d["study"].__setitem__(
                "gate_cases", d["study"]["gate_cases"][:8]
            ),
            "gate_cases must be exactly",
            id="eight_gate_cases",
        ),
        pytest.param(
            lambda d: d["study"].__setitem__(
                "v1_regular_failures", d["study"]["v1_regular_failures"][:4]
            ),
            "v1_regular_failures must be exactly",
            id="four_v1_failures",
        ),
        pytest.param(
            lambda d: d["study"].__setitem__(
                "smooth_control_cases", ["regular_euro_atm_put"]
            ),
            "smooth_control_cases must be exactly",
            id="one_control",
        ),
        pytest.param(
            lambda d: d["study"].__setitem__("case_design_digest", "cases-" + "0" * 64),
            "case_design_digest",
            id="unpinned_case_digest",
        ),
        pytest.param(
            lambda d: d["study"].__setitem__(
                "selection_order", [ORDER_PROBE_GRID, CANDIDATE_GRID]
            ),
            "selection_order must be exactly",
            id="second_candidate",
        ),
        pytest.param(
            lambda d: d["study"].__setitem__("order_probe_grids", [REFERENCE_GRID]),
            "order_probe_grids must be exactly",
            id="probe_becomes_reference",
        ),
        pytest.param(
            lambda d: d["grids"][1].__setitem__("role", "reference"),
            "grid roles must be",
            id="candidate_role_changed",
        ),
        pytest.param(
            lambda d: d["grids"].pop(3), "exactly 4 tables", id="anchor_rung_dropped"
        ),
        pytest.param(
            lambda d: d["criteria"].__setitem__("minimum_supported_bump_order", 1.2),
            "must stay 1.5",
            id="changed_order_lower_bound",
        ),
        pytest.param(
            lambda d: d["criteria"].__setitem__("maximum_supported_stencil_order", 3.0),
            "must stay 2.5",
            id="changed_stencil_order_upper_bound",
        ),
        pytest.param(
            lambda d: d["richardson"].__setitem__("minimum_supported_observed_order", 1.2),
            "observed-order band must stay",
            id="changed_richardson_band",
        ),
        pytest.param(
            lambda d: d["criteria"].__setitem__("shape_absolute_floor", 1.0e-6),
            "must stay 1e-08",
            id="changed_shape_floor",
        ),
        pytest.param(
            lambda d: d["criteria"].__setitem__(
                "fixed_bump_charge_scheme", "one_over_one_minus_two_to_minus_one/1"
            ),
            "fixed_bump_charge_scheme",
            id="changed_k_construction",
        ),
        pytest.param(
            lambda d: d["criteria"].__setitem__(
                "delta_flat_epsilon_rule", "small=E/h, large=E/(2h)"
            ),
            "delta_flat_epsilon_rule",
            id="changed_delta_epsilon_rule",
        ),
        pytest.param(
            lambda d: d["criteria"].__setitem__(
                "vega_flat_epsilon_rule", "small=err(h), large=err(2h)"
            ),
            "vega_flat_epsilon_rule",
            id="changed_vega_epsilon_rule",
        ),
        pytest.param(
            lambda d: d["criteria"].__setitem__(
                "grid_stencil_bump_bias_charge_applied", True
            ),
            "grid_stencil_bump_bias_charge_applied",
            id="e2_gains_a_bias_charge",
        ),
        pytest.param(
            lambda d: d["criteria"].__setitem__("gamma_is_evaluation_only", False),
            "gamma_is_evaluation_only",
            id="gamma_becomes_gating",
        ),
        pytest.param(
            lambda d: d["eligibility"].__setitem__("gamma_eligibility", "sometimes"),
            "gamma_eligibility must be 'never'",
            id="gamma_becomes_eligible",
        ),
        pytest.param(
            lambda d: d["eligibility"].__setitem__("structural_c1_applies_to", ["delta"]),
            "structural_c1_applies_to",
            id="c1_drops_vega",
        ),
        pytest.param(
            lambda d: d["eligibility"].__setitem__(
                "identity_excludes_contract_multiplier", False
            ),
            "identity_excludes_contract_multiplier",
            id="multiplier_enters_identity",
        ),
        pytest.param(
            lambda d: d["eligibility"].__setitem__("identity_includes_settlement", False),
            "identity_includes_settlement",
            id="settlement_leaves_identity",
        ),
        pytest.param(
            lambda d: d["dominance"].__setitem__("control", "european_control_on_every_node"),
            "european_dominance_control_at_centre_only",
            id="dominance_no_longer_d2",
        ),
        pytest.param(
            lambda d: d["dominance"].__setitem__(
                "residual_measure", "relative_maximum_lcp_residual"
            ),
            "absolute_maximum_lcp_residual",
            id="relative_residual",
        ),
        pytest.param(
            lambda d: d["dominance"].__setitem__("is_a_rigorous_bound", True),
            "is_a_rigorous_bound must be false",
            id="claims_a_bound",
        ),
        pytest.param(
            lambda d: d["richardson"].__setitem__("role", "candidate"),
            "reference_validation_only",
            id="richardson_promoted",
        ),
        pytest.param(
            lambda d: d["solver"].__setitem__("psor_maximum_iterations", 50),
            "must stay 200000",
            id="changed_psor_ceiling",
        ),
        pytest.param(
            lambda d: d["solver"].__setitem__("psor_tolerance", 1.0e-9),
            "must stay 1e-11",
            id="changed_residual_rule",
        ),
        pytest.param(
            lambda d: d["bumps"].__setitem__("spot", [0.25, 0.5, 1.0]),
            "must stay",
            id="changed_bump_ladder",
        ),
        pytest.param(
            lambda d: d["lifecycle"].__setitem__("authorizes_dataset_generation", True),
            "authorizes_dataset_generation must be false",
            id="authorizes_a_dataset",
        ),
        pytest.param(
            lambda d: d["lifecycle"].__setitem__(
                "selection_requires_fresh_top_level_approval", False
            ),
            "selection_requires_fresh_top_level_approval",
            id="self_approves",
        ),
        pytest.param(
            lambda d: d["lifecycle"].__setitem__(
                "confirmation_statuses", ["not_run", "pending", "run", "rerun"]
            ),
            "confirmation_statuses must be exactly",
            id="extra_lifecycle_state",
        ),
        pytest.param(
            lambda d: d["gating"].__setitem__(
                "gating_checks", [*v2.GATING_CHECKS, "fixed_bump_delta_validation"]
            ),
            "gating_checks must be exactly",
            id="delta_validation_gates",
        ),
        pytest.param(
            lambda d: d["gating"].__setitem__(
                "price_critical_solves", [*v2.PRICE_CRITICAL_SOLVES, "grid_800x400/base"]
            ),
            "price_critical_solves must be exactly",
            id="probe_becomes_price_critical",
        ),
        pytest.param(
            lambda d: d["gating"].__setitem__("anchor_rung_failure_is_descriptive", False),
            "anchor_rung_failure_is_descriptive",
            id="anchor_starts_gating",
        ),
        pytest.param(
            lambda d: d["gating"].__setitem__("stress_case_failure_is_descriptive", False),
            "stress_case_failure_is_descriptive",
            id="stress_starts_gating",
        ),
    ],
)
def test_the_parser_rejects_every_material_mutation(mutate, match: str) -> None:
    document = _document()
    mutate(document)
    with pytest.raises(LabelPolicyV2Error, match=match):
        _parse(document)


@pytest.mark.parametrize("key", sorted(v2.UNCHANGED_V1_CAPS))
def test_a_loosened_v1_cap_is_rejected(key: str) -> None:
    document = _document()
    document["criteria"][key] *= 10.0
    with pytest.raises(LabelPolicyV2Error, match="unchanged v1 cap"):
        _parse(document)


def test_the_four_v1_caps_equal_the_frozen_v1_snapshot(config: PolicyConfigV2) -> None:
    frozen_v1 = json.loads(RESULT_V1.read_text())["predeclared_criteria"]
    for key in v2.UNCHANGED_V1_CAPS:
        assert getattr(config.criteria, key) == frozen_v1[key]


def test_every_configured_centre_and_bumped_spot_is_an_exact_node() -> None:
    document = _document()
    document["cases"][0]["spot"] = 100.3
    with pytest.raises(LabelPolicyV2Error, match=r"order changed|not an exact interior node"):
        _parse(document)


def test_an_unknown_configuration_key_is_rejected() -> None:
    document = _document()
    document["criteria"]["extra_knob"] = 1.0
    with pytest.raises(LabelPolicyV2Error, match="unknown keys"):
        _parse(document)


# ---------------------------------------------------------------------------
# Finding 8: executable-source provenance
# ---------------------------------------------------------------------------


def test_the_executable_source_inventory_covers_every_relied_on_source() -> None:
    assert set(v2.EXECUTABLE_SOURCE_INVENTORY) == {
        "v2_runner_module",
        "v2_freeze_script",
        "v1_label_policy_module",
        "canonical_payload_module",
        "package_init_module",
        "pde_header",
        "pde_implementation",
        "pde_binding_source",
        "option_header",
        "option_implementation",
    }
    for relative in v2.EXECUTABLE_SOURCE_INVENTORY.values():
        assert (PROJECT_ROOT / relative).is_file(), relative
    # Every first-party file the PDE path's include graph reaches is inventoried.
    assert set(v2.PDE_INCLUDE_CLOSURE) <= set(v2.EXECUTABLE_SOURCE_INVENTORY.values())


def test_source_digests_and_their_composite_reconcile() -> None:
    digests = v2.executable_source_digests()
    assert set(digests) == {
        f"{name}_sha256" for name in v2.EXECUTABLE_SOURCE_INVENTORY
    }
    composite = v2.executable_source_composite_digest(digests)
    recorded = {**digests, "executable_source_composite_digest": composite}
    assert v2.verify_executable_source_digests(recorded, where="test") == digests
    mutated = dict(recorded)
    mutated["pde_header_sha256"] = "0" * 64
    with pytest.raises(LabelPolicyV2Error, match="pde_header_sha256 does not match"):
        v2.verify_executable_source_digests(mutated, where="test")


def test_the_source_digest_limitation_is_stated(config: PolicyConfigV2) -> None:
    assert "do not prove the loaded _pde extension binary" in v2.SOURCE_DIGEST_LIMITATION
    provenance = v2._source_provenance(config)
    assert provenance["source_digest_limitation"] == v2.SOURCE_DIGEST_LIMITATION
    # Build-time digests are recorded separately and are self-reported.
    assert "reported_binding_pde_header_sha256" in provenance


def test_a_missing_inventory_source_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(LabelPolicyV2Error, match="must run from a source checkout"):
        v2.executable_source_digests(root=tmp_path)


# ---------------------------------------------------------------------------
# Finding 1: corrected E1 flat-branch residual scales
# ---------------------------------------------------------------------------


def test_charge_factor_is_the_declared_constant() -> None:
    assert v2.FIXED_BUMP_CHARGE_FACTOR == 1.0 / (1.0 - 2.0**-1.5)
    assert math.isclose(v2.FIXED_BUMP_CHARGE_FACTOR, 1.5469181606780271, rel_tol=0.0)


def test_delta_flat_epsilons_are_three_e_over_two_h_and_four_h() -> None:
    scale, bump = 8.0e-10, 0.5
    small, large = delta_flat_epsilons(base_price_residual_scale=scale, primary_bump=bump)
    assert small == pytest.approx(3.0 * scale / (2.0 * bump), rel=0.0, abs=0.0)
    assert large == pytest.approx(3.0 * scale / (4.0 * bump), rel=0.0, abs=0.0)
    # error(D(h)) + error(D(2h)) and error(D(2h)) + error(D(4h)), written out.
    assert small == pytest.approx(scale / bump + scale / (2.0 * bump))
    assert large == pytest.approx(scale / (2.0 * bump) + scale / (4.0 * bump))
    assert small == 2.0 * large


def test_vega_flat_epsilons_sum_adjacent_rung_error_scales() -> None:
    errors = [1.6e-7, 8.0e-8, 4.0e-8]
    small, large = vega_flat_epsilons(errors)
    assert small == errors[0] + errors[1]
    assert large == errors[1] + errors[2]
    with pytest.raises(LabelPolicyV2Error, match="three rung error scales"):
        vega_flat_epsilons([1.0, 2.0])


def _fixed_bump(**overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "quantity": "delta",
        "estimate_h": 0.5,
        "estimate_2h": 0.5 + 4.0e-5,
        "estimate_4h": 0.5 + 2.0e-4,
        "reference_estimate": 0.5,
        "epsilon_small": 2.4e-9,
        "epsilon_large": 1.2e-9,
        "cap": 1.0e-3,
        "minimum_order": 1.5,
        "maximum_order": 2.5,
    }
    arguments.update(overrides)
    return evaluate_fixed_bump_validation(**arguments)


def test_resolved_branch_requires_a_supported_observed_order() -> None:
    outcome = _fixed_bump()
    assert outcome["branch"] == "resolved"
    assert outcome["observed_bump_order"] == pytest.approx(2.0)
    assert outcome["verdict"] == "pass"
    assert outcome["bias_charge"] == pytest.approx(v2.FIXED_BUMP_CHARGE_FACTOR * 4.0e-5)


def test_resolved_branch_fails_an_unsupported_observed_order() -> None:
    outcome = _fixed_bump(estimate_4h=0.5 + 8.0e-5)
    assert outcome["branch"] == "resolved"
    assert outcome["checks"]["observed_bump_order_supported"] is False
    assert outcome["verdict"] == "fail"


@pytest.mark.parametrize(
    ("h", "two_h", "four_h"),
    [
        pytest.param(0.5, 0.5, 0.5 + 1.0e-4, id="zero_small_increment"),
        pytest.param(0.5, 0.5 + 1.0e-4, 0.5 + 1.0e-4, id="zero_large_increment"),
    ],
)
def test_a_zero_denominator_resolved_order_can_never_pass(
    h: float, two_h: float, four_h: float
) -> None:
    outcome = _fixed_bump(
        estimate_h=h, estimate_2h=two_h, estimate_4h=four_h, epsilon_small=1e-12,
        epsilon_large=5e-13,
    )
    assert outcome["branch"] == "resolved"
    assert outcome["observed_bump_order"] is None
    assert outcome["checks"]["observed_bump_order_supported"] is False
    assert outcome["verdict"] == "fail"


def test_a_non_finite_estimate_can_never_pass() -> None:
    outcome = _fixed_bump(estimate_4h=math.nan)
    assert outcome["checks"]["observed_bump_order_supported"] is False
    assert outcome["verdict"] == "fail"


def test_flat_branch_uses_each_increment_against_its_own_tolerance() -> None:
    # Small increment inside epsilon_small but large increment outside
    # epsilon_large: not flat, because the tolerances are per increment.
    outcome = _fixed_bump(
        estimate_h=0.5,
        estimate_2h=0.5 + 1.0e-9,
        estimate_4h=0.5 + 1.0e-9 + 4.0e-9,
        epsilon_small=2.4e-9,
        epsilon_large=1.2e-9,
    )
    assert outcome["branch"] == "resolved"
    flat = _fixed_bump(
        estimate_h=0.5,
        estimate_2h=0.5 + 1.0e-9,
        estimate_4h=0.5 + 1.0e-9 + 1.0e-9,
        epsilon_small=2.4e-9,
        epsilon_large=1.2e-9,
    )
    assert flat["branch"] == "flat"
    assert flat["bias_charge"] == pytest.approx(v2.FIXED_BUMP_CHARGE_FACTOR * 2.4e-9)
    assert flat["verdict"] == "pass"


def test_combined_budget_rather_than_the_reference_error_alone_decides() -> None:
    outcome = _fixed_bump(
        estimate_2h=0.5 + 5.0e-4,
        estimate_4h=0.5 + 2.5e-3,
        reference_estimate=0.5 + 3.0e-4,
    )
    assert outcome["reference_absolute_error"] < outcome["absolute_error_cap"]
    assert outcome["combined_absolute_error"] > outcome["absolute_error_cap"]
    assert outcome["verdict"] == "fail"


def test_a_residual_scale_charge_above_the_cap_is_numerically_invalid() -> None:
    outcome = _fixed_bump(
        estimate_2h=0.5, estimate_4h=0.5, epsilon_small=1.0e-2, epsilon_large=5.0e-3
    )
    assert outcome["numerically_invalid_residual_scale"] is True
    assert "delta_numerically_invalid_residual_scale" in outcome["reasons"]
    assert outcome["verdict"] == "fail"


def test_fixed_bump_publishes_both_epsilons_and_every_predeclared_field() -> None:
    outcome = _fixed_bump()
    for key in (
        "branch",
        "observed_bump_order",
        "delta_small",
        "delta_large",
        "epsilon_small",
        "epsilon_large",
        "bias_charge",
        "reference_absolute_error",
        "combined_absolute_error",
        "absolute_error_cap",
        "charge_scheme",
        "verdict",
    ):
        assert key in outcome


# ---------------------------------------------------------------------------
# E2
# ---------------------------------------------------------------------------


def _grid_stencil(**overrides: Any) -> dict[str, Any]:
    error = 1.0e-3
    arguments: dict[str, Any] = {
        "coarse_delta": 0.5 + error,
        "candidate_delta": 0.5 + error / 4.0,
        "fine_delta": 0.5 + error / 16.0,
        "cap": 1.0e-3,
        "assumed_order": 2.0,
        "minimum_order": 1.5,
        "maximum_order": 2.5,
    }
    arguments.update(overrides)
    return evaluate_grid_stencil_delta(**arguments)


def test_grid_stencil_reference_is_the_richardson_of_the_candidate_and_fine_pair() -> None:
    outcome = _grid_stencil()
    assert outcome["observed_stencil_order"] == pytest.approx(2.0)
    assert outcome["reference_stencil_delta"] == pytest.approx(0.5)
    assert outcome["absolute_error"] == pytest.approx(1.0e-3 / 4.0)
    assert outcome["bump_bias_charge_applied"] is False
    assert outcome["verdict"] == "pass"


def test_grid_stencil_single_rule_is_the_delta_cap() -> None:
    outcome = _grid_stencil(cap=1.0e-5)
    assert "grid_stencil_absolute_error_exceeds_cap" in outcome["reasons"]
    assert outcome["verdict"] == "fail"


def test_unsupported_stencil_order_is_named() -> None:
    outcome = _grid_stencil(coarse_delta=0.5 + 1.0e-3 / 2.0)
    assert "grid_stencil_observed_order_unsupported" in outcome["reasons"]
    assert outcome["verdict"] == "fail"


def test_a_degenerate_stencil_order_is_unsupported() -> None:
    outcome = _grid_stencil(coarse_delta=0.5 + 1.0e-3 / 4.0)
    assert outcome["observed_stencil_order"] is None
    assert outcome["verdict"] == "fail"


# ---------------------------------------------------------------------------
# Shape allowance and residual scales
# ---------------------------------------------------------------------------


def test_allowance_is_the_larger_of_the_floor_and_the_scale() -> None:
    tight = shape_allowance(scale=1.0e-9, floor=1.0e-8, price_cap=5.0e-4)
    assert (tight.allowance, tight.exceeds_price_cap) == (1.0e-8, False)
    loose = shape_allowance(scale=1.0e-6, floor=1.0e-8, price_cap=5.0e-4)
    assert loose.allowance == 1.0e-6


def test_an_allowance_is_never_widened_past_the_price_cap() -> None:
    outcome = shape_allowance(scale=1.0e-2, floor=1.0e-8, price_cap=5.0e-4)
    assert outcome.exceeds_price_cap is True
    assert outcome.allowance == 5.0e-4


def test_residual_scale_formulas_are_the_predeclared_ones() -> None:
    assert v2.price_residual_scale(time_steps=800, maximum_lcp_residual=-2.0e-12) == (
        800 * 2.0e-12
    )
    assert delta_residual_scale(
        time_steps=800, maximum_lcp_residual=1.0e-12, spot_step=0.25
    ) == pytest.approx(800 * 1.0e-12 / 0.25)
    assert vega_residual_scale(
        down_time_steps=800,
        down_maximum_lcp_residual=1.0e-12,
        up_time_steps=800,
        up_maximum_lcp_residual=3.0e-12,
        vega_bump=0.005,
    ) == pytest.approx((800 * 1.0e-12 + 800 * 3.0e-12) / (2 * 0.005))
    assert v2.dominance_scale(
        american_time_steps=800,
        american_maximum_lcp_residual=1.0e-12,
        european_time_steps=800,
        european_maximum_lcp_residual=2.0e-12,
    ) == pytest.approx(800 * 3.0e-12)


# ---------------------------------------------------------------------------
# Synthetic solve records
# ---------------------------------------------------------------------------

_PRICE = 60.0
_RESIDUAL = 1.0e-12
_STENCIL_ERROR = 1.0e-3
_VEGA_SLOPE = 12.0
_CONVEXITY = 0.005
_SCALE_BY_GRID = {
    ORDER_PROBE_GRID: 1.0,
    CANDIDATE_GRID: 0.25,
    REFERENCE_GRID: 0.0625,
    ANCHOR_GRID: 0.015625,
}


def _true_delta(case: CaseV2) -> float:
    return 0.5 if case.option_type == "call" else -0.5


def _solve(
    case: CaseV2, config: PolicyConfigV2, grid_name: str, role: str, **overrides: Any
) -> SurfaceSolve:
    grid = config.grid(grid_name)
    style, volatility = v2.planned_role_state(case, role)
    price = _PRICE + _VEGA_SLOPE * (volatility - case.volatility)
    if role == v2.DOMINANCE_CONTROL_ROLE:
        price = _PRICE - 1.0
    bumped: dict[str, float] = {}
    if role == v2.BASE_ROLE and grid_name in {CANDIDATE_GRID, REFERENCE_GRID}:
        for bump in config.bumps.spot:
            for direction, sign in (("down", -1.0), ("up", 1.0)):
                bumped[v2.spot_bump_key(bump, direction)] = (
                    _PRICE + sign * _true_delta(case) * bump + _CONVEXITY * bump * bump
                )
    step, intervals = v2.expected_spot_grid(
        strike=case.strike,
        spot_maximum=config.solver.spot_maximum,
        spot_intervals=grid.spot_intervals,
    )
    solve = SurfaceSolve(
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
        centre_delta=_true_delta(case) + _STENCIL_ERROR * _SCALE_BY_GRID[grid_name],
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
    solve = dataclasses.replace(
        solve, pricing_grid_identity=expected_pricing_grid_identity(case, config, solve)
    )
    if overrides:
        solve = dataclasses.replace(solve, **overrides)
    return solve


def _solves(case: CaseV2, config: PolicyConfigV2) -> dict[tuple[str, str], SurfaceSolve]:
    return {
        key: _solve(case, config, key[0], key[1])
        for key in v2.required_solve_keys(case, config)
    }


def _replace(
    solves: dict[tuple[str, str], SurfaceSolve], key: tuple[str, str], **overrides: Any
) -> dict[tuple[str, str], SurfaceSolve]:
    updated = dict(solves)
    updated[key] = dataclasses.replace(updated[key], **overrides)
    return updated


def _drop(
    case: CaseV2, solves: dict[tuple[str, str], SurfaceSolve], key: tuple[str, str]
) -> tuple[dict[tuple[str, str], SurfaceSolve], list[SolveFailure]]:
    """Remove one solve and describe it as a caught solver exception."""
    remaining = {other: solve for other, solve in solves.items() if other != key}
    failure = SolveFailure(
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


@pytest.fixture()
def american_case(config: PolicyConfigV2) -> CaseV2:
    return config.case(AMERICAN_CASE)


def test_a_clean_case_passes_and_is_delta_and_vega_eligible(
    config: PolicyConfigV2, american_case: CaseV2
) -> None:
    outcome = evaluate_case(american_case, config, _solves(american_case, config))
    assert outcome["passes"] is True
    assert outcome["delta_eligibility"]["eligible"] is True
    assert outcome["vega_eligibility"]["eligible"] is True
    assert outcome["fixed_bump_delta_validation"]["branch"] == "flat"
    assert outcome["fixed_bump_vega_validation"]["branch"] == "flat"
    assert outcome["vega_eligibility"]["value"] == pytest.approx(_VEGA_SLOPE)
    assert outcome["support"] == {
        "price_critical_solves_usable": True,
        "delta_supporting_solves_usable": True,
        "vega_supporting_solves_usable": True,
        "anchor_evidence_complete": True,
    }


# ---------------------------------------------------------------------------
# Finding 2 and 4: the role-aware gate/eligibility truth table
# ---------------------------------------------------------------------------


def _identity(solves, case, config):
    """The baseline truth-table row: nothing is mutated."""
    return solves, []


def _row(case_name: str, mutate, price: bool, delta: bool, vega: bool, anchor: bool):
    return pytest.param(case_name, mutate, price, delta, vega, anchor)


def _mutate_price_error(solves, case, config):
    return _replace(
        solves, (REFERENCE_GRID, v2.BASE_ROLE), centre_price=_PRICE + 1.0e-2
    ), []


def _mutate_shape(solves, case, config):
    control = solves[(CANDIDATE_GRID, v2.DOMINANCE_CONTROL_ROLE)]
    return _replace(
        solves,
        (CANDIDATE_GRID, v2.DOMINANCE_CONTROL_ROLE),
        centre_price=control.centre_price + 1.0 + 1.0e-3,
    ), []


def _mutate_price_scale(solves, case, config):
    return _replace(
        solves, (CANDIDATE_GRID, v2.BASE_ROLE), maximum_lcp_residual=1.0e-5
    ), []


def _mutate_non_exact_node(solves, case, config):
    return _replace(
        solves, (CANDIDATE_GRID, v2.BASE_ROLE), centre_spot_is_exact_node=False
    ), []


def _mutate_identity(solves, case, config):
    return _replace(
        solves, (REFERENCE_GRID, v2.BASE_ROLE), pricing_grid_identity="pg-" + "0" * 64
    ), []


def _mutate_delta_e1(solves, case, config):
    # A resolved, unsupported ladder: the three bump deltas no longer converge.
    solve = solves[(CANDIDATE_GRID, v2.BASE_ROLE)]
    bumped = dict(solve.bumped_prices)
    bumped[v2.spot_bump_key(2.0, "up")] += 4.0e-3
    return _replace(solves, (CANDIDATE_GRID, v2.BASE_ROLE), bumped_prices=bumped), []


def _mutate_delta_e2(solves, case, config):
    return _replace(
        solves,
        (CANDIDATE_GRID, v2.BASE_ROLE),
        centre_delta=_true_delta(case) + 0.5,
    ), []


def _mutate_stencil_order(solves, case, config):
    return _replace(
        solves,
        (ORDER_PROBE_GRID, v2.BASE_ROLE),
        centre_delta=_true_delta(case) + _STENCIL_ERROR * 0.25 + 1.0e-9,
    ), []


def _mutate_delta_scale(solves, case, config):
    return _replace(solves, (CANDIDATE_GRID, v2.BASE_ROLE), spot_step=1.0e-9), []


def _mutate_delta_auxiliary(solves, case, config):
    return _drop(case, solves, (ORDER_PROBE_GRID, v2.BASE_ROLE))


def _mutate_delta_epsilon(solves, case, config):
    # A residual big enough that K * epsilon_delta_small exceeds the delta cap
    # while M*R stays inside the price cap: 1e-7 * 800 = 8e-5 <= 5e-4, and
    # 3 * 8e-5 / (2 * 0.5) * K = 3.7e-4 ... so push to 2e-7.
    return _replace(
        solves, (CANDIDATE_GRID, v2.BASE_ROLE), maximum_lcp_residual=4.0e-7
    ), []


def _mutate_vega_e1(solves, case, config):
    up = solves[(CANDIDATE_GRID, v2.volatility_role(0.02, "up"))]
    return _replace(
        solves,
        (CANDIDATE_GRID, v2.volatility_role(0.02, "up")),
        centre_price=up.centre_price + 1.0,
    ), []


def _mutate_c1(solves, case, config):
    return _replace(
        solves,
        (CANDIDATE_GRID, v2.BASE_ROLE),
        centre_greek_eligible=False,
        centre_greek_eligibility_reason="regime_stencil_not_uniform",
    ), []


def _mutate_vega_availability(solves, case, config):
    return _replace(
        solves, (CANDIDATE_GRID, v2.volatility_role(0.005, "up")), centre_price=math.inf
    ), []


def _mutate_vega_regime(solves, case, config):
    return _replace(
        solves,
        (CANDIDATE_GRID, v2.volatility_role(0.005, "up")),
        centre_exercise_state="exercise",
    ), []


def _mutate_vega_convention(solves, case, config):
    return _replace(
        solves,
        (CANDIDATE_GRID, v2.volatility_role(0.005, "up")),
        volatility=case.volatility + 0.006,
    ), []


def _mutate_vega_auxiliary(solves, case, config):
    return _drop(case, solves, (CANDIDATE_GRID, v2.volatility_role(0.01, "down")))


def _mutate_vega_scale(solves, case, config):
    # 800 * 5e-7 = 4e-4 <= the price cap, while the vega residual scale
    # (2 * 4e-4) / (2 * 0.005) = 0.08 exceeds the 5e-2 vega cap.
    updated = solves
    for direction in ("down", "up"):
        updated = _replace(
            updated,
            (CANDIDATE_GRID, v2.volatility_role(0.005, direction)),
            maximum_lcp_residual=5.0e-7,
        )
    return updated, []


def _mutate_anchor(solves, case, config):
    return _drop(case, solves, (ANCHOR_GRID, v2.BASE_ROLE))


def _mutate_stress_price_critical(solves, case, config):
    return _drop(case, solves, (CANDIDATE_GRID, v2.BASE_ROLE))


TRUTH_TABLE = [
    _row(AMERICAN_CASE, _identity, True, True, True, True),
    _row(
        AMERICAN_CASE,
        lambda s, c, k: _drop(c, s, (CANDIDATE_GRID, v2.BASE_ROLE)),
        False,
        False,
        False,
        True,
    ),
    _row(
        AMERICAN_CASE,
        lambda s, c, k: _drop(c, s, (REFERENCE_GRID, v2.BASE_ROLE)),
        False,
        False,
        False,
        True,
    ),
    _row(
        AMERICAN_CASE,
        lambda s, c, k: _drop(c, s, (CANDIDATE_GRID, v2.DOMINANCE_CONTROL_ROLE)),
        False,
        False,
        False,
        True,
    ),
    _row(AMERICAN_CASE, _mutate_price_error, False, True, True, True),
    _row(AMERICAN_CASE, _mutate_shape, False, True, True, True),
    _row(AMERICAN_CASE, _mutate_non_exact_node, False, False, False, True),
    _row(AMERICAN_CASE, _mutate_identity, False, False, False, True),
    _row(AMERICAN_CASE, _mutate_price_scale, False, False, False, True),
    _row(AMERICAN_CASE, _mutate_delta_e1, True, False, True, True),
    _row(AMERICAN_CASE, _mutate_delta_e2, True, False, True, True),
    _row(AMERICAN_CASE, _mutate_stencil_order, True, False, True, True),
    _row(AMERICAN_CASE, _mutate_delta_scale, True, False, True, True),
    _row(AMERICAN_CASE, _mutate_delta_auxiliary, True, False, True, True),
    _row(AMERICAN_CASE, _mutate_delta_epsilon, True, False, True, True),
    _row(AMERICAN_CASE, _mutate_vega_e1, True, True, False, True),
    _row(AMERICAN_CASE, _mutate_c1, True, False, False, True),
    _row(AMERICAN_CASE, _mutate_vega_availability, True, True, False, True),
    _row(AMERICAN_CASE, _mutate_vega_regime, True, True, False, True),
    _row(AMERICAN_CASE, _mutate_vega_convention, True, True, False, True),
    _row(AMERICAN_CASE, _mutate_vega_auxiliary, True, True, False, True),
    _row(AMERICAN_CASE, _mutate_vega_scale, True, True, False, True),
    _row(EUROPEAN_ANCHOR_CASE, _identity, True, True, True, True),
    _row(EUROPEAN_ANCHOR_CASE, _mutate_anchor, True, True, True, False),
    _row(STRESS_CASE, _mutate_stress_price_critical, False, False, False, True),
]


@pytest.mark.parametrize(
    ("case_name", "mutate", "price_passes", "delta_eligible", "vega_eligible", "anchor_complete"),
    TRUTH_TABLE,
)
def test_gate_and_eligibility_truth_table(
    config: PolicyConfigV2,
    case_name: str,
    mutate,
    price_passes: bool,
    delta_eligible: bool,
    vega_eligible: bool,
    anchor_complete: bool,
) -> None:
    case = config.case(case_name)
    solves, failures = mutate(_solves(case, config), case, config)
    outcome = evaluate_case(case, config, solves, failures)
    assert outcome["passes"] is price_passes
    assert outcome["delta_eligibility"]["eligible"] is delta_eligible
    assert outcome["vega_eligibility"]["eligible"] is vega_eligible
    assert outcome["support"]["anchor_evidence_complete"] is anchor_complete
    # A gamma failure can never appear among the gating checks.
    assert set(outcome["checks"]) == set(v2.GATING_CHECKS)


def test_a_stress_case_failure_never_reaches_the_policy_gate(
    config: PolicyConfigV2,
) -> None:
    case = config.case(STRESS_CASE)
    solves, failures = _drop(case, _solves(case, config), (CANDIDATE_GRID, v2.BASE_ROLE))
    outcome = evaluate_case(case, config, solves, failures)
    assert outcome["passes"] is False
    assert outcome["gate_eligible"] is False
    assert outcome["counts_toward_selection"] is False
    for entry in outcome["solve_failures"]:
        assert entry["criticality"] == v2.STRESS_DESCRIPTIVE
    rows = [
        evaluate_case(config.case(name), config, _solves(config.case(name), config))
        for name in config.remediation_cases
        if name != STRESS_CASE
    ]
    rows.append(outcome)
    summary = stage_outcome(rows, config)
    assert summary["descriptive_failures"] == [STRESS_CASE]
    assert summary["all_gate_eligible_cases_pass"] is True
    assert select_accuracy_policy(summary, config) == CANDIDATE_GRID


def test_an_anchor_failure_is_descriptive_and_recorded(config: PolicyConfigV2) -> None:
    case = config.case(EUROPEAN_ANCHOR_CASE)
    solves, failures = _drop(case, _solves(case, config), (ANCHOR_GRID, v2.BASE_ROLE))
    outcome = evaluate_case(case, config, solves, failures)
    assert outcome["passes"] is True
    assert outcome["delta_eligibility"]["eligible"] is True
    assert outcome["vega_eligibility"]["eligible"] is True
    assert outcome["support"]["anchor_evidence_complete"] is False
    assert outcome["anchor_validation"]["anchor_evidence_complete"] is False
    assert outcome["anchor_validation"]["gates"] is False
    assert outcome["solve_failures"][0]["criticality"] == v2.ANCHOR_DESCRIPTIVE


def test_gamma_never_gates_and_is_never_supervision_eligible(
    config: PolicyConfigV2, american_case: CaseV2
) -> None:
    solves = _replace(
        _solves(american_case, config), (CANDIDATE_GRID, v2.BASE_ROLE), centre_gamma=5.0
    )
    outcome = evaluate_case(american_case, config, solves)
    gamma = outcome["gamma_evaluation_only"]
    assert gamma["absolute_error"] > config.criteria.gamma_absolute_error
    assert gamma["within_cap"] is False
    assert gamma["gates"] is False
    assert gamma["supervision_eligible"] is False
    assert "gamma_absolute_error" not in outcome["checks"]
    assert outcome["passes"] is True


def test_structural_c1_ineligibility_denies_both_greeks_and_not_the_price(
    config: PolicyConfigV2, american_case: CaseV2
) -> None:
    solves, _ = _mutate_c1(_solves(american_case, config), american_case, config)
    outcome = evaluate_case(american_case, config, solves)
    assert outcome["passes"] is True
    assert outcome["delta_eligibility"]["eligible"] is False
    assert outcome["vega_eligibility"]["eligible"] is False
    assert (
        outcome["vega_eligibility"]["predicates"]["structural_c1_greek_eligibility"] is False
    )
    assert "c1_regime_stencil_not_uniform" in outcome["delta_eligibility"]["reasons"]
    assert "c1_regime_stencil_not_uniform" in outcome["vega_eligibility"]["reasons"]


def test_a_dominance_gap_inside_the_residual_scale_no_longer_fails(
    config: PolicyConfigV2, american_case: CaseV2
) -> None:
    """The v1 failure mode: a 2.7e-8 gap against a 1e-8 fixed tolerance."""
    solves = _solves(american_case, config)
    control = solves[(CANDIDATE_GRID, v2.DOMINANCE_CONTROL_ROLE)]
    solves = _replace(
        solves,
        (CANDIDATE_GRID, v2.DOMINANCE_CONTROL_ROLE),
        centre_price=control.centre_price + 1.0 + 2.7e-8,
        maximum_lcp_residual=5.0e-11,
    )
    outcome = evaluate_case(american_case, config, solves)
    dominance = outcome["shape"]["american_dominance"]
    assert dominance["gap"] == pytest.approx(-2.7e-8)
    assert dominance["allowance"] > config.criteria.shape_absolute_floor
    assert dominance["passes"] is True
    assert outcome["shape"]["is_a_rigorous_bound"] is False
    assert outcome["passes"] is True


def test_a_shape_scale_above_the_price_cap_fails_the_check_itself(
    config: PolicyConfigV2, american_case: CaseV2
) -> None:
    # Each solve's own price residual scale stays inside the price cap
    # (800 * 4e-7 = 3.2e-4), but their sum, 6.4e-4, does not.
    solves = _replace(
        _solves(american_case, config),
        (CANDIDATE_GRID, v2.BASE_ROLE),
        maximum_lcp_residual=4.0e-7,
    )
    solves = _replace(
        solves, (CANDIDATE_GRID, v2.DOMINANCE_CONTROL_ROLE), maximum_lcp_residual=4.0e-7
    )
    outcome = evaluate_case(american_case, config, solves)
    assert outcome["checks"]["price_critical_residual_scale_within_price_cap"] is True
    dominance = outcome["shape"]["american_dominance"]
    assert dominance["residual_scale_exceeds_price_cap"] is True
    assert dominance["allowance"] == config.criteria.price_absolute_error
    assert dominance["passes"] is False
    assert outcome["passes"] is False


# ---------------------------------------------------------------------------
# Finding 3: real solver exceptions, per case and solve role
# ---------------------------------------------------------------------------


class _InjectedSolverError(RuntimeError):
    """Stands in for a binding exception such as PsorConvergenceError."""


def _fake_surface(case: CaseV2, config: PolicyConfigV2, kwargs: Mapping[str, Any]) -> dict:
    strike = float(kwargs["strike"])
    intervals_requested = int(kwargs["spot_intervals"])
    volatility = float(kwargs["volatility"])
    step, intervals = v2.expected_spot_grid(
        strike=strike,
        spot_maximum=float(kwargs["spot_maximum"]),
        spot_intervals=intervals_requested,
    )
    grid_name = {800: ORDER_PROBE_GRID, 1600: CANDIDATE_GRID, 3200: REFERENCE_GRID,
                 6400: ANCHOR_GRID}[intervals_requested]
    true_delta = _true_delta(case)
    spots = [index * step for index in range(intervals + 1)]
    values = [
        _PRICE
        + true_delta * (spot - case.spot)
        + _CONVEXITY * (spot - case.spot) ** 2
        + _VEGA_SLOPE * (volatility - case.volatility)
        for spot in spots
    ]
    if kwargs["exercise_style"] == "european" and case.exercise_style == "american":
        values = [value - 1.0 for value in values]
    stencil = true_delta + _STENCIL_ERROR * _SCALE_BY_GRID[grid_name]
    count = len(spots)
    european = kwargs["exercise_style"] == "european"
    return {
        "surface_input": {
            "option_type": kwargs["option_type"],
            "exercise_style": kwargs["exercise_style"],
            "strike": strike,
            "valuation_time": float(kwargs["valuation_time"]),
            "expiry_time": float(kwargs["expiry_time"]),
            "volatility": volatility,
            "continuous_carry": float(kwargs["continuous_carry"]),
            "curve_times": list(kwargs["curve_times"]),
            "curve_log_discounts": list(kwargs["curve_log_discounts"]),
            "dividends": [list(item) for item in kwargs["dividends"]],
            "dividends_declared": True,
            "settlement": kwargs["settlement"],
            "contract_multiplier": float(kwargs["contract_multiplier"]),
            "spot_intervals": intervals_requested,
            "time_steps": int(kwargs["time_steps"]),
            "spot_maximum": float(kwargs["spot_maximum"]),
            "rannacher_steps": int(kwargs["rannacher_steps"]),
            "psor_tolerance": float(kwargs["psor_tolerance"]),
            "psor_relaxation": float(kwargs["psor_relaxation"]),
            "psor_maximum_iterations": int(kwargs["psor_maximum_iterations"]),
            "boundary_exclusion_nodes": int(kwargs["boundary_exclusion_nodes"]),
        },
        "solver_status": "discrete_system_converged",
        "time_steps": int(kwargs["time_steps"]),
        "maximum_lcp_residual": _RESIDUAL,
        "spot_step": step,
        "spot_intervals": intervals,
        "spot_nodes": spots,
        "values": values,
        "deltas": [stencil] * count,
        "gammas": [0.02] * count,
        "greek_eligible": [True] * count,
        "greek_eligibility_reasons": ["eligible"] * count,
        "exercise_states": [("no_obstacle" if european else "continuation")] * count,
        "backward_inductions": 1,
        "linear_solves": int(kwargs["time_steps"]),
        "psor_solves": 0 if european else int(kwargs["time_steps"]),
        "psor_total_iterations": 0 if european else 7 * int(kwargs["time_steps"]),
    }


def _throwing_solver(case: CaseV2, config: PolicyConfigV2, *, throw_at: set[tuple[str, str]]):
    """A fake binding that raises for the named (grid, role) solves."""
    grid_of = {800: ORDER_PROBE_GRID, 1600: CANDIDATE_GRID, 3200: REFERENCE_GRID,
               6400: ANCHOR_GRID}
    calls: list[tuple[str, str]] = []

    def solver(**kwargs: Any) -> dict:
        grid_name = grid_of[int(kwargs["spot_intervals"])]
        volatility = float(kwargs["volatility"])
        if kwargs["exercise_style"] != case.exercise_style:
            role = v2.DOMINANCE_CONTROL_ROLE
        elif volatility == case.volatility:
            role = v2.BASE_ROLE
        else:
            bump = abs(volatility - case.volatility)
            rung = min(config.bumps.volatility, key=lambda b: abs(b - bump))
            role = v2.volatility_role(
                rung, "up" if volatility > case.volatility else "down"
            )
        calls.append((grid_name, role))
        if (grid_name, role) in throw_at:
            raise _InjectedSolverError(
                f"PSOR failed at segment 3 in {PROJECT_ROOT}/cpp/src/x.cpp\nline two"
            )
        return _fake_surface(case, config, kwargs)

    solver.calls = calls  # type: ignore[attr-defined]
    return solver


@pytest.mark.parametrize(
    ("case_name", "throw_at", "criticality", "price", "delta", "vega", "anchor"),
    [
        pytest.param(
            AMERICAN_CASE,
            (CANDIDATE_GRID, v2.BASE_ROLE),
            v2.PRICE_CRITICAL,
            False,
            False,
            False,
            True,
            id="price_critical_exception",
        ),
        pytest.param(
            AMERICAN_CASE,
            (CANDIDATE_GRID, "volatility_0.01_up"),
            v2.VEGA_ONLY,
            True,
            True,
            False,
            True,
            id="vega_auxiliary_exception",
        ),
        pytest.param(
            AMERICAN_CASE,
            (ORDER_PROBE_GRID, v2.BASE_ROLE),
            v2.DELTA_ONLY,
            True,
            False,
            True,
            True,
            id="delta_auxiliary_exception",
        ),
        pytest.param(
            EUROPEAN_ANCHOR_CASE,
            (ANCHOR_GRID, v2.BASE_ROLE),
            v2.ANCHOR_DESCRIPTIVE,
            True,
            True,
            True,
            False,
            id="anchor_exception",
        ),
        pytest.param(
            STRESS_CASE,
            (CANDIDATE_GRID, v2.BASE_ROLE),
            v2.STRESS_DESCRIPTIVE,
            False,
            False,
            False,
            True,
            id="stress_exception",
        ),
    ],
)
def test_a_thrown_solver_exception_is_recorded_and_scoped_to_its_role(
    config: PolicyConfigV2,
    case_name: str,
    throw_at: tuple[str, str],
    criticality: str,
    price: bool,
    delta: bool,
    vega: bool,
    anchor: bool,
) -> None:
    case = config.case(case_name)
    solver = _throwing_solver(case, config, throw_at={throw_at})
    solves, failures, timings = collect_case_solves(case, config, solver=solver)
    # The stage continues: every planned solve was attempted.
    assert len(timings) == len(v2.required_solve_keys(case, config))
    assert len(solves) == len(timings) - 1
    assert len(failures) == 1
    failure = failures[0]
    assert (failure.case, failure.grid, failure.role) == (case.name, *throw_at)
    assert failure.option_type == case.option_type
    assert failure.criticality == criticality
    assert failure.exception_class == "_InjectedSolverError"
    assert "\n" not in failure.message
    assert str(PROJECT_ROOT) not in failure.message
    assert "<repo>" in failure.message

    outcome = evaluate_case(case, config, solves, failures)
    assert outcome["passes"] is price
    assert outcome["delta_eligibility"]["eligible"] is delta
    assert outcome["vega_eligibility"]["eligible"] is vega
    assert outcome["support"]["anchor_evidence_complete"] is anchor


def test_a_clean_fake_solver_reproduces_the_synthetic_baseline(
    config: PolicyConfigV2, american_case: CaseV2
) -> None:
    solver = _throwing_solver(american_case, config, throw_at=set())
    solves, failures, _ = collect_case_solves(american_case, config, solver=solver)
    assert failures == []
    outcome = evaluate_case(american_case, config, solves, failures)
    assert outcome["passes"] is True
    assert outcome["delta_eligibility"]["eligible"] is True
    assert outcome["vega_eligibility"]["eligible"] is True


# ---------------------------------------------------------------------------
# Finding 9: independent analytic Black-Scholes checks
# ---------------------------------------------------------------------------

_BS = {"strike": 100.0, "maturity": 1.0, "rate": 0.05, "dividend_yield": 0.0}


def _bs_price(spot: float, volatility: float) -> float:
    return float(
        black_scholes(option_type="call", spot=spot, volatility=volatility, **_BS)["price"]
    )


def test_analytic_black_scholes_fixed_bump_delta_ladder() -> None:
    """E1 on a closed-form ladder, with every expected value written out here."""
    spot, sigma, h = 100.0, 0.2, 0.5
    slope = 3.0e-4  # a deliberate candidate bias the reference does not carry
    ladder = {}
    for bump in (h, 2.0 * h, 4.0 * h):
        up = _bs_price(spot + bump, sigma) + slope * (spot + bump)
        down = _bs_price(spot - bump, sigma) + slope * (spot - bump)
        ladder[bump] = (up - down) / (2.0 * bump)
    reference = (_bs_price(spot + h, sigma) - _bs_price(spot - h, sigma)) / (2.0 * h)

    delta_small = abs(ladder[h] - ladder[2 * h])
    delta_large = abs(ladder[2 * h] - ladder[4 * h])
    charge = 1.0 / (1.0 - 2.0**-1.5)
    scale = 800 * 1.0e-12
    epsilon_small = 3.0 * scale / (2.0 * h)
    epsilon_large = 3.0 * scale / (4.0 * h)
    assert delta_small > epsilon_small  # a genuinely resolved ladder
    order = math.log2(delta_large / delta_small)
    # Second order up to the ladder's own higher-order Black-Scholes terms.
    assert order == pytest.approx(2.0, abs=5.0e-3)
    expected_reference_error = abs(ladder[h] - reference)
    assert expected_reference_error == pytest.approx(slope, rel=1.0e-9)
    expected_combined = expected_reference_error + charge * delta_small

    outcome = evaluate_fixed_bump_validation(
        quantity="delta",
        estimate_h=ladder[h],
        estimate_2h=ladder[2 * h],
        estimate_4h=ladder[4 * h],
        reference_estimate=reference,
        epsilon_small=epsilon_small,
        epsilon_large=epsilon_large,
        cap=1.0e-3,
        minimum_order=1.5,
        maximum_order=2.5,
    )
    assert outcome["branch"] == "resolved"
    assert outcome["observed_bump_order"] == pytest.approx(order, rel=0.0, abs=0.0)
    assert outcome["delta_small"] == pytest.approx(delta_small, rel=0.0, abs=0.0)
    assert outcome["bias_charge"] == pytest.approx(charge * delta_small, rel=0.0, abs=0.0)
    assert outcome["combined_absolute_error"] == pytest.approx(
        expected_combined, rel=0.0, abs=0.0
    )
    assert outcome["verdict"] == "pass"
    assert expected_combined < 1.0e-3


def test_analytic_black_scholes_fixed_bump_delta_ladder_exceeds_the_budget() -> None:
    spot, sigma, h = 100.0, 0.2, 0.5
    slope = 9.7e-4
    ladder = {}
    for bump in (h, 2.0 * h, 4.0 * h):
        up = _bs_price(spot + bump, sigma) + slope * (spot + bump)
        down = _bs_price(spot - bump, sigma) + slope * (spot - bump)
        ladder[bump] = (up - down) / (2.0 * bump)
    reference = (_bs_price(spot + h, sigma) - _bs_price(spot - h, sigma)) / (2.0 * h)
    scale = 800 * 1.0e-12
    outcome = evaluate_fixed_bump_validation(
        quantity="delta",
        estimate_h=ladder[h],
        estimate_2h=ladder[2 * h],
        estimate_4h=ladder[4 * h],
        reference_estimate=reference,
        epsilon_small=3.0 * scale / (2.0 * h),
        epsilon_large=3.0 * scale / (4.0 * h),
        cap=1.0e-3,
        minimum_order=1.5,
        maximum_order=2.5,
    )
    charge = 1.0 / (1.0 - 2.0**-1.5)
    expected = slope + charge * abs(ladder[h] - ladder[2 * h])
    assert expected > 1.0e-3
    assert outcome["combined_absolute_error"] == pytest.approx(expected, rel=1.0e-12)
    assert outcome["checks"]["combined_absolute_error_within_cap"] is False
    assert outcome["verdict"] == "fail"


def test_analytic_black_scholes_fixed_bump_vega_ladder() -> None:
    spot, sigma, eta = 100.0, 0.2, 0.005
    ladder = {}
    for bump in (eta, 2.0 * eta, 4.0 * eta):
        ladder[bump] = (
            _bs_price(spot, sigma + bump) - _bs_price(spot, sigma - bump)
        ) / (2.0 * bump)
    reference = ladder[eta]

    # err(b) = (M_down_b * R_down_b + M_up_b * R_up_b) / (2b), written out.
    residual, steps = 1.0e-12, 800
    errors = [(steps * residual + steps * residual) / (2.0 * bump) for bump in
              (eta, 2.0 * eta, 4.0 * eta)]
    epsilon_small = errors[0] + errors[1]
    epsilon_large = errors[1] + errors[2]

    delta_small = abs(ladder[eta] - ladder[2 * eta])
    delta_large = abs(ladder[2 * eta] - ladder[4 * eta])
    assert delta_small > epsilon_small
    order = math.log2(delta_large / delta_small)
    assert order == pytest.approx(2.0, abs=2.0e-2)
    charge = 1.0 / (1.0 - 2.0**-1.5)

    outcome = evaluate_fixed_bump_validation(
        quantity="vega",
        estimate_h=ladder[eta],
        estimate_2h=ladder[2 * eta],
        estimate_4h=ladder[4 * eta],
        reference_estimate=reference,
        epsilon_small=epsilon_small,
        epsilon_large=epsilon_large,
        cap=5.0e-2,
        minimum_order=1.5,
        maximum_order=2.5,
    )
    assert outcome["branch"] == "resolved"
    assert outcome["epsilon_small"] == pytest.approx(epsilon_small, rel=0.0, abs=0.0)
    assert outcome["epsilon_large"] == pytest.approx(epsilon_large, rel=0.0, abs=0.0)
    assert outcome["reference_absolute_error"] == 0.0
    assert outcome["combined_absolute_error"] == pytest.approx(
        charge * delta_small, rel=0.0, abs=0.0
    )
    assert outcome["verdict"] == "pass"
    # The ladder's own centered vega is close to the analytic Black-Scholes vega.
    analytic = float(
        black_scholes(option_type="call", spot=spot, volatility=sigma, **_BS)["vega"]
    )
    # Finite-bump truncation, which is exactly what the bias charge covers.
    assert ladder[eta] == pytest.approx(analytic, rel=1.0e-4)


def test_analytic_black_scholes_vega_ladder_flat_branch() -> None:
    """With a residual scale above the ladder's own movement, E1 goes flat."""
    spot, sigma, eta = 100.0, 0.2, 0.005
    ladder = [
        (_bs_price(spot, sigma + bump) - _bs_price(spot, sigma - bump)) / (2.0 * bump)
        for bump in (eta, 2.0 * eta, 4.0 * eta)
    ]
    residual, steps = 1.0e-7, 800
    errors = [(2 * steps * residual) / (2.0 * bump) for bump in (eta, 2 * eta, 4 * eta)]
    epsilon_small, epsilon_large = errors[0] + errors[1], errors[1] + errors[2]
    assert abs(ladder[0] - ladder[1]) <= epsilon_small
    assert abs(ladder[1] - ladder[2]) <= epsilon_large
    charge = 1.0 / (1.0 - 2.0**-1.5)

    outcome = evaluate_fixed_bump_validation(
        quantity="vega",
        estimate_h=ladder[0],
        estimate_2h=ladder[1],
        estimate_4h=ladder[2],
        reference_estimate=ladder[0],
        epsilon_small=epsilon_small,
        epsilon_large=epsilon_large,
        cap=5.0e-2,
        minimum_order=1.5,
        maximum_order=2.5,
    )
    assert outcome["branch"] == "flat"
    assert outcome["bias_charge"] == pytest.approx(charge * epsilon_small, rel=0.0, abs=0.0)
    assert outcome["verdict"] == "pass"


def test_analytic_black_scholes_grid_stencil_delta() -> None:
    """A second-order stencil ladder whose Richardson reference is the BS delta."""
    spot, sigma = 100.0, 0.2
    analytic = float(
        black_scholes(option_type="call", spot=spot, volatility=sigma, **_BS)["delta"]
    )
    error = 2.0e-4  # candidate-rung grid error; halving the step divides it by four
    coarse = analytic + 4.0 * error
    candidate = analytic + error
    fine = analytic + error / 4.0

    expected_order = math.log2(abs(coarse - candidate) / abs(candidate - fine))
    assert expected_order == pytest.approx(2.0)
    expected_reference = fine + (fine - candidate) / 3.0
    assert expected_reference == pytest.approx(analytic, abs=1.0e-15)

    outcome = evaluate_grid_stencil_delta(
        coarse_delta=coarse,
        candidate_delta=candidate,
        fine_delta=fine,
        cap=1.0e-3,
        assumed_order=2.0,
        minimum_order=1.5,
        maximum_order=2.5,
    )
    assert outcome["observed_stencil_order"] == pytest.approx(expected_order, rel=0.0, abs=0.0)
    assert outcome["reference_stencil_delta"] == pytest.approx(
        expected_reference, rel=0.0, abs=0.0
    )
    assert outcome["absolute_error"] == pytest.approx(error, rel=1.0e-12)
    assert outcome["verdict"] == "pass"

    coarser = evaluate_grid_stencil_delta(
        coarse_delta=analytic + 4.0 * 2.0e-3,
        candidate_delta=analytic + 2.0e-3,
        fine_delta=analytic + 2.0e-3 / 4.0,
        cap=1.0e-3,
        assumed_order=2.0,
        minimum_order=1.5,
        maximum_order=2.5,
    )
    assert coarser["absolute_error"] == pytest.approx(2.0e-3, rel=1.0e-12)
    assert coarser["verdict"] == "fail"


# ---------------------------------------------------------------------------
# Finding 11: solve reuse and accounting
# ---------------------------------------------------------------------------


def test_the_solve_plan_reuses_one_base_surface_per_grid(
    config: PolicyConfigV2, american_case: CaseV2
) -> None:
    plan = v2.required_solve_keys(american_case, config)
    assert sum(1 for grid, role in plan if role == v2.BASE_ROLE) == 3
    assert plan.count((CANDIDATE_GRID, v2.DOMINANCE_CONTROL_ROLE)) == 1
    assert sum(1 for _, role in plan if role.startswith("volatility_")) == 8
    assert len(plan) == 12
    # No scalar centre or spot-bump duplication: the bumped prices ride on the
    # base surface of the same grid.
    solves = _solves(american_case, config)
    base = solves[(CANDIDATE_GRID, v2.BASE_ROLE)]
    assert len(base.bumped_prices) == 6


def test_planned_solve_counts_are_the_declared_ones(config: PolicyConfigV2) -> None:
    remediation = planned_solve_count(config, "remediation")
    confirmation = planned_solve_count(config, "confirmation")
    assert remediation == 116
    assert confirmation == 323
    assert remediation + confirmation == 439


def test_solve_accounting_reconciles(config: PolicyConfigV2) -> None:
    rows = _stage_rows(config, "remediation")
    accounting = solve_accounting(rows, config, "remediation")
    assert accounting["planned_surface_solves"] == 116
    assert accounting["attempted_surface_solves"] == 116
    assert accounting["completed_surface_solves"] == 116
    assert accounting["solver_exceptions"] == 0
    assert accounting["backward_inductions"] == 116
    assert accounting["psor_total_iterations"] > 0
    assert all(accounting["reconciliation"].values())
    assert sum(accounting["attempted_by_criticality"].values()) == 116
    assert sum(accounting["attempted_by_case_classification"].values()) == 116


def test_solve_accounting_counts_exceptions(config: PolicyConfigV2) -> None:
    rows = []
    for name in config.stage_case_names("remediation"):
        case = config.case(name)
        solves = _solves(case, config)
        failures: list[SolveFailure] = []
        if name == AMERICAN_CASE:
            solves, failures = _drop(case, solves, (CANDIDATE_GRID, v2.BASE_ROLE))
        rows.append(evaluate_case(case, config, solves, failures))
    accounting = solve_accounting(rows, config, "remediation")
    assert accounting["attempted_surface_solves"] == 116
    assert accounting["completed_surface_solves"] == 115
    assert accounting["solver_exceptions"] == 1
    assert accounting["backward_inductions"] == 115
    assert all(accounting["reconciliation"].values())


# ---------------------------------------------------------------------------
# Stage outcome, selection and lifecycle
# ---------------------------------------------------------------------------


def _stage_rows(config: PolicyConfigV2, stage: str = "remediation") -> list[dict[str, Any]]:
    return [
        evaluate_case(config.case(name), config, _solves(config.case(name), config))
        for name in config.stage_case_names(stage)
    ]


def test_only_the_nine_regular_remediation_cases_decide_selection(
    config: PolicyConfigV2,
) -> None:
    outcome = stage_outcome(_stage_rows(config), config)
    assert outcome["gate_eligible_case_count"] == 9
    assert outcome["all_gate_eligible_cases_pass"] is True
    assert select_accuracy_policy(outcome, config) == CANDIDATE_GRID


def test_one_gate_eligible_failure_refuses_selection(config: PolicyConfigV2) -> None:
    rows = _stage_rows(config)
    rows[0]["checks"]["price_absolute_error"] = False
    rows[0]["passes"] = False
    outcome = stage_outcome(rows, config)
    assert outcome["gate_eligible_failure_count"] == 1
    assert select_accuracy_policy(outcome, config) == v2.NO_POLICY_SELECTED


def test_state_one_remediation_failed_is_terminal_and_freezable(
    config: PolicyConfigV2,
) -> None:
    lifecycle = lifecycle_block(
        stage="remediation", all_gate_eligible_cases_pass=False, config=config
    )
    assert lifecycle["remediation_status"] == "failed"
    assert lifecycle["confirmation_status"] == "not_run"
    assert lifecycle["confirmation_not_run_reason"] == "remediation_failed"
    assert (lifecycle["terminal"], lifecycle["freezable"]) == (True, True)
    assert lifecycle["selected_accuracy_policy"] == v2.NO_POLICY_SELECTED


def test_state_two_remediation_passed_carries_no_selected_policy_field(
    config: PolicyConfigV2,
) -> None:
    lifecycle = lifecycle_block(
        stage="remediation", all_gate_eligible_cases_pass=True, config=config
    )
    assert lifecycle["confirmation_status"] == "pending"
    assert (lifecycle["terminal"], lifecycle["freezable"]) == (False, False)
    assert "selected_accuracy_policy" not in lifecycle


def test_state_three_confirmation_run_selects_pending_fresh_approval(
    config: PolicyConfigV2,
) -> None:
    selected = lifecycle_block(
        stage="confirmation", all_gate_eligible_cases_pass=True, config=config
    )
    assert selected["selected_accuracy_policy"] == CANDIDATE_GRID
    assert selected["selection_pending_fresh_top_level_approval"] is True
    assert selected["authorizes_dataset_generation"] is False
    refused = lifecycle_block(
        stage="confirmation", all_gate_eligible_cases_pass=False, config=config
    )
    assert refused["selected_accuracy_policy"] == v2.NO_POLICY_SELECTED


# ---------------------------------------------------------------------------
# Finding 5: authoritative, semantic confirmation entry
# ---------------------------------------------------------------------------


def _stage_report(
    config: PolicyConfigV2, *, stage: str = "remediation", fail: bool = False
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name in config.stage_case_names(stage):
        case = config.case(name)
        solves = _solves(case, config)
        failures: list[SolveFailure] = []
        if fail and name == config.gate_cases[0]:
            solves, failures = _drop(case, solves, (CANDIDATE_GRID, v2.BASE_ROLE))
        rows.append(evaluate_case(case, config, solves, failures))
    outcome = stage_outcome(rows, config)
    accounting = solve_accounting(rows, config, stage)
    lifecycle = lifecycle_block(
        stage=stage,
        all_gate_eligible_cases_pass=bool(outcome["all_gate_eligible_cases_pass"]),
        config=config,
    )
    report = build_report(
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
    # Round-trip through JSON so a verifier sees exactly what a reader sees.
    return json.loads(canonical_json(report))


def test_a_clean_remediation_report_verifies_and_admits_confirmation(
    config: PolicyConfigV2,
) -> None:
    report = _stage_report(config)
    recomputed = require_confirmation_entry(report, config)
    assert recomputed.stage == "remediation"
    assert recomputed.lifecycle["remediation_status"] == "passed"
    assert recomputed.accounting["attempted_surface_solves"] == 116


def _recording_solver():
    calls: list[Any] = []

    def solver(**kwargs: Any) -> dict:
        calls.append(kwargs)
        raise AssertionError("the solver must not be reached")

    solver.calls = calls  # type: ignore[attr-defined]
    return solver


def _relabel_failed_as_passed(report: dict[str, Any]) -> None:
    """Rewrite a failed/not_run lifecycle into a well-formed passed/pending one."""
    lifecycle = report["lifecycle"]
    lifecycle.update(
        {
            "remediation_status": "passed",
            "confirmation_status": "pending",
            "terminal": False,
            "freezable": False,
            "confirmation_not_run_reason": None,
        }
    )
    lifecycle.pop("selected_accuracy_policy", None)
    lifecycle.pop("selection_pending_fresh_top_level_approval", None)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        pytest.param(
            _relabel_failed_as_passed,
            "disagrees with the value recomputed",
            id="failed_not_run_relabelled_passed_pending",
        ),
        pytest.param(
            lambda r: r["lifecycle"].update(
                {"remediation_status": "passed", "confirmation_status": "pending"}
            ),
            "object key set is wrong",
            id="nonterminal_lifecycle_keeping_a_selection_field",
        ),
        pytest.param(
            lambda r: r["cases"][0]["checks"].update({"price_absolute_error": True}),
            # Now rejected at admission, by the exact failed_checks_by_name
            # reconciliation; the later recomputation gate still stands behind it.
            r"disagrees with the decisions its own raw solve records imply"
            r"|failed_checks_by_name key set is wrong",
            id="flipped_gate_boolean",
        ),
        pytest.param(
            lambda r: r["cases"][0].update({"passes": True}),
            "disagrees with the decisions",
            id="flipped_case_verdict",
        ),
        pytest.param(
            lambda r: r["cases"][0]["delta_eligibility"].update({"eligible": True}),
            "disagrees with the decisions",
            id="fabricated_delta_eligibility",
        ),
        pytest.param(
            lambda r: r["stage_outcome"].update({"gate_eligible_failure_count": 0}),
            "stage_outcome disagrees",
            id="edited_aggregate",
        ),
        pytest.param(
            lambda r: r["cases"].pop(2),
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
            lambda r: r["cases"][0]["state"].update({"name": "regular_euro_atm_put"}),
            "duplicated=|unexpected=",
            id="renamed_case",
        ),
        pytest.param(
            lambda r: r["cases"][0].update({"unexpected_block": 1}),
            r"unknown fields|object key set is wrong",
            id="unknown_case_key",
        ),
        pytest.param(
            lambda r: r.update({"unexpected_block": 1}),
            r"unknown fields|object key set is wrong",
            id="unknown_report_key",
        ),
        pytest.param(
            lambda r: r["source"].update({"pde_header_sha256": "0" * 64}),
            "does not match the repository",
            id="altered_source_digest",
        ),
        pytest.param(
            lambda r: r["source"].update(
                {"executable_source_composite_digest": "src-" + "0" * 64}
            ),
            "executable_source_composite_digest does not match",
            id="altered_composite_digest",
        ),
        pytest.param(
            lambda r: r.update({"criteria_digest": "crit-" + "0" * 64}),
            "criteria_digest does not match",
            id="altered_criteria_digest",
        ),
        pytest.param(
            lambda r: r["criteria_block"]["criteria"].update({"shape_absolute_floor": 1e-6}),
            "criteria_block does not match",
            id="altered_criteria_block",
        ),
        pytest.param(
            lambda r: r["predeclared_criteria"].update({"delta_absolute_error": 1.0}),
            "predeclared_criteria does not match",
            id="altered_predeclared_criteria",
        ),
        pytest.param(
            lambda r: r["source"].update({"raw_config_sha256": "0" * 64}),
            "raw_config_sha256 does not match",
            id="foreign_config",
        ),
        pytest.param(
            lambda r: r["non_claims"].pop(),
            "non_claims must be the predeclared statements",
            id="dropped_non_claim",
        ),
        pytest.param(
            lambda r: r["cases"][0]["solves"].pop(),
            r"does not reconcile|solve_problems key set is wrong",
            id="dropped_solve_record",
        ),
        pytest.param(
            lambda r: r["cases"][0]["solves"][0].update({"maximum_lcp_residual": 1.0}),
            r"disagrees with the decisions|solve_problems key set is wrong",
            id="edited_raw_residual",
        ),
    ],
)
def test_confirmation_entry_rejects_a_mutated_remediation_report(
    config: PolicyConfigV2, mutate, match: str
) -> None:
    report = _stage_report(config, fail=True)
    mutate(report)
    solver = _recording_solver()
    with pytest.raises(LabelPolicyV2Error, match=match):
        require_confirmation_entry(report, config)
    assert solver.calls == []


def test_a_genuinely_failed_remediation_is_refused_before_the_solver(
    config: PolicyConfigV2,
) -> None:
    report = _stage_report(config, fail=True)
    solver = _recording_solver()
    with pytest.raises(LabelPolicyV2Error, match="recomputed remediation stage did not pass"):
        require_confirmation_entry(report, config)
    assert solver.calls == []


def test_confirmation_entry_refuses_a_confirmation_stage_report(
    config: PolicyConfigV2,
) -> None:
    report = _stage_report(config, stage="confirmation")
    with pytest.raises(LabelPolicyV2Error, match="requires a remediation-stage report"):
        require_confirmation_entry(report, config)


def test_confirmation_entry_refuses_a_criterion_changed_after_the_result() -> None:
    original = load_policy_config_v2(CONFIG_V2)
    report = _stage_report(original)
    document = _document()
    document["criteria"]["shape_absolute_floor"] = 1.0e-6
    with pytest.raises(LabelPolicyV2Error, match="must stay 1e-08"):
        parse_policy_config_v2(
            document, source_name="x.toml", raw_config_sha256=original.raw_config_sha256
        )
    # And a report produced under a different raw configuration is refused too.
    foreign = parse_policy_config_v2(
        _document(), source_name="x.toml", raw_config_sha256="f" * 64
    )
    with pytest.raises(LabelPolicyV2Error, match="raw_config_sha256 does not match"):
        require_confirmation_entry(report, foreign)


def test_verify_stage_report_accepts_an_untouched_confirmation_report(
    config: PolicyConfigV2,
) -> None:
    report = _stage_report(config, stage="confirmation")
    recomputed = verify_stage_report(report, config)
    assert recomputed.stage == "confirmation"
    assert recomputed.accounting["attempted_surface_solves"] == 323
    assert recomputed.lifecycle["selected_accuracy_policy"] == CANDIDATE_GRID


# ---------------------------------------------------------------------------
# Finding 10: one-shot enforcement is honest, and --overwrite is gone
# ---------------------------------------------------------------------------


def test_the_stage_cli_has_no_overwrite_flag() -> None:
    options = {
        option
        for action in v2.build_parser()._actions
        for option in action.option_strings
    }
    assert "--overwrite" not in options
    assert options == {"-h", "--help", "--config", "--stage", "--output-directory",
                       "--remediation-report"}


def test_the_stage_cli_refuses_a_nonempty_output_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "report.json").write_text("{}", encoding="utf-8")
    code = v2.main(
        [
            "--config",
            str(CONFIG_V2),
            "--stage",
            "remediation",
            "--output-directory",
            str(output),
        ]
    )
    assert code == 2
    assert "refusing to run into non-empty" in capsys.readouterr().err


def test_one_shot_enforcement_is_documented_as_best_effort(config: PolicyConfigV2) -> None:
    lifecycle = lifecycle_block(
        stage="remediation", all_gate_eligible_cases_pass=False, config=config
    )
    assert lifecycle["one_shot_enforcement"] == "procedural_and_provenance_backed"
    text = " ".join(v2.NON_CLAIMS)
    assert "best effort" in text
    assert "another directory or another clone cannot be detected" in text


# ---------------------------------------------------------------------------
# Report and publication
# ---------------------------------------------------------------------------


def test_the_report_publishes_every_predeclared_non_claim(config: PolicyConfigV2) -> None:
    report = _stage_report(config, fail=True)
    text = " ".join(report["non_claims"])
    assert "unavailable per production row" in text
    assert "cannot re-estimate its own vega bump bias" in text
    assert "do not validate the surrounding hyperrectangle" in text
    assert "Task 9C-C2b2 must separately predeclare" in text
    assert "No dataset and no training input is authorized" in text
    assert "cannot authenticate a fully coordinated fabricated numerical report" in text
    assert report["eligibility_contract"]["authorized_training_input_statuses"] == []
    assert not v2.AUTHORIZED_TRAINING_INPUT_STATUSES


def test_the_report_is_canonical_finite_json(config: PolicyConfigV2) -> None:
    report = _stage_report(config, fail=True)
    assert canonical_json(report).endswith("\n")
    with pytest.raises(LabelPolicyV2Error, match="finite canonical JSON"):
        canonical_json({"broken": math.inf})


def test_publication_is_deterministic_and_refuses_to_overwrite(
    config: PolicyConfigV2, tmp_path: Path
) -> None:
    report = _stage_report(config, fail=True)
    left, right = tmp_path / "left", tmp_path / "right"
    paths = write_outputs(copy.deepcopy(report), left)
    assert sorted(path.name for path in paths) == [
        "cases.csv",
        "eligibility.csv",
        "failures.csv",
        "report.json",
    ]
    write_outputs(copy.deepcopy(report), right)
    for name in ("report.json", "cases.csv", "eligibility.csv", "failures.csv"):
        assert (left / name).read_bytes() == (right / name).read_bytes()
    with pytest.raises(LabelPolicyV2Error, match="refusing to overwrite"):
        write_outputs(report, left)


def test_the_failures_csv_carries_the_structured_exception_record(
    config: PolicyConfigV2, tmp_path: Path
) -> None:
    report = _stage_report(config, fail=True)
    write_outputs(report, tmp_path / "run")
    text = (tmp_path / "run" / "failures.csv").read_text()
    assert "criticality" in text.splitlines()[0]
    assert v2.PRICE_CRITICAL in text
    assert "PsorConvergenceError" in text


# ---------------------------------------------------------------------------
# v1 separation
# ---------------------------------------------------------------------------


def test_v2_cannot_collide_with_any_v1_artefact() -> None:
    assert v2.SCHEMA_VERSION != "pde-label-policy-pilot/1"
    assert v2.REPORT_SCHEMA_VERSION != "pde-label-policy-report/1"
    assert CONFIG_V2 != CONFIG_V1
    v1_snapshot = json.loads(RESULT_V1.read_text())
    assert v1_snapshot["recommendation"]["selected_accuracy_policy"] == "no_policy_selected"
    assert (
        tuple(v1_snapshot["policy_outcomes"][CANDIDATE_GRID]["regular_failures"])
        == v2.V1_REGULAR_FAILURES
    )


def test_contract_multiplier_is_reporting_only_and_settlement_is_not() -> None:
    echo = {
        "option_type": "put",
        "exercise_style": "american",
        "strike": 100.0,
        "valuation_time": 0.0,
        "expiry_time": 1.0,
        "volatility": 0.22,
        "continuous_carry": 0.0,
        "curve_times": [0.0, 1.0],
        "curve_log_discounts": [0.0, -0.06],
        "dividends": [],
        "settlement": "cash",
        "contract_multiplier": 100.0,
        "dividends_declared": True,
        "spot_intervals": 1600,
        "time_steps": 800,
        "spot_maximum": 400.0,
        "rannacher_steps": 2,
        "psor_tolerance": 1.0e-11,
        "psor_relaxation": 1.2,
        "psor_maximum_iterations": 200000,
        "boundary_exclusion_nodes": 2,
    }
    assert pricing_grid_identity(echo) == pricing_grid_identity(
        {**echo, "contract_multiplier": 1.0}
    )
    assert pricing_grid_identity(echo) != pricing_grid_identity(
        {**echo, "settlement": "physical"}
    )


def test_criteria_digest_is_stable_and_moves_with_the_case_design(
    config: PolicyConfigV2,
) -> None:
    assert criteria_digest(config) == criteria_digest(load_policy_config_v2(CONFIG_V2))
    assert config.case_design_digest in canonical_json(v2.criteria_block(config))


# ---------------------------------------------------------------------------
# The frozen price-reference rule: raw 3200x1600 centre node, unconditional
# ---------------------------------------------------------------------------


def test_the_price_reference_is_the_raw_reference_grid_centre_node(
    config: PolicyConfigV2, american_case: CaseV2
) -> None:
    solves = _solves(american_case, config)
    outcome = evaluate_case(american_case, config, solves)
    price = outcome["price"]
    assert price["reference_method"] == v2.PRICE_REFERENCE_METHOD == "raw_grid_3200x1600_center"
    assert price["reference_grid"] == REFERENCE_GRID
    assert price["candidate_grid"] == CANDIDATE_GRID
    # Bitwise the raw centre values, with nothing extrapolated in between.
    assert price["candidate"] == solves[(CANDIDATE_GRID, v2.BASE_ROLE)].centre_price
    assert price["reference"] == solves[(REFERENCE_GRID, v2.BASE_ROLE)].centre_price
    assert price["absolute_error"] == abs(price["candidate"] - price["reference"])


def test_the_price_reference_tracks_the_reference_grid_alone(
    config: PolicyConfigV2, american_case: CaseV2
) -> None:
    """A Richardson or fallback reference would not move one-for-one."""
    shift = 3.0e-4
    solves = _solves(american_case, config)
    reference = solves[(REFERENCE_GRID, v2.BASE_ROLE)]
    moved = _replace(
        solves, (REFERENCE_GRID, v2.BASE_ROLE), centre_price=reference.centre_price + shift
    )
    outcome = evaluate_case(american_case, config, moved)
    assert outcome["price"]["reference"] == reference.centre_price + shift
    assert outcome["price"]["absolute_error"] == pytest.approx(shift, rel=1.0e-12)
    assert outcome["passes"] is True


def test_the_order_probe_price_cannot_affect_the_price_reference_or_verdict(
    config: PolicyConfigV2, american_case: CaseV2
) -> None:
    """grid_800x400 is price-irrelevant: only its stencil delta is consumed."""
    baseline = evaluate_case(american_case, config, _solves(american_case, config))
    solves = _replace(
        _solves(american_case, config), (ORDER_PROBE_GRID, v2.BASE_ROLE), centre_price=1234.5
    )
    mutated = evaluate_case(american_case, config, solves)
    assert mutated["price"] == baseline["price"]
    assert mutated["checks"] == baseline["checks"]
    assert mutated["passes"] is baseline["passes"] is True
    assert mutated["grid_stencil_delta_validation"] == baseline[
        "grid_stencil_delta_validation"
    ]
    assert mutated["delta_eligibility"] == baseline["delta_eligibility"]
    assert mutated["vega_eligibility"] == baseline["vega_eligibility"]


def test_the_order_probe_delta_moves_only_the_stencil_order_diagnostic(
    config: PolicyConfigV2, american_case: CaseV2
) -> None:
    baseline = evaluate_case(american_case, config, _solves(american_case, config))
    probe = _solves(american_case, config)[(ORDER_PROBE_GRID, v2.BASE_ROLE)]
    solves = _replace(
        _solves(american_case, config),
        (ORDER_PROBE_GRID, v2.BASE_ROLE),
        centre_delta=probe.centre_delta + 5.0e-4,
    )
    mutated = evaluate_case(american_case, config, solves)
    stencil_before = baseline["grid_stencil_delta_validation"]
    stencil_after = mutated["grid_stencil_delta_validation"]
    # The observed order moves; the reference and the price do not.
    assert stencil_after["observed_stencil_order"] != stencil_before["observed_stencil_order"]
    assert stencil_after["reference_stencil_delta"] == stencil_before[
        "reference_stencil_delta"
    ]
    assert stencil_after["absolute_error"] == stencil_before["absolute_error"]
    assert mutated["price"] == baseline["price"]
    assert mutated["passes"] is True


def test_an_unsupported_probe_order_denies_delta_but_never_the_price(
    config: PolicyConfigV2, american_case: CaseV2
) -> None:
    probe = _solves(american_case, config)[(ORDER_PROBE_GRID, v2.BASE_ROLE)]
    solves = _replace(
        _solves(american_case, config),
        (ORDER_PROBE_GRID, v2.BASE_ROLE),
        centre_delta=probe.centre_delta + 5.0e-4,
    )
    outcome = evaluate_case(american_case, config, solves)
    assert outcome["grid_stencil_delta_validation"]["checks"][
        "observed_stencil_order_supported"
    ] is False
    assert outcome["delta_eligibility"]["eligible"] is False
    assert outcome["passes"] is True


def test_the_anchor_rung_cannot_move_the_price_reference(config: PolicyConfigV2) -> None:
    case = config.case(EUROPEAN_ANCHOR_CASE)
    baseline = evaluate_case(case, config, _solves(case, config))
    solves = _replace(
        _solves(case, config), (ANCHOR_GRID, v2.BASE_ROLE), centre_price=999.0
    )
    mutated = evaluate_case(case, config, solves)
    assert mutated["price"] == baseline["price"]
    assert mutated["passes"] is True
    assert mutated["anchor_validation"]["anchor_price"] == 999.0
    assert mutated["anchor_validation"]["gates"] is False


def test_richardson_never_touches_the_price_reference(config: PolicyConfigV2) -> None:
    contract = richardson_contract_block(config)
    assert contract["applies_to_price_reference"] is False
    assert contract["applies_to"] == ["grid_stencil_delta_reference", "anchor_diagnostics"]
    conventions = conventions_block(config)
    assert conventions["price_reference_is_richardson_extrapolated"] is False
    assert conventions["price_reference_has_conditional_fallback"] is False
    assert conventions["order_probe_is_price_irrelevant"] is True


def test_no_document_claims_a_richardson_extrapolated_v2_price_reference() -> None:
    """v1's reference was extrapolated; no v2 statement may say v2's is."""
    contract = (PROJECT_ROOT / "docs/pde-numerical-contract.md").read_text()
    section = contract[contract.index("## Task 9C-C3: label-policy v2") :]
    assert 'price_reference_method = "raw_grid_3200x1600_center"' in section
    assert "No price Richardson extrapolation exists" in section
    assert "no conditional" in section
    assert "`grid_800x400` is price-irrelevant" in section
    # Any surviving mention of the v1 phrasing must name v1 on the same line.
    for line in section.splitlines():
        if "selectively Richardson-extrapolated" in line:
            assert "v1" in line, line

    for document in (
        PROJECT_ROOT / "docs/tasks/active/task-9c-c3-label-policy-v2.md",
        PROJECT_ROOT / "configs/pde_label_policy_pilot_v2.toml",
        PROJECT_ROOT / "scripts/freeze_pde_label_policy_v2_results.py",
    ):
        text = document.read_text()
        assert "selectively Richardson-extrapolated" not in text, document
    # The freeze tool's own snapshot limitation says the opposite, explicitly.
    freeze_text = (
        PROJECT_ROOT / "scripts/freeze_pde_label_policy_v2_results.py"
    ).read_text()
    assert "never " in freeze_text and "Richardson-extrapolated" in freeze_text



# ---------------------------------------------------------------------------
# Finding: pinned protocol strings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("key", "constant"),
    [
        ("name", v2.STUDY_NAME),
        ("delta_method", v2.DELTA_METHOD),
        ("vega_method", v2.VEGA_METHOD),
        ("price_reference_method", v2.PRICE_REFERENCE_METHOD),
    ],
)
def test_a_protocol_string_is_pinned_not_descriptive(key: str, constant: str) -> None:
    document = _document()
    assert document["study"][key] == constant
    document["study"][key] = "something_plausible_but_different"
    with pytest.raises(LabelPolicyV2Error, match=f"\\[study\\].{key} must be exactly"):
        _parse(document)


def test_the_report_is_built_from_the_pinned_protocol_constants(
    config: PolicyConfigV2,
) -> None:
    report = _stage_report(config, fail=True)
    assert report["study"]["name"] == v2.STUDY_NAME
    assert report["conventions"]["delta_method"] == v2.DELTA_METHOD
    assert report["conventions"]["vega_method"] == v2.VEGA_METHOD
    assert report["conventions"]["price_reference_method"] == v2.PRICE_REFERENCE_METHOD
    assert report["validation_rules"]["price"]["reference_method"] == (
        v2.PRICE_REFERENCE_METHOD
    )
    block = v2.criteria_block(config)
    assert block["price_reference_method"] == v2.PRICE_REFERENCE_METHOD
    assert block["study_name"] == v2.STUDY_NAME


# ---------------------------------------------------------------------------
# Exact nested schemas
# ---------------------------------------------------------------------------

NESTED_SECTIONS = [
    "study",
    "conventions",
    "validation_rules",
    "eligibility_contract",
    "richardson_contract",
    "solver",
    "performance",
]


@pytest.mark.parametrize("section", NESTED_SECTIONS)
def test_an_unknown_field_in_a_nested_section_is_rejected(
    config: PolicyConfigV2, section: str
) -> None:
    report = _stage_report(config, fail=True)
    report[section]["injected_field"] = 1
    solver = _recording_solver()
    with pytest.raises(LabelPolicyV2Error):
        require_confirmation_entry(report, config)
    assert solver.calls == []


@pytest.mark.parametrize("section", NESTED_SECTIONS)
def test_a_removed_field_in_a_nested_section_is_rejected(
    config: PolicyConfigV2, section: str
) -> None:
    report = _stage_report(config, fail=True)
    report[section].pop(sorted(report[section])[0])
    with pytest.raises(LabelPolicyV2Error):
        require_confirmation_entry(report, config)


@pytest.mark.parametrize(
    "path",
    [
        ("validation_rules", "fixed_bump"),
        ("validation_rules", "grid_stencil"),
        ("validation_rules", "price"),
        ("validation_rules", "shape"),
        ("performance", "host"),
        ("criteria_block", "lifecycle"),
    ],
)
def test_an_unknown_field_two_levels_deep_is_rejected(
    config: PolicyConfigV2, path: tuple[str, str]
) -> None:
    report = _stage_report(config, fail=True)
    report[path[0]][path[1]]["injected_field"] = 1
    with pytest.raises(LabelPolicyV2Error):
        require_confirmation_entry(report, config)
    report = _stage_report(config, fail=True)
    block = report[path[0]][path[1]]
    block.pop(sorted(block)[0])
    with pytest.raises(LabelPolicyV2Error):
        require_confirmation_entry(report, config)


def test_a_grid_record_is_schema_checked(config: PolicyConfigV2) -> None:
    report = _stage_report(config, fail=True)
    report["grids"][2]["injected_field"] = 1
    with pytest.raises(LabelPolicyV2Error, match=r"grids|object key set is wrong"):
        require_confirmation_entry(report, config)
    report = _stage_report(config, fail=True)
    report["grids"].append(dict(report["grids"][0]))
    with pytest.raises(LabelPolicyV2Error):
        require_confirmation_entry(report, config)


def test_a_retyped_nested_value_is_rejected(config: PolicyConfigV2) -> None:
    report = _stage_report(config, fail=True)
    report["solver"]["spot_intervals_typo"] = None
    with pytest.raises(LabelPolicyV2Error):
        require_confirmation_entry(report, config)
    report = _stage_report(config, fail=True)
    report["performance"]["timed_solve_attempts"] = -1
    with pytest.raises(LabelPolicyV2Error, match=r"integer >= 0"):
        require_confirmation_entry(report, config)
    report = _stage_report(config, fail=True)
    report["performance"]["wall_seconds"] = "fast"
    with pytest.raises(LabelPolicyV2Error, match=r"expected finite real number"):
        require_confirmation_entry(report, config)


# ---------------------------------------------------------------------------
# Every published aggregate is recomputed
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


@pytest.mark.parametrize("block", ["stage_outcome", "solve_accounting"])
def test_every_published_aggregate_is_recomputed_and_rejected_if_edited(
    config: PolicyConfigV2, block: str
) -> None:
    baseline = _stage_report(config, fail=True)
    keys = sorted(baseline[block])
    assert keys, block
    accepted: list[str] = []
    for key in keys:
        report = copy.deepcopy(baseline)
        report[block][key] = _mutated_value(report[block][key])
        try:
            require_confirmation_entry(report, config)
        except LabelPolicyV2Error:
            continue
        accepted.append(key)
    assert accepted == [], f"{block} aggregates accepted after mutation: {accepted}"


@pytest.mark.parametrize(
    ("block", "key", "value"),
    [
        ("solve_accounting", "planned_surface_solves", 999),
        ("solve_accounting", "psor_total_iterations", 999),
        ("stage_outcome", "descriptive_failure_count", 999),
        (
            "stage_outcome",
            "failed_checks_by_name",
            {"price_absolute_error": ["regular_euro_atm_put"]},
        ),
    ],
)
def test_the_named_aggregate_mutations_are_rejected(
    config: PolicyConfigV2, block: str, key: str, value: Any
) -> None:
    report = _stage_report(config, fail=True)
    report[block][key] = value
    solver = _recording_solver()
    with pytest.raises(
        LabelPolicyV2Error,
        match=r"disagrees with the value recomputed|key set is wrong|value is wrong",
    ):
        require_confirmation_entry(report, config)
    assert solver.calls == []


# ---------------------------------------------------------------------------
# Provenance: the newly tracked option translation unit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["option_header", "option_implementation"])
def test_a_mutation_to_the_option_translation_unit_invalidates_verification(
    config: PolicyConfigV2, name: str
) -> None:
    report = _stage_report(config, fail=True)
    report["source"][f"{name}_sha256"] = "0" * 64
    with pytest.raises(LabelPolicyV2Error, match=f"{name}_sha256 does not match"):
        require_confirmation_entry(report, config)


def test_the_inventory_matches_the_pde_include_closure() -> None:
    assert set(v2.PDE_INCLUDE_CLOSURE) == {
        v2.EXECUTABLE_SOURCE_INVENTORY[name]
        for name in (
            "pde_binding_source",
            "pde_header",
            "pde_implementation",
            "option_header",
            "option_implementation",
        )
    }
    # The walk itself: every first-party include edge reachable from the PDE
    # path resolves to a file already in the inventory.
    inventoried = set(v2.EXECUTABLE_SOURCE_INVENTORY.values())
    for relative in v2.PDE_INCLUDE_CLOSURE:
        text = (PROJECT_ROOT / relative).read_text()
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith('#include "'):
                continue
            included = stripped.split('"')[1]
            assert included.startswith("dp/"), (relative, included)
            resolved = f"cpp/include/{included}"
            assert resolved in inventoried, (relative, resolved)


# ---------------------------------------------------------------------------
# Strict JSON: duplicate object keys and non-standard constants
# ---------------------------------------------------------------------------


def _inject_duplicate(text: str, anchor: str, line: str) -> str:
    """Insert ``line`` just inside the first ``{`` following ``anchor``.

    Textual surgery on the serialized file, because a duplicate key cannot be
    expressed as a Python dict: ``json.loads`` keeps only the last value, so a
    dict-level test would prove nothing.
    """
    start = text.index(anchor)
    brace = text.index("{", start)
    return f"{text[: brace + 1]}\n{line}{text[brace + 1 :]}"


def _sentinels(monkeypatch: pytest.MonkeyPatch) -> tuple[list[Any], list[Any]]:
    """Replace run_stage and the solver with counters that must stay empty."""
    stage_calls: list[Any] = []
    solver_calls: list[Any] = []

    def fake_run_stage(*args: Any, **kwargs: Any) -> dict[str, Any]:
        stage_calls.append((args, kwargs))
        raise AssertionError("run_stage must not be reached")

    def fake_solver(**kwargs: Any) -> dict[str, Any]:
        solver_calls.append(kwargs)
        raise AssertionError("the solver must not be reached")

    monkeypatch.setattr(v2, "run_stage", fake_run_stage)
    monkeypatch.setattr(v2, "pde_valuation_surface", fake_solver)
    return stage_calls, solver_calls


@pytest.mark.parametrize(
    ("anchor", "line", "expected_path"),
    [
        pytest.param("{", '  "schema_version": "duplicated",', "$.schema_version",
                     id="duplicate_top_level_key"),
        pytest.param('"study": ', '    "name": "duplicated",', ".study.name",
                     id="duplicate_nested_study_name"),
        pytest.param('"validation_rules": ', '    "gating_checks": [],',
                     ".validation_rules.gating_checks", id="duplicate_in_validation_rules"),
        pytest.param('"solves": [', '        "role": "duplicated",',
                     ".cases[0].solves[0].role", id="duplicate_inside_an_array_object"),
        pytest.param('"host": ', '      "system": "duplicated",',
                     ".performance.host.system", id="duplicate_three_levels_deep"),
    ],
)
def test_a_serialized_duplicate_key_is_refused_before_anything_runs(
    config: PolicyConfigV2,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    anchor: str,
    line: str,
    expected_path: str,
) -> None:
    stage_calls, solver_calls = _sentinels(monkeypatch)
    text = v2.canonical_json(_stage_report(config, fail=True))
    path = tmp_path / "report.json"
    path.write_text(_inject_duplicate(text, anchor, line), encoding="utf-8")

    with pytest.raises(LabelPolicyV2Error, match="repeats object keys") as raised:
        v2.load_strict_json(path, description="remediation report")
    assert expected_path in str(raised.value)

    output = tmp_path / "run"
    code = v2.main(
        [
            "--config", str(CONFIG_V2),
            "--stage", "confirmation",
            "--output-directory", str(output),
            "--remediation-report", str(path),
        ]
    )
    assert code == 2
    assert stage_calls == []
    assert solver_calls == []
    assert not output.exists() or not list(output.iterdir())


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_a_non_standard_json_constant_is_refused(
    config: PolicyConfigV2, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, constant: str
) -> None:
    stage_calls, solver_calls = _sentinels(monkeypatch)
    text = v2.canonical_json(_stage_report(config, fail=True))
    assert '"wall_seconds": 0.0' in text
    path = tmp_path / "report.json"
    path.write_text(text.replace('"wall_seconds": 0.0', f'"wall_seconds": {constant}'), "utf-8")

    with pytest.raises(LabelPolicyV2Error, match="non-standard JSON constant"):
        v2.load_strict_json(path, description="remediation report")

    output = tmp_path / "run"
    code = v2.main(
        [
            "--config", str(CONFIG_V2),
            "--stage", "confirmation",
            "--output-directory", str(output),
            "--remediation-report", str(path),
        ]
    )
    assert code == 2
    assert stage_calls == []
    assert solver_calls == []


def test_the_strict_loader_still_accepts_a_clean_document(
    config: PolicyConfigV2, tmp_path: Path
) -> None:
    path = tmp_path / "report.json"
    path.write_text(v2.canonical_json(_stage_report(config, fail=True)), encoding="utf-8")
    loaded = v2.load_strict_json(path, description="remediation report")
    assert loaded["schema_version"] == v2.REPORT_SCHEMA_VERSION
    require_confirmation_entry(_stage_report(config), config)


def test_the_duplicate_check_reaches_every_nesting_depth() -> None:
    for text, expected in (
        ('{"a": 1, "a": 2}', "$.a"),
        ('{"o": {"b": 1, "b": 2}}', ".o.b"),
        ('{"l": [{"c": 1, "c": 2}]}', ".l[0].c"),
        ('{"l": [[{"d": 1, "d": 2}]]}', ".l[0][0].d"),
    ):
        with pytest.raises(LabelPolicyV2Error, match="repeats object keys") as raised:
            v2.strict_json_loads(text, description="fixture")
        assert expected in str(raised.value)


# ---------------------------------------------------------------------------
# Exhaustive recursive type validation
# ---------------------------------------------------------------------------


def _incompatible(value: Any) -> Any:
    """A value that no correct spec for this node can accept."""
    if type(value) is bool:
        return 1  # bool is a subclass of int; this is the adversarial case
    if type(value) is int:
        return True
    if type(value) is float:
        return "not a number"
    if type(value) is str:
        return 7
    if type(value) is list:
        return 7
    if type(value) is dict:
        return 7
    return {"injected_incompatible": True}  # for null


def _nodes(document: Any, path: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    found = [path]
    if isinstance(document, dict):
        for key in document:
            found.extend(_nodes(document[key], (*path, key)))
    elif isinstance(document, list):
        for index, item in enumerate(document):
            found.extend(_nodes(item, (*path, index)))
    return found


def _normalise(path: tuple[Any, ...]) -> str:
    return "".join("[]" if isinstance(part, int) else f".{part}" for part in path)


def _distinct_positions(document: Any) -> dict[str, tuple[Any, ...]]:
    positions: dict[str, tuple[Any, ...]] = {}
    for path in _nodes(document):
        if not path:
            continue
        positions.setdefault(_normalise(path), path)
    return positions


def _set_at(document: Any, path: tuple[Any, ...], value: Any) -> None:
    target = document
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value


def test_every_serialized_report_position_is_type_validated(
    config: PolicyConfigV2,
) -> None:
    """Substitute an incompatible type at every distinct schema position."""
    baseline = _stage_report(config, fail=True)
    v2.validate_report_schema(baseline)
    positions = _distinct_positions(baseline)
    assert len(positions) > 250, len(positions)
    accepted: list[str] = []
    for name, path in sorted(positions.items()):
        document = copy.deepcopy(baseline)
        _set_at(document, path, _incompatible(_read_at(baseline, path)))
        try:
            v2.validate_report_schema(document)
        except LabelPolicyV2Error:
            continue
        accepted.append(name)
    assert accepted == [], f"report positions accepted an incompatible type: {accepted}"


def _read_at(document: Any, path: tuple[Any, ...]) -> Any:
    target = document
    for part in path:
        target = target[part]
    return target


def test_a_confirmation_report_is_type_validated_too(config: PolicyConfigV2) -> None:
    baseline = _stage_report(config, stage="confirmation", fail=False)
    v2.validate_report_schema(baseline)
    positions = _distinct_positions(baseline)
    accepted = []
    for name, path in sorted(positions.items()):
        document = copy.deepcopy(baseline)
        _set_at(document, path, _incompatible(_read_at(baseline, path)))
        try:
            v2.validate_report_schema(document)
        except LabelPolicyV2Error:
            continue
        accepted.append(name)
    assert accepted == [], accepted


@pytest.mark.parametrize(
    ("path", "value"),
    [
        pytest.param(("cases", 1, "solves", 0, "centre_spot_is_exact_node"), 1,
                     id="centre_spot_is_exact_node_true_to_1"),
        pytest.param(("cases", 1, "solves", 0, "centre_greek_eligible"), 1,
                     id="centre_greek_eligible_true_to_1"),
        pytest.param(("cases", 1, "solves", 0, "bumped_spots_are_exact_nodes"), 1,
                     id="bumped_spots_are_exact_nodes_true_to_1"),
        pytest.param(("cases", 0, "fixed_bump_delta_validation", "branch"), 7,
                     id="fixed_bump_delta_branch_null_to_7"),
        pytest.param(("cases", 0, "solve_failures", 0, "exception_class"), 7,
                     id="solve_failures_exception_class_to_7"),
        pytest.param(("cases", 0, "delta_eligibility", "reasons"), 7,
                     id="delta_ineligibility_reasons_to_7"),
    ],
)
def test_the_named_adversarial_type_mutations_are_rejected(
    config: PolicyConfigV2,
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[Any, ...],
    value: Any,
) -> None:
    stage_calls, solver_calls = _sentinels(monkeypatch)
    report = _stage_report(config, fail=True)
    _set_at(report, path, value)
    with pytest.raises(LabelPolicyV2Error, match="expected"):
        require_confirmation_entry(report, config)
    assert stage_calls == []
    assert solver_calls == []


def test_type_errors_name_the_path_the_expectation_and_the_observation(
    config: PolicyConfigV2,
) -> None:
    report = _stage_report(config, fail=True)
    _set_at(report, ("cases", 1, "solves", 0, "centre_greek_eligible"), 1)
    with pytest.raises(LabelPolicyV2Error) as raised:
        v2.validate_report_schema(report)
    message = str(raised.value)
    assert "cases[1].solves[0].centre_greek_eligible" in message
    assert "expected boolean" in message
    assert "observed int 1" in message


def test_the_schema_validates_before_any_dataclass_is_constructed(
    config: PolicyConfigV2, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[Any] = []
    original = v2.surface_solve_from_record

    def recording(record: Mapping[str, Any]) -> v2.SurfaceSolve:
        built.append(record)
        return original(record)

    monkeypatch.setattr(v2, "surface_solve_from_record", recording)
    report = _stage_report(config, fail=True)
    _set_at(report, ("cases", 1, "solves", 0, "centre_greek_eligible"), 1)
    with pytest.raises(LabelPolicyV2Error):
        v2.recompute_stage(report, config)
    assert built == []


def test_a_null_is_accepted_only_where_declared(config: PolicyConfigV2) -> None:
    report = _stage_report(config, fail=True)
    # Declared nullable: the ladder was not computable for the failing case.
    assert report["cases"][0]["fixed_bump_delta_validation"]["branch"] is None
    v2.validate_report_schema(report)
    # Not nullable.
    _set_at(report, ("cases", 0, "passes"), None)
    with pytest.raises(LabelPolicyV2Error, match="expected boolean"):
        v2.validate_report_schema(report)


# ---------------------------------------------------------------------------
# Role-exact bumped_prices
# ---------------------------------------------------------------------------


def _all_solve_positions(report: Mapping[str, Any]) -> dict[tuple[str, str], tuple[int, int]]:
    """Locate one serialized solve per distinct (grid, role) in the report."""
    found: dict[tuple[str, str], tuple[int, int]] = {}
    for case_index, case in enumerate(report["cases"]):
        for solve_index, solve in enumerate(case["solves"]):
            found.setdefault(
                (solve["grid_name"], solve["role"]), (case_index, solve_index)
            )
    return found


def test_the_bumped_price_vocabulary_is_the_pinned_ladder(config: PolicyConfigV2) -> None:
    assert set(v2.CANONICAL_SPOT_BUMP_KEYS) == {
        v2.spot_bump_key(bump, direction)
        for bump in config.bumps.spot
        for direction in ("down", "up")
    }
    assert len(v2.CANONICAL_SPOT_BUMP_KEYS) == 6


@pytest.mark.parametrize("stage", ["remediation", "confirmation"])
def test_every_solve_role_carries_exactly_its_required_bump_keys(
    config: PolicyConfigV2, stage: str
) -> None:
    report = _stage_report(config, stage=stage, fail=stage == "remediation")
    positions = _all_solve_positions(report)
    # Every distinct role that this stage actually solves.
    assert len(positions) == 13, sorted(positions)
    for (grid_name, role), (case_index, solve_index) in sorted(positions.items()):
        published = report["cases"][case_index]["solves"][solve_index]["bumped_prices"]
        expected = v2.expected_bumped_price_keys(config, grid_name, role)
        assert set(published) == expected, (grid_name, role)
        if role == v2.BASE_ROLE and grid_name in {CANDIDATE_GRID, REFERENCE_GRID}:
            assert len(expected) == 6
        else:
            assert expected == frozenset()


def _solve_role_cases(stage: str = "remediation") -> list[Any]:
    config = load_policy_config_v2(CONFIG_V2)
    report = _stage_report(config, stage=stage, fail=True)
    return [
        pytest.param(grid_name, role, id=f"{grid_name}/{role}")
        for grid_name, role in sorted(_all_solve_positions(report))
    ]


@pytest.mark.parametrize(("grid_name", "role"), _solve_role_cases())
def test_an_unexpected_bumped_price_key_is_rejected_for_every_role(
    config: PolicyConfigV2,
    monkeypatch: pytest.MonkeyPatch,
    grid_name: str,
    role: str,
) -> None:
    stage_calls, solver_calls = _sentinels(monkeypatch)
    report = _stage_report(config, fail=True)
    case_index, solve_index = _all_solve_positions(report)[(grid_name, role)]
    report["cases"][case_index]["solves"][solve_index]["bumped_prices"][
        "unexpected_serialized_key"
    ] = 1.0
    with pytest.raises(LabelPolicyV2Error, match=r"expected|key set is wrong") as raised:
        require_confirmation_entry(report, config)
    assert "bumped_prices" in str(raised.value)
    assert stage_calls == []
    assert solver_calls == []


@pytest.mark.parametrize(("grid_name", "role"), _solve_role_cases())
def test_a_foreign_canonical_bump_key_is_rejected_for_every_role(
    config: PolicyConfigV2,
    monkeypatch: pytest.MonkeyPatch,
    grid_name: str,
    role: str,
) -> None:
    """A key that is canonical for a base solve, injected into any other role."""
    stage_calls, solver_calls = _sentinels(monkeypatch)
    report = _stage_report(config, fail=True)
    case_index, solve_index = _all_solve_positions(report)[(grid_name, role)]
    published = report["cases"][case_index]["solves"][solve_index]["bumped_prices"]
    foreign = next(
        key for key in v2.CANONICAL_SPOT_BUMP_KEYS if key not in published
    ) if len(published) < len(v2.CANONICAL_SPOT_BUMP_KEYS) else None
    if foreign is None:
        pytest.skip("this role already carries the whole canonical ladder")
    published[foreign] = 1.0
    with pytest.raises(LabelPolicyV2Error, match="key set is wrong") as raised:
        require_confirmation_entry(report, config)
    message = str(raised.value)
    assert f"role '{role}'" in message
    assert foreign in message
    assert "expected=" in message
    assert stage_calls == []
    assert solver_calls == []


@pytest.mark.parametrize("required_key", list(v2.CANONICAL_SPOT_BUMP_KEYS))
@pytest.mark.parametrize("grid_name", [CANDIDATE_GRID, REFERENCE_GRID])
def test_removing_any_required_bump_key_is_rejected(
    config: PolicyConfigV2,
    monkeypatch: pytest.MonkeyPatch,
    grid_name: str,
    required_key: str,
) -> None:
    stage_calls, solver_calls = _sentinels(monkeypatch)
    report = _stage_report(config, fail=True)
    case_index, solve_index = _all_solve_positions(report)[(grid_name, v2.BASE_ROLE)]
    del report["cases"][case_index]["solves"][solve_index]["bumped_prices"][required_key]
    with pytest.raises(LabelPolicyV2Error, match="key set is wrong") as raised:
        require_confirmation_entry(report, config)
    message = str(raised.value)
    assert f"missing=['{required_key}']" in message
    assert f"grid '{grid_name}'" in message
    assert stage_calls == []
    assert solver_calls == []


@pytest.mark.parametrize(
    ("value", "match"),
    [
        pytest.param(True, "finite real number", id="bool"),
        pytest.param(float("inf"), "finite real number", id="infinite"),
        pytest.param("60.0", "finite real number", id="string"),
        pytest.param(None, "finite real number", id="null"),
    ],
)
def test_a_bumped_price_value_must_be_a_finite_real(
    config: PolicyConfigV2, monkeypatch: pytest.MonkeyPatch, value: Any, match: str
) -> None:
    stage_calls, solver_calls = _sentinels(monkeypatch)
    report = _stage_report(config, fail=True)
    case_index, solve_index = _all_solve_positions(report)[(CANDIDATE_GRID, v2.BASE_ROLE)]
    published = report["cases"][case_index]["solves"][solve_index]["bumped_prices"]
    published[v2.CANONICAL_SPOT_BUMP_KEYS[0]] = value
    with pytest.raises(LabelPolicyV2Error, match=match):
        require_confirmation_entry(report, config)
    assert stage_calls == []
    assert solver_calls == []


def test_no_extra_bump_entry_can_reach_surface_solve_from_record(
    config: PolicyConfigV2, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[Any] = []
    original = v2.surface_solve_from_record
    monkeypatch.setattr(
        v2,
        "surface_solve_from_record",
        lambda record: (built.append(record), original(record))[1],
    )
    report = _stage_report(config, fail=True)
    case_index, solve_index = _all_solve_positions(report)[(CANDIDATE_GRID, v2.BASE_ROLE)]
    report["cases"][case_index]["solves"][solve_index]["bumped_prices"]["spot_4_up"] = 1.0
    with pytest.raises(LabelPolicyV2Error):
        v2.recompute_stage(report, config)
    assert built == []


# ---------------------------------------------------------------------------
# MapSpec key-closure audit
# ---------------------------------------------------------------------------


def _map_specs(spec: Any) -> list[tuple[str, Any]]:
    return [
        (path, node)
        for path, node in v2.iter_schema_specs(spec)
        if isinstance(node, v2.MapSpec)
    ]


def test_every_report_map_is_key_constrained_or_contextually_validated() -> None:
    unbounded = [
        path
        for path, node in _map_specs(v2.REPORT_SCHEMA)
        if not node.has_key_constraint and path not in v2.CONTEXTUAL_MAP_PATHS
    ]
    assert unbounded == [], f"MapSpecs with arbitrary string keys: {unbounded}"
    # Every registered contextual path must actually exist in the schema.
    present = {path for path, _ in _map_specs(v2.REPORT_SCHEMA)}
    assert present >= v2.CONTEXTUAL_MAP_PATHS, v2.CONTEXTUAL_MAP_PATHS - present


def test_the_audit_would_have_caught_the_bumped_prices_defect() -> None:
    """The pre-fix spec — an unconstrained MapSpec — must fail the audit."""
    defective = v2.ObjectSpec({"bumped_prices": v2.MapSpec(v2.RealSpec())})
    unbounded = [
        path
        for path, node in _map_specs(defective)
        if not node.has_key_constraint and path not in v2.CONTEXTUAL_MAP_PATHS
    ]
    assert unbounded == [".bumped_prices"]
    # And the fixed spec passes only because it now carries the vocabulary.
    fixed = v2.ObjectSpec(
        {"bumped_prices": v2.MapSpec(v2.RealSpec(), key_enum=v2.CANONICAL_SPOT_BUMP_KEYS)}
    )
    assert all(node.has_key_constraint for _, node in _map_specs(fixed))


def test_the_report_map_classification_is_recorded() -> None:
    classification = {
        ".cases[].solves[].bumped_prices": "context-derived",
        ".cases[].solve_criticality": "context-derived",
        ".cases[].solve_problems": "context-derived",
        ".stage_outcome.failed_checks_by_name": "context-derived",
        ".solve_accounting.attempted_by_solve_role": "context-derived",
        ".solve_accounting.attempted_by_criticality": "context-derived",
        ".solve_accounting.attempted_by_case_classification": "context-derived",
    }
    assert {path for path, _ in _map_specs(v2.REPORT_SCHEMA)} == set(classification)
    # Every report map is now exactly reconciled, not merely key-bounded.
    assert set(classification.values()) == {"context-derived"}
    assert set(classification) == set(v2.EXACTLY_DERIVABLE_MAP_PATHS)


# ---------------------------------------------------------------------------
# Digest and identity fields
# ---------------------------------------------------------------------------

#: Any field whose name says "digest" must be typed as one. The audit below
#: checks the schema against this name-based sweep, so a digest field typed as
#: a plain string is a failure rather than an omission nobody notices.
_DIGEST_LIKE_SUFFIXES = ("_sha256", "_digest")
_DIGEST_LIKE_NAMES = frozenset(
    {
        "pricing_grid_identity",
        "vega_convention_id",
        "expected_vega_convention_id",
    }
)


def _is_digest_like(field: str) -> bool:
    return field.endswith(_DIGEST_LIKE_SUFFIXES) or field in _DIGEST_LIKE_NAMES


def _digest_spec_of(spec: Any) -> Any | None:
    inner = spec.inner if isinstance(spec, v2.NullableSpec) else spec
    if isinstance(inner, v2.StrSpec) and (inner.hex_digest or inner.digest_prefix):
        return inner
    return None


def _schema_digest_fields(schema: Any) -> dict[str, tuple[Any, bool]]:
    """Every digest-like schema position, with its spec and nullability."""
    found: dict[str, tuple[Any, bool]] = {}
    for path, spec in v2.iter_schema_specs(schema):
        field = path.rsplit(".", 1)[-1].removesuffix("[]").removesuffix("{}")
        if not _is_digest_like(field):
            continue
        if isinstance(spec, (v2.ObjectSpec, v2.ArraySpec, v2.MapSpec)):
            continue
        found.setdefault(path, (spec, isinstance(spec, v2.NullableSpec)))
    return found


def test_every_digest_like_report_field_is_typed_as_a_digest() -> None:
    plain = [
        path
        for path, (spec, _) in _schema_digest_fields(v2.REPORT_SCHEMA).items()
        if _digest_spec_of(spec) is None
    ]
    assert plain == [], f"digest-like fields typed as unrestricted strings: {plain}"
    fields = _schema_digest_fields(v2.REPORT_SCHEMA)
    # The three the review named, now constrained rather than free strings.
    for name in (
        "reported_binding_pde_header_sha256",
        "reported_binding_pde_source_sha256",
        "reported_binding_pde_implementation_sha256",
    ):
        spec = _digest_spec_of(fields[f".source.{name}"][0])
        assert spec is not None and spec.hex_digest is True, name
    assert len(fields) >= 20, len(fields)


def _bad_digests(spec: Any) -> list[tuple[str, Any]]:
    prefix = f"{spec.digest_prefix}-" if spec.digest_prefix else ""
    return [
        ("not_a_digest", "not-a-digest"),
        ("sixty_three_hex", prefix + "0" * 63),
        ("sixty_five_hex", prefix + "0" * 65),
        ("uppercase_hex", prefix + "0" * 63 + "A"),
        ("non_hex_characters", prefix + "0" * 63 + "z"),
        ("integer", 7),
    ]


def _digest_field_params(schema: Any) -> list[Any]:
    params: list[Any] = []
    for path, (spec, nullable) in sorted(_schema_digest_fields(schema).items()):
        inner = _digest_spec_of(spec)
        assert inner is not None, path
        cases = _bad_digests(inner)
        if not nullable:
            cases.append(("null", None))
        if inner.digest_prefix:
            cases.append(("wrong_prefix", "other-" + "0" * 64))
        params.extend(
            pytest.param(path, value, id=f"{path}-{label}") for label, value in cases
        )
    return params


def _positions_for(document: Any, normalised: str) -> list[tuple[Any, ...]]:
    """Every concrete path in ``document`` matching a normalised schema path."""
    return [
        path
        for path in _nodes(document)
        if path and _normalise(path) == normalised
    ]


@pytest.mark.parametrize(("path", "value"), _digest_field_params(v2.REPORT_SCHEMA))
def test_every_report_digest_field_rejects_a_malformed_value(
    config: PolicyConfigV2, path: str, value: Any
) -> None:
    report = _stage_report(config, fail=True)
    concrete = _positions_for(report, path)
    assert concrete, path
    _set_at(report, concrete[0], value)
    with pytest.raises(LabelPolicyV2Error, match="expected"):
        v2.validate_report_schema(report)


def test_a_malformed_digest_is_refused_before_run_stage_and_the_solver(
    config: PolicyConfigV2, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_calls, solver_calls = _sentinels(monkeypatch)
    report = _stage_report(config, fail=True)
    report["source"]["reported_binding_pde_header_sha256"] = "not-a-digest"
    with pytest.raises(LabelPolicyV2Error, match="lowercase hex SHA-256 digest"):
        require_confirmation_entry(report, config)
    assert stage_calls == []
    assert solver_calls == []


# ---------------------------------------------------------------------------
# Systematic object-key closure
# ---------------------------------------------------------------------------


def _structural_key(document: Any, path: tuple[Any, ...]) -> str:
    """A path key that keeps role, lifecycle and discriminator variants apart.

    Array positions are *not* collapsed by index where the record's validation
    depends on a discriminator: a solve is keyed by its grid and role, a case
    by its classification and outcome shape, a failure by its criticality.
    """
    parts: list[str] = []
    for depth, part in enumerate(path):
        if isinstance(part, int):
            parent = _read_at(document, tuple(path[:depth]))
            record = parent[part]
            container = parts[-1] if parts else ""
            if container == ".solves" and isinstance(record, dict):
                parts.append(f"[{record['grid_name']}/{record['role']}]")
            elif container == ".cases" and isinstance(record, dict):
                shape = (
                    record["state"]["classification"],
                    record["gate_eligible"],
                    record["anchor_validation"] is not None,
                    bool(record["solve_failures"]),
                )
                parts.append(f"[{'/'.join(map(str, shape))}]")
            elif container == ".solve_failures" and isinstance(record, dict):
                parts.append(f"[{record['criticality']}]")
            elif container == ".grids" and isinstance(record, dict):
                parts.append(f"[{record['name']}]")
            else:
                parts.append("[]")
        else:
            parts.append(f".{part}")
    return "".join(parts)


def _object_positions(document: Any) -> dict[str, tuple[Any, ...]]:
    """Every distinct structural position holding a JSON object."""
    positions: dict[str, tuple[Any, ...]] = {}
    for path in _nodes(document):
        if isinstance(_read_at(document, path), dict):
            positions.setdefault(_structural_key(document, path), path)
    return positions


def _report_variants(config: PolicyConfigV2) -> dict[str, dict[str, Any]]:
    return {
        "remediation_passed": _stage_report(config, stage="remediation", fail=False),
        "remediation_failed": _stage_report(config, stage="remediation", fail=True),
        "confirmation_passed": _stage_report(config, stage="confirmation", fail=False),
        "confirmation_failed": _stage_report(config, stage="confirmation", fail=True),
    }


def _fast_admit(document: Any, config: PolicyConfigV2) -> None:
    v2.validate_report_schema(document)
    v2.validate_report_context(document, config)


def _closure_survivors(
    baseline: dict[str, Any], config: PolicyConfigV2, positions: dict[str, tuple[Any, ...]]
) -> list[str]:
    """Apply add/remove-key mutations; return the ones nothing rejected."""
    survivors: list[str] = []
    for name, path in sorted(positions.items()):
        added = copy.deepcopy(baseline)
        target = _read_at(added, path) if path else added
        target["unexpected_serialized_key"] = 1
        if not _rejected(added, config, f"{name}: added key"):
            survivors.append(f"{name}: added key")
        for key in sorted(_read_at(baseline, path) if path else baseline):
            removed = copy.deepcopy(baseline)
            container = _read_at(removed, path) if path else removed
            del container[key]
            if not _rejected(removed, config, f"{name}: removed {key}"):
                survivors.append(f"{name}: removed {key}")
    return survivors


def _rejected(document: Any, config: PolicyConfigV2, label: str) -> bool:
    del label
    try:
        _fast_admit(document, config)
    except LabelPolicyV2Error:
        return True
    # Anything the schema and the contextual pass admit must still be caught by
    # the full authoritative path, which recomputes every aggregate.
    try:
        verify_stage_report(document, config)
    except LabelPolicyV2Error:
        return True
    return False


REPORT_VARIANTS = (
    "remediation_passed",
    "remediation_failed",
    "confirmation_passed",
    "confirmation_failed",
)


@pytest.mark.parametrize("variant", REPORT_VARIANTS)
def test_object_key_closure_for_every_report_lifecycle_variant(
    config: PolicyConfigV2, variant: str
) -> None:
    baseline = _report_variants(config)[variant]
    _fast_admit(baseline, config)
    positions = _object_positions(baseline)
    assert len(positions) > 40, (variant, len(positions))
    survivors = _closure_survivors(baseline, config, positions)
    assert survivors == [], f"{variant} accepted key mutations: {survivors}"


def test_the_closure_sweep_covers_every_solve_role_variant(
    config: PolicyConfigV2,
) -> None:
    baseline = _stage_report(config, stage="confirmation", fail=True)
    positions = [
        name
        for name in _object_positions(baseline)
        if ".solves[" in name and name.endswith("]")
    ]
    # Solves are keyed inside their case discriminator, so the sweep visits each
    # role once per distinct case shape rather than collapsing them all.
    roles = {name.rsplit(".solves", 1)[1] for name in positions}
    assert len(roles) == 13, sorted(roles)
    assert len(positions) > len(roles), len(positions)
    for grid_name, role in sorted(_all_solve_positions(baseline)):
        assert f"[{grid_name}/{role}]" in roles


# ---------------------------------------------------------------------------
# Exactly derivable maps: raw-record reconciliation before construction
# ---------------------------------------------------------------------------
#
# Bounding a map's key vocabulary still admits a valid-vocabulary key carrying
# a fabricated value, and still admits the removal of a key that had to be
# there. These three maps have exact contents that follow from the versioned
# configuration and the raw solve, failure and check records, so they are
# reconciled exactly at the admission boundary — before any dataclass exists,
# before run_stage, before the solver.


def _construction_sentinels(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[Any], list[Any], list[Any]]:
    """run_stage, the solver, and every solve rebuild must all stay unreached."""
    stage_calls, solver_calls = _sentinels(monkeypatch)
    built: list[Any] = []
    original = v2.surface_solve_from_record

    def recording(record: Mapping[str, Any]) -> v2.SurfaceSolve:
        built.append(record)
        return original(record)

    monkeypatch.setattr(v2, "surface_solve_from_record", recording)
    return stage_calls, solver_calls, built


def _second_case_name(report: dict[str, Any]) -> str:
    return str(report["cases"][1]["state"]["name"])


def _swap_failed_check_value(report: dict[str, Any]) -> None:
    report["stage_outcome"]["failed_checks_by_name"]["price_absolute_error"] = [
        _second_case_name(report)
    ]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        pytest.param(
            lambda r: r["cases"][0]["solve_problems"].pop("grid_1600x800/base"),
            r"solve_problems key set is wrong",
            id="solve_problems_removed_an_actual_problem",
        ),
        pytest.param(
            lambda r: r["cases"][0]["solve_problems"].update(
                {"grid_3200x1600/base": ["solve_absent"]}
            ),
            r"solve_problems key set is wrong",
            id="solve_problems_added_an_in_plan_key_with_no_problem",
        ),
        pytest.param(
            lambda r: r["cases"][0]["solve_problems"].update(
                {"grid_1600x800/base": ["solve_absent"]}
            ),
            r"solve_problems value is wrong",
            id="solve_problems_changed_a_value",
        ),
        pytest.param(
            lambda r: r["cases"][0]["solve_problems"].update(
                {"grid_1600x800/base": ["solver_exception", "non_finite_centre_price"]}
            ),
            r"solve_problems value is wrong",
            id="solve_problems_appended_a_problem",
        ),
        pytest.param(
            lambda r: r["stage_outcome"]["failed_checks_by_name"].update(
                {"spot_convexity": ["regular_euro_negative_rate_call"]}
            ),
            r"failed_checks_by_name key set is wrong",
            id="failed_checks_added_an_allowed_name",
        ),
        pytest.param(
            lambda r: r["stage_outcome"]["failed_checks_by_name"].pop(
                "price_absolute_error"
            ),
            r"failed_checks_by_name key set is wrong",
            id="failed_checks_removed_an_expected_name",
        ),
        pytest.param(
            _swap_failed_check_value,
            r"failed_checks_by_name value is wrong",
            id="failed_checks_changed_a_value",
        ),
        pytest.param(
            lambda r: r["solve_accounting"]["attempted_by_case_classification"].pop(
                "stress"
            ),
            r"attempted_by_case_classification key set is wrong",
            id="classification_counts_removed_a_classification",
        ),
        pytest.param(
            lambda r: r["solve_accounting"]["attempted_by_case_classification"].update(
                {"regular": 104}
            ),
            r"attempted_by_case_classification value is wrong",
            id="classification_counts_changed_a_count",
        ),
    ],
)
def test_a_derivable_map_mutation_is_rejected_before_anything_is_constructed(
    config: PolicyConfigV2, monkeypatch: pytest.MonkeyPatch, mutate: Any, match: str
) -> None:
    stage_calls, solver_calls, built = _construction_sentinels(monkeypatch)
    report = _stage_report(config, fail=True)
    mutate(report)
    with pytest.raises(LabelPolicyV2Error, match=match):
        require_confirmation_entry(report, config)
    assert stage_calls == []
    assert solver_calls == []
    assert built == []


def _regular_only_report(config: PolicyConfigV2) -> dict[str, Any]:
    """A report whose rows are all regular, so 'stress' is legitimately absent.

    Both classifications appear in every real stage, so this is the only way to
    exercise the rule that a classification the stage does not cover is omitted
    rather than published with a zero count.
    """
    report = _stage_report(config, fail=True)
    report["cases"] = [
        row
        for row in report["cases"]
        if config.case(str(row["state"]["name"])).classification == "regular"
    ]
    report["solve_accounting"]["attempted_by_case_classification"] = (
        v2.raw_attempted_by_case_classification(report["cases"], config)
    )
    report["solve_accounting"]["attempted_by_solve_role"] = {
        key: value
        for key, value in report["solve_accounting"]["attempted_by_solve_role"].items()
        if any(key in row["solve_plan"] for row in report["cases"])
    }
    report["solve_accounting"]["attempted_by_criticality"] = {
        key: value
        for key, value in report["solve_accounting"]["attempted_by_criticality"].items()
        if any(key in row["solve_criticality"] for row in report["cases"])
    }
    return report


def test_an_absent_classification_must_be_omitted_not_published_as_zero(
    config: PolicyConfigV2, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_calls, solver_calls, built = _construction_sentinels(monkeypatch)
    report = _regular_only_report(config)
    assert set(report["solve_accounting"]["attempted_by_case_classification"]) == {
        "regular"
    }
    # The omission itself is admissible.
    v2.validate_report_context(report, config)
    # Publishing the uncovered classification as a zero count is not.
    report["solve_accounting"]["attempted_by_case_classification"]["stress"] = 0
    with pytest.raises(
        LabelPolicyV2Error, match=r"attempted_by_case_classification key set is wrong"
    ):
        v2.validate_report_context(report, config)
    assert stage_calls == []
    assert solver_calls == []
    assert built == []


def test_a_problem_free_solve_must_not_be_published_with_an_empty_problem_list(
    config: PolicyConfigV2, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_calls, solver_calls, built = _construction_sentinels(monkeypatch)
    report = _stage_report(config, fail=True)
    report["cases"][0]["solve_problems"]["grid_3200x1600/base"] = []
    with pytest.raises(LabelPolicyV2Error, match=r"solve_problems key set is wrong"):
        require_confirmation_entry(report, config)
    assert stage_calls == []
    assert solver_calls == []
    assert built == []


@pytest.mark.parametrize("variant", REPORT_VARIANTS)
def test_the_raw_helpers_equal_the_recomputed_aggregates(
    config: PolicyConfigV2, variant: str
) -> None:
    """One derivation, used twice: admission and recomputation must agree."""
    report = _report_variants(config)[variant]
    recomputed = v2.recompute_stage(report, config)
    rows = report["cases"]
    assert v2.raw_failed_checks_by_name(rows, config) == (
        recomputed.outcome["failed_checks_by_name"]
    )
    assert v2.raw_attempted_by_case_classification(rows, config) == (
        recomputed.accounting["attempted_by_case_classification"]
    )
    assert len(recomputed.cases) == len(rows)
    for row, rebuilt in zip(rows, recomputed.cases, strict=True):
        case = config.case(str(row["state"]["name"]))
        records = {
            (str(solve["grid_name"]), str(solve["role"])): solve
            for solve in row["solves"]
        }
        failed = {
            (str(entry["grid"]), str(entry["role"])) for entry in row["solve_failures"]
        }
        derived = v2.raw_solve_problems(case, config, records, failed)
        assert derived == rebuilt["solve_problems"]
        assert derived == row["solve_problems"]


def test_every_exactly_derivable_map_is_contextually_reconciled() -> None:
    """An enum or pattern key constraint alone is not sufficient for these."""
    assert v2.EXACTLY_DERIVABLE_MAP_PATHS <= v2.CONTEXTUAL_MAP_PATHS
    present = {path for path, _ in _map_specs(v2.REPORT_SCHEMA)}
    assert present == v2.EXACTLY_DERIVABLE_MAP_PATHS
    merely_bounded = [
        path
        for path, node in _map_specs(v2.REPORT_SCHEMA)
        if path in v2.EXACTLY_DERIVABLE_MAP_PATHS
        and node.has_key_constraint
        and path not in v2.CONTEXTUAL_MAP_PATHS
    ]
    assert merely_bounded == [], merely_bounded
