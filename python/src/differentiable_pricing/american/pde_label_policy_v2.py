"""Task 9C-C3 PDE label-policy v2: predeclared criteria, lifecycle and runner.

Task 9C-B (v1) met every predeclared absolute-error cap at 1600x800 and still
refused to select a policy, on two kinds of check that are not error caps:
bump-ladder movement of the *reference* delta/vega, and one American-dominance
violation the size of the solver's own accumulated PSOR residual. This module
implements the separately predeclared v2 answer to both, plus the one-shot
remediation/confirmation lifecycle the active task specifies.

Everything decision-making here is a pure function of already-measured
numbers, so the whole criterion is exercised by cheap synthetic and analytic
tests. Only :func:`collect_case_solves` calls the compiled engine, and only the
manual CLI calls that.

Three structural properties are worth naming up front.

* **Role-aware decisions.** Each solve is classified price-critical,
  delta-only, vega-only, anchor-descriptive or stress-descriptive, and a
  failure only ever propagates to the decisions its role supports. A
  Greek-specific failure can never veto an otherwise valid price policy.
* **Failures are data, not control flow.** A solver exception is caught per
  case and per solve role, recorded structurally, and the stage continues, so
  a complete report is always emitted.
* **One derivation writes and re-verifies.** Every published case block is
  rebuilt from the published raw solve records by :func:`recompute_stage`,
  which both the confirmation-entry validator and the freeze tool use. A
  published summary is never compared against itself.

Nothing in this module is a dataset, a training input, or an approved label
policy. ``AUTHORIZED_TRAINING_INPUT_STATUSES`` is empty and task 9C-C3
authorizes no dataset generation whatever its outcome.

Read alongside ``docs/pde-numerical-contract.md`` ("Task 9C-C3: label-policy
v2") and ``docs/tasks/active/task-9c-c3-label-policy-v2.md``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
import platform
import re
import resource
import sys
import tempfile
import time
import tomllib
from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from differentiable_pricing import _pde, pde_valuation_surface
from differentiable_pricing.american.pde_surface_harvest import canonical_payload

SCHEMA_VERSION: Final = "pde-label-policy-pilot-v2/1"
REPORT_SCHEMA_VERSION: Final = "pde-label-policy-v2-report/1"
CRITERIA_SCHEME_VERSION: Final = "pde-label-policy-v2-criteria/1"
IDENTITY_SCHEME_VERSION: Final = "pde-label-policy-v2-identity/1"
VEGA_CONVENTION_SCHEME_VERSION: Final = "pde-label-policy-v2-vega-convention/1"
PROVENANCE_SCHEME_VERSION: Final = "pde-label-policy-v2-provenance/1"

ENGINE_NAME: Final = "dp::finite_difference_price/v1"
SURFACE_ENGINE_NAME: Final = "dp::finite_difference_valuation_surface/v1"

#: Canonical protocol strings. These are pinned, not descriptive prose: the
#: parser requires each one exactly, the report is built from these constants,
#: and report verification compares against them.
STUDY_NAME: Final = "pde-label-policy-pilot-v2"
DELTA_METHOD: Final = "grid_stencil_primary_with_fixed_bump_validation"
VEGA_METHOD: Final = "centered_volatility_bump_from_separate_surfaces"

#: The frozen v2 price-reference rule. The price reference is the **raw**
#: exact-node centre value of the 3200x1600 base surface: unconditional, never
#: Richardson-extrapolated, with no conditional fallback. Richardson survives
#: only as a validation tool for the E2 grid-stencil delta and for the
#: explicitly descriptive anchor diagnostics.
PRICE_REFERENCE_METHOD: Final = "raw_grid_3200x1600_center"

PROJECT_ROOT: Final = Path(__file__).resolve().parents[4]
_SHA256_TEXT: Final = re.compile(r"^[0-9a-f]{64}$")
#: `<grid>/<role>` keys, the only shape a solve-role map key may take.
_SOLVE_ROLE_KEY: Final = re.compile(r"^grid_\d+x\d+/[A-Za-z0-9_.]+$")

ORDER_PROBE_GRID: Final = "grid_800x400"
CANDIDATE_GRID: Final = "grid_1600x800"
REFERENCE_GRID: Final = "grid_3200x1600"
ANCHOR_GRID: Final = "grid_6400x3200"
GRID_NAMES: Final = (ORDER_PROBE_GRID, CANDIDATE_GRID, REFERENCE_GRID, ANCHOR_GRID)
GRID_ROLES: Final = ("order_probe", "candidate", "reference", "anchor")
GRID_DIMENSIONS: Final = ((800, 400), (1600, 800), (3200, 1600), (6400, 3200))

NO_POLICY_SELECTED: Final = "no_policy_selected"
NON_GATING_REMEDIATION_CASE: Final = "stress_american_put_exercise_boundary"

#: The five v1 grid_1600x800 regular failures, from the frozen v1 snapshot.
V1_REGULAR_FAILURES: Final = (
    "regular_euro_negative_rate_call",
    "regular_euro_high_rate_call",
    "regular_american_put_early_exercise",
    "regular_american_put_negative_rate_control",
    "regular_american_one_dividend_call",
)
#: The three smooth regression controls, confirmed from the task spec.
SMOOTH_CONTROL_CASES: Final = (
    "regular_euro_atm_put",
    "regular_euro_deep_itm_call_long_high_vol",
    "regular_euro_deep_otm_put_long",
)
ANCHOR_CASES: Final = ("regular_euro_atm_call", NON_GATING_REMEDIATION_CASE)
REMEDIATION_CASES: Final = (
    *V1_REGULAR_FAILURES,
    *SMOOTH_CONTROL_CASES,
    "regular_euro_atm_call",
    NON_GATING_REMEDIATION_CASE,
)
#: Exactly the nine of the ten that decide pass/fail.
REMEDIATION_GATE_CASES: Final = REMEDIATION_CASES[:-1]
REMEDIATION_CASE_COUNT: Final = 10
GATING_REMEDIATION_CASE_COUNT: Final = 9
CONFIRMATION_CASE_COUNT: Final = 28

#: Digest of the 28 case states, in order. Pinned so a replaced anchor, a
#: reordered case or an edited case state is rejected at parse time.
CASE_DESIGN_DIGEST: Final = "cases-c2d4ca8210cebd2c951c87052b22f519daf5d14bd1875e56802a9d04fcc3d91d"

#: ``1 / (1 - 2**-1.5)``. The bias charge multiplies the flat-branch tolerance
#: or the resolved-branch increment by this factor to bound the *remaining*
#: extrapolation tail of a geometric ladder whose worst supported order is
#: 1.5, so a criterion comparing only consecutive rungs cannot understate the
#: residual bias.
FIXED_BUMP_CHARGE_FACTOR: Final = 1.0 / (1.0 - 2.0**-1.5)
FIXED_BUMP_CHARGE_SCHEME: Final = "one_over_one_minus_two_to_minus_three_halves/1"
DELTA_FLAT_EPSILON_RULE: Final = "small=3E/(2h), large=3E/(4h), E=M_base*R_base"
VEGA_FLAT_EPSILON_RULE: Final = (
    "small=err(h)+err(2h), large=err(2h)+err(4h), "
    "err(b)=(M_down_b*R_down_b+M_up_b*R_up_b)/(2b)"
)
GRID_STENCIL_REFERENCE_RULE: Final = (
    "grid_richardson_of_candidate_and_fine_stencil_deltas"
)
GRID_STENCIL_ORDER_RULE: Final = (
    "observed_order_from_order_probe_candidate_and_reference"
)
SUPPORTED_ORDER_BAND: Final = (1.5, 2.5)

#: Empty by construction. Task 9C-C3 decides a label policy; it authorizes no
#: dataset and no training input, whichever way it resolves.
AUTHORIZED_TRAINING_INPUT_STATUSES: Final[frozenset[str]] = frozenset()

BASE_ROLE: Final = "base"
DOMINANCE_CONTROL_ROLE: Final = "european_dominance_control"

PRICE_CRITICAL: Final = "price_critical"
DELTA_ONLY: Final = "delta_only"
VEGA_ONLY: Final = "vega_only"
ANCHOR_DESCRIPTIVE: Final = "anchor_descriptive"
STRESS_DESCRIPTIVE: Final = "stress_descriptive"
SOLVE_CRITICALITIES: Final = (
    PRICE_CRITICAL,
    DELTA_ONLY,
    VEGA_ONLY,
    ANCHOR_DESCRIPTIVE,
    STRESS_DESCRIPTIVE,
)

CLASSIFICATIONS: Final = frozenset({"regular", "stress"})
OPTION_TYPES: Final = frozenset({"call", "put"})
EXERCISE_STYLES: Final = frozenset({"european", "american"})

STAGES: Final = ("remediation", "confirmation")
REMEDIATION_STATUSES: Final = ("passed", "failed")
CONFIRMATION_STATUSES: Final = ("not_run", "pending", "run")

#: Fields that define the pricing/grid identity a v2 result must reproduce.
#: ``contract_multiplier`` is absent deliberately: it never enters pricing
#: arithmetic and stays reporting-only. ``settlement`` is present: it names a
#: genuinely different contract term.
PRICING_IDENTITY_FIELDS: Final = (
    "option_type",
    "exercise_style",
    "strike",
    "valuation_time",
    "expiry_time",
    "volatility",
    "continuous_carry",
    "curve_times",
    "curve_log_discounts",
    "dividends",
    "settlement",
)
GRID_IDENTITY_FIELDS: Final = (
    "spot_intervals",
    "time_steps",
    "spot_maximum",
    "rannacher_steps",
    "psor_tolerance",
    "psor_relaxation",
    "psor_maximum_iterations",
    "boundary_exclusion_nodes",
)
#: Present in the surface echo, never in the identity.
REPORTING_ONLY_ECHO_FIELDS: Final = ("contract_multiplier", "dividends_declared")

#: Every gating check is scoped to price-critical solves and price shape.
GATING_CHECKS: Final = (
    "price_critical_solves_returned_and_converged",
    "price_critical_exact_node_identity",
    "price_critical_pricing_and_grid_identity",
    "price_critical_residual_scale_within_price_cap",
    "price_absolute_error",
    "american_dominance",
    "american_intrinsic_bound",
    "spot_monotonicity",
    "spot_convexity",
)
NON_GATING_PUBLISHED_CHECKS: Final = (
    "gamma_absolute_error",
    "fixed_bump_delta_validation",
    "fixed_bump_vega_validation",
    "grid_stencil_delta_validation",
    "anchor_rung_evidence",
)
COMMON_ELIGIBILITY_PREDICATES: Final = (
    "exact_pricing_and_grid_identity",
    "exact_solver_settings_identity",
    "exact_node_no_interpolation",
    "structural_c1_greek_eligibility",
    "discrete_system_converged",
    "surface_price_residual_scales_within_price_cap",
)
STRUCTURAL_C1_APPLIES_TO: Final = ("delta", "vega")
DELTA_ELIGIBILITY_PREDICATES: Final = (
    "delta_supporting_solves_available",
    "delta_residual_scale_within_delta_cap",
    "fixed_bump_delta_validation_passed",
    "fixed_bump_delta_residual_scale_within_delta_cap",
    "grid_stencil_delta_validation_passed",
)
VEGA_ELIGIBILITY_PREDICATES: Final = (
    "vega_supporting_solves_available",
    "vega_residual_scale_within_vega_cap",
    "vega_numerically_available",
    "vega_bump_exercise_states_equal",
    "exact_vega_convention_and_bump_identity",
    "fixed_bump_vega_validation_passed",
    "fixed_bump_vega_residual_scale_within_vega_cap",
)

PRICE_CRITICAL_SOLVES: Final = (
    f"{CANDIDATE_GRID}/{BASE_ROLE}",
    f"{REFERENCE_GRID}/{BASE_ROLE}",
    f"{CANDIDATE_GRID}/{DOMINANCE_CONTROL_ROLE}",
)
DELTA_ONLY_SOLVES: Final = (f"{ORDER_PROBE_GRID}/{BASE_ROLE}",)
VEGA_ONLY_SOLVES: Final = (
    f"{CANDIDATE_GRID}/volatility_*",
    f"{REFERENCE_GRID}/volatility_*",
)
ANCHOR_DESCRIPTIVE_SOLVES: Final = (f"{ANCHOR_GRID}/{BASE_ROLE}",)

#: Executable sources the runner and the freeze path rely on, relative to the
#: repository root. Every one is hashed into the report's provenance block and
#: reconciled by both the confirmation-entry validator and the freeze tool.
#:
#: ``v1_label_policy_module`` is recorded even though v2 does not import it:
#: v2's formulas are deliberately written against that revision of the v1
#: study, and a reader must be able to see which revision that was.
EXECUTABLE_SOURCE_INVENTORY: Final = {
    "v2_runner_module": "python/src/differentiable_pricing/american/pde_label_policy_v2.py",
    "v2_freeze_script": "scripts/freeze_pde_label_policy_v2_results.py",
    "v1_label_policy_module": "python/src/differentiable_pricing/american/pde_label_policy.py",
    "canonical_payload_module": (
        "python/src/differentiable_pricing/american/pde_surface_harvest.py"
    ),
    "package_init_module": "python/src/differentiable_pricing/__init__.py",
    "pde_header": "cpp/include/dp/finite_difference_pde.hpp",
    "pde_implementation": "cpp/src/finite_difference_pde.cpp",
    "pde_binding_source": "bindings/python/pde_module.cpp",
    # `dp/option.hpp` is included by both the PDE header and the binding, and
    # `cpp/src/option.cpp` defines the parse/payoff symbols they call. Walking
    # the include graph of the PDE path gives exactly this closure: the
    # binding TU, the PDE header and implementation, and these two.
    "option_header": "cpp/include/dp/option.hpp",
    "option_implementation": "cpp/src/option.cpp",
}

#: What the include-graph walk found, recorded so a later reader can check the
#: claim rather than re-derive it. `_pde` links the whole `dp_core` archive,
#: which also contains `binomial_tree.cpp`, `black_scholes.cpp`,
#: `least_squares_monte_carlo.cpp` and `smooth_mlp.cpp`. None of those is
#: reachable from the PDE path's include graph and the PDE path calls no symbol
#: they define, so they are deliberately **not** in the inventory. `CMakeLists.txt`
#: selects what is compiled but is build definition rather than executable
#: source, and is likewise not inventoried; that is a stated limit, not an
#: oversight.
PDE_INCLUDE_CLOSURE: Final = (
    "bindings/python/pde_module.cpp",
    "cpp/include/dp/finite_difference_pde.hpp",
    "cpp/src/finite_difference_pde.cpp",
    "cpp/include/dp/option.hpp",
    "cpp/src/option.cpp",
)

#: Stated rather than glossed: hashing these files records which sources were
#: present in the checkout. It does **not** prove the loaded ``_pde`` extension
#: was compiled from them. No such proof exists in this repository today; the
#: binding exposes its own build-time digests, which are recorded separately
#: and are themselves self-reported by the build.
SOURCE_DIGEST_LIMITATION: Final = (
    "Executable-source digests record the repository files present when the "
    "stage ran. They do not prove the loaded _pde extension binary was built "
    "from those sources; no such proof exists here. The binding's own "
    "build-time digests are recorded alongside them and are self-reported by "
    "the build."
)

#: Published verbatim in every v2 report.
NON_CLAIMS: Final = (
    "A validation reference error and an observed convergence order are "
    "available only for a case this study solved on its own reference and "
    "order-probe rungs. They are unavailable per production row, so a "
    "production row can never carry its own measured reference error or "
    "observed order.",
    "A production row is priced with one volatility bump. A one-bump row "
    "cannot re-estimate its own vega bump bias; the bias charge is estimated "
    "here, on this evidence set, and is not recomputable downstream.",
    "The 28 evidence cases are isolated points. They do not validate the "
    "surrounding hyperrectangle of parameters, and no interpolation between "
    "them is licensed by this study.",
    "Task 9C-C2b2 must separately predeclare its own generation domain. It "
    "does not inherit one from this task's case list.",
    "No dataset and no training input is authorized by task 9C-C3, whichever "
    "way it resolves. AUTHORIZED_TRAINING_INPUT_STATUSES is empty.",
    "The American dominance and intrinsic allowances are operational "
    "price-error scale estimates built from the solver's own accumulated LCP "
    "residual. is_a_rigorous_bound is false; they are not certified bounds.",
    "Gamma is evaluation-only. Its absolute error is published and never "
    "gates, and gamma is never supervision-eligible in this task.",
    "Richardson extrapolation is a reference and validation technique here, "
    "never a candidate label policy (decision log DEC-010).",
    "Semantic verification of a stage report recomputes every published "
    "decision from the published raw numbers. It detects an inconsistent or "
    "partial mutation. Without a signature or a re-solve it cannot "
    "authenticate a fully coordinated fabricated numerical report, and "
    "neither the confirmation-entry validator nor the freeze tool re-solves.",
    "One-shot status is procedural, provenance-backed and independently "
    "reviewed. The nonempty-output refusal and the snapshot-overwrite refusal "
    "are best effort: a second run in another directory or another clone "
    "cannot be detected from inside this runner.",
    SOURCE_DIGEST_LIMITATION,
)

_TOP_LEVEL_KEYS: Final = frozenset(
    {
        "schema_version",
        "study",
        "solver",
        "grids",
        "bumps",
        "richardson",
        "criteria",
        "gating",
        "eligibility",
        "dominance",
        "lifecycle",
        "cases",
    }
)
_STUDY_KEYS: Final = frozenset(
    {
        "name",
        "engine",
        "surface_engine",
        "price_units",
        "delta_method",
        "vega_method",
        "price_reference_method",
        "curve_construction",
        "selection_order",
        "order_probe_grids",
        "anchor_cases",
        "confirmation_case_count",
        "case_design_digest",
        "remediation_cases",
        "gate_cases",
        "non_gating_remediation_cases",
        "v1_regular_failures",
        "smooth_control_cases",
    }
)
_SOLVER_KEYS: Final = frozenset(
    {
        "spot_maximum",
        "rannacher_steps",
        "psor_tolerance",
        "psor_relaxation",
        "psor_maximum_iterations",
        "settlement",
        "contract_multiplier",
        "boundary_exclusion_nodes",
    }
)
_GRID_KEYS: Final = frozenset({"name", "spot_intervals", "time_steps", "role"})
_BUMP_KEYS: Final = frozenset(
    {"spot", "primary_spot", "volatility", "primary_volatility", "ladder_ratio"}
)
_RICHARDSON_KEYS: Final = frozenset(
    {
        "role",
        "coarse_to_fine_ratio",
        "assumed_order",
        "minimum_supported_observed_order",
        "maximum_supported_observed_order",
    }
)
_CRITERIA_FLOAT_KEYS: Final = (
    "price_absolute_error",
    "delta_absolute_error",
    "gamma_absolute_error",
    "vega_absolute_error_per_unit_volatility",
    "shape_absolute_floor",
    "minimum_supported_bump_order",
    "maximum_supported_bump_order",
    "minimum_supported_stencil_order",
    "maximum_supported_stencil_order",
)
_CRITERIA_KEYS: Final = frozenset(
    {
        *_CRITERIA_FLOAT_KEYS,
        "gamma_is_evaluation_only",
        "fixed_bump_charge_scheme",
        "delta_flat_epsilon_rule",
        "vega_flat_epsilon_rule",
        "grid_stencil_reference_rule",
        "grid_stencil_order_rule",
        "grid_stencil_bump_bias_charge_applied",
    }
)
_GATING_KEYS: Final = frozenset(
    {
        "gating_checks",
        "non_gating_published_checks",
        "price_critical_solves",
        "delta_only_solves",
        "vega_only_solves",
        "anchor_descriptive_solves",
        "regular_case_price_critical_failure_fails_candidate",
        "anchor_rung_failure_is_descriptive",
        "stress_case_failure_is_descriptive",
    }
)
_ELIGIBILITY_KEYS: Final = frozenset(
    {
        "common_predicates",
        "structural_c1_applies_to",
        "delta_predicates",
        "vega_predicates",
        "gamma_eligibility",
        "identity_excludes_contract_multiplier",
        "identity_includes_settlement",
    }
)
_DOMINANCE_KEYS: Final = frozenset({"control", "residual_measure", "is_a_rigorous_bound"})
_LIFECYCLE_KEYS: Final = frozenset(
    {
        "stages",
        "remediation_statuses",
        "confirmation_statuses",
        "selection_requires_fresh_top_level_approval",
        "authorizes_dataset_generation",
        "authorizes_training_input",
        "one_shot_enforcement",
    }
)
_CASE_KEYS: Final = frozenset(
    {
        "name",
        "classification",
        "description",
        "option_type",
        "exercise_style",
        "spot",
        "strike",
        "expiry_time",
        "rate",
        "continuous_carry",
        "volatility",
        "dividends",
    }
)

#: The four v1 absolute-error caps, carried over unchanged and deliberately.
UNCHANGED_V1_CAPS: Final = {
    "price_absolute_error": 5.0e-4,
    "delta_absolute_error": 1.0e-3,
    "gamma_absolute_error": 2.0e-4,
    "vega_absolute_error_per_unit_volatility": 5.0e-2,
}
#: Every other fixed threshold and convention v2 pins, beyond those four caps.
PINNED_CRITERIA: Final = {
    "shape_absolute_floor": 1.0e-8,
    "minimum_supported_bump_order": SUPPORTED_ORDER_BAND[0],
    "maximum_supported_bump_order": SUPPORTED_ORDER_BAND[1],
    "minimum_supported_stencil_order": SUPPORTED_ORDER_BAND[0],
    "maximum_supported_stencil_order": SUPPORTED_ORDER_BAND[1],
    "gamma_is_evaluation_only": True,
    "fixed_bump_charge_scheme": FIXED_BUMP_CHARGE_SCHEME,
    "delta_flat_epsilon_rule": DELTA_FLAT_EPSILON_RULE,
    "vega_flat_epsilon_rule": VEGA_FLAT_EPSILON_RULE,
    "grid_stencil_reference_rule": GRID_STENCIL_REFERENCE_RULE,
    "grid_stencil_order_rule": GRID_STENCIL_ORDER_RULE,
    "grid_stencil_bump_bias_charge_applied": False,
}
#: Solver settings and the PSOR iteration ceiling are part of the
#: predeclaration too: changing the residual rule changes every scale.
PINNED_SOLVER: Final = {
    "spot_maximum": 400.0,
    "rannacher_steps": 2,
    "psor_tolerance": 1.0e-11,
    "psor_relaxation": 1.2,
    "psor_maximum_iterations": 200000,
    "settlement": "cash",
    "boundary_exclusion_nodes": 2,
}
#: Every spot-bump price key the protocol can ever produce, derived from the
#: pinned bump ladder. The static schema restricts `bumped_prices` to exactly
#: this vocabulary; the contextual validator then requires the exact subset a
#: given solve role must carry.
CANONICAL_SPOT_BUMP_KEYS: Final = (
    "spot_0.5_down",
    "spot_0.5_up",
    "spot_1_down",
    "spot_1_up",
    "spot_2_down",
    "spot_2_up",
)

PINNED_BUMPS: Final = {
    "spot": (0.5, 1.0, 2.0),
    "primary_spot": 0.5,
    "volatility": (0.005, 0.01, 0.02),
    "primary_volatility": 0.005,
    "ladder_ratio": 2,
}


class LabelPolicyV2Error(RuntimeError):
    """Raised when the v2 contract, lifecycle, execution, or publication fails."""


# ---------------------------------------------------------------------------
# Executable-source provenance
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise LabelPolicyV2Error(f"cannot hash executable source '{path}': {error}") from error


def executable_source_digests(*, root: Path = PROJECT_ROOT) -> dict[str, str]:
    """Digest every executable source the v2 runner and freeze path rely on.

    The v2 study is a source-checkout study: a missing inventory file is an
    error rather than a silently thinner provenance block.
    """
    digests: dict[str, str] = {}
    for name, relative in sorted(EXECUTABLE_SOURCE_INVENTORY.items()):
        path = root / relative
        if not path.is_file():
            raise LabelPolicyV2Error(
                f"executable source '{relative}' is missing; the v2 study must run "
                "from a source checkout so its provenance is complete"
            )
        digests[f"{name}_sha256"] = _sha256_file(path)
    return digests


def executable_source_composite_digest(digests: Mapping[str, str]) -> str:
    """One digest over the whole inventory, so a single field pins all of it."""
    expected = {f"{name}_sha256" for name in EXECUTABLE_SOURCE_INVENTORY}
    if set(digests) != expected:
        missing = sorted(expected - set(digests))
        unknown = sorted(set(digests) - expected)
        raise LabelPolicyV2Error(
            "executable-source inventory mismatch; "
            f"missing={missing} unknown={unknown}"
        )
    payload = {
        "scheme": PROVENANCE_SCHEME_VERSION,
        "inventory": {name: EXECUTABLE_SOURCE_INVENTORY[name] for name in sorted(
            EXECUTABLE_SOURCE_INVENTORY
        )},
        "digests": dict(sorted(digests.items())),
    }
    return _digest("src", payload)


def verify_executable_source_digests(
    recorded: Mapping[str, Any], *, root: Path = PROJECT_ROOT, where: str
) -> dict[str, str]:
    """Reconcile a recorded inventory against the current repository files."""
    actual = executable_source_digests(root=root)
    for name, digest in sorted(actual.items()):
        if recorded.get(name) != digest:
            raise LabelPolicyV2Error(
                f"{where}.{name} does not match the repository: recorded "
                f"{recorded.get(name)!r}, file hashes to {digest}"
            )
    composite = executable_source_composite_digest(actual)
    if recorded.get("executable_source_composite_digest") != composite:
        raise LabelPolicyV2Error(
            f"{where}.executable_source_composite_digest does not match the repository"
        )
    return actual



# ---------------------------------------------------------------------------
# Strict JSON loading
# ---------------------------------------------------------------------------


def _duplicate_key_paths(document: Any, offenders: Mapping[int, Sequence[str]]) -> list[str]:
    """Walk a parsed document top-down to locate the offending objects by identity."""
    located: list[str] = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, dict):
            keys = offenders.get(id(value))
            if keys is not None:
                located.extend(f"{path or '$'}.{key}" for key in keys)
            for key in value:
                walk(value[key], f"{path}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    walk(document, "")
    return located


def strict_json_loads(text: str, *, description: str) -> Any:
    """Parse JSON, refusing duplicate object keys and non-standard constants.

    ``json.loads`` silently keeps the **last** value of a repeated key, so a
    duplicate is a silent data loss: by the time anything downstream — including
    canonical re-serialisation — looks at the document, one of the two values
    is already gone. Detection therefore has to happen inside the parser.

    ``object_pairs_hook`` sees the raw key/value pairs of every object at every
    nesting depth, including objects inside arrays, so nothing escapes. The
    offending objects are remembered by identity and their paths are recovered
    by one top-down walk afterwards, so the error names the duplicate key *and*
    where it sits.

    ``NaN``, ``Infinity`` and ``-Infinity`` are Python extensions to JSON, not
    JSON. They are refused rather than accepted as floats.
    """
    offenders: dict[int, list[str]] = {}
    retained: list[dict[str, Any]] = []

    def object_pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: set[str] = set()
        duplicates: list[str] = []
        for key, _ in pairs:
            if key in seen and key not in duplicates:
                duplicates.append(key)
            seen.add(key)
        obj = dict(pairs)
        if duplicates:
            # Keep a reference so the id() stays valid for the path walk.
            retained.append(obj)
            offenders[id(obj)] = duplicates
        return obj

    def parse_constant(name: str) -> Any:
        raise LabelPolicyV2Error(
            f"{description} contains the non-standard JSON constant '{name}'; "
            "NaN, Infinity and -Infinity are not valid JSON and are refused"
        )

    try:
        document = json.loads(
            text, object_pairs_hook=object_pairs_hook, parse_constant=parse_constant
        )
    except json.JSONDecodeError as error:
        raise LabelPolicyV2Error(f"{description} is not valid JSON: {error}") from error
    if offenders:
        paths = _duplicate_key_paths(document, offenders)
        if not paths:  # pragma: no cover - defensive; identity walk always finds them
            paths = sorted({key for keys in offenders.values() for key in keys})
        raise LabelPolicyV2Error(
            f"{description} repeats object keys, which JSON parsing would silently "
            f"collapse: {', '.join(sorted(paths))}"
        )
    return document


def load_strict_json(path: Path, *, description: str) -> Any:
    """Read and strictly parse one JSON file on the task 9C-C3 evidence path."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise LabelPolicyV2Error(f"cannot read {description} '{path}': {error}") from error
    return strict_json_loads(text, description=f"{description} '{path}'")


# ---------------------------------------------------------------------------
# Exact recursive type schemas
# ---------------------------------------------------------------------------
#
# Everything below validates *serialized* JSON, before any dataclass is
# constructed. `bool` is a subclass of `int` in Python, so every rule uses
# `type(value) is ...` rather than `isinstance`, and no value is ever coerced.


def _schema_error(path: str, expected: str, value: Any) -> LabelPolicyV2Error:
    return LabelPolicyV2Error(
        f"{path or '$'}: expected {expected}, observed "
        f"{type(value).__name__} {value!r}"
    )


@dataclass(frozen=True, slots=True)
class BoolSpec:
    """Exactly ``True`` or ``False``; ``1`` and ``0`` are rejected."""

    def validate(self, value: Any, path: str) -> None:
        if type(value) is not bool:
            raise _schema_error(path, "boolean", value)


@dataclass(frozen=True, slots=True)
class IntSpec:
    minimum: int | None = None
    maximum: int | None = None

    def validate(self, value: Any, path: str) -> None:
        if type(value) is not int:
            raise _schema_error(path, "integer", value)
        if self.minimum is not None and value < self.minimum:
            raise _schema_error(path, f"integer >= {self.minimum}", value)
        if self.maximum is not None and value > self.maximum:
            raise _schema_error(path, f"integer <= {self.maximum}", value)


@dataclass(frozen=True, slots=True)
class RealSpec:
    minimum: float | None = None
    maximum: float | None = None

    def validate(self, value: Any, path: str) -> None:
        if type(value) not in (int, float):
            raise _schema_error(path, "finite real number", value)
        number = float(value)
        if not math.isfinite(number):
            raise _schema_error(path, "finite real number", value)
        if self.minimum is not None and number < self.minimum:
            raise _schema_error(path, f"real >= {self.minimum}", value)
        if self.maximum is not None and number > self.maximum:
            raise _schema_error(path, f"real <= {self.maximum}", value)


@dataclass(frozen=True, slots=True)
class StrSpec:
    enum: tuple[str, ...] | None = None
    allow_empty: bool = False
    hex_digest: bool = False
    digest_prefix: str | None = None

    def validate(self, value: Any, path: str) -> None:
        if type(value) is not str:
            raise _schema_error(path, "string", value)
        if not value and not self.allow_empty:
            raise _schema_error(path, "non-empty string", value)
        if self.enum is not None and value not in self.enum:
            raise _schema_error(path, f"one of {list(self.enum)}", value)
        if self.hex_digest and not _SHA256_TEXT.fullmatch(value):
            raise _schema_error(path, "lowercase hex SHA-256 digest", value)
        if self.digest_prefix is not None:
            head, separator, tail = value.partition("-")
            if separator != "-" or head != self.digest_prefix or not _SHA256_TEXT.fullmatch(tail):
                raise _schema_error(
                    path,
                    f"'{self.digest_prefix}-' followed by a lowercase hex SHA-256 digest",
                    value,
                )


@dataclass(frozen=True, slots=True)
class ArraySpec:
    item: Any
    minimum_length: int = 0

    def validate(self, value: Any, path: str) -> None:
        if type(value) is not list:
            raise _schema_error(path, "array", value)
        if len(value) < self.minimum_length:
            raise _schema_error(path, f"array of at least {self.minimum_length}", value)
        for index, item in enumerate(value):
            self.item.validate(item, f"{path}[{index}]")


@dataclass(frozen=True, slots=True)
class ObjectSpec:
    fields: Mapping[str, Any]

    def validate(self, value: Any, path: str) -> None:
        if type(value) is not dict:
            raise _schema_error(path, "object", value)
        missing = sorted(set(self.fields) - set(value))
        unknown = sorted(set(value) - set(self.fields))
        if missing or unknown:
            raise LabelPolicyV2Error(
                f"{path or '$'}: object key set is wrong; missing={missing} "
                f"unknown={unknown}"
            )
        for key, spec in self.fields.items():
            spec.validate(value[key], f"{path}.{key}")


@dataclass(frozen=True, slots=True)
class MapSpec:
    """An object whose keys are open but whose values share one exact type."""

    value_spec: Any
    key_enum: tuple[str, ...] | None = None
    key_pattern: Any = None

    def validate(self, value: Any, path: str) -> None:
        if type(value) is not dict:
            raise _schema_error(path, "object", value)
        for key in value:
            if type(key) is not str or not key:
                raise _schema_error(f"{path}.<key>", "non-empty string key", key)
            if self.key_enum is not None and key not in self.key_enum:
                raise _schema_error(f"{path}.{key}", f"key in {list(self.key_enum)}", key)
            if self.key_pattern is not None and not self.key_pattern.fullmatch(key):
                raise _schema_error(
                    f"{path}.{key}", f"key matching {self.key_pattern.pattern}", key
                )
            self.value_spec.validate(value[key], f"{path}.{key}")

    @property
    def has_key_constraint(self) -> bool:
        """Whether the static schema alone bounds this map's key set."""
        return self.key_enum is not None or self.key_pattern is not None


@dataclass(frozen=True, slots=True)
class NullableSpec:
    """The only way a ``null`` is ever accepted."""

    inner: Any

    def validate(self, value: Any, path: str) -> None:
        if value is None:
            return
        self.inner.validate(value, path)


def validate_against_schema(spec: Any, value: Any, *, where: str) -> None:
    """Entry point: validate one serialized document against its exact schema."""
    spec.validate(value, where)


def iter_schema_specs(spec: Any, path: str = "") -> list[tuple[str, Any]]:
    """Enumerate ``(normalised path, spec)`` for every node of a schema tree.

    Used by the key-closure audit, which requires every ``MapSpec`` to carry
    either an explicit key constraint or a registered contextual validator.
    """
    found: list[tuple[str, Any]] = [(path, spec)]
    if isinstance(spec, ObjectSpec):
        for key, child in spec.fields.items():
            found.extend(iter_schema_specs(child, f"{path}.{key}"))
    elif isinstance(spec, ArraySpec):
        found.extend(iter_schema_specs(spec.item, f"{path}[]"))
    elif isinstance(spec, MapSpec):
        found.extend(iter_schema_specs(spec.value_spec, f"{path}{{}}"))
    elif isinstance(spec, NullableSpec):
        found.extend(iter_schema_specs(spec.inner, path))
    return found

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GridV2:
    name: str
    spot_intervals: int
    time_steps: int
    role: str


@dataclass(frozen=True, slots=True)
class CaseV2:
    name: str
    classification: str
    description: str
    option_type: str
    exercise_style: str
    spot: float
    strike: float
    expiry_time: float
    rate: float
    continuous_carry: float
    volatility: float
    dividends: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class SolverSettingsV2:
    spot_maximum: float
    rannacher_steps: int
    psor_tolerance: float
    psor_relaxation: float
    psor_maximum_iterations: int
    settlement: str
    contract_multiplier: float
    boundary_exclusion_nodes: int


@dataclass(frozen=True, slots=True)
class BumpLadder:
    spot: tuple[float, ...]
    primary_spot: float
    volatility: tuple[float, ...]
    primary_volatility: float
    ladder_ratio: int


@dataclass(frozen=True, slots=True)
class RichardsonSettingsV2:
    role: str
    coarse_to_fine_ratio: int
    assumed_order: float
    minimum_supported_observed_order: float
    maximum_supported_observed_order: float


@dataclass(frozen=True, slots=True)
class CriteriaV2:
    price_absolute_error: float
    delta_absolute_error: float
    gamma_absolute_error: float
    vega_absolute_error_per_unit_volatility: float
    shape_absolute_floor: float
    minimum_supported_bump_order: float
    maximum_supported_bump_order: float
    minimum_supported_stencil_order: float
    maximum_supported_stencil_order: float
    gamma_is_evaluation_only: bool
    fixed_bump_charge_scheme: str
    delta_flat_epsilon_rule: str
    vega_flat_epsilon_rule: str
    grid_stencil_reference_rule: str
    grid_stencil_order_rule: str
    grid_stencil_bump_bias_charge_applied: bool


@dataclass(frozen=True, slots=True)
class PolicyConfigV2:
    name: str
    selection_order: tuple[str, ...]
    order_probe_grids: tuple[str, ...]
    anchor_cases: tuple[str, ...]
    remediation_cases: tuple[str, ...]
    gate_cases: tuple[str, ...]
    non_gating_remediation_cases: tuple[str, ...]
    confirmation_case_count: int
    case_design_digest: str
    price_reference_method: str
    solver: SolverSettingsV2
    grids: tuple[GridV2, ...]
    bumps: BumpLadder
    richardson: RichardsonSettingsV2
    criteria: CriteriaV2
    dominance_control: str
    dominance_residual_measure: str
    cases: tuple[CaseV2, ...]
    source_name: str
    raw_config_sha256: str

    def grid(self, name: str) -> GridV2:
        for candidate in self.grids:
            if candidate.name == name:
                return candidate
        raise LabelPolicyV2Error(f"unknown grid '{name}'")

    def case(self, name: str) -> CaseV2:
        for candidate in self.cases:
            if candidate.name == name:
                return candidate
        raise LabelPolicyV2Error(f"unknown case '{name}'")

    def stage_case_names(self, stage: str) -> tuple[str, ...]:
        if stage == "remediation":
            return self.remediation_cases
        if stage == "confirmation":
            return tuple(case.name for case in self.cases)
        raise LabelPolicyV2Error(f"unknown stage '{stage}'")

    def is_gate_eligible(self, case: CaseV2) -> bool:
        """A case decides pass/fail only when it is regular and not excluded."""
        return (
            case.classification == "regular"
            and case.name not in self.non_gating_remediation_cases
        )


def case_design_digest(cases: Sequence[CaseV2]) -> str:
    """Digest the whole case design, states and order together."""
    payload = {
        "scheme": CRITERIA_SCHEME_VERSION,
        "cases": [
            {
                **{
                    key: value
                    for key, value in asdict(case).items()
                    if key != "dividends"
                },
                "dividends": [list(dividend) for dividend in case.dividends],
            }
            for case in cases
        ],
    }
    return _digest("cases", payload)


def load_policy_config_v2(path: Path | str) -> PolicyConfigV2:
    """Load and strictly validate the versioned v2 configuration."""
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise LabelPolicyV2Error(f"cannot read configuration '{source}': {error}") from error
    try:
        document = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise LabelPolicyV2Error(f"configuration is not valid UTF-8 TOML: {error}") from error
    return parse_policy_config_v2(
        document,
        source_name=source.name,
        raw_config_sha256=hashlib.sha256(raw).hexdigest(),
    )


def parse_policy_config_v2(
    document: Mapping[str, Any], *, source_name: str, raw_config_sha256: str
) -> PolicyConfigV2:
    """Validate an already parsed v2 configuration against the predeclaration.

    Every material fixed value is pinned against a module constant, so a
    mutation to any of them is rejected here rather than discovered in a
    result.
    """
    _reject_unknown(document, _TOP_LEVEL_KEYS, "top-level")
    if _require_string(document, "schema_version", "top-level") != SCHEMA_VERSION:
        raise LabelPolicyV2Error(f"schema_version must be '{SCHEMA_VERSION}'")

    study = _require_table(document, "study", "[study]")
    _reject_unknown(study, _STUDY_KEYS, "[study]")
    # Every protocol string is pinned. None of them is free descriptive text:
    # a report is built from these same constants and verified against them.
    for key, expected in (
        ("name", STUDY_NAME),
        ("engine", ENGINE_NAME),
        ("surface_engine", SURFACE_ENGINE_NAME),
        ("price_units", "currency_per_share"),
        ("delta_method", DELTA_METHOD),
        ("vega_method", VEGA_METHOD),
        ("price_reference_method", PRICE_REFERENCE_METHOD),
        ("curve_construction", "flat_continuously_compounded_from_case_rate"),
    ):
        if _require_string(study, key, "[study]") != expected:
            raise LabelPolicyV2Error(f"[study].{key} must be exactly '{expected}'")

    for key, expected_tuple in (
        ("selection_order", (CANDIDATE_GRID,)),
        ("order_probe_grids", (ORDER_PROBE_GRID,)),
        ("anchor_cases", ANCHOR_CASES),
        ("remediation_cases", REMEDIATION_CASES),
        ("gate_cases", REMEDIATION_GATE_CASES),
        ("non_gating_remediation_cases", (NON_GATING_REMEDIATION_CASE,)),
        ("v1_regular_failures", V1_REGULAR_FAILURES),
        ("smooth_control_cases", SMOOTH_CONTROL_CASES),
    ):
        actual = _require_string_array(study, key, "[study]")
        if actual != expected_tuple:
            raise LabelPolicyV2Error(
                f"[study].{key} must be exactly {list(expected_tuple)}, got {list(actual)}"
            )
    if _require_positive_int(study, "confirmation_case_count", "[study]") != (
        CONFIRMATION_CASE_COUNT
    ):
        raise LabelPolicyV2Error(
            f"[study].confirmation_case_count must be {CONFIRMATION_CASE_COUNT}"
        )
    declared_case_digest = _require_string(study, "case_design_digest", "[study]")

    solver = _parse_solver(document)
    grids = _parse_grids(document)
    bumps = _parse_bumps(document)
    richardson = _parse_richardson(document)
    criteria = _parse_criteria(document)
    _parse_gating(document)
    _parse_eligibility(document)
    _parse_lifecycle(document)
    dominance = _require_table(document, "dominance", "[dominance]")
    _reject_unknown(dominance, _DOMINANCE_KEYS, "[dominance]")
    dominance_control = _require_string(dominance, "control", "[dominance]")
    if dominance_control != "european_dominance_control_at_centre_only":
        raise LabelPolicyV2Error(
            "[dominance].control must be 'european_dominance_control_at_centre_only' (D2)"
        )
    residual_measure = _require_string(dominance, "residual_measure", "[dominance]")
    if residual_measure != "absolute_maximum_lcp_residual":
        raise LabelPolicyV2Error(
            "[dominance].residual_measure must be 'absolute_maximum_lcp_residual'"
        )
    if _require_bool(dominance, "is_a_rigorous_bound", "[dominance]") is not False:
        raise LabelPolicyV2Error("[dominance].is_a_rigorous_bound must be false")

    cases = _parse_cases(document, solver, bumps)
    names = tuple(case.name for case in cases)
    if len(set(names)) != len(names):
        raise LabelPolicyV2Error("case names must be unique")
    if len(cases) != CONFIRMATION_CASE_COUNT:
        raise LabelPolicyV2Error(
            f"the confirmation set must contain exactly {CONFIRMATION_CASE_COUNT} cases"
        )
    actual_case_digest = case_design_digest(cases)
    if actual_case_digest != CASE_DESIGN_DIGEST:
        raise LabelPolicyV2Error(
            "the 28 case states or their order changed: expected "
            f"{CASE_DESIGN_DIGEST}, computed {actual_case_digest}"
        )
    if declared_case_digest != CASE_DESIGN_DIGEST:
        raise LabelPolicyV2Error(
            "[study].case_design_digest does not pin the predeclared case design"
        )
    _validate_remediation_set(cases)

    config = PolicyConfigV2(
        name=STUDY_NAME,
        selection_order=(CANDIDATE_GRID,),
        order_probe_grids=(ORDER_PROBE_GRID,),
        anchor_cases=ANCHOR_CASES,
        remediation_cases=REMEDIATION_CASES,
        gate_cases=REMEDIATION_GATE_CASES,
        non_gating_remediation_cases=(NON_GATING_REMEDIATION_CASE,),
        confirmation_case_count=CONFIRMATION_CASE_COUNT,
        case_design_digest=actual_case_digest,
        price_reference_method=PRICE_REFERENCE_METHOD,
        solver=solver,
        grids=grids,
        bumps=bumps,
        richardson=richardson,
        criteria=criteria,
        dominance_control=dominance_control,
        dominance_residual_measure=residual_measure,
        cases=cases,
        source_name=source_name,
        raw_config_sha256=raw_config_sha256,
    )
    _validate_spot_grid_alignment(config)
    return config


def _parse_solver(document: Mapping[str, Any]) -> SolverSettingsV2:
    table = _require_table(document, "solver", "[solver]")
    _reject_unknown(table, _SOLVER_KEYS, "[solver]")
    solver = SolverSettingsV2(
        spot_maximum=_require_positive_float(table, "spot_maximum", "[solver]"),
        rannacher_steps=_require_nonnegative_int(table, "rannacher_steps", "[solver]"),
        psor_tolerance=_require_positive_float(table, "psor_tolerance", "[solver]"),
        psor_relaxation=_require_positive_float(table, "psor_relaxation", "[solver]"),
        psor_maximum_iterations=_require_positive_int(
            table, "psor_maximum_iterations", "[solver]"
        ),
        settlement=_require_string(table, "settlement", "[solver]"),
        contract_multiplier=_require_positive_float(table, "contract_multiplier", "[solver]"),
        boundary_exclusion_nodes=_require_positive_int(
            table, "boundary_exclusion_nodes", "[solver]"
        ),
    )
    for key, expected in PINNED_SOLVER.items():
        if getattr(solver, key) != expected:
            raise LabelPolicyV2Error(
                f"[solver].{key} is part of the predeclaration and must stay {expected!r}"
            )
    if solver.boundary_exclusion_nodes < int(_pde.pde_regime_stencil_radius):
        raise LabelPolicyV2Error(
            "[solver].boundary_exclusion_nodes must be at least the regime stencil radius"
        )
    return solver


def _parse_grids(document: Mapping[str, Any]) -> tuple[GridV2, ...]:
    raw = document.get("grids")
    if not isinstance(raw, list) or len(raw) != len(GRID_NAMES):
        raise LabelPolicyV2Error(f"top-level grids must contain exactly {len(GRID_NAMES)} tables")
    grids: list[GridV2] = []
    for index, item in enumerate(raw):
        context = f"grids[{index}]"
        if not isinstance(item, Mapping):
            raise LabelPolicyV2Error(f"{context} must be a table")
        _reject_unknown(item, _GRID_KEYS, context)
        grids.append(
            GridV2(
                name=_require_string(item, "name", context),
                spot_intervals=_require_positive_int(item, "spot_intervals", context),
                time_steps=_require_positive_int(item, "time_steps", context),
                role=_require_string(item, "role", context),
            )
        )
    if tuple(grid.name for grid in grids) != GRID_NAMES:
        raise LabelPolicyV2Error(f"grid names and order must be {GRID_NAMES}")
    if tuple((grid.spot_intervals, grid.time_steps) for grid in grids) != GRID_DIMENSIONS:
        raise LabelPolicyV2Error("grid dimensions must be the declared 800x400..6400x3200 ladder")
    if tuple(grid.role for grid in grids) != GRID_ROLES:
        raise LabelPolicyV2Error(f"grid roles must be {GRID_ROLES}")
    return tuple(grids)


def _parse_bumps(document: Mapping[str, Any]) -> BumpLadder:
    table = _require_table(document, "bumps", "[bumps]")
    _reject_unknown(table, _BUMP_KEYS, "[bumps]")
    ladder = BumpLadder(
        spot=_require_positive_float_array(table, "spot", "[bumps]"),
        primary_spot=_require_positive_float(table, "primary_spot", "[bumps]"),
        volatility=_require_positive_float_array(table, "volatility", "[bumps]"),
        primary_volatility=_require_positive_float(table, "primary_volatility", "[bumps]"),
        ladder_ratio=_require_positive_int(table, "ladder_ratio", "[bumps]"),
    )
    for key, expected in PINNED_BUMPS.items():
        if getattr(ladder, key) != expected:
            raise LabelPolicyV2Error(
                f"[bumps].{key} is part of the predeclaration and must stay {expected!r}"
            )
    for name, sizes, primary in (
        ("spot", ladder.spot, ladder.primary_spot),
        ("volatility", ladder.volatility, ladder.primary_volatility),
    ):
        if sizes != (primary, 2.0 * primary, 4.0 * primary):
            raise LabelPolicyV2Error(
                f"[bumps].{name} must be exactly (h, 2h, 4h) in float64 arithmetic"
            )
    return ladder


def _parse_richardson(document: Mapping[str, Any]) -> RichardsonSettingsV2:
    table = _require_table(document, "richardson", "[richardson]")
    _reject_unknown(table, _RICHARDSON_KEYS, "[richardson]")
    settings = RichardsonSettingsV2(
        role=_require_string(table, "role", "[richardson]"),
        coarse_to_fine_ratio=_require_positive_int(table, "coarse_to_fine_ratio", "[richardson]"),
        assumed_order=_require_positive_float(table, "assumed_order", "[richardson]"),
        minimum_supported_observed_order=_require_float(
            table, "minimum_supported_observed_order", "[richardson]"
        ),
        maximum_supported_observed_order=_require_float(
            table, "maximum_supported_observed_order", "[richardson]"
        ),
    )
    if settings.role != "reference_validation_only":
        raise LabelPolicyV2Error("[richardson].role must be 'reference_validation_only'")
    if settings.coarse_to_fine_ratio != 2 or settings.assumed_order != 2.0:
        raise LabelPolicyV2Error("v2 Richardson must use a factor-two, second-order formula")
    band = (
        settings.minimum_supported_observed_order,
        settings.maximum_supported_observed_order,
    )
    if band != SUPPORTED_ORDER_BAND:
        raise LabelPolicyV2Error(
            f"[richardson] observed-order band must stay {list(SUPPORTED_ORDER_BAND)}"
        )
    return settings


def _parse_criteria(document: Mapping[str, Any]) -> CriteriaV2:
    table = _require_table(document, "criteria", "[criteria]")
    _reject_unknown(table, _CRITERIA_KEYS, "[criteria]")
    values: dict[str, Any] = {
        key: _require_positive_float(table, key, "[criteria]") for key in _CRITERIA_FLOAT_KEYS
    }
    criteria = CriteriaV2(
        **values,
        gamma_is_evaluation_only=_require_bool(table, "gamma_is_evaluation_only", "[criteria]"),
        fixed_bump_charge_scheme=_require_string(table, "fixed_bump_charge_scheme", "[criteria]"),
        delta_flat_epsilon_rule=_require_string(table, "delta_flat_epsilon_rule", "[criteria]"),
        vega_flat_epsilon_rule=_require_string(table, "vega_flat_epsilon_rule", "[criteria]"),
        grid_stencil_reference_rule=_require_string(
            table, "grid_stencil_reference_rule", "[criteria]"
        ),
        grid_stencil_order_rule=_require_string(table, "grid_stencil_order_rule", "[criteria]"),
        grid_stencil_bump_bias_charge_applied=_require_bool(
            table, "grid_stencil_bump_bias_charge_applied", "[criteria]"
        ),
    )
    for key, expected in UNCHANGED_V1_CAPS.items():
        if getattr(criteria, key) != expected:
            raise LabelPolicyV2Error(
                f"[criteria].{key} must stay at the unchanged v1 cap {expected!r}"
            )
    for key, expected in PINNED_CRITERIA.items():
        if getattr(criteria, key) != expected:
            raise LabelPolicyV2Error(
                f"[criteria].{key} is part of the predeclaration and must stay {expected!r}"
            )
    return criteria


def _parse_gating(document: Mapping[str, Any]) -> None:
    table = _require_table(document, "gating", "[gating]")
    _reject_unknown(table, _GATING_KEYS, "[gating]")
    for key, expected in (
        ("gating_checks", GATING_CHECKS),
        ("non_gating_published_checks", NON_GATING_PUBLISHED_CHECKS),
        ("price_critical_solves", PRICE_CRITICAL_SOLVES),
        ("delta_only_solves", DELTA_ONLY_SOLVES),
        ("vega_only_solves", VEGA_ONLY_SOLVES),
        ("anchor_descriptive_solves", ANCHOR_DESCRIPTIVE_SOLVES),
    ):
        if _require_string_array(table, key, "[gating]") != expected:
            raise LabelPolicyV2Error(f"[gating].{key} must be exactly {list(expected)}")
    for key in (
        "regular_case_price_critical_failure_fails_candidate",
        "anchor_rung_failure_is_descriptive",
        "stress_case_failure_is_descriptive",
    ):
        if not _require_bool(table, key, "[gating]"):
            raise LabelPolicyV2Error(f"[gating].{key} must be true")


def _parse_eligibility(document: Mapping[str, Any]) -> None:
    table = _require_table(document, "eligibility", "[eligibility]")
    _reject_unknown(table, _ELIGIBILITY_KEYS, "[eligibility]")
    for key, expected in (
        ("common_predicates", COMMON_ELIGIBILITY_PREDICATES),
        ("structural_c1_applies_to", STRUCTURAL_C1_APPLIES_TO),
        ("delta_predicates", DELTA_ELIGIBILITY_PREDICATES),
        ("vega_predicates", VEGA_ELIGIBILITY_PREDICATES),
    ):
        if _require_string_array(table, key, "[eligibility]") != expected:
            raise LabelPolicyV2Error(f"[eligibility].{key} must be exactly {list(expected)}")
    if _require_string(table, "gamma_eligibility", "[eligibility]") != "never":
        raise LabelPolicyV2Error("[eligibility].gamma_eligibility must be 'never'")
    if not _require_bool(table, "identity_excludes_contract_multiplier", "[eligibility]"):
        raise LabelPolicyV2Error(
            "[eligibility].identity_excludes_contract_multiplier must be true"
        )
    if not _require_bool(table, "identity_includes_settlement", "[eligibility]"):
        raise LabelPolicyV2Error("[eligibility].identity_includes_settlement must be true")


def _parse_lifecycle(document: Mapping[str, Any]) -> None:
    table = _require_table(document, "lifecycle", "[lifecycle]")
    _reject_unknown(table, _LIFECYCLE_KEYS, "[lifecycle]")
    for key, expected in (
        ("stages", STAGES),
        ("remediation_statuses", REMEDIATION_STATUSES),
        ("confirmation_statuses", CONFIRMATION_STATUSES),
    ):
        if _require_string_array(table, key, "[lifecycle]") != expected:
            raise LabelPolicyV2Error(f"[lifecycle].{key} must be exactly {list(expected)}")
    if not _require_bool(table, "selection_requires_fresh_top_level_approval", "[lifecycle]"):
        raise LabelPolicyV2Error(
            "[lifecycle].selection_requires_fresh_top_level_approval must be true"
        )
    for key in ("authorizes_dataset_generation", "authorizes_training_input"):
        if _require_bool(table, key, "[lifecycle]") is not False:
            raise LabelPolicyV2Error(f"[lifecycle].{key} must be false")
    if (
        _require_string(table, "one_shot_enforcement", "[lifecycle]")
        != "procedural_and_provenance_backed"
    ):
        raise LabelPolicyV2Error(
            "[lifecycle].one_shot_enforcement must be 'procedural_and_provenance_backed'"
        )


def _parse_cases(
    document: Mapping[str, Any], solver: SolverSettingsV2, bumps: BumpLadder
) -> tuple[CaseV2, ...]:
    raw = document.get("cases")
    if not isinstance(raw, list) or not raw:
        raise LabelPolicyV2Error("top-level cases must be a non-empty array of tables")
    return tuple(_parse_case(item, index, solver, bumps) for index, item in enumerate(raw))


def _parse_case(value: Any, index: int, solver: SolverSettingsV2, bumps: BumpLadder) -> CaseV2:
    context = f"cases[{index}]"
    if not isinstance(value, Mapping):
        raise LabelPolicyV2Error(f"{context} must be a table")
    _reject_unknown(value, _CASE_KEYS, context)
    classification = _require_string(value, "classification", context)
    option_type = _require_string(value, "option_type", context)
    exercise_style = _require_string(value, "exercise_style", context)
    if classification not in CLASSIFICATIONS:
        raise LabelPolicyV2Error(f"{context}.classification must be regular or stress")
    if option_type not in OPTION_TYPES:
        raise LabelPolicyV2Error(f"{context}.option_type must be call or put")
    if exercise_style not in EXERCISE_STYLES:
        raise LabelPolicyV2Error(f"{context}.exercise_style must be european or american")
    spot = _require_positive_float(value, "spot", context)
    strike = _require_positive_float(value, "strike", context)
    expiry = _require_positive_float(value, "expiry_time", context)
    volatility = _require_positive_float(value, "volatility", context)
    rate = _require_float(value, "rate", context)
    carry = _require_float(value, "continuous_carry", context)
    if spot - max(bumps.spot) <= 0.0:
        raise LabelPolicyV2Error(f"{context} spot bump ladder crosses S <= 0")
    if volatility - max(bumps.volatility) <= 0.0:
        raise LabelPolicyV2Error(f"{context} volatility bump ladder crosses sigma <= 0")
    if solver.spot_maximum <= max(spot + max(bumps.spot), strike):
        raise LabelPolicyV2Error(f"{context} bumped spot or strike reaches the spatial boundary")
    raw_dividends = value.get("dividends")
    if not isinstance(raw_dividends, list):
        raise LabelPolicyV2Error(f"{context}.dividends must be an explicit array")
    dividends: list[tuple[float, float]] = []
    for dividend_index, raw in enumerate(raw_dividends):
        if not isinstance(raw, list) or len(raw) != 2:
            raise LabelPolicyV2Error(
                f"{context}.dividends[{dividend_index}] must be [time, amount]"
            )
        ex_time = _as_finite_float(raw[0], f"{context}.dividends[{dividend_index}][0]")
        amount = _as_finite_float(raw[1], f"{context}.dividends[{dividend_index}][1]")
        if not 0.0 < ex_time < expiry or amount <= 0.0:
            raise LabelPolicyV2Error(
                f"{context} dividend must have 0 < time < expiry and amount > 0"
            )
        dividends.append((ex_time, amount))
    if any(later[0] <= earlier[0] for earlier, later in itertools.pairwise(dividends)):
        raise LabelPolicyV2Error(f"{context}.dividends must be strictly time ordered")
    return CaseV2(
        name=_require_string(value, "name", context),
        classification=classification,
        description=_require_string(value, "description", context),
        option_type=option_type,
        exercise_style=exercise_style,
        spot=spot,
        strike=strike,
        expiry_time=expiry,
        rate=rate,
        continuous_carry=carry,
        volatility=volatility,
        dividends=tuple(dividends),
    )


def _validate_remediation_set(cases: Sequence[CaseV2]) -> None:
    """Check the pinned remediation/gate sets against the parsed case design."""
    by_name = {case.name: case for case in cases}
    unknown = sorted(set(REMEDIATION_CASES) - set(by_name))
    if unknown:
        raise LabelPolicyV2Error(
            f"the remediation set names cases absent from the design: {', '.join(unknown)}"
        )
    if len(REMEDIATION_CASES) != REMEDIATION_CASE_COUNT:
        raise LabelPolicyV2Error("the remediation set must contain ten cases")
    stress = [name for name in REMEDIATION_CASES if by_name[name].classification == "stress"]
    if stress != [NON_GATING_REMEDIATION_CASE]:
        raise LabelPolicyV2Error(
            "the remediation set must contain exactly one stress case, the declared anchor"
        )
    gating = tuple(
        name for name in REMEDIATION_CASES if by_name[name].classification == "regular"
    )
    if gating != REMEDIATION_GATE_CASES or len(gating) != GATING_REMEDIATION_CASE_COUNT:
        raise LabelPolicyV2Error(
            f"exactly {GATING_REMEDIATION_CASE_COUNT} regular remediation cases decide pass/fail"
        )


def _validate_spot_grid_alignment(config: PolicyConfigV2) -> None:
    """Every configured centre and bumped spot must be an exact grid node."""
    for case in config.cases:
        offsets = [0.0]
        for bump in config.bumps.spot:
            offsets.extend((-bump, bump))
        for grid in config.grids:
            spot_step, intervals = expected_spot_grid(
                strike=case.strike,
                spot_maximum=config.solver.spot_maximum,
                spot_intervals=grid.spot_intervals,
            )
            for offset in offsets:
                spot = case.spot + offset
                index = round(spot / spot_step)
                if not 0 < index < intervals or index * spot_step != spot:
                    raise LabelPolicyV2Error(
                        f"case '{case.name}' spot {spot!r} is not an exact interior node of "
                        f"grid '{grid.name}'; v2 forbids interpolated labels"
                    )


def expected_spot_grid(
    *, strike: float, spot_maximum: float, spot_intervals: int
) -> tuple[float, int]:
    """Reproduce the solver's node placement: (actual step, actual intervals).

    This mirrors the C++ rule that puts a node exactly on the strike. It is a
    duplicated derivation on purpose, exactly as task 9C-C2a's is, and must be
    updated in the same change as any alteration of the C++ grid rule.
    """
    target_step = spot_maximum / spot_intervals
    strike_index = max(1, round(strike / target_step))
    step = strike / strike_index
    intervals = math.ceil((spot_maximum / step) - 1.0e-9)
    if intervals <= strike_index:
        intervals = strike_index + 1
    return step, intervals


# ---------------------------------------------------------------------------
# Identities
# ---------------------------------------------------------------------------


def _digest(prefix: str, payload: Mapping[str, Any]) -> str:
    return f"{prefix}-{hashlib.sha256(canonical_payload(payload).encode('utf-8')).hexdigest()}"


def criteria_block(config: PolicyConfigV2) -> dict[str, Any]:
    """Every predeclared decision rule the confirmation stage must reuse."""
    return {
        "scheme": CRITERIA_SCHEME_VERSION,
        "criteria": asdict(config.criteria),
        "selection_order": list(config.selection_order),
        "order_probe_grids": list(config.order_probe_grids),
        "grids": [asdict(grid) for grid in config.grids],
        "bumps": asdict(config.bumps),
        "richardson": asdict(config.richardson),
        "solver": {
            key: value
            for key, value in asdict(config.solver).items()
            if key != "contract_multiplier"
        },
        "gating_checks": list(GATING_CHECKS),
        "non_gating_published_checks": list(NON_GATING_PUBLISHED_CHECKS),
        "price_critical_solves": list(PRICE_CRITICAL_SOLVES),
        "delta_only_solves": list(DELTA_ONLY_SOLVES),
        "vega_only_solves": list(VEGA_ONLY_SOLVES),
        "anchor_descriptive_solves": list(ANCHOR_DESCRIPTIVE_SOLVES),
        "common_predicates": list(COMMON_ELIGIBILITY_PREDICATES),
        "structural_c1_applies_to": list(STRUCTURAL_C1_APPLIES_TO),
        "delta_predicates": list(DELTA_ELIGIBILITY_PREDICATES),
        "vega_predicates": list(VEGA_ELIGIBILITY_PREDICATES),
        "gamma_eligibility": "never",
        "study_name": STUDY_NAME,
        "delta_method": DELTA_METHOD,
        "vega_method": VEGA_METHOD,
        "price_reference_method": PRICE_REFERENCE_METHOD,
        "price_reference_grid": REFERENCE_GRID,
        "dominance_control": config.dominance_control,
        "dominance_residual_measure": config.dominance_residual_measure,
        "fixed_bump_charge_factor": FIXED_BUMP_CHARGE_FACTOR,
        "fixed_bump_charge_scheme": FIXED_BUMP_CHARGE_SCHEME,
        "supported_order_band": list(SUPPORTED_ORDER_BAND),
        "case_design_digest": config.case_design_digest,
        "remediation_cases": list(config.remediation_cases),
        "gate_cases": list(config.gate_cases),
        "non_gating_remediation_cases": list(config.non_gating_remediation_cases),
        "anchor_cases": list(config.anchor_cases),
        "lifecycle": {
            "stages": list(STAGES),
            "remediation_statuses": list(REMEDIATION_STATUSES),
            "confirmation_statuses": list(CONFIRMATION_STATUSES),
            "selection_requires_fresh_top_level_approval": True,
            "authorizes_dataset_generation": False,
            "authorizes_training_input": False,
        },
    }


def criteria_digest(config: PolicyConfigV2) -> str:
    """Digest the whole predeclared decision rule set."""
    return _digest("crit", criteria_block(config))


def pricing_grid_identity(surface_input: Mapping[str, Any]) -> str:
    """Digest the pricing and grid identity of one accepted solver input."""
    missing = [
        key
        for key in (*PRICING_IDENTITY_FIELDS, *GRID_IDENTITY_FIELDS)
        if key not in surface_input
    ]
    if missing:
        raise LabelPolicyV2Error(
            f"surface input echo is missing identity fields: {', '.join(missing)}"
        )
    payload = {
        "scheme": IDENTITY_SCHEME_VERSION,
        **{key: surface_input[key] for key in PRICING_IDENTITY_FIELDS},
        **{key: surface_input[key] for key in GRID_IDENTITY_FIELDS},
    }
    return _digest("pg", payload)


def vega_convention_identity(*, bump: float) -> str:
    """Digest the exact v2 vega convention, including its absolute bump."""
    _require_positive_finite_number(bump, "vega bump")
    payload = {
        "scheme": VEGA_CONVENTION_SCHEME_VERSION,
        "formula": "(price(sigma + eta) - price(sigma - eta)) / (2 * eta)",
        "bump": bump,
        "bump_units": "absolute_volatility",
        "vega_units": "per_unit_absolute_volatility",
        "point_conversion": "vega / 100",
        "node_matching": "exact_node_index_and_bitwise_spot",
        "roles": ["sigma_down", "base", "sigma_up"],
    }
    return _digest("vega", payload)


# ---------------------------------------------------------------------------
# Numerical primitives
# ---------------------------------------------------------------------------


def centered_difference(up: float, down: float, bump: float) -> float:
    """Return the centered first difference of two bumped prices."""
    _require_positive_finite_number(bump, "centered-difference bump")
    return (up - down) / (2.0 * bump)


def vega_per_volatility_point(vega_per_unit: float) -> float:
    """Convert per-unit absolute-volatility vega to one percentage point."""
    return vega_per_unit / 100.0


def richardson_extrapolate(coarse: float, fine: float, assumed_order: float = 2.0) -> float:
    """Extrapolate a factor-two pair: fine + (fine - coarse) / (2**p - 1)."""
    _require_positive_finite_number(assumed_order, "Richardson assumed order")
    return fine + (fine - coarse) / (2.0**assumed_order - 1.0)


def observed_order(coarse: float, medium: float, fine: float) -> float | None:
    """Return log2(|coarse - medium| / |medium - fine|), or None if undefined.

    A zero denominator, a zero numerator or a non-finite result yields None,
    and None can never satisfy the supported band.
    """
    numerator = abs(coarse - medium)
    denominator = abs(medium - fine)
    if not (math.isfinite(numerator) and math.isfinite(denominator)):
        return None
    if numerator == 0.0 or denominator == 0.0:
        return None
    value = math.log2(numerator / denominator)
    return value if math.isfinite(value) else None


def order_is_supported(order: float | None, minimum: float, maximum: float) -> bool:
    """An undefined or non-finite observed order never counts as supported."""
    return order is not None and math.isfinite(order) and minimum <= order <= maximum


def price_residual_scale(*, time_steps: int, maximum_lcp_residual: float) -> float:
    """Operational price-error scale of one solve: steps x absolute LCP residual."""
    return float(time_steps) * abs(float(maximum_lcp_residual))


def delta_residual_scale(
    *, time_steps: int, maximum_lcp_residual: float, spot_step: float
) -> float:
    """``M_base * R_base / spot_step``: the grid-stencil delta's residual scale."""
    _require_positive_finite_number(spot_step, "spot step")
    return price_residual_scale(
        time_steps=time_steps, maximum_lcp_residual=maximum_lcp_residual
    ) / float(spot_step)


def vega_residual_scale(
    *,
    down_time_steps: int,
    down_maximum_lcp_residual: float,
    up_time_steps: int,
    up_maximum_lcp_residual: float,
    vega_bump: float,
) -> float:
    """``(M_down * R_down + M_up * R_up) / (2 * vega_bump)``."""
    _require_positive_finite_number(vega_bump, "vega bump")
    down = price_residual_scale(
        time_steps=down_time_steps, maximum_lcp_residual=down_maximum_lcp_residual
    )
    up = price_residual_scale(
        time_steps=up_time_steps, maximum_lcp_residual=up_maximum_lcp_residual
    )
    return (down + up) / (2.0 * float(vega_bump))


def delta_flat_epsilons(*, base_price_residual_scale: float, primary_bump: float) -> tuple[
    float, float
]:
    """Return ``(epsilon_small, epsilon_large)`` for the fixed-bump delta ladder.

    Each rung carries its own estimator error scale: with ``E = M_base *
    R_base`` the centered difference at bump ``b`` has error scale ``E / b``.
    The two ladder increments therefore inherit different tolerances,

    ``epsilon_small = E/h + E/(2h) = 3E/(2h)``,
    ``epsilon_large = E/(2h) + E/(4h) = 3E/(4h)``.

    No cancellation is assumed between the node errors of one shared surface:
    the prices come from the same solve, but the bound adds the two rung error
    scales rather than claiming they offset.
    """
    _require_positive_finite_number(primary_bump, "primary spot bump")
    scale = abs(float(base_price_residual_scale))
    return 3.0 * scale / (2.0 * primary_bump), 3.0 * scale / (4.0 * primary_bump)


def vega_flat_epsilons(errors_by_bump: Sequence[float]) -> tuple[float, float]:
    """Return ``(epsilon_small, epsilon_large)`` for the fixed-bump vega ladder.

    ``errors_by_bump`` holds ``err(h), err(2h), err(4h)`` where
    ``err(b) = (M_down_b * R_down_b + M_up_b * R_up_b) / (2b)``. Every
    sigma-down and sigma-up solve on the ladder therefore enters the
    tolerance explicitly.
    """
    if len(errors_by_bump) != 3:
        raise LabelPolicyV2Error("the vega ladder needs exactly three rung error scales")
    first, second, third = (abs(float(value)) for value in errors_by_bump)
    return first + second, second + third


def threshold_pass(error: float | None, threshold: float) -> bool:
    """Inclusive predeclared absolute-error threshold; None never passes."""
    return error is not None and math.isfinite(error) and error <= threshold


# ---------------------------------------------------------------------------
# E1: fixed-bump delta/vega validation
# ---------------------------------------------------------------------------


def evaluate_fixed_bump_validation(
    *,
    quantity: str,
    estimate_h: float,
    estimate_2h: float,
    estimate_4h: float,
    reference_estimate: float,
    epsilon_small: float,
    epsilon_large: float,
    cap: float,
    minimum_order: float,
    maximum_order: float,
) -> dict[str, Any]:
    """Validate a fixed-bump Greek against a refined estimator at the same bump.

    The reference uses the *same* primary bump convention, so its finite-bump
    truncation and the candidate's are the same quantity and are charged once:
    one combined budget of
    ``reference_absolute_error + bias_charge <= cap``.

    Each ladder increment is compared against its own tolerance, because the
    two increments are built from rungs with different estimator error scales.

    * **resolved** -- at least one increment exceeds its tolerance, so
      ``p = log2(Delta_large / Delta_small)`` is meaningful and is required to
      lie inside the supported band; an undefined, non-finite or
      zero-denominator ``p`` can never satisfy it. The bias charge is
      ``K * Delta_small``.
    * **flat** -- both increments sit within their own tolerances, so ``p`` is
      a ratio of noise and decides nothing; the bias charge falls back to
      ``K * epsilon_small``.

    ``K * epsilon_small > cap`` means the solver's own noise floor, charged at
    ``K``, already exceeds the Greek cap. That is recorded as
    ``numerically_invalid_residual_scale``; it denies **this Greek** and never
    vetoes the price policy.
    """
    delta_small = abs(estimate_h - estimate_2h)
    delta_large = abs(estimate_2h - estimate_4h)
    order = observed_order(estimate_4h, estimate_2h, estimate_h)
    flat = delta_small <= epsilon_small and delta_large <= epsilon_large
    branch = "flat" if flat else "resolved"
    bias_charge = FIXED_BUMP_CHARGE_FACTOR * (epsilon_small if flat else delta_small)
    reference_absolute_error = abs(estimate_h - reference_estimate)
    combined_absolute_error = reference_absolute_error + bias_charge
    residual_scale_charge = FIXED_BUMP_CHARGE_FACTOR * epsilon_small
    numerically_invalid = not threshold_pass(residual_scale_charge, cap)
    checks = {
        "observed_bump_order_supported": (
            True if flat else order_is_supported(order, minimum_order, maximum_order)
        ),
        "ladder_within_residual_scale": (
            (delta_small <= epsilon_small and delta_large <= epsilon_large) if flat else True
        ),
        "combined_absolute_error_within_cap": threshold_pass(combined_absolute_error, cap),
        "residual_scale_within_cap": not numerically_invalid,
    }
    reasons = [f"{quantity}_{name}_failed" for name, passed in checks.items() if not passed]
    if numerically_invalid:
        reasons.append(f"{quantity}_numerically_invalid_residual_scale")
    return {
        "quantity": quantity,
        "branch": branch,
        "estimate_at_h": estimate_h,
        "estimate_at_2h": estimate_2h,
        "estimate_at_4h": estimate_4h,
        "delta_small": delta_small,
        "delta_large": delta_large,
        "observed_bump_order": order,
        "charge_factor": FIXED_BUMP_CHARGE_FACTOR,
        "charge_scheme": FIXED_BUMP_CHARGE_SCHEME,
        "epsilon_small": epsilon_small,
        "epsilon_large": epsilon_large,
        "bias_charge": bias_charge,
        "reference_estimate": reference_estimate,
        "reference_absolute_error": reference_absolute_error,
        "combined_absolute_error": combined_absolute_error,
        "absolute_error_cap": cap,
        "checks": checks,
        "numerically_invalid_residual_scale": numerically_invalid,
        "reasons": sorted(set(reasons)),
        "verdict": "pass" if all(checks.values()) else "fail",
    }


# ---------------------------------------------------------------------------
# E2: production grid-stencil delta validation
# ---------------------------------------------------------------------------


def evaluate_grid_stencil_delta(
    *,
    coarse_delta: float,
    candidate_delta: float,
    fine_delta: float,
    cap: float,
    assumed_order: float,
    minimum_order: float,
    maximum_order: float,
) -> dict[str, Any]:
    """Validate the production nodewise grid-stencil delta.

    The reference is the grid-Richardson of the candidate/fine stencil pair,
    and the stencil order is measured from the coarse/candidate/fine triple.
    No bump-bias charge is added: the grid-stencil reference converges toward
    the mathematical derivative rather than toward a finite-bump difference,
    so the single rule is ``|candidate - reference| <= cap``.
    """
    reference = richardson_extrapolate(candidate_delta, fine_delta, assumed_order)
    order = observed_order(coarse_delta, candidate_delta, fine_delta)
    order_supported = order_is_supported(order, minimum_order, maximum_order)
    absolute_error = abs(candidate_delta - reference)
    checks = {
        "observed_stencil_order_supported": order_supported,
        "absolute_error_within_cap": threshold_pass(absolute_error, cap),
    }
    reasons: list[str] = []
    if not order_supported:
        reasons.append("grid_stencil_observed_order_unsupported")
    if not checks["absolute_error_within_cap"]:
        reasons.append("grid_stencil_absolute_error_exceeds_cap")
    return {
        "coarse_stencil_delta": coarse_delta,
        "candidate_stencil_delta": candidate_delta,
        "fine_stencil_delta": fine_delta,
        "reference_stencil_delta": reference,
        "reference_definition": GRID_STENCIL_REFERENCE_RULE,
        "order_definition": GRID_STENCIL_ORDER_RULE,
        "observed_stencil_order": order,
        "absolute_error": absolute_error,
        "absolute_error_cap": cap,
        "bump_bias_charge_applied": False,
        "checks": checks,
        "reasons": reasons,
        "verdict": "pass" if all(checks.values()) else "fail",
    }


def unavailable_grid_stencil(cap: float, reason: str) -> dict[str, Any]:
    """The E2 block when a supporting stencil delta could not be produced."""
    return {
        "coarse_stencil_delta": None,
        "candidate_stencil_delta": None,
        "fine_stencil_delta": None,
        "reference_stencil_delta": None,
        "reference_definition": GRID_STENCIL_REFERENCE_RULE,
        "order_definition": GRID_STENCIL_ORDER_RULE,
        "observed_stencil_order": None,
        "absolute_error": None,
        "absolute_error_cap": cap,
        "bump_bias_charge_applied": False,
        "checks": {
            "observed_stencil_order_supported": False,
            "absolute_error_within_cap": False,
        },
        "reasons": [reason],
        "verdict": "fail",
    }


# ---------------------------------------------------------------------------
# Shape: dominance and intrinsic
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ShapeAllowance:
    scale: float
    floor: float
    allowance: float
    exceeds_price_cap: bool


def shape_allowance(*, scale: float, floor: float, price_cap: float) -> ShapeAllowance:
    """Return the operational shape allowance and whether the scale is invalid.

    The allowance is the larger of the predeclared absolute floor and the
    corresponding operational price-error scale estimate, and it is never
    widened past the price cap.
    """
    exceeds = not (math.isfinite(scale) and scale <= price_cap)
    bounded = price_cap if exceeds else scale
    return ShapeAllowance(
        scale=scale, floor=floor, allowance=max(floor, bounded), exceeds_price_cap=exceeds
    )


def dominance_scale(
    *,
    american_time_steps: int,
    american_maximum_lcp_residual: float,
    european_time_steps: int,
    european_maximum_lcp_residual: float,
) -> float:
    """Operational price-error scale of an American-versus-European comparison."""
    return price_residual_scale(
        time_steps=american_time_steps, maximum_lcp_residual=american_maximum_lcp_residual
    ) + price_residual_scale(
        time_steps=european_time_steps, maximum_lcp_residual=european_maximum_lcp_residual
    )


def intrinsic_scale(*, american_time_steps: int, american_maximum_lcp_residual: float) -> float:
    """Operational price-error scale of the American intrinsic comparison."""
    return price_residual_scale(
        time_steps=american_time_steps, maximum_lcp_residual=american_maximum_lcp_residual
    )


# ---------------------------------------------------------------------------
# Measured inputs
# ---------------------------------------------------------------------------

SOLVE_RECORD_FIELDS: Final = (
    "grid_name",
    "role",
    "exercise_style",
    "volatility",
    "solver_status",
    "time_steps",
    "maximum_lcp_residual",
    "spot_step",
    "spot_intervals",
    "pricing_grid_identity",
    "centre_node_index",
    "centre_spot_is_exact_node",
    "centre_price",
    "centre_delta",
    "centre_gamma",
    "centre_greek_eligible",
    "centre_greek_eligibility_reason",
    "centre_exercise_state",
    "backward_inductions",
    "linear_solves",
    "psor_solves",
    "psor_total_iterations",
    "bumped_prices",
    "bumped_spots_are_exact_nodes",
)


#: The one solver status that means the discrete system was actually solved.
CONVERGED_SOLVER_STATUS: Final = "discrete_system_converged"


@dataclass(frozen=True, slots=True)
class SurfaceSolve:
    """One valuation-time surface solve, reduced to what the v2 criteria read."""

    grid_name: str
    role: str
    exercise_style: str
    volatility: float
    solver_status: str
    time_steps: int
    maximum_lcp_residual: float
    spot_step: float
    spot_intervals: int
    pricing_grid_identity: str
    centre_node_index: int
    centre_spot_is_exact_node: bool
    centre_price: float
    centre_delta: float | None
    centre_gamma: float | None
    centre_greek_eligible: bool
    centre_greek_eligibility_reason: str
    centre_exercise_state: str
    backward_inductions: int = 1
    linear_solves: int = 0
    psor_solves: int = 0
    psor_total_iterations: int = 0
    bumped_prices: Mapping[str, float] = field(default_factory=dict)
    bumped_spots_are_exact_nodes: bool = True

    @property
    def converged(self) -> bool:
        return self.solver_status == CONVERGED_SOLVER_STATUS

    @property
    def price_residual_scale(self) -> float:
        return price_residual_scale(
            time_steps=self.time_steps, maximum_lcp_residual=self.maximum_lcp_residual
        )

    def record(self) -> dict[str, Any]:
        """Publish every raw field a verifier needs to recompute the decisions."""
        payload = asdict(self)
        payload["bumped_prices"] = {
            key: float(value) for key, value in sorted(self.bumped_prices.items())
        }
        return payload


@dataclass(frozen=True, slots=True)
class SolveFailure:
    """A structured record of one solve that did not produce a usable surface."""

    case: str
    grid: str
    role: str
    option_type: str
    exercise_style: str
    criticality: str
    exception_class: str
    message: str

    def record(self) -> dict[str, Any]:
        return asdict(self)


def surface_solve_from_record(record: Mapping[str, Any]) -> SurfaceSolve:
    """Rebuild a solve from a published record, rejecting an unknown field."""
    unknown = sorted(set(record) - set(SOLVE_RECORD_FIELDS))
    missing = sorted(set(SOLVE_RECORD_FIELDS) - set(record))
    if unknown or missing:
        raise LabelPolicyV2Error(
            f"published solve record has unknown={unknown} missing={missing} fields"
        )
    bumped = record["bumped_prices"]
    if not isinstance(bumped, Mapping):
        raise LabelPolicyV2Error("published solve record bumped_prices must be an object")
    return SurfaceSolve(
        **{key: record[key] for key in SOLVE_RECORD_FIELDS if key != "bumped_prices"},
        bumped_prices={str(key): float(value) for key, value in bumped.items()},
    )


def spot_bump_key(bump: float, direction: str) -> str:
    """Stable key for one bumped spot price on the base surface."""
    if direction not in {"down", "up"}:
        raise LabelPolicyV2Error("bump direction must be 'down' or 'up'")
    return f"spot_{number_token(bump)}_{direction}"


def volatility_role(bump: float, direction: str) -> str:
    """Stable solve role for one volatility-bumped surface."""
    if direction not in {"down", "up"}:
        raise LabelPolicyV2Error("bump direction must be 'down' or 'up'")
    return f"volatility_{number_token(bump)}_{direction}"


def planned_role_state(case: CaseV2, role: str) -> tuple[str, float]:
    """Return the ``(exercise_style, volatility)`` a role is required to solve."""
    if role == DOMINANCE_CONTROL_ROLE:
        return "european", case.volatility
    if role == BASE_ROLE:
        return case.exercise_style, case.volatility
    if not role.startswith("volatility_"):
        raise LabelPolicyV2Error(f"unknown solve role '{role}'")
    token, direction = role[len("volatility_") :].rsplit("_", 1)
    if direction not in {"down", "up"}:
        raise LabelPolicyV2Error(f"unknown solve role '{role}'")
    bump = float(token)
    return case.exercise_style, case.volatility + (bump if direction == "up" else -bump)


def solve_criticality(case: CaseV2, grid_name: str, role: str) -> str:
    """Classify what a solve's failure is allowed to veto.

    A stress case can veto nothing, so every one of its solves is descriptive.
    Otherwise the extra 6400x3200 anchor rung is descriptive, the 800x400
    order probe supports only the delta stencil order, the volatility ladder
    supports only vega, and the 1600/3200 base solves plus the centre European
    dominance control are price-critical.
    """
    if case.classification == "stress":
        return STRESS_DESCRIPTIVE
    if grid_name == ANCHOR_GRID:
        return ANCHOR_DESCRIPTIVE
    if role.startswith("volatility_"):
        return VEGA_ONLY
    if grid_name == ORDER_PROBE_GRID:
        return DELTA_ONLY
    return PRICE_CRITICAL


def required_solve_keys(case: CaseV2, config: PolicyConfigV2) -> tuple[tuple[str, str], ...]:
    """Return every ``(grid, role)`` a case needs, in canonical order."""
    keys: list[tuple[str, str]] = [
        (ORDER_PROBE_GRID, BASE_ROLE),
        (CANDIDATE_GRID, BASE_ROLE),
        (REFERENCE_GRID, BASE_ROLE),
    ]
    for bump in config.bumps.volatility:
        for direction in ("down", "up"):
            keys.append((CANDIDATE_GRID, volatility_role(bump, direction)))
    for direction in ("down", "up"):
        keys.append((REFERENCE_GRID, volatility_role(config.bumps.primary_volatility, direction)))
    if case.exercise_style == "american":
        keys.append((CANDIDATE_GRID, DOMINANCE_CONTROL_ROLE))
    if case.name in config.anchor_cases:
        keys.append((ANCHOR_GRID, BASE_ROLE))
    return tuple(keys)


def planned_solve_count(config: PolicyConfigV2, stage: str) -> int:
    """How many surface solves a stage attempts, before any injected failure."""
    return sum(
        len(required_solve_keys(config.case(name), config))
        for name in config.stage_case_names(stage)
    )


# ---------------------------------------------------------------------------
# Pure raw-record derivations
# ---------------------------------------------------------------------------
#
# Three published maps have exact contents that follow from the versioned
# configuration and from raw per-solve, per-failure and per-check records
# alone. Each is derived exactly once, here, and that one derivation is used
# twice: at the admission boundary, against the still-raw validated report
# dictionaries and before any dataclass exists, and again during aggregate
# recomputation. There is deliberately no second implementation of these
# semantics that could drift out of step with the first.


def _expected_pricing_grid_identity_raw(
    case: CaseV2,
    config: PolicyConfigV2,
    *,
    grid_name: str,
    exercise_style: str,
    volatility: float,
) -> str:
    """Rebuild the identity a solve must carry from raw fields alone."""
    grid = config.grid(grid_name)
    return pricing_grid_identity(
        {
            "option_type": case.option_type,
            "exercise_style": exercise_style,
            "strike": case.strike,
            "valuation_time": 0.0,
            "expiry_time": case.expiry_time,
            "volatility": volatility,
            "continuous_carry": case.continuous_carry,
            "curve_times": [0.0, case.expiry_time],
            "curve_log_discounts": [0.0, -case.rate * case.expiry_time],
            "dividends": [list(dividend) for dividend in case.dividends],
            "settlement": config.solver.settlement,
            "spot_intervals": grid.spot_intervals,
            "time_steps": grid.time_steps,
            "spot_maximum": config.solver.spot_maximum,
            "rannacher_steps": config.solver.rannacher_steps,
            "psor_tolerance": config.solver.psor_tolerance,
            "psor_relaxation": config.solver.psor_relaxation,
            "psor_maximum_iterations": config.solver.psor_maximum_iterations,
            "boundary_exclusion_nodes": config.solver.boundary_exclusion_nodes,
        }
    )


def _raw_identity_matches_plan(
    case: CaseV2,
    config: PolicyConfigV2,
    grid_name: str,
    role: str,
    record: Mapping[str, Any],
) -> bool:
    """Bind one raw solve record to the exact state its role had to solve."""
    exercise_style, volatility = planned_role_state(case, role)
    if (
        record["grid_name"] != grid_name
        or record["exercise_style"] != exercise_style
        or record["volatility"] != volatility
    ):
        return False
    return record["pricing_grid_identity"] == _expected_pricing_grid_identity_raw(
        case,
        config,
        grid_name=str(record["grid_name"]),
        exercise_style=str(record["exercise_style"]),
        volatility=float(record["volatility"]),
    )


def solve_record_problems(
    case: CaseV2,
    config: PolicyConfigV2,
    grid_name: str,
    role: str,
    record: Mapping[str, Any] | None,
    *,
    solver_raised: bool,
) -> list[str]:
    """Every problem one planned solve exhibits, from its raw record alone.

    An absent record is itself the problem, and which problem it is depends on
    whether the stage recorded a solver exception for that role.
    """
    if record is None:
        return ["solver_exception" if solver_raised else "solve_absent"]
    issues: list[str] = []
    if record["solver_status"] != CONVERGED_SOLVER_STATUS:
        issues.append(f"solver_status_{record['solver_status']}")
    if not record["centre_spot_is_exact_node"]:
        issues.append("centre_spot_not_an_exact_node")
    if not record["bumped_spots_are_exact_nodes"]:
        issues.append("bumped_spot_not_an_exact_node")
    if not math.isfinite(record["centre_price"]):
        issues.append("non_finite_centre_price")
    if not _raw_identity_matches_plan(case, config, grid_name, role, record):
        issues.append("pricing_or_grid_identity_mismatch")
    scale = price_residual_scale(
        time_steps=record["time_steps"],
        maximum_lcp_residual=record["maximum_lcp_residual"],
    )
    if not threshold_pass(scale, config.criteria.price_absolute_error):
        issues.append("price_residual_scale_above_price_cap")
    return issues


def raw_solve_problems(
    case: CaseV2,
    config: PolicyConfigV2,
    records: Mapping[tuple[str, str], Mapping[str, Any]],
    failed_keys: AbstractSet[tuple[str, str]],
) -> dict[str, list[str]]:
    """The exact ``solve_problems`` map for one case, in canonical form.

    Canonical form: one entry per *planned* solve that has at least one
    problem, keyed ``"<grid>/<role>"``, sorted by key, whose value is the
    sorted list of that solve's problems. A problem-free planned solve is
    omitted entirely; there are no empty-list entries.
    """
    problems: dict[str, list[str]] = {}
    for grid_name, role in required_solve_keys(case, config):
        issues = solve_record_problems(
            case,
            config,
            grid_name,
            role,
            records.get((grid_name, role)),
            solver_raised=(grid_name, role) in failed_keys,
        )
        if issues:
            problems[f"{grid_name}/{role}"] = sorted(issues)
    return dict(sorted(problems.items()))


def raw_failed_checks_by_name(
    cases: Sequence[Mapping[str, Any]], config: PolicyConfigV2
) -> dict[str, list[str]]:
    """The exact ``failed_checks_by_name`` map, in canonical form.

    Canonical form: one entry per gating check that at least one *gate-eligible*
    case failed, sorted by check name, whose value is the sorted list of the
    names of the cases that failed it. Gate eligibility is taken from the
    versioned configuration, never from the row's own claim. A check no
    gate-eligible case failed is omitted; there are no empty-list entries, and
    a descriptive case never contributes.
    """
    failed: dict[str, list[str]] = {}
    for row in cases:
        name = str(_mapping(row["state"], "case.state")["name"])
        if not config.is_gate_eligible(config.case(name)):
            continue
        for check, passed in _mapping(row["checks"], f"case '{name}'.checks").items():
            if not passed:
                failed.setdefault(str(check), []).append(name)
    return {key: sorted(value) for key, value in sorted(failed.items())}


def raw_attempted_by_case_classification(
    cases: Sequence[Mapping[str, Any]], config: PolicyConfigV2
) -> dict[str, int]:
    """The exact ``attempted_by_case_classification`` map, in canonical form.

    Canonical form: one entry per case classification actually present in the
    stage, sorted by classification, whose value is the number of solves the
    configuration plans for the cases of that classification. A classification
    the stage does not cover is omitted; there are no zero-count entries.
    """
    counts: dict[str, int] = {}
    for row in cases:
        case = config.case(str(_mapping(row["state"], "case.state")["name"]))
        counts[case.classification] = counts.get(case.classification, 0) + len(
            required_solve_keys(case, config)
        )
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Case evaluation
# ---------------------------------------------------------------------------


def evaluate_case(
    case: CaseV2,
    config: PolicyConfigV2,
    solves: Mapping[tuple[str, str], SurfaceSolve],
    failures: Sequence[SolveFailure] = (),
) -> dict[str, Any]:
    """Apply every predeclared v2 rule to one case's measured solves.

    ``solves`` may be incomplete: a solve that raised is absent and is
    described by ``failures``. What a missing or unusable solve vetoes depends
    entirely on the role it plays.
    """
    criteria = config.criteria
    required = required_solve_keys(case, config)
    failure_index = {(item.grid, item.role): item for item in failures}
    unexpected = sorted(set(failure_index) - set(required))
    if unexpected:
        raise LabelPolicyV2Error(
            f"case '{case.name}' reports failures for solves it never required: {unexpected}"
        )

    # The same pure raw-record derivation the admission boundary already ran.
    problems = raw_solve_problems(
        case,
        config,
        {key: solve.record() for key, solve in solves.items()},
        frozenset(failure_index),
    )
    usable = {key: f"{key[0]}/{key[1]}" not in problems for key in required}

    by_criticality: dict[str, list[str]] = {name: [] for name in SOLVE_CRITICALITIES}
    for key in required:
        grid_name, role = key
        by_criticality[solve_criticality(case, grid_name, role)].append(f"{grid_name}/{role}")

    def _all_usable(criticality: str) -> bool:
        return all(
            usable[key]
            for key in required
            if solve_criticality(case, key[0], key[1]) == criticality
        )

    price_support = _all_usable(PRICE_CRITICAL)
    delta_support = price_support and _all_usable(DELTA_ONLY)
    vega_support = price_support and _all_usable(VEGA_ONLY)
    anchor_key = (ANCHOR_GRID, BASE_ROLE)
    anchor_required = anchor_key in required
    anchor_complete = (not anchor_required) or usable[anchor_key]

    candidate = solves.get((CANDIDATE_GRID, BASE_ROLE))
    reference = solves.get((REFERENCE_GRID, BASE_ROLE))
    probe = solves.get((ORDER_PROBE_GRID, BASE_ROLE))

    price_error = (
        abs(candidate.centre_price - reference.centre_price)
        if price_support and candidate is not None and reference is not None
        else None
    )
    gamma_error = (
        _optional_absolute_error(candidate.centre_gamma, reference.centre_gamma)
        if candidate is not None and reference is not None
        else None
    )

    fixed_bump_delta = _fixed_bump_delta(config, candidate, reference, price_support)
    fixed_bump_vega = _fixed_bump_vega(config, solves, vega_support)
    grid_stencil = _grid_stencil(config, probe, candidate, reference, delta_support)
    shape = _shape_outcome(case, config, candidate, solves, price_support)
    anchor = _anchor_outcome(case, config, solves, reference, anchor_complete)

    gate_checks = {
        "price_critical_solves_returned_and_converged": _no_issue(
            case,
            required,
            problems,
            PRICE_CRITICAL,
            ("solver_exception", "solve_absent", "solver_status"),
        ),
        "price_critical_exact_node_identity": _no_issue(
            case, required, problems, PRICE_CRITICAL, ("not_an_exact_node",)
        ),
        "price_critical_pricing_and_grid_identity": _no_issue(
            case,
            required,
            problems,
            PRICE_CRITICAL,
            ("pricing_or_grid_identity_mismatch", "non_finite_centre_price"),
        ),
        "price_critical_residual_scale_within_price_cap": _no_issue(
            case, required, problems, PRICE_CRITICAL, ("price_residual_scale_above_price_cap",)
        ),
        "price_absolute_error": threshold_pass(price_error, criteria.price_absolute_error),
        "american_dominance": shape["american_dominance"]["passes"],
        "american_intrinsic_bound": shape["american_intrinsic_bound"]["passes"],
        "spot_monotonicity": shape["spot_monotonicity"]["passes"],
        "spot_convexity": shape["spot_convexity"]["passes"],
    }
    if set(gate_checks) != set(GATING_CHECKS):
        raise LabelPolicyV2Error("the gating check set drifted from its predeclaration")

    structural_c1 = bool(candidate is not None and candidate.centre_greek_eligible)
    structural_reason = (
        candidate.centre_greek_eligibility_reason
        if candidate is not None
        else "candidate_base_solve_unavailable"
    )
    delta_eligibility = _delta_eligibility(
        config,
        candidate,
        delta_support=delta_support,
        structural_c1=structural_c1,
        structural_reason=structural_reason,
        fixed_bump=fixed_bump_delta,
        grid_stencil=grid_stencil,
    )
    vega_eligibility = _vega_eligibility(
        config,
        case,
        solves,
        vega_support=vega_support,
        structural_c1=structural_c1,
        structural_reason=structural_reason,
        fixed_bump=fixed_bump_vega,
    )

    gate_eligible = config.is_gate_eligible(case)
    return {
        "state": _case_state(case),
        "gate_eligible": gate_eligible,
        "counts_toward_selection": gate_eligible,
        "solve_plan": [f"{grid}/{role}" for grid, role in required],
        "solve_criticality": {
            name: sorted(values) for name, values in sorted(by_criticality.items()) if values
        },
        "solves": [
            solves[(grid, role)].record()
            for grid, role in required
            if (grid, role) in solves
        ],
        "solve_failures": [
            failure_index[key].record() for key in required if key in failure_index
        ],
        "solve_problems": {key: list(value) for key, value in problems.items()},
        "support": {
            "price_critical_solves_usable": price_support,
            "delta_supporting_solves_usable": delta_support,
            "vega_supporting_solves_usable": vega_support,
            "anchor_evidence_complete": anchor_complete,
        },
        "price": {
            # The reference is the raw exact-node centre value of the
            # 3200x1600 base surface, unconditionally. Nothing here is
            # Richardson-extrapolated and there is no fallback branch.
            "candidate": None if candidate is None else candidate.centre_price,
            "candidate_grid": CANDIDATE_GRID,
            "reference": None if reference is None else reference.centre_price,
            "reference_grid": REFERENCE_GRID,
            "reference_method": PRICE_REFERENCE_METHOD,
            "absolute_error": price_error,
            "absolute_error_cap": criteria.price_absolute_error,
        },
        "gamma_evaluation_only": {
            "candidate": None if candidate is None else candidate.centre_gamma,
            "reference": None if reference is None else reference.centre_gamma,
            "absolute_error": gamma_error,
            "absolute_error_cap": criteria.gamma_absolute_error,
            "within_cap": (
                None
                if gamma_error is None
                else threshold_pass(gamma_error, criteria.gamma_absolute_error)
            ),
            "gates": False,
            "supervision_eligible": False,
            "reason": "gamma_is_evaluation_only",
        },
        "fixed_bump_delta_validation": fixed_bump_delta,
        "fixed_bump_vega_validation": fixed_bump_vega,
        "grid_stencil_delta_validation": grid_stencil,
        "shape": shape,
        "anchor_validation": anchor,
        "checks": gate_checks,
        "delta_eligibility": delta_eligibility,
        "vega_eligibility": vega_eligibility,
        "passes": all(gate_checks.values()),
    }


def _no_issue(
    case: CaseV2,
    required: Sequence[tuple[str, str]],
    problems: Mapping[str, Sequence[str]],
    criticality: str,
    markers: Sequence[str],
) -> bool:
    """True when no solve of ``criticality`` reported any of ``markers``."""
    for grid_name, role in required:
        if solve_criticality(case, grid_name, role) != criticality:
            continue
        for issue in problems.get(f"{grid_name}/{role}", ()):
            if any(marker in issue for marker in markers):
                return False
    return True


def expected_pricing_grid_identity(
    case: CaseV2, config: PolicyConfigV2, solve: SurfaceSolve
) -> str:
    """Rebuild the identity a solve must carry from the configuration alone."""
    return _expected_pricing_grid_identity_raw(
        case,
        config,
        grid_name=solve.grid_name,
        exercise_style=solve.exercise_style,
        volatility=solve.volatility,
    )


def _unavailable_fixed_bump(quantity: str, cap: float, reason: str) -> dict[str, Any]:
    """The E1 record when the ladder could not be built.

    Every field the available record carries is present and explicitly null, so
    the serialized key set of a fixed-bump record is uniform and the schema can
    demand it exactly instead of tolerating absent keys.
    """
    quantity_fields: dict[str, Any] = (
        {"base_price_residual_scale": None}
        if quantity == "delta"
        else {"rung_residual_scales": None, "vega_per_volatility_point": None}
    )
    return {
        "bumps": None,
        "primary_bump": None,
        "epsilon_definition": None,
        **quantity_fields,
        "quantity": quantity,
        "branch": None,
        "estimate_at_h": None,
        "estimate_at_2h": None,
        "estimate_at_4h": None,
        "delta_small": None,
        "delta_large": None,
        "observed_bump_order": None,
        "charge_factor": FIXED_BUMP_CHARGE_FACTOR,
        "charge_scheme": FIXED_BUMP_CHARGE_SCHEME,
        "epsilon_small": None,
        "epsilon_large": None,
        "bias_charge": None,
        "reference_estimate": None,
        "reference_absolute_error": None,
        "combined_absolute_error": None,
        "absolute_error_cap": cap,
        "checks": {
            "observed_bump_order_supported": False,
            "ladder_within_residual_scale": False,
            "combined_absolute_error_within_cap": False,
            "residual_scale_within_cap": False,
        },
        "numerically_invalid_residual_scale": False,
        "reasons": [reason],
        "verdict": "fail",
    }


def _fixed_bump_delta(
    config: PolicyConfigV2,
    candidate: SurfaceSolve | None,
    reference: SurfaceSolve | None,
    supported: bool,
) -> dict[str, Any]:
    cap = config.criteria.delta_absolute_error
    if not supported or candidate is None or reference is None:
        return _unavailable_fixed_bump("delta", cap, "delta_supporting_solves_unavailable")
    try:
        estimates = [_bump_delta(candidate, bump) for bump in config.bumps.spot]
        reference_estimate = _bump_delta(reference, config.bumps.primary_spot)
    except LabelPolicyV2Error:
        return _unavailable_fixed_bump("delta", cap, "bumped_price_missing")
    epsilon_small, epsilon_large = delta_flat_epsilons(
        base_price_residual_scale=candidate.price_residual_scale,
        primary_bump=config.bumps.primary_spot,
    )
    outcome = evaluate_fixed_bump_validation(
        quantity="delta",
        estimate_h=estimates[0],
        estimate_2h=estimates[1],
        estimate_4h=estimates[2],
        reference_estimate=reference_estimate,
        epsilon_small=epsilon_small,
        epsilon_large=epsilon_large,
        cap=cap,
        minimum_order=config.criteria.minimum_supported_bump_order,
        maximum_order=config.criteria.maximum_supported_bump_order,
    )
    outcome["bumps"] = list(config.bumps.spot)
    outcome["primary_bump"] = config.bumps.primary_spot
    outcome["epsilon_definition"] = DELTA_FLAT_EPSILON_RULE
    outcome["base_price_residual_scale"] = candidate.price_residual_scale
    return outcome


def _bump_delta(solve: SurfaceSolve, bump: float) -> float:
    try:
        up = float(solve.bumped_prices[spot_bump_key(bump, "up")])
        down = float(solve.bumped_prices[spot_bump_key(bump, "down")])
    except KeyError as error:
        raise LabelPolicyV2Error(
            f"solve '{solve.grid_name}/{solve.role}' is missing a bumped price: {error}"
        ) from error
    return centered_difference(up, down, bump)


def _fixed_bump_vega(
    config: PolicyConfigV2,
    solves: Mapping[tuple[str, str], SurfaceSolve],
    supported: bool,
) -> dict[str, Any]:
    cap = config.criteria.vega_absolute_error_per_unit_volatility
    primary = config.bumps.primary_volatility
    keys = [
        (CANDIDATE_GRID, volatility_role(bump, direction))
        for bump in config.bumps.volatility
        for direction in ("down", "up")
    ] + [
        (REFERENCE_GRID, volatility_role(primary, direction)) for direction in ("down", "up")
    ]
    if not supported or any(key not in solves for key in keys):
        return _unavailable_fixed_bump("vega", cap, "vega_supporting_solves_unavailable")

    estimates: list[float] = []
    rung_errors: list[float] = []
    for bump in config.bumps.volatility:
        down = solves[(CANDIDATE_GRID, volatility_role(bump, "down"))]
        up = solves[(CANDIDATE_GRID, volatility_role(bump, "up"))]
        estimates.append(centered_difference(up.centre_price, down.centre_price, bump))
        rung_errors.append(
            vega_residual_scale(
                down_time_steps=down.time_steps,
                down_maximum_lcp_residual=down.maximum_lcp_residual,
                up_time_steps=up.time_steps,
                up_maximum_lcp_residual=up.maximum_lcp_residual,
                vega_bump=bump,
            )
        )
    reference_down = solves[(REFERENCE_GRID, volatility_role(primary, "down"))]
    reference_up = solves[(REFERENCE_GRID, volatility_role(primary, "up"))]
    reference_estimate = centered_difference(
        reference_up.centre_price, reference_down.centre_price, primary
    )
    epsilon_small, epsilon_large = vega_flat_epsilons(rung_errors)
    outcome = evaluate_fixed_bump_validation(
        quantity="vega",
        estimate_h=estimates[0],
        estimate_2h=estimates[1],
        estimate_4h=estimates[2],
        reference_estimate=reference_estimate,
        epsilon_small=epsilon_small,
        epsilon_large=epsilon_large,
        cap=cap,
        minimum_order=config.criteria.minimum_supported_bump_order,
        maximum_order=config.criteria.maximum_supported_bump_order,
    )
    outcome["bumps"] = list(config.bumps.volatility)
    outcome["primary_bump"] = primary
    outcome["epsilon_definition"] = VEGA_FLAT_EPSILON_RULE
    outcome["rung_residual_scales"] = rung_errors
    outcome["vega_per_volatility_point"] = vega_per_volatility_point(estimates[0])
    return outcome


def _grid_stencil(
    config: PolicyConfigV2,
    probe: SurfaceSolve | None,
    candidate: SurfaceSolve | None,
    reference: SurfaceSolve | None,
    supported: bool,
) -> dict[str, Any]:
    cap = config.criteria.delta_absolute_error
    if not supported:
        return unavailable_grid_stencil(cap, "delta_supporting_solves_unavailable")
    deltas = [
        None if solve is None else solve.centre_delta
        for solve in (probe, candidate, reference)
    ]
    if any(value is None or not math.isfinite(value) for value in deltas):
        return unavailable_grid_stencil(cap, "grid_stencil_delta_unavailable")
    return evaluate_grid_stencil_delta(
        coarse_delta=float(deltas[0]),
        candidate_delta=float(deltas[1]),
        fine_delta=float(deltas[2]),
        cap=cap,
        assumed_order=config.richardson.assumed_order,
        minimum_order=config.criteria.minimum_supported_stencil_order,
        maximum_order=config.criteria.maximum_supported_stencil_order,
    )


def _shape_outcome(
    case: CaseV2,
    config: PolicyConfigV2,
    candidate: SurfaceSolve | None,
    solves: Mapping[tuple[str, str], SurfaceSolve],
    price_support: bool,
) -> dict[str, Any]:
    floor = config.criteria.shape_absolute_floor
    price_cap = config.criteria.price_absolute_error
    american = case.exercise_style == "american"
    intrinsic_value = (
        max(case.spot - case.strike, 0.0)
        if case.option_type == "call"
        else max(case.strike - case.spot, 0.0)
    )
    control = solves.get((CANDIDATE_GRID, DOMINANCE_CONTROL_ROLE))
    available = price_support and candidate is not None

    if american and available and control is not None:
        dominance = shape_allowance(
            scale=dominance_scale(
                american_time_steps=candidate.time_steps,
                american_maximum_lcp_residual=candidate.maximum_lcp_residual,
                european_time_steps=control.time_steps,
                european_maximum_lcp_residual=control.maximum_lcp_residual,
            ),
            floor=floor,
            price_cap=price_cap,
        )
        intrinsic = shape_allowance(
            scale=intrinsic_scale(
                american_time_steps=candidate.time_steps,
                american_maximum_lcp_residual=candidate.maximum_lcp_residual,
            ),
            floor=floor,
            price_cap=price_cap,
        )
        dominance_gap = candidate.centre_price - control.centre_price
        intrinsic_gap = candidate.centre_price - intrinsic_value
        dominance_block = {
            "applicable": True,
            "control_price": control.centre_price,
            "gap": dominance_gap,
            "scale": dominance.scale,
            "floor": dominance.floor,
            "allowance": dominance.allowance,
            "residual_scale_exceeds_price_cap": dominance.exceeds_price_cap,
            "passes": dominance_gap >= -dominance.allowance
            and not dominance.exceeds_price_cap,
        }
        intrinsic_block = {
            "applicable": True,
            "intrinsic_value": intrinsic_value,
            "gap": intrinsic_gap,
            "scale": intrinsic.scale,
            "floor": intrinsic.floor,
            "allowance": intrinsic.allowance,
            "residual_scale_exceeds_price_cap": intrinsic.exceeds_price_cap,
            "passes": intrinsic_gap >= -intrinsic.allowance and not intrinsic.exceeds_price_cap,
        }
    else:
        # A European case poses neither comparison; an unusable price-critical
        # solve already fails its own gating check, so the shape checks stay
        # neutral rather than failing the same solve twice.
        dominance_block = {
            "applicable": False,
            "control_price": None,
            "gap": None,
            "scale": None,
            "floor": floor,
            "allowance": None,
            "residual_scale_exceeds_price_cap": False,
            "passes": True,
        }
        intrinsic_block = {
            "applicable": False,
            "intrinsic_value": intrinsic_value,
            "gap": None,
            "scale": None,
            "floor": floor,
            "allowance": None,
            "residual_scale_exceeds_price_cap": False,
            "passes": True,
        }

    delta = candidate.centre_delta if available and candidate is not None else None
    gamma = candidate.centre_gamma if available and candidate is not None else None
    if not available:
        monotone = True
        convex = True
    else:
        monotone = delta is not None and (
            (delta >= -floor) if case.option_type == "call" else (delta <= floor)
        )
        convex = gamma is not None and gamma >= -floor
    return {
        "measure": "absolute_maximum_lcp_residual",
        "is_a_rigorous_bound": False,
        "allowance_definition": (
            "max(shape_absolute_floor, operational price-error scale estimate), never "
            "widened past price_absolute_error"
        ),
        "american_dominance": dominance_block,
        "american_intrinsic_bound": intrinsic_block,
        "spot_monotonicity": {"delta": delta, "floor": floor, "passes": monotone},
        "spot_convexity": {"gamma": gamma, "floor": floor, "passes": convex},
    }


def _anchor_outcome(
    case: CaseV2,
    config: PolicyConfigV2,
    solves: Mapping[tuple[str, str], SurfaceSolve],
    reference: SurfaceSolve | None,
    complete: bool,
) -> dict[str, Any] | None:
    if case.name not in config.anchor_cases:
        return None
    anchor = solves.get((ANCHOR_GRID, BASE_ROLE))
    return {
        "grid": ANCHOR_GRID,
        "role": "retained_anchor_rung",
        "anchor_evidence_complete": complete and anchor is not None,
        "anchor_price": None if anchor is None else anchor.centre_price,
        "reference_price": None if reference is None else reference.centre_price,
        "price_absolute_gap": (
            None
            if anchor is None or reference is None
            else abs(reference.centre_price - anchor.centre_price)
        ),
        "anchor_stencil_delta": None if anchor is None else anchor.centre_delta,
        "reference_stencil_delta": None if reference is None else reference.centre_delta,
        "stencil_delta_absolute_gap": (
            None
            if anchor is None or reference is None
            else _optional_absolute_error(reference.centre_delta, anchor.centre_delta)
        ),
        "gates": False,
    }


def _delta_eligibility(
    config: PolicyConfigV2,
    candidate: SurfaceSolve | None,
    *,
    delta_support: bool,
    structural_c1: bool,
    structural_reason: str,
    fixed_bump: Mapping[str, Any],
    grid_stencil: Mapping[str, Any],
) -> dict[str, Any]:
    scale = (
        delta_residual_scale(
            time_steps=candidate.time_steps,
            maximum_lcp_residual=candidate.maximum_lcp_residual,
            spot_step=candidate.spot_step,
        )
        if candidate is not None
        else None
    )
    predicates = {
        "delta_supporting_solves_available": delta_support,
        "structural_c1_greek_eligibility": structural_c1,
        "delta_residual_scale_within_delta_cap": threshold_pass(
            scale, config.criteria.delta_absolute_error
        ),
        "fixed_bump_delta_validation_passed": fixed_bump["verdict"] == "pass",
        "fixed_bump_delta_residual_scale_within_delta_cap": not fixed_bump[
            "numerically_invalid_residual_scale"
        ],
        "grid_stencil_delta_validation_passed": grid_stencil["verdict"] == "pass",
    }
    reasons = [name for name, ok in predicates.items() if not ok]
    if not structural_c1:
        reasons.append(f"c1_{structural_reason}")
    reasons.extend(fixed_bump["reasons"])
    reasons.extend(grid_stencil["reasons"])
    return {
        "quantity": "delta",
        "estimator": "nodewise_grid_stencil_on_candidate_grid",
        "value": None if candidate is None else candidate.centre_delta,
        "residual_scale": scale,
        "residual_scale_definition": "M_base * R_base / spot_step",
        "residual_scale_cap": config.criteria.delta_absolute_error,
        "predicates": predicates,
        "eligible": all(predicates.values()),
        "reasons": sorted(set(reasons)),
    }


def _vega_eligibility(
    config: PolicyConfigV2,
    case: CaseV2,
    solves: Mapping[tuple[str, str], SurfaceSolve],
    *,
    vega_support: bool,
    structural_c1: bool,
    structural_reason: str,
    fixed_bump: Mapping[str, Any],
) -> dict[str, Any]:
    primary = config.bumps.primary_volatility
    expected_convention = vega_convention_identity(bump=primary)
    base = solves.get((CANDIDATE_GRID, BASE_ROLE))
    down = solves.get((CANDIDATE_GRID, volatility_role(primary, "down")))
    up = solves.get((CANDIDATE_GRID, volatility_role(primary, "up")))

    if down is None or up is None or base is None:
        predicates = {
            "vega_supporting_solves_available": False,
            "structural_c1_greek_eligibility": structural_c1,
            "vega_residual_scale_within_vega_cap": False,
            "vega_numerically_available": False,
            "vega_bump_exercise_states_equal": False,
            "exact_vega_convention_and_bump_identity": False,
            "fixed_bump_vega_validation_passed": False,
            "fixed_bump_vega_residual_scale_within_vega_cap": True,
        }
        return {
            "quantity": "vega_per_unit_volatility",
            "estimator": "centered_volatility_bump_at_primary_rung",
            "value": None,
            "vega_per_volatility_point": None,
            "vega_convention_id": None,
            "expected_vega_convention_id": expected_convention,
            "observed_sigma_down": None if down is None else down.volatility,
            "observed_sigma_up": None if up is None else up.volatility,
            "planned_sigma_down": planned_role_state(
                case, volatility_role(primary, "down")
            )[1],
            "planned_sigma_up": planned_role_state(case, volatility_role(primary, "up"))[1],
            "vega_bump": primary,
            "exercise_states": {"base": None, "sigma_down": None, "sigma_up": None},
            "residual_scale": None,
            "residual_scale_definition": "(M_down * R_down + M_up * R_up) / (2 * vega_bump)",
            "residual_scale_cap": config.criteria.vega_absolute_error_per_unit_volatility,
            "predicates": predicates,
            "eligible": False,
            "reasons": sorted(
                {*(name for name, ok in predicates.items() if not ok), *fixed_bump["reasons"]}
            ),
        }

    scale = vega_residual_scale(
        down_time_steps=down.time_steps,
        down_maximum_lcp_residual=down.maximum_lcp_residual,
        up_time_steps=up.time_steps,
        up_maximum_lcp_residual=up.maximum_lcp_residual,
        vega_bump=primary,
    )
    # The bump is checked against the volatilities the plan requires, bitwise,
    # not recovered by differencing them: (up - down) / 2 is not exactly the
    # declared bump in float64, and a tolerance here would be a hole.
    planned_down = planned_role_state(case, volatility_role(primary, "down"))[1]
    planned_up = planned_role_state(case, volatility_role(primary, "up"))[1]
    bump_matches_plan = down.volatility == planned_down and up.volatility == planned_up
    convention = vega_convention_identity(bump=primary) if bump_matches_plan else None
    value = centered_difference(up.centre_price, down.centre_price, primary)
    available = math.isfinite(value) and all(
        math.isfinite(solve.centre_price) for solve in (down, up)
    )
    states_equal = (
        base.centre_exercise_state == down.centre_exercise_state == up.centre_exercise_state
    )
    predicates = {
        "vega_supporting_solves_available": vega_support,
        "structural_c1_greek_eligibility": structural_c1,
        "vega_residual_scale_within_vega_cap": threshold_pass(
            scale, config.criteria.vega_absolute_error_per_unit_volatility
        ),
        "vega_numerically_available": available,
        "vega_bump_exercise_states_equal": states_equal,
        "exact_vega_convention_and_bump_identity": (
            bump_matches_plan and convention == expected_convention
        ),
        "fixed_bump_vega_validation_passed": fixed_bump["verdict"] == "pass",
        "fixed_bump_vega_residual_scale_within_vega_cap": not fixed_bump[
            "numerically_invalid_residual_scale"
        ],
    }
    reasons = [name for name, ok in predicates.items() if not ok]
    if not structural_c1:
        reasons.append(f"c1_{structural_reason}")
    reasons.extend(fixed_bump["reasons"])
    return {
        "quantity": "vega_per_unit_volatility",
        "estimator": "centered_volatility_bump_at_primary_rung",
        "value": value if available else None,
        "vega_per_volatility_point": vega_per_volatility_point(value) if available else None,
        "vega_convention_id": convention,
        "expected_vega_convention_id": expected_convention,
        "observed_sigma_down": down.volatility,
        "observed_sigma_up": up.volatility,
        "planned_sigma_down": planned_down,
        "planned_sigma_up": planned_up,
        "vega_bump": primary,
        "exercise_states": {
            "base": base.centre_exercise_state,
            "sigma_down": down.centre_exercise_state,
            "sigma_up": up.centre_exercise_state,
        },
        "residual_scale": scale,
        "residual_scale_definition": "(M_down * R_down + M_up * R_up) / (2 * vega_bump)",
        "residual_scale_cap": config.criteria.vega_absolute_error_per_unit_volatility,
        "predicates": predicates,
        "eligible": all(predicates.values()),
        "reasons": sorted(set(reasons)),
    }


def _optional_absolute_error(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return abs(float(left) - float(right))


def _case_state(case: CaseV2) -> dict[str, Any]:
    state = asdict(case)
    state["dividends"] = [list(dividend) for dividend in case.dividends]
    state["valuation_time"] = 0.0
    state["curve_times"] = [0.0, case.expiry_time]
    state["curve_log_discounts"] = [0.0, -case.rate * case.expiry_time]
    state["moneyness_spot_over_strike"] = case.spot / case.strike
    return state


# ---------------------------------------------------------------------------
# Stage outcome, accounting and selection
# ---------------------------------------------------------------------------


def stage_outcome(cases: Sequence[Mapping[str, Any]], config: PolicyConfigV2) -> dict[str, Any]:
    """Summarise one stage: which gate-eligible cases passed, and eligibility."""
    gating_failures = [
        str(row["state"]["name"])
        for row in cases
        if row["counts_toward_selection"] and not row["passes"]
    ]
    descriptive_failures = [
        str(row["state"]["name"])
        for row in cases
        if not row["counts_toward_selection"] and not row["passes"]
    ]
    return {
        "candidate": CANDIDATE_GRID,
        "gate_eligible_case_count": sum(1 for row in cases if row["counts_toward_selection"]),
        "gate_eligible_failure_count": len(gating_failures),
        "gate_eligible_failures": gating_failures,
        "all_gate_eligible_cases_pass": not gating_failures,
        "failed_checks_by_name": raw_failed_checks_by_name(cases, config),
        "descriptive_failure_count": len(descriptive_failures),
        "descriptive_failures": descriptive_failures,
        "delta_eligible_cases": sorted(
            str(row["state"]["name"]) for row in cases if row["delta_eligibility"]["eligible"]
        ),
        "vega_eligible_cases": sorted(
            str(row["state"]["name"]) for row in cases if row["vega_eligibility"]["eligible"]
        ),
        "gamma_eligible_cases": [],
        "anchor_evidence_incomplete_cases": sorted(
            str(row["state"]["name"])
            for row in cases
            if not row["support"]["anchor_evidence_complete"]
        ),
        "selection_order": list(config.selection_order),
        "regular_cases_only_decide_selection": True,
        "stress_case_is_descriptive_anchor_evidence": True,
        "criteria_were_not_loosened": True,
    }


def solve_accounting(
    cases: Sequence[Mapping[str, Any]], config: PolicyConfigV2, stage: str
) -> dict[str, Any]:
    """Reconcile every solve the stage planned, attempted, and completed."""
    planned = planned_solve_count(config, stage)
    attempted = sum(len(row["solve_plan"]) for row in cases)
    completed = sum(len(row["solves"]) for row in cases)
    exceptions = sum(len(row["solve_failures"]) for row in cases)
    by_role: dict[str, int] = {}
    by_criticality: dict[str, int] = {}
    for row in cases:
        for name, members in row["solve_criticality"].items():
            by_criticality[name] = by_criticality.get(name, 0) + len(members)
        for key in row["solve_plan"]:
            by_role[key] = by_role.get(key, 0) + 1

    def _total(field_name: str) -> int:
        return sum(int(solve[field_name]) for row in cases for solve in row["solves"])

    accounting = {
        "planned_surface_solves": planned,
        "attempted_surface_solves": attempted,
        "completed_surface_solves": completed,
        "solver_exceptions": exceptions,
        "backward_inductions": _total("backward_inductions"),
        "linear_solves": _total("linear_solves"),
        "psor_solves": _total("psor_solves"),
        "psor_total_iterations": _total("psor_total_iterations"),
        "attempted_by_solve_role": dict(sorted(by_role.items())),
        "attempted_by_criticality": dict(sorted(by_criticality.items())),
        "attempted_by_case_classification": raw_attempted_by_case_classification(
            cases, config
        ),
        "reconciliation": {
            "planned_equals_attempted": planned == attempted,
            "attempted_equals_completed_plus_exceptions": attempted
            == completed + exceptions,
            "backward_inductions_equal_completed_solves": _total("backward_inductions")
            == completed,
        },
    }
    if not all(accounting["reconciliation"].values()):
        raise LabelPolicyV2Error(f"solve accounting does not reconcile: {accounting}")
    return accounting


def select_accuracy_policy(outcome: Mapping[str, Any], config: PolicyConfigV2) -> str:
    """Select the single declared candidate, or refuse."""
    for policy in config.selection_order:
        if policy == CANDIDATE_GRID and bool(outcome["all_gate_eligible_cases_pass"]):
            return policy
    return NO_POLICY_SELECTED


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def lifecycle_block(
    *, stage: str, all_gate_eligible_cases_pass: bool, config: PolicyConfigV2
) -> dict[str, Any]:
    """Return exactly one of the three supported lifecycle states."""
    if stage not in STAGES:
        raise LabelPolicyV2Error(f"unknown stage '{stage}'")
    common = {
        "stage": stage,
        "raw_config_sha256": config.raw_config_sha256,
        "criteria_digest": criteria_digest(config),
        "case_design_digest": config.case_design_digest,
        "protocol": "one_shot_predeclared",
        "criteria_were_not_loosened": True,
        "one_shot_enforcement": "procedural_and_provenance_backed",
        "authorizes_dataset_generation": False,
        "authorizes_training_input": False,
    }
    if stage == "remediation" and not all_gate_eligible_cases_pass:
        return {
            **common,
            "remediation_status": "failed",
            "confirmation_status": "not_run",
            "confirmation_not_run_reason": "remediation_failed",
            "terminal": True,
            "freezable": True,
            "selected_accuracy_policy": NO_POLICY_SELECTED,
            "selection_pending_fresh_top_level_approval": False,
        }
    if stage == "remediation":
        # State 2 carries no selected-policy field at all.
        return {
            **common,
            "remediation_status": "passed",
            "confirmation_status": "pending",
            "confirmation_not_run_reason": None,
            "terminal": False,
            "freezable": False,
        }
    selected = CANDIDATE_GRID if all_gate_eligible_cases_pass else NO_POLICY_SELECTED
    return {
        **common,
        "remediation_status": "passed",
        "confirmation_status": "run",
        "confirmation_not_run_reason": None,
        "terminal": True,
        "freezable": True,
        "selected_accuracy_policy": selected,
        "selection_pending_fresh_top_level_approval": selected != NO_POLICY_SELECTED,
    }


# ---------------------------------------------------------------------------
# Semantic recomputation, shared by the confirmation gate and the freeze tool
# ---------------------------------------------------------------------------

REPORT_KEYS: Final = frozenset(
    {
        "schema_version",
        "lifecycle",
        "study",
        "conventions",
        "predeclared_criteria",
        "criteria_block",
        "criteria_digest",
        "validation_rules",
        "eligibility_contract",
        "richardson_contract",
        "source",
        "solver",
        "grids",
        "cases",
        "stage_outcome",
        "solve_accounting",
        "non_claims",
        "performance",
    }
)
REPORT_SOURCE_KEYS: Final = frozenset(
    {
        "build_configuration",
        "config_name",
        "cxx_compiler",
        "data_provenance",
        "executable_source_composite_digest",
        "executable_source_inventory",
        "private_market_data_used",
        "raw_config_sha256",
        "reported_binding_pde_header_sha256",
        "reported_binding_pde_implementation_sha256",
        "reported_binding_pde_source_sha256",
        "source_digest_limitation",
        *(f"{name}_sha256" for name in EXECUTABLE_SOURCE_INVENTORY),
    }
)
CASE_REPORT_KEYS: Final = frozenset(
    {
        "state",
        "gate_eligible",
        "counts_toward_selection",
        "solve_plan",
        "solve_criticality",
        "solves",
        "solve_failures",
        "solve_problems",
        "support",
        "price",
        "gamma_evaluation_only",
        "fixed_bump_delta_validation",
        "fixed_bump_vega_validation",
        "grid_stencil_delta_validation",
        "shape",
        "anchor_validation",
        "checks",
        "delta_eligibility",
        "vega_eligibility",
        "passes",
    }
)
PERFORMANCE_KEYS: Final = frozenset(
    {
        "wall_seconds",
        "timed_solve_attempts",
        "timed_solve_seconds",
        "peak_resident_memory_bytes",
        "peak_resident_memory_increase_bytes",
        "projection_caveat",
        "host",
    }
)
PERFORMANCE_HOST_KEYS: Final = frozenset({"system", "machine", "python"})
GRID_RECORD_KEYS: Final = frozenset({"name", "spot_intervals", "time_steps", "role"})
FAILURE_RECORD_FIELDS: Final = (
    "case",
    "grid",
    "role",
    "option_type",
    "exercise_style",
    "criticality",
    "exception_class",
    "message",
)


@dataclass(frozen=True, slots=True)
class RecomputedStage:
    """Everything a verifier derives again from a report's raw numbers."""

    stage: str
    cases: list[dict[str, Any]]
    outcome: dict[str, Any]
    accounting: dict[str, Any]
    lifecycle: dict[str, Any]


def recompute_stage(report: Mapping[str, Any], config: PolicyConfigV2) -> RecomputedStage:
    """Rebuild every published decision from the report's raw solve records.

    This is the single derivation the confirmation-entry validator and the
    freeze tool both use. It never reads a published verdict, count or
    lifecycle field: it reads only the raw per-solve numbers and the case
    identities, and re-runs the same evaluation the runner ran.

    It is semantic verification. It detects an inconsistent or partial
    mutation. It cannot authenticate a fully coordinated fabricated numerical
    report, because it does not re-solve.
    """
    # Exact recursive type validation of the serialized document, before any
    # SurfaceSolve or case object is constructed from it.
    validate_report_schema(report)
    validate_report_context(report, config)
    _reject_unknown_report_keys(report)
    stage = _report_stage(report, config)
    expected_names = list(config.stage_case_names(stage))
    published = _sequence(report.get("cases"), "report.cases")
    published_names = [
        _require_string(_mapping(row.get("state"), "report.cases[].state"), "name", "state")
        for row in (_mapping(item, "report.cases[]") for item in published)
    ]
    if published_names != expected_names:
        missing = sorted(set(expected_names) - set(published_names))
        unknown = sorted(set(published_names) - set(expected_names))
        duplicated = sorted(
            {name for name in published_names if published_names.count(name) > 1}
        )
        raise LabelPolicyV2Error(
            f"report.cases must be exactly the {stage} case set, in order; "
            f"missing={missing} unexpected={unknown} duplicated={duplicated}"
        )

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(published):
        row = _mapping(item, f"report.cases[{index}]")
        _exact_keys(row, CASE_REPORT_KEYS, f"report.cases[{index}]")
        case = config.case(published_names[index])
        solves, failures = _rebuild_case_inputs(row, case, config, f"report.cases[{index}]")
        rows.append(evaluate_case(case, config, solves, failures))

    outcome = stage_outcome(rows, config)
    accounting = solve_accounting(rows, config, stage)
    lifecycle = lifecycle_block(
        stage=stage,
        all_gate_eligible_cases_pass=bool(outcome["all_gate_eligible_cases_pass"]),
        config=config,
    )
    return RecomputedStage(
        stage=stage, cases=rows, outcome=outcome, accounting=accounting, lifecycle=lifecycle
    )


def _report_stage(report: Mapping[str, Any], config: PolicyConfigV2) -> str:
    lifecycle = _mapping(report.get("lifecycle"), "report.lifecycle")
    stage = _require_string(lifecycle, "stage", "report.lifecycle")
    if stage not in STAGES:
        raise LabelPolicyV2Error(f"report.lifecycle.stage must be one of {list(STAGES)}")
    study = _mapping(report.get("study"), "report.study")
    if _require_string(study, "stage", "report.study") != stage:
        raise LabelPolicyV2Error("report.study.stage and report.lifecycle.stage disagree")
    del config
    return stage


def _rebuild_case_inputs(
    row: Mapping[str, Any], case: CaseV2, config: PolicyConfigV2, where: str
) -> tuple[dict[tuple[str, str], SurfaceSolve], list[SolveFailure]]:
    required = required_solve_keys(case, config)
    plan = [f"{grid}/{role}" for grid, role in required]
    if list(_sequence(row.get("solve_plan"), f"{where}.solve_plan")) != plan:
        raise LabelPolicyV2Error(f"{where}.solve_plan is not the required solve set")

    solves: dict[tuple[str, str], SurfaceSolve] = {}
    for index, item in enumerate(_sequence(row.get("solves"), f"{where}.solves")):
        record = _mapping(item, f"{where}.solves[{index}]")
        grid = _require_string(record, "grid_name", f"{where}.solves[{index}]")
        role = _require_string(record, "role", f"{where}.solves[{index}]")
        key = (grid, role)
        if key not in required:
            raise LabelPolicyV2Error(f"{where}.solves[{index}] is not a required solve")
        if key in solves:
            raise LabelPolicyV2Error(f"{where}.solves[{index}] duplicates {grid}/{role}")
        validate_solve_record_context(record, config, where=f"{where}.solves[{index}]")
        solves[key] = surface_solve_from_record(record)

    failures: list[SolveFailure] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(_sequence(row.get("solve_failures"), f"{where}.solve_failures")):
        record = _mapping(item, f"{where}.solve_failures[{index}]")
        unknown = sorted(set(record) - set(FAILURE_RECORD_FIELDS))
        missing = sorted(set(FAILURE_RECORD_FIELDS) - set(record))
        if unknown or missing:
            raise LabelPolicyV2Error(
                f"{where}.solve_failures[{index}] has unknown={unknown} missing={missing}"
            )
        failure = SolveFailure(**{key: record[key] for key in FAILURE_RECORD_FIELDS})
        key = (failure.grid, failure.role)
        if key in seen or key in solves:
            raise LabelPolicyV2Error(
                f"{where}.solve_failures[{index}] duplicates or contradicts a solve"
            )
        expected_criticality = solve_criticality(case, failure.grid, failure.role)
        if failure.criticality != expected_criticality:
            raise LabelPolicyV2Error(
                f"{where}.solve_failures[{index}].criticality must be "
                f"'{expected_criticality}'"
            )
        seen.add(key)
        failures.append(failure)
    return solves, failures


def compare_recomputed_stage(
    report: Mapping[str, Any], recomputed: RecomputedStage, *, where: str
) -> None:
    """Require every published block to equal what the raw numbers imply."""
    published_cases = list(report["cases"])
    for index, (published, rebuilt) in enumerate(
        zip(published_cases, recomputed.cases, strict=True)
    ):
        if canonical_payload(_json_ready(published)) != canonical_payload(_json_ready(rebuilt)):
            differing = sorted(
                key
                for key in CASE_REPORT_KEYS
                if canonical_payload(_json_ready(published.get(key)))
                != canonical_payload(_json_ready(rebuilt.get(key)))
            )
            raise LabelPolicyV2Error(
                f"{where}.cases[{index}] ('{rebuilt['state']['name']}') disagrees with the "
                f"decisions its own raw solve records imply; differing blocks: {differing}"
            )
    for key, rebuilt in (
        ("stage_outcome", recomputed.outcome),
        ("solve_accounting", recomputed.accounting),
        ("lifecycle", recomputed.lifecycle),
    ):
        published = report.get(key)
        if canonical_payload(_json_ready(published)) != canonical_payload(_json_ready(rebuilt)):
            raise LabelPolicyV2Error(
                f"{where}.{key} disagrees with the value recomputed from the case rows"
            )


def _json_ready(value: Any) -> Any:
    """Normalise a value for canonical comparison (tuples become lists)."""
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        return value
    raise LabelPolicyV2Error(f"cannot compare a {type(value).__name__} canonically")


def verify_stage_report(
    report: Mapping[str, Any],
    config: PolicyConfigV2,
    *,
    root: Path = PROJECT_ROOT,
    where: str = "report",
) -> RecomputedStage:
    """Authoritative semantic verification of one v2 stage report.

    Nothing published is trusted. The schema is exact, the provenance digests
    are reconciled against repository files, the criteria block and its digest
    are rebuilt from the externally supplied configuration, and every case
    decision, aggregate count and lifecycle field is recomputed from the raw
    per-solve numbers.

    The stated limit: this detects inconsistent or partial mutation. It cannot
    authenticate a fully coordinated fabricated numerical report, because it
    does not re-solve.
    """
    validate_report_schema(report, where=where)
    validate_report_context(report, config, where=where)
    _reject_unknown_report_keys(report)
    if _require_string(report, "schema_version", where) != REPORT_SCHEMA_VERSION:
        raise LabelPolicyV2Error(f"{where}.schema_version must be '{REPORT_SCHEMA_VERSION}'")

    source = _mapping(report.get("source"), f"{where}.source")
    _exact_keys(source, REPORT_SOURCE_KEYS, f"{where}.source")
    if _require_bool(source, "private_market_data_used", f"{where}.source"):
        raise LabelPolicyV2Error(f"{where}.source.private_market_data_used must be false")
    if source["raw_config_sha256"] != config.raw_config_sha256:
        raise LabelPolicyV2Error(
            f"{where}.source.raw_config_sha256 does not match the supplied configuration"
        )
    inventory = _mapping(source.get("executable_source_inventory"), f"{where}.source")
    if dict(inventory) != dict(EXECUTABLE_SOURCE_INVENTORY):
        raise LabelPolicyV2Error(
            f"{where}.source.executable_source_inventory is not the declared inventory"
        )
    verify_executable_source_digests(source, root=root, where=f"{where}.source")
    if source["source_digest_limitation"] != SOURCE_DIGEST_LIMITATION:
        raise LabelPolicyV2Error(f"{where}.source.source_digest_limitation was edited")

    stage = _report_stage(report, config)
    published_case_count = len(_sequence(report.get("cases"), f"{where}.cases"))
    gate_eligible_case_count = sum(
        1
        for name in config.stage_case_names(stage)
        if config.is_gate_eligible(config.case(name))
    )
    # Every config-derived section is rebuilt and compared as a whole, so an
    # unknown, missing, duplicated or retyped field anywhere inside any of them
    # is rejected -- not only at the top level.
    for key, expected in (
        (
            "study",
            study_block(
                config=config,
                stage=stage,
                case_count=published_case_count,
                gate_eligible_case_count=gate_eligible_case_count,
            ),
        ),
        ("conventions", conventions_block(config)),
        ("criteria_block", criteria_block(config)),
        ("validation_rules", validation_rules_block(config)),
        ("eligibility_contract", eligibility_contract_block()),
        ("richardson_contract", richardson_contract_block(config)),
        ("solver", asdict(config.solver)),
        ("grids", [asdict(grid) for grid in config.grids]),
    ):
        _require_section_matches(report, key, expected, where=where)
    for index, grid in enumerate(_sequence(report["grids"], f"{where}.grids")):
        _exact_keys(
            _mapping(grid, f"{where}.grids[{index}]"),
            GRID_RECORD_KEYS,
            f"{where}.grids[{index}]",
        )
    _validate_performance(report, where=where)
    expected_digest = criteria_digest(config)
    if report.get("criteria_digest") != expected_digest:
        raise LabelPolicyV2Error(f"{where}.criteria_digest does not match the configuration")
    _require_section_matches(
        report, "predeclared_criteria", asdict(config.criteria), where=where
    )
    if list(report.get("non_claims", ())) != list(NON_CLAIMS):
        raise LabelPolicyV2Error(f"{where}.non_claims must be the predeclared statements")

    lifecycle = _mapping(report.get("lifecycle"), f"{where}.lifecycle")
    if lifecycle.get("raw_config_sha256") != config.raw_config_sha256:
        raise LabelPolicyV2Error(f"{where}.lifecycle.raw_config_sha256 does not match")
    if lifecycle.get("criteria_digest") != expected_digest:
        raise LabelPolicyV2Error(f"{where}.lifecycle.criteria_digest does not match")
    if lifecycle.get("case_design_digest") != config.case_design_digest:
        raise LabelPolicyV2Error(f"{where}.lifecycle.case_design_digest does not match")

    recomputed = recompute_stage(report, config)
    compare_recomputed_stage(report, recomputed, where=where)
    return recomputed



def _require_section_matches(
    report: Mapping[str, Any], key: str, expected: Any, *, where: str
) -> None:
    """Require a whole published section to equal what the configuration implies.

    Comparing the canonical payload of the entire section is strictly stronger
    than an exact key set: an unknown key, a missing key, a duplicated entry, a
    retyped value and an edited value are all rejected, at every nesting depth.
    """
    published = report.get(key)
    if canonical_payload(_json_ready(published)) == canonical_payload(_json_ready(expected)):
        return
    detail = ""
    if isinstance(published, Mapping) and isinstance(expected, Mapping):
        unknown = sorted(set(published) - set(expected))
        missing = sorted(set(expected) - set(published))
        differing = sorted(
            name
            for name in set(published) & set(expected)
            if canonical_payload(_json_ready(published[name]))
            != canonical_payload(_json_ready(expected[name]))
        )
        detail = f"; unknown={unknown} missing={missing} differing={differing}"
    raise LabelPolicyV2Error(
        f"{where}.{key} does not match the externally supplied configuration{detail}"
    )


def _validate_performance(report: Mapping[str, Any], *, where: str) -> None:
    """Exact keys and types for the one section that is runtime measurement."""
    performance = _mapping(report.get("performance"), f"{where}.performance")
    _exact_keys(performance, PERFORMANCE_KEYS, f"{where}.performance")
    for key in ("wall_seconds", "timed_solve_seconds"):
        value = performance[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise LabelPolicyV2Error(f"{where}.performance.{key} must be a number")
        if not math.isfinite(float(value)) or float(value) < 0.0:
            raise LabelPolicyV2Error(f"{where}.performance.{key} must be finite and >= 0")
    for key in (
        "timed_solve_attempts",
        "peak_resident_memory_bytes",
        "peak_resident_memory_increase_bytes",
    ):
        value = performance[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise LabelPolicyV2Error(
                f"{where}.performance.{key} must be a non-negative integer"
            )
    _require_string(performance, "projection_caveat", f"{where}.performance")
    host = _mapping(performance["host"], f"{where}.performance.host")
    _exact_keys(host, PERFORMANCE_HOST_KEYS, f"{where}.performance.host")
    for key in sorted(PERFORMANCE_HOST_KEYS):
        _require_string(host, key, f"{where}.performance.host")


# ---------------------------------------------------------------------------
# The exact report schema
# ---------------------------------------------------------------------------

EXERCISE_STATES: Final = (
    "continuation",
    "exercise",
    "numerically_indifferent",
    "no_obstacle",
)
SOLVER_STATUSES: Final = ("discrete_system_converged", "psor_iteration_limit_exceeded")
FIXED_BUMP_BRANCHES: Final = ("flat", "resolved")
VERDICTS: Final = ("pass", "fail")

_DIGEST = StrSpec(hex_digest=True)
#: Prefixed identity digests. Each is `<prefix>-<64 lowercase hex>`, so an
#: unrestricted string is never accepted where an identity belongs.
_CRITERIA_DIGEST = StrSpec(digest_prefix="crit")
_CASE_DESIGN_DIGEST = StrSpec(digest_prefix="cases")
_SOURCE_COMPOSITE_DIGEST = StrSpec(digest_prefix="src")
_PRICING_GRID_IDENTITY = StrSpec(digest_prefix="pg")
_VEGA_CONVENTION_IDENTITY = StrSpec(digest_prefix="vega")
_NAME = StrSpec()
_TEXT = StrSpec()
_FLAG = BoolSpec()
_COUNT = IntSpec(minimum=0)
_NAMES = ArraySpec(_NAME)
_MAYBE_REAL = NullableSpec(RealSpec())
_MAYBE_NAME = NullableSpec(_NAME)

_CHECKS_SPEC = ObjectSpec({name: _FLAG for name in GATING_CHECKS})

_SOLVE_SPEC = ObjectSpec(
    {
        "grid_name": StrSpec(enum=GRID_NAMES),
        "role": _NAME,
        "exercise_style": StrSpec(enum=tuple(sorted(EXERCISE_STYLES))),
        "volatility": RealSpec(minimum=0.0),
        "solver_status": StrSpec(enum=SOLVER_STATUSES),
        "time_steps": IntSpec(minimum=1),
        "maximum_lcp_residual": RealSpec(),
        "spot_step": RealSpec(minimum=0.0),
        "spot_intervals": IntSpec(minimum=1),
        "pricing_grid_identity": _PRICING_GRID_IDENTITY,
        "centre_node_index": _COUNT,
        "centre_spot_is_exact_node": _FLAG,
        "centre_price": RealSpec(),
        "centre_delta": _MAYBE_REAL,
        "centre_gamma": _MAYBE_REAL,
        "centre_greek_eligible": _FLAG,
        "centre_greek_eligibility_reason": _NAME,
        "centre_exercise_state": StrSpec(enum=EXERCISE_STATES),
        "backward_inductions": _COUNT,
        "linear_solves": _COUNT,
        "psor_solves": _COUNT,
        "psor_total_iterations": _COUNT,
        "bumped_prices": MapSpec(RealSpec(), key_enum=CANONICAL_SPOT_BUMP_KEYS),
        "bumped_spots_are_exact_nodes": _FLAG,
    }
)

_SOLVE_FAILURE_SPEC = ObjectSpec(
    {
        "case": _NAME,
        "grid": StrSpec(enum=GRID_NAMES),
        "role": _NAME,
        "option_type": StrSpec(enum=tuple(sorted(OPTION_TYPES))),
        "exercise_style": StrSpec(enum=tuple(sorted(EXERCISE_STYLES))),
        "criticality": StrSpec(enum=SOLVE_CRITICALITIES),
        "exception_class": _NAME,
        "message": StrSpec(allow_empty=True),
    }
)

_FIXED_BUMP_SPEC = ObjectSpec(
    {
        "quantity": StrSpec(enum=("delta", "vega")),
        "branch": NullableSpec(StrSpec(enum=FIXED_BUMP_BRANCHES)),
        "estimate_at_h": _MAYBE_REAL,
        "estimate_at_2h": _MAYBE_REAL,
        "estimate_at_4h": _MAYBE_REAL,
        "delta_small": _MAYBE_REAL,
        "delta_large": _MAYBE_REAL,
        "observed_bump_order": _MAYBE_REAL,
        "charge_factor": RealSpec(minimum=1.0),
        "charge_scheme": StrSpec(enum=(FIXED_BUMP_CHARGE_SCHEME,)),
        "epsilon_small": _MAYBE_REAL,
        "epsilon_large": _MAYBE_REAL,
        "bias_charge": _MAYBE_REAL,
        "reference_estimate": _MAYBE_REAL,
        "reference_absolute_error": _MAYBE_REAL,
        "combined_absolute_error": _MAYBE_REAL,
        "absolute_error_cap": RealSpec(minimum=0.0),
        "checks": ObjectSpec(
            {
                "observed_bump_order_supported": _FLAG,
                "ladder_within_residual_scale": _FLAG,
                "combined_absolute_error_within_cap": _FLAG,
                "residual_scale_within_cap": _FLAG,
            }
        ),
        "numerically_invalid_residual_scale": _FLAG,
        "reasons": _NAMES,
        "verdict": StrSpec(enum=VERDICTS),
        # Present only when the ladder was computable; the runner adds them
        # together, so they are declared nullable rather than optional.
        "bumps": NullableSpec(ArraySpec(RealSpec(minimum=0.0), minimum_length=3)),
        "primary_bump": _MAYBE_REAL,
        "epsilon_definition": _MAYBE_NAME,
    }
)
_FIXED_BUMP_DELTA_SPEC = ObjectSpec(
    {**_FIXED_BUMP_SPEC.fields, "base_price_residual_scale": _MAYBE_REAL}
)
_FIXED_BUMP_VEGA_SPEC = ObjectSpec(
    {
        **_FIXED_BUMP_SPEC.fields,
        "rung_residual_scales": NullableSpec(ArraySpec(RealSpec(), minimum_length=3)),
        "vega_per_volatility_point": _MAYBE_REAL,
    }
)

_GRID_STENCIL_SPEC = ObjectSpec(
    {
        "coarse_stencil_delta": _MAYBE_REAL,
        "candidate_stencil_delta": _MAYBE_REAL,
        "fine_stencil_delta": _MAYBE_REAL,
        "reference_stencil_delta": _MAYBE_REAL,
        "reference_definition": StrSpec(enum=(GRID_STENCIL_REFERENCE_RULE,)),
        "order_definition": StrSpec(enum=(GRID_STENCIL_ORDER_RULE,)),
        "observed_stencil_order": _MAYBE_REAL,
        "absolute_error": _MAYBE_REAL,
        "absolute_error_cap": RealSpec(minimum=0.0),
        "bump_bias_charge_applied": _FLAG,
        "checks": ObjectSpec(
            {
                "observed_stencil_order_supported": _FLAG,
                "absolute_error_within_cap": _FLAG,
            }
        ),
        "reasons": _NAMES,
        "verdict": StrSpec(enum=VERDICTS),
    }
)

_SHAPE_COMPARISON_FIELDS = {
    "applicable": _FLAG,
    "gap": _MAYBE_REAL,
    "scale": _MAYBE_REAL,
    "floor": RealSpec(minimum=0.0),
    "allowance": _MAYBE_REAL,
    "residual_scale_exceeds_price_cap": _FLAG,
    "passes": _FLAG,
}
_SHAPE_SPEC = ObjectSpec(
    {
        "measure": StrSpec(enum=("absolute_maximum_lcp_residual",)),
        "is_a_rigorous_bound": _FLAG,
        "allowance_definition": _TEXT,
        "american_dominance": ObjectSpec(
            {**_SHAPE_COMPARISON_FIELDS, "control_price": _MAYBE_REAL}
        ),
        "american_intrinsic_bound": ObjectSpec(
            {**_SHAPE_COMPARISON_FIELDS, "intrinsic_value": RealSpec(minimum=0.0)}
        ),
        "spot_monotonicity": ObjectSpec(
            {"delta": _MAYBE_REAL, "floor": RealSpec(minimum=0.0), "passes": _FLAG}
        ),
        "spot_convexity": ObjectSpec(
            {"gamma": _MAYBE_REAL, "floor": RealSpec(minimum=0.0), "passes": _FLAG}
        ),
    }
)

_ELIGIBILITY_COMMON = {
    "quantity": _NAME,
    "estimator": _NAME,
    "value": _MAYBE_REAL,
    "residual_scale": _MAYBE_REAL,
    "residual_scale_definition": _TEXT,
    "residual_scale_cap": RealSpec(minimum=0.0),
    "eligible": _FLAG,
    "reasons": _NAMES,
}
_DELTA_ELIGIBILITY_SPEC = ObjectSpec(
    {
        **_ELIGIBILITY_COMMON,
        "predicates": ObjectSpec(
            {
                "structural_c1_greek_eligibility": _FLAG,
                **{name: _FLAG for name in DELTA_ELIGIBILITY_PREDICATES},
            }
        ),
    }
)
_VEGA_ELIGIBILITY_SPEC = ObjectSpec(
    {
        **_ELIGIBILITY_COMMON,
        "predicates": ObjectSpec(
            {
                "structural_c1_greek_eligibility": _FLAG,
                **{name: _FLAG for name in VEGA_ELIGIBILITY_PREDICATES},
            }
        ),
        "vega_per_volatility_point": _MAYBE_REAL,
        "vega_convention_id": NullableSpec(_VEGA_CONVENTION_IDENTITY),
        "expected_vega_convention_id": _VEGA_CONVENTION_IDENTITY,
        "observed_sigma_down": _MAYBE_REAL,
        "observed_sigma_up": _MAYBE_REAL,
        "planned_sigma_down": _MAYBE_REAL,
        "planned_sigma_up": _MAYBE_REAL,
        "vega_bump": RealSpec(minimum=0.0),
        "exercise_states": ObjectSpec(
            {
                "base": NullableSpec(StrSpec(enum=EXERCISE_STATES)),
                "sigma_down": NullableSpec(StrSpec(enum=EXERCISE_STATES)),
                "sigma_up": NullableSpec(StrSpec(enum=EXERCISE_STATES)),
            }
        ),
    }
)

_CASE_STATE_SPEC = ObjectSpec(
    {
        "name": _NAME,
        "classification": StrSpec(enum=tuple(sorted(CLASSIFICATIONS))),
        "description": _TEXT,
        "option_type": StrSpec(enum=tuple(sorted(OPTION_TYPES))),
        "exercise_style": StrSpec(enum=tuple(sorted(EXERCISE_STYLES))),
        "spot": RealSpec(minimum=0.0),
        "strike": RealSpec(minimum=0.0),
        "expiry_time": RealSpec(minimum=0.0),
        "rate": RealSpec(),
        "continuous_carry": RealSpec(),
        "volatility": RealSpec(minimum=0.0),
        "dividends": ArraySpec(ArraySpec(RealSpec(), minimum_length=2)),
        "valuation_time": RealSpec(),
        "curve_times": ArraySpec(RealSpec(), minimum_length=2),
        "curve_log_discounts": ArraySpec(RealSpec(), minimum_length=2),
        "moneyness_spot_over_strike": RealSpec(minimum=0.0),
    }
)

_ANCHOR_SPEC = NullableSpec(
    ObjectSpec(
        {
            "grid": StrSpec(enum=(ANCHOR_GRID,)),
            "role": StrSpec(enum=("retained_anchor_rung",)),
            "anchor_evidence_complete": _FLAG,
            "anchor_price": _MAYBE_REAL,
            "reference_price": _MAYBE_REAL,
            "price_absolute_gap": _MAYBE_REAL,
            "anchor_stencil_delta": _MAYBE_REAL,
            "reference_stencil_delta": _MAYBE_REAL,
            "stencil_delta_absolute_gap": _MAYBE_REAL,
            "gates": _FLAG,
        }
    )
)

_CASE_SPEC = ObjectSpec(
    {
        "state": _CASE_STATE_SPEC,
        "gate_eligible": _FLAG,
        "counts_toward_selection": _FLAG,
        "solve_plan": ArraySpec(_NAME, minimum_length=1),
        "solve_criticality": MapSpec(_NAMES, key_enum=SOLVE_CRITICALITIES),
        "solves": ArraySpec(_SOLVE_SPEC),
        "solve_failures": ArraySpec(_SOLVE_FAILURE_SPEC),
        "solve_problems": MapSpec(_NAMES, key_pattern=_SOLVE_ROLE_KEY),
        "support": ObjectSpec(
            {
                "price_critical_solves_usable": _FLAG,
                "delta_supporting_solves_usable": _FLAG,
                "vega_supporting_solves_usable": _FLAG,
                "anchor_evidence_complete": _FLAG,
            }
        ),
        "price": ObjectSpec(
            {
                "candidate": _MAYBE_REAL,
                "candidate_grid": StrSpec(enum=(CANDIDATE_GRID,)),
                "reference": _MAYBE_REAL,
                "reference_grid": StrSpec(enum=(REFERENCE_GRID,)),
                "reference_method": StrSpec(enum=(PRICE_REFERENCE_METHOD,)),
                "absolute_error": _MAYBE_REAL,
                "absolute_error_cap": RealSpec(minimum=0.0),
            }
        ),
        "gamma_evaluation_only": ObjectSpec(
            {
                "candidate": _MAYBE_REAL,
                "reference": _MAYBE_REAL,
                "absolute_error": _MAYBE_REAL,
                "absolute_error_cap": RealSpec(minimum=0.0),
                "within_cap": NullableSpec(_FLAG),
                "gates": _FLAG,
                "supervision_eligible": _FLAG,
                "reason": StrSpec(enum=("gamma_is_evaluation_only",)),
            }
        ),
        "fixed_bump_delta_validation": _FIXED_BUMP_DELTA_SPEC,
        "fixed_bump_vega_validation": _FIXED_BUMP_VEGA_SPEC,
        "grid_stencil_delta_validation": _GRID_STENCIL_SPEC,
        "shape": _SHAPE_SPEC,
        "anchor_validation": _ANCHOR_SPEC,
        "checks": _CHECKS_SPEC,
        "delta_eligibility": _DELTA_ELIGIBILITY_SPEC,
        "vega_eligibility": _VEGA_ELIGIBILITY_SPEC,
        "passes": _FLAG,
    }
)

_GRID_SPEC = ObjectSpec(
    {
        "name": StrSpec(enum=GRID_NAMES),
        "spot_intervals": IntSpec(minimum=1),
        "time_steps": IntSpec(minimum=1),
        "role": StrSpec(enum=GRID_ROLES),
    }
)
_SOLVER_SPEC = ObjectSpec(
    {
        "spot_maximum": RealSpec(minimum=0.0),
        "rannacher_steps": _COUNT,
        "psor_tolerance": RealSpec(minimum=0.0),
        "psor_relaxation": RealSpec(minimum=0.0, maximum=2.0),
        "psor_maximum_iterations": IntSpec(minimum=1),
        "settlement": StrSpec(enum=("cash", "physical")),
        "contract_multiplier": RealSpec(minimum=0.0),
        "boundary_exclusion_nodes": IntSpec(minimum=1),
    }
)
_CRITERIA_SPEC = ObjectSpec(
    {
        **{name: RealSpec(minimum=0.0) for name in _CRITERIA_FLOAT_KEYS},
        "gamma_is_evaluation_only": _FLAG,
        "fixed_bump_charge_scheme": StrSpec(enum=(FIXED_BUMP_CHARGE_SCHEME,)),
        "delta_flat_epsilon_rule": StrSpec(enum=(DELTA_FLAT_EPSILON_RULE,)),
        "vega_flat_epsilon_rule": StrSpec(enum=(VEGA_FLAT_EPSILON_RULE,)),
        "grid_stencil_reference_rule": StrSpec(enum=(GRID_STENCIL_REFERENCE_RULE,)),
        "grid_stencil_order_rule": StrSpec(enum=(GRID_STENCIL_ORDER_RULE,)),
        "grid_stencil_bump_bias_charge_applied": _FLAG,
    }
)
_RICHARDSON_SPEC = ObjectSpec(
    {
        "role": StrSpec(enum=("reference_validation_only",)),
        "coarse_to_fine_ratio": IntSpec(minimum=1),
        "assumed_order": RealSpec(minimum=0.0),
        "minimum_supported_observed_order": RealSpec(),
        "maximum_supported_observed_order": RealSpec(),
    }
)
_CRITERIA_BLOCK_SPEC = ObjectSpec(
    {
        "scheme": StrSpec(enum=(CRITERIA_SCHEME_VERSION,)),
        "criteria": _CRITERIA_SPEC,
        "selection_order": _NAMES,
        "order_probe_grids": _NAMES,
        "grids": ArraySpec(_GRID_SPEC, minimum_length=4),
        "bumps": ObjectSpec(
            {
                "spot": ArraySpec(RealSpec(minimum=0.0), minimum_length=3),
                "primary_spot": RealSpec(minimum=0.0),
                "volatility": ArraySpec(RealSpec(minimum=0.0), minimum_length=3),
                "primary_volatility": RealSpec(minimum=0.0),
                "ladder_ratio": IntSpec(minimum=1),
            }
        ),
        "richardson": _RICHARDSON_SPEC,
        "solver": ObjectSpec(
            {
                key: spec
                for key, spec in _SOLVER_SPEC.fields.items()
                if key != "contract_multiplier"
            }
        ),
        "gating_checks": _NAMES,
        "non_gating_published_checks": _NAMES,
        "price_critical_solves": _NAMES,
        "delta_only_solves": _NAMES,
        "vega_only_solves": _NAMES,
        "anchor_descriptive_solves": _NAMES,
        "common_predicates": _NAMES,
        "structural_c1_applies_to": _NAMES,
        "delta_predicates": _NAMES,
        "vega_predicates": _NAMES,
        "gamma_eligibility": StrSpec(enum=("never",)),
        "study_name": StrSpec(enum=(STUDY_NAME,)),
        "delta_method": StrSpec(enum=(DELTA_METHOD,)),
        "vega_method": StrSpec(enum=(VEGA_METHOD,)),
        "price_reference_method": StrSpec(enum=(PRICE_REFERENCE_METHOD,)),
        "price_reference_grid": StrSpec(enum=(REFERENCE_GRID,)),
        "dominance_control": _NAME,
        "dominance_residual_measure": _NAME,
        "fixed_bump_charge_factor": RealSpec(minimum=1.0),
        "fixed_bump_charge_scheme": StrSpec(enum=(FIXED_BUMP_CHARGE_SCHEME,)),
        "supported_order_band": ArraySpec(RealSpec(), minimum_length=2),
        "case_design_digest": _CASE_DESIGN_DIGEST,
        "remediation_cases": _NAMES,
        "gate_cases": _NAMES,
        "non_gating_remediation_cases": _NAMES,
        "anchor_cases": _NAMES,
        "lifecycle": ObjectSpec(
            {
                "stages": _NAMES,
                "remediation_statuses": _NAMES,
                "confirmation_statuses": _NAMES,
                "selection_requires_fresh_top_level_approval": _FLAG,
                "authorizes_dataset_generation": _FLAG,
                "authorizes_training_input": _FLAG,
            }
        ),
    }
)

_LIFECYCLE_SPEC = ObjectSpec(
    {
        "stage": StrSpec(enum=STAGES),
        "raw_config_sha256": _DIGEST,
        "criteria_digest": _CRITERIA_DIGEST,
        "case_design_digest": _CASE_DESIGN_DIGEST,
        "protocol": StrSpec(enum=("one_shot_predeclared",)),
        "criteria_were_not_loosened": _FLAG,
        "one_shot_enforcement": StrSpec(enum=("procedural_and_provenance_backed",)),
        "authorizes_dataset_generation": _FLAG,
        "authorizes_training_input": _FLAG,
        "remediation_status": StrSpec(enum=REMEDIATION_STATUSES),
        "confirmation_status": StrSpec(enum=CONFIRMATION_STATUSES),
        "confirmation_not_run_reason": NullableSpec(_NAME),
        "terminal": _FLAG,
        "freezable": _FLAG,
    }
)
_TERMINAL_LIFECYCLE_SPEC = ObjectSpec(
    {
        **_LIFECYCLE_SPEC.fields,
        "selected_accuracy_policy": StrSpec(enum=(NO_POLICY_SELECTED, CANDIDATE_GRID)),
        "selection_pending_fresh_top_level_approval": _FLAG,
    }
)

_SOURCE_SPEC = ObjectSpec(
    {
        "config_name": _NAME,
        "raw_config_sha256": _DIGEST,
        **{f"{name}_sha256": _DIGEST for name in EXECUTABLE_SOURCE_INVENTORY},
        "executable_source_inventory": ObjectSpec(
            {name: _NAME for name in EXECUTABLE_SOURCE_INVENTORY}
        ),
        "executable_source_composite_digest": _SOURCE_COMPOSITE_DIGEST,
        "reported_binding_pde_header_sha256": _DIGEST,
        "reported_binding_pde_source_sha256": _DIGEST,
        "reported_binding_pde_implementation_sha256": _DIGEST,
        "source_digest_limitation": _TEXT,
        "build_configuration": _TEXT,
        "cxx_compiler": _TEXT,
        "data_provenance": _NAME,
        "private_market_data_used": _FLAG,
    }
)

_STAGE_OUTCOME_SPEC = ObjectSpec(
    {
        "candidate": StrSpec(enum=(CANDIDATE_GRID,)),
        "gate_eligible_case_count": _COUNT,
        "gate_eligible_failure_count": _COUNT,
        "gate_eligible_failures": _NAMES,
        "all_gate_eligible_cases_pass": _FLAG,
        "failed_checks_by_name": MapSpec(_NAMES, key_enum=GATING_CHECKS),
        "descriptive_failure_count": _COUNT,
        "descriptive_failures": _NAMES,
        "delta_eligible_cases": _NAMES,
        "vega_eligible_cases": _NAMES,
        "gamma_eligible_cases": _NAMES,
        "anchor_evidence_incomplete_cases": _NAMES,
        "selection_order": _NAMES,
        "regular_cases_only_decide_selection": _FLAG,
        "stress_case_is_descriptive_anchor_evidence": _FLAG,
        "criteria_were_not_loosened": _FLAG,
    }
)

_SOLVE_ACCOUNTING_SPEC = ObjectSpec(
    {
        "planned_surface_solves": _COUNT,
        "attempted_surface_solves": _COUNT,
        "completed_surface_solves": _COUNT,
        "solver_exceptions": _COUNT,
        "backward_inductions": _COUNT,
        "linear_solves": _COUNT,
        "psor_solves": _COUNT,
        "psor_total_iterations": _COUNT,
        "attempted_by_solve_role": MapSpec(_COUNT, key_pattern=_SOLVE_ROLE_KEY),
        "attempted_by_criticality": MapSpec(_COUNT, key_enum=SOLVE_CRITICALITIES),
        "attempted_by_case_classification": MapSpec(
            _COUNT, key_enum=tuple(sorted(CLASSIFICATIONS))
        ),
        "reconciliation": ObjectSpec(
            {
                "planned_equals_attempted": _FLAG,
                "attempted_equals_completed_plus_exceptions": _FLAG,
                "backward_inductions_equal_completed_solves": _FLAG,
            }
        ),
    }
)

PERFORMANCE_SPEC: Final = ObjectSpec(
    {
        "wall_seconds": RealSpec(minimum=0.0),
        "timed_solve_attempts": _COUNT,
        "timed_solve_seconds": RealSpec(minimum=0.0),
        "peak_resident_memory_bytes": _COUNT,
        "peak_resident_memory_increase_bytes": _COUNT,
        "projection_caveat": _TEXT,
        "host": ObjectSpec({"system": _TEXT, "machine": _TEXT, "python": _TEXT}),
    }
)

#: The complete serialized v2 stage report. Every leaf has an exact type; a
#: `null` is accepted only where `NullableSpec` says so.
REPORT_SCHEMA: Final = ObjectSpec(
    {
        "schema_version": StrSpec(enum=(REPORT_SCHEMA_VERSION,)),
        "lifecycle": _LIFECYCLE_SPEC,
        "study": ObjectSpec(
            {
                "name": StrSpec(enum=(STUDY_NAME,)),
                "stage": StrSpec(enum=STAGES),
                "objective": _NAME,
                "case_count": IntSpec(minimum=1),
                "gate_eligible_case_count": _COUNT,
                "remediation_cases": _NAMES,
                "gate_cases": _NAMES,
                "non_gating_remediation_cases": _NAMES,
                "anchor_cases": _NAMES,
                "confirmation_case_count": IntSpec(minimum=1),
                "expensive_stage_is_excluded_from_ci": _FLAG,
            }
        ),
        "conventions": ObjectSpec(
            {
                "value": _NAME,
                "delta": _NAME,
                "gamma": _NAME,
                "vega_per_unit_volatility": _TEXT,
                "vega_per_volatility_point": _NAME,
                "spot_bumps": ArraySpec(RealSpec(minimum=0.0), minimum_length=3),
                "primary_spot_bump": RealSpec(minimum=0.0),
                "volatility_bumps": ArraySpec(RealSpec(minimum=0.0), minimum_length=3),
                "primary_volatility_bump": RealSpec(minimum=0.0),
                "delta_method": StrSpec(enum=(DELTA_METHOD,)),
                "vega_method": StrSpec(enum=(VEGA_METHOD,)),
                "production_delta_estimator": _NAME,
                "price_candidate": _NAME,
                "price_reference_method": StrSpec(enum=(PRICE_REFERENCE_METHOD,)),
                "price_reference_grid": StrSpec(enum=(REFERENCE_GRID,)),
                "price_reference_is_richardson_extrapolated": _FLAG,
                "price_reference_has_conditional_fallback": _FLAG,
                "order_probe_is_price_irrelevant": _FLAG,
                "node_source": _NAME,
                "theta_and_rho_implemented": _FLAG,
            }
        ),
        "predeclared_criteria": _CRITERIA_SPEC,
        "criteria_block": _CRITERIA_BLOCK_SPEC,
        "criteria_digest": _CRITERIA_DIGEST,
        "validation_rules": ObjectSpec(
            {
                "fixed_bump": ObjectSpec(
                    {
                        "charge_factor": RealSpec(minimum=1.0),
                        "charge_scheme": StrSpec(enum=(FIXED_BUMP_CHARGE_SCHEME,)),
                        "combined_budget": _TEXT,
                        "delta_epsilon_rule": StrSpec(enum=(DELTA_FLAT_EPSILON_RULE,)),
                        "vega_epsilon_rule": StrSpec(enum=(VEGA_FLAT_EPSILON_RULE,)),
                        "resolved_branch": _TEXT,
                        "flat_branch": _TEXT,
                        "supported_order_band": ArraySpec(RealSpec(), minimum_length=2),
                    }
                ),
                "grid_stencil": ObjectSpec(
                    {
                        "candidate": _TEXT,
                        "reference": StrSpec(enum=(GRID_STENCIL_REFERENCE_RULE,)),
                        "order_probe": StrSpec(enum=(GRID_STENCIL_ORDER_RULE,)),
                        "order_probe_grid": StrSpec(enum=(ORDER_PROBE_GRID,)),
                        "supported_order_band": ArraySpec(RealSpec(), minimum_length=2),
                        "bump_bias_charge_applied": _FLAG,
                    }
                ),
                "price": ObjectSpec(
                    {
                        "candidate": _NAME,
                        "reference_method": StrSpec(enum=(PRICE_REFERENCE_METHOD,)),
                        "reference_grid": StrSpec(enum=(REFERENCE_GRID,)),
                        "richardson_extrapolated": _FLAG,
                        "conditional_fallback": _FLAG,
                        "absolute_error_cap": RealSpec(minimum=0.0),
                    }
                ),
                "shape": ObjectSpec(
                    {
                        "control": _NAME,
                        "residual_measure": _NAME,
                        "is_a_rigorous_bound": _FLAG,
                        "applies_to": _NAMES,
                    }
                ),
                "gating_checks": _NAMES,
                "non_gating_published_checks": _NAMES,
                "solve_criticality_classes": _NAMES,
                "price_critical_solves": _NAMES,
                "delta_only_solves": _NAMES,
                "vega_only_solves": _NAMES,
                "anchor_descriptive_solves": _NAMES,
            }
        ),
        "eligibility_contract": ObjectSpec(
            {
                "common_predicates": _NAMES,
                "structural_c1_applies_to": _NAMES,
                "delta_predicates": _NAMES,
                "vega_predicates": _NAMES,
                "gamma_eligibility": StrSpec(enum=("never",)),
                "identity_excludes_contract_multiplier": _FLAG,
                "identity_includes_settlement": _FLAG,
                "authorized_training_input_statuses": ArraySpec(_NAME),
            }
        ),
        "richardson_contract": ObjectSpec(
            {
                **_RICHARDSON_SPEC.fields,
                "formula": _TEXT,
                "is_never_a_candidate_label_policy": _FLAG,
                "applies_to": _NAMES,
                "applies_to_price_reference": _FLAG,
            }
        ),
        "source": _SOURCE_SPEC,
        "solver": _SOLVER_SPEC,
        "grids": ArraySpec(_GRID_SPEC, minimum_length=4),
        "cases": ArraySpec(_CASE_SPEC, minimum_length=1),
        "stage_outcome": _STAGE_OUTCOME_SPEC,
        "solve_accounting": _SOLVE_ACCOUNTING_SPEC,
        "non_claims": ArraySpec(_TEXT, minimum_length=1),
        "performance": PERFORMANCE_SPEC,
    }
)
#: The same report with the terminal lifecycle variant, which carries the two
#: selection fields the nonterminal state must not have.
TERMINAL_REPORT_SCHEMA: Final = ObjectSpec(
    {**REPORT_SCHEMA.fields, "lifecycle": _TERMINAL_LIFECYCLE_SPEC}
)


def validate_report_schema(report: Any, *, where: str = "report") -> None:
    """Exact recursive type validation of a serialized stage report.

    Runs before any ``SurfaceSolve``, case object or confirmation decision is
    constructed. The nonterminal remediation state carries no selected-policy
    field, so the schema is chosen by the lifecycle's own status pair before
    anything else is read.
    """
    if type(report) is not dict:
        raise _schema_error(where, "object", report)
    lifecycle = report.get("lifecycle")
    terminal = not (
        type(lifecycle) is dict
        and lifecycle.get("remediation_status") == "passed"
        and lifecycle.get("confirmation_status") == "pending"
    )
    schema = TERMINAL_REPORT_SCHEMA if terminal else REPORT_SCHEMA
    validate_against_schema(schema, report, where=where)



# ---------------------------------------------------------------------------
# Contextual validation: exact key sets a static schema cannot know
# ---------------------------------------------------------------------------
#
# The static schema bounds every map key to a vocabulary or a syntax. Some maps
# additionally have an *exact* required key set that depends on the case, the
# grid and the solve role, which only the versioned configuration can supply.
# Those are validated here, at the admission boundary, before any dataclass is
# constructed, any stage is run and any solver is called.

#: Every map whose exact key set is derived from the configuration rather than
#: bounded by the static schema alone. The key-closure audit requires every
#: `MapSpec` to be either statically constrained or registered here.
CONTEXTUAL_MAP_PATHS: Final = frozenset(
    {
        ".cases[].solves[].bumped_prices",
        ".cases[].solve_criticality",
        ".cases[].solve_problems",
        ".stage_outcome.failed_checks_by_name",
        ".solve_accounting.attempted_by_solve_role",
        ".solve_accounting.attempted_by_criticality",
        ".solve_accounting.attempted_by_case_classification",
    }
)

#: Every map whose *exact contents* — key set and value at each key — follow
#: from the configuration and the raw records. For these, an enum or pattern
#: key constraint is not sufficient: bounding the vocabulary still admits a
#: valid-vocabulary key with a fabricated value, or the removal of a key that
#: had to be there. Each must therefore also be reconciled exactly, at the
#: admission boundary, which is why this set is required to be a subset of
#: :data:`CONTEXTUAL_MAP_PATHS`.
EXACTLY_DERIVABLE_MAP_PATHS: Final = frozenset(
    {
        ".cases[].solves[].bumped_prices",
        ".cases[].solve_criticality",
        ".cases[].solve_problems",
        ".stage_outcome.failed_checks_by_name",
        ".solve_accounting.attempted_by_solve_role",
        ".solve_accounting.attempted_by_criticality",
        ".solve_accounting.attempted_by_case_classification",
    }
)

if not EXACTLY_DERIVABLE_MAP_PATHS <= CONTEXTUAL_MAP_PATHS:  # pragma: no cover
    raise LabelPolicyV2Error(
        "a map whose exact contents are derivable is not contextually reconciled: "
        f"{sorted(EXACTLY_DERIVABLE_MAP_PATHS - CONTEXTUAL_MAP_PATHS)}"
    )


def expected_bumped_price_keys(
    config: PolicyConfigV2, grid_name: str, role: str
) -> frozenset[str]:
    """The exact ``bumped_prices`` key set one solve role must carry.

    A base surface on the candidate or reference grid carries the whole spot
    ladder, because those bumped spots are exact nodes of that same solve.
    Every other role — the order probe, the volatility ladder, the European
    dominance control, the anchor rung — carries none.
    """
    if role == BASE_ROLE and grid_name in {CANDIDATE_GRID, REFERENCE_GRID}:
        return frozenset(
            spot_bump_key(bump, direction)
            for bump in config.bumps.spot
            for direction in ("down", "up")
        )
    return frozenset()


def _require_exact_keys(
    published: Mapping[str, Any], expected: AbstractSet[str], *, where: str, subject: str
) -> None:
    unexpected = sorted(set(published) - set(expected))
    missing = sorted(set(expected) - set(published))
    if unexpected or missing:
        raise LabelPolicyV2Error(
            f"{where}: {subject} key set is wrong; unexpected={unexpected} "
            f"missing={missing} expected={sorted(expected)}"
        )


def _require_exact_map(
    published: Mapping[str, Any], expected: Mapping[str, Any], *, where: str, subject: str
) -> None:
    """Require a published map to equal, key for key and value for value, the
    map derived from the configuration and the raw records."""
    _require_exact_keys(published, frozenset(expected), where=where, subject=subject)
    wrong = sorted(key for key in expected if published[key] != expected[key])
    if wrong:
        detail = "; ".join(
            f"{key}: published={published[key]!r} derived={expected[key]!r}" for key in wrong
        )
        raise LabelPolicyV2Error(
            f"{where}: {subject} value is wrong at {wrong}; {detail}"
        )


def validate_solve_record_context(
    record: Mapping[str, Any], config: PolicyConfigV2, *, where: str
) -> None:
    """Require one serialized solve's ``bumped_prices`` to match its own role."""
    grid_name = str(record["grid_name"])
    role = str(record["role"])
    _require_exact_keys(
        record["bumped_prices"],
        expected_bumped_price_keys(config, grid_name, role),
        where=f"{where} (grid '{grid_name}', role '{role}').bumped_prices",
        subject="bumped_prices",
    )


def validate_report_context(
    report: Mapping[str, Any], config: PolicyConfigV2, *, where: str = "report"
) -> None:
    """Validate every context-derived key set in a serialized stage report.

    Runs immediately after the static schema and before anything is
    constructed from the document.
    """
    stage = _require_string(
        _mapping(report.get("study"), f"{where}.study"), "stage", f"{where}.study"
    )
    if stage not in STAGES:
        raise LabelPolicyV2Error(f"{where}.study.stage must be one of {list(STAGES)}")

    stage_roles: set[str] = set()
    stage_criticalities: set[str] = set()
    case_rows: list[Mapping[str, Any]] = []
    for index, entry in enumerate(_sequence(report.get("cases"), f"{where}.cases")):
        case_where = f"{where}.cases[{index}]"
        row = _mapping(entry, case_where)
        case_rows.append(row)
        name = _require_string(
            _mapping(row.get("state"), f"{case_where}.state"), "name", "state"
        )
        case = config.case(name)
        required = required_solve_keys(case, config)
        plan = [f"{grid}/{role}" for grid, role in required]
        stage_roles.update(plan)

        criticalities: dict[str, list[str]] = {}
        for grid_name, role in required:
            criticalities.setdefault(
                solve_criticality(case, grid_name, role), []
            ).append(f"{grid_name}/{role}")
        stage_criticalities.update(criticalities)
        _require_exact_keys(
            _mapping(row.get("solve_criticality"), f"{case_where}.solve_criticality"),
            frozenset(criticalities),
            where=f"{case_where}.solve_criticality",
            subject="solve_criticality",
        )

        problems = _mapping(row.get("solve_problems"), f"{case_where}.solve_problems")
        unexpected = sorted(set(problems) - set(plan))
        if unexpected:
            raise LabelPolicyV2Error(
                f"{case_where}.solve_problems names solves outside this case's plan: "
                f"{unexpected}; the plan is {plan}"
            )

        records: dict[tuple[str, str], Mapping[str, Any]] = {}
        for position, solve in enumerate(
            _sequence(row.get("solves"), f"{case_where}.solves")
        ):
            record = _mapping(solve, f"{case_where}.solves[{position}]")
            validate_solve_record_context(
                record, config, where=f"{case_where}.solves[{position}]"
            )
            key = (str(record["grid_name"]), str(record["role"]))
            if key in records:
                raise LabelPolicyV2Error(
                    f"{case_where}.solves publishes '{key[0]}/{key[1]}' more than once"
                )
            records[key] = record
        off_plan = sorted(f"{grid}/{role}" for grid, role in set(records) - set(required))
        if off_plan:
            raise LabelPolicyV2Error(
                f"{case_where}.solves publishes solves outside this case's plan: "
                f"{off_plan}; the plan is {plan}"
            )

        failed_keys: set[tuple[str, str]] = set()
        for position, failure in enumerate(
            _sequence(row.get("solve_failures"), f"{case_where}.solve_failures")
        ):
            entry_map = _mapping(failure, f"{case_where}.solve_failures[{position}]")
            failed_keys.add((str(entry_map["grid"]), str(entry_map["role"])))
        off_plan = sorted(f"{grid}/{role}" for grid, role in failed_keys - set(required))
        if off_plan:
            raise LabelPolicyV2Error(
                f"{case_where}.solve_failures names solves outside this case's plan: "
                f"{off_plan}; the plan is {plan}"
            )

        # Exact reconciliation, from the raw solve, failure and configuration
        # records only, before any dataclass exists.
        _require_exact_map(
            problems,
            raw_solve_problems(case, config, records, failed_keys),
            where=f"{case_where}.solve_problems",
            subject="solve_problems",
        )

    accounting = _mapping(report.get("solve_accounting"), f"{where}.solve_accounting")
    _require_exact_keys(
        _mapping(
            accounting.get("attempted_by_solve_role"),
            f"{where}.solve_accounting.attempted_by_solve_role",
        ),
        frozenset(stage_roles),
        where=f"{where}.solve_accounting.attempted_by_solve_role",
        subject="attempted_by_solve_role",
    )
    _require_exact_keys(
        _mapping(
            accounting.get("attempted_by_criticality"),
            f"{where}.solve_accounting.attempted_by_criticality",
        ),
        frozenset(stage_criticalities),
        where=f"{where}.solve_accounting.attempted_by_criticality",
        subject="attempted_by_criticality",
    )
    _require_exact_map(
        _mapping(
            accounting.get("attempted_by_case_classification"),
            f"{where}.solve_accounting.attempted_by_case_classification",
        ),
        raw_attempted_by_case_classification(case_rows, config),
        where=f"{where}.solve_accounting.attempted_by_case_classification",
        subject="attempted_by_case_classification",
    )
    outcome = _mapping(report.get("stage_outcome"), f"{where}.stage_outcome")
    _require_exact_map(
        _mapping(
            outcome.get("failed_checks_by_name"),
            f"{where}.stage_outcome.failed_checks_by_name",
        ),
        raw_failed_checks_by_name(case_rows, config),
        where=f"{where}.stage_outcome.failed_checks_by_name",
        subject="failed_checks_by_name",
    )


def require_confirmation_entry(
    remediation_report: Mapping[str, Any],
    config: PolicyConfigV2,
    *,
    root: Path = PROJECT_ROOT,
) -> RecomputedStage:
    """Refuse to start confirmation unless a *recomputed* remediation passed.

    Six lifecycle fields are not trusted. The whole remediation report is
    semantically verified first — schema, provenance digests, criteria block
    and digest, every per-case check, every aggregate, and the lifecycle
    itself, all recomputed from raw numbers — and only the recomputed
    remediation status decides. A report whose lifecycle claims
    ``passed``/``pending`` while its own case rows imply ``failed``/``not_run``
    is rejected here, before any solver call.
    """
    recomputed = verify_stage_report(
        remediation_report, config, root=root, where="remediation report"
    )
    if recomputed.stage != "remediation":
        raise LabelPolicyV2Error(
            "confirmation requires a remediation-stage report, got stage "
            f"'{recomputed.stage}'"
        )
    if recomputed.lifecycle["remediation_status"] != "passed":
        raise LabelPolicyV2Error(
            "confirmation is prohibited: the recomputed remediation stage did not pass "
            f"(remediation_status = '{recomputed.lifecycle['remediation_status']}')"
        )
    if recomputed.lifecycle["confirmation_status"] != "pending":
        raise LabelPolicyV2Error(
            "confirmation is prohibited: the recomputed remediation state is not "
            f"awaiting confirmation ('{recomputed.lifecycle['confirmation_status']}')"
        )
    published_cases = {
        str(row["state"]["name"]) for row in remediation_report["cases"]
    }
    if published_cases != set(config.remediation_cases):
        raise LabelPolicyV2Error(
            "confirmation is prohibited: the remediation report does not cover the "
            "predeclared ten-case remediation set"
        )
    return recomputed


# ---------------------------------------------------------------------------
# The expensive path
# ---------------------------------------------------------------------------


def _sanitize_message(text: str) -> str:
    """Collapse an exception message to one bounded, path-free line."""
    flattened = " ".join(str(text).split())
    root = str(PROJECT_ROOT)
    flattened = flattened.replace(root, "<repo>")
    return flattened[:400]


def collect_case_solves(
    case: CaseV2, config: PolicyConfigV2, *, solver: Any = pde_valuation_surface
) -> tuple[dict[tuple[str, str], SurfaceSolve], list[SolveFailure], list[dict[str, Any]]]:
    """Run every surface solve one case needs, recording failures structurally.

    One base surface per grid supplies the centre price, the nodewise stencil
    delta/gamma and every spot-bumped price, because each bumped spot is an
    exact node of that same grid. Only the volatility ladder and the centre
    European dominance control need further solves.

    An exception from the binding is caught here, recorded with its case,
    grid/rung, solve role, contract type, exception class, sanitized message
    and criticality, and the stage continues so a complete report is emitted.
    """
    solves: dict[tuple[str, str], SurfaceSolve] = {}
    failures: list[SolveFailure] = []
    timings: list[dict[str, Any]] = []
    for grid_name, role in required_solve_keys(case, config):
        grid = config.grid(grid_name)
        exercise_style, volatility = planned_role_state(case, role)
        criticality = solve_criticality(case, grid_name, role)
        started = time.perf_counter()
        try:
            solve = _solve_surface(
                case,
                config,
                grid,
                role=role,
                exercise_style=exercise_style,
                volatility=volatility,
                solver=solver,
            )
        except BaseException as error:
            # Recorded structurally and never swallowed silently: the stage must
            # still emit a complete report.
            if isinstance(error, (KeyboardInterrupt, SystemExit, MemoryError)):
                raise
            failures.append(
                SolveFailure(
                    case=case.name,
                    grid=grid_name,
                    role=role,
                    option_type=case.option_type,
                    exercise_style=exercise_style,
                    criticality=criticality,
                    exception_class=type(error).__name__,
                    message=_sanitize_message(str(error)),
                )
            )
        else:
            solves[(grid_name, role)] = solve
        timings.append(
            {
                "case": case.name,
                "grid": grid_name,
                "role": role,
                "criticality": criticality,
                "seconds": time.perf_counter() - started,
            }
        )
    return solves, failures, timings


def _solve_surface(
    case: CaseV2,
    config: PolicyConfigV2,
    grid: GridV2,
    *,
    role: str,
    exercise_style: str,
    volatility: float,
    solver: Any,
) -> SurfaceSolve:
    result = solver(
        option_type=case.option_type,
        exercise_style=exercise_style,
        strike=case.strike,
        valuation_time=0.0,
        expiry_time=case.expiry_time,
        volatility=volatility,
        continuous_carry=case.continuous_carry,
        curve_times=[0.0, case.expiry_time],
        curve_log_discounts=[0.0, -case.rate * case.expiry_time],
        dividends=[list(dividend) for dividend in case.dividends],
        settlement=config.solver.settlement,
        contract_multiplier=config.solver.contract_multiplier,
        spot_intervals=grid.spot_intervals,
        time_steps=grid.time_steps,
        spot_maximum=config.solver.spot_maximum,
        rannacher_steps=config.solver.rannacher_steps,
        psor_tolerance=config.solver.psor_tolerance,
        psor_relaxation=config.solver.psor_relaxation,
        psor_maximum_iterations=config.solver.psor_maximum_iterations,
        boundary_exclusion_nodes=config.solver.boundary_exclusion_nodes,
        query_spots=[],
    )
    return surface_solve_from_result(case, config, grid, result, role=role)


def surface_solve_from_result(
    case: CaseV2,
    config: PolicyConfigV2,
    grid: GridV2,
    result: Mapping[str, Any],
    *,
    role: str,
) -> SurfaceSolve:
    """Reduce one returned surface to the fields the v2 criteria consume."""
    echo = result.get("surface_input")
    if not isinstance(echo, Mapping):
        raise LabelPolicyV2Error(
            f"case '{case.name}' grid '{grid.name}' role '{role}' returned no surface_input echo"
        )
    spots = list(result["spot_nodes"])
    values = list(result["values"])
    deltas = list(result["deltas"])
    gammas = list(result["gammas"])
    eligible = list(result["greek_eligible"])
    reasons = list(result["greek_eligibility_reasons"])
    states = list(result["exercise_states"])
    step = float(result["spot_step"])

    centre_index, centre_exact = _exact_node(case.spot, spots, step)
    bumped: dict[str, float] = {}
    bumps_exact = True
    if role == BASE_ROLE and grid.name in {CANDIDATE_GRID, REFERENCE_GRID}:
        for bump in config.bumps.spot:
            for direction, sign in (("down", -1.0), ("up", 1.0)):
                index, exact = _exact_node(case.spot + sign * bump, spots, step)
                bumps_exact = bumps_exact and exact
                if exact:
                    bumped[spot_bump_key(bump, direction)] = float(values[index])
    return SurfaceSolve(
        grid_name=grid.name,
        role=role,
        exercise_style=str(echo["exercise_style"]),
        volatility=float(echo["volatility"]),
        solver_status=str(result["solver_status"]),
        time_steps=int(result["time_steps"]),
        maximum_lcp_residual=float(result["maximum_lcp_residual"]),
        spot_step=step,
        spot_intervals=int(result["spot_intervals"]),
        pricing_grid_identity=pricing_grid_identity(echo),
        centre_node_index=centre_index,
        centre_spot_is_exact_node=centre_exact,
        centre_price=float(values[centre_index]),
        centre_delta=None if deltas[centre_index] is None else float(deltas[centre_index]),
        centre_gamma=None if gammas[centre_index] is None else float(gammas[centre_index]),
        centre_greek_eligible=bool(eligible[centre_index]),
        centre_greek_eligibility_reason=str(reasons[centre_index]),
        centre_exercise_state=str(states[centre_index]),
        backward_inductions=int(result["backward_inductions"]),
        linear_solves=int(result["linear_solves"]),
        psor_solves=int(result["psor_solves"]),
        psor_total_iterations=int(result["psor_total_iterations"]),
        bumped_prices=bumped,
        bumped_spots_are_exact_nodes=bumps_exact,
    )


def _exact_node(spot: float, spot_nodes: Sequence[float], spot_step: float) -> tuple[int, bool]:
    """Locate a spot on the solved grid, requiring a bitwise-equal node."""
    index = round(spot / spot_step)
    if not 0 <= index < len(spot_nodes):
        raise LabelPolicyV2Error(f"spot {spot!r} lies outside the solved grid")
    return index, spot_nodes[index] == spot


def run_stage(
    config: PolicyConfigV2, *, stage: str, solver: Any = pde_valuation_surface
) -> dict[str, Any]:
    """Run one stage end to end and return its report. Expensive; manual only."""
    if stage not in STAGES:
        raise LabelPolicyV2Error(f"unknown stage '{stage}'")
    started = time.perf_counter()
    initial_peak = _peak_rss_bytes()
    timings: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for name in config.stage_case_names(stage):
        case = config.case(name)
        solves, failures, case_timings = collect_case_solves(case, config, solver=solver)
        timings.extend(case_timings)
        rows.append(evaluate_case(case, config, solves, failures))
    outcome = stage_outcome(rows, config)
    accounting = solve_accounting(rows, config, stage)
    lifecycle = lifecycle_block(
        stage=stage,
        all_gate_eligible_cases_pass=bool(outcome["all_gate_eligible_cases_pass"]),
        config=config,
    )
    return build_report(
        config=config,
        stage=stage,
        cases=rows,
        outcome=outcome,
        accounting=accounting,
        lifecycle=lifecycle,
        performance=_performance(timings=timings, started=started, initial_peak=initial_peak),
    )


def study_block(
    *, config: PolicyConfigV2, stage: str, case_count: int, gate_eligible_case_count: int
) -> dict[str, Any]:
    """The report's `study` section, derived from the configuration alone."""
    return {
        "name": STUDY_NAME,
        "stage": stage,
        "objective": "predeclared_pde_label_policy_v2_stability_and_shape_criterion",
        "case_count": case_count,
        "gate_eligible_case_count": gate_eligible_case_count,
        "remediation_cases": list(config.remediation_cases),
        "gate_cases": list(config.gate_cases),
        "non_gating_remediation_cases": list(config.non_gating_remediation_cases),
        "anchor_cases": list(config.anchor_cases),
        "confirmation_case_count": config.confirmation_case_count,
        "expensive_stage_is_excluded_from_ci": True,
    }


def conventions_block(config: PolicyConfigV2) -> dict[str, Any]:
    """The report's `conventions` section, derived from the configuration."""
    return {
        "value": "currency_per_share",
        "delta": "dV/dS",
        "gamma": "d2V/dS2_per_currency_unit",
        "vega_per_unit_volatility": "dV/dsigma where sigma is absolute volatility",
        "vega_per_volatility_point": "vega_per_unit_volatility/100",
        "spot_bumps": list(config.bumps.spot),
        "primary_spot_bump": config.bumps.primary_spot,
        "volatility_bumps": list(config.bumps.volatility),
        "primary_volatility_bump": config.bumps.primary_volatility,
        "delta_method": DELTA_METHOD,
        "vega_method": VEGA_METHOD,
        "production_delta_estimator": "nodewise_grid_stencil_on_candidate_grid",
        "price_candidate": f"exact_node_centre_value_on_{CANDIDATE_GRID}",
        "price_reference_method": PRICE_REFERENCE_METHOD,
        "price_reference_grid": REFERENCE_GRID,
        "price_reference_is_richardson_extrapolated": False,
        "price_reference_has_conditional_fallback": False,
        "order_probe_is_price_irrelevant": True,
        "node_source": "exact_grid_nodes_only_no_interpolation",
        "theta_and_rho_implemented": False,
    }


def validation_rules_block(config: PolicyConfigV2) -> dict[str, Any]:
    """The report's `validation_rules` section, derived from the configuration."""
    return {
        "fixed_bump": {
            "charge_factor": FIXED_BUMP_CHARGE_FACTOR,
            "charge_scheme": FIXED_BUMP_CHARGE_SCHEME,
            "combined_budget": "reference_absolute_error + bias_charge <= cap",
            "delta_epsilon_rule": DELTA_FLAT_EPSILON_RULE,
            "vega_epsilon_rule": VEGA_FLAT_EPSILON_RULE,
            "resolved_branch": "requires a defined observed order in the supported band",
            "flat_branch": "requires each increment within its own tolerance",
            "supported_order_band": list(SUPPORTED_ORDER_BAND),
        },
        "grid_stencil": {
            "candidate": f"nodewise grid-stencil delta on {CANDIDATE_GRID}",
            "reference": GRID_STENCIL_REFERENCE_RULE,
            "order_probe": GRID_STENCIL_ORDER_RULE,
            "order_probe_grid": ORDER_PROBE_GRID,
            "supported_order_band": list(SUPPORTED_ORDER_BAND),
            "bump_bias_charge_applied": False,
        },
        "price": {
            "candidate": f"exact_node_centre_value_on_{CANDIDATE_GRID}",
            "reference_method": PRICE_REFERENCE_METHOD,
            "reference_grid": REFERENCE_GRID,
            "richardson_extrapolated": False,
            "conditional_fallback": False,
            "absolute_error_cap": config.criteria.price_absolute_error,
        },
        "shape": {
            "control": config.dominance_control,
            "residual_measure": config.dominance_residual_measure,
            "is_a_rigorous_bound": False,
            "applies_to": ["american_dominance", "american_intrinsic_bound"],
        },
        "gating_checks": list(GATING_CHECKS),
        "non_gating_published_checks": list(NON_GATING_PUBLISHED_CHECKS),
        "solve_criticality_classes": list(SOLVE_CRITICALITIES),
        "price_critical_solves": list(PRICE_CRITICAL_SOLVES),
        "delta_only_solves": list(DELTA_ONLY_SOLVES),
        "vega_only_solves": list(VEGA_ONLY_SOLVES),
        "anchor_descriptive_solves": list(ANCHOR_DESCRIPTIVE_SOLVES),
    }


def eligibility_contract_block() -> dict[str, Any]:
    """The report's `eligibility_contract` section, from module constants."""
    return {
        "common_predicates": list(COMMON_ELIGIBILITY_PREDICATES),
        "structural_c1_applies_to": list(STRUCTURAL_C1_APPLIES_TO),
        "delta_predicates": list(DELTA_ELIGIBILITY_PREDICATES),
        "vega_predicates": list(VEGA_ELIGIBILITY_PREDICATES),
        "gamma_eligibility": "never",
        "identity_excludes_contract_multiplier": True,
        "identity_includes_settlement": True,
        "authorized_training_input_statuses": sorted(AUTHORIZED_TRAINING_INPUT_STATUSES),
    }


def richardson_contract_block(config: PolicyConfigV2) -> dict[str, Any]:
    """The report's `richardson_contract` section.

    Richardson is a validation tool for the E2 grid-stencil delta and for the
    descriptive anchor diagnostics. It is never a candidate label policy and
    **never touches the price reference**.
    """
    return {
        **asdict(config.richardson),
        "formula": "fine + (fine - coarse) / 3 = (4*fine - coarse) / 3",
        "is_never_a_candidate_label_policy": True,
        "applies_to": ["grid_stencil_delta_reference", "anchor_diagnostics"],
        "applies_to_price_reference": False,
    }


def build_report(
    *,
    config: PolicyConfigV2,
    stage: str,
    cases: Sequence[Mapping[str, Any]],
    outcome: Mapping[str, Any],
    accounting: Mapping[str, Any],
    lifecycle: Mapping[str, Any],
    performance: Mapping[str, Any],
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Assemble the deterministic v2 report document."""
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "lifecycle": dict(lifecycle),
        "study": study_block(
            config=config,
            stage=stage,
            case_count=len(cases),
            gate_eligible_case_count=outcome["gate_eligible_case_count"],
        ),
        "conventions": conventions_block(config),
        "predeclared_criteria": asdict(config.criteria),
        "criteria_block": criteria_block(config),
        "criteria_digest": criteria_digest(config),
        "validation_rules": validation_rules_block(config),
        "eligibility_contract": eligibility_contract_block(),
        "richardson_contract": richardson_contract_block(config),
        "source": _source_provenance(config, root=root),
        "solver": asdict(config.solver),
        "grids": [asdict(grid) for grid in config.grids],
        "cases": list(cases),
        "stage_outcome": dict(outcome),
        "solve_accounting": dict(accounting),
        "non_claims": list(NON_CLAIMS),
        "performance": dict(performance),
    }


def _source_provenance(config: PolicyConfigV2, *, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    digests = executable_source_digests(root=root)
    return {
        "config_name": config.source_name,
        "raw_config_sha256": config.raw_config_sha256,
        **digests,
        "executable_source_inventory": dict(EXECUTABLE_SOURCE_INVENTORY),
        "executable_source_composite_digest": executable_source_composite_digest(digests),
        # Self-reported by the build, recorded beside the file digests rather
        # than conflated with them.
        "reported_binding_pde_header_sha256": str(_pde.__pde_header_sha256__),
        "reported_binding_pde_source_sha256": str(_pde.__pde_source_sha256__),
        "reported_binding_pde_implementation_sha256": str(_pde.__pde_implementation_sha256__),
        "source_digest_limitation": SOURCE_DIGEST_LIMITATION,
        "build_configuration": str(_pde.__build_configuration__),
        "cxx_compiler": str(_pde.__cxx_compiler__),
        "data_provenance": "fixed_synthetic_public_configuration_only",
        "private_market_data_used": False,
    }


def _performance(
    *, timings: Sequence[Mapping[str, Any]], started: float, initial_peak: int
) -> dict[str, Any]:
    final_peak = _peak_rss_bytes()
    return {
        "wall_seconds": time.perf_counter() - started,
        "timed_solve_attempts": len(timings),
        "timed_solve_seconds": sum(float(row["seconds"]) for row in timings),
        "peak_resident_memory_bytes": final_peak,
        "peak_resident_memory_increase_bytes": max(0, final_peak - initial_peak),
        "projection_caveat": (
            "Wall time here measures this evidence run only. It is not a "
            "production-feasibility claim and no label-throughput projection is made."
        ),
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
    }


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


# ---------------------------------------------------------------------------
# Publication
# ---------------------------------------------------------------------------


def write_outputs(
    report: Mapping[str, Any], output_directory: Path, *, overwrite: bool = False
) -> list[Path]:
    """Atomically publish the deterministic JSON report and its CSV views.

    ``overwrite`` exists for tests and for a deliberate in-process rewrite. The
    stage CLI never sets it: a v2 stage runs once, into an empty directory.
    """
    if output_directory.exists() and not output_directory.is_dir():
        raise LabelPolicyV2Error(f"output path '{output_directory}' is not a directory")
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "report.json": canonical_json(report),
        "cases.csv": _cases_csv(report),
        "eligibility.csv": _eligibility_csv(report),
        "failures.csv": _failures_csv(report),
    }
    paths: list[Path] = []
    for name, payload in outputs.items():
        destination = output_directory / name
        if destination.exists() and not overwrite:
            raise LabelPolicyV2Error(f"refusing to overwrite existing output '{destination}'")
        _write_atomic(destination, payload.encode("utf-8"))
        paths.append(destination)
    return paths


def _cases_csv(report: Mapping[str, Any]) -> str:
    header = [
        "case",
        "classification",
        "gate_eligible",
        "price",
        "price_absolute_error",
        "grid_stencil_delta",
        "grid_stencil_delta_error",
        "fixed_bump_delta_combined_error",
        "fixed_bump_vega_combined_error",
        "gamma",
        "gamma_absolute_error",
        "price_policy_passes",
        "delta_eligible",
        "vega_eligible",
        "anchor_evidence_complete",
    ]
    rows: list[list[Any]] = []
    for case in report["cases"]:
        stencil = case["grid_stencil_delta_validation"]
        rows.append(
            [
                case["state"]["name"],
                case["state"]["classification"],
                case["gate_eligible"],
                case["price"]["candidate"],
                case["price"]["absolute_error"],
                stencil["candidate_stencil_delta"],
                stencil["absolute_error"],
                case["fixed_bump_delta_validation"]["combined_absolute_error"],
                case["fixed_bump_vega_validation"]["combined_absolute_error"],
                case["gamma_evaluation_only"]["candidate"],
                case["gamma_evaluation_only"]["absolute_error"],
                case["passes"],
                case["delta_eligibility"]["eligible"],
                case["vega_eligibility"]["eligible"],
                case["support"]["anchor_evidence_complete"],
            ]
        )
    return _csv_text(header, rows)


def _eligibility_csv(report: Mapping[str, Any]) -> str:
    header = ["case", "quantity", "eligible", "value", "residual_scale", "reasons"]
    rows: list[list[Any]] = []
    for case in report["cases"]:
        for key in ("delta_eligibility", "vega_eligibility"):
            block = case[key]
            rows.append(
                [
                    case["state"]["name"],
                    block["quantity"],
                    block["eligible"],
                    block["value"],
                    block["residual_scale"],
                    ";".join(block["reasons"]),
                ]
            )
        rows.append(
            [
                case["state"]["name"],
                "gamma",
                False,
                case["gamma_evaluation_only"]["candidate"],
                None,
                "gamma_is_evaluation_only",
            ]
        )
    return _csv_text(header, rows)


def _failures_csv(report: Mapping[str, Any]) -> str:
    header = [
        "case",
        "classification",
        "gate_eligible",
        "grid",
        "role",
        "criticality",
        "exception_class",
        "message",
    ]
    rows: list[list[Any]] = []
    for case in report["cases"]:
        for failure in case["solve_failures"]:
            rows.append(
                [
                    case["state"]["name"],
                    case["state"]["classification"],
                    case["gate_eligible"],
                    failure["grid"],
                    failure["role"],
                    failure["criticality"],
                    failure["exception_class"],
                    failure["message"],
                ]
            )
    return _csv_text(header, rows)


def _csv_text(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def canonical_json(document: Any) -> str:
    """Sorted keys, two-space indent, trailing newline, no non-finite number."""
    try:
        return json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n"
    except (TypeError, ValueError) as error:
        raise LabelPolicyV2Error(f"report is not finite canonical JSON: {error}") from error


def _write_atomic(destination: Path, payload: bytes) -> None:
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except OSError as error:
        raise LabelPolicyV2Error(f"cannot publish '{destination}': {error}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--remediation-report",
        type=Path,
        default=None,
        help=(
            "Passing remediation report.json. Required by --stage confirmation, "
            "which semantically re-verifies it and refuses to start unless the "
            "recomputed remediation result is passed/pending."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        # Best effort, and no more: a second run in another directory or clone
        # cannot be detected here. One-shot status is procedural.
        if arguments.output_directory.exists() and list(
            arguments.output_directory.iterdir()
        ):
            raise LabelPolicyV2Error(
                f"refusing to run into non-empty '{arguments.output_directory}'; a v2 "
                "stage runs once, into an empty directory"
            )
        config = load_policy_config_v2(arguments.config)
        if arguments.stage == "confirmation":
            if arguments.remediation_report is None:
                raise LabelPolicyV2Error(
                    "confirmation is prohibited without --remediation-report"
                )
            previous = load_strict_json(
                arguments.remediation_report, description="remediation report"
            )
            require_confirmation_entry(previous, config)
        elif arguments.remediation_report is not None:
            raise LabelPolicyV2Error("--remediation-report belongs to --stage confirmation")
        report = run_stage(config, stage=arguments.stage)
        paths = write_outputs(report, arguments.output_directory)
    except LabelPolicyV2Error as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    lifecycle = report["lifecycle"]
    print(
        f"wrote {len(paths)} artifacts; stage={lifecycle['stage']} "
        f"remediation_status={lifecycle['remediation_status']} "
        f"confirmation_status={lifecycle['confirmation_status']} "
        f"selected_accuracy_policy={lifecycle.get('selected_accuracy_policy', '<absent>')}"
    )
    return 0


# ---------------------------------------------------------------------------
# Small strict parsers
# ---------------------------------------------------------------------------


def number_token(value: float) -> str:
    """Stable ``.17g`` token for a bump size used as a dictionary key."""
    return format(value, ".17g")


def _reject_unknown_report_keys(report: Mapping[str, Any]) -> None:
    _exact_keys(_mapping(report, "report"), REPORT_KEYS, "report")


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LabelPolicyV2Error(f"{where} must be an object")
    return value


def _sequence(value: Any, where: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise LabelPolicyV2Error(f"{where} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], where: str) -> None:
    present = set(value)
    missing = sorted(expected - present)
    unknown = sorted(present - expected)
    if missing:
        raise LabelPolicyV2Error(f"{where} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise LabelPolicyV2Error(f"{where} has unknown fields: {', '.join(unknown)}")


def _require_positive_finite_number(value: float, context: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise LabelPolicyV2Error(f"{context} must be finite and positive")


def _reject_unknown(table: Mapping[str, Any], allowed: frozenset[str], context: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise LabelPolicyV2Error(f"{context} has unknown keys: {', '.join(unknown)}")
    missing = sorted(allowed - set(table))
    if missing:
        raise LabelPolicyV2Error(f"{context} is missing keys: {', '.join(missing)}")


def _require_table(table: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    value = table.get(key)
    if not isinstance(value, Mapping):
        raise LabelPolicyV2Error(f"{context} must be a table")
    return value


def _require_string(table: Mapping[str, Any], key: str, context: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise LabelPolicyV2Error(f"{context}.{key} must be a non-empty string")
    return value


def _require_bool(table: Mapping[str, Any], key: str, context: str) -> bool:
    value = table.get(key)
    if not isinstance(value, bool):
        raise LabelPolicyV2Error(f"{context}.{key} must be a boolean")
    return value


def _require_string_array(table: Mapping[str, Any], key: str, context: str) -> tuple[str, ...]:
    value = table.get(key)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise LabelPolicyV2Error(f"{context}.{key} must be a non-empty string array")
    return tuple(value)


def _as_finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabelPolicyV2Error(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise LabelPolicyV2Error(f"{context} must be finite")
    return result


def _require_float(table: Mapping[str, Any], key: str, context: str) -> float:
    if key not in table:
        raise LabelPolicyV2Error(f"{context}.{key} is required")
    return _as_finite_float(table[key], f"{context}.{key}")


def _require_positive_float(table: Mapping[str, Any], key: str, context: str) -> float:
    value = _require_float(table, key, context)
    if value <= 0.0:
        raise LabelPolicyV2Error(f"{context}.{key} must be positive")
    return value


def _require_positive_float_array(
    table: Mapping[str, Any], key: str, context: str
) -> tuple[float, ...]:
    value = table.get(key)
    if not isinstance(value, list) or not value:
        raise LabelPolicyV2Error(f"{context}.{key} must be a non-empty numeric array")
    parsed = tuple(_as_finite_float(item, f"{context}.{key}") for item in value)
    if any(item <= 0.0 for item in parsed) or len(set(parsed)) != len(parsed):
        raise LabelPolicyV2Error(f"{context}.{key} must contain unique positive values")
    return parsed


def _require_positive_int(table: Mapping[str, Any], key: str, context: str) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LabelPolicyV2Error(f"{context}.{key} must be a positive integer")
    return value


def _require_nonnegative_int(table: Mapping[str, Any], key: str, context: str) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LabelPolicyV2Error(f"{context}.{key} must be a non-negative integer")
    return value


if __name__ == "__main__":
    sys.exit(main())
