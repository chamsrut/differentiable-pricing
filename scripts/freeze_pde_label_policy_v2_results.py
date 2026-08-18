#!/usr/bin/env python3
"""Freeze a reviewed task 9C-C3 (label-policy v2) stage report as a snapshot.

The raw stage report is expensive to produce and is deliberately ignored by
git (``artifacts/``). This tool distils it into a small, strictly versioned,
canonically serialised snapshot that *is* checked in, so the repository carries
the reviewed evidence without the multi-megabyte report.

Two explicit modes, one of which must always be named:

``--extract --report R --output S``
    Extract a fresh snapshot from a reviewed raw report. Nothing published in
    that report is trusted: the schema is exact and rejects unknown keys, the
    criteria block and its digest are rebuilt from the externally supplied
    checked-in configuration, every executable-source digest is reconciled
    against the repository, and every per-case verdict, Greek eligibility,
    aggregate count and lifecycle field is **recomputed from the raw per-solve
    numbers**. Refuses to overwrite an existing ``S`` without ``--update``.

``--check --output S``
    Validate the checked-in snapshot the same way, without writing and without
    reading the ignored raw report. **A missing snapshot is a failure, not a
    pass**: this mode exists to enforce a snapshot that is supposed to be
    there.

Only a *terminal* lifecycle state may be frozen. Task 9C-C3 supports three
lifecycle states and exactly two of them are terminal; a passing remediation
report awaiting confirmation is nonterminal and this tool refuses it.

**Stated limit.** Recomputation detects an inconsistent or partial mutation,
including one whose file hashes were regenerated. It cannot authenticate a
fully coordinated fabricated numerical report: this tool does not re-solve,
and no signature scheme exists here.

Nothing here is wired into ``scripts/check.sh`` or CI by the change that
introduced it, and no v2 snapshot exists yet.

Exit codes: ``0`` success, ``2`` any validation, provenance, or I/O failure.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python/src"))

from differentiable_pricing.american import pde_label_policy_v2 as v2  # noqa: E402

DEFAULT_SNAPSHOT: Final = (
    PROJECT_ROOT / "docs/results/american_pde_label_policy_v2_results_v1.json"
)
DEFAULT_CONFIG: Final = PROJECT_ROOT / "configs/pde_label_policy_pilot_v2.toml"

SNAPSHOT_SCHEMA_VERSION: Final = "american-pde-label-policy-v2-results/1"
REPORT_SCHEMA_VERSION: Final = v2.REPORT_SCHEMA_VERSION
SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")

#: The gating checks scoped to price-critical solves. A case failing any of
#: them has no usable price reference and therefore no price error at all.
PRICE_CRITICAL_SOLVE_CHECKS: Final = frozenset(
    {
        "price_critical_solves_returned_and_converged",
        "price_critical_exact_node_identity",
        "price_critical_pricing_and_grid_identity",
        "price_critical_residual_scale_within_price_cap",
    }
)

SNAPSHOT_KEYS: Final = frozenset(
    {
        "schema_version",
        "source",
        "lifecycle",
        "study",
        "predeclared_criteria",
        "criteria_digest",
        "outcome",
        "solve_accounting",
        "cases",
        "non_claims",
        "limitations",
    }
)
SNAPSHOT_SOURCE_KEYS: Final = frozenset(
    {
        "build_configuration",
        "config_file",
        "cxx_compiler",
        "data_provenance",
        "executable_source_composite_digest",
        "executable_source_inventory",
        "private_market_data_used",
        "raw_config_sha256",
        "reported_binding_pde_header_sha256",
        "reported_binding_pde_implementation_sha256",
        "reported_binding_pde_source_sha256",
        "report_filename",
        "report_schema_version",
        "report_sha256",
        "source_digest_limitation",
        *(f"{name}_sha256" for name in v2.EXECUTABLE_SOURCE_INVENTORY),
    }
)
SNAPSHOT_STUDY_KEYS: Final = frozenset(
    {
        "anchor_cases",
        "candidate",
        "case_count",
        "confirmation_case_count",
        "gate_cases",
        "gate_eligible_case_count",
        "name",
        "non_gating_remediation_cases",
        "price_reference_method",
        "remediation_cases",
        "stage",
    }
)
SNAPSHOT_OUTCOME_KEYS: Final = frozenset(
    {
        "all_gate_eligible_cases_pass",
        "anchor_evidence_incomplete_cases",
        "candidate",
        "criteria_were_not_loosened",
        "delta_eligible_cases",
        "descriptive_failure_count",
        "descriptive_failures",
        "failed_checks_by_name",
        "gamma_eligible_cases",
        "gate_eligible_case_count",
        "gate_eligible_failure_count",
        "gate_eligible_failures",
        "regular_cases_only_decide_selection",
        "selection_order",
        "stress_case_is_descriptive_anchor_evidence",
        "vega_eligible_cases",
    }
)
SNAPSHOT_CASE_KEYS: Final = frozenset(
    {
        "anchor_evidence_complete",
        "classification",
        "delta_ineligibility_reasons",
        "delta_label_eligible",
        "failed_checks",
        "fixed_bump_delta_branch",
        "fixed_bump_delta_combined_absolute_error",
        "fixed_bump_vega_branch",
        "fixed_bump_vega_combined_absolute_error",
        "gamma_absolute_error",
        "gamma_label_eligible",
        "gate_eligible",
        "grid_stencil_delta_absolute_error",
        "grid_stencil_observed_order",
        "linear_solves",
        "name",
        "passes",
        "price_absolute_error",
        "psor_solves",
        "psor_total_iterations",
        "solve_failures",
        "vega_ineligibility_reasons",
        "vega_label_eligible",
    }
)
SNAPSHOT_LIFECYCLE_KEYS: Final = frozenset(
    {
        "authorizes_dataset_generation",
        "authorizes_training_input",
        "case_design_digest",
        "confirmation_not_run_reason",
        "confirmation_status",
        "criteria_digest",
        "criteria_were_not_loosened",
        "freezable",
        "one_shot_enforcement",
        "protocol",
        "raw_config_sha256",
        "remediation_status",
        "selected_accuracy_policy",
        "selection_pending_fresh_top_level_approval",
        "stage",
        "terminal",
    }
)
SNAPSHOT_ACCOUNTING_KEYS: Final = frozenset(
    {
        "attempted_by_case_classification",
        "attempted_by_criticality",
        "attempted_by_solve_role",
        "attempted_surface_solves",
        "backward_inductions",
        "completed_surface_solves",
        "linear_solves",
        "planned_surface_solves",
        "psor_solves",
        "psor_total_iterations",
        "reconciliation",
        "solver_exceptions",
    }
)

LIMITATIONS: Final = (
    "This snapshot records one stage of a one-shot, predeclared protocol. Its "
    "criteria were fixed before the stage ran and are never revised inside "
    "task 9C-C3 after a result has been observed.",
    "A selected candidate is a candidate pending fresh top-level independent "
    "approval, not an accepted label policy, and it never authorizes dataset "
    "generation or neural training on its own.",
    "The price reference is the raw exact-node centre value of this study's "
    "own finer 3200x1600 rung. It is unconditional and is never "
    "Richardson-extrapolated. It is an internal numerical reference, not "
    "independent truth. Richardson appears only in the grid-stencil delta "
    "reference and in the descriptive anchor diagnostics.",
    "The American dominance and intrinsic allowances are operational "
    "price-error scale estimates built from the solver's own accumulated LCP "
    "residual. They are explicitly not certified bounds.",
    "Gamma is evaluation-only throughout. Its error is published and gates "
    "nothing, and no row treats it as a supervised target.",
    "Every verdict in this snapshot was recomputed from the raw per-solve "
    "numbers of the report it was extracted from, against the checked-in "
    "configuration. That detects inconsistent or partial mutation, including "
    "one with regenerated file hashes. It cannot authenticate a fully "
    "coordinated fabricated numerical report: this tool does not re-solve.",
    "Executable-source digests record which repository files were present. "
    "They do not prove the loaded extension binary was built from them.",
    "All evidence is synthetic. No market data, quoted price, or calibration "
    "target enters this study.",
)



# --------------------------------------------------------------------------
# The exact snapshot schema
# --------------------------------------------------------------------------
#
# Same discipline as the report schema in the runner: exact recursive types on
# the *serialized* document, `type(value) is ...` throughout so a JSON `true`
# can never be an int and an int can never be a bool, and `null` accepted only
# where declared. Validation runs before any aggregate is recomputed.

_FLAG = v2.BoolSpec()
_COUNT = v2.IntSpec(minimum=0)
_NAME = v2.StrSpec()
_TEXT = v2.StrSpec()
_DIGEST = v2.StrSpec(hex_digest=True)
_CRITERIA_DIGEST = v2.StrSpec(digest_prefix="crit")
_CASE_DESIGN_DIGEST = v2.StrSpec(digest_prefix="cases")
_SOURCE_COMPOSITE_DIGEST = v2.StrSpec(digest_prefix="src")
_NAMES = v2.ArraySpec(_NAME)
_MAYBE_REAL = v2.NullableSpec(v2.RealSpec())

SNAPSHOT_CASE_SPEC: Final = v2.ObjectSpec(
    {
        "name": _NAME,
        "classification": v2.StrSpec(enum=("regular", "stress")),
        "gate_eligible": _FLAG,
        "passes": _FLAG,
        "failed_checks": v2.ArraySpec(v2.StrSpec(enum=v2.GATING_CHECKS)),
        "price_absolute_error": _MAYBE_REAL,
        "gamma_absolute_error": _MAYBE_REAL,
        "gamma_label_eligible": _FLAG,
        "delta_label_eligible": _FLAG,
        "delta_ineligibility_reasons": _NAMES,
        "vega_label_eligible": _FLAG,
        "vega_ineligibility_reasons": _NAMES,
        "fixed_bump_delta_branch": v2.NullableSpec(
            v2.StrSpec(enum=v2.FIXED_BUMP_BRANCHES)
        ),
        "fixed_bump_delta_combined_absolute_error": _MAYBE_REAL,
        "fixed_bump_vega_branch": v2.NullableSpec(
            v2.StrSpec(enum=v2.FIXED_BUMP_BRANCHES)
        ),
        "fixed_bump_vega_combined_absolute_error": _MAYBE_REAL,
        "grid_stencil_delta_absolute_error": _MAYBE_REAL,
        "grid_stencil_observed_order": _MAYBE_REAL,
        "anchor_evidence_complete": _FLAG,
        "linear_solves": _COUNT,
        "psor_solves": _COUNT,
        "psor_total_iterations": _COUNT,
        "solve_failures": v2.ArraySpec(
            v2.ObjectSpec(
                {
                    "grid": v2.StrSpec(enum=v2.GRID_NAMES),
                    "role": _NAME,
                    "criticality": v2.StrSpec(enum=v2.SOLVE_CRITICALITIES),
                    "exception_class": _NAME,
                }
            )
        ),
    }
)

SNAPSHOT_SCHEMA: Final = v2.ObjectSpec(
    {
        "schema_version": v2.StrSpec(enum=(SNAPSHOT_SCHEMA_VERSION,)),
        "source": v2.ObjectSpec(
            {
                "build_configuration": _TEXT,
                "config_file": _NAME,
                "cxx_compiler": _TEXT,
                "data_provenance": _NAME,
                "executable_source_composite_digest": _SOURCE_COMPOSITE_DIGEST,
                "executable_source_inventory": v2.ObjectSpec(
                    {name: _NAME for name in v2.EXECUTABLE_SOURCE_INVENTORY}
                ),
                "private_market_data_used": _FLAG,
                "raw_config_sha256": _DIGEST,
                "report_filename": _NAME,
                "report_schema_version": v2.StrSpec(enum=(REPORT_SCHEMA_VERSION,)),
                "report_sha256": _DIGEST,
                "reported_binding_pde_header_sha256": _DIGEST,
                "reported_binding_pde_implementation_sha256": _DIGEST,
                "reported_binding_pde_source_sha256": _DIGEST,
                "source_digest_limitation": _TEXT,
                **{
                    f"{name}_sha256": _DIGEST for name in v2.EXECUTABLE_SOURCE_INVENTORY
                },
            }
        ),
        "lifecycle": v2.ObjectSpec(
            {
                "stage": v2.StrSpec(enum=v2.STAGES),
                "raw_config_sha256": _DIGEST,
                "criteria_digest": _CRITERIA_DIGEST,
                "case_design_digest": _CASE_DESIGN_DIGEST,
                "protocol": v2.StrSpec(enum=("one_shot_predeclared",)),
                "criteria_were_not_loosened": _FLAG,
                "one_shot_enforcement": v2.StrSpec(
                    enum=("procedural_and_provenance_backed",)
                ),
                "authorizes_dataset_generation": _FLAG,
                "authorizes_training_input": _FLAG,
                "remediation_status": v2.StrSpec(enum=v2.REMEDIATION_STATUSES),
                "confirmation_status": v2.StrSpec(enum=v2.CONFIRMATION_STATUSES),
                "confirmation_not_run_reason": v2.NullableSpec(_NAME),
                "terminal": _FLAG,
                "freezable": _FLAG,
                "selected_accuracy_policy": v2.StrSpec(
                    enum=(v2.NO_POLICY_SELECTED, v2.CANDIDATE_GRID)
                ),
                "selection_pending_fresh_top_level_approval": _FLAG,
            }
        ),
        "study": v2.ObjectSpec(
            {
                "anchor_cases": _NAMES,
                "candidate": v2.StrSpec(enum=(v2.CANDIDATE_GRID,)),
                "case_count": v2.IntSpec(minimum=1),
                "confirmation_case_count": v2.IntSpec(minimum=1),
                "gate_cases": _NAMES,
                "gate_eligible_case_count": _COUNT,
                "name": v2.StrSpec(enum=(v2.STUDY_NAME,)),
                "non_gating_remediation_cases": _NAMES,
                "price_reference_method": v2.StrSpec(
                    enum=(v2.PRICE_REFERENCE_METHOD,)
                ),
                "remediation_cases": _NAMES,
                "stage": v2.StrSpec(enum=v2.STAGES),
            }
        ),
        "predeclared_criteria": v2.ObjectSpec(
            {
                **{
                    name: v2.RealSpec(minimum=0.0)
                    for name in v2._CRITERIA_FLOAT_KEYS
                },
                "gamma_is_evaluation_only": _FLAG,
                "fixed_bump_charge_scheme": v2.StrSpec(
                    enum=(v2.FIXED_BUMP_CHARGE_SCHEME,)
                ),
                "delta_flat_epsilon_rule": v2.StrSpec(
                    enum=(v2.DELTA_FLAT_EPSILON_RULE,)
                ),
                "vega_flat_epsilon_rule": v2.StrSpec(enum=(v2.VEGA_FLAT_EPSILON_RULE,)),
                "grid_stencil_reference_rule": v2.StrSpec(
                    enum=(v2.GRID_STENCIL_REFERENCE_RULE,)
                ),
                "grid_stencil_order_rule": v2.StrSpec(
                    enum=(v2.GRID_STENCIL_ORDER_RULE,)
                ),
                "grid_stencil_bump_bias_charge_applied": _FLAG,
            }
        ),
        "criteria_digest": _CRITERIA_DIGEST,
        "outcome": v2.ObjectSpec(
            {
                "all_gate_eligible_cases_pass": _FLAG,
                "anchor_evidence_incomplete_cases": _NAMES,
                "candidate": v2.StrSpec(enum=(v2.CANDIDATE_GRID,)),
                "criteria_were_not_loosened": _FLAG,
                "delta_eligible_cases": _NAMES,
                "descriptive_failure_count": _COUNT,
                "descriptive_failures": _NAMES,
                "failed_checks_by_name": v2.MapSpec(_NAMES, key_enum=v2.GATING_CHECKS),
                "gamma_eligible_cases": _NAMES,
                "gate_eligible_case_count": _COUNT,
                "gate_eligible_failure_count": _COUNT,
                "gate_eligible_failures": _NAMES,
                "regular_cases_only_decide_selection": _FLAG,
                "selection_order": _NAMES,
                "stress_case_is_descriptive_anchor_evidence": _FLAG,
                "vega_eligible_cases": _NAMES,
            }
        ),
        "solve_accounting": v2.ObjectSpec(
            {
                "attempted_by_case_classification": v2.MapSpec(
                    _COUNT, key_enum=("regular", "stress")
                ),
                "attempted_by_criticality": v2.MapSpec(
                    _COUNT, key_enum=v2.SOLVE_CRITICALITIES
                ),
                "attempted_by_solve_role": v2.MapSpec(
                    _COUNT, key_pattern=v2._SOLVE_ROLE_KEY
                ),
                "attempted_surface_solves": _COUNT,
                "backward_inductions": _COUNT,
                "completed_surface_solves": _COUNT,
                "linear_solves": _COUNT,
                "planned_surface_solves": _COUNT,
                "psor_solves": _COUNT,
                "psor_total_iterations": _COUNT,
                "reconciliation": v2.ObjectSpec(
                    {
                        "planned_equals_attempted": _FLAG,
                        "attempted_equals_completed_plus_exceptions": _FLAG,
                        "backward_inductions_equal_completed_solves": _FLAG,
                    }
                ),
                "solver_exceptions": _COUNT,
            }
        ),
        "cases": v2.ArraySpec(SNAPSHOT_CASE_SPEC, minimum_length=1),
        "non_claims": v2.ArraySpec(_TEXT, minimum_length=1),
        "limitations": v2.ArraySpec(_TEXT, minimum_length=1),
    }
)


class FreezeError(RuntimeError):
    """Raised when the report, the snapshot, or repository provenance is invalid."""


# --------------------------------------------------------------------------
# Small strict readers
# --------------------------------------------------------------------------


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FreezeError(f"{where} must be an object")
    return value


def _sequence(value: Any, where: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise FreezeError(f"{where} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], where: str) -> None:
    present = set(value)
    missing = sorted(expected - present)
    unknown = sorted(present - expected)
    if missing:
        raise FreezeError(f"{where} is missing required fields: {', '.join(missing)}")
    if unknown:
        raise FreezeError(f"{where} has unknown fields: {', '.join(unknown)}")


def _string(container: Mapping[str, Any], key: str, where: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value:
        raise FreezeError(f"{where}.{key} must be a non-empty string")
    return value


def _digest_field(container: Mapping[str, Any], key: str, where: str) -> str:
    value = _string(container, key, where)
    if not SHA256_PATTERN.fullmatch(value):
        raise FreezeError(f"{where}.{key} must be a lowercase hex SHA-256 digest")
    return value


def _bool(container: Mapping[str, Any], key: str, where: str) -> bool:
    value = container.get(key)
    if not isinstance(value, bool):
        raise FreezeError(f"{where}.{key} must be a boolean")
    return value


def _integer(container: Mapping[str, Any], key: str, where: str, *, minimum: int) -> int:
    value = container.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise FreezeError(f"{where}.{key} must be an integer")
    if value < minimum:
        raise FreezeError(f"{where}.{key} must be at least {minimum}")
    return value


def _optional_finite(container: Mapping[str, Any], key: str, where: str) -> float | None:
    value = container.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FreezeError(f"{where}.{key} must be a number or null")
    number = float(value)
    if not math.isfinite(number):
        raise FreezeError(f"{where}.{key} must be finite, got {value!r}")
    return number


def _string_array(container: Mapping[str, Any], key: str, where: str) -> list[str]:
    value = _sequence(container.get(key), f"{where}.{key}")
    if any(not isinstance(item, str) for item in value):
        raise FreezeError(f"{where}.{key} must be an array of strings")
    return list(value)


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise FreezeError(f"cannot hash '{path}': {error}") from error


def load_expected_config(config_path: Path) -> v2.PolicyConfigV2:
    """Load the externally supplied, checked-in v2 configuration."""
    try:
        return v2.load_policy_config_v2(config_path)
    except v2.LabelPolicyV2Error as error:
        raise FreezeError(f"cannot use expected configuration '{config_path}': {error}") from error


# --------------------------------------------------------------------------
# Report validation: recompute, never trust
# --------------------------------------------------------------------------


def validate_report(
    report: Any, config: v2.PolicyConfigV2, *, root: Path = PROJECT_ROOT
) -> v2.RecomputedStage:
    """Semantically verify a raw v2 stage report against the expected config.

    Delegates to the runner's single derivation so a published field is never
    compared against itself, then additionally requires the state to be
    terminal, because only a terminal outcome may be frozen.
    """
    payload = _mapping(report, "report")
    try:
        recomputed = v2.verify_stage_report(payload, config, root=root, where="report")
    except v2.LabelPolicyV2Error as error:
        raise FreezeError(str(error)) from error
    lifecycle = recomputed.lifecycle
    if not lifecycle["freezable"]:
        raise FreezeError(
            "report.lifecycle is a nonterminal state: a passing remediation awaiting "
            "confirmation is deliberately not freezable"
        )
    if not lifecycle["terminal"]:
        raise FreezeError("report.lifecycle is not terminal and may not be frozen")
    return recomputed


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def extract_snapshot(
    report: Any,
    config: v2.PolicyConfigV2,
    *,
    report_filename: str,
    report_sha256: str,
    root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Build the frozen snapshot from a semantically verified raw report."""
    if not SHA256_PATTERN.fullmatch(report_sha256):
        raise FreezeError("report_sha256 must be a lowercase hex SHA-256 digest")
    recomputed = validate_report(report, config, root=root)
    payload = _mapping(report, "report")
    source = payload["source"]
    study = payload["study"]
    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "source": {
            "build_configuration": source["build_configuration"],
            "config_file": source["config_name"],
            "cxx_compiler": source["cxx_compiler"],
            "data_provenance": source["data_provenance"],
            "executable_source_composite_digest": source[
                "executable_source_composite_digest"
            ],
            "executable_source_inventory": dict(source["executable_source_inventory"]),
            "private_market_data_used": source["private_market_data_used"],
            "raw_config_sha256": source["raw_config_sha256"],
            "report_filename": report_filename,
            "report_schema_version": payload["schema_version"],
            "report_sha256": report_sha256,
            "reported_binding_pde_header_sha256": source[
                "reported_binding_pde_header_sha256"
            ],
            "reported_binding_pde_implementation_sha256": source[
                "reported_binding_pde_implementation_sha256"
            ],
            "reported_binding_pde_source_sha256": source[
                "reported_binding_pde_source_sha256"
            ],
            "source_digest_limitation": source["source_digest_limitation"],
            **{
                f"{name}_sha256": source[f"{name}_sha256"]
                for name in v2.EXECUTABLE_SOURCE_INVENTORY
            },
        },
        "lifecycle": dict(recomputed.lifecycle),
        "study": {
            "anchor_cases": list(study["anchor_cases"]),
            "candidate": v2.CANDIDATE_GRID,
            "case_count": len(recomputed.cases),
            "confirmation_case_count": study["confirmation_case_count"],
            "gate_cases": list(study["gate_cases"]),
            "gate_eligible_case_count": recomputed.outcome["gate_eligible_case_count"],
            "name": study["name"],
            "non_gating_remediation_cases": list(study["non_gating_remediation_cases"]),
            "price_reference_method": v2.PRICE_REFERENCE_METHOD,
            "remediation_cases": list(study["remediation_cases"]),
            "stage": recomputed.stage,
        },
        "predeclared_criteria": dict(payload["predeclared_criteria"]),
        "criteria_digest": v2.criteria_digest(config),
        "outcome": {key: recomputed.outcome[key] for key in sorted(SNAPSHOT_OUTCOME_KEYS)},
        "solve_accounting": {
            key: recomputed.accounting[key] for key in sorted(SNAPSHOT_ACCOUNTING_KEYS)
        },
        "cases": [_extract_case(case) for case in recomputed.cases],
        "non_claims": list(v2.NON_CLAIMS),
        "limitations": list(LIMITATIONS),
    }
    validate_snapshot(snapshot, config, root=root)
    return snapshot


def _extract_case(case: Mapping[str, Any]) -> dict[str, Any]:
    vega = case["fixed_bump_vega_validation"]
    stencil = case["grid_stencil_delta_validation"]
    return {
        "anchor_evidence_complete": bool(case["support"]["anchor_evidence_complete"]),
        "classification": case["state"]["classification"],
        "delta_ineligibility_reasons": list(case["delta_eligibility"]["reasons"]),
        "delta_label_eligible": bool(case["delta_eligibility"]["eligible"]),
        "failed_checks": sorted(name for name, passed in case["checks"].items() if not passed),
        "fixed_bump_delta_branch": case["fixed_bump_delta_validation"]["branch"],
        "fixed_bump_delta_combined_absolute_error": case["fixed_bump_delta_validation"][
            "combined_absolute_error"
        ],
        "fixed_bump_vega_branch": vega["branch"],
        "fixed_bump_vega_combined_absolute_error": vega["combined_absolute_error"],
        "gamma_absolute_error": case["gamma_evaluation_only"]["absolute_error"],
        "gamma_label_eligible": False,
        "gate_eligible": bool(case["gate_eligible"]),
        "grid_stencil_delta_absolute_error": stencil["absolute_error"],
        "grid_stencil_observed_order": stencil["observed_stencil_order"],
        # The three totals the distilled row cannot otherwise imply, so the
        # stage accounting stays independently derivable from the case rows.
        "linear_solves": sum(int(solve["linear_solves"]) for solve in case["solves"]),
        "name": case["state"]["name"],
        "passes": bool(case["passes"]),
        "price_absolute_error": case["price"]["absolute_error"],
        "psor_solves": sum(int(solve["psor_solves"]) for solve in case["solves"]),
        "psor_total_iterations": sum(
            int(solve["psor_total_iterations"]) for solve in case["solves"]
        ),
        "solve_failures": [
            {
                "criticality": failure["criticality"],
                "exception_class": failure["exception_class"],
                "grid": failure["grid"],
                "role": failure["role"],
            }
            for failure in case["solve_failures"]
        ],
        "vega_ineligibility_reasons": list(case["vega_eligibility"]["reasons"]),
        "vega_label_eligible": bool(case["vega_eligibility"]["eligible"]),
    }


# --------------------------------------------------------------------------
# Snapshot validation
# --------------------------------------------------------------------------


def validate_snapshot(
    snapshot: Any, config: v2.PolicyConfigV2, *, root: Path = PROJECT_ROOT
) -> Mapping[str, Any]:
    """Validate a frozen snapshot against the checked-in configuration.

    Every threshold, role, case set and lifecycle field is checked against the
    configuration and the runner's own constants, not against the snapshot's
    own claims. Aggregate counts are recomputed from the snapshot's case rows.
    """
    # Exact recursive type validation of the serialized snapshot, before any
    # aggregate is recomputed or any value is read as a number.
    try:
        v2.validate_against_schema(SNAPSHOT_SCHEMA, snapshot, where="snapshot")
    except v2.LabelPolicyV2Error as error:
        raise FreezeError(str(error)) from error
    payload = _mapping(snapshot, "snapshot")
    validate_snapshot_context(payload, config)
    _exact_keys(payload, SNAPSHOT_KEYS, "snapshot")
    if _string(payload, "schema_version", "snapshot") != SNAPSHOT_SCHEMA_VERSION:
        raise FreezeError(f"snapshot schema_version must be '{SNAPSHOT_SCHEMA_VERSION}'")

    _validate_snapshot_source(payload, config, root=root)
    lifecycle = _validate_snapshot_lifecycle(payload, config)
    _validate_snapshot_study(payload, lifecycle)
    _validate_snapshot_criteria(payload, config)
    outcome = _mapping(payload["outcome"], "snapshot.outcome")
    _exact_keys(outcome, SNAPSHOT_OUTCOME_KEYS, "snapshot.outcome")
    _exact_keys(
        _mapping(payload["solve_accounting"], "snapshot.solve_accounting"),
        SNAPSHOT_ACCOUNTING_KEYS,
        "snapshot.solve_accounting",
    )
    _validate_snapshot_cases(payload, outcome, config)
    _validate_snapshot_accounting(payload, config)

    if list(_sequence(payload["non_claims"], "snapshot.non_claims")) != list(v2.NON_CLAIMS):
        raise FreezeError("snapshot.non_claims must be the runner's predeclared statements")
    if list(_sequence(payload["limitations"], "snapshot.limitations")) != list(LIMITATIONS):
        raise FreezeError("snapshot.limitations must be this tool's declared limitations")
    return payload



#: The snapshot's context-derived map. Everything else is statically bounded.
CONTEXTUAL_MAP_PATHS: Final = frozenset({".solve_accounting.attempted_by_solve_role"})


def validate_snapshot_context(
    snapshot: Mapping[str, Any], config: v2.PolicyConfigV2
) -> None:
    """Require the snapshot's role map to be exactly the stage's own plan keys.

    Runs immediately after the static schema, before any aggregate is
    recomputed. The static schema bounds the key *syntax*; only the
    configuration and the case list give the exact set.
    """
    stage = _string(_mapping(snapshot["study"], "snapshot.study"), "stage", "snapshot.study")
    if stage not in v2.STAGES:
        raise FreezeError(f"snapshot.study.stage must be one of {list(v2.STAGES)}")
    expected: set[str] = set()
    for index, entry in enumerate(_sequence(snapshot["cases"], "snapshot.cases")):
        case = _mapping(entry, f"snapshot.cases[{index}]")
        declared = config.case(_string(case, "name", f"snapshot.cases[{index}]"))
        expected.update(
            f"{grid}/{role}"
            for grid, role in v2.required_solve_keys(declared, config)
        )
    published = _mapping(
        _mapping(snapshot["solve_accounting"], "snapshot.solve_accounting")[
            "attempted_by_solve_role"
        ],
        "snapshot.solve_accounting.attempted_by_solve_role",
    )
    unexpected = sorted(set(published) - expected)
    missing = sorted(expected - set(published))
    if unexpected or missing:
        raise FreezeError(
            "snapshot.solve_accounting.attempted_by_solve_role key set is wrong; "
            f"unexpected={unexpected} missing={missing} expected={sorted(expected)}"
        )


def _validate_snapshot_source(
    payload: Mapping[str, Any], config: v2.PolicyConfigV2, *, root: Path
) -> None:
    source = _mapping(payload["source"], "snapshot.source")
    _exact_keys(source, SNAPSHOT_SOURCE_KEYS, "snapshot.source")
    # The composite is a prefixed digest, not a bare hex string, so it is
    # reconciled below rather than pattern-matched here.
    _string(source, "executable_source_composite_digest", "snapshot.source")
    for key in (
        "raw_config_sha256",
        "report_sha256",
        "reported_binding_pde_header_sha256",
        "reported_binding_pde_implementation_sha256",
        "reported_binding_pde_source_sha256",
        *(f"{name}_sha256" for name in v2.EXECUTABLE_SOURCE_INVENTORY),
    ):
        _digest_field(source, key, "snapshot.source")
    for key in ("config_file", "report_filename"):
        if "/" in _string(source, key, "snapshot.source") or "\\" in source[key]:
            raise FreezeError(f"snapshot.source.{key} must be a bare filename")
    if _string(source, "report_schema_version", "snapshot.source") != REPORT_SCHEMA_VERSION:
        raise FreezeError(
            f"snapshot.source.report_schema_version must be '{REPORT_SCHEMA_VERSION}'"
        )
    if _bool(source, "private_market_data_used", "snapshot.source"):
        raise FreezeError("snapshot.source.private_market_data_used must be false")
    if source["raw_config_sha256"] != config.raw_config_sha256:
        raise FreezeError(
            "snapshot.source.raw_config_sha256 does not match the checked-in configuration"
        )
    if dict(_mapping(source["executable_source_inventory"], "snapshot.source")) != dict(
        v2.EXECUTABLE_SOURCE_INVENTORY
    ):
        raise FreezeError("snapshot.source.executable_source_inventory was edited")
    if source["source_digest_limitation"] != v2.SOURCE_DIGEST_LIMITATION:
        raise FreezeError("snapshot.source.source_digest_limitation was edited")
    try:
        v2.verify_executable_source_digests(source, root=root, where="snapshot.source")
    except v2.LabelPolicyV2Error as error:
        raise FreezeError(str(error)) from error


def _validate_snapshot_lifecycle(
    payload: Mapping[str, Any], config: v2.PolicyConfigV2
) -> Mapping[str, Any]:
    lifecycle = _mapping(payload["lifecycle"], "snapshot.lifecycle")
    stage = _string(lifecycle, "stage", "snapshot.lifecycle")
    if stage not in v2.STAGES:
        raise FreezeError(f"snapshot.lifecycle.stage must be one of {list(v2.STAGES)}")
    # A terminal state always carries the selected-policy field; the
    # nonterminal state 2 never does and can never be frozen. The block must
    # equal one the runner itself can produce, rebuilt from the configuration.
    _exact_keys(lifecycle, SNAPSHOT_LIFECYCLE_KEYS, "snapshot.lifecycle")
    supported = [
        v2.lifecycle_block(stage=stage, all_gate_eligible_cases_pass=passed, config=config)
        for passed in (False, True)
    ]
    if dict(lifecycle) not in [dict(block) for block in supported]:
        raise FreezeError(
            "snapshot.lifecycle is not a supported, self-consistent terminal state"
        )
    if not lifecycle["terminal"] or not lifecycle["freezable"]:
        raise FreezeError("snapshot.lifecycle records a state that may not be frozen")
    if lifecycle["criteria_digest"] != _string(payload, "criteria_digest", "snapshot"):
        raise FreezeError("snapshot.criteria_digest and snapshot.lifecycle disagree")
    if lifecycle["criteria_digest"] != v2.criteria_digest(config):
        raise FreezeError(
            "snapshot.criteria_digest does not match the checked-in configuration"
        )
    if lifecycle["case_design_digest"] != config.case_design_digest:
        raise FreezeError("snapshot.lifecycle.case_design_digest does not match the design")
    return lifecycle


def _validate_snapshot_study(
    payload: Mapping[str, Any], lifecycle: Mapping[str, Any]
) -> None:
    study = _mapping(payload["study"], "snapshot.study")
    _exact_keys(study, SNAPSHOT_STUDY_KEYS, "snapshot.study")
    if _string(study, "candidate", "snapshot.study") != v2.CANDIDATE_GRID:
        raise FreezeError(f"snapshot.study.candidate must be '{v2.CANDIDATE_GRID}'")
    for key, expected in (
        ("name", v2.STUDY_NAME),
        ("price_reference_method", v2.PRICE_REFERENCE_METHOD),
    ):
        if _string(study, key, "snapshot.study") != expected:
            raise FreezeError(f"snapshot.study.{key} must be exactly '{expected}'")
    if _string(study, "stage", "snapshot.study") != lifecycle["stage"]:
        raise FreezeError("snapshot.study.stage and snapshot.lifecycle.stage disagree")
    _integer(study, "case_count", "snapshot.study", minimum=1)
    _integer(study, "gate_eligible_case_count", "snapshot.study", minimum=0)
    if _integer(study, "confirmation_case_count", "snapshot.study", minimum=1) != (
        v2.CONFIRMATION_CASE_COUNT
    ):
        raise FreezeError(
            f"snapshot.study.confirmation_case_count must be {v2.CONFIRMATION_CASE_COUNT}"
        )
    for key, expected in (
        ("remediation_cases", list(v2.REMEDIATION_CASES)),
        ("gate_cases", list(v2.REMEDIATION_GATE_CASES)),
        ("non_gating_remediation_cases", [v2.NON_GATING_REMEDIATION_CASE]),
        ("anchor_cases", list(v2.ANCHOR_CASES)),
    ):
        if _string_array(study, key, "snapshot.study") != expected:
            raise FreezeError(f"snapshot.study.{key} must be exactly {expected}")


def _validate_snapshot_criteria(
    payload: Mapping[str, Any], config: v2.PolicyConfigV2
) -> None:
    criteria = _mapping(payload["predeclared_criteria"], "snapshot.predeclared_criteria")
    expected = dataclasses.asdict(config.criteria)
    if dict(criteria) != expected:
        differing = sorted(
            key for key in set(criteria) | set(expected) if criteria.get(key) != expected.get(key)
        )
        raise FreezeError(
            "snapshot.predeclared_criteria does not match the checked-in configuration; "
            f"differing: {differing}"
        )


def _validate_snapshot_cases(
    payload: Mapping[str, Any], outcome: Mapping[str, Any], config: v2.PolicyConfigV2
) -> None:
    study = payload["study"]
    stage = study["stage"]
    expected_names = list(config.stage_case_names(stage))
    cases = _sequence(payload["cases"], "snapshot.cases")
    names = [
        _string(_mapping(case, f"snapshot.cases[{index}]"), "name", f"snapshot.cases[{index}]")
        for index, case in enumerate(cases)
    ]
    if names != expected_names:
        missing = sorted(set(expected_names) - set(names))
        unknown = sorted(set(names) - set(expected_names))
        duplicated = sorted({name for name in names if names.count(name) > 1})
        raise FreezeError(
            f"snapshot.cases must be exactly the {stage} case set, in order; "
            f"missing={missing} unexpected={unknown} duplicated={duplicated}"
        )
    if len(cases) != study["case_count"]:
        raise FreezeError("snapshot.study.case_count disagrees with snapshot.cases")

    gating = 0
    failures: list[str] = []
    descriptive: list[str] = []
    failed_by_check: dict[str, list[str]] = {}
    delta_eligible: list[str] = []
    vega_eligible: list[str] = []
    anchor_incomplete: list[str] = []
    for index, entry in enumerate(cases):
        where = f"snapshot.cases[{index}]"
        case = _mapping(entry, where)
        _exact_keys(case, SNAPSHOT_CASE_KEYS, where)
        name = names[index]
        declared = config.case(name)
        if _string(case, "classification", where) != declared.classification:
            raise FreezeError(f"{where}.classification disagrees with the case design")
        for key in (
            "fixed_bump_delta_combined_absolute_error",
            "fixed_bump_vega_combined_absolute_error",
            "gamma_absolute_error",
            "grid_stencil_delta_absolute_error",
            "grid_stencil_observed_order",
            "price_absolute_error",
        ):
            _optional_finite(case, key, where)
        if _bool(case, "gamma_label_eligible", where):
            raise FreezeError(f"{where}: gamma is never label-eligible")
        if _bool(case, "gate_eligible", where) != config.is_gate_eligible(declared):
            raise FreezeError(f"{where}.gate_eligible disagrees with the predeclared gate set")
        passes = _bool(case, "passes", where)
        failed = _string_array(case, "failed_checks", where)
        if passes != (not failed):
            raise FreezeError(f"{where}.passes disagrees with {where}.failed_checks")
        unknown = sorted(set(failed) - set(v2.GATING_CHECKS))
        if unknown:
            raise FreezeError(f"{where}.failed_checks names non-gating checks: {unknown}")
        for failure in _sequence(case["solve_failures"], f"{where}.solve_failures"):
            block = _mapping(failure, f"{where}.solve_failures[]")
            _exact_keys(
                block,
                frozenset({"criticality", "exception_class", "grid", "role"}),
                f"{where}.solve_failures[]",
            )
            if _string(block, "criticality", where) not in v2.SOLVE_CRITICALITIES:
                raise FreezeError(f"{where}.solve_failures[] has an unknown criticality")
        _validate_snapshot_case_consistency(case, config, where)
        if case["gate_eligible"]:
            gating += 1
            if not passes:
                failures.append(name)
            for check in failed:
                failed_by_check.setdefault(check, []).append(name)
        elif not passes:
            descriptive.append(name)
        if _bool(case, "delta_label_eligible", where):
            delta_eligible.append(name)
        if _bool(case, "vega_label_eligible", where):
            vega_eligible.append(name)
        if not _bool(case, "anchor_evidence_complete", where):
            anchor_incomplete.append(name)

    recomputed = {
        "gate_eligible_case_count": gating,
        "gate_eligible_failure_count": len(failures),
        "gate_eligible_failures": sorted(failures),
        "all_gate_eligible_cases_pass": not failures,
        "delta_eligible_cases": sorted(delta_eligible),
        "vega_eligible_cases": sorted(vega_eligible),
        "anchor_evidence_incomplete_cases": sorted(anchor_incomplete),
        "gamma_eligible_cases": [],
        "descriptive_failures": sorted(descriptive),
        "descriptive_failure_count": len(descriptive),
        "candidate": v2.CANDIDATE_GRID,
    }
    failed_by_check = {key: sorted(value) for key, value in sorted(failed_by_check.items())}
    for key, want in recomputed.items():
        got = outcome[key]
        if isinstance(want, list):
            got = sorted(got)
        if got != want:
            raise FreezeError(f"snapshot.outcome.{key} is {got!r}, recomputed {want!r}")
    published_checks = _mapping(
        outcome["failed_checks_by_name"], "snapshot.outcome.failed_checks_by_name"
    )
    normalised: dict[str, list[str]] = {}
    for key, value in published_checks.items():
        names = _sequence(value, f"snapshot.outcome.failed_checks_by_name.{key}")
        if any(not isinstance(item, str) for item in names):
            raise FreezeError(
                f"snapshot.outcome.failed_checks_by_name.{key} must be an array of names"
            )
        normalised[key] = sorted(names)
    if normalised != failed_by_check:
        raise FreezeError(
            f"snapshot.outcome.failed_checks_by_name is {normalised!r}, "
            f"recomputed {failed_by_check!r}"
        )
    if study["gate_eligible_case_count"] != gating:
        raise FreezeError("snapshot.study.gate_eligible_case_count disagrees with the rows")
    for key in (
        "criteria_were_not_loosened",
        "regular_cases_only_decide_selection",
        "stress_case_is_descriptive_anchor_evidence",
    ):
        if not _bool(outcome, key, "snapshot.outcome"):
            raise FreezeError(f"snapshot.outcome.{key} must be true")
    if _string_array(outcome, "selection_order", "snapshot.outcome") != [v2.CANDIDATE_GRID]:
        raise FreezeError("snapshot.outcome.selection_order must be the single candidate")

    lifecycle = payload["lifecycle"]
    if lifecycle["stage"] == "confirmation":
        want = v2.NO_POLICY_SELECTED if failures else v2.CANDIDATE_GRID
        if lifecycle["selected_accuracy_policy"] != want:
            raise FreezeError("snapshot.lifecycle.selected_accuracy_policy disagrees with rows")
    elif not failures:
        raise FreezeError(
            "a frozen remediation snapshot must record at least one gate-eligible "
            "failure; a passing remediation is nonterminal and may not be frozen"
        )


def _validate_snapshot_case_consistency(
    case: Mapping[str, Any], config: v2.PolicyConfigV2, where: str
) -> None:
    """Cross-check one distilled case row against the configured thresholds.

    A snapshot does not carry the raw per-solve numbers, so it cannot re-run
    the full evaluation. What it can do — and does here — is refuse a row whose
    published errors contradict its own published verdicts against the
    checked-in caps and order band.
    """
    criteria = config.criteria
    price_error = case["price_absolute_error"]
    failed = set(case["failed_checks"])
    price_failed = "price_absolute_error" in failed
    # A case whose price-critical support failed has no computable price error.
    if failed & PRICE_CRITICAL_SOLVE_CHECKS:
        if price_error is not None:
            raise FreezeError(
                f"{where}: a case with an unusable price-critical solve cannot publish a "
                f"price error, got {price_error!r}"
            )
        if not price_failed:
            raise FreezeError(
                f"{where}: an unavailable price error must fail the price check"
            )
    elif price_error is None:
        raise FreezeError(f"{where}: a supported case must publish its price error")
    elif (float(price_error) > criteria.price_absolute_error) != price_failed:
        raise FreezeError(
            f"{where}.price_absolute_error={price_error!r} contradicts its own "
            f"failed_checks against the {criteria.price_absolute_error} cap"
        )

    delta_eligible = bool(case["delta_label_eligible"])
    for key, cap in (
        ("grid_stencil_delta_absolute_error", criteria.delta_absolute_error),
        ("fixed_bump_delta_combined_absolute_error", criteria.delta_absolute_error),
    ):
        value = case[key]
        if delta_eligible and (value is None or float(value) > cap):
            raise FreezeError(
                f"{where}: delta cannot be eligible with {key}={value!r} against a {cap} cap"
            )
    order = case["grid_stencil_observed_order"]
    band = (criteria.minimum_supported_stencil_order, criteria.maximum_supported_stencil_order)
    if delta_eligible and (order is None or not band[0] <= float(order) <= band[1]):
        raise FreezeError(
            f"{where}: delta cannot be eligible with an unsupported stencil order {order!r}"
        )

    vega_eligible = bool(case["vega_label_eligible"])
    vega_error = case["fixed_bump_vega_combined_absolute_error"]
    vega_cap = criteria.vega_absolute_error_per_unit_volatility
    if vega_eligible and (vega_error is None or float(vega_error) > vega_cap):
        raise FreezeError(
            f"{where}: vega cannot be eligible with a combined error of {vega_error!r} "
            f"against a {vega_cap} cap"
        )
    if delta_eligible and case["fixed_bump_delta_branch"] is None:
        raise FreezeError(f"{where}: delta cannot be eligible with no fixed-bump branch")
    if vega_eligible and case["fixed_bump_vega_branch"] is None:
        raise FreezeError(f"{where}: vega cannot be eligible with no fixed-bump branch")

    # A failure's criticality decides exactly what it is allowed to veto.
    for failure in case["solve_failures"]:
        criticality = failure["criticality"]
        if criticality == v2.PRICE_CRITICAL and case["passes"]:
            raise FreezeError(
                f"{where}: a price-critical solve failure cannot leave the price policy passing"
            )
        if criticality in {v2.PRICE_CRITICAL, v2.DELTA_ONLY} and delta_eligible:
            raise FreezeError(f"{where}: a {criticality} failure cannot leave delta eligible")
        if criticality in {v2.PRICE_CRITICAL, v2.VEGA_ONLY} and vega_eligible:
            raise FreezeError(f"{where}: a {criticality} failure cannot leave vega eligible")
        if criticality == v2.ANCHOR_DESCRIPTIVE and case["anchor_evidence_complete"]:
            raise FreezeError(
                f"{where}: an anchor-rung failure cannot leave the anchor evidence complete"
            )

def _validate_snapshot_accounting(
    payload: Mapping[str, Any], config: v2.PolicyConfigV2
) -> None:
    """Recompute the whole stage accounting from the snapshot's own case rows.

    Nothing stored in ``solve_accounting`` is read back. The plan comes from
    the configuration, the per-role/criticality/class breakdowns are pure
    functions of the configuration and the case list, the exception counts come
    from each row's own failure records, and the three engine totals come from
    the immutable per-case fields the extractor derives from the raw solves.
    """
    stage = payload["study"]["stage"]
    cases = payload["cases"]
    published = _mapping(payload["solve_accounting"], "snapshot.solve_accounting")

    planned = v2.planned_solve_count(config, stage)
    attempted = 0
    exceptions = 0
    by_role: dict[str, int] = {}
    by_criticality: dict[str, int] = {}
    by_case_class: dict[str, int] = {}
    linear = 0
    psor = 0
    iterations = 0
    for index, entry in enumerate(cases):
        where = f"snapshot.cases[{index}]"
        case = _mapping(entry, where)
        declared = config.case(_string(case, "name", where))
        required = v2.required_solve_keys(declared, config)
        attempted += len(required)
        exceptions += len(_sequence(case["solve_failures"], f"{where}.solve_failures"))
        by_case_class[declared.classification] = (
            by_case_class.get(declared.classification, 0) + len(required)
        )
        for grid_name, role in required:
            key = f"{grid_name}/{role}"
            by_role[key] = by_role.get(key, 0) + 1
            criticality = v2.solve_criticality(declared, grid_name, role)
            by_criticality[criticality] = by_criticality.get(criticality, 0) + 1
        linear += _integer(case, "linear_solves", where, minimum=0)
        psor += _integer(case, "psor_solves", where, minimum=0)
        iterations += _integer(case, "psor_total_iterations", where, minimum=0)

    completed = attempted - exceptions
    recomputed = {
        "planned_surface_solves": planned,
        "attempted_surface_solves": attempted,
        "completed_surface_solves": completed,
        "solver_exceptions": exceptions,
        "backward_inductions": completed,
        "linear_solves": linear,
        "psor_solves": psor,
        "psor_total_iterations": iterations,
        "attempted_by_solve_role": dict(sorted(by_role.items())),
        "attempted_by_criticality": dict(sorted(by_criticality.items())),
        "attempted_by_case_classification": dict(sorted(by_case_class.items())),
        "reconciliation": {
            "planned_equals_attempted": planned == attempted,
            "attempted_equals_completed_plus_exceptions": attempted
            == completed + exceptions,
            "backward_inductions_equal_completed_solves": completed == completed,
        },
    }
    for key, want in recomputed.items():
        got = published[key]
        if got != want:
            raise FreezeError(
                f"snapshot.solve_accounting.{key} is {got!r}, recomputed {want!r}"
            )
    if not all(recomputed["reconciliation"].values()):
        raise FreezeError(
            f"snapshot.solve_accounting does not reconcile: {recomputed['reconciliation']}"
        )

# --------------------------------------------------------------------------
# Serialisation and CLI
# --------------------------------------------------------------------------


def serialise(snapshot: Mapping[str, Any]) -> str:
    """Canonical text form: sorted keys, two-space indent, trailing newline."""
    return json.dumps(snapshot, indent=2, sort_keys=True, ensure_ascii=True) + "\n"


def write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically via a same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def load_json(path: Path, description: str) -> Any:
    """Read one evidence JSON file through the shared strict loader.

    Duplicate object keys and the non-standard `NaN`/`Infinity` constants are
    refused *inside the parser*, before extraction, snapshot writing or any
    canonical-text comparison: by the time an ordinary parse has finished, one
    value of a repeated key has already been discarded.
    """
    try:
        return v2.load_strict_json(path, description=description)
    except v2.LabelPolicyV2Error as error:
        raise FreezeError(str(error)) from error


def snapshot_from_report(
    report_path: Path, config: v2.PolicyConfigV2, *, root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    """Load, hash, semantically verify, and extract a snapshot from a report."""
    report = load_json(report_path, "report")
    return extract_snapshot(
        report,
        config,
        report_filename=report_path.name,
        report_sha256=_sha256_file(report_path),
        root=root,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--extract",
        action="store_true",
        help="Extract a snapshot from a reviewed raw report. Requires --report.",
    )
    mode.add_argument(
        "--check",
        action="store_true",
        help=(
            "Validate the checked-in snapshot and its repository provenance without "
            "writing. A missing snapshot fails."
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=(
            "Reviewed raw v2 stage report.json. Required with --extract; optional "
            "with --check to also detect staleness."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_SNAPSHOT,
        help="Frozen snapshot path to write, or to validate under --check.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Checked-in v2 configuration every criterion is recomputed against.",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Permit overwriting an existing snapshot when extracting.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        config = load_expected_config(arguments.config)
        if arguments.check:
            if arguments.update:
                raise FreezeError("--update cannot be combined with --check")
            if not arguments.output.is_file():
                raise FreezeError(
                    f"expected snapshot '{arguments.output}' is absent; --check enforces "
                    "a snapshot that is supposed to exist and never passes without one"
                )
            snapshot = validate_snapshot(load_json(arguments.output, "snapshot"), config)
            current = arguments.output.read_text(encoding="utf-8")
            if current != serialise(snapshot):
                raise FreezeError(
                    f"snapshot '{arguments.output}' is not canonically serialised; "
                    f"regenerate it with {Path(__file__).name}"
                )
            if arguments.report is not None:
                fresh = snapshot_from_report(arguments.report, config)
                if serialise(fresh) != current:
                    raise FreezeError(
                        f"snapshot '{arguments.output}' is stale with respect to "
                        f"'{arguments.report}'; regenerate it with --update"
                    )
            print(f"snapshot ok: {arguments.output}")
            return 0

        if arguments.report is None:
            raise FreezeError("--extract requires --report")
        if arguments.output.exists() and not arguments.update:
            raise FreezeError(
                f"refusing to overwrite existing snapshot '{arguments.output}'; "
                "pass --update to replace it deliberately"
            )
        snapshot = snapshot_from_report(arguments.report, config)
        write_atomic(arguments.output, serialise(snapshot))
        print(f"wrote {arguments.output}")
    except FreezeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
