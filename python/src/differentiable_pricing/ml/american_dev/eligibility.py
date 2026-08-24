"""Task 9H assessability: computed and persisted **before** anything is scored.

This is a separate module, a separate human-invoked script and a separate
artifact for one reason: the ordering has to be externally verifiable rather
than a matter of execution intention. Whether the ``>= 99%`` primary pass rate
can honestly be reported as a number is settled by the contracts and the label
operator alone -- **no model output is read here** -- and the price-fidelity
analysis refuses to score a row set until the matching artifact exists.

What the ordering protects against
----------------------------------
The declared eligibility rule anchors on the *label's* discretization error, not
the *model's*. Where the lattice is well converged it admits rows whose vega is
almost arbitrarily small, and on those rows the primary statistic is decided by
the denominator before the model is consulted. Deciding afterwards which rows
"should" have counted would be choosing a denominator with the answer already in
view. Computing it first, from a declaration whose digest is pinned in the
artifact, removes that freedom.

What it does not do
-------------------
It creates no gate on a model, changes no tolerance, and never replaces the
declared eligibility rule. Its secondary view is descriptive.

No model is loaded. No checkpoint is read. Nothing is trained.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from ..american_pilot import physical_features
from ..artifact import write_json_atomic
from .attempts import PROTOCOL_CONFIG_PATH, assert_split_allowed, load_toml
from .frozen import _guarded_artifact_path, declaration_state, diagnostic_provenance
from .heldout import EVALUABLE_HALVES, evaluable_columns
from .tolerance import (
    ASSESSABILITY_DEGENERATE_LIMIT,
    SECONDARY_VIEW_DISCLOSURE,
    VEGA_DEGENERACY_NORMALIZED,
    assessability,
    declaration,
    eligible_vega_diagnostics,
)

ELIGIBILITY_SCHEMA: Final = "american-dev-eligibility/1"

#: The locked Task 9G dataset. Task 9H trains and evaluates on this and nothing
#: else, and the manifest is stripped to the two reachable splits before
#: anything reads it.
DATASET_DIRECTORY: Final = "data/american-option-v1"
DATASET_MANIFEST: Final = "data/american-option-v1/manifest.json"

#: Where each row set's artifact is written, beneath the ignored artifacts tree.
DEFAULT_OUTPUT_TEMPLATE: Final = "artifacts/task-9h/price/eligibility-{row_set}-v1.json"

#: The row sets this phase may assess. ``validation`` is the partition the loop
#: has selected against; ``H1`` is the reserved train half evaluated once. The
#: reserved half H2 is absent, and :mod:`.heldout` refuses it independently.
ASSESSABLE_ROW_SETS: Final = ("validation", *EVALUABLE_HALVES)


class EligibilityError(RuntimeError):
    """Raised when assessability is missing, stale or does not match its rows."""


def assert_row_set(row_set: str) -> str:
    """Resolve a row-set name, refusing anything this phase may not assess."""
    name = str(row_set)
    if name not in ASSESSABLE_ROW_SETS:
        raise EligibilityError(
            f"unknown row set {row_set!r}; this phase assesses {list(ASSESSABLE_ROW_SETS)}"
        )
    if name in ("validation",):
        assert_split_allowed(name)
    return name


def row_set_identity(row_set: str, columns: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Count and sample-identifier digest of the rows being assessed or scored.

    Digested over **sorted** identifiers, so the value names the row *set* and
    two derivations that select the same rows agree regardless of enumeration
    order. This is what ties an eligibility artifact to the rows a later scoring
    run actually reads.
    """
    identifiers = columns.get("sample_id")
    if identifiers is None:
        raise EligibilityError("a row set must supply sample_id to be identified")
    ordered = sorted(str(value) for value in identifiers)
    return {
        "row_set": assert_row_set(row_set),
        "rows": len(ordered),
        "sample_id_sha256": hashlib.sha256("\0".join(ordered).encode()).hexdigest(),
    }


def load_row_set(project_root: Path, row_set: str) -> dict[str, np.ndarray]:
    """Load one assessable row set's columns, through the existing guards.

    ``validation`` is the partition itself. ``H1`` is the reserved train half,
    obtained by loading ``train`` and passing it through
    :func:`.heldout.evaluable_columns` -- the single door that refuses the
    reserved half H2. Dataset identity is verified against the locked Task 9G
    values on the way, by the workbench's own verification.
    """
    from .workbench import load_partition, locked_dataset_identity, restricted_manifest

    name = assert_row_set(row_set)
    protocol = load_toml(Path(project_root) / PROTOCOL_CONFIG_PATH)
    locked = locked_dataset_identity(protocol)
    dataset = Path(project_root) / DATASET_DIRECTORY
    manifest = restricted_manifest(
        json.loads((Path(project_root) / DATASET_MANIFEST).read_text(encoding="utf-8"))
    )
    split = "validation" if name == "validation" else "train"
    columns = load_partition(dataset, manifest, split)
    if name == "validation":
        del locked
        return columns
    return evaluable_columns(columns, name)


def _public(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    """Strip the private per-row arrays a report must not carry."""
    return {key: value for key, value in diagnostics.items() if not key.startswith("_")}


def assess_rows(row_set: str, columns: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Assessability of one row set, without provenance. Reads no model output.

    Split from :func:`assess` so the measurement and the source-state attestation
    are separable: the attestation demands a clean worktree, and a test
    exercising the arithmetic should not have to commit to run.
    """
    identity = row_set_identity(row_set, columns)
    tensor = torch.as_tensor(physical_features(columns), dtype=torch.float64)
    european = np.asarray(columns["european_crr_price"], dtype=np.float64)
    diagnostics = eligible_vega_diagnostics(tensor, european)
    verdict = assessability(diagnostics)
    return {
        "schema_version": ELIGIBILITY_SCHEMA,
        "task": "task-9h-american-pricer-development",
        "row_set": identity,
        "status": {
            "reads_a_model": False,
            "reads_a_prediction": False,
            "trains_a_model": False,
            "computed_before_scoring": True,
            "creates_a_model_gate": False,
        },
        "tolerance": declaration(),
        "vega_diagnostics": _public(diagnostics),
        "assessability": verdict,
        "secondary_view_disclosure": SECONDARY_VIEW_DISCLOSURE,
    }


def assess(
    project_root: Path,
    row_set: str,
    columns: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """:func:`assess_rows` plus the source-state attestation.

    The attestation requires a clean tracked worktree, which is what stops a
    decision-bearing artifact from being produced before the protocol commit.
    """
    return {
        **assess_rows(row_set, columns),
        "provenance": diagnostic_provenance(project_root),
    }


def write(
    project_root: Path,
    row_set: str,
    columns: Mapping[str, np.ndarray],
    *,
    output: str | None = None,
    overwrite: bool = False,
) -> tuple[dict[str, Any], Path]:
    """Persist one row set's assessability artifact."""
    name = assert_row_set(row_set)
    relative = output or DEFAULT_OUTPUT_TEMPLATE.format(row_set=name)
    path = _guarded_artifact_path(project_root, relative, where="eligibility artifact path")
    if path.exists() and not overwrite:
        raise EligibilityError(
            f"'{relative}' already exists; assessability is computed once per row set, and "
            "recomputing it after seeing a score is exactly what the ordering forbids"
        )
    report = assess(project_root, name, columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, report, overwrite=overwrite)
    return report, path


def load(project_root: Path, row_set: str, *, output: str | None = None) -> dict[str, Any]:
    """Read one row set's assessability artifact, or explain that it is missing."""
    name = assert_row_set(row_set)
    relative = output or DEFAULT_OUTPUT_TEMPLATE.format(row_set=name)
    path = _guarded_artifact_path(project_root, relative, where="eligibility artifact path")
    if not path.is_file():
        raise EligibilityError(
            f"'{relative}' does not exist; assessability must be computed and persisted "
            f"before {name!r} is scored. Run the eligibility analysis first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def assert_matches(
    record: Mapping[str, Any],
    project_root: Path,
    row_set: str,
    columns: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Refuse to score unless the artifact really describes these rows and rules.

    Four things must line up, and each closes a different way the ordering could
    be defeated:

    * the **row-set identity**, so an artifact computed for one partition cannot
      license scoring another;
    * the **row digest**, so the artifact cannot describe a different subset of
      the same partition;
    * the **declaration digests**, so a tolerance edited after assessability was
      computed invalidates it;
    * the **protocol commit**, so an artifact from a different source state is
      not silently reused.
    """
    if record.get("schema_version") != ELIGIBILITY_SCHEMA:
        raise EligibilityError(
            f"eligibility artifact schema must be {ELIGIBILITY_SCHEMA!r}, got "
            f"{record.get('schema_version')!r}"
        )
    identity = row_set_identity(row_set, columns)
    recorded = record.get("row_set", {})
    for key in ("row_set", "rows", "sample_id_sha256"):
        if recorded.get(key) != identity[key]:
            raise EligibilityError(
                f"the eligibility artifact was computed for {key}={recorded.get(key)!r} but "
                f"scoring was asked for {identity[key]!r}; assessability describes exactly "
                "the rows it was computed on"
            )
    current = declaration_state(project_root)
    recorded_provenance = record.get("provenance", {})
    if recorded_provenance.get("declaration_digests") != current["declaration_digests"]:
        raise EligibilityError(
            "the declarations changed after this eligibility artifact was written; "
            "recompute assessability under the current protocol rather than scoring "
            "against a stale one"
        )
    if recorded_provenance.get("protocol_commit") != current["protocol_commit"]:
        raise EligibilityError(
            f"the eligibility artifact was written under protocol commit "
            f"{str(recorded_provenance.get('protocol_commit'))[:12]}… but the working tree is "
            f"at {str(current['protocol_commit'])[:12]}…; an artifact from another source "
            "state is not reused"
        )
    return dict(record)


def compact(record: Mapping[str, Any]) -> dict[str, Any]:
    """The small, agent-readable view: is the primary statistic assessable?"""
    degeneracy = record["vega_diagnostics"]["degeneracy"]
    return {
        "row_set": record["row_set"]["row_set"],
        "rows": record["row_set"]["rows"],
        "protocol_commit": record["provenance"]["protocol_commit"],
        "eligible_rows": record["vega_diagnostics"]["eligible_rows"],
        "eligible_fraction": record["vega_diagnostics"]["eligible_fraction"],
        "degenerate_eligible_fraction": degeneracy["degenerate_eligible_fraction"],
        "degeneracy_threshold_normalized_vega": VEGA_DEGENERACY_NORMALIZED,
        "degenerate_eligible_limit": ASSESSABILITY_DEGENERATE_LIMIT,
        "eligible_fraction_below_1e-4_normalized_vega": degeneracy[
            "eligible_fraction_below_1e-4_normalized_vega"
        ],
        "secondary_view_rows": record["vega_diagnostics"]["secondary_view"]["rows"],
        "verdict": record["assessability"]["verdict"],
        "primary_statistic_assessable": record["assessability"]["primary_statistic_assessable"],
    }


def conditional_subset(
    diagnostics: Mapping[str, Any], columns: Sequence[Any] | None = None
) -> np.ndarray:
    """The non-degenerate eligible subset, for the conditional pass rate.

    Reported when the primary statistic is NOT ASSESSABLE. It is a **different
    statistic** from the declaration and every report must say so; this function
    exists so that difference is explicit in the code rather than implied.
    """
    del columns
    eligible = np.asarray(diagnostics["_eligible"], dtype=bool)
    degenerate = np.asarray(diagnostics["_degenerate"], dtype=bool)
    return eligible & ~degenerate
