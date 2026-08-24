"""Task 9H held-out development rows: the H1/H2 carve, declared before use.

**These are ``train`` rows.** Every Task 9G and Task 9H training run used the
same 32,768 of ``train``'s 200,000 rows -- the lowest
``SHA-256(salt || NUL || sample_id)`` under salt ``task-9h-train-subset-v1``.
The complement is exactly computable and 167,232 rows have never been read by
any model in this project. This module carves 50,000 of them by a documented
deterministic rule and splits them, **in advance**, into two disjoint,
separately named halves of 25,000 each -- the same size as ``validation``, so
metrics are directly comparable.

* **H1** is used **once**, in this diagnostic phase.
* **H2** is reserved and untouched, for a final candidate before any
  confirmation study. **No code path here can score it.**

**"Held out" here never means a final partition.** The forbidden-token list in
:mod:`.attempts` reserves ``holdout`` -- a different word -- for the partitions
Task 9H may never reach, and this module deliberately does not use it. H1 and H2
are reserved *train* rows: same generator, same distribution, drawn from a split
Task 9H is explicitly permitted to read. They are **not** a substitute for an
independently generated partition and every report must describe them as what
they are.

**H2 is unreachable by construction, not by convention.**
:func:`assert_half_evaluable` refuses every half but H1, and every function that
turns a half into rows calls it first. :func:`carve` returns both halves'
indices because the declaration has to hash them, and that is the only place H2
indices exist.

**No selection gate may be created from H1.** It bounds accumulated selection
bias; it does not become a new criterion.

The carve rule is pure arithmetic over sample identifiers and needs no training
stack, so nothing here imports PyTorch at module scope. :func:`declare_carve`
defers its two artifact imports into the function body to keep that true.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from .attempts import AttemptError, assert_split_allowed, select_rows_by_hash

HELDOUT_SCHEMA: Final = "american-dev-heldout-declaration/1"

#: Where the declaration is written, beneath the ignored artifacts tree.
DEFAULT_OUTPUT: Final = "artifacts/task-9h/heldout/heldout-declaration-v1.json"

#: The partition the carve is taken from. ``train`` is one of the two splits
#: Task 9H may read, and the carve never leaves it.
CARVE_PARTITION: Final = "train"

#: The training-subset rule every Task 9G and Task 9H run used, restated here as
#: constants so the complement is computed from the same definition rather than
#: from a remembered number.
TRAIN_SUBSET_SALT: Final = "task-9h-train-subset-v1"
TRAIN_SUBSET_BUDGET: Final = 32768

#: The carve rule. One salt, one rank order, fully reproducible from these
#: constants and the partition's ``sample_id`` column alone.
CARVE_SALT: Final = "task-9h-heldout-v1"
CARVE_SIZE: Final = 50_000
HALF_SIZE: Final = 25_000

HALVES: Final = ("H1", "H2")
#: The only half any code path in this phase may turn into rows.
EVALUABLE_HALVES: Final = ("H1",)

CARVE_RULE: Final = (
    "T = lowest 32768 of SHA-256('task-9h-train-subset-v1' || NUL || sample_id) over "
    "train's 200000 rows, sample_id tie-break -- the rows every Task 9G and Task 9H run "
    "trained on. C = train \\ T, 167232 rows never read by any model. Rank C by "
    "SHA-256('task-9h-heldout-v1' || NUL || sample_id), sample_id tie-break; the lowest "
    "50000 form the carve. H1 = carve ranks 1..25000, H2 = carve ranks 25001..50000."
)

EXCHANGEABILITY: Final = (
    "H1 and H2 differ only in SHA-256 rank under a fixed salt, which is independent of "
    "every feature and of the label, so the two halves are exchangeable draws from the "
    "same distribution as train and validation"
)

NOT_AN_INDEPENDENT_PARTITION: Final = (
    "H1 is reserved train rows, not an independently generated partition: same generator, "
    "same sampling design, same declared domain. It bounds accumulated selection bias; it "
    "does not establish out-of-distribution behaviour and it is not a confirmation set."
)

NO_SELECTION_GATE: Final = (
    "H1 is evaluated once, in this diagnostic phase. No threshold, gate or model-selection "
    "rule may be derived from it, and H2 is not touched in this task under any circumstances."
)


class HeldoutAccessError(AttemptError):
    """Raised when anything would turn a reserved half into evaluable rows.

    Subclasses :class:`.attempts.AttemptError` so a caller that already fails
    closed on partition-identity problems fails closed on this one too.
    """


@dataclass(frozen=True)
class HeldoutCarve:
    """The carve, as row positions into the ``train`` partition's own order."""

    train_rows: int
    training_subset: np.ndarray
    complement: np.ndarray
    carve: np.ndarray
    h1: np.ndarray
    h2: np.ndarray


def assert_half_evaluable(half: str) -> str:
    """Refuse every half but H1. The single gate on turning a half into rows."""
    name = str(half)
    if name not in HALVES:
        raise HeldoutAccessError(f"unknown held-out half {half!r}; the halves are {list(HALVES)}")
    if name not in EVALUABLE_HALVES:
        raise HeldoutAccessError(
            f"half {name!r} is reserved and is not evaluated in this task under any "
            f"circumstances; only {list(EVALUABLE_HALVES)} may be turned into rows"
        )
    return name


def _rank_key(salt: str, sample_id: Any) -> tuple[bytes, str]:
    """The same key :func:`.attempts.select_rows_by_hash` ranks by."""
    return hashlib.sha256(f"{salt}\0{sample_id}".encode()).digest(), str(sample_id)


def carve(sample_ids: Sequence[Any]) -> HeldoutCarve:
    """Derive the training subset, its complement, the carve and both halves.

    Pure: given the partition's ``sample_id`` column in its stored order, this
    returns the same positions on every machine and in every Python version, and
    it reads no other column, no label and no model.

    The training subset is recomputed with :func:`.attempts.select_rows_by_hash`
    -- the function the attempts themselves used -- rather than reproduced here,
    so the complement cannot drift from the rows that were actually trained on.
    """
    ids = list(sample_ids)
    rows = len(ids)
    if rows != len(set(str(sample_id) for sample_id in ids)):
        raise HeldoutAccessError("sample identifiers must be unique to carve deterministically")
    if rows <= TRAIN_SUBSET_BUDGET + CARVE_SIZE:
        raise HeldoutAccessError(
            f"{rows} rows cannot supply a {TRAIN_SUBSET_BUDGET}-row training subset and a "
            f"disjoint {CARVE_SIZE}-row carve"
        )
    training_subset = select_rows_by_hash(ids, TRAIN_SUBSET_BUDGET, TRAIN_SUBSET_SALT)
    trained = np.zeros(rows, dtype=bool)
    trained[training_subset] = True
    complement = np.flatnonzero(~trained)
    if complement.size != rows - TRAIN_SUBSET_BUDGET:
        raise HeldoutAccessError("the complement is not the training subset's exact complement")
    ranked = sorted((*_rank_key(CARVE_SALT, ids[index]), int(index)) for index in complement)
    carved = np.asarray([entry[2] for entry in ranked[:CARVE_SIZE]], dtype=np.int64)
    return HeldoutCarve(
        train_rows=rows,
        training_subset=np.sort(training_subset),
        complement=complement,
        carve=carved,
        h1=carved[:HALF_SIZE],
        h2=carved[HALF_SIZE:CARVE_SIZE],
    )


def half_index(carved: HeldoutCarve, half: str) -> np.ndarray:
    """Row positions of one **evaluable** half. Refuses a reserved half."""
    name = assert_half_evaluable(half)
    return np.sort(carved.h1) if name == "H1" else np.sort(carved.h2)


def _identity(sample_ids: Sequence[Any], index: np.ndarray) -> dict[str, Any]:
    """Count and digest of one row set, over its sorted sample identifiers.

    Sorted so the digest identifies the *set*, not an ordering: two derivations
    that select the same rows agree even if they enumerated them differently.
    """
    selected = sorted(str(sample_ids[position]) for position in index)
    digest = hashlib.sha256("\0".join(selected).encode()).hexdigest()
    return {"rows": len(selected), "sample_id_sha256": digest}


def declare(sample_ids: Sequence[Any]) -> dict[str, Any]:
    """The full declaration: rule, counts, digests and verified disjointness.

    Written **before** either half is evaluated. Disjointness is asserted, not
    described: a carve that overlapped the training rows would make the whole
    point of H1 -- rows no model could have seen -- false.
    """
    carved = carve(sample_ids)
    sets = {
        "training_subset": set(carved.training_subset.tolist()),
        "H1": set(carved.h1.tolist()),
        "H2": set(carved.h2.tolist()),
    }
    if len(sets["H1"]) != HALF_SIZE or len(sets["H2"]) != HALF_SIZE:
        raise HeldoutAccessError(f"each half must hold exactly {HALF_SIZE} distinct rows")
    disjoint = {
        "H1_and_H2": not (sets["H1"] & sets["H2"]),
        "H1_and_training_subset": not (sets["H1"] & sets["training_subset"]),
        "H2_and_training_subset": not (sets["H2"] & sets["training_subset"]),
    }
    if not all(disjoint.values()):
        raise HeldoutAccessError(f"the carve is not disjoint as declared: {disjoint}")
    return {
        "schema_version": HELDOUT_SCHEMA,
        "partition": assert_split_allowed(CARVE_PARTITION),
        "rule": CARVE_RULE,
        "constants": {
            "train_subset_salt": TRAIN_SUBSET_SALT,
            "train_subset_budget": TRAIN_SUBSET_BUDGET,
            "carve_salt": CARVE_SALT,
            "carve_size": CARVE_SIZE,
            "half_size": HALF_SIZE,
            "tie_break": "sample_id",
        },
        "counts": {
            "train_rows": carved.train_rows,
            "training_subset": int(carved.training_subset.size),
            "complement_never_read": int(carved.complement.size),
            "carve": int(carved.carve.size),
        },
        "halves": {
            "H1": {
                **_identity(sample_ids, carved.h1),
                "use": "evaluated once, in this diagnostic phase",
                "evaluable": True,
            },
            "H2": {
                **_identity(sample_ids, carved.h2),
                "use": "reserved for a final candidate before any confirmation study",
                "evaluable": False,
            },
        },
        "training_subset_identity": _identity(sample_ids, carved.training_subset),
        "disjointness_verified": disjoint,
        "exchangeability": EXCHANGEABILITY,
        "not_an_independent_partition": NOT_AN_INDEPENDENT_PARTITION,
        "no_selection_gate": NO_SELECTION_GATE,
    }


def evaluable_columns(
    columns: Mapping[str, np.ndarray], half: str
) -> dict[str, np.ndarray]:
    """Restrict a loaded ``train`` partition to one evaluable half's rows.

    The single door between the carve and anything that computes a metric, and
    it is closed on H2.
    """
    name = assert_half_evaluable(half)
    sample_ids = columns.get("sample_id")
    if sample_ids is None:
        raise HeldoutAccessError("the partition must supply sample_id to be carved")
    index = half_index(carve(list(sample_ids)), name)
    return {key: np.asarray(value)[index] for key, value in columns.items()}


# ---------------------------------------------------------------------------
# The declaration artifact
# ---------------------------------------------------------------------------


def declare_carve(
    project_root: Any,
    *,
    output: str = DEFAULT_OUTPUT,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Persist the carve declaration, **before** either half is evaluated.

    Writing it requires a clean tracked worktree, through the shared provenance
    helper: a declaration whose source state cannot be named is not evidence
    that anything was declared in advance.

    The two artifact imports are deferred into this function so importing this
    module stays free of the training stack.
    """
    from pathlib import Path

    from ..artifact import write_json_atomic
    from .eligibility import DATASET_DIRECTORY, DATASET_MANIFEST
    from .frozen import _guarded_artifact_path, diagnostic_provenance
    from .workbench import load_partition, restricted_manifest

    project_root = Path(project_root)
    path = _guarded_artifact_path(project_root, output, where="carve declaration path")
    if path.exists() and not overwrite:
        raise HeldoutAccessError(
            f"'{output}' already exists; the carve is declared once, and redeclaring it after "
            "a half has been evaluated is exactly what declaring in advance rules out"
        )
    import json as _json

    manifest = restricted_manifest(
        _json.loads((project_root / DATASET_MANIFEST).read_text(encoding="utf-8"))
    )
    columns = load_partition(project_root / DATASET_DIRECTORY, manifest, CARVE_PARTITION)
    report = {
        "schema_version": HELDOUT_SCHEMA,
        "task": "task-9h-american-pricer-development",
        **declare(list(columns["sample_id"])),
        "provenance": diagnostic_provenance(project_root),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, report, overwrite=overwrite)
    return report


def compact_carve(report: Mapping[str, Any]) -> dict[str, Any]:
    """The small, agent-readable view of the declaration."""
    return {
        "protocol_commit": report.get("provenance", {}).get("protocol_commit"),
        "partition": report["partition"],
        "counts": report["counts"],
        "H1_rows": report["halves"]["H1"]["rows"],
        "H1_sample_id_sha256": report["halves"]["H1"]["sample_id_sha256"],
        "H2_rows": report["halves"]["H2"]["rows"],
        "H2_sample_id_sha256": report["halves"]["H2"]["sample_id_sha256"],
        "H2_evaluable": report["halves"]["H2"]["evaluable"],
        "disjointness_verified": report["disjointness_verified"],
        "not_an_independent_partition": report["not_an_independent_partition"],
    }
