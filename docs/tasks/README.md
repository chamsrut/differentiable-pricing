# Task index

Every task specification lives at a **stable path** under
`tasks/active/`, whatever its lifecycle state. This repository marks task
status **in place** and never relocates a spec: a completed or deferred spec
stays where it was written so that every link recorded in
[../decision-log.md](../decision-log.md) — an append-only file whose entries
are never edited — keeps resolving, and so that the two source files that name
a spec path directly keep working
(`python/tests/test_pde_label_policy_v2.py` and
`python/src/differentiable_pricing/american/pde_label_policy_v2.py`, the latter
digest-pinned inside a frozen snapshot's executable-source inventory).

This file is that lifecycle grouping. It is the index; the directory layout is
not. The authoritative status of any task is the `## Status` block inside its
own spec.

## Active

Exactly one forward-looking task is active.

| Task | Spec | State |
|---|---|---|
| 9I — Architecture v2.3 transition, Phase 0 | [active/task-9i-architecture-v2-3-phase-0.md](active/task-9i-architecture-v2-3-phase-0.md) | `Active` — two parallel exploratory workstreams; opens no final partition and trains no new attempt |

Its normative parent is
[../american-neural-architecture-freeze-v2.3.md](../american-neural-architecture-freeze-v2.3.md).

## Completed

Terminal. The recorded outcome — positive or negative — is a historical fact
and is never revised.

| Task | Spec | Outcome |
|---|---|---|
| 9C-C3 — PDE label-policy v2 | [active/task-9c-c3-label-policy-v2.md](active/task-9c-c3-label-policy-v2.md) | `Completed` — selected `grid_1600x800`, externally approved (DEC-025) |
| 9D — local data-holdings catalogue and integrity audit | [active/task-9d-data-holdings-audit.md](active/task-9d-data-holdings-audit.md) | `Completed` — catalogue published, admitted nothing (DEC-031) |
| 9E — CRR dataset admission | [active/task-9e-crr-dataset-admission.md](active/task-9e-crr-dataset-admission.md) | Closed **conditional and partial**: mapping-only admission approved (DEC-035); the cross-check and semantic-coverage gates were not discharged and the near-duplicate gate cannot be satisfied retroactively for this dataset version (DEC-034). Terminal for this dataset version; authorizes no training |
| 9G — American neural-pricer feasibility pilot | [active/task-9g-american-neural-pricer-pilot.md](active/task-9g-american-neural-pricer-pilot.md) | `validation_gates_failed` / `failure_to_learn` — frozen negative result (DEC-038); final partition unconsumed; `final-evaluate` forbidden under that protocol |

## Deferred / on hold

Not started, not rejected. Each states its own resumption condition; none is
work to pick up.

| Task | Spec | Resumption condition |
|---|---|---|
| 9C-C2b2 — deterministic parallel/resumable generation | [active/task-9c-c2b2-parallel-resumable-generation.md](active/task-9c-c2b2-parallel-resumable-generation.md) | The XSP/SPY phase requires dataset-scale PDE generation (DEC-029) |
| 9F — private object storage and vendor ingestion | [active/task-9f-remote-data-access-plan.md](active/task-9f-remote-data-access-plan.md) | A recorded decision that the project needs machine-to-machine access to its own archive |

## Archived exploratory line

**Task 9H — adaptive American price-model development** is not on this branch.
Its specification, attempt ledger, interim research brief, and decision entries
`DEC-041`–`DEC-048` live only on the unmerged branch
`experiment/task-9h-american-pricer-development`, archived at tag
`task-9h-v1-exploratory-pre-v2.3` (commit `3950ed0`). Nothing it produced is a
project result; it is exploratory input to the v2 design. Task 9I carries
forward only the Greek/head diagnostics named in its own spec.

Decision IDs `DEC-039`–`DEC-048` are reserved by that archived line and are
therefore intentionally absent from this branch's linear decision history.

## Writing a new task

Copy [TEMPLATE.md](TEMPLATE.md) to `active/task-<id>-<short-name>.md` and fill
every section. On completion, mark the spec `Completed` **in place**, update
[../project-state.md](../project-state.md), append a
[../decision-log.md](../decision-log.md) entry, and move the row in this file
from "Active" to "Completed".
