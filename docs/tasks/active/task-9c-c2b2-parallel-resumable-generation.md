# Task 9C-C2b2: deterministic parallel/resumable generation infrastructure

## Status

`Deferred` — **not started, and not rejected.** No infrastructure, no pilot,
no dataset.

This task became reachable when task 9C-C3 produced an accepted label policy
(`grid_1600x800`, [../../decision-log.md](../../decision-log.md) DEC-025). It
supersedes nothing: 9C-C3 is `Completed` and terminal
([task-9c-c3-label-policy-v2.md](task-9c-c3-label-policy-v2.md)).

It was then **deferred** by the stage-2 roadmap lock
([../../decision-log.md](../../decision-log.md) DEC-028, DEC-029). Dataset-scale
PDE generation is not needed until the XSP/SPY phase, and building the
machinery before the experiment that justifies it is the ordering DEC-012
already ruled against in the small.

**Resumption condition:** the XSP/SPY phase requires dataset-scale PDE
generation. Until then no work proceeds here.

**Nothing below is rejected or superseded.** The objective, scope boundaries,
`OPEN` conventions, gates and stop conditions in this file remain the
specification to resume from, unchanged. This file stays at this path — the
repository marks task status in place and has no `tasks/deferred/` location —
so every link to it stays valid.

The single active task is
[task-9d-data-holdings-audit.md](task-9d-data-holdings-audit.md) (DEC-030).

## Objective

Can PDE label generation under the accepted policy be made **deterministic,
parallel and resumable** — producing byte-identical output regardless of worker
count, chunk order, or how many times a run is interrupted and resumed?

A positive answer is infrastructure that demonstrably satisfies that property
on bounded validation and one small, manually invoked pilot. A negative answer
— a determinism or resumption property that cannot be met without abandoning
grouped partitioning or the accepted policy — is a valid, recordable outcome
that stops here rather than being worked around downstream.

**This task does not produce a dataset.** It produces the machinery that a
later, separately gated task would use.

## Motivation

Task 9C-B measured the cost that makes this necessary: thirteen scalar solves
per state and grid, 1,638 solves and about 126.6 million PSOR iterations for 28
states. Tasks 9C-C1/9C-C2a/9C-C2b1 attacked that algorithmically first — one
solve returning a whole valuation-time surface, grouped partitioning, a
three-surface vega — deliberately **before** buying parallelism
([../../decision-log.md](../../decision-log.md) DEC-012). Task 9C-C3 then
supplied the missing precondition: a policy worth generating against.
Parallelism is now the remaining throughput lever, and it is only worth
building because the algorithmic work already happened.

The ordering constraint from DEC-012 still binds in the small: parallelism may
not paper over a correctness or leakage property. Determinism and grouped
partitioning are requirements of this task, not optimizations of it.

## Authoritative inputs

- [../../pde-numerical-contract.md](../../pde-numerical-contract.md) — the PDE
  solver contract, the task 9C-C1 surface, the task 9C-C2a grouped-harvesting
  rules, and "Task 9C-C3: label-policy v2" including "Outcome: the accepted v2
  result". Authoritative for every accepted number; this file restates none.
- [../../results/american_pde_label_policy_v2_results_v1.json](../../results/american_pde_label_policy_v2_results_v1.json)
  — the frozen, externally approved terminal evidence for the accepted policy.
  **Read-only. Never edited, regenerated, reformatted or refreshed.**
- [../../decision-log.md](../../decision-log.md) DEC-004 (surface reuse),
  DEC-005 (group-level partitioning before any solve), DEC-006 (rows are not an
  effective sample size), DEC-009 (gamma evaluation-only), DEC-011 (no approved
  training-input status), DEC-012 (algorithmic reuse before clusters), DEC-014
  (no final dataset before an accepted policy), DEC-025/DEC-026/DEC-027 (the
  accepted policy, the 18/22 delta reading, and snapshot immutability).
- [../../architecture.md](../../architecture.md) — artifact identity and
  provenance rules any new output must satisfy.

## In scope

- Deterministic parallel execution of independent solve groups: identical
  output bytes for any worker count, any chunk size, and any scheduling order.
- Resumability: an interrupted run resumes without recomputing completed
  groups and without producing output that differs from an uninterrupted run.
- A manifest/provenance discipline that makes an interrupted or partial run
  **unmistakable** as incomplete, per the task 9C-C2a precedent, so a partial
  output can never read as a finished one.
- Bounded validation of both properties, cheap enough to live in the test
  suite: determinism across worker counts, and resume-equals-uninterrupted.
- Predeclaring this task's **own generation domain**. It is **not** inherited
  from task 9C-C3's 28 evidence cases, which are isolated points that validate
  no surrounding hyperrectangle.
- One **small, manually invoked pilot** demonstrating the machinery end to end
  under the accepted policy, with raw output under ignored `artifacts/`.

## Out of scope

- **Production dataset generation.** This task must not claim or launch one,
  and its pilot is a demonstration of machinery, not a dataset. A production
  run is a separate, separately gated task.
- **Neural training** of any American surrogate, and any change to
  `AUTHORIZED_TRAINING_INPUT_STATUSES` (DEC-011, DEC-014).
- Any reinterpretation of task 9C-C3: no re-running a stage, no widening the
  supported order band, no revisiting the four delta exclusions (DEC-026), and
  no edit to the frozen v2 snapshot or config (DEC-027).
- Any change to gamma's evaluation-only status (DEC-009) or to Richardson's
  validation-only role (DEC-010).
- Any change to the C++ engine, its bindings, or `pde_surface_harvest` — all of
  which are digest-pinned in the frozen v2 snapshot's executable-source
  inventory. A change to any of them breaks that snapshot's provenance
  reconciliation and needs its own recorded decision first.
- Buying or configuring cluster hardware. "Parallel" here means deterministic
  multi-worker execution, not a scaling procurement.

## Predeclared conventions

Fixed and versioned **before** the pilot run, in this task's own configuration:

- **The generation domain** — the parameter regions labels will be generated
  over. Still `OPEN`; it must be written down and versioned before the pilot,
  and it is not inherited from 9C-C3's case list.
- **Group definition and partition assignment** — unchanged from DEC-005:
  assigned before any solve, and a group may never straddle
  `train`/`validation`/`interpolation_test`.
- **Determinism claim and its preconditions** — exactly which inputs must be
  identical for byte identity to be claimed, stated as explicitly as task
  9C-C2a states its own. Still `OPEN`.
- **Pilot size** — small, and fixed before the run so it cannot grow into a
  production run after the fact. Still `OPEN`.
- **Label content per row** — price, and delta/vega only where the accepted
  policy makes them eligible; gamma is carried as evaluation-only, never as a
  supervised target.

Do not invent a value for any `OPEN` item during a run. Resolve and version it
first, and record the resolution in the decision log.

## Acceptance gates

- **Entry:** an accepted label policy exists to generate against — satisfied by
  DEC-025 — and every `OPEN` convention above is resolved and versioned.
- **Determinism gate:** output bytes identical across at least two worker
  counts and two chunk sizes, under the stated preconditions.
- **Resumption gate:** a run interrupted and resumed produces output
  byte-identical to an uninterrupted run of the same configuration.
- **Leakage gate:** no group straddles a partition, checked mechanically, not
  by inspection.
- **Exit:** the gates above pass, the bounded validation lives in the test
  suite, and the small pilot has been run once, manually, with its outcome
  recorded. Exit does **not** authorize a production dataset.
- **Failure:** a determinism or resumption property that cannot be met is
  recorded as a negative result with its cause. It is not worked around by
  weakening grouped partitioning or the accepted policy.

## Manual-run protocol

The pilot is a manual, terminal-invoked job, per `AGENTS.md` ("No
agent-supervised expensive numerical runs"). No hook, CI job or agent may
launch, schedule, or gate on it. Bounded determinism/resumption validation is
cheap and belongs in the test suite; the pilot does not.

The exact commands are defined when this task's runner and configuration are
implemented, and will live in
[../../pde-numerical-contract.md](../../pde-numerical-contract.md) — not
duplicated here.

## Artifact policy

Raw pilot output goes to ignored `artifacts/`, never committed, and is refused
by `verify_training_input_publication` exactly as the 9C-C2a and 9C-C2b1
demonstrations are (DEC-011). A manifest is written **last** and pins the
digests of everything it describes, so an interrupted run cannot read as a
completed one.

Whether this task freezes a `docs/results/` snapshot is itself `OPEN`: its
output is machinery plus a demonstration, not a numerical acceptance result. If
one is frozen, it gets its own designated freeze/check script and a `--check`
wired into `scripts/check.sh` and CI, following the task 9C-C3 precedent.

## Review requirements

`code-reviewer` for the generation, scheduling and resumption code.
`numerical-reviewer` for determinism, partitioning and any claim about label
content or eligibility. Both are separate-context **preliminary** review
([../../agent-system.md](../../agent-system.md)).

Accepting this task as complete — and, later, authorizing any production
dataset — is a **material approval** requiring a fresh top-level session with
no anchoring on the implementer's reasoning.

## Stop conditions

- Any `OPEN` convention above is still unresolved when the pilot is about to
  run: stop and resolve it first.
- The pilot is about to exceed its predeclared size, or is being described as a
  production dataset: stop. That is a separate task with its own gate.
- Determinism or resumption cannot be achieved without weakening grouped
  partitioning: stop and record the negative result.
- Any change becomes necessary to a file in the frozen v2 snapshot's
  executable-source inventory: stop and record a decision first, because it
  breaks terminal evidence provenance.

## Completion report

Record, unconditionally: which gates passed and failed; the determinism and
resumption evidence; the pilot's exact size and outcome; the resolved values of
every convention that was `OPEN`; an update to
[../../project-state.md](../../project-state.md) stating the new current state
and the exact next task; and a new entry in
[../../decision-log.md](../../decision-log.md) recording the outcome, positive
or negative. State explicitly that no production dataset and no training input
was authorized by this task.
