# Task 9I: Architecture v2.3 transition — Phase 0

## Status

`Active` — **exploratory only.** Two independent workstreams run in parallel.
Neither may open a final partition, train a new neural attempt, retune a frozen
result, or produce a project result.

This task is opened by [../../decision-log.md](../../decision-log.md) DEC-049,
which adopts
[../../american-neural-architecture-freeze-v2.3.md](../../american-neural-architecture-freeze-v2.3.md)
as the normative parent of the American neural-pricer roadmap and supersedes
the previous Task 9H development roadmap.

## Objective

Produce the two exploratory inputs that Architecture Freeze v2.3 requires
before its Phase 1 (freezing the v2 architecture and the acceptance/reference
protocol) can begin:

- **A — head/representation guidance** from the carried-forward Greek
  diagnostics, answering: is the observed Greek behaviour a
  learning/representation problem, a deployment-head problem, or a
  projection-design problem, and is the Greek reference converged enough to
  support any continuous-American Greek claim at all?
- **B — native-backend feasibility evidence**, answering: is a native C++
  implementation of a frozen deployed neural pricing function worth the full
  engineering investment, measured against a matched native CRR request?

A negative or inconclusive answer on either side is a valid completion. Neither
answer decides which model becomes the v2 confirmation candidate; the freeze
already settles that it cannot be E2c.

## Motivation

Task 9G's locked run returned a frozen negative result
(`failure_to_learn`, DEC-038). The Task 9H exploratory line that followed it
reached a recorded price leader and then paused its attempt loop for a
zero-training diagnostic phase; that phase was predeclared and implemented but
its decision-bearing grids had not been scored. External review of Architecture
Freeze v2 then produced v2.3, which reorganizes the remaining work: the Greek
diagnostics and the native prototype become **independent parallel exploratory
workstreams**, and the v2 confirmation model must be retrained from a dataset
reproducible from committed source.

Finishing the diagnostics is therefore worth doing — they inform the v2
head/representation design — while E2c's weights are worth keeping only as a
concrete specimen for the native prototype.

Freeze sections: §0A, §13.2, §23 (Phase 0A and Phase 0B).

## Authoritative inputs

- [../../american-neural-architecture-freeze-v2.3.md](../../american-neural-architecture-freeze-v2.3.md)
  — normative parent. This spec does not restate its rules, thresholds,
  formulas, or sequencing; where the two differ, the freeze wins.
- [../../architecture.md](../../architecture.md) — language boundary, artifact
  identity, `american_forward_carry_v1`, the exact European-to-American lift.
- [../../american-crr-contract.md](../../american-crr-contract.md) — CRR
  lattice, adjacent-average semantics, complexity.
- [../../research-contract.md](../../research-contract.md) — partition
  discipline, metrics, provisional gates, non-claims.
- [../../decision-log.md](../../decision-log.md) DEC-034, DEC-036, DEC-038,
  DEC-049.
- [../../results/american_neural_pilot_results_v1.json](../../results/american_neural_pilot_results_v1.json)
  — frozen, never regenerated or reinterpreted.
- The archived Task 9H exploratory line: branch
  `experiment/task-9h-american-pricer-development`, tag
  `task-9h-v1-exploratory-pre-v2.3` (commit `3950ed0`). It carries the 9H
  workbench, the append-only attempt ledger, the interim research brief, the
  predeclared diagnostic protocol, the E2c checkpoint, and decision entries
  `DEC-041`–`DEC-048`. **None of it is merged into `main` or into this
  branch**, and none of it is a project result.

## In scope

### Workstream A — finish the carried-forward Greek/head diagnostics

Carry forward, from the archived 9H line, **only** the diagnostics the freeze
names in §23 Phase 0A, and complete them:

1. the depth-convergence qualification propagation;
2. recording the reference-contract digest **and** the reference protocol
   commit in Grid-1 artifacts;
3. the normalized-Vega degeneracy units repair (freeze §17.1) and the
   three-bucket Vega-excess report (§17.3);
4. removal of Gamma degeneracy as a reported statistic. Gamma has no
   degeneracy metric: the Vega metric exists only because the declared
   downstream IV Newton workflow divides by Vega, and no declared workflow
   divides by Gamma (§17.2, §17.5). Gamma is assessed on accuracy,
   sign/convexity behaviour, and crossover curvature instead;
5. the dimensional-invariance regression test for B.2 eligibility (§6);
6. then the runs: Grid 1 validation, Grid 1 H1, Grid 2 spot crossover, Grid 2b
   volatility crossover.

The output is head/representation guidance in the four-way form the freeze
states in §23 Phase 0A.

### Workstream B — native C++ E2c feasibility prototype

Independently, and in parallel, build the minimal native prototype the freeze
specifies in §13.2: the exact current frozen E2c deployed function in float64,
with the exact feature representation, MLP, analytic European leg, intrinsic
leg, and smooth floor/projection — no retraining, no architecture shrinkage,
no float32. Establish numerical equivalence against fixed Python test vectors
**first**, then benchmark batch 1 and batch 8 against the same matched native
C++ CRR adjacent-average request under the §14 protocol.

E2c is a concrete specimen here, nothing more.

## Out of scope

- **Any neural training attempt.** No new attempt is spent in this task, and no
  attempt budget is consumed.
- **Opening `interpolation_test` or any other final partition**, by any code
  path, including hashing, stat-ing, counting, or importing it. Task 9G's
  `final-evaluate` remains **forbidden** under its own protocol; a future final
  evaluation requires a new predeclared protocol and a fresh final partition
  (DEC-038).
- **Retaining or promoting E2c's weights as the v2 confirmation model.** The
  freeze settles this: the v1 teacher dataset is not reproducible from
  committed source, so E2c is not eligible (§0A.2, §13.2).
- **Freezing the v2 architecture, application tolerances, reference-error
  budget, attempt budget, sampling design, or native benchmark contract.** That
  is the freeze's Phase 1 and needs its own task.
- **Restoring the reproducible American generator or generating any v2
  partition.** That is Phase 2.
- **Rerunning, retuning against, regenerating, reformatting, or reinterpreting
  any frozen snapshot under `docs/results/`.**
- **Native Greek latency claims.** No native Greek code exists; the freeze
  forbids advertising one until it does (§16.1).
- Merging the archived 9H branch wholesale. Only the named diagnostics are
  carried forward, and each carry-forward is a reviewed change on its own.

## Predeclared conventions

The freeze fixes the semantics; this task fixes nothing numerically new.
Carried forward unchanged from the freeze and the archived 9H declaration:

- the application price-tolerance family — the vol-equivalent criterion where
  assessable, and the flat normalized `5.5e-4` practical screen (freeze §9.3);
- the per-row resolvability rule and the assessability rule that reports a
  primary pass rate `NOT ASSESSABLE` when more than 1% of eligible rows are
  vega-degenerate;
- the Vega degeneracy threshold, applied in **declared normalized units** —
  `|Vega_physical| / A ≤ 1e-8` with `A = S exp(-qT)` — to the surrogate and the
  reference identically (freeze §17.1);
- the three mutually exclusive Vega-excess buckets (freeze §17.3);
- **no Gamma-degeneracy statistic is defined, required, or expected.** Freeze
  §17.5 settles this: a degeneracy statistic exists for Vega because the
  declared downstream IV Newton workflow divides by Vega, and none is defined
  for Gamma unless a future declared workflow has a denominator or failure
  mechanism that makes Gamma-near-zero operationally relevant. This task does
  not invent one for symmetry, and its absence is not an open item;
- H1 is evaluated once; **H2 is unreachable by construction** and is not
  touched in this task.

Still `OPEN` and **not** invented here — each must be resolved by a recorded
decision before the run it gates:

- `OPEN` — the exact native benchmark machine, thread budget, affinity, and
  timing clock for workstream B. The §14 protocol constrains the shape; the
  concrete values must be recorded before the first reported timing.
- `OPEN` — which native dense/tanh implementation is acceptable as a maintained
  dependency (freeze §24, open question 7). The prototype measures candidates;
  it does not select one.

## Acceptance gates

This task is exploratory; its gates are **completion gates, not pass/fail
criteria on a model**. It completes when all of the following hold.

**Workstream A**

- A1 — every repair in "In scope" A.1–A.5 is implemented, tested, and
  committed **before** any Grid-1 model result is observed, and the commit
  record states explicitly that no Grid-1 model result had been observed under
  either the old or the repaired degeneracy definition when the repair was
  made (freeze §23 Phase 0A.6).
- A2 — any artifact produced under a superseded semantic definition is
  preserved byte-for-byte, and the repaired result is published under a new
  schema/path with an explicit `supersedes` link. Nothing decision-bearing is
  overwritten (freeze §23 Phase 0A.7).
- A3 — Grid 1 validation, Grid 1 H1, Grid 2, and Grid 2b have each run, and
  each conclusion carries its reference-convergence qualification. Where the
  Greek reference is not converged under its predeclared rule, the report
  states the reference limitation and makes **no** continuous-American Greek
  claim (freeze §16.5).
- A4 — the head/representation guidance is recorded in the four-way form of
  freeze §23 Phase 0A.

**Workstream B**

- B1 — native/Python numerical equivalence against the fixed test vectors is
  demonstrated **before** any timing is reported.
- B2 — matched batch-1 and batch-8 timings exist under the §14 protocol,
  measured in the same executable, process, dtype, and thread budget, with no
  timed call falling back into Python.
- B3 — the report distinguishes measured timings from engineering estimates.
  Memory-bandwidth and FLOP arithmetic may be recorded as a plausibility check
  and **never** as a latency result; no 10x/20x/40x speedup is asserted
  (freeze §13.3, §0.12).

**On failure of any gate:** the task stops at that gate and reports the
failure. An inconclusive prototype (B) or a non-converged Greek reference (A)
is a completion with a negative finding, not a reason to relax a declared rule.

## Manual-run protocol

Every numerical run in this task is a **manual, human-invoked terminal job**,
per [../../../AGENTS.md](../../../AGENTS.md). No hook, CI job, or agent may
launch one, schedule one, or treat its completion as a gate. An agent may state
the documented command; a human runs it and reviews the raw report.

Exact commands are not fixed here because the carried-forward workbench is not
yet on this branch. Before the first run, this section must be updated to name
the exact command, its order, and its preconditions — including the clean
tracked worktree requirement the 9H workbench already enforces.

Ordering is structural, not advisory: the assessability artifact is produced by
its own command and its own artifact, and price fidelity refuses to score a row
set whose eligibility artifact does not match its rows, declaration digests, and
protocol commit.

## Artifact policy

- Raw output goes to ignored `artifacts/`. Nothing under `artifacts/` is
  evidence, and nothing there is committed.
- **Nothing in this task freezes a `docs/results/` snapshot.** These are
  exploratory diagnostics; the freeze reserves frozen confirmatory evidence for
  the v2 campaign's Phase 5.
- Diagnostic artifacts record the reference-contract digest, the reference
  protocol commit, the attempt identity, and the source digests they ran from.
- Superseded artifacts are preserved byte-for-byte with a `supersedes` link, per
  gate A2. No historical decision-bearing evidence is overwritten.
- Attempt-ledger discipline is unchanged: append-only, an existing entry is
  never rewritten. This task records **zero** new neural attempts.

## Review requirements

- `code-reviewer` on every non-trivial diff, including each carry-forward of
  9H machinery and the native prototype.
- `numerical-reviewer` on the degeneracy-units repair, the depth-convergence
  propagation, the eligibility dimensional-invariance test, the Grid
  interpretations, and the native/Python equivalence check.
- Both are **preliminary** and are never described as final approval.
- Material approval — accepting the head/representation guidance as an input to
  the v2 freeze, or accepting the native prototype as evidence that a native
  backend is worth building — requires a **fresh top-level session** with no
  prior anchoring, per [../../agent-system.md](../../agent-system.md).

## Stop conditions

Work stops before completion if any of the following occurs.

- Any code path opens, hashes, stats, imports, counts, or inspects
  `interpolation_test` or another final partition.
- A Grid-1 model result is observed before the semantic repairs of A1 are
  committed. The declared definition may not then be changed.
- A carried-forward artifact turns out to have been produced under a superseded
  semantic definition and cannot be preserved byte-for-byte alongside its
  repaired successor.
- A neural training attempt is started, or the 9H attempt budget is consumed.
- The native prototype cannot reproduce the frozen Python E2c function to the
  declared tolerance. Timings are then not reported at all — an unvalidated
  native path is not a faster path.
- Any declared `OPEN` convention above is needed by a run and has not been
  resolved by a recorded decision. It is not invented to fill the gap.

## Completion report

On completion, whatever the outcome, record:

- the head/representation guidance (A) and the native feasibility finding (B),
  each with its qualifications and its `NOT ASSESSABLE` / non-converged
  determinations stated plainly;
- which gates A1–A4 and B1–B3 passed and which failed;
- confirmation that no final partition was opened, no neural attempt was spent,
  and no frozen snapshot was regenerated or reinterpreted;
- the update to [../../project-state.md](../../project-state.md) — current
  evidence, limitations, and the exact next task, which on success is the
  freeze's **Phase 1**: committing the v2 architecture and the
  acceptance/reference protocol;
- a new [../../decision-log.md](../../decision-log.md) entry, cross-linked to
  DEC-049.
