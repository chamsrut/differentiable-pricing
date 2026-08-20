# Task 9E: CRR dataset admission — semantic suitability, loader support, integrity and leakage gates

## Status

`Blocked — conditionally admitted, predeclared gates incomplete`. The
implemented integrity, internal-identity, identifier-disjointness, and
exact-state-disjointness checks passed, but task 9E as predeclared did **not**
complete: the independent price cross-check and semantic-coverage judgement
were not performed, and the near-duplicate threshold was not fixed before its
distances were observed. The latter gate cannot be retroactively satisfied for
this dataset version.

The dataset is conditionally admitted only for learning the known
continuous-yield CRR mapping. That is not acceptance of converged
American-price accuracy, and it authorizes no training. A fresh top-level
review of the reconciled admission remains outstanding.

It became the active task when task 9D completed
([task-9d-data-holdings-audit.md](task-9d-data-holdings-audit.md), `Completed`).
Task 9C-C2b2 stays `Deferred`
([task-9c-c2b2-parallel-resumable-generation.md](task-9c-c2b2-parallel-resumable-generation.md)),
task 9C-C3 stays `Completed`
([task-9c-c3-label-policy-v2.md](task-9c-c3-label-policy-v2.md)), and task 9F
stays `On hold`
([task-9f-remote-data-access-plan.md](task-9f-remote-data-access-plan.md)).

**Reconciled outcome.** The admission record is
[../../american-crr-dataset-admission.md](../../american-crr-dataset-admission.md).
The dataset is **conditionally admitted solely for learning the known
continuous-yield American CRR mapping** under
`american-option-dataset/1`. The seven raw inputs are feature-sufficient but
not minimal; task 9G predeclares `american_forward_carry_v1`. No row was
dropped and no observed result or frozen gate was changed.

What was delivered: the generating configuration and the two label-policy pilot
configurations recovered verbatim from `49ef72a` and digest-pinned by a test;
the "Label policy v1" contract section recovered into
[../../american-crr-contract.md](../../american-crr-contract.md); the schema
contract ported as `data/american_schema.py` (**schema only — no generation
machinery, so the dataset still cannot be regenerated from tracked sources**);
strict named-schema loader support with the European path unchanged; and the
integrity and leakage gates in `data/american_admission.py`, with **75 focused
tests** on small fixtures (27 schema, 27 gate, 14 loader, 7
feature-sufficiency), none of which reads the Git-ignored dataset.

The independent LSM or PDE price cross-check remains outstanding and must be
completed or explicitly resolved before training is authorized. The missing
semantic-coverage judgement bounds the conclusion to feasibility for the known
mapping. The already-observed near-duplicate distances remain descriptive;
inventing a threshold now would not discharge the original gate.

The implementation also omits two mandatory row/manifest invariants: every
row's `label_policy` must equal `manifest.label_policy.name`, and every row's
`label_steps` must equal `manifest.label_policy.steps`. Task 9G must implement
and test these before training, and its entry checks must pin generator version
`1.0.0` and the recovered dataset configuration digest.

**This task authorizes no training.** The first American training run is a
separate task with its own gate, and it is not defined here.

## Objective

Is the local candidate CRR dataset fit to be the phase-1 learnability
experiment's input, and can this repository load it without guessing?

Task 9D catalogued the dataset and found its internal consistency exact and its
provenance partially open
([../../data-holdings-catalogue.md](../../data-holdings-catalogue.md)). Cataloguing
is not admission. This task decides admission: it assesses **semantic
suitability**, adds **named-schema loader support** so the dataset can be read
under an explicit schema contract rather than by coincidence, and establishes
**integrity and leakage gates** that a later training task can check
mechanically.

A **positive** answer is a dataset admitted under a named schema, with gates
that pass and are wired into the test suite. A **negative** answer — a semantic
defect, an unresolvable provenance gap, or a leakage property that fails — is
an equally valid completion that stops phase-1 training against this dataset
until separately resolved.

**This task trains nothing and generates nothing.**

## Motivation

Phase 1 of the locked roadmap (DEC-028) needs an input. The only candidate is
the local CRR dataset, and task 9D established both why it is promising and why
it cannot be used as-is:

- every internal identity holds exactly, every manifest claim reconciles
  against the bytes, and the pricing oracle's composite digest reproduces from
  this branch's C++ sources — so the labels come from a known engine;
- but the generator, its configuration, its tests and the contract section
  describing its label policy are **not on this branch**, its label policy has
  **no frozen evidence**, and the existing loader **rejects its schema by
  design**.

Admission is where those are resolved or recorded as permanent limits. Doing it
before training keeps a negative finding cheap.

## Authoritative inputs

- [../../data-holdings-catalogue.md](../../data-holdings-catalogue.md) — the task 9D
  audit; authoritative for what was measured, and for the 12 unresolved
  limitations this task inherits.
- [../../research-contract.md](../../research-contract.md) — "The locked stage-2
  roadmap", "Data protocol" (named partitions; no random split across rows
  sharing paths, grids, scenarios, or near-duplicate contract state),
  "Provisional gates", "Non-claims".
- [../../architecture.md](../../architecture.md) — artifact identity and provenance
  rules any admitted artefact must satisfy.
- [../../american-crr-contract.md](../../american-crr-contract.md) — the CRR engine
  on **this** branch. Note that its "Label policy v1" section exists only on
  `feat/american-dataset-v1`; reconciling that is in scope.
- [../../decision-log.md](../../decision-log.md) DEC-001, DEC-011, DEC-014,
  DEC-028, DEC-030, DEC-031, DEC-032.
- The dataset's own `manifest.json` and `diagnostics.json`, read as **claims to
  be checked**, exactly as in task 9D.

## In scope

### 1. Semantic suitability audit

Beyond internal consistency, which task 9D already established:

- whether the sampling design, the 11 strata and their weights, and the
  moneyness / maturity / volatility / rate / yield coverage make this a
  *useful* American-learnability probe rather than merely a valid table;
- the balance of informative versus degenerate rows — in particular the share
  carrying **no early-exercise signal at all** (`zero_premium_rows` 41,966 of
  200,000 in `train`; `no_early_exercise_rows` 25,648), and whether that share
  is a coverage defect, a faithful reflection of the domain, or something to
  weight around;
- whether the paired American/European-on-one-lattice row design supports the
  European→American transfer comparison (H2) or quietly distorts it;
- whether `interpolation_test` alone suffices for the phase-1 claim, given that
  no boundary, extrapolation/OOD, or scenario partition exists;
- an **independent price cross-check** on a small, predeclared sample of rows
  against a numerically unrelated engine already on this branch (LSM, or the
  PDE oracle at zero cash dividend), sized so it is a cheap consistency probe,
  **not** a study. Every consistency check in task 9D was internal to one
  lattice; this is the first external one.

### 2. Provenance resolution

- Decide and record how `feat/american-dataset-v1` is handled: brought forward,
  kept as a referenced read-only provenance branch, or declared unrecoverable
  for practical purposes. The branch is not an ancestor of `main` and is 94
  files divergent from HEAD; the decision is a real one and must be recorded,
  not defaulted.
- Decide whether the CRR label policy needs a frozen `docs/results/` snapshot
  and a contract section on this branch before admission, or whether its
  ignored `artifacts/` reports plus digest linkage are sufficient — and say
  which, explicitly.
- Record any provenance gap that remains open as a permanent limitation of
  anything trained on this dataset.

### 3. Named-schema loader support

- Extend the loading path so a dataset is read under an **explicitly named
  schema** rather than by exact equality against the single European table.
  Today `ml/dataset.py` compares with `TABLE_SCHEMA` via `schema.equals(...)`
  and `data/diagnose.load_manifest` rejects any `schema_version` other than
  `european-option-dataset/1`, so `american-option-dataset/1` fails closed —
  correct behaviour that this task replaces with a deliberate one.
- Registration is explicit: a schema is supported because it is named and
  versioned, never because a file happens to parse. Loading an unknown or
  unregistered schema stays a hard error.
- The European path's behaviour must not change. Its frozen results are
  terminal and any drift in how they load is a regression.
- Manifest-digest verification on load stays mandatory for every schema.

### 4. Integrity and leakage gates

Predeclared, mechanically checkable, and wired into the test suite so they run
without the dataset present or skip explicitly when it is absent:

- **Integrity:** manifest digests match the files; row counts match; the
  declared schema matches exactly; no nulls and no non-finite values; domain
  and per-stratum bound compliance.
- **Internal identity:** the label-averaging, premium, adjacent-gap and
  log-moneyness identities, and American dominance, at the tolerances task 9D
  measured (exact, except intrinsic dominance which is exact to rounding).
- **Leakage:** no `sample_id` and no exact contract state shared between
  partitions, plus a **near-duplicate gate whose threshold is predeclared and
  versioned before it is evaluated**. Task 9D's measured nearest-neighbour
  distances are descriptive; choosing a threshold to fit them afterwards is
  exactly the post-hoc criterion this repository forbids.

### 5. The admission decision itself

A recorded, explicit admission or refusal, with the limitations that survive
it. Admission means: this dataset may be used as the phase-1 learnability
input, under a named schema, subject to the recorded limits. It means nothing
more.

## Out of scope

- **Training any network**, fine-tuning, transfer experiments, hyperparameter
  search, or any learning curve. Phase 1's training run is a separate,
  separately gated task.
- **Regenerating, transforming, resampling, reformatting, moving, or deleting
  the dataset**, or any other holding. Read-only, as task 9D was.
- Generating a new dataset of any kind, American or European.
- Any PDE label generation, and any resumption of task 9C-C2b2.
- Any SPY or XSP work — phase 2 is not started by this task.
- Implementing anything in task 9F: no S3, no bucket, no credentials, no vendor
  API call, no upload.
- Any change to the frozen PDE label-policy evidence, its config, or its runner
  (DEC-027). **The accepted PDE label policy remains frozen evidence; this task
  does not reinterpret or rerun it.**
- Any change to `AUTHORIZED_TRAINING_INPUT_STATUSES` beyond what an explicit,
  recorded admission decision requires — and if this task's answer is refusal,
  no change at all.
- Revising the locked three-phase roadmap (DEC-028).

## Predeclared conventions

Fixed and versioned **before** the corresponding check is evaluated:

- **The named schema identifier and its version** for the American dataset, and
  the registration mechanism. `OPEN`.
- **The near-duplicate leakage threshold**, its metric, and its feature
  standardization — written down before it is computed again. `OPEN`.
- **The cross-check sample size, the rows selected, the comparison engine, and
  the agreement tolerance**, all fixed before the cross-check runs. `OPEN`.
- **What counts as a degenerate row** for the suitability assessment, and
  whether any threshold on their share is a gate or a diagnostic. `OPEN`.
- **The admission criteria themselves** — which findings block admission and
  which are recorded limits. `OPEN`, and they must be fixed before the
  suitability audit's results are read.

Do not invent a value for any `OPEN` item mid-task. Resolve and version it
first, and record the resolution in the decision log.

## Predeclared acceptance gates (preserved)

These are the original task 9E gates. They are preserved so the reconciled
outcome cannot silently weaken or rewrite them after results were observed.

- **Entry:** task 9D is complete and its catalogue exists; every `OPEN`
  convention above is resolved and versioned before the check it governs runs.
- **Loader gate:** the American dataset loads under its named schema with
  manifest-digest verification enforced; an unregistered schema and a
  digest mismatch both still fail closed; the European path is unchanged.
- **Integrity gate:** every integrity and internal-identity check above passes.
- **Leakage gate:** identifier and exact-state disjointness hold, and the
  predeclared near-duplicate threshold is met.
- **Cross-check gate:** the predeclared sample agrees with the independent
  engine within the predeclared tolerance.
- **Provenance gate:** the branch and label-policy-evidence decisions are
  recorded, and every remaining gap is written down as a limitation.
- **Exit:** the gates pass, the gates live in the test suite, the admission
  decision is recorded in [../../project-state.md](../../project-state.md) and
  [../../decision-log.md](../../decision-log.md), and the next task is named.
  Exit authorizes **the dataset as a phase-1 input and nothing else** — no
  training run is authorized by it.
- **Failure:** any gate that fails is recorded as a refusal or a conditional
  admission with its cause. It is not worked around by weakening the gate, by
  regenerating the dataset, or by dropping the offending rows.

### Reconciled gate outcome

- **Entry:** not fully discharged; the near-duplicate convention stayed open
  until after its distances were observed.
- **Loader:** discharged for named-schema loading and manifest/file digest
  verification, but the row/manifest label-policy name and step invariants are
  missing and must be added before training.
- **Integrity and internal identity:** discharged over all 250,000 rows for the
  implemented checks.
- **Leakage:** identifier and exact-state disjointness discharged;
  near-duplicate gate not discharged and impossible to satisfy retroactively
  for this dataset version.
- **Cross-check:** not discharged; no numerically independent American-price
  comparison was run.
- **Semantic suitability:** not discharged; no coverage judgement was made.
- **Exit:** not discharged as an unconditional admission. The honest outcome is
  conditional admission for learning the known CRR mapping, pending fresh
  review and the task 9G entry gates.

## Manual-run protocol

**No long numerical study is involved.** Loading, hashing, and column
reductions over a 47 MB dataset are cheap and belong in ordinary tooling and
the test suite.

The one exception is the independent cross-check in scope item 1: it is sized
to be a cheap probe, and if the predeclared sample ever grows to something that
is a *study*, it stops being part of this task and becomes its own manually
invoked job under the `AGENTS.md` rule. No hook, CI job, or agent may launch a
long numerical run.

## Artifact policy

Loader and gate code, the named-schema registration, and the tests are tracked
in Git. Any cross-check output goes to ignored `artifacts/`. This task freezes
**no** `docs/results/` snapshot unless the provenance decision concludes that
the CRR label policy needs one — in which case that snapshot gets its own
designated freeze/check script and a `--check` wired into `scripts/check.sh`
and CI, following the task 9C-C3 precedent.

The dataset itself is never committed and never moves.

## Review requirements

`code-reviewer` for the loader, schema registration, and gate code.
`numerical-reviewer` for the semantic suitability assessment, the cross-check,
the leakage gate, and every claim about label provenance or sampling. Both are
separate-context **preliminary** review
([../../agent-system.md](../../agent-system.md)).

**Admission is a material approval** and requires a fresh top-level session
with no anchoring on the implementer's reasoning. A single-session conclusion
that the dataset is admissible is not admission.

## Stop conditions

- Any action would modify, regenerate, move, or delete a holding: stop.
- A gate is about to be adjusted after its result is known: stop. That is the
  post-hoc criterion change this repository forbids.
- The suitability audit turns into training, or a "quick" fit is proposed to
  see whether the data works: stop. Training is a separate gate.
- A provenance gap is about to be papered over with an assumed value — a
  default multiplier, an assumed exercise style, an inferred config: stop and
  record the gap.
- The cross-check is about to grow into a study: stop and re-scope it as its
  own manual job.
- The dataset is about to be described as no-dividend, or its continuous yield
  as equivalent to a discrete dividend schedule: stop and fix the wording
  (DEC-001).

## Completion report

Record, unconditionally: the admission decision — admitted, conditionally
admitted, or refused — and its basis; which gates passed and failed; the
resolved value of every `OPEN` convention; the semantic suitability findings;
the cross-check result; the provenance decisions on
`feat/american-dataset-v1` and on the label policy's evidence; every limitation
that survives admission; an update to
[../../project-state.md](../../project-state.md) stating the new current state and
the exact next task; and a new entry in
[../../decision-log.md](../../decision-log.md).

State explicitly that no network was trained, no dataset was regenerated or
modified, and — if admission was refused — that nothing was admitted.
