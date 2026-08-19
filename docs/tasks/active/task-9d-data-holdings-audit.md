# Task 9D: local data-holdings catalogue and integrity audit

## Status

`Completed` — **the catalogue exists and every exit gate is discharged.** This
is no longer the active task; the active task is
[task-9e-crr-dataset-admission.md](task-9e-crr-dataset-admission.md)
([../../decision-log.md](../../decision-log.md) DEC-032).

It became the active task when the stage-2 roadmap was locked (DEC-028) and
task 9C-C2b2 was deferred (DEC-029). Task 9C-C2b2's specification stays in
place at
[task-9c-c2b2-parallel-resumable-generation.md](task-9c-c2b2-parallel-resumable-generation.md),
marked `Deferred`, to be resumed only when the XSP/SPY phase needs
dataset-scale PDE generation. Task 9C-C3 is `Completed` and terminal
([task-9c-c3-label-policy-v2.md](task-9c-c3-label-policy-v2.md)). The deferred
half of this task — private object storage and entitlement-aware vendor
ingestion — is written up as
[task-9f-remote-data-access-plan.md](task-9f-remote-data-access-plan.md),
status `On hold`.

**Outcome.** The catalogue is
[../../data-holdings-catalogue.md](../../data-holdings-catalogue.md). Every holding
under `data/` is catalogued; every material fact is marked as a claim read from
metadata or a check recomputed from the bytes; nothing was admitted.

The headline results, in the catalogue's terms and not restated numerically
here beyond what a reader needs to know the shape of the answer:

- The candidate CRR dataset's every internal identity holds, its every manifest
  claim reconciles against the bytes, its partitions share no identifier and no
  exact contract state, and its oracle's composite source digest **reproduces
  from this branch's tracked C++ sources**.
- Its generator, configuration, tests, and the contract section describing its
  label policy are **not on this branch or on `main`** — they exist only on the
  unmerged branch `feat/american-dataset-v1` — and its label policy has **no
  frozen `docs/results/` evidence** on any branch.
- The market archive verifies **119 of 119** checksummed files with zero
  mismatches; three of the four processed trees are byte-identical to each
  other; and `exercise_style` and `contract_multiplier` are **100 % null** in
  the processed option definitions, which is a phase-2 constraint the XSP/SPY
  study must plan around rather than assume away.

Twelve unresolved limitations are recorded in the catalogue's section 8 and are
inherited by task 9E. **Nothing was admitted, nothing was uploaded, no vendor
API was called, no label was regenerated, the ML loader was not modified, and
no network was trained.**

**Review status.** The `numerical-reviewer` this spec requires for the CRR
dataset section was **not** invoked in the session that produced the catalogue,
and no fresh top-level session has yet materially accepted its findings. Per
`AGENTS.md`, a single-session conclusion is preliminary: `Completed` here means
the task's own exit gates are discharged and its deliverables exist, not that
its findings have been independently approved. Task 9E's admission decision is
where that approval becomes load-bearing, and 9E requires it explicitly.

## Objective

What, exactly, does this project already hold on local disk, and is any of it
admissible as project evidence? The task answers that by producing a written
catalogue and integrity audit of the existing local data holdings — the
candidate American CRR dataset and the Databento/FRED market-data holdings —
plus a deferred plan for private object storage and entitlement-aware vendor
ingestion.

A **positive** answer is a catalogue precise enough that a later task can
decide admission mechanically: every holding named, with schema, format,
partitions, size, row/record counts, provenance, digests where appropriate,
tracking/ignore status, and known limitations. A **negative** answer — a
holding whose provenance cannot be reconstructed, whose generator is not
recoverable, or whose contents do not match its own manifest — is an equally
valid completion, and it stops phase-1 training against that holding until a
separately gated decision resolves it.

**This task admits nothing.** Cataloguing a dataset is not accepting it.

## Motivation

Phase 1 of the locked roadmap (DEC-028) is a learnability experiment against a
continuous-dividend-yield American CRR dataset. **A local candidate CRR
dataset reportedly exists, but it has not yet been catalogued, validated,
reproducibly admitted, or accepted as project evidence.** It is local,
Git-ignored and untracked, and insufficiently catalogued (DEC-030).

Training on an uncatalogued holding would put the project's first American
result on evidence it cannot describe — the exact failure the frozen-evidence
and provenance rules in [../../architecture.md](../../architecture.md) exist to
prevent. Auditing first is cheap; discovering a provenance hole after a
training run is not.

The market-data holdings need the same treatment for a different reason: tasks
9A/9B established read-only what the proprietary archive can and cannot supply
(DEC-015), but the *inventory* of what is on disk, under which entitlement,
and in what shape, is not written down anywhere a later XSP/SPY task can read.

## Authoritative inputs

- [../../research-contract.md](../../research-contract.md) — "The locked
  stage-2 roadmap", "Data protocol", "Real-market inputs", "Non-claims".
- [../../architecture.md](../../architecture.md) — artifact identity and
  provenance rules any catalogued artefact is described against.
- [../../market-state-reconstruction-contract.md](../../market-state-reconstruction-contract.md)
  — what the market archive was already established to supply and not supply.
- [../../american-crr-contract.md](../../american-crr-contract.md) — the CRR
  engine the candidate dataset's labels claim to come from.
- [../../decision-log.md](../../decision-log.md) DEC-001 (continuous yield is
  not a discrete cash dividend), DEC-011 (no approved training-input status),
  DEC-014 (no dataset before its own gate), DEC-015 (the archive stays out of
  Git), DEC-028, DEC-029, DEC-030.
- The candidate dataset's own on-disk manifest and diagnostics, read **as
  claims to be checked**, never as established provenance.

## In scope

Produce a written catalogue and integrity audit covering:

- **The candidate American CRR dataset** under Git-ignored `data/`. Describe it
  precisely as a **continuous-dividend-yield American CRR** dataset — never as
  a no-dividend dataset. Record: schema and schema version; file format and
  compression; partition names and their separation; on-disk sizes; row counts
  per partition and in total; the label policy the manifest claims; the oracle
  and build provenance it claims; seed and sampling provenance; digests where
  appropriate; ignore/tracking status; and known limitations, including every
  caveat the manifest itself records.
- **Generator recoverability.** The generator, its configuration, and its tests
  are **not present on the current branch or on `main`**; they exist on the
  unmerged branch `feat/american-dataset-v1` (commit `49ef72a`, "Preserve
  synthetic American dataset prototype"), which adds
  `configs/american_option_dataset_v1.toml`,
  `configs/american_label_policy_pilot_v1.toml`,
  `configs/american_label_policy_pilot_v2.toml`, and the
  `differentiable_pricing.data.american_*` modules. The audit records this
  fact, whether the manifest's referenced config digest matches that branch's
  file, and what reproducing the dataset would therefore require. It does
  **not** merge, cherry-pick, or run any of it.
- **The Databento/FRED market-data holdings** under Git-ignored `data/` —
  vendors present, dataset/schema identifiers, date coverage, file formats,
  record counts, sizes, existing manifests and checksum files, ignore status,
  entitlement provenance where it is recorded, and known gaps.
- **A deferred plan** for private S3 storage/retrieval and entitlement-aware
  Databento ingestion: what would be stored, under what layout and retention,
  how integrity would be verified on retrieval, how entitlements would be
  represented, and what would have to be true before any of it is built.
  A plan only.

Where the catalogue lands: this task's own written output under `docs/`, in the
smallest form that serves it. It records **claims and verified checks
separately**, exactly as the frozen-evidence rules require elsewhere: a number
read out of a manifest is a claim; a number recomputed from the files is a
check.

## Out of scope

- **Uploading any data anywhere.** No object-storage write, no bucket, no
  sync.
- **Implementing S3 access** — no client, no credentials, no configuration.
  The S3 work product of this task is a plan.
- **Calling Databento** or any other vendor API, and any new ingestion code.
- **Regenerating CRR labels**, or regenerating, transforming, moving,
  reformatting, or deleting any existing holding.
- **Modifying the ML loader** or any training code, and **training any
  network**.
- **Admitting** any holding as project evidence or as a training input.
  Admission is a separate, separately gated decision; this task supplies the
  catalogue it would be made from, and `AUTHORIZED_TRAINING_INPUT_STATUSES`
  stays empty (DEC-011).
- Merging or running anything from `feat/american-dataset-v1`.
- Any change to the frozen PDE label-policy evidence, its config, or its
  runner (DEC-027). **The accepted PDE label policy remains frozen evidence;
  this task does not reinterpret or rerun it.**
- Extending provenance machinery, harvesting infrastructure, Greek-policy
  machinery, or cloud access.

## Predeclared conventions

- **Claim versus check.** Every catalogued fact is labelled as either a claim
  read from an artefact's own metadata or a check recomputed by this task.
  A claim is never promoted to a check by restating it.
- **Digest discipline.** Where a digest is recorded, it is SHA-256 over file
  bytes, and the file it covers is named alongside it.
- **Vocabulary.** The CRR dataset is a *continuous-dividend-yield American
  CRR* dataset. It is never called a no-dividend dataset, and a continuous
  yield is never described as equivalent to a discrete cash-distribution
  schedule (DEC-001).
- **No proprietary content leaves local disk.** The catalogue may record
  shapes, counts, coverage, digests and file names for the market archive; it
  never copies quote-level content into Git (DEC-015).
- **Admission threshold** — what would make a holding admissible as a training
  input — is `OPEN` and is **not** invented here. It is set by the later,
  separately gated admission decision. This task reports what an admission
  decision would need, not what it should conclude.

## Acceptance gates

- **Entry:** the roadmap is locked (DEC-028) and task 9C-C2b2 is deferred
  (DEC-029), so this is the single active task.
- **Coverage gate:** every holding under `data/` is either catalogued or
  explicitly listed as deliberately out of scope with a reason. No silent
  omission.
- **Separation gate:** every catalogued fact is marked claim or check; no
  unmarked fact appears.
- **Non-mutation gate:** no file under `data/`, `artifacts/`, `runs/`,
  `docs/results/`, or `docs/figures/` is created, modified, moved, or deleted
  by this task, and no code, test, config, script, or workflow changes.
- **Reproducibility gate:** for the CRR dataset, the audit states exactly what
  would be required to regenerate it and whether that is currently possible
  from tracked sources alone. "Not currently possible from tracked sources" is
  an acceptable — and expected — finding.
- **Exit:** the catalogue exists, the deferred S3/Databento plan exists,
  [../../project-state.md](../../project-state.md) states the new current state
  and the next task, and a decision-log entry records the audit's findings.
  Exit authorizes **no** admission, **no** upload, and **no** training.
- **Failure:** a holding whose provenance cannot be reconstructed is recorded
  as such, with what is missing. It is not repaired, regenerated, or given a
  provenance it does not have.

## Manual-run protocol

**None — this task runs no numerical study.** It performs read-only inspection
of local files: listing, sizing, hashing, and reading metadata. Those are cheap
and are not subject to the `AGENTS.md` manual-run rule, which governs long
numerical studies (CRR convergence, LSM cross-check, PDE label-policy pilots,
market ingestion/reconstruction). None of those may be launched by this task,
by a hook, by CI, or by an agent.

Reading a parquet file's metadata and row count is inspection. Re-deriving a
label, re-solving a PDE, or re-running a generator is not, and is out of scope.

## Artifact policy

This task's output is documentation under `docs/`, tracked in Git. It writes
nothing under `artifacts/`, `data/`, or `runs/`, and it freezes **no**
`docs/results/` snapshot: an inventory is not a numerical acceptance result,
and the frozen-evidence class has one generator/validator script per family by
design. If a later admission decision needs a machine-checkable artefact, that
belongs to that decision's task, with its own designated script and `--check`.

The catalogue must never embed proprietary quote-level content; it records
metadata about the archive, not the archive.

## Review requirements

`code-reviewer` is not required — this task changes no code. If it does change
code, that is a scope violation and a stop condition, not a review question.

`numerical-reviewer` is required for the CRR dataset section, because that
section makes claims about label provenance, sampling, partition separation,
and the label policy the dataset asserts. Both review kinds remain
separate-context **preliminary** review
([../../agent-system.md](../../agent-system.md)).

Accepting the audit's findings — and any later decision to admit a holding as a
training input — is a **material approval** requiring a fresh top-level session
with no anchoring on the implementer's reasoning.

## Stop conditions

- Any action would write, move, delete, or regenerate data: stop.
- Any action would require credentials, a network call to a vendor, or object
  storage: stop; that is the deferred plan, not this task.
- The audit is being used to argue for admission rather than to describe:
  stop. Admission is a separate gate.
- A holding's contents contradict its own manifest: stop cataloguing that
  holding as if the manifest were right, and record the contradiction as the
  finding.
- The candidate CRR dataset is about to be described as no-dividend, or a
  continuous yield as equivalent to a discrete dividend schedule: stop and fix
  the wording (DEC-001).

## Completion report

Record, unconditionally: the catalogue itself; which gates passed and failed;
every holding found, and every one deliberately excluded with its reason; the
claim/check split for each material fact; the reproducibility finding for the
CRR dataset, including the generator's branch situation; the deferred
S3/Databento plan; an update to
[../../project-state.md](../../project-state.md) stating the new current state
and the exact next task; and a new entry in
[../../decision-log.md](../../decision-log.md) recording the findings.

State explicitly that no data was uploaded, no S3 access was implemented, no
vendor API was called, no CRR labels were regenerated, no ML loader was
modified, no network was trained, and that no holding was admitted as project
evidence or as a training input by this task.
