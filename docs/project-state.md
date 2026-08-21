# Project state

The canonical, living handoff. If this file and a narrative document (README,
`docs/agentic-workflow.md`) disagree, this file wins — see
[documentation-map.md](documentation-map.md) for the full precedence order.
It does not win against code, versioned config, frozen evidence, or a
normative contract; it summarizes and links those rather than restating
their numbers.

## Update rule

Whoever starts, changes the scope of, or completes a milestone updates this
file — "Current implementation state," "Exact next task," and if needed
"Completed milestones" — in the same change, and adds a corresponding entry
to [decision-log.md](decision-log.md). A stale "next task" here is treated
as a defect in this file, not a fact about the project.

## Research thesis

Can a smooth neural surrogate reproduce a trusted derivative pricer's prices
**and** its useful risk sensitivities, across a declared domain, at
materially lower end-to-end latency, with an error profile suitable for a
stated decision? Full hypotheses, stage reference standards, and acceptance
methodology: [research-contract.md](research-contract.md). This file does
not restate them.

## Current research story

Stage 1 (European options) is complete and independently, fresh-seed
replicated — terminal, never rerun. Stage 2 (American options) now has three
numerically independent reference engines (CRR tree, LSM, discrete-dividend
PDE oracle), and — as of task 9C-C3 — **an accepted PDE label policy**. The
first attempt to turn the PDE oracle into a labeling policy (task 9C-B)
returned a negative, frozen result: `no_policy_selected`. Rather than respond
to that cost by adding parallel compute, the project spent tasks
9C-C1/9C-C2a/9C-C2b1 on algorithmic reuse — one solve returning a whole
valuation-time surface, grouped partitioning that keeps harvested rows honest
about correlation, and a three-surface vega design — before revisiting the
labeling-policy question itself. That revisit, task 9C-C3, ran its two
predeclared stages once each and selected `grid_1600x800`; a fresh top-level
independent session then returned **APPROVE POLICY AND FREEZE**. What is
unblocked is *labeling*, and only that: **no accepted production American
dataset and no American neural training exist yet**. The local CRR dataset's
later conditional, mapping-only admission does not change that production
claim. In parallel, a real-market track (tasks 9A/9B)
audited a three-session proprietary quote archive and established, read-only,
which pricing inputs it can and cannot supply — it produces no price, label, or
calibrated value and stays fully out of Git.

The stage-2 roadmap has since been **locked**
([decision-log.md](decision-log.md) DEC-028) and the story reordered around the
question the project actually exists to answer: can a neural surrogate price
American options with useful accuracy while delivering materially faster
inference than the numerical method that generated its labels, and does that
speedup support faster implied-volatility inversion and volatility-surface
construction? The cheapest evidence for that question needs no new label
generation, so the first experiment runs against an existing
continuous-dividend-yield American CRR baseline rather than against new PDE
labels. Accordingly task 9C-C2b2 is **deferred, not rejected** (DEC-029), and
the next step was a documentation and integrity audit of what this project
already holds on local disk (DEC-030). That audit, task 9D, is now complete: the
catalogue is [data-holdings-catalogue.md](data-holdings-catalogue.md), it admits
nothing, and it leaves the candidate CRR dataset catalogued-but-not-admitted
with twelve recorded limitations (DEC-031). Task 9E added named-schema support
and internal gates, but reconciliation found that its independent cross-check,
near-duplicate gate, and semantic-coverage judgement were not discharged. The
dataset is now conditionally admitted only for learning the known CRR mapping,
not for converged American-price accuracy (DEC-034). Task 9G's bounded protocol
and implementation were built (DEC-036), approved at cumulative commit
`8d27c23` (DEC-037), and **merged** at `e930454`. The human operator then ran
the locked `run-to-validation` once, and **it returned a negative result**:
`status=validation_gates_failed`, `outcome=failure_to_learn`. Every entry gate
passed — the independent PDE mapping check passed 21 of 21 rows before
optimization, and the exact European-to-American transfer lift passed all eight
pinned probes — but **neither arm passed every validation gate**. Transfer's
validation RMSE was strictly better than scratch's, which is a within-pilot
observation confounded by the arm-seeded shuffle and is not evidence of transfer
value. **No final evaluation occurred**: `final_evaluation_attempts = 0`,
`final_partition_consumed = false`, and final evaluation is forbidden under this
protocol because its final-entry rule failed. The frozen evidence is
[results/american_neural_pilot_results_v1.json](results/american_neural_pilot_results_v1.json)
(DEC-038). **One seed and one budget do not establish H2**, and this result is
not H2 evidence.
**The accepted PDE label policy remains frozen evidence; this roadmap change
does not reinterpret or rerun it.**

## Completed milestones

| Milestone | Evidence |
|---|---|
| European neural-pricer replication | [results/european_replication_results_v1.json](results/european_replication_results_v1.json), frozen-terminal |
| American CRR reference (scalar + parallel batch) | [american-crr-contract.md](american-crr-contract.md) |
| American LSM cross-check | [results/american_lsm_crosscheck_results_v1.json](results/american_lsm_crosscheck_results_v1.json) |
| Task 9A: market ingestion / feasibility audit | [architecture.md](architecture.md), local-only, no artefact staged |
| Task 9B: market-state reconstruction | [market-state-reconstruction-contract.md](market-state-reconstruction-contract.md), local-only |
| Task 9C-A: discrete-dividend PDE oracle | [pde-numerical-contract.md](pde-numerical-contract.md) |
| Task 9C-B: PDE label-policy pilot v1 | [results/american_pde_label_policy_results_v1.json](results/american_pde_label_policy_results_v1.json) — frozen `no_policy_selected` |
| Task 9C-C1: valuation-time surface | [pde-numerical-contract.md](pde-numerical-contract.md) (Task 9C-C1 section) |
| Task 9C-C2a: leakage-safe grouped harvesting | [pde-numerical-contract.md](pde-numerical-contract.md) (Task 9C-C2a section) |
| Task 9C-C2b1: authoritative verification and three-surface vega | [pde-numerical-contract.md](pde-numerical-contract.md) (Task 9C-C2b1 section) |
| Task 9C-C3 predeclaration (criteria, lifecycle, runner, freeze tool) | [pde-numerical-contract.md](pde-numerical-contract.md) (Task 9C-C3 section) |
| Task 9C-C3: PDE label-policy v2 — **accepted `grid_1600x800`** | [results/american_pde_label_policy_v2_results_v1.json](results/american_pde_label_policy_v2_results_v1.json) — frozen terminal, externally approved ([decision-log.md](decision-log.md) DEC-025) |
| Task 9D: local data-holdings catalogue and integrity audit | [data-holdings-catalogue.md](data-holdings-catalogue.md) — audit record, admits nothing ([decision-log.md](decision-log.md) DEC-031) |
| Task 9G: American neural-pricer feasibility pilot — **negative `failure_to_learn`** | [results/american_neural_pilot_results_v1.json](results/american_neural_pilot_results_v1.json) — frozen, `validation_gates_failed`, final partition unconsumed ([decision-log.md](decision-log.md) DEC-038); **fresh top-level review returned `APPROVE TASK 9G RESULT FOR MERGE`** (DEC-039), pinned by `python/tests/test_american_neural_pilot_results_snapshot.py` |

## Current implementation state

- C++ reference engines: Black–Scholes, CRR (scalar + parallel batch), LSM,
  discrete-dividend PDE (scalar price, and the valuation-time surface API).
- Python: dataset generation/diagnostics and training for the European
  surrogate (complete); study runners for CRR convergence, LSM cross-check,
  the 9C-B pilot, and PDE surface/vega harvesting (all exploratory, none
  feeding a production dataset); read-only market ingestion/reconstruction.
- **An accepted production label policy exists: `grid_1600x800`** (task
  9C-C3, [decision-log.md](decision-log.md) DEC-025). It covers price on all
  22 regular cases, delta on 18 of them and vega on all 22; gamma stays
  evaluation-only. **No accepted, versioned PDE-labelled American
  neural-training dataset exists**, and an accepted label policy is not one.
  Three distinct things must not be conflated here:
  - **tracked, versioned, reproducible demo code and configuration** —
    `scripts/demo_pde_surface_harvest.py`,
    `scripts/demo_pde_surface_vega_harvest.py`, and their configs
    (`configs/pde_surface_harvest_demo_v1.toml`,
    `configs/pde_surface_vega_harvest_demo_v1.toml`) are checked into Git
    and reproducible by construction — running them is deterministic;
  - **ignored, unfrozen local outputs those demos produce** — running that
    tracked code writes rows/reports beneath ignored `artifacts/`; those
    outputs are never committed, are refused by the training-input gate
    ([decision-log.md](decision-log.md) DEC-011), and are not themselves
    tracked, reproducible evidence in the sense a frozen `docs/results/`
    snapshot is — they demonstrate the harvesting machinery on one run, not
    a dataset;
  - **an accepted, versioned training dataset** — does not exist in either
    form above. Neither the tracked demo code nor any local run of it
    constitutes one. It requires an accepted label policy — which task 9C-C3
    has now supplied — **and** its own separately gated generation task, which
    has not run.
  No American neural surrogate exists. No parallel or resumable label
  generation exists.
- **The local candidate CRR dataset is catalogued and conditionally admitted
  only for learning the known continuous-yield CRR mapping.** The admission
  record is
  [american-crr-dataset-admission.md](american-crr-dataset-admission.md). Fresh
  independent review returned `APPROVE PLANNING RECONCILIATION` (DEC-035),
  discharging material review only for this conditional, mapping-only
  admission. This is not acceptance of converged American-price accuracy and
  authorizes no training. Task
  9E recovered the generating configuration and the two label-policy pilot
  configurations verbatim from `49ef72a` (digest-pinned by a test), recovered the
  "Label policy v1" section into
  [american-crr-contract.md](american-crr-contract.md), ported the schema
  contract as `data/american_schema.py` — **schema only, so the dataset still
  cannot be regenerated from tracked sources** — added strict named-schema loader
  support with the European path unchanged, and added integrity,
  internal-identity, identifier-disjointness, and exact-state-disjointness
  checks in `data/american_admission.py`; those implemented checks passed over
  all 250,000 rows. The independent numerical cross-check and semantic-coverage
  judgement were not performed. No near-duplicate threshold was predeclared
  before the distances were observed, so that gate cannot be satisfied
  retroactively. Task 9G now enforces each row's `label_policy` and
  `label_steps` against the manifest policy and pins generator version `1.0.0`
  and recovered config digest `d18485c6…`. The raw dataset is feature-sufficient, but
  `american_raw_physical_v1` is not minimal. The pilot uses
  `american_forward_carry_v1`: encoded type, `log(F/K)`, `sigma*sqrt(T)`, `rT`,
  and `qT`, with target `V/(S*exp(-q*T))`. The three-input European
  `forward_normalized_v1` representation remains rejected for American labels,
  because two contracts sharing a forward and a total volatility differ by
  1.2–1.3 % in normalized American price while agreeing to machine precision
  in European price.
- The dataset itself sits under Git-ignored `data/american-option-v1/`: 250,000 rows
  of synthetic **continuous-dividend-yield American CRR** labels — not a
  no-dividend dataset, and a continuous yield is not a discrete
  cash-distribution schedule (DEC-001). Task 9D verified every internal
  identity and every manifest claim against the bytes, and reproduced the
  labelling oracle's composite source digest from this branch's C++ sources;
  it also established that the generator, its configuration, its tests and the
  contract section describing its label policy were **not on this branch and not
  on `main`** (they exist on the unmerged branch `feat/american-dataset-v1`), and
  that the label policy has **no frozen `docs/results/` evidence** on any branch.
  Full detail:
  [data-holdings-catalogue.md](data-holdings-catalogue.md) (DEC-031). Task 9E
  resolved the loader gap and the configuration/contract recovery; the generator
  gap and the label policy's missing frozen evidence survive as recorded
  limitations (DEC-033).
- Task 9G supplies the strict five-input representation, exact task-specific
  European lift, American artifact schema, protocol validator, validation/final
  lifecycle runner, metrics/diagnostics, matched latency and synthetic-IV
  implementations, and raw-report freeze/check tool. It is **merged at
  `e930454`, executed once, and closed with a negative result**. The recovered
  source lift was preflighted and re-verified in the run at eight physical
  probes after train-only scaling: maximum absolute price difference
  `1.4210854715202004e-14`, maximum relative difference
  `3.6214823824845716e-13`. The validation audit snapshot uses fixed-size
  sufficient statistics rather than partition rows; the separately retained
  21-case PDE evidence is synthetic and protocol-pinned, and all 21 rows passed.
  The protocol records the arm-seeded shuffle confound, the fixed-domain PDE
  truncation limitation, and the limits of offline snapshot authentication, and
  all three survive into the frozen result. **An American neural surrogate was
  trained but not accepted**: neither arm met the predeclared feasibility gates,
  the latency-speedup and IV-error gates also failed, and no final evaluation
  ran. The offline snapshot check
  `python3 scripts/freeze_american_neural_pilot_results.py --check` now runs in
  both `scripts/check.sh` and CI; it reads only tracked files and reruns no
  pricing, training, or IV inversion.
- **No accepted, versioned PDE-labelled SPY training dataset exists**, and no
  SPY neural surrogate exists. **No American neural surrogate has yet been
  trained and accepted** — task 9G trained two arms and accepted neither.
- **Task 9H supplies price-only development infrastructure, and six attempts
  have been run and recorded.** The package
  `python/src/differentiable_pricing/ml/american_dev/`
  holds five modules: `attempts` (partition guard and its single forbidden-token
  list, row/seed selection, digests, clean-tree and committed-source checks,
  strict configuration validation, attempt-log rules; PyTorch-free),
  `representation` (the five `american_forward_carry_v1` coordinates, the
  European anchor, conditioning features, heads, physical reconstruction),
  `models` (a dense and a residual network plus dispatch), `workbench` (one
  attempt end to end) and `geometry` (the exploratory validation-set geometry of
  the binding constraints, including the CRR-versus-analytic European comparator
  discrepancy; it reads one partition, trains nothing and writes no attempt
  evidence). Three heads are implemented and dispatched: `direct`,
  `premium_over_european` and `smooth_lower_floor`, the last a smooth one-sided
  projection onto `max(analytic European, intrinsic)` at a predeclared
  normalized temperature of `1e-4` that preserves the direct price target
  (DEC-044).
  Evaluation reuses Task 9G's `sliced_metrics`, `shape_diagnostics` and
  `assess_arm` against `configs/american_neural_pilot_acceptance_v1.toml`, so
  the criterion is shared code rather than a copied number, and an attempt may
  not point at another acceptance file or section. Before an attempt reserves
  anything, the workbench requires its configuration and every digest-recorded
  source file to be tracked at `HEAD` and unmodified, and pins the manifest,
  `train` and `validation` identities to the ones
  `configs/american_neural_pilot_protocol_v1.toml` locked for Task 9G; it then
  reuses Task 9G's row-level `verify_partition_policy` on both partitions
  (DEC-042). The scripts are the human-only
  `scripts/run_american_dev_attempt.py` (`run`, `status`), the human-only
  `scripts/analyze_american_dev_geometry.py` (`analyze`, `show`) and the
  offline `scripts/american_dev_attempts.py` (`record`, `check`; the `check`
  mode is wired into `scripts/check.sh` and CI). Configurations are the six
  immutable `configs/american_dev_attempt_scratch_*.toml`. The append-only
  attempt log is `docs/attempts/task-9h-attempt-log.jsonl`; it holds **six**
  attempts, all `outcome=criterion_not_met`: `scratch_direct_control_v1`
  (normalized RMSE 5.73e-3, 10,121 bound / 1,460 shape violations),
  `scratch_capacity_v1` (3.50e-3, 9,083 / 854), `scratch_american_premium_v1`
  (6.98e-3, 1,116 / 330 — zero European-comparator violations),
  `scratch_conditioning_v1` (4.22e-3, 9,205 / 1,537) and
  `scratch_residual_architecture_v1` (2.11e-3, 8,569 / 683 — all three
  price-error gates passed) and `scratch_residual_premium_v1` (6.85e-3, 6,925 /
  312 — the premium head at residual capacity, which reproduced the small
  premium arm's price cost and left 5,839 stored-CRR comparator violations
  despite an enforced analytic European anchor), recorded at `193ab11`,
  `3c4fdb8`, `7099c57`, `15bba65`, `32258bf` and `207536c`. Those are
  development measurements selected against `validation`, not project results.
  The schema-1 validation-geometry report has been produced and lives beneath
  the ignored `artifacts/` tree (DEC-043). **The schema-2 comparator-discrepancy
  extension and the E2 candidate `scratch_residual_smooth_floor_v1` are
  predeclared and not yet run** (DEC-044); a predeclared configuration is a
  plan, not a measurement. No Greek, latency, implied-volatility or transfer
  machinery exists in task 9H.

## Exact next task

**Task 9H: adaptive American price-model development**, on branch
`experiment/task-9h-american-pricer-development` —
[tasks/active/task-9h-american-pricer-development.md](tasks/active/task-9h-american-pricer-development.md)
([decision-log.md](decision-log.md) DEC-041). It iterates on capacity,
representation, target and architecture until an American **price** model meets
a fixed development criterion on `train` and `validation`.

**It is development, not confirmatory research.** `validation` is used
repeatedly for selection, so **nothing task 9H produces is a project result**:
anything it selects carries selection bias and requires a separately predeclared
confirmation on a fresh final partition, as its own task with its own gates and
its own review. `interpolation_test` and every other final partition stay
inaccessible for the whole loop, and the task 9H runner exposes **no
final-evaluation command**.

**The criterion is Task 9G's, reused rather than restated**: normalized
RMSE `<= 0.003`, p99 `<= 0.015`, maximum `<= 0.08`, and zero material bound or
shape violations. Task 9H reads it from
`configs/american_neural_pilot_acceptance_v1.toml` and applies it through
`ml.american_pilot`, so it cannot be loosened after an attempt fails without
breaking the digest-pinned Task 9G protocol check.

**Scope is price only.** Greeks, latency, implied volatility and transfer
learning are separate follow-up stages that begin only after a candidate works;
none of their machinery is built in advance.

Its current state is **infrastructure implemented and hardened, one attempt
recorded**: five immutable attempt configurations, a human-only runner, an
offline attempt-log recorder/checker, and an append-only attempt log holding the
failed `scratch_direct_control_v1` control. The human invokes every training
run; agents implement code and analyze compact summaries and launch nothing.

The next attempt is `scratch_capacity_v1`, whose configuration already exists and
is unchanged. Nothing has been run for it, and the control is **not** rerun.

**Task 9G is closed and terminal.** Fresh top-level review of the result-only
change at `a4fd9f2` (merged as PR #27, `main` at `5c0ef6a`) returned **APPROVE
TASK 9G RESULT FOR MERGE** ([decision-log.md](decision-log.md) DEC-039). That
verdict accepts the recorded negative outcome and the snapshot as authoritative
for its numbers; it approves no H2 result, no converged American-price truth, no
American Greek accuracy, and no latency conclusion, and it does not authorize
`final-evaluate`, which stays forbidden under the task 9G protocol. A
non-material process deviation during result preparation — a read-only SHA-256
of the ignored validation report, taken contrary to a session instruction — is
recorded for honesty as DEC-040; it changed no evidence, ran no numerical work
and accessed no final partition. The task 9G snapshot is now additionally pinned
by `python/tests/test_american_neural_pilot_results_snapshot.py`, which pins its
SHA-256 `8a125c81…` and invokes the designated offline checker in-process.

### The task 9G result-only change, as reviewed and merged

The result-only change on branch `results/task-9g-american-neural-pilot-v1`
added the frozen snapshot
[results/american_neural_pilot_results_v1.json](results/american_neural_pilot_results_v1.json)
(distilled from validation report SHA-256 `9a4acdfd…`), wired
`python3 scripts/freeze_american_neural_pilot_results.py --check` into
`scripts/check.sh` and CI, and updated the task spec, this file, and the
decision log. It changed no config, threshold, seed, protocol value, training
code, evaluation code, runner behavior, freezer semantics, or frozen evidence.
It is merged at `a4fd9f2` (PR #27, `main` at `5c0ef6a`) and approved by DEC-039.

**The recorded result is `status=validation_gates_failed`,
`outcome=failure_to_learn`.** Every entry gate passed, including the
independent PDE mapping check at 21 of 21 rows and the exact transfer lift at
all eight pinned probes; neither arm then passed every validation gate.
Transfer's validation RMSE was strictly better than scratch's — a within-pilot
observation confounded by the arm-seeded epoch shuffle, not evidence of transfer
value, and not a converted pass. **No final evaluation occurred**:
`final_evaluation_attempts = 0`, `final_partition_consumed = false`, and
`final-evaluate` is **forbidden** under this protocol because
`validation_final_entry_passed = false` and `second_attempt_allowed = false`.
`interpolation_test` was never opened, hashed, imported, or counted, and it is
never tuned against. **One seed and one budget do not establish H2.**

What 9E delivered: the generating configuration and both label-policy pilot
configurations recovered verbatim from `49ef72a` and digest-pinned by a test;
the "Label policy v1" section recovered into
[american-crr-contract.md](american-crr-contract.md); the schema contract
ported as `data/american_schema.py` — **schema only, no generation machinery**;
strict named-schema loader support with the European path unchanged and unknown
schemas failing closed; the implemented integrity and exact-disjointness checks in
`data/american_admission.py`; and 75 focused tests on small fixtures, none of
which reads the Git-ignored dataset.

What 9E did not complete: an **independent LSM or PDE cross-check** of these
labels and a **semantic-coverage judgement** of the sampling design. The
near-duplicate threshold was not predeclared before task 9D measured the
distances, so that gate cannot be satisfied retroactively for this dataset
version and the measurements stay descriptive. The current checks also omit
the row/manifest label-policy name and step equalities. These facts make the
admission conditional and mapping-only; they are not all-gates completion.

**Task 9D is `Completed`.** Its catalogue is
[data-holdings-catalogue.md](data-holdings-catalogue.md) — every holding under
`data/` catalogued, every fact marked claim or check, and nothing admitted
(DEC-031). Of the twelve unresolved limitations recorded there, task 9E closed
the loader-rejection and configuration/contract-recovery items; the rest survive
in [american-crr-dataset-admission.md](american-crr-dataset-admission.md),
"Limitations that survive admission".

**Task 9F is `On hold`** —
[tasks/active/task-9f-remote-data-access-plan.md](tasks/active/task-9f-remote-data-access-plan.md),
private object storage and entitlement-aware Databento ingestion. It is a plan,
not work: every one of its predeclared conventions is `OPEN`, and none was
invented. Licensed OPRA/Databento content may not be redistributed publicly;
"reproducible" there means reproducible by an authorized user who independently
holds the required vendor entitlements. **Cloud storage and vendor ingestion are
deferred reproducibility work, not prerequisites for the first CRR learnability
experiment.**

**Task 9C-C2b2 is `Deferred`**, not deleted and not rejected (DEC-029). Its
specification stays in place at
[tasks/active/task-9c-c2b2-parallel-resumable-generation.md](tasks/active/task-9c-c2b2-parallel-resumable-generation.md)
and is the specification to resume from **when the XSP/SPY phase requires
dataset-scale PDE generation**, and not before.

Task 9C-C3 is **complete and terminal**. Its two stages ran once each,
manually; the confirmation stage selected `grid_1600x800`, and a fresh
top-level independent session returned **APPROVE POLICY AND FREEZE**. The
frozen evidence is
[results/american_pde_label_policy_v2_results_v1.json](results/american_pde_label_policy_v2_results_v1.json)
(SHA-256 `75d9402f071323065f8398ccd2cf427e2fb6fa1e663aef90ec7cbcf9c46a1186`,
distilled from confirmation report
`f9bf3f8fd636498b09fae20df8e42e976c68d2b70b28fdff20ab93752a7e130e`). Its
numbers live in [pde-numerical-contract.md](pde-numerical-contract.md) ("Task
9C-C3: label-policy v2", "Outcome: the accepted v2 result") and are not
restated here. Nothing about 9C-C3 is rerunnable or revisable; its ten
remediation cases and its 28 confirmation cases are consumed evidence. **The
accepted PDE label policy remains frozen evidence; the roadmap lock does not
reinterpret or rerun it.**

## Locked roadmap

The stage-2 order is locked by [decision-log.md](decision-log.md) DEC-028 and
stated normatively in [research-contract.md](research-contract.md), "The
locked stage-2 roadmap". It exists to answer one question: can a neural
surrogate price American options with useful accuracy while delivering
materially faster inference than the numerical method that generated its
labels, and does that speedup support faster implied-volatility inversion and
volatility-surface construction?

1. **Continuous-dividend-yield American CRR baseline.** Use the existing local
   candidate CRR dataset — after task 9D has catalogued it and a separate gate
   has admitted it — to test American-price learnability, scratch versus
   European-transfer training, inference accuracy, inference scaling against
   CRR's $O(N^2)$ lattice cost, and a small implied-volatility/surface
   reconstruction. It is a continuous-dividend-yield dataset; it **cannot**
   model SPY cash dividends (DEC-001).
2. **XSP/SPY real-instrument study.** XSP (European, cash-settled) is the
   control; SPY (American, discrete deterministic cash distributions, early
   exercise) is the target. Learn PDE prices for contracts grounded in the
   available market universe, compare inference against the PDE, and
   reproduce and evaluate real implied-volatility surfaces. The existing PDE
   solver and the accepted v2 label policy remain valuable inputs here. This
   is the phase whose dataset-scale generation would resume task 9C-C2b2.
3. **Deferred commodity extension.** Corn options are the leading future
   candidate — American exercise into futures, seasonality, and the futures
   curve give a genuinely different cross-asset test. Not on the critical
   path: not designed, not scoped for data acquisition, not implemented now.

**Crypto is explicitly excluded** from the active roadmap: European-only
crypto options do not advance the American-option research question.

**Cloud storage and vendor ingestion are deferred reproducibility work, not
prerequisites for the first CRR learnability experiment.**

## Entry/exit gates

Project-level gates (fixed acceptance criteria, `validation`/
`interpolation_test` separation, consumed-partition discipline) are in
[research-contract.md](research-contract.md), "Provisional gates" and
"Data protocol." They are not restated here.

Task 9C-C3's gates are **all discharged** and are recorded here as history,
not as pending work: the criteria and ten-case remediation set were
predeclared and versioned; remediation passed in full under unadjusted
criteria, which admitted the 28-case confirmation run; confirmation ran once
and selected `grid_1600x800`; the terminal snapshot was frozen; and the
**fresh top-level independent approval** that the exit gate required was
obtained (**APPROVE POLICY AND FREEZE**, [decision-log.md](decision-log.md)
DEC-025). Full detail:
[tasks/active/task-9c-c3-label-policy-v2.md](tasks/active/task-9c-c3-label-policy-v2.md),
now marked `Completed`.

Task 9D's gates are **all discharged** and are recorded here as history, not
as pending work: every holding under `data/` is catalogued; every material fact
is marked claim or check; the CRR dataset's reproducibility situation is stated,
including that its generator is on neither this branch nor `main`; the deferred
S3/Databento plan exists as task 9F; and this file and the decision log are
updated. Nothing was admitted, uploaded, regenerated, or trained. Full detail:
[data-holdings-catalogue.md](data-holdings-catalogue.md) and
[tasks/active/task-9d-data-holdings-audit.md](tasks/active/task-9d-data-holdings-audit.md),
now marked `Completed`.

Task 9E's gates, and their exact status:

- **Entry:** **not fully discharged** — task 9D was complete, but the
  near-duplicate convention stayed open until after its distances were
  observed.
- **Loader gate:** discharged — the dataset loads under a named, versioned
  schema with manifest-digest verification enforced; unknown schemas and
  unregistered `schema_version` values fail closed with no permissive fallback;
  the European loading path is unchanged and its tests pass untouched. Task 9G
  now implements and tests the row `label_policy` and `label_steps` equalities.
- **Integrity and internal-identity gates:** discharged — every gate in
  `data/american_admission.py` ran over all 250,000 rows and passed, and the
  gates live in the test suite on small fixtures.
- **Leakage gate:** **partially discharged** — identifier and exact
  contract-state disjointness hold. The near-duplicate half was not run as a
  gate: no threshold was predeclared before task 9D observed the distances, so
  it cannot be satisfied retroactively for this dataset version.
- **Cross-check gate:** **not discharged.** An independent LSM or PDE
  cross-check of these labels remains open; every identity verified so far is
  internal to one lattice.
- **Semantic-coverage judgement:** **not discharged.** The recorded zero-
  premium and no-exercise counts were not judged for suitability.
- **Provenance gate:** discharged — the branch and label-policy-evidence
  decisions are recorded, with the surviving gaps written down.
- **Exit:** **not discharged as an unconditional admission.** The reconciled
  outcome is conditional admission only for learning the known CRR mapping. It
  received fresh independent approval as `APPROVE PLANNING RECONCILIATION`
  (DEC-035), and it is not acceptance of converged American-price accuracy.
- **Explicitly not authorized:** training any network, and regenerating,
  transforming, moving or deleting any dataset. None occurred.

Task 9H's gates, and their exact status:

- **Entry:** discharged — task 9G is terminal and its result is approved
  (DEC-039), and task 9H has its own predeclared specification (DEC-041).
- **Partition gate:** enforced in code — only `train` and `validation` are
  reachable; any other split name or resolved path fails closed; the dataset
  manifest is stripped to those two splits before any consumer sees it; and
  there is no final-evaluation entry point at all. Re-checked offline by
  `python3 scripts/american_dev_attempts.py check`.
- **Clean-tree and identity gate:** enforced in code — a real attempt refuses
  to run with tracked worktree modifications and records the exact commit,
  configuration digest, source digests, dataset digests and selected row
  identities.
- **Immutability gate:** enforced — an attempt is never overwritten, the
  attempt log refuses a duplicate ID, and the offline check re-verifies that
  every logged attempt's configuration still hashes to the digest that attempt
  recorded.
- **Criterion gate:** fixed before the first attempt and **not revisable
  because an attempt fails.** It is Task 9G's `[validation_final_entry]`
  section, read from a digest-pinned file and applied through Task 9G's own
  evaluation functions.
- **Exit:** **not discharged.** Nothing task 9H selects is a result; a candidate
  that meets the criterion requires a separately predeclared confirmation on a
  fresh final partition, as its own task. Greeks, latency, implied volatility
  and transfer learning are separate follow-up stages after that.
- **Explicitly not authorized:** any final-partition access; any agent-, hook-
  or CI-launched training run; and editing any task 9G protocol, config,
  runner, freezer or snapshot.

Task 9F's gates are **dormant** — it is `On hold`, its every predeclared
convention is `OPEN`, and its entry requires a recorded decision to start.
Its redistribution gate binds whenever it does start: no artefact it produces
may let an unentitled party obtain licensed content.

Task 9C-C2b2's gates are **unchanged and dormant** while it is deferred. Its
entry still requires an accepted label policy (satisfied by DEC-025) plus its
own predeclared generation domain, which it does **not** inherit from C3's 28
evidence cases; its exit still authorizes no production dataset and no neural
training (DEC-014, DEC-025).

## Known limitations and non-claims

- Harvested rows from one surface are **correlated**, not independent
  samples — they are outputs of one solve.
- Partitioning is by **independent solve group**, assigned before any
  solve; a group can never straddle `train`/`validation`/
  `interpolation_test`.
- Row count is **not** an effective sample size, whether counted per group
  or per attempted surface — see
  [decision-log.md](decision-log.md) DEC-006.
- Gamma remains **evaluation-only**; no current row treats it as a
  supervised target (DEC-009).
- Vega is supervision-eligible **only where task 9C-C3 measured it to be**:
  22 of 22 regular evidence cases (DEC-007, DEC-025).
  `vega_numerically_available` on a harvested row is still not the same claim,
  and eligibility
  was established on 28 isolated evidence cases, which **do not validate the
  surrounding parameter hyperrectangle**.
- Delta is supervision-eligible on **18 of 22** regular cases. That is
  conservative **observed-order measurability**, not four inaccurate deltas:
  all four exclusions passed their stencil-delta validation and were excluded
  only because the factor-two order was undefined or outside the predeclared
  band (DEC-026). The band is not widened; reconsidering it needs a separately
  versioned task.
- Task 9C-C3's American dominance and intrinsic allowances are **operational
  price-error scale estimates, not certified bounds** — every v2 report
  publishes `is_a_rigorous_bound = false` (DEC-016).
- The accepted policy `grid_1600x800` covers **price** on 22/22 regular
  cases, with delta and vega eligibility decided separately, case by case
  (DEC-017, DEC-025). Gamma is not covered at all.
- The accepted policy authorizes **labeling only**. The frozen v2 report
  records `authorizes_dataset_generation = false` and
  `authorizes_training_input = false`; no dataset and no training input is
  authorized by it (DEC-014, DEC-025).
- One v2 stress case, `stress_euro_short_low_vol_atm`, **fails** its price
  check descriptively (2.9943e-3) with a correspondingly large
  evaluation-only gamma error. Stress cases decide no selection, and this is
  retained as honest evidence of a regime where the accepted policy is not
  accurate — not as a resolved defect.
- `selection_pending_fresh_top_level_approval = true` in the frozen v2
  snapshot is **historical and correct** — the report predates the approval.
  DEC-025 records the approval; the snapshot is never edited to flip it
  (DEC-027).
- Task 9C-B's `no_policy_selected` is a **valid, complete negative result**,
  not an unfinished task — it is frozen and never reinterpreted (DEC-003).
- The local candidate CRR dataset is **conditionally admitted only for learning
  the known continuous-yield CRR mapping** (DEC-033, DEC-034, DEC-035). Its
  fresh material review is discharged only for that bounded reconciliation.
  Admission is not training authorization or acceptance of converged
  American-price accuracy, and nothing in the dataset may be cited as a project
  result on the strength of admission alone.
- The CRR dataset is **still not reproducible from tracked sources alone**.
  Task 9E recovered its configuration, its two pilot configurations and its
  label-policy contract section, but deliberately **not** the generation
  machinery, which stays on the unmerged branch `feat/american-dataset-v1`
  (DEC-033).
- The CRR label policy has **no frozen `docs/results/` evidence** on any branch.
  Its selection rests on two ignored local artifacts, and its own contract
  section records that the selecting pilot was **not an uncontaminated
  predeclaration** and that the uncontaminated gates would have chosen
  `N = 4096` rather than `N = 1024` (DEC-031, DEC-033).
- Every identity verified in tasks 9D and 9E is internal to one lattice. Task
  9G's locked run supplied the first agreement evidence from a numerically
  unrelated engine: the independent PDE mapping check passed on **21 rows**
  (DEC-038). That is a 21-row mapping-consistency check, **not** a
  dataset-scale cross-check of the 250,000 CRR labels and **not** converged
  American-price truth; the broader cross-check gate remains open
  (DEC-033, DEC-034).
- Task 9E's near-duplicate measurements are descriptive. Because the distances
  were observed before a threshold was predeclared, its near-duplicate gate
  cannot be satisfied retroactively for this dataset version (DEC-034).
- Task 9G closes the admission code's row `label_policy` / `label_steps` gap and
  pins generator version `1.0.0` plus recovered configuration digest
  `d18485c6…`. Its independent PDE check has now **executed and passed on 21 of
  21 rows** inside the locked run, as mapping-consistency evidence only
  (DEC-034, DEC-036, DEC-038).
- The European `forward_normalized_v1` representation is **inadmissible for
  American labels**: it determines a European price exactly but loses the
  separate dependence on rate and dividend yield that the early-exercise
  boundary carries (DEC-033). `american_forward_carry_v1` adds `rT` and `qT`
  to those three inputs; `american_raw_physical_v1` is feature-sufficient but
  not minimal (DEC-034).
- The frozen European replication's ignored original `weights.npz` has been
  recovered locally and preflighted at SHA-256 `42670774…`, alongside manifest
  SHA-256 `054ca945…`. Runtime checks repeat both identities; the files remain
  untracked and no silent retraining or substitution is allowed (DEC-034,
  DEC-036).
- Task 9G is a one-seed, one-budget feasibility pilot and cannot establish H2.
  A promising outcome would have required a separate replication with at least
  five seeds and several predeclared label budgets (DEC-034). **Its outcome was
  not promising**: `failure_to_learn` (DEC-038).
- Task 9G's frozen result is a **negative** one and is never reinterpreted as a
  partial success. Neither arm passed every validation gate; the latency-speedup
  and IV-error gates also failed; and `transfer_validation_rmse_strictly_better
  = true` is a within-pilot comparison confounded by the arm-seeded epoch
  shuffle, so it is **not** evidence that transfer initialization helps
  (DEC-038).
- Task 9G's independent PDE check passed 21 of 21 rows, but it is
  **mapping-consistency evidence only** — not converged American-price truth and
  not semantic-coverage evidence — and its fixed-domain 400x200 versus 800x400
  refinement pair does not independently bound domain-truncation error
  (DEC-038).
- Task 9G's `interpolation_test` partition is **unconsumed**
  (`final_evaluation_attempts = 0`, `final_partition_consumed = false`), and
  final evaluation is **forbidden** under the 9G protocol. Any future final
  evaluation needs a new predeclared protocol and a fresh final partition
  (DEC-038).
- Offline snapshot checking detects internal inconsistency and tracked-input
  drift, but **cannot authenticate a fully coordinated fabricated raw report and
  snapshot** (DEC-036, DEC-038).
- In the processed market partitions, **`exercise_style` and
  `contract_multiplier` are 100 % null**. The XSP-European / SPY-American
  distinction that phase 2 depends on is an external convention, not something
  the local bytes carry (DEC-031).
- The candidate CRR dataset carries a **continuous dividend yield**. That is
  **not** a discrete cash-distribution schedule, so it **cannot model SPY cash
  dividends**, and it is never described as a no-dividend dataset (DEC-001).
- **No accepted, versioned PDE-labelled SPY training dataset exists**, and
  **no American neural surrogate has yet been trained and accepted.**
- **Cloud storage and vendor ingestion are deferred reproducibility work, not
  prerequisites for the first CRR learnability experiment** (DEC-030).
- **Nothing task 9H produces is a project result.** It selects against
  `validation`, repeatedly, so every task 9H number carries selection bias. A
  selected candidate requires a separately predeclared confirmation on a fresh
  final partition, as its own task with its own gates and its own review
  (DEC-041).
- **Task 9H measures prices only.** It makes no Greek, latency,
  implied-volatility or transfer-learning claim, and builds none of that
  machinery. Those are separate follow-up stages that begin only after a
  candidate meets the development criterion.
- Task 9H's conventions could not be added to
  [architecture.md](architecture.md) or any other `docs/*-contract.md`: those
  files are digest-pinned as task 9G tracked inputs and editing them would break
  the historical task 9G protocol check that `scripts/check.sh` and CI run. They
  live in the task 9H specification instead, which is the authority for them
  until a future task can promote them.
- Every exploratory publication produced so far (9C-C2a, 9C-C2b1
  demonstrations) is **refused** by `verify_training_input_publication`,
  because `APPROVED_TRAINING_INPUT_STATUSES` is empty today (DEC-011). This
  is by design, not a bug to fix incidentally.

## Operational constraints

- Long numerical studies are manual, terminal-invoked jobs only — never CI,
  hook-, or agent-triggered. See `AGENTS.md`.
- The proprietary market archive and everything derived from it stay out of
  Git; CI never sees it (DEC-015).
- The full local gate needs the `train` extras
  (`pip install -e '.[dev,train]'`); a missing `pytest`/PyTorch fails the
  gate rather than silently skipping the Python suite.
- `clang-format` formatting is checked locally only when the tool happens to
  be installed, in the post-edit hook and in `./scripts/check.sh`; **it is
  not enforced in CI today.** Do not report `./scripts/check.sh` as having
  verified formatting unless `clang-format` was actually present and ran.
  Full detail: [agent-system.md](agent-system.md), "Clang-format baseline
  problem."
- `.githooks/pre-commit` is not activated by a fresh clone automatically; it
  requires `scripts/install-git-hooks.sh` to be run once. See
  `CONTRIBUTING.md`.

## Technical debt

- The clang-format/CI gap above (tracked, not yet fixed — deferred to a
  future PR per [agent-system.md](agent-system.md)).
- Pre-commit hook activation is manual and undocumented in prior
  `CONTRIBUTING.md` revisions; documented as of this change but still not
  automatic.
- `.claude/skills/` and any additional read-only subagent beyond
  `code-reviewer`/`numerical-reviewer` are planned, not implemented — see
  [agent-system.md](agent-system.md).
- The v2 fixed-bump check `ladder_within_residual_scale` is a **dead
  reporting field**: its condition can never fail in either branch, so it
  never contributed to any verdict. Harmless, and deliberately **not** fixed
  here — the runner is in the frozen snapshot's executable-source inventory,
  so editing it would break the provenance reconciliation of terminal
  evidence. Cleanup belongs to a later, separately versioned study (DEC-027).
- The v2 snapshot links its consumed remediation report only indirectly
  (shared `criteria_digest` / `raw_config_sha256`, plus the confirmation
  report digest). A future schema version may store the consumed
  remediation-report content digest directly — a forward-looking improvement,
  not a defect in the frozen result (DEC-027).

## Artifact/contract map

| Study/engine | Contract | Frozen result |
|---|---|---|
| European replication | [research-contract.md](research-contract.md) | [results/european_replication_results_v1.json](results/european_replication_results_v1.json) |
| European validation | [architecture.md](architecture.md) | [results/european_validation_results_v1.json](results/european_validation_results_v1.json) |
| American CRR | [american-crr-contract.md](american-crr-contract.md) | none frozen (cross-check role only) |
| American LSM | [american-lsm-contract.md](american-lsm-contract.md) | [results/american_lsm_crosscheck_results_v1.json](results/american_lsm_crosscheck_results_v1.json) |
| PDE oracle / surface / harvest / vega | [pde-numerical-contract.md](pde-numerical-contract.md) | [results/american_pde_label_policy_results_v1.json](results/american_pde_label_policy_results_v1.json) (v1 pilot only; 9C-C1/C2a/C2b1 are exploratory infrastructure with no frozen snapshot) |
| PDE label policy v2 (task 9C-C3) | [pde-numerical-contract.md](pde-numerical-contract.md) (Task 9C-C3 section) | [results/american_pde_label_policy_v2_results_v1.json](results/american_pde_label_policy_v2_results_v1.json) — frozen terminal, accepted `grid_1600x800`. Enforced by `scripts/freeze_pde_label_policy_v2_results.py --check` in `scripts/check.sh` and CI, and pinned by `python/tests/test_pde_label_policy_v2_results_snapshot.py` |
| Market ingestion (9A) / reconstruction (9B) | [market-state-reconstruction-contract.md](market-state-reconstruction-contract.md) | none — local-only, never staged |
| Local data holdings (9D audit) | [data-holdings-catalogue.md](data-holdings-catalogue.md) | none — audit record, not frozen evidence, no generator script |
| CRR dataset conditional admission (9E) | [american-crr-dataset-admission.md](american-crr-dataset-admission.md), [american-crr-contract.md](american-crr-contract.md) ("Label policy v1") | none — conditional mapping-only admission, not frozen evidence; implemented checks live in `data/american_admission.py` and incomplete gates are recorded |
| American price-model development loop (9H) | [tasks/active/task-9h-american-pricer-development.md](tasks/active/task-9h-american-pricer-development.md) | none, and none is expected — task 9H produces development records, not frozen evidence. Its recorded search lives in the append-only `docs/attempts/task-9h-attempt-log.jsonl` (six entries, all `criterion_not_met`) and is checked offline by `python3 scripts/american_dev_attempts.py check` in `scripts/check.sh` and CI (DEC-041, DEC-042, DEC-043, DEC-044). The exploratory validation-geometry reports are written beneath the ignored `artifacts/` tree and are never committed |
| American neural-pricer feasibility pilot (9G) | [architecture.md](architecture.md) ("Phase-1 American surrogate representation"), [tasks/active/task-9g-american-neural-pricer-pilot.md](tasks/active/task-9g-american-neural-pricer-pilot.md) | [results/american_neural_pilot_results_v1.json](results/american_neural_pilot_results_v1.json) — frozen negative result, `validation_gates_failed` / `failure_to_learn`, final partition unconsumed. Enforced by `scripts/freeze_american_neural_pilot_results.py --check` in `scripts/check.sh` and CI (DEC-036, DEC-037, DEC-038); result acceptance pending fresh top-level review |

## New-agent checklist

1. Read `AGENTS.md` in full.
2. Read this file in full.
3. Read the active task spec under `tasks/active/`.
4. Read the normative contract for whatever you are about to touch.
5. Check `git status` and the current branch before doing anything else.
6. Do not run an expensive numerical study yourself; report the command if
   asked.
7. Do not commit or push without an explicit instruction in the current
   conversation.
8. If you are reviewing rather than implementing, remember your findings
   are preliminary — see [agent-system.md](agent-system.md).
