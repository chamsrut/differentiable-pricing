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
not for converged American-price accuracy (DEC-034). The active task is task 9G,
the protocol and implementation for one bounded feasibility pilot; execution is
blocked on its entry gates.
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
  retroactively. The current checks also do not enforce each row's
  `label_policy` and `label_steps` against the manifest policy; task 9G must add
  those invariants and pin generator version `1.0.0` and recovered config digest
  `d18485c6…` before training. The raw dataset is feature-sufficient, but
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
- **No accepted, versioned PDE-labelled SPY training dataset exists**, and no
  SPY neural surrogate exists. **No American neural surrogate has yet been
  trained and accepted.**

## Exact next task

**Task 9G: the protocol-and-implementation PR for the bounded continuous-yield
American neural-pricer feasibility pilot** —
[tasks/active/task-9g-american-neural-pricer-pilot.md](tasks/active/task-9g-american-neural-pricer-pilot.md)
([decision-log.md](decision-log.md) DEC-034). Its first entry gate is a fresh
top-level review of task 9E's reconciled **conditional** admission; that gate
is discharged by `APPROVE PLANNING RECONCILIATION` (DEC-035). The PR must
implement the row/manifest policy invariants, lock dataset and source-
artifact identities, implement and verify the exact European-to-American lift,
and freeze seeds, the single training budget, metrics, latency cases, and IV
cases. It must be reviewed and merged before any manual experiment run.

**No training is authorized or started.** Task 9G is active at protocol and
implementation scope only. The outstanding independent numerical cross-check
must be completed or explicitly resolved before a training run is authorized;
`interpolation_test` remains unavailable until the one-shot final evaluation.

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
  the European loading path is unchanged and its tests pass untouched. Two
  mandatory pre-training invariants are missing: row `label_policy` and
  `label_steps` are not checked against the manifest policy name and steps.
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
- **No independent cross-check of the CRR labels has been run.** Every identity
  verified in tasks 9D and 9E is internal to one lattice; agreement with a
  numerically unrelated engine is untested and must be completed or explicitly
  resolved before training is authorized (DEC-033, DEC-034).
- Task 9E's near-duplicate measurements are descriptive. Because the distances
  were observed before a threshold was predeclared, its near-duplicate gate
  cannot be satisfied retroactively for this dataset version (DEC-034).
- The implemented admission checks do not enforce row `label_policy` or
  `label_steps` against the manifest policy. Those invariants, generator version
  `1.0.0`, and recovered configuration digest `d18485c6…` are mandatory task 9G
  entry checks (DEC-034).
- The European `forward_normalized_v1` representation is **inadmissible for
  American labels**: it determines a European price exactly but loses the
  separate dependence on rate and dividend yield that the early-exercise
  boundary carries (DEC-033). `american_forward_carry_v1` adds `rT` and `qT`
  to those three inputs; `american_raw_physical_v1` is feature-sufficient but
  not minimal (DEC-034).
- The frozen European replication records weights SHA-256 `42670774…`, but the
  original `weights.npz` is not tracked. Transfer cannot start unless that exact
  artifact is recovered; no silent retraining or substitution is allowed
  (DEC-034).
- Task 9G is a one-seed, one-budget feasibility pilot and cannot establish H2.
  A promising outcome requires a separate replication with at least five seeds
  and several predeclared label budgets (DEC-034).
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
| American neural-pricer feasibility pilot (9G) | [architecture.md](architecture.md) ("Phase-1 American surrogate representation"), [tasks/active/task-9g-american-neural-pricer-pilot.md](tasks/active/task-9g-american-neural-pricer-pilot.md) | none — protocol and implementation not yet built; no run authorized |

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
