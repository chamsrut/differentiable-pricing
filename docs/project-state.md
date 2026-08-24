# Project state

The canonical, living handoff: what the project believes, what it has actually
shown, and the exact next task. If this file and a narrative document (README,
[agentic-workflow.md](agentic-workflow.md)) disagree, this file wins. It does
**not** win against code, versioned configuration, frozen evidence, or a
normative contract — it summarizes and links those rather than restating their
numbers. Full precedence: [documentation-map.md](documentation-map.md).

**Update rule.** Whoever starts, changes the scope of, or completes a milestone
updates this file in the same change and appends a
[decision-log.md](decision-log.md) entry. A stale "exact next task" here is a
defect in this file, not a fact about the project.

---

## Research thesis

Can a smooth neural surrogate reproduce a trusted derivative pricer's prices
**and** its useful risk sensitivities, across a declared domain, at materially
lower end-to-end latency, with an error profile suitable for a stated decision?

Hypotheses, stage reference standards, metrics, and acceptance methodology:
[research-contract.md](research-contract.md). Not restated here.

The American specialization the current work answers, from the locked stage-2
roadmap (DEC-028): can a neural surrogate price American options with useful
accuracy while delivering materially faster inference than the numerical method
that generated its labels — and does that speedup carry through to
implied-volatility inversion and surface construction?

---

## Current accepted architecture

**[american-neural-architecture-freeze-v2.3.md](american-neural-architecture-freeze-v2.3.md)
is the normative parent of the American neural-pricer roadmap** (DEC-049). It
is a contract, not narrative; where any other document disagrees with it about
the American neural design, it wins. This file does not duplicate it.

What it settles, in one paragraph each:

- **The teacher is reproducible C++ CRR.** `L_N = ½[CRR_N + CRR_{N+1}]`, with
  both components persisted. `N` is not a network input, and `CRR_N` is not
  recoverable from an `N+1` tree.
- **Base label and deep reference are separate objects.** Final evaluation
  reports the three-way decomposition `V_NN − L_N`, `V_NN − L_ref`, and
  `L_N − L_ref`, so teacher discretization is never silently charged to the
  network. Price-reference and per-Greek reference depths may differ.
- **All v2 partitions are generated atomically**, by one invocation of one
  committed generator configuration, bound by one manifest. "Untouched final"
  becomes a property of a precommitted sampling design rather than of a second
  sampling event.
- **The primary final interpolation set is neutrally sampled.** Model-specific
  crossover and projection scans are separate adversarial stress diagnostics,
  never the headline denominator.
- **Adaptive-development controls survive the rewrite**: append-only attempt
  ledger, declared attempt budget, development holdouts, final-partition access
  guard, one-shot final evaluation, no post-final tuning under the same claim.
- **The model artifact is deterministic and non-pickle**, and native C++
  inference — not PyTorch — is the production latency path. No speedup is
  asserted until native timings exist.
- **E2c cannot be the v2 confirmation model**, because its teacher dataset is
  not reproducible from committed source. It survives only as a concrete
  specimen for the native prototype.

---

## Current implementation and evidence

**Stage 1 — European: complete, replicated, terminal.** A precommitted protocol
ran once end to end on a fresh 250,000-row dataset and a new seed; all 11
validation and locked-final gates passed with zero material European-bound
violations. `interpolation_test` is permanently consumed and is never rerun.
Frozen:
[results/european_replication_results_v1.json](results/european_replication_results_v1.json),
[results/european_validation_results_v1.json](results/european_validation_results_v1.json).

**Stage 2 — American reference engines: three, numerically independent.** C++
Black–Scholes with analytic Greeks; CRR (scalar and parallel batch); LSM policy
fitting and valuation, frozen as a cross-check
([results/american_lsm_crosscheck_results_v1.json](results/american_lsm_crosscheck_results_v1.json));
and a Crank–Nicolson discrete-dividend PDE oracle with a valuation-time
price/delta/gamma surface.

**An accepted PDE label policy exists — and authorizes labeling only.** Task
9C-C3 selected `grid_1600x800` and a fresh top-level session approved it
(DEC-025). It covers price on 22/22 regular cases, delta on 18/22, vega on
22/22; gamma is evaluation-only. The frozen report records
`authorizes_dataset_generation = false` and `authorizes_training_input = false`.
Frozen:
[results/american_pde_label_policy_v2_results_v1.json](results/american_pde_label_policy_v2_results_v1.json).

**The local CRR dataset is conditionally admitted, mapping-only.** 250,000 rows
of synthetic continuous-dividend-yield American CRR labels, Git-ignored and
untracked. Admission covers learning the known continuous-yield CRR mapping and
nothing else (DEC-033, DEC-034, DEC-035);
[american-crr-dataset-admission.md](american-crr-dataset-admission.md). It is
**not** regenerable from committed source — the generator is on the unmerged
branch `feat/american-dataset-v1` — which is precisely why v2.3 requires a
restored generator before the confirmation campaign.

**Task 9G — the one American neural experiment that has run — failed.** Its
locked `run-to-validation` was invoked once by the human operator. Every entry
gate passed, including an independent PDE mapping check at 21/21 rows and the
exact European-to-American transfer lift at all eight pinned probes. Neither
arm then passed every validation gate. Recorded outcome:
`status=validation_gates_failed`, `outcome=failure_to_learn`. Frozen:
[results/american_neural_pilot_results_v1.json](results/american_neural_pilot_results_v1.json)
(DEC-038); merged to `main` at `5c0ef6a` (PR #27).

**Task 9H — archived exploratory line, not merged, not a result.** The adaptive
price-model development loop that followed 9G reached a recorded price leader,
recorded E2c, and paused its attempt loop for a zero-training diagnostic phase
whose grids were predeclared and implemented but not scored. All of it — the
workbench, the append-only attempt ledger, the interim research brief, the
E2c checkpoint, and decision entries `DEC-041`–`DEC-048` — lives **only** on the
unmerged branch `experiment/task-9h-american-pricer-development`, archived at
tag `task-9h-v1-exploratory-pre-v2.3` (commit `3950ed0`). Its roadmap is
superseded by v2.3 (DEC-049); its evidence is preserved as exploratory history.

**Not built:** no accepted, versioned American training dataset; no accepted
American neural surrogate; no reproducible American generator on this branch;
no deterministic C++-loadable model artifact; no native C++ inference path; no
SPY dataset or surrogate. C++ deployment today is `SmoothMlp` inference only.

---

## Current limitations and non-claims

- **Every neural-pricer result is synthetic.** No market data, quoted price, or
  calibration target enters any dataset, training run, or reported metric on
  the surrogate side. The separate read-only market track (tasks 9A/9B) does
  read real quotes, but produces no price, Greek, label, or calibrated value and
  feeds no dataset or training run; the two are never mixed.
- **No American neural surrogate has been trained and accepted.** Task 9G
  trained two arms and accepted neither. Its `failure_to_learn` is a valid,
  complete negative result and is never reinterpreted as a partial success —
  `transfer_validation_rmse_strictly_better = true` is a within-pilot
  observation confounded by the arm-seeded epoch shuffle, not evidence that
  transfer initialization helps (DEC-038).
- **One seed and one budget do not establish H2.** Any H2 claim needs a
  separately predeclared replication with at least five seeds and several label
  budgets (DEC-034).
- **Task 9G's `interpolation_test` is unconsumed and stays that way.**
  `final_evaluation_attempts = 0`, `final_partition_consumed = false`, and
  `final-evaluate` is **forbidden** under that protocol because its final-entry
  rule failed. Any future final evaluation needs a new predeclared protocol and
  a fresh final partition (DEC-038).
- **An accepted label policy is not an accepted dataset**, and neither is
  tracked demo code or its ignored local output. All three are separately
  gated (DEC-011, DEC-014, DEC-025).
- **The CRR dataset is not reproducible from committed source**, its label
  policy has **no frozen `docs/results/` evidence** on any branch, and its own
  contract section records that uncontaminated gates would have chosen
  `N = 4096` rather than `N = 1024` (DEC-031, DEC-033).
- **The CRR dataset carries a continuous dividend yield.** That is not a
  discrete cash-distribution schedule, so it cannot model SPY cash dividends,
  and it is never described as a no-dividend dataset (DEC-001).
- **Cross-engine agreement is thin.** Task 9G's independent PDE check passed
  21/21 rows as *mapping-consistency* evidence only — not a dataset-scale
  cross-check of 250,000 labels, not converged American-price truth, and not
  semantic-coverage evidence. The broader cross-check gate remains open
  (DEC-034, DEC-038).
- **Task 9E's near-duplicate gate cannot be satisfied retroactively** for this
  dataset version, because no threshold was predeclared before task 9D observed
  the distances. Those measurements stay descriptive (DEC-034).
- **Greek claims are bounded by reference quality.** Agreement with a
  finite-depth CRR Greek does not establish a continuous-American Greek unless
  the numerical Greek reference is itself adequately converged. Where it is not,
  the reference limitation is reported and no continuous-American claim is made
  (freeze §15, §16.5). **There is no Gamma-degeneracy statistic**, and none is
  expected: the Vega metric exists because the declared downstream IV Newton
  workflow divides by Vega, and Gamma gets one only if a future declared
  workflow makes Gamma-near-zero operationally relevant. Gamma is assessed on
  accuracy, sign/convexity behaviour, and crossover curvature (freeze §17.5).
- **No latency claim.** No native inference path exists, so no measured speedup
  exists. Memory-bandwidth and FLOP arithmetic are plausibility checks, never
  latency results (freeze §13.3).
- **In-envelope interpolation only.** No boundary, extrapolation/OOD, or
  scenario-shock partition has been built or evaluated.
- **No calibrated curve or volatility surface exists.** The real-market track
  (tasks 9A/9B) reads three sessions of a proprietary quote archive read-only
  and produces no price, Greek, label, or calibrated value. In its processed
  partitions `exercise_style` and `contract_multiplier` are 100% null, so the
  XSP/SPY distinction phase 2 depends on is an external convention, not
  something the local bytes carry (DEC-031).
- **Offline snapshot checking has a known limit.** It detects internal
  inconsistency and tracked-input drift, but cannot authenticate a fully
  coordinated fabricated raw report plus snapshot (DEC-036, DEC-038).
- **`selection_pending_fresh_top_level_approval = true` in the frozen v2
  snapshot is historical and correct.** The report predates its approval; the
  snapshot is never edited to flip it (DEC-027).

Study-specific non-claims that remain in force — correlated harvested rows,
group-level partitioning, effective sample size, gamma's evaluation-only status,
the `stress_euro_short_low_vol_atm` descriptive failure, and the
`is_a_rigorous_bound = false` scale estimates — are stated in
[pde-numerical-contract.md](pde-numerical-contract.md) and the linked decision
entries (DEC-006, DEC-007, DEC-009, DEC-016, DEC-017, DEC-026), which remain
authoritative for them.

---

## Exact next task

**Task 9I — the Architecture v2.3 Phase 0 transition** —
[tasks/active/task-9i-architecture-v2-3-phase-0.md](tasks/active/task-9i-architecture-v2-3-phase-0.md)
(DEC-049). Two independent, **exploratory** workstreams run in parallel:

**A. Finish only the carried-forward Greek/head diagnostics** from the archived
Task 9H line: the depth-convergence qualification propagation, the
reference-digest and protocol-commit recording, the normalized-Vega degeneracy
units repair with its three-bucket excess report, the removal of Gamma
degeneracy from decision-bearing interpretation, and the B.2 eligibility
dimensional-invariance test — all committed **before** any Grid-1 model result
is observed — then Grid 1 validation, Grid 1 H1, Grid 2 spot crossover, and
Grid 2b volatility crossover.

**B. Independently build the native C++ E2c feasibility prototype**: the exact
frozen deployed function in float64, numerically equivalent to fixed Python
test vectors first, then benchmarked at batch 1 and batch 8 against a matched
native CRR adjacent-average request.

**These are exploratory inputs to the v2 design, not confirmation work.**
Neither may open a final partition, spend a neural attempt, or promote E2c's
weights toward the v2 confirmation model. A negative or inconclusive answer on
either side is a valid completion.

After Phase 0 is understood, the freeze's **Phase 1** — committing the v2
architecture, application tolerances, reference-error budget, Greek-reference
rules, attempt budget, sampling design, final-access policy, and native
benchmark contract — is the next task, and needs its own spec and review.

Everything else under `tasks/active/` is completed, deferred, or on hold. The
lifecycle grouping is [tasks/README.md](tasks/README.md); specs are marked in
place and never relocated so that decision-log links keep resolving.

---

## Major completed milestones

| Milestone | Outcome | Evidence |
|---|---|---|
| European neural-pricer replication | Passed all 11 gates; terminal | [results/european_replication_results_v1.json](results/european_replication_results_v1.json) |
| American CRR reference (scalar + parallel batch) | Complete | [american-crr-contract.md](american-crr-contract.md) |
| American LSM cross-check | Complete, frozen | [results/american_lsm_crosscheck_results_v1.json](results/american_lsm_crosscheck_results_v1.json) |
| Tasks 9A/9B: market feasibility audit and state reconstruction | Complete, local-only, nothing staged | [market-state-reconstruction-contract.md](market-state-reconstruction-contract.md) |
| Task 9C-A: discrete-dividend PDE oracle | Complete | [pde-numerical-contract.md](pde-numerical-contract.md) |
| Task 9C-B: PDE label-policy pilot v1 | Frozen `no_policy_selected` — a valid negative result | [results/american_pde_label_policy_results_v1.json](results/american_pde_label_policy_results_v1.json) |
| Tasks 9C-C1/C2a/C2b1: valuation-time surface, grouped harvesting, three-surface vega | Complete, exploratory infrastructure; no frozen snapshot | [pde-numerical-contract.md](pde-numerical-contract.md) |
| Task 9C-C3: PDE label-policy v2 | Accepted `grid_1600x800`, externally approved; labeling only | [results/american_pde_label_policy_v2_results_v1.json](results/american_pde_label_policy_v2_results_v1.json) (DEC-025) |
| Task 9D: local data-holdings catalogue and integrity audit | Complete; admits nothing | [data-holdings-catalogue.md](data-holdings-catalogue.md) (DEC-031) |
| Task 9E: CRR dataset admission | Closed conditional and partial — mapping-only; cross-check, semantic-coverage, and near-duplicate gates not discharged | [american-crr-dataset-admission.md](american-crr-dataset-admission.md) (DEC-034, DEC-035) |
| Task 9G: American neural-pricer feasibility pilot | **Negative** — `validation_gates_failed` / `failure_to_learn`; final partition unconsumed | [results/american_neural_pilot_results_v1.json](results/american_neural_pilot_results_v1.json) (DEC-038) |
| Task 9H: adaptive American price-model development | Archived exploratory; superseded by v2.3; not a project result | branch `experiment/task-9h-american-pricer-development`, tag `task-9h-v1-exploratory-pre-v2.3` (DEC-049) |

---

## Operational constraints

- **Long numerical studies are manual, human-invoked terminal jobs only** —
  never CI-, hook-, or agent-triggered. See [../AGENTS.md](../AGENTS.md).
- **Frozen evidence is never hand-edited.** Each snapshot family has one
  designated generator/validator script;
  `scripts/freeze_pde_label_policy_v2_results.py --check` and
  `scripts/freeze_american_neural_pilot_results.py --check` run offline in
  `scripts/check.sh` and CI, reading only tracked files.
- **Five contracts are content-digest-pinned** by task 9G's frozen protocol
  (`configs/american_neural_pilot_protocol_v1.toml`, enforced in
  `scripts/check.sh` and CI): [architecture.md](architecture.md),
  [research-contract.md](research-contract.md),
  [american-crr-contract.md](american-crr-contract.md),
  [american-crr-dataset-admission.md](american-crr-dataset-admission.md), and
  [pde-numerical-contract.md](pde-numerical-contract.md). Editing any of them by
  one byte fails that gate. They therefore still describe the pre-v2.3 plan in
  places; where they disagree with the freeze about the American neural design,
  **the freeze wins**, and they are read as the record of what task 9G built.
  See [documentation-map.md](documentation-map.md), "Digest-pinned documents."
- **The proprietary market archive and everything derived from it stay out of
  Git**; CI never sees it (DEC-015).
- The full local gate needs `pip install -e '.[dev,train]'`; a missing
  pytest/PyTorch fails the gate rather than silently skipping the Python suite.
- **`clang-format` is not enforced in CI**, and locally it runs only when the
  tool happens to be installed. Never report `./scripts/check.sh` as having
  verified C++ formatting unless `clang-format` actually ran. Detail:
  [agent-system.md](agent-system.md), "Clang-format baseline problem."
- `.githooks/pre-commit` is inactive in a fresh clone until
  `scripts/install-git-hooks.sh` is run once.

## Known technical debt

- The clang-format/CI gap above — tracked, deferred to a future PR.
- `ladder_within_residual_scale` in the v2 runner is a dead reporting field
  whose condition can never fail. Deliberately **not** fixed: the runner is
  digest-pinned in a frozen snapshot's executable-source inventory, so editing
  it would break provenance reconciliation of terminal evidence (DEC-027).
- The v2 snapshot links its consumed remediation report only indirectly. A
  future schema version may store that digest directly — a forward-looking
  improvement, not a defect in the frozen result (DEC-027).
- `.claude/skills/` and any read-only triage subagent beyond `code-reviewer` /
  `numerical-reviewer` are planned, not implemented
  ([agent-system.md](agent-system.md)).

---

## Authoritative links

| Topic | Document |
|---|---|
| American neural-pricer architecture (normative parent) | [american-neural-architecture-freeze-v2.3.md](american-neural-architecture-freeze-v2.3.md) |
| Hypotheses, gates, data protocol, non-claims | [research-contract.md](research-contract.md) |
| Language boundary, artifact contract, stage-1 model math, phase-1 American representation | [architecture.md](architecture.md) |
| CRR lattice and convergence semantics | [american-crr-contract.md](american-crr-contract.md) |
| LSM estimand, uncertainty, control variate | [american-lsm-contract.md](american-lsm-contract.md) |
| PDE oracle, surface, harvesting, label policies v1/v2 | [pde-numerical-contract.md](pde-numerical-contract.md) |
| Market-state reconstruction and its input contract | [market-state-reconstruction-contract.md](market-state-reconstruction-contract.md) |
| CRR dataset conditional admission | [american-crr-dataset-admission.md](american-crr-dataset-admission.md) |
| Local data holdings (9D audit) | [data-holdings-catalogue.md](data-holdings-catalogue.md) |
| Decisions, append-only | [decision-log.md](decision-log.md) |
| Which document wins | [documentation-map.md](documentation-map.md) |
| Task index and lifecycle | [tasks/README.md](tasks/README.md) |
| Agent roles, review independence, planned pieces | [agent-system.md](agent-system.md), [agentic-workflow.md](agentic-workflow.md) |

## New-agent checklist

1. Read [../AGENTS.md](../AGENTS.md) in full.
2. Read this file in full.
3. Read
   [american-neural-architecture-freeze-v2.3.md](american-neural-architecture-freeze-v2.3.md)
   if the work touches the American neural-pricer roadmap.
4. Read the active task spec, [tasks/README.md](tasks/README.md) for lifecycle.
5. Read the normative contract for whatever you are about to touch.
6. Check `git status` and the current branch before doing anything else.
7. Do not run an expensive numerical study yourself; report the command.
8. Do not commit or push without an explicit instruction in the current
   conversation.
9. If you are reviewing rather than implementing, your findings are
   **preliminary** — see [agent-system.md](agent-system.md).
