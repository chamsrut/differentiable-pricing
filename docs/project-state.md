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
unblocked is *labeling*, and only that: **no American dataset and no American
neural training exist yet**, and both remain separately gated behind task
9C-C2b2's infrastructure and a bounded pilot. In parallel, a real-market track (tasks 9A/9B) audited a
three-session proprietary quote archive and established, read-only, which
pricing inputs it can and cannot supply — it produces no price, label, or
calibrated value and stays fully out of Git.

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

## Exact next task

**Task 9C-C2b2: deterministic parallel/resumable production-generation
infrastructure, using the accepted policy** —
[tasks/active/task-9c-c2b2-parallel-resumable-generation.md](tasks/active/task-9c-c2b2-parallel-resumable-generation.md).

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
remediation cases and its 28 confirmation cases are consumed evidence.

C2b2's initial scope is **infrastructure, bounded validation and one small,
manually invoked pilot**. It must **not** claim or launch a production dataset.
Dataset-scale generation and neural training stay separately gated, exactly as
before — the accepted policy unblocks *labeling*, not *a dataset*
([decision-log.md](decision-log.md) DEC-014, DEC-025). The frozen v2 report
itself authorizes neither: it records
`authorizes_dataset_generation = false` and
`authorizes_training_input = false`, and
`AUTHORIZED_TRAINING_INPUT_STATUSES` is still empty.

## Roadmap to training

In order, now that task 9C-C3 has resolved positively:

1. ~~Task 9C-C3 remediation, then confirmation~~ — **done**, accepted
   `grid_1600x800` (DEC-025).
2. A bounded task 9C-C2b2: deterministic parallel/resumable generation
   infrastructure, bounded validation, and one small manually invoked pilot —
   the current active task.
3. A small, grouped pilot dataset under the accepted policy, separately
   gated.
4. The first neural training run and its learning curves.
5. Scale the dataset only if the pilot run justifies it.
6. Real-quote price/Greek/implied-volatility/surface comparisons — informed
   by, but not a calibration built from, tasks 9A/9B.

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

Task 9C-C2b2's specific gates:

- **Entry:** an accepted label policy exists to generate against — satisfied
  by DEC-025 — and C2b2 predeclares its own generation domain, which it does
  **not** inherit from C3's 28 evidence cases.
- **Exit of the initial scope:** deterministic parallel/resumable
  infrastructure, bounded validation, and one small manually invoked pilot
  whose output is a demonstration of the machinery, not a dataset.
- **Explicitly not authorized by entry:** production dataset generation and
  neural training. Each needs its own gate; the frozen v2 report authorizes
  neither (DEC-014, DEC-025).

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
