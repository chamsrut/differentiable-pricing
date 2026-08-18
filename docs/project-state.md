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
PDE oracle), but the first attempt to turn the PDE oracle into a labeling
policy (task 9C-B) returned a negative, frozen result:
`no_policy_selected`. Rather than respond to that cost by adding parallel
compute, the project spent tasks 9C-C1/9C-C2a/9C-C2b1 on algorithmic reuse —
one solve returning a whole valuation-time surface, grouped partitioning
that keeps harvested rows honest about correlation, and a three-surface
vega design — before revisiting the labeling-policy question itself. That
revisit is task 9C-C3, the current active task. No American dataset and no
American neural training exist yet; both are blocked on a policy being
accepted. In parallel, a real-market track (tasks 9A/9B) audited a
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
| Task 9C-C3 predeclaration (criteria, lifecycle, runner, freeze tool) | [pde-numerical-contract.md](pde-numerical-contract.md) (Task 9C-C3 section) — no run executed, no snapshot frozen |

## Current implementation state

- C++ reference engines: Black–Scholes, CRR (scalar + parallel batch), LSM,
  discrete-dividend PDE (scalar price, and the valuation-time surface API).
- Python: dataset generation/diagnostics and training for the European
  surrogate (complete); study runners for CRR convergence, LSM cross-check,
  the 9C-B pilot, and PDE surface/vega harvesting (all exploratory, none
  feeding a production dataset); read-only market ingestion/reconstruction.
- No production label policy exists. No accepted, versioned PDE-labelled
  American neural-training dataset exists. Three distinct things must not
  be conflated here:
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
    constitutes one; that requires an accepted label policy (task 9C-C3 or
    later) and its own dataset-generation task.
  No American neural surrogate exists. No parallel or resumable label
  generation exists.

## Exact next task

**Task 9C-C3: run the predeclared v2 remediation stage, manually, from a
terminal** —
[tasks/active/task-9c-c3-label-policy-v2.md](tasks/active/task-9c-c3-label-policy-v2.md).

The predeclaration is **complete and versioned**: every value the task spec
previously flagged `OPEN` is resolved (decision log DEC-016, DEC-017,
DEC-018), and the criteria, lifecycle, runner and freeze/check tool exist with
cheap tests. Nothing has been run. Concretely, what exists now is
`configs/pde_label_policy_pilot_v2.toml`,
`python/src/differentiable_pricing/american/pde_label_policy_v2.py`,
`scripts/freeze_pde_label_policy_v2_results.py`, and their two test modules.

The next action is a **human-run** ten-case remediation stage — the exact
command is in [pde-numerical-contract.md](pde-numerical-contract.md) ("Task
9C-C3: label-policy v2", "Manual run protocol"). Nine regular cases decide
pass/fail; `stress_american_put_exercise_boundary` is descriptive anchor
evidence. Only if the remediation passes does the unchanged 28-case
confirmation run, once, against the same unadjusted criteria — and the runner
**refuses** to start confirmation without a passing remediation report
carrying the same raw-config and criteria digests. No dataset generation or
training happens inside this task, whichever way it resolves.

## Roadmap to training

In order, after task 9C-C3 resolves (positively or negatively):

1. Task 9C-C3 remediation, then (conditionally) confirmation — see above.
2. A bounded task 9C-C2b2: parallel/resumable generation, only once a
   policy exists to generate against.
3. A small, grouped pilot dataset under that policy.
4. The first neural training run and its learning curves.
5. Scale the dataset only if the pilot run justifies it.
6. Real-quote price/Greek/implied-volatility/surface comparisons — informed
   by, but not a calibration built from, tasks 9A/9B.

## Entry/exit gates

Project-level gates (fixed acceptance criteria, `validation`/
`interpolation_test` separation, consumed-partition discipline) are in
[research-contract.md](research-contract.md), "Provisional gates" and
"Data protocol." They are not restated here.

Task 9C-C3's specific gates:

- **Entry to the remediation run:** v2 stability/shape criterion and the
  ten-case remediation set predeclared and versioned; v1 remains untouched.
  One-shot: these are not revisable inside the task after a result is seen.
- **Entry to the 28-case confirmation run:** the remediation run passed, in
  full, under the same, unadjusted criteria — and only then; a remediation
  failure terminates the task (`confirmation_status = not_run`) rather than
  triggering a redesign-and-retry.
- **Exit:** every terminal outcome freezes a snapshot. Either a remediation-
  or confirmation-stage `no_policy_selected`, or, on confirmation success, a
  **selected candidate pending fresh top-level independent approval**
  (price, and separately, delta/vega where eligible) — not an automatically
  accepted policy. Full detail:
  [tasks/active/task-9c-c3-label-policy-v2.md](tasks/active/task-9c-c3-label-policy-v2.md).

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
- Vega is **numerically available** in the harvested rows but is **not**
  supervision-eligible until task 9C-C3 decides stability (DEC-007). C3's
  criterion for that decision is now predeclared; it has not been run, so no
  vega is supervision-eligible today.
- Task 9C-C3's American dominance and intrinsic allowances are **operational
  price-error scale estimates, not certified bounds** — every v2 report
  publishes `is_a_rigorous_bound = false` (DEC-016).
- A v2 confirmation success would select a **price** candidate pending fresh
  top-level approval; delta and vega eligibility are decided separately, case
  by case (DEC-017). Nothing is selected today.
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

## Artifact/contract map

| Study/engine | Contract | Frozen result |
|---|---|---|
| European replication | [research-contract.md](research-contract.md) | [results/european_replication_results_v1.json](results/european_replication_results_v1.json) |
| European validation | [architecture.md](architecture.md) | [results/european_validation_results_v1.json](results/european_validation_results_v1.json) |
| American CRR | [american-crr-contract.md](american-crr-contract.md) | none frozen (cross-check role only) |
| American LSM | [american-lsm-contract.md](american-lsm-contract.md) | [results/american_lsm_crosscheck_results_v1.json](results/american_lsm_crosscheck_results_v1.json) |
| PDE oracle / surface / harvest / vega | [pde-numerical-contract.md](pde-numerical-contract.md) | [results/american_pde_label_policy_results_v1.json](results/american_pde_label_policy_results_v1.json) (v1 pilot only; 9C-C1/C2a/C2b1 are exploratory infrastructure with no frozen snapshot) |
| PDE label policy v2 (task 9C-C3) | [pde-numerical-contract.md](pde-numerical-contract.md) (Task 9C-C3 section) | none — predeclared, not yet run. Its canonical snapshot path is `docs/results/american_pde_label_policy_v2_results_v1.json`; `scripts/freeze_pde_label_policy_v2_results.py --check` correctly fails while it is absent |
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
