# Decision log

Append-only. Entries are identified by a stable ID, never by a calendar date
— this repository does not record wall-clock dates as evidence. Do not edit
or delete a past entry; if a later decision changes an outcome, add a new
entry and cross-link it with "Supersedes" / "Superseded by".

Each entry states context, decision, and consequences at the level a later
task needs to avoid relitigating it. It does not restate the numerical
formulas or full result tables — those live in the linked contract or
snapshot, which remains authoritative for the numbers.

---

### DEC-001 — PDE oracle over LSM/CRR for dividend-bearing labels

- **Status:** Active
- **Context:** The CRR tree and the LSM engine both carry a continuous
  dividend yield; a real American equity option pays discrete cash dividends,
  which a continuous yield does not represent.
- **Decision:** Task 9C-A added a third, numerically independent reference —
  a Crank–Nicolson finite-difference solver with an explicit discrete cash
  dividend jump — as the engine intended to eventually supply price/Greek
  labels for dividend-bearing American contracts.
- **Consequences:** Label generation work for dividend-bearing contracts
  targets the PDE oracle, not CRR or LSM. CRR and LSM keep their existing
  role (DEC-002).
- **Authoritative links:** [pde-numerical-contract.md](pde-numerical-contract.md),
  [research-contract.md](research-contract.md)

### DEC-002 — CRR and LSM remain cross-checks, not production label sources

- **Status:** Active
- **Context:** CRR and LSM agree with each other on the contracts both can
  price, but neither is market truth and neither prices discrete dividends.
- **Decision:** CRR and LSM are kept as an internal cross-check pair
  (stage-2 agreement evidence) rather than extended to become the
  dividend-bearing label source.
- **Consequences:** Their frozen cross-check evidence is descriptive
  agreement between two numerical methods inside one model, not a
  production-readiness claim, and is never reused as an acceptance criterion
  for a label-policy study.
- **Authoritative links:** [american-crr-contract.md](american-crr-contract.md),
  [american-lsm-contract.md](american-lsm-contract.md),
  [results/american_lsm_crosscheck_results_v1.json](results/american_lsm_crosscheck_results_v1.json)

### DEC-003 — Task 9C-B v1 pilot is frozen `no_policy_selected`

- **Status:** Frozen-terminal
- **Context:** The task 9C-B pilot tested whether a fixed grid could supply
  price/delta/gamma/vega labels by centered bumps across three candidate
  policies and 28 predeclared cases.
- **Decision:** The frozen recommendation is
  `selected_accuracy_policy = no_policy_selected`, with
  `criteria_were_not_loosened = true`. The result is accepted as-is and is
  never edited, rerun, or reinterpreted.
- **Consequences:** No American label policy exists from v1. No American
  dataset generation or American neural training may begin from v1 alone.
  The pilot's cost measurement motivated DEC-004 and DEC-012 instead of
  simply buying more compute. Task 9C-C3 (v2) is a new, separately
  predeclared study, not a rerun of this one.
- **Authoritative links:**
  [results/american_pde_label_policy_results_v1.json](results/american_pde_label_policy_results_v1.json),
  [pde-numerical-contract.md](pde-numerical-contract.md) (Task 9C-B section),
  `python/tests/test_pde_label_policy_results_snapshot.py`

### DEC-004 — Surface reuse: one solve, many outputs

- **Status:** Active
- **Context:** The 9C-B pilot needed thirteen independent scalar solves per
  state and grid to assemble four labels, discarding the full valuation-time
  solution each time.
- **Decision:** Task 9C-C1 added `pde_valuation_surface`: one backward
  induction returns the whole valuation-time slice (price, delta, gamma,
  exercise classification, Greek eligibility) plus many requested spots
  evaluated against that single solve.
- **Consequences:** Per-spot re-solving is no longer necessary for reading
  price/delta/gamma off an already-solved grid. The scalar API and its
  $O(N_S)$ working memory are unchanged; a surface query at the scalar spot
  reproduces the scalar price bitwise.
- **Authoritative links:** [pde-numerical-contract.md](pde-numerical-contract.md)
  (Task 9C-C1 section)

### DEC-005 — Group-level partitioning, before any solve

- **Status:** Active
- **Context:** Harvesting many spot rows from one surface, and later three
  surfaces (base/sigma-down/sigma-up) for vega, creates rows that are not
  statistically independent of each other.
- **Decision:** `plan_harvest` assigns partitions (`train` /
  `validation` / `interpolation_test`) by economic **group** identity,
  computed with no solver argument, so assignment provably precedes every
  solve and no group can straddle a partition.
- **Consequences:** A call/put pair or a sigma-down/base/sigma-up triple
  cannot be split across partitions. Duplicate economic states fail before
  the first solve rather than silently leaking.
- **Authoritative links:** [pde-numerical-contract.md](pde-numerical-contract.md)
  (Task 9C-C2a section)

### DEC-006 — Independent groups are not the same count as harvested rows

- **Status:** Active
- **Context:** One design group can emit many rows (many spots, and in the
  vega design, three surfaces), so "rows generated" and "independent
  numerical work" are different quantities.
- **Decision:** Reports publish `raw_rows_per_group` (dataset expansion) and
  `raw_rows_per_attempted_surface` (numerical-work reuse) side by side, each
  with its own integer numerator and denominator.
- **Consequences:** Neither ratio, nor the group count itself, may be
  reported or treated as a statistical effective sample size.
- **Authoritative links:** [pde-numerical-contract.md](pde-numerical-contract.md)
  (Task 9C-C2a section, "Two different yields, always reported together")

### DEC-007 — Three-surface vega (base / sigma-down / sigma-up)

- **Status:** Active
- **Context:** Vega needs a volatility bump, but a naive bump-and-reprice
  breaks the "one solve, many rows" design and its partition guarantees.
- **Decision:** A design declares either exactly `["base"]` or exactly
  `["base", "sigma_down", "sigma_up"]`. With the triple, each leg is solved
  at `sigma - eta`, `sigma`, `sigma + eta`; vega is the centered difference
  of the two bumped prices, per unit absolute volatility. Price, delta, and
  gamma still come only from the base surface. Group atomicity holds: a leg
  whose bumped sibling failed retains nothing.
- **Consequences:** `vega_numerically_available` is a narrow numerical flag,
  not a supervision-eligibility claim. Vega supervision eligibility is
  **undecided** and is governed by the active task 9C-C3 (see
  [tasks/active/task-9c-c3-label-policy-v2.md](tasks/active/task-9c-c3-label-policy-v2.md)),
  not by DEC-009 — DEC-009 concerns gamma only and does not extend to vega.
- **Authoritative links:** [pde-numerical-contract.md](pde-numerical-contract.md)
  (Task 9C-C2b1 section, "Three surfaces, one vega")

### DEC-008 — Economic identity and vega-label identity are different identities

- **Status:** Active
- **Context:** A base pricing node (`row_id`) and that same node evaluated
  under a specific vega convention are both meaningful, but not
  interchangeable, identities.
- **Decision:** An available vega additionally carries
  `vega_label_record_id`, derived from `row_id` and a canonical
  `vega_convention_id`. Equal `row_id` means the same base pricing node;
  equal `vega_label_record_id` means that node under the same vega
  convention. Equal `row_id` with different convention IDs is not an
  identical label record.
- **Consequences:** Downstream consumers must match on the identity that
  actually matters for their use (economic state vs. labeled vega record)
  and must not conflate the two.
- **Authoritative links:** [pde-numerical-contract.md](pde-numerical-contract.md)
  (Task 9C-C2b1 section, "What the vega columns are, and what they are not")

### DEC-009 — Gamma stays evaluation-only

- **Status:** Active
- **Context:** Gamma is read from the same valuation-time slice as delta,
  but no task to date has established a supervision-eligibility rule for it.
- **Decision:** Gamma remains an evaluation-only quantity; no current
  harvested row treats it as a supervised training target.
- **Consequences:** Any future proposal to supervise on gamma needs its own
  predeclared eligibility rule, analogous to what task 9C-C3 is doing for
  vega/delta stability. Not automatically inherited from delta's rule.
- **Authoritative links:** [pde-numerical-contract.md](pde-numerical-contract.md)
  (Task 9C-C1 and 9C-C2b1 sections)

### DEC-010 — Richardson extrapolation is validation-only

- **Status:** Active
- **Context:** Richardson extrapolation was the most accurate candidate
  where the PDE solution is smooth, but its observed factor-two order fell
  outside the predeclared `[1.5, 2.5]` support band on a large minority of
  cases, concentrated at early exercise, dividends, short maturities, and
  deep-in-the-money kinks.
- **Decision:** Richardson is kept as a reference/validation technique, not
  promoted to a label policy. Unsupported observed order is preserved and
  reported as a result, never repaired by assuming second order.
- **Consequences:** No label policy (v1 or v2) may select Richardson as its
  production candidate without first resolving the unsupported-order cases;
  none has been proposed to date.
- **Authoritative links:** [pde-numerical-contract.md](pde-numerical-contract.md)
  (Task 9C-B section), [results/american_pde_label_policy_results_v1.json](results/american_pde_label_policy_results_v1.json)

### DEC-011 — Authoritative verification requires an externally supplied expected configuration

- **Status:** Active
- **Context:** A publication that only checks its own stored digests can be
  internally consistent and still not authentic — a stored digest cannot
  vouch for itself.
- **Decision:** Two guarantees are named and kept separate.
  `verify_publication` proves only self-consistency. Verification against a
  configuration is a distinct entry point, `verify_publication_authoritatively`,
  which replans the harvest deterministically from an externally supplied
  expected configuration (no PDE solve) and recomputes digests rather than
  trusting stored ones. `verify_training_input_publication` requires the
  authoritative path and then refuses every study status, because
  `APPROVED_TRAINING_INPUT_STATUSES` is empty today.
- **Consequences:** No exploratory publication in this repository — including
  the 9C-C2a and 9C-C2b1 demonstrations — currently passes as an approved
  training input, by construction, regardless of how clean its own
  self-consistency check looks.
- **Authoritative links:** [pde-numerical-contract.md](pde-numerical-contract.md)
  (Task 9C-C2b1 section, "Two guarantees, named separately and kept separate")

### DEC-012 — Algorithmic reuse before clusters

- **Status:** Active
- **Context:** The 9C-B pilot's dominant cost was thirteen independent
  scalar solves per state and grid; naively scaling that with more workers
  was available but was not chosen first.
- **Decision:** The next milestone after 9C-B (task 9C-C1) was to read a
  valuation-time slice and its delta/gamma out of a single solve, not to
  add worker pools or cluster execution. Parallel/resumable generation
  remains explicitly deferred (still not implemented as of 9C-C2b1).
- **Consequences:** Any future proposal to scale label generation with
  parallel or cluster execution should first show that the algorithmic
  reuse path (surface solves, grouped harvesting, shared multi-surface
  solves) has been exhausted for the case at hand.
- **Authoritative links:** [research-contract.md](research-contract.md)
  ("The discrete-dividend PDE oracle and its label-policy pilot"),
  [pde-numerical-contract.md](pde-numerical-contract.md) ("Not implemented
  by task 9C-C1")

### DEC-013 — Long numerical studies are run manually, from a terminal

- **Status:** Active
- **Context:** CRR convergence, the LSM cross-check, PDE label-policy
  pilots, and the market ingestion/reconstruction studies are all expensive
  and, for the market studies, gated on private data.
- **Decision:** None of these runs is part of CI, a hook, or an
  agent-triggered action. Each is an explicit, documented CLI invocation a
  human runs and reviews before any snapshot is frozen.
- **Consequences:** An agent session may state the documented command; it
  must not run it as part of an automated gate and must not treat a study's
  completion as self-certifying.
- **Authoritative links:** [project-state.md](project-state.md)
  ("Operational constraints"), AGENTS.md ("No agent-supervised expensive
  numerical runs")

### DEC-014 — No final American dataset before an accepted label policy

- **Status:** Active
- **Context:** Task 9C-B (v1) returned `no_policy_selected`; no policy has
  been accepted since.
- **Decision:** American dataset generation and American neural training
  stay blocked until a label-policy study (task 9C-C3 or later) is accepted.
- **Consequences:** Any dataset-generation or training work proposed before
  that acceptance is out of scope and should be declined or redirected to
  the blocking task.
- **Authoritative links:** [project-state.md](project-state.md),
  [tasks/active/task-9c-c3-label-policy-v2.md](tasks/active/task-9c-c3-label-policy-v2.md)

### DEC-015 — No private market archive, or anything derived from it, in Git

- **Status:** Active
- **Context:** Tasks 9A/9B read a proprietary, licensed quote-level archive.
- **Decision:** The raw archive, every Parquet file derived from it, and
  every report/figure/fitted value derived from it stay out of Git. Tests
  for that code path use synthetic fixtures built with the vendor encoder,
  so CI runs the whole suite without the archive.
- **Consequences:** No PR may stage anything under `data/market-feasibility-v1/`
  or a Parquet/report/figure derived from it. This is a hard git-safety rule,
  not a style preference.
- **Authoritative links:** AGENTS.md ("Git, data, and frozen-result safety"),
  [market-state-reconstruction-contract.md](market-state-reconstruction-contract.md)
