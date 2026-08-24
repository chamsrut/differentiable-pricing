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

### DEC-016 — Task 9C-C3 takes the residual/scale-aware path, not a tightened solve

- **Status:** Active
- **Context:** Task 9C-C3's spec left one binary choice open: make the
  American-dominance check residual/scale-aware, or tighten the solve until
  the existing fixed `1e-8` tolerance is no longer smaller than the solver's
  own accumulated PSOR residual. v1's single dominance failure was a 2.7e-8
  gap against that 1e-8 tolerance — the size of the residual itself.
- **Decision:** The residual/scale-aware path. The dominance and intrinsic
  allowances become
  `max(shape_absolute_floor, operational price-error scale estimate)`, built
  from the **absolute** `maximum_lcp_residual` accumulated over the time steps
  actually taken, and are **never widened past the price absolute-error cap**.
  A scale above that cap records `residual_scale_exceeds_price_cap` and fails
  the shape check itself. The v1 fixed tolerance survives only as the floor.
  The same residual-scale reasoning supplies the fixed-bump noise floors, and
  those are **per ladder increment**, not one shared tolerance: with
  `E = M_base * R_base` the delta ladder uses
  `epsilon_small = 3E/(2h)` and `epsilon_large = 3E/(4h)`, and the vega ladder
  uses `err(h)+err(2h)` and `err(2h)+err(4h)` with
  `err(b) = (M_down_b*R_down_b + M_up_b*R_up_b)/(2b)`. Each rung carries its own
  estimator error scale, and **no cancellation is claimed between the node
  errors of one shared surface** — the tolerance adds the rung scales rather
  than assuming they offset.
- **Consequences:** These allowances are **operational price-error scale
  estimates, not certified bounds**; every v2 report publishes
  `is_a_rigorous_bound = false` and no later document may describe them as
  bounds. Tightening the solve remains available to a future, separately
  versioned study; it is not what C3 tests. The four v1 absolute-error caps
  are reused unchanged and deliberately, so that v2's question — can a revised
  *stability and shape* rule pass? — stays comparable to v1's answer.
- **Authoritative links:** [pde-numerical-contract.md](pde-numerical-contract.md)
  ("Task 9C-C3: label-policy v2"),
  [tasks/active/task-9c-c3-label-policy-v2.md](tasks/active/task-9c-c3-label-policy-v2.md),
  `configs/pde_label_policy_pilot_v2.toml`

### DEC-017 — The v2 gate decides price; delta and vega eligibility are separate

- **Status:** Active
- **Context:** v1 failed a whole case when the *reference* Greek moved across
  the bump ladder, so an unstable Greek could veto an accurate price. Task
  9C-C3's spec, in contrast, restricts a confirmation-stage selection to
  **price** and requires delta and vega each to pass their own explicit
  stability-eligibility test, allowing a case to be price-selected and
  simultaneously delta- or vega-ineligible.
- **Decision:** v2 separates the two explicitly, versions the split in
  `[gating]`, and makes it **role-aware**. Every solve is classified
  `price_critical`, `delta_only`, `vega_only`, `anchor_descriptive` or
  `stress_descriptive`, and a failure propagates only to the decisions its role
  supports. The nine gate-eligible regular cases decide pass/fail on
  price-critical solve health, the price cap and price shape (dominance,
  intrinsic, monotonicity, convexity) only. The fixed-bump validation (E1,
  delta and vega), the grid-stencil validation (E2, delta), the Greek residual
  scales and a Greek's numerically invalid residual scale all decide
  **eligibility** for that Greek and never the price gate. An anchor-rung
  failure records `anchor_evidence_complete = false` and vetoes nothing; any
  stress-case failure is descriptive. Structural task 9C-C1 centre eligibility
  is **common** to delta and vega, not delta-only. A solver exception is caught
  per case and per solve role, recorded structurally, and the stage continues.
- **Consequences:** A v2 confirmation success selects a **price** candidate
  pending fresh top-level approval, with delta and vega granted case by case.
  A Greek-specific failure can never veto an otherwise valid price policy.
  Gamma is outside this entirely (DEC-009): its error is published, gates
  nothing, and never becomes a label. `spot_convexity` gates as a sign
  property of the price surface, not as a gamma accuracy criterion. An
  executable truth table over every row of the matrix is a test, not prose.
- **Authoritative links:** [pde-numerical-contract.md](pde-numerical-contract.md)
  ("Task 9C-C3: label-policy v2", "What gates, and what only decides
  eligibility"),
  [tasks/active/task-9c-c3-label-policy-v2.md](tasks/active/task-9c-c3-label-policy-v2.md)

### DEC-018 — The v2 production delta is the exact-node grid-stencil delta

- **Status:** Active
- **Context:** v1 derived delta by bumping the spot and re-solving, which
  costs extra solves, carries finite-bump truncation the label can never shed,
  and was the quantity whose reference kept moving across the bump ladder.
  Task 9C-C1 already returns a nodewise stencil delta from a single solve
  (DEC-004).
- **Decision:** v2's production delta is the task 9C-C1 nodewise grid-stencil
  delta read at an **exact grid node** of the candidate surface. Every
  configured centre and bumped spot must be an exact interior node of every
  grid the study runs — checked at configuration-parse time and again bitwise
  against the returned node vector — so no label is ever interpolated. The
  fixed-bump ladder is retained as *validation* of that label, not as the
  label. One base surface per grid therefore supplies the centre price, the
  stencil delta and gamma, and every spot-bumped price.
- **Consequences:** The grid-stencil validation adds no bump-bias charge,
  because its grid-Richardson reference converges toward the mathematical
  derivative rather than toward a finite-bump difference. A production row
  cannot carry its own reference error or observed order, and a one-bump row
  cannot re-estimate its own vega bump bias; both limits are published as
  non-claims in every v2 report. Task 9C-C2b2 must predeclare its own
  generation domain — the 28 evidence cases validate isolated points, not the
  surrounding hyperrectangle — and no dataset or training input is authorized
  by task 9C-C3 whatever its outcome (DEC-014).
- **Authoritative links:** [pde-numerical-contract.md](pde-numerical-contract.md)
  ("Task 9C-C3: label-policy v2", "E2: production grid-stencil delta
  validation"), [pde-numerical-contract.md](pde-numerical-contract.md) (Task
  9C-C1 section)

### DEC-019 — v2 verification recomputes, and says what it cannot prove

- **Status:** Active
- **Context:** A stage report and a frozen snapshot both invite the same
  mistake the harvesting work already named (DEC-011): reading a published
  verdict back and calling that verification. Task 9C-C3's confirmation gate
  and its freeze tool each consume a report that a later reader cannot
  re-derive by hand.
- **Decision:** Both consume the **externally supplied checked-in
  configuration** and recompute rather than trust. One shared derivation
  rebuilds every case decision from the report's raw per-solve numbers and
  re-derives the aggregates, the solve accounting and the lifecycle; the
  schema is exact at every level; the criteria block and its digest are
  rebuilt from the configuration; and an inventory of **every executable
  source the runner and freeze path rely on** — v2 runner, v2 freeze script,
  the v1 label-policy module, the `canonical_payload` module, the package
  init, the PDE header, the PDE implementation and the binding source — is
  digested individually plus as one composite and reconciled against the
  repository files. Confirmation refuses to start unless the **recomputed**
  remediation result is `passed`/`pending`, before any solver call.
- **Consequences:** Three limits are published rather than glossed. Semantic
  verification detects inconsistent or partial mutation, including one whose
  file hashes were regenerated; it **cannot** authenticate a fully coordinated
  fabricated numerical report, because nothing re-solves. Source digests record
  which files were present; they **do not** prove the loaded extension binary
  was built from them, so the binding's self-reported build digests are kept in
  separate `reported_binding_*` fields. One-shot enforcement — the nonempty
  output refusal and the snapshot-overwrite refusal, with no `--overwrite` flag
  on the stage CLI at all — is **best effort**: a second run in another
  directory or clone cannot be detected, so one-shot status is procedural,
  provenance-backed and independently reviewed.
- **Authoritative links:** [pde-numerical-contract.md](pde-numerical-contract.md)
  ("Task 9C-C3: label-policy v2", "Confirmation entry is authoritative and
  semantic", "The freeze tool recomputes rather than trusts",
  "Executable-source provenance", "One-shot enforcement, stated honestly")

### DEC-020 — The v2 price reference is the raw 3200x1600 centre node

- **Status:** Active
- **Context:** v1's price reference was the 3200x1600 rung *selectively
  Richardson-extrapolated with 1600x800*, applied only where the observed
  factor-two order fell inside the supported band. That is a conditional
  reference: whether it extrapolates depends on a measured quantity, so two
  cases can be judged against structurally different references, and the
  800x400 rung silently participates in the price verdict through the order
  estimate.
- **Decision:** v2 uses an **unconditional raw** price reference: the
  exact-node centre value of the 3200x1600 base surface, pinned as
  `price_reference_method = "raw_grid_3200x1600_center"` in the configuration,
  in the criteria digest and in every report, and rejected at parse time and at
  report-verification time if it differs. There is no price Richardson
  extrapolation and no conditional price-reference fallback. `grid_800x400` is
  solved only to estimate the E2 delta observed order and is price-irrelevant.
- **Consequences:** Richardson keeps exactly two roles in v2 — the E2
  grid-stencil delta reference and the descriptive anchor diagnostics — and
  `richardson_contract` publishes `applies_to_price_reference = false`
  (DEC-010 is unchanged). The 6400x3200 anchor stays descriptive and can never
  veto ordinary price selection. All four v1 absolute caps stay unchanged, so
  the comparison with v1 remains meaningful; only the reference construction
  differs, and that difference is recorded here rather than inferred. Tests
  assert that moving the 800x400 price leaves the price block, the gating
  checks and both eligibility verdicts bitwise unchanged, while moving its
  stencil delta moves the observed order and nothing else.
- **Authoritative links:** [pde-numerical-contract.md](pde-numerical-contract.md)
  ("Task 9C-C3: label-policy v2", "The price reference: raw 3200x1600 centre
  node, unconditional"), `configs/pde_label_policy_pilot_v2.toml`

### DEC-021 — v2 verification is exact at every level and derives every aggregate

- **Status:** Active
- **Context:** DEC-019 established that v2 verification recomputes rather than
  trusts. Two gaps survived it: strict schema checking stopped at the top level
  of a report, and the frozen snapshot retained aggregates — the stage solve
  accounting in particular — that its own distilled rows could not reproduce.
- **Decision:** Every config-derived report section is now **rebuilt from the
  configuration and compared as a whole canonical payload** (`study`,
  `conventions`, `criteria_block`, `validation_rules` and its sub-objects,
  `eligibility_contract`, `richardson_contract`, `predeclared_criteria`,
  `solver`, each `grids[]` record), which rejects unknown, missing, duplicated
  and retyped fields at any depth; `performance` gets an exact key set with
  per-field type and range checks. The freeze validator recomputes **every**
  published aggregate from the lowest-level rows, and where a retained
  aggregate was not derivable the minimal immutable per-case fields were added
  to the snapshot rows — implicit linear solves, PSOR solves and PSOR
  iterations — rather than keeping an unverifiable number. The four canonical
  protocol strings (`study.name`, `delta_method`, `vega_method`,
  `price_reference_method`) became module constants that the report is built
  from and the parser requires exactly. `cpp/include/dp/option.hpp` and
  `cpp/src/option.cpp` joined the executable-source inventory after walking the
  PDE path's include graph.
- **Consequences:** No published number in a v2 report or snapshot is accepted
  on its own authority. The include-graph walk is recorded as
  `PDE_INCLUDE_CLOSURE` and re-derived by a test, and two build facts are
  stated as limits rather than folded in: `_pde` links other `dp_core`
  translation units that the PDE path's include graph never reaches, and
  `CMakeLists.txt` is build definition rather than executable source, so
  neither is inventoried. The coordinated-fabrication limit of DEC-019 is
  unchanged: nothing here re-solves.
- **Authoritative links:** [pde-numerical-contract.md](pde-numerical-contract.md)
  ("Task 9C-C3: label-policy v2", "Confirmation entry is authoritative and
  semantic", "The freeze tool recomputes rather than trusts",
  "Executable-source provenance")

### DEC-022 — Task 9C-C3 evidence JSON is parsed strictly and typed exactly

- **Status:** Active
- **Context:** DEC-019 and DEC-021 made v2 verification recompute every
  decision and compare every config-derived section. Two holes remained below
  that layer, both in how a document becomes Python objects. `json.loads`
  silently keeps the last value of a duplicated object key, so a report could
  carry two `"passes"` entries and lose one before any check ran; and it
  accepts `NaN`/`Infinity`, which are not JSON. Separately, `bool` is a
  subclass of `int` in Python, so an `isinstance`-based check — or dataclass
  construction, truthiness or arithmetic — would accept a JSON `1` where a
  flag belongs.
- **Decision:** One shared strict loader serves every evidence JSON read
  (confirmation's remediation report, the freeze tool's report read, and both
  snapshot reads). It refuses duplicate object keys inside the parser at every
  depth, including objects nested in arrays, naming the key and its path, and
  refuses the non-standard constants. Separately, a centralized exact
  recursive schema validates the whole serialized document — `REPORT_SCHEMA`
  and `SNAPSHOT_SCHEMA` — **before** any `SurfaceSolve`, case object or
  confirmation decision is constructed. Every rule uses `type(value) is ...`,
  objects require their exact key set, and `null` is accepted only where a
  field is explicitly declared nullable.
- **Consequences:** Malformed evidence is refused before `run_stage`, before
  any solver call, before snapshot extraction, before snapshot writing and
  before any canonical comparison — asserted with sentinel counters that must
  stay empty. Because exact key sets leave no room for optional fields, the
  unavailable fixed-bump record now publishes its full key set with explicit
  nulls; that is a serialization-shape change only and touches no criterion,
  no threshold and no numerical result. Two schema-driven mutation tests walk
  every distinct serialized position of a report and of a snapshot and require
  an incompatible type at each to be rejected, so a future field the schema
  forgets fails a test rather than passing silently.
- **Authoritative links:** [pde-numerical-contract.md](pde-numerical-contract.md)
  ("Task 9C-C3: label-policy v2", "Evidence JSON is parsed strictly", "Exact
  recursive type validation")

### DEC-023 — v2 schemas close object keys and type every digest

- **Status:** Active
- **Context:** DEC-022 made the v2 schemas exact on *types*. Three holes
  remained on *keys and formats*. `cases[].solves[].bumped_prices` was an
  unconstrained map, so a fabricated bump key — or a key belonging to a
  different solve role — passed validation. Three
  `reported_binding_pde_*_sha256` fields were typed as unrestricted strings.
  And the exhaustive tests mutated types only, never key sets, so neither gap
  could surface.
- **Decision:** Three closures, none of which touches a criterion, a
  threshold, the price reference, the accounting or the provenance inventory.
  First, `bumped_prices` is role-exact: the static schema restricts keys to
  the six canonical spot-bump names derived from the pinned ladder, and a
  contextual validator requires the exact subset that solve's grid and role
  must carry, at the admission boundary and again immediately before
  construction. Second, every digest field — including the three named ones —
  is typed as a 64-hex digest or as a prefixed identity (`crit-`, `cases-`,
  `src-`, `pg-`, `vega-`), and a name-based audit fails if a digest-like field
  is ever typed as a plain string. Third, every `MapSpec` in both schemas is
  classified as enum-keyed, pattern-keyed or context-derived; an audit test
  fails any map that is neither statically constrained nor registered with a
  contextual validator, and **no map in either schema is free-form**.
- **Consequences:** The key-closure audit, run against the pre-fix
  `bumped_prices` spec, reports it as unbounded — the defect cannot recur
  unnoticed. Systematic add-a-key and remove-each-key sweeps now run over all
  four report lifecycle variants and all three terminal snapshot variants,
  keyed by structural discriminator so every solve role and case shape is
  covered rather than collapsed by array index. Anything the schema and
  contextual pass admit is re-checked through the full authoritative path.
- **Authoritative links:** [pde-numerical-contract.md](pde-numerical-contract.md)
  ("Task 9C-C3: label-policy v2", "Key closure: no map means \"arbitrary
  string keys\"", "Digests are typed as digests", "Systematic object-key
  closure")

### DEC-024 — three derivable v2 report maps are reconciled exactly, not merely key-bounded

- **Status:** accepted (task 9C-C3 PR 1, pre-run).
- **Context:** DEC-023 closed key *vocabularies*: every `MapSpec` became
  enum-keyed, pattern-keyed or context-derived. That still admits two things
  for a map whose exact contents are derivable — a valid-vocabulary key
  carrying a fabricated value, and the removal of a key that had to be there.
  Three report maps were in that position: `cases[].solve_problems` (a subset
  check only), `stage_outcome.failed_checks_by_name` and
  `solve_accounting.attempted_by_case_classification` (enum-keyed subsets
  whose exact contents were pinned only by the later aggregate
  recomputation, after dataclass construction).
- **Decision:** Derive the exact contents of all three from the versioned
  configuration and the still-raw solve, failure and check records, and
  reconcile them at the admission boundary — before any `SurfaceSolve` or
  other case dataclass exists, before `run_stage`, before the solver. The
  derivations are `raw_solve_problems` (over `solve_record_problems`),
  `raw_failed_checks_by_name` and `raw_attempted_by_case_classification`.
  They are not a second implementation: `evaluate_case`, `stage_outcome` and
  `solve_accounting` were refactored to call the same three functions, so one
  derivation serves both admission and recomputation. `_require_exact_map`
  compares key set *and* value at each key, so an added key, a removed key
  and a changed value are all rejected. The canonical representations are
  unchanged and are now stated explicitly: a problem-free solve, a check
  nobody failed, and a classification the stage does not cover are each
  omitted, never published as an empty list or a zero count. Gate eligibility
  for `failed_checks_by_name` is read from the configuration, never from the
  row's own `counts_toward_selection`.
- **Consequences:** `EXACTLY_DERIVABLE_MAP_PATHS` records every report map
  whose exact contents are derivable and is required to be a subset of
  `CONTEXTUAL_MAP_PATHS`; an audit test fails any such map that carries only
  a key constraint. Every map in the report schema is now exactly reconciled.
  Several existing mutation tests now fail earlier, at admission rather than
  at recomputation, with a more specific message; their expectations were
  widened to accept either, which keeps the later gate under test. No
  criterion, threshold, price reference, accounting definition, lifecycle
  rule or provenance entry changed, and the snapshot schema was left alone —
  the freeze tool already recomputes every snapshot aggregate.
- **Authoritative links:** [pde-numerical-contract.md](pde-numerical-contract.md)
  ("Task 9C-C3: label-policy v2", "Key closure: no map means \"arbitrary
  string keys\"")

### DEC-025 — Task 9C-C3 selected `grid_1600x800`, and a fresh top-level session accepted it

- **Status:** Active. Supersedes nothing; DEC-003 (v1 is frozen-terminal)
  remains in force and untouched.
- **Context:** Task 9C-B (v1) met every predeclared absolute-error cap at
  1600x800 and still returned `no_policy_selected`, failing five regular cases
  on stability and shape rather than on accuracy. Task 9C-C3 predeclared a
  revised stability/shape rule — not a loosened cap — and asked, once, whether
  it passes those five cases without destabilising cases v1 already handled.
- **Decision:** Both stages ran once, manually, against criteria fixed before
  execution. Remediation passed all ten cases; the unchanged 28-case
  confirmation set then ran once and selected **`grid_1600x800`**. The reviewed
  confirmation report
  (`f9bf3f8fd636498b09fae20df8e42e976c68d2b70b28fdff20ab93752a7e130e`) was
  frozen by the designated tool into
  [results/american_pde_label_policy_v2_results_v1.json](results/american_pde_label_policy_v2_results_v1.json)
  (`75d9402f071323065f8398ccd2cf427e2fb6fa1e663aef90ec7cbcf9c46a1186`). A
  **fresh top-level independent session**, with no anchoring on the
  implementer's reasoning, then reviewed that frozen evidence and returned
  **APPROVE POLICY AND FREEZE**. That external approval is what converts the
  selected candidate into an accepted policy; no single-session or subagent
  review could.
- **The accepted result, exactly:** accuracy policy `grid_1600x800`; price
  selected on **22/22** numerically valid regular cases; delta
  supervision-eligible on **18/22**; vega supervision-eligible on **22/22**;
  gamma **evaluation-only, 0/22** (DEC-009 unchanged);
  `criteria_were_not_loosened = true`; worst regular price error **2.6867e-4**
  against the unchanged **5e-4** cap. All **323** confirmation solves completed
  with **zero exceptions**, over **176,130** linear solves, **144,086** PSOR
  solves and **34,888,292** PSOR iterations, in approximately **1,586.4 s** of
  wall time. One descriptive stress failure, `stress_euro_short_low_vol_atm`,
  price error **2.9943e-3** with a correspondingly large evaluation-only gamma
  error; **stress cases do not decide selection** and this one decided nothing.
- **Two things this decision explicitly does not grant.** First,
  `regular_american_put_negative_rate_control` passes its negative
  American-dominance gap **only** through the predeclared residual-scale-aware
  operational allowance, which is a price-error scale estimate from the
  solver's own accumulated LCP residual and **is not a rigorous error bound**
  (`is_a_rigorous_bound = false`, DEC-016). Second, **no dataset and no
  training input is authorized by the frozen report itself**
  (`authorizes_dataset_generation = false`,
  `authorizes_training_input = false`, `AUTHORIZED_TRAINING_INPUT_STATUSES`
  empty). An accepted label policy is not an accepted dataset; task 9C-C2b2 and
  its pilot are separately gated, and DEC-014 still binds.
- **Consequences:** The label-policy blocker recorded since DEC-003 is cleared
  for **price, delta (where eligible) and vega (where eligible)** only. Task
  9C-C2b2 — deterministic parallel/resumable generation infrastructure — becomes
  the next technical milestone, at infrastructure and bounded-pilot scope. v1
  remains immutable historical `no_policy_selected` evidence: v2 answered a new,
  separately predeclared question on a separately versioned config and runner
  and **does not reinterpret v1** (DEC-003).
- **Authoritative links:** [pde-numerical-contract.md](pde-numerical-contract.md)
  ("Task 9C-C3: label-policy v2", "Outcome: the accepted v2 result"),
  [results/american_pde_label_policy_v2_results_v1.json](results/american_pde_label_policy_v2_results_v1.json),
  `python/tests/test_pde_label_policy_v2_results_snapshot.py`

### DEC-026 — 18/22 delta eligibility is order measurability, not four inaccurate deltas

- **Status:** Active.
- **Context:** DEC-025 grants delta supervision eligibility on 18 of 22 regular
  cases. Read carelessly, "4 of 22 excluded" invites the conclusion that the
  policy produces four bad deltas, and then invites widening the order band to
  "recover" them. Both readings are wrong, and the second would be a
  post-result criterion change.
- **Decision:** Record the exclusions as what the snapshot actually says. All
  four excluded cases carry `grid_stencil_delta_validation_passed` — the
  stencil delta was inside the unchanged 1e-3 cap — and were excluded **solely**
  by `grid_stencil_observed_order_unsupported`:
  - `regular_american_put_high_rate` — stencil delta exactly $-1$, so the
    factor-two order is **undefined**;
  - `regular_american_high_carry_call` — stencil delta exactly $+1$, order
    likewise undefined;
  - `regular_euro_deep_otm_call_short_low_vol` — observed order approximately
    $23.75$;
  - `regular_american_one_dividend_call` — observed order approximately
    $3.027$, **despite** an E2 error of approximately $9.71 \times 10^{-7}$,
    well below the 1e-3 cap.
  The first two are the saturated ends of a delta, where an exactly flat
  stencil leaves no increment to measure an order from. Declining to supervise a
  delta whose convergence order could not be measured is the conservative
  direction and was predeclared.
- **Consequences:** The supported order band is **not** widened and eligibility
  is **not** changed. Any reconsideration — of the band, of the flat-delta case,
  or of these four cases — requires a **separately versioned task and config**,
  exactly as 9C-C3 was separately versioned from 9C-B. It is never a revision
  inside 9C-C3, and the four cases are already-consumed one-shot evidence.
- **Authoritative links:** [pde-numerical-contract.md](pde-numerical-contract.md)
  ("Task 9C-C3: label-policy v2", "F1: what 18/22 delta eligibility means"),
  DEC-018, DEC-025

### DEC-027 — The v2 snapshot is enforced frozen evidence; its pending-approval field stays historical

- **Status:** Active.
- **Context:** `scripts/freeze_pde_label_policy_v2_results.py` was deliberately
  left out of `scripts/check.sh` and CI at predeclaration time, because
  `--check` fails when the snapshot it enforces is absent. A snapshot now
  exists. Separately, the frozen snapshot records
  `selection_pending_fresh_top_level_approval = true`, which is no longer the
  project's state — the approval has since happened — and that mismatch is an
  obvious temptation to "correct" the file.
- **Decision:** Two rules. First, `--check` is now the **designated** validator
  and runs in `scripts/check.sh` and in the `python` CI job, **once each**; it
  reads only checked-in files and never the ignored raw report, so it passes
  with `artifacts/` absent.
  `python/tests/test_pde_label_policy_v2_results_snapshot.py` pins the snapshot
  digest and every conclusion alongside it. Second,
  `selection_pending_fresh_top_level_approval = true` is **historical and
  correct**: the report was generated before the external approval existed and
  accurately records the state at generation time. Completion of that approval
  is recorded by DEC-025, which post-dates the snapshot. **Neither the report
  nor the snapshot is ever edited to flip that field**, or reformatted,
  regenerated or refreshed for any other reason.
- **Known technical debt, recorded not fixed:** the fixed-bump check
  `ladder_within_residual_scale` is **dead** — in the `flat` branch its
  condition is exactly the branch predicate, and in the `resolved` branch it is
  literally `True` — so it can never fail and never contributed to any verdict.
  It is a harmless reporting field. It is not removed here, because the runner
  is in the frozen snapshot's executable-source inventory and editing it would
  break the provenance reconciliation of terminal evidence. Any cleanup belongs
  to a later, separately versioned study.
- **Known limit, recorded not fixed:** the snapshot links the consumed
  remediation report only indirectly, via the confirmation report's digest and
  the shared `criteria_digest` / `raw_config_sha256` the confirmation stage
  required to match. A future schema version may store the consumed
  remediation-report **content digest** directly. That is a forward-looking
  improvement for a later study, not a defect in this result and not a reason to
  regenerate this snapshot.
- **Authoritative links:** [pde-numerical-contract.md](pde-numerical-contract.md)
  ("Task 9C-C3: label-policy v2", "The snapshot is immutable, and enforced",
  "Future provenance hardening"), [architecture.md](architecture.md), DEC-003,
  DEC-025

### DEC-028 — The stage-2 roadmap is locked to CRR learnability, then XSP/SPY, then a deferred commodity extension

- **Status:** Active.
- **Context:** After DEC-025 accepted `grid_1600x800`, the project's stated
  next step was infrastructure (task 9C-C2b2) leading to a PDE-labelled
  American dataset. That ordering answered "how do we generate labels at
  scale?" before the project had answered "is an American surrogate worth
  generating labels for?" The research question the project actually exists to
  answer is a **speed** question, and the cheapest evidence for it does not
  need new label generation at all.
- **Decision:** Lock the stage-2 roadmap, in this order, as a new appended
  section of [research-contract.md](research-contract.md) ("The locked stage-2
  roadmap"):
  1. **Continuous-dividend-yield American CRR baseline** — the first
     learnability experiment: American-price learnability, scratch versus
     European-transfer training (H2), inference accuracy, inference scaling
     against CRR's $O(N^2)$ lattice cost, and a small implied-volatility
     inversion and surface reconstruction.
  2. **XSP/SPY real-instrument study** — XSP (European, cash-settled) as the
     control, SPY (American, discrete deterministic cash distributions, early
     exercise) as the target; learn PDE prices for contracts grounded in the
     available market universe, compare inference against the PDE, and
     reproduce and evaluate real implied-volatility surfaces. The existing PDE
     solver and the accepted v2 label policy remain valuable inputs here.
  3. **Deferred commodity extension** — corn options are the leading future
     candidate, because American exercise into futures, seasonality and the
     futures curve give a genuinely different cross-asset test. Not designed,
     not scoped for data acquisition, not implemented now.
  **Crypto is explicitly excluded** from the active roadmap: the liquid crypto
  option universe is European-only and therefore does not advance the
  American-option question. That is a scope decision, not a numerical one.
- **The central question this locks the roadmap to:** can a neural surrogate
  price American options with useful accuracy while delivering materially
  faster inference than the numerical method that generated its labels, and
  does that speedup support faster implied-volatility inversion and
  volatility-surface construction? Latency claims stay governed by
  [research-contract.md](research-contract.md) "Metrics" — end-to-end, against
  the label-generating method, under matched conditions.
- **Consequences:** Phase ordering is not to be relitigated task by task. A
  later task may report evidence that a phase is misordered; it may not
  quietly reorder them. **The accepted PDE label policy remains frozen
  evidence; this roadmap change does not reinterpret or rerun it** (DEC-025,
  DEC-026, DEC-027). This decision changed documentation only: no code, test,
  config, script, frozen result, figure, or data file was touched, and no
  numerical run was performed.
- **Authoritative links:** [research-contract.md](research-contract.md) ("The
  locked stage-2 roadmap"), [project-state.md](project-state.md), DEC-029,
  DEC-030

### DEC-029 — Task 9C-C2b2 is deferred, not deleted and not rejected

- **Status:** Active.
- **Context:** Task 9C-C2b2 (deterministic parallel/resumable PDE-label
  generation) was the active task under the pre-DEC-028 ordering. Under the
  locked roadmap, dataset-scale PDE generation is not needed until the
  XSP/SPY phase. Building it now would be infrastructure ahead of the
  experiment that justifies it — the same failure mode DEC-012 already ruled
  against in the small.
- **Decision:** Task 9C-C2b2 is **deferred**. Its specification stays at
  [tasks/active/task-9c-c2b2-parallel-resumable-generation.md](tasks/active/task-9c-c2b2-parallel-resumable-generation.md)
  — the repository marks task status in place and has no `tasks/completed/` or
  `tasks/deferred/` location — with `Status: Deferred` and the resumption
  condition stated in the file. Nothing in it is rejected: its objective,
  gates, `OPEN` conventions and stop conditions remain the specification to
  resume from.
- **Resumption condition:** the XSP/SPY phase requires dataset-scale PDE
  generation. Until then, no work proceeds on it.
- **Consequences:** Deferring it does not weaken anything it was protecting.
  Grouped partitioning (DEC-005), gamma's evaluation-only status (DEC-009),
  the empty `AUTHORIZED_TRAINING_INPUT_STATUSES` (DEC-011), and the rule that
  an accepted policy is not an accepted dataset (DEC-014, DEC-025) all still
  bind. In particular, this deferral is **not** a licence for some other task
  to generate PDE labels at scale without deterministic, resumable,
  leakage-safe machinery.
- **Authoritative links:**
  [tasks/active/task-9c-c2b2-parallel-resumable-generation.md](tasks/active/task-9c-c2b2-parallel-resumable-generation.md),
  DEC-012, DEC-014, DEC-025, DEC-028

### DEC-030 — The local candidate CRR dataset is audited before it is admitted, and task 9D is the next active task

- **Status:** Active.
- **Context:** Phase 1 of the locked roadmap (DEC-028) runs against a
  continuous-dividend-yield American CRR dataset. A candidate exists locally
  under Git-ignored `data/`, reportedly about 250,000 rows. Nothing about it
  has been established by this repository's evidence rules: it is untracked,
  its provenance is not catalogued here, and its generator does not exist on
  the current branch or on `main`.
- **Decision:** **A local candidate CRR dataset reportedly exists, but it has
  not yet been catalogued, validated, reproducibly admitted, or accepted as
  project evidence.** It is not admitted, not transformed, not regenerated,
  and not trained on until a documentation and integrity audit — **task 9D**,
  [tasks/active/task-9d-data-holdings-audit.md](tasks/active/task-9d-data-holdings-audit.md)
  — has catalogued it. Task 9D is the single active task and is
  documentation-and-audit scope only.
- **What task 9D covers:** the existing CRR dataset; the existing
  Databento/FRED market-data holdings; schemas, formats, partitions, sizes,
  row/record counts, provenance, hashes where appropriate, tracking/ignore
  status, and known limitations; and a **deferred plan** for private S3
  storage/retrieval and entitlement-aware Databento ingestion.
- **What task 9D must not do:** upload data, implement S3 access, call
  Databento, regenerate CRR labels, modify the ML loader, or train a network.
- **Two wordings that must not drift:** a continuous dividend yield is not a
  discrete cash-distribution schedule, so the phase-1 CRR dataset **cannot
  model SPY cash dividends** and must never be described as a no-dividend
  dataset (DEC-001). And **cloud storage and vendor ingestion are deferred
  reproducibility work, not prerequisites for the first CRR learnability
  experiment** — task 9D plans them, it does not build them.
- **Consequences:** No accepted, versioned PDE-labelled SPY training dataset
  exists, and no American neural surrogate has yet been trained and accepted.
  Whether the candidate dataset is admitted at all is task 9D's finding to
  report and a later, separately gated decision to make; a negative audit
  finding is a valid outcome that stops phase-1 training until it is resolved.
- **Authoritative links:**
  [tasks/active/task-9d-data-holdings-audit.md](tasks/active/task-9d-data-holdings-audit.md),
  [research-contract.md](research-contract.md) ("The locked stage-2 roadmap",
  "Non-claims"), DEC-001, DEC-011, DEC-015, DEC-028

### DEC-031 — Task 9D catalogued the local data holdings; the catalogue admits nothing

- **Status:** Active.
- **Context:** DEC-030 routed the single active task to a documentation and
  integrity audit of the existing local data holdings, because phase 1 of the
  locked roadmap (DEC-028) needs an input and the only candidate was an
  uncatalogued local dataset.
- **Decision:** The audit is complete and its written output is
  [data-holdings-catalogue.md](data-holdings-catalogue.md). Every holding under
  `data/` is catalogued; every material fact is marked either **[check]**
  (recomputed in the audit from local bytes or Git objects) or **[claim]**
  (copied from metadata or an existing report and not independently verified).
  A claim is never promoted to a check by restatement. The catalogue is an
  audit record at one point in time: it is **not** a normative contract, **not**
  frozen evidence, has no generator script, and is superseded by re-auditing
  rather than regeneration.
- **What the audit established, positively:** the candidate CRR dataset's
  internal identities all hold (label averaging, early-exercise premium,
  adjacent-step gap, log-moneyness, American dominance — exact; intrinsic
  dominance exact to rounding); every manifest claim reconciles against the
  bytes, including all per-partition, per-stratum and label-diagnostic counts;
  no nulls and no non-finite values anywhere; full domain and per-stratum bound
  compliance; no identifier and no exact contract state shared between
  partitions; and the manifest's composite `crr_implementation_sha256`
  **reproduces exactly from this branch's tracked C++ and binding sources**, so
  the labelling oracle is byte-identical to the CRR engine at HEAD. The market
  archive verifies 119 of 119 checksummed files with zero mismatches.
- **What the audit established, negatively:** the CRR dataset's generator,
  configuration and tests are **not on this branch and not on `main`** — they
  exist only on the unmerged branch `feat/american-dataset-v1`, which is not an
  ancestor of `main`; its label policy has **no frozen `docs/results/`
  snapshot** on any branch and its governing contract section exists only on
  that other branch; its Parquet files carry no embedded provenance; and the
  existing ML loader rejects its schema by design. In the processed market
  partitions, `exercise_style` and `contract_multiplier` are **100 % null**, so
  the XSP-European / SPY-American distinction phase 2 depends on is an external
  convention, not something these bytes carry.
- **Consequences:** **The catalogue admits nothing.** Cataloguing a holding is
  not accepting it; `AUTHORIZED_TRAINING_INPUT_STATUSES` stays empty (DEC-011);
  and **no accepted, versioned American training dataset exists** in any class.
  Twelve unresolved limitations are recorded in the catalogue's section 8 and
  are inherited by task 9E (DEC-032). The audit modified no data, ran no
  numerical study, called no vendor API, and changed no source code.
- **Review status, recorded not glossed:** the `numerical-reviewer` the task
  spec requires was not invoked in the session that produced the catalogue, and
  no fresh top-level session has materially accepted its findings. `Completed`
  means the exit gates are discharged and the deliverables exist; it does not
  mean independently approved. Task 9E's admission decision is where that
  approval becomes load-bearing.
- **Authoritative links:** [data-holdings-catalogue.md](data-holdings-catalogue.md),
  [tasks/active/task-9d-data-holdings-audit.md](tasks/active/task-9d-data-holdings-audit.md),
  DEC-015, DEC-028, DEC-030, DEC-032

### DEC-032 — Task 9E, CRR dataset admission, is the next active task; task 9F is on hold

- **Status:** Active.
- **Context:** Task 9D established what the candidate CRR dataset is and what
  about it is still open (DEC-031). The next bounded step is neither training
  nor regeneration: it is deciding whether the dataset is fit for phase 1 and
  making this repository able to read it under an explicit schema contract.
  Separately, the deferred half of 9D — remote storage and vendor ingestion —
  needed somewhere to live without becoming work.
- **Decision:** Two task routings.
  1. **Task 9E — CRR dataset admission** is the single active task:
     [tasks/active/task-9e-crr-dataset-admission.md](tasks/active/task-9e-crr-dataset-admission.md).
     Its scope is exactly three things plus the decision they support:
     **semantic suitability** (is this a useful learnability probe, not merely
     a valid table, including one small predeclared cross-check against a
     numerically unrelated engine — the first external check this dataset has
     had); **named-schema loader support** (a dataset is read because its
     schema is named and versioned, never because a file happens to parse; the
     European path is unchanged and unknown schemas still fail closed); and
     **integrity and leakage gates** wired into the test suite, with the
     near-duplicate threshold predeclared and versioned **before** it is
     evaluated. Provenance decisions on `feat/american-dataset-v1` and on the
     label policy's missing frozen evidence are recorded there too.
  2. **Task 9F — private object storage and entitlement-aware vendor
     ingestion** is `On hold`:
     [tasks/active/task-9f-remote-data-access-plan.md](tasks/active/task-9f-remote-data-access-plan.md).
     It is a plan, not work. Its every predeclared convention is `OPEN` and
     none was invented.
- **Explicitly excluded from task 9E:** training any network, and regenerating,
  transforming, moving or deleting any dataset. Both are separate gates. Phase
  1's training run does not begin at 9E's exit; 9E authorizes the dataset as a
  phase-1 input and nothing more, and only a fresh top-level session can grant
  even that.
- **The redistribution boundary, recorded once so 9F cannot drift:** licensed
  OPRA and Databento content may not be redistributed publicly. "Reproducible"
  in 9F means reproducible by an authorized user who independently holds the
  required vendor entitlements — never downloadable by anyone with the
  repository. **Cloud storage and vendor ingestion are deferred reproducibility
  work, not prerequisites for the first CRR learnability experiment.**
- **Consequences:** The locked three-phase roadmap (DEC-028) is unchanged; this
  entry routes tasks within phase 1, it does not reorder phases. Task 9C-C2b2
  stays `Deferred` (DEC-029) and task 9C-C3 stays `Completed` and terminal.
  **The accepted PDE label policy remains frozen evidence; neither 9E nor 9F
  reinterprets or reruns it.**
- **Authoritative links:**
  [tasks/active/task-9e-crr-dataset-admission.md](tasks/active/task-9e-crr-dataset-admission.md),
  [tasks/active/task-9f-remote-data-access-plan.md](tasks/active/task-9f-remote-data-access-plan.md),
  [data-holdings-catalogue.md](data-holdings-catalogue.md), DEC-011, DEC-014,
  DEC-015, DEC-028, DEC-030, DEC-031

### DEC-033 — The CRR dataset is admitted for one bounded experiment, and for nothing else

- **Status:** Active.
- **Context:** DEC-032 routed the active task to 9E, which had to answer one
  question: is `data/american-option-v1/` a valid, reproducible input to a
  bounded American CRR-network experiment? Task 9D had catalogued it and left
  the loader unable to read its schema, its generator on an unmerged branch, and
  its label policy without frozen evidence.
- **Decision:** **Admitted, solely for the bounded continuous-yield American CRR
  learnability and latency experiment**, under the schema
  `american-option-dataset/1` and the representation `american_raw_physical_v1`.
  The record is
  [american-crr-dataset-admission.md](american-crr-dataset-admission.md). Every
  gate in `python/src/differentiable_pricing/data/american_admission.py` ran
  over all 250,000 rows and passed; no gate was weakened and no row was dropped.
  **The admission is a single-session conclusion and its material approval is
  outstanding**, so task 9E is `Implemented, pending fresh review`, not
  `Completed`.
- **What admission does not grant:** it authorizes **no training** — the first
  American training run is a separate task with its own gate, which this session
  did not define, start, or route to. It grants no SPY or PDE training, no
  production label-quality claim, no certified Greek, and no discrete-dividend
  coverage. **A continuous dividend yield is not a discrete SPY cash-dividend
  schedule**, and this schema cannot represent one.
- **Provenance recovered, and its limit.** Read-only Git-object inspection of
  `feat/american-dataset-v1` at `49ef72a` — no merge, no cherry-pick, no
  regenerated row. `configs/american_option_dataset_v1.toml` and both
  label-policy pilot configurations were ported **verbatim** and are digest-
  pinned by a test, so the manifest's config link now verifies from tracked
  sources; the "Label policy v1" section was recovered into
  [american-crr-contract.md](american-crr-contract.md); and the schema contract
  was ported as `data/american_schema.py`. **The generation machinery was
  deliberately not ported**, so the dataset **still cannot be regenerated from
  tracked sources alone**. The manifest's composite `crr_implementation_sha256`
  reproduces exactly from this branch's C++ and binding sources, so the
  labelling oracle is byte-identical to the CRR engine at HEAD.
- **The label policy's evidence status, recorded not glossed:** the two pilot
  reports that selected `steps = 1024` were **ignored local artifacts under
  `artifacts/`, never frozen `docs/results/` snapshots**, and they remain so. No
  snapshot for this label policy exists on any branch and task 9E invented none.
  The recovered contract section states that pilot v2 was written after v1's
  results were seen and is **not an uncontaminated predeclaration**, and that
  re-scored under v1's original p99 gate the selection would have been
  `N = 4096`. That is carried forward as a property of the policy.
- **Feature sufficiency:** the dataset carries every state variable a
  constant-parameter American CRR price depends on. `raw_physical_v1`'s seven
  raw features suffice; the European **`forward_normalized_v1` representation
  does not** and is recorded as rejected. Measured with the CRR engine, two
  contracts sharing an option type, `log(F/K)` and `sigma*sqrt(T)` but differing
  in rate and dividend yield agree to machine precision in normalized European
  price and differ by **1.2–1.3 %** in normalized American price. The minimal
  versioned American representation is therefore `american_raw_physical_v1`.
- **Loader:** datasets are now selected by **explicit schema name and version**
  from a registry of two, with exact per-version schema validation, mandatory
  manifest-digest verification, and **no permissive compatibility fallback**;
  an unregistered `schema_version` fails before any file is opened. The
  **European path is unchanged** and its tests pass untouched. The American
  target and its paired European comparator are exposed by name.
- **Two gates deliberately not claimed.** The **near-duplicate** leakage
  threshold was not predeclared — task 9D measured those distances first, so any
  threshold now would be post-hoc; the measurement stays descriptive and no gate
  was built from it. The **independent cross-check** against a numerically
  unrelated engine was scoped out by the implementing prompt and **was not
  run**; every identity verified so far is internal to one lattice, and that
  cross-check remains open work.
- **Consequences:** phase 1 of the locked roadmap (DEC-028) has an input, and
  only that. The locked three-phase roadmap is unchanged. Task 9C-C2b2 stays
  `Deferred`, task 9C-C3 stays `Completed` and terminal, and task 9F stays
  `On hold`. **The accepted PDE label policy remains frozen evidence; this task
  does not reinterpret or rerun it.** No network was trained, no inference was
  benchmarked, no implied volatility was reconstructed, and no dataset was
  regenerated or modified.
- **Authoritative links:**
  [american-crr-dataset-admission.md](american-crr-dataset-admission.md),
  [american-crr-contract.md](american-crr-contract.md) ("Label policy v1"),
  [tasks/active/task-9e-crr-dataset-admission.md](tasks/active/task-9e-crr-dataset-admission.md),
  [data-holdings-catalogue.md](data-holdings-catalogue.md), DEC-001, DEC-011,
  DEC-028, DEC-031, DEC-032

### DEC-034 — Task 9E is conditional, and task 9G is a bounded feasibility pilot

- **Status:** Active. Supersedes DEC-033 only where DEC-033 describes task 9E
  as an all-gates admission or calls `american_raw_physical_v1` minimal;
  DEC-033 remains append-only history and all of its recorded measurements and
  limitations stand.
- **Context:** Reconciliation against task 9E's predeclared specification found
  that the independent LSM/PDE price cross-check and semantic-coverage
  judgement were not performed, and that the near-duplicate threshold was not
  predeclared before task 9D observed the distances. It also found that the
  implemented gates do not bind each row's `label_policy` and `label_steps` to
  `manifest.label_policy.name` and `.steps`. Separately, the seven raw American
  inputs are feature-sufficient but not a minimal representation, and the
  frozen European source weights are recorded in evidence but not tracked.
- **Admission decision:** The dataset is **conditionally admitted only for
  learning the known continuous-yield CRR mapping**. This is not acceptance of
  converged American-price accuracy and authorizes no training by itself. The
  independent numerical cross-check remains outstanding and must be completed
  or explicitly resolved before training is authorized. The near-duplicate
  measurements remain descriptive: task 9E's predeclared gate cannot be
  satisfied retroactively for this dataset version, and no threshold is
  invented now. Material acceptance of the reconciled conditional admission
  still requires fresh top-level review.
- **Dataset entry invariant:** Before training, implementation and tests must
  require every row's `label_policy == manifest.label_policy.name` and
  `label_steps == manifest.label_policy.steps`. Experiment entry checks also
  pin generator version `1.0.0` and recovered configuration SHA-256
  `d18485c66b92c720c57bef6820e7f6cdb7204159c2dcf8d47d8f9c744cb28c98`.
- **Representation:** The phase-1 network uses
  `american_forward_carry_v1`: encoded option type, `log(F/K)`,
  `sigma*sqrt(T)`, `rT`, and `qT`, with target
  `V/(S*exp(-q*T))` and physical reconstruction by multiplying by
  `S*exp(-q*T)`. It is sufficient because
  `log(S/K) = log(F/K) - rT + qT`. `american_raw_physical_v1` remains
  feature-sufficient but is not minimal. The measured rejection of the
  three-input European representation stands; this decision extends it with
  the two missing carry coordinates rather than reinterpreting it.
- **Transfer lift and source gate:** Task 9G copies the three shared first-layer
  columns of the frozen three-input European network, adds zero first-layer
  columns for `rT` and `qT`, copies all later weights and biases, and preserves
  or algebraically rebases standardization so the unconstrained physical
  function is identical. It uses unconstrained weights, not the European bounds
  projection as an American constraint, and verifies the physical identity at
  fixed probes to float64 numerical precision before fine-tuning. An impossible
  exact lift stops the experiment; no general framework is built around it.
  The original `weights.npz` must be recovered and match frozen SHA-256
  `42670774f736383e50818b6e6c1db9374a77988173e35423ffc34b3c4297ecb8`.
  Missing or mismatching weights stop the transfer arm; silent retraining or
  substitution is forbidden.
- **Pilot boundary:** Task 9G is one architecture, one scratch run, one transfer
  run, one budget, validation checkpoint selection, and one-shot
  `interpolation_test`, with no sweep. It reports the paired European CRR price
  as a no-learning baseline and positive early-exercise-premium rows
  separately; benchmarks the actual adjacent-average operation
  `0.5*(CRR(N)+CRR(N+1))` over the fixed depth ladder; and performs one small
  model-consistent IV experiment. Negative transfer and failure to learn are
  valid. One seed and one budget cannot establish H2; a promising pilot leads
  to a separately predeclared replication with at least five seeds and several
  label budgets.
- **Latency and IV language:** Reference and neural timing match request shapes,
  batch sizes, thread budgets, warm-ups, repetitions, and recorded
  hardware/software metadata. Neural timing includes physical feature
  transformation and output reconstruction. One maturity is an IV smile slice;
  multiple maturities are required for an IV surface. Synthetic phase 1 makes
  no bid--ask-relative market claim.
- **Lifecycle:** The required order is planning-reconciliation PR;
  protocol-and-implementation PR reviewed and merged before execution; manual
  locked run; result-only PR with gates, seeds, training code, and evaluation
  code unchanged; fresh review of the result. A branch first reviewed only
  after results is not the predeclaration mechanism.
- **Consequences:** Task 9G is the exact next task, at protocol-and-
  implementation scope. No experiment was run by this decision. The locked
  CRR → XSP/SPY → deferred commodity roadmap is unchanged; task 9C-C2b2 remains
  deferred, task 9F remains on hold, and the accepted PDE policy and all frozen
  evidence remain untouched.
- **Authoritative links:**
  [american-crr-dataset-admission.md](american-crr-dataset-admission.md),
  [architecture.md](architecture.md) ("Phase-1 American surrogate
  representation"), [research-contract.md](research-contract.md) ("Phase-1
  feasibility pilot versus an H2 replication"),
  [tasks/active/task-9g-american-neural-pricer-pilot.md](tasks/active/task-9g-american-neural-pricer-pilot.md),
  DEC-028, DEC-032, DEC-033

### DEC-035 — Fresh review approves only the task 9E conditional planning reconciliation

- **Status:** Active. This records review state only; it does not change
  DEC-034's experiment scope, representation, roadmap, thresholds, or workflow.
- **Verdict:** The fresh independent reviewer returned exactly
  `APPROVE PLANNING RECONCILIATION`, with no BLOCKER, MAJOR, MINOR, or NOTE
  findings.
- **Approved boundary:** The review approves task 9E's reconciled conditional
  admission solely for learning the known continuous-yield CRR mapping. It
  discharges the fresh-review gate for that mapping-only planning
  reconciliation.
- **What the approval does not do:** It does not establish converged
  American-price accuracy; authorize training; satisfy or waive the independent
  numerical cross-check; satisfy or waive the semantic-coverage judgement;
  retroactively satisfy the near-duplicate gate; or approve a neural-pricer
  result.
- **Consequences:** Task 9G's protocol-and-implementation PR is the exact next
  task. Before training, the row/manifest policy invariants and dataset and
  generator/config identity pins must be implemented; the independent
  cross-check must be completed or explicitly resolved under fresh review; the
  European source weights must be recovered and digest-verified; the exact
  transfer lift must be verified; and seeds, training budget, metrics, latency
  shapes, and IV cases must be frozen. No experiment is authorized by this
  decision.
- **Authoritative links:**
  [american-crr-dataset-admission.md](american-crr-dataset-admission.md),
  [project-state.md](project-state.md),
  [tasks/active/task-9e-crr-dataset-admission.md](tasks/active/task-9e-crr-dataset-admission.md),
  [tasks/active/task-9g-american-neural-pricer-pilot.md](tasks/active/task-9g-american-neural-pricer-pilot.md),
  DEC-033, DEC-034

### DEC-036 — Lock Task 9G's bounded protocol and implementation before execution

- **Status:** Active. Implements DEC-034; it does not execute or interpret the
  pilot.
- **Source and dataset gates:** The ignored unconstrained European manifest and
  weights were recovered and preflighted at SHA-256 `054ca945…` and
  `42670774…`; the existing loader reconstructed the expected float64
  `3 -> [64,64,64] -> 1` tanh model. The master protocol pins schema
  `american-option-dataset/1`, generator `1.0.0`, configuration `d18485c6…`,
  manifest `25285892…`, train `ff7a114f…`, validation `6f53a72e…`, and locked
  final `f006a17c…`. Runtime entry checks open and hash train and validation
  only; the final file remains unavailable before the atomic one-shot
  reservation.
- **Admission invariant closed:** every loaded row must now satisfy
  `label_policy == manifest.label_policy.name` and
  `label_steps == manifest.label_policy.steps`. Focused fixtures cover pass,
  name mismatch, step mismatch, and mixed-column mismatch. Existing European
  and American named-schema loading is unchanged.
- **Representation and lift:** `american_forward_carry_v1` is implemented only
  in task-specific code, with target and physical reconstruction inside the
  PyTorch graph and no output projection. American standardization is fitted on
  the selected train rows only. The task-specific lift rebases the three shared
  first-layer coordinates and output layer algebraically, sets the `rT` and
  `qT` columns exactly to zero, and copies all other parameters. Against the
  recovered source and actual locked train-only scaling, eight physical probes
  passed `rtol=atol=1e-12`; maximum absolute price difference was
  `1.4210854715202004e-14`, maximum relative difference
  `3.6214823824845716e-13`.
- **Fixed pilot design:** exactly scratch seed `2909056561` and transfer seed
  `2009073353`, each publicly derived as the first four big-endian bytes of
  SHA-256 over its stable label; 32,768 deterministic train rows; one
  `5 -> [64,64,64] -> 1` float64 CPU model per arm; price-only AdamW at
  `1e-3` with `1e-6` weight decay and a 120-epoch cosine schedule; batch size
  2,048; four threads; no early stopping; minimum validation standardized-target
  MSE checkpoint with earliest exact tie. There is no sweep, recovery run, or
  extra arm.
- **Independent check:** the existing PDE is a valid unrelated comparator under
  its normative equation by setting `continuous_carry=q`, a flat rate curve,
  and an explicitly empty cash-dividend schedule. One validation row per
  `(stratum, option_type)` is selected by the lowest salted SHA-256 of
  `sample_id`, using identifiers and physical inputs only. The 21 selected IDs
  and inputs are pinned before execution. Coarse `400x200` and fine `800x400`
  grids use a four-times spot/strike domain; normalized refinement and fine-PDE
  versus CRR differences must be at most `5e-4` and `1e-3`. This is only a
  mapping-consistency check. The refinement pair changes resolution while
  holding `spot_maximum` fixed, so it does not independently bound domain-
  truncation error. It has not run.
- **Evidence and lifecycle:** physical and normalized MAE/RMSE/p95/p99/max are
  locked overall and by option type, expiry, moneyness, volatility, premium,
  and exercise status. Bound and local monotonicity/convexity diagnostics make
  no Greek claim. The no-learning baseline is stored `european_crr_price`.
  Latency uses actual adjacent averages at `[256,512,1024,2048,4096]`, matched
  single/batch-eight requests, thread budgets, two warm-ups and seven
  repetitions. Six multi-maturity synthetic cases use one safeguarded bisection
  rule and are called an IV surface. The runner exposes only
  `run-to-validation`, `status`, and confirmed `final-evaluate`; an atomic marker
  consumes the sole final attempt before the final partition is touched. A
  caught error or interruption after reservation writes a strict consumed-
  failure report, records it in the ledger, forbids retry, and remains
  freezable without manufacturing final metrics.
- **Pilot confound:** scratch and transfer retain their locked, distinct arm
  seeds. Those seeds control epoch permutations as well as initialization, so
  this one-seed feasibility pilot cannot attribute an observed arm difference
  solely to transfer initialization.
- **Artifacts and freezing:** the new `american-neural-artifact/1` schema pins
  scaling, representation, architecture, dataset/protocol/code identity,
  deterministic NPZ weights, and transfer lineage. The designated raw-report
  validator and snapshot freeze/check tool reject schema, finiteness, digest,
  lifecycle, and recomputation defects. The compact validation/final partition
  audit retains only per-model/per-slice sufficient statistics needed for
  offline recomputation; that audit does not copy sample IDs, economic inputs,
  labels, or predictions from the ignored dataset/report. Separately, the
  independent PDE entry-check block deliberately retains its 21 protocol-pinned
  identifiers, synthetic economic inputs, CRR labels, and solve evidence. Those
  synthetic cases create no market- or proprietary-data claim. Offline
  `--check` detects internal inconsistency and tracked-input drift, but cannot
  authenticate a fully coordinated fabricated raw report and snapshot. No Task
  9G result snapshot or figure is created by this implementation change.
- **Execution state and next action:** no PDE cross-check, optimization,
  training, latency study, IV inversion, or final evaluation ran. Same-session
  preliminary code and numerical reviews were used to resolve implementation
  findings; they are not approval. The next action is a fresh top-level review
  and merge. Only then may a human invoke the locked `run-to-validation`
  command. One seed and one budget cannot establish H2.
- **Authoritative links:**
  `configs/american_neural_pilot_protocol_v1.toml`,
  [tasks/active/task-9g-american-neural-pricer-pilot.md](tasks/active/task-9g-american-neural-pricer-pilot.md),
  [architecture.md](architecture.md), [research-contract.md](research-contract.md),
  DEC-034, DEC-035

### DEC-037 — Fresh review approves Task 9G's protocol and implementation for merge

- **Status:** Active. This is an append-only implementation-review record; it
  approves no numerical result and changes no locked experimental value.
- **Verdict and reviewed identity:** A fresh top-level reviewer inspected the
  complete cumulative Task 9G branch at commit `8d27c23` and returned exactly
  `APPROVE TASK 9G IMPLEMENTATION FOR MERGE`.
- **Approved boundary:** The verdict approves the Task 9G protocol and
  implementation for merge and, after merge, the subsequent human invocation
  of the locked `python scripts/run_american_neural_pilot.py
  run-to-validation` command. It does not itself authorize execution before
  merge, approve any result, or authorize `final-evaluate`.
- **Minor observations before final cumulative review:** The three
  non-blocking MINOR observations were corrected after the reviewed commit and
  before final cumulative review: the duplicated approval wording in project
  state was removed; compact-statistic mutation coverage now uses distinct,
  non-zero, multi-slice errors and directly mutates squared sums, maxima,
  quantile interpolation and ranks; and the 25,000-validation plus 25,000-final
  size fixture now populates both option types and every expiry, moneyness,
  volatility, premium and exercise-status slice.
- **Execution state:** No PDE cross-check, training, latency benchmark, IV
  inversion, final evaluation or result approval occurred. No numerical result
  exists. The approval-recording/minor-fix commit requires final cumulative
  review and the implementation branch must be merged before the human locked
  validation run.
- **Non-claims preserved:** This approval establishes no H2 result, converged-
  price accuracy, American Greek accuracy, market or bid--ask performance, OOD
  behavior, discrete-dividend applicability, or portable-latency conclusion.
  `final-evaluate` remains separately gated on the validation outcome and its
  required review.
- **Authoritative links:**
  [project-state.md](project-state.md),
  [tasks/active/task-9g-american-neural-pricer-pilot.md](tasks/active/task-9g-american-neural-pricer-pilot.md),
  DEC-034, DEC-035, DEC-036

### DEC-038 — Task 9G's locked run failed its validation gates: frozen `failure_to_learn`

- **Status:** Frozen-terminal for task 9G's numerical result. This is an
  append-only result record; it changes no locked experimental value, no
  threshold, no seed, and no earlier decision. It does not itself constitute
  fresh material approval of the result.
- **Context:** DEC-036 locked task 9G's bounded protocol and implementation, and
  DEC-037 recorded the fresh top-level `APPROVE TASK 9G IMPLEMENTATION FOR
  MERGE` verdict at cumulative commit `8d27c23`. That approval covered
  implementation merge and one subsequent human-invoked `run-to-validation`
  only.
- **Executed identity:** The implementation merged at
  `e930454d1c0869e22778a99ae22505f093889e73`. The human operator then invoked
  `python scripts/run_american_neural_pilot.py run-to-validation` exactly once,
  under protocol `configs/american_neural_pilot_protocol_v1.toml` (SHA-256
  `6600a46ea2132bd3c834645ddb704ccc71069a0cca84d3762b4c4ca753aac591`). The
  consumed raw validation report has SHA-256
  `9a4acdfd3b96eee3d29ef9c12c467ea304297dc44d561f56fdaeb2292ba999a9`; it stays
  beneath ignored paths and is never edited or regenerated.
- **Decision:** Freeze the run's outcome as the terminal task 9G result:
  `status = validation_gates_failed`, `outcome = failure_to_learn`, recorded at
  `outcome.phase = "validation"` with `lifecycle.state = "validation_terminal"`.
  This is one of the four predeclared honest outcomes, not an unfinished task,
  and it is never rerun, retuned, or reinterpreted as a partial success.
- **Entry gates, all discharged:** The independent PDE mapping check ran before
  optimization and **passed on 21 of 21 rows**. The exact European-to-American
  transfer lift **passed** at all eight protocol-pinned probes — maximum
  absolute difference `1.4210854715202004e-14`, maximum relative difference
  `3.6214823824845716e-13`, against `1e-12` tolerances. Dataset, generator,
  configuration and source-artifact identities all verified. No gate was
  waived, loosened, or retried.
- **Why it failed:** **Neither arm passed every validation gate.** `scratch`
  failed all five predeclared checks; `transfer` passed only
  `normalized_p99_absolute_error` and failed the other four. Both
  `all_iv_errors_passed` and `all_reference_speedups_passed` are `false`.
- **Transfer versus scratch:** `transfer_validation_rmse_strictly_better =
  true` — transfer's validation RMSE was strictly better than scratch's. This
  is a within-pilot comparison only. The two locked arm seeds also produce
  different epoch shuffle permutations, so this one-seed pilot cannot attribute
  the difference solely to transfer initialization. It is not evidence that
  transfer initialization helps, and it does not convert a failed gate into a
  pass.
- **Final-partition non-consumption:** `final_evaluation_attempts = 0` and
  `final_partition_consumed = false`. `interpolation_test` was never opened,
  hashed, imported, or counted, by any code path or by any agent.
- **Final evaluation is forbidden:** `validation_final_entry_passed = false`
  and `second_attempt_allowed = false`, so the protocol's final-entry rule is
  unmet and `final-evaluate` may not be invoked under this protocol, now or
  later. Any future final evaluation requires a new predeclared protocol and a
  fresh final partition. The unconsumed partition is never tuned against, and a
  failed pilot is never rescued by opening it.
- **Frozen evidence:** [results/american_neural_pilot_results_v1.json](results/american_neural_pilot_results_v1.json),
  schema `american-neural-pilot-result/1`, validated offline by
  `python3 scripts/freeze_american_neural_pilot_results.py --check`, which is
  now wired into both `scripts/check.sh` and CI. That check reads only tracked
  files and reruns no pricing, training, latency measurement, or IV inversion.
  The snapshot is authoritative for every number and is never hand-edited or
  regenerated.
- **Non-claims:** This result establishes **no** H2 result — **one seed and one
  budget do not establish H2** — and no converged American-price truth, American
  Greek accuracy, OOD or extrapolation behavior, discrete-dividend
  applicability, SPY performance, market calibration, bid--ask-relative
  accuracy, production readiness, or live trading value. The 21-row PDE check is
  mapping-consistency evidence only, not converged truth and not
  semantic-coverage evidence, and its fixed-domain 400x200 versus 800x400
  refinement pair does not independently bound domain-truncation error. The
  measured speedups failed their gate and would in any case be conditional on
  the exact matched benchmark contract, not portable beyond the recorded
  hardware, software, request shapes and thread budgets. The IV experiment is
  model-consistent synthetic evidence spanning maturities `0.25` and `2.0` —
  correctly an IV **surface** — and reports no market fit. Offline snapshot
  checking detects internal inconsistency and tracked-input drift but cannot
  authenticate a fully coordinated fabricated raw report and snapshot. Task 9E's
  admission remains conditional and mapping-only, and this result does not
  upgrade it.
- **Next intended task, recorded as intent only:** **task 9H**, an explicitly
  exploratory American neural-pricer development loop on `train` and
  `validation` data only. Claude or Codex may iteratively implement capacity,
  feature, target and architecture experiments; the human runs every one of them
  locally under the standing manual-run rule; every attempt and every failure is
  documented, including abandoned ones; and `interpolation_test` and every other
  final partition remain inaccessible throughout. Because selection would happen
  against `validation`, any model 9H selects carries selection bias and is not a
  result: it requires a separate, freshly predeclared confirmation on a fresh
  partition after development ends, as its own task with its own gates and its
  own review. This entry neither implements, scopes, nor authorizes task 9H.
- **Authoritative links:**
  [results/american_neural_pilot_results_v1.json](results/american_neural_pilot_results_v1.json),
  [project-state.md](project-state.md),
  [tasks/active/task-9g-american-neural-pricer-pilot.md](tasks/active/task-9g-american-neural-pricer-pilot.md),
  DEC-034, DEC-035, DEC-036, DEC-037

### DEC-049 — Adopt Architecture Freeze v2.3 as the normative parent of the American neural-pricer roadmap

- **Status:** Active. **Supersedes** the Task 9H development roadmap — the
  adaptive `train`/`validation` attempt loop recorded on the archived
  exploratory line — as the plan for reaching an American neural result. It
  supersedes **no** frozen evidence, reinterprets **no** recorded outcome, and
  reruns **nothing**.
- **Provenance note on numbering.** Decision IDs `DEC-039`–`DEC-048` are
  reserved by the archived Task 9H exploratory line — branch
  `experiment/task-9h-american-pricer-development`, tag
  `task-9h-v1-exploratory-pre-v2.3` (commit `3950ed0`) — and are therefore
  intentionally absent from this branch's linear history. That branch is not
  intended to be merged wholesale; reserving the range keeps decision IDs
  globally unique across preserved project history. This entry does not
  restate, alter, or replace those decisions.
- **Context.** Task 9G's locked run produced a frozen negative result
  (`failure_to_learn`, DEC-038). The Task 9H exploratory line that followed
  reached a recorded price leader, recorded the E2c candidate, and paused its
  attempt loop for a zero-training diagnostic phase whose grids were
  predeclared and implemented but not scored. External review of Architecture
  Freeze v2 then produced v2.3, which reorganizes the remaining work rather
  than continuing the 9H loop.
- **Decision.** Adopt
  [american-neural-architecture-freeze-v2.3.md](american-neural-architecture-freeze-v2.3.md)
  as the **normative parent** of the American neural-pricer roadmap. It is a
  contract, not narrative: it sits in tier 2 of the source-of-truth hierarchy
  and, for the American neural design, ahead of
  [architecture.md](architecture.md) and the other `*-contract.md` files. It is
  versioned, not amended in place — a change means a new freeze document
  adopted by its own decision entry.
- **All prior Task 9H evidence is preserved as exploratory historical
  evidence.** Its specification, append-only attempt ledger, interim research
  brief, predeclared diagnostic protocol, E2c checkpoint, and decision entries
  `DEC-041`–`DEC-048` remain intact at the archive tag above. Nothing there is
  deleted, rewritten, or reinterpreted. Equally, nothing there is a project
  result: selection happened repeatedly against `validation`, so every 9H
  outcome carries selection bias and is exploratory input to the v2 design
  only. Adopting v2.3 does not retroactively make v2.3 the plan those
  experiments were run under.
- **E2c cannot be the v2 confirmation model.** Its teacher dataset is not
  reproducible from committed source — the American generator is on the
  unmerged branch `feat/american-dataset-v1` and was never carried onto `main`
  (DEC-031, DEC-033) — so no confirmation claim can rest on it. E2c survives
  only as a concrete 199k-parameter specimen for the native C++ feasibility
  prototype, and the remaining Greek diagnostics inform the **head and
  representation design**, not whether historical E2c weights can be retained.
- **The v2 confirmation campaign requires all of the following**, each a
  structural requirement rather than a preference:
  1. a **reproducible generator** — the American dataset must be regenerable
     from committed source plus a committed configuration;
  2. **atomic dataset generation** — every v2 partition (train, validation,
     development holdouts, the neutrally sampled final interpolation set, and
     the predeclared economic stress set) emitted by one invocation of one
     committed generator configuration and bound by one manifest, so that
     "untouched final" is a property of a precommitted sampling design and not
     of a second sampling event;
  3. **retraining** — a new model trained under that dataset, under a declared
     attempt budget with development-holdout and final-access controls intact;
  4. **independent numerical-reference convergence** — price-reference and
     per-Greek reference depths selected on a partition-free convergence set
     against a predeclared application/reference tolerance, never against the
     model's achieved error, with additive PDE evidence from outside the CRR
     lattice family;
  5. **deterministic artifact export** — a non-pickle, C++-loadable artifact
     with a manifest binding feature order, architecture, scaling, head
     parameters, weight hash, dataset identity, and provenance;
  6. **native C++ inference** — the timed production path, with no timed call
     falling back into Python;
  7. **one-shot final evaluation** — the untouched final partition opened
     exactly once, with no post-final tuning under the same confirmation claim.
- **No final or interpolation partition has been consumed by this roadmap
  change.** This is a documentation-only adoption. Task 9G's
  `interpolation_test` remains unconsumed (`final_evaluation_attempts = 0`,
  `final_partition_consumed = false`) and its `final-evaluate` remains
  **forbidden** under the 9G protocol; the archived 9H line never opened a final
  partition; and v2.3 introduces no new pass/fail threshold and authorizes no
  training, no generation, and no evaluation.
- **Consequences.**
  - The next task is **task 9I**, the v2.3 Phase 0 transition: finish only the
    carried-forward Greek/head diagnostics, and independently build the native
    C++ E2c feasibility prototype. Both are exploratory inputs to the v2 design,
    not confirmation work
    ([tasks/active/task-9i-architecture-v2-3-phase-0.md](tasks/active/task-9i-architecture-v2-3-phase-0.md)).
  - The freeze's Phase 1 — committing the v2 architecture, application
    tolerances, reference-error budget, Greek-reference rules, attempt budget,
    sampling design, final-access policy, and native benchmark contract —
    follows Phase 0 and needs its own spec and review.
  - Task specifications keep their existing paths and are marked in place; the
    lifecycle grouping is [tasks/README.md](tasks/README.md). No spec was
    relocated, so every task link recorded in earlier entries of this
    append-only log still resolves, and no earlier entry was edited.
  - The locked stage-2 roadmap (DEC-028) is unchanged: v2.3 governs *how* the
    phase-1 American neural experiment is built and confirmed, not the phase
    order.
  - Five normative contracts — [architecture.md](architecture.md),
    [research-contract.md](research-contract.md),
    [american-crr-contract.md](american-crr-contract.md),
    [american-crr-dataset-admission.md](american-crr-dataset-admission.md), and
    [pde-numerical-contract.md](pde-numerical-contract.md) — are
    content-digest-pinned by task 9G's frozen protocol and could not be revised
    by this adoption without failing that gate. They are left byte-identical and
    are now read as the record of what task 9G built; where they disagree with
    the freeze about the American neural design, the freeze wins by the
    precedence order. Recorded in
    [documentation-map.md](documentation-map.md), "Digest-pinned documents".
- **Authoritative links:**
  [american-neural-architecture-freeze-v2.3.md](american-neural-architecture-freeze-v2.3.md),
  [tasks/active/task-9i-architecture-v2-3-phase-0.md](tasks/active/task-9i-architecture-v2-3-phase-0.md),
  [tasks/README.md](tasks/README.md),
  [project-state.md](project-state.md),
  [documentation-map.md](documentation-map.md),
  DEC-001, DEC-028, DEC-031, DEC-033, DEC-034, DEC-036, DEC-038
