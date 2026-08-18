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
