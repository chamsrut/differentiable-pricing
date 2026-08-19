# Research contract

## Primary question

Can a smooth neural surrogate approximate a trusted derivative pricer and its
useful risk sensitivities, across a declared domain, with lower end-to-end
latency and an error profile suitable for a stated decision?

The project is not trying to show that a network can interpolate an arbitrary
grid. It is testing accuracy, derivative fidelity, generalization, latency,
and whether transfer learning reduces the number of expensive labels needed.

## Hypotheses

### H1: European option sanity check

A compact network trained on Black--Scholes labels can reproduce prices and
delta/vega over a bounded domain. Analytic formulas must expose normalization,
derivative, sampling, and implementation errors before complex products are
introduced.

### H2: European to American transfer

Starting from a European-option representation reduces American-option sample
requirements or training time compared with a matched randomly initialized
model, while reaching equal or better out-of-sample price and Greek error.

### H3: European to Bermudan swaption transfer

A model initialized from European-swaption training reduces the label budget
needed for Bermudan swaption accuracy versus random initialization. The
comparison must control architecture, optimizer, schedules, wall-clock budget,
seeds, and training examples.

## Stages and reference standards

| Stage | Minimum reference | Required checks |
|---|---|---|
| European option | Analytic Black--Scholes | closed-form prices/Greeks, parity, monotonicity, MC convergence |
| American option | Converged CRR tree | European lower bound where applicable, exercise-boundary behaviour, step convergence |
| European swaption | Independently checked Hull--White 1F implementation | curve/convention fixtures, limiting cases, calibration reconstruction |
| Bermudan swaption | Converged tree/PDE or out-of-sample LSM | lower bounds, exercise-policy separation, path/step/basis convergence |

Reference-pricer numerical error must be materially smaller than the surrogate
acceptance threshold. Otherwise the experiment measures label noise.

The stage-2 scalar tree implementation and its current numerical limitations
are specified in
[american-crr-contract.md](american-crr-contract.md). Adjacent-step agreement
is a convergence diagnostic, not a proof of accuracy; the label-generation
protocol must measure convergence across the complete sampling domain.
The exploratory stage-2 convergence configuration uses named stress cases, an
internal high-step CRR reference, analytic European controls, and a separate
probability-feasibility grid. It does not select the production label step
count by itself.

The independent stage-2 LSM contract is specified in
[american-lsm-contract.md](american-lsm-contract.md). It freezes a policy on
one antithetic path stream and values it on another, reports pair-aware Monte
Carlo uncertainty, and varies path count, exercise-grid density, and basis
degree one factor at a time. Its confidence interval covers valuation sampling
error only; policy suboptimality and Bermudan exercise-grid bias remain
separate numerical effects.

The reviewed stage-2 LSM/CRR cross-check evidence is frozen in
[results/american_lsm_crosscheck_results_v1.json](results/american_lsm_crosscheck_results_v1.json)
(schema `american-lsm-crosscheck-results/1`), extracted programmatically from
the reviewed raw report by `scripts/freeze_american_lsm_results.py`. Seven
experiments over nine pinned regimes, 63 case rows. The primary experiment
reaches a mean CRR-minus-LSM gap of 0.02380 price units with a mean
valuation-only standard error of 0.01263.

That evidence is descriptive, not a gate. In particular:

- the stochastic interval-containment count is a diagnostic with no nominal
  rate and **must not** be promoted to an acceptance criterion;
- deterministic zero-width cases are summarised separately from stochastic
  interval comparisons and are excluded from the reported minimum
  variance-reduction ratio;
- the CRR reference is an internal cross-check, not exact American truth and
  not market truth.

**This study selects no production label policy.** Task 8E separately
predeclares and runs the CRR label-policy calibration study, with its own
gates fixed before results are observed. No number frozen by task 8D may be
reused as a task 8E acceptance criterion.

### The discrete-dividend PDE oracle and its label-policy pilot

Neither the tree nor the LSM engine prices an explicit cash dividend, which a
real American equity option needs. Task 9C-A added a third, numerically
unrelated reference: a one-dimensional Crank--Nicolson finite-difference solver
with Rannacher damping and a PSOR solve of the American obstacle, specified in
[pde-numerical-contract.md](pde-numerical-contract.md). It prices only; it
exposes no sensitivity.

Task 9C-B then asked, under criteria frozen in
`configs/pde_label_policy_pilot_v1.toml`, whether a fixed grid can supply
price, delta, gamma and vega labels by centered bumps. The frozen answer in
[results/american_pde_label_policy_results_v1.json](results/american_pde_label_policy_results_v1.json)
is `no_policy_selected`, with `criteria_were_not_loosened = true`. The
1600x800 candidate met every predeclared absolute-error cap on all 22 regular
cases and still failed five of them on bump stability and on an
American-dominance shape check; Richardson extrapolation was the most accurate
candidate where the solution is smooth but had unsupported observed order in 11
of the 28 cases, so it is a reference technique rather than a label policy.

Three consequences bound later work at the time. **No American label policy
existed**, so no American dataset generation and no American neural training
could begin. A negative selection is a result, not a licence to relax the
criteria that produced it. And the pilot's measured cost — thirteen scalar
solves per state and grid — is why the next milestone (task 9C-C1) was to read
a valuation-time slice and its delta and gamma out of a single solve, not to
buy more cores.

Task 9C-C3 later asked a **new, separately predeclared** question — whether a
revised stability/shape rule, with v1's four absolute-error caps carried over
unchanged, can pass the cases v1 failed — and answered it positively: the
accepted accuracy policy is `grid_1600x800`, externally approved by a fresh
top-level session ([decision-log.md](decision-log.md) DEC-025). That does not
reinterpret v1, whose snapshot stays immutable and whose negative selection
stands. Two limits survive it. The accepted policy covers **labeling only**:
the frozen report authorizes neither dataset generation nor training input, so
an American dataset and American neural training are still separately gated and
have not begun. And no number from either study may be reused as an acceptance
criterion for a later one.

## The locked stage-2 roadmap

This section is appended, not a rewrite: it fixes the order in which the
stage-2 American work is attempted, and it does not alter any rule, hypothesis,
threshold, or non-claim stated above. It was locked as part of a
documentation-only roadmap reset ([decision-log.md](decision-log.md) DEC-028).

### The question this roadmap exists to answer

Can a neural surrogate price American options with **useful accuracy** while
delivering **materially faster inference than the numerical method that
generated its labels**, and does that speedup carry through to **faster
implied-volatility inversion and volatility-surface construction**?

This is a specialization of the primary question above, not a replacement for
it. "Useful accuracy" and "materially faster" are still governed by the stage
reference standards, "Metrics", and "Provisional gates": end-to-end latency
must count feature, transfer, batching and derivative costs, and a speedup
claim measured against anything other than the label-generating method under
matched conditions does not answer this question.

### Phase 1 — continuous-dividend-yield American CRR baseline

The first learnability experiment runs against a **continuous-dividend-yield
American CRR** dataset, not a no-dividend one: its underlying carries a
continuous yield `q`, which is a different modelling object from a discrete
cash-distribution schedule and is never treated as equivalent to one
(DEC-001).

Its purpose is bounded and entirely about learnability and speed:

- whether American prices are learnable at all at useful accuracy;
- scratch training versus European-transfer initialization (H2);
- out-of-sample inference accuracy;
- inference scaling against CRR's $O(N^2)$ lattice cost;
- a small implied-volatility inversion and surface reconstruction built on
  surrogate inference.

**A local candidate CRR dataset reportedly exists, but it has not yet been
catalogued, validated, reproducibly admitted, or accepted as project
evidence.** It is local, Git-ignored and untracked, and insufficiently
catalogued. It must be audited before it is admitted as a training input or
used to train anything.

**Update, task 9D.** The audit that paragraph called for has since run. The
dataset is now **catalogued** — [data-holdings-catalogue.md](data-holdings-catalogue.md),
[decision-log.md](decision-log.md) DEC-031 — and the other three conditions are
unchanged: it has **not** been validated as suitable, **not** reproducibly
admitted, and **not** accepted as project evidence. It remains local,
Git-ignored and untracked, its generator and configuration are on an unmerged
branch, and its label policy has no frozen `docs/results/` evidence. Admission
is task 9E (DEC-032). Nothing above is relaxed by this update.

**Update, task 9E.** Admission has since been decided:
[american-crr-dataset-admission.md](american-crr-dataset-admission.md),
[decision-log.md](decision-log.md) DEC-033. The dataset is **admitted solely for
the bounded continuous-yield American CRR learnability and latency experiment
described in phase 1 above**, under the schema `american-option-dataset/1` and
the representation `american_raw_physical_v1`, with every integrity and leakage
gate passing over all 250,000 rows. Four limits stand. Admission **authorizes no
training** — the first American training run is a separate task with its own
gate. The admission is a **single-session conclusion whose material approval is
outstanding**. The dataset is **still not regenerable from tracked sources
alone**: task 9E recovered its configuration, its pilot configurations and its
label-policy contract section, but deliberately not the generation machinery.
And its label policy **still has no frozen `docs/results/` evidence**, its
selection resting on ignored local artifacts. Nothing above is relaxed by this
update, and phase 1 remains a continuous-yield experiment that says nothing
about discrete dividends.

### Phase 2 — the XSP/SPY real-instrument study

Phase 2 moves to contracts grounded in the available market universe, with a
deliberate control/target pair:

- **XSP** — European, cash-settled — is the **control**;
- **SPY** — American, with discrete deterministic cash distributions and
  early exercise — is the **target**.

Its goal is to learn PDE prices for those contracts, compare surrogate
inference against the PDE, and reproduce and evaluate real implied-volatility
surfaces. The existing discrete-dividend PDE solver
([pde-numerical-contract.md](pde-numerical-contract.md)) and the accepted v2
label policy remain valuable inputs to this phase.

**No accepted, versioned PDE-labelled SPY training dataset exists**, and no
SPY neural surrogate exists.

### Phase 3 — deferred commodity extension

Commodity options are **not on the current critical path**. Corn options are
the leading future candidate, because American exercise into futures,
seasonality, and the futures curve would supply a genuinely different
cross-asset test of the same question. This phase is not designed, not
scoped for data acquisition, and not implemented now.

### Explicitly excluded: crypto

Crypto options are excluded from the active roadmap. The liquid crypto option
universe is European-only, so it does not advance the American-option research
question this roadmap exists to answer. Exclusion here is a scope decision,
not a numerical judgement.

### What this roadmap does not change

- **The accepted PDE label policy remains frozen evidence; this roadmap
  change does not reinterpret or rerun it.** `grid_1600x800` and its
  eligibility readings stand exactly as frozen (DEC-025, DEC-026, DEC-027).
- **No American neural surrogate has yet been trained and accepted.**
- **Cloud storage and vendor ingestion are deferred reproducibility work, not
  prerequisites for the first CRR learnability experiment.**

## Data protocol

Each generated row or partition records:

- contract and model inputs in explicit units;
- price and available reference sensitivities;
- reference method and version;
- seed and random-number policy where applicable;
- convergence controls and estimated numerical error;
- feature-domain identifier and generation timestamp.

Sample in transformed economic coordinates where useful (for example log
moneyness rather than raw spot and strike). Keep named partitions:

- interpolation test: held-out samples inside the training envelope;
- boundary test: short expiry, low volatility, deep moneyness, and exercise
  frontiers;
- extrapolation/OOD test: explicitly outside at least one training bound;
- scenario test: coherent curve/surface shocks rather than independent rows.

Do not random-split rows originating from the same paths, grids, curve
scenario, or near-duplicate contract state.

### Real-market inputs

Tasks 9A and 9B establish where a real-market pricing input could come from,
and nothing more. 9A audits a three-session proprietary quote archive read-only
and reports scoped capabilities; 9B fits same-minute put-call parity and
classifies each PDE input as directly observed, pointwise identified, jointly
identifiable only, external convention, or unavailable
([market-state-reconstruction-contract.md](market-state-reconstruction-contract.md)).

Neither produces a price, a label, a Greek, an implied volatility, or a
calibrated curve or surface. Discount factors and forwards are identified at
quoted expiries as knots with no interpolation through them; dividend amounts,
borrow and carry, and corporate-action adjustments are not available from that
archive. Every artefact of both studies is derived from proprietary quote-level
data and stays out of Git, so their tests use synthetic fixtures and CI never
sees the archive. Every threshold in both is exploratory and was chosen after
the archive was observed; none gates a replication partition.

## Metrics

Report at least:

- price MAE, RMSE, relative error where stable, p95, p99, and maximum;
- delta/vega for equity stages and PV01/vega for rates stages;
- arbitrage and shape violations: bounds, monotonicity, convexity where
  applicable;
- reference-pricer time, feature time, model time, derivative time, and total
  latency for fixed batch sizes;
- model size and peak memory;
- error sliced by expiry, moneyness, volatility, exercise region, and OOD flag.

All latency results require warm-up, pinned software/hardware metadata,
multiple repetitions, and uncertainty intervals. GPU and CPU results are not
interchangeable.

## Transfer-learning experiment

For each target product compare:

1. random initialization;
2. source-product pretrained initialization;
3. optionally, frozen trunk followed by partial/full fine-tuning.

Use at least five training seeds and several label budgets. Select
hyperparameters without looking at the final test set. The primary transfer
claim is the target-label budget needed to cross a predeclared error gate, not
the prettiest single learning curve.

Negative transfer is a valid result and must be reported.

## Provisional gates

Stage-specific gates live in versioned configuration. Before generating the
first large dataset, replace provisional values with tolerances connected to
a use case. Passing average error alone is never sufficient: tail errors,
shape constraints, cross-language parity, and derivative checks must pass.

The first European fresh-seed replication is bound by
`configs/european_neural_replication_protocol_v1.toml`. That protocol preserves
the development sampler and selected model except for independently derived
dataset and training seeds, pins every input file by SHA-256, permits validation
for checkpoint selection, and permits one final evaluation on the fresh
`interpolation_test`. It must be committed before replication data generation.
If the final gate fails, record the failure and version a new protocol; do not
tune against the consumed partition.

Protocol v1 completed with one final-evaluation attempt and passed every
frozen validation and final gate. Its immutable, digest-linked outcome is
`docs/results/european_replication_results_v1.json`. The final partition is
consumed permanently. The result supports synthetic in-envelope European
interpolation only; it does not relax the non-claims below or authorize further
tuning against that dataset.

## Non-claims

- A network derivative is not an exact Greek of the reference model.
- A synthetic-label experiment is not evidence of live trading alpha.
- Faster kernel inference is not faster end-to-end pricing unless feature,
  transfer, batching, and derivative costs are counted.
- Successful interpolation is not proof of extrapolation.
- A valuation-only Monte Carlo interval is not a total-error bar. It excludes
  policy-fitting error and exercise-grid bias, so an interval that excludes a
  tree reference is not by itself evidence of a defect.
- Agreement between LSM and a high-step CRR tree is agreement between two
  numerical methods inside one model. It is not evidence that the model
  describes market prices.
- Meeting the task 9C-B accuracy caps at one grid is not a label policy. The
  pilot selected none, and its idealized worker projections are not a
  feasibility claim.
- An accepted label policy is not an accepted dataset. Task 9C-C3's accepted
  `grid_1600x800` authorizes labeling within its measured eligibility — price,
  delta where eligible, vega where eligible, never gamma — and authorizes no
  dataset generation and no training input.
- Task 9C-C3's 28 evidence cases are isolated points. They do not validate the
  surrounding parameter hyperrectangle, and no interpolation between them is
  licensed. A later generation task predeclares its own domain rather than
  inheriting one.
- Task 9C-C3's residual-scale-aware dominance and intrinsic allowances are
  operational price-error scale estimates, not certified bounds
  (`is_a_rigorous_bound = false`).
- A three-session archive audit is a feasibility and identifiability study, not
  a historical market study, and it establishes no calibration capability. No
  dividend amount, borrow rate, or American implied volatility has been
  inferred anywhere in this repository.
- A continuous dividend yield is **not** a discrete cash-distribution
  schedule. The phase-1 CRR dataset carries a continuous yield and therefore
  **cannot model SPY cash dividends**; nothing learned on it transfers as a
  claim about discrete-dividend American pricing (DEC-001, DEC-030).
- A local candidate CRR dataset reportedly exists, but it has not yet been
  catalogued, validated, reproducibly admitted, or accepted as project
  evidence. Its row count, its parquet files, and its manifest are not evidence
  until an audit establishes them; no number from it may be cited as a project
  result before then (DEC-030).
- Task 9D catalogued that dataset and admitted nothing. Cataloguing a holding
  is not accepting it: the dataset is still not validated as suitable, not
  reproducibly admitted, and not accepted as project evidence, and no number
  from it may be cited as a project result before task 9E's admission gate and
  a fresh top-level approval (DEC-031, DEC-032).
- Task 9E admitted that dataset for **one bounded experiment only** — the
  continuous-yield American CRR learnability and latency experiment — and for
  nothing else. Admission is not training authorization, not a production
  label-quality claim, not a certified Greek, and not discrete-dividend
  coverage; its material approval is outstanding; the dataset remains
  unregenerable from tracked sources; and its label policy still has no frozen
  evidence (DEC-033).
- No independent cross-check of the CRR labels against a numerically unrelated
  engine has been run. Every identity verified in tasks 9D and 9E is internal to
  one lattice (DEC-033).
- No accepted, versioned PDE-labelled SPY training dataset exists, and no
  American neural surrogate has yet been trained and accepted.
