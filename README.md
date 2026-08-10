# Differentiable Pricing

A research platform for testing whether neural surrogates can reproduce
derivative prices **and useful risk sensitivities** at materially lower
inference latency than their reference pricers.

The expensive part of re-pricing a book is rarely one price: it is the Greeks,
the scenario grid, and the exercise-aware models behind them. A smooth network
is attractive because one reverse-mode pass returns the value and every input
sensitivity at once. Whether those sensitivities are *trustworthy* is
falsifiable, so the project starts where truth is known and moves outward one
controlled stage at a time.

> A network derivative is exact **for the learned network**, not automatically
> an exact market Greek. It becomes a credible Greek only after feature and
> target normalization are reversed and the result is validated against
> analytic, algorithmic, or convergence-tested bump references.

The first oracle is a C++20 Black–Scholes implementation with analytic Greeks.
Four planned stages add one difficulty at a time — European option, American
option, European swaption, Bermudan swaption — each with its own trusted
reference pricer and a transfer experiment against random initialization. Every
stage must beat acceptance criteria fixed *before* results are observed, on a
partition that has informed no model choice
([docs/research-contract.md](docs/research-contract.md)).

**Stage 1 is complete and independently replicated. Stage 2 now has three
independent reference engines — a CRR tree, an LSM cross-check, and a
discrete-dividend finite-difference PDE oracle — plus a frozen label-policy
pilot that selected no policy, and still no neural work. A parallel real-market
track has audited a three-session quote archive and established which pricer
inputs it can and cannot supply.**

---

## Headline evidence

### Stage 1 complete: fresh-seed European replication

A precommitted protocol was executed once, end to end, on a **newly generated
250,000-row dataset and a new initialization seed**. Architecture, objective,
domain, row counts, optimizer, budget, evaluation bands, output constraint, and
every acceptance gate were bound by SHA-256 before the data existed. Training
and checkpoint selection saw only `train` and `validation`; the fresh
`interpolation_test` was evaluated once and is now permanently consumed.

| Frozen check | Validation | Locked final | Limit | Final budget used |
|---|---:|---:|---:|---:|
| Price / spot RMSE | 0.0001799 | **0.0001805** | 0.0005 | 36.1% |
| Price / spot P99 absolute error | 0.0004290 | **0.0004246** | 0.002 | 21.2% |
| Delta RMSE | 0.002613 | **0.002942** | 0.01 | 29.4% |
| Gamma RMSE | 0.001679 | **0.001937** | 0.002 | 96.8% |
| Vega RMSE | 0.2452 | **0.2627** | 1.0 | 26.3% |
| Theta RMSE | 0.1359 | **0.1311** | 1.5 | 8.7% |
| Rho RMSE | 0.2840 | **0.2901** | 2.0 | 14.5% |
| Tail price / spot RMSE | 0.00006315 | **0.00005324** | 0.0015 | 3.5% |
| Tail delta RMSE | 0.003029 | **0.002817** | 0.03 | 9.4% |
| Tail gamma RMSE | 0.002075 | **0.002002** | 0.005 | 40.0% |
| Material European-bound violations | 0 | **0** | 0 | Pass |

![Validation and locked-final errors divided by their frozen acceptance limits, all bars below the limit line, with gamma closest to it](docs/figures/european_replication_gate_margins.svg)

> A fresh-seed differential neural surrogate replicated its synthetic
> in-domain European-option pricing and Greek accuracy, passing all 11
> precommitted validation and locked-final gates with zero material
> European-bound violations.

Three qualifications belong with that claim, not in a footnote.

**Gamma is the least comfortable result.** Its locked-final RMSE consumed 96.8%
of the frozen allowance — a legitimate pass, but curvature is the first metric
to stress in any boundary or out-of-domain study.

**Zero arbitrage violations is structural, not learned.** The trained network
alone violated the discounted lower bound materially on 1,584 locked-final
rows; the deterministic European-bounds projection adjusted 1,586 prices (6.34%
of the partition), all at the lower bound, driving material violations to zero,
with the bounded and unconstrained artifacts sharing one weight digest.

![European-bounds projection: 1,584 material violations before enforcement, zero after](docs/figures/european_replication_projection.svg)

**The six reported Greeks are not six independent confirmations.**

| Greek | Relationship to the training objective |
|---|---|
| Price | Directly supervised. |
| Delta, vega | Directly supervised, through the transformed first derivatives $u_x$ and $u_v$. |
| Rho, theta | Not supervised; algebraic reweightings of the same learned $(u,u_x,u_v)$ on the unconstrained branch, so they chiefly confirm physical-unit reconstruction. |
| Gamma | Not in the objective; requires second-order autograd through $u_{xx}$. First-derivative supervision plausibly regularizes it indirectly, but that is an unablated explanation, not a measured causal result. |
| European bounds | Not learned; imposed structurally by the projection. |

Where the projection is active the reported derivatives follow the active
discounted bound instead, and no derivative is unique exactly at a kink. The
metrics above come from the constrained model that was shipped, so they already
reflect whichever branch applied at each row.

The immutable snapshot
[`docs/results/european_replication_results_v1.json`](docs/results/european_replication_results_v1.json)
records commits, seeds, toolchain, digests, byte-identical weight lineage,
every metric, gate decisions, and the projection intervention; its ledger
records `final_evaluation_attempts = 1` and `status = replication_passed`.
Digests are traceability, not an authenticated attestation of who ran a run.

### Stage 2: two independent American references agree

American work is the active second stage and is so far entirely about
*reference pricers*, not networks. Two numerically unrelated methods price the
same nine pinned regimes: a deterministic **Cox–Ross–Rubinstein tree** refined
to an 8,192/8,193-step adjacent average, and a **Longstaff–Schwartz Monte
Carlo** engine that fits a stopping policy on one antithetic path stream and
values that frozen policy on a second. Seven experiments, 63 case rows;
`CRR−LSM` is in price units against the CRR adjacent average.

| Experiment | Role | Steps | Degree | Train paths | Valuation paths | Mean CRR−LSM | Max abs gap | Mean valuation-only SE | Stochastic cases | In-interval | Deterministic cases | Min applicable VR ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `paths-low-v1` | path convergence | 64 | 2 | 16,384 | 32,768 | 0.02513 | 0.13529 | 0.02508 | 6 | 4 | 3 | 1.0096 |
| `steps-low-v1` | exercise grid convergence | 32 | 2 | 32,768 | 65,536 | 0.02808 | 0.12666 | 0.01781 | 6 | 3 | 3 | 1.0001 |
| `reference-v1` | reference | 64 | 2 | 32,768 | 65,536 | 0.02713 | 0.07977 | 0.01799 | 6 | 2 | 3 | 1.0033 |
| `steps-high-v1` | exercise grid convergence | 128 | 2 | 32,768 | 65,536 | 0.03298 | 0.14736 | 0.01784 | 6 | 2 | 3 | 1.0120 |
| `degree-one-v1` | basis sensitivity | 64 | 1 | 32,768 | 65,536 | 0.60422 | 2.43419 | 0.01828 | 6 | 1 | 3 | 1.0345 |
| `degree-three-v1` | basis sensitivity | 64 | 3 | 32,768 | 65,536 | 0.02828 | 0.11119 | 0.01778 | 6 | 3 | 3 | 1.0047 |
| `paths-high-primary-v1` | primary | 64 | 2 | 65,536 | 131,072 | 0.02380 | 0.07394 | 0.01263 | 6 | 0 | 3 | 1.0088 |

![Mean and maximum CRR-minus-LSM gap per experiment with valuation-only standard errors, degree one far worse than every other arm](docs/figures/american_lsm_experiment_comparison.svg)

The primary experiment `paths-high-primary-v1` reaches a **mean gap of 0.02380
price units and a worst single case of 0.07394**, on option values between 1.4
and 30. Three of the nine cases are deterministic — their valuation estimator
has exactly zero variance — and they agree with the tree to 4.3e-13, 1.4e-06,
and 1.5e-05.

![Primary-experiment case prices with valuation-only standard errors, deterministic zero-width cases marked separately](docs/figures/american_lsm_primary_cases.svg)

#### How to read the gap without misreading it

LSM sits below the finer-grid CRR reference in **every** stochastic case. That
is the expected ordering, not a defect: for a fixed learned Bermudan stopping
policy the expectation is no greater than the same-grid optimal stopping value.
Three sources of difference must be kept apart, and the reported interval
measures only the first:

* **Valuation noise** — sampling error around an already-frozen policy, the
  `Mean valuation-only SE` column. It shrinks as $n^{-1/2}$: along the path
  ladder it falls 0.02508 → 0.01799 → 0.01263, ratios of 1.3946 and 1.4244 at
  full snapshot precision against the $\sqrt{2}\approx1.4142$ a doubling
  implies — deviations of −1.4% and +0.7%.
* **Policy-fitting bias** — the learned rule is not optimal for its own grid.
* **Exercise-grid bias** — a 64-date Bermudan is not a continuously exercisable
  American.

The last two are systematic and do not shrink with valuation paths, so the
interval has no nominal coverage rate against the CRR reference and
`In-interval` is a diagnostic, **not an acceptance gate**: it falls 4 → 2 → 0
along the ladder precisely because added paths narrow the interval around a
fixed bias, while the point estimates keep agreeing to roughly 0.02 price
units. Deterministic zero-width cases cannot contain a finite-step tree value
however close the agreement, so they are reported separately rather than
converted into successes by an invented tolerance.

#### Sensitivity findings

![Sensitivity of the mean CRR-minus-LSM gap to basis degree, exercise-grid density, and path count](docs/figures/american_lsm_sensitivity.svg)

* **Degree one is inadequate**: the linear basis gives a mean gap of 0.60422,
  more than twenty times the quadratic arm. **Quadratic and cubic are
  comparable** — 0.02713 versus 0.02828, more than an order of magnitude below
  either arm's ≈0.018 mean valuation-only standard error. With one training
  seed per experiment that means "not separated by noise", not "equal".
* **More exercise dates did not help at a fixed path budget**: 32 → 64 → 128
  dates moved the mean gap 0.02808 → 0.02713 → 0.03298, because refining the
  grid adds regression dates without adding paths to fit them on.
* **More paths cut valuation noise but improved the gap only modestly**: along
  the 16,384 → 32,768 → 65,536 training-path ladder it moves 0.02513 → 0.02713
  → 0.02380 — real, small, and **not monotone**, consistent with a residual
  systematic bias.
* **Every applicable finite control variate reduced variance**, minimum ratio
  1.0001, with deterministic cases excluded from that minimum.
* **Regression fallbacks appear in sparse deep-OTM regions**: 4 in the primary
  experiment, all in `deep_otm_put`.

These are consequences of the method, documented in
[docs/american-lsm-contract.md](docs/american-lsm-contract.md), and must not be
tuned away.

#### What the American results do not establish

One-factor geometric Brownian motion with constant rate, dividend yield, and
volatility. All evidence is synthetic: **no market data**, quoted price, or
calibration target enters this study. The CRR reference is an internal
cross-check between two very different numerical methods — **not exact American
truth** and **not market truth** — and agreement with it says nothing about
whether GBM describes live option prices. These results **select no production
label policy**: task 8E predeclares the CRR label-policy calibration study
separately, and nothing here may be reused as its acceptance criterion.

Per-case rows, provenance digests, and declared limitations:
[docs/results/american_lsm_crosscheck_results_v1.json](docs/results/american_lsm_crosscheck_results_v1.json).

### Task 9C: a strong PDE oracle and a useful negative label-policy result

The tree and the Monte Carlo engine both carry a *continuous* dividend yield,
and a cash dividend is not a yield. Task 9C-A therefore added
`dp::finite_difference_price`: a deterministic one-dimensional Crank–Nicolson
solver with Rannacher damping, an M-matrix upwinding rule, a monotone cubic
Hermite dividend jump, and a projected successive over-relaxation (PSOR) solve
of the American linear complementarity problem, converged on the LCP residual
rather than the iterate change
([docs/pde-numerical-contract.md](docs/pde-numerical-contract.md)).

Task 9C-B then asked one predeclared question: can a fixed grid produce price,
delta, gamma and vega labels by centered price bumps, accurately and stably
enough to supervise a network? Twenty-eight cases — 22 regular, 6 stress —
three candidate policies in a declared order, and frozen caps in
`configs/pde_label_policy_pilot_v1.toml`. The answers point in different
directions and both matter.

**The oracle itself is numerically promising.** Against the study's internal
reference — 3200×1600, selectively Richardson-extrapolated with 1600×800 only
where the observed factor-two order supported it — the 1600×800 candidate met
every predeclared absolute-error cap across all 22 regular cases:

| Label | Worst regular-case error, 1600×800 | Frozen cap |
|---|---:|---:|
| Price | 3.5822e-4 | 5e-4 |
| Delta | 1.7330e-5 | 1e-3 |
| Gamma | 4.5225e-6 | 2e-4 |
| Vega, per unit volatility | 2.9727e-3 | 5e-2 |

**No universal four-label policy was selected.** The frozen recommendation is
`selected_accuracy_policy = no_policy_selected`, with
`criteria_were_not_loosened = true`. The 1600×800 candidate failed 5 of the 22
regular cases on checks that are not error caps: four on Greek bump stability,
where the *reference* delta or vega itself moved across the bump ladder by more
than the frozen variation allowance, and one on the American-dominance shape
check, where a negative-rate American put priced 2.7e-8 below its European
control against a 1e-8 shape tolerance. That gap is of the order of the
solver's own accumulated residual, but a criterion frozen before the run is not
relaxed after it. The cheaper 800×400 candidate failed 11 regular cases and the
Richardson pair 9, so the declared order had no passing member.

**Richardson extrapolation is a validation technique, not a label policy.**
Where the solution is smooth it was the most accurate candidate by an order of
magnitude — worst regular price error 5.056e-5, worst delta 1.03e-6. But the
observed factor-two order fell outside the predeclared `[1.5, 2.5]` support
band in 11 of the 28 cases, concentrated exactly where labels are hardest:
early exercise, discrete dividends, short maturities, and deep-in-the-money
kinks. Unsupported order was preserved as a result rather than repaired by
assuming second order. Ten cases are additionally flagged unsuitable for Greek
supervision while their price rows are retained.

**The operational bottleneck is scalar bump-and-reprice label generation.** The
pilot took 1,638 solves and about 126.6 million PSOR iterations to produce four
labels for 28 states — 2.19 hours of single-core wall time. Extrapolated to
250,000 four-label states that is 104.9 serial hours at 800×400 and 792.5 at
1600×800 (13.1 and 99.1 hours at eight *ideal* workers). Those projections
assume perfect independent-worker scaling and exclude scheduling, memory
contention, failures and dataset I/O: they are evidence that naive scalar
generation is not operationally acceptable, **not** a cluster or production
benchmark.

Every per-case error, check outcome and observed order is frozen in
[docs/results/american_pde_label_policy_results_v1.json](docs/results/american_pde_label_policy_results_v1.json).
The negative result is the pilot working as designed: it priced out a naive
labelling strategy for two hours of compute, not after a dataset had been
generated on it.

### Real-market inputs: what a three-session archive can supply (tasks 9A–9B)

A separate track asks where a discrete-dividend American pricer's inputs would
come from on real quotes. It has produced no pricer, no calibration and no
dataset, and every artefact it generates is derived from proprietary
quote-level data and stays out of Git.

**Task 9A** audits a three-session archive read-only: it verifies the SHA-256
manifest before decoding, consults the vendor's per-request completeness
statement, never modifies a raw file, and writes normalized Parquet beneath an
ignored tree. It reports *scoped* capabilities — archive integrity, ingestion
feasibility, XSP European surface readiness, SPY American calibration
readiness, replication readiness — rather than one verdict, over coverage
denominators spanning the whole resolved instrument universe. Its structural
surface minima are properties of the method; its session-count minima were
chosen after the archive was observed and are labelled
`provisional_pilot_target`, not desk-grade.

**Task 9B** fits put-call parity in the same minute, $y = C - P = a + bK$, and
reports what each PDE input's provenance would be
([docs/market-state-reconstruction-contract.md](docs/market-state-reconstruction-contract.md)).
Discount factors and forwards are **pointwise identified at quoted expiries,
with no curve interpolated through those knots**. A non-negative fitted slope is
published as a contradiction rather than clamped, and bid-ask widths weight the
fit as executable-liquidity measures, never as standard errors. The SPY
quantity has exactly one name, the **American parity carry residual**
$Q = S - D\,\mathrm{median}_i\,\tilde{F}_i$, which is not a dividend, a
dividend present value, a borrow rate, a SPY forward, or an independent market
input. The SPY ex-date 2026-06-18 is `officially_scheduled` from the issuer's
published distribution schedule: that fixes the date only, and no dividend
amount is described as observed or inferred anywhere. Variation across the
three intraday snapshots is reported and deliberately not decomposed — it may
combine genuine market movement, quote microstructure, a changing filtered
strike set, and fitting variation.

---

## Project status

| Component | Status | Evidence |
|---|---|---|
| Black–Scholes prices and analytic Greeks (C++20) | Complete | known values, parity, analytic delta, invalid inputs, Python parity |
| Smooth `tanh` MLP with reverse-mode input derivatives (C++20) | Complete | C++ reverse-pass and binding parity tests |
| European dataset generator and diagnostics | Complete | hash-verified manifests, leakage and bound checks |
| European surrogate: price → forward-normalized → differential → bounds-projected | Complete | [validation snapshot](docs/results/european_validation_results_v1.json) |
| Fresh-seed European replication | **Complete, locked, terminal** | [replication snapshot](docs/results/european_replication_results_v1.json) |
| CRR American tree, scalar and parallel batch (C++20) | Complete | bit-identical serial/parallel prices |
| LSM American cross-check (C++20) | **Complete, frozen** | [cross-check snapshot](docs/results/american_lsm_crosscheck_results_v1.json) |
| Discrete-dividend American PDE oracle, scalar, price-only (C++20) | Complete | [PDE contract](docs/pde-numerical-contract.md), refinement ladders, cross-engine tests |
| PDE four-label policy pilot (task 9C-B) | **Complete, frozen — `no_policy_selected`** | [pilot snapshot](docs/results/american_pde_label_policy_results_v1.json) |
| Real-market feasibility audit (task 9A) | Complete, local only | scoped capabilities; no artefact staged |
| Market-state reconstruction (task 9B) | Complete, local only | [reconstruction contract](docs/market-state-reconstruction-contract.md) |
| Valuation-time surface, internally consistent American Greeks (task 9C-C1) | **Not started — next milestone** | design only |
| American label policy v2; American dataset, training, transfer; swaption stages 3–4 | Not started | blocked on 9C-C |
| C++ artifact loading and deployment | Not implemented | `SmoothMlp` inference only |
| Latency claims; calibrated curves and surfaces; OOD partitions | Not established | benchmarks are evidence, not gates |

No generated dataset, weight file, raw study report, or market-derived artefact
is checked in. The repository carries versioned snapshots under `docs/results/`
and deterministic, CI-checked SVG figures under `docs/figures/` rendered from
those snapshots alone.

---

## Architecture

```text
cpp/ bindings/python/ python/   C++ core, pybind11 boundary, Python package
configs/                        Versioned experiment assumptions
docs/                           Contracts and research protocol
docs/results/ docs/figures/     Frozen snapshots and rendered plots
scripts/                        Developer checks and study runners
.claude/ .githooks/ .github/    Review agents, local hooks, CI
```

**C++** owns reference pricing and analytic sensitivities, CRR early-exercise
pricing with a parallel price-only batch boundary, LSM policy fitting and
valuation, the finite-difference PDE oracle with discrete cash dividends, input
validation, and reverse-mode derivatives of the deployed network. **Python**
owns sampling, partitioning, lineage, training, evaluation, study runners, and
the read-only market pipelines. Pricing formulas are never duplicated in Python
to make a test pass. [docs/architecture.md](docs/architecture.md).

---

## Quick start

Prerequisites: CMake 3.20+, a C++20 compiler (recent GCC, Clang, or MSVC),
Python 3.11+, and Git.

```bash
# C++ core only
cmake -S . -B build/dev -DDP_BUILD_PYTHON_BINDINGS=OFF -DDP_WARNINGS_AS_ERRORS=ON
cmake --build build/dev --parallel
ctest --test-dir build/dev --output-on-failure

# Python package, compiled extension included
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e '.[dev,train]'

# Full local gate: configs, protocols, snapshots, figures, lint, C++ tests, Python suite
./scripts/check.sh
```

The `train` extra adds PyTorch and is required by everything under
`python/tests/ml/`; for dataset work without it install `.[data,dev]`, and the
optional `market` extra adds the vendor DBN reader only the real ingestion
pipeline needs. The gate needs the editable install — a missing pytest fails it
rather than skipping the Python suite — and `--quick`, used by the pre-commit
hook, stops after the C++ tests. Use `-DCMAKE_BUILD_TYPE=Release` for any
recorded performance experiment.

The pricers emit machine-readable JSON:

```bash
./build/dev/dp_pricer call 100 100 1 0.05 0 0.20
./build/dev/dp_american_pricer american put 100 100 1 0.05 0 0.20 2048
```

The American executable reports the $N$- and $(N+1)$-step tree prices, their
average, their absolute gap, and exercise-region diagnostics. That adjacent gap
is a refinement diagnostic, **not** a pure parity measure or a certified error
bound ([docs/american-crr-contract.md](docs/american-crr-contract.md)).

---

## Reproduce the European study

All commands assume the editable install. Generated datasets, artifacts, and
reports are gitignored by design.

### Generate and diagnose the dataset

```bash
python -m differentiable_pricing.data.generate \
  --config configs/european_option_dataset_v1.toml \
  --output data/european-option-v1

python -m differentiable_pricing.data.diagnose \
  --dataset data/european-option-v1 \
  --output data/european-option-v1/diagnostics.json
```

Generation writes three splits plus a timestamp-free `manifest.json` recording
schema and generator versions, the seed derivation, the configuration SHA-256,
the oracle identity, units and conventions, row counts, and a SHA-256 per file,
so the same configuration and toolchain reproduce byte-identical output. Every
label comes from the compiled C++ oracle, and partitions are drawn from
independent streams rather than random-split from one table.

The sampling box is a **provisional synthetic engineering range, not a
calibrated market distribution**. Because the three factors are drawn
independently, a minority of rows land in the short-maturity, low-volatility,
deep-moneyness corner where the price underflows towards zero and delta
saturates; the manifest's `label_diagnostics` block counts them per partition.
Those **extreme-$z$ rows must be evaluated as their own slice**: they price to
within float64 noise of intrinsic value, so relative price error over them is
meaningless.

The diagnostic tool hashes *every* declared Parquet file **before** opening any
of them, then reports per-split statistics, standardized-moneyness bands,
rowwise no-arbitrage checks, and leakage checks on duplicate identifiers and
duplicate **economic states**. These are dataset diagnostics, not
model-performance metrics: a clean report is a precondition for an experiment,
not the result of one.

### Train, constrain, and evaluate

Training reads and verifies only `train.parquet` and `validation.parquet`; it
never opens `interpolation_test.parquet`.

```bash
python -m differentiable_pricing.ml.train \
  --dataset data/european-option-v1 \
  --config configs/european_neural_forward_differential_v1.toml \
  --output artifacts/european-neural-forward-differential-v1

python -m differentiable_pricing.ml.constrain \
  --artifact artifacts/european-neural-forward-differential-v1 \
  --output artifacts/european-neural-forward-differential-bounded-v1

python -m differentiable_pricing.ml.evaluate \
  --dataset data/european-option-v1 \
  --artifact artifacts/european-neural-forward-differential-bounded-v1 \
  --partition validation \
  --output artifacts/european-neural-forward-differential-bounded-v1/validation-evaluation.json
```

Substitute `european_neural_baseline_v1.toml`, `european_neural_long_v1.toml`,
`european_neural_wide_v2.toml`, or
`european_neural_forward_normalized_v1.toml` for the earlier arms.

Every run is CPU-only and float64, fits all feature and target scales on the
training partition alone, and uses only dense layers, `tanh` hidden
activations, and one linear output — matching the C++ `SmoothMlp` contract.
Reference Greek columns are never features, and artifacts follow the versioned
format in [docs/architecture.md](docs/architecture.md). Repeated training is
byte-identical in one fixed runtime, but PyTorch promises nothing across
releases or thread configurations, so a changed runtime is a new experiment
even with the same seed. `constrain` is not another training run: it verifies
the source artifact, copies `weights.npz` byte-for-byte, records the source
digests, and refuses to overwrite a directory or stack the same constraint
twice.

### The mathematics the code implements

The **forward-normalized representation** uses the exact homogeneity of the
Black–Scholes price to reduce seven raw inputs to three economic coordinates,
for spot $S$, strike $K$, maturity $T$, rate $r$, dividend yield $q$, and
volatility $\sigma$:

$$
F = S e^{(r-q)T}, \qquad
x = \log(F/K), \qquad
v = \sigma\sqrt{T}, \qquad
A = S e^{-qT}.
$$

The network learns $u = V/A = f(\text{option type}, x, v)$ and the wrapper
reconstructs $V = A u$ inside the PyTorch graph, so physical-unit Greeks still
follow from autograd. **Differential training** adds first-derivative
supervision — analytic delta and vega labels fix $u_x,u_v$ exactly — and the
**bounds projection** then clips the price into the discounted European
interval, changing the output contract only, never the weights. Every identity,
including those labels, the algebraic rho/theta reconstruction and the bound
definitions, is written out in
[docs/architecture.md](docs/architecture.md#stage-1-european-surrogate-model).

### What the controlled experiments showed

These values come from the same 25,000-row `validation` partition and therefore
describe **model selection, not a locked final evaluation**. Each row changes
one main factor relative to its control — the capacity arms hold seed, batch
size, optimizer, learning rate, weight decay, dtype, device, and feature order
fixed — and thresholds live in `configs/european_neural_acceptance_v1.toml`.

| Experiment | Main controlled change | Price/spot RMSE | Delta RMSE | Gamma RMSE | Material bound violations | Gate result |
|---|---|---:|---:|---:|---:|---|
| Raw baseline | 3×64, price-only, seven raw inputs | 0.003482 | 0.03577 | 0.006962 | 2,211 | Fail |
| Longer | Same model, 100 → 300 epochs | 0.001898 | 0.02236 | 0.005432 | 1,743 | Fail |
| Wider | Four 128-unit layers | 0.001515 | 0.02025 | 0.005368 | 1,393 | Fail |
| Forward | 3×64 on $(\text{type},\log(F/K),\sigma\sqrt{T})$ | 0.0009008 | 0.01995 | 0.006551 | 1,626 | Fail |
| Differential | Add analytic delta/vega supervision | 0.0002171 | 0.002536 | 0.001854 | 75 | Arbitrage only |
| Bounded | Project unchanged differential weights onto $[L,U]$ | **0.0002168** | **0.002474** | **0.001504** | **0** | **Pass** |

![Validation errors divided by their acceptance limits, on a logarithmic vertical axis, falling sharply at the differential-training step](docs/figures/european_validation_error_progression.svg)

![Material no-arbitrage violations across controlled experiments, dropping from 2,211 to zero](docs/figures/european_validation_arbitrage_progression.svg)

- **More of the same was not enough**: more epochs and more width improved
  price and delta, yet every gate still failed by a wide margin.
- **Financial coordinates mattered more than width for price**: forward
  normalization gave the strongest price-only improvement — and made gamma
  *worse*, so accurate prices do not imply accurate curvature.
- **Differential supervision was the decisive learning change**, cutting price,
  delta, vega, theta, and rho errors several-fold to nearly an order of
  magnitude, and carrying gamma — the one genuinely out-of-objective signal —
  across its gate.
- **Structure removed the remaining failure without retraining**: 76 of 25,000
  rows (`0.304%`) were projected, all onto the lower bound, weights
  byte-identical.

These versioned gates were used during the development study, but their
chronology is **not** independently established by repository history: the
configuration and the results arrive in the same change set. The fresh-seed
protocol fixes that by binding the same acceptance configuration by SHA-256 in
a protocol-only commit. Plotted inputs and intervention counts:
[`docs/results/european_validation_results_v1.json`](docs/results/european_validation_results_v1.json).

```bash
python scripts/plot_european_validation_results.py          # regenerate
python scripts/plot_european_validation_results.py --check  # CI staleness gate
```

### The frozen replication protocol

`configs/european_neural_replication_protocol_v1.toml` is the lock behind the
headline result: it binds the development result, both dataset configurations,
the training configuration, the acceptance gates, the projection, and the
one-shot final-evaluation policy. Both new seeds are the first unsigned 32 bits
of SHA-256 over public labels recorded in the protocol, so no favourable seed
could be chosen after seeing a result.

```bash
python scripts/check_european_replication_protocol.py       # validate the lock only
python scripts/run_european_replication.py run-to-validation
python scripts/run_european_replication.py status
python scripts/run_european_replication.py final-evaluate \
  --confirm-locked-final-evaluation                          # one-shot, terminal
python scripts/plot_european_replication_results.py --check
```

`run-to-validation` generates and diagnoses the fresh dataset, trains the frozen
model, derives the bounded artifact, and evaluates `validation` against every
frozen gate; it never touches `interpolation_test`. Pre-existing output paths
are a hard failure — there is no overwrite or resume flag — and the runner
records a final-evaluation attempt *before* starting it, so an interruption
still consumes the single permitted attempt. The audit ledger at
`runs/european-neural-replication-v1/execution.json` records commits, clean
`main`/`origin/main` state, runtime versions, commands and outputs, artifact
lineage, digests, and gate decisions.

**Protocol v1 is terminal.** Its final partition was consumed exactly once and
a future change needs a new protocol and a newly generated final partition
([docs/research-contract.md](docs/research-contract.md)).

Evaluation reports give MAE, RMSE, p95, p99, and maximum absolute error
overall, by call/put, and by standardized-moneyness band. Price error is
divided by spot, never by option price, because near-zero prices make that
ratio unstable. A constrained artifact additionally reports bound activations,
maximum adjustment, and the unconstrained network's own metrics, so the
intervention stays visible.

---

## Reproduce the American studies

All four studies are deliberately **outside CI** because their production-sized
grids are expensive. Reinstall the editable package after changing the C++
binding, then:

```bash
python -m differentiable_pricing.american.convergence \
  --config configs/american_crr_convergence_v1.toml \
  --output artifacts/american-crr-convergence-v1.json

python -m differentiable_pricing.american.lsm_crosscheck \
  --config configs/american_lsm_crosscheck_v1.toml \
  --output artifacts/american-lsm-crosscheck-v1.json

python -m differentiable_pricing.american.pde_refinement
```

The **CRR convergence report** covers named exercise regimes over a step ladder
against the internal 8,192/8,193-step average, with exercise-boundary samples,
analytic European controls, both signs of rate for no-dividend calls, and the
strict $0<p<1$ minimum-feasible-step distribution over a separate
$(T,r,q,\sigma)$ grid. It records build provenance but no wall-clock fields, so
numerical diagnostics never mix with machine-specific performance evidence. The
high-step average is a refinement reference from the same algorithm, not
independent truth — which is why the LSM cross-check exists
([docs/american-crr-contract.md](docs/american-crr-contract.md)).

The **LSM cross-check** reuses nine SHA-256-pinned CRR regimes and varies path
count, exercise-grid density, and polynomial degree one factor at a time. The
engine treats each antithetic pair average as one independent sample, reports
raw and control-variate standard errors with a 95% interval, and refuses a
training request whose estimated working set exceeds the configured limit.
Formulas, bias decomposition, RNG contract, regression method, and memory
accounting: [docs/american-lsm-contract.md](docs/american-lsm-contract.md).

The **PDE refinement study** reports how the oracle's price moves as the spot
grid, the time grid, and the truncated domain are refined, one axis at a time
and then jointly. It selects nothing.

Refreeze the LSM evidence and its figures only from a reviewed report:

```bash
python scripts/freeze_american_lsm_results.py \
  --report artifacts/american-lsm-crosscheck-review-fixed-v1.json \
  --output docs/results/american_lsm_crosscheck_results_v1.json --update
python scripts/plot_american_lsm_results.py

python scripts/freeze_american_lsm_results.py --check   # CI-safe, artifacts/ absent
python scripts/plot_american_lsm_results.py --check
```

### The label-policy pilot

The task 9C-B pilot is the most expensive study here — 2.19 hours of measured
single-core wall time — and is run only when its numerical evidence is
requested. The second command recomputes every price while reusing the first
run's measured timing block, so all deterministic JSON and CSV output can be
compared byte for byte:

```bash
python -m differentiable_pricing.american.pde_label_policy \
  --config configs/pde_label_policy_pilot_v1.toml \
  --output-directory artifacts/pde-label-policy-pilot-v1-run1
python -m differentiable_pricing.american.pde_label_policy \
  --config configs/pde_label_policy_pilot_v1.toml \
  --output-directory artifacts/pde-label-policy-pilot-v1-run2 \
  --performance-source artifacts/pde-label-policy-pilot-v1-run1/report.json \
  --verify-identical-to artifacts/pde-label-policy-pilot-v1-run1
```

Its frozen outcome is already in the repository, so nothing in CI or the local
gate reruns it; the checked-in snapshot is digest-pinned by
`python/tests/test_pde_label_policy_results_snapshot.py`.

### Benchmarking the CRR paths

```bash
python scripts/benchmark_american_crr.py \
  --config configs/american_crr_convergence_v1.toml \
  --steps 1024 --batch-sizes 1,16,64 --thread-counts 1,2,4 \
  --warmups 1 --repetitions 5 \
  --output artifacts/american-crr-benchmark-local.json
```

The report records every repetition plus CPU affinity, platform, compiler,
build configuration, batch size, and effective worker count, and requires
identical prices from every configuration. Timings are never checked in or used
as CI gates.

---

## Reproduce the market studies

Both need the private archive and the `market` extra, both write only beneath
ignored trees, and neither runs in CI: the market tests build vendor-encoded
fixtures programmatically and skip decode-backed cases when the extra is
absent.

```bash
python -m pip install -e '.[dev,market]'

python -m differentiable_pricing.market.ingest \
  --config configs/market_feasibility_v1.toml \
  --output artifacts/market-feasibility-v1-audit3/feasibility-report-v3.json

python -m differentiable_pricing.market.reconstruct \
  --config configs/market_state_reconstruction_v1.toml
```

The reconstruction verifies the 9A configuration, report, and raw-manifest
digests before any arithmetic and aborts with nothing written on a mismatch.
Its report, tables, and figures are derived from proprietary quote-level data:
never stage them, nor any fitted curve or inferred market value.

---

## Methodology, contracts, and frozen evidence

| Document | Contents |
|---|---|
| [docs/research-contract.md](docs/research-contract.md) | Hypotheses, stage reference standards, data protocol, metrics, replication gates, non-claims |
| [docs/architecture.md](docs/architecture.md) | Language boundary, snapshot discipline, artifact contract, stage-1 model math |
| [docs/american-crr-contract.md](docs/american-crr-contract.md) | Lattice, recursion, exercise metadata, complexity, convergence semantics |
| [docs/american-lsm-contract.md](docs/american-lsm-contract.md) | Estimand, policy/valuation separation, uncertainty, control variate, overflow rejection |
| [docs/pde-numerical-contract.md](docs/pde-numerical-contract.md) | Discrete-dividend PDE oracle: equation, discount interpolation, dividend jump, boundaries, PSOR residual, complexity, scope, and the 9C-B pilot design |
| [docs/market-state-reconstruction-contract.md](docs/market-state-reconstruction-contract.md) | Parity fitting, identifiability classes, forbidden names, task 9C input contract |
| [docs/agentic-workflow.md](docs/agentic-workflow.md) | Agent roles, guardrails, review loop |
| [CLAUDE.md](CLAUDE.md) | Operating contract: commands, numerical non-negotiables, coding rules |

Three disciplines run through all of them.

**Partition roles are enforced, not assumed.** `validation` selects
architecture and hyperparameters; `interpolation_test` must be named explicitly
and is never tuned against. Once any result informs a model change that split is
consumed and a fresh-seed replication is required for the final estimate.

**Expensive evidence is frozen, not transcribed.** A snapshot family's
generator checks the raw report against an exact key schema, recomputes every
summary from its own rows, extracts each number programmatically, and emits
canonical timestamp-free JSON. Every `--check` runs in CI on checked-in files
only, so no gate depends on an ignored artifact.

**Agent-assisted engineering is part of the method, and it is constrained.** A
deterministic post-edit hook validates edited files without rewriting them, and
two read-only clean-context agents (`code-reviewer`, `numerical-reviewer`)
return findings on a diff without touching it. They produce evidence, not
authority: deterministic tests, CI, and human ownership of assumptions and
claims remain decisive
([docs/agentic-workflow.md](docs/agentic-workflow.md)).

---

## Limitations and non-claims

- **Every learned result is synthetic.** No market data, quoted price, or
  calibration target enters any dataset, training run, or reported metric. The
  market track reads real quotes but produces no price, Greek, label, or
  calibrated model.
- **No calibrated curve or volatility surface exists.** Task 9B identifies
  discount and forward *knots* pointwise at quoted expiries; nothing
  interpolates a curve through them, no implied volatility is inverted, and no
  surface is fitted.
- **The real-market evidence is three sessions.** It is a feasibility and
  identifiability study, not a historical market study, and it establishes no
  SPY American calibration capability. Dividend amounts, borrow and carry, and
  corporate-action adjustments are unavailable in that archive; the SPY
  ex-*date* is officially scheduled and its amount is unverified.
- **No production label policy exists.** The task 9C-B pilot returned
  `no_policy_selected`, so American dataset generation and American neural
  training remain blocked.
- **In-envelope interpolation only.** No boundary, extrapolation/OOD, or
  scenario-shock partition has been built or evaluated.
- **No latency claim.** Analytic Black–Scholes can easily be faster than this
  network. A speed comparison only becomes meaningful for genuinely expensive
  references — trees, PDE, LSM — and only end-to-end, with feature transforms,
  serialization, batching, and derivative cost all counted. The pilot's worker
  projections assume ideal scaling and are not feasibility claims.
- **No production or deployment claim.** C++ artifact loading is not
  implemented; the shipped inference core is `SmoothMlp` alone.
- **No American Greek, dataset, or neural result yet**, and no Greek validation
  independent of the analytic reference. The PDE oracle exposes prices only;
  its pilot Greeks came from external bump-and-reprice.

---

## Next milestone: task 9C-C1

The pilot located the cost precisely: four labels per state took thirteen
scalar solves at one grid — a center plus three symmetric bump pairs per axis —
and every one of them discards the whole valuation-time solution to keep a
single number. **Task 9C-C1 is designed and not implemented.** It will:

- expose the valuation-time value slice $V(S)$ from one PDE solve;
- derive delta and gamma from that slice, so price and both spatial Greeks come
  from one internally consistent solution instead of independent solves;
- expose the exercise/continuation classification already computed by the
  obstacle solve;
- predeclare a Greek-eligibility rule around nonsmooth and free-boundary
  stencils, since the pilot showed the reference Greek itself is unstable
  exactly there;
- preserve the existing scalar API and its $O(N_S)$ working memory.

The sequence after it, in order and none of it implemented: **9C-C2**, a
three-surface vega with deterministic batching and parallel throughput;
**9C-C3**, a predeclared label-policy v2 evaluated on a small remediation set,
followed only conditionally by a full confirmation run; then American dataset
generation and the European→American transfer experiment; and only after that,
evaluation of price, Greek, latency, implied-volatility and surface-calibration
behaviour against real quotes.

Nothing in that list exists today: no surface extraction, no surface batching,
no vega surface, no parallel sharding, no faster LCP solver, no American
training dataset, no American surrogate, no implied-volatility inversion, and
no volatility-surface calibration.

---

## Contributing

Read [CLAUDE.md](CLAUDE.md) and
[docs/research-contract.md](docs/research-contract.md) first. Every change
should state the numerical assumption it changes, add a regression test, and
pass `./scripts/check.sh`. Full workflow, test-partition rules, snapshot
refresh procedure, and pull-request expectations:
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT; see [LICENSE](LICENSE).
