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
partition that has informed no model choice. Stage-by-stage reference standards
and the full experimental contract:
[docs/research-contract.md](docs/research-contract.md).

**Stage 1 is complete and independently replicated. Stage 2 has both of its
reference pricers built and cross-checked, and no neural work yet.**

---

## Headline results

### Stage 1 complete: fresh-seed European replication

A precommitted protocol was executed once, end to end, on a **newly generated
250,000-row dataset and a new initialization seed**. Architecture, objective,
domain, row counts, optimizer, training budget, evaluation bands, output
constraint, and every acceptance gate were bound by SHA-256 before the data
existed. Training and checkpoint selection saw only `train` and `validation`;
the fresh `interpolation_test` was evaluated once and is now permanently
consumed.

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
of the frozen allowance — a legitimate pass, since the threshold was committed
before the fresh dataset existed, but curvature is the first metric to stress
in any boundary or out-of-domain study.

**Zero arbitrage violations is structural, not learned.** The trained network
alone violated the discounted lower bound materially on 1,584 locked-final
rows; the deterministic European-bounds projection adjusted 1,586 prices (6.34%
of the partition), all at the lower bound, driving material violations to zero.
Bounded and unconstrained artifacts share the identical weight digest
`42670774f736383e50818b6e6c1db9374a77988173e35423ffc34b3c4297ecb8`.

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
records commits, seeds, toolchain, dataset/report/artifact digests,
byte-identical weight lineage, every metric, gate decisions, and the projection
intervention. The execution ledger records `final_evaluation_attempts = 1`,
`status = replication_passed`, a clean `main` worktree, and source commit
`5c47081e265a4f1256b733d398079ca02f2f290e`; the untracked evidence bundle's
reviewed SHA-256 is
`86859008443a41d86c310c45db9abf784cb3131a32d4a83dca13d9e4a68afb29`. Digests are
traceability, not an authenticated attestation of who ran the experiment.

### Stage 2 in progress: two independent American references agree

American work is the active second stage and is so far entirely about
*reference pricers*, not networks. Two numerically unrelated methods price the
same nine pinned regimes: a deterministic **Cox–Ross–Rubinstein tree** refined
to an 8,192/8,193-step adjacent average, and an independently structured
**Longstaff–Schwartz Monte Carlo** engine that fits a stopping policy on one
antithetic path stream and values that frozen policy on a second. Seven
experiments, 63 case rows; `CRR−LSM` is in price units against the CRR adjacent
average.

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
is the expected ordering, not a defect: LSM values a *fixed learned Bermudan
stopping policy*, and for a fixed policy the expectation is no greater than the
same-grid optimal stopping value. Three sources of difference must be kept
apart, and the reported interval measures only the first:

* **Valuation noise** — sampling error around an already-frozen policy, the
  `Mean valuation-only SE` column. It shrinks as $n^{-1/2}$: along the path
  ladder it falls 0.02508 → 0.01799 → 0.01263, ratios of 1.3946 and 1.4244
  against the $\sqrt{2}\approx1.4142$ a doubling implies, deviations of −1.4%
  and +0.7%. (Those ratios use full snapshot precision; recomputing them from
  the five-decimal column gives 1.394 and 1.424.)
* **Policy-fitting bias** — the learned rule is not optimal for its own grid.
* **Exercise-grid bias** — a 64-date Bermudan is not a continuously exercisable
  American.

The last two are systematic and do not shrink with valuation paths, so the
interval has no nominal coverage rate against the CRR reference and
`In-interval` is a diagnostic, **not an acceptance gate**: it falls 4 → 2 → 0
along the ladder precisely because added paths narrow the interval around a
fixed bias. Deterministic zero-width cases cannot contain a finite-step tree
value however close the agreement, so they are reported separately rather than
converted into successes by an invented tolerance. Interval containment and
point-estimate agreement are different questions — containment falls to zero
while the point estimates keep agreeing to roughly 0.02 price units.

#### Sensitivity findings

![Sensitivity of the mean CRR-minus-LSM gap to basis degree, exercise-grid density, and path count](docs/figures/american_lsm_sensitivity.svg)

* **Degree one is inadequate**: at 64 dates and 32,768 training paths the
  linear basis gives a mean gap of 0.60422 and a worst case of 2.43419, more
  than twenty times the quadratic arm. **Quadratic and cubic are comparable**
  here — 0.02713 versus 0.02828, a difference more than an order of magnitude
  below either arm's ≈0.018 mean valuation-only standard error. With one
  training seed per experiment that means "not separated by valuation noise",
  not "replicated as equal".
* **More exercise dates did not help at a fixed path budget**: 32 → 64 → 128
  dates moved the mean gap 0.02808 → 0.02713 → 0.03298, because refining the
  grid adds regression dates without adding paths to fit them on.
* **More paths cut valuation noise but improved the gap only modestly**: along
  the 16,384 → 32,768 → 65,536 training-path ladder it moves 0.02513 → 0.02713
  → 0.02380 — real, small, and **not monotone**, consistent with a residual
  systematic bias.
* **Every applicable finite control variate reduced variance**, minimum ratio
  1.0001; deterministic cases are excluded from that minimum (undefined 0/0)
  and variance-removing controls count as infinite rather than folded in.
* **Regression fallbacks appear in sparse deep-OTM regions**: 4 in the primary
  experiment, all in `deep_otm_put`, ranging from 2 to 9 across experiments.

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
label policy**: task 8E separately predeclares and runs the CRR label-policy
calibration study, and nothing here may be reused as its acceptance criterion.

Per-case rows, provenance digests, and declared limitations:
[docs/results/american_lsm_crosscheck_results_v1.json](docs/results/american_lsm_crosscheck_results_v1.json).

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
| American label policy | **Not started — Task 8E** | gates fixed before results |
| American dataset, training, transfer; swaption stages 3–4 | Not started | blocked on the label policy |
| C++ artifact loading and deployment | Not implemented | `SmoothMlp` inference only |
| Latency claims; market data, curves, surfaces, OOD | Out of scope so far | benchmarks are evidence, not gates |

No generated dataset, weight file, or raw study report is checked in. The
repository carries versioned snapshots under `docs/results/` and deterministic,
CI-checked SVG figures under `docs/figures/` rendered from those snapshots
alone.

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
valuation, input validation, and reverse-mode derivatives of the deployed
network. **Python** owns sampling, partitioning, lineage, training, and
evaluation. Pricing formulas are never duplicated in Python to make a test
pass. [docs/architecture.md](docs/architecture.md).

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
`python/tests/ml/`; for dataset work without PyTorch install `.[data,dev]`. The
gate needs the editable install — a missing pytest fails it rather than
skipping the Python suite — and `--quick`, used by the pre-commit hook, stops
after the C++ tests. Single-configuration generators default to
`RelWithDebInfo` because the CRR reference is quadratic in its step count; use
`-DCMAKE_BUILD_TYPE=Release` for any recorded performance experiment.

The pricers emit machine-readable JSON:

```bash
./build/dev/dp_pricer call 100 100 1 0.05 0 0.20
./build/dev/dp_american_pricer american put 100 100 1 0.05 0 0.20 2048
```

The American executable reports the $N$- and $(N+1)$-step tree prices, their
average, their absolute gap, and exercise-region diagnostics. That adjacent gap
is a refinement diagnostic that also absorbs time-mesh and exercise-frontier
changes: it is **not** a pure parity measure or a certified error bound
([docs/american-crr-contract.md](docs/american-crr-contract.md)).

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

Generation writes the three splits plus a `manifest.json` recording schema and
generator versions, the base seed and stream derivation, the configuration
SHA-256, the oracle identity, units and conventions, row counts, and a SHA-256
per file. It has no wall-clock timestamp, so the same configuration and
toolchain reproduce byte-identical output; treat a different toolchain as a new
dataset until hashes are compared. Every label comes from the compiled C++
oracle, and partitions are drawn from independent streams rather than
random-split from one table.

The sampling box is a **provisional synthetic engineering range, not a
calibrated market distribution**. Because the three factors are drawn
independently, a minority of rows land in the short-maturity, low-volatility,
deep-moneyness corner where the price underflows towards zero and delta
saturates; the manifest's `label_diagnostics` block counts them per partition
so downstream metrics can stratify rather than average them away. The boundary
partition those rows belong in is still to be built.

The diagnostic tool hashes *every* declared Parquet file **before** opening any
of them, then writes one deterministic, timestamp-free report: per-split
statistics, standardized-moneyness bands over
`absolute_z = |log_forward_moneyness| / (volatility * sqrt(maturity))`, rowwise
no-arbitrage checks at a documented float64 tolerance of `1e-12` relative to
`max(discounted_spot, discounted_strike)`, and leakage checks on duplicate
identifiers and duplicate **economic states** (exact float64 equality over the
seven input columns; near-duplicate detection needs a declared quantization
this project has not fixed). It exits `0` with no findings, `3` when something
was reported, and `2` when the dataset cannot be trusted enough to describe.

These are **dataset diagnostics, not model-performance metrics**: a clean
report is a precondition for an experiment, not the result of one. And
**extreme-$z$ rows must be evaluated as their own slice** — they price to
within float64 noise of intrinsic value, so relative price error over them is
meaningless; do not delete them and do not average them away.

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
Reference Greek columns are never features. Artifacts follow the versioned
format in [docs/architecture.md](docs/architecture.md): canonical metadata plus
SHA-256-linked, non-pickled `weights.npz` loaded with `allow_pickle=False`.
Seeds, thread count, and library versions are recorded, and repeated training
is tested for byte-identical output in one fixed runtime — but PyTorch does not
promise identical results across releases or thread configurations, so a
changed runtime is a new experiment even with the same seed.

`constrain` is not another training run: it verifies the source artifact,
copies `weights.npz` byte-for-byte, records the source digests, publishes
atomically, and refuses to overwrite a directory or stack the same constraint
twice. Source and derived artifacts must report the same weight digest.

### The mathematics the code implements

Features and targets are standardized with statistics from `train` only, and
the physical wrapper undoes the target transform inside the autograd graph.

The **forward-normalized representation** uses the exact homogeneity of the
Black–Scholes price to reduce seven raw inputs to three economic coordinates:

$$
F = S e^{(r-q)T}, \qquad
x = \log(F/K), \qquad
v = \sigma\sqrt{T}, \qquad
A = S e^{-qT},
$$

for spot $S$, strike $K$, maturity $T$, rate $r$, dividend yield $q$, and
volatility $\sigma$. The network learns the dimensionless
$u = V/A = f(\text{option type}, x, v)$ and the wrapper reconstructs $V = A u$,
all inside the PyTorch graph, so physical-unit Greeks still follow from
autograd.

**Differential training** adds first-derivative supervision. The analytic delta
and vega labels fix the coordinate derivatives $u_x,u_v$ exactly; converted to
the standardized coordinates $z_x,z_v$, the configured loss is

$$
\operatorname{MSE}(\hat y,y)
+\frac{1}{2}\left[
  \operatorname{MSE}(\partial_{z_x}\hat y,\partial_{z_x}y)
  +\operatorname{MSE}(\partial_{z_v}\hat y,\partial_{z_v}y)
\right].
$$

Those labels are analytic in the declared coordinates; finite differences are
never used as training labels. Rho and theta are then fixed algebraically by
$(u,u_x,u_v)$, which is why the Greek-evidence table above treats them as
reconstruction checks rather than independent derivative learning.

The **European no-arbitrage bounds projection** changes only the output
contract — not the data, objective, architecture, or weights — clipping the
network price into the discounted European interval $[L,U]$ built from
$A=Se^{-qT}$ and $B=Ke^{-rT}$. The Black–Scholes price lies inside that
interval, so projection cannot increase pointwise absolute price error. It is
piecewise-differentiable: away from a clipping boundary autograd returns either
the learned-network Greeks or the active bound's physical derivatives, while at
that boundary and at the intrinsic-value kink the derivative is not unique — a
limitation the artifact and the evaluation report both state. The projection is
a versioned part of the model, not a reporting adjustment.

Every identity above, including the $u_x,u_v$ labels, the rho/theta algebra,
and the bound definitions, is written out in
[docs/architecture.md](docs/architecture.md#stage-1-european-surrogate-model).

### What the controlled experiments showed

These values come from the same 25,000-row `validation` partition and therefore
describe **model selection, not a locked final evaluation**. Each row changes
one main factor relative to its control; thresholds live in
`configs/european_neural_acceptance_v1.toml`.

| Experiment | Main controlled change | Price/spot RMSE | Delta RMSE | Gamma RMSE | Material bound violations | Gate result |
|---|---|---:|---:|---:|---:|---|
| Raw baseline | 3×64, price-only, seven raw inputs | 0.003482 | 0.03577 | 0.006962 | 2,211 | Fail |
| Longer | Same model, 100 → 300 epochs | 0.001898 | 0.02236 | 0.005432 | 1,743 | Fail |
| Wider | Four 128-unit layers | 0.001515 | 0.02025 | 0.005368 | 1,393 | Fail |
| Forward | 3×64 on $(\text{type},\log(F/K),\sigma\sqrt{T})$ | 0.0009008 | 0.01995 | 0.006551 | 1,626 | Fail |
| Differential | Add analytic delta/vega supervision | 0.0002171 | 0.002536 | 0.001854 | 75 | Arbitrage only |
| Bounded | Project unchanged differential weights onto $[L,U]$ | **0.0002168** | **0.002474** | **0.001504** | **0** | **Pass** |

The capacity arms held seed, batch size, optimizer, learning rate, weight
decay, dtype, device, and feature order fixed, varying only depth/width
(`european_neural_long_v1.toml`, 64×64×64 for 300 epochs;
`european_neural_wide_v2.toml`, four 128-unit layers) against the 100-epoch
`european_neural_baseline_v1.toml`.

![Validation errors divided by their acceptance limits, on a logarithmic vertical axis, falling sharply at the differential-training step](docs/figures/european_validation_error_progression.svg)

![Material no-arbitrage violations across controlled experiments, dropping from 2,211 to zero](docs/figures/european_validation_arbitrage_progression.svg)

- **More of the same was not enough**: 100 → 300 epochs roughly halved price
  error, and the wider model improved price and delta again, yet every gate
  still failed by a wide margin.
- **Financial coordinates mattered more than width for price**: forward
  normalization gave the strongest price-only improvement — and made gamma
  *worse*, so accurate prices do not imply accurate curvature.
- **Differential supervision was the decisive learning change**, cutting price,
  delta, vega, theta, and rho errors several-fold to nearly an order of
  magnitude. Gamma also crossed its gate, the one genuinely out-of-objective
  signal in that row.
- **Structure removed the remaining failure without retraining**: 76 of 25,000
  rows (`0.304%`) were projected, all onto the lower bound, maximum adjustment
  `0.102481`, weights byte-identical. Violations fell from 75 to zero while
  every headline metric improved slightly.

Gamma was already the tightest headline gate here, at about 75% of its allowed
RMSE. These versioned gates were used during the development study, but their
chronology is **not** independently established by repository history: the
configuration and the results arrive in the same change set. The fresh-seed
protocol fixes that by binding the same acceptance configuration by SHA-256 in
a protocol-only commit. Plotted inputs, weight digests, and intervention
counts:
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

Evaluation reports link the dataset manifest, training configuration, artifact
manifest, weight digests, and output constraint, and give MAE, RMSE, p95, p99,
and maximum absolute error overall, by call/put, and by standardized-moneyness
band. Price error is also divided by spot, but never by option price, because
near-zero prices make that ratio unstable. No-arbitrage violations are counted
at both the float64 reporting tolerance and a material relative tolerance of
`1e-6`, and a constrained artifact adds bound activations, maximum adjustment,
and the unconstrained network's own metrics so the intervention stays visible.
Reports label `validation` as `model_selection` and `interpolation_test` as
`locked_final_evaluation`.

---

## Reproduce the American study

Both studies are deliberately **outside CI** because their production-sized
grids are expensive. Reinstall the editable package after changing the C++
binding, then:

```bash
python -m differentiable_pricing.american.convergence \
  --config configs/american_crr_convergence_v1.toml \
  --output artifacts/american-crr-convergence-v1.json

python -m differentiable_pricing.american.lsm_crosscheck \
  --config configs/american_lsm_crosscheck_v1.toml \
  --output artifacts/american-lsm-crosscheck-v1.json
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
training request whose estimated working set exceeds the configured limit. For
any admitted input it either returns fully finite results or raises; it never
returns a NaN price or a corrupted standard error. Formulas, bias
decomposition, RNG contract, regression method, and memory accounting:
[docs/american-lsm-contract.md](docs/american-lsm-contract.md).

Refreeze the evidence and its figures only from a reviewed report:

```bash
python scripts/freeze_american_lsm_results.py \
  --report artifacts/american-lsm-crosscheck-review-fixed-v1.json \
  --output docs/results/american_lsm_crosscheck_results_v1.json --update
python scripts/plot_american_lsm_results.py

python scripts/freeze_american_lsm_results.py --check   # CI-safe, artifacts/ absent
python scripts/plot_american_lsm_results.py --check
```

### Benchmarking the CRR paths

```bash
python scripts/benchmark_american_crr.py \
  --config configs/american_crr_convergence_v1.toml \
  --steps 1024 --batch-sizes 1,16,64 --thread-counts 1,2,4 \
  --warmups 1 --repetitions 5 \
  --output artifacts/american-crr-benchmark-local.json
```

The report records every repetition plus CPU affinity, platform, compiler,
build configuration, batch size, and effective worker count, rotates scalar and
batch modes to reduce fixed-order thermal bias, and requires identical prices
from every configuration. Timings are never checked in or used as CI gates. The
recombining tree remains $O(N^2)$ because it visits $N(N+1)/2$ nodes; batch
workers cut elapsed time but not total work, so the relevant baseline is the
smallest convergence-supported $N$ built in `Release`.

---

## Methodology, contracts, and frozen evidence

| Document | Contents |
|---|---|
| [docs/research-contract.md](docs/research-contract.md) | Hypotheses, stage reference standards, data protocol, metrics, replication gates, non-claims |
| [docs/architecture.md](docs/architecture.md) | Language boundary, snapshot discipline, artifact contract, stage-1 model math |
| [docs/american-crr-contract.md](docs/american-crr-contract.md) | Lattice, recursion, exercise metadata, complexity, convergence semantics |
| [docs/american-lsm-contract.md](docs/american-lsm-contract.md) | Estimand, policy/valuation separation, uncertainty, control variate, overflow rejection |
| [docs/agentic-workflow.md](docs/agentic-workflow.md) | Agent roles, guardrails, review loop |
| [CLAUDE.md](CLAUDE.md) | Operating contract: commands, numerical non-negotiables, coding rules |

Three disciplines run through all of them.

**Partition roles are enforced, not assumed.** `validation` selects
architecture and hyperparameters; `interpolation_test` must be named explicitly
and is never tuned against. Once any result informs a model change that split is
consumed and a fresh-seed replication is required for the final estimate.

**Expensive evidence is frozen, not transcribed.** One generator/validator per
snapshot family checks the raw report against an exact key schema, recomputes
every summary from its own rows, extracts each number programmatically, and
emits canonical timestamp-free JSON. Every `--check` runs in CI on checked-in
files only, so no gate depends on an ignored artifact.

**Agent-assisted engineering is part of the method, and it is constrained.** A
deterministic post-edit hook validates edited Python, JSON, TOML, and C++
formatting without rewriting files, and two read-only clean-context agents
(`code-reviewer`, `numerical-reviewer`) return findings on a diff without
touching it. They produce evidence, not authority: deterministic tests, CI, and
human ownership of assumptions and claims remain decisive
([docs/agentic-workflow.md](docs/agentic-workflow.md)).

---

## Limitations

- **Synthetic only.** No market data, quoted price, or calibration target
  enters any result here; there is no calibrated curve and no
  implied-volatility surface.
- **In-envelope interpolation only.** No boundary, extrapolation/OOD, or
  scenario-shock partition has been built or evaluated.
- **No latency claim.** Analytic Black–Scholes can easily be faster than this
  network. A speed comparison only becomes meaningful for genuinely expensive
  references — trees, PDE, LSM — and only end-to-end, with feature transforms,
  serialization, batching, and derivative cost all counted.
- **No production or deployment claim.** C++ artifact loading is not
  implemented; the shipped inference core is `SmoothMlp` alone.
- **No Greek validation independent of the analytic reference.** Greeks are
  compared with Black–Scholes closed forms in the declared model.
- **No American Greek, dataset, or neural result exists yet.** The stage-2
  evidence is agreement between two numerical methods inside one model.
- Digests are traceability to reviewed external evidence, not an authenticated
  attestation of who executed a run.

---

## Next milestone: Task 8E

The next task is the **predeclared CRR label-policy calibration study**: fix,
before any results are observed, the step policy and reference-error budget
that American training labels must satisfy across the complete proposed
sampling domain.

**No American dataset generation and no American neural training may begin
until that label policy is frozen**, and nothing frozen by the task 8D
cross-check may be reused as a task 8E acceptance criterion. Only then should
exercise-aware train, validation, boundary, and locked-test partitions be
built, and only then should the European→American transfer experiment start.

LSM does not replace the tree in this one-dimensional problem; it is
independent validation and preparation for path-dependent or higher-dimensional
products where recombination stops keeping the state space tractable.

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
