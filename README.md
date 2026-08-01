# Differentiable Pricing

A research platform for testing whether neural surrogates can reproduce
derivative prices and useful sensitivities at materially lower inference
latency than their reference pricers.

The project deliberately starts with a result we can falsify. The first
oracle is a C++20 Black--Scholes implementation with analytic Greeks. A small,
smooth C++ neural network can return both its value and the exact derivative
of that network with respect to its inputs. Python will own data generation,
training, experiment tracking, and model export.

> The derivative of a surrogate is exact **for the learned network**, not
> automatically an exact market Greek. It becomes a credible Greek only after
> normalization is reversed and the result is validated against analytic,
> algorithmic, or high-accuracy bump-and-revalue references.

## Research sequence

| Stage | Reference pricer | Why it exists | Transfer experiment |
|---|---|---|---|
| 1. European option | Black--Scholes, then Monte Carlo | Prove price and Greek learning against analytic truth | Baseline representation |
| 2. American option | High-step CRR tree, then LSM | Add an exercise boundary and non-smooth behaviour | European weights vs random initialization |
| 3. European swaption | Hull--White 1F, later G2++ | Add curves, rates conventions, and model calibration | Rates baseline |
| 4. Bermudan swaption | Tree/PDE or carefully validated LSM | Add multiple exercise dates and expensive labels | European-swaption weights vs random initialization |

Each stage must beat fixed acceptance criteria on an untouched,
distribution-aware test set before the next stage is treated as credible. The
full experimental contract is in
[docs/research-contract.md](docs/research-contract.md).

## What works now

- C++20 Black--Scholes call/put prices and analytic delta, gamma, vega, theta,
  and rho.
- A deterministic C++20 Cox--Ross--Rubinstein tree for European and American
  calls and puts, with adjacent-step convergence diagnostics and
  early-exercise-region metadata.
- A price-only CRR batch boundary with deterministic ordering, indexed
  preflight failures, explicit worker counts, reusable per-worker \(O(N)\)
  memory, and bit-identical serial/parallel prices.
- A deterministic C++20 Longstaff--Schwartz cross-check with separate
  policy-training and valuation paths, antithetic pair-aware uncertainty,
  analytic European control variates, stable continuation regression, an
  explicit training-memory guard, and total price-domain overflow rejection
  in place of silent non-finite results.
- A command-line pricer with machine-readable JSON output.
- A scalar-output `tanh` MLP implemented in C++, including a manual
  reverse-mode pass for input derivatives.
- Python bindings for both components through pybind11.
- C++ unit tests for known values, put--call parity, analytic delta, invalid
  inputs, and the MLP reverse pass.
- Python parity tests for the installed extension.
- A deterministic, versioned European-option dataset generator whose only
  pricing oracle is the compiled C++ binding.
- A deterministic dataset diagnostic tool: hash verification, per-split
  distribution statistics, standardized-moneyness stratification, and rowwise
  no-arbitrage bound checks.
- A deterministic CPU/float64 price-training baseline with train-only
  normalization, validation early stopping, and versioned integrity-checked
  artifacts.
- A forward-normalized differential-training experiment that supervises price,
  delta, and vega information while leaving gamma as an out-of-objective test.
- A deterministic, piecewise-differentiable European-bounds projection that
  can derive a new artifact from verified weights without retraining.
- A separate untouched-test evaluator that differentiates physical price
  through the saved transforms and reports price, delta, gamma, vega, theta,
  and rho error by option type and standardized-moneyness band.
- A precommitted fresh-seed replication protocol and guarded execution runner
  that consumed the locked final partition exactly once and passed all 11
  frozen acceptance gates.
- An immutable replication snapshot with full SHA-256 lineage and
  deterministic, CI-checked SVG figures.
- Deterministic repository checks, CI, a Claude Code post-edit hook, and two
  read-only clean-context review agents.

No generated dataset or trained weight file is checked into the repository.
Small, versioned snapshots and deterministic plots document both the
model-selection path and the completed fresh-seed replication. The raw
datasets, weights, reports, and execution ledger remain external evidence
identified by SHA-256. The replication is a locked final result for synthetic
in-envelope European options, not a market-pricing, OOD, latency, or production
claim.

## Repository map

```text
cpp/                    C++ pricing and inference core
bindings/python/        pybind11 boundary
python/                 Python package and tests
configs/                Versioned experiment assumptions
docs/                   Architecture and research contract
docs/results/           Versioned development and replication summaries
docs/figures/           Deterministically rendered result plots
.claude/                Claude Code hook and review agents
.githooks/              Optional local Git hooks
.github/workflows/      Continuous integration
scripts/                Reproducible developer checks
```

## Prerequisites

- CMake 3.20 or later
- A C++20 compiler (recent GCC, Clang, or MSVC)
- Python 3.11 or later
- Git
- Optional: GitHub CLI (`gh`) for one-command remote creation

## Build the C++ core

From the repository root:

```bash
cmake -S . -B build/dev \
  -DDP_BUILD_PYTHON_BINDINGS=OFF \
  -DDP_WARNINGS_AS_ERRORS=ON
cmake --build build/dev --parallel
ctest --test-dir build/dev --output-on-failure
```

Single-configuration generators default to `RelWithDebInfo`, because the CRR
reference is quadratic in its step count. Use
`-DCMAKE_BUILD_TYPE=Release` for a recorded performance experiment; latency
claims must still include hardware and repeated-run metadata.

Try the executable:

```bash
./build/dev/dp_pricer call 100 100 1 0.05 0 0.20
./build/dev/dp_american_pricer american put 100 100 1 0.05 0 0.20 2048
```

The American executable reports the raw \(N\)-step tree price, the
\((N+1)\)-step price, their average, their absolute gap, and exercise-region
diagnostics. Adjacent-step averaging reduces the visible even/odd
strike-alignment oscillation. For an American tree, however, the adjacent
discrepancy also includes time-mesh and exercise-frontier changes; it is
**not** a pure parity measure or certified error bound. A label-generation
configuration must establish convergence over its entire declared domain
before treating a step count as adequate. The numerical contract and formulas
are in
[docs/american-crr-contract.md](docs/american-crr-contract.md).

### Run the American CRR convergence study

Reinstall the editable package after changing the C++ binding, then run the
versioned exploratory study:

```bash
python -m pip install -e '.[dev,train]'
python -m differentiable_pricing.american.convergence \
  --config configs/american_crr_convergence_v1.toml \
  --output artifacts/american-crr-convergence-v1.json
```

The report contains:

- American adjacent-step prices over a declared step ladder;
- absolute differences from the internal 8,192/8,193-step CRR average;
- exercise-region snapshots over a separate diagnostic step ladder, including
  boundary samples at declared physical-time fractions;
- European-tree comparisons with analytic Black--Scholes prices;
- explicit positive-rate no-dividend-call and negative-rate-call exercise
  controls;
- the strict \(0<p<1\) minimum-feasible-candidate-step distribution over a
  separate \((T,r,q,\sigma)\) grid;
- compiler, build configuration, and a composite hash of every C++/binding
  source file that affects the recorded CRR and analytic-control results;
- no wall-clock fields, so numerical diagnostics are not mixed with
  machine-specific performance evidence.

The high-step CRR average is a refinement reference from the same algorithm,
not exact or independent truth. The independent LSM protocol below
cross-checks it, and both methods' observed errors must inform a separately
versioned label policy rather than silently becoming one.

### Benchmark the scalar and parallel batch paths

```bash
python scripts/benchmark_american_crr.py \
  --config configs/american_crr_convergence_v1.toml \
  --steps 1024 \
  --batch-sizes 1,16,64 \
  --thread-counts 1,2,4 \
  --warmups 1 \
  --repetitions 5 \
  --output artifacts/american-crr-benchmark-local.json
```

The benchmark records every repetition, CPU affinity, platform, C++ compiler,
build configuration, batch size, requested thread count, and effective worker
count. Scalar and batch modes are cyclically rotated within repetitions to
reduce fixed-order thermal bias. Every configuration must return exactly
identical prices. Timings are deliberately not checked into source control or
used as CI gates: a later neural comparison must use the same hardware,
affinity, batch size, accuracy target, and effective core budget.

The recombining tree remains \(O(N^2)\) in arithmetic because it visits
\(N(N+1)/2\) nodes. Batch workers reduce elapsed time and reuse allocations;
they do not change the total work. The relevant reference baseline is
therefore the smallest convergence-supported \(N\), built in `Release`, rather
than an arbitrary large tree.

### Run the independent LSM cross-check

The LSM engine learns a stopping policy on one antithetic path stream and
evaluates that frozen policy on a second stream. It treats each antithetic pair
average as one independent sample, reports raw and European-control-variate
standard errors and a 95% interval, and rejects a training request before
allocation when its estimated bulk working set exceeds the configured limit.
The full formulas, bias decomposition, RNG contract, regression method, and
memory accounting are in
[docs/american-lsm-contract.md](docs/american-lsm-contract.md).

After rebuilding the binding, run the versioned cross-check:

```bash
python -m pip install -e '.[dev,train]'
python -m differentiable_pricing.american.lsm_crosscheck \
  --config configs/american_lsm_crosscheck_v1.toml \
  --output artifacts/american-lsm-crosscheck-v1.json
```

The configuration reuses nine SHA-256-pinned named CRR regimes and varies path
count, exercise-grid density, and polynomial degree one factor at a time. The
primary experiment uses 64 exercise intervals, 65,536 policy-training paths,
131,072 valuation paths, and a quadratic continuation basis. Those settings
are exploratory until the generated report is reviewed.

The LSM interval quantifies valuation noise for a **fixed learned policy**. It
does not include policy-fitting error or the gap between discrete Bermudan and
continuous American exercise. The 8,192/8,193 CRR adjacent average is likewise
an internal constant-parameter-model reference, not a market quote. Agreement
between them is useful evidence that two very different numerical methods are
not making a large silent error; it is not evidence that GBM describes live
option prices.

#### Reading the report without misreading it

The interval is **valuation-only** and has no nominal coverage rate against the
CRR reference, the same-grid optimal value, or the continuous American value.
`crr_reference_inside_stochastic_valuation_interval_cases` is **not** a
calibration statistic and is not an acceptance gate. Policy-fitting error and
the discrete-exercise gap are systematic and do not shrink with valuation
paths, while sampling error shrinks at \(n^{-1/2}\); adding paths therefore
narrows the interval around a fixed bias, and the count is expected to fall
towards zero. A low count is evidence that noise has been driven below the
bias, not evidence of a failure. Each experiment summary carries these caveats
inline, so they travel with the numbers.

Cases whose valuation estimator has exactly zero variance --- immediate
exercise at time zero, or structural suppression where the control variate
reproduces the payoff exactly --- have a single-point interval that cannot
contain a finite-step tree value however close the agreement. They are counted
and reported separately, with their tree differences given as differences
rather than converted into successes by an invented tolerance.

Four observed results are expected behaviour, documented in
[docs/american-lsm-contract.md](docs/american-lsm-contract.md), and should not
be tuned away: LSM sitting below CRR (a fixed policy on a coarse grid is a
lower bound); degree one being clearly inadequate; degree three not improving
on degree two at these path counts; and a denser exercise grid making results
worse when policy-training paths are held fixed.

The engine also rejects rather than corrupts. The shared vanilla-input
validator admits every finite rate, dividend yield and volatility, and a finite
log spot does not imply a representable spot, so every derived price-domain
quantity is validated where it is produced. For any admitted input the engine
either returns fully finite results or raises; it never returns a NaN price or
a silently corrupted standard error.

## American cross-check numerical results

The reviewed cross-check evidence is frozen in
[docs/results/american_lsm_crosscheck_results_v1.json](docs/results/american_lsm_crosscheck_results_v1.json).
The raw report stays ignored under `artifacts/`; the snapshot is a compact,
strictly versioned extraction of it, generated programmatically rather than
transcribed. Regenerate and validate it with:

```bash
python scripts/freeze_american_lsm_results.py \
  --report artifacts/american-lsm-crosscheck-review-fixed-v1.json \
  --output docs/results/american_lsm_crosscheck_results_v1.json --update
python scripts/freeze_american_lsm_results.py --check
python scripts/plot_american_lsm_results.py --check
```

`--check` needs only checked-in files: it validates the snapshot internally and
reconciles its configuration and C++ provenance digests against the current
repository sources, so CI enforces it without the ignored artifact.

### Frozen experiment table

Seven experiments over nine SHA-256-pinned CRR regimes, 63 case rows. `CRR−LSM`
is in price units against the 8,192/8,193-step CRR adjacent average.

| Experiment | Role | Steps | Degree | Train paths | Valuation paths | Mean CRR−LSM | Max abs gap | Mean valuation-only SE | Stochastic cases | In-interval | Deterministic cases | Min applicable VR ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `paths-low-v1` | path convergence | 64 | 2 | 16,384 | 32,768 | 0.02513 | 0.13529 | 0.02508 | 6 | 4 | 3 | 1.0096 |
| `steps-low-v1` | exercise grid convergence | 32 | 2 | 32,768 | 65,536 | 0.02808 | 0.12666 | 0.01781 | 6 | 3 | 3 | 1.0001 |
| `reference-v1` | reference | 64 | 2 | 32,768 | 65,536 | 0.02713 | 0.07977 | 0.01799 | 6 | 2 | 3 | 1.0033 |
| `steps-high-v1` | exercise grid convergence | 128 | 2 | 32,768 | 65,536 | 0.03298 | 0.14736 | 0.01784 | 6 | 2 | 3 | 1.0120 |
| `degree-one-v1` | basis sensitivity | 64 | 1 | 32,768 | 65,536 | 0.60422 | 2.43419 | 0.01828 | 6 | 1 | 3 | 1.0345 |
| `degree-three-v1` | basis sensitivity | 64 | 3 | 32,768 | 65,536 | 0.02828 | 0.11119 | 0.01778 | 6 | 3 | 3 | 1.0047 |
| `paths-high-primary-v1` | primary | 64 | 2 | 65,536 | 131,072 | 0.02380 | 0.07394 | 0.01263 | 6 | 0 | 3 | 1.0088 |

![Experiment comparison](docs/figures/american_lsm_experiment_comparison.svg)

### Primary result and how to read it

The primary experiment is `paths-high-primary-v1`: 64 exercise intervals,
degree-two basis, 65,536 policy-training paths, 131,072 independent valuation
paths. Its mean gap to the CRR reference is **0.02380** price units and its
largest single-case gap is **0.07394**, on option values between 1.4 and 30.

LSM sits below the finer-grid CRR reference in every stochastic case. That is
the **expected ordering**, not a defect: LSM evaluates a *fixed learned
Bermudan stopping policy* on a discrete exercise grid, and for a fixed policy
the expectation is no greater than the same-grid optimal stopping value.

Three sources of difference must be kept apart, and only the first is measured
by the reported interval:

* **Valuation noise** — sampling error of the valuation stream around an
  already-frozen policy. This is what `Valuation-only SE` reports. It shrinks
  as \(n^{-1/2}\); across the path ladder it falls 0.02508 → 0.01799 → 0.01263,
  ratios of 1.3946 and 1.4244 against the \(\sqrt{2}\approx1.4142\) per
  doubling that implies, deviations of −1.4% and +0.7%. (Those ratios use full
  snapshot precision; recomputing them from the five-decimal table above gives
  1.394 and 1.424.)
* **Policy-fitting bias** — the learned stopping rule is not the optimal rule
  for its own grid. Systematic; does not shrink with valuation paths.
* **Exercise-grid bias** — a 64-date Bermudan is not a continuously
  exercisable American. Systematic; does not shrink with valuation paths.

The reported interval covers **only** the first. It therefore has no nominal
coverage rate against the CRR reference, and the `In-interval` column is a
diagnostic, **not an acceptance gate**. It falls 4 → 2 → 0 along the path
ladder precisely because added paths narrow the interval around a fixed bias.
A low count means noise has been driven below the bias.

| Case | Type | S | K | T | r | q | σ | LSM | CRR ref | CRR−LSM | Valuation-only SE | z | Early ex. | CV coef | VR ratio | Fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `standard_atm_put` | put | 100 | 100 | 1 | 0.05 | 0 | 0.2 | 6.044896 | 6.090415 | +0.04552 | 0.01039 | 4.38 | 0.3675 | 0.3542 | 1.3967 | 0 |
| `deep_itm_put` | put | 70 | 100 | 1 | 0.05 | 0 | 0.25 | 30.000000 | 30.000000 | +4.26e-13 | 0 (zero-width) | zero-width | 1.0000 | 0.0000 | n/a | 0 |
| `deep_otm_put` | put | 130 | 100 | 1 | 0.05 | 0 | 0.25 | 1.446386 | 1.457718 | +0.01133 | 0.00574 | 1.97 | 0.0739 | 0.7712 | 3.7646 | 4 |
| `short_maturity_put` | put | 100 | 100 | 0.05 | 0.05 | 0 | 0.2 | 1.670262 | 1.676875 | +0.00661 | 0.00261 | 2.54 | 0.3028 | 0.4745 | 1.8285 | 0 |
| `high_volatility_put` | put | 100 | 100 | 1 | 0.05 | 0 | 0.8 | 28.460000 | 28.533940 | +0.07394 | 0.02559 | 2.89 | 0.5541 | -0.0864 | 1.0088 | 0 |
| `negative_rate_put` | put | 100 | 100 | 1 | -0.01 | 0 | 0.2 | 8.518075 | 8.518074 | -1.43e-06 | 0 (zero-width) | zero-width | 0.0000 | 1.0000 | infinite | 0 |
| `negative_rate_call` | call | 100 | 100 | 1 | -0.01 | 0 | 0.2 | 7.536196 | 7.568589 | +0.03239 | 0.01452 | 2.23 | 0.2941 | 0.5028 | 1.9797 | 0 |
| `non_dividend_call_control` | call | 100 | 100 | 1 | 0.05 | 0 | 0.2 | 10.450584 | 10.450568 | -1.51e-05 | 0 (zero-width) | zero-width | 0.0000 | 1.0000 | infinite | 0 |
| `high_dividend_call` | call | 100 | 100 | 1 | 0.01 | 0.1 | 0.25 | 6.731179 | 6.775596 | +0.04442 | 0.01692 | 2.63 | 0.3195 | 0.3306 | 1.3490 | 0 |

![Primary cases](docs/figures/american_lsm_primary_cases.svg)

Three of the nine cases are **deterministic**: their valuation estimator has
exactly zero variance, from immediate exercise at time zero
(`deep_itm_put`) or from a control variate that reproduces the payoff exactly
(`negative_rate_put`, `non_dividend_call_control`, both of which never exercise
early). Their interval is a single point, so containment of any finite-step
tree value is arithmetically impossible however close the agreement — and their
agreement is very close, at 4.3e-13, 1.4e-06 and 1.5e-05. They are reported
separately and are excluded from every stochastic statistic in the table above.

### Sensitivity findings

![Sensitivity](docs/figures/american_lsm_sensitivity.svg)

* **Degree one is inadequate.** At a fixed 64-date grid and 32,768 training
  paths, the linear basis gives a mean gap of 0.60422 and a worst case of
  2.43419 — more than twenty times the quadratic arm. **Quadratic and cubic
  are comparable** at this path budget: 0.02713 versus 0.02828, a difference of
  0.00115 that is roughly 0.14 combined valuation-only standard errors. Note
  this bounds the difference against *valuation* noise only — each experiment
  ran one training seed, so seed-to-seed variation of the fitted policy is not
  measured here and "comparable" means "not separated by valuation noise", not
  "replicated as equal".
* **More exercise dates did not help at a fixed path budget.** Holding training
  paths at 32,768, going 32 → 64 → 128 dates moved the mean gap 0.02808 →
  0.02713 → 0.03298. Refining the grid adds regression dates without adding
  paths to fit them on, so policy-fitting error grows faster than grid bias
  falls.
* **More paths cut valuation noise; the mean gap improved only modestly.**
  Along the 16,384 → 32,768 → 65,536 training-path ladder the mean valuation-only
  SE falls cleanly (0.02508 → 0.01799 → 0.01263), while the mean gap moves
  0.02513 → 0.02713 → 0.02380. The end-to-end improvement is real but small and
  **not monotone**; the intermediate rise is within the noise of the mean, so
  this supports a residual systematic bias that paths alone do not remove.
* **Every applicable finite control variate achieved a variance-reduction ratio
  of at least approximately one** — the minimum over all seven experiments is
  1.0001, so the European control never materially *hurt*. Deterministic cases
  are excluded from that minimum: their ratio is an undefined 0/0, and cases
  whose control removes all variance are counted as infinite rather than
  folded into it.
* **Regression fallbacks appear in sparse deep-OTM regions.** The primary
  experiment records 4 constant-fallback dates, all in `deep_otm_put`, where
  too few paths are in the money at a given date to fit the continuation basis
  and the fit falls back to a constant. Across experiments the count ranges
  from 2 to 9.

### Scope and what these results do not establish

One-factor geometric Brownian motion with constant rate, dividend yield and
volatility. All evidence is synthetic: no market data, quoted price, or
calibration target enters this study. The CRR reference is an internal
cross-check between two very different numerical methods — it is **not** exact
American truth and **not** market truth, and agreement with it says nothing
about whether GBM describes live option prices.

These results **select no production label policy**. Task 8E separately
predeclares and runs the CRR label-policy calibration study; nothing here may
be reused as its acceptance criterion.

## Build the Python package

The editable install compiles the C++ extension:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,train]'
pytest -q
ruff check .
```

On Windows PowerShell, activate with
`.venv\Scripts\Activate.ps1`.

`./scripts/check.sh` runs this same Python suite after the C++ tests, so it
needs the editable install above. The `--quick` mode used by the pre-commit
hook stops after the C++ tests.

For generation and diagnostics without PyTorch, install the smaller data
extra:

```bash
python -m pip install -e '.[data,dev]'
```

## Where the data comes from

No purchased or historical market dataset is required for the first
experiments. Pricing labels are synthetic:

1. sample economically valid model and contract parameters;
2. evaluate the trusted reference pricer;
3. persist inputs, price, reference sensitivities, seed, pricer version, and
   numerical error estimate;
4. split by parameter regions, not by shuffled duplicate rows.

### Generate the stage-1 European-option dataset

Dataset generation needs NumPy and PyArrow but not PyTorch:

```bash
python -m pip install -e '.[data,dev]'
python -m differentiable_pricing.data.generate \
  --config configs/european_option_dataset_v1.toml \
  --output data/european-option-v1
```

This writes `train.parquet`, `validation.parquet`,
`interpolation_test.parquet`, and a `manifest.json` recording the schema and
generator versions, the base seed and stream derivation, the configuration
SHA-256, the oracle identity, the units and conventions, row counts, and a
SHA-256 per file. The manifest deliberately contains no wall-clock timestamp,
so regenerating from the same configuration reproduces byte-identical output —
for the toolchain the manifest records under `runtime`. NumPy does not
guarantee its generator streams across versions and PyArrow stamps its version
into the Parquet footer, so treat a different toolchain as a new dataset until
the hashes are compared. Every label comes from the compiled C++ oracle; the
partitions are drawn from independent streams rather than random-split from one
table. Generated files are ignored by Git.

The sampling box in the configuration is a provisional synthetic engineering
range, not a calibrated market distribution. Because the three factors are
drawn independently, a minority of rows land in the short-maturity,
low-volatility, deep-moneyness corner where the price underflows towards zero
and delta saturates; the manifest's `label_diagnostics` block counts them per
partition so downstream metrics can stratify rather than average them away.
The boundary partition those rows really belong in is still to be built.

### Diagnose a generated dataset

Once a dataset exists, describe it deterministically:

```bash
python -m differentiable_pricing.data.diagnose \
  --dataset data/european-option-v1 \
  --output data/european-option-v1/diagnostics.json
```

The tool validates `manifest.json` and verifies the SHA-256 of *every* declared
Parquet file **before** it opens any of them, then writes one versioned JSON
report (`diagnostics_schema_version`) containing, per split:

- row count, call/put counts, and finite/non-finite counts;
- min, max, mean, standard deviation, and p01/p05/p50/p95/p99 for every input,
  the price, and every Greek;
- standardized-moneyness bands over
  `absolute_z = |log_forward_moneyness| / (volatility * sqrt(maturity))`:
  core (`<= 4`), tail (`4 < z <= 8`), extreme (`> 8`), with counts and
  proportions;
- near-zero-price and saturated-delta counts recomputed under the thresholds
  **recorded in that dataset's manifest**, and compared with the counts the
  manifest recorded;
- rowwise European no-arbitrage bound checks, with violation counts and the
  raw maximum violation magnitude;
- observed per-column ranges against the sampling domain the manifest
  declares;
- duplicate sample identifiers, duplicate *economic states*, intersections of
  both between splits, and manifest/Parquet row-count and schema agreement.

The report has no timestamp and no run identifier, and every list is sorted, so
two runs over identical input bytes produce a byte-identical file. As with the
generator, float summaries are reproducible for a fixed toolchain; counts and
digests are exact everywhere.

Exit status is `0` when no findings were raised, `3` when the dataset is
readable but something was reported (duplicate identifiers, cross-split
identifier leakage, bound violations, non-finite values, disagreement with the
manifest), and `2` when the dataset could not be trusted enough to describe at
all (missing, undeclared, or hash-mismatched Parquet file, malformed manifest,
wrong schema, unknown `option_type`, or a manifest row count that disagrees
with the Parquet footer). `manifest.json` is not itself hashed, so that last
check is the only guard against a manifest edited in isolation.

**How to read the output**

- These are **dataset diagnostics, not model-performance metrics.** Every
  number describes the stored bytes. Nothing here is evidence that a surrogate
  can price anything, and a clean report is a precondition for an experiment,
  not a result of one.
- **Extreme-`z` rows must be evaluated separately.** They price to within
  float64 noise of intrinsic value, so they carry almost no gradient signal and
  relative price error over them is meaningless. Stratify them out of headline
  metrics and report them as their own slice; do not delete them and do not
  average them away.
- Bound violations are counted against a documented float64 tolerance of
  `1e-12` relative to `max(discounted_spot, discounted_strike)`, about four
  orders of magnitude above rounding noise. The reported maximum magnitudes are
  raw, so remaining head room stays visible.
- Two leakage checks are reported and they are not equivalent. Sample
  identifiers are `<split>-<index>` by construction, so an identifier
  intersection means a corrupted identifier column, not leakage. The
  **state** check — exact float64 equality over `(option_type, spot, strike,
  maturity, rate, dividend_yield, volatility)` — is the one the research
  contract asks for. It finds duplicated states, not *near*-duplicate ones;
  near-duplicate detection needs a declared quantization of the input space,
  which this project has not fixed yet.
- Statistics are pooled over a whole split, not conditioned on the moneyness
  band. Before quoting a p95 or p99 of price, gamma, or vega as a property of
  the tradable region, recompute it per band from the stored columns.
- **Black--Scholes is analytic and will often be faster than the network.**
  Stage 1 is a *correctness* experiment: it exists to expose normalization,
  derivative, sampling, and implementation errors against closed-form truth. It
  is not, and cannot be, a latency-win claim. A latency argument only becomes
  meaningful for references that are genuinely expensive (trees, PDE, LSM), and
  then only end-to-end, with feature transforms, serialization, batching, and
  derivative cost all counted.

The generated dataset and its diagnostics live under `data/`, which Git
ignores.

Black--Scholes provides exact labels at stage 1. Later, trees/PDE/Monte Carlo
provide labels with convergence and standard-error diagnostics. Market data
becomes useful when calibrating realistic curves, surfaces, and parameter
distributions; it is not needed to prove that the learning and differentiation
machinery is correct.

## Train the stage-1 neural baseline

Training reads and verifies only `train.parquet` and `validation.parquet`.
It does not hash or open `interpolation_test.parquet`; that split remains
untouched until the separate evaluation command.

```bash
python -m differentiable_pricing.ml.train \
  --dataset data/european-option-v1 \
  --config configs/european_neural_baseline_v1.toml \
  --output artifacts/european-neural-baseline-v1
```

The pinned baseline is CPU-only and float64. It encodes puts as `-1` and calls
as `+1`, fits every feature mean/scale and the price mean/scale on the training
partition only, and minimizes normalized price MSE. The architecture is only
dense layers, `tanh` hidden activations, and one linear output, matching the
existing C++ `SmoothMlp` contract. Reference Greek columns are never features
or training targets.

For physical features \(x_i\) and price \(V\), the baseline uses

\[
z_i=\frac{x_i-\mu_i}{s_i},\qquad
y=\frac{V-\mu_V}{s_V},\qquad
\mathcal{L}_{\mathrm{price}}=\operatorname{MSE}(f_\theta(z),y),
\]

with every \(\mu\) and \(s\) estimated from `train` only. The physical wrapper
undoes the target transform inside the autograd graph.

The artifact directory contains canonical metadata plus SHA-256-linked,
non-pickled `weights.npz` arrays. Loading uses `allow_pickle=False`, verifies
the digest, feature order, dtype, layer shapes, and finite values, then rebuilds
the network. PyTorch `.pt` checkpoints are outside this contract and must only
ever be loaded from a trusted source.

The seed, deterministic-algorithm setting, thread count, and Python/NumPy/
PyTorch/platform versions are recorded. Repeated training is tested for
byte-identical weights and metadata in one fixed runtime; PyTorch does not
promise identical floating-point results across releases, hardware, or thread
configurations, so a changed runtime is a new experiment even with the same
TOML seed.

### Controlled capacity experiments

The first baseline is deliberately frozen. Two additional configurations
change one experimental factor at a time:

| Configuration | Hidden layers | Maximum epochs | Question |
|---|---:|---:|---|
| `european_neural_baseline_v1.toml` | 64×64×64 | 100 | Initial price-only baseline |
| `european_neural_long_v1.toml` | 64×64×64 | 300 | Was the baseline stopped too early? |
| `european_neural_wide_v2.toml` | 128×128×128×128 | 300 | Does additional capacity improve the same objective? |

The long and wide runs keep the seed, batch size, optimizer, learning rate,
weight decay, dtype, device, and feature order fixed. Their validation reports
are compared against the versioned gates in
`configs/european_neural_acceptance_v1.toml`; the gates must not be relaxed
after seeing a result. Repository history does not by itself establish that
these v1 gates predate the results they judge — see the note above the results
table.

The original baseline's interpolation-test report has already informed further
model work, so that split is consumed for this experiment series. Do not use it
to choose between the long and wide candidates. After selecting on validation,
generate a fresh-seed dataset, retrain the frozen winner, and evaluate that
new dataset's interpolation test exactly once.

Train the controlled candidates into separate artifact directories:

```bash
python -m differentiable_pricing.ml.train \
  --dataset data/european-option-v1 \
  --config configs/european_neural_long_v1.toml \
  --output artifacts/european-neural-long-v1

python -m differentiable_pricing.ml.train \
  --dataset data/european-option-v1 \
  --config configs/european_neural_wide_v2.toml \
  --output artifacts/european-neural-wide-v2
```

### Forward-normalized representation experiment

If capacity alone does not meet the fixed gates, the next controlled
experiment changes the financial representation while returning to the
3×64 long-control architecture. For spot \(S\), strike \(K\), maturity \(T\),
rate \(r\), dividend yield \(q\), and volatility \(\sigma\), it computes

\[
F = S e^{(r-q)T}, \qquad
x = \log(F/K), \qquad
v = \sigma\sqrt{T}, \qquad
A = S e^{-qT}.
\]

The network learns the dimensionless relationship

\[
\frac{V}{A} = f(\text{option type}, x, v),
\]

and the physical wrapper reconstructs \(V=A f\). This uses the exact
homogeneity of the Black–Scholes price to reduce seven raw inputs to three
economic coordinates. The transforms and reconstruction remain inside the
PyTorch graph, so physical-unit Greeks still follow from autograd. This
experiment remains price-only and does not guarantee no-arbitrage bounds.

The model size, optimizer, seed, batch size, and training budget exactly match
`european_neural_long_v1.toml`; only the representation changes:

```bash
python -m differentiable_pricing.ml.train \
  --dataset data/european-option-v1 \
  --config configs/european_neural_forward_normalized_v1.toml \
  --output artifacts/european-neural-forward-normalized-v1

python -m differentiable_pricing.ml.evaluate \
  --dataset data/european-option-v1 \
  --artifact artifacts/european-neural-forward-normalized-v1 \
  --partition validation \
  --output artifacts/european-neural-forward-normalized-v1/validation-evaluation.json
```

### Differential-training experiment

Price accuracy does not imply derivative accuracy. The next controlled
experiment keeps the forward-normalized representation, 3×64 network, seed,
optimizer, and 300-epoch budget fixed, but adds first-derivative supervision.
Writing \(u=V/A\), the existing analytic delta and vega labels determine the
coordinate derivatives exactly:

\[
u_x=e^{qT}\Delta-u, \qquad
u_v=\frac{\text{Vega}}{A\sqrt{T}}.
\]

These derivatives are converted to the standardized network coordinates using
the feature and target scales fitted on the training split. With \(y\) denoting
the standardized target and \(z_x,z_v\) the standardized coordinates, the
configured loss is

\[
\operatorname{MSE}(\hat y,y)
+\frac{1}{2}\left[
  \operatorname{MSE}(\partial_{z_x}\hat y,\partial_{z_x}y)
  +\operatorname{MSE}(\partial_{z_v}\hat y,\partial_{z_v}y)
\right].
\]

Training uses delta and vega labels only. The reported Greeks therefore carry
different evidential weight, and it is worth being precise about which is
which:

| Greek | Relationship to the training objective |
|---|---|
| Price | Directly supervised by the price term. |
| Delta, vega | Directly supervised, through the transformed first derivatives \(u_x\) and \(u_v\). |
| Rho, theta | Not supervised, but algebraic reweightings of the same learned \((u,u_x,u_v)\) on the unconstrained branch. |
| Gamma | Not in the objective; requires second-order autograd through \(u_{xx}\). |
| European bounds | Not learned at all; imposed structurally by the projection below. |

Delta and vega are supervised through the differential targets \(u_x\) and
\(u_v\). Rho and theta are algebraic reweightings of the same learned
\((u,u_x,u_v)\), so they primarily validate physical-unit reconstruction rather
than independent derivative learning. Concretely, under the forward-normalized
representation,

\[
\rho=\frac{\partial V}{\partial r}=A\,T\,u_x,
\qquad
\theta=-\frac{\partial V}{\partial T}
 =q A u-A(r-q)u_x-\frac{A u_v v}{2T},
\]

so once \((u,u_x,u_v)\) are fixed at a point, rho and theta are determined
there. Those identities describe the unconstrained branch. Where the
European-bounds projection below is active, the shipped model's price is the
discounted bound rather than \(A\,u\), so the reported derivatives follow that
bound instead, and no derivative is unique exactly at a projection kink. The
evaluation metrics are computed from the constrained model that is actually
shipped, so they already reflect whichever branch applied at each row; the
projection is part of the mathematical model, not a reporting adjustment. Gamma
is the only reported Greek requiring out-of-objective curvature \(u_{xx}\),
although differential training can regularize it indirectly by constraining the
first derivative between labelled points. At inference, every Greek is still
computed from autograd rather than emitted as a separate network output.

```bash
python -m differentiable_pricing.ml.train \
  --dataset data/european-option-v1 \
  --config configs/european_neural_forward_differential_v1.toml \
  --output artifacts/european-neural-forward-differential-v1

python -m differentiable_pricing.ml.evaluate \
  --dataset data/european-option-v1 \
  --artifact artifacts/european-neural-forward-differential-v1 \
  --partition validation \
  --output artifacts/european-neural-forward-differential-v1/validation-evaluation.json
```

### European no-arbitrage bounds projection

The differential model can still make small price-bound violations because
its scalar output is unconstrained. This experiment changes only the output
contract, not the training data, objective, architecture, or weights. For
discounted spot \(A=S e^{-qT}\) and discounted strike \(B=K e^{-rT}\), the
model computes the usual European interval

\[
\begin{aligned}
L_\text{call}&=\max(A-B,0), & U_\text{call}&=A,\\
L_\text{put}&=\max(B-A,0),  & U_\text{put}&=B,
\end{aligned}
\]

then returns

\[
V_\text{bounded}=\min\!\left(U,\max(L,V_\text{network})\right).
\]

The Black--Scholes reference price is inside this interval, so interval
projection cannot increase pointwise absolute price error. It is
piecewise-differentiable: away from a clipping boundary, autograd returns
either the learned-network Greeks or the active bound's physical derivatives.
At the clipping boundary and the intrinsic-value kink, the derivative is not
unique; the derived artifact and evaluation report state this limitation
explicitly.

Derive a bounded artifact from the already-trained differential model:

```bash
python -m differentiable_pricing.ml.constrain \
  --artifact artifacts/european-neural-forward-differential-v1 \
  --output artifacts/european-neural-forward-differential-bounded-v1

python -m differentiable_pricing.ml.evaluate \
  --dataset data/european-option-v1 \
  --artifact artifacts/european-neural-forward-differential-bounded-v1 \
  --partition validation \
  --output artifacts/european-neural-forward-differential-bounded-v1/validation-evaluation.json
```

The derivation command verifies the source artifact, copies `weights.npz`
byte-for-byte, records the source manifest and weight SHA-256 digests, and
publishes the result atomically. It refuses to overwrite an existing
directory or to stack the same constraint twice. This is not another training
run: the source and derived artifacts must report the same weight digest.

## What the controlled experiments showed

The following values come from the same 25,000-row `validation` partition and
therefore describe model selection, not a locked final evaluation. Each row
changes one main factor relative to its control. The acceptance thresholds live
in `configs/european_neural_acceptance_v1.toml`. These versioned gates were used
during the initial development study, but their chronology is not independently
established by repository history: the configuration and the results arrive in
the same change set, so nothing here proves the thresholds preceded the numbers
they judge. The fresh-seed protocol now binds its acceptance configuration by
SHA-256; merging that protocol before any replication execution makes the
ordering checkable rather than merely asserted.

| Experiment | Main controlled change | Price/spot RMSE | Delta RMSE | Gamma RMSE | Material bound violations | Gate result |
|---|---|---:|---:|---:|---:|---|
| Raw baseline | 3×64, price-only, seven raw inputs | 0.003482 | 0.03577 | 0.006962 | 2,211 | Fail |
| Longer | Same model, 100 → 300 epochs | 0.001898 | 0.02236 | 0.005432 | 1,743 | Fail |
| Wider | Four 128-unit layers | 0.001515 | 0.02025 | 0.005368 | 1,393 | Fail |
| Forward | 3×64 on \((\text{type},\log(F/K),\sigma\sqrt{T})\) | 0.0009008 | 0.01995 | 0.006551 | 1,626 | Fail |
| Differential | Add analytic delta/vega supervision | 0.0002171 | 0.002536 | 0.001854 | 75 | Arbitrage only |
| Bounded | Project unchanged differential weights onto \([L,U]\) | **0.0002168** | **0.002474** | **0.001504** | **0** | **Pass** |

![Validation errors divided by their acceptance limits, on a logarithmic vertical axis](docs/figures/european_validation_error_progression.svg)

![Material no-arbitrage violations across controlled experiments](docs/figures/european_validation_arbitrage_progression.svg)

The progression exposed several useful distinctions:

- **Optimization budget helped, but did not solve the problem.** Extending the
  original network from 100 to 300 epochs roughly halved price error, while
  price, slope, curvature, and arbitrage gates still failed.
- **Capacity gave diminishing returns.** The wider model improved price and
  delta modestly over the long control but remained far outside the fixed
  gates.
- **Financial coordinates mattered more than width for price.** Forward
  normalization used Black--Scholes homogeneity to remove redundant scale and
  delivered the strongest price-only improvement. Gamma became worse,
  demonstrating that accurate prices do not imply accurate curvature.
- **Differential supervision was the decisive learning change.** Supplying
  analytic information about \(u_x\) and \(u_v\) reduced price, delta, vega,
  theta, and rho errors several-fold to nearly an order of magnitude. Those
  five are not five independent confirmations: price, delta, and vega are the
  supervised quantities themselves, and rho and theta are algebraic
  reweightings of the same learned \((u,u_x,u_v)\), so their improvement
  chiefly confirms that physical-unit reconstruction is correct. Gamma—never a
  target, and requiring out-of-objective curvature \(u_{xx}\)—also crossed its
  gate, which is the one genuinely out-of-objective signal in this row, though
  first-derivative supervision plausibly regularizes curvature indirectly.
- **Structure removed the remaining failure without retraining.** Only 76 of
  25,000 rows (`0.304%`) were projected, all onto the lower bound. The maximum
  price adjustment was `0.102481`; the copied weights remained byte-identical.
  Violations fell from 75 material cases to zero, while every headline error
  metric improved slightly.

The selected bounded model passes all validation gates in
`configs/european_neural_acceptance_v1.toml`. That is a model-selection result
on the `validation` partition, judged against gates whose chronology repository
history does not independently establish; it is not a locked interpolation-test
result. Gamma is the tightest headline gate at about 75% of its allowed RMSE, so
the fresh replication must confirm curvature rather than treating this run as
conclusive.

The exact plotted inputs, weight digests, intervention counts, and non-claim
status live in
[`docs/results/european_validation_results_v1.json`](docs/results/european_validation_results_v1.json).
Regenerate both SVGs without an additional plotting dependency:

```bash
python scripts/plot_european_validation_results.py
python scripts/plot_european_validation_results.py --check
```

## Frozen fresh-seed replication protocol

`configs/european_neural_replication_protocol_v1.toml` locks the next
experiment before any new labels or results exist. It binds the development
result, original and replication dataset configurations, selected training
configuration, and acceptance gates by SHA-256. It also declares the selected
European-bounds projection, output locations, and the one-shot final-evaluation
policy.

The replication is intentionally not a sampling ablation. Relative to the
development study:

- the dataset identity and base seed change;
- the training experiment identity and initialization seed change;
- the generator, oracle, row counts, domain, sampler, representation,
  architecture, objective, optimizer, training budget, evaluation bands,
  acceptance gates, and output constraint remain identical.

Both new seeds are the first unsigned 32 bits of SHA-256 over public labels
recorded in the protocol. This makes the choices reproducible and prevents
selecting a favorable seed after seeing a result. The protocol permits
validation for checkpoint selection and one evaluation of the new
`interpolation_test`. A failure must be recorded; any subsequent tuning
requires a new protocol and a newly generated final partition.

Validate the lock without generating data or importing PyTorch:

```bash
python scripts/check_european_replication_protocol.py
```

Do not execute the replication from an unmerged protocol branch or a dirty
worktree. The protocol-only commit must precede dataset generation in
repository history.

Task 7B's execution runner, rather than this static validator, must also refuse
pre-existing output directories and record the protocol and source commit,
clean-worktree state, runtime and toolchain versions, artifact lineage, and
evidence for the single final-partition evaluation.

After the runner itself is merged to `main`, post-merge CI is green, and the
worktree is clean, execute every stage through validation:

```bash
python scripts/run_european_replication.py run-to-validation
python scripts/run_european_replication.py status
```

This generates and diagnoses the fresh dataset, trains the frozen model,
derives the European-bounded artifact, evaluates `validation`, and evaluates
all frozen acceptance gates. It never performs model evaluation on
`interpolation_test`. Existing dataset, artifact, report, or run-ledger paths
cause a hard failure; the runner has no overwrite or resume flag.

The audit ledger is written atomically to
`runs/european-neural-replication-v1/execution.json`. It records the protocol
and source commits, clean `main`/`origin/main` state, runtime versions, exact
commands and outputs, artifact lineage, file digests, and validation-gate
decisions. A failed stage remains recorded and requires the failure policy in
the frozen protocol; deleting outputs and trying the same protocol again is
not an accepted research result. The local ledger is traceability evidence,
not an authenticated attestation; the terminal result and its digests must be
reviewed and committed as a separate, immutable result snapshot.

## Evaluate prices and learned Greeks

Evaluation requires the partition role to be explicit. Validation reports may
guide model selection and can be regenerated while comparing candidates:

```bash
python -m differentiable_pricing.ml.evaluate \
  --dataset data/european-option-v1 \
  --artifact artifacts/european-neural-baseline-v1 \
  --partition validation \
  --output artifacts/european-neural-baseline-v1/validation-evaluation.json
```

Only when the ledger says `ready_for_final`, inspect the validation report and
then intentionally consume the locked final evaluation:

```bash
python scripts/run_european_replication.py final-evaluate \
  --confirm-locked-final-evaluation
python scripts/run_european_replication.py status
```

The runner records the attempt before starting evaluation. A process
interruption therefore still consumes the one permitted attempt, and an
existing final report is never overwritten. Passing or failing the final gates
is terminal for this protocol; the final split must not guide another model
change.

## Fresh-seed replication result

The frozen protocol completed successfully on a new 250,000-row dataset and a
new initialization seed. Training and checkpoint selection read only `train`
and `validation`; the new `interpolation_test` was then evaluated once. The
ledger records `final_evaluation_attempts = 1`, `status =
replication_passed`, a clean `main` worktree, and source commit
`5c47081e265a4f1256b733d398079ca02f2f290e`.

The central result is not merely that a second training run converged. It is
that a precommitted design reproduced its price and Greek accuracy on an
independently generated final partition without changing the architecture,
objective, domain, row counts, optimizer, training budget, evaluation bands,
or gates.

| Frozen check | Validation | Locked final | Limit | Final budget |
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

![Validation and locked-final errors divided by their frozen acceptance limits](docs/figures/european_replication_gate_margins.svg)

Gamma is the least comfortable result: its locked-final RMSE consumed `96.8%`
of the frozen allowance. That is still a legitimate pass because the threshold
was committed before the fresh dataset existed, but it is the first metric to
stress in a future boundary or OOD study.

The zero-violation result also needs the right interpretation. The trained
network alone violated the discounted lower bound materially on 1,584 locked
final rows. The deterministic European-bounds projection changed 1,586 prices
(`6.34%` of the partition), all at the lower bound, and reduced material
violations to zero. The bounded and unconstrained artifacts have the identical
weight digest
`42670774f736383e50818b6e6c1db9374a77988173e35423ffc34b3c4297ecb8`;
the no-arbitrage guarantee is structural, not learned.

![Rows affected by the European-bounds projection before and after enforcement](docs/figures/european_replication_projection.svg)

This supports a deliberately narrow claim:

> A fresh-seed differential neural surrogate replicated its synthetic
> in-domain European-option pricing and Greek accuracy, passing all 11
> precommitted validation and locked-final gates with zero material
> European-bound violations.

The Greek results do not provide six independent confirmations. Price is
directly supervised; delta and vega are supervised through \(u_x\) and \(u_v\);
rho and theta are algebraic reweightings of the learned \((u,u_x,u_v)\) on the
unconstrained branch. On the 1,586 locked-final rows where the bounds
projection is active, the reported derivatives follow the active discounted
bound instead, and derivatives are not unique at the projection kinks; the
metrics above already reflect whichever branch applied. Gamma is the only
reported Greek outside the objective, although first-derivative supervision
plausibly regularizes it indirectly — that remains an unablated explanation
rather than a measured causal result.

The immutable snapshot records the protocol and source commits, seeds,
toolchain, dataset/report/artifact digests, byte-identical weight lineage,
validation and final metrics, gate decisions, and projection intervention:
[`docs/results/european_replication_results_v1.json`](docs/results/european_replication_results_v1.json).
The raw evidence bundle is deliberately not tracked; its reviewed SHA-256 is
`86859008443a41d86c310c45db9abf784cb3131a32d4a83dca13d9e4a68afb29`.
These digests provide traceability, not an authenticated attestation of who ran
the experiment.

Regenerate or verify both replication figures without another plotting
dependency:

```bash
python scripts/plot_european_replication_results.py
python scripts/plot_european_replication_results.py --check
```

The physical price reconstruction is inside the differentiable graph. Autograd
therefore includes both input and target scaling when it computes delta,
gamma, vega, rho, and theta (`-d price / d maturity`). The deterministic JSON
report links the exact dataset manifest, training configuration, artifact
manifest, weight digests, output constraint, and constrained artifact's source
artifact. It reports MAE, RMSE, p95, p99, and maximum absolute error overall,
by call/put, and for core/tail/extreme standardized moneyness. For price it
also reports error after division by spot; it does not divide by option price
because near-zero prices make that ratio unstable.
Learned-price European no-arbitrage violations are counted separately at both
the float64 reporting tolerance and a material relative tolerance of `1e-6`.
For a constrained artifact, the report also gives the number of lower/upper
bound activations, maximum absolute price adjustment, and the unconstrained
network's price and arbitrage metrics so the projection's intervention remains
visible.
The report labels validation as `model_selection` and interpolation test as
`locked_final_evaluation`.

This run proves or falsifies the plumbing, not the economic thesis:

- a network derivative is exact for the learned function, not automatically a
  Black--Scholes or market Greek;
- the report covers synthetic in-envelope interpolation only;
- no boundary, extrapolation/OOD, latency, or C++ deployment claim is made;
- analytic Black--Scholes can easily be faster than this network. A speed
  comparison becomes interesting only when the reference is genuinely
  expensive and end-to-end costs are measured.

## Agent-assisted development

`CLAUDE.md` is the checked-in operating contract. Claude Code can discover the
project agents under `.claude/agents/`:

- `code-reviewer`: independent, findings-first software review;
- `numerical-reviewer`: model, Greek, convergence, leakage, and experiment
  validity review.

Both agents are read-only and start with their own context. Ask Claude Code:

```text
Use code-reviewer to review the current diff. Do not modify files.
Use numerical-reviewer to audit the pricing and validation assumptions.
```

`.claude/settings.json` runs a deterministic post-edit hook. It validates
edited Python, JSON, TOML, and (when installed) C++ formatting without
rewriting files. Inspect it in Claude Code with `/hooks`.

These agents produce evidence and findings; they do not replace deterministic
tests, CI, or human ownership. See
[docs/agentic-workflow.md](docs/agentic-workflow.md).

## Initialize the local Git repository

After extracting this starter:

```bash
cd differentiable-pricing
git init -b main
./scripts/install-git-hooks.sh
./scripts/check.sh
git add .
git commit -m "Initialize differentiable pricing research platform"
```

If your Git version does not support `git init -b main`, run `git init` and
then `git branch -M main`.

## Create and push the GitHub repository

Authenticate once, then create the remote from this existing directory:

```bash
gh auth login
gh repo create differentiable-pricing \
  --private \
  --source=. \
  --remote=origin \
  --push \
  --description="C++ pricing, Python training, and differentiable neural surrogates"
```

Change `--private` to `--public` when you are ready to use it in applications.
Do not ask GitHub to add a README or `.gitignore`: this repository already has
both, and an independently initialized remote can create an avoidable merge.

## Immediate next milestone

Stage 1 is complete at its stated synthetic in-domain scope. The American CRR
engine now has diagnostic and price-only paths, a deterministic parallel batch
boundary, and a versioned exploratory convergence study. The independent LSM
engine and its path/step/basis cross-check protocol are now implemented. The
next action is to run and review that cross-check, then freeze a label policy
tied to an explicit reference-error budget before building exercise-aware
train, validation, boundary, and locked-test partitions. Only then should the
European-to-American transfer experiment begin.

LSM does not replace the tree in this one-dimensional problem. Its purpose
here is independent validation and preparation for path-dependent or
higher-dimensional products where recombination no longer keeps the state
space tractable.

## License

MIT; see [LICENSE](LICENSE).
