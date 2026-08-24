# Differentiable Pricing

A research platform for testing whether neural surrogates can reproduce
derivative prices **and useful risk sensitivities** at materially lower
inference cost than their reference pricers.

---

## The research question

Re-pricing a book is rarely one price: it is the Greeks, the scenario grid, and
the exercise-aware models behind them. A smooth network is attractive because
one reverse-mode pass returns the value and every **first-order** input
sensitivity together — delta, vega, rho, theta. Second-order quantities such as
gamma are not free: they need a further differentiation pass, which is one
reason curvature is numerically more demanding than the first-order
sensitivities. Whether any of those sensitivities are *trustworthy* is
falsifiable, so this project starts where truth is known and moves outward one
controlled stage at a time.

> A network derivative is exact **for the learned network**, not automatically
> an exact market Greek. It becomes a credible Greek only after feature and
> target normalization are reversed and the result is validated against
> analytic, algorithmic, or convergence-tested bump references.

Four planned stages add one difficulty at a time — European option, American
option, European swaption, Bermudan swaption — each with its own trusted
reference pricer. Where transfer is part of the hypothesis, transferred
initialization is compared with a matched randomly initialized model. Every
stage must beat acceptance criteria fixed *before* results are observed, on a
partition that has informed no model choice
([docs/research-contract.md](docs/research-contract.md)).

**Every neural-pricer result here is synthetic.** No market data, quoted price,
or calibration target enters any dataset, training run, or reported metric on
the surrogate side; labels come from the project's own reference pricers.

A separate, **read-only market-feasibility track** does touch real data: it
reads three sessions of a proprietary quote archive to establish which pricing
inputs that archive can and cannot supply. It produces no price, Greek, label,
or calibrated value, it feeds no dataset or training run, and neither it nor
anything derived from it enters Git. The two tracks are never mixed: no result
below is market-validated, and no market-derived quantity is a project result.

---

## Stage 1 — European: complete and replicated

A precommitted protocol ran once, end to end, on a **newly generated
250,000-row dataset and a new initialization seed**. Architecture, objective,
domain, row counts, optimizer, budget, evaluation bands, output constraint, and
every acceptance gate were bound by SHA-256 before the data existed. Training
and checkpoint selection saw only `train` and `validation`; the fresh
`interpolation_test` was evaluated once and is now permanently consumed.

| Frozen check | Validation | Locked final | Limit | Budget used |
|---|---:|---:|---:|---:|
| Price / spot RMSE | 0.0001799 | **0.0001805** | 0.0005 | 36.1% |
| Price / spot P99 absolute error | 0.0004290 | **0.0004246** | 0.002 | 21.2% |
| Delta RMSE | 0.002613 | **0.002942** | 0.01 | 29.4% |
| Gamma RMSE | 0.001679 | **0.001937** | 0.002 | 96.8% |
| Vega RMSE | 0.2452 | **0.2627** | 1.0 | 26.3% |
| Theta RMSE | 0.1359 | **0.1311** | 1.5 | 8.7% |
| Rho RMSE | 0.2840 | **0.2901** | 2.0 | 14.5% |
| Material European-bound violations | 0 | **0** | 0 | Pass |

Three tail metrics (price, delta, gamma) also passed; all eleven gates are in
the frozen snapshot.

![Validation and locked-final errors divided by their frozen acceptance limits, all bars below the limit line, with gamma closest to it](docs/figures/european_replication_gate_margins.svg)

**Three qualifications belong with that result, not in a footnote.**

- **Gamma is the least comfortable pass.** Its locked-final RMSE consumed 96.8%
  of the frozen allowance. Curvature is the first metric to stress in any
  boundary or out-of-domain study.
- **Zero arbitrage violations is structural, not learned.** The trained network
  alone violated the discounted lower bound materially on 1,584 locked-final
  rows; a deterministic European-bounds projection adjusted 1,586 prices (6.34%
  of the partition), all at the lower bound, driving material violations to
  zero. The bounded and unconstrained artifacts share one weight digest.
- **The six reported Greeks are not six independent confirmations.** Price,
  delta, and vega are directly supervised. Rho and theta are algebraic
  reweightings of the same learned quantities, so they chiefly confirm
  physical-unit reconstruction. Gamma is not in the objective at all and
  requires second-order autograd; first-derivative supervision plausibly
  regularizes it, but that is an unablated explanation, not a measured result.

Frozen evidence:
[`docs/results/european_replication_results_v1.json`](docs/results/european_replication_results_v1.json)
records commits, seeds, toolchain, digests, byte-identical weight lineage, every
metric, gate decisions, and the projection intervention, with
`final_evaluation_attempts = 1` and `status = replication_passed`. Digests are
traceability, not an authenticated attestation of who ran a run.

---

## Stage 2 — American: the problem, and the reference architecture

An American option can be exercised at any time before expiry, so there is no
closed-form price. The value solves a free-boundary problem, and the exercise
boundary depends separately on the rate and the dividend yield — which is why
the three-input representation that determines a European price exactly is
**inadmissible** for American labels.

That has two consequences for this project. Labels must come from a numerical
method, so **label error is now part of the error budget**. And that method is
expensive — CRR backward induction is $O(N^2)$ per option — which is exactly
what makes a surrogate worth testing.

### Three numerically independent reference engines

| Engine | Role | Contract | Frozen evidence |
|---|---|---|---|
| **CRR binomial tree** (C++20, scalar + parallel batch) | Teacher and reference for continuous-yield American prices. Reports the $N$- and $(N+1)$-step prices and their adjacent average | [american-crr-contract.md](docs/american-crr-contract.md) | none — cross-check role |
| **Longstaff–Schwartz Monte Carlo** (C++20) | Independent cross-check. Fits a stopping policy on one antithetic path stream and values that frozen policy on a second | [american-lsm-contract.md](docs/american-lsm-contract.md) | [snapshot](docs/results/american_lsm_crosscheck_results_v1.json) |
| **Crank–Nicolson PDE oracle** (C++20) | Discrete cash dividends, PSOR obstacle solve, valuation-time price/delta/gamma surface | [pde-numerical-contract.md](docs/pde-numerical-contract.md) | [v1 pilot](docs/results/american_pde_label_policy_results_v1.json), [v2 policy](docs/results/american_pde_label_policy_v2_results_v1.json) |

#### CRR and LSM agree, and what that agreement is worth

Two numerically unrelated methods price the same nine pinned regimes: a
deterministic CRR tree refined to an 8,192/8,193-step adjacent average, and an
LSM engine that fits a stopping policy on one antithetic path stream and values
that frozen policy on a second. Seven experiments, 63 case rows; `CRR−LSM` is in
price units against the CRR adjacent average.

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

The primary experiment reaches a **mean gap of 0.02380 price units and a worst
single case of 0.07394**, on option values between 1.4 and 30. Three of the nine
cases are deterministic — their valuation estimator has exactly zero variance —
and they agree with the tree to 4.3e-13, 1.4e-06, and 1.5e-05.

![Primary-experiment case prices with valuation-only standard errors, deterministic zero-width cases marked separately](docs/figures/american_lsm_primary_cases.svg)

**Reading the gap correctly matters more than its size.** LSM sits below the
finer-grid CRR reference in every stochastic case, which is the expected
ordering rather than a defect. Three sources of difference must be kept apart,
and the reported interval measures only the first:

- **Valuation noise** — sampling error around an already-frozen policy, the
  `Mean valuation-only SE` column. It shrinks as $n^{-1/2}$: along the path
  ladder it falls 0.02508 → 0.01799 → 0.01263.
- **Policy-fitting bias** — the learned rule is not optimal for its own grid.
- **Exercise-grid bias** — a 64-date Bermudan is not a continuously exercisable
  American.

The last two are systematic and do not shrink with valuation paths, so the
interval has no nominal coverage rate against the CRR reference. `In-interval`
is a diagnostic, **not an acceptance gate**: it falls 4 → 2 → 0 along the ladder
precisely because added paths narrow the interval around a fixed bias while the
point estimates keep agreeing to roughly 0.02 price units.

![Sensitivity of the mean CRR-minus-LSM gap to basis degree, exercise-grid density, and path count](docs/figures/american_lsm_sensitivity.svg)

Degree one is inadequate — a mean gap of 0.60422, more than twenty times the
quadratic arm — while quadratic and cubic are comparable and not separated by
noise. More exercise dates did not help at a fixed path budget, and more paths
cut valuation noise while improving the gap only modestly and **not**
monotonically, consistent with a residual systematic bias. These are
consequences of the method, documented in
[docs/american-lsm-contract.md](docs/american-lsm-contract.md), and must not be
tuned away.

**What this does not establish.** One-factor geometric Brownian motion with
constant rate, dividend yield, and volatility. All evidence is synthetic: **no
market data**, quoted price, or calibration target enters this study. The CRR
reference is an internal cross-check between two very different numerical
methods — **not exact American truth** and **not market truth** — and agreement
with it says nothing about whether GBM describes live option prices. These
results **select no production label policy**: task 8E predeclared the CRR
label-policy calibration study separately, and nothing here may be reused as its
acceptance criterion. Neither engine prices discrete dividends, which is why the
PDE oracle exists: a real American equity option pays discrete cash, which a
continuous yield does not represent.

Per-case rows, provenance digests, and declared limitations:
[docs/results/american_lsm_crosscheck_results_v1.json](docs/results/american_lsm_crosscheck_results_v1.json).

### Label policy

Turning the PDE oracle into a labeling policy took two studies. The first
(task 9C-B) returned a frozen negative — `no_policy_selected` — which is a
complete result, not an unfinished task. The second (task 9C-C3) was separately
predeclared, ran its two stages once each, selected **`grid_1600x800`**, and was
approved by a fresh independent review.

That policy authorizes **labeling only**: price on 22/22 regular cases, delta on
18/22, vega on 22/22, gamma not at all. Its frozen report records
`authorizes_dataset_generation = false` and `authorizes_training_input = false`.
An accepted label policy is not an accepted dataset.

---

## The neural surrogate: Architecture Freeze v2.3

[**docs/american-neural-architecture-freeze-v2.3.md**](docs/american-neural-architecture-freeze-v2.3.md)
is the normative parent of the American neural-pricer roadmap. It is a
contract, and it wins over every other document about the American neural
design. In brief:

- **The teacher is reproducible C++ CRR.** The base label is
  $L_N = \tfrac12[CRR_N + CRR_{N+1}]$, with both components persisted. Depth $N$
  is not a network input, and $CRR_N$ is not recoverable from an $N+1$ tree —
  changing $N$ changes the whole lattice.
- **Base label and deep reference are separate objects.** Final evaluation
  reports the decomposition $V_{NN}-L_N$, $V_{NN}-L_{ref}$, and $L_N-L_{ref}$,
  so teacher discretization is never silently charged to the network.
  Price-reference and per-Greek reference depths may differ, because
  differentiation amplifies finite-depth lattice error differently.
- **Unseen state and deeper tree test different things.** "Out of sample" means
  an unseen option state $x$; changing $N$ is a reference-quality axis.
- **All partitions are generated atomically** — train, validation, development
  holdouts, the neutrally sampled final interpolation set, and a predeclared
  economic stress set — from one invocation of one committed generator
  configuration, bound by one manifest. "Untouched final" becomes a property of
  a precommitted sampling design, not of a second sampling event.
- **Adaptive-development controls are architectural**: append-only attempt
  ledger, declared attempt budget, development holdouts, final-partition access
  guard, one-shot final evaluation, no post-final tuning under the same claim.
- **The model artifact is deterministic and non-pickle**, loadable from C++.
- **Native C++ inference is the production latency path.** The question is not
  whether PyTorch beats CRR; it is whether the exact frozen deployed function,
  implemented natively, beats the matched native CRR request. No speedup is
  asserted until native timings exist.
- **Greek claims are bounded by reference quality.** Agreement with a
  finite-depth lattice Greek does not establish a continuous-American Greek
  unless the reference is itself converged.

The complexity claim the architecture permits, and no more: standard CRR
backward induction is $O(N^2)$ time per option; a frozen network with $P$
parameters has fixed $O(P)$ inference work, which is $O(1)$ with respect to
lattice depth $N$ — not constant with respect to model size.

---

## Current status

| Component | Status |
|---|---|
| Black–Scholes prices and analytic Greeks (C++20) | Complete |
| Smooth `tanh` MLP with reverse-mode input derivatives (C++20) | Complete |
| European dataset generator, diagnostics, training, evaluation | Complete |
| Fresh-seed European replication | **Complete, locked, terminal** |
| CRR tree, LSM cross-check, discrete-dividend PDE oracle (C++20) | Complete |
| PDE label policy | **Accepted `grid_1600x800`** — labeling only |
| Local continuous-yield CRR dataset | Conditionally admitted, **mapping-only**; not regenerable from committed source |
| American neural-pricer feasibility pilot (task 9G) | **Complete — negative.** `validation_gates_failed` / `failure_to_learn`; final partition unconsumed |
| Adaptive development loop (task 9H) | Archived exploratory; roadmap superseded by v2.3; not a project result |
| Architecture Freeze v2.3 Phase 0 (task 9I) | **Active** — Greek/head diagnostics and a native C++ feasibility prototype, in parallel |
| Reproducible American generator, v2 dataset, v2 training | Not started |
| Deterministic artifact export, native C++ inference | Not implemented |
| Swaption stages 3–4 | Not started |

The current, authoritative state and the exact next task:
[docs/project-state.md](docs/project-state.md).

No generated dataset, weight file, raw study report, or market-derived artefact
is checked in. The repository carries versioned snapshots under `docs/results/`
and deterministic, CI-checked SVG figures under `docs/figures/` rendered from
those snapshots alone.

---

## Key limitations

- **No American neural surrogate has been trained and accepted.** Task 9G
  trained two arms and accepted neither. Every entry gate passed — including an
  independent PDE mapping check at 21/21 rows and an exact
  European-to-American transfer lift at eight pinned probes — and then neither
  arm met the predeclared validation gates. That negative is frozen and is
  never reinterpreted as a partial success.
- **One seed and one budget cannot establish a transfer hypothesis.** Task 9G's
  better transfer-arm validation RMSE is confounded by the arm-seeded epoch
  shuffle. A real claim needs at least five seeds and several label budgets.
- **The local CRR dataset is not reproducible from committed source.** Its
  generator lives on an unmerged branch, and its label policy has no frozen
  evidence on any branch. This is exactly why v2.3 makes a reproducible
  generator a structural requirement.
- **Cross-engine agreement on the dataset is thin.** The 21-row PDE check is
  mapping-consistency evidence only — not a dataset-scale cross-check of
  250,000 labels, and not converged American-price truth.
- **No latency claim.** No native inference path exists yet, so no measured
  speedup exists. FLOP and memory-bandwidth arithmetic are plausibility checks,
  never latency results.
- **In-envelope interpolation only.** No boundary, extrapolation/OOD, or
  scenario-shock partition has been built or evaluated.
- **No calibrated curve or volatility surface exists.** The real-market track
  reads three sessions of a proprietary archive read-only and produces no price,
  Greek, label, or calibrated value. Dividend amounts, borrow and carry, and
  corporate-action adjustments are unavailable in it.
- **No production or deployment claim.** C++ artifact loading is not
  implemented; the shipped inference core is `SmoothMlp` alone.

Full, decision-linked list: [docs/project-state.md](docs/project-state.md),
"Current limitations and non-claims".

---

## Build, test, run

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

# Full local gate
./scripts/check.sh
```

The `train` extra adds PyTorch and is required by everything under
`python/tests/ml/`; for dataset work without it install `.[data,dev]`, and the
optional `market` extra adds the vendor DBN reader.

`./scripts/check.sh` runs `ruff check` only when `ruff` is importable and
`clang-format --dry-run --Werror` only when `clang-format` is installed —
**neither is unconditional, and CI does not check C++ formatting at all.** Full
mode always configures and builds C++ fresh, runs the C++ tests, then the full
Python suite; a missing pytest fails the gate rather than skipping it.
`--quick`, used by the opt-in pre-commit hook, skips the Python suite and runs
C++ tests only if `build/check` already exists. **GitHub Actions CI is the
authoritative clean-environment gate**, since local results depend on machine
state. Use `-DCMAKE_BUILD_TYPE=Release` for any recorded performance
experiment.

The pricers emit machine-readable JSON:

```bash
./build/dev/dp_pricer call 100 100 1 0.05 0 0.20
./build/dev/dp_american_pricer american put 100 100 1 0.05 0 0.20 2048
```

The American executable reports the $N$- and $(N+1)$-step prices, their average,
their absolute gap, and exercise-region diagnostics. That adjacent gap is a
refinement diagnostic, **not** a parity measure or a certified error bound.

### Reproduce the European study

```bash
python -m differentiable_pricing.data.generate \
  --config configs/european_option_dataset_v1.toml \
  --output data/european-option-v1

python -m differentiable_pricing.data.diagnose \
  --dataset data/european-option-v1 \
  --output data/european-option-v1/diagnostics.json

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

Generation writes three splits plus a timestamp-free `manifest.json` recording
schema and generator versions, seed derivation, configuration SHA-256, oracle
identity, units, row counts, and a SHA-256 per file, so the same configuration
and toolchain reproduce byte-identical output. Partitions are drawn from
independent streams, not random-split from one table. Training reads and
verifies only `train.parquet` and `validation.parquet` and never opens
`interpolation_test.parquet`. `constrain` is not another training run: it
verifies the source artifact, copies `weights.npz` byte-for-byte, and refuses to
stack the same constraint twice.

The sampling box is a **provisional synthetic engineering range, not a
calibrated market distribution**; extreme-$z$ rows price to within float64 noise
of intrinsic value and must be evaluated as their own slice.

### Reproduce the American reference studies

Deliberately **outside CI** — their production-sized grids are expensive, and
per `AGENTS.md` every long numerical run is a manual, human-invoked terminal
job.

```bash
python -m differentiable_pricing.american.convergence \
  --config configs/american_crr_convergence_v1.toml \
  --output artifacts/american-crr-convergence-v1.json

python -m differentiable_pricing.american.lsm_crosscheck \
  --config configs/american_lsm_crosscheck_v1.toml \
  --output artifacts/american-lsm-crosscheck-v1.json

python -m differentiable_pricing.american.pde_refinement

python scripts/demo_pde_valuation_surface.py
```

Frozen snapshots are refrozen only from a reviewed raw report, by the one
designated script for that family — never by hand:

```bash
python scripts/freeze_american_lsm_results.py --check   # offline, tracked files only
python scripts/freeze_pde_label_policy_v2_results.py --check
python scripts/freeze_american_neural_pilot_results.py --check
```

All three `--check` invocations run in `scripts/check.sh` and in CI. They read
only tracked files and rerun no pricing, training, latency measurement, or IV
inversion.

The market studies need the private archive and the `market` extra, write only
beneath ignored trees, and never run in CI. Their reports and figures derive
from proprietary quote-level data and are never staged.

---

## Documentation

| Document | Contents |
|---|---|
| [docs/project-state.md](docs/project-state.md) | Current state and the exact next task — more current than this file wherever they disagree |
| [docs/american-neural-architecture-freeze-v2.3.md](docs/american-neural-architecture-freeze-v2.3.md) | Normative parent of the American neural-pricer roadmap |
| [docs/research-contract.md](docs/research-contract.md) | Hypotheses, stage reference standards, data protocol, metrics, gates, non-claims |
| [docs/architecture.md](docs/architecture.md) | Language boundary, snapshot discipline, artifact contract, stage-1 model math |
| [docs/american-crr-contract.md](docs/american-crr-contract.md) | Lattice, recursion, exercise metadata, complexity, convergence semantics |
| [docs/american-lsm-contract.md](docs/american-lsm-contract.md) | Estimand, policy/valuation separation, uncertainty, control variate |
| [docs/pde-numerical-contract.md](docs/pde-numerical-contract.md) | PDE oracle, valuation-time surface, harvesting, label policies v1 and v2 |
| [docs/market-state-reconstruction-contract.md](docs/market-state-reconstruction-contract.md) | Parity fitting, identifiability classes, input contract |
| [docs/decision-log.md](docs/decision-log.md) | Append-only record of decisions a later task must not relitigate |
| [docs/documentation-map.md](docs/documentation-map.md) | Which document wins when two disagree |
| [docs/tasks/README.md](docs/tasks/README.md) | Task index and lifecycle |
| [docs/agentic-workflow.md](docs/agentic-workflow.md) | Agent roles, guardrails, review loop |

Three disciplines run through all of them.

**Partition roles are enforced, not assumed.** `validation` selects
architecture and hyperparameters; a final partition must be named explicitly and
is never tuned against. Once any result informs a model change, that split is
consumed and a fresh-seed replication is required for the final estimate.

**Expensive evidence is frozen, not transcribed.** A snapshot family's generator
checks the raw report against an exact key schema, recomputes every summary from
its own rows, extracts each number programmatically, and emits canonical
timestamp-free JSON. Every `--check` runs in CI on checked-in files only, so no
gate depends on an ignored artifact.

**Agent-assisted engineering is part of the method, and it is constrained.** A
deterministic post-edit hook validates edited files without rewriting them, and
two read-only clean-context review agents return findings on a diff without
touching it. They produce evidence, not authority: their findings are
**preliminary**, and deterministic tests, CI, and human ownership of assumptions
and claims remain decisive
([docs/agentic-workflow.md](docs/agentic-workflow.md),
[docs/agent-system.md](docs/agent-system.md)).

---

## Repository layout

```text
cpp/ bindings/python/ python/   C++ core, pybind11 boundary, Python package
configs/                        Versioned experiment assumptions
docs/                           Contracts, decisions, and research protocol
docs/results/ docs/figures/     Frozen snapshots and rendered plots
scripts/                        Developer checks and study runners
.claude/ .githooks/ .github/    Review agents, local hooks, CI
```

**C++** owns reference pricing and analytic sensitivities, CRR early-exercise
pricing with a parallel price-only batch boundary, LSM policy fitting and
valuation, the finite-difference PDE oracle with discrete cash dividends and its
valuation-time surface, input validation, and reverse-mode derivatives of the
deployed network. **Python** owns sampling, partitioning, lineage, training,
evaluation, study runners, and the read-only market pipelines. Pricing formulas
are never duplicated in Python to make a test pass.

---

## Contributing

Read [AGENTS.md](AGENTS.md) (or [CLAUDE.md](CLAUDE.md) if you are using Claude
Code), [docs/project-state.md](docs/project-state.md), and
[docs/documentation-map.md](docs/documentation-map.md) first. Every change
should state the numerical assumption it changes, add a regression test, and
pass `./scripts/check.sh`. Full workflow, test-partition rules, snapshot refresh
procedure, and pull-request expectations: [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT; see [LICENSE](LICENSE).
