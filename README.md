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
- Deterministic repository checks, CI, a Claude Code post-edit hook, and two
  read-only clean-context review agents.

The training pipeline is intentionally the next milestone, rather than
placeholder code pretending experiments have already been run.

## Repository map

```text
cpp/                    C++ pricing and inference core
bindings/python/        pybind11 boundary
python/                 Python package and tests
configs/                Versioned experiment assumptions
docs/                   Architecture and research contract
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

Try the executable:

```bash
./build/dev/dp_pricer call 100 100 1 0.05 0 0.20
```

## Build the Python package

The editable install compiles the C++ extension:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,data]'
pytest -q
ruff check .
```

On Windows PowerShell, activate with
`.venv\Scripts\Activate.ps1`.

`./scripts/check.sh` runs this same Python suite after the C++ tests, so it
needs the editable install above. The `--quick` mode used by the pre-commit
hook stops after the C++ tests.

Training dependencies are separate because PyTorch is not needed to use the
pricing library:

```bash
python -m pip install -e '.[train,dev]'
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

The stage-1 generator now exists. Next: add the boundary, extrapolation, and
scenario partitions the research contract requires, then define the
price/Greek/latency acceptance gates before any training run. That keeps the
neural network from becoming an impressive-looking answer to an
underspecified question.

## License

MIT; see [LICENSE](LICENSE).
