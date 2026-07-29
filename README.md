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
python -m pip install -e '.[dev]'
pytest -q
ruff check .
```

On Windows PowerShell, activate with
`.venv\Scripts\Activate.ps1`.

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

Build the stage-1 data generator and write rows to a versioned Parquet schema.
Before training anything, define parameter bounds, sampling distributions,
train/validation/test regions, and the price/Greek/latency acceptance gates.
That keeps the neural network from becoming an impressive-looking answer to an
underspecified question.

## License

MIT; see [LICENSE](LICENSE).
