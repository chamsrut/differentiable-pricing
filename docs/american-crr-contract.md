# American-option CRR numerical contract

## Purpose and scope

The Cox--Ross--Rubinstein (CRR) engine is the first deterministic reference
pricer for the American-option stage. It prices European and American vanilla
calls and puts under the same constant-parameter lognormal dynamics as the
Black--Scholes implementation:

$$
\frac{dS_t}{S_t} = (r-q)\,dt + \sigma\,dW_t.
$$

It is deliberately a scalar reference implementation. It establishes
exercise logic, convergence tests, and permanent analytic fixtures before
batch generation or neural training is introduced.

## Lattice and recursion

For $N$ time steps,

$$
\Delta t=\frac{T}{N}, \qquad
u=e^{\sigma\sqrt{\Delta t}}, \qquad d=u^{-1},
$$

$$
p=\frac{e^{(r-q)\Delta t}-d}{u-d}, \qquad
D=e^{-r\Delta t}.
$$

The implementation requires $0<p<1$. A rejected coarse tree is not repaired
by clipping $p$: increase the step count or revise inputs until the lattice
is compatible with the declared model.

At maturity,

$$
V_N(S)=
\begin{cases}
\max(S-K,0), & \text{call},\\
\max(K-S,0), & \text{put}.
\end{cases}
$$

European backward induction uses

$$
C_{n,j}=D\left[pV_{n+1,j+1}+(1-p)V_{n+1,j}\right],
\qquad V_{n,j}=C_{n,j}.
$$

For an American option,

$$
V_{n,j}=\max\left(\Phi(S_{n,j}), C_{n,j}\right).
$$

The price recursion uses the exact floating-point maximum above. Exercise
metadata is deliberately more conservative: it records exercise only when

$$
\Phi(S_{n,j}) - C_{n,j} >
10^{-12}\max\left(1,\lvert\Phi(S_{n,j})\rvert,\lvert C_{n,j}\rvert\right).
$$

This scale-aware indifference band prevents roundoff at an economically tied
node from creating a false exercise frontier without changing the computed
price. The reported earliest exercise step is a time index, with zero denoting
the valuation date; it is not a fitted or smoothed boundary.
The library result also stores one optional boundary point per pre-expiry
layer: the greatest exercised spot for a put and the smallest exercised spot
for a call. The CLI reports how many such layers exist rather than emitting
the full vector.

## Complexity, batching, and determinism

The tree uses $O(N^2)$ arithmetic and rolling $O(N)$ memory rather than
storing the $O(N^2)$ lattice. European pricing allocates no exercise-boundary
or backward spot-state vector. Given the same floating-point toolchain and
inputs, it is deterministic and uses no random numbers.

The price-only API runs the identical node recursion without retaining the
exercise boundary. The batch API validates every request serially before
starting workers, preserves input order, and assigns independent contracts to
up to the requested number of worker threads. The effective count is
$W=\min(\text{requested threads},B)$ for a non-empty batch of $B$ rows.
Every worker reuses private rolling value and spot buffers, so total storage is
$O(B+WN_{\max})$ rather than $O(BN_{\max})$. A worker processes each tree
serially: changing the worker count does not alter the floating-point operation
order within a price, and serial and parallel results are required to be
bit-identical.

Batch parallelism reduces wall-clock time but not total work:

$$
\text{work}=O\left(\sum_{i=1}^{B}N_i^2\right).
$$

Parallelizing the nodes inside one tree is deliberately deferred. The
in-place recursion would need double buffering and a synchronization barrier
at every time layer. That can be revisited only if profiling shows single-point
latency, rather than batch throughput, is the relevant bottleneck.

The scalar `crr_binomial` API rejects more than 16,384 steps. The adjacent-step
API—and therefore the CLI—accepts at most 16,383 because it also computes
$N+1$. These operational ceilings do not imply that every smaller request is
fast or numerically accurate. Single-configuration CMake builds default to
`RelWithDebInfo`; use an explicit `Release` configuration for recorded
performance experiments. Task 8B must record hardware, build configuration,
throughput, and uncertainty rather than turning a machine-specific time into a
portable CI threshold.

## Convergence output

`crr_adjacent_step_estimate` computes separate $N$- and $(N+1)$-step
trees and reports

$$
\bar V_N=\frac{V_N+V_{N+1}}{2},
\qquad
g_N=\lvert V_N-V_{N+1}\rvert.
$$

CRR prices can oscillate with step parity because the strike moves relative to
the terminal lattice. The adjacent average often reduces that visible
oscillation. For an American option, $N$ versus $N+1$ also changes the
exercise-time mesh and discrete frontier, so $g_N$ is an
adjacent-refinement discrepancy rather than a pure parity diagnostic. Neither
$g_N$ nor $g_N/2$ is a certified truncation-error bound. The exploratory
study in
`configs/american_crr_convergence_v1.toml` covers named exercise regimes and
uses an 8,192/8,193 adjacent average as an internal high-step reference. It
also prices the corresponding European contracts against analytic
Black--Scholes values, checks both signs of rates for no-dividend calls,
samples exercise boundaries across multiple refinements, and maps the minimum
feasible *tested candidate* step count over a separate grid in
$(T,r,q,\sigma)$. The report explicitly does not treat the same-engine
high-step value as independent truth.

Production of learning labels still requires a separately versioned sampling
and acceptance protocol over the complete proposed data domain, including
exercise-frontier and short-maturity cases. A batch generator must fail or
flag an unsupported state explicitly; it must never silently drop rejected
rows.

## Label policy v1

**Recovered by task 9E.** This section was written on the unmerged branch
`feat/american-dataset-v1` alongside the generator that produced
`data/american-option-v1/`, and was brought onto this branch by task 9E so that
the label policy the dataset declares is under contract here. Its content is
unchanged except for one cross-reference to the preceding section.

**Evidence status, stated plainly.** The two pilot reports this section
describes were **ignored local artifacts under `artifacts/`, never frozen
`docs/results/` snapshots**, and they remain so: no `docs/results/` snapshot
exists for this label policy on any branch. The pilot configurations are now
tracked (`configs/american_label_policy_pilot_v1.toml`,
`configs/american_label_policy_pilot_v2.toml`) and their digests match what the
reports record, but the reports themselves are not project evidence in the sense
a frozen snapshot is. Read every number below as a recovered claim, not as
frozen evidence.

`american-crr-adjacent-average/1` labels a contract state with

$$
\bar V_N=\frac{V_N^{\mathrm{A}}+V_{N+1}^{\mathrm{A}}}{2},
\qquad N=1024,
$$

and records both components, their gap $g_N$, and the identically averaged
European CRR price $\bar V_N^{\mathrm{E}}$ from the same lattice. The stored
early-exercise premium is $\bar V_N-\bar V_N^{\mathrm{E}}$: a difference between
two identically discretized quantities, not a premium over an exact European
value. Because the American value is a nodewise maximum over the European one on
one lattice and rounding to nearest is monotone, that premium is non-negative
**exactly**, with no tolerance.

The analytic European price is stored alongside as the one genuinely
independent control column; every other price in the table shares the labelling
lattice.

### How $N$ was chosen

The exploratory convergence study above covers named exercise regimes. A production label
policy has to hold over the whole sampling domain, so the step count was chosen
by a separate pilot that draws 2,048 rows from the production mixture through a
stream of its own, `label_policy_pilot_v1`, and therefore neither consumes nor
overlaps any partition. Errors are normalised by spot so they are directly
comparable with the price-over-spot surrogate gates in
`configs/european_neural_acceptance_v1.toml`.

**Pilot v1 (`configs/american_label_policy_pilot_v1.toml`) failed.** It was
predeclared, executed once over the ladder $\{256,512,1024,2048\}$ against an
8,192/8,193 reference, and **no candidate met its gates**. The binding gate was
a p99 discrepancy of $10^{-5}$; the best candidate reached
$1.17\times10^{-5}$. That configuration and its report are kept unchanged as
the recorded failure.

**Pilot v2 (`configs/american_label_policy_pilot_v2.toml`) made the
selection.** It was written *after* v1's results were seen and is therefore
**not an uncontaminated predeclaration**; the file says so in its own header.
What changed:

- the worst-row gate, the European control, and the theorem control are
  **byte-for-byte the v1 values**;
- the p99 gate moved from $10^{-5}$ to $2\times10^{-4}$. The v1 value contained
  a category error rather than a stricter judgement: it compared a p99 label
  statistic against a threshold derived from the surrogate *RMSE* gate
  ($5\times10^{-4}$) rather than from the surrogate's own p99 gate
  ($2\times10^{-3}$);
- the ladder was extended to 4,096 and the internal reference strengthened to
  16,383/16,384, which makes every measured discrepancy **larger**, not smaller.

| $N$ | max $\lvert\bar V_N-\bar V_{\mathrm{ref}}\rvert/S$ | p99 | mean | max European vs Black--Scholes | under v2 gates | under v1 gates |
|---|---|---|---|---|---|---|
| 512 | $7.88\times10^{-5}$ | $5.62\times10^{-5}$ | $1.36\times10^{-5}$ | $6.78\times10^{-5}$ | fails | fails |
| **1024** | $3.25\times10^{-5}$ | $2.63\times10^{-5}$ | $6.64\times10^{-6}$ | $3.43\times10^{-5}$ | **selected** | fails p99 |
| 2048 | $2.64\times10^{-5}$ | $1.26\times10^{-5}$ | $3.09\times10^{-6}$ | $1.58\times10^{-5}$ | passes | fails p99 |
| 4096 | $8.30\times10^{-6}$ | $6.15\times10^{-6}$ | $1.38\times10^{-6}$ | $7.98\times10^{-6}$ | passes | **would be selected** |

v2 gates: max $\le5\times10^{-5}$, p99 $\le2\times10^{-4}$, European control
$\le5\times10^{-5}$, non-dividend call control $\le10^{-12}$. v1 gates are
identical except p99 $\le10^{-5}$. The selection rule is the smallest candidate
meeting every gate.

### The relaxed threshold changed the answer

**Read the last column before quoting $N=1024$.** Re-scoring this same pilot run
under v1's original p99 gate of $10^{-5}$ — every other gate unchanged — the
smallest candidate meeting every gate is $N=4096$, not $N=1024$. The relaxation
from $10^{-5}$ to $2\times10^{-4}$ is therefore **not** a neutral correction
that happened to leave the outcome alone: it is what makes a step count four
times smaller, and roughly sixteen times cheaper per row, admissible.

Both readings are on the table and the difference is real:

- $N=1024$ is what the *corrected* derivation selects. That derivation is
  defensible on its own terms: a p99 label statistic belongs against the
  surrogate's own p99 acceptance gate ($2\times10^{-3}$), not against its RMSE
  gate ($5\times10^{-4}$) divided by fifty.
- $N=4096$ is what survives the *uncontaminated* numbers, the ones written down
  before any result was seen.

The worst-row gate — the one number that never moved — is satisfied from
$N=1024$ upward, and it is what rules out $N=512$. Everything above $N=1024$ is
a question of how much label-noise head room the project wants to buy, and at
what compute cost. That is a project decision, recorded here rather than
resolved by a threshold edit.

### What the pilot does and does not establish

Averaging earns its place rather than being assumed: at $N=1024$ the
adjacent average halves the discrepancy of the single $N$-step price, mean
$6.64\times10^{-6}$ against $1.32\times10^{-5}$ and max $3.25\times10^{-5}$
against $7.38\times10^{-5}$.

Both controls hold exactly. The minimum early-exercise premium is $0$ at every
candidate, never negative, and the non-dividend call control at a non-negative
rate is $0$ to the last bit.

The per-row gap is **not** an error bar. Across pilot rows the ratio
$\lvert\bar V_N-\bar V_{\mathrm{ref}}\rvert/(g_N/2)$ has median $0.66$ but a
95th percentile near $9.8$ and a maximum in the hundreds. Half the gap
understates the measured discrepancy for a substantial minority of states, so
$g_N/2$ must be read as a cheap per-row stability signal, never quoted as a
bound.

The reference remains a same-engine high-step average. The only independent
number in the study is the European CRR leg against analytic Black--Scholes,
and it bounds the discretization error of the lattice the American branch
shares. The discrepancy is broadly spread across strata rather than
concentrated: at $N=1024$ every stratum's mean lies within roughly one order of
magnitude of every other, which is why extending the ladder, not a per-stratum
step rule, was the remedy for v1's failure.

### The pilot's tail statistics are estimated from 2,048 rows

A worst-row statistic over 2,048 draws is not a worst-row statistic over
250,000, and the production dataset measures the gap directly. The European
control is the one pilot quantity that can be recomputed on every production
row, because it needs no high-step reference:

| | pilot, 2,048 rows | production, 250,000 rows |
|---|---|---|
| max European CRR vs Black--Scholes, over spot | $3.43\times10^{-5}$ | $5.09\times10^{-5}$ |

The production worst row is $1.48\times$ the pilot's, and **2 rows in 250,000
(0.0008%), both in `long_maturity`, exceed the $5\times10^{-5}$ gate the pilot
passed**, by 1.8%. That is not a dataset defect — this comparison is a
descriptive control column, not a gate the dataset is required to clear — but it
is direct evidence that a 2,048-row pilot understates the extreme tail of a
250,000-row draw by roughly half an order of magnitude.

Read the worst-row numbers in the selection table as estimates with that much
slack, not as bounds. The same caveat applies to the adjacent-average
discrepancy, whose production tail cannot be measured without a 16,383-step
reference over every row, which is why it is not reported here. A pilot with
several independent streams, reporting the spread of the worst-row statistic
rather than one point estimate, would close this gap and is the obvious next
refinement if the step count is ever revisited.

## Permanent validation properties

The deterministic C++ suite checks:

- European-tree convergence to analytic Black--Scholes prices;
- equality of European and American calls when $q=0$ and $r>0$;
- a positive early-exercise premium for a representative American put;
- early exercise for a representative high-dividend American call;
- intrinsic and simple upper bounds;
- spot monotonicity;
- suppression of spurious exercise diagnostics at zero-rate indifference;
- improved price stability under step refinement;
- exact parity among the diagnostic, price-only, serial-batch, and
  parallel-batch paths;
- deterministic batch ordering and indexed failure of invalid rows;
- rejection of zero/excessive step counts and invalid risk-neutral
  probabilities.

These properties are independent of a hard-coded American price produced by
the same implementation. Later reference snapshots must record the step
policy and convergence evidence rather than presenting one CRR output as
exact truth.

## Why CRR precedes Least-Squares Monte Carlo

CRR is the stronger first oracle in this one-factor Markov setting: it is
deterministic, exposes the exercise decision at every node, and can be refined
without regression or sampling noise. Least-Squares Monte Carlo (LSM) was
implemented afterwards as a separate engine, once the tree contract was stable
([american-lsm-contract.md](american-lsm-contract.md)).

The LSM implementation must:

- estimate the continuation policy on training paths;
- value the frozen policy on independent paths;
- report path-count, time-step, basis, and seed sensitivity;
- cross-check prices and exercise behaviour against converged CRR results in
  their shared one-factor domain.

LSM becomes the practical reference when path dependence or multiple state
variables make recombining trees unsuitable. Agreement with a same-model CRR
tree is a validation step, not evidence that either model matches market
prices.

## Current non-claims

- No American Greek is exposed by this engine, and none by the finite-difference
  oracle either. The task 9C-B pilot obtained Greeks by bumping and repricing
  outside both engines, and selected no policy.
- The batch boundary is price-only; American Greeks are not exposed.
- The checked-in convergence configuration is exploratory and does not yet
  authorize a label step count.
- This engine carries a continuous dividend yield. It does not price an
  explicit cash dividend; that contract belongs to the finite-difference oracle
  ([pde-numerical-contract.md](pde-numerical-contract.md)).
- Runtime measurements are machine-specific evidence and are not CI gates.
- No American neural result exists yet. **Task 9E conditionally qualifies the
  dataset only for learning the known CRR mapping**:
  `data/american-option-v1/` is a prospective phase-1 input
  ([american-crr-dataset-admission.md](american-crr-dataset-admission.md),
  [decision-log.md](decision-log.md) DEC-033, DEC-034). This is not acceptance
  of converged American-price accuracy. It is Git-ignored, not
  regenerable from tracked sources, its label policy has no frozen evidence, and
  its admission authorizes no training. The independent numerical cross-check,
  missing row/manifest policy invariants, and task 9G entry gates remain
  outstanding. **No American neural result exists**, and no network has been
  trained.
- Continuous exercise is approximated by exercise at every lattice time
  layer and requires step convergence.
- Inputs are constant $r$, $q$, and $\sigma$; there is no calibrated
  curve or volatility surface.
- This is model-value infrastructure, not evidence of market fit or trading
  alpha.
