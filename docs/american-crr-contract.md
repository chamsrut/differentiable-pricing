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
\Phi(S_{n,j})-C_{n,j}
>
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
without regression or sampling noise. Least-Squares Monte Carlo (LSM) will be
implemented as a separate engine after the tree contract is stable.

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

- No American Greek is exposed yet.
- The batch boundary is price-only; American Greeks are not exposed.
- The checked-in convergence configuration is exploratory and does not yet
  authorize a label step count.
- Runtime measurements are machine-specific evidence and are not CI gates.
- No American training dataset or neural result exists yet.
- Continuous exercise is approximated by exercise at every lattice time
  layer and requires step convergence.
- Inputs are constant $r$, $q$, and $\sigma$; there is no calibrated
  curve or volatility surface.
- This is model-value infrastructure, not evidence of market fit or trading
  alpha.
