# American-option CRR numerical contract

## Purpose and scope

The Cox--Ross--Rubinstein (CRR) engine is the first deterministic reference
pricer for the American-option stage. It prices European and American vanilla
calls and puts under the same constant-parameter lognormal dynamics as the
Black--Scholes implementation:

\[
\frac{dS_t}{S_t} = (r-q)\,dt + \sigma\,dW_t.
\]

It is deliberately a scalar reference implementation. It establishes
exercise logic, convergence tests, and permanent analytic fixtures before
batch generation or neural training is introduced.

## Lattice and recursion

For \(N\) time steps,

\[
\Delta t=\frac{T}{N}, \qquad
u=e^{\sigma\sqrt{\Delta t}}, \qquad d=u^{-1},
\]

\[
p=\frac{e^{(r-q)\Delta t}-d}{u-d}, \qquad
D=e^{-r\Delta t}.
\]

The implementation requires \(0<p<1\). A rejected coarse tree is not repaired
by clipping \(p\): increase the step count or revise inputs until the lattice
is compatible with the declared model.

At maturity,

\[
V_N(S)=
\begin{cases}
\max(S-K,0), & \text{call},\\
\max(K-S,0), & \text{put}.
\end{cases}
\]

European backward induction uses

\[
C_{n,j}=D\left[pV_{n+1,j+1}+(1-p)V_{n+1,j}\right],
\qquad V_{n,j}=C_{n,j}.
\]

For an American option,

\[
V_{n,j}=\max\left(\Phi(S_{n,j}), C_{n,j}\right).
\]

The price recursion uses the exact floating-point maximum above. Exercise
metadata is deliberately more conservative: it records exercise only when

\[
\Phi(S_{n,j})-C_{n,j}
>
10^{-12}\max\left(1,\lvert\Phi(S_{n,j})\rvert,\lvert C_{n,j}\rvert\right).
\]

This scale-aware indifference band prevents roundoff at an economically tied
node from creating a false exercise frontier without changing the computed
price. The reported earliest exercise step is a time index, with zero denoting
the valuation date; it is not a fitted or smoothed boundary.
The library result also stores one optional boundary point per pre-expiry
layer: the greatest exercised spot for a put and the smallest exercised spot
for a call. The CLI reports how many such layers exist rather than emitting
the full vector.

## Complexity and determinism

The tree uses \(O(N^2)\) arithmetic and rolling \(O(N)\) memory rather than
storing the \(O(N^2)\) lattice. European pricing allocates no exercise-boundary
or backward spot-state vector. Given the same floating-point toolchain and
inputs, it is deterministic and uses no random numbers.

The scalar `crr_binomial` API rejects more than 16,384 steps. The adjacent-step
API—and therefore the CLI—accepts at most 16,383 because it also computes
\(N+1\). These operational ceilings do not imply that every smaller request is
fast or numerically accurate. Single-configuration CMake builds default to
`RelWithDebInfo`; use an explicit `Release` configuration for recorded
performance experiments. Task 8B must record hardware, build configuration,
throughput, and uncertainty rather than turning a machine-specific time into a
portable CI threshold.

## Convergence output

`crr_adjacent_step_estimate` computes separate \(N\)- and \((N+1)\)-step
trees and reports

\[
\bar V_N=\frac{V_N+V_{N+1}}{2},
\qquad
g_N=\lvert V_N-V_{N+1}\rvert.
\]

CRR prices can oscillate with step parity because the strike moves relative to
the terminal lattice. The adjacent average often reduces that visible
oscillation. Neither \(g_N\) nor \(g_N/2\) is a certified truncation-error
bound. Production of learning labels requires a separately versioned
convergence study over the full parameter domain, including exercise-frontier
and short-maturity cases. It must also map the minimum feasible step count
under the strict \(0<p<1\) rule. A batch generator must fail or flag an
unsupported state explicitly; it must never silently drop rejected rows.

## Permanent validation properties

The deterministic C++ suite checks:

- European-tree convergence to analytic Black--Scholes prices;
- equality of European and American calls when \(q=0\) and \(r>0\);
- a positive early-exercise premium for a representative American put;
- early exercise for a representative high-dividend American call;
- intrinsic and simple upper bounds;
- spot monotonicity;
- suppression of spurious exercise diagnostics at zero-rate indifference;
- improved price stability under step refinement;
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
- No batch API or Python binding is exposed yet.
- No American training dataset or neural result exists yet.
- Continuous exercise is approximated by exercise at every lattice time
  layer and requires step convergence.
- Inputs are constant \(r\), \(q\), and \(\sigma\); there is no calibrated
  curve or volatility surface.
- This is model-value infrastructure, not evidence of market fit or trading
  alpha.
