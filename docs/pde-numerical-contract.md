# Finite-difference PDE numerical contract (task 9C-A)

## Purpose and scope

`dp::finite_difference_price` is a deterministic one-dimensional
finite-difference oracle for European and American vanilla calls and puts with
**explicit discrete cash dividends**. It is the first engine in this repository
that prices a contract the CRR and LSM engines cannot: those two carry a
continuous dividend yield, and a cash dividend is not a yield.

**Task 9C-A supports deterministic pricing only.** Greeks, optimized batching,
label-policy selection and frozen acceptance gates are task 9C-B. Nothing in
this document is an acceptance gate, and the first grid policy is explicitly
**not** claimed to be label-grade: no grid, step count, domain maximum or PSOR
setting here has been justified as fit for generating training labels.

The public surface is `cpp/include/dp/finite_difference_pde.hpp` and the `_pde`
Python extension. The oracle is bound from `bindings/python/pde_module.cpp`,
its own translation unit, so the CRR and LSM implementation digests that the
frozen American LSM cross-check snapshot records stay byte-stable.

## Equation and sign conventions

Between dividend dates the value $V(t,S)$ satisfies

$$
V_t+\tfrac12\sigma^2S^2V_{SS}+(r(t)-c)SV_S-r(t)V=0,
$$

solved backward from expiry $T$ to the valuation time $t_0$.

- $r(t)$ is the instantaneous rate implied by the declared discount curve. It
  is never a scalar argument and never defaults.
- $c$ is the **continuous carry**, subtracted from the risk-neutral drift. It
  is carried as `std::optional<double>` and a disengaged optional is rejected:
  a caller with no borrow or carry information must say so rather than pass a
  zero that reads as an observation.
- $\sigma$ is constant for task 9C-A.
- Times are ACT/365F year fractions measured from the discount curve's origin.
  `valuation_time` may be nonzero; `expiry_time` must be strictly later.
- Prices are **per share**. `contract_multiplier` and `settlement` are carried
  because the task 9B input contract lists them, and neither enters any
  pricing arithmetic. A regression test asserts that changing the multiplier
  does not change the price.

Nothing defaults. A missing rate, carry, dividend list, settlement convention
or multiplier is an error.

## Discount curve and interpolation

The curve is a pair of vectors: strictly increasing ACT/365F `times` and
`log_discounts`, beginning exactly at $(0,0)$.

Log discounts are interpolated **linearly**, so the discount factor is the
exponential of an interpolation in log space. Positivity therefore holds by
construction and a zero rate is a zero slope rather than a quotient. The
implied instantaneous rate is piecewise constant, equal on the segment
$[\tau_k,\tau_{k+1}]$ to

$$
r_k=-\frac{L_{k+1}-L_k}{\tau_{k+1}-\tau_k}.
$$

The curve must **bracket** $[t_0,T]$. Evaluation outside the declared knots
throws; it is never extrapolated, and a curve that stops before expiry is
rejected rather than extended flat.

## Cash dividends and the jump condition

A cash dividend of amount $d>0$ at ex-time $t_d$ maps the spot as

$$
S_{t_d^+}=\max(S_{t_d^-}-d,0),
$$

so the value function is carried backward across the event by

$$
V(t_d^-,S)=V\bigl(t_d^+,\max(S-d,0)\bigr)
$$

for a European option and

$$
V(t_d^-,S)=\max\Bigl(\Phi(S),\;V\bigl(t_d^+,\max(S-d,0)\bigr)\Bigr)
$$

for an American option, where $\Phi$ is the exercise payoff. The American form
is what makes **exercise immediately before the ex-date** available, which is
the only reason a non-dividend-paying American call would ever be exercised
early.

The shifted spot generally falls between grid nodes. It is interpolated with
**Fritsch--Carlson monotone cubic Hermite** interpolation. The limiter is
load-bearing: an unlimited cubic through option values can overshoot near the
payoff kink and manufacture a local arbitrage that no later time step removes.

Rejected, each with its own message: a nonpositive or non-finite amount (a zero
dividend is an event that does nothing, which is a different statement from
declaring no event), an ex-time outside the **open** interval $(t_0,T)$, an
unsorted schedule, and a repeated ex-time. An empty schedule must be stated
through `CashDividendSchedule::declared_none()`; a default-constructed schedule
is *undeclared* and is rejected.

## Spatial discretization and boundaries

The spot grid is uniform on $[0,S_{\max}]$ with $S_i=ih$.

`spot_intervals` and `spot_maximum` are **targets**. The solver places a node
exactly on the strike — the terminal payoff kink is the dominant error source
for a vanilla contract, and letting it fall at an arbitrary point inside a cell
makes the observed convergence order oscillate with the grid — which perturbs
both. `PdeResult` reports `spot_intervals`, `spot_maximum`, `spot_step` and
`strike_node_index` as actually used, and a refinement ladder must quote those.

**The grid is not scaled to the expiry, and the solver does not detect an
under-resolved one.** Accuracy requires the spot step to be small against the
diffusion width $\sigma S\sqrt{T-t_0}$. For a one-year at-the-money option at
$\sigma=0.2$ that width is $20$ and a step of $0.5$ is comfortable; for a
2.6-hour option it is $0.35$ and the same step of $0.5$ prices the option $30\%$
low while still reporting
`solver_status = discrete_system_converged` and
`discretization_accuracy = not_assessed`. Nothing in the engine flags this — it
is ordinary discretization error, not a failure of the implicit linear or LCP
solve — so the caller owns the choice and a refinement ladder is the only thing
that exposes it. A regression test asserts both sides of this: the resolved
grid matching Black--Scholes, and the under-resolved grid being visibly wrong
while carrying the explicit `not_assessed` discretization status.

Interior rows use central differences,

$$
l_i=\tfrac12\sigma^2i^2-\tfrac12(r-c)i,\qquad
u_i=\tfrac12\sigma^2i^2+\tfrac12(r-c)i,\qquad
d_i=-(l_i+u_i)-r,
$$

which are independent of $h$. Where central differencing would make $l_i$ or
$u_i$ negative — that is where convection dominates diffusion, $|r-c|>\sigma^2
i$, which can only happen at the smallest few $i$ — that row switches to
one-sided upwinding in the drift direction. This keeps every off-diagonal
non-negative, which is what makes the implicit operator an M-matrix and PSOR
monotone. `PdeResult::upwinded_rows` reports the largest number of rows
upwinded in any one time segment; a nonzero value is a fact about the grid, not
a warning.

Boundaries:

- **$S=0$** needs no imposed condition. Both the diffusion and the convection
  term carry a factor of $S$ and vanish, so the equation degenerates to
  $V_t-rV=0$ and node $0$ is stepped as that ordinary differential equation
  inside the same system. For an American put the obstacle then correctly
  holds the node at $K$.
- **$S=S_{\max}$** is Dirichlet. A deep out-of-the-money put is worth zero. A
  deep in-the-money call is worth the discounted forward intrinsic, which the
  discrete dividends reduce by their carried-back present value:

  $$
  V(t,S_{\max})=S_{\max}e^{-c(T-t)}-\sum_{t_d>t}d_j\,P(t,t_d)\,e^{-c(T-t_d)}-KP(t,T),
  $$

  floored at zero and, for an American call, raised to $S_{\max}-K$. This is
  exact only in the limit $S_{\max}\to\infty$, so **domain truncation is a
  reported sensitivity, not a free choice**. The refinement study runs a
  truncation ladder and both test suites assert that widening the domain from
  $4K$ to $8K$ moves the price by less than $10^{-6}$ while a domain of
  $1.5K$ is measurably more truncated.

## Time discretization and Rannacher damping

Every dividend ex-time and every discount-curve knot strictly inside
$(t_0,T)$ becomes an endpoint of a uniformly stepped segment, so **no time step
ever straddles a rate change or a jump**. The requested `time_steps` are
distributed across segments in proportion to their length with at least one
step each; the total actually taken is reported as `time_steps`, and
`aligned_times` reports the segment endpoints. A request with fewer steps than
segments is rejected rather than silently coarsened.

Each step advances the $\theta$-scheme

$$
(I-\theta\,\Delta t\,L)\,V^{n}=(I+(1-\theta)\,\Delta t\,L)\,V^{n+1},
$$

with $\theta=\tfrac12$ (Crank--Nicolson). Because every off-diagonal is
non-negative, diagonal dominance of the implicit operator reduces to
$1+\theta\,\Delta t\,r>0$; a step size that breaks it is rejected.

**Rannacher damping.** The terminal payoff and every dividend jump are
non-smooth, and Crank--Nicolson does not damp the high-frequency components
that non-smoothness excites — the price still converges but the observed order
degrades, and derivatives degrade further. Following Giles and Carter, the
first `rannacher_steps` grid steps after the terminal payoff **and after every
dividend jump** are each replaced by two fully implicit half steps. The default
in every fixture here is `rannacher_steps = 2`, that is four implicit half
steps per restart. `PdeResult` reports `damped_half_steps` and
`crank_nicolson_steps` separately. Setting `rannacher_steps = 0` is permitted
so the effect can be measured.

The damping budget is counted in **steps taken since the restart**, not per
aligned segment. If the segment a restart falls in has fewer steps than
`rannacher_steps`, the remainder is spent on the following segment. That is
deliberate: the damping has to cover the first steps after the non-smooth
event, and a curve knot happening to fall nearby is not a reason to stop
damping early.

The valuation-to-expiry span must exceed the time-alignment tolerance. Below it
the two instants merge into one aligned time, leaving no segment to step over;
that is rejected rather than allowed to return the undiscounted terminal
payoff.

## The American obstacle: LCP and PSOR

American exercise is not a max applied after a linear solve; it is a linear
complementarity problem. With obstacle $\psi_i=\Phi(S_i)$, each step solves

$$
Ax\ge b,\qquad x\ge\psi,\qquad (Ax-b)_i\,(x-\psi)_i=0 .
$$

The solver is projected successive over-relaxation, chosen as the initial
transparent deterministic method rather than the fastest one:

$$
x_i\leftarrow\max\!\left(\psi_i,\;x_i+\omega\,\frac{b_i-(Ax)_i}{A_{ii}}\right),
$$

sweeping in fixed index order from the previous time level clamped to the
obstacle. Operator splitting methods in the Ikonen--Toivanen family are the
natural next step and are not implemented here.

**Convergence is checked on the LCP residual, not on the iterate change.** The
natural residual

$$
\rho=\max_i\bigl|\min\bigl(x_i-\psi_i,\;(Ax-b)_i\bigr)\bigr|
$$

is zero exactly at a solution of the complementarity system: it is nonzero if
feasibility fails on either side *or* if complementarity fails with both slacks
positive, which an iterate-change test cannot see. It is normalised by
$\max(1,\lVert b\rVert_\infty)$ and compared against `psor_tolerance`.
`PdeResult` reports both the absolute and the normalised maximum over every
solve.

Exceeding `psor_maximum_iterations` throws `dp::PdePsorFailure`
(`PsorConvergenceError`, a `RuntimeError` subclass, in Python), carrying the
segment index, the iteration count, the residual reached and the tolerance. A
non-converged discrete-system price is **never returned**. `PdeSolverStatus`
therefore distinguishes only the operational solve result: a returned result
always carries `discrete_system_converged`, while
`psor_iteration_limit_exceeded` is observable only on that exception.

This solver status is deliberately not an accuracy claim. `PdeResult` also
reports `PdeDiscretizationAccuracy::not_assessed`, exposed in Python as
`discretization_accuracy = "not_assessed"`, until an external refinement or
reference comparison has measured continuum discretization error for the
requested contract. The exploratory refinement report includes both fields on
each rung for the same reason.

A European step is not an LCP and is solved directly by the Thomas algorithm.
The same residual measurement is still taken — without an obstacle it
degenerates to $\lVert Ax-b\rVert_\infty$ — so the reported field is meaningful
for both styles.

**One consequence worth stating plainly:** `psor_tolerance` bounds the residual
of a *single* implicit solve, so the solver's contribution to a price
accumulates over the time steps. At the settings used in this repository's
tests ($10^{-11}$ relative, 400 steps) it is observed near $10^{-7}$, and
identities that are exact in exact arithmetic — an American call with
non-negative rates, zero carry and no dividend equalling its European value —
are asserted within that band rather than to machine precision.

## Determinism, complexity and memory

The solver is single-threaded, has no randomness, sweeps in a fixed order and
reassociates no reduction, so repeated runs on the same inputs are **bitwise**
identical. Tests assert bitwise equality of the price, the PSOR iteration total
and the residual.

Let $N$ be the spot intervals actually used, $M$ the time steps actually taken
and $D$ the number of declared dividends. Work is $O(NM+MD)$ for a European
contract and $O(NM\bar{k}+MD)$ for an American one, where $\bar{k}$ is the mean
PSOR sweep count per step (reported as `psor_total_iterations` and
`psor_maximum_iterations_used`). The $MD$ term is the call's upper-boundary sum
over the dividends still ahead, evaluated once per implicit step; it is
negligible whenever $D\ll N$, which the $D\le 512$ ceiling makes the normal
case, and it is zero for a put. Working memory is $O(N)$: a fixed number of
node-length vectors and no space-time array. The only other allocations are the
small `aligned_times` and `dividend_events` vectors, whose length is the number
of declared knots and dividends.

## Supported and unsupported

Supported:

- European and American vanilla calls and puts.
- Piecewise-linear log-discount term structure with a piecewise-constant
  implied rate.
- Explicitly declared continuous carry, including zero and negative values.
- Zero or more discrete cash dividends, with the exercise decision available
  immediately before each ex-date.
- Negative rates, subject to the diagonal-dominance check.
- Nonzero valuation time.

Not supported in task 9C-A, and not silently approximated:

- Greeks of any kind. The grid carries no sensitivity output and none should be
  read off it. This is task 9C-B.
- Non-constant or state-dependent volatility.
- Proportional dividends, dividend curves, or any inference of a dividend from
  market data. Task 9B established that the archive carries no independent
  dividend ground truth.
- Batched or vectorised evaluation, and any performance claim. The engine is
  scalar.
- Barriers, Bermudan schedules, multiple assets, and stochastic rates.
- Any label policy, acceptance gate or frozen snapshot.

## Validation

C++ (`cpp/tests/test_main.cpp`) and Python
(`python/tests/test_pde_oracle_binding.py`) cover the same twelve
requirements independently:

1. European calls and puts without dividends against analytic Black--Scholes,
   across positive, zero-carry, positive-carry and negative-rate cases.
2. American options without cash dividends against the existing high-step CRR
   engine, for puts and for calls with a carry large enough to make early
   exercise optimal.
3. A non-dividend American call with non-negative rates and zero carry
   reproducing its European value.
4. American value never below intrinsic or below the corresponding European
   value.
5. Zero and negative cash dividends rejected, ex-times outside the open
   interval rejected, unsorted and duplicated schedules rejected, and an
   undeclared schedule rejected.
6. Increasing a cash dividend lowering a call and raising a put, monotonically
   across a ladder of amounts.
7. The jump mapping itself, including the $S<d$ branch, and a put whose spot
   sits below its dividend landing on the absorbing $S=0$ node and pricing to
   $KP(0,T)$.
8. Exercise immediately before a dividend: under a dividend large enough to
   leave the stock deep out of the money, the American call collapses onto a
   European call expiring at the ex-time and exceeds its own European
   counterpart by more than a full point.
9. Curve knots and dividend ex-times appearing exactly in `aligned_times`, the
   recorded dividend events indexing into it, and refusal to extrapolate on
   either side of the curve or to price against a curve that stops short of
   expiry.
10. Invalid grids, schedules, curves, rates, volatility, carry and conventions,
    and PSOR non-convergence as an explicit typed failure.
11. Bitwise determinism across repeated runs, and the contract multiplier not
    entering the arithmetic.
12. The under-resolved short-expiry fixture returning
    `solver_status = discrete_system_converged` and
    `discretization_accuracy = not_assessed` while still being visibly
    inaccurate against Black--Scholes.
13. A test-only independent European one-cash-dividend reference under flat
    rate, carry and volatility:
    $V(0)=D(0,t_d)E[BS(\max(S(t_d^-)-d,0),K,T-t_d)]$, evaluated by adaptive
    quadrature over the pre-dividend lognormal distribution. The quadrature is
    first checked against a stricter tolerance/domain, then a fine PDE grid is
    compared against it for one material-dividend call and one put.
14. Refinement ladders in space, in time, jointly, and across the truncated
    domain.

Regression tolerances are deliberately conservative and are stated as
cross-engine or solver bands, not as scientific thresholds. The refinement
study reports convergence numerically:

```bash
python -m differentiable_pricing.american.pde_refinement
```

It is exploratory, it selects nothing, and it is not part of CI.

## References

- Giles and Carter, *Convergence analysis of Crank--Nicolson and Rannacher
  time-marching*, <https://people.maths.ox.ac.uk/~gilesm/files/giles_carter.pdf>
- Ikonen and Toivanen, *Operator splitting methods for American option pricing*,
  <https://www.sciencedirect.com/science/article/pii/S0893965904804968/pdf>

## Non-claims

- No grid policy here is label-grade, and none has been justified against an
  intended estimator.
- The CRR comparison is a cross-engine gap. Both engines carry their own
  discretization error and neither is truth for the other.
- The American dividend ladder is measured against the study's own finest rung.
  That is internal self-consistency, not an independent check: no closed form
  and no second engine in this repository prices an American discrete cash
  dividend. The European one-cash-dividend test uses a separate quadrature
  reference only for European options under flat parameters.
- The upper boundary condition is asymptotic. Domain truncation error is
  measured, not eliminated.
- Nothing here reads, calibrates to, or implies any real market quote.

## Task 9C-B label-policy pilot

Task 9C-B leaves the scalar C++ solver unchanged and derives price, delta,
gamma and vega labels by deterministic centered price bumps in
`differentiable_pricing.american.pde_label_policy`. Its frozen design is
`configs/pde_label_policy_pilot_v1.toml`; generated evidence remains ignored
under `artifacts/`.

Every bumped solve for a case and grid uses the same domain target, actual
nodes, time grid, curve, carry and cash-dividend schedule. The runner checks
the actual grid signature and aborts if it changes. Spot and volatility bump
ladders are both reported. Vega is reported per unit absolute volatility and
per volatility point (the former divided by 100); theta and rho remain out of
scope.

Candidate grids are 800x400, 1600x800, and the signed second-order Richardson
pair `(4 * V_1600 - V_800) / 3`. The main reference is 3200x1600, selectively
extrapolated with 1600x800 only when the observed factor-two order lies inside
the predeclared support interval. The 6400x3200 rung is restricted to two
declared anchors. Unsupported observed order is preserved as a result and
never repaired by assuming second order.

Only regular cases decide selection and every one must pass every frozen price,
Greek, bump-stability, bound and shape check. Stress cases remain in all JSON
and CSV output; an exercise-boundary or payoff-kink row may be retained for
price evaluation while being flagged unsuitable for Greek supervision. If no
candidate clears the regular design, the only permitted recommendation is
`no_policy_selected`.
