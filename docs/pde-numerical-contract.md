# Finite-difference PDE numerical contract (task 9C-A)

## Purpose and scope

`dp::finite_difference_price` is a deterministic one-dimensional
finite-difference oracle for European and American vanilla calls and puts with
**explicit discrete cash dividends**. It is the first engine in this repository
that prices a contract the CRR and LSM engines cannot: those two carry a
continuous dividend yield, and a cash dividend is not a yield.

**Task 9C-A supports deterministic pricing only.** Nothing in this document
outside the task 9C-B section is an acceptance gate, and the first grid policy
is explicitly **not** claimed to be label-grade: no grid, step count, domain
maximum or PSOR setting here has been justified as fit for generating training
labels. Task 9C-B tested that question separately and answered it negatively
(below); the engine itself is unchanged by it.

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
$u_i$ negative — that is where convection dominates diffusion,
$|r-c|>\sigma^2 i$, which can only happen at the smallest few $i$ — that row
switches to one-sided upwinding in the drift direction. This keeps every off-diagonal
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

- Greeks of any kind *from the scalar entry point*. `PdeResult` carries no
  sensitivity output and none should be read off it. Task 9C-B derived Greeks
  *outside* the engine by bumping and repricing. Reading delta and gamma off the
  valuation-time slice is task 9C-C1 and is implemented on a separate entry
  point, `finite_difference_valuation_surface`, described below; the scalar API
  is unchanged.
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

## Task 9C-C1 valuation-time surface

Backward induction already computes the whole valuation-time vector $V(S_i)$ and
the scalar API then throws all of it away to keep one interpolated number. Task
9C-C1 exposes that vector, and nothing else about the solve's history.

`dp::finite_difference_valuation_surface(contract, grid, settings)` returns a
`PdeValuationSurface`; `dp::evaluate_valuation_surface(surface, spots)` queries
an already-solved one; `dp::finite_difference_valuation_surface_at(...)` does
both so the common case cannot accidentally solve per spot. In Python the
combined form is `pde_valuation_surface(...)`.

### The solved domain does not depend on any requested spot

`PdeSurfaceContract` is the economic state with the spot removed, and
`PdeContract::surface_state()` is how the scalar path reaches it. The spot grid
is built from the strike and the grid settings alone. The scalar entry point
keeps its own domain checks and then interpolates the same slice at its own
spot, so **the scalar API and its results are unchanged**: a surface query at
the scalar spot reproduces the scalar price *bitwise*, not to a tolerance,
because it is the same `monotone_evaluate` call on the same solved vector. Both
suites assert equality with `==`.

### What one solve returns

Per node: `index`, `spot`, `value`, `obstacle`, `obstacle_slack`,
`lcp_linear_residual`, `exercise_state`, `delta`, `gamma`, `greek_eligible` and
`greek_eligibility_reason`. Alongside them the surface carries the whole
`PdeSolveDiagnostics` block — solver status, discretization-accuracy status,
actual grid, time and Rannacher counts, PSOR solve and iteration counts,
residuals, aligned times and dividend events — plus `valuation_time`,
`exercise_classification_scale`, `boundary_exclusion_nodes`,
`regime_stencil_radius` and `backward_inductions`. `backward_inductions` is
always $1$: it is reported so a caller can show that $32$ queries cost one
solve and not $32$ solves. `PdeValuationSurface::validate()` rejects a node
vector whose length disagrees with the reported grid or whose index and spot
order is not ascending.

The Python result additionally always carries a **mandatory** `surface_input`
section: the normalized pricing and numerical inputs the solver accepted, taken
from the constructed contract, grid and surface settings rather than from the
caller's arguments. It lets a consumer reconstruct the identity of the call that
produced a result instead of trusting its own record of what it asked for. It
binds the result to its declared inputs and is **not** evidence that the
algorithm used them correctly. The field list, the reporting-only fields excluded
from that identity, and how task 9C-C2a enforces it are in the 9C-C2a section
below. The scalar API is unchanged and carries no echo.

### Delta and gamma

Both are read off the solved slice. **No additional PDE solve, and no bump, is
involved.** They are written in the actual node coordinates, with
$h^-_i=S_i-S_{i-1}$ and $h^+_i=S_{i+1}-S_i$:

$$
\Delta_i=\frac{(h^-_i)^2V_{i+1}+\left((h^+_i)^2-(h^-_i)^2\right)V_i-(h^+_i)^2V_{i-1}}{h^-_ih^+_i(h^-_i+h^+_i)},
$$

$$
\Gamma_i=\frac{2\left(h^-_iV_{i+1}-(h^+_i+h^-_i)V_i+h^+_iV_{i-1}\right)}{h^-_ih^+_i(h^-_i+h^+_i)}.
$$

On the uniform grid built today these reduce to the centered formulas
$(V_{i+1}-V_{i-1})/(2h)$ and $(V_{i+1}-2V_i+V_{i-1})/h^2$. They are deliberately
not written that way: a later nonuniform grid must not be able to inherit a
uniform-grid formula silently.

`delta` and `gamma` are `std::optional<double>`, engaged exactly when a centered
stencil exists and every value entering it is finite. **The two domain-boundary
nodes report no number at all.** A one-sided difference there would look
plausible and is not the same estimator; offering it would put a different
quantity in a training column under the same name.

### Exercise classification

The classification is read off the final implicit system, never off a post-hoc
decimal tolerance. With obstacle $\psi$, solved slice $x$, obstacle slack
$s=x-\psi$ and linear residual $\rho=Ax-b$ of that final system, the solver's
own convergence contract bounds the natural LCP residual by

$$
\varepsilon=\texttt{psor-tolerance}\times\max(1,\lVert b\rVert_\infty),
$$

reported as `exercise_classification_scale`. A node is classified only when one
of the two complementary slacks is certified nonzero at that scale:

- `continuation` when $s_i>\varepsilon$;
- `exercise` when $s_i\le\varepsilon$ and $\rho_i>\varepsilon$;
- `numerically_indifferent` when both slacks sit inside $\varepsilon$, so
  strict complementarity cannot be certified;
- `no_obstacle` for every node of a European contract, where no complementarity
  problem is posed at all. It is not a quiet `continuation`.

`numerically_indifferent` is a real population, not a rare edge. In the
reference American put fixture (800 intervals, $S_{\max}=400$, $K=100$) it is
one contiguous block of 135 nodes at $S\in[333,400]$, that is $S\ge 3.33K$,
where the value ($\le 9.7\times10^{-10}$), the obstacle and the linear residual
all sit at or below $\varepsilon\approx 10^{-9}$. It is the truncation tail, 503
nodes away from the free boundary at $S=81.5$, and its nodes are
information-free: their computed $\Delta$ is at most $9.6\times10^{-11}$ against
an analytic European reference of $-9.8\times10^{-11}$ at $S=333$.

That is the observed geometry in one fixture and it is **not** what the rule is
based on. The rule is based on the meaning of the classification: an uncertified
regime makes the derivative's interpretation uncertified. Nothing guarantees the
band stays in the tail — an indifferent band can also open at the free boundary,
where slack and residual vanish together — and there a regime-uniform interior
would otherwise be declared eligible. The exclusion is therefore applied to the
node's own state, before the stencil test. It makes the rule strictly stronger
and can only remove rows, never add them.

Counts on that fixture, by exercise state and eligibility: continuation
499 eligible / 4 refused, exercise 157 / 6, numerically indifferent 0 / 135
(656 of 801 nodes eligible). The refused continuation and exercise nodes are the
free-boundary band and the domain-edge buffer.

### Structural Greek eligibility

The rule was fixed in code and tests before any surface numbers were inspected.
A node is ineligible for the first applicable reason in this declared
precedence order, and the reason is returned as a stable machine-readable name:

1. `centered_stencil_unavailable` — index $0$ or the last index.
2. `inside_domain_boundary_buffer` — index below `boundary_exclusion_nodes` or
   above `spot_intervals - boundary_exclusion_nodes`. The buffer is a **node
   count**, declared by the caller and reported back, and it must be at least
   the regime stencil radius.
3. `non_finite_stencil_value` — a stencil value or a resulting derivative is not
   finite. Defensive: the solver already rejects a non-finite solution.
4. `unresolved_exercise_state` — the node's own classification is
   `numerically_indifferent`. The difference quotient there is a well-defined
   number, but the solver cannot certify which quantity it estimates: the
   derivative of an obstacle-clamped payoff, or of a free PDE solution. The
   refusal is structural, not a count-driven exclusion. It must hold wherever an
   indifferent band appears, including a band wide enough to have a
   regime-uniform interior *at the free boundary*, which is exactly where the
   9C-B pilot found the reference Greek least stable. The stencil value is still
   returned; only its admissibility as a label is denied.
5. `regime_stencil_not_uniform` — some node of the **five-node** stencil
   $i-2,\dots,i+2$ carries a different exercise classification. Five nodes is
   the smallest window that contains the three-node derivative stencil of both
   immediate neighbours, so a node cannot be eligible while a neighbour whose
   value entered its own stencil sits in a different regime.
   `pde_regime_stencil_radius = 2` is a fixed header constant, not a tuning
   knob.

A pure exercise-region node stays **eligible**: its whole stencil is inside one
regime, and $V=\Phi$ there makes its price and its intrinsic derivatives
mathematically valid. In the reference put fixture those nodes return
$\Delta=-1$ and $\Gamma=0$ to within $10^{-12}$. They are correct and nearly
information-free, which is a sampling-density question, not an eligibility
question: `exercise_state` is returned separately so task 9C-C2 can limit their
density. **No post-hoc sampling quota is implemented here.**

The exclusion around the free boundary is local. If the last exercise node has
index $t-1$ and the first non-exercise node index $t$, exactly the band
$[t-2,t+1]$ is refused; $t-3$ and $t+2$ remain eligible.

### Multi-spot evaluation

One solve serves every requested spot. Requested order is preserved; duplicates
are neither removed nor reordered and each carries `first_occurrence_index`,
the position of the first bitwise-equal spot, so a consumer can see the
repetition instead of inferring it. A spot outside the solved domain — not
finite, not positive, or at or beyond the actual `spot_maximum` — is rejected
with `std::invalid_argument` naming the offending query index. **Nothing is
clamped and nothing is extrapolated.** The check runs before the solve in the
combined entry point, so a bad request never costs one.

A query's price is the same monotone cubic Hermite interpolation the scalar API
uses. Its `left_node_index` and `right_node_index` come from the same cell
lookup that produced that price, so a query can never report bracketing nodes
other than its own.

**A query's delta and gamma are interpolated nodewise discrete Greeks, not
derivatives of that price interpolant.** They are the linear blend across the
containing cell of the two bracketing nodes' centered difference quotients,
returned **only** when both those nodes are Greek-eligible; two eligible
neighbours necessarily share a regime, so an engaged query Greek never straddles
the free boundary.

The distinction is measurable, not cosmetic. Differentiating the monotone
Hermite price interpolant gives a different number: at mid-cell positions on the
reference fixtures the two estimators disagree by up to $6.0\times10^{-5}$
(European call) and $8.1\times10^{-5}$ (American put) in delta, about
$3\times10^{-4}$ relative — larger than the nodewise delta's own agreement with
Black--Scholes on the same grid. An off-grid $(\text{price},\Delta)$ pair from
this API is therefore **not** internally consistent to better than that scale,
and neither estimator is corrected towards the other.

A query placed exactly on a grid node is the consistent case and is exact: the
blend weight is zero, so its value, delta and gamma are bitwise the node's own.
Both suites assert that with `==`. Task 9C-C2 should prefer node-aligned
harvesting where price and Greek must come from one interpolant. Query eligibility carries
the same reason vocabulary as node eligibility, taken from whichever bracketing
node is ineligible. `exercise_state` on a query is engaged only when both
bracketing nodes agree, so a query sitting across the boundary reports no regime
rather than a guessed one.

### Determinism, complexity and memory

The surface adds no randomness, no reassociated reduction and no parallelism.
Repeated solves are bitwise identical in every node value, Greek, classification
and in `exercise_classification_scale`.

**Only the valuation-time level is retained.** The marching loop holds the same
fixed number of node-length vectors it always did, plus one node-length residual
vector read off the final system; no space-time array is formed and no time
history is exposed. Working memory is therefore $O(N_S)$, unchanged by $M$, and
the returned surface is $O(N_S)$ as well. Both suites assert the structural
consequence: quadrupling the time steps leaves the returned node count and the
aligned-time vector unchanged. That is a structural regression test on the
design, not a memory measurement.

### Not implemented by task 9C-C1

Volatility bumps and vega surfaces; any label dataset, Parquet writer or
partitioning code; worker pools, threading, multiprocessing or cluster
execution; progress or checkpoint infrastructure; Richardson label generation; a
replacement for PSOR; label policy v2; neural training; implied-volatility
inversion or surface calibration. **Task 9C-B remains `no_policy_selected`** and
nothing in this section revisits it: no threshold, config or frozen result of
that pilot is touched, and no grid here is label-grade.

No throughput is claimed. `scripts/demo_pde_valuation_surface.py` is a
correctness demonstration on two small cases and its wall-clock lines are not a
production-feasibility measurement and must not be extrapolated to a label
budget.

### Task 9C-C2a: harvesting correlated spot rows under grouped partitioning

The rules recorded in the next section were written before any dataset existed.
Task 9C-C2a implements the structural half of them:
`differentiable_pricing.american.pde_surface_harvest`, driven by a versioned
TOML design such as `configs/pde_surface_harvest_demo_v1.toml`. It is
**exploratory infrastructure**. It generates no production dataset, computes no
vega, runs no worker pool, replaces no LCP solver, trains nothing, and selects
no label policy.

```bash
python scripts/demo_pde_surface_harvest.py
```

#### Three counts, never one

A flattened dataset conflates three different numbers, so all three are reported
globally and per partition:

- **raw harvested row count** — one row per harvested node;
- **independent design-group count** — distinct base economic states that
  produced rows. It is a *design* count. It is deliberately **not** called a
  statistical effective sample size, and no estimator has been fitted to it;
- **surface-lifecycle counts** — planned, attempted, solver-returned, solver-failed,
  pipeline-successful, post-solve-pipeline-failed, retained and discarded surfaces are
  separate integers. Attempted increments immediately before the solver call;
  solver-returned increments immediately after it returns, before validation or harvesting.
  A returned surface that later fails is never described as a solver failure.

Every report carries the statement that rows sharing a `surface_id` are
correlated outputs of one backward induction, and rows sharing a
`partition_group_id` are correlated outputs of shared numerical work.

#### Two different yields, always reported together

Rows per *group* and rows per *solve* answer different questions and differ by
however many surfaces a group emits — a factor of four in the shipped design,
which declares two option types and two exercise styles. Quoting the first as a
computational multiplier would overstate the numerical-work reuse by exactly
that factor, so all three ratios below are always published side by side, each
carrying its own integer numerator and denominator:

- `raw_rows_per_group` = rows / independent design groups. **Dataset expansion
  per design point.** It is not work saved and must never be quoted as a
  computational speedup.
- `raw_rows_per_attempted_surface` = rows / every solver call attempted,
  including raising calls, invalid returned surfaces and pipeline-successful surfaces later
  discarded with a failed group. This is the honest **rows-per-attempt
  multiplier** and the conservative denominator.
- `raw_rows_per_retained_surface` = rows / pipeline-successful surfaces whose rows were kept. It
  equals the previous ratio when nothing was discarded and exceeds it otherwise;
  it is published beside the attempted-solve figure so the flattering
  denominator can never appear alone.

Each ratio is an object with `numerator`, `denominator` and `value`. Both terms
are integer counts that appear elsewhere in the same scope, `value` is exactly
their float64 quotient, and a zero denominator gives `0.0` rather than a
non-finite number. `reconcile_report` verifies every one of these against the
scope's own counts, and separately that
`planned_surface_count == attempted_surface_count`,
`attempted_surface_count == solver_returned_surface_count + solver_failed_surface_count`,
`solver_returned_surface_count == pipeline_successful_surface_count +
post_solve_pipeline_failed_surface_count`, and `pipeline_successful_surface_count ==
retained_surface_count + discarded_surface_count`. Failure records equal solver failures plus
post-solve pipeline failures. Backward inductions and available returned diagnostics are counted
when the solver returns, even if validation or harvesting later fails; nothing is invented for a
raising call.
The definitions travel inside the report under `interpretation.yield_definitions`.
**Neither ratio is a statistical effective sample size.**

#### Versioned canonical identities

Each identity is `sha256` over a canonical JSON payload: sorted keys, compact
separators, and every float rendered as a `.17g` tagged string rather than left
to a JSON float writer. Nothing depends on Python's `hash()`, on dictionary or
insertion order, on locale, on platform float `repr`, or on any path. The
identity scheme version is inside the hashed payload, so changing the scheme
changes every digest.

**Negative zero is normalized to positive zero before serialization.** IEEE-754
makes $-0.0 = 0.0$ true while `.17g` prints them differently, so without the
normalization two states that compare equal everywhere — and price identically —
would receive different identities and could land in different partitions. It is
not a hypothetical: a flat curve's log discount is $-r(T-t_0)$, which is exactly
`-0.0` at a zero rate, and a declared zero carry can arrive signed either way.
The normalization applies at every depth, so nested curve and dividend inputs are
covered, and it touches nothing else: `value == 0.0` is true for the two signed
zeros and for no other magnitude, subnormals included. Non-finite values remain
rejected outright.

- **`partition_group_id`** identifies the base non-spot economic state: strike,
  valuation and expiry times, **base volatility**, continuous carry, the whole
  discount curve, the cash-dividend schedule and settlement. It excludes the
  queried spot, the harvested node index, the option type, the exercise style,
  the volatility-bump role, the numerical grid and solver settings, and the
  contract multiplier. It also excludes the scenario's *name*: a name is
  documentation, and admitting it would let two identical economic states claim
  to be different design points and defeat duplicate detection.

  **`contract_multiplier` is excluded because this document already states that
  it never enters pricing arithmetic**: prices are per share, and a regression
  test asserts that changing the multiplier leaves the price and every reported
  diagnostic bitwise unchanged. Two candidates differing only in it are the same
  pricing state, so they are rejected as a duplicate economic group before any
  solve. Letting a reporting convention split them would manufacture a second
  design point out of nothing and could place two identical pricing states in
  different partitions. The multiplier stays in row and report metadata, where it
  is a reporting fact rather than an identity. `settlement` is deliberately *not*
  excluded even though the current engine does not use it either: it names a
  genuinely different contract term whose economic content a later engine may
  price, whereas a multiplier can only ever scale a reported number.

  Option type and exercise style are excluded as the **conservative** choice. A
  call and a put on one base state are linked by put--call parity, and a
  European contract is the natural dominance control of the American one on the
  same state — the same pairing task 9C-B's shape checks used. Those are solver
  siblings of one scenario, so they are kept together rather than allowed to
  straddle partitions. The grid is excluded because one economic state solved on
  two grids is still one state; excluding it can only merge groups, never split
  them, so it cannot create a leak.
- **`solver_input_id`**, also used as **`surface_id`**, identifies the actual PDE
  call. Its canonical descriptor contains strike, option type, exercise style,
  valuation and expiry, the discount curve, dividends, continuous carry,
  settlement, volatility **actually passed**, and every numerical grid and
  solver setting. It excludes `partition_group_id`, scenario name, surface role,
  the reporting-only multiplier and descriptive metadata. A role is provenance,
  not pricing identity, so identical calls cannot acquire different IDs.
- **`row_id`** identifies one harvested node: the `surface_id` and the node
  index. The grid is already pinned inside `surface_id`, so the index determines
  the spot exactly and no float enters the digest.

`SURFACE_ROLES` is `base`, `sigma_down`, `sigma_up`. Before partition assignment,
the planner constructs every actual solver descriptor. If two groups, two roles
in one group, or repeated candidates claim the same `solver_input_id`, C2a
rejects the design before assignment or solving. It does not merge components or
reuse solves. Task 9C-C2b may later introduce explicit solve reuse. **No vega is
computed.**

#### Partitioning, before any solve

`plan_harvest(config)` takes **no solver argument and calls none**. Assignment
consumes only candidate group identities, the algorithm version and the seed, so
an outcome-dependent assignment is structurally impossible rather than a matter
of statement ordering.

The algorithm is `grouped_quota_by_sorted_digest/1`. Candidate groups and every
actual solver input are canonicalized and duplicate-checked; a repeated economic
state or actual PDE call is an error raised **before partition assignment and the
first solve**. Survivors are sorted by their own seeded
assignment digest and sliced contiguously into `train`, `validation` and
`interpolation_test` — the repository's established partition names, asserted
equal to `data.config.SPLIT_NAMES` — using largest-remainder counts from the
config-versioned weights, with ties broken by declared partition order and a
deterministic repair pass enforcing `minimum_groups_per_partition`.

Consequences, each covered by a test: every row of a group lands in exactly one
partition; a future base/sigma-down/sigma-up triple cannot cross partitions;
reordering candidates inside an already parsed configuration changes no
assignment or output; and the chunk size changes nothing either, because chunking
only groups the loop.
The rule is deliberately **not incremental**: adding a candidate re-derives every
assignment, which is why the design is versioned with the configuration.

#### Exact-node harvesting

Harvesting uses **exact grid nodes only**. `harvest.node_source` must be
`exact_grid_nodes` and any other value is rejected; `query_spots` is always
empty, so the task 9C-C1 off-grid query API is untouched and unused. Task
9C-C1's own measurement is the reason: an off-grid price and an off-grid delta
come from two different estimators and are not internally consistent to better
than about $3\times10^{-4}$ relative.

Rejection precedence is fixed and first-applicable, so every node is counted
exactly once and the counters reconcile against the node total:

1. `truncation_endpoint` — index $0$ or the last index;
2. `boundary_buffer` — inside the surface contract's declared
   `boundary_exclusion_nodes`, which must be at least the regime stencil radius;
3. `outside_moneyness_window` — outside the predeclared interior window. That
   window is a *sampling* choice and is separate from the structural buffer
   above;
4. `non_finite_price`;
5. `quota` — an admissible node its regime's quota did not select.

The structural admissibility fields are **copied, never recomputed or relaxed**.
`exercise_state`, `greek_eligible` and `greek_eligibility_reason` come through as
the engine reported them, the numerical `delta` and `gamma` are exposed
*separately* from `delta_label_eligible` and `gamma_label_eligible`, and three
invariants are asserted live rather than assumed: a `numerically_indifferent`
node reported as Greek-eligible, an eligible node with reason other than
`eligible` or with a missing or non-finite derivative, and an ineligible node
with reason `eligible` are all errors. A `numerically_indifferent` row is
therefore **retained for price and refused for Greeks**, with its stencil number
still visible.

#### Regime density and shortfalls

Pure exercise-region rows are correct but information-light, and numerically
indifferent rows are diagnostic only, so `[harvest.quota]` declares a per-surface
cap for each regime. Selection runs **only after** the group's partition is
already fixed and follows one predeclared, outcome-independent rule,
`evenly_spaced_by_ascending_node_index/1`: it looks at positions, never at
prices, Greeks or errors. With a quota no larger than the population the spacing
is at least one position, so the selected indices strictly increase and no
duplicate can be produced; a duplicate would still be counted and removed, and a
repeated `row_id` anywhere is a hard failure.

A regime the contract style cannot produce — `exercise` for a European contract,
`no_obstacle` for an American one — is *inapplicable*, not a shortfall. Genuine
shortfalls report requested, achieved, the admissible population and the deficit,
per surface and summed per partition and globally.

The shipped demonstration values are exploratory. **They are not a production
sampling policy.**

#### Output, failure and determinism

`report.json` and `rows.csv` are published atomically, then `manifest.json`
**last**, carrying the SHA-256 and byte length of both. A run interrupted
mid-publication therefore leaves a directory `verify_publication` rejects rather
than one that reads as a completed dataset. Every published list is emitted in
canonical order, and no deterministic file contains a hostname, an absolute path,
a timestamp or a runtime measurement; the demonstration prints elapsed time to
the terminal only. Provenance carries both `raw_config_sha256`, the exact TOML
bytes, and `semantic_config_sha256`, canonical parsed semantics with scenario and
set-like declaration ordering normalized.

**Byte identity is claimed only under stated preconditions**, published in the
report as `determinism.byte_identity_preconditions`:

- identical configuration TOML bytes;
- identical source-name provenance, that is the same `config_name`;
- identical runner code and compiled engine;
- differences confined to harmless candidate ordering or chunk size.

The source *name* is a precondition, not a detail. `config_name` is recorded
provenance, so **identical bytes loaded under a different filename** keep the
semantic digest, the raw digest, every identity, every partition assignment and
`rows.csv` byte for byte — while `report.json` differs in that one field, and the
manifest differs because it pins the report's hash. That is a provenance
difference, not non-determinism, and it has its own check:
`verify_semantic_identity` compares the two publications with `config_name` set
aside and still requires `rows.csv` to match exactly.
`verify_byte_identity` remains the stricter check and is valid only when every
precondition above holds. Semantically equivalent *reordered* TOML is the other
case: it preserves the semantic digest, identities, assignments and `rows.csv`,
while its raw digest intentionally differs.

#### The mandatory `surface_input` echo

The task 9C-C1 surface result carries a **mandatory** `surface_input` section
built from the constructed `dp::PdeSurfaceContract`, `dp::PdeGrid` and
`dp::PdeSurfaceSettings` — the normalized inputs the solver actually accepted,
not the caller's own strings. It is additive; the scalar API is untouched and
carries no echo.

Its nineteen identity fields are option type, exercise style, strike, valuation
and expiry time, volatility, continuous carry, curve times, curve log discounts,
dividends, settlement, spot intervals, time steps, spot maximum, Rannacher steps,
PSOR tolerance, relaxation and iteration ceiling, and the boundary exclusion
buffer. Two further fields are reporting metadata that must never enter the
identity: `contract_multiplier`, which does not enter pricing arithmetic, and
`dividends_declared`, which distinguishes a declared empty schedule from an
undeclared one and is checked rather than hashed.

The echo reports **requested** grid targets. The adjusted grid the solver used
stays in the diagnostics, so the two remain distinguishable and the actual-solve
identity is defined by the inputs that select the solve.

**Task 9C-C2a treats the echo as mandatory and there is no optional-echo success
path.** Before any row is harvested, the harvester rebuilds the canonical solver
descriptor from the returned echo and requires its digest to equal the planned
`solver_input_id`; it compares field by field first so a mismatch names the
offending input, and it rejects a missing section, a missing field or an unknown
field. A missing or mismatched echo counts as one attempted, solver-returned,
post-solve-pipeline-failed surface,
produces a failure record with full group and surface identity, produces no rows,
and prevents its group from being retained.

The echo **binds the API result to the declared solver inputs. It is not
independent evidence that the numerical algorithm used them correctly.**
Numerical correctness remains established by the task 9C-C1 validation suite.

A raising solver or returned surface that fails plan/result binding is isolated
and recorded with its full group and surface identity, stage, error type and
message. The result binder verifies the mandatory input echo, the adjusted spot
and time grid diagnostics, solver tolerance, stencil buffer, vector length and
exact uniform spot node against the planned actual-solver descriptor. Its group
is marked `failed` and contributes
**no** rows: a partially solved group is not a coherent design point, and
admitting one would put a silently thinner state into a partition. The solves
that did complete are still reported as solves, with `retained = false`, so the
solve count never quietly shrinks.

#### One derivation, used to write and to reconcile

`expected_surface_metadata` and `expected_row_metadata` derive the immutable
metadata of a surface record and of a row from the planned solver descriptor,
the planned partition and group, and — for a row — the exact node index. The
same two functions build the records when writing and rebuild them when
reconciling, so a published field is never compared against itself.

Every surface record's identity-linked metadata is compared as a whole:
`surface_id`, `solver_input_id`, group, partition, scenario name, role, option
type, exercise style, strike, actual volatility, settings digest and the full
solver descriptor, plus its solved status and its retention against its group's
status. Every row's immutable columns are rebuilt and compared likewise:
partition, group, surface and row IDs, role, option and exercise type, strike and
actual volatility, valuation and expiry time, carry, rate, settlement, dividend
count, contract multiplier as reporting metadata, settings digest, grid, and the
spot derived from the exact planned grid rather than copied from the row.

The counts are **recomputed from the actual rows**, never trusted: rows per
partition, group and surface; selected rows per surface and per regime; counts by
exercise state; counts by price, delta and gamma eligibility; counts by
ineligibility reason; and duplicate and cross-partition intersections. Each is
compared against the report. Numerical row invariants are checked too: a finite
price whenever price-eligible, finite eligible Greeks, eligibility agreeing with
its reason, and `numerically_indifferent` never Greek-eligible.

One limit is stated rather than overclaimed. The **rejected-node tallies cannot
be independently reconstructed from published data**: the nodes they count were
discarded and are not republished, so only the recorded surface audit supplies
those facts. They are reconciled as integer accounting — selected plus rejected
equals the node count — while the *selected* side is reconciled directly against
the rows that actually belong to that surface, per regime, together with their
node indices and per-regime quota ceilings.

`reconcile_report` is a live invariant, not a test helper, and
`verify_publication` runs the full semantic comparison after hash verification.
Semantic reconciliation rejects **single-sided mutations and inconsistent
report/plan/row combinations even when ordinary file hashes have been
regenerated** to match the edited files: a changed row volatility, strike, option
type, node index, spot, settings digest, eligibility or exercise state; a changed
surface-record volatility, settings or solver descriptor; and a changed
classification or eligibility count in the report are each rejected, because each
is rebuilt from the published plan or recomputed from the actual rows rather than
read back from the field under test.

#### What that reconciliation does not cover on its own

That guarantee is about *consistency*, not authenticity, and the gap was
recorded here rather than left for a reader to discover. Five plan fields carry
no digest of their own, so a **coordinated rewrite that changes the published
plan and every corresponding row in the same way passes `verify_publication`
undetected**:

- `scenario_metadata.rate`
- `scenario_metadata.contract_multiplier`
- `scenario_name`
- `surface_role`
- `settings_digest`

Three of them are in fact derivable from data that *is* anchored —
`settings_digest` from the solver descriptor's own grid and solver settings,
`rate` from `curve_log_discounts` and `expiry_time`, and a `base` role from the
actual volatility equalling the group's base volatility — and are simply not
recomputed today. `scenario_name` and `contract_multiplier` are free reporting
text and would need a plan-level digest instead.

The scope of the gap, stated exactly:

- It does **not** affect `solver_input_id`, `surface_id` or `row_id`, each of
  which is a digest of its own payload; nor partition assignment; nor any pricing
  label; nor the current exploratory demonstration, which publishes nothing that
  is consumed.
- Published task 9C-C2a harvests must therefore **not** be treated as
  authoritative downstream training inputs on the strength of
  `verify_publication` alone.
- Anchoring or recomputing these five fields was a **required first step of task
  9C-C2b**, before any published harvest is read as an input rather than as a
  demonstration. Task 9C-C2b1 does it, in a **separate, explicitly named**
  entry point; the section below states exactly which guarantee is which, and
  `verify_publication` deliberately keeps the weaker one.

Finally, `_expected_spot_grid_payload` and `_expected_time_grid_payload`
reconstruct the solver's node placement and time alignment in Python. That
duplication is intentional: it is the cross-check that binds a returned surface
to its planned grid, and a divergence fails loudly as a validation error rather
than silently. It is **not** an independent implementation, and it must be
updated in the same change as any future alteration of the C++ grid rule.

#### Not implemented by task 9C-C2a

Vega surfaces and the sigma-bump solves themselves beyond the role identity;
parallel or batched execution; Parquet output; production dataset generation;
label policy v2; neural training; checkpointing or resumption; any acceptance
gate. **Task 9C-B remains `no_policy_selected`** and nothing here reads, reruns,
edits or reinterprets its frozen evidence. Task 9C-C2b1, below, adds the vega
triple and authoritative verification and nothing else from that list.

### Task 9C-C2b1: authoritative verification and three-surface vega

Task 9C-C2b1 does exactly two things: it closes the five recorded plan-metadata
gaps behind a separately named verification entry point, and it computes vega
from three surfaces per contract leg. It adds no worker pool, no resumability,
no Parquet, no production dataset, no label policy v2, no training, no new LCP
solver and no acceptance gate, and it does not rerun the task 9C-B pilot.

The row, report, manifest and configuration schemas move to `/3`. The **identity
scheme stays at `pde-surface-harvest-identity/2`**: no identity payload gained,
lost or reordered a field, so every `partition_group_id`, `solver_input_id` and
`row_id` a task 9C-C2a run produced still identifies the same thing.

Vega label identity is deliberately separate. `vega_convention_id` is SHA-256 over a canonical
versioned payload containing the centered formula, canonical absolute bump $eta$, per-unit
absolute-volatility unit, divide-by-100 point conversion, the canonical meanings of
`sigma_down`, `base` and `sigma_up`, and exact-node-index plus bitwise-spot matching.
`vega_label_record_id` is derived from the economic `row_id` and that convention ID. Therefore:

- equal `row_id` means the same base economic and numerical pricing node;
- equal `vega_label_record_id` means the same node under the same vega convention;
- equal `row_id` with unequal `vega_convention_id` is not an identical label record.

A base-only design publishes the convention and label-record identities as absent. A
three-surface row publishes its convention ID; only a row with an available vega label publishes
a label-record ID. Stored digests distinguish conventions but are not authenticity evidence.
Authoritative verification rederives the convention from the externally supplied expected
configuration and verifies every row identity against it.

#### Two guarantees, named separately and kept separate

`verify_publication` is **self-contained consistency verification** and is
unchanged in kind. It proves a publication agrees with itself: file hashes match
the manifest, each identity is a digest of its own payload, each immutable row
and surface field is rebuilt from the published plan, and every count is
recomputed from the actual rows. **A digest stored inside a publication cannot
make that publication authentic**, and this document does not claim otherwise.

`verify_publication_authoritatively(directory, expected_config=...)` is
**authoritative verification against an externally supplied expected
configuration**, either an already parsed config or a path to the versioned
TOML. It:

1. verifies the publication's `raw_config_sha256` and `semantic_config_sha256`
   against that configuration;
2. **replans the harvest deterministically** with `plan_harvest`, which takes no
   solver argument and calls none, so nothing is priced;
3. compares the complete published plan against the replanned one, field by
   field first so a mismatch names the offending field, then as a whole
   canonical payload;
4. thereby anchors `scenario_metadata.rate`,
   `scenario_metadata.contract_multiplier`, `scenario_name`, `surface_role` and
   `settings_digest` against the supplied configuration rather than inferring
   any of them from the publication;
5. **recomputes `settings_digest`** from the published solver descriptor's own
   grid and solver settings rather than trusting the published value;
6. compares every configuration-derived report section — `vega_identity`, `study`,
   `partitioning`, `harvest_rules` — and the `numerical_settings` table, all
   built by the same `_config_report_sections` derivation that wrote them;
7. finally runs the whole consistency pass with every row and surface record
   rebuilt from the **externally derived** plan. It is a strict superset of
   `verify_publication`'s semantic stage.

The report carries both statements verbatim under `verification`, so a reader of
an artifact cannot mistake one for the other.

`verify_training_input_publication` is the gate a downstream training consumer
must use. It always runs the authoritative path — consistency-only verification
is explicitly insufficient for a training input — and then refuses any study
status outside `APPROVED_TRAINING_INPUT_STATUSES`, which is **empty**. Nothing
this module produces is an approved training input today.

Mutation tests cover all five fields: a coordinated rewrite of the plan, every
affected row and every regenerated file hash is rejected by authoritative
verification, with the offending field named. Four of the five —
`rate`, `contract_multiplier`, `scenario_name`, `settings_digest` — are also
shown *passing* consistency-only verification, which is what makes the two
guarantees genuinely different rather than nominally different. `surface_role`
is the exception and is tested as such: the vega design below pins each role to
the volatility its group actually solves, and a solved volatility lives inside
`solver_input` and therefore inside the surface's own digest, so relabelling a
role is now detectable without an external configuration too.

#### Three surfaces, one vega

A configuration declares either exactly `["base"]` or exactly the complete
triple `["base", "sigma_down", "sigma_up"]`. A partial set such as
`["base", "sigma_up"]` is **rejected at parse time**: the convention below is
centered, an asymmetric pair cannot feed it, and admitting such a design would
invite a one-sided vega under the same column name later.

With the triple, each contract leg — one `(option_type, exercise_style)` of one
scenario — is solved three times, at $\sigma-\eta$, $\sigma$ and $\sigma+\eta$,
and

$$
\mathrm{vega}=\frac{V(\sigma+\eta)-V(\sigma-\eta)}{2\eta}.
$$

- $\eta$ is an **absolute** volatility bump declared as
  `volatility_bump` under `[surfaces]`. A scenario with $\sigma-\eta\le 0$ is
  rejected during configuration parsing, before planning and therefore before
  the first solve.
- Vega is published **per unit absolute volatility**.
  `vega_per_volatility_point = vega / 100` is published beside it as a
  **reporting** conversion only; nothing consumes it.
- **Price, delta and gamma come only from the base surface.** The base surface
  is the leg's only row source; `sigma_down` and `sigma_up` contribute two
  prices per node and nothing else. Their whole node vector is accounted for
  under the rejection reason `non_base_role`, so the per-surface identity
  `selected + rejected == node_count` still holds for every surface.
- **Vega comes only from the two bumped prices.** The base price does not enter
  it.
- The three surfaces are required to sit on **bitwise the same spot grid** —
  intervals, maximum, step, strike node and the whole node vector — and rows are
  matched by exact node index with the bumped node's spot required to equal the
  row's spot bitwise before its price is used. Volatility does not enter the
  grid rule, so those checks should never fire; they exist because a silently
  misaligned pair would difference two prices at two different spots and publish
  the result as a derivative.
- `solver_input_id` already contains the volatility actually passed, so the
  three solves of one leg carry three distinct identities and no alias can merge
  them.
- The roles of one leg are members of one partition group by construction, which
  is the rule declared before any vega existed. A triple therefore cannot
  straddle a partition.
- **Group atomicity is preserved.** The two bumped surfaces are solved first and
  the base surface last, so a leg's vega is assembled inside the same guarded
  step that harvests its rows. Every planned surface is still attempted —
  `planned == attempted` is a reconciliation invariant and a failure may never
  become a skipped solve — but a base surface whose bumped sibling failed raises
  at stage `surface_harvest` and its group retains nothing. A solver return is counted before
  that stage; the base is then a post-solve pipeline failure, while only surfaces that pass the
  whole pipeline enter retained/discarded accounting.

#### What the vega columns are, and what they are not

`vega_numerically_available` is a **numerical-availability flag and nothing
more**: it says this row carries a finite vega computed from its own group's
three surfaces at the same exact grid node. It is deliberately not named like
`delta_label_eligible` or `gamma_label_eligible`, and **no vega is called
supervision-eligible here**. Whether a vega is stable enough to supervise is
task 9C-C3's decision.

The inputs that decision needs travel with the row so it can be made without
re-solving: `vega_bump`, `price_sigma_down`, `price_sigma_up`, the base `price`,
and `exercise_state_sigma_down` / `exercise_state_sigma_up`. One-sided
differences, the second difference and any bump ladder follow from the first
four. The last two are there because a node can exercise at $\sigma$ and
continue at $\sigma+\eta$ — the free boundary moves with volatility — and a
centered difference across that regime change estimates neither one-sided
derivative. That is published raw, per row, and counted descriptively in
`counts_by_vega_bump_exercise_regime`; it decides nothing here.

A row whose bumped price is not finite keeps its price, delta and gamma, reports
`vega_numerically_available = false`, and publishes **no** vega quantity at all
rather than a zero or a NaN. Its `vega_bump` and the two bumped regimes stay,
because they are design and classification facts rather than derived numbers.

**Gamma supervision policy is unchanged and undecided.** Gamma remains
evaluation-only, exactly as task 9C-C2a left it.

#### Recomputed, never trusted

Every vega-related count is recomputed from the actual rows during
reconciliation: `counts_by_vega_numerical_availability` and
`counts_by_vega_bump_exercise_regime` globally and per partition, and
`vega_available_row_count` per surface. Each published row's vega is recomputed
from that row's own `price_sigma_up`, `price_sigma_down` and `vega_bump` with
exactly the operations that produced it and compared **bitwise**, and
`vega_per_volatility_point` against `vega / 100` likewise. `vega_bump` is
anchored separately and exactly against the plan: `_role_volatility` computes
the bumped volatilities as `base -/+ bump` in float64, so recomputing them from
the row's published bump reproduces the planned volatilities bitwise and no
tolerance is involved.

Editing `vega`, either bumped price, the bump, either vega identity, the point conversion, the
availability flag, a bumped regime, or any vega count is therefore rejected even
when every ordinary file hash has been regenerated.

What this does **not** prove is that the two bumped prices are the ones the
solver returned. That is the same limit the base `price` column has always had,
and it is stated rather than papered over: reconciliation binds published
numbers to published inputs and to the plan; it does not re-solve.

#### Measured against independent references

Neither figure below is an acceptance gate, neither is an accuracy claim, and
no threshold from the frozen task 9C-B pilot is reused as a criterion.

- **European call against Black--Scholes**, at 800 spot intervals and 400 time
  steps with $\eta=0.01$ across three volatilities: the worst absolute error
  against the analytic derivative is $5.839\times10^{-2}$ per unit volatility,
  while against the *analytic centered difference at the same $\eta$* it is
  $9.634\times10^{-3}$, falling to $2.407\times10^{-3}$ at 1600/800. Error against
  analytic Black--Scholes vega contains both PDE-grid error and finite-bump truncation. The
  same-$\eta$ analytic centered comparison isolates the PDE-grid component pointwise: grid
  refinement reduces that component but cannot remove the finite-bump component. The printed
  worst-case maxima may occur at different rows and must not be subtracted to estimate
  truncation error. This is exactly the kind of fact task 9C-C3 needs.
- **American put against a twice-refined centered PDE control** (a control
  surface at double the spot intervals and double the time steps, where every
  coarse node is a refined node at bitwise the same spot): the worst absolute
  difference over all harvested rows is $3.478\times10^{-2}$ and over
  Greek-eligible rows $1.992\times10^{-2}$. The worst row is one the surface
  contract already refuses as a Greek label, with reason
  `regime_stencil_not_uniform` — the free-boundary band. **Vega is not gated by
  that flag**, because it comes from prices rather than from a stencil, and that
  is precisely the stability question left open.
- A pure exercise-region American put node has vega zero to solver scale while
  **both** bumped surfaces are still in that region at the same node, and a
  visibly non-zero vega when the up-bump crosses into continuation. Both cases
  are asserted.

#### Demonstration

```bash
python scripts/demo_pde_surface_vega_harvest.py
```

Driven by `configs/pde_surface_vega_harvest_demo_v1.toml`: nine scenarios, one
contract leg, three surfaces per group, 27 solves. It reports groups, planned
  surfaces per group, the full attempted/returned/pipeline lifecycle and retained/discarded
  counts, rows per independent group beside rows per attempted surface, the two vega
comparisons above, partitions and integrity counts, both verification
guarantees, the training-input gate refusing, byte identity under reversed
candidate order and a different chunk size, and wall time on the terminal only.
It is descriptive evidence. It selects no label policy, generates no dataset,
and is not a throughput claim.

Rows per group and rows per attempted surface now differ by a factor of **three**
in that design, because a vega costs three backward inductions per row set.
Quoting rows per group as a computational multiplier would overstate the
numerical-work reuse by exactly that factor.

#### Not implemented by task 9C-C2b1

Parallel workers; resumability or checkpointing; dataset-scale generation;
label policy v2; neural training; a replacement for PSOR; Parquet output; any
acceptance gate; any vega supervision-eligibility or stability rule; any change
to gamma supervision policy. **Task 9C-B remains `no_policy_selected`** and
nothing here reads, reruns, edits or reinterprets its frozen evidence.

### Mandatory rules for the task 9C-C2 dataset contract

These are recorded here now, before any dataset exists, because the leakage they
prevent is invisible once rows are flattened. **One surface yielding many rows
does not make those rows statistically independent.**

- All spot rows harvested from the same PDE surface are correlated numerical
  outputs of one solve.
- The sigma-minus, base-sigma and sigma-plus surfaces later used for vega belong
  to the **same surface group**.
- Train/validation/test assignment happens by surface or economic group
  **before solving**, never row by row afterwards.
- No surface group may straddle partitions.
- Reports must show both raw row counts and independent surface-group counts.
- Duplicate economic states must be checked at both row level and group level.
- Learning-curve size is measured primarily in independent non-spot parameter
  groups, not only in harvested rows.
- Harvesting must use a predeclared interior window away from the truncation
  boundary. The `boundary_exclusion_nodes` buffer here is a structural
  admissibility rule, not that window.
- Exercise-region rows are correct but low-information, and their sampling
  density will be controlled by a predeclared task 9C-C2 policy. That policy is
  separate from the structural refusal of uncertified-regime rows above: those
  are excluded by the engine and are not a sampling choice.
- Row counts must be reported broken down by exercise state and eligibility
  reason, so a partition cannot silently consist of near-intrinsic rows.

A group should eventually contain every output sharing the same underlying
non-spot economic state and related numerical construction, including call/put
or volatility-bump relatives wherever the sampling contract determines that
cross-partition separation could leak information.

Task 9C-C2a implements every rule above and fixes that identifier schema; see
the preceding section. The rules are kept here in their original form because
they were declared before any dataset existed, and a later implementation must
be readable against what was required rather than against itself. What 9C-C2a
does **not** supply is the rest of task 9C-C2: no vega triple is solved, no
production dataset is generated, and learning-curve measurement in independent
groups remains future work.

### Validation

C++ (`cpp/tests/test_main.cpp`) and Python
(`python/tests/test_pde_surface_binding.py`) cover the same requirements
independently:

1. The scalar API and its status semantics unchanged, and a surface query at the
   scalar spot reproducing the scalar price bitwise for European, American,
   and cash-dividend contracts.
2. Node vector lengths, ascending order, finite values, node spots at $ih$, and
   the reported grid, buffer, stencil radius and solve count.
3. European price, delta and gamma against analytic Black--Scholes at every
   eligible node in a smooth interior window, on both a fine and a coarser grid.
4. Call and put monotonicity, convexity of the value slice, and non-negative
   gamma on eligible nodes, for both exercise styles.
5. American obstacle dominance at every discrete node.
6. Pure exercise-region put nodes: value intrinsic to within
   $\varepsilon$, $\Delta=-1$ and $\Gamma=0$ to $10^{-12}$, classified
   `exercise`, and structurally eligible.
7. Continuation-region classification with certified obstacle slack.
8. The whole classification rule replayed against $s_i$, $\rho_i$ and
   $\varepsilon$ on every node.
9. Nodes whose stencil crosses the free boundary ineligible with
   `regime_stencil_not_uniform`, and the exclusion band exactly $[t-2,t+1]$;
   every `numerically_indifferent` node ineligible, with
   `unresolved_exercise_state` on the interior of the band and its difference
   quotient still returned.
10. Query delta and gamma equal the linear blend of their bracketing nodewise
    Greeks bitwise, differ from the derivative of the query price interpolant,
    and reduce to the node's own Greeks bitwise for a query on a grid node.
11. Domain-boundary nodes ineligible with `centered_stencil_unavailable` and no
    delta or gamma at all; buffer nodes ineligible with
    `inside_domain_boundary_buffer` while their stencil still exists.
12. A discrete-dividend surface: events recorded, obstacle dominance held, and
    the scalar identity preserved. No analytic comparator is used, because the
    cash-dividend jump invalidates Black--Scholes.
13. Negative rate with positive carry against Black--Scholes; a negative-rate
    American put having **no** exercise region and collapsing onto its European
    surface; and a negative-rate carrying American call whose exercise region is
    the upper spot range.
14. Thirty-two queries from one solve: `backward_inductions = 1`, and PSOR solve
    and iteration totals identical to a one-query run of the same contract.
15. Out-of-domain, non-positive and non-finite queries rejected by index.
16. Duplicate and unsorted queries preserved in order with correct
    `first_occurrence_index` and identical values for repeats.
17. Bitwise-identical repeated solves.
18. The returned surface unchanged in size by quadrupling the time steps.
19. Rejected surface requests: undersized and oversized boundary buffers, a
    domain not containing the strike, an undeclared carry, and the absence of a
    spot argument on the surface entry point.

Greek bands against Black--Scholes are conservative cross-check bands, not
accuracy claims and not acceptance gates.

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

### Outcome

The pilot was run once and its reviewed report is frozen in
[results/american_pde_label_policy_results_v1.json](results/american_pde_label_policy_results_v1.json),
digest-pinned by `python/tests/test_pde_label_policy_results_snapshot.py`. It
recorded `selected_accuracy_policy = no_policy_selected` with
`criteria_were_not_loosened = true`. Read it as three separate findings.

1. **The accuracy caps were met at 1600x800.** Across all 22 regular cases the
   worst absolute errors against the study's internal reference were 3.5822e-4
   in price (cap 5e-4), 1.7330e-5 in delta (cap 1e-3), 4.5225e-6 in gamma (cap
   2e-4) and 2.9727e-3 in vega per unit volatility (cap 5e-2).
2. **Selection failed on stability and shape, not on those caps.** The same
   candidate failed 5 regular cases: four where the *reference* delta or vega
   itself moved across the bump ladder by more than the frozen variation
   allowance, and one negative-rate American put whose price fell 2.7e-8 below
   its European dominance control against a 1e-8 shape tolerance. That gap is
   the size of the solver's accumulated PSOR residual, and the criterion was
   still not loosened. The 800x400 candidate failed 11 regular cases and the
   Richardson pair 9, so the declared order had no passing member.
3. **Richardson is a reference technique, not a label policy.** Where the
   solution is smooth it was the most accurate candidate by an order of
   magnitude, but the observed factor-two order fell outside the supported
   `[1.5, 2.5]` band in 11 of the 28 cases — concentrated at early exercise,
   dividends, short maturities and deep-in-the-money kinks. Unsupported order
   was preserved as a result rather than repaired by assuming second order.

The cost is the other result: 1,638 solves and about 126.6 million PSOR
iterations produced four labels for 28 states in 2.19 hours of single-core wall
time, because each state and grid costs thirteen independent solves. The
report's 250,000-label projections are idealized independent-worker
extrapolations of that measurement and are not production-feasibility claims.

No production label policy therefore exists, and none of these numbers may be
reused as an acceptance criterion for a later label-policy study.

### Provenance of the frozen snapshot after task 9C-C1

The snapshot's `source` block records the PDE header, source and composite
implementation digests **as they were when the pilot ran**. Task 9C-C1 changed
`cpp/include/dp/finite_difference_pde.hpp` and
`cpp/src/finite_difference_pde.cpp`, so those three recorded digests no longer
equal the current files, and that is correct: they are **historical task 9C-B
evidence**, not a statement about HEAD. They are frozen along with the rest of
the file by its pinned SHA-256 and must never be refreshed to match a later
build — doing so would silently transfer provenance the pilot never had.
`python/tests/test_pde_label_policy_results_snapshot.py` deliberately reconciles
only the config and runner digests against current files, because those two
inputs are the ones a rerun would have to reproduce exactly; it makes no claim
about the PDE digests either way.

Confidence that the current engine still reproduces the pilot's scalar
behaviour therefore rests on evidence, not on a digest: the scalar entry point
was shown to be **byte-identical** to the pre-9C-C1 implementation across seven
contracts covering both exercise styles, zero/one/two cash dividends, a negative
rate and an upwinded grid — price and every reported diagnostic printed at full
precision — and on the unchanged task 9C-A regression suites. Rerunning the
pilot to refresh provenance is not permitted: protocol-wise its partition was
consumed once, and its outcome is terminal.


## Task 9C-C3: label-policy v2

Task 9C-B met every predeclared absolute-error cap at 1600x800 and still
refused to select a policy, on two kinds of check that are **not** error caps:
bump-ladder movement of the *reference* delta and vega, and one
American-dominance violation whose size was the solver's own accumulated PSOR
residual. Task 9C-C3 predeclares a separate, one-shot answer to both. It is a
new versioned study, not a rerun: `configs/pde_label_policy_pilot_v1.toml`,
`python/src/differentiable_pricing/american/pde_label_policy.py` and
[results/american_pde_label_policy_results_v1.json](results/american_pde_label_policy_results_v1.json)
are read-only here and are byte-unchanged by it (decision log DEC-003).

**This section is the predeclaration. No v2 remediation or confirmation run has
been executed, and no v2 snapshot exists under `docs/results/`.** The
implementation delivers the versioned configuration
(`configs/pde_label_policy_pilot_v2.toml`), the criteria, eligibility rules and
lifecycle (`python/src/differentiable_pricing/american/pde_label_policy_v2.py`),
the freeze/check tool (`scripts/freeze_pde_label_policy_v2_results.py`) and
their tests, and nothing else.

### The locked protocol

- Ten remediation cases execute: the five v1 grid_1600x800 regular failures,
  three smooth regression controls (`regular_euro_atm_put`,
  `regular_euro_deep_itm_call_long_high_vol`,
  `regular_euro_deep_otm_put_long`), and the two declared anchors.
- **Nine regular cases decide pass/fail.**
  `stress_american_put_exercise_boundary` is mandatory descriptive/anchor
  evidence and never gates.
- `selection_order = ["grid_1600x800"]`. There is exactly one candidate.
- `grid_800x400` is an **order probe only** and can never be selected.
- Richardson stays **reference/validation-only** (DEC-010).
- The four v1 absolute-error caps are unchanged: price $5\times10^{-4}$, delta
  $10^{-3}$, gamma $2\times10^{-4}$, vega $5\times10^{-2}$ per unit
  volatility.
- **Gamma is evaluation-only and never gates.** Its error is published; it
  enters no pass/fail decision and no row is gamma-supervision-eligible
  (DEC-009).
- The 6400x3200 anchor rung is retained, for the two declared anchor cases, as
  descriptive evidence that gates nothing.
- Shape uses **D2**: one European dominance control, solved at the centre spot
  only, on the candidate grid, at the base volatility.
- A regular-case **price-critical** failure fails the candidate. An anchor-rung
  failure and any stress-case failure are descriptive.
- **Task 9C-C3 authorizes no dataset and no training input**, whichever way it
  resolves. `AUTHORIZED_TRAINING_INPUT_STATUSES` is empty.

The 28 confirmation case states are byte-for-byte the v1 design's case states.

### The price reference: raw 3200x1600 centre node, unconditional

This is a frozen v2 design decision and differs deliberately from v1.

- **Candidate price:** the exact-node centre value of the 1600x800 base
  surface.
- **Price reference:** the exact-node centre value of the **raw** 3200x1600
  base surface. `price_reference_method = "raw_grid_3200x1600_center"`, pinned
  in the configuration, in the criteria digest, and rejected at parse time and
  at report-verification time if it differs.
- **No price Richardson extrapolation exists, and no conditional
  price-reference fallback exists.** There is one branch.
- **`grid_800x400` is price-irrelevant.** It is solved only to estimate the E2
  delta observed order; its centre price enters no price quantity, and tests
  assert that changing it leaves the price block, the gating checks and both
  eligibility verdicts bitwise unchanged, while changing its stencil delta
  moves the observed order and nothing else.
- **Richardson remains a validation and reference technique** for the E2
  grid-stencil delta reference and for the explicitly descriptive anchor
  diagnostics, and only there (DEC-010). `richardson_contract` publishes
  `applies_to_price_reference = false`.
- **The 6400x3200 anchor stays descriptive** and can never veto ordinary price
  selection; a test asserts that moving the anchor price leaves the price block
  unchanged.
- All four v1 absolute caps remain unchanged.

v1's reference *was* selectively Richardson-extrapolated; that historical
statement belongs to task 9C-B and is not v2's rule.

### Everything material is pinned in the parser

The configuration is not merely validated for shape; every material fixed
value is compared against a module constant and rejected if it moved. That
covers the 28 case states **and their order** (through a pinned
`case_design_digest`), the exact ten remediation cases, the exact nine gate
cases, the two anchors, the five v1 failures and the three smooth controls,
every grid name/dimension/role, `selection_order`, the order band, the $K$
construction and its scheme version, the shape floor, the four caps, the
four canonical protocol strings (`study.name`, `delta_method`, `vega_method`
and `price_reference_method` — module constants the report is *built from*,
never free descriptive text), the
solver's residual settings and PSOR iteration ceiling, the E1 and E2
convention strings, gamma's evaluation-only status, the D2 dominance control,
Richardson's validation-only role, the reporting-only status of
`contract_multiplier`, the identity status of `settlement`, the lifecycle
vocabulary, and the anti-approval fields
(`selection_requires_fresh_top_level_approval`,
`authorizes_dataset_generation`, `authorizes_training_input`). Unknown and
missing keys are both errors, in every table.

### Exact nodes only, and one solve per grid

Every configured centre spot and every configured bumped spot must be an
**exact interior node** of every grid the study runs, on all four rungs,
checked at parse time and again bitwise against the returned node vector.
Nothing is interpolated and nothing is clamped.

That is what lets one base surface per grid supply the centre price, the
nodewise stencil delta and gamma, **and** every spot-bumped price: the bumped
spots are nodes of the same solve. There is no scalar centre solve and no
spot-bump duplication. Per case: one base surface per grid, six
volatility-bumped surfaces on the candidate grid, two on the reference grid,
one European dominance control for an American case, and one anchor surface
for an anchor case.

The stage therefore attempts **116** surface solves in remediation and **323**
in confirmation, **439** across a successful two-stage path. Every report
publishes a `solve_accounting` block — planned, attempted, completed, solver
exceptions, backward inductions, implicit linear solves, PSOR solves, total
PSOR iterations, counts by solve role, by criticality and by case class — with
three reconciliation identities that must hold:
`planned == attempted`,
`attempted == completed + exceptions`, and
`backward_inductions == completed`.

### E1: fixed-bump delta and vega validation

Let $D(h)$, $D(2h)$, $D(4h)$ be the candidate's centered fixed-bump estimate of
one Greek on the strict geometric ladder $(h,2h,4h)$. Write

$$
\Delta_{\text{small}}=|D(h)-D(2h)|,\qquad
\Delta_{\text{large}}=|D(2h)-D(4h)|,
$$

$$
K=\frac{1}{1-2^{-1.5}}\approx 1.5469,\qquad
p=\log_2\!\frac{\Delta_{\text{large}}}{\Delta_{\text{small}}}.
$$

The reference is a refined estimator using **the same primary bump
convention**, so its finite-bump truncation and the candidate's are the same
quantity and are charged **once**: one combined budget,

$$
\text{reference absolute error}+\text{bias charge}\le\text{unchanged Greek cap}.
$$

**Each ladder increment carries its own tolerance**, because the rungs it is
built from have different estimator error scales.

**Delta.** With $E=M_{\text{base}}R_{\text{base}}$ — the base surface's time
steps times its **absolute** `maximum_lcp_residual` — the centered difference
at bump $b$ has error scale $E/b$, so

$$
\varepsilon^{\Delta}_{\text{small}}=\frac{E}{h}+\frac{E}{2h}=\frac{3E}{2h},
\qquad
\varepsilon^{\Delta}_{\text{large}}=\frac{E}{2h}+\frac{E}{4h}=\frac{3E}{4h}.
$$

The bumped prices come from one shared surface, and **no cancellation between
its node errors is claimed**: the tolerance adds the two rung error scales.

**Vega.** With

$$
\mathrm{err}(b)=\frac{M_{\downarrow,b}R_{\downarrow,b}+M_{\uparrow,b}R_{\uparrow,b}}{2b},
$$

$$
\varepsilon^{\nu}_{\text{small}}=\mathrm{err}(h)+\mathrm{err}(2h),\qquad
\varepsilon^{\nu}_{\text{large}}=\mathrm{err}(2h)+\mathrm{err}(4h).
$$

Every sigma-down and sigma-up solve on the ladder — six on the candidate grid
— is solved, published and enters the tolerance explicitly.

Two branches:

- **Resolved** — at least one increment exceeds its own tolerance. Then $p$ is
  meaningful and is required to lie in $[1.5,2.5]$; an undefined, non-finite or
  zero-denominator $p$ can never satisfy it. The bias charge is
  $K\,\Delta_{\text{small}}$.
- **Flat** — both increments sit within their own tolerances. $p$ is a ratio of
  noise and decides nothing; the bias charge is
  $K\,\varepsilon_{\text{small}}$.

If $K\varepsilon_{\text{small}}$ exceeds the Greek cap, the solver's own noise
floor already exceeds the cap the check is written against. That is recorded as
`numerically_invalid_residual_scale`. It denies **that Greek** and never
vetoes the price policy.

Published per case and quantity: branch, $p$ when defined,
$\Delta_{\text{small}}$, $\Delta_{\text{large}}$, both epsilons, the bias
charge, the reference error, the combined error, the cap, and the verdict.

### E2: production grid-stencil delta validation

The production delta is the **nodewise grid-stencil delta** read off the
candidate surface at the exact centre node — the task 9C-C1 estimator.

- **Candidate:** the nodewise stencil delta on `grid_1600x800`.
- **Reference:** the grid-Richardson of the 1600/3200 stencil deltas.
- **Order:** measured from the 800/1600/3200 stencil deltas; required in
  $[1.5,2.5]$.
- **Single rule:**
  $|\text{candidate}-\text{reference}|\le\text{delta absolute error cap}$.

**No bump-bias charge is added**, because the grid-stencil reference converges
toward the mathematical derivative. An unsupported order names
`grid_stencil_observed_order_unsupported`; a violated rule names
`grid_stencil_absolute_error_exceeds_cap`. Either makes delta ineligible.

### Dominance and intrinsic: operational scale estimates, not bounds

Both use the **absolute** `maximum_lcp_residual`:

$$
\text{dominance scale}=M_{\text{A}}R_{\text{A}}+M_{\text{E}}R_{\text{E}},\qquad
\text{intrinsic scale}=M_{\text{A}}R_{\text{A}},
$$

$$
\text{allowance}=\max(\text{shape absolute floor},\ \text{corresponding scale}).
$$

If the corresponding scale exceeds the **price** cap, the case records
`residual_scale_exceeds_price_cap`, the allowance is **never widened past the
price cap**, and the check itself fails. These apply only to
`american_dominance` and `american_intrinsic_bound`, are **operational
price-error scale estimates**, and every report publishes
`is_a_rigorous_bound = false`.

### Role-aware decisions: what a failure is allowed to veto

Every solve is classified, and a failure only ever propagates to the decisions
its role supports.

| Solve | Criticality |
|---|---|
| `grid_1600x800/base`, `grid_3200x1600/base`, `grid_1600x800/european_dominance_control` | `price_critical` |
| `grid_800x400/base` | `delta_only` |
| `grid_1600x800/volatility_*`, `grid_3200x1600/volatility_*` | `vega_only` |
| `grid_6400x3200/base` | `anchor_descriptive` |
| every solve of a stress case | `stress_descriptive` |

The 800/1600/3200 solves that validate the candidate/reference protocol are
therefore distinguished from the extra descriptive 6400 anchor rung.

**A price-policy failure on a gate-eligible regular case** is any of: a
candidate/base price-solve failure; a failure of a solve needed to construct
the price reference; a centre European dominance-control failure; the
candidate price error above its cap; a price shape/bounds failure; a
non-finite, identity-mismatched or non-exact-node failure on a price-critical
solve; or a price-critical residual scale above the price cap.

**Delta-only ineligibility:** delta E1 failure; delta E2 failure or an
unsupported stencil order; a delta-specific residual-scale failure; a
delta-specific auxiliary (order-probe) solve failure;
$K\varepsilon^{\Delta}_{\text{small}}$ above the delta cap.

**Vega-only ineligibility:** vega E1 failure; structural C1 ineligibility; a
vega availability failure; a base/down/up regime mismatch; a vega convention or
bump mismatch; a sigma auxiliary solve failure; a vega-specific residual-scale
failure; $K\varepsilon^{\nu}_{\text{small}}$ above the vega cap.

**A Greek-specific failure never vetoes an otherwise valid price policy.** An
anchor-rung failure is descriptive: it records
`anchor_evidence_complete = false` plus the failure, and vetoes nothing. A
failure anywhere in the stress case is descriptive and never reaches the gate.
Gamma remains descriptive throughout.

An executable truth table covering every row above is asserted in
`python/tests/test_pde_label_policy_v2.py`.

### Structural C1 eligibility is common to both Greeks

The centre node's task 9C-C1 `greek_eligible` flag is part of the **common**
structural predicate. A row with `centre_greek_eligible = false` is delta
**and** vega ineligible; its price is unaffected unless an independent price
criterion fails. An uncertified regime makes a bumped price difference at that
node as uninterpretable as a stencil difference — this is a correction to an
earlier draft in which C1 constrained delta only.

Beyond C1, delta requires
$M_{\text{base}}R_{\text{base}}/\text{spot step}\le$ delta cap, and vega
requires
$(M_{\downarrow}R_{\downarrow}+M_{\uparrow}R_{\uparrow})/(2\eta)\le$ vega cap,
numerical availability, equal base/sigma-down/sigma-up exercise states, and a
vega convention whose bumped volatilities equal the planned ones **bitwise** —
the bump is never recovered by differencing two volatilities, because
$(\sigma_\uparrow-\sigma_\downarrow)/2$ is not exactly $\eta$ in float64.

`contract_multiplier` is **not** part of the pricing/grid identity and not part
of any eligibility predicate. `settlement` **is** part of the contract
identity.

### Solver exceptions are data, not control flow

Each solve is wrapped individually. An exception thrown by the binding is
caught and recorded as a structured failure carrying the case, the grid/rung,
the solve role, the option and exercise type, the exception class, a sanitized
single-line message with repository paths redacted, and the solve's
criticality. **The stage then continues**, so a complete report is always
emitted and the accounting still reconciles. `KeyboardInterrupt`, `SystemExit`
and `MemoryError` are re-raised rather than recorded.

The outcomes, by role: a regular price-critical exception fails the price
candidate; a regular vega-auxiliary exception makes vega ineligible only; a
delta-auxiliary (order-probe) exception makes delta ineligible only; an extra
6400 anchor exception is descriptive; any stress-case exception is
descriptive. Tests inject a throwing fake solver, not merely a returned
non-converged record.

### Lifecycle: exactly three supported states

| # | stage | remediation_status | confirmation_status | terminal | freezable | selection |
|---|---|---|---|---|---|---|
| 1 | `remediation` | `failed` | `not_run` | yes | yes | `no_policy_selected`, reason `remediation_failed` |
| 2 | `remediation` | `passed` | `pending` | no | **no** | **no selected-policy field at all** |
| 3 | `confirmation` | `passed` | `run` | yes | yes | `no_policy_selected`, or `grid_1600x800` pending fresh top-level approval |

There is no fourth state, and the freeze tool refuses state 2.

### Evidence JSON is parsed strictly

Every JSON read on the task 9C-C3 evidence path goes through one shared strict
loader — the confirmation stage's remediation report, the freeze tool's report
read, and both snapshot reads under `--extract` and `--check`.

- **Duplicate object keys are refused inside the parser**, at every nesting
  depth and inside objects nested in arrays, via `object_pairs_hook`. This
  cannot be delegated to canonical re-serialisation: ordinary parsing keeps
  only the *last* value of a repeated key, so by the time anything downstream
  looks at the document one value has already been discarded. The error names
  the duplicated key and its path — the offending objects are remembered by
  identity during parsing and their paths recovered by one top-down walk.
- **`NaN`, `Infinity` and `-Infinity` are refused.** They are Python
  extensions to JSON, not JSON, and `json` accepts them by default.

The refusal happens before `run_stage`, before any solver call, before
snapshot extraction, before snapshot writing and before any canonical-text
comparison. Tests inject duplicates into the **serialized file** — a Python
dict cannot express one — and assert that sentinel `run_stage` and solver call
counts stay empty and that no snapshot file is written.

### Exact recursive type validation

Before any `SurfaceSolve`, case object or confirmation decision is
constructed, the whole serialized document is validated against a centralized
exact schema: `REPORT_SCHEMA` in the runner and `SNAPSHOT_SCHEMA` in the freeze
tool, both built from the same small combinator set.

The rules are deliberately narrow, and none of them is a truthiness, an
arithmetic operation, a canonical comparison or an `isinstance` call —
**`bool` is a subclass of `int` in Python**, so `isinstance(True, int)` is
true and would let a JSON `1` masquerade as a flag:

| Kind | Rule |
|---|---|
| Boolean | `type(value) is bool` — `1` and `0` are rejected |
| Integer | `type(value) is int`, never `bool`, with range bounds |
| Real | `type(value) in (int, float)`, never `bool`, finite, with range bounds |
| String | `type(value) is str`, non-empty by default, plus exact enum vocabularies |
| Array | `type(value) is list`, every element validated recursively |
| Object | `type(value) is dict` with its **exact** key set, every leaf validated |
| Null | accepted only under an explicit `NullableSpec` |

No value is ever coerced. Coverage is exhaustive over every raw `SurfaceSolve`
field, the solve diagnostics and accounting fields, solve-failure records,
check records, every case-result field, the eligibility blocks and their
reason arrays, both fixed-bump records, the grid-stencil record, the price and
shape blocks, the stage outcome, `performance`, and every snapshot
study/provenance/criteria/outcome/accounting/case field — including every
nested list member and enum string. The unavailable fixed-bump record now
publishes its whole key set with explicit nulls, so a record's serialized
shape is uniform and the schema can demand it exactly rather than tolerate
absent keys.

Errors name the path, the expectation and the observation, for example
`cases[1].solves[0].centre_greek_eligible: expected boolean, observed int 1`.

Two schema-driven mutation tests walk **every distinct serialized position** of
a valid remediation report and of a valid snapshot, substitute an incompatible
type at each (a `bool` becomes `1`, an `int` becomes `true`, a `null` becomes
an object, and so on), and require every one to be rejected. They assert a
floor on the number of positions visited, so a future field that the schema
forgets shows up as a failure rather than as silence.

### Key closure: no map means "arbitrary string keys"

Exact types are not enough on their own: a JSON object whose keys are open
admits fabricated entries even when every value is well typed. Every map in
both schemas is therefore classified and constrained.

| Map | Class | Constraint |
|---|---|---|
| `cases[].solves[].bumped_prices` | context-derived | static vocabulary of the six canonical spot-bump keys, then the **exact** subset that solve's role requires |
| `cases[].solve_criticality` | context-derived | keys restricted to the criticality vocabulary, then exactly the classes that case's plan produces |
| `cases[].solve_problems` | context-derived | `<grid>/<role>` syntax, then **exactly** the problems that case's raw solve and failure records imply |
| `solve_accounting.attempted_by_solve_role` | context-derived | `<grid>/<role>` syntax, then exactly the stage's plan keys |
| `solve_accounting.attempted_by_criticality` | context-derived | criticality vocabulary, then exactly the classes present |
| `solve_accounting.attempted_by_case_classification` | context-derived | `regular` / `stress`, then **exactly** the per-classification planned-solve counts |
| `stage_outcome.failed_checks_by_name` | context-derived | the gating-check vocabulary, then **exactly** the failing gate-eligible cases per check |
| snapshot `outcome.failed_checks_by_name`, `attempted_by_case_classification`, `attempted_by_criticality` | enum-keyed subset | the exact set is fixed by the snapshot's own full recomputation |
| snapshot `attempted_by_solve_role` | context-derived | `<grid>/<role>` syntax, then exactly the stage's plan keys |

**No map is genuinely free-form.** An audit test enumerates every `MapSpec` in
both schemas and fails unless it carries an explicit key constraint *or* is
registered in `CONTEXTUAL_MAP_PATHS` with a contextual validator behind it.
The same audit, run against the pre-fix `bumped_prices` spec, reports it as
unbounded — which is how the defect is now prevented from recurring.

**A key constraint alone is not sufficient when the contents are derivable.**
Bounding a vocabulary still admits a valid-vocabulary key carrying a
fabricated value, and still admits the removal of a key that had to be there.
Every report map whose exact contents follow from the configuration and the
raw records is therefore listed in `EXACTLY_DERIVABLE_MAP_PATHS`, which the
module requires to be a subset of `CONTEXTUAL_MAP_PATHS`; a second audit test
fails any such map that is merely key-bounded. After this closure, every map
in the report schema is exactly reconciled, none is merely bounded.

Three of them are reconciled against derivations shared with the aggregate
recomputation rather than reimplemented:

| Map | Canonical rule | Shared derivation |
|---|---|---|
| `cases[].solve_problems` | one entry per *planned* solve that has at least one problem, sorted by key, value sorted; a problem-free solve is omitted, never published as an empty list | `raw_solve_problems`, over `solve_record_problems` per planned role |
| `stage_outcome.failed_checks_by_name` | one entry per gating check at least one *gate-eligible* case failed, sorted by name, value the sorted failing case names; gate eligibility comes from the configuration, never from the row's own claim; a check nobody failed is omitted | `raw_failed_checks_by_name` |
| `solve_accounting.attempted_by_case_classification` | one entry per classification the stage actually covers, sorted, value the number of solves the configuration plans for those cases; an uncovered classification is omitted, never published as zero | `raw_attempted_by_case_classification` |

`evaluate_case`, `stage_outcome` and `solve_accounting` call the same three
functions, so there is one implementation of these semantics, not two that can
drift. A test asserts the helper output equals the recomputed aggregate for all
four lifecycle variants. The full admission order is: strict JSON parse →
recursive static schema → raw-record contextual exact reconciliation →
dataclass construction → aggregate recomputation → confirmation admission and
`run_stage`. Sentinel tests wrap `surface_solve_from_record`, `run_stage` and
`pde_valuation_surface` and require all three call lists to stay empty for
every one of these mutations.

`bumped_prices` is the sharpest case, because the required set genuinely
differs by role:

| Solve role | Required `bumped_prices` keys |
|---|---|
| `grid_1600x800/base`, `grid_3200x1600/base` | all six: `spot_0.5_{down,up}`, `spot_1_{down,up}`, `spot_2_{down,up}` |
| `grid_800x400/base`, `grid_6400x3200/base` | none |
| `grid_1600x800/volatility_*`, `grid_3200x1600/volatility_*` | none |
| `grid_1600x800/european_dominance_control` | none |

The static schema restricts keys to the canonical six, derived from the pinned
bump ladder; `validate_report_context` then requires the exact subset for that
solve's grid and role, and errors name the solve path, its grid and role, the
unexpected and missing keys, and the expected set. Contextual validation runs
at the admission boundary — immediately after the static schema and **before**
any `SurfaceSolve` is constructed, any stage is run and any solver is called —
and is repeated immediately before construction in `_rebuild_case_inputs`, so
no extra map entry can reach `surface_solve_from_record`.

### Digests are typed as digests

Every field that is a digest is typed as one, not as an unrestricted string:
plain `sha256` fields require 64 lowercase hex characters, and each prefixed
identity requires its own prefix — `crit-`, `cases-`, `src-`, `pg-`, `vega-` —
followed by 64 lowercase hex characters. That covers the ten
executable-source digests, the raw-config and report digests, the three
`reported_binding_pde_*_sha256` fields, the criteria and case-design digests
in the report, the criteria block, the lifecycle and the snapshot, the
executable-source composite digest, every solve's `pricing_grid_identity`, and
both vega convention identities.

A name-based audit sweeps every schema position whose field name ends in
`_sha256` or `_digest`, or is a known identity, and fails if it is not typed
as a digest — so a future digest field added as a plain string is a failing
test rather than a silent hole. Parametrized mutation tests replace every one
of those fields, in both the report and the snapshot, with a non-digest, 63
hex characters, 65 hex characters, uppercase hex, non-hex characters, an
integer, a null (where not nullable) and a wrong prefix, and require rejection
at the schema boundary.

### Systematic object-key closure

Beyond type mutation, both schemas are swept for key closure. For every
distinct **structural** object position — array records are keyed by their
discriminator, so a solve is distinguished by grid and role, a case by
classification/gate-eligibility/anchor/failure shape, a failure by criticality,
a grid by name, rather than collapsed by index — the sweep adds an unexpected
key and, separately, removes each required key one at a time, and requires
every mutation to be rejected. It runs over all four report lifecycle variants
(remediation-passed, remediation-failed, confirmation-passed,
confirmation-failed) and all three terminal snapshot variants. Anything the
schema and contextual pass admit is re-checked through the full authoritative
path, so a removal inside a legitimately subset-keyed map is still caught by
aggregate recomputation.

### Confirmation entry is authoritative and semantic

Confirmation does **not** trust the six lifecycle fields of the remediation
report it is handed. Before any solve, `require_confirmation_entry`:

1. requires the exact report schema and rejects unknown, missing, duplicated or
   retyped fields **at every level**, not merely the top: `study`,
   `conventions`, `criteria_block`, `validation_rules` (and each of its
   `fixed_bump`, `grid_stencil`, `price` and `shape` sub-objects),
   `eligibility_contract`, `richardson_contract`, `predeclared_criteria`,
   `solver`, every `grids[]` record, and every case, solve, failure and check
   record. Each config-derived section is rebuilt from the configuration and
   compared as a whole canonical payload, which is strictly stronger than a key
   set. `performance` is the one runtime section, and it gets an exact key set
   plus per-field type and range checks;
2. validates the source/provenance block and reconciles the **whole executable
   source inventory** against the repository files;
3. verifies `raw_config_sha256` and the criteria digest against the
   externally supplied checked-in configuration, and rebuilds the entire
   `criteria_block` from it;
4. requires every expected case, in order, with its exact required solve roles,
   rejecting a missing, duplicated, renamed or unexpected case;
5. **recomputes every per-case check from the raw per-solve numbers** by
   re-running the same `evaluate_case` the runner ran, then recomputes price
   verdicts, delta eligibility, vega eligibility, gate counts, the aggregates,
   the solve accounting, `remediation_status` and `confirmation_status`;
6. requires the *recomputed* result to be `passed`/`pending`.

A report whose lifecycle claims `passed`/`pending` while its own case rows
imply `failed`/`not_run` is rejected **before the solver is called at all** —
asserted by a test that keeps a recording solver and requires its call list to
stay empty.

**Stated honestly:** semantic verification detects an inconsistent or partial
mutation, including one whose file hashes were regenerated. Without a
signature or a re-solve it **cannot** authenticate a fully coordinated
fabricated numerical report, and neither the confirmation gate nor the freeze
tool re-solves.

### The freeze tool recomputes rather than trusts

`scripts/freeze_pde_label_policy_v2_results.py` names its mode explicitly.
`--extract` runs the same authoritative verification above against the
checked-in configuration, additionally requires a **terminal** lifecycle state,
and only then distils the snapshot. `--check` revalidates the checked-in
snapshot: exact keys everywhere, all fixed thresholds and roles (not only the
four v1 caps) against the configuration, the exact ten-execution and nine-gate
case sets, the recomputed criteria digest, the runner and every executable
shared-source digest reconciled against repository files, and **every published
aggregate recomputed from the lowest-level rows**. The plan comes from the
configuration; the per-role, per-criticality and per-case-class breakdowns are
pure functions of the configuration and the case list; solver-exception counts
come from each row's own failure records; completed solves and backward
inductions follow from the accounting identities; and the three engine totals —
implicit linear solves, PSOR solves and PSOR iterations — come from minimal
immutable per-case fields the extractor derives from the raw solves, precisely
so that no retained aggregate is underivable. `failed_checks_by_name`, the
gating and descriptive failure counts, every verdict total and every
eligibility total are recomputed the same way, and each distilled row is
cross-checked against the caps and order band so a fabricated boolean or error
contradicts something. **A missing snapshot fails.** Mutation tests cover a
fabricated `price.absolute_error`, an unknown case key, a changed
`shape_absolute_floor`, a changed order bound, a flipped gate boolean, a
removed and a duplicated case, an altered lifecycle and an altered source
digest — each with hashes regenerated. The coordinated-rewrite limitation
above applies here too: the freeze tool does not re-solve.

### Executable-source provenance

Every report and snapshot records an individually named digest for each source
the runner and freeze path rely on — the v2 runner, the v2 freeze script, the
v1 label-policy module the v2 formulas were written against, the
`pde_surface_harvest` module supplying `canonical_payload`, the package
`__init__`, the PDE header, the PDE implementation, the Python binding source,
and **`cpp/include/dp/option.hpp` and `cpp/src/option.cpp`** — plus one
canonical composite digest over the whole inventory. Both the
confirmation-entry validator and the freeze tool reconcile every one of them
against the repository files, and a missing inventory file is an error: the v2
study is a source-checkout study.

The include graph of the PDE path was walked to confirm the inventory is
complete. Its first-party closure is exactly five files —
`bindings/python/pde_module.cpp` (which includes `dp/finite_difference_pde.hpp`
and `dp/option.hpp`), `cpp/include/dp/finite_difference_pde.hpp` (which
includes `dp/option.hpp`), `cpp/src/finite_difference_pde.cpp`,
`cpp/include/dp/option.hpp` (standard headers only) and `cpp/src/option.cpp`.
Every other include on that path is a C++ standard header. That closure is
`PDE_INCLUDE_CLOSURE` in the runner, and a test re-walks the `#include "dp/..."`
edges to prove no first-party edge escapes the inventory.

Two facts about the build are recorded rather than silently folded in. `_pde`
links the whole `dp_core` archive, which also compiles `binomial_tree.cpp`,
`black_scholes.cpp`, `least_squares_monte_carlo.cpp` and `smooth_mlp.cpp`;
none is reachable from the PDE path's include graph and the PDE path calls no
symbol they define, so they are deliberately outside the inventory. And
`CMakeLists.txt` selects what is compiled but is build definition rather than
executable source, so it is likewise outside it. Both are stated limits, not
oversights.

**These digests do not prove the loaded `_pde` extension binary was built from
those sources.** No such proof exists in this repository. The binding's own
build-time digests are recorded separately, under `reported_binding_*`, and
are self-reported by the build.

### One-shot enforcement, stated honestly

The stage CLI has **no `--overwrite` flag**. It refuses to run into a nonempty
output directory, and the freeze tool refuses to overwrite an existing
snapshot without `--update`. Both are **best effort**: a second run in another
directory or another clone cannot be detected from inside the runner. One-shot
status is **procedural, provenance-backed and independently reviewed**, and
the lifecycle block says so
(`one_shot_enforcement = procedural_and_provenance_backed`).

### Manual run protocol

Both stages are manual, terminal-invoked jobs (`AGENTS.md`, "No
agent-supervised expensive numerical runs"). No hook, CI job or agent may
launch or schedule either one.

```bash
python -m differentiable_pricing.american.pde_label_policy_v2 \
  --config configs/pde_label_policy_pilot_v2.toml \
  --stage remediation \
  --output-directory artifacts/pde-label-policy-v2-remediation

python -m differentiable_pricing.american.pde_label_policy_v2 \
  --config configs/pde_label_policy_pilot_v2.toml \
  --stage confirmation \
  --remediation-report artifacts/pde-label-policy-v2-remediation/report.json \
  --output-directory artifacts/pde-label-policy-v2-confirmation
```

Freezing a **remediation-terminal** outcome (remediation failed) uses the
remediation report:

```bash
python scripts/freeze_pde_label_policy_v2_results.py \
  --extract \
  --report artifacts/pde-label-policy-v2-remediation/report.json \
  --output docs/results/american_pde_label_policy_v2_results_v1.json
python scripts/freeze_pde_label_policy_v2_results.py --check
```

Freezing a **confirmation-terminal** outcome uses the confirmation report:

```bash
python scripts/freeze_pde_label_policy_v2_results.py \
  --extract \
  --report artifacts/pde-label-policy-v2-confirmation/report.json \
  --output docs/results/american_pde_label_policy_v2_results_v1.json
python scripts/freeze_pde_label_policy_v2_results.py --check
```

The snapshot filename `american_pde_label_policy_v2_results_v1.json` is the
canonical one and is identical in the script's default, in this document, in
`CONTRIBUTING.md` and in the PR 2 plan.

### Stated limits of this evidence

Published verbatim in every v2 report:

- **Validation reference errors and observed orders are unavailable per
  production row.**
- **A one-bump production row cannot re-estimate its own vega bump bias.**
- **The 28 evidence cases do not validate their surrounding hyperrectangle.**
- **Task 9C-C2b2 must separately predeclare its own generation domain.**
- **No dataset and no training input is authorized by task 9C-C3.**
- Semantic verification cannot authenticate a fully coordinated fabricated
  numerical report.
- One-shot enforcement is best effort and procedural.
- Source digests do not prove the loaded binary's provenance.

### Not implemented by the task 9C-C3 predeclaration change

The remediation run; the confirmation run; any v2 result snapshot; any change
to `scripts/check.sh` or CI; parallel or resumable generation; dataset
generation; neural training; any change to the C++ engine or its bindings; any
change to `pde_surface_harvest`. **Task 9C-B remains `no_policy_selected`.**
