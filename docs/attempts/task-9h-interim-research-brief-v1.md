# Task 9H interim research brief v1

**A self-contained request for independent scientific advice.**

This document is written to be read detached from its repository. Everything
needed to understand the project, the experimental history, the evidence and
the pending decision is reproduced here. No repository access is required.

Every number in this brief is copied from evidence whose SHA-256 digests were
verified before writing (see the provenance appendix, §16). Quantities computed
from those numbers rather than copied are labelled **[derived]**.

---

# 1. Executive summary

**What the project is trying to achieve.** The project is a research platform
testing a single falsifiable claim: that a smooth neural surrogate can
reproduce a trusted derivative pricer's *prices and its useful risk
sensitivities*, across a declared domain, at materially lower end-to-end
latency, with an error profile good enough for a stated decision. It proceeds
in stages of increasing difficulty, each with its own independently trusted
reference pricer and acceptance criteria fixed before results are observed.
Stage 1 (European options against analytic Black–Scholes) is complete and
independently replicated on a fresh seed and a fresh dataset. Stage 2
(American options against a converged CRR lattice) is where the project now is.

**Why American options are harder than European options.** A European price is
a smooth closed-form function of its inputs. An American price is the value
function of an optimal-stopping problem: the solution of a free-boundary
problem with no closed form, whose value function has a curvature ridge that
follows a moving early-exercise boundary and whose second derivative in spot is
discontinuous across that boundary in the limit. The map from contract state to
price is therefore not analytic: it is continuous and, away from the boundary,
smooth, but it contains a state-dependent kink locus. It also has *structural*
properties that a good pricer must respect — the price dominates both intrinsic
value and the matched European price, is bounded above, is monotone and convex
in spot, and monotone in volatility. A regression that is accurate on average
can still violate every one of these, and a violated lower bound is a
qualitatively different defect from a large residual: it makes the surrogate
unusable as a pricing function even where its error is small.

**Why a differentiable neural surrogate is useful.** The expensive part of
re-pricing a book is not one price; it is the Greeks, the scenario grid and the
exercise-aware models behind them. A CRR lattice costs $O(N^2)$ per price and
exposes no analytic sensitivity, so every Greek costs additional full solves. A
smooth network returns the value and *every* input sensitivity in one
reverse-mode pass, and its differentiability propagates: implied-volatility
inversion becomes a cheap, well-conditioned root-find with an exact derivative,
and a volatility surface can be calibrated directly in price space by gradient
descent through the pricer. That is the payoff the project is chasing. It is
also why *structural* validity matters more than it would for a plain
regression: an inverted price must be monotone in volatility for the inversion
to be well posed, and a calibration that differentiates through a non-convex,
non-monotone surrogate inherits its pathologies.

**The present outcome.** Five development attempts have run under Task 9H, an
explicitly exploratory loop on `train` and `validation` data only. The fixed
criterion has five conditions: three on price error and two on structural
diagnostics. The most recent attempt — a residual architecture with a plain
unconstrained output — **passed all three price-error conditions for the first
time**, with normalized RMSE 0.00211 against a 0.003 limit, p99 0.00816 against
0.015 and maximum 0.03110 against 0.08. The same model recorded **8,569
material bound violations and 683 material shape violations** on 25,000
validation rows, against a limit of zero for each. Separately, a small model
with a premium-over-European output head reduced bound violations from ~9,000
to 1,116 and shape violations to 330 — by far the best structural behaviour
observed — while its price tails got materially *worse* than the control's.

No model is accepted, and no attempt has met the complete criterion. Price
accuracy has been achieved by one architecture; structural validity has not
been achieved by any. Because Task 9H selects against `validation` repeatedly
and by design, none of its numbers is a project result.

**The decision required from the adviser.** The next experiment has
deliberately **not** been selected. The evidence separates cleanly into two
partial successes obtained by two different mechanisms — one architectural, one
in the output parameterization — that have never been combined, tested against
each other at matched capacity, or tested against a longer training budget that
the best-epoch evidence suggests may be binding. The adviser is asked which
single experiment is most informative next, and how the architecture,
output-head and training-budget hypotheses should be separated (§14).

---

# 2. Ultimate research objective

## The intended research story

The neural American pricer is not the deliverable. It is the first link in a
four-link chain:

1. **An accurate, differentiable American-price surrogate** that approximates a
   trusted American solver over a declared domain.
2. **Fast, stable sensitivities.** Because the surrogate is a smooth function
   of its inputs, delta, gamma, vega, theta and rho come from automatic
   differentiation in one pass, at a cost that does not scale with the number
   of Greeks requested — as opposed to bumped lattice solves, which do.
3. **Accelerated model-consistent implied-volatility extraction.** Inverting a
   quoted American price for its implied volatility is a scalar root-find in
   $\sigma$. With a lattice, each iteration is an $O(N^2)$ solve and the
   derivative $\partial V/\partial\sigma$ must itself be bumped. With a
   differentiable surrogate, each iteration is a forward pass and the vega is
   exact for the network, so a safeguarded Newton iteration converges in a
   handful of cheap steps.
4. **Joint calibration of a smooth volatility surface directly to American
   option prices.** With a differentiable pricer, a parametric volatility
   surface can be fitted by minimizing a price-space objective through the
   pricer itself, with analytic gradients with respect to the surface
   parameters — avoiding the conventional two-step route of inverting each
   quote to an IV point and then smoothing the resulting IV cloud.

## The intended comparison

The research design is a three-arm comparison of end-to-end pipelines that
produce the same object — a fitted volatility surface consistent with American
option prices:

| Arm | Pipeline | Role |
|---|---|---|
| **A. Numerical benchmark** | CRR (or PDE) American pricer → pointwise IV inversion → surface fit to the IV points | The trusted, slow reference. Defines correctness and the latency to beat. |
| **B. Neural pointwise inversion** | Neural American pricer → safeguarded pointwise IV inversion → surface fit | Substitutes the surrogate for the lattice in the *same* pipeline shape as A. Isolates the effect of replacing the pricer. |
| **C. End-to-end calibration** | Volatility-surface parameterization → neural American pricer → direct price-space calibration by gradient descent | Uses differentiability structurally, not just for speed. Skips the intermediate IV cloud entirely. |

Arm A defines truth and cost. Arm B tests whether the surrogate can be dropped
into a conventional pipeline. Arm C tests the differentiability claim itself.

## Where the current work sits

**The neural American pricer is the core dependency of all three arms.** Arms B
and C are not merely downstream in the sense of "later"; they are unbuildable
until arm B's pricer exists and is trustworthy. Concretely, at the time of
writing:

- **No accepted American price model exists.** That is the subject of this brief.
- **No Greek result exists.** No American Greek has been supervised, measured
  or claimed in this project. The datasets carry no Greek labels.
- **No IV extraction result exists** beyond a six-case model-consistent
  synthetic diagnostic recorded in a terminal and negative pilot (§6).
- **No surface calibration machinery exists**, in either arm B or arm C form.
- **No latency conclusion exists** for any accepted model.

Greeks, IV extraction and end-to-end surface calibration are **downstream
stages, not established results**. None of their machinery is built
speculatively, and no claim in this brief depends on them.

---

# 3. Current modeling scope and non-claims

## What is in scope, precisely

| Aspect | Current scope |
|---|---|
| **Dividends** | Continuous dividend yield $q$, constant per contract, sampled over $[0, 0.12]$. |
| **Volatility** | Constant $\sigma$ within a contract, sampled over $[0.05, 0.8]$. |
| **Rate** | Constant $r$ within a contract, sampled over $[-0.02, 0.12]$ — negative rates included. |
| **Yield** | Constant $q$ within a contract, as above. |
| **Contract types** | American call and American put, 50/50 by construction in every partition. |
| **Label source** | Synthetic labels from a CRR binomial lattice under the adjacent-average policy (§4). |
| **Domain** | Spot $[50, 150]$, log-moneyness $[-0.7, 0.7]$, maturity $[0.0192, 3.0]$ years. |

## Explicit non-claims

- **No discrete dividends.** The underlying carries a *continuous yield*. There
  is no ex-date, no cash amount and no dividend schedule anywhere in the data
  schema. A continuous yield is a different modelling object from a discrete
  cash distribution and is never treated as equivalent to one. Nothing learned
  here transfers as a claim about discrete-dividend American pricing.
- **No market quotes and no bid–ask evidence.** Phase 1 is entirely synthetic.
  There is no bid, no ask, and no market-relative error measure. A separate
  read-only market-data track exists but has produced no price, no label, no
  Greek, no implied volatility and no calibrated curve or surface.
- **No SPY or XSP claim.** Real-instrument work is a later phase against
  contracts the current dataset cannot represent (SPY pays discrete cash
  dividends).
- **No out-of-distribution or extrapolation claim.** The dataset has no
  boundary, extrapolation/OOD or scenario partition — only an interpolation
  test partition exists, and it is untouched. Nothing here says anything about
  behaviour outside the declared training envelope.
- **No accepted Greek result.** No Greek is labelled, supervised, measured or
  claimed.
- **No production or trading claim.** Nothing here is production-ready and
  nothing supports a trading decision.

## What the present task actually is

**The present task is learning a known mapping, not proving that mapping is
market truth.** The labels are CRR lattice outputs. The dataset is admitted
*conditionally and only* for the purpose of learning its known
continuous-yield CRR mapping; it is explicitly **not** accepted as evidence of
converged American-price accuracy, and its semantic coverage of the option
universe was never independently judged. The question under test is therefore:
*can a network reproduce this specific, well-defined, deterministic function to
a declared tolerance while respecting its structural properties?* Whether that
function is itself a converged approximation to the true American price, and
whether the sampled states are representative of anything, are separate
questions that this task does not address and does not claim to have answered.

---

# 4. Data, representation and target

## Dataset identity and generator

A single local dataset, `american-option-v1`, generated by a deterministic
C++20 CRR lattice (`dp::crr_binomial` v0.1.0, built Release with GNU 13.3.0),
under schema `american-option-dataset/1` and generator version `1.0.0`. It is
not tracked in version control; its identity is pinned by SHA-256 in the
experiment protocol and re-verified before every training run:

| File | Rows | SHA-256 |
|---|---:|---|
| `train.parquet` | 200,000 | `ff7a114f5bee260dba04c99afc78b0c9098fd9dfdd18c7a9dc952ce249ef50a9` |
| `validation.parquet` | 25,000 | `6f53a72e26a917f44324d3df86bffc4924eaf4739655f0563e63a0ad91f839c9` |
| `interpolation_test.parquet` | 25,000 | `f006a17cf5818f5950b0402e8942e6738726f2547e2e3c309038a836a1ed1058` |
| `manifest.json` | — | `252858929f3a981717dafc54c1639e1bd8159c5ad000aec7a0cc0cf24e9630d2` |

250,000 rows total, 29 columns, zero nulls and zero non-finite values. Every
row's `split` column matches its file. `sample_id` is unique within each
partition and the three partitions share no `sample_id`. Taking the full
contract state as the key, there are **zero exact duplicate states anywhere in
the 250,000 rows and zero exact state overlaps between any two partitions**. A
near-duplicate diagnostic (standardized nearest-neighbour distance to the
training set) found a minimum distance of 0.0116 and 0.12–0.14% of held-out
rows within 0.05; this is descriptive only — no near-duplicate threshold was
predeclared, so it is not a gate.

## Partition lifecycle

- `train` — used for fitting. Both the terminal Task 9G pilot and every Task 9H
  attempt fit on a deterministic 32,768-row subset of it (see below).
- `validation` — used for checkpoint selection and for every metric reported in
  this brief. All 25,000 rows are used. Task 9H selects against it
  **repeatedly and deliberately**, which is the source of the selection bias
  disclosed throughout.
- `interpolation_test` — the final partition. **It has never been opened.** In
  Task 9G it was never opened, hashed, imported or counted, and the frozen
  result records `final_evaluation_attempts = 0` and
  `final_partition_consumed = false`. Task 9H's code cannot reach it: the
  workbench fails closed on any final-partition token in a split name or a
  resolved path, strips the dataset manifest to the two reachable splits before
  anything else sees it, and exposes no final-evaluation command at all. Every
  one of the five attempt reports records `final_partition_touched = false` and
  `partitions_opened = ["train", "validation"]`.

**The final partition remains untouched.** Under the terminal pilot's protocol
a final evaluation is now *forbidden*, because that protocol's final-entry rule
failed. Any future confirmation requires a new predeclared protocol and a fresh
final partition.

## Row counts actually used

| Purpose | Rows | Partition |
|---|---:|---|
| Task 9G training (both arms) | 32,768 | `train` |
| Task 9H training (all five attempts) | 32,768 | `train` |
| Task 9G validation metrics (both arms + baseline) | 25,000 | `validation` |
| Task 9H validation metrics (all five attempts) | 25,000 | `validation` |

The Task 9H training subset is selected deterministically as the 32,768 rows
with the lowest `SHA-256(salt ‖ NUL ‖ sample_id)`, with `sample_id` as the
tie-break, under the fixed salt `task-9h-train-subset-v1`. **All five attempts
therefore see exactly the same 32,768 rows**, making the comparison paired by
construction.

## Label semantics: adjacent $N$ / $N+1$ CRR averaging

The label policy is `american-crr-adjacent-average/1`:

$$
V^{\text{label}} = \tfrac{1}{2}\bigl(\mathrm{CRR}(N) + \mathrm{CRR}(N+1)\bigr),
\qquad N = 1024 .
$$

Averaging adjacent step counts suppresses the well-known even/odd oscillation
of binomial American prices around the continuum limit. `label_steps` is the
constant 1024 in every row. This identity was recomputed on all 250,000 rows
and holds **bitwise, at zero tolerance**.

Adjacent-step agreement is a convergence *diagnostic*, not a proof of accuracy.
The CRR reference is an internal cross-check; it is not exact American truth
and not market truth.

## The paired European CRR baseline

Every row carries **both** an American price and a European price computed on
the *same lattice* with the *same* adjacent-average policy. The row is
therefore a paired American/European observation rather than a row tagged with
one exercise style. Consequences, all recomputed bitwise on all 250,000 rows:

- `early_exercise_premium == american_price − european_crr_price` exactly;
- `american_price ≥ european_crr_price` on every row, minimum difference
  exactly 0.0 (American dominance is exact, because the lattice is shared);
- `american_price ≥ intrinsic_value` up to floating point (largest shortfall
  −4.83e-13, i.e. rounding).

The stored European price serves two roles: it is the **no-learning baseline**
(its error against the American label *is* the early-exercise premium), and it
is the comparator in one of the two lower-bound diagnostics (§5).

Of the 25,000 validation rows, **19,668 have strictly positive early-exercise
premium and 5,332 have zero premium; 21,763 show exercise activity on the
lattice and 3,237 do not.** These sub-populations are reported separately
throughout.

## The five-input representation `american_forward_carry_v1`

Let $F = S e^{(r-q)T}$ and $A = S e^{-qT}$. The network's five inputs, in order,
are:

1. **encoded option type** ($+1$ call, $-1$ put);
2. $x = \log(F/K)$ — `log_forward_moneyness`;
3. $v = \sigma\sqrt{T}$ — `total_volatility`;
4. $a = rT$ — `rate_time`;
5. $b = qT$ — `yield_time`.

Feature and target standardization parameters are fitted on the selected
training rows only and are part of the model artifact.

## The target and its reconstruction

$$
u = \frac{V}{A} = \frac{V}{S e^{-qT}}, \qquad\text{reconstructed as}\qquad V = S e^{-qT} \, u .
$$

The physical wrapper computes the five coordinates from the physical contract
and performs the reconstruction *inside the differentiable graph*, so the
composed object is a differentiable map from physical contract parameters to a
physical price.

## Why the representation is mathematically sufficient for this model class

The pricing problem is scale-invariant in $(S, K)$ jointly, so dividing by $A$
removes one degree of freedom without loss. The remaining constant-parameter
CRR state is recoverable from the five coordinates, because

$$
\log(S/K) = \log(F/K) - rT + qT = x - a + b .
$$

The coordinates therefore retain spot moneyness *and* the separate $r$ and $q$
dependence of the early-exercise boundary — which matters, since the boundary
for a call is driven by $q$ and for a put by $r$, and a representation
collapsing them into the single carry $r-q$ would be insufficient. A
seven-input raw physical representation is also feature-sufficient but is not
minimal; the five-input form is the one used.

## Why European / intrinsic conditioning features are redundant but potentially useful

Two optional conditioning features were tested:

- `european_price_ratio` $= E(\text{state})/A$, where $E$ is the analytic
  continuous-yield Black–Scholes European price;
- `intrinsic_ratio` $= \max(\omega(S-K),0)/A$.

Both are **deterministic functions of inputs the network already receives**.
They add no economic information whatsoever, and in the information-theoretic
sense they are exactly redundant. They may nonetheless help as *conditioning*:
the European price is a closed-form expression involving $\Phi(\cdot)$ that a
tanh MLP would otherwise have to synthesize from scratch, and the intrinsic
ratio encodes the kink at the money that the network must otherwise learn. The
candidate therefore tests conditioning — whether handing the network an
expensive-to-synthesize function as an input makes the residual map easier —
not extra knowledge.

---

# 5. Fixed development criterion

## The criterion, unchanged

Task 9H's definition of "works" is **Task 9G's validation criterion, verbatim**.
It is read at runtime from a digest-pinned configuration file
(`configs/american_neural_pilot_acceptance_v1.toml`, section
`[validation_final_entry]`, SHA-256
`8ef937872e3cda84b36c766e373630dd6fb5c3471e25f032fe09c56ed310402e`) and applied
through the same code that judged Task 9G. No threshold is restated in Task 9H
code or configuration, so it cannot be quietly loosened after an attempt fails.

| Check | Threshold |
|---|---|
| normalized RMSE | $\le 0.003$ |
| normalized p99 absolute error | $\le 0.015$ |
| normalized maximum absolute error | $\le 0.08$ |
| material bound violations | $0$ |
| material shape violations | $0$ |

All five must hold simultaneously. Errors are normalized by the discounted spot
$A = S e^{-qT}$ — the same quantity that scales the target — so a normalized
error is a price error as a fraction of the discounted underlying, and the RMSE
limit of 0.003 is 30 basis points of $A$.

## The diagnostics, defined exactly

A violation is counted as **material** when it exceeds a tolerance of
$10^{-6} \times A$ for that row, i.e. one part per million of the discounted
spot. This excludes floating-point noise from the counts. `maximum_violation`
values quoted in this brief are in **physical price units**, not normalized.

Let $\hat V$ be the model's reconstructed price, $\omega = +1$ for a call and
$-1$ for a put, and $E^{\mathrm{CRR}}$ the stored paired European CRR price.

### Bound diagnostics (four; summed into `material_bound_violations`)

| Name | Violation measure | Eligible rows | Rationale |
|---|---|---:|---|
| `intrinsic_lower_bound` | $\max(\omega(S-K),0) - \hat V$ | all 25,000 | An American option can be exercised now, so it cannot be worth less than its payoff. |
| `european_comparator_lower_bound` | $E^{\mathrm{CRR}} - \hat V$ | all 25,000 | An American option dominates the otherwise identical European option. The comparator is the *stored CRR* European leg on the same lattice. |
| `american_call_upper_bound` | $\hat V - S$ on calls, $-\infty$ otherwise | 12,500 | A call is never worth more than the underlying. |
| `american_put_rate_aware_upper_bound` | $\hat V - K e^{\max(-rT,\,0)}$ on puts, $-\infty$ otherwise | 12,500 | A put is capped by the strike; the exponential factor keeps the bound valid under negative rates. |

### Shape diagnostics (three; summed into `material_shape_violations`)

Evaluated by finite bumps in the *physical* inputs, with the model re-evaluated
at bumped states. The spot bump is $\pm 1\%$ multiplicative; the volatility
bump is $\pm 0.01$ absolute. A row is **eligible** only if both bumped states
stay inside the declared domain (spot in $[50,150]$, log-moneyness in
$[-0.7,0.7]$, volatility in $[0.05,0.8]$) — which is why eligible-row counts are
below 25,000.

| Name | Violation measure | Eligible rows |
|---|---|---:|
| `spot_monotonicity` | calls: $\max(\hat V_{S^-} - \hat V,\; \hat V - \hat V_{S^+})$; puts: $\max(\hat V - \hat V_{S^-},\; \hat V_{S^+} - \hat V)$ | 24,470 |
| `spot_convexity` | $2\hat V - \hat V_{S^-} - \hat V_{S^+}$ | 24,470 |
| `volatility_monotonicity` | $\max(\hat V_{\sigma^-} - \hat V,\; \hat V - \hat V_{\sigma^+})$ | 24,758 |

That is: a call must be non-decreasing in spot and a put non-increasing; both
must be convex in spot and non-decreasing in volatility. Note that the shape
checks require **three extra forward passes per direction**, so they probe the
model's local behaviour, not just its values at the sampled states.

## What "zero violations" is and is not

**Zero observed violations is a development criterion over the declared
validation diagnostics, not a mathematical global guarantee.** Precisely:

- It is measured on 25,000 specific validation states (fewer for the shape
  checks, which have eligibility conditions), not over the continuum.
- The shape checks use finite bumps at a fixed scale. A violation at a
  different bump scale, or between the sampled states, would not be seen.
- The materiality tolerance of $10^{-6}A$ means sub-tolerance violations are
  not counted.
- Passing does not establish that the network is *globally* monotone, convex,
  or bounded — only that no counterexample appeared among the diagnostics run.

A model that passes has therefore demonstrated the *absence of detected
counterexamples* on a declared, finite battery. That is a meaningful engineering
bar, and it is deliberately a hard one (any single material violation fails),
but it is not a proof.

---

# 6. Task 9G chronology and result

Task 9G was the project's **confirmatory** American pilot: one predeclared
protocol, executed exactly once by a human operator, with every seed,
threshold, metric and case digest-locked before execution. It is now **terminal
and closed**, with an approved negative result. Its purpose here is to supply
the control against which Task 9H's development loop is measured, and to
explain why development — rather than a multi-seed replication — was the right
next move.

## Entry gates, all discharged before optimization

**Independent PDE mapping cross-check — passed 21 of 21 rows.** Before any
optimization, 21 validation states were re-priced with a numerically unrelated
engine: a one-dimensional Crank–Nicolson finite-difference solver with
Rannacher damping and a PSOR solve of the American obstacle, on a coarse
$400\times200$ and a fine $800\times400$ grid. All 21 agreed with the CRR label
within the predeclared tolerance. This is **mapping-consistency evidence only**
— it says the CRR labels are reproducible by an independent discretization. It
is not converged American-price truth, and because the refinement pair holds
`spot_maximum` fixed while changing resolution, it does not independently bound
domain-truncation error.

**Exact European-to-American transfer lift — passed at all eight probes.** The
transfer arm was constructed by an exact, bounded weight lift from the frozen
stage-1 European network (three inputs → five inputs): copy the three shared
first-layer columns, add **zero-initialized** columns for $rT$ and $qT$, copy
every remaining weight and bias, and algebraically rebase the standardization
so the represented physical function is unchanged. Before fine-tuning, the
lifted five-input network reproduced the source model's fully reconstructed
physical predictions at eight fixed protocol-pinned probe points:

| Quantity | Observed | Tolerance |
|---|---:|---:|
| maximum absolute difference | $1.4210854715202004\times10^{-14}$ | $1\times10^{-12}$ |
| maximum relative difference | $3.6214823824845716\times10^{-13}$ | $1\times10^{-12}$ |

This is **exact-lift evidence**: the American model at initialization *is* the
European model, to float64 precision, as a physical function. No general
transfer framework was built, and the European bounds projection was
deliberately not carried into the American output.

## The fixed one-seed / one-budget protocol

Two arms — `scratch` (random initialization) and `transfer` (the exact lift) —
under **one** training budget, **one** optimizer schedule, **one** row budget,
one checkpoint rule and one evaluation pipeline. One fixed seed per arm. No
hyperparameter sweep, no extra seed, no ablation, no recovery run. Architecture:
a float64 CPU `tanh` MLP with hidden dimensions $[64,64,64]$ — the same hidden
architecture as the frozen European source, which is what makes the exact lift
possible. Checkpoint selection used only `train` and `validation`.

A structural confound is recorded in the protocol itself: **the two arm seeds
also produce different epoch shuffle permutations**, not only different
initializations, so this one-seed pilot cannot attribute an observed arm
difference solely to transfer initialization.

## The result table

All figures on the 25,000-row `validation` partition, normalized by $A$.
The baseline is the stored paired European CRR price used with no learning at
all; its error against the American label *is* the early-exercise premium.

| Metric | `scratch` | `transfer` | European CRR baseline | Limit |
|---|---:|---:|---:|---:|
| normalized MAE | 0.004105128974273842 | 0.0017265443806429982 | 0.005560348888468108 | — |
| **normalized RMSE** | **0.006189463437549645** | **0.003035918363785017** | 0.0168479978486027 | **0.003** |
| normalized p95 absolute error | 0.01269144962082391 | 0.005674938764602986 | 0.02928215082641918 | — |
| **normalized p99 absolute error** | **0.022446213823595962** | **0.011377020296823961** | 0.080199881481468 | **0.015** |
| **normalized maximum absolute error** | **0.10502849577445739** | **0.08698284181815397** | 0.27191261705750525 | **0.08** |
| physical RMSE (price units) | 0.6132432400268641 | 0.29515611446193435 | 1.6983739861761729 | — |
| physical maximum absolute error | 7.718184408092569 | 7.136887689864807 | 34.57574365035366 | — |
| **material bound violations** | **9,482** | **8,513** | n/a | **0** |
| **material shape violations** | **1,168** | **947** | n/a | **0** |
| rows | 25,000 | 25,000 | 25,000 | — |

Bound- and shape-violation breakdown:

| Check | `scratch` violations (max, price units) | `transfer` violations (max, price units) |
|---|---|---|
| `intrinsic_lower_bound` | 1,657 (6.941219292028130) | 1,006 (4.241263272462451) |
| `european_comparator_lower_bound` | 7,825 (7.718184408092569) | 7,507 (7.136887689864807) |
| `american_call_upper_bound` | 0 (0.0) | 0 (0.0) |
| `american_put_rate_aware_upper_bound` | 0 (0.0) | 0 (0.0) |
| `spot_monotonicity` | 147 (0.3704368839162191) | 98 (0.21866537182921775) |
| `spot_convexity` | 924 (0.06839277958837897) | 529 (0.11194016425471887) |
| `volatility_monotonicity` | 97 (0.15743059067582976) | 320 (0.49910144416901403) |

### Gate outcomes

| Gate | `scratch` | `transfer` |
|---|---|---|
| normalized RMSE $\le 0.003$ | **FAIL** | **FAIL** (0.003036 vs 0.003) |
| normalized p99 $\le 0.015$ | **FAIL** | **PASS** |
| normalized maximum $\le 0.08$ | **FAIL** | **FAIL** (0.08698 vs 0.08) |
| material bound violations $= 0$ | **FAIL** | **FAIL** |
| material shape violations $= 0$ | **FAIL** | **FAIL** |
| **overall** | **FAIL (0/5)** | **FAIL (1/5)** |

**Transfer improvement over scratch.** Transfer's validation RMSE was strictly
better: 0.003036 versus 0.006190, a **50.95% reduction [derived]**; MAE fell
57.94% **[derived]** and maximum absolute error 17.18% **[derived]**. Bound
violations fell from 9,482 to 8,513 and shape violations from 1,168 to 947.
**This is a within-pilot observation, not evidence of transfer value**: with one
seed per arm and the arm-seeded shuffle confound, the difference cannot be
attributed to the transfer initialization. It also converts no failed gate into
a pass — transfer missed RMSE by 1.2% and maximum error by 8.7% **[derived]**.

## Latency ladder

A matched CRR-depth ladder $N \in \{256, 512, 1024, 2048, 4096\}$, with the
numerical comparator always the actual adjacent-average operation
$\tfrac12(\mathrm{CRR}(N)+\mathrm{CRR}(N+1))$. Neural end-to-end timing includes
feature transformation, standardization, model execution, inverse target
transformation and physical reconstruction. Seven repetitions; the reported
interval is a distribution-free order-statistic interval for the median with
actual coverage 0.984375. The interpretation depth is $N=1024$ — the dataset's
own label operation — and the protocol's threshold is a median end-to-end
speedup $\ge 10$ at every request shape.

| Shape | Depth | `scratch` median speedup | `transfer` median speedup |
|---|---:|---:|---:|
| single (batch 1, 1 thread) | **1024** | **4.041006376142777** | **4.427329993731772** |
| batch-8 (4 threads) | **1024** | **11.201979103596603** | **12.457299447562114** |
| single | 4096 | 53.01260524504374 | 87.03540144892827 |
| batch-8 | 4096 | 194.38333888406459 | 312.4295084268767 |

At the reference depth, the batch-8 shape cleared 10× but the single-request
shape did not (≈4×), so `all_reference_speedups_passed = false`. Any speedup is
conditional on the exact matched benchmark contract and does not generalize
beyond the recorded hardware, software, request shapes and thread budgets.

## Synthetic implied-volatility cases

Six protocol-pinned synthetic CRR cases spanning two maturities (0.25 and 2.0),
so the output is correctly termed an **IV surface** rather than a smile slice.
Each case inverts the American price for volatility under identical brackets and
stopping rules, comparing IV recovered from the label with IV recovered from
each arm. Zero inversion failures. The threshold is a maximum absolute
volatility error $\le 0.01$.

| Case | label IV | `scratch` abs. IV error | `transfer` abs. IV error |
|---|---:|---:|---:|
| `call_t025_k90` | 0.180000 | **0.034922992810606956** | **0.021478531882166862** |
| `call_t025_k100` | 0.200000 | 0.03427434526383877 | 0.0037742704153060913 |
| `put_t025_k110` | 0.240000 | 0.01685558259487152 | 0.013367688283324242 |
| `put_t2_k90` | 0.300000 | 0.0015427954494953156 | 0.0016853883862495422 |
| `call_t2_k100` | 0.280000 | 0.005712 | 0.002641 |
| `put_t2_k110` | 0.320000 | 0.008609 | 0.001601 |
| **maximum** | — | **0.034922992810606956** | **0.021478531882166862** |

Both arms exceeded the 0.01 threshold, so `all_iv_errors_passed = false`. The
short-maturity cases are the worst for both arms — consistent with the price
map being steepest in $\sigma$ where $\sqrt{T}$ is smallest, so a fixed price
error inverts to a larger volatility error. This is model-consistent synthetic
evidence only; it reports neither market fit nor bid–ask-relative performance.

## Locked outcome

`status = validation_gates_failed`, `outcome = failure_to_learn`,
`lifecycle.state = validation_terminal`,
`lifecycle.validation_final_entry_passed = false`,
`lifecycle.second_attempt_allowed = false`,
`lifecycle.final_evaluation_attempts = 0`,
`lifecycle.final_partition_consumed = false`.

Because the protocol's final-entry rule failed, **final evaluation is forbidden
under that protocol, now and later**. `interpolation_test` was never opened,
hashed, imported or counted, and it stays unconsumed. The result was frozen and
then materially approved by a fresh independent top-level review. That approval
accepts the recorded negative outcome; it approves no positive claim.

## Why 9G justified adaptive development rather than H2 replication

The project's hypothesis H2 concerns whether European-to-American transfer
reduces the label or time budget needed to reach equal-or-better price and
Greek error. Testing H2 requires a *replication*: at least five seeds and
several predeclared label budgets, all fixed in advance.

Task 9G was explicitly a **feasibility probe**, not an H2 test, and its
predeclared rule was that a *promising* pilot would justify predeclaring an H2
replication. The pilot was not promising: neither arm reached the accuracy
gates, the reference-depth speedup failed at one request shape, and the IV
errors exceeded their threshold. Running a five-seed replication of an
architecture that has not been shown to reach the accuracy bar at *any* seed
would be an expensive way to measure the variance of a failure.

The logically prior question is a different one: **does any architecture, under
this representation and this label set, learn the mapping at useful accuracy at
all?** That question is exploratory by nature — it requires trying things,
looking at validation, and trying the next thing. It cannot be predeclared,
and its answers cannot be confirmatory. Hence Task 9H: an explicitly
adaptive, explicitly biased development loop, whose output is a *recorded
search* rather than a result, and whose eventual candidate — if one emerges —
must be confirmed by a separate task with a new protocol and a fresh partition.

---

# 7. Task 9H development methodology

## What Task 9H is

**Explicitly exploratory and adaptive.** Its purpose is to iterate on capacity,
representation, target and architecture until an American *price* model meets
the fixed criterion of §5 on `train` and `validation`.

**`train` and `validation` only.** No code path may open, hash, stat, import,
count or inspect `interpolation_test` or any other final partition. The guard is
enforced in three places: the workbench strips the dataset manifest to the two
reachable splits before anything else sees it; a single forbidden-token
definition serves both the runtime guard and an offline static scan, so the two
cannot drift apart; and the runner exposes no final-evaluation command and no
flag that adds one. Retained manifest entries are guarded too — a `train` entry
pointing at `../interpolation_test.parquet` is refused rather than read.

**Every attempt and failure is recorded.** Successful, failed and abandoned
alike, in a single append-only JSONL log at a canonical path. An existing entry
is never rewritten. Once an attempt's output directory exists, the attempt ID is
*spent*: a crash after that point is recorded as an infrastructure failure and
the retry gets a new ID and a new configuration, because deleting the directory
to reuse the ID would erase the evidence that the first run happened.

**Clean committed source for every run.** The runner refuses to start with
tracked worktree modifications, *and separately* requires the selected
configuration and every file whose digest it records to be tracked at `HEAD` and
byte-identical to its `HEAD` blob. A merely clean `git status` would still admit
a brand-new untracked module, in which case the recorded commit would not
describe what ran. All five attempt reports record
`tracked_worktree_clean = true`.

**Immutable used configurations.** A configuration that has run is never edited;
a changed idea gets a new attempt ID and a new file. An offline checker
re-verifies that every logged attempt's configuration still hashes to the digest
that attempt recorded.

**Dataset identity is pinned to Task 9G's.** Before training, the manifest and
both reachable partitions must hash to the digests the Task 9G protocol already
locked, and the row-level label-policy verification is re-run over both
partitions using Task 9G's own code. A mismatch fails closed — otherwise an
attempt could train on a regenerated dataset and still be compared against the
Task 9G control.

**The human invokes every training run.** Agents implement code and
configuration and analyze compact summaries; long numerical runs are manual,
terminal-invoked jobs only.

## What is held fixed across all five attempts

| Held fixed | Value |
|---|---|
| Training rows | The same 32,768 rows from `train`, same salt, same selection rule |
| Validation set | All 25,000 rows |
| Shuffle seed | 3544330082 (identical in all five) |
| Representation | `american_forward_carry_v1` |
| Target / reconstruction | $u = V/(Se^{-qT})$; $V = Se^{-qT}u$ |
| Optimizer | AdamW, lr $10^{-3}$, weight decay $10^{-6}$, $\beta = (0.9, 0.999)$, $\varepsilon = 10^{-8}$ |
| Schedule | Cosine annealing, period 120 epochs, minimum lr 0.0 |
| Epoch budget | 120 |
| Batch size | 2,048 (evaluation 4,096) |
| Shuffle mechanism | `torch.randperm` from the shuffle seed, once per epoch |
| Checkpoint rule | Minimum standardized-target MSE on `validation`; earliest epoch wins exact ties |
| Precision / device | float64, CPU, 4 threads, deterministic algorithms enabled |
| Activation | `tanh` (all five) |
| Criterion | §5, read from the digest-pinned file |

**Initialization seeds differ per attempt**, derived from a per-attempt label
(`"<attempt_id>/init"` → first four bytes of its SHA-256, unsigned big-endian).
The shuffle label is shared, so every attempt walks the same rows in the same
order. See §12 for why the differing initialization seed is a confound.

## Why nothing here is a result

Selection happens against `validation`, repeatedly and deliberately: attempt
$k+1$ is chosen *because of* what attempts $1..k$ measured on `validation`. That
is the point of the loop, and it is exactly what makes the output biased. Every
recorded attempt carries the annotation *"measured against validation, which
this loop selects on repeatedly; a development measurement, not a project
result."*

**No Task 9H result is confirmatory.** A model selected by this loop requires a
separately predeclared confirmation, as its own task, with its own gates, its
own independent review, and a **fresh final partition** — not the untouched
`interpolation_test`, whose governing protocol already forbids final evaluation.

## Chronology

1. **`scratch_direct_control_v1`** — direct control: re-measure the Task 9G
   architecture and target on this branch as the baseline.
2. **`scratch_capacity_v1`** — larger dense capacity: was 9G capacity-limited?
3. **`scratch_american_premium_v1`** — premium over European: is the residual
   over a known European price an easier map?
4. **`scratch_conditioning_v1`** — European/intrinsic conditioning: is
   conditioning, not capacity, the binding constraint?
5. **`scratch_residual_architecture_v1`** — residual architecture: do identity
   shortcuts optimize better than a plain dense stack at comparable size?

---

# 8. Exact five-attempt comparison

All price metrics are on the 25,000-row `validation` partition, normalized by
$A = Se^{-qT}$. Every number is copied from a hash-verified attempt report,
summary and log entry (§16). Nothing in this table is recomputed or rounded
except where a value is shown truncated for width.

| Field | `scratch_direct_control_v1` | `scratch_capacity_v1` | `scratch_american_premium_v1` | `scratch_conditioning_v1` | `scratch_residual_architecture_v1` |
|---|---|---|---|---|---|
| **Source commit** | `82fe0c5a88a8d0dd4dd802ad005856adc9af6c42` | `85b99deede4e825af6242b2cad2c0df346b7e7a6` | `3c4fdb8f1f37b057d45c1e6b8bacb7b3517e36ee` | `7099c57ab79729b1ac96e2ac3ad9e58742b2358c` | `15bba65eea168a21df4fe7a799ccf793d494234f` |
| **Hypothesis** | The Task 9G architecture and target re-measured on this branch as the control every later attempt is compared against. | Task 9G failed to learn because the network was capacity-limited; a materially wider and deeper direct-output model reaches the price criterion. | Learning the American premium over an explicit European anchor is easier than learning the whole price map. | Conditioning, not capacity, is the binding constraint: at approximately control capacity, deterministic European and intrinsic features make the map materially easier to fit. | Identity shortcuts let a deeper stack of comparable inputs optimize better than a plain dense stack. |
| **Architecture** | `smooth_mlp`, hidden $[64,64,64]$, `tanh` | `smooth_mlp`, hidden $[256,256,256,256]$, `tanh` | `smooth_mlp`, hidden $[64,64,64]$, `tanh` | `smooth_mlp`, hidden $[64,64,64]$, `tanh` | `smooth_residual`, width 128, 6 blocks, `tanh` |
| **Features** | the five base coordinates | the five base coordinates | the five base coordinates | five base + `european_price_ratio` + `intrinsic_ratio` (7) | the five base coordinates |
| **Output head** | `direct` | `direct` | `premium_over_european` | `direct` | `direct` |
| **Parameter count** | 8,769 | 199,169 | 8,769 | 8,897 | 199,041 |
| **Initialization seed** | 3705668152 | 184978734 | 141483577 | 4147426778 | 3349003535 |
| **Shuffle seed** | 3544330082 | 3544330082 | 3544330082 | 3544330082 | 3544330082 |
| **Selected rows** | 32,768 (`train`) | 32,768 (`train`) | 32,768 (`train`) | 32,768 (`train`) | 32,768 (`train`) |
| **Best epoch** (of 120) | 120 | 119 | 120 | 119 | 117 |
| **normalized MAE** | 0.0037018087268151914 | 0.0022342033895660627 | 0.002970927993220988 | 0.002341956694778215 | **0.0013192895808582478** |
| **normalized RMSE** (limit 0.003) | 0.005725170440175143 | 0.003500912482274842 | 0.006982320321969395 | 0.004223135357900415 | **0.0021119419277138224** |
| **normalized p95** | 0.011703292404853077 | 0.0069888389983263155 | 0.010827970563730312 | 0.007361223397307244 | **0.0043419273415797694** |
| **normalized p99** (limit 0.015) | 0.020336109253536835 | 0.012952537821785156 | 0.03173820532293383 | 0.015688608195316058 | **0.008156614446419056** |
| **normalized maximum** (limit 0.08) | 0.11872539195311793 | 0.0659807379251851 | 0.13769917936282078 | 0.14799914435966932 | **0.0310974345996657** |
| **material bound violations** (limit 0) | 10,121 | 9,083 | **1,116** | 9,205 | 8,569 |
| **material shape violations** (limit 0) | 1,460 | 854 | **330** | 1,537 | 683 |
| gate: normalized RMSE | FAIL | FAIL | FAIL | FAIL | **PASS** |
| gate: normalized p99 | FAIL | PASS | FAIL | FAIL | **PASS** |
| gate: normalized maximum | FAIL | PASS | FAIL | FAIL | **PASS** |
| gate: material bound violations | FAIL | FAIL | FAIL | FAIL | FAIL |
| gate: material shape violations | FAIL | FAIL | FAIL | FAIL | FAIL |
| **gates passed** | 0 / 5 | 2 / 5 | 0 / 5 | 0 / 5 | **3 / 5** |
| **overall result** | `criterion_not_met` | `criterion_not_met` | `criterion_not_met` | `criterion_not_met` | `criterion_not_met` |
| **report SHA-256** | `f63f5786f395d3aa0206d979eab5c72046f5c740ea70a0bfe28b5f07d28834ab` | `8bebc3a43fd228d4724faefc66ac73e49738d7f3725458a19c6282a9b964bb98` | `847c42b3ba1e520bb39937e5b7bc7208f2e6c50d68d0cc47792237f4a237d209` | `2deb8cd300cbcc74430bf562a2faeb66a502c7c55710e9ac3a9e6701aac5653c` | `9614fc7b0de83cbfd488420bf90a91e391513265611a8f910d8173afbccf1e4b` |
| **summary SHA-256** | `ed2ab53ddeb01cafb5b0d5b5110da2b800a0a1a9fb1af5ed54e4df1e1bcc4ab6` | `55a96875f64599b11ad383034ee02e75166b88fa439ee9e70640e0ccedf62072` | `36731a275bd1fa84fea4262f3740c62aabc4bdd07e413f4325eb3ef2dbbd7b9c` | `beb305c111ce80a857d92b6286fbda52e6416f85be212c7a69510c8776cc91c3` | `dcdd03d6f20dd27634ce0c777ddfb184ad4414e8cfb985774d2db728f5451ca3` |

## Bound and shape breakdown, all five attempts

Counts are material violations; the parenthesized value is the maximum
violation in **physical price units**.

| Check | eligible | control | capacity | premium | conditioning | residual |
|---|---:|---|---|---|---|---|
| `intrinsic_lower_bound` | 25,000 | 1,595 (5.333765506932167) | 1,213 (4.101252829123254) | 1,116 (13.574191741755271) | 778 (7.02756258512278) | 1,041 (3.254580774016512) |
| `european_comparator_lower_bound` | 25,000 | 8,526 (9.51022228844161) | 7,870 (4.84869843115905) | **0 (0.0)** | 8,427 (13.532546834067773) | 7,528 (1.9031321376395454) |
| `american_call_upper_bound` | 12,500 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| `american_put_rate_aware_upper_bound` | 12,500 | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) | 0 (0.0) |
| **bound total** | — | **10,121** | **9,083** | **1,116** | **9,205** | **8,569** |
| `spot_monotonicity` | 24,470 | 163 (0.19625059586178395) | 74 (0.16347444710638315) | **20 (0.008668559450156765)** | 275 (0.2920217243297998) | 87 (0.11237652654770613) |
| `spot_convexity` | 24,470 | 1,140 (0.06869358444259888) | 650 (0.09308269581723039) | **80 (0.016394755943423434)** | 869 (0.07658396084545416) | 353 (0.12830713976001107) |
| `volatility_monotonicity` | 24,758 | 157 (0.44938586050007245) | 130 (0.43686285405755143) | 230 (0.6523074724152167) | 393 (0.18746898018277136) | 243 (0.21550668954995444) |
| **shape total** | — | **1,460** | **854** | **330** | **1,537** | **683** |

Two observations hold across every attempt without exception: **both upper
bounds are satisfied on every eligible row in every model**, and the
`european_comparator_lower_bound` is the single largest violation family in
every model except the premium-head one, where it is exactly zero.

---

# 9. Per-attempt findings

## 9.1 `scratch_direct_control_v1` — the baseline

**Changed from its comparator:** nothing. It re-measures the Task 9G `scratch`
configuration on this branch.
**Held fixed:** everything in §7's fixed table, plus the $[64,64,64]$ `tanh`
architecture, the direct head and the five base features.
**Hypothesis tested:** that the Task 9G architecture, re-run under the Task 9H
harness with the Task 9H row subset and seeds, reproduces a comparable failure
and can serve as the reference point for every later attempt.

**Exact result.** Normalized RMSE 0.005725170440175143, p99
0.020336109253536835, maximum 0.11872539195311793, MAE 0.0037018087268151914,
p95 0.011703292404853077. 10,121 material bound violations and 1,460 material
shape violations. Best epoch 120 of 120. **All five gates failed.**

**What was learned.** The control fails in the same qualitative way Task 9G's
`scratch` arm failed, at the same order of magnitude on every metric (9G
scratch: RMSE 0.006189, p99 0.022446, maximum 0.105028, 9,482 bound / 1,168
shape violations). The Task 9H harness therefore reproduces the phenomenon
being studied, and every later attempt has a stable comparator that will not
drift. The best epoch being the *last* epoch (120) is the first signal that the
budget may be binding.

**What cannot be inferred.** The control is not numerically identical to 9G's
scratch arm and should not be treated as a replication of it: the row subset,
the initialization seed and the shuffle seed all differ. Differences between
this control and 9G scratch are uninformative about anything.

## 9.2 `scratch_capacity_v1` — capacity substantially improves price accuracy

**Changed from its comparator (the control):** hidden dimensions
$[64,64,64] \to [256,256,256,256]$; 8,769 → 199,169 parameters, a factor of
**22.71 [derived]**.
**Held fixed:** representation, target, direct head, features, optimizer,
schedule, budget, batch size, rows, shuffle seed.
**Hypothesis tested:** that Task 9G failed because the network was
capacity-limited, and a materially wider and deeper *direct-output* model
reaches the price criterion.

**Exact result.** Normalized RMSE fell from 0.005725170440175143 to
0.003500912482274842 — a **38.85% reduction [derived]**, though still **16.70%
above the 0.003 limit [derived]**. p99 fell to 0.012952537821785156 (**PASS**)
and maximum to 0.0659807379251851 (**PASS**). Bound violations fell from 10,121
to 9,083 (−10.3% [derived]) and shape violations from 1,460 to 854 (−41.5%
[derived]). Best epoch 119. **Two of five gates passed** — the first time any
gate passed in this loop.

**What was learned.** Capacity is a real and substantial lever on *price
accuracy*: two of the three price gates flipped to passing. Capacity is a weak
lever on *structural validity*: a 22.7-fold parameter increase removed about a
tenth of the bound violations. Those two facts together are the central
observation of the whole loop — the price-accuracy problem and the structural
problem respond to different interventions.

**What cannot be inferred.** That capacity is *sufficient* for price accuracy —
RMSE still failed. That capacity is *irrelevant* to structure — the violations
did fall, just far too little. That 199,169 is near any optimum; only two
capacities were tried, on one seed each. Best epoch 119 of 120 means
optimization may not have plateaued, so part of the residual RMSE gap may be
budget rather than capacity.

## 9.3 `scratch_american_premium_v1` — the premium head greatly reduces violations but worsens price tails

**Changed from its comparator:** the output head only, `direct` →
`premium_over_european`. Architecture, parameter count (8,769), features and
everything else match the control exactly.
**Held fixed:** everything else, including the small $[64,64,64]$ network.
**Hypothesis tested:** that learning the American *premium* over an explicit
European anchor is an easier map than learning the whole price map.

**The head, stated exactly.** The network produces the normalized value

$$
u = \frac{E}{A} + s \cdot \mathrm{softplus}(\text{raw}),
$$

where $E$ is the **analytic continuous-yield Black–Scholes European price**
computed inside the differentiable graph, $A = Se^{-qT}$ and $s$ is the fitted
target scale. Two qualifications are recorded in the specification and are worth
restating for a numerical reader: (i) the bound is **non-strict** — `softplus`
underflows to exactly zero in float64 for a sufficiently negative
pre-activation, collapsing the price onto the anchor rather than below it; and
(ii) the bound is **not bitwise on the reconstructed physical price**, because
the reconstruction multiplies by $A$ and the price carries an $A\cdot(E/A)$
floating-point round-trip that can land one unit in the last place below $E$ —
a relative shortfall of order $10^{-16}$, immaterial against a $3\times10^{-3}$
criterion, but a near-bound rather than an exact one.

Note also that the anchor is the *analytic Black–Scholes* European price, while
the `european_comparator_lower_bound` diagnostic compares against the *stored
CRR* European leg. These differ by the lattice's own discretization error, so
the observed zero violations on that check are an empirical fact about this
dataset rather than an algebraic identity.

**Exact result.** **Bound violations collapsed from 9,083 (capacity) and 10,121
(control) to 1,116, and shape violations from 854 / 1,460 to 330** — by a wide
margin the best structural behaviour observed anywhere in this project,
including both Task 9G arms. Within that, `european_comparator_lower_bound`
went to **exactly zero on all 25,000 rows** (from 8,526 in the control), and
`spot_convexity` to 80 (from 1,140) and `spot_monotonicity` to 20 (from 163),
each with a maximum violation an order of magnitude smaller than any other
model's.

Price accuracy went the other way. Normalized RMSE rose to 0.006982320321969395
— **21.96% worse than the control [derived]** and the worst of all five
attempts. p99 rose to 0.03173820532293383 and maximum to 0.13769917936282078.
MAE, by contrast, was 0.002970927993220988, *better* than the control's
0.0037018087268151914. Best epoch 120. **All five gates failed.**

**What was learned.** The European anchor is powerful structural information:
supplying it as an explicit, architecturally enforced floor eliminated the
single largest violation family outright and cut convexity and spot-monotonicity
violations by roughly an order of magnitude. That structural improvement is not
a by-product of better fitting — this model fits worse — so it is attributable
to the parameterization itself.

The MAE-improves-while-RMSE-worsens pattern is diagnostic: the head made the
*typical* row better and the *tail* rows much worse. The slice evidence (§11)
localizes that tail: calls (RMSE 0.009388 vs puts 0.003061), long expiries
(expiry $\ge 2$: RMSE 0.010711, maximum 0.137699) and positive moneyness
($\ge 0.2$: RMSE 0.012711, p99 0.058051). These are precisely the regions where
the American-minus-European premium for a call is largest and most sensitive to
$q$ — the early-exercise region for a dividend-paying call.

**What cannot be inferred.** **This does not establish that premium prediction
is intrinsically inaccurate.** The head was tested only at the control's 8,769
parameters — the smallest capacity in the loop — against a target (the premium)
whose dynamic range and curvature differ substantially from the price's, and
under a target scaling $s$ fitted for the price rather than the premium. A
plausible reading is capacity or scale starvation on a harder-to-represent
quantity, not an intrinsic defect of the parameterization; another is that the
softplus introduces a genuine bias. **These are not separated by the available
evidence.** Nor can it be inferred that the residual 1,116 bound violations are
irreducible: they are entirely `intrinsic_lower_bound` violations, which the
European anchor does not address, since a European put (and a European call on a
dividend payer) can itself sit below intrinsic value.

## 9.4 `scratch_conditioning_v1` — conditioning improves average accuracy but not structure

**Changed from its comparator (the control):** two conditioning features added,
`european_price_ratio` and `intrinsic_ratio`, taking the input from 5 to 7
coordinates and the parameter count from 8,769 to 8,897 — a **1.5% increase
[derived]**, i.e. capacity is effectively held at the control's.
**Held fixed:** architecture shape, direct head, everything else.
**Hypothesis tested:** that conditioning, not capacity, is the binding
constraint — that at approximately control capacity, deterministic European and
intrinsic features make the map materially easier to fit.

**Exact result.** Normalized RMSE improved from the control's
0.005725170440175143 to 0.004223135357900415 — **26.24% better than the control
[derived]**, but still **20.6% worse than the capacity model's 0.003500912482274842
[derived]**. MAE 0.002341956694778215, essentially matching the capacity
model's 0.0022342033895660627 at 4.5% of its parameters. p99 0.015688608195316058
narrowly missed the 0.015 limit (**FAIL** by 4.6% [derived]). Maximum
0.14799914435966932 failed materially — **the worst maximum of all five
attempts**. Bound violations 9,205 and shape violations 1,537 — the latter the
**worst shape count in the loop**, worse than the control's 1,460. Best epoch
119. **All five gates failed.**

**What was learned.** Redundant conditioning is genuinely useful for *average*
fit: two extra deterministic inputs bought roughly two-thirds of the capacity
model's MAE improvement at 1/22 of its parameters. It did nothing for the tail
(the maximum error got worse than the control's) and nothing for structure
(shape violations got worse). Handing the network the European price as an
*input* is materially weaker than imposing it as an architectural *floor*
(§9.3): as an input the network is free to ignore or over-ride it, and it
evidently does so in the tail.

**What cannot be inferred.** That conditioning is useless — it is the
cheapest average-accuracy improvement observed. That conditioning and capacity
are substitutes; they were never tested together. That these two features are
the right ones; a third feature (`european_minus_intrinsic`) exists in the code
and was never tested. That the worse shape count is attributable to
conditioning rather than to this attempt's particular initialization seed —
with one seed per attempt, a 5% difference in violation counts between the
control (1,460) and this attempt (1,537) is not separable from seed noise.

## 9.5 `scratch_residual_architecture_v1` — all three price gates pass; structure does not

**Changed from its comparator (the capacity model):** the architecture,
`smooth_mlp` $[256,256,256,256]$ → `smooth_residual` with width 128 and 6
blocks. Parameter counts are 199,169 and 199,041 — **within 0.06% of each other
[derived]**, deliberately, so the comparison is at matched capacity.
**Held fixed:** direct head, five base features, representation, target,
optimizer, schedule, budget, rows, shuffle seed.
**Hypothesis tested:** that identity shortcuts let a deeper stack of comparable
size optimize better than a plain dense stack.

The block is $h \mapsto h + W_2\,\phi(W_1 h + b_1) + b_2$ — an identity
shortcut and nothing else. There is deliberately **no batch or layer
normalization and no dropout** anywhere in either architecture, because both
make a single row's prediction depend on batch composition or on a random mask,
which is not a property a pricing function may have.

**Exact result.** **All three price-error gates passed.**

| Metric | Value | Limit | Budget used [derived] |
|---|---:|---:|---:|
| normalized RMSE | 0.0021119419277138224 | 0.003 | 70.4% |
| normalized p99 | 0.008156614446419056 | 0.015 | 54.4% |
| normalized maximum | 0.0310974345996657 | 0.08 | 38.9% |

MAE 0.0013192895808582478 and p95 0.0043419273415797694 are likewise the best
in the loop. Against the *comparably sized* dense model, RMSE fell **39.67%
[derived]**; against the control, **63.11% [derived]**. Best epoch 117 — the
earliest best epoch of the five, though still within the final 3% of the budget.

Structurally it failed: **8,569 material bound violations and 683 material
shape violations**. Both counts are the *second* best in the loop, behind the
premium head by factors of 7.7 and 2.1 respectively **[derived]**, and the
bound count is only 5.7% below the capacity model's 9,083 **[derived]**. The
one notable structural improvement is qualitative rather than in the counts: the
maximum `european_comparator_lower_bound` violation fell to 1.9031 price units,
the smallest of any direct-output model (control 9.5102, capacity 4.8487,
conditioning 13.5325) — the model breaches the European floor about as often as
its peers, but by much less.

**What was learned.** At matched parameter count, residual connections
substantially outperform a plain dense stack on every price-error statistic.
Combined with §9.2, this establishes that the price-approximation difficulty was
never a representational-capacity ceiling in the function-class sense; it was at
least in part an *optimization* problem, which identity shortcuts address. The
price side of the criterion is, on this evidence, solved.

Equally: solving price accuracy did essentially nothing for structural validity.
A model that is 63% more accurate than the control still violates the European
lower bound on 30.1% of validation rows **[derived]**. **This is the strongest
evidence in the loop that the two failure modes are causally independent**, and
it identifies the unconstrained direct output — not the network's expressive
power, and not its optimization — as the remaining obstacle.

**What cannot be inferred.** That the residual architecture would pass the price
gates at a different seed; one seed was run. That width 128 × 6 blocks is
optimal, or that the improvement comes from depth rather than from the shortcut
per se — depth and topology changed together. That the price margin is robust:
the RMSE result uses 70.4% of its allowance, and the checkpoint was selected by
looking at this very validation set, so the true generalization RMSE on an
unseen partition is expected to be somewhat worse. That combining it with a
constrained head will preserve the price accuracy — untested.

---

# 10. Cross-attempt synthesis

## Evidence

Statements below are direct readings of hash-verified measurements. Each holds
for the specific configurations run, on one seed each, on this validation set.

1. **Capacity affects price accuracy.** Holding head, features, optimizer,
   budget and rows fixed, a 22.71-fold parameter increase reduced normalized
   RMSE from 0.005725170440175143 to 0.003500912482274842 and flipped the p99
   and maximum gates from fail to pass.
2. **Residual connections outperform the comparably sized dense model.** At
   199,041 versus 199,169 parameters — a 0.06% difference — the residual stack
   reduced RMSE by 39.67% **[derived]** (0.0021119419277138224 vs
   0.003500912482274842), p99 by 37.0% **[derived]** and maximum by 52.9%
   **[derived]**, and reduced bound violations by 5.7% and shape violations by
   20.0% **[derived]**.
3. **The premium head changes structural behaviour strongly.** Holding
   architecture, capacity and features exactly fixed against the control,
   switching the output head reduced material bound violations from 10,121 to
   1,116 (−89.0% **[derived]**) and material shape violations from 1,460 to 330
   (−77.4% **[derived]**), taking `european_comparator_lower_bound` violations
   from 8,526 to exactly 0. It simultaneously raised normalized RMSE by 21.96%
   **[derived]**, p99 by 56.1% **[derived]** and maximum by 16.0% **[derived]**,
   while *lowering* MAE by 19.7% **[derived]**.
4. **Conditioning alone is not sufficient.** At 1.5% extra parameters,
   conditioning improved RMSE by 26.24% **[derived]** over the control but left
   every gate failing, produced the worst maximum error in the loop
   (0.14799914435966932) and the worst shape-violation count (1,537).
5. **All best epochs are near the end of the 120-epoch budget:** 120, 119, 120,
   119 and 117. Not one attempt selected a checkpoint before epoch 117 —
   97.5% of the budget **[derived]**.
6. **No model reaches the structural criterion.** The minimum observed material
   bound violations across all five attempts and both Task 9G arms is 1,116
   (0.5% of eligible rows in the worst family), and the minimum shape
   violations is 330. The threshold is zero for both.
7. **Both upper bounds hold everywhere.** `american_call_upper_bound` and
   `american_put_rate_aware_upper_bound` recorded exactly zero material
   violations and exactly 0.0 maximum violation in **all five** Task 9H attempts
   and in **both** Task 9G arms.

## Inferences

These are interpretations, offered as such. Each is stated with the evidence it
rests on and the evidence it lacks.

- **The residual architecture appears to solve price approximation under the
  current validation criterion.** It passed all three price-error gates with
  29.6%, 45.6% and 61.1% of the respective allowances unused **[derived]**.
  Caveats: one seed; a validation-selected checkpoint on the same set the
  metrics are read from; no fresh-partition confirmation.
- **Unconstrained direct output appears to be the dominant structural
  weakness.** Four models with a direct head span a 2.7-fold range in RMSE
  (0.00211 to 0.00570) and a 22.7-fold range in parameter count, yet their bound
  violations lie in a narrow band of 8,569–10,121 — a 1.18-fold range. The one
  model with a constrained head sits at 1,116, an order of magnitude below all
  of them, *despite being the least accurate model in the loop*. Violation count
  tracks the output parameterization, not the price accuracy.
- **The premium result may indicate a useful inductive bias, but it was tested
  only at control capacity, so it does not establish that premium prediction is
  intrinsically inaccurate.** The confound is complete and unresolved: head and
  capacity were never varied independently. A premium head at 199k parameters
  has never been run. Alternative readings — excessive output bias from the
  softplus, a target scale $s$ fitted for the price rather than the premium,
  insufficient capacity for a higher-curvature target — are all consistent with
  the data and are not separated by it.
- **A residual-plus-premium interaction is currently untested.** The two
  mechanisms that produced the loop's two partial successes have never appeared
  in the same model. Whether they compose, interfere, or trade off is unknown.
  It is worth noting explicitly that composition is *not* guaranteed: the
  premium head constrains the output manifold, which could plausibly cost some
  of the residual model's price accuracy.
- **Longer training might improve errors but has no mechanism that guarantees
  structural validity.** Every attempt's best epoch sits in the last 3% of its
  budget, which is the classic signature of an under-trained cosine schedule —
  the learning rate reaches its floor and the model is still improving. That
  makes a budget extension a live hypothesis for the price metrics. But there
  is no argument, and no evidence in this loop, that more gradient steps on an
  unconstrained MSE objective would drive bound violations to *exactly* zero:
  the objective does not penalize them, four models spanning very different
  accuracies all violate at a similar rate, and the best-fitting model of the
  five still breaches the European floor on 30.1% of rows **[derived]**.

---

# 11. Slice-level failure analysis

Every attempt reports metrics on 17 slices. What follows summarizes the
recurring difficult regions; it does not reproduce all slices.

## Long expiry

The `expiry ≥ 2` slice (3,275 rows) is the worst or near-worst expiry bucket in
**every** direct-output model, and the gap widens with maturity:

| Model | expiry <0.25 RMSE | expiry 0.25–1 | expiry 1–2 | **expiry ≥2** |
|---|---:|---:|---:|---:|
| control | 0.004206 | 0.005981 | 0.005074 | **0.007824** |
| capacity | 0.002519 | 0.003173 | 0.003238 | **0.005309** |
| premium | 0.001856 | 0.004255 | 0.007826 | **0.010711** |
| conditioning | 0.001592 | 0.002845 | 0.004160 | **0.007414** |
| residual | 0.001930 | 0.002129 | 0.002037 | **0.002432** |

The premium head's degradation with maturity is the steepest of the five — its
long-expiry RMSE is 5.8× its short-expiry RMSE **[derived]** — consistent with
the American-minus-European premium growing with $T$ and the small network
lacking the capacity or scale to track it. The residual model is the flattest
across maturity (1.26× **[derived]**), which is part of why it passes.

## Extreme moneyness, both signs

Direct-output models are worst in the **negative** tail:

| Model | moneyness ≤ −0.2 (5,095 rows) | −0.2 to 0.2 (14,798) | ≥ 0.2 (5,107) |
|---|---:|---:|---:|
| control | **0.007937** | 0.004340 | 0.006559 |
| capacity | **0.004442** | 0.002916 | 0.003960 |
| conditioning | **0.007162** | 0.002745 | 0.003781 |
| residual | **0.002397** | 0.002043 | 0.002003 |
| premium | 0.003521 | 0.004727 | **0.012711** |

The premium head **inverts this pattern**: it is best in the negative tail and
by far the worst in the positive tail (RMSE 0.012711, p99 0.058051, maximum
0.137699 — the model's global maximum). Deep-OTM contracts have small premiums
and small European anchors, which the premium parameterization handles well;
deep-ITM contracts, where the early-exercise premium is largest, are where it
breaks down. The conditioning model's global maximum (0.147999, the loop's
worst) also sits in the negative-moneyness slice.

## Low and high volatility

Low volatility is the harder end for every model:

| Model | vol < 0.15 (2,323 rows) | 0.15–0.5 (19,469) | ≥ 0.5 (3,208) |
|---|---:|---:|---:|
| control | **0.008466** | 0.005240 | 0.006077 |
| capacity | **0.005730** | 0.003118 | 0.003570 |
| premium | **0.008520** | 0.007144 | 0.004196 |
| conditioning | 0.004952 | 0.003699 | **0.006180** |
| residual | **0.004136** | 0.001821 | 0.001502 |

For the residual model the low-volatility slice is a clear outlier: RMSE
0.004136 against 0.001821 in the body — **2.27× the mid-volatility figure
[derived]** — and it carries the model's global maximum error
(0.0310974345996657). Low volatility means a sharp early-exercise boundary and a
near-kinked value function, which is exactly where a smooth network should
struggle. The conditioning model is the sole exception to the pattern, being
worst at high volatility.

## Zero-premium and no-exercise rows

These two overlapping sub-populations (5,332 zero-premium and 3,237 no-exercise
of 25,000) are where the American price *coincides with* the European price:

| Model | premium: zero | premium: positive | exercise: none | exercise: observed |
|---|---:|---:|---:|---:|
| control | 0.006209 | 0.005587 | 0.006242 | 0.005644 |
| capacity | 0.003393 | 0.003530 | 0.003322 | 0.003527 |
| **premium** | **0.001781** | **0.007817** | **0.001821** | **0.007451** |
| conditioning | 0.004323 | 0.004196 | 0.005222 | 0.004054 |
| residual | 0.002126 | 0.002108 | 0.002123 | 0.002110 |

The premium head shows the loop's single most dramatic slice effect: on
zero-premium rows its RMSE is 0.001781, the best any model achieves on any
comparable slice, and on positive-premium rows it is 0.007817 — a **4.39×
ratio [derived]**. This is exactly what the parameterization predicts: where
the true American price equals the European anchor, the network need only drive
the softplus term toward zero, and it does so very well. Where a genuine premium
must be produced, it does poorly. The residual model, by contrast, is
essentially flat across this split (0.002126 vs 0.002108), meaning its accuracy
does not depend on whether early exercise matters.

## Call behaviour under the premium head

Option type is nearly neutral for four of the five models but strongly
asymmetric for the premium head:

| Model | calls (12,500) RMSE | puts (12,500) RMSE | call/put ratio [derived] |
|---|---:|---:|---:|
| control | 0.005973 | 0.005466 | 1.09 |
| capacity | 0.003508 | 0.003493 | 1.00 |
| **premium** | **0.009388** | **0.003061** | **3.07** |
| conditioning | 0.003433 | 0.004887 | 0.70 |
| residual | 0.001979 | 0.002237 | 0.88 |

Under the premium head, calls have 3.07× the RMSE of puts, p99 0.043412 versus
0.010708, and the model's global maximum error (0.137699) is a call. The
mechanism is plausible: an American call on a continuous-yield underlying has a
premium that is zero when $q=0$ and grows with $q$, so the call premium surface
has a degenerate region and a steep dependence on $qT$ — a harder object for a
small network than the put premium, which is driven by $r$ and is generally
better behaved. **This is a mechanism consistent with the data, not a
demonstrated cause.**

## Do the same rows drive maximum errors across models?

**Row-level identity is not available in the compact summaries.** The attempt
reports contain per-slice aggregates (count, sum of absolute error, sum of
squared error, maximum absolute error, and interpolated quantile brackets) but
**no per-row predictions, no `sample_id` for any extreme row, and no error
histogram**. It is therefore **not possible to state from this evidence whether
the same validation rows drive the maximum error across models.**

What *can* be said at slice granularity, and is worth noting as circumstantial:

- Within the residual model, the value 0.023464588018936583 appears as the
  maximum absolute error of six different slices simultaneously —
  `expiry ≥ 2`, `moneyness ≤ −0.2`, `volatility ≥ 0.5`, `option_type:put`,
  `premium_status:zero` and `exercise_status:no_exercise`. Since a row belongs
  to exactly one bucket of each slice family, this is consistent with **one**
  row — a long-dated, deep-OTM, high-volatility, zero-premium, no-exercise put
  — attaining that maximum in all six. The model's distinct global maximum,
  0.0310974345996657, is the maximum of the complementary buckets
  (`expiry 1–2`, `moneyness −0.2 to 0.2`, `volatility < 0.15`,
  `option_type:call`, `premium_status:positive`,
  `exercise_status:exercise_observed`), consistent with a single
  low-volatility, near-the-money, medium-dated call with a positive premium.
- The regions themselves recur across models — long expiry, extreme moneyness
  and low volatility are difficult for every direct-output model — but region
  recurrence is much weaker than row recurrence, and the premium head's
  inversion of the moneyness pattern shows the regions are not model-invariant.

Establishing row-level overlap would require a report extension (per-row error
export or top-$k$ `sample_id` capture), which is a code change and has not been
made.

---

# 12. Experimental limitations and confounds

1. **Different initialization seeds across attempts.** Each attempt derives its
   initialization seed from its own attempt ID, so all five differ (3705668152,
   184978734, 141483577, 4147426778, 3349003535). This is unavoidable in the
   sense that the architectures differ in shape and cannot share a weight
   tensor, but it means every between-attempt difference carries a seed
   component. The shuffle seed *is* shared (3544330082), so at least the data
   ordering is identical.
2. **Only one seed per attempt.** No variance estimate exists for any reported
   number. Differences smaller than seed noise cannot be distinguished from it,
   and the magnitude of seed noise in this setting has never been measured. The
   ~5% differences in violation counts between similar models are well inside
   plausible seed noise; the ~40–60% differences in RMSE probably are not, but
   that is a judgement, not a measurement.
3. **Adaptive validation reuse.** Every attempt selects its checkpoint on
   `validation` and is judged on `validation`, and each attempt was *designed*
   in response to earlier attempts' `validation` results. Reported metrics are
   therefore optimistically biased by an unquantified amount, and the bias grows
   with the number of attempts. The residual model's RMSE of 0.00211 should be
   read as an upper bound on its quality, not an estimate of its
   fresh-partition performance.
4. **The design is not a complete factorial.** Two architecture families
   (dense, residual), two capacities (~8.8k, ~199k), two heads (direct,
   premium) and two feature sets (5, 7) define a 16-cell space; five cells have
   been run. Missing in particular: premium at high capacity, residual with
   premium, residual with conditioning, conditioning at high capacity, and any
   cell with more than one factor changed from a run neighbour.
5. **The premium head was tested only with the small MLP.** This is the single
   most consequential gap. The head's effect and the small capacity are
   completely confounded, so the premium arm's poor price accuracy admits at
   least three unseparated explanations (§9.3, §10).
6. **Capacity and architecture changed together in some comparisons.** The
   residual model versus the *control* changes both topology and a 22.7-fold
   parameter count. Only the residual-versus-capacity comparison is
   capacity-matched (199,041 vs 199,169), and that is the comparison the
   architecture claim rests on. Similarly, conditioning versus capacity changes
   both features and capacity.
7. **The training budget may be insufficient.** All five best epochs are 117 or
   later out of 120 under a cosine schedule annealing to a zero floor. No
   attempt shows a clear plateau. Part of every reported error may be
   optimization shortfall rather than model limitation, and the ranking of
   architectures at 120 epochs need not be the ranking at 400.
8. **Labels are CRR mapping targets.** The learning target is
   $\tfrac12(\mathrm{CRR}(1024)+\mathrm{CRR}(1025))$, not the true American
   price. Error measured here is error against that mapping. The dataset is
   admitted only for learning this known mapping and is explicitly not accepted
   as converged-price evidence; its own discretization error is not bounded by
   anything in this brief.
9. **The zero-violation criterion is empirical over finite diagnostics.** As
   §5 sets out: 25,000 sampled states, fixed bump scales, a $10^{-6}A$
   materiality tolerance, and eligibility restrictions that exclude 2–5% of rows
   from the shape checks. Zero observed violations would not be a proof of
   global structural validity.
10. **No latency, Greek, IV or final-partition evaluation exists in Task 9H.**
    Its scope is price only, by design. The Task 9G latency and IV numbers in §6
    belong to a different, terminal experiment with a different architecture, and
    they are not properties of any Task 9H model. No Task 9H model has been
    differentiated, timed, inverted, or evaluated on any held-out partition.

---

# 13. Neutral candidate next experiments

Listed without ranking or recommendation. "Config-only" means a new immutable
configuration file against existing, already-committed code; "requires code"
means new model or loss machinery must be written, reviewed and committed first.

### 13.1 Residual architecture + premium-over-European head

- **Targeted failure:** the residual model's 8,569 bound violations, of which
  7,528 are `european_comparator_lower_bound` — exactly the family the premium
  head drove to zero.
- **Exact changes from `scratch_residual_architecture_v1`:** `head`
  `direct` → `premium_over_european`. Nothing else. New attempt ID, new
  initialization seed (derived from the new ID).
- **Config-only.** Both components exist and are dispatched.
- **Confounds:** the new initialization seed; the target scale $s$ is fitted on
  the price and is reused for the softplus premium term, which may be
  mis-scaled for a premium target; the anchor is analytic Black–Scholes while
  the diagnostic comparator is the CRR European leg.
- **Support:** bound violations drop to the residual `intrinsic_lower_bound`
  count alone (order $10^3$) with price gates still passing → the two mechanisms
  compose and the remaining problem is narrowly the intrinsic bound. **Reject:**
  price gates fail as they did in the small premium arm → the head's price cost
  is not a capacity artifact and is intrinsic to the parameterization.

### 13.2 Residual architecture + European/intrinsic conditioning

- **Targeted failure:** the residual model's low-volatility slice (RMSE
  0.004136 vs 0.001821 in the body) and its global maximum, where an explicit
  intrinsic feature should help most.
- **Exact changes from `scratch_residual_architecture_v1`:**
  `conditioning_features` `[]` → `["european_price_ratio", "intrinsic_ratio"]`;
  input dimension 5 → 7 (parameter count rises by ~256).
- **Config-only.**
- **Confounds:** new seed; conditioning's only prior test was at control
  capacity, so its interaction with a large residual stack is unmeasured; on the
  prior evidence conditioning *worsened* shape violations.
- **Support:** low-volatility and extreme-moneyness slice errors converge toward
  the body, with violations no worse. **Reject:** no slice improvement, or
  violations rise as they did in the small conditioning arm → conditioning does
  not survive scale.

### 13.3 Residual + premium + conditioning

- **Targeted failure:** both at once — the premium head's structural gain and
  conditioning's average-accuracy gain on top of the residual price accuracy.
- **Exact changes:** both of the above, together.
- **Config-only.**
- **Confounds:** the most confounded of the candidates — three factors differ
  from the residual comparator and two from either single-factor arm, so a
  failure would be uninterpretable and a success would not attribute credit.
  Note that `european_price_ratio` and the premium anchor use the *same*
  analytic European price, so the conditioning feature is partly redundant with
  the head.
- **Support:** all five gates pass. **Reject:** anything else — in which case
  the result is nearly uninformative about which component was responsible.

### 13.4 Longer-budget residual architecture, new attempt ID

- **Targeted failure:** best epoch 117/120 under a cosine schedule annealed to
  zero, in all five attempts — the signature of an incomplete anneal.
- **Exact changes from `scratch_residual_architecture_v1`:** `training.epochs`
  and `optimizer.schedule_period_epochs` 120 → a larger matched value (e.g. 360
  or 480). Everything else identical. The configuration is immutable after use,
  so this is necessarily a new attempt ID and file.
- **Config-only.**
- **Confounds:** new seed; changing both `epochs` and `schedule_period_epochs`
  changes the *shape* of the learning-rate trajectory, not just its length, so
  this is not a pure "more steps" experiment; a longer budget with
  validation-selected checkpointing increases the selection bias per attempt.
- **Support:** best epoch lands well inside the new budget and the price gates
  clear by a wider margin, with violations roughly unchanged → the budget was
  binding for price and is irrelevant to structure. **Reject:** best epoch is
  again at the end → the schedule shape, not the length, is what the best-epoch
  signal was reporting. Either outcome is informative about a confound that
  currently affects *all* five recorded attempts.

### 13.5 Differentiable lower/upper-bound parameterization

- **Targeted failure:** the 1,116 residual `intrinsic_lower_bound` violations
  that the European anchor does *not* fix (a European put, or a European call on
  a dividend payer, can itself sit below intrinsic), plus the
  `european_comparator_lower_bound` family — enforced jointly and
  architecturally rather than one at a time.
- **Exact changes:** a new head that maps the raw output through a smooth
  monotone bijection onto $[L, U]$ where
  $L = \max(\text{intrinsic}, E)$ and $U$ is the type-appropriate upper bound
  ($S$ for calls, $Ke^{\max(-rT,0)}$ for puts) — for instance
  $\hat u = \ell + (\upsilon - \ell)\,\sigma(\text{raw})$ in normalized units.
  A smooth $\max$ (or a softplus-based smoothing) is needed at the $L$ kink to
  keep the composite twice differentiable for Greeks.
- **Requires code:** a new head in the representation module, its dispatch, its
  configuration validation and its tests. It also requires deciding whether $E$
  is the analytic or the CRR European price, since only the latter matches the
  diagnostic.
- **Confounds:** a bounded output cannot represent a violation, so the two bound
  diagnostics become *tautologically* satisfied and lose their diagnostic value;
  the sigmoid saturates near the bounds, which may hurt exactly the deep-ITM and
  zero-premium regions where the price sits at a bound; the smoothed $\max$
  introduces a bias of its own; and it does nothing for the three *shape*
  checks.
- **Support:** bound violations become structurally zero *and* price gates hold
  and shape violations do not rise. **Reject:** price gates fail, or shape
  violations rise, → the constraint costs more than it buys.

### 13.6 Structural bound / monotonicity / convexity penalties in the loss

- **Targeted failure:** all seven diagnostics at once, including the three shape
  checks that no architectural output constraint addresses.
- **Exact changes:** add penalty terms to the training objective — hinge
  penalties on bound residuals evaluated at training states, and finite-bump or
  autograd-based penalties on $\partial \hat V/\partial S$ sign,
  $\partial^2 \hat V/\partial S^2$ sign and $\partial \hat V/\partial\sigma$
  sign — with one or more weights, optionally with collocation points sampled
  outside the training rows.
- **Requires code:** loss machinery, bump/derivative computation inside
  training, weight configuration and validation, tests. It also introduces the
  loop's first *tunable hyperparameter*, which multiplies the search space and
  the selection bias.
- **Confounds:** penalties are enforced at *training* states while the criterion
  is measured at *validation* states, so a penalty can fit the training
  constraint set without generalizing; the penalty weight trades price accuracy
  against structure along a curve that must itself be explored; derivative
  penalties triple or quadruple the cost per step, interacting with the
  budget confound in §12.7; and penalising exactly the seven declared
  diagnostics risks optimizing the metric rather than the property.
- **Support:** violations fall by orders of magnitude across *all* seven checks
  at modest price cost. **Reject:** violations fall only on penalized-and-measured
  checks while other structure degrades, or price gates fail at every usable
  weight.

---

# 14. Questions for the independent adviser

1. **What is the single most informative next experiment?** Given five recorded
   attempts, one seed each, an unmeasured seed variance and an adaptive loop
   that pays selection-bias cost per attempt, which one experiment maximizes
   information gained per validation exposure?
2. **Does the evidence justify combining the residual architecture with the
   premium head?** Or does composing the loop's two partial successes risk
   producing a result that is uninterpretable because the head's price cost was
   never separated from its capacity confound?
3. **Is the premium result evidence for a useful structural bias, an excessive
   output bias, insufficient capacity, or some combination?** Which measurement
   would separate them most cheaply — a premium head at 199k parameters, a
   premium head with a premium-fitted target scale, or something else?
4. **Should training duration be tested before changing the objective?** All
   five best epochs land at 117–120 of 120. Is resolving that confound a
   prerequisite for interpreting any further architecture or head comparison, or
   is it a lower-value branch than the structural question?
5. **Should bounds be enforced architecturally or penalized in the loss?**
   Architectural enforcement makes two diagnostics tautological and may saturate
   at the bounds; a penalty preserves the diagnostics' meaning but adds
   hyperparameters, cost, and train-versus-validation generalization risk. Which
   is the more defensible route for a research result, and does the answer
   change for a *development* criterion versus a confirmatory one?
6. **How should monotonicity and convexity be addressed while preserving useful
   differentiability?** Options include input-convex network structure,
   monotone-by-construction layers, derivative penalties, or accepting a small
   violation budget. The downstream goal requires *smooth, trustworthy first and
   second derivatives*, so a construction that achieves monotonicity by
   introducing kinks or clamps is self-defeating. What is the right trade?
7. **Is zero observed violation an appropriate development target?** It is
   inherited unchanged from a confirmatory protocol where it gated entry to a
   final evaluation. Is an exact-zero count over finite, finite-bump diagnostics
   the right objective for a development loop, or does it invite overfitting to
   the diagnostic battery — and if the latter, what would a better-posed
   structural target look like, given that it must be fixed before it is tested
   against?
8. **What is the smallest sequence that separates architecture, output-head and
   training-budget hypotheses?** Concretely: what is the minimal set of
   additional runs — and in what order — that would let one attribute price
   accuracy and structural validity to distinct causes with the confounds in §12
   resolved rather than accumulated?
9. **When should architecture exploration stop?** What evidence would justify
   concluding that the current representation, label set and model class cannot
   satisfy the criterion, as opposed to concluding that the search has not yet
   found the right configuration? Is there a principled stopping rule for an
   adaptive loop of this kind?
10. **What evidence should be required before fresh-partition confirmation?**
    A confirmation costs a fresh final partition, which is consumed
    irreversibly. Given the selection bias accumulated by then, what should be
    demanded first — multiple seeds at the candidate configuration? a
    validation-margin threshold well inside the gates? a held-out development
    split carved from `train`? — and how should the confirmation protocol's
    thresholds relate to the development criterion?

---

# 15. Current status

- **Five Task 9H attempts have completed.** `scratch_direct_control_v1`,
  `scratch_capacity_v1`, `scratch_american_premium_v1`,
  `scratch_conditioning_v1`, `scratch_residual_architecture_v1`. Each ran once,
  from a clean committed source tree, on the same 32,768 training rows with the
  same shuffle seed, optimizer, schedule, budget, batch size and checkpoint
  rule, and each is recorded in the append-only attempt log.
- **None passed the complete criterion.** All five recorded
  `outcome = criterion_not_met`. Best result: 3 of 5 gates.
- **The residual architecture is the current price-accuracy leader.**
  `scratch_residual_architecture_v1`: normalized RMSE 0.0021119419277138224,
  p99 0.008156614446419056, maximum 0.0310974345996657 — the first and only
  attempt to pass all three price-error gates.
- **The premium head is the current structural-violation leader.**
  `scratch_american_premium_v1`: 1,116 material bound violations and 330
  material shape violations, with `european_comparator_lower_bound` violations
  at exactly zero — an order of magnitude better than any other model on bounds,
  while failing all three price gates.
- **No combined experiment has run.** No model has both a residual architecture
  and a constrained head; no model has both high capacity and a constrained
  head; no model has run for more than 120 epochs.
- **No model is accepted.** Every Task 9H measurement is a development
  measurement selected against `validation`, carrying selection bias by
  construction. Nothing here is a project result and nothing here may be cited
  as one.
- **Final partitions remain untouched.** `interpolation_test` has never been
  opened, hashed, stat-ed, imported, counted or inspected. All five attempt
  reports record `final_partition_touched = false` and
  `partitions_opened = ["train", "validation"]`. Under Task 9G's protocol a
  final evaluation is forbidden; any future confirmation needs a new predeclared
  protocol and a fresh final partition.
- **The next experiment is deliberately not yet selected, pending independent
  advice.** No sixth attempt has been designed, no sixth configuration has been
  written, and no code has been changed in preparing this brief.

---

# 16. Provenance appendix

Every digest below was recomputed and verified before this brief was written.
Verification covered: each report's and summary's own SHA-256; the agreement of
each attempt's configuration digest as recorded in its report, its summary and
the attempt log against the configuration file's actual bytes; the agreement of
the criterion digest in all three places against the acceptance file's actual
bytes; the agreement of every `committed_source` git blob ID against the
tracked file; the agreement of every logged `source_digests` entry against that
attempt's own commit; and the equality of the overall normalized metrics,
violation counts and gate objects across report, summary and log for all five
attempts. **All checks passed.**

## Commits

| Commit | Subject | Role |
|---|---|---|
| `e930454d1c0869e22778a99ae22505f093889e73` | Task 9G implementation merge | The source state of the single locked Task 9G run |
| `5c0ef6a` | Merge PR #27 — Task 9G frozen result | Task 9G result-only closure |
| `82fe0c5a88a8d0dd4dd802ad005856adc9af6c42` | `experiment: add Task 9H price-model development workbench` | ran `scratch_direct_control_v1` |
| `193ab11` | `experiment: record Task 9H direct-control failure` | logged attempt 1 |
| `85b99deede4e825af6242b2cad2c0df346b7e7a6` | `fix: harden Task 9H attempt provenance and partition guards` | ran `scratch_capacity_v1` |
| `3c4fdb8f1f37b057d45c1e6b8bacb7b3517e36ee` | `experiment: record Task 9H capacity failure` | ran `scratch_american_premium_v1` |
| `7099c57ab79729b1ac96e2ac3ad9e58742b2358c` | `experiment: record Task 9H premium-head failure` | ran `scratch_conditioning_v1` |
| `15bba65eea168a21df4fe7a799ccf793d494234f` | `experiment: record Task 9H conditioning failure` | ran `scratch_residual_architecture_v1` |
| `32258bf` | `experiment: record Task 9H residual-model failure` | logged attempt 5; branch head |

Each attempt ran from the commit in the row that names it, with
`tracked_worktree_clean = true` recorded in its report; the *next* commit
records that attempt's result. Branch:
`experiment/task-9h-american-pricer-development`.

## Configurations (SHA-256 of file bytes)

| Path | SHA-256 |
|---|---|
| `configs/american_dev_attempt_scratch_direct_control_v1.toml` | `53e519a282adb259a1babeb3e5820bf6ac0f712d42cc58d23ed623e956fc8ee7` |
| `configs/american_dev_attempt_scratch_capacity_v1.toml` | `70c0afe6a4b4231938de59da27e676c2689229343f1e4b3bf27d5b2193e58446` |
| `configs/american_dev_attempt_scratch_american_premium_v1.toml` | `61ed7e12176e485712f5692fec0cb609d72bcc693afcaca036a991ed12f0dc82` |
| `configs/american_dev_attempt_scratch_conditioning_v1.toml` | `d460683cb769f35072a52e9ac64e696bf1d828d62bdc50cf7ab8264847392263` |
| `configs/american_dev_attempt_scratch_residual_architecture_v1.toml` | `c05c3489f4db251873dd17aa8fbaf8f5bb4e246e5830d9c7c9ee2b928c975a21` |
| `configs/american_neural_pilot_acceptance_v1.toml` (the criterion) | `8ef937872e3cda84b36c766e373630dd6fb5c3471e25f032fe09c56ed310402e` |
| `configs/american_neural_pilot_protocol_v1.toml` (dataset identity lock) | `6600a46ea2132bd3c834645ddb704ccc71069a0cca84d3762b4c4ca753aac591` |
| `configs/american_option_dataset_v1.toml` (generator config, recovered) | `d18485c66b92c720c57bef6820e7f6cdb7204159c2dcf8d47d8f9c744cb28c98` |

The five attempt configurations differ from one another **only** in: the header
comment, `attempt_id`, `hypothesis`, `seeds.initialization_label`,
`paths.output_directory`, `paths.ledger`, and the single field each one is
testing (`architecture.hidden_dimensions`, `architecture.name`/`width`/`blocks`,
`head`, or `conditioning_features`). This was verified by direct textual
comparison against the control.

## Attempt reports and summaries (SHA-256 of file bytes)

| Attempt | `attempt-report.json` | `summary.json` |
|---|---|---|
| `scratch_direct_control_v1` | `f63f5786f395d3aa0206d979eab5c72046f5c740ea70a0bfe28b5f07d28834ab` | `ed2ab53ddeb01cafb5b0d5b5110da2b800a0a1a9fb1af5ed54e4df1e1bcc4ab6` |
| `scratch_capacity_v1` | `8bebc3a43fd228d4724faefc66ac73e49738d7f3725458a19c6282a9b964bb98` | `55a96875f64599b11ad383034ee02e75166b88fa439ee9e70640e0ccedf62072` |
| `scratch_american_premium_v1` | `847c42b3ba1e520bb39937e5b7bc7208f2e6c50d68d0cc47792237f4a237d209` | `36731a275bd1fa84fea4262f3740c62aabc4bdd07e413f4325eb3ef2dbbd7b9c` |
| `scratch_conditioning_v1` | `2deb8cd300cbcc74430bf562a2faeb66a502c7c55710e9ac3a9e6701aac5653c` | `beb305c111ce80a857d92b6286fbda52e6416f85be212c7a69510c8776cc91c3` |
| `scratch_residual_architecture_v1` | `9614fc7b0de83cbfd488420bf90a91e391513265611a8f910d8173afbccf1e4b` | `dcdd03d6f20dd27634ce0c777ddfb184ad4414e8cfb985774d2db728f5451ca3` |

Reports and summaries live under an ignored artifacts tree and are not tracked
in version control; the append-only attempt log at
`docs/attempts/task-9h-attempt-log.jsonl` is the tracked record and carries the
same metrics, verified equal.

## Task 9G frozen evidence

| Item | Digest |
|---|---|
| `docs/results/american_neural_pilot_results_v1.json` | `8a125c81d469c320bc3a2cea9709f695e11c0a3a4ed59b985b88b93e3f3cdedc` |
| Task 9G raw validation report (ignored, distilled from) | `9a4acdfd3b96eee3d29ef9c12c467ea304297dc44d561f56fdaeb2292ba999a9` |
| Frozen European source-model `weights.npz` (ignored, transfer source) | `42670774f736383e50818b6e6c1db9374a77988173e35423ffc34b3c4297ecb8` |
| European source artifact manifest | `054ca945de851b4e5b2a10455e93be44546770410bcf698e0051175fe8d19f51` |

## Runtime environment of record

Python 3.11.15, PyTorch 2.13.0+cpu, NumPy 2.4.6, PyArrow 25.0.0; Linux
6.18.33.2-microsoft-standard-WSL2 x86_64 with glibc 2.39; GNU 13.3.0, Release
build; 8 logical CPUs. All training is float64 on CPU with deterministic
algorithms enabled and a 4-thread budget.
