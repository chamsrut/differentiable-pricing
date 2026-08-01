# American LSM numerical contract

This document specifies the independent Monte Carlo cross-check for the
constant-parameter, one-factor American-option stage. It implements the
least-squares Monte Carlo (LSM) method introduced by
[Longstaff and Schwartz (2001)](https://people.math.ethz.ch/~hjfurrer/teaching/LongstaffSchwartzAmericanOptionsLeastSquareMonteCarlo.pdf).
The implementation is a second numerical method under the same risk-neutral
GBM assumptions as Black--Scholes and CRR. It is not market calibration and
does not make the model assumptions more realistic.

## Dynamics and estimand

Under the pricing measure,

\[
\frac{dS_t}{S_t}=(r-q)\,dt+\sigma\,dW_t.
\]

For \(M\) equal exercise intervals, \(t_m=mT/M\). The implementation estimates
the value of a learned stopping policy exercisable on
\(\{0,t_1,\ldots,t_M\}\). This is a Bermudan approximation to continuous
American exercise.

Four errors must not be conflated:

1. the finite exercise grid is worth no more than continuous exercise;
2. a learned policy is generally suboptimal even on that grid;
3. the learned policy varies with the finite policy-training sample;
4. valuation of the fixed policy has Monte Carlo sampling error.

Only item 4 is described by the reported standard error and confidence
interval. A fixed stopping policy produces a lower-bound expectation for the
same-grid optimal stopping problem. A finite-sample estimate, especially after
a zero-mean control-variate adjustment, can still lie above a tree value.

### The reported interval is valuation-only

The interval is **not** a coverage statement about the same-grid optimal value,
the continuously exercisable American value, or any tree reference. It has no
nominal rate against those quantities, so no count derived from it is an
acceptance gate and none is used as one.

Items 1--3 are systematic: they do not shrink as valuation paths increase.
Item 4 does, at the usual \(n^{-1/2}\). Adding valuation paths therefore
narrows the interval around a fixed policy-and-grid bias, and the number of
cases whose interval contains a finer-grid reference is expected to **fall**
towards zero rather than approach any target fraction. Reading `0/N` as failed
95% coverage inverts the meaning of the experiment: it is evidence that
valuation noise has been driven below the bias, which is the point of adding
paths.

For this reason the report separates two things a single containment count
would conflate:

- **stochastic cases**, whose interval has positive width and whose
  containment is informative about the size of the bias relative to noise;
- **deterministic zero-width cases**, whose valuation estimator has exactly
  zero variance. These arise from immediate exercise at time zero and from
  structural suppression, where the control variate reproduces the payoff
  exactly. Their interval is a single point, so containing a finite-step tree
  value is arithmetically impossible no matter how close the agreement.
  Counting them as containment failures would misrepresent agreement at the
  \(10^{-13}\)--\(10^{-5}\) level as disagreement, so their tree differences
  are reported separately instead.

No exact-recovery tolerance is invented to convert those differences into
successes. They are reported as differences.

## Independent policy fitting and valuation

The two explicit 64-bit seeds identify separately seeded streams:

- `training_seed` generates the paths used to fit the stopping rule;
- `valuation_seed` generates fresh paths used only after that rule is frozen.

The engine rejects equal seeds, and the study derives both with different
SHA-256 domain labels. This separation avoids the upward look-ahead bias
caused by evaluating a fitted exercise rule on the paths used to fit it.

At expiry, cash flow is intrinsic value. At each earlier exercise date, the
training pass:

1. selects in-the-money paths;
2. discounts each path's current future stopping cash flow back to that date;
3. regresses it on a polynomial in standardized log moneyness;
4. exercises when intrinsic value strictly exceeds the nonnegative fitted
   continuation value.

For \(x=\log(S/K)\), the date-specific coordinate is

\[
z=\frac{x-\bar{x}}{s_x},\qquad
\widehat C(x)=\max\left(0,\sum_{j=0}^{d}\beta_j z^j\right).
\]

The degree is restricted to \(1\le d\le3\). Modified Gram--Schmidt QR with a
reorthogonalization pass solves the small least-squares problem. Insufficient
or numerically rank-deficient samples fall back to the mean discounted cash
flow, and every fallback is reported. Increasing polynomial degree without a
commensurate path study is not harmless; the path/basis trade-off is the
subject of
[Glasserman and Yu (2004)](https://arxiv.org/pdf/math/0503556).

The time-zero choice is also fixed from the training sample before valuation.
For a call with \(r\ge0\) and \(q\le0\), the engine applies the structural
no-early-exercise result rather than allowing sampling noise to invent an
exercise region. It likewise suppresses early put exercise when \(r\le0\) and
\(q\ge0\): the European lower bound
\(K e^{-r\tau}-S e^{-q\tau}\) is then at least \(K-S\), so continuation
dominates intrinsic value.

## Antithetic sampling and uncertainty

Both streams use a versioned SplitMix64 plus Box--Muller normal generator.
Each random normal sequence drives a positive-shock path and its antithetic
negative-shock path. The estimator treats the average of those two payoffs as
one independent observation. Consequently,

\[
n_{\mathrm{eff}}=\frac{P_{\mathrm{valuation}}}{2},
\qquad
\operatorname{SE}(\bar Z)
=\sqrt{\frac{s_Z^2}{n_{\mathrm{eff}}}}.
\]

Treating all paths as independent would understate uncertainty because the two
members of a pair are correlated. The reported 95% interval is the
large-sample normal approximation

\[
\bar Z\ \pm\ 1.959963984540054\,\operatorname{SE}(\bar Z).
\]

## European control variate

On every valuation path, let \(X\) be the discounted payoff from the frozen
American stopping policy and \(Y\) the discounted European payoff at maturity.
The known expectation of \(Y\) is the analytic Black--Scholes price
\(V_{\mathrm{BS}}\). The coefficient is estimated from antithetic pair
averages on the policy-training stream and then frozen before the independent
valuation stream:

\[
\widehat\beta_{\mathrm{train}}
=\frac{\widehat{\operatorname{Cov}}_{\mathrm{train}}(X,Y)}
       {\widehat{\operatorname{Var}}_{\mathrm{train}}(Y)},
\qquad
Z=X-\widehat\beta_{\mathrm{train}}
       \left(Y-V_{\mathrm{BS}}\right).
\]

Although the same training paths also fit the policy, no coefficient is
estimated on the valuation sample. Conditional on the frozen policy and
coefficient, \(\mathbb E[Z]=\mathbb E[X]\). The report retains raw and adjusted
prices and standard errors, the fitted coefficient, the European Monte Carlo
and analytic prices, and the realized out-of-sample variance-reduction ratio.

### Applicability of the reported ratio

The variance-reduction ratio is \(\widehat{\operatorname{Var}}(X)/
\widehat{\operatorname{Var}}(Z)\). Two degenerate cases must not be read the
same way, so applicability is reported explicitly:

- If the training policy chooses **immediate exercise**, the payoff is already
  deterministic. No valuation simulation runs, the engine uses coefficient
  zero, and both variances are exactly zero. The ratio is the undefined form
  \(0/0\). It is marked `variance_reduction_applicable = false`, is emitted as
  JSON `null`, and is **excluded from every variance-reduction summary**: such
  a case can neither set nor lower a reported minimum. A placeholder value here
  would silently become the reported worst case and hide the genuine one.
- If an **applicable** control variate removes all variance --- as structural
  suppression does, where \(X\equiv Y\) path by path --- the ratio is a
  positive infinity. That is a measurement, not an undefined form. It is
  counted in `variance_reduction_infinite_cases`, emitted as JSON `null`, and
  excluded from the finite minimum.

`minimum_variance_reduction_ratio` therefore means: the minimum over applicable
finite stochastic cases.

The same applies to the European Monte Carlo fields. An immediate-exercise
policy draws no valuation paths, so no European Monte Carlo observation exists.
`european_monte_carlo_sampled` is false and both sampled fields are emitted as
`null` rather than being filled with the closed-form value, which would present
an analytic price as an observation. The separate analytic European price is
always reported.

## Memory and complexity

Policy fitting stores double-precision log spots and is
\(O(P_{\mathrm{train}}M)\) in memory and roughly
\(O(P_{\mathrm{train}}Md)\) in arithmetic. Valuation streams each antithetic
pair and uses \(O(M)\) memory.

The engine computes a deterministic estimate of all bulk training storage
exactly once, at the public entry point: the path matrix, cash flows, stopping
indices, maximum-size regression vectors and QR columns, and accumulated policy
coefficients. The limit is enforced against that single value before any bulk
allocation, and the same value is what the result reports, so the Python
preflight and the engine can never disagree. Allocator metadata and
implementation-specific `std::vector` object overhead are excluded. The
estimate deliberately carries one extra double per training path as a
conservative margin; it is an upper bound, not a tight one.

## Price-domain overflow rejection

The shared `VanillaOptionInput` validator admits **every** finite rate,
dividend yield and volatility. That boundary is also used by Black--Scholes and
CRR, so tightening it is a separate compatibility decision and is not made
here. The LSM engine is instead made safe for every input that validator
currently admits.

A finite log spot is not sufficient. The simulated state is a random walk in
\(\log S\), and \(\exp(x)\) leaves double precision near \(x\approx709.78\)
while \(x\) itself remains far inside the representable range. A drift large
enough to push the walk past that bound produced an infinite spot, while the
matching discount factor underflowed to exactly zero; their product was the
IEEE indeterminate form \(\infty\times0=\mathrm{NaN}\), which then propagated
silently through the online moments into a returned "price". A large negative
rate overflowed the discount factor directly.

The engine now validates every price-domain quantity where it is produced ---
drift and diffusion, each exponentiated spot, every discount factor, intrinsic
values, discounted cash flows, regression responses and constant fallbacks,
control-variate observations, antithetic pair averages, online-moment inputs
and accumulators, and every reported price, standard error and interval bound
--- and throws `std::overflow_error` before the value can reach a regression or
a statistic. Exponent expressions are passed through unchanged, so the checks
do not alter any arithmetic.

The contract is therefore total: for any input the shared validator admits,
`least_squares_monte_carlo` either returns fully finite results or throws. It
never returns NaN, an infinity, or a corrupted statistic. Extreme but genuinely
representable inputs are still priced; only inputs that leave double precision
are rejected.

## Cross-check protocol

`configs/american_lsm_crosscheck_v1.toml` pins:

- nine named regimes from the CRR convergence study by file SHA-256;
- the high-step 8,192/8,193 CRR adjacent average;
- one-factor-at-a-time path-count, exercise-grid, and basis-degree experiments;
- an explicit primary experiment;
- a base seed and deterministic per-case/per-experiment stream derivation;
- a 512 MiB policy-training working-set ceiling.

Run it after rebuilding the binding:

```bash
python -m differentiable_pricing.american.lsm_crosscheck \
  --config configs/american_lsm_crosscheck_v1.toml \
  --output artifacts/american-lsm-crosscheck-v1.json
```

The report is deterministic for the recorded compiler/runtime toolchain and
contains no timing fields. Its CRR comparison is an independent-algorithm
cross-check within one model, not a comparison with market prices and not a
formal proof that either discretization is exact.

The strongest regression signal against discount-time and stopping-index errors
is not in this report: it is the early-exercise-premium cross-check in the C++
test suite, which compares \(V_{\mathrm{LSM}}-V_{\mathrm{BS}}\) against the CRR
American-minus-European premium. Subtracting the shared European component
cancels most of the diffusion noise, so a mis-discounted exercise cash flow
shows up first order. Run the C++ tests as well as the study when changing the
recursion.

## Expected behaviour in the observed study

The following are consequences of the method, not defects, and must not be
"fixed" by moving thresholds or redesigning the experiment.

**LSM sits below CRR.** The value of a fixed, learned policy on a coarse
Bermudan grid is a lower bound for the same-grid optimal value, which is itself
a lower bound for continuous exercise. The finer-grid CRR average therefore
exceeds it, and every observed gap has that sign. Because the gap is systematic
while the standard error shrinks at \(n^{-1/2}\), increasing valuation paths
tightens the interval around the gap rather than closing it, which is exactly
why the stochastic containment count falls as paths increase.

**Degree one is inadequate here.** A linear continuation basis cannot represent
the curvature of the continuation value near the exercise boundary. In the
recorded study its mean gap is roughly twenty times the quadratic reference's
and far larger than any standard error, so this is a demonstrated deficiency
rather than sampling noise.

**Quadratic is the defensible current basis, not a proven optimum.** Degree
three does not improve on degree two at the current path counts; the two are
indistinguishable in the recorded evidence, while the cubic basis estimates one
more coefficient per regression. Quadratic is the reasonable default pending a
joint path/basis study. Nothing here proves it optimal.

**More exercise dates need more policy-training paths.** Doubling the exercise
grid at a fixed path budget spreads the same training sample across twice as
many regression dates. The in-the-money sample at each date thins, the
constant-fallback count rises, and the fitted policy gets worse: the mean gap
widens rather than narrowing. This is the path/basis/grid trade-off of
[Glasserman and Yu (2004)](https://arxiv.org/pdf/math/0503556), and it means an
exercise-grid refinement is only meaningful alongside a path increase.

**Deep out-of-the-money contracts fall back often.** With few or no
in-the-money training paths at early dates, the constant fallback is the
correct response to an ill-posed regression, and it is reported per date rather
than hidden. Note that `minimum_relative_r_diagonal` is a worst-column
conditioning indicator relative to the constant column: it is at most one by
construction, and fallback rows report zero rather than a fitted value.
