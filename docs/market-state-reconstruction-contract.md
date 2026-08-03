# Market-state identifiability and reconstruction (task 9B)

## Purpose and scope

This study answers one question: which of the economic inputs a
discrete-dividend American PDE pricer needs can be recovered from the
three-session `market-feasibility-v1` archive, which are only jointly
identifiable, and which require external data or an explicit assumption.

Its deliverable is the **input contract for task 9C**. It is not a pricer.

Explicitly out of scope, and absent from every artefact this study produces:

- solving any PDE;
- generating synthetic labels;
- training any network;
- computing any Greek;
- inverting any implied volatility;
- fitting any volatility surface.

The study reuses task 9A's normalized Parquet and provenance. It builds no
second ingestion path and no general optimizer.

## Units and conventions

- Strikes and premiums are USD per share, exactly as quoted. **No contract
  multiplier is applied anywhere.** Put-call parity in per-share units is
  multiplier-free, and the archive's OPRA definition records leave the
  multiplier undefined, so assuming one would be an unchecked constant.
- Timestamps are UTC internally. Snapshot and expiry instants are declared in
  exchange-local wall-clock time and converted per date through a real time
  zone, so a daylight-saving transition moves the instant instead of silently
  misaligning it.
- Year fractions are ACT/365 fixed between two *instants*:

  $$
  \tau = \frac{t_{\mathrm{expiry}} - t_{\mathrm{snapshot}}}{365 \times 86400},
  $$

  with both in seconds. This is a calendar-day convention, not a business-day
  one, and not a money-market ACT/360.
- One-minute schemas stamp each interval at its close, so a snapshot labelled
  `10:30` covers `(10:29, 10:30]`. It is an interval-close observation, not an
  instantaneous one.

## European reconstruction

For a European call and put on the same underlying with the same strike $K$
and expiry $T$, quoted in the **same minute**,

$$
y(K,T) = C(K,T) - P(K,T) = D(T)\,[F(T) - K] = a(T) + b(T)\,K,
$$

so a two-parameter regression of $y$ on $K$ identifies the whole state at that
expiry:

$$
D(T) = -b(T), \qquad F(T) = \frac{a(T)}{D(T)}, \qquad
r(T) = -\frac{\log D(T)}{\tau}.
$$

### The slope sign is mandatory

$b = -D$, and a discount factor is positive, so a fitted slope that is not
strictly negative means the identity did not hold on that data. It is reported
under its own name and the offending slope is published as computed. No
absolute value is taken anywhere: clamping a negative discount factor to a
small positive number converts a detected contradiction into an undetectable
one.

### Executable bounds, not error bars

A quote is an interval. The combination could have been transacted anywhere in

$$
y_{\mathrm{lower}} = C_{\mathrm{bid}} - P_{\mathrm{ask}}, \qquad
y_{\mathrm{upper}} = C_{\mathrm{ask}} - P_{\mathrm{bid}},
$$

and for a fitted $\hat{y}$ the outside-spread error is

$$
e(\hat{y}) =
\begin{cases}
0, & y_{\mathrm{lower}} \le \hat{y} \le y_{\mathrm{upper}},\\
y_{\mathrm{lower}} - \hat{y}, & \hat{y} < y_{\mathrm{lower}},\\
\hat{y} - y_{\mathrm{upper}}, & \hat{y} > y_{\mathrm{upper}}.
\end{cases}
$$

This is a statement about executability, not about statistical significance.
**A bid-ask width is never treated as a standard error.** No confidence
interval, standard error or p-value is derived anywhere in this study.

### Pairing

Only standard-root contracts that are tradable in the *same minute* with an
identical expiry and an identical strike are paired. A call from one minute and
a put from the next are not a pair: parity is an identity between simultaneous
prices. Duplicate legs are rejected rather than silently resolved, so the
result cannot depend on input order. Tradability is task 9A's `tradable_core`:
finite positive sizes on both sides, a positive bid, a strictly positive
spread, a book graded `ok` or `stale`, and no vendor flag.

### Fitting

Two deterministic closed-form methods are reported side by side:

- unweighted least squares;
- weighted least squares with $w_i = 1/\Delta_i^2$, where
  $\Delta_i = y_{\mathrm{upper},i} - y_{\mathrm{lower},i}$ is the combined call
  and put bid-ask width.

The weight is an executable-liquidity measure. It is deliberately not described
as an independent statistical standard error, no distributional assumption is
attached to it, and nothing derived from it is an interval estimate.

Both solve in strike-centered coordinates and map back, so the reported
intercept is exact for the uncentered model while the arithmetic never forms
the badly scaled raw normal equations. Two condition numbers are reported: the
2-norm condition number of the raw weighted design $[\,1\;K\,]$, which is what
a naive solve would face, and the same quantity for the centered design
actually used. The small eigenvalue is obtained from
$\det = S_w S_{xx}$ rather than from $S_w S_{kk} - S_k^2$, which cancels to
nothing when strikes sit far from the origin — the regime every listed chain is
in.

### Degeneracy is reported, never clamped

Each of these is a distinct fact with a distinct remedy and gets its own name:
insufficient pairs, a singular design, a poorly conditioned design, a
non-finite solution, a non-negative slope, a non-positive discount factor, a
non-positive forward, a non-positive weight. Ill conditioning is a **warning**
— the fit is still published, flagged, and left to the reader. The others
invalidate.

### Strike windows

The forward anchor is the paired strike minimizing $|C_{\mathrm{mid}} -
P_{\mathrm{mid}}|$, which is the strike nearest the forward and needs no
discount factor, dividend or spot to find. Three relative half-widths about
that anchor are reported for every expiry, snapshot and method. **The spread
across them is a result, not an error bar to be minimized, and no window may be
selected after seeing the results.**

### Rate interpretation

$r = -\log D / \tau$ is withheld for zero-DTE expiries and for any $\tau$ below
a declared floor. $D$ and $F$ are still fitted and published for them. The
reason is arithmetic: at a $\tau$ of hours the smallest representable change in
a quoted premium moves $-\log D/\tau$ by whole percentage points, so the
resulting number describes the quote grid rather than the funding market.

### Expiry-time assumption

The archive's OPRA definition records carry an expiration **date** stamped at
midnight UTC and no settlement metadata: no settlement time, no AM/PM
settlement flag and no exercise style. A fixed local expiry instant is
therefore **assumed** and every $\tau$ in the report is labelled with that
assumption. It is wrong for AM-settled expiries and an approximation for
PM-settled ones. The induced error in $\tau$ is at most a few hours, which is
negligible at six months and material at one day — which is exactly why short
expiries are excluded from rate interpretation.

## The SPY American diagnostic

European parity is **not** an identity for American options. Given $D(T)$ from
an exactly matched XSP expiry, and American midpoints $C_A$, $P_A$, the study
defines only

$$
\tilde{F}_i(T) = \frac{C_A - P_A + D(T)\,K_i}{D(T)}, \qquad
\tilde{Q}(T) = S_{\mathrm{SPY}} - D(T)\,\mathrm{median}_i\,\tilde{F}_i(T).
$$

$\tilde{Q}$ is called the **American parity carry residual**. It is never
called:

- an observed dividend;
- an exact dividend present value;
- a borrow rate;
- an exact SPY forward;
- an independent market input.

Those five names are declared in the configuration and the parser rejects a
configuration that drops any of them.

It contains, inseparably: the present value of dividends expected before
expiry; the early-exercise premia of the American call and put, which do not
cancel; quote noise on four legs; the error in a partial-venue spot; and any
contract difference between the SPY and XSP listings that share an expiration
date. This archive separates none of them.

The median across strikes is used precisely so one exercised leg cannot move
the level.

### Detecting early-exercise contamination

Under exact European parity $\tilde{F}$ is constant in $K$, so a systematic
slope against the strike is contamination or misspecification, not noise.
Measuring that slope inside the window the discount factor was fitted on is
useless: least-squares orthogonality makes the European slope zero by
construction. The study therefore uses an **out-of-window control**: $D$ from
the narrowest window is applied to pairs in the widest, identically for SPY and
for the European XSP chain. The two slopes are then comparable, and the finding
is reported as a ratio between two measured quantities rather than against an
invented absolute level.

### Ex-dates

The archive holds **no dividend data of its own** — task 9A's
`discrete_dividend_schedule` probe finds the declared path present but empty,
and this study re-probes it. The scheduled dates come instead from the issuer's
published calendar, cited in the configuration:

- **Source:** *SPDR Dividend and Capital Gain Distribution Schedule*, State
  Street Global Advisors,
  <https://www.ssga.com/library-content/products/fund-data/etfs/us/distribution/SPDR_Dividend_Distribution_Schedule.pdf>
  (the table's quarterly rows apply to both SPY and MDY).
- **Ex-date:** 2026-06-18 · **Record date:** 2026-06-18 · **Payable date:**
  2026-07-31.

The status is `officially_scheduled`, and that label is deliberately narrow. It
establishes **the scheduled dates only**. It carries no cash amount, so nothing
here may describe a dividend amount as observed, inferred or validated. It
measures nothing about the options market either: whether, when and by how much
a scheduled distribution is reflected in quoted prices is a market question this
three-session archive cannot answer. The parser enforces both limits by
requiring `amount_status` and `economic_effect_status` to stay `not_verified`.

The dates enter no arithmetic. They label which cross-session transition
contains the scheduled event; the comparison itself is a plain cross-session
difference in $\tilde{Q}$ at a fixed expiry. Where the largest step falls in
that transition, it is reported as **temporally aligned with, and qualitatively
consistent with, the scheduled ex-dividend event** — an alignment in time and
sign, and nothing more. It does not infer or validate the cash dividend amount,
and it does not separate the residual into dividend, borrow, early-exercise,
ETF/index-basis, and quote and fitting components, none of which this archive
separates. **The schedule is never adjusted to fit a result.**

## Rate controls

Comparison against SOFR, the DGS constant-maturity Treasury series, and
outright SR3 settlement-implied rates ($100 - \mathrm{settlement}$) is
**descriptive only**.

- A Treasury constant-maturity yield is a par yield on government credit. It is
  not an OIS zero rate: different credit, different collateral, different
  compounding, and a par quotation rather than a zero one.
- An SR3 settlement implies a futures rate on compounded SOFR over a forward
  three-month accrual period. It is not a zero-coupon yield to that period's
  end, and the futures/forward convexity difference is not modelled.

No accrual schedule, business-day convention, compounding convention or
convexity adjustment is implemented, so **no artefact of this study is a
bootstrapped OIS curve** and none may be described as one.

## Provenance

Three digests are verified before any arithmetic runs, and a mismatch aborts
the run with nothing written:

1. task 9A's configuration file, cross-checked against the digest 9A recorded
   for itself;
2. task 9A's feasibility report;
3. the raw archive's SHA-256 manifest, cross-checked against the digest 9A
   recorded for it.

The processed root, schema version and audit version the 9A report declares are
checked too. The declared dividend, borrow and corporate-action source paths
are **re-probed** at the moment of use rather than quoted from an older report,
because their state is the most consequential fact about SPY identifiability.
The probe distinguishes `absent`, `present_but_empty` and `present`: a SHA-256
manifest lists files, so an empty directory is invisible to it.

## Identifiability classification

Each PDE input is classified as one of:

| Class | Meaning |
| --- | --- |
| `directly_observed` | read from a field of the archive with no model between the file and the number |
| `robustly_inferred` | recovered by an estimator this study ran, whose stability over the axes it was run on was measured and met stated criteria |
| `pointwise_identified_curve_unconstructed` | identified on its own at the discrete points the data quotes, but the exploratory stability bar was missed and/or no continuous curve through those knots was built |
| `jointly_identifiable_only` | appears in the data only inside a sum or product with another unknown, so no amount of this data separates it |
| `external_convention` | not in the data at all; supplied by a rule the study declares |
| `unavailable` | needed, not in the data, and not supplyable by convention |

`pointwise_identified_curve_unconstructed` exists because the alternative would
be a false statement. Put-call parity determines $D(T)$ and $F(T)$ at each
quoted expiry **on their own**, from an over-determined set of same-minute
pairs, with no second unknown to be confounded with — unlike borrow and
dividends, which are genuinely joint and keep `jointly_identifiable_only`. A
missed stability bar bounds how precisely those knots are pinned; it cannot
revoke their identifiability. The exploratory `5e-4` threshold may therefore
change the **stability** statement and never the **pointwise identifiability**
statement. The class also records the second, independent fact that this study
builds knots and no interpolation through them, so no continuous discount or
forward curve exists yet; that interpolation rule is a modelling choice task 9C
must make.

The remaining verdicts are **logical**: they follow from what a field is,
not from how large a number came out. Only `robustly_inferred` consults
measured numbers, through two thresholds — a bid-ask containment floor and a
ceiling on the spread in $D$ across windows and methods — that are declared
with their provenance as exploratory criteria chosen after this archive was
observed. Every quantity they are compared against is reported beside them, so
the classification can be re-derived under a different criterion without
rerunning the study.

The stability criterion is applied to the spread **at a fixed instant**, across
strike windows and fit methods. The spread across the three intraday snapshots
is reported separately, and deliberately not used as the criterion: it is not
the estimator disagreeing with itself at one instant, but it is not one thing
either. It may combine genuine market movement, quote microstructure, a change
in which strikes pass the tradability and window filters at each snapshot, and
fitting variation. This study reports it and does not decompose it.

## Proposed task 9C input contract

Specified here, **not implemented**. Task 9C owns the solver; the point of
stating the interface now is that each field's real-market provenance is
decided by the identifiability matrix, so the interface cannot quietly acquire
a field no data can fill.

| Field | Type | Real-market provenance |
| --- | --- | --- |
| `spot` | `double` | directly observed |
| `strike` | `double` | directly observed |
| `option_type` | `enum class OptionType` | directly observed |
| `valuation_time` | `double` (years) | directly observed |
| `expiry_time` | `double` (years) | date observed; instant **assumed** |
| `discount_curve` | `PiecewiseDiscountCurve` | reconstructed at listed expiries |
| `dividends` | `CashDividendSchedule` | **unavailable** |
| `continuous_carry` | `std::optional<double>` | **jointly identifiable only** |
| `volatility` | `double` | model-implied; not produced here |
| `exercise_style` | `enum class ExerciseStyle` | external convention |
| `settlement` | `SettlementConvention` | external convention |
| `contract_multiplier` | `double` | external convention |

Design points that are part of the contract rather than of an implementation:

- The discount curve is carried as times and **log-discounts**, evaluated as
  the exponential of an interpolation in log-discount space, so positivity
  holds by construction and a zero rate is a slope rather than a quotient. The
  interpolation rule is versioned with the contract.
- Curve nodes must bracket `[valuation_time, expiry_time]` so no extrapolation
  is ever silent.
- `continuous_carry` is `std::optional` on purpose: a caller with no borrow
  information must be able to say so, rather than pass a zero that reads as an
  observation.
- An empty dividend schedule must be stated explicitly by the caller, not
  inferred from a default-constructed vector. It is a meaningfully different
  contract, not a default.
- `contract_multiplier` is carried for reporting contract-level cash only. The
  solver works in per-share units and the multiplier must not enter the pricing
  arithmetic.

Fields are marked in the generated report as synthetic training inputs,
must-be-reconstructed, or both.

## Running it

The study reads task 9A's processed Parquet, which is proprietary and
git-ignored, so it runs locally and never in CI. Its tests use synthetic
fixtures only and run everywhere.

```bash
python -m differentiable_pricing.market.reconstruct \
  --config configs/market_state_reconstruction_v1.toml
```

Outputs — a JSON report, two CSV tables and SVG figures — are written
atomically beneath the configured, git-ignored root. **Every one of them is
derived from proprietary quote-level data.** Fitted market curves, inferred
market values, reconstructed quotes and generated figures must never be staged.
Only code, configuration, tests and this document may appear in Git.

Two output guarantees, which are not the same guarantee:

- Each artefact is replaced atomically, and the whole study completes — every
  table and figure rendered — before anything is written, so a computation
  failure leaves the previous outputs entirely untouched.
- The artefact *set* is not transactional. Each file is a separate rename, so an
  I/O failure part-way through publishing can leave some artefacts refreshed and
  others stale. The JSON report is therefore published **last**: a stale report
  beside fresh tables reads as an interrupted run, whereas a fresh report beside
  stale tables would read as a complete one.

## Non-claims

- Three sessions, two of them consecutive. Nothing here is a distributional
  claim.
- Every setting is exploratory and was chosen after the archive was observed.
  None is a predeclared replication setting and none gates a replication
  partition.
- The expiry instant is assumed; the archive carries no settlement metadata.
- Bid-ask widths are executable-liquidity measures, never standard errors.
- The carry residual is not a dividend, not a dividend present value, not a
  borrow rate, not a SPY forward and not an independent market input.
- No dividend is described as observed or inferred anywhere.
- The reconstructed forward is the **cash-settled index contract's** forward.
  It is not the SPY forward, and the ETF/index basis is not measured here.
- No OIS curve is bootstrapped.
- The underlying feed is a partial-venue consolidation, so its half-spread is a
  lower bound on the spot's error, not a measurement of it.
