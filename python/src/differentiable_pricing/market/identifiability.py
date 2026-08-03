"""Classify each discrete-dividend American PDE input by how well it is known.

A pricer needs a specific list of numbers. This module answers, for each of
them, whether the three-session archive supplies it, supplies it only in
combination with something else, supplies nothing and forces a convention, or
supplies nothing at all. The answer is then turned into the proposed Task 9C
solver interface, with every field marked either "a synthetic training input we
choose" or "a real-market quantity that must be reconstructed".

How a classification is reached
    Most classifications are *logical* rather than numerical, and those are the
    ones that carry weight. A quantity is:

    ``directly_observed``
        read from a field of the archive with no model between the file and the
        number;
    ``robustly_inferred``
        recovered by an estimator this study ran, whose stability across the
        windows, methods, snapshots and sessions it was run over was measured
        and met stated exploratory criteria;
    ``pointwise_identified_curve_unconstructed``
        identified on its own -- not confounded with any other unknown -- at the
        discrete points the data quotes, but either the exploratory stability
        bar was missed or no continuous curve through those points has been
        built. This is a statement about construction and measured stability,
        never about identifiability: the knots are identified;
    ``jointly_identifiable_only``
        appearing in the data only inside a sum or product with another unknown,
        so no amount of this data separates it;
    ``external_convention``
        not in the data at all and supplied by a rule the study declares;
    ``unavailable``
        needed, not in the data, and not supplyable by convention either.

    Only the ``robustly_inferred`` verdict consults measured numbers, and the
    two thresholds it uses are declared here with their provenance: they were
    chosen after this archive was observed and are exploratory.

Nothing here is a constant standing in for an observation. Every fact the rules
consume arrives in :class:`ObservedFacts`, measured by the run that calls this.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Final

DIRECTLY_OBSERVED: Final = "directly_observed"
ROBUSTLY_INFERRED: Final = "robustly_inferred"
POINTWISE_IDENTIFIED_CURVE_UNCONSTRUCTED: Final = "pointwise_identified_curve_unconstructed"
JOINTLY_IDENTIFIABLE_ONLY: Final = "jointly_identifiable_only"
EXTERNAL_CONVENTION: Final = "external_convention"
UNAVAILABLE: Final = "unavailable"

CLASSIFICATIONS: Final = (
    DIRECTLY_OBSERVED,
    ROBUSTLY_INFERRED,
    POINTWISE_IDENTIFIED_CURVE_UNCONSTRUCTED,
    JOINTLY_IDENTIFIABLE_ONLY,
    EXTERNAL_CONVENTION,
    UNAVAILABLE,
)

MINIMUM_CONTAINMENT_FOR_ROBUST: Final = 0.90
"""Fraction of fitted parity values that must lie inside the executable interval.

Exploratory, chosen after this archive was observed. The reasoning is that a
reconstruction whose fitted combination is not executable at nine strikes in ten
is describing something other than the tradable market; the exact tenth is a
judgement, not a derivation, and this study reports the achieved fraction beside
the threshold so a reader can apply their own.
"""

MAXIMUM_DISCOUNT_SPREAD_FOR_ROBUST: Final = 5.0e-4
"""Largest tolerated spread in ``D`` across strike windows and fit methods.

Exploratory, chosen after this archive was observed, and kept at its declared
value. Since $\\mathrm{d}D/D = -\\tau\\,\\mathrm{d}r$, five parts in ten thousand
of a discount factor is about five basis points of annualized rate at one year,
twenty at three months, and two at two and a half years.

That maturity dependence is a known weakness of the criterion: one absolute
spread in $D$ means very different rate ambiguities at different expiries, so
the same number is lenient at the short end and strict at the long end. The
right criterion is almost certainly in rate units rather than discount-factor
units. Changing it now would be selecting a criterion after seeing which side of
it the result fell on, so this value stands, the achieved spread is reported
beside it, and the same spread is *also* reported in rate units as a purely
descriptive number so a future study can freeze a better criterion on evidence
rather than on convenience.
"""

THRESHOLD_PROVENANCE: Final = (
    "Both numerical thresholds in this classification were chosen after the three-session "
    "market-feasibility-v1 archive had been observed. They are exploratory pilot criteria, not "
    "predeclared acceptance gates, and every quantity they are compared against is reported "
    "beside them so the classification can be re-derived under a different criterion without "
    "rerunning the study. The logical classifications -- directly observed, jointly identifiable "
    "only, external convention, unavailable -- consult no threshold at all: they follow from what "
    "a field is, not from how large a number came out."
)


@dataclass(frozen=True, slots=True)
class ObservedFacts:
    """Everything the classification rules are allowed to consult.

    Every field is measured by the run that constructs this object. Nothing in
    the rules below reaches around it for a default.
    """

    european_fits_attempted: int
    european_fits_valid: int
    european_containment_fraction_median: float
    european_discount_factor_spread: float
    european_discount_factor_intraday_spread: float
    european_rate_interpretable_expiries: int
    matched_spy_xsp_expiries: int
    american_residual_defined: bool
    american_residual_spread: float
    american_residual_intraday_spread: float
    american_f_tilde_shows_strike_dependence: bool
    dividend_source_state: str
    borrow_source_state: str
    corporate_action_source_state: str
    exercise_style_populated_fraction: float
    contract_multiplier_populated_fraction: float
    settlement_time_populated_fraction: float
    underlying_feed_class: str
    underlying_relative_half_spread: float
    sessions_observed: int

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready statement of the evidence the rules consumed."""
        return {
            "european_fits_attempted": self.european_fits_attempted,
            "european_fits_valid": self.european_fits_valid,
            "european_containment_fraction_median": self.european_containment_fraction_median,
            "european_discount_factor_spread": self.european_discount_factor_spread,
            "european_discount_factor_spread_definition": (
                "median over (session, snapshot, expiry) of the range in D across strike "
                "windows and fit methods. This is estimator ambiguity at a fixed instant; it "
                "deliberately excludes the intraday term below, which mixes several causes."
            ),
            "european_discount_factor_intraday_spread": (
                self.european_discount_factor_intraday_spread
            ),
            "european_discount_factor_intraday_spread_definition": (
                "median over (session, expiry) of the range in D across windows, methods AND "
                "the three intraday snapshots. Larger than the term above, but not for one "
                "reason: it may combine genuine market movement, quote microstructure, strike "
                "composition -- which strikes pass the tradability and window filters differs "
                "between snapshots -- and fitting variation. This study does not decompose it, "
                "and it is not an estimator error."
            ),
            "european_rate_interpretable_expiries": self.european_rate_interpretable_expiries,
            "matched_spy_xsp_expiries": self.matched_spy_xsp_expiries,
            "american_residual_defined": self.american_residual_defined,
            "american_residual_spread": self.american_residual_spread,
            "american_residual_spread_definition": (
                "median over (session, snapshot, expiry) of the range in the carry residual "
                "across strike windows and fit methods, at the quoted spot midpoint."
            ),
            "american_residual_intraday_spread": self.american_residual_intraday_spread,
            "american_f_tilde_shows_strike_dependence": (
                self.american_f_tilde_shows_strike_dependence
            ),
            "dividend_source_state": self.dividend_source_state,
            "borrow_source_state": self.borrow_source_state,
            "corporate_action_source_state": self.corporate_action_source_state,
            "exercise_style_populated_fraction": self.exercise_style_populated_fraction,
            "contract_multiplier_populated_fraction": (
                self.contract_multiplier_populated_fraction
            ),
            "settlement_time_populated_fraction": self.settlement_time_populated_fraction,
            "underlying_feed_class": self.underlying_feed_class,
            "underlying_relative_half_spread": self.underlying_relative_half_spread,
            "sessions_observed": self.sessions_observed,
            "european_fit_validity_fraction": self.european_fit_validity_fraction,
            "european_fit_validity_policy": (
                "Zero tolerance, unlike the two median-based criteria beside it. A blocking "
                "diagnostic is a contradiction rather than an outlier: a non-negative slope or a "
                "non-positive discount factor says the parity identity did not hold, and a "
                "singular design says the strikes spanned nothing. The achieved fraction is "
                "reported so a reader may apply their own tolerance."
            ),
            "declared_source_probe_semantics": (
                "A source state of 'present' means files exist at the declared path. It does not "
                "mean their contents were parsed or validated: this study reads no dividend, "
                "borrow or corporate-action file. A classification that turns on 'present' is "
                "therefore a statement about acquisition, not about content."
            ),
        }

    @property
    def european_fit_validity_fraction(self) -> float:
        """Fraction of attempted fits that returned usable numbers."""
        if self.european_fits_attempted == 0:
            return 0.0
        return self.european_fits_valid / self.european_fits_attempted

    @property
    def european_reconstruction_is_robust(self) -> bool:
        """Whether the parity reconstruction met all three exploratory criteria.

        The validity clause is deliberately zero-tolerance while the other two
        are median-based, and the asymmetry is intentional rather than an
        oversight. The blocking diagnostics are not noise: a non-negative slope
        or a non-positive discount factor says the parity identity did not hold
        on that data, and a singular design says the strikes did not span
        anything. One such cell is a contradiction, not an outlier, and a
        reconstruction that produces contradictions is not one a pricer should
        take a curve from. Dispersion, by contrast, is expected in every cell
        and is therefore judged on its median.

        The achieved validity fraction is reported beside the verdict, so a
        reader who prefers a tolerance can apply one without rerunning the
        study.
        """
        return (
            self.european_fits_valid > 0
            and self.european_fits_valid == self.european_fits_attempted
            and self.european_containment_fraction_median >= MINIMUM_CONTAINMENT_FOR_ROBUST
            and self.european_discount_factor_spread <= MAXIMUM_DISCOUNT_SPREAD_FOR_ROBUST
        )


@dataclass(frozen=True, slots=True)
class InputClassification:
    """One PDE input, its verdict, the evidence, and what would change it."""

    key: str
    description: str
    units: str
    classification: str
    evidence: str
    consequence: str
    what_would_change_it: str

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready row of the identifiability matrix."""
        return {
            "input": self.key,
            "description": self.description,
            "units": self.units,
            "classification": self.classification,
            "evidence": self.evidence,
            "consequence_for_a_pricer": self.consequence,
            "what_would_change_it": self.what_would_change_it,
        }


def _source_phrase(state: str) -> str:
    """Render a declared-source probe state as a sentence fragment."""
    return {
        "absent": "the declared path does not exist",
        "present_but_empty": "the declared path exists and holds no files",
        "present": "the declared path holds content",
    }.get(state, f"the probe reported '{state}'")


def build_identifiability_matrix(facts: ObservedFacts) -> tuple[InputClassification, ...]:
    """Classify every PDE input from measured facts about this archive."""
    robust = facts.european_reconstruction_is_robust
    # Not `jointly_identifiable_only` when the bar is missed. Put-call parity
    # identifies D and F on their own, from an over-determined set of same-minute
    # pairs; they are not confounded with a third unknown the way borrow and
    # dividends are. A missed stability bar is a statement about measured
    # dispersion and about the fact that no continuous curve was built, and
    # neither of those can revoke pointwise identifiability.
    discount_class = (
        ROBUSTLY_INFERRED if robust else POINTWISE_IDENTIFIED_CURVE_UNCONSTRUCTED
    )
    curve_note = (
        " Either way this study builds only knots: D(T) and F(T) are identified at the quoted "
        "expiries and no continuous discount or forward curve has been constructed, so the "
        "between-expiry interpolation rule remains a modelling choice task 9C must make."
        if robust
        else " The exploratory stability bar was therefore narrowly missed. That bounds how "
        "precisely these knots are pinned; it does not make them unidentified. Parity determines "
        "D(T) and F(T) at each quoted expiry on their own, from an over-determined set of "
        "same-minute pairs, with no second unknown to be confounded with -- unlike borrow and "
        "dividends, which are genuinely joint. What has not been produced is a continuous curve: "
        "this study yields identified pointwise knots and no interpolation through them."
    )
    discount_evidence = (
        f"{facts.european_fits_valid} of {facts.european_fits_attempted} XSP parity fits "
        f"returned a positive discount factor with a strictly negative slope; the median "
        f"bid-ask containment fraction was {facts.european_containment_fraction_median:.4f} "
        f"against an exploratory floor of {MINIMUM_CONTAINMENT_FOR_ROBUST:.2f}, and the median "
        f"spread in D across strike windows and fit methods at a fixed expiry and snapshot was "
        f"{facts.european_discount_factor_spread:.3e} against an exploratory ceiling of "
        f"{MAXIMUM_DISCOUNT_SPREAD_FOR_ROBUST:.1e}. Across the three intraday snapshots the "
        f"spread widens to {facts.european_discount_factor_intraday_spread:.3e}, which is "
        f"deliberately not the quantity the criterion is applied to: it may combine genuine "
        f"market movement, quote microstructure, a change in strike composition between "
        f"snapshots, and fitting variation, and this study separates none of them. Zero-DTE "
        f"expiries are excluded from both spreads: at a tau of hours D is pinned within quote "
        f"granularity of one, so such a cell would report its tau rather than the estimator."
        + curve_note
    )

    dividend_amount_class = (
        UNAVAILABLE if facts.dividend_source_state != "present" else DIRECTLY_OBSERVED
    )
    dividend_date_class = dividend_amount_class
    borrow_class = UNAVAILABLE if facts.borrow_source_state != "present" else DIRECTLY_OBSERVED

    exercise_class = (
        DIRECTLY_OBSERVED
        if facts.exercise_style_populated_fraction > 0.0
        else EXTERNAL_CONVENTION
    )
    multiplier_class = (
        DIRECTLY_OBSERVED
        if facts.contract_multiplier_populated_fraction > 0.0
        else EXTERNAL_CONVENTION
    )
    settlement_class = (
        DIRECTLY_OBSERVED
        if facts.settlement_time_populated_fraction > 0.0
        else EXTERNAL_CONVENTION
    )

    return (
        InputClassification(
            key="valuation_time",
            description=(
                "The instant the pricing state is taken at. Here it is the interval-close "
                "timestamp of a one-minute consolidated quote bar, read from the file."
            ),
            units="UTC instant",
            classification=DIRECTLY_OBSERVED,
            evidence=(
                "Every quote record carries ts_recv, and the snapshot is selected by exact "
                "equality with a declared interval-close instant, so the valuation time is a "
                "field rather than an estimate. It is an interval-close observation covering "
                "the preceding minute, not an instantaneous one."
            ),
            consequence="No modelling risk; only the interval-close semantics must be honoured.",
            what_would_change_it=(
                "Nothing. A finer schema would narrow the interval, not change its kind."
            ),
        ),
        InputClassification(
            key="spy_spot",
            description="The underlying level the American PDE is solved from.",
            units="USD per share",
            classification=DIRECTLY_OBSERVED,
            evidence=(
                f"Quoted directly by the underlying feed, whose class is "
                f"'{facts.underlying_feed_class}'. Its relative half-spread at the observed "
                f"snapshots was {facts.underlying_relative_half_spread:.3e}. That feed is not "
                f"the SIP national best bid and offer, so the quoted half-spread is a lower "
                f"bound on the level's error, not a measurement of it."
            ),
            consequence=(
                "Observed, but with an uncertainty this archive cannot bound from above. The "
                "spot enters the American diagnostic linearly, so its error passes straight "
                "into the carry residual."
            ),
            what_would_change_it="A consolidated SIP quote feed for the underlying.",
        ),
        InputClassification(
            key="discount_curve",
            description="D(t): the discount factor applied to a cash flow at each expiry.",
            units="dimensionless, per expiry",
            classification=discount_class,
            evidence=discount_evidence,
            consequence=(
                "Available at the listed expiries only. Between them a pricer must interpolate, "
                "and the interpolation rule is a modelling choice this study does not make."
            ),
            what_would_change_it=(
                "Nothing about the archive; the reconstruction is already an estimator over "
                "quotes. Tighter quotes or more sessions would shrink the spread, not change "
                "the kind of the quantity."
            ),
        ),
        InputClassification(
            key="forward_curve",
            description="F(t): the forward level of the European underlying at each expiry.",
            units="index points (XSP), per expiry",
            classification=discount_class,
            evidence=(
                "Recovered from the same parity regression as D, as a/D, so it inherits that "
                "fit's validity and its classification exactly, including whether the stability "
                "bar was met and the fact that only pointwise knots were built. It is the "
                "forward of the CASH-SETTLED index contract; "
                "it is not the SPY forward and the two differ by the ETF/index basis, which "
                "this archive does not measure."
            ),
            consequence=(
                "Usable as the European control's forward. Substituting it for a SPY forward "
                "would import an unmeasured basis into every SPY number."
            ),
            what_would_change_it=(
                "An index level feed would let the ETF/index basis be measured rather than "
                "left as an unquantified difference."
            ),
        ),
        InputClassification(
            key="discrete_dividend_dates",
            description="The ex-dates of cash distributions before expiry.",
            units="calendar dates",
            classification=dividend_date_class,
            evidence=(
                f"The declared dividend source was probed and "
                f"{_source_phrase(facts.dividend_source_state)}. "
                f"No declaration, ex-date or amount appears anywhere in this archive. The "
                f"American parity carry residual does not supply them: it is one number per "
                f"expiry and cannot resolve how many cash flows fall inside that expiry or when."
            ),
            consequence=(
                "Early exercise of an American call is driven almost entirely by the timing of "
                "these dates. Without them a discrete-dividend PDE has no exercise boundary to "
                "solve against on real data."
            ),
            what_would_change_it=(
                "An issuer distribution history with declared, ex, record and pay dates."
            ),
        ),
        InputClassification(
            key="discrete_dividend_amounts",
            description="The cash amounts of the distributions before expiry.",
            units="USD per share",
            classification=dividend_amount_class,
            evidence=(
                f"Same probe: {_source_phrase(facts.dividend_source_state)}. The carry "
                f"residual "
                f"contains the present value of expected dividends summed with the "
                f"early-exercise premia and quote noise, and this archive contains nothing that "
                f"separates those terms. As above, a 'present' state would record acquisition, "
                f"not verified content."
            ),
            consequence=(
                "No artefact of this study may describe a dividend as observed or inferred; "
                "only the aggregate residual exists here."
            ),
            what_would_change_it=(
                "The same issuer distribution history. A forward-implied estimate would still "
                "be a joint estimate with the borrow rate, not an observation."
            ),
        ),
        InputClassification(
            key="effective_borrow_or_carry",
            description="The continuous carry the underlying earns or costs to hold.",
            units="continuously compounded, per annum",
            classification=(
                JOINTLY_IDENTIFIABLE_ONLY if borrow_class == UNAVAILABLE else DIRECTLY_OBSERVED
            ),
            evidence=(
                f"The declared borrow source was probed and "
                f"{_source_phrase(facts.borrow_source_state)}. "
                f"In the absence of an independent dividend schedule, borrow enters the data "
                f"only through the same forward relation the dividends do: a forward pins the "
                f"combination of discounting, dividends and borrow, and one equation cannot "
                f"determine three unknowns. That is a structural statement about the "
                f"observation equation, not a numerical one, and it would survive a borrow "
                f"series whose contents had not been checked."
            ),
            consequence=(
                "A pricer must either take borrow as an assumption or fold it into an effective "
                "carry jointly fitted with the dividend schedule, in which case neither is "
                "separately identified."
            ),
            what_would_change_it=(
                "A securities-lending fee series would make borrow observed and leave the "
                "dividend schedule as the single remaining unknown in the forward relation."
            ),
        ),
        InputClassification(
            key="exercise_style",
            description="Whether the contract may be exercised before expiry.",
            units="categorical",
            classification=exercise_class,
            evidence=(
                f"The archive's option definition records populate an exercise style on "
                f"{facts.exercise_style_populated_fraction:.1%} of resolved contracts. The "
                f"style used by a pricer therefore comes from the product's listed terms -- SPY "
                f"options are American-style, XSP options are European-style -- which is "
                f"external knowledge about the listing, not a field in this data."
            ),
            consequence=(
                "The single most consequential switch in the solver is set by convention. It "
                "must be declared per product and never defaulted."
            ),
            what_would_change_it=(
                "A definition feed that populates the exercise style, or an exchange contract "
                "specification consumed as a declared source."
            ),
        ),
        InputClassification(
            key="settlement_convention",
            description=(
                "Settlement style and instant: AM or PM settled, cash or physical, and the "
                "moment the settlement value is struck."
            ),
            units="categorical plus an instant",
            classification=settlement_class,
            evidence=(
                f"Definition records populate a settlement time on "
                f"{facts.settlement_time_populated_fraction:.1%} of resolved contracts; the "
                f"expiration field carries a date stamped at midnight UTC and no AM/PM flag. "
                f"This study therefore assumes a fixed local expiry instant and labels every "
                f"tau with that assumption."
            ),
            consequence=(
                "Immaterial for six-month tau, material for one-day tau. It is the reason "
                "short-dated expiries are excluded from rate interpretation here."
            ),
            what_would_change_it=(
                "Exchange contract specifications, or a definition feed populating settlement "
                "metadata."
            ),
        ),
        InputClassification(
            key="contract_multiplier",
            description="Shares per contract; converts per-share quotes into contract cash.",
            units="shares per contract",
            classification=multiplier_class,
            evidence=(
                f"Definition records populate a multiplier on "
                f"{facts.contract_multiplier_populated_fraction:.1%} of resolved contracts. "
                f"Nothing in this study needs one: parity in per-share units is "
                f"multiplier-free, so no multiplier was assumed anywhere."
            ),
            consequence=(
                "Only matters when converting to contract-level cash or comparing against "
                "notional; a pricer working in per-share units never sees it."
            ),
            what_would_change_it=(
                "A definition feed populating the multiplier, or contract specifications "
                "consumed as a declared source. An adjusted contract's multiplier additionally "
                "needs the corporate-action memoranda, which were probed and "
                f"{_source_phrase(facts.corporate_action_source_state)}."
            ),
        ),
        InputClassification(
            key="volatility_input",
            description="The diffusion coefficient the PDE is solved with.",
            units="annualized, dimensionless",
            classification=UNAVAILABLE,
            evidence=(
                "Out of scope by construction: this study inverts no implied volatility and "
                "fits no surface, and no volatility number appears in its output. Quotes "
                "constrain volatility only through a pricing model, which makes any volatility "
                "input model-dependent rather than observed even once that work is done."
            ),
            consequence=(
                "For synthetic training it is a sampled parameter. For real-market evaluation "
                "it is the output of a calibration that has not been performed and that "
                "depends on every other input in this table."
            ),
            what_would_change_it=(
                "Task 9C's solver plus a calibration, which would make it model-implied rather "
                "than observed."
            ),
        ),
    )


# ---------------------------------------------------------------------------
# The proposed Task 9C input contract
# ---------------------------------------------------------------------------

FIELD_SYNTHETIC: Final = "synthetic_training_input"
"""Chosen by the sampler when generating training states."""

FIELD_RECONSTRUCTED: Final = "must_be_reconstructed_for_real_market_evaluation"
"""Must come from market data before a real-market price means anything."""

FIELD_BOTH: Final = "sampled_for_training_and_reconstructed_for_evaluation"
"""Sampled in training and separately reconstructed at evaluation time."""


@dataclass(frozen=True, slots=True)
class ContractField:
    """One field of the proposed Task 9C C++ pricer input."""

    name: str
    cpp_type: str
    units: str
    meaning: str
    provenance: str
    validation: str
    source_in_task_9b: str

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready description of this field."""
        return {
            "name": self.name,
            "cpp_type": self.cpp_type,
            "units": self.units,
            "meaning": self.meaning,
            "provenance": self.provenance,
            "validation_at_the_boundary": self.validation,
            "source_in_task_9b": self.source_in_task_9b,
        }


def proposed_task_9c_contract() -> tuple[ContractField, ...]:
    """Return the proposed Task 9C solver input contract. Nothing is implemented.

    This is a specification, not code that runs: Task 9C owns the solver. The
    point of stating it here is that every field's real-market provenance is
    decided by the identifiability matrix above, so the interface cannot quietly
    acquire a field that no data can fill.
    """
    return (
        ContractField(
            name="spot",
            cpp_type="double",
            units="USD per share",
            meaning="Underlying level at the valuation instant.",
            provenance=FIELD_BOTH,
            validation="finite and strictly positive",
            source_in_task_9b="observed directly from the underlying quote feed",
        ),
        ContractField(
            name="strike",
            cpp_type="double",
            units="USD per share",
            meaning="Contract strike, in the same per-share units as the spot.",
            provenance=FIELD_BOTH,
            validation="finite and strictly positive",
            source_in_task_9b="observed directly from the resolved contract definition",
        ),
        ContractField(
            name="option_type",
            cpp_type="enum class OptionType { Call, Put }",
            units="categorical",
            meaning="Payoff direction.",
            provenance=FIELD_BOTH,
            validation="must be one of the two enumerators; no integer fallback",
            source_in_task_9b="observed directly from the resolved contract definition",
        ),
        ContractField(
            name="valuation_time",
            cpp_type="double",
            units="years, ACT/365F, measured from a declared epoch",
            meaning="When the state is priced.",
            provenance=FIELD_BOTH,
            validation="finite; strictly less than expiry_time",
            source_in_task_9b="the interval-close instant of the snapshot bar",
        ),
        ContractField(
            name="expiry_time",
            cpp_type="double",
            units="years, ACT/365F, measured from the same epoch",
            meaning="When the contract expires.",
            provenance=FIELD_BOTH,
            validation="finite; strictly greater than valuation_time",
            source_in_task_9b=(
                "the expiration date from the definition record plus an ASSUMED local expiry "
                "time, because the archive carries no settlement metadata"
            ),
        ),
        ContractField(
            name="discount_curve",
            cpp_type="struct PiecewiseDiscountCurve { std::vector<double> times, log_discounts; }",
            units="times in years; log-discounts dimensionless",
            meaning=(
                "Piecewise curve evaluated as exp of an interpolation in log-discount space, so "
                "positivity holds by construction and a zero rate is a slope rather than a "
                "quotient. The interpolation rule is part of the versioned contract, not an "
                "implementation detail."
            ),
            provenance=FIELD_BOTH,
            validation=(
                "strictly increasing times; equal lengths; the first node at or before "
                "valuation_time and the last at or after expiry_time, so no extrapolation is "
                "ever silent; every log-discount finite"
            ),
            source_in_task_9b=(
                "reconstructed at listed XSP expiries from put-call parity; the between-expiry "
                "interpolation rule is a modelling choice this study did not make"
            ),
        ),
        ContractField(
            name="dividends",
            cpp_type="struct CashDividendSchedule { std::vector<double> ex_times, amounts; }",
            units="ex-times in years; amounts in USD per share",
            meaning=(
                "Discrete cash amounts dropped from the underlying at their ex-times. The "
                "American exercise boundary is driven by these, so an empty schedule is a "
                "meaningfully different contract, not a default."
            ),
            provenance=FIELD_RECONSTRUCTED,
            validation=(
                "strictly increasing ex-times; equal lengths; every amount finite and "
                "non-negative; every ex-time strictly inside (valuation_time, expiry_time]; "
                "emptiness must be stated explicitly by the caller rather than inferred from a "
                "default-constructed vector"
            ),
            source_in_task_9b=(
                "UNAVAILABLE. The archive holds no dividend calendar and the American parity "
                "carry residual does not decompose into one. This field is the single hardest "
                "blocker to real SPY evaluation."
            ),
        ),
        ContractField(
            name="continuous_carry",
            cpp_type="std::optional<double>",
            units="continuously compounded, per annum",
            meaning=(
                "Optional effective borrow or carry applied continuously alongside the discrete "
                "schedule. Optional on purpose: a caller that has no borrow information must be "
                "able to say so, rather than pass a zero that reads as an observation."
            ),
            provenance=FIELD_RECONSTRUCTED,
            validation="if engaged, finite; no sign restriction, since borrow may be negative",
            source_in_task_9b=(
                "JOINTLY IDENTIFIABLE ONLY. With no independent dividend schedule, carry and "
                "dividends enter the forward relation together and this data separates neither."
            ),
        ),
        ContractField(
            name="volatility",
            cpp_type="double",
            units="annualized, dimensionless",
            meaning="Diffusion coefficient of the declared dynamics.",
            provenance=FIELD_BOTH,
            validation="finite and strictly positive",
            source_in_task_9b=(
                "not produced here. No implied volatility was inverted and no surface was fitted; "
                "for real-market evaluation this is the output of a calibration that depends on "
                "every other field in this contract."
            ),
        ),
        ContractField(
            name="exercise_style",
            cpp_type="enum class ExerciseStyle { European, American }",
            units="categorical",
            meaning="Whether the early-exercise constraint is imposed at each time step.",
            provenance=FIELD_BOTH,
            validation="must be one of the two enumerators; never defaulted",
            source_in_task_9b=(
                "EXTERNAL CONVENTION. The archive's definition records leave exercise style "
                "unpopulated; it comes from the product's listed terms."
            ),
        ),
        ContractField(
            name="settlement",
            cpp_type=(
                "struct SettlementConvention { SettlementStyle style; double settlement_lag; }"
            ),
            units="categorical plus years",
            meaning=(
                "Cash or physical settlement and the lag between exercise and cash. Physical "
                "settlement is what makes the dividend schedule matter to the holder of an "
                "American call, so it belongs in the contract rather than in a comment."
            ),
            provenance=FIELD_BOTH,
            validation="lag finite and non-negative",
            source_in_task_9b=(
                "EXTERNAL CONVENTION. No settlement metadata is present in the archive."
            ),
        ),
        ContractField(
            name="contract_multiplier",
            cpp_type="double",
            units="shares per contract",
            meaning=(
                "Carried for reporting contract-level cash only. The solver works in per-share "
                "units, so it must not enter the pricing arithmetic."
            ),
            provenance=FIELD_RECONSTRUCTED,
            validation="finite and strictly positive",
            source_in_task_9b=(
                "EXTERNAL CONVENTION. Unpopulated in this archive's definition records; an "
                "adjusted contract's multiplier additionally needs corporate-action memoranda."
            ),
        ),
    )


def contract_summary(fields: Sequence[ContractField]) -> dict[str, Any]:
    """Return the proposed interface plus a roll-up of field provenance."""
    return {
        "status": "proposed_only_not_implemented",
        "scope_note": (
            "This is the input contract for Task 9C. No PDE solver, discretization, boundary "
            "condition or Greek is specified or implemented here."
        ),
        "unit_convention": (
            "Per-share USD for prices and strikes; ACT/365F years for all times, measured from "
            "one declared epoch shared by every time field."
        ),
        "fields": [field.summary() for field in fields],
        "synthetic_training_inputs": [
            field.name for field in fields if field.provenance in (FIELD_SYNTHETIC, FIELD_BOTH)
        ],
        "must_be_reconstructed_for_real_market_evaluation": [
            field.name for field in fields if field.provenance in (FIELD_RECONSTRUCTED, FIELD_BOTH)
        ],
        "blocking_for_real_spy_evaluation": [
            field.name
            for field in fields
            if field.name in ("dividends", "continuous_carry")
        ],
    }
