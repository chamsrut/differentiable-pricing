"""Put-call parity reconstruction of a discount factor and a forward.

This module is the arithmetic of Task 9B and nothing else: no file access, no
Parquet, no configuration, no vendor types. Everything here is a pure function
of numbers so it can be tested against synthetic fixtures whose exact answer is
known in advance.

The European identity
    For a European call and put on the same underlying, with the same strike
    $K$ and the same expiry $T$, quoted at the same instant,

        y(K, T) = C(K, T) - P(K, T) = D(T) [F(T) - K] = a(T) + b(T) K

    so a regression of the call-minus-put combination on the strike identifies
    the whole term structure at that expiry:

        D(T) = -b(T),    F(T) = a(T) / D(T),    r(T) = -log D(T) / tau.

    The slope sign is not a convention and not a fitting detail: $b = -D$, and a
    discount factor is positive, so a fitted slope that is not strictly negative
    means the identity did not hold on this data. That is reported as a failure,
    never repaired by taking an absolute value.

Executable bounds rather than error bars
    A quote is an interval, not a point. The combination could have been
    transacted anywhere inside

        y_lower = C_bid - P_ask,     y_upper = C_ask - P_bid

    so a fitted value inside that interval is consistent with the market and one
    outside it is not. :func:`outside_spread_error` measures only the excursion
    beyond the interval and is exactly zero inside it. This is a statement about
    executability, not about statistical significance: nothing in this module
    treats a bid-ask width as a standard error, and no confidence interval is
    derived anywhere.

Degeneracy is reported, never clamped
    A fit can fail in several distinct ways -- too few pairs, a singular or
    ill-conditioned design, a non-finite solution, a non-negative slope, a
    non-positive forward -- and they have different causes and different
    remedies. Each is reported under its own name and the offending value is
    published as it came out of the arithmetic. Clamping a negative discount
    factor to a small positive number would convert a detected contradiction
    into an undetectable one.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

SECONDS_PER_YEAR_ACT365: Final = 365.0 * 86_400.0
"""ACT/365 fixed, measured between instants rather than between dates."""

OPTION_TYPE_CALL: Final = "C"
OPTION_TYPE_PUT: Final = "P"

METHOD_UNWEIGHTED: Final = "unweighted_least_squares"
METHOD_WEIGHTED: Final = "weighted_least_squares_bidask"
KNOWN_METHODS: Final = (METHOD_UNWEIGHTED, METHOD_WEIGHTED)

DIAGNOSTIC_INSUFFICIENT_PAIRS: Final = "insufficient_pairs"
DIAGNOSTIC_SINGULAR: Final = "singular_design"
DIAGNOSTIC_ILL_CONDITIONED: Final = "poorly_conditioned_design"
DIAGNOSTIC_NON_FINITE: Final = "non_finite_solution"
DIAGNOSTIC_SLOPE_SIGN: Final = "non_negative_slope_violates_b_equals_minus_D"
DIAGNOSTIC_NON_POSITIVE_DISCOUNT: Final = "non_positive_discount_factor"
DIAGNOSTIC_NON_POSITIVE_FORWARD: Final = "non_positive_forward"
DIAGNOSTIC_NON_POSITIVE_WEIGHT: Final = "non_positive_weight"
DIAGNOSTIC_NON_POSITIVE_SPOT: Final = "non_positive_spot"

RATE_WITHHELD_ZERO_DTE: Final = "zero_dte"
RATE_WITHHELD_SHORT_TAU: Final = "tau_below_minimum_for_rate_interpretation"
RATE_WITHHELD_INVALID_DISCOUNT: Final = "discount_factor_not_positive"


class ParityError(ValueError):
    """Raised when an input to the parity arithmetic is not usable at all."""


def _finite(value: float) -> bool:
    return isinstance(value, float | int) and math.isfinite(float(value))


# ---------------------------------------------------------------------------
# Quotes and pairs
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OptionQuoteRecord:
    """One tradable option quote observed at one synchronization instant.

    ``timestamp_ns`` is the interval-close instant of the one-minute bar. It is
    carried per leg rather than assumed common, because "same minute" is a
    pairing requirement that has to be checked rather than trusted.
    """

    timestamp_ns: int
    expiry: dt.date
    strike: float
    option_type: str
    bid: float
    ask: float

    def validate(self) -> None:
        """Reject a quote that cannot take part in the arithmetic."""
        if self.option_type not in (OPTION_TYPE_CALL, OPTION_TYPE_PUT):
            raise ParityError(f"option_type must be 'C' or 'P', got {self.option_type!r}")
        if not _finite(self.strike) or self.strike <= 0.0:
            raise ParityError(f"strike must be finite and positive, got {self.strike!r}")
        if not _finite(self.bid) or not _finite(self.ask):
            raise ParityError(f"quote must be finite, got bid={self.bid!r} ask={self.ask!r}")
        if self.ask < self.bid:
            raise ParityError(f"crossed quote: bid={self.bid!r} exceeds ask={self.ask!r}")


@dataclass(frozen=True, slots=True)
class ParityPair:
    """A same-minute call and put sharing an expiry and a strike."""

    timestamp_ns: int
    expiry: dt.date
    strike: float
    call_bid: float
    call_ask: float
    put_bid: float
    put_ask: float

    @property
    def call_mid(self) -> float:
        """Midpoint of the call's quoted interval."""
        return 0.5 * (self.call_bid + self.call_ask)

    @property
    def put_mid(self) -> float:
        """Midpoint of the put's quoted interval."""
        return 0.5 * (self.put_bid + self.put_ask)

    @property
    def y_mid(self) -> float:
        """Midpoint combination ``C_mid - P_mid``, the regression's response."""
        return self.call_mid - self.put_mid

    @property
    def y_lower(self) -> float:
        """Cheapest executable combination, ``C_bid - P_ask``."""
        return self.call_bid - self.put_ask

    @property
    def y_upper(self) -> float:
        """Dearest executable combination, ``C_ask - P_bid``."""
        return self.call_ask - self.put_bid

    @property
    def combined_width(self) -> float:
        """Width of the executable interval: the two bid-ask widths added.

        This is a liquidity measure. It is not a standard error and nothing in
        this module treats it as one.
        """
        return self.y_upper - self.y_lower


def build_pairs(records: Iterable[OptionQuoteRecord]) -> tuple[ParityPair, ...]:
    """Pair calls with puts of identical expiry, strike and *instant*.

    A call from one minute and a put from the next are not a pair. Parity is an
    identity between simultaneous prices, so the timestamps must match exactly;
    a leg with no same-minute counterpart is dropped rather than matched to the
    nearest one. Duplicate legs -- the same instrument, minute, expiry, strike
    and side twice -- are rejected, because silently keeping one of them would
    make the result depend on input order.
    """
    calls: dict[tuple[int, dt.date, float], OptionQuoteRecord] = {}
    puts: dict[tuple[int, dt.date, float], OptionQuoteRecord] = {}
    for record in records:
        record.validate()
        key = (record.timestamp_ns, record.expiry, record.strike)
        book = calls if record.option_type == OPTION_TYPE_CALL else puts
        if key in book:
            raise ParityError(
                f"duplicate {record.option_type} quote for strike {record.strike} "
                f"expiring {record.expiry.isoformat()} at instant {record.timestamp_ns}"
            )
        book[key] = record

    pairs = [
        ParityPair(
            timestamp_ns=key[0],
            expiry=key[1],
            strike=key[2],
            call_bid=call.bid,
            call_ask=call.ask,
            put_bid=puts[key].bid,
            put_ask=puts[key].ask,
        )
        for key, call in calls.items()
        if key in puts
    ]
    pairs.sort(key=lambda pair: (pair.timestamp_ns, pair.expiry, pair.strike))
    return tuple(pairs)


def group_by_expiry(
    pairs: Iterable[ParityPair],
) -> dict[dt.date, tuple[ParityPair, ...]]:
    """Bucket pairs by expiry, keeping each bucket sorted by strike."""
    grouped: dict[dt.date, list[ParityPair]] = {}
    for pair in pairs:
        grouped.setdefault(pair.expiry, []).append(pair)
    return {
        expiry: tuple(sorted(bucket, key=lambda pair: pair.strike))
        for expiry, bucket in sorted(grouped.items())
    }


def forward_anchor_strike(pairs: Sequence[ParityPair]) -> float:
    """Return the paired strike minimizing ``|C_mid - P_mid|``.

    Near the forward the call and put midpoints coincide, so this is the
    cheapest usable forward anchor: it needs no discount factor, no dividend and
    no spot. Ties are broken by the smaller strike so the choice is
    deterministic rather than dependent on iteration order.
    """
    if not pairs:
        raise ParityError("cannot anchor a strike window with no pairs")
    return min(pairs, key=lambda pair: (abs(pair.y_mid), pair.strike)).strike


def select_strike_window(
    pairs: Sequence[ParityPair], *, anchor: float, relative_half_width: float
) -> tuple[ParityPair, ...]:
    """Return the pairs whose strike lies within a relative band about ``anchor``."""
    if not _finite(anchor) or anchor <= 0.0:
        raise ParityError(f"anchor must be finite and positive, got {anchor!r}")
    if not _finite(relative_half_width) or relative_half_width <= 0.0:
        raise ParityError(
            f"relative_half_width must be finite and positive, got {relative_half_width!r}"
        )
    lower = anchor * (1.0 - relative_half_width)
    upper = anchor * (1.0 + relative_half_width)
    return tuple(pair for pair in pairs if lower <= pair.strike <= upper)


# ---------------------------------------------------------------------------
# Executable-interval errors
# ---------------------------------------------------------------------------


def outside_spread_error(fitted: float, lower: float, upper: float) -> float:
    """Return how far ``fitted`` falls outside the executable interval.

    Zero inside ``[lower, upper]``; otherwise the signed excursion's magnitude,
    ``lower - fitted`` below and ``fitted - upper`` above. Always non-negative.
    """
    if not _finite(fitted) or not _finite(lower) or not _finite(upper):
        return math.nan
    if upper < lower:
        raise ParityError(f"inverted executable interval: [{lower}, {upper}]")
    if fitted < lower:
        return lower - fitted
    if fitted > upper:
        return fitted - upper
    return 0.0


# ---------------------------------------------------------------------------
# The two-parameter fit
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LineFit:
    """The solved line ``y = intercept + slope * K`` and its conditioning.

    The solve is done in strike-centered coordinates and mapped back, so
    ``intercept`` is exact for the uncentered model while the arithmetic never
    forms the badly scaled raw normal equations. Both condition numbers are
    reported: ``condition_number`` is the 2-norm condition number of the raw
    weighted design matrix ``[1, K]`` -- what a naive solve would have faced --
    and ``condition_number_centered`` is the same quantity for the centered
    design actually used.
    """

    intercept: float
    slope: float
    condition_number: float
    condition_number_centered: float
    weight_sum: float
    strike_dispersion: float
    singular: bool

    def evaluate(self, strike: float) -> float:
        """Return the fitted combination value at ``strike``."""
        return self.intercept + self.slope * strike


def fit_line(
    strikes: Sequence[float], values: Sequence[float], weights: Sequence[float] | None = None
) -> LineFit:
    """Solve a deterministic weighted least-squares line in centered coordinates.

    ``weights`` of ``None`` means unweighted. All weights must be finite and
    strictly positive: a zero weight is a silently dropped observation and a
    negative one is not a least-squares problem.
    """
    count = len(strikes)
    if count != len(values):
        raise ParityError(f"strike/value length mismatch: {count} vs {len(values)}")
    if count == 0:
        raise ParityError("cannot fit a line through no points")
    if weights is None:
        weight_list = [1.0] * count
    else:
        if len(weights) != count:
            raise ParityError(f"weight length mismatch: {len(weights)} vs {count}")
        weight_list = [float(weight) for weight in weights]
        for weight in weight_list:
            if not _finite(weight) or weight <= 0.0:
                raise ParityError(f"weights must be finite and positive, got {weight!r}")

    weight_total = math.fsum(weight_list)
    sum_k = math.fsum(w * k for w, k in zip(weight_list, strikes, strict=True))
    sum_kk = math.fsum(w * k * k for w, k in zip(weight_list, strikes, strict=True))
    mean_k = sum_k / weight_total
    mean_y = math.fsum(w * y for w, y in zip(weight_list, values, strict=True)) / weight_total
    centered_kk = math.fsum(
        w * (k - mean_k) * (k - mean_k) for w, k in zip(weight_list, strikes, strict=True)
    )
    centered_ky = math.fsum(
        w * (k - mean_k) * (y - mean_y)
        for w, k, y in zip(weight_list, strikes, values, strict=True)
    )

    # The 2x2 weighted Gram matrix of the raw design [[Sw, Sk], [Sk, Skk]].
    # Its determinant is exactly Sw * centered_kk, which is the numerically
    # stable route to the small eigenvalue: forming it as Sw*Skk - Sk^2 cancels
    # to nothing when strikes sit far from the origin, which is precisely the
    # regime every listed option chain is in.
    trace = weight_total + sum_kk
    gap = weight_total - sum_kk
    discriminant = math.sqrt(gap * gap + 4.0 * sum_k * sum_k)
    largest = 0.5 * (trace + discriminant)
    determinant = weight_total * centered_kk
    smallest = determinant / largest if largest > 0.0 else 0.0

    singular = not math.isfinite(centered_kk) or centered_kk <= 0.0
    if singular:
        return LineFit(
            intercept=math.nan,
            slope=math.nan,
            condition_number=math.inf,
            condition_number_centered=math.inf,
            weight_sum=weight_total,
            strike_dispersion=centered_kk,
            singular=True,
        )

    slope = centered_ky / centered_kk
    intercept = mean_y - slope * mean_k
    condition = math.sqrt(largest / smallest) if smallest > 0.0 else math.inf
    centered_high = max(weight_total, centered_kk)
    centered_low = min(weight_total, centered_kk)
    condition_centered = math.sqrt(centered_high / centered_low) if centered_low > 0.0 else math.inf
    return LineFit(
        intercept=intercept,
        slope=slope,
        condition_number=condition,
        condition_number_centered=condition_centered,
        weight_sum=weight_total,
        strike_dispersion=centered_kk,
        singular=False,
    )


def bidask_weights(pairs: Sequence[ParityPair]) -> tuple[float, ...]:
    """Return ``1 / width^2`` weights from the combined call/put bid-ask width.

    The width is executable-liquidity information: a combination quoted two
    cents wide pins the parity line far more tightly than one quoted a dollar
    wide. It is deliberately *not* called a standard error, no distributional
    assumption is attached to it, and no interval estimate is derived from it.

    A pair with a non-positive or non-finite width cannot be weighted this way;
    :exc:`ParityError` is raised rather than substituting a default, so the
    caller reports the degeneracy instead of absorbing it.
    """
    weights: list[float] = []
    for pair in pairs:
        width = pair.combined_width
        if not _finite(width) or width <= 0.0:
            raise ParityError(
                f"pair at strike {pair.strike} has a non-positive combined bid-ask "
                f"width ({width!r}); it cannot carry an inverse-width weight"
            )
        weights.append(1.0 / (width * width))
    return tuple(weights)


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def year_fraction_act365(start: dt.datetime, end: dt.datetime) -> float:
    """Return the ACT/365-fixed year fraction between two aware instants.

    Both instants must be timezone-aware. A naive datetime here would silently
    mean "whatever the machine's local zone is", which is exactly the class of
    assumption this project forbids leaving implicit.
    """
    if start.tzinfo is None or end.tzinfo is None:
        raise ParityError("both instants must be timezone-aware")
    return (end - start).total_seconds() / SECONDS_PER_YEAR_ACT365


# ---------------------------------------------------------------------------
# The European reconstruction
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ParityFit:
    """A reconstructed ``(D, F, r)`` at one expiry, with its full diagnostics.

    ``discount_factor`` and ``forward`` are published exactly as the arithmetic
    produced them, including when they are negative or non-finite. ``valid``
    says whether they may be used; ``diagnostics`` says why not.
    """

    method: str
    relative_half_width: float
    anchor_strike: float
    pair_count: int
    strike_min: float
    strike_max: float
    tau_years: float
    intercept: float
    slope: float
    discount_factor: float
    forward: float
    rate: float | None
    rate_withheld_reason: str | None
    condition_number: float
    condition_number_centered: float
    rms_mid_residual: float
    max_abs_mid_residual: float
    containment_fraction: float
    rms_outside_spread_error: float
    max_outside_spread_error: float
    diagnostics: tuple[str, ...]

    @property
    def valid(self) -> bool:
        """Whether ``D`` and ``F`` are usable numbers.

        An ill-conditioned design is a warning, not an invalidation: the fit is
        still published, flagged, and left to the reader.
        """
        blocking = set(self.diagnostics) - {DIAGNOSTIC_ILL_CONDITIONED}
        return not blocking

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready record of this fit."""
        return {
            "method": self.method,
            "relative_half_width": self.relative_half_width,
            "anchor_strike": self.anchor_strike,
            "pair_count": self.pair_count,
            "strike_min": self.strike_min,
            "strike_max": self.strike_max,
            "tau_years": self.tau_years,
            "intercept_a": self.intercept,
            "slope_b": self.slope,
            "discount_factor_D": self.discount_factor,
            "forward_F": self.forward,
            "rate_r": self.rate,
            "rate_withheld_reason": self.rate_withheld_reason,
            "condition_number": _json_float(self.condition_number),
            "condition_number_centered": _json_float(self.condition_number_centered),
            "rms_mid_residual": self.rms_mid_residual,
            "max_abs_mid_residual": self.max_abs_mid_residual,
            "bid_ask_containment_fraction": self.containment_fraction,
            "rms_outside_spread_error": self.rms_outside_spread_error,
            "max_outside_spread_error": self.max_outside_spread_error,
            "diagnostics": list(self.diagnostics),
            "valid": self.valid,
        }


def _json_float(value: float) -> float | str:
    """Represent an infinity as a string so the JSON stays standard."""
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    if math.isnan(value):
        return "nan"
    return value


def fit_parity(
    pairs: Sequence[ParityPair],
    *,
    method: str,
    tau_years: float,
    relative_half_width: float,
    anchor_strike: float,
    minimum_pairs: int,
    condition_number_warning: float,
    minimum_rate_tau_years: float,
    is_zero_dte: bool,
) -> ParityFit:
    """Reconstruct ``D``, ``F`` and ``r`` from one expiry's same-minute pairs.

    Nothing is clamped. A fitted slope that is not strictly negative contradicts
    ``b = -D`` for a positive discount factor and is reported as such with the
    offending slope intact.
    """
    if method not in KNOWN_METHODS:
        raise ParityError(f"unknown fit method {method!r}; expected one of {list(KNOWN_METHODS)}")

    diagnostics: list[str] = []
    strikes = [pair.strike for pair in pairs]
    count = len(pairs)
    strike_min = min(strikes) if strikes else math.nan
    strike_max = max(strikes) if strikes else math.nan

    if count < minimum_pairs:
        return _degenerate_fit(
            method=method,
            relative_half_width=relative_half_width,
            anchor_strike=anchor_strike,
            count=count,
            strike_min=strike_min,
            strike_max=strike_max,
            tau_years=tau_years,
            diagnostics=(DIAGNOSTIC_INSUFFICIENT_PAIRS,),
        )

    values = [pair.y_mid for pair in pairs]
    weights: Sequence[float] | None = None
    if method == METHOD_WEIGHTED:
        try:
            weights = bidask_weights(pairs)
        except ParityError:
            return _degenerate_fit(
                method=method,
                relative_half_width=relative_half_width,
                anchor_strike=anchor_strike,
                count=count,
                strike_min=strike_min,
                strike_max=strike_max,
                tau_years=tau_years,
                diagnostics=(DIAGNOSTIC_NON_POSITIVE_WEIGHT,),
            )

    line = fit_line(strikes, values, weights)
    if line.singular:
        return _degenerate_fit(
            method=method,
            relative_half_width=relative_half_width,
            anchor_strike=anchor_strike,
            count=count,
            strike_min=strike_min,
            strike_max=strike_max,
            tau_years=tau_years,
            diagnostics=(DIAGNOSTIC_SINGULAR,),
            condition_number=line.condition_number,
            condition_number_centered=line.condition_number_centered,
        )

    if line.condition_number > condition_number_warning:
        diagnostics.append(DIAGNOSTIC_ILL_CONDITIONED)

    if not math.isfinite(line.intercept) or not math.isfinite(line.slope):
        diagnostics.append(DIAGNOSTIC_NON_FINITE)

    discount = -line.slope
    if math.isfinite(line.slope) and line.slope >= 0.0:
        diagnostics.append(DIAGNOSTIC_SLOPE_SIGN)
    if not (math.isfinite(discount) and discount > 0.0):
        diagnostics.append(DIAGNOSTIC_NON_POSITIVE_DISCOUNT)
        forward = math.nan
    else:
        forward = line.intercept / discount
        if not (math.isfinite(forward) and forward > 0.0):
            diagnostics.append(DIAGNOSTIC_NON_POSITIVE_FORWARD)

    rate: float | None = None
    withheld: str | None = None
    if not (math.isfinite(discount) and discount > 0.0):
        withheld = RATE_WITHHELD_INVALID_DISCOUNT
    elif is_zero_dte:
        withheld = RATE_WITHHELD_ZERO_DTE
    elif tau_years <= 0.0 or tau_years < minimum_rate_tau_years:
        withheld = RATE_WITHHELD_SHORT_TAU
    else:
        rate = -math.log(discount) / tau_years

    residuals = [pair.y_mid - line.evaluate(pair.strike) for pair in pairs]
    rms_residual = math.sqrt(math.fsum(value * value for value in residuals) / count)
    max_residual = max(abs(value) for value in residuals)

    excursions = [
        outside_spread_error(line.evaluate(pair.strike), pair.y_lower, pair.y_upper)
        for pair in pairs
    ]
    contained = sum(1 for value in excursions if value == 0.0)
    rms_outside = math.sqrt(math.fsum(value * value for value in excursions) / count)
    max_outside = max(excursions)

    return ParityFit(
        method=method,
        relative_half_width=relative_half_width,
        anchor_strike=anchor_strike,
        pair_count=count,
        strike_min=strike_min,
        strike_max=strike_max,
        tau_years=tau_years,
        intercept=line.intercept,
        slope=line.slope,
        discount_factor=discount,
        forward=forward,
        rate=rate,
        rate_withheld_reason=withheld,
        condition_number=line.condition_number,
        condition_number_centered=line.condition_number_centered,
        rms_mid_residual=rms_residual,
        max_abs_mid_residual=max_residual,
        containment_fraction=contained / count,
        rms_outside_spread_error=rms_outside,
        max_outside_spread_error=max_outside,
        diagnostics=tuple(diagnostics),
    )


def _degenerate_fit(
    *,
    method: str,
    relative_half_width: float,
    anchor_strike: float,
    count: int,
    strike_min: float,
    strike_max: float,
    tau_years: float,
    diagnostics: tuple[str, ...],
    condition_number: float = math.inf,
    condition_number_centered: float = math.inf,
) -> ParityFit:
    """Return a fit that failed, carrying its reason and no invented numbers."""
    return ParityFit(
        method=method,
        relative_half_width=relative_half_width,
        anchor_strike=anchor_strike,
        pair_count=count,
        strike_min=strike_min,
        strike_max=strike_max,
        tau_years=tau_years,
        intercept=math.nan,
        slope=math.nan,
        discount_factor=math.nan,
        forward=math.nan,
        rate=None,
        rate_withheld_reason=diagnostics[0],
        condition_number=condition_number,
        condition_number_centered=condition_number_centered,
        rms_mid_residual=math.nan,
        max_abs_mid_residual=math.nan,
        containment_fraction=math.nan,
        rms_outside_spread_error=math.nan,
        max_outside_spread_error=math.nan,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------
# The American diagnostic
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CarryResidual:
    """The American parity carry residual and the dispersion it was taken from.

    This is a residual. It is not an observed dividend, not an exact dividend
    present value, not a borrow rate, not an exact SPY forward, and not an
    independent market input. It contains the present value of expected
    dividends, the non-cancelling early-exercise premia of an American call and
    put, quote noise on four legs, and the error in a partial-venue spot, none
    of which this archive separates.
    """

    strike_count: int
    strike_min: float
    strike_max: float
    spot: float
    discount_factor: float
    median_f_tilde: float
    mean_f_tilde: float
    f_tilde_min: float
    f_tilde_max: float
    f_tilde_iqr: float
    f_tilde_mad: float
    carry_residual: float
    strike_slope: float
    strike_slope_condition: float
    valid: bool
    diagnostics: tuple[str, ...]

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready record naming the quantity correctly."""
        return {
            "quantity": "american_parity_carry_residual",
            "strike_count": self.strike_count,
            "strike_min": self.strike_min,
            "strike_max": self.strike_max,
            "spy_spot_used": self.spot,
            "discount_factor_from_matched_xsp_expiry": self.discount_factor,
            "median_f_tilde": self.median_f_tilde,
            "mean_f_tilde": self.mean_f_tilde,
            "f_tilde_min": self.f_tilde_min,
            "f_tilde_max": self.f_tilde_max,
            "f_tilde_iqr": self.f_tilde_iqr,
            "f_tilde_median_absolute_deviation": self.f_tilde_mad,
            "carry_residual_Q_tilde": self.carry_residual,
            "f_tilde_slope_per_unit_strike": self.strike_slope,
            "f_tilde_slope_design_condition_number": _json_float(self.strike_slope_condition),
            "valid": self.valid,
            "diagnostics": list(self.diagnostics),
        }


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    if count % 2 == 1:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _quantile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated quantile on the sorted sample. Deterministic."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def american_carry_residual(
    pairs: Sequence[ParityPair],
    *,
    spot: float,
    discount_factor: float,
    minimum_strikes: int,
) -> CarryResidual:
    """Compute the American parity carry residual at one matched expiry.

    Per strike, ``F_tilde_i = [C_A - P_A + D K_i] / D`` from the American
    midpoints and the discount factor of the *matched XSP expiry*; then
    ``Q_tilde = S_SPY - D * median_i F_tilde_i``.

    Under exact European parity ``F_tilde`` would be constant in the strike, so
    its dispersion and its slope against the strike are the observables that
    expose early-exercise contamination. They are reported, not removed.
    """
    diagnostics: list[str] = []
    count = len(pairs)
    if count < minimum_strikes:
        diagnostics.append(DIAGNOSTIC_INSUFFICIENT_PAIRS)
    if not (_finite(discount_factor) and discount_factor > 0.0):
        diagnostics.append(DIAGNOSTIC_NON_POSITIVE_DISCOUNT)
    if not _finite(spot) or spot <= 0.0:
        diagnostics.append(DIAGNOSTIC_NON_POSITIVE_SPOT)
    if diagnostics:
        return CarryResidual(
            strike_count=count,
            strike_min=min((pair.strike for pair in pairs), default=math.nan),
            strike_max=max((pair.strike for pair in pairs), default=math.nan),
            spot=spot,
            discount_factor=discount_factor,
            median_f_tilde=math.nan,
            mean_f_tilde=math.nan,
            f_tilde_min=math.nan,
            f_tilde_max=math.nan,
            f_tilde_iqr=math.nan,
            f_tilde_mad=math.nan,
            carry_residual=math.nan,
            strike_slope=math.nan,
            strike_slope_condition=math.inf,
            valid=False,
            diagnostics=tuple(diagnostics),
        )

    strikes = [pair.strike for pair in pairs]
    f_tilde = [
        (pair.y_mid + discount_factor * pair.strike) / discount_factor for pair in pairs
    ]
    median = _median(f_tilde)
    residual = spot - discount_factor * median
    slope_fit = fit_line(strikes, f_tilde)
    absolute_deviations = [abs(value - median) for value in f_tilde]
    return CarryResidual(
        strike_count=count,
        strike_min=min(strikes),
        strike_max=max(strikes),
        spot=spot,
        discount_factor=discount_factor,
        median_f_tilde=median,
        mean_f_tilde=math.fsum(f_tilde) / count,
        f_tilde_min=min(f_tilde),
        f_tilde_max=max(f_tilde),
        f_tilde_iqr=_quantile(f_tilde, 0.75) - _quantile(f_tilde, 0.25),
        f_tilde_mad=_median(absolute_deviations),
        carry_residual=residual,
        strike_slope=slope_fit.slope,
        strike_slope_condition=slope_fit.condition_number,
        valid=True,
        diagnostics=(),
    )


def dispersion_summary(values: Iterable[float]) -> Mapping[str, float]:
    """Return a deterministic min/median/max/spread summary of a sensitivity axis."""
    sample = [float(value) for value in values if _finite(value)]
    if not sample:
        return {"count": 0.0}
    return {
        "count": float(len(sample)),
        "min": min(sample),
        "median": _median(sample),
        "max": max(sample),
        "range": max(sample) - min(sample),
    }
