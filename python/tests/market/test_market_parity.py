"""Put-call parity arithmetic, against synthetic fixtures with known answers.

Every fixture here is constructed from a chosen ``(D, F)`` so the recovered
values can be compared against the truth rather than against a previous run.
No real market data is involved and none is needed: the identity being tested is
exact, so an approximate check would be testing the fixture, not the code.
"""

from __future__ import annotations

import datetime as dt
import math
from zoneinfo import ZoneInfo

import pytest
from differentiable_pricing.market.parity import (
    DIAGNOSTIC_ILL_CONDITIONED,
    DIAGNOSTIC_INSUFFICIENT_PAIRS,
    DIAGNOSTIC_NON_POSITIVE_DISCOUNT,
    DIAGNOSTIC_NON_POSITIVE_SPOT,
    DIAGNOSTIC_NON_POSITIVE_WEIGHT,
    DIAGNOSTIC_SINGULAR,
    DIAGNOSTIC_SLOPE_SIGN,
    METHOD_UNWEIGHTED,
    METHOD_WEIGHTED,
    RATE_WITHHELD_SHORT_TAU,
    RATE_WITHHELD_ZERO_DTE,
    OptionQuoteRecord,
    ParityError,
    ParityPair,
    american_carry_residual,
    bidask_weights,
    build_pairs,
    fit_line,
    fit_parity,
    forward_anchor_strike,
    group_by_expiry,
    outside_spread_error,
    select_strike_window,
    year_fraction_act365,
)

EXPIRY = dt.date(2026, 9, 18)
INSTANT = 1_781_706_600_000_000_000
NEW_YORK = ZoneInfo("America/New_York")


def synthetic_pair(
    strike: float,
    *,
    discount: float,
    forward: float,
    half_width: float = 0.01,
    call_shift: float = 0.0,
    put_shift: float = 0.0,
    expiry: dt.date = EXPIRY,
    timestamp_ns: int = INSTANT,
) -> ParityPair:
    """Build a pair whose midpoints satisfy ``C - P = D (F - K)`` exactly.

    The call and put midpoints are split around an arbitrary put level, so the
    combination is exact while the individual legs carry no information the
    parity fit is entitled to use.
    """
    combination = discount * (forward - strike)
    put_mid = 10.0
    call_mid = combination + put_mid
    return ParityPair(
        timestamp_ns=timestamp_ns,
        expiry=expiry,
        strike=strike,
        call_bid=call_mid - half_width + call_shift,
        call_ask=call_mid + half_width + call_shift,
        put_bid=put_mid - half_width + put_shift,
        put_ask=put_mid + half_width + put_shift,
    )


def synthetic_chain(
    *, discount: float, forward: float, strikes: list[float], half_width: float = 0.01
) -> list[ParityPair]:
    return [
        synthetic_pair(strike, discount=discount, forward=forward, half_width=half_width)
        for strike in strikes
    ]


def fit(pairs: list[ParityPair], *, method: str = METHOD_UNWEIGHTED, tau: float = 0.25):
    return fit_parity(
        pairs,
        method=method,
        tau_years=tau,
        relative_half_width=0.2,
        anchor_strike=forward_anchor_strike(pairs),
        minimum_pairs=4,
        condition_number_warning=1.0e8,
        minimum_rate_tau_years=1.0 / 365.0,
        is_zero_dte=False,
    )


# ---------------------------------------------------------------------------
# Exact recovery and the slope sign
# ---------------------------------------------------------------------------


def test_exact_discount_and_forward_recovery() -> None:
    """A noiseless chain must return the D and F it was generated from."""
    discount, forward = 0.98765, 753.25
    result = fit(synthetic_chain(discount=discount, forward=forward,
                                 strikes=[700.0, 720.0, 750.0, 780.0, 800.0]))
    assert result.valid
    assert result.discount_factor == pytest.approx(discount, rel=1e-12)
    assert result.forward == pytest.approx(forward, rel=1e-12)
    assert result.rms_mid_residual == pytest.approx(0.0, abs=1e-9)


def test_slope_is_minus_the_discount_factor() -> None:
    """The mandated sign convention: b = -D, so the fitted slope is negative."""
    result = fit(synthetic_chain(discount=0.97, forward=740.0,
                                 strikes=[700.0, 725.0, 750.0, 775.0]))
    assert result.slope < 0.0
    assert result.slope == pytest.approx(-result.discount_factor, rel=1e-12)


def test_rate_matches_minus_log_discount_over_tau() -> None:
    """r = -log(D)/tau, with tau supplied by the caller's day count."""
    discount, tau = 0.99, 0.25
    result = fit(
        synthetic_chain(discount=discount, forward=750.0,
                        strikes=[700.0, 725.0, 750.0, 775.0]),
        tau=tau,
    )
    assert result.rate == pytest.approx(-math.log(discount) / tau, rel=1e-12)


def test_positive_slope_is_reported_not_repaired() -> None:
    """A chain whose combination increases in the strike contradicts b = -D.

    The offending slope and the resulting negative discount factor are published
    exactly as computed. Taking an absolute value here would turn a detected
    contradiction into an undetectable one.
    """
    pairs = [
        synthetic_pair(strike, discount=-0.98, forward=750.0)
        for strike in (700.0, 725.0, 750.0, 775.0)
    ]
    result = fit(pairs)
    assert result.slope > 0.0
    assert DIAGNOSTIC_SLOPE_SIGN in result.diagnostics
    assert DIAGNOSTIC_NON_POSITIVE_DISCOUNT in result.diagnostics
    assert not result.valid
    assert result.discount_factor < 0.0
    assert math.isnan(result.forward)
    assert result.rate is None


# ---------------------------------------------------------------------------
# Executable intervals
# ---------------------------------------------------------------------------


def test_bid_ask_interval_construction() -> None:
    """y_lower = C_bid - P_ask and y_upper = C_ask - P_bid, never the reverse."""
    pair = ParityPair(
        timestamp_ns=INSTANT, expiry=EXPIRY, strike=750.0,
        call_bid=12.0, call_ask=12.5, put_bid=9.0, put_ask=9.4,
    )
    assert pair.y_lower == pytest.approx(12.0 - 9.4)
    assert pair.y_upper == pytest.approx(12.5 - 9.0)
    assert pair.y_lower < pair.y_mid < pair.y_upper
    assert pair.combined_width == pytest.approx((12.5 - 12.0) + (9.4 - 9.0))


def test_outside_spread_error_is_zero_inside_and_signed_magnitude_outside() -> None:
    assert outside_spread_error(3.0, 2.0, 4.0) == 0.0
    assert outside_spread_error(2.0, 2.0, 4.0) == 0.0
    assert outside_spread_error(4.0, 2.0, 4.0) == 0.0
    assert outside_spread_error(1.25, 2.0, 4.0) == pytest.approx(0.75)
    assert outside_spread_error(4.5, 2.0, 4.0) == pytest.approx(0.5)


def test_outside_spread_error_rejects_an_inverted_interval() -> None:
    with pytest.raises(ParityError):
        outside_spread_error(3.0, 4.0, 2.0)


def test_noiseless_fit_is_contained_and_a_shifted_chain_is_not() -> None:
    """Containment is an executability statement about the fitted line."""
    contained = fit(synthetic_chain(discount=0.99, forward=750.0,
                                    strikes=[700.0, 725.0, 750.0, 775.0, 800.0]))
    assert contained.containment_fraction == 1.0
    assert contained.max_outside_spread_error == 0.0

    # Move one leg by far more than the quoted width: the line still fits the
    # other four, so the fitted value at the moved strike lands outside its own
    # executable interval.
    pairs = synthetic_chain(discount=0.99, forward=750.0,
                            strikes=[700.0, 725.0, 750.0, 775.0, 800.0])
    pairs[2] = synthetic_pair(750.0, discount=0.99, forward=750.0, call_shift=5.0)
    displaced = fit(pairs)
    assert displaced.containment_fraction < 1.0
    assert displaced.max_outside_spread_error > 0.0
    assert displaced.rms_outside_spread_error > 0.0


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def records(*specs: tuple[int, str, float]) -> list[OptionQuoteRecord]:
    return [
        OptionQuoteRecord(
            timestamp_ns=stamp, expiry=EXPIRY, strike=strike,
            option_type=right, bid=1.0, ask=1.1,
        )
        for stamp, right, strike in specs
    ]


def test_pairing_requires_the_same_minute() -> None:
    """A call from one minute and a put from the next are not a pair."""
    minute = 60_000_000_000
    paired = build_pairs(records((INSTANT, "C", 750.0), (INSTANT, "P", 750.0)))
    assert len(paired) == 1

    unpaired = build_pairs(records((INSTANT, "C", 750.0), (INSTANT + minute, "P", 750.0)))
    assert unpaired == ()


def test_pairing_requires_the_same_strike_and_expiry() -> None:
    assert build_pairs(records((INSTANT, "C", 750.0), (INSTANT, "P", 755.0))) == ()
    other = OptionQuoteRecord(
        timestamp_ns=INSTANT, expiry=dt.date(2026, 10, 16), strike=750.0,
        option_type="P", bid=1.0, ask=1.1,
    )
    assert build_pairs([*records((INSTANT, "C", 750.0)), other]) == ()


def test_duplicate_legs_are_rejected_rather_than_silently_resolved() -> None:
    with pytest.raises(ParityError, match="duplicate"):
        build_pairs(records((INSTANT, "C", 750.0), (INSTANT, "C", 750.0)))


def test_crossed_and_malformed_quotes_are_rejected() -> None:
    crossed = OptionQuoteRecord(
        timestamp_ns=INSTANT, expiry=EXPIRY, strike=750.0,
        option_type="C", bid=2.0, ask=1.0,
    )
    with pytest.raises(ParityError, match="crossed"):
        build_pairs([crossed])
    with pytest.raises(ParityError, match="strike"):
        build_pairs(
            [
                OptionQuoteRecord(
                    timestamp_ns=INSTANT, expiry=EXPIRY, strike=-1.0,
                    option_type="C", bid=1.0, ask=1.1,
                )
            ]
        )


def test_grouping_and_anchor_and_window_selection() -> None:
    chain = synthetic_chain(discount=0.99, forward=750.0,
                            strikes=[600.0, 700.0, 750.0, 800.0, 900.0])
    grouped = group_by_expiry(chain)
    assert list(grouped) == [EXPIRY]
    assert [pair.strike for pair in grouped[EXPIRY]] == [600.0, 700.0, 750.0, 800.0, 900.0]

    # |C_mid - P_mid| = |D (F - K)| is minimized at the strike nearest F.
    assert forward_anchor_strike(chain) == 750.0
    window = select_strike_window(chain, anchor=750.0, relative_half_width=0.10)
    assert [pair.strike for pair in window] == [700.0, 750.0, 800.0]


def test_pairs_are_ordered_deterministically() -> None:
    forward_order = build_pairs(records((INSTANT, "C", 800.0), (INSTANT, "P", 800.0),
                                        (INSTANT, "C", 700.0), (INSTANT, "P", 700.0)))
    reverse_order = build_pairs(records((INSTANT, "P", 700.0), (INSTANT, "C", 800.0),
                                        (INSTANT, "P", 800.0), (INSTANT, "C", 700.0)))
    assert [pair.strike for pair in forward_order] == [700.0, 800.0]
    assert forward_order == reverse_order


# ---------------------------------------------------------------------------
# Weighted fitting
# ---------------------------------------------------------------------------


def test_weighted_fit_is_deterministic_and_order_independent() -> None:
    chain = synthetic_chain(discount=0.99, forward=750.0,
                            strikes=[700.0, 725.0, 750.0, 775.0, 800.0])
    first = fit(chain, method=METHOD_WEIGHTED)
    second = fit(list(reversed(chain)), method=METHOD_WEIGHTED)
    assert first.discount_factor == second.discount_factor
    assert first.forward == second.forward


def test_weights_are_the_inverse_square_of_the_combined_width() -> None:
    pairs = [
        synthetic_pair(700.0, discount=0.99, forward=750.0, half_width=0.01),
        synthetic_pair(800.0, discount=0.99, forward=750.0, half_width=0.10),
    ]
    weights = bidask_weights(pairs)
    assert weights[0] == pytest.approx(1.0 / (0.04**2))
    assert weights[1] == pytest.approx(1.0 / (0.40**2))
    assert weights[0] > weights[1]


def test_weighting_pulls_the_fit_toward_the_tightest_quotes() -> None:
    """A wide, displaced quote moves the unweighted fit more than the weighted one."""
    chain = synthetic_chain(discount=0.99, forward=750.0,
                            strikes=[700.0, 725.0, 750.0, 775.0])
    chain.append(
        synthetic_pair(800.0, discount=0.99, forward=750.0, half_width=2.0, call_shift=3.0)
    )
    unweighted = fit(chain, method=METHOD_UNWEIGHTED)
    weighted = fit(chain, method=METHOD_WEIGHTED)
    assert abs(weighted.discount_factor - 0.99) < abs(unweighted.discount_factor - 0.99)


def test_a_zero_width_pair_cannot_carry_an_inverse_width_weight() -> None:
    locked = ParityPair(
        timestamp_ns=INSTANT, expiry=EXPIRY, strike=750.0,
        call_bid=12.0, call_ask=12.0, put_bid=9.0, put_ask=9.0,
    )
    with pytest.raises(ParityError, match="non-positive combined bid-ask"):
        bidask_weights([locked])

    chain = synthetic_chain(discount=0.99, forward=750.0,
                            strikes=[700.0, 725.0, 750.0])
    result = fit([*chain, locked], method=METHOD_WEIGHTED)
    assert DIAGNOSTIC_NON_POSITIVE_WEIGHT in result.diagnostics
    assert not result.valid


# ---------------------------------------------------------------------------
# Singular and invalid fits
# ---------------------------------------------------------------------------


def test_identical_strikes_are_a_singular_design() -> None:
    pairs = [synthetic_pair(750.0, discount=0.99, forward=750.0) for _ in range(4)]
    # Pairing would reject duplicates, so build the fit directly.
    result = fit_parity(
        pairs,
        method=METHOD_UNWEIGHTED,
        tau_years=0.25,
        relative_half_width=0.2,
        anchor_strike=750.0,
        minimum_pairs=4,
        condition_number_warning=1.0e8,
        minimum_rate_tau_years=1.0 / 365.0,
        is_zero_dte=False,
    )
    assert DIAGNOSTIC_SINGULAR in result.diagnostics
    assert not result.valid
    assert math.isnan(result.discount_factor)
    assert result.rate is None


def test_insufficient_pairs_is_its_own_diagnostic() -> None:
    result = fit(synthetic_chain(discount=0.99, forward=750.0, strikes=[700.0, 750.0]))
    assert result.diagnostics == (DIAGNOSTIC_INSUFFICIENT_PAIRS,)
    assert not result.valid


def test_ill_conditioning_warns_without_invalidating() -> None:
    """A poorly conditioned design is flagged; the fit is still published."""
    chain = synthetic_chain(discount=0.99, forward=750.0,
                            strikes=[750.0, 750.000001, 750.000002, 750.000003])
    result = fit_parity(
        chain,
        method=METHOD_UNWEIGHTED,
        tau_years=0.25,
        relative_half_width=0.2,
        anchor_strike=750.0,
        minimum_pairs=4,
        condition_number_warning=1.0e6,
        minimum_rate_tau_years=1.0 / 365.0,
        is_zero_dte=False,
    )
    assert DIAGNOSTIC_ILL_CONDITIONED in result.diagnostics
    assert result.valid, "conditioning is a warning, not an invalidation"
    assert result.condition_number > 1.0e6


def test_centering_beats_the_raw_design_on_conditioning() -> None:
    line = fit_line([700.0, 725.0, 750.0, 775.0], [10.0, 5.0, 0.0, -5.0])
    assert line.condition_number_centered < line.condition_number
    assert line.slope == pytest.approx(-0.2)


def test_fit_line_rejects_malformed_input() -> None:
    with pytest.raises(ParityError, match="length mismatch"):
        fit_line([1.0, 2.0], [1.0])
    with pytest.raises(ParityError, match="no points"):
        fit_line([], [])
    with pytest.raises(ParityError, match="positive"):
        fit_line([1.0, 2.0], [1.0, 2.0], [1.0, 0.0])


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def test_act365_year_fraction_is_calendar_based() -> None:
    start = dt.datetime(2026, 6, 17, 14, 30, tzinfo=dt.UTC)
    end = start + dt.timedelta(days=365)
    assert year_fraction_act365(start, end) == pytest.approx(1.0)
    assert year_fraction_act365(start, start + dt.timedelta(days=1)) == pytest.approx(1.0 / 365.0)


def test_year_fraction_requires_aware_instants() -> None:
    naive = dt.datetime(2026, 6, 17, 14, 30)
    with pytest.raises(ParityError, match="timezone-aware"):
        year_fraction_act365(naive, naive + dt.timedelta(days=1))


def test_expiry_time_assumption_shifts_tau_by_the_assumed_hours() -> None:
    """The assumed local expiry instant is what makes tau well defined.

    The archive stamps expiration at midnight UTC, so an implementation that
    took the field literally would price a PM-settled contract as expiring
    sixteen hours early. The difference is negligible at six months and is
    whole percentage points of annualized rate at one day, which is why short
    expiries are excluded from rate interpretation.
    """
    snapshot = dt.datetime(2026, 6, 17, 14, 30, tzinfo=dt.UTC)
    expiry = dt.date(2026, 6, 18)
    midnight = dt.datetime.combine(expiry, dt.time(0, 0), tzinfo=dt.UTC)
    assumed = dt.datetime.combine(expiry, dt.time(16, 0), tzinfo=NEW_YORK).astimezone(dt.UTC)
    tau_midnight = year_fraction_act365(snapshot, midnight)
    tau_assumed = year_fraction_act365(snapshot, assumed)
    assert tau_assumed > tau_midnight
    assert (tau_assumed - tau_midnight) * 365.0 * 24.0 == pytest.approx(20.0)

    discount = 0.99995
    rate_gap = abs(-math.log(discount) / tau_assumed + math.log(discount) / tau_midnight)
    assert rate_gap > 0.005, "a one-day tau is highly sensitive to the expiry assumption"


def test_zero_dte_and_short_tau_withhold_the_rate_but_keep_D_and_F() -> None:
    chain = synthetic_chain(discount=0.99999, forward=750.0,
                            strikes=[700.0, 725.0, 750.0, 775.0])
    zero_dte = fit_parity(
        chain, method=METHOD_UNWEIGHTED, tau_years=0.0006,
        relative_half_width=0.2, anchor_strike=750.0, minimum_pairs=4,
        condition_number_warning=1.0e8, minimum_rate_tau_years=1.0 / 365.0,
        is_zero_dte=True,
    )
    assert zero_dte.rate is None
    assert zero_dte.rate_withheld_reason == RATE_WITHHELD_ZERO_DTE
    assert zero_dte.valid and zero_dte.discount_factor == pytest.approx(0.99999)

    short = fit_parity(
        chain, method=METHOD_UNWEIGHTED, tau_years=0.5 / 365.0,
        relative_half_width=0.2, anchor_strike=750.0, minimum_pairs=4,
        condition_number_warning=1.0e8, minimum_rate_tau_years=1.0 / 365.0,
        is_zero_dte=False,
    )
    assert short.rate is None
    assert short.rate_withheld_reason == RATE_WITHHELD_SHORT_TAU


# ---------------------------------------------------------------------------
# The American diagnostic
# ---------------------------------------------------------------------------


def test_european_chain_gives_a_flat_f_tilde_and_the_expected_residual_sign() -> None:
    """With exact parity, F_tilde is constant and Q = S - D F.

    A spot above ``D F`` -- the ordinary situation for a dividend-paying
    underlying whose forward sits below the compounded spot -- gives a strictly
    positive residual.
    """
    discount, forward, spot = 0.99, 750.0, 745.0
    chain = synthetic_chain(discount=discount, forward=forward,
                            strikes=[700.0, 725.0, 750.0, 775.0, 800.0])
    result = american_carry_residual(chain, spot=spot, discount_factor=discount,
                                     minimum_strikes=5)
    assert result.valid
    assert result.median_f_tilde == pytest.approx(forward, rel=1e-12)
    assert result.f_tilde_iqr == pytest.approx(0.0, abs=1e-9)
    assert result.strike_slope == pytest.approx(0.0, abs=1e-12)
    assert result.carry_residual == pytest.approx(spot - discount * forward, rel=1e-12)
    assert result.carry_residual > 0.0


def test_residual_is_negative_when_the_discounted_forward_exceeds_spot() -> None:
    discount, forward, spot = 0.99, 800.0, 745.0
    chain = synthetic_chain(discount=discount, forward=forward,
                            strikes=[750.0, 775.0, 800.0, 825.0, 850.0])
    result = american_carry_residual(chain, spot=spot, discount_factor=discount,
                                     minimum_strikes=5)
    assert result.carry_residual < 0.0


def test_early_exercise_contamination_produces_a_strike_dependent_f_tilde() -> None:
    """An early-exercise premium on the in-the-money leg tilts F_tilde in K.

    A deep in-the-money American put on a physically settled underlying carries
    an exercise premium that grows with how far in the money it is. Adding that
    premium to the put alone -- as early exercise does -- lowers ``C - P`` most
    at the highest strikes, which is exactly a downward slope in F_tilde.
    """
    discount, forward = 0.99, 750.0
    strikes = [700.0, 725.0, 750.0, 775.0, 800.0]
    chain = [
        synthetic_pair(
            strike,
            discount=discount,
            forward=forward,
            put_shift=max(0.0, 0.02 * (strike - forward)),
        )
        for strike in strikes
    ]
    contaminated = american_carry_residual(chain, spot=745.0, discount_factor=discount,
                                           minimum_strikes=5)
    clean = american_carry_residual(
        synthetic_chain(discount=discount, forward=forward, strikes=strikes),
        spot=745.0, discount_factor=discount, minimum_strikes=5,
    )
    assert contaminated.strike_slope < -1.0e-3
    assert abs(contaminated.strike_slope) > 100.0 * abs(clean.strike_slope)
    assert contaminated.f_tilde_iqr > clean.f_tilde_iqr
    assert contaminated.f_tilde_max - contaminated.f_tilde_min > 1.0


def test_residual_reports_rather_than_clamps_an_invalid_discount_factor() -> None:
    chain = synthetic_chain(discount=0.99, forward=750.0,
                            strikes=[700.0, 725.0, 750.0, 775.0, 800.0])
    result = american_carry_residual(chain, spot=745.0, discount_factor=-0.5,
                                     minimum_strikes=5)
    assert not result.valid
    assert DIAGNOSTIC_NON_POSITIVE_DISCOUNT in result.diagnostics
    assert math.isnan(result.carry_residual)


def test_residual_requires_a_minimum_number_of_strikes() -> None:
    chain = synthetic_chain(discount=0.99, forward=750.0, strikes=[700.0, 750.0])
    result = american_carry_residual(chain, spot=745.0, discount_factor=0.99,
                                     minimum_strikes=5)
    assert not result.valid
    assert DIAGNOSTIC_INSUFFICIENT_PAIRS in result.diagnostics


def test_median_is_robust_to_a_single_contaminated_strike() -> None:
    """The median is used precisely so one exercised leg cannot move the level."""
    discount, forward = 0.99, 750.0
    strikes = [700.0, 725.0, 750.0, 775.0, 800.0]
    chain = synthetic_chain(discount=discount, forward=forward, strikes=strikes)
    chain[0] = synthetic_pair(700.0, discount=discount, forward=forward, put_shift=25.0)
    result = american_carry_residual(chain, spot=745.0, discount_factor=discount,
                                     minimum_strikes=5)
    assert result.median_f_tilde == pytest.approx(forward, rel=1e-12)
    # Raising the put lowers C - P and so lowers F_tilde at that strike; the
    # mean follows it and the median does not.
    assert result.mean_f_tilde < result.median_f_tilde


@pytest.mark.parametrize("spot", [0.0, -1.0, math.inf, math.nan])
def test_an_unusable_spot_is_named_rather_than_absorbed(spot: float) -> None:
    chain = synthetic_chain(discount=0.99, forward=750.0,
                            strikes=[700.0, 725.0, 750.0, 775.0, 800.0])
    result = american_carry_residual(chain, spot=spot, discount_factor=0.99,
                                     minimum_strikes=5)
    assert not result.valid
    assert DIAGNOSTIC_NON_POSITIVE_SPOT in result.diagnostics
    assert math.isnan(result.carry_residual)
