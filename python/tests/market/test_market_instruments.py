"""Contracts must come from definitions, and disagreements must not be repaired.

Identity is ``(publisher_id, instrument_id)``, definitions are applied in
effective-time order with delete actions honoured, and a well-formed symbol with
an unexpected root is a non-standard *contract*, not a parse failure.
"""

from __future__ import annotations

import datetime as dt

import pytest
from differentiable_pricing.market.instruments import (
    REFERENCE_SELECTION_NEAREST_OUTRIGHT,
    REFERENCE_SELECTION_SINGLE,
    InstrumentError,
    InstrumentKey,
    build_future_universe,
    build_option_universe,
    future_definition_rows,
    is_outright_future,
    option_definition_rows,
    parse_osi_symbol,
    resolve_option_definition,
)
from market_fixtures import (
    FakeDefinition,
    future_definition,
    option_definition,
    osi,
    scaled,
)

SESSION = dt.date(2026, 7, 30)


class TestOsiParsing:
    def test_parses_a_standard_symbol(self) -> None:
        parsed = parse_osi_symbol("SPY   270115C00620000")
        assert parsed.root == "SPY"
        assert parsed.expiration_date == dt.date(2027, 1, 15)
        assert parsed.option_type == "C"
        assert parsed.strike == 620.0

    def test_parses_a_fractional_strike(self) -> None:
        parsed = parse_osi_symbol("XSP   260930P00512500")
        assert parsed.option_type == "P"
        assert parsed.strike == 512.5

    def test_parses_a_six_character_root(self) -> None:
        parsed = parse_osi_symbol(osi("ABCDEF", dt.date(2026, 12, 18), "C", 1.5))
        assert parsed.root == "ABCDEF"
        assert parsed.strike == 1.5

    def test_parses_an_adjusted_root(self) -> None:
        parsed = parse_osi_symbol(osi("SPY1", dt.date(2026, 9, 18), "C", 600.0))
        assert parsed.root == "SPY1"
        assert parsed.strike == 600.0

    @pytest.mark.parametrize(
        "symbol",
        [
            "SPY   270115X00620000",  # not a call or a put
            "SPY   271315C00620000",  # impossible month
            "SPY   270230C00620000",  # impossible day
            "SPY   270115C0062000",  # strike too short
            "SPY270115C00620000000",  # wrong overall layout
            "",
        ],
    )
    def test_rejects_malformed_symbols(self, symbol: str) -> None:
        with pytest.raises(InstrumentError):
            parse_osi_symbol(symbol)

    def test_rejects_a_non_string(self) -> None:
        with pytest.raises(InstrumentError):
            parse_osi_symbol(620.0)  # type: ignore[arg-type]


class TestOptionResolution:
    def test_resolves_a_consistent_definition(self) -> None:
        record = option_definition(instrument_id=7, strike=612.5, right="P")
        contract = resolve_option_definition(record, product_root="SPY")
        assert contract.instrument_id == 7
        assert contract.option_type == "P"
        assert contract.strike == pytest.approx(612.5)
        assert contract.expiration_date == dt.date(2026, 9, 18)
        assert contract.is_call is False
        assert contract.is_standard_root is True
        assert contract.time_to_maturity_days(SESSION) == 50

    def test_identity_carries_the_publisher(self) -> None:
        contract = resolve_option_definition(
            option_definition(instrument_id=7, publisher_id=31)
        )
        assert contract.key == InstrumentKey(31, 7)

    def test_absent_multiplier_and_exercise_style_stay_absent(self) -> None:
        contract = resolve_option_definition(option_definition(instrument_id=1))
        assert contract.contract_multiplier is None
        assert contract.exercise_style is None
        assert contract.unit_of_measure_quantity is None

    def test_exercise_style_is_read_when_a_cfi_code_carries_it(self) -> None:
        american = resolve_option_definition(option_definition(instrument_id=1, cfi="OCASPS"))
        european = resolve_option_definition(option_definition(instrument_id=2, cfi="OPEICS"))
        assert american.exercise_style == "american"
        assert european.exercise_style == "european"

    def test_strike_disagreement_is_a_failure_not_a_repair(self) -> None:
        record = option_definition(instrument_id=1, strike=600.0)
        record.strike_price = scaled(601.0)
        with pytest.raises(InstrumentError, match="encodes strike"):
            resolve_option_definition(record)

    def test_expiry_disagreement_is_a_failure(self) -> None:
        record = option_definition(instrument_id=1)
        record.expiration = int(
            dt.datetime(2026, 9, 19, 20, tzinfo=dt.UTC).timestamp()
        ) * 1_000_000_000
        with pytest.raises(InstrumentError, match="encodes expiry"):
            resolve_option_definition(record)

    def test_right_disagreement_is_a_failure(self) -> None:
        record = option_definition(instrument_id=1, right="C")
        record.instrument_class = "P"
        with pytest.raises(InstrumentError, match="definition class"):
            resolve_option_definition(record)

    def test_absent_strike_is_a_failure(self) -> None:
        record = option_definition(instrument_id=1)
        record.strike_price = 2**63 - 1
        with pytest.raises(InstrumentError, match="no strike price"):
            resolve_option_definition(record)

    def test_a_spread_class_is_not_an_outright_option(self) -> None:
        record = option_definition(instrument_id=1)
        record.instrument_class = "T"
        with pytest.raises(InstrumentError, match="not an outright option"):
            resolve_option_definition(record)


class TestNonStandardRoots:
    """An adjusted contract is an observation about the contract, not a decode error."""

    def test_an_unexpected_root_resolves_and_is_marked_non_standard(self) -> None:
        contract = resolve_option_definition(
            option_definition(instrument_id=1, root="SPY1"), product_root="SPY"
        )
        assert contract.is_standard_root is False
        assert contract.root == "SPY1"
        assert contract.strike == pytest.approx(600.0)

    def test_a_non_standard_root_is_not_counted_as_a_resolution_failure(self) -> None:
        universe = build_option_universe(
            [
                option_definition(instrument_id=1, root="SPY"),
                option_definition(instrument_id=2, root="SPY1"),
            ],
            product_root="SPY",
        )
        assert universe.failures == []
        assert len(universe.contracts) == 2
        assert len(universe.standard()) == 1
        assert len(universe.non_standard()) == 1
        assert next(iter(universe.non_standard().values())).root == "SPY1"

    def test_a_genuine_parse_failure_is_still_a_failure(self) -> None:
        broken = option_definition(instrument_id=3, root="SPY")
        broken.raw_symbol = "not-an-osi-symbol"
        universe = build_option_universe([broken], product_root="SPY")
        assert len(universe.failures) == 1
        assert universe.failures[0].reason == "option_definition_unresolvable"
        assert universe.contracts == {}

    def test_definition_rows_carry_the_standard_root_flag(self) -> None:
        universe = build_option_universe(
            [option_definition(instrument_id=1, root="SPY7")], product_root="SPY"
        )
        rows = list(option_definition_rows(universe.contracts.values()))
        assert rows[0]["is_standard_root"] is False
        assert rows[0]["root"] == "SPY7"


class TestEffectiveTimeAndLifecycle:
    def test_a_later_effective_definition_wins(self) -> None:
        first = option_definition(instrument_id=5, strike=600.0, ts_recv=100)
        second = option_definition(instrument_id=5, strike=605.0, ts_recv=200)
        universe = build_option_universe([first, second])
        assert universe.contracts[InstrumentKey(30, 5)].strike == pytest.approx(605.0)
        assert universe.counters.superseded_definitions == 1

    def test_file_order_does_not_override_effective_time(self) -> None:
        """The corrected definition must win even when it arrives first in the file."""
        newer = option_definition(instrument_id=5, strike=605.0, ts_recv=200)
        older = option_definition(instrument_id=5, strike=600.0, ts_recv=100)
        universe = build_option_universe([newer, older])
        assert universe.contracts[InstrumentKey(30, 5)].strike == pytest.approx(605.0)
        assert universe.counters.out_of_order_definitions == 1
        assert universe.counters.superseded_definitions == 0

    def test_a_delete_action_retires_the_instrument(self) -> None:
        live = option_definition(instrument_id=5, strike=600.0, ts_recv=100)
        retired = option_definition(
            instrument_id=5, strike=600.0, ts_recv=200, security_update_action="D"
        )
        universe = build_option_universe([live, retired])
        assert universe.contracts == {}
        assert universe.counters.deleted_definitions == 1

    def test_a_later_add_resurrects_a_deleted_instrument(self) -> None:
        retired = option_definition(
            instrument_id=5, strike=600.0, ts_recv=100, security_update_action="D"
        )
        readded = option_definition(instrument_id=5, strike=610.0, ts_recv=200)
        universe = build_option_universe([retired, readded])
        assert universe.contracts[InstrumentKey(30, 5)].strike == pytest.approx(610.0)

    def test_an_unparseable_delete_still_retires_the_instrument(self) -> None:
        live = option_definition(instrument_id=5, strike=600.0, ts_recv=100)
        retired = option_definition(instrument_id=5, ts_recv=200, security_update_action="D")
        retired.raw_symbol = "garbage"
        universe = build_option_universe([live, retired])
        assert universe.contracts == {}
        assert universe.failures == []

    def test_the_same_id_from_two_publishers_does_not_collide(self) -> None:
        first = option_definition(instrument_id=5, strike=600.0, publisher_id=30)
        second = option_definition(instrument_id=5, strike=650.0, publisher_id=31)
        universe = build_option_universe([first, second])
        assert set(universe.contracts) == {InstrumentKey(30, 5), InstrumentKey(31, 5)}
        assert universe.counters.superseded_definitions == 0

    def test_rows_are_emitted_in_identity_order(self) -> None:
        universe = build_option_universe(
            [
                option_definition(instrument_id=9, strike=610.0),
                option_definition(instrument_id=2, strike=600.0),
            ]
        )
        ids = [row["instrument_id"] for row in option_definition_rows(universe.contracts.values())]
        assert ids == [2, 9]


class TestFuturesFiltering:
    def test_outrights_and_multi_leg_instruments_are_separate_views(self) -> None:
        records = [
            future_definition(instrument_id=1, raw_symbol="ESU6", instrument_class="F"),
            future_definition(instrument_id=2, raw_symbol="ESZ6", instrument_class="F"),
            future_definition(instrument_id=3, raw_symbol="ESU6-ESZ6", instrument_class="S"),
            future_definition(instrument_id=4, raw_symbol="ESZ6-ESH7", instrument_class="S"),
            future_definition(instrument_id=5, raw_symbol="ES-MIX", instrument_class="M"),
        ]
        universe = build_future_universe(records)
        assert universe.definition_records == 5
        assert len(universe.contracts) == 5
        assert sorted(item.raw_symbol for item in universe.outrights().values()) == [
            "ESU6",
            "ESZ6",
        ]
        assert sorted(item.raw_symbol for item in universe.multi_leg().values()) == [
            "ES-MIX",
            "ESU6-ESZ6",
            "ESZ6-ESH7",
        ]

    def test_leg_count_is_not_the_discriminator(self) -> None:
        """CME spread definitions here report zero legs, so class must decide."""
        spread = future_definition(
            instrument_id=3, raw_symbol="ESU6-ESZ6", instrument_class="S", leg_count=0
        )
        assert getattr(spread, "leg_count", 0) == 0
        assert is_outright_future(spread) is False

    def test_outright_is_identified_directly(self) -> None:
        outright = future_definition(instrument_id=1, raw_symbol="ESU6", instrument_class="F")
        assert is_outright_future(outright) is True

    def test_a_definition_without_a_class_is_a_failure(self) -> None:
        broken = FakeDefinition(instrument_id=1, raw_symbol="ESU6", instrument_class="")
        universe = build_future_universe([broken])
        assert universe.contracts == {}
        assert universe.failures[0].reason == "future_definition_unresolvable"

    def test_expired_outrights_are_excluded_from_the_unexpired_view(self) -> None:
        universe = build_future_universe(
            [
                future_definition(
                    instrument_id=1, raw_symbol="ESM6", instrument_class="F",
                    expiry=dt.date(2026, 6, 18),
                ),
                future_definition(
                    instrument_id=2, raw_symbol="ESU6", instrument_class="F",
                    expiry=dt.date(2026, 9, 18),
                ),
            ]
        )
        assert len(universe.outrights()) == 2
        assert sorted(
            item.raw_symbol for item in universe.unexpired_outrights(SESSION).values()
        ) == ["ESU6"]

    def test_rows_carry_the_outright_flag(self) -> None:
        universe = build_future_universe(
            [
                future_definition(instrument_id=2, raw_symbol="ESU6-ESZ6", instrument_class="S"),
                future_definition(instrument_id=1, raw_symbol="ESU6", instrument_class="F"),
            ]
        )
        rows = list(future_definition_rows(universe.contracts.values()))
        assert [row["instrument_id"] for row in rows] == [1, 2]
        assert [row["is_outright"] for row in rows] == [True, False]


class TestReferenceSelection:
    """A spread must never become the futures reference."""

    def universe(self):
        return build_future_universe(
            [
                future_definition(
                    instrument_id=10, raw_symbol="ESM6", instrument_class="F",
                    expiry=dt.date(2026, 6, 18),
                ),
                future_definition(
                    instrument_id=11, raw_symbol="ESU6", instrument_class="F",
                    expiry=dt.date(2026, 9, 18),
                ),
                future_definition(
                    instrument_id=12, raw_symbol="ESZ6", instrument_class="F",
                    expiry=dt.date(2026, 12, 18),
                ),
                future_definition(
                    instrument_id=13, raw_symbol="ESM6-ESU6", instrument_class="S",
                    expiry=dt.date(2026, 6, 18),
                ),
            ]
        )

    def test_the_nearest_unexpired_outright_wins(self) -> None:
        chosen, detail = self.universe().select_reference(
            REFERENCE_SELECTION_NEAREST_OUTRIGHT, dt.date(2026, 6, 17)
        )
        assert chosen is not None
        assert chosen.raw_symbol == "ESM6"
        assert chosen.is_outright is True
        assert "ESM6" in detail

    def test_a_spread_is_never_selected_even_when_it_expires_first(self) -> None:
        universe = build_future_universe(
            [
                future_definition(
                    instrument_id=13, raw_symbol="ESM6-ESU6", instrument_class="S",
                    expiry=dt.date(2026, 6, 18),
                ),
                future_definition(
                    instrument_id=11, raw_symbol="ESU6", instrument_class="F",
                    expiry=dt.date(2026, 9, 18),
                ),
            ]
        )
        chosen, _ = universe.select_reference(
            REFERENCE_SELECTION_NEAREST_OUTRIGHT, dt.date(2026, 6, 17)
        )
        assert chosen is not None
        assert chosen.raw_symbol == "ESU6"

    def test_an_expired_front_contract_rolls_to_the_next(self) -> None:
        chosen, _ = self.universe().select_reference(
            REFERENCE_SELECTION_NEAREST_OUTRIGHT, dt.date(2026, 6, 19)
        )
        assert chosen is not None
        assert chosen.raw_symbol == "ESU6"

    def test_no_eligible_outright_yields_no_reference_and_an_explanation(self) -> None:
        chosen, detail = self.universe().select_reference(
            REFERENCE_SELECTION_NEAREST_OUTRIGHT, dt.date(2030, 1, 1)
        )
        assert chosen is None
        assert "no outright contract expires" in detail

    def test_single_instrument_requires_exactly_one(self) -> None:
        chosen, detail = self.universe().select_reference(
            REFERENCE_SELECTION_SINGLE, dt.date(2026, 6, 17)
        )
        assert chosen is None
        assert "not defined" in detail

    def test_an_unknown_rule_is_an_error(self) -> None:
        with pytest.raises(InstrumentError, match="unknown reference selection rule"):
            self.universe().select_reference("whatever_quotes_first", dt.date(2026, 6, 17))
