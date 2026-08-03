"""The Task 9B configuration must reject anything that would hide an assumption.

The interesting cases are not typos. They are configurations that parse fine and
would quietly change what the study means: pairing across minutes, interpreting a
zero-DTE rate, labelling an assumed expiry instant as observed, or dropping one
of the names the American residual may never be given.
"""

from __future__ import annotations

import copy
import tomllib
from pathlib import Path
from typing import Any

import pytest
from differentiable_pricing.market.config import ConfigError
from differentiable_pricing.market.state_config import (
    SCHEMA_VERSION,
    load_state_config,
    parse_state_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs/market_state_reconstruction_v1.toml"


@pytest.fixture
def document() -> dict[str, Any]:
    with CONFIG_PATH.open("rb") as stream:
        return tomllib.load(stream)


def parse(document: dict[str, Any]):
    return parse_state_config(document, source_name="test.toml", source_sha256="0" * 64)


def test_the_checked_in_configuration_loads_and_hashes_itself() -> None:
    config = load_state_config(CONFIG_PATH)
    assert config.schema_version == SCHEMA_VERSION
    assert len(config.source_sha256) == 64
    assert config.settings_status == "exploratory_pilot"
    assert not config.settings_are_frozen, (
        "these settings were chosen after the archive was observed and must not "
        "claim to be frozen"
    )


def test_declared_snapshots_windows_and_methods_are_the_intended_ones(
    document: dict[str, Any]
) -> None:
    config = parse(document)
    assert [moment.isoformat() for moment in config.snapshots.times_local] == [
        "10:30:00",
        "13:00:00",
        "15:45:00",
    ]
    assert config.strike_windows.relative_half_widths == (0.10, 0.15, 0.20)
    assert set(config.fit.methods) == {
        "unweighted_least_squares",
        "weighted_least_squares_bidask",
    }
    assert config.time.day_count == "ACT/365F"
    assert config.time.expiry_time_local.isoformat() == "16:00:00"


def test_unknown_keys_are_rejected(document: dict[str, Any]) -> None:
    document["unexpected"] = 1
    with pytest.raises(ConfigError, match="unknown keys"):
        parse(document)


def test_pairing_across_minutes_is_refused(document: dict[str, Any]) -> None:
    document["pairing"]["require_same_minute"] = False
    with pytest.raises(ConfigError, match="simultaneous prices"):
        parse(document)


def test_zero_dte_rate_interpretation_is_refused(document: dict[str, Any]) -> None:
    document["time"]["zero_dte_rate_interpretation"] = True
    with pytest.raises(ConfigError, match="quote grid"):
        parse(document)


def test_an_expiry_instant_may_not_be_labelled_observed(document: dict[str, Any]) -> None:
    document["time"]["expiry_time_status"] = "observed"
    with pytest.raises(ConfigError, match="no settlement metadata"):
        parse(document)


def test_a_day_count_that_is_not_implemented_is_refused(document: dict[str, Any]) -> None:
    document["time"]["day_count"] = "ACT/360"
    with pytest.raises(ConfigError, match="day count"):
        parse(document)


def test_every_forbidden_name_for_the_residual_is_required(document: dict[str, Any]) -> None:
    for index in range(len(document["american"]["forbidden_names"])):
        trimmed = copy.deepcopy(document)
        del trimmed["american"]["forbidden_names"][index]
        with pytest.raises(ConfigError, match="forbidden_names is missing"):
            parse(trimmed)


def test_the_official_spy_distribution_schedule_is_pinned(document: dict[str, Any]) -> None:
    """The issuer's published dates, cited by title and URL."""
    schedule = parse(document).american.ex_date_schedule
    assert schedule.status == "officially_scheduled"
    assert [value.isoformat() for value in schedule.ex_dates] == ["2026-06-18"]
    assert [value.isoformat() for value in schedule.record_dates] == ["2026-06-18"]
    assert [value.isoformat() for value in schedule.payable_dates] == ["2026-07-31"]
    assert schedule.source_title == "SPDR Dividend and Capital Gain Distribution Schedule"
    assert schedule.source_publisher == "State Street Global Advisors"
    assert schedule.source_url.endswith("SPDR_Dividend_Distribution_Schedule.pdf")
    assert set(schedule.applies_to) == {"SPY", "MDY"}


def test_the_schedule_verifies_dates_only(document: dict[str, Any]) -> None:
    """A published calendar states dates. It states no amount and measures no effect."""
    schedule = parse(document).american.ex_date_schedule
    assert schedule.dates_are_officially_scheduled is True
    assert schedule.amount_is_verified is False
    assert schedule.economic_effect_is_verified is False
    summary = parse(document).summary()["ex_date_schedule"]
    assert summary["amount_status"] == "not_verified"
    assert summary["economic_effect_status"] == "not_verified"
    assert "SCHEDULED DATES only" in summary["scope"]


def test_a_schedule_status_other_than_officially_scheduled_is_refused(
    document: dict[str, Any]
) -> None:
    document["american"]["ex_date_schedule"]["status"] = "observed"
    with pytest.raises(ConfigError, match="officially_scheduled"):
        parse(document)


@pytest.mark.parametrize("field", ["amount_status", "economic_effect_status"])
def test_the_schedule_may_not_claim_a_verified_amount_or_effect(
    document: dict[str, Any], field: str
) -> None:
    document["american"]["ex_date_schedule"][field] = "verified"
    with pytest.raises(ConfigError, match="must be 'not_verified'"):
        parse(document)


def test_the_schedule_dates_must_line_up(document: dict[str, Any]) -> None:
    document["american"]["ex_date_schedule"]["record_dates"] = []
    with pytest.raises(ConfigError, match="non-empty list of dates"):
        parse(document)
    document["american"]["ex_date_schedule"]["record_dates"] = ["2026-06-18", "2026-09-17"]
    with pytest.raises(ConfigError, match="equal length"):
        parse(document)


def test_a_payable_date_before_its_ex_date_is_refused(document: dict[str, Any]) -> None:
    document["american"]["ex_date_schedule"]["payable_dates"] = ["2026-06-01"]
    with pytest.raises(ConfigError, match="precedes its ex-date"):
        parse(document)


def test_an_inexact_expiry_match_is_refused(document: dict[str, Any]) -> None:
    document["american"]["require_exact_expiry_match"] = False
    with pytest.raises(ConfigError, match="neighbouring expiry"):
        parse(document)


def test_the_two_option_products_must_differ(document: dict[str, Any]) -> None:
    document["pairing"]["american_option_product"] = document["pairing"][
        "european_option_product"
    ]
    with pytest.raises(ConfigError, match="must differ"):
        parse(document)


def test_an_unimplemented_fit_method_is_refused(document: dict[str, Any]) -> None:
    document["fit"]["methods"] = ["total_least_squares"]
    with pytest.raises(ConfigError, match="not implemented"):
        parse(document)


def test_a_two_point_minimum_is_refused(document: dict[str, Any]) -> None:
    document["fit"]["minimum_pairs_per_fit"] = 2
    with pytest.raises(ConfigError, match="at least 3"):
        parse(document)


def test_a_malformed_digest_is_refused(document: dict[str, Any]) -> None:
    document["provenance"]["feasibility_report_sha256"] = "not-a-digest"
    with pytest.raises(ConfigError, match="SHA-256"):
        parse(document)


def test_strike_windows_must_be_relative_fractions(document: dict[str, Any]) -> None:
    document["strike_windows"]["relative_half_widths"] = [0.1, 1.5]
    with pytest.raises(ConfigError, match="strictly between 0 and 1"):
        parse(document)


def test_summary_states_that_settlement_metadata_is_absent(document: dict[str, Any]) -> None:
    summary = parse(document).summary()
    assert "absent from the archive" in summary["settlement_metadata"]
    assert summary["expiry_time_status"] == "assumed"
    assert summary["settings_are_frozen"] is False


def test_a_path_escaping_the_project_root_is_refused(document: dict[str, Any]) -> None:
    """A configured path is joined to the repository root, so `..` must not parse."""
    document["provenance"]["processed_root"] = "../escape"
    with pytest.raises(ConfigError, match="relative path without"):
        parse(document)
    document["provenance"]["processed_root"] = "/absolute/escape"
    with pytest.raises(ConfigError, match="relative path without"):
        parse(document)


def test_an_output_filename_may_not_carry_a_directory(document: dict[str, Any]) -> None:
    document["output"]["report_filename"] = "nested/report.json"
    with pytest.raises(ConfigError, match="bare file name"):
        parse(document)
