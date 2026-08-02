"""The frozen audit configuration must stay valid, and its loader must stay strict.

This module is the lightweight reproducibility check: it validates the versioned
configuration that ships with the repository without reading a single byte of the
private archive, so it runs identically in CI and on a machine that has the data.

It also pins the declarations that keep the audit honest: the external sources
declared as probes, the tradability predicate, and the structural capability
minima. Those exist so no conclusion can be hardcoded, so the tests assert that
they are present and well-formed rather than merely parseable.
"""

from __future__ import annotations

import copy
import datetime as dt
import tomllib
from pathlib import Path
from typing import Any

import pytest
from differentiable_pricing.market.config import (
    CAPABILITIES,
    SCHEMA_VERSION,
    ConfigError,
    load_market_config,
    parse_market_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "market_feasibility_v1.toml"


@pytest.fixture
def document() -> dict[str, Any]:
    with CONFIG_PATH.open("rb") as stream:
        return tomllib.load(stream)


def parse(document: dict[str, Any]) -> Any:
    return parse_market_config(document, source_name="test.toml", source_sha256="0" * 64)


def product(document: dict[str, Any], key: str) -> dict[str, Any]:
    return next(item for item in document["products"] if item["key"] == key)


class TestFrozenConfiguration:
    def test_the_checked_in_configuration_loads(self) -> None:
        config = load_market_config(CONFIG_PATH)
        assert config.schema_version == SCHEMA_VERSION
        assert len(config.source_sha256) == 64

    def test_it_declares_three_dates_with_one_held_out(self) -> None:
        config = load_market_config(CONFIG_PATH)
        assert [entry.date for entry in config.dates] == [
            dt.date(2026, 6, 17),
            dt.date(2026, 6, 18),
            dt.date(2026, 7, 30),
        ]
        assert [entry.date for entry in config.held_out_dates()] == [dt.date(2026, 7, 30)]

    def test_it_declares_every_acquired_product(self) -> None:
        config = load_market_config(CONFIG_PATH)
        assert {spec.key for spec in config.products} == {
            "spy-options",
            "xsp-options",
            "spy-underlying",
            "es-futures",
            "sr3-futures",
        }
        assert config.product("spy-options").role == "primary_option_chain"
        assert config.product("spy-options").definition_schema == "definition"
        assert config.product("spy-options").quote_schema == "cbbo-1m"

    def test_sr3_declares_no_quote_schema(self) -> None:
        """The archive holds no SR3 quotes; the configuration must not claim any."""
        assert load_market_config(CONFIG_PATH).product("sr3-futures").quote_schema is None

    def test_the_session_grid_is_390_minutes_on_every_declared_date(self) -> None:
        config = load_market_config(CONFIG_PATH)
        for entry in config.dates:
            minutes = config.session_minutes_utc(entry.date)
            assert len(minutes) == 390
            assert minutes[0] - dt.timedelta(minutes=1) == config.session.window_utc(
                entry.date
            )[0]
            assert minutes[-1] == config.session.window_utc(entry.date)[1]

    def test_the_grid_follows_daylight_saving_rather_than_a_fixed_offset(self) -> None:
        config = load_market_config(CONFIG_PATH)
        summer_open, _ = config.session.window_utc(dt.date(2026, 7, 30))
        winter_open, _ = config.session.window_utc(dt.date(2026, 12, 15))
        assert summer_open.hour == 13
        assert winter_open.hour == 14

    def test_output_roots_are_git_ignored_trees(self) -> None:
        config = load_market_config(CONFIG_PATH)
        assert config.processed_root.startswith("data/")
        assert config.report_root.startswith("artifacts/")

    def test_every_synchronization_pair_names_declared_products(self) -> None:
        config = load_market_config(CONFIG_PATH)
        keys = {spec.key for spec in config.products}
        for pair in config.synchronization_pairs:
            assert pair.option_product in keys
            assert pair.reference_product in keys
        primary = [pair for pair in config.synchronization_pairs if pair.role == "primary"]
        assert len(primary) == 1
        assert primary[0].option_product == "spy-options"
        assert primary[0].reference_product == "spy-underlying"

    def test_fred_series_are_declared_as_daily_controls_only(self) -> None:
        config = load_market_config(CONFIG_PATH)
        assert {spec.series_id for spec in config.fred_series} == {
            "SOFR",
            "DGS1MO",
            "DGS3MO",
            "DGS6MO",
            "DGS1",
            "DGS2",
        }
        assert all("daily" in spec.role for spec in config.fred_series)


class TestReferenceAttribution:
    """No product may inherit another product's underlying by accident."""

    def test_the_equity_feed_is_declared_partial_venue_not_nbbo(self) -> None:
        spec = load_market_config(CONFIG_PATH).product("spy-underlying")
        assert spec.feed_class == "partial_venue_consolidated"
        assert spec.is_nbbo_grade is False

    def test_the_american_chain_names_its_own_underlying(self) -> None:
        spec = load_market_config(CONFIG_PATH).product("spy-options")
        assert spec.underlying_product == "spy-underlying"
        assert spec.proxy_underlying_product is None

    def test_the_cash_settled_control_declares_an_explicit_caveated_proxy(self) -> None:
        spec = load_market_config(CONFIG_PATH).product("xsp-options")
        assert spec.underlying_product is None
        assert spec.proxy_underlying_product == "spy-underlying"
        assert spec.proxy_caveat is not None
        assert "not in this archive" in spec.proxy_caveat
        assert "index" in spec.settles_on

    def test_the_futures_reference_declares_an_outright_selection_rule(self) -> None:
        spec = load_market_config(CONFIG_PATH).product("es-futures")
        assert spec.reference_selection == "nearest_unexpired_outright"

    def test_a_quote_product_without_a_feed_class_is_rejected(
        self, document: dict[str, Any]
    ) -> None:
        del product(document, "spy-underlying")["feed_class"]
        with pytest.raises(ConfigError, match="no feed_class"):
            parse(document)

    def test_an_unknown_feed_class_is_rejected(self, document: dict[str, Any]) -> None:
        product(document, "spy-underlying")["feed_class"] = "the_whole_market"
        with pytest.raises(ConfigError, match="feed_class 'the_whole_market' is unknown"):
            parse(document)

    def test_declaring_both_a_real_and_a_proxy_underlying_is_rejected(
        self, document: dict[str, Any]
    ) -> None:
        product(document, "xsp-options")["underlying_product"] = "spy-underlying"
        with pytest.raises(ConfigError, match="one reference attribution, not two"):
            parse(document)

    def test_a_proxy_without_a_caveat_is_rejected(self, document: dict[str, Any]) -> None:
        del product(document, "xsp-options")["proxy_caveat"]
        with pytest.raises(ConfigError, match="without proxy_caveat"):
            parse(document)

    def test_an_underlying_that_is_not_a_product_is_rejected(
        self, document: dict[str, Any]
    ) -> None:
        product(document, "spy-options")["underlying_product"] = "nasdaq"
        with pytest.raises(ConfigError, match="is not a declared product"):
            parse(document)

    def test_a_reference_without_a_selection_rule_is_rejected(
        self, document: dict[str, Any]
    ) -> None:
        del product(document, "spy-underlying")["reference_selection"]
        with pytest.raises(ConfigError, match="declares no reference_selection rule"):
            parse(document)

    def test_an_unknown_selection_rule_is_rejected(self, document: dict[str, Any]) -> None:
        product(document, "es-futures")["reference_selection"] = "first_quote_wins"
        with pytest.raises(ConfigError, match="reference_selection 'first_quote_wins'"):
            parse(document)


class TestDeclaredSources:
    """External inputs must be declared as probes, never hardcoded as absent."""

    def test_every_required_input_is_declared_as_a_probe(self) -> None:
        config = load_market_config(CONFIG_PATH)
        keys = {source.key for source in config.declared_sources}
        assert {
            "discrete_dividend_schedule",
            "corporate_action_and_contract_adjustment_history",
            "securities_lending_borrow_rate",
        } <= keys

    def test_each_probe_names_a_path_a_kind_and_a_capability(self) -> None:
        for source in load_market_config(CONFIG_PATH).declared_sources:
            assert source.path
            assert source.kind in {"file", "directory"}
            assert source.capability in CAPABILITIES
            assert source.requirement in {"required", "recommended"}
            assert source.why
            assert source.candidate_source

    def test_a_source_naming_an_unknown_capability_is_rejected(
        self, document: dict[str, Any]
    ) -> None:
        document["declared_sources"][0]["capability"] = "world_domination"
        with pytest.raises(ConfigError, match="unknown capabilities"):
            parse(document)

    def test_an_unknown_source_kind_is_rejected(self, document: dict[str, Any]) -> None:
        document["declared_sources"][0]["kind"] = "spreadsheet"
        with pytest.raises(ConfigError, match="kind 'spreadsheet' is unknown"):
            parse(document)

    def test_a_repeated_source_key_is_rejected(self, document: dict[str, Any]) -> None:
        document["declared_sources"].append(copy.deepcopy(document["declared_sources"][0]))
        with pytest.raises(ConfigError, match="repeats source key"):
            parse(document)


class TestTradabilityDeclaration:
    def test_structural_requirements_are_declared(self) -> None:
        spec = load_market_config(CONFIG_PATH).tradability
        assert spec.require_positive_sizes is True
        assert spec.require_positive_bid is True
        assert spec.require_positive_spread is True
        assert spec.exclude_vendor_flagged is True

    def test_several_exploratory_spread_thresholds_are_declared(self) -> None:
        """A single value would be an acceptance threshold chosen after the fact.

        The narrower claim these thresholds support is that no capability check
        consults any of them; the session minima make no such claim.
        """
        thresholds = load_market_config(CONFIG_PATH).tradability
        assert len(thresholds.exploratory_relative_spread_thresholds) >= 3
        assert thresholds.exploratory_relative_spread_thresholds == tuple(
            sorted(thresholds.exploratory_relative_spread_thresholds)
        )

    def test_a_single_threshold_is_rejected(self, document: dict[str, Any]) -> None:
        document["tradability"]["exploratory_relative_spread_thresholds"] = [0.05]
        with pytest.raises(ConfigError, match="at least two numbers"):
            parse(document)

    def test_a_non_positive_threshold_is_rejected(self, document: dict[str, Any]) -> None:
        document["tradability"]["exploratory_relative_spread_thresholds"] = [0.0, 0.05]
        with pytest.raises(ConfigError, match="strictly positive"):
            parse(document)

    def test_a_non_boolean_switch_is_rejected(self, document: dict[str, Any]) -> None:
        document["tradability"]["require_positive_sizes"] = "yes"
        with pytest.raises(ConfigError, match="must be a boolean"):
            parse(document)


class TestCapabilityMinima:
    def test_the_replication_minimum_exceeds_the_declared_date_count(self) -> None:
        """Three sessions cannot satisfy a replication protocol under any threshold."""
        config = load_market_config(CONFIG_PATH)
        assert config.capabilities.minimum_sessions > len(config.dates)
        assert config.capabilities.minimum_held_out_sessions > len(config.held_out_dates())

    def test_surface_minima_are_structural_and_positive(self) -> None:
        capabilities = load_market_config(CONFIG_PATH).capabilities
        assert capabilities.minimum_expiries_with_paired_strikes >= 1
        assert capabilities.minimum_paired_strikes_per_expiry >= 2

    def test_the_session_minima_are_declared_provisional_not_structural(self) -> None:
        """They were chosen after this archive was observed and must say so."""
        capabilities = load_market_config(CONFIG_PATH).capabilities
        assert capabilities.session_minima_status == "provisional_pilot_target"
        assert capabilities.session_minima_are_frozen is False
        provenance = capabilities.session_minima_provenance
        assert "after ingesting and examining" in provenance
        assert "provisional" in provenance
        assert "frozen before a pilot partition is consumed" in provenance

    def test_the_structural_minima_claim_independence_from_this_archive(self) -> None:
        provenance = load_market_config(CONFIG_PATH).capabilities.structural_minima_provenance
        assert "independently of this archive" in provenance
        assert "No observation of this archive informed either number" in provenance

    def test_the_provenance_is_serialized_with_the_numbers(self) -> None:
        summary = load_market_config(CONFIG_PATH).capabilities.summary()
        assert summary["minimum_sessions"] == 60
        assert summary["session_minima_status"] == "provisional_pilot_target"
        assert summary["session_minima_are_frozen"] is False
        assert summary["session_minima_provenance"]
        assert summary["structural_minima_provenance"]

    def test_an_unknown_minima_status_is_rejected(self, document: dict[str, Any]) -> None:
        document["capabilities"]["session_minima_status"] = "desk_grade"
        with pytest.raises(ConfigError, match="session_minima_status 'desk_grade'"):
            parse(document)

    def test_omitting_the_provenance_is_rejected(self, document: dict[str, Any]) -> None:
        del document["capabilities"]["session_minima_provenance"]
        with pytest.raises(ConfigError, match="session_minima_provenance"):
            parse(document)

    def test_more_held_out_than_total_sessions_is_rejected(
        self, document: dict[str, Any]
    ) -> None:
        document["capabilities"]["minimum_held_out_sessions"] = 10_000
        with pytest.raises(ConfigError, match="exceeds minimum_sessions"):
            parse(document)


class TestLoaderStrictness:
    def test_an_unknown_top_level_key_is_rejected(self, document: dict[str, Any]) -> None:
        document["surprise"] = 1
        with pytest.raises(ConfigError, match="unknown keys"):
            parse(document)

    def test_an_unknown_nested_key_is_rejected(self, document: dict[str, Any]) -> None:
        document["session"]["lunch_break"] = "12:00"
        with pytest.raises(ConfigError, match=r"\[session\] has unknown keys"):
            parse(document)

    def test_a_wrong_schema_version_is_rejected(self, document: dict[str, Any]) -> None:
        document["schema_version"] = "market-feasibility/999"
        with pytest.raises(ConfigError, match="unsupported schema_version"):
            parse(document)

    def test_a_repeated_date_is_rejected(self, document: dict[str, Any]) -> None:
        document["dates"].append(copy.deepcopy(document["dates"][0]))
        with pytest.raises(ConfigError, match="repeats date"):
            parse(document)

    def test_an_unknown_date_role_is_rejected(self, document: dict[str, Any]) -> None:
        document["dates"][0]["role"] = "test"
        with pytest.raises(ConfigError, match="role 'test' is unknown"):
            parse(document)

    def test_dropping_the_held_out_date_is_rejected(self, document: dict[str, Any]) -> None:
        document["dates"] = [
            entry for entry in document["dates"] if entry["role"] != "held_out"
        ]
        with pytest.raises(ConfigError, match="no held_out date"):
            parse(document)

    def test_a_repeated_product_key_is_rejected(self, document: dict[str, Any]) -> None:
        document["products"].append(copy.deepcopy(document["products"][0]))
        with pytest.raises(ConfigError, match="repeats product key"):
            parse(document)

    def test_an_option_product_without_definitions_is_rejected(
        self, document: dict[str, Any]
    ) -> None:
        del product(document, "spy-options")["definition_schema"]
        with pytest.raises(ConfigError, match="option product without a definition_schema"):
            parse(document)

    def test_a_quote_schema_outside_the_schemas_list_is_rejected(
        self, document: dict[str, Any]
    ) -> None:
        product(document, "spy-options")["quote_schema"] = "bbo-1m"
        with pytest.raises(ConfigError, match="is not in its schemas list"):
            parse(document)

    def test_a_definition_schema_used_as_a_quote_schema_is_rejected(
        self, document: dict[str, Any]
    ) -> None:
        product(document, "spy-options")["quote_schema"] = "definition"
        with pytest.raises(ConfigError, match="is not a quote schema"):
            parse(document)

    def test_a_pair_referencing_a_quoteless_product_is_rejected(
        self, document: dict[str, Any]
    ) -> None:
        document["synchronization_pairs"][0]["reference_product"] = "sr3-futures"
        with pytest.raises(ConfigError, match="has no quote schema"):
            parse(document)

    def test_a_pair_referencing_an_undeclared_product_is_rejected(
        self, document: dict[str, Any]
    ) -> None:
        document["synchronization_pairs"][0]["reference_product"] = "vix-futures"
        with pytest.raises(ConfigError, match="is not a declared product"):
            parse(document)

    def test_an_absolute_path_is_rejected(self, document: dict[str, Any]) -> None:
        document["archive"]["root"] = "/etc"
        with pytest.raises(ConfigError, match="must be a relative path"):
            parse(document)

    def test_a_traversing_path_is_rejected(self, document: dict[str, Any]) -> None:
        document["output"]["processed_root"] = "data/../../elsewhere"
        with pytest.raises(ConfigError, match="must be a relative path"):
            parse(document)

    def test_a_session_that_closes_before_it_opens_is_rejected(
        self, document: dict[str, Any]
    ) -> None:
        document["session"]["close_local"] = "08:00"
        with pytest.raises(ConfigError, match="closes at or before it opens"):
            parse(document)

    def test_a_minute_count_that_contradicts_the_window_is_rejected(
        self, document: dict[str, Any]
    ) -> None:
        document["session"]["expected_minutes"] = 400
        with pytest.raises(ConfigError, match="expected_minutes is 400"):
            parse(document)

    def test_a_grid_that_does_not_divide_the_session_is_rejected(
        self, document: dict[str, Any]
    ) -> None:
        document["synchronization"]["frequency_seconds"] = 7
        with pytest.raises(ConfigError, match="not an exact multiple"):
            parse(document)

    def test_an_unknown_time_zone_is_rejected(self, document: dict[str, Any]) -> None:
        document["session"]["time_zone"] = "Mars/Olympus_Mons"
        with pytest.raises(ConfigError, match="is not available"):
            parse(document)

    def test_non_increasing_bucket_edges_are_rejected(self, document: dict[str, Any]) -> None:
        document["audit"]["moneyness_bucket_edges"] = [1.0, 0.5]
        with pytest.raises(ConfigError, match="strictly increasing"):
            parse(document)

    def test_more_manifest_entries_than_files_is_rejected(
        self, document: dict[str, Any]
    ) -> None:
        document["archive"]["expected_manifest_entries"] = 10_000
        with pytest.raises(ConfigError, match="exceeds expected_total_files"):
            parse(document)

    def test_a_missing_file_is_reported_clearly(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigError, match="cannot read configuration"):
            load_market_config(tmp_path / "absent.toml")

    def test_invalid_toml_is_reported_clearly(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.toml"
        path.write_text("this is not = = toml", encoding="utf-8")
        with pytest.raises(ConfigError, match="not valid UTF-8 TOML"):
            load_market_config(path)
