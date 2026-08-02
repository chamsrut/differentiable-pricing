"""Declared sources must be *observed*, and the three states must stay distinct.

The defect this module guards against is an audit that encodes "the dividend
data is missing" as a constant. Every assertion here checks that the answer
changes when the archive changes.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest
from differentiable_pricing.market.provenance import (
    SOURCE_ABSENT,
    SOURCE_PRESENT,
    SOURCE_PRESENT_BUT_EMPTY,
    ConditionError,
    probe_declared_source,
    read_vendor_conditions,
)
from market_fixtures import write_condition

PROBE = {
    "key": "discrete_dividend_schedule",
    "requirement": "required",
    "capability": "spy_american_calibration_readiness",
    "why": "because the study needs it",
    "candidate_source": "an issuer distribution history",
}


class TestDeclaredSourceProbes:
    def test_a_missing_directory_is_absent(self, tmp_path: Path) -> None:
        observed = probe_declared_source(
            tmp_path, path="raw/issuer", kind="directory", **PROBE
        )
        assert observed.state == SOURCE_ABSENT
        assert observed.satisfied is False
        assert observed.file_count == 0

    def test_an_empty_directory_is_distinguishable_from_a_missing_one(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "raw" / "issuer").mkdir(parents=True)
        observed = probe_declared_source(
            tmp_path, path="raw/issuer", kind="directory", **PROBE
        )
        assert observed.state == SOURCE_PRESENT_BUT_EMPTY
        assert observed.state != SOURCE_ABSENT
        assert observed.satisfied is False
        assert "invisible to manifest verification" in observed.detail

    def test_a_populated_directory_is_present_and_satisfied(self, tmp_path: Path) -> None:
        target = tmp_path / "raw" / "issuer"
        target.mkdir(parents=True)
        (target / "distributions.csv").write_text("ex_date,amount\n", encoding="utf-8")
        observed = probe_declared_source(
            tmp_path, path="raw/issuer", kind="directory", **PROBE
        )
        assert observed.state == SOURCE_PRESENT
        assert observed.satisfied is True
        assert observed.file_count == 1
        assert observed.total_bytes > 0

    def test_a_nested_file_counts_as_content(self, tmp_path: Path) -> None:
        nested = tmp_path / "raw" / "issuer" / "2026"
        nested.mkdir(parents=True)
        (nested / "q2.csv").write_text("x", encoding="utf-8")
        observed = probe_declared_source(
            tmp_path, path="raw/issuer", kind="directory", **PROBE
        )
        assert observed.state == SOURCE_PRESENT

    def test_a_zero_byte_file_is_present_but_empty(self, tmp_path: Path) -> None:
        (tmp_path / "raw").mkdir(parents=True)
        (tmp_path / "raw" / "dividends.csv").touch()
        observed = probe_declared_source(
            tmp_path, path="raw/dividends.csv", kind="file", **PROBE
        )
        assert observed.state == SOURCE_PRESENT_BUT_EMPTY
        assert observed.satisfied is False

    def test_a_file_with_content_is_present(self, tmp_path: Path) -> None:
        (tmp_path / "raw").mkdir(parents=True)
        (tmp_path / "raw" / "dividends.csv").write_text("ex,amount\n", encoding="utf-8")
        observed = probe_declared_source(
            tmp_path, path="raw/dividends.csv", kind="file", **PROBE
        )
        assert observed.state == SOURCE_PRESENT
        assert observed.satisfied is True

    def test_a_missing_file_is_absent(self, tmp_path: Path) -> None:
        observed = probe_declared_source(
            tmp_path, path="raw/dividends.csv", kind="file", **PROBE
        )
        assert observed.state == SOURCE_ABSENT

    def test_the_summary_carries_the_state_and_the_reason(self, tmp_path: Path) -> None:
        observed = probe_declared_source(
            tmp_path, path="raw/issuer", kind="directory", **PROBE
        )
        summary = observed.summary()
        assert summary["state"] == SOURCE_ABSENT
        assert summary["satisfied"] is False
        assert summary["why_the_study_needs_it"] == "because the study needs it"
        assert summary["candidate_source"] == "an issuer distribution history"


class TestVendorConditions:
    def test_available_dates_are_parsed(self, tmp_path: Path) -> None:
        directory = tmp_path / "raw" / "request"
        write_condition(directory, ["2026-06-17", "2026-06-18"])
        entries = read_vendor_conditions(directory / "condition.json", relative_to=tmp_path)
        assert [entry.date for entry in entries] == [
            dt.date(2026, 6, 17),
            dt.date(2026, 6, 18),
        ]
        assert all(entry.is_available for entry in entries)
        assert entries[0].request_path == "raw/request"

    def test_a_degraded_date_is_not_available(self, tmp_path: Path) -> None:
        directory = tmp_path / "raw" / "request"
        write_condition(directory, ["2026-06-17"], condition="degraded")
        entries = read_vendor_conditions(directory / "condition.json", relative_to=tmp_path)
        assert entries[0].is_available is False
        assert entries[0].condition == "degraded"

    def test_an_unparseable_date_is_kept_verbatim(self, tmp_path: Path) -> None:
        directory = tmp_path / "raw" / "request"
        directory.mkdir(parents=True)
        (directory / "condition.json").write_text(
            json.dumps([{"date": "not-a-date", "condition": "available"}]), encoding="utf-8"
        )
        entries = read_vendor_conditions(directory / "condition.json", relative_to=tmp_path)
        assert entries[0].date is None
        assert entries[0].raw_date == "not-a-date"

    def test_malformed_json_is_an_error_not_an_absence(self, tmp_path: Path) -> None:
        directory = tmp_path / "raw" / "request"
        directory.mkdir(parents=True)
        (directory / "condition.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(ConditionError, match="cannot read"):
            read_vendor_conditions(directory / "condition.json", relative_to=tmp_path)

    def test_a_non_list_document_is_an_error(self, tmp_path: Path) -> None:
        directory = tmp_path / "raw" / "request"
        directory.mkdir(parents=True)
        (directory / "condition.json").write_text('{"date": "2026-06-17"}', encoding="utf-8")
        with pytest.raises(ConditionError, match="not a list"):
            read_vendor_conditions(directory / "condition.json", relative_to=tmp_path)
