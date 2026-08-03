"""End-to-end Task 9B behaviour against a synthetic Task 9A output tree.

The tree is generated from a chosen ``(D, F)`` and a chosen carry, so the study
has a known right answer at every step: the reconstruction must return the
generating parameters, the American residual must return the generating carry,
and the identifiability matrix must classify each input from what the synthetic
archive actually contains rather than from a constant.

No private market data is involved. Nothing here reads the real archive.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest
from differentiable_pricing.market.identifiability import (
    DIRECTLY_OBSERVED,
    EXTERNAL_CONVENTION,
    JOINTLY_IDENTIFIABLE_ONLY,
    MAXIMUM_DISCOUNT_SPREAD_FOR_ROBUST,
    POINTWISE_IDENTIFIED_CURVE_UNCONSTRUCTED,
    ROBUSTLY_INFERRED,
    UNAVAILABLE,
    ObservedFacts,
    build_identifiability_matrix,
)
from differentiable_pricing.market.reconstruct import (
    ProvenanceError,
    ReconstructionError,
    build_tables,
    run_and_write,
    run_study,
    verify_provenance,
    write_atomically,
)
from differentiable_pricing.market.state_config import parse_state_config
from state_fixtures import (
    CARRY,
    EUROPEAN_ONLY_EXPIRY,
    EXPIRIES,
    RATE,
    SESSIONS,
    SPOT_MID,
    ZERO_DTE_EXPIRY,
    build_state_config_document,
    build_task_9a_outputs,
    discount_factor,
    european_forward,
)


@pytest.fixture
def synthetic_study(tmp_path: Path):
    """A synthetic 9A tree plus a Task 9B configuration pinned to it."""
    outputs = build_task_9a_outputs(tmp_path / "task9a")
    document = build_state_config_document(outputs)
    config = parse_state_config(
        document, source_name="synthetic.toml", source_sha256="1" * 64
    )
    return config, outputs


def matrix_row(document: dict[str, Any], key: str) -> dict[str, Any]:
    for row in document["identifiability"]["matrix"]:
        if row["input"] == key:
            return row
    raise AssertionError(f"no identifiability row for {key!r}")


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_provenance_passes_on_an_untouched_tree(synthetic_study) -> None:
    config, outputs = synthetic_study
    checks, report = verify_provenance(config, outputs["base"])
    assert all(check.satisfied for check in checks)
    names = {check.name for check in checks}
    assert {
        "task_9a_config_sha256",
        "task_9a_report_sha256",
        "raw_archive_manifest_sha256",
        "task_9a_report_agrees_on_its_config_digest",
        "task_9a_report_agrees_on_the_source_manifest_digest",
        "task_9a_processed_root",
        "processed_root_exists",
    } <= names
    assert report["audit_version"] == "2.0.0"


def test_a_tampered_report_aborts_before_any_arithmetic(synthetic_study, tmp_path) -> None:
    config, outputs = synthetic_study
    report = json.loads(outputs["report_path"].read_text(encoding="utf-8"))
    report["audit_version"] = "9.9.9"
    outputs["report_path"].write_text(json.dumps(report, indent=2, sort_keys=True),
                                      encoding="utf-8")
    destination = tmp_path / "out"
    with pytest.raises(ProvenanceError, match="does not match what this study pinned"):
        run_and_write(config, outputs["base"], destination)
    assert not destination.exists(), "a failed provenance check must write nothing"


def test_a_tampered_manifest_aborts(synthetic_study) -> None:
    config, outputs = synthetic_study
    manifest = outputs["base"] / outputs["archive_root"] / outputs["manifest_relative"]
    manifest.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="raw_archive_manifest_sha256"):
        verify_provenance(config, outputs["base"])


def test_a_report_disagreeing_with_itself_aborts(synthetic_study) -> None:
    """The report's own record of its config digest is cross-checked."""
    _, outputs = synthetic_study
    report = json.loads(outputs["report_path"].read_text(encoding="utf-8"))
    report["config"]["source_sha256"] = "f" * 64
    text = json.dumps(report, indent=2, sort_keys=True)
    outputs["report_path"].write_text(text, encoding="utf-8")
    # Repin the report digest so only the internal disagreement remains.
    document = build_state_config_document(outputs)
    import hashlib

    document["provenance"]["feasibility_report_sha256"] = hashlib.sha256(
        outputs["report_path"].read_bytes()
    ).hexdigest()
    repinned = parse_state_config(document, source_name="x.toml", source_sha256="1" * 64)
    with pytest.raises(ProvenanceError, match="agrees_on_its_config_digest"):
        verify_provenance(repinned, outputs["base"])


def test_a_missing_report_is_a_clear_failure(synthetic_study) -> None:
    config, outputs = synthetic_study
    outputs["report_path"].unlink()
    with pytest.raises(ProvenanceError, match="is missing"):
        verify_provenance(config, outputs["base"])


def test_a_missing_processed_partition_is_reported(synthetic_study) -> None:
    config, outputs = synthetic_study
    partition = (
        outputs["base"]
        / outputs["processed_relative"]
        / "product=xsp-options"
        / "schema=cbbo-1m"
        / f"date={SESSIONS[0].isoformat()}"
    )
    for path in partition.iterdir():
        path.unlink()
    partition.rmdir()
    with pytest.raises(ReconstructionError, match="missing quote partition"):
        run_study(config, outputs["base"])


# ---------------------------------------------------------------------------
# Recovery of the generating parameters
# ---------------------------------------------------------------------------


def test_the_study_recovers_the_generating_discount_and_forward(synthetic_study) -> None:
    config, outputs = synthetic_study
    document, outcomes = run_study(config, outputs["base"])
    assert outcomes

    for outcome in outcomes:
        for result in outcome.european:
            for fit in result.fits.values():
                assert fit.valid, fit.diagnostics
                assert fit.discount_factor == pytest.approx(
                    discount_factor(result.tau_years), rel=1e-9
                )
                assert fit.forward == pytest.approx(
                    european_forward(result.tau_years), rel=1e-9
                )
                assert fit.containment_fraction == 1.0
                assert fit.max_outside_spread_error == pytest.approx(0.0, abs=1e-9)
                assert fit.slope < 0.0
                if result.is_zero_dte:
                    # D and F are still exact; only the rate is withheld.
                    assert fit.rate is None
                    assert fit.rate_withheld_reason == "zero_dte"
                else:
                    assert fit.rate == pytest.approx(RATE, rel=1e-6)

    facts = document["identifiability"]["observed_facts"]
    assert facts["european_fits_valid"] == facts["european_fits_attempted"]
    assert facts["european_containment_fraction_median"] == 1.0


def test_only_exactly_matched_expiries_carry_an_american_diagnostic(
    synthetic_study,
) -> None:
    config, outputs = synthetic_study
    _, outcomes = run_study(config, outputs["base"])
    for outcome in outcomes:
        matched = {result.expiry for result in outcome.american if result.matched}
        expected = set(EXPIRIES)
        if outcome.date == ZERO_DTE_EXPIRY:
            expected.add(ZERO_DTE_EXPIRY)
        assert matched == expected
        assert EUROPEAN_ONLY_EXPIRY not in matched


def test_the_carry_residual_recovers_the_generating_carry(synthetic_study) -> None:
    """Q = S - D * median F_tilde must equal ``S (1 - exp(-q tau))``.

    The median is taken over strikes, and the synthetic early-exercise premium
    is applied only to strikes above the forward, so the median strike is
    uncontaminated and the level is exact.
    """
    config, outputs = synthetic_study
    _, outcomes = run_study(config, outputs["base"])
    checked = 0
    for outcome in outcomes:
        for result in outcome.american:
            if not result.matched or result.is_zero_dte:
                continue
            residual = result.residuals[(0.2, "unweighted_least_squares", "mid")]
            assert residual.valid
            expected = SPOT_MID * (1.0 - math.exp(-CARRY * result.tau_years))
            assert residual.carry_residual == pytest.approx(expected, rel=1e-6)
            assert residual.carry_residual > 0.0
            checked += 1
    assert checked > 0


def test_the_spot_variants_move_the_residual_by_the_quoted_width(synthetic_study) -> None:
    config, outputs = synthetic_study
    _, outcomes = run_study(config, outputs["base"])
    for outcome in outcomes:
        for result in outcome.american:
            if not result.matched or result.is_zero_dte:
                continue
            bid = result.residuals[(0.2, "unweighted_least_squares", "bid")]
            ask = result.residuals[(0.2, "unweighted_least_squares", "ask")]
            assert ask.carry_residual - bid.carry_residual == pytest.approx(
                outcome.underlying_ask - outcome.underlying_bid, rel=1e-9
            )


def test_early_exercise_contamination_is_detected_against_the_european_control(
    synthetic_study,
) -> None:
    """The American chain tilts in the strike; the European control does not."""
    config, outputs = synthetic_study
    document, _ = run_study(config, outputs["base"])
    dependence = document["american_diagnostic"]["sensitivity"]["strike_dependence"]
    assert dependence["strike_dependence_indicated"] is True
    assert dependence["spy_abs_slope"]["median"] > 0.0
    assert dependence["xsp_european_control_abs_slope"]["median"] == pytest.approx(
        0.0, abs=1e-9
    )


# ---------------------------------------------------------------------------
# Identifiability
# ---------------------------------------------------------------------------


def test_identifiability_follows_the_synthetic_archive_not_a_constant(
    synthetic_study,
) -> None:
    config, outputs = synthetic_study
    document, _ = run_study(config, outputs["base"])

    assert matrix_row(document, "valuation_time")["classification"] == DIRECTLY_OBSERVED
    assert matrix_row(document, "spy_spot")["classification"] == DIRECTLY_OBSERVED
    assert matrix_row(document, "discount_curve")["classification"] == ROBUSTLY_INFERRED
    assert matrix_row(document, "forward_curve")["classification"] == ROBUSTLY_INFERRED
    assert matrix_row(document, "discrete_dividend_dates")["classification"] == UNAVAILABLE
    assert matrix_row(document, "discrete_dividend_amounts")["classification"] == UNAVAILABLE
    assert (
        matrix_row(document, "effective_borrow_or_carry")["classification"]
        == JOINTLY_IDENTIFIABLE_ONLY
    )
    assert matrix_row(document, "exercise_style")["classification"] == EXTERNAL_CONVENTION
    assert matrix_row(document, "settlement_convention")["classification"] == EXTERNAL_CONVENTION
    assert matrix_row(document, "contract_multiplier")["classification"] == EXTERNAL_CONVENTION
    assert matrix_row(document, "volatility_input")["classification"] == UNAVAILABLE


def test_the_declared_source_states_are_distinguished(synthetic_study) -> None:
    """An empty directory and a missing one are different facts with different fixes."""
    config, outputs = synthetic_study
    document, _ = run_study(config, outputs["base"])
    probes = document["provenance"]["declared_source_reprobe"]
    assert probes["dividend"]["state"] == "present_but_empty"
    assert probes["corporate_action"]["state"] == "present_but_empty"
    assert probes["borrow"]["state"] == "absent"


def test_a_populated_exercise_style_flips_the_classification(synthetic_study) -> None:
    """The classification is driven by the observation, so changing it changes the verdict."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    config, outputs = synthetic_study
    for date in SESSIONS:
        directory = (
            outputs["base"]
            / outputs["processed_relative"]
            / "product=spy-options"
            / "schema=definition"
            / f"date={date.isoformat()}"
        )
        target = next(iter(directory.glob("*.parquet")))
        table = pq.read_table(directory)
        styles = pa.array(["A"] * table.num_rows, pa.string())
        updated = table.set_column(
            table.schema.get_field_index("exercise_style"), "exercise_style", styles
        )
        pq.write_table(updated, target)

    document, _ = run_study(config, outputs["base"])
    assert matrix_row(document, "exercise_style")["classification"] == DIRECTLY_OBSERVED


def test_the_report_never_names_the_residual_a_dividend(synthetic_study) -> None:
    config, outputs = synthetic_study
    document, _ = run_study(config, outputs["base"])
    text = json.dumps(document).lower()
    for phrase in ("observed dividend", "exact dividend present value", "exact spy forward"):
        # The phrases appear only in the list of names that are forbidden.
        forbidden = document["american_diagnostic"]["forbidden_names"]
        assert phrase in [name.lower() for name in forbidden]
    assert "american_parity_carry_residual" in text
    assert document["american_diagnostic"]["name"] == "american_parity_carry_residual"


def test_the_proposed_contract_marks_the_unfillable_fields(synthetic_study) -> None:
    config, outputs = synthetic_study
    document, _ = run_study(config, outputs["base"])
    contract = document["task_9c_input_contract"]
    assert contract["status"] == "proposed_only_not_implemented"
    assert set(contract["blocking_for_real_spy_evaluation"]) == {
        "dividends",
        "continuous_carry",
    }
    names = {field["name"] for field in contract["fields"]}
    assert {
        "spot", "strike", "option_type", "valuation_time", "expiry_time",
        "discount_curve", "dividends", "continuous_carry", "volatility",
        "exercise_style", "settlement", "contract_multiplier",
    } <= names


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def test_outputs_are_written_deterministically(synthetic_study, tmp_path) -> None:
    config, outputs = synthetic_study
    first = tmp_path / "first"
    second = tmp_path / "second"
    written = run_and_write(config, outputs["base"], first)
    run_and_write(config, outputs["base"], second)
    assert written
    for path in written:
        twin = second / path.relative_to(first)
        assert path.read_bytes() == twin.read_bytes(), f"{path.name} is not reproducible"


def test_the_report_is_standard_json_with_no_nan_literals(synthetic_study, tmp_path) -> None:
    config, outputs = synthetic_study
    root = tmp_path / "out"
    run_and_write(config, outputs["base"], root)
    raw = (root / config.output.report_filename).read_text(encoding="utf-8")
    assert "NaN" not in raw and "Infinity" not in raw
    json.loads(raw, parse_constant=_reject_constant)


def _reject_constant(value: str):
    raise AssertionError(f"non-standard JSON constant {value!r} in the report")


def test_tables_and_figures_are_published(synthetic_study, tmp_path) -> None:
    config, outputs = synthetic_study
    root = tmp_path / "out"
    run_and_write(config, outputs["base"], root)
    assert (root / config.output.table_filename).is_file()
    assert (root / config.output.american_table_filename).is_file()
    figures = sorted((root / config.output.figure_subdirectory).glob("*.svg"))
    assert figures
    for figure in figures:
        text = figure.read_text(encoding="utf-8")
        assert text.startswith("<svg") and text.rstrip().endswith("</svg>")


def test_the_european_table_carries_every_reported_column(synthetic_study) -> None:
    config, outputs = synthetic_study
    _, outcomes = run_study(config, outputs["base"])
    (header, rows), (american_header, american_rows) = build_tables(outcomes)
    for column in (
        "discount_factor_D", "forward_F", "rate_r", "condition_number",
        "containment_fraction", "rms_outside_spread_error", "max_outside_spread_error",
        "diagnostics",
    ):
        assert column in header
    assert rows
    for column in ("carry_residual", "f_tilde_strike_slope", "spot_variant"):
        assert column in american_header
    assert american_rows


def test_a_failed_write_leaves_no_temporary_file(tmp_path) -> None:
    """The atomic writer removes its temporary file on every failure path."""
    destination = tmp_path / "nested" / "report.json"
    destination.parent.mkdir(parents=True)
    write_atomically(destination, b"first\n")
    assert destination.read_bytes() == b"first\n"

    # A directory at the destination makes os.replace fail after the temporary
    # file has been written.
    blocked = tmp_path / "nested" / "blocked.json"
    blocked.mkdir()
    with pytest.raises(ReconstructionError, match="cannot publish"):
        write_atomically(blocked, b"second\n")
    strays = [path for path in blocked.parent.iterdir() if path.name.startswith(".")]
    assert strays == [], f"temporary files left behind: {strays}"
    assert destination.read_bytes() == b"first\n"


def test_a_failed_run_leaves_previous_outputs_untouched(synthetic_study, tmp_path) -> None:
    config, outputs = synthetic_study
    root = tmp_path / "out"
    run_and_write(config, outputs["base"], root)
    original = (root / config.output.report_filename).read_bytes()

    manifest = outputs["base"] / outputs["archive_root"] / outputs["manifest_relative"]
    manifest.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ProvenanceError):
        run_and_write(config, outputs["base"], root)
    assert (root / config.output.report_filename).read_bytes() == original


# ---------------------------------------------------------------------------
# Verdicts and scope
# ---------------------------------------------------------------------------


def test_the_report_states_its_scope_and_its_non_claims(synthetic_study) -> None:
    config, outputs = synthetic_study
    document, _ = run_study(config, outputs["base"])
    out_of_scope = document["scope"]["explicitly_out_of_scope"]
    assert any("PDE" in item for item in out_of_scope)
    assert any("implied volatility" in item for item in out_of_scope)
    assert any("volatility surface" in item for item in out_of_scope)
    assert len(document["limitations_and_non_claims"]) >= 8


def test_every_verdict_question_is_answered(synthetic_study) -> None:
    config, outputs = synthetic_study
    document, _ = run_study(config, outputs["base"])
    verdicts = document["verdicts"]
    assert len(verdicts) == 7
    assert all(key[0].isdigit() for key in verdicts)
    assert verdicts["1_is_xsp_discount_and_forward_reconstruction_numerically_stable"][
        "answer"
    ] == "yes"


# ---------------------------------------------------------------------------
# Regressions from review
# ---------------------------------------------------------------------------


def test_zero_dte_expiries_are_excluded_from_the_stability_aggregates(
    synthetic_study,
) -> None:
    """A tau of hours pins D near one; pooling that cell understates the spread.

    The zero-DTE cell must be published per expiry and excluded from the
    aggregate the ``robustly_inferred`` verdict is derived from, exactly as it
    is excluded from rate interpretation and from the American aggregates.
    """
    config, outputs = synthetic_study
    document, outcomes = run_study(config, outputs["base"])

    zero_dte = [
        result
        for outcome in outcomes
        for result in outcome.european
        if result.is_zero_dte
    ]
    assert zero_dte, "the fixture must contain a same-session expiry"
    assert document["european_sensitivity"]["zero_dte_expiries_excluded"] == len(zero_dte)
    assert document["american_diagnostic"]["sensitivity"]["zero_dte_expiries_excluded"] > 0

    # Published per expiry despite the exclusion.
    published = [
        entry
        for date_entry in document["per_date"].values()
        for snapshot in date_entry["snapshots"]
        for entry in snapshot["european_expiries"]
        if entry["is_zero_dte"]
    ]
    assert published

    # The aggregate must equal the one computed with zero-DTE cells dropped.
    ranges = []
    for outcome in outcomes:
        for result in outcome.european:
            if result.is_zero_dte:
                continue
            values = [fit.discount_factor for fit in result.fits.values() if fit.valid]
            if len(values) > 1:
                ranges.append(max(values) - min(values))
    expected = sorted(ranges)[len(ranges) // 2] if len(ranges) % 2 else 0.5 * (
        sorted(ranges)[len(ranges) // 2 - 1] + sorted(ranges)[len(ranges) // 2]
    )
    assert document["identifiability"]["observed_facts"][
        "european_discount_factor_spread"
    ] == pytest.approx(expected)


def test_the_validity_fraction_is_reported_beside_the_zero_tolerance_verdict(
    synthetic_study,
) -> None:
    config, outputs = synthetic_study
    document, _ = run_study(config, outputs["base"])
    facts = document["identifiability"]["observed_facts"]
    assert facts["european_fit_validity_fraction"] == pytest.approx(1.0)
    assert "Zero tolerance" in facts["european_fit_validity_policy"]
    assert "not mean their contents were parsed" in facts["declared_source_probe_semantics"]


def test_a_malformed_underlying_quote_is_refused_rather_than_priced(
    synthetic_study,
) -> None:
    """The spot enters the residual linearly, so a crossed one must not pass.

    The option legs are re-validated downstream; the spot has no second reader,
    so it is checked unconditionally rather than trusting the tradability
    filter to have caught it.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    config, outputs = synthetic_study
    directory = (
        outputs["base"]
        / outputs["processed_relative"]
        / "product=spy-underlying"
        / "schema=bbo-1m"
        / f"date={SESSIONS[0].isoformat()}"
    )
    target = next(iter(directory.glob("*.parquet")))
    table = pq.read_table(directory)
    crossed = pa.array([900.0] * table.num_rows, pa.float64())
    updated = table.set_column(
        table.schema.get_field_index("bid_price"), "bid_price", crossed
    )
    pq.write_table(updated, target)

    with pytest.raises(ReconstructionError, match="not a usable spot"):
        run_study(config, outputs["base"])


def test_an_underlying_that_does_not_resolve_to_one_quote_is_refused(
    synthetic_study,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    config, outputs = synthetic_study
    directory = (
        outputs["base"]
        / outputs["processed_relative"]
        / "product=spy-underlying"
        / "schema=bbo-1m"
        / f"date={SESSIONS[0].isoformat()}"
    )
    target = next(iter(directory.glob("*.parquet")))
    table = pq.read_table(directory)
    # Two distinct instruments quoting in the same minute: there is no rule here
    # for choosing between them, so the run must stop rather than pick one.
    index = table.schema.get_field_index("instrument_id")
    ids = table.column("instrument_id").to_pylist()
    shifted = table.set_column(
        index,
        table.schema.field(index),
        pa.array([value + 1 for value in ids], table.schema.field(index).type),
    )
    pq.write_table(pa.concat_tables([table, shifted]), target)

    with pytest.raises(ReconstructionError, match="exactly one is required"):
        run_study(config, outputs["base"])


def test_a_missing_processed_root_fails_the_provenance_check(synthetic_study) -> None:
    import shutil

    config, outputs = synthetic_study
    shutil.rmtree(outputs["base"] / outputs["processed_relative"])
    with pytest.raises(ProvenanceError, match="processed_root_exists"):
        verify_provenance(config, outputs["base"])


def test_the_report_is_published_after_the_tables_and_figures(synthetic_study, tmp_path):
    """The index artefact goes last, so an interrupted publish is visible.

    A stale report beside fresh tables reads as an interrupted run. A fresh
    report beside stale tables would read as a complete one and would not be.
    """
    config, outputs = synthetic_study
    root = tmp_path / "out"
    written = run_and_write(config, outputs["base"], root)
    assert written[-1].name == config.output.report_filename
    assert written[0].name == config.output.table_filename


# ---------------------------------------------------------------------------
# The scheduled ex-date, and what it does and does not establish
# ---------------------------------------------------------------------------


def ex_date_block(document: dict[str, Any]) -> dict[str, Any]:
    return document["american_diagnostic"]["ex_date_comparison"]


def test_the_report_carries_the_official_schedule_not_a_hypothesis(
    synthetic_study,
) -> None:
    config, outputs = synthetic_study
    document, _ = run_study(config, outputs["base"])
    schedule = ex_date_block(document)["schedule"]
    assert schedule["status"] == "officially_scheduled"
    assert schedule["ex_dates"] == ["2026-06-18"]
    assert schedule["record_dates"] == ["2026-06-18"]
    assert schedule["payable_dates"] == ["2026-07-31"]
    assert schedule["source_publisher"] == "State Street Global Advisors"
    assert "ssga.com" in schedule["source_url"]

    text = json.dumps(document)
    assert "unverified_hypothesis" not in text
    assert "hypothesized_ex_date" not in text


def test_the_session_transition_crosses_the_scheduled_ex_date(synthetic_study) -> None:
    config, outputs = synthetic_study
    document, _ = run_study(config, outputs["base"])
    block = ex_date_block(document)
    transitions = block["session_transitions"]
    assert transitions, "the fixture must span the scheduled ex-date"
    crossing = [row for row in transitions if row["crosses_scheduled_ex_date"]]
    assert crossing == [
        row
        for row in transitions
        if row["from_session"] == "2026-06-17" and row["to_session"] == "2026-06-18"
    ]
    assert block["largest_transition"]["crosses_scheduled_ex_date"] is True
    assert block["scheduled_ex_date_alignment"][
        "largest_step_crosses_the_scheduled_ex_date"
    ] is True


def test_the_alignment_reading_claims_alignment_and_nothing_more(
    synthetic_study,
) -> None:
    """No mismatch language, and no inference of the amount or its components."""
    config, outputs = synthetic_study
    document, _ = run_study(config, outputs["base"])
    block = ex_date_block(document)
    reading = block["scheduled_ex_date_alignment"]["reading"]

    assert "temporally aligned with" in reading
    assert "qualitatively consistent with" in reading
    assert "scheduled ex-dividend event" in reading
    assert "mismatch" not in reading.lower()

    lowered = (reading + block["interpretation_warning"] + block["schedule"]["scope"]).lower()
    assert "does not infer or validate the cash dividend amount" in lowered
    for component in ("borrow", "early-exercise", "basis", "fitting"):
        assert component in lowered

    whole = json.dumps(document).lower()
    assert "mismatch between an unverified background assumption" not in whole


# ---------------------------------------------------------------------------
# Pointwise identification versus the exploratory stability bar
# ---------------------------------------------------------------------------


def facts_with_spread(spread: float) -> ObservedFacts:
    """Observed facts identical but for the discount-factor spread."""
    return ObservedFacts(
        european_fits_attempted=100,
        european_fits_valid=100,
        european_containment_fraction_median=1.0,
        european_discount_factor_spread=spread,
        european_discount_factor_intraday_spread=2.0 * spread,
        european_rate_interpretable_expiries=90,
        matched_spy_xsp_expiries=30,
        american_residual_defined=True,
        american_residual_spread=0.4,
        american_residual_intraday_spread=1.2,
        american_f_tilde_shows_strike_dependence=True,
        dividend_source_state="present_but_empty",
        borrow_source_state="absent",
        corporate_action_source_state="present_but_empty",
        exercise_style_populated_fraction=0.0,
        contract_multiplier_populated_fraction=0.0,
        settlement_time_populated_fraction=0.0,
        underlying_feed_class="partial_venue_consolidated",
        underlying_relative_half_spread=2.0e-5,
        sessions_observed=3,
    )


def classification_of(facts: ObservedFacts, key: str) -> dict[str, Any]:
    for row in build_identifiability_matrix(facts):
        if row.key == key:
            return row.summary()
    raise AssertionError(f"no row for {key!r}")


@pytest.mark.parametrize("key", ["discount_curve", "forward_curve"])
def test_a_narrowly_missed_stability_bar_keeps_pointwise_identification(key: str) -> None:
    """The exploratory threshold bounds precision. It cannot revoke identifiability."""
    missed = facts_with_spread(MAXIMUM_DISCOUNT_SPREAD_FOR_ROBUST * 1.004)
    assert not missed.european_reconstruction_is_robust
    row = classification_of(missed, key)
    assert row["classification"] == POINTWISE_IDENTIFIED_CURVE_UNCONSTRUCTED
    assert row["classification"] != JOINTLY_IDENTIFIABLE_ONLY

    met = facts_with_spread(MAXIMUM_DISCOUNT_SPREAD_FOR_ROBUST * 0.5)
    assert met.european_reconstruction_is_robust
    assert classification_of(met, key)["classification"] == ROBUSTLY_INFERRED


def test_the_missed_bar_evidence_says_identified_but_uninterpolated() -> None:
    missed = facts_with_spread(MAXIMUM_DISCOUNT_SPREAD_FOR_ROBUST * 1.004)
    evidence = classification_of(missed, "discount_curve")["evidence"]
    assert "does not make them unidentified" in evidence
    assert "no second unknown to be confounded with" in evidence
    assert "no interpolation through them" in evidence


def test_genuinely_confounded_quantities_keep_the_joint_class() -> None:
    """Borrow and dividends really are confounded; the new class must not leak there."""
    for facts in (
        facts_with_spread(MAXIMUM_DISCOUNT_SPREAD_FOR_ROBUST * 1.004),
        facts_with_spread(MAXIMUM_DISCOUNT_SPREAD_FOR_ROBUST * 0.5),
    ):
        row = classification_of(facts, "effective_borrow_or_carry")
        assert row["classification"] == JOINTLY_IDENTIFIABLE_ONLY
        assert row["classification"] != POINTWISE_IDENTIFIED_CURVE_UNCONSTRUCTED


def test_no_continuous_curve_is_ever_claimed(synthetic_study) -> None:
    """Whether or not the bar is met, this study builds knots and no curve."""
    config, outputs = synthetic_study
    document, _ = run_study(config, outputs["base"])
    row = matrix_row(document, "discount_curve")
    assert "no continuous discount or forward curve has been constructed" in row["evidence"]
    assert "interpolate" in row["consequence_for_a_pricer"]

    curve_field = next(
        field
        for field in document["task_9c_input_contract"]["fields"]
        if field["name"] == "discount_curve"
    )
    assert "interpolation rule is a modelling choice this study did not make" in (
        curve_field["source_in_task_9b"]
    )


# ---------------------------------------------------------------------------
# Cross-snapshot variation is not attributed to one cause
# ---------------------------------------------------------------------------


def test_cross_snapshot_variation_is_not_called_market_movement(synthetic_study) -> None:
    config, outputs = synthetic_study
    document, _ = run_study(config, outputs["base"])
    axis = document["european_sensitivity"]["axis_definitions"][
        "across_snapshots_and_windows_and_methods"
    ]
    facts = document["identifiability"]["observed_facts"]
    combined = axis + facts["european_discount_factor_intraday_spread_definition"]
    for cause in (
        "market movement",
        "quote microstructure",
        "strike composition",
        "fitting variation",
    ):
        assert cause in combined
    assert "not attributable to one cause" in axis

    whole = json.dumps(document)
    assert "which is the market moving between" not in whole
    assert "because the market moved between" not in whole
