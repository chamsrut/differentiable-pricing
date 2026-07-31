"""Cross-language tests for the price-only and parallel CRR boundary."""

from __future__ import annotations

import threading

import pytest
from differentiable_pricing import (
    crr_diagnostics,
    crr_lattice_parameters,
    crr_price,
    crr_price_batch,
)


def _columns() -> dict[str, list[object]]:
    return {
        "option_types": ["put", "call", "put", "call"],
        "exercise_styles": ["american", "american", "american", "european"],
        "spots": [100.0, 90.0, 130.0, 110.0],
        "strikes": [100.0, 100.0, 100.0, 100.0],
        "maturities": [1.0, 0.25, 2.0, 0.5],
        "rates": [0.05, 0.01, -0.01, 0.03],
        "dividend_yields": [0.0, 0.08, 0.0, 0.02],
        "volatilities": [0.2, 0.35, 0.5, 0.15],
        "steps": [257, 384, 193, 128],
    }


def _batch(columns: dict[str, list[object]], threads: int) -> dict[str, list[object]]:
    return crr_price_batch(
        columns["option_types"],
        columns["exercise_styles"],
        columns["spots"],
        columns["strikes"],
        columns["maturities"],
        columns["rates"],
        columns["dividend_yields"],
        columns["volatilities"],
        columns["steps"],
        threads,
    )


def test_known_american_put_price_only_value() -> None:
    result = crr_price("put", "american", 100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 2048)
    assert result["price"] == pytest.approx(6.08999882273936, abs=1.0e-13)
    assert result["steps"] == 2048
    assert 0.0 < result["risk_neutral_probability"] < 1.0


def test_scalar_crr_price_releases_the_python_gil() -> None:
    started = threading.Event()
    stop = threading.Event()
    counter = [0]

    def increment() -> None:
        started.set()
        while not stop.is_set():
            counter[0] += 1

    worker = threading.Thread(target=increment)
    worker.start()
    try:
        assert started.wait(timeout=1.0)
        before = counter[0]
        crr_price("put", "american", 100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 8192)
        after = counter[0]
    finally:
        stop.set()
        worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert after > before


def test_american_put_diagnostics_share_the_price_recursion() -> None:
    price = crr_price("put", "american", 100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 512)
    diagnostics = crr_diagnostics(
        "put", "american", 100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 512
    )
    assert diagnostics["price"] == price["price"]
    assert diagnostics["early_exercise_nodes"] > 0
    assert diagnostics["earliest_exercise_step"] is not None
    assert len(diagnostics["exercise_boundary_by_step"]) == 512
    assert any(value is not None for value in diagnostics["exercise_boundary_by_step"])


def test_lattice_parameters_match_price_probability() -> None:
    lattice = crr_lattice_parameters(100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 257)
    price = crr_price("put", "american", 100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 257)
    assert lattice["risk_neutral_probability"] == price["risk_neutral_probability"]
    assert lattice["up_factor"] * lattice["down_factor"] == pytest.approx(1.0)
    assert lattice["time_step"] == pytest.approx(1.0 / 257.0)


def test_serial_and_parallel_batches_are_bit_identical_and_ordered() -> None:
    columns = _columns()
    serial = _batch(columns, 1)
    parallel = _batch(columns, 4)
    assert serial == parallel
    assert serial["steps"] == columns["steps"]

    scalar_prices = [
        crr_price(
            columns["option_types"][index],
            columns["exercise_styles"][index],
            columns["spots"][index],
            columns["strikes"][index],
            columns["maturities"][index],
            columns["rates"][index],
            columns["dividend_yields"][index],
            columns["volatilities"][index],
            columns["steps"][index],
        )["price"]
        for index in range(len(columns["steps"]))
    ]
    assert serial["price"] == scalar_prices


def test_empty_batch_is_supported() -> None:
    columns = {name: [] for name in _columns()}
    result = _batch(columns, 1)
    assert result == {
        "price": [],
        "steps": [],
        "risk_neutral_probability": [],
    }


def test_batch_rejects_mismatched_columns() -> None:
    columns = _columns()
    columns["spots"] = columns["spots"][:-1]
    with pytest.raises(ValueError, match="spots must have the same length"):
        _batch(columns, 1)


def test_batch_identifies_invalid_row_before_starting_workers() -> None:
    columns = _columns()
    columns["spots"][2] = -1.0
    with pytest.raises(ValueError, match="index 2"):
        _batch(columns, 4)


@pytest.mark.parametrize("threads", [0, 257])
def test_batch_rejects_invalid_thread_counts(threads: int) -> None:
    with pytest.raises(ValueError, match="thread count"):
        _batch(_columns(), threads)


def test_batch_identifies_invalid_enum_row() -> None:
    columns = _columns()
    columns["option_types"][1] = "straddle"
    with pytest.raises(ValueError, match="index 1"):
        _batch(columns, 2)


def test_lattice_feasibility_is_strict() -> None:
    with pytest.raises(ValueError, match="strictly between zero and one"):
        crr_lattice_parameters(100.0, 100.0, 1.0, 1.0, 0.0, 0.01, 1)
