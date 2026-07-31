#!/usr/bin/env python3
"""Benchmark scalar and batch CRR pricing without imposing portable timing gates.

The command uses deterministic contract rows from the convergence
configuration. It records raw repetitions, warm-up count, CPU affinity,
compiler/build metadata, and exact output digests. Timings are meaningful only
on the recorded machine and are never consumed by CI as pass/fail thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from differentiable_pricing import _core
from differentiable_pricing.american.convergence import (
    ConvergenceCase,
    ConvergenceError,
    load_convergence_config,
    write_report,
)


def _positive_csv(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("every value must be positive")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("values must not contain duplicates")
    return result


def _rows(cases: Sequence[ConvergenceCase], size: int, steps: int) -> list[dict[str, Any]]:
    return [
        {
            "option_type": case.option_type,
            "exercise_style": "american",
            "spot": case.spot,
            "strike": case.strike,
            "maturity": case.maturity,
            "rate": case.rate,
            "dividend_yield": case.dividend_yield,
            "volatility": case.volatility,
            "steps": steps,
        }
        for index in range(size)
        for case in (cases[index % len(cases)],)
    ]


def _cpu_model() -> str:
    processor_identifier = os.environ.get("PROCESSOR_IDENTIFIER")
    if processor_identifier:
        return processor_identifier
    with suppress(OSError):
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith(("model name", "Hardware")):
                return line.partition(":")[2].strip()
    return platform.processor()


def _batch_columns(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    return {
        name: [row[name] for row in rows]
        for name in (
            "option_type",
            "exercise_style",
            "spot",
            "strike",
            "maturity",
            "rate",
            "dividend_yield",
            "volatility",
            "steps",
        )
    }


def _run_batch(columns: Mapping[str, list[Any]], thread_count: int) -> list[float]:
    result = _core.crr_price_batch(
        columns["option_type"],
        columns["exercise_style"],
        columns["spot"],
        columns["strike"],
        columns["maturity"],
        columns["rate"],
        columns["dividend_yield"],
        columns["volatility"],
        columns["steps"],
        thread_count,
    )
    return [float(value) for value in result["price"]]


def _run_scalar(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    return [
        float(
            _core.crr_price(
                row["option_type"],
                row["exercise_style"],
                row["spot"],
                row["strike"],
                row["maturity"],
                row["rate"],
                row["dividend_yield"],
                row["volatility"],
                row["steps"],
            )["price"]
        )
        for row in rows
    ]


def _timing_summary(
    durations: Sequence[float], last_result: Sequence[float]
) -> dict[str, Any]:
    ordered = sorted(durations)
    p95_position = 0.95 * (len(ordered) - 1)
    lower = math.floor(p95_position)
    upper = math.ceil(p95_position)
    weight = p95_position - lower
    p95 = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    digest = hashlib.sha256(
        json.dumps(last_result, allow_nan=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "durations_seconds": list(durations),
        "minimum_seconds": min(durations),
        "median_seconds": statistics.median(durations),
        "p95_seconds": p95,
        "maximum_seconds": max(durations),
        "output_sha256": digest,
    }


def run_benchmark(
    *,
    config_path: Path,
    steps: int,
    batch_sizes: Sequence[int],
    thread_counts: Sequence[int],
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    """Execute every benchmark mode and return the machine-specific report."""
    if steps <= 0:
        raise ConvergenceError("steps must be positive")
    if not batch_sizes or any(size <= 0 or size > 1_000_000 for size in batch_sizes):
        raise ConvergenceError("batch sizes must lie in [1, 1000000]")
    if not thread_counts or any(count <= 0 for count in thread_counts):
        raise ConvergenceError("thread counts must be positive")
    if warmups < 0 or repetitions <= 0:
        raise ConvergenceError("warmups must be non-negative and repetitions positive")
    config = load_convergence_config(config_path)
    if steps > int(_core.maximum_crr_steps):
        raise ConvergenceError(f"steps exceed {_core.maximum_crr_steps}")
    if any(count > int(_core.maximum_crr_batch_threads) for count in thread_counts):
        raise ConvergenceError(
            f"thread counts exceed {_core.maximum_crr_batch_threads}"
        )

    measurements: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        rows = _rows(config.cases, batch_size, steps)
        columns = _batch_columns(rows)
        expected_prices = _run_scalar(rows)
        modes: list[dict[str, Any]] = [
            {
                "mode": "scalar_python_loop",
                "batch_size": batch_size,
                "requested_thread_count": 1,
                "effective_worker_count": 1,
                "action": lambda rows=rows: _run_scalar(rows),
            }
        ]
        for thread_count in thread_counts:
            modes.append(
                {
                    "mode": "batch",
                    "batch_size": batch_size,
                    "requested_thread_count": thread_count,
                    "effective_worker_count": min(thread_count, batch_size),
                    "action": (
                        lambda columns=columns, thread_count=thread_count: _run_batch(
                            columns, thread_count
                        )
                    ),
                }
            )

        for _ in range(warmups):
            for mode in modes:
                if mode["action"]() != expected_prices:
                    raise ConvergenceError(
                        "benchmark warmups or modes produced non-identical prices"
                    )

        durations_by_mode: list[list[float]] = [[] for _ in modes]
        last_results: list[list[float]] = [[] for _ in modes]
        for repetition in range(repetitions):
            offset = repetition % len(modes)
            order = [*range(offset, len(modes)), *range(0, offset)]
            for mode_index in order:
                started = time.perf_counter_ns()
                result = modes[mode_index]["action"]()
                stopped = time.perf_counter_ns()
                if result != expected_prices:
                    raise ConvergenceError(
                        "benchmark repetitions or modes produced non-identical prices"
                    )
                durations_by_mode[mode_index].append(
                    (stopped - started) * 1.0e-9
                )
                last_results[mode_index] = result

        for mode, durations, last_result in zip(
            modes, durations_by_mode, last_results, strict=True
        ):
            timing = _timing_summary(durations, last_result)
            measurements.append(
                {
                    key: value for key, value in mode.items() if key != "action"
                }
                | {
                    "throughput_at_median_rows_per_second": batch_size
                    / timing["median_seconds"],
                    **timing,
                }
            )

    affinity: list[int] | None = None
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))
    return {
        "schema_version": "american-crr-benchmark/1",
        "warning": (
            "Machine-specific exploratory timings; not a CI gate and not directly "
            "comparable with a neural benchmark unless hardware, affinity, accuracy, "
            "batch size, and thread budget match."
        ),
        "source": {
            "config_file": config.source_name,
            "config_sha256": config.source_sha256,
            "study_name": config.name,
        },
        "runtime": {
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_model": _cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "cpu_affinity": affinity,
            "core_version": str(_core.__version__),
            "crr_header_sha256": str(_core.__crr_header_sha256__),
            "crr_implementation_sha256": str(
                _core.__crr_implementation_sha256__
            ),
            "crr_source_sha256": str(_core.__crr_source_sha256__),
            "build_configuration": str(_core.__build_configuration__),
            "cxx_compiler": str(_core.__cxx_compiler__),
        },
        "protocol": {
            "steps": steps,
            "batch_sizes": list(batch_sizes),
            "thread_counts": list(thread_counts),
            "untimed_validation_prepasses_per_batch_size": 1,
            "warmups": warmups,
            "repetitions": repetitions,
            "clock": "time.perf_counter_ns",
            "measurement_order": (
                "deterministic cyclic rotation of scalar and batch modes within "
                "each repetition"
            ),
        },
        "measurements": measurements,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1024)
    parser.add_argument("--batch-sizes", type=_positive_csv, default=(1, 16, 64))
    parser.add_argument("--thread-counts", type=_positive_csv, default=(1, 2, 4))
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=5)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the benchmark CLI."""
    arguments = build_parser().parse_args(argv)
    if arguments.steps <= 0 or arguments.warmups < 0 or arguments.repetitions <= 0:
        print(
            "error: steps and repetitions must be positive; warmups must be non-negative",
            file=sys.stderr,
        )
        return 2
    try:
        if arguments.output.exists():
            raise ConvergenceError(
                f"refusing to overwrite existing report '{arguments.output}'"
            )
        report = run_benchmark(
            config_path=arguments.config,
            steps=arguments.steps,
            batch_sizes=arguments.batch_sizes,
            thread_counts=arguments.thread_counts,
            warmups=arguments.warmups,
            repetitions=arguments.repetitions,
        )
        write_report(report, arguments.output)
    except ConvergenceError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "measurements": len(report["measurements"]),
                "output": str(arguments.output),
                "steps": arguments.steps,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
