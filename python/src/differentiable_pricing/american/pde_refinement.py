"""Exploratory refinement study for the task 9C-A finite-difference oracle.

The study reports how the oracle's price moves as the spot grid, the time grid
and the truncated spot domain are refined, one axis at a time and then jointly.
It is evidence, not a gate:

- No threshold here is an acceptance criterion, and none was chosen after
  seeing a result. The observed orders are printed as measurements.
- No grid policy is selected. Task 9C-A supports deterministic pricing only;
  label-policy selection and frozen acceptance gates belong to task 9C-B.
- The Black--Scholes and CRR references carry their own error. Against CRR the
  study reports a cross-engine gap, never a convergence order.

Run it when numerical evidence is requested. It is deliberately not part of CI:
the finest American case is a production-sized PSOR solve, and the whole study
takes roughly ten minutes of single-core time.

    python -m differentiable_pricing.american.pde_refinement
    python -m differentiable_pricing.american.pde_refinement --output report.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from differentiable_pricing import _pde, black_scholes, crr_price, pde_price

STUDY_VERSION: Final = "american-pde-refinement/1"
PROVENANCE: Final = "exploratory_pilot"

# One base contract per family, declared before any result was observed.
BASE_SPOT: Final = 100.0
BASE_STRIKE: Final = 100.0
BASE_EXPIRY: Final = 1.0
BASE_RATE: Final = 0.05
BASE_CARRY: Final = 0.0
BASE_VOLATILITY: Final = 0.2
BASE_SPOT_MAXIMUM: Final = 400.0
BASE_RANNACHER_STEPS: Final = 2
BASE_PSOR_TOLERANCE: Final = 1.0e-11
BASE_PSOR_RELAXATION: Final = 1.2
BASE_PSOR_ITERATION_LIMIT: Final = 200_000
CRR_REFERENCE_STEPS: Final = 8192
DIVIDEND_SCHEDULE: Final = ((0.35, 1.5), (0.75, 1.5))


@dataclass(frozen=True)
class Rung:
    """One point of a refinement ladder."""

    spot_intervals: int
    time_steps: int
    spot_maximum: float
    price: float
    spot_step: float
    time_steps_used: int
    solver_status: str
    discretization_accuracy: str
    psor_total_iterations: int
    maximum_relative_lcp_residual: float
    seconds: float
    error: float | None
    order: float | None


def _solve(
    *,
    option_type: str,
    exercise_style: str,
    spot_intervals: int,
    time_steps: int,
    spot_maximum: float,
    dividends: Sequence[tuple[float, float]] = (),
    carry: float = BASE_CARRY,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    result = pde_price(
        option_type=option_type,
        exercise_style=exercise_style,
        spot=BASE_SPOT,
        strike=BASE_STRIKE,
        valuation_time=0.0,
        expiry_time=BASE_EXPIRY,
        volatility=BASE_VOLATILITY,
        continuous_carry=carry,
        curve_times=[0.0, BASE_EXPIRY],
        curve_log_discounts=[0.0, -BASE_RATE * BASE_EXPIRY],
        dividends=list(dividends),
        settlement="cash",
        contract_multiplier=100.0,
        spot_intervals=spot_intervals,
        time_steps=time_steps,
        spot_maximum=spot_maximum,
        rannacher_steps=BASE_RANNACHER_STEPS,
        psor_tolerance=BASE_PSOR_TOLERANCE,
        psor_relaxation=BASE_PSOR_RELAXATION,
        psor_maximum_iterations=BASE_PSOR_ITERATION_LIMIT,
    )
    return result, time.perf_counter() - started


def _ladder(
    *,
    option_type: str,
    exercise_style: str,
    levels: Sequence[tuple[int, int, float]],
    reference: float | None,
    dividends: Sequence[tuple[float, float]] = (),
    carry: float = BASE_CARRY,
) -> list[Rung]:
    rungs: list[Rung] = []
    for spot_intervals, time_steps, spot_maximum in levels:
        result, seconds = _solve(
            option_type=option_type,
            exercise_style=exercise_style,
            spot_intervals=spot_intervals,
            time_steps=time_steps,
            spot_maximum=spot_maximum,
            dividends=dividends,
            carry=carry,
        )
        price = float(result["price"])
        error = None if reference is None else abs(price - reference)
        order = None
        if error is not None and rungs and rungs[-1].error not in (None, 0.0) and error > 0.0:
            previous = rungs[-1].error
            assert previous is not None
            order = math.log2(previous / error)
        rungs.append(
            Rung(
                spot_intervals=spot_intervals,
                time_steps=time_steps,
                spot_maximum=spot_maximum,
                price=price,
                spot_step=float(result["spot_step"]),
                time_steps_used=int(result["time_steps"]),
                solver_status=str(result["solver_status"]),
                discretization_accuracy=str(result["discretization_accuracy"]),
                psor_total_iterations=int(result["psor_total_iterations"]),
                maximum_relative_lcp_residual=float(result["maximum_relative_lcp_residual"]),
                seconds=seconds,
                error=error,
                order=order,
            )
        )
    return rungs


def _print_table(title: str, note: str, rungs: Sequence[Rung], error_label: str) -> None:
    print(f"\n{title}")
    print(f"  {note}")
    header = (
        f"  {'S nodes':>8} {'dS':>9} {'steps':>7} {'S_max':>7} "
        f"{'price':>16} {error_label:>12} {'ratio':>7} {'PSOR its':>10} {'sec':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for rung in rungs:
        error = "" if rung.error is None else f"{rung.error:.3e}"
        order = "" if rung.order is None else f"{2 ** rung.order:.2f}x"
        print(
            f"  {rung.spot_intervals:>8} {rung.spot_step:>9.4f} {rung.time_steps_used:>7} "
            f"{rung.spot_maximum:>7.0f} {rung.price:>16.10f} {error:>12} {order:>7} "
            f"{rung.psor_total_iterations:>10} {rung.seconds:>7.2f}"
        )


def run_study() -> dict[str, Any]:
    """Run every ladder and return the report payload."""
    european_reference = float(
        black_scholes(
            "call", BASE_SPOT, BASE_STRIKE, BASE_EXPIRY, BASE_RATE, BASE_CARRY, BASE_VOLATILITY
        )["price"]
    )
    american_reference = float(
        crr_price(
            "put",
            "american",
            BASE_SPOT,
            BASE_STRIKE,
            BASE_EXPIRY,
            BASE_RATE,
            BASE_CARRY,
            BASE_VOLATILITY,
            CRR_REFERENCE_STEPS,
        )["price"]
    )

    ladders: dict[str, dict[str, Any]] = {}

    ladders["european_space"] = {
        "note": (
            "European call, time steps held at 3200 so the spatial term dominates; "
            "reference is analytic Black-Scholes."
        ),
        "error_label": "|BS error|",
        "rungs": _ladder(
            option_type="call",
            exercise_style="european",
            levels=[
                (200, 3200, BASE_SPOT_MAXIMUM),
                (400, 3200, BASE_SPOT_MAXIMUM),
                (800, 3200, BASE_SPOT_MAXIMUM),
                (1600, 3200, BASE_SPOT_MAXIMUM),
            ],
            reference=european_reference,
        ),
    }

    ladders["european_time"] = {
        "note": (
            "European call, spot intervals held at 3200 so the time term dominates; "
            "reference is analytic Black-Scholes. The residual spatial error at this "
            "resolution is about 3.9e-5 and floors the finest rungs: the ratio decaying "
            "there is that floor, not a loss of time accuracy."
        ),
        "error_label": "|BS error|",
        "rungs": _ladder(
            option_type="call",
            exercise_style="european",
            levels=[
                (3200, 25, BASE_SPOT_MAXIMUM),
                (3200, 50, BASE_SPOT_MAXIMUM),
                (3200, 100, BASE_SPOT_MAXIMUM),
                (3200, 200, BASE_SPOT_MAXIMUM),
                (3200, 400, BASE_SPOT_MAXIMUM),
            ],
            reference=european_reference,
        ),
    }

    ladders["european_joint"] = {
        "note": "European call, both axes refined together; reference is analytic Black-Scholes.",
        "error_label": "|BS error|",
        "rungs": _ladder(
            option_type="call",
            exercise_style="european",
            levels=[
                (200, 100, BASE_SPOT_MAXIMUM),
                (400, 200, BASE_SPOT_MAXIMUM),
                (800, 400, BASE_SPOT_MAXIMUM),
                (1600, 800, BASE_SPOT_MAXIMUM),
                (3200, 1600, BASE_SPOT_MAXIMUM),
            ],
            reference=european_reference,
        ),
    }

    domain_rungs = _ladder(
        option_type="call",
        exercise_style="european",
        levels=[
            (150, 400, 150.0),
            (200, 400, 200.0),
            (400, 400, 400.0),
            (800, 400, 800.0),
            (1600, 400, 1600.0),
        ],
        reference=european_reference,
    )
    ladders["european_domain"] = {
        "note": (
            "European call, spot step held at 1.0 while the truncated domain widens. "
            "The residual error is the fixed space/time error, not truncation."
        ),
        "error_label": "|BS error|",
        "rungs": domain_rungs,
    }

    ladders["american_joint"] = {
        "note": (
            "American put, both axes refined together. The reference is CRR at "
            f"{CRR_REFERENCE_STEPS} steps, which carries its own error, so this column "
            "is a cross-engine gap and the ratio is not a convergence order."
        ),
        "error_label": "|CRR gap|",
        "rungs": _ladder(
            option_type="put",
            exercise_style="american",
            levels=[
                (200, 100, BASE_SPOT_MAXIMUM),
                (400, 200, BASE_SPOT_MAXIMUM),
                (800, 400, BASE_SPOT_MAXIMUM),
                (1600, 800, BASE_SPOT_MAXIMUM),
            ],
            reference=american_reference,
        ),
    }

    # No closed form and no independent engine covers discrete cash dividends,
    # so this ladder is measured against its own finest rung. That is internal
    # self-consistency, not truth.
    dividend_levels = [
        (200, 100, BASE_SPOT_MAXIMUM),
        (400, 200, BASE_SPOT_MAXIMUM),
        (800, 400, BASE_SPOT_MAXIMUM),
        (1600, 800, BASE_SPOT_MAXIMUM),
        (3200, 1600, BASE_SPOT_MAXIMUM),
    ]
    finest, _ = _solve(
        option_type="call",
        exercise_style="american",
        spot_intervals=6400,
        time_steps=3200,
        spot_maximum=BASE_SPOT_MAXIMUM,
        dividends=DIVIDEND_SCHEDULE,
    )
    ladders["american_dividend_self_consistency"] = {
        "note": (
            "American call with cash dividends of 1.5 at t=0.35 and t=0.75, measured "
            "against this study's own 6400x3200 rung. Self-consistency, not truth: no "
            "closed form and no independent engine covers this contract."
        ),
        "error_label": "|self gap|",
        "rungs": _ladder(
            option_type="call",
            exercise_style="american",
            levels=dividend_levels,
            reference=float(finest["price"]),
            dividends=DIVIDEND_SCHEDULE,
        ),
    }

    return {
        "study_version": STUDY_VERSION,
        "provenance": PROVENANCE,
        "selects_a_label_policy": False,
        "engine": {
            "pde_header_sha256": str(_pde.__pde_header_sha256__),
            "pde_source_sha256": str(_pde.__pde_source_sha256__),
            "pde_implementation_sha256": str(_pde.__pde_implementation_sha256__),
            "build_configuration": str(_pde.__build_configuration__),
            "cxx_compiler": str(_pde.__cxx_compiler__),
        },
        "contract": {
            "spot": BASE_SPOT,
            "strike": BASE_STRIKE,
            "expiry_time": BASE_EXPIRY,
            "rate": BASE_RATE,
            "continuous_carry": BASE_CARRY,
            "volatility": BASE_VOLATILITY,
            "dividend_schedule": [list(item) for item in DIVIDEND_SCHEDULE],
        },
        "solver": {
            "rannacher_steps": BASE_RANNACHER_STEPS,
            "psor_tolerance": BASE_PSOR_TOLERANCE,
            "psor_relaxation": BASE_PSOR_RELAXATION,
            "psor_maximum_iterations": BASE_PSOR_ITERATION_LIMIT,
        },
        "status_contract": {
            "solver_status": "discrete_system_converged is a linear/LCP solve result only",
            "discretization_accuracy": "not_assessed unless an external reference or refinement "
            "study measures continuum discretization error",
        },
        "references": {
            "european_black_scholes": european_reference,
            "american_crr": american_reference,
            "american_crr_steps": CRR_REFERENCE_STEPS,
            "american_dividend_finest": float(finest["price"]),
        },
        "ladders": {
            name: {
                "note": ladder["note"],
                "error_label": ladder["error_label"],
                "rungs": [asdict(rung) for rung in ladder["rungs"]],
            }
            for name, ladder in ladders.items()
        },
    }


def print_report(report: dict[str, Any]) -> None:
    """Print every ladder as a table."""
    print(f"{report['study_version']}  provenance={report['provenance']}")
    print(
        f"engine pde_implementation_sha256={report['engine']['pde_implementation_sha256'][:16]}..."
        f"  build={report['engine']['build_configuration']}"
    )
    references = report["references"]
    print(
        f"references: Black-Scholes call={references['european_black_scholes']:.10f}  "
        f"CRR-{references['american_crr_steps']} American put={references['american_crr']:.10f}"
    )
    status_contract = report["status_contract"]
    print(
        f"status: solver_status={status_contract['solver_status']}; "
        f"discretization_accuracy={status_contract['discretization_accuracy']}"
    )
    titles = {
        "european_space": "1. Spot-grid refinement",
        "european_time": "2. Time-grid refinement",
        "european_joint": "3. Joint refinement, European",
        "european_domain": "4. Spot-domain truncation",
        "american_joint": "5. Joint refinement, American put",
        "american_dividend_self_consistency": "6. Joint refinement, American call with dividends",
    }
    for name, ladder in report["ladders"].items():
        _print_table(
            titles.get(name, name),
            ladder["note"],
            [Rung(**rung) for rung in ladder["rungs"]],
            ladder["error_label"],
        )
    print(
        "\nThe 'ratio' column is the factor by which the previous rung's error shrank. "
        "\nFor a second-order scheme joint refinement gives 4x. No value here is an "
        "\nacceptance gate and no grid policy is selected: that is task 9C-B."
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional path for a JSON copy of the printed report",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the study, print it, and optionally write the JSON report."""
    arguments = build_parser().parse_args(argv)
    report = run_study()
    print_report(report)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nwrote {arguments.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
