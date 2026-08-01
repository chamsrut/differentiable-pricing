#!/usr/bin/env python3
"""Render deterministic SVG figures from the frozen American LSM cross-check snapshot.

Every figure is built only from ``docs/results/american_lsm_crosscheck_results_v1.json``.
The raw report is never read here, so the figures stay reproducible in CI where
the ignored artifact is absent. Output contains no timestamps, random
identifiers, or environment-dependent metadata; ``--check`` fails when a
checked-in figure differs from a fresh render.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final
from xml.sax.saxutils import escape

sys.path.insert(0, str(Path(__file__).resolve().parent))

from freeze_american_lsm_results import (
    FreezeError,
    load_json,
    validate_snapshot,
    write_atomic,
)

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT: Final = (
    PROJECT_ROOT / "docs/results/american_lsm_crosscheck_results_v1.json"
)
DEFAULT_OUTPUT_DIRECTORY: Final = PROJECT_ROOT / "docs/figures"

INK: Final = "#172033"
MUTED: Final = "#536078"
GRID: Final = "#dce2ea"
AXIS: Final = "#9aa4b8"
MEAN_GAP: Final = "#2563eb"
MAX_GAP: Final = "#7c3aed"
NOISE: Final = "#d97706"
DETERMINISTIC: Final = "#0f9d76"


class PlotError(RuntimeError):
    """Raised when the snapshot cannot be rendered or a checked-in figure is stale."""


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 14,
    weight: int = 400,
    anchor: str = "start",
    fill: str = INK,
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">'
        f"{escape(value)}</text>"
    )


def _line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str,
    width: float = 1.0,
    dash: str | None = None,
) -> str:
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width:.1f}"{dashed}/>'
    )


def _rect(
    x: float, y: float, width: float, height: float, *, fill: str, radius: float = 3.0
) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(width, 0.0):.1f}" '
        f'height="{height:.1f}" rx="{radius:.1f}" fill="{fill}"/>'
    )


def _document(width: int, height: int, title: str, body: Sequence[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        f"<title>{escape(title)}</title>\n"
        "<style>text { font-family: Inter, ui-sans-serif, system-ui, sans-serif; }</style>\n"
        f'<rect width="{width}" height="{height}" rx="18" fill="#ffffff"/>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def _swatch(x: float, y: float, label: str, colour: str, *, size: int = 12) -> list[str]:
    return [
        _rect(x, y - 8, 22.0, 9.0, fill=colour, radius=4.0),
        _text(x + 30, y, label, size=size, fill="#354158"),
    ]


def _decade_bounds(values: Sequence[float]) -> tuple[float, float]:
    """Return inclusive powers of ten bracketing every strictly positive value."""
    positive = [v for v in values if v > 0.0]
    if not positive:
        raise PlotError("a logarithmic axis needs at least one positive value")
    low = math.floor(math.log10(min(positive)))
    high = math.ceil(math.log10(max(positive)))
    if high == low:
        high = low + 1
    return float(low), float(high)


def _decade_label(exponent: int) -> str:
    if exponent >= 0:
        return str(10**exponent)
    return f"0.{'0' * (-exponent - 1)}1"


def _wrap(value: str, limit: int) -> list[str]:
    """Greedily wrap ``value`` to at most ``limit`` characters per line.

    Character counting rather than font metrics keeps the render deterministic
    and dependency-free; the limits are chosen conservatively for the 11px
    label size actually used.
    """
    lines: list[str] = []
    current = ""
    for word in value.split():
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > limit:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _experiment_comparison(snapshot: Mapping[str, Any]) -> str:
    experiments = list(snapshot["experiments"])
    width, height = 1180, 740
    left, right, top = 300.0, 120.0, 200.0
    chart_width = width - left - right
    row_height = 68.0
    bar_height = 13.0

    series = (
        ("mean_crr_reference_minus_lsm", "Mean CRR - LSM", MEAN_GAP),
        ("maximum_absolute_crr_reference_minus_lsm", "Max |CRR - LSM|", MAX_GAP),
        ("mean_stochastic_valuation_standard_error", "Mean valuation-only SE", NOISE),
    )
    values = [abs(float(e[key])) for e in experiments for key, _, _ in series]
    low, high = _decade_bounds(values)

    def x_of(value: float) -> float:
        if value <= 0.0:
            return left
        position = (math.log10(value) - low) / (high - low)
        return left + chart_width * min(max(position, 0.0), 1.0)

    body = [
        _text(36, 44, "American LSM cross-check: experiment comparison", size=25, weight=700),
        _text(
            36,
            72,
            "Price units, log scale. CRR - LSM is the gap to the 8,192-step CRR "
            "adjacent-average reference.",
            size=14,
            fill=MUTED,
        ),
        _text(
            36,
            94,
            "The standard error is valuation sampling noise only. It is NOT a "
            "total-error bar: it excludes policy-fitting",
            size=13,
            fill="#b42318",
        ),
        _text(
            36,
            113,
            "error and discrete-exercise (grid) bias, which are systematic and do "
            "not shrink with more valuation paths.",
            size=13,
            fill="#b42318",
        ),
    ]
    legend_x = 36.0
    for _, label, colour in series:
        body.extend(_swatch(legend_x, 141, label, colour))
        legend_x += 250.0

    for exponent in range(int(low), int(high) + 1):
        x = x_of(10.0**exponent)
        body.append(
            _line(x, top - 16, x, top + row_height * len(experiments) - 20, stroke=GRID)
        )
        body.append(
            _text(
                x,
                top - 26,
                _decade_label(exponent),
                size=12,
                anchor="middle",
                fill="#657189",
            )
        )
    body.append(
        _text(
            left + chart_width / 2.0,
            top + row_height * len(experiments) + 18,
            "Price difference (log scale)",
            size=13,
            anchor="middle",
            fill=MUTED,
        )
    )

    for index, experiment in enumerate(experiments):
        y = top + index * row_height
        primary = experiment["role"] == "primary"
        # Three stacked lines keep the longest role name inside the left
        # margin at GitHub's rendered width instead of overflowing the canvas.
        body.append(
            _text(
                left - 16,
                y + 2,
                str(experiment["name"]),
                size=13,
                weight=700 if primary else 500,
                anchor="end",
            )
        )
        body.append(
            _text(left - 16, y + 19, str(experiment["role"]), size=11, anchor="end", fill=MUTED)
        )
        body.append(
            _text(
                left - 16,
                y + 35,
                f"{experiment['exercise_steps']} steps | "
                f"degree {experiment['polynomial_degree']} | "
                f"{experiment['training_paths']:,} train",
                size=11,
                anchor="end",
                fill=MUTED,
            )
        )
        for offset, (key, _, colour) in enumerate(series):
            value = abs(float(experiment[key]))
            bar_y = y - 8 + offset * (bar_height + 3.0)
            body.append(_rect(left, bar_y, x_of(value) - left, bar_height, fill=colour))
            body.append(
                _text(
                    x_of(value) + 8,
                    bar_y + 11,
                    f"{value:.5f}",
                    size=11,
                    fill="#354158",
                )
            )
    return _document(width, height, "American LSM experiment comparison", body)


def _primary_cases(snapshot: Mapping[str, Any]) -> str:
    cases = list(snapshot["primary_cases"])
    primary_name = snapshot["study"]["primary_experiment"]
    width, height = 1180, 740
    left, right, top = 260.0, 150.0, 220.0
    chart_width = width - left - right
    row_height = 50.0

    spans = []
    for case in cases:
        gap = float(case["crr_reference_minus_lsm"])
        error = float(case["valuation_only_standard_error"])
        spans.extend((gap - error, gap + error, 0.0))
    low, high = min(spans), max(spans)
    padding = (high - low) * 0.08 or 1.0
    low, high = low - padding, high + padding

    def x_of(value: float) -> float:
        return left + chart_width * (value - low) / (high - low)

    body = [
        _text(36, 44, f"Primary experiment case results ({primary_name})", size=25, weight=700),
        _text(
            36,
            72,
            "CRR - LSM per case, with a +/- 1 valuation-only standard error bar. "
            "Positive means LSM sits below the",
            size=14,
            fill=MUTED,
        ),
        _text(
            36,
            92,
            "CRR reference, which is the expected ordering for a fixed learned "
            "stopping policy on a coarser grid.",
            size=14,
            fill=MUTED,
        ),
        _text(
            36,
            120,
            "The bar covers valuation sampling noise only. Policy-fitting error "
            "and exercise-grid bias are NOT included,",
            size=13,
            fill="#b42318",
        ),
        _text(
            36,
            139,
            "so a bar excluding zero is not evidence of a defect and is not an "
            "acceptance gate.",
            size=13,
            fill="#b42318",
        ),
    ]
    body.extend(_swatch(36, 172, "Stochastic case (+/- 1 valuation-only SE)", MEAN_GAP))
    body.extend(
        _swatch(490, 172, "Deterministic case (zero-width interval)", DETERMINISTIC)
    )

    zero_x = x_of(0.0)
    body.append(
        _line(
            zero_x,
            top - 22,
            zero_x,
            top + row_height * len(cases) - 18,
            stroke="#94a3b8",
            width=1.5,
        )
    )
    body.append(_text(zero_x, top - 32, "0", size=12, anchor="middle", fill="#657189"))
    for fraction in (0.25, 0.5, 0.75, 1.0):
        value = low + (high - low) * fraction
        x = x_of(value)
        body.append(_line(x, top - 22, x, top + row_height * len(cases) - 18, stroke=GRID))
        body.append(
            _text(x, top - 32, f"{value:.3f}", size=12, anchor="middle", fill="#657189")
        )
    body.append(
        _text(
            left + chart_width / 2.0,
            top + row_height * len(cases) + 14,
            "CRR reference minus LSM price (price units, linear scale)",
            size=13,
            anchor="middle",
            fill=MUTED,
        )
    )

    for index, case in enumerate(cases):
        y = top + index * row_height
        gap = float(case["crr_reference_minus_lsm"])
        error = float(case["valuation_only_standard_error"])
        deterministic = bool(case["valuation_interval_is_zero_width"])
        colour = DETERMINISTIC if deterministic else MEAN_GAP
        body.append(_text(left - 18, y + 4, str(case["name"]), size=13, weight=500, anchor="end"))
        body.append(
            _text(
                left - 18,
                y + 21,
                f"{case['option_type']} S={case['spot']:g} K={case['strike']:g} "
                f"T={case['maturity']:g} r={case['rate']:g} q={case['dividend_yield']:g} "
                f"v={case['volatility']:g}",
                size=10,
                anchor="end",
                fill=MUTED,
            )
        )
        if deterministic:
            # A zero-width interval has no bar to draw; a hollow marker keeps it
            # visually separate from the stochastic cases rather than implying a
            # vanishingly small but real uncertainty.
            body.append(
                f'<circle cx="{x_of(gap):.1f}" cy="{y:.1f}" r="6" fill="#ffffff" '
                f'stroke="{colour}" stroke-width="2.5"/>'
            )
        else:
            body.append(
                _line(x_of(gap - error), y, x_of(gap + error), y, stroke=colour, width=3.0)
            )
            for edge in (gap - error, gap + error):
                body.append(_line(x_of(edge), y - 6, x_of(edge), y + 6, stroke=colour, width=2.0))
            body.append(
                f'<circle cx="{x_of(gap):.1f}" cy="{y:.1f}" r="5" fill="{colour}"/>'
            )
        label = (
            f"{gap:+.2e}  zero-width"
            if deterministic
            else f"{gap:+.5f} +/- {error:.5f}"
        )
        body.append(_text(left + chart_width + 12, y + 4, label, size=11, fill="#354158"))
    return _document(width, height, "American LSM primary case results", body)


def _sensitivity(snapshot: Mapping[str, Any]) -> str:
    by_name = {str(e["name"]): e for e in snapshot["experiments"]}
    panels = (
        (
            "Basis degree",
            "at 64 steps, 32,768 training paths",
            # reference-v1 is the degree-two arm; ordering the tuple by degree
            # keeps the rendered bars in 1, 2, 3 order.
            ("degree-one-v1", "reference-v1", "degree-three-v1"),
            lambda e: f"degree {e['polynomial_degree']}",
            "Degree one is inadequate; quadratic and cubic are comparable.",
        ),
        (
            "Exercise dates",
            "at degree 2, 32,768 training paths",
            ("steps-low-v1", "reference-v1", "steps-high-v1"),
            lambda e: f"{e['exercise_steps']} steps",
            "More exercise dates at fixed training paths did not improve the gap.",
        ),
        (
            "Path budget",
            "at degree 2, 64 steps",
            ("paths-low-v1", "reference-v1", "paths-high-primary-v1"),
            lambda e: f"{e['training_paths']:,} train",
            "More paths cut valuation noise; the mean gap improved only modestly.",
        ),
    )
    width, height = 1180, 560
    panel_width = (width - 72 - 40) / 3.0
    body = [
        _text(36, 44, "Path, exercise-grid and basis sensitivity", size=25, weight=700),
        _text(
            36,
            72,
            "Each panel varies one factor. Bars are the experiment mean CRR - LSM "
            "gap (log scale); the amber marker is",
            size=14,
            fill=MUTED,
        ),
        _text(
            36,
            92,
            "the mean valuation-only standard error, which measures sampling noise "
            "alone and not total error.",
            size=14,
            fill=MUTED,
        ),
    ]

    values = [
        abs(float(by_name[name][key]))
        for _, _, names, _, _ in panels
        for name in names
        for key in ("mean_crr_reference_minus_lsm", "mean_stochastic_valuation_standard_error")
    ]
    low, high = _decade_bounds(values)
    chart_top = 172.0
    chart_height = 250.0

    def y_of(value: float) -> float:
        if value <= 0.0:
            return chart_top + chart_height
        position = (math.log10(value) - low) / (high - low)
        return chart_top + chart_height * (1.0 - min(max(position, 0.0), 1.0))

    for panel_index, (title, subtitle, names, label_of, finding) in enumerate(panels):
        origin = 72.0 + panel_index * (panel_width + 20.0)
        body.append(_text(origin, chart_top - 34, title, size=15, weight=700))
        body.append(_text(origin, chart_top - 16, subtitle, size=11, fill=MUTED))
        for exponent in range(int(low), int(high) + 1):
            y = y_of(10.0**exponent)
            body.append(_line(origin, y, origin + panel_width - 24.0, y, stroke=GRID))
            if panel_index == 0:
                body.append(
                    _text(
                        origin - 8,
                        y + 4,
                        _decade_label(exponent),
                        size=11,
                        anchor="end",
                        fill="#657189",
                    )
                )
        body.append(
            _line(
                origin,
                chart_top + chart_height,
                origin + panel_width - 24.0,
                chart_top + chart_height,
                stroke=AXIS,
            )
        )
        slot = (panel_width - 24.0) / len(names)
        for slot_index, name in enumerate(names):
            experiment = by_name[name]
            gap = abs(float(experiment["mean_crr_reference_minus_lsm"]))
            noise = abs(float(experiment["mean_stochastic_valuation_standard_error"]))
            bar_width = slot * 0.46
            x = origin + slot * slot_index + (slot - bar_width) / 2.0
            top_y = y_of(gap)
            primary = experiment["role"] == "primary"
            body.append(
                _rect(
                    x,
                    top_y,
                    bar_width,
                    chart_top + chart_height - top_y,
                    fill=MEAN_GAP if not primary else "#1d4ed8",
                    radius=2.0,
                )
            )
            body.append(
                _text(
                    x + bar_width / 2.0,
                    top_y - 8,
                    f"{gap:.4f}",
                    size=10,
                    anchor="middle",
                    fill="#354158",
                )
            )
            body.append(
                _line(
                    x - 3.0,
                    y_of(noise),
                    x + bar_width + 3.0,
                    y_of(noise),
                    stroke=NOISE,
                    width=2.5,
                )
            )
            body.append(
                _text(
                    x + bar_width / 2.0,
                    chart_top + chart_height + 18,
                    label_of(experiment),
                    size=11,
                    anchor="middle",
                )
            )
            if primary:
                body.append(
                    _text(
                        x + bar_width / 2.0,
                        chart_top + chart_height + 33,
                        "(primary)",
                        size=9,
                        anchor="middle",
                        fill=MUTED,
                    )
                )
        for line_index, line in enumerate(_wrap(finding, 44)):
            body.append(
                _text(
                    origin,
                    chart_top + chart_height + 62 + line_index * 16,
                    line,
                    size=11,
                    fill="#354158",
                )
            )

    body.extend(_swatch(72, height - 26, "Mean CRR - LSM gap (log scale)", MEAN_GAP, size=11))
    body.append(_line(430, height - 30, 452, height - 30, stroke=NOISE, width=3.0))
    body.append(
        _text(460, height - 26, "Mean valuation-only standard error", size=11, fill="#354158")
    )
    return _document(width, height, "American LSM sensitivity", body)


def render_figures(snapshot_path: Path) -> dict[str, str]:
    """Validate the snapshot and return deterministic figure contents by filename."""
    try:
        snapshot = validate_snapshot(load_json(snapshot_path, "snapshot"))
    except FreezeError as error:
        raise PlotError(str(error)) from error
    return {
        "american_lsm_experiment_comparison.svg": _experiment_comparison(snapshot),
        "american_lsm_primary_cases.svg": _primary_cases(snapshot),
        "american_lsm_sensitivity.svg": _sensitivity(snapshot),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if checked-in figures differ from a fresh deterministic render.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        figures = render_figures(arguments.snapshot)
        if arguments.check:
            for name, content in figures.items():
                path = arguments.output_directory / name
                try:
                    current = path.read_text(encoding="utf-8")
                except OSError as error:
                    raise PlotError(f"cannot read checked-in figure '{path}': {error}") from error
                if current != content:
                    raise PlotError(
                        f"checked-in figure '{path}' is stale; rerun {Path(__file__).name}"
                    )
        else:
            arguments.output_directory.mkdir(parents=True, exist_ok=True)
            for name, content in figures.items():
                # Atomic per figure: a failure mid-write must not leave a
                # truncated SVG that a later --check would compare against.
                write_atomic(arguments.output_directory / name, content)
    except PlotError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
