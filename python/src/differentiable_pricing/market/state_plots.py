"""Deterministic SVG diagnostics for the Task 9B reconstruction report.

Figures are hand-rendered SVG rather than produced by a plotting library. That
keeps the project free of a new dependency whose output is version-sensitive,
and it makes the bytes reproducible: given the same report the same file comes
out, with no timestamps, no random identifiers and no font metrics baked in.

Every figure here is derived from proprietary quote-level data and is written
beneath a git-ignored root. None of them may be committed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final
from xml.sax.saxutils import escape

INK: Final = "#172033"
MUTED: Final = "#536078"
GRID: Final = "#dce2ea"
AXIS: Final = "#9aa4b8"
BACKGROUND: Final = "#ffffff"

SERIES_COLOURS: Final = (
    "#2563eb",
    "#0f9d76",
    "#d97706",
    "#7c3aed",
    "#dc2626",
    "#0891b2",
)
"""Cycled in declaration order, so a series keeps its colour across renders."""

MARGIN_LEFT: Final = 88.0
MARGIN_RIGHT: Final = 210.0
MARGIN_TOP: Final = 68.0
MARGIN_BOTTOM: Final = 74.0


class PlotError(RuntimeError):
    """Raised when a figure cannot be rendered from the values supplied."""


@dataclass(frozen=True, slots=True)
class Series:
    """One labelled polyline-and-marker series."""

    label: str
    points: tuple[tuple[float, float], ...]
    marker_only: bool = False


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 13,
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
    x1: float, y1: float, x2: float, y2: float, *, stroke: str, width: float = 1.0,
    dash: str | None = None,
) -> str:
    dashed = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
        f'stroke="{stroke}" stroke-width="{width:.1f}"{dashed}/>'
    )


def _document(width: int, height: int, title: str, body: Sequence[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        f"<title>{escape(title)}</title>\n"
        "<style>text { font-family: Inter, ui-sans-serif, system-ui, sans-serif; }</style>\n"
        f'<rect width="{width}" height="{height}" rx="18" fill="{BACKGROUND}"/>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def _nice_bounds(low: float, high: float) -> tuple[float, float, float]:
    """Return ``(low, high, step)`` spanning the data on a round grid.

    Deterministic: no data-dependent branching beyond the magnitude of the
    range, so the same inputs always produce the same ticks.
    """
    if not math.isfinite(low) or not math.isfinite(high):
        raise PlotError("axis bounds must be finite")
    if high < low:
        raise PlotError(f"inverted axis bounds [{low}, {high}]")
    if high == low:
        pad = abs(high) * 0.05 if high != 0.0 else 0.5
        low, high = low - pad, high + pad
    span = high - low
    raw_step = span / 5.0
    magnitude = 10.0 ** math.floor(math.log10(raw_step))
    for multiple in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = multiple * magnitude
        if step >= raw_step:
            break
    start = math.floor(low / step) * step
    stop = math.ceil(high / step) * step
    return start, stop, step


def _format_tick(value: float, step: float) -> str:
    if step >= 1.0:
        return f"{value:,.0f}"
    digits = min(8, max(0, -math.floor(math.log10(step)) + 1))
    return f"{value:.{digits}f}"


def render_panel(
    *,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    series: Sequence[Series],
    footnote: str,
    width: int = 980,
    height: int = 560,
) -> str:
    """Render one titled scatter/line panel with a legend and a footnote."""
    usable = [entry for entry in series if entry.points]
    if not usable:
        raise PlotError(f"figure '{title}' has no plottable points")
    xs = [x for entry in usable for x, _ in entry.points]
    ys = [y for entry in usable for _, y in entry.points]
    if not all(math.isfinite(value) for value in (*xs, *ys)):
        raise PlotError(f"figure '{title}' contains a non-finite coordinate")

    x_low, x_high, x_step = _nice_bounds(min(xs), max(xs))
    y_low, y_high, y_step = _nice_bounds(min(ys), max(ys))
    plot_width = width - MARGIN_LEFT - MARGIN_RIGHT
    plot_height = height - MARGIN_TOP - MARGIN_BOTTOM

    def sx(value: float) -> float:
        return MARGIN_LEFT + (value - x_low) / (x_high - x_low) * plot_width

    def sy(value: float) -> float:
        return MARGIN_TOP + plot_height - (value - y_low) / (y_high - y_low) * plot_height

    body: list[str] = [
        _text(MARGIN_LEFT, 30, title, size=17, weight=600),
        _text(MARGIN_LEFT, 50, subtitle, size=12, fill=MUTED),
    ]

    ticks = round((y_high - y_low) / y_step)
    for index in range(ticks + 1):
        value = y_low + index * y_step
        y = sy(value)
        body.append(_line(MARGIN_LEFT, y, MARGIN_LEFT + plot_width, y, stroke=GRID))
        body.append(
            _text(MARGIN_LEFT - 10, y + 4, _format_tick(value, y_step), size=11,
                  anchor="end", fill=MUTED)
        )
    ticks = round((x_high - x_low) / x_step)
    for index in range(ticks + 1):
        value = x_low + index * x_step
        x = sx(value)
        body.append(
            _line(x, MARGIN_TOP, x, MARGIN_TOP + plot_height, stroke=GRID, dash="2 4")
        )
        body.append(
            _text(x, MARGIN_TOP + plot_height + 20, _format_tick(value, x_step), size=11,
                  anchor="middle", fill=MUTED)
        )

    body.append(
        _line(MARGIN_LEFT, MARGIN_TOP + plot_height, MARGIN_LEFT + plot_width,
              MARGIN_TOP + plot_height, stroke=AXIS, width=1.4)
    )
    body.append(
        _line(MARGIN_LEFT, MARGIN_TOP, MARGIN_LEFT, MARGIN_TOP + plot_height,
              stroke=AXIS, width=1.4)
    )

    for index, entry in enumerate(usable):
        colour = SERIES_COLOURS[index % len(SERIES_COLOURS)]
        ordered = sorted(entry.points)
        if not entry.marker_only and len(ordered) > 1:
            path = " ".join(
                f"{'M' if position == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}"
                for position, (x, y) in enumerate(ordered)
            )
            body.append(
                f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="1.6"/>'
            )
        for x, y in ordered:
            body.append(
                f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.6" fill="{colour}"/>'
            )
        legend_y = MARGIN_TOP + 16 + index * 22
        legend_x = MARGIN_LEFT + plot_width + 24
        body.append(
            f'<rect x="{legend_x:.1f}" y="{legend_y - 8:.1f}" width="18" height="8" '
            f'rx="4" fill="{colour}"/>'
        )
        body.append(_text(legend_x + 26, legend_y, entry.label, size=11, fill="#354158"))

    body.append(
        _text(MARGIN_LEFT + plot_width / 2, height - 34, x_label, size=12,
              anchor="middle", fill=MUTED)
    )
    body.append(
        f'<text x="{20:.1f}" y="{MARGIN_TOP + plot_height / 2:.1f}" font-size="12" '
        f'text-anchor="middle" fill="{MUTED}" '
        f'transform="rotate(-90 20 {MARGIN_TOP + plot_height / 2:.1f})">'
        f"{escape(y_label)}</text>"
    )
    body.append(_text(MARGIN_LEFT, height - 14, footnote, size=10, fill=MUTED))
    return _document(width, height, title, body)
