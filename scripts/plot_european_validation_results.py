#!/usr/bin/env python3
"""Render deterministic SVGs from the versioned European validation snapshot."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final
from xml.sax.saxutils import escape

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS: Final = (
    PROJECT_ROOT / "docs/results/european_validation_results_v1.json"
)
DEFAULT_ACCEPTANCE: Final = (
    PROJECT_ROOT / "configs/european_neural_acceptance_v1.toml"
)
DEFAULT_OUTPUT_DIRECTORY: Final = PROJECT_ROOT / "docs/figures"
EXPECTED_EXPERIMENT_IDS: Final = (
    "raw_baseline",
    "long_control",
    "wide_control",
    "forward_normalized",
    "differential",
    "bounded_differential",
)
SHA256_PATTERN: Final = re.compile(r"^[0-9a-f]{64}$")
ERROR_SERIES: Final = (
    ("price_over_spot_rmse", "Price / spot", "#2563eb"),
    ("delta_rmse", "Delta", "#d97706"),
    ("gamma_rmse", "Gamma", "#7c3aed"),
    ("vega_rmse", "Vega", "#059669"),
)
GATE_PATHS: Final = {
    "price_over_spot_rmse": ("overall", "price_over_spot_rmse_max"),
    "price_over_spot_p99_absolute_error": (
        "overall",
        "price_over_spot_p99_absolute_error_max",
    ),
    "delta_rmse": ("overall", "delta_rmse_max"),
    "gamma_rmse": ("overall", "gamma_rmse_max"),
    "vega_rmse": ("overall", "vega_rmse_max"),
    "theta_rmse": ("overall", "theta_rmse_max"),
    "rho_rmse": ("overall", "rho_rmse_max"),
    "tail_price_over_spot_rmse": ("tail", "price_over_spot_rmse_max"),
    "tail_delta_rmse": ("tail", "delta_rmse_max"),
    "tail_gamma_rmse": ("tail", "gamma_rmse_max"),
}


class PlotError(RuntimeError):
    """Raised when the result snapshot or generated figures are inconsistent."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PlotError(f"cannot load result snapshot '{path}': {error}") from error
    if not isinstance(payload, dict):
        raise PlotError("result snapshot must be a JSON object")
    return payload


def _load_toml(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            payload = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise PlotError(f"cannot load acceptance configuration '{path}': {error}") from error
    if not isinstance(payload, dict):
        raise PlotError("acceptance configuration must be a TOML table")
    return payload


def _number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PlotError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise PlotError(f"{where} must be finite and non-negative")
    return result


def _validate(
    results: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if results.get("schema_version") != "european-validation-study/v1":
        raise PlotError("result snapshot schema is unsupported")
    if results.get("partition") != "validation":
        raise PlotError("result snapshot must describe the validation partition")
    if results.get("rows") != 25_000:
        raise PlotError("result snapshot must contain 25,000 validation rows")
    # Every experiment is evaluated against one dataset, so the snapshot names
    # its manifest digest once, at the result-set level.
    dataset_manifest = results.get("dataset_manifest_sha256")
    if (
        not isinstance(dataset_manifest, str)
        or SHA256_PATTERN.fullmatch(dataset_manifest) is None
    ):
        raise PlotError(
            "result snapshot must record a valid dataset_manifest_sha256 for the "
            "dataset shared by every experiment"
        )
    experiments = results.get("experiments")
    if not isinstance(experiments, list) or not all(
        isinstance(item, dict) for item in experiments
    ):
        raise PlotError("result snapshot experiments must be a list of objects")
    identifiers = tuple(item.get("id") for item in experiments)
    if identifiers != EXPECTED_EXPERIMENT_IDS:
        raise PlotError(
            f"experiment order must be {list(EXPECTED_EXPERIMENT_IDS)}, "
            f"got {list(identifiers)}"
        )

    for experiment in experiments:
        identifier = experiment["id"]
        label = experiment.get("label")
        if not isinstance(label, str) or not label:
            raise PlotError(f"experiment {identifier} has no label")
        # Provenance is required uniformly: a weights digest alone does not say
        # which artifact produced the reported metrics, so every row must also
        # name its artifact directory and that artifact's manifest digest.
        for required_digest in ("weights_sha256", "artifact_manifest_sha256"):
            value = experiment.get(required_digest)
            if value is None:
                raise PlotError(
                    f"experiment {identifier} is missing {required_digest}; every "
                    f"experiment must carry complete provenance"
                )
            if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
                raise PlotError(
                    f"experiment {identifier} has an invalid {required_digest}"
                )
        artifact = experiment.get("artifact")
        if not isinstance(artifact, str) or not artifact.startswith("artifacts/"):
            raise PlotError(
                f"experiment {identifier} must name its artifact directory under "
                f"'artifacts/'"
            )
        for optional_digest in ("source_manifest_sha256", "source_weights_sha256"):
            value = experiment.get(optional_digest)
            if value is not None and (
                not isinstance(value, str)
                or SHA256_PATTERN.fullmatch(value) is None
            ):
                raise PlotError(
                    f"experiment {identifier} has an invalid {optional_digest}"
                )
        metrics = experiment.get("metrics")
        if not isinstance(metrics, dict):
            raise PlotError(f"experiment {identifier} metrics must be an object")
        for metric in GATE_PATHS:
            _number(metrics.get(metric), f"experiment {identifier}.{metric}")
        no_arbitrage = experiment.get("no_arbitrage")
        if not isinstance(no_arbitrage, dict):
            raise PlotError(f"experiment {identifier} no_arbitrage must be an object")
        violations = no_arbitrage.get("material_violations")
        if isinstance(violations, bool) or not isinstance(violations, int):
            raise PlotError(
                f"experiment {identifier}.material_violations must be an integer"
            )
        if violations < 0 or violations > results["rows"]:
            raise PlotError(
                f"experiment {identifier}.material_violations is outside row count"
            )

    artifacts = [experiment["artifact"] for experiment in experiments]
    if len(set(artifacts)) != len(artifacts):
        raise PlotError("each experiment must name a distinct artifact directory")
    manifests = [experiment["artifact_manifest_sha256"] for experiment in experiments]
    if len(set(manifests)) != len(manifests):
        raise PlotError(
            "each experiment must have a distinct artifact manifest digest; a "
            "repeated digest means two rows report the same artifact"
        )

    bounded = experiments[-1]
    differential = experiments[-2]
    source_digest = bounded.get("source_weights_sha256")
    if source_digest != differential["weights_sha256"]:
        raise PlotError("bounded artifact source digest does not match differential weights")
    if bounded["weights_sha256"] != source_digest:
        raise PlotError("bounded artifact did not preserve its source weights")
    # Traceability metadata, not authentication: this records which manifest the
    # bounded artifact was derived from. It matches the digest the differential
    # row reports, but nothing here re-reads the source artifact, so it links
    # the two rows rather than proving the source exists or is unmodified.
    if bounded.get("source_manifest_sha256") != differential["artifact_manifest_sha256"]:
        raise PlotError("bounded artifact source manifest does not match differential")
    if bounded["artifact_manifest_sha256"] == differential["artifact_manifest_sha256"]:
        raise PlotError(
            "bounded artifact must have its own manifest: adding the output "
            "constraint changes the manifest even though the weights are identical"
        )
    if bounded.get("output_constraint") != "european_bounds_v1":
        raise PlotError("selected artifact does not declare the European bounds")
    projection = bounded.get("projection_diagnostics")
    if not isinstance(projection, dict):
        raise PlotError("selected artifact has no projection diagnostics")
    active_rows = projection.get("active_rows")
    lower_rows = projection.get("lower_bound_activations")
    upper_rows = projection.get("upper_bound_activations")
    if (
        isinstance(active_rows, bool)
        or not isinstance(active_rows, int)
        or isinstance(lower_rows, bool)
        or not isinstance(lower_rows, int)
        or isinstance(upper_rows, bool)
        or not isinstance(upper_rows, int)
        or active_rows != lower_rows + upper_rows
        or not 0 <= active_rows <= results["rows"]
    ):
        raise PlotError("selected artifact projection counts are inconsistent")
    for metric, (section, key) in GATE_PATHS.items():
        gate_table = gates.get(section)
        if not isinstance(gate_table, dict):
            raise PlotError(f"acceptance configuration is missing [{section}]")
        limit = _number(gate_table.get(key), f"acceptance {section}.{key}")
        value = _number(bounded["metrics"][metric], f"bounded {metric}")
        if value > limit:
            raise PlotError(f"selected bounded model fails {metric}: {value} > {limit}")
    no_arbitrage_gates = gates.get("no_arbitrage")
    if not isinstance(no_arbitrage_gates, dict):
        raise PlotError("acceptance configuration is missing [no_arbitrage]")
    maximum_material_violations = _number(
        no_arbitrage_gates.get("material_violations_max"),
        "acceptance no_arbitrage.material_violations_max",
    )
    if (
        bounded["no_arbitrage"]["material_violations"]
        > maximum_material_violations
    ):
        raise PlotError("selected bounded model fails the material-arbitrage gate")
    return experiments


def _text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 14,
    weight: int = 400,
    anchor: str = "start",
    fill: str = "#172033",
) -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="{anchor}" fill="{fill}">'
        f"{escape(value)}</text>"
    )


def _rotated_text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 14,
    weight: int = 400,
    fill: str = "#172033",
) -> str:
    """Render a vertical, bottom-to-top axis label centred on ``(x, y)``."""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-weight="{weight}" text-anchor="middle" fill="{fill}" '
        f'transform="rotate(-90 {x:.1f} {y:.1f})">'
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


def _document(
    width: int,
    height: int,
    title: str,
    body: Sequence[str],
) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        f"<title>{escape(title)}</title>\n"
        "<style>text { font-family: Inter, ui-sans-serif, system-ui, sans-serif; }</style>\n"
        f'<rect width="{width}" height="{height}" rx="18" fill="#ffffff"/>\n'
        + "\n".join(body)
        + "\n</svg>\n"
    )


def _error_progression(
    experiments: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Any],
) -> str:
    width, height = 1000, 570
    left, right, top, bottom = 116.0, 38.0, 112.0, 118.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_positions = [
        left + index * plot_width / (len(experiments) - 1)
        for index in range(len(experiments))
    ]

    def y_position(ratio: float) -> float:
        return top + (1.0 - math.log10(ratio)) * plot_height / 2.0

    body = [
        _text(36, 42, "Validation error progression", size=25, weight=700),
        _text(
            36,
            70,
            "Each metric is divided by its acceptance limit; below 1.0 passes. "
            "Vertical axis is log scale.",
            size=14,
            fill="#536078",
        ),
        # The axis is log10-spaced, so it is named as such rather than leaving
        # the reader to infer it from the irregular tick spacing.
        _rotated_text(
            30.0,
            top + plot_height / 2.0,
            "RMSE ÷ acceptance limit (log scale)",
            size=13,
            fill="#536078",
        ),
    ]
    legend_x = 420.0
    for index, (_, label, color) in enumerate(ERROR_SERIES):
        x = legend_x + index * 137.0
        body.append(_line(x, 91, x + 22, 91, stroke=color, width=3.0))
        body.append(_text(x + 29, 96, label, size=12, fill="#354158"))

    ticks = (0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
    for tick in ticks:
        y = y_position(tick)
        is_gate = tick == 1.0
        body.append(
            _line(
                left,
                y,
                width - right,
                y,
                stroke="#dc2626" if is_gate else "#dce2ea",
                width=1.5 if is_gate else 1.0,
                dash="7 6" if is_gate else None,
            )
        )
        body.append(
            _text(
                left - 14,
                y + 5,
                f"{tick:g}x",
                size=12,
                anchor="end",
                fill="#b42318" if is_gate else "#657189",
            )
        )
    body.append(_text(left + 8, y_position(1.0) - 10, "Pass gate", size=12, fill="#b42318"))

    for metric, _, color in ERROR_SERIES:
        section, key = GATE_PATHS[metric]
        limit = float(gates[section][key])
        ratios = [
            float(experiment["metrics"][metric]) / limit
            for experiment in experiments
        ]
        points = " ".join(
            f"{x:.1f},{y_position(ratio):.1f}"
            for x, ratio in zip(x_positions, ratios, strict=True)
        )
        body.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        for x, ratio in zip(x_positions, ratios, strict=True):
            y = y_position(ratio)
            body.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#ffffff" '
                f'stroke="{color}" stroke-width="3"/>'
            )

    for x, experiment in zip(x_positions, experiments, strict=True):
        body.append(_line(x, top + plot_height, x, top + plot_height + 7, stroke="#7c879b"))
        body.append(
            _text(
                x,
                top + plot_height + 30,
                str(experiment["label"]),
                size=12,
                weight=600,
                anchor="middle",
            )
        )
    body.append(
        _text(
            width / 2,
            height - 28,
            "Ordered controlled experiments on the same 25,000-row validation partition",
            size=12,
            anchor="middle",
            fill="#657189",
        )
    )
    return _document(width, height, "Validation error progression", body)


def _arbitrage_progression(
    experiments: Sequence[Mapping[str, Any]],
) -> str:
    width, height = 1000, 540
    left, right, top, bottom = 92.0, 38.0, 102.0, 110.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    maximum = 2_500
    step = plot_width / len(experiments)
    bar_width = min(82.0, step * 0.58)
    colors = ("#64748b", "#64748b", "#64748b", "#7c3aed", "#d97706", "#059669")

    def y_position(value: float) -> float:
        return top + plot_height * (1.0 - value / maximum)

    body = [
        _text(36, 42, "Material no-arbitrage violations", size=25, weight=700),
        _text(
            36,
            70,
            "Count above the fixed 1e-6 relative tolerance; the acceptance gate is zero.",
            size=14,
            fill="#536078",
        ),
    ]
    for tick in range(0, maximum + 1, 500):
        y = y_position(tick)
        body.append(_line(left, y, width - right, y, stroke="#dce2ea"))
        body.append(
            _text(
                left - 14,
                y + 5,
                f"{tick:,}",
                size=12,
                anchor="end",
                fill="#657189",
            )
        )

    for index, (experiment, color) in enumerate(
        zip(experiments, colors, strict=True)
    ):
        value = int(experiment["no_arbitrage"]["material_violations"])
        x = left + step * index + (step - bar_width) / 2.0
        y = y_position(value)
        visible_height = max(top + plot_height - y, 3.0)
        visible_y = top + plot_height - visible_height
        body.append(
            f'<rect x="{x:.1f}" y="{visible_y:.1f}" width="{bar_width:.1f}" '
            f'height="{visible_height:.1f}" rx="7" fill="{color}"/>'
        )
        body.append(
            _text(
                x + bar_width / 2.0,
                visible_y - 10,
                f"{value:,}",
                size=13,
                weight=700,
                anchor="middle",
                fill=color,
            )
        )
        body.append(
            _text(
                x + bar_width / 2.0,
                top + plot_height + 29,
                str(experiment["label"]),
                size=12,
                weight=600,
                anchor="middle",
            )
        )

    body.append(
        _text(
            width / 2,
            height - 24,
            "The bounded artifact reuses the differential model's exact weight bytes.",
            size=12,
            anchor="middle",
            fill="#657189",
        )
    )
    return _document(width, height, "Material no-arbitrage violations", body)


def render_figures(
    results_path: Path,
    acceptance_path: Path,
) -> dict[str, str]:
    """Validate the snapshot and return deterministic figure contents."""
    results = _load_json(results_path)
    gates = _load_toml(acceptance_path)
    experiments = _validate(results, gates)
    return {
        "european_validation_error_progression.svg": _error_progression(
            experiments,
            gates,
        ),
        "european_validation_arbitrage_progression.svg": _arbitrage_progression(
            experiments,
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--acceptance", type=Path, default=DEFAULT_ACCEPTANCE)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if checked-in figures differ from a fresh deterministic render.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        figures = render_figures(arguments.results, arguments.acceptance)
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
                (arguments.output_directory / name).write_text(
                    content,
                    encoding="utf-8",
                )
    except PlotError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
