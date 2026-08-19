"""Small, self-contained ``american-option-dataset/1`` fixtures.

Every fixture is a handful of rows written to ``tmp_path``. Nothing here reads
``data/american-option-v1/``: the 47 MB local candidate dataset is Git-ignored
and absent in CI, so no test may depend on it.

Rows are constructed so that every identity the schema asserts holds exactly by
construction, which lets a test break exactly one of them and watch the
corresponding gate fire.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from differentiable_pricing.data.american_schema import (
    LABEL_POLICY_NAME,
    SCHEMA_VERSION,
    TABLE_SCHEMA,
)

_DIGEST = "0" * 64
LABEL_STEPS = 32


def _row(
    index: int,
    split: str,
    *,
    option_type: str,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    volatility: float,
    american_at_steps: float,
    american_at_steps_plus_one: float,
    european_at_steps: float,
    european_at_steps_plus_one: float,
) -> dict[str, Any]:
    american = 0.5 * (american_at_steps + american_at_steps_plus_one)
    european = 0.5 * (european_at_steps + european_at_steps_plus_one)
    intrinsic = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
    return {
        "sample_id": f"{split}-{index}",
        "split": split,
        "stratum": "core",
        "exercise_regime": "mixed",
        "option_type": option_type,
        "label_policy": LABEL_POLICY_NAME,
        "spot": spot,
        "strike": strike,
        "log_moneyness": math.log(spot / strike),
        "maturity": maturity,
        "rate": rate,
        "dividend_yield": dividend_yield,
        "volatility": volatility,
        "american_price": american,
        "american_price_at_steps": american_at_steps,
        "american_price_at_steps_plus_one": american_at_steps_plus_one,
        "adjacent_step_gap": abs(american_at_steps - american_at_steps_plus_one),
        "european_crr_price": european,
        "european_crr_price_at_steps": european_at_steps,
        "european_crr_price_at_steps_plus_one": european_at_steps_plus_one,
        "early_exercise_premium": american - european,
        "european_black_scholes_price": european,
        "intrinsic_value": intrinsic,
        "risk_neutral_probability": 0.5,
        "exercise_activity_fraction": 0.25,
        "label_steps": LABEL_STEPS,
        "early_exercise_nodes": 4,
        "earliest_exercise_step": 3,
        "exercise_boundary_layers": 2,
    }


def rows_for_split(split: str, count: int, *, offset: int = 0) -> list[dict[str, Any]]:
    """``count`` distinct, identity-consistent rows for one partition."""
    rows: list[dict[str, Any]] = []
    for index in range(count):
        step = offset + index
        option_type = "call" if step % 2 == 0 else "put"
        spot = 100.0 + step
        strike = 95.0 + 0.5 * step
        intrinsic = (
            max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
        )
        # Prices are built above intrinsic so the fixture satisfies the exercise
        # bound the gate checks, rather than passing it by accident.
        floor = intrinsic + 1.0 + 0.01 * step
        rows.append(
            _row(
                index,
                split,
                option_type=option_type,
                spot=spot,
                strike=strike,
                maturity=0.5 + 0.01 * step,
                rate=0.02 + 0.001 * step,
                dividend_yield=0.01 + 0.0005 * step,
                volatility=0.2 + 0.001 * step,
                american_at_steps=floor + 0.002,
                american_at_steps_plus_one=floor,
                european_at_steps=floor - 0.5,
                european_at_steps_plus_one=floor - 0.5,
            )
        )
    return rows


def table_from_rows(rows: list[dict[str, Any]]) -> pa.Table:
    columns = {name: [row[name] for row in rows] for name in TABLE_SCHEMA.names}
    return pa.table(columns, schema=TABLE_SCHEMA)


def _domain(all_rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    domain: dict[str, list[float]] = {}
    for field in (
        "spot",
        "log_moneyness",
        "maturity",
        "rate",
        "dividend_yield",
        "volatility",
    ):
        values = [row[field] for row in all_rows]
        low = min(values) - 1.0
        high = max(values) + 1.0
        domain[field] = [low, high]
    return domain


def _label_diagnostics(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "rows": len(rows),
        "zero_price_rows": sum(1 for row in rows if row["american_price"] == 0.0),
        "near_zero_price_rows": 0,
        "zero_premium_rows": sum(
            1 for row in rows if row["early_exercise_premium"] == 0.0
        ),
        "no_early_exercise_rows": sum(
            1 for row in rows if row["early_exercise_nodes"] == 0
        ),
        "at_intrinsic_rows": 0,
        "large_adjacent_gap_rows": 0,
    }


def write_dataset(
    directory: Path,
    *,
    counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Write a complete, valid three-partition fixture dataset.

    Returns the manifest that was written, so a test can mutate a copy of it
    and re-write it to exercise one specific rejection.
    """
    counts = counts or {"train": 6, "validation": 3, "interpolation_test": 3}
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    offset = 0
    tables: dict[str, pa.Table] = {}
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "validation", "interpolation_test"):
        rows = rows_for_split(split, counts[split], offset=offset)
        offset += counts[split]
        split_rows[split] = rows
        tables[split] = table_from_rows(rows)

    files = []
    for split, table in tables.items():
        path = directory / f"{split}.parquet"
        pq.write_table(table, path, compression="snappy", version="2.6")
        files.append(
            {
                "split": split,
                "file": path.name,
                "rows": table.num_rows,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    every_row = [row for rows in split_rows.values() for row in rows]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator_version": "1.0.0",
        "dataset_name": "american-option-fixture",
        "columns": list(TABLE_SCHEMA.names),
        "config": {"file": "american_option_dataset_v1.toml", "sha256": _DIGEST},
        "oracle": {
            "name": "dp::crr_binomial",
            "crr_header_sha256": _DIGEST,
            "crr_implementation_sha256": _DIGEST,
            "crr_source_sha256": _DIGEST,
        },
        "label_policy": {"name": LABEL_POLICY_NAME, "steps": LABEL_STEPS},
        "row_counts": {split: counts[split] for split in counts},
        "files": files,
        "domain": _domain(every_row),
        "label_diagnostics": {
            "splits": {
                split: _label_diagnostics(rows) for split, rows in split_rows.items()
            }
        },
    }
    write_manifest(directory, manifest)
    return manifest


def write_manifest(directory: Path, manifest: dict[str, Any]) -> Path:
    path = Path(directory) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    return path


def rewrite_split(directory: Path, split: str, table: pa.Table) -> None:
    """Overwrite one partition's Parquet file without touching the manifest."""
    pq.write_table(
        table, Path(directory) / f"{split}.parquet", compression="snappy", version="2.6"
    )
