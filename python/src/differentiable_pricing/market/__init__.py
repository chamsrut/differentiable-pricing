"""Deterministic, memory-bounded ingestion and feasibility audit of raw market data.

This package answers scoped questions about an acquired raw archive: is it
intact, can it be ingested, and what does it support. It decodes, normalizes and
measures. It does not price, fit, calibrate or train, and it never invents an
input that was not acquired.

Put-call parity fitting, implied forwards and dividend inference are explicitly
out of scope here and belong to task 9B.
"""

from __future__ import annotations

__all__ = [
    "SCHEMA_VERSION",
]

SCHEMA_VERSION = "market-feasibility/2"
"""Only archive-audit schema version this package accepts."""
