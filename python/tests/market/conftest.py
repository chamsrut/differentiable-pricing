"""Pytest fixtures for the market ingestion tests.

The builders live in :mod:`market_fixtures` so test modules can import them
directly; only fixtures that need pytest's lifecycle belong here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from market_fixtures import build_synthetic_archive


@pytest.fixture
def synthetic_archive(tmp_path: Path) -> Path:
    """A tiny archive that verifies cleanly against its own manifest."""
    return build_synthetic_archive(tmp_path / "archive")
