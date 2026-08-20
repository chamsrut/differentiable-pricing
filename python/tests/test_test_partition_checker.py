"""Lightweight static checks for indirect PyTorch test dependencies."""

from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _checker():
    path = PROJECT_ROOT / "scripts/check_test_partition.py"
    spec = importlib.util.spec_from_file_location("test_partition_checker", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dynamic_repository_script_import_is_classified_as_torch_backed(tmp_path) -> None:
    script = tmp_path / "scripts" / "torch_runner.py"
    script.parent.mkdir(parents=True)
    script.write_text("import torch\n", encoding="utf-8")
    test = tmp_path / "python" / "tests" / "test_runner.py"
    test.parent.mkdir(parents=True)
    test.write_text(
        """
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def _module():
    path = PROJECT_ROOT / "scripts/torch_runner.py"
    spec = importlib.util.spec_from_file_location("fixture_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
""",
        encoding="utf-8",
    )

    assert _checker()._requires_torch(test, tmp_path) == {"scripts/torch_runner.py -> torch"}


def test_unrelated_local_shadow_does_not_hide_dynamic_torch_script(tmp_path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "torch_runner.py").write_text("import torch\n", encoding="utf-8")
    (scripts / "safe.py").write_text("import json\n", encoding="utf-8")
    test = tmp_path / "python" / "tests" / "test_runner.py"
    test.parent.mkdir(parents=True)
    test.write_text(
        """
import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PROJECT_ROOT / "scripts/torch_runner.py"

def _module():
    spec = importlib.util.spec_from_file_location("fixture_runner", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def unrelated():
    SCRIPT = PROJECT_ROOT / "scripts/safe.py"
    return SCRIPT
""",
        encoding="utf-8",
    )

    assert _checker()._requires_torch(test, tmp_path) == {"scripts/torch_runner.py -> torch"}
