"""Regression tests for the fail-closed legacy Colab profiler."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]
LEGACY_PROFILER = PROJECT / "scripts" / "colab_profile.py"


def test_import_is_side_effect_free() -> None:
    spec = importlib.util.spec_from_file_location(
        "retired_colab_profile_under_test", LEGACY_PROFILER
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)

    spec.loader.exec_module(module)

    assert callable(module.main)


def test_execution_fails_closed_and_points_to_supported_paths() -> None:
    result = subprocess.run(
        [sys.executable, str(LEGACY_PROFILER)],
        cwd=PROJECT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "retired and cannot launch profiling jobs" in result.stderr
    assert "scripts/profile.sh" in result.stderr
    assert "scripts/colab_train_loop.py" in result.stderr


def test_source_has_no_process_or_provisioning_surface() -> None:
    source = LEGACY_PROFILER.read_text(encoding="utf-8")

    assert "subprocess" not in source
    assert "os.system" not in source
    assert "colab start" not in source
    assert "colab exec" not in source
    assert "gcloud storage" not in source
