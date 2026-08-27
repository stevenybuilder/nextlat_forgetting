from __future__ import annotations

import ast
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "colab_h3_expansion_launcher.py"


def test_d40_colab_cell_launcher_has_no_cell_unsafe_file_identity() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    ast.parse(source)
    assert "__file__" not in source
    assert 'LAUNCHER_UPLOAD = "/content/h3-expansion-launcher.py"' in source
    assert "sha256_file(LAUNCHER_UPLOAD)" in source
    assert "source_generation" in source and "if_generation_match" in source
    assert "gcloud storage" not in source


def test_d40_launcher_relays_and_invokes_only_frozen_durable_shape() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "subprocess.Popen(" in source
    assert "stderr=subprocess.STDOUT" in source
    assert "[h3-d40-heartbeat]" in source
    assert '"run_h3_expansion_durable.py"' in source
    assert '"--chunk-size", "1000", "--batch-size", "64"' in source
    assert "raise SystemExit(rc)" in source
