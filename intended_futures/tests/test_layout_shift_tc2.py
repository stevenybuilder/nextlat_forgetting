from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).parents[1]


def _condition(first_touch: str | None, progress: float, *, patched: bool) -> dict:
    return {
        "first_touch": first_touch,
        "target_a_progress": progress,
        "replans": [
            {
                "patch_receipt": {
                    "calls_seen": 1 if patched else 0,
                    "calls_patched": 1 if patched else 0,
                    "shape_mismatches": 0,
                }
            }
        ],
    }


def test_residual_observer_can_pass_without_prompt_mean_leakage(tmp_path):
    config = json.loads(
        (ROOT / "config" / "layout_shift_tc2.json").read_text(encoding="utf-8")
    )
    config["status"] = "frozen"
    config["population"]["expected_split_units"] = {
        "observer_fit": 20,
        "observer_validation": 10,
        "causal_test": 0,
    }
    config["observer"]["validation_gate"]["minimum_positive_family_cosines"] = 2
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest = {
        "study": config["study"],
        "config_sha256": config_sha,
        "manifest_sha256": "synthetic-manifest",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    receipt = {
        "study": config["study"],
        "config_sha256": config_sha,
        "manifest_sha256": manifest["manifest_sha256"],
    }
    receipt_path = tmp_path / "runtime.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    receipt_sha = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    records = tmp_path / "records"
    records.mkdir()
    index = 0
    for family_index, family in enumerate(("family-a", "family-b")):
        family_offset = np.asarray([2.0 * family_index, -1.0 * family_index])
        for sample in (1, 2, 3):
            split = "observer_fit" if sample in (1, 2) else "observer_validation"
            for level in range(1, 6):
                residual = np.asarray([0.2 * level, (-1.0) ** level * 0.1])
                target_xy = family_offset + residual
                metadata = {
                    "study": config["study"],
                    "split": split,
                    "family_id": family,
                    "level": level,
                    "sample": sample,
                    "runtime_receipt_sha256": receipt_sha,
                }
                np.savez_compressed(
                    records / f"record-{index:03d}.npz",
                    metadata_json=np.asarray(json.dumps(metadata)),
                    activation_difference=target_xy.astype(np.float16),
                    action_difference=target_xy.astype(np.float32),
                    target_difference_xy=target_xy,
                    target_difference_xyz=np.concatenate((target_xy, [0.0])),
                )
                index += 1
    analysis = tmp_path / "analysis.json"
    controller = tmp_path / "controller.npz"
    clearance = tmp_path / "clearance.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "fit_layout_shift_controller.py"),
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--runtime-receipt",
            str(receipt_path),
            "--input",
            str(records),
            "--analysis-output",
            str(analysis),
            "--controller-output",
            str(controller),
            "--clearance-output",
            str(clearance),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(analysis.read_text(encoding="utf-8"))
    assert result["advance_to_causal_test"] is True
    assert result["validation"]["layout_residual_r2"] > 0.9
    assert controller.exists()
    assert clearance.exists()


def test_causal_analysis_requires_layout_advantage_over_prompt_mean(tmp_path):
    config = json.loads(
        (ROOT / "config" / "layout_shift_tc2.json").read_text(encoding="utf-8")
    )
    config["population"]["expected_split_units"]["causal_test"] = 9
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    records = tmp_path / "records"
    records.mkdir()
    for index in range(9):
        payload = {
            "study": config["study"],
            "stimulus_id": f"family-{index}",
            "family_id": f"family-{index}",
            "valid": True,
            "conditions": {
                "task_a_clean": _condition("a", 0.2, patched=False),
                "task_b_clean": _condition("b", 0.0, patched=False),
                "minimum_norm": _condition("a", 0.2, patched=True),
                "prompt_mean_controller": _condition("b", 0.0, patched=True),
                "matched_random": _condition("b", 0.0, patched=True),
                "full_replay": _condition("a", 0.2, patched=True),
            },
        }
        (records / f"family-{index}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    output = tmp_path / "analysis.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_layout_shift_causal.py"),
            "--config",
            str(config_path),
            "--input",
            str(records),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["manipulation_passed"] is True
    assert result["compact_target_control_supported"] is True
    assert result["compact_checks"]["layout_progress_over_prompt_mean"] is True
