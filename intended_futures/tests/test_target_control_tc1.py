import json
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).parents[1]


def _patched_condition(first_touch: str, action_value: float) -> dict:
    return {
        "first_touch": first_touch,
        "donor_progress": 0.3 if first_touch == "donor" else 0.0,
        "first_action_chunk": [[action_value] * 7 for _ in range(10)],
        "replans": [
            {
                "patch_receipt": {
                    "calls_seen": 1,
                    "calls_patched": 1,
                    "shape_mismatches": 0,
                }
            }
        ],
    }


def _clean_condition(first_touch: str, action_value: float) -> dict:
    return {
        "first_touch": first_touch,
        "donor_progress": 0.3 if first_touch == "donor" else 0.0,
        "first_action_chunk": [[action_value] * 7 for _ in range(10)],
        "replans": [{"patch_receipt": {"calls_seen": 0, "calls_patched": 0}}],
    }


def test_tc1_analysis_requires_selective_controller_and_working_replay(tmp_path):
    protocol = json.loads(
        (ROOT / "config" / "target_control_tc1.json").read_text(
            encoding="utf-8"
        )
    )
    protocol["population"]["expected_causal_test_units"] = 12
    config_path = tmp_path / "protocol.json"
    config_path.write_text(json.dumps(protocol), encoding="utf-8")
    records = tmp_path / "records"
    records.mkdir()
    for index, scene in enumerate(protocol["population"]["scene_ids"]):
        payload = {
            "study": protocol["study"],
            "split": "causal_test",
            "valid": True,
            "stimulus_id": f"{scene}-state-30",
            "scene_id": scene,
            "conditions": {
                "donor_clean": _clean_condition("donor", 1.0),
                "recipient_clean": _clean_condition("recipient", 0.0),
                "minimum_norm": _patched_condition("donor", 0.9),
                "donor_projection": _patched_condition("donor", 0.8),
                "matched_random": _patched_condition("recipient", 0.1),
                "full_replay": _patched_condition("donor", 1.0),
            },
        }
        (records / f"{scene}-state-{30 + index:02d}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )
    output = tmp_path / "analysis.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "analyze_target_control_tc1.py"),
            "--tc1-config",
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
    analysis = json.loads(output.read_text(encoding="utf-8"))
    assert analysis["manipulation_passed"] is True
    assert analysis["compact_target_control_supported"] is True
    assert analysis["advance_to_reserve"] is True
    assert analysis["final_positive_claim_supported"] is False


def test_failed_observer_gate_cannot_create_controller_or_clearance(tmp_path):
    protocol = json.loads(
        (ROOT / "config" / "target_control_tc1.json").read_text(encoding="utf-8")
    )
    protocol["population"]["expected_observer_fit_units"] = 2
    protocol["population"]["expected_observer_validation_units"] = 2
    protocol["observer"]["validation_gate"]["minimum_positive_scene_cosines"] = 2
    config_path = tmp_path / "protocol.json"
    config_path.write_text(json.dumps(protocol), encoding="utf-8")
    protocol_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest = {
        "study": protocol["study"],
        "protocol_sha256": protocol_sha,
        "manifest_sha256": "synthetic-manifest",
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    runtime = {
        "study": protocol["study"],
        "tc1_config_sha256": protocol_sha,
        "manifest_sha256": manifest["manifest_sha256"],
    }
    runtime_path = tmp_path / "runtime.json"
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")
    runtime_sha = hashlib.sha256(runtime_path.read_bytes()).hexdigest()
    records = tmp_path / "records"
    records.mkdir()
    examples = [
        ("observer_fit", "scene-a", [1.0, 0.0, 0.0], [1.0, 0.0]),
        ("observer_fit", "scene-b", [0.0, 1.0, 0.0], [0.0, 1.0]),
        ("observer_validation", "scene-a", [1.0, 0.0, 0.0], [0.8, 0.2]),
        ("observer_validation", "scene-b", [0.0, 1.0, 0.0], [0.2, 0.8]),
    ]
    for index, (split, scene, target, activation) in enumerate(examples):
        metadata = {
            "study": protocol["study"],
            "split": split,
            "scene_id": scene,
            "subject_a": "object-a",
            "subject_b": "object-b",
            "prompt_a_id": f"prompt-a-{scene}",
            "prompt_b_id": f"prompt-b-{scene}",
            "runtime_receipt_sha256": runtime_sha,
        }
        np.savez_compressed(
            records / f"scene-{index:02d}-state-10.npz",
            metadata_json=np.asarray(json.dumps(metadata)),
            activation_difference=np.asarray(activation, dtype=np.float16),
            target_difference=np.asarray(target, dtype=np.float64),
        )
    analysis_path = tmp_path / "analysis.json"
    controller_path = tmp_path / "controller.npz"
    clearance_path = tmp_path / "clearance.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "fit_target_controller.py"),
            "--tc1-config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--runtime-receipt",
            str(runtime_path),
            "--input",
            str(records),
            "--analysis-output",
            str(analysis_path),
            "--controller-output",
            str(controller_path),
            "--clearance-output",
            str(clearance_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 2, completed.stderr
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    assert analysis["advance_to_causal_test"] is False
    assert analysis["checks"]["residual_sse_reduction_over_exact_prompt_pair"] is False
    assert not controller_path.exists()
    assert not clearance_path.exists()
