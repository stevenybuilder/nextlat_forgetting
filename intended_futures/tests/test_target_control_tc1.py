import json
from pathlib import Path
import subprocess
import sys


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
