#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


def _scene_means(values: list[float], scenes: list[str]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    group_array = np.asarray(scenes)
    return {
        str(scene): float(np.mean(array[group_array == scene]))
        for scene in np.unique(group_array)
    }


def _receipt_is_exact(condition: dict[str, Any]) -> bool:
    for replan in condition["replans"]:
        receipt = replan["patch_receipt"]
        if (
            int(receipt["calls_seen"]) <= 0
            or int(receipt["calls_patched"]) != int(receipt["calls_seen"])
            or int(receipt.get("shape_mismatches", 0)) != 0
        ):
            return False
    return True


def _action_recovery(
    condition_action: np.ndarray,
    donor_action: np.ndarray,
    recipient_action: np.ndarray,
) -> float:
    donor = np.asarray(donor_action, dtype=np.float64).reshape(-1)
    recipient = np.asarray(recipient_action, dtype=np.float64).reshape(-1)
    condition = np.asarray(condition_action, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(recipient - donor))
    if denominator <= 1e-12:
        raise ValueError("clean donor and recipient action chunks are identical")
    return 1.0 - float(np.linalg.norm(condition - donor)) / denominator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tc1-config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("causal_test", "reserve"), default="causal_test")
    args = parser.parse_args()

    project_src = Path(__file__).parents[1] / "src"
    sys.path.insert(0, str(project_src))
    from intended_futures.statistics import (
        clustered_bootstrap_mean,
        exact_group_sign_flip_pvalue,
    )

    protocol = json.loads(args.tc1_config.read_text(encoding="utf-8"))
    expected = (
        int(protocol["population"]["expected_causal_test_units"])
        if args.split == "causal_test"
        else len(protocol["population"]["scene_ids"])
        * len(protocol["population"]["reserve_state_indices"])
    )
    records = []
    for path in sorted(args.input.glob("scene-*-state-*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("study") == protocol["study"] and record.get("split") == args.split:
            records.append(record)
    if len(records) != expected:
        raise RuntimeError(f"expected {expected} TC1 records, found {len(records)}")
    invalid = [record for record in records if not record.get("valid", False)]
    valid = [record for record in records if record.get("valid", False)]
    required_conditions = {condition["condition_id"] for condition in protocol["conditions"]}
    for record in valid:
        if set(record["conditions"]) != required_conditions:
            raise RuntimeError(f"{record['stimulus_id']} has incomplete TC1 conditions")

    contrast_specs = {
        "minimum_norm_minus_recipient": ("minimum_norm", "recipient_clean"),
        "minimum_norm_minus_random": ("minimum_norm", "matched_random"),
        "full_replay_minus_recipient": ("full_replay", "recipient_clean"),
        "full_replay_minus_random": ("full_replay", "matched_random"),
        "donor_projection_minus_recipient": ("donor_projection", "recipient_clean"),
    }
    scenes = [str(record["scene_id"]) for record in valid]
    contrasts: dict[str, Any] = {}
    analysis_seed = int(protocol["endpoints"]["analysis_seed"])
    for offset, (name, (left_id, right_id)) in enumerate(contrast_specs.items()):
        touch_values = [
            float(record["conditions"][left_id]["first_touch"] == "donor")
            - float(record["conditions"][right_id]["first_touch"] == "donor")
            for record in valid
        ]
        progress_values = [
            float(record["conditions"][left_id]["donor_progress"])
            - float(record["conditions"][right_id]["donor_progress"])
            for record in valid
        ]
        touch_interval = clustered_bootstrap_mean(
            touch_values,
            scenes,
            repetitions=10000,
            seed=analysis_seed + offset,
        )
        progress_interval = clustered_bootstrap_mean(
            progress_values,
            scenes,
            repetitions=10000,
            seed=analysis_seed + 100 + offset,
        )
        touch_scene = _scene_means(touch_values, scenes)
        contrasts[name] = {
            "donor_first_touch_difference": float(np.mean(touch_values)),
            "donor_first_touch_scene_cluster_95ci": [
                touch_interval["lower"],
                touch_interval["upper"],
            ],
            "positive_touch_scenes": sum(value > 0 for value in touch_scene.values()),
            "scene_touch_effects": touch_scene,
            "exact_scene_sign_flip_p": exact_group_sign_flip_pvalue(touch_values, scenes),
            "donor_progress_difference_meters": float(np.mean(progress_values)),
            "donor_progress_scene_cluster_95ci_meters": [
                progress_interval["lower"],
                progress_interval["upper"],
            ],
        }

    condition_summary: dict[str, Any] = {}
    for condition_id in sorted(required_conditions):
        donor_touches = [
            record["conditions"][condition_id]["first_touch"] == "donor"
            for record in valid
        ]
        recipient_touches = [
            record["conditions"][condition_id]["first_touch"] == "recipient"
            for record in valid
        ]
        summary: dict[str, Any] = {
            "donor_first_touch_rate": float(np.mean(donor_touches)),
            "recipient_first_touch_rate": float(np.mean(recipient_touches)),
            "mean_donor_progress_meters": float(
                np.mean(
                    [record["conditions"][condition_id]["donor_progress"] for record in valid]
                )
            ),
        }
        if condition_id not in {"donor_clean", "recipient_clean"}:
            recoveries = [
                _action_recovery(
                    np.asarray(record["conditions"][condition_id]["first_action_chunk"]),
                    np.asarray(record["conditions"]["donor_clean"]["first_action_chunk"]),
                    np.asarray(record["conditions"]["recipient_clean"]["first_action_chunk"]),
                )
                for record in valid
            ]
            summary["mean_first_action_donor_recovery_fraction"] = float(
                np.mean(recoveries)
            )
            summary["positive_action_recovery_scenes"] = sum(
                value > 0 for value in _scene_means(recoveries, scenes).values()
            )
        condition_summary[condition_id] = summary

    full_receipts_exact = all(
        _receipt_is_exact(record["conditions"]["full_replay"]) for record in valid
    )
    controller_receipts_exact = all(
        _receipt_is_exact(record["conditions"][condition_id])
        for record in valid
        for condition_id in ("minimum_norm", "donor_projection", "matched_random")
    )
    manipulation_spec = protocol["decision_rule"]["manipulation_check_required"]
    compact_spec = protocol["decision_rule"]["compact_target_control_supported_only_if_all"]
    manipulation = contrasts["full_replay_minus_recipient"]
    primary = contrasts["minimum_norm_minus_recipient"]
    selective = contrasts["minimum_norm_minus_random"]
    manipulation_checks = {
        "minimum_touch_effect": manipulation["donor_first_touch_difference"]
        >= float(manipulation_spec["full_replay_minus_recipient_touch_minimum"]),
        "confidence_interval_lower_above_zero": manipulation[
            "donor_first_touch_scene_cluster_95ci"
        ][0]
        > 0,
        "positive_scene_effects": manipulation["positive_touch_scenes"]
        >= int(manipulation_spec["positive_scene_effects_minimum"]),
        "exact_patch_receipts": full_receipts_exact,
    }
    manipulation_passed = all(manipulation_checks.values())
    compact_checks = {
        "minimum_norm_minus_recipient_touch": primary["donor_first_touch_difference"]
        >= float(compact_spec["minimum_norm_minus_recipient_touch_minimum"]),
        "minimum_norm_minus_random_touch": selective["donor_first_touch_difference"]
        >= float(compact_spec["minimum_norm_minus_random_touch_minimum"]),
        "recipient_confidence_interval_lower_above_zero": primary[
            "donor_first_touch_scene_cluster_95ci"
        ][0]
        > 0,
        "random_confidence_interval_lower_above_zero": selective[
            "donor_first_touch_scene_cluster_95ci"
        ][0]
        > 0,
        "positive_scene_effects": primary["positive_touch_scenes"]
        >= int(compact_spec["positive_scene_effects_minimum"]),
        "exact_patch_receipts": controller_receipts_exact,
        "invalid_units": len(invalid) <= int(compact_spec["invalid_units_maximum"]),
    }
    compact_supported = manipulation_passed and all(compact_checks.values())
    if not manipulation_passed:
        interpretation = "assay_inconclusive_full_replay_failed"
    elif compact_supported:
        interpretation = "compact_target_control_supported_in_frozen_population"
    else:
        interpretation = "compact_target_control_not_supported_for_frozen_controller"
    advance_to_reserve = args.split == "causal_test" and compact_supported
    final_positive_claim = args.split == "reserve" and compact_supported
    output = {
        "study": protocol["study"],
        "split": args.split,
        "status": "complete",
        "records": len(records),
        "valid_units": len(valid),
        "invalid_units": len(invalid),
        "condition_summary": condition_summary,
        "contrasts": contrasts,
        "receipt_checks": {
            "full_replay_exact": full_receipts_exact,
            "controller_conditions_exact": controller_receipts_exact,
        },
        "manipulation_checks": manipulation_checks,
        "manipulation_passed": manipulation_passed,
        "compact_checks": compact_checks,
        "compact_target_control_supported": compact_supported,
        "advance_to_reserve": advance_to_reserve,
        "final_positive_claim_supported": final_positive_claim,
        "interpretation": interpretation,
        "claim_boundary": protocol["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
