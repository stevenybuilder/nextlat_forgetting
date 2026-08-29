#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


def _family_means(values: list[float], families: list[str]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    groups = np.asarray(families)
    return {
        str(family): float(np.mean(array[groups == family]))
        for family in sorted(set(groups))
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from intended_futures.statistics import (
        clustered_bootstrap_mean,
        exact_group_sign_flip_pvalue,
    )

    config = json.loads(args.config.read_text(encoding="utf-8"))
    records = []
    for path in sorted(args.input.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("study") == config["study"] and "conditions" in record:
            records.append(record)
    expected = int(config["population"]["expected_split_units"]["causal_test"])
    if len(records) != expected:
        raise RuntimeError(f"expected {expected} causal records, found {len(records)}")
    invalid = [record for record in records if not record.get("valid", False)]
    valid = [record for record in records if record.get("valid", False)]
    required = {condition["condition_id"] for condition in config["conditions"]}
    for record in valid:
        if set(record["conditions"]) != required:
            raise RuntimeError(f"{record['stimulus_id']} has incomplete conditions")
    families = [str(record["family_id"]) for record in valid]
    seed = int(config["causal_decision_rule"]["analysis_seed"])
    specs = {
        "minimum_norm_minus_task_b": ("minimum_norm", "task_b_clean"),
        "minimum_norm_minus_random": ("minimum_norm", "matched_random"),
        "minimum_norm_minus_prompt_mean": ("minimum_norm", "prompt_mean_controller"),
        "full_replay_minus_task_b": ("full_replay", "task_b_clean"),
    }
    contrasts: dict[str, Any] = {}
    for offset, (name, (left, right)) in enumerate(specs.items()):
        touch = [
            float(record["conditions"][left]["first_touch"] == "a")
            - float(record["conditions"][right]["first_touch"] == "a")
            for record in valid
        ]
        progress = [
            float(record["conditions"][left]["target_a_progress"])
            - float(record["conditions"][right]["target_a_progress"])
            for record in valid
        ]
        touch_interval = clustered_bootstrap_mean(
            touch, families, repetitions=10000, seed=seed + offset
        )
        progress_interval = clustered_bootstrap_mean(
            progress, families, repetitions=10000, seed=seed + 100 + offset
        )
        touch_family = _family_means(touch, families)
        progress_family = _family_means(progress, families)
        contrasts[name] = {
            "target_a_first_touch_difference": float(np.mean(touch)),
            "target_a_first_touch_family_cluster_95ci": [
                touch_interval["lower"],
                touch_interval["upper"],
            ],
            "positive_touch_families": sum(value > 0 for value in touch_family.values()),
            "family_touch_effects": touch_family,
            "exact_family_sign_flip_p": exact_group_sign_flip_pvalue(touch, families),
            "target_a_progress_difference_meters": float(np.mean(progress)),
            "target_a_progress_family_cluster_95ci_meters": [
                progress_interval["lower"],
                progress_interval["upper"],
            ],
            "positive_progress_families": sum(
                value > 0 for value in progress_family.values()
            ),
            "family_progress_effects": progress_family,
        }

    full_receipts_exact = all(
        _receipt_is_exact(record["conditions"]["full_replay"]) for record in valid
    )
    controller_receipts_exact = all(
        _receipt_is_exact(record["conditions"][condition])
        for record in valid
        for condition in ("minimum_norm", "prompt_mean_controller", "matched_random")
    )
    rules = config["causal_decision_rule"]
    manipulation_spec = rules["manipulation_check_required"]
    compact_spec = rules["compact_control_supported_only_if_all"]
    manipulation = contrasts["full_replay_minus_task_b"]
    primary = contrasts["minimum_norm_minus_task_b"]
    random_control = contrasts["minimum_norm_minus_random"]
    layout_control = contrasts["minimum_norm_minus_prompt_mean"]
    manipulation_checks = {
        "minimum_touch_effect": manipulation["target_a_first_touch_difference"]
        >= float(manipulation_spec["full_replay_minus_task_b_target_a_touch_minimum"]),
        "confidence_interval_lower_above_zero": manipulation[
            "target_a_first_touch_family_cluster_95ci"
        ][0]
        > 0,
        "positive_family_effects": manipulation["positive_touch_families"]
        >= int(manipulation_spec["positive_family_effects_minimum"]),
        "exact_patch_receipts": full_receipts_exact,
    }
    manipulation_passed = all(manipulation_checks.values())
    compact_checks = {
        "minimum_norm_minus_task_b_touch": primary["target_a_first_touch_difference"]
        >= float(compact_spec["minimum_norm_minus_task_b_target_a_touch_minimum"]),
        "minimum_norm_minus_random_touch": random_control[
            "target_a_first_touch_difference"
        ]
        >= float(compact_spec["minimum_norm_minus_random_target_a_touch_minimum"]),
        "touch_confidence_lowers_above_zero": primary[
            "target_a_first_touch_family_cluster_95ci"
        ][0]
        > 0
        and random_control["target_a_first_touch_family_cluster_95ci"][0] > 0,
        "layout_progress_over_prompt_mean": layout_control[
            "target_a_progress_difference_meters"
        ]
        >= float(
            compact_spec["minimum_actual_minus_prompt_mean_target_a_progress_meters"]
        ),
        "layout_progress_confidence_lower_above_zero": layout_control[
            "target_a_progress_family_cluster_95ci_meters"
        ][0]
        > 0,
        "positive_family_effects": primary["positive_touch_families"]
        >= int(compact_spec["positive_family_effects_minimum"]),
        "exact_patch_receipts": controller_receipts_exact,
        "invalid_units": len(invalid) <= int(compact_spec["invalid_units_maximum"]),
    }
    compact_supported = manipulation_passed and all(compact_checks.values())
    if not manipulation_passed:
        interpretation = "assay_inconclusive_full_replay_failed"
    elif compact_supported:
        interpretation = "layout_specific_compact_target_control_supported_in_pilot"
    else:
        interpretation = "layout_specific_compact_target_control_not_supported"
    output = {
        "study": config["study"],
        "status": "complete",
        "records": len(records),
        "valid_units": len(valid),
        "invalid_units": len(invalid),
        "families": len(set(families)),
        "contrasts": contrasts,
        "receipt_checks": {
            "full_replay_exact": full_receipts_exact,
            "controller_conditions_exact": controller_receipts_exact,
        },
        "manipulation_checks": manipulation_checks,
        "manipulation_passed": manipulation_passed,
        "compact_checks": compact_checks,
        "compact_target_control_supported": compact_supported,
        "interpretation": interpretation,
        "claim_boundary": config["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
