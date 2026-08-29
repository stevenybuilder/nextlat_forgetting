#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left_flat = np.asarray(left, dtype=np.float64).reshape(-1)
    right_flat = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat))
    if denominator <= 1e-12:
        return float("nan")
    return float(np.dot(left_flat, right_flat) / denominator)


def _scene_means(values: list[float], scenes: list[str]) -> dict[str, float]:
    value_array = np.asarray(values, dtype=np.float64)
    scene_array = np.asarray(scenes)
    return {
        str(scene): float(np.mean(value_array[scene_array == scene]))
        for scene in np.unique(scene_array)
    }


def _standardized(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    standard_deviation = float(np.std(array, ddof=1))
    if standard_deviation <= 1e-12:
        return float("inf") if float(np.mean(array)) > 0 else 0.0
    return float(np.mean(array) / standard_deviation)


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
    parser.add_argument("--m0-config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project_src = Path(__file__).parents[1] / "src"
    sys.path.insert(0, str(project_src))
    from intended_futures.statistics import clustered_bootstrap_mean

    protocol = json.loads(args.m0_config.read_text(encoding="utf-8"))
    expected = int(protocol["population"]["expected_units"])
    records = []
    for path in sorted(args.input.glob("scene-*-state-*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("study") == protocol["study"]:
            records.append(record)
    invalid_records = [record for record in records if not record.get("valid", False)]
    valid_records = [record for record in records if record.get("valid", False)]
    if len(records) != expected:
        raise RuntimeError(f"expected {expected} M0 records, found {len(records)}")

    gates = protocol["gate"]
    sites_summary: dict[str, Any] = {}
    for site in protocol["sites"]:
        site_id = str(site["site_id"])
        full_id = f"{site_id}_full"
        random_id = f"{site_id}_random"
        action_margins: list[float] = []
        random_margins: list[float] = []
        margin_advantages: list[float] = []
        progress_effects: list[float] = []
        scenes: list[str] = []
        donor_first_full = 0
        donor_first_random = 0
        donor_first_recipient = 0
        exact_receipts = True

        for record in valid_records:
            conditions = record["conditions"]
            required = {"donor_clean", "recipient_clean", full_id, random_id}
            if not required.issubset(conditions):
                raise RuntimeError(f"record {record['stimulus_id']} lacks M0 conditions for {site_id}")
            donor_action = np.asarray(conditions["donor_clean"]["first_action_chunk"])
            recipient_action = np.asarray(conditions["recipient_clean"]["first_action_chunk"])
            full_action = np.asarray(conditions[full_id]["first_action_chunk"])
            random_action = np.asarray(conditions[random_id]["first_action_chunk"])
            full_margin = _cosine(full_action, donor_action) - _cosine(
                full_action, recipient_action
            )
            random_margin = _cosine(random_action, donor_action) - _cosine(
                random_action, recipient_action
            )
            action_margins.append(full_margin)
            random_margins.append(random_margin)
            margin_advantages.append(full_margin - random_margin)
            progress_effects.append(
                float(conditions[full_id]["donor_progress"])
                - float(conditions["recipient_clean"]["donor_progress"])
            )
            scenes.append(str(record["scene_id"]))
            donor_first_full += int(conditions[full_id]["first_touch"] == "donor")
            donor_first_random += int(conditions[random_id]["first_touch"] == "donor")
            donor_first_recipient += int(
                conditions["recipient_clean"]["first_touch"] == "donor"
            )
            exact_receipts = exact_receipts and _receipt_is_exact(conditions[full_id])
            exact_receipts = exact_receipts and _receipt_is_exact(conditions[random_id])

        action_scene = _scene_means(action_margins, scenes)
        advantage_scene = _scene_means(margin_advantages, scenes)
        progress_scene = _scene_means(progress_effects, scenes)
        progress_scene_values = list(progress_scene.values())
        action_interval = clustered_bootstrap_mean(
            action_margins,
            scenes,
            repetitions=10000,
            seed=81810 + int(site["layer"]),
        )
        progress_interval = clustered_bootstrap_mean(
            progress_effects,
            scenes,
            repetitions=10000,
            seed=81910 + int(site["layer"]),
        )
        checks = {
            "exact_all_call_receipts": exact_receipts,
            "positive_action_margin_scenes": sum(value > 0 for value in action_scene.values())
            >= int(gates["minimum_positive_scene_margins"]),
            "mean_action_margin": float(np.mean(action_margins))
            >= float(gates["minimum_donor_action_similarity_margin"]),
            "margin_over_random": float(np.mean(margin_advantages))
            >= float(gates["minimum_margin_over_random"]),
            "standardized_donor_progress": _standardized(progress_scene_values)
            >= float(gates["minimum_standardized_donor_progress"]),
            "positive_donor_progress_scenes": sum(value > 0 for value in progress_scene_values)
            >= int(gates["minimum_positive_donor_progress_scenes"]),
            "invalid_units": len(invalid_records) <= int(gates["maximum_invalid_units"]),
        }
        sites_summary[site_id] = {
            "site": site,
            "valid_units": len(valid_records),
            "mean_donor_action_similarity_margin": float(np.mean(action_margins)),
            "action_margin_cluster_bootstrap_95ci": [
                action_interval["lower"],
                action_interval["upper"],
            ],
            "positive_action_margin_scenes": sum(value > 0 for value in action_scene.values()),
            "mean_margin_advantage_over_random": float(np.mean(margin_advantages)),
            "positive_margin_advantage_scenes": sum(
                value > 0 for value in advantage_scene.values()
            ),
            "mean_donor_progress_effect_meters": float(np.mean(progress_effects)),
            "donor_progress_cluster_bootstrap_95ci_meters": [
                progress_interval["lower"],
                progress_interval["upper"],
            ],
            "standardized_scene_donor_progress": _standardized(progress_scene_values),
            "positive_donor_progress_scenes": sum(value > 0 for value in progress_scene_values),
            "donor_first_touch_rates": {
                "recipient_clean": donor_first_recipient / len(valid_records) if valid_records else None,
                "full_replay": donor_first_full / len(valid_records) if valid_records else None,
                "random_direction": donor_first_random / len(valid_records) if valid_records else None,
            },
            "scene_action_margins": action_scene,
            "scene_progress_effects_meters": progress_scene,
            "checks": checks,
            "gate_passed": all(checks.values()),
        }

    passed_sites = [
        site_id for site_id, summary in sites_summary.items() if summary["gate_passed"]
    ]
    output = {
        "study": protocol["study"],
        "status": "complete" if len(records) == expected else "incomplete",
        "records": len(records),
        "valid_units": len(valid_records),
        "invalid_units": len(invalid_records),
        "sites": sites_summary,
        "passed_sites": passed_sites,
        "advance_to_tc1": bool(passed_sites),
        "claim_boundary": "engineering manipulation check; not a target-control effect",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

