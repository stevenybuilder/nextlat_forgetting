#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tc1-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.tc1_config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    shards = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    if len(shards) != len(args.inputs) or not shards:
        raise RuntimeError("preflight merge received no shards")
    expected_shard_count = len(shards)
    if {int(shard["scene_shard_count"]) for shard in shards} != {expected_shard_count}:
        raise RuntimeError("preflight shard-count metadata differs from input count")
    if {int(shard["scene_shard_index"]) for shard in shards} != set(range(expected_shard_count)):
        raise RuntimeError("preflight shard indices are incomplete or duplicated")
    if any(
        shard.get("passed") is not True
        or shard.get("model_outcomes_observed") is not False
        or shard.get("study") != protocol["study"]
        or shard.get("manifest_sha256") != manifest["manifest_sha256"]
        for shard in shards
    ):
        raise RuntimeError("a preflight shard failed or has mismatched provenance")
    scenes = [scene for shard in shards for scene in shard["scenes"]]
    scene_ids = [str(scene["scene_id"]) for scene in scenes]
    expected_scene_ids = sorted(protocol["population"]["scene_ids"])
    if sorted(scene_ids) != expected_scene_ids or len(set(scene_ids)) != len(scene_ids):
        raise RuntimeError("merged preflight scenes are incomplete or duplicated")
    split_counts = {
        split: sum(int(shard["split_counts"].get(split, 0)) for shard in shards)
        for split in ("observer_fit", "observer_validation", "causal_test", "reserve")
    }
    expected_split_counts = {
        "observer_fit": int(protocol["population"]["expected_observer_fit_units"]),
        "observer_validation": int(
            protocol["population"]["expected_observer_validation_units"]
        ),
        "causal_test": int(protocol["population"]["expected_causal_test_units"]),
        "reserve": len(expected_scene_ids)
        * len(protocol["population"]["reserve_state_indices"]),
    }
    if split_counts != expected_split_counts:
        raise RuntimeError(
            f"merged preflight split counts differ: {split_counts} != {expected_split_counts}"
        )
    output = {
        "study": protocol["study"],
        "manifest_sha256": manifest["manifest_sha256"],
        "passed": True,
        "model_outcomes_observed": False,
        "execution": f"{expected_shard_count} simulator-only scene shards",
        "scenes_validated": len(scenes),
        "states_validated": sum(split_counts.values()),
        "split_counts": split_counts,
        "minimum_pair_distance_meters": min(
            float(shard["minimum_pair_distance_meters"]) for shard in shards
        ),
        "scenes": sorted(scenes, key=lambda scene: scene["scene_id"]),
        "shard_files": [str(path) for path in args.inputs],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                key: output[key]
                for key in ("passed", "scenes_validated", "states_validated", "minimum_pair_distance_meters")
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
