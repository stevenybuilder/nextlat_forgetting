#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-config", type=Path, required=True)
    parser.add_argument("--tc1-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--libero-cf-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-seed", type=int, default=7)
    parser.add_argument("--scene-shard-count", type=int, default=1)
    parser.add_argument("--scene-shard-index", type=int, default=0)
    args = parser.parse_args()

    project_src = Path(__file__).parents[1] / "src"
    sys.path.insert(0, str(project_src))
    from intended_futures.config import load_config
    from intended_futures.stimuli import validate_subject_positions

    pilot = load_config(args.pilot_config)
    protocol = json.loads(args.tc1_config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (
        args.scene_shard_count <= 0
        or not 0 <= args.scene_shard_index < args.scene_shard_count
    ):
        raise ValueError("scene shard index must be inside a positive shard count")
    if manifest["study"] != protocol["study"]:
        raise RuntimeError("TC1 manifest study differs from the protocol")
    if manifest["protocol_sha256"] != __import__("hashlib").sha256(
        args.tc1_config.read_bytes()
    ).hexdigest():
        raise RuntimeError("TC1 manifest is not bound to this protocol file")
    renderer = protocol["runtime_contract"]["simulator_renderer"]
    if os.environ.get("MUJOCO_GL") != renderer or os.environ.get("PYOPENGL_PLATFORM") != renderer:
        raise RuntimeError(f"renderer environment must be fixed to {renderer!r}")

    sys.path.insert(0, str(args.libero_cf_root))
    from eval.main_cf import _get_libero_env
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[manifest["suite"]]()
    contract = pilot["benchmark"]["workspace_position_contract"]
    rows_by_scene: dict[str, list[dict]] = {}
    for row in manifest["rows"]:
        rows_by_scene.setdefault(str(row["scene_id"]), []).append(row)
    selected_scene_ids = [
        scene_id
        for index, scene_id in enumerate(sorted(rows_by_scene))
        if index % args.scene_shard_count == args.scene_shard_index
    ]
    rows_by_scene = {scene_id: rows_by_scene[scene_id] for scene_id in selected_scene_ids}

    scene_summaries = []
    minimum_pair_distance = float("inf")
    split_counts: dict[str, int] = {}
    for scene_id, rows in sorted(rows_by_scene.items()):
        task_index = int(rows[0]["task_index"])
        task = suite.get_task(task_index)
        states = suite.get_task_init_states(task_index)
        maximum_index = max(int(row["initial_state_index"]) for row in rows)
        if maximum_index >= len(states):
            raise RuntimeError(
                f"{scene_id} has only {len(states)} official states; requested {maximum_index}"
            )
        env, _ = _get_libero_env(task, 256, args.environment_seed)
        scene_minimum = float("inf")
        try:
            for row in rows:
                state_index = int(row["initial_state_index"])
                env.reset()
                obs = env.set_init_state(states[state_index])
                for _ in range(10):
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                subjects = [
                    str(row["task_a"]["intended_subject"]),
                    str(row["task_b"]["intended_subject"]),
                ]
                positions = validate_subject_positions(obs, subjects, contract)
                separation = float(np.linalg.norm(positions[subjects[0]] - positions[subjects[1]]))
                scene_minimum = min(scene_minimum, separation)
                minimum_pair_distance = min(minimum_pair_distance, separation)
                split = str(row["split"])
                split_counts[split] = split_counts.get(split, 0) + 1
        finally:
            env.close()
        scene_summaries.append(
            {
                "scene_id": scene_id,
                "task_index": task_index,
                "official_state_count": int(len(states)),
                "states_validated": len(rows),
                "minimum_pair_distance_meters": scene_minimum,
            }
        )
        print(
            json.dumps(
                {
                    "event": "scene_preflight_complete",
                    "scene_id": scene_id,
                    "states_validated": len(rows),
                    "minimum_pair_distance_meters": scene_minimum,
                }
            ),
            flush=True,
        )

    expected_per_split = {
        "observer_fit": len(selected_scene_ids)
        * len(protocol["population"]["observer_fit_state_indices"]),
        "observer_validation": len(selected_scene_ids)
        * len(protocol["population"]["observer_validation_state_indices"]),
        "causal_test": len(selected_scene_ids)
        * len(protocol["population"]["causal_test_state_indices"]),
        "reserve": len(selected_scene_ids)
        * len(protocol["population"]["reserve_state_indices"]),
    }
    if split_counts != expected_per_split:
        raise RuntimeError(f"split counts differ: {split_counts} != {expected_per_split}")
    summary = {
        "study": protocol["study"],
        "manifest_sha256": manifest["manifest_sha256"],
        "passed": True,
        "model_outcomes_observed": False,
        "scene_shard_count": args.scene_shard_count,
        "scene_shard_index": args.scene_shard_index,
        "scenes_validated": len(scene_summaries),
        "states_validated": sum(split_counts.values()),
        "split_counts": split_counts,
        "minimum_pair_distance_meters": minimum_pair_distance,
        "scenes": scene_summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("passed", "states_validated", "minimum_pair_distance_meters")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
