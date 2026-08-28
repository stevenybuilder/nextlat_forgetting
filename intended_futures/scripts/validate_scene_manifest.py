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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--libero-cf-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-seed", type=int, default=7)
    args = parser.parse_args()

    project_src = Path(__file__).parents[1] / "src"
    sys.path.insert(0, str(project_src))
    from intended_futures.config import load_config
    from intended_futures.stimuli import validate_subject_positions

    config = load_config(args.config)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["study"] != config["study"] or manifest["manifest_sha256"] != config["manifest_sha256"]:
        raise RuntimeError("manifest does not match the frozen configuration")
    renderer = config["runtime_contract"]["simulator_renderer"]
    if os.environ.get("MUJOCO_GL") != renderer or os.environ.get("PYOPENGL_PLATFORM") != renderer:
        raise RuntimeError(f"renderer environment must be fixed to {renderer!r}")

    sys.path.insert(0, str(args.libero_cf_root))
    from eval.main_cf import _get_libero_env
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[manifest["suite"]]()
    contract = config["benchmark"]["workspace_position_contract"]
    subjects = sorted({prompt["subject"] for prompt in config["benchmark"]["prompts"].values()})
    scene_summaries = []
    for scene in config["benchmark"]["scenes"]:
        task_index = int(scene["task_index"])
        task = suite.get_task(task_index)
        states = suite.get_task_init_states(task_index)
        env, _ = _get_libero_env(task, 256, args.environment_seed)
        positions_by_subject = {subject: [] for subject in subjects}
        try:
            for state_index in config["sampling"]["initial_state_indices"]:
                env.reset()
                obs = env.set_init_state(states[state_index])
                for _ in range(10):
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                positions = validate_subject_positions(obs, subjects, contract)
                for subject, position in positions.items():
                    positions_by_subject[subject].append(position)
        finally:
            env.close()
        scene_summaries.append(
            {
                "scene_id": scene["scene_id"],
                "task_index": task_index,
                "task_file": task.bddl_file,
                "states_validated": len(config["sampling"]["initial_state_indices"]),
                "position_ranges": {
                    subject: {
                        "minimum": np.min(np.asarray(values), axis=0).tolist(),
                        "maximum": np.max(np.asarray(values), axis=0).tolist(),
                    }
                    for subject, values in positions_by_subject.items()
                },
            }
        )
    summary = {
        "study": config["study"],
        "manifest_sha256": manifest["manifest_sha256"],
        "passed": True,
        "scenes_validated": len(scene_summaries),
        "states_validated": sum(scene["states_validated"] for scene in scene_summaries),
        "subjects_required_in_every_scene": subjects,
        "scenes": scene_summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("passed", "scenes_validated", "states_validated")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
