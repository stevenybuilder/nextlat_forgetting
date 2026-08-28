#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def _atomic_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _policy_observation(obs: dict[str, Any], prompt: str, image_tools: Any, quat2axisangle: Any) -> dict[str, Any]:
    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return {
        "observation/image": image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224)),
        "observation/wrist_image": image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224)),
        "observation/state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
        "prompt": prompt,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--libero-cf-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--environment-seed", type=int, default=7)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["study"] != config["study"] or manifest["manifest_sha256"] != config["manifest_sha256"]:
        raise RuntimeError("manifest does not match the frozen configuration")
    renderer = str(config["runtime_contract"]["simulator_renderer"])
    if os.environ.get("MUJOCO_GL") != renderer or os.environ.get("PYOPENGL_PLATFORM") != renderer:
        raise RuntimeError(f"renderer environment must be fixed to {renderer!r}")

    sys.path.insert(0, str(args.libero_cf_root))
    from eval.main_cf import _get_libero_env, _quat2axisangle
    from intended_futures.stimuli import validate_subject_positions
    from libero.libero import benchmark
    from openpi_client import image_tools
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    rows = manifest["rows"][: args.limit]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["scene_id"]].append(row)
    suite = benchmark.get_benchmark_dict()[manifest["suite"]]()
    client = WebsocketClientPolicy(args.host, args.port)
    contract = config["benchmark"]["workspace_position_contract"]
    completed = 0
    started = time.monotonic()
    for scene_id, scene_rows in grouped.items():
        task_index = int(scene_rows[0]["task_index"])
        task = suite.get_task(task_index)
        if task.bddl_file != Path(scene_rows[0]["task_file"]).name:
            raise RuntimeError(f"benchmark task mapping changed for {scene_id}")
        states = suite.get_task_init_states(task_index)
        env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.environment_seed)
        try:
            for row in scene_rows:
                destination = args.output / f"{row['stimulus_id']}.npz"
                if destination.exists():
                    with np.load(destination, allow_pickle=False) as existing:
                        metadata = json.loads(str(existing["metadata_json"].item()))
                    if metadata["stimulus_id"] != row["stimulus_id"] or metadata["noise_seed"] != row["noise_seed"]:
                        raise RuntimeError(f"preexisting record identity mismatch: {destination}")
                    completed += 1
                    continue
                env.reset()
                obs = env.set_init_state(states[int(row["initial_state_index"])])
                for _ in range(10):
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                subjects = [row["task_a"]["intended_subject"], row["task_b"]["intended_subject"]]
                positions = validate_subject_positions(obs, subjects, contract)
                target_a = positions[subjects[0]]
                target_b = positions[subjects[1]]
                separation = float(np.linalg.norm(target_b - target_a))
                request = _policy_observation(obs, row["task_a"]["prompt"], image_tools, _quat2axisangle)
                request.update(
                    {
                        "__intended_futures_mode": "extract_pair",
                        "__prompt_a": row["task_a"]["prompt"],
                        "__prompt_b": row["task_b"]["prompt"],
                        "__noise_seed": int(row["noise_seed"]),
                    }
                )
                response = client.infer(request)
                metadata = {
                    "study": config["study"],
                    "manifest_sha256": manifest["manifest_sha256"],
                    "stimulus_id": row["stimulus_id"],
                    "scene_id": scene_id,
                    "task_index": task_index,
                    "initial_state_index": int(row["initial_state_index"]),
                    "noise_seed": int(row["noise_seed"]),
                    "prompt_a": row["task_a"]["prompt"],
                    "prompt_b": row["task_b"]["prompt"],
                    "subject_a": row["task_a"]["intended_subject"],
                    "subject_b": row["task_b"]["intended_subject"],
                    "subject_separation": separation,
                    "denoising_calls_a": response["denoising_calls_a"],
                    "denoising_calls_b": response["denoising_calls_b"],
                    "pair_infer_ms": float(response["pair_infer_ms"]),
                }
                arrays: dict[str, Any] = {
                    "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
                    "eef_position": np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
                    "target_a_position": target_a,
                    "target_b_position": target_b,
                    "actions_a": np.asarray(response["actions_a"], dtype=np.float32),
                    "actions_b": np.asarray(response["actions_b"], dtype=np.float32),
                }
                for layer, activation in response["activations_a"].items():
                    arrays[f"activation_a_layer_{layer}"] = np.asarray(activation, dtype=np.float32)
                for layer, activation in response["activations_b"].items():
                    arrays[f"activation_b_layer_{layer}"] = np.asarray(activation, dtype=np.float32)
                _atomic_npz(destination, arrays)
                completed += 1
                print(json.dumps({"event": "pair_complete", "stimulus_id": row["stimulus_id"], "completed": completed}), flush=True)
        finally:
            env.close()
    summary = {
        "study": config["study"],
        "manifest_sha256": manifest["manifest_sha256"],
        "planned": len(rows),
        "completed_or_preexisting": completed,
        "elapsed_seconds": time.monotonic() - started,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if completed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
