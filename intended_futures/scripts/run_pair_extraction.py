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


def _body_position(env: Any, subject: str, resolver: Any) -> np.ndarray:
    body_name = resolver(env, subject)
    if not body_name:
        raise RuntimeError(f"could not resolve simulator body for {subject}")
    sim = env.env.sim
    body_id = sim.model.body_name2id(body_name)
    position = np.asarray(sim.data.body_xpos[body_id], dtype=np.float64).copy()
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise RuntimeError(f"invalid body position for {subject}: {position}")
    return position


def _policy_observation(obs: dict[str, Any], prompt: str, image_tools: Any, quat2axisangle: Any) -> dict[str, Any]:
    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, 224, 224))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, 224, 224))
    return {
        "observation/image": image,
        "observation/wrist_image": wrist,
        "observation/state": np.concatenate(
            (
                obs["robot0_eef_pos"],
                quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
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
    expected_renderer = str(config["runtime_contract"]["simulator_renderer"])
    if os.environ.get("MUJOCO_GL") != expected_renderer or os.environ.get("PYOPENGL_PLATFORM") != expected_renderer:
        raise RuntimeError(f"renderer environment must be fixed to {expected_renderer!r}")
    sys.path.insert(0, str(args.libero_cf_root))
    from eval.main_cf import _get_libero_env, _quat2axisangle, _resolve_instance_root_body_name
    from libero.libero import benchmark
    from openpi_client import image_tools
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["study"] != config["study"] or manifest["manifest_sha256"] != config["manifest_sha256"]:
        raise RuntimeError("manifest does not match the frozen configuration")
    rows = manifest["rows"][: args.limit]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["pair_id"]].append(row)
    task_suite = benchmark.get_benchmark_dict()[manifest["suite"]]()
    client = WebsocketClientPolicy(args.host, args.port)
    completed = 0
    started = time.monotonic()

    for pair_id, pair_rows in grouped.items():
        first = pair_rows[0]
        task_a_index = int(first["task_a"]["task_index"])
        task_b_index = int(first["task_b"]["task_index"])
        task_a = task_suite.get_task(task_a_index)
        states_a = task_suite.get_task_init_states(task_a_index)
        states_b = task_suite.get_task_init_states(task_b_index)
        if states_a.shape != states_b.shape or not np.array_equal(states_a, states_b):
            raise RuntimeError(f"loaded initial states are not identical for {pair_id}")
        env, _ = _get_libero_env(task_a, LIBERO_ENV_RESOLUTION, args.environment_seed)
        try:
            for row in pair_rows:
                destination = args.output / f"{row['stimulus_id']}.npz"
                if destination.exists():
                    with np.load(destination, allow_pickle=False) as existing:
                        metadata = json.loads(str(existing["metadata_json"].item()))
                    if metadata["stimulus_id"] != row["stimulus_id"] or metadata["noise_seed"] != row["noise_seed"]:
                        raise RuntimeError(f"preexisting record identity mismatch: {destination}")
                    completed += 1
                    continue

                env.reset()
                obs = env.set_init_state(states_a[int(row["initial_state_index"])])
                for _ in range(10):
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
                target_a = _body_position(env, row["task_a"]["intended_subject"], _resolve_instance_root_body_name)
                target_b = _body_position(env, row["task_b"]["intended_subject"], _resolve_instance_root_body_name)
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
                    "study": manifest["study"],
                    "manifest_sha256": manifest["manifest_sha256"],
                    "stimulus_id": row["stimulus_id"],
                    "pair_id": pair_id,
                    "initial_state_index": int(row["initial_state_index"]),
                    "noise_seed": int(row["noise_seed"]),
                    "prompt_a": row["task_a"]["prompt"],
                    "prompt_b": row["task_b"]["prompt"],
                    "subject_a": row["task_a"]["intended_subject"],
                    "subject_b": row["task_b"]["intended_subject"],
                    "denoising_calls_a": response["denoising_calls_a"],
                    "denoising_calls_b": response["denoising_calls_b"],
                    "pair_infer_ms": float(response["pair_infer_ms"]),
                }
                arrays: dict[str, Any] = {
                    "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
                    "eef_position": eef,
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
        "study": manifest["study"],
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
