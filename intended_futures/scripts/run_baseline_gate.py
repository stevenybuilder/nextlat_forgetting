#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


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


def _valid_action_chunk(actions: Any, *, replan_steps: int) -> tuple[bool, str | None, np.ndarray | None]:
    try:
        array = np.asarray(actions, dtype=np.float64)
    except Exception as error:
        return False, f"action conversion failed: {error}", None
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[0] < replan_steps or array.shape[1] != 7:
        return False, f"unexpected action shape: {array.shape}", None
    if not np.all(np.isfinite(array)):
        return False, "action chunk contains non-finite values", None
    return True, None, array


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--libero-cf-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--environment-seed", type=int, default=7)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    expected_renderer = str(config["runtime_contract"]["simulator_renderer"])
    if os.environ.get("MUJOCO_GL") != expected_renderer or os.environ.get("PYOPENGL_PLATFORM") != expected_renderer:
        raise RuntimeError(f"renderer environment must be fixed to {expected_renderer!r}")
    sys.path.insert(0, str(args.libero_cf_root))
    from eval.main import _get_libero_env, _quat2axisangle
    from intended_futures.gates import evaluate_baseline_gate
    from libero.libero import benchmark
    from openpi_client import image_tools
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    frozen = config["benchmark"]["baseline_gate"]
    task_indices = [int(value) for value in frozen["task_indices"]]
    state_indices = [int(value) for value in frozen["initial_state_indices"]]
    expected_ids = [f"task-{task_index:02d}-state-{state_index:02d}" for task_index in task_indices for state_index in state_indices]
    task_suite = benchmark.get_benchmark_dict()[frozen["suite"]]()
    client = WebsocketClientPolicy(args.host, args.port)
    args.output.mkdir(parents=True, exist_ok=True)
    records_path = args.output / "episodes.json"
    records: list[dict[str, Any]] = []
    if records_path.exists():
        records = json.loads(records_path.read_text(encoding="utf-8"))
    existing = {str(record["episode_id"]): record for record in records}
    if len(existing) != len(records) or not set(existing).issubset(expected_ids):
        raise RuntimeError("preexisting baseline records do not match the frozen episode population")

    started = time.monotonic()
    for task_index in task_indices:
        task = task_suite.get_task(task_index)
        initial_states = task_suite.get_task_init_states(task_index)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.environment_seed)
        try:
            for state_index in state_indices:
                episode_id = f"task-{task_index:02d}-state-{state_index:02d}"
                if episode_id in existing:
                    continue
                record: dict[str, Any] = {
                    "episode_id": episode_id,
                    "task_index": task_index,
                    "initial_state_index": state_index,
                    "prompt": str(task_description),
                    "valid": True,
                    "success": False,
                    "steps": 0,
                    "policy_calls": 0,
                    "error": None,
                }
                episode_started = time.monotonic()
                try:
                    env.reset()
                    obs = env.set_init_state(initial_states[state_index])
                    action_plan: deque[np.ndarray] = deque()
                    total_limit = int(frozen["max_steps"]) + int(frozen["wait_steps"])
                    timestep = 0
                    while timestep < total_limit:
                        if timestep < int(frozen["wait_steps"]):
                            obs, _, done, _ = env.step(LIBERO_DUMMY_ACTION)
                            timestep += 1
                            if done:
                                raise RuntimeError("environment terminated during stabilization")
                            continue
                        if not action_plan:
                            request = _policy_observation(obs, str(task_description), image_tools, _quat2axisangle)
                            outputs = client.infer(request)
                            record["policy_calls"] += 1
                            valid, error, action_chunk = _valid_action_chunk(
                                outputs.get("actions"), replan_steps=int(config["model"]["replan_steps"])
                            )
                            if not valid or action_chunk is None:
                                raise RuntimeError(error)
                            action_plan.extend(action_chunk[: int(config["model"]["replan_steps"])] )
                        action = action_plan.popleft()
                        obs, _, done, _ = env.step(action.tolist())
                        timestep += 1
                        if done:
                            record["success"] = True
                            break
                    record["steps"] = timestep
                except Exception as error:
                    record["valid"] = False
                    record["error"] = f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
                record["elapsed_seconds"] = time.monotonic() - episode_started
                records.append(record)
                existing[episode_id] = record
                _atomic_json(records_path, records)
                print(json.dumps({key: record[key] for key in ("episode_id", "valid", "success", "steps", "policy_calls")}), flush=True)
        finally:
            env.close()

    ordered = [existing[episode_id] for episode_id in expected_ids if episode_id in existing]
    gate = evaluate_baseline_gate(
        ordered,
        expected_episode_ids=expected_ids,
        minimum_successes=int(frozen["minimum_successes"]),
        invalid_episodes_allowed=int(frozen["invalid_episodes_allowed"]),
    )
    summary = {
        "study": config["study"],
        "suite": frozen["suite"],
        "task_indices": task_indices,
        "initial_state_indices": state_indices,
        "no_retries_or_replacements": not bool(frozen["retry_failed_or_invalid_episodes"]),
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        **gate,
    }
    _atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if gate["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
