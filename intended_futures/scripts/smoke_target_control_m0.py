#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def _policy_observation(
    obs: dict[str, Any], prompt: str, image_tools: Any, quat2axisangle: Any
) -> dict[str, Any]:
    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    return {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(image, 224, 224)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist, 224, 224)
        ),
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
    parser.add_argument("--m0-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--libero-cf-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--environment-seed", type=int, default=7)
    args = parser.parse_args()

    protocol = json.loads(args.m0_config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selected_scenes = set(protocol["population"]["scene_ids"])
    selected_states = set(protocol["population"]["initial_state_indices"])
    try:
        row = next(
            row
            for row in manifest["rows"]
            if row["scene_id"] in selected_scenes
            and row["initial_state_index"] not in selected_states
        )
    except StopIteration as error:
        raise RuntimeError("manifest has no state excluded from the frozen M0 population") from error

    sys.path.insert(0, str(args.libero_cf_root))
    from eval.main_cf import _get_libero_env, _quat2axisangle
    from libero.libero import benchmark
    from openpi_client import image_tools
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    suite = benchmark.get_benchmark_dict()[manifest["suite"]]()
    task_index = int(row["task_index"])
    task = suite.get_task(task_index)
    initial_states = suite.get_task_init_states(task_index)
    env, _ = _get_libero_env(task, 256, args.environment_seed)
    client = WebsocketClientPolicy(args.host, args.port)
    donor_prompt = str(row["task_a"]["prompt"])
    recipient_prompt = str(row["task_b"]["prompt"])
    rollout = protocol["rollout"]
    receipts: dict[str, Any] = {}

    try:
        obs = env.set_init_state(initial_states[int(row["initial_state_index"])])
        for _ in range(10):
            obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
        for condition in protocol["conditions"]:
            if condition["patch_kind"] == "none":
                continue
            site = next(
                site for site in protocol["sites"] if site["site_id"] == condition["site_id"]
            )
            request = _policy_observation(obs, recipient_prompt, image_tools, _quat2axisangle)
            request.update(
                {
                    "__intended_futures_mode": "causal_action",
                    "__recipient_prompt": recipient_prompt,
                    "__donor_prompt": donor_prompt,
                    "__noise_seed": int(row["noise_seed"])
                    + int(rollout["noise_seed_offset"]),
                    "__selected_pathway": site["pathway"],
                    "__selected_layer": int(site["layer"]),
                    "__patch_kind": condition["patch_kind"],
                    "__patch_schedule": condition["patch_schedule"],
                    "__random_direction_seed": int(row["noise_seed"])
                    + int(rollout["random_direction_seed_offset"])
                    + int(site["layer"]) * 100,
                }
            )
            response = client.infer(request)
            actions = np.asarray(response["actions"], dtype=np.float64)
            if actions.shape != (10, 7) or not np.all(np.isfinite(actions)):
                raise RuntimeError(f"invalid smoke-test action chunk: {actions.shape}")
            receipt = response["patch_receipt"]
            if (
                int(receipt["calls_seen"]) <= 0
                or int(receipt["calls_patched"]) != int(receipt["calls_seen"])
                or int(receipt.get("shape_mismatches", 0)) != 0
            ):
                raise RuntimeError(f"non-exact smoke-test patch receipt: {receipt}")
            receipts[condition["condition_id"]] = {
                "calls_seen": int(receipt["calls_seen"]),
                "calls_patched": int(receipt["calls_patched"]),
                "shape_mismatches": int(receipt.get("shape_mismatches", 0)),
                "action_shape": list(actions.shape),
                "finite_actions": True,
            }
    finally:
        env.close()

    output = {
        "study": protocol["study"],
        "status": "passed",
        "excluded_stimulus_id": row["stimulus_id"],
        "receipts": receipts,
    }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
