#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import numpy as np


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _progress(start: np.ndarray, end: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(start - target) - np.linalg.norm(end - target))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-config", type=Path, required=True)
    parser.add_argument("--causal-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--subspace", type=Path, required=True)
    parser.add_argument("--libero-cf-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--environment-seed", type=int, default=7)
    args = parser.parse_args()

    project_src = Path(__file__).parents[1] / "src"
    sys.path.insert(0, str(project_src))
    from intended_futures.config import load_config
    from intended_futures.geometry import random_orthonormal_basis
    from intended_futures.stimuli import validate_subject_positions

    pilot = load_config(args.pilot_config)
    causal = json.loads(args.causal_config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if causal["parent_study"] != pilot["study"] or causal["parent_manifest_sha256"] != manifest["manifest_sha256"]:
        raise RuntimeError("causal protocol does not match the parent pilot")
    if _sha256(args.subspace) != causal["future_subspace_sha256"]:
        raise RuntimeError("future subspace hash differs from the frozen causal protocol")
    renderer = pilot["runtime_contract"]["simulator_renderer"]
    if os.environ.get("MUJOCO_GL") != renderer or os.environ.get("PYOPENGL_PLATFORM") != renderer:
        raise RuntimeError(f"renderer environment must be fixed to {renderer!r}")

    with np.load(args.subspace, allow_pickle=False) as fitted:
        selected_layer = int(fitted["selected_layer"])
        basis = np.asarray(fitted["basis"], dtype=np.float32)
        input_shape = tuple(int(value) for value in fitted["input_shape"])
    if selected_layer != int(causal["selected_layer"]) or basis.shape[0] != int(np.prod(input_shape)):
        raise RuntimeError("selected subspace metadata violates the causal protocol")
    random_basis = random_orthonormal_basis(
        basis.shape[0], basis.shape[1], seed=int(causal["sampling"]["random_basis_seed"])
    ).astype(np.float32)

    sys.path.insert(0, str(args.libero_cf_root))
    from eval.main_cf import _get_libero_env, _quat2axisangle
    from libero.libero import benchmark
    from openpi_client import image_tools
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    selected_states = set(int(value) for value in causal["sampling"]["initial_state_indices"])
    selected_scenes = set(str(value) for value in causal["sampling"]["scene_ids"])
    rows = [
        row
        for row in manifest["rows"]
        if row["scene_id"] in selected_scenes and int(row["initial_state_index"]) in selected_states
    ]
    if len(rows) != int(causal["sampling"]["expected_causal_units"]):
        raise RuntimeError("causal row count differs from the frozen protocol")
    client = WebsocketClientPolicy(args.host, args.port)
    suite = benchmark.get_benchmark_dict()[manifest["suite"]]()
    conditions = list(causal["rollout"]["conditions"])
    contract = pilot["benchmark"]["workspace_position_contract"]
    args.output.mkdir(parents=True, exist_ok=True)
    completed = 0
    invalid = 0
    started = time.monotonic()

    for row in rows:
        destination = args.output / f"{row['stimulus_id']}.json"
        if destination.exists():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if existing["stimulus_id"] != row["stimulus_id"] or existing["noise_seed"] != row["noise_seed"]:
                raise RuntimeError(f"preexisting causal record identity mismatch: {destination}")
            completed += 1
            invalid += int(not existing["valid"])
            continue
        task_index = int(row["task_index"])
        task = suite.get_task(task_index)
        states = suite.get_task_init_states(task_index)
        env, _ = _get_libero_env(task, 256, args.environment_seed)
        record: dict[str, Any] = {
            "study": causal["study"],
            "parent_manifest_sha256": manifest["manifest_sha256"],
            "stimulus_id": row["stimulus_id"],
            "scene_id": row["scene_id"],
            "task_index": task_index,
            "initial_state_index": int(row["initial_state_index"]),
            "noise_seed": int(row["noise_seed"]),
            "donor_prompt": row["task_a"]["prompt"],
            "recipient_prompt": row["task_b"]["prompt"],
            "donor_subject": row["task_a"]["intended_subject"],
            "recipient_subject": row["task_b"]["intended_subject"],
            "selected_layer": selected_layer,
            "valid": True,
            "error": None,
            "conditions": {},
        }
        order_rng = np.random.default_rng(
            int(causal["sampling"]["condition_order_seed"])
            + int(row["task_index"]) * 100
            + int(row["initial_state_index"])
        )
        condition_order = [conditions[index] for index in order_rng.permutation(len(conditions))]
        record["condition_order"] = condition_order
        try:
            for condition in condition_order:
                env.reset()
                obs = env.set_init_state(states[int(row["initial_state_index"])])
                for _ in range(10):
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                subjects = [row["task_a"]["intended_subject"], row["task_b"]["intended_subject"]]
                positions = validate_subject_positions(obs, subjects, contract)
                donor_target = positions[subjects[0]].copy()
                recipient_target = positions[subjects[1]].copy()
                start_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
                replans = []
                done = False
                for replan_index in range(int(causal["rollout"]["replans"])):
                    request = _policy_observation(obs, row["task_b"]["prompt"], image_tools, _quat2axisangle)
                    request.update(
                        {
                            "__intended_futures_mode": "causal_action",
                            "__recipient_prompt": row["task_b"]["prompt"],
                            "__donor_prompt": row["task_a"]["prompt"],
                            "__noise_seed": int(row["noise_seed"])
                            + int(causal["sampling"]["replan_noise_seed_offset"])
                            + replan_index,
                            "__selected_layer": selected_layer,
                            "__patch_kind": condition,
                            "__future_basis": basis,
                            "__random_basis": random_basis,
                        }
                    )
                    response = client.infer(request)
                    actions = np.asarray(response["actions"], dtype=np.float64)
                    execute = int(causal["rollout"]["execute_actions_per_replan"])
                    if actions.ndim != 2 or actions.shape[0] < execute or actions.shape[1] != 7 or not np.all(np.isfinite(actions)):
                        raise RuntimeError(f"invalid causal action chunk: {actions.shape}")
                    executed = 0
                    for action in actions[:execute]:
                        obs, _, done, _ = env.step(action.tolist())
                        executed += 1
                        if done:
                            break
                    replans.append(
                        {
                            "replan_index": replan_index,
                            "noise_seed": int(response["noise_seed"]),
                            "actions_executed": executed,
                            "patch_receipt": response["patch_receipt"],
                        }
                    )
                    if done:
                        break
                end_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
                record["conditions"][condition] = {
                    "start_eef": start_eef.tolist(),
                    "end_eef": end_eef.tolist(),
                    "donor_target": donor_target.tolist(),
                    "recipient_target": recipient_target.tolist(),
                    "donor_progress": _progress(start_eef, end_eef, donor_target),
                    "recipient_progress": _progress(start_eef, end_eef, recipient_target),
                    "environment_done": bool(done),
                    "replans": replans,
                }
        except Exception as error:
            record["valid"] = False
            record["error"] = f"{type(error).__name__}: {error}\n{traceback.format_exc()}"
        finally:
            env.close()
        _atomic_json(destination, record)
        completed += 1
        invalid += int(not record["valid"])
        print(
            json.dumps(
                {
                    "event": "causal_unit_complete",
                    "stimulus_id": row["stimulus_id"],
                    "completed": completed,
                    "valid": record["valid"],
                }
            ),
            flush=True,
        )

    summary = {
        "study": causal["study"],
        "parent_manifest_sha256": manifest["manifest_sha256"],
        "expected": len(rows),
        "completed_or_preexisting": completed,
        "invalid": invalid,
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(args.output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if completed == len(rows) and invalid == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
