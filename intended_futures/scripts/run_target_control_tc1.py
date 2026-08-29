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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


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


def _progress(start: np.ndarray, end: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(start - target) - np.linalg.norm(end - target))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-config", type=Path, required=True)
    parser.add_argument("--tc1-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--clearance", type=Path, required=True)
    parser.add_argument("--libero-cf-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--environment-seed", type=int, default=7)
    parser.add_argument("--split", choices=("causal_test", "reserve"), default="causal_test")
    args = parser.parse_args()

    project_src = Path(__file__).parents[1] / "src"
    sys.path.insert(0, str(project_src))
    from intended_futures.config import load_config
    from intended_futures.contacts import touched_instances
    from intended_futures.stimuli import validate_subject_positions

    pilot = load_config(args.pilot_config)
    protocol = json.loads(args.tc1_config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    runtime_receipt = json.loads(args.runtime_receipt.read_text(encoding="utf-8"))
    clearance = json.loads(args.clearance.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen":
        raise RuntimeError("TC1 causal collection requires a frozen protocol")
    if manifest["protocol_sha256"] != _sha256(args.tc1_config):
        raise RuntimeError("TC1 manifest is not bound to this protocol")
    if (
        runtime_receipt.get("study") != protocol["study"]
        or runtime_receipt.get("tc1_config_sha256") != _sha256(args.tc1_config)
        or runtime_receipt.get("manifest_sha256") != manifest["manifest_sha256"]
        or runtime_receipt.get("gpu_count") != 1
        or runtime_receipt.get("distributed") is not False
    ):
        raise RuntimeError("runtime receipt does not certify this causal run")
    if (
        clearance.get("authorization") != "GO_CAUSAL_TEST"
        or clearance.get("all_observer_gates_passed") is not True
        or clearance.get("protocol_sha256") != _sha256(args.tc1_config)
        or clearance.get("manifest_sha256") != manifest["manifest_sha256"]
        or clearance.get("controller_sha256") != _sha256(args.controller)
        or clearance.get("runtime_receipt_sha256") != _sha256(args.runtime_receipt)
    ):
        raise RuntimeError("observer clearance does not authorize this causal run")
    renderer = protocol["runtime_contract"]["simulator_renderer"]
    if os.environ.get("MUJOCO_GL") != renderer or os.environ.get("PYOPENGL_PLATFORM") != renderer:
        raise RuntimeError(f"renderer environment must be fixed to {renderer!r}")

    sys.path.insert(0, str(args.libero_cf_root))
    from eval.main_cf import _get_libero_env, _quat2axisangle
    from libero.libero import benchmark
    from openpi_client import image_tools
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    rows = [row for row in manifest["rows"] if row["split"] == args.split]
    expected = (
        int(protocol["population"]["expected_causal_test_units"])
        if args.split == "causal_test"
        else len(protocol["population"]["scene_ids"])
        * len(protocol["population"]["reserve_state_indices"])
    )
    if len(rows) != expected:
        raise RuntimeError(f"causal test has {len(rows)} rows, expected {expected}")
    conditions = list(protocol["conditions"])
    if len({condition["condition_id"] for condition in conditions}) != len(conditions):
        raise RuntimeError("TC1 condition identifiers are not unique")
    site = protocol["site"]
    rollout = protocol["rollout"]
    suite = benchmark.get_benchmark_dict()[manifest["suite"]]()
    contract = pilot["benchmark"]["workspace_position_contract"]
    client = WebsocketClientPolicy(args.host, args.port)
    args.output.mkdir(parents=True, exist_ok=True)
    completed = 0
    invalid = 0
    started = time.monotonic()

    for row in rows:
        destination = args.output / f"{row['stimulus_id']}.json"
        if destination.exists():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if (
                existing.get("study") != protocol["study"]
                or existing.get("stimulus_id") != row["stimulus_id"]
                or existing.get("split") != args.split
                or existing.get("controller_sha256") != _sha256(args.controller)
            ):
                raise RuntimeError(f"preexisting TC1 record identity mismatch: {destination}")
            completed += 1
            invalid += int(not existing["valid"])
            continue

        task_index = int(row["task_index"])
        task = suite.get_task(task_index)
        initial_states = suite.get_task_init_states(task_index)
        env, _ = _get_libero_env(task, 256, args.environment_seed)
        donor_prompt = str(row["task_a"]["prompt"])
        recipient_prompt = str(row["task_b"]["prompt"])
        donor_subject = str(row["task_a"]["intended_subject"])
        recipient_subject = str(row["task_b"]["intended_subject"])
        record: dict[str, Any] = {
            "study": protocol["study"],
            "protocol_sha256": _sha256(args.tc1_config),
            "manifest_sha256": manifest["manifest_sha256"],
            "controller_sha256": _sha256(args.controller),
            "clearance_sha256": _sha256(args.clearance),
            "runtime_receipt_sha256": _sha256(args.runtime_receipt),
            "stimulus_id": row["stimulus_id"],
            "split": args.split,
            "scene_id": row["scene_id"],
            "task_index": task_index,
            "initial_state_index": int(row["initial_state_index"]),
            "donor_prompt": donor_prompt,
            "recipient_prompt": recipient_prompt,
            "donor_subject": donor_subject,
            "recipient_subject": recipient_subject,
            "valid": True,
            "error": None,
            "conditions": {},
        }
        order_rng = np.random.default_rng(
            int(rollout["condition_order_seed"])
            + task_index * 100
            + int(row["initial_state_index"])
        )
        condition_order = [
            conditions[index] for index in order_rng.permutation(len(conditions))
        ]
        record["condition_order"] = [condition["condition_id"] for condition in condition_order]

        try:
            for condition in condition_order:
                env.reset()
                obs = env.set_init_state(initial_states[int(row["initial_state_index"])])
                for _ in range(10):
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                positions = validate_subject_positions(
                    obs, [donor_subject, recipient_subject], contract
                )
                donor_target = positions[donor_subject].copy()
                recipient_target = positions[recipient_subject].copy()
                desired_target_delta = donor_target - recipient_target
                start_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
                prompt_side = str(condition["prompt_side"])
                active_prompt = donor_prompt if prompt_side == "donor" else recipient_prompt
                other_prompt = recipient_prompt if prompt_side == "donor" else donor_prompt
                patch_kind = str(condition["patch_kind"])
                if patch_kind == "full_donor":
                    patch_schedule = "all_calls_replay"
                elif patch_kind == "none":
                    patch_schedule = "all_calls_delta"
                else:
                    patch_schedule = "all_calls_delta"
                replans: list[dict[str, Any]] = []
                first_action_chunk: list[list[float]] | None = None
                first_touch: str | None = None
                first_touch_step: int | None = None
                touched_donor = False
                touched_recipient = False
                done = False
                executed_total = 0

                for replan_index in range(int(rollout["replans"])):
                    request = _policy_observation(obs, active_prompt, image_tools, _quat2axisangle)
                    request.update(
                        {
                            "__intended_futures_mode": "causal_action",
                            "__recipient_prompt": active_prompt,
                            "__donor_prompt": other_prompt if patch_kind == "none" else donor_prompt,
                            "__noise_seed": int(row["noise_seed"])
                            + int(rollout["noise_seed_offset"])
                            + replan_index,
                            "__selected_pathway": str(site["pathway"]),
                            "__selected_layer": int(site["layer"]),
                            "__patch_kind": patch_kind,
                            "__patch_schedule": patch_schedule,
                            "__desired_target_delta": desired_target_delta,
                            "__random_direction_seed": int(row["noise_seed"])
                            + int(rollout["random_direction_seed_offset"])
                            + replan_index,
                        }
                    )
                    response = client.infer(request)
                    actions = np.asarray(response["actions"], dtype=np.float64)
                    execute = int(rollout["execute_actions_per_replan"])
                    if (
                        actions.ndim != 2
                        or actions.shape[0] < execute
                        or actions.shape[1] != 7
                        or not np.all(np.isfinite(actions))
                    ):
                        raise RuntimeError(f"invalid TC1 action chunk: {actions.shape}")
                    if first_action_chunk is None:
                        first_action_chunk = actions.tolist()
                    if patch_kind not in {"none", "full_donor"}:
                        receipt_hash = response["patch_receipt"].get(
                            "controller_artifact_sha256"
                        )
                        if receipt_hash != _sha256(args.controller):
                            raise RuntimeError("server used a different target-controller artifact")

                    executed_this_replan = 0
                    for action in actions[:execute]:
                        obs, _, done, _ = env.step(action.tolist())
                        executed_this_replan += 1
                        executed_total += 1
                        touched = touched_instances(env, [donor_subject, recipient_subject])
                        touched_donor = touched_donor or donor_subject in touched
                        touched_recipient = touched_recipient or recipient_subject in touched
                        if first_touch is None and touched:
                            if donor_subject in touched and recipient_subject in touched:
                                first_touch = "both"
                            elif donor_subject in touched:
                                first_touch = "donor"
                            else:
                                first_touch = "recipient"
                            first_touch_step = executed_total
                        if done or (
                            bool(rollout["stop_on_first_candidate_touch"])
                            and first_touch is not None
                        ):
                            break
                    replans.append(
                        {
                            "replan_index": replan_index,
                            "noise_seed": int(response["noise_seed"]),
                            "actions_executed": executed_this_replan,
                            "patch_receipt": response["patch_receipt"],
                        }
                    )
                    if done or (
                        bool(rollout["stop_on_first_candidate_touch"])
                        and first_touch is not None
                    ):
                        break

                if first_action_chunk is None:
                    raise RuntimeError("TC1 condition produced no action chunk")
                end_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
                record["conditions"][condition["condition_id"]] = {
                    "prompt_side": prompt_side,
                    "patch_kind": patch_kind,
                    "first_action_chunk": first_action_chunk,
                    "start_eef": start_eef.tolist(),
                    "end_eef": end_eef.tolist(),
                    "donor_target": donor_target.tolist(),
                    "recipient_target": recipient_target.tolist(),
                    "donor_progress": _progress(start_eef, end_eef, donor_target),
                    "recipient_progress": _progress(start_eef, end_eef, recipient_target),
                    "first_touch": first_touch,
                    "first_touch_step": first_touch_step,
                    "touched_donor": touched_donor,
                    "touched_recipient": touched_recipient,
                    "environment_done": bool(done),
                    "actions_executed": executed_total,
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
                    "event": "tc1_unit_complete",
                    "stimulus_id": row["stimulus_id"],
                    "completed": completed,
                    "valid": record["valid"],
                }
            ),
            flush=True,
        )

    summary = {
        "study": protocol["study"],
        "split": args.split,
        "manifest_sha256": manifest["manifest_sha256"],
        "controller_sha256": _sha256(args.controller),
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
