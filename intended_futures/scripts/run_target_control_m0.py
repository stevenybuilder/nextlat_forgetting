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


def _manifest_digest(manifest: dict[str, Any]) -> str:
    payload = {
        key: value for key, value in manifest.items() if key != "manifest_sha256"
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


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


def _validate_protocol(protocol: dict[str, Any], manifest: dict[str, Any]) -> None:
    if protocol.get("status") != "frozen":
        raise ValueError("target-control protocol must be frozen")
    if protocol["parent"]["manifest_sha256"] != manifest.get("manifest_sha256"):
        raise ValueError("target-control protocol and stimulus manifest differ")
    sites = {site["site_id"]: site for site in protocol["sites"]}
    if set(sites) != {"paligemma_l13", "expert_l9"}:
        raise ValueError("M0 sites differ from the frozen two-site assay")
    condition_ids = [condition["condition_id"] for condition in protocol["conditions"]]
    if len(condition_ids) != len(set(condition_ids)):
        raise ValueError("M0 condition IDs must be unique")
    for condition in protocol["conditions"]:
        if condition["patch_kind"] != "none" and condition.get("site_id") not in sites:
            raise ValueError(f"condition references unknown site: {condition}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-config", type=Path, required=True)
    parser.add_argument("--m0-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--libero-cf-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--environment-seed", type=int, default=7)
    args = parser.parse_args()

    project_src = Path(__file__).parents[1] / "src"
    sys.path.insert(0, str(project_src))
    from intended_futures.config import load_config
    from intended_futures.contacts import touched_instances
    from intended_futures.stimuli import validate_subject_positions

    pilot = load_config(args.pilot_config)
    protocol = json.loads(args.m0_config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    _validate_protocol(protocol, manifest)
    if _manifest_digest(manifest) != protocol["parent"]["manifest_sha256"]:
        raise RuntimeError("manifest content differs from the frozen M0 protocol")

    renderer = protocol["runtime_contract"]["simulator_renderer"]
    if os.environ.get("MUJOCO_GL") != renderer or os.environ.get("PYOPENGL_PLATFORM") != renderer:
        raise RuntimeError(f"renderer environment must be fixed to {renderer!r}")

    sys.path.insert(0, str(args.libero_cf_root))
    from eval.main_cf import _get_libero_env, _quat2axisangle
    from libero.libero import benchmark
    from openpi_client import image_tools
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    selected_scenes = set(protocol["population"]["scene_ids"])
    selected_states = set(int(index) for index in protocol["population"]["initial_state_indices"])
    rows = [
        row
        for row in manifest["rows"]
        if row["scene_id"] in selected_scenes
        and int(row["initial_state_index"]) in selected_states
    ]
    if len(rows) != int(protocol["population"]["expected_units"]):
        raise RuntimeError("M0 row count differs from the frozen protocol")

    sites = {site["site_id"]: site for site in protocol["sites"]}
    conditions = list(protocol["conditions"])
    rollout = protocol["rollout"]
    suite = benchmark.get_benchmark_dict()[manifest["suite"]]()
    client = WebsocketClientPolicy(args.host, args.port)
    contract = pilot["benchmark"]["workspace_position_contract"]
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
            ):
                raise RuntimeError(f"preexisting M0 record identity mismatch: {destination}")
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
            "config_sha256": _sha256(args.m0_config),
            "manifest_sha256": manifest["manifest_sha256"],
            "stimulus_id": row["stimulus_id"],
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
        condition_order = [conditions[index] for index in order_rng.permutation(len(conditions))]
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
                start_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
                prompt_side = str(condition["prompt_side"])
                active_prompt = donor_prompt if prompt_side == "donor" else recipient_prompt
                opposite_prompt = recipient_prompt if prompt_side == "donor" else donor_prompt
                site = sites.get(condition.get("site_id"), sites["expert_l9"])
                patch_kind = str(condition["patch_kind"])
                patch_schedule = str(condition.get("patch_schedule", "first_call"))
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
                            "__donor_prompt": opposite_prompt if patch_kind == "none" else donor_prompt,
                            "__noise_seed": int(row["noise_seed"])
                            + int(rollout["noise_seed_offset"])
                            + replan_index,
                            "__selected_pathway": str(site["pathway"]),
                            "__selected_layer": int(site["layer"]),
                            "__patch_kind": patch_kind,
                            "__patch_schedule": patch_schedule,
                            "__random_direction_seed": int(row["noise_seed"])
                            + int(rollout["random_direction_seed_offset"])
                            + int(site["layer"]) * 100
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
                        raise RuntimeError(f"invalid M0 action chunk: {actions.shape}")
                    if first_action_chunk is None:
                        first_action_chunk = actions.tolist()

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
                    raise RuntimeError("M0 condition produced no action chunk")
                end_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
                record["conditions"][condition["condition_id"]] = {
                    "prompt_side": prompt_side,
                    "patch_kind": patch_kind,
                    "patch_schedule": patch_schedule,
                    "site": site,
                    "start_eef": start_eef.tolist(),
                    "end_eef": end_eef.tolist(),
                    "donor_target": donor_target.tolist(),
                    "recipient_target": recipient_target.tolist(),
                    "donor_progress": _progress(start_eef, end_eef, donor_target),
                    "recipient_progress": _progress(start_eef, end_eef, recipient_target),
                    "first_action_chunk": first_action_chunk,
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
                    "event": "target_control_m0_unit_complete",
                    "stimulus_id": row["stimulus_id"],
                    "completed": completed,
                    "valid": record["valid"],
                }
            ),
            flush=True,
        )

    summary = {
        "study": protocol["study"],
        "config_sha256": _sha256(args.m0_config),
        "manifest_sha256": manifest["manifest_sha256"],
        "expected": len(rows),
        "completed_or_preexisting": completed,
        "invalid": invalid,
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(args.output / "collection_summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if completed == len(rows) and invalid == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
