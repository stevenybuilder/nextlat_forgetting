#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--clearance", type=Path, required=True)
    parser.add_argument("--libero-plus-root", type=Path, required=True)
    parser.add_argument("--libero-cf-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--environment-seed", type=int, default=7)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from intended_futures.contacts import touched_instances
    from intended_futures.stimuli import validate_subject_positions

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    receipt = json.loads(args.runtime_receipt.read_text(encoding="utf-8"))
    clearance = json.loads(args.clearance.read_text(encoding="utf-8"))
    if config.get("status") != "frozen":
        raise RuntimeError("causal collection requires a frozen config")
    if manifest.get("config_sha256") != _sha256(args.config):
        raise RuntimeError("manifest is not bound to this config")
    if (
        receipt.get("study") != config["study"]
        or receipt.get("config_sha256") != _sha256(args.config)
        or receipt.get("manifest_sha256") != manifest["manifest_sha256"]
        or receipt.get("gpu_count") != 1
        or receipt.get("distributed") is not False
    ):
        raise RuntimeError("runtime receipt does not certify this causal run")
    if (
        clearance.get("authorization") != "GO_CAUSAL_TEST"
        or clearance.get("all_observer_gates_passed") is not True
        or clearance.get("config_sha256") != _sha256(args.config)
        or clearance.get("manifest_sha256") != manifest["manifest_sha256"]
        or clearance.get("controller_sha256") != _sha256(args.controller)
    ):
        raise RuntimeError("observer clearance does not authorize this causal run")
    renderer = config["runtime_contract"]["simulator_renderer"]
    if os.environ.get("MUJOCO_GL") != renderer or os.environ.get("PYOPENGL_PLATFORM") != renderer:
        raise RuntimeError(f"renderer environment must be fixed to {renderer!r}")

    # The level/sample assets use standard LIBERO objects. Keep the already validated
    # LIBERO-CF simulator package and read only BDDL/init assets from the pinned Plus tree.
    sys.path.insert(0, str(args.libero_cf_root))
    import torch
    from eval.main_cf import _quat2axisangle
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import image_tools
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    with np.load(args.controller, allow_pickle=False) as artifact:
        family_mean_xyz = json.loads(str(artifact["family_mean_target_xyz_json"].item()))
    rows = [row for row in manifest["rows"] if row["split"] == "causal_test"]
    expected = int(config["population"]["expected_split_units"]["causal_test"])
    if len(rows) != expected:
        raise RuntimeError(f"causal split has {len(rows)} rows, expected {expected}")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[: args.limit]
    conditions = list(config["conditions"])
    bddl_root = args.libero_plus_root / "libero" / "libero" / "bddl_files"
    init_root = args.libero_plus_root / "libero" / "libero" / "init_files"
    contract = config["stimulus"]["workspace_position_contract"]
    rollout = config["rollout"]
    site = config["site"]
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
                existing.get("study") != config["study"]
                or existing.get("manifest_sha256") != manifest["manifest_sha256"]
                or existing.get("controller_sha256") != _sha256(args.controller)
            ):
                raise RuntimeError(f"preexisting record identity mismatch: {destination}")
            completed += 1
            invalid += int(not existing["valid"])
            continue

        bddl = bddl_root / row["task_file"]
        init_file = init_root / row["init_file"]
        if _sha256(bddl) != row["bddl_sha256"] or _sha256(init_file) != row["init_sha256"]:
            raise RuntimeError(f"upstream stimulus hash mismatch for {row['stimulus_id']}")
        states = np.asarray(torch.load(init_file), dtype=np.float64).reshape(1, -1)
        prompt_a = str(row["task_a"]["prompt"])
        prompt_b = str(row["task_b"]["prompt"])
        subject_a = str(row["task_a"]["intended_subject"])
        subject_b = str(row["task_b"]["intended_subject"])
        record: dict[str, Any] = {
            "study": config["study"],
            "config_sha256": _sha256(args.config),
            "manifest_sha256": manifest["manifest_sha256"],
            "runtime_receipt_sha256": _sha256(args.runtime_receipt),
            "controller_sha256": _sha256(args.controller),
            "clearance_sha256": _sha256(args.clearance),
            "stimulus_id": row["stimulus_id"],
            "family_id": row["family_id"],
            "source_task_id": int(row["source_task_id"]),
            "difficulty_level": int(row["difficulty_level"]),
            "level": int(row["level"]),
            "sample": int(row["sample"]),
            "valid": True,
            "error": None,
            "conditions": {},
        }
        order_rng = np.random.default_rng(
            int(rollout["condition_order_seed"]) + int(row["source_task_id"])
        )
        ordered_conditions = [
            conditions[index] for index in order_rng.permutation(len(conditions))
        ]
        record["condition_order"] = [item["condition_id"] for item in ordered_conditions]

        env = OffScreenRenderEnv(
            bddl_file_name=bddl, camera_heights=256, camera_widths=256
        )
        env.seed(args.environment_seed)
        try:
            for condition in ordered_conditions:
                env.reset()
                obs = env.set_init_state(states[0])
                for _ in range(10):
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
                positions = validate_subject_positions(
                    obs, [subject_a, subject_b], contract
                )
                target_a = positions[subject_a].copy()
                target_b = positions[subject_b].copy()
                actual_delta = target_a - target_b
                if "simulator_target_difference_xy" in row:
                    target_reference = "simulator_fixed_state"
                    reference_delta = np.asarray(
                        row["simulator_target_difference_xy"], dtype=np.float64
                    )
                    tolerance = float(
                        config["stimulus"][
                            "maximum_manifest_to_simulator_xy_error_meters"
                        ]
                    )
                else:
                    target_reference = "bddl_region_center"
                    reference_delta = np.asarray(
                        row["bddl_target_difference_xy"], dtype=np.float64
                    )
                    tolerance = float(
                        config["stimulus"][
                            "maximum_bddl_to_simulator_xy_error_meters"
                        ]
                    )
                if float(np.linalg.norm(actual_delta[:2] - reference_delta)) > tolerance:
                    raise RuntimeError(
                        f"simulator target positions disagree with {target_reference} preflight"
                    )
                actual_delta[2] = 0.0
                family_delta = np.asarray(
                    family_mean_xyz[row["family_id"]], dtype=np.float64
                )
                start_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).copy()
                condition_id = str(condition["condition_id"])
                active_prompt = prompt_a if condition["prompt_side"] == "a" else prompt_b
                other_prompt = prompt_b if condition["prompt_side"] == "a" else prompt_a
                patch_kind = str(condition["patch_kind"])
                desired_delta = (
                    family_delta
                    if condition_id == "prompt_mean_controller"
                    else actual_delta
                )
                patch_schedule = (
                    "all_calls_replay" if patch_kind == "full_donor" else "all_calls_delta"
                )
                replans: list[dict[str, Any]] = []
                first_action_chunk: list[list[float]] | None = None
                first_touch: str | None = None
                first_touch_step: int | None = None
                executed_total = 0
                done = False

                for replan_index in range(int(rollout["replans"])):
                    request = _policy_observation(
                        obs, active_prompt, image_tools, _quat2axisangle
                    )
                    request.update(
                        {
                            "__intended_futures_mode": "causal_action",
                            "__recipient_prompt": active_prompt,
                            "__donor_prompt": other_prompt if patch_kind == "none" else prompt_a,
                            "__noise_seed": int(row["noise_seed"])
                            + int(rollout["noise_seed_offset"])
                            + replan_index,
                            "__selected_pathway": str(site["pathway"]),
                            "__selected_layer": int(site["layer"]),
                            "__patch_kind": patch_kind,
                            "__patch_schedule": patch_schedule,
                            "__desired_target_delta": desired_delta,
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
                        raise RuntimeError(f"invalid action chunk {actions.shape}")
                    if first_action_chunk is None:
                        first_action_chunk = actions.tolist()
                    if patch_kind in {"minimum_norm_target", "random_controller"}:
                        if response["patch_receipt"].get("controller_artifact_sha256") != _sha256(args.controller):
                            raise RuntimeError("server used a different controller artifact")
                    executed_this_replan = 0
                    for action in actions[:execute]:
                        obs, _, done, _ = env.step(action.tolist())
                        executed_this_replan += 1
                        executed_total += 1
                        touched = touched_instances(env, [subject_a, subject_b])
                        if first_touch is None and touched:
                            if subject_a in touched and subject_b in touched:
                                first_touch = "both"
                            elif subject_a in touched:
                                first_touch = "a"
                            else:
                                first_touch = "b"
                            first_touch_step = executed_total
                        if done or (
                            bool(rollout["stop_on_first_candidate_touch"])
                            and first_touch is not None
                        ):
                            break
                    replans.append(
                        {
                            "replan_index": replan_index,
                            "actions_executed": executed_this_replan,
                            "patch_receipt": response["patch_receipt"],
                        }
                    )
                    if done or (
                        bool(rollout["stop_on_first_candidate_touch"])
                        and first_touch is not None
                    ):
                        break
                end_eef = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
                record["conditions"][condition_id] = {
                    "prompt": active_prompt,
                    "patch_kind": patch_kind,
                    "desired_target_delta": desired_delta.tolist(),
                    "first_touch": first_touch,
                    "first_touch_step": first_touch_step,
                    "target_a_progress": float(
                        np.linalg.norm(start_eef - target_a)
                        - np.linalg.norm(end_eef - target_a)
                    ),
                    "target_b_progress": float(
                        np.linalg.norm(start_eef - target_b)
                        - np.linalg.norm(end_eef - target_b)
                    ),
                    "terminal_target_a_distance": float(np.linalg.norm(end_eef - target_a)),
                    "terminal_target_b_distance": float(np.linalg.norm(end_eef - target_b)),
                    "first_action_chunk": first_action_chunk,
                    "actions_executed": executed_total,
                    "replans": replans,
                }
        except Exception as error:
            record["valid"] = False
            record["error"] = f"{type(error).__name__}: {error}"
            invalid += 1
        finally:
            env.close()
        _atomic_json(destination, record)
        completed += 1
        print(
            json.dumps(
                {
                    "event": "layout_causal_unit_complete",
                    "stimulus_id": row["stimulus_id"],
                    "completed": completed,
                    "valid": record["valid"],
                }
            ),
            flush=True,
        )

    summary = {
        "study": config["study"],
        "planned": len(rows),
        "completed_or_preexisting": completed,
        "invalid": invalid,
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if completed == len(rows) and invalid == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
