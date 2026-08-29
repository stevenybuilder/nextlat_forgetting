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


def _atomic_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
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
    parser.add_argument("--libero-plus-root", type=Path, required=True)
    parser.add_argument("--libero-cf-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--environment-seed", type=int, default=7)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    project_src = Path(__file__).parents[1] / "src"
    sys.path.insert(0, str(project_src))
    from intended_futures.stimuli import validate_subject_positions

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    receipt = json.loads(args.runtime_receipt.read_text(encoding="utf-8"))
    if config.get("status") != "frozen":
        raise RuntimeError("observer collection requires a frozen config")
    if manifest.get("config_sha256") != _sha256(args.config):
        raise RuntimeError("manifest is not bound to this config")
    if (
        receipt.get("study") != config["study"]
        or receipt.get("written_before_model_outcomes") is not True
        or receipt.get("config_sha256") != _sha256(args.config)
        or receipt.get("manifest_sha256") != manifest["manifest_sha256"]
        or receipt.get("gpu_count") != 1
        or receipt.get("distributed") is not False
    ):
        raise RuntimeError("runtime receipt does not certify this observer run")
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

    rows = [
        row
        for row in manifest["rows"]
        if row["split"] in {"observer_fit", "observer_validation"}
    ]
    expected = sum(
        int(config["population"]["expected_split_units"][split])
        for split in ("observer_fit", "observer_validation")
    )
    if len(rows) != expected:
        raise RuntimeError(f"observer has {len(rows)} rows, expected {expected}")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[: args.limit]
    bddl_root = args.libero_plus_root / "libero" / "libero" / "bddl_files"
    init_root = args.libero_plus_root / "libero" / "libero" / "init_files"
    layer = int(config["site"]["layer"])
    contract = config["stimulus"]["workspace_position_contract"]
    client = WebsocketClientPolicy(args.host, args.port)
    args.output.mkdir(parents=True, exist_ok=True)
    completed = 0
    started = time.monotonic()

    for row in rows:
        destination = args.output / f"{row['stimulus_id']}.npz"
        if destination.exists():
            with np.load(destination, allow_pickle=False) as existing:
                metadata = json.loads(str(existing["metadata_json"].item()))
            if (
                metadata.get("study") != config["study"]
                or metadata.get("manifest_sha256") != manifest["manifest_sha256"]
                or metadata.get("split") != row["split"]
            ):
                raise RuntimeError(f"preexisting record identity mismatch: {destination}")
            completed += 1
            continue

        bddl = bddl_root / row["task_file"]
        init_file = init_root / row["init_file"]
        if _sha256(bddl) != row["bddl_sha256"] or _sha256(init_file) != row["init_sha256"]:
            raise RuntimeError(f"upstream stimulus hash mismatch for {row['stimulus_id']}")
        env = OffScreenRenderEnv(
            bddl_file_name=bddl, camera_heights=256, camera_widths=256
        )
        env.seed(args.environment_seed)
        try:
            states = torch.load(init_file)
            states = np.asarray(states, dtype=np.float64).reshape(1, -1)
            env.reset()
            obs = env.set_init_state(states[0])
            for _ in range(10):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            subjects = [
                str(row["task_a"]["intended_subject"]),
                str(row["task_b"]["intended_subject"]),
            ]
            positions = validate_subject_positions(obs, subjects, contract)
            request = _policy_observation(
                obs, str(row["task_a"]["prompt"]), image_tools, _quat2axisangle
            )
            request.update(
                {
                    "__intended_futures_mode": "extract_pair",
                    "__prompt_a": row["task_a"]["prompt"],
                    "__prompt_b": row["task_b"]["prompt"],
                    "__noise_seed": int(row["noise_seed"]),
                }
            )
            response = client.infer(request)
            key = str(layer)
            activation_a = np.asarray(
                response["paligemma_activations_a"][key], dtype=np.float32
            )
            activation_b = np.asarray(
                response["paligemma_activations_b"][key], dtype=np.float32
            )
            if activation_a.shape != activation_b.shape or activation_a.ndim != 2:
                raise RuntimeError(
                    f"invalid activation shapes {activation_a.shape}, {activation_b.shape}"
                )
            activation_difference = activation_a - activation_b
            archived = activation_difference.astype(np.float16)
            if not np.all(np.isfinite(archived)):
                raise RuntimeError("activation archive contains non-finite values")
            target_xyz = positions[subjects[0]] - positions[subjects[1]]
            controller_target_xyz = np.asarray(
                [target_xyz[0], target_xyz[1], 0.0], dtype=np.float64
            )
            bddl_target_xy = np.asarray(row["bddl_target_difference_xy"], dtype=np.float64)
            if "simulator_target_difference_xy" in row:
                target_reference = "simulator_fixed_state"
                reference_target_xy = np.asarray(
                    row["simulator_target_difference_xy"], dtype=np.float64
                )
                tolerance = float(
                    config["stimulus"][
                        "maximum_manifest_to_simulator_xy_error_meters"
                    ]
                )
            else:
                target_reference = "bddl_region_center"
                reference_target_xy = bddl_target_xy
                tolerance = float(
                    config["stimulus"]["maximum_bddl_to_simulator_xy_error_meters"]
                )
            if float(np.linalg.norm(target_xyz[:2] - reference_target_xy)) > tolerance:
                raise RuntimeError(
                    f"simulator target positions disagree with {target_reference} preflight"
                )
            metadata = {
                "study": config["study"],
                "manifest_sha256": manifest["manifest_sha256"],
                "config_sha256": _sha256(args.config),
                "runtime_receipt_sha256": _sha256(args.runtime_receipt),
                "stimulus_id": row["stimulus_id"],
                "family_id": row["family_id"],
                "source_task_id": int(row["source_task_id"]),
                "difficulty_level": int(row["difficulty_level"]),
                "level": int(row["level"]),
                "sample": int(row["sample"]),
                "split": row["split"],
                "noise_seed": int(row["noise_seed"]),
                "prompt_a_id": row["task_a"]["prompt_id"],
                "prompt_b_id": row["task_b"]["prompt_id"],
                "subject_a": subjects[0],
                "subject_b": subjects[1],
                "pathway": config["site"]["pathway"],
                "layer": layer,
                "activation_shape": list(activation_difference.shape),
                "archive_dtype": "float16",
                "target_reference": target_reference,
                "pair_infer_ms": float(response["pair_infer_ms"]),
            }
            actions_a = np.asarray(response["actions_a"], dtype=np.float32)
            actions_b = np.asarray(response["actions_b"], dtype=np.float32)
            _atomic_npz(
                destination,
                {
                    "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
                    "activation_difference": archived,
                    "target_difference_xy": target_xyz[:2].astype(np.float64),
                    "target_difference_xyz": controller_target_xyz,
                    "bddl_target_difference_xy": bddl_target_xy,
                    "reference_target_difference_xy": reference_target_xy,
                    "action_difference": (actions_a - actions_b).astype(np.float32),
                    "actions_a": actions_a,
                    "actions_b": actions_b,
                },
            )
            completed += 1
            print(
                json.dumps(
                    {
                        "event": "layout_observer_pair_complete",
                        "stimulus_id": row["stimulus_id"],
                        "completed": completed,
                    }
                ),
                flush=True,
            )
        finally:
            env.close()

    summary = {
        "study": config["study"],
        "manifest_sha256": manifest["manifest_sha256"],
        "planned": len(rows),
        "completed_or_preexisting": completed,
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if completed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
