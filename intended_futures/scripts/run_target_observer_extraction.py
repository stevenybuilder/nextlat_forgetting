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


def _atomic_npz(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    parser.add_argument("--pilot-config", type=Path, required=True)
    parser.add_argument("--tc1-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--libero-cf-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--environment-seed", type=int, default=7)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    project_src = Path(__file__).parents[1] / "src"
    sys.path.insert(0, str(project_src))
    from intended_futures.config import load_config
    from intended_futures.stimuli import validate_subject_positions

    pilot = load_config(args.pilot_config)
    protocol = json.loads(args.tc1_config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    runtime_receipt = json.loads(args.runtime_receipt.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen":
        raise RuntimeError("target-observer extraction requires a frozen TC1 protocol")
    if manifest["study"] != protocol["study"]:
        raise RuntimeError("TC1 manifest study differs from the protocol")
    if manifest["protocol_sha256"] != _sha256(args.tc1_config):
        raise RuntimeError("TC1 manifest is not bound to this protocol")
    if (
        runtime_receipt.get("study") != protocol["study"]
        or runtime_receipt.get("written_before_model_outcomes") is not True
        or runtime_receipt.get("tc1_config_sha256") != _sha256(args.tc1_config)
        or runtime_receipt.get("manifest_sha256") != manifest["manifest_sha256"]
        or runtime_receipt.get("gpu_count") != 1
        or runtime_receipt.get("distributed") is not False
    ):
        raise RuntimeError("runtime receipt does not certify this frozen observer run")
    renderer = protocol["runtime_contract"]["simulator_renderer"]
    if os.environ.get("MUJOCO_GL") != renderer or os.environ.get("PYOPENGL_PLATFORM") != renderer:
        raise RuntimeError(f"renderer environment must be fixed to {renderer!r}")

    sys.path.insert(0, str(args.libero_cf_root))
    from eval.main_cf import _get_libero_env, _quat2axisangle
    from libero.libero import benchmark
    from openpi_client import image_tools
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    rows = [
        row for row in manifest["rows"]
        if row["split"] in {"observer_fit", "observer_validation"}
    ]
    expected = (
        int(protocol["population"]["expected_observer_fit_units"])
        + int(protocol["population"]["expected_observer_validation_units"])
    )
    if len(rows) != expected:
        raise RuntimeError(f"observer extraction has {len(rows)} rows, expected {expected}")
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        rows = rows[: args.limit]
    suite = benchmark.get_benchmark_dict()[manifest["suite"]]()
    contract = pilot["benchmark"]["workspace_position_contract"]
    layer = int(protocol["site"]["layer"])
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
                metadata.get("study") != protocol["study"]
                or metadata.get("manifest_sha256") != manifest["manifest_sha256"]
                or metadata.get("split") != row["split"]
            ):
                raise RuntimeError(f"preexisting observer record identity mismatch: {destination}")
            completed += 1
            continue

        task_index = int(row["task_index"])
        task = suite.get_task(task_index)
        states = suite.get_task_init_states(task_index)
        env, _ = _get_libero_env(task, 256, args.environment_seed)
        try:
            env.reset()
            obs = env.set_init_state(states[int(row["initial_state_index"])])
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
            donor = np.asarray(response["paligemma_activations_a"][key], dtype=np.float32)
            recipient = np.asarray(response["paligemma_activations_b"][key], dtype=np.float32)
            if donor.shape != recipient.shape or donor.ndim != 2:
                raise RuntimeError(
                    f"PaliGemma activation shapes differ or are not token-by-hidden: "
                    f"{donor.shape}, {recipient.shape}"
                )
            difference = donor - recipient
            if not np.all(np.isfinite(difference)):
                raise RuntimeError("PaliGemma activation difference is non-finite")
            archived = difference.astype(np.float16)
            if not np.all(np.isfinite(archived)):
                raise RuntimeError("float16 activation archive overflowed")
            metadata = {
                "study": protocol["study"],
                "manifest_sha256": manifest["manifest_sha256"],
                "protocol_sha256": _sha256(args.tc1_config),
                "runtime_receipt_sha256": _sha256(args.runtime_receipt),
                "stimulus_id": row["stimulus_id"],
                "scene_id": row["scene_id"],
                "split": row["split"],
                "initial_state_index": int(row["initial_state_index"]),
                "noise_seed": int(row["noise_seed"]),
                "prompt_a_id": row["task_a"]["prompt_id"],
                "prompt_b_id": row["task_b"]["prompt_id"],
                "subject_a": subjects[0],
                "subject_b": subjects[1],
                "pathway": protocol["site"]["pathway"],
                "layer": layer,
                "activation_shape": list(difference.shape),
                "archive_dtype": "float16",
                "source_inference_dtype": protocol["runtime_contract"]["inference_dtype"],
                "pair_infer_ms": float(response["pair_infer_ms"]),
            }
            _atomic_npz(
                destination,
                {
                    "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
                    "activation_difference": archived,
                    "target_difference": (
                        positions[subjects[0]] - positions[subjects[1]]
                    ).astype(np.float64),
                    "actions_a": np.asarray(response["actions_a"], dtype=np.float32),
                    "actions_b": np.asarray(response["actions_b"], dtype=np.float32),
                },
            )
            completed += 1
            print(
                json.dumps(
                    {
                        "event": "observer_pair_complete",
                        "stimulus_id": row["stimulus_id"],
                        "completed": completed,
                        "activation_shape": list(difference.shape),
                    }
                ),
                flush=True,
            )
        finally:
            env.close()

    summary = {
        "study": protocol["study"],
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
