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


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--libero-plus-root", type=Path, required=True)
    parser.add_argument("--libero-cf-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--environment-seed", type=int, default=7)
    args = parser.parse_args()

    project_src = Path(__file__).parents[1] / "src"
    sys.path.insert(0, str(project_src))
    from intended_futures.provenance import git_commit
    from intended_futures.stimuli import validate_subject_positions

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if config.get("status") != "frozen":
        raise RuntimeError("geometry measurement requires a frozen source design")
    if manifest.get("config_sha256") != _sha256(args.config):
        raise RuntimeError("manifest is not bound to this config")
    if git_commit(args.libero_plus_root) != config["upstream"]["libero_plus_commit"]:
        raise RuntimeError("LIBERO-Plus commit differs from the frozen source design")
    parent_receipt = json.loads(
        (Path(__file__).parents[1] / "results" / "target_control_tc1" / "runtime_receipt.json")
        .read_text(encoding="utf-8")
    )
    if git_commit(args.libero_cf_root) != parent_receipt["libero_cf_commit"]:
        raise RuntimeError("LIBERO-CF commit differs from the validated parent runtime")
    renderer = config["runtime_contract"]["simulator_renderer"]
    if os.environ.get("MUJOCO_GL") != renderer or os.environ.get("PYOPENGL_PLATFORM") != renderer:
        raise RuntimeError(f"renderer environment must be fixed to {renderer!r}")

    sys.path.insert(0, str(args.libero_cf_root))
    import torch
    from libero.libero.envs import OffScreenRenderEnv

    rows = manifest["rows"]
    if len(rows) != int(config["population"]["expected_manifest_rows"]):
        raise RuntimeError("source manifest row count differs from the frozen design")
    bddl_root = args.libero_plus_root / "libero" / "libero" / "bddl_files"
    init_root = args.libero_plus_root / "libero" / "libero" / "init_files"
    contract = config["stimulus"]["workspace_position_contract"]
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    for index, row in enumerate(rows, start=1):
        destination = args.output / f"{row['stimulus_id']}.json"
        if destination.exists():
            existing = json.loads(destination.read_text(encoding="utf-8"))
            if (
                existing.get("study") != config["study"]
                or existing.get("manifest_sha256") != manifest["manifest_sha256"]
                or existing.get("stimulus_id") != row["stimulus_id"]
            ):
                raise RuntimeError(f"preexisting geometry identity mismatch: {destination}")
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
            states = np.asarray(torch.load(init_file), dtype=np.float64).reshape(1, -1)
            env.reset()
            obs = env.set_init_state(states[0])
            for _ in range(10):
                obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
            subjects = [
                str(row["task_a"]["intended_subject"]),
                str(row["task_b"]["intended_subject"]),
            ]
            positions = validate_subject_positions(obs, subjects, contract)
            target_a = np.asarray(positions[subjects[0]], dtype=np.float64)
            target_b = np.asarray(positions[subjects[1]], dtype=np.float64)
            difference = target_a - target_b
            bddl_difference = np.asarray(
                row["bddl_target_difference_xy"], dtype=np.float64
            )
            payload = {
                "study": config["study"],
                "manifest_sha256": manifest["manifest_sha256"],
                "stimulus_id": row["stimulus_id"],
                "family_id": row["family_id"],
                "split": row["split"],
                "level": int(row["level"]),
                "sample": int(row["sample"]),
                "bddl_sha256": row["bddl_sha256"],
                "init_sha256": row["init_sha256"],
                "simulator_target_xyz_a": target_a.tolist(),
                "simulator_target_xyz_b": target_b.tolist(),
                "simulator_target_difference_xy": difference[:2].tolist(),
                "simulator_target_separation_meters": float(
                    np.linalg.norm(difference[:2])
                ),
                "bddl_target_difference_xy": bddl_difference.tolist(),
                "bddl_to_simulator_difference_error_meters": float(
                    np.linalg.norm(difference[:2] - bddl_difference)
                ),
                "model_outcomes_observed": False,
            }
            _atomic_json(destination, payload)
        finally:
            env.close()
        print(
            json.dumps(
                {
                    "event": "layout_geometry_measured",
                    "stimulus_id": row["stimulus_id"],
                    "completed": index,
                }
            ),
            flush=True,
        )

    records = []
    for row in rows:
        path = args.output / f"{row['stimulus_id']}.json"
        records.append(json.loads(path.read_text(encoding="utf-8")))
    summary = {
        "study": config["study"],
        "source_manifest_sha256": manifest["manifest_sha256"],
        "records": len(records),
        "maximum_bddl_to_simulator_difference_error_meters": max(
            float(record["bddl_to_simulator_difference_error_meters"])
            for record in records
        ),
        "minimum_simulator_target_separation_meters": min(
            float(record["simulator_target_separation_meters"]) for record in records
        ),
        "model_outcomes_observed": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(args.summary, summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
