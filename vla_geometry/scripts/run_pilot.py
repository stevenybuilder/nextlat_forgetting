#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vima-root", type=Path, required=True)
    parser.add_argument("--vimabench-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-validation", type=Path)
    parser.add_argument(
        "--mode", choices=("smoke", "representation", "behavior"), required=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit-cells", type=int)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    project_src = Path(__file__).parents[1] / "src"
    sys.path.insert(0, str(project_src))
    sys.path.insert(0, str(args.vima_root))
    sys.path.insert(0, str(args.vimabench_root))

    from vla_geometry.grid import build_cells, get_factor_order, load_config
    from vla_geometry.io import (
        atomic_write_json,
        atomic_write_npz,
        read_npz_record,
        runtime_provenance,
    )
    from vla_geometry.runner import load_policy, run_episode
    from vla_geometry.seeds import resolve_all_seed_maps

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(0)
    np.random.seed(0)

    config = load_config(args.config)
    if config.get("status") != "frozen":
        raise RuntimeError("refusing to run a non-frozen scientific configuration")
    contract = config["runtime_contract"]
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(contract["cublas_workspace_config"])
    all_cells = build_cells(config["factors"], get_factor_order(config))
    seed_maps = resolve_all_seed_maps(config, all_cells)
    cells = all_cells
    if args.mode == "smoke":
        cells = [cells[0], cells[-1]]
        closed_loop = True
    elif args.mode == "representation":
        closed_loop = False
    else:
        closed_loop = True
    if args.limit_cells is not None:
        cells = cells[: args.limit_cells]

    args.output.mkdir(parents=True, exist_ok=True)
    seed_manifest_path = args.output / "seed_manifest.json"
    if seed_manifest_path.exists():
        existing_manifest = json.loads(seed_manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != seed_maps:
            raise RuntimeError("existing seed manifest differs from frozen configuration")
    else:
        atomic_write_json(seed_manifest_path, seed_maps)
    provenance_path = args.output / "provenance.json"
    provenance = runtime_provenance(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        vima_root=args.vima_root,
        vimabench_root=args.vimabench_root,
        project_root=Path(__file__).parents[1],
    )
    gpu = provenance["gpu"]
    if gpu is None or int(gpu["count"]) != int(contract["num_gpus"]):
        raise RuntimeError(
            f"visible GPU count {None if gpu is None else gpu['count']} "
            f"!= frozen {contract['num_gpus']}"
        )
    if bool(contract["distributed"]):
        raise RuntimeError("this runner does not support a distributed runtime")
    if os.environ.get("WORLD_SIZE") not in (None, "1"):
        raise RuntimeError(f"unexpected WORLD_SIZE={os.environ['WORLD_SIZE']}")
    if args.device != "cuda:0":
        raise RuntimeError(f"device {args.device!r} violates the frozen cuda:0 contract")
    expected_runtime = contract["expected_runtime"]
    runtime_checks = {
        "python": provenance["python"],
        "torch": provenance["torch"],
        "cuda_runtime": provenance["cuda_runtime"],
        "cudnn": provenance["cudnn"],
        "gpu_name": gpu["name"],
        "numpy": provenance["numpy"],
        "transformers": provenance["transformers"],
        "gym": provenance["gym"],
    }
    mismatches = {
        key: {"actual": runtime_checks[key], "expected": expected}
        for key, expected in expected_runtime.items()
        if str(runtime_checks[key]) != str(expected)
    }
    if mismatches:
        raise RuntimeError(f"runtime contract mismatch: {mismatches}")
    expected = config["model"]["checkpoint_expected_bytes"]
    if provenance["checkpoint_bytes"] != expected:
        raise RuntimeError(
            f"checkpoint bytes {provenance['checkpoint_bytes']} != frozen {expected}"
        )
    expected_sha256 = config["model"]["checkpoint_sha256"]
    if provenance["checkpoint_sha256"] != expected_sha256:
        raise RuntimeError(
            "checkpoint SHA-256 "
            f"{provenance['checkpoint_sha256']} != frozen {expected_sha256}"
        )
    for key in ("vima_commit", "vimabench_commit"):
        frozen = config["upstream"][key]
        if provenance[key] != frozen:
            raise RuntimeError(f"{key}={provenance[key]} != frozen {frozen}")
    if not provenance["vima_worktree_clean"] or not provenance["vimabench_worktree_clean"]:
        raise RuntimeError("pinned upstream worktree contains uncommitted changes")
    if config["benchmark"]["task"] == "manipulate_old_neighbor":
        if args.adapter_validation is None or not args.adapter_validation.exists():
            raise RuntimeError("Task 16 requires --adapter-validation from the frozen source")
        adapter_validation = json.loads(
            args.adapter_validation.read_text(encoding="utf-8")
        )
        required_adapter_values = {
            "passed": True,
            "policy_loaded": False,
            "actions_executed": False,
            "rewards_or_outcomes_observed": False,
            "config_sha256": provenance["config_sha256"],
            "source_tree_sha256": provenance["source_tree_sha256"],
            "cells_checked": len(all_cells),
        }
        adapter_mismatches = {
            key: {"actual": adapter_validation.get(key), "expected": expected}
            for key, expected in required_adapter_values.items()
            if adapter_validation.get(key) != expected
        }
        if adapter_mismatches:
            raise RuntimeError(f"adapter validation mismatch: {adapter_mismatches}")
        provenance["adapter_validation"] = required_adapter_values
    if provenance_path.exists():
        existing_provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if existing_provenance != provenance:
            raise RuntimeError("existing run provenance differs; use a new output directory")
    else:
        atomic_write_json(provenance_path, provenance)

    import torch

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)

    policy = load_policy(args.checkpoint, args.device)
    parameter_dtypes = {str(parameter.dtype) for parameter in policy.parameters()}
    if contract["precision"] == "float32" and parameter_dtypes != {"torch.float32"}:
        raise RuntimeError(f"policy dtype mismatch: {sorted(parameter_dtypes)}")
    completed = 0
    failed = 0
    started = time.monotonic()
    mode_dir = args.output / args.mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        seeds = seed_maps[args.mode][cell["cell_id"]]
        for seed in seeds:
            suffix = "npz" if not closed_loop else "json"
            destination = mode_dir / f"{cell['cell_id']}-seed{seed}.{suffix}"
            failure_path = mode_dir / f"{cell['cell_id']}-seed{seed}.failure.json"
            if destination.exists() and failure_path.exists():
                raise RuntimeError(f"both success and failure records exist for {destination}")
            if destination.exists():
                if closed_loop:
                    existing_metadata = json.loads(destination.read_text(encoding="utf-8"))
                else:
                    existing_activation, existing_metadata = read_npz_record(destination)
                    if not np.all(np.isfinite(existing_activation)):
                        raise RuntimeError(f"non-finite preexisting activation: {destination}")
                expected_metadata = {
                    "cell_id": cell["cell_id"],
                    "seed": int(seed),
                    "mode": args.mode,
                    "closed_loop": closed_loop,
                    "valid": True,
                }
                for factor in get_factor_order(config):
                    expected_metadata[factor] = str(cell[factor])
                metadata_mismatches = {
                    key: {"actual": existing_metadata.get(key), "expected": expected}
                    for key, expected in expected_metadata.items()
                    if existing_metadata.get(key) != expected
                }
                if metadata_mismatches:
                    raise RuntimeError(
                        f"preexisting record mismatch in {destination}: {metadata_mismatches}"
                    )
                completed += 1
                continue
            if failure_path.exists():
                failed += 1
                continue
            try:
                result = run_episode(
                    policy=policy,
                    config=config,
                    cell=cell,
                    seed=int(seed),
                    device=args.device,
                    vima_root=args.vima_root,
                    closed_loop=closed_loop,
                )
                result.metadata["mode"] = args.mode
                result.metadata["elapsed_wall_seconds"] = time.monotonic() - started
                if closed_loop:
                    atomic_write_json(destination, result.metadata)
                else:
                    if result.activation is None:
                        raise AssertionError("representation run returned no activation")
                    atomic_write_npz(
                        destination,
                        activation=result.activation,
                        metadata=result.metadata,
                    )
                completed += 1
                print(
                    json.dumps(
                        {
                            "event": "episode_complete",
                            "mode": args.mode,
                            "cell_id": cell["cell_id"],
                            "seed": seed,
                            "success": result.metadata["success"],
                            "completed": completed,
                            "failed": failed,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            except Exception as error:
                failed += 1
                failure = {
                    "cell_id": cell["cell_id"],
                    "seed": int(seed),
                    "mode": args.mode,
                    "valid": False,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
                atomic_write_json(
                    failure_path, failure
                )
                print(json.dumps({"event": "episode_failure", **failure}), flush=True)
    summary = {
        "mode": args.mode,
        "planned": sum(
            len(seed_maps[args.mode][cell["cell_id"]])
            for cell in cells
        ),
        "completed_or_preexisting": completed,
        "failed": failed,
        "wall_seconds": time.monotonic() - started,
    }
    atomic_write_json(args.output / f"{args.mode}_summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
