#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--parent-runtime-receipt", type=Path, required=True)
    parser.add_argument("--libero-plus-root", type=Path, required=True)
    parser.add_argument("--libero-cf-root", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--simulator-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project_root = Path(__file__).parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from intended_futures.manifest import sha256_file
    from intended_futures.provenance import checkpoint_tree, git_commit, sha256_source_tree

    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    parent = json.loads(args.parent_runtime_receipt.read_text(encoding="utf-8"))
    if config.get("status") != "frozen":
        raise RuntimeError("runtime receipt requires a frozen config")
    config_sha = sha256_file(args.config)
    if manifest.get("config_sha256") != config_sha:
        raise RuntimeError("manifest is not bound to the frozen config")
    if (
        preflight.get("all_checks_passed") is not True
        or preflight.get("model_outcomes_observed") is not False
        or preflight.get("manifest_sha256") != manifest.get("manifest_sha256")
        or int(preflight.get("rows", -1))
        != int(config["population"]["expected_manifest_rows"])
    ):
        raise RuntimeError("pre-model geometry preflight does not certify the population")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("receipt requires CUDA_VISIBLE_DEVICES=0")

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("receipt requires exactly one visible CUDA GPU")
    gpu_name = torch.cuda.get_device_name(0)
    if config["runtime_contract"]["gpu_class"] not in gpu_name:
        raise RuntimeError(f"GPU {gpu_name!r} violates the runtime contract")
    libero_plus_commit = git_commit(args.libero_plus_root)
    libero_cf_commit = git_commit(args.libero_cf_root)
    openpi_commit = git_commit(args.openpi_root)
    if libero_plus_commit != config["upstream"]["libero_plus_commit"]:
        raise RuntimeError("LIBERO-Plus commit violates the frozen config")
    if openpi_commit != config["upstream"]["openpi_reference_commit"]:
        raise RuntimeError("OpenPI commit violates the frozen config")
    if libero_cf_commit != parent["libero_cf_commit"]:
        raise RuntimeError("LIBERO-CF commit differs from the validated parent runtime")
    checkpoint = checkpoint_tree(args.checkpoint)
    if checkpoint["tree_sha256"] != parent["parent_checkpoint_tree_sha256"]:
        raise RuntimeError("checkpoint tree differs from the validated parent checkpoint")
    simulator_python = subprocess.run(
        [str(args.simulator_python), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    receipt = {
        "study": config["study"],
        "written_before_model_outcomes": True,
        "config_sha256": config_sha,
        "manifest_file_sha256": sha256_file(args.manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "preflight_file_sha256": sha256_file(args.preflight),
        "preflight_rows": int(preflight["rows"]),
        "preflight_pooled_layout_residual_rms_meters": float(
            preflight["pooled_layout_residual_rms_meters"]
        ),
        "parent_runtime_receipt_sha256": sha256_file(args.parent_runtime_receipt),
        "checkpoint": checkpoint,
        "source_tree_sha256": sha256_source_tree(project_root),
        "libero_plus_commit": libero_plus_commit,
        "libero_cf_commit": libero_cf_commit,
        "openpi_commit": openpi_commit,
        "policy_python": platform.python_version(),
        "simulator_python": simulator_python,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": gpu_name,
        "gpu_count": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "distributed": False,
        "simulator_renderer": config["runtime_contract"]["simulator_renderer"],
        "site": config["site"],
        "conditions": [condition["condition_id"] for condition in config["conditions"]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in ("study", "config_sha256", "source_tree_sha256", "gpu_name")
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
