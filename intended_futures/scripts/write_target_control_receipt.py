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
    parser.add_argument("--m0-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parent-runtime-receipt", type=Path, required=True)
    parser.add_argument("--libero-cf-root", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--simulator-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project_root = Path(__file__).parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from intended_futures.manifest import sha256_file
    from intended_futures.provenance import git_commit, sha256_source_tree

    protocol = json.loads(args.m0_config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    parent = json.loads(args.parent_runtime_receipt.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen":
        raise RuntimeError("M0 protocol is not frozen")
    if (
        protocol["parent"]["manifest_sha256"] != manifest["manifest_sha256"]
        or parent["manifest_sha256"] != manifest["manifest_sha256"]
    ):
        raise RuntimeError("M0 inputs do not share the frozen parent manifest")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("M0 receipt requires CUDA_VISIBLE_DEVICES=0")

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("M0 receipt requires exactly one visible CUDA GPU")
    gpu_name = torch.cuda.get_device_name(0)
    if protocol["runtime_contract"]["gpu_class"] not in gpu_name:
        raise RuntimeError(f"GPU {gpu_name!r} violates the M0 runtime contract")
    libero_commit = git_commit(args.libero_cf_root)
    openpi_commit = git_commit(args.openpi_root)
    if libero_commit != protocol["upstream"]["libero_cf_commit"]:
        raise RuntimeError("LIBERO-CF commit violates the M0 protocol")
    if openpi_commit != protocol["upstream"]["openpi_reference_commit"]:
        raise RuntimeError("OpenPI commit violates the M0 protocol")
    simulator_python = subprocess.run(
        [str(args.simulator_python), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    receipt = {
        "study": protocol["study"],
        "written_before_m0_outcomes": True,
        "m0_config_sha256": sha256_file(args.m0_config),
        "manifest_file_sha256": sha256_file(args.manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "parent_runtime_receipt_sha256": sha256_file(args.parent_runtime_receipt),
        "parent_checkpoint_tree_sha256": parent["checkpoint"]["tree_sha256"],
        "source_tree_sha256": sha256_source_tree(project_root),
        "libero_cf_commit": libero_commit,
        "openpi_commit": openpi_commit,
        "policy_python": platform.python_version(),
        "simulator_python": simulator_python,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": gpu_name,
        "gpu_count": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "distributed": False,
        "simulator_renderer": protocol["runtime_contract"]["simulator_renderer"],
        "expected_units": protocol["population"]["expected_units"],
        "sites": protocol["sites"],
        "conditions": [
            condition["condition_id"] for condition in protocol["conditions"]
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in ("study", "m0_config_sha256", "source_tree_sha256", "gpu_name")
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
