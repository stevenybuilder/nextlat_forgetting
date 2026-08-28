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
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--libero-cf-root", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project_root = Path(__file__).parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from intended_futures.config import load_config
    from intended_futures.manifest import sha256_file
    from intended_futures.provenance import checkpoint_tree, git_commit, sha256_source_tree

    config = load_config(args.config)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["study"] != config["study"]:
        raise ValueError("manifest and configuration study IDs differ")
    if manifest["manifest_sha256"] != config.get("manifest_sha256", manifest["manifest_sha256"]):
        raise ValueError("manifest hash differs from frozen configuration")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != config["runtime_contract"]["cuda_visible_devices"]:
        raise RuntimeError("CUDA_VISIBLE_DEVICES violates the frozen runtime contract")

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("runtime receipt must be written from the OpenPI environment") from error
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("runtime receipt requires exactly one visible CUDA GPU")
    gpu_name = torch.cuda.get_device_name(0)
    if gpu_name != config["runtime_contract"]["primary_gpu"]:
        raise RuntimeError(f"GPU {gpu_name!r} != frozen {config['runtime_contract']['primary_gpu']!r}")

    receipt = {
        "study": config["study"],
        "written_before_pilot_outcomes": True,
        "config_sha256": sha256_file(args.config),
        "manifest_file_sha256": sha256_file(args.manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "source_tree_sha256": sha256_source_tree(project_root),
        "checkpoint": checkpoint_tree(args.checkpoint),
        "libero_cf_commit": git_commit(args.libero_cf_root),
        "openpi_commit": git_commit(args.openpi_root),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": gpu_name,
        "gpu_count": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "simulator_renderer": config["runtime_contract"]["simulator_renderer"],
        "distributed": False,
    }
    if receipt["libero_cf_commit"] != config["upstream"]["libero_cf_commit"]:
        raise RuntimeError("LIBERO-CF commit violates the frozen configuration")
    if receipt["openpi_commit"] != config["upstream"]["openpi_reference_commit"]:
        raise RuntimeError("OpenPI commit violates the frozen configuration")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in ("study", "config_sha256", "manifest_sha256", "source_tree_sha256", "gpu_name")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
