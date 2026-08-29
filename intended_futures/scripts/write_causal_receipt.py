#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-config", type=Path, required=True)
    parser.add_argument("--causal-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--subspace", type=Path, required=True)
    parser.add_argument("--parent-runtime-receipt", type=Path, required=True)
    parser.add_argument("--libero-cf-root", type=Path, required=True)
    parser.add_argument("--openpi-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project_root = Path(__file__).parents[1]
    sys.path.insert(0, str(project_root / "src"))
    from intended_futures.config import load_config
    from intended_futures.manifest import sha256_file
    from intended_futures.provenance import git_commit, sha256_source_tree

    pilot = load_config(args.pilot_config)
    causal = json.loads(args.causal_config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    parent_receipt = json.loads(args.parent_runtime_receipt.read_text(encoding="utf-8"))
    if (
        causal["parent_study"] != pilot["study"]
        or causal["parent_manifest_sha256"] != manifest["manifest_sha256"]
        or parent_receipt["study"] != pilot["study"]
    ):
        raise RuntimeError("causal inputs do not share the frozen parent study")
    if sha256_file(args.subspace) != causal["future_subspace_sha256"]:
        raise RuntimeError("causal subspace differs from the frozen hash")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != pilot["runtime_contract"]["cuda_visible_devices"]:
        raise RuntimeError("CUDA_VISIBLE_DEVICES violates the parent runtime contract")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("causal receipt requires exactly one visible CUDA GPU")
    gpu_name = torch.cuda.get_device_name(0)
    if gpu_name != pilot["runtime_contract"]["primary_gpu"]:
        raise RuntimeError("causal GPU differs from the parent runtime contract")
    receipt = {
        "study": causal["study"],
        "written_before_causal_outcomes": True,
        "parent_study": pilot["study"],
        "parent_manifest_sha256": manifest["manifest_sha256"],
        "pilot_config_sha256": sha256_file(args.pilot_config),
        "causal_config_sha256": sha256_file(args.causal_config),
        "subspace_sha256": sha256_file(args.subspace),
        "parent_runtime_receipt_sha256": sha256_file(args.parent_runtime_receipt),
        "source_tree_sha256": sha256_source_tree(project_root),
        "libero_cf_commit": git_commit(args.libero_cf_root),
        "openpi_commit": git_commit(args.openpi_root),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": gpu_name,
        "gpu_count": torch.cuda.device_count(),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "distributed": False,
        "selected_layer": causal["selected_layer"],
        "causal_units": causal["sampling"]["expected_causal_units"],
        "conditions": causal["rollout"]["conditions"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in ("study", "causal_config_sha256", "source_tree_sha256", "gpu_name")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
