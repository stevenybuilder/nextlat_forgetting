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
    parser.add_argument("--tc1-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
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

    protocol = json.loads(args.tc1_config.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    parent = json.loads(args.parent_runtime_receipt.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen":
        raise RuntimeError("TC1 runtime receipt requires a frozen protocol")
    protocol_sha = sha256_file(args.tc1_config)
    if manifest.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("TC1 manifest is not bound to the frozen protocol")
    if (
        preflight.get("passed") is not True
        or preflight.get("model_outcomes_observed") is not False
        or preflight.get("manifest_sha256") != manifest.get("manifest_sha256")
    ):
        raise RuntimeError("simulator-only preflight does not certify this manifest")
    if int(preflight.get("states_validated", -1)) != int(
        protocol["population"]["expected_manifest_rows"]
    ):
        raise RuntimeError("preflight did not validate the full frozen population")
    minimum_separation = float(preflight.get("minimum_pair_distance_meters", -1.0))
    if minimum_separation < float(protocol["population"]["minimum_pair_distance_meters"]):
        raise RuntimeError("preflight violates the frozen target-separation contract")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("TC1 receipt requires CUDA_VISIBLE_DEVICES=0")

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("TC1 receipt requires exactly one visible CUDA GPU")
    gpu_name = torch.cuda.get_device_name(0)
    if protocol["runtime_contract"]["gpu_class"] not in gpu_name:
        raise RuntimeError(f"GPU {gpu_name!r} violates the TC1 runtime contract")
    libero_commit = git_commit(args.libero_cf_root)
    openpi_commit = git_commit(args.openpi_root)
    if libero_commit != protocol["upstream"]["libero_cf_commit"]:
        raise RuntimeError("LIBERO-CF commit violates the TC1 protocol")
    if openpi_commit != protocol["upstream"]["openpi_reference_commit"]:
        raise RuntimeError("OpenPI commit violates the TC1 protocol")
    simulator_python = subprocess.run(
        [str(args.simulator_python), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    receipt = {
        "study": protocol["study"],
        "written_before_model_outcomes": True,
        "tc1_config_sha256": protocol_sha,
        "manifest_file_sha256": sha256_file(args.manifest),
        "manifest_sha256": manifest["manifest_sha256"],
        "preflight_file_sha256": sha256_file(args.preflight),
        "preflight_states_validated": int(preflight["states_validated"]),
        "preflight_minimum_pair_distance_meters": minimum_separation,
        "parent_runtime_receipt_sha256": sha256_file(args.parent_runtime_receipt),
        "parent_checkpoint_tree_sha256": parent["parent_checkpoint_tree_sha256"],
        "source_tree_sha256": sha256_source_tree(project_root),
        "libero_cf_commit": libero_commit,
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
        "simulator_renderer": protocol["runtime_contract"]["simulator_renderer"],
        "site": protocol["site"],
        "conditions": [condition["condition_id"] for condition in protocol["conditions"]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in ("study", "tc1_config_sha256", "source_tree_sha256", "gpu_name")
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
