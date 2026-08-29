#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
import socket
import sys

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-name", default="pi05_libero")
    parser.add_argument("--layers", default="5,11,17")
    parser.add_argument("--paligemma-layers", default="")
    parser.add_argument("--denoising-calls", type=int, default=10)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-controller", type=Path)
    args = parser.parse_args()

    project_src = Path(__file__).parents[1] / "src"
    sys.path.insert(0, str(project_src))
    from intended_futures.instrumented_policy import InstrumentedPairedPolicy
    from openpi.policies import policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as training_config

    if args.device != "cuda:0":
        raise RuntimeError("the frozen pilot requires cuda:0")
    layers = [int(value) for value in args.layers.split(",")]
    paligemma_layers = [
        int(value) for value in args.paligemma_layers.split(",") if value.strip()
    ]
    base_policy = policy_config.create_trained_policy(
        training_config.get_config(args.config_name),
        args.checkpoint,
        pytorch_device=args.device,
    )
    target_controller = None
    if args.target_controller is not None:
        with np.load(args.target_controller, allow_pickle=False) as artifact:
            target_controller = {
                "pathway": str(artifact["pathway"].item()),
                "layer": int(artifact["layer"].item()),
                "beta": np.asarray(artifact["beta"], dtype=np.float32),
                "inverse_ridge_fraction": float(artifact["inverse_ridge_fraction"].item()),
                "maximum_norm_fraction_of_full_donor_delta": float(
                    artifact["maximum_norm_fraction_of_full_donor_delta"].item()
                ),
                "artifact_sha256": hashlib.sha256(args.target_controller.read_bytes()).hexdigest(),
            }
    policy = InstrumentedPairedPolicy(
        base_policy,
        layer_indices=layers,
        expected_denoising_calls=args.denoising_calls,
        paligemma_layer_indices=paligemma_layers,
        target_controller=target_controller,
    )
    logging.info(
        "Serving instrumented policy on %s with expert layers %s and PaliGemma layers %s",
        socket.gethostname(),
        layers,
        paligemma_layers,
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    raise SystemExit(main())
