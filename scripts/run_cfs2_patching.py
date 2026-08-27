#!/usr/bin/env python3
"""Run CFS-2 parent-to-adapted activation patching for one completed branch.

This is inference-only.  It writes per-probe baseline margins plus matching-parent,
unrelated-anchor, and norm-matched random-control effects for every requested layer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import pathlib
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import yaml


REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cfs2 import patching as P  # noqa: E402
from lurestar import representations as R  # noqa: E402
from lurestar.validate import solve_line, token_ids  # noqa: E402


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(left)
    for key, item in right.items():
        if isinstance(item, Mapping) and isinstance(value.get(key), Mapping):
            value[key] = merge(value[key], item)
        else:
            value[key] = item
    return value


def load_retention(path: pathlib.Path) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    ids, lines, correct, competitors = [], [], [], []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise P.CFS2PatchingError(f"retention manifest line {number} is invalid") from exc
        if row.get("schema") != "nextlat_forgetting/cfs2_retention_manifest/1":
            raise P.CFS2PatchingError("retention manifest contains a non-CFS-2 row")
        line = str(row["line"])
        solved = solve_line(line)
        ids.append(str(row["probe_id"]))
        lines.append(line)
        correct.append(int(solved.path[1]))
        competitors.append([int(arm[0]) for arm in solved.arms[1:]])
    if not lines or len(set(ids)) != len(ids):
        raise P.CFS2PatchingError("retention probes must be nonempty and uniquely identified")
    tokens = np.asarray([token_ids(line, eos=True) for line in lines], dtype=np.int64)
    if tokens.ndim != 2 or tokens.shape[1] != 69:
        raise P.CFS2PatchingError(f"CFS-2 G(5,5) tokens must be N x 69, got {tokens.shape}")
    return tokens, ids, np.asarray(correct, dtype=np.int64), np.asarray(competitors, dtype=np.int64)


def load_nextlat_model(
    *, upstream: pathlib.Path, config_path: pathlib.Path, checkpoint: pathlib.Path, device: str
):
    torch = importlib.import_module("torch")
    upstream_text = str(upstream)
    if upstream_text not in sys.path:
        sys.path.insert(0, upstream_text)
    nextlat = importlib.import_module("models.model_nextlat")
    defaults = yaml.safe_load((upstream / "defaults.yaml").read_text(encoding="utf-8"))
    body = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(defaults, Mapping) or not isinstance(body, Mapping):
        raise P.CFS2PatchingError("NextLat defaults/config must be mappings")
    config = merge(defaults, body)
    model, trainer = config["model"], config["trainer"]
    args = {
        "n_layer": int(model["n_layer"]), "n_head": int(model["n_head"]),
        "n_embd": int(model["n_embd"]), "dropout": float(model["dropout"]),
        "bias": bool(model["bias"]), "vocab_size": 106, "block_size": 69,
        "context_length": 62, "eos_token_id": 104,
        "use_fused": bool(trainer.get("use_fused_kernels", False)),
        "mtp_horizon": int(model["mtp_horizon"]), "lambda_kl": float(model["lambda_kl"]),
        "lambda_mse": float(model["lambda_mse"]), "lambda_ce": float(model["lambda_ce"]),
        "proj_factor": float(model["proj_factor"]),
        "compute_hidden_state_rank": bool(model.get("compute_hidden_state_rank", False)),
    }
    wrapper = nextlat.NextLat(nextlat.NextLatConfig(**args))
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping) or not isinstance(raw.get("model"), Mapping):
        raise P.CFS2PatchingError(f"checkpoint lacks a model state dictionary: {checkpoint}")
    try:
        wrapper.model.load_state_dict(raw["model"], strict=True)
    except RuntimeError as exc:
        raise P.CFS2PatchingError(f"checkpoint is incompatible with CFS-2 NextLat: {checkpoint}") from exc
    return wrapper.model.to(device).eval()


def capture_dataset(model, tokens: np.ndarray, *, layers: Sequence[int], batch_size: int, device: str):
    torch = importlib.import_module("torch")
    states = {layer: [] for layer in layers}
    logits = []
    for start in range(0, tokens.shape[0], batch_size):
        batch = torch.as_tensor(tokens[start:start + batch_size], dtype=torch.long, device=device)
        captured = P.capture_states_and_logits(model, batch, layers=layers)
        for layer in layers:
            states[layer].append(P.tensor_to_numpy(captured.states[layer]))
        logits.append(P.tensor_to_numpy(captured.logits))
    return {layer: np.concatenate(parts) for layer, parts in states.items()}, np.concatenate(logits)


def patched_dataset_logits(
    model,
    tokens: np.ndarray,
    replacement: np.ndarray,
    *,
    layer: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    torch = importlib.import_module("torch")
    parts = []
    for start in range(0, tokens.shape[0], batch_size):
        stop = min(tokens.shape[0], start + batch_size)
        batch = torch.as_tensor(tokens[start:stop], dtype=torch.long, device=device)
        value = torch.as_tensor(replacement[start:stop], device=device)
        parts.append(P.tensor_to_numpy(P.forward_with_patch(model, batch, value, layer=layer)))
    return np.concatenate(parts)


def margins(logits: np.ndarray, correct: np.ndarray, competitors: np.ndarray) -> np.ndarray:
    return np.asarray(R.branch_margin(logits, correct, competitors), dtype=np.float32)


def atomic_npz(path: pathlib.Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run(args: argparse.Namespace) -> dict[str, Any]:
    upstream = pathlib.Path(args.upstream_root).resolve()
    config = pathlib.Path(args.config).resolve()
    parent_checkpoint = pathlib.Path(args.parent_checkpoint).resolve()
    adapted_checkpoint = pathlib.Path(args.adapted_checkpoint).resolve()
    retention = pathlib.Path(args.retention_manifest).resolve()
    output = pathlib.Path(args.output).resolve()
    layers = tuple(int(value) for value in args.layers.split(","))
    token_matrix, probe_ids, correct, competitors = load_retention(retention)

    parent = load_nextlat_model(upstream=upstream, config_path=config,
                                checkpoint=parent_checkpoint, device=args.device)
    parent_states, _parent_logits = capture_dataset(
        parent, token_matrix, layers=layers, batch_size=args.batch_size, device=args.device
    )
    del parent
    torch = importlib.import_module("torch")
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()

    adapted = load_nextlat_model(upstream=upstream, config_path=config,
                                 checkpoint=adapted_checkpoint, device=args.device)
    adapted_states, baseline_logits = capture_dataset(
        adapted, token_matrix, layers=layers, batch_size=args.batch_size, device=args.device
    )
    baseline_margin = margins(baseline_logits, correct, competitors)
    unrelated = P.fixed_derangement(len(probe_ids), args.analysis_seed)
    arrays: dict[str, Any] = {
        "schema": np.asarray("nextlat_forgetting/cfs2_activation_patching/1"),
        "branch_id": np.asarray(args.branch_id),
        "probe_ids": np.asarray(probe_ids, dtype="U32"),
        "patch_position": np.asarray(P.PATCH_POSITION, dtype=np.int64),
        "patch_layers": np.asarray(layers, dtype=np.int64),
        "analysis_seed": np.asarray(args.analysis_seed, dtype=np.int64),
        "baseline_margin": baseline_margin,
        "parent_checkpoint_sha256": np.asarray(sha256_file(parent_checkpoint)),
        "adapted_checkpoint_sha256": np.asarray(sha256_file(adapted_checkpoint)),
        "retention_manifest_sha256": np.asarray(sha256_file(retention)),
    }

    # An adapted-state self patch must be numerically identical to the unpatched pass.
    first_stop = min(args.batch_size, token_matrix.shape[0])
    first_tokens = torch.as_tensor(token_matrix[:first_stop], dtype=torch.long, device=args.device)
    for layer in layers:
        self_patch = P.forward_with_patch(
            adapted, first_tokens,
            torch.as_tensor(adapted_states[layer][:first_stop], device=args.device), layer=layer,
        )
        if not torch.allclose(
            self_patch.float().cpu(), torch.as_tensor(baseline_logits[:first_stop]),
            atol=args.parity_atol, rtol=args.parity_rtol,
        ):
            raise P.CFS2PatchingError(f"adapted-state no-op parity failed at layer {layer}")

        controls = {
            "parent_state": parent_states[layer],
            "unrelated_anchor": parent_states[layer][unrelated],
            "norm_matched_random_subspace": P.norm_matched_random_replacement(
                adapted_states[layer], parent_states[layer],
                seed=args.analysis_seed + 1009 * (layer + 1),
            ),
        }
        for name, replacement in controls.items():
            patched_logits = patched_dataset_logits(
                adapted, token_matrix, replacement, layer=layer,
                batch_size=args.batch_size, device=args.device,
            )
            patched_margin = margins(patched_logits, correct, competitors)
            arrays[f"layer_{layer}_{name}_margin"] = patched_margin
            arrays[f"layer_{layer}_patch_{name}_effect"] = patched_margin - baseline_margin

    atomic_npz(output, arrays)
    return {
        "status": "COMPLETE_WITH_NAMED_CONTROLS",
        "branch_id": args.branch_id,
        "output": str(output),
        "sha256": sha256_file(output),
        "n_probes": len(probe_ids),
        "layers": list(layers),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--branch-id", required=True)
    parser.add_argument("--upstream-root", default=str(REPO / "upstream/NextLat"))
    parser.add_argument("--config", default=str(REPO / "configs/cfs2_nextlat_adapt.yaml"))
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--adapted-checkpoint", required=True)
    parser.add_argument("--retention-manifest", default=str(REPO / "manifests/cfs2/retention.jsonl"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--layers", default=",".join(map(str, P.DEFAULT_PATCH_LAYERS)))
    parser.add_argument("--analysis-seed", type=int, default=20260824)
    parser.add_argument("--parity-atol", type=float, default=1e-5)
    parser.add_argument("--parity-rtol", type=float, default=1e-5)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    try:
        result = run(args)
    except (OSError, ValueError, P.CFS2PatchingError) as exc:
        print(f"[run_cfs2_patching] FAILED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

