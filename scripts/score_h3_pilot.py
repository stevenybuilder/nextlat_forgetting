#!/usr/bin/env python3
"""Restartably score frozen H3 candidates with the sole non-confirmatory BST pilot.

The job is an explicit JSON object binding the exact local checkpoint/config/tokenizer, the
generation receipt, every input manifest, and this scorer.  No path is discovered from a runs or
results directory.  Completed chunks are atomically published and verified before reuse.
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
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from lurestar import h3_precompute as H  # noqa: E402
from lurestar.validate import token_ids, validate_line  # noqa: E402


JOB_SCHEMA = "nextlat_forgetting/h3_pilot_score_job/1"
CHUNK_SCHEMA = "nextlat_forgetting/h3_pilot_score_chunk/1"


def _record(value: Any, base: pathlib.Path, label: str) -> tuple[pathlib.Path, str]:
    if not isinstance(value, Mapping):
        raise H.PrecomputeRefused(f"{label} must be a path/SHA-256 record")
    path_value, expected = value.get("path"), value.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected, str) or len(expected) != 64:
        raise H.PrecomputeRefused(f"{label} has an invalid path/hash")
    path = pathlib.Path(path_value)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file() or H.sha256_file(path) != expected:
        raise H.PrecomputeRefused(f"{label} does not match its frozen SHA-256")
    return path, expected


def _load_job(path: pathlib.Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != JOB_SCHEMA:
        raise H.PrecomputeRefused(f"score job schema must be {JOB_SCHEMA}")
    if payload.get("role") != "non_confirmatory_engineering_pilot":
        raise H.PrecomputeRefused("score job is not explicitly non-confirmatory")
    if payload.get("confirmatory_inputs") is not False or payload.get("confirmatory_results") is not False:
        raise H.PrecomputeRefused("confirmatory inputs/results are forbidden")
    if payload.get("model_family") != "bst" or payload.get("seed") != H.PILOT_SEED or payload.get("training_step") != H.PILOT_STEP:
        raise H.PrecomputeRefused("score job attempts to substitute a different pilot")
    base = path.parent
    checkpoint, checkpoint_sha = _record(payload.get("checkpoint"), base, "pilot checkpoint")
    config, config_sha = _record(payload.get("config"), base, "pilot config")
    tokenizer, tokenizer_sha = _record(payload.get("tokenizer"), base, "upstream tokenizer")
    freeze, freeze_sha = _record(payload.get("pilot_freeze"), base, "pilot freeze")
    scorer, scorer_sha = _record(payload.get("scorer"), base, "pilot scorer")
    adaptation, adaptation_sha = _record(payload.get("adaptation_contract"), base, "adaptation contract")
    if checkpoint_sha != H.PILOT_CHECKPOINT_SHA256 or config_sha != H.PILOT_CONFIG_SHA256:
        raise H.PrecomputeRefused("job checkpoint/config does not match the sole frozen pilot")
    if tokenizer != (ROOT / "upstream/NextLat/data/stargraph.py").resolve():
        raise H.PrecomputeRefused("job must bind the actual pinned upstream tokenizer")
    if scorer != pathlib.Path(__file__).resolve() or scorer_sha != H.sha256_file(__file__):
        raise H.PrecomputeRefused("job does not bind the executing scorer")
    freeze_payload = json.loads(freeze.read_text(encoding="utf-8"))
    if freeze_payload.get("checkpoint", {}).get("sha256") != checkpoint_sha:
        raise H.PrecomputeRefused("pilot freeze does not bind this checkpoint")
    if freeze_payload.get("materialized_config", {}).get("sha256") != config_sha:
        raise H.PrecomputeRefused("pilot freeze does not bind this config")
    if freeze_payload.get("source_bindings", {}).get("scorer_sha256") != scorer_sha:
        raise H.PrecomputeRefused("pilot freeze predates or disagrees with this scorer")
    if freeze_payload.get("source_bindings", {}).get("tokenizer_sha256") != tokenizer_sha:
        raise H.PrecomputeRefused("pilot freeze does not bind this tokenizer")
    if freeze_payload.get("source_bindings", {}).get("adaptation_contract_sha256") != adaptation_sha:
        raise H.PrecomputeRefused("pilot freeze does not bind the common BST CE implementation")
    if payload.get("upstream_commit") != H.UPSTREAM_COMMIT:
        raise H.PrecomputeRefused("score job does not bind the pinned upstream commit")
    inputs = []
    raw_inputs = payload.get("inputs")
    if not isinstance(raw_inputs, list) or len(raw_inputs) != 7:
        raise H.PrecomputeRefused("score job must list the exact seven frozen input pools")
    allowed_names = {
        "b_near", "b_far", "b_mid", "acquisition_near", "acquisition_mid", "acquisition_far",
        "candidate_generation_receipt",
    }
    seen = set()
    for item in raw_inputs:
        if not isinstance(item, Mapping) or item.get("name") not in allowed_names:
            raise H.PrecomputeRefused("unknown score input role")
        name = str(item["name"])
        if name in seen:
            raise H.PrecomputeRefused("duplicate score input role")
        input_path, digest = _record(item, base, f"input {name}")
        inputs.append((name, input_path, digest))
        seen.add(name)
    if seen != allowed_names:
        raise H.PrecomputeRefused("score input roles are incomplete")
    output = pathlib.Path(str(payload.get("output_dir", "")))
    if not output.is_absolute():
        output = base / output
    payload["_bound"] = {
        "checkpoint": checkpoint, "config": config, "tokenizer": tokenizer,
        "freeze": freeze, "scorer": scorer, "adaptation": adaptation,
        "inputs": inputs, "output": output.resolve(), "freeze_sha256": freeze_sha,
        "tokenizer_sha256": tokenizer_sha, "adaptation_sha256": adaptation_sha,
    }
    return payload


def _input_rows(bound: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, path, _digest in bound["inputs"]:
        if name == "candidate_generation_receipt":
            continue
        for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            row = json.loads(raw)
            line = row.get("line")
            if not isinstance(line, str) or row.get("prompt_sha256") != H.prompt_sha(line):
                raise H.PrecomputeRefused(f"{name}:{number} has invalid prompt identity")
            validate_line(line)
            rows.append({"pool": name, "prompt_sha256": row["prompt_sha256"], "line": line})
    expected = H.MID_COUNT + H.MID_CANDIDATE_COUNT * 2 + H.ACQUISITION_CANDIDATE_COUNT * 3
    if len(rows) != expected or len({row["prompt_sha256"] for row in rows}) != expected:
        raise H.PrecomputeRefused(f"score population must contain exactly {expected} unique items")
    return rows


def _load_model(job: Mapping[str, Any], device: str):
    torch = importlib.import_module("torch")
    from extract_lurestar_evidence import _import_models, _load_materialized_config, _model_args
    bound = job["_bound"]
    _gpt, _nextlat, bst_mod, stargraph = _import_models(ROOT / "upstream/NextLat")
    config = _load_materialized_config(bound["config"], upstream=ROOT / "upstream/NextLat", arm="bst", seed=H.PILOT_SEED)
    wrapper = bst_mod.BST(bst_mod.BSTConfig(**_model_args(config, "bst")))
    raw = torch.load(bound["checkpoint"], map_location="cpu", weights_only=False)
    if raw.get("training_steps") != H.PILOT_STEP:
        raise H.PrecomputeRefused("checkpoint payload is not exact training step 500")
    wrapper.encoder.load_state_dict(raw["encoder"], strict=True)
    wrapper.text_head.load_state_dict(raw["text_head"], strict=True)
    wrapper.encoder.to(device).eval()
    wrapper.text_head.to(device).eval()
    tokenizer = stargraph.Tokenizer(100)
    probe = "1,2|2,3/1,3=1,2,3"
    if tokenizer.encode(probe) != token_ids(probe):
        raise H.PrecomputeRefused("actual upstream tokenizer disagrees with frozen oracle")
    return wrapper, tokenizer


def _tokenize(tokenizer: Any, lines: Sequence[str]) -> np.ndarray:
    arrays = [np.asarray(tokenizer.tokenize(line)[0], dtype=np.int64) for line in lines]
    if {len(row) for row in arrays} != {69}:
        raise H.PrecomputeRefused("pilot scorer received a non-G(5,5) token row")
    return np.stack(arrays)


def _score(wrapper: Any, tokenizer: Any, rows: Sequence[Mapping[str, Any]], device: str) -> np.ndarray:
    torch = importlib.import_module("torch")
    from lurestar.adaptation import _masked_targets, bst_next_token_logits
    tokens = torch.as_tensor(_tokenize(tokenizer, [str(row["line"]) for row in rows]), dtype=torch.long, device=device)
    with torch.inference_mode():
        logits = bst_next_token_logits(wrapper, tokens[:, :-1])
        targets = _masked_targets(wrapper, tokens)
        per_token = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100,
            reduction="none",
        ).reshape(tokens.shape[0], -1)
        mask = targets.ne(-100)
        losses = (per_token * mask).sum(1) / mask.sum(1)
    if not torch.isfinite(losses).all():
        raise H.PrecomputeRefused("pilot scorer produced a nonfinite loss")
    # ``tolist`` also works in the local verification environment whose old torch wheel was
    # compiled against NumPy 1.x; production Colab has a matching torch/NumPy pair.
    return np.asarray(losses.float().cpu().tolist(), dtype=np.float32)


def _chunk_paths(output: pathlib.Path, start: int, stop: int) -> tuple[pathlib.Path, pathlib.Path]:
    stem = f"loss-{start:06d}-{stop:06d}"
    return output / "chunks" / f"{stem}.jsonl", output / "chunks" / f"{stem}.receipt.json"


def _write_chunk(output: pathlib.Path, job_sha: str, rows: Sequence[Mapping[str, Any]], losses: np.ndarray, start: int, stop: int) -> None:
    path, receipt_path = _chunk_paths(output, start, stop)
    records = [{
        "schema": H.LOSS_TABLE_SCHEMA,
        "ordinal": start + offset,
        "pool": row["pool"],
        "prompt_sha256": row["prompt_sha256"],
        "loss": float(losses[offset]),
    } for offset, row in enumerate(rows)]
    digest = H.create_or_verify(path, b"".join(H.canonical_json(record) for record in records))
    H.create_or_verify(receipt_path, H.canonical_json({
        "schema": CHUNK_SCHEMA, "job_sha256": job_sha, "start": start, "stop": stop,
        "row_count": stop - start, "loss_chunk_sha256": digest,
    }))


def _verified_chunk(output: pathlib.Path, job_sha: str, start: int, stop: int) -> bytes | None:
    path, receipt_path = _chunk_paths(output, start, stop)
    if not path.exists() and not receipt_path.exists():
        return None
    if not path.is_file() or not receipt_path.is_file():
        raise H.PrecomputeRefused("partial/corrupt pilot chunk blocks restart")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt != {
        "schema": CHUNK_SCHEMA, "job_sha256": job_sha, "start": start, "stop": stop,
        "row_count": stop - start, "loss_chunk_sha256": H.verify_sidecar(path),
    }:
        raise H.PrecomputeRefused("pilot chunk receipt identity mismatch")
    H.verify_sidecar(receipt_path)
    return path.read_bytes()


def run_score(job_path: pathlib.Path, *, device: str, chunk_size: int, batch_size: int) -> dict[str, Any]:
    if chunk_size <= 0 or batch_size <= 0:
        raise H.PrecomputeRefused("chunk/batch sizes must be positive")
    job_sha = H.sha256_file(job_path)
    job = _load_job(job_path)
    bound, output = job["_bound"], job["_bound"]["output"]
    identity = {"schema": JOB_SCHEMA, "job_sha256": job_sha}
    H.create_or_verify(output / "job_identity.json", H.canonical_json(identity))
    rows = _input_rows(bound)
    model = tokenizer = None
    chunks: list[bytes] = []
    for start in range(0, len(rows), chunk_size):
        stop = min(start + chunk_size, len(rows))
        existing = _verified_chunk(output, job_sha, start, stop)
        if existing is None:
            if model is None:
                model, tokenizer = _load_model(job, device)
            parts = []
            for begin in range(start, stop, batch_size):
                end = min(begin + batch_size, stop)
                parts.append(_score(model, tokenizer, rows[begin:end], device))
            _write_chunk(output, job_sha, rows[start:stop], np.concatenate(parts), start, stop)
            existing = _verified_chunk(output, job_sha, start, stop)
        assert existing is not None
        chunks.append(existing)
    table_payload = b"".join(chunks)
    table_sha = H.create_or_verify(output / "pilot_losses.jsonl", table_payload)
    receipt = {
        "schema": H.LOSS_TABLE_SCHEMA,
        "status": "COMPLETE_NONCONFIRMATORY_PILOT_SCORING",
        "job_sha256": job_sha,
        "pilot_freeze_sha256": bound["freeze_sha256"],
        "checkpoint_sha256": H.PILOT_CHECKPOINT_SHA256,
        "config_sha256": H.PILOT_CONFIG_SHA256,
        "tokenizer_sha256": bound["tokenizer_sha256"],
        "adaptation_contract_sha256": bound["adaptation_sha256"],
        "loss_table_sha256": table_sha,
        "row_count": len(rows),
        "confirmatory_inputs_inspected": False,
        "confirmatory_results_inspected": False,
    }
    H.create_or_verify(output / "pilot_scoring_receipt.json", H.canonical_json(receipt))
    return receipt


def plan(job_path: pathlib.Path | None) -> dict[str, Any]:
    return {
        "schema": JOB_SCHEMA,
        "mode": "READ_ONLY_PLAN",
        "job_present": bool(job_path and job_path.is_file()),
        "frozen_pilot": {"family": "bst", "seed": H.PILOT_SEED, "step": H.PILOT_STEP, "checkpoint_sha256": H.PILOT_CHECKPOINT_SHA256},
        "score_items": H.MID_COUNT + 2 * H.MID_CANDIDATE_COUNT + 3 * H.ACQUISITION_CANDIDATE_COUNT,
        "estimated_a100_minutes": [3, 12],
        "durability": "atomic create-only chunks; restart verifies job and chunk SHA-256",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("plan", "score"), required=True)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--job", type=pathlib.Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk-size", type=int, default=1_000)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args(argv)
    if args.plan and args.mode != "plan":
        raise H.PrecomputeRefused("--plan cannot accompany scoring")
    if args.mode == "plan":
        result = plan(args.job)
    else:
        if args.job is None:
            raise H.PrecomputeRefused("--mode score requires --job")
        result = run_score(args.job.resolve(), device=args.device, chunk_size=args.chunk_size, batch_size=args.batch_size)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except H.PrecomputeRefused as exc:
        raise SystemExit(f"BLOCK: {exc}")
