#!/usr/bin/env python3
"""Evaluate one frozen NextLat checkpoint for the retrospective basin case study.

This leaves the historical competence evaluator unchanged. It reproduces that evaluator's exact
five-token greedy metric and adds one predeclared diagnostic: teacher-forced performance at path
position 2, the first nontrivial edge choice after conditioning on the gold source node.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import os
import pathlib
import sys
from collections.abc import Sequence
from typing import Any


REPO = pathlib.Path(__file__).resolve().parents[2]
BASE_EVALUATOR_PATH = REPO / "scripts" / "evaluate_base_competence.py"
_spec = importlib.util.spec_from_file_location("_frozen_base_evaluator", BASE_EVALUATOR_PATH)
if _spec is None or _spec.loader is None:  # pragma: no cover - broken installation
    raise RuntimeError("cannot load the frozen base competence evaluator")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)


SCHEMA = "nextlat_forgetting/basin_checkpoint_evaluation/1"
PROGRESS_SCHEMA = "nextlat_forgetting/basin_checkpoint_evaluation_progress/1"
BRANCH_PATH_POSITION = 2
BRANCH_HIDDEN_INDEX = 63


class BasinEvaluationError(RuntimeError):
    """A case-study provenance, runtime, or measurement precondition failed."""


def summarize_margins(*, count: int, total: float, total_sq: float,
                      minimum: float, maximum: float) -> dict[str, float | int]:
    """Return stable population summaries without retaining prompt-level model outputs."""
    if count <= 0:
        raise BasinEvaluationError("cannot summarize an empty margin collection")
    mean = total / count
    variance = max(0.0, total_sq / count - mean * mean)
    return {
        "count": count,
        "mean": mean,
        "population_std": math.sqrt(variance),
        "min": minimum,
        "max": maximum,
    }


def _fresh_progress(identity: dict[str, Any], *, total: int) -> dict[str, Any]:
    return {
        "schema": PROGRESS_SCHEMA,
        **identity,
        "next_index": 0,
        "total": total,
        "exact_correct": 0,
        "token_correct": [0] * E.TARGET_TOKENS,
        "branch_correct": 0,
        "branch_margin_count": 0,
        "branch_margin_sum": 0.0,
        "branch_margin_sum_sq": 0.0,
        "branch_margin_min": None,
        "branch_margin_max": None,
    }


def load_progress(path: pathlib.Path, identity: dict[str, Any], *, total: int) -> dict[str, Any]:
    """Resume only an evaluation with exactly matching scientific provenance."""
    fresh = _fresh_progress(identity, total=total)
    if not path.is_file():
        return fresh
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BasinEvaluationError("progress file is unreadable or invalid") from exc
    if not isinstance(saved, dict) or set(saved) != set(fresh):
        raise BasinEvaluationError("progress schema changed")
    mutable = {
        "next_index", "exact_correct", "token_correct", "branch_correct",
        "branch_margin_count", "branch_margin_sum", "branch_margin_sum_sq",
        "branch_margin_min", "branch_margin_max",
    }
    for key, expected in fresh.items():
        if key not in mutable and saved.get(key) != expected:
            raise BasinEvaluationError(f"progress provenance mismatch for {key}")
    next_index = saved["next_index"]
    counts = [saved["exact_correct"], saved["branch_correct"], saved["branch_margin_count"]]
    token_correct = saved["token_correct"]
    if (
        not isinstance(next_index, int) or isinstance(next_index, bool)
        or not 0 <= next_index <= total
        or any(not isinstance(value, int) or isinstance(value, bool)
               or not 0 <= value <= next_index for value in counts)
        or not isinstance(token_correct, list) or len(token_correct) != E.TARGET_TOKENS
        or any(not isinstance(value, int) or isinstance(value, bool)
               or not 0 <= value <= next_index for value in token_correct)
        or saved["branch_margin_count"] != next_index
    ):
        raise BasinEvaluationError("progress counters are invalid")
    for key in ("branch_margin_sum", "branch_margin_sum_sq"):
        if not isinstance(saved[key], (int, float)) or not math.isfinite(saved[key]):
            raise BasinEvaluationError(f"progress {key} is invalid")
    if next_index:
        for key in ("branch_margin_min", "branch_margin_max"):
            if not isinstance(saved[key], (int, float)) or not math.isfinite(saved[key]):
                raise BasinEvaluationError(f"progress {key} is invalid")
    return saved


def _nextlat_logits(model: Any, tokens: Any) -> Any:
    """Return next-token logits through the pinned upstream NextLat model surface."""
    return model.model(tokens)[:, -1, :]


def evaluate_checkpoint(
    *, job_id: str, seed: int, step: int, checkpoint: pathlib.Path,
    expected_checkpoint_sha256: str, config_path: pathlib.Path,
    source_config: pathlib.Path, dataset: pathlib.Path,
    manifests: Sequence[pathlib.Path], upstream: pathlib.Path,
    output: pathlib.Path, batch_size: int,
) -> dict[str, Any]:
    """Run the provenance-bound greedy and first-decision evaluation."""
    expected_job = f"nextlat-s{seed}-base"
    if job_id != expected_job:
        raise BasinEvaluationError(f"job identity {job_id!r} does not equal {expected_job!r}")
    if step <= 0 or batch_size <= 0:
        raise BasinEvaluationError("step and batch size must be positive")
    checkpoint = checkpoint.resolve()
    dataset = dataset.resolve()
    upstream = upstream.resolve()
    if not checkpoint.is_file():
        raise BasinEvaluationError(f"checkpoint is missing: {checkpoint}")
    checkpoint_sha = E.sha256_file(checkpoint)
    if checkpoint_sha != expected_checkpoint_sha256:
        raise BasinEvaluationError("checkpoint SHA-256 does not match the frozen artifact roster")
    if not dataset.is_file() or dataset.name != E.EXPECTED_DATASET_NAME:
        raise BasinEvaluationError(f"dataset must be {E.EXPECTED_DATASET_NAME}")

    manifest_records = E.verify_manifest_binding(dataset, manifests)
    config = E._load_and_validate_config(
        config_path.resolve(), source_config.resolve(), model_name="nextlat", seed=seed,
        upstream=upstream,
    )
    examples: list[tuple[list[int], list[int]]] = []
    for line_number, line in enumerate(dataset.read_text(encoding="utf-8").splitlines(), 1):
        try:
            examples.append(E.validate_g55_tokens(E.encode_stargraph_line(line)))
        except E.EvaluationError as exc:
            raise BasinEvaluationError(f"invalid held-out row {line_number}: {exc}") from exc
    if len(examples) != E.EXPECTED_EXAMPLES:
        raise BasinEvaluationError(
            f"held-out corpus has {len(examples)} rows, expected {E.EXPECTED_EXAMPLES}"
        )

    evaluator_path = pathlib.Path(__file__).resolve()
    identity = {
        "job_id": job_id,
        "seed": seed,
        "step": step,
        "checkpoint_sha256": checkpoint_sha,
        "dataset_sha256": E.sha256_file(dataset),
        "evaluator_sha256": E.sha256_file(evaluator_path),
        "base_evaluator_sha256": E.sha256_file(BASE_EVALUATOR_PATH),
        "config_sha256": E.sha256_file(source_config.resolve()),
        "materialized_config_sha256": E.sha256_file(config_path.resolve()),
        "manifest_sha256s": sorted(record["sha256"] for record in manifest_records),
        "upstream_commit": E.PINNED_UPSTREAM_COMMIT,
        "decoding": E.DECODING,
        "branch_path_position": BRANCH_PATH_POSITION,
        "branch_hidden_index": BRANCH_HIDDEN_INDEX,
    }
    progress_path = output.with_name(output.name + ".progress.json")
    progress = load_progress(progress_path, identity, total=len(examples))

    try:
        import lightning as L
        import torch
    except ImportError as exc:
        raise BasinEvaluationError("pinned torch/lightning runtime is unavailable") from exc

    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False

    prior_cwd = pathlib.Path.cwd()
    sys.path.insert(0, str(upstream))
    try:
        os.chdir(upstream)
        from core_train import initialize_model
        from data.stargraph import Tokenizer

        fabric = L.Fabric()
        fabric.seed_everything(seed)
        model = initialize_model(
            fabric, config, Tokenizer(E.MAX_NODES), initialize_optimizer=False,
            checkpoint_path=str(checkpoint),
        )
        model.eval()

        exact_correct = int(progress["exact_correct"])
        token_correct = [int(value) for value in progress["token_correct"]]
        branch_correct = int(progress["branch_correct"])
        margin_count = int(progress["branch_margin_count"])
        margin_sum = float(progress["branch_margin_sum"])
        margin_sum_sq = float(progress["branch_margin_sum_sq"])
        margin_min = progress["branch_margin_min"]
        margin_max = progress["branch_margin_max"]
        resume_index = int(progress["next_index"])

        inference_context = getattr(torch, "inference_mode", contextlib.nullcontext)
        with inference_context():
            for start in range(resume_index, len(examples), batch_size):
                batch = examples[start:start + batch_size]
                prefix = torch.tensor(
                    [item[0] for item in batch], dtype=torch.long, device=fabric.device
                )
                targets = torch.tensor(
                    [item[1] for item in batch], dtype=torch.long, device=fabric.device
                )

                predictions = E._torch_greedy(model, "nextlat", prefix, torch)
                matches = predictions.eq(targets)
                exact_correct += int(matches.all(dim=1).sum().item())
                token_correct = [
                    old + int(new)
                    for old, new in zip(token_correct, matches.sum(dim=0).tolist())
                ]

                branch_context = torch.cat((prefix, targets[:, :1]), dim=1)
                logits = _nextlat_logits(model, branch_context).float()
                gold = targets[:, 1]
                predicted = torch.argmax(logits, dim=-1)
                branch_correct += int(predicted.eq(gold).sum().item())
                gold_logits = logits.gather(1, gold[:, None]).squeeze(1)
                alternatives = logits.clone()
                alternatives.scatter_(1, gold[:, None], float("-inf"))
                margins = (gold_logits - alternatives.max(dim=1).values).double().cpu()
                values = margins.tolist()
                margin_count += len(values)
                margin_sum += float(margins.sum().item())
                margin_sum_sq += float((margins * margins).sum().item())
                batch_min = min(values)
                batch_max = max(values)
                margin_min = batch_min if margin_min is None else min(float(margin_min), batch_min)
                margin_max = batch_max if margin_max is None else max(float(margin_max), batch_max)

                E.atomic_write_json(progress_path, {
                    "schema": PROGRESS_SCHEMA,
                    **identity,
                    "next_index": start + len(batch),
                    "total": len(examples),
                    "exact_correct": exact_correct,
                    "token_correct": token_correct,
                    "branch_correct": branch_correct,
                    "branch_margin_count": margin_count,
                    "branch_margin_sum": margin_sum,
                    "branch_margin_sum_sq": margin_sum_sq,
                    "branch_margin_min": margin_min,
                    "branch_margin_max": margin_max,
                })
    finally:
        os.chdir(prior_cwd)

    total = len(examples)
    margin_summary = summarize_margins(
        count=margin_count, total=margin_sum, total_sq=margin_sum_sq,
        minimum=float(margin_min), maximum=float(margin_max),
    )
    result = {
        "schema": SCHEMA,
        **identity,
        "exact_path_accuracy": {
            "correct": exact_correct, "total": total, "value": exact_correct / total,
        },
        "per_token_accuracy": [
            {"position": index + 1, "correct": count, "total": total, "value": count / total}
            for index, count in enumerate(token_correct)
        ],
        "teacher_forced_first_decision": {
            "path_position": BRANCH_PATH_POSITION,
            "hidden_index": BRANCH_HIDDEN_INDEX,
            "conditioning": "gold source node at path position 1",
            "accuracy": {
                "correct": branch_correct, "total": total, "value": branch_correct / total,
            },
            "gold_logit_margin": {
                "definition": "gold next-node logit minus largest non-gold logit",
                **margin_summary,
            },
        },
        "runtime": {
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "device": str(fabric.device),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        },
    }
    E.atomic_write_json(output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--step", required=True, type=int)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)
    try:
        result = evaluate_checkpoint(
            job_id=args.job_id, seed=args.seed, step=args.step,
            checkpoint=pathlib.Path(args.checkpoint),
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            config_path=pathlib.Path(args.config), source_config=pathlib.Path(args.source_config),
            dataset=pathlib.Path(args.dataset),
            manifests=[pathlib.Path(path) for path in args.manifest],
            upstream=pathlib.Path(args.upstream), output=pathlib.Path(args.output),
            batch_size=args.batch_size,
        )
    except (BasinEvaluationError, E.EvaluationError, OSError, ValueError) as exc:
        print(f"[evaluate_basin_checkpoint] REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
