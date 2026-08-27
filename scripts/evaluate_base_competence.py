#!/usr/bin/env python
"""Deterministically score a trained base checkpoint on held-out Path-Star G(5,5).

The paper-comparable validation loop samples from the model.  This evaluator is the separate,
binding competence gate: it performs explicit argmax decoding for exactly five answer tokens and
writes integer exact-path counts.  It never calls upstream ``generate`` because those methods
divide by ``temperature`` before sampling; passing the frozen semantic value ``temperature=0``
there would create infinities rather than a well-defined greedy decoder.

Production usage (normally through ``evaluate_trained_bases.py``)::

    fabric run --devices 1 --precision bf16-mixed \
      scripts/evaluate_base_competence.py --upstream /content/project/upstream/NextLat \
      --job-id nextlat-s1234-base --model nextlat --seed 1234 \
      --checkpoint ...pt --config .../materialized_config.yaml \
      --source-config configs/nextlat_lurestar.yaml \
      --dataset /content/lurestar/data/stargraph/graph_5_5_test_20000.txt \
      --manifest /content/lurestar/manifests/corpus.sha256 --output ...json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import pathlib
import re
import subprocess
import sys
from typing import Callable, Iterable, Sequence

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from lurestar.durable_checkpoint import atomic_write_json, sha256_file  # noqa: E402


SCHEMA = "nextlat_forgetting/exact_path_evaluation/1"
PINNED_UPSTREAM_COMMIT = "3770be6009cea2b3c455a9ce7f2ca88b504bb955"
DECODING = {"strategy": "greedy", "top_k": 1, "temperature": 0.0}
EXPECTED_DATASET_NAME = "graph_5_5_test_20000.txt"
EXPECTED_EXAMPLES = 20_000
MAX_NODES = 100
TARGET_TOKENS = 5
PREFIX_TOKENS = 63
SEQUENCE_TOKENS = 69
VOCAB_SIZE = 106  # upstream StarGraphDataModule: maxNodes + 5 + one slack row
MODEL_FLAGS = {
    "gpt": (False, False),
    "nextlat": (False, True),
    "bst": (True, False),
}


class EvaluationError(RuntimeError):
    """A fail-closed evaluator precondition or integrity failure."""


def encode_stargraph_line(line: str, *, max_nodes: int = MAX_NODES) -> list[int]:
    """Replay upstream ``Tokenizer.encode`` without importing its GPU dependency surface."""
    text = line.strip()
    if not text:
        raise EvaluationError("held-out dataset contains an empty line")
    separators = {"|": max_nodes, "=": max_nodes + 1, "/": max_nodes + 2}
    encoded: list[int] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == ",":
            index += 1
            continue
        if char.isdigit():
            end = index + 1
            while end < len(text) and text[end].isdigit():
                end += 1
            value = int(text[index:end])
            if not 0 <= value < max_nodes:
                raise EvaluationError(f"node id {value} is outside [0, {max_nodes})")
            encoded.append(value)
            index = end
            continue
        if char not in separators:
            raise EvaluationError(f"unsupported character {char!r} in held-out dataset")
        encoded.append(separators[char])
        index += 1
    encoded.append(max_nodes + 4)  # upstream Tokenizer.eos_token_id
    return encoded


def validate_g55_tokens(tokens: Sequence[int]) -> tuple[list[int], list[int]]:
    """Validate exact upstream G(5,5) sequence geometry and split prefix/answer."""
    if len(tokens) != SEQUENCE_TOKENS:
        raise EvaluationError(
            f"held-out row has {len(tokens)} tokens, expected G(5,5) length {SEQUENCE_TOKENS}"
        )
    if tokens.count(MAX_NODES + 1) != 1 or tokens[PREFIX_TOKENS - 1] != MAX_NODES + 1:
        raise EvaluationError("held-out row does not have '=' at frozen token index 62")
    if tokens[-1] != MAX_NODES + 4:
        raise EvaluationError("held-out row lacks the upstream EOS token")
    prefix = list(tokens[:PREFIX_TOKENS])
    answer = list(tokens[PREFIX_TOKENS:-1])
    if len(answer) != TARGET_TOKENS:
        raise EvaluationError("held-out row does not contain exactly five answer tokens")
    return prefix, answer


def greedy_decode_lists(
    prefixes: Sequence[Sequence[int]],
    *,
    steps: int,
    next_tokens: Callable[[Sequence[Sequence[int]]], Sequence[int]],
) -> list[list[int]]:
    """Backend-neutral autoregressive greedy loop used by production and exact fixtures."""
    sequences = [list(prefix) for prefix in prefixes]
    for _ in range(steps):
        predicted = list(next_tokens(sequences))
        if len(predicted) != len(sequences):
            raise EvaluationError("next-token function returned the wrong batch size")
        for sequence, token in zip(sequences, predicted):
            if not isinstance(token, int) or isinstance(token, bool):
                raise EvaluationError("next-token function returned a non-integer token")
            sequence.append(token)
    return [sequence[-steps:] for sequence in sequences]


def exact_path_counts(
    predictions: Sequence[Sequence[int]], targets: Sequence[Sequence[int]]
) -> dict[str, object]:
    """Return exact integer counts; partial paths never count as solved."""
    if len(predictions) != len(targets) or not targets:
        raise EvaluationError("predictions and targets must have the same nonzero batch size")
    correct = 0
    for prediction, target in zip(predictions, targets):
        if len(prediction) != TARGET_TOKENS or len(target) != TARGET_TOKENS:
            raise EvaluationError("exact-path inputs must each contain exactly five tokens")
        correct += int(list(prediction) == list(target))
    total = len(targets)
    return {"correct": correct, "total": total, "value": correct / total}


def load_progress(path: pathlib.Path, identity: dict, *, total: int) -> dict:
    """Load a provenance-bound partial evaluation, or return a fresh accumulator."""
    fresh = {
        "schema": "nextlat_forgetting/exact_path_evaluation_progress/1",
        **identity,
        "next_index": 0,
        "correct": 0,
        "token_correct": [0] * TARGET_TOKENS,
        "total": total,
    }
    if not path.is_file():
        return fresh
    try:
        progress = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise EvaluationError("partial evaluation progress is not valid JSON") from exc
    for key, expected in fresh.items():
        if key in {"next_index", "correct", "token_correct"}:
            continue
        if progress.get(key) != expected:
            raise EvaluationError(f"partial evaluation progress {key} provenance mismatch")
    next_index = progress.get("next_index")
    correct = progress.get("correct")
    token_correct = progress.get("token_correct")
    if (
        not isinstance(next_index, int) or isinstance(next_index, bool)
        or not 0 <= next_index <= total
        or not isinstance(correct, int) or isinstance(correct, bool)
        or not 0 <= correct <= next_index
        or not isinstance(token_correct, list) or len(token_correct) != TARGET_TOKENS
        or any(not isinstance(value, int) or isinstance(value, bool)
               or not 0 <= value <= next_index for value in token_correct)
    ):
        raise EvaluationError("partial evaluation progress contains invalid counts")
    return progress


def verify_manifest_binding(dataset: pathlib.Path, manifests: Sequence[pathlib.Path]) -> list[dict]:
    """Require an exact sha256sum-style row binding the held-out dataset in a frozen manifest."""
    dataset = dataset.resolve()
    dataset_sha = sha256_file(dataset)
    records = []
    bound = False
    row = re.compile(r"^([0-9a-fA-F]{64})\s+\*?(.+?)(?:\s+\d+)?$")
    for manifest in manifests:
        manifest = manifest.resolve()
        if not manifest.is_file():
            raise EvaluationError(f"evaluation manifest is missing: {manifest}")
        for raw in manifest.read_text(encoding="utf-8").splitlines():
            match = row.match(raw.strip())
            if not match:
                continue
            digest, named = match.groups()
            if pathlib.Path(named).name == dataset.name and digest.lower() == dataset_sha:
                bound = True
        records.append({"path": str(manifest), "sha256": sha256_file(manifest)})
    if not bound:
        raise EvaluationError(
            f"no supplied manifest exactly binds {dataset.name} to SHA-256 {dataset_sha}"
        )
    return records


def _git_head(upstream: pathlib.Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EvaluationError("cannot determine upstream git commit") from exc


def _load_and_validate_config(
    config_path: pathlib.Path, source_config: pathlib.Path, *, model_name: str, seed: int,
    upstream: pathlib.Path,
):
    """Load the exact materialized training config and assert frozen scientific identity."""
    import yaml
    from omegaconf import OmegaConf

    config_path = config_path.resolve()
    source_config = source_config.resolve()
    for label, path in (("materialized config", config_path), ("source config", source_config)):
        if not path.is_file():
            raise EvaluationError(f"{label} is missing: {path}")
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise EvaluationError("materialized config is not a mapping")
    if int(document.get("seed", -1)) != seed:
        raise EvaluationError("materialized config seed does not match evaluated job")
    expected_bst, expected_nextlat = MODEL_FLAGS[model_name]
    if bool(document.get("use_bst", False)) != expected_bst:
        raise EvaluationError("materialized config use_bst flag does not match model identity")
    if bool(document.get("use_nextlat", False)) != expected_nextlat:
        raise EvaluationError("materialized config use_nextlat flag does not match model identity")
    if document.get("data", {}).get("dataset") != "stargraph":
        raise EvaluationError("competence evaluator only accepts the StarGraph dataset")
    provenance = document.get("provenance", {})
    if provenance.get("upstream_commit") != PINNED_UPSTREAM_COMMIT:
        raise EvaluationError("materialized config does not bind the pinned upstream commit")
    if _git_head(upstream) != PINNED_UPSTREAM_COMMIT:
        raise EvaluationError("upstream checkout is not at the pinned commit")

    default = OmegaConf.load(str(upstream / "defaults.yaml"))
    config = OmegaConf.merge(default, OmegaConf.load(str(config_path)))
    # These are measured by the exact held-out data geometry, just as StarGraphDataModule does.
    config.model.vocab_size = VOCAB_SIZE
    config.model.context_length = PREFIX_TOKENS - 1
    config.model.block_size = SEQUENCE_TOKENS
    config.data.device_batch_size = int(config.data.effective_batch_size)
    config.data.micro_batch_size = int(config.data.effective_batch_size)
    return config


def _torch_greedy(model, model_name: str, prefix, torch):
    """Return five explicit argmax tokens; never dispatch to sampling-based ``generate``."""
    generated = prefix
    for _ in range(TARGET_TOKENS):
        cropped = generated[:, -int(model.config.block_size):]
        if model_name == "bst":
            eos = torch.full(
                (cropped.size(0), 1), int(model.config.eos_token_id),
                dtype=torch.long, device=cropped.device,
            )
            _, backward = model.encoder(eos, compute_forward=False, compute_backward=True)
            forward, _ = model.encoder(cropped, compute_forward=True, compute_backward=False)
            logits = model.text_head(forward[:, -1:, :], backward)[:, 0, :]
            # Upstream BST retains a singleton sequence axis here and squeezes it only after
            # softmax in ``generate``.  We do the equivalent before argmax.
            if logits.dim() == 3:
                logits = logits.squeeze(1)
        else:
            logits = model.model(cropped)[:, -1, :]
        token = torch.argmax(logits.float(), dim=-1, keepdim=True)
        generated = torch.cat((generated, token), dim=1)
    return generated[:, -TARGET_TOKENS:]


def evaluate_checkpoint(
    *, job_id: str, model_name: str, seed: int, checkpoint: pathlib.Path,
    config_path: pathlib.Path, source_config: pathlib.Path, dataset: pathlib.Path,
    manifests: Sequence[pathlib.Path], upstream: pathlib.Path, output: pathlib.Path,
    batch_size: int,
) -> dict:
    """Execute the complete pinned evaluation and atomically emit its provenance receipt."""
    expected_job = f"{model_name}-s{seed}-base"
    if model_name not in MODEL_FLAGS:
        raise EvaluationError(f"unsupported model {model_name!r}")
    if job_id != expected_job:
        raise EvaluationError(f"job identity {job_id!r} does not equal {expected_job!r}")
    if batch_size <= 0:
        raise EvaluationError("batch size must be positive")
    checkpoint = checkpoint.resolve()
    dataset = dataset.resolve()
    upstream = upstream.resolve()
    if not checkpoint.is_file():
        raise EvaluationError(f"checkpoint is missing: {checkpoint}")
    if not dataset.is_file() or dataset.name != EXPECTED_DATASET_NAME:
        raise EvaluationError(f"dataset must be the frozen {EXPECTED_DATASET_NAME}")
    manifest_records = verify_manifest_binding(dataset, manifests)
    config = _load_and_validate_config(
        config_path, source_config, model_name=model_name, seed=seed, upstream=upstream
    )

    examples: list[tuple[list[int], list[int]]] = []
    for line_number, line in enumerate(dataset.read_text(encoding="utf-8").splitlines(), 1):
        try:
            examples.append(validate_g55_tokens(encode_stargraph_line(line)))
        except EvaluationError as exc:
            raise EvaluationError(f"invalid held-out row {line_number}: {exc}") from exc
    if len(examples) != EXPECTED_EXAMPLES:
        raise EvaluationError(
            f"held-out corpus has {len(examples)} rows, expected exactly {EXPECTED_EXAMPLES}"
        )

    evaluator_path = pathlib.Path(__file__).resolve()
    progress_identity = {
        "job_id": job_id,
        "model": model_name,
        "seed": seed,
        "checkpoint_sha256": sha256_file(checkpoint),
        "dataset_sha256": sha256_file(dataset),
        "evaluator_sha256": sha256_file(evaluator_path),
        "config_sha256": sha256_file(source_config),
        "materialized_config_sha256": sha256_file(config_path),
        "manifest_sha256s": sorted(record["sha256"] for record in manifest_records),
        "upstream_commit": PINNED_UPSTREAM_COMMIT,
        "decoding": DECODING,
    }
    progress_path = output.with_name(output.name + ".progress.json")
    progress = load_progress(progress_path, progress_identity, total=len(examples))

    # Import the pinned trainer only after all cheap integrity checks pass.
    try:
        import torch
        import lightning as L
    except ImportError as exc:
        raise EvaluationError("pinned torch/lightning runtime is unavailable") from exc
    sys.path.insert(0, str(upstream))
    prior_cwd = pathlib.Path.cwd()
    try:
        os.chdir(upstream)
        from core_train import initialize_model
        from data.stargraph import Tokenizer

        fabric = L.Fabric()
        fabric.seed_everything(seed)
        tokenizer = Tokenizer(MAX_NODES)
        model = initialize_model(
            fabric, config, tokenizer, initialize_optimizer=False,
            checkpoint_path=str(checkpoint),
        )
        model.eval()
        correct = int(progress["correct"])
        token_correct = [int(value) for value in progress["token_correct"]]
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
                predictions = _torch_greedy(model, model_name, prefix, torch)
                matches = predictions.eq(targets)
                correct += int(matches.all(dim=1).sum().item())
                per_token = matches.sum(dim=0).tolist()
                token_correct = [a + int(b) for a, b in zip(token_correct, per_token)]
                atomic_write_json(progress_path, {
                    "schema": "nextlat_forgetting/exact_path_evaluation_progress/1",
                    **progress_identity,
                    "next_index": start + len(batch),
                    "correct": correct,
                    "token_correct": token_correct,
                    "total": len(examples),
                })
    finally:
        os.chdir(prior_cwd)

    total = len(examples)
    result = {
        "schema": SCHEMA,
        "job_id": job_id,
        "model": model_name,
        "seed": seed,
        **{key: value for key, value in progress_identity.items()
           if key not in {"job_id", "model", "seed"}},
        "exact_path_accuracy": {"correct": correct, "total": total, "value": correct / total},
        "per_token_accuracy": [
            {"position": index + 1, "correct": count, "total": total, "value": count / total}
            for index, count in enumerate(token_correct)
        ],
        "runtime": {
            "torch": str(torch.__version__),
            "cuda": str(torch.version.cuda),
            "device": str(fabric.device),
        },
    }
    atomic_write_json(output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--model", required=True, choices=sorted(MODEL_FLAGS))
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", required=True, help="run's materialized_config.yaml")
    ap.add_argument("--source-config", required=True, help="hash-bound generated project config")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args(argv)
    try:
        result = evaluate_checkpoint(
            job_id=args.job_id, model_name=args.model, seed=args.seed,
            checkpoint=pathlib.Path(args.checkpoint), config_path=pathlib.Path(args.config),
            source_config=pathlib.Path(args.source_config), dataset=pathlib.Path(args.dataset),
            manifests=[pathlib.Path(path) for path in args.manifest],
            upstream=pathlib.Path(args.upstream), output=pathlib.Path(args.output),
            batch_size=args.batch_size,
        )
    except (EvaluationError, OSError, ValueError) as exc:
        print(f"[evaluate_base_competence] REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
