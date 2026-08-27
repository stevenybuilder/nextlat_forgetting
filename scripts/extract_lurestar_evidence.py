#!/usr/bin/env python3
"""Restartable, hash-bound GPU extraction for reduced confirmatory Lure-Star analysis.

This is the only command in the analysis path that imports the pinned NextLat model code or
touches checkpoints.  It deliberately writes small, independently verified chunks before
assembling an evidence NPZ.  A preemption can therefore lose at most the in-flight chunk.

The command consumes a frozen JSON job rather than discovering files in a run directory.  Every
base config, base checkpoint, stimulus manifest, and permanent D40 H3 exclusion are named with
their SHA-256. H3 was prospectively and permanently dropped after the one-shot D40 feasibility
gate left four items unmatched. Consequently this command emits only the frozen H1/H2 ``E_score``
population. Adaptation checkpoints, near/mid/far arrays, mechanism probes, and H3 analysis inputs
are refused rather than ignored. This is a scientific boundary, not a compatibility mode.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import yaml

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from lurestar import evaluate as E  # noqa: E402
from lurestar import representations as R  # noqa: E402
from lurestar.validate import solve_line, token_ids  # noqa: E402


PINNED_UPSTREAM_COMMIT = "3770be6009cea2b3c455a9ce7f2ca88b504bb955"
JOB_SCHEMA = "nextlat_forgetting/lurestar_evidence_extraction_job/3"
PROGRESS_SCHEMA = "nextlat_forgetting/lurestar_evidence_progress/1"
EVIDENCE_SCHEMA = "nextlat_forgetting/lurestar_evidence/4"
H1_CONDITIONS = ("base", "repeat", "near_safe", "near_critical", "far_critical")
ARMS = ("nextlat", "bst", "gpt")
LOCAL_MEASUREMENT_SOURCE_PATHS = (
    "src/lurestar/representations.py",
    "src/lurestar/evaluate.py",
)
H3_BLOCK_PATH = (_REPO / "manifests" / "h3_selected" / "PERMANENT_H3_BLOCK.json").resolve()
H3_BLOCK_SHA256 = "82d526ad5cb6ac5fb942790488a6b766e59b816acb27ed405a00852f40925778"
H3_BLOCK_SIDECAR_SHA256 = "24b47f2d49e084d4b09e39393938294d7ef1a7ba6e5bbf90d56b4e7145a65d0b"
H3_BLOCK_FORBIDDEN = (
    "candidate_expansion", "caliper_change", "weighting", "unmatched_restriction",
    "pilot_substitution", "matching_amendment",
)
H3_BLOCK_DOCUMENT = {
    "combined_loss_sha256": "814058a162e12fde36c7204dd30798b63bfbf02294fce768046070672e5afece",
    "expanded_manifest_sha256": "2effd4e13d384786546c71cc61b4138dc97f082e3992bf3cdf398e6bf93264f1",
    "forbidden": list(H3_BLOCK_FORBIDDEN),
    "no_further_amendments_permitted": True,
    "reason": "D40_ONE_SHOT_EXPANSION_REMAINS_INFEASIBLE",
    "schema": "nextlat_forgetting/h3_mid_expansion/1",
    "status": "PERMANENT_H3_BLOCK",
    "unmatched_count": 4,
    "unmatched_identity_sha256":
        "ab4fb10a1e049912fb3e24046cf1498b1027e489864e076d91c10044cef82bf6",
}


class ExtractionRefused(RuntimeError):
    """Raised when provenance or a frozen scientific identity cannot be verified."""


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _atomic_bytes(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: pathlib.Path, value: Any) -> None:
    _atomic_bytes(path, _canonical_json(value))


def _atomic_npz(path: pathlib.Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve_record(record: Any, *, base: pathlib.Path, label: str) -> dict[str, str]:
    if not isinstance(record, Mapping):
        raise ExtractionRefused(f"{label} must be a path/SHA-256 record")
    raw_path, digest = record.get("path"), record.get("sha256")
    if not isinstance(raw_path, str) or not _is_sha(digest):
        raise ExtractionRefused(f"{label} has an invalid path or SHA-256")
    path = pathlib.Path(raw_path)
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.is_file():
        raise ExtractionRefused(f"{label} does not exist: {path}")
    actual = sha256_file(path)
    if actual != digest:
        raise ExtractionRefused(f"{label} SHA-256 mismatch: expected {digest}, got {actual}")
    return {"path": str(path), "sha256": digest}


def _load_jsonl(path: pathlib.Path, *, label: str) -> list[dict]:
    rows: list[dict] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ExtractionRefused(f"{label}:{number} is invalid JSON") from exc
        if not isinstance(row, dict):
            raise ExtractionRefused(f"{label}:{number} must be a JSON object")
        rows.append(row)
    if not rows:
        raise ExtractionRefused(f"{label} is empty")
    return rows


def _merge(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict:
    out = dict(left)
    for key, value in right.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass(frozen=True)
class BoundJob:
    path: pathlib.Path
    digest: str
    payload: dict
    arm: str
    seed: int
    upstream: pathlib.Path
    configs: dict[str, dict[str, str]]
    checkpoints: dict[str, dict[str, str]]
    inputs: dict[str, dict[str, str]]
    h3_permanent_block: dict[str, str]


def _verify_h3_permanent_block(record: Any, *, base: pathlib.Path) -> dict[str, str]:
    """Verify the exact canonical D40 block and its independently hashed sidecar."""
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256", "sidecar"}:
        raise ExtractionRefused("permanent H3 block record must bind path, SHA-256, and sidecar")
    if not isinstance(record["sidecar"], Mapping) or set(record["sidecar"]) != {"path", "sha256"}:
        raise ExtractionRefused("permanent H3 block sidecar record must bind only path and SHA-256")
    bound = _resolve_record(record, base=base, label="canonical permanent H3 block")
    path = pathlib.Path(bound["path"])
    if path != H3_BLOCK_PATH or bound["sha256"] != H3_BLOCK_SHA256:
        raise ExtractionRefused("job must bind the exact canonical permanent H3 block")
    sidecar_record = record.get("sidecar") if isinstance(record, Mapping) else None
    sidecar = _resolve_record(sidecar_record, base=base, label="permanent H3 block sidecar")
    sidecar_path = pathlib.Path(sidecar["path"])
    if sidecar_path != pathlib.Path(f"{H3_BLOCK_PATH}.sha256"):
        raise ExtractionRefused("job must bind the canonical permanent H3 block sidecar")
    if sidecar["sha256"] != H3_BLOCK_SIDECAR_SHA256:
        raise ExtractionRefused("canonical permanent H3 block sidecar hash changed")
    fields = sidecar_path.read_text(encoding="utf-8").strip().split()
    if fields != [H3_BLOCK_SHA256, H3_BLOCK_PATH.name]:
        raise ExtractionRefused("permanent H3 block sidecar content or filename binding changed")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExtractionRefused("permanent H3 block is invalid JSON") from exc
    if document != H3_BLOCK_DOCUMENT:
        raise ExtractionRefused("permanent H3 block semantics changed")
    return {
        "path": str(path), "sha256": H3_BLOCK_SHA256,
        "sidecar_path": str(sidecar_path), "sidecar_sha256": sidecar["sha256"],
    }


def load_job(path: os.PathLike[str] | str) -> BoundJob:
    job_path = pathlib.Path(path).resolve()
    try:
        payload = json.loads(job_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExtractionRefused("extraction job is unreadable") from exc
    if payload.get("schema") != JOB_SCHEMA:
        raise ExtractionRefused(f"job schema must be {JOB_SCHEMA}")
    allowed_job_fields = {
        "schema", "arm", "seed", "upstream_commit", "upstream_path",
        "upstream_source_sha256", "local_measurement_source_sha256",
        "configs", "checkpoints", "frozen_inputs",
        "h3_permanent_block", "extraction",
    }
    if set(payload) != allowed_job_fields:
        raise ExtractionRefused(
            "reduced H1/H2 job has missing or unexpected fields; H3 attempts are forbidden"
        )
    arm = str(payload.get("arm", "")).lower()
    seed = payload.get("seed")
    if arm not in ARMS or not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ExtractionRefused("job needs a known arm and nonnegative integer seed")
    if payload.get("upstream_commit") != PINNED_UPSTREAM_COMMIT:
        raise ExtractionRefused("job does not bind the pinned NextLat commit")
    base = job_path.parent
    upstream_raw = payload.get("upstream_path", "upstream/NextLat")
    upstream = pathlib.Path(str(upstream_raw))
    if not upstream.is_absolute():
        upstream = base / upstream
    upstream = upstream.resolve()
    if not (upstream / "models" / "model_gpt.py").is_file():
        raise ExtractionRefused(f"pinned NextLat integration is absent: {upstream}")

    configs_raw = payload.get("configs")
    checkpoints_raw = payload.get("checkpoints")
    inputs_raw = payload.get("frozen_inputs")
    if not all(isinstance(value, Mapping) for value in (configs_raw, checkpoints_raw, inputs_raw)):
        raise ExtractionRefused("configs, checkpoints and frozen_inputs must be objects")
    if set(configs_raw) != {"base"} or set(checkpoints_raw) != {"base"}:
        raise ExtractionRefused("reduced H1/H2 job permits only the base config and checkpoint")
    for label, record in (
        ("base config", configs_raw["base"]),
        ("base checkpoint", checkpoints_raw["base"]),
        ("E_lure frozen input", inputs_raw.get("e_lure")),
    ):
        if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
            raise ExtractionRefused(f"{label} record must bind only path and SHA-256")
    branch_names = ("base",)
    configs = {
        name: _resolve_record(configs_raw.get(name), base=base, label=f"{name} config")
        for name in branch_names
    }
    checkpoints = {
        name: _resolve_record(checkpoints_raw.get(name), base=base, label=f"{name} checkpoint")
        for name in branch_names
    }
    required_inputs = ("e_lure",)
    if set(inputs_raw) != set(required_inputs):
        raise ExtractionRefused("reduced H1/H2 job permits only the E_lure frozen input")
    inputs = {
        name: _resolve_record(inputs_raw.get(name), base=base, label=f"frozen input {name}")
        for name in required_inputs
    }
    block = _verify_h3_permanent_block(payload.get("h3_permanent_block"), base=base)
    extraction = payload.get("extraction")
    if not isinstance(extraction, Mapping):
        raise ExtractionRefused("job lacks frozen extraction policy")
    frozen = {
        "whitener_count": 400,
        "scored_count": 1600,
    }
    if set(extraction) != set(frozen):
        raise ExtractionRefused("reduced H1/H2 extraction policy has unexpected or missing fields")
    for key, expected in frozen.items():
        if extraction.get(key) != expected:
            raise ExtractionRefused(f"extraction.{key} must be frozen to {expected!r}")
    local_sources = payload.get("local_measurement_source_sha256")
    if not isinstance(local_sources, Mapping) or set(local_sources) != set(
        LOCAL_MEASUREMENT_SOURCE_PATHS
    ):
        raise ExtractionRefused(
            "job must bind exactly the local representations.py and evaluate.py measurement sources"
        )
    for relative in LOCAL_MEASUREMENT_SOURCE_PATHS:
        digest = local_sources.get(relative)
        source = (_REPO / relative).resolve()
        if not _is_sha(digest) or not source.is_file() or sha256_file(source) != digest:
            raise ExtractionRefused(f"local measurement source identity failed for {relative}")
    return BoundJob(
        path=job_path, digest=sha256_file(job_path), payload=payload, arm=arm, seed=seed,
        upstream=upstream, configs=configs, checkpoints=checkpoints, inputs=inputs,
        h3_permanent_block=block,
    )


def verify_upstream(job: BoundJob) -> None:
    """Refuse a different checkout when git metadata exists; bind source hashes always."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(job.upstream), "rev-parse", "HEAD"],
            text=True, capture_output=True, check=False, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        proc = None
    if proc is not None and proc.returncode == 0 and proc.stdout.strip() != PINNED_UPSTREAM_COMMIT:
        raise ExtractionRefused(
            f"NextLat checkout is {proc.stdout.strip()}, expected {PINNED_UPSTREAM_COMMIT}"
        )
    expected = job.payload.get("upstream_source_sha256")
    if not isinstance(expected, Mapping):
        raise ExtractionRefused("job must bind upstream_source_sha256")
    for relative in (
        "data/stargraph.py", "models/model_base.py", "models/model_gpt.py",
        "models/model_nextlat.py", "models/model_bst.py",
    ):
        digest = expected.get(relative)
        source = job.upstream / relative
        if not _is_sha(digest) or not source.is_file() or sha256_file(source) != digest:
            raise ExtractionRefused(f"pinned upstream source identity failed for {relative}")


def _load_materialized_config(path: pathlib.Path, *, upstream: pathlib.Path,
                              arm: str, seed: int) -> dict:
    defaults_path = upstream / "defaults.yaml"
    defaults = yaml.safe_load(defaults_path.read_text(encoding="utf-8"))
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ExtractionRefused(f"config is not a mapping: {path}")
    config = _merge(defaults, body)
    if int(config.get("seed", -1)) != seed:
        raise ExtractionRefused(f"config seed does not match job seed {seed}")
    expected_flags = {
        "gpt": (False, False), "nextlat": (False, True), "bst": (True, False),
    }[arm]
    if (bool(config.get("use_bst")), bool(config.get("use_nextlat"))) != expected_flags:
        raise ExtractionRefused(f"config model flags do not identify arm {arm}")
    model = config["model"]
    for key, expected in (("n_layer", 12), ("n_head", 6), ("n_embd", 384)):
        if int(model.get(key, -1)) != expected:
            raise ExtractionRefused(f"config {key} must remain paper-scale {expected}")
    # These are deterministically set by StarGraphDataModule.update_config for G(5,5).
    model["vocab_size"], model["block_size"], model["context_length"] = 106, 69, 62
    return config


def _import_models(upstream: pathlib.Path):
    upstream_text = str(upstream)
    if upstream_text not in sys.path:
        sys.path.insert(0, upstream_text)
    return (
        importlib.import_module("models.model_gpt"),
        importlib.import_module("models.model_nextlat"),
        importlib.import_module("models.model_bst"),
        importlib.import_module("data.stargraph"),
    )


def _model_args(config: Mapping[str, Any], arm: str) -> dict:
    model, trainer, data = config["model"], config["trainer"], config["data"]
    args = {
        "n_layer": int(model["n_layer"]), "n_head": int(model["n_head"]),
        "n_embd": int(model["n_embd"]), "block_size": int(model["block_size"]),
        "bias": bool(model["bias"]), "vocab_size": int(model["vocab_size"]),
        "dropout": float(model["dropout"]), "eos_token_id": R.EOS_TOKEN_ID,
        "use_fused": bool(trainer.get("use_fused_kernels", False)),
        "context_length": int(model["context_length"]),
    }
    if arm == "gpt":
        args.update(goal_range=data.get("goal_range", [25, 75]), fim_token_id=-1,
                    is_fim_mode=False,
                    compute_hidden_state_rank=bool(model.get("compute_hidden_state_rank", False)))
    elif arm == "nextlat":
        for key in ("mtp_horizon", "lambda_kl", "lambda_mse", "lambda_ce", "proj_factor",
                    "compute_hidden_state_rank"):
            args[key] = model[key]
    else:
        for key in ("bst_pair_minimum_gap", "bst_pair_maximum_gap", "bst_pair_subsample_rate",
                    "bst_single_gap_prediction_mode"):
            args[key] = model[key]
    return args


def load_model(job: BoundJob, *, device: str):
    """Instantiate the actual pinned arm from the job's sole base checkpoint."""
    torch = importlib.import_module("torch")
    gpt_mod, nextlat_mod, bst_mod, _stargraph = _import_models(job.upstream)
    config = _load_materialized_config(
        pathlib.Path(job.configs["base"]["path"]), upstream=job.upstream,
        arm=job.arm, seed=job.seed
    )
    classes = {
        "gpt": (gpt_mod.GPT, gpt_mod.GPTConfig),
        "nextlat": (nextlat_mod.NextLat, nextlat_mod.NextLatConfig),
        "bst": (bst_mod.BST, bst_mod.BSTConfig),
    }
    wrapper_cls, config_cls = classes[job.arm]
    wrapper = wrapper_cls(config_cls(**_model_args(config, job.arm)))
    raw = torch.load(job.checkpoints["base"]["path"], map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping):
        raise ExtractionRefused("base checkpoint payload is not a mapping")
    try:
        if job.arm == "bst":
            if not isinstance(raw.get("encoder"), Mapping) or not isinstance(raw.get("text_head"), Mapping):
                raise ExtractionRefused("BST checkpoint lacks encoder/text_head state dictionaries")
            wrapper.encoder.load_state_dict(raw["encoder"], strict=True)
            wrapper.text_head.load_state_dict(raw["text_head"], strict=True)
        else:
            if not isinstance(raw.get("model"), Mapping):
                raise ExtractionRefused(f"{job.arm} checkpoint lacks model state dictionary")
            wrapper.model.load_state_dict(raw["model"], strict=True)
    except RuntimeError as exc:
        raise ExtractionRefused("base checkpoint is incompatible with its bound config") from exc
    if job.arm == "bst":
        wrapper.encoder.to(device).eval()
        wrapper.text_head.to(device).eval()
    else:
        wrapper.model.to(device).eval()
    return wrapper, config


def load_tokenizer(job: BoundJob):
    """Use the actual upstream tokenizer and prove it agrees with the local frozen oracle."""
    *_models, stargraph = _import_models(job.upstream)
    tokenizer = stargraph.Tokenizer(100)
    probe = "1,2|2,3/1,3=1,2,3"
    if tokenizer.encode(probe) != token_ids(probe):
        raise ExtractionRefused("upstream stargraph tokenization disagrees with the frozen oracle")
    if tokenizer.eos_token_id != R.EOS_TOKEN_ID:
        raise ExtractionRefused("upstream stargraph EOS identity changed")
    return tokenizer


def tokenize_lines(tokenizer, lines: Sequence[str]) -> np.ndarray:
    rows = [np.asarray(tokenizer.tokenize(line)[0], dtype=np.int64) for line in lines]
    lengths = {row.size for row in rows}
    if lengths != {69}:
        raise ExtractionRefused(f"G(5,5) token rows must all have length 69, got {sorted(lengths)}")
    tokens = np.stack(rows)
    R.resolve_extraction_indices(tokens)
    return tokens


def _branch_targets(lines: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    correct, competitors = [], []
    for line in lines:
        solved = solve_line(line)
        correct.append(solved.path[1])
        competitors.append([arm[0] for arm in solved.arms[1:]])
    return np.asarray(correct, dtype=np.int64), np.asarray(competitors, dtype=np.int64)


def _inner(wrapper, arm: str):
    return wrapper if arm == "bst" else wrapper.model


def _extract(wrapper, arm: str, tokens: np.ndarray, *, batch_size: int, device: str) -> dict:
    return R.extract_positions(
        _inner(wrapper, arm), tokens, architecture=arm, batch_size=batch_size, device=device,
        capture_blocks=True,
    )


def _margin(logits: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    correct, competitors = _branch_targets(lines)
    return R.branch_margin(logits[:, 1, :], correct, competitors)


def _greedy_exact_path(wrapper, arm: str, tokens: np.ndarray, *, batch_size: int,
                       device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generated/true five-token paths and exact indicator using explicit argmax."""
    torch = importlib.import_module("torch")
    values = np.asarray(tokens, dtype=np.int64)
    predicted = []
    for start in range(0, values.shape[0], batch_size):
        generated = torch.tensor(
            values[start:start + batch_size, :63].tolist(), dtype=torch.long, device=device
        )
        for _step in range(5):
            out = R.forward_all_states(_inner(wrapper, arm), generated, architecture=arm)
            next_token = torch.argmax(out["logits"][:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        tail = generated[:, 63:68].detach().cpu()
        try:
            predicted.append(tail.numpy())
        except RuntimeError:
            predicted.append(np.asarray(tail.tolist(), dtype=np.int64))
    generated_tail = np.concatenate(predicted, axis=0)
    true_tail = values[:, 63:68]
    indicator = E.exact_path_accuracy(generated_tail, true_tail)
    return generated_tail, true_tail, indicator




class ChunkStore:
    """Atomic, restartable chunks bound to one immutable extraction job."""

    def __init__(self, root: os.PathLike[str] | str, *, job_sha256: str):
        self.root = pathlib.Path(root).resolve()
        self.job_sha256 = job_sha256
        self.root.mkdir(parents=True, exist_ok=True)
        identity = self.root / "job.json"
        if identity.exists():
            try:
                prior = json.loads(identity.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ExtractionRefused("progress job identity is corrupt") from exc
            if prior != {"schema": PROGRESS_SCHEMA, "job_sha256": job_sha256}:
                raise ExtractionRefused("progress directory belongs to a different extraction job")
        else:
            _atomic_json(identity, {"schema": PROGRESS_SCHEMA, "job_sha256": job_sha256})

    def paths(self, group: str, start: int, stop: int) -> tuple[pathlib.Path, pathlib.Path]:
        stem = f"{group}-{start:06d}-{stop:06d}"
        return self.root / f"{stem}.npz", self.root / f"{stem}.json"

    def load(self, group: str, start: int, stop: int) -> dict[str, np.ndarray] | None:
        path, receipt_path = self.paths(group, start, stop)
        if not path.is_file() or not receipt_path.is_file():
            return None
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        expected = {
            "schema": PROGRESS_SCHEMA, "job_sha256": self.job_sha256,
            "group": group, "start": start, "stop": stop,
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ExtractionRefused(f"stale or cross-job chunk receipt: {receipt_path}")
        if receipt.get("npz_sha256") != sha256_file(path):
            raise ExtractionRefused(f"completed chunk fails SHA-256: {path}")
        with np.load(path, allow_pickle=False) as z:
            arrays = {name: np.asarray(z[name]) for name in z.files}
        if not arrays or any(value.shape[0] != stop - start for value in arrays.values()):
            raise ExtractionRefused(f"chunk arrays do not have their bound row count: {path}")
        return arrays

    def produce(
        self, group: str, start: int, stop: int,
        producer: Callable[[], Mapping[str, Any]],
    ) -> dict[str, np.ndarray]:
        existing = self.load(group, start, stop)
        if existing is not None:
            return existing
        arrays = {key: np.asarray(value) for key, value in producer().items()}
        if not arrays or any(value.shape[0] != stop - start for value in arrays.values()):
            raise ExtractionRefused(f"producer for {group} returned a wrong-sized array")
        path, receipt_path = self.paths(group, start, stop)
        _atomic_npz(path, arrays)
        _atomic_json(receipt_path, {
            "schema": PROGRESS_SCHEMA, "job_sha256": self.job_sha256,
            "group": group, "start": start, "stop": stop,
            "npz_sha256": sha256_file(path), "arrays": sorted(arrays),
        })
        return arrays


def _chunks(n: int, size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, n, size):
        yield start, min(start + size, n)


def _concat(chunks: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not chunks:
        raise ExtractionRefused("cannot concatenate zero chunks")
    keys = set(chunks[0])
    if any(set(chunk) != keys for chunk in chunks):
        raise ExtractionRefused("chunk array schemas disagree")
    return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in keys}


def _h1_chunk(job: BoundJob, model, tokenizer, rows: Sequence[dict], *, batch_size: int,
              device: str) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    ids = []
    for row in rows:
        conditions = row.get("conditions")
        if not isinstance(conditions, Mapping) or any(name not in conditions for name in H1_CONDITIONS):
            raise ExtractionRefused("E_lure row lacks the complete five-condition quartet")
        base_key = conditions["base"].get("graph_key")
        if not _is_sha(base_key):
            raise ExtractionRefused("E_lure base graph_key is not a SHA-256 identity")
        ids.append(base_key)
    result["h1_item_ids"] = np.asarray(ids, dtype="U64")
    for condition in H1_CONDITIONS:
        lines = [str(row["conditions"][condition]["line"]) for row in rows]
        tokens = tokenize_lines(tokenizer, lines)
        out = _extract(model, job.arm, tokens, batch_size=batch_size, device=device)
        result[f"{condition}_hidden_psi"] = out["hidden"][:, 0, :]
        result[f"{condition}_hidden_branch"] = out["hidden"][:, 1, :]
        intermediate = np.asarray(out.get("intermediate_hidden"))
        if intermediate.ndim != 4 or intermediate.shape[1:3] != (12, 2):
            raise ExtractionRefused("mandatory intermediate extraction is not N x 12 x 2 x D")
        result[f"{condition}_hidden_intermediate"] = intermediate
        result[f"{condition}_logits_branch"] = out["logits"][:, 1, :]
        result[f"{condition}_margin"] = _margin(out["logits"], lines)
        correct, _competitors = _branch_targets(lines)
        result[f"{condition}_first_branch_accuracy"] = E.first_branch_accuracy(
            out["logits"][:, 1, :], correct
        )
        generated_path, true_path, exact_accuracy = _greedy_exact_path(
            model, job.arm, tokens, batch_size=batch_size, device=device
        )
        result[f"{condition}_generated_path"] = generated_path
        result[f"{condition}_true_path"] = true_path
        result[f"{condition}_exact_path_accuracy"] = exact_accuracy
        if job.arm == "bst":
            if "hidden_texthead" not in out:
                raise ExtractionRefused("BST extraction omitted the mandatory TextHead state")
            result[f"{condition}_hidden_texthead"] = out["hidden_texthead"][:, 0, :]
    return result


def _identity_sha(values: Sequence[Any]) -> str:
    """Match evaluator v2: one ordered UTF-8 identity per LF-terminated row."""
    strings = [str(value) for value in values]
    if any("\n" in value or "\r" in value for value in strings):
        raise ExtractionRefused("identity values may not contain CR/LF")
    payload = "".join(value + "\n" for value in strings).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _fit_source_sha(states: np.ndarray, row_ids: Sequence[str]) -> str:
    """Canonical content hash of the exact floating-point rows used to fit a whitener."""
    values = np.ascontiguousarray(np.asarray(states, dtype="<f8"))
    normalized_ids = [str(value) for value in row_ids]
    if len(normalized_ids) != values.shape[0]:
        raise ExtractionRefused("whitener fit-source row IDs do not align with fit rows")
    header = _canonical_json({
        "dtype": "float64-le", "shape": list(values.shape),
        "ordered_row_ids": normalized_ids,
    })
    return hashlib.sha256(header + values.tobytes(order="C")).hexdigest()


def _whitener_fields(prefix: str, whitener: R.Whitener, calibration_ids: Sequence[str],
                     calibration_base_ids: Sequence[str], fit_states: np.ndarray) -> dict[str, np.ndarray]:
    report = whitener.report()
    ids = np.asarray([str(value) for value in calibration_ids], dtype="U129")
    if ids.size != whitener.n_pool or len(set(ids.tolist())) != ids.size:
        raise ExtractionRefused(f"{prefix} whitener calibration identities are not one-to-one")
    return {
        f"{prefix}_shrinkage": np.asarray(report["shrinkage"], dtype=np.float64),
        f"{prefix}_shrinkage_rule": np.asarray("ledoit_wolf_with_1e-3_floor"),
        f"{prefix}_condition_number": np.asarray(
            report["condition_number"], dtype=np.float64
        ),
        f"{prefix}_calibration_ids": ids,
        f"{prefix}_calibration_ids_sha256": np.asarray(_identity_sha(ids.tolist())),
        f"{prefix}_calibration_base_ids": np.asarray(calibration_base_ids, dtype="U64"),
        f"{prefix}_calibration_base_ids_sha256": np.asarray(
            _identity_sha(calibration_base_ids)
        ),
        f"{prefix}_fit_source_sha256": np.asarray(_fit_source_sha(fit_states, ids.tolist())),
        f"{prefix}_n_pool": np.asarray(whitener.n_pool, dtype=np.int64),
        f"{prefix}_n_features": np.asarray(whitener.n_features, dtype=np.int64),
    }


def _secondary_distances(
    raw: Mapping[str, np.ndarray], *, state_suffix: str, white: np.ndarray,
    scored: np.ndarray, score_ids: np.ndarray, whitener_prefix: str,
) -> dict[str, np.ndarray]:
    states = {condition: raw[f"{condition}_{state_suffix}"] for condition in H1_CONDITIONS}
    pool = R.CenteringPool.from_conditions(**states)
    fit_states = np.vstack([states[condition][white] for condition in H1_CONDITIONS])
    fit_ids = [
        f"{raw['h1_item_ids'][index]}:{condition}"
        for condition in H1_CONDITIONS for index in white
    ]
    calibration_base_ids = [str(raw["h1_item_ids"][index]) for index in white]
    fit_group_ids = [
        str(raw["h1_item_ids"][index])
        for _condition in H1_CONDITIONS for index in white
    ]
    whitener = R.Whitener.fit(
        fit_states, item_ids=fit_ids, group_ids=fit_group_ids
    )
    centered = E.psi_distances_centered_cosine(
        states["base"][scored], states["near_critical"][scored],
        states["near_safe"][scored], centering_pool=pool,
    )
    whitened = E.psi_distances_whitened(
        states["base"][scored], states["near_critical"][scored],
        states["near_safe"][scored], whitener=whitener, item_ids=score_ids.tolist(),
        group_ids=score_ids.tolist(),
    )
    return {
        "d_critical_centered_cosine": centered["d_critical"],
        "d_safe_centered_cosine": centered["d_safe"],
        "d_critical_whitened": whitened["d_critical"],
        "d_safe_whitened": whitened["d_safe"],
        **_whitener_fields(
            whitener_prefix, whitener, fit_ids, calibration_base_ids, fit_states
        ),
    }


def _intermediate_distances(
    raw: Mapping[str, np.ndarray], *, white: np.ndarray, scored: np.ndarray,
    score_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    n_score = scored.size
    centered_critical = np.empty((12, 2, n_score), dtype=np.float64)
    centered_safe = np.empty_like(centered_critical)
    whitened_critical = np.empty_like(centered_critical)
    whitened_safe = np.empty_like(centered_critical)
    shrinkage = np.empty((12, 2), dtype=np.float64)
    condition_number = np.empty((12, 2), dtype=np.float64)
    fit_source = np.empty((12, 2), dtype="U64")
    fit_ids_sha256 = np.empty((12, 2), dtype="U64")
    n_features = np.empty((12, 2), dtype=np.int64)
    fit_dtype = np.empty((12, 2), dtype="U16")
    fit_shape = np.empty((12, 2, 2), dtype=np.int64)
    shrinkage_rule = np.empty((12, 2), dtype="U32")
    ids = np.asarray(raw["h1_item_ids"])
    calibration_base_ids = [str(ids[index]) for index in white]
    fit_ids = [f"{ids[index]}:{condition}" for condition in H1_CONDITIONS for index in white]
    fit_ids_array = np.asarray(fit_ids, dtype="U80")
    fit_group_ids = [
        str(ids[index]) for _condition in H1_CONDITIONS for index in white
    ]
    for block in range(12):
        for position in range(2):
            states = {
                condition: np.asarray(raw[f"{condition}_hidden_intermediate"])[
                    :, block, position, :
                ]
                for condition in H1_CONDITIONS
            }
            pool = R.CenteringPool.from_conditions(**states)
            fit_states = np.vstack([states[condition][white] for condition in H1_CONDITIONS])
            whitener = R.Whitener.fit(
                fit_states, item_ids=fit_ids, group_ids=fit_group_ids
            )
            centered = E.psi_distances_centered_cosine(
                states["base"][scored], states["near_critical"][scored],
                states["near_safe"][scored], centering_pool=pool,
            )
            whitened = E.psi_distances_whitened(
                states["base"][scored], states["near_critical"][scored],
                states["near_safe"][scored], whitener=whitener,
                item_ids=score_ids.tolist(), group_ids=score_ids.tolist(),
            )
            centered_critical[block, position] = centered["d_critical"]
            centered_safe[block, position] = centered["d_safe"]
            whitened_critical[block, position] = whitened["d_critical"]
            whitened_safe[block, position] = whitened["d_safe"]
            shrinkage[block, position] = whitener.shrinkage
            condition_number[block, position] = whitener.condition_number
            fit_source[block, position] = _fit_source_sha(fit_states, fit_ids)
            fit_ids_sha256[block, position] = _identity_sha(fit_ids)
            n_features[block, position] = whitener.n_features
            # _fit_source_sha canonicalizes these exact values to little-endian float64.
            fit_dtype[block, position] = "float64-le"
            fit_shape[block, position] = np.asarray(fit_states.shape, dtype=np.int64)
            shrinkage_rule[block, position] = "ledoit_wolf_with_1e-3_floor"
    return {
        "secondary_intermediate_blocks": np.arange(12, dtype=np.int64),
        "secondary_intermediate_positions": np.asarray([62, 63], dtype=np.int64),
        "secondary_intermediate_d_critical_centered_cosine": centered_critical,
        "secondary_intermediate_d_safe_centered_cosine": centered_safe,
        "secondary_intermediate_d_critical_whitened": whitened_critical,
        "secondary_intermediate_d_safe_whitened": whitened_safe,
        "secondary_intermediate_whitener_shrinkage": shrinkage,
        "secondary_intermediate_whitener_condition_number": condition_number,
        "secondary_intermediate_whitener_fit_source_sha256": fit_source,
        "secondary_intermediate_whitener_fit_ids": fit_ids_array,
        "secondary_intermediate_whitener_fit_ids_sha256": fit_ids_sha256,
        "secondary_intermediate_whitener_n_features": n_features,
        "secondary_intermediate_whitener_fit_dtype": fit_dtype,
        "secondary_intermediate_whitener_fit_shape": fit_shape,
        "secondary_intermediate_whitener_shrinkage_rule": shrinkage_rule,
        "secondary_intermediate_calibration_base_ids": np.asarray(
            calibration_base_ids, dtype="U64"
        ),
        "secondary_intermediate_calibration_base_ids_sha256": np.asarray(
            _identity_sha(calibration_base_ids)
        ),
        "secondary_intermediate_status": np.asarray(
            "AVAILABLE_ALL_BLOCKS_0_11_PRE_FINAL_NORM_POSITIONS_62_63"
        ),
    }


def _h1_finalize(raw: Mapping[str, np.ndarray], *, whitener_count: int,
                 scored_count: int) -> tuple[dict[str, np.ndarray], R.CenteringPool, R.Whitener]:
    ids = np.asarray(raw["h1_item_ids"])
    if len(set(ids.tolist())) != ids.size or ids.size != whitener_count + scored_count:
        raise ExtractionRefused("H1 identities do not form the frozen 400/1,600 split")
    order = np.argsort(ids, kind="stable")
    white, scored = order[:whitener_count], order[whitener_count:]
    pool = R.CenteringPool.from_conditions(**{
        condition: raw[f"{condition}_hidden_psi"] for condition in H1_CONDITIONS
    })
    fit_states = np.vstack([
        raw[f"{condition}_hidden_psi"][white] for condition in H1_CONDITIONS
    ])
    fit_ids = [f"{ids[index]}:{condition}" for condition in H1_CONDITIONS for index in white]
    calibration_base_ids = [str(ids[index]) for index in white]
    fit_group_ids = [
        str(ids[index]) for _condition in H1_CONDITIONS for index in white
    ]
    whitener = R.Whitener.fit(fit_states, item_ids=fit_ids, group_ids=fit_group_ids)
    score_ids = ids[scored]
    centered = E.psi_distances_centered_cosine(
        raw["base_hidden_psi"][scored], raw["near_critical_hidden_psi"][scored],
        raw["near_safe_hidden_psi"][scored], centering_pool=pool,
    )
    whitened = E.psi_distances_whitened(
        raw["base_hidden_psi"][scored], raw["near_critical_hidden_psi"][scored],
        raw["near_safe_hidden_psi"][scored], whitener=whitener,
        item_ids=score_ids.tolist(),
        group_ids=score_ids.tolist(),
    )
    secondary_index63 = _secondary_distances(
        raw, state_suffix="hidden_branch", white=white, scored=scored,
        score_ids=score_ids, whitener_prefix="whitener",
    )
    intermediate = _intermediate_distances(
        raw, white=white, scored=scored, score_ids=score_ids
    )
    base_scored = raw["base_hidden_psi"][scored]
    critical_scored = raw["near_critical_hidden_psi"][scored]
    safe_scored = raw["near_safe_hidden_psi"][scored]
    evidence = {
        "h1_item_ids": score_ids,
        "h1_item_ids_sha256": np.asarray(_identity_sha(score_ids.tolist())),
        "d_critical": centered["d_critical"], "d_safe": centered["d_safe"],
        "d_repeat": R.centered_cosine_distance(
            raw["base_hidden_psi"][scored], raw["repeat_hidden_psi"][scored], mean=pool.mean
        ),
        "d_critical_whitened": whitened["d_critical"],
        "d_safe_whitened": whitened["d_safe"],
        "critical_margin": raw["near_critical_margin"][scored],
        "base_margin": raw["base_margin"][scored],
        **{
            f"behavior_{condition}_first_branch_accuracy": raw[
                f"{condition}_first_branch_accuracy"
            ][scored]
            for condition in H1_CONDITIONS
        },
        **{
            f"behavior_{condition}_exact_path_accuracy": raw[
                f"{condition}_exact_path_accuracy"
            ][scored]
            for condition in H1_CONDITIONS
        },
        **{
            f"behavior_{condition}_generated_path": raw[f"{condition}_generated_path"][scored]
            for condition in H1_CONDITIONS
        },
        **{
            f"behavior_{condition}_true_path": raw[f"{condition}_true_path"][scored]
            for condition in H1_CONDITIONS
        },
        "npsi": np.asarray(E.normalized_psi(centered["d_critical"], centered["d_safe"])[0]),
        "npsi_whitened": np.asarray(E.normalized_psi(
            whitened["d_critical"], whitened["d_safe"]
        )[0]),
        "secondary_raw_cosine_d_critical": R.cosine_distance_raw(
            base_scored, critical_scored
        ),
        "secondary_raw_cosine_d_safe": R.cosine_distance_raw(base_scored, safe_scored),
        "secondary_uncentered_euclidean_d_critical": np.linalg.norm(
            base_scored - critical_scored, axis=1
        ),
        "secondary_uncentered_euclidean_d_safe": np.linalg.norm(
            base_scored - safe_scored, axis=1
        ),
        "secondary_exact_path_status": np.asarray("AVAILABLE_EXPLICIT_ARGMAX_5_TOKENS"),
        **_whitener_fields(
            "whitener", whitener, fit_ids, calibration_base_ids, fit_states
        ),
        **{
            f"secondary_index63_{key}": value for key, value in secondary_index63.items()
        },
        **intermediate,
    }
    texthead_keys = [f"{condition}_hidden_texthead" for condition in H1_CONDITIONS]
    if all(key in raw for key in texthead_keys):
        secondary_texthead = _secondary_distances(
            raw, state_suffix="hidden_texthead", white=white, scored=scored,
            score_ids=score_ids, whitener_prefix="whitener",
        )
        evidence.update({
            "secondary_bst_texthead_status": np.asarray("AVAILABLE_BST_ONLY"),
            **{
                f"secondary_bst_texthead_{key}": value
                for key, value in secondary_texthead.items()
            },
        })
    elif any(key in raw for key in texthead_keys):
        raise ExtractionRefused("BST TextHead secondary is only partially present")
    else:
        evidence["secondary_bst_texthead_status"] = np.asarray("NOT_APPLICABLE_NON_BST")
    return evidence, pool, whitener


def assemble_evidence(job: BoundJob, h1_raw: Mapping[str, np.ndarray],
                      output: pathlib.Path) -> dict:
    """Assemble the exact H1/H2 evidence schema; H3 arrays have no accepted call surface."""
    extraction = job.payload["extraction"]
    h1, _centering_pool, _whitener = _h1_finalize(
        h1_raw, whitener_count=int(extraction["whitener_count"]),
        scored_count=int(extraction["scored_count"]),
    )
    arrays = {
        "evidence_schema": np.asarray(EVIDENCE_SCHEMA), "arm": np.asarray(job.arm),
        "seed": np.asarray(job.seed), "base_checkpoint_sha256": np.asarray(job.checkpoints["base"]["sha256"]),
        "h3_permanent_block_sha256": np.asarray(job.h3_permanent_block["sha256"]),
        "h3_permanent_block_sidecar_sha256": np.asarray(
            job.h3_permanent_block["sidecar_sha256"]
        ),
        "local_representations_sha256": np.asarray(
            job.payload["local_measurement_source_sha256"][
                "src/lurestar/representations.py"
            ]
        ),
        "local_evaluate_sha256": np.asarray(
            job.payload["local_measurement_source_sha256"]["src/lurestar/evaluate.py"]
        ),
        **h1,
    }
    for key, value in arrays.items():
        if np.asarray(value).dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise ExtractionRefused(f"final evidence contains non-finite {key}")
    _atomic_npz(output, arrays)
    identity_domains = {
        "h1_quartet": {"count": int(h1["h1_item_ids"].size),
                       "item_ids_sha256": str(h1["h1_item_ids_sha256"])},
    }
    receipt = {
        "schema": EVIDENCE_SCHEMA, "job": {"path": str(job.path), "sha256": job.digest},
        "arm": job.arm, "seed": job.seed, "checkpoints": job.checkpoints,
        "configs": job.configs, "frozen_inputs": job.inputs,
        "h3_permanent_block": job.h3_permanent_block,
        "local_measurement_sources": {
            relative: {
                "path": str((_REPO / relative).resolve()),
                "sha256": job.payload["local_measurement_source_sha256"][relative],
            }
            for relative in LOCAL_MEASUREMENT_SOURCE_PATHS
        },
        "evidence": {"path": str(output.resolve()), "sha256": sha256_file(output)},
        "population_counts": {"h1": int(h1["h1_item_ids"].size), "h3": 0},
        "identity_domains": identity_domains,
        "excluded": {
            "h3": True, "adaptation_checkpoints": True, "mechanism_probes": True,
            "h3_analysis": True,
        },
    }
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    _atomic_json(receipt_path, receipt)
    return receipt


def run(job_path: pathlib.Path, output: pathlib.Path, progress_dir: pathlib.Path, *,
        chunk_size: int, batch_size: int, device: str) -> dict:
    if chunk_size < 1 or batch_size < 1:
        raise ExtractionRefused("chunk-size and batch-size must be positive")
    job = load_job(job_path)
    verify_upstream(job)
    store = ChunkStore(progress_dir, job_sha256=job.digest)
    tokenizer = load_tokenizer(job)
    e_rows = _load_jsonl(pathlib.Path(job.inputs["e_lure"]["path"]), label="E_lure")
    if len(e_rows) != 2000:
        raise ExtractionRefused("frozen design requires exactly 2,000 E_lure rows")
    model, _config = load_model(job, device=device)

    h1_chunks = []
    for start, stop in _chunks(len(e_rows), chunk_size):
        h1_chunks.append(store.produce(
            "h1", start, stop,
            lambda start=start, stop=stop: _h1_chunk(
                job, model, tokenizer, e_rows[start:stop],
                batch_size=batch_size, device=device,
            ),
        ))
    return assemble_evidence(job, _concat(h1_chunks), output.resolve())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--progress-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    try:
        receipt = run(
            pathlib.Path(args.job), pathlib.Path(args.output), pathlib.Path(args.progress_dir),
            chunk_size=args.chunk_size, batch_size=args.batch_size, device=args.device,
        )
    except (ExtractionRefused, OSError, ValueError, RuntimeError) as exc:
        print(f"[extract_lurestar_evidence] REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
