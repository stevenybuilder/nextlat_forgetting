#!/usr/bin/env python3
"""Materialize and run the complete, outcome-blind Lure-Star evaluation matrix.

The command is deliberately transactional at both scientific boundaries:

* all 15 base parents and every pre-existing evaluation artifact are preflighted before the
  first extractor subprocess is permitted; and
* the evaluator is permitted exactly once, only after all 15 evidence/receipt pairs satisfy
  the extractor/evaluator's live schema and immutable identity bindings.

This orchestrator reads identities, schemas, shapes, and hashes.  It never reads a distance,
margin, accuracy, PSI, or other model outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, Mapping, Sequence

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from lurestar.durable_checkpoint import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)
from run_matrix import (  # noqa: E402
    DONE,
    Ledger,
    MODELS,
    SEEDS,
    verify_base_competence_receipt,
)


SCHEMA = "nextlat_forgetting/lurestar_evaluation_materialization/1"
PLAN_SCHEMA = "nextlat_forgetting/lurestar_evaluation_plan/1"
INVENTORY_NAME = "lurestar_evaluation_inventory.sha256"
RECEIPT_NAME = "lurestar_evaluation_materialization.json"
UPSTREAM_SOURCE_PATHS = (
    "data/stargraph.py",
    "models/model_base.py",
    "models/model_gpt.py",
    "models/model_nextlat.py",
    "models/model_bst.py",
)
LOCAL_MEASUREMENT_SOURCE_PATHS = (
    "src/lurestar/representations.py",
    "src/lurestar/evaluate.py",
)


class MaterializationRefused(RuntimeError):
    """A complete-matrix, provenance, or artifact-integrity condition failed."""


@dataclass(frozen=True)
class EvaluationCell:
    arm: str
    seed: int
    job_id: str
    parent: dict[str, Any]
    source_config: dict[str, str]
    materialized_config: dict[str, str]
    checkpoint: dict[str, str]
    job_path: pathlib.Path
    job_payload: dict[str, Any]
    output_path: pathlib.Path
    progress_root: pathlib.Path


def _is_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _record(path: os.PathLike[str] | str, *, expected: str | None = None) -> dict[str, str]:
    resolved = pathlib.Path(path).resolve()
    if not resolved.is_file():
        raise MaterializationRefused(f"required artifact is missing: {resolved}")
    digest = sha256_file(resolved)
    if expected is not None and digest != expected:
        raise MaterializationRefused(
            f"artifact SHA-256 mismatch for {resolved}: expected {expected}, got {digest}"
        )
    return {"path": str(resolved), "sha256": digest}


def _load_module(path: pathlib.Path, name: str) -> ModuleType:
    path = path.resolve()
    if not path.is_file():
        raise MaterializationRefused(f"evaluation source is missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise MaterializationRefused(f"cannot import evaluation source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _canonical_identity_sha(values: Sequence[Any]) -> str:
    strings = [str(value) for value in values]
    if any("\n" in value or "\r" in value for value in strings):
        raise MaterializationRefused("identity values may not contain CR/LF")
    return hashlib.sha256("".join(value + "\n" for value in strings).encode()).hexdigest()


def _stimulus_identity(path: pathlib.Path, expected_sha256: str) -> dict[str, Any]:
    record = _record(path, expected=expected_sha256)
    base_ids: list[str] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(raw)
            base_id = row["conditions"]["base"]["graph_key"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise MaterializationRefused(
                f"E_lure row {line_number} lacks its canonical base graph identity"
            ) from exc
        if not _is_sha(base_id):
            raise MaterializationRefused(
                f"E_lure row {line_number} base graph_key is not lowercase SHA-256"
            )
        base_ids.append(base_id)
    if len(base_ids) != 2000 or len(set(base_ids)) != 2000:
        raise MaterializationRefused("E_lure must contain exactly 2,000 unique base identities")
    ordered = sorted(base_ids)
    calibration, scored = ordered[:400], ordered[400:]
    return {
        **record,
        "base_count": 2000,
        "calibration_count": 400,
        "scored_count": 1600,
        "calibration_ids_sha256": _canonical_identity_sha(calibration),
        "scored_ids_sha256": _canonical_identity_sha(scored),
    }


def _extractor_policy(extractor: ModuleType) -> dict[str, Any]:
    factory = getattr(extractor, "default_extraction_policy", None)
    if callable(factory):
        policy = factory()
    elif isinstance(getattr(extractor, "EXTRACTION_POLICY", None), Mapping):
        policy = dict(extractor.EXTRACTION_POLICY)
    else:
        # The v3 job contract exposes only these two immutable fields.  Future semantic
        # additions must expose one of the authorities above instead of being guessed here.
        policy = {"whitener_count": 400, "scored_count": 1600}
    if not isinstance(policy, Mapping):
        raise MaterializationRefused("extractor extraction-policy authority is invalid")
    return json.loads(json.dumps(dict(policy), sort_keys=True))


def _materialized_config(parent: Mapping[str, Any], checkpoint: pathlib.Path) -> dict[str, str]:
    path = checkpoint.parent / "materialized_config.yaml"
    record = _record(path)
    out_root = pathlib.Path(str(parent.get("out_root", ""))).resolve()
    try:
        relative = str(path.resolve().relative_to(out_root))
    except ValueError as exc:
        raise MaterializationRefused("materialized config is outside its parent output root") from exc
    artifacts = parent.get("artifacts")
    if not isinstance(artifacts, Mapping) or artifacts.get(relative) != record["sha256"]:
        raise MaterializationRefused(
            "materialized config is not hash-bound by the DONE parent artifact ledger"
        )
    return record


def _parent_cell(
    parent: Mapping[str, Any], *, arm: str, seed: int, root: pathlib.Path,
    upstream: pathlib.Path, upstream_hashes: dict[str, str], e_lure: dict[str, Any],
    local_measurement_hashes: dict[str, str], extractor: ModuleType,
    parent_validator: Callable[..., Any],
) -> EvaluationCell:
    job_id = f"{arm}-s{seed}-base"
    if parent.get("status") != DONE:
        raise MaterializationRefused(f"{job_id} is not DONE")
    if (
        parent.get("job_id") != job_id
        or parent.get("model") != arm
        or parent.get("seed") != seed
        or parent.get("phase") != "base"
    ):
        raise MaterializationRefused(f"{job_id} parent identity is inconsistent")
    try:
        # The production validator re-verifies the complete training artifact set before
        # checking the competence chain.  Keeping this as one injectable boundary lets tests
        # prove the all-15 barrier without manufacturing scientific outcome receipts.
        parent_validator(parent, expected_job_id=job_id, model=arm, seed=seed)
    except RuntimeError as exc:
        raise MaterializationRefused(f"{job_id} competence/training provenance failed: {exc}") from exc

    checkpoint = _record(
        str(parent.get("final_checkpoint", "")),
        expected=str(parent.get("final_checkpoint_sha256", "")),
    )
    source_config = _record(
        str(parent.get("config", "")), expected=str(parent.get("config_sha256", ""))
    )
    materialized = _materialized_config(parent, pathlib.Path(checkpoint["path"]))

    block_path = pathlib.Path(str(extractor.H3_BLOCK_PATH)).resolve()
    block = _record(block_path, expected=str(extractor.H3_BLOCK_SHA256))
    block_sidecar = _record(
        pathlib.Path(f"{block_path}.sha256"),
        expected=str(extractor.H3_BLOCK_SIDECAR_SHA256),
    )
    cell_root = root / "cells" / job_id
    job_path = root / "jobs" / f"{job_id}.json"
    payload = {
        "schema": str(extractor.JOB_SCHEMA),
        "arm": arm,
        "seed": seed,
        "upstream_commit": str(extractor.PINNED_UPSTREAM_COMMIT),
        "upstream_path": str(upstream),
        "upstream_source_sha256": upstream_hashes,
        "local_measurement_source_sha256": local_measurement_hashes,
        "configs": {"base": materialized},
        "checkpoints": {"base": checkpoint},
        "frozen_inputs": {
            "e_lure": {"path": e_lure["path"], "sha256": e_lure["sha256"]}
        },
        "h3_permanent_block": {
            **block,
            "sidecar": block_sidecar,
        },
        "extraction": _extractor_policy(extractor),
    }
    return EvaluationCell(
        arm=arm, seed=seed, job_id=job_id, parent=dict(parent),
        source_config=source_config, materialized_config=materialized,
        checkpoint=checkpoint, job_path=job_path, job_payload=payload,
        output_path=cell_root / "evidence.npz", progress_root=cell_root / "progress",
    )


def build_plan(
    *, ledger_path: pathlib.Path, upstream: pathlib.Path, e_lure_path: pathlib.Path,
    e_lure_sha256: str, evaluation_root: pathlib.Path, extractor_path: pathlib.Path,
    evaluator_path: pathlib.Path, parent_validator: Callable[..., Any] = verify_base_competence_receipt,
    extractor_module: ModuleType | None = None, evaluator_module: ModuleType | None = None,
) -> tuple[dict[str, Any], list[EvaluationCell], ModuleType, ModuleType]:
    """Preflight the entire matrix without writing artifacts or invoking subprocesses."""
    if not _is_sha(e_lure_sha256):
        raise MaterializationRefused("an explicit lowercase E_lure SHA-256 is required")
    extractor = extractor_module or _load_module(extractor_path, "_lurestar_live_extractor")
    evaluator = evaluator_module or _load_module(evaluator_path, "_lurestar_live_evaluator")
    for module, names in (
        (extractor, ("JOB_SCHEMA", "EVIDENCE_SCHEMA", "PINNED_UPSTREAM_COMMIT")),
        (evaluator, ("SCHEMA", "REPORT_SCHEMA", "RECEIPT_SCHEMA")),
    ):
        missing = [name for name in names if not isinstance(getattr(module, name, None), str)]
        if missing:
            raise MaterializationRefused(f"evaluation source lacks schema authority: {missing}")

    upstream = upstream.resolve()
    if not upstream.is_dir():
        raise MaterializationRefused(f"pinned upstream checkout is missing: {upstream}")
    upstream_hashes = {
        relative: _record(upstream / relative)["sha256"] for relative in UPSTREAM_SOURCE_PATHS
    }
    local_measurement_sources = {
        relative: _record(_REPO / relative) for relative in LOCAL_MEASUREMENT_SOURCE_PATHS
    }
    local_measurement_hashes = {
        relative: record["sha256"] for relative, record in local_measurement_sources.items()
    }
    e_lure = _stimulus_identity(e_lure_path.resolve(), e_lure_sha256)
    states = Ledger(ledger_path).states()
    wanted = {(arm, seed): f"{arm}-s{seed}-base" for arm in MODELS for seed in SEEDS}
    missing = sorted(job for job in wanted.values() if job not in states)
    extra_base = sorted(
        job for job, state in states.items()
        if state.get("phase") == "base" and job not in set(wanted.values())
    )
    if missing or extra_base:
        raise MaterializationRefused(
            f"ledger base matrix differs from exact 15 cells; missing={missing}, extra={extra_base}"
        )

    cells: list[EvaluationCell] = []
    # Every parent, including the final cell, is validated before this function returns.
    for arm in MODELS:
        for seed in SEEDS:
            cells.append(_parent_cell(
                states[wanted[(arm, seed)]], arm=arm, seed=seed,
                root=evaluation_root.resolve(), upstream=upstream,
                upstream_hashes=upstream_hashes, e_lure=e_lure,
                local_measurement_hashes=local_measurement_hashes,
                extractor=extractor, parent_validator=parent_validator,
            ))
    if len(cells) != 15 or len({str(cell.output_path) for cell in cells}) != 15 or len(
        {str(cell.progress_root) for cell in cells}
    ) != 15:
        raise MaterializationRefused("evaluation cells do not have 15 unique output/progress roots")

    plan = {
        "schema": PLAN_SCHEMA,
        "outcomes_inspected": False,
        "expected_arms": list(MODELS),
        "expected_seeds": list(SEEDS),
        "cell_count": 15,
        "extractor": _record(extractor_path),
        "evaluator": _record(evaluator_path),
        "local_measurement_sources": local_measurement_sources,
        "schemas": {
            "job": extractor.JOB_SCHEMA,
            "evidence": extractor.EVIDENCE_SCHEMA,
            "manifest": evaluator.SCHEMA,
            "report": evaluator.REPORT_SCHEMA,
            "evaluation_receipt": evaluator.RECEIPT_SCHEMA,
        },
        "upstream": {
            "path": str(upstream), "commit": extractor.PINNED_UPSTREAM_COMMIT,
            "source_sha256": upstream_hashes,
        },
        "e_lure": e_lure,
        "cells": [
            {
                "arm": cell.arm, "seed": cell.seed, "job_id": cell.job_id,
                "source_config": cell.source_config,
                "materialized_config": cell.materialized_config,
                "checkpoint": cell.checkpoint,
                "job_path": str(cell.job_path), "output_path": str(cell.output_path),
                "progress_root": str(cell.progress_root),
            }
            for cell in cells
        ],
    }
    return plan, cells, extractor, evaluator


def _expected_evidence_fields(evaluator: ModuleType, arm: str) -> set[str]:
    authority = getattr(evaluator, "expected_evidence_fields", None)
    if callable(authority):
        fields = authority(arm)
    else:
        fields = set(getattr(evaluator, "REQUIRED_ARRAYS", ())) | set(
            getattr(evaluator, "BOUND_SCALARS", ())
        )
        if arm == "bst":
            fields |= set(getattr(evaluator, "BST_SECONDARY_ARRAYS", ()))
            fields |= set(getattr(evaluator, "BST_WHITENER_FIELDS", ()))
    if not fields:
        raise MaterializationRefused("evaluator exposes no evidence-field authority")
    return set(fields)


def validate_evidence(
    cell: EvaluationCell, *, extractor: ModuleType, evaluator: ModuleType,
    expected_score_sha256: str,
) -> dict[str, Any]:
    """Validate evidence identity and schema without reading any outcome array values."""
    output = cell.output_path.resolve()
    receipt_path = output.with_suffix(output.suffix + ".receipt.json")
    if not output.is_file() or not receipt_path.is_file():
        raise MaterializationRefused(f"{cell.job_id} evidence/receipt pair is incomplete")
    output_sha = sha256_file(output)
    expected_fields = _expected_evidence_fields(evaluator, cell.arm)
    with np.load(output, allow_pickle=False) as arrays:
        if set(arrays.files) != expected_fields:
            raise MaterializationRefused(
                f"{cell.job_id} evidence fields differ; "
                f"missing={sorted(expected_fields-set(arrays.files))}, "
                f"extra={sorted(set(arrays.files)-expected_fields)}"
            )
        scalar_expected = {
            "evidence_schema": str(extractor.EVIDENCE_SCHEMA),
            "arm": cell.arm,
            "seed": str(cell.seed),
            "base_checkpoint_sha256": cell.checkpoint["sha256"],
            "h3_permanent_block_sha256": str(extractor.H3_BLOCK_SHA256),
            "h3_permanent_block_sidecar_sha256": str(extractor.H3_BLOCK_SIDECAR_SHA256),
            "local_representations_sha256": cell.job_payload[
                "local_measurement_source_sha256"
            ]["src/lurestar/representations.py"],
            "local_evaluate_sha256": cell.job_payload["local_measurement_source_sha256"][
                "src/lurestar/evaluate.py"
            ],
        }
        for key, expected in scalar_expected.items():
            if key not in arrays.files or np.asarray(arrays[key]).size != 1:
                raise MaterializationRefused(f"{cell.job_id} lacks scalar identity {key}")
            observed = str(np.asarray(arrays[key]).reshape(-1)[0])
            if observed != expected:
                raise MaterializationRefused(f"{cell.job_id} evidence identity {key} disagrees")
        if "h1_item_ids" not in arrays.files:
            raise MaterializationRefused(f"{cell.job_id} lacks scored-item identities")
        ids = np.asarray(arrays["h1_item_ids"]).ravel()
        if ids.size != 1600 or _canonical_identity_sha(ids.tolist()) != expected_score_sha256:
            raise MaterializationRefused(f"{cell.job_id} scored-item identity differs from E_score")
        if "h1_item_ids_sha256" in arrays.files:
            embedded = str(np.asarray(arrays["h1_item_ids_sha256"]).reshape(-1)[0])
            if embedded != expected_score_sha256:
                raise MaterializationRefused(f"{cell.job_id} embedded E_score hash disagrees")

    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationRefused(f"{cell.job_id} evidence receipt is invalid") from exc
    expected_receipt = {
        "schema": str(extractor.EVIDENCE_SCHEMA),
        "arm": cell.arm,
        "seed": cell.seed,
    }
    if any(receipt.get(key) != value for key, value in expected_receipt.items()):
        raise MaterializationRefused(f"{cell.job_id} evidence receipt identity disagrees")
    if receipt.get("job") != {
        "path": str(cell.job_path.resolve()), "sha256": sha256_file(cell.job_path)
    }:
        raise MaterializationRefused(f"{cell.job_id} evidence receipt job binding disagrees")
    if receipt.get("evidence") != {"path": str(output), "sha256": output_sha}:
        raise MaterializationRefused(f"{cell.job_id} evidence receipt output binding disagrees")
    if receipt.get("checkpoints") != {"base": cell.checkpoint}:
        raise MaterializationRefused(f"{cell.job_id} evidence receipt checkpoint disagrees")
    expected_local_sources = {
        relative: {
            "path": str((_REPO / relative).resolve()),
            "sha256": cell.job_payload["local_measurement_source_sha256"][relative],
        }
        for relative in LOCAL_MEASUREMENT_SOURCE_PATHS
    }
    if receipt.get("local_measurement_sources") != expected_local_sources:
        raise MaterializationRefused(
            f"{cell.job_id} evidence receipt local measurement binding disagrees"
        )
    domains = receipt.get("identity_domains", {})
    if domains.get("h1_quartet") != {
        "count": 1600, "item_ids_sha256": expected_score_sha256
    }:
        raise MaterializationRefused(f"{cell.job_id} evidence receipt E_score domain disagrees")
    return {
        "arm": cell.arm, "seed": cell.seed,
        "base_checkpoint": cell.checkpoint,
        "evidence_npz": str(output), "evidence_sha256": output_sha,
        "receipt": {"path": str(receipt_path), "sha256": sha256_file(receipt_path)},
    }


def _existing_status(
    cells: Sequence[EvaluationCell], *, extractor: ModuleType, evaluator: ModuleType,
    expected_score_sha256: str,
) -> dict[str, dict[str, Any] | None]:
    """Preflight every existing output before any missing cell may be invoked."""
    status: dict[str, dict[str, Any] | None] = {}
    for cell in cells:
        output = cell.output_path
        receipt = output.with_suffix(output.suffix + ".receipt.json")
        if not output.exists() and not receipt.exists():
            status[cell.job_id] = None
            continue
        # One-sided or stale artifacts are refusals, never silently replaced.
        status[cell.job_id] = validate_evidence(
            cell, extractor=extractor, evaluator=evaluator,
            expected_score_sha256=expected_score_sha256,
        )
    return status


def _relay(command: list[str], *, cwd: pathlib.Path) -> int:
    print("+ " + " ".join(command), flush=True)
    process = subprocess.Popen(
        command, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print("  | " + line.rstrip(), flush=True)
    return process.wait()


def _manifest_payload(
    *, plan: Mapping[str, Any], evaluator: ModuleType, cells: Sequence[dict[str, Any]],
    n_boot: int,
) -> dict[str, Any]:
    block_path = pathlib.Path(str(plan["cells"][0]["job_path"]))
    # Read the already-materialized job, which is the single authority for the exact block record.
    job = json.loads(block_path.read_text(encoding="utf-8"))
    return {
        "schema": str(evaluator.SCHEMA),
        "analysis_seed": 73021,
        "expected_arms": list(getattr(evaluator, "E").ARMS),
        "expected_seeds": list(SEEDS),
        "n_boot": n_boot,
        "identity_domains": {
            "h1_quartet": {
                "count": 1600,
                "item_ids_sha256": plan["e_lure"]["scored_ids_sha256"],
            }
        },
        "h3_permanent_block": job["h3_permanent_block"],
        "extractor": plan["extractor"],
        "local_measurement_sources": plan["local_measurement_sources"],
        "frozen_inputs": {
            "e_lure": {
                "path": plan["e_lure"]["path"], "sha256": plan["e_lure"]["sha256"]
            }
        },
        "cells": [
            {key: value for key, value in cell.items() if key != "receipt"}
            for cell in cells
        ],
    }


def _inventory(records: Mapping[str, str]) -> str:
    if len(records) != len(set(records)) or any(not _is_sha(value) for value in records.values()):
        raise MaterializationRefused("inventory contains duplicate names or invalid digests")
    return "".join(f"{records[name]}  {name}\n" for name in sorted(records))


def _verify_local_measurement_sources(plan: Mapping[str, Any]) -> None:
    records = plan.get("local_measurement_sources")
    if not isinstance(records, Mapping) or set(records) != set(LOCAL_MEASUREMENT_SOURCE_PATHS):
        raise MaterializationRefused("plan local measurement-source inventory changed")
    for relative in LOCAL_MEASUREMENT_SOURCE_PATHS:
        expected = records[relative]
        if not isinstance(expected, Mapping) or set(expected) != {"path", "sha256"}:
            raise MaterializationRefused(f"plan local source record is malformed for {relative}")
        current = (_REPO / relative).resolve()
        if pathlib.Path(str(expected["path"])).resolve() != current:
            raise MaterializationRefused(f"plan local source path changed for {relative}")
        if not _is_sha(expected["sha256"]) or not current.is_file() or sha256_file(
            current
        ) != expected["sha256"]:
            raise MaterializationRefused(f"local measurement source changed for {relative}")


def _existing_evaluation(
    *, manifest_path: pathlib.Path, report_path: pathlib.Path, evaluator: ModuleType,
    local_measurement_sources: Mapping[str, Any] | None = None,
) -> tuple[pathlib.Path, pathlib.Path] | None:
    """Return a valid prior receipt/sidecar pair, refuse every partial or stale triple."""
    receipt_path = report_path.with_suffix(report_path.suffix + ".receipt.json")
    sidecar_path = receipt_path.with_suffix(receipt_path.suffix + ".sha256")
    present = [path.exists() for path in (report_path, receipt_path, sidecar_path)]
    if not any(present):
        return None
    if not all(present):
        raise MaterializationRefused("prior evaluator artifacts are incomplete/stale")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationRefused("prior evaluation receipt is invalid") from exc
    if (
        receipt.get("schema") != evaluator.RECEIPT_SCHEMA
        or receipt.get("manifest") != {
            "path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)
        }
        or receipt.get("report") != {
            "path": str(report_path.resolve()), "sha256": sha256_file(report_path)
        }
        or (
            local_measurement_sources is not None
            and receipt.get("local_measurement_sources") != dict(local_measurement_sources)
        )
    ):
        raise MaterializationRefused("prior evaluation receipt is stale for this manifest/report")
    fields = sidecar_path.read_text(encoding="utf-8").strip().split()
    if not fields or fields[0].lower() != sha256_file(receipt_path):
        raise MaterializationRefused("prior evaluation receipt sidecar is stale")
    return receipt_path, sidecar_path


def execute(
    *, ledger_path: pathlib.Path, upstream: pathlib.Path, e_lure_path: pathlib.Path,
    e_lure_sha256: str, evaluation_root: pathlib.Path, extractor_path: pathlib.Path,
    evaluator_path: pathlib.Path, python: str = sys.executable, device: str = "cuda",
    chunk_size: int = 16, batch_size: int = 64, n_boot: int = 10_000,
    dry_run: bool = False, print_plan: bool = False,
    command_runner: Callable[..., int] = _relay,
    parent_validator: Callable[..., Any] = verify_base_competence_receipt,
    extractor_module: ModuleType | None = None, evaluator_module: ModuleType | None = None,
) -> dict[str, Any]:
    if chunk_size < 1 or batch_size < 1 or n_boot < 100:
        raise MaterializationRefused("chunk-size/batch-size must be positive and n_boot >= 100")
    plan, cells, extractor, evaluator = build_plan(
        ledger_path=ledger_path, upstream=upstream, e_lure_path=e_lure_path,
        e_lure_sha256=e_lure_sha256, evaluation_root=evaluation_root,
        extractor_path=extractor_path, evaluator_path=evaluator_path,
        parent_validator=parent_validator, extractor_module=extractor_module,
        evaluator_module=evaluator_module,
    )
    if print_plan or dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return {"status": "DRY_RUN", "plan": plan}
    _verify_local_measurement_sources(plan)

    # This checks all 15 outputs before one extractor is allowed to start.
    existing = _existing_status(
        cells, extractor=extractor, evaluator=evaluator,
        expected_score_sha256=plan["e_lure"]["scored_ids_sha256"],
    )
    evaluation_root.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        atomic_write_json(cell.job_path, cell.job_payload)
    atomic_write_json(evaluation_root / "plan.json", plan)

    # Validate each live job through the extractor's own parser before GPU work.
    for cell in cells:
        _verify_local_measurement_sources(plan)
        try:
            bound = extractor.load_job(cell.job_path)
        except Exception as exc:
            raise MaterializationRefused(f"{cell.job_id} live extractor job refused: {exc}") from exc
        if getattr(bound, "digest", None) != sha256_file(cell.job_path):
            raise MaterializationRefused(f"{cell.job_id} live extractor did not bind its job hash")

    for cell in cells:
        if existing[cell.job_id] is not None:
            continue
        _verify_local_measurement_sources(plan)
        command = [
            python, str(extractor_path.resolve()), "--job", str(cell.job_path),
            "--output", str(cell.output_path), "--progress-dir", str(cell.progress_root),
            "--chunk-size", str(chunk_size), "--batch-size", str(batch_size),
            "--device", device,
        ]
        if command_runner(command, cwd=upstream.resolve()) != 0:
            raise MaterializationRefused(f"{cell.job_id} extractor failed")

    evidence = [
        validate_evidence(
            cell, extractor=extractor, evaluator=evaluator,
            expected_score_sha256=plan["e_lure"]["scored_ids_sha256"],
        )
        for cell in cells
    ]
    _verify_local_measurement_sources(plan)
    manifest = _manifest_payload(plan=plan, evaluator=evaluator, cells=evidence, n_boot=n_boot)
    manifest_path = evaluation_root / "evaluation_manifest.json"
    atomic_write_json(manifest_path, manifest)
    report_path = evaluation_root / "report.json"
    prior_evaluation = _existing_evaluation(
        manifest_path=manifest_path, report_path=report_path, evaluator=evaluator,
        local_measurement_sources=plan["local_measurement_sources"],
    )
    evaluator_invocations = 0
    if prior_evaluation is None:
        _verify_local_measurement_sources(plan)
        command = [
            python, str(evaluator_path.resolve()), "--manifest", str(manifest_path),
            "--output", str(report_path), "--seeds", *[str(seed) for seed in SEEDS],
            "--n-boot", str(n_boot),
        ]
        if command_runner(command, cwd=_REPO) != 0:
            raise MaterializationRefused("Lure-Star evaluator failed")
        evaluator_invocations = 1
    _verify_local_measurement_sources(plan)
    evaluation_receipt = report_path.with_suffix(report_path.suffix + ".receipt.json")
    evaluation_sidecar = evaluation_receipt.with_suffix(evaluation_receipt.suffix + ".sha256")
    for path in (report_path, evaluation_receipt, evaluation_sidecar):
        if not path.is_file():
            raise MaterializationRefused(f"evaluator omitted required artifact: {path}")
    try:
        receipt_document = json.loads(evaluation_receipt.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MaterializationRefused("evaluation receipt is invalid JSON") from exc
    if (
        receipt_document.get("schema") != evaluator.RECEIPT_SCHEMA
        or receipt_document.get("manifest") != {
            "path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)
        }
        or receipt_document.get("report") != {
            "path": str(report_path.resolve()), "sha256": sha256_file(report_path)
        }
        or receipt_document.get("local_measurement_sources") != plan[
            "local_measurement_sources"
        ]
    ):
        raise MaterializationRefused("evaluation receipt identity/hash binding disagrees")

    records: dict[str, str] = {
        "plan.json": sha256_file(evaluation_root / "plan.json"),
        "evaluation_manifest.json": sha256_file(manifest_path),
        "report.json": sha256_file(report_path),
        "report.json.receipt.json": sha256_file(evaluation_receipt),
        "report.json.receipt.json.sha256": sha256_file(evaluation_sidecar),
        "inputs/e_lure": plan["e_lure"]["sha256"],
        "sources/extractor": plan["extractor"]["sha256"],
        "sources/evaluator": plan["evaluator"]["sha256"],
        **{
            f"sources/{relative}": plan["local_measurement_sources"][relative]["sha256"]
            for relative in LOCAL_MEASUREMENT_SOURCE_PATHS
        },
    }
    for cell, validated in zip(cells, evidence):
        records[f"jobs/{cell.job_id}.json"] = sha256_file(cell.job_path)
        records[f"cells/{cell.job_id}/evidence.npz"] = validated["evidence_sha256"]
        records[f"cells/{cell.job_id}/evidence.npz.receipt.json"] = validated["receipt"][
            "sha256"
        ]
        records[f"parents/{cell.job_id}/checkpoint"] = cell.checkpoint["sha256"]
        records[f"parents/{cell.job_id}/source_config"] = cell.source_config["sha256"]
        records[f"parents/{cell.job_id}/materialized_config"] = cell.materialized_config[
            "sha256"
        ]
    inventory_path = evaluation_root / INVENTORY_NAME
    atomic_write_text(inventory_path, _inventory(records))
    receipt = {
        "schema": SCHEMA,
        "status": "COMPLETE",
        "outcomes_inspected": False,
        "cell_count": 15,
        "extractor_invocations_permitted_after_complete_preflight": True,
        "evaluator_invocations_this_execution": evaluator_invocations,
        "evaluator_invocations_maximum_per_execution": 1,
        "inventory": {
            "path": str(inventory_path.resolve()), "sha256": sha256_file(inventory_path),
            "entry_count": len(records),
        },
        "plan": {
            "path": str((evaluation_root / "plan.json").resolve()),
            "sha256": sha256_file(evaluation_root / "plan.json"),
        },
        "manifest": {"path": str(manifest_path.resolve()), "sha256": sha256_file(manifest_path)},
        "evaluation_receipt": {
            "path": str(evaluation_receipt.resolve()), "sha256": sha256_file(evaluation_receipt)
        },
        "local_measurement_sources": plan["local_measurement_sources"],
    }
    atomic_write_json(evaluation_root / RECEIPT_NAME, receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--e-lure", required=True)
    parser.add_argument("--e-lure-sha256", required=True)
    parser.add_argument("--evaluation-root", required=True)
    parser.add_argument(
        "--extractor", default=str(_REPO / "scripts" / "extract_lurestar_evidence.py")
    )
    parser.add_argument(
        "--evaluator", default=str(_REPO / "scripts" / "evaluate_lurestar_checkpoints.py")
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = execute(
            ledger_path=pathlib.Path(args.ledger), upstream=pathlib.Path(args.upstream),
            e_lure_path=pathlib.Path(args.e_lure), e_lure_sha256=args.e_lure_sha256,
            evaluation_root=pathlib.Path(args.evaluation_root),
            extractor_path=pathlib.Path(args.extractor), evaluator_path=pathlib.Path(args.evaluator),
            python=args.python, device=args.device, chunk_size=args.chunk_size,
            batch_size=args.batch_size, n_boot=args.n_boot,
            dry_run=args.dry_run, print_plan=args.print_plan,
        )
    except (MaterializationRefused, OSError, ValueError) as exc:
        print(f"[materialize_lurestar_evaluation] REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
