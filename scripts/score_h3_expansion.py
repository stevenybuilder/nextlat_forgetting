#!/usr/bin/env python3
"""Score only the 135,000 prospective D40 middle candidates.

The scientific forward/loss primitives are imported from the byte-frozen D39 scorer and its exact
SHA-256 is mandatory. This file adds population validation and restartable orchestration only.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections.abc import Mapping
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from lurestar import h3_expansion as E  # noqa: E402
from lurestar import h3_precompute as H  # noqa: E402
import score_h3_pilot as FROZEN  # noqa: E402

JOB_SCHEMA = "nextlat_forgetting/h3_expansion_score_job/1"
EXCLUSION_DOMAIN_SHA256 = {
    "training": "d13199b00c41d74325931cfecb15e3cf876d5e7d999c3257aaf4962e44827d76",
    "e_lure": "f67765e6ea2afd4156c9d03ad0271afe224f1a54ddf1afc82118fcc3e4541495",
    "b_near": "7e4a414fc51c693e850fb5a0e01a651e3e78cb01304ddf1704cf11aad5314528",
    "b_far": "364978600eb73a6e9044e812dd974fe6a2df509b7f256079dc3c7d2ec8ab99e3",
    "a_pair": "e70fb087b6b1dd6fa7129303bbc4bcc30843c327fcab168937976295cbf2dd10",
    "b_mid_d39": E.D39_MID_SHA256,
    "acquisition_near": "f152d2f263900e760aefff67085f202b659692ea186c024f53b0e812adb46053",
    "acquisition_mid": "85096001d85ad7ef0dbe001b712d78c3b697d0b08f84fafbeef51b11e2c9512d",
    "acquisition_far": "d279d29388f3315049c4693eba52b56f6ffbb5e1ba5d4dcb0c25e915e404c717",
}


def load_job(path: pathlib.Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != JOB_SCHEMA or raw.get("role") != "non_confirmatory_d40_expansion":
        raise E.ExpansionRefused("invalid D40 scoring job schema/role")
    if raw.get("confirmatory_inputs") is not False or raw.get("confirmatory_results") is not False:
        raise E.ExpansionRefused("confirmatory inputs/results are forbidden")
    if raw.get("model_family") != "bst" or raw.get("seed") != 1234 or raw.get("training_step") != 500:
        raise E.ExpansionRefused("D40 must reuse the sole D39 pilot")
    if raw.get("upstream_commit") != H.UPSTREAM_COMMIT:
        raise E.ExpansionRefused("D40 upstream commit changed after the prospective freeze")
    base = path.parent
    records = {}
    for key in ("checkpoint", "config", "tokenizer", "frozen_scorer", "expansion_scorer", "adaptation_contract",
                "d40_decision", "generation_receipt", "generation_code_receipt",
                "generation_domain_receipt", "new_manifest"):
        records[key] = FROZEN._record(raw.get(key), base, key)
    if records["checkpoint"][1] != H.PILOT_CHECKPOINT_SHA256 or records["config"][1] != H.PILOT_CONFIG_SHA256:
        raise E.ExpansionRefused("D40 pilot checkpoint/config substitution")
    if records["frozen_scorer"][0] != (ROOT / "scripts/score_h3_pilot.py").resolve() or records["frozen_scorer"][1] != E.D39_SCORER_SHA256:
        raise E.ExpansionRefused("D40 does not call the exact frozen D39 scorer")
    if records["expansion_scorer"][0] != pathlib.Path(__file__).resolve() or records["expansion_scorer"][1] != H.sha256_file(__file__):
        raise E.ExpansionRefused("D40 expansion scorer changed after job freeze")
    if records["d40_decision"][1] != E.D40_DECISION_SHA256:
        raise E.ExpansionRefused("D40 decision changed after generation")
    generation = json.loads(records["generation_receipt"][0].read_text())
    if generation.get("status") != "D40_CANDIDATES_FROZEN_BEFORE_SCORING":
        raise E.ExpansionRefused("D40 generation receipt is not pre-scoring frozen")
    if generation.get("outputs", {}).get("b_mid_new_135000.jsonl") != records["new_manifest"][1]:
        raise E.ExpansionRefused("D40 generation receipt does not bind the new manifest")
    code_receipt = json.loads(records["generation_code_receipt"][0].read_text())
    if (
        code_receipt.get("status") != "D40_GENERATOR_BYTES_BOUND_BEFORE_SCORING"
        or code_receipt.get("decision_sha256") != E.D40_DECISION_SHA256
        or code_receipt.get("artifacts", {}).get("generation_receipt_sha256")
        != records["generation_receipt"][1]
        or code_receipt.get("artifacts", {}).get("new_manifest_sha256")
        != records["new_manifest"][1]
        or code_receipt.get("artifacts", {}).get("generation_domain_receipt_sha256")
        != records["generation_domain_receipt"][1]
        or code_receipt.get("generator_library_sha256")
        != H.sha256_file(ROOT / "src/lurestar/h3_expansion.py")
        or code_receipt.get("generator_cli_sha256")
        != H.sha256_file(ROOT / "scripts/materialize_h3_expansion.py")
        or code_receipt.get("confirmatory_inputs_inspected") is not False
        or code_receipt.get("confirmatory_results_inspected") is not False
    ):
        raise E.ExpansionRefused("D40 generator-byte receipt is invalid or does not bind this job")
    domain = json.loads(records["generation_domain_receipt"][0].read_text())
    if (
        domain.get("status") != "D40_EXCLUSION_DOMAINS_BOUND_BEFORE_SCORING"
        or domain.get("generation_receipt_sha256") != records["generation_receipt"][1]
        or domain.get("new_manifest_sha256") != records["new_manifest"][1]
        or domain.get("exclusion_domain_sha256") != EXCLUSION_DOMAIN_SHA256
        or domain.get("confirmatory_inputs_inspected") is not False
        or domain.get("confirmatory_results_inspected") is not False
    ):
        raise E.ExpansionRefused("D40 exclusion-domain receipt is invalid or incomplete")
    rows = E._rows(records["new_manifest"][0])
    if len(rows) != E.NEW_COUNT or len({row.get("prompt_sha256") for row in rows}) != E.NEW_COUNT:
        raise E.ExpansionRefused("D40 score population is not 135,000 unique new rows")
    if any(row.get("schema") != E.SCHEMA or row.get("pool") != "B_mid" for row in rows):
        raise E.ExpansionRefused("D40 scorer received a non-expansion row")
    for index, row in enumerate(rows):
        line = row.get("line")
        if (
            not isinstance(line, str)
            or H.prompt_sha(line) != row.get("prompt_sha256")
            or H.canonical_key_from_line(line) != row.get("graph_key")
        ):
            raise E.ExpansionRefused(f"D40 score row {index} identity does not hash its line")
        H.validate_line(line)
    output = pathlib.Path(str(raw.get("output_dir", "")))
    if not output.is_absolute():
        output = base / output
    raw["_bound"] = {
        "checkpoint": records["checkpoint"][0], "config": records["config"][0],
        "tokenizer": records["tokenizer"][0], "adaptation": records["adaptation_contract"][0],
        "new_manifest": records["new_manifest"][0], "rows": rows,
        "output": output.resolve(),
        "hashes": {key: value[1] for key, value in records.items()},
    }
    return raw


def run_score(job_path: pathlib.Path, *, device: str, chunk_size: int, batch_size: int) -> dict[str, Any]:
    if chunk_size <= 0 or batch_size <= 0:
        raise E.ExpansionRefused("positive chunk/batch sizes required")
    job_sha, job = H.sha256_file(job_path), load_job(job_path)
    output, rows = job["_bound"]["output"], job["_bound"]["rows"]
    H.create_or_verify(output / "job_identity.json", H.canonical_json({
        "schema": JOB_SCHEMA, "job_sha256": job_sha,
        "scientific_scorer_sha256": E.D39_SCORER_SHA256,
    }))
    # Frozen loader consumes only these bound fields plus seed/model identity.
    frozen_job = {"_bound": {"checkpoint": job["_bound"]["checkpoint"],
                              "config": job["_bound"]["config"]}}
    model = tokenizer = None
    chunks: list[bytes] = []
    score_rows = [{"pool": "b_mid_d40", "prompt_sha256": row["prompt_sha256"],
                   "line": row["line"]} for row in rows]
    for start in range(0, len(rows), chunk_size):
        stop = min(start + chunk_size, len(rows))
        existing = FROZEN._verified_chunk(output, job_sha, start, stop)
        if existing is None:
            if model is None:
                model, tokenizer = FROZEN._load_model(frozen_job, device)
            pieces = [
                FROZEN._score(model, tokenizer, score_rows[begin:min(begin + batch_size, stop)], device)
                for begin in range(start, stop, batch_size)
            ]
            FROZEN._write_chunk(
                output, job_sha, score_rows[start:stop], np.concatenate(pieces), start, stop
            )
            existing = FROZEN._verified_chunk(output, job_sha, start, stop)
        assert existing is not None
        chunks.append(existing)
    table_sha = H.create_or_verify(output / "expansion_losses_135000.jsonl", b"".join(chunks))
    receipt = {
        "schema": E.LOSS_SCHEMA, "status": "COMPLETE_D40_EXPANSION_SCORING",
        "job_sha256": job_sha, "row_count": E.NEW_COUNT,
        "loss_table_sha256": table_sha,
        "checkpoint_sha256": H.PILOT_CHECKPOINT_SHA256,
        "config_sha256": H.PILOT_CONFIG_SHA256,
        "tokenizer_sha256": job["_bound"]["hashes"]["tokenizer"],
        "scientific_scorer_sha256": E.D39_SCORER_SHA256,
        "new_manifest_sha256": job["_bound"]["hashes"]["new_manifest"],
        "generation_receipt_sha256": job["_bound"]["hashes"]["generation_receipt"],
        "generation_code_receipt_sha256": job["_bound"]["hashes"]["generation_code_receipt"],
        "generation_domain_receipt_sha256": job["_bound"]["hashes"]["generation_domain_receipt"],
        "confirmatory_inputs_inspected": False, "confirmatory_results_inspected": False,
    }
    H.create_or_verify(output / "expansion_scoring_receipt.json", H.canonical_json(receipt))
    return receipt


def plan(job_path: pathlib.Path) -> dict[str, Any]:
    job = load_job(job_path)
    return {"schema": JOB_SCHEMA, "mode": "READ_ONLY_PLAN",
            "job_sha256": H.sha256_file(job_path), "rows": len(job["_bound"]["rows"]),
            "scientific_scorer_sha256": E.D39_SCORER_SHA256,
            "checkpoint_sha256": H.PILOT_CHECKPOINT_SHA256, "gpu_launched": False}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("plan", "score"), required=True)
    ap.add_argument("--job", type=pathlib.Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--chunk-size", type=int, default=1_000)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args(argv)
    result = plan(args.job.resolve()) if args.mode == "plan" else run_score(
        args.job.resolve(), device=args.device, chunk_size=args.chunk_size, batch_size=args.batch_size
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except E.ExpansionRefused as exc:
        raise SystemExit(f"BLOCK: {exc}")
