#!/usr/bin/env python3
"""Generation-bound GCS durability wrapper for D40 expansion scoring."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any, Mapping

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from lurestar import h3_expansion as E  # noqa: E402
from lurestar import h3_precompute as H  # noqa: E402
import run_h3_pilot_durable as BASE  # noqa: E402
import score_h3_expansion as SCORE  # noqa: E402
import score_h3_pilot as FROZEN  # noqa: E402

TRANSPORT_SCHEMA = "nextlat_forgetting/h3_expansion_gcs_transport/1"
STATE_SCHEMA = "nextlat_forgetting/h3_expansion_gcs_state/1"


def raw_job(path: pathlib.Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    transport, remote = payload.get("durable_transport"), payload.get("durable_gcs")
    if not isinstance(transport, Mapping) or not isinstance(remote, Mapping):
        raise E.ExpansionRefused("D40 job lacks durable transport bindings")
    driver = pathlib.Path(str(transport.get("path", "")))
    if not driver.is_absolute():
        driver = path.parent / driver
    driver = driver.resolve()
    if driver != pathlib.Path(__file__).resolve() or transport.get("sha256") != H.sha256_file(driver):
        raise E.ExpansionRefused("D40 durable driver path/hash mismatch")
    for key in ("bucket", "base_prefix", "project"):
        if not isinstance(remote.get(key), str) or not remote[key]:
            raise E.ExpansionRefused(f"D40 durable_gcs.{key} is not frozen")
    scoring = payload.get("scoring")
    if not isinstance(scoring, Mapping) or scoring.get("chunk_size") != 1_000:
        raise E.ExpansionRefused("D40 job must freeze chunk_size at 1,000")
    if not isinstance(scoring.get("batch_size"), int) or scoring["batch_size"] <= 0:
        raise E.ExpansionRefused("D40 job must freeze a positive batch_size")
    return payload


def run(job_path: pathlib.Path, *, adc: pathlib.Path, bootstrap: bool, chunk_size: int,
        batch_size: int) -> dict[str, Any]:
    raw = raw_job(job_path)
    if chunk_size != raw["scoring"]["chunk_size"] or batch_size != raw["scoring"]["batch_size"]:
        raise E.ExpansionRefused("runtime chunk/batch sizes must equal the frozen D40 job")
    if not adc.is_file() or adc.stat().st_mode & 0o077:
        raise E.ExpansionRefused("uploaded ADC must exist with mode 0600")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(adc.resolve())
    os.environ["GOOGLE_CLOUD_PROJECT"] = str(raw["durable_gcs"]["project"])
    versions = BASE._bootstrap_runtime() if bootstrap else BASE._verify_runtime_without_bootstrap()
    from google.cloud import storage

    bucket = storage.Client(project=raw["durable_gcs"]["project"]).bucket(raw["durable_gcs"]["bucket"])
    for role in ("checkpoint", "config"):
        record = raw[role]
        target = pathlib.Path(record["path"])
        if not target.is_absolute():
            target = job_path.parent / target
        BASE._download_frozen_input(bucket, record, target.resolve())
    job_sha, driver_sha = H.sha256_file(job_path), H.sha256_file(__file__)
    durable = BASE.GcsDurability(
        bucket, base_prefix=raw["durable_gcs"]["base_prefix"],
        job_sha256=job_sha, driver_sha256=driver_sha,
    )
    bound = SCORE.load_job(job_path)
    output = bound["_bound"]["output"]
    original_verified = FROZEN._verified_chunk

    def durable_verified(local_output: pathlib.Path, local_job_sha: str, start: int, stop: int):
        if local_job_sha != job_sha:
            raise E.ExpansionRefused("D40 scorer/transport job SHA mismatch")
        existing = original_verified(local_output, local_job_sha, start, stop)
        if existing is None and durable.restore_chunk(local_output, start, stop):
            existing = original_verified(local_output, local_job_sha, start, stop)
        if existing is not None:
            durable.commit_chunk(local_output, start, stop)
        return existing

    FROZEN._verified_chunk = durable_verified
    try:
        receipt = SCORE.run_score(
            job_path, device="cuda", chunk_size=chunk_size, batch_size=batch_size
        )
    finally:
        FROZEN._verified_chunk = original_verified
    loss_path = output / "expansion_losses_135000.jsonl"
    receipt_path = output / "expansion_scoring_receipt.json"
    loss_record = durable.put_file("final/expansion_losses_135000.jsonl", loss_path,
                                   kind="d40_expansion_loss_table")
    receipt_record = durable.put_file("final/expansion_scoring_receipt.json", receipt_path,
                                      kind="d40_expansion_scoring_receipt")
    state = {
        "schema": STATE_SCHEMA, "complete": True, "job_sha256": job_sha,
        "driver_sha256": driver_sha, "runtime": versions, "row_count": E.NEW_COUNT,
        "scientific_scorer_sha256": E.D39_SCORER_SHA256,
        "new_manifest_sha256": bound["_bound"]["hashes"]["new_manifest"],
        "loss_table": loss_record, "scoring_receipt": receipt_record,
    }
    state_record = durable.put_bytes("state.json", H.canonical_json(state), kind="complete_state")
    if json.loads(durable.get_committed("state.json", state_record)) != state:
        raise E.ExpansionRefused("D40 final durable state read-back mismatch")
    H.create_or_verify(output / "durable_state.json", H.canonical_json(state))
    return {**state, "state": state_record,
            "remote_prefix": f"gs://{bucket.name}/{durable.prefix}"}


def plan(job_path: pathlib.Path) -> dict[str, Any]:
    raw = raw_job(job_path)
    scientific = SCORE.plan(job_path)
    job_sha = H.sha256_file(job_path)
    return {
        "schema": TRANSPORT_SCHEMA, "mode": "READ_ONLY_PLAN", "rows": E.NEW_COUNT,
        "job_sha256": job_sha, "scientific_scorer_sha256": scientific["scientific_scorer_sha256"],
        "remote_prefix": f"gs://{raw['durable_gcs']['bucket']}/{raw['durable_gcs']['base_prefix']}/{job_sha}",
        "chunk_size": raw["scoring"]["chunk_size"],
        "batch_size": raw["scoring"]["batch_size"],
        "maximum_uncommitted_items": raw["scoring"]["chunk_size"],
        "publication_order": ["chunk_data", "chunk_receipt", "chunk_commit",
                              "final_table", "final_receipt", "state_last"],
        "credentials_persisted": False, "gpu_launched": False,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("plan", "run"), required=True)
    ap.add_argument("--job", type=pathlib.Path, required=True)
    ap.add_argument("--adc", type=pathlib.Path, default=pathlib.Path("/content/adc.json"))
    ap.add_argument("--bootstrap", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=1_000)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args(argv)
    result = plan(args.job.resolve()) if args.mode == "plan" else run(
        args.job.resolve(), adc=args.adc.resolve(), bootstrap=args.bootstrap,
        chunk_size=args.chunk_size, batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except E.ExpansionRefused as exc:
        raise SystemExit(f"BLOCK: {exc}")
