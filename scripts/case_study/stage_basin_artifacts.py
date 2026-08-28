#!/usr/bin/env python3
"""Stage the generation- and hash-bound basin case-study artifacts from private GCS.

The bucket is supplied at runtime and is intentionally never written to the public receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import tempfile
from typing import Any


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: pathlib.Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp.write(encoded)
        tmp_path = pathlib.Path(tmp.name)
    os.replace(tmp_path, path)


def _download(bucket: Any, *, object_name: str, destination: pathlib.Path,
              expected_sha256: str, expected_size: int | None = None,
              generation: str | None = None) -> dict[str, Any]:
    expected_generation = int(generation) if generation is not None else None
    blob = bucket.blob(object_name, generation=expected_generation)
    blob.reload()
    observed_generation = int(blob.generation)
    if expected_generation is not None and observed_generation != expected_generation:
        raise RuntimeError(f"generation mismatch for {object_name}")
    if expected_size is not None and int(blob.size) != expected_size:
        raise RuntimeError(f"remote size mismatch for {object_name}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256_file(destination) == expected_sha256:
        status = "already_present_and_verified"
    else:
        partial = destination.with_name(destination.name + ".partial")
        blob.download_to_filename(
            str(partial), if_generation_match=observed_generation, checksum="auto"
        )
        if expected_size is not None and partial.stat().st_size != expected_size:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"downloaded size mismatch for {object_name}")
        if sha256_file(partial) != expected_sha256:
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"downloaded SHA-256 mismatch for {object_name}")
        os.replace(partial, destination)
        status = "downloaded_and_verified"
    return {
        "object_name": object_name,
        "generation": str(observed_generation),
        "size": destination.stat().st_size,
        "sha256": expected_sha256,
        "local_path": str(destination),
        "status": status,
    }


def stage(*, freeze_path: pathlib.Path, bucket_name: str,
          output_root: pathlib.Path) -> dict[str, Any]:
    try:
        from google.cloud import storage
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("google-cloud-storage is not installed") from exc

    freeze_path = freeze_path.resolve()
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("schema") != "nextlat_forgetting/basin_case_study_freeze/1":
        raise RuntimeError("unexpected basin freeze schema")
    if freeze["runtime_controls"]["new_training_authorized"] is not False:
        raise RuntimeError("freeze unexpectedly authorizes training")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    records: list[dict[str, Any]] = []
    test = freeze["data"]["test"]
    records.append(_download(
        bucket, object_name=test["object_name"],
        destination=output_root / "data" / "graph_5_5_test_20000.txt",
        expected_sha256=test["sha256"], expected_size=int(test["size"]),
        generation=test["generation"],
    ))

    for run in freeze["runs"]:
        job_id = run["job_id"]
        seed = int(run["seed"])
        object_root = f"lurestar/runs/{job_id}/{job_id}-seed{seed}"
        run_root = output_root / "runs" / job_id
        records.append(_download(
            bucket, object_name=f"{object_root}/materialized_config.yaml",
            destination=run_root / "materialized_config.yaml",
            expected_sha256=run["materialized_config_sha256"],
        ))
        records.append(_download(
            bucket, object_name=f"{object_root}/version_0/metrics.csv",
            destination=run_root / "metrics.csv",
            expected_sha256=run["metrics_sha256"],
        ))
        for checkpoint in run["checkpoints"]:
            records.append(_download(
                bucket, object_name=f"{object_root}/{checkpoint['filename']}",
                destination=run_root / "checkpoints" / checkpoint["filename"],
                expected_sha256=checkpoint["sha256"],
                expected_size=int(checkpoint["size"]), generation=checkpoint["generation"],
            ))

    receipt = {
        "schema": "nextlat_forgetting/basin_artifact_staging_receipt/1",
        "storage_backend": "private_gcs",
        "bucket_name_recorded": False,
        "freeze": {"path": str(freeze_path), "sha256": sha256_file(freeze_path)},
        "artifact_count": len(records),
        "artifacts": records,
    }
    atomic_json(output_root / "staging_receipt.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    receipt = stage(
        freeze_path=pathlib.Path(args.freeze), bucket_name=args.bucket,
        output_root=pathlib.Path(args.output_root).resolve(),
    )
    print(json.dumps({"artifact_count": receipt["artifact_count"], "status": "PASS"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
