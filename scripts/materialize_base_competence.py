#!/usr/bin/env python
"""Materialize and ledger-bind a base exact-path competence evaluation.

This tool consumes a real evaluator's structured output; it has no ``--accuracy`` escape hatch.
It verifies the evaluated checkpoint, held-out dataset, evaluator source, manifests, deterministic
decoding regime, and exact integer counts before atomically writing the canonical receipt and
promoting the append-only run ledger from TRAINED to DONE. Re-running a valid promotion is a no-op.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from lurestar.durable_checkpoint import atomic_write_json, atomic_write_text, sha256_file  # noqa: E402
from run_matrix import (  # noqa: E402
    COMPETENCE_DECODING,
    COMPETENCE_RECEIPT,
    COMPETENCE_RECEIPT_SIDECAR,
    DONE,
    TRAINED,
    Ledger,
    competence_identity_from_paths,
    verify_base_competence_receipt,
    verify_parent_training_artifacts,
)


def _record(path: pathlib.Path) -> dict[str, str]:
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"required provenance file is missing: {path}")
    return {"path": str(path), "sha256": sha256_file(path)}


def materialize(
    *, ledger_path: pathlib.Path, job_id: str, evaluator_output_path: pathlib.Path,
    evaluator_path: pathlib.Path, dataset_path: pathlib.Path,
    manifest_paths: list[pathlib.Path],
) -> dict:
    ledger = Ledger(ledger_path)
    parent = ledger.state_of(job_id)
    if not parent:
        raise RuntimeError(f"no ledger state exists for {job_id}")
    if parent.get("status") == DONE:
        return verify_base_competence_receipt(
            parent,
            expected_job_id=job_id,
            model=str(parent.get("model")),
            seed=int(parent.get("seed")),
        )
    if parent.get("status") != TRAINED:
        raise RuntimeError(f"{job_id} is {parent.get('status')}, not TRAINED")
    if parent.get("phase") != "base":
        raise RuntimeError("competence receipts can only promote base jobs")
    if not manifest_paths:
        raise RuntimeError("at least one evaluation manifest is required")

    # This comparison happens before any receipt write.  Evaluation inputs are a property of
    # the base job that was frozen before training, never a post-hoc choice made after seeing the
    # checkpoint or a preliminary score.
    verify_parent_training_artifacts(parent)
    frozen_identity = parent.get("competence_identity")
    current_identity = competence_identity_from_paths(
        evaluator_path, dataset_path, manifest_paths
    )
    if not isinstance(frozen_identity, dict) or current_identity != frozen_identity:
        raise RuntimeError(
            "evaluation inputs do not exactly equal the pre-training competence identity"
        )

    evaluator_output = _record(evaluator_output_path)
    evaluator = current_identity["evaluator"]
    dataset = current_identity["dataset"]
    manifests = current_identity["manifests"]
    try:
        raw = json.loads(evaluator_output_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError("evaluator output is not valid JSON") from exc
    if raw.get("schema") != "nextlat_forgetting/exact_path_evaluation/1":
        raise RuntimeError("evaluator output has an unsupported schema")

    expected = {
        "job_id": job_id,
        "model": parent.get("model"),
        "seed": parent.get("seed"),
        "checkpoint_sha256": parent.get("final_checkpoint_sha256"),
        "dataset_sha256": dataset["sha256"],
        "evaluator_sha256": evaluator["sha256"],
        "manifest_sha256s": sorted(record["sha256"] for record in manifests),
        "decoding": COMPETENCE_DECODING,
    }
    for key, value in expected.items():
        if raw.get(key) != value:
            raise RuntimeError(
                f"evaluator output {key} does not match frozen provenance: "
                f"{raw.get(key)!r} != {value!r}"
            )
    checkpoint_path = pathlib.Path(str(parent.get("final_checkpoint", "")))
    checkpoint_sha = parent.get("final_checkpoint_sha256")
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != checkpoint_sha:
        raise RuntimeError("TRAINED parent checkpoint is missing or its SHA no longer matches")

    out_root = pathlib.Path(str(parent.get("out_root", "")))
    receipt_path = out_root / COMPETENCE_RECEIPT
    sidecar_path = out_root / COMPETENCE_RECEIPT_SIDECAR
    receipt = {
        "schema": "nextlat_forgetting/base_competence/1",
        "job_id": job_id,
        "model": parent["model"],
        "seed": parent["seed"],
        "phase": "base",
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_sha,
        },
        "evaluator": evaluator,
        "evaluator_output": evaluator_output,
        "evaluation_dataset": dataset,
        "manifests": manifests,
        "decoding": COMPETENCE_DECODING,
        "competence_identity": frozen_identity,
        "exact_path_accuracy": raw.get("exact_path_accuracy"),
    }
    atomic_write_json(receipt_path, receipt)
    receipt_sha = sha256_file(receipt_path)
    atomic_write_text(sidecar_path, f"{receipt_sha}  {receipt_path.name}\n")

    promoted = {
        key: value for key, value in parent.items() if key not in ("seq", "ts", "status")
    }
    competence_artifacts = {
        COMPETENCE_RECEIPT: receipt_sha,
        COMPETENCE_RECEIPT_SIDECAR: sha256_file(sidecar_path),
    }
    promoted.update({
        "job_id": job_id,
        "status": DONE,
        "supersedes": parent.get("seq"),
        "artifacts": dict(parent.get("artifacts", {}), **competence_artifacts),
        "evaluation_artifacts": competence_artifacts,
    })
    # Validate the exact entry before making it authoritative in the append-only ledger.
    verify_base_competence_receipt(
        promoted,
        expected_job_id=job_id,
        model=str(parent["model"]),
        seed=int(parent["seed"]),
    )
    ledger.append(promoted)
    return receipt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--evaluator-output", required=True)
    ap.add_argument("--evaluator", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--manifest", action="append", required=True)
    args = ap.parse_args(argv)
    try:
        receipt = materialize(
            ledger_path=pathlib.Path(args.ledger),
            job_id=args.job_id,
            evaluator_output_path=pathlib.Path(args.evaluator_output),
            evaluator_path=pathlib.Path(args.evaluator),
            dataset_path=pathlib.Path(args.dataset),
            manifest_paths=[pathlib.Path(path) for path in args.manifest],
        )
    except RuntimeError as exc:
        print(f"[materialize_base_competence] REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
