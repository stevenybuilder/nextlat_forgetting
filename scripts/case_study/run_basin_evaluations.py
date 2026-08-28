#!/usr/bin/env python3
"""Run every checkpoint in the frozen retrospective basin artifact roster."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
import os
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


def _scientific_output(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "nextlat_forgetting/basin_checkpoint_evaluation/1":
        raise RuntimeError(f"unexpected evaluation schema in {path}")
    return value


def run(*, project_root: pathlib.Path, artifact_root: pathlib.Path,
        upstream: pathlib.Path, output_root: pathlib.Path,
        batch_size: int) -> dict[str, Any]:
    freeze_path = project_root / "manifests/case_study/basin/artifacts.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    evaluator = project_root / "scripts/case_study/evaluate_basin_checkpoint.py"
    source_config = project_root / "configs/nextlat_lurestar.yaml"
    corpus_manifest = project_root / "manifests/corpus.sha256"
    dataset = artifact_root / "data/graph_5_5_test_20000.txt"
    if freeze["runtime_controls"]["new_training_authorized"] is not False:
        raise RuntimeError("training is not forbidden by the freeze")
    if sha256_file(dataset) != freeze["data"]["test"]["sha256"]:
        raise RuntimeError("held-out dataset hash mismatch")

    result_records: list[dict[str, Any]] = []
    for run_spec in freeze["runs"]:
        job_id = run_spec["job_id"]
        seed = int(run_spec["seed"])
        run_root = artifact_root / "runs" / job_id
        materialized = run_root / "materialized_config.yaml"
        if sha256_file(materialized) != run_spec["materialized_config_sha256"]:
            raise RuntimeError(f"materialized config mismatch for {job_id}")
        for checkpoint in run_spec["checkpoints"]:
            step = int(checkpoint["step"])
            checkpoint_path = run_root / "checkpoints" / checkpoint["filename"]
            output = output_root / job_id / f"step_{step}.json"
            command = [
                "fabric", "run", "--devices", "1", "--precision", "bf16-mixed",
                str(evaluator),
                "--job-id", job_id,
                "--seed", str(seed),
                "--step", str(step),
                "--checkpoint", str(checkpoint_path),
                "--expected-checkpoint-sha256", checkpoint["sha256"],
                "--config", str(materialized),
                "--source-config", str(source_config),
                "--dataset", str(dataset),
                "--manifest", str(corpus_manifest),
                "--upstream", str(upstream),
                "--output", str(output),
                "--batch-size", str(batch_size),
            ]
            output.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(command, cwd=upstream, check=True)
            result = _scientific_output(output)
            if result["checkpoint_sha256"] != checkpoint["sha256"] or result["step"] != step:
                raise RuntimeError(f"evaluation identity mismatch for {job_id} step {step}")
            result_records.append({
                "job_id": job_id, "seed": seed, "step": step,
                "path": str(output), "sha256": sha256_file(output),
                "exact_path_accuracy": result["exact_path_accuracy"]["value"],
                "first_decision_accuracy": result["teacher_forced_first_decision"]["accuracy"]["value"],
            })

    reference_spec = freeze["runs"][0]["checkpoints"][-1]
    reference = output_root / freeze["runs"][0]["job_id"] / f"step_{reference_spec['step']}.json"
    repeat = output_root / freeze["runs"][0]["job_id"] / f"step_{reference_spec['step']}_repeat.json"
    run_root = artifact_root / "runs" / freeze["runs"][0]["job_id"]
    repeat_command = [
        "fabric", "run", "--devices", "1", "--precision", "bf16-mixed", str(evaluator),
        "--job-id", freeze["runs"][0]["job_id"],
        "--seed", str(freeze["runs"][0]["seed"]),
        "--step", str(reference_spec["step"]),
        "--checkpoint", str(run_root / "checkpoints" / reference_spec["filename"]),
        "--expected-checkpoint-sha256", reference_spec["sha256"],
        "--config", str(run_root / "materialized_config.yaml"),
        "--source-config", str(source_config),
        "--dataset", str(dataset),
        "--manifest", str(corpus_manifest),
        "--upstream", str(upstream),
        "--output", str(repeat),
        "--batch-size", str(batch_size),
    ]
    subprocess.run(repeat_command, cwd=upstream, check=True)
    deterministic_repeat_pass = reference.read_bytes() == repeat.read_bytes()
    if not deterministic_repeat_pass:
        raise RuntimeError("clean repeated evaluation changed scientific JSON")

    index = {
        "schema": "nextlat_forgetting/basin_evaluation_index/1",
        "freeze": {"path": str(freeze_path), "sha256": sha256_file(freeze_path)},
        "evaluator": {"path": str(evaluator), "sha256": sha256_file(evaluator)},
        "upstream_commit": freeze["software"]["upstream_commit"],
        "result_count": len(result_records),
        "results": result_records,
        "deterministic_repeat": {
            "status": "PASS",
            "reference": str(reference),
            "repeat": str(repeat),
            "byte_identical": True,
            "sha256": sha256_file(reference),
        },
    }
    atomic_json(output_root / "evaluation_index.json", index)
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise SystemExit("batch size must be positive")
    index = run(
        project_root=pathlib.Path(args.project_root).resolve(),
        artifact_root=pathlib.Path(args.artifact_root).resolve(),
        upstream=pathlib.Path(args.upstream).resolve(),
        output_root=pathlib.Path(args.output_root).resolve(),
        batch_size=args.batch_size,
    )
    print(json.dumps({"result_count": index["result_count"], "status": "PASS"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
