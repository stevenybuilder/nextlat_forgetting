#!/usr/bin/env python
"""Evaluate every TRAINED base job and atomically promote valid receipts to DONE.

This is the fail-closed bridge between ``run_matrix.py --phase base`` and any adaptation plan.
It is intentionally idempotent: a verified DONE job is skipped, a valid raw result left by a
disconnect is materialized without repeating inference, and a failed evaluation never mutates the
append-only ledger.  Adaptation remains impossible because ``run_matrix.py`` accepts only DONE
parents with verified competence receipts.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import subprocess
import sys

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "scripts"))

from lurestar.durable_checkpoint import sha256_file  # noqa: E402
from materialize_base_competence import materialize  # noqa: E402
from run_matrix import (  # noqa: E402
    DONE,
    MODELS,
    SEEDS,
    TRAINED,
    Ledger,
    competence_identity_from_paths,
    verify_base_competence_receipt,
    verify_parent_training_artifacts,
)


class LifecycleError(RuntimeError):
    """A production evaluation lifecycle invariant failed."""


def expected_job_ids(models: list[str], seeds: list[int]) -> list[str]:
    return [f"{model}-s{seed}-base" for model in models for seed in seeds]


def materialized_config_for(parent: dict) -> pathlib.Path:
    checkpoint = pathlib.Path(str(parent.get("final_checkpoint", ""))).resolve()
    path = checkpoint.parent / "materialized_config.yaml"
    if not path.is_file():
        raise LifecycleError(f"materialized training config is missing beside {checkpoint}")
    return path


def verify_parent_inputs(parent: dict, job_id: str) -> tuple[pathlib.Path, pathlib.Path]:
    if parent.get("status") not in (TRAINED, DONE):
        raise LifecycleError(f"{job_id} is {parent.get('status') or 'missing'}, not TRAINED/DONE")
    if parent.get("phase") != "base" or parent.get("job_id") != job_id:
        raise LifecycleError(f"{job_id} ledger identity is not a base job")
    checkpoint = pathlib.Path(str(parent.get("final_checkpoint", ""))).resolve()
    expected_checkpoint_sha = parent.get("final_checkpoint_sha256")
    if not checkpoint.is_file() or sha256_file(checkpoint) != expected_checkpoint_sha:
        raise LifecycleError(f"{job_id} checkpoint is missing or does not match the ledger")
    source_config = pathlib.Path(str(parent.get("config", ""))).resolve()
    expected_config_sha = parent.get("config_sha256")
    if not source_config.is_file() or sha256_file(source_config) != expected_config_sha:
        raise LifecycleError(f"{job_id} source config is missing or does not match the ledger")
    try:
        verify_parent_training_artifacts(parent)
    except RuntimeError as exc:
        raise LifecycleError(f"{job_id} has invalid training artifacts: {exc}") from exc
    return checkpoint, source_config


def evaluator_command(
    *, evaluator: pathlib.Path, parent: dict, checkpoint: pathlib.Path,
    materialized_config: pathlib.Path, source_config: pathlib.Path, dataset: pathlib.Path,
    manifests: list[pathlib.Path], upstream: pathlib.Path, output: pathlib.Path,
    devices: int, precision: str, batch_size: int,
) -> list[str]:
    command = [
        "fabric", "run", "--devices", str(devices), "--precision", precision,
        str(evaluator),
        "--job-id", str(parent["job_id"]),
        "--model", str(parent["model"]),
        "--seed", str(parent["seed"]),
        "--checkpoint", str(checkpoint),
        "--config", str(materialized_config),
        "--source-config", str(source_config),
        "--dataset", str(dataset),
        "--upstream", str(upstream),
        "--output", str(output),
        "--batch-size", str(batch_size),
    ]
    for manifest in manifests:
        command.extend(("--manifest", str(manifest)))
    return command


def relay(command: list[str], *, cwd: pathlib.Path) -> int:
    print("+ " + shlex.join(command), flush=True)
    process = subprocess.Popen(
        command, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print("  | " + line.rstrip(), flush=True)
    return process.wait()


def run_lifecycle(
    *, ledger_path: pathlib.Path, models: list[str], seeds: list[int],
    evaluator: pathlib.Path, dataset: pathlib.Path, manifests: list[pathlib.Path],
    upstream: pathlib.Path, devices: int, precision: str, batch_size: int,
    command_runner=relay,
) -> dict[str, str]:
    ledger = Ledger(ledger_path)
    states = ledger.states()
    wanted = expected_job_ids(models, seeds)
    missing = [job_id for job_id in wanted if job_id not in states]
    if missing:
        raise LifecycleError(f"base evaluation cannot start; ledger jobs are missing: {missing}")
    if not evaluator.is_file():
        raise LifecycleError(f"production evaluator is missing: {evaluator}")
    if not dataset.is_file():
        raise LifecycleError(f"held-out dataset is missing: {dataset}")
    if not manifests or any(not path.is_file() for path in manifests):
        raise LifecycleError("one or more frozen evaluation manifests are missing")
    try:
        requested_identity = competence_identity_from_paths(evaluator, dataset, manifests)
    except (RuntimeError, FileNotFoundError) as exc:
        raise LifecycleError(f"invalid requested competence identity: {exc}") from exc

    outcomes: dict[str, str] = {}
    for job_id in wanted:
        parent = ledger.state_of(job_id)
        assert parent is not None
        checkpoint, source_config = verify_parent_inputs(parent, job_id)
        if parent.get("competence_identity") != requested_identity:
            raise LifecycleError(
                f"{job_id} evaluator/dataset/manifest identity differs from the identity "
                "frozen before base training"
            )
        if parent["status"] == DONE:
            verify_base_competence_receipt(
                parent, expected_job_id=job_id,
                model=str(parent["model"]), seed=int(parent["seed"]),
            )
            outcomes[job_id] = "verified-DONE"
            continue

        materialized_config = materialized_config_for(parent)
        output = pathlib.Path(str(parent["out_root"])) / "evaluation" / "exact_path_raw.json"

        # A disconnect after atomic evaluator output but before promotion must not repeat GPU
        # inference.  Try the complete provenance/materialization check first.
        if output.is_file():
            try:
                materialize(
                    ledger_path=ledger_path, job_id=job_id,
                    evaluator_output_path=output, evaluator_path=evaluator,
                    dataset_path=dataset, manifest_paths=manifests,
                )
                outcomes[job_id] = "recovered-raw-and-promoted"
                continue
            except RuntimeError as exc:
                print(f"[{job_id}] stale/incomplete raw result will be replaced: {exc}", flush=True)

        command = evaluator_command(
            evaluator=evaluator, parent=parent, checkpoint=checkpoint,
            materialized_config=materialized_config, source_config=source_config,
            dataset=dataset, manifests=manifests, upstream=upstream, output=output,
            devices=devices, precision=precision, batch_size=batch_size,
        )
        if command_runner(command, cwd=upstream) != 0:
            raise LifecycleError(f"{job_id} deterministic evaluator failed; ledger remains TRAINED")
        materialize(
            ledger_path=ledger_path, job_id=job_id,
            evaluator_output_path=output, evaluator_path=evaluator,
            dataset_path=dataset, manifest_paths=manifests,
        )
        outcomes[job_id] = "evaluated-and-promoted"
    return outcomes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument(
        "--evaluator", default=str(_REPO / "scripts" / "evaluate_base_competence.py")
    )
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--precision", default="bf16-mixed")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args(argv)
    if args.devices != 1:
        print("[evaluate_trained_bases] REFUSED: competence gate is frozen to one device",
              file=sys.stderr)
        return 2
    try:
        outcomes = run_lifecycle(
            ledger_path=pathlib.Path(args.ledger), models=list(args.models),
            seeds=list(args.seeds), evaluator=pathlib.Path(args.evaluator).resolve(),
            dataset=pathlib.Path(args.dataset).resolve(),
            manifests=[pathlib.Path(path).resolve() for path in args.manifest],
            upstream=pathlib.Path(args.upstream).resolve(), devices=args.devices,
            precision=args.precision, batch_size=args.batch_size,
        )
    except (LifecycleError, RuntimeError, OSError, ValueError) as exc:
        print(f"[evaluate_trained_bases] REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(outcomes, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
