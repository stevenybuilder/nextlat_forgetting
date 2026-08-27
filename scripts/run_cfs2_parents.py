#!/usr/bin/env python3
"""Train the three independent CFS-2-only NextLat parents.

This is a small provider-neutral launcher around the existing base MatrixRunner.
It has no CFS adaptation or evaluation path and cannot change the frozen parent
roster or 20,000-update target.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import typing as t

import yaml


REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from lurestar.durable_checkpoint import DurableSync  # noqa: E402
from run_matrix import (  # noqa: E402
    DEFAULT_COMPETENCE_EVALUATOR,
    DEFAULT_COMPETENCE_MANIFESTS,
    DEFAULT_MANIFESTS,
    FabricLauncher,
    JobSpec,
    Ledger,
    MatrixRunner,
    TRAINING_TERMINAL,
    validate_matrix,
)


CFS2_PARENT_SEEDS = (2234, 2235, 2236)
CFS2_PARENT_STEPS = 20_000
CFS2_PARENT_CONFIG = REPO / "configs" / "cfs2_nextlat_parent.yaml"
CFS2_PARENT_DECISION = REPO / "docs" / "DECISION_CFS2_STIMULUS_REPAIR.md"


class CFS2ParentError(RuntimeError):
    """The CFS-2-only parent contract is missing or changed."""


def source_parent_id(seed: int) -> str:
    if seed not in CFS2_PARENT_SEEDS:
        raise CFS2ParentError(f"seed {seed} is outside the frozen CFS-2-only roster")
    return f"nextlat-s{seed}-cfs2-base"


def validate_parent_config(path: os.PathLike[str] | str = CFS2_PARENT_CONFIG) -> pathlib.Path:
    config_path = pathlib.Path(path).resolve()
    try:
        value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        reference = yaml.safe_load((REPO / "configs" / "nextlat_lurestar.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CFS2ParentError("CFS-2 parent or reference NextLat config is unreadable") from exc
    if not isinstance(value, dict) or not isinstance(reference, dict):
        raise CFS2ParentError("CFS-2 parent and reference configs must be mappings")
    contract = value.get("cfs2_parent")
    if contract != {
        "schema": "nextlat_forgetting/cfs2_parent_config/1",
        "exact_training_steps": CFS2_PARENT_STEPS,
        "seeds": list(CFS2_PARENT_SEEDS),
        "architecture_and_objective_equal_to_planned_nextlat_bases": True,
    }:
        raise CFS2ParentError("CFS-2 parent config lacks the frozen parent contract")
    if value.get("use_nextlat") is not True or value.get("use_bst") is not False:
        raise CFS2ParentError("CFS-2 parent must select NextLat only")
    for section in ("data", "model", "optimizer", "lr_scheduler"):
        if value.get(section) != reference.get(section):
            raise CFS2ParentError(f"CFS-2 parent {section} differs from the planned NextLat base")
    trainer = value.get("trainer")
    if not isinstance(trainer, dict) or trainer.get("train_batches") != CFS2_PARENT_STEPS or trainer.get("init_from") != "scratch":
        raise CFS2ParentError("CFS-2 parent must train from scratch for exactly 20,000 updates")
    return config_path


def build_cfs2_parent_jobs(
    root: os.PathLike[str] | str,
    *,
    seeds: t.Sequence[int] = CFS2_PARENT_SEEDS,
    config: os.PathLike[str] | str = CFS2_PARENT_CONFIG,
    competence_evaluator: os.PathLike[str] | str = DEFAULT_COMPETENCE_EVALUATOR,
    competence_dataset: os.PathLike[str] | str | None = None,
    competence_manifests: t.Sequence[os.PathLike[str] | str] = DEFAULT_COMPETENCE_MANIFESTS,
) -> list[JobSpec]:
    requested = tuple(int(seed) for seed in seeds)
    if len(set(requested)) != len(requested) or any(seed not in CFS2_PARENT_SEEDS for seed in requested):
        raise CFS2ParentError("requested seeds must be a unique subset of 2234, 2235, 2236")
    config_path = validate_parent_config(config)
    root_path = pathlib.Path(root).resolve()
    if competence_dataset is None:
        staged = root_path / "data" / "stargraph" / "graph_5_5_test_20000.txt"
        competence_dataset = staged if staged.is_file() else REPO / "data" / "stargraph" / "graph_5_5_test_20000.txt"
    manifests = tuple(DEFAULT_MANIFESTS["base"]) + (str(CFS2_PARENT_DECISION),)
    jobs = [
        JobSpec(
            job_id=source_parent_id(seed), model="nextlat", seed=seed,
            phase="base", condition=None, config=str(config_path),
            out_root=str(root_path / "runs" / "cfs2_parents" / "nextlat" / f"seed{seed}" / "base" / "_"),
            manifests=manifests, competence_evaluator=str(competence_evaluator),
            competence_dataset=str(competence_dataset),
            competence_manifests=tuple(str(path) for path in competence_manifests),
            train_batches=CFS2_PARENT_STEPS, final_artifacts=(), overrides=(),
        )
        for seed in requested
    ]
    validate_matrix(jobs)
    return jobs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--ledger", default=None)
    parser.add_argument("--upstream", default=str(REPO / "upstream" / "NextLat"))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(CFS2_PARENT_SEEDS))
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--strategy", default="ddp")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--bucket", default=os.environ.get("LURESTAR_BUCKET"))
    parser.add_argument("--gcs-prefix", default="cfs2/parents")
    args = parser.parse_args(argv)

    root = pathlib.Path(args.root).resolve()
    ledger_path = pathlib.Path(args.ledger).resolve() if args.ledger else root / "control" / "cfs2_parent_run_ledger.json"
    jobs = build_cfs2_parent_jobs(root, seeds=args.seeds)
    if args.print_plan:
        print(json.dumps([job.to_dict() for job in jobs], indent=2))
        return 0
    sync = DurableSync(args.bucket, args.gcs_prefix, "parent-matrix", logger=print) if args.bucket else None
    launcher = FabricLauncher(
        args.upstream, devices=args.devices, precision=args.precision,
        strategy=args.strategy, dry_run=args.dry_run,
    )
    states = MatrixRunner(Ledger(ledger_path), launcher, sync=sync).run(jobs)
    incomplete = [job.job_id for job in jobs if states.get(job.job_id, {}).get("status") not in TRAINING_TERMINAL]
    if incomplete:
        print(f"[cfs2-parents] not terminal: {incomplete}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

