#!/usr/bin/env python3
"""Crash-safe runner for the repaired CFS-2 factorial adaptation matrix.

The runner is intentionally unable to evaluate.  It can only make the exact
8-parent x 2-episode x 4-arm matrix durable under ``runs/cfs2``.  A CFS-1
manifest, ledger, path, or parent alias cannot enter this runner accidentally.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import os
import pathlib
import pickle
import subprocess
import sys
import time
import typing as t

import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cfs2.adaptation import (  # noqa: E402
    CFS2_ARMS, CFS2_CONTRACT, CFS2_EPISODES, CFS2_EVALUATION_INPUTS,
    CFS2_PARENT_SEEDS, CFS2_UPDATE_STEPS, canonical_json_sha256,
    cfs2_branch_order, contract_sha256, sha256_file, validate_nextlat_ce_only_config,
    validate_update_manifest,
)
from lurestar.durable_checkpoint import (  # noqa: E402
    DurableCheckpointer, DurableSync, atomic_write_json, pickle_serializer,
)


CFS2_LEDGER_SCHEMA = "nextlat_forgetting/cfs2_run_ledger/1"
CFS2_COMPLETION_SCHEMA = "nextlat_forgetting/cfs2_training_completion/1"
CFS2_CLONE_SCHEMA = "nextlat_forgetting/cfs2_clone_provenance/1"
CFS2_READINESS_SCHEMA = "nextlat_forgetting/cfs2_pre_evaluation_readiness/1"
CFS2_PARENT_LINEAGE_SCHEMA = "nextlat_forgetting/cfs2_parent_lineage_receipt/1"
CFS2_PARENT_LINEAGE_SCHEMA_V2 = "nextlat_forgetting/cfs2_parent_lineage_receipt/2"
CFS2_TRAINED, CFS2_RUNNING, CFS2_INTERRUPTED, CFS2_FAILED, CFS2_STALE = (
    "TRAINED", "RUNNING", "INTERRUPTED", "FAILED", "STALE"
)
TERMINAL = frozenset((CFS2_TRAINED, "DONE"))
PARENT_STEPS = 20_000
CONFIG_RELATIVE = pathlib.Path("configs/cfs2_nextlat_adapt.yaml")


class CFS2MatrixError(RuntimeError):
    """A CFS-2 input, clone, lineage, durability, or readiness rule failed."""


@dataclasses.dataclass(frozen=True)
class CFS2Job:
    job_id: str
    parent_id: str
    seed: int
    episode: int
    arm: str
    overlap: str
    future_relation: str
    config: str
    out_root: str
    update_bank: str
    update_bank_sha256: str
    manifest_path: str
    manifest_sha256: str

    @property
    def experiment_name(self) -> str:
        return self.job_id

    @property
    def experiment_dir(self) -> str:
        return str(pathlib.Path(self.out_root) / self.experiment_name)

    def to_dict(self) -> dict[str, t.Any]:
        return dataclasses.asdict(self) | {
            "experiment_name": self.experiment_name, "experiment_dir": self.experiment_dir,
        }


@dataclasses.dataclass(frozen=True)
class ParentCheckpoint:
    parent_id: str
    source_parent_id: str
    seed: int
    path: str
    sha256: str
    step: int
    optimizer_present: bool
    rng_present: bool
    lineage_receipt_sha256: str | None


@dataclasses.dataclass
class ResumePlan:
    spec: CFS2Job
    parent: ParentCheckpoint
    fresh: bool
    resume_step: int
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None

    @property
    def target_step(self) -> int:
        return self.parent.step + CFS2_UPDATE_STEPS


@dataclasses.dataclass(frozen=True)
class LaunchResult:
    returncode: int
    detail: str = ""


class CFS2Ledger:
    """Append-only CFS-2 state, separate from every legacy experiment ledger."""

    def __init__(self, path: os.PathLike[str] | str) -> None:
        self.path = pathlib.Path(path)

    def entries(self) -> list[dict[str, t.Any]]:
        if not self.path.is_file():
            return []
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CFS2MatrixError("CFS-2 ledger is unreadable") from exc
        if value.get("schema") != CFS2_LEDGER_SCHEMA or not isinstance(value.get("entries"), list):
            raise CFS2MatrixError("CFS-2 ledger schema/entries are invalid")
        return list(value["entries"])

    def append(self, entry: t.Mapping[str, t.Any]) -> dict[str, t.Any]:
        entries = self.entries()
        record = dict(entry) | {"seq": len(entries), "ts": time.time()}
        entries.append(record)
        atomic_write_json(self.path, {"schema": CFS2_LEDGER_SCHEMA, "entries": entries})
        return record

    def states(self) -> dict[str, dict[str, t.Any]]:
        return {entry["job_id"]: entry for entry in self.entries() if isinstance(entry.get("job_id"), str)}


def parent_id_for_seed(seed: int) -> str:
    if seed not in CFS2_PARENT_SEEDS:
        raise ValueError("seed is not in the frozen CFS-2 parent roster")
    return f"nextlat-seed{seed}" + ("-base" if seed < 2000 else "-cfs2-base")


def cfs2_job_id(seed: int, episode: int, arm: str) -> str:
    if seed not in CFS2_PARENT_SEEDS or episode not in CFS2_EPISODES or arm not in CFS2_ARMS:
        raise ValueError("CFS-2 job identity is outside the frozen matrix")
    return f"cfs2-nextlat-seed{seed}-episode{episode}-{arm}"


def _load_yaml(path: pathlib.Path) -> dict[str, t.Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CFS2MatrixError("CFS-2 config is unreadable") from exc
    if not isinstance(value, dict):
        raise CFS2MatrixError("CFS-2 config must be a mapping")
    validate_nextlat_ce_only_config(value)
    cfs2 = value.get("cfs2")
    data = value.get("data")
    if not isinstance(cfs2, dict) or cfs2.get("adaptation_steps") != CFS2_UPDATE_STEPS:
        raise CFS2MatrixError("CFS-2 config lacks the exact 500-update contract")
    if not isinstance(data, dict) or cfs2.get("unbound_bank_sentinel") not in {
        data.get("stargraph_train_data_path"), data.get("stargraph_test_data_path")
    }:
        raise CFS2MatrixError("CFS-2 config is not safely unbound")
    return value


def build_cfs2_matrix(root: os.PathLike[str] | str, manifest_path: os.PathLike[str] | str,
                      *, config: os.PathLike[str] | str = REPO / CONFIG_RELATIVE) -> tuple[list[CFS2Job], dict[str, t.Any]]:
    root_path = pathlib.Path(root).resolve()
    manifest = validate_update_manifest(manifest_path)
    config_path = pathlib.Path(config).resolve()
    _load_yaml(config_path)
    jobs: list[CFS2Job] = []
    for seed in CFS2_PARENT_SEEDS:
        for episode in CFS2_EPISODES:
            for arm in CFS2_ARMS:
                overlap, relation = arm.split("_", 1)
                bank = manifest["episodes"][episode]["arms"][arm]
                jobs.append(CFS2Job(
                    job_id=cfs2_job_id(seed, episode, arm), parent_id=parent_id_for_seed(seed),
                    seed=seed, episode=episode, arm=arm, overlap=overlap, future_relation=relation,
                    config=str(config_path),
                    out_root=str(root_path / "runs" / "cfs2" / "nextlat" / f"seed{seed}" / f"episode{episode}" / arm),
                    update_bank=str(pathlib.Path(manifest["root"]) / bank["path"]),
                    update_bank_sha256=bank["sha256"], manifest_path=manifest["path"], manifest_sha256=manifest["sha256"],
                ))
    if len(jobs) != 64 or len({job.job_id for job in jobs}) != 64:
        raise CFS2MatrixError("CFS-2 matrix must contain exactly 64 unique branches")
    outputs = [pathlib.Path(job.out_root).resolve() for job in jobs]
    if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(outputs) for b in outputs[i + 1:]):
        raise CFS2MatrixError("CFS-2 branch roots overlap")
    return jobs, manifest


def _parent_state_loader(path: pathlib.Path) -> dict[str, t.Any]:
    try:
        import torch  # type: ignore
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        with path.open("rb") as handle:
            value = pickle.load(handle)
    if not isinstance(value, dict):
        raise CFS2MatrixError("parent checkpoint did not deserialize to a mapping")
    return value


def _verify_parent(record: t.Mapping[str, t.Any], *, expected_id: str, expected_seed: int,
                   loader: t.Callable[[pathlib.Path], dict[str, t.Any]]) -> ParentCheckpoint:
    if record.get("job_id") != expected_id or record.get("model") != "nextlat" or record.get("status") not in TERMINAL:
        raise CFS2MatrixError(f"parent {expected_id} is not a terminal NextLat base")
    path_text, digest, step = record.get("final_checkpoint"), record.get("final_checkpoint_sha256"), record.get("step")
    if not isinstance(path_text, str) or not isinstance(digest, str) or len(digest) != 64 or step != PARENT_STEPS:
        raise CFS2MatrixError(f"parent {expected_id} lacks exact 20,000-step checkpoint binding")
    path = pathlib.Path(path_text).resolve()
    if not path.is_file() or sha256_file(path) != digest:
        raise CFS2MatrixError(f"parent {expected_id} checkpoint SHA is stale")
    state = loader(path)
    if state.get("training_steps") != PARENT_STEPS or not isinstance(state.get("model"), dict) or not isinstance(state.get("optimizer"), dict) or not isinstance(state.get("lurestar_rng_state_v1"), dict):
        raise CFS2MatrixError("CFS-2 requires exact parent model, optimizer, RNG, and 20,000 steps")
    return ParentCheckpoint(expected_id, expected_id, expected_seed, str(path), digest, PARENT_STEPS, True, True, None)


def _lineage_aliases(lineage_path: os.PathLike[str] | str | None, parent_ledger: pathlib.Path,
                     ledger_sha: str, latest: Mapping[str, dict[str, t.Any]],
                     loader: t.Callable[[pathlib.Path], dict[str, t.Any]]) -> dict[str, ParentCheckpoint]:
    if lineage_path is None:
        return {}
    receipt_path = pathlib.Path(lineage_path).resolve()
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CFS2MatrixError("CFS-2 parent-lineage receipt is unreadable") from exc
    if receipt.get("schema") == CFS2_PARENT_LINEAGE_SCHEMA_V2:
        return _lineage_v2_aliases(
            receipt, receipt_path=receipt_path, parent_ledger=parent_ledger,
            ledger_sha=ledger_sha, latest=latest, loader=loader,
        )
    if set(receipt) != {"schema", "status", "source_parent_ledger", "aliases"} or receipt.get("schema") != CFS2_PARENT_LINEAGE_SCHEMA or receipt.get("status") != "FROZEN":
        raise CFS2MatrixError("CFS-2 parent-lineage receipt schema/status is invalid")
    source = receipt["source_parent_ledger"]
    if not isinstance(source, dict) or set(source) != {"path", "sha256"} or source.get("sha256") != ledger_sha:
        raise CFS2MatrixError("CFS-2 lineage receipt does not hash-bind the supplied parent ledger")
    source_path = pathlib.Path(str(source["path"]))
    if not source_path.is_absolute():
        source_path = receipt_path.parent / source_path
    if source_path.resolve() != parent_ledger.resolve():
        raise CFS2MatrixError("CFS-2 lineage receipt names a different parent ledger")
    aliases = receipt.get("aliases")
    if not isinstance(aliases, list) or len(aliases) != 3:
        raise CFS2MatrixError("CFS-2 needs exactly three explicit CFS-only aliases")
    out: dict[str, ParentCheckpoint] = {}
    receipt_sha = sha256_file(receipt_path)
    for seed in (2234, 2235, 2236):
        canonical, source_id = parent_id_for_seed(seed), f"nextlat-seed{seed}-cfs1-base"
        rows = [row for row in aliases if isinstance(row, dict) and row.get("seed") == seed]
        if len(rows) != 1:
            raise CFS2MatrixError(f"CFS-2 lineage receipt is missing unique alias for seed {seed}")
        row = rows[0]
        if set(row) != {"seed", "canonical_parent_id", "source_parent_id", "source_ledger_entry_sha256", "parent_checkpoint"} or row.get("canonical_parent_id") != canonical or row.get("source_parent_id") != source_id:
            raise CFS2MatrixError(f"CFS-2 alias identity is invalid for seed {seed}")
        source_record = latest.get(source_id)
        if source_record is None or row.get("source_ledger_entry_sha256") != canonical_json_sha256(source_record):
            raise CFS2MatrixError(f"CFS-2 alias source ledger entry is not hash-bound for seed {seed}")
        parent = _verify_parent(source_record, expected_id=source_id, expected_seed=seed, loader=loader)
        checkpoint = row.get("parent_checkpoint")
        if checkpoint != {"path": parent.path, "sha256": parent.sha256}:
            raise CFS2MatrixError(f"CFS-2 alias checkpoint differs from source parent for seed {seed}")
        out[canonical] = dataclasses.replace(parent, parent_id=canonical, source_parent_id=source_id,
                                             lineage_receipt_sha256=receipt_sha)
    return out


def _lineage_v2_aliases(
    receipt: t.Mapping[str, t.Any], *, receipt_path: pathlib.Path,
    parent_ledger: pathlib.Path, ledger_sha: str,
    latest: t.Mapping[str, dict[str, t.Any]],
    loader: t.Callable[[pathlib.Path], dict[str, t.Any]],
) -> dict[str, ParentCheckpoint]:
    """Resolve the eight real run-ledger IDs to CFS-2's canonical parent IDs."""
    if set(receipt) != {"schema", "status", "source_ledgers", "parent_ledger", "parents"} or receipt.get("status") != "FROZEN":
        raise CFS2MatrixError("CFS-2 v2 parent-lineage receipt schema/status is invalid")
    bound = receipt.get("parent_ledger")
    if not isinstance(bound, dict) or set(bound) != {"path", "sha256"} or bound.get("sha256") != ledger_sha:
        raise CFS2MatrixError("CFS-2 v2 receipt does not hash-bind the supplied parent ledger")
    bound_path = pathlib.Path(str(bound["path"]))
    if not bound_path.is_absolute():
        bound_path = receipt_path.parent / bound_path
    if bound_path.resolve() != parent_ledger.resolve():
        raise CFS2MatrixError("CFS-2 v2 receipt names a different parent ledger")

    raw_sources = receipt.get("source_ledgers")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise CFS2MatrixError("CFS-2 v2 receipt lacks source-ledger bindings")
    sources: dict[str, str] = {}
    source_states: dict[str, dict[str, dict[str, t.Any]]] = {}
    for source in raw_sources:
        if not isinstance(source, dict) or set(source) != {"path", "sha256"}:
            raise CFS2MatrixError("CFS-2 v2 source-ledger binding is invalid")
        source_path = pathlib.Path(str(source["path"]))
        if not source_path.is_absolute():
            source_path = receipt_path.parent / source_path
        source_path = source_path.resolve()
        digest = source.get("sha256")
        if not source_path.is_file() or not isinstance(digest, str) or sha256_file(source_path) != digest:
            raise CFS2MatrixError("CFS-2 v2 source ledger is missing or stale")
        if str(source_path) in sources:
            raise CFS2MatrixError("CFS-2 v2 source-ledger bindings are duplicated")
        sources[str(source_path)] = digest
        try:
            source_document = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CFS2MatrixError("CFS-2 v2 source ledger is unreadable") from exc
        source_entries = source_document.get("entries")
        if not isinstance(source_entries, list):
            raise CFS2MatrixError("CFS-2 v2 source ledger lacks append-only entries")
        source_states[str(source_path)] = {
            entry["job_id"]: entry for entry in source_entries
            if isinstance(entry, dict) and isinstance(entry.get("job_id"), str)
        }

    rows = receipt.get("parents")
    if not isinstance(rows, list) or len(rows) != len(CFS2_PARENT_SEEDS):
        raise CFS2MatrixError("CFS-2 v2 receipt must bind exactly eight parents")
    receipt_sha = sha256_file(receipt_path)
    out: dict[str, ParentCheckpoint] = {}
    for seed in CFS2_PARENT_SEEDS:
        canonical = parent_id_for_seed(seed)
        source_id = f"nextlat-s{seed}-base" if seed < 2000 else f"nextlat-s{seed}-cfs2-base"
        matching = [row for row in rows if isinstance(row, dict) and row.get("seed") == seed]
        if len(matching) != 1:
            raise CFS2MatrixError(f"CFS-2 v2 receipt lacks unique parent seed {seed}")
        row = matching[0]
        if set(row) != {
            "seed", "canonical_parent_id", "source_parent_id", "source_ledger",
            "source_ledger_entry_sha256", "parent_checkpoint",
        } or row.get("canonical_parent_id") != canonical or row.get("source_parent_id") != source_id:
            raise CFS2MatrixError(f"CFS-2 v2 parent identity is invalid for seed {seed}")
        source_binding = row.get("source_ledger")
        if not isinstance(source_binding, dict) or set(source_binding) != {"path", "sha256"}:
            raise CFS2MatrixError(f"CFS-2 v2 source binding is invalid for seed {seed}")
        source_path = pathlib.Path(str(source_binding["path"]))
        if not source_path.is_absolute():
            source_path = receipt_path.parent / source_path
        if sources.get(str(source_path.resolve())) != source_binding.get("sha256"):
            raise CFS2MatrixError(f"CFS-2 v2 source binding is not in the frozen ledger set for seed {seed}")
        record = latest.get(source_id)
        if record is None or row.get("source_ledger_entry_sha256") != canonical_json_sha256(record):
            raise CFS2MatrixError(f"CFS-2 v2 parent entry is not hash-bound for seed {seed}")
        original = source_states[str(source_path.resolve())].get(source_id)
        if original != record or canonical_json_sha256(original) != row.get("source_ledger_entry_sha256"):
            raise CFS2MatrixError(f"CFS-2 v2 parent entry differs from its original ledger for seed {seed}")
        parent = _verify_parent(record, expected_id=source_id, expected_seed=seed, loader=loader)
        checkpoint = row.get("parent_checkpoint")
        if checkpoint != {"path": parent.path, "sha256": parent.sha256, "training_steps": PARENT_STEPS}:
            raise CFS2MatrixError(f"CFS-2 v2 checkpoint differs from its source entry for seed {seed}")
        out[canonical] = dataclasses.replace(
            parent, parent_id=canonical, source_parent_id=source_id,
            lineage_receipt_sha256=receipt_sha,
        )
    return out


def load_parents(parent_ledger: os.PathLike[str] | str, *, lineage_receipt: os.PathLike[str] | str | None = None,
                 loader: t.Callable[[pathlib.Path], dict[str, t.Any]] = _parent_state_loader) -> dict[str, ParentCheckpoint]:
    path = pathlib.Path(parent_ledger).resolve()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CFS2MatrixError("CFS-2 parent ledger is unreadable") from exc
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise CFS2MatrixError("CFS-2 parent ledger lacks append-only entries")
    latest = {entry["job_id"]: entry for entry in entries if isinstance(entry, dict) and isinstance(entry.get("job_id"), str)}
    aliases = _lineage_aliases(lineage_receipt, path, sha256_file(path), latest, loader)
    parents: dict[str, ParentCheckpoint] = {}
    for seed in CFS2_PARENT_SEEDS:
        canonical = parent_id_for_seed(seed)
        if canonical in latest:
            parent = _verify_parent(latest[canonical], expected_id=canonical, expected_seed=seed, loader=loader)
        elif canonical in aliases:
            parent = aliases[canonical]
        else:
            requirement = "; a hash-bound CFS-2 lineage receipt is required for a CFS-1 alias" if seed >= 2000 else ""
            raise CFS2MatrixError(f"missing CFS-2 parent {canonical}{requirement}")
        parents[canonical] = parent
    return parents


class CFS2FabricLauncher:
    def __init__(self, upstream: os.PathLike[str] | str, *, devices: int = 1, precision: str = "bf16-mixed",
                 strategy: str = "ddp", dry_run: bool = False, echo: t.Callable[[str], None] = print) -> None:
        self.upstream, self.devices, self.precision, self.strategy, self.dry_run, self.echo = (
            pathlib.Path(upstream).resolve(), devices, precision, strategy, dry_run, echo)

    def command(self, plan: ResumePlan) -> list[str]:
        checkpoint = plan.parent.path if plan.fresh else plan.checkpoint_path
        if not checkpoint or plan.target_step != PARENT_STEPS + CFS2_UPDATE_STEPS:
            raise CFS2MatrixError("CFS-2 resume/target-step contract is invalid")
        return ["fabric", "run", "--devices", str(self.devices), "--strategy", self.strategy, "--precision", self.precision,
                "train.py", "--config", plan.spec.config, "--checkpoint_path", checkpoint,
                f"seed={plan.spec.seed}", f"trainer.out_dir={pathlib.Path(plan.spec.out_root).resolve()}",
                f"trainer.experiment_name={plan.spec.experiment_name}", "trainer.init_from=resume",
                f"trainer.train_batches={plan.target_step}", "trainer.compile=false", "trainer.log_to_wandb=false",
                "trainer.save_recovery_checkpoint=50", f"data.stargraph_train_data_path={plan.spec.update_bank}",
                f"data.stargraph_test_data_path={plan.spec.update_bank}"]

    def __call__(self, plan: ResumePlan) -> LaunchResult:
        command = self.command(plan)
        self.echo("[cfs2] " + plan.spec.job_id + ": " + " ".join(command))
        if self.dry_run:
            return LaunchResult(0, "dry-run")
        proc = subprocess.Popen(command, cwd=str(self.upstream), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=dict(os.environ, PYTHONUNBUFFERED="1"))
        assert proc.stdout is not None
        tail: collections.deque[str] = collections.deque(maxlen=50)
        for line in proc.stdout:
            line = line.rstrip("\n"); tail.append(line); self.echo(line)
        return LaunchResult(proc.wait(), "returncode=%d; tail=%s" % (proc.returncode, "\n".join(tail)))


class CFS2Runner:
    def __init__(self, ledger: CFS2Ledger, launcher: t.Callable[[ResumePlan], LaunchResult], *, serializer=None,
                 sync: DurableSync | None = None, echo: t.Callable[[str], None] = print) -> None:
        self.ledger, self.launcher, self.serializer, self.sync, self.echo = ledger, launcher, serializer, sync, echo

    def checkpointer(self, job: CFS2Job) -> DurableCheckpointer:
        return DurableCheckpointer(job.out_root, job.job_id, experiment_name=job.experiment_name,
                                   serializer=self.serializer, sync=self.sync, logger=self.echo)

    def identity(self, job: CFS2Job) -> dict[str, t.Any]:
        _load_yaml(pathlib.Path(job.config)); bank, manifest = pathlib.Path(job.update_bank), pathlib.Path(job.manifest_path)
        if not bank.is_file() or sha256_file(bank) != job.update_bank_sha256:
            raise CFS2MatrixError(f"{job.job_id} CFS-2 update-stream hash changed")
        if not manifest.is_file() or sha256_file(manifest) != job.manifest_sha256:
            raise CFS2MatrixError(f"{job.job_id} CFS-2 update manifest hash changed")
        return {"parent_id": job.parent_id, "seed": job.seed, "episode": job.episode, "arm": job.arm,
                "overlap": job.overlap, "future_relation": job.future_relation,
                "out_root": str(pathlib.Path(job.out_root).resolve()), "config_sha256": sha256_file(job.config),
                "update_bank_sha256": job.update_bank_sha256, "update_manifest_sha256": job.manifest_sha256}

    def _clone_receipt(self, job: CFS2Job, parent: ParentCheckpoint) -> pathlib.Path:
        path = pathlib.Path(job.out_root) / "clone_provenance.json"
        value = {"schema": CFS2_CLONE_SCHEMA, "job_id": job.job_id, "parent_id": parent.parent_id,
                 "source_parent_id": parent.source_parent_id, "parent_checkpoint": {"path": parent.path, "sha256": parent.sha256, "training_steps": parent.step},
                 "optimizer_state": "present_in_exact_parent_checkpoint", "rng_state": "present_in_exact_parent_checkpoint",
                 "lineage_receipt_sha256": parent.lineage_receipt_sha256, "clone_mode": "pinned_train_checkpoint_path_restore",
                 "full_parameter": True, "adaptation_steps": CFS2_UPDATE_STEPS, "contract": CFS2_CONTRACT, "contract_sha256": contract_sha256()}
        if path.is_file():
            try: existing = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc: raise CFS2MatrixError("CFS-2 clone receipt is corrupt") from exc
            if existing != value: raise CFS2MatrixError("CFS-2 clone receipt disagrees with exact parent clone")
        else: atomic_write_json(path, value)
        return path

    def _plan(self, job: CFS2Job, parent: ParentCheckpoint) -> ResumePlan:
        checkpointer = self.checkpointer(job); checkpointer.adopt_existing(); record = checkpointer.resolve()
        if record is not None and record.step > parent.step + CFS2_UPDATE_STEPS:
            raise CFS2MatrixError("CFS-2 recovery checkpoint exceeds +500 target")
        return ResumePlan(job, parent, record is None, record.step if record else parent.step,
                          record.path if record else None, record.sha256 if record else None)

    def _terminalize(self, job: CFS2Job, parent: ParentCheckpoint, manifest: dict[str, t.Any], states: dict[str, dict[str, t.Any]], final, clone: pathlib.Path, *, recovered: bool) -> dict[str, t.Any]:
        if final.step - parent.step != CFS2_UPDATE_STEPS:
            entry = self.ledger.append({"job_id": job.job_id, "status": CFS2_FAILED, "reason": "not exact +500 CFS-2 updates", **self.identity(job)})
            states[job.job_id] = entry; return entry
        experiment = pathlib.Path(job.experiment_dir); config = experiment / "materialized_config.yaml"; metrics = sorted(experiment.glob("version_*/metrics.csv"))
        required = [clone, config, *metrics]
        if any(not item.is_file() or item.stat().st_size == 0 for item in required):
            raise CFS2MatrixError(f"{job.job_id} lacks durable CFS-2 trainer artifacts")
        artifacts = {str(item.relative_to(job.out_root)): sha256_file(item) for item in required}
        completion = pathlib.Path(job.out_root) / "cfs2_training_completion.json"
        value = {"schema": CFS2_COMPLETION_SCHEMA, "job_id": job.job_id, "parent_id": parent.parent_id,
                 "source_parent_id": parent.source_parent_id, "seed": job.seed, "episode": job.episode,
                 "condition": {"arm": job.arm, "overlap": job.overlap, "future_relation": job.future_relation},
                 "parent_checkpoint": {"path": parent.path, "sha256": parent.sha256, "training_steps": parent.step},
                 "branch_checkpoint": {"path": final.path, "sha256": final.sha256, "training_steps": final.step},
                 "adaptation": {"updates": CFS2_UPDATE_STEPS, "absolute_target_step": parent.step + CFS2_UPDATE_STEPS,
                                "full_parameter": True, "loss": "teacher_forced_next_token_cross_entropy", "contract": CFS2_CONTRACT, "contract_sha256": contract_sha256()},
                 "clone_provenance": {"path": clone.name, "sha256": sha256_file(clone)},
                 "lineage_receipt_sha256": parent.lineage_receipt_sha256,
                 "inputs": {"update_manifest_sha256": job.manifest_sha256, "update_bank_sha256": job.update_bank_sha256,
                            "generator_receipt_sha256": manifest["generator_receipt"]["sha256"], "construction_receipt_sha256": manifest["construction_receipt"]["sha256"],
                            "generator_manifest": manifest["generator_manifest"], "retention_probe_manifest": manifest["retention_probes"],
                            "adaptation_stream_manifest": {"path": str(pathlib.Path(job.update_bank).resolve()), "sha256": job.update_bank_sha256},
                            "global_control_manifest": manifest["global_control_manifest"], "evaluation_inputs": {key: value["sha256"] for key, value in manifest["evaluation_inputs"].items()},
                            "state_interchange_activation_patching": manifest["state_interchange_activation_patching"]},
                 "pre_post_paired_endpoints": list(CFS2_EVALUATION_INPUTS),
                 "scientific_evaluation_started": False, "training_artifacts": artifacts}
        atomic_write_json(completion, value); artifacts[completion.name] = sha256_file(completion)
        self.checkpointer(job).finalize()
        entry = self.ledger.append({"job_id": job.job_id, "status": CFS2_TRAINED, "step": final.step, "updates": CFS2_UPDATE_STEPS,
                                    "parent_checkpoint_sha256": parent.sha256, "parent_steps": parent.step,
                                    "final_checkpoint": final.path, "final_checkpoint_sha256": final.sha256,
                                    "completion_sha256": sha256_file(completion), "recovered_without_launch": recovered,
                                    "artifacts": artifacts, **self.identity(job)})
        states[job.job_id] = entry
        if self.sync is not None:
            # Publish all immutable evidence first; mutable terminal ledger state last.
            branch_root = pathlib.Path(job.out_root).resolve()
            for relative in artifacts:
                self.sync.push(branch_root / relative, f"runs/cfs2/{job.job_id}/{relative}")
            final_path = pathlib.Path(final.path); sidecar = final_path.with_name(final_path.name + ".meta.json")
            if not sidecar.is_file(): raise CFS2MatrixError("final CFS-2 checkpoint sidecar vanished")
            self.sync.push(final_path, f"runs/cfs2/{job.job_id}/{final_path.name}")
            self.sync.push(sidecar, f"runs/cfs2/{job.job_id}/{sidecar.name}")
            self.sync.push(self.ledger.path, "control/cfs2_run_ledger.json")
        return entry

    def run_job(self, job: CFS2Job, parent: ParentCheckpoint, manifest: dict[str, t.Any], states: dict[str, dict[str, t.Any]]) -> dict[str, t.Any]:
        prior = states.get(job.job_id); identity = self.identity(job)
        if prior and any(key in prior and prior[key] != value for key, value in identity.items()):
            raise CFS2MatrixError(f"{job.job_id} identity changed; refusing resume")
        if prior and prior.get("status") in TERMINAL:
            completion = pathlib.Path(job.out_root) / "cfs2_training_completion.json"; checkpoint = pathlib.Path(str(prior.get("final_checkpoint", "")))
            if completion.is_file() and checkpoint.is_file() and sha256_file(completion) == prior.get("completion_sha256") and sha256_file(checkpoint) == prior.get("final_checkpoint_sha256"):
                return prior
            self.ledger.append({"job_id": job.job_id, "status": CFS2_STALE, "reason": "terminal artifacts failed verification", **identity})
        clone = self._clone_receipt(job, parent); plan = self._plan(job, parent)
        if plan.resume_step == plan.target_step:
            final = self.checkpointer(job).resolve()
            if final is None: raise CFS2MatrixError("CFS-2 durable terminal checkpoint vanished")
            return self._terminalize(job, parent, manifest, states, final, clone, recovered=True)
        self.ledger.append({"job_id": job.job_id, "status": CFS2_RUNNING, "parent_checkpoint_sha256": parent.sha256,
                            "parent_steps": parent.step, "resume_step": plan.resume_step, **identity})
        result = self.launcher(plan); checkpoint = self.checkpointer(job); checkpoint.adopt_existing(); final = checkpoint.resolve()
        if result.returncode != 0:
            entry = self.ledger.append({"job_id": job.job_id, "status": CFS2_INTERRUPTED if final else CFS2_FAILED,
                                        "reason": result.detail, "step": final.step if final else plan.resume_step, **identity})
            states[job.job_id] = entry; return entry
        if final is None: raise CFS2MatrixError("CFS-2 launcher exited zero without durable checkpoint")
        return self._terminalize(job, parent, manifest, states, final, clone, recovered=False)

    def run(self, jobs: t.Sequence[CFS2Job], parents: t.Mapping[str, ParentCheckpoint], manifest: dict[str, t.Any]) -> dict[str, dict[str, t.Any]]:
        # Atomic pre-launch validation prevents a bad 64th input from buying a selective subset.
        for job in jobs:
            if job.parent_id not in parents: raise CFS2MatrixError(f"{job.job_id} lacks verified parent")
            self.identity(job)
        states = self.ledger.states(); by_id = {job.job_id: job for job in jobs}
        for job_id in cfs2_branch_order(manifest, list(by_id)):
            self.run_job(by_id[job_id], parents[by_id[job_id].parent_id], manifest, states)
        assert_clone_parity(states, jobs)
        return states


def assert_clone_parity(states: t.Mapping[str, t.Mapping[str, t.Any]], jobs: t.Sequence[CFS2Job]) -> None:
    by_parent: dict[str, set[str]] = {}
    for job in jobs:
        record = states.get(job.job_id)
        if record and record.get("status") in TERMINAL:
            digest = record.get("parent_checkpoint_sha256")
            if not isinstance(digest, str) or len(digest) != 64: raise CFS2MatrixError("terminal CFS-2 state lacks parent SHA")
            by_parent.setdefault(job.parent_id, set()).add(digest)
    if any(len(digests) != 1 for digests in by_parent.values()):
        raise CFS2MatrixError("CFS-2 exact-clone parity failed")


def preflight_cfs2_evaluation(root: os.PathLike[str] | str, jobs: t.Sequence[CFS2Job], ledger: CFS2Ledger,
                              manifest: t.Mapping[str, t.Any]) -> pathlib.Path:
    states = ledger.states(); missing = [job.job_id for job in jobs if states.get(job.job_id, {}).get("status") not in TERMINAL]
    if missing: raise CFS2MatrixError(f"CFS-2 evaluation is blocked; nonterminal branches: {missing}")
    branches = []
    for job in sorted(jobs, key=lambda item: item.job_id):
        state = states[job.job_id]; completion = pathlib.Path(job.out_root) / "cfs2_training_completion.json"
        if not completion.is_file() or sha256_file(completion) != state.get("completion_sha256"):
            raise CFS2MatrixError(f"{job.job_id} completion receipt missing/stale")
        receipt = json.loads(completion.read_text(encoding="utf-8")); inputs = receipt.get("inputs", {})
        if receipt.get("schema") != CFS2_COMPLETION_SCHEMA or receipt.get("job_id") != job.job_id or receipt.get("parent_id") != job.parent_id or receipt.get("scientific_evaluation_started") is not False or receipt.get("adaptation", {}).get("updates") != CFS2_UPDATE_STEPS or receipt.get("branch_checkpoint", {}).get("sha256") != state.get("final_checkpoint_sha256") or receipt.get("parent_checkpoint", {}).get("sha256") != state.get("parent_checkpoint_sha256"):
            raise CFS2MatrixError(f"{job.job_id} completion cannot authorize CFS-2 evaluation")
        if inputs.get("update_manifest_sha256") != job.manifest_sha256 or inputs.get("update_bank_sha256") != job.update_bank_sha256 or set(inputs.get("evaluation_inputs", {})) != set(CFS2_EVALUATION_INPUTS) or inputs.get("state_interchange_activation_patching", {}).get("sha256") != manifest["state_interchange_activation_patching"]["sha256"]:
            raise CFS2MatrixError(f"{job.job_id} CFS-2 endpoint/state commitment binding is incomplete")
        branches.append({"job_id": job.job_id, "parent_id": job.parent_id, "seed": job.seed, "episode": job.episode,
                         "overlap": job.overlap, "future_relation": job.future_relation,
                         "completion_sha256": state["completion_sha256"], "parent_checkpoint_sha256": state["parent_checkpoint_sha256"],
                         "branch_checkpoint_sha256": state["final_checkpoint_sha256"]})
    assert_clone_parity(states, jobs)
    output = pathlib.Path(root).resolve() / "cfs2_pre_evaluation_readiness.json"
    atomic_write_json(output, {"schema": CFS2_READINESS_SCHEMA, "status": "ALL_64_BRANCHES_TRAINED",
                               "n_branches": 64, "update_manifest_sha256": manifest["sha256"],
                               "evaluation_input_sha256s": {key: value["sha256"] for key, value in manifest["evaluation_inputs"].items()},
                               "state_interchange_activation_patching": manifest["state_interchange_activation_patching"],
                               "branches": branches, "scientific_evaluation_started": False})
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True); parser.add_argument("--manifest", required=True); parser.add_argument("--parent-ledger", required=True)
    parser.add_argument("--parent-lineage-receipt"); parser.add_argument("--ledger"); parser.add_argument("--upstream", default=str(REPO / "upstream" / "NextLat"))
    parser.add_argument("--config", default=str(REPO / CONFIG_RELATIVE)); parser.add_argument("--bucket"); parser.add_argument("--gcs-prefix", default="cfs2")
    parser.add_argument("--dry-run", action="store_true"); parser.add_argument("--print-plan", action="store_true"); parser.add_argument("--preflight-evaluation", action="store_true")
    args = parser.parse_args(argv); jobs, manifest = build_cfs2_matrix(args.root, args.manifest, config=args.config)
    ledger = CFS2Ledger(args.ledger or str(pathlib.Path(args.root) / "cfs2_run_ledger.json"))
    if args.print_plan or args.dry_run: print(json.dumps([job.to_dict() for job in jobs], indent=2, sort_keys=True)); return 0
    parents = load_parents(args.parent_ledger, lineage_receipt=args.parent_lineage_receipt)
    if args.preflight_evaluation: print(preflight_cfs2_evaluation(args.root, jobs, ledger, manifest)); return 0
    sync = DurableSync(args.bucket, args.gcs_prefix, "cfs2", logger=print) if args.bucket else None
    states = CFS2Runner(ledger, CFS2FabricLauncher(args.upstream), sync=sync).run(jobs, parents, manifest)
    return 0 if all(states.get(job.job_id, {}).get("status") in TERMINAL for job in jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
