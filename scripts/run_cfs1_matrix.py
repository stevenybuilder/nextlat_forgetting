#!/usr/bin/env python3
"""Crash-safe, isolated runner for CFS-1 adaptation branches.

This runner never invokes an evaluator.  It has one job: fork every eligible
NextLat parent checkpoint into the frozen 8 parents x 2 episodes x 4 arms
matrix, run exactly 500 CE-only full-parameter updates, and make every branch
durable under the distinct ``cfs1/`` namespace.  Evaluation is intentionally
blocked until :func:`preflight_cfs1_evaluation` verifies all 64 branches.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
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

from cfs1.adaptation import (  # noqa: E402
    CFS1_ARMS, CFS1_CONTRACT, CFS1_EPISODES, CFS1_PARENT_SEEDS,
    CFS1_UPDATE_STEPS, EVALUATION_INPUTS, cfs1_branch_order, contract_sha256,
    sha256_file, validate_nextlat_ce_only_config, validate_update_manifest,
)
from lurestar.durable_checkpoint import (  # noqa: E402
    DurableCheckpointer, DurableSync, atomic_write_json, pickle_serializer,
)


CFS1_LEDGER_SCHEMA = "nextlat_forgetting/cfs1_run_ledger/1"
CFS1_COMPLETION_SCHEMA = "nextlat_forgetting/cfs1_training_completion/1"
CFS1_CLONE_SCHEMA = "nextlat_forgetting/cfs1_clone_provenance/1"
CFS1_READINESS_SCHEMA = "nextlat_forgetting/cfs1_pre_evaluation_readiness/1"
CFS1_TRAINED = "TRAINED"
CFS1_PENDING = "PENDING"
CFS1_RUNNING = "RUNNING"
CFS1_INTERRUPTED = "INTERRUPTED"
CFS1_FAILED = "FAILED"
CFS1_STALE = "STALE"
TERMINAL = frozenset((CFS1_TRAINED, "DONE"))
PARENT_STEPS = 20_000
CONFIG_RELATIVE = pathlib.Path("configs/cfs1_nextlat_adapt.yaml")
CFS1_STIMULUS_BLOCK_FILENAME = "STIMULUS_VALIDITY_BLOCK.json"
CFS1_STIMULUS_BLOCK_SCHEMA = "nextlat_forgetting/cfs1_stimulus_validity_block/1"


class CFS1MatrixError(RuntimeError):
    """A CFS-1 identity, clone, durability, or readiness rule failed closed."""


@dataclasses.dataclass(frozen=True)
class CFS1Job:
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
        # Contains ``seed`` so pinned train.py does not add a second suffix.
        return self.job_id

    @property
    def experiment_dir(self) -> str:
        return str(pathlib.Path(self.out_root) / self.experiment_name)

    def to_dict(self) -> dict[str, t.Any]:
        return dataclasses.asdict(self) | {"experiment_name": self.experiment_name,
                                           "experiment_dir": self.experiment_dir}


@dataclasses.dataclass(frozen=True)
class ParentCheckpoint:
    parent_id: str
    seed: int
    path: str
    sha256: str
    step: int
    optimizer_present: bool
    rng_present: bool


@dataclasses.dataclass
class ResumePlan:
    spec: CFS1Job
    parent: ParentCheckpoint
    fresh: bool
    resume_step: int
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None

    @property
    def target_step(self) -> int:
        return self.parent.step + CFS1_UPDATE_STEPS


@dataclasses.dataclass
class LaunchResult:
    returncode: int
    detail: str = ""


class CFS1Ledger:
    """Append-only CFS-1 ledger, intentionally not the legacy H3/HMM ledger."""

    def __init__(self, path: os.PathLike[str] | str) -> None:
        self.path = pathlib.Path(path)

    def entries(self) -> list[dict[str, t.Any]]:
        if not self.path.is_file():
            return []
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CFS1MatrixError(f"CFS-1 ledger is unreadable: {self.path}") from exc
        if document.get("schema") != CFS1_LEDGER_SCHEMA or not isinstance(document.get("entries"), list):
            raise CFS1MatrixError("CFS-1 ledger schema/entries are invalid")
        return list(document["entries"])

    def append(self, entry: t.Mapping[str, t.Any]) -> dict[str, t.Any]:
        entries = self.entries()
        record = dict(entry)
        record["seq"] = len(entries)
        record["ts"] = time.time()
        entries.append(record)
        atomic_write_json(self.path, {"schema": CFS1_LEDGER_SCHEMA, "entries": entries})
        return record

    def states(self) -> dict[str, dict[str, t.Any]]:
        out: dict[str, dict[str, t.Any]] = {}
        for entry in self.entries():
            if isinstance(entry.get("job_id"), str):
                out[entry["job_id"]] = entry
        return out


def cfs1_job_id(seed: int, episode: int, arm: str) -> str:
    if seed not in CFS1_PARENT_SEEDS:
        raise ValueError(f"CFS-1 seed {seed} is not in the frozen parent set")
    if episode not in CFS1_EPISODES or arm not in CFS1_ARMS:
        raise ValueError("CFS-1 episode or arm is not frozen")
    return f"cfs1-nextlat-seed{seed}-episode{episode}-{arm}"


def parent_id_for_seed(seed: int) -> str:
    if seed not in CFS1_PARENT_SEEDS:
        raise ValueError(f"CFS-1 seed {seed} is not in the frozen parent set")
    # The original five are the planned Lure-Star bases.  The three added causal-study
    # parents are deliberately distinct CFS-1-only bases, so their lineage cannot be
    # mistaken for an unplanned sixth/eighth legacy replicate.
    suffix = "-base" if seed < 2000 else "-cfs1-base"
    return f"nextlat-seed{seed}{suffix}"


def _load_yaml(path: pathlib.Path) -> dict[str, t.Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise CFS1MatrixError(f"CFS-1 config is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise CFS1MatrixError("CFS-1 config must be a mapping")
    validate_nextlat_ce_only_config(value)
    cfs = value.get("cfs1")
    if not isinstance(cfs, dict) or cfs.get("adaptation_steps") != CFS1_UPDATE_STEPS:
        raise CFS1MatrixError("CFS-1 config lacks the exact 500-update contract")
    if cfs.get("unbound_bank_sentinel") not in {
            value.get("data", {}).get("stargraph_train_data_path"),
            value.get("data", {}).get("stargraph_test_data_path")}:
        raise CFS1MatrixError("CFS-1 config is not safely unbound")
    return value


def require_cfs1_stimulus_clearance(manifest_path: os.PathLike[str] | str) -> None:
    """Fail closed when an outcome-blind construct-validity block is present.

    The block is deliberately adjacent to the frozen CFS-1 manifest, rather than
    embedded in it, so the original construction remains byte-for-byte auditable.
    CFS-2 has a separate namespace and runner; this function cannot silently turn
    the confounded CFS-1 construction into the successor study.
    """
    manifest_path = pathlib.Path(manifest_path).resolve()
    block_path = manifest_path.parent / CFS1_STIMULUS_BLOCK_FILENAME
    if not block_path.exists():
        raise CFS1MatrixError(
            "CFS-1 stimulus-validity disposition is missing; retired study fails closed"
        )
    sidecar = block_path.with_name(block_path.name + ".sha256")
    try:
        document = json.loads(block_path.read_text(encoding="utf-8"))
        expected_sha = sidecar.read_text(encoding="utf-8").split()[0]
    except (OSError, json.JSONDecodeError, IndexError) as exc:
        raise CFS1MatrixError("CFS-1 stimulus-validity disposition is unreadable") from exc
    if document.get("schema") != CFS1_STIMULUS_BLOCK_SCHEMA:
        raise CFS1MatrixError("CFS-1 stimulus-validity disposition has the wrong schema")
    if expected_sha != sha256_file(block_path):
        raise CFS1MatrixError("CFS-1 stimulus-validity disposition hash is stale")
    if document.get("status") != "CLEARED_FOR_CONFIRMATORY_LAUNCH":
        reason_doc = document.get("reason")
        reason = (reason_doc.get("summary", "construct-validity block is active")
                  if isinstance(reason_doc, dict) else "construct-validity block is active")
        raise CFS1MatrixError(f"CFS-1 confirmatory launch is blocked: {reason}")


def build_cfs1_matrix(root: os.PathLike[str] | str, manifest_path: os.PathLike[str] | str,
                      *, config: os.PathLike[str] | str = REPO / CONFIG_RELATIVE) -> tuple[list[CFS1Job], dict[str, t.Any]]:
    """Build the exact 8 x 2 x 4 CFS-1 adaptation matrix from opaque inputs."""
    root = pathlib.Path(root).resolve()
    require_cfs1_stimulus_clearance(manifest_path)
    manifest = validate_update_manifest(manifest_path)
    config_path = pathlib.Path(config).resolve()
    _load_yaml(config_path)
    jobs: list[CFS1Job] = []
    for seed in CFS1_PARENT_SEEDS:
        for episode in CFS1_EPISODES:
            episode_doc = manifest["episodes"][episode]
            for arm in CFS1_ARMS:
                update = episode_doc["arms"][arm]
                overlap, future_relation = arm.split("_", 1)
                jobs.append(CFS1Job(
                    job_id=cfs1_job_id(seed, episode, arm), parent_id=parent_id_for_seed(seed),
                    seed=seed, episode=episode, arm=arm,
                    overlap=overlap, future_relation=future_relation, config=str(config_path),
                    out_root=str(root / "runs" / "cfs1" / "nextlat" / f"seed{seed}" /
                                 f"episode{episode}" / arm),
                    update_bank=str(pathlib.Path(manifest["root"]) / update["path"]),
                    update_bank_sha256=update["sha256"], manifest_path=manifest["path"],
                    manifest_sha256=manifest["sha256"],
                ))
    if len(jobs) != 64 or len({job.job_id for job in jobs}) != 64:
        raise CFS1MatrixError("CFS-1 matrix is not exactly 64 unique branches")
    roots = [pathlib.Path(job.out_root).resolve() for job in jobs]
    if any(a == b or a in b.parents or b in a.parents
           for index, a in enumerate(roots) for b in roots[index + 1:]):
        raise CFS1MatrixError("CFS-1 branch output roots overlap")
    return jobs, manifest


def _parent_state_loader(path: pathlib.Path) -> dict[str, t.Any]:
    """Load only locally generated checkpoints; tests may inject pickle equivalents."""
    try:
        import torch  # type: ignore
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        with path.open("rb") as stream:
            state = pickle.load(stream)
    if not isinstance(state, dict):
        raise CFS1MatrixError("parent checkpoint did not deserialize to a mapping")
    return state


def verify_parent_checkpoint(record: t.Mapping[str, t.Any], *, expected_seed: int,
                             loader: t.Callable[[pathlib.Path], dict[str, t.Any]] = _parent_state_loader) -> ParentCheckpoint:
    """Verify the immutable base clone source, including optimizer and RNG state."""
    if record.get("job_id") != parent_id_for_seed(expected_seed) or record.get("model") != "nextlat":
        raise CFS1MatrixError(f"seed {expected_seed} parent id/model is wrong")
    if record.get("status") not in TERMINAL:
        raise CFS1MatrixError(f"seed {expected_seed} parent is not TRAINED/DONE")
    path_value, expected_sha, step = (record.get("final_checkpoint"),
                                      record.get("final_checkpoint_sha256"), record.get("step"))
    if not isinstance(path_value, str) or not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise CFS1MatrixError(f"seed {expected_seed} parent lacks checkpoint SHA")
    if step != PARENT_STEPS:
        raise CFS1MatrixError(f"seed {expected_seed} parent must be exactly {PARENT_STEPS} steps")
    path = pathlib.Path(path_value).resolve()
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise CFS1MatrixError(f"seed {expected_seed} parent checkpoint hash is stale")
    state = loader(path)
    if state.get("training_steps") != PARENT_STEPS:
        raise CFS1MatrixError(f"seed {expected_seed} parent checkpoint does not encode 20,000 steps")
    if not isinstance(state.get("model"), dict) or not isinstance(state.get("optimizer"), dict):
        raise CFS1MatrixError("CFS-1 requires the exact parent model and optimizer state")
    if not isinstance(state.get("lurestar_rng_state_v1"), dict):
        raise CFS1MatrixError("CFS-1 requires the exact parent RNG state")
    return ParentCheckpoint(parent_id=parent_id_for_seed(expected_seed), seed=expected_seed,
                            path=str(path), sha256=expected_sha, step=PARENT_STEPS,
                            optimizer_present=True, rng_present=True)


def load_parents(parent_ledger: os.PathLike[str] | str, *, loader=_parent_state_loader) -> dict[str, ParentCheckpoint]:
    """Read latest states from either a generic append-only ledger or CFS parent ledger."""
    path = pathlib.Path(parent_ledger)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CFS1MatrixError(f"parent ledger is unreadable: {path}") from exc
    entries = document.get("entries")
    if not isinstance(entries, list):
        raise CFS1MatrixError("parent ledger lacks append-only entries")
    latest: dict[str, dict[str, t.Any]] = {}
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("job_id"), str):
            latest[entry["job_id"]] = entry
    parents = {}
    for seed in CFS1_PARENT_SEEDS:
        parent_id = parent_id_for_seed(seed)
        if parent_id not in latest:
            raise CFS1MatrixError(f"missing CFS-1 parent {parent_id}")
        parents[parent_id] = verify_parent_checkpoint(latest[parent_id], expected_seed=seed, loader=loader)
    return parents


class CFS1FabricLauncher:
    """Launch one existing NextLat branch while streaming every child line."""

    def __init__(self, upstream: os.PathLike[str] | str, *, devices: int = 1,
                 precision: str = "bf16-mixed", strategy: str = "ddp", dry_run: bool = False,
                 echo: t.Callable[[str], None] = print) -> None:
        self.upstream = pathlib.Path(upstream).resolve()
        self.devices, self.precision, self.strategy, self.dry_run, self.echo = (
            devices, precision, strategy, dry_run, echo)

    def command(self, plan: ResumePlan) -> list[str]:
        if plan.parent.step != PARENT_STEPS or plan.target_step != PARENT_STEPS + CFS1_UPDATE_STEPS:
            raise CFS1MatrixError("CFS-1 parent/target step contract changed")
        checkpoint = plan.checkpoint_path if not plan.fresh else plan.parent.path
        if not checkpoint:
            raise CFS1MatrixError("CFS-1 branch has no verified checkpoint to resume")
        return [
            "fabric", "run", "--devices", str(self.devices), "--strategy", self.strategy,
            "--precision", self.precision, "train.py", "--config", plan.spec.config,
            "--checkpoint_path", checkpoint,
            f"seed={plan.spec.seed}", f"trainer.out_dir={pathlib.Path(plan.spec.out_root).resolve()}",
            f"trainer.experiment_name={plan.spec.experiment_name}", "trainer.init_from=resume",
            f"trainer.train_batches={plan.target_step}", "trainer.compile=false",
            "trainer.log_to_wandb=false", "trainer.save_recovery_checkpoint=50",
            f"data.stargraph_train_data_path={plan.spec.update_bank}",
            f"data.stargraph_test_data_path={plan.spec.update_bank}",
        ]

    def __call__(self, plan: ResumePlan) -> LaunchResult:
        command = self.command(plan)
        self.echo("[cfs1] " + plan.spec.job_id + ": " + " ".join(command))
        if self.dry_run:
            return LaunchResult(0, "dry-run")
        proc = subprocess.Popen(command, cwd=str(self.upstream), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1,
                                env=dict(os.environ, PYTHONUNBUFFERED="1"))
        assert proc.stdout is not None
        tail: collections.deque[str] = collections.deque(maxlen=50)
        for line in proc.stdout:
            line = line.rstrip("\n")
            tail.append(line)
            self.echo(line)
        rc = proc.wait()
        return LaunchResult(rc, f"returncode={rc}; tail=" + "\\n".join(tail))


class CFS1Runner:
    def __init__(self, ledger: CFS1Ledger, launcher: t.Callable[[ResumePlan], LaunchResult], *,
                 serializer=None, sync: DurableSync | None = None,
                 echo: t.Callable[[str], None] = print) -> None:
        self.ledger, self.launcher, self.serializer, self.sync, self.echo = (
            ledger, launcher, serializer, sync, echo)

    def checkpointer(self, job: CFS1Job) -> DurableCheckpointer:
        return DurableCheckpointer(job.out_root, job.job_id, experiment_name=job.experiment_name,
                                   serializer=self.serializer, sync=self.sync, logger=self.echo)

    def identity(self, job: CFS1Job) -> dict[str, t.Any]:
        config = pathlib.Path(job.config)
        bank = pathlib.Path(job.update_bank)
        manifest = pathlib.Path(job.manifest_path)
        _load_yaml(config)
        if not bank.is_file() or sha256_file(bank) != job.update_bank_sha256:
            raise CFS1MatrixError(f"{job.job_id} update bank hash changed")
        if not manifest.is_file() or sha256_file(manifest) != job.manifest_sha256:
            raise CFS1MatrixError(f"{job.job_id} update manifest hash changed")
        return {
            "parent_id": job.parent_id, "seed": job.seed, "episode": job.episode,
            "arm": job.arm, "overlap": job.overlap,
            "future_relation": job.future_relation, "out_root": str(pathlib.Path(job.out_root).resolve()),
            "config_sha256": sha256_file(config), "update_bank_sha256": job.update_bank_sha256,
            "update_manifest_sha256": job.manifest_sha256,
        }

    def _assert_identity(self, job: CFS1Job, prior: t.Mapping[str, t.Any] | None) -> None:
        if not prior:
            return
        current = self.identity(job)
        for key, value in current.items():
            if key in prior and prior[key] != value:
                raise CFS1MatrixError(f"{job.job_id} identity changed at {key}; refusing resume")

    def _clone_receipt(self, job: CFS1Job, parent: ParentCheckpoint) -> pathlib.Path:
        path = pathlib.Path(job.out_root) / "clone_provenance.json"
        document = {
            "schema": CFS1_CLONE_SCHEMA, "job_id": job.job_id, "parent_id": parent.parent_id,
            "parent_checkpoint": {"path": parent.path, "sha256": parent.sha256,
                                  "training_steps": parent.step},
            "optimizer_state": "present_in_exact_parent_checkpoint",
            "rng_state": "present_in_exact_parent_checkpoint",
            "clone_mode": "pinned_train_checkpoint_path_restore",
            "full_parameter": True, "adaptation_steps": CFS1_UPDATE_STEPS,
            "contract": CFS1_CONTRACT, "contract_sha256": contract_sha256(),
        }
        if path.is_file():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise CFS1MatrixError(f"{job.job_id} clone receipt is corrupt") from exc
            if existing != document:
                raise CFS1MatrixError(f"{job.job_id} clone receipt disagrees with parent clone")
        else:
            atomic_write_json(path, document)
        return path

    def plan(self, job: CFS1Job, parent: ParentCheckpoint) -> ResumePlan:
        ck = self.checkpointer(job)
        ck.adopt_existing()
        record = ck.resolve()
        if record is not None and record.step > parent.step + CFS1_UPDATE_STEPS:
            raise CFS1MatrixError(f"{job.job_id} recovered checkpoint overruns exact +500 target")
        return ResumePlan(job, parent, record is None, record.step if record else parent.step,
                          record.path if record else None, record.sha256 if record else None)

    def _artifacts(self, job: CFS1Job, clone: pathlib.Path) -> dict[str, str]:
        root = pathlib.Path(job.out_root)
        experiment = pathlib.Path(job.experiment_dir)
        materialized = experiment / "materialized_config.yaml"
        metrics = sorted(experiment.glob("version_*/metrics.csv"))
        required = [clone, materialized, *metrics]
        missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
        if missing:
            raise CFS1MatrixError(f"{job.job_id} lacks durable trainer artifacts: {missing}")
        return {str(path.relative_to(root)): sha256_file(path) for path in required}

    def _completion(self, job: CFS1Job, parent: ParentCheckpoint, final, clone: pathlib.Path,
                    artifacts: dict[str, str], manifest: dict[str, t.Any]) -> pathlib.Path:
        path = pathlib.Path(job.out_root) / "cfs1_training_completion.json"
        document = {
            "schema": CFS1_COMPLETION_SCHEMA, "job_id": job.job_id, "parent_id": parent.parent_id,
            "seed": job.seed, "episode": job.episode,
            "condition": {"arm": job.arm, "overlap": job.overlap,
                          "future_relation": job.future_relation},
            "parent_checkpoint": {"path": parent.path, "sha256": parent.sha256,
                                  "training_steps": parent.step},
            "branch_checkpoint": {"path": final.path, "sha256": final.sha256,
                                  "training_steps": final.step},
            "adaptation": {"updates": CFS1_UPDATE_STEPS, "absolute_target_step": parent.step + CFS1_UPDATE_STEPS,
                           "full_parameter": True, "loss": "teacher_forced_next_token_cross_entropy",
                           "contract": CFS1_CONTRACT, "contract_sha256": contract_sha256()},
            "clone_provenance": {"path": clone.name, "sha256": sha256_file(clone)},
            "inputs": {
                "update_manifest_sha256": job.manifest_sha256,
                "update_bank_sha256": job.update_bank_sha256,
                "generator_receipt_sha256": manifest["generator_receipt"]["sha256"],
                "construction_receipt_sha256": manifest["construction_receipt"]["sha256"],
                # These names deliberately match the evaluator's opaque branch-evidence
                # contract.  They bind generated inputs, never outcomes.
                "generator_manifest": manifest["generator_manifest"],
                "retention_probe_manifest": manifest["retention_probes"],
                "adaptation_stream_manifest": {
                    "path": str(pathlib.Path(job.update_bank).resolve()),
                    "sha256": job.update_bank_sha256,
                },
                "global_control_manifest": manifest["global_control_manifest"],
                "evaluation_inputs": {name: value["sha256"]
                                      for name, value in manifest["evaluation_inputs"].items()},
            },
            "training_artifacts": artifacts,
            "scientific_evaluation_started": False,
        }
        atomic_write_json(path, document)
        return path

    def _sync_state_last(self, job: CFS1Job, artifacts: t.Iterable[pathlib.Path]) -> None:
        if self.sync is None:
            return
        # Checkpoints/receipts first.  The ledger is the only mutable terminal state and is
        # deliberately published last, so a reconnect never observes a completed branch whose
        # input or completion evidence is absent.
        for artifact in artifacts:
            try:
                relative = artifact.resolve().relative_to(pathlib.Path(job.out_root).resolve())
            except ValueError as exc:
                raise CFS1MatrixError(f"CFS-1 durable artifact escapes branch root: {artifact}") from exc
            self.sync.push(artifact, f"runs/{job.job_id}/{relative.as_posix()}")
        self.sync.push(self.ledger.path, "control/cfs1_run_ledger.json")

    def run_job(self, job: CFS1Job, parent: ParentCheckpoint, manifest: dict[str, t.Any],
                states: dict[str, dict[str, t.Any]]) -> dict[str, t.Any]:
        prior = states.get(job.job_id)
        self._assert_identity(job, prior)
        if prior and prior.get("status") in TERMINAL:
            completion = pathlib.Path(job.out_root) / "cfs1_training_completion.json"
            checkpoint = pathlib.Path(str(prior.get("final_checkpoint", "")))
            if (completion.is_file() and checkpoint.is_file() and
                    sha256_file(completion) == prior.get("completion_sha256") and
                    sha256_file(checkpoint) == prior.get("final_checkpoint_sha256")):
                return prior
            prior = self.ledger.append({"job_id": job.job_id, "status": CFS1_STALE,
                                        "reason": "terminal CFS-1 artifacts failed verification",
                                        "supersedes": prior.get("seq"), **self.identity(job)})
            states[job.job_id] = prior

        clone = self._clone_receipt(job, parent)
        plan = self.plan(job, parent)
        if plan.resume_step == plan.target_step:
            final = self.checkpointer(job).resolve()
            if final is None:
                raise CFS1MatrixError(f"{job.job_id} durable terminal checkpoint vanished")
            return self._terminalize(job, parent, manifest, states, final, clone, recovered=True)
        if plan.resume_step > plan.target_step:
            raise CFS1MatrixError(f"{job.job_id} cannot spend an over-target optimizer update")
        self.ledger.append({"job_id": job.job_id, "status": CFS1_RUNNING,
                            "parent_checkpoint_sha256": parent.sha256, "parent_steps": parent.step,
                            "resume_step": plan.resume_step, **self.identity(job)})
        result = self.launcher(plan)
        ck = self.checkpointer(job)
        ck.adopt_existing()
        final = ck.resolve()
        if result.returncode != 0:
            entry = self.ledger.append({"job_id": job.job_id,
                                        "status": CFS1_INTERRUPTED if final else CFS1_FAILED,
                                        "reason": result.detail, "step": final.step if final else plan.resume_step,
                                        "parent_checkpoint_sha256": parent.sha256, **self.identity(job)})
            states[job.job_id] = entry
            return entry
        if final is None:
            raise CFS1MatrixError(f"{job.job_id} exited 0 without a durable checkpoint")
        return self._terminalize(job, parent, manifest, states, final, clone, recovered=False)

    def _terminalize(self, job, parent, manifest, states, final, clone, *, recovered: bool):
        updates = final.step - parent.step
        if updates != CFS1_UPDATE_STEPS:
            entry = self.ledger.append({"job_id": job.job_id, "status": CFS1_FAILED,
                                        "reason": f"exact CFS-1 update failure: {updates} != {CFS1_UPDATE_STEPS}",
                                        "step": final.step, "updates": updates,
                                        "parent_checkpoint_sha256": parent.sha256, **self.identity(job)})
            states[job.job_id] = entry
            return entry
        artifacts = self._artifacts(job, clone)
        completion = self._completion(job, parent, final, clone, artifacts, manifest)
        artifacts[completion.name] = sha256_file(completion)
        self.checkpointer(job).finalize()
        entry = self.ledger.append({"job_id": job.job_id, "status": CFS1_TRAINED,
                                    "step": final.step, "updates": updates,
                                    "parent_checkpoint_sha256": parent.sha256,
                                    "parent_steps": parent.step, "final_checkpoint": final.path,
                                    "final_checkpoint_sha256": final.sha256,
                                    "completion_sha256": sha256_file(completion),
                                    "recovered_without_launch": recovered,
                                    "artifacts": artifacts, **self.identity(job)})
        states[job.job_id] = entry
        root = pathlib.Path(job.out_root)
        durable_artifacts = [root / relative for relative in artifacts]
        final_path = pathlib.Path(final.path)
        final_sidecar = final_path.with_name(final_path.name + ".meta.json")
        if not final_sidecar.is_file():
            raise CFS1MatrixError(f"{job.job_id} final checkpoint sidecar disappeared before state commit")
        self._sync_state_last(job, [*durable_artifacts, final_path, final_sidecar])
        return entry

    def run(self, jobs: t.Sequence[CFS1Job], parents: t.Mapping[str, ParentCheckpoint],
            manifest: dict[str, t.Any]) -> dict[str, dict[str, t.Any]]:
        # This is intentionally a matrix-wide preflight, not a per-branch convenience
        # check.  A tampered 64th bank must stop before the first paid branch is launched;
        # otherwise execution order could turn an input-integrity failure into a selectively
        # completed subset of the randomized intervention matrix.
        for job in jobs:
            if job.parent_id not in parents:
                raise CFS1MatrixError(f"{job.job_id} lacks verified parent")
            self.identity(job)
        states = self.ledger.states()
        ordered_ids = cfs1_branch_order(manifest, [job.job_id for job in jobs])
        by_id = {job.job_id: job for job in jobs}
        for job_id in ordered_ids:
            job = by_id[job_id]
            parent = parents.get(job.parent_id)
            if parent is None:
                raise CFS1MatrixError(f"{job.job_id} lacks verified parent")
            self.run_job(job, parent, manifest, states)
        assert_clone_parity(states, jobs)
        return states


def assert_clone_parity(states: t.Mapping[str, t.Mapping[str, t.Any]], jobs: t.Sequence[CFS1Job]) -> None:
    """Every completed branch for a parent must bind the same exact parent checkpoint."""
    grouped: dict[str, set[str]] = {}
    for job in jobs:
        state = states.get(job.job_id)
        if not state or state.get("status") not in TERMINAL:
            continue
        digest = state.get("parent_checkpoint_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise CFS1MatrixError(f"{job.job_id} terminal state lacks parent SHA")
        grouped.setdefault(job.parent_id, set()).add(digest)
    for parent_id, digests in grouped.items():
        if len(digests) != 1:
            raise CFS1MatrixError(f"CFS-1 clone parity failed for {parent_id}: {sorted(digests)}")


def preflight_cfs1_evaluation(root: os.PathLike[str] | str, jobs: t.Sequence[CFS1Job],
                               ledger: CFS1Ledger, manifest: t.Mapping[str, t.Any]) -> pathlib.Path:
    """Atomically authorize evaluation only after every branch has durable paired evidence."""
    states = ledger.states()
    missing = [job.job_id for job in jobs if states.get(job.job_id, {}).get("status") not in TERMINAL]
    if missing:
        raise CFS1MatrixError(f"CFS-1 evaluation is blocked; nonterminal branches: {missing}")
    bindings: list[dict[str, t.Any]] = []
    for job in sorted(jobs, key=lambda item: item.job_id):
        state = states[job.job_id]
        completion_path = pathlib.Path(job.out_root) / "cfs1_training_completion.json"
        if not completion_path.is_file() or sha256_file(completion_path) != state.get("completion_sha256"):
            raise CFS1MatrixError(f"{job.job_id} completion receipt is missing/stale")
        try:
            receipt = json.loads(completion_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CFS1MatrixError(f"{job.job_id} completion receipt is invalid") from exc
        expected = {
            "job_id": job.job_id, "parent_id": job.parent_id,
            "seed": job.seed,
        }
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise CFS1MatrixError(f"{job.job_id} completion receipt identity is wrong")
        if (receipt.get("adaptation", {}).get("updates") != CFS1_UPDATE_STEPS or
                receipt.get("branch_checkpoint", {}).get("sha256") != state.get("final_checkpoint_sha256") or
                receipt.get("parent_checkpoint", {}).get("sha256") != state.get("parent_checkpoint_sha256") or
                receipt.get("scientific_evaluation_started") is not False):
            raise CFS1MatrixError(f"{job.job_id} completion receipt cannot authorize evaluation")
        inputs = receipt.get("inputs", {})
        if (inputs.get("update_manifest_sha256") != job.manifest_sha256 or
                inputs.get("update_bank_sha256") != job.update_bank_sha256 or
                set(inputs.get("evaluation_inputs", {})) != set(EVALUATION_INPUTS) or
                inputs.get("generator_manifest", {}).get("sha256") !=
                    manifest["generator_manifest"]["sha256"] or
                inputs.get("retention_probe_manifest", {}).get("sha256") !=
                    manifest["retention_probes"]["sha256"] or
                inputs.get("adaptation_stream_manifest", {}).get("sha256") != job.update_bank_sha256 or
                inputs.get("global_control_manifest", {}).get("sha256") !=
                    manifest["global_control_manifest"]["sha256"]):
            raise CFS1MatrixError(f"{job.job_id} evaluator inputs are not completely bound")
        bindings.append({"job_id": job.job_id, "completion_sha256": state["completion_sha256"],
                         "parent_checkpoint_sha256": state["parent_checkpoint_sha256"],
                         "branch_checkpoint_sha256": state["final_checkpoint_sha256"]})
    assert_clone_parity(states, jobs)
    readiness = pathlib.Path(root).resolve() / "cfs1_pre_evaluation_readiness.json"
    atomic_write_json(readiness, {
        "schema": CFS1_READINESS_SCHEMA, "status": "ALL_BRANCHES_TRAINED",
        "n_branches": len(jobs), "update_manifest_sha256": manifest["sha256"],
        "evaluation_input_sha256s": {key: value["sha256"]
                                      for key, value in manifest["evaluation_inputs"].items()},
        "branches": bindings, "scientific_evaluation_started": False,
    })
    return readiness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--parent-ledger", required=True)
    parser.add_argument("--ledger", default=None)
    parser.add_argument("--upstream", default=str(REPO / "upstream" / "NextLat"))
    parser.add_argument("--config", default=str(REPO / CONFIG_RELATIVE))
    parser.add_argument("--bucket")
    parser.add_argument("--gcs-prefix", default="cfs1")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--preflight-evaluation", action="store_true")
    args = parser.parse_args(argv)
    jobs, manifest = build_cfs1_matrix(args.root, args.manifest, config=args.config)
    ledger = CFS1Ledger(args.ledger or str(pathlib.Path(args.root) / "cfs1_run_ledger.json"))
    if args.print_plan or args.dry_run:
        print(json.dumps([job.to_dict() for job in jobs], indent=2, sort_keys=True))
        return 0
    parents = load_parents(args.parent_ledger)
    if args.preflight_evaluation:
        print(preflight_cfs1_evaluation(args.root, jobs, ledger, manifest))
        return 0
    sync = DurableSync(args.bucket, args.gcs_prefix, "cfs1", logger=print) if args.bucket else None
    runner = CFS1Runner(ledger, CFS1FabricLauncher(args.upstream), sync=sync)
    states = runner.run(jobs, parents, manifest)
    return 0 if all(states.get(job.job_id, {}).get("status") in TERMINAL for job in jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
