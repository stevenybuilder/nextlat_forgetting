#!/usr/bin/env python
"""Idempotent runner for the confirmatory job matrix (spec section 9, "Idempotent runner").

Running this twice must be a no-op the second time, and running it after a Colab disconnect must
pick each job up from its newest *verified* checkpoint. Concretely it:

  1. reads `results/run_ledger.json`;
  2. skips `DONE` jobs only when every recorded artifact still hashes to the recorded value;
  3. resumes incomplete jobs from the newest valid checkpoint, rolling back one if the newest
     is corrupt (`DurableCheckpointer.resolve`);
  4. preserves config, seed, manifests and output root across the resume, and refuses to launch
     if any of them changed under it;
  5. writes atomic `metrics/step_{step}.json` keyed by `(run_id, step)`;
  6. marks `DONE` only after the final evaluation artifacts exist and verify.

The single most dangerous property it enforces is the output-root separation. Upstream's resume
pointers `recovery_ckpt` and `latest_ckpt` live at `trainer.out_dir`, one directory *above* the
experiment directory (`core_train.py:944-948`, `core_train.py:970-974`), and `init_from: resume`
reads them from there (`core_train.py:139-151`). The shipped configs give every algorithm and
every sweep seed the same `out_dir: output/stargraph` (`gpt_stargraph_5_5.yaml:14`), so whichever
job wrote last owns the pointer. If an H3 `far` branch ever resumed from the `near` branch's
pointer, both branches would silently share a parent and the near-minus-far contrast -- the whole
of H3 -- would be measuring nothing. `validate_matrix` makes that unrepresentable.

Usage::

    python scripts/run_matrix.py --root gs-mirror/lurestar --print-plan
    python scripts/run_matrix.py --root /content/lurestar --phase base
    python scripts/run_matrix.py --root /content/lurestar --retry-sync
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import subprocess
import sys
import time
import typing as t

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from lurestar.durable_checkpoint import (  # noqa: E402
    DurableCheckpointer,
    DurableSync,
    atomic_write_json,
    sha256_file,
)

MODELS = ("gpt", "nextlat")
SEEDS = (1234, 1235, 1236)          # PROGRAM.md: frozen, three preregistered confirmatory seeds
CONDITIONS = ("near", "far")

PENDING, RUNNING, INTERRUPTED, FAILED, DONE, STALE = (
    "PENDING", "RUNNING", "INTERRUPTED", "FAILED", "DONE", "STALE",
)

DEFAULT_FINAL_ARTIFACTS = ("final_summary.json",)


# --------------------------------------------------------------------------------------
# job identity
# --------------------------------------------------------------------------------------

def job_id(model: str, seed: int, phase: str, condition: str | None = None) -> str:
    """Deterministic ids: `nextlat-s1234-base`, `gpt-s1235-adapt-near`.

    Deterministic because the ledger, the output root and the GCS prefix are all keyed by it;
    a job id that depended on iteration order or a timestamp would make a resume a new job.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}")
    parts = [model, f"s{int(seed)}", phase]
    if condition is not None:
        parts.append(condition)
    return "-".join(parts)


@dataclasses.dataclass(frozen=True)
class JobSpec:
    job_id: str
    model: str
    seed: int
    phase: str                      # "base" | "adapt"
    condition: str | None           # None | "near" | "far"
    config: str
    out_root: str
    manifests: tuple[str, ...] = ()
    parent_job_id: str | None = None
    train_batches: int = 20000
    final_artifacts: tuple[str, ...] = DEFAULT_FINAL_ARTIFACTS
    overrides: tuple[str, ...] = ()

    @property
    def experiment_name(self) -> str:
        return self.job_id

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def build_matrix(
    root: os.PathLike | str,
    *,
    models: t.Sequence[str] = MODELS,
    seeds: t.Sequence[int] = SEEDS,
    config_for: t.Callable[[str, str, str | None], str] | None = None,
    manifests: t.Mapping[str, t.Sequence[str]] | None = None,
    base_steps: int = 20000,
    adapt_steps: int = 500,
    final_artifacts: t.Sequence[str] = DEFAULT_FINAL_ARTIFACTS,
) -> list[JobSpec]:
    """One base job per (model, seed); one `near` and one `far` adaptation job hanging off it.

    Directory layout is the spec's: `runs/{model}/{seed}/{phase}/{condition}/`.
    """
    root = pathlib.Path(root)
    if config_for is None:
        def config_for(model, phase, condition):  # noqa: ARG001
            suffix = "base" if phase == "base" else "adapt"
            return str(_REPO / "configs" / f"{model}_lurestar_{suffix}.yaml")

    manifests = manifests or {}
    jobs: list[JobSpec] = []
    for model in models:
        for seed in seeds:
            base_id = job_id(model, seed, "base")
            jobs.append(JobSpec(
                job_id=base_id, model=model, seed=seed, phase="base", condition=None,
                config=config_for(model, "base", None),
                out_root=str(root / "runs" / model / str(seed) / "base" / "_"),
                manifests=tuple(manifests.get("base", ())),
                train_batches=base_steps,
                final_artifacts=tuple(final_artifacts),
            ))
            for cond in CONDITIONS:
                jobs.append(JobSpec(
                    job_id=job_id(model, seed, "adapt", cond),
                    model=model, seed=seed, phase="adapt", condition=cond,
                    config=config_for(model, "adapt", cond),
                    out_root=str(root / "runs" / model / str(seed) / "adapt" / cond),
                    manifests=tuple(manifests.get(cond, ())),
                    parent_job_id=base_id,
                    train_batches=adapt_steps,
                    final_artifacts=tuple(final_artifacts),
                ))
    validate_matrix(jobs)
    return jobs


def validate_matrix(jobs: t.Sequence[JobSpec]) -> None:
    """No two jobs may share, or nest inside, an output root; ids must be unique.

    Nesting matters as much as equality: an out_root that contains another job's out_root would
    put one job's `recovery_ckpt` on the resume search path of the other.
    """
    seen_ids: set[str] = set()
    roots: list[tuple[str, pathlib.Path]] = []
    for j in jobs:
        if j.job_id in seen_ids:
            raise ValueError(f"duplicate job id {j.job_id}")
        seen_ids.add(j.job_id)
        roots.append((j.job_id, pathlib.Path(j.out_root).resolve()))
    for i, (id_a, a) in enumerate(roots):
        for id_b, b in roots[i + 1:]:
            if a == b:
                raise ValueError(
                    f"{id_a} and {id_b} share out_root {a}; upstream's resume pointer lives at "
                    "out_dir level (core_train.py:944-948) and would cross branches"
                )
            if a in b.parents or b in a.parents:
                raise ValueError(f"{id_a} and {id_b} have nested out_roots: {a} vs {b}")


# --------------------------------------------------------------------------------------
# ledger
# --------------------------------------------------------------------------------------

class Ledger:
    """Append-only run ledger (PROGRAM.md invariant 1).

    Entries are never rewritten. A wrong entry is corrected by appending a superseding entry
    carrying a `reason`. The file is rewritten atomically as a whole each time, but the list of
    entries only ever grows, so the history of every job is recoverable.
    """

    def __init__(self, path: os.PathLike | str) -> None:
        self.path = pathlib.Path(path)

    def entries(self) -> list[dict]:
        if not self.path.is_file():
            return []
        doc = json.loads(self.path.read_text())
        return doc.get("entries", [])

    def append(self, entry: dict) -> dict:
        entries = self.entries()
        record = dict(entry)
        record.setdefault("ts", time.time())
        record["seq"] = len(entries)
        entries.append(record)
        atomic_write_json(self.path, {"schema": 1, "entries": entries})
        return record

    def state_of(self, job: str) -> dict | None:
        latest = None
        for e in self.entries():
            if e.get("job_id") == job:
                latest = e
        return latest

    def states(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for e in self.entries():
            if "job_id" in e:
                out[e["job_id"]] = e
        return out


# --------------------------------------------------------------------------------------
# artifacts and metrics
# --------------------------------------------------------------------------------------

def hash_artifacts(out_root: os.PathLike | str, rels: t.Sequence[str]) -> dict[str, str]:
    root = pathlib.Path(out_root)
    missing = [r for r in rels if not (root / r).is_file()]
    if missing:
        raise FileNotFoundError(f"missing final artifacts under {root}: {missing}")
    return {r: sha256_file(root / r) for r in rels}


def verify_artifacts(out_root: os.PathLike | str, recorded: t.Mapping[str, str]) -> tuple[bool, str]:
    root = pathlib.Path(out_root)
    for rel, want in recorded.items():
        p = root / rel
        if not p.is_file():
            return False, f"artifact {rel} is missing"
        got = sha256_file(p)
        if got != want:
            return False, f"artifact {rel} hash {got[:12]} != recorded {want[:12]}"
    return True, "ok"


def write_step_metrics(out_root: os.PathLike | str, run_id: str, step: int, payload: dict) -> pathlib.Path:
    """Atomic `metrics/step_{step}.json` keyed by `(run_id, step)`.

    Keyed means keyed: rewriting the same step from a *different* run id is a bug (two jobs
    sharing an output root), so it raises instead of silently overwriting.
    """
    path = pathlib.Path(out_root) / "metrics" / f"step_{int(step)}.json"
    if path.is_file():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = {}
        if existing.get("run_id") not in (None, run_id):
            raise ValueError(
                f"{path} already belongs to run {existing['run_id']!r}, refusing to overwrite "
                f"with {run_id!r} -- two jobs are sharing an output root"
            )
    body = dict(payload)
    body["run_id"] = run_id
    body["step"] = int(step)
    atomic_write_json(path, body)
    return path


# --------------------------------------------------------------------------------------
# resume planning
# --------------------------------------------------------------------------------------

@dataclasses.dataclass
class ResumePlan:
    spec: JobSpec
    fresh: bool
    resume_step: int = 0
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    rolled_back_from: str | None = None
    parent_checkpoint: str | None = None
    parent_checkpoint_sha256: str | None = None

    @property
    def init_from(self) -> str:
        return "scratch" if self.fresh else "resume"

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["spec"] = self.spec.to_dict()
        d["init_from"] = self.init_from
        return d


@dataclasses.dataclass
class LaunchResult:
    returncode: int
    final_step: int | None = None
    detail: str = ""


class FabricLauncher:
    """Builds and runs the real single-GPU command. Child output is relayed line by line.

    Two transport bugs already cost a smoke test each (docs/RUNLOG.md): a child's stdout does
    not reach the `colab exec` stream unless it is relayed in-process, and piping the command
    into `tail` returns *tail's* exit status, so a crashed run reports RC=0. Neither is repeated
    here: output is relayed through a pipe we read ourselves, and the returncode is the child's.
    """

    def __init__(self, upstream_dir: os.PathLike | str, *, devices: int = 1,
                 precision: str = "bf16-mixed", strategy: str = "ddp",
                 python: str = sys.executable, dry_run: bool = False,
                 echo: t.Callable[[str], None] = print) -> None:
        self.upstream_dir = pathlib.Path(upstream_dir)
        self.devices = devices
        self.precision = precision
        self.strategy = strategy
        self.python = python
        self.dry_run = dry_run
        self.echo = echo

    def command(self, plan: ResumePlan) -> list[str]:
        spec = plan.spec
        cmd = [
            "fabric", "run",
            "--devices", str(self.devices),
            "--strategy", self.strategy,
            "--precision", self.precision,
            "train.py", "--config", spec.config,
        ]
        if plan.parent_checkpoint and plan.fresh:
            # train.py:262-264; --checkpoint_path takes precedence over init_from
            # (core_train.py:130) and restores weights+optimizer+step, which is how an H3
            # branch starts from the frozen base parent.
            cmd += ["--checkpoint_path", plan.parent_checkpoint]
        # Everything else is an OmegaConf dotlist override (train.py:265, train.py:349).
        cmd += [
            f"seed={spec.seed}",
            f"trainer.out_dir={pathlib.Path(spec.out_root).resolve()}",
            f"trainer.experiment_name={spec.experiment_name}",
            f"trainer.init_from={plan.init_from}",
            f"trainer.train_batches={spec.train_batches}",
            "trainer.compile=false",          # spec section 8; README.md:117-122
            "trainer.log_to_wandb=false",
            "trainer.save_recovery_checkpoint=250",
        ]
        cmd += list(spec.overrides)
        return cmd

    def __call__(self, plan: ResumePlan) -> LaunchResult:
        cmd = self.command(plan)
        self.echo(f"[run_matrix] {plan.spec.job_id}: " + " ".join(cmd))
        if self.dry_run:
            return LaunchResult(0, None, "dry-run")
        proc = subprocess.Popen(
            cmd, cwd=str(self.upstream_dir), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            env=dict(os.environ, PYTHONUNBUFFERED="1"),
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            self.echo(line.rstrip("\n"))
        rc = proc.wait()
        return LaunchResult(rc, None, f"returncode={rc}")


# --------------------------------------------------------------------------------------
# the runner
# --------------------------------------------------------------------------------------

class MatrixRunner:
    def __init__(self, ledger: Ledger, launcher: t.Callable[[ResumePlan], LaunchResult], *,
                 serializer=None, sync: DurableSync | None = None,
                 echo: t.Callable[[str], None] = print) -> None:
        self.ledger = ledger
        self.launcher = launcher
        self.serializer = serializer
        self.sync = sync
        self.echo = echo

    def checkpointer(self, spec: JobSpec) -> DurableCheckpointer:
        return DurableCheckpointer(
            spec.out_root, spec.job_id, experiment_name=spec.experiment_name,
            serializer=self.serializer, sync=self.sync, logger=self.echo,
        )

    # ---- identity guard --------------------------------------------------------------
    def _identity(self, spec: JobSpec) -> dict:
        """Config, seed, manifests and output root, hashed. Spec section 9.3 item 4."""
        cfg = pathlib.Path(spec.config)
        return {
            "seed": spec.seed,
            "out_root": str(pathlib.Path(spec.out_root).resolve()),
            "config": str(cfg),
            "config_sha256": sha256_file(cfg) if cfg.is_file() else None,
            "manifest_sha256": {
                m: (sha256_file(m) if pathlib.Path(m).is_file() else None)
                for m in spec.manifests
            },
        }

    def _check_identity(self, spec: JobSpec, prior: dict | None) -> None:
        if not prior:
            return
        now = self._identity(spec)
        for key in ("seed", "out_root", "config_sha256", "manifest_sha256"):
            if key in prior and prior[key] is not None and prior[key] != now[key]:
                raise RuntimeError(
                    f"{spec.job_id}: {key} changed since the last ledger entry "
                    f"({prior[key]!r} -> {now[key]!r}). A resume must preserve config, seed, "
                    "manifest and output root; refusing to continue this job."
                )

    # ---- planning ---------------------------------------------------------------------
    def plan(self, spec: JobSpec, states: t.Mapping[str, dict]) -> ResumePlan:
        ck = self.checkpointer(spec)
        before = {r.path for r in ck.read_index()}
        rec = ck.resolve()
        after = {r.path for r in ck.read_index()}
        rolled_back = sorted(before - after)

        parent_ckpt = parent_sha = None
        if spec.parent_job_id:
            parent = states.get(spec.parent_job_id)
            if not parent or parent.get("status") != DONE:
                raise RuntimeError(
                    f"{spec.job_id} needs parent {spec.parent_job_id} to be DONE first"
                )
            parent_ckpt = parent.get("final_checkpoint")
            parent_sha = parent.get("final_checkpoint_sha256")
            if not parent_ckpt or not parent_sha:
                raise RuntimeError(f"parent {spec.parent_job_id} recorded no final checkpoint")
            got = sha256_file(parent_ckpt) if pathlib.Path(parent_ckpt).is_file() else None
            if got != parent_sha:
                raise RuntimeError(
                    f"parent checkpoint {parent_ckpt} hash {str(got)[:12]} != recorded "
                    f"{parent_sha[:12]}; the H3 branches would not share a parent"
                )

        return ResumePlan(
            spec=spec,
            fresh=rec is None,
            resume_step=rec.step if rec else 0,
            checkpoint_path=rec.path if rec else None,
            checkpoint_sha256=rec.sha256 if rec else None,
            rolled_back_from=rolled_back[0] if rolled_back else None,
            parent_checkpoint=parent_ckpt,
            parent_checkpoint_sha256=parent_sha,
        )

    # ---- run ---------------------------------------------------------------------------
    def run_job(self, spec: JobSpec, states: dict[str, dict]) -> dict:
        prior = states.get(spec.job_id)

        if prior and prior.get("status") == DONE:
            ok, reason = verify_artifacts(spec.out_root, prior.get("artifacts", {}))
            if ok and prior.get("final_checkpoint_sha256"):
                p = pathlib.Path(prior["final_checkpoint"])
                ok = p.is_file() and sha256_file(p) == prior["final_checkpoint_sha256"]
                reason = "ok" if ok else "final checkpoint hash mismatch"
            if ok:
                self.echo(f"[run_matrix] {spec.job_id}: DONE, hashes verify, skipping")
                return prior
            # Never silently rerun a DONE job: append a superseding entry saying why.
            entry = self.ledger.append({
                "job_id": spec.job_id, "status": STALE,
                "reason": f"DONE entry failed verification: {reason}",
                "supersedes": prior.get("seq"), **self._identity(spec),
            })
            states[spec.job_id] = entry
            prior = entry

        self._check_identity(spec, prior)
        plan = self.plan(spec, states)
        self.ledger.append({
            "job_id": spec.job_id, "status": RUNNING, "step": plan.resume_step,
            "resumed_from": plan.checkpoint_path,
            "resumed_from_sha256": plan.checkpoint_sha256,
            "rolled_back_from": plan.rolled_back_from,
            "parent_job_id": spec.parent_job_id,
            "parent_checkpoint_sha256": plan.parent_checkpoint_sha256,
            **self._identity(spec),
        })

        result = self.launcher(plan)

        ck = self.checkpointer(spec)
        final = ck.resolve()
        if result.returncode != 0:
            entry = self.ledger.append({
                "job_id": spec.job_id,
                "status": INTERRUPTED if final is not None else FAILED,
                "step": final.step if final else plan.resume_step,
                "reason": result.detail or f"returncode={result.returncode}",
                "parent_job_id": spec.parent_job_id,
                "parent_checkpoint_sha256": plan.parent_checkpoint_sha256,
                **self._identity(spec),
            })
            states[spec.job_id] = entry
            return entry

        try:
            artifacts = hash_artifacts(spec.out_root, spec.final_artifacts)
        except FileNotFoundError as exc:
            entry = self.ledger.append({
                "job_id": spec.job_id, "status": FAILED, "reason": str(exc),
                "step": final.step if final else plan.resume_step,
                "parent_job_id": spec.parent_job_id,
                "parent_checkpoint_sha256": plan.parent_checkpoint_sha256,
                **self._identity(spec),
            })
            states[spec.job_id] = entry
            return entry

        if final is None:
            entry = self.ledger.append({
                "job_id": spec.job_id, "status": FAILED,
                "reason": "job exited 0 but left no verified checkpoint",
                **self._identity(spec),
            })
            states[spec.job_id] = entry
            return entry

        ck.finalize()   # clear the stale recovery pointer; core_train.py:145-151
        entry = self.ledger.append({
            "job_id": spec.job_id, "status": DONE, "step": final.step,
            "final_checkpoint": final.path, "final_checkpoint_sha256": final.sha256,
            "artifacts": artifacts,
            "parent_job_id": spec.parent_job_id,
            "parent_checkpoint_sha256": plan.parent_checkpoint_sha256,
            **self._identity(spec),
        })
        states[spec.job_id] = entry
        self.echo(f"[run_matrix] {spec.job_id}: DONE at step {final.step}")
        return entry

    def run(self, jobs: t.Sequence[JobSpec]) -> dict[str, dict]:
        validate_matrix(jobs)
        states = self.ledger.states()
        for spec in jobs:
            self.run_job(spec, states)
        assert_branch_parity(states, jobs)
        return states


def assert_branch_parity(states: t.Mapping[str, dict], jobs: t.Sequence[JobSpec]) -> None:
    """Near and far must record the same `parent_checkpoint_sha256` (spec section 9.3).

    They are the two arms of the H3 contrast. If they hang off different parents, the
    near-minus-far erosion difference confounds the branch with the starting point.
    """
    by_parent: dict[str, dict[str, str | None]] = {}
    for spec in jobs:
        if spec.phase != "adapt" or spec.parent_job_id is None:
            continue
        entry = states.get(spec.job_id)
        if not entry or entry.get("status") != DONE:
            continue
        by_parent.setdefault(spec.parent_job_id, {})[spec.condition or "?"] = entry.get(
            "parent_checkpoint_sha256"
        )
    for parent, arms in by_parent.items():
        if len(arms) < 2:
            continue
        shas = set(arms.values())
        if len(shas) != 1 or None in shas:
            raise RuntimeError(
                f"H3 branches off {parent} do not share a parent checkpoint: {arms}"
            )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="durable run root, e.g. /content/lurestar")
    ap.add_argument("--ledger", default=str(_REPO / "results" / "run_ledger.json"))
    ap.add_argument("--upstream", default=str(_REPO / "upstream" / "NextLat"))
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--phase", choices=["base", "adapt", "all"], default="all")
    ap.add_argument("--only", nargs="*", default=None, help="explicit job ids")
    ap.add_argument("--base-steps", type=int, default=20000)
    ap.add_argument("--adapt-steps", type=int, default=500)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--precision", default="bf16-mixed")
    ap.add_argument("--strategy", default="ddp",
                    help="ddp even on one device: it gives a DistributedSampler whose order is "
                         "reproducible across resumes (docs/UPSTREAM_REPORT.md section 3.5 item 5)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--print-plan", action="store_true")
    ap.add_argument("--bucket", default=os.environ.get("LURESTAR_BUCKET"))
    ap.add_argument("--gcs-prefix", default="lurestar")
    ap.add_argument("--retry-sync", action="store_true",
                    help="drain the NEEDS_SYNC queue left by a failed GCS push and exit")
    a = ap.parse_args(argv)

    if a.retry_sync:
        if not a.bucket:
            print("--retry-sync needs --bucket or $LURESTAR_BUCKET", file=sys.stderr)
            return 2
        failed = 0
        for spec in build_matrix(a.root, models=a.models, seeds=a.seeds):
            sync = DurableSync(a.bucket, a.gcs_prefix, spec.job_id, logger=print)
            for res in sync.retry_pending():
                print(("  ok   " if res.ok else "  FAIL ") + res.remote)
                failed += 0 if res.ok else 1
        return 1 if failed else 0

    jobs = build_matrix(
        a.root, models=a.models, seeds=a.seeds,
        base_steps=a.base_steps, adapt_steps=a.adapt_steps,
    )
    if a.phase != "all":
        jobs = [j for j in jobs if j.phase == a.phase]
    if a.only:
        wanted = set(a.only)
        jobs = [j for j in jobs if j.job_id in wanted]

    if a.print_plan:
        print(json.dumps([j.to_dict() for j in jobs], indent=2))
        return 0

    sync = DurableSync(a.bucket, a.gcs_prefix, "matrix", logger=print) if a.bucket else None
    launcher = FabricLauncher(
        a.upstream, devices=a.devices, precision=a.precision,
        strategy=a.strategy, dry_run=a.dry_run,
    )
    runner = MatrixRunner(Ledger(a.ledger), launcher, sync=sync)
    states = runner.run(jobs)
    not_done = [j.job_id for j in jobs if states.get(j.job_id, {}).get("status") != DONE]
    if not_done:
        print(f"[run_matrix] not DONE: {not_done}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
