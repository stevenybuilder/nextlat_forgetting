"""The idempotent-runner contract (spec section 9, "Idempotent runner").

The runner is exercised against a fake launcher that writes real checkpoints through the real
`DurableCheckpointer`, so the resume, rollback and verification paths are the production ones;
only the `fabric run` subprocess is stubbed. `test_fabric_command_*` pins the real command.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from lurestar.durable_checkpoint import DurableCheckpointer, pickle_serializer, sha256_file
from run_matrix import (
    DONE,
    FAILED,
    INTERRUPTED,
    STALE,
    FabricLauncher,
    JobSpec,
    LaunchResult,
    Ledger,
    MatrixRunner,
    ResumePlan,
    assert_branch_parity,
    build_matrix,
    job_id,
    validate_matrix,
    write_step_metrics,
)

SER = pickle_serializer()


# --------------------------------------------------------------------------------------
# a fake launcher that writes real checkpoints
# --------------------------------------------------------------------------------------

class FakeLauncher:
    """Trains `steps_per_call` further steps, checkpointing every 10, then stops or fails."""

    def __init__(self, total: int = 50, stop_at: dict[str, int] | None = None,
                 fail: set[str] | None = None):
        self.total = total
        self.stop_at = stop_at or {}
        self.fail = fail or set()
        self.calls: list[ResumePlan] = []

    def __call__(self, plan: ResumePlan) -> LaunchResult:
        self.calls.append(plan)
        spec = plan.spec
        ck = DurableCheckpointer(spec.out_root, spec.job_id,
                                 experiment_name=spec.experiment_name, serializer=SER)
        stop = self.stop_at.get(spec.job_id, self.total)
        step = plan.resume_step
        while step < min(stop, self.total):
            step += 1
            if step % 10 == 0:
                ck.save({"step": step, "job": spec.job_id,
                         "parent": plan.parent_checkpoint_sha256}, step)
                write_step_metrics(spec.out_root, spec.job_id, step,
                                   {"loss": 1.0 / step})
        if spec.job_id in self.fail:
            return LaunchResult(1, step, "simulated crash")
        if step < self.total:
            return LaunchResult(137, step, "simulated runtime disconnect")
        ck.save({"step": step, "job": spec.job_id,
                 "parent": plan.parent_checkpoint_sha256}, step, kind="final")
        (pathlib.Path(spec.out_root) / "final_summary.json").write_text(
            json.dumps({"job_id": spec.job_id, "step": step}) + "\n")
        return LaunchResult(0, step, "ok")


@pytest.fixture()
def matrix(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("trainer:\n  train_batches: 50\n")
    jobs = build_matrix(tmp_path / "root", models=("gpt",), seeds=(1234,),
                        config_for=lambda m, p, c: str(cfg),
                        base_steps=50, adapt_steps=50)
    return jobs, cfg


def make_runner(tmp_path, launcher) -> MatrixRunner:
    return MatrixRunner(Ledger(tmp_path / "run_ledger.json"), launcher,
                        serializer=SER, echo=lambda m: None)


# --------------------------------------------------------------------------------------
# identity and layout
# --------------------------------------------------------------------------------------

def test_job_ids_are_deterministic_and_match_the_spec():
    assert job_id("nextlat", 1234, "base") == "nextlat-s1234-base"
    assert job_id("gpt", 1235, "adapt", "near") == "gpt-s1235-adapt-near"
    assert job_id("gpt", 1235, "adapt", "far") == "gpt-s1235-adapt-far"
    with pytest.raises(ValueError):
        job_id("llama", 1234, "base")


def test_every_branch_gets_its_own_output_root(tmp_path):
    jobs = build_matrix(tmp_path / "r")
    assert len(jobs) == 2 * 3 * 3          # 2 models x 3 seeds x (base, near, far)
    roots = [pathlib.Path(j.out_root).resolve() for j in jobs]
    assert len(set(roots)) == len(roots)
    near = next(j for j in jobs if j.job_id == "nextlat-s1234-adapt-near")
    far = next(j for j in jobs if j.job_id == "nextlat-s1234-adapt-far")
    base = next(j for j in jobs if j.job_id == "nextlat-s1234-base")
    assert near.out_root != far.out_root != base.out_root
    assert near.parent_job_id == far.parent_job_id == base.job_id


def test_shared_or_nested_output_roots_are_rejected(tmp_path):
    """A crossed resume pointer silently destroys H3, so make it unrepresentable."""
    a = JobSpec("a", "gpt", 1234, "adapt", "near", "c.yaml", str(tmp_path / "out"))
    b = JobSpec("b", "gpt", 1234, "adapt", "far", "c.yaml", str(tmp_path / "out"))
    with pytest.raises(ValueError, match="share out_root"):
        validate_matrix([a, b])
    c = JobSpec("c", "gpt", 1234, "adapt", "far", "c.yaml", str(tmp_path / "out" / "far"))
    with pytest.raises(ValueError, match="nested out_roots"):
        validate_matrix([a, c])
    with pytest.raises(ValueError, match="duplicate job id"):
        validate_matrix([a, a])


# --------------------------------------------------------------------------------------
# idempotency
# --------------------------------------------------------------------------------------

def test_second_run_is_a_no_op_when_hashes_verify(tmp_path, matrix):
    jobs, _ = matrix
    launcher = FakeLauncher()
    runner = make_runner(tmp_path, launcher)
    states = runner.run(jobs)
    assert {j.job_id: states[j.job_id]["status"] for j in jobs} == {j.job_id: DONE for j in jobs}
    first_calls = len(launcher.calls)
    assert first_calls == len(jobs)

    states2 = make_runner(tmp_path, launcher).run(jobs)
    assert len(launcher.calls) == first_calls, "a verified DONE job was relaunched"
    assert all(states2[j.job_id]["status"] == DONE for j in jobs)


def test_tampered_final_artifact_supersedes_done_and_reruns(tmp_path, matrix):
    jobs, _ = matrix
    base = jobs[0]
    launcher = FakeLauncher()
    make_runner(tmp_path, launcher).run(jobs)
    calls_before = len(launcher.calls)

    art = pathlib.Path(base.out_root) / "final_summary.json"
    art.write_text(art.read_text() + "tampered\n")

    ledger = Ledger(tmp_path / "run_ledger.json")
    n_entries = len(ledger.entries())
    runner = MatrixRunner(ledger, launcher, serializer=SER, echo=lambda m: None)
    states = runner.run([base])

    assert len(launcher.calls) > calls_before
    entries = [e for e in ledger.entries() if e["job_id"] == base.job_id]
    stale = [e for e in entries if e["status"] == STALE]
    assert stale and "failed verification" in stale[-1]["reason"]
    assert stale[-1]["supersedes"] is not None
    assert states[base.job_id]["status"] == DONE
    # append-only: nothing was rewritten, the history only grew
    assert len(ledger.entries()) > n_entries


def test_ledger_is_append_only_and_seq_is_monotonic(tmp_path):
    ledger = Ledger(tmp_path / "l.json")
    ledger.append({"job_id": "a", "status": "RUNNING"})
    ledger.append({"job_id": "a", "status": DONE})
    ledger.append({"job_id": "a", "status": STALE, "reason": "wrong", "supersedes": 1})
    entries = ledger.entries()
    assert [e["seq"] for e in entries] == [0, 1, 2]
    assert [e["status"] for e in entries] == ["RUNNING", DONE, STALE]
    assert ledger.state_of("a")["status"] == STALE
    assert ledger.state_of("nope") is None


# --------------------------------------------------------------------------------------
# resume and rollback
# --------------------------------------------------------------------------------------

def test_interrupted_job_resumes_from_the_newest_valid_checkpoint(tmp_path, matrix):
    jobs, _ = matrix
    base = jobs[0]
    launcher = FakeLauncher(stop_at={base.job_id: 32})
    ledger = Ledger(tmp_path / "run_ledger.json")
    runner = MatrixRunner(ledger, launcher, serializer=SER, echo=lambda m: None)
    runner.run_job(base, ledger.states())
    assert ledger.state_of(base.job_id)["status"] == INTERRUPTED
    assert ledger.state_of(base.job_id)["step"] == 30

    launcher.stop_at = {}
    runner2 = MatrixRunner(ledger, launcher, serializer=SER, echo=lambda m: None)
    runner2.run_job(base, ledger.states())
    assert launcher.calls[-1].resume_step == 30
    assert launcher.calls[-1].init_from == "resume"
    assert ledger.state_of(base.job_id)["status"] == DONE
    assert ledger.state_of(base.job_id)["step"] == 50


def test_corrupt_newest_checkpoint_rolls_back_one(tmp_path, matrix):
    jobs, _ = matrix
    base = jobs[0]
    launcher = FakeLauncher(stop_at={base.job_id: 32})
    ledger = Ledger(tmp_path / "run_ledger.json")
    MatrixRunner(ledger, launcher, serializer=SER, echo=lambda m: None).run_job(
        base, ledger.states())

    ck = DurableCheckpointer(base.out_root, base.job_id,
                             experiment_name=base.experiment_name, serializer=SER)
    newest = ck.resolve()
    assert newest.step == 30
    with open(newest.path, "r+b") as fh:
        fh.truncate(5)

    launcher.stop_at = {}
    MatrixRunner(ledger, launcher, serializer=SER, echo=lambda m: None).run_job(
        base, ledger.states())
    plan = launcher.calls[-1]
    assert plan.resume_step == 20, "should have rolled back exactly one checkpoint"
    assert plan.rolled_back_from == newest.path
    running = [e for e in ledger.entries()
               if e["job_id"] == base.job_id and e["status"] == "RUNNING"]
    assert running[-1]["rolled_back_from"] == newest.path


def test_job_that_exits_zero_without_artifacts_is_not_marked_done(tmp_path, matrix):
    jobs, _ = matrix
    base = jobs[0]

    class NoArtifacts(FakeLauncher):
        def __call__(self, plan):
            res = super().__call__(plan)
            (pathlib.Path(plan.spec.out_root) / "final_summary.json").unlink(missing_ok=True)
            return res

    ledger = Ledger(tmp_path / "run_ledger.json")
    MatrixRunner(ledger, NoArtifacts(), serializer=SER, echo=lambda m: None).run_job(
        base, ledger.states())
    entry = ledger.state_of(base.job_id)
    assert entry["status"] == FAILED
    assert "missing final artifacts" in entry["reason"]


def test_resume_refuses_a_changed_config(tmp_path, matrix):
    jobs, cfg = matrix
    base = jobs[0]
    launcher = FakeLauncher(stop_at={base.job_id: 32})
    ledger = Ledger(tmp_path / "run_ledger.json")
    MatrixRunner(ledger, launcher, serializer=SER, echo=lambda m: None).run_job(
        base, ledger.states())

    cfg.write_text("trainer:\n  train_batches: 999\n")   # the frozen surface moved
    with pytest.raises(RuntimeError, match="config_sha256 changed"):
        MatrixRunner(ledger, launcher, serializer=SER, echo=lambda m: None).run_job(
            base, ledger.states())


# --------------------------------------------------------------------------------------
# H3 branch parity
# --------------------------------------------------------------------------------------

def test_near_and_far_record_the_same_parent_checkpoint_sha(tmp_path, matrix):
    jobs, _ = matrix
    states = make_runner(tmp_path, FakeLauncher()).run(jobs)
    base, near, far = (states[j] for j in
                       ("gpt-s1234-base", "gpt-s1234-adapt-near", "gpt-s1234-adapt-far"))
    assert near["parent_checkpoint_sha256"] == far["parent_checkpoint_sha256"]
    assert near["parent_checkpoint_sha256"] == base["final_checkpoint_sha256"]
    assert sha256_file(base["final_checkpoint"]) == base["final_checkpoint_sha256"]


def test_branch_parity_check_catches_a_crossed_parent(tmp_path):
    jobs = build_matrix(tmp_path / "r", models=("gpt",), seeds=(1234,))
    states = {
        "gpt-s1234-adapt-near": {"status": DONE, "parent_checkpoint_sha256": "a" * 64},
        "gpt-s1234-adapt-far": {"status": DONE, "parent_checkpoint_sha256": "b" * 64},
    }
    with pytest.raises(RuntimeError, match="do not share a parent checkpoint"):
        assert_branch_parity(states, jobs)
    states["gpt-s1234-adapt-far"]["parent_checkpoint_sha256"] = "a" * 64
    assert_branch_parity(states, jobs)


def test_adaptation_job_will_not_start_before_its_parent_is_done(tmp_path, matrix):
    jobs, _ = matrix
    near = next(j for j in jobs if j.condition == "near")
    ledger = Ledger(tmp_path / "run_ledger.json")
    with pytest.raises(RuntimeError, match="needs parent"):
        MatrixRunner(ledger, FakeLauncher(), serializer=SER, echo=lambda m: None).run_job(
            near, ledger.states())


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------

def test_step_metrics_are_atomic_and_keyed_by_run_and_step(tmp_path):
    write_step_metrics(tmp_path, "gpt-s1234-base", 250, {"loss": 0.5})
    p = tmp_path / "metrics" / "step_250.json"
    body = json.loads(p.read_text())
    assert body == {"run_id": "gpt-s1234-base", "step": 250, "loss": 0.5}
    assert list((tmp_path / "metrics").glob("*.partial")) == []

    # rewriting the same key from the same run is fine (a resume replays it)
    write_step_metrics(tmp_path, "gpt-s1234-base", 250, {"loss": 0.5})
    # from a different run it is a shared-output-root bug
    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_step_metrics(tmp_path, "gpt-s1234-adapt-near", 250, {"loss": 0.1})


# --------------------------------------------------------------------------------------
# the real command
# --------------------------------------------------------------------------------------

def test_fabric_command_pins_the_single_gpu_paper_configuration(tmp_path):
    spec = JobSpec("nextlat-s1234-base", "nextlat", 1234, "base", None,
                   "configs/nextlat.yaml", str(tmp_path / "runs" / "nextlat" / "1234" / "base"))
    cmd = FabricLauncher(tmp_path, dry_run=True).command(
        ResumePlan(spec=spec, fresh=True))
    assert cmd[:2] == ["fabric", "run"]
    assert "--devices" in cmd and cmd[cmd.index("--devices") + 1] == "1"
    assert cmd[cmd.index("--precision") + 1] == "bf16-mixed"
    assert "trainer.compile=false" in cmd                    # spec section 8
    assert "trainer.save_recovery_checkpoint=250" in cmd     # spec section 9.2 item 1
    assert "seed=1234" in cmd
    assert f"trainer.out_dir={pathlib.Path(spec.out_root).resolve()}" in cmd
    assert "trainer.experiment_name=nextlat-s1234-base" in cmd
    assert "trainer.init_from=scratch" in cmd
    assert not any(c.startswith("trainer.out_dir=output/") for c in cmd)


def test_fabric_command_resumes_and_branches_from_a_parent(tmp_path):
    spec = JobSpec("gpt-s1235-adapt-near", "gpt", 1235, "adapt", "near",
                   "configs/gpt_adapt.yaml", str(tmp_path / "near"),
                   parent_job_id="gpt-s1235-base", train_batches=500,
                   overrides=("model.lambda_mse=0.0", "model.lambda_kl=0.0"))
    launcher = FabricLauncher(tmp_path, dry_run=True)

    fresh = launcher.command(ResumePlan(spec=spec, fresh=True,
                                        parent_checkpoint="/d/base/final.pt"))
    assert "--checkpoint_path" in fresh
    assert fresh[fresh.index("--checkpoint_path") + 1] == "/d/base/final.pt"
    assert "trainer.train_batches=500" in fresh
    assert "model.lambda_mse=0.0" in fresh and "model.lambda_kl=0.0" in fresh

    # once the branch has its own checkpoints, the parent must not be re-applied
    resumed = launcher.command(ResumePlan(spec=spec, fresh=False, resume_step=250,
                                          parent_checkpoint="/d/base/final.pt"))
    assert "--checkpoint_path" not in resumed
    assert "trainer.init_from=resume" in resumed
