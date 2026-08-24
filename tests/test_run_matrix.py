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
    MODELS,
    SEEDS,
    assert_branch_parity,
    build_matrix,
    default_config_for,
    default_overrides_for,
    job_id,
    upstream_experiment_dir_name,
    validate_matrix,
    write_step_metrics,
)

SER = pickle_serializer()


# --------------------------------------------------------------------------------------
# a fake launcher that writes real checkpoints
# --------------------------------------------------------------------------------------

class FakeLauncher:
    """Trains up to the OFFSET step target, checkpointing every 10, then stops or fails.

    The offset is the point. `--checkpoint_path` restores `training_steps`
    (`models/model_base.py:437`), `core_train.py:309` seeds `self.step` from it, and
    `core_train.py:569` returns as soon as `self.step > trainer.train_batches`. So a branch
    off a P-step parent begins at step P and must be asked for `P + adapt_steps`, not
    `adapt_steps`. This launcher reproduces that arithmetic, which is what makes `updates`
    (steps taken beyond the parent) a real quantity in these tests instead of a relabelled
    step counter -- and what lets `ignore_parent_offset=True` reproduce deviation D-19.

    It also writes its checkpoints into `spec.experiment_dir_name`, i.e. the directory
    upstream would really have created after `train.py:98-99` appended `-seed{seed}` (D-18).
    """

    def __init__(self, stop_at: dict[str, int] | None = None,
                 fail: set[str] | None = None, ignore_parent_offset: bool = False):
        self.stop_at = stop_at or {}
        self.fail = fail or set()
        self.ignore_parent_offset = ignore_parent_offset
        self.calls: list[ResumePlan] = []
        self.updates: dict[str, int] = {}

    @staticmethod
    def _ckpt(spec: JobSpec) -> DurableCheckpointer:
        return DurableCheckpointer(spec.out_root, spec.job_id,
                                   experiment_name=spec.experiment_dir_name, serializer=SER)

    def target(self, plan: ResumePlan) -> int:
        """What `FabricLauncher.command` puts in `trainer.train_batches=`."""
        if self.ignore_parent_offset:
            return plan.spec.train_batches      # the D-19 bug, verbatim
        return (plan.parent_steps or 0) + plan.spec.train_batches

    def __call__(self, plan: ResumePlan) -> LaunchResult:
        self.calls.append(plan)
        spec = plan.spec
        ck = self._ckpt(spec)
        target = self.target(plan)
        start = plan.resume_step or (plan.parent_steps or 0)
        stop = self.stop_at.get(spec.job_id, target)
        step = start
        while step < min(stop, target):
            step += 1
            self.updates[spec.job_id] = self.updates.get(spec.job_id, 0) + 1
            if step % 10 == 0:
                ck.save({"step": step, "job": spec.job_id,
                         "parent": plan.parent_checkpoint_sha256}, step)
                write_step_metrics(spec.out_root, spec.job_id, step,
                                   {"loss": 1.0 / step})
        if spec.job_id in self.fail:
            return LaunchResult(1, step, "simulated crash")
        if step < target:
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
    assert job_id("bst", 1236, "base") == "bst-s1236-base"
    assert job_id("bst", 1236, "adapt", "far") == "bst-s1236-adapt-far"
    with pytest.raises(ValueError):
        job_id("llama", 1234, "base")


def test_the_matrix_is_three_arms(tmp_path):
    """Spec sec.8: gpt, nextlat AND bst, at the three preregistered seeds.

    BST is the competence-matched control (docs/DECISION_D20_competence_gate.md, "Superseded
    in part"): the paper's Figure 6 has GPT at ~18.6% -- 1/d, i.e. chance -- and BST at
    ~99.9%, so NextLat-vs-GPT cannot separate objective from task success and
    NextLat-vs-BST can. Dropping the arm silently downgrades the primary contrast, and
    nothing else in this repository would notice, so the arity is asserted here.
    """
    assert MODELS == ("gpt", "nextlat", "bst")
    assert SEEDS == (1234, 1235, 1236)

    jobs = build_matrix(tmp_path / "r")
    base = [j for j in jobs if j.phase == "base"]
    adapt = [j for j in jobs if j.phase == "adapt"]
    assert len(base) == 9, "3 models x 3 seeds"
    assert len(adapt) == 18, "3 models x 3 seeds x {near, far}"
    assert len(jobs) == 27

    assert {j.model for j in jobs} == {"gpt", "nextlat", "bst"}
    for model in MODELS:
        assert len([j for j in base if j.model == model]) == 3
        assert len([j for j in adapt if j.model == model]) == 6
        assert {j.seed for j in jobs if j.model == model} == set(SEEDS)


def test_every_branch_gets_its_own_output_root(tmp_path):
    """No two of the 27 jobs may share an output root, across arms as well as within one.

    Upstream resolves `recovery_ckpt` / `latest_ckpt` at `trainer.out_dir`
    (core_train.py:944-948, 970-974), one directory ABOVE the experiment directory, and
    `init_from: resume` reads them from there. A shared root therefore lets whichever job
    wrote last own the other's resume pointer -- and between a `near` and a `far` branch
    that silently gives them one parent and empties the H3 contrast.
    """
    jobs = build_matrix(tmp_path / "r")
    assert len(jobs) == 3 * 3 * 3          # 3 models x 3 seeds x (base, near, far)
    roots = [pathlib.Path(j.out_root).resolve() for j in jobs]
    assert len(set(roots)) == len(roots) == 27
    # and no root may nest inside another, which validate_matrix enforces for real
    for i, a in enumerate(roots):
        for b in roots[i + 1:]:
            assert a not in b.parents and b not in a.parents

    for model in MODELS:
        for seed in SEEDS:
            near = next(j for j in jobs if j.job_id == job_id(model, seed, "adapt", "near"))
            far = next(j for j in jobs if j.job_id == job_id(model, seed, "adapt", "far"))
            base = next(j for j in jobs if j.job_id == job_id(model, seed, "base"))
            assert len({near.out_root, far.out_root, base.out_root}) == 3
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
    """A branch must ask for parent_steps + adapt_steps, not adapt_steps.

    `--checkpoint_path` restores `training_steps` (`models/model_base.py:437`), the trainer
    seeds `self.step` from it (`core_train.py:309`), and the loop returns as soon as
    `self.step > trainer.train_batches` (`core_train.py:569`). Asking for
    `trainer.train_batches=500` off a 20,000-step parent buys ZERO adaptation updates.
    `docs/UPSTREAM_REPORT.md` section 3.4 names this trap explicitly.
    """
    spec = JobSpec("gpt-s1235-adapt-near", "gpt", 1235, "adapt", "near",
                   "configs/gpt_adapt.yaml", str(tmp_path / "near"),
                   parent_job_id="gpt-s1235-base", train_batches=500,
                   overrides=("model.lambda_mse=0.0", "model.lambda_kl=0.0"))
    launcher = FabricLauncher(tmp_path, dry_run=True)

    fresh = launcher.command(ResumePlan(spec=spec, fresh=True,
                                        parent_checkpoint="/d/base/final.pt",
                                        parent_steps=20000))
    assert "--checkpoint_path" in fresh
    assert fresh[fresh.index("--checkpoint_path") + 1] == "/d/base/final.pt"
    assert "trainer.train_batches=20500" in fresh
    assert "trainer.train_batches=500" not in fresh, (
        "500 off a 20,000-step parent returns at core_train.py:569 without one update"
    )
    assert "model.lambda_mse=0.0" in fresh and "model.lambda_kl=0.0" in fresh

    # the offset survives the resume: the branch's OWN checkpoints also carry the offset
    # counter, so dropping it on the second launch breaks the resume exactly the same way.
    resumed = launcher.command(ResumePlan(spec=spec, fresh=False, resume_step=20250,
                                          parent_checkpoint="/d/base/final.pt",
                                          parent_steps=20000))
    assert "--checkpoint_path" not in resumed
    assert "trainer.init_from=resume" in resumed
    assert "trainer.train_batches=20500" in resumed


def test_branch_command_is_refused_without_the_parent_step_count(tmp_path):
    """Unknown parent step count must be a refusal, never a silent `train_batches=500`."""
    spec = JobSpec("gpt-s1235-adapt-far", "gpt", 1235, "adapt", "far",
                   "configs/adapt_far.yaml", str(tmp_path / "far"),
                   parent_job_id="gpt-s1235-base", train_batches=500)
    with pytest.raises(ValueError, match="without the parent's step count"):
        FabricLauncher(tmp_path, dry_run=True).command(
            ResumePlan(spec=spec, fresh=True, parent_checkpoint="/d/base/final.pt"))


def test_base_job_command_carries_no_offset(tmp_path):
    spec = JobSpec("gpt-s1234-base", "gpt", 1234, "base", None,
                   "configs/gpt_lurestar.yaml", str(tmp_path / "base"), train_batches=20000)
    cmd = FabricLauncher(tmp_path, dry_run=True).command(ResumePlan(spec=spec, fresh=True))
    assert "trainer.train_batches=20000" in cmd
    assert "--checkpoint_path" not in cmd


# --------------------------------------------------------------------------------------
# the defaults the runner actually ships with
# --------------------------------------------------------------------------------------

def test_default_matrix_points_at_configs_that_exist(tmp_path):
    """`configs/{model}_lurestar_base.yaml` never existed; a job cannot launch without one."""
    jobs = build_matrix(tmp_path / "r")
    for j in jobs:
        assert pathlib.Path(j.config).is_file(), j.config
    assert build_matrix(tmp_path / "r", require_configs=False)  # opt-out still available
    with pytest.raises(FileNotFoundError, match="configs that do not exist"):
        build_matrix(tmp_path / "r", config_for=lambda m, p, c: str(tmp_path / "nope.yaml"))


def test_near_and_far_do_not_share_a_config(tmp_path):
    """Handing both H3 arms one config collapses the contrast the out_root split protects."""
    jobs = {j.job_id: j for j in build_matrix(tmp_path / "r", models=("gpt",), seeds=(1234,))}
    near = jobs["gpt-s1234-adapt-near"]
    far = jobs["gpt-s1234-adapt-far"]
    assert near.config != far.config
    assert pathlib.Path(near.config).name == "adapt_near.yaml"
    assert pathlib.Path(far.config).name == "adapt_far.yaml"


def test_gpt_adaptation_flips_use_nextlat_off(tmp_path):
    """configs/adapt_*.yaml are derived from the NextLat YAML and set `use_nextlat: true`."""
    jobs = {j.job_id: j for j in build_matrix(tmp_path / "r", seeds=(1234,))}
    assert "use_nextlat=false" in jobs["gpt-s1234-adapt-near"].overrides
    assert "use_nextlat=false" in jobs["gpt-s1234-adapt-far"].overrides
    assert jobs["nextlat-s1234-adapt-near"].overrides == ()
    cmd = FabricLauncher(tmp_path, dry_run=True).command(
        ResumePlan(spec=jobs["gpt-s1234-adapt-near"], fresh=True,
                   parent_checkpoint="/d/p.pt", parent_steps=20000))
    assert "use_nextlat=false" in cmd


def test_bst_adaptation_restates_its_whole_objective(tmp_path):
    """The BST branch needs three overrides, not one, and the third is the easy one to miss.

    `configs/adapt_*.yaml` is a copy of the NextLat G(5,5) YAML, which never mentions
    `bst_pair_minimum_gap`, so the merge falls through to `defaults.yaml:98` = 1 while the
    BST base parent was trained at 2 (`bst_stargraph_5_5.yaml:42`, read into BSTConfig at
    `core_train.py:80`). Adapting a gap-2 parent under a gap-1 objective would change the
    loss between base and adaptation in the BST arm alone -- a confound inside the arm that
    exists to remove a confound.
    """
    jobs = {j.job_id: j for j in build_matrix(tmp_path / "r", seeds=(1234,))}
    for cond in ("near", "far"):
        ov = jobs[f"bst-s1234-adapt-{cond}"].overrides
        assert "use_bst=true" in ov
        assert "use_nextlat=false" in ov
        assert "model.bst_pair_minimum_gap=2" in ov

    cmd = FabricLauncher(tmp_path, dry_run=True).command(
        ResumePlan(spec=jobs["bst-s1234-adapt-near"], fresh=True,
                   parent_checkpoint="/d/p.pt", parent_steps=20000))
    for override in ("use_bst=true", "use_nextlat=false", "model.bst_pair_minimum_gap=2"):
        assert override in cmd
    # base jobs need no flag overrides: each arm has its own copied upstream YAML
    assert jobs["bst-s1234-base"].overrides == ()
    assert default_overrides_for("bst", "base") == ()
    with pytest.raises(ValueError, match="unknown model"):
        default_overrides_for("llama", "adapt")


def test_each_arm_gets_its_own_base_config(tmp_path):
    """Three arms, three copied upstream YAMLs. A shared base file would erase the arm."""
    assert default_config_for("bst", "base", None).endswith("configs/bst_lurestar.yaml")
    assert default_config_for("gpt", "base", None).endswith("configs/gpt_lurestar.yaml")
    assert default_config_for("nextlat", "base", None).endswith(
        "configs/nextlat_lurestar.yaml")
    with pytest.raises(ValueError, match="unknown model"):
        default_config_for("llama", "base", None)

    jobs = build_matrix(tmp_path / "r")
    base_cfgs = {j.model: j.config for j in jobs if j.phase == "base"}
    assert len(set(base_cfgs.values())) == 3
    for cfg in base_cfgs.values():
        assert pathlib.Path(cfg).is_file(), cfg


def test_default_matrix_carries_the_dataset_manifests(tmp_path):
    """Spec section 9.3 item 4: a resume must preserve the manifests, so they must be wired."""
    for j in build_matrix(tmp_path / "r", models=("gpt",), seeds=(1234,)):
        assert j.manifests, j.job_id
        for m in j.manifests:
            assert pathlib.Path(m).is_file(), m


# --------------------------------------------------------------------------------------
# the identity guard
# --------------------------------------------------------------------------------------

def test_identity_guard_refuses_a_config_that_was_absent_when_the_job_started(tmp_path):
    """A recorded `None` used to disable the guard forever. It must be a refusal instead."""
    cfg = tmp_path / "cfg.yaml"
    spec = JobSpec("gpt-s1234-base", "gpt", 1234, "base", None, str(cfg), str(tmp_path / "o"))
    runner = make_runner(tmp_path, FakeLauncher())

    with pytest.raises(FileNotFoundError, match="does not exist"):
        runner._identity(spec)                       # never record config_sha256=None

    cfg.write_text("trainer:\n  train_batches: 50\n")
    prior = dict(runner._identity(spec))
    runner._check_identity(spec, prior)              # unchanged config is fine

    cfg.write_text("trainer:\n  train_batches: 999999\n")
    with pytest.raises(RuntimeError, match="config_sha256 changed"):
        runner._check_identity(spec, prior)

    # and a legacy ledger entry that pinned None is a refusal, not a free pass
    legacy = dict(prior, config_sha256=None)
    with pytest.raises(RuntimeError, match="config_sha256 changed"):
        runner._check_identity(spec, legacy)


def test_identity_guard_refuses_a_moved_manifest(tmp_path):
    man = tmp_path / "corpus.sha256"
    man.write_text("d13199b0  graph_5_5_sample_200000.txt\n")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("trainer:\n  train_batches: 50\n")
    spec = JobSpec("gpt-s1234-base", "gpt", 1234, "base", None, str(cfg),
                   str(tmp_path / "o"), manifests=(str(man),))
    runner = make_runner(tmp_path, FakeLauncher())
    prior = dict(runner._identity(spec))
    man.write_text("00000000  graph_5_5_sample_200000.txt\n")
    with pytest.raises(RuntimeError, match="manifest_sha256 changed"):
        runner._check_identity(spec, prior)


# --------------------------------------------------------------------------------------
# adoption of checkpoints upstream wrote
# --------------------------------------------------------------------------------------

def test_runner_resumes_from_a_checkpoint_upstream_wrote(tmp_path):
    """The production launcher is upstream `train.py`; it writes through fabric.save, not us.

    Without adoption the index is empty for every real job, `resolve()` returns None, and the
    runner plans `init_from=scratch` on top of a valid 15,000-step checkpoint.
    """
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("trainer:\n  train_batches: 20000\n")
    out = tmp_path / "runs" / "gpt" / "1234" / "base"
    # D-18: train.py:98-99 appends "-seed1234" because "gpt-s1234-base" has no "seed" in it
    exp = out / "gpt-s1234-base-seed1234"
    exp.mkdir(parents=True)
    save, _ = SER
    # exactly what core_train.py:961 and core_train.py:774-777 leave on disk
    for name, step in (("recovery_ckpt_iter_15000.pt", 15000),
                       ("ckpt_iter_14000_0.4412.pt", 14000)):
        with open(exp / name, "wb") as fh:
            save({"training_steps": step}, fh)
    (out / "recovery_ckpt").write_text(str(exp / "recovery_ckpt_iter_15000.pt"))

    spec = JobSpec("gpt-s1234-base", "gpt", 1234, "base", None, str(cfg), str(out),
                   train_batches=20000)
    runner = make_runner(tmp_path, FakeLauncher())
    plan = runner.plan(spec, {})
    assert not plan.fresh, "an upstream checkpoint on disk must not plan a scratch restart"
    assert plan.init_from == "resume"
    assert plan.resume_step == 15000
    assert (out / "recovery_ckpt").is_file(), "the upstream pointer must survive planning"
    # the float in upstream's validation-checkpoint name must not be read as the step
    assert sorted(r.step for r in runner.checkpointer(spec).read_index()) == [14000, 15000]


def test_a_torn_upstream_checkpoint_rolls_back_to_the_validation_checkpoint(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("trainer:\n  train_batches: 20000\n")
    out = tmp_path / "runs" / "gpt" / "1234" / "base"
    exp = out / "gpt-s1234-base-seed1234"      # D-18
    exp.mkdir(parents=True)
    save, _ = SER
    for name, step in (("recovery_ckpt_iter_15000.pt", 15000),
                       ("ckpt_iter_14000_0.4412.pt", 14000)):
        with open(exp / name, "wb") as fh:
            save({"training_steps": step}, fh)
    torn = exp / "recovery_ckpt_iter_15000.pt"
    (out / "recovery_ckpt").write_text(str(torn))
    with open(torn, "r+b") as fh:      # killed mid fabric.save: model_base.py:417 is not atomic
        fh.truncate(7)

    spec = JobSpec("gpt-s1234-base", "gpt", 1234, "base", None, str(cfg), str(out))
    plan = make_runner(tmp_path, FakeLauncher()).plan(spec, {})
    assert plan.resume_step == 14000
    assert (out / "recovery_ckpt").read_text().strip() == str(exp / "ckpt_iter_14000_0.4412.pt")
