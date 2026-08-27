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
from materialize_base_competence import main as competence_main
from materialize_base_competence import materialize as materialize_base_competence
from run_matrix import (
    COMPETENCE_DECODING,
    COMPETENCE_RECEIPT,
    COMPETENCE_RECEIPT_SIDECAR,
    COMPLETION_SUMMARY,
    DONE,
    FAILED,
    INTERRUPTED,
    STALE,
    TRAINED,
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
    verify_base_competence_receipt,
    verified_adaptation_manifests,
    verified_h3_permanent_block,
    main,
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
    the guarded runtime patch stops once `self.step >= trainer.train_batches`. A branch off a
    P-step parent must therefore be asked for the absolute target `P + adapt_steps`, not the
    relative `adapt_steps`. This launcher reproduces that arithmetic, which makes `updates`
    (steps taken beyond the parent) a real quantity in these tests instead of a relabelled
    step counter -- and lets `ignore_parent_offset=True` reproduce a target under-run.

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
            return plan.spec.train_batches      # intentionally wrong relative target
        return (plan.parent_steps or 0) + plan.spec.train_batches

    @staticmethod
    def _write_upstream_artifacts(spec: JobSpec, step: int) -> None:
        """Reproduce the files `train.py` + CSVLogger write, not a toy-only summary."""
        experiment = pathlib.Path(spec.checkpoint_dir)
        (experiment / "version_0").mkdir(parents=True, exist_ok=True)
        (experiment / "materialized_config.yaml").write_text(
            f"seed: {spec.seed}\ntrainer:\n  train_batches: {step}\n"
        )
        (experiment / "version_0" / "metrics.csv").write_text(
            f"step,train_loss\n{step},0.1\n"
        )
        contract = pathlib.Path(spec.out_root) / "metrics" / "step_0_contract.json"
        contract.parent.mkdir(parents=True, exist_ok=True)
        contract.write_text(json.dumps({
            "schema": 1,
            "job_id": spec.job_id,
            "adaptation": (
                {
                    "contract": "h3_full_parameter_next_token_ce_v1",
                    "contract_sha256": sha256_file(
                        pathlib.Path(__file__).resolve().parents[1]
                        / "src/lurestar/adaptation.py"
                    ),
                    "full_parameter": True,
                    "loss": "teacher_forced_next_token_cross_entropy",
                    "bst_dense_prefix_suffix_objective": False if spec.model == "bst" else None,
                    "bst_backward_input": (
                        "item_independent_lone_eos" if spec.model == "bst" else None
                    ),
                }
                if spec.phase == "adapt" else None
            ),
        }, sort_keys=True) + "\n")

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
        self._write_upstream_artifacts(spec, step)
        return LaunchResult(0, step, "ok")


@pytest.fixture()
def matrix(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("trainer:\n  train_batches: 50\n")
    evaluator = tmp_path / "fixture_evaluator.py"
    evaluator.write_text("# deterministic test-only exact-path evaluator\n")
    dataset = tmp_path / "graph_5_5_test_20000.txt"
    dataset.write_text("held-out path-star fixture\n")
    manifest = tmp_path / "fixture_test_dataset.sha256"
    manifest.write_text(f"{sha256_file(dataset)}  {dataset.name}\n")
    jobs = build_matrix(tmp_path / "root", models=("gpt",), seeds=(1234,),
                        config_for=lambda m, p, c: str(cfg),
                        base_steps=50, adapt_steps=50,
                        competence_evaluator=evaluator, competence_dataset=dataset,
                        competence_manifests=(manifest,))
    return jobs, cfg


def make_runner(tmp_path, launcher) -> MatrixRunner:
    return MatrixRunner(Ledger(tmp_path / "run_ledger.json"), launcher,
                        serializer=SER, echo=lambda m: None)


def test_recovery_barrier_terminalizes_exact_checkpoint_and_quarantines_poison(
        matrix, tmp_path) -> None:
    base = matrix[0][0]
    ck = DurableCheckpointer(
        base.out_root, base.job_id, experiment_name=base.experiment_dir_name, serializer=SER)
    final = ck.save({"training_steps": 50, "job": base.job_id}, 50, kind="final")
    FakeLauncher._write_upstream_artifacts(base, 50)
    experiment = pathlib.Path(base.checkpoint_dir)
    version0 = experiment / "version_0" / "metrics.csv"
    poison = experiment / "version_1" / "metrics.csv"
    poison.parent.mkdir(parents=True)
    poison.write_text("step,train_loss\n51,0.0\n")
    materialized = experiment / "materialized_config.yaml"
    step0 = pathlib.Path(base.out_root) / "metrics" / "step_0_contract.json"
    provenance = {
        "checkpoint_creation_source_sha256": "a" * 64,
        "successor_terminalization_source_sha256": "b" * 64,
        "checkpoint_generation": "123",
        "checkpoint_sha256": final.sha256,
        "recovery_receipt_sha256": "c" * 64,
    }
    barrier = {base.job_id: {
        "job_id": base.job_id, "target_step": 50,
        "checkpoint_path": final.path, "checkpoint_sha256": final.sha256,
        "recovery_provenance": provenance,
        "authoritative_artifacts": {
            str(path.resolve()): sha256_file(path)
            for path in (materialized, version0, step0)
        },
    }}

    def forbidden_launcher(_plan):
        raise AssertionError("recovered exact-target job invoked the trainer")

    ledger = Ledger(tmp_path / "successor-ledger.json")
    states = MatrixRunner(
        ledger, forbidden_launcher, serializer=SER, echo=lambda _: None,
        recovery_barrier=barrier).run([base])

    assert states[base.job_id]["status"] == TRAINED
    assert states[base.job_id]["recovery_provenance"] == provenance
    summary = json.loads((pathlib.Path(base.out_root) / COMPLETION_SUMMARY).read_text())
    assert summary["recovery_provenance"] == provenance
    assert not poison.exists()
    assert (pathlib.Path(base.out_root) / "quarantine" / "unbound_retry_telemetry" /
            poison.relative_to(base.out_root)).is_file()


def test_recovery_barrier_overstep_aborts_before_any_launcher(matrix, tmp_path) -> None:
    base = matrix[0][0]
    ck = DurableCheckpointer(
        base.out_root, base.job_id, experiment_name=base.experiment_dir_name, serializer=SER)
    over = ck.save({"training_steps": 51}, 51, kind="final")
    calls = []
    runner = MatrixRunner(
        Ledger(tmp_path / "successor-ledger.json"), lambda plan: calls.append(plan),
        serializer=SER, echo=lambda _: None,
        recovery_barrier={base.job_id: {
            "target_step": 50, "checkpoint_path": over.path,
            "checkpoint_sha256": over.sha256, "authoritative_artifacts": {},
            "recovery_provenance": {},
        }})

    with pytest.raises(RuntimeError, match="no launcher was invoked"):
        runner.run([base])
    assert calls == []


def test_adoption_rejects_filename_payload_training_step_mismatch(matrix) -> None:
    base = matrix[0][0]
    path = pathlib.Path(base.checkpoint_dir) / "recovery_ckpt_iter_50.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    save, _ = SER
    with path.open("wb") as handle:
        save({"training_steps": 51}, handle)
    ck = DurableCheckpointer(
        base.out_root, base.job_id, experiment_name=base.experiment_dir_name, serializer=SER)

    with pytest.raises(Exception, match="loaded training_steps 51 != filename step 50"):
        ck.adopt(path)


def _attach_competence_fixture(
    ledger: Ledger, spec: JobSpec, *, accuracy: float = 0.95
) -> dict:
    """Explicitly emulate a separately executed, hash-recorded scientific evaluator."""
    prior = ledger.state_of(spec.job_id)
    assert prior and prior["status"] == TRAINED
    assert spec.competence_evaluator and spec.competence_dataset
    evaluator = pathlib.Path(spec.competence_evaluator).resolve()
    dataset = pathlib.Path(spec.competence_dataset).resolve()
    manifests = [pathlib.Path(path).resolve() for path in spec.competence_manifests]
    evaluator_output = pathlib.Path(spec.out_root) / "evaluation" / "raw_metrics.json"
    evaluator_output.parent.mkdir(parents=True, exist_ok=True)
    total = 100
    correct = round(accuracy * total)
    evaluator_output.write_text(json.dumps({
        "schema": "nextlat_forgetting/exact_path_evaluation/1",
        "job_id": spec.job_id,
        "model": spec.model,
        "seed": spec.seed,
        "checkpoint_sha256": prior["final_checkpoint_sha256"],
        "dataset_sha256": sha256_file(dataset),
        "evaluator_sha256": sha256_file(evaluator),
        "manifest_sha256s": sorted(sha256_file(path) for path in manifests),
        "decoding": COMPETENCE_DECODING,
        "exact_path_accuracy": {
            "correct": correct, "total": total, "value": correct / total,
        },
    }, sort_keys=True) + "\n")
    receipt_path = pathlib.Path(spec.out_root) / COMPETENCE_RECEIPT
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "nextlat_forgetting/base_competence/1",
        "job_id": spec.job_id,
        "model": spec.model,
        "seed": spec.seed,
        "phase": "base",
        "checkpoint": {
            "path": prior["final_checkpoint"],
            "sha256": prior["final_checkpoint_sha256"],
        },
        "evaluator": {"path": str(evaluator), "sha256": sha256_file(evaluator)},
        "evaluator_output": {
            "path": str(evaluator_output), "sha256": sha256_file(evaluator_output),
        },
        "evaluation_dataset": {"path": str(dataset), "sha256": sha256_file(dataset)},
        "manifests": [
            {"path": str(path), "sha256": sha256_file(path)} for path in manifests
        ],
        "decoding": COMPETENCE_DECODING,
        "competence_identity": prior["competence_identity"],
        "exact_path_accuracy": {
            "correct": correct,
            "total": total,
            "value": correct / total,
        },
    }
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    receipt_sha = sha256_file(receipt_path)
    sidecar = pathlib.Path(spec.out_root) / COMPETENCE_RECEIPT_SIDECAR
    sidecar.write_text(f"{receipt_sha}  {receipt_path.name}\n")
    promoted = {
        key: value for key, value in prior.items() if key not in ("seq", "ts", "status")
    }
    promoted.update({
        "job_id": spec.job_id,
        "status": DONE,
        "supersedes": prior["seq"],
        "artifacts": dict(
            prior["artifacts"],
            **{
                COMPETENCE_RECEIPT: receipt_sha,
                COMPETENCE_RECEIPT_SIDECAR: sha256_file(sidecar),
            },
        ),
        "evaluation_artifacts": {
            COMPETENCE_RECEIPT: receipt_sha,
            COMPETENCE_RECEIPT_SIDECAR: sha256_file(sidecar),
        },
    })
    return ledger.append(promoted)


def run_with_competence_fixtures(tmp_path, jobs, launcher):
    """Run base, explicitly attach evaluated receipts, then run adaptation branches."""
    runner = make_runner(tmp_path, launcher)
    base = [job for job in jobs if job.phase == "base"]
    adapt = [job for job in jobs if job.phase == "adapt"]
    runner.run(base)
    for spec in base:
        accuracy = 0.20 if spec.model == "gpt" else 0.95
        _attach_competence_fixture(runner.ledger, spec, accuracy=accuracy)
    return runner.run(adapt)


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
    """Spec sec.8: gpt, nextlat AND bst, at the five preregistered seeds.

    BST is the competence-matched control (docs/DECISION_D20_competence_gate.md, "Superseded
    in part"): the paper's Figure 6 has GPT at ~18.6% -- 1/d, i.e. chance -- and BST at
    ~99.9%, so NextLat-vs-GPT cannot separate objective from task success and
    NextLat-vs-BST can. Dropping the arm silently downgrades the primary contrast, and
    nothing else in this repository would notice, so the arity is asserted here.
    """
    assert MODELS == ("gpt", "nextlat", "bst")
    assert SEEDS == (1234, 1235, 1236, 1237, 1238)

    jobs = build_matrix(tmp_path / "r")
    base = [j for j in jobs if j.phase == "base"]
    adapt = [j for j in jobs if j.phase == "adapt"]
    assert len(base) == 15, "3 models x 5 seeds"
    assert len(adapt) == 45, "3 models x 5 seeds x {near, mid, far}"
    assert len(jobs) == 60

    assert {j.model for j in jobs} == {"gpt", "nextlat", "bst"}
    for model in MODELS:
        assert len([j for j in base if j.model == model]) == 5
        assert len([j for j in adapt if j.model == model]) == 15
        assert {j.seed for j in jobs if j.model == model} == set(SEEDS)


def test_every_branch_gets_its_own_output_root(tmp_path):
    """No two of the 45 jobs may share an output root, across arms as well as within one.

    Upstream resolves `recovery_ckpt` / `latest_ckpt` at `trainer.out_dir`
    (core_train.py:944-948, 970-974), one directory ABOVE the experiment directory, and
    `init_from: resume` reads them from there. A shared root therefore lets whichever job
    wrote last own the other's resume pointer -- and between a `near` and a `far` branch
    that silently gives them one parent and empties the H3 contrast.
    """
    jobs = build_matrix(tmp_path / "r")
    assert len(jobs) == 3 * 5 * 4          # 3 models x 5 seeds x (base, near, mid, far)
    roots = [pathlib.Path(j.out_root).resolve() for j in jobs]
    assert len(set(roots)) == len(roots) == 60
    # and no root may nest inside another, which validate_matrix enforces for real
    for i, a in enumerate(roots):
        for b in roots[i + 1:]:
            assert a not in b.parents and b not in a.parents

    for model in MODELS:
        for seed in SEEDS:
            near = next(j for j in jobs if j.job_id == job_id(model, seed, "adapt", "near"))
            mid = next(j for j in jobs if j.job_id == job_id(model, seed, "adapt", "mid"))
            far = next(j for j in jobs if j.job_id == job_id(model, seed, "adapt", "far"))
            base = next(j for j in jobs if j.job_id == job_id(model, seed, "base"))
            assert len({near.out_root, mid.out_root, far.out_root, base.out_root}) == 4
            assert near.parent_job_id == mid.parent_job_id == far.parent_job_id == base.job_id


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
    states = run_with_competence_fixtures(tmp_path, jobs, launcher)
    assert {j.job_id: states[j.job_id]["status"] for j in jobs} == {
        j.job_id: (DONE if j.phase == "base" else TRAINED) for j in jobs
    }
    first_calls = len(launcher.calls)
    assert first_calls == len(jobs)

    states2 = make_runner(tmp_path, launcher).run(jobs)
    assert len(launcher.calls) == first_calls, "a verified DONE job was relaunched"
    assert all(
        states2[j.job_id]["status"] == (DONE if j.phase == "base" else TRAINED)
        for j in jobs
    )


def test_tampered_completion_receipt_is_rebuilt_from_exact_checkpoint_without_rerun(
    tmp_path, matrix
):
    jobs, _ = matrix
    base = jobs[0]
    launcher = FakeLauncher()
    run_with_competence_fixtures(tmp_path, jobs, launcher)
    calls_before = len(launcher.calls)

    art = pathlib.Path(base.out_root) / COMPLETION_SUMMARY
    art.write_text(art.read_text() + "tampered\n")

    ledger = Ledger(tmp_path / "run_ledger.json")
    n_entries = len(ledger.entries())
    runner = MatrixRunner(ledger, launcher, serializer=SER, echo=lambda m: None)
    states = runner.run([base])

    assert len(launcher.calls) == calls_before
    entries = [e for e in ledger.entries() if e["job_id"] == base.job_id]
    stale = [e for e in entries if e["status"] == STALE]
    assert stale and "failed verification" in stale[-1]["reason"]
    assert stale[-1]["supersedes"] is not None
    assert states[base.job_id]["status"] == TRAINED
    assert states[base.job_id]["recovered_without_launch"] is True
    assert sha256_file(art) == states[base.job_id]["artifacts"][COMPLETION_SUMMARY]
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
    assert ledger.state_of(base.job_id)["status"] == TRAINED
    assert ledger.state_of(base.job_id)["step"] == 50


def test_exact_target_checkpoint_is_terminalized_without_launching(tmp_path, matrix):
    """A lost terminal ledger write must not turn step 50 into step 51 on recovery."""
    base = matrix[0][0]
    launcher = FakeLauncher()
    ledger = Ledger(tmp_path / "run_ledger.json")
    runner = MatrixRunner(ledger, launcher, serializer=SER, echo=lambda m: None)
    experiment = pathlib.Path(base.checkpoint_dir)
    experiment.mkdir(parents=True, exist_ok=True)
    checkpoint = experiment / "final_ckpt_iter_50.pt"
    save, _ = SER
    with checkpoint.open("wb") as handle:
        save({"training_steps": 50, "job": base.job_id}, handle)
    FakeLauncher._write_upstream_artifacts(base, 50)

    entry = runner.run_job(base, ledger.states())

    assert launcher.calls == [], "exact-target recovery must consume no optimizer step"
    assert entry["status"] == TRAINED
    assert entry["step"] == entry["updates"] == 50
    assert entry["final_checkpoint"] == str(checkpoint.resolve())
    assert entry["final_checkpoint_sha256"] == sha256_file(checkpoint)
    assert entry["recovered_without_launch"] is True
    assert not any(item["status"] == "RUNNING" for item in ledger.entries())
    summary = json.loads((pathlib.Path(base.out_root) / COMPLETION_SUMMARY).read_text())
    assert summary["checkpoint"]["sha256"] == sha256_file(checkpoint)


def test_recovered_checkpoint_beyond_exact_target_fails_without_launch(tmp_path, matrix):
    """An already-overshot checkpoint is protocol evidence, never a resume starting point."""
    base = matrix[0][0]
    launcher = FakeLauncher()
    ledger = Ledger(tmp_path / "run_ledger.json")
    runner = MatrixRunner(ledger, launcher, serializer=SER, echo=lambda m: None)
    experiment = pathlib.Path(base.checkpoint_dir)
    experiment.mkdir(parents=True, exist_ok=True)
    checkpoint = experiment / "final_ckpt_iter_51.pt"
    save, _ = SER
    with checkpoint.open("wb") as handle:
        save({"training_steps": 51, "job": base.job_id}, handle)

    entry = runner.run_job(base, ledger.states())

    assert launcher.calls == [], "over-target recovery must fail before launcher invocation"
    assert entry["status"] == FAILED
    assert entry["step"] == entry["updates"] == 51
    assert "exceeds exact absolute target 50" in entry["reason"]
    assert "refusing to launch" in entry["reason"]
    assert not any(item["status"] == "RUNNING" for item in ledger.entries())


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
            (pathlib.Path(plan.spec.checkpoint_dir) / "materialized_config.yaml").unlink()
            return res

    ledger = Ledger(tmp_path / "run_ledger.json")
    MatrixRunner(ledger, NoArtifacts(), serializer=SER, echo=lambda m: None).run_job(
        base, ledger.states())
    entry = ledger.state_of(base.job_id)
    assert entry["status"] == FAILED
    assert "upstream training artifacts are incomplete" in entry["reason"]


def test_runner_writes_completion_receipt_from_real_upstream_artifacts(tmp_path, matrix):
    """Upstream does not write final_summary.json; the runner creates it after verification."""
    jobs, _ = matrix
    base = jobs[0]
    ledger = Ledger(tmp_path / "run_ledger.json")
    MatrixRunner(ledger, FakeLauncher(), serializer=SER, echo=lambda m: None).run_job(
        base, ledger.states())

    entry = ledger.state_of(base.job_id)
    summary_path = pathlib.Path(base.out_root) / COMPLETION_SUMMARY
    summary = json.loads(summary_path.read_text())
    assert entry["status"] == TRAINED
    assert entry["status"] != DONE, "training alone is not completed scientific evaluation"
    assert summary["kind"] == "training_completion"
    assert summary["job_id"] == base.job_id
    assert summary["step"] == summary["updates"] == 50
    assert summary["checkpoint"]["sha256"] == entry["final_checkpoint_sha256"]
    assert any(p.endswith("materialized_config.yaml") for p in summary["training_artifacts"])
    assert any(p.endswith("metrics.csv") for p in summary["training_artifacts"])
    assert entry["artifacts"][COMPLETION_SUMMARY] == sha256_file(summary_path)


def test_done_requires_caller_required_scientific_evaluation_artifacts(tmp_path, matrix):
    jobs, _ = matrix
    original = jobs[0]
    spec = JobSpec(
        original.job_id, original.model, original.seed, original.phase, original.condition,
        original.config, original.out_root, manifests=original.manifests,
        competence_evaluator=original.competence_evaluator,
        competence_dataset=original.competence_dataset,
        competence_manifests=original.competence_manifests,
        train_batches=original.train_batches, final_artifacts=("evaluation.json",),
    )

    class EvaluatedLauncher(FakeLauncher):
        def __call__(self, plan):
            result = super().__call__(plan)
            (pathlib.Path(plan.spec.out_root) / "evaluation.json").write_text(
                json.dumps({"schema": "confirmatory_evaluation/1"}) + "\n"
            )
            return result

    ledger = Ledger(tmp_path / "run_ledger.json")
    MatrixRunner(ledger, EvaluatedLauncher(), serializer=SER, echo=lambda m: None).run_job(
        spec, ledger.states()
    )
    entry = ledger.state_of(spec.job_id)
    assert entry["status"] == DONE
    assert "evaluation.json" in entry["artifacts"]


def test_trained_job_promotes_to_done_after_eval_without_relaunch(tmp_path, matrix):
    jobs, _ = matrix
    original = jobs[0]
    launcher = FakeLauncher()
    ledger = Ledger(tmp_path / "run_ledger.json")
    runner = MatrixRunner(ledger, launcher, serializer=SER, echo=lambda m: None)
    runner.run_job(original, ledger.states())
    assert ledger.state_of(original.job_id)["status"] == TRAINED
    calls = len(launcher.calls)

    (pathlib.Path(original.out_root) / "evaluation.json").write_text(
        json.dumps({"schema": "confirmatory_evaluation/1"}) + "\n"
    )
    evaluated = JobSpec(
        original.job_id, original.model, original.seed, original.phase, original.condition,
        original.config, original.out_root, manifests=original.manifests,
        competence_evaluator=original.competence_evaluator,
        competence_dataset=original.competence_dataset,
        competence_manifests=original.competence_manifests,
        train_batches=original.train_batches, final_artifacts=("evaluation.json",),
    )
    states = ledger.states()
    runner.run_job(evaluated, states)
    entry = ledger.state_of(original.job_id)
    assert len(launcher.calls) == calls, "evaluation promotion must not rerun paid training"
    assert entry["status"] == DONE
    assert entry["supersedes"] is not None
    assert entry["evaluation_artifacts"]["evaluation.json"] == sha256_file(
        pathlib.Path(original.out_root) / "evaluation.json"
    )


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
    states = run_with_competence_fixtures(tmp_path, jobs, FakeLauncher())
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


def test_trained_only_parent_is_rejected_before_adaptation_planning(tmp_path, matrix):
    jobs, _ = matrix
    base = next(j for j in jobs if j.phase == "base")
    near = next(j for j in jobs if j.condition == "near")
    runner = make_runner(tmp_path, FakeLauncher())
    runner.run([base])
    assert runner.ledger.state_of(base.job_id)["status"] == TRAINED
    with pytest.raises(RuntimeError, match="TRAINED-only"):
        runner.plan(near, runner.ledger.states())


@pytest.mark.parametrize(
    ("model", "accuracy", "passes"),
    (("gpt", 0.20, True), ("nextlat", 0.89, False), ("bst", 0.89, False),
     ("nextlat", 0.90, True), ("bst", 0.95, True)),
)
def test_competence_threshold_applies_to_nextlat_and_bst_but_not_gpt(
    tmp_path, model, accuracy, passes
):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("trainer:\n  train_batches: 10\n")
    jobs = build_matrix(
        tmp_path / "root", models=(model,), seeds=(1234,),
        config_for=lambda m, p, c: str(cfg), base_steps=10, adapt_steps=5,
    )
    base = next(j for j in jobs if j.phase == "base")
    near = next(j for j in jobs if j.condition == "near")
    runner = make_runner(tmp_path, FakeLauncher())
    runner.run([base])
    _attach_competence_fixture(runner.ledger, base, accuracy=accuracy)
    if passes:
        plan = runner.plan(near, runner.ledger.states())
        assert plan.parent_checkpoint_sha256
    else:
        with pytest.raises(RuntimeError, match="below the preregistered 0.90"):
            runner.plan(near, runner.ledger.states())


def test_competence_receipt_rejects_tampering_and_identity_mismatch(tmp_path, matrix):
    jobs, _ = matrix
    base = next(j for j in jobs if j.phase == "base")
    near = next(j for j in jobs if j.condition == "near")
    runner = make_runner(tmp_path, FakeLauncher())
    runner.run([base])
    done = _attach_competence_fixture(runner.ledger, base)

    receipt_path = pathlib.Path(base.out_root) / COMPETENCE_RECEIPT
    original = receipt_path.read_text()
    receipt_path.write_text(original + " ")
    with pytest.raises(RuntimeError, match="tampered competence artifact"):
        runner.plan(near, runner.ledger.states())

    receipt_path.write_text(original)
    receipt = json.loads(original)
    receipt["seed"] = 1235
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    receipt_sha = sha256_file(receipt_path)
    sidecar = pathlib.Path(base.out_root) / COMPETENCE_RECEIPT_SIDECAR
    sidecar.write_text(f"{receipt_sha}  {receipt_path.name}\n")
    rebound = dict(done)
    rebound["artifacts"] = dict(done["artifacts"])
    rebound["artifacts"][COMPETENCE_RECEIPT] = receipt_sha
    rebound["artifacts"][COMPETENCE_RECEIPT_SIDECAR] = sha256_file(sidecar)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        verify_base_competence_receipt(
            rebound, expected_job_id=base.job_id, model=base.model, seed=base.seed
        )


def test_competence_receipt_rejects_evaluator_or_checkpoint_hash_mismatch(tmp_path, matrix):
    jobs, _ = matrix
    base = next(j for j in jobs if j.phase == "base")
    runner = make_runner(tmp_path, FakeLauncher())
    runner.run([base])
    done = _attach_competence_fixture(runner.ledger, base)
    receipt_path = pathlib.Path(base.out_root) / COMPETENCE_RECEIPT
    receipt = json.loads(receipt_path.read_text())

    evaluator = pathlib.Path(receipt["evaluator"]["path"])
    evaluator.write_text("# changed evaluator\n")
    with pytest.raises(RuntimeError, match="evaluator.*SHA"):
        verify_base_competence_receipt(
            done, expected_job_id=base.job_id, model=base.model, seed=base.seed
        )

    evaluator.write_text("# deterministic test-only exact-path evaluator\n")
    checkpoint = pathlib.Path(done["final_checkpoint"])
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="checkpoint is missing or no longer matches"):
        verify_base_competence_receipt(
            done, expected_job_id=base.job_id, model=base.model, seed=base.seed
        )


def _production_competence_inputs(tmp_path, base, parent, *, decoding=None):
    assert base.competence_evaluator and base.competence_dataset
    evaluator = pathlib.Path(base.competence_evaluator)
    dataset = pathlib.Path(base.competence_dataset)
    manifest = pathlib.Path(base.competence_manifests[0])
    output = tmp_path / "exact_path_metrics.json"
    output.write_text(json.dumps({
        "schema": "nextlat_forgetting/exact_path_evaluation/1",
        "job_id": base.job_id,
        "model": base.model,
        "seed": base.seed,
        "checkpoint_sha256": parent["final_checkpoint_sha256"],
        "dataset_sha256": sha256_file(dataset),
        "evaluator_sha256": sha256_file(evaluator),
        "manifest_sha256s": [sha256_file(manifest)],
        "decoding": COMPETENCE_DECODING if decoding is None else decoding,
        "exact_path_accuracy": {"correct": 95, "total": 100, "value": 0.95},
    }, sort_keys=True) + "\n")
    return output, evaluator, dataset, manifest


def test_production_materializer_atomically_promotes_trained_parent_without_relaunch(
    tmp_path, matrix, capsys
):
    jobs, _ = matrix
    base = next(j for j in jobs if j.phase == "base")
    ledger = Ledger(tmp_path / "run_ledger.json")
    launcher = FakeLauncher()
    MatrixRunner(ledger, launcher, serializer=SER, echo=lambda m: None).run_job(
        base, ledger.states()
    )
    calls = len(launcher.calls)
    parent = ledger.state_of(base.job_id)
    output, evaluator, dataset, manifest = _production_competence_inputs(
        tmp_path, base, parent
    )
    assert competence_main([
        "--ledger", str(ledger.path),
        "--job-id", base.job_id,
        "--evaluator-output", str(output),
        "--evaluator", str(evaluator),
        "--dataset", str(dataset),
        "--manifest", str(manifest),
    ]) == 0
    receipt = json.loads(capsys.readouterr().out)
    done = ledger.state_of(base.job_id)
    assert len(launcher.calls) == calls
    assert done["status"] == DONE
    assert receipt["decoding"] == COMPETENCE_DECODING
    assert receipt["evaluation_dataset"]["sha256"] == sha256_file(dataset)
    assert receipt["manifests"][0]["sha256"] == sha256_file(manifest)
    assert not list(pathlib.Path(base.out_root).rglob("*.partial"))
    verify_base_competence_receipt(
        done, expected_job_id=base.job_id, model=base.model, seed=base.seed
    )


@pytest.mark.parametrize("wrong", ("dataset", "decoding"))
def test_production_materializer_fails_closed_on_wrong_data_or_decoding(
    tmp_path, matrix, wrong
):
    jobs, _ = matrix
    base = next(j for j in jobs if j.phase == "base")
    ledger = Ledger(tmp_path / "run_ledger.json")
    MatrixRunner(ledger, FakeLauncher(), serializer=SER, echo=lambda m: None).run_job(
        base, ledger.states()
    )
    parent = ledger.state_of(base.job_id)
    decoding = (
        {"strategy": "sample", "top_k": 50, "temperature": 1.0}
        if wrong == "decoding" else None
    )
    output, evaluator, dataset, manifest = _production_competence_inputs(
        tmp_path, base, parent, decoding=decoding
    )
    if wrong == "dataset":
        dataset.write_text("different held-out corpus\n")
    with pytest.raises(RuntimeError, match="dataset SHA|dataset_sha256|held-out dataset|decoding"):
        materialize_base_competence(
            ledger_path=ledger.path,
            job_id=base.job_id,
            evaluator_output_path=output,
            evaluator_path=evaluator,
            dataset_path=dataset,
            manifest_paths=[manifest],
        )
    assert ledger.state_of(base.job_id)["status"] == TRAINED


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
    seeds `self.step` from it (`core_train.py:309`). Under the guarded `>=` stop-rule patch,
    exactly 500 updates therefore require the absolute target 20,500.
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
        "500 is a relative count, not the absolute target for a 20,000-step parent"
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


def test_near_mid_and_far_do_not_share_a_config(tmp_path):
    """Each frozen H3 intervention has its own bank-bound config."""
    jobs = {j.job_id: j for j in build_matrix(tmp_path / "r", models=("gpt",), seeds=(1234,))}
    near = jobs["gpt-s1234-adapt-near"]
    mid = jobs["gpt-s1234-adapt-mid"]
    far = jobs["gpt-s1234-adapt-far"]
    assert len({near.config, mid.config, far.config}) == 3
    assert pathlib.Path(near.config).name == "adapt_near.yaml"
    assert pathlib.Path(mid.config).name == "adapt_mid.yaml"
    assert pathlib.Path(far.config).name == "adapt_far.yaml"


def test_gpt_adaptation_flips_use_nextlat_off(tmp_path):
    """configs/adapt_*.yaml are derived from the NextLat YAML and set `use_nextlat: true`."""
    jobs = {j.job_id: j for j in build_matrix(tmp_path / "r", seeds=(1234,))}
    assert "use_nextlat=false" in jobs["gpt-s1234-adapt-near"].overrides
    assert "use_nextlat=false" in jobs["gpt-s1234-adapt-mid"].overrides
    assert "use_nextlat=false" in jobs["gpt-s1234-adapt-far"].overrides
    assert jobs["nextlat-s1234-adapt-near"].overrides == ()
    cmd = FabricLauncher(tmp_path, dry_run=True).command(
        ResumePlan(spec=jobs["gpt-s1234-adapt-near"], fresh=True,
                   parent_checkpoint="/d/p.pt", parent_steps=20000))
    assert "use_nextlat=false" in cmd


def test_bst_adaptation_selects_architecture_without_dense_objective_knobs(tmp_path):
    """Runtime owns the common CE estimand; the dense BST pair-gap knob is ineligible."""
    jobs = {j.job_id: j for j in build_matrix(tmp_path / "r", seeds=(1234,))}
    for cond in ("near", "mid", "far"):
        ov = jobs[f"bst-s1234-adapt-{cond}"].overrides
        assert "use_bst=true" in ov
        assert "use_nextlat=false" in ov
        assert not any(value.startswith("model.bst_pair_") for value in ov)

    cmd = FabricLauncher(tmp_path, dry_run=True).command(
        ResumePlan(spec=jobs["bst-s1234-adapt-near"], fresh=True,
                   parent_checkpoint="/d/p.pt", parent_steps=20000))
    for override in ("use_bst=true", "use_nextlat=false"):
        assert override in cmd
    assert not any(value.startswith("model.bst_pair_") for value in cmd)
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


def _write_hashed(path: pathlib.Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    digest = sha256_file(path)
    pathlib.Path(f"{path}.sha256").write_text(f"{digest}  {path.name}\n")
    return digest


def _complete_adaptation_gate(tmp_path: pathlib.Path) -> pathlib.Path:
    source_dir = tmp_path / "sources"
    pilot = {
        "pilot": {
            "role": "non_confirmatory",
            "checkpoint_sha256": "a" * 64,
            "loss_table_sha256": "b" * 64,
            "selector_code_sha256": "c" * 64,
            "frozen_before_confirmatory": True,
            "inspected_confirmatory_checkpoints": False,
            "inspected_confirmatory_results": False,
            "optimized_h3_outcomes": False,
        }
    }
    acquisition = {
        "frozen_before_confirmatory": True,
        "inspected_confirmatory_checkpoints": False,
        "inspected_confirmatory_results": False,
        "optimized_h3_outcomes": False,
        "disjoint_from_training": True,
        "matched_target_path_distribution": True,
        "matched_pilot_loss_deciles": True,
    }
    source_payloads = {
        "near_manifest": "near\n",
        "mid_candidates": "mid candidates\n",
        "mid_selection": json.dumps(pilot) + "\n",
        "far_candidates": "far candidates\n",
        "far_selection": json.dumps(pilot) + "\n",
        "near_validation": "near validation\n",
        "mid_validation": "mid validation\n",
        "far_validation": "far validation\n",
        "acquisition_provenance": json.dumps(acquisition) + "\n",
    }
    sources = {}
    for label, payload in source_payloads.items():
        path = source_dir / f"{label}.jsonl"
        digest = _write_hashed(path, payload)
        sources[label] = {"path": str(path), "sha256": digest}

    adapt_dir = tmp_path / "adapt"
    outputs = {}
    for name in (
        "graph_5_5_bnear_5000.txt", "graph_5_5_bmid_5000.txt", "graph_5_5_bfar_5000.txt",
        "graph_5_5_bnearval_2000.txt", "graph_5_5_bmidval_2000.txt",
        "graph_5_5_bfarval_2000.txt",
    ):
        outputs[name] = _write_hashed(adapt_dir / name, f"{name}\n")
    receipt = {
        "schema_version": 1,
        "status": "materialized",
        "scientific_selection_performed": False,
        "sources": sources,
        "outputs": outputs,
    }
    _write_hashed(adapt_dir / "adaptation_banks.json", json.dumps(receipt) + "\n")
    return adapt_dir


def test_adaptation_gate_refuses_missing_or_near_only_receipts(tmp_path):
    with pytest.raises(RuntimeError, match="lacks file or sidecar"):
        verified_adaptation_manifests(tmp_path / "missing")

    adapt_dir = _complete_adaptation_gate(tmp_path)
    receipt_path = adapt_dir / "adaptation_banks.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["outputs"].pop("graph_5_5_bfar_5000.txt")
    _write_hashed(receipt_path, json.dumps(receipt) + "\n")
    with pytest.raises(RuntimeError, match="receipt is incomplete"):
        verified_adaptation_manifests(adapt_dir)


def test_complete_adaptation_gate_binds_all_banks_sidecars_and_pilot_inputs(tmp_path):
    adapt_dir = _complete_adaptation_gate(tmp_path)
    paths = tuple(pathlib.Path(p) for p in verified_adaptation_manifests(adapt_dir))
    assert len(paths) == 2 + 2 * 6 + 2 * 9
    assert all(path.is_file() for path in paths)
    assert sum(path.name.endswith(".sha256") for path in paths) == 16

    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("trainer:\n  train_batches: 50\n")
    spec = JobSpec(
        "gpt-s1234-adapt-near", "gpt", 1234, "adapt", "near", str(cfg),
        str(tmp_path / "run"), manifests=tuple(str(path) for path in paths),
    )
    identity = make_runner(tmp_path, FakeLauncher())._identity(spec)
    assert set(identity["manifest_sha256"]) == {str(path) for path in paths}

    bank = adapt_dir / "graph_5_5_bfar_5000.txt"
    bank.write_text(bank.read_text() + "tampered\n")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verified_adaptation_manifests(adapt_dir)


def test_cli_permits_only_reduced_base_phase_after_permanent_h3_block(tmp_path, capsys):
    common = [
        "--root", str(tmp_path / "runs"), "--models", "gpt", "--seeds", "1234",
        "--print-plan", "--adaptation-manifest-dir", str(tmp_path / "missing"),
    ]
    assert main([*common, "--phase", "base"]) == 0
    capsys.readouterr()
    assert main([*common, "--phase", "adapt"]) == 2
    assert main([*common, "--phase", "all"]) == 2
    assert "permanently excluded Lure-Star H3" in capsys.readouterr().err


def test_cli_base_plan_binds_exact_permanent_h3_exclusion(tmp_path, capsys):
    args = [
        "--root", str(tmp_path / "runs"), "--models", "gpt", "--seeds", "1234",
        "--print-plan", "--phase", "base",
    ]

    assert main(args) == 0
    plan = json.loads(capsys.readouterr().out)
    identity_paths = set(plan[0]["manifests"])
    block, sidecar = verified_h3_permanent_block()
    assert {block, sidecar} <= identity_paths
    assert not any("adaptation_banks" in path for path in identity_paths)


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


def test_base_identity_freezes_evaluator_heldout_dataset_and_evaluation_manifests(
    tmp_path, matrix
):
    base = next(job for job in matrix[0] if job.phase == "base")
    runner = make_runner(tmp_path, FakeLauncher())
    frozen = runner._identity(base)
    competence = frozen["competence_identity"]
    assert competence["evaluator"]["sha256"] == sha256_file(base.competence_evaluator)
    assert competence["dataset"]["sha256"] == sha256_file(base.competence_dataset)
    assert competence["manifests"] == [{
        "path": str(pathlib.Path(base.competence_manifests[0]).resolve()),
        "sha256": sha256_file(base.competence_manifests[0]),
    }]

    evaluator = pathlib.Path(base.competence_evaluator)
    original = evaluator.read_text()
    evaluator.write_text(original + "# post-hoc change\n")
    try:
        with pytest.raises(RuntimeError, match="competence_identity changed"):
            runner._check_identity(base, frozen)
    finally:
        evaluator.write_text(original)


def test_materializer_refuses_if_any_parent_training_artifact_changed(tmp_path, matrix):
    base = next(job for job in matrix[0] if job.phase == "base")
    ledger = Ledger(tmp_path / "run_ledger.json")
    MatrixRunner(ledger, FakeLauncher(), serializer=SER, echo=lambda m: None).run_job(
        base, ledger.states()
    )
    parent = ledger.state_of(base.job_id)
    output, evaluator, dataset, manifest = _production_competence_inputs(
        tmp_path, base, parent
    )
    metrics_rel = next(rel for rel in parent["artifacts"] if rel.endswith("metrics.csv"))
    (pathlib.Path(parent["out_root"]) / metrics_rel).write_text("tampered\n")
    with pytest.raises(RuntimeError, match="training artifact verification failed"):
        materialize_base_competence(
            ledger_path=ledger.path, job_id=base.job_id,
            evaluator_output_path=output, evaluator_path=evaluator,
            dataset_path=dataset, manifest_paths=[manifest],
        )
    assert ledger.state_of(base.job_id)["status"] == TRAINED


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


# --------------------------------------------------------------------------------------
# D-18: the directory upstream actually creates
# --------------------------------------------------------------------------------------

def test_the_runner_predicts_the_seed_suffixed_checkpoint_directory(tmp_path):
    """D-18. `train.py:98-99` renames the experiment before any path is built::

        if "seed" not in experiment_name:
            experiment_name = experiment_name + f"-seed{config.seed}"

    then overwrites `config.trainer.experiment_name` (train.py:125) and joins it onto
    `trainer.out_dir` for every checkpoint (core_train.py:933,959). `bst-s1236-base` does
    not contain the substring "seed" -- `s1236` is not `seed` -- so the real directory is
    `bst-s1236-base-seed1236`. A runner that looks in `{out_root}/{job_id}/` finds an empty
    path, `resolve()` returns None, and `run_job` writes "job exited 0 but left no verified
    checkpoint": a finished 20,000-step run recorded FAILED.
    """
    assert upstream_experiment_dir_name("gpt-s1234-base", 1234) == "gpt-s1234-base-seed1234"
    assert upstream_experiment_dir_name("bst-s1236-adapt-far", 1236) == \
        "bst-s1236-adapt-far-seed1236"
    # a name that already carries "seed" is left alone, exactly as upstream leaves it alone
    assert upstream_experiment_dir_name("nextlat-seed1234-base", 1234) == \
        "nextlat-seed1234-base"

    for job in build_matrix(tmp_path / "r"):
        assert "seed" not in job.experiment_name, job.experiment_name
        assert job.experiment_dir_name == f"{job.job_id}-seed{job.seed}"
        assert job.checkpoint_dir == str(
            pathlib.Path(job.out_root) / f"{job.job_id}-seed{job.seed}")
        assert job.to_dict()["checkpoint_dir"] == job.checkpoint_dir


@pytest.mark.parametrize("model", MODELS)
def test_a_finished_run_in_the_real_directory_is_found_for_every_arm(tmp_path, model):
    """The whole matrix, not just GPT, must be hashed at the path upstream wrote to."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("trainer:\n  train_batches: 20000\n")
    out = tmp_path / "runs" / model / "1234" / "base"
    spec = JobSpec(job_id(model, 1234, "base"), model, 1234, "base", None, str(cfg),
                   str(out), train_batches=20000)

    exp = pathlib.Path(spec.checkpoint_dir)
    exp.mkdir(parents=True)
    assert exp.name.endswith("-seed1234")
    save, _ = SER
    with open(exp / "recovery_ckpt_iter_15000.pt", "wb") as fh:
        save({"training_steps": 15000}, fh)
    (out / "recovery_ckpt").write_text(str(exp / "recovery_ckpt_iter_15000.pt"))

    plan = make_runner(tmp_path, FakeLauncher()).plan(spec, {})
    assert not plan.fresh and plan.resume_step == 15000


def test_checkpoints_under_the_unsuffixed_job_id_are_not_mistaken_for_the_run(tmp_path):
    """Negative control for D-18: the unsuffixed directory is not where upstream writes."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("trainer:\n  train_batches: 20000\n")
    out = tmp_path / "runs" / "bst" / "1236" / "base"
    spec = JobSpec("bst-s1236-base", "bst", 1236, "base", None, str(cfg), str(out),
                   train_batches=20000)

    wrong = out / "bst-s1236-base"          # what a job-id-only prediction would use
    wrong.mkdir(parents=True)
    save, _ = SER
    with open(wrong / "recovery_ckpt_iter_15000.pt", "wb") as fh:
        save({"training_steps": 15000}, fh)

    plan = make_runner(tmp_path, FakeLauncher()).plan(spec, {})
    assert plan.fresh, "a stray directory must not be adopted as this job's history"
    assert pathlib.Path(spec.checkpoint_dir).name == "bst-s1236-base-seed1236"


# --------------------------------------------------------------------------------------
# three-arm H3 parity
# --------------------------------------------------------------------------------------

def test_near_mid_and_far_share_one_parent_in_every_arm(tmp_path):
    """Three base parents and nine branches all retain per-arm parent identity.

    Run for real through the production `MatrixRunner`, `Ledger` and `DurableCheckpointer`
    over all three arms at one seed: near, mid, and far must record the SAME
    `parent_checkpoint_sha256` as each other and as their own base's final checkpoint, and
    the three arms must not collide on one parent.
    """
    jobs = build_matrix(tmp_path / "root", seeds=(1234,), base_steps=50, adapt_steps=50)
    assert len(jobs) == 12
    launcher = FakeLauncher()
    states = run_with_competence_fixtures(tmp_path, jobs, launcher)
    assert [states[j.job_id]["status"] for j in jobs] == [
        DONE if j.phase == "base" else TRAINED for j in jobs
    ]

    parents = []
    for model in MODELS:
        base = states[job_id(model, 1234, "base")]
        near = states[job_id(model, 1234, "adapt", "near")]
        mid = states[job_id(model, 1234, "adapt", "mid")]
        far = states[job_id(model, 1234, "adapt", "far")]
        assert near["parent_checkpoint_sha256"] == mid["parent_checkpoint_sha256"], model
        assert mid["parent_checkpoint_sha256"] == far["parent_checkpoint_sha256"], model
        assert near["parent_checkpoint_sha256"] == base["final_checkpoint_sha256"], model
        assert sha256_file(base["final_checkpoint"]) == base["final_checkpoint_sha256"]
        parents.append(base["final_checkpoint_sha256"])
    assert len(set(parents)) == 3, "the three arms must not share a base checkpoint"
    assert_branch_parity(states, jobs)


def test_branch_parity_catches_a_crossed_parent_in_the_bst_arm(tmp_path):
    jobs = build_matrix(tmp_path / "r", models=("bst",), seeds=(1236,))
    states = {
        "bst-s1236-adapt-near": {"status": DONE, "parent_checkpoint_sha256": "a" * 64},
        "bst-s1236-adapt-far": {"status": DONE, "parent_checkpoint_sha256": "b" * 64},
    }
    with pytest.raises(RuntimeError, match="do not share a parent checkpoint"):
        assert_branch_parity(states, jobs)
    states["bst-s1236-adapt-far"]["parent_checkpoint_sha256"] = "a" * 64
    assert_branch_parity(states, jobs)


# --------------------------------------------------------------------------------------
# Exact-update completion guard
# --------------------------------------------------------------------------------------

class OffsetBlindLauncher(FakeLauncher):
    """Success-shaped under-run fixture: one update, then a clean exit.

    It represents any transport/target bug that exits 0 with a verified checkpoint but fails
    the exact update contract. The production runtime's guarded ``>=`` patch and absolute
    parent-plus-500 target are tested independently.
    """

    def __call__(self, plan: ResumePlan) -> LaunchResult:
        self.calls.append(plan)
        spec = plan.spec
        step = (plan.parent_steps or 0) + 1
        self.updates[spec.job_id] = 1
        ck = self._ckpt(spec)
        ck.save({"step": step, "job": spec.job_id,
                 "parent": plan.parent_checkpoint_sha256}, step, kind="final")
        return LaunchResult(0, step, "ok")


def _done_base(tmp_path, model, ledger):
    jobs = build_matrix(tmp_path / "root", models=(model,), seeds=(1234,),
                        base_steps=50, adapt_steps=50)
    base = jobs[0]
    MatrixRunner(ledger, FakeLauncher(), serializer=SER, echo=lambda m: None).run_job(
        base, ledger.states())
    assert ledger.state_of(base.job_id)["status"] == TRAINED
    _attach_competence_fixture(
        ledger, base, accuracy=0.20 if model == "gpt" else 0.95
    )
    assert ledger.state_of(base.job_id)["status"] == DONE
    return jobs


def test_exact_target_adaptation_recovery_preserves_parent_offset(tmp_path):
    """A branch target is parent step plus requested updates, including on adoption."""
    ledger = Ledger(tmp_path / "run_ledger.json")
    jobs = _done_base(tmp_path, "nextlat", ledger)
    near = next(job for job in jobs if job.condition == "near")
    parent = ledger.state_of(near.parent_job_id)
    launcher = FakeLauncher()
    runner = MatrixRunner(ledger, launcher, serializer=SER, echo=lambda m: None)
    checkpoint = runner.checkpointer(near).save(
        {
            "step": 100,
            "job": near.job_id,
            "parent": parent["final_checkpoint_sha256"],
        },
        100,
        kind="final",
    )
    FakeLauncher._write_upstream_artifacts(near, 100)

    entry = runner.run_job(near, ledger.states())

    assert launcher.calls == []
    assert entry["status"] == TRAINED
    assert entry["parent_steps"] == 50
    assert entry["step"] == 100
    assert entry["updates"] == 50
    assert entry["final_checkpoint_sha256"] == checkpoint.sha256
    assert entry["recovered_without_launch"] is True


@pytest.mark.parametrize("model", MODELS)
def test_an_adaptation_job_with_no_updates_is_never_marked_done(tmp_path, model):
    """A success-shaped under-run is a failure in every arm.

    The job exits 0 and leaves a verified checkpoint, so process and checkpoint verification
    pass. The only thing wrong
    with it is that no adaptation happened -- and an H3 branch that trained nothing still
    produces an erosion number, which is why this has to be caught here rather than noticed
    later in the analysis.
    """
    ledger = Ledger(tmp_path / "run_ledger.json")
    jobs = _done_base(tmp_path, model, ledger)
    near = next(j for j in jobs if j.condition == "near")

    blind = OffsetBlindLauncher()
    MatrixRunner(ledger, blind, serializer=SER, echo=lambda m: None).run_job(
        near, ledger.states())

    entry = ledger.state_of(near.job_id)
    assert blind.updates[near.job_id] == 1, "the launcher really did exit 0 after one step"
    assert entry["status"] == FAILED, "a one-update adaptation must not be DONE"
    assert entry["updates"] == 1
    assert entry["step"] == 51
    assert "request was exactly 50" in entry["reason"]


@pytest.mark.parametrize("model", MODELS)
def test_an_adaptation_job_that_takes_zero_updates_is_not_marked_done(tmp_path, model):
    """Same guard against the exact-zero shape: `train_batches` left at the parent's step."""
    ledger = Ledger(tmp_path / "run_ledger.json")
    jobs = _done_base(tmp_path, model, ledger)
    far = next(j for j in jobs if j.condition == "far")

    blind = FakeLauncher(ignore_parent_offset=True)
    MatrixRunner(ledger, blind, serializer=SER, echo=lambda m: None).run_job(
        far, ledger.states())

    entry = ledger.state_of(far.job_id)
    assert blind.updates.get(far.job_id, 0) == 0
    assert entry["status"] == FAILED
    assert entry["updates"] == 0
    assert "request was exactly 50" in entry["reason"]


def test_a_success_shaped_overrun_is_rejected(tmp_path, matrix):
    jobs, _ = matrix
    base = jobs[0]

    class OverrunLauncher(FakeLauncher):
        def __call__(self, plan):
            super().__call__(plan)
            step = plan.spec.train_batches + 1
            self._ckpt(plan.spec).save(
                {"step": step, "job": plan.spec.job_id}, step, kind="final"
            )
            return LaunchResult(0, step, "success-shaped overrun")

    ledger = Ledger(tmp_path / "run_ledger.json")
    MatrixRunner(ledger, OverrunLauncher(), serializer=SER, echo=lambda m: None).run_job(
        base, ledger.states()
    )
    entry = ledger.state_of(base.job_id)
    assert entry["status"] == FAILED
    assert entry["updates"] == 51
    assert "request was exactly 50" in entry["reason"]


@pytest.mark.parametrize("model", MODELS)
def test_a_healthy_adaptation_records_its_update_count_in_the_ledger(tmp_path, model):
    """`step` is upstream's offset-carrying counter; `updates` is what PROGRAM.md freezes."""
    ledger = Ledger(tmp_path / "run_ledger.json")
    jobs = _done_base(tmp_path, model, ledger)
    base = jobs[0]
    assert ledger.state_of(base.job_id)["updates"] == 50
    assert ledger.state_of(base.job_id)["parent_steps"] is None

    launcher = FakeLauncher()
    runner = MatrixRunner(ledger, launcher, serializer=SER, echo=lambda m: None)
    for spec in jobs[1:]:
        runner.run_job(spec, ledger.states())
        entry = ledger.state_of(spec.job_id)
        assert entry["status"] == TRAINED, entry.get("reason")
        assert entry["parent_steps"] == 50, "the branch started from the 50-step parent"
        assert entry["step"] == 100, "upstream's counter carries the parent's offset"
        assert entry["updates"] == 50 == spec.train_batches


def test_branch_command_offsets_train_batches_in_every_arm(tmp_path):
    """The exact parent-plus-500 target is emitted for all arms and branches."""
    jobs = {j.job_id: j for j in
            build_matrix(tmp_path / "r", seeds=(1234,), adapt_steps=500)}
    launcher = FabricLauncher(tmp_path, dry_run=True)
    for model in MODELS:
        for cond in ("near", "far"):
            spec = jobs[job_id(model, 1234, "adapt", cond)]
            cmd = launcher.command(ResumePlan(
                spec=spec, fresh=True, parent_checkpoint="/d/base/final.pt",
                parent_steps=20000))
            assert "trainer.train_batches=20500" in cmd
            assert "trainer.train_batches=500" not in cmd, (
                f"{spec.job_id}: relative 500 is not the 20,500 absolute target"
            )
            asked = int(next(c for c in cmd
                             if c.startswith("trainer.train_batches=")).split("=", 1)[1])
            assert asked - 20000 == spec.train_batches == 500
            # and an unknown parent step count is a refusal, never a silent 500
            with pytest.raises(ValueError, match="without the parent's step count"):
                launcher.command(ResumePlan(spec=spec, fresh=True,
                                            parent_checkpoint="/d/base/final.pt"))
