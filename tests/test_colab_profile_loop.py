"""Focused regressions for the hardened Colab profiling launcher."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import tarfile
import base64
import hashlib
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT / "scripts" / "colab_profile_loop.py"
PROFILE_RESUME = PROJECT / "scripts" / "profile_resume.py"


def load_launcher():
    spec = importlib.util.spec_from_file_location("colab_profile_loop_under_test", LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_profile_resume():
    spec = importlib.util.spec_from_file_location("profile_resume_under_test", PROFILE_RESUME)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket = bucket
        self.name = name
        self.metadata = {}
        self.size = None
        self.generation = None
        self.md5_hash = None
        self.crc32c = None

    def upload_from_filename(self, path, if_generation_match=None):
        if if_generation_match == 0 and self.name in self.bucket.payloads:
            raise RuntimeError("precondition failed")
        self.bucket.events.append(("file", self.name))
        self.bucket.payloads[self.name] = Path(path).read_bytes()
        self.bucket.metadata[self.name] = dict(self.metadata or {})
        self.bucket.generations[self.name] = self.bucket.generations.get(self.name, 0) + 1
        self.reload()

    def upload_from_string(self, payload, content_type=None):
        del content_type
        self.bucket.events.append(("string", self.name))
        if isinstance(payload, str):
            payload = payload.encode()
        self.bucket.payloads[self.name] = bytes(payload)
        self.bucket.metadata[self.name] = dict(self.metadata or {})
        self.bucket.generations[self.name] = self.bucket.generations.get(self.name, 0) + 1
        self.reload()

    def download_as_bytes(self):
        if self.name not in self.bucket.payloads:
            raise FileNotFoundError(self.name)
        return self.bucket.payloads[self.name]

    def download_to_filename(self, path):
        Path(path).write_bytes(self.download_as_bytes())

    def reload(self):
        payload = self.bucket.payloads.get(self.name)
        self.size = len(payload) if payload is not None else None
        self.metadata = dict(self.bucket.metadata.get(self.name, self.metadata or {}))
        self.generation = self.bucket.generations.get(self.name)
        if payload is not None:
            self.md5_hash = base64.b64encode(hashlib.md5(payload).digest()).decode("ascii")


class FakeBucket:
    def __init__(self):
        self.payloads = {}
        self.metadata = {}
        self.events = []
        self.generations = {}

    def blob(self, name, generation=None):
        del generation
        return FakeBlob(self, name)


def make_input_identity(launcher, suffix: str = "0") -> dict:
    objects = []
    for index, prefix in enumerate(launcher.INPUT_PREFIXES):
        objects.append({
            "name": f"{prefix}/input-{index}.bin",
            "generation": f"{index + 1}{suffix}",
            "size_bytes": index + 10,
            "md5_base64": f"md5-{index}-{suffix}",
            "crc32c_base64": f"crc-{index}-{suffix}",
        })
    return launcher.build_remote_input_identity(objects)


def make_job(root: Path, launcher, name: str) -> None:
    jobs = root / "gate" / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    steps, warmup = launcher.EXPECTED_GATE_JOBS[name]
    log = jobs / f"{name}.log"
    log.write_text("step output\n")
    probe = jobs / f"{name}.probe.123.json"
    probe.write_text(json.dumps({
        "peak_allocated_bytes": 10,
        "peak_reserved_bytes": 20,
        "exit": "ok",
    }))
    (jobs / f"{name}.job.json").write_text(json.dumps({
        "job": name,
        "returncode": 0,
        "steps": steps,
        "warmup_steps": warmup,
        "log": str(log),
    }))


def test_spec_is_frozen_content_addressed_and_a100_only() -> None:
    launcher = load_launcher()
    digest = "a" * 64
    inputs = make_input_identity(launcher)
    spec = launcher.build_spec(digest, inputs, "gpu-a100-test")

    launcher.validate_spec(spec)
    assert spec["gpu"] == "a100"
    assert spec["precision"] == "bf16-mixed"
    assert spec["source_object"] == f"lurestar/source/project-{digest}.tar.gz"
    assert spec["remote_inputs"] == inputs
    assert spec["input_identity_sha256"] == inputs["identity_sha256"]
    assert spec["nonconfirmatory"] is True
    assert set(spec["gate_jobs"]) == set(launcher.EXPECTED_GATE_JOBS)
    assert set(spec["smoke_jobs"]) == set(launcher.EXPECTED_SMOKE_JOBS)

    drifted = json.loads(json.dumps(spec))
    drifted["gate_jobs"]["lurestar-gpt"]["steps"] = 499
    with pytest.raises(launcher.ProfileError, match="contract hash"):
        launcher.validate_spec(drifted)


def test_package_is_reproducible_and_excludes_secrets_bulk_and_results(tmp_path: Path) -> None:
    launcher = load_launcher()
    for name in ("scripts", "configs", "src"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "keep.txt").write_text(name)
    for name in ("data", "results", "upstream", ".secrets"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "drop.txt").write_text(name)
    (tmp_path / "adc.json").write_text("secret")

    archive = launcher.package_project(tmp_path, tmp_path / ".agent_state" / "src.tar.gz")
    first = archive.read_bytes()
    os.utime(tmp_path / "scripts" / "keep.txt", (2_000_000_000, 2_000_000_000))
    launcher.package_project(tmp_path, archive)
    assert first == archive.read_bytes()
    with tarfile.open(archive) as bundle:
        names = set(bundle.getnames())
    assert "scripts/keep.txt" in names
    assert not any(name.startswith(("data/", "results/", "upstream/", ".secrets/"))
                   for name in names)
    assert "adc.json" not in names


def test_gate_validation_requires_exact_steps_logs_and_peak_vram(tmp_path: Path) -> None:
    launcher = load_launcher()
    make_job(tmp_path, launcher, "lurestar-gpt")
    launcher.validate_gate_group(tmp_path, ("lurestar-gpt",))

    probe = tmp_path / "gate" / "jobs" / "lurestar-gpt.probe.123.json"
    probe.write_text(json.dumps({
        "peak_allocated_bytes": None, "peak_reserved_bytes": 20, "exit": "ok",
    }))
    with pytest.raises(launcher.ProfileError, match="peak VRAM"):
        launcher.validate_gate_group(tmp_path, ("lurestar-gpt",))


def test_gate_allows_diagnostic_probe_but_binds_one_success_to_final_attempt(
        tmp_path: Path) -> None:
    launcher = load_launcher()
    make_job(tmp_path, launcher, "lurestar-bst")
    jobs = tmp_path / "gate" / "jobs"
    ledger = jobs / "lurestar-bst.attempts.json"
    ledger.write_text(json.dumps({
        "schema": "nextlat_forgetting/profile_attempts/1", "job": "lurestar-bst",
        "target_steps": 500, "warmup_steps": 100,
        "attempts": [
            {"attempt": 0, "resume_step": 0, "version_start_index": 0},
            {"attempt": 1, "resume_step": 250, "version_start_index": 1},
        ],
    }))
    manifest_path = jobs / "lurestar-bst.job.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({"attempt_ledger": str(ledger), "attempt": 1})
    manifest_path.write_text(json.dumps(manifest))
    original_probe = jobs / "lurestar-bst.probe.123.json"
    original_probe.write_text(json.dumps({
        "exit": "signal-15", "profile_attempt": 0,
        "peak_allocated_bytes": 10, "peak_reserved_bytes": 20,
    }))
    terminal_probe = jobs / "lurestar-bst.probe.456.json"
    terminal_probe.write_text(json.dumps({
        "exit": "ok", "profile_attempt": 1,
        "peak_allocated_bytes": 10, "peak_reserved_bytes": 20,
    }))
    launcher.validate_gate_group(tmp_path, ("lurestar-bst",))

    prior_success = jobs / "lurestar-bst.probe.789.json"
    prior_success.write_text(json.dumps({
        "exit": "ok", "profile_attempt": 0,
        "peak_allocated_bytes": 10, "peak_reserved_bytes": 20,
    }))
    launcher.validate_gate_group(tmp_path, ("lurestar-bst",))

    duplicate_terminal = jobs / "lurestar-bst.probe.999.json"
    duplicate_terminal.write_text(json.dumps({
        "exit": "ok", "profile_attempt": 1,
        "peak_allocated_bytes": 10, "peak_reserved_bytes": 20,
    }))
    with pytest.raises(launcher.ProfileError, match="final attempt"):
        launcher.validate_gate_group(tmp_path, ("lurestar-bst",))


def test_smoke_resume_validation_is_per_job(tmp_path: Path) -> None:
    launcher = load_launcher()
    jobs = tmp_path / "smoke" / "jobs"
    jobs.mkdir(parents=True)
    log = jobs / "hmm-smoke-gpt.log"
    probe = jobs / "hmm-smoke-gpt.probe.1.json"
    log.write_text("step output\n")
    probe.write_text(json.dumps({"peak_allocated_bytes": 1, "peak_reserved_bytes": 2}))
    (jobs / "hmm-smoke-gpt.json").write_text(json.dumps({
        "job": "hmm-smoke-gpt", "returncode": 0, "steps": 50,
        "log": str(log), "probe": str(probe),
        "materialized_configs": ["config.yaml"], "checkpoints": ["step50.pt"],
    }))

    launcher.validate_smoke_job(tmp_path, "hmm-smoke-gpt")
    with pytest.raises(launcher.ProfileError, match="hmm-smoke-nextlat"):
        launcher.validate_smoke(tmp_path)


def test_summary_refuses_missing_job_or_required_measurement(tmp_path: Path) -> None:
    launcher = load_launcher()
    records = {}
    for name in launcher.EXPECTED_GATE_JOBS:
        records[name] = {
            "returncode": 0,
            "missing_required": [],
            **{field: 1.0 for field in launcher.REQUIRED_PROFILE_FIELDS},
        }
    summary = {"records": records, "projection": {"incomplete_for": []}}
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary))
    assert launcher.validate_profile_summary(path) == summary

    records["hmm-nextlat"]["peak_reserved_gb"] = None
    path.write_text(json.dumps(summary))
    with pytest.raises(launcher.ProfileError, match="peak_reserved_gb"):
        launcher.validate_profile_summary(path)

    records.pop("hmm-nextlat")
    path.write_text(json.dumps(summary))
    with pytest.raises(launcher.ProfileError, match="exact five"):
        launcher.validate_profile_summary(path)


def test_state_is_uploaded_after_artifacts_and_success_terminal_requires_complete(tmp_path: Path) -> None:
    launcher = load_launcher()
    spec = launcher.build_spec("b" * 64, make_input_identity(launcher), "gpu-a100-test")
    root = tmp_path / "profile"
    root.mkdir()
    (root / "raw.log").write_text("valuable partial output\n")
    bucket = FakeBucket()
    durability = launcher.ProfileDurability(bucket, root, spec, logger=lambda *_: None)

    state = durability.sync_once(complete=False)
    assert bucket.events[-1][1].endswith("/state.json")
    assert state["generation"] == 1
    with pytest.raises(launcher.ProfileError, match="complete committed state"):
        durability.publish_terminal(state, success=True)

    complete = durability.sync_once(complete=True)
    terminal = durability.publish_terminal(complete, success=True)
    assert terminal["success"] is True and terminal["complete"] is True
    assert bucket.events[-1][1].endswith("/sessions/gpu-a100-test/terminal.json")
    with pytest.raises(launcher.ProfileError, match="downgrade"):
        durability.sync_once(complete=False)


def test_restore_rejects_corruption_and_round_trips_partial_value(tmp_path: Path) -> None:
    launcher = load_launcher()
    spec = launcher.build_spec("c" * 64, make_input_identity(launcher), "gpu-a100-test")
    source = tmp_path / "first"
    source.mkdir()
    (source / "job.log").write_text("retained compute\n")
    bucket = FakeBucket()
    launcher.ProfileDurability(bucket, source, spec, logger=lambda *_: None).sync_once()

    restored = tmp_path / "second"
    restored.mkdir()
    durability = launcher.ProfileDurability(bucket, restored, spec, logger=lambda *_: None)
    assert durability.restore() == 1
    assert (restored / "job.log").read_text() == "retained compute\n"

    artifact_name = next(name for name in bucket.payloads if name.endswith("job.log"))
    bucket.payloads[artifact_name] = b"corrupt"
    with pytest.raises(launcher.ProfileError, match="failed verification"):
        launcher.ProfileDurability(
            bucket, tmp_path / "third", spec, logger=lambda *_: None).restore()


def test_committed_state_survives_later_interrupted_mutable_file_sync(tmp_path: Path) -> None:
    launcher = load_launcher()
    spec = launcher.build_spec("e" * 64, make_input_identity(launcher), "gpu-a100-test")
    source = tmp_path / "source"
    source.mkdir()
    log = source / "job.log"
    log.write_text("first durable value\n")
    bucket = FakeBucket()
    durability = launcher.ProfileDurability(bucket, source, spec, logger=lambda *_: None)
    first = durability.sync_once()
    state_name = spec["profile_prefix"] + "/state.json"
    first_state_payload = bucket.payloads[state_name]
    first_state_metadata = dict(bucket.metadata[state_name])
    first_remote = first["artifacts"][0]["remote"]

    log.write_text("later value before interrupted state commit\n")
    second = durability.sync_once()
    second_remote = second["artifacts"][0]["remote"]
    assert first_remote != second_remote
    assert bucket.payloads[first_remote] == b"first durable value\n"

    # Simulate loss before the newer pointer becomes authoritative: the older committed state
    # still names immutable bytes and therefore restores exactly.
    bucket.payloads[state_name] = first_state_payload
    bucket.metadata[state_name] = first_state_metadata
    restored = tmp_path / "restored"
    restored.mkdir()
    launcher.ProfileDurability(bucket, restored, spec, logger=lambda *_: None).restore()
    assert (restored / "job.log").read_text() == "first durable value\n"


def test_changing_telemetry_is_deferred_without_blocking_checkpoint_commit(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = load_launcher()
    spec = launcher.build_spec("9" * 64, make_input_identity(launcher), "gpu-a100-test")
    source = tmp_path / "source"
    source.mkdir()
    metrics = source / "metrics.csv"
    metrics.write_text("step,loss\n0,1.0\n")
    first_checkpoint = source / "recovery-step-100.pt"
    first_checkpoint.write_bytes(b"checkpoint-100")
    bucket = FakeBucket()
    durability = launcher.ProfileDurability(bucket, source, spec, logger=lambda *_: None)
    first = durability.sync_once()
    first_metrics = next(
        artifact for artifact in first["artifacts"]
        if artifact["relative_path"] == "metrics.csv"
    )

    second_checkpoint = source / "recovery-step-200.pt"
    second_checkpoint.write_bytes(b"checkpoint-200")
    original_copy = launcher.shutil.copyfileobj
    raced = False

    def append_during_copy(source_stream, destination_stream, length=0):
        nonlocal raced
        original_copy(source_stream, destination_stream, length=length)
        if Path(source_stream.name) == metrics and not raced:
            raced = True
            with open(metrics, "a") as stream:
                stream.write("1,0.9\n")
                stream.flush()
                os.fsync(stream.fileno())

    monkeypatch.setattr(launcher.shutil, "copyfileobj", append_during_copy)
    second = durability.sync_once()

    assert raced is True
    assert second["deferred_live_mutable"] == [{
        "relative_path": "metrics.csv",
        "reason": "changed_during_snapshot",
        "retained_sha256": first_metrics["sha256"],
    }]
    second_by_path = {artifact["relative_path"]: artifact
                      for artifact in second["artifacts"]}
    assert second_by_path["metrics.csv"] == first_metrics
    assert second_by_path["recovery-step-200.pt"]["sha256"] == launcher.sha256_file(
        second_checkpoint)
    assert bucket.events[-1][1].endswith("/state.json")

    restored = tmp_path / "restored"
    restored.mkdir()
    launcher.ProfileDurability(bucket, restored, spec, logger=lambda *_: None).restore()
    assert (restored / "metrics.csv").read_text() == "step,loss\n0,1.0\n"
    assert (restored / "recovery-step-200.pt").read_bytes() == b"checkpoint-200"

    # The terminal transaction must include the now-stable final telemetry, not the retained
    # prefix. The racing hook is one-shot, so the next snapshot captures both metric rows.
    complete = durability.sync_once(complete=True)
    assert complete["deferred_live_mutable"] == []
    final_metrics = next(
        artifact for artifact in complete["artifacts"]
        if artifact["relative_path"] == "metrics.csv"
    )
    assert final_metrics["sha256"] == launcher.sha256_file(metrics)
    assert final_metrics["sha256"] != first_metrics["sha256"]


def test_runtime_loss_restores_exact_checkpoint_and_plans_only_remaining_updates(
        tmp_path: Path) -> None:
    launcher = load_launcher()
    resume = load_profile_resume()
    spec = launcher.build_spec("7" * 64, make_input_identity(launcher), "gpu-a100-test")
    profile = tmp_path / "profile"
    jobs = profile / "gate" / "jobs"
    jobs.mkdir(parents=True)
    out_dir = profile / "gate" / "root" / "runs" / "bst" / "seed1234" / "base"
    experiment = "bst-seed1234-base"

    fresh = resume.plan_attempt(
        jobs_dir=jobs, job="lurestar-bst", out_dir=out_dir,
        experiment=experiment, steps=500, warmup_steps=100)
    assert fresh["resume_step"] == 0 and fresh["attempt"] == 0

    version0 = out_dir / experiment / "version_0"
    version0.mkdir(parents=True)
    (version0 / "metrics.csv").write_text("step,steps_per_sec\n0,1.0\n")
    checkpoint = out_dir / experiment / "recovery_ckpt_iter_250.pt"
    checkpoint.write_bytes(b"exact optimizer state at step 250")
    digest = launcher.sha256_file(checkpoint)
    checkpoint.with_name(checkpoint.name + ".meta.json").write_text(json.dumps({
        "size_bytes": checkpoint.stat().st_size,
        "sha256": digest,
        "training_steps": 250,
    }))
    (out_dir / "recovery_ckpt").write_text(str(checkpoint.resolve()))

    bucket = FakeBucket()
    launcher.ProfileDurability(bucket, profile, spec, logger=lambda *_: None).sync_once()
    archived = tmp_path / "lost-runtime-filesystem"
    profile.rename(archived)
    profile.mkdir()
    restored = launcher.ProfileDurability(bucket, profile, spec, logger=lambda *_: None)
    assert restored.restore() > 0

    replacement = resume.plan_attempt(
        jobs_dir=profile / "gate" / "jobs", job="lurestar-bst",
        out_dir=out_dir, experiment=experiment, steps=500, warmup_steps=100)
    assert replacement["action"] == "run"
    assert replacement["resume_step"] == 250
    assert replacement["attempt"] == 1
    ledger = json.loads((profile / "gate/jobs/lurestar-bst.attempts.json").read_text())
    assert [attempt["resume_step"] for attempt in ledger["attempts"]] == [0, 250]

    drifted = launcher.build_spec("6" * 64, make_input_identity(launcher), "gpu-a100-test")
    drifted["profile_prefix"] = spec["profile_prefix"]
    with pytest.raises(launcher.ProfileError, match="does not match this source contract"):
        launcher.ProfileDurability(
            bucket, tmp_path / "drifted", drifted, logger=lambda *_: None).restore()


def test_completed_profile_job_receipt_skips_relaunch(tmp_path: Path) -> None:
    resume = load_profile_resume()
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    out_dir = tmp_path / "root" / "runs" / "gpt" / "seed1234" / "base"
    experiment = "gpt-seed1234-base"
    checkpoint_dir = out_dir / experiment
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "recovery_ckpt_iter_500.pt"
    checkpoint.write_bytes(b"final exact optimizer state")
    checkpoint.with_name(checkpoint.name + ".meta.json").write_text(json.dumps({
        "size_bytes": checkpoint.stat().st_size,
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "training_steps": 500,
    }))
    (out_dir / "recovery_ckpt").write_text(str(checkpoint.resolve()))
    log = jobs / "lurestar-gpt.log"
    log.write_text("completed\n")
    probe = jobs / "lurestar-gpt.probe.1.json"
    probe.write_text(json.dumps({"exit": "ok", "peak_allocated_bytes": 1,
                                 "peak_reserved_bytes": 2}))
    (jobs / "lurestar-gpt.job.json").write_text(json.dumps({
        "job": "lurestar-gpt", "returncode": 0, "steps": 500,
        "warmup_steps": 100, "log": str(log),
        "probe_glob": str(jobs / "lurestar-gpt.probe.*.json"),
        "out_dir": str(out_dir), "experiment_name": experiment,
    }))

    plan = resume.plan_attempt(
        jobs_dir=jobs, job="lurestar-gpt", out_dir=out_dir,
        experiment=experiment, steps=500, warmup_steps=100)

    assert plan["action"] == "skip"
    assert plan["resume_step"] == 500
    assert not (jobs / "lurestar-gpt.attempts.json").exists()


def test_target_checkpoint_and_probe_repairs_manifest_without_optimizer_replay(
        tmp_path: Path) -> None:
    resume = load_profile_resume()
    jobs = tmp_path / "jobs"
    jobs.mkdir()
    out_dir = tmp_path / "root" / "runs" / "bst" / "seed1234" / "base"
    experiment = "bst-seed1234-base"
    fresh = resume.plan_attempt(
        jobs_dir=jobs, job="lurestar-bst", out_dir=out_dir,
        experiment=experiment, steps=500, warmup_steps=100)
    assert fresh["action"] == "run" and fresh["resume_step"] == 0
    version = out_dir / experiment / "version_0"
    version.mkdir(parents=True)
    (version / "metrics.csv").write_text("step,steps_per_sec\n499,1.0\n")
    checkpoint = out_dir / experiment / "recovery_ckpt_iter_500.pt"
    checkpoint.write_bytes(b"target state")
    checkpoint.with_name(checkpoint.name + ".meta.json").write_text(json.dumps({
        "size_bytes": checkpoint.stat().st_size,
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "training_steps": 500,
    }))
    (out_dir / "recovery_ckpt").write_text(str(checkpoint.resolve()))
    (jobs / "lurestar-bst.probe.123.json").write_text(json.dumps({
        "exit": "ok", "profile_attempt": 0,
        "peak_allocated_bytes": 1, "peak_reserved_bytes": 2,
    }))
    (jobs / "lurestar-bst.log").write_text("terminal training output\n")

    repair = resume.plan_attempt(
        jobs_dir=jobs, job="lurestar-bst", out_dir=out_dir,
        experiment=experiment, steps=500, warmup_steps=100)

    assert repair == {
        "action": "finalize", "resume_step": 500,
        "ledger": str(jobs / "lurestar-bst.attempts.json"), "attempt": 0,
    }


def test_complete_sync_refuses_to_commit_unstable_live_telemetry(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = load_launcher()
    spec = launcher.build_spec("8" * 64, make_input_identity(launcher), "gpu-a100-test")
    source = tmp_path / "source"
    source.mkdir()
    telemetry = source / "active.gpu.csv"
    telemetry.write_text("timestamp,utilization\n")
    bucket = FakeBucket()
    durability = launcher.ProfileDurability(bucket, source, spec, logger=lambda *_: None)
    original_copy = launcher.shutil.copyfileobj

    def append_during_copy(source_stream, destination_stream, length=0):
        original_copy(source_stream, destination_stream, length=length)
        with open(telemetry, "a") as stream:
            stream.write("now,99\n")
            stream.flush()
            os.fsync(stream.fileno())

    monkeypatch.setattr(launcher.shutil, "copyfileobj", append_during_copy)
    with pytest.raises(launcher.ProfileError, match="requires stable live artifacts"):
        durability.sync_once(complete=True)

    assert spec["profile_prefix"] + "/state.json" not in bucket.payloads
    assert durability.complete_committed is False


def test_remote_input_identity_binds_every_prefix_generation_and_hash() -> None:
    launcher = load_launcher()
    inputs = make_input_identity(launcher)
    launcher.validate_remote_input_identity(inputs)
    spec = launcher.build_spec("f" * 64, inputs, "gpu-a100-test")

    drifted_inputs = json.loads(json.dumps(inputs))
    drifted_inputs["objects"][0]["generation"] = "999"
    with pytest.raises(launcher.ProfileError, match="identity/hash mismatch"):
        launcher.build_spec("f" * 64, drifted_inputs, "gpu-a100-test")

    changed = make_input_identity(launcher, "9")
    changed_spec = launcher.build_spec("f" * 64, changed, "gpu-a100-test")
    assert changed_spec["profile_id"] != spec["profile_id"]
    assert changed_spec["contract_sha256"] != spec["contract_sha256"]


def test_restore_refuses_state_from_different_remote_input_inventory(tmp_path: Path) -> None:
    launcher = load_launcher()
    source = tmp_path / "source"
    source.mkdir()
    (source / "value.txt").write_text("valuable\n")
    bucket = FakeBucket()
    first_spec = launcher.build_spec(
        "1" * 64, make_input_identity(launcher, "1"), "gpu-a100-test")
    launcher.ProfileDurability(
        bucket, source, first_spec, logger=lambda *_: None).sync_once()

    second_spec = launcher.build_spec(
        "1" * 64, make_input_identity(launcher, "2"), "gpu-a100-test")
    # Force lookup of the old pointer to directly exercise the restore identity check.
    second_spec["profile_prefix"] = first_spec["profile_prefix"]
    with pytest.raises(launcher.ProfileError, match="does not match this source contract"):
        launcher.ProfileDurability(
            bucket, tmp_path / "restore", second_spec, logger=lambda *_: None).restore()


def test_explicit_audited_salvage_can_promote_orphans_but_default_cannot(
        tmp_path: Path) -> None:
    launcher = load_launcher()
    inputs = make_input_identity(launcher)
    old_spec = launcher.build_spec("1" * 64, inputs, "gpu-a100-old")
    old_root = tmp_path / "old"
    old_root.mkdir()
    (old_root / "valuable.pt").write_bytes(b"audited orphan")
    bucket = FakeBucket()
    old_state = launcher.ProfileDurability(
        bucket, old_root, old_spec, logger=lambda *_: None).sync_once()

    target_sha = "2" * 64
    receipt = {
        "schema": launcher.SALVAGE_SCHEMA,
        "authorization": "GO",
        "source_sha256": old_spec["source_sha256"],
        "source_profile_id": old_spec["profile_id"],
        "target_source_sha256": target_sha,
        "input_identity_sha256": inputs["identity_sha256"],
        "remote_inputs": inputs,
        "training_compatibility": {
            "verdict": "BYTE_IDENTICAL",
            "source_sha256": old_spec["source_sha256"],
            "target_source_sha256": target_sha,
            "compared_surface_sha256": "3" * 64,
        },
        "audit": {"auditor": "independent-read-only-agent",
                  "audited_at": "2026-08-24T12:00:00Z", "evidence_sha256": "4" * 64},
        "artifacts": old_state["artifacts"],
        "artifact_fingerprint": launcher.canonical_sha256({
            "artifacts": old_state["artifacts"]}),
        "completed_jobs": [],
        "resume_steps": {"lurestar-bst": 250},
    }
    target_spec = launcher.build_spec(
        target_sha, inputs, "gpu-a100-new", salvage_receipt=receipt)
    target_root = tmp_path / "target"
    target_root.mkdir()
    durability = launcher.ProfileDurability(
        bucket, target_root, target_spec, logger=lambda *_: None)

    assert durability.restore() == 0
    assert not (target_root / "valuable.pt").exists()
    assert durability.restore(salvage_receipt=receipt) == 1
    assert (target_root / "valuable.pt").read_bytes() == b"audited orphan"

    tampered = json.loads(json.dumps(receipt))
    tampered["resume_steps"]["lurestar-bst"] = 501
    with pytest.raises(launcher.ProfileError, match="resume step"):
        launcher.build_spec(target_sha, inputs, salvage_receipt=tampered)


def test_salvage_materializes_attempt_zero_boundary_for_segmented_summary(
        tmp_path: Path) -> None:
    launcher = load_launcher()
    output = tmp_path / "profile"
    metrics = (output / "gate/root/runs/bst/seed1234/base/bst-seed1234-base/"
               "version_0/metrics.csv")
    metrics.parent.mkdir(parents=True)
    metrics.write_text("step,steps_per_sec\n100,1.0\n")
    receipt = {"resume_steps": {"lurestar-bst": 250}}

    launcher.materialize_salvage_attempt_ledgers(output, receipt)

    ledger = json.loads((output / "gate/jobs/lurestar-bst.attempts.json").read_text())
    assert ledger["attempts"] == [
        {"attempt": 0, "resume_step": 0, "version_start_index": 0}]
    assert ledger["salvage_boundary"] == 250


def test_teardown_is_owned_scoped_and_requires_two_zero_burn_quota_reads() -> None:
    launcher = load_launcher()
    teardown_source = inspect.getsource(launcher.teardown_owned_runtime)
    assert '["colab", "stop", "--session", sid]' in teardown_source
    stopped = []
    settled = {"active_runtimes": 0, "burn_rate_hourly": 0.0, "paid_balance": 1000.0}
    result = launcher.teardown_owned_runtime(
        "gpu-a100-owned",
        stopper=lambda sid: stopped.append(sid),
        status_reader=lambda: ({"status": "no_runtime"}, {"status": "no_runtime"}),
        quota_reader=lambda: (settled, settled),
    )
    assert stopped == ["gpu-a100-owned"]
    assert result == settled

    with pytest.raises(launcher.ProfileError, match="zero runtime/burn twice"):
        launcher.teardown_owned_runtime(
            "gpu-a100-owned",
            stopper=lambda _sid: None,
            status_reader=lambda: ({"status": "no_runtime"}, {"status": "no_runtime"}),
            quota_reader=lambda: (settled, {**settled, "burn_rate_hourly": 5.3}),
        )


def test_teardown_fails_closed_when_runtime_does_not_disappear() -> None:
    launcher = load_launcher()
    calls = []
    with pytest.raises(launcher.ProfileError, match="two-read no-runtime"):
        launcher.teardown_owned_runtime(
            "gpu-a100-owned",
            stopper=lambda sid: calls.append(sid),
            status_reader=lambda: ({"status": "connected"}, {"status": "connected"}),
            quota_reader=lambda: pytest.fail("quota must not be accepted before stopped status"),
        )
    assert calls == ["gpu-a100-owned", "gpu-a100-owned"]


def test_monitor_preserves_advancing_runtime_and_stops_on_terminal() -> None:
    launcher = load_launcher()
    spec = launcher.build_spec("d" * 64, make_input_identity(launcher), "gpu-a100-test")
    generations = iter([0, 1, 1])
    terminals = iter([None, None, {"success": True, "complete": True}])

    outcome = launcher.monitor_owned_runtime(
        spec,
        status_reader=lambda: ({"status": "connected"}, {"status": "connected"}),
        generation_reader=lambda _spec: next(generations),
        terminal_reader=lambda _spec: next(terminals),
        stall_windows=3,
    )

    assert outcome["reason"] == "terminal"
    assert outcome["terminal"]["success"] is True


def test_runtime_path_uses_python_gcs_and_invokes_profile_sh() -> None:
    launcher = load_launcher()
    source = inspect.getsource(launcher.runtime_driver)

    assert "google.cloud import storage" in source
    assert "storage.Client(project=GCP_PROJECT)" in source
    assert "gcloud" not in source
    assert '"profile.sh"' in source
    assert '"--lurestar-only"' in source
    assert '"--hmm-only"' in source
    assert "trainer.train_batches=50" in inspect.getsource(launcher.run_smoke_job)
    assert "PROFILE_COMPLETE=True" in source


def test_status_disagreement_is_uncertain() -> None:
    launcher = load_launcher()
    assert launcher.agreed_runtime_state(
        {"status": "no_runtime"}, {"status": "connected"}) == "uncertain"
    assert launcher.agreed_runtime_state(
        {"status": "no_runtime"}, {"status": "no_runtime"}) == "gone"
    assert launcher.agreed_runtime_state(
        {"status": "connected"}, {"status": "connected"}) == "active"
