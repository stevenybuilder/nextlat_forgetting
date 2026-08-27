import importlib.util
import io
import json
import os
import pathlib
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import types

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts" / "colab_recovery_gate.py"
SPEC = importlib.util.spec_from_file_location("colab_recovery_gate", PATH)
gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gate
SPEC.loader.exec_module(gate)


class _MemoryBucket:
    """Small create-only GCS model used to replay immutable commits."""

    def __init__(self):
        self.objects = {}
        self.calls = []

    def blob(self, name):
        bucket = self

        class Blob:
            def _put(self, payload, kwargs):
                bucket.calls.append((name, bytes(payload), dict(kwargs)))
                if kwargs.get("if_generation_match") == 0 and name in bucket.objects:
                    raise RuntimeError("412 conditionNotMet")
                bucket.objects[name] = bytes(payload)

            def upload_from_filename(self, filename, **kwargs):
                self._put(pathlib.Path(filename).read_bytes(), kwargs)

            def upload_from_string(self, payload, **kwargs):
                if isinstance(payload, str):
                    payload = payload.encode()
                self._put(payload, kwargs)

            def download_to_filename(self, filename):
                pathlib.Path(filename).write_bytes(bucket.objects[name])

            def download_as_text(self):
                return bucket.objects[name].decode()

        return Blob()


def _install_fake_checkpoint_verifier(monkeypatch, states):
    module = types.ModuleType("lurestar_runtime")

    def verify_checkpoint(path, *, deserialize, require_metadata):
        assert deserialize is True and require_metadata is True
        path = pathlib.Path(path)
        state = states.get(path.name, states.get("*"))
        if state is None:
            raise AssertionError("unexpected checkpoint %s" % path)
        return {"sha256": gate.sha256_file(path)}, state

    module.verify_checkpoint = verify_checkpoint
    monkeypatch.setitem(sys.modules, "lurestar_runtime", module)


def test_preregistration_is_nonconfirmatory_and_frozen():
    digest = "a" * 64
    spec = gate.build_spec(digest, "rg-%s-1234567890-deadbeef" % digest[:12])
    assert spec["gpu"] == "t4"
    assert spec["precision"] == "16-mixed"
    assert spec["seed"] not in gate.CONFIRMATORY_SEEDS
    assert spec["train_steps"] == 300
    assert spec["interrupt_step"] == 150
    assert spec["checkpoint_every"] == 50
    assert spec["data"]["first_generator_seed"] == 1_000_000
    assert spec["nonconfirmatory"] is True
    gate.validate_spec(spec)


@pytest.mark.parametrize("field,value", [
    ("seed", 1234), ("gpu", "a100"), ("precision", "bf16-mixed"),
    ("train_steps", 301), ("interrupt_step", 149),
])
def test_sidecar_science_drift_fails_closed(field, value):
    digest = "b" * 64
    spec = gate.build_spec(digest, "rg-%s-1234567890-feedface" % digest[:12])
    spec[field] = value
    with pytest.raises(gate.GateError):
        gate.validate_spec(spec)


def test_content_addressed_source_is_required():
    digest = "c" * 64
    spec = gate.build_spec(digest, "rg-%s-1234567890-1234abcd" % digest[:12])
    spec["source_object"] = "lurestar/source/project.tar.gz"
    spec.pop("preregistration_sha256")
    with pytest.raises(gate.GateError, match="content-addressed"):
        gate.validate_spec(spec)


def test_preregistration_hash_detects_tampering():
    digest = "d" * 64
    spec = gate.build_spec(digest, "rg-%s-1234567890-abcd1234" % digest[:12])
    spec["data"]["first_generator_seed"] += 1
    with pytest.raises(gate.GateError):
        gate.validate_spec(spec)


def test_receipt_is_append_only_jsonl(tmp_path):
    path = tmp_path / "receipt.jsonl"
    gate.append_receipt(path, {"event": "one"})
    gate.append_receipt(path, {"event": "two"})
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["event"] for row in rows] == ["one", "two"]
    assert all("recorded_at_unix" in row for row in rows)


def test_two_status_reads_must_agree():
    gone = {"status": "no_runtime"}
    active = {"status": "connected"}
    assert gate.agreed_runtime_state(gone, gone) == "gone"
    assert gate.agreed_runtime_state(active, active) == "active"
    assert gate.agreed_runtime_state(gone, active) == "uncertain"


def test_upload_retries_on_same_session_without_reprovisioning(tmp_path):
    calls = []
    outcomes = [(1, "kernel write failed"), (1, "kernel busy"), (0, "uploaded")]

    def runner(argv, *, check=False):
        calls.append((argv, check))
        return outcomes.pop(0)

    attempts = gate.upload_with_retry(
        "gpu-t4-fixed-session", tmp_path / "job.json", "/content/job.json",
        runner=runner,
    )

    assert attempts == 3
    assert len(calls) == 3
    assert all(call[0][3] == "gpu-t4-fixed-session" for call in calls)
    assert all(call[1] is False for call in calls)


def test_recovery_training_command_enables_deterministic_cuda_contract(tmp_path):
    command, env = gate._training_command(
        tmp_path / "project", tmp_path / "upstream", tmp_path / "branch",
        tmp_path / "train.txt", tmp_path / "test.txt", resume=True,
    )
    assert command
    assert env["LURESTAR_DETERMINISTIC_RUNTIME"] == "1"
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert env["NVIDIA_TF32_OVERRIDE"] == "0"


def test_training_observation_latch_survives_bounded_tail_eviction(tmp_path):
    marker = "Fast forwarded data to step 150"
    code = (
        "print(%r); "
        "[print('progress-%%04d' %% i) for i in range(5100)]"
    ) % marker
    observation_path = tmp_path / "observations" / "reference.json"
    captured, observations = gate._finish_training(
        [sys.executable, "-c", code], dict(os.environ),
        observation_path=observation_path,
    )
    assert marker not in captured
    assert "progress-5099" in captured
    assert gate._observed_fast_forward(observations, 150) is True
    assert json.loads(observation_path.read_text()) == {
        "schema": gate.TrainingObservationLatch.SCHEMA,
        "line_count": 1,
        "events": [{"kind": "data_fast_forward", "step": 150, "line_number": 1}],
    }


def test_training_observation_latch_rejects_unstructured_or_wrong_step():
    latch = gate.TrainingObservationLatch()
    latch.observe_line("Fast forwarded data to step not-a-number")
    latch.observe_line("Fast forwarded data to step 149")
    observations = latch.snapshot()
    assert gate._observed_fast_forward(observations, 150) is False
    assert gate._observed_fast_forward({"events": observations["events"]}, 149) is False


def test_upload_retry_fails_closed_after_bound(tmp_path):
    def runner(_argv, *, check=False):
        assert check is False
        return 1, "kernel write failed"

    with pytest.raises(gate.GateError, match="failed 2 times"):
        gate.upload_with_retry(
            "gpu-t4-fixed-session", tmp_path / "job.json", "/content/job.json",
            attempts=2, runner=runner,
        )


def test_parse_cli_json_tolerates_prefix_and_rejects_nonobject():
    assert gate.parse_cli_json('noise\n{"session":"abc"}\n')["session"] == "abc"
    with pytest.raises(gate.GateError):
        gate.parse_cli_json("[]")


def test_archive_is_byte_reproducible_and_excludes_sensitive_trees(tmp_path):
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    (root / "scripts" / "x.py").write_text("x = 1\n")
    (root / ".secrets").mkdir()
    (root / ".secrets" / "adc.json").write_text("secret")
    (root / "data").mkdir()
    (root / "data" / "confirmatory.txt").write_text("forbidden")
    (root / "scripts" / "nested").mkdir()
    (root / "scripts" / "nested" / "adc.json").write_text("secret")
    (root / "scripts" / "nested" / ".env").write_text("secret")
    first, second = tmp_path / "a.tar.gz", tmp_path / "b.tar.gz"
    gate.package_project(root, first)
    gate.package_project(root, second)
    assert gate.sha256_file(first) == gate.sha256_file(second)
    import tarfile
    with tarfile.open(first) as archive:
        names = archive.getnames()
    assert "scripts/x.py" in names
    assert not any("secret" in name or name.startswith("data") for name in names)
    assert not any(name.endswith(("adc.json", ".env")) for name in names)


@pytest.mark.parametrize("member_kind", ["traversal", "symlink", "hardlink"])
def test_safe_extract_rejects_paths_and_links(tmp_path, member_kind):
    archive_path = tmp_path / (member_kind + ".tar")
    with tarfile.open(archive_path, "w") as archive:
        if member_kind == "traversal":
            member = tarfile.TarInfo("../outside.txt")
            payload = b"escape"
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        else:
            member = tarfile.TarInfo("unsafe-link")
            member.type = tarfile.SYMTYPE if member_kind == "symlink" else tarfile.LNKTYPE
            member.linkname = "../outside.txt"
            archive.addfile(member)
    with pytest.raises(gate.GateError, match="traversal|link"):
        gate._safe_extract(archive_path, tmp_path / "extract")
    assert not (tmp_path / "outside.txt").exists()


def test_safe_extract_preserves_regular_file_and_executable_mode(tmp_path):
    archive_path = tmp_path / "safe.tar"
    with tarfile.open(archive_path, "w") as archive:
        directory = tarfile.TarInfo("scripts")
        directory.type = tarfile.DIRTYPE
        archive.addfile(directory)
        member = tarfile.TarInfo("scripts/run.sh")
        payload = b"#!/bin/sh\nexit 0\n"
        member.size = len(payload)
        member.mode = 0o755
        archive.addfile(member, io.BytesIO(payload))
    destination = tmp_path / "extract"
    gate._safe_extract(archive_path, destination)
    extracted = destination / "scripts" / "run.sh"
    assert extracted.read_bytes() == payload
    assert extracted.stat().st_mode & 0o111


def test_kill_tree_catches_nested_workers_in_escaped_process_groups(tmp_path):
    pid_path = tmp_path / "nested-pids.txt"
    grandchild_code = "import time; time.sleep(600)"
    child_code = (
        "import os,pathlib,subprocess,sys,time; "
        "grand=subprocess.Popen([sys.executable,'-c',%r],start_new_session=True); "
        "pathlib.Path(%r).write_text(str(os.getpid())+' '+str(grand.pid)); "
        "time.sleep(600)"
    ) % (grandchild_code, str(pid_path))
    parent_code = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c',%r],start_new_session=True); "
        "time.sleep(600)"
    ) % child_code
    process = subprocess.Popen(
        [sys.executable, "-c", parent_code], start_new_session=True
    )
    recorded = [process.pid]
    try:
        deadline = time.time() + 10
        while time.time() < deadline and not pid_path.is_file():
            time.sleep(0.02)
        assert pid_path.is_file()
        recorded.extend(map(int, pid_path.read_text().split()))
        deadline = time.time() + 10
        before = gate._process_table()
        while time.time() < deadline and not set(recorded).issubset(before):
            time.sleep(0.02)
            before = gate._process_table()
        assert set(recorded).issubset(before)
        assert len({before[pid]["pgid"] for pid in recorded}) == 3
        result = gate._kill_process_tree(process, timeout=10)
        assert result["all_captured_identities_gone"] is True
        assert set(recorded).issubset(result["captured_pids"])
        assert len(result["captured_process_groups"]) >= 3
        after = gate._process_table()
        assert not set(recorded).intersection(after)
    finally:
        for pid in reversed(recorded):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass


def test_progress_snapshot_commits_metrics_before_state_and_event(tmp_path):
    calls = []

    class Blob:
        def __init__(self, name):
            self.name = name

        def upload_from_filename(self, filename, **kwargs):
            calls.append(("file", self.name, pathlib.Path(filename).read_bytes(), kwargs))

        def upload_from_string(self, payload, **kwargs):
            calls.append(("string", self.name, payload, kwargs))

    class Bucket:
        def blob(self, name):
            return Blob(name)

    metric = tmp_path / "active" / "runs" / "version_0" / "metrics.csv"
    metric.parent.mkdir(parents=True)
    metric.write_text("step,loss\n1,2.0\n")
    pointer = tmp_path / "active" / "runs" / "recovery_ckpt"
    pointer.write_text("/content/checkpoint.pt\n")
    spec = {
        "gate_id": "rg-aaaaaaaaaaaa-1234567890-deadbeef",
        "source_sha256": "a" * 64,
        "event_prefix": "events",
    }
    state = gate._publish_progress_snapshot(Bucket(), spec, tmp_path, 7)
    progress_calls = [call for call in calls if "/progress/000007/" in call[1]]
    assert [call[0] for call in progress_calls] == ["string", "string"]
    assert progress_calls[0][1].endswith("/artifacts/active/runs/version_0/metrics.csv")
    assert progress_calls[0][2] == metric.read_bytes()
    assert progress_calls[1][1].endswith("/state.json")
    assert all(call[3]["if_generation_match"] == 0 for call in progress_calls)
    assert calls[-1][1].startswith("events/")
    assert state["checkpoint_pointers"]["active/runs/recovery_ckpt"] == \
        "/content/checkpoint.pt"


def test_resume_commit_is_artifact_first_create_only_and_replay_safe(monkeypatch, tmp_path):
    out_dir = tmp_path / "base"
    checkpoint_dir = out_dir / "experiment"
    checkpoint_dir.mkdir(parents=True)
    checkpoints = []
    for step in (100, 150):
        checkpoint = checkpoint_dir / ("recovery_ckpt_iter_%d.pt" % step)
        checkpoint.write_bytes(("checkpoint-%d" % step).encode())
        checkpoint.with_name(checkpoint.name + ".meta.json").write_text("{}\n")
        checkpoints.append(checkpoint)
    (out_dir / "recovery_ckpt").write_text(str(checkpoints[-1]) + "\n")
    _write_metrics(
        checkpoint_dir / "version_0" / "metrics.csv",
        [{"step": 150, "loss": 0.5, "lr": 0.001, "steps_per_sec": 1}],
    )
    _install_fake_checkpoint_verifier(monkeypatch, {
        "*": {"training_steps": gate.INTERRUPT_STEP,
              "lurestar_rng_state_v1": {"torch": [1, 2, 3]}},
    })
    bucket = _MemoryBucket()
    spec = {
        "gate_id": "rg-aaaaaaaaaaaa-1234567890-deadbeef",
        "source_sha256": "a" * 64,
        "resume_prefix": "recovery/resume",
    }

    state = gate._upload_resume_snapshot(bucket, spec, out_dir, checkpoints[-1])
    state_name = "recovery/resume/state.json"
    assert bucket.calls[-1][0] == state_name
    assert all(call[2].get("if_generation_match") == 0 for call in bucket.calls)
    assert gate._assert_committed_lineage(checkpoints[-1], state) == \
        gate.sha256_file(checkpoints[-1])
    alternate_path = tmp_path / "same-bytes-wrong-path.pt"
    alternate_path.write_bytes(checkpoints[-1].read_bytes())
    with pytest.raises(gate.GateError, match="path differs"):
        gate._assert_committed_lineage(alternate_path, state)
    committed = dict(bucket.objects)

    checkpoints[-1].write_bytes(b"replayed-different-bytes")
    with pytest.raises(RuntimeError, match="412"):
        gate._upload_resume_snapshot(bucket, spec, out_dir, checkpoints[-1])
    assert bucket.objects == committed
    with pytest.raises(gate.GateError, match="bytes differ"):
        gate._assert_committed_lineage(checkpoints[-1], state)


def test_resume_snapshot_round_trip_restores_exact_shared_lineage(monkeypatch, tmp_path):
    gate_id = "rg-roundtrip-%s" % tmp_path.name
    runtime_root = tmp_path / "runtime" / gate_id
    out_dir = runtime_root / "active-lineage" / "runs" / "gpt" / "seed18181" / "base"
    checkpoint_dir = out_dir / "experiment"
    checkpoint_dir.mkdir(parents=True)
    checkpoints = []
    try:
        for step in (100, gate.INTERRUPT_STEP):
            checkpoint = checkpoint_dir / ("recovery_ckpt_iter_%d.pt" % step)
            checkpoint.write_bytes(("checkpoint-payload-%d" % step).encode())
            checkpoint.with_name(checkpoint.name + ".meta.json").write_text(
                json.dumps({"step": step}, sort_keys=True) + "\n"
            )
            checkpoints.append(checkpoint)
        target = checkpoints[-1]
        pointer = out_dir / "recovery_ckpt"
        pointer.write_text(str(target.resolve()) + "\n")
        _write_metrics(
            checkpoint_dir / "version_0" / "metrics.csv",
            [{"step": gate.INTERRUPT_STEP, "loss": 0.5, "lr": 0.001,
              "steps_per_sec": 1}],
        )
        _install_fake_checkpoint_verifier(monkeypatch, {
            "*": {
                "training_steps": gate.INTERRUPT_STEP,
                "lurestar_rng_state_v1": {"torch": [1, 2, 3]},
            },
        })
        bucket = _MemoryBucket()
        spec = {
            "gate_id": gate_id,
            "source_sha256": "a" * 64,
            "resume_prefix": "recovery/%s/resume" % gate_id,
        }

        committed = gate._upload_resume_snapshot(bucket, spec, out_dir, target)
        downloaded = json.loads(
            bucket.blob("%s/state.json" % spec["resume_prefix"]).download_as_text()
        )
        assert downloaded == committed
        expected = {
            artifact["local_path"]: {
                "bytes": pathlib.Path(artifact["local_path"]).read_bytes(),
                "sha256": artifact["sha256"],
                "size_bytes": artifact["size_bytes"],
            }
            for artifact in committed["artifacts"]
        }
        assert committed["checkpoint"] == str(target.resolve())

        # The retained local continuation and the restored continuation must share
        # one immutable, path-bound step-150 parent.
        retained_lineage = gate._assert_committed_lineage(target, committed)
        for local_path in expected:
            pathlib.Path(local_path).unlink()
        assert not target.exists()
        assert not pointer.exists()

        restored = gate._restore_resume_snapshot(
            bucket, spec, downloaded, runtime_root=runtime_root
        )
        assert restored == target.resolve()
        assert set(expected) == {
            str(pathlib.Path(artifact["local_path"]).resolve())
            for artifact in downloaded["artifacts"]
        }
        for local_path, original in expected.items():
            local = pathlib.Path(local_path)
            assert local.is_file()
            assert local.read_bytes() == original["bytes"]
            assert local.stat().st_size == original["size_bytes"]
            assert gate.sha256_file(local) == original["sha256"]
        assert pointer.read_text() == str(target.resolve()) + "\n"
        restored_lineage = gate._assert_committed_lineage(restored, downloaded)
        assert retained_lineage == restored_lineage == committed["checkpoint_sha256"]
    finally:
        shutil.rmtree(runtime_root, ignore_errors=True)


def test_final_commit_is_artifact_first_create_only_and_replay_safe(monkeypatch, tmp_path):
    checkpoint = tmp_path / "ckpt_iter_300.pt"
    checkpoint.write_bytes(b"final-checkpoint")
    checkpoint.with_name(checkpoint.name + ".meta.json").write_text("{}\n")
    payload = {
        "training_steps": gate.TRAIN_STEPS,
        gate.AMP_SCALER_KEY: {"scale": 1024.0},
    }
    _install_fake_checkpoint_verifier(monkeypatch, {"*": payload})
    bucket = _MemoryBucket()
    spec = {"gate_id": "rg-aaaaaaaaaaaa-1234567890-deadbeef",
            "source_sha256": "a" * 64}

    state = gate._publish_final_checkpoint(bucket, spec, "reference", checkpoint)
    assert bucket.calls[-1][0].endswith("/state.json")
    assert all(call[2].get("if_generation_match") == 0 for call in bucket.calls)
    committed = dict(bucket.objects)

    checkpoint.write_bytes(b"different-final")
    with pytest.raises(RuntimeError, match="412"):
        gate._publish_final_checkpoint(bucket, spec, "reference", checkpoint)
    assert bucket.objects == committed
    assert state["checkpoint_sha256"] != gate.sha256_file(checkpoint)


def _write_metrics(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["step", "loss", "lr", "steps_per_sec"]
        )
        writer.writeheader()
        writer.writerows(rows)


def test_metrics_merge_versions_without_gaps_or_telemetry_comparison(tmp_path):
    clean = tmp_path / "clean"
    resumed = tmp_path / "resumed"
    rows = [
        {"step": step, "loss": 1.0 / (step + 1), "lr": 0.0005,
         "steps_per_sec": 10 + step}
        for step in range(1, 5)
    ]
    _write_metrics(clean / "experiment" / "version_0" / "metrics.csv", rows)
    _write_metrics(resumed / "experiment" / "version_0" / "metrics.csv", rows[:2])
    resumed_rows = [dict(row, steps_per_sec=999) for row in rows[2:]]
    _write_metrics(resumed / "experiment" / "version_1" / "metrics.csv", resumed_rows)
    result = gate.compare_metric_histories(clean, resumed)
    assert result["ok"] is True
    assert result["clean"]["step_count"] == 4
    assert result["ignored_telemetry"] == ["steps_per_sec", "tokens_per_sec"]


def test_metrics_compare_manual_optimization_by_logger_segment_and_row(tmp_path):
    reference = tmp_path / "reference"
    recovered = tmp_path / "recovered"
    first = [
        {"step": 0, "loss": 0.9, "lr": 0.0005, "steps_per_sec": 1},
        {"step": 0, "loss": 0.8, "lr": 0.0005, "steps_per_sec": 1},
    ]
    second = [
        {"step": 0, "loss": 0.7, "lr": 0.0005, "steps_per_sec": 1},
        {"step": 0, "loss": 0.6, "lr": 0.0005, "steps_per_sec": 1},
    ]
    _write_metrics(reference / "experiment" / "version_0" / "metrics.csv", first)
    _write_metrics(reference / "experiment" / "version_1" / "metrics.csv", second)
    _write_metrics(recovered / "experiment" / "version_0" / "metrics.csv", first)
    _write_metrics(
        recovered / "experiment" / "version_1" / "metrics.csv",
        [dict(row, steps_per_sec=999) for row in second],
    )

    result = gate.compare_metric_histories(reference, recovered)
    assert result["ok"] is True
    assert result["clean"]["mode"] == "logger_segment_row"
    assert result["clean"]["segments"] == {"0": 2, "1": 2}


def test_metrics_manual_optimization_rejects_missing_resume_row(tmp_path):
    reference = tmp_path / "reference"
    recovered = tmp_path / "recovered"
    rows = [
        {"step": 0, "loss": 0.9, "lr": 0.0005, "steps_per_sec": 1},
        {"step": 0, "loss": 0.8, "lr": 0.0005, "steps_per_sec": 1},
    ]
    _write_metrics(reference / "experiment" / "version_0" / "metrics.csv", rows)
    _write_metrics(recovered / "experiment" / "version_0" / "metrics.csv", rows[:1])
    result = gate.compare_metric_histories(reference, recovered)
    assert result["ok"] is False
    assert result["mismatch"] == "optimizer-step sets differ"


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
def test_metrics_reject_nonfinite_values(tmp_path, bad):
    root = tmp_path / "metrics"
    _write_metrics(
        root / "experiment" / "version_0" / "metrics.csv",
        [{"step": 1, "loss": bad, "lr": 0.0005, "steps_per_sec": 1}],
    )
    with pytest.raises(gate.GateError, match="non-finite metric"):
        gate._normalized_metrics(root)


def test_metrics_reject_duplicate_version_across_experiments(tmp_path):
    row = [{"step": 0, "loss": 1.0, "lr": 0.0005, "steps_per_sec": 1}]
    _write_metrics(tmp_path / "a" / "version_0" / "metrics.csv", row)
    _write_metrics(tmp_path / "b" / "version_0" / "metrics.csv", row)
    with pytest.raises(gate.GateError, match="duplicate CSVLogger version"):
        gate._normalized_metrics(tmp_path)


def test_metrics_reject_gap_duplicate_and_value_drift(tmp_path):
    clean = tmp_path / "clean"
    resumed = tmp_path / "resumed"
    base = [
        {"step": 1, "loss": 1.0, "lr": 0.0005, "steps_per_sec": 1},
        {"step": 2, "loss": 0.9, "lr": 0.0005, "steps_per_sec": 1},
        {"step": 3, "loss": 0.8, "lr": 0.0005, "steps_per_sec": 1},
    ]
    _write_metrics(clean / "experiment" / "version_0" / "metrics.csv", base)
    _write_metrics(resumed / "experiment" / "version_0" / "metrics.csv", base[:1])
    _write_metrics(resumed / "experiment" / "version_1" / "metrics.csv", base[2:])
    with pytest.raises(gate.GateError, match="gap"):
        gate.compare_metric_histories(clean, resumed)

    resumed2 = tmp_path / "resumed2"
    _write_metrics(resumed2 / "experiment" / "version_0" / "metrics.csv", base[:2])
    _write_metrics(resumed2 / "experiment" / "version_1" / "metrics.csv", base[1:])
    with pytest.raises(gate.GateError, match="duplicate"):
        gate.compare_metric_histories(clean, resumed2)

    resumed3 = tmp_path / "resumed3"
    drifted = [dict(row) for row in base]
    drifted[1]["loss"] = 0.7
    _write_metrics(resumed3 / "experiment" / "version_0" / "metrics.csv", drifted)
    assert gate.compare_metric_histories(clean, resumed3)["ok"] is False


def _run_final_comparison(monkeypatch, tmp_path, *, recovered_scaler, omit_scaler=None):
    fake_torch = types.ModuleType("torch")
    fake_torch.is_tensor = lambda _value: False
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    reference_path = tmp_path / "reference.pt"
    recovered_path = tmp_path / "recovered.pt"
    reference_path.write_bytes(b"reference-final")
    recovered_path.write_bytes(b"recovered-final")
    common = {
        "training_steps": gate.TRAIN_STEPS,
        "model": {}, "optimizer": {}, "lr_scheduler_state": {"step": 300},
        "lurestar_rng_state_v1": {"torch": [1, 2, 3]},
    }
    reference = dict(common)
    recovered = dict(common)
    reference[gate.AMP_SCALER_KEY] = {"scale": 1024.0, "growth_tracker": 7}
    recovered[gate.AMP_SCALER_KEY] = recovered_scaler
    if omit_scaler == "reference":
        reference.pop(gate.AMP_SCALER_KEY)
    elif omit_scaler == "recovered":
        recovered.pop(gate.AMP_SCALER_KEY)
    _install_fake_checkpoint_verifier(monkeypatch, {
        reference_path.name: reference, recovered_path.name: recovered,
    })
    monkeypatch.setattr(gate, "_probe_logits", lambda _payload: [1.0])
    monkeypatch.setattr(gate, "compare_metric_histories", lambda _left, _right: {"ok": True})
    observations = {
        "schema": gate.TrainingObservationLatch.SCHEMA,
        "line_count": 1,
        "events": [{"kind": "data_fast_forward", "step": 150, "line_number": 1}],
    }
    return gate.compare_final_checkpoints(
        reference_path, recovered_path,
        reference_out=tmp_path / "reference", recovered_out=tmp_path / "recovered",
        reference_observations=observations,
        recovered_observations=observations,
        lineage_sha="f" * 64,
    )


def test_final_comparison_requires_exact_fp16_grad_scaler_equality(monkeypatch, tmp_path):
    result = _run_final_comparison(
        monkeypatch, tmp_path,
        recovered_scaler={"scale": 1024.0, "growth_tracker": 7},
    )
    assert result["passed"] is True
    assert result["checks"]["amp_grad_scaler"]["ok"] is True
    data_position = result["checks"]["data_position"]
    assert data_position["observation_schema"] == gate.TrainingObservationLatch.SCHEMA
    assert data_position["reference_observation"]["events"][0]["step"] == 150
    assert data_position["recovered_observation"]["events"][0]["step"] == 150

    result = _run_final_comparison(
        monkeypatch, tmp_path,
        recovered_scaler={"scale": 512.0, "growth_tracker": 7},
    )
    assert result["passed"] is False
    assert result["checks"]["amp_grad_scaler"]["ok"] is False


def test_tree_metrics_supports_exact_uint32_numpy_rng_state():
    np = pytest.importorskip("numpy")
    left = np.asarray([0, 2**32 - 1], dtype=np.uint32)
    same = left.copy()
    changed = np.asarray([0, 2**32 - 2], dtype=np.uint32)
    assert gate._tree_metrics(left, same, atol=0.0, rtol=0.0)["ok"] is True
    result = gate._tree_metrics(left, changed, atol=0.0, rtol=0.0)
    assert result["ok"] is False
    assert result["mismatch"].endswith("exact ndarray mismatch")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_tree_metrics_rejects_nonfinite_scalar_and_numpy_state(bad):
    np = pytest.importorskip("numpy")
    scalar = gate._tree_metrics(bad, bad, atol=1.0, rtol=1.0)
    assert scalar["ok"] is False
    assert scalar["mismatch"].endswith("non-finite scalar")
    array = gate._tree_metrics(
        np.asarray([bad], dtype=np.float64), np.asarray([bad], dtype=np.float64),
        atol=1.0, rtol=1.0,
    )
    assert array["ok"] is False
    assert array["mismatch"].endswith("non-finite ndarray")


def test_tree_metrics_rejects_nonfinite_tensor_state(monkeypatch):
    class FakeTensor:
        shape = (1,)
        dtype = "float32"

        def numel(self):
            return 1

        def is_floating_point(self):
            return True

        def is_complex(self):
            return False

        def detach(self):
            return self

        def cpu(self):
            return self

        def to(self, _dtype):
            return self

    class FiniteResult:
        def all(self):
            return False

    fake_torch = types.ModuleType("torch")
    fake_torch.float64 = "float64"
    fake_torch.is_tensor = lambda value: isinstance(value, FakeTensor)
    fake_torch.isfinite = lambda _value: FiniteResult()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    result = gate._tree_metrics(FakeTensor(), FakeTensor(), atol=1.0, rtol=1.0)
    assert result["ok"] is False
    assert result["mismatch"].endswith("non-finite tensor")


@pytest.mark.parametrize("missing", ["reference", "recovered"])
def test_final_comparison_refuses_missing_fp16_grad_scaler(monkeypatch, tmp_path, missing):
    with pytest.raises(gate.GateError, match="lacks AMP GradScaler"):
        _run_final_comparison(
            monkeypatch, tmp_path,
            recovered_scaler={"scale": 1024.0, "growth_tracker": 7},
            omit_scaler=missing,
        )


def test_exec_timeout_still_records_diagnostic_and_verifies_teardown(monkeypatch, tmp_path):
    monkeypatch.setattr(gate, "PROJECT_ROOT", tmp_path)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "x.py").write_text("x = 1\n")
    calls = []

    def runner(argv, *, check=True, relay=True, max_lines=200):
        calls.append(list(argv))
        if argv[:3] == ["colab", "quota", "--json"]:
            return 0, json.dumps({"paid_balance": 1500.0})
        if argv[:2] == ["colab", "start"]:
            return 0, json.dumps({"session": "gpu-t4-timeout"})
        if argv[:2] == ["colab", "exec"]:
            return 124, "transport disconnected after durable runtime snapshots"
        if argv[:3] == ["colab", "status", "--json"]:
            return 0, json.dumps({"status": "connected"})
        if argv[:3] == ["gcloud", "storage", "cat"]:
            return 1, "result not committed"
        return 0, "ok"

    monkeypatch.setattr(gate, "run_argv", runner)
    monkeypatch.setattr(
        gate, "status_pair",
        lambda *args, **kwargs: ({"status": "no_runtime"}, {"status": "no_runtime"}),
    )
    monkeypatch.setattr(
        gate, "quota_pair",
        lambda *args, **kwargs: (
            {"active_runtimes": 0, "burn_rate_hourly": 0.0, "paid_balance": 1499.0},
            {"active_runtimes": 0, "burn_rate_hourly": 0.0, "paid_balance": 1499.0},
        ),
    )
    receipt = tmp_path / "receipt.jsonl"
    with pytest.raises(gate.GateError, match="durable gate result is absent"):
        gate.host_main(run=True, receipt_path=receipt)

    exec_index = next(i for i, argv in enumerate(calls) if argv[:2] == ["colab", "exec"])
    stop_index = next(i for i, argv in enumerate(calls) if argv[:2] == ["colab", "stop"])
    assert stop_index > exec_index
    assert calls[stop_index] == ["colab", "stop", "--session", "gpu-t4-timeout"]
    events = [json.loads(line)["event"] for line in receipt.read_text().splitlines()]
    assert "EXEC_RETURNED" in events
    assert "HOST_DIAGNOSTIC" in events
    assert events[-1] == "SESSION_STOP_VERIFIED"


def test_host_defaults_to_prepare_only(monkeypatch, tmp_path):
    monkeypatch.setattr(gate, "PROJECT_ROOT", tmp_path)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "x.py").write_text("x = 1\n")
    assert gate.host_main(run=False, receipt_path=tmp_path / "receipt.jsonl") == 0
    rows = (tmp_path / "receipt.jsonl").read_text().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["event"] == "PREREGISTERED"
