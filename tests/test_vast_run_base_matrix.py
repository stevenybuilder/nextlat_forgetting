from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from types import SimpleNamespace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("vast_run_base_matrix_tested", ROOT / "scripts/vast_run_base_matrix.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ledger(tmp_path: Path, entries: list[dict]) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "run_ledger.json"
    path.write_text(json.dumps({"schema": "test", "entries": entries}))
    return path


def _trained(job_id: str = "gpt-s1234-base") -> dict:
    return {"job_id": job_id, "status": "TRAINED", "phase": "base", "step": 20_000}


def _terminal_local_state(root: Path, job_id: str = "gpt-s1234-base") -> dict:
    out_root = root / "runs" / "gpt" / "seed1234" / "base"
    out_root.mkdir(parents=True)
    artifact = out_root / "materialized_config.yaml"
    artifact.write_text("frozen: true\n")
    checkpoint = out_root / "final.ckpt"
    checkpoint.write_bytes(b"exact-terminal-checkpoint")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        **_trained(job_id), "out_root": str(out_root),
        "artifacts": {"materialized_config.yaml": digest(artifact)},
        "final_checkpoint": str(checkpoint), "final_checkpoint_sha256": digest(checkpoint),
    }


def test_vast_passes_frozen_source_snapshot_manifest_path_to_base_evaluator() -> None:
    module = _module()
    data_dir, manifest_dir = module.vast_base_stage_paths()
    assert data_dir == str(module.ROOT / "data" / "stargraph")
    assert manifest_dir == str(module.PROJECT / "manifests")
    assert manifest_dir != str(module.ROOT / "manifests")


def test_vast_quarantines_trained_evaluator_identity_mismatch_without_restart(tmp_path: Path) -> None:
    module = _module(); ledger = _ledger(tmp_path, [_trained()])
    disposition = module.classify_vast_failure(
        ledger=ledger, project=tmp_path, job_ids=("gpt-s1234-base",), returncode=2,
        identity_checker=lambda *_: "gpt-s1234-base evaluator/dataset/manifest identity differs from frozen identity",
    )
    assert disposition.retry is False
    assert disposition.kind == "SCIENTIFIC_IDENTITY_OR_SCHEMA_MISMATCH"
    receipt = module.write_vast_quarantine(tmp_path, disposition, ledger=ledger, returncode=2)
    payload = json.loads(receipt.read_text())
    assert payload["status"] == "QUARANTINED_NO_AUTORESTART"
    assert payload["scientific_outcomes_opened"] is False
    assert payload["failure"]["kind"] == disposition.kind


def test_vast_retries_only_incomplete_training_or_explicit_transport(tmp_path: Path) -> None:
    module = _module()
    incomplete = _ledger(tmp_path / "incomplete", [{"job_id": "gpt-s1234-base", "status": "INTERRUPTED", "step": 5_000}])
    retry = module.classify_vast_failure(
        ledger=incomplete, project=tmp_path, job_ids=("gpt-s1234-base",), returncode=1,
        identity_checker=lambda *_: None,
    )
    assert retry.retry is True and retry.kind == "INCOMPLETE_TRAINING"
    trained = _ledger(tmp_path / "trained", [_trained()])
    no_retry = module.classify_vast_failure(
        ledger=trained, project=tmp_path, job_ids=("gpt-s1234-base",), returncode=2,
        identity_checker=lambda *_: None,
    )
    assert no_retry.retry is False and no_retry.kind == "POST_TRAINING_OR_UNKNOWN_FAILURE"
    transport = module.classify_vast_failure(
        ledger=trained, project=tmp_path, job_ids=("gpt-s1234-base",), returncode=75,
        exception=RuntimeError("GCS connection reset during sync"), identity_checker=lambda *_: None,
    )
    assert transport.retry is True and transport.kind == "TRANSIENT_TRANSPORT"


def test_vast_ledger_first_restore_skips_old_recovery_download_only_after_terminal_hash_verification(tmp_path: Path) -> None:
    module = _module(); state = _terminal_local_state(tmp_path)
    ledger = _ledger(tmp_path, [state])
    remote = ["lurestar/runs/gpt-s1234-base/state.json"]
    needed, reason = module.vast_restore_required(
        ledger=ledger, root=tmp_path, state_names=remote, selected_jobs=("gpt-s1234-base",),
    )
    assert needed is False
    assert "locally hash-verified" in reason
    Path(state["final_checkpoint"]).write_bytes(b"tampered")
    needed, reason = module.vast_restore_required(
        ledger=ledger, root=tmp_path, state_names=remote, selected_jobs=("gpt-s1234-base",),
    )
    assert needed is True
    assert "exact scoped restore" in reason


def test_vast_restore_keeps_job_scope_and_restores_incomplete_selected_state(tmp_path: Path) -> None:
    module = _module(); ledger = _ledger(tmp_path, [{**_trained(), "status": "INTERRUPTED", "step": 4_000}])
    remote = [
        "lurestar/runs/gpt-s1234-base/state.json",
        "lurestar/runs/nextlat-s1234-base/state.json",
        "lurestar/runs/nextlat-s1234-hmm-control/state.json",
    ]
    jobs = module.remote_base_state_jobs(remote, selected_jobs=("gpt-s1234-base",))
    assert jobs == {"gpt-s1234-base"}
    needed, reason = module.vast_restore_required(
        ledger=ledger, root=tmp_path, state_names=remote, selected_jobs=("gpt-s1234-base",),
    )
    assert needed is True
    assert "not a terminal base state" in reason


def test_vast_supervisor_restart_preserves_newer_verified_local_active_checkpoint(
        tmp_path: Path) -> None:
    module = _module()
    out_root = tmp_path / "runs" / "gpt" / "1235" / "base"
    checkpoint = out_root / "gpt-s1235-base" / "recovery_ckpt_iter_14000.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"verified-local-checkpoint")
    pointer = out_root / "recovery_ckpt"
    pointer.write_text(str(checkpoint))
    ledger = _ledger(tmp_path, [{
        "job_id": "gpt-s1235-base", "status": "RUNNING", "step": 11_000,
        "out_root": str(out_root),
    }])

    class Durability:
        def _pointer_targets(self, root):
            assert root == out_root
            return [(pointer, checkpoint)]

        def _verify_checkpoint_for_sync(self, path):
            assert path == checkpoint
            return checkpoint.with_name(checkpoint.name + ".meta.json")

    candidate = module.verified_local_active_checkpoint(
        Durability(), ledger, ("gpt-s1235-base", "gpt-s1236-base"),
    )
    assert candidate == {
        "job_id": "gpt-s1235-base", "step": 14_000, "ledger_step": 11_000,
        "checkpoint": str(checkpoint),
    }


def test_vast_upload_wrapper_uses_bounded_resumable_timeout_without_adapter_retry_loop() -> None:
    module = _module()

    class Blob:
        def __init__(self):
            self.metadata = None
            self.chunk_size = None
            self.calls = []

        def upload_from_filename(self, filename, *args, **kwargs):
            self.calls.append((filename, args, kwargs))
            return "uploaded"

    class Bucket:
        name = "test-bucket"

        def __init__(self):
            self.raw = Blob()

        def blob(self, *args, **kwargs):
            assert args == ("runs/bst-s1236-base/recovery.ckpt",)
            assert kwargs == {"generation": 9}
            return self.raw

        def list_blobs(self, *args, **kwargs):
            return iter(())

    raw_bucket = Bucket()
    bucket = module.VastUploadBucket(raw_bucket)
    blob = bucket.blob("runs/bst-s1236-base/recovery.ckpt", generation=9)
    blob.metadata = {"sha256": "a" * 64}
    assert raw_bucket.raw.metadata == {"sha256": "a" * 64}
    assert raw_bucket.raw.chunk_size == module.VAST_UPLOAD_CHUNK_BYTES
    assert blob.upload_from_filename("/runs/recovery.ckpt", if_generation_match=0) == "uploaded"
    _, _, kwargs = raw_bucket.raw.calls[-1]
    assert kwargs == {"if_generation_match": 0, "timeout": module.VAST_UPLOAD_TIMEOUT, "retry": None}
    blob.upload_from_filename("/runs/recovery.ckpt", timeout=17)
    assert raw_bucket.raw.calls[-1][2]["timeout"] == 17
    assert list(bucket.list_blobs(prefix="lurestar/runs/")) == []


def test_vast_upload_wrapper_reuses_exact_committed_remote_bytes(tmp_path: Path) -> None:
    module = _module()
    checkpoint = tmp_path / "recovery_ckpt_iter_14000.pt"
    checkpoint.write_bytes(b"already-durable-checkpoint")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    class Blob:
        name = "lurestar/runs/gpt-s1235-base/recovery_ckpt_iter_14000.pt"
        size = checkpoint.stat().st_size
        metadata = {"sha256": digest}
        generation = 17

        def __init__(self):
            self.chunk_size = None
            self.uploads = 0

        def reload(self):
            return None

        def upload_from_filename(self, *_args, **_kwargs):
            self.uploads += 1

    raw = Blob()
    blob = module.VastUploadBlob(raw)
    blob.metadata = {"sha256": digest}
    assert blob.upload_from_filename(str(checkpoint)) is None
    assert raw.uploads == 0
    assert raw.metadata == {"sha256": digest}


def test_vast_background_sync_failure_never_aborts_training() -> None:
    module = _module()

    class StopAfterTwoAttempts:
        def __init__(self):
            self.calls = 0

        def wait(self, _interval):
            self.calls += 1
            return self.calls > 2

    class Durability:
        def __init__(self):
            self.calls = 0

        def sync_once(self, ledger, *, ledger_object):
            assert ledger == "/runs/ledger.json"
            assert ledger_object == "run_ledger-worker.json"
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("checkpoint pointer changed before state commit")
            return {"gpt-s1235-base": {"step": 15_000}}

    durability = Durability()
    assert module.vast_background_sync_loop(
        StopAfterTwoAttempts(), durability, "/runs/ledger.json",
        "run_ledger-worker.json", interval=0,
    ) is None
    assert durability.calls == 2


def test_vast_recovery_cadence_is_model_specific(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    assert module.VAST_FAST_RECOVERY_STEPS == 5_000
    assert module.VAST_BST_RECOVERY_STEPS == 1_000

    # This is the exact expression injected into the provider-local run_matrix copy.
    for model, expected in (("gpt", 5_000), ("nextlat", 5_000), ("bst", 1_000)):
        spec = SimpleNamespace(model=model)
        actual = 5_000 if spec.model in {"gpt", "nextlat"} else 1_000
        assert actual == expected

    project = tmp_path / "project"
    runner = project / "scripts" / "run_matrix.py"
    runner.parent.mkdir(parents=True)
    runner.write_text(
        "def command(spec):\n"
        "    return [\n"
        "        \"trainer.save_recovery_checkpoint=250\",\n"
        "    ]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "PROJECT", project)
    module.apply_vast_operational_overrides()
    patched = runner.read_text(encoding="utf-8")
    compile(patched, str(runner), "exec")
    namespace = {}
    exec(patched, namespace)
    assert namespace["command"](SimpleNamespace(model="gpt"))[-1].endswith("=5000")
    assert namespace["command"](SimpleNamespace(model="nextlat"))[-1].endswith("=5000")
    assert namespace["command"](SimpleNamespace(model="bst"))[-1].endswith("=1000")
