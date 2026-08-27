from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


PROJECT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT / "scripts" / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exact = load_script("repair_hmm_exact_target_state")
soft = load_script("restore_soft_deleted_objects")


def test_soft_delete_recovery_dry_run_binds_exact_generation(monkeypatch):
    uri = "gs://bucket/lurestar/runs/job/state.json"
    live = b"poisoned\n"
    wanted = b"terminal\n"
    wanted_sha = hashlib.sha256(wanted).hexdigest()
    monkeypatch.setattr(soft, "describe", lambda value: {
        "generation": "22", "custom_fields": {"sha256": hashlib.sha256(live).hexdigest()}
    })
    monkeypatch.setattr(soft, "object_bytes", lambda value: live)
    monkeypatch.setattr(soft, "soft_deleted_record", lambda value, generation: {
        "metadata": {"generation": generation, "metadata": {"sha256": wanted_sha}}
    })

    result = soft.recover_one(
        {"uri": uri, "generation": "11", "sha256": wanted_sha}, apply=False
    )

    assert result["action"] == "would_restore"
    assert result["requested_generation"] == "11"
    assert result["expected_sha256"] == wanted_sha
    assert "--g22--sha256-" in result["previous_live_archive"]
    assert "--g11--sha256-" in result["restored_archive"]


def test_soft_delete_recovery_is_idempotent_when_exact_bytes_are_live(monkeypatch):
    uri = "gs://bucket/lurestar/runs/job/state.json"
    payload = b"terminal\n"
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(soft, "describe", lambda value: {
        "generation": "33", "custom_fields": {"sha256": digest}
    })
    monkeypatch.setattr(soft, "object_bytes", lambda value: payload)

    result = soft.recover_one(
        {"uri": uri, "generation": "11", "sha256": digest}, apply=False
    )

    assert result["action"] == "already_restored_dry_run"
    assert result["restored_live_generation"] == "33"


def test_soft_delete_recovery_can_resume_absent_live_after_transaction(monkeypatch):
    uri = "gs://bucket/lurestar/runs/job/state.json"
    wanted_sha = "b" * 64
    transaction = {
        "schema": soft.RECEIPT_SCHEMA,
        "uri": uri,
        "requested_generation": "11",
        "expected_sha256": wanted_sha,
        "previous_live_generation": "22",
        "previous_live_sha256": "a" * 64,
        "previous_live_archive": "gs://bucket/archive-before",
        "restored_archive": "gs://bucket/archive-restored",
        "scientific_metrics_inspected": False,
    }
    tx_uri = soft.transaction_uri(uri, "11")
    monkeypatch.setattr(
        soft, "describe", lambda value: (_ for _ in ()).throw(soft.RecoveryError("absent"))
    )
    monkeypatch.setattr(
        soft, "object_bytes",
        lambda value: (json.dumps(transaction) + "\n").encode() if value == tx_uri else b"",
    )
    monkeypatch.setattr(soft, "soft_deleted_record", lambda value, generation: {
        "metadata": {"generation": generation, "metadata": {"sha256": wanted_sha}}
    })

    result = soft.recover_one(
        {"uri": uri, "generation": "11", "sha256": wanted_sha}, apply=False
    )

    assert result["action"] == "would_resume_restore"
    assert result["transaction_uri"] == tx_uri


def _interrupted_transaction(uri: str, wanted_sha: str) -> dict:
    return {
        "schema": soft.RECEIPT_SCHEMA,
        "uri": uri,
        "requested_generation": "11",
        "expected_sha256": wanted_sha,
        "previous_live_generation": "22",
        "previous_live_sha256": "a" * 64,
        "previous_live_archive": "gs://bucket/archive-before",
        "restored_archive": "gs://bucket/archive-restored",
        "scientific_metrics_inspected": False,
    }


def test_soft_delete_recovery_apply_resumes_and_reads_restored_generation(monkeypatch):
    uri = "gs://bucket/lurestar/runs/job/state.json"
    wanted = b"terminal\n"
    wanted_sha = hashlib.sha256(wanted).hexdigest()
    transaction = _interrupted_transaction(uri, wanted_sha)
    tx_uri = soft.transaction_uri(uri, "11")
    descriptions = iter([
        soft.RecoveryError("absent"),
        {"generation": "44", "custom_fields": {"sha256": wanted_sha}},
    ])

    def fake_describe(value):
        result = next(descriptions)
        if isinstance(result, Exception):
            raise result
        return result

    reads = []

    def fake_bytes(value):
        reads.append(value)
        if value == tx_uri:
            return (json.dumps(transaction) + "\n").encode()
        if value == uri + "#44":
            return wanted
        if value == transaction["restored_archive"]:
            raise soft.RecoveryError("not archived yet")
        raise AssertionError(value)

    commands = []
    monkeypatch.setattr(soft, "describe", fake_describe)
    monkeypatch.setattr(soft, "object_bytes", fake_bytes)
    monkeypatch.setattr(soft, "soft_deleted_record", lambda value, generation: {
        "metadata": {"generation": generation, "metadata": {"sha256": wanted_sha}}
    })
    monkeypatch.setattr(soft, "command", lambda *args: commands.append(args) or b"")
    monkeypatch.setattr(soft, "copy_and_verify", lambda *args: None)

    result = soft.recover_one(
        {"uri": uri, "generation": "11", "sha256": wanted_sha}, apply=True
    )

    assert result["action"] == "restored_after_interruption"
    assert result["restored_live_generation"] == "44"
    assert uri + "#44" in reads
    assert ("gcloud", "storage", "restore", uri + "#11", "--if-generation-match=0") in commands


def test_soft_delete_recovery_apply_rejects_wrong_restored_metadata(monkeypatch):
    uri = "gs://bucket/lurestar/runs/job/state.json"
    wanted = b"terminal\n"
    wanted_sha = hashlib.sha256(wanted).hexdigest()
    transaction = _interrupted_transaction(uri, wanted_sha)
    tx_uri = soft.transaction_uri(uri, "11")
    descriptions = iter([
        soft.RecoveryError("absent"),
        {"generation": "44", "custom_fields": {"sha256": "0" * 64}},
    ])

    def fake_describe(value):
        result = next(descriptions)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(soft, "describe", fake_describe)
    monkeypatch.setattr(
        soft, "object_bytes",
        lambda value: ((json.dumps(transaction) + "\n").encode()
                       if value == tx_uri else wanted),
    )
    monkeypatch.setattr(soft, "soft_deleted_record", lambda value, generation: {
        "metadata": {"generation": generation, "metadata": {"sha256": wanted_sha}}
    })
    monkeypatch.setattr(soft, "command", lambda *args: b"")

    with pytest.raises(soft.RecoveryError, match="resumed restore metadata hash mismatch"):
        soft.recover_one(
            {"uri": uri, "generation": "11", "sha256": wanted_sha}, apply=True
        )


def test_exact_target_repair_selects_only_target_checkpoint(monkeypatch):
    bucket = "bucket"
    prefix = "lurestar"
    job_id = "gpt-seed1234-hmm-persistent_moderate"
    state_uri = f"gs://{bucket}/{prefix}/runs/{job_id}/state.json"
    out_root = f"/content/lurestar/runs/hmm_family/persistent_moderate/gpt/seed1234/base"
    state = {
        "schema": exact.STATE_SCHEMA,
        "run_id": job_id,
        "status": "FAILED",
        "step": 3001,
        "out_root": out_root,
        "source_snapshot_sha256": "a" * 64,
    }
    checkpoint_uri = (
        f"gs://{bucket}/{prefix}/runs/{job_id}/{job_id}/ckpt_iter_3000_x.pt"
    )
    checkpoint = b"checkpoint"
    metadata = (json.dumps({
        "run_id": job_id,
        "step": 3000,
        "sha256": hashlib.sha256(checkpoint).hexdigest(),
        "size_bytes": len(checkpoint),
        "path": f"{out_root}/{job_id}/ckpt_iter_3000_x.pt",
    }) + "\n").encode()
    state_payload = (json.dumps(state) + "\n").encode()

    def fake_bytes(uri):
        return {
            state_uri + "#123": state_payload,
            checkpoint_uri + "#123": checkpoint,
            checkpoint_uri + ".meta.json#123": metadata,
        }[uri]

    def fake_describe(uri):
        if uri == state_uri:
            return {
                "name": uri.removeprefix(f"gs://{bucket}/"),
                "generation": "123",
                "size": len(state_payload),
                "custom_fields": {"sha256": hashlib.sha256(state_payload).hexdigest()},
            }
        payload = fake_bytes(uri + "#123")
        return {
            "name": uri.removeprefix(f"gs://{bucket}/"),
            "generation": "123",
            "size": len(payload),
        }

    monkeypatch.setattr(exact, "gcs_bytes", fake_bytes)
    monkeypatch.setattr(exact, "describe", fake_describe)
    monkeypatch.setattr(exact, "unique_object", lambda pattern: checkpoint_uri)
    monkeypatch.setattr(exact, "checkpoint_payload_step", lambda payload: 3000)

    result = exact.repair_one(
        bucket=bucket, prefix=prefix, job_id=job_id, target_step=3000,
        source_sha256="a" * 64, apply=False,
    )

    assert result["action"] == "would_repair"
    assert result["checkpoint_sha256"] == hashlib.sha256(checkpoint).hexdigest()
    assert result["checkpoint_generation"] == "123"
    assert result["checkpoint_uri"].endswith("ckpt_iter_3000_x.pt")


def test_exact_target_repair_revalidates_already_exact_state(monkeypatch):
    bucket = "bucket"
    job_id = "gpt-seed1234-hmm-persistent_moderate"
    state_uri = f"gs://{bucket}/lurestar/runs/{job_id}/state.json"
    local_checkpoint = f"/content/lurestar/runs/x/{job_id}/ckpt_iter_3000_x.pt"
    state = {
        "schema": exact.STATE_SCHEMA,
        "run_id": job_id,
        "status": "TRAINED",
        "step": 3000,
        "out_root": "/content/lurestar/runs/x",
        "source_snapshot_sha256": "a" * 64,
        "checkpoint": {
            "path": local_checkpoint, "sha256": "b" * 64,
            "size_bytes": 10, "generation": "7",
        },
        "artifacts": [
            {"local_path": local_checkpoint, "remote": "lurestar/runs/x/c.pt",
             "generation": "7"},
            {"local_path": local_checkpoint + ".meta.json",
             "remote": "lurestar/runs/x/c.pt.meta.json", "generation": "8"},
        ],
    }
    payload = (json.dumps(state) + "\n").encode()
    monkeypatch.setattr(exact, "describe", lambda uri: {
        "generation": "9", "custom_fields": {"sha256": hashlib.sha256(payload).hexdigest()}
    })
    monkeypatch.setattr(exact, "gcs_bytes", lambda uri: payload)
    monkeypatch.setattr(
        exact, "validate_checkpoint_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(exact.RepairError("corrupt sidecar")),
    )

    with pytest.raises(exact.RepairError, match="corrupt sidecar"):
        exact.repair_one(
            bucket=bucket, prefix="lurestar", job_id=job_id, target_step=3000,
            source_sha256="a" * 64, apply=False,
        )


def _complete_exact_state(job_id: str, local_checkpoint: str) -> dict:
    return {
        "schema": exact.STATE_SCHEMA,
        "run_id": job_id,
        "status": "TRAINED",
        "step": 3000,
        "out_root": "/content/lurestar/runs/x",
        "source_snapshot_sha256": "a" * 64,
        "checkpoint": {
            "path": local_checkpoint, "sha256": "b" * 64,
            "size_bytes": 10, "generation": "7",
        },
        "artifacts": [
            {"local_path": local_checkpoint, "remote": "lurestar/runs/x/c.pt",
             "sha256": "b" * 64, "size_bytes": 10, "generation": "7"},
            {"local_path": local_checkpoint + ".meta.json",
             "remote": "lurestar/runs/x/c.pt.meta.json",
             "sha256": "c" * 64, "size_bytes": 20, "generation": "8"},
        ],
        "recovery_candidates": [{
            "path": local_checkpoint,
            "metadata_path": local_checkpoint + ".meta.json",
            "sha256": "b" * 64,
            "size_bytes": 10,
            "generation": "7",
            "metadata_sha256": "c" * 64,
            "metadata_size_bytes": 20,
            "metadata_generation": "8",
            "step": 3000,
        }],
    }


def test_exact_target_repair_accepts_complete_generation_bound_state(monkeypatch):
    bucket = "bucket"
    job_id = "gpt-seed1234-hmm-persistent_moderate"
    state_uri = f"gs://{bucket}/lurestar/runs/{job_id}/state.json"
    local_checkpoint = f"/content/lurestar/runs/x/{job_id}/ckpt_iter_3000_x.pt"
    state = _complete_exact_state(job_id, local_checkpoint)
    payload = (json.dumps(state) + "\n").encode()
    monkeypatch.setattr(exact, "describe", lambda uri: {
        "generation": "9", "custom_fields": {"sha256": hashlib.sha256(payload).hexdigest()}
    })
    monkeypatch.setattr(exact, "gcs_bytes", lambda uri: payload)
    monkeypatch.setattr(exact, "validate_checkpoint_bundle", lambda *args, **kwargs: (
        {"local_path": local_checkpoint, "remote": "lurestar/runs/x/c.pt",
         "sha256": "b" * 64, "size_bytes": 10, "generation": "7"},
        {"local_path": local_checkpoint + ".meta.json",
         "remote": "lurestar/runs/x/c.pt.meta.json",
         "sha256": "c" * 64, "size_bytes": 20, "generation": "8"},
    ))

    result = exact.repair_one(
        bucket=bucket, prefix="lurestar", job_id=job_id, target_step=3000,
        source_sha256="a" * 64, apply=False,
    )

    assert result["action"] == "already_exact_terminal"
    assert result["checkpoint_generation"] == "7"


def test_exact_target_repair_rejects_incomplete_recovery_candidate(monkeypatch):
    bucket = "bucket"
    job_id = "gpt-seed1234-hmm-persistent_moderate"
    state_uri = f"gs://{bucket}/lurestar/runs/{job_id}/state.json"
    local_checkpoint = f"/content/lurestar/runs/x/{job_id}/ckpt_iter_3000_x.pt"
    state = _complete_exact_state(job_id, local_checkpoint)
    del state["recovery_candidates"][0]["metadata_generation"]
    payload = (json.dumps(state) + "\n").encode()
    monkeypatch.setattr(exact, "describe", lambda uri: {
        "generation": "9", "custom_fields": {"sha256": hashlib.sha256(payload).hexdigest()}
    })
    monkeypatch.setattr(exact, "gcs_bytes", lambda uri: payload)
    monkeypatch.setattr(exact, "validate_checkpoint_bundle", lambda *args, **kwargs: (
        {"local_path": local_checkpoint, "remote": "lurestar/runs/x/c.pt",
         "sha256": "b" * 64, "size_bytes": 10, "generation": "7"},
        {"local_path": local_checkpoint + ".meta.json",
         "remote": "lurestar/runs/x/c.pt.meta.json",
         "sha256": "c" * 64, "size_bytes": 20, "generation": "8"},
    ))

    with pytest.raises(exact.RepairError, match="complete recovery candidate"):
        exact.repair_one(
            bucket=bucket, prefix="lurestar", job_id=job_id, target_step=3000,
            source_sha256="a" * 64, apply=False,
        )


def test_state_repair_upload_uses_generation_precondition(monkeypatch):
    calls = []
    monkeypatch.setattr(exact, "run", lambda *args, **kwargs: calls.append(args) or b"")

    exact.upload_bytes(
        "gs://bucket/state.json", b"{}\n", commit_metadata=True,
        if_generation_match="77",
    )

    assert calls
    assert "--if-generation-match=77" in calls[0]
    assert any(value.startswith("--custom-metadata=sha256=") for value in calls[0])


@pytest.mark.parametrize("value", [3000.9, "3000", True])
def test_checkpoint_payload_step_requires_a_literal_integer(monkeypatch, value):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        load=lambda *args, **kwargs: {"training_steps": value}
    ))

    with pytest.raises(exact.RepairError, match="integer training_steps"):
        exact.checkpoint_payload_step(b"fixture")
