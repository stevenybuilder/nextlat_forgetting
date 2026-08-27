from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "profile_salvage", ROOT / "scripts" / "create_profile_salvage_receipt.py"
)
assert SPEC and SPEC.loader
salvage = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = salvage
SPEC.loader.exec_module(salvage)

LAUNCHER_SPEC = importlib.util.spec_from_file_location(
    "profile_loop_for_salvage_test", ROOT / "scripts" / "colab_profile_loop.py"
)
assert LAUNCHER_SPEC and LAUNCHER_SPEC.loader
launcher = importlib.util.module_from_spec(LAUNCHER_SPEC)
LAUNCHER_SPEC.loader.exec_module(launcher)

class FakeBackend:
    def __init__(self):
        self.records = []
        self.payloads = {}
        self.creates = []
        self.next_generation = 10_000

    def add(self, name, payload, generation, *, custom=True):
        record = salvage.ObjectRecord(
            name=name,
            generation=str(generation),
            size_bytes=len(payload),
            md5_base64=salvage.md5_base64(payload),
            crc32c_base64=salvage.crc32c_base64(payload),
            custom_sha256=salvage.sha256(payload) if custom else None,
        )
        self.records.append(record)
        self.payloads[(name, str(generation))] = payload
        return record

    def artifact(self, relative, payload, generation):
        digest = salvage.sha256(payload)
        name = "%s/artifacts/sha256/%s/%s" % (salvage.PROFILE_PREFIX, digest, relative)
        return self.add(name, payload, generation)

    def list(self, prefix):
        return [item for item in self.records if item.name.startswith(prefix)]

    def read(self, name, generation):
        return self.payloads[(name, str(generation))]

    def resolve(self, name):
        matches = [item for item in self.records if item.name == name]
        return max(matches, key=lambda item: int(item.generation)) if matches else None

    def create(self, name, payload):
        if any(item.name == name for item in self.records):
            raise salvage.SalvageError("create-only collision")
        self.next_generation += 1
        record = self.add(name, payload, self.next_generation)
        self.creates.append(name)
        return record


def source_archive(members):
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz") as bundle:
        for name, body in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.size = len(body)
            bundle.addfile(info, io.BytesIO(body))
    return raw.getvalue()


@pytest.fixture
def evidence(monkeypatch, tmp_path):
    old_profile = b"""import os\nX = 1\ndef validate_gate_group():\n    return 'old'\nclass ProfileDurability:\n    pass\ndef runtime_driver():\n    return X\n"""
    new_profile = b"""import os\nimport tempfile\nX = 1\ndef validate_gate_group():\n    return 'new'\nclass ProfileDurability:\n    value = 'new'\ndef runtime_driver():\n    return X\n"""
    members = {
        "scripts/colab_profile_loop.py": old_profile,
        "configs/frozen.yaml": b"seed: 1234\n",
    }
    archive = source_archive(members)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "scripts" / "colab_profile_loop.py").write_bytes(new_profile)
    (tmp_path / "scripts" / "create_profile_salvage_receipt.py").write_text("# new tool\n")
    (tmp_path / "tests" / "test_profile_salvage_receipt.py").write_text("# new test\n")
    (tmp_path / "configs" / "frozen.yaml").write_bytes(members["configs/frozen.yaml"])

    backend = FakeBackend()
    source_digest = salvage.sha256(archive)
    remote_inputs = launcher.build_remote_input_identity([
        {
            "name": prefix + "/fixture.bin", "generation": str(index + 1),
            "size_bytes": index + 1, "md5_base64": "md5-%d" % index,
            "crc32c_base64": "crc-%d" % index,
        }
        for index, prefix in enumerate(launcher.INPUT_PREFIXES)
    ])
    monkeypatch.setattr(salvage, "EXECUTED_SOURCE_SHA256", source_digest)
    monkeypatch.setattr(salvage, "EXECUTED_SOURCE_OBJECT", "lurestar/source/project-%s.tar.gz" % source_digest)
    monkeypatch.setattr(salvage, "EXECUTED_SOURCE_GENERATION", "77")
    monkeypatch.setattr(salvage, "INPUT_IDENTITY_SHA256", remote_inputs["identity_sha256"])
    profile_id = "a100-%s-%s" % (source_digest[:12], remote_inputs["identity_sha256"][:12])
    monkeypatch.setattr(salvage, "PROFILE_ID", profile_id)
    monkeypatch.setattr(salvage, "PROFILE_PREFIX", "lurestar/profiles/" + profile_id)
    backend.add(salvage.EXECUTED_SOURCE_OBJECT, archive, 77, custom=False)

    generation = 100
    base_record = backend.artifact("provenance/runtime.json", b"{}\n", generation)
    base_state = {
        "schema": "nextlat_forgetting/colab_profile_state/2",
        "profile_id": salvage.PROFILE_ID,
        "source_sha256": source_digest,
        "input_identity_sha256": salvage.INPUT_IDENTITY_SHA256,
        "remote_inputs": remote_inputs,
        "generation": 2,
        "complete": False,
        "artifact_fingerprint": "base-fingerprint",
        "artifacts": [{
            "relative_path": "provenance/runtime.json",
            "remote": base_record.name,
            "object_generation": base_record.generation,
            "sha256": salvage.sha256(b"{}\n"),
            "size_bytes": 3,
        }],
    }

    def artifact(relative, payload):
        nonlocal generation
        generation += 1
        return backend.artifact(relative, payload, generation)

    for model in ("gpt", "nextlat"):
        job = "lurestar-" + model
        root = "gate/root/runs/%s/seed1234/base" % model
        exp = "%s-seed1234-base" % model
        checkpoint_path = root + "/%s/ckpt_iter_500_final.pt" % exp
        checkpoint = (model + "-checkpoint").encode()
        job_doc = {
            "job": job, "returncode": 0, "steps": 500, "warmup_steps": 100,
            "log": "/old/" + job + ".log",
        }
        probe = {
            "exit": "ok", "peak_allocated_bytes": 1, "peak_reserved_bytes": 2,
            "cuda": {"device_name": "NVIDIA A100-SXM4-40GB", "bf16_supported": True},
        }
        metadata = {
            "training_steps": 500, "sha256": salvage.sha256(checkpoint),
            "size_bytes": len(checkpoint),
        }
        artifact("gate/jobs/%s.gpu.csv" % job, b"timestamp,memory\n")
        artifact("gate/jobs/%s.job.json" % job, salvage.canonical_bytes(job_doc))
        artifact("gate/jobs/%s.log" % job, b"complete\n")
        artifact("gate/jobs/%s.probe.1.json" % job, salvage.canonical_bytes(probe))
        artifact(root + "/.lurestar_job_identity.json", b"{}\n")
        artifact(root + "/metrics/step_0_contract.json", b"{}\n")
        artifact(root + "/latest_ckpt", ("/old/" + checkpoint_path.split("/", 7)[-1]).encode())
        artifact(root + "/%s/materialized_config.yaml" % exp, b"seed: 1234\n")
        artifact(root + "/%s/version_0/metrics.csv" % exp, b"step\n500\n")
        artifact(checkpoint_path, checkpoint)
        artifact(checkpoint_path + ".meta.json", salvage.canonical_bytes(metadata))

    bst_root = "gate/root/runs/bst/seed1234/base"
    bst_exp = bst_root + "/bst-seed1234-base"
    bst_checkpoint = b"bst-recovery-step-250"
    bst_pointer = ("/content/old/bst-seed1234-base/recovery_ckpt_iter_250.pt").encode()
    bst_meta = salvage.canonical_bytes({
        "training_steps": 250, "rng_state": True,
        "sha256": salvage.sha256(bst_checkpoint), "size_bytes": len(bst_checkpoint),
    })
    monkeypatch.setattr(salvage, "BST_RECOVERY_SHA256", salvage.sha256(bst_checkpoint))
    monkeypatch.setattr(salvage, "BST_RECOVERY_META_SHA256", salvage.sha256(bst_meta))
    monkeypatch.setattr(salvage, "BST_RECOVERY_POINTER_SHA256", salvage.sha256(bst_pointer))
    artifact(bst_root + "/.lurestar_job_identity.json", b"{}\n")
    artifact(bst_exp + "/materialized_config.yaml", b"seed: 1234\n")
    artifact(bst_root + "/metrics/step_0_contract.json", b"{}\n")
    artifact(bst_root + "/recovery_ckpt", bst_pointer)
    artifact(bst_exp + "/version_0/metrics.csv", b"step\n249\n")
    artifact(bst_exp + "/recovery_ckpt_iter_250.pt", bst_checkpoint)
    artifact(bst_exp + "/recovery_ckpt_iter_250.pt.meta.json", bst_meta)
    artifact(bst_exp + "/ckpt_iter_250_validation.pt", b"excluded-validation")
    monkeypatch.setattr(salvage, "BST_FINAL_VALIDATION_SHA256", salvage.sha256(b"excluded-validation"))
    # Multiple mutable versions must never create an implicit/ambiguous resume choice.
    artifact("gate/jobs/lurestar-bst.log", b"step 200\n")
    artifact("gate/jobs/lurestar-bst.log", b"step 320\n")
    artifact(bst_exp + "/version_0/metrics.csv", b"step\n320\n")

    state_payload = salvage.canonical_bytes(base_state)
    monkeypatch.setattr(salvage, "BASE_STATE_SHA256", salvage.sha256(state_payload))
    monkeypatch.setattr(salvage, "BASE_STATE_GENERATION", "88")
    backend.add(salvage.PROFILE_PREFIX + "/state.json", state_payload, 88)
    _, target_bodies = salvage.tree_member_hashes(tmp_path)
    target_payload = source_archive(target_bodies)
    target_archive = tmp_path / ".target-source.tar.gz"
    target_archive.write_bytes(target_payload)
    return backend, tmp_path, target_archive, salvage.sha256(target_payload)


def test_prepare_validates_completed_jobs_and_only_step250_bst_resume(evidence):
    backend, root, target_archive, _ = evidence
    attestation, state, clearance, pointer = salvage.prepare(
        backend, root, verify_payloads=True, target_source_archive=target_archive)

    assert attestation["schema"] == salvage.ATTESTATION_SCHEMA
    assert attestation["dual_source_binding"]["orchestration"]["equal"] is True
    assert set(state["completed_jobs"]) == {"lurestar-gpt", "lurestar-nextlat"}
    assert state["resume_jobs"]["lurestar-bst"] == {
        "step": 250, "checkpoint_sha256": salvage.BST_RECOVERY_SHA256,
    }
    paths = {item["relative_path"] for item in state["artifacts"]}
    assert "gate/jobs/lurestar-bst.log" not in paths
    assert "gate/root/runs/bst/seed1234/base/bst-seed1234-base/version_0/metrics.csv" in paths
    assert clearance["authorization"] == "GO"
    assert pointer["historical_state_overwritten"] is False


def test_apply_is_create_only_and_publishes_pointer_last(evidence):
    backend, root, target_archive, _ = evidence
    attestation, state, clearance, pointer = salvage.prepare(
        backend, root, verify_payloads=True, target_source_archive=target_archive)
    old_state = backend.read(salvage.PROFILE_PREFIX + "/state.json", "88")
    result = salvage.publish(backend, attestation, state, clearance, pointer)

    assert backend.creates[-1] == salvage.PROFILE_PREFIX + "/salvage/current.json"
    assert "/attestation.json" in backend.creates[0]
    assert "/promoted-state.json" in backend.creates[1]
    assert "/clearance.json" in backend.creates[2]
    assert backend.read(salvage.PROFILE_PREFIX + "/state.json", "88") == old_state
    assert result["document"]["historical_state_overwritten"] is False
    created = list(backend.creates)
    retried = salvage.publish(backend, attestation, state, clearance, pointer)
    assert backend.creates == created
    assert retried == result


def test_readback_rejects_corrupt_bytes(evidence):
    backend, root, target_archive, _ = evidence
    record = next(item for item in backend.records if item.name.endswith("lurestar-gpt.job.json"))
    backend.payloads[(record.name, record.generation)] = b"corrupt"
    with pytest.raises(salvage.SalvageError, match="size mismatch"):
        salvage.prepare(backend, root, verify_payloads=True,
                        target_source_archive=target_archive)


def test_source_attestation_rejects_nonallowlisted_change(evidence):
    backend, root, target_archive, _ = evidence
    (root / "configs" / "frozen.yaml").write_text("seed: 9999\n")
    with pytest.raises(salvage.SalvageError, match="non-orchestration source changed"):
        salvage.prepare(backend, root, verify_payloads=False,
                        target_source_archive=target_archive)


def test_source_attestation_rejects_profile_training_contract_change(evidence):
    backend, root, target_archive, _ = evidence
    path = root / "scripts" / "colab_profile_loop.py"
    path.write_text(path.read_text().replace("return X", "return X + 1"))
    with pytest.raises(salvage.SalvageError, match="training-contract projection changed"):
        salvage.prepare(backend, root, verify_payloads=False,
                        target_source_archive=target_archive)


def test_target_archive_must_exactly_match_audited_tree(evidence):
    backend, root, target_archive, _ = evidence
    target_archive.write_bytes(source_archive({"unexpected.txt": b"drift\n"}))
    with pytest.raises(salvage.SalvageError, match="does not exactly represent"):
        salvage.prepare(backend, root, verify_payloads=False,
                        target_source_archive=target_archive)


def test_bst_wrong_pointer_is_not_salvageable(evidence):
    backend, root, target_archive, _ = evidence
    record = next(item for item in backend.records
                  if item.name.endswith("/gate/root/runs/bst/seed1234/base/recovery_ckpt")
                  and ("/sha256/%s/" % salvage.BST_RECOVERY_POINTER_SHA256) in item.name)
    backend.payloads[(record.name, record.generation)] = b"/wrong/recovery_ckpt_iter_125.pt"
    with pytest.raises(salvage.SalvageError):
        salvage.prepare(backend, root, verify_payloads=True,
                        target_source_archive=target_archive)


def test_clearance_round_trips_through_profile_loop_consumer(evidence):
    backend, root, target_archive, target_sha = evidence
    _, _, clearance, _ = salvage.prepare(
        backend, root, verify_payloads=True, target_source_archive=target_archive)

    launcher.validate_salvage_receipt(
        clearance, target_source_sha256=target_sha,
        remote_inputs=clearance["remote_inputs"])
    spec = launcher.build_spec(
        target_sha, clearance["remote_inputs"], salvage_receipt=clearance)
    assert spec["salvage_receipt"] == clearance


def test_crc32c_known_vector():
    assert salvage.crc32c_base64(b"abc") == "Nks/tw=="
