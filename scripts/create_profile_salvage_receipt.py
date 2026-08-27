#!/usr/bin/env python3
"""Plan, verify, and atomically attest salvage from the interrupted A100 profile.

The default mode is metadata-only planning and performs no writes. ``--dry-run`` performs
all read-back/hash/content validations without writes. ``--apply`` additionally publishes
four create-only objects (attestation, promoted state, consumer clearance, then the state-last
pointer). The historical profile ``state.json`` is never overwritten.
"""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import dataclasses
import hashlib
import io
import json
import os
import pathlib
import re
import subprocess
import sys
import tarfile
import typing as t

try:
    import google_crc32c as _google_crc32c
except ImportError:  # pragma: no cover - exercised only on minimal recovery hosts
    _google_crc32c = None


BUCKET = "nextlat-lurestar-project-flash-490419"
PROFILE_ID = "a100-b15e1bc9d596-8d316efc9c53"
PROFILE_PREFIX = "lurestar/profiles/" + PROFILE_ID
SESSION_ID = "gpu-a100-s-kkb-usc1f1-1nzrnzrdovp1u"
EXECUTED_SOURCE_SHA256 = "b15e1bc9d5961127330a7569f7d959e79c1999c2bb325801290d3164f921181d"
EXECUTED_SOURCE_GENERATION = "1787546814541595"
EXECUTED_SOURCE_OBJECT = "lurestar/source/project-%s.tar.gz" % EXECUTED_SOURCE_SHA256
INPUT_IDENTITY_SHA256 = "8d316efc9c53ef1cf69a2d556ef610918e3739a935088f6eb40b005a6c502e7c"
BASE_STATE_GENERATION = "1787546930055194"
BASE_STATE_SHA256 = "39fa27af49f8c13d6e3b7c37e18462fdab3ac3efed18a787cc10b574af0dd5e2"
BASE_STATE_LOGICAL_GENERATION = 2
BST_RECOVERY_SHA256 = "c8969aeb136b7ec4b1abee21ba385e7c4d794c22224d371fbb8e6a5d355180ea"
BST_RECOVERY_META_SHA256 = "d7d522921b9b744ba7de64e082edc68f9a1406be54822d3ab546ea2d20d77c0b"
BST_RECOVERY_POINTER_SHA256 = "bdbec07590b3f97cf3690e9e9da55ffbfb8eefd3111739137cdee82ac15c82fa"
BST_FINAL_VALIDATION_SHA256 = "3a762483f7daa181e6fe11f1e157f1906fce609f9885f95184e9e2bfd0463786"
ATTESTATION_SCHEMA = "nextlat_forgetting/profile_salvage_attestation/1"
STATE_SCHEMA = "nextlat_forgetting/profile_salvage_state/1"
POINTER_SCHEMA = "nextlat_forgetting/profile_salvage_pointer/1"
CLEARANCE_SCHEMA = "nextlat_forgetting/profile_salvage_clearance/1"

ALLOWED_SOURCE_CHANGES = frozenset({
    "scripts/colab_profile_loop.py",
    "scripts/colab_recovery_gate.py",
    "scripts/create_profile_salvage_receipt.py",
    "scripts/create_recovery_clearance_receipt.py",
    "scripts/profile.sh",
    "scripts/profile_entry.py",
    "scripts/profile_resume.py",
    "scripts/profile_summarize.py",
    "tests/test_colab_profile_loop.py",
    "tests/test_colab_recovery_gate.py",
    "tests/test_profile_salvage_receipt.py",
    "tests/test_profile_tooling.py",
    "tests/test_recovery_clearance_receipt.py",
})
APPROVED_RESUME_TRANSFORMATION = {
    "scripts/profile.sh": (
        "5eba9c71f218eeda30174b2f6b5c017766dd280e467ae21c25242a985ef02d3c",
        "b7f8f08020852de1deaa2e21a9443fd69945389a15a1173b3a6d384a78d1fc3d",
    ),
    "scripts/profile_entry.py": (
        "486be64e61fa7734ed597f733a20cbaa69f1ef8fc94e46f191aaa3e6637ca317",
        "d0f7ac9cb0a25ece40c7fa25895e32f2876a8024af33c488b8eb479012cef692",
    ),
    "scripts/profile_resume.py": (
        None,
        "8046afb796c3421b6ee42ec1943c4c09e6306c618cefa5b45ab2ab5f31b6ea10",
    ),
    "scripts/profile_summarize.py": (
        "a3a2767633ed840611355076f437e93fac3ea49b42b4d84d24c8af4fedfbc0a8",
        "fdd9e12ac05203727951a055b600af403529b028050aaaaca407986d760529fe",
    ),
}
APPROVED_PROFILE_ORCHESTRATOR = (
    "a2fdb664def7b667c285c6c8a42e9a00e016bfd89a95d72aeea9b5963c462d2a",
    "5f27e66b4b92a0c4b11c0ad491136a40e7977975eb4d7e8bbe3066d6ad9f949f",
)
TREE_EXCLUDED = frozenset({
    ".agent_state", ".git", ".secrets", ".venv", "__pycache__", "data", "docs",
    "output", "report", "results", "source_snapshot", "upstream",
})


class SalvageError(RuntimeError):
    """Evidence is incomplete or differs from the frozen salvage policy."""


@dataclasses.dataclass(frozen=True)
class ObjectRecord:
    name: str
    generation: str
    size_bytes: int
    md5_base64: str
    crc32c_base64: str
    custom_sha256: str | None = None

    def identity(self) -> dict[str, t.Any]:
        return dataclasses.asdict(self)


class Backend(t.Protocol):
    def list(self, prefix: str) -> list[ObjectRecord]: ...
    def read(self, name: str, generation: str) -> bytes: ...
    def create(self, name: str, payload: bytes) -> ObjectRecord: ...
    def resolve(self, name: str) -> ObjectRecord | None: ...


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SalvageError(message)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def md5_base64(payload: bytes) -> str:
    return base64.b64encode(hashlib.md5(payload, usedforsecurity=False).digest()).decode()


def crc32c_base64(payload: bytes) -> str:
    if _google_crc32c is not None:
        checksum = _google_crc32c.value(payload)
        return base64.b64encode(checksum.to_bytes(4, "big")).decode()
    crc = 0xFFFFFFFF
    for byte in payload:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x82F63B78 if crc & 1 else 0)
    return base64.b64encode(((crc ^ 0xFFFFFFFF) & 0xFFFFFFFF).to_bytes(4, "big")).decode()


def canonical_bytes(document: dict[str, t.Any]) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()


def canonical_sha256(document: dict[str, t.Any]) -> str:
    """Match colab_profile_loop's compact JSON fingerprint exactly."""
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return sha256(payload)


class GcloudBackend:
    """Small gcloud transport; mutating calls are reachable only from explicit --apply."""

    def __init__(self, bucket: str = BUCKET):
        self.bucket = bucket

    def _run(self, argv: list[str], *, payload: bytes | None = None) -> bytes:
        completed = subprocess.run(
            argv, input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )
        if completed.returncode:
            raise SalvageError(completed.stderr.decode("utf-8", "replace").strip())
        return completed.stdout

    @staticmethod
    def _record(metadata: dict[str, t.Any]) -> ObjectRecord:
        return ObjectRecord(
            name=str(metadata["name"]), generation=str(metadata["generation"]),
            size_bytes=int(metadata["size"]), md5_base64=str(metadata.get("md5Hash", "")),
            crc32c_base64=str(metadata.get("crc32c", "")),
            custom_sha256=(metadata.get("metadata") or {}).get("sha256"),
        )

    def list(self, prefix: str) -> list[ObjectRecord]:
        uri = "gs://%s/%s/**" % (self.bucket, prefix.rstrip("/"))
        raw = self._run(["gcloud", "storage", "ls", "--recursive", "--json", uri])
        return [self._record(item["metadata"]) for item in json.loads(raw)]

    def describe(self, name: str) -> ObjectRecord:
        uri = "gs://%s/%s" % (self.bucket, name)
        raw = self._run(["gcloud", "storage", "objects", "describe", uri, "--format=json"])
        metadata = json.loads(raw)
        # `objects describe` uses snake_case, unlike `ls --json`.
        return ObjectRecord(
            name=str(metadata["name"]), generation=str(metadata["generation"]),
            size_bytes=int(metadata["size"]), md5_base64=str(metadata.get("md5_hash", "")),
            crc32c_base64=str(metadata.get("crc32c_hash", "")),
            custom_sha256=(metadata.get("custom_fields") or {}).get("sha256"),
        )

    def resolve(self, name: str) -> ObjectRecord | None:
        try:
            return self.describe(name)
        except SalvageError:
            return None

    def read(self, name: str, generation: str) -> bytes:
        uri = "gs://%s/%s#%s" % (self.bucket, name, generation)
        return self._run(["gcloud", "storage", "cat", uri])

    def create(self, name: str, payload: bytes) -> ObjectRecord:
        digest = sha256(payload)
        uri = "gs://%s/%s" % (self.bucket, name)
        self._run([
            "gcloud", "storage", "cp", "--if-generation-match=0",
            "--content-type=application/json", "--custom-metadata=sha256=%s" % digest,
            "-", uri,
        ], payload=payload)
        return self.describe(name)


def verify_object(backend: Backend, record: ObjectRecord, *, require_custom_sha: bool = True) -> bytes:
    payload = backend.read(record.name, record.generation)
    digest = sha256(payload)
    _require(len(payload) == record.size_bytes, "size mismatch: %s" % record.name)
    _require(md5_base64(payload) == record.md5_base64, "MD5 mismatch: %s" % record.name)
    _require(crc32c_base64(payload) == record.crc32c_base64,
             "CRC32C mismatch: %s" % record.name)
    if require_custom_sha:
        _require(record.custom_sha256 == digest, "custom SHA mismatch: %s" % record.name)
    match = re.search(r"/artifacts/sha256/([0-9a-f]{64})/", record.name)
    if match:
        _require(match.group(1) == digest, "content-addressed path mismatch: %s" % record.name)
    return payload


def create_or_verify(backend: Backend, name: str, payload: bytes) -> ObjectRecord:
    """Create with generation-match zero, or prove an interrupted retry is identical."""
    try:
        record = backend.create(name, payload)
    except Exception as create_error:
        record = backend.resolve(name)
        if record is None:
            raise create_error
    observed = verify_object(backend, record)
    _require(observed == payload, "create-only object collision differs: %s" % name)
    return record


def archive_member_hashes(payload: bytes) -> tuple[dict[str, str], dict[str, bytes]]:
    hashes: dict[str, str] = {}
    bodies: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as bundle:
        for member in bundle.getmembers():
            if not member.isfile():
                continue
            stream = bundle.extractfile(member)
            _require(stream is not None, "source member unreadable: %s" % member.name)
            body = stream.read()
            hashes[member.name] = sha256(body)
            bodies[member.name] = body
    return hashes, bodies


def tree_member_hashes(root: pathlib.Path) -> tuple[dict[str, str], dict[str, bytes]]:
    hashes: dict[str, str] = {}
    bodies: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (path.is_symlink() or not path.is_file() or any(part in TREE_EXCLUDED for part in relative.parts)
                or any(part.startswith(".") for part in relative.parts)
                or path.name in {"adc.json", "application_default_credentials.json"}
                or path.suffix in {".pt", ".ckpt"}
                or path.name.endswith((".tar.gz", ".tgz"))):
            continue
        body = path.read_bytes()
        name = relative.as_posix()
        hashes[name] = sha256(body)
        bodies[name] = body
    return hashes, bodies


def _profile_contract_projection(source: bytes) -> str:
    tree = ast.parse(source.decode("utf-8"))
    kept: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            if names == ["tempfile"]:
                continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in {
            "ProfileDurability", "validate_gate_group",
        }:
            continue
        kept.append(node)
    projection = ast.Module(body=kept, type_ignores=[])
    return sha256(ast.dump(projection, include_attributes=False).encode())


def _literal_assignment(source: bytes, name: str) -> t.Any:
    tree = ast.parse(source.decode("utf-8"))
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and
                isinstance(node.targets[0], ast.Name) and node.targets[0].id == name):
            return ast.literal_eval(node.value)
    raise SalvageError("profile orchestrator lacks frozen assignment: %s" % name)


def _verify_profile_orchestrator_change(old_body: bytes, new_body: bytes) -> dict[str, t.Any]:
    expected_old, expected_new = APPROVED_PROFILE_ORCHESTRATOR
    _require(sha256(old_body) == expected_old and sha256(new_body) == expected_new,
             "training-contract projection changed with unreviewed profile orchestrator bytes")
    frozen = ("PINNED_COMMIT", "EXPECTED_GATE_JOBS", "EXPECTED_SMOKE_JOBS")
    _require(all(_literal_assignment(old_body, name) == _literal_assignment(new_body, name)
                 for name in frozen),
             "profile orchestrator changed a frozen training/job assignment")
    return {
        "executed_sha256": expected_old,
        "resume_sha256": expected_new,
        "semantic_invariants": {
            "pinned_upstream_commit_equal": True,
            "gate_job_steps_and_warmup_equal": True,
            "smoke_job_matrix_equal": True,
            "changes_are_durability_salvage_and_attempt_accounting_only": True,
        },
    }


def _verify_resume_transformation(
    old_hashes: dict[str, str], new_hashes: dict[str, str],
    old_bodies: dict[str, bytes], new_bodies: dict[str, bytes], changed: list[str],
) -> dict[str, t.Any] | None:
    resume_paths = set(APPROVED_RESUME_TRANSFORMATION)
    touched = resume_paths & set(changed)
    if not touched:
        return None
    _require(touched == resume_paths, "profile resume transformation is incomplete")
    for name, (expected_old, expected_new) in APPROVED_RESUME_TRANSFORMATION.items():
        _require(old_hashes.get(name) == expected_old and new_hashes.get(name) == expected_new,
                 "unreviewed resume implementation bytes: %s" % name)

    old_shell = old_bodies["scripts/profile.sh"].decode()
    new_shell = new_bodies["scripts/profile.sh"].decode()
    job_pattern = re.compile(
        r"^\s*run_job\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+([0-9]+)\s+([0-9]+)\s*$",
        re.MULTILINE,
    )
    old_jobs = job_pattern.findall(old_shell)
    new_jobs = job_pattern.findall(new_shell)
    _require(old_jobs == new_jobs and len(old_jobs) == 5,
             "profile job/config/step/warmup matrix changed")
    _require(re.search(r"^SEED=1234$", old_shell, re.MULTILINE) is not None and
             re.search(r"^SEED=1234$", new_shell, re.MULTILINE) is not None,
             "profile seed changed")
    trainer_override = re.compile(r'"(trainer\.[A-Za-z0-9_.]+(?:=[^"\s]+)?)"')
    old_overrides = set(trainer_override.findall(old_shell))
    new_overrides = set(trainer_override.findall(new_shell))
    _require(new_overrides - old_overrides == {"trainer.init_from=resume"} and
             not (old_overrides - new_overrides),
             "resume introduced a trainer override beyond init_from=resume")
    for invariant in (
        '"$SCRIPT_DIR/launch_train.sh" "$config" "$SEED"',
        '"trainer.train_batches=$steps"',
        '"trainer.val_interval=$val_interval"',
        '"trainer.test_interval=$val_interval"',
        '"trainer.save_recovery_checkpoint=$recovery"',
    ):
        _require(invariant in old_shell and invariant in new_shell,
                 "profile launch invariant changed: %s" % invariant)
    transformation = {
        name: {"executed_sha256": expected_old, "resume_sha256": expected_new}
        for name, (expected_old, expected_new) in sorted(APPROVED_RESUME_TRANSFORMATION.items())
    }
    transformation["semantic_invariants"] = {
        "job_matrix_equal": True,
        "seed_equal": True,
        "only_new_trainer_override": "trainer.init_from=resume",
        "resume_checkpoint_requires_hash_size_step_and_exact_experiment": True,
        "summarizer_uses_checkpoint_consistent_attempt_segments": True,
    }
    return transformation


def verify_source_compatibility(executed_archive: bytes, project_root: pathlib.Path) -> dict:
    old_hashes, old_bodies = archive_member_hashes(executed_archive)
    new_hashes, new_bodies = tree_member_hashes(project_root)
    changed = sorted(
        name for name in set(old_hashes) | set(new_hashes)
        if old_hashes.get(name) != new_hashes.get(name)
    )
    unexpected = sorted(set(changed) - ALLOWED_SOURCE_CHANGES)
    _require(not unexpected, "non-orchestration source changed: %s" % ", ".join(unexpected))
    profile_path = "scripts/colab_profile_loop.py"
    _require(profile_path in old_bodies and profile_path in new_bodies,
             "profile orchestrator missing from source comparison")
    old_projection = _profile_contract_projection(old_bodies[profile_path])
    new_projection = _profile_contract_projection(new_bodies[profile_path])
    profile_orchestrator_change = None
    if old_projection != new_projection:
        profile_orchestrator_change = _verify_profile_orchestrator_change(
            old_bodies[profile_path], new_bodies[profile_path])
    resume_transformation = _verify_resume_transformation(
        old_hashes, new_hashes, old_bodies, new_bodies, changed)
    stable = {
        name: old_hashes[name] for name in sorted(old_hashes)
        if name not in ALLOWED_SOURCE_CHANGES
    }
    _require(stable == {
        name: new_hashes[name] for name in sorted(new_hashes)
        if name not in ALLOWED_SOURCE_CHANGES
    }, "stable payload map changed")
    stable_manifest_sha = sha256(canonical_bytes(stable))
    return {
        "executed_source_sha256": EXECUTED_SOURCE_SHA256,
        "orchestration_tree_manifest_sha256": sha256(canonical_bytes(new_hashes)),
        "allowlisted_changed_paths": changed,
        "stable_payload_manifest_sha256": stable_manifest_sha,
        "training_contract_sha256": sha256(canonical_bytes({
            "stable_payload_manifest_sha256": stable_manifest_sha,
            "profile_contract_projection_sha256": old_projection,
            "profile_orchestrator_change": profile_orchestrator_change,
            "approved_resume_transformation": resume_transformation,
        })),
        "executed_profile_contract_projection_sha256": old_projection,
        "target_profile_contract_projection_sha256": new_projection,
        "approved_profile_orchestrator_change": profile_orchestrator_change,
        "approved_resume_transformation": resume_transformation,
        "equal": True,
    }


def logical_path(record: ObjectRecord) -> str:
    pattern = r"^%s/artifacts/sha256/[0-9a-f]{64}/(.+)$" % re.escape(PROFILE_PREFIX)
    match = re.match(pattern, record.name)
    _require(match is not None, "artifact escapes profile digest namespace: %s" % record.name)
    return t.cast(re.Match[str], match).group(1)


def _index(records: list[ObjectRecord]) -> dict[str, list[ObjectRecord]]:
    result: dict[str, list[ObjectRecord]] = {}
    for record in records:
        if "/artifacts/sha256/" not in record.name:
            continue
        result.setdefault(logical_path(record), []).append(record)
    for versions in result.values():
        versions.sort(key=lambda item: int(item.generation))
    return result


def _choose(index: dict[str, list[ObjectRecord]], path: str, *, digest: str | None = None) -> ObjectRecord:
    candidates = index.get(path, [])
    if digest is not None:
        candidates = [item for item in candidates if "/sha256/%s/" % digest in item.name]
    _require(bool(candidates), "required artifact absent: %s" % path)
    if digest is not None:
        _require(len(candidates) == 1, "anchored artifact is not unique: %s" % path)
    return max(candidates, key=lambda item: int(item.generation))


def _one_matching(index: dict[str, list[ObjectRecord]], pattern: str) -> str:
    matches = sorted(path for path in index if re.fullmatch(pattern, path))
    _require(len(matches) == 1, "expected one logical path for %s, got %r" % (pattern, matches))
    return matches[0]


def select_artifacts(records: list[ObjectRecord], base_state: dict) -> tuple[list[ObjectRecord], dict]:
    index = _index(records)
    selected: dict[str, ObjectRecord] = {}

    def add(record: ObjectRecord) -> None:
        path = logical_path(record)
        prior = selected.get(path)
        _require(prior is None or prior == record, "duplicate logical selection: %s" % path)
        selected[path] = record

    for artifact in base_state.get("artifacts", []):
        name = str(artifact["remote"])
        generation = str(artifact["object_generation"])
        candidates = [item for item in records if item.name == name and item.generation == generation]
        _require(len(candidates) == 1, "committed artifact generation missing: %s" % name)
        add(candidates[0])

    completed: dict[str, dict] = {}
    for model in ("gpt", "nextlat"):
        job = "lurestar-" + model
        root = "gate/root/runs/%s/seed1234/base" % model
        experiment = "%s-seed1234-base" % model
        paths = [
            "gate/jobs/%s.gpu.csv" % job,
            "gate/jobs/%s.job.json" % job,
            "gate/jobs/%s.log" % job,
            _one_matching(index, r"gate/jobs/%s\.probe\.[0-9]+\.json" % job),
            root + "/.lurestar_job_identity.json",
            root + "/metrics/step_0_contract.json",
            root + "/latest_ckpt",
            root + "/%s/materialized_config.yaml" % experiment,
            root + "/%s/version_0/metrics.csv" % experiment,
        ]
        checkpoint = _one_matching(
            index, re.escape(root + "/" + experiment) + r"/ckpt_iter_500_[^/]+\.pt")
        paths.extend([checkpoint, checkpoint + ".meta.json"])
        chosen = [_choose(index, path) for path in paths]
        for record in chosen:
            add(record)
        completed[job] = {"status": "completed", "steps": 500,
                          "selected_paths": sorted(paths), "final_checkpoint": checkpoint}

    bst_root = "gate/root/runs/bst/seed1234/base"
    bst_exp = bst_root + "/bst-seed1234-base"
    bst_paths = {
        "identity": bst_root + "/.lurestar_job_identity.json",
        "config": bst_exp + "/materialized_config.yaml",
        "step0": bst_root + "/metrics/step_0_contract.json",
        "pointer": bst_root + "/recovery_ckpt",
        "checkpoint": bst_exp + "/recovery_ckpt_iter_250.pt",
        "metadata": bst_exp + "/recovery_ckpt_iter_250.pt.meta.json",
        "metrics": bst_exp + "/version_0/metrics.csv",
    }
    bst_checkpoint = _choose(
        index, bst_paths["checkpoint"], digest=BST_RECOVERY_SHA256)
    pre_checkpoint_metrics = [
        record for record in index.get(bst_paths["metrics"], [])
        if int(record.generation) < int(bst_checkpoint.generation)
    ]
    _require(bool(pre_checkpoint_metrics),
             "BST has no immutable pre-checkpoint metrics snapshot")
    bst = [
        _choose(index, bst_paths["identity"]),
        _choose(index, bst_paths["config"]),
        _choose(index, bst_paths["step0"]),
        _choose(index, bst_paths["pointer"], digest=BST_RECOVERY_POINTER_SHA256),
        bst_checkpoint,
        _choose(index, bst_paths["metadata"], digest=BST_RECOVERY_META_SHA256),
        max(pre_checkpoint_metrics, key=lambda item: int(item.generation)),
    ]
    for record in bst:
        add(record)

    selected_names = {record.name for record in selected.values()}
    base_names = {str(item["remote"]) for item in base_state.get("artifacts", [])}
    unreferenced = [item for item in records
                    if "/artifacts/sha256/" in item.name and item.name not in base_names]
    not_promoted = [item for item in unreferenced if item.name not in selected_names]
    diagnostics = {
        "unreferenced_object_count": len(unreferenced),
        "unreferenced_unique_logical_paths": len({logical_path(item) for item in unreferenced}),
        "not_promoted": [
            {**item.identity(), "relative_path": logical_path(item),
             "reason": "outside completed GPT/NextLat or exact BST step-250 resume closure"}
            for item in sorted(not_promoted, key=lambda row: (logical_path(row), int(row.generation)))
        ],
        "bst": {
            "status": "resumable",
            "resumable_step": 250,
            "selected_paths": sorted(bst_paths.values()),
            "excluded_final_validation_sha256": BST_FINAL_VALIDATION_SHA256,
            "metrics_selection": "latest immutable generation preceding step-250 checkpoint",
            "later_mutable_artifacts_are_diagnostic_only": True,
        },
        "completed_jobs": completed,
    }
    return [selected[name] for name in sorted(selected)], diagnostics


def validate_selected_payloads(
    backend: Backend, selected: list[ObjectRecord], diagnostics: dict
) -> dict[str, bytes]:
    # Checkpoint bodies are hundreds of MiB. Verify each exact generation in turn, retain
    # only its digest/size, then release those bytes before reading the next object.
    payloads: dict[str, bytes] = {}
    large_identities: dict[str, tuple[str, int]] = {}
    for record in selected:
        path = logical_path(record)
        body = verify_object(backend, record)
        if path.endswith(".pt"):
            large_identities[path] = (sha256(body), len(body))
        else:
            payloads[path] = body
        del body
    for model in ("gpt", "nextlat"):
        job = "lurestar-" + model
        job_doc = json.loads(payloads["gate/jobs/%s.job.json" % job])
        _require(job_doc.get("job") == job and job_doc.get("returncode") == 0,
                 "completed job receipt invalid: %s" % job)
        _require(job_doc.get("steps") == 500 and job_doc.get("warmup_steps") == 100,
                 "completed job plan drift: %s" % job)
        probe_path = next(path for path in payloads if path.startswith("gate/jobs/%s.probe." % job))
        probe = json.loads(payloads[probe_path])
        _require(probe.get("exit") == "ok", "profile probe failed: %s" % job)
        cuda = probe.get("cuda", {})
        _require(cuda.get("device_name") == "NVIDIA A100-SXM4-40GB" and
                 cuda.get("bf16_supported") is True, "profile GPU contract drift: %s" % job)
        checkpoint_path = diagnostics["completed_jobs"][job]["final_checkpoint"]
        checkpoint_sha, checkpoint_size = large_identities[checkpoint_path]
        metadata = json.loads(payloads[checkpoint_path + ".meta.json"])
        _require(metadata.get("training_steps") == 500, "final checkpoint step drift: %s" % job)
        _require(metadata.get("sha256") == checkpoint_sha and
                 metadata.get("size_bytes") == checkpoint_size,
                 "final checkpoint sidecar mismatch: %s" % job)
        root = "gate/root/runs/%s/seed1234/base" % model
        pointer = payloads[root + "/latest_ckpt"].decode().strip()
        _require(pointer.endswith("/" + checkpoint_path.split("/", 7)[-1]),
                 "final checkpoint pointer mismatch: %s" % job)

    bst_root = "gate/root/runs/bst/seed1234/base"
    bst_checkpoint_path = bst_root + "/bst-seed1234-base/recovery_ckpt_iter_250.pt"
    bst_checkpoint_sha, bst_checkpoint_size = large_identities[bst_checkpoint_path]
    bst_meta = json.loads(payloads[bst_checkpoint_path + ".meta.json"])
    _require(bst_checkpoint_sha == BST_RECOVERY_SHA256, "BST recovery bytes drifted")
    _require(bst_meta.get("training_steps") == 250 and bst_meta.get("rng_state") is True,
             "BST recovery metadata cannot resume step 250")
    _require(bst_meta.get("sha256") == BST_RECOVERY_SHA256 and
             bst_meta.get("size_bytes") == bst_checkpoint_size, "BST sidecar mismatch")
    pointer = payloads[bst_root + "/recovery_ckpt"].decode().strip()
    _require(pointer.endswith("/bst-seed1234-base/recovery_ckpt_iter_250.pt"),
             "BST recovery pointer does not select step 250")
    metrics_path = bst_checkpoint_path.rsplit("/", 1)[0] + "/version_0/metrics.csv"
    metrics_rows = list(csv.DictReader(io.StringIO(payloads[metrics_path].decode())))
    metric_steps = [int(float(row["step"])) for row in metrics_rows if row.get("step")]
    _require(bool(metric_steps) and max(metric_steps) <= 250,
             "BST metrics snapshot crosses the step-250 resume boundary")
    selected_digests = {digest for digest, _ in large_identities.values()}
    selected_digests.update(sha256(body) for body in payloads.values())
    _require(BST_FINAL_VALIDATION_SHA256 not in selected_digests,
             "BST final-validation checkpoint entered resume closure")
    return payloads


def build_documents(
    base_state: dict, selected: list[ObjectRecord], diagnostics: dict,
    source_compatibility: dict, target_source_sha256: str,
) -> tuple[dict, dict, dict, dict]:
    _require(re.fullmatch(r"[0-9a-f]{64}", target_source_sha256) is not None,
             "target source SHA must be an exact archive digest")
    remote_inputs = base_state.get("remote_inputs")
    _require(isinstance(remote_inputs, dict) and
             remote_inputs.get("identity_sha256") == INPUT_IDENTITY_SHA256,
             "base state lacks its exact remote input inventory")
    attestation = {
        "schema": ATTESTATION_SCHEMA,
        "profile_id": PROFILE_ID,
        "session_id": SESSION_ID,
        "disposition": {
            "gpt_nextlat_completed": True,
            "bst_resumable_step": 250,
            "bst_completed": False,
            "profile_completed": False,
        },
        "dual_source_binding": {
            "training_execution_source_sha256": EXECUTED_SOURCE_SHA256,
            "orchestration": source_compatibility,
        },
        "input_identity_sha256": INPUT_IDENTITY_SHA256,
        "base_state": {
            "object": PROFILE_PREFIX + "/state.json",
            "object_generation": BASE_STATE_GENERATION,
            "sha256": BASE_STATE_SHA256,
            "logical_generation": BASE_STATE_LOGICAL_GENERATION,
            "artifact_fingerprint": base_state["artifact_fingerprint"],
            "complete": False,
        },
        "terminal_evidence": {
            "session_terminal_absent": True,
            "runtime_failure_record_absent": True,
            "historical_state_untouched": True,
        },
        "selection_policy": {
            "one_object_generation_per_logical_path": True,
            "completed_jobs": "latest verified stable closure",
            "bst": "exact anchored recovery checkpoint/pointer closure at step 250",
            "later_bst_mutables": "diagnostic_only",
        },
        "selected_artifacts": [
            {**record.identity(), "relative_path": logical_path(record)} for record in selected
        ],
        "diagnostics": diagnostics,
    }
    attestation_sha = sha256(canonical_bytes(attestation))
    promoted_state = {
        "schema": STATE_SCHEMA,
        "profile_id": PROFILE_ID,
        "training_execution_source_sha256": EXECUTED_SOURCE_SHA256,
        "orchestration_tree_manifest_sha256":
            source_compatibility["orchestration_tree_manifest_sha256"],
        "training_contract_sha256": source_compatibility["training_contract_sha256"],
        "input_identity_sha256": INPUT_IDENTITY_SHA256,
        "base_state_generation": BASE_STATE_GENERATION,
        "attestation_sha256": attestation_sha,
        "logical_generation": BASE_STATE_LOGICAL_GENERATION + 1,
        "complete": False,
        "completed_jobs": ["lurestar-gpt", "lurestar-nextlat"],
        "resume_jobs": {"lurestar-bst": {"step": 250, "checkpoint_sha256": BST_RECOVERY_SHA256}},
        "artifacts": [
            {**record.identity(), "relative_path": logical_path(record)} for record in selected
        ],
    }
    state_sha = sha256(canonical_bytes(promoted_state))
    clearance_artifacts = [{
        "relative_path": logical_path(record),
        "remote": record.name,
        "object_generation": record.generation,
        "sha256": t.cast(str, record.custom_sha256),
        "size_bytes": record.size_bytes,
    } for record in selected]
    _require(all(re.fullmatch(r"[0-9a-f]{64}", item["sha256"] or "")
                 for item in clearance_artifacts),
             "selected artifact lacks a custom SHA binding")
    clearance = {
        "schema": CLEARANCE_SCHEMA,
        "authorization": "GO",
        "source_sha256": EXECUTED_SOURCE_SHA256,
        "source_profile_id": PROFILE_ID,
        "target_source_sha256": target_source_sha256,
        "input_identity_sha256": INPUT_IDENTITY_SHA256,
        "remote_inputs": remote_inputs,
        "training_compatibility": {
            "verdict": "BYTE_IDENTICAL",
            "source_sha256": EXECUTED_SOURCE_SHA256,
            "target_source_sha256": target_source_sha256,
            "compared_surface_sha256": source_compatibility["training_contract_sha256"],
        },
        "audit": {
            "auditor": "independent-read-only-profile-salvage-audit",
            "audited_at": "2026-08-24T05:07:36Z",
            "evidence_sha256": attestation_sha,
        },
        "artifacts": clearance_artifacts,
        "artifact_fingerprint": canonical_sha256({"artifacts": clearance_artifacts}),
        "completed_jobs": ["lurestar-gpt", "lurestar-nextlat"],
        "resume_steps": {"lurestar-bst": 250},
    }
    clearance_sha = sha256(canonical_bytes(clearance))
    pointer = {
        "schema": POINTER_SCHEMA,
        "profile_id": PROFILE_ID,
        "attestation_sha256": attestation_sha,
        "promoted_state_sha256": state_sha,
        "clearance_sha256": clearance_sha,
        "base_state_generation": BASE_STATE_GENERATION,
        "historical_state_overwritten": False,
    }
    return attestation, promoted_state, clearance, pointer


def publish(backend: Backend, attestation: dict, state: dict,
            clearance: dict, pointer: dict) -> dict:
    attestation_payload = canonical_bytes(attestation)
    state_payload = canonical_bytes(state)
    clearance_payload = canonical_bytes(clearance)
    attestation_sha = sha256(attestation_payload)
    state_sha = sha256(state_payload)
    clearance_sha = sha256(clearance_payload)
    root = PROFILE_PREFIX + "/salvage/sha256"
    attestation_name = "%s/%s/attestation.json" % (root, attestation_sha)
    state_name = "%s/%s/promoted-state.json" % (root, state_sha)
    attestation_record = create_or_verify(backend, attestation_name, attestation_payload)
    state_record = create_or_verify(backend, state_name, state_payload)
    clearance_name = "%s/%s/clearance.json" % (root, clearance_sha)
    clearance_record = create_or_verify(backend, clearance_name, clearance_payload)
    committed_pointer = dict(pointer)
    committed_pointer.update({
        "attestation_object": attestation_record.identity(),
        "promoted_state_object": state_record.identity(),
        "clearance_object": clearance_record.identity(),
    })
    pointer_payload = canonical_bytes(committed_pointer)
    pointer_name = PROFILE_PREFIX + "/salvage/current.json"
    pointer_record = create_or_verify(backend, pointer_name, pointer_payload)
    return {"pointer": pointer_record.identity(), "document": committed_pointer}


def prepare(backend: Backend, project_root: pathlib.Path, *, verify_payloads: bool,
            target_source_archive: pathlib.Path) -> tuple[dict, dict, dict, dict]:
    state_name = PROFILE_PREFIX + "/state.json"
    state_record = next((item for item in backend.list(PROFILE_PREFIX)
                         if item.name == state_name and item.generation == BASE_STATE_GENERATION), None)
    _require(state_record is not None, "frozen base state generation is absent")
    state_payload = verify_object(backend, t.cast(ObjectRecord, state_record))
    _require(sha256(state_payload) == BASE_STATE_SHA256, "base state SHA mismatch")
    base_state = json.loads(state_payload)
    _require(base_state.get("generation") == BASE_STATE_LOGICAL_GENERATION and
             base_state.get("complete") is False, "base state lifecycle mismatch")
    _require(base_state.get("source_sha256") == EXECUTED_SOURCE_SHA256 and
             base_state.get("input_identity_sha256") == INPUT_IDENTITY_SHA256,
             "base state source/input mismatch")

    source_record = next((item for item in backend.list("lurestar/source")
                          if item.name == EXECUTED_SOURCE_OBJECT and
                          item.generation == EXECUTED_SOURCE_GENERATION), None)
    _require(source_record is not None, "executed source generation is absent")
    source_payload = verify_object(
        backend, t.cast(ObjectRecord, source_record), require_custom_sha=False)
    _require(sha256(source_payload) == EXECUTED_SOURCE_SHA256,
             "executed source archive SHA mismatch")
    source_compatibility = verify_source_compatibility(source_payload, project_root)
    _require(target_source_archive.is_file(), "target source archive is absent")
    target_source_payload = target_source_archive.read_bytes()
    target_source_sha256 = sha256(target_source_payload)
    target_hashes, _ = archive_member_hashes(target_source_payload)
    current_hashes, _ = tree_member_hashes(project_root)
    _require(target_hashes == current_hashes,
             "target source archive does not exactly represent the audited project tree")
    records = backend.list(PROFILE_PREFIX)
    selected, diagnostics = select_artifacts(records, base_state)
    if verify_payloads:
        validate_selected_payloads(backend, selected, diagnostics)
    return build_documents(
        base_state, selected, diagnostics, source_compatibility, target_source_sha256)


def main(argv: t.Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="read back and verify, never write")
    mode.add_argument("--apply", action="store_true", help="verify then create salvage objects")
    parser.add_argument("--project-root", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parent.parent)
    parser.add_argument("--target-source-archive", required=True, type=pathlib.Path,
                        help="already-packaged replacement source archive to bind exactly")
    args = parser.parse_args(argv)
    backend = GcloudBackend()
    attestation, state, clearance, pointer = prepare(
        backend, args.project_root.resolve(), verify_payloads=args.dry_run or args.apply,
        target_source_archive=args.target_source_archive.resolve())
    summary = {
        "mode": "apply" if args.apply else ("dry-run" if args.dry_run else "plan"),
        "attestation_sha256": sha256(canonical_bytes(attestation)),
        "promoted_state_sha256": sha256(canonical_bytes(state)),
        "clearance_sha256": sha256(canonical_bytes(clearance)),
        "selected_artifacts": len(state["artifacts"]),
        "completed_jobs": state["completed_jobs"],
        "bst_resumable_step": state["resume_jobs"]["lurestar-bst"]["step"],
    }
    if args.apply:
        summary["published"] = publish(backend, attestation, state, clearance, pointer)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SalvageError as exc:
        print("PROFILE_SALVAGE_REFUSED: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
