from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "upload_frozen_inputs", ROOT / "scripts/upload_frozen_inputs.py"
)
U = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = U
SPEC.loader.exec_module(U)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def make_project(tmp_path: pathlib.Path, files: dict[str, bytes] | None = None) -> pathlib.Path:
    root = tmp_path / "project"
    files = files or {
        "data/hmm/a.npy": b"hmm",
        "data/hmm_family/r1/train.npy": b"family",
        "manifests/a.json": b"{}\n",
    }
    for relative, payload in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    inventory = root / U.INVENTORY_RELATIVE
    inventory.parent.mkdir(parents=True, exist_ok=True)
    inventory.write_text("".join(
        "%s  %s\n" % (digest(payload), relative)
        for relative, payload in sorted(files.items())
    ))
    return root


class FakeBackend:
    def __init__(self):
        self.objects = {}
        self.generation = 100
        self.calls = []
        self.corrupt_reads = set()
        self.fail_after_create = set()

    def resolve(self, name):
        self.calls.append(("resolve", name))
        item = self.objects.get(name)
        return None if item is None else item[0]

    def create_file(self, name, local_path, sha256):
        self.calls.append(("create", name, sha256))
        if name in self.objects:
            raise U.UploadError("precondition failed")
        payload = pathlib.Path(local_path).read_bytes()
        self.generation += 1
        record = U.RemoteObject(name, str(self.generation), len(payload), sha256)
        self.objects[name] = (record, payload)
        if name in self.fail_after_create:
            raise U.UploadError("connection dropped after commit")

    def download_exact(self, name, generation, destination):
        self.calls.append(("download", name, generation))
        record, payload = self.objects[name]
        assert generation == record.generation
        if name in self.corrupt_reads:
            payload += b"corrupt"
        pathlib.Path(destination).write_bytes(payload)


class MutatingBackend(FakeBackend):
    def __init__(self, inventory):
        super().__init__()
        self.inventory = inventory
        self.mutated = False

    def create_file(self, name, local_path, sha256):
        super().create_file(name, local_path, sha256)
        if not self.mutated:
            self.inventory.write_bytes(self.inventory.read_bytes() + b"\n")
            self.mutated = True


def test_plan_validates_locally_without_backend_calls_or_receipt(tmp_path):
    root = make_project(tmp_path)
    backend = FakeBackend()
    plan = U.upload_bundle(root, backend, bucket="bucket", plan=True)
    assert plan["status"] == "PLAN"
    assert plan["object_count"] == 3
    assert backend.calls == []
    assert not (root / U.RECEIPT_RELATIVE).exists()
    assert any("/corpus/hmm_family/r1/train.npy" in row["name"] for row in plan["objects"])


@pytest.mark.parametrize("line", [
    "0" * 64 + " data/hmm/a.npy\n",
    "A" * 64 + "  data/hmm/a.npy\n",
    "0" * 64 + "  ../escape\n",
    "0" * 64 + "  configs/not-allowed.yaml\n",
    "0" * 64 + "  manifests/manifest_inventory.sha256\n",
])
def test_strict_inventory_rejects_malformed_or_forbidden_lines(tmp_path, line):
    root = make_project(tmp_path)
    (root / U.INVENTORY_RELATIVE).write_text(line)
    with pytest.raises(U.UploadError):
        U.parse_inventory(root)


def test_rejects_duplicate_unsorted_bad_sha_and_symlink(tmp_path):
    root = make_project(tmp_path)
    inventory = root / U.INVENTORY_RELATIVE
    good = digest(b"hmm")
    inventory.write_text(
        "%s  data/hmm/a.npy\n%s  data/hmm/a.npy\n" % (good, good)
    )
    with pytest.raises(U.UploadError, match="duplicate"):
        U.parse_inventory(root)
    inventory.write_text(
        "%s  manifests/a.json\n%s  data/hmm/a.npy\n" % (digest(b"{}\n"), good)
    )
    with pytest.raises(U.UploadError, match="not sorted"):
        U.parse_inventory(root)
    inventory.write_text("%s  data/hmm/a.npy\n" % ("0" * 64))
    with pytest.raises(U.UploadError, match="SHA mismatch"):
        U.parse_inventory(root)
    (root / "data/hmm/a.npy").unlink()
    (root / "outside").write_bytes(b"hmm")
    (root / "data/hmm/a.npy").symlink_to(root / "outside")
    inventory.write_text("%s  data/hmm/a.npy\n" % good)
    with pytest.raises(U.UploadError, match="symlinked"):
        U.parse_inventory(root)


def test_apply_uploads_payloads_then_inventory_and_writes_atomic_receipt(tmp_path):
    root = make_project(tmp_path)
    backend = FakeBackend()
    result = U.upload_bundle(root, backend, bucket="bucket")
    creates = [call[1] for call in backend.calls if call[0] == "create"]
    assert creates[-1].endswith("/manifests/manifest_inventory.sha256")
    assert all(not name.endswith("manifest_inventory.sha256") for name in creates[:-1])
    assert result["status"] == "COMPLETE"
    assert all(row["generation"].isdigit() for row in result["objects"])
    assert result["commit"]["generation"].isdigit()
    receipt = json.loads((root / U.RECEIPT_RELATIVE).read_text())
    assert receipt == result


def test_interrupted_precommit_upload_reuses_exact_objects(tmp_path):
    root = make_project(tmp_path)
    backend = FakeBackend()
    _, bundle_sha, entries = U.parse_inventory(root)
    first_name = "lurestar/input_bundles/%s/%s" % (bundle_sha, entries[0].remote_suffix)
    backend.fail_after_create.add(first_name)
    result = U.upload_bundle(root, backend, bucket="bucket")
    assert result["status"] == "COMPLETE"
    assert [c[1] for c in backend.calls if c[0] == "create"].count(first_name) == 1


def test_committed_bundle_may_be_verified_but_never_filled(tmp_path):
    root = make_project(tmp_path)
    backend = FakeBackend()
    first = U.upload_bundle(root, backend, bucket="bucket")
    backend.calls.clear()
    assert U.upload_bundle(root, backend, bucket="bucket") == first
    assert not [call for call in backend.calls if call[0] == "create"]
    missing_name = first["objects"][0]["name"]
    del backend.objects[missing_name]
    with pytest.raises(U.UploadError, match="committed bundle is missing"):
        U.upload_bundle(root, backend, bucket="bucket")
    assert not [call for call in backend.calls if call[0] == "create"]


@pytest.mark.parametrize("mutation", ["metadata", "size", "readback"])
def test_remote_collision_or_readback_mismatch_fails_closed(tmp_path, mutation):
    root = make_project(tmp_path)
    backend = FakeBackend()
    _, bundle_sha, entries = U.parse_inventory(root)
    entry = entries[0]
    name = "lurestar/input_bundles/%s/%s" % (bundle_sha, entry.remote_suffix)
    record = U.RemoteObject(name, "9", entry.size_bytes, entry.sha256)
    if mutation == "metadata":
        record = U.RemoteObject(name, "9", entry.size_bytes, "0" * 64)
    elif mutation == "size":
        record = U.RemoteObject(name, "9", entry.size_bytes + 1, entry.sha256)
    backend.objects[name] = (record, entry.local_path.read_bytes())
    if mutation == "readback":
        backend.corrupt_reads.add(name)
    with pytest.raises(U.UploadError):
        U.upload_bundle(root, backend, bucket="bucket")
    assert not (root / U.RECEIPT_RELATIVE).exists()


def test_differing_existing_receipt_is_never_overwritten(tmp_path):
    root = make_project(tmp_path)
    receipt = root / U.RECEIPT_RELATIVE
    receipt.parent.mkdir(parents=True)
    receipt.write_text('{"different":true}\n')
    with pytest.raises(U.UploadError, match="receipt differs"):
        U.upload_bundle(root, FakeBackend(), bucket="bucket")
    assert receipt.read_text() == '{"different":true}\n'


def test_inventory_change_during_upload_prevents_commit_and_receipt(tmp_path):
    root = make_project(tmp_path)
    backend = MutatingBackend(root / U.INVENTORY_RELATIVE)
    with pytest.raises(U.UploadError, match="inventory changed"):
        U.upload_bundle(root, backend, bucket="bucket")
    assert not any(name.endswith("manifest_inventory.sha256")
                   for name in backend.objects)
    assert not (root / U.RECEIPT_RELATIVE).exists()


def test_gcloud_backend_uses_generation_precondition_metadata_and_exact_read(tmp_path):
    calls = []

    def runner(argv, **kwargs):
        calls.append(argv)
        if "describe" in argv:
            payload = json.dumps({
                "name": "prefix/object", "generation": "123", "size": "3",
                "custom_fields": {"sha256": digest(b"abc")},
            }).encode()
            return subprocess.CompletedProcess(argv, 0, payload, b"")
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    backend = U.GcloudBackend("bucket", runner=runner)
    local = tmp_path / "x"
    local.write_bytes(b"abc")
    backend.create_file("prefix/object", local, digest(b"abc"))
    assert "--if-generation-match=0" in calls[0]
    assert "--custom-metadata=sha256=%s" % digest(b"abc") in calls[0]
    record = backend.resolve("prefix/object")
    assert record.generation == "123"
    destination = tmp_path / "out"
    backend.download_exact("prefix/object", "123", destination)
    assert "gs://bucket/prefix/object#123" in calls[-1]


def test_gcloud_describe_distinguishes_absent_from_transport_failure():
    def missing(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, b"", b"ERROR: Not Found")

    assert U.GcloudBackend("bucket", runner=missing).resolve("x") is None

    def broken(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 1, b"", b"authentication failed")

    with pytest.raises(U.UploadError, match="describe failed"):
        U.GcloudBackend("bucket", runner=broken).resolve("x")
