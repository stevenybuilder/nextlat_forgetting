from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from lurestar import h3_precompute as H  # noqa: E402
import materialize_adaptation_banks as B  # noqa: E402
import score_h3_pilot as S  # noqa: E402
import run_h3_pilot_durable as D  # noqa: E402


class _Missing(Exception):
    code = 404


class _Conflict(Exception):
    code = 412


class _FakeBlob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name
        self.metadata = None
        self.generation = None

    def upload_from_filename(self, filename, if_generation_match):
        assert if_generation_match == 0
        if self.name in self.bucket.objects:
            raise _Conflict()
        generation = self.bucket.next_generation
        self.bucket.next_generation += 1
        self.bucket.objects[self.name] = {
            "payload": pathlib.Path(filename).read_bytes(), "generation": generation,
            "metadata": dict(self.metadata or {}),
        }
        self.generation = generation

    def reload(self):
        if self.name not in self.bucket.objects:
            raise _Missing()
        record = self.bucket.objects[self.name]
        self.generation, self.metadata = record["generation"], dict(record["metadata"])

    def download_as_bytes(self, if_generation_match):
        self.reload()
        assert int(if_generation_match) == int(self.generation)
        return self.bucket.objects[self.name]["payload"]


class _FakeBucket:
    name = "fake-bucket"

    def __init__(self):
        self.objects = {}
        self.next_generation = 1

    def blob(self, name):
        return _FakeBlob(self, name)


def _write(path: pathlib.Path, payload: bytes) -> str:
    return H.create_or_verify(path, payload)


def _rows(path: pathlib.Path, count: int) -> list[dict]:
    return [json.loads(raw) for raw in path.read_text().splitlines()[:count]]


def test_create_or_verify_is_idempotent_and_never_overwrites(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "frozen.json"
    digest = H.create_or_verify(target, b"one\n")
    assert digest == hashlib.sha256(b"one\n").hexdigest()
    assert H.create_or_verify(target, b"one\n") == digest
    with pytest.raises(H.PrecomputeRefused, match="overwrite"):
        H.create_or_verify(target, b"two\n")
    assert target.read_bytes() == b"one\n"


def test_mid_generator_is_solver_verified_varied_and_deterministic(monkeypatch) -> None:
    near = _rows(ROOT / "manifests/b_near.jsonl", 2)
    monkeypatch.setattr(H, "MID_COUNT", 2)
    monkeypatch.setattr(H, "MID_CANDIDATE_COUNT", 6)
    prompts, graphs = H._identity_sets(near)
    first = H.generate_mid_candidates(near, prompts, graphs)
    second = H.generate_mid_candidates(near, prompts, graphs)
    assert first == second
    assert len(first) == 6
    assert all(row["solver_verified"] for row in first)
    assert all(H.prompt_sha(row["line"]) == row["prompt_sha256"] for row in first)
    for i in range(2):
        distances = {round(row["normalized_edge_disagreement"], 12) for row in first[3*i:3*i+3]}
        assert len(distances) == 3


def test_acquisition_candidates_are_independent_and_structurally_ordered(monkeypatch) -> None:
    monkeypatch.setattr(H, "ACQUISITION_CANDIDATE_COUNT", 4)
    pools = H.generate_acquisition_candidates(set(), set())
    assert {key: len(value) for key, value in pools.items()} == {"near": 4, "mid": 4, "far": 4}
    prompts = [row["prompt_sha256"] for rows in pools.values() for row in rows]
    graphs = [row["graph_key"] for rows in pools.values() for row in rows]
    assert len(prompts) == len(set(prompts)) == 12
    assert len(graphs) == len(set(graphs)) == 12
    for index in range(4):
        assert pools["near"][index]["anchor_structural_distance"] < pools["mid"][index]["anchor_structural_distance"]
        assert pools["mid"][index]["anchor_structural_distance"] < pools["far"][index]["anchor_structural_distance"]


def test_pilot_freeze_names_one_checkpoint_and_prohibits_substitution() -> None:
    freeze = H.pilot_freeze_payload(
        generator_sha256="a" * 64, scorer_sha256="b" * 64,
        tokenizer_sha256="c" * 64, adaptation_contract_sha256="d" * 64,
    )
    assert freeze["sole_pilot"] is True
    assert freeze["substitution_or_reselection_permitted"] is False
    assert freeze["checkpoint"]["sha256"] == H.PILOT_CHECKPOINT_SHA256
    assert freeze["profile_state"]["generation"] == 12
    assert freeze["selection_rule"]["confirmatory_checkpoints_permitted"] is False


def test_score_job_refuses_checkpoint_substitution(tmp_path: pathlib.Path) -> None:
    job = tmp_path / "job.json"
    job.write_text(json.dumps({
        "schema": S.JOB_SCHEMA, "role": "non_confirmatory_engineering_pilot",
        "confirmatory_inputs": False, "confirmatory_results": False,
        "model_family": "bst", "seed": 999, "training_step": 500,
    }))
    with pytest.raises(H.PrecomputeRefused, match="substitute"):
        S._load_job(job)


def test_real_score_job_binds_all_53000_inputs_without_loading_model() -> None:
    job = S._load_job((ROOT / ".agent_state/pilot/h3-score-job.json").resolve())
    rows = S._input_rows(job["_bound"])
    assert len(rows) == 53_000
    assert len({row["prompt_sha256"] for row in rows}) == 53_000
    assert job["_bound"]["checkpoint"].name.startswith("bst-seed1234-step500-")


def test_atomic_score_chunk_is_restartable_and_corruption_blocks(tmp_path: pathlib.Path) -> None:
    rows = [{"pool": "b_near", "prompt_sha256": "a" * 64, "line": "unused"}]
    import numpy as np
    S._write_chunk(tmp_path, "b" * 64, rows, np.asarray([0.25]), 0, 1)
    payload = S._verified_chunk(tmp_path, "b" * 64, 0, 1)
    assert payload is not None and b'"loss":0.25' in payload
    path, _receipt = S._chunk_paths(tmp_path, 0, 1)
    path.write_bytes(b"tampered\n")
    with pytest.raises(H.PrecomputeRefused, match="sidecar mismatch"):
        S._verified_chunk(tmp_path, "b" * 64, 0, 1)


def test_gcs_chunk_commit_is_create_only_read_back_and_restorable(tmp_path: pathlib.Path) -> None:
    job_sha, driver_sha = "b" * 64, "c" * 64
    rows = [{"pool": "b_near", "prompt_sha256": "a" * 64, "line": "unused"}]
    import numpy as np
    S._write_chunk(tmp_path, job_sha, rows, np.asarray([0.25]), 0, 1)
    bucket = _FakeBucket()
    durable = D.GcsDurability(
        bucket, base_prefix="lurestar/h3-test", job_sha256=job_sha,
        driver_sha256=driver_sha,
    )
    first = durable.commit_chunk(tmp_path, 0, 1)
    second = durable.commit_chunk(tmp_path, 0, 1)
    assert first == second
    data, receipt = S._chunk_paths(tmp_path, 0, 1)
    data.unlink()
    pathlib.Path(f"{data}.sha256").unlink()
    receipt.unlink()
    pathlib.Path(f"{receipt}.sha256").unlink()
    assert durable.restore_chunk(tmp_path, 0, 1) is True
    assert S._verified_chunk(tmp_path, job_sha, 0, 1) is not None
    assert all(record["metadata"]["job_sha256"] == job_sha for record in bucket.objects.values())


def test_durable_plan_binds_remote_prefix_to_exact_job_sha() -> None:
    job_path = (ROOT / ".agent_state/pilot/h3-score-job.json").resolve()
    result = D.plan(job_path)
    assert result["job_sha256"] == H.sha256_file(job_path)
    assert result["remote_prefix"].endswith("/" + result["job_sha256"])
    assert result["credentials_persisted"] is False


def test_full_selector_outputs_are_accepted_by_materialization_gate(tmp_path: pathlib.Path, monkeypatch) -> None:
    # Shrink only counts; retain the exact frozen algorithms and real solver-valid stimuli.
    monkeypatch.setattr(H, "MID_COUNT", 2)
    monkeypatch.setattr(H, "MID_CANDIDATE_COUNT", 6)
    monkeypatch.setattr(H, "ACQUISITION_COUNT", 10)
    monkeypatch.setattr(H, "ACQUISITION_CANDIDATE_COUNT", 20)
    monkeypatch.setattr(B, "NEAR_COUNT", 2)
    monkeypatch.setattr(B, "MID_COUNT", 2)
    monkeypatch.setattr(B, "FAR_CANDIDATE_COUNT", 6)
    monkeypatch.setattr(B, "MID_CANDIDATE_COUNT", 6)
    monkeypatch.setattr(B, "VALIDATION_COUNT", 10)

    root = tmp_path / "project"
    manifests, candidates, selected = root / "manifests", root / "candidate", root / "selected"
    manifests.mkdir(parents=True)
    near = _rows(ROOT / "manifests/b_near.jsonl", 2)
    far_all = _rows(ROOT / "manifests/b_far.jsonl", 20)
    near_prompts, near_graphs = H._identity_sets(near)
    far = []
    for row in far_all:
        if row["prompt_sha256"] not in near_prompts and row["graph_key"] not in near_graphs:
            far.append(row)
        if len(far) == 6:
            break
    _write(manifests / "b_near.jsonl", H._jsonl_payload(near))
    _write(manifests / "b_far.jsonl", H._jsonl_payload(far))
    mid = H.generate_mid_candidates(near, near_prompts | {r["prompt_sha256"] for r in far}, near_graphs | {r["graph_key"] for r in far})
    _write(candidates / "b_mid_candidates.jsonl", H._jsonl_payload(mid))
    acquisitions = H.generate_acquisition_candidates(
        near_prompts | {r["prompt_sha256"] for r in far + mid},
        near_graphs | {r["graph_key"] for r in far + mid},
    )
    for branch, rows in acquisitions.items():
        _write(candidates / f"acquisition_{branch}_candidates.jsonl", H._jsonl_payload(rows))
    freeze = H.pilot_freeze_payload(
        generator_sha256=H.sha256_file(ROOT / "src/lurestar/h3_precompute.py"),
        scorer_sha256=H.sha256_file(ROOT / "scripts/score_h3_pilot.py"),
        tokenizer_sha256=H.sha256_file(ROOT / "upstream/NextLat/data/stargraph.py"),
        adaptation_contract_sha256=H.sha256_file(ROOT / "src/lurestar/adaptation.py"),
    )
    _write(candidates / "pilot_freeze.json", H.canonical_json(freeze))

    losses = []
    def add(rows, values, pool):
        for row, loss in zip(rows, values):
            losses.append({"schema": H.LOSS_TABLE_SCHEMA, "pool": pool, "prompt_sha256": row["prompt_sha256"], "loss": loss})
    add(near, [0.10, 0.40], "b_near")
    add(far, [0.60, 0.10, 0.40, 0.50, 0.20, 0.30], "b_far")
    # The first candidate paired to each near is in the matching global decile and caliper.
    add(mid, [0.10, 0.20, 0.30, 0.40, 0.50, 0.60], "b_mid")
    for branch, rows in acquisitions.items():
        add(rows, [float(i) for i in range(20)], f"acquisition_{branch}")
    loss_path = tmp_path / "pilot_losses.jsonl"
    _write(loss_path, H._jsonl_payload(losses))

    receipt = H.select_outputs(root=root, candidate_dir=candidates, loss_table=loss_path, output_dir=selected)
    assert receipt["status"] == "H3_SELECTIONS_FROZEN"
    near_items = B.load_manifest(manifests / "b_near.jsonl", 2, "B_near")
    far_items = B.load_manifest(manifests / "b_far.jsonl", 6, "B_far")
    mid_items = B.load_manifest(candidates / "b_mid_candidates.jsonl", 6, "B_mid")
    assert len(B.select_far(near_items, far_items, selected / "far_selection.json",
                            near_sha256=H.verify_sidecar(manifests / "b_near.jsonl"),
                            candidates_sha256=H.verify_sidecar(manifests / "b_far.jsonl"))) == 2
    assert len(B.select_mid(near_items, mid_items, selected / "mid_selection.json",
                            near_sha256=H.verify_sidecar(manifests / "b_near.jsonl"),
                            candidates_sha256=H.verify_sidecar(candidates / "b_mid_candidates.jsonl"))) == 2
    acquisition_hashes = {
        branch: H.verify_sidecar(selected / f"acquisition_{branch}.jsonl")
        for branch in ("near", "mid", "far")
    }
    B.verify_acquisition_provenance(
        selected / "acquisition_provenance.json",
        near_sha256=acquisition_hashes["near"], mid_sha256=acquisition_hashes["mid"],
        far_sha256=acquisition_hashes["far"],
    )
