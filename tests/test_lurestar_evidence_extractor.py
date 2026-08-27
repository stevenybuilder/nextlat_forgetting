from __future__ import annotations

import ast
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "extract_lurestar_evidence", ROOT / "scripts/extract_lurestar_evidence.py"
)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def _record(path: pathlib.Path) -> dict:
    return {"path": str(path), "sha256": M.sha256_file(path)}


def _block_record() -> dict:
    record = _record(M.H3_BLOCK_PATH)
    record["sidecar"] = _record(pathlib.Path(f"{M.H3_BLOCK_PATH}.sha256"))
    return record


def test_chunk_store_is_atomic_restartable_and_hash_verified(tmp_path):
    store = M.ChunkStore(tmp_path / "progress", job_sha256="a" * 64)
    calls = []

    def produce():
        calls.append(1)
        return {"x": np.arange(4), "y": np.ones((4, 2))}

    first = store.produce("h1", 0, 4, produce)
    second = store.produce("h1", 0, 4, produce)
    assert calls == [1]
    assert np.array_equal(first["x"], second["x"])
    chunk, _receipt = store.paths("h1", 0, 4)
    chunk.write_bytes(chunk.read_bytes() + b"tamper")
    with pytest.raises(M.ExtractionRefused, match="fails SHA-256"):
        store.load("h1", 0, 4)


def test_progress_directory_refuses_a_different_job(tmp_path):
    M.ChunkStore(tmp_path, job_sha256="a" * 64)
    with pytest.raises(M.ExtractionRefused, match="different extraction job"):
        M.ChunkStore(tmp_path, job_sha256="b" * 64)


def test_greedy_paths_use_explicit_argmax_and_return_auditable_tokens(monkeypatch):
    torch = pytest.importorskip("torch")
    target = np.asarray([7, 8, 9, 10, 11], dtype=np.int64)
    tokens = np.zeros((3, 69), dtype=np.int64)
    tokens[:, 63:68] = target

    class Wrapper:
        model = object()

        def generate(self, *_args, **_kwargs):
            raise AssertionError("stochastic upstream generate must not be called")

    def forward(_inner, sequence, *, architecture):
        step = sequence.shape[1] - 63
        logits = torch.zeros((sequence.shape[0], sequence.shape[1], 106), device=sequence.device)
        logits[:, -1, int(target[step])] = 10.0
        return {"logits": logits, "hidden": torch.zeros_like(logits[..., :6])}

    monkeypatch.setattr(M.R, "forward_all_states", forward)
    generated, truth, indicator = M._greedy_exact_path(
        Wrapper(), "gpt", tokens, batch_size=2, device="cpu"
    )
    assert generated.shape == truth.shape == (3, 5)
    assert np.array_equal(generated, truth)
    assert indicator.tolist() == [1.0, 1.0, 1.0]


def _bound_job(tmp_path: pathlib.Path) -> M.BoundJob:
    job_path = tmp_path / "job.json"
    job_path.write_text("{}")
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"base")
    config = tmp_path / "base.yaml"
    config.write_text("base")
    return M.BoundJob(
        path=job_path,
        digest=M.sha256_file(job_path),
        payload={
            "extraction": {
                "whitener_count": 400,
                "scored_count": 1600,
            },
            "local_measurement_source_sha256": {
                relative: M.sha256_file(ROOT / relative)
                for relative in M.LOCAL_MEASUREMENT_SOURCE_PATHS
            },
        },
        arm="gpt",
        seed=1234,
        upstream=tmp_path,
        configs={"base": _record(config)},
        checkpoints={"base": _record(checkpoint)},
        inputs={},
        h3_permanent_block={
            "path": str(M.H3_BLOCK_PATH),
            "sha256": M.H3_BLOCK_SHA256,
            "sidecar_path": str(M.H3_BLOCK_PATH) + ".sha256",
            "sidecar_sha256": M.sha256_file(str(M.H3_BLOCK_PATH) + ".sha256"),
        },
    )


def _h1_raw(seed=4):
    rng = np.random.default_rng(seed)
    n, d = 2000, 6
    base = rng.normal(size=(n, d))
    out = {"h1_item_ids": np.asarray([f"{i:064x}" for i in range(n)], dtype="U64")}
    for index, condition in enumerate(M.H1_CONDITIONS):
        state = base if condition == "base" else base + (index + 1) * rng.normal(
            scale=.05, size=(n, d)
        )
        out[f"{condition}_hidden_psi"] = state
        out[f"{condition}_hidden_branch"] = state + 0.01 * rng.normal(size=(n, d))
        out[f"{condition}_hidden_intermediate"] = rng.normal(size=(n, 12, 2, d))
        out[f"{condition}_margin"] = rng.normal(size=n)
        out[f"{condition}_first_branch_accuracy"] = rng.integers(0, 2, size=n).astype(float)
        true_path = rng.integers(0, 100, size=(n, 5), dtype=np.int64)
        out[f"{condition}_true_path"] = true_path
        out[f"{condition}_generated_path"] = true_path.copy()
        out[f"{condition}_exact_path_accuracy"] = np.ones(n)
    return out


def test_assembly_emits_npsi_whitener_audit_and_labeled_secondaries(tmp_path):
    job = _bound_job(tmp_path)
    output = tmp_path / "evidence.npz"
    receipt = M.assemble_evidence(job, _h1_raw(), output)
    with np.load(output, allow_pickle=False) as z:
        required = {
            "npsi", "npsi_whitened", "whitener_shrinkage",
            "whitener_condition_number", "whitener_calibration_ids",
            "whitener_calibration_ids_sha256", "whitener_fit_source_sha256",
            "secondary_index63_d_critical_centered_cosine",
            "secondary_raw_cosine_d_critical",
            "secondary_uncentered_euclidean_d_critical",
            "behavior_base_first_branch_accuracy", "behavior_near_critical_first_branch_accuracy",
            "behavior_base_generated_path", "behavior_base_true_path",
            "secondary_intermediate_status", "secondary_exact_path_status",
            "secondary_intermediate_d_critical_centered_cosine",
            "secondary_intermediate_whitener_fit_ids",
            "secondary_intermediate_whitener_fit_ids_sha256",
            "secondary_intermediate_whitener_n_features",
            "secondary_intermediate_whitener_fit_dtype",
            "secondary_intermediate_whitener_fit_shape",
            "secondary_intermediate_whitener_shrinkage_rule",
            "local_representations_sha256", "local_evaluate_sha256",
        }
        assert required <= set(z.files)
        assert z["h1_item_ids"].size == 1600
        assert z["whitener_calibration_ids"].size == 2000
        assert float(z["npsi"]) == pytest.approx(
            M.E.normalized_psi(z["d_critical"], z["d_safe"])[0]
        )
        assert str(z["secondary_bst_texthead_status"]) == "NOT_APPLICABLE_NON_BST"
        assert z["secondary_intermediate_d_critical_centered_cosine"].shape == (12, 2, 1600)
        assert z["secondary_intermediate_whitener_fit_ids"].shape == (2000,)
        assert z["secondary_intermediate_whitener_fit_ids_sha256"].shape == (12, 2)
        assert z["secondary_intermediate_whitener_fit_shape"].shape == (12, 2, 2)
        assert np.all(z["secondary_intermediate_whitener_fit_dtype"] == "float64-le")
        assert z["behavior_base_generated_path"].shape == (1600, 5)
        assert str(z["h3_permanent_block_sha256"]) == M.H3_BLOCK_SHA256
        assert not any(name.startswith(("near_", "mid_", "far_")) for name in z.files)
        assert not any("gradient" in name or "adaptation" in name for name in z.files)
    assert receipt["population_counts"] == {"h1": 1600, "h3": 0}
    assert receipt["local_measurement_sources"] == {
        relative: {"path": str((ROOT / relative).resolve()), "sha256": M.sha256_file(ROOT / relative)}
        for relative in M.LOCAL_MEASUREMENT_SOURCE_PATHS
    }
    assert receipt["excluded"] == {
        "h3": True, "adaptation_checkpoints": True, "mechanism_probes": True,
        "h3_analysis": True,
    }


def test_identity_hash_is_order_sensitive_and_has_an_unambiguous_delimiter():
    assert M._identity_sha(["ab", "c"]) != M._identity_sha(["a", "bc"])
    assert M._identity_sha(["a", "b"]) != M._identity_sha(["b", "a"])


def test_whitener_fit_source_hash_binds_ordered_ids_shape_dtype_and_bytes():
    states = np.arange(12, dtype=np.float32).reshape(3, 4)
    ids = ["a", "b", "c"]
    digest = M._fit_source_sha(states, ids)
    assert digest != M._fit_source_sha(states, list(reversed(ids)))
    changed = states.copy()
    changed[0, 0] += 1
    assert digest != M._fit_source_sha(changed, ids)
    with pytest.raises(M.ExtractionRefused, match="do not align"):
        M._fit_source_sha(states, ids[:2])


def _job_file(tmp_path: pathlib.Path) -> pathlib.Path:
    upstream = tmp_path / "upstream"
    source_paths = (
        "data/stargraph.py", "models/model_base.py", "models/model_gpt.py",
        "models/model_nextlat.py", "models/model_bst.py",
    )
    for relative in source_paths:
        path = upstream / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    config = tmp_path / "base.yaml"
    config.write_text("seed: 1234\n")
    checkpoint = tmp_path / "base.pt"
    checkpoint.write_bytes(b"base")
    e_lure = tmp_path / "e_lure.jsonl"
    e_lure.write_text('{"item_id": 1}\n')
    payload = {
        "schema": M.JOB_SCHEMA,
        "arm": "gpt",
        "seed": 1234,
        "upstream_commit": M.PINNED_UPSTREAM_COMMIT,
        "upstream_path": str(upstream),
        "upstream_source_sha256": {
            relative: M.sha256_file(upstream / relative) for relative in source_paths
        },
        "local_measurement_source_sha256": {
            relative: M.sha256_file(ROOT / relative)
            for relative in M.LOCAL_MEASUREMENT_SOURCE_PATHS
        },
        "configs": {"base": _record(config)},
        "checkpoints": {"base": _record(checkpoint)},
        "frozen_inputs": {"e_lure": _record(e_lure)},
        "h3_permanent_block": _block_record(),
        "extraction": {
            "whitener_count": 400,
            "scored_count": 1600,
        },
    }
    path = tmp_path / "job.json"
    path.write_text(json.dumps(payload))
    return path


def test_job_binds_only_base_and_exact_canonical_permanent_block(tmp_path):
    path = _job_file(tmp_path)
    job = M.load_job(path)
    M.verify_upstream(job)
    assert set(job.checkpoints) == {"base"}
    assert set(job.configs) == {"base"}
    assert set(job.inputs) == {"e_lure"}
    assert job.h3_permanent_block["sha256"] == M.H3_BLOCK_SHA256


def test_job_refuses_mutated_local_measurement_source_binding(tmp_path):
    path = _job_file(tmp_path)
    payload = json.loads(path.read_text())
    payload["local_measurement_source_sha256"][
        "src/lurestar/representations.py"
    ] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(M.ExtractionRefused, match="local measurement source identity failed"):
        M.load_job(path)


@pytest.mark.parametrize("forbidden", ["near", "mid", "far"])
def test_job_refuses_every_adaptation_checkpoint(tmp_path, forbidden):
    path = _job_file(tmp_path)
    payload = json.loads(path.read_text())
    checkpoint = tmp_path / f"{forbidden}.pt"
    checkpoint.write_bytes(forbidden.encode())
    payload["checkpoints"][forbidden] = _record(checkpoint)
    path.write_text(json.dumps(payload))
    with pytest.raises(M.ExtractionRefused, match="only the base"):
        M.load_job(path)


@pytest.mark.parametrize("forbidden", ["h3_pairs", "near_gradient_probe", "adaptation_trainer"])
def test_job_refuses_h3_or_mechanism_frozen_inputs(tmp_path, forbidden):
    path = _job_file(tmp_path)
    payload = json.loads(path.read_text())
    artifact = tmp_path / forbidden
    artifact.write_text(forbidden)
    payload["frozen_inputs"][forbidden] = _record(artifact)
    path.write_text(json.dumps(payload))
    with pytest.raises(M.ExtractionRefused, match="only the E_lure"):
        M.load_job(path)


def test_job_refuses_noncanonical_block_hash(tmp_path):
    path = _job_file(tmp_path)
    payload = json.loads(path.read_text())
    payload["h3_permanent_block"]["sha256"] = "0" * 64
    path.write_text(json.dumps(payload))
    with pytest.raises(M.ExtractionRefused, match="SHA-256 mismatch"):
        M.load_job(path)


def test_production_source_has_no_retired_h3_execution_surface_or_legacy_bindings():
    source = (ROOT / "scripts/extract_lurestar_evidence.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_names = {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert function_names.isdisjoint({
        "_validated_h3_row", "_h3_chunk", "_gradient_chunk", "_shadow_update_delta",
        "_exact_path", "_cosine_drift", "_forward_grad", "_parameters",
    })
    for retired_binding in (
        "retired/lurestar_evidence/2", "EXACT_ADAPTATION_LOSS", "gradient_controls",
        "near_checkpoint_sha256", "mid_checkpoint_sha256", "far_checkpoint_sha256",
        "actual_update_contract_sha256", "jacobian_projection_seed",
        "jacobian_projection_count", "_extraction_optimizer_state",
        "_extraction_training_steps",
    ):
        assert retired_binding not in source
