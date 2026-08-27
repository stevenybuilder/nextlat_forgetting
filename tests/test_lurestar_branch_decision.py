"""Prospective D46 H1-BD-1 contract tests; synthetic arrays only, never model outputs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path, monkeypatch):
    # Reuse the existing synthetic base-only evidence fixture.  It creates no checkpoint
    # output or scientific result; all values are test-local random arrays.
    legacy_test = _load("legacy_h1_fixture", "tests/test_lurestar_checkpoint_evaluator.py")
    branch = _load("h1_branch_decision", "scripts/evaluate_lurestar_branch_decision.py")
    seeds = (11, 12)
    monkeypatch.setattr(branch, "CANONICAL_SEEDS", seeds)
    return branch, legacy_test._manifest(tmp_path, seeds=seeds), seeds


def _rewrite_evidence(manifest_path: Path, array_key: str, value: float) -> None:
    manifest = json.loads(manifest_path.read_text())
    cell = manifest["cells"][0]
    evidence = Path(cell["evidence_npz"])
    with np.load(evidence, allow_pickle=False) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}
    arrays[array_key] = np.full_like(arrays[array_key], value, dtype=np.float64)
    np.savez(evidence, **arrays)
    cell["evidence_sha256"] = hashlib.sha256(evidence.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


def test_h1bd_is_a_complete_prospective_h63_analysis_and_keeps_h62_control(tmp_path, monkeypatch):
    branch, manifest, seeds = _fixture(tmp_path, monkeypatch)
    report, receipt = branch.evaluate_manifest(manifest, expected_seeds=seeds)
    assert report["status"] == "COMPLETE"
    assert report["analysis_id"] == "H1-BD-1"
    assert report["extraction_index"] == 63
    assert report["h1_legacy_control"] == {
        "classification_unmodified": True,
        "extraction_index": 62,
        "role": "originally_registered_delimiter_state_control",
    }
    assert len(report["cells"]) == 3 * len(seeds)
    assert set(report["seed_level_contrasts"]) == {
        "h63_centered_cosine", "h63_whitened_euclidean",
    }
    assert report["metric_name_mapping"]["whitened_euclidean"] == (
        "held-out-whitened Mahalanobis distance "
        "(Euclidean norm after the held-out whitening transform)"
    )
    assert receipt["legacy_h1_h62_unmodified"] is True
    assert receipt["h63_branch_decision_index"] == 63


def test_h1bd_never_reads_or_reports_h62_primary_distance_arrays(tmp_path, monkeypatch):
    branch, manifest, seeds = _fixture(tmp_path, monkeypatch)
    # h62's frozen arrays are deliberately poisoned in this synthetic fixture and its
    # manifest hash is updated. H1-BD still validates/evaluates h63 only; it neither
    # promotes nor uses h62 outcomes to decide the new analysis.
    _rewrite_evidence(manifest, "d_critical", np.nan)
    _rewrite_evidence(manifest, "d_safe", np.nan)
    report, _receipt = branch.evaluate_manifest(manifest, expected_seeds=seeds)
    assert report["extraction_index"] == 63
    assert all(set(cell) == {"identity", "h63_psi", "h63_whitener_audit"}
               for cell in report["cells"])


def test_h1bd_refuses_nonfinite_branch_decision_arrays(tmp_path, monkeypatch):
    branch, manifest, seeds = _fixture(tmp_path, monkeypatch)
    _rewrite_evidence(manifest, "secondary_index63_d_critical_centered_cosine", np.nan)
    with pytest.raises(branch.BranchDecisionRefused, match="h63 arrays are invalid"):
        branch.evaluate_manifest(manifest, expected_seeds=seeds)


def test_h1bd_refuses_a_partial_base_matrix(tmp_path, monkeypatch):
    branch, manifest, seeds = _fixture(tmp_path, monkeypatch)
    document = json.loads(manifest.read_text())
    document["cells"].pop()
    manifest.write_text(json.dumps(document))
    with pytest.raises(branch.BranchDecisionRefused, match="complete frozen 15-cell base matrix"):
        branch.evaluate_manifest(manifest, expected_seeds=seeds)


def test_h1bd_refuses_a_freeze_receipt_that_allows_legacy_h1_mutation(tmp_path, monkeypatch):
    branch, manifest, seeds = _fixture(tmp_path, monkeypatch)
    receipt = json.loads(branch.FREEZE_RECEIPT_PATH.read_text())
    receipt["legacy_h1_unchanged"] = False
    copied = tmp_path / "forged-freeze.json"
    copied.write_text(json.dumps(receipt))
    monkeypatch.setattr(branch, "FREEZE_RECEIPT_PATH", copied)
    with pytest.raises(branch.BranchDecisionRefused, match="freeze receipt semantics changed"):
        branch.evaluate_manifest(manifest, expected_seeds=seeds)
