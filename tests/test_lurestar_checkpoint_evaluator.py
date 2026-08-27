from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import pathlib

import numpy as np
import pytest

from lurestar import evaluate as E
from lurestar.durable_checkpoint import sha256_file


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_lurestar_checkpoints", ROOT / "scripts/evaluate_lurestar_checkpoints.py"
)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(M)


def _record(path: pathlib.Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def _block_record() -> dict:
    record = _record(M.H3_BLOCK_PATH)
    record["sidecar"] = _record(pathlib.Path(f"{M.H3_BLOCK_PATH}.sha256"))
    return record


def _manifest(tmp_path: pathlib.Path, *, seeds=(11, 12)):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(4)
    h1_ids = np.asarray([f"{i:064x}" for i in range(400, 2000)])
    calibration_ids = np.asarray([
        f"{i:064x}:{condition}"
        for condition in ("base", "repeat", "near_safe", "near_critical", "far_critical")
        for i in range(400)
    ])
    local_measurement_sources = {
        relative: _record(ROOT / relative)
        for relative in ("src/lurestar/representations.py", "src/lurestar/evaluate.py")
    }
    identity_domains = {
        "h1_quartet": {
            "count": h1_ids.size,
            "item_ids_sha256": M.item_ids_sha256(h1_ids),
        },
    }
    block_sidecar_sha = sha256_file(pathlib.Path(f"{M.H3_BLOCK_PATH}.sha256"))
    cells = []
    for arm in E.ARMS:
        for seed in seeds:
            checkpoint = tmp_path / f"{arm}-{seed}-base.pt"
            checkpoint.write_bytes(f"{arm}:{seed}:base".encode())
            checkpoint_record = _record(checkpoint)
            h1_critical = rng.uniform(0.05, 0.6, h1_ids.size)
            h1_safe = rng.uniform(0.01, 0.2, h1_ids.size)
            h1_critical_w = h1_critical * 2.0
            h1_safe_w = rng.uniform(0.01, 0.2, h1_ids.size)
            common = {
                "npsi": np.asarray(E.normalized_psi(h1_critical, h1_safe)[0]),
                "npsi_whitened": np.asarray(E.normalized_psi(h1_critical_w, h1_safe_w)[0]),
                "secondary_raw_cosine_d_critical": h1_critical * 0.9,
                "secondary_raw_cosine_d_safe": h1_safe * 0.9,
                "secondary_uncentered_euclidean_d_critical": h1_critical * 1.2,
                "secondary_uncentered_euclidean_d_safe": h1_safe * 1.2,
                "secondary_index63_d_critical_centered_cosine": h1_critical * 0.8,
                "secondary_index63_d_safe_centered_cosine": h1_safe * 0.8,
                "secondary_index63_d_critical_whitened": h1_critical_w * 0.8,
                "secondary_index63_d_safe_whitened": h1_safe_w * 0.8,
                "secondary_intermediate_status": np.asarray(
                    "AVAILABLE_ALL_BLOCKS_0_11_PRE_FINAL_NORM_POSITIONS_62_63"
                ),
                "secondary_exact_path_status": np.asarray("AVAILABLE_EXPLICIT_ARGMAX_5_TOKENS"),
                "secondary_bst_texthead_status": np.asarray(
                    "AVAILABLE_BST_ONLY" if arm == "bst" else "NOT_APPLICABLE_NON_BST"
                ),
                "secondary_intermediate_blocks": np.arange(12),
                "secondary_intermediate_positions": np.asarray([62, 63]),
                "secondary_intermediate_d_critical_centered_cosine": np.broadcast_to(
                    h1_critical, (12, 2, h1_ids.size)
                ).copy(),
                "secondary_intermediate_d_safe_centered_cosine": np.broadcast_to(
                    h1_safe, (12, 2, h1_ids.size)
                ).copy(),
                "secondary_intermediate_d_critical_whitened": np.broadcast_to(
                    h1_critical_w, (12, 2, h1_ids.size)
                ).copy(),
                "secondary_intermediate_d_safe_whitened": np.broadcast_to(
                    h1_safe_w, (12, 2, h1_ids.size)
                ).copy(),
                "secondary_intermediate_whitener_shrinkage": np.full((12, 2), .2),
                "secondary_intermediate_whitener_condition_number": np.full((12, 2), 3.0),
                "secondary_intermediate_whitener_fit_source_sha256": np.full(
                    (12, 2), hashlib.sha256(f"{arm}:{seed}:layers".encode()).hexdigest()
                ),
                "secondary_intermediate_whitener_fit_ids": calibration_ids.copy(),
                "secondary_intermediate_whitener_fit_ids_sha256": np.full(
                    (12, 2), M.item_ids_sha256(calibration_ids)
                ),
                "secondary_intermediate_whitener_n_features": np.full((12, 2), 384),
                "secondary_intermediate_whitener_fit_dtype": np.full(
                    (12, 2), "float64-le"
                ),
                "secondary_intermediate_whitener_fit_shape": np.broadcast_to(
                    np.asarray([2000, 384]), (12, 2, 2)
                ).copy(),
                "secondary_intermediate_whitener_shrinkage_rule": np.full(
                    (12, 2), "ledoit_wolf_with_1e-3_floor"
                ),
                "secondary_intermediate_calibration_base_ids": np.asarray(
                    [f"{i:064x}" for i in range(400)]
                ),
                "secondary_intermediate_calibration_base_ids_sha256": np.asarray(
                    M.item_ids_sha256(np.asarray([f"{i:064x}" for i in range(400)]))
                ),
            }
            for condition in ("base", "repeat", "near_safe", "near_critical", "far_critical"):
                first = rng.integers(0, 2, h1_ids.size).astype(float)
                truth = rng.integers(0, 100, size=(h1_ids.size, 5), dtype=np.int64)
                generated = truth.copy()
                wrong = rng.random(h1_ids.size) < .2
                generated[wrong, 0] = (generated[wrong, 0] + 1) % 100
                common[f"behavior_{condition}_first_branch_accuracy"] = first
                common[f"behavior_{condition}_true_path"] = truth
                common[f"behavior_{condition}_generated_path"] = generated
                common[f"behavior_{condition}_exact_path_accuracy"] = (~wrong).astype(float)
            for prefix in ("whitener", "secondary_index63_whitener"):
                common.update({
                    f"{prefix}_shrinkage": np.asarray(0.2),
                    f"{prefix}_shrinkage_rule": np.asarray("ledoit_wolf_with_1e-3_floor"),
                    f"{prefix}_condition_number": np.asarray(3.0),
                    f"{prefix}_calibration_ids": calibration_ids,
                    f"{prefix}_calibration_ids_sha256": np.asarray(
                        M.item_ids_sha256(calibration_ids)
                    ),
                    f"{prefix}_calibration_base_ids": np.asarray(
                        [f"{i:064x}" for i in range(400)]
                    ),
                    f"{prefix}_calibration_base_ids_sha256": np.asarray(
                        M.item_ids_sha256(np.asarray([f"{i:064x}" for i in range(400)]))
                    ),
                    f"{prefix}_fit_source_sha256": np.asarray(
                        hashlib.sha256(f"{arm}:{seed}:{prefix}".encode()).hexdigest()
                    ),
                    f"{prefix}_n_pool": np.asarray(2000),
                    f"{prefix}_n_features": np.asarray(6),
                })
            if arm == "bst":
                common.update({
                    "secondary_bst_texthead_d_critical_centered_cosine": h1_critical * .7,
                    "secondary_bst_texthead_d_safe_centered_cosine": h1_safe * .7,
                    "secondary_bst_texthead_d_critical_whitened": h1_critical_w * .7,
                    "secondary_bst_texthead_d_safe_whitened": h1_safe_w * .7,
                })
                prefix = "secondary_bst_texthead_whitener"
                common.update({
                    f"{prefix}_shrinkage": np.asarray(0.2),
                    f"{prefix}_shrinkage_rule": np.asarray("ledoit_wolf_with_1e-3_floor"),
                    f"{prefix}_condition_number": np.asarray(3.0),
                    f"{prefix}_calibration_ids": calibration_ids,
                    f"{prefix}_calibration_ids_sha256": np.asarray(
                        M.item_ids_sha256(calibration_ids)
                    ),
                    f"{prefix}_calibration_base_ids": np.asarray(
                        [f"{i:064x}" for i in range(400)]
                    ),
                    f"{prefix}_calibration_base_ids_sha256": np.asarray(
                        M.item_ids_sha256(np.asarray([f"{i:064x}" for i in range(400)]))
                    ),
                    f"{prefix}_fit_source_sha256": np.asarray(
                        hashlib.sha256(f"{arm}:{seed}:{prefix}".encode()).hexdigest()
                    ),
                    f"{prefix}_n_pool": np.asarray(2000),
                    f"{prefix}_n_features": np.asarray(6),
                })
            evidence = tmp_path / f"{arm}-{seed}-evidence.npz"
            np.savez(
                evidence,
                evidence_schema=np.asarray("nextlat_forgetting/lurestar_evidence/4"),
                arm=np.asarray(arm),
                seed=np.asarray(seed),
                base_checkpoint_sha256=np.asarray(checkpoint_record["sha256"]),
                h3_permanent_block_sha256=np.asarray(M.H3_BLOCK_SHA256),
                h3_permanent_block_sidecar_sha256=np.asarray(block_sidecar_sha),
                local_representations_sha256=np.asarray(
                    local_measurement_sources["src/lurestar/representations.py"]["sha256"]
                ),
                local_evaluate_sha256=np.asarray(
                    local_measurement_sources["src/lurestar/evaluate.py"]["sha256"]
                ),
                h1_item_ids=h1_ids,
                h1_item_ids_sha256=np.asarray(M.item_ids_sha256(h1_ids)),
                d_critical=h1_critical,
                d_safe=h1_safe,
                d_repeat=rng.uniform(0.01, 0.2, h1_ids.size),
                d_critical_whitened=h1_critical_w,
                d_safe_whitened=h1_safe_w,
                critical_margin=rng.normal(2.0, 0.2, h1_ids.size) - 0.3 * h1_critical,
                base_margin=rng.normal(2.0, 0.2, h1_ids.size),
                **common,
            )
            cells.append({
                "arm": arm,
                "seed": seed,
                "base_checkpoint": checkpoint_record,
                "evidence_npz": str(evidence),
                "evidence_sha256": sha256_file(evidence),
            })
    extractor = tmp_path / "extractor.py"
    extractor.write_text("# frozen test extractor\n")
    stimulus = tmp_path / "e_lure.jsonl"
    stimulus.write_text('{"item_id": 1}\n')
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "schema": M.SCHEMA,
        "analysis_seed": 73021,
        "expected_arms": list(E.ARMS),
        "expected_seeds": list(seeds),
        "n_boot": 100,
        "identity_domains": identity_domains,
        "h3_permanent_block": _block_record(),
        "extractor": _record(extractor),
        "local_measurement_sources": local_measurement_sources,
        "frozen_inputs": {"e_lure": _record(stimulus)},
        "cells": cells,
    }))
    return path


def test_report_emits_npsi_student_t_iut_and_nested_h2(tmp_path):
    manifest = _manifest(tmp_path)
    report, receipt = M.evaluate_manifest(manifest, expected_seeds=[11, 12], n_boot=100)
    assert report["schema"] == M.REPORT_SCHEMA
    assert len(report["cells"]) == 6
    assert set(report["seed_level_contrasts"]) == {"h1_psi", "h1_psi_whitened"}
    assert report["h3_exclusion"]["h3_analysis_included"] is False
    assert report["h3_exclusion"]["block_sha256"] == M.H3_BLOCK_SHA256
    assert "h3_model_by_distance_interaction" not in report
    for cell in report["cells"]:
        assert set(cell) == {
            "identity", "h1_psi", "h1_psi_whitened", "safe_lure_invariance", "h2",
            "whitener_audit", "mandatory_secondaries", "five_condition_behavior",
        }
        assert np.isfinite(cell["h1_psi"]["npsi"])
        assert cell["h1_psi_whitened"]["metric"] == "whitened_euclidean"
        assert set(cell["h2"]) == {
            "margin_primary", "first_branch_accuracy_secondary", "exact_path_accuracy_secondary",
        }
        h2 = cell["h2"]["margin_primary"]
        for metric in ("centered_cosine", "whitened_euclidean"):
            assert set(("M0", "M1", "delta_r2_heldout")) <= set(h2[metric])
            assert h2[metric]["folds_reused_exactly"] is True
        assert cell["whitener_audit"]["primary"]["n_pool"] == 2000
        assert len(cell["mandatory_secondaries"]["intermediate_layers"]["fixed_inventory"]) == 24
        assert cell["five_condition_behavior"]["base"]["exact_path_accuracy"]["n"] == 1600
        assert cell["h2"]["first_branch_accuracy_secondary"]["centered_cosine"][
            "status"
        ] == "estimable_nested_linear_probability_model"
        assert cell["identity"]["identity_domains"]["h1_quartet"]["count"] == 1600
        assert set(cell["identity"]) == {
            "arm", "seed", "base_checkpoint", "evidence_npz", "identity_domains",
        }
    assert receipt["manifest"]["sha256"] == sha256_file(manifest)
    assert receipt["h3_permanent_block"]["sha256"] == M.H3_BLOCK_SHA256
    assert receipt["local_measurement_sources"] == {
        relative: _record(ROOT / relative)
        for relative in ("src/lurestar/representations.py", "src/lurestar/evaluate.py")
    }
    assert len(receipt["inputs"]) == 6
    assert report["h1_confirmatory_classification"]["rule"].startswith("intersection-union")
    primary = report["seed_level_contrasts"]["h1_psi"][
        "contrasts_in_preregistered_order"
    ][0]
    assert primary["ci"]["method"].startswith("two-sided paired Student-t")
    assert len(primary["leave_one_seed_out"]) == 2


def test_cli_atomically_writes_report_receipt_and_hash_sidecar(tmp_path):
    manifest = _manifest(tmp_path)
    output = tmp_path / "final.json"
    assert M.main([
        "--manifest", str(manifest), "--output", str(output),
        "--seeds", "11", "12", "--n-boot", "100",
    ]) == 0
    receipt_path = tmp_path / "final.json.receipt.json"
    assert output.is_file() and receipt_path.is_file()
    receipt = json.loads(receipt_path.read_text())
    assert receipt["report"]["sha256"] == sha256_file(output)
    assert (tmp_path / "final.json.receipt.json.sha256").read_text().split()[0] == sha256_file(
        receipt_path
    )


def test_h1_intersection_union_classifier_has_all_four_frozen_states():
    seeds = range(5)

    def arm_report(differences):
        return E.three_arm_contrasts(
            {
                "nextlat": {seed: float(differences[seed]) for seed in seeds},
                "bst": {seed: 0.0 for seed in seeds},
                "gpt": {seed: -0.1 for seed in seeds},
            },
            rng=np.random.default_rng(0), n_boot=100,
        )

    strong = arm_report([1.0, 1.1, .9, 1.2, .8])
    unresolved = arm_report([.4, -.2, .3, -.1, .2])
    negative = arm_report([-1.0, -1.1, -.9, -1.2, -.8])
    assert M._h1_intersection_union(strong, strong)["classification"] == (
        "metric-robust confirmatory support"
    )
    assert M._h1_intersection_union(unresolved, unresolved)["classification"] == (
        "directionally consistent but unresolved evidence"
    )
    assert M._h1_intersection_union(strong, negative)["classification"] == (
        "metric-dependent evidence"
    )
    decision = M._h1_intersection_union(negative, negative)
    assert decision["classification"] == "no support"
    assert "nPSI" in decision["inputs_forbidden_from_rescue"]


@pytest.mark.parametrize("field", ["npsi", "whitener_condition_number", "whitener_fit_source_sha256"])
def test_evaluator_refuses_tampered_npsi_or_whitener_audit_fields(tmp_path, field):
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    cell = payload["cells"][0]
    evidence = pathlib.Path(cell["evidence_npz"])
    with np.load(evidence, allow_pickle=False) as z:
        arrays = {key: np.asarray(z[key]) for key in z.files}
    arrays[field] = np.asarray("bad" if field.endswith("sha256") else -1.0)
    np.savez(evidence, **arrays)
    cell["evidence_sha256"] = sha256_file(evidence)
    manifest.write_text(json.dumps(payload))
    report, receipt = M.evaluate_manifest(manifest, expected_seeds=[11, 12], n_boot=100)
    assert report["status"] == receipt["status"] == "INVALID_INCOMPLETE"
    assert report["invalid_cells"][0]["arm"] == "nextlat"


def test_intermediate_whitener_provenance_is_complete_and_strict(tmp_path):
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    cell = payload["cells"][0]
    evidence = pathlib.Path(cell["evidence_npz"])
    with np.load(evidence, allow_pickle=False) as z:
        original = {key: np.asarray(z[key]) for key in z.files}

    mutations = (
        ("secondary_intermediate_whitener_shrinkage", lambda x: np.full((12, 2), -0.1)),
        ("secondary_intermediate_whitener_condition_number", lambda x: np.full((12, 2), .5)),
        (
            "secondary_intermediate_whitener_fit_source_sha256",
            lambda x: np.full((12, 2), "Z" * 64),
        ),
        (
            "secondary_intermediate_whitener_fit_ids",
            lambda x: np.asarray(["0" * 64 + ":wrong_condition", *x.tolist()[1:]]),
        ),
        (
            "secondary_intermediate_whitener_n_features",
            lambda x: np.full((12, 2), 383),
        ),
        (
            "secondary_intermediate_whitener_fit_dtype",
            lambda x: np.full((12, 2), "float32-le"),
        ),
        (
            "secondary_intermediate_whitener_fit_shape",
            lambda x: np.broadcast_to(np.asarray([2000, 383]), (12, 2, 2)).copy(),
        ),
    )
    for field, mutate in mutations:
        arrays = dict(original)
        arrays[field] = mutate(original[field])
        np.savez(evidence, **arrays)
        cell["evidence_sha256"] = sha256_file(evidence)
        manifest.write_text(json.dumps(payload))
        report, _receipt = M.evaluate_manifest(
            manifest, expected_seeds=[11, 12], n_boot=100
        )
        assert report["status"] == "INVALID_INCOMPLETE", field
        assert report["invalid_cells"][0]["arm"] == "nextlat"


def test_analysis_time_primary_h2_failure_becomes_terminal_invalid_cell(tmp_path):
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    cell = payload["cells"][0]
    evidence = pathlib.Path(cell["evidence_npz"])
    with np.load(evidence, allow_pickle=False) as z:
        arrays = {key: np.asarray(z[key]) for key in z.files}
    arrays["base_margin"] = np.ones(M.EXPECTED_H1_ITEMS)
    np.savez(evidence, **arrays)
    cell["evidence_sha256"] = sha256_file(evidence)
    manifest.write_text(json.dumps(payload))
    report, receipt = M.evaluate_manifest(manifest, expected_seeds=[11, 12], n_boot=100)
    assert report["status"] == receipt["status"] == "INVALID_INCOMPLETE"
    invalid = report["invalid_cells"][0]
    assert invalid["arm"] == "nextlat" and invalid["seed"] == 11
    assert invalid["reason_code"] == "H2_PRIMARY_INVALID"
    assert "constant within a training fold" in invalid["reason"]
    assert receipt["invalid_cells"] == report["invalid_cells"]


def test_analysis_time_degenerate_secondary_npsi_becomes_terminal_invalid_cell(tmp_path):
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    cell = payload["cells"][0]
    evidence = pathlib.Path(cell["evidence_npz"])
    with np.load(evidence, allow_pickle=False) as z:
        arrays = {key: np.asarray(z[key]) for key in z.files}
    for field in (
        "secondary_intermediate_d_critical_centered_cosine",
        "secondary_intermediate_d_safe_centered_cosine",
    ):
        arrays[field] = arrays[field].copy()
        arrays[field][0, 0] = 0.0
    np.savez(evidence, **arrays)
    cell["evidence_sha256"] = sha256_file(evidence)
    manifest.write_text(json.dumps(payload))
    report, receipt = M.evaluate_manifest(manifest, expected_seeds=[11, 12], n_boot=100)
    assert report["status"] == receipt["status"] == "INVALID_INCOMPLETE"
    invalid = report["invalid_cells"][0]
    assert invalid["arm"] == "nextlat" and invalid["seed"] == 11
    assert invalid["reason_code"] == "NPSI_INVALID"


def test_evidence_refuses_local_measurement_hash_substitution(tmp_path):
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    cell = payload["cells"][0]
    evidence = pathlib.Path(cell["evidence_npz"])
    with np.load(evidence, allow_pickle=False) as z:
        arrays = {key: np.asarray(z[key]) for key in z.files}
    arrays["local_representations_sha256"] = np.asarray("0" * 64)
    np.savez(evidence, **arrays)
    cell["evidence_sha256"] = sha256_file(evidence)
    manifest.write_text(json.dumps(payload))
    report, _receipt = M.evaluate_manifest(manifest, expected_seeds=[11, 12], n_boot=100)
    assert report["status"] == "INVALID_INCOMPLETE"
    assert "local_representations_sha256 disagrees" in report["invalid_cells"][0]["reason"]


def test_binary_h2_secondary_reports_ceiling_without_invalidating_primary(tmp_path):
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    cell = payload["cells"][0]
    evidence = pathlib.Path(cell["evidence_npz"])
    with np.load(evidence, allow_pickle=False) as z:
        arrays = {key: np.asarray(z[key]) for key in z.files}
    for endpoint in ("first_branch_accuracy", "exact_path_accuracy"):
        arrays[f"behavior_base_{endpoint}"] = np.ones(M.EXPECTED_H1_ITEMS)
        arrays[f"behavior_near_critical_{endpoint}"] = np.ones(M.EXPECTED_H1_ITEMS)
    arrays["behavior_base_generated_path"] = arrays["behavior_base_true_path"].copy()
    arrays["behavior_near_critical_generated_path"] = arrays[
        "behavior_near_critical_true_path"
    ].copy()
    np.savez(evidence, **arrays)
    cell["evidence_sha256"] = sha256_file(evidence)
    manifest.write_text(json.dumps(payload))
    report, _receipt = M.evaluate_manifest(manifest, expected_seeds=[11, 12], n_boot=100)
    first = report["cells"][0]
    status = "not_estimable_due_to_ceiling/constant training-fold predictor"
    assert first["h2"]["first_branch_accuracy_secondary"]["centered_cosine"]["status"] == status
    assert first["h2"]["exact_path_accuracy_secondary"]["centered_cosine"]["status"] == status
    assert first["h2"]["margin_primary"]["both_metric_classification"] in {
        "metric-robust incremental predictive support", "metric-dependent evidence", "inconclusive"
    }


def test_evaluator_recomputes_exact_path_indicator_from_durable_generated_tokens(tmp_path):
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    cell = payload["cells"][0]
    evidence = pathlib.Path(cell["evidence_npz"])
    with np.load(evidence, allow_pickle=False) as z:
        arrays = {key: np.asarray(z[key]) for key in z.files}
    arrays["behavior_base_generated_path"] = arrays["behavior_base_generated_path"].copy()
    arrays["behavior_base_generated_path"][0, 0] += 1
    np.savez(evidence, **arrays)
    cell["evidence_sha256"] = sha256_file(evidence)
    manifest.write_text(json.dumps(payload))
    report, receipt = M.evaluate_manifest(manifest, expected_seeds=[11, 12], n_boot=100)
    assert report["status"] == receipt["status"] == "INVALID_INCOMPLETE"
    assert "indicator disagrees with tokens" in report["invalid_cells"][0]["reason"]


def test_invalid_cell_emits_terminal_incomplete_report_and_receipt(tmp_path):
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    cell = payload["cells"][0]
    evidence = pathlib.Path(cell["evidence_npz"])
    with np.load(evidence, allow_pickle=False) as z:
        arrays = {key: np.asarray(z[key]) for key in z.files}
    arrays.pop("npsi")
    np.savez(evidence, **arrays)
    cell["evidence_sha256"] = sha256_file(evidence)
    manifest.write_text(json.dumps(payload))
    report, receipt = M.evaluate_manifest(manifest, expected_seeds=[11, 12], n_boot=100)
    assert report["status"] == report["primary_status"] == "INVALID_INCOMPLETE"
    assert receipt["status"] == receipt["primary_status"] == "INVALID_INCOMPLETE"
    invalid = report["invalid_cells"]
    assert invalid and set(invalid[0]) == {"arm", "seed", "evidence_npz", "reason_code", "reason"}
    assert invalid[0]["reason_code"] == "NPSI_INVALID"
    assert receipt["invalid_cells"] == invalid


def test_terminal_report_emits_non_equivalence_null_interpretations(tmp_path):
    report, _receipt = M.evaluate_manifest(
        _manifest(tmp_path), expected_seeds=[11, 12], n_boot=100
    )
    assert set(report["nulls"]) == {"H1", "H2"}
    for hypothesis in ("H1", "H2"):
        assert "not resolved at the detectable effect size" in report["nulls"][hypothesis][
            "non_support_interpretation"
        ]
        assert "never evidence of equivalence" in report["nulls"][hypothesis][
            "non_support_interpretation"
        ]
    assert report["manipulation_failures"] == M.MANIPULATION_FAILURES


@pytest.mark.parametrize("field", ["invalid_cells", "nulls", "manipulation_failures"])
def test_terminal_contract_rejects_removed_or_changed_nulls_manipulation_and_invalid_cells(
    tmp_path, field
):
    report, _receipt = M.evaluate_manifest(
        _manifest(tmp_path), expected_seeds=[11, 12], n_boot=100
    )
    removed = dict(report)
    removed.pop(field)
    with pytest.raises(M.EvaluationRefused, match="terminal report"):
        M._validate_terminal_contract(removed)
    changed = dict(report)
    changed[field] = {"tampered": True}
    with pytest.raises(M.EvaluationRefused, match="terminal report"):
        M._validate_terminal_contract(changed)


def test_refuses_incomplete_matrix_and_tampered_base_checkpoint(tmp_path):
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["cells"].pop()
    manifest.write_text(json.dumps(payload))
    with pytest.raises(M.EvaluationRefused, match="complete arm/seed matrix"):
        M.evaluate_manifest(manifest, expected_seeds=[11, 12], n_boot=100)

    manifest = _manifest(tmp_path / "fresh")
    payload = json.loads(manifest.read_text())
    pathlib.Path(payload["cells"][0]["base_checkpoint"]["path"]).write_bytes(b"tampered")
    report, _receipt = M.evaluate_manifest(manifest, expected_seeds=[11, 12], n_boot=100)
    assert report["status"] == "INVALID_INCOMPLETE"
    assert "SHA-256" in report["invalid_cells"][0]["reason"]


def test_refuses_h1_length_coercion_instead_of_truncating_or_padding(tmp_path):
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    cell = payload["cells"][0]
    evidence = pathlib.Path(cell["evidence_npz"])
    with np.load(evidence, allow_pickle=False) as z:
        arrays = {key: np.asarray(z[key]) for key in z.files}
    arrays["d_critical"] = arrays["d_critical"][:-1]
    np.savez(evidence, **arrays)
    cell["evidence_sha256"] = sha256_file(evidence)
    manifest.write_text(json.dumps(payload))
    report, _receipt = M.evaluate_manifest(manifest, expected_seeds=[11, 12], n_boot=100)
    assert report["status"] == "INVALID_INCOMPLETE"
    assert "truncation/padding is forbidden" in report["invalid_cells"][0]["reason"]


def test_refuses_wrong_h1_confirmatory_count_before_loading_cells(tmp_path):
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["identity_domains"]["h1_quartet"]["count"] = 1000
    manifest.write_text(json.dumps(payload))
    with pytest.raises(M.EvaluationRefused, match="exactly 1600"):
        M.evaluate_manifest(manifest, expected_seeds=[11, 12], n_boot=100)


@pytest.mark.parametrize("branch", ["near", "mid", "far"])
def test_refuses_every_adaptation_checkpoint_field(tmp_path, branch):
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    checkpoint = tmp_path / f"{branch}.pt"
    checkpoint.write_bytes(branch.encode())
    payload["cells"][0][f"{branch}_checkpoint"] = _record(checkpoint)
    manifest.write_text(json.dumps(payload))
    report, _receipt = M.evaluate_manifest(manifest, expected_seeds=[11, 12], n_boot=100)
    assert report["status"] == "INVALID_INCOMPLETE"
    assert "only arm, seed, base checkpoint" in report["invalid_cells"][0]["reason"]


@pytest.mark.parametrize(
    "field",
    ["h3_item_ids_near", "margin_after_mid", "gradient_dot_far", "adaptation_item_ids_near"],
)
def test_refuses_any_h3_adaptation_or_mechanism_array(tmp_path, field):
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    cell = payload["cells"][0]
    evidence = pathlib.Path(cell["evidence_npz"])
    with np.load(evidence, allow_pickle=False) as z:
        arrays = {key: np.asarray(z[key]) for key in z.files}
    arrays[field] = np.ones(10)
    np.savez(evidence, **arrays)
    cell["evidence_sha256"] = sha256_file(evidence)
    manifest.write_text(json.dumps(payload))
    report, _receipt = M.evaluate_manifest(manifest, expected_seeds=[11, 12], n_boot=100)
    assert report["status"] == "INVALID_INCOMPLETE"
    assert "extra=" in report["invalid_cells"][0]["reason"]


def test_refuses_h3_analysis_control_or_old_schema(tmp_path):
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["h3_control_contract"] = {"attempt": True}
    manifest.write_text(json.dumps(payload))
    with pytest.raises(M.EvaluationRefused, match="H3 analysis controls are forbidden"):
        M.evaluate_manifest(manifest, expected_seeds=[11, 12], n_boot=100)

    manifest = _manifest(tmp_path / "old")
    payload = json.loads(manifest.read_text())
    payload["schema"] = "nextlat_forgetting/lurestar_evaluation_manifest/2"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(M.EvaluationRefused, match="manifest schema"):
        M.evaluate_manifest(manifest, expected_seeds=[11, 12], n_boot=100)


def test_refuses_noncanonical_permanent_block_hash(tmp_path):
    manifest = _manifest(tmp_path)
    payload = json.loads(manifest.read_text())
    payload["h3_permanent_block"]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload))
    with pytest.raises(M.EvaluationRefused, match="SHA-256 verification"):
        M.evaluate_manifest(manifest, expected_seeds=[11, 12], n_boot=100)


def test_production_source_has_no_retired_h3_analysis_surface_or_legacy_schema():
    source = (ROOT / "scripts/evaluate_lurestar_checkpoints.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function_names = {
        node.name for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assigned_names = {
        target.id for node in ast.walk(tree) if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name)
    }
    assert function_names.isdisjoint({
        "_h3_incremental_models", "_load_cell", "_cell_metrics",
        "_retired_h3_evaluate_manifest",
    })
    assert assigned_names.isdisjoint({
        "BRANCHES", "EXPECTED_H3_ITEMS", "H3_SHARED_ARRAYS", "H3_BRANCH_STEMS",
        "ADAPTATION_BRANCH_STEMS", "VALIDATION_BRANCH_STEMS", "UNTOUCHED_ARRAYS",
    })
    assert "retired/lurestar_evidence/2" not in source


def test_hmm_legacy_named_diagnostics_remain_part_of_the_separate_hmm_contract():
    evaluator = (ROOT / "scripts/evaluate_hmm_checkpoints.py").read_text(encoding="utf-8")
    aggregate = (ROOT / "src/hmm_geometry/aggregate.py").read_text(encoding="utf-8")
    estimator = (ROOT / "src/hmm_geometry/evaluate.py").read_text(encoding="utf-8")
    required = {
        "h3_posterior_decoding_len32", "h3_future_distribution_decoding_len32",
        "h3_posterior_decoding_len64", "h3_future_distribution_decoding_len64",
    }
    assert all(name in evaluator for name in required)
    assert all(name in aggregate for name in required)
    assert "def h3_posterior_decodability(" in estimator
