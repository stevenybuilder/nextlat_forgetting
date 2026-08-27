from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import materialize_adaptation_banks as banks  # noqa: E402


def _line(index: int) -> str:
    # The gate validates serialization identity, while the generator/validator owns graph
    # correctness. Rotating edges makes the canonical graph identity unique per fixture.
    edges = [f"{(index + j) % 100},{(index + j + 1) % 100}" for j in range(20)]
    return f"{'|'.join(edges)}/{index % 100},{(index + 20) % 100}={index % 100},{(index + 1) % 100}"


def _record(index: int, pool: str, *, paired_near: dict | None = None) -> dict:
    line = _line(index)
    record = {
        "line": line,
        "pool": pool,
        "prompt_sha256": banks._prompt_sha256(line),
        "graph_key": banks._semantic_graph_key(line),
    }
    if pool == "B_mid":
        assert paired_near is not None
        record.update(
            paired_near_prompt_sha256=paired_near["prompt_sha256"],
            solver_verified=True,
        )
    return record


def _write_hashed(path: Path, payload: str) -> str:
    path.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode()).hexdigest()
    Path(f"{path}.sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return digest


def _write_manifest(path: Path, records: list[dict]) -> str:
    return _write_hashed(path, "".join(json.dumps(r, sort_keys=True) + "\n" for r in records))


def _pilot_artifact(path: Path, near: list[dict], far: list[dict], near_sha: str, far_sha: str,
                    far_indices: list[int]) -> None:
    artifact = {
        "schema_version": 1,
        "purpose": banks.PURPOSE,
        "selection_method": banks.METHOD,
        "near_bank_sha256": near_sha,
        "candidate_bank_sha256": far_sha,
        "pilot": {
            "role": "non_confirmatory",
            "frozen_before_confirmatory": True,
            "inspected_confirmatory_checkpoints": False,
            "inspected_confirmatory_results": False,
            "optimized_h3_outcomes": False,
            "checkpoint_sha256": "a" * 64,
            "loss_table_sha256": "b" * 64,
            "selector_code_sha256": "c" * 64,
            "created_at_utc": "2026-08-23T00:00:00Z",
        },
        "selection": [
            {
                "near_prompt_sha256": near[i]["prompt_sha256"],
                "far_prompt_sha256": far[j]["prompt_sha256"],
                "near_loss_quantile": (i + 0.5) / len(near),
                "far_loss_quantile": (i + 0.5) / len(near),
            }
            for i, j in enumerate(far_indices)
        ],
    }
    _write_hashed(path, json.dumps(artifact, sort_keys=True) + "\n")


def _mid_artifact(path: Path, near: list[dict], mid: list[dict], near_sha: str, mid_sha: str,
                  mid_indices: list[int]) -> None:
    selected = set(mid_indices)
    candidate_table = []
    distances = []
    for index, record in enumerate(mid):
        near_record = next(
            value for value in near
            if value["prompt_sha256"] == record["paired_near_prompt_sha256"]
        )
        near_item = banks.BankItem(
            near_record["line"], near_record["prompt_sha256"], near_record["graph_key"]
        )
        mid_item = banks.BankItem(
            record["line"], record["prompt_sha256"], record["graph_key"],
            record["paired_near_prompt_sha256"], True,
        )
        distance = banks.normalized_edge_disagreement(near_item, mid_item)
        eligible = index in selected
        candidate_table.append({
            "mid_prompt_sha256": record["prompt_sha256"],
            "near_loss_decile": near.index(near_record),
            "mid_loss_decile": near.index(near_record),
            "pilot_loss_absolute_difference": 0.0 if eligible else 0.2,
            "normalized_edge_disagreement": distance,
            "eligible": eligible,
        })
        if eligible:
            distances.append(distance)
    distances.sort()
    middle = len(distances) // 2
    median = distances[middle] if len(distances) % 2 else (
        distances[middle - 1] + distances[middle]
    ) / 2
    artifact = {
        "schema_version": 1,
        "purpose": banks.MID_PURPOSE,
        "selection_method": banks.MID_METHOD,
        "near_bank_sha256": near_sha,
        "candidate_bank_sha256": mid_sha,
        "distance_quantile": banks.MID_DISTANCE_QUANTILE,
        "pilot_loss_caliper": 0.1,
        "eligible_median_normalized_edge_disagreement": median,
        "tie_break": "candidate_prompt_sha256_ascending",
        "pilot": {
            "role": "non_confirmatory",
            "frozen_before_confirmatory": True,
            "inspected_confirmatory_checkpoints": False,
            "inspected_confirmatory_results": False,
            "optimized_h3_outcomes": False,
            "checkpoint_sha256": "d" * 64,
            "loss_table_sha256": "e" * 64,
            "selector_code_sha256": "f" * 64,
            "created_at_utc": "2026-08-24T00:00:00Z",
        },
        "candidate_table": candidate_table,
        "selection": [
            {
                "near_prompt_sha256": near[i]["prompt_sha256"],
                "mid_prompt_sha256": mid[j]["prompt_sha256"],
                "normalized_edge_disagreement": candidate_table[j]["normalized_edge_disagreement"],
            }
            for i, j in enumerate(mid_indices)
        ],
    }
    _write_hashed(path, json.dumps(artifact, sort_keys=True) + "\n")


def _acquisition_artifact(path: Path, near_sha: str, mid_sha: str, far_sha: str, count: int) -> None:
    _write_hashed(path, json.dumps({
        "schema_version": 1,
        "purpose": banks.ACQUISITION_PURPOSE,
        "selection_method": banks.ACQUISITION_METHOD,
        "frozen_before_confirmatory": True,
        "inspected_confirmatory_checkpoints": False,
        "inspected_confirmatory_results": False,
        "optimized_h3_outcomes": False,
        "disjoint_from_training": True,
        "matched_target_path_distribution": True,
        "matched_pilot_loss_deciles": True,
        "selector_code_sha256": "a" * 64,
        "bank_sha256": {"near": near_sha, "mid": mid_sha, "far": far_sha},
        "counts": {"near": count, "mid": count, "far": count},
    }, sort_keys=True) + "\n")


def test_shipped_near_manifest_materializes_exact_frozen_bank(tmp_path: Path) -> None:
    assert banks.main(["--near-only", "--output-dir", str(tmp_path)]) == 0
    output = tmp_path / banks.OUTPUT_NAMES["near"]
    expected = "".join(
        json.loads(raw)["line"] + "\n"
        for raw in (ROOT / "manifests/b_near.jsonl").read_text().splitlines()
    )
    assert output.read_text() == expected
    assert len(output.read_text().splitlines()) == 5_000
    assert banks.verify_sidecar(output) == hashlib.sha256(expected.encode()).hexdigest()


def test_full_mode_refuses_without_pilot_and_independent_validation(tmp_path: Path) -> None:
    with pytest.raises(banks.GateError, match="full materialization is gated"):
        banks.main(["--output-dir", str(tmp_path)])
    assert not tmp_path.exists() or not list(tmp_path.iterdir())


def test_manifest_hash_sidecar_is_mandatory_and_verified(tmp_path: Path) -> None:
    manifest = tmp_path / "near.jsonl"
    manifest.write_text("{}\n")
    with pytest.raises(banks.GateError, match="needs a SHA-256 sidecar"):
        banks.load_manifest(manifest, 1, "B_near")
    Path(f"{manifest}.sha256").write_text(f"{'0' * 64}  near.jsonl\n")
    with pytest.raises(banks.GateError, match="SHA-256 mismatch"):
        banks.load_manifest(manifest, 1, "B_near")


def test_valid_pilot_mapping_selects_recorded_items_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(banks, "NEAR_COUNT", 3)
    near_records = [_record(i, "B_near") for i in range(3)]
    far_records = [_record(i + 20, "B_far") for i in range(6)]
    near_path, far_path = tmp_path / "near.jsonl", tmp_path / "far.jsonl"
    near_sha = _write_manifest(near_path, near_records)
    far_sha = _write_manifest(far_path, far_records)
    artifact = tmp_path / "selection.json"
    _pilot_artifact(artifact, near_records, far_records, near_sha, far_sha, [1, 3, 5])
    near = banks.load_manifest(near_path, 3, "B_near")
    far = banks.load_manifest(far_path, 6, "B_far")
    selected = banks.select_far(
        near, far, artifact, near_sha256=near_sha, candidates_sha256=far_sha
    )
    assert [x.prompt_sha256 for x in selected] == [far_records[i]["prompt_sha256"] for i in (1, 3, 5)]


def test_mid_mapping_requires_structural_median_and_pilot_loss_decile(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(banks, "MID_COUNT", 2)
    monkeypatch.setattr(banks, "MID_CANDIDATE_COUNT", 4)
    near_records = [_record(i, "B_near") for i in range(2)]
    # The final two candidates are exactly ten edge shifts away: disagreement = 10/20.
    mid_records = [
        _record(30, "B_mid", paired_near=near_records[0]),
        _record(31, "B_mid", paired_near=near_records[1]),
        _record(10, "B_mid", paired_near=near_records[0]),
        _record(11, "B_mid", paired_near=near_records[1]),
    ]
    near_path, mid_path = tmp_path / "near.jsonl", tmp_path / "mid.jsonl"
    near_sha = _write_manifest(near_path, near_records)
    mid_sha = _write_manifest(mid_path, mid_records)
    artifact = tmp_path / "mid-selection.json"
    _mid_artifact(artifact, near_records, mid_records, near_sha, mid_sha, [2, 3])
    selected = banks.select_mid(
        banks.load_manifest(near_path, 2, "B_near"),
        banks.load_manifest(mid_path, 4, "B_mid"), artifact,
        near_sha256=near_sha, candidates_sha256=mid_sha,
    )
    assert [item.prompt_sha256 for item in selected] == [
        mid_records[i]["prompt_sha256"] for i in (2, 3)
    ]
    payload = json.loads(artifact.read_text())
    payload["candidate_table"][2]["mid_loss_decile"] = 9
    _write_hashed(artifact, json.dumps(payload, sort_keys=True) + "\n")
    with pytest.raises(banks.GateError, match="eligibility does not follow"):
        banks.select_mid(
            banks.load_manifest(near_path, 2, "B_near"),
            banks.load_manifest(mid_path, 4, "B_mid"), artifact,
            near_sha256=near_sha, candidates_sha256=mid_sha,
        )


def test_d40_mid_contract_requires_permanent_one_shot_stop(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(banks, "MID_EXPANDED_CANDIDATE_COUNT", 150)
    artifact = tmp_path / "mid-selection-d40.json"
    payload = {
        "schema_version": 1,
        "purpose": banks.MID_D40_PURPOSE,
        "selection_method": banks.MID_D40_METHOD,
        "permanent_block_if_any_unmatched": True,
        "no_further_amendments_permitted": True,
        "combined_loss_table_sha256": "a" * 64,
    }
    _write_hashed(artifact, json.dumps(payload, sort_keys=True) + "\n")
    assert banks.mid_candidate_count_from_selection(artifact) == 150
    payload["no_further_amendments_permitted"] = False
    _write_hashed(artifact, json.dumps(payload, sort_keys=True) + "\n")
    with pytest.raises(banks.GateError, match="permanent one-shot"):
        banks.mid_candidate_count_from_selection(artifact)


def test_first_n_far_candidates_are_explicitly_refused(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(banks, "NEAR_COUNT", 3)
    near_records = [_record(i, "B_near") for i in range(3)]
    far_records = [_record(i + 20, "B_far") for i in range(6)]
    near_path, far_path = tmp_path / "near.jsonl", tmp_path / "far.jsonl"
    near_sha = _write_manifest(near_path, near_records)
    far_sha = _write_manifest(far_path, far_records)
    artifact = tmp_path / "selection.json"
    _pilot_artifact(artifact, near_records, far_records, near_sha, far_sha, [0, 1, 2])
    with pytest.raises(banks.GateError, match="first 5,000 candidates"):
        banks.select_far(
            banks.load_manifest(near_path, 3, "B_near"),
            banks.load_manifest(far_path, 6, "B_far"),
            artifact,
            near_sha256=near_sha,
            candidates_sha256=far_sha,
        )


def test_reordering_near_pairs_or_mismatching_quantiles_is_refused(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(banks, "NEAR_COUNT", 2)
    near_records = [_record(i, "B_near") for i in range(2)]
    far_records = [_record(i + 20, "B_far") for i in range(4)]
    near_path, far_path = tmp_path / "near.jsonl", tmp_path / "far.jsonl"
    near_sha = _write_manifest(near_path, near_records)
    far_sha = _write_manifest(far_path, far_records)
    artifact = tmp_path / "selection.json"
    _pilot_artifact(artifact, near_records, far_records, near_sha, far_sha, [1, 3])
    payload = json.loads(artifact.read_text())
    payload["selection"][0]["near_prompt_sha256"] = near_records[1]["prompt_sha256"]
    payload["selection"][1]["near_prompt_sha256"] = near_records[0]["prompt_sha256"]
    _write_hashed(artifact, json.dumps(payload, sort_keys=True) + "\n")
    near = banks.load_manifest(near_path, 2, "B_near")
    far = banks.load_manifest(far_path, 4, "B_far")
    with pytest.raises(banks.GateError, match="paired item order"):
        banks.select_far(near, far, artifact, near_sha256=near_sha, candidates_sha256=far_sha)

    _pilot_artifact(artifact, near_records, far_records, near_sha, far_sha, [1, 3])
    payload = json.loads(artifact.read_text())
    payload["selection"][0]["far_loss_quantile"] += 0.1
    _write_hashed(artifact, json.dumps(payload, sort_keys=True) + "\n")
    with pytest.raises(banks.GateError, match="same loss quantile"):
        banks.select_far(near, far, artifact, near_sha256=near_sha, candidates_sha256=far_sha)


@pytest.mark.parametrize(
    "field,value",
    [
        ("role", "confirmatory"),
        ("frozen_before_confirmatory", False),
        ("inspected_confirmatory_checkpoints", True),
        ("inspected_confirmatory_results", True),
        ("optimized_h3_outcomes", True),
    ],
)
def test_confirmatory_or_outcome_tuned_selection_is_refused(
    tmp_path: Path, monkeypatch, field: str, value
) -> None:
    monkeypatch.setattr(banks, "NEAR_COUNT", 2)
    near_records = [_record(i, "B_near") for i in range(2)]
    far_records = [_record(i + 20, "B_far") for i in range(4)]
    near_path, far_path = tmp_path / "near.jsonl", tmp_path / "far.jsonl"
    near_sha = _write_manifest(near_path, near_records)
    far_sha = _write_manifest(far_path, far_records)
    artifact = tmp_path / "selection.json"
    _pilot_artifact(artifact, near_records, far_records, near_sha, far_sha, [1, 3])
    payload = json.loads(artifact.read_text())
    payload["pilot"][field] = value
    _write_hashed(artifact, json.dumps(payload, sort_keys=True) + "\n")
    with pytest.raises(banks.GateError, match=f"pilot.{field}"):
        banks.select_far(
            banks.load_manifest(near_path, 2, "B_near"),
            banks.load_manifest(far_path, 4, "B_far"),
            artifact,
            near_sha256=near_sha,
            candidates_sha256=far_sha,
        )


def test_validation_inputs_need_exact_count_hashes_and_disjoint_graphs(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(banks, "VALIDATION_COUNT", 2)
    near_val = tmp_path / "near_val.txt"
    _write_hashed(near_val, _line(50) + "\n" + _line(51) + "\n")
    loaded = banks.load_validation(near_val)
    assert len(loaded) == 2
    with pytest.raises(banks.GateError, match="not independent"):
        banks.require_disjoint({"near_validation": loaded, "far_validation": loaded})


def test_output_names_and_receipt_are_config_compatible(tmp_path: Path) -> None:
    item = banks.BankItem(
        line=_line(70),
        prompt_sha256=banks._prompt_sha256(_line(70)),
        graph_key=banks._semantic_graph_key(_line(70)),
    )
    receipt = banks.write_banks(tmp_path, {"far": [item]}, {"fixture": {"sha256": "f" * 64}})
    assert (tmp_path / "graph_5_5_bfar_5000.txt").is_file()
    assert receipt["scientific_selection_performed"] is False
    assert banks.verify_sidecar(tmp_path / "adaptation_banks.json")


def test_source_paths_are_portable_between_host_and_colab(tmp_path: Path) -> None:
    project = tmp_path / "project"
    inside = project / "manifests" / "selection.json"
    inside.parent.mkdir(parents=True)
    inside.write_text("{}\n")
    outside = tmp_path / "external.json"
    outside.write_text("{}\n")

    assert banks.portable_source_path(inside, project) == "manifests/selection.json"
    assert banks.portable_source_path(outside, project) == str(outside.resolve())


def test_manifest_inventory_binds_nested_adaptation_and_hmm_inputs(tmp_path: Path) -> None:
    project = tmp_path / "project"
    adapt = project / "manifests" / "adapt"
    hmm = project / "data" / "hmm"
    hmm_family = project / "data" / "hmm_family" / "persistent"
    adapt.mkdir(parents=True)
    hmm.mkdir(parents=True)
    hmm_family.mkdir(parents=True)
    (adapt / "bank.txt").write_text("bank\n")
    (hmm / "train.npy").write_bytes(b"array")
    (hmm_family / "seed1234.npy").write_bytes(b"family-array")

    digest = banks.refresh_manifest_inventory(project)
    inventory = project / "manifests" / "manifest_inventory.sha256"
    rows = inventory.read_text().splitlines()

    assert banks.sha256_file(inventory) == digest
    assert any(row.endswith("  manifests/adapt/bank.txt") for row in rows)
    assert any(row.endswith("  data/hmm/train.npy") for row in rows)
    assert any(row.endswith("  data/hmm_family/persistent/seed1234.npy") for row in rows)
    assert not any("manifest_inventory.sha256" in row for row in rows)
    assert banks.verify_sidecar(inventory) == digest


def test_full_gate_writes_all_six_config_filenames(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(banks, "NEAR_COUNT", 2)
    monkeypatch.setattr(banks, "FAR_CANDIDATE_COUNT", 4)
    monkeypatch.setattr(banks, "MID_COUNT", 2)
    monkeypatch.setattr(banks, "MID_CANDIDATE_COUNT", 4)
    monkeypatch.setattr(banks, "VALIDATION_COUNT", 2)
    near_records = [_record(i, "B_near") for i in range(2)]
    far_records = [_record(i + 20, "B_far") for i in range(4)]
    mid_records = [
        _record(30, "B_mid", paired_near=near_records[0]),
        _record(31, "B_mid", paired_near=near_records[1]),
        _record(10, "B_mid", paired_near=near_records[0]),
        _record(11, "B_mid", paired_near=near_records[1]),
    ]
    near_path, far_path, mid_path = (
        tmp_path / "near.jsonl", tmp_path / "far.jsonl", tmp_path / "mid.jsonl"
    )
    near_sha = _write_manifest(near_path, near_records)
    far_sha = _write_manifest(far_path, far_records)
    mid_sha = _write_manifest(mid_path, mid_records)
    selection = tmp_path / "selection.json"
    _pilot_artifact(selection, near_records, far_records, near_sha, far_sha, [1, 3])
    mid_selection = tmp_path / "mid-selection.json"
    _mid_artifact(mid_selection, near_records, mid_records, near_sha, mid_sha, [2, 3])
    near_validation, mid_validation, far_validation = (
        tmp_path / "near-val.txt", tmp_path / "mid-val.txt", tmp_path / "far-val.txt"
    )
    near_val_sha = _write_hashed(near_validation, _line(50) + "\n" + _line(51) + "\n")
    mid_val_sha = _write_hashed(mid_validation, _line(60) + "\n" + _line(61) + "\n")
    far_val_sha = _write_hashed(far_validation, _line(70) + "\n" + _line(71) + "\n")
    acquisition = tmp_path / "acquisition.json"
    _acquisition_artifact(acquisition, near_val_sha, mid_val_sha, far_val_sha, 2)
    output = tmp_path / "out"

    assert banks.main([
        "--near-manifest", str(near_path),
        "--far-candidates", str(far_path),
        "--far-selection", str(selection),
        "--mid-candidates", str(mid_path),
        "--mid-selection", str(mid_selection),
        "--near-validation", str(near_validation),
        "--mid-validation", str(mid_validation),
        "--far-validation", str(far_validation),
        "--acquisition-provenance", str(acquisition),
        "--output-dir", str(output),
    ]) == 0
    assert {path.name for path in output.glob("graph_5_5_*.txt")} == set(banks.OUTPUT_NAMES.values())
    for name in banks.OUTPUT_NAMES.values():
        assert banks.verify_sidecar(output / name)
