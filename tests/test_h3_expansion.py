from __future__ import annotations

import json
import pathlib
import sys
import hashlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from lurestar import h3_expansion as E  # noqa: E402
from lurestar.generate import graph_from_line  # noqa: E402
import run_h3_expansion_durable as DURABLE  # noqa: E402
import score_h3_expansion as SCORE  # noqa: E402
import finalize_h3_d40 as FINALIZE  # noqa: E402


def _near() -> dict:
    return json.loads((ROOT / "manifests/b_near.jsonl").read_text().splitlines()[0])


def test_d40_candidate_rng_is_deterministic_and_separate() -> None:
    base = graph_from_line(_near()["line"])
    first = E._candidate(base, item=0, rewires=3, slot=8, attempt=4).serialize()
    assert first == E._candidate(base, item=0, rewires=3, slot=8, attempt=4).serialize()
    assert first != E._candidate(base, item=0, rewires=3, slot=8, attempt=5).serialize()
    assert E.MASTER_SEED != 2_026_082_401 and E.RNG_NAMESPACE == 40


def _balanced() -> list[dict]:
    rows = []
    for near in ("a" * 64, "b" * 64):
        for rewires in E.STRATA:
            for slot in range(10):
                identity = f"{near}:{rewires}:{slot}"
                rows.append({
                    "paired_near_prompt_sha256": near, "rewire_count": rewires,
                    "prompt_sha256": __import__("hashlib").sha256((identity + "p").encode()).hexdigest(),
                    "graph_key": __import__("hashlib").sha256((identity + "g").encode()).hexdigest(),
                })
    return rows


def test_exact_10_per_rewire_class_gate_and_count_mutation() -> None:
    rows = _balanced()
    E.require_exact_strata(rows, ["a" * 64, "b" * 64])
    with pytest.raises(E.ExpansionRefused, match="exactly balanced"):
        E.require_exact_strata(rows[:-1], ["a" * 64, "b" * 64])


def test_duplicate_prompt_or_graph_mutation_is_refused() -> None:
    rows = _balanced()
    rows[-1]["graph_key"] = rows[0]["graph_key"]
    with pytest.raises(E.ExpansionRefused, match="reuses"):
        E.require_exact_strata(rows, ["a" * 64, "b" * 64])


def test_d40_frozen_count_arithmetic() -> None:
    assert E.ORIGINAL_PER_NEAR + len(E.STRATA) * E.NEW_PER_STRATUM == E.TOTAL_PER_NEAR
    assert E.NEAR_COUNT * E.TOTAL_PER_NEAR == E.EXPANDED_COUNT
    assert E.NEAR_COUNT * len(E.STRATA) * E.NEW_PER_STRATUM == E.NEW_COUNT
    assert 53_000 + E.NEW_COUNT == E.COMBINED_LOSS_COUNT


def test_frozen_scorer_and_d39_artifacts_remain_byte_identical() -> None:
    from lurestar.h3_precompute import sha256_file
    assert sha256_file(ROOT / "scripts/score_h3_pilot.py") == E.D39_SCORER_SHA256
    assert sha256_file(ROOT / "manifests/h3_precompute/b_mid_candidates.jsonl") == E.D39_MID_SHA256
    assert sha256_file(ROOT / ".agent_state/pilot/h3_pilot_score/pilot_losses.jsonl") == E.D39_LOSS_SHA256


def test_d40_frozen_job_binds_exact_population_and_durable_prefix() -> None:
    job = ROOT / ".agent_state/pilot/h3-expansion-score-job.json"
    bound = SCORE.load_job(job)
    plan = DURABLE.plan(job)
    assert len(bound["_bound"]["rows"]) == E.NEW_COUNT
    assert bound["_bound"]["hashes"]["frozen_scorer"] == E.D39_SCORER_SHA256
    assert plan["job_sha256"] == "393c933e9e616cd24a4b7a9b408203b0c22002c39cf97f2d72b03176fe45482a"
    assert plan["remote_prefix"].endswith("/" + plan["job_sha256"])
    assert plan["maximum_uncommitted_items"] == 1_000


def _portable_job(tmp_path: pathlib.Path) -> tuple[pathlib.Path, dict]:
    source = ROOT / ".agent_state/pilot/h3-expansion-score-job.json"
    payload = json.loads(source.read_text())
    for value in payload.values():
        if isinstance(value, dict) and "path" in value:
            path = pathlib.Path(value["path"])
            value["path"] = str((source.parent / path).resolve()) if not path.is_absolute() else str(path)
    payload["output_dir"] = str(tmp_path / "score")
    target = tmp_path / "job.json"
    return target, payload


def test_job_refuses_mutated_generator_code_receipt(tmp_path: pathlib.Path) -> None:
    target, payload = _portable_job(tmp_path)
    receipt = json.loads((ROOT / "manifests/h3_expansion/generation_code_receipt.json").read_text())
    receipt["confirmatory_results_inspected"] = True
    mutated = tmp_path / "generation-code.json"
    mutated.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    payload["generation_code_receipt"] = {
        "path": str(mutated), "sha256": hashlib.sha256(mutated.read_bytes()).hexdigest()
    }
    target.write_text(json.dumps(payload))
    with pytest.raises(E.ExpansionRefused, match="generator-byte receipt"):
        SCORE.load_job(target)


def test_job_refuses_frozen_scorer_hash_substitution(tmp_path: pathlib.Path) -> None:
    target, payload = _portable_job(tmp_path)
    payload["frozen_scorer"]["sha256"] = "0" * 64
    target.write_text(json.dumps(payload))
    with pytest.raises(E.D39.PrecomputeRefused, match="does not match its frozen SHA-256"):
        SCORE.load_job(target)


def test_durable_job_refuses_larger_uncommitted_chunk(tmp_path: pathlib.Path) -> None:
    target, payload = _portable_job(tmp_path)
    payload["scoring"]["chunk_size"] = 2_000
    target.write_text(json.dumps(payload))
    with pytest.raises(E.ExpansionRefused, match="chunk_size at 1,000"):
        DURABLE.raw_job(target)


def test_job_refuses_upstream_commit_substitution(tmp_path: pathlib.Path) -> None:
    target, payload = _portable_job(tmp_path)
    payload["upstream_commit"] = "0" * 40
    target.write_text(json.dumps(payload))
    with pytest.raises(E.ExpansionRefused, match="upstream commit"):
        SCORE.load_job(target)


def test_unchanged_d39_auxiliary_path_skips_failed_d39_mid(tmp_path: pathlib.Path) -> None:
    receipt = FINALIZE.auxiliary_d39(
        root=ROOT,
        loss_table=ROOT / ".agent_state/pilot/h3_pilot_score/pilot_losses.jsonl",
        candidate_dir=ROOT / "manifests/h3_precompute",
        output_dir=tmp_path,
    )
    assert receipt["status"] == "D40_UNCHANGED_D39_AUXILIARY_SELECTIONS_FROZEN"
    assert len(json.loads((tmp_path / "far_selection.json").read_text())["selection"]) == 5_000
    provenance = json.loads((tmp_path / "acquisition_provenance.json").read_text())
    assert provenance["counts"] == {"near": 2_000, "mid": 2_000, "far": 2_000}
    assert all((tmp_path / f"acquisition_{branch}.jsonl").is_file()
               for branch in ("near", "mid", "far"))
