from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from cfs1.adaptation import (
    CFS1_ARMS, CFS1_EPISODES, CFS1_UPDATE_STEPS, CFS1AdaptationError,
    cfs1_branch_order, validate_nextlat_ce_only_config, validate_update_manifest,
)


def _artifact(root: Path, name: str, payload: str = "frozen") -> dict[str, str]:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _manifest(root: Path) -> Path:
    construction = _artifact(root, "receipts/construction.json")
    generator = _artifact(root, "receipts/generator.json")
    generator_manifest = _artifact(root, "manifests/global.json")
    retention = _artifact(root, "manifests/retention.jsonl")
    global_controls = _artifact(root, "manifests/global_controls.jsonl")
    inputs = {name: _artifact(root, f"probes/{name}.jsonl") for name in (
        "margin", "retention_ce", "retention_exact_path", "global_controls", "state_drift",
        "pregeometry")}
    episodes = []
    for episode in CFS1_EPISODES:
        arms = {}
        for arm in CFS1_ARMS:
            overlap, relation = arm.split("_", 1)
            arms[arm] = {**_artifact(root, f"updates/graph_5_5_cfs1_episode{episode}_{arm}.txt", arm),
                         "overlap": overlap, "future_relation": relation}
        episodes.append({"episode": episode,
                         "episode_sha256": ("a" if episode == 0 else "b") * 64, "arms": arms})
    document = {
        "schema": "nextlat_forgetting/cfs1_update_manifest/1", "status": "FROZEN",
        "construction": {"model_outcomes_inspected": False, "training_outcomes_inspected": False,
                         "retention_outcomes_inspected": False, "matching": "construction_matched",
                         "randomized_assignment": True, "receipt": construction},
        "generator_receipt": generator,
        "generator_manifest": generator_manifest,
        "retention_probes": retention,
        "global_control_manifest": global_controls,
        "design": {"model": "nextlat", "adaptation_steps": CFS1_UPDATE_STEPS,
                   "full_parameter": True, "loss": "teacher_forced_next_token_cross_entropy",
                   "arms": list(CFS1_ARMS), "episodes": list(CFS1_EPISODES)},
        "execution_order_algorithm": "sha256-sort-v1", "execution_order_salt_sha256": "c" * 64,
        "evaluation_inputs": inputs, "episodes": episodes,
    }
    path = root / "cfs1_update_manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_manifest_requires_all_outcome_blind_arms_and_paired_evaluation_inputs(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    parsed = validate_update_manifest(manifest)
    assert parsed["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert set(parsed["episodes"]) == set(CFS1_EPISODES)
    assert set(parsed["evaluation_inputs"]) == {
        "margin", "retention_ce", "retention_exact_path", "global_controls", "state_drift",
        "pregeometry"}

    document = json.loads(manifest.read_text())
    document["construction"]["retention_outcomes_inspected"] = True
    manifest.write_text(json.dumps(document))
    with pytest.raises(CFS1AdaptationError, match="outcome-blind"):
        validate_update_manifest(manifest)


def test_manifest_refuses_swapped_arm_identity_or_missing_probe_binding(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    document = json.loads(manifest.read_text())
    document["episodes"][0]["arms"]["high_different"]["overlap"] = "low"
    manifest.write_text(json.dumps(document))
    with pytest.raises(CFS1AdaptationError, match="overlap binding"):
        validate_update_manifest(manifest)

    manifest = _manifest(tmp_path / "second")
    document = json.loads(manifest.read_text())
    document["evaluation_inputs"].pop("pregeometry")
    manifest.write_text(json.dumps(document))
    with pytest.raises(CFS1AdaptationError, match="evaluation input membership"):
        validate_update_manifest(manifest)


def test_hash_randomized_order_is_deterministic_and_outcome_free(tmp_path: Path) -> None:
    parsed = validate_update_manifest(_manifest(tmp_path))
    ids = ["cfs1-c", "cfs1-a", "cfs1-b"]
    assert cfs1_branch_order(parsed, ids) == cfs1_branch_order(parsed, list(reversed(ids)))


def test_nextlat_config_contract_rejects_auxiliary_loss_or_bst() -> None:
    config = {"use_nextlat": True, "use_bst": False,
              "model": {"lambda_mse": 0.0, "lambda_kl": 0.0, "lambda_ce": 0.0}}
    assert validate_nextlat_ce_only_config(config)["full_parameter"] is True
    with pytest.raises(CFS1AdaptationError, match="auxiliary"):
        validate_nextlat_ce_only_config({**config, "model": {**config["model"], "lambda_kl": 1.0}})
