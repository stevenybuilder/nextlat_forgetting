"""Immutable CFS-2 adaptation envelope.

CFS-2 is the repaired successor to CFS-1.  This module deliberately owns a
different schema, contract name, and namespace: CFS-1 streams or manifests can
never be accepted as CFS-2 inputs merely because their surface happens to look
similar.  It validates identities and hashes only; graph solving remains in
``cfs2.validate`` and no model or outcome is opened here.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from typing import Any, Mapping


CFS2_MANIFEST_SCHEMA = "nextlat_forgetting/cfs2_update_manifest/1"
CFS2_ADAPTATION_SCHEMA = "nextlat_forgetting/cfs2_adaptation/1"
CFS2_CONTRACT = "cfs2_full_parameter_next_token_ce_v1"
CFS2_UPDATE_STEPS = 500
CFS2_PARENT_SEEDS = (1234, 1235, 1236, 1237, 1238, 2234, 2235, 2236)
CFS2_EPISODES = (0, 1)
CFS2_ARMS = ("high_different", "low_different", "high_same", "low_same")
CFS2_ORDER_ALGORITHM = "sha256-sort-v1"
CFS2_EVALUATION_INPUTS = (
    "margin", "retention_ce", "retention_exact_path", "global_controls",
    "state_drift", "pregeometry",
)
CFS2_EXPECTED_OVERLAPS = {
    "high_same": 18, "high_different": 18, "low_same": 8, "low_different": 8,
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


class CFS2AdaptationError(RuntimeError):
    """Raised when CFS-2's outcome-blind execution envelope is not exact."""


def sha256_file(path: str | pathlib.Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def contract_sha256(path: str | pathlib.Path | None = None) -> str:
    return sha256_file(pathlib.Path(path) if path is not None else pathlib.Path(__file__))


def canonical_json_sha256(value: Any) -> str:
    try:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CFS2AdaptationError("CFS-2 receipt is not canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CFS2AdaptationError(message)


def _sha(value: Any, label: str) -> str:
    _require(isinstance(value, str) and _SHA256.fullmatch(value) is not None,
             f"{label} must be a lowercase SHA-256")
    return value


def _artifact(root: pathlib.Path, value: Any, label: str) -> dict[str, str]:
    _require(isinstance(value, Mapping) and set(value) == {"path", "sha256"},
             f"{label} must bind exactly path and SHA-256")
    relative, digest = value["path"], _sha(value["sha256"], f"{label}.sha256")
    _require(isinstance(relative, str) and relative, f"{label}.path is invalid")
    pure = pathlib.PurePosixPath(relative)
    _require(not pure.is_absolute() and all(part not in ("", ".", "..") for part in pure.parts),
             f"{label}.path must be safe and relative")
    path = (root / pure).resolve()
    _require(root == path or root in path.parents, f"{label} escapes manifest root")
    _require(path.is_file() and not path.is_symlink(), f"{label} is absent or symlinked")
    _require(sha256_file(path) == digest, f"{label} SHA-256 mismatch")
    return {"path": str(pure), "sha256": digest}


def validate_nextlat_ce_only_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Check CFS-2's sole permitted branch objective."""
    _require(config.get("use_nextlat") is True and config.get("use_bst", False) is False,
             "CFS-2 must select NextLat and never BST")
    model = config.get("model")
    _require(isinstance(model, Mapping), "CFS-2 config lacks model")
    coefficients = {key: model.get(key) for key in ("lambda_mse", "lambda_kl", "lambda_ce")}
    _require(all(value == 0.0 for value in coefficients.values()),
             "CFS-2 auxiliary NextLat coefficients must all be exactly zero")
    return {
        "schema": CFS2_ADAPTATION_SCHEMA, "contract": CFS2_CONTRACT,
        "contract_sha256": contract_sha256(), "family": "nextlat", "full_parameter": True,
        "loss": "teacher_forced_next_token_cross_entropy",
        "nextlat_auxiliary_coefficients": coefficients,
    }


def validate_update_manifest(path: str | pathlib.Path) -> dict[str, Any]:
    """Validate the frozen repaired-stimulus envelope without reading its lines."""
    manifest_path = pathlib.Path(path).resolve()
    _require(manifest_path.is_file() and not manifest_path.is_symlink(), "CFS-2 manifest is absent")
    root = manifest_path.parent.resolve()
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CFS2AdaptationError("CFS-2 manifest is unreadable") from exc
    _require(isinstance(document, Mapping) and document.get("schema") == CFS2_MANIFEST_SCHEMA,
             "unexpected CFS-2 update-manifest schema")
    _require(document.get("status") == "FROZEN", "CFS-2 manifest is not frozen")
    construction = document.get("construction")
    _require(isinstance(construction, Mapping), "CFS-2 manifest lacks construction")
    for key in ("model_outcomes_inspected", "training_outcomes_inspected", "retention_outcomes_inspected"):
        _require(construction.get(key) is False, f"CFS-2 construction must be outcome-blind: {key}")
    _require(construction.get("matching") == "construction_matched" and
             construction.get("randomized_assignment") is True,
             "CFS-2 construction/randomization binding is invalid")
    construction_receipt = _artifact(root, construction.get("receipt"), "construction.receipt")
    exact_overlap = construction.get("exact_total_edge_overlap")
    _require(exact_overlap == CFS2_EXPECTED_OVERLAPS,
             "CFS-2 must bind exact repaired 18/18/8/8 overlap artifacts")
    design = document.get("design")
    _require(isinstance(design, Mapping), "CFS-2 manifest lacks design")
    _require(design.get("model") == "nextlat" and design.get("adaptation_steps") == CFS2_UPDATE_STEPS and
             design.get("full_parameter") is True and
             design.get("loss") == "teacher_forced_next_token_cross_entropy",
             "CFS-2 must bind 500 full-parameter CE-only NextLat updates")
    _require(tuple(design.get("arms", ())) == CFS2_ARMS and tuple(design.get("episodes", ())) == CFS2_EPISODES,
             "CFS-2 arms/episodes are not frozen")
    _require(document.get("execution_order_algorithm") == CFS2_ORDER_ALGORITHM,
             "CFS-2 execution order algorithm is invalid")
    salt = _sha(document.get("execution_order_salt_sha256"), "execution_order_salt_sha256")
    inputs = document.get("evaluation_inputs")
    # Materialized receipts use canonical JSON (sorted keys), so the semantic order is
    # carried explicitly rather than inferred from serialized-object key order.
    _require(isinstance(inputs, Mapping) and set(inputs) == set(CFS2_EVALUATION_INPUTS) and
             tuple(document.get("evaluation_input_order", ())) == CFS2_EVALUATION_INPUTS,
             "CFS-2 evaluation input membership/order is not frozen")
    normalized_inputs = {name: _artifact(root, inputs[name], f"evaluation_inputs.{name}")
                         for name in CFS2_EVALUATION_INPUTS}
    state_commitment = _artifact(root, document.get("state_interchange_activation_patching"),
                                 "state interchange/activation-patching commitment")
    generator_receipt = _artifact(root, document.get("generator_receipt"), "generator_receipt")
    generator_manifest = _artifact(root, document.get("generator_manifest"), "generator_manifest")
    retention = _artifact(root, document.get("retention_probes"), "retention_probes")
    global_controls = _artifact(root, document.get("global_control_manifest"), "global_control_manifest")
    raw_episodes = document.get("episodes")
    _require(isinstance(raw_episodes, list) and len(raw_episodes) == 2, "CFS-2 requires two episodes")
    episodes: dict[int, dict[str, Any]] = {}
    for expected_episode, raw_episode in zip(CFS2_EPISODES, raw_episodes):
        _require(isinstance(raw_episode, Mapping) and raw_episode.get("episode") == expected_episode,
                 "CFS-2 episode identity is not canonical")
        _sha(raw_episode.get("episode_sha256"), f"episode{expected_episode}.episode_sha256")
        arms = raw_episode.get("arms")
        _require(isinstance(arms, Mapping) and set(arms) == set(CFS2_ARMS),
                 "CFS-2 arm membership is not canonical")
        normalized_arms: dict[str, dict[str, str]] = {}
        for arm in CFS2_ARMS:
            record = arms[arm]
            _require(isinstance(record, Mapping), f"CFS-2 {arm} is malformed")
            overlap, relation = arm.split("_", 1)
            _require(record.get("overlap") == overlap and record.get("future_relation") == relation,
                     f"CFS-2 {arm} factor binding is wrong")
            _require(record.get("total_edge_overlap") == CFS2_EXPECTED_OVERLAPS[arm],
                     f"CFS-2 {arm} does not bind repaired overlap")
            artifact = _artifact(root, {"path": record.get("path"), "sha256": record.get("sha256")},
                                 f"episode{expected_episode}.{arm}")
            _require(artifact["path"].startswith("streams/graph_5_5_cfs2_") and
                     artifact["path"].endswith(".txt"), "CFS-2 must use CFS-2 raw graph streams")
            normalized_arms[arm] = artifact
        episodes[expected_episode] = {"episode": expected_episode,
                                      "episode_sha256": raw_episode["episode_sha256"], "arms": normalized_arms}
    parent_identity = document.get("parent_identity")
    _require(isinstance(parent_identity, Mapping) and tuple(parent_identity.get("seeds", ())) == CFS2_PARENT_SEEDS and
             parent_identity.get("cfs_only_alias_requires_hash_bound_lineage_receipt") is True,
             "CFS-2 parent identity/lineage rule is missing")
    return {
        "path": str(manifest_path), "sha256": sha256_file(manifest_path), "root": str(root),
        "construction_receipt": construction_receipt, "generator_receipt": generator_receipt,
        "generator_manifest": generator_manifest, "retention_probes": retention,
        "global_control_manifest": global_controls, "evaluation_inputs": normalized_inputs,
        "state_interchange_activation_patching": state_commitment,
        "execution_order_salt_sha256": salt, "episodes": episodes,
    }


def cfs2_branch_order(manifest: Mapping[str, Any], job_ids: tuple[str, ...] | list[str]) -> list[str]:
    salt = _sha(manifest.get("execution_order_salt_sha256"), "execution_order_salt_sha256")
    _require(len(job_ids) == len(set(job_ids)), "CFS-2 branch IDs must be unique")
    return sorted(job_ids, key=lambda job_id: hashlib.sha256(f"{salt}:{job_id}".encode()).hexdigest())
