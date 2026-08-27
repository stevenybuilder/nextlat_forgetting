"""CFS-1's isolated, common adaptation contract.

This module intentionally does not import or modify the retired Lure-Star H3
adapter.  CFS-1 is a new NextLat-only randomized intervention experiment.  The
generator owns the update streams; this module owns the narrow contract the
branch runner consumes: two prespecified episodes, four construction-matched
arms, exactly 500 full-parameter teacher-forced CE updates per branch, and no
auxiliary NextLat loss.

``validate_update_manifest`` is deliberately an opaque boundary.  It checks
identity, outcome-blindness, and the complete arm/episode surface without
reimplementing the graph generator's solver or looking at model outcomes.  A
generator may add fields, but it cannot omit or alter this public contract.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
from typing import Any, Mapping


CFS1_MANIFEST_SCHEMA = "nextlat_forgetting/cfs1_update_manifest/1"
CFS1_ADAPTATION_SCHEMA = "nextlat_forgetting/cfs1_adaptation/1"
CFS1_CONTRACT = "cfs1_full_parameter_next_token_ce_v1"
CFS1_UPDATE_STEPS = 500
CFS1_PARENT_SEEDS = (1234, 1235, 1236, 1237, 1238, 2234, 2235, 2236)
# Episode identities are numeric experimental factors, not presentation labels.  Their
# ordering here is part of the frozen execution and paired-analysis contract.
CFS1_EPISODES = (0, 1)
CFS1_ARMS = (
    "high_different",
    "low_different",
    "high_same",
    "low_same",
)
CFS1_ORDER_ALGORITHM = "sha256-sort-v1"
EVALUATION_INPUTS = (
    "margin",
    "retention_ce",
    "retention_exact_path",
    "global_controls",
    "state_drift",
    "pregeometry",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class CFS1AdaptationError(RuntimeError):
    """The new causal-forgetting intervention contract is not satisfied."""


def sha256_file(path: str | pathlib.Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def contract_sha256(path: str | pathlib.Path | None = None) -> str:
    return sha256_file(pathlib.Path(path) if path is not None else pathlib.Path(__file__))


def canonical_json_sha256(document: Any) -> str:
    try:
        payload = json.dumps(document, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CFS1AdaptationError("CFS-1 contract document is not canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CFS1AdaptationError(message)


def _sha(value: object, label: str) -> str:
    _require(isinstance(value, str) and _SHA256.fullmatch(value) is not None,
             f"{label} must be a lowercase SHA-256")
    return str(value)


def _safe_relative(value: object, label: str) -> str:
    _require(isinstance(value, str) and value, f"{label} must be a nonempty relative path")
    pure = pathlib.PurePosixPath(value)
    _require(not pure.is_absolute() and all(part not in ("", ".", "..") for part in pure.parts),
             f"{label} must be a safe relative path")
    return str(pure)


def _artifact(root: pathlib.Path, record: object, label: str) -> dict[str, str]:
    _require(isinstance(record, Mapping), f"{label} must be an artifact mapping")
    relative = _safe_relative(record.get("path"), f"{label}.path")
    expected = _sha(record.get("sha256"), f"{label}.sha256")
    path = (root / relative).resolve()
    _require(root == path or root in path.parents, f"{label} escapes the manifest root")
    _require(path.is_file() and not path.is_symlink(), f"{label} artifact is absent or symlinked")
    actual = sha256_file(path)
    _require(actual == expected, f"{label} SHA-256 mismatch")
    return {"path": relative, "sha256": expected}


def _load_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CFS1AdaptationError(f"CFS-1 update manifest is invalid: {path}") from exc
    _require(isinstance(document, dict), "CFS-1 update manifest must be a JSON object")
    return document


def validate_update_manifest(path: str | pathlib.Path) -> dict[str, Any]:
    """Verify CFS-1's opaque generator interface and all training input hashes.

    Required generator surface (relative paths are rooted at this manifest):

    ``schema``, ``status: FROZEN``, outcome-blind ``construction`` receipt,
    ``design`` (NextLat, 500 CE-only full-parameter updates), a fixed SHA sorting
    salt, one untouched retention-probe artifact, and both episodes with every
    one of the four arms.  The runner returns a normalized, immutable-friendly
    mapping and never reads graph content beyond hashing it.
    """
    manifest_path = pathlib.Path(path).resolve()
    _require(manifest_path.is_file() and not manifest_path.is_symlink(),
             "CFS-1 update manifest is absent or symlinked")
    root = manifest_path.parent.resolve()
    document = _load_json(manifest_path)
    _require(document.get("schema") == CFS1_MANIFEST_SCHEMA,
             "unexpected CFS-1 update manifest schema")
    _require(document.get("status") == "FROZEN", "CFS-1 update manifest is not frozen")

    construction = document.get("construction")
    _require(isinstance(construction, Mapping), "CFS-1 manifest lacks construction receipt")
    for key in ("model_outcomes_inspected", "training_outcomes_inspected",
                "retention_outcomes_inspected"):
        _require(construction.get(key) is False,
                 f"CFS-1 construction must be outcome-blind: {key}")
    _require(construction.get("matching") == "construction_matched",
             "CFS-1 may only use construction-matched updates")
    _require(construction.get("randomized_assignment") is True,
             "CFS-1 arms must be randomized before adaptation")
    construction_receipt = _artifact(root, construction.get("receipt"), "construction.receipt")
    generator_receipt = _artifact(root, document.get("generator_receipt"), "generator_receipt")
    generator_manifest = _artifact(root, document.get("generator_manifest"), "generator_manifest")
    retention_probes = _artifact(root, document.get("retention_probes"), "retention_probes")
    global_control_manifest = _artifact(
        root, document.get("global_control_manifest"), "global_control_manifest"
    )

    design = document.get("design")
    _require(isinstance(design, Mapping), "CFS-1 manifest lacks design")
    _require(design.get("model") == "nextlat", "CFS-1 primary intervention is NextLat-only")
    _require(design.get("adaptation_steps") == CFS1_UPDATE_STEPS,
             "CFS-1 must request exactly 500 adaptation updates")
    _require(design.get("full_parameter") is True,
             "CFS-1 adaptation must update all parameters")
    _require(design.get("loss") == "teacher_forced_next_token_cross_entropy",
             "CFS-1 adaptation must use teacher-forced next-token CE")
    _require(tuple(design.get("arms", ())) == CFS1_ARMS,
             "CFS-1 arm order/membership is not frozen")
    _require(tuple(design.get("episodes", ())) == CFS1_EPISODES,
             "CFS-1 episode order/membership is not frozen")
    _require(document.get("execution_order_algorithm") == CFS1_ORDER_ALGORITHM,
             "CFS-1 has no approved hash-randomized execution algorithm")
    execution_salt = _sha(document.get("execution_order_salt_sha256"),
                          "execution_order_salt_sha256")

    evaluation_inputs = document.get("evaluation_inputs")
    _require(isinstance(evaluation_inputs, Mapping)
             and tuple(evaluation_inputs.keys()) == EVALUATION_INPUTS,
             "CFS-1 evaluation input membership/order is not frozen")
    normalized_inputs = {
        name: _artifact(root, evaluation_inputs[name], f"evaluation_inputs.{name}")
        for name in EVALUATION_INPUTS
    }
    raw_episodes = document.get("episodes")
    _require(isinstance(raw_episodes, list) and len(raw_episodes) == len(CFS1_EPISODES),
             "CFS-1 requires exactly two episodes")
    normalized_episodes: dict[str, dict[str, Any]] = {}
    for expected_id, episode in zip(CFS1_EPISODES, raw_episodes):
        _require(isinstance(episode, Mapping), "CFS-1 episode must be a mapping")
        _require(episode.get("episode") == expected_id,
                 "CFS-1 numeric episode IDs must be canonical and ordered")
        _sha(episode.get("episode_sha256"), f"episode{expected_id}.episode_sha256")
        arms = episode.get("arms")
        _require(isinstance(arms, Mapping) and tuple(arms.keys()) == CFS1_ARMS,
                 f"{expected_id} arm membership/order is not canonical")
        normalized_arms = {}
        for arm in CFS1_ARMS:
            arm_record = arms[arm]
            _require(isinstance(arm_record, Mapping), f"episode{expected_id}.{arm} must be a mapping")
            expected_overlap, expected_relation = arm.split("_", 1)
            _require(arm_record.get("overlap") == expected_overlap,
                     f"episode{expected_id}.{arm} overlap binding is wrong")
            _require(arm_record.get("future_relation") == expected_relation,
                     f"episode{expected_id}.{arm} future-relation binding is wrong")
            bank = _artifact(root, arm_record, f"episode{expected_id}.arms.{arm}")
            bank_name = pathlib.PurePosixPath(bank["path"]).name
            _require(bank_name.startswith("graph_5_5_") and bank_name.endswith(".txt"),
                     f"episode{expected_id}.{arm} must bind a raw graph_5_5_*.txt training bank")
            normalized_arms[arm] = bank
        normalized_episodes[expected_id] = {
            "episode": expected_id,
            "episode_sha256": str(episode["episode_sha256"]),
            "arms": normalized_arms,
        }

    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "root": str(root),
        "schema": CFS1_MANIFEST_SCHEMA,
        "construction_receipt": construction_receipt,
        "generator_receipt": generator_receipt,
        "generator_manifest": generator_manifest,
        "retention_probes": retention_probes,
        "global_control_manifest": global_control_manifest,
        "evaluation_inputs": normalized_inputs,
        "execution_order_salt_sha256": execution_salt,
        "episodes": normalized_episodes,
    }


def cfs1_branch_order(manifest: Mapping[str, Any], job_ids: list[str] | tuple[str, ...]) -> list[str]:
    """Return the outcome-independent, manifest-committed branch execution order."""
    salt = _sha(manifest.get("execution_order_salt_sha256"), "execution_order_salt_sha256")
    _require(len(set(job_ids)) == len(job_ids), "CFS-1 branch ids must be unique")
    return sorted(job_ids, key=lambda job_id: hashlib.sha256(
        (salt + ":" + job_id).encode("utf-8")).hexdigest())


def validate_nextlat_ce_only_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless a branch config represents CFS-1's common objective."""
    _require(config.get("use_nextlat") is True and config.get("use_bst", False) is False,
             "CFS-1 branch config must select NextLat and never BST")
    model = config.get("model")
    _require(isinstance(model, Mapping), "CFS-1 branch config lacks model block")
    coefficients = {key: model.get(key) for key in ("lambda_mse", "lambda_kl", "lambda_ce")}
    _require(all(value == 0.0 for value in coefficients.values()),
             "CFS-1 NextLat auxiliary coefficients must all be exactly zero")
    return {
        "schema": CFS1_ADAPTATION_SCHEMA,
        "contract": CFS1_CONTRACT,
        "contract_sha256": contract_sha256(),
        "family": "nextlat",
        "full_parameter": True,
        "loss": "teacher_forced_next_token_cross_entropy",
        "nextlat_auxiliary_coefficients": coefficients,
    }
