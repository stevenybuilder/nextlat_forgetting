#!/usr/bin/env python
"""Build truthful, source-bound evidence for the eleven preregistration gates.

The lifecycle is deliberately split to avoid a circular attestation:

``prepare``
    Create deterministic scientific-input manifests that must be included in the source archive.
``attest``
    After the immutable source archive and external gate inputs exist, execute the fixed fixture
    tests and create archive-excluded attestations.  The evidence index is published last.
``--diagnose``
    Perform the same read-only preflight without executing tests or writing any file.

This command cannot create a freeze PASS receipt and cannot launch compute.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import stat
import subprocess
import sys
import tarfile
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import validate_preregistration as V  # noqa: E402
from lurestar.validate import canonical_key_from_line  # noqa: E402


PREPARE_SCHEMA = "nextlat_forgetting/preregistration_preparation/1"
EXECUTION_SCHEMA = "nextlat_forgetting/preregistration_test_execution/1"
CONDITIONS = ("base", "repeat", "near_safe", "near_critical", "far_critical")
PREPARED = {
    "split_receipt": pathlib.Path("manifests/preregistration/eval_split.json"),
    "five_condition_manifest": pathlib.Path("manifests/preregistration/e_lure_conditions.json"),
    "disjointness_receipt": pathlib.Path("manifests/preregistration/pool_disjointness.json"),
}
ARCHIVE = pathlib.Path(".agent_state/project.tar.gz")
EVIDENCE = pathlib.Path(".agent_state/preregistration-evidence.json")
AMENDMENT = pathlib.Path("docs/PREREGISTRATION_AMENDMENT_2026-08-24.md")
SPEC = pathlib.Path("nextlat_v4_predictive_geometry_spec.md")
BUILDER = pathlib.Path("scripts/build_preregistration_evidence.py")
HMM_FAMILY = pathlib.Path("manifests/hmm_family.json")
HMM_MATERIALIZATION = pathlib.Path("manifests/hmm_family_materialization.json")
HMM_INVENTORY = pathlib.Path("manifests/hmm_family_inventory.sha256")
ADAPTATION_RECEIPT = pathlib.Path("manifests/adapt/adaptation_banks.json")
H3_BLOCK = V.H3_BLOCK_PATH
FULL_TEST_RECEIPT = pathlib.Path(".agent_state/confirmatory-test-receipt.json")
REVIEW_RECEIPT = pathlib.Path(".agent_state/confirmatory-review-receipt.json")

SKIP_DIRS = {
    ".venv", ".git", "__pycache__", "data", "output", "results", ".secrets",
    ".agent_state", "report", "upstream",
}
SKIP_TOP_FILES = {"HANDOFF.md"}


class EvidenceBlocked(RuntimeError):
    """A fail-closed preregistration precondition was not met."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    return sha256_bytes(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8"))


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def create_only_bytes(path: pathlib.Path, body: bytes) -> None:
    """Atomically create ``path`` or prove the existing bytes are identical."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != body:
            raise EvidenceBlocked(f"refusing to replace different frozen artifact: {path}")
        return
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def create_only_json(path: pathlib.Path, value: object) -> None:
    create_only_bytes(path, _json_bytes(value))


def _read_json(path: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceBlocked(f"{label} is missing or invalid: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceBlocked(f"{label} must be a JSON object: {path}")
    return value


def _permanent_h3_block(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Return the exact D40 block and sidecar, or fail closed on any deviation."""
    path = root / H3_BLOCK
    sidecar = path.with_name(path.name + ".sha256")
    value = _read_json(path, "permanent H3 block")
    if sha256_file(path) != V.H3_BLOCK_SHA256 or value != V.H3_BLOCK_DOCUMENT:
        raise EvidenceBlocked("permanent H3 block is mutated or is not the frozen D40 outcome")
    expected_sidecar = f"{V.H3_BLOCK_SHA256}  {path.name}\n".encode()
    if not sidecar.is_file() or sidecar.read_bytes() != expected_sidecar:
        raise EvidenceBlocked("permanent H3 block sidecar is missing or stale")
    return path, sidecar


def _jsonl(path: pathlib.Path, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise EvidenceBlocked(f"{label} is missing: {path}")
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidenceBlocked(f"{label} line {number} is invalid JSON") from exc
        if not isinstance(row, dict):
            raise EvidenceBlocked(f"{label} line {number} is not an object")
        rows.append(row)
    return rows


def build_split_documents(e_lure: pathlib.Path) -> tuple[dict, dict]:
    """Return the immutable split and exact five-condition manifests."""
    rows = _jsonl(e_lure, "E_lure manifest")
    if len(rows) != 2_000:
        raise EvidenceBlocked(f"E_lure must contain exactly 2000 quartets, found {len(rows)}")
    ranked: list[tuple[str, int, dict[str, Any]]] = []
    seen_ids: set[int] = set()
    seen_base_serializations: set[str] = set()
    for row in rows:
        quartet_id = row.get("quartet_id")
        conditions = row.get("conditions")
        if isinstance(quartet_id, bool) or not isinstance(quartet_id, int) or quartet_id in seen_ids:
            raise EvidenceBlocked("E_lure quartet_id values must be unique integers")
        if not isinstance(conditions, dict) or any(name not in conditions for name in CONDITIONS):
            raise EvidenceBlocked(f"quartet {quartet_id} lacks the exact five required conditions")
        base = conditions["base"]
        if not isinstance(base, dict) or not isinstance(base.get("line"), str):
            raise EvidenceBlocked(f"quartet {quartet_id} has no canonical base serialization")
        serialization = base["line"]
        if serialization in seen_base_serializations:
            raise EvidenceBlocked("E_lure contains duplicate canonical base serializations")
        seen_ids.add(quartet_id)
        seen_base_serializations.add(serialization)
        ranked.append((sha256_bytes(serialization.encode("utf-8")), quartet_id, row))
    ranked.sort(key=lambda item: (item[0], item[1]))
    e_white, e_score = ranked[:400], ranked[400:]

    def memberships(items: Sequence[tuple[str, int, dict[str, Any]]]) -> list[dict[str, Any]]:
        return [
            {"quartet_id": quartet_id, "base_serialization_sha256": base_sha}
            for base_sha, quartet_id, _ in items
        ]

    project_root = e_lure.resolve().parents[1]
    source = {"path": str(e_lure.resolve().relative_to(project_root)), "sha256": sha256_file(e_lure)}
    split = {
        "schema": "nextlat_forgetting/eval_split_source/1",
        "membership_rule": "ascending_sha256_of_canonical_base_serialization",
        "canonical_base_serialization": "UTF-8 bytes of conditions.base.line without newline",
        "source": source,
        "e_white": memberships(e_white),
        "e_score": memberships(e_score),
        "counts": {"e_white": 400, "e_score": 1600, "overlap": 0},
    }
    records = []
    for base_sha, quartet_id, row in ranked:
        condition_records: dict[str, dict[str, Any]] = {}
        for name in CONDITIONS:
            value = row["conditions"][name]
            if not isinstance(value, dict) or set(("line", "prompt_sha256", "graph_key", "answer")) - set(value):
                raise EvidenceBlocked(f"quartet {quartet_id}/{name} identity is incomplete")
            line = value["line"]
            if not isinstance(line, str):
                raise EvidenceBlocked(f"quartet {quartet_id}/{name} line is not text")
            condition_records[name] = {
                "line_sha256": sha256_bytes(line.encode("utf-8")),
                "prompt_sha256": value["prompt_sha256"],
                "graph_key": value["graph_key"],
                "answer": value["answer"],
            }
        if set(condition_records) != set(CONDITIONS):  # defensive exactness
            raise EvidenceBlocked("five-condition record contains a missing or extra condition")
        records.append({
            "quartet_id": quartet_id,
            "base_serialization_sha256": base_sha,
            "conditions": condition_records,
        })
    five = {
        "schema": "nextlat_forgetting/e_lure_conditions_source/1",
        "conditions": list(CONDITIONS),
        "e_score_count": 1600,
        "source": source,
        "records": records,
    }
    return split, five


def _line_graph_keys(path: pathlib.Path, label: str) -> set[str]:
    if not path.is_file():
        raise EvidenceBlocked(f"{label} is missing: {path}")
    keys: set[str] = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            key = canonical_key_from_line(line)
        except Exception as exc:
            raise EvidenceBlocked(f"{label} line {number} is not a valid Path-Star item") from exc
        if key in keys:
            raise EvidenceBlocked(f"{label} contains a duplicate canonical graph")
        keys.add(key)
    return keys


def _eval_graph_keys(e_lure: pathlib.Path) -> set[str]:
    keys: set[str] = set()
    for row in _jsonl(e_lure, "E_lure manifest"):
        for name in CONDITIONS:
            try:
                key = canonical_key_from_line(row["conditions"][name]["line"])
            except Exception as exc:
                raise EvidenceBlocked(f"invalid E_lure condition {row.get('quartet_id')}/{name}") from exc
            keys.add(key)
    return keys


def pairwise_overlap_counts(domains: Mapping[str, set[str]]) -> dict[str, int]:
    labels = sorted(domains)
    return {
        f"{left}__{right}": len(domains[left] & domains[right])
        for index, left in enumerate(labels) for right in labels[index + 1:]
    }


def require_pairwise_disjoint(domains: Mapping[str, set[str]]) -> dict[str, int]:
    overlaps = pairwise_overlap_counts(domains)
    bad = {name: count for name, count in overlaps.items() if count}
    if bad:
        raise EvidenceBlocked(f"canonical graph pools are not pairwise disjoint: {bad}")
    return overlaps


def _adaptation_outputs(root: pathlib.Path) -> tuple[dict[str, pathlib.Path], dict]:
    receipt_path = root / ADAPTATION_RECEIPT
    receipt = _read_json(receipt_path, "adaptation-bank materialization receipt")
    wanted = {
        "graph_5_5_bnear_5000.txt", "graph_5_5_bmid_5000.txt", "graph_5_5_bfar_5000.txt",
        "graph_5_5_bnearval_2000.txt", "graph_5_5_bmidval_2000.txt",
        "graph_5_5_bfarval_2000.txt",
    }
    outputs = receipt.get("outputs")
    if (receipt.get("status") != "materialized" or receipt.get("schema_version") != 1 or
            not isinstance(outputs, dict) or set(outputs) != wanted or
            receipt.get("scientific_selection_performed") is not False):
        raise EvidenceBlocked("adaptation-bank receipt is not a complete outcome-blind six-bank freeze")
    paths: dict[str, pathlib.Path] = {}
    for name in sorted(wanted):
        path = receipt_path.parent / name
        count = 5_000 if "val_" not in name and "val" not in name else 2_000
        if not path.is_file() or outputs[name] != sha256_file(path):
            raise EvidenceBlocked(f"adaptation output hash mismatch: {path}")
        if len(path.read_text(encoding="utf-8").splitlines()) != count:
            raise EvidenceBlocked(f"adaptation output count mismatch: {path}")
        paths[name] = path
    return paths, receipt


def _validate_adaptation_inputs(
    root: pathlib.Path, outputs: Mapping[str, pathlib.Path], receipt: Mapping[str, Any],
) -> dict[str, pathlib.Path]:
    """Replay the frozen selectors; do not trust a materialization receipt's claims alone."""
    import materialize_adaptation_banks as banks

    sources = receipt.get("sources")
    required = {
        "near_manifest", "far_candidates", "far_selection", "mid_candidates", "mid_selection",
        "near_validation", "mid_validation", "far_validation", "acquisition_provenance",
    }
    if not isinstance(sources, dict) or set(sources) != required:
        raise EvidenceBlocked(
            f"adaptation source role mismatch: missing={sorted(required-set(sources or {}))}, "
            f"extra={sorted(set(sources or {})-required)}"
        )
    paths: dict[str, pathlib.Path] = {}
    for role in sorted(required):
        record = sources[role]
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise EvidenceBlocked(f"adaptation source {role} binding is malformed")
        path = (root / str(record["path"])).resolve()
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise EvidenceBlocked(f"adaptation source {role} hash mismatch")
        try:
            banks.verify_sidecar(path)
        except banks.GateError as exc:
            raise EvidenceBlocked(f"adaptation source {role} sidecar mismatch: {exc}") from exc
        paths[role] = path
    try:
        near_sha = banks.verify_sidecar(paths["near_manifest"])
        far_sha = banks.verify_sidecar(paths["far_candidates"])
        mid_sha = banks.verify_sidecar(paths["mid_candidates"])
        near = banks.load_manifest(paths["near_manifest"], banks.NEAR_COUNT, "B_near")
        far_candidates = banks.load_manifest(
            paths["far_candidates"], banks.FAR_CANDIDATE_COUNT, "B_far"
        )
        mid_candidate_count = banks.mid_candidate_count_from_selection(paths["mid_selection"])
        mid_candidates = banks.load_manifest(paths["mid_candidates"], mid_candidate_count, "B_mid")
        mid = banks.select_mid(
            near, mid_candidates, paths["mid_selection"],
            near_sha256=near_sha, candidates_sha256=mid_sha,
        )
        far = banks.select_far(
            near, far_candidates, paths["far_selection"],
            near_sha256=near_sha, candidates_sha256=far_sha,
        )
        validations = {
            label: banks.load_validation(paths[f"{label}_validation"])
            for label in ("near", "mid", "far")
        }
        validation_hashes = {
            label: banks.verify_sidecar(paths[f"{label}_validation"])
            for label in ("near", "mid", "far")
        }
        banks.verify_acquisition_provenance(
            paths["acquisition_provenance"],
            near_sha256=validation_hashes["near"],
            mid_sha256=validation_hashes["mid"],
            far_sha256=validation_hashes["far"],
        )
        selected = {"near": near, "mid": mid, "far": far, **{
            f"{label}_validation": values for label, values in validations.items()
        }}
        banks.require_matched_path_distributions({
            label: values for label, values in selected.items() if "validation" not in label
        })
        banks.require_matched_path_distributions(validations)
        banks.require_disjoint(selected)
    except banks.GateError as exc:
        raise EvidenceBlocked(f"adaptation scientific replay failed: {exc}") from exc
    for label, values in selected.items():
        expected = "".join(item.line + "\n" for item in values).encode()
        output = outputs[banks.OUTPUT_NAMES[label]]
        if output.read_bytes() != expected:
            raise EvidenceBlocked(f"adaptation output differs from replayed selection: {output}")
    return paths


def build_disjointness_document(root: pathlib.Path) -> dict:
    h3_block, h3_sidecar = _permanent_h3_block(root)
    training = root / "data/stargraph/graph_5_5_sample_200000.txt"
    e_lure = root / "manifests/e_lure.jsonl"
    domains = {
        "training": _line_graph_keys(training, "training corpus"),
        "evaluation": _eval_graph_keys(e_lure),
    }
    overlaps = require_pairwise_disjoint(domains)
    return {
        "schema": "nextlat_forgetting/pool_disjointness_source/1",
        "identity": "canonical solver graph key (edge ordering and recorded answer ignored)",
        "domains": {
            name: {"count": len(values), "set_sha256": canonical_sha256(sorted(values))}
            for name, values in sorted(domains.items())
        },
        "pairwise_overlap_counts": overlaps,
        "checks": {
            "all_pairwise_overlap_count": sum(overlaps.values()),
            "training_evaluation_overlap_count": len(domains["training"] & domains["evaluation"]),
            # H3 was prospectively dropped by D40, so no adaptation pool is permitted to exist in
            # the confirmatory design. This zero is an exclusion claim, not an untested overlap.
            "adaptation_evaluation_overlap_count": 0,
        },
        "sources": {
            "training": {"path": str(training.relative_to(root)), "sha256": sha256_file(training)},
            "evaluation": {"path": str(e_lure.relative_to(root)), "sha256": sha256_file(e_lure)},
            "h3_permanent_block": {
                "path": str(h3_block.relative_to(root)), "sha256": sha256_file(h3_block),
            },
            "h3_permanent_block_sidecar": {
                "path": str(h3_sidecar.relative_to(root)), "sha256": sha256_file(h3_sidecar),
            },
        },
    }


def prepare(root: pathlib.Path) -> dict:
    e_lure = root / "manifests/e_lure.jsonl"
    split, five = build_split_documents(e_lure)
    create_only_json(root / PREPARED["split_receipt"], split)
    create_only_json(root / PREPARED["five_condition_manifest"], five)
    # This is intentionally last: a missing or mutated permanent H3 block yields a truthful BLOCK,
    # while the already deterministic split files remain useful and retry-safe.
    disjointness = build_disjointness_document(root)
    create_only_json(root / PREPARED["disjointness_receipt"], disjointness)
    return {
        "schema": PREPARE_SCHEMA,
        "status": "READY_FOR_SOURCE_SNAPSHOT",
        "artifacts": {
            role: {"path": str(path), "sha256": sha256_file(root / path)}
            for role, path in PREPARED.items()
        },
    }


def _included_current_files(root: pathlib.Path) -> dict[str, tuple[str, int]]:
    result: dict[str, tuple[str, int]] = {}
    for top in sorted(root.iterdir(), key=lambda p: p.name):
        if top.name in SKIP_DIRS or top.name in SKIP_TOP_FILES or top.name.startswith("."):
            continue
        candidates: Iterable[pathlib.Path] = [top] if top.is_file() else top.rglob("*")
        for path in candidates:
            relative = path.relative_to(root)
            parts = relative.parts
            if any(part in SKIP_DIRS for part in parts):
                continue
            basename = path.name
            if (basename == ".env" or basename.startswith(".env.") or
                    basename in {"adc.json", "application_default_credentials.json"} or
                    str(relative).endswith((".pt", ".ckpt", ".tar.gz"))):
                continue
            if path.is_symlink() or not path.is_file():
                continue
            result[relative.as_posix()] = (sha256_file(path), stat.S_IMODE(path.stat().st_mode))
    return result


def assert_archive_fresh(root: pathlib.Path, archive: pathlib.Path) -> str:
    if not archive.is_file():
        raise EvidenceBlocked(f"source archive is missing: {archive}")
    archived: dict[str, tuple[str, int]] = {}
    try:
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar.getmembers():
                if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
                    raise EvidenceBlocked(f"source archive has unsafe member: {member.name}")
                if not member.isfile():
                    continue
                stream = tar.extractfile(member)
                if stream is None:
                    raise EvidenceBlocked(f"source archive member is unreadable: {member.name}")
                archived[member.name] = (sha256_bytes(stream.read()), member.mode & 0o777)
    except (tarfile.TarError, OSError) as exc:
        raise EvidenceBlocked(f"source archive is invalid: {exc}") from exc
    current = _included_current_files(root)
    if archived != current:
        missing = sorted(set(current) - set(archived))
        extra = sorted(set(archived) - set(current))
        changed = sorted(name for name in set(current) & set(archived) if current[name] != archived[name])
        raise EvidenceBlocked(
            f"source archive is stale: missing={missing[:8]}, extra={extra[:8]}, changed={changed[:8]}"
        )
    return sha256_file(archive)


def _hash_inventory_paths(root: pathlib.Path, inventory: pathlib.Path) -> list[pathlib.Path]:
    seen: set[pathlib.Path] = set()
    lines = inventory.read_text(encoding="utf-8").splitlines()
    for number, line in enumerate(lines, 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise EvidenceBlocked(f"HMM inventory line {number} is malformed")
        path = (root / match.group(2)).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise EvidenceBlocked("HMM inventory path escapes the project") from exc
        if path in seen or not path.is_file() or sha256_file(path) != match.group(1):
            raise EvidenceBlocked(f"HMM inventory entry is missing, duplicate, or stale: {path}")
        seen.add(path)
    if not seen:
        raise EvidenceBlocked("HMM inventory is empty")
    return sorted(seen)


def _verify_hash_inventory(root: pathlib.Path, inventory: pathlib.Path) -> int:
    return len(_hash_inventory_paths(root, inventory))


def _literal_schema_constants(path: pathlib.Path, names: set[str]) -> dict[str, str]:
    """Read top-level literal schema constants without importing scientific runtime code."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise EvidenceBlocked(f"schema authority is unreadable: {path}: {exc}") from exc
    values: dict[str, str] = {}
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        value = statement.value
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id in names:
                values[target.id] = value.value
    if set(values) != names:
        raise EvidenceBlocked(
            f"schema authority {path} lacks literal constants: {sorted(names-set(values))}"
        )
    return values


def _semantic_sources(root: pathlib.Path, role: str, source_sha: str) -> tuple[list[pathlib.Path], dict]:
    """Return genuine raw subjects and any payload additions for a role."""
    del source_sha
    if role in PREPARED:
        path = root / PREPARED[role]
        _read_json(path, role)
        split, five = build_split_documents(root / "manifests/e_lure.jsonl")
        expected = (split if role == "split_receipt" else five if role == "five_condition_manifest"
                    else build_disjointness_document(root))
        if path.read_bytes() != _json_bytes(expected):
            raise EvidenceBlocked(f"{role} differs from its deterministic source inputs")
        raw = [path, root / "manifests/e_lure.jsonl"]
        if role == "disjointness_receipt":
            h3_block, h3_sidecar = _permanent_h3_block(root)
            raw.extend([
                root / "data/stargraph/graph_5_5_sample_200000.txt",
                h3_block, h3_sidecar,
            ])
        return raw, {}
    if role in V.H3_BLOCK_ROLES:
        path, sidecar = _permanent_h3_block(root)
        return [path, sidecar], {"subject": {
            "path": str(path.resolve()), "sha256": sha256_file(path),
            "schema": V.H3_BLOCK_SCHEMA,
        }}
    if role in {"hmm_family_manifest", "hmm_materialization_receipt", "hmm_te_receipt"}:
        family_path, material_path, inventory = (
            root / HMM_FAMILY, root / HMM_MATERIALIZATION, root / HMM_INVENTORY,
        )
        family = _read_json(family_path, "HMM family")
        material = _read_json(material_path, "HMM materialization")
        if (family.get("schema") != V.ARTIFACT_SCHEMAS["8"]["hmm_family_manifest"] or
                family.get("required_regimes") != list(V.REGIMES) or
                family.get("primary_regime") is not None):
            raise EvidenceBlocked("HMM family is not the exact three-regime amendment family")
        inventory_paths = _hash_inventory_paths(root, inventory)
        n_inventory = len(inventory_paths)
        if (material.get("schema") != V.ARTIFACT_SCHEMAS["8"]["hmm_materialization_receipt"] or
                material.get("status") != "complete" or material.get("n_artifacts") != n_inventory or
                material.get("required_regimes") != list(V.REGIMES) or
                material.get("model_outcomes_inspected") is not False or
                material.get("inventory_sha256") != sha256_file(inventory) or
                material.get("family_sha256") != family.get("payload_sha256")):
            raise EvidenceBlocked("HMM family materialization receipt is incomplete or stale")
        certs: dict[str, dict[str, Any]] = {}
        for regime in V.REGIMES:
            try:
                te = family["regimes"][regime]["linear_certificate"]["matrices"][
                    "transition_times_emission"
                ]
                rank, sigma = te["rank"], te["sigma_min"]
            except (KeyError, TypeError) as exc:
                raise EvidenceBlocked(f"HMM {regime} TE certificate is missing") from exc
            if rank != 4 or isinstance(sigma, bool) or not isinstance(sigma, (int, float)) or sigma <= .05:
                raise EvidenceBlocked(f"HMM {regime} fails rank(TE)=4 and sigma_min(TE)>0.05")
            certs[regime] = {"rank_te": rank, "sigma_min_te": sigma}
        paths = [family_path, material_path, inventory, *inventory_paths]
        if role == "hmm_family_manifest":
            return paths, {"subject": {
                "path": str(family_path), "sha256": sha256_file(family_path),
                "schema": V.ARTIFACT_SCHEMAS["8"][role],
            }}
        if role == "hmm_materialization_receipt":
            return paths, {"subject": {
                "path": str(material_path), "sha256": sha256_file(material_path),
                "schema": V.ARTIFACT_SCHEMAS["8"][role],
            }}
        return paths, {
            "te_certificates": certs, "rank_required": 4,
            "sigma_min_exclusive_threshold": 0.05,
        }
    if role == "lurestar_schema_receipt":
        extractor = root / "scripts/extract_lurestar_evidence.py"
        evaluator = root / "scripts/evaluate_lurestar_checkpoints.py"
        estimator = root / "src/lurestar/evaluate.py"
        representations = root / "src/lurestar/representations.py"
        extractor_tests = root / "tests/test_lurestar_evidence_extractor.py"
        evaluator_tests = root / "tests/test_lurestar_checkpoint_evaluator.py"
        representation_tests = root / "tests/test_representations.py"
        materializer = root / "scripts/materialize_lurestar_evaluation.py"
        materializer_tests = root / "tests/test_materialize_lurestar_evaluation.py"
        hmm_evaluator = root / "scripts/evaluate_hmm_checkpoints.py"
        hmm_runner = root / "scripts/run_hmm_matrix.py"
        hmm_aggregate = root / "src/hmm_geometry/aggregate.py"
        hmm_runner_tests = root / "tests/test_run_hmm_matrix.py"
        hmm_family_tests = root / "tests/test_hmm_family.py"
        extractor_schemas = _literal_schema_constants(
            extractor, {"JOB_SCHEMA", "PROGRESS_SCHEMA", "EVIDENCE_SCHEMA"},
        )
        evaluator_schemas = _literal_schema_constants(
            evaluator, {"SCHEMA", "REPORT_SCHEMA", "RECEIPT_SCHEMA"},
        )
        observed = {
            "extraction_job": extractor_schemas["JOB_SCHEMA"],
            "extraction_progress": extractor_schemas["PROGRESS_SCHEMA"],
            "evidence_npz": extractor_schemas["EVIDENCE_SCHEMA"],
            "evidence_receipt": extractor_schemas["EVIDENCE_SCHEMA"],
            "evaluation_manifest": evaluator_schemas["SCHEMA"],
            "confirmatory_report": evaluator_schemas["REPORT_SCHEMA"],
            "evaluation_receipt": evaluator_schemas["RECEIPT_SCHEMA"],
        }
        hmm_schemas = (
            _literal_schema_constants(hmm_evaluator, {"SCHEMA"})["SCHEMA"],
            _literal_schema_constants(hmm_aggregate, {"SCHEMA"})["SCHEMA"],
        )
        try:
            witnesses = V.derive_lurestar_semantic_witnesses(root)
        except ValueError as exc:
            raise EvidenceBlocked(f"Lure-Star semantic witnesses are incomplete: {exc}") from exc
        h3_block, h3_sidecar = _permanent_h3_block(root)
        witness_names = set(witnesses)
        core = {
            "npsi_formula_and_denominator", "paired_student_t_and_loso",
            "exact_sha_base_id_folds", "nested_h2_m0_delta_r2_identical_folds",
            "extractor_npsi_and_audit", "report_schema_and_required_statistics",
            "h1_four_state_classifier", "binary_h2_secondary_ceiling_status",
            "all_12_hooks_parity_and_cleanup", "bst_forward_only_all_12_hooks",
            "whitener_exact_mahalanobis_parity", "whitener_heldout_claim_fail_closed",
            "atomic_lurestar_exact_15_dry_run",
            "atomic_lurestar_invalid_fifteenth_zero_invocations",
            "atomic_lurestar_stale_fifteenth_zero_invocations",
            "atomic_lurestar_exact_cell_set", "atomic_hmm_exact_30_acceptance",
            "atomic_hmm_exact_30_refusal", "hmm_fisher_z_exact_and_boundary_fail_closed",
            "hmm_two_sided_sign_flip_floor", "hmm_two_sided_mde_and_exact_family",
            "hmm_null_and_heterogeneity_report_only",
        }
        additions = {
            "schemas": list(dict.fromkeys((*observed.values(), *hmm_schemas))),
            "missing_metrics_refused": "invalid_cells_terminal_schema" in witness_names,
            "extra_metrics_refused": "h3_mechanism_array_refusal" in witness_names,
            "invalid_cells_emitted": {
                "tampered_field_invalid_emission", "invalid_cells_terminal_schema",
                "terminal_required_fields_fail_closed",
            }.issubset(witness_names),
            "nulls_emitted": {
                "non_equivalence_nulls_and_manipulation_failures",
                "terminal_required_fields_fail_closed",
            }.issubset(witness_names),
            "manipulation_failures_emitted": {
                "non_equivalence_nulls_and_manipulation_failures",
                "terminal_required_fields_fail_closed",
            }.issubset(witness_names),
            "lurestar_schema_contract": observed,
            "lurestar_confirmatory_scope": (
                "base_only_h1_h2" if "base_only_checkpoint_scope" in witness_names else "unverified"
            ),
            "h1_h2_metrics_preserved": core.issubset(witness_names),
            "permanent_h3_exclusion_required": (
                "base_only_checkpoint_scope" in witness_names
            ),
            "h3_fields_refused": "h3_analysis_and_old_schema_refusal" in witness_names,
            "adaptation_fields_refused": "adaptation_checkpoint_refusal" in witness_names,
            "mechanism_fields_refused": "h3_mechanism_array_refusal" in witness_names,
            "semantic_witnesses": witnesses,
        }
        return [
            extractor, evaluator, estimator, representations, extractor_tests,
            evaluator_tests, representation_tests, materializer, materializer_tests,
            hmm_evaluator, hmm_runner, hmm_aggregate, hmm_runner_tests, hmm_family_tests,
            h3_block, h3_sidecar,
        ], additions
    if role == "full_suite_receipt":
        path = root / FULL_TEST_RECEIPT
        value = _read_json(path, "full-suite clearance receipt")
        if (value.get("schema") != V.ARTIFACT_SCHEMAS["11"][role] or
                value.get("source_sha256") != sha256_file(root / ARCHIVE) or
                value.get("outcome") != "PASS" or value.get("exit_code") != 0 or
                isinstance(value.get("tests_passed"), bool) or
                not isinstance(value.get("tests_passed"), int) or value["tests_passed"] <= 0):
            raise EvidenceBlocked("full-suite receipt is not a passing exact-source receipt")
        return [path], {"exit_code": 0, "tests_passed": value["tests_passed"]}
    if role == "independent_review_receipt":
        path = root / REVIEW_RECEIPT
        value = _read_json(path, "independent-review clearance receipt")
        report = root / str(value.get("report_path", ""))
        if (value.get("schema") != V.ARTIFACT_SCHEMAS["11"][role] or
                value.get("source_sha256") != sha256_file(root / ARCHIVE) or
                value.get("verdict") != "PASS" or not str(value.get("reviewer", "")).strip() or
                not report.is_file() or value.get("report_sha256") != sha256_file(report)):
            raise EvidenceBlocked("independent review is not a passing exact-source receipt")
        return [path, report], {"reviewer": value["reviewer"]}
    return [], {}


TEST_NODES: dict[str, tuple[str, ...]] = {
    "split_receipt": (
        "tests/test_build_preregistration_evidence.py::test_split_is_hash_sorted_exact_and_disjoint",
    ),
    "five_condition_manifest": (
        "tests/test_build_preregistration_evidence.py::test_five_condition_manifest_is_exact",
    ),
    "disjointness_receipt": (
        "tests/test_build_preregistration_evidence.py::test_pairwise_disjointness_mutation_blocks",
    ),
    "whitener_fixture_receipt": (
        "tests/test_representations.py::test_whitened_euclidean_equals_mahalanobis_under_the_same_covariance",
        "tests/test_representations.py::test_whitener_refuses_to_score_items_from_its_own_fitting_pool",
    ),
    "metric_fixture_receipt": (
        "tests/test_representations.py::test_centered_cosine_matches_a_naive_loop_reference",
        "tests/test_representations.py::test_psi_under_both_metrics_agrees_on_the_sign_of_a_planted_effect",
    ),
    **{
        role: (
            "tests/test_build_preregistration_evidence.py::"
            "test_shipped_permanent_h3_block_is_exact_and_immutable",
        )
        for role in V.H3_BLOCK_ROLES
    },
    "hmm_family_manifest": (
        "tests/test_hmm_family.py::test_shipped_family_is_amendment_exact",
    ),
    "hmm_materialization_receipt": (
        "tests/test_hmm_family.py::test_family_orchestration_is_exactly_30_isolated_jobs",
    ),
    "hmm_te_receipt": (
        "tests/test_hmm_family.py::test_family_is_model_blind_complete_and_certifies_primary_sigma_min",
        "tests/test_hmm_family.py::test_family_pair_candidates_are_selected_on_exact_future_js",
    ),
    "aggregate_fixture_receipt": (
        "tests/test_hmm_family.py::test_aggregate_requires_all_regimes_models_seeds_and_metrics",
    ),
    "multiplicity_fixture_receipt": (
        "tests/test_hmm_family.py::test_exact_sign_flip_discreteness_is_explicit",
        "tests/test_hmm_family.py::test_aggregate_requires_all_regimes_models_seeds_and_metrics",
    ),
    "lurestar_schema_receipt": (
        *V.LURESTAR_SEMANTIC_TEST_NODES,
    ),
    "hmm_schema_receipt": (
        "tests/test_hmm_family.py::test_aggregate_requires_all_regimes_models_seeds_and_metrics",
    ),
}


ROLE_MODULES: dict[str, tuple[pathlib.Path, ...]] = {
    "whitener_fixture_receipt": (pathlib.Path("src/lurestar/evaluate.py"),),
    "metric_fixture_receipt": (pathlib.Path("src/lurestar/evaluate.py"),),
    **{role: () for role in V.H3_BLOCK_ROLES},
    "hmm_family_manifest": (pathlib.Path("src/hmm_geometry/family.py"),),
    "hmm_materialization_receipt": (pathlib.Path("scripts/materialize_hmm_family.py"),),
    "hmm_te_receipt": (pathlib.Path("src/hmm_geometry/family.py"),),
    "aggregate_fixture_receipt": (pathlib.Path("src/hmm_geometry/aggregate.py"),),
    "multiplicity_fixture_receipt": (pathlib.Path("src/hmm_geometry/aggregate.py"),),
    "lurestar_schema_receipt": (
        pathlib.Path("scripts/extract_lurestar_evidence.py"),
        pathlib.Path("scripts/evaluate_lurestar_checkpoints.py"),
        pathlib.Path("scripts/materialize_lurestar_evaluation.py"),
        pathlib.Path("scripts/run_hmm_matrix.py"),
        pathlib.Path("src/lurestar/evaluate.py"),
        pathlib.Path("src/lurestar/representations.py"),
        pathlib.Path("src/hmm_geometry/aggregate.py"),
        pathlib.Path("tests/test_lurestar_evidence_extractor.py"),
        pathlib.Path("tests/test_lurestar_checkpoint_evaluator.py"),
        pathlib.Path("tests/test_representations.py"),
        pathlib.Path("tests/test_materialize_lurestar_evaluation.py"),
        pathlib.Path("tests/test_run_hmm_matrix.py"),
        pathlib.Path("tests/test_hmm_family.py"),
    ),
    "hmm_schema_receipt": (pathlib.Path("src/hmm_geometry/aggregate.py"),),
}


def _test_count(output: str) -> int:
    matches = re.findall(r"(?:^|\s)(\d+) passed(?:\s|,|$)", output)
    return int(matches[-1]) if matches else 0


def run_role_tests(root: pathlib.Path, role: str, source_sha: str,
                   state_root: pathlib.Path,
                   semantic_subject: Mapping[str, Any] | None = None) -> pathlib.Path:
    nodes = TEST_NODES[role]
    command = [sys.executable, "-m", "pytest", *nodes, "-q"]
    completed = subprocess.run(
        command, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        check=False,
    )
    count = _test_count(completed.stdout)
    status = "PASS" if completed.returncode == 0 and count > 0 else "FAIL"
    modules = [BUILDER, *ROLE_MODULES.get(role, ())]
    execution = {
        "schema": EXECUTION_SCHEMA,
        "status": status,
        "role": role,
        "source_archive_sha256": source_sha,
        "command": command,
        "pytest_nodes": list(nodes),
        "exit_code": completed.returncode,
        "tests_passed": count,
        "output_sha256": sha256_bytes(completed.stdout.encode()),
        "producer": {"path": str((root / BUILDER).resolve()), "sha256": sha256_file(root / BUILDER)},
        "modules": [
            {"path": str((root / path).resolve()), "sha256": sha256_file(root / path)}
            for path in modules
        ],
        "semantic_subject": dict(semantic_subject) if semantic_subject is not None else {
            key: V.EXPECTED_CHECKS[next(
                gate for gate, roles in V.ARTIFACT_SCHEMAS.items() if role in roles
            )][key] for key in V.ROLE_CHECK_KEYS[role]
        },
    }
    execution_path = state_root / "tests" / f"{role}.execution.json"
    strict_path = state_root / "tests" / f"{role}.json"
    create_only_json(execution_path, execution)
    strict: dict[str, object] = {
        "schema": V.TEST_EVIDENCE_SCHEMA,
        "status": status,
        "role": role,
        "source_archive_sha256": source_sha,
        "exit_code": completed.returncode,
        "tests_passed": count,
    }
    if role == "lurestar_schema_receipt":
        if semantic_subject is None or not isinstance(
            semantic_subject.get("semantic_witnesses"), dict
        ):
            raise EvidenceBlocked("Lure-Star tests lack derived semantic witnesses")
        strict.update({
            "pytest_nodes": list(nodes),
            "modules": execution["modules"],
            "semantic_witnesses_sha256": canonical_sha256(
                semantic_subject["semantic_witnesses"]
            ),
        })
    create_only_json(strict_path, strict)
    if status != "PASS":
        raise EvidenceBlocked(
            f"targeted fixture tests failed for {role}; output_sha256={execution['output_sha256']}"
        )
    return strict_path


def _derived_test_evidence(root: pathlib.Path, role: str, source_sha: str,
                           state_root: pathlib.Path, tests_passed: int = 1) -> pathlib.Path:
    path = state_root / "tests" / f"{role}.json"
    create_only_json(path, {
        "schema": V.TEST_EVIDENCE_SCHEMA, "status": "PASS", "role": role,
        "source_archive_sha256": source_sha, "exit_code": 0, "tests_passed": tests_passed,
    })
    return path


def _role_payload(gate: str, role: str, extras: Mapping[str, Any]) -> dict:
    if gate == "10" and role == "lurestar_schema_receipt":
        required = set(V.ROLE_CHECK_KEYS[role]) | {"semantic_witnesses"}
        if set(extras) != required:
            raise EvidenceBlocked(
                "Lure-Star gate-10 payload was not fully derived from sources/witnesses: "
                f"missing={sorted(required-set(extras))}, extra={sorted(set(extras)-required)}"
            )
        return {"claim": role, **dict(extras)}
    checks = V.EXPECTED_CHECKS[gate]
    payload = {"claim": role, **{key: checks[key] for key in V.ROLE_CHECK_KEYS[role]}}
    payload.update(extras)
    return payload


def _binding(path: pathlib.Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def attest(root: pathlib.Path) -> dict:
    archive = (root / ARCHIVE).resolve()
    source_sha = assert_archive_fresh(root, archive)
    state_root = root / ".agent_state" / "preregistration" / source_sha
    producer = root / BUILDER
    gates: dict[str, dict[str, Any]] = {}
    for gate, schemas in V.ARTIFACT_SCHEMAS.items():
        checks = json.loads(json.dumps(V.EXPECTED_CHECKS[gate]))
        artifacts: list[dict[str, str]] = []
        if gate == "1":
            for role, path, schema in (
                ("amendment", root / AMENDMENT, V.ARTIFACT_SCHEMAS[gate]["amendment"]),
                ("authoritative_spec", root / SPEC, V.ARTIFACT_SCHEMAS[gate]["authoritative_spec"]),
                ("source_snapshot", archive, V.ARTIFACT_SCHEMAS[gate]["source_snapshot"]),
            ):
                if not path.is_file():
                    raise EvidenceBlocked(f"gate 1 artifact missing: {path}")
                artifacts.append({"role": role, "path": str(path), "sha256": sha256_file(path),
                                  "schema": schema})
            gates[gate] = {
                "schema": f"nextlat_forgetting/preregistration_gate_{gate}/1",
                "artifacts": artifacts, "checks": checks,
            }
            continue

        if gate == "8":
            _sources, additions = _semantic_sources(root, "hmm_te_receipt", source_sha)
            checks["te_certificates"] = additions["te_certificates"]
        for role, schema in schemas.items():
            raw_sources, additions = _semantic_sources(root, role, source_sha)
            payload = _role_payload(gate, role, additions)
            if role in {"full_suite_receipt", "independent_review_receipt"}:
                test_path = _derived_test_evidence(
                    root, role, source_sha, state_root,
                    int(additions.get("tests_passed", 1)),
                )
            else:
                test_path = run_role_tests(
                    root, role, source_sha, state_root,
                    {key: value for key, value in payload.items() if key != "claim"},
                )
            execution_path = state_root / "tests" / f"{role}.execution.json"
            source_bindings = [_binding(archive), *(_binding(path) for path in raw_sources)]
            if execution_path.is_file():
                source_bindings.append(_binding(execution_path))
            # Preserve order while refusing duplicate semantic bindings.
            deduped: list[dict[str, str]] = []
            seen: set[str] = set()
            for item in source_bindings:
                if item["path"] not in seen:
                    seen.add(item["path"])
                    deduped.append(item)
            document = {
                "schema": schema,
                "attestation_schema": V.ATTESTATION_SCHEMA,
                "status": "PASS",
                "role": role,
                "source_archive_sha256": source_sha,
                "payload_sha256": V.canonical_json_sha256(payload),
                "payload": payload,
                "producer": _binding(producer),
                "source_bindings": deduped,
                "test_bindings": [_binding(test_path)],
            }
            output = state_root / f"g{gate}-{role}.json"
            create_only_json(output, document)
            artifacts.append({
                "role": role, "path": str(output.resolve()), "sha256": sha256_file(output),
                "schema": schema,
            })
        gates[gate] = {
            "schema": f"nextlat_forgetting/preregistration_gate_{gate}/1",
            "artifacts": artifacts, "checks": checks,
        }
    evidence = {"schema": V.SCHEMA, "gates": gates}
    # Validate through a candidate first. Its one expected global issue is its noncanonical path;
    # every scientific gate must already pass before the public index is created.
    candidate = state_root / "preregistration-evidence.candidate.json"
    create_only_json(candidate, evidence)
    candidate_result = V.validate(candidate, amendment=root / AMENDMENT, spec=root / SPEC)
    expected_path_issue = (
        "evidence index must live at the archive-excluded "
        f"{(root / EVIDENCE).resolve()} path"
    )
    if (candidate_result.get("status") != "BLOCK" or
            candidate_result.get("global_issues") != [expected_path_issue] or
            any(gate.get("status") != "PASS" for gate in candidate_result.get("gates", []))):
        raise EvidenceBlocked(f"candidate evidence failed validation: {candidate_result}")
    # Validator requires the canonical .agent_state path. This is the final published evidence
    # index, not the separate freeze PASS receipt and not launch authorization.
    create_only_json(root / EVIDENCE, evidence)
    result = V.validate(root / EVIDENCE, amendment=root / AMENDMENT, spec=root / SPEC)
    if result.get("status") != "PASS":
        raise EvidenceBlocked(f"constructed evidence did not validate: {result}")
    return {
        "schema": V.SCHEMA, "status": "EVIDENCE_READY_NOT_LAUNCH_CLEARANCE",
        "source_archive_sha256": source_sha, "evidence": _binding(root / EVIDENCE),
        "remaining_action": "run validate_preregistration.py to mint the separate freeze receipt",
    }


def diagnose(root: pathlib.Path) -> dict:
    issues: list[str] = []
    prepared: dict[str, dict[str, Any]] = {}
    try:
        split, five = build_split_documents(root / "manifests/e_lure.jsonl")
    except EvidenceBlocked as exc:
        issues.append(str(exc))
    else:
        wanted = {"split_receipt": split, "five_condition_manifest": five}
        for role, value in wanted.items():
            path = root / PREPARED[role]
            prepared[role] = {
                "path": str(path), "exists": path.is_file(),
                "exact": path.is_file() and path.read_bytes() == _json_bytes(value),
            }
            if not prepared[role]["exact"]:
                issues.append(f"{role} is missing or differs from deterministic preparation")
    try:
        disjoint = build_disjointness_document(root)
    except EvidenceBlocked as exc:
        issues.append(str(exc))
    else:
        path = root / PREPARED["disjointness_receipt"]
        prepared["disjointness_receipt"] = {
            "path": str(path), "exists": path.is_file(),
            "exact": path.is_file() and path.read_bytes() == _json_bytes(disjoint),
        }
        if not prepared["disjointness_receipt"]["exact"]:
            issues.append("disjointness_receipt is missing or differs from deterministic preparation")
    try:
        source_sha = assert_archive_fresh(root, root / ARCHIVE)
    except EvidenceBlocked as exc:
        source_sha = None
        issues.append(str(exc))
    missing_roles: list[str] = []
    # Raw scientific subjects can be diagnosed even before the archive exists. Fixture-only roles
    # are implementation-ready when their producer modules and fixed pytest nodes exist; attest
    # will actually execute them only after a fresh archive has been made.
    diagnostic_sha = source_sha or "0" * 64
    for gate, roles in V.ARTIFACT_SCHEMAS.items():
        if gate == "1":
            continue
        for role in roles:
            try:
                _semantic_sources(root, role, diagnostic_sha)
                for module in (BUILDER, *ROLE_MODULES.get(role, ())):
                    if not (root / module).is_file():
                        raise EvidenceBlocked(f"producer module is missing: {module}")
                for node in TEST_NODES.get(role, ()):
                    if not (root / node.split("::", 1)[0]).is_file():
                        raise EvidenceBlocked(f"targeted pytest module is missing: {node}")
            except EvidenceBlocked as exc:
                missing_roles.append(role)
                issues.append(f"{role}: {exc}")
    return {
        "schema": "nextlat_forgetting/preregistration_evidence_diagnostic/1",
        "status": "READY" if not issues else "BLOCK",
        "read_only": True,
        "source_archive_sha256": source_sha,
        "prepared": prepared,
        "missing_roles": sorted(set(missing_roles)),
        "issues": issues,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("prepare", "attest"))
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--project-root", type=pathlib.Path, default=ROOT)
    args = parser.parse_args(argv)
    if args.diagnose == (args.mode is not None):
        parser.error("choose exactly one of prepare, attest, or --diagnose")
    root = args.project_root.resolve()
    try:
        result = diagnose(root) if args.diagnose else (
            prepare(root) if args.mode == "prepare" else attest(root)
        )
    except EvidenceBlocked as exc:
        result = {"schema": PREPARE_SCHEMA, "status": "BLOCK", "issues": [str(exc)]}
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2 if result.get("status") == "BLOCK" else 0


if __name__ == "__main__":
    raise SystemExit(main())
