"""Independent, model-blind validation for CFS-1 construction artifacts.

CFS-1 intentionally does not consume checkpoints, losses, learned distances, or
pilot selections.  This module verifies serialized Path-Star examples from scratch
using the project's independent graph solver, and separately verifies the
construction-level randomisation and balance contracts.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from lurestar.validate import (
    GraphError,
    canonical_key_from_line,
    parse_line,
    sha256_text,
    token_ids,
    validate_line,
)


RETENTION_SCHEMA = "nextlat_forgetting/cfs1_retention_manifest/1"
UPDATE_SCHEMA = "nextlat_forgetting/cfs1_update_manifest/1"
GLOBAL_SCHEMA = "nextlat_forgetting/cfs1_global_manifest/1"
GLOBAL_CONTROL_SCHEMA = "nextlat_forgetting/cfs1_global_control_manifest/1"
RECEIPT_SCHEMA = "nextlat_forgetting/cfs1_materialization_receipt/1"
CONDITIONS = (
    ("high", "same"),
    ("high", "different"),
    ("low", "same"),
    ("low", "different"),
)


class CFS1ValidationError(ValueError):
    """An immutable CFS-1 artifact is malformed or violates the protocol."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    return sha256_bytes(path.read_bytes())


def prompt_sha256(line: str) -> str:
    return sha256_text(parse_line(line).prompt)


def edge_overlap(left: str, right: str) -> int:
    """Order-independent edge overlap, derived from serialized lines."""
    return len(set(parse_line(left).edges) & set(parse_line(right).edges))


def _node_frequency(line: str) -> Counter[int]:
    return Counter(token_ids(parse_line(line).prompt))


def _answer_sha256(line: str) -> str:
    return sha256_text(",".join(str(value) for value in parse_line(line).answer))


def line_witness(line: str) -> dict[str, Any]:
    """Return all trusted identity fields, after solver certification."""
    try:
        solved = validate_line(line)
    except GraphError as exc:
        raise CFS1ValidationError(f"invalid Path-Star line: {exc}") from exc
    parsed = parse_line(line)
    return {
        "prompt_sha256": prompt_sha256(line),
        "graph_key": canonical_key_from_line(line),
        "answer_sha256": _answer_sha256(line),
        "source": parsed.source,
        "goal": parsed.goal,
        "answer": list(solved.path),
        "prompt_token_length": len(token_ids(parsed.prompt)),
        "answer_token_length": len(parsed.answer),
    }


@dataclass(frozen=True)
class LegacyIndex:
    """Identifiers reserved by corpus and every legacy experiment artifact."""

    prompt_hashes: frozenset[str]
    graph_keys: frozenset[str]
    identifiers: frozenset[str]
    sources: tuple[str, ...]


def _walk_identifiers(value: Any, *, out: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "line" and isinstance(child, str) and "/" in child and "=" in child:
                try:
                    out.add(prompt_sha256(child))
                    out.add(canonical_key_from_line(child))
                except (CFS1ValidationError, GraphError, ValueError):
                    pass
            if key == "id" or key.endswith("_id"):
                if isinstance(child, (str, int)):
                    out.add(str(child))
            _walk_identifiers(child, out=out)
    elif isinstance(value, list):
        for child in value:
            _walk_identifiers(child, out=out)


def build_legacy_index(root: pathlib.Path) -> LegacyIndex:
    """Index corpus, H1/H2/H3/HMM manifests, and their public identifiers.

    CFS-1's own directory is explicitly excluded: a rerun must verify the same
    artifact, rather than treating an already-materialized CFS-1 bank as legacy
    contamination.  This is a data-identity check only; no model result is opened.
    """
    root = pathlib.Path(root)
    prompts: set[str] = set()
    graphs: set[str] = set()
    identifiers: set[str] = set()
    sources: list[str] = []

    corpus = root / "data/stargraph/graph_5_5_sample_200000.txt"
    if corpus.exists():
        sources.append(str(corpus.relative_to(root)))
        with corpus.open() as handle:
            for raw in handle:
                line = raw.strip()
                if not line:
                    continue
                prompts.add(prompt_sha256(line))
                graphs.add(canonical_key_from_line(line))

    manifests = root / "manifests"
    if manifests.exists():
        for path in sorted(manifests.rglob("*")):
            if not path.is_file() or "cfs1" in path.relative_to(manifests).parts:
                continue
            if path.suffix not in {".json", ".jsonl", ".txt"}:
                continue
            try:
                raw = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            sources.append(str(path.relative_to(root)))
            if path.suffix == ".txt":
                for line in raw.splitlines():
                    if "/" in line and "=" in line:
                        try:
                            prompts.add(prompt_sha256(line))
                            graphs.add(canonical_key_from_line(line))
                        except (GraphError, ValueError):
                            continue
                continue
            try:
                values: Iterable[Any] = (
                    [json.loads(line) for line in raw.splitlines() if line.strip()]
                    if path.suffix == ".jsonl"
                    else [json.loads(raw)]
                )
            except json.JSONDecodeError:
                continue
            for value in values:
                _walk_identifiers(value, out=identifiers)
                if isinstance(value, Mapping) and isinstance(value.get("line"), str):
                    line = value["line"]
                    try:
                        prompts.add(prompt_sha256(line))
                        graphs.add(canonical_key_from_line(line))
                    except (GraphError, ValueError):
                        pass
    return LegacyIndex(
        prompt_hashes=frozenset(prompts), graph_keys=frozenset(graphs),
        identifiers=frozenset(identifiers), sources=tuple(sources),
    )


def _require_record_fields(record: Mapping[str, Any], fields: Sequence[str], label: str) -> None:
    missing = [field for field in fields if field not in record]
    if missing:
        raise CFS1ValidationError(f"{label} missing fields {missing}")


def validate_retention_records(
    records: Sequence[Mapping[str, Any]], *, expected_count: int, legacy: LegacyIndex | None = None
) -> dict[str, Mapping[str, Any]]:
    if len(records) != expected_count:
        raise CFS1ValidationError(f"retention count {len(records)} != {expected_count}")
    by_probe: dict[str, Mapping[str, Any]] = {}
    seen_prompt: set[str] = set()
    seen_graph: set[str] = set()
    for row in records:
        _require_record_fields(row, ("schema", "probe_id", "line", "prompt_sha256", "graph_key", "answer_sha256"), "retention")
        if row["schema"] != RETENTION_SCHEMA:
            raise CFS1ValidationError("retention schema mismatch")
        probe_id = str(row["probe_id"])
        if probe_id in by_probe:
            raise CFS1ValidationError(f"duplicate retention probe_id {probe_id}")
        witness = line_witness(str(row["line"]))
        for key in ("prompt_sha256", "graph_key", "answer_sha256"):
            if row[key] != witness[key]:
                raise CFS1ValidationError(f"retention {probe_id} has stale {key}")
        if witness["prompt_sha256"] in seen_prompt or witness["graph_key"] in seen_graph:
            raise CFS1ValidationError(f"retention {probe_id} duplicates a prior probe identity")
        if legacy and (witness["prompt_sha256"] in legacy.prompt_hashes or witness["graph_key"] in legacy.graph_keys or probe_id in legacy.identifiers):
            raise CFS1ValidationError(f"retention {probe_id} collides with legacy/corpus identity")
        seen_prompt.add(witness["prompt_sha256"])
        seen_graph.add(witness["graph_key"])
        by_probe[probe_id] = row
    return by_probe


def validate_bundle(
    retention: Sequence[Mapping[str, Any]],
    updates: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    codebook: Mapping[str, Any],
    *,
    expected_probes: int,
    expected_updates: int,
    global_controls: Sequence[Mapping[str, Any]] | None = None,
    expected_global_controls: int | None = None,
    legacy: LegacyIndex | None = None,
) -> None:
    """Validate CFS-1's four banks independently of generator bookkeeping."""
    probes = validate_retention_records(retention, expected_count=expected_probes, legacy=legacy)
    if set(updates) != set(CONDITIONS):
        raise CFS1ValidationError("CFS-1 must contain exactly high/low x same/different banks")
    units: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = {}
    all_prompts: set[str] = {str(row["prompt_sha256"]) for row in retention}
    all_graphs: set[str] = {str(row["graph_key"]) for row in retention}
    global_token_counts: dict[tuple[str, str], Counter[int]] = {condition: Counter() for condition in CONDITIONS}
    answer_lengths: dict[tuple[str, str], Counter[int]] = {condition: Counter() for condition in CONDITIONS}
    probe_counts: dict[tuple[str, str], Counter[str]] = {condition: Counter() for condition in CONDITIONS}
    for condition, rows in updates.items():
        if len(rows) != expected_updates:
            raise CFS1ValidationError(f"{condition} count {len(rows)} != {expected_updates}")
        seen_units: set[str] = set()
        for row in rows:
            _require_record_fields(row, ("schema", "update_id", "unit_id", "probe_id", "line", "prompt_sha256", "graph_key", "answer_sha256", "condition", "edge_overlap_with_probe", "future_same_as_probe"), "update")
            if row["schema"] != UPDATE_SCHEMA:
                raise CFS1ValidationError("update schema mismatch")
            if tuple(row["condition"].get(key) for key in ("overlap", "future_relation")) != condition:
                raise CFS1ValidationError(f"update condition field disagrees with bank {condition}")
            unit_id, probe_id = str(row["unit_id"]), str(row["probe_id"])
            if unit_id in seen_units:
                raise CFS1ValidationError(f"duplicate {condition} unit {unit_id}")
            if probe_id not in probes:
                raise CFS1ValidationError(f"update {unit_id} names unknown probe {probe_id}")
            witness = line_witness(str(row["line"]))
            for key in ("prompt_sha256", "graph_key", "answer_sha256"):
                if row[key] != witness[key]:
                    raise CFS1ValidationError(f"update {unit_id} has stale {key}")
            overlap = edge_overlap(str(probes[probe_id]["line"]), str(row["line"]))
            if row["edge_overlap_with_probe"] != overlap:
                raise CFS1ValidationError(f"update {unit_id} reports the wrong edge overlap")
            same = witness["answer"] == line_witness(str(probes[probe_id]["line"]))["answer"]
            if bool(row["future_same_as_probe"]) != same:
                raise CFS1ValidationError(f"update {unit_id} reports the wrong future relation")
            if same != (condition[1] == "same"):
                raise CFS1ValidationError(f"update {unit_id} violates its future relation")
            if witness["prompt_sha256"] in all_prompts or witness["graph_key"] in all_graphs:
                raise CFS1ValidationError(f"update {unit_id} duplicates a CFS-1 graph/prompt identity")
            if legacy and (witness["prompt_sha256"] in legacy.prompt_hashes or witness["graph_key"] in legacy.graph_keys or str(row["update_id"]) in legacy.identifiers or unit_id in legacy.identifiers):
                raise CFS1ValidationError(f"update {unit_id} collides with legacy/corpus identity")
            seen_units.add(unit_id)
            all_prompts.add(witness["prompt_sha256"])
            all_graphs.add(witness["graph_key"])
            global_token_counts[condition].update(token_ids(parse_line(str(row["line"])).prompt))
            answer_lengths[condition][witness["answer_token_length"]] += 1
            probe_counts[condition][probe_id] += 1
            units.setdefault(unit_id, {})[condition] = row

    expected_overlaps = {
        ("high", "same"): 18, ("high", "different"): 18,
        ("low", "same"): 8, ("low", "different"): 7,
    }
    for unit_id, rows in units.items():
        if set(rows) != set(CONDITIONS):
            raise CFS1ValidationError(f"unit {unit_id} is not represented in all four banks")
        probe_ids = {str(row["probe_id"]) for row in rows.values()}
        if len(probe_ids) != 1:
            raise CFS1ValidationError(f"unit {unit_id} mixes retention probes")
        all_lines = [str(row["line"]) for row in rows.values()]
        if len({tuple(sorted(_node_frequency(line).items())) for line in all_lines}) != 1:
            raise CFS1ValidationError(f"unit {unit_id} changes the prompt token multiset")
        if len({(parse_line(line).source, parse_line(line).goal) for line in all_lines}) != 1:
            raise CFS1ValidationError(f"unit {unit_id} changes source/goal tokens")
        same_answer = line_witness(str(rows[("high", "same")]["line"]))["answer"]
        if same_answer != line_witness(str(rows[("low", "same")]["line"]))["answer"]:
            raise CFS1ValidationError(f"unit {unit_id} does not answer-balance same banks")
        different_answer = line_witness(str(rows[("high", "different")]["line"]))["answer"]
        if different_answer != line_witness(str(rows[("low", "different")]["line"]))["answer"]:
            raise CFS1ValidationError(f"unit {unit_id} does not answer-balance different banks")
        if same_answer == different_answer:
            raise CFS1ValidationError(f"unit {unit_id} has no future intervention")
        for condition, row in rows.items():
            if row["edge_overlap_with_probe"] != expected_overlaps[condition]:
                raise CFS1ValidationError(f"unit {unit_id} {condition} has unexpected overlap")

    if len({tuple(sorted(counter.items())) for counter in global_token_counts.values()}) != 1:
        raise CFS1ValidationError("the four update banks are not token-balanced")
    if len({tuple(sorted(counter.items())) for counter in answer_lengths.values()}) != 1:
        raise CFS1ValidationError("the four update banks are not path-length balanced")
    if len({tuple(sorted(counter.items())) for counter in probe_counts.values()}) != 1:
        raise CFS1ValidationError("the four update banks do not share the same probe codebook")
    counts = Counter(str(row["probe_id"]) for row in updates[("high", "same")])
    if set(counts) != set(probes) or set(counts.values()) - {2, 3}:
        raise CFS1ValidationError("each retention probe must appear exactly two or three times")
    if sum(value == 3 for value in counts.values()) != expected_updates - 2 * expected_probes:
        raise CFS1ValidationError("hash codebook has the wrong number of extra probe assignments")
    if codebook.get("schema") != "nextlat_forgetting/cfs1_hash_codebook/1":
        raise CFS1ValidationError("CFS-1 hash codebook schema mismatch")
    if codebook.get("n_probes") != expected_probes or codebook.get("n_updates") != expected_updates:
        raise CFS1ValidationError("CFS-1 codebook count mismatch")
    declared = codebook.get("unit_order")
    if not isinstance(declared, list) or set(declared) != set(units) or len(declared) != expected_updates:
        raise CFS1ValidationError("CFS-1 codebook must bind every update unit exactly once")
    if global_controls is None:
        if expected_global_controls is not None:
            raise CFS1ValidationError("CFS-1 requires a global-control manifest")
        return
    if expected_global_controls is None or len(global_controls) != expected_global_controls:
        raise CFS1ValidationError("CFS-1 global-control count mismatch")
    control_ids: set[str] = set()
    for row in global_controls:
        _require_record_fields(row, ("schema", "control_id", "line", "prompt_sha256", "graph_key", "answer_sha256"), "global control")
        if row["schema"] != GLOBAL_CONTROL_SCHEMA:
            raise CFS1ValidationError("global-control schema mismatch")
        control_id = str(row["control_id"])
        if control_id in control_ids:
            raise CFS1ValidationError(f"duplicate global control id {control_id}")
        witness = line_witness(str(row["line"]))
        for key in ("prompt_sha256", "graph_key", "answer_sha256"):
            if row[key] != witness[key]:
                raise CFS1ValidationError(f"global control {control_id} has stale {key}")
        if witness["prompt_sha256"] in all_prompts or witness["graph_key"] in all_graphs:
            raise CFS1ValidationError(f"global control {control_id} collides with CFS-1 identity")
        if legacy and (witness["prompt_sha256"] in legacy.prompt_hashes or witness["graph_key"] in legacy.graph_keys or control_id in legacy.identifiers):
            raise CFS1ValidationError(f"global control {control_id} collides with legacy/corpus identity")
        control_ids.add(control_id)
        all_prompts.add(witness["prompt_sha256"])
        all_graphs.add(witness["graph_key"])
