"""Independent validation for the balanced, outcome-blind CFS-2 stimuli."""

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


RETENTION_SCHEMA = "nextlat_forgetting/cfs2_retention_manifest/1"
UPDATE_SCHEMA = "nextlat_forgetting/cfs2_update_manifest/1"
GLOBAL_CONTROL_SCHEMA = "nextlat_forgetting/cfs2_global_control_manifest/1"
RECEIPT_SCHEMA = "nextlat_forgetting/cfs2_materialization_receipt/1"
CONDITIONS = (
    ("high", "same"),
    ("high", "different"),
    ("low", "same"),
    ("low", "different"),
)
EXPECTED_OVERLAPS = {
    ("high", "same"): 18,
    ("high", "different"): 18,
    ("low", "same"): 8,
    ("low", "different"): 8,
}


class CFS2ValidationError(ValueError):
    """A CFS-2 artifact is malformed or violates the repaired protocol."""


def canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    return sha256_bytes(path.read_bytes())


def prompt_sha256(line: str) -> str:
    return sha256_text(parse_line(line).prompt)


def edge_overlap(left: str, right: str) -> int:
    return len(set(parse_line(left).edges) & set(parse_line(right).edges))


def overlap_decomposition(probe_line: str, update_line: str) -> tuple[int, int]:
    """Return shared update-answer edges and all other shared graph edges."""
    probe_edges = set(parse_line(probe_line).edges)
    update_answer = tuple(parse_line(update_line).answer)
    update_answer_edges = set(zip(update_answer, update_answer[1:]))
    answer_overlap = len(probe_edges & update_answer_edges)
    return answer_overlap, edge_overlap(probe_line, update_line) - answer_overlap


def _node_frequency(line: str) -> Counter[int]:
    return Counter(token_ids(parse_line(line).prompt))


def _answer_sha256(line: str) -> str:
    return sha256_text(",".join(str(value) for value in parse_line(line).answer))


def line_witness(line: str) -> dict[str, Any]:
    try:
        solved = validate_line(line)
    except GraphError as exc:
        raise CFS2ValidationError(f"invalid Path-Star line: {exc}") from exc
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
    prompt_hashes: frozenset[str]
    graph_keys: frozenset[str]
    identifiers: frozenset[str]
    sources: tuple[str, ...]


def _walk_manifest(value: Any, *, lines: list[str], identifiers: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key == "line" and isinstance(child, str) and "/" in child and "=" in child:
                lines.append(child)
            if (key == "id" or key.endswith("_id")) and isinstance(child, (str, int)):
                identifiers.add(str(child))
            _walk_manifest(child, lines=lines, identifiers=identifiers)
    elif isinstance(value, list):
        for child in value:
            _walk_manifest(child, lines=lines, identifiers=identifiers)


def build_legacy_index(root: pathlib.Path) -> LegacyIndex:
    """Index data identities without opening model outcomes or pilot losses.

    The index includes the training corpus and Path-Star stimulus manifests.  It
    explicitly refuses files whose names indicate losses, results, metrics, or
    checkpoints; those are neither needed nor scientifically permissible inputs
    to an outcome-blind stimulus repair.
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
                if line:
                    prompts.add(prompt_sha256(line))
                    graphs.add(canonical_key_from_line(line))

    manifests = root / "manifests"
    prohibited_name_parts = ("loss", "result", "metric", "checkpoint", "evidence")
    if manifests.exists():
        for path in sorted(manifests.rglob("*")):
            if not path.is_file() or "cfs2" in path.relative_to(manifests).parts:
                continue
            if path.suffix not in {".json", ".jsonl", ".txt"}:
                continue
            if any(part in path.name.lower() for part in prohibited_name_parts):
                continue
            try:
                raw = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if path.suffix == ".txt":
                candidate_lines = [
                    line for line in raw.splitlines() if "/" in line and "=" in line
                ]
                values: Iterable[Any] = ()
            else:
                try:
                    values = (
                        [json.loads(line) for line in raw.splitlines() if line.strip()]
                        if path.suffix == ".jsonl"
                        else [json.loads(raw)]
                    )
                except json.JSONDecodeError:
                    continue
                candidate_lines = []
            for value in values:
                _walk_manifest(value, lines=candidate_lines, identifiers=identifiers)
            found_pathstar = False
            for line in candidate_lines:
                try:
                    prompts.add(prompt_sha256(line))
                    graphs.add(canonical_key_from_line(line))
                    found_pathstar = True
                except (GraphError, ValueError):
                    continue
            if found_pathstar:
                sources.append(str(path.relative_to(root)))
    return LegacyIndex(
        prompt_hashes=frozenset(prompts),
        graph_keys=frozenset(graphs),
        identifiers=frozenset(identifiers),
        sources=tuple(sources),
    )


def _require_fields(
    row: Mapping[str, Any], fields: Sequence[str], label: str
) -> None:
    missing = [field for field in fields if field not in row]
    if missing:
        raise CFS2ValidationError(f"{label} missing fields {missing}")


def validate_retention_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_count: int,
    legacy: LegacyIndex | None = None,
) -> dict[str, Mapping[str, Any]]:
    if len(records) != expected_count:
        raise CFS2ValidationError(
            f"retention count {len(records)} != {expected_count}"
        )
    by_probe: dict[str, Mapping[str, Any]] = {}
    seen_prompt: set[str] = set()
    seen_graph: set[str] = set()
    for row in records:
        _require_fields(
            row,
            (
                "schema",
                "probe_id",
                "line",
                "prompt_sha256",
                "graph_key",
                "answer_sha256",
            ),
            "retention",
        )
        if row["schema"] != RETENTION_SCHEMA:
            raise CFS2ValidationError("retention schema mismatch")
        probe_id = str(row["probe_id"])
        if probe_id in by_probe:
            raise CFS2ValidationError(f"duplicate retention probe_id {probe_id}")
        witness = line_witness(str(row["line"]))
        for key in ("prompt_sha256", "graph_key", "answer_sha256"):
            if row[key] != witness[key]:
                raise CFS2ValidationError(f"retention {probe_id} has stale {key}")
        if witness["prompt_sha256"] in seen_prompt or witness["graph_key"] in seen_graph:
            raise CFS2ValidationError(f"retention {probe_id} duplicates a prior identity")
        if legacy and (
            witness["prompt_sha256"] in legacy.prompt_hashes
            or witness["graph_key"] in legacy.graph_keys
            or probe_id in legacy.identifiers
        ):
            raise CFS2ValidationError(
                f"retention {probe_id} collides with legacy/corpus identity"
            )
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
    probes = validate_retention_records(
        retention, expected_count=expected_probes, legacy=legacy
    )
    if set(updates) != set(CONDITIONS):
        raise CFS2ValidationError(
            "CFS-2 must contain exactly high/low x same/different banks"
        )
    units: dict[str, dict[tuple[str, str], Mapping[str, Any]]] = {}
    all_prompts = {str(row["prompt_sha256"]) for row in retention}
    all_graphs = {str(row["graph_key"]) for row in retention}
    global_token_counts = {condition: Counter() for condition in CONDITIONS}
    answer_lengths = {condition: Counter() for condition in CONDITIONS}
    probe_counts = {condition: Counter() for condition in CONDITIONS}
    for condition, rows in updates.items():
        if len(rows) != expected_updates:
            raise CFS2ValidationError(
                f"{condition} count {len(rows)} != {expected_updates}"
            )
        seen_units: set[str] = set()
        for row in rows:
            _require_fields(
                row,
                (
                    "schema",
                    "update_id",
                    "unit_id",
                    "probe_id",
                    "line",
                    "prompt_sha256",
                    "graph_key",
                    "answer_sha256",
                    "condition",
                    "edge_overlap_with_probe",
                    "answer_edge_overlap_with_probe",
                    "nonanswer_edge_overlap_with_probe",
                    "future_same_as_probe",
                ),
                "update",
            )
            if row["schema"] != UPDATE_SCHEMA:
                raise CFS2ValidationError("update schema mismatch")
            declared_condition = tuple(
                row["condition"].get(key)
                for key in ("overlap", "future_relation")
            )
            if declared_condition != condition:
                raise CFS2ValidationError(
                    f"update condition disagrees with bank {condition}"
                )
            unit_id, probe_id = str(row["unit_id"]), str(row["probe_id"])
            if unit_id in seen_units:
                raise CFS2ValidationError(f"duplicate {condition} unit {unit_id}")
            if probe_id not in probes:
                raise CFS2ValidationError(f"update {unit_id} names unknown probe")
            witness = line_witness(str(row["line"]))
            for key in ("prompt_sha256", "graph_key", "answer_sha256"):
                if row[key] != witness[key]:
                    raise CFS2ValidationError(f"update {unit_id} has stale {key}")
            overlap = edge_overlap(str(probes[probe_id]["line"]), str(row["line"]))
            if row["edge_overlap_with_probe"] != overlap:
                raise CFS2ValidationError(f"update {unit_id} reports wrong edge overlap")
            answer_overlap, nonanswer_overlap = overlap_decomposition(
                str(probes[probe_id]["line"]), str(row["line"])
            )
            if row["answer_edge_overlap_with_probe"] != answer_overlap:
                raise CFS2ValidationError(
                    f"update {unit_id} reports wrong answer-edge overlap"
                )
            if row["nonanswer_edge_overlap_with_probe"] != nonanswer_overlap:
                raise CFS2ValidationError(
                    f"update {unit_id} reports wrong nonanswer-edge overlap"
                )
            same = (
                witness["answer"]
                == line_witness(str(probes[probe_id]["line"]))["answer"]
            )
            if bool(row["future_same_as_probe"]) != same:
                raise CFS2ValidationError(f"update {unit_id} reports wrong future relation")
            if same != (condition[1] == "same"):
                raise CFS2ValidationError(f"update {unit_id} violates future relation")
            if witness["prompt_sha256"] in all_prompts or witness["graph_key"] in all_graphs:
                raise CFS2ValidationError(f"update {unit_id} duplicates CFS-2 identity")
            if legacy and (
                witness["prompt_sha256"] in legacy.prompt_hashes
                or witness["graph_key"] in legacy.graph_keys
                or str(row["update_id"]) in legacy.identifiers
                or unit_id in legacy.identifiers
            ):
                raise CFS2ValidationError(
                    f"update {unit_id} collides with legacy/corpus identity"
                )
            seen_units.add(unit_id)
            all_prompts.add(witness["prompt_sha256"])
            all_graphs.add(witness["graph_key"])
            global_token_counts[condition].update(
                token_ids(parse_line(str(row["line"])).prompt)
            )
            answer_lengths[condition][witness["answer_token_length"]] += 1
            probe_counts[condition][probe_id] += 1
            units.setdefault(unit_id, {})[condition] = row

    if len(units) != expected_updates:
        raise CFS2ValidationError("CFS-2 unit count mismatch")
    for unit_id, rows in units.items():
        if set(rows) != set(CONDITIONS):
            raise CFS2ValidationError(f"unit {unit_id} lacks a factorial cell")
        probe_ids = {str(row["probe_id"]) for row in rows.values()}
        if len(probe_ids) != 1:
            raise CFS2ValidationError(f"unit {unit_id} mixes retention probes")
        lines = [str(row["line"]) for row in rows.values()]
        if len({tuple(sorted(_node_frequency(line).items())) for line in lines}) != 1:
            raise CFS2ValidationError(f"unit {unit_id} changes prompt token multiset")
        if len(
            {
                (parse_line(line).source, parse_line(line).goal)
                for line in lines
            }
        ) != 1:
            raise CFS2ValidationError(f"unit {unit_id} changes source/goal")
        same_high = line_witness(str(rows[("high", "same")]["line"]))["answer"]
        same_low = line_witness(str(rows[("low", "same")]["line"]))["answer"]
        diff_high = line_witness(str(rows[("high", "different")]["line"]))["answer"]
        diff_low = line_witness(str(rows[("low", "different")]["line"]))["answer"]
        if same_high != same_low:
            raise CFS2ValidationError(f"unit {unit_id} fails same answer balance")
        if diff_high != diff_low:
            raise CFS2ValidationError(f"unit {unit_id} fails different answer balance")
        if same_high == diff_high:
            raise CFS2ValidationError(f"unit {unit_id} has no future intervention")
        for condition, row in rows.items():
            if row["edge_overlap_with_probe"] != EXPECTED_OVERLAPS[condition]:
                raise CFS2ValidationError(
                    f"unit {unit_id} {condition} has unexpected overlap"
                )
        answer_expected = {
            ("high", "same"): 4,
            ("low", "same"): 4,
            ("high", "different"): 3,
            ("low", "different"): 3,
        }
        nonanswer_expected = {
            condition: EXPECTED_OVERLAPS[condition] - answer_expected[condition]
            for condition in CONDITIONS
        }
        for condition, row in rows.items():
            if row["answer_edge_overlap_with_probe"] != answer_expected[condition]:
                raise CFS2ValidationError(
                    f"unit {unit_id} {condition} has unexpected answer-edge overlap"
                )
            if row["nonanswer_edge_overlap_with_probe"] != nonanswer_expected[condition]:
                raise CFS2ValidationError(
                    f"unit {unit_id} {condition} has unexpected nonanswer overlap"
                )
        if (
            rows[("high", "same")]["nonanswer_edge_overlap_with_probe"]
            - rows[("low", "same")]["nonanswer_edge_overlap_with_probe"]
            != rows[("high", "different")]["nonanswer_edge_overlap_with_probe"]
            - rows[("low", "different")]["nonanswer_edge_overlap_with_probe"]
        ):
            raise CFS2ValidationError(
                f"unit {unit_id} changes the nonanswer high-low contrast by relation"
            )
        if (
            rows[("high", "same")]["edge_overlap_with_probe"]
            != rows[("high", "different")]["edge_overlap_with_probe"]
            or rows[("low", "same")]["edge_overlap_with_probe"]
            != rows[("low", "different")]["edge_overlap_with_probe"]
        ):
            raise CFS2ValidationError(
                f"unit {unit_id} does not balance overlap within level"
            )

    if len({tuple(sorted(counter.items())) for counter in global_token_counts.values()}) != 1:
        raise CFS2ValidationError("the four update banks are not token-balanced")
    if len({tuple(sorted(counter.items())) for counter in answer_lengths.values()}) != 1:
        raise CFS2ValidationError("the four update banks are not answer-length balanced")
    if len({tuple(sorted(counter.items())) for counter in probe_counts.values()}) != 1:
        raise CFS2ValidationError("the four banks do not share one probe codebook")
    counts = Counter(str(row["probe_id"]) for row in updates[("high", "same")])
    if set(counts) != set(probes) or set(counts.values()) - {2, 3}:
        raise CFS2ValidationError("each retention probe must appear two or three times")
    if sum(value == 3 for value in counts.values()) != expected_updates - 2 * expected_probes:
        raise CFS2ValidationError("hash codebook has wrong extra assignment count")
    if codebook.get("schema") != "nextlat_forgetting/cfs2_hash_codebook/1":
        raise CFS2ValidationError("CFS-2 hash codebook schema mismatch")
    if codebook.get("n_probes") != expected_probes or codebook.get("n_updates") != expected_updates:
        raise CFS2ValidationError("CFS-2 codebook count mismatch")
    declared = codebook.get("unit_order")
    if not isinstance(declared, list) or len(declared) != expected_updates or set(declared) != set(units):
        raise CFS2ValidationError("CFS-2 codebook does not bind every unit")
    episodes = codebook.get("episodes")
    if not isinstance(episodes, list) or [row.get("episode") for row in episodes] != [0, 1]:
        raise CFS2ValidationError("CFS-2 requires two fixed episodes")
    if any(set(row.get("unit_order", ())) != set(units) for row in episodes):
        raise CFS2ValidationError("CFS-2 episode order does not cover every unit")

    if global_controls is None:
        if expected_global_controls is not None:
            raise CFS2ValidationError("CFS-2 requires global controls")
        return
    if expected_global_controls is None or len(global_controls) != expected_global_controls:
        raise CFS2ValidationError("CFS-2 global-control count mismatch")
    control_ids: set[str] = set()
    for row in global_controls:
        _require_fields(
            row,
            (
                "schema",
                "control_id",
                "line",
                "prompt_sha256",
                "graph_key",
                "answer_sha256",
            ),
            "global control",
        )
        if row["schema"] != GLOBAL_CONTROL_SCHEMA:
            raise CFS2ValidationError("global-control schema mismatch")
        control_id = str(row["control_id"])
        if control_id in control_ids:
            raise CFS2ValidationError(f"duplicate global control {control_id}")
        witness = line_witness(str(row["line"]))
        for key in ("prompt_sha256", "graph_key", "answer_sha256"):
            if row[key] != witness[key]:
                raise CFS2ValidationError(f"global control {control_id} has stale {key}")
        if witness["prompt_sha256"] in all_prompts or witness["graph_key"] in all_graphs:
            raise CFS2ValidationError(f"global control {control_id} duplicates identity")
        if legacy and (
            witness["prompt_sha256"] in legacy.prompt_hashes
            or witness["graph_key"] in legacy.graph_keys
            or control_id in legacy.identifiers
        ):
            raise CFS2ValidationError(
                f"global control {control_id} collides with legacy/corpus identity"
            )
        control_ids.add(control_id)
        all_prompts.add(witness["prompt_sha256"])
        all_graphs.add(witness["graph_key"])
