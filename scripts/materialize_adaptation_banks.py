#!/usr/bin/env python3
"""Materialize H3 adaptation text banks without making a scientific choice.

This script is a gate, not an exploratory selector.  It can serialize the already-frozen
B_near manifest directly.  B_mid and B_far are deliberately impossible to serialize until
a separate non-confirmatory pilot has produced hash-pinned candidate tables and mappings.
Likewise, the independent validation banks must be supplied as separately hashed inputs.

The tool never opens a checkpoint, results directory, metrics file, or ledger.  Its only
inputs are immutable manifests and provenance artifacts.  It recomputes the preregistered
middle-distance ranking solely to verify the supplied mapping and never inspects an H3
outcome or chooses a rule from confirmatory results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


NEAR_COUNT = 5_000
MID_COUNT = 5_000
FAR_CANDIDATE_COUNT = 15_000
MID_CANDIDATE_COUNT = 15_000
MID_EXPANDED_CANDIDATE_COUNT = 150_000
VALIDATION_COUNT = 2_000
SCHEMA_VERSION = 1
PURPOSE = "h3_far_loss_quantile_match"
METHOD = "non_confirmatory_pilot_loss_quantile_match"
MID_PURPOSE = "h3_mid_structural_distance_match"
MID_METHOD = "frozen_structural_median_with_pilot_loss_decile_caliper"
MID_D40_PURPOSE = "h3_mid_structural_distance_match_d40"
MID_D40_METHOD = "d40_one_shot_expanded_structural_median_with_unchanged_pilot_caliper"
MID_DISTANCE_QUANTILE = 0.5
ACQUISITION_PURPOSE = "h3_independent_acquisition_banks"
ACQUISITION_METHOD = "model_blind_structural_then_frozen_pilot_loss_decile"
OUTPUT_NAMES = {
    "near": "graph_5_5_bnear_5000.txt",
    "mid": "graph_5_5_bmid_5000.txt",
    "far": "graph_5_5_bfar_5000.txt",
    "near_validation": "graph_5_5_bnearval_2000.txt",
    "mid_validation": "graph_5_5_bmidval_2000.txt",
    "far_validation": "graph_5_5_bfarval_2000.txt",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class GateError(ValueError):
    """The requested materialization would violate the frozen H3 design."""


@dataclass(frozen=True)
class BankItem:
    line: str
    prompt_sha256: str
    graph_key: str
    paired_near_prompt_sha256: str | None = None
    solver_verified: bool | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_source_path(path: Path, project_root: Path) -> str:
    """Use a project-relative identity when an input moves from the host to Colab."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(project_root.resolve()))
    except ValueError:
        return str(resolved)


def _require_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise GateError(f"{field} must be a lowercase SHA-256")
    return value


def verify_sidecar(path: Path) -> str:
    """Verify ``path`` against its required ``<path>.sha256`` sibling."""
    sidecar = Path(f"{path}.sha256")
    if not path.is_file():
        raise GateError(f"input does not exist: {path}")
    if not sidecar.is_file():
        raise GateError(f"immutable input needs a SHA-256 sidecar: {sidecar}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if not fields:
        raise GateError(f"empty SHA-256 sidecar: {sidecar}")
    expected = _require_sha(fields[0], str(sidecar))
    actual = sha256_file(path)
    if actual != expected:
        raise GateError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
    return actual


def _semantic_graph_key(line: str) -> str:
    """Hash graph/query content independently of a manifest's metadata."""
    if "/" not in line or "=" not in line:
        raise GateError("stargraph line must contain both '/' and '=' delimiters")
    graph_and_query = line.split("=", 1)[0]
    body, query = graph_and_query.split("/", 1)
    edges = body.split("|")
    if len(edges) != 20 or any(edge.count(",") != 1 for edge in edges):
        raise GateError("stargraph line must contain exactly 20 serialized edges")
    canonical = f"{'|'.join(sorted(edges))}/{query}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _prompt_sha256(line: str) -> str:
    """Match the generator: hash through and including the answer delimiter."""
    if "=" not in line:
        raise GateError("stargraph line lacks '=' answer delimiter")
    return hashlib.sha256(line[: line.index("=") + 1].encode("utf-8")).hexdigest()


def _edges(line: str) -> frozenset[str]:
    return frozenset(line.split("/", 1)[0].split("|"))


def _path_length(line: str) -> int:
    return len(line.split("=", 1)[1].split(","))


def normalized_edge_disagreement(left: BankItem, right: BankItem) -> float:
    """Model-blind graph distance used to define the frozen middle branch."""
    left_edges, right_edges = _edges(left.line), _edges(right.line)
    if len(left_edges) != 20 or len(right_edges) != 20:
        raise GateError("structural distance requires two 20-edge path-star graphs")
    return 1.0 - len(left_edges & right_edges) / 20.0


def load_manifest(path: Path, expected_count: int, expected_pool: str) -> list[BankItem]:
    verify_sidecar(path)
    items: list[BankItem] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            record = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GateError(f"{path}:{number}: invalid JSON: {exc}") from exc
        if record.get("pool") != expected_pool:
            raise GateError(
                f"{path}:{number}: pool must be {expected_pool!r}, got {record.get('pool')!r}"
            )
        line = record.get("line")
        prompt_sha = record.get("prompt_sha256")
        graph_key = record.get("graph_key")
        if not isinstance(line, str) or not line.strip():
            raise GateError(f"{path}:{number}: missing serialized stargraph line")
        _require_sha(prompt_sha, f"{path}:{number}: prompt_sha256")
        _require_sha(graph_key, f"{path}:{number}: graph_key")
        if _prompt_sha256(line) != prompt_sha:
            raise GateError(f"{path}:{number}: prompt_sha256 does not hash line")
        if _semantic_graph_key(line) != graph_key:
            raise GateError(f"{path}:{number}: graph_key does not match line")
        paired_near = record.get("paired_near_prompt_sha256")
        solver_verified = record.get("solver_verified")
        if expected_pool == "B_mid":
            _require_sha(paired_near, f"{path}:{number}: paired_near_prompt_sha256")
            if solver_verified is not True:
                raise GateError(f"{path}:{number}: B_mid candidate must be solver_verified")
        items.append(BankItem(
            line=line,
            prompt_sha256=prompt_sha,
            graph_key=graph_key,
            paired_near_prompt_sha256=paired_near,
            solver_verified=solver_verified,
        ))
    _require_count_and_uniqueness(items, expected_count, str(path))
    return items


def load_validation(path: Path) -> list[BankItem]:
    """Load a separately generated validation JSONL or raw text bank."""
    verify_sidecar(path)
    items: list[BankItem] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            raise GateError(f"{path}:{number}: blank lines are not allowed")
        if raw.lstrip().startswith("{"):
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise GateError(f"{path}:{number}: invalid JSON: {exc}") from exc
            line = record.get("line")
            if not isinstance(line, str):
                raise GateError(f"{path}:{number}: JSONL validation item lacks 'line'")
            prompt_sha = record.get("prompt_sha256") or _prompt_sha256(line)
            if _prompt_sha256(line) != prompt_sha:
                raise GateError(f"{path}:{number}: prompt_sha256 does not hash line")
            graph_key = record.get("graph_key") or _semantic_graph_key(line)
            if _semantic_graph_key(line) != graph_key:
                raise GateError(f"{path}:{number}: graph_key does not match line")
        else:
            line = raw
            prompt_sha = _prompt_sha256(line)
            graph_key = _semantic_graph_key(line)
        _require_sha(prompt_sha, f"{path}:{number}: prompt_sha256")
        _require_sha(graph_key, f"{path}:{number}: graph_key")
        items.append(BankItem(line=line, prompt_sha256=prompt_sha, graph_key=graph_key))
    _require_count_and_uniqueness(items, VALIDATION_COUNT, str(path))
    return items


def _require_count_and_uniqueness(items: Sequence[BankItem], count: int, label: str) -> None:
    if len(items) != count:
        raise GateError(f"{label}: expected exactly {count} items, got {len(items)}")
    prompts = [item.prompt_sha256 for item in items]
    graphs = [item.graph_key for item in items]
    if len(set(prompts)) != count:
        raise GateError(f"{label}: prompt hashes are not unique")
    if len(set(graphs)) != count:
        raise GateError(f"{label}: graph keys are not unique")


def _read_json_artifact(path: Path, label: str) -> Mapping[str, Any]:
    verify_sidecar(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GateError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise GateError(f"{label} must be a JSON object")
    return value


def verify_acquisition_provenance(
    path: Path, *, near_sha256: str, mid_sha256: str, far_sha256: str
) -> Mapping[str, Any]:
    """Bind independent acquisition sets to an outcome-blind frozen selection receipt."""
    artifact = _read_json_artifact(path, "acquisition provenance")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise GateError(f"acquisition provenance schema_version must be {SCHEMA_VERSION}")
    if (
        artifact.get("purpose") != ACQUISITION_PURPOSE
        or artifact.get("selection_method") != ACQUISITION_METHOD
    ):
        raise GateError("acquisition provenance does not declare the frozen outcome-blind rule")
    required = {
        "frozen_before_confirmatory": True,
        "inspected_confirmatory_checkpoints": False,
        "inspected_confirmatory_results": False,
        "optimized_h3_outcomes": False,
        "disjoint_from_training": True,
        "matched_target_path_distribution": True,
        "matched_pilot_loss_deciles": True,
    }
    for field, expected in required.items():
        if artifact.get(field) != expected:
            raise GateError(f"acquisition provenance {field} must be {expected!r}")
    _require_sha(artifact.get("selector_code_sha256"), "acquisition selector_code_sha256")
    actual = {"near": near_sha256, "mid": mid_sha256, "far": far_sha256}
    if artifact.get("bank_sha256") != actual:
        raise GateError("acquisition provenance does not bind these near/mid/far banks")
    if artifact.get("counts") != {label: VALIDATION_COUNT for label in actual}:
        raise GateError("acquisition provenance does not bind the frozen bank counts")
    return artifact


def select_far(
    near: Sequence[BankItem],
    candidates: Sequence[BankItem],
    artifact_path: Path,
    *,
    near_sha256: str,
    candidates_sha256: str,
) -> list[BankItem]:
    """Validate a pilot mapping and return its already-selected far items in paired order."""
    artifact = _read_json_artifact(artifact_path, "far selection artifact")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise GateError(f"far selection schema_version must be {SCHEMA_VERSION}")
    if artifact.get("purpose") != PURPOSE or artifact.get("selection_method") != METHOD:
        raise GateError("far selection must declare the frozen H3 pilot loss-quantile method")
    if artifact.get("near_bank_sha256") != near_sha256:
        raise GateError("far selection was not made against this B_near manifest")
    if artifact.get("candidate_bank_sha256") != candidates_sha256:
        raise GateError("far selection was not made from this B_far candidate manifest")

    pilot = artifact.get("pilot")
    if not isinstance(pilot, dict):
        raise GateError("far selection artifact lacks pilot provenance")
    required_truths = {
        "role": "non_confirmatory",
        "frozen_before_confirmatory": True,
        "inspected_confirmatory_checkpoints": False,
        "inspected_confirmatory_results": False,
        "optimized_h3_outcomes": False,
    }
    for field, expected in required_truths.items():
        if pilot.get(field) != expected:
            raise GateError(f"pilot.{field} must be {expected!r}")
    for field in ("checkpoint_sha256", "loss_table_sha256", "selector_code_sha256"):
        _require_sha(pilot.get(field), f"pilot.{field}")
    if not isinstance(pilot.get("created_at_utc"), str) or not pilot["created_at_utc"].endswith("Z"):
        raise GateError("pilot.created_at_utc must be an explicit UTC timestamp ending in Z")

    mapping = artifact.get("selection")
    if not isinstance(mapping, list) or len(mapping) != NEAR_COUNT:
        raise GateError(f"selection must contain exactly {NEAR_COUNT} paired records")
    near_by_sha = {item.prompt_sha256: item for item in near}
    far_by_sha = {item.prompt_sha256: item for item in candidates}
    seen_near: set[str] = set()
    seen_far: set[str] = set()
    selected: list[BankItem] = []
    for index, pair in enumerate(mapping):
        if not isinstance(pair, dict):
            raise GateError(f"selection[{index}] must be an object")
        near_sha = _require_sha(pair.get("near_prompt_sha256"), f"selection[{index}].near_prompt_sha256")
        far_sha = _require_sha(pair.get("far_prompt_sha256"), f"selection[{index}].far_prompt_sha256")
        near_q = pair.get("near_loss_quantile")
        far_q = pair.get("far_loss_quantile")
        if not all(isinstance(q, (int, float)) and not isinstance(q, bool) and math.isfinite(q) and 0 <= q <= 1
                   for q in (near_q, far_q)):
            raise GateError(f"selection[{index}] quantiles must be finite numbers in [0, 1]")
        if near_sha not in near_by_sha or far_sha not in far_by_sha:
            raise GateError(f"selection[{index}] references an item outside its frozen source bank")
        if near_sha != near[index].prompt_sha256:
            raise GateError(
                f"selection[{index}] is not in frozen B_near order; paired item order must match"
            )
        if near_sha in seen_near or far_sha in seen_far:
            raise GateError(f"selection[{index}] reuses a near or far item")
        if abs(float(near_q) - float(far_q)) > 1e-12:
            raise GateError(f"selection[{index}] does not map the same loss quantile")
        if _path_length(near[index].line) != _path_length(far_by_sha[far_sha].line):
            raise GateError(f"selection[{index}] does not match target-path length")
        seen_near.add(near_sha)
        seen_far.add(far_sha)
        selected.append(far_by_sha[far_sha])
    if seen_near != set(near_by_sha):
        raise GateError("selection does not pair every B_near item exactly once")
    first_five_thousand = {item.prompt_sha256 for item in candidates[:NEAR_COUNT]}
    if {item.prompt_sha256 for item in selected} == first_five_thousand:
        raise GateError("refusing B_far: selection is the first 5,000 candidates in file order")
    return selected


def select_mid(
    near: Sequence[BankItem],
    candidates: Sequence[BankItem],
    artifact_path: Path,
    *,
    near_sha256: str,
    candidates_sha256: str,
) -> list[BankItem]:
    """Validate the frozen model-blind/pilot-caliper middle-bank mapping."""
    artifact = _read_json_artifact(artifact_path, "mid selection artifact")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise GateError(f"mid selection schema_version must be {SCHEMA_VERSION}")
    contract = (artifact.get("purpose"), artifact.get("selection_method"))
    if contract == (MID_PURPOSE, MID_METHOD):
        expected_candidate_count = MID_CANDIDATE_COUNT
    elif contract == (MID_D40_PURPOSE, MID_D40_METHOD):
        expected_candidate_count = MID_EXPANDED_CANDIDATE_COUNT
        if artifact.get("permanent_block_if_any_unmatched") is not True:
            raise GateError("D40 mid selection must permanently block if any near item is unmatched")
        if artifact.get("no_further_amendments_permitted") is not True:
            raise GateError("D40 mid selection must prohibit further matching amendments")
        combined_loss_sha = _require_sha(
            artifact.get("combined_loss_table_sha256"), "combined_loss_table_sha256"
        )
    else:
        raise GateError("mid selection does not declare the frozen structural/pilot rule")
    if artifact.get("near_bank_sha256") != near_sha256:
        raise GateError("mid selection was not made against this B_near manifest")
    if artifact.get("candidate_bank_sha256") != candidates_sha256:
        raise GateError("mid selection was not made from this B_mid candidate manifest")
    if float(artifact.get("distance_quantile", -1)) != MID_DISTANCE_QUANTILE:
        raise GateError(f"mid distance_quantile must be frozen at {MID_DISTANCE_QUANTILE}")
    loss_caliper = artifact.get("pilot_loss_caliper")
    if (
        not isinstance(loss_caliper, (int, float)) or isinstance(loss_caliper, bool)
        or not math.isfinite(float(loss_caliper)) or float(loss_caliper) < 0
    ):
        raise GateError("mid pilot_loss_caliper must be a frozen finite nonnegative number")
    if artifact.get("tie_break") != "candidate_prompt_sha256_ascending":
        raise GateError("mid selection tie-break must be candidate SHA-256 ascending")

    pilot = artifact.get("pilot")
    required_truths = {
        "role": "non_confirmatory",
        "frozen_before_confirmatory": True,
        "inspected_confirmatory_checkpoints": False,
        "inspected_confirmatory_results": False,
        "optimized_h3_outcomes": False,
    }
    if not isinstance(pilot, dict):
        raise GateError("mid selection artifact lacks pilot provenance")
    for field, expected in required_truths.items():
        if pilot.get(field) != expected:
            raise GateError(f"pilot.{field} must be {expected!r}")
    for field in ("checkpoint_sha256", "loss_table_sha256", "selector_code_sha256"):
        _require_sha(pilot.get(field), f"pilot.{field}")
    if contract == (MID_D40_PURPOSE, MID_D40_METHOD) and pilot.get("loss_table_sha256") != combined_loss_sha:
        raise GateError("D40 pilot provenance does not bind the combined loss table")
    if not isinstance(pilot.get("created_at_utc"), str) or not pilot["created_at_utc"].endswith("Z"):
        raise GateError("pilot.created_at_utc must be an explicit UTC timestamp ending in Z")

    near_by_sha = {item.prompt_sha256: item for item in near}
    candidate_by_sha = {item.prompt_sha256: item for item in candidates}
    table = artifact.get("candidate_table")
    if not isinstance(table, list) or len(table) != expected_candidate_count:
        raise GateError(
            f"mid candidate_table must contain all {expected_candidate_count} candidates"
        )
    eligible_by_near: dict[str, list[tuple[float, str, BankItem]]] = {
        sha: [] for sha in near_by_sha
    }
    seen_table: set[str] = set()
    eligible_distances: list[float] = []
    for index, row in enumerate(table):
        if not isinstance(row, dict):
            raise GateError(f"candidate_table[{index}] must be an object")
        mid_sha = _require_sha(row.get("mid_prompt_sha256"), f"candidate_table[{index}].mid_prompt_sha256")
        if mid_sha not in candidate_by_sha or mid_sha in seen_table:
            raise GateError(f"candidate_table[{index}] references or reuses an invalid candidate")
        candidate = candidate_by_sha[mid_sha]
        near_sha = candidate.paired_near_prompt_sha256
        if near_sha not in near_by_sha:
            raise GateError(f"candidate_table[{index}] is not paired to this B_near bank")
        near_decile, mid_decile = row.get("near_loss_decile"), row.get("mid_loss_decile")
        loss_difference = row.get("pilot_loss_absolute_difference")
        if (
            not isinstance(near_decile, int) or isinstance(near_decile, bool)
            or not isinstance(mid_decile, int) or isinstance(mid_decile, bool)
            or near_decile not in range(10) or mid_decile not in range(10)
            or not isinstance(loss_difference, (int, float)) or isinstance(loss_difference, bool)
            or not math.isfinite(float(loss_difference)) or float(loss_difference) < 0
        ):
            raise GateError(f"candidate_table[{index}] has invalid frozen pilot-loss values")
        distance = normalized_edge_disagreement(near_by_sha[near_sha], candidate)
        declared = row.get("normalized_edge_disagreement")
        if not isinstance(declared, (int, float)) or abs(float(declared) - distance) > 1e-12:
            raise GateError(f"candidate_table[{index}] structural distance is incorrect")
        path_match = _path_length(near_by_sha[near_sha].line) == _path_length(candidate.line)
        should_be_eligible = (
            near_decile == mid_decile
            and float(loss_difference) <= float(loss_caliper) + 1e-12
            and path_match
            and candidate.solver_verified is True
        )
        if row.get("eligible") is not should_be_eligible:
            raise GateError(f"candidate_table[{index}] eligibility does not follow frozen rules")
        if should_be_eligible:
            eligible_by_near[near_sha].append((distance, mid_sha, candidate))
            eligible_distances.append(distance)
        seen_table.add(mid_sha)
    if seen_table != set(candidate_by_sha):
        raise GateError("mid candidate_table does not cover the candidate manifest exactly")
    if not eligible_distances or any(not values for values in eligible_by_near.values()):
        raise GateError("mid pilot caliper leaves one or more B_near items without a candidate")
    ordered_distances = sorted(eligible_distances)
    middle = len(ordered_distances) // 2
    median_distance = (
        ordered_distances[middle] if len(ordered_distances) % 2
        else (ordered_distances[middle - 1] + ordered_distances[middle]) / 2.0
    )
    declared_median = artifact.get("eligible_median_normalized_edge_disagreement")
    if not isinstance(declared_median, (int, float)) or abs(float(declared_median) - median_distance) > 1e-12:
        raise GateError("mid selection does not bind the eligible candidate median")

    mapping = artifact.get("selection")
    if not isinstance(mapping, list) or len(mapping) != MID_COUNT:
        raise GateError(f"mid selection must contain exactly {MID_COUNT} paired records")
    seen: set[str] = set()
    selected: list[BankItem] = []
    for index, pair in enumerate(mapping):
        if not isinstance(pair, dict):
            raise GateError(f"selection[{index}] must be an object")
        near_sha = _require_sha(pair.get("near_prompt_sha256"), f"selection[{index}].near_prompt_sha256")
        mid_sha = _require_sha(pair.get("mid_prompt_sha256"), f"selection[{index}].mid_prompt_sha256")
        if near_sha != near[index].prompt_sha256:
            raise GateError(f"selection[{index}] is not in frozen B_near order")
        if mid_sha not in candidate_by_sha or mid_sha in seen:
            raise GateError(f"selection[{index}] references or reuses an invalid B_mid candidate")
        mid = candidate_by_sha[mid_sha]
        distance = normalized_edge_disagreement(near[index], mid)
        declared = pair.get("normalized_edge_disagreement")
        if not isinstance(declared, (int, float)) or isinstance(declared, bool) or abs(float(declared) - distance) > 1e-12:
            raise GateError(f"selection[{index}] structural distance is absent or incorrect")
        expected = min(
            eligible_by_near[near_sha], key=lambda value: (abs(value[0] - median_distance), value[1])
        )
        if mid_sha != expected[1]:
            raise GateError(
                f"selection[{index}] is not the SHA-tiebroken candidate closest to median"
            )
        seen.add(mid_sha)
        selected.append(mid)
    if {item.prompt_sha256 for item in selected} == {
        item.prompt_sha256 for item in candidates[:MID_COUNT]
    }:
        raise GateError("refusing B_mid: selection is the first 5,000 candidates in file order")
    return selected


def mid_candidate_count_from_selection(path: Path) -> int:
    """Resolve the prospectively frozen D39/D40 population before loading a large manifest."""
    artifact = _read_json_artifact(path, "mid selection artifact")
    contract = (artifact.get("purpose"), artifact.get("selection_method"))
    if contract == (MID_PURPOSE, MID_METHOD):
        return MID_CANDIDATE_COUNT
    if contract == (MID_D40_PURPOSE, MID_D40_METHOD):
        if (
            artifact.get("permanent_block_if_any_unmatched") is not True
            or artifact.get("no_further_amendments_permitted") is not True
        ):
            raise GateError("D40 mid selection lacks its permanent one-shot stopping rule")
        _require_sha(artifact.get("combined_loss_table_sha256"), "combined_loss_table_sha256")
        return MID_EXPANDED_CANDIDATE_COUNT
    raise GateError("mid selection does not declare a supported frozen structural/pilot rule")


def require_disjoint(banks: Mapping[str, Sequence[BankItem]]) -> None:
    labels = list(banks)
    for left_index, left in enumerate(labels):
        for right in labels[left_index + 1 :]:
            left_prompts = {item.prompt_sha256 for item in banks[left]}
            right_prompts = {item.prompt_sha256 for item in banks[right]}
            prompt_overlap = left_prompts & right_prompts
            graph_overlap = {item.graph_key for item in banks[left]} & {
                item.graph_key for item in banks[right]
            }
            if prompt_overlap or graph_overlap:
                raise GateError(
                    f"{left} and {right} are not independent: {len(prompt_overlap)} prompt "
                    f"and {len(graph_overlap)} graph-key collisions"
                )


def require_matched_path_distributions(banks: Mapping[str, Sequence[BankItem]]) -> None:
    """Require identical target-path-length histograms across intervention banks."""
    reference_label: str | None = None
    reference: dict[int, int] | None = None
    for label, items in banks.items():
        histogram: dict[int, int] = {}
        for item in items:
            length = _path_length(item.line)
            histogram[length] = histogram.get(length, 0) + 1
        if reference is None:
            reference_label, reference = label, histogram
        elif histogram != reference:
            raise GateError(
                f"{label} target-path distribution {histogram} does not match "
                f"{reference_label} {reference}"
            )


def _atomic_write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = Path(f"{path}.sha256")
    sidecar_payload = f"{digest}  {path.name}\n".encode()
    fd, temporary = tempfile.mkstemp(prefix=f".{sidecar.name}.", dir=sidecar.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(sidecar_payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, sidecar)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return digest


def refresh_manifest_inventory(project_root: Path) -> str:
    """Atomically bind every staged manifest and HMM array for runtime verification."""
    project_root = project_root.resolve()
    inventory = project_root / "manifests" / "manifest_inventory.sha256"
    excluded = {inventory.resolve(), Path(f"{inventory}.sha256").resolve()}
    paths: list[Path] = []
    for base in (
        project_root / "manifests",
        project_root / "data" / "hmm",
        project_root / "data" / "hmm_family",
    ):
        if base.is_dir():
            paths.extend(
                path for path in base.rglob("*")
                if path.is_file() and not path.name.endswith(".partial")
                and path.resolve() not in excluded
            )
    paths.sort(key=lambda item: str(item.resolve().relative_to(project_root)))
    payload = ("\n".join(
        f"{sha256_file(path)}  {path.resolve().relative_to(project_root)}" for path in paths
    ) + "\n").encode("utf-8")
    return _atomic_write(inventory, payload)


def write_banks(output_dir: Path, banks: Mapping[str, Sequence[BankItem]], sources: Mapping[str, Any]) -> dict[str, Any]:
    output_hashes: dict[str, str] = {}
    for label, items in banks.items():
        name = OUTPUT_NAMES[label]
        payload = ("\n".join(item.line for item in items) + "\n").encode("utf-8")
        output_hashes[name] = _atomic_write(output_dir / name, payload)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "materialized",
        "scientific_selection_performed": False,
        "sources": dict(sources),
        "outputs": output_hashes,
        "checks": [
            "exact_counts",
            "unique_prompt_hashes",
            "unique_graph_keys",
            "pairwise_train_validation_disjointness",
            "matched_target_path_distributions",
            "required_input_sidecars",
        ],
    }
    receipt_payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    _atomic_write(output_dir / "adaptation_banks.json", receipt_payload)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--near-manifest", type=Path, default=root / "manifests/b_near.jsonl")
    parser.add_argument("--far-candidates", type=Path, default=root / "manifests/b_far.jsonl")
    parser.add_argument("--far-selection", type=Path)
    parser.add_argument("--mid-candidates", type=Path)
    parser.add_argument("--mid-selection", type=Path)
    parser.add_argument("--near-validation", type=Path)
    parser.add_argument("--mid-validation", type=Path)
    parser.add_argument("--far-validation", type=Path)
    parser.add_argument("--acquisition-provenance", type=Path)
    parser.add_argument("--output-dir", type=Path, default=root / "manifests/adapt")
    parser.add_argument(
        "--near-only",
        action="store_true",
        help="materialize only frozen B_near; no far or validation claim is made",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    near_sha = verify_sidecar(args.near_manifest)
    near = load_manifest(args.near_manifest, NEAR_COUNT, "B_near")
    banks: dict[str, Sequence[BankItem]] = {"near": near}
    sources: dict[str, Any] = {
        "near_manifest": {
            "path": portable_source_path(args.near_manifest, project_root),
            "sha256": near_sha,
        }
    }
    if not args.near_only:
        missing = [
            flag
            for flag, value in (
                ("--far-selection", args.far_selection),
                ("--mid-candidates", args.mid_candidates),
                ("--mid-selection", args.mid_selection),
                ("--near-validation", args.near_validation),
                ("--mid-validation", args.mid_validation),
                ("--far-validation", args.far_validation),
                ("--acquisition-provenance", args.acquisition_provenance),
            )
            if value is None
        ]
        if missing:
            raise GateError(
                "full materialization is gated; missing independent frozen inputs: " + ", ".join(missing)
            )
        far_sha = verify_sidecar(args.far_candidates)
        candidates = load_manifest(args.far_candidates, FAR_CANDIDATE_COUNT, "B_far")
        mid_sha = verify_sidecar(args.mid_candidates)
        mid_candidate_count = mid_candidate_count_from_selection(args.mid_selection)
        mid_candidates = load_manifest(args.mid_candidates, mid_candidate_count, "B_mid")
        mid = select_mid(
            near,
            mid_candidates,
            args.mid_selection,
            near_sha256=near_sha,
            candidates_sha256=mid_sha,
        )
        selected = select_far(
            near,
            candidates,
            args.far_selection,
            near_sha256=near_sha,
            candidates_sha256=far_sha,
        )
        near_validation_sha = verify_sidecar(args.near_validation)
        mid_validation_sha = verify_sidecar(args.mid_validation)
        far_validation_sha = verify_sidecar(args.far_validation)
        near_validation = load_validation(args.near_validation)
        mid_validation = load_validation(args.mid_validation)
        far_validation = load_validation(args.far_validation)
        verify_acquisition_provenance(
            args.acquisition_provenance,
            near_sha256=near_validation_sha,
            mid_sha256=mid_validation_sha,
            far_sha256=far_validation_sha,
        )
        banks.update(
            mid=mid,
            far=selected,
            near_validation=near_validation,
            mid_validation=mid_validation,
            far_validation=far_validation,
        )
        sources.update(
            far_candidates={"path": portable_source_path(args.far_candidates, project_root),
                            "sha256": far_sha},
            far_selection={"path": portable_source_path(args.far_selection, project_root),
                           "sha256": verify_sidecar(args.far_selection)},
            mid_candidates={"path": portable_source_path(args.mid_candidates, project_root),
                            "sha256": mid_sha},
            mid_selection={"path": portable_source_path(args.mid_selection, project_root),
                           "sha256": verify_sidecar(args.mid_selection)},
            near_validation={"path": portable_source_path(args.near_validation, project_root),
                             "sha256": near_validation_sha},
            mid_validation={"path": portable_source_path(args.mid_validation, project_root),
                            "sha256": mid_validation_sha},
            far_validation={"path": portable_source_path(args.far_validation, project_root),
                            "sha256": far_validation_sha},
            acquisition_provenance={
                "path": portable_source_path(args.acquisition_provenance, project_root),
                "sha256": verify_sidecar(args.acquisition_provenance),
            },
        )
    require_matched_path_distributions({
        label: items for label, items in banks.items() if "validation" not in label
    })
    validation_banks = {
        label: items for label, items in banks.items() if "validation" in label
    }
    if validation_banks:
        require_matched_path_distributions(validation_banks)
    require_disjoint(banks)
    receipt = write_banks(args.output_dir, banks, sources)
    try:
        args.output_dir.resolve().relative_to(project_root.resolve())
    except ValueError:
        pass
    else:
        refresh_manifest_inventory(project_root)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateError as exc:
        raise SystemExit(f"REFUSED: {exc}") from exc
