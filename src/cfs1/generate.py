"""Fresh, model-blind construction of CFS-1 causal-forgetting banks.

Each update unit is a solver-certified G(5,5) Path-Star quartet tied to one
untouched retention probe.  The four cells are constructed, not selected:

``high/same`` and ``high/different`` preserve 18 of 20 edges; ``low/same``
and ``low/different`` preserve the minimum compatible 8 and 7 edges,
respectively.  All four retain the same source, goal, node-token multiset,
prompt length, answer length, probe assignment, and deterministic stream slot.
No model, loss, learned representation, caliper, or pilot data is imported.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from lurestar.validate import ARM_LEN, MAX_NODES, NUM_ARMS

from . import validate as V


MASTER_SEED = 20260824
N_PROBES = 2_000
N_UPDATES = 5_000
MAX_CANDIDATE_ATTEMPTS = 10_000


@dataclass(frozen=True)
class Graph:
    source: int
    goal: int
    arms: tuple[tuple[int, ...], ...]
    order: tuple[int, ...]

    def edge(self, structural_index: int) -> tuple[int, int]:
        arm, depth = divmod(structural_index, ARM_LEN)
        return (self.source if depth == 0 else self.arms[arm][depth - 1], self.arms[arm][depth])

    def serialize(self) -> str:
        edges = "|".join(f"{u},{v}" for u, v in (self.edge(index) for index in self.order))
        paths = [arm for arm in self.arms if arm[-1] == self.goal]
        if len(paths) != 1:
            raise ValueError("CFS-1 graph has no unique goal arm")
        answer = ",".join(str(node) for node in (self.source,) + paths[0])
        return f"{edges}/{self.source},{self.goal}={answer}"


@dataclass(frozen=True)
class CFS1Bundle:
    """Named construction result; avoids unsafe positional API drift."""

    retention: list[dict[str, Any]]
    updates: dict[tuple[str, str], list[dict[str, Any]]]
    global_controls: list[dict[str, Any]]
    codebook: dict[str, Any]


def _hash_int(*parts: object) -> int:
    data = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest(), "big")


def _ranked(values: Sequence[int], *parts: object) -> list[int]:
    return sorted(values, key=lambda value: (_hash_int(MASTER_SEED, *parts, value), value))


def _permutation(values: Sequence[int], *parts: object) -> list[int]:
    return _ranked(list(values), *parts)


def _base_graph(probe_index: int, attempt: int, *, domain: str = "retention") -> Graph:
    """Derive a unique node set and serialized order solely from public seed/index."""
    nodes = _permutation(list(range(MAX_NODES)), domain, probe_index, "attempt", attempt)[: NUM_ARMS * ARM_LEN + 1]
    source = nodes[0]
    arms = tuple(tuple(nodes[1 + arm * ARM_LEN: 1 + (arm + 1) * ARM_LEN]) for arm in range(NUM_ARMS))
    goal = arms[0][-1]
    order = tuple(_permutation(list(range(NUM_ARMS * ARM_LEN)), domain, "edge-order", probe_index, attempt))
    return Graph(source=source, goal=goal, arms=arms, order=order)


def _graph_with_arms(base: Graph, arms: Sequence[Sequence[int]]) -> Graph:
    return Graph(base.source, base.goal, tuple(tuple(arm) for arm in arms), base.order)


def _suffix_swap(base: Graph, left: int, right: int, depth: int) -> Graph:
    """Two-edge suffix swap; ``depth`` uses zero-based arm-node positions."""
    if not (0 < depth < ARM_LEN):
        raise ValueError("CFS-1 swaps require an internal arm depth")
    old_left, old_right = list(base.arms[left]), list(base.arms[right])
    arms = [list(arm) for arm in base.arms]
    arms[left] = old_left[:depth] + old_right[depth:]
    arms[right] = old_right[:depth] + old_left[depth:]
    return _graph_with_arms(base, arms)


def _derangement(values: Sequence[int], forbidden: Sequence[int], *, tag: object) -> list[int]:
    """A deterministic permutation with no row-wise equality to ``forbidden``."""
    ordered = _permutation(values, "derangement", tag)
    for shift in range(len(ordered)):
        candidate = ordered[shift:] + ordered[:shift]
        if all(left != right for left, right in zip(candidate, forbidden)):
            return candidate
    raise AssertionError("no derangement exists for four or more values")


def _low_same(base: Graph, variant: int) -> Graph:
    """Retain only the forced source fan-out and unchanged goal path (8 edges)."""
    groups = list(range(NUM_ARMS))
    positions: list[list[int]] = [groups]
    previous = groups
    for depth in range(1, ARM_LEN):
        rest = _derangement(groups[1:], previous[1:], tag=("low-same", variant, depth))
        current = [0] + rest
        positions.append(current)
        previous = current
    arms = [[base.arms[positions[depth][arm]][depth] for depth in range(ARM_LEN)] for arm in groups]
    return _graph_with_arms(base, arms)


def _low_different(base: Graph, partner: int, variant: int) -> Graph:
    """Match the high-different answer while retaining only its forced 7 edges."""
    groups = list(range(NUM_ARMS))
    # High-different at depth 2 has target path partner[0], partner[1], 0[2], 0[3].
    p0 = [partner] + [group for group in groups if group != partner]
    rest1 = _derangement([group for group in groups if group != partner], p0[1:], tag=("low-different", variant, 1))
    p1 = [partner] + rest1
    rest2 = _derangement(groups[1:], p1[1:], tag=("low-different", variant, 2))
    p2 = [0] + rest2
    rest3 = _derangement(groups[1:], p2[1:], tag=("low-different", variant, 3))
    p3 = [0] + rest3
    positions = [p0, p1, p2, p3]
    arms = [[base.arms[positions[depth][arm]][depth] for depth in range(ARM_LEN)] for arm in groups]
    return _graph_with_arms(base, arms)


def _variants(occurrence: int) -> tuple[tuple[int, int], int]:
    """Three distinct, fixed construction codes suffice for 2/3 probe reuse."""
    safe_pairs = ((1, 2), (1, 3), (2, 3))
    if occurrence not in range(len(safe_pairs)):
        raise ValueError("CFS-1 codebook permits at most three copies per probe")
    return safe_pairs[occurrence], occurrence + 1


def _unit_graphs(base: Graph, occurrence: int) -> Mapping[tuple[str, str], Graph]:
    safe_pair, partner = _variants(occurrence)
    high_same = _suffix_swap(base, safe_pair[0], safe_pair[1], 2)
    high_different = _suffix_swap(base, 0, partner, 2)
    low_same = _low_same(base, occurrence)
    low_different = _low_different(base, partner, occurrence)
    return {
        ("high", "same"): high_same,
        ("high", "different"): high_different,
        ("low", "same"): low_same,
        ("low", "different"): low_different,
    }


def make_codebook(n_probes: int = N_PROBES, n_updates: int = N_UPDATES) -> dict[str, Any]:
    if not n_probes > 0 or not 2 * n_probes <= n_updates <= 3 * n_probes:
        raise ValueError("CFS-1 requires every probe to appear exactly two or three times")
    extra = n_updates - 2 * n_probes
    triple_probes = set(_ranked(list(range(n_probes)), "triple-probe")[:extra])
    assignments = [(probe, occurrence) for probe in range(n_probes) for occurrence in range(2 + (probe in triple_probes))]
    assignments = sorted(assignments, key=lambda pair: (_hash_int(MASTER_SEED, "unit-order", *pair), pair))
    unit_order = [f"cfs1-unit-{slot:05d}" for slot in range(n_updates)]
    unit_map = [
        {"unit_id": unit_order[slot], "stream_position": slot, "probe_index": probe, "occurrence": occurrence}
        for slot, (probe, occurrence) in enumerate(assignments)
    ]
    episodes = []
    for episode in (0, 1):
        order = sorted(unit_order, key=lambda unit: (_hash_int(MASTER_SEED, "episode", episode, unit), unit))
        episodes.append({"episode": episode, "unit_order": order, "unit_order_sha256": V.sha256_bytes(V.canonical_json(order))})
    return {
        "schema": "nextlat_forgetting/cfs1_hash_codebook/1",
        "master_seed": MASTER_SEED,
        "n_probes": n_probes,
        "n_updates": n_updates,
        "triple_probe_indices": sorted(triple_probes),
        "assignment_rule": "sha256(master_seed|unit-order|probe_index|occurrence), ascending",
        "unit_order": unit_order,
        "units": unit_map,
        "episodes": episodes,
    }


def _retention_record(probe_id: str, graph: Graph, candidate_attempt: int) -> dict[str, Any]:
    line = graph.serialize()
    return {
        "schema": V.RETENTION_SCHEMA,
        "probe_id": probe_id,
        "candidate_attempt": candidate_attempt,
        "line": line,
        **{key: value for key, value in V.line_witness(line).items() if key in {"prompt_sha256", "graph_key", "answer_sha256"}},
    }


def _global_control_record(control_id: str, graph: Graph, candidate_attempt: int) -> dict[str, Any]:
    line = graph.serialize()
    return {
        "schema": V.GLOBAL_CONTROL_SCHEMA,
        "control_id": control_id,
        "candidate_attempt": candidate_attempt,
        "line": line,
        **{key: value for key, value in V.line_witness(line).items() if key in {"prompt_sha256", "graph_key", "answer_sha256"}},
    }


def _update_record(
    *, unit: Mapping[str, Any], probe_id: str, graph: Graph, base_line: str,
    overlap: str, future_relation: str,
) -> dict[str, Any]:
    line = graph.serialize()
    witness = V.line_witness(line)
    return {
        "schema": V.UPDATE_SCHEMA,
        "update_id": f"cfs1-update-{overlap}-{future_relation}-{unit['stream_position']:05d}",
        "unit_id": unit["unit_id"],
        "stream_position": unit["stream_position"],
        "probe_id": probe_id,
        "probe_occurrence": unit["occurrence"],
        "condition": {"overlap": overlap, "future_relation": future_relation},
        "line": line,
        "prompt_sha256": witness["prompt_sha256"],
        "graph_key": witness["graph_key"],
        "answer_sha256": witness["answer_sha256"],
        "edge_overlap_with_probe": V.edge_overlap(base_line, line),
        "future_same_as_probe": witness["answer"] == V.line_witness(base_line)["answer"],
        "counterbalance_code": f"occurrence-{unit['occurrence']}",
    }


def build_bundle(
    *, n_probes: int = N_PROBES, n_updates: int = N_UPDATES, legacy: V.LegacyIndex | None = None
) -> CFS1Bundle:
    """Create all CFS-1 records in memory, failing before any write on contamination."""
    codebook = make_codebook(n_probes, n_updates)
    assignments_by_probe: dict[int, list[Mapping[str, Any]]] = {}
    for unit in codebook["units"]:
        assignments_by_probe.setdefault(unit["probe_index"], []).append(unit)
    retention: list[dict[str, Any]] = []
    updates: dict[tuple[str, str], list[dict[str, Any]]] = {condition: [] for condition in V.CONDITIONS}
    local_prompts: set[str] = set()
    local_graphs: set[str] = set()
    for probe_index in range(n_probes):
        probe_id = f"cfs1-retention-{probe_index:05d}"
        if legacy and probe_id in legacy.identifiers:
            raise V.CFS1ValidationError(f"fresh CFS-1 probe id collides with legacy identifier {probe_id}")
        for attempt in range(MAX_CANDIDATE_ATTEMPTS):
            base = _base_graph(probe_index, attempt, domain="retention")
            candidate_retention = _retention_record(probe_id, base, attempt)
            candidate_updates: dict[tuple[str, str], list[dict[str, Any]]] = {condition: [] for condition in V.CONDITIONS}
            candidate_witnesses = [(candidate_retention["prompt_sha256"], candidate_retention["graph_key"])]
            for unit in assignments_by_probe[probe_index]:
                for condition, graph in _unit_graphs(base, int(unit["occurrence"])).items():
                    row = _update_record(unit=unit, probe_id=probe_id, graph=graph, base_line=candidate_retention["line"], overlap=condition[0], future_relation=condition[1])
                    candidate_updates[condition].append(row)
                    candidate_witnesses.append((row["prompt_sha256"], row["graph_key"]))
            prompt_values = [value[0] for value in candidate_witnesses]
            graph_values = [value[1] for value in candidate_witnesses]
            contaminated = (
                len(set(prompt_values)) != len(prompt_values)
                or len(set(graph_values)) != len(graph_values)
                or bool(set(prompt_values) & local_prompts)
                or bool(set(graph_values) & local_graphs)
                or bool(legacy and (set(prompt_values) & legacy.prompt_hashes or set(graph_values) & legacy.graph_keys))
            )
            if contaminated:
                continue
            retention.append(candidate_retention)
            local_prompts.update(prompt_values)
            local_graphs.update(graph_values)
            for condition, rows in candidate_updates.items():
                updates[condition].extend(rows)
            break
        else:
            raise V.CFS1ValidationError(f"unable to find uncontaminated CFS-1 probe {probe_index}")
    global_controls: list[dict[str, Any]] = []
    for control_index in range(n_probes):
        control_id = f"cfs1-global-control-{control_index:05d}"
        if legacy and control_id in legacy.identifiers:
            raise V.CFS1ValidationError(f"fresh CFS-1 global-control id collides with legacy identifier {control_id}")
        for attempt in range(MAX_CANDIDATE_ATTEMPTS):
            row = _global_control_record(control_id, _base_graph(control_index, attempt, domain="global-control"), attempt)
            if row["prompt_sha256"] in local_prompts or row["graph_key"] in local_graphs:
                continue
            if legacy and (row["prompt_sha256"] in legacy.prompt_hashes or row["graph_key"] in legacy.graph_keys):
                continue
            global_controls.append(row)
            local_prompts.add(row["prompt_sha256"])
            local_graphs.add(row["graph_key"])
            break
        else:
            raise V.CFS1ValidationError(f"unable to find uncontaminated global control {control_index}")
    V.validate_bundle(retention, updates, codebook, expected_probes=n_probes, expected_updates=n_updates, global_controls=global_controls, expected_global_controls=n_probes, legacy=legacy)
    return CFS1Bundle(retention=retention, updates=updates, global_controls=global_controls, codebook=codebook)
