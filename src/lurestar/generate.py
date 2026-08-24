"""Lure-Star matched-quartet generator (spec §5).

Every stimulus is a G(5,5) Path-Star graph serialized in upstream's exact format
(``upstream/NextLat/data/stargraph/prepare.py:70-73``).  A quartet holds one base graph
and its controlled perturbations:

===================  ===========================================================
condition            construction
===================  ===========================================================
``base``             the graph as sampled, under edge order ``pi_base``
``repeat``           the same graph under a different edge order ``pi_repeat``
``near_safe``        depth-k suffix swap between two distractor arms, on ``base``
``near_critical``    depth-k suffix swap between the goal arm and a distractor
                     arm, on ``base``
``near_safe_aligned``the same safe swap applied to ``repeat`` — see below
``far_critical``     the same node multiset repartitioned into a low-overlap graph
===================  ===========================================================

A depth-k suffix swap between arms ``A = [s,a1..a4]`` and ``B = [s,b1..b4]`` rewrites
``(a_{k-1} -> a_k)`` and ``(b_{k-1} -> b_k)`` into ``(a_{k-1} -> b_k)`` and
``(b_{k-1} -> a_k)``.  The tails are untouched, so **exactly two serialized head tokens
change**, and the node multiset, the node frequencies and the per-node degree pair are
preserved exactly.  When the goal arm is one of the two, the goal *node* ends up
terminating the other arm: the goal token in the ``/source,goal`` field is unchanged
while the correct first branch and the whole path change.

Why ``near_safe_aligned`` exists, in one paragraph — the full argument, including the
impossibility proof for the spec's literal wording, is in ``docs/STIMULUS_DESIGN.md``.
The critical swap must edit the goal arm's depth-k edge; the safe swap must not.  Those
are different edges, so under one edge ordering they occupy different serialized slots
and no choice of ordering can make the two edits land on the same absolute token
positions.  We therefore enforce two invariants instead of one impossible one:

* **LS-1** (primary, shared anchor): ``near_safe`` and ``near_critical`` are each an
  exact two-token edit of the *same* ``base`` string, their edited slot pairs are
  disjoint and have **identical gap** ``q - p``, and which of the two candidate slot
  pairs goes to which condition is decided by a fair coin — so edit position is
  exchangeable between the two conditions by construction.
* **LS-2** (robustness, matched anchor): ``near_safe_aligned`` is an exact two-token
  edit of ``repeat`` at the **same absolute token positions** as ``near_critical``'s
  edit of ``base``.  Both distances are then "how far does a two-token head swap at
  slots (p,q) move the state", differing only in whether the future changes.

Determinism is a pure function of ``(master_seed, item_index)`` via
``numpy.random.default_rng([master_seed, index, ...])``, so nothing depends on the
worker count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import pathlib
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .validate import (
    ARM_LEN,
    MAX_NODES,
    NUM_ARMS,
    Edge,
    GraphError,
    TrainingIndex,
    canonical_graph_key,
    canonical_key_from_line,
    check_quartet,
    edge_slot_token_positions,
    parse_line,
    sha256_text,
    solve_edges,
)

N_EDGES = NUM_ARMS * ARM_LEN
#: A depth-0 swap exchanges whole arms and leaves the edge *set* untouched, so the
#: usable suffix depths are the arm positions 1..ARM_LEN-1 (edge depths 2..4).
SWAP_DEPTHS = tuple(range(1, ARM_LEN))


# --------------------------------------------------------------------------------------
# Graph representation
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Graph:
    """A Path-Star graph *plus* the serialized order of its edges.

    ``arms[a][d]`` is the node at position ``d`` of arm ``a``.  The structural edge index
    of that node's incoming edge is ``a * ARM_LEN + d``; ``order[t]`` is the structural
    index of the edge printed in serialized slot ``t``.  The goal arm is whichever arm
    ends at ``goal`` — it is *not* assumed to be arm 0, because a critical swap moves the
    goal onto another arm.
    """

    source: int
    goal: int
    arms: Tuple[Tuple[int, ...], ...]
    order: Tuple[int, ...]

    def edge(self, structural: int) -> Edge:
        a, d = divmod(structural, ARM_LEN)
        tail = self.source if d == 0 else self.arms[a][d - 1]
        return (tail, self.arms[a][d])

    def edges_in_order(self) -> Tuple[Edge, ...]:
        return tuple(self.edge(e) for e in self.order)

    def slot_of(self) -> List[int]:
        slots = [0] * len(self.order)
        for t, e in enumerate(self.order):
            slots[e] = t
        return slots

    def goal_arm(self) -> int:
        hits = [a for a, arm in enumerate(self.arms) if arm[-1] == self.goal]
        if len(hits) != 1:
            raise GraphError(f"goal {self.goal} terminates {len(hits)} arms")
        return hits[0]

    def path(self) -> Tuple[int, ...]:
        return (self.source,) + self.arms[self.goal_arm()]

    def serialize(self) -> str:
        body = "|".join(f"{u},{v}" for u, v in self.edges_in_order())
        answer = ",".join(str(n) for n in self.path())
        return f"{body}/{self.source},{self.goal}={answer}"

    def prompt(self) -> str:
        line = self.serialize()
        return line[: line.index("=") + 1]


def graph_from_line(line: str) -> Graph:
    """Recover a :class:`Graph`, *including its serialized edge order*, from a line.

    ``serialize(graph_from_line(x)) == x`` for every well-formed line; the acceptance
    tests assert that round trip against real upstream corpus lines, which is what makes
    this representation trustworthy for the A_pair pool.
    """
    p = parse_line(line)
    solved = solve_edges(p.edges, p.source, p.goal)
    structural: Dict[Edge, int] = {}
    for a, arm in enumerate(solved.arms):
        for d, node in enumerate(arm):
            tail = p.source if d == 0 else arm[d - 1]
            structural[(tail, node)] = a * ARM_LEN + d
    order = tuple(structural[e] for e in p.edges)
    return Graph(source=p.source, goal=p.goal, arms=solved.arms, order=order)


def suffix_swap(graph: Graph, arm_x: int, arm_y: int, depth: int) -> Graph:
    """Swap the depth-``depth`` suffixes of two arms, keeping every other edge in place.

    The two rewritten edges keep their own serialized slots (their tails did not change),
    and each deeper edge of arm ``x`` inherits the slot of the corresponding edge of arm
    ``y`` — because after the swap it *is* that edge.  Result: exactly two head tokens
    move in the serialized string.
    """
    if arm_x == arm_y:
        raise ValueError("suffix swap needs two distinct arms")
    if depth not in SWAP_DEPTHS:
        raise ValueError(f"depth {depth} outside {SWAP_DEPTHS}")

    arms = [list(a) for a in graph.arms]
    ax, ay = list(arms[arm_x]), list(arms[arm_y])
    arms[arm_x] = ax[:depth] + ay[depth:]
    arms[arm_y] = ay[:depth] + ax[depth:]

    slot_of = graph.slot_of()
    new_slot = list(slot_of)
    for d in range(depth + 1, ARM_LEN):
        new_slot[arm_x * ARM_LEN + d] = slot_of[arm_y * ARM_LEN + d]
        new_slot[arm_y * ARM_LEN + d] = slot_of[arm_x * ARM_LEN + d]

    order = [0] * N_EDGES
    for e, t in enumerate(new_slot):
        order[t] = e
    return Graph(
        source=graph.source,
        goal=graph.goal,
        arms=tuple(tuple(a) for a in arms),
        order=tuple(order),
    )


def swap_edit_slots(graph: Graph, arm_x: int, arm_y: int, depth: int) -> Tuple[int, int]:
    """Serialized slots whose head token a ``suffix_swap`` would rewrite."""
    slot_of = graph.slot_of()
    return (slot_of[arm_x * ARM_LEN + depth], slot_of[arm_y * ARM_LEN + depth])


# --------------------------------------------------------------------------------------
# Sampling helpers.  All randomness flows through one numpy Generator per item.
# --------------------------------------------------------------------------------------
def _order_with_pins(rng: np.random.Generator, pins: Dict[int, int]) -> Tuple[int, ...]:
    """A uniformly random slot order subject to ``pins[slot] = structural_edge``."""
    order: List[Optional[int]] = [None] * N_EDGES
    used = set()
    for slot, edge in pins.items():
        if order[slot] is not None:
            raise ValueError(f"two edges pinned to slot {slot}")
        order[slot] = edge
        used.add(edge)
    free = [e for e in range(N_EDGES) if e not in used]
    shuffled = [int(x) for x in rng.permutation(np.asarray(free, dtype=np.int64))]
    it = iter(shuffled)
    for t in range(N_EDGES):
        if order[t] is None:
            order[t] = next(it)
    return tuple(int(x) for x in order)  # type: ignore[arg-type]


def _gap_matched_slot_pairs(
    rng: np.random.Generator,
) -> Tuple[Tuple[int, int], Tuple[int, int], int]:
    """Two disjoint slot pairs with identical gap ``q - p`` (invariant LS-1).

    Gap 19 is excluded because ``(0,19)`` is then the only pair, so no disjoint partner
    exists.  For every gap in 1..18 at least two disjoint pairs exist.
    """
    while True:
        gap = int(rng.integers(1, N_EDGES - 1))  # 1..18
        starts = np.arange(N_EDGES - gap)
        perm = rng.permutation(starts)
        a = int(perm[0])
        partners = [int(b) for b in perm[1:] if abs(int(b) - a) != gap]
        if not partners:
            continue
        b = partners[0]
        return (a, a + gap), (b, b + gap), gap


def _sample_base_graph(rng: np.random.Generator, max_nodes: int) -> Tuple[int, Tuple[Tuple[int, ...], ...]]:
    """21 distinct node ids, source first, then five arms of four — as ``prepare.py:11-24``."""
    nodes = rng.permutation(max_nodes)[: N_EDGES + 1]
    source = int(nodes[0])
    arms = tuple(
        tuple(int(x) for x in nodes[1 + a * ARM_LEN : 1 + (a + 1) * ARM_LEN])
        for a in range(NUM_ARMS)
    )
    return source, arms


def sample_far_graph(
    rng: np.random.Generator,
    source: int,
    goal: int,
    other_nodes: Sequence[int],
    base_edges: frozenset,
    base_path: Tuple[int, ...],
    max_overlap: int,
    max_tries: int = 500,
) -> Tuple[Graph, int, int]:
    """Repartition the same node multiset into a low-edge-overlap valid graph.

    Source and goal tokens are held fixed (so the ``/source,goal`` field and every node
    frequency are preserved); the other 19 nodes are permuted freely and the goal is
    placed at the terminal position of arm 0.  Rejection sampling enforces
    ``|E_far & E_base| <= max_overlap`` and a different path.  The number of tries is
    recorded so the manifest shows how hard the constraint bit.
    """
    pool = np.asarray([n for n in other_nodes if n != goal], dtype=np.int64)
    for attempt in range(1, max_tries + 1):
        perm = rng.permutation(pool)
        arm0 = tuple(int(x) for x in perm[: ARM_LEN - 1]) + (goal,)
        rest = tuple(
            tuple(int(x) for x in perm[ARM_LEN - 1 + a * ARM_LEN : ARM_LEN - 1 + (a + 1) * ARM_LEN])
            for a in range(NUM_ARMS - 1)
        )
        order = tuple(int(x) for x in rng.permutation(N_EDGES))
        graph = Graph(source=source, goal=goal, arms=(arm0,) + rest, order=order)
        overlap = len(frozenset(graph.edges_in_order()) & base_edges)
        if overlap <= max_overlap and graph.path() != base_path:
            return graph, overlap, attempt
    raise RuntimeError(
        f"far-critical rejection sampling failed after {max_tries} tries "
        f"(max_overlap={max_overlap})"
    )


# --------------------------------------------------------------------------------------
# Quartet construction
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class QuartetConfig:
    max_nodes: int = MAX_NODES
    far_max_edge_overlap: int = 2
    far_max_tries: int = 500
    self_check: bool = True


def _condition_record(graph: Graph) -> Dict:
    line = graph.serialize()
    prompt = line[: line.index("=") + 1]
    return {
        "line": line,
        "prompt_sha256": sha256_text(prompt),
        "answer": list(graph.path()),
        "graph_key": canonical_graph_key(graph.edges_in_order(), graph.source, graph.goal),
    }


def make_quartet(
    master_seed: int,
    index: int,
    cfg: QuartetConfig = QuartetConfig(),
    slot_pairs: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
) -> Dict:
    """Build one matched quartet.  Pure function of ``(master_seed, index, cfg)``.

    ``slot_pairs`` overrides the gap-matched slot-pair sampler with two explicit pairs.
    It exists so the LS-1 negative control can build a quartet that *violates* the
    invariant and prove the checker catches it; nothing in the production path passes it,
    and ``cfg.self_check`` will reject the result when the pairs are not gap-matched.
    """
    rng = np.random.default_rng([master_seed, index])

    source, arms = _sample_base_graph(rng, cfg.max_nodes)
    goal = arms[0][-1]

    depth = int(rng.choice(np.asarray(SWAP_DEPTHS)))
    critical_arm = int(rng.integers(1, NUM_ARMS))
    distractors = [a for a in range(1, NUM_ARMS) if a != critical_arm]
    safe_arms = tuple(
        int(x) for x in rng.choice(np.asarray(distractors), size=2, replace=False)
    )

    if slot_pairs is None:
        pair_a, pair_b, gap = _gap_matched_slot_pairs(rng)
    else:
        pair_a, pair_b = ((int(x), int(y)) for x, y in slot_pairs)
        gap = pair_a[1] - pair_a[0]
    if rng.random() < 0.5:  # fair coin: edit position is exchangeable across conditions
        crit_slots, safe_slots = pair_a, pair_b
    else:
        crit_slots, safe_slots = pair_b, pair_a

    goal_first = bool(rng.random() < 0.5)
    safe_first = bool(rng.random() < 0.5)
    crit_edges = (0 * ARM_LEN + depth, critical_arm * ARM_LEN + depth)
    safe_edges = (safe_arms[0] * ARM_LEN + depth, safe_arms[1] * ARM_LEN + depth)

    base_pins = {
        crit_slots[0]: crit_edges[0 if goal_first else 1],
        crit_slots[1]: crit_edges[1 if goal_first else 0],
        safe_slots[0]: safe_edges[0 if safe_first else 1],
        safe_slots[1]: safe_edges[1 if safe_first else 0],
    }
    base = Graph(source, goal, arms, _order_with_pins(rng, base_pins))

    aligned_first = bool(rng.random() < 0.5)
    repeat_pins = {
        crit_slots[0]: safe_edges[0 if aligned_first else 1],
        crit_slots[1]: safe_edges[1 if aligned_first else 0],
    }
    repeat = Graph(source, goal, arms, _order_with_pins(rng, repeat_pins))

    near_critical = suffix_swap(base, 0, critical_arm, depth)
    near_safe = suffix_swap(base, safe_arms[0], safe_arms[1], depth)
    near_safe_aligned = suffix_swap(repeat, safe_arms[0], safe_arms[1], depth)

    base_edges = frozenset(base.edges_in_order())
    other_nodes = [n for arm in arms for n in arm]
    far_critical, overlap, tries = sample_far_graph(
        rng,
        source,
        goal,
        other_nodes,
        base_edges,
        base.path(),
        cfg.far_max_edge_overlap,
        cfg.far_max_tries,
    )

    record = {
        "quartet_id": index,
        "master_seed": master_seed,
        "rng_key": [master_seed, index],
        "depth": depth,
        "critical_arm": critical_arm,
        "safe_arms": list(safe_arms),
        "slot_gap": gap,
        "edit_slots": {
            "near_critical": list(swap_edit_slots(base, 0, critical_arm, depth)),
            "near_safe": list(swap_edit_slots(base, safe_arms[0], safe_arms[1], depth)),
            "near_safe_aligned": list(
                swap_edit_slots(repeat, safe_arms[0], safe_arms[1], depth)
            ),
        },
        "far_edge_overlap": overlap,
        "far_tries": tries,
        "conditions": {
            "base": _condition_record(base),
            "repeat": _condition_record(repeat),
            "near_safe": _condition_record(near_safe),
            "near_critical": _condition_record(near_critical),
            "near_safe_aligned": _condition_record(near_safe_aligned),
            "far_critical": _condition_record(far_critical),
        },
    }
    record["edit_token_positions"] = {
        cond: [edge_slot_token_positions(s)[1] for s in slots]
        for cond, slots in record["edit_slots"].items()
    }
    if cfg.self_check:
        problems = check_quartet(record, far_max_edge_overlap=cfg.far_max_edge_overlap)
        if problems:
            raise GraphError(f"quartet {index} failed self-check: {problems}")
    return record


# --------------------------------------------------------------------------------------
# Pools (spec §5 "Data pools")
# --------------------------------------------------------------------------------------
def _quartet_shard(args) -> List[Dict]:
    master_seed, lo, hi, cfg = args
    return [make_quartet(master_seed, i, cfg) for i in range(lo, hi)]


def build_e_lure(
    master_seed: int,
    n_quartets: int,
    cfg: QuartetConfig = QuartetConfig(),
    workers: int = 1,
) -> List[Dict]:
    """The held-out H1/H2 pool.  Identical output for any ``workers``."""
    if workers <= 1:
        return [make_quartet(master_seed, i, cfg) for i in range(n_quartets)]
    step = max(1, -(-n_quartets // (workers * 4)))
    jobs = [
        (master_seed, lo, min(lo + step, n_quartets), cfg)
        for lo in range(0, n_quartets, step)
    ]
    # Explicit context: the start method must not be inherited from whatever imported
    # this module, because a fork-vs-spawn difference would be a silent behaviour change
    # in a function whose whole contract is worker-count independence.
    with mp.get_context("spawn").Pool(workers) as pool:
        parts = pool.map(_quartet_shard, jobs)
    return [rec for part in parts for rec in part]


def near_lure_of(graph: Graph, critical_arm: int, depth: int) -> Graph:
    """The near-critical lure of a graph whose goal arm is arm 0."""
    if graph.goal_arm() != 0:
        raise ValueError("expected the goal arm to be arm 0")
    return suffix_swap(graph, 0, critical_arm, depth)


def build_a_pair_pools(
    master_seed: int,
    train_lines: Sequence[str],
    n_items: int,
    near_per_item: int,
    far_per_item: int,
    cfg: QuartetConfig = QuartetConfig(),
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """``A_pair`` (base-training items) plus their ``B_near`` / ``B_far`` lure banks.

    Selection is model-blind by construction: line indices are drawn from a seeded
    ``Generator`` and nothing about any model is consulted.  ``B_far`` is deliberately a
    *candidate bank* larger than ``B_near`` — spec §6 requires the far items actually
    used to be chosen later by a loss-quantile match on a non-confirmatory pilot
    checkpoint, and that mapping is then frozen.
    """
    rng = np.random.default_rng([master_seed, 0])
    picks = rng.choice(len(train_lines), size=n_items, replace=False)
    picks = np.sort(picks)

    a_pair: List[Dict] = []
    b_near: List[Dict] = []
    b_far: List[Dict] = []

    combos = [(a, d) for a in range(1, NUM_ARMS) for d in SWAP_DEPTHS]
    for item_id, line_index in enumerate(int(x) for x in picks):
        line = train_lines[line_index].rstrip("\n")
        graph = graph_from_line(line)
        if graph.serialize() != line:
            raise GraphError(f"round trip failed for training line {line_index}")
        base_rec = _condition_record(graph)
        a_pair.append(
            {
                "item_id": item_id,
                "train_line_index": line_index,
                "pool": "A_pair",
                **base_rec,
            }
        )

        n_rng = np.random.default_rng([master_seed, 1, item_id])
        chosen = n_rng.choice(len(combos), size=near_per_item, replace=False)
        for k, ci in enumerate(int(x) for x in chosen):
            arm, depth = combos[ci]
            lure = near_lure_of(graph, arm, depth)
            b_near.append(
                {
                    "item_id": f"{item_id}:{k}",
                    "parent_item_id": item_id,
                    "parent_train_line_index": line_index,
                    "pool": "B_near",
                    "kind": "near_critical",
                    "critical_arm": arm,
                    "depth": depth,
                    "edit_slots": list(swap_edit_slots(graph, 0, arm, depth)),
                    **_condition_record(lure),
                }
            )

        f_rng = np.random.default_rng([master_seed, 2, item_id])
        base_edges = frozenset(graph.edges_in_order())
        others = [n for a in graph.arms for n in a]
        for k in range(far_per_item):
            far, overlap, tries = sample_far_graph(
                f_rng,
                graph.source,
                graph.goal,
                others,
                base_edges,
                graph.path(),
                cfg.far_max_edge_overlap,
                cfg.far_max_tries,
            )
            b_far.append(
                {
                    "item_id": f"{item_id}:{k}",
                    "parent_item_id": item_id,
                    "parent_train_line_index": line_index,
                    "pool": "B_far",
                    "kind": "far_critical",
                    "edge_overlap": overlap,
                    "tries": tries,
                    **_condition_record(far),
                }
            )
    return a_pair, b_near, b_far


# --------------------------------------------------------------------------------------
# Manifest I/O
# --------------------------------------------------------------------------------------
def write_jsonl(records: Sequence[Dict], path: pathlib.Path) -> str:
    """Write JSONL atomically and drop a ``.sha256`` sidecar next to it."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
    tmp.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n"
    )
    return digest


def read_jsonl(path) -> List[Dict]:
    with pathlib.Path(path).open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def leaked_quartet_ids(e_lure: Sequence[Dict], index: TrainingIndex) -> List[int]:
    """Quartet ids with any condition present in the training corpus.

    The membership test recomputes the canonical key and the prompt hash **from
    ``c["line"]``**, never from the record's own ``graph_key`` / ``prompt_sha256``
    fields.  A gate that consults a self-reported hash cannot see a record whose hash is
    stale, tampered with, or produced by a buggy ``_condition_record`` — which is exactly
    how this project's first leakage check managed to be vacuous.
    ``validate.check_quartet`` separately cross-checks that the stored fields agree with
    the line, so the two guards are independent.
    """
    return [
        r["quartet_id"]
        for r in e_lure
        for c in r["conditions"].values()
        if canonical_key_from_line(c["line"]) in index.graph_keys
        or sha256_text(parse_line(c["line"]).prompt) in index.prompt_hashes
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = pathlib.Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Build the Lure-Star stimulus pools.")
    ap.add_argument("--master-seed", type=int, default=20260823)
    ap.add_argument("--n-quartets", type=int, default=2000)
    ap.add_argument("--n-a-pair", type=int, default=1000)
    ap.add_argument("--near-per-item", type=int, default=5)
    ap.add_argument("--far-per-item", type=int, default=15)
    ap.add_argument("--far-max-edge-overlap", type=int, default=2)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--train-file", default=str(root / "data/stargraph/graph_5_5_sample_200000.txt"))
    ap.add_argument("--out-dir", default=str(root / "manifests"))
    ap.add_argument("--skip-leakage-check", action="store_true")
    a = ap.parse_args(argv)

    cfg = QuartetConfig(far_max_edge_overlap=a.far_max_edge_overlap)
    out = pathlib.Path(a.out_dir)

    e_lure = build_e_lure(a.master_seed, a.n_quartets, cfg, a.workers)
    train_lines = pathlib.Path(a.train_file).read_text().splitlines()
    a_pair, b_near, b_far = build_a_pair_pools(
        a.master_seed, train_lines, a.n_a_pair, a.near_per_item, a.far_per_item, cfg
    )

    if not a.skip_leakage_check:
        index = TrainingIndex.build(a.train_file)
        leaked = leaked_quartet_ids(e_lure, index)
        if leaked:
            raise SystemExit(f"E_lure leaked into training: quartets {leaked[:10]}")
        a_keys = {r["graph_key"] for r in a_pair}
        b_keys = {r["graph_key"] for r in b_near} | {r["graph_key"] for r in b_far}
        e_keys = {c["graph_key"] for r in e_lure for c in r["conditions"].values()}
        if e_keys & (a_keys | b_keys):
            raise SystemExit("E_lure overlaps the H3 pools")

    digests = {
        "e_lure.jsonl": write_jsonl(e_lure, out / "e_lure.jsonl"),
        "a_pair.jsonl": write_jsonl(a_pair, out / "a_pair.jsonl"),
        "b_near.jsonl": write_jsonl(b_near, out / "b_near.jsonl"),
        "b_far.jsonl": write_jsonl(b_far, out / "b_far.jsonl"),
    }
    overlaps = np.array([r["far_edge_overlap"] for r in e_lure])
    provenance = {
        "generator": "src/lurestar/generate.py",
        "master_seed": a.master_seed,
        "config": asdict(cfg),
        "counts": {
            "e_lure_quartets": len(e_lure),
            "a_pair": len(a_pair),
            "b_near": len(b_near),
            "b_far": len(b_far),
        },
        "train_file": a.train_file,
        "invariants": ["LS-1", "LS-2"],
        "far_edge_overlap_histogram": {
            str(v): int((overlaps == v).sum()) for v in sorted(set(overlaps.tolist()))
        },
        "sha256": digests,
    }
    (out / "stimuli_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
