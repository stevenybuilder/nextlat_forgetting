"""Independent solver and validator for Path-Star graphs and Lure-Star quartets.

Nothing in this module is allowed to trust construction bookkeeping.  Every function
here takes a *serialized line* or a raw edge list and re-derives the graph, the arms and
the source-to-goal path from scratch, so a generator bug cannot validate itself.  The
generator calls into this module for its own self-check, and the acceptance tests call
into it again on the written manifests.

Upstream anchors (pinned commit ``3770be6009cea2b3c455a9ce7f2ca88b504bb955``):

* serialized grammar — ``upstream/NextLat/data/stargraph/prepare.py:70-73``
* graph construction (source popped first, ``numOfPathsFromSource`` arms of
  ``lenOfEachPath - 1`` edges, arm ``p == 0`` carries the goal) — ``prepare.py:8-36``
* tokenizer: ``,`` is dropped, ``|``/``=``/``/`` map to ``maxNodes + 0/1/2``, one EOS
  (``maxNodes + 4``) is appended per line — ``upstream/NextLat/data/stargraph.py:9-57``
"""

from __future__ import annotations

import hashlib
import pathlib
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

# --------------------------------------------------------------------------------------
# G(5,5) constants.  ``ARM_LEN`` is ``path_length - 1`` edges per arm (prepare.py:21).
# --------------------------------------------------------------------------------------
NUM_ARMS = 5
ARM_LEN = 4
MAX_NODES = 100

Edge = Tuple[int, int]


class GraphError(ValueError):
    """A serialized line is not a valid G(d,l) Path-Star instance."""


# --------------------------------------------------------------------------------------
# Tokenizer — a faithful re-implementation of upstream ``Tokenizer`` (stargraph.py:9-57)
# --------------------------------------------------------------------------------------
_DIGITS = frozenset("0123456789")


def token_ids(text: str, max_nodes: int = MAX_NODES, eos: bool = False) -> List[int]:
    """Token ids for ``text`` under the upstream stargraph tokenizer.

    Commas carry no token (``stargraph.py:35-37``); ``|``/``=``/``/`` map to
    ``max_nodes``/``max_nodes + 1``/``max_nodes + 2``.  ``eos=True`` appends
    ``max_nodes + 4``, which ``tokenize`` (``stargraph.py:51-57``) does for a full line.
    """
    special = {"|": max_nodes, "=": max_nodes + 1, "/": max_nodes + 2, "$": max_nodes + 3}
    out: List[int] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == ",":
            i += 1
            continue
        j = i
        while j < n and text[j] in _DIGITS:
            j += 1
        if j > i:
            value = int(text[i:j])
            if not 0 <= value < max_nodes:
                raise GraphError(f"node id {value} outside 0..{max_nodes - 1}")
            out.append(value)
            i = j
        else:
            ch = text[i]
            if ch not in special:
                raise GraphError(f"unexpected character {ch!r} in {text[:40]!r}")
            out.append(special[ch])
            i += 1
    if eos:
        out.append(max_nodes + 4)
    return out


def edge_slot_token_positions(slot: int) -> Tuple[int, int]:
    """Token indices of the (tail, head) of serialized edge ``slot``.

    Each slot contributes ``tail, head, separator`` to the token stream, so slot ``t``
    owns indices ``3t`` and ``3t + 1``.  Checked against the verified tokenization in
    ``docs/UPSTREAM_REPORT.md`` §1.6: slot 19's head sits at index 58, ``/`` at 59,
    source at 60, goal at 61, ``=`` at 62.
    """
    return 3 * slot, 3 * slot + 1


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ParsedLine:
    edges: Tuple[Edge, ...]
    source: int
    goal: int
    answer: Tuple[int, ...]
    prompt: str
    line: str


def parse_line(line: str) -> ParsedLine:
    """Split one serialized example into its four fields (``prepare.py:70-73``)."""
    line = line.rstrip("\n")
    if line.count("/") != 1 or line.count("=") != 1:
        raise GraphError(f"line must contain exactly one '/' and one '=': {line[:60]!r}")
    edge_text, rest = line.split("/", 1)
    query, answer_text = rest.split("=", 1)
    if "," not in query:
        raise GraphError("query field must be 'source,goal'")
    src_text, goal_text = query.split(",", 1)

    edges: List[Edge] = []
    for token in edge_text.split("|"):
        parts = token.split(",")
        if len(parts) != 2:
            raise GraphError(f"edge {token!r} is not 'u,v'")
        edges.append((int(parts[0]), int(parts[1])))

    answer = tuple(int(p) for p in answer_text.split(","))
    return ParsedLine(
        edges=tuple(edges),
        source=int(src_text),
        goal=int(goal_text),
        answer=answer,
        prompt=line[: line.index("=") + 1],
        line=line,
    )


# --------------------------------------------------------------------------------------
# The solver
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SolvedGraph:
    source: int
    goal: int
    arms: Tuple[Tuple[int, ...], ...]  # serialized-order arms, goal arm listed first
    path: Tuple[int, ...]


def solve_edges(
    edges: Sequence[Edge],
    source: int,
    goal: int,
    num_arms: int = NUM_ARMS,
    arm_len: int = ARM_LEN,
    max_nodes: int = MAX_NODES,
) -> SolvedGraph:
    """Recover the arms and the source-to-goal path *from the edge list alone*.

    Raises :class:`GraphError` on anything that is not a valid G(``num_arms``,
    ``arm_len + 1``) star: wrong edge count, a repeated edge, a node with the wrong
    in/out degree, an arm of the wrong length, overlapping arms, or a goal that is not
    the terminal node of exactly one arm.
    """
    n_expected = num_arms * arm_len
    if len(edges) != n_expected:
        raise GraphError(f"expected {n_expected} edges, got {len(edges)}")
    if len(set(edges)) != n_expected:
        raise GraphError("duplicate edge in edge list")

    adj: Dict[int, List[int]] = defaultdict(list)
    indeg: Counter = Counter()
    nodes: Set[int] = set()
    for u, v in edges:
        if not (0 <= u < max_nodes and 0 <= v < max_nodes):
            raise GraphError(f"node id outside 0..{max_nodes - 1} in edge ({u},{v})")
        if u == v:
            raise GraphError(f"self loop at {u}")
        adj[u].append(v)
        indeg[v] += 1
        nodes.add(u)
        nodes.add(v)

    if len(nodes) != n_expected + 1:
        raise GraphError(f"expected {n_expected + 1} distinct nodes, got {len(nodes)}")
    if source not in nodes:
        raise GraphError("source is not a node of the graph")
    if goal not in nodes:
        raise GraphError("goal is not a node of the graph")
    if source == goal:
        raise GraphError("source equals goal")
    if indeg[source] != 0:
        raise GraphError("source has an incoming edge")
    for node in nodes:
        if node == source:
            continue
        if indeg[node] != 1:
            raise GraphError(f"node {node} has in-degree {indeg[node]}, expected 1")
    if len(adj[source]) != num_arms:
        raise GraphError(f"source out-degree {len(adj[source])}, expected {num_arms}")

    arms: List[Tuple[int, ...]] = []
    seen: Set[int] = set()
    for first in adj[source]:
        arm: List[int] = []
        cur = first
        for step in range(arm_len):
            if cur in seen or cur == source:
                raise GraphError("arms are not disjoint")
            seen.add(cur)
            arm.append(cur)
            succ = adj.get(cur, [])
            if step < arm_len - 1:
                if len(succ) != 1:
                    raise GraphError(
                        f"interior node {cur} has out-degree {len(succ)}, expected 1"
                    )
                cur = succ[0]
            elif succ:
                raise GraphError(f"arm terminal {cur} has out-degree {len(succ)}")
        arms.append(tuple(arm))

    if seen | {source} != nodes:
        raise GraphError("arms do not cover every node")

    goal_arms = [i for i, arm in enumerate(arms) if arm[-1] == goal]
    if len(goal_arms) != 1:
        raise GraphError(
            f"goal {goal} terminates {len(goal_arms)} arms, expected exactly 1"
        )
    g = goal_arms[0]
    ordered = (arms[g],) + tuple(arm for i, arm in enumerate(arms) if i != g)
    return SolvedGraph(
        source=source, goal=goal, arms=ordered, path=(source,) + arms[g]
    )


def solve_line(line: str, **kwargs) -> SolvedGraph:
    p = parse_line(line)
    return solve_edges(p.edges, p.source, p.goal, **kwargs)


def validate_line(line: str, **kwargs) -> SolvedGraph:
    """Solve ``line`` and check that its recorded answer is the solver's path."""
    p = parse_line(line)
    solved = solve_edges(p.edges, p.source, p.goal, **kwargs)
    if p.answer != solved.path:
        raise GraphError(
            f"recorded answer {p.answer} != solver path {solved.path}"
        )
    return solved


# --------------------------------------------------------------------------------------
# Canonical identities and leakage
# --------------------------------------------------------------------------------------
def canonical_graph_key(edges: Iterable[Edge], source: int, goal: int) -> str:
    """Order-invariant identity of a graph: same key iff same edge set + source + goal.

    Two serializations of one graph (base and repeat) collide here on purpose — that is
    what makes this a stronger leakage test than hashing the prompt string.

    The canonical order is a **lexicographic sort of the formatted edge strings**, not a
    numeric sort of the pairs.  Both are total orders, but they disagree ("10,5" < "2,3"
    as text, the reverse as numbers), and :meth:`TrainingIndex.build` reaches the same key
    without parsing 4 million integers.  The two implementations must agree; the leakage
    positive control in ``tests/test_lure_generator.py`` is what enforces that, and it is
    what caught this exact mismatch the first time round.
    """
    body = "|".join(sorted(f"{u},{v}" for u, v in edges))
    return hashlib.sha256(f"{body}/{source},{goal}".encode()).hexdigest()


def canonical_key_from_line(line: str) -> str:
    p = parse_line(line)
    return canonical_graph_key(p.edges, p.source, p.goal)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class TrainingIndex:
    """Membership index over a training corpus, for the no-leakage acceptance test."""

    prompt_hashes: Set[str]
    graph_keys: Set[str]
    n_lines: int
    path: str

    @classmethod
    def build(cls, path) -> "TrainingIndex":
        path = pathlib.Path(path)
        prompt_hashes: Set[str] = set()
        graph_keys: Set[str] = set()
        n = 0
        sha = hashlib.sha256
        with path.open("r") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    continue
                n += 1
                cut = line.index("=") + 1
                prompt_hashes.add(sha(line[:cut].encode()).hexdigest())
                edge_text, rest = line.split("/", 1)
                query = rest.split("=", 1)[0]
                body = "|".join(sorted(edge_text.split("|")))
                graph_keys.add(sha(f"{body}/{query}".encode()).hexdigest())
        return cls(prompt_hashes, graph_keys, n, str(path))

    def contains_prompt(self, prompt: str) -> bool:
        return sha256_text(prompt) in self.prompt_hashes

    def contains_graph(self, line: str) -> bool:
        return canonical_key_from_line(line) in self.graph_keys


# --------------------------------------------------------------------------------------
# Quartet invariants (spec §5) — see docs/STIMULUS_DESIGN.md for the exact statement
# --------------------------------------------------------------------------------------
#: Conditions written for every quartet.  ``near_safe_aligned`` is this project's
#: position-exact robustness variant; see docs/STIMULUS_DESIGN.md.
CONDITIONS = (
    "base",
    "repeat",
    "near_safe",
    "near_critical",
    "near_safe_aligned",
    "far_critical",
)

#: Which serialization each condition is a two-token edit of.  ``near_safe`` is anchored
#: on ``base`` (the spec-literal PSI); ``near_safe_aligned`` is anchored on ``repeat``,
#: which is what buys identical absolute edit positions against ``near_critical``.
NEAR_ANCHOR = {
    "near_safe": "base",
    "near_critical": "base",
    "near_safe_aligned": "repeat",
}


def token_diff_positions(a: str, b: str, max_nodes: int = MAX_NODES) -> List[int]:
    """Indices where the token streams of two equal-length prompts differ."""
    ta = token_ids(a, max_nodes)
    tb = token_ids(b, max_nodes)
    if len(ta) != len(tb):
        raise GraphError(f"prompt token lengths differ: {len(ta)} vs {len(tb)}")
    return [i for i, (x, y) in enumerate(zip(ta, tb)) if x != y]


def node_frequency(prompt: str, max_nodes: int = MAX_NODES) -> Counter:
    """Multiset of node tokens in a prompt (specials excluded)."""
    return Counter(t for t in token_ids(prompt, max_nodes) if t < max_nodes)


def degree_sequence(edges: Iterable[Edge]) -> Tuple[Tuple[int, int], ...]:
    """Sorted per-node ``(out_degree, in_degree)`` multiset."""
    out: Counter = Counter()
    inn: Counter = Counter()
    nodes: Set[int] = set()
    for u, v in edges:
        out[u] += 1
        inn[v] += 1
        nodes.add(u)
        nodes.add(v)
    return tuple(sorted((out[n], inn[n]) for n in nodes))


#: Default cap on ``|E_far & E_base|``, mirroring ``QuartetConfig.far_max_edge_overlap``.
#: It lives here as well as in the generator because the checker is what certifies a
#: written manifest, and a checker that has never heard of the cap will certify a near
#: lure relabelled as the far control.
FAR_MAX_EDGE_OVERLAP = 2


def _depth_of_head_node(solved: "SolvedGraph") -> Dict[int, int]:
    """``node -> its 0-based position along its own arm`` for a solved graph.

    The head of the depth-``d`` edge of an arm is the node at arm position ``d``, so this
    is exactly "which suffix depth does the edge ending at this node sit at".  Derived
    from the solver, never from generator bookkeeping.
    """
    out: Dict[int, int] = {}
    for arm in solved.arms:
        for d, node in enumerate(arm):
            out[node] = d
    return out


def check_quartet(
    record: Mapping,
    max_nodes: int = MAX_NODES,
    far_max_edge_overlap: int = FAR_MAX_EDGE_OVERLAP,
) -> List[str]:
    """Re-derive every spec §5 matching requirement from a written quartet record.

    Returns a list of human-readable violations; empty means the quartet passes.  This
    reads only ``record["conditions"][*]["line"]`` and the recorded edit slots — never
    the generator's internal graph objects, and never a stored hash or a stored answer:
    ``graph_key``, ``prompt_sha256`` and ``answer`` are all recomputed from ``line``,
    because the no-leakage gate consults ``graph_key`` and a gate that reads a
    self-reported field is not a gate.
    """
    problems: List[str] = []
    cond = record["conditions"]
    missing = [c for c in CONDITIONS if c not in cond]
    if missing:
        return [f"missing conditions: {missing}"]

    lines = {c: cond[c]["line"] for c in CONDITIONS}
    parsed = {}
    solved = {}
    for c, line in lines.items():
        try:
            parsed[c] = parse_line(line)
            solved[c] = validate_line(line)
        except GraphError as exc:
            problems.append(f"{c}: {exc}")
    if problems:
        return problems

    base = solved["base"]
    base_p = parsed["base"]

    # --- answers -----------------------------------------------------------------
    for c in ("repeat", "near_safe", "near_safe_aligned"):
        if solved[c].path != base.path:
            problems.append(f"{c}: path {solved[c].path} != base path {base.path}")
    if solved["near_critical"].path == base.path:
        problems.append("near_critical: path unchanged")
    if solved["near_critical"].path[1] == base.path[1]:
        problems.append("near_critical: first branch unchanged")
    if solved["far_critical"].path == base.path:
        problems.append("far_critical: path unchanged")

    # --- invariants shared by every condition -------------------------------------
    base_freq = node_frequency(base_p.prompt, max_nodes)
    base_deg = degree_sequence(base_p.edges)
    base_prompt_len = len(token_ids(base_p.prompt, max_nodes))
    for c in CONDITIONS:
        p = parsed[c]
        if p.source != base_p.source:
            problems.append(f"{c}: source changed")
        if p.goal != base_p.goal:
            problems.append(f"{c}: goal token changed")
        if len(p.answer) != len(base_p.answer):
            problems.append(f"{c}: answer length changed")
        if len(token_ids(p.prompt, max_nodes)) != base_prompt_len:
            problems.append(f"{c}: prompt token length changed")
        freq = node_frequency(p.prompt, max_nodes)
        if c == "far_critical":
            # Spec §5 asks far-critical only for "the same node multiset"; it is a
            # repartition, so a node that terminated an arm in base may sit mid-arm in
            # far.  The node *set* and the multiset of per-node frequencies are still
            # exactly preserved, and that is what is asserted here.
            if set(freq) != set(base_freq):
                problems.append("far_critical: node set changed")
            if sorted(freq.values()) != sorted(base_freq.values()):
                problems.append("far_critical: node frequency multiset changed")
        elif freq != base_freq:
            problems.append(f"{c}: node multiset/frequency changed")
        if degree_sequence(p.edges) != base_deg:
            problems.append(f"{c}: degree sequence changed")

    # --- repeat is an order-only reshuffle ----------------------------------------
    if set(parsed["repeat"].edges) != set(base_p.edges):
        problems.append("repeat: edge set changed")
    if parsed["repeat"].edges == base_p.edges:
        problems.append("repeat: edge order identical to base")

    # --- the two-token edits at their recorded slots -------------------------------
    slots = record["edit_slots"]
    for c, anchor in NEAR_ANCHOR.items():
        expected = sorted(
            i for s in slots[c] for i in (edge_slot_token_positions(s)[1],)
        )
        got = sorted(token_diff_positions(parsed[anchor].prompt, parsed[c].prompt, max_nodes))
        if got != expected:
            problems.append(
                f"{c}: prompt differs from {anchor} at token positions {got}, "
                f"expected exactly the head tokens {expected}"
            )

    # --- the matched-position invariant (docs/STIMULUS_DESIGN.md, LS-1/LS-2) --------
    pc = sorted(slots["near_critical"])
    ps = sorted(slots["near_safe"])
    pa = sorted(slots["near_safe_aligned"])
    if pc[1] - pc[0] != ps[1] - ps[0]:
        problems.append(
            f"LS-1 violated: critical slot gap {pc[1] - pc[0]} != safe slot gap "
            f"{ps[1] - ps[0]}"
        )
    if pa != pc:
        problems.append(f"LS-2 violated: aligned slots {pa} != critical slots {pc}")
    if len(set(pc) | set(ps)) != 4:
        problems.append("critical and safe edits share a serialized slot")

    # --- LS-0: the two near lures are swaps at ONE common suffix depth --------------
    # Graph validity already forces the two edges *within* one swap to share a depth (a
    # head transposition across unequal depths leaves an arm of the wrong length, which
    # the solver rejects).  It does NOT force the safe swap and the critical swap to use
    # the same depth, and depth is the magnitude of the manipulation: a depth-1 swap
    # moves three nodes per arm, a depth-3 swap moves one.  Both are two-token prompt
    # edits, so every token-level assertion above passes on a depth-mismatched pair.
    derived_depths: Dict[str, List[int]] = {}
    for c, anchor in NEAR_ANCHOR.items():
        depth_of = _depth_of_head_node(solved[anchor])
        anchor_edges = parsed[anchor].edges
        ds: List[int] = []
        for s in slots[c]:
            if not 0 <= s < len(anchor_edges):
                problems.append(f"{c}: edit slot {s} outside the edge list")
                continue
            head = anchor_edges[s][1]
            if head not in depth_of:
                problems.append(f"{c}: edited head {head} is not an arm node of {anchor}")
                continue
            ds.append(depth_of[head])
        derived_depths[c] = ds
        if len(set(ds)) != 1:
            problems.append(f"{c}: edited edges sit at different suffix depths {ds}")
        elif not 1 <= ds[0] <= ARM_LEN - 1:
            problems.append(f"{c}: suffix depth {ds[0]} outside 1..{ARM_LEN - 1}")
    flat = {d for ds in derived_depths.values() for d in ds}
    if len(flat) > 1:
        problems.append(
            "LS-0 violated: the near lures are swaps at different suffix depths "
            + str({c: sorted(set(ds)) for c, ds in derived_depths.items()})
        )
    elif "depth" in record and flat and record["depth"] != next(iter(flat)):
        problems.append(
            f"recorded depth {record['depth']} != re-derived depth {next(iter(flat))}"
        )

    # --- far-critical is a low-overlap repartition of the same nodes ---------------
    overlap = len(set(parsed["far_critical"].edges) & set(base_p.edges))
    if overlap != record["far_edge_overlap"]:
        problems.append(
            f"far_critical: recorded overlap {record['far_edge_overlap']} != "
            f"recomputed {overlap}"
        )
    if overlap > far_max_edge_overlap:
        problems.append(
            f"far_critical: edge overlap {overlap} exceeds the cap "
            f"{far_max_edge_overlap} — this is not a far control"
        )

    # --- stored identities must be recomputable from the line ----------------------
    # generate.py's leakage gate and the leakage tests read ``graph_key`` and
    # ``prompt_sha256`` off the record.  Recompute both here so a stale, tampered or
    # buggy identity field is a violation instead of a silent pass.
    for c in CONDITIONS:
        entry = cond[c]
        p = parsed[c]
        if "graph_key" in entry:
            want = canonical_graph_key(p.edges, p.source, p.goal)
            if entry["graph_key"] != want:
                problems.append(f"{c}: graph_key does not match its line")
        if "prompt_sha256" in entry:
            if entry["prompt_sha256"] != sha256_text(p.prompt):
                problems.append(f"{c}: prompt_sha256 does not match its line")
        if "answer" in entry:
            if tuple(entry["answer"]) != p.answer:
                problems.append(f"{c}: recorded answer does not match its line")
    return problems
