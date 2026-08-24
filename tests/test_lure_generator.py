"""Acceptance tests for the Lure-Star matched-quartet generator (spec §5).

Design rule for this file: every assertion must be able to fail on wrong input. Where an
invariant is cheap to satisfy accidentally, there is a paired *negative control* that
corrupts a real record and asserts the checker catches it. The solver is never allowed to
read the recorded answer before computing its own, and the leakage test is never allowed
to trust construction — it hashes against the actual 200,000-line training file and is
paired with a positive control proving the index would flag a training item.

The invariants LS-1 and LS-2 are stated and justified in ``docs/STIMULUS_DESIGN.md``.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np
import pytest

from lurestar.generate import (
    ARM_LEN,
    _condition_record,
    N_EDGES,
    NUM_ARMS,
    SWAP_DEPTHS,
    QuartetConfig,
    build_a_pair_pools,
    build_e_lure,
    graph_from_line,
    leaked_quartet_ids,
    make_quartet,
    read_jsonl,
    suffix_swap,
    swap_edit_slots,
    write_jsonl,
)
from lurestar.validate import (
    CONDITIONS,
    MAX_NODES,
    NEAR_ANCHOR,
    GraphError,
    TrainingIndex,
    canonical_key_from_line,
    check_quartet,
    degree_sequence,
    edge_slot_token_positions,
    node_frequency,
    parse_line,
    sha256_text,
    solve_line,
    token_diff_positions,
    token_ids,
    validate_line,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data" / "stargraph" / "graph_5_5_sample_200000.txt"
MASTER_SEED = 20260823
N_QUARTETS = 1000  # spec §5: "For at least 1,000 quartets, verify automatically"

#: The verbatim G(5,5) line and its tokenization recorded in docs/UPSTREAM_REPORT.md §1.6,
#: reproduced there from upstream's own Tokenizer.
UPSTREAM_LINE = (
    "49,97|65,62|36,85|51,38|61,45|49,12|64,17|5,33|12,79|49,64|62,51|45,74|"
    "49,61|74,27|17,36|32,68|97,53|79,32|49,65|53,5/49,33=49,97,53,5,33"
)
UPSTREAM_TOKENS = [
    49, 97, 100, 65, 62, 100, 36, 85, 100, 51, 38, 100, 61, 45, 100, 49, 12, 100,
    64, 17, 100, 5, 33, 100, 12, 79, 100, 49, 64, 100, 62, 51, 100, 45, 74, 100,
    49, 61, 100, 74, 27, 100, 17, 36, 100, 32, 68, 100, 97, 53, 100, 79, 32, 100,
    49, 65, 100, 53, 5, 102, 49, 33, 101, 49, 97, 53, 5, 33, 104,
]


# ======================================================================================
# Fixtures
# ======================================================================================
@pytest.fixture(scope="session")
def quartets():
    return build_e_lure(MASTER_SEED, N_QUARTETS, QuartetConfig(), workers=1)


@pytest.fixture(scope="session")
def train_lines():
    if not TRAIN_FILE.exists():
        pytest.skip(f"training corpus missing at {TRAIN_FILE}")
    return TRAIN_FILE.read_text().splitlines()


@pytest.fixture(scope="session")
def train_index():
    if not TRAIN_FILE.exists():
        pytest.skip(f"training corpus missing at {TRAIN_FILE}")
    return TrainingIndex.build(TRAIN_FILE)


@pytest.fixture(scope="session")
def h3_pools(train_lines):
    return build_a_pair_pools(MASTER_SEED, train_lines, 200, 5, 15, QuartetConfig())


# ======================================================================================
# 1. Format fidelity — our representation must be upstream's, byte for byte
# ======================================================================================
def test_tokenizer_matches_the_verified_upstream_tokenization():
    ids = token_ids(UPSTREAM_LINE, MAX_NODES, eos=True)
    assert ids == UPSTREAM_TOKENS
    assert len(ids) == 69
    assert ids.index(MAX_NODES + 1) == 62  # '=' at index 62


def test_edge_slot_token_positions_land_on_the_real_endpoints():
    parsed = parse_line(UPSTREAM_LINE)
    ids = token_ids(parsed.prompt, MAX_NODES)
    assert len(ids) == 63
    for slot, (u, v) in enumerate(parsed.edges):
        tail_i, head_i = edge_slot_token_positions(slot)
        assert ids[tail_i] == u, f"slot {slot} tail"
        assert ids[head_i] == v, f"slot {slot} head"
    # Negative control: an off-by-one slot mapping must NOT reproduce the edge list,
    # otherwise the assertions above would hold for a wrong formula too.
    shifted = [
        (ids[3 * t + 1], ids[3 * t + 2]) for t in range(len(parsed.edges) - 1)
    ]
    assert shifted != list(parsed.edges[: len(shifted)])


def test_graph_round_trips_real_corpus_lines_byte_for_byte(train_lines):
    picks = list(range(0, len(train_lines), 4001))
    assert len(picks) >= 49
    for i in picks:
        line = train_lines[i]
        graph = graph_from_line(line)
        assert graph.serialize() == line, f"round trip failed on training line {i}"
        assert graph.path() == parse_line(line).answer


def test_upstream_corpus_answers_are_reproduced_by_our_solver(train_lines):
    for i in range(0, len(train_lines), 3571):
        solved = validate_line(train_lines[i])
        assert len(solved.arms) == NUM_ARMS
        assert all(len(a) == ARM_LEN for a in solved.arms)
        assert len(solved.path) == ARM_LEN + 1


# ======================================================================================
# 2. The solver must be independent, and must reject wrong input
# ======================================================================================
def test_solver_ignores_the_recorded_answer(train_lines):
    """The path comes from the edge list, not from the '=' field."""
    line = train_lines[0]
    prompt, answer = line.split("=")
    wrong = prompt + "=" + ",".join(reversed(answer.split(",")))
    # solve_line still finds the true path ...
    assert solve_line(wrong).path == parse_line(line).answer
    # ... and validate_line rejects the corrupted answer.
    with pytest.raises(GraphError, match="solver path"):
        validate_line(wrong)


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda l: l.replace("|", "", 1), id="dropped_edge_separator"),
        pytest.param(lambda l: l.split("|", 1)[1], id="missing_edge"),
        pytest.param(lambda l: l + "|1,2", id="extra_edge"),
    ],
)
def test_solver_rejects_structurally_broken_lines(train_lines, mutate):
    with pytest.raises((GraphError, ValueError)):
        validate_line(mutate(train_lines[1]))


def test_solver_rejects_a_goal_that_is_not_an_arm_terminal(train_lines):
    line = train_lines[2]
    parsed = parse_line(line)
    solved = solve_line(line)
    interior = solved.arms[0][1]  # depth-2 node: never an arm terminal
    broken = line.replace(f"/{parsed.source},{parsed.goal}=", f"/{parsed.source},{interior}=")
    with pytest.raises(GraphError, match="terminates 0 arms"):
        solve_line(broken)


def test_solver_rejects_an_edge_rewired_into_a_second_parent(train_lines):
    """Redirect one edge so a node gains in-degree 2 — the exact failure a buggy swap makes."""
    line = train_lines[3]
    parsed = parse_line(line)
    (u0, v0), (u1, v1) = parsed.edges[0], parsed.edges[1]
    edges = list(parsed.edges)
    edges[0] = (u0, v1)  # v1 now has two parents, v0 has none
    body = "|".join(f"{a},{b}" for a, b in edges)
    broken = f"{body}/{parsed.source},{parsed.goal}=" + ",".join(str(n) for n in parsed.answer)
    with pytest.raises(GraphError):
        solve_line(broken)


# ======================================================================================
# 3. Spec §5 acceptance suite over >= 1,000 quartets
# ======================================================================================
def test_pool_size_meets_the_spec_minimum(quartets):
    assert len(quartets) >= 1000


def test_every_graph_is_solver_verified_valid(quartets):
    for rec in quartets:
        for cond in CONDITIONS:
            solved = validate_line(rec["conditions"][cond]["line"])
            assert len(solved.arms) == NUM_ARMS
            assert all(len(a) == ARM_LEN for a in solved.arms)


def test_full_invariant_checker_passes_on_every_quartet(quartets):
    failures = [(r["quartet_id"], check_quartet(r)) for r in quartets]
    failures = [f for f in failures if f[1]]
    assert not failures, failures[:5]


def test_answers_are_identical_for_base_repeat_and_safe(quartets):
    for rec in quartets:
        base = tuple(rec["conditions"]["base"]["answer"])
        for cond in ("repeat", "near_safe", "near_safe_aligned"):
            got = tuple(solve_line(rec["conditions"][cond]["line"]).path)
            assert got == base, (rec["quartet_id"], cond, got, base)


def test_near_critical_changes_first_branch_and_full_path(quartets):
    for rec in quartets:
        base = solve_line(rec["conditions"]["base"]["line"]).path
        crit = solve_line(rec["conditions"]["near_critical"]["line"]).path
        assert crit != base
        assert crit[1] != base[1], "first branch unchanged"
        assert crit[0] == base[0], "source moved"
        assert crit[-1] == base[-1], "goal node moved"


def test_near_critical_keeps_the_goal_token_but_moves_the_goal_arm(quartets):
    """Spec §5's subtle claim, asserted directly rather than assumed."""
    for rec in quartets:
        base_p = parse_line(rec["conditions"]["base"]["line"])
        crit_p = parse_line(rec["conditions"]["near_critical"]["line"])
        assert crit_p.goal == base_p.goal
        assert crit_p.prompt.split("/")[1] == base_p.prompt.split("/")[1]
        base_solved = solve_line(base_p.line)
        crit_solved = solve_line(crit_p.line)
        # The goal arm is a different node sequence, and its first node changed.
        assert crit_solved.arms[0] != base_solved.arms[0]
        assert crit_solved.arms[0][0] != base_solved.arms[0][0]
        # The old goal arm's suffix survives, now terminating the goal in another arm.
        depth = rec["depth"]
        assert crit_solved.arms[0][depth:] == base_solved.arms[0][depth:]


def test_far_critical_changes_the_path(quartets):
    for rec in quartets:
        base = solve_line(rec["conditions"]["base"]["line"]).path
        far = solve_line(rec["conditions"]["far_critical"]["line"]).path
        assert far != base


def test_both_near_lures_change_exactly_two_endpoint_tokens_at_recorded_slots(quartets):
    for rec in quartets:
        prompts = {c: parse_line(rec["conditions"][c]["line"]).prompt for c in CONDITIONS}
        for cond, anchor in NEAR_ANCHOR.items():
            diff = token_diff_positions(prompts[anchor], prompts[cond])
            expected = sorted(
                edge_slot_token_positions(s)[1] for s in rec["edit_slots"][cond]
            )
            assert diff == expected, (rec["quartet_id"], cond, diff, expected)
            assert len(diff) == 2
            # The changed tokens are heads (index % 3 == 1), i.e. endpoints of an edge.
            assert all(i % 3 == 1 for i in diff)


def test_token_edit_distance_from_base_is_equal_for_safe_and_critical(quartets):
    for rec in quartets:
        base = parse_line(rec["conditions"]["base"]["line"]).prompt
        d_safe = len(token_diff_positions(base, parse_line(rec["conditions"]["near_safe"]["line"]).prompt))
        d_crit = len(token_diff_positions(base, parse_line(rec["conditions"]["near_critical"]["line"]).prompt))
        assert d_safe == d_crit == 2, (rec["quartet_id"], d_safe, d_crit)


def test_LS1_edit_slot_pairs_are_disjoint_with_identical_gap(quartets):
    for rec in quartets:
        pc = sorted(rec["edit_slots"]["near_critical"])
        ps = sorted(rec["edit_slots"]["near_safe"])
        assert pc[1] - pc[0] == ps[1] - ps[0] == rec["slot_gap"]
        assert len(set(pc) | set(ps)) == 4
        assert all(0 <= s < N_EDGES for s in pc + ps)


def test_LS2_aligned_safe_edit_uses_the_same_absolute_positions_as_critical(quartets):
    for rec in quartets:
        assert sorted(rec["edit_slots"]["near_safe_aligned"]) == sorted(
            rec["edit_slots"]["near_critical"]
        )
        assert sorted(rec["edit_token_positions"]["near_safe_aligned"]) == sorted(
            rec["edit_token_positions"]["near_critical"]
        )


def test_edit_position_is_exchangeable_between_safe_and_critical(quartets):
    """LS-1's coin flip must actually be a coin flip, not a constant."""
    earlier_is_critical = sum(
        1
        for r in quartets
        if min(r["edit_slots"]["near_critical"]) < min(r["edit_slots"]["near_safe"])
    )
    n = len(quartets)
    # +/- 4 binomial SDs around n/2; a constant assignment (0 or n) fails by miles.
    tol = 4 * (0.25 * n) ** 0.5
    assert abs(earlier_is_critical - n / 2) < tol, (earlier_is_critical, n)
    crit_pos = np.array([s for r in quartets for s in r["edit_slots"]["near_critical"]])
    safe_pos = np.array([s for r in quartets for s in r["edit_slots"]["near_safe"]])
    assert abs(crit_pos.mean() - safe_pos.mean()) < 0.5, (crit_pos.mean(), safe_pos.mean())


def test_source_goal_multiset_degrees_and_lengths_are_preserved(quartets):
    for rec in quartets:
        base_p = parse_line(rec["conditions"]["base"]["line"])
        base_freq = node_frequency(base_p.prompt)
        base_deg = degree_sequence(base_p.edges)
        base_prompt_len = len(token_ids(base_p.prompt))
        assert base_prompt_len == 63
        for cond in CONDITIONS:
            p = parse_line(rec["conditions"][cond]["line"])
            assert p.source == base_p.source
            assert p.goal == base_p.goal
            assert len(token_ids(p.prompt)) == base_prompt_len
            assert len(p.answer) == len(base_p.answer) == ARM_LEN + 1
            assert degree_sequence(p.edges) == base_deg
            freq = node_frequency(p.prompt)
            if cond == "far_critical":
                assert set(freq) == set(base_freq)
                assert sorted(freq.values()) == sorted(base_freq.values())
            else:
                assert freq == base_freq, (rec["quartet_id"], cond)


def test_repeat_is_an_order_only_reshuffle(quartets):
    for rec in quartets:
        base_p = parse_line(rec["conditions"]["base"]["line"])
        rep_p = parse_line(rec["conditions"]["repeat"]["line"])
        assert set(rep_p.edges) == set(base_p.edges)
        assert rep_p.edges != base_p.edges
        assert canonical_key_from_line(rep_p.line) == canonical_key_from_line(base_p.line)


def test_all_six_conditions_are_distinct_strings(quartets):
    for rec in quartets:
        lines = {rec["conditions"][c]["line"] for c in CONDITIONS}
        assert len(lines) == len(CONDITIONS), rec["quartet_id"]


def test_far_critical_edge_overlap_is_low_and_reported(quartets):
    overlaps = []
    for rec in quartets:
        base_edges = set(parse_line(rec["conditions"]["base"]["line"]).edges)
        far_edges = set(parse_line(rec["conditions"]["far_critical"]["line"]).edges)
        ov = len(base_edges & far_edges)
        assert ov == rec["far_edge_overlap"]
        overlaps.append(ov)
    overlaps = np.array(overlaps)
    hist = {int(v): int((overlaps == v).sum()) for v in range(N_EDGES + 1) if (overlaps == v).any()}
    tries = np.array([r["far_tries"] for r in quartets])
    print(
        f"\nfar-critical edge overlap over {len(quartets)} quartets (out of {N_EDGES} edges):"
        f"\n  histogram        {hist}"
        f"\n  mean {overlaps.mean():.3f}  max {overlaps.max()}  "
        f"fraction<=1 {(overlaps <= 1).mean():.3f}"
        f"\n  rejection tries: mean {tries.mean():.3f}  max {tries.max()}  "
        f"first-try acceptance {(tries == 1).mean():.3f}"
    )
    assert overlaps.max() <= 2
    assert overlaps.mean() < 2.0
    # And the near lures are, by contrast, 18/20 overlapping — the control is real.
    near_ov = [
        len(
            set(parse_line(r["conditions"]["base"]["line"]).edges)
            & set(parse_line(r["conditions"]["near_critical"]["line"]).edges)
        )
        for r in quartets[:200]
    ]
    assert set(near_ov) == {N_EDGES - 2}


# ======================================================================================
# 4. Negative controls — the suite above must be capable of failing
# ======================================================================================
def test_checker_rejects_a_condition_taken_from_another_quartet(quartets):
    caught = 0
    for i in range(200):
        rec = json.loads(json.dumps(quartets[i]))
        rec["conditions"]["near_safe"] = quartets[i + 1]["conditions"]["near_safe"]
        if check_quartet(rec):
            caught += 1
    assert caught == 200


def test_checker_rejects_shuffled_condition_assignment(quartets):
    """If the suite passed on shuffled data it would not be a test."""
    caught = 0
    for i in range(200):
        rec = json.loads(json.dumps(quartets[i]))
        rec["conditions"]["near_critical"], rec["conditions"]["near_safe"] = (
            rec["conditions"]["near_safe"],
            rec["conditions"]["near_critical"],
        )
        if check_quartet(rec):
            caught += 1
    assert caught == 200


def test_checker_rejects_an_aligned_lure_anchored_on_the_wrong_string(quartets):
    """near_safe_aligned edits `repeat`; anchoring it on `base` must be caught."""
    rec = json.loads(json.dumps(quartets[0]))
    rec["conditions"]["near_safe_aligned"] = rec["conditions"]["near_safe"]
    problems = check_quartet(rec)
    assert problems and any("near_safe_aligned" in p for p in problems), problems


def test_checker_rejects_a_genuinely_gap_mismatched_quartet():
    """Build a real, fully consistent quartet that violates LS-1, and catch it.

    Every line here is valid, every lure is still an exact two-token edit at its own
    recorded slots — the *only* thing wrong is that the critical pair spans 3 slots and
    the safe pair spans 11. If the checker only re-derived edit positions it would pass
    this, so LS-1 has to be a separate assertion, and this proves it is.
    """
    rogue = make_quartet(
        MASTER_SEED, 0, QuartetConfig(self_check=False), slot_pairs=((2, 5), (7, 18))
    )
    assert sorted(rogue["edit_slots"]["near_critical"]) in ([2, 5], [7, 18])
    problems = check_quartet(rogue)
    assert problems, "a gap-mismatched quartet must not pass"
    assert all("LS-1" in p for p in problems), problems

    # The gap-matched version of the same item passes, so the failure is the gap alone.
    ok = make_quartet(
        MASTER_SEED, 0, QuartetConfig(self_check=False), slot_pairs=((2, 5), (7, 10))
    )
    assert not check_quartet(ok)


def test_self_check_refuses_to_emit_a_gap_mismatched_quartet():
    with pytest.raises(GraphError, match="LS-1"):
        make_quartet(MASTER_SEED, 0, QuartetConfig(), slot_pairs=((2, 5), (7, 18)))


def test_checker_rejects_near_lures_at_MISMATCHED_SUFFIX_DEPTHS(quartets):
    """P0 regression. A safe swap at one depth and a critical swap at another is an
    unmatched pair: depth is the magnitude of the perturbation (a depth-1 swap moves
    three nodes per arm, a depth-3 swap moves one), yet BOTH are two-token prompt edits,
    so every token-level assertion in this file passes on such a quartet. The checker
    must re-derive the depth from the solved anchor and refuse.
    """
    caught = 0
    tried = 0
    for rec in quartets[:400]:
        base = graph_from_line(rec["conditions"]["base"]["line"])
        s0, s1 = rec["safe_arms"]
        for d2 in SWAP_DEPTHS:
            if d2 == rec["depth"]:
                continue
            slots = sorted(swap_edit_slots(base, s0, s1, d2))
            pc = sorted(rec["edit_slots"]["near_critical"])
            # Only the depth may differ: keep the pair disjoint from critical's and give
            # it the same gap, so LS-1 still holds and depth is the ONLY violation.
            if slots[1] - slots[0] != pc[1] - pc[0] or set(slots) & set(pc):
                continue
            rogue = json.loads(json.dumps(rec))
            rogue["conditions"]["near_safe"] = _condition_record(
                suffix_swap(base, s0, s1, d2)
            )
            rogue["edit_slots"]["near_safe"] = slots
            rogue["edit_token_positions"]["near_safe"] = [
                edge_slot_token_positions(x)[1] for x in slots
            ]
            tried += 1
            problems = check_quartet(rogue)
            assert problems, (rec["quartet_id"], rec["depth"], d2)
            assert any("LS-0" in p for p in problems), problems
            # ... and nothing else is wrong with it: LS-1 still passes.
            assert not any("LS-1" in p for p in problems), problems
            caught += 1
            break
    assert tried >= 10, f"only {tried} depth-mismatched rogues constructible"
    assert caught == tried


def test_checker_rejects_a_near_lure_relabelled_as_the_FAR_CONTROL(quartets):
    """P0 regression. far_critical's defining property is low edge overlap. Without an
    explicit cap the checker certified an 18/20-overlap near lure as the far control,
    which is the control H3's primary near-minus-far contrast is measured against.
    """
    for rec in quartets[:50]:
        base = graph_from_line(rec["conditions"]["base"]["line"])
        near = suffix_swap(base, 0, rec["critical_arm"], rec["depth"])
        rogue = json.loads(json.dumps(rec))
        rogue["conditions"]["far_critical"] = _condition_record(near)
        overlap = len(
            set(parse_line(near.serialize()).edges)
            & set(parse_line(rec["conditions"]["base"]["line"]).edges)
        )
        assert overlap == N_EDGES - 2
        rogue["far_edge_overlap"] = overlap  # honestly recorded — only the cap catches it
        problems = check_quartet(rogue)
        assert any("exceeds the cap" in p for p in problems), problems
    # The cap is a parameter, and a loosened cap must actually loosen it.
    assert not any(
        "exceeds the cap" in p
        for p in check_quartet(quartets[0], far_max_edge_overlap=2)
    )
    tight = check_quartet(quartets[0], far_max_edge_overlap=-1)
    assert any("exceeds the cap" in p for p in tight), tight


@pytest.mark.parametrize("field", ["graph_key", "prompt_sha256", "answer"])
def test_checker_recomputes_stored_identities_and_rejects_a_wrong_one(quartets, field):
    """P0 regression. generate.py's no-leakage gate and the leakage tests consult
    ``graph_key``. A record whose stored key does not match its own line would sail
    through both — a training item copied verbatim with a bogus key reads as clean.
    """
    for cond in CONDITIONS:
        rec = json.loads(json.dumps(quartets[3]))
        entry = rec["conditions"][cond]
        entry[field] = "0" * 64 if field != "answer" else list(reversed(entry["answer"]))
        problems = check_quartet(rec)
        assert any(cond in p and field.split("_")[0] in p for p in problems), (
            cond, field, problems
        )


def test_leakage_gate_recomputes_the_key_instead_of_trusting_the_record(
    quartets, train_index, train_lines
):
    """P0 regression on the CLI gate itself (``generate.leaked_quartet_ids``).

    A verbatim training line smuggled into a quartet with a falsified ``graph_key`` and
    ``prompt_sha256`` must still be flagged. A gate that reads the stored fields returns
    "clean" here, which is precisely how a leakage guarantee becomes decorative.
    """
    clean = json.loads(json.dumps(quartets[:5]))
    assert leaked_quartet_ids(clean, train_index) == []

    poisoned = json.loads(json.dumps(quartets[:5]))
    line = train_lines[77]
    poisoned[2]["conditions"]["far_critical"] = {
        "line": line,
        "graph_key": "0" * 64,          # forged
        "prompt_sha256": "0" * 64,      # forged
        "answer": list(parse_line(line).answer),
    }
    assert leaked_quartet_ids(poisoned, train_index) == [poisoned[2]["quartet_id"]]

    # And a *reshuffled* training graph — a different prompt string, same graph — must
    # be caught too, otherwise only verbatim copies would be.
    g = graph_from_line(line)
    reshuffled = type(g)(g.source, g.goal, g.arms, tuple(int(x) for x in np.random.default_rng(5).permutation(N_EDGES)))
    assert reshuffled.serialize() != line
    poisoned[2]["conditions"]["far_critical"]["line"] = reshuffled.serialize()
    assert leaked_quartet_ids(poisoned, train_index) == [poisoned[2]["quartet_id"]]


def test_checker_rejects_a_tampered_answer(quartets):
    rec = json.loads(json.dumps(quartets[0]))
    line = rec["conditions"]["repeat"]["line"]
    head, tail = line.split("=")
    nodes = tail.split(",")
    nodes[1], nodes[2] = nodes[2], nodes[1]
    rec["conditions"]["repeat"]["line"] = head + "=" + ",".join(nodes)
    assert check_quartet(rec)


def test_checker_rejects_a_tampered_far_overlap_count(quartets):
    rec = json.loads(json.dumps(quartets[0]))
    rec["far_edge_overlap"] = rec["far_edge_overlap"] + 7
    assert any("far_critical" in p for p in check_quartet(rec))


def test_suffix_swap_at_depth_zero_is_refused():
    """Depth 0 would exchange whole arms and leave the edge set untouched."""
    rec = make_quartet(MASTER_SEED, 0)
    g = graph_from_line(rec["conditions"]["base"]["line"])
    with pytest.raises(ValueError):
        suffix_swap(g, 0, 1, 0)
    with pytest.raises(ValueError):
        suffix_swap(g, 1, 1, 2)
    assert set(SWAP_DEPTHS) == {1, 2, 3}


# ======================================================================================
# 5. Determinism
# ======================================================================================
def test_generation_is_deterministic_under_a_recorded_seed():
    a = build_e_lure(MASTER_SEED, 64, QuartetConfig(), workers=1)
    b = build_e_lure(MASTER_SEED, 64, QuartetConfig(), workers=1)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    c = build_e_lure(MASTER_SEED + 1, 64, QuartetConfig(), workers=1)
    assert json.dumps(a, sort_keys=True) != json.dumps(c, sort_keys=True)


def test_generation_is_identical_under_a_different_worker_count():
    one = build_e_lure(MASTER_SEED, 96, QuartetConfig(), workers=1)
    many = build_e_lure(MASTER_SEED, 96, QuartetConfig(), workers=4)
    assert json.dumps(one, sort_keys=True) == json.dumps(many, sort_keys=True)


def test_a_quartet_is_a_pure_function_of_seed_and_index(quartets):
    for i in (0, 17, 499, N_QUARTETS - 1):
        assert make_quartet(MASTER_SEED, i) == quartets[i]


# ======================================================================================
# 6. Leakage — checked by hashing, never by trusting construction
# ======================================================================================
def test_training_index_flags_a_real_training_item(train_index, train_lines):
    """Positive control: without this the leakage test could pass by being broken."""
    for i in (0, 1234, 199_999):
        line = train_lines[i]
        assert train_index.contains_prompt(line[: line.index("=") + 1])
        assert train_index.contains_graph(line)
    assert train_index.n_lines == 200_000


def test_no_evaluation_item_appears_in_the_training_corpus(quartets, train_index):
    for rec in quartets:
        for cond in CONDITIONS:
            entry = rec["conditions"][cond]
            prompt = parse_line(entry["line"]).prompt
            assert entry["prompt_sha256"] == sha256_text(prompt)
            assert entry["prompt_sha256"] not in train_index.prompt_hashes, (
                rec["quartet_id"],
                cond,
            )
            assert entry["graph_key"] not in train_index.graph_keys, (
                rec["quartet_id"],
                cond,
            )


def test_e_lure_is_disjoint_from_the_h3_pools(quartets, h3_pools):
    a_pair, b_near, b_far = h3_pools
    h3_keys = {r["graph_key"] for r in a_pair + b_near + b_far}
    e_keys = {c["graph_key"] for r in quartets for c in r["conditions"].values()}
    assert not (e_keys & h3_keys)
    h3_prompts = {r["prompt_sha256"] for r in a_pair + b_near + b_far}
    e_prompts = {c["prompt_sha256"] for r in quartets for c in r["conditions"].values()}
    assert not (e_prompts & h3_prompts)


# ======================================================================================
# 7. H3 pools
# ======================================================================================
def test_a_pair_items_really_are_training_items(h3_pools, train_index, train_lines):
    a_pair, _, _ = h3_pools
    assert len(a_pair) == 200
    assert len({r["train_line_index"] for r in a_pair}) == 200
    for rec in a_pair:
        assert rec["line"] == train_lines[rec["train_line_index"]]
        assert train_index.contains_graph(rec["line"])
        validate_line(rec["line"])


def test_a_pair_selection_is_model_blind_and_reproducible(train_lines):
    first = build_a_pair_pools(MASTER_SEED, train_lines, 40, 2, 2, QuartetConfig())
    second = build_a_pair_pools(MASTER_SEED, train_lines, 40, 2, 2, QuartetConfig())
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    other = build_a_pair_pools(MASTER_SEED + 1, train_lines, 40, 2, 2, QuartetConfig())
    assert [r["train_line_index"] for r in first[0]] != [r["train_line_index"] for r in other[0]]


def test_b_near_items_are_two_token_lures_of_their_parent(h3_pools):
    a_pair, b_near, _ = h3_pools
    parents = {r["item_id"]: r for r in a_pair}
    assert len(b_near) == 5 * len(a_pair)
    for rec in b_near:
        parent = parents[rec["parent_item_id"]]
        p_prompt = parse_line(parent["line"]).prompt
        l_prompt = parse_line(rec["line"]).prompt
        diff = token_diff_positions(p_prompt, l_prompt)
        assert diff == sorted(edge_slot_token_positions(s)[1] for s in rec["edit_slots"])
        assert len(diff) == 2
        lure_path = validate_line(rec["line"]).path
        parent_path = validate_line(parent["line"]).path
        assert lure_path != parent_path
        assert lure_path[1] != parent_path[1]
        assert lure_path[0] == parent_path[0] and lure_path[-1] == parent_path[-1]


def test_b_near_lures_of_one_parent_are_distinct(h3_pools):
    _, b_near, _ = h3_pools
    by_parent = {}
    for rec in b_near:
        by_parent.setdefault(rec["parent_item_id"], []).append(rec["line"])
    for parent, lines in by_parent.items():
        assert len(set(lines)) == len(lines), parent


def test_b_far_bank_is_large_low_overlap_and_valid(h3_pools):
    a_pair, b_near, b_far = h3_pools
    assert len(b_far) == 15 * len(a_pair)
    assert len(b_far) > len(b_near), "B_far must be an oversized candidate bank (spec §6)"
    parents = {r["item_id"]: r for r in a_pair}
    for rec in b_far:
        parent = parents[rec["parent_item_id"]]
        p = parse_line(parent["line"])
        f = parse_line(rec["line"])
        assert rec["edge_overlap"] == len(set(p.edges) & set(f.edges)) <= 2
        assert f.source == p.source and f.goal == p.goal
        assert validate_line(rec["line"]).path != validate_line(parent["line"]).path


# ======================================================================================
# 8. Manifests
# ======================================================================================
def test_manifests_round_trip_with_a_sha256_sidecar(tmp_path, quartets):
    path = tmp_path / "e_lure.jsonl"
    digest = write_jsonl(quartets[:50], path)
    sidecar = tmp_path / "e_lure.jsonl.sha256"
    assert sidecar.read_text().split()[0] == digest
    import hashlib

    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    assert read_jsonl(path) == quartets[:50]
    assert not list(tmp_path.glob("*.partial"))


def test_cli_builds_every_pool_and_its_hashes(tmp_path, train_lines):
    small_train = tmp_path / "graph_5_5_sample_2000.txt"
    small_train.write_text("\n".join(train_lines[:2000]) + "\n")
    out = tmp_path / "manifests"
    rc = subprocess.run(
        [
            sys.executable, "-m", "lurestar.generate",
            "--master-seed", str(MASTER_SEED),
            "--n-quartets", "40",
            "--n-a-pair", "10",
            "--near-per-item", "3",
            "--far-per-item", "4",
            "--workers", "1",
            "--train-file", str(small_train),
            "--out-dir", str(out),
        ],
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
    )
    assert rc.returncode == 0, rc.stderr[-3000:]
    prov = json.loads((out / "stimuli_provenance.json").read_text())
    assert prov["counts"] == {
        "e_lure_quartets": 40, "a_pair": 10, "b_near": 30, "b_far": 40
    }
    assert prov["invariants"] == ["LS-1", "LS-2"]
    assert prov["far_edge_overlap_histogram"]
    import hashlib

    for name, digest in prov["sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
        assert (out / f"{name}.sha256").read_text().split()[0] == digest
    for rec in read_jsonl(out / "e_lure.jsonl"):
        assert not check_quartet(rec)


def test_shipped_manifests_match_the_recorded_seed():
    """The artifact in manifests/ must be the one this code produces from its seed."""
    import hashlib

    man = ROOT / "manifests"
    prov_path = man / "stimuli_provenance.json"
    if not prov_path.exists():
        pytest.skip("manifests/ not built yet; run `python -m lurestar.generate`")
    prov = json.loads(prov_path.read_text())
    for name, digest in prov["sha256"].items():
        assert hashlib.sha256((man / name).read_bytes()).hexdigest() == digest, name
        assert (man / f"{name}.sha256").read_text().split()[0] == digest

    shipped = read_jsonl(man / "e_lure.jsonl")
    assert len(shipped) == prov["counts"]["e_lure_quartets"]
    fresh = build_e_lure(prov["master_seed"], 200, QuartetConfig(), workers=1)
    assert shipped[:200] == fresh
    for rec in shipped[::7]:
        assert not check_quartet(rec)
