"""Prospective D40 one-shot expansion of the H3 middle candidate pool."""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from . import h3_precompute as D39
from .generate import SWAP_DEPTHS, graph_from_line, suffix_swap
from .validate import TrainingIndex, canonical_key_from_line, parse_line, validate_line


SCHEMA = "nextlat_forgetting/h3_mid_expansion/1"
LOSS_SCHEMA = D39.LOSS_TABLE_SCHEMA
MASTER_SEED = 2_026_082_402
RNG_NAMESPACE = 40
NEAR_COUNT = 5_000
ORIGINAL_PER_NEAR = 3
NEW_PER_STRATUM = 9
STRATA = (1, 2, 3)
TOTAL_PER_NEAR = 30
NEW_COUNT = 135_000
EXPANDED_COUNT = 150_000
COMBINED_LOSS_COUNT = 188_000
D39_LOSS_SHA256 = "a562057ead0852cb2a5dd5e68f3e50b34d9f299e1cabb707b6a269f37bbc7f13"
D39_MID_SHA256 = "df4a1a18ba4f5b2eb18e13a9dfb69e7c08b9d952044ea8392eead996123ecf8f"
D39_SCORER_SHA256 = "f907f00eda179c23d261a78e05efc43be33084240bb3ae691a4e29bf3a2b0954"
D40_DECISION_SHA256 = "b471988d1caca2cc9d40e44cf5c714028b9d2c8f63b4709a2b81608e767702ab"
MID_LOSS_CALIPER = 0.1


class ExpansionRefused(D39.PrecomputeRefused):
    """D40 generation, scoring, combination, or selection must stop."""


def _rows(path: pathlib.Path) -> list[dict[str, Any]]:
    return D39._read_jsonl(path)


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return D39._jsonl_payload(rows)


def _receipt(path: pathlib.Path, label: str) -> tuple[dict[str, Any], str]:
    digest = D39.verify_sidecar(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExpansionRefused(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ExpansionRefused(f"{label} must be a JSON object")
    return value, digest


def _identities(rows: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
    return D39._identity_sets(rows)


def _candidate(base, *, item: int, rewires: int, slot: int, attempt: int):
    rng = np.random.default_rng([MASTER_SEED, RNG_NAMESPACE, item, rewires, slot, attempt])
    graph = base
    for _ in range(rewires):
        arms = [int(value) for value in rng.choice(5, size=2, replace=False)]
        graph = suffix_swap(graph, arms[0], arms[1], int(rng.choice(SWAP_DEPTHS)))
    return graph


def require_exact_strata(
    rows: Sequence[Mapping[str, Any]], near_shas: Sequence[str], *, per_class: int = 10
) -> None:
    """Mutation-resistant count/identity gate for the frozen balanced construction."""
    expected_near = set(near_shas)
    counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    prompts, graphs = [], []
    for index, row in enumerate(rows):
        near_sha, rewires = row.get("paired_near_prompt_sha256"), row.get("rewire_count")
        if near_sha not in expected_near or rewires not in STRATA:
            raise ExpansionRefused(f"expanded row {index} has invalid near/rewire stratum")
        counts[str(near_sha)][int(rewires)] += 1
        prompts.append(str(row.get("prompt_sha256")))
        graphs.append(str(row.get("graph_key")))
    required = {rewires: per_class for rewires in STRATA}
    if set(counts) != expected_near or any(dict(counts[sha]) != required for sha in expected_near):
        raise ExpansionRefused("expanded pool is not exactly balanced across 1/2/3 rewires")
    if len(prompts) != len(set(prompts)) or len(graphs) != len(set(graphs)):
        raise ExpansionRefused("expanded pool reuses a prompt or canonical graph identity")


def generate(
    *, root: pathlib.Path, output_dir: pathlib.Path,
) -> dict[str, Any]:
    decision = root / "docs/DECISION_D40_h3_overlap_expansion.md"
    if D39.sha256_file(decision) != D40_DECISION_SHA256:
        raise ExpansionRefused("D40 decision must be frozen before candidate generation")
    near_path = root / "manifests/b_near.jsonl"
    original_path = root / "manifests/h3_precompute/b_mid_candidates.jsonl"
    if D39.verify_sidecar(original_path) != D39_MID_SHA256:
        raise ExpansionRefused("the immutable D39 middle pool changed")
    near, original = _rows(near_path), _rows(original_path)
    if len(near) != NEAR_COUNT or len(original) != NEAR_COUNT * ORIGINAL_PER_NEAR:
        raise ExpansionRefused("D39 near/middle counts changed")
    by_near: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in original:
        near_sha = str(row.get("paired_near_prompt_sha256"))
        rewires = row.get("rewire_count")
        if rewires not in STRATA or rewires in by_near[near_sha]:
            raise ExpansionRefused("D39 pool is not exactly one original per rewire class")
        by_near[near_sha][int(rewires)] = row

    frozen_paths = [
        root / "manifests/b_near.jsonl", root / "manifests/b_far.jsonl",
        root / "manifests/a_pair.jsonl", root / "manifests/e_lure.jsonl",
        *(root / "manifests/h3_precompute" / f"acquisition_{branch}_candidates.jsonl"
          for branch in ("near", "mid", "far")),
    ]
    frozen_rows = [row for path in frozen_paths for row in _rows(path)] + original
    prompts, graphs = _identities(frozen_rows)
    training_path = root / "data/stargraph/graph_5_5_sample_200000.txt"
    training = TrainingIndex.build(training_path)
    prompts |= training.prompt_hashes
    graphs |= training.graph_keys

    expanded: list[dict[str, Any]] = []
    new: list[dict[str, Any]] = []
    for item_index, near_row in enumerate(near):
        near_sha = str(near_row["prompt_sha256"])
        if set(by_near[near_sha]) != set(STRATA):
            raise ExpansionRefused(f"near row {item_index} lacks its three D39 candidates")
        base = graph_from_line(str(near_row["line"]))
        near_parsed = parse_line(str(near_row["line"]))
        near_path_length = len(validate_line(str(near_row["line"])).path)
        for rewires in STRATA:
            # Retain the exact parsed D39 record, not a reconstructed equivalent.
            expanded.append(by_near[near_sha][rewires])
            for slot in range(NEW_PER_STRATUM):
                accepted = None
                for attempt in range(4096):
                    graph = _candidate(
                        base, item=item_index, rewires=rewires, slot=slot, attempt=attempt
                    )
                    line = graph.serialize()
                    try:
                        solved = validate_line(line)
                    except Exception:
                        continue
                    parsed = parse_line(line)
                    psha, gsha = D39.prompt_sha(line), canonical_key_from_line(line)
                    if (
                        parsed.source != near_parsed.source or parsed.goal != near_parsed.goal
                        or len(solved.path) != near_path_length
                        or psha in prompts or gsha in graphs
                    ):
                        continue
                    accepted = {
                        "schema": SCHEMA, "pool": "B_mid",
                        "item_id": f"d40:{item_index}:{rewires}:{slot}",
                        "paired_near_prompt_sha256": near_sha,
                        "line": line, "prompt_sha256": psha, "graph_key": gsha,
                        "solver_verified": True,
                        "normalized_edge_disagreement": D39.structural_distance(
                            str(near_row["line"]), line
                        ),
                        "construction": "d40_deterministic_sequential_suffix_swaps",
                        "rewire_count": rewires, "stratum_slot": slot,
                        "accepted_attempt": attempt, "master_seed": MASTER_SEED,
                        "rng_namespace": RNG_NAMESPACE,
                    }
                    break
                if accepted is None:
                    raise ExpansionRefused(
                        f"could not generate D40 item={item_index} stratum={rewires} slot={slot}"
                    )
                prompts.add(accepted["prompt_sha256"])
                graphs.add(accepted["graph_key"])
                expanded.append(accepted)
                new.append(accepted)
    if len(new) != NEW_COUNT or len(expanded) != EXPANDED_COUNT:
        raise AssertionError("D40 exact candidate count failure")
    # Counts, not distance diversity, are frozen. Verify exact 10/class per near.
    require_exact_strata(expanded, [str(row["prompt_sha256"]) for row in near])

    outputs = {
        "b_mid_new_135000.jsonl": D39.create_or_verify(
            output_dir / "b_mid_new_135000.jsonl", _jsonl(new)
        ),
        "b_mid_expanded_150000.jsonl": D39.create_or_verify(
            output_dir / "b_mid_expanded_150000.jsonl", _jsonl(expanded)
        ),
    }
    receipt = {
        "schema": SCHEMA, "status": "D40_CANDIDATES_FROZEN_BEFORE_SCORING",
        "confirmatory_inputs_inspected": False, "confirmatory_results_inspected": False,
        "paid_compute_used": False, "master_seed": MASTER_SEED,
        "rng_namespace": RNG_NAMESPACE,
        "counts": {"near": NEAR_COUNT, "retained": 15_000, "new": NEW_COUNT,
                   "expanded": EXPANDED_COUNT, "per_near": TOTAL_PER_NEAR,
                   "per_rewire_class_per_near": 10},
        "inputs": {
            "d40_decision_sha256": D40_DECISION_SHA256,
            "d39_mid_sha256": D39_MID_SHA256,
            "near_sha256": D39.verify_sidecar(near_path),
            "training_sha256": D39.sha256_file(training_path),
        },
        "outputs": outputs,
        "checks": ["new_rng_namespace", "independent_solver", "source_goal_path_match",
                   "global_prompt_graph_uniqueness", "frozen_domain_disjointness",
                   "exact_10_per_rewire_class", "distance_diversity_not_required"],
    }
    D39.create_or_verify(output_dir / "generation_receipt.json", D39.canonical_json(receipt))
    return receipt


def combine_losses(
    *, original_loss: pathlib.Path, expansion_loss: pathlib.Path, expanded_manifest: pathlib.Path,
    scoring_receipt: pathlib.Path, durable_state: pathlib.Path,
    generation_receipt: pathlib.Path, generation_domain_receipt: pathlib.Path,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    if D39.verify_sidecar(original_loss) != D39_LOSS_SHA256:
        raise ExpansionRefused("frozen D39 loss table changed")
    original_rows = _rows(original_loss)
    expansion_rows = _rows(expansion_loss)
    expanded = _rows(expanded_manifest)
    if len(original_rows) != 53_000 or len(expansion_rows) != NEW_COUNT or len(expanded) != EXPANDED_COUNT:
        raise ExpansionRefused("D40 loss/manifest counts are incomplete")
    original_ids = {str(row["prompt_sha256"]) for row in original_rows}
    expansion_ids = {str(row["prompt_sha256"]) for row in expansion_rows}
    expected_new = {
        str(row["prompt_sha256"]) for row in expanded if row.get("schema") == SCHEMA
    }
    if len(original_ids) != 53_000 or len(expansion_ids) != NEW_COUNT or original_ids & expansion_ids:
        raise ExpansionRefused("D40 loss identities are duplicate or overlapping")
    if expansion_ids != expected_new:
        raise ExpansionRefused("expanded loss table does not cover exactly the 135,000 new rows")
    expansion_sha = D39.verify_sidecar(expansion_loss)
    expanded_sha = D39.verify_sidecar(expanded_manifest)
    scoring, scoring_sha = _receipt(scoring_receipt, "D40 scoring receipt")
    durable, durable_sha = _receipt(durable_state, "D40 durable state")
    generation, generation_sha = _receipt(generation_receipt, "D40 generation receipt")
    domain, domain_sha = _receipt(generation_domain_receipt, "D40 exclusion-domain receipt")
    new_sha = generation.get("outputs", {}).get("b_mid_new_135000.jsonl")
    if (
        generation.get("status") != "D40_CANDIDATES_FROZEN_BEFORE_SCORING"
        or generation.get("outputs", {}).get("b_mid_expanded_150000.jsonl") != expanded_sha
        or domain.get("status") != "D40_EXCLUSION_DOMAINS_BOUND_BEFORE_SCORING"
        or domain.get("generation_receipt_sha256") != generation_sha
        or domain.get("expanded_manifest_sha256") != expanded_sha
        or domain.get("new_manifest_sha256") != new_sha
    ):
        raise ExpansionRefused("D40 generation/domain lineage is incomplete")
    if (
        scoring.get("status") != "COMPLETE_D40_EXPANSION_SCORING"
        or scoring.get("row_count") != NEW_COUNT
        or scoring.get("loss_table_sha256") != expansion_sha
        or scoring.get("new_manifest_sha256") != new_sha
        or scoring.get("generation_receipt_sha256") != generation_sha
        or scoring.get("generation_domain_receipt_sha256") != domain_sha
        or scoring.get("scientific_scorer_sha256") != D39_SCORER_SHA256
        or scoring.get("checkpoint_sha256") != D39.PILOT_CHECKPOINT_SHA256
        or scoring.get("config_sha256") != D39.PILOT_CONFIG_SHA256
        or scoring.get("tokenizer_sha256") != D39.sha256_file(
            pathlib.Path(__file__).resolve().parents[2] / "upstream/NextLat/data/stargraph.py"
        )
        or scoring.get("confirmatory_inputs_inspected") is not False
        or scoring.get("confirmatory_results_inspected") is not False
    ):
        raise ExpansionRefused("D40 scoring receipt does not bind the exact expansion")
    if (
        durable.get("complete") is not True or durable.get("row_count") != NEW_COUNT
        or durable.get("job_sha256") != scoring.get("job_sha256")
        or durable.get("scientific_scorer_sha256") != D39_SCORER_SHA256
        or durable.get("new_manifest_sha256") != new_sha
        or durable.get("loss_table", {}).get("sha256") != expansion_sha
        or durable.get("scoring_receipt", {}).get("sha256") != scoring_sha
    ):
        raise ExpansionRefused("D40 loss table lacks a complete generation-bound durable state")
    combined = original_rows + expansion_rows
    combined_sha = D39.create_or_verify(
        output_dir / "combined_pilot_losses_188000.jsonl", _jsonl(combined)
    )
    receipt = {
        "schema": SCHEMA, "status": "D40_COMBINED_LOSS_TABLE_COMPLETE",
        "logical_order": ["frozen_d39_53000", "new_d40_mid_135000"],
        "row_count": COMBINED_LOSS_COUNT, "unique_identity_count": COMBINED_LOSS_COUNT,
        "sources": {"d39_loss_sha256": D39_LOSS_SHA256,
                    "d40_loss_sha256": expansion_sha,
                    "expanded_manifest_sha256": expanded_sha,
                    "generation_receipt_sha256": generation_sha,
                    "generation_domain_receipt_sha256": domain_sha,
                    "scoring_receipt_sha256": scoring_sha,
                    "durable_state_sha256": durable_sha,
                    "score_job_sha256": scoring["job_sha256"]},
        "output_sha256": combined_sha,
        "scientific_scorer_sha256": D39_SCORER_SHA256,
        "checkpoint_sha256": D39.PILOT_CHECKPOINT_SHA256,
        "config_sha256": D39.PILOT_CONFIG_SHA256,
        "tokenizer_sha256": D39.sha256_file(
            pathlib.Path(__file__).resolve().parents[2] / "upstream/NextLat/data/stargraph.py"
        ),
    }
    D39.create_or_verify(output_dir / "combined_loss_receipt.json", D39.canonical_json(receipt))
    return receipt


def select_mid(
    *, root: pathlib.Path, expanded_manifest: pathlib.Path, combined_loss: pathlib.Path,
    combined_receipt: pathlib.Path, output_dir: pathlib.Path,
) -> dict[str, Any]:
    near_path = root / "manifests/b_near.jsonl"
    near, mid = _rows(near_path), _rows(expanded_manifest)
    if len(near) != NEAR_COUNT or len(mid) != EXPANDED_COUNT:
        raise ExpansionRefused("D40 selection requires exact 5,000/150,000 inputs")
    losses, loss_sha = D39.load_loss_table(combined_loss)
    receipt, receipt_sha = _receipt(combined_receipt, "D40 combined-loss receipt")
    if (
        len(losses) != COMBINED_LOSS_COUNT
        or receipt.get("status") != "D40_COMBINED_LOSS_TABLE_COMPLETE"
        or receipt.get("row_count") != COMBINED_LOSS_COUNT
        or receipt.get("unique_identity_count") != COMBINED_LOSS_COUNT
        or receipt.get("output_sha256") != loss_sha
        or receipt.get("sources", {}).get("expanded_manifest_sha256")
        != D39.verify_sidecar(expanded_manifest)
        or receipt.get("sources", {}).get("d39_loss_sha256") != D39_LOSS_SHA256
        or receipt.get("scientific_scorer_sha256") != D39_SCORER_SHA256
    ):
        raise ExpansionRefused("D40 combined table/receipt is incomplete or has wrong lineage")
    for field in ("generation_receipt_sha256", "generation_domain_receipt_sha256",
                  "scoring_receipt_sha256", "durable_state_sha256", "score_job_sha256"):
        value = receipt.get("sources", {}).get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise ExpansionRefused(f"D40 combined receipt lacks {field}")
    expanded_ids = {str(row.get("prompt_sha256")) for row in mid}
    if len(expanded_ids) != EXPANDED_COUNT or not expanded_ids.issubset(losses):
        raise ExpansionRefused("D40 combined table does not cover the exact expanded population")
    near_rank, mid_rank = D39._rank(near, losses), D39._rank(mid, losses)
    near_by_sha = {str(row["prompt_sha256"]): row for row in near}
    eligible: dict[str, list[tuple[float, str]]] = {sha: [] for sha in near_by_sha}
    table, distances = [], []
    for row in mid:
        msha, nsha = str(row["prompt_sha256"]), str(row["paired_near_prompt_sha256"])
        if nsha not in near_by_sha:
            raise ExpansionRefused("expanded candidate points outside B_near")
        line = str(row["line"])
        validate_line(line)
        if D39.prompt_sha(line) != msha or canonical_key_from_line(line) != row.get("graph_key"):
            raise ExpansionRefused("expanded candidate identity does not hash its serialized line")
        distance = D39.structural_distance(str(near_by_sha[nsha]["line"]), str(row["line"]))
        difference = abs(losses[nsha].loss - losses[msha].loss)
        path_match = len(parse_line(str(near_by_sha[nsha]["line"])).answer) == len(
            parse_line(str(row["line"])).answer
        )
        ok = near_rank[nsha][2] == mid_rank[msha][2] and difference <= MID_LOSS_CALIPER + 1e-12 and path_match
        table.append({"mid_prompt_sha256": msha, "near_loss_decile": near_rank[nsha][2],
                      "mid_loss_decile": mid_rank[msha][2],
                      "pilot_loss_absolute_difference": difference,
                      "normalized_edge_disagreement": distance, "eligible": ok})
        if ok:
            eligible[nsha].append((distance, msha)); distances.append(distance)
    missing = sorted(sha for sha, values in eligible.items() if not values)
    if missing:
        block = {
            "schema": SCHEMA, "status": "PERMANENT_H3_BLOCK",
            "reason": "D40_ONE_SHOT_EXPANSION_REMAINS_INFEASIBLE",
            "unmatched_count": len(missing), "unmatched_identity_sha256": hashlib.sha256(
                "".join(value + "\n" for value in missing).encode()
            ).hexdigest(),
            "no_further_amendments_permitted": True,
            "forbidden": ["candidate_expansion", "caliper_change", "weighting",
                          "unmatched_restriction", "pilot_substitution", "matching_amendment"],
            "combined_loss_sha256": loss_sha,
            "expanded_manifest_sha256": D39.verify_sidecar(expanded_manifest),
        }
        D39.create_or_verify(output_dir / "PERMANENT_H3_BLOCK.json", D39.canonical_json(block))
        raise ExpansionRefused(f"PERMANENT H3 BLOCK: {len(missing)} near items remain unmatched")
    median = float(np.median(np.asarray(distances, dtype=np.float64)))
    selection, used = [], set()
    for row in near:
        nsha = str(row["prompt_sha256"])
        distance, msha = min(eligible[nsha], key=lambda value: (abs(value[0] - median), value[1]))
        if msha in used:
            raise ExpansionRefused("D40 selected a middle candidate twice")
        used.add(msha)
        selection.append({"near_prompt_sha256": nsha, "mid_prompt_sha256": msha,
                          "normalized_edge_disagreement": distance})
    artifact = {
        "schema_version": 1, "purpose": "h3_mid_structural_distance_match_d40",
        "selection_method": "d40_one_shot_expanded_structural_median_with_unchanged_pilot_caliper",
        "near_bank_sha256": D39.verify_sidecar(near_path),
        "candidate_bank_sha256": D39.verify_sidecar(expanded_manifest),
        "distance_quantile": 0.5, "pilot_loss_caliper": MID_LOSS_CALIPER,
        "eligible_median_normalized_edge_disagreement": median,
        "tie_break": "candidate_prompt_sha256_ascending",
        "permanent_block_if_any_unmatched": True, "no_further_amendments_permitted": True,
        "pilot": D39._pilot_provenance(
            D39.verify_sidecar(root / "manifests/h3_precompute/pilot_freeze.json"),
            loss_sha, D39.sha256_file(pathlib.Path(__file__)),
        ),
        "combined_loss_table_sha256": loss_sha,
        "combined_loss_receipt_sha256": receipt_sha,
        "candidate_table": table, "selection": selection,
    }
    artifact_sha = D39.create_or_verify(output_dir / "mid_selection_d40.json", D39.canonical_json(artifact))
    receipt = {"schema": SCHEMA, "status": "D40_ALL_5000_MATCHED",
               "matched_count": NEAR_COUNT, "selected_unique_count": len(used),
               "eligible_candidate_count": len(distances), "global_distance_median": median,
               "selection_sha256": artifact_sha, "combined_loss_sha256": loss_sha}
    D39.create_or_verify(output_dir / "selection_receipt.json", D39.canonical_json(receipt))
    return receipt


def plan(root: pathlib.Path, output_dir: pathlib.Path) -> dict[str, Any]:
    return {"schema": SCHEMA, "mode": "READ_ONLY_PLAN", "master_seed": MASTER_SEED,
            "rng_namespace": RNG_NAMESPACE,
            "counts": {"retained": 15_000, "new": NEW_COUNT, "expanded": EXPANDED_COUNT,
                       "new_per_rewire_class_per_near": NEW_PER_STRATUM},
            "would_write": [str(output_dir / "b_mid_new_135000.jsonl"),
                            str(output_dir / "b_mid_expanded_150000.jsonl"),
                            str(output_dir / "generation_receipt.json")],
            "gpu_launched": False}
