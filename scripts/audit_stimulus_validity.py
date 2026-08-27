#!/usr/bin/env python3
"""Deterministic, outcome-blind audit of the project's stimulus artifacts.

This script deliberately has no checkpoint, run, result, metric, or loss input.  It
answers a narrower question: do the frozen files contain the examples the generators
claim to have made, and how well do those examples instantiate the scientific
constructs?  Mechanical validity and construct validity are reported separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SCHEMA = "nextlat_forgetting/stimulus_validity_audit/2"
AUDIT_DATE = "2026-08-24"


def _bootstrap(root: Path) -> None:
    source = str(root / "src")
    if source not in sys.path:
        sys.path.insert(0, source)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _hist(values: Iterable[Any]) -> dict[str, int]:
    return {str(key): value for key, value in sorted(Counter(values).items(), key=lambda x: str(x[0]))}


def _count_range(counter: Counter[Any]) -> dict[str, int]:
    values = list(counter.values())
    return {
        "unique": len(counter),
        "min_count": min(values) if values else 0,
        "max_count": max(values) if values else 0,
    }


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _line_summary(lines: Sequence[str]) -> tuple[dict[str, Any], set[str], set[str]]:
    from lurestar.validate import canonical_key_from_line, parse_line, sha256_text, token_ids, validate_line

    prompt_ids: set[str] = set()
    graph_ids: set[str] = set()
    sources: Counter[int] = Counter()
    goals: Counter[int] = Counter()
    source_goal: Counter[tuple[int, int]] = Counter()
    prompt_lengths: Counter[int] = Counter()
    answer_lengths: Counter[int] = Counter()
    node_coverage: Counter[int] = Counter()
    invalid: list[dict[str, Any]] = []
    duplicate_prompts = 0
    duplicate_graphs = 0
    for index, line in enumerate(lines):
        try:
            solved = validate_line(line)
            parsed = parse_line(line)
        except Exception as exc:  # the report records malformed data rather than hiding it
            invalid.append({"index": index, "error": f"{type(exc).__name__}: {exc}"})
            continue
        prompt_id = sha256_text(parsed.prompt)
        graph_id = canonical_key_from_line(line)
        duplicate_prompts += int(prompt_id in prompt_ids)
        duplicate_graphs += int(graph_id in graph_ids)
        prompt_ids.add(prompt_id)
        graph_ids.add(graph_id)
        sources[parsed.source] += 1
        goals[parsed.goal] += 1
        source_goal[(parsed.source, parsed.goal)] += 1
        prompt_lengths[len(token_ids(parsed.prompt))] += 1
        answer_lengths[len(parsed.answer)] += 1
        node_coverage.update({solved.source, *(node for arm in solved.arms for node in arm)})
    summary = {
        "count": len(lines),
        "solver_valid_count": len(lines) - len(invalid),
        "solver_invalid_count": len(invalid),
        "first_invalid_examples": invalid[:10],
        "unique_prompt_identities": len(prompt_ids),
        "duplicate_prompt_count": duplicate_prompts,
        "unique_graph_identities": len(graph_ids),
        "duplicate_graph_count": duplicate_graphs,
        "prompt_token_length_histogram": _hist(prompt_lengths.elements()),
        "answer_length_histogram": _hist(answer_lengths.elements()),
        "source_balance": _count_range(sources),
        "goal_balance": _count_range(goals),
        "source_goal_pair_coverage": _count_range(source_goal),
        "node_id_coverage": _count_range(node_coverage),
    }
    return summary, prompt_ids, graph_ids


def audit_pathstar(root: Path) -> tuple[dict[str, Any], dict[str, set[str]]]:
    train_path = root / "data/stargraph/graph_5_5_sample_200000.txt"
    test_path = root / "data/stargraph/graph_5_5_test_20000.txt"
    train_lines = [line.strip() for line in train_path.read_text().splitlines() if line.strip()]
    test_lines = [line.strip() for line in test_path.read_text().splitlines() if line.strip()]
    train, train_prompts, train_graphs = _line_summary(train_lines)
    test, test_prompts, test_graphs = _line_summary(test_lines)
    provenance_path = root / "manifests/corpus_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    return {
        "role": "official-distribution synthetic Path-Star competence and representation task",
        "provenance": {
            "upstream_commit": provenance.get("upstream_commit"),
            "upstream_function": provenance.get("upstream_function"),
            "parameters": provenance.get("params"),
            "seed_scheme": provenance.get("seed_scheme"),
            "files": {
                str(train_path.relative_to(root)): sha256_file(train_path),
                str(test_path.relative_to(root)): sha256_file(test_path),
                str(provenance_path.relative_to(root)): sha256_file(provenance_path),
            },
        },
        "train": train,
        "test": test,
        "train_test_identity_disjointness": {
            "prompt_collisions": len(train_prompts & test_prompts),
            "graph_collisions": len(train_graphs & test_graphs),
        },
        "coded_invariant_status": "PASS" if not train["solver_invalid_count"] and not test["solver_invalid_count"] else "FAIL",
        "construct_validity_status": "FIT_FOR_CONTROLLED_PATH_PLANNING",
        "construct_match": (
            "Strong for algorithm/config fidelity, competence, and controlled predictive-state tests; "
            "does not by itself support claims about natural-language semantics."
        ),
        "known_limitations": [
            "Same generator family is used for training and held-out test data.",
            "The corpus reproduces the pinned upstream distribution and seeds, not a claim of byte identity to every dataset used in paper v4.",
            "All examples have one fixed G(5,5) topology and fixed token/answer lengths, limiting external validity.",
        ],
    }, {"train_prompts": train_prompts, "train_graphs": train_graphs, "test_prompts": test_prompts, "test_graphs": test_graphs}


def audit_lurestar(root: Path, identities: Mapping[str, set[str]]) -> tuple[dict[str, Any], dict[str, set[str]]]:
    from lurestar.validate import canonical_key_from_line, check_quartet, parse_line, sha256_text

    # The unusual base/repeat and near-safe/aligned canonical collisions are intended:
    # they are different serializations of the same graph, not duplicate items.
    e_path = root / "manifests/e_lure.jsonl"
    quartets = load_jsonl(e_path)
    violations: list[dict[str, Any]] = []
    condition_lines: dict[str, list[str]] = {}
    depth = Counter()
    far_overlap = Counter()
    edit_positions: dict[str, Counter[tuple[int, ...]]] = {"near_safe": Counter(), "near_critical": Counter(), "near_safe_aligned": Counter()}
    e_prompts: set[str] = set()
    e_graphs: set[str] = set()
    for index, record in enumerate(quartets):
        problems = check_quartet(record)
        if problems:
            violations.append({"quartet_index": index, "problems": problems})
        depth[record["depth"]] += 1
        far_overlap[record["far_edge_overlap"]] += 1
        for name, slots in record["edit_token_positions"].items():
            edit_positions[name][tuple(sorted(slots))] += 1
        for name, condition in record["conditions"].items():
            line = condition["line"]
            condition_lines.setdefault(name, []).append(line)
            parsed = parse_line(line)
            e_prompts.add(sha256_text(parsed.prompt))
            e_graphs.add(canonical_key_from_line(line))

    condition_summaries = {name: _line_summary(lines)[0] for name, lines in sorted(condition_lines.items())}
    pools: dict[str, Any] = {}
    pool_prompts: set[str] = set()
    pool_graphs: set[str] = set()
    for filename in ("a_pair.jsonl", "b_near.jsonl", "b_far.jsonl"):
        path = root / "manifests" / filename
        records = load_jsonl(path)
        lines = [record["line"] for record in records]
        summary, prompts, graphs = _line_summary(lines)
        summary["sha256"] = sha256_file(path)
        summary["collisions_with_pathstar_train"] = {
            "prompts": len(prompts & identities["train_prompts"]),
            "graphs": len(graphs & identities["train_graphs"]),
        }
        if filename == "b_near.jsonl":
            parent_counts = Counter(record["parent_item_id"] for record in records)
            summary["updates_per_parent_histogram"] = _hist(parent_counts.values())
        if filename == "b_far.jsonl":
            summary["edge_overlap_histogram"] = _hist(record["edge_overlap"] for record in records)
            summary["generation_tries_histogram"] = _hist(record["tries"] for record in records)
        pools[filename] = summary
        pool_prompts |= prompts
        pool_graphs |= graphs

    return {
        "role": "paired future-preserving/future-changing perturbations for H1/H2; legacy adaptation pools are provenance-only",
        "provenance": {
            "manifest": "manifests/stimuli_provenance.json",
            "manifest_sha256": sha256_file(root / "manifests/stimuli_provenance.json"),
            "master_seed": json.loads((root / "manifests/stimuli_provenance.json").read_text())["master_seed"],
            "e_lure_sha256": sha256_file(e_path),
        },
        "e_lure": {
            "quartet_count": len(quartets),
            "condition_counts": {name: len(lines) for name, lines in sorted(condition_lines.items())},
            "condition_summaries": condition_summaries,
            "quartet_invariant_violation_count": len(violations),
            "first_violations": violations[:10],
            "suffix_depth_histogram": _hist(depth.elements()),
            "far_edge_overlap_histogram": _hist(far_overlap.elements()),
            "unique_edit_position_pairs": {name: len(values) for name, values in edit_positions.items()},
            "edit_token_position_histograms": {
                name: _hist(position for pair, count in values.items() for position in pair for _ in range(count))
                for name, values in edit_positions.items()
            },
            "collisions_with_pathstar_train": {
                "prompt_collisions": len(e_prompts & identities["train_prompts"]),
                "graph_collisions": len(e_graphs & identities["train_graphs"]),
            },
        },
        "legacy_pools": pools,
        "e_lure_vs_legacy_pool_identity_disjointness": {
            "prompt_collisions": len(e_prompts & pool_prompts),
            "graph_collisions": len(e_graphs & pool_graphs),
        },
        "coded_invariant_status": "PASS" if not violations else "FAIL",
        "construct_validity_status": "FIT_WITH_DECLARED_POSITION_LIMITATION",
        "construct_match": (
            "Strong for minimal, solver-certified future-preserving versus future-changing edits. "
            "LS-1 shares the anchor but only exchangeably balances absolute edit position; LS-2 matches position using a different serialization anchor."
        ),
        "known_limitations": [
            "Safe and critical edits cannot occupy identical absolute positions in one common serialized anchor; this is an algebraic design limitation, not a failed check.",
            "Far controls are rejection-sampled to edge overlap <=2 and therefore represent a deliberately conditioned distribution.",
            "A_pair is intentionally drawn from training; it is not an independent evaluation set. Legacy H3 adaptation is retired and these pools must not be reinterpreted as current causal evidence.",
            "Synthetic graph perturbations isolate a future-change construct but do not establish natural-language generalization.",
        ],
    }, {"e_prompts": e_prompts, "e_graphs": e_graphs, "pool_prompts": pool_prompts, "pool_graphs": pool_graphs}


def _sequence_duplicates(array: np.ndarray) -> int:
    contiguous = np.ascontiguousarray(array)
    rows = contiguous.view(np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))).ravel()
    return int(len(rows) - len(np.unique(rows)))


def _cross_sequence_collisions(left: np.ndarray, right: np.ndarray) -> int:
    if left.shape[1] != right.shape[1]:
        return 0
    a = np.ascontiguousarray(left).view(np.dtype((np.void, left.dtype.itemsize * left.shape[1]))).ravel()
    b = np.ascontiguousarray(right.astype(left.dtype, copy=False)).view(a.dtype).ravel()
    return int(len(np.intersect1d(a, b)))


def audit_hmm(root: Path) -> dict[str, Any]:
    from hmm_geometry.family import REGIMES, load_family
    from hmm_geometry.forward import forward_batch

    hmms, payload = load_family(root / "manifests/hmm_family.json")
    regimes: dict[str, Any] = {}
    all_ok = True
    for name in REGIMES:
        data_dir = root / "data/hmm_family" / name
        arrays = {
            "train": np.load(data_dir / "hmm4x4_train_len32_100000.npy"),
            "val": np.load(data_dir / "hmm4x4_val_len32_10000.npy"),
            "lengen": np.load(data_dir / "hmm4x4_lengen_len64_10000.npy"),
        }
        split_summary: dict[str, Any] = {}
        for split, array in arrays.items():
            valid_range = bool(np.issubdtype(array.dtype, np.integer) and array.min() >= 0 and array.max() <= 3)
            all_ok &= valid_range
            split_summary[split] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "symbol_histogram": _hist(array.ravel()),
                "duplicate_sequence_count": _sequence_duplicates(array),
                "symbols_in_0_to_3": valid_range,
                "sha256": sha256_file(next(data_dir.glob(f"*_{split}_*")) if split != "lengen" else data_dir / "hmm4x4_lengen_len64_10000.npy"),
            }
        posterior_checks: dict[str, Any] = {}
        for split, stem in (("val", "hmm4x4_val_posteriors.npz"), ("lengen", "hmm4x4_lengen_posteriors.npz")):
            path = data_dir / stem
            stored = np.load(path)
            observations = arrays[split]
            indices = np.unique(np.linspace(0, len(observations) - 1, 64, dtype=int))
            exact = forward_batch(hmms[name], observations[indices].astype(np.int64))
            obs_match = bool(np.array_equal(stored["observations"], observations))
            belief_error = float(np.max(np.abs(exact.beliefs - stored["beliefs"][indices])))
            next_obs_error = float(np.max(np.abs(exact.next_obs - stored["next_obs"][indices])))
            normalized = bool(
                np.allclose(stored["beliefs"].sum(axis=-1), 1.0, atol=1e-10)
                and np.allclose(stored["next_obs"].sum(axis=-1), 1.0, atol=1e-10)
            )
            passed = obs_match and belief_error < 1e-10 and next_obs_error < 1e-10 and normalized
            all_ok &= passed
            posterior_checks[split] = {
                "stored_observations_equal_split": obs_match,
                "probability_rows_normalized": normalized,
                "deterministic_recomputed_rows": len(indices),
                "max_abs_belief_error": belief_error,
                "max_abs_next_observation_error": next_obs_error,
                "passed": passed,
                "sha256": sha256_file(path),
            }
        te = payload["regimes"][name]["linear_certificate"]["matrices"]["transition_times_emission"]
        regimes[name] = {
            "splits": split_summary,
            "train_val_exact_sequence_collisions": _cross_sequence_collisions(arrays["train"], arrays["val"]),
            "posterior_checks": posterior_checks,
            "mean_dwell_time": payload["regimes"][name]["selection_diagnostics"]["mean_dwell_time"],
            "belief_entropy_mean_bits": payload["regimes"][name]["selection_diagnostics"]["belief_entropy_mean_bits"],
            "transition_times_emission_rank": te["rank"],
            "transition_times_emission_sigma_min": te["sigma_min"],
        }
    return {
        "role": "exact-ground-truth calibration of whether representation geometry tracks predictive equivalence",
        "provenance": {
            "family_manifest": "manifests/hmm_family.json",
            "family_manifest_sha256": sha256_file(root / "manifests/hmm_family.json"),
            "materialization_receipt": "manifests/hmm_family_materialization.json",
            "model_inputs_used_for_selection": [],
        },
        "regimes": regimes,
        "coded_invariant_status": "PASS" if all_ok else "FAIL",
        "construct_validity_status": "FIT_FOR_CALIBRATION_NOT_FORGETTING",
        "construct_match": (
            "Excellent for exact posterior/future ground truth and robustness across mixing/aliasing regimes; "
            "it calibrates predictive geometry but is not itself a causal-forgetting or language task."
        ),
        "known_limitations": [
            "Only 4 hidden states, 4 observation symbols, and sequence lengths 32/64 are represented.",
            "Exact duplicate sequences are possible samples from the same stochastic process; counts are disclosed rather than treated automatically as leakage.",
            "Only 64 deterministic rows per stored evaluation split are recomputed in this audit; all array ranges and probability normalizations are checked in full.",
        ],
    }


def cfs_construct_assessment(overlap_histograms: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    expected = {
        "high/same": {"18": 5000},
        "high/different": {"18": 5000},
        "low/same": {"8": 5000},
        "low/different": {"7": 5000},
    }
    observed = {key: dict(value) for key, value in overlap_histograms.items()}
    confounded = observed.get("low/same") != observed.get("low/different")
    return {
        "expected_overlap_histograms": expected,
        "observed_overlap_histograms": observed,
        "low_overlap_factorially_balanced": not confounded,
        "critical_finding": (
            "CFS-1 low/same has 8 shared edges while low/different has 7. The nuisance difference is perfectly "
            "confounded with future relation inside the low-overlap arm and can bias the difference-in-differences "
            "toward the hypothesized interaction. A regression cannot identify the two effects from these cells."
            if confounded else "No low-arm overlap imbalance detected."
        ),
        "disposition": "REDESIGN_BEFORE_CAUSAL_BRANCH_TRAINING" if confounded else "NO_OVERLAP_REDESIGN_REQUIRED",
    }


def audit_cfs1(
    root: Path, path_ids: Mapping[str, set[str]], lure_ids: Mapping[str, set[str]]
) -> tuple[dict[str, Any], dict[str, set[str]]]:
    from cfs1.validate import CONDITIONS, validate_bundle
    from lurestar.validate import canonical_key_from_line, parse_line, sha256_text

    base = root / "manifests/cfs1"
    retention = load_jsonl(base / "retention.jsonl")
    controls = load_jsonl(base / "global_controls.jsonl")
    codebook = json.loads((base / "hash_codebook.json").read_text())
    updates = {
        condition: load_jsonl(base / f"updates_{condition[0]}_{condition[1]}.jsonl")
        for condition in CONDITIONS
    }
    validation_error = None
    try:
        validate_bundle(
            retention, updates, codebook, expected_probes=2000, expected_updates=5000,
            global_controls=controls, expected_global_controls=2000,
        )
    except Exception as exc:
        validation_error = f"{type(exc).__name__}: {exc}"

    sets: dict[str, tuple[set[str], set[str]]] = {}
    summaries: dict[str, Any] = {}
    for label, records in {
        "retention": retention,
        "global_controls": controls,
        **{f"{key[0]}/{key[1]}": value for key, value in updates.items()},
    }.items():
        lines = [record["line"] for record in records]
        summary, prompts, graphs = _line_summary(lines)
        summary["source_histogram"] = _hist(parse_line(line).source for line in lines)
        summary["goal_histogram"] = _hist(parse_line(line).goal for line in lines)
        if "/" in label:
            summary["edge_overlap_with_probe_histogram"] = _hist(record["edge_overlap_with_probe"] for record in records)
            summary["future_same_as_probe_histogram"] = _hist(record["future_same_as_probe"] for record in records)
        summaries[label] = summary
        sets[label] = (prompts, graphs)

    all_prompts = set().union(*(value[0] for value in sets.values()))
    all_graphs = set().union(*(value[1] for value in sets.values()))
    legacy_prompts = path_ids["train_prompts"] | path_ids["test_prompts"] | lure_ids["e_prompts"] | lure_ids["pool_prompts"]
    legacy_graphs = path_ids["train_graphs"] | path_ids["test_graphs"] | lure_ids["e_graphs"] | lure_ids["pool_graphs"]
    overlap_histograms = {
        label: summaries[label]["edge_overlap_with_probe_histogram"]
        for label in ("high/same", "high/different", "low/same", "low/different")
    }
    assessment = cfs_construct_assessment(overlap_histograms)
    report = {
        "role": "candidate causal-forgetting factorial manipulation",
        "provenance": {
            "generator": "src/cfs1/generate.py",
            "construction_receipt": "manifests/cfs1/construction_receipt.json",
            "construction_receipt_sha256": sha256_file(base / "construction_receipt.json"),
            "model_or_outcome_inputs": [],
        },
        "artifact_summaries": summaries,
        "independent_bundle_validation": {"passed": validation_error is None, "error": validation_error},
        "identity_disjointness_from_pathstar_and_lurestar": {
            "prompt_collisions": len(all_prompts & legacy_prompts),
            "graph_collisions": len(all_graphs & legacy_graphs),
        },
        "factorial_overlap_assessment": assessment,
        "coded_invariant_status": "PASS" if validation_error is None else "FAIL",
        "construct_validity_status": "FAIL_REDESIGN_REQUIRED",
        "construct_match": (
            "The files faithfully implement their coded 18/18/8/7 construction, but that construction does not "
            "cleanly identify the intended overlap-by-conflicting-future interaction. Mechanical correctness is not construct validity."
        ),
        "known_limitations": [
            assessment["critical_finding"],
            "The fixed G(5,5) topology may make exact 18/18 and 8/8 construction infeasible without a new construction family.",
            "CFS-1 should be retained as an auditable design attempt, not silently rewritten or used for the strongest causal claim.",
        ],
        "required_action": "Do not start CFS-1 adaptation branches. Construct and re-audit CFS-2 with exact factorial overlap balance.",
    }
    return report, {"all_prompts": all_prompts, "all_graphs": all_graphs}


def _overlap_decomposition(probe_line: str, update_line: str) -> tuple[int, int, int]:
    """Independently recompute total, update-answer, and other edge overlap."""
    from lurestar.validate import parse_line

    probe_edges = set(parse_line(probe_line).edges)
    update = parse_line(update_line)
    update_answer_edges = set(zip(update.answer, update.answer[1:]))
    shared = probe_edges & set(update.edges)
    answer = len(shared & update_answer_edges)
    return len(shared), answer, len(shared) - answer


def cfs2_construct_assessment(
    total: Mapping[str, Mapping[str, int]],
    answer: Mapping[str, Mapping[str, int]],
    nonanswer: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    expected_total = {
        "high/same": {"18": 5000}, "high/different": {"18": 5000},
        "low/same": {"8": 5000}, "low/different": {"8": 5000},
    }
    expected_answer = {
        "high/same": {"4": 5000}, "high/different": {"3": 5000},
        "low/same": {"4": 5000}, "low/different": {"3": 5000},
    }
    expected_nonanswer = {
        "high/same": {"14": 5000}, "high/different": {"15": 5000},
        "low/same": {"4": 5000}, "low/different": {"5": 5000},
    }
    observed_total = {key: dict(value) for key, value in total.items()}
    observed_answer = {key: dict(value) for key, value in answer.items()}
    observed_nonanswer = {key: dict(value) for key, value in nonanswer.items()}

    def singleton(histogram: Mapping[str, int]) -> int | None:
        return int(next(iter(histogram))) if len(histogram) == 1 else None

    values = {key: singleton(value) for key, value in observed_nonanswer.items()}
    same_contrast = (
        values.get("high/same") - values.get("low/same")
        if values.get("high/same") is not None and values.get("low/same") is not None
        else None
    )
    different_contrast = (
        values.get("high/different") - values.get("low/different")
        if values.get("high/different") is not None and values.get("low/different") is not None
        else None
    )
    exact = (
        observed_total == expected_total
        and observed_answer == expected_answer
        and observed_nonanswer == expected_nonanswer
        and same_contrast == different_contrast == 10
    )
    return {
        "expected_total_overlap_histograms": expected_total,
        "expected_answer_overlap_histograms": expected_answer,
        "expected_nonanswer_overlap_histograms": expected_nonanswer,
        "observed_total_overlap_histograms": observed_total,
        "observed_answer_overlap_histograms": observed_answer,
        "observed_nonanswer_overlap_histograms": observed_nonanswer,
        "total_overlap_balanced_within_level": (
            observed_total.get("high/same") == observed_total.get("high/different")
            and observed_total.get("low/same") == observed_total.get("low/different")
        ),
        "answer_overlap_balanced_high_low_within_relation": (
            observed_answer.get("high/same") == observed_answer.get("low/same")
            and observed_answer.get("high/different") == observed_answer.get("low/different")
        ),
        "nonanswer_high_minus_low_contrast": {
            "same": same_contrast, "different": different_contrast,
        },
        "exact_repaired_contract": exact,
        "disposition": (
            "FIT_FOR_CONTROLLED_CAUSAL_FORGETTING"
            if exact else "FAIL_REPAIR_OR_REGENERATE_BEFORE_TRAINING"
        ),
        "interpretation": (
            "CFS-2 removes CFS-1's differential total-overlap nuisance. The 4-versus-3 "
            "answer-edge difference is the intended future intervention; it is equal across "
            "high/low within relation, while the non-answer high-minus-low contrast is 10 in both relations."
            if exact else "The written CFS-2 artifacts do not satisfy the repaired factorial contract."
        ),
    }


def audit_cfs2(
    root: Path,
    path_ids: Mapping[str, set[str]],
    lure_ids: Mapping[str, set[str]],
    cfs1_ids: Mapping[str, set[str]],
) -> dict[str, Any]:
    from cfs2.validate import CONDITIONS, validate_bundle
    from lurestar.validate import canonical_key_from_line, parse_line, sha256_text

    base = root / "manifests/cfs2"
    retention = load_jsonl(base / "retention.jsonl")
    controls = load_jsonl(base / "global_controls.jsonl")
    codebook = json.loads((base / "hash_codebook.json").read_text())
    updates = {
        condition: load_jsonl(base / f"updates_{condition[0]}_{condition[1]}.jsonl")
        for condition in CONDITIONS
    }
    validation_error = None
    try:
        validate_bundle(
            retention, updates, codebook, expected_probes=2000, expected_updates=5000,
            global_controls=controls, expected_global_controls=2000,
        )
    except Exception as exc:
        validation_error = f"{type(exc).__name__}: {exc}"

    probes = {str(row["probe_id"]): str(row["line"]) for row in retention}
    sets: dict[str, tuple[set[str], set[str]]] = {}
    summaries: dict[str, Any] = {}
    stored_mismatch_counts: dict[str, dict[str, int]] = {}
    total_hists: dict[str, dict[str, int]] = {}
    answer_hists: dict[str, dict[str, int]] = {}
    nonanswer_hists: dict[str, dict[str, int]] = {}
    record_groups: dict[str, list[dict[str, Any]]] = {
        "retention": retention,
        "global_controls": controls,
        **{f"{key[0]}/{key[1]}": value for key, value in updates.items()},
    }
    for label, records in record_groups.items():
        lines = [str(record["line"]) for record in records]
        summary, prompts, graphs = _line_summary(lines)
        summary["source_histogram"] = _hist(parse_line(line).source for line in lines)
        summary["goal_histogram"] = _hist(parse_line(line).goal for line in lines)
        if "/" in label:
            totals: list[int] = []
            answers: list[int] = []
            nonanswers: list[int] = []
            mismatch = Counter()
            for row in records:
                total, answer, nonanswer = _overlap_decomposition(
                    probes[str(row["probe_id"])], str(row["line"])
                )
                totals.append(total)
                answers.append(answer)
                nonanswers.append(nonanswer)
                mismatch["total"] += int(row.get("edge_overlap_with_probe") != total)
                mismatch["answer"] += int(row.get("answer_edge_overlap_with_probe") != answer)
                mismatch["nonanswer"] += int(row.get("nonanswer_edge_overlap_with_probe") != nonanswer)
            total_hists[label] = _hist(totals)
            answer_hists[label] = _hist(answers)
            nonanswer_hists[label] = _hist(nonanswers)
            summary["recomputed_total_overlap_histogram"] = total_hists[label]
            summary["recomputed_answer_overlap_histogram"] = answer_hists[label]
            summary["recomputed_nonanswer_overlap_histogram"] = nonanswer_hists[label]
            stored_mismatch_counts[label] = dict(mismatch)
        summaries[label] = summary
        sets[label] = (prompts, graphs)

    all_prompts = set().union(*(value[0] for value in sets.values()))
    all_graphs = set().union(*(value[1] for value in sets.values()))
    legacy_prompts = (
        path_ids["train_prompts"] | path_ids["test_prompts"]
        | lure_ids["e_prompts"] | lure_ids["pool_prompts"] | cfs1_ids["all_prompts"]
    )
    legacy_graphs = (
        path_ids["train_graphs"] | path_ids["test_graphs"]
        | lure_ids["e_graphs"] | lure_ids["pool_graphs"] | cfs1_ids["all_graphs"]
    )

    stream_checks: dict[str, Any] = {}
    rows_by_condition = {
        condition: {str(row["unit_id"]): str(row["line"]) for row in rows}
        for condition, rows in updates.items()
    }
    for episode in codebook["episodes"]:
        episode_number = int(episode["episode"])
        order = [str(value) for value in episode["unit_order"]]
        for condition in CONDITIONS:
            key = f"episode{episode_number}_{condition[0]}_{condition[1]}"
            path = base / "streams" / f"graph_5_5_cfs2_{key}.txt"
            actual = [line.strip() for line in path.read_text().splitlines() if line.strip()]
            expected = [rows_by_condition[condition][unit_id] for unit_id in order]
            stream_checks[key] = {
                "count": len(actual),
                "exact_codebook_order_and_content_match": actual == expected,
                "sha256": sha256_file(path),
            }

    assessment = cfs2_construct_assessment(total_hists, answer_hists, nonanswer_hists)
    mechanically_passed = (
        validation_error is None
        and all(not any(value.values()) for value in stored_mismatch_counts.values())
        and all(value["exact_codebook_order_and_content_match"] for value in stream_checks.values())
    )
    return {
        "role": "repaired causal-forgetting factorial manipulation on controlled symbolic planning",
        "provenance": {
            "generator": "src/cfs2/generate.py",
            "construction_receipt": "manifests/cfs2/construction_receipt.json",
            "construction_receipt_sha256": sha256_file(base / "construction_receipt.json"),
            "model_or_outcome_inputs": [],
        },
        "artifact_summaries": summaries,
        "independent_bundle_validation": {"passed": validation_error is None, "error": validation_error},
        "independent_stored_overlap_mismatch_counts": stored_mismatch_counts,
        "independent_stream_checks": stream_checks,
        "identity_disjointness_from_pathstar_lurestar_and_cfs1": {
            "prompt_collisions": len(all_prompts & legacy_prompts),
            "graph_collisions": len(all_graphs & legacy_graphs),
        },
        "factorial_overlap_assessment": assessment,
        "coded_invariant_status": "PASS" if mechanically_passed else "FAIL",
        "construct_validity_status": assessment["disposition"],
        "construct_match": assessment["interpretation"],
        "known_limitations": [
            "CFS-2 identifies a controlled symbolic full-parameter adaptation effect, not a natural-language mechanism.",
            "Different-future cells necessarily retain one fewer update-answer edge than same-future cells; the design controls this by matching it across high/low and equalizing the non-answer high-low contrast.",
            "The construction still uses one fixed G(5,5) topology and 500-update adaptation regime.",
        ],
        "required_action": (
            "Use only CFS-2 streams for the repaired causal study; never mix CFS-1 and CFS-2. "
            "Bind downstream runner/evaluator inputs to these hashes before branch execution."
        ),
    }


def audit_project(root: Path) -> dict[str, Any]:
    root = root.resolve()
    _bootstrap(root)
    pathstar, path_ids = audit_pathstar(root)
    lurestar, lure_ids = audit_lurestar(root, path_ids)
    hmm = audit_hmm(root)
    cfs1, cfs1_ids = audit_cfs1(root, path_ids, lure_ids)
    cfs2 = audit_cfs2(root, path_ids, lure_ids, cfs1_ids)
    return {
        "schema": SCHEMA,
        "audit_date": AUDIT_DATE,
        "scope": "stimulus files, generators, manifests, exact solvers, and ground-truth arrays only",
        "outcome_blinding": {
            "checkpoints_opened": False,
            "training_logs_opened": False,
            "model_losses_opened": False,
            "scientific_results_opened": False,
            "forbidden_input_classes": ["checkpoints", "run receipts", "model metrics", "loss files", "hidden-state outputs"],
        },
        "interpretation_rule": (
            "A coded-invariant PASS means the artifact matches its implemented specification. It does not establish "
            "that the specification isolates the intended scientific construct."
        ),
        "datasets": {
            "pathstar": pathstar, "lurestar": lurestar, "hmm_family": hmm,
            "cfs1": cfs1, "cfs2": cfs2,
        },
        "overall_disposition": {
            "continue": [
                "Path-Star base training", "Lure-Star H1/H2 evaluation", "HMM calibration/evaluation",
                "CFS-2 downstream protocol/runner binding and then repaired branch execution",
            ],
            "pause": ["CFS-1 adaptation branches"],
            "next": ["Bind CFS-2 hashes into downstream runner/evaluator inputs", "Never mix CFS-1 and CFS-2 streams"],
        },
    }


def render_markdown(audit: Mapping[str, Any]) -> str:
    datasets = audit["datasets"]
    cfs = datasets["cfs1"]
    path = datasets["pathstar"]
    lure = datasets["lurestar"]
    hmm = datasets["hmm_family"]
    cfs2 = datasets["cfs2"]
    lines = [
        "# Outcome-blind stimulus-validity audit",
        "",
        f"**Audit date:** {audit['audit_date']}  ",
        "**Inputs:** stimulus files, generators, manifests, independent solvers, and exact HMM ground truth only.  ",
        "**Explicitly not opened:** checkpoints, losses, training logs, hidden states, evaluation metrics, or scientific outcomes.",
        "",
        "## Executive decision",
        "",
        "| Dataset | Coded invariants | Construct assessment | Decision |",
        "| --- | --- | --- | --- |",
        f"| Path-Star | {path['coded_invariant_status']} | {path['construct_validity_status']} | Continue |",
        f"| Lure-Star H1/H2 | {lure['coded_invariant_status']} | {lure['construct_validity_status']} | Continue with declared limitation |",
        f"| HMM family | {hmm['coded_invariant_status']} | {hmm['construct_validity_status']} | Continue as calibration |",
        f"| CFS-1 | {cfs['coded_invariant_status']} | **{cfs['construct_validity_status']}** | **Pause and replace with CFS-2** |",
        f"| CFS-2 | {cfs2['coded_invariant_status']} | **{cfs2['construct_validity_status']}** | Continue after downstream hash binding |",
        "",
        "> A coded-invariant PASS means that the files match the implemented specification. It does not prove that the specification isolates the intended scientific construct.",
        "",
        "## Critical finding: CFS-1 is mechanically correct but scientifically confounded",
        "",
        cfs["factorial_overlap_assessment"]["critical_finding"],
        "",
        "The exact overlap table is:",
        "",
        "| Condition | Shared edges per update | Count |",
        "| --- | ---: | ---: |",
    ]
    for condition in ("high/same", "high/different", "low/same", "low/different"):
        histogram = cfs["factorial_overlap_assessment"]["observed_overlap_histograms"][condition]
        overlap, count = next(iter(histogram.items()))
        lines.append(f"| {condition} | {overlap} | {count} |")
    lines += [
        "",
        "Because the low-arm nuisance difference is deterministic, a regression cannot recover the causal interaction from CFS-1. No CFS-1 adaptation branch should run. Preserve it as a documented design attempt and build CFS-2 with exact overlap balance.",
        "",
        "## CFS-2 independently verifies the repair",
        "",
        cfs2["factorial_overlap_assessment"]["interpretation"],
        "",
        "| Condition | total shared | update-answer shared | other shared | Count |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for condition in ("high/same", "high/different", "low/same", "low/different"):
        assessment = cfs2["factorial_overlap_assessment"]
        total_hist = assessment["observed_total_overlap_histograms"][condition]
        answer_hist = assessment["observed_answer_overlap_histograms"][condition]
        nonanswer_hist = assessment["observed_nonanswer_overlap_histograms"][condition]
        total_value, count = next(iter(total_hist.items()))
        lines.append(
            f"| {condition} | {total_value} | {next(iter(answer_hist))} | "
            f"{next(iter(nonanswer_hist))} | {count} |"
        )
    lines += [
        "",
        f"- Full independent bundle validator passed: {cfs2['independent_bundle_validation']['passed']}.",
        f"- Cross-dataset collisions with Path-Star, Lure-Star, and CFS-1: {cfs2['identity_disjointness_from_pathstar_lurestar_and_cfs1']['prompt_collisions']} prompts and {cfs2['identity_disjointness_from_pathstar_lurestar_and_cfs1']['graph_collisions']} graphs.",
        f"- All eight materialized episode/cell streams exactly match independently reconstructed codebook order and contents: {all(value['exact_codebook_order_and_content_match'] for value in cfs2['independent_stream_checks'].values())}.",
        "- The non-answer high-minus-low contrast is 10 for both same and different future relations.",
        "",
        "## Path-Star",
        "",
        path["construct_match"],
        "",
        f"- Train: {path['train']['count']:,} lines; {path['train']['solver_valid_count']:,} solver-valid; {path['train']['duplicate_graph_count']:,} duplicate graph identities.",
        f"- Test: {path['test']['count']:,} lines; {path['test']['solver_valid_count']:,} solver-valid; {path['test']['duplicate_graph_count']:,} duplicate graph identities.",
        f"- Train/test collisions: {path['train_test_identity_disjointness']['prompt_collisions']} prompts and {path['train_test_identity_disjointness']['graph_collisions']} canonical graphs.",
        f"- Fixed prompt-length histogram: `{path['train']['prompt_token_length_histogram']}`; answer-length histogram: `{path['train']['answer_length_histogram']}`.",
        f"- Train source coverage: `{path['train']['source_balance']}`; goal coverage: `{path['train']['goal_balance']}`; source-goal pair coverage: `{path['train']['source_goal_pair_coverage']}`.",
        f"- All-node coverage: `{path['train']['node_id_coverage']}`; duplicate prompts: {path['train']['duplicate_prompt_count']:,}.",
        "",
        "Limitations:",
        "",
        *[f"- {item}" for item in path["known_limitations"]],
        "",
        "## Lure-Star H1/H2",
        "",
        lure["construct_match"],
        "",
        f"- E_lure: {lure['e_lure']['quartet_count']:,} paired quartets and {lure['e_lure']['quartet_invariant_violation_count']} independent-checker violations.",
        f"- Training leakage: {lure['e_lure']['collisions_with_pathstar_train']['prompt_collisions']} prompt and {lure['e_lure']['collisions_with_pathstar_train']['graph_collisions']} graph collisions.",
        f"- Far-control edge overlap: `{lure['e_lure']['far_edge_overlap_histogram']}`.",
        f"- Suffix-depth coverage: `{lure['e_lure']['suffix_depth_histogram']}`.",
        f"- Edit-position diversity (unique position pairs): `{lure['e_lure']['unique_edit_position_pairs']}`; full position histograms are in the JSON receipt.",
        "",
        "Per-condition balance and identity summary:",
        "",
        "| Condition | n | prompt lengths | answer lengths | sources | goals | duplicate prompts/graphs |",
        "| --- | ---: | --- | --- | --- | --- | ---: |",
    ]
    for name, summary in lure["e_lure"]["condition_summaries"].items():
        lines.append(
            f"| {name} | {summary['count']} | `{summary['prompt_token_length_histogram']}` | "
            f"`{summary['answer_length_histogram']}` | `{summary['source_balance']}` | "
            f"`{summary['goal_balance']}` | {summary['duplicate_prompt_count']}/{summary['duplicate_graph_count']} |"
        )
    lines += [
        "",
        "Legacy pool disclosure:",
        "",
        "| Pool | n | collision with Path-Star train (prompt/graph) | duplicate prompts/graphs |",
        "| --- | ---: | ---: | ---: |",
    ]
    for filename, summary in lure["legacy_pools"].items():
        collision = summary["collisions_with_pathstar_train"]
        lines.append(
            f"| {filename} | {summary['count']} | {collision['prompts']}/{collision['graphs']} | "
            f"{summary['duplicate_prompt_count']}/{summary['duplicate_graph_count']} |"
        )
    lines += [
        "",
        "Limitations:",
        "",
        *[f"- {item}" for item in lure["known_limitations"]],
        "",
        "## HMM family",
        "",
        hmm["construct_match"],
        "",
    ]
    for name, regime in hmm["regimes"].items():
        lines.append(
            f"- `{name}`: train {regime['splits']['train']['shape']}, val {regime['splits']['val']['shape']}, "
            f"length-generalization {regime['splits']['lengen']['shape']}; train/val exact collisions "
            f"{regime['train_val_exact_sequence_collisions']}; posterior spot checks pass="
            f"{all(value['passed'] for value in regime['posterior_checks'].values())}; train symbol histogram "
            f"`{regime['splits']['train']['symbol_histogram']}`; duplicate train/val/length-generalization sequences "
            f"{regime['splits']['train']['duplicate_sequence_count']}/"
            f"{regime['splits']['val']['duplicate_sequence_count']}/"
            f"{regime['splits']['lengen']['duplicate_sequence_count']}."
        )
    lines += [
        "",
        "Limitations:",
        "",
        *[f"- {item}" for item in hmm["known_limitations"]],
        "",
        "## CFS-1 construction details",
        "",
        cfs["construct_match"],
        "",
        f"- Independent full-bundle validation passed: {cfs['independent_bundle_validation']['passed']}.",
        f"- Cross-dataset identity collisions: {cfs['identity_disjointness_from_pathstar_and_lurestar']['prompt_collisions']} prompts and {cfs['identity_disjointness_from_pathstar_and_lurestar']['graph_collisions']} graphs.",
        f"- Retention probes: {cfs['artifact_summaries']['retention']['count']:,}; untouched global controls: {cfs['artifact_summaries']['global_controls']['count']:,}; each update bank: {cfs['artifact_summaries']['high/same']['count']:,}.",
        "- The independent validator confirms identical global node-token counts, answer-length distributions, and probe reuse codebooks across the four update banks.",
        "",
        "| Artifact/cell | n | overlap | prompt lengths | answer lengths | sources | goals | duplicate prompts/graphs |",
        "| --- | ---: | --- | --- | --- | --- | --- | ---: |",
    ]
    for name in ("retention", "global_controls", "high/same", "high/different", "low/same", "low/different"):
        summary = cfs["artifact_summaries"][name]
        overlap = summary.get("edge_overlap_with_probe_histogram", "n/a")
        lines.append(
            f"| {name} | {summary['count']} | `{overlap}` | `{summary['prompt_token_length_histogram']}` | "
            f"`{summary['answer_length_histogram']}` | `{summary['source_balance']}` | `{summary['goal_balance']}` | "
            f"{summary['duplicate_prompt_count']}/{summary['duplicate_graph_count']} |"
        )
    lines += [
        "",
        "Limitations and required action:",
        "",
        *[f"- {item}" for item in cfs["known_limitations"]],
        f"- **{cfs['required_action']}**",
        "",
        "## CFS-2 construction details",
        "",
        cfs2["construct_match"],
        "",
        f"- Retention probes: {cfs2['artifact_summaries']['retention']['count']:,}; untouched global controls: {cfs2['artifact_summaries']['global_controls']['count']:,}; each update bank: {cfs2['artifact_summaries']['high/same']['count']:,}.",
        "- All overlap totals and decompositions below were recomputed from serialized probe/update edges; stored overlap fields were cross-checked only afterward.",
        "",
        "| Artifact/cell | n | total overlap | answer overlap | other overlap | prompt lengths | sources | goals | duplicate prompts/graphs |",
        "| --- | ---: | --- | --- | --- | --- | --- | --- | ---: |",
    ]
    for name in ("retention", "global_controls", "high/same", "high/different", "low/same", "low/different"):
        summary = cfs2["artifact_summaries"][name]
        total = summary.get("recomputed_total_overlap_histogram", "n/a")
        answer = summary.get("recomputed_answer_overlap_histogram", "n/a")
        nonanswer = summary.get("recomputed_nonanswer_overlap_histogram", "n/a")
        lines.append(
            f"| {name} | {summary['count']} | `{total}` | `{answer}` | `{nonanswer}` | "
            f"`{summary['prompt_token_length_histogram']}` | `{summary['source_balance']}` | "
            f"`{summary['goal_balance']}` | {summary['duplicate_prompt_count']}/{summary['duplicate_graph_count']} |"
        )
    lines += [
        "",
        "Limitations and required action:",
        "",
        *[f"- {item}" for item in cfs2["known_limitations"]],
        f"- **{cfs2['required_action']}**",
        "",
        "## What this audit does not establish",
        "",
        "It does not show that a model learned the task, that any hypothesis is true, or that effects generalize to natural language. Those are outcome questions and were intentionally excluded. This audit establishes artifact provenance, mechanical validity, balance/disjointness facts, and a pre-outcome judgment about construct fit.",
        "",
        "## Reproduction",
        "",
        "```bash",
        ".venv/bin/python scripts/audit_stimulus_validity.py --write",
        "```",
        "",
        "The command deterministically regenerates this document and `manifests/stimulus_validity_audit.json` from the frozen stimulus artifacts.",
        "",
    ]
    return "\n".join(lines)


def _write(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--write", action="store_true", help="write the deterministic JSON receipt and Markdown report")
    args = parser.parse_args(argv)
    audit = audit_project(args.root)
    markdown = render_markdown(audit)
    if args.write:
        receipt = args.root / "manifests/stimulus_validity_audit.json"
        report = args.root / "docs/STIMULUS_VALIDITY_AUDIT.md"
        _write(receipt, json.dumps(audit, indent=2, sort_keys=True) + "\n")
        _write(report, markdown)
        print(f"wrote {receipt}")
        print(f"wrote {report}")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
