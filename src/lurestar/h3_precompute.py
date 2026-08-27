"""Outcome-blind, immutable pre-compute primitives for Lure-Star H3.

The module deliberately separates three phases:

``generate``
    Uses only frozen corpora/stimuli and deterministic RNG streams.  It creates exactly
    15,000 paired middle-distance candidates plus oversized independent acquisition pools.
``score``
    Is implemented by :mod:`scripts.score_h3_pilot`; it may inspect only the single frozen
    non-confirmatory BST pilot and writes restartable loss chunks.
``select``
    Applies the rule frozen here before scoring.  It never opens a model or a confirmatory
    result and fails closed if the pilot caliper makes the requested banks infeasible.

All durable writes use create-or-verify semantics.  An existing byte-identical artifact is
accepted for restartability; an existing different artifact is never replaced.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import tempfile
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .generate import (
    Graph,
    SWAP_DEPTHS,
    _sample_base_graph,
    graph_from_line,
    sample_far_graph,
    suffix_swap,
)
from .validate import (
    MAX_NODES,
    TrainingIndex,
    canonical_key_from_line,
    parse_line,
    sha256_text,
    validate_line,
)


SCHEMA = "nextlat_forgetting/h3_precompute/1"
PILOT_FREEZE_SCHEMA = "nextlat_forgetting/h3_pilot_freeze/1"
LOSS_TABLE_SCHEMA = "nextlat_forgetting/h3_pilot_loss_table/1"
SELECTION_SCHEMA_VERSION = 1
MID_COUNT = 5_000
MID_CANDIDATE_COUNT = 15_000
ACQUISITION_COUNT = 2_000
ACQUISITION_CANDIDATE_COUNT = 6_000
MASTER_SEED = 2_026_082_401
PILOT_SEED = 1_234
PILOT_STEP = 500
PILOT_CHECKPOINT_SHA256 = (
    "1f5f00611e33ada0ac0a778f9d45bef9e174f1bbeedfaaa3491018a9bf400176"
)
PILOT_CONFIG_SHA256 = (
    "03e1b9e4a1a2a7e44b68cf69a1534a90a695685004ea5ebe79822e0bd9472e98"
)
PILOT_STATE_SHA256 = (
    "5ea74eb05e72759fbaa2cded94ee758300cfe9057d5a7d6f8ef81d566afa577a"
)
PILOT_PROFILE_ID = "a100-be81d2f1e79c-8d316efc9c53"
PILOT_STATE_GENERATION = 12
PILOT_STATE_URI = (
    "gs://nextlat-lurestar-project-flash-490419/lurestar/profiles/"
    f"{PILOT_PROFILE_ID}/state.json"
)
PILOT_CHECKPOINT_OBJECT = (
    "lurestar/profiles/a100-be81d2f1e79c-8d316efc9c53/artifacts/sha256/"
    f"{PILOT_CHECKPOINT_SHA256}/gate/root/runs/bst/seed1234/base/"
    "bst-seed1234-base/ckpt_iter_500_1.1343.pt"
)
PILOT_CHECKPOINT_GENERATION = "1787553307238486"
PILOT_CONFIG_OBJECT = (
    "lurestar/profiles/a100-be81d2f1e79c-8d316efc9c53/artifacts/sha256/"
    f"{PILOT_CONFIG_SHA256}/gate/root/runs/bst/seed1234/base/"
    "bst-seed1234-base/materialized_config.yaml"
)
PILOT_CONFIG_GENERATION = "1787552617534322"
PILOT_CHECKPOINT_META_SHA256 = (
    "585480bc93fffc80e02c989208085c2855a8bc8d65b61db9b2168b419dac96b9"
)
UPSTREAM_COMMIT = "3770be6009cea2b3c455a9ce7f2ca88b504bb955"
MID_LOSS_CALIPER = 0.1


class PrecomputeRefused(RuntimeError):
    """A pre-compute action would violate a frozen identity or selection rule."""


def sha256_file(path: os.PathLike[str] | str) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def create_or_verify(path: os.PathLike[str] | str, payload: bytes) -> str:
    """Atomically create ``path`` or verify that its existing bytes are identical."""
    target = pathlib.Path(path)
    digest = hashlib.sha256(payload).hexdigest()
    sidecar = pathlib.Path(f"{target}.sha256")
    if target.exists():
        if not target.is_file() or target.read_bytes() != payload:
            raise PrecomputeRefused(f"refusing to overwrite frozen artifact: {target}")
        expected_sidecar = f"{digest}  {target.name}\n".encode()
        if not sidecar.is_file() or sidecar.read_bytes() != expected_sidecar:
            raise PrecomputeRefused(f"frozen artifact has missing/stale sidecar: {target}")
        return digest
    if sidecar.exists():
        raise PrecomputeRefused(f"orphan sidecar blocks create-only output: {sidecar}")
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".partial", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        # Link gives create-only publication even if another controller races us.
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != payload:
                raise PrecomputeRefused(f"concurrent writer published different bytes: {target}")
        expected_sidecar = f"{digest}  {target.name}\n".encode()
        fd2, temporary_sidecar = tempfile.mkstemp(
            prefix=f".{sidecar.name}.", suffix=".partial", dir=target.parent
        )
        try:
            with os.fdopen(fd2, "wb") as handle:
                handle.write(expected_sidecar)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary_sidecar, sidecar)
            except FileExistsError:
                if sidecar.read_bytes() != expected_sidecar:
                    raise PrecomputeRefused(f"concurrent writer published stale sidecar: {sidecar}")
        finally:
            pathlib.Path(temporary_sidecar).unlink(missing_ok=True)
    finally:
        pathlib.Path(temporary).unlink(missing_ok=True)
    return digest


def verify_sidecar(path: os.PathLike[str] | str) -> str:
    target = pathlib.Path(path)
    sidecar = pathlib.Path(f"{target}.sha256")
    if not target.is_file() or not sidecar.is_file():
        raise PrecomputeRefused(f"missing frozen input or SHA-256 sidecar: {target}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    actual = sha256_file(target)
    if not fields or fields[0] != actual:
        raise PrecomputeRefused(f"SHA-256 sidecar mismatch: {target}")
    return actual


def _read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    verify_sidecar(path)
    rows: list[dict[str, Any]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise PrecomputeRefused(f"{path}:{number}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise PrecomputeRefused(f"{path}:{number}: row must be an object")
        rows.append(row)
    return rows


def _jsonl_payload(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json(dict(row)) for row in rows)


def prompt_sha(line: str) -> str:
    return sha256_text(line[: line.index("=") + 1])


def structural_distance(left: str, right: str) -> float:
    a, b = set(parse_line(left).edges), set(parse_line(right).edges)
    return 1.0 - len(a & b) / 20.0


def _identity_sets(rows: Iterable[Mapping[str, Any]]) -> tuple[set[str], set[str]]:
    prompts: set[str] = set()
    graphs: set[str] = set()
    for row in rows:
        if "conditions" in row:
            records = row["conditions"].values()
        else:
            records = (row,)
        for record in records:
            line = str(record["line"])
            prompts.add(prompt_sha(line))
            graphs.add(canonical_key_from_line(line))
    return prompts, graphs


def _check_new_line(line: str, prompts: set[str], graphs: set[str]) -> tuple[str, str]:
    validate_line(line)
    prompt, graph = prompt_sha(line), canonical_key_from_line(line)
    if prompt in prompts or graph in graphs:
        raise PrecomputeRefused("candidate identity collides with a frozen domain")
    return prompt, graph


def _mid_candidate(base: Graph, *, item: int, slot: int, attempt: int) -> Graph:
    rng = np.random.default_rng([MASTER_SEED, 11, item, slot, attempt])
    graph = base
    # One/two/three rewires give a model-blind structural spread centered well below far.
    for _ in range(slot + 1):
        arms = [int(value) for value in rng.choice(5, size=2, replace=False)]
        graph = suffix_swap(graph, arms[0], arms[1], int(rng.choice(SWAP_DEPTHS)))
    return graph


def generate_mid_candidates(
    near: Sequence[Mapping[str, Any]], excluded_prompts: set[str], excluded_graphs: set[str]
) -> list[dict[str, Any]]:
    if len(near) != MID_COUNT:
        raise PrecomputeRefused(f"B_near must contain exactly {MID_COUNT} rows")
    prompts, graphs = set(excluded_prompts), set(excluded_graphs)
    rows: list[dict[str, Any]] = []
    for item_index, near_row in enumerate(near):
        near_line = str(near_row["line"])
        base = graph_from_line(near_line)
        used_distances: set[float] = set()
        for slot in range(3):
            accepted: tuple[str, str, str, float, int] | None = None
            for attempt in range(512):
                graph = _mid_candidate(base, item=item_index, slot=slot, attempt=attempt)
                line = graph.serialize()
                distance = structural_distance(near_line, line)
                rounded = round(distance, 12)
                if rounded == 0.0 or rounded in used_distances or distance >= 0.9:
                    continue
                try:
                    psha, gsha = _check_new_line(line, prompts, graphs)
                except PrecomputeRefused:
                    continue
                accepted = line, psha, gsha, distance, attempt
                break
            if accepted is None:
                raise PrecomputeRefused(
                    f"could not construct three distinct B_mid candidates for near row {item_index}"
                )
            line, psha, gsha, distance, attempt = accepted
            prompts.add(psha)
            graphs.add(gsha)
            used_distances.add(round(distance, 12))
            rows.append({
                "schema": SCHEMA,
                "pool": "B_mid",
                "item_id": f"{item_index}:{slot}",
                "paired_near_prompt_sha256": str(near_row["prompt_sha256"]),
                "line": line,
                "prompt_sha256": psha,
                "graph_key": gsha,
                "solver_verified": True,
                "normalized_edge_disagreement": distance,
                "construction": "deterministic_sequential_suffix_swaps",
                "rewire_count": slot + 1,
                "accepted_attempt": attempt,
                "master_seed": MASTER_SEED,
            })
    if len(rows) != MID_CANDIDATE_COUNT:
        raise AssertionError("internal B_mid count failure")
    return rows


def _acquisition_triplet(index: int, attempt: int) -> dict[str, tuple[str, float, int]]:
    rng = np.random.default_rng([MASTER_SEED, 29, index, attempt])
    source, arms = _sample_base_graph(rng, MAX_NODES)
    goal = arms[0][-1]
    anchor = Graph(source, goal, arms, tuple(int(x) for x in rng.permutation(20)))
    near_arms = [int(x) for x in rng.choice(5, size=2, replace=False)]
    near = suffix_swap(anchor, near_arms[0], near_arms[1], int(rng.choice(SWAP_DEPTHS)))

    mid: Graph | None = None
    for mid_attempt in range(64):
        mrng = np.random.default_rng([MASTER_SEED, 31, index, attempt, mid_attempt])
        candidate = anchor
        for _ in range(3 + mid_attempt % 3):
            pair = [int(x) for x in mrng.choice(5, size=2, replace=False)]
            candidate = suffix_swap(candidate, pair[0], pair[1], int(mrng.choice(SWAP_DEPTHS)))
        distance = structural_distance(anchor.serialize(), candidate.serialize())
        if 0.15 <= distance <= 0.6 and candidate.serialize() != near.serialize():
            mid = candidate
            break
    if mid is None:
        raise PrecomputeRefused("could not generate a middle acquisition candidate")
    nodes = [node for arm in anchor.arms for node in arm]
    far, overlap, far_attempts = sample_far_graph(
        rng, anchor.source, anchor.goal, nodes, frozenset(anchor.edges_in_order()),
        anchor.path(), max_overlap=2, max_tries=500,
    )
    return {
        "near": (near.serialize(), structural_distance(anchor.serialize(), near.serialize()), 1),
        "mid": (mid.serialize(), structural_distance(anchor.serialize(), mid.serialize()), 3 + mid_attempt % 3),
        "far": (far.serialize(), structural_distance(anchor.serialize(), far.serialize()), far_attempts),
    }


def generate_acquisition_candidates(
    excluded_prompts: set[str], excluded_graphs: set[str]
) -> dict[str, list[dict[str, Any]]]:
    prompts, graphs = set(excluded_prompts), set(excluded_graphs)
    output = {branch: [] for branch in ("near", "mid", "far")}
    for index in range(ACQUISITION_CANDIDATE_COUNT):
        accepted: dict[str, tuple[str, float, int]] | None = None
        accepted_attempt = -1
        for attempt in range(512):
            triplet = _acquisition_triplet(index, attempt)
            local_prompts, local_graphs = set(), set()
            try:
                for line, _distance, _tries in triplet.values():
                    validate_line(line)
                    psha, gsha = prompt_sha(line), canonical_key_from_line(line)
                    if (
                        psha in prompts or psha in local_prompts
                        or gsha in graphs or gsha in local_graphs
                    ):
                        raise PrecomputeRefused(
                            "acquisition candidate collides with a frozen domain"
                        )
                    local_prompts.add(psha)
                    local_graphs.add(gsha)
            except PrecomputeRefused:
                continue
            accepted, accepted_attempt = triplet, attempt
            break
        if accepted is None:
            raise PrecomputeRefused(f"could not generate independent acquisition triplet {index}")
        for branch, (line, distance, construction_tries) in accepted.items():
            psha, gsha = prompt_sha(line), canonical_key_from_line(line)
            prompts.add(psha)
            graphs.add(gsha)
            output[branch].append({
                "schema": SCHEMA,
                "pool": f"B_{branch}_validation_candidate",
                "item_id": f"acquisition:{index}:{branch}",
                "line": line,
                "prompt_sha256": psha,
                "graph_key": gsha,
                "solver_verified": True,
                "anchor_structural_distance": distance,
                "construction": f"independent_{branch}_structural_candidate",
                "accepted_attempt": accepted_attempt,
                "construction_tries": construction_tries,
                "master_seed": MASTER_SEED,
            })
    return output


def pilot_freeze_payload(
    *, generator_sha256: str, scorer_sha256: str, tokenizer_sha256: str,
    adaptation_contract_sha256: str,
) -> dict[str, Any]:
    return {
        "schema": PILOT_FREEZE_SCHEMA,
        "status": "FROZEN_BEFORE_PILOT_SCORING",
        "role": "non_confirmatory_engineering_pilot",
        "sole_pilot": True,
        "substitution_or_reselection_permitted": False,
        "model_family": "bst",
        "seed": PILOT_SEED,
        "training_step": PILOT_STEP,
        "checkpoint": {
            "sha256": PILOT_CHECKPOINT_SHA256,
            "remote_object": PILOT_CHECKPOINT_OBJECT,
            "object_generation": PILOT_CHECKPOINT_GENERATION,
            "size_bytes": 567_659_011,
        },
        "materialized_config": {
            "sha256": PILOT_CONFIG_SHA256,
            "remote_object": PILOT_CONFIG_OBJECT,
            "object_generation": PILOT_CONFIG_GENERATION,
        },
        "checkpoint_metadata_sha256": PILOT_CHECKPOINT_META_SHA256,
        "profile_state": {
            "profile_id": PILOT_PROFILE_ID,
            "uri": PILOT_STATE_URI,
            "generation": PILOT_STATE_GENERATION,
            "sha256": PILOT_STATE_SHA256,
        },
        "upstream_commit": UPSTREAM_COMMIT,
        "selection_rule": {
            "far": "exact empirical quantile ranks 3*r+1 after (loss,SHA256) ordering",
            "mid": "same pilot-loss decile; absolute-loss caliper <=0.1; closest to global eligible structural-distance median; SHA256 tie-break",
            "acquisition": "exactly 200 items per pilot-loss decile per branch; SHA256 tie-break",
            "confirmatory_outcomes_consulted": False,
            "confirmatory_checkpoints_permitted": False,
        },
        "source_bindings": {
            "generator_sha256": generator_sha256,
            "scorer_sha256": scorer_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "adaptation_contract_sha256": adaptation_contract_sha256,
        },
        "rationale": (
            "The recovered checkpoint predates confirmatory outcomes and already paid the 500-step "
            "engineering cost. Reusing it only to freeze nuisance matching avoids new model choice "
            "and compute; it cannot support a scientific effect claim."
        ),
    }


def materialize_candidates(
    *, root: pathlib.Path, output_dir: pathlib.Path, scorer_path: pathlib.Path
) -> dict[str, Any]:
    inputs = {
        name: root / "manifests" / name
        for name in ("b_near.jsonl", "b_far.jsonl", "a_pair.jsonl", "e_lure.jsonl")
    }
    hashes = {name: verify_sidecar(path) for name, path in inputs.items()}
    near = _read_jsonl(inputs["b_near.jsonl"])
    far = _read_jsonl(inputs["b_far.jsonl"])
    if len(far) != MID_CANDIDATE_COUNT:
        raise PrecomputeRefused("B_far must remain the frozen 15,000-item candidate bank")
    frozen_rows = near + far + _read_jsonl(inputs["a_pair.jsonl"]) + _read_jsonl(inputs["e_lure.jsonl"])
    excluded_prompts, excluded_graphs = _identity_sets(frozen_rows)
    train_path = root / "data/stargraph/graph_5_5_sample_200000.txt"
    training = TrainingIndex.build(train_path)
    excluded_prompts |= training.prompt_hashes
    excluded_graphs |= training.graph_keys

    mid = generate_mid_candidates(near, excluded_prompts, excluded_graphs)
    mid_prompts, mid_graphs = _identity_sets(mid)
    acquisition = generate_acquisition_candidates(
        excluded_prompts | mid_prompts, excluded_graphs | mid_graphs
    )
    outputs: dict[str, str] = {}
    for name, rows in {
        "b_mid_candidates.jsonl": mid,
        **{f"acquisition_{branch}_candidates.jsonl": rows for branch, rows in acquisition.items()},
    }.items():
        outputs[name] = create_or_verify(output_dir / name, _jsonl_payload(rows))
    freeze = pilot_freeze_payload(
        generator_sha256=sha256_file(pathlib.Path(__file__)),
        scorer_sha256=sha256_file(scorer_path),
        tokenizer_sha256=sha256_file(root / "upstream/NextLat/data/stargraph.py"),
        adaptation_contract_sha256=sha256_file(root / "src/lurestar/adaptation.py"),
    )
    outputs["pilot_freeze.json"] = create_or_verify(
        output_dir / "pilot_freeze.json", canonical_json(freeze)
    )
    receipt = {
        "schema": SCHEMA,
        "status": "CPU_CANDIDATES_FROZEN",
        "scientific_outcomes_inspected": False,
        "paid_compute_used": False,
        "master_seed": MASTER_SEED,
        "counts": {
            "mid": len(mid),
            **{f"acquisition_{branch}": len(rows) for branch, rows in acquisition.items()},
        },
        "inputs": hashes,
        "training_corpus": {"path": str(train_path.relative_to(root)), "sha256": sha256_file(train_path)},
        "outputs": outputs,
        "checks": [
            "independent_solver_verified",
            "prompt_and_graph_identity_disjoint_from_training_e_lure_b_near_b_far",
            "pairwise_candidate_identity_disjointness",
            "exact_counts",
            "structural_variation",
            "create_only_publication",
        ],
    }
    create_or_verify(output_dir / "candidate_generation_receipt.json", canonical_json(receipt))
    return receipt


@dataclass(frozen=True)
class LossRow:
    pool: str
    prompt_sha256: str
    loss: float


def load_loss_table(path: pathlib.Path) -> tuple[dict[str, LossRow], str]:
    digest = verify_sidecar(path)
    rows = _read_jsonl(path)
    out: dict[str, LossRow] = {}
    for index, row in enumerate(rows):
        if row.get("schema") != LOSS_TABLE_SCHEMA:
            raise PrecomputeRefused(f"loss row {index} has wrong schema")
        sha = row.get("prompt_sha256")
        loss = row.get("loss")
        if not isinstance(sha, str) or len(sha) != 64 or sha in out:
            raise PrecomputeRefused(f"loss row {index} has invalid/duplicate identity")
        if not isinstance(loss, (int, float)) or isinstance(loss, bool) or not math.isfinite(loss):
            raise PrecomputeRefused(f"loss row {index} has nonfinite loss")
        out[sha] = LossRow(str(row.get("pool")), sha, float(loss))
    return out, digest


def _rank(rows: Sequence[Mapping[str, Any]], losses: Mapping[str, LossRow]) -> dict[str, tuple[int, float, int]]:
    for row in rows:
        sha = str(row["prompt_sha256"])
        if sha not in losses:
            raise PrecomputeRefused(f"pilot loss table omits candidate {sha}")
    ordered = sorted(rows, key=lambda row: (losses[str(row["prompt_sha256"])].loss, str(row["prompt_sha256"])))
    n = len(ordered)
    return {
        str(row["prompt_sha256"]): (rank, (rank + 0.5) / n, min(9, rank * 10 // n))
        for rank, row in enumerate(ordered)
    }


def _pilot_provenance(freeze_sha: str, loss_sha: str, selector_sha: str) -> dict[str, Any]:
    return {
        "role": "non_confirmatory",
        "frozen_before_confirmatory": True,
        "inspected_confirmatory_checkpoints": False,
        "inspected_confirmatory_results": False,
        "optimized_h3_outcomes": False,
        "checkpoint_sha256": PILOT_CHECKPOINT_SHA256,
        "loss_table_sha256": loss_sha,
        "selector_code_sha256": selector_sha,
        "pilot_freeze_sha256": freeze_sha,
        "created_at_utc": "2026-08-24T07:40:00Z",
    }


def select_outputs(
    *, root: pathlib.Path, candidate_dir: pathlib.Path, loss_table: pathlib.Path,
    output_dir: pathlib.Path,
) -> dict[str, Any]:
    near_path, far_path = root / "manifests/b_near.jsonl", root / "manifests/b_far.jsonl"
    mid_path = candidate_dir / "b_mid_candidates.jsonl"
    freeze_path = candidate_dir / "pilot_freeze.json"
    near, far, mid = _read_jsonl(near_path), _read_jsonl(far_path), _read_jsonl(mid_path)
    if (len(near), len(far), len(mid)) != (MID_COUNT, MID_CANDIDATE_COUNT, MID_CANDIDATE_COUNT):
        raise PrecomputeRefused("selection inputs do not have frozen 5k/15k/15k counts")
    freeze_sha = verify_sidecar(freeze_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    source_root = pathlib.Path(__file__).resolve().parents[2]
    if freeze != pilot_freeze_payload(
        generator_sha256=sha256_file(pathlib.Path(__file__)),
        scorer_sha256=sha256_file(source_root / "scripts/score_h3_pilot.py"),
        tokenizer_sha256=sha256_file(source_root / "upstream/NextLat/data/stargraph.py"),
        adaptation_contract_sha256=sha256_file(source_root / "src/lurestar/adaptation.py"),
    ):
        raise PrecomputeRefused("pilot freeze is stale or does not designate the sole frozen pilot")
    losses, loss_sha = load_loss_table(loss_table)
    selector_sha = sha256_file(pathlib.Path(__file__))
    pilot = _pilot_provenance(freeze_sha, loss_sha, selector_sha)

    near_rank, far_rank, mid_rank = _rank(near, losses), _rank(far, losses), _rank(mid, losses)
    far_order = sorted(far, key=lambda row: far_rank[str(row["prompt_sha256"])][0])
    far_selection = []
    for near_row in near:
        nsha = str(near_row["prompt_sha256"])
        rank, quantile, _decile = near_rank[nsha]
        chosen = far_order[3 * rank + 1]
        fsha = str(chosen["prompt_sha256"])
        if abs(far_rank[fsha][1] - quantile) > 1e-15:
            raise AssertionError("the frozen 3:1 far quantile identity failed")
        far_selection.append({
            "near_prompt_sha256": nsha,
            "far_prompt_sha256": fsha,
            "near_loss_quantile": quantile,
            "far_loss_quantile": quantile,
        })
    far_artifact = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "purpose": "h3_far_loss_quantile_match",
        "selection_method": "non_confirmatory_pilot_loss_quantile_match",
        "near_bank_sha256": verify_sidecar(near_path),
        "candidate_bank_sha256": verify_sidecar(far_path),
        "pilot": pilot,
        "selection": far_selection,
    }

    near_by_sha = {str(row["prompt_sha256"]): row for row in near}
    eligible: dict[str, list[tuple[float, str]]] = {sha: [] for sha in near_by_sha}
    candidate_table = []
    eligible_distances: list[float] = []
    for row in mid:
        msha = str(row["prompt_sha256"])
        nsha = str(row["paired_near_prompt_sha256"])
        if nsha not in near_by_sha:
            raise PrecomputeRefused("B_mid candidate points outside frozen B_near")
        distance = structural_distance(str(near_by_sha[nsha]["line"]), str(row["line"]))
        difference = abs(losses[nsha].loss - losses[msha].loss)
        same_decile = near_rank[nsha][2] == mid_rank[msha][2]
        path_match = len(parse_line(str(near_by_sha[nsha]["line"])).answer) == len(
            parse_line(str(row["line"])).answer
        )
        is_eligible = same_decile and difference <= MID_LOSS_CALIPER + 1e-12 and path_match
        candidate_table.append({
            "mid_prompt_sha256": msha,
            "near_loss_decile": near_rank[nsha][2],
            "mid_loss_decile": mid_rank[msha][2],
            "pilot_loss_absolute_difference": difference,
            "normalized_edge_disagreement": distance,
            "eligible": is_eligible,
        })
        if is_eligible:
            eligible[nsha].append((distance, msha))
            eligible_distances.append(distance)
    missing = [sha for sha, values in eligible.items() if not values]
    if missing:
        raise PrecomputeRefused(
            f"mid pilot caliper is infeasible for {len(missing)} near items; no rule change permitted"
        )
    median = float(np.median(np.asarray(eligible_distances, dtype=np.float64)))
    mid_selection = []
    selected_mid: set[str] = set()
    for near_row in near:
        nsha = str(near_row["prompt_sha256"])
        distance, msha = min(eligible[nsha], key=lambda value: (abs(value[0] - median), value[1]))
        if msha in selected_mid:
            raise PrecomputeRefused("mid mapping attempts to reuse a candidate")
        selected_mid.add(msha)
        mid_selection.append({
            "near_prompt_sha256": nsha,
            "mid_prompt_sha256": msha,
            "normalized_edge_disagreement": distance,
        })
    mid_artifact = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "purpose": "h3_mid_structural_distance_match",
        "selection_method": "frozen_structural_median_with_pilot_loss_decile_caliper",
        "near_bank_sha256": verify_sidecar(near_path),
        "candidate_bank_sha256": verify_sidecar(mid_path),
        "distance_quantile": 0.5,
        "pilot_loss_caliper": MID_LOSS_CALIPER,
        "eligible_median_normalized_edge_disagreement": median,
        "tie_break": "candidate_prompt_sha256_ascending",
        "pilot": pilot,
        "candidate_table": candidate_table,
        "selection": mid_selection,
    }

    acquisition_selected: dict[str, list[dict[str, Any]]] = {}
    acquisition_hashes: dict[str, str] = {}
    acquisition_inputs: dict[str, str] = {}
    for branch in ("near", "mid", "far"):
        path = candidate_dir / f"acquisition_{branch}_candidates.jsonl"
        rows = _read_jsonl(path)
        if len(rows) != ACQUISITION_CANDIDATE_COUNT:
            raise PrecomputeRefused(f"{branch} acquisition pool is not oversized 6,000")
        ranks = _rank(rows, losses)
        by_decile: dict[int, list[dict[str, Any]]] = {value: [] for value in range(10)}
        for row in rows:
            by_decile[ranks[str(row["prompt_sha256"])][2]].append(row)
        chosen: list[dict[str, Any]] = []
        for decile in range(10):
            ordered = sorted(by_decile[decile], key=lambda row: str(row["prompt_sha256"]))
            if len(ordered) < ACQUISITION_COUNT // 10:
                raise PrecomputeRefused(f"{branch} acquisition decile {decile} is infeasible")
            chosen.extend(ordered[: ACQUISITION_COUNT // 10])
        chosen.sort(key=lambda row: str(row["item_id"]))
        acquisition_selected[branch] = chosen
        output_path = output_dir / f"acquisition_{branch}.jsonl"
        acquisition_hashes[branch] = create_or_verify(output_path, _jsonl_payload(chosen))
        acquisition_inputs[branch] = verify_sidecar(path)
    # Generation already proves cross-domain disjointness. Recheck selected identities locally.
    selected_prompt_sets = {
        branch: {str(row["prompt_sha256"]) for row in rows}
        for branch, rows in acquisition_selected.items()
    }
    if any(selected_prompt_sets[a] & selected_prompt_sets[b] for a, b in (("near", "mid"), ("near", "far"), ("mid", "far"))):
        raise PrecomputeRefused("selected acquisition banks overlap")

    far_sha = create_or_verify(output_dir / "far_selection.json", canonical_json(far_artifact))
    mid_sha = create_or_verify(output_dir / "mid_selection.json", canonical_json(mid_artifact))
    acquisition_artifact = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "purpose": "h3_independent_acquisition_banks",
        "selection_method": "model_blind_structural_then_frozen_pilot_loss_decile",
        "frozen_before_confirmatory": True,
        "inspected_confirmatory_checkpoints": False,
        "inspected_confirmatory_results": False,
        "optimized_h3_outcomes": False,
        "disjoint_from_training": True,
        "matched_target_path_distribution": True,
        "matched_pilot_loss_deciles": True,
        "selector_code_sha256": selector_sha,
        "bank_sha256": acquisition_hashes,
        "counts": {branch: ACQUISITION_COUNT for branch in acquisition_hashes},
        "candidate_bank_sha256": acquisition_inputs,
        "pilot": pilot,
        "per_decile_counts": {branch: {str(d): 200 for d in range(10)} for branch in acquisition_hashes},
    }
    acquisition_sha = create_or_verify(
        output_dir / "acquisition_provenance.json", canonical_json(acquisition_artifact)
    )
    receipt = {
        "schema": SCHEMA,
        "status": "H3_SELECTIONS_FROZEN",
        "confirmatory_results_inspected": False,
        "pilot_freeze_sha256": freeze_sha,
        "pilot_loss_table_sha256": loss_sha,
        "outputs": {
            "far_selection.json": far_sha,
            "mid_selection.json": mid_sha,
            "acquisition_provenance.json": acquisition_sha,
            **{f"acquisition_{key}.jsonl": value for key, value in acquisition_hashes.items()},
        },
        "mid_global_eligible_distance_p50": median,
        "mid_eligible_count": len(eligible_distances),
    }
    create_or_verify(output_dir / "selection_receipt.json", canonical_json(receipt))
    return receipt


def plan(*, root: pathlib.Path, candidate_dir: pathlib.Path) -> dict[str, Any]:
    inputs = [
        root / "data/stargraph/graph_5_5_sample_200000.txt",
        *(root / "manifests" / name for name in ("b_near.jsonl", "b_far.jsonl", "a_pair.jsonl", "e_lure.jsonl")),
    ]
    return {
        "schema": SCHEMA,
        "mode": "READ_ONLY_PLAN",
        "would_write": [
            str(candidate_dir / "b_mid_candidates.jsonl"),
            *(str(candidate_dir / f"acquisition_{branch}_candidates.jsonl") for branch in ("near", "mid", "far")),
            str(candidate_dir / "pilot_freeze.json"),
            str(candidate_dir / "candidate_generation_receipt.json"),
        ],
        "inputs_present": {str(path): path.is_file() for path in inputs},
        "counts": {
            "mid_candidates": MID_CANDIDATE_COUNT,
            "acquisition_candidates_per_branch": ACQUISITION_CANDIDATE_COUNT,
            "pilot_score_total": MID_COUNT + MID_CANDIDATE_COUNT * 2 + ACQUISITION_CANDIDATE_COUNT * 3,
        },
        "estimated_cpu_generation_minutes": [2, 8],
        "estimated_candidate_disk_mib": [20, 45],
        "estimated_pilot_scoring_gpu_minutes_a100": [3, 12],
        "next_action_after_generation": (
            "download the frozen generation-12 checkpoint/config by exact object generation, "
            "create a hash-bound score job, then run score_h3_pilot.py --mode score"
        ),
    }
