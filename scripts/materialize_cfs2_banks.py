#!/usr/bin/env python3
"""Materialize immutable, model-blind, overlap-balanced CFS-2 banks."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile
from typing import Any, Mapping, Sequence


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfs2 import generate as G  # noqa: E402
from cfs2 import validate as V  # noqa: E402


ARM_ORDER = ("high_different", "low_different", "high_same", "low_same")
EVALUATION_INPUT_ORDER = (
    "margin", "retention_ce", "retention_exact_path", "global_controls",
    "state_drift", "pregeometry",
)
ORDER_SALT = "20260824|cfs2-repaired-stimulus|execution-order"


def _reference(path: str, digest: str) -> dict[str, str]:
    return {"path": path, "sha256": digest}


def _branch_plan() -> list[dict[str, Any]]:
    """CFS-2's lattice; reusing a CFS-1-only parent needs a later receipt."""
    planned = [f"nextlat-seed{seed}-base" for seed in range(1234, 1239)]
    cfs2_only = [f"nextlat-seed{seed}-cfs2-base" for seed in range(2234, 2237)]
    return [
        {
            "parent_id": parent_id,
            "parent_role": "planned_base" if parent_id in planned else "cfs2_only_compatible_base",
            "episode": episode,
            "overlap": arm.split("_", 1)[0],
            "future_relation": arm.split("_", 1)[1],
            "branch_id": f"cfs2-{parent_id}-e{episode}-{arm}",
            "adaptation_steps": 500,
        }
        for parent_id in planned + cfs2_only
        for episode in (0, 1)
        for arm in ARM_ORDER
    ]


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(V.canonical_json(row) for row in rows)


def _write_immutable(path: pathlib.Path, payload: bytes) -> str:
    digest = V.sha256_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise V.CFS2ValidationError(
                f"refusing to overwrite immutable artifact {path}"
            )
    else:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != payload:
                    raise V.CFS2ValidationError(
                        f"concurrent immutable write disagreed for {path}"
                    )
            finally:
                pathlib.Path(temporary).unlink(missing_ok=True)
        except BaseException:
            pathlib.Path(temporary).unlink(missing_ok=True)
            raise
    sidecar_path = pathlib.Path(f"{path}.sha256")
    sidecar = f"{digest}  {path.name}\n".encode()
    if sidecar_path.exists() and sidecar_path.read_bytes() != sidecar:
        raise V.CFS2ValidationError(
            f"refusing to overwrite immutable sidecar {sidecar_path}"
        )
    if not sidecar_path.exists():
        sidecar_path.write_bytes(sidecar)
    return digest


def _trainer_stream(
    rows: Sequence[Mapping[str, Any]], unit_order: Sequence[str]
) -> bytes:
    by_unit = {str(row["unit_id"]): row for row in rows}
    if set(by_unit) != set(unit_order) or len(by_unit) != len(unit_order):
        raise V.CFS2ValidationError(
            "trainer stream cannot bind every ordered unit exactly once"
        )
    return (
        "\n".join(str(by_unit[unit]["line"]) for unit in unit_order) + "\n"
    ).encode()


def materialize(
    *,
    root: pathlib.Path,
    output_dir: pathlib.Path,
    n_probes: int,
    n_updates: int,
    dry_run: bool,
) -> dict[str, Any]:
    legacy = V.build_legacy_index(root)
    bundle = G.build_bundle(
        n_probes=n_probes, n_updates=n_updates, legacy=legacy
    )
    if dry_run:
        return {
            "schema": "nextlat_forgetting/cfs2_no_model_dry_run/1",
            "status": "PASS",
            "model_imported": False,
            "model_outputs_opened": False,
            "scientific_outcomes_opened": False,
            "counts": {
                "retention": len(bundle.retention),
                "global_controls": len(bundle.global_controls),
                "each_update_bank": n_updates,
                "episodes": 2,
            },
            "exact_edge_overlap": {
                "high_same": 18,
                "high_different": 18,
                "low_same": 8,
                "low_different": 8,
            },
            "legacy_sources_indexed": len(legacy.sources),
        }

    payloads: dict[str, bytes] = {
        "retention.jsonl": _jsonl(bundle.retention),
        "global_controls.jsonl": _jsonl(bundle.global_controls),
        "hash_codebook.json": V.canonical_json(bundle.codebook),
    }
    for condition, rows in bundle.updates.items():
        payloads[f"updates_{condition[0]}_{condition[1]}.jsonl"] = _jsonl(rows)
    bindings = {
        name: _write_immutable(output_dir / name, payload)
        for name, payload in payloads.items()
    }
    episode_orders = {
        int(item["episode"]): item for item in bundle.codebook["episodes"]
    }
    stream_bindings: dict[str, str] = {}
    for episode in (0, 1):
        for arm in ARM_ORDER:
            overlap, relation = arm.split("_", 1)
            relative = f"streams/graph_5_5_cfs2_episode{episode}_{arm}.txt"
            payload = _trainer_stream(
                bundle.updates[(overlap, relation)],
                episode_orders[episode]["unit_order"],
            )
            stream_bindings[f"episode{episode}_{arm}"] = _write_immutable(
                output_dir / relative, payload
            )

    generator_path = root / "src/cfs2/generate.py"
    validator_path = root / "src/cfs2/validate.py"
    construction = {
        "schema": "nextlat_forgetting/cfs2_construction_receipt/1",
        "status": "FROZEN",
        "successor_to": "CFS-1 stimulus construction; no CFS-1 artifact modified",
        "outcome_blind": {
            "model_imported": False,
            "model_outputs_opened": False,
            "scientific_outcomes_opened": False,
            "loss_pilot_caliper_or_learned_matching_used": False,
        },
        "generator": {
            "path": "src/cfs2/generate.py",
            "sha256": V.sha256_file(generator_path),
        },
        "validator": {
            "path": "src/cfs2/validate.py",
            "sha256": V.sha256_file(validator_path),
        },
        "counts": {
            "retention": n_probes,
            "global_controls": n_probes,
            "each_update_bank": n_updates,
            "episodes": 2,
        },
        "factorial_contract": {
            "high_same_edge_overlap": 18,
            "high_different_edge_overlap": 18,
            "low_same_edge_overlap": 8,
            "low_different_edge_overlap": 8,
            "within_overlap_future_relation_exactly_balanced": True,
            "source_goal_node_multiset_prompt_length_answer_length_balanced": True,
            "answers_equal_high_low_within_relation": True,
            "solver_validated": True,
            "legacy_and_corpus_disjoint": True,
        },
        "artifacts": {
            name: {"path": name, "sha256": digest}
            for name, digest in bindings.items()
        },
        "trainer_streams": stream_bindings,
    }
    construction_sha = _write_immutable(
        output_dir / "construction_receipt.json", V.canonical_json(construction)
    )
    # A separate execution envelope preserves the original construction and
    # materialization receipts byte-for-byte while adding execution/evaluation bindings.
    execution_generator = {
        "schema": "nextlat_forgetting/cfs2_execution_generator_receipt/1",
        "status": "FROZEN",
        "generator": {"path": "src/cfs2/generate.py", "sha256": V.sha256_file(generator_path)},
        "validator": {"path": "src/cfs2/validate.py", "sha256": V.sha256_file(validator_path)},
        "outcome_blind": True,
        "prohibited_inputs": ["checkpoint", "loss", "pilot", "caliper", "learned_distance", "scientific_outcome"],
    }
    execution_generator_sha = _write_immutable(
        output_dir / "cfs2_execution_generator_receipt.json", V.canonical_json(execution_generator)
    )
    state_commitment = {
        "schema": "nextlat_forgetting/cfs2_state_interchange_activation_patching_commitment/1",
        "status": "COMMITTED_BEFORE_CFS2_ADAPTATION",
        "role": "secondary local readout intervention; never global causal mediation proof",
        "required_after_complete_branch_matrix": True,
        "interventions": ["state_interchange", "activation_patching"],
        "named_controls": [
            "patch_parent_state_effect", "patch_unrelated_anchor_effect",
            "patch_norm_matched_random_subspace_effect",
        ],
        "construction_receipt_sha256": construction_sha,
        "scientific_outcomes_opened": False,
    }
    state_commitment_sha = _write_immutable(
        output_dir / "state_interchange_activation_patching_commitment.json", V.canonical_json(state_commitment)
    )
    episodes = []
    for episode in (0, 1):
        arms: dict[str, Any] = {}
        for arm in ARM_ORDER:
            overlap, relation = arm.split("_", 1)
            relative = f"streams/graph_5_5_cfs2_episode{episode}_{arm}.txt"
            arms[arm] = {
                **_reference(relative, stream_bindings[f"episode{episode}_{arm}"]),
                "overlap": overlap, "future_relation": relation,
                "total_edge_overlap": {"high": 18, "low": 8}[overlap],
                "provenance_manifest": _reference(
                    f"updates_{overlap}_{relation}.jsonl", bindings[f"updates_{overlap}_{relation}.jsonl"]
                ),
                "unit_order_sha256": episode_orders[episode]["unit_order_sha256"],
            }
        episodes.append({"episode": episode, "episode_sha256": episode_orders[episode]["unit_order_sha256"], "arms": arms})
    evaluation_inputs = {
        "margin": _reference("retention.jsonl", bindings["retention.jsonl"]),
        "retention_ce": _reference("retention.jsonl", bindings["retention.jsonl"]),
        "retention_exact_path": _reference("retention.jsonl", bindings["retention.jsonl"]),
        "global_controls": _reference("global_controls.jsonl", bindings["global_controls.jsonl"]),
        "state_drift": _reference("retention.jsonl", bindings["retention.jsonl"]),
        "pregeometry": _reference("retention.jsonl", bindings["retention.jsonl"]),
    }
    if tuple(evaluation_inputs) != EVALUATION_INPUT_ORDER:
        raise AssertionError("CFS-2 evaluation-input order drift")
    update_manifest = {
        "schema": "nextlat_forgetting/cfs2_update_manifest/1", "status": "FROZEN",
        "construction": {
            "model_outcomes_inspected": False, "training_outcomes_inspected": False,
            "retention_outcomes_inspected": False, "matching": "construction_matched",
            "randomized_assignment": True,
            "receipt": _reference("construction_receipt.json", construction_sha),
            "exact_total_edge_overlap": {"high_same": 18, "high_different": 18, "low_same": 8, "low_different": 8},
        },
        "generator_receipt": _reference("cfs2_execution_generator_receipt.json", execution_generator_sha),
        "generator_manifest": _reference("hash_codebook.json", bindings["hash_codebook.json"]),
        "retention_probes": _reference("retention.jsonl", bindings["retention.jsonl"]),
        "global_control_manifest": _reference("global_controls.jsonl", bindings["global_controls.jsonl"]),
        "design": {"model": "nextlat", "adaptation_steps": 500, "full_parameter": True,
                   "loss": "teacher_forced_next_token_cross_entropy", "arms": list(ARM_ORDER), "episodes": [0, 1]},
        "execution_order_algorithm": "sha256-sort-v1",
        "execution_order_salt_sha256": V.sha256_bytes(ORDER_SALT.encode("utf-8")),
        "evaluation_inputs": evaluation_inputs,
        "evaluation_input_order": list(EVALUATION_INPUT_ORDER),
        "state_interchange_activation_patching": _reference("state_interchange_activation_patching_commitment.json", state_commitment_sha),
        "episodes": episodes,
        "parent_identity": {"seeds": [1234, 1235, 1236, 1237, 1238, 2234, 2235, 2236],
                            "canonical_cfs2_only_parent_suffix": "-cfs2-base",
                            "cfs_only_alias_requires_hash_bound_lineage_receipt": True},
        "branch_plan": _branch_plan(),
        "successor_to": "CFS-1 repaired stimulus study; CFS-1 streams are prohibited",
    }
    update_manifest_sha = _write_immutable(
        output_dir / "cfs2_update_manifest.json", V.canonical_json(update_manifest)
    )
    execution_receipt = {
        "schema": "nextlat_forgetting/cfs2_execution_envelope_receipt/1", "status": "PASS",
        "construction_receipt_sha256": construction_sha,
        "execution_generator_receipt_sha256": execution_generator_sha,
        "state_interchange_activation_patching_commitment_sha256": state_commitment_sha,
        "cfs2_update_manifest_sha256": update_manifest_sha,
        "exact_total_edge_overlap": update_manifest["construction"]["exact_total_edge_overlap"],
        "n_branches": 64, "scientific_outcomes_opened": False,
    }
    execution_receipt_sha = _write_immutable(
        output_dir / "cfs2_execution_envelope_receipt.json", V.canonical_json(execution_receipt)
    )
    materialization = {
        "schema": V.RECEIPT_SCHEMA,
        "status": "PASS",
        "construction_receipt_sha256": construction_sha,
        "generator_sha256": V.sha256_file(generator_path),
        "validator_sha256": V.sha256_file(validator_path),
        "retention_sha256": bindings["retention.jsonl"],
        "global_controls_sha256": bindings["global_controls.jsonl"],
        "adaptation_sha256": {
            name: digest
            for name, digest in bindings.items()
            if name.startswith("updates_")
        },
        "trainer_stream_sha256": stream_bindings,
        "codebook_sha256": bindings["hash_codebook.json"],
    }
    receipt_sha = _write_immutable(
        output_dir / "materialization_receipt.json",
        V.canonical_json(materialization),
    )
    return {
        "status": "PASS",
        "output_dir": str(output_dir),
        "construction_receipt_sha256": construction_sha,
        "materialization_receipt_sha256": receipt_sha,
        "cfs2_update_manifest_sha256": update_manifest_sha,
        "cfs2_execution_envelope_receipt_sha256": execution_receipt_sha,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument(
        "--output-dir", type=pathlib.Path, default=ROOT / "manifests/cfs2"
    )
    parser.add_argument("--n-probes", type=int, default=G.N_PROBES)
    parser.add_argument("--n-updates", type=int, default=G.N_UPDATES)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = materialize(
        root=args.root.resolve(),
        output_dir=args.output_dir.resolve(),
        n_probes=args.n_probes,
        n_updates=args.n_updates,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except V.CFS2ValidationError as exc:
        raise SystemExit(f"BLOCK: {exc}")
