#!/usr/bin/env python3
"""Materialize immutable, model-blind CFS-1 retention and update manifests."""

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

from cfs1 import generate as G  # noqa: E402
from cfs1 import validate as V  # noqa: E402


# This order is an externally consumed experimental contract (see cfs1.adaptation),
# not a presentation preference.  The outer envelope is intentionally serialized
# without key sorting so this order survives JSON decoding.
ARM_ORDER = ("high_different", "low_different", "high_same", "low_same")
EVALUATION_INPUT_ORDER = (
    "margin", "retention_ce", "retention_exact_path", "global_controls",
    "state_drift", "pregeometry",
)
ORDER_SALT = "20260824|unit-order"


def _jsonl(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(V.canonical_json(row) for row in rows)


def _ordered_json(document: Mapping[str, Any]) -> bytes:
    """Serialize the public runner envelope with its contractual mapping order."""
    return (json.dumps(document, sort_keys=False, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def _reference(path: str, digest: str) -> dict[str, str]:
    return {"path": path, "sha256": digest}


def _trainer_stream(rows: Sequence[Mapping[str, Any]], unit_order: Sequence[str]) -> bytes:
    by_unit = {str(row["unit_id"]): row for row in rows}
    if set(by_unit) != set(unit_order) or len(by_unit) != len(unit_order):
        raise V.CFS1ValidationError("trainer stream cannot bind every hash-ordered unit exactly once")
    return ("\n".join(str(by_unit[unit]["line"]) for unit in unit_order) + "\n").encode()


def _write_immutable(path: pathlib.Path, payload: bytes) -> str:
    """Create once; an identical rerun is accepted, divergent bytes are refused."""
    digest = V.sha256_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise V.CFS1ValidationError(f"refusing to overwrite immutable artifact {path}")
    else:
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != payload:
                    raise V.CFS1ValidationError(f"concurrent immutable write disagreed for {path}")
            finally:
                pathlib.Path(temporary).unlink(missing_ok=True)
        except BaseException:
            pathlib.Path(temporary).unlink(missing_ok=True)
            raise
    sidecar = f"{digest}  {path.name}\n".encode()
    sidecar_path = pathlib.Path(f"{path}.sha256")
    if sidecar_path.exists() and sidecar_path.read_bytes() != sidecar:
        raise V.CFS1ValidationError(f"refusing to overwrite immutable sidecar {sidecar_path}")
    if not sidecar_path.exists():
        sidecar_path.write_bytes(sidecar)
    return digest


def _branch_plan() -> list[dict[str, Any]]:
    planned = [f"nextlat-seed{seed}-base" for seed in range(1234, 1239)]
    cfs_only = [f"nextlat-seed{seed}-cfs1-base" for seed in range(2234, 2237)]
    return [
        {
            "parent_id": parent_id,
            "parent_role": "planned_base" if parent_id in planned else "cfs1_only_base",
            "episode": episode,
            "overlap": overlap,
            "future_relation": future_relation,
            "branch_id": f"cfs1-{parent_id}-e{episode}-{overlap}-{future_relation}",
            "adaptation_steps": 500,
        }
        for parent_id in planned + cfs_only
        for episode in (0, 1)
        for overlap, future_relation in V.CONDITIONS
    ]


def materialize(*, root: pathlib.Path, output_dir: pathlib.Path, n_probes: int, n_updates: int, dry_run: bool) -> dict[str, Any]:
    legacy = V.build_legacy_index(root)
    bundle = G.build_bundle(n_probes=n_probes, n_updates=n_updates, legacy=legacy)
    retention, updates = bundle.retention, bundle.updates
    global_controls, codebook = bundle.global_controls, bundle.codebook
    if dry_run:
        return {
            "schema": "nextlat_forgetting/cfs1_no_model_dry_run/1",
            "status": "PASS",
            "model_imported": False,
            "model_outputs_opened": False,
            "loss_or_pilot_selection_used": False,
            "counts": {"retention": len(retention), "each_update_bank": n_updates, "global_controls": len(global_controls)},
            "legacy_sources_indexed": len(legacy.sources),
        }
    payloads: dict[str, bytes] = {
        "retention.jsonl": _jsonl(retention),
        "global_controls.jsonl": _jsonl(global_controls),
        "hash_codebook.json": V.canonical_json(codebook),
    }
    for overlap, future_relation in V.CONDITIONS:
        payloads[f"updates_{overlap}_{future_relation}.jsonl"] = _jsonl(updates[(overlap, future_relation)])
    bindings = {name: _write_immutable(output_dir / name, payload) for name, payload in payloads.items()}
    episode_orders = {int(item["episode"]): item for item in codebook["episodes"]}
    stream_bindings: dict[tuple[int, str], str] = {}
    for episode in (0, 1):
        unit_order = episode_orders[episode]["unit_order"]
        for arm in ARM_ORDER:
            overlap, future_relation = arm.split("_", 1)
            payload = _trainer_stream(updates[(overlap, future_relation)], unit_order)
            relative = f"streams/graph_5_5_cfs1_episode{episode}_{arm}.txt"
            stream_bindings[(episode, arm)] = _write_immutable(output_dir / relative, payload)
    generator = root / "src/cfs1/generate.py"
    validator = root / "src/cfs1/validate.py"
    generator_receipt = {
        "schema": "nextlat_forgetting/cfs1_generator_receipt/1",
        "generator": {"path": "src/cfs1/generate.py", "sha256": V.sha256_file(generator)},
        "validator": {"path": "src/cfs1/validate.py", "sha256": V.sha256_file(validator)},
        "model_blind": True,
        "prohibited_inputs": ["checkpoint", "loss", "pilot", "caliper", "middle_match", "learned_distance", "scientific_outcome"],
    }
    generator_receipt_sha = _write_immutable(output_dir / "generator_receipt.json", V.canonical_json(generator_receipt))
    generator_manifest = {
        "schema": "nextlat_forgetting/cfs1_generator_manifest/1",
        "generator_receipt": _reference("generator_receipt.json", generator_receipt_sha),
        "generator": generator_receipt["generator"],
        "validator": generator_receipt["validator"],
        "hash_codebook": _reference("hash_codebook.json", bindings["hash_codebook.json"]),
        "construction_mode": "model_blind_solver_verified_construction_matched",
    }
    generator_manifest_sha = _write_immutable(output_dir / "generator_manifest.json", V.canonical_json(generator_manifest))
    construction_receipt = {
        "schema": "nextlat_forgetting/cfs1_construction_receipt/1",
        "status": "FROZEN",
        "outcome_blind": {
            "model_outcomes_inspected": False,
            "training_outcomes_inspected": False,
            "retention_outcomes_inspected": False,
        },
        "generator_receipt": _reference("generator_receipt.json", generator_receipt_sha),
        "retention": {**_reference("retention.jsonl", bindings["retention.jsonl"]), "count": n_probes},
        "adaptation_arms": {
            f"{overlap}_{future_relation}": {
                **_reference(f"updates_{overlap}_{future_relation}.jsonl", bindings[f"updates_{overlap}_{future_relation}.jsonl"]), "count": n_updates,
            }
            for overlap, future_relation in V.CONDITIONS
        },
        "global_controls": {**_reference("global_controls.jsonl", bindings["global_controls.jsonl"]), "count": n_probes},
        "hash_codebook": _reference("hash_codebook.json", bindings["hash_codebook.json"]),
        "construction": {
            "model_blind": True,
            "no_loss_pilot_caliper_or_middle_matching": True,
            "solver_validated": True,
            "same_relation_answer_balanced_high_low": True,
            "token_balanced_high_low_x_same_different": True,
        },
    }
    construction_receipt_sha = _write_immutable(output_dir / "construction_receipt.json", V.canonical_json(construction_receipt))
    episodes = []
    for episode in (0, 1):
        arms: dict[str, Any] = {}
        for arm in ARM_ORDER:
            overlap, future_relation = arm.split("_", 1)
            arms[arm] = {
                **_reference(f"streams/graph_5_5_cfs1_episode{episode}_{arm}.txt", stream_bindings[(episode, arm)]),
                "overlap": overlap,
                "future_relation": future_relation,
                "provenance_manifest": _reference(
                    f"updates_{overlap}_{future_relation}.jsonl",
                    bindings[f"updates_{overlap}_{future_relation}.jsonl"],
                ),
                "unit_order_sha256": episode_orders[episode]["unit_order_sha256"],
            }
        episodes.append({
            "episode": episode,
            "episode_sha256": episode_orders[episode]["unit_order_sha256"],
            "arms": arms,
        })
    evaluation_inputs = {
        "margin": _reference("retention.jsonl", bindings["retention.jsonl"]),
        "retention_ce": _reference("retention.jsonl", bindings["retention.jsonl"]),
        "retention_exact_path": _reference("retention.jsonl", bindings["retention.jsonl"]),
        "global_controls": _reference("global_controls.jsonl", bindings["global_controls.jsonl"]),
        "state_drift": _reference("retention.jsonl", bindings["retention.jsonl"]),
        "pregeometry": _reference("retention.jsonl", bindings["retention.jsonl"]),
    }
    if tuple(evaluation_inputs) != EVALUATION_INPUT_ORDER:
        raise AssertionError("CFS-1 evaluation input order drift")
    # This opaque envelope is the sole evaluator/runner input surface. It binds six
    # paired evaluator inputs and the trainer-ready raw Path-Star stream per arm/episode.
    outer_manifest = {
        "schema": V.UPDATE_SCHEMA,
        "status": "FROZEN",
        "construction": {
            "model_outcomes_inspected": False,
            "training_outcomes_inspected": False,
            "retention_outcomes_inspected": False,
            "matching": "construction_matched",
            "randomized_assignment": True,
            "receipt": _reference("construction_receipt.json", construction_receipt_sha),
        },
        "generator_receipt": _reference("generator_receipt.json", generator_receipt_sha),
        "generator_manifest": _reference("generator_manifest.json", generator_manifest_sha),
        "retention_probes": _reference("retention.jsonl", bindings["retention.jsonl"]),
        "global_control_manifest": _reference("global_controls.jsonl", bindings["global_controls.jsonl"]),
        "design": {
            "model": "nextlat", "adaptation_steps": 500, "full_parameter": True,
            "loss": "teacher_forced_next_token_cross_entropy", "arms": list(ARM_ORDER), "episodes": [0, 1],
        },
        "execution_order_algorithm": "sha256-sort-v1",
        "execution_order_salt_sha256": V.sha256_bytes(ORDER_SALT.encode()),
        "evaluation_inputs": evaluation_inputs,
        "episodes": episodes,
        "parent_roster": [
            {"parent_id": f"nextlat-seed{seed}-base", "parent_role": "planned_base"}
            for seed in range(1234, 1239)
        ] + [
            {"parent_id": f"nextlat-seed{seed}-cfs1-base", "parent_role": "cfs1_only_base"}
            for seed in range(2234, 2237)
        ],
        "branch_plan": _branch_plan(),
        "sha_sort": {
            "algorithm": "sha256",
            "salt": "20260824|unit-order",
            "order": "ascending SHA256(UTF-8 salt|probe_index|occurrence), then numeric pair",
            "codebook_sha256": bindings["hash_codebook.json"],
        },
    }
    outer_sha = _write_immutable(output_dir / "cfs1_update_manifest.json", _ordered_json(outer_manifest))
    receipt = {
        "schema": V.RECEIPT_SCHEMA,
        "status": "PASS",
        "generator_sha256": V.sha256_file(generator),
        "validator_sha256": V.sha256_file(validator),
        "generator_receipt_sha256": generator_receipt_sha,
        "generator_manifest_sha256": generator_manifest_sha,
        "construction_receipt_sha256": construction_receipt_sha,
        "retention_sha256": bindings["retention.jsonl"],
        "adaptation_sha256": {name: digest for name, digest in bindings.items() if name.startswith("updates_")},
        "trainer_stream_sha256": {f"episode{episode}_{arm}": digest for (episode, arm), digest in stream_bindings.items()},
        "global_controls_sha256": bindings["global_controls.jsonl"],
        "cfs1_update_manifest_sha256": outer_sha,
        "codebook_sha256": bindings["hash_codebook.json"],
        "no_model_dry_run": {"model_imported": False, "model_outputs_opened": False, "loss_or_pilot_selection_used": False},
    }
    receipt_sha = _write_immutable(output_dir / "materialization_receipt.json", V.canonical_json(receipt))
    return {"status": "PASS", "output_dir": str(output_dir), "cfs1_update_manifest_sha256": outer_sha, "receipt_sha256": receipt_sha}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--output-dir", type=pathlib.Path, default=ROOT / "manifests/cfs1")
    parser.add_argument("--n-probes", type=int, default=G.N_PROBES)
    parser.add_argument("--n-updates", type=int, default=G.N_UPDATES)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = materialize(root=args.root.resolve(), output_dir=args.output_dir.resolve(), n_probes=args.n_probes, n_updates=args.n_updates, dry_run=args.dry_run)
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except V.CFS1ValidationError as exc:
        raise SystemExit(f"BLOCK: {exc}")
