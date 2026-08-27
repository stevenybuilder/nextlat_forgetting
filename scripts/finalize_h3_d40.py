#!/usr/bin/env python3
"""Integrated D40 finalizer: one-shot mid plus unchanged D39 far/acquisition outputs."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lurestar import h3_expansion as E  # noqa: E402
from lurestar import h3_precompute as H  # noqa: E402


def auxiliary_d39(*, root: pathlib.Path, loss_table: pathlib.Path,
                  candidate_dir: pathlib.Path, output_dir: pathlib.Path) -> dict[str, Any]:
    """Materialize only the unchanged D39 far/acquisition choices, never the failed D39 mid."""
    if H.verify_sidecar(loss_table) != E.D39_LOSS_SHA256:
        raise E.ExpansionRefused("D40 auxiliary selection requires the exact complete D39 loss table")
    near_path, far_path = root / "manifests/b_near.jsonl", root / "manifests/b_far.jsonl"
    near, far = E._rows(near_path), E._rows(far_path)
    if len(near) != H.MID_COUNT or len(far) != H.MID_CANDIDATE_COUNT:
        raise E.ExpansionRefused("D39 far inputs no longer have exact 5k/15k counts")
    losses, loss_sha = H.load_loss_table(loss_table)
    near_rank, far_rank = H._rank(near, losses), H._rank(far, losses)
    far_order = sorted(far, key=lambda row: far_rank[str(row["prompt_sha256"])][0])
    far_selection = []
    for near_row in near:
        nsha = str(near_row["prompt_sha256"])
        rank, quantile, _ = near_rank[nsha]
        chosen = far_order[3 * rank + 1]
        fsha = str(chosen["prompt_sha256"])
        if abs(far_rank[fsha][1] - quantile) > 1e-15:
            raise E.ExpansionRefused("unchanged D39 far quantile identity failed")
        far_selection.append({
            "near_prompt_sha256": nsha, "far_prompt_sha256": fsha,
            "near_loss_quantile": quantile, "far_loss_quantile": quantile,
        })
    freeze_path = candidate_dir / "pilot_freeze.json"
    freeze_sha = H.verify_sidecar(freeze_path)
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze != H.pilot_freeze_payload(
        generator_sha256=H.sha256_file(H.__file__),
        scorer_sha256=H.sha256_file(root / "scripts/score_h3_pilot.py"),
        tokenizer_sha256=H.sha256_file(root / "upstream/NextLat/data/stargraph.py"),
        adaptation_contract_sha256=H.sha256_file(root / "src/lurestar/adaptation.py"),
    ):
        raise E.ExpansionRefused("D39 pilot freeze is stale or does not designate the sole pilot")
    pilot = H._pilot_provenance(freeze_sha, loss_sha, H.sha256_file(H.__file__))
    far_artifact = {
        "schema_version": H.SELECTION_SCHEMA_VERSION,
        "purpose": "h3_far_loss_quantile_match",
        "selection_method": "non_confirmatory_pilot_loss_quantile_match",
        "near_bank_sha256": H.verify_sidecar(near_path),
        "candidate_bank_sha256": H.verify_sidecar(far_path),
        "pilot": pilot, "selection": far_selection,
    }

    acquisition_selected: dict[str, list[dict[str, Any]]] = {}
    acquisition_hashes: dict[str, str] = {}
    acquisition_inputs: dict[str, str] = {}
    for branch in ("near", "mid", "far"):
        path = candidate_dir / f"acquisition_{branch}_candidates.jsonl"
        rows = E._rows(path)
        if len(rows) != H.ACQUISITION_CANDIDATE_COUNT:
            raise E.ExpansionRefused(f"{branch} acquisition pool is not exact 6,000")
        ranks = H._rank(rows, losses)
        by_decile: dict[int, list[dict[str, Any]]] = {value: [] for value in range(10)}
        for row in rows:
            by_decile[ranks[str(row["prompt_sha256"])][2]].append(row)
        chosen: list[dict[str, Any]] = []
        for decile in range(10):
            ordered = sorted(by_decile[decile], key=lambda row: str(row["prompt_sha256"]))
            if len(ordered) < H.ACQUISITION_COUNT // 10:
                raise E.ExpansionRefused(f"{branch} acquisition decile {decile} is infeasible")
            chosen.extend(ordered[: H.ACQUISITION_COUNT // 10])
        chosen.sort(key=lambda row: str(row["item_id"]))
        acquisition_selected[branch] = chosen
        acquisition_hashes[branch] = H.create_or_verify(
            output_dir / f"acquisition_{branch}.jsonl", H._jsonl_payload(chosen)
        )
        acquisition_inputs[branch] = H.verify_sidecar(path)
    identity_sets = {
        key: {str(row["prompt_sha256"]) for row in rows}
        for key, rows in acquisition_selected.items()
    }
    if any(identity_sets[a] & identity_sets[b]
           for a, b in (("near", "mid"), ("near", "far"), ("mid", "far"))):
        raise E.ExpansionRefused("unchanged D39 acquisition selections overlap")
    far_sha = H.create_or_verify(output_dir / "far_selection.json", H.canonical_json(far_artifact))
    acquisition = {
        "schema_version": H.SELECTION_SCHEMA_VERSION,
        "purpose": "h3_independent_acquisition_banks",
        "selection_method": "model_blind_structural_then_frozen_pilot_loss_decile",
        "frozen_before_confirmatory": True,
        "inspected_confirmatory_checkpoints": False,
        "inspected_confirmatory_results": False,
        "optimized_h3_outcomes": False,
        "disjoint_from_training": True,
        "matched_target_path_distribution": True,
        "matched_pilot_loss_deciles": True,
        "selector_code_sha256": H.sha256_file(H.__file__),
        "bank_sha256": acquisition_hashes,
        "counts": {branch: H.ACQUISITION_COUNT for branch in acquisition_hashes},
        "candidate_bank_sha256": acquisition_inputs,
        "pilot": pilot,
        "per_decile_counts": {
            branch: {str(decile): H.ACQUISITION_COUNT // 10 for decile in range(10)}
            for branch in acquisition_hashes
        },
    }
    acquisition_sha = H.create_or_verify(
        output_dir / "acquisition_provenance.json", H.canonical_json(acquisition)
    )
    receipt = {
        "schema": E.SCHEMA, "status": "D40_UNCHANGED_D39_AUXILIARY_SELECTIONS_FROZEN",
        "confirmatory_results_inspected": False,
        "d39_loss_sha256": loss_sha, "d39_pilot_freeze_sha256": freeze_sha,
        "outputs": {"far_selection.json": far_sha,
                    "acquisition_provenance.json": acquisition_sha,
                    **{f"acquisition_{key}.jsonl": value for key, value in acquisition_hashes.items()}},
    }
    H.create_or_verify(output_dir / "d40_auxiliary_receipt.json", H.canonical_json(receipt))
    return receipt


def finalize(*, root: pathlib.Path, candidate_dir: pathlib.Path, loss_table: pathlib.Path,
             combined_loss: pathlib.Path, combined_receipt: pathlib.Path,
             output_dir: pathlib.Path) -> dict[str, Any]:
    mid = E.select_mid(
        root=root, expanded_manifest=root / "manifests/h3_expansion/b_mid_expanded_150000.jsonl",
        combined_loss=combined_loss, combined_receipt=combined_receipt, output_dir=output_dir,
    )
    auxiliary = auxiliary_d39(
        root=root, loss_table=loss_table, candidate_dir=candidate_dir, output_dir=output_dir,
    )
    receipt = {
        "schema": E.SCHEMA, "status": "D40_SIX_BANK_SELECTION_INPUTS_FROZEN",
        "mid_selection_sha256": H.verify_sidecar(output_dir / "mid_selection_d40.json"),
        "mid_receipt_sha256": H.verify_sidecar(output_dir / "selection_receipt.json"),
        "auxiliary_receipt_sha256": H.verify_sidecar(output_dir / "d40_auxiliary_receipt.json"),
        "mid": mid, "auxiliary": auxiliary,
        "confirmatory_results_inspected": False,
    }
    H.create_or_verify(output_dir / "d40_finalize_receipt.json", H.canonical_json(receipt))
    return receipt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("plan", "auxiliary", "finalize"), required=True)
    ap.add_argument("--root", type=pathlib.Path, default=ROOT)
    ap.add_argument("--candidate-dir", type=pathlib.Path,
                    default=ROOT / "manifests/h3_precompute")
    ap.add_argument("--loss-table", type=pathlib.Path,
                    default=ROOT / ".agent_state/pilot/h3_pilot_score/pilot_losses.jsonl")
    ap.add_argument("--combined-loss", type=pathlib.Path,
                    default=ROOT / "manifests/h3_expansion/combined_pilot_losses_188000.jsonl")
    ap.add_argument("--combined-receipt", type=pathlib.Path,
                    default=ROOT / "manifests/h3_expansion/combined_loss_receipt.json")
    ap.add_argument("--output-dir", type=pathlib.Path, default=ROOT / "manifests/h3_selected")
    args = ap.parse_args(argv)
    values = {key: value.resolve() for key, value in vars(args).items() if isinstance(value, pathlib.Path)}
    if args.mode == "plan":
        result = {"schema": E.SCHEMA, "mode": "READ_ONLY_PLAN", "gpu_launched": False,
                  "sequence": ["validate D40 combined lineage", "select D40 mid exactly 5000",
                               "materialize unchanged D39 far", "materialize 3 D39 acquisition banks",
                               "publish integrated receipt last"]}
    elif args.mode == "auxiliary":
        result = auxiliary_d39(root=values["root"], loss_table=values["loss_table"],
                               candidate_dir=values["candidate_dir"], output_dir=values["output_dir"])
    else:
        result = finalize(root=values["root"], candidate_dir=values["candidate_dir"],
                          loss_table=values["loss_table"], combined_loss=values["combined_loss"],
                          combined_receipt=values["combined_receipt"], output_dir=values["output_dir"])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (E.ExpansionRefused, H.PrecomputeRefused) as exc:
        raise SystemExit(f"BLOCK: {exc}")
