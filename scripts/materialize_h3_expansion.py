#!/usr/bin/env python3
"""Plan, generate, combine, or select the prospective D40 middle expansion."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lurestar import h3_expansion as E  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("plan", "generate", "combine", "select"), required=True)
    ap.add_argument("--root", type=pathlib.Path, default=ROOT)
    ap.add_argument("--candidate-dir", type=pathlib.Path, default=ROOT / "manifests/h3_expansion")
    ap.add_argument("--original-loss", type=pathlib.Path,
                    default=ROOT / ".agent_state/pilot/h3_pilot_score/pilot_losses.jsonl")
    ap.add_argument("--expansion-loss", type=pathlib.Path)
    ap.add_argument("--combined-loss", type=pathlib.Path)
    ap.add_argument("--scoring-receipt", type=pathlib.Path)
    ap.add_argument("--durable-state", type=pathlib.Path)
    ap.add_argument("--combined-receipt", type=pathlib.Path)
    ap.add_argument("--selection-dir", type=pathlib.Path, default=ROOT / "manifests/h3_selected")
    args = ap.parse_args(argv)
    root, candidate_dir = args.root.resolve(), args.candidate_dir.resolve()
    if args.mode == "plan":
        result = E.plan(root, candidate_dir)
    elif args.mode == "generate":
        result = E.generate(root=root, output_dir=candidate_dir)
    elif args.mode == "combine":
        if args.expansion_loss is None or args.scoring_receipt is None or args.durable_state is None:
            raise E.ExpansionRefused(
                "--mode combine requires --expansion-loss, --scoring-receipt, and --durable-state"
            )
        result = E.combine_losses(
            original_loss=args.original_loss.resolve(),
            expansion_loss=args.expansion_loss.resolve(),
            expanded_manifest=candidate_dir / "b_mid_expanded_150000.jsonl",
            scoring_receipt=args.scoring_receipt.resolve(),
            durable_state=args.durable_state.resolve(),
            generation_receipt=candidate_dir / "generation_receipt.json",
            generation_domain_receipt=candidate_dir / "generation_domain_receipt.json",
            output_dir=candidate_dir,
        )
    else:
        combined = args.combined_loss or candidate_dir / "combined_pilot_losses_188000.jsonl"
        combined_receipt = args.combined_receipt or candidate_dir / "combined_loss_receipt.json"
        result = E.select_mid(
            root=root, expanded_manifest=candidate_dir / "b_mid_expanded_150000.jsonl",
            combined_loss=combined.resolve(), combined_receipt=combined_receipt.resolve(),
            output_dir=args.selection_dir.resolve(),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except E.ExpansionRefused as exc:
        raise SystemExit(f"BLOCK: {exc}")
