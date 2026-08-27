#!/usr/bin/env python3
"""Plan, generate, or select the immutable H3 pilot pre-compute artifacts."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lurestar import h3_precompute as H  # noqa: E402


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("plan", "generate", "select"), required=True)
    ap.add_argument("--plan", action="store_true", help="read-only alias; requires --mode plan")
    ap.add_argument("--root", type=pathlib.Path, default=ROOT)
    ap.add_argument("--candidate-dir", type=pathlib.Path, default=ROOT / "manifests/h3_precompute")
    ap.add_argument("--selection-dir", type=pathlib.Path, default=ROOT / "manifests/h3_selected")
    ap.add_argument("--loss-table", type=pathlib.Path)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.plan and args.mode != "plan":
        raise H.PrecomputeRefused("--plan is read-only and cannot accompany a write mode")
    root = args.root.resolve()
    if args.mode == "plan":
        result = H.plan(root=root, candidate_dir=args.candidate_dir.resolve())
    elif args.mode == "generate":
        result = H.materialize_candidates(
            root=root,
            output_dir=args.candidate_dir.resolve(),
            scorer_path=root / "scripts/score_h3_pilot.py",
        )
    else:
        if args.loss_table is None:
            raise H.PrecomputeRefused("--mode select requires --loss-table")
        result = H.select_outputs(
            root=root,
            candidate_dir=args.candidate_dir.resolve(),
            loss_table=args.loss_table.resolve(),
            output_dir=args.selection_dir.resolve(),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except H.PrecomputeRefused as exc:
        raise SystemExit(f"BLOCK: {exc}")
