#!/usr/bin/env python
"""Create the immutable, fail-closed GPT-vs-NextLat HMM family aggregate."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from hmm_geometry.aggregate import (  # noqa: E402
    MODELS,
    SEEDS,
    HMMAggregationError,
    aggregate,
    load_complete_receipts,
)
from hmm_geometry.family import REGIMES  # noqa: E402
from lurestar.durable_checkpoint import atomic_write_json, atomic_write_text, sha256_file  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", action="append", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    try:
        expected_cells = len(REGIMES) * len(MODELS) * len(SEEDS)
        if len(args.receipt) != expected_cells:
            raise HMMAggregationError(
                f"exactly {expected_cells} frozen receipt arguments are required; "
                "operational recovery subsets cannot produce a confirmatory aggregate"
            )
        receipts = load_complete_receipts(args.receipt)
        payload = aggregate(receipts)
        if args.output.exists():
            existing = json.loads(args.output.read_text(encoding="utf-8"))
            if existing != payload:
                raise HMMAggregationError("refusing to replace a different frozen aggregate")
        else:
            atomic_write_json(args.output, payload)
        sidecar = args.output.with_name(args.output.name + ".sha256")
        expected = f"{sha256_file(args.output)}  {args.output.name}\n"
        if sidecar.exists() and sidecar.read_text(encoding="utf-8") != expected:
            raise HMMAggregationError("aggregate sidecar disagrees with output")
        if not sidecar.exists():
            atomic_write_text(sidecar, expected)
    except (HMMAggregationError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"REFUSED/FAILED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
