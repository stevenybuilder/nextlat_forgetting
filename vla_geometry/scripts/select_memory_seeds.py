#!/usr/bin/env python3
"""Resolve preregistered seeds by official Task-16 factors without querying the policy.

Selection depends only on the pinned VIMA-Bench generator's target shape, receptacle shape, and
neighbor direction. No action, reward, success, or model output is evaluated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from vla_geometry.grid import build_cells, get_factor_order, load_config
    from vla_geometry.seeds import resolve_all_seed_maps

    config = load_config(args.config)
    order = get_factor_order(config)
    cells = build_cells(config["factors"], order)
    manifest = resolve_all_seed_maps(config, cells)

    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(json.dumps({"seed_manifest": str(args.output), "cells": len(cells)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
