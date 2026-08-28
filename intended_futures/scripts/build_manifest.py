#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from intended_futures.config import load_config
    from intended_futures.manifest import build_matched_manifest, write_manifest

    manifest = build_matched_manifest(load_config(args.config), args.libero_root)
    write_manifest(args.output, manifest)
    print(f"wrote {len(manifest['rows'])} matched stimuli to {args.output}")
    print(f"manifest_sha256={manifest['manifest_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
