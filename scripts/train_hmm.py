#!/usr/bin/env python
"""Register the external HMM datamodule and delegate to pinned NextLat training.

Run this script with the upstream repository as the working directory so its ``defaults.yaml``
and import layout remain authoritative, for example::

    cd /content/nextlat
    fabric run --devices 1 --precision bf16-mixed \
      /content/lurestar/scripts/train_hmm.py --config /content/lurestar/configs/gpt_hmm.yaml

No file below ``upstream/`` is changed; registration mutates only the imported module's in-memory
``DATAMODULES`` dictionary.
"""
from __future__ import annotations

import argparse
import importlib
import pathlib
import sys
import typing as t

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _upstream_layout(path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    root = path.expanduser().resolve()
    train_py = root / "train.py"
    defaults = root / "defaults.yaml"
    if not train_py.is_file() or not defaults.is_file():
        raise FileNotFoundError(
            f"expected pinned NextLat train.py and defaults.yaml under {root}; "
            "run the shim from the NextLat repository or pass --upstream-root"
        )
    return root, defaults


def _import_upstream_train(upstream_root: pathlib.Path):
    src = str(PROJECT_ROOT / "src")
    upstream = str(upstream_root)
    if src not in sys.path:
        sys.path.insert(0, src)
    if upstream not in sys.path:
        sys.path.insert(0, upstream)
    return importlib.import_module("train")


def register_hmm(upstream_train) -> None:
    from hmm_geometry.datamodule import HMMBeliefDataModule

    upstream_train.DATAMODULES["hmm_belief"] = HMMBeliefDataModule


def _load_merged_config(config_path: pathlib.Path, defaults_path: pathlib.Path,
                        overrides: t.Sequence[str]):
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:  # pragma: no cover - runtime dependency, absent on test host
        raise RuntimeError("HMM training requires the pinned NextLat OmegaConf dependency") from exc

    config_path = config_path.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"HMM config not found: {config_path}")
    base = OmegaConf.load(config_path)
    if "sweep" in base and base.sweep is not None:
        raise ValueError(
            "scripts/train_hmm.py intentionally does not run sweep configs; launch each "
            "preregistered seed as an explicit job"
        )
    return OmegaConf.merge(
        OmegaConf.load(defaults_path), base, OmegaConf.from_dotlist(list(overrides))
    )


def main(argv: t.Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train NextLat/GPT on the frozen HMM corpus")
    parser.add_argument("-c", "--config", required=True, type=pathlib.Path)
    parser.add_argument("--no_pbar", action="store_true")
    parser.add_argument("--shard", action="store_true")
    parser.add_argument("--checkpoint_path")
    parser.add_argument(
        "--upstream-root",
        type=pathlib.Path,
        default=pathlib.Path.cwd(),
        help="directory containing pinned train.py and defaults.yaml (default: current directory)",
    )
    args, overrides = parser.parse_known_args(argv)

    upstream_root, defaults_path = _upstream_layout(args.upstream_root)
    upstream_train = _import_upstream_train(upstream_root)
    register_hmm(upstream_train)
    config = _load_merged_config(args.config, defaults_path, overrides)
    upstream_train.do_train(
        config,
        hide_progress_bar=args.no_pbar,
        use_sharding=args.shard,
        checkpoint_path=args.checkpoint_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
