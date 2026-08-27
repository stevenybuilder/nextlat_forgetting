#!/usr/bin/env python3
"""Materialize the exact eight-parent, hash-bound CFS-2 lineage receipt."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile
import typing as t


REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

from cfs2.adaptation import canonical_json_sha256, sha256_file  # noqa: E402
from run_cfs2_matrix import (  # noqa: E402
    CFS2_PARENT_LINEAGE_SCHEMA_V2,
    PARENT_STEPS,
    load_parents,
    parent_id_for_seed,
)
from run_cfs2_parents import CFS2_PARENT_SEEDS, source_parent_id  # noqa: E402


ALL_PARENT_SEEDS = (1234, 1235, 1236, 1237, 1238, *CFS2_PARENT_SEEDS)
PARENT_LEDGER_SCHEMA = "nextlat_forgetting/cfs2_parent_ledger/1"


class CFS2LineageError(RuntimeError):
    """The source ledgers cannot identify the frozen eight-parent roster."""


def expected_source_id(seed: int) -> str:
    if seed in CFS2_PARENT_SEEDS:
        return source_parent_id(seed)
    if seed in ALL_PARENT_SEEDS:
        return f"nextlat-s{seed}-base"
    raise CFS2LineageError(f"seed {seed} is outside the frozen eight-parent roster")


def _read_ledger(path: pathlib.Path) -> list[dict[str, t.Any]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CFS2LineageError(f"unreadable source ledger: {path}") from exc
    entries = document.get("entries")
    if not isinstance(entries, list) or any(not isinstance(entry, dict) for entry in entries):
        raise CFS2LineageError(f"source ledger lacks mapping entries: {path}")
    return entries


def _validate_source_entry(entry: dict[str, t.Any], *, seed: int) -> None:
    source_id = expected_source_id(seed)
    expected = {
        "job_id": source_id, "model": "nextlat", "seed": seed,
        "phase": "base", "step": PARENT_STEPS, "updates": PARENT_STEPS,
    }
    if any(entry.get(key) != value for key, value in expected.items()):
        raise CFS2LineageError(f"{source_id} is not the exact 20,000-update NextLat parent")
    if entry.get("status") not in {"TRAINED", "DONE"}:
        raise CFS2LineageError(f"{source_id} is not terminal")
    raw_path, digest = entry.get("final_checkpoint"), entry.get("final_checkpoint_sha256")
    if not isinstance(raw_path, str) or not isinstance(digest, str) or len(digest) != 64:
        raise CFS2LineageError(f"{source_id} lacks a checkpoint path/SHA")
    checkpoint = pathlib.Path(raw_path).resolve()
    if not checkpoint.is_file() or checkpoint.is_symlink() or sha256_file(checkpoint) != digest:
        raise CFS2LineageError(f"{source_id} checkpoint is missing or stale")


def _immutable_json(path: pathlib.Path, value: t.Mapping[str, t.Any]) -> None:
    payload = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise CFS2LineageError(f"refusing to overwrite different frozen artifact: {path}")
        return
    descriptor, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = pathlib.Path(raw_tmp)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_lineage(
    ledgers: t.Sequence[os.PathLike[str] | str],
    *,
    output_dir: os.PathLike[str] | str,
) -> tuple[pathlib.Path, pathlib.Path]:
    source_paths = [pathlib.Path(path).resolve() for path in ledgers]
    if not source_paths or len(set(source_paths)) != len(source_paths):
        raise CFS2LineageError("provide one or more unique source ledgers")
    sources = [{"path": str(path), "sha256": sha256_file(path)} for path in source_paths]
    candidates: dict[str, list[tuple[dict[str, t.Any], pathlib.Path]]] = {}
    for path in source_paths:
        latest: dict[str, dict[str, t.Any]] = {}
        for entry in _read_ledger(path):
            job_id = entry.get("job_id")
            if isinstance(job_id, str):
                latest[job_id] = entry
        for source_id, entry in latest.items():
            candidates.setdefault(source_id, []).append((entry, path))

    selected: list[tuple[int, dict[str, t.Any], pathlib.Path]] = []
    for seed in ALL_PARENT_SEEDS:
        source_id = expected_source_id(seed)
        rows = candidates.get(source_id, [])
        if len(rows) != 1:
            raise CFS2LineageError(f"expected exactly one latest source for {source_id}; found {len(rows)}")
        entry, ledger_path = rows[0]
        _validate_source_entry(entry, seed=seed)
        selected.append((seed, entry, ledger_path))

    output = pathlib.Path(output_dir).resolve()
    parent_ledger = output / "cfs2_parent_ledger.json"
    _immutable_json(parent_ledger, {
        "schema": PARENT_LEDGER_SCHEMA, "status": "FROZEN",
        "entries": [entry for _, entry, _ in selected],
    })
    parent_rows = []
    source_by_path = {row["path"]: row for row in sources}
    for seed, entry, ledger_path in selected:
        parent_rows.append({
            "seed": seed,
            "canonical_parent_id": parent_id_for_seed(seed),
            "source_parent_id": expected_source_id(seed),
            "source_ledger": source_by_path[str(ledger_path)],
            "source_ledger_entry_sha256": canonical_json_sha256(entry),
            "parent_checkpoint": {
                "path": str(pathlib.Path(entry["final_checkpoint"]).resolve()),
                "sha256": entry["final_checkpoint_sha256"],
                "training_steps": PARENT_STEPS,
            },
        })
    receipt = output / "cfs2_parent_lineage_receipt.json"
    _immutable_json(receipt, {
        "schema": CFS2_PARENT_LINEAGE_SCHEMA_V2,
        "status": "FROZEN",
        "source_ledgers": sources,
        "parent_ledger": {"path": str(parent_ledger), "sha256": sha256_file(parent_ledger)},
        "parents": parent_rows,
    })
    # Exercise the production loader, including the checkpoint payload checks, before returning.
    load_parents(parent_ledger, lineage_receipt=receipt)
    return parent_ledger, receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", action="append", required=True, help="source run ledger; repeatable")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    ledger, receipt = materialize_lineage(args.ledger, output_dir=args.output_dir)
    print(json.dumps({
        "status": "FROZEN", "parents": len(ALL_PARENT_SEEDS),
        "parent_ledger": str(ledger), "lineage_receipt": str(receipt),
        "lineage_receipt_sha256": sha256_file(receipt),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
