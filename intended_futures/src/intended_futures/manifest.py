from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any


_LANGUAGE_RE = re.compile(r"\(:language\s+(.+?)\)", re.IGNORECASE)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _language(path: Path) -> str:
    match = _LANGUAGE_RE.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"could not parse :language from {path}")
    return " ".join(match.group(1).split())


def discover_identical_state_pairs(libero_root: str | Path, suite: str) -> list[dict[str, Any]]:
    root = Path(libero_root)
    bddl_dir = root / "libero" / "bddl_files" / suite
    init_dir = root / "libero" / "init_files" / suite
    if not bddl_dir.is_dir() or not init_dir.is_dir():
        raise FileNotFoundError(f"LIBERO-CF suite not found below {root}")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task_index, bddl_path in enumerate(sorted(bddl_dir.glob("*.bddl"))):
        init_path = init_dir / f"{bddl_path.stem}.pruned_init"
        if not init_path.exists():
            raise FileNotFoundError(f"missing official initial states: {init_path}")
        init_sha256 = sha256_file(init_path)
        groups[init_sha256].append(
            {
                "task_index": task_index,
                "task_file": f"{suite}/{bddl_path.name}",
                "task_stem": bddl_path.stem,
                "prompt": _language(bddl_path),
                "bddl_sha256": sha256_file(bddl_path),
                "init_sha256": init_sha256,
            }
        )

    result = []
    for init_sha256, tasks in sorted(groups.items()):
        prompts = {task["prompt"] for task in tasks}
        if len(tasks) == 2 and len(prompts) == 2:
            result.append({"init_sha256": init_sha256, "tasks": sorted(tasks, key=lambda row: row["task_index"])})
    return sorted(result, key=lambda pair: pair["tasks"][0]["task_index"])


def build_matched_manifest(config: dict[str, Any], libero_root: str | Path) -> dict[str, Any]:
    suite = config["benchmark"]["suite"]
    discovered = discover_identical_state_pairs(libero_root, suite)
    frozen_pairs = config["benchmark"]["matched_pairs"]
    if len(discovered) != len(frozen_pairs):
        raise ValueError(f"discovered {len(discovered)} exact pairs, expected {len(frozen_pairs)}")

    rows: list[dict[str, Any]] = []
    for frozen, actual in zip(frozen_pairs, discovered):
        actual_indices = [task["task_index"] for task in actual["tasks"]]
        if actual_indices != frozen["task_indices"]:
            raise ValueError(
                f"pair {frozen['pair_id']} task indices changed: {actual_indices} != {frozen['task_indices']}"
            )
        for side, (task, subject) in enumerate(
            zip(actual["tasks"], frozen["intended_subjects"])
        ):
            task["side"] = "a" if side == 0 else "b"
            task["intended_subject"] = subject
            task["intended_receptacle"] = frozen["intended_receptacle"]
        for state_index in config["sampling"]["initial_state_indices"]:
            rows.append(
                {
                    "stimulus_id": f"{frozen['pair_id']}-state-{state_index:02d}",
                    "pair_id": frozen["pair_id"],
                    "initial_state_index": state_index,
                    "init_sha256": actual["init_sha256"],
                    "task_a": actual["tasks"][0],
                    "task_b": actual["tasks"][1],
                    "noise_seed": int(config["sampling"]["noise_seed_base"] + len(rows)),
                }
            )

    if len(rows) != config["sampling"]["expected_matched_pairs"]:
        raise AssertionError("manifest row count violates frozen design")
    payload = {
        "study": config["study"],
        "suite": suite,
        "pair_selection": "byte_identical_init_file_distinct_language_v1",
        "rows": rows,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
