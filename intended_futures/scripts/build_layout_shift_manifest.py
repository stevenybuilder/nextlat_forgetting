#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_digest(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--libero-plus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from intended_futures.libero_plus import (
        parse_language,
        parse_layout_name,
        prompt_source_for_support,
        task_geometry,
    )

    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = args.libero_plus_root
    expected_commit = config["upstream"]["libero_plus_commit"]
    actual_commit = (
        __import__("subprocess")
        .check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True)
        .strip()
    )
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"LIBERO-Plus commit {actual_commit} differs from {expected_commit}"
        )

    suite = config["population"]["suite"]
    bddl_dir = root / "libero" / "libero" / "bddl_files" / suite
    init_dir = (
        root
        / "libero"
        / "libero"
        / "init_files"
        / "libero_newobj"
        / suite
    )
    classification_path = (
        root / "libero" / "libero" / "benchmark" / "task_classification.json"
    )
    classification = json.loads(classification_path.read_text(encoding="utf-8"))
    excluded = set(config["population"]["excluded_families"])
    split_by_sample = {
        int(sample): split
        for split, samples in config["population"]["split_samples"].items()
        for sample in samples
    }
    rows: list[dict[str, Any]] = []
    for entry in classification[suite]:
        if entry.get("category") != "Objects Layout":
            continue
        name = str(entry["name"])
        try:
            family, level, sample = parse_layout_name(name)
        except ValueError:
            continue
        if family in excluded or sample not in split_by_sample:
            continue
        bddl = bddl_dir / f"{name}.bddl"
        init_file = init_dir / f"{name}.pruned_init"
        base_bddl = bddl_dir / f"{family}.bddl"
        for required in (bddl, init_file, base_bddl):
            if not required.is_file():
                raise FileNotFoundError(required)
        geometry = task_geometry(bddl)
        support_a = geometry["supports"]["akita_black_bowl_1"]
        support_b = geometry["supports"]["akita_black_bowl_2"]
        prompt_a_source = prompt_source_for_support(support_a)
        prompt_b_source = prompt_source_for_support(support_b)
        if prompt_a_source != family:
            raise RuntimeError(
                f"base prompt/support mismatch for {name}: {prompt_a_source}"
            )
        prompt_sources = []
        for prompt_source in (prompt_a_source, prompt_b_source):
            source_path = bddl_dir / f"{prompt_source}.bddl"
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            prompt_sources.append(
                {
                    "id": prompt_source,
                    "text": parse_language(source_path.read_text(encoding="utf-8")),
                    "source_file": f"{suite}/{source_path.name}",
                    "source_sha256": _sha256(source_path),
                }
            )
        target_a = geometry["subject_xy"]["akita_black_bowl_1"]
        target_b = geometry["subject_xy"]["akita_black_bowl_2"]
        family_id = family.removeprefix("pick_up_the_black_bowl_").removesuffix(
            "_and_place_it_on_the_plate"
        )
        rows.append(
            {
                "stimulus_id": f"{family_id}-l{level}-s{sample}",
                "family_id": family_id,
                "source_task_id": int(entry["id"]),
                "source_task_name": name,
                "difficulty_level": int(entry["difficulty_level"]),
                "level": level,
                "sample": sample,
                "split": split_by_sample[sample],
                "task_file": f"{suite}/{bddl.name}",
                "bddl_sha256": _sha256(bddl),
                "init_file": f"libero_newobj/{suite}/{init_file.name}",
                "init_sha256": _sha256(init_file),
                "noise_seed": int(config["population"]["noise_seed_base"])
                + len(rows),
                "task_a": {
                    "prompt_id": prompt_sources[0]["id"],
                    "prompt": prompt_sources[0]["text"],
                    "prompt_source_file": prompt_sources[0]["source_file"],
                    "prompt_source_sha256": prompt_sources[0]["source_sha256"],
                    "intended_subject": "akita_black_bowl_1",
                    "initial_support": support_a,
                },
                "task_b": {
                    "prompt_id": prompt_sources[1]["id"],
                    "prompt": prompt_sources[1]["text"],
                    "prompt_source_file": prompt_sources[1]["source_file"],
                    "prompt_source_sha256": prompt_sources[1]["source_sha256"],
                    "intended_subject": "akita_black_bowl_2",
                    "initial_support": support_b,
                },
                "bddl_target_xy_a": target_a.tolist(),
                "bddl_target_xy_b": target_b.tolist(),
                "bddl_target_difference_xy": (target_a - target_b).tolist(),
            }
        )
    rows.sort(key=lambda row: (row["family_id"], row["level"], row["sample"]))
    for index, row in enumerate(rows):
        row["noise_seed"] = int(config["population"]["noise_seed_base"]) + index
    counts = {
        split: sum(row["split"] == split for row in rows)
        for split in config["population"]["split_samples"]
    }
    expected_counts = config["population"]["expected_split_units"]
    if counts != expected_counts:
        raise RuntimeError(f"manifest split counts {counts} differ from {expected_counts}")
    payload: dict[str, Any] = {
        "study": config["study"],
        "suite": suite,
        "design": "libero_plus_visible_target_displacement_matched_prompts_v1",
        "config_sha256": _sha256(args.config),
        "libero_plus_commit": actual_commit,
        "classification_sha256": _sha256(classification_path),
        "rows": rows,
    }
    payload["manifest_sha256"] = _canonical_digest(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "rows": len(rows),
                "split_counts": counts,
                "families": len({row["family_id"] for row in rows}),
                "manifest_sha256": payload["manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
