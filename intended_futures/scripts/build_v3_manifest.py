#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


LANGUAGE_RE = re.compile(r"\(:language\s+(.+?)\)", re.IGNORECASE)


def _language(path: Path) -> str:
    match = LANGUAGE_RE.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"missing language field: {path}")
    return " ".join(match.group(1).split())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--libero-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
    from intended_futures.config import load_config

    config = load_config(args.config)
    suite = config["benchmark"]["suite"]
    bddl_root = args.libero_root / "libero" / "bddl_files"
    bddl_dir = bddl_root / suite
    init_dir = args.libero_root / "libero" / "init_files" / suite
    bddl_files = sorted(bddl_dir.glob("*.bddl"))
    prompts = config["benchmark"]["prompts"]
    for prompt_id, prompt in prompts.items():
        if "source_file" in prompt:
            source = bddl_root / str(prompt["source_suite"]) / str(prompt["source_file"])
        else:
            source = bddl_files[int(prompt["source_task_index"])]
        if _language(source) != prompt["text"]:
            raise RuntimeError(f"official prompt changed for {prompt_id}: {source}")

    rows = []
    for scene in config["benchmark"]["scenes"]:
        task_index = int(scene["task_index"])
        bddl = bddl_files[task_index]
        init_file = init_dir / f"{bddl.stem}.pruned_init"
        if not init_file.exists():
            raise FileNotFoundError(init_file)
        prompt_a = prompts[scene["contrast"][0]]
        prompt_b = prompts[scene["contrast"][1]]
        for state_index in config["sampling"]["initial_state_indices"]:
            rows.append(
                {
                    "stimulus_id": f"{scene['scene_id']}-state-{state_index:02d}",
                    "scene_id": scene["scene_id"],
                    "task_index": task_index,
                    "task_file": f"{suite}/{bddl.name}",
                    "bddl_sha256": _sha256(bddl),
                    "init_sha256": _sha256(init_file),
                    "initial_state_index": state_index,
                    "noise_seed": int(config["sampling"]["noise_seed_base"] + len(rows)),
                    "task_a": {
                        "prompt_id": scene["contrast"][0],
                        "prompt": prompt_a["text"],
                        "intended_subject": prompt_a["subject"],
                    },
                    "task_b": {
                        "prompt_id": scene["contrast"][1],
                        "prompt": prompt_b["text"],
                        "intended_subject": prompt_b["subject"],
                    },
                }
            )
    if len(rows) != config["sampling"]["expected_matched_pairs"]:
        raise RuntimeError("manifest row count violates v3 configuration")
    payload = {
        "study": config["study"],
        "suite": suite,
        "design": config["benchmark"]["design"],
        "rows": rows,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "manifest_sha256": payload["manifest_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
