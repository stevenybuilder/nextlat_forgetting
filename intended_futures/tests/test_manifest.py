import json
from pathlib import Path

from intended_futures.config import load_config
from intended_futures.manifest import build_matched_manifest


ROOT = Path(__file__).parents[1]


def _write_task(root: Path, suite: str, index: int, prompt: str, init: bytes) -> None:
    name = f"{index + 1:02d}-task-{index}"
    bddl = root / "libero" / "bddl_files" / suite / f"{name}.bddl"
    states = root / "libero" / "init_files" / suite / f"{name}.pruned_init"
    bddl.parent.mkdir(parents=True, exist_ok=True)
    states.parent.mkdir(parents=True, exist_ok=True)
    bddl.write_text(f"(define (problem x) (:language {prompt}) (:goal (And)))\n", encoding="utf-8")
    states.write_bytes(init)


def test_frozen_config_is_self_consistent():
    config = load_config(ROOT / "config" / "pilot.json")
    assert config["sampling"]["expected_matched_pairs"] == 50
    assert config["runtime_contract"]["num_gpus"] == 1


def test_manifest_uses_only_exact_initial_state_matches(tmp_path):
    config = json.loads((ROOT / "config" / "pilot.json").read_text(encoding="utf-8"))
    suite = config["benchmark"]["suite"]
    for pair_index, pair in enumerate(config["benchmark"]["matched_pairs"]):
        shared = f"pair-{pair_index}".encode()
        _write_task(tmp_path, suite, pair["task_indices"][0], f"prompt a {pair_index}", shared)
        _write_task(tmp_path, suite, pair["task_indices"][1], f"prompt b {pair_index}", shared)
    # The production task indices are sparse, so materialize unmatched filler tasks.
    existing = set(sum((pair["task_indices"] for pair in config["benchmark"]["matched_pairs"]), []))
    for task_index in range(15):
        if task_index not in existing:
            _write_task(tmp_path, suite, task_index, f"unmatched {task_index}", f"unique-{task_index}".encode())

    manifest = build_matched_manifest(config, tmp_path)
    assert len(manifest["rows"]) == 50
    assert len({row["stimulus_id"] for row in manifest["rows"]}) == 50
    assert all(row["task_a"]["init_sha256"] == row["task_b"]["init_sha256"] for row in manifest["rows"])
    assert all(row["task_a"]["prompt"] != row["task_b"]["prompt"] for row in manifest["rows"])
