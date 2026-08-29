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


def test_v3_same_scene_design_is_balanced():
    config = load_config(ROOT / "config" / "pilot_v3.json")
    scenes = config["benchmark"]["scenes"]
    prompts = config["benchmark"]["prompts"]
    assert len(scenes) == 12
    assert config["sampling"]["expected_matched_pairs"] == 120
    side_a = [prompts[scene["contrast"][0]]["subject"] for scene in scenes]
    side_b = [prompts[scene["contrast"][1]]["subject"] for scene in scenes]
    for subject in set(side_a + side_b):
        assert side_a.count(subject) == side_b.count(subject) == 4


def test_v4_replacement_scene_preserves_balance() -> None:
    config = load_config(ROOT / "config" / "pilot_v4.json")
    scenes = config["benchmark"]["scenes"]
    prompts = config["benchmark"]["prompts"]
    assert {scene["task_index"] for scene in scenes} == {0, 1, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14}
    side_a = [prompts[scene["contrast"][0]]["subject"] for scene in scenes]
    side_b = [prompts[scene["contrast"][1]]["subject"] for scene in scenes]
    assert all(side_a.count(subject) == side_b.count(subject) == 4 for subject in set(side_a + side_b))


def test_tc1_frozen_protocol_uses_four_disjoint_untouched_state_splits() -> None:
    protocol = json.loads(
        (ROOT / "config" / "target_control_tc1.json").read_text(
            encoding="utf-8"
        )
    )
    population = protocol["population"]
    split_fields = [
        "observer_fit_state_indices",
        "observer_validation_state_indices",
        "causal_test_state_indices",
        "reserve_state_indices",
    ]
    split_sets = [set(population[field]) for field in split_fields]
    assert all(len(indices) == 10 for indices in split_sets)
    assert set.union(*split_sets) == set(range(10, 50))
    assert all(
        left.isdisjoint(right)
        for index, left in enumerate(split_sets)
        for right in split_sets[index + 1 :]
    )
    assert set(population["previously_observed_state_indices"]).isdisjoint(
        set.union(*split_sets)
    )
    assert population["expected_manifest_rows"] == 12 * 40
    assert population["minimum_pair_distance_meters"] == 0.05
    assert protocol["status"] == "frozen"
    assert protocol["site"] == {
        "site_id": "paligemma_l13",
        "pathway": "paligemma",
        "layer": 13,
        "indexing": "zero_based",
        "token_pooling": "none",
        "selection_rule": "pre-outcome Action Atlas goal-classification maximum, then retained because full replay redirected target behavior in M0",
    }


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
