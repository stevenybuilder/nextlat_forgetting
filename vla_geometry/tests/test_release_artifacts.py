import json
from pathlib import Path

from vla_geometry.grid import build_cells, get_factor_order, load_config
from vla_geometry.io import sha256_file, sha256_source_tree
from vla_geometry.seeds import resolve_all_seed_maps


ROOT = Path(__file__).parents[1]
RESULTS = ROOT / "results"


def _load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def test_task16_release_artifacts_match_frozen_code_config_and_seed_population():
    config_path = ROOT / "config" / "memory_pilot.json"
    config = load_config(config_path)
    cells = build_cells(config["factors"], get_factor_order(config))
    provenance = _load("task16_pilot_provenance.json")
    manifest = _load("task16_seed_manifest.json")
    adapter = _load("task16_adapter_validation.json")
    analysis = _load("task16_pilot_analysis.json")

    assert provenance["config_sha256"] == sha256_file(config_path)
    assert provenance["source_tree_sha256"] == sha256_source_tree(ROOT)
    assert manifest == resolve_all_seed_maps(config, cells)
    assert adapter["passed"] is True
    assert adapter["rewards_or_outcomes_observed"] is False
    assert adapter["config_sha256"] == provenance["config_sha256"]
    assert adapter["source_tree_sha256"] == provenance["source_tree_sha256"]
    assert adapter["resets_checked"] == 64
    assert sum(row["trials"] for row in analysis["cells"]) == 640
    assert sum(row["successes"] for row in analysis["cells"]) == 315
    assert analysis["primary"]["advance_to_causal_stage"] is False


def test_task16_release_summaries_are_complete_and_failure_free():
    expected = {"smoke": 4, "representation": 256, "behavior": 640}
    for mode, planned in expected.items():
        summary = _load(f"task16_{mode}_summary.json")
        assert summary["mode"] == mode
        assert summary["planned"] == planned
        assert summary["completed_or_preexisting"] == planned
        assert summary["failed"] == 0
