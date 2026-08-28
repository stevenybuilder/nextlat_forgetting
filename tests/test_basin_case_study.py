"""Scientific-contract tests for the retrospective basin case study."""

from __future__ import annotations

import importlib.util
import csv
import json
import math
from pathlib import Path
from xml.etree import ElementTree

import pytest


PROJECT = Path(__file__).resolve().parents[1]
EVALUATOR = PROJECT / "scripts" / "case_study" / "evaluate_basin_checkpoint.py"
FREEZE = PROJECT / "manifests" / "case_study" / "basin" / "artifacts.json"
REPORT_BUILDER = PROJECT / "scripts" / "case_study" / "build_basin_report.py"
RELEASE = PROJECT / "results" / "studies" / "basin_case_study" / "retrospective-2026-08-27"


def load_module():
    spec = importlib.util.spec_from_file_location("basin_evaluator_under_test", EVALUATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = load_module()


def load_report_module():
    spec = importlib.util.spec_from_file_location("basin_report_under_test", REPORT_BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


R = load_report_module()


def test_freeze_is_retrospective_and_forbids_population_inference_and_training() -> None:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert freeze["status"] == "FROZEN_BEFORE_NEW_TRAJECTORY_EVALUATION"
    assert freeze["design"]["retrospective_outcome_selected"] is True
    assert freeze["design"]["outcomes_known_before_freeze"] is True
    assert freeze["design"]["population_inference_authorized"] is False
    assert freeze["design"]["formal_null_hypothesis_test"] is None
    assert freeze["runtime_controls"]["new_training_authorized"] is False
    assert freeze["runtime_controls"]["gpu_count"] == 1
    assert freeze["runtime_controls"]["full_host_gpu_fraction"] == 1.0
    assert freeze["runtime_controls"]["spend_stop_usd"] == 5.0


def test_freeze_includes_every_checkpoint_once_and_excludes_recovery_files() -> None:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    runs = {run["seed"]: run for run in freeze["runs"]}
    assert set(runs) == {1234, 1235}
    assert [item["step"] for item in runs[1234]["checkpoints"]] == [
        1000, 3000, 5000, 7000, 9000, 12000, 15000, 17000, 19000, 20000,
    ]
    assert [item["step"] for item in runs[1235]["checkpoints"]] == [
        2000, 4000, 6000, 8000, 10000, 12000, 15000, 17000, 19000, 20000,
    ]
    checkpoints = [item for run in runs.values() for item in run["checkpoints"]]
    assert len({item["sha256"] for item in checkpoints}) == 20
    assert all("recovery" not in item["filename"] for item in checkpoints)
    assert all(len(item["sha256"]) == 64 and item["size"] == 263100035 for item in checkpoints)


def test_margin_summary_uses_population_moments() -> None:
    values = [-2.0, 0.0, 3.0, 5.0]
    summary = B.summarize_margins(
        count=len(values), total=sum(values), total_sq=sum(value * value for value in values),
        minimum=min(values), maximum=max(values),
    )
    assert summary["mean"] == 1.5
    assert summary["population_std"] == pytest.approx(math.sqrt(7.25))
    assert summary["min"] == -2.0
    assert summary["max"] == 5.0


def test_progress_refuses_provenance_changes(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json.progress.json"
    identity = {"job_id": "nextlat-s1234-base", "checkpoint_sha256": "a" * 64}
    fresh = B.load_progress(path, identity, total=20_000)
    saved = dict(
        fresh,
        next_index=2,
        exact_correct=1,
        token_correct=[2, 1, 1, 1, 1],
        branch_correct=1,
        branch_margin_count=2,
        branch_margin_sum=0.5,
        branch_margin_sum_sq=1.25,
        branch_margin_min=-0.5,
        branch_margin_max=1.0,
    )
    path.write_text(json.dumps(saved), encoding="utf-8")
    assert B.load_progress(path, identity, total=20_000) == saved
    with pytest.raises(B.BasinEvaluationError, match="provenance mismatch"):
        B.load_progress(path, dict(identity, checkpoint_sha256="b" * 64), total=20_000)


def test_branch_metric_is_bound_to_first_nontrivial_path_decision() -> None:
    assert B.BRANCH_PATH_POSITION == 2
    assert B.BRANCH_HIDDEN_INDEX == 63
    source = EVALUATOR.read_text(encoding="utf-8")
    assert "torch.use_deterministic_algorithms(True)" in source
    assert "targets[:, :1]" in source
    assert "gold = targets[:, 1]" in source
    assert "torch.argmax" in source


def test_stager_never_persists_private_bucket_name() -> None:
    source = (PROJECT / "scripts/case_study/stage_basin_artifacts.py").read_text(encoding="utf-8")
    assert '"bucket_name_recorded": False' in source
    assert '"bucket_name": bucket_name' not in source


def test_runner_uses_one_device_and_refuses_nonidentical_repeat() -> None:
    source = (PROJECT / "scripts/case_study/run_basin_evaluations.py").read_text(encoding="utf-8")
    assert source.count('"--devices", "1"') == 2
    assert '"--precision", "bf16-mixed"' in source
    assert "reference.read_bytes() == repeat.read_bytes()" in source
    assert "clean repeated evaluation changed scientific JSON" in source


def test_tracked_release_reports_the_two_run_claim_boundary() -> None:
    summary = json.loads((RELEASE / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema"] == "nextlat_forgetting/basin_case_study_summary/1"
    assert summary["design"]["retrospective_outcome_selected"] is True
    assert summary["design"]["run_count"] == 2
    assert summary["design"]["population_inference_authorized"] is False
    assert summary["design"]["heldout_examples_are_not_independent_training_runs"] is True
    assert summary["deterministic_repeat"]["status"] == "PASS"
    assert summary["results"]["seed_1234"]["interval_notation"] == "(1000, 3000]"
    assert summary["results"]["seed_1234"]["step_3000_exact_path_accuracy"] == 0.99015
    assert summary["results"]["seed_1235"]["step_20000_exact_path_accuracy"] == 0.18255
    assert summary["results"]["seed_1235"][
        "step_20000_first_decision_gold_margin_mean"
    ] == pytest.approx(-4.231727400970459)
    discrepancy = summary["historical_evaluator_comparison"]
    assert discrepancy["historical_correct"] == 3663
    assert discrepancy["new_correct"] == 3651
    assert discrepancy["correct_count_difference_historical_minus_new"] == 12
    assert discrepancy["absolute_accuracy_difference"] == pytest.approx(0.0006)
    assert summary["cost"]["new_training_usd"] == 0.0
    assert summary["cost"]["checkpoint_evaluation_session_usd_approx"] == 0.382


def test_tracked_release_has_twenty_rows_and_valid_svg() -> None:
    with (RELEASE / "checkpoint_summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 20
    assert [int(row["step"]) for row in rows if int(row["seed"]) == 1234] == [
        1000, 3000, 5000, 7000, 9000, 12000, 15000, 17000, 19000, 20000,
    ]
    assert [int(row["step"]) for row in rows if int(row["seed"]) == 1235] == [
        2000, 4000, 6000, 8000, 10000, 12000, 15000, 17000, 19000, 20000,
    ]
    for name in ["exact_path_accuracy.svg", "first_decision_margin.svg", "loss_trajectory.svg"]:
        root = ElementTree.parse(RELEASE / "figures" / name).getroot()
        assert root.tag.endswith("svg")
        assert root.find("{http://www.w3.org/2000/svg}title") is not None
        assert root.find("{http://www.w3.org/2000/svg}desc") is not None


def test_report_outputs_do_not_publish_private_runtime_paths() -> None:
    forbidden = ["/workspace/", ".agent_state/", "private_gcs", "object_name", "bucket_name"]
    for path in RELEASE.rglob("*"):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        assert not any(value in text for value in forbidden), path


def test_report_builder_is_byte_deterministic_with_local_bundle(tmp_path: Path) -> None:
    input_root = PROJECT / ".agent_state" / "basin_case_study" / "extracted"
    if not input_root.exists():
        pytest.skip("local ignored checkpoint-evaluation bundle is not present")
    R.build(input_root, PROJECT, tmp_path)
    expected = sorted(path.relative_to(RELEASE) for path in RELEASE.rglob("*") if path.is_file())
    observed = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*") if path.is_file())
    assert observed == expected
    for relative in expected:
        assert (tmp_path / relative).read_bytes() == (RELEASE / relative).read_bytes()
