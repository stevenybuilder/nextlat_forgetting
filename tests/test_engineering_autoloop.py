"""Focused tests for the engineering-only, anti-p-hacking iteration boundary."""

from __future__ import annotations

import json
import pathlib
import sys

import pytest

from engineering_autoloop import (
    CHECKS,
    LoopError,
    abort_candidate,
    begin_candidate,
    evaluate_candidate,
    validate_check_registry,
)


PASS = {"pass": (sys.executable, "-c", "print('engineering ok')")}
FAIL = {"fail": (sys.executable, "-c", "raise SystemExit(7)")}


def project(tmp_path: pathlib.Path) -> pathlib.Path:
    """Minimal tree; missing frozen files are themselves part of the snapshot."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "worker.py").write_text("VALUE = 1\n")
    (tmp_path / "PROGRAM.md").write_text("frozen\n")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "model.yaml").write_text("seed: 1234\n")
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "bank.json").write_text("{}\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_gate.py").write_text("def test_gate(): assert True\n")
    return tmp_path


def paths(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    return root / ".git" / "active.json", root / "results" / "engineering_loop.jsonl"


def begin(root: pathlib.Path, *, registry=PASS, now=100.0, checks=("pass",)):
    active, log = paths(root)
    return begin_candidate(
        root,
        candidate_id="candidate-1",
        candidate_paths=["scripts/worker.py"],
        time_budget_seconds=30,
        max_checks=1,
        checks=list(checks),
        active_path=active,
        log_path=log,
        now=now,
        registry=registry,
    )


def test_one_changed_candidate_passes_without_reading_scientific_results(tmp_path) -> None:
    root = project(tmp_path)
    active, log = paths(root)
    begin(root)
    (root / "scripts" / "worker.py").write_text("VALUE = 2\n")

    times = iter([101.0, 101.1, 101.2])
    record = evaluate_candidate(active, now=lambda: next(times))

    assert record["accepted"] is True
    assert record["candidate_before_sha256"] != record["candidate_after_sha256"]
    assert record["frozen_before_sha256"] == record["frozen_after_sha256"]
    assert record["worktree_action"].startswith("none")
    events = [json.loads(line)["event"] for line in log.read_text().splitlines()]
    assert events == ["BEGIN", "EVALUATE"]
    assert not active.exists()


def test_scientific_surface_change_blocks_checks_and_acceptance(tmp_path) -> None:
    root = project(tmp_path)
    active, _ = paths(root)
    begin(root)
    (root / "scripts" / "worker.py").write_text("VALUE = 2\n")
    (root / "configs" / "model.yaml").write_text("seed: 9999\n")

    record = evaluate_candidate(active, now=lambda: 101.0)

    assert record["accepted"] is False
    assert record["checks"] == []
    assert "scientific frozen surface changed" in record["reasons"]


def test_candidate_cannot_own_tests_configs_or_outcome_files(tmp_path) -> None:
    root = project(tmp_path)
    active, log = paths(root)
    for candidate in ("tests/test_gate.py", "configs/model.yaml", "results/live_numbers.json"):
        with pytest.raises(LoopError):
            begin_candidate(
                root, candidate_id="bad", candidate_paths=[candidate],
                time_budget_seconds=30, max_checks=1, checks=["pass"],
                active_path=active, log_path=log, now=100, registry=PASS,
            )


def test_candidate_path_cannot_escape_the_repository(tmp_path) -> None:
    root = project(tmp_path)
    active, log = paths(root)
    with pytest.raises(LoopError, match="scoped repository path"):
        begin_candidate(
            root, candidate_id="escape", candidate_paths=["../outside.py"],
            time_budget_seconds=30, max_checks=1, checks=["pass"],
            active_path=active, log_path=log, now=100, registry=PASS,
        )


def test_only_one_candidate_can_be_active(tmp_path) -> None:
    root = project(tmp_path)
    active, log = paths(root)
    begin(root)
    with pytest.raises(LoopError, match="already active"):
        begin_candidate(
            root, candidate_id="candidate-2", candidate_paths=["scripts/other.py"],
            time_budget_seconds=30, max_checks=1, checks=["pass"],
            active_path=active, log_path=log, now=101, registry=PASS,
        )


def test_fixed_budget_expires_without_running_checks(tmp_path) -> None:
    root = project(tmp_path)
    active, _ = paths(root)
    begin(root, now=100)
    (root / "scripts" / "worker.py").write_text("VALUE = 2\n")

    record = evaluate_candidate(active, now=lambda: 131.0)

    assert record["accepted"] is False
    assert record["checks"] == []
    assert "fixed time budget exceeded" in record["reasons"]


def test_failed_check_is_audited_and_never_reverts_candidate(tmp_path) -> None:
    root = project(tmp_path)
    active, _ = paths(root)
    begin(root, registry=FAIL, checks=("fail",))
    candidate = root / "scripts" / "worker.py"
    candidate.write_text("VALUE = 7\n")
    times = iter([101.0, 101.1, 101.2])

    record = evaluate_candidate(active, now=lambda: next(times))

    assert record["accepted"] is False
    assert record["checks"][0]["returncode"] == 7
    assert candidate.read_text() == "VALUE = 7\n"
    assert record["worktree_action"].startswith("none")


def test_abort_is_append_only_and_leaves_diff_in_place(tmp_path) -> None:
    root = project(tmp_path)
    active, log = paths(root)
    begin(root)
    candidate = root / "scripts" / "worker.py"
    candidate.write_text("VALUE = 3\n")
    before = log.read_bytes()

    record = abort_candidate(active, "smaller candidate preferred", now=102)

    assert record["event"] == "ABORT"
    assert log.read_bytes().startswith(before)
    assert candidate.read_text() == "VALUE = 3\n"
    assert not active.exists()


def test_registry_rejects_outcome_reads_and_destructive_cleanup() -> None:
    validate_check_registry(CHECKS)
    with pytest.raises(LoopError, match="prohibited outcomes"):
        validate_check_registry({"bad": ("python", "read_psi.py")})
    with pytest.raises(LoopError, match="destructive"):
        validate_check_registry({"bad": ("git", "reset", "--hard")})


def test_budget_and_check_set_are_fixed_at_begin(tmp_path) -> None:
    root = project(tmp_path)
    active, log = paths(root)
    with pytest.raises(LoopError, match="exceeds fixed"):
        begin_candidate(
            root, candidate_id="too-many", candidate_paths=["scripts/worker.py"],
            time_budget_seconds=30, max_checks=1, checks=["a", "b"],
            active_path=active, log_path=log, now=100,
            registry={"a": PASS["pass"], "b": PASS["pass"]},
        )
