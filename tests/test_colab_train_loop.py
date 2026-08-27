"""Focused regressions for the dual-role Colab sweep driver."""

from __future__ import annotations

import importlib.util
import inspect
import io
import json
import os
import shutil
import stat
import sys
import tarfile
import threading
from pathlib import Path

import pytest


PROJECT = Path(__file__).resolve().parents[1]
DRIVER_PATH = PROJECT / "scripts" / "colab_train_loop.py"


def _load_driver():
    spec = importlib.util.spec_from_file_location("colab_train_loop_under_test", DRIVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_verified_checkpoint(path: Path, driver, *, step: int) -> None:
    path.write_bytes((json.dumps({"training_steps": step, "model": {"weight": [1.0]}})
                      + "\n").encode())
    metadata = {
        "schema": 1,
        "path": str(path.resolve()),
        "sha256": driver.sha256_file(path),
        "size_bytes": path.stat().st_size,
        "training_steps": step,
        "rng_state": True,
    }
    path.with_name(path.name + ".meta.json").write_text(json.dumps(metadata))


def _runtime_spec(driver, base: dict, *, session_id: str | None = None) -> dict:
    source_sha = "a" * 64
    input_sha = "b" * 64
    document = {
        **base,
        "source_sha256": source_sha,
        "source_object": f"{driver.PREFIX}/source/project-{source_sha}.tar.gz",
        "input_bundle_sha256": input_sha,
        "input_bundle_prefix": f"{driver.PREFIX}/input_bundles/{input_sha}",
    }
    if session_id is not None:
        document["session_id"] = session_id
    return document


def test_confirmatory_job_spec_guard_accepts_only_frozen_scopes() -> None:
    driver = _load_driver()

    assert driver.validate_confirmatory_job_spec({
        "gpu": "a100", "run_matrix_args": ["--phase", "base"],
    })["phase"] == "base"
    assert driver.validate_confirmatory_job_spec({
        "runner": "lurestar", "gpu": "a100",
        "run_matrix_args": ["--phase", "base", "--only", "bst-s1238-base"],
    })["only"] == ("bst-s1238-base",)
    for phase in ("train", "evaluate"):
        assert driver.validate_confirmatory_job_spec({
            "runner": "hmm", "runner_phase": phase, "family": True, "gpu": "a100",
        }) == {"runner": "hmm", "phase": phase, "family": True}


@pytest.mark.parametrize("spec,message", [
    ({"gpu": "a100", "run_matrix_args": ["--phase", "adapt"]}, "excludes H3"),
    ({"gpu": "a100", "run_matrix_args": ["--phase", "all"]}, "base-only"),
    ({"gpu": "a100", "run_matrix_args": ["--phase", "base", "--models", "gpt"]},
     "permits only"),
    ({"gpu": "a100", "run_matrix_args": ["--phase", "base", "--only",
                                             "gpt-s1234-adapt-near"]}, "excludes H3"),
    ({"gpu": "t4", "run_matrix_args": ["--phase", "base"]}, "exactly a100"),
    ({"gpu": "a100", "max_attempts": 21,
      "run_matrix_args": ["--phase", "base"]}, "from 1 through"),
    ({"gpu": "a100", "hard_stop_balance_cu": 1,
      "run_matrix_args": ["--phase", "base"]}, "may not be below"),
    ({"gpu": "a100", "run_matrix_args": ["--phase", "base"], "extra": True},
     "unknown fields"),
    ({"runner": "hmm", "runner_phase": "train", "family": False, "gpu": "a100"},
     "complete frozen family"),
    ({"runner": "hmm", "runner_phase": "train", "family": True, "gpu": "a100",
      "models": ["gpt"]}, "unknown fields"),
    ({"runner": "hmm", "runner_phase": "train", "family": True, "gpu": "a100",
      "only": ["gpt-seed1234-hmm-persistent_moderate"]}, "unknown fields"),
    ({"runner": "hmm", "runner_phase": "train", "family": True, "gpu": "a100",
      "regimes": ["persistent_moderate"]}, "unknown fields"),
    ({"runner": "hmm", "runner_phase": "profile", "family": True, "gpu": "a100"},
     "exactly train or evaluate"),
])
def test_confirmatory_job_spec_guard_rejects_scope_expansion(spec, message) -> None:
    driver = _load_driver()
    with pytest.raises(SystemExit, match=message):
        driver.validate_confirmatory_job_spec(spec)


def test_runtime_job_spec_guard_binds_generated_overlay_and_session() -> None:
    driver = _load_driver()
    base = {"runner": "hmm", "runner_phase": "train", "family": True, "gpu": "a100"}
    runtime = _runtime_spec(driver, base, session_id="gpu-a100-test")

    assert driver.validate_confirmatory_job_spec(
        runtime, runtime_overlay=True, require_session=True)["family"] is True
    runtime["source_object"] = f"{driver.PREFIX}/source/project-{'c' * 64}.tar.gz"
    with pytest.raises(SystemExit, match="source object/hash mismatch"):
        driver.validate_confirmatory_job_spec(
            runtime, runtime_overlay=True, require_session=True)


def test_predecessor_source_is_strict_distinct_and_job_spec_bound(tmp_path) -> None:
    driver = _load_driver()
    predecessor = "c" * 64
    base = {
        "runner": "hmm", "runner_phase": "train", "family": True, "gpu": "a100",
        "predecessor_source_sha256": predecessor,
        "recovery_job_ids": list(driver.D41_RECOVERY_JOB_IDS),
        "recovery_receipt_sha256": "e" * 64,
    }
    runtime = _runtime_spec(driver, base, session_id="gpu-a100-test")
    runtime["recovery_receipt_object"] = (
        f"{driver.PREFIX}/recovery_receipts/{'e' * 64}.json")
    runtime["recovery_receipt_generation"] = "123"

    assert driver.validate_confirmatory_job_spec(
        runtime, runtime_overlay=True, require_session=True)["phase"] == "train"
    for invalid in ("C" * 64, "c" * 63, "not-a-hash"):
        with pytest.raises(SystemExit, match="predecessor source hash"):
            driver.validate_confirmatory_job_spec({**base, "predecessor_source_sha256": invalid})
    with pytest.raises(SystemExit, match="must differ"):
        driver.validate_confirmatory_job_spec(
            {**runtime, "predecessor_source_sha256": runtime["source_sha256"]},
            runtime_overlay=True, require_session=True)

    assert driver.canonical_json_sha256(base) != driver.canonical_json_sha256({
        **base, "predecessor_source_sha256": "d" * 64})


def _d43_spec(driver, *, receipt_sha="f" * 64) -> dict:
    return {
        "runner": "hmm", "runner_phase": "train", "family": True, "gpu": "a100",
        "predecessor_source_sha256": "c" * 64,
        "recovery_job_ids": list(driver.D41_RECOVERY_JOB_IDS),
        "recovery_receipt_sha256": "e" * 64,
        "continuation_gate": driver.D43_CONTINUATION_GATE,
        "continuation_gate_schema": driver.D43_CONTINUATION_SCHEMA,
        "continuation_receipt_sha256": receipt_sha,
    }


def test_d43_job_spec_requires_explicit_exact_gate_schema_receipt_and_d41_recovery() -> None:
    driver = _load_driver()
    valid = _d43_spec(driver)
    assert driver.validate_confirmatory_job_spec(valid)["phase"] == "train"
    for mutation in (
        {"continuation_gate": "d41"},
        {"continuation_gate_schema": "nextlat_forgetting/d43_measurement_amendment/0"},
        {"continuation_receipt_sha256": "F" * 64},
        {"predecessor_source_sha256": None, "recovery_job_ids": None,
         "recovery_receipt_sha256": None},
    ):
        with pytest.raises(SystemExit, match="D43 continuation"):
            driver.validate_confirmatory_job_spec({**valid, **mutation})
    # A receipt hash alone can never silently select D43.
    with pytest.raises(SystemExit, match="exact gate/schema/receipt"):
        driver.validate_confirmatory_job_spec({
            **valid, "continuation_gate": None, "continuation_gate_schema": None})


def _write_fake_d43_bundle(root: Path, driver, *, source_sha="a" * 64,
                           state_mutation=None, receipt_mutation=None):
    state = root / ".agent_state"
    state.mkdir(parents=True, exist_ok=True)
    pending = [
        f"{model}-seed{seed}-hmm-{regime}"
        for regime in ("persistent_moderate", "fast_mixing_moderate",
                       "persistent_high_aliasing")
        for model in ("gpt", "nextlat") for seed in range(1234, 1239)
        if f"{model}-seed{seed}-hmm-{regime}" not in driver.D41_RECOVERY_JOB_IDS
    ]
    continuation = {
        "training_started": True,
        "completed_job_ids": list(driver.D41_RECOVERY_JOB_IDS),
        "pending_job_ids": pending, "evaluated_job_ids": [],
        "scientific_evaluations_started": False,
        "scientific_metrics_inspected": False, "evaluator_invocations": 0,
    }
    if state_mutation:
        continuation.update(state_mutation)
    continuation_path = state / "d43-continuation-state.json"
    continuation_path.write_text(json.dumps(continuation) + "\n")
    jobs = [{
        "job_id": job_id,
        "created_under_predecessor_source_sha256": "c" * 64,
        "consumed_read_only_by_successor_source_sha256": source_sha,
    } for job_id in driver.D41_RECOVERY_JOB_IDS]
    receipt = {
        "schema": driver.D43_CONTINUATION_SCHEMA, "status": "PASS",
        "authorization": "MEASUREMENT_AMENDMENT_GO",
        "confirmatory_lifecycle": {
            "training_started": True, "completed_hmm_training_cells": 10,
            "total_hmm_training_cells": 30,
            "scientific_evaluations_started": False,
            "scientific_evaluations_inspected": False,
        },
        "archives": {"d43_measurement_successor": {"sha256": source_sha}},
        "exact_ten_checkpoint_lineage": {
            "predecessor_to_successor_provenance": jobs},
        "outcome_blind_atomic_continuation": {"atomic_continuation_state": {
            "path": ".agent_state/d43-continuation-state.json"}},
    }
    if receipt_mutation:
        receipt_mutation(receipt)
    receipt_path = state / "d43-measurement-amendment-receipt.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n")
    gate_script = root / "scripts/d43_measurement_amendment_gate.py"
    gate_script.parent.mkdir(parents=True, exist_ok=True)
    gate_script.write_text(
        "import json, pathlib\n"
        "ORIGINAL_PREDECESSOR_SHA256 = '" + "c" * 64 + "'\n"
        "EXACT_TEN_JOB_IDS = " + repr(tuple(driver.D41_RECOVERY_JOB_IDS)) + "\n"
        "ALL_HMM_JOB_IDS = " + repr(tuple([*driver.D41_RECOVERY_JOB_IDS, *pending])) + "\n"
        "def validate_receipt(root, *archives):\n"
        "    return json.loads((pathlib.Path(root)/'.agent_state/d43-measurement-amendment-receipt.json').read_text())\n"
    )
    return receipt_path, receipt, continuation


@pytest.mark.parametrize("state_mutation,receipt_mutation,message", [
    ({"completed_job_ids": [*list(_load_driver().D41_RECOVERY_JOB_IDS), "extra-job"]}, None,
     "lifecycle/partition/provenance"),
    ({"evaluated_job_ids": ["gpt-seed1234-hmm-persistent_moderate"]}, None,
     "lifecycle/partition/provenance"),
    ({"evaluator_invocations": 1}, None, "lifecycle/partition/provenance"),
    (None, lambda receipt: receipt["archives"]["d43_measurement_successor"].update(
        {"sha256": "b" * 64}), "lifecycle/partition/provenance"),
    (None, lambda receipt: receipt["confirmatory_lifecycle"].update(
        {"training_started": False}), "lifecycle/partition/provenance"),
])
def test_d43_launch_bundle_recomputes_and_refuses_nonatomic_or_wrong_source(
        tmp_path, state_mutation, receipt_mutation, message) -> None:
    driver = _load_driver()
    receipt_path, _receipt, _state = _write_fake_d43_bundle(
        tmp_path, driver, state_mutation=state_mutation,
        receipt_mutation=receipt_mutation)
    spec = _d43_spec(driver, receipt_sha=driver.sha256_file(receipt_path))
    with pytest.raises(SystemExit, match=message):
        driver.validate_d43_continuation_bundle(tmp_path, spec, "a" * 64)


def test_d43_launch_bundle_binds_exact_receipt_and_dual_source_lineage(tmp_path) -> None:
    driver = _load_driver()
    receipt_path, _receipt, _state = _write_fake_d43_bundle(tmp_path, driver)
    spec = _d43_spec(driver, receipt_sha=driver.sha256_file(receipt_path))
    binding = driver.validate_d43_continuation_bundle(tmp_path, spec, "a" * 64)
    assert binding["receipt_sha256"] == spec["continuation_receipt_sha256"]
    assert binding["predecessor_source_sha256"] == "c" * 64
    assert binding["source_sha256"] == "a" * 64
    assert binding["completed_job_ids"] == list(driver.D41_RECOVERY_JOB_IDS)
    receipt_path.write_text("{}\n")
    with pytest.raises(SystemExit, match="missing or hash-stale"):
        driver.validate_d43_continuation_bundle(tmp_path, spec, "a" * 64)


class _StopAfter:
    def __init__(self, calls):
        self.calls = calls
        self.count = 0

    def wait(self, _interval):
        self.count += 1
        return self.count > self.calls


def test_sync_failure_circuit_breaker_self_heals_and_resets_counter(tmp_path) -> None:
    driver = _load_driver()

    class Durability:
        outcomes = iter((RuntimeError("one"), RuntimeError("two"), None,
                         RuntimeError("one-again"), RuntimeError("two-again"), None))

        def sync_once(self, *_args, **_kwargs):
            outcome = next(self.outcomes)
            if outcome:
                raise outcome
            return {}

    abort = threading.Event()
    result = driver.durable_sync_loop(
        _StopAfter(6), abort, Durability(), "ledger.json", "hmm_run_ledger.json",
        tmp_path / "diagnostic.json", source_sha256="a" * 64, interval=0)
    assert result is None
    assert not abort.is_set()
    assert not (tmp_path / "diagnostic.json").exists()


def test_three_consecutive_sync_failures_abort_and_atomically_retain_diagnostic(tmp_path) -> None:
    driver = _load_driver()

    class Durability:
        calls = 0

        def sync_once(self, *_args, **_kwargs):
            self.calls += 1
            raise RuntimeError("auth unavailable")

    durability = Durability()
    abort = threading.Event()
    diagnostic = tmp_path / "diagnostic.json"
    result = driver.durable_sync_loop(
        _StopAfter(10), abort, durability, "ledger.json", "hmm_run_ledger.json",
        diagnostic, source_sha256="a" * 64,
        predecessor_source_sha256="c" * 64, interval=0)
    assert abort.is_set()
    assert durability.calls == driver.MAX_CONSECUTIVE_SYNC_FAILURES
    assert result["training_complete"] is False
    assert json.loads(diagnostic.read_text()) == result
    assert not diagnostic.with_name(diagnostic.name + ".partial").exists()


def test_interruptible_command_stops_on_sync_abort_before_more_paid_work() -> None:
    driver = _load_driver()
    abort = threading.Event()
    abort.set()
    rc, output = driver.sh(
        f"{sys.executable} -c 'import time; time.sleep(30)'",
        check=False, quiet=True, abort_event=abort)
    assert rc != 0
    assert "TRAINING_COMPLETE=True" not in output


def test_sh_uses_bash_for_process_substitution(capsys) -> None:
    driver = _load_driver()

    rc, output = driver.sh("cat <(printf 'verified\\n')", quiet=True)

    assert rc == 0
    assert output == "verified"
    assert "verified" in capsys.readouterr().out


def test_sh_keeps_200_human_lines_and_can_capture_structured_output_unbounded() -> None:
    driver = _load_driver()

    _, bounded = driver.sh("seq 1 250", quiet=True, silent=True)
    _, complete = driver.sh("seq 1 250", quiet=True, silent=True, max_lines=None)

    assert len(bounded.splitlines()) == 200
    assert bounded.splitlines()[0] == "51"
    assert len(complete.splitlines()) == 250
    assert complete.splitlines()[0] == "1"


def test_ledger_progress_uses_latest_append_only_status_per_job() -> None:
    driver = _load_driver()
    document = {
        "schema": 1,
        "entries": [
            {"seq": 0, "job_id": "gpt-1234", "status": "DONE"},
            {"seq": 1, "job_id": "nextlat-1234", "status": "RUNNING"},
            {"seq": 2, "job_id": "gpt-1234", "status": "STALE"},
            {"seq": 3, "job_id": "nextlat-1234", "status": "DONE"},
            {"seq": 4, "reason": "global record without a job"},
        ],
    }

    assert driver.ledger_progress(document) == (1, 2)
    assert driver.ledger_progress({"schema": 1, "entries": []}) == (0, 0)
    assert driver.ledger_progress({
        "schema": 1,
        "entries": [{"job_id": "trained", "status": "TRAINED"}],
    }) == (1, 1)


def test_driver_terminates_training_loop_without_claiming_evaluation_done() -> None:
    driver = _load_driver()
    source = inspect.getsource(driver)

    assert 'marker.get("training_complete") is True' in source
    assert '"TRAINING_COMPLETE=True" in out' not in source
    assert "TRAINING MATRIX COMPLETE" in source
    assert "SWEEP COMPLETE" not in source
    assert "ALL_DONE=True" not in source


def test_status_pair_requires_agreement_before_lifecycle_action(monkeypatch) -> None:
    driver = _load_driver()
    replies = iter([
        (0, '{"status":"no_runtime"}'),
        (0, '{"status":"connected","session":"gpu-a100-test"}'),
    ])
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return next(replies)

    monkeypatch.setattr(driver, "sh", run)

    first, second = driver.colab_status_pair(delay_seconds=0, sleeper=lambda _: None)

    assert driver.agreed_runtime_state(first, second) == "uncertain"
    assert driver.agreed_runtime_state({"status": "no_runtime"},
                                       {"status": "no_runtime"}) == "gone"
    assert driver.agreed_runtime_state({"status": "connected"},
                                       {"status": "connected"}) == "active"
    assert all(kwargs["max_lines"] is None for _, kwargs in calls)


def test_quota_pair_requires_two_full_agreeing_reads(monkeypatch) -> None:
    driver = _load_driver()
    replies = iter([
        (0, '{"paid_balance":1781.0,"active_runtimes":0,"burn_rate_hourly":0}'),
        (0, '{"paid_balance":1781.0,"active_runtimes":0,"burn_rate_hourly":0}'),
    ])
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return next(replies)

    monkeypatch.setattr(driver, "sh", run)
    first, second = driver.colab_quota_pair(delay_seconds=0, sleeper=lambda _: None)

    assert driver.agreed_paid_balance(first, second) == 1781.0
    assert len(calls) == 2
    assert all(call[0][0] == "colab quota --json" for call in calls)
    assert all(kwargs["max_lines"] is None for _, kwargs in calls)


@pytest.mark.parametrize(
    "quota_pair,status_pair,message",
    [
        (({"paid_balance": 1500.0}, {"paid_balance": 1499.0}),
         ({"status": "no_runtime"}, {"status": "no_runtime"}), "quota reads disagree"),
        (({"paid_balance": 1188.61}, {"paid_balance": 1188.61}),
         ({"status": "no_runtime"}, {"status": "no_runtime"}), "hard stop reached"),
        (({"paid_balance": 1500.0}, {"paid_balance": 1500.0}),
         ({"status": "no_runtime"}, {"status": "connected"}), "do not agree"),
    ],
)
def test_every_provision_authorization_fails_closed_on_quota_status_or_floor(
        quota_pair, status_pair, message) -> None:
    driver = _load_driver()

    with pytest.raises(SystemExit, match=message):
        driver.authorize_provisioning(
            1188.61, quota_reader=lambda: quota_pair, status_reader=lambda: status_pair)


def test_provision_authorization_uses_paired_quota_then_fresh_paired_status() -> None:
    driver = _load_driver()
    calls = []

    def quotas():
        calls.append("quota-pair")
        return ({"paid_balance": 1500.0, "active_runtimes": 0},
                {"paid_balance": 1500.0, "active_runtimes": 0})

    def statuses():
        calls.append("status-pair")
        return ({"status": "no_runtime"}, {"status": "no_runtime"})

    assert driver.authorize_provisioning(
        1188.61, quota_reader=quotas, status_reader=statuses) == 1500.0
    assert calls == ["quota-pair", "status-pair"]
    loop_source = inspect.getsource(driver._owned_loop)
    authorize = loop_source.index("authorize_provisioning(hard_floor)")
    start = loop_source.index('colab start --gpu', authorize)
    assert authorize < start


def test_active_runtime_identity_must_match_owned_session() -> None:
    driver = _load_driver()

    driver.require_owned_session(
        {"status": "connected", "session": "gpu-owned"},
        {"status": "connected", "session": "gpu-owned"}, "gpu-owned")
    with pytest.raises(driver.OwnershipUncertain, match="ownership is uncertain"):
        driver.require_owned_session(
            {"status": "connected", "session": "gpu-someone-else"},
            {"status": "connected", "session": "gpu-someone-else"}, "gpu-owned")
    for first, second in (
        ({"status": "connected"}, {"status": "connected"}),
        ({"status": "connected", "session": "gpu-owned"}, {"status": "connected"}),
        ({"status": "connected", "session": "gpu-owned"},
         {"status": "connected", "session": "gpu-changed"}),
    ):
        with pytest.raises(driver.OwnershipUncertain):
            driver.require_owned_session(first, second, "gpu-owned")


def test_ownership_uncertain_handoff_is_atomic_and_strictly_read_only(tmp_path) -> None:
    driver = _load_driver()
    path = driver.write_ownership_uncertain_diagnostic(
        tmp_path, expected_session_id="gpu-owned",
        status_first={"status": "connected", "session": "gpu-owned"},
        status_second={"status": "connected", "session": "gpu-changed"},
        stage="active_runtime_monitor")
    document = json.loads(path.read_text())
    assert document["status"] == "OWNERSHIP_UNCERTAIN_READ_ONLY"
    assert document["forbidden_actions"] == ["stop", "start", "exec", "upload"]
    assert not path.with_name(path.name + ".partial").exists()


def test_owned_loop_checks_sid_before_upload_and_uncertainty_before_mutation() -> None:
    driver = _load_driver()
    source = inspect.getsource(driver._owned_loop)
    preflight = source.index("preflight_first, preflight_second = colab_status_pair()")
    publish = source.index("publish_immutable_host_file")
    assert preflight < publish
    monitor_reason = source.index('if reason == "ownership_uncertain"')
    stop = source.index('if reason in {"terminal", "stalled"}')
    start = source.index('colab start --gpu')
    assert monitor_reason < stop
    assert preflight < start


def test_driver_sync_abort_forces_nonzero_and_can_never_claim_completion() -> None:
    driver = _load_driver()
    source = inspect.getsource(driver.driver)
    assert "if sync_abort.is_set():\n            rc = 74" in source
    assert "not sync_abort.is_set() and rc == 0" in source
    assert "return int(rc)" in source


def test_final_sync_does_not_require_state_for_untouched_pending_jobs() -> None:
    driver = _load_driver()
    document = {
        "schema": 1,
        "entries": [
            {"job_id": "untouched", "status": "PENDING"},
            {"job_id": "started", "status": "RUNNING"},
            {"job_id": "finished", "status": "DONE"},
        ],
    }

    assert driver.state_required_jobs(document) == {"started", "finished"}


def test_base_terminal_requires_done_not_merely_trained() -> None:
    driver = _load_driver()
    document = {
        "entries": [
            {"job_id": "a", "status": "TRAINED"},
            {"job_id": "b", "status": "DONE"},
        ]
    }
    assert driver.ledger_progress(document) == (2, 2)
    assert driver.ledger_progress(document, terminal_statuses={"DONE"}) == (1, 2)
    assert driver.requested_phase(["--phase", "base"]) == "base"
    assert driver.requested_phase(["--models", "gpt"]) is None


def test_package_includes_project_runtime_code_and_excludes_local_bulk(tmp_path) -> None:
    driver = _load_driver()
    for directory in ("src", "scripts", "configs", "manifests", "docs"):
        path = tmp_path / directory
        path.mkdir()
        (path / "kept.txt").write_text(directory)
    for directory in ("data", "results", "upstream", ".secrets"):
        path = tmp_path / directory
        path.mkdir()
        (path / "excluded.txt").write_text(directory)
    (tmp_path / "HANDOFF.md").write_text("mutable operations log\n")

    archive = driver.package(str(tmp_path))

    with tarfile.open(archive) as tf:
        names = set(tf.getnames())
    for directory in ("src", "scripts", "configs", "manifests", "docs"):
        assert f"{directory}/kept.txt" in names
    assert not any(name.startswith(("data/", "results/", "upstream/", ".secrets/"))
                   for name in names)
    assert "HANDOFF.md" not in names


def test_package_excludes_nested_credentials_and_links(tmp_path) -> None:
    driver = _load_driver()
    source = tmp_path / "src"
    source.mkdir()
    (source / "kept.py").write_text("VALUE = 1\n")
    (source / ".env").write_text("TOKEN=secret\n")
    (source / ".env.production").write_text("TOKEN=secret\n")
    (source / "adc.json").write_text('{"refresh_token":"secret"}\n')
    (source / "application_default_credentials.json").write_text("{}\n")
    (source / "link.py").symlink_to(source / "kept.py")

    with tarfile.open(driver.package(str(tmp_path))) as archive:
        names = set(archive.getnames())

    assert "src/kept.py" in names
    for excluded in ("src/.env", "src/.env.production", "src/adc.json",
                     "src/application_default_credentials.json", "src/link.py"):
        assert excluded not in names


def test_safe_extract_rejects_traversal_and_links(tmp_path) -> None:
    driver = _load_driver()
    traversal = tmp_path / "traversal.tar"
    with tarfile.open(traversal, "w") as archive:
        info = tarfile.TarInfo("../escaped.txt")
        payload = b"nope"
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    with pytest.raises(RuntimeError, match="path traversal"):
        driver.safe_extract_tar(traversal, tmp_path / "out-traversal")
    assert not (tmp_path / "escaped.txt").exists()

    linked = tmp_path / "linked.tar"
    with tarfile.open(linked, "w") as archive:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/tmp/outside"
        archive.addfile(info)
    with pytest.raises(RuntimeError, match="link or special file"):
        driver.safe_extract_tar(linked, tmp_path / "out-link")


def test_source_snapshot_installs_fresh_and_refuses_overlay(tmp_path) -> None:
    driver = _load_driver()
    source = tmp_path / "source"
    source.mkdir()
    (source / "run.py").write_text("VALUE = 1\n")
    archive = tmp_path / "source.tar"
    with tarfile.open(archive, "w") as tf:
        tf.add(source / "run.py", arcname="run.py")
    destination = tmp_path / "project"

    driver.install_source_snapshot(archive, destination)
    assert (destination / "run.py").read_text() == "VALUE = 1\n"

    with pytest.raises(RuntimeError, match="nonempty source destination"):
        driver.install_source_snapshot(archive, destination)


def test_package_is_reproducible_across_source_and_wall_clock_metadata(tmp_path) -> None:
    driver = _load_driver()
    source = tmp_path / "src"
    source.mkdir()
    payload = source / "kept.py"
    payload.write_text("VALUE = 1\n")

    first = Path(driver.package(str(tmp_path))).read_bytes()
    os.utime(source, (2_000_000_000, 2_000_000_000))
    os.utime(payload, (2_000_000_000, 2_000_000_000))
    second = Path(driver.package(str(tmp_path))).read_bytes()

    assert first == second


def test_runtime_spec_binds_a_content_addressed_source_snapshot(tmp_path) -> None:
    driver = _load_driver()
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "kept.py").write_text("VALUE = 1\n")
    (tmp_path / "manifests").mkdir()
    inventory = tmp_path / "manifests" / "manifest_inventory.sha256"
    inventory.write_text("a" * 64 + "  manifests/example.json\n")

    archive = driver.package(str(tmp_path))
    spec_path, spec = driver.prepare_runtime_spec(
        str(tmp_path), {"gpu": "a100", "run_matrix_args": ["--phase", "base"]}, archive)

    digest = driver.sha256_file(archive)
    assert spec["source_sha256"] == digest
    assert spec["source_object"] == f"lurestar/source/project-{digest}.tar.gz"
    input_digest = driver.sha256_file(inventory)
    assert spec["input_bundle_sha256"] == input_digest
    assert spec["input_bundle_prefix"] == f"lurestar/input_bundles/{input_digest}"
    assert json.loads(spec_path.read_text()) == spec


def _write_confirmatory_receipts(root: Path, driver, spec: dict) -> str:
    for relative in driver.CONFIRMATORY_PROTOCOL_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen protocol: {relative}\n")
    validator = root / "scripts" / "validate_preregistration.py"
    validator.parent.mkdir(parents=True, exist_ok=True)
    validator.write_text(
        "import json, pathlib\n"
        "def validate(evidence, *, amendment, spec):\n"
        "    path = pathlib.Path(evidence).parent / 'synthetic-recomputed-preregistration.json'\n"
        "    return json.loads(path.read_text())\n"
    )
    fixture_manifest = root / "manifests" / "synthetic.json"
    fixture_manifest.parent.mkdir(parents=True, exist_ok=True)
    fixture_manifest.write_text('{"fixture":true}\n')
    inventory = fixture_manifest.parent / "manifest_inventory.sha256"
    inventory.write_text(
        f"{driver.sha256_file(fixture_manifest)}  manifests/synthetic.json\n"
    )
    state = root / ".agent_state"
    state.mkdir(exist_ok=True)
    test_path = state / "confirmatory-test-receipt.json"
    review_path = state / "confirmatory-review-receipt.json"
    report = root / "docs" / "INDEPENDENT_CONFIRMATORY_REVIEW.md"
    report.write_text("VERDICT: PASS\n")
    archive = Path(driver.package(str(root)))
    source_sha = driver.sha256_file(archive)
    input_sha = driver.sha256_file(inventory)
    input_prefix = f"lurestar/input_bundles/{input_sha}"
    (state / "input-bundle-upload.json").write_text(json.dumps({
        "schema": driver.INPUT_BUNDLE_UPLOAD_SCHEMA,
        "status": "COMPLETE",
        "bucket": driver.BUCKET,
        "bundle_prefix": input_prefix,
        "input_bundle_sha256": input_sha,
        "object_count": 1,
        "objects": [{
            "local_path": "manifests/synthetic.json",
            "name": f"{input_prefix}/manifests/synthetic.json",
            "generation": "101",
            "size_bytes": fixture_manifest.stat().st_size,
            "sha256": driver.sha256_file(fixture_manifest),
        }],
        "commit": {
            "local_path": "manifests/manifest_inventory.sha256",
            "name": f"{input_prefix}/manifests/manifest_inventory.sha256",
            "generation": "102",
            "size_bytes": inventory.stat().st_size,
            "sha256": input_sha,
        },
    }, sort_keys=True) + "\n")
    test_path.write_text(json.dumps({
        "schema": driver.CONFIRMATORY_TEST_SCHEMA,
        "outcome": "PASS",
        "exit_code": 0,
        "tests_passed": 1,
        "source_sha256": source_sha,
    }, sort_keys=True) + "\n")
    review_path.write_text(json.dumps({
        "schema": driver.CONFIRMATORY_REVIEW_SCHEMA,
        "verdict": "PASS",
        "reviewer": "independent-test-reviewer",
        "report_path": "docs/INDEPENDENT_CONFIRMATORY_REVIEW.md",
        "report_sha256": driver.sha256_file(report),
        "source_sha256": source_sha,
    }, sort_keys=True) + "\n")
    evidence_path = state / "preregistration-evidence.json"
    evidence = {
        "schema": driver.PREREGISTRATION_EVIDENCE_SCHEMA,
        "gates": {str(gate): {} for gate in range(1, 12)},
    }
    evidence["gates"]["1"] = {"artifacts": [{
        "role": "source_snapshot",
        "path": ".agent_state/project.tar.gz",
        "sha256": source_sha,
        "schema": "binary/source-snapshot",
    }]}
    evidence["gates"]["11"] = {"artifacts": [
        {
            "role": "full_suite_receipt",
            "path": ".agent_state/confirmatory-test-receipt.json",
            "sha256": driver.sha256_file(test_path),
            "schema": driver.FULL_TEST_SUITE_SCHEMA,
        },
        {
            "role": "independent_review_receipt",
            "path": ".agent_state/confirmatory-review-receipt.json",
            "sha256": driver.sha256_file(review_path),
            "schema": driver.INDEPENDENT_SCIENTIFIC_REVIEW_SCHEMA,
        },
    ]}
    evidence_path.write_text(json.dumps(evidence, sort_keys=True) + "\n")
    amendment = root / "docs" / "PREREGISTRATION_AMENDMENT_2026-08-24.md"
    specification = root / "nextlat_v4_predictive_geometry_spec.md"
    preregistration_path = state / "preregistration-freeze-receipt.json"
    preregistration_document = {
        "schema": driver.PREREGISTRATION_FREEZE_SCHEMA,
        "status": "PASS",
        "all_eleven_gates_pass": True,
        "authority": {
            "amendment": {"path": str(amendment), "sha256": driver.sha256_file(amendment)},
            "spec": {"path": str(specification), "sha256": driver.sha256_file(specification)},
            "evidence": {"path": str(evidence_path), "sha256": driver.sha256_file(evidence_path)},
            "validator": {"path": str(validator), "sha256": driver.sha256_file(validator)},
        },
        "missing_gate_blocks": [],
        "extra_gate_blocks": [],
        "global_issues": [],
        "gates": [{"gate": gate, "status": "PASS", "issues": []}
                  for gate in range(1, 12)],
        "meaning": "pre-compute design frozen; no scientific outcome evaluated",
    }
    preregistration_path.write_text(json.dumps(preregistration_document, sort_keys=True) + "\n")
    (state / "synthetic-recomputed-preregistration.json").write_text(
        json.dumps(preregistration_document, sort_keys=True) + "\n")
    clearance = {
        "schema": driver.CONFIRMATORY_CLEARANCE_SCHEMA,
        "authorization": "GO",
        "source_sha256": source_sha,
        "job_spec_sha256": driver.canonical_json_sha256(spec),
        "input_bundle": driver.validate_input_bundle_receipt(root),
        "protocol_bindings": {
            relative: driver.sha256_file(root / relative)
            for relative in driver.CONFIRMATORY_PROTOCOL_PATHS
        },
        "test_receipt_sha256": driver.sha256_file(test_path),
        "review_receipt_sha256": driver.sha256_file(review_path),
        "preregistration": driver.validate_preregistration_pass_receipt(
            root, source_sha, preregistration_path),
    }
    (state / "confirmatory-clearance.json").write_text(
        json.dumps(clearance, sort_keys=True) + "\n")
    return source_sha


def test_confirmatory_clearance_binds_source_spec_protocol_tests_and_review(tmp_path) -> None:
    driver = _load_driver()
    spec = {"gpu": "a100", "run_matrix_args": ["--phase", "base"]}
    source_sha = _write_confirmatory_receipts(tmp_path, driver, spec)

    clearance = driver.validate_confirmatory_clearance(tmp_path, spec, source_sha)

    assert clearance["authorization"] == "GO"
    (tmp_path / "docs" / "FOUNDATIONS.md").write_text("changed after review\n")
    with pytest.raises(SystemExit, match="protocol binding mismatch"):
        driver.validate_confirmatory_clearance(tmp_path, spec, source_sha)


@pytest.mark.parametrize("verdict", ["BLOCK", "FAIL"])
def test_nonpassing_independent_review_explicitly_refuses_launch(
        tmp_path, verdict) -> None:
    driver = _load_driver()
    spec = {"gpu": "a100", "run_matrix_args": ["--phase", "base"]}
    source_sha = _write_confirmatory_receipts(tmp_path, driver, spec)
    review_path = tmp_path / ".agent_state" / "confirmatory-review-receipt.json"
    review = json.loads(review_path.read_text())
    review["verdict"] = verdict
    review_path.write_text(json.dumps(review, sort_keys=True) + "\n")
    # Model an attacker updating only the outer binding. The semantic review verdict
    # remains independently re-read and must still block the paid launch.
    clearance_path = tmp_path / ".agent_state" / "confirmatory-clearance.json"
    clearance = json.loads(clearance_path.read_text())
    clearance["review_receipt_sha256"] = driver.sha256_file(review_path)
    clearance_path.write_text(json.dumps(clearance, sort_keys=True) + "\n")

    with pytest.raises(
            SystemExit,
            match="confirmatory review receipt is not a passing source-bound receipt"):
        driver.validate_confirmatory_clearance(tmp_path, spec, source_sha)


@pytest.mark.parametrize("mutation, message", [
    ("source", "source binding mismatch"),
    ("spec", "job-spec binding mismatch"),
    ("inputs", "input-bundle upload receipt"),
    ("tests", "confirmatory test receipt binding mismatch"),
    ("review", "confirmatory review receipt binding mismatch"),
])
def test_confirmatory_clearance_fails_closed_on_stale_evidence(
        tmp_path, mutation, message) -> None:
    driver = _load_driver()
    spec = {"gpu": "a100", "run_matrix_args": ["--phase", "base"]}
    source_sha = _write_confirmatory_receipts(tmp_path, driver, spec)
    if mutation == "source":
        source_sha = "c" * 64
    elif mutation == "spec":
        spec = {**spec, "max_attempts": 19}
    elif mutation == "inputs":
        (tmp_path / ".agent_state" / "input-bundle-upload.json").write_text("{}\n")
    elif mutation == "tests":
        (tmp_path / ".agent_state" / "confirmatory-test-receipt.json").write_text("{}\n")
    else:
        (tmp_path / ".agent_state" / "confirmatory-review-receipt.json").write_text("{}\n")

    with pytest.raises(SystemExit, match=message):
        driver.validate_confirmatory_clearance(tmp_path, spec, source_sha)


def test_rehashed_malicious_clearance_still_refuses_adaptation_scope(tmp_path) -> None:
    driver = _load_driver()
    safe = {"gpu": "a100", "run_matrix_args": ["--phase", "base"]}
    source_sha = _write_confirmatory_receipts(tmp_path, driver, safe)
    malicious = {"gpu": "a100", "run_matrix_args": ["--phase", "adapt"]}
    path = tmp_path / ".agent_state" / "confirmatory-clearance.json"
    clearance = json.loads(path.read_text())
    clearance["job_spec_sha256"] = driver.canonical_json_sha256(malicious)
    path.write_text(json.dumps(clearance, sort_keys=True) + "\n")

    with pytest.raises(SystemExit, match="excludes H3 and adaptation"):
        driver.validate_confirmatory_clearance(tmp_path, malicious, source_sha)


def test_host_dispatch_refuses_malicious_spec_before_packaging(
        tmp_path, monkeypatch) -> None:
    driver = _load_driver()
    state = tmp_path / ".agent_state"
    state.mkdir()
    (state / "job_spec.json").write_text(json.dumps({
        "gpu": "a100", "run_matrix_args": ["--phase", "base"],
        "adaptation_manifest": "attacker-controlled.json",
    }))
    calls = []
    monkeypatch.setattr(driver, "package", lambda *_args: calls.append("package"))

    with pytest.raises(SystemExit, match="unknown fields"):
        driver._owned_loop(str(tmp_path))

    assert calls == []


def test_driver_dispatch_refuses_malicious_runtime_spec_before_gpu_or_credentials(
        tmp_path, monkeypatch) -> None:
    driver = _load_driver()
    malicious = _runtime_spec(
        driver,
        {"gpu": "a100", "run_matrix_args": ["--phase", "adapt"]},
        session_id="gpu-a100-owned",
    )
    sidecar = tmp_path / "job_spec.json"
    sidecar.write_text(json.dumps(malicious) + "\n")
    calls = []
    monkeypatch.setattr(driver, "SPEC_PATH", str(sidecar))
    monkeypatch.setattr(driver, "secure_adc", lambda *_args: calls.append("secure_adc"))
    monkeypatch.setattr(driver, "verify_runtime_gpu", lambda *_args, **_kwargs:
                        calls.append("verify_runtime_gpu"))

    with pytest.raises(SystemExit, match="excludes H3 and adaptation"):
        driver.driver()

    assert calls == []


def test_remote_input_preflight_checks_exact_receipt_generations_before_compute(
        tmp_path, monkeypatch) -> None:
    driver = _load_driver()
    spec = {"gpu": "a100", "run_matrix_args": ["--phase", "base"]}
    _write_confirmatory_receipts(tmp_path, driver, spec)
    binding = driver.validate_input_bundle_receipt(tmp_path)
    receipt = json.loads(
        (tmp_path / ".agent_state" / "input-bundle-upload.json").read_text())
    records = {record["name"]: record for record in [*receipt["objects"], receipt["commit"]]}

    class Remote:
        def __init__(self, record):
            self.generation = record["generation"]
            self.size_bytes = record["size_bytes"]
            self.custom_sha256 = record["sha256"]

    class Backend:
        def __init__(self, _bucket):
            pass

        def resolve(self, name):
            return Remote(records[name]) if name in records else None

        def download_exact(self, name, generation, destination):
            assert name == receipt["commit"]["name"]
            assert generation == receipt["commit"]["generation"]
            destination.write_bytes(
                (tmp_path / "manifests" / "manifest_inventory.sha256").read_bytes())

    fake = type("Uploader", (), {"GcloudBackend": Backend})
    monkeypatch.setattr(driver, "_load_input_uploader", lambda _root: fake)
    assert driver.verify_remote_input_bundle(tmp_path, binding) is True

    records[receipt["objects"][0]["name"]]["generation"] = "999"
    with pytest.raises(SystemExit, match="absent or changed"):
        driver.verify_remote_input_bundle(tmp_path, binding)


def test_controller_lock_is_create_only_and_archives_provably_dead_local_owner(
        tmp_path, monkeypatch) -> None:
    driver = _load_driver()
    monkeypatch.setattr(driver.socket, "gethostname", lambda: "same-host")
    lock = tmp_path / ".agent_state" / "colab-controller.lock"

    with driver.controller_lock(tmp_path) as owner:
        assert json.loads(lock.read_text())["controller_id"] == owner["controller_id"]
        with pytest.raises(SystemExit, match="another Colab controller"):
            with driver.controller_lock(tmp_path):
                pass
    assert not lock.exists()

    lock.write_text(json.dumps({
        "schema": "nextlat_forgetting/colab_controller_lock/1",
        "controller_id": "dead-owner",
        "hostname": "same-host",
        "pid": 99999999,
    }))
    monkeypatch.setattr(driver, "_local_pid_alive", lambda _pid: False)
    with driver.controller_lock(tmp_path):
        assert lock.exists()
    assert len(list(lock.parent.glob("colab-controller.lock.stale.*"))) == 1


def test_runtime_bootstrap_is_required_and_receives_both_roots(tmp_path, monkeypatch) -> None:
    driver = _load_driver()
    project = tmp_path / "project with spaces"
    upstream = tmp_path / "upstream with spaces"
    project.mkdir()
    upstream.mkdir()
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        receipt = (json.dumps({
            "schema": "nextlat_forgetting/runtime_patch/1",
            "patch_version": 5,
            "upstream_commit": driver.PINNED,
            "generated_at_unix": driver.time.time(),
        }, indent=2, sort_keys=True) + "\n").encode()
        (upstream / ".lurestar_runtime_patch_receipt.json").write_bytes(receipt)
        audit = project / "source_snapshot" / "runtime_patch"
        audit.mkdir(parents=True, exist_ok=True)
        (audit / "runtime_patch_receipt.json").write_bytes(receipt)

    monkeypatch.setattr(driver, "sh", run)

    with pytest.raises(SystemExit, match="runtime_bootstrap.py is absent"):
        driver.apply_runtime_bootstrap(str(project), str(upstream))
    assert calls == []

    scripts = project / "scripts"
    scripts.mkdir()
    (scripts / "runtime_bootstrap.py").write_text("# integration hook\n")
    result = driver.apply_runtime_bootstrap(str(project), str(upstream))
    assert result["applied_receipt_sha256"] == result["audit_receipt_sha256"]
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command.startswith(sys.executable)
    assert "runtime_bootstrap.py" in command
    assert "--project-root" in command and "--upstream" in command
    assert kwargs == {"cwd": str(project)}


def test_runtime_bootstrap_rejects_shipped_static_audit_in_place_of_emitted_receipt(
        tmp_path, monkeypatch) -> None:
    driver = _load_driver()
    project = tmp_path / "project"
    upstream = tmp_path / "upstream"
    (project / "scripts").mkdir(parents=True)
    upstream.mkdir()
    (project / "scripts" / "runtime_bootstrap.py").write_text("# integration hook\n")
    audit = project / "source_snapshot" / "runtime_patch"
    audit.mkdir(parents=True)
    (audit / "runtime_patch_receipt.json").write_text('{"static":true}\n')

    def run(_command, **_kwargs):
        (upstream / ".lurestar_runtime_patch_receipt.json").write_text(json.dumps({
            "schema": "nextlat_forgetting/runtime_patch/1",
            "patch_version": 5,
            "upstream_commit": driver.PINNED,
            "generated_at_unix": driver.time.time(),
        }) + "\n")

    monkeypatch.setattr(driver, "sh", run)
    with pytest.raises(SystemExit, match="differs from the receipt emitted"):
        driver.apply_runtime_bootstrap(project, upstream)


def test_d41_runtime_fingerprint_records_applied_and_audit_receipt_hashes() -> None:
    driver = _load_driver()
    receipt = json.loads(
        (PROJECT / ".agent_state" / "d41-exact-ten-recovery-receipt.json").read_text())
    audit = PROJECT / "source_snapshot" / "runtime_patch" / "runtime_patch_receipt.json"
    actual = dict(receipt["runtime_equivalence"]["expected_successor_contract"])

    fingerprint = driver.verify_d41_runtime_equivalence(
        receipt, actual, project_root=PROJECT, patch_receipt_path=audit,
        audit_patch_receipt_path=audit)

    digest = driver.sha256_file(audit)
    assert fingerprint["runtime_patch_applied_receipt_sha256"] == digest
    assert fingerprint["runtime_patch_audit_receipt_sha256"] == digest
    assert fingerprint["runtime_patch_receipt_projection_sha256"] == (
        receipt["runtime_equivalence"]["runtime_patch"]
        ["expected_receipt_projection_sha256"])


def test_session_spec_is_bound_before_runtime_inputs_are_uploaded(tmp_path, monkeypatch) -> None:
    driver = _load_driver()
    runtime_spec_file = tmp_path / "job_spec.runtime.json"
    runtime_spec = _runtime_spec(
        driver, {"gpu": "a100", "run_matrix_args": ["--phase", "base"]})
    runtime_spec_file.write_text(json.dumps(runtime_spec) + "\n")
    events = []
    real_prepare = driver.prepare_session_spec

    def prepare(path, spec, session_id):
        events.append(("prepare", session_id))
        return real_prepare(path, spec, session_id)

    def run(command, **kwargs):
        del kwargs
        events.append(("upload", command))
        return 0, ""

    monkeypatch.setattr(driver, "prepare_session_spec", prepare)
    monkeypatch.setattr(driver, "sh", run)

    driver.upload_session_inputs(
        "gpu-owned-1", "/tmp/adc.json", runtime_spec_file, runtime_spec)

    assert events[0] == ("prepare", "gpu-owned-1")
    assert [event[0] for event in events] == ["prepare", "upload", "upload"]
    assert json.loads(runtime_spec_file.read_text())["session_id"] == "gpu-owned-1"


def test_upload_retries_transient_failure_on_same_session() -> None:
    driver = _load_driver()
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return (1, "kernel write failed") if len(calls) == 1 else (0, "ok")

    attempts = driver.upload_with_retry(
        "gpu-owned-1", "/tmp/input.json", "/content/input.json", runner=run,
        sleeper=lambda seconds: None)

    assert attempts == 2
    assert len(calls) == 2
    assert all("--session gpu-owned-1" in command for command, _ in calls)
    assert all(kwargs == {"check": False} for _, kwargs in calls)


def test_upload_retry_fails_closed_after_bound() -> None:
    driver = _load_driver()
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return 1, "still unavailable"

    with pytest.raises(SystemExit, match="failed 3 times on the same session"):
        driver.upload_with_retry(
            "gpu-owned-1", "/tmp/input.json", "/content/input.json", runner=run,
            sleeper=lambda seconds: None)
    assert len(calls) == 3


def test_base_jobs_expand_in_frozen_model_seed_order_and_respect_only() -> None:
    driver = _load_driver()

    cells = driver.selected_base_jobs([
        "--phase", "base", "--models", "gpt", "nextlat", "--seeds", "1234", "1235",
    ])
    assert [cell[2] for cell in cells] == [
        "gpt-s1234-base", "gpt-s1235-base",
        "nextlat-s1234-base", "nextlat-s1235-base",
    ]
    assert driver.selected_base_jobs([
        "--phase", "base", "--only", "bst-s1238-base",
    ]) == [("bst", 1238, "bst-s1238-base")]
    with pytest.raises(SystemExit, match="unknown or non-base"):
        driver.selected_base_jobs(["--phase", "base", "--only", "gpt-s1234-adapt-near"])


def test_base_stages_evaluate_and_sync_before_next_training_stage(monkeypatch) -> None:
    driver = _load_driver()
    commands = []
    events = []

    def run(command, **kwargs):
        del kwargs
        commands.append(command)
        events.append("evaluate" if "evaluate_trained_bases.py" in command else "train")
        return 0, ""

    class Durability:
        def sync_once(self, ledger, ledger_object):
            del ledger, ledger_object
            job = "gpt-s1234-base" if events.count("sync") == 0 else "gpt-s1235-base"
            events.append("sync")
            return {job: {"step": 20000}}

    monkeypatch.setattr(driver, "sh", run)
    rc = driver.run_lurestar_base_stages(
        project="/content/project", root="/content/lurestar", ledger="/content/ledger.json",
        upstream="/content/upstream", args=["--phase", "base", "--models", "gpt",
        "--seeds", "1234", "1235"], data_dir="/content/data",
        manifest_dir="/content/manifests", durability=Durability())

    assert rc == 0
    assert events == ["train", "evaluate", "sync", "train", "evaluate", "sync"]
    assert "--only gpt-s1234-base" in commands[0]
    assert "--models gpt --seeds 1234" in commands[1]


def test_hmm_recovery_stage_durably_commits_all_ten_before_remaining_launcher(
        tmp_path) -> None:
    driver = _load_driver()
    ledger = tmp_path / "hmm_run_ledger.json"
    events = []
    provenance = {"checkpoint_creation_source_sha256": "a" * 64}
    entries = [
        {"seq": index, "job_id": job_id, "status": "TRAINED", "step": 3000,
         "recovery_provenance": provenance}
        for index, job_id in enumerate(driver.D41_RECOVERY_JOB_IDS)
    ]

    def run(command, **kwargs):
        del kwargs
        events.append("terminalize")
        assert "--recovery-barrier" in command
        assert "--only" in command
        ledger.write_text(json.dumps({"schema": 1, "entries": entries}))
        return 0, ""

    class Durability:
        def sync_once(self, path, ledger_object):
            assert Path(path) == ledger
            assert ledger_object == "hmm_run_ledger.json"
            events.append("durable")
            return {
                job_id: {"status": "TRAINED", "step": 3000,
                         "recovery_provenance": provenance}
                for job_id in driver.D41_RECOVERY_JOB_IDS
            }

    committed = driver.run_hmm_recovery_terminalization_stage(
        ["python", "run_hmm_matrix.py", "--family"], tmp_path / "barrier.json",
        driver.D41_RECOVERY_JOB_IDS, Durability(), ledger, command_runner=run)

    assert events == ["terminalize", "durable"]
    assert tuple(committed) == driver.D41_RECOVERY_JOB_IDS


def test_driver_restore_and_atomic_exact_ten_barrier_precede_any_hmm_launcher() -> None:
    driver = _load_driver()
    source = inspect.getsource(driver.driver)

    restore = source.index("restored = durability.restore()")
    barrier = source.index("recovery_barrier = write_runtime_recovery_barrier")
    launcher = source.index("run_hmm_recovery_terminalization_stage(")

    assert restore < barrier < launcher


def test_metrics_are_durable_before_first_checkpoint_without_committing_state(tmp_path) -> None:
    driver = _load_driver()
    root = tmp_path / "runtime"
    out = root / "runs" / "job"
    metrics = out / "experiment-seed1" / "version_0" / "metrics.csv"
    metrics.parent.mkdir(parents=True)
    metrics.write_text("step,loss\n7,1.25\n")
    materialized = out / "experiment-seed1" / "materialized_config.yaml"
    materialized.write_text("trainer:\n  max_steps: 3000\n")
    bucket = _FakeBucket()
    durability = driver.RuntimeDurability(
        bucket, root, prefix="lurestar", source_sha256="a" * 64,
        checkpoint_loader=lambda _: {})

    state = durability.sync_job({
        "job_id": "job", "status": "RUNNING", "out_root": str(out.resolve())})

    assert state is None
    assert "lurestar/runs/job/state.json" not in bucket.payloads
    receipt_name = "lurestar/runs/job/telemetry/latest.json"
    receipt = json.loads(bucket.payloads[receipt_name])
    assert receipt["resumable"] is False
    assert receipt["source_snapshot_sha256"] == "a" * 64
    remote_names = {item["remote"] for item in receipt["artifacts"]}
    assert any(name.endswith("metrics.csv") for name in remote_names)
    assert any(name.endswith("materialized_config.yaml") for name in remote_names)


def test_evaluation_progress_and_receipts_are_in_durability_set(tmp_path) -> None:
    driver = _load_driver()
    root = tmp_path / "runtime"
    out = root / "runs" / "job"
    evaluation = out / "evaluation"
    evaluation.mkdir(parents=True)
    (evaluation / "exact_path_raw.json.progress.json").write_text('{"next_index":128}\n')
    (evaluation / "base_competence.json").write_text('{"status":"pass"}\n')
    (evaluation / "base_competence.json.sha256").write_text("a" * 64 + "\n")
    bucket = _FakeBucket()
    durability = driver.RuntimeDurability(
        bucket, root, prefix="lurestar", source_sha256="b" * 64,
        checkpoint_loader=lambda _: {})

    durability.sync_job({
        "job_id": "job", "status": "TRAINED", "out_root": str(out.resolve())})

    receipt = json.loads(bucket.payloads["lurestar/runs/job/telemetry/latest.json"])
    names = {item["remote"] for item in receipt["artifacts"]}
    assert any(name.endswith("exact_path_raw.json.progress.json") for name in names)
    assert any(name.endswith("base_competence.json") for name in names)
    assert any(name.endswith("base_competence.json.sha256") for name in names)


def test_hmm_representation_cache_round_trips_recursively_with_progress_last(tmp_path) -> None:
    driver = _load_driver()
    root = tmp_path / "runtime"
    out = root / "runs" / "job"
    evaluation = out / "evaluation"
    chunks = evaluation / "representation_cache" / "chunks"
    chunks.mkdir(parents=True)
    chunk = chunks / "pairs-000000.npz"
    chunk.write_bytes(b"immutable representation arrays")
    chunk_sha = driver.sha256_file(chunk)
    sidecar = chunk.with_name(chunk.name + ".sha256")
    sidecar.write_text(f"{chunk_sha}  {chunk.name}\n")
    progress = evaluation / "representation_cache" / "progress.json"
    progress.write_text(json.dumps({
        "schema": "nextlat_forgetting/hmm_representation_cache/1",
        "identity": {"job_id": "job"},
        "identity_sha256": "1" * 64,
        "chunks": {"pairs-000000": {
            "path": str(chunk.resolve()),
            "sha256": chunk_sha,
            "sidecar": str(sidecar.resolve()),
            "sidecar_sha256": driver.sha256_file(sidecar),
            "bytes": chunk.stat().st_size,
        }},
    }, sort_keys=True))
    manifest = evaluation / "representation_manifest.json"
    manifest.write_text('{"schema":"nextlat_forgetting/hmm_representation_plan/1"}\n')
    final_receipt = evaluation / "hmm_geometry.json"
    final_receipt.write_text(json.dumps({
        "representation_manifest": {
            "path": str(manifest.resolve()), "sha256": driver.sha256_file(manifest)},
        "representation_cache": {
            "progress": {
                "path": str(progress.resolve()), "sha256": driver.sha256_file(progress)},
        },
    }))
    final_sidecar = evaluation / "hmm_geometry.json.sha256"
    final_sidecar.write_text(driver.sha256_file(final_receipt) + "\n")
    expected = {
        path.relative_to(evaluation): path.read_bytes()
        for path in (chunk, sidecar, progress, manifest, final_receipt, final_sidecar)
    }
    bucket = _FakeBucket()
    durability = driver.RuntimeDurability(
        bucket, root, prefix="lurestar", source_sha256="d" * 64,
        checkpoint_loader=lambda _: {})

    assert durability.sync_job({
        "job_id": "job", "status": "RUNNING", "out_root": str(out.resolve())}) is None

    receipt = json.loads(bucket.payloads["lurestar/runs/job/telemetry/latest.json"])
    assert receipt["resumable"] is True
    assert receipt["evaluation_cache"]["n_chunks"] == 1
    assert receipt["artifacts"][-1] == receipt["evaluation_cache"]["progress"]
    remote_names = {item["remote"] for item in receipt["artifacts"]}
    assert any(name.endswith("representation_cache/chunks/pairs-000000.npz")
               for name in remote_names)
    assert any(name.endswith("representation_cache/chunks/pairs-000000.npz.sha256")
               for name in remote_names)
    assert any(name.endswith("representation_cache/progress.json") for name in remote_names)
    assert any(name.endswith("representation_manifest.json") for name in remote_names)

    shutil.rmtree(evaluation)
    driver.RuntimeDurability(
        bucket, root, prefix="lurestar", source_sha256="d" * 64,
        checkpoint_loader=lambda _: {}).restore()

    for relative, payload in expected.items():
        assert (evaluation / relative).read_bytes() == payload
    assert driver.sha256_file(final_receipt) == final_sidecar.read_text().strip()
    restored_progress = json.loads(progress.read_text())
    assert driver.sha256_file(chunk) == restored_progress["chunks"]["pairs-000000"]["sha256"]


def test_hmm_cache_restore_does_not_publish_progress_when_a_chunk_is_corrupt(tmp_path) -> None:
    driver = _load_driver()
    root = tmp_path / "runtime"
    out = root / "runs" / "job"
    chunks = out / "evaluation" / "representation_cache" / "chunks"
    chunks.mkdir(parents=True)
    chunk = chunks / "pairs-000000.npz"
    chunk.write_bytes(b"valid")
    chunk_sha = driver.sha256_file(chunk)
    sidecar = chunk.with_name(chunk.name + ".sha256")
    sidecar.write_text(f"{chunk_sha}  {chunk.name}\n")
    progress = chunks.parent / "progress.json"
    progress.write_text(json.dumps({
        "schema": "nextlat_forgetting/hmm_representation_cache/1",
        "identity": {"job_id": "job"}, "identity_sha256": "2" * 64,
        "chunks": {"pairs-000000": {
            "path": str(chunk.resolve()), "sha256": chunk_sha,
            "sidecar": str(sidecar.resolve()),
            "sidecar_sha256": driver.sha256_file(sidecar),
        }},
    }))
    bucket = _FakeBucket()
    durability = driver.RuntimeDurability(bucket, root, source_sha256="e" * 64)
    durability.sync_job({
        "job_id": "job", "status": "RUNNING", "out_root": str(out.resolve())})
    receipt = json.loads(bucket.payloads["lurestar/runs/job/telemetry/latest.json"])
    chunk_remote = next(item["remote"] for item in receipt["artifacts"]
                        if item["local_path"] == str(chunk.resolve()))
    bucket.payloads[chunk_remote] = b"corrupt"
    shutil.rmtree(out / "evaluation")

    with pytest.raises(RuntimeError, match="restored telemetry failed verification"):
        driver.RuntimeDurability(bucket, root, source_sha256="e" * 64).restore()

    assert not progress.exists()


def test_optional_telemetry_failure_cannot_block_checkpoint_state_commit(tmp_path) -> None:
    driver = _load_driver()
    root = tmp_path / "runtime"
    out = root / "runs" / "job"
    experiment = out / "job-seed1"
    experiment.mkdir(parents=True)
    checkpoint = experiment / "recovery_ckpt_iter_250.pt"
    _write_verified_checkpoint(checkpoint, driver, step=250)
    (out / "recovery_ckpt").write_text(str(checkpoint.resolve()))
    bucket = _FakeBucket()
    messages = []
    durability = driver.RuntimeDurability(
        bucket, root, prefix="lurestar", source_sha256="c" * 64,
        checkpoint_loader=lambda path: json.loads(Path(path).read_text()),
        logger=messages.append)
    durability._sync_telemetry = lambda *_: (_ for _ in ()).throw(
        RuntimeError("live file changed"))

    state = durability.sync_job({
        "job_id": "job", "status": "RUNNING", "out_root": str(out.resolve())})

    assert state["step"] == 250
    assert "lurestar/runs/job/state.json" in bucket.payloads
    assert any("skipped after checkpoint commit" in message for message in messages)


def test_runtime_checkpoint_verification_rejects_filename_sidecar_loaded_step_mismatch(
        tmp_path) -> None:
    driver = _load_driver()
    root = tmp_path / "runtime"
    checkpoint = root / "runs/job/job/recovery_ckpt_iter_3000.pt"
    checkpoint.parent.mkdir(parents=True)
    _write_verified_checkpoint(checkpoint, driver, step=3000)
    metadata = json.loads((Path(str(checkpoint) + ".meta.json")).read_text())
    metadata["training_steps"] = 3001
    Path(str(checkpoint) + ".meta.json").write_text(json.dumps(metadata))

    with pytest.raises(RuntimeError, match="metadata does not match payload"):
        driver.RuntimeDurability(
            _FakeBucket(), root,
            checkpoint_loader=lambda path: json.loads(Path(path).read_text()),
        )._verify_checkpoint_for_sync(checkpoint, expected_step=3000)


def test_runtime_checkpoint_verification_normalizes_legacy_then_requires_canonical_step(
        tmp_path) -> None:
    driver = _load_driver()
    root = tmp_path / "runtime"
    checkpoint = root / "runs/job/job/recovery_ckpt_iter_3000.pt"
    checkpoint.parent.mkdir(parents=True)
    _write_verified_checkpoint(checkpoint, driver, step=3000)
    sidecar = Path(str(checkpoint) + ".meta.json")
    metadata = json.loads(sidecar.read_text())
    metadata["step"] = metadata.pop("training_steps")
    metadata["run_id"] = "job"
    sidecar.write_text(json.dumps(metadata))
    durability = driver.RuntimeDurability(
        _FakeBucket(), root,
        checkpoint_loader=lambda path: json.loads(Path(path).read_text()))
    binding = {
        "uri": "gs://test-bucket/lurestar/legacy-sidecar.json",
        "generation": "123",
        "sha256": driver.sha256_file(sidecar),
        "size_bytes": sidecar.stat().st_size,
    }

    with pytest.raises(RuntimeError, match="lacks canonical training_steps"):
        durability._verify_checkpoint_for_sync(
            checkpoint, expected_step=3000, expected_run_id="job")
    durability._verify_checkpoint_for_sync(
        checkpoint, expected_step=3000, expected_run_id="job",
        legacy_sidecar_binding=binding)
    metadata = json.loads(sidecar.read_text())
    assert metadata["step"] == metadata["training_steps"] == 3000
    durability._verify_checkpoint_for_sync(
        checkpoint, expected_step=3000, expected_run_id="job")

    metadata["training_steps"] = 3001
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="invalid step identity"):
        durability._verify_checkpoint_for_sync(
            checkpoint, expected_step=3000, expected_run_id="job")

    metadata["step"] = metadata["training_steps"] = True
    sidecar.write_text(json.dumps(metadata))
    with pytest.raises(RuntimeError, match="invalid step identity"):
        durability._verify_checkpoint_for_sync(
            checkpoint, expected_step=3000, expected_run_id="job")


@pytest.mark.parametrize("field,value", [("run_id", "other-job"), ("path", "/other.pt")])
def test_runtime_checkpoint_verification_keeps_sidecar_identity_strict(
        tmp_path, field, value) -> None:
    driver = _load_driver()
    root = tmp_path / "runtime"
    checkpoint = root / "runs/job/job/recovery_ckpt_iter_3000.pt"
    checkpoint.parent.mkdir(parents=True)
    _write_verified_checkpoint(checkpoint, driver, step=3000)
    sidecar = Path(str(checkpoint) + ".meta.json")
    metadata = json.loads(sidecar.read_text())
    metadata["run_id"] = "job"
    metadata[field] = value
    sidecar.write_text(json.dumps(metadata))

    with pytest.raises(RuntimeError, match="metadata does not match payload"):
        driver.RuntimeDurability(
            _FakeBucket(), root,
            checkpoint_loader=lambda path: json.loads(Path(path).read_text()),
        )._verify_checkpoint_for_sync(
            checkpoint, expected_step=3000, expected_run_id="job")


def test_predecessor_ledger_is_archived_before_fresh_successor_ledger(tmp_path) -> None:
    driver = _load_driver()
    root = tmp_path / "runtime"
    root.mkdir()
    ledger = root / "hmm_run_ledger.json"
    ledger.write_text(json.dumps({
        "schema": 1, "entries": [{"job_id": "poison", "status": "FAILED", "step": 3001}],
    }) + "\n")
    bucket = _FakeBucket()
    durability = driver.RuntimeDurability(bucket, root, prefix="lurestar")

    archived = driver.archive_predecessor_hmm_ledger(
        durability, ledger, "a" * 64)

    assert archived and Path(archived).is_file()
    assert json.loads(ledger.read_text()) == {"schema": 1, "entries": []}
    assert any(name.startswith("lurestar/recovery_audit/predecessor_ledgers/")
               for name in bucket.payloads)


def test_d41_legacy_generation_omission_migrates_then_successor_retry_is_strict_and_immutable(
        tmp_path) -> None:
    driver = _load_driver()
    bucket = _FakeBucket()
    root = tmp_path / "runtime"
    predecessor = "3" * 64
    successor = "4" * 64
    receipt_sha = "5" * 64
    receipt_jobs = []
    predecessor_objects = {}
    materialized_config_objects = {}
    mutable_operational_objects = {}

    def upload(local: Path, remote: str) -> dict:
        blob = bucket.blob(remote)
        digest = driver.sha256_file(local)
        blob.metadata = {"sha256": digest}
        blob.upload_from_filename(str(local))
        return {
            "uri": f"gs://{bucket.name}/{remote}",
            "generation": str(blob.generation),
            "sha256": digest,
            "size_bytes": local.stat().st_size,
        }

    for job_id in driver.D41_RECOVERY_JOB_IDS:
        out = root / "runs" / job_id
        experiment = out / "experiment"
        experiment.mkdir(parents=True)
        checkpoint = experiment / "recovery_ckpt_iter_3000.pt"
        _write_verified_checkpoint(checkpoint, driver, step=3000)
        sidecar = checkpoint.with_name(checkpoint.name + ".meta.json")
        legacy_sidecar = json.loads(sidecar.read_text())
        legacy_sidecar["step"] = legacy_sidecar.pop("training_steps")
        legacy_sidecar["run_id"] = job_id
        sidecar.write_text(json.dumps(legacy_sidecar, indent=2, sort_keys=True) + "\n")
        config = experiment / "materialized_config.yaml"
        config.write_text("trainer:\n  train_batches: 3000\n")
        durable_index = out / "durable_index.json"
        durable_index.write_text(json.dumps({
            "schema": 1, "records": [{"step": 3000, "path": str(checkpoint.resolve())}],
        }, sort_keys=True) + "\n")
        pointer = out / "latest_ckpt"
        pointer.write_text(str(checkpoint.resolve()))
        base = f"lurestar/runs/{job_id}"
        checkpoint_object = upload(checkpoint, f"{base}/experiment/{checkpoint.name}")
        sidecar_object = upload(sidecar, f"{base}/experiment/{sidecar.name}")
        config_object = upload(config, f"{base}/experiment/{config.name}")
        durable_index_object = upload(durable_index, f"{base}/durable_index.json")
        pointer_object = upload(pointer, f"{base}/latest_ckpt")
        state = {
            "schema": driver.STATE_SCHEMA,
            "run_id": job_id,
            "status": "TRAINED",
            "step": 3000,
            "out_root": str(out.resolve()),
            "pointer": str(pointer.resolve()),
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "sha256": checkpoint_object["sha256"],
                "size_bytes": checkpoint_object["size_bytes"],
            },
            "recovery_candidates": [{
                "path": str(checkpoint.resolve()),
                "sha256": checkpoint_object["sha256"],
                "size_bytes": checkpoint_object["size_bytes"],
                "step": 3000,
                "metadata_path": str(sidecar.resolve()),
                "metadata_sha256": sidecar_object["sha256"],
            }],
            "artifacts": [
                {"local_path": str(checkpoint.resolve()),
                 "remote": checkpoint_object["uri"].split(f"gs://{bucket.name}/", 1)[1],
                 **{key: checkpoint_object[key]
                    for key in ("sha256", "size_bytes")}},
                {"local_path": str(sidecar.resolve()),
                 "remote": sidecar_object["uri"].split(f"gs://{bucket.name}/", 1)[1],
                 **{key: sidecar_object[key]
                    for key in ("sha256", "size_bytes")}},
                {"local_path": str(config.resolve()),
                 "remote": config_object["uri"].split(f"gs://{bucket.name}/", 1)[1],
                 **{key: config_object[key]
                    for key in ("sha256", "size_bytes")}},
                {"local_path": str(durable_index.resolve()),
                 "remote": durable_index_object["uri"].split(
                     f"gs://{bucket.name}/", 1)[1],
                 **{key: durable_index_object[key]
                    for key in ("sha256", "size_bytes")}},
                {"local_path": str(pointer.resolve()),
                 "remote": pointer_object["uri"].split(f"gs://{bucket.name}/", 1)[1],
                 **{key: pointer_object[key]
                    for key in ("sha256", "size_bytes")}},
            ],
            "source_snapshot_sha256": predecessor,
        }
        state_path = tmp_path / f"{job_id}-state.json"
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        state_object = upload(state_path, f"{base}/state.json")
        receipt_jobs.append({
            "job_id": job_id,
            "model": job_id.split("-seed", 1)[0],
            "seed": int(job_id.split("seed", 1)[1].split("-", 1)[0]),
            "regime": "persistent_moderate",
            "target_step": 3000,
            "predecessor_source_sha256": predecessor,
            "state_object": state_object,
            "checkpoint_object": checkpoint_object,
            "sidecar_object": sidecar_object,
            "verification": {
                "state_trained_exact_target": True,
                "checkpoint_bytes_sha256_verified": True,
                "sidecar_bytes_sha256_verified": True,
                "sidecar_binds_checkpoint": True,
                "source_identity_verified": True,
                "payload_training_steps_verified": True,
            },
        })
        predecessor_objects[job_id] = (checkpoint_object, sidecar_object)
        materialized_config_objects[job_id] = config_object
        mutable_operational_objects[job_id] = (
            str(durable_index.resolve()), durable_index_object,
            str(pointer.resolve()), pointer_object,
        )

    receipt = {
        "schema": driver.D41_RECOVERY_RECEIPT_SCHEMA,
        "status": "PASS",
        "scientific_metrics_inspected": False,
        "jobs": receipt_jobs,
    }
    shutil.rmtree(root / "runs")
    loader = lambda path: json.loads(Path(path).read_text())

    # Literal live failure shape: later retries rewrote mutable durability indexes/pointers at
    # the same object names after the terminal predecessor state committed them.
    for (durable_local, durable_object, pointer_local,
         pointer_object) in mutable_operational_objects.values():
        del durable_local, pointer_local
        for binding, payload in (
            (durable_object, b'{"schema":1,"records":["later retry"]}\n'),
            (pointer_object, b"/content/later-retry/checkpoint.pt\n"),
        ):
            remote = binding["uri"].split(f"gs://{bucket.name}/", 1)[1]
            blob = bucket.blob(remote)
            blob.metadata = {"sha256": driver.hashlib.sha256(payload).hexdigest()}
            blob.upload_from_string(payload)

    # Materialized config is scientific execution identity, not a regenerable operational
    # index.  It stays hash-strict even on the legacy predecessor migration path.
    first_job = driver.D41_RECOVERY_JOB_IDS[0]
    config_binding = materialized_config_objects[first_job]
    config_remote = config_binding["uri"].split(f"gs://{bucket.name}/", 1)[1]
    original_config = bucket.versions[(config_remote, config_binding["generation"])][0]
    poisoned = bucket.blob(config_remote)
    poisoned.metadata = {"sha256": driver.hashlib.sha256(b"changed config\n").hexdigest()}
    poisoned.upload_from_string(b"changed config\n")
    with pytest.raises(RuntimeError, match="restored artifact failed verification"):
        driver.RuntimeDurability(
            bucket, root, source_sha256=successor,
            predecessor_source_sha256=predecessor, recovery_receipt=receipt,
            recovery_receipt_sha256=receipt_sha, checkpoint_loader=loader,
            logger=lambda _message: None).restore()
    shutil.rmtree(root / "runs", ignore_errors=True)
    config_restore = tmp_path / "materialized_config.restore.yaml"
    config_restore.write_bytes(original_config)
    upload(config_restore, config_remote)

    # The a962 predecessor state did not carry object generations.  Missing legacy fields are
    # accepted only because the clearance receipt supplies and byte-verifies the exact object.
    # A legacy field that is present but disagrees with the receipt remains a hard failure.
    first_record = receipt_jobs[0]
    predecessor_state_name = first_record["state_object"]["uri"].split(
        f"gs://{bucket.name}/", 1)[1]
    original_predecessor_state = bucket.payloads[predecessor_state_name]
    mismatched_state = json.loads(original_predecessor_state)
    mismatched_state["recovery_candidates"][0]["generation"] = "999999"
    mismatched_path = tmp_path / "mismatched-predecessor-state.json"
    mismatched_path.write_text(json.dumps(mismatched_state, indent=2, sort_keys=True) + "\n")
    first_record["state_object"] = upload(mismatched_path, predecessor_state_name)
    with pytest.raises(RuntimeError, match="exact checkpoint is absent or ambiguous"):
        driver.RuntimeDurability(
            bucket, root, source_sha256=successor,
            predecessor_source_sha256=predecessor, recovery_receipt=receipt,
            recovery_receipt_sha256=receipt_sha, checkpoint_loader=loader,
            logger=lambda _message: None).restore()
    shutil.rmtree(root / "runs", ignore_errors=True)
    original_path = tmp_path / "original-predecessor-state.json"
    original_path.write_bytes(original_predecessor_state)
    first_record["state_object"] = upload(original_path, predecessor_state_name)

    for durable_local, _durable_object, _pointer_local, _pointer_object in (
            mutable_operational_objects.values()):
        stale_local = Path(durable_local)
        stale_local.parent.mkdir(parents=True, exist_ok=True)
        stale_local.write_text('{"stale":"partial prior restore"}\n')

    durability = driver.RuntimeDurability(
        bucket, root, source_sha256=successor,
        predecessor_source_sha256=predecessor, recovery_receipt=receipt,
        recovery_receipt_sha256=receipt_sha, checkpoint_loader=loader,
        logger=lambda _message: None)
    restored = durability.restore()

    def remote_snapshot():
        return (
            dict(bucket.payloads), dict(bucket.metadata), dict(bucket.generations),
            dict(bucket.versions),
        )

    def assert_remote_snapshot_unchanged(before):
        assert bucket.payloads == before[0]
        assert bucket.metadata == before[1]
        assert bucket.generations == before[2]
        assert bucket.versions == before[3]

    # A receipt-bound sync that lost restore provenance must reject before any artifact can be
    # routed or uploaded. In particular, a normalized sidecar must never fall through to the
    # mutable predecessor object name and only then fail.
    first_job = driver.D41_RECOVERY_JOB_IDS[0]
    first_state = restored[first_job]
    provenance_probe = driver.RuntimeDurability(
        bucket, root, source_sha256=successor,
        predecessor_source_sha256=predecessor, recovery_receipt=receipt,
        recovery_receipt_sha256=receipt_sha, checkpoint_loader=loader,
        logger=lambda _message: None)
    before_remote = remote_snapshot()
    with pytest.raises(RuntimeError, match="lacks predecessor recovery provenance"):
        provenance_probe.sync_job({
            "job_id": first_job, "status": "TRAINED",
            "out_root": first_state["out_root"],
        })
    assert_remote_snapshot_unchanged(before_remote)

    wrong_provenance = dict(first_state["recovery_provenance"])
    wrong_provenance["sidecar_normalization"] = dict(
        wrong_provenance["sidecar_normalization"], to_sha256="f" * 64)
    before_remote = remote_snapshot()
    with pytest.raises(RuntimeError, match="normalized sidecar differs"):
        provenance_probe.sync_job({
            "job_id": first_job, "status": "TRAINED",
            "out_root": first_state["out_root"],
            "recovery_provenance": wrong_provenance,
        })
    assert_remote_snapshot_unchanged(before_remote)

    valid_barrier = driver.write_runtime_recovery_barrier(root, restored, {
        "source_sha256": successor,
        "predecessor_source_sha256": predecessor,
        "recovery_receipt_sha256": receipt_sha,
        "recovery_job_ids": list(driver.D41_RECOVERY_JOB_IDS),
    })
    assert tuple(json.loads(valid_barrier.read_text())["job_ids"]) == (
        driver.D41_RECOVERY_JOB_IDS)

    missing_artifacts = dict(restored)
    missing_artifacts[first_job] = dict(restored[first_job], artifacts=[])
    with pytest.raises(SystemExit, match="no authoritative completion artifacts"):
        driver.write_runtime_recovery_barrier(root, missing_artifacts, {
            "source_sha256": successor,
            "predecessor_source_sha256": predecessor,
            "recovery_receipt_sha256": receipt_sha,
            "recovery_job_ids": list(driver.D41_RECOVERY_JOB_IDS),
        })

    # A migrated D41 target can coexist with stale local recovery files left by the failed
    # 3,001 retry.  The target-only terminal transaction must neither normalize nor upload a
    # 2,750 remnant.  Conversely, making that remnant an active pointer is a pre-upload refusal.
    exact_checkpoint = Path(first_state["restored_checkpoint"])
    legacy_remnant = exact_checkpoint.with_name("recovery_ckpt_iter_2750.pt")
    _write_verified_checkpoint(legacy_remnant, driver, step=2750)
    legacy_sidecar = legacy_remnant.with_name(legacy_remnant.name + ".meta.json")
    legacy_metadata = json.loads(legacy_sidecar.read_text())
    legacy_metadata["step"] = legacy_metadata.pop("training_steps")
    legacy_metadata["run_id"] = first_job
    legacy_sidecar.write_text(json.dumps(legacy_metadata, indent=2, sort_keys=True) + "\n")
    legacy_sidecar_before = legacy_sidecar.read_bytes()
    stale_pointer = Path(first_state["out_root"]) / "recovery_ckpt"
    stale_pointer.write_text(str(legacy_remnant.resolve()))
    before_remote = remote_snapshot()
    with pytest.raises(RuntimeError, match="receipt-bound pointers do not name one exact target"):
        durability.sync_job({
            "job_id": first_job, "status": "TRAINED",
            "out_root": first_state["out_root"],
            "recovery_provenance": first_state["recovery_provenance"],
        })
    assert_remote_snapshot_unchanged(before_remote)
    assert legacy_sidecar.read_bytes() == legacy_sidecar_before
    assert str(legacy_remnant) not in durability._sidecar_normalizations
    stale_pointer.unlink()

    successor_generations = {}
    for job_id in driver.D41_RECOVERY_JOB_IDS:
        state = restored[job_id]
        normalized_sidecar = Path(state["restored_checkpoint"] + ".meta.json")
        normalized_metadata = json.loads(normalized_sidecar.read_text())
        assert normalized_metadata["step"] == normalized_metadata["training_steps"] == 3000
        normalization = state["recovery_provenance"]["sidecar_normalization"]
        assert normalization["from_sha256"] == predecessor_objects[job_id][1]["sha256"]
        assert normalization["to_sha256"] == driver.sha256_file(normalized_sidecar)
        assert normalization["from_sha256"] != normalization["to_sha256"]
        synced = durability.sync_job({
            "job_id": job_id,
            "status": "TRAINED",
            "out_root": state["out_root"],
            "recovery_provenance": state["recovery_provenance"],
        })
        provenance = synced["recovery_provenance"]
        checkpoint_object, sidecar_object = predecessor_objects[job_id]
        assert provenance["checkpoint_generation"] == checkpoint_object["generation"]
        assert provenance["sidecar_generation"] == sidecar_object["generation"]
        assert provenance["predecessor_checkpoint_object"] == checkpoint_object
        assert provenance["predecessor_sidecar_object"] == sidecar_object
        assert provenance["successor_checkpoint_object"]["uri"] != checkpoint_object["uri"]
        assert provenance["successor_sidecar_object"]["uri"] != sidecar_object["uri"]
        assert provenance["successor_sidecar_object"]["sha256"] == (
            provenance["sidecar_normalization"]["to_sha256"])
        durable_local = mutable_operational_objects[job_id][0]
        assert not Path(durable_local).exists()
        # The local restored state records the outcome-blind skip. It is not copied into the
        # successor state commit after terminalization regenerates operational bookkeeping.
        assert any(item["local_path"] == durable_local for item in
                   restored[job_id]["restore_provenance"]
                   ["skipped_regenerable_predecessor_artifacts"])
        if job_id == first_job:
            assert {item["path"] for item in synced["recovery_candidates"]} == {
                str(exact_checkpoint)}
            assert str(legacy_remnant) not in {
                item["local_path"] for item in synced["artifacts"]}
            assert not any(legacy_remnant.name in remote for remote in bucket.payloads)
            assert legacy_sidecar.read_bytes() == legacy_sidecar_before
            assert str(legacy_remnant) not in durability._sidecar_normalizations
        successor_generations[job_id] = (
            dict(provenance["successor_checkpoint_object"]),
            dict(provenance["successor_sidecar_object"]),
        )

    # Once state belongs to the successor, generation fields are mandatory.  The successor
    # state is the only authority for retry objects, so omission cannot inherit legacy grace.
    successor_state_name = f"lurestar/runs/{first_job}/state.json"
    original_successor_state = bucket.payloads[successor_state_name]
    missing_generation_state = json.loads(original_successor_state)
    missing_generation_state["recovery_candidates"][0].pop("generation", None)
    missing_generation_path = tmp_path / "missing-successor-generation-state.json"
    missing_generation_path.write_text(
        json.dumps(missing_generation_state, indent=2, sort_keys=True) + "\n")
    upload(missing_generation_path, successor_state_name)
    shutil.rmtree(root / "runs")
    with pytest.raises(RuntimeError, match="exact checkpoint is absent or ambiguous"):
        driver.RuntimeDurability(
            bucket, root, source_sha256=successor,
            predecessor_source_sha256=predecessor, recovery_receipt=receipt,
            recovery_receipt_sha256=receipt_sha, checkpoint_loader=loader,
            logger=lambda _message: None).restore()
    restored_successor_path = tmp_path / "restored-successor-state.json"
    restored_successor_path.write_bytes(original_successor_state)
    upload(restored_successor_path, successor_state_name)

    # Make every receipt-pinned predecessor payload unavailable.  A second disconnect can
    # succeed only if current-source restore uses the separately immutable successor objects.
    for checkpoint_object, sidecar_object in predecessor_objects.values():
        for binding in (checkpoint_object, sidecar_object):
            remote = binding["uri"].split(f"gs://{bucket.name}/", 1)[1]
            bucket.versions.pop((remote, binding["generation"]))
    shutil.rmtree(root / "runs", ignore_errors=True)
    retry = driver.RuntimeDurability(
        bucket, root, source_sha256=successor,
        predecessor_source_sha256=predecessor, recovery_receipt=receipt,
        recovery_receipt_sha256=receipt_sha, checkpoint_loader=loader,
        logger=lambda _message: None)
    recovered = retry.restore()
    for job_id in driver.D41_RECOVERY_JOB_IDS:
        state = retry.sync_job({
            "job_id": job_id, "status": "TRAINED",
            "out_root": recovered[job_id]["out_root"],
            "recovery_provenance": recovered[job_id]["recovery_provenance"],
        })
        provenance = state["recovery_provenance"]
        assert provenance["successor_checkpoint_object"] == successor_generations[job_id][0]
        assert provenance["successor_sidecar_object"] == successor_generations[job_id][1]


class _FakeBlob:
    def __init__(self, bucket, name, generation=None):
        self.bucket = bucket
        self.name = name
        self.requested_generation = None if generation is None else str(generation)
        self.generation = self.requested_generation
        self.metadata = {}
        self.size = None

    def upload_from_filename(self, path, if_generation_match=None):
        if if_generation_match == 0 and self.name in self.bucket.payloads:
            raise RuntimeError("PreconditionFailed")
        if self.requested_generation is not None:
            raise RuntimeError("cannot upload a historical generation")
        self.bucket.payloads[self.name] = Path(path).read_bytes()
        self.bucket.metadata[self.name] = dict(self.metadata or {})
        self.bucket.generations[self.name] = self.bucket.generations.get(self.name, 0) + 1
        generation = str(self.bucket.generations[self.name])
        self.bucket.versions[(self.name, generation)] = (
            self.bucket.payloads[self.name], dict(self.bucket.metadata[self.name]))
        self.reload()

    def upload_from_string(self, payload, content_type=None):
        del content_type
        if isinstance(payload, str):
            payload = payload.encode()
        if self.requested_generation is not None:
            raise RuntimeError("cannot upload a historical generation")
        self.bucket.payloads[self.name] = bytes(payload)
        self.bucket.metadata[self.name] = dict(self.metadata or {})
        self.bucket.generations[self.name] = self.bucket.generations.get(self.name, 0) + 1
        generation = str(self.bucket.generations[self.name])
        self.bucket.versions[(self.name, generation)] = (
            self.bucket.payloads[self.name], dict(self.bucket.metadata[self.name]))
        self.reload()

    def download_as_bytes(self):
        if self.requested_generation is not None:
            return self.bucket.versions[(self.name, self.requested_generation)][0]
        return self.bucket.payloads[self.name]

    def download_to_filename(self, path):
        Path(path).write_bytes(self.download_as_bytes())

    def reload(self):
        if self.requested_generation is not None:
            version = self.bucket.versions.get((self.name, self.requested_generation))
            payload = version[0] if version is not None else None
            metadata = version[1] if version is not None else {}
        else:
            payload = self.bucket.payloads.get(self.name)
            metadata = self.bucket.metadata.get(self.name, {})
            current = self.bucket.generations.get(self.name)
            self.generation = str(current) if current is not None else None
        self.size = len(payload) if payload is not None else None
        self.metadata = dict(metadata)


class _FakeBucket:
    name = "test-bucket"

    def __init__(self):
        self.payloads = {}
        self.metadata = {}
        self.generations = {}
        self.versions = {}
        self.blobs = {}

    def blob(self, name, generation=None):
        key = (name, None if generation is None else str(generation))
        return self.blobs.setdefault(key, _FakeBlob(self, name, generation))

    def list_blobs(self, prefix):
        return [self.blob(name) for name in sorted(self.payloads) if name.startswith(prefix)]


def test_adc_is_restricted_before_runtime_authentication(tmp_path) -> None:
    driver = _load_driver()
    adc = tmp_path / "adc.json"
    adc.write_text("{}")
    os.chmod(adc, 0o644)

    driver.secure_adc(adc)

    assert stat.S_IMODE(adc.stat().st_mode) == 0o600


def test_runtime_driver_contains_no_gcloud_storage_invocation() -> None:
    driver = _load_driver()
    source = inspect.getsource(driver.driver)
    downloader_source = inspect.getsource(driver.download_required_input_corpora)
    assert "gcloud storage" not in source
    assert 'download_prefix(input_bundle_prefix + "/manifests"' in source
    assert 'input_bundle_prefix + "/corpus/hmm_family"' in downloader_source
    assert 'input_bundle_prefix + "/corpus/hmm"' not in downloader_source
    assert "RUNTIME_INPUT_SUBSET_VERIFIED" in source
    assert "storage.Client(project=GCP_PROJECT)" in source


@pytest.mark.parametrize("phase", ["train", "evaluate"])
def test_hmm_family_only_bundle_reaches_runtime_dispatch_inputs(tmp_path, phase) -> None:
    driver = _load_driver()

    class Downloads:
        def __init__(self):
            self.calls = []

        def download_prefix(self, prefix, destination):
            self.calls.append((prefix, destination))
            return 1

    durability = Downloads()
    bundle = driver.PREFIX + "/input_bundles/" + "a" * 64
    dispatch = {"runner": "hmm", "phase": phase, "family": True}
    resolved = driver.download_required_input_corpora(
        durability, tmp_path / "data", bundle, dispatch)

    assert durability.calls == [
        (bundle + "/corpus/hmm_family", str(tmp_path / "data" / "hmm_family")),
    ]
    assert resolved == {
        bundle + "/corpus/hmm_family": str(tmp_path / "data" / "hmm_family")}
    assert not (tmp_path / "data" / "hmm").exists()
    assert (tmp_path / "data" / "hmm_family").is_dir()


def test_hmm_family_download_refuses_missing_family_before_optimizer_dispatch(tmp_path) -> None:
    driver = _load_driver()

    class Downloads:
        def __init__(self):
            self.calls = []

        def download_prefix(self, prefix, _destination):
            self.calls.append(prefix)
            return 0

    durability = Downloads()
    bundle = driver.PREFIX + "/input_bundles/" + "a" * 64
    with pytest.raises(SystemExit, match="frozen HMM-family arrays"):
        driver.download_required_input_corpora(
            durability, tmp_path / "data", bundle,
            {"runner": "hmm", "phase": "train", "family": True})
    assert durability.calls == [bundle + "/corpus/hmm_family"]

    source = inspect.getsource(driver.driver)
    download_index = source.index("download_required_input_corpora(")
    verify_index = source.index("verify_runtime_input_subset(")
    runner_index = source.index('runner = dispatch["runner"]')
    child_dispatch_index = source.index('if runner == "hmm":', runner_index)
    assert download_index < verify_index < runner_index < child_dispatch_index


def test_lurestar_runtime_does_not_pull_any_hmm_corpus(tmp_path) -> None:
    driver = _load_driver()

    class Downloads:
        def __init__(self):
            self.calls = []

        def download_prefix(self, prefix, destination):
            self.calls.append((prefix, destination))
            raise AssertionError("Lure-Star must not request an HMM corpus")

    durability = Downloads()
    assert driver.download_required_input_corpora(
        durability, tmp_path / "data",
        driver.PREFIX + "/input_bundles/" + "a" * 64,
        {"runner": "lurestar", "phase": "base", "only": ()}) == {}
    assert durability.calls == []


def test_runtime_subset_verification_is_runner_exact(tmp_path) -> None:
    driver = _load_driver()
    runtime = tmp_path / "runtime"
    manifests = runtime / "manifests"
    family = runtime / "data" / "hmm_family" / "regime"
    legacy = runtime / "data" / "hmm"
    manifests.mkdir(parents=True)
    family.mkdir(parents=True)
    legacy.mkdir(parents=True)
    manifest = manifests / "frozen.json"
    family_array = family / "train.npy"
    legacy_array = legacy / "train.npy"
    manifest.write_text("frozen\n")
    family_array.write_bytes(b"family")
    legacy_array.write_bytes(b"legacy")
    inventory = manifests / "manifest_inventory.sha256"
    inventory.write_text("\n".join([
        f"{driver.sha256_file(manifest)}  manifests/frozen.json",
        f"{driver.sha256_file(family_array)}  data/hmm_family/regime/train.npy",
        f"{driver.sha256_file(legacy_array)}  data/hmm/train.npy",
    ]) + "\n")

    verified = driver.verify_runtime_input_subset(
        inventory, runtime, {"runner": "hmm", "phase": "train", "family": True})
    assert verified == {"manifests/": 1, "data/hmm_family/": 1}

    family_array.unlink()
    with pytest.raises(SystemExit, match="required runtime input is absent"):
        driver.verify_runtime_input_subset(
            inventory, runtime, {"runner": "hmm", "phase": "train", "family": True})

    # The same frozen bundle remains valid for Lure-Star without downloading any HMM arrays.
    verified = driver.verify_runtime_input_subset(
        inventory, runtime, {"runner": "lurestar", "phase": "base", "only": ()})
    assert verified == {"manifests/": 1}


def test_runtime_driver_has_validated_hmm_dispatch_and_separate_ledger() -> None:
    driver = _load_driver()
    source = inspect.getsource(driver.driver)
    assert "validate_confirmatory_job_spec" in source
    assert "require_session=True" in source
    assert 'ledger_object = "hmm_run_ledger.json"' in source
    assert '"run_hmm_matrix.py"' in source
    assert '"--snapshot-root", root' in source
    assert '"--data-root", root' in source
    assert '"--phase", phase, "--family", "--driver-managed-durability"' in source
    assert '"--driver-managed-durability"' in source
    assert 'phase = dispatch["phase"]' in source


def test_sync_uses_requested_noncolliding_ledger_object(tmp_path) -> None:
    driver = _load_driver()
    bucket = _FakeBucket()
    root = tmp_path / "runtime"
    root.mkdir()
    ledger = root / "hmm_run_ledger.json"
    ledger.write_text('{"schema":1,"entries":[]}\n')
    durability = driver.RuntimeDurability(bucket, root)

    assert durability.sync_once(ledger, ledger_object="hmm_run_ledger.json") == {}
    assert "lurestar/hmm_run_ledger.json" in bucket.payloads
    assert "lurestar/run_ledger.json" not in bucket.payloads
    with pytest.raises(RuntimeError, match="one filename"):
        durability.sync_once(ledger, ledger_object="../escape.json")


def test_runtime_bootstrap_installs_pinned_requirements_without_replacing_torch() -> None:
    driver = _load_driver()
    source = inspect.getsource(driver.driver)

    assert "grep -v '^torch' requirements.txt" in source
    assert "expected_torch_version=original_torch" in source
    assert "verify_runtime_gpu(requested_gpu)" in source


def test_during_run_sync_commits_verified_state_and_restores_exact_paths(tmp_path) -> None:
    driver = _load_driver()
    bucket = _FakeBucket()
    root = tmp_path / "lurestar"
    out = root / "runs" / "gpt" / "seed1234" / "base"
    experiment = out / "gpt-s1234-base-seed1234"
    version = experiment / "version_0"
    version.mkdir(parents=True)
    checkpoint = experiment / "recovery_ckpt_iter_250.pt"
    _write_verified_checkpoint(checkpoint, driver, step=250)
    original_checkpoint = checkpoint.read_bytes()
    previous = experiment / "recovery_ckpt_iter_0.pt"
    _write_verified_checkpoint(previous, driver, step=0)
    pointer = out / "recovery_ckpt"
    pointer.write_text(str(checkpoint.resolve()))
    (experiment / "materialized_config.yaml").write_text("seed: 1234\n")
    (version / "metrics.csv").write_text("step,loss\n250,0.1\n")
    metrics = out / "metrics"
    metrics.mkdir()
    (metrics / "step_250.json").write_text('{"step": 250}\n')
    evaluation = out / "evaluation"
    evaluation.mkdir()
    competence = evaluation / "base_competence.json"
    competence.write_text('{"status":"pass"}\n')
    patch_audit = root / "source_snapshot" / "runtime_patch"
    patch_audit.mkdir(parents=True)
    (patch_audit / "runtime_patch.diff").write_text("verified diff\n")
    (patch_audit / "runtime_patch_receipt.json").write_text('{"schema": 1}\n')
    ledger = root / "run_ledger.json"
    ledger.write_text(json.dumps({
        "schema": 1,
        "entries": [{
            "job_id": "gpt-s1234-base",
            "status": "RUNNING",
            "out_root": str(out.resolve()),
            "artifacts": {"evaluation/base_competence.json":
                          driver.sha256_file(competence)},
        }],
    }))
    loader = lambda path: json.loads(Path(path).read_text())
    durability = driver.RuntimeDurability(bucket, root, logger=lambda message: None,
                                          checkpoint_loader=loader)

    states = durability.sync_once(ledger)

    state = states["gpt-s1234-base"]
    assert state["step"] == 250
    assert state["checkpoint"]["path"] == str(checkpoint.resolve())
    assert state["checkpoint"]["sha256"] == driver.sha256_file(checkpoint)
    remote_state = "lurestar/runs/gpt-s1234-base/state.json"
    assert json.loads(bucket.payloads[remote_state]) == state
    assert bucket.metadata[remote_state]["sha256"]
    remote_names = set(bucket.payloads)
    assert "lurestar/runs/gpt-s1234-base/recovery_ckpt" in remote_names
    assert any(name.endswith("recovery_ckpt_iter_250.pt") for name in remote_names)
    assert any(name.endswith("recovery_ckpt_iter_0.pt") for name in remote_names)
    assert any(name.endswith("runtime_patch/runtime_patch.diff") for name in remote_names)
    assert any(name.endswith("runtime_patch/runtime_patch_receipt.json")
               for name in remote_names)
    assert any(name.endswith("materialized_config.yaml") for name in remote_names)
    assert any(name.endswith("metrics.csv") for name in remote_names)
    assert any(name.endswith("evaluation/base_competence.json") for name in remote_names)

    shutil.rmtree(root / "runs")
    restored = driver.RuntimeDurability(
        bucket, root, logger=lambda message: None, checkpoint_loader=loader).restore()

    assert restored["gpt-s1234-base"]["step"] == 250
    assert checkpoint.read_bytes() == original_checkpoint
    assert previous.is_file()
    assert pointer.read_text().strip() == str(checkpoint.resolve())
    assert driver.sha256_file(checkpoint) == state["checkpoint"]["sha256"]
    assert competence.read_text() == '{"status":"pass"}\n'

    newest_remote = next(
        name for name in bucket.payloads if name.endswith("recovery_ckpt_iter_250.pt"))
    bucket.payloads[newest_remote] = b"corrupt-newest-generation"
    shutil.rmtree(root / "runs")
    recovered = driver.RuntimeDurability(
        bucket, root, logger=lambda message: None, checkpoint_loader=loader).restore()

    assert recovered["gpt-s1234-base"]["restored_step"] == 0
    assert previous.is_file()
    assert pointer.read_text().strip() == str(previous.resolve())


def test_fast_exit_progress_requires_a_durable_step_or_checkpoint_advance() -> None:
    driver = _load_driver()
    before = {"job": {"step": 250, "checkpoint_sha256": "aaa"}}

    assert not driver.durable_progress_advanced(before, before)
    assert driver.durable_progress_advanced(
        before, {"job": {"step": 500, "checkpoint_sha256": "bbb"}})
    assert driver.durable_progress_advanced(
        before, {"job": {"step": 250, "checkpoint_sha256": "bbb"}})


def test_monitor_preserves_active_session_while_durable_progress_advances() -> None:
    driver = _load_driver()
    statuses = iter([
        ({"status": "connected", "session": "gpu-owned"},
         {"status": "connected", "session": "gpu-owned"}),
        ({"status": "connected", "session": "gpu-owned"},
         {"status": "connected", "session": "gpu-owned"}),
    ])
    progress = iter([
        {"job": {"step": 250, "checkpoint_sha256": "a"}},
        {"job": {"step": 500, "checkpoint_sha256": "b"}},
        {"job": {"step": 500, "checkpoint_sha256": "b"}},
    ])
    markers = iter([None, {
        "schema": "nextlat_forgetting/runtime_terminal/1",
        "session_id": "gpu-owned",
        "source_snapshot_sha256": "source",
        "training_complete": True,
    }])
    stop_calls = []

    outcome = driver.monitor_owned_runtime(
        "gpu-owned", "source", {"job": {"step": 0, "checkpoint_sha256": "z"}},
        status_reader=lambda: next(statuses),
        progress_reader=lambda: next(progress),
        terminal_reader=lambda *_: next(markers),
        stall_limit=1,
    )

    assert outcome["reason"] == "terminal"
    assert outcome["progress"]["job"]["step"] == 500
    assert stop_calls == []


def test_monitor_treats_telemetry_as_liveness_but_not_resume_progress() -> None:
    driver = _load_driver()
    statuses = iter([
        ({"status": "connected", "session": "gpu-owned"},
         {"status": "connected", "session": "gpu-owned"}),
        ({"status": "connected", "session": "gpu-owned"},
         {"status": "connected", "session": "gpu-owned"}),
    ])
    activities = iter([
        {"job": {"synced_at": 1.0}},
        {"job": {"synced_at": 2.0}},
    ])
    markers = iter([None, {
        "schema": "nextlat_forgetting/runtime_terminal/1",
        "session_id": "gpu-owned",
        "source_snapshot_sha256": "source",
        "training_complete": True,
    }])
    unchanged = {"job": {"step": 0, "checkpoint_sha256": "a"}}

    outcome = driver.monitor_owned_runtime(
        "gpu-owned", "source", unchanged,
        status_reader=lambda: next(statuses), progress_reader=lambda: unchanged,
        terminal_reader=lambda *_: next(markers),
        activity_reader=lambda: next(activities), stall_limit=1)

    assert outcome["reason"] == "terminal"
    assert outcome["progress"] == unchanged


@pytest.mark.parametrize("status_pair", [
    ({"status": "connected"}, {"status": "connected"}),
    ({"status": "connected", "session": "gpu-owned"},
     {"status": "connected", "session": "gpu-changed"}),
])
def test_monitor_revalidates_sid_and_returns_read_only_on_uncertain_ownership(
        status_pair) -> None:
    driver = _load_driver()
    calls = []
    outcome = driver.monitor_owned_runtime(
        "gpu-owned", "source", {}, status_reader=lambda: status_pair,
        progress_reader=lambda: calls.append("progress") or {},
        terminal_reader=lambda *_: calls.append("terminal") or None,
        stall_limit=1)
    assert outcome["reason"] == "ownership_uncertain"
    # Identity is checked before either terminal state or progress can authorize mutation.
    assert calls == []


def test_stop_requires_terminal_or_stalled_and_two_read_gone_verdict(tmp_path) -> None:
    driver = _load_driver()
    session_file = tmp_path / ".colab_session"
    session_file.write_text("gpu-owned\n")
    stops = []

    with pytest.raises(ValueError, match="non-terminal reason"):
        driver.stop_owned_runtime(
            "gone", session_file, stopper=lambda: stops.append("stop"),
            status_reader=lambda: ({"status": "no_runtime"}, {"status": "no_runtime"}))
    assert stops == []
    assert session_file.is_file()

    driver.stop_owned_runtime(
        "terminal", session_file, stopper=lambda: stops.append("stop"),
        status_reader=lambda: ({"status": "no_runtime"}, {"status": "no_runtime"}))
    assert stops == ["stop"]
    assert not session_file.exists()

    session_file.write_text("gpu-owned\n")
    driver.stop_owned_runtime(
        "input-upload-failed", session_file, stopper=lambda: stops.append("stop-input"),
        status_reader=lambda: ({"status": "no_runtime"}, {"status": "no_runtime"}))
    assert stops[-1] == "stop-input"
    assert not session_file.exists()


def test_loop_monitors_owned_active_runtime_before_any_exec() -> None:
    driver = _load_driver()
    source = inspect.getsource(driver._owned_loop)
    active_branch = source.index('if runtime_state == "active":')
    monitor_call = source.index("outcome = monitor_owned_runtime", active_branch)
    exec_call = source.index("colab exec --session", active_branch)

    assert monitor_call < exec_call


def test_restore_rejects_state_json_whose_committed_hash_does_not_verify(tmp_path) -> None:
    driver = _load_driver()
    bucket = _FakeBucket()
    root = tmp_path / "lurestar"
    state_name = "lurestar/runs/job/state.json"
    payload = json.dumps({
        "schema": driver.STATE_SCHEMA,
        "run_id": "job",
        "checkpoint": {"path": str(root / "runs/job/a.pt"), "sha256": "x",
                       "size_bytes": 1},
        "artifacts": [],
        "pointer": str(root / "runs/job/recovery_ckpt"),
    }).encode()
    bucket.payloads[state_name] = payload
    bucket.metadata[state_name] = {"sha256": "not-the-payload-hash"}

    with pytest.raises(RuntimeError, match="state.json hash metadata mismatch"):
        driver.RuntimeDurability(bucket, root, logger=lambda message: None).restore()


def test_restore_refuses_predecessor_without_exact_recovery_receipt(tmp_path) -> None:
    driver = _load_driver()
    bucket = _FakeBucket()
    root = tmp_path / "lurestar"
    out = root / "runs" / "job"
    experiment = out / "job-seed1234"
    experiment.mkdir(parents=True)
    checkpoint = experiment / "recovery_ckpt_iter_250.pt"
    _write_verified_checkpoint(checkpoint, driver, step=250)
    pointer = out / "recovery_ckpt"
    pointer.write_text(str(checkpoint.resolve()))
    old_source = "3" * 64
    current_source = "4" * 64
    entry = {"job_id": "job", "status": "RUNNING", "out_root": str(out.resolve())}
    loader = lambda path: json.loads(Path(path).read_text())
    driver.RuntimeDurability(
        bucket, root, source_sha256=old_source, checkpoint_loader=loader,
        logger=lambda _: None).sync_job(entry)
    shutil.rmtree(root / "runs")
    messages = []

    with pytest.raises(ValueError, match="exact PASS D41 receipt"):
        driver.RuntimeDurability(
            bucket, root, source_sha256=current_source,
            predecessor_source_sha256=old_source, checkpoint_loader=loader,
            logger=messages.append)


@pytest.mark.parametrize("predecessor", [None, "5" * 64])
def test_restore_refuses_undeclared_or_arbitrary_predecessor_source(
        tmp_path, predecessor) -> None:
    driver = _load_driver()
    bucket = _FakeBucket()
    root = tmp_path / "lurestar"
    out = root / "runs" / "job"
    experiment = out / "job-seed1234"
    experiment.mkdir(parents=True)
    checkpoint = experiment / "recovery_ckpt_iter_250.pt"
    _write_verified_checkpoint(checkpoint, driver, step=250)
    (out / "recovery_ckpt").write_text(str(checkpoint.resolve()))
    loader = lambda path: json.loads(Path(path).read_text())
    driver.RuntimeDurability(
        bucket, root, source_sha256="3" * 64, checkpoint_loader=loader,
        logger=lambda _: None).sync_job(
            {"job_id": "job", "status": "RUNNING", "out_root": str(out.resolve())})
    shutil.rmtree(root / "runs")

    if predecessor is None:
        with pytest.raises(RuntimeError, match="source snapshot does not match"):
            driver.RuntimeDurability(
                bucket, root, source_sha256="4" * 64,
                predecessor_source_sha256=predecessor, checkpoint_loader=loader,
                logger=lambda _: None).restore()
    else:
        with pytest.raises(ValueError, match="exact PASS D41 receipt"):
            driver.RuntimeDurability(
                bucket, root, source_sha256="4" * 64,
                predecessor_source_sha256=predecessor, checkpoint_loader=loader,
                logger=lambda _: None)


def test_trained_job_uses_verified_ledger_checkpoint_when_upstream_left_no_pointer(tmp_path) -> None:
    driver = _load_driver()
    bucket = _FakeBucket()
    root = tmp_path / "lurestar"
    out = root / "runs" / "job"
    experiment = out / "job-seed1234"
    experiment.mkdir(parents=True)
    checkpoint = experiment / "ckpt_iter_20000_0.1.pt"
    _write_verified_checkpoint(checkpoint, driver, step=20_000)
    entry = {
        "job_id": "job",
        "status": "TRAINED",
        "out_root": str(out.resolve()),
        "final_checkpoint": str(checkpoint.resolve()),
        "final_checkpoint_sha256": driver.sha256_file(checkpoint),
    }

    state = driver.RuntimeDurability(
        bucket, root, logger=lambda message: None,
        checkpoint_loader=lambda path: json.loads(Path(path).read_text())).sync_job(entry)

    latest = out / "latest_ckpt"
    assert latest.read_text() == str(checkpoint.resolve())
    assert state["pointer"] == str(latest.resolve())
    assert state["step"] == 20000
