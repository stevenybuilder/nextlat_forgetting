from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "colab_cfs1_loop.py"


def _module():
    spec = importlib.util.spec_from_file_location("colab_cfs1_loop_tested", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _spec(module, *, session=True):
    source, manifest, parents = "a" * 64, "b" * 64, "c" * 64
    spec = {
        "schema": module.CFS1_SPEC_SCHEMA, "runner": "cfs1_adaptation", "gpu": "a100",
        "source_sha256": source, "source_object": f"cfs1/source/project-{source}.tar.gz",
        "update_manifest_sha256": manifest,
        "update_manifest_object": f"cfs1/inputs/{manifest}/cfs1_update_manifest.json",
        "parent_ledger_sha256": parents,
        "parent_ledger_object": f"cfs1/inputs/{parents}/parents.json",
    }
    if session:
        spec["session_id"] = "gpu-a100-cfs1-test"
    return spec


def test_strict_cfs1_spec_and_owned_runtime_pairing() -> None:
    module = _module()
    spec = _spec(module)
    assert module.validate_cfs1_job_spec(spec, require_session=True)["source_sha256"] == "a" * 64
    argv = module.runtime_runner_argv(spec, bucket="unit-test-bucket")
    assert "--gcs-prefix" in argv and argv[argv.index("--gcs-prefix") + 1] == "cfs1"
    module.require_owned_session({"state": "running", "session_id": spec["session_id"]},
                                 {"state": "idle", "session_id": spec["session_id"]},
                                 spec["session_id"])
    with pytest.raises(module.CFS1ColabError, match="ownership is uncertain"):
        module.require_owned_session({"state": "running", "session_id": spec["session_id"]},
                                     {"state": "idle", "session_id": "gpu-a100-other"},
                                     spec["session_id"])
    with pytest.raises(module.CFS1ColabError, match="unknown"):
        module.validate_cfs1_job_spec({**spec, "phase": "evaluate"})


def test_state_is_published_only_after_artifacts_and_sync_failure_is_durable(tmp_path: Path) -> None:
    module = _module()
    artifact, state = tmp_path / "completion.json", tmp_path / "ledger.json"
    artifact.write_text("completion")
    state.write_text("ledger")
    calls = []

    def upload(path, remote, digest):
        assert path.read_bytes()
        assert len(digest) == 64
        calls.append(remote)

    durability = module.CFS1StateLastDurability("cfs1", upload)
    remotes = durability.publish_state_last(job_id="cfs1-nextlat-seed1234-episode0-high_different",
                                            artifacts=[artifact], state=state)
    assert remotes[-1] == "cfs1/control/cfs1_run_ledger.json"
    assert calls == remotes

    failures = []
    def fail():
        failures.append(1)
        raise RuntimeError("offline")
    diagnostic = tmp_path / "sync.json"
    with pytest.raises(module.CFS1ColabError, match="circuit breaker"):
        module.sync_with_circuit_breaker(fail, diagnostic=diagnostic, source_sha256="a" * 64)
    assert len(failures) == 3
    assert json.loads(diagnostic.read_text())["training_complete"] is False
