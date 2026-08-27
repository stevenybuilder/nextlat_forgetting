from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "materialize_lurestar_evaluation",
    ROOT / "scripts" / "materialize_lurestar_evaluation.py",
)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: pathlib.Path, content: str | bytes) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode() if isinstance(content, str) else content)
    return path


def _fake_modules(tmp_path: pathlib.Path):
    block = _write(tmp_path / "PERMANENT_H3_BLOCK.json", "{}\n")
    sidecar = _write(tmp_path / "PERMANENT_H3_BLOCK.json.sha256", "block sidecar\n")
    extractor = SimpleNamespace(
        JOB_SCHEMA="test/lurestar_job/9",
        EVIDENCE_SCHEMA="test/lurestar_evidence/9",
        PINNED_UPSTREAM_COMMIT="a" * 40,
        H3_BLOCK_PATH=block,
        H3_BLOCK_SHA256=_sha(block),
        H3_BLOCK_SIDECAR_SHA256=_sha(sidecar),
        EXTRACTION_POLICY={"whitener_count": 400, "scored_count": 1600},
    )
    evaluator = SimpleNamespace(
        SCHEMA="test/lurestar_manifest/9",
        REPORT_SCHEMA="test/lurestar_report/9",
        RECEIPT_SCHEMA="test/lurestar_receipt/9",
        E=SimpleNamespace(ARMS=("nextlat", "bst", "gpt")),
        REQUIRED_ARRAYS={"h1_item_ids"},
        BOUND_SCALARS={
            "evidence_schema", "arm", "seed", "base_checkpoint_sha256",
            "h3_permanent_block_sha256", "h3_permanent_block_sidecar_sha256",
            "h1_item_ids_sha256",
            "local_representations_sha256", "local_evaluate_sha256",
        },
        BST_SECONDARY_ARRAYS=set(),
        BST_WHITENER_FIELDS=set(),
    )
    extractor_path = _write(tmp_path / "extractor.py", "# extractor\n")
    evaluator_path = _write(tmp_path / "evaluator.py", "# evaluator\n")
    return extractor, evaluator, extractor_path, evaluator_path


def _project(tmp_path: pathlib.Path):
    upstream = tmp_path / "upstream"
    for relative in M.UPSTREAM_SOURCE_PATHS:
        _write(upstream / relative, relative + "\n")

    stimulus = tmp_path / "e_lure.jsonl"
    rows = []
    for index in range(2000):
        identity = f"{index:064x}"
        rows.append(json.dumps({
            "conditions": {"base": {"graph_key": identity}},
        }, sort_keys=True))
    _write(stimulus, "\n".join(rows) + "\n")

    entries = []
    for arm in M.MODELS:
        for seed in M.SEEDS:
            job_id = f"{arm}-s{seed}-base"
            out_root = tmp_path / "runs" / arm / str(seed) / "base"
            run_root = out_root / f"{job_id}-seed{seed}"
            checkpoint = _write(run_root / "final.pt", f"{job_id}:checkpoint\n")
            materialized = _write(
                run_root / "materialized_config.yaml", f"seed: {seed}\n"
            )
            source_config = _write(
                tmp_path / "configs" / f"{arm}-{seed}.yaml", f"seed: {seed}\n"
            )
            relative_materialized = str(materialized.relative_to(out_root))
            entries.append({
                "seq": len(entries),
                "job_id": job_id,
                "status": M.DONE,
                "model": arm,
                "seed": seed,
                "phase": "base",
                "out_root": str(out_root),
                "final_checkpoint": str(checkpoint),
                "final_checkpoint_sha256": _sha(checkpoint),
                "config": str(source_config),
                "config_sha256": _sha(source_config),
                "artifacts": {relative_materialized: _sha(materialized)},
            })
    ledger = tmp_path / "ledger.json"
    _write(ledger, json.dumps({"schema": 1, "entries": entries}, sort_keys=True) + "\n")
    extractor, evaluator, extractor_path, evaluator_path = _fake_modules(tmp_path)
    return {
        "ledger": ledger,
        "upstream": upstream,
        "stimulus": stimulus,
        "stimulus_sha": _sha(stimulus),
        "evaluation_root": tmp_path / "evaluation",
        "extractor": extractor,
        "evaluator": evaluator,
        "extractor_path": extractor_path,
        "evaluator_path": evaluator_path,
    }


def _kwargs(project: dict, **updates):
    values = {
        "ledger_path": project["ledger"],
        "upstream": project["upstream"],
        "e_lure_path": project["stimulus"],
        "e_lure_sha256": project["stimulus_sha"],
        "evaluation_root": project["evaluation_root"],
        "extractor_path": project["extractor_path"],
        "evaluator_path": project["evaluator_path"],
        "extractor_module": project["extractor"],
        "evaluator_module": project["evaluator"],
    }
    values.update(updates)
    return values


def test_dry_run_preflights_exact_15_and_writes_nothing(tmp_path):
    project = _project(tmp_path)
    validated = []

    def validator(parent, **identity):
        validated.append((parent["job_id"], identity))
        return {"valid": True}

    result = M.execute(**_kwargs(project), parent_validator=validator, dry_run=True)
    assert result["status"] == "DRY_RUN"
    assert result["plan"]["cell_count"] == 15
    assert len(validated) == 15
    assert len({cell["output_path"] for cell in result["plan"]["cells"]}) == 15
    assert len({cell["progress_root"] for cell in result["plan"]["cells"]}) == 15
    assert set(result["plan"]["local_measurement_sources"]) == set(
        M.LOCAL_MEASUREMENT_SOURCE_PATHS
    )
    assert not project["evaluation_root"].exists()


def test_invalid_fifteenth_parent_permits_zero_subprocess_invocations(tmp_path):
    project = _project(tmp_path)
    invocations = []
    checked = []

    def validator(parent, **_identity):
        checked.append(parent["job_id"])
        if parent["job_id"] == "bst-s1238-base":
            raise RuntimeError("invalid final competence receipt")

    def runner(command, *, cwd):
        invocations.append((command, cwd))
        return 0

    with pytest.raises(M.MaterializationRefused, match="invalid final competence receipt"):
        M.execute(
            **_kwargs(project), parent_validator=validator, command_runner=runner,
        )
    assert checked[-1] == "bst-s1238-base"
    assert len(checked) == 15
    assert invocations == []
    assert not project["evaluation_root"].exists()


def test_stale_fifteenth_evidence_permits_zero_subprocess_invocations(tmp_path):
    project = _project(tmp_path)
    invocations = []
    stale = (
        project["evaluation_root"] / "cells" / "bst-s1238-base" / "evidence.npz"
    )
    _write(stale, b"partial stale evidence")

    with pytest.raises(M.MaterializationRefused, match="evidence/receipt pair is incomplete"):
        M.execute(
            **_kwargs(project), parent_validator=lambda *args, **kwargs: None,
            command_runner=lambda command, *, cwd: invocations.append(command) or 0,
        )
    assert invocations == []


def test_missing_or_extra_base_cell_is_refused_during_preflight(tmp_path):
    project = _project(tmp_path)
    document = json.loads(project["ledger"].read_text())
    document["entries"].pop()
    document["entries"].append({
        "seq": 14, "job_id": "gpt-s9999-base", "status": M.DONE,
        "model": "gpt", "seed": 9999, "phase": "base",
    })
    project["ledger"].write_text(json.dumps(document))
    with pytest.raises(M.MaterializationRefused, match="missing=.*bst-s1238-base.*extra"):
        M.build_plan(
            **_kwargs(project), parent_validator=lambda *args, **kwargs: None,
        )


def test_e_lure_requires_explicit_hash_and_exact_400_1600_identity_split(tmp_path):
    project = _project(tmp_path)
    with pytest.raises(M.MaterializationRefused, match="explicit lowercase E_lure SHA"):
        M.build_plan(
            **_kwargs(project, e_lure_sha256="not-a-sha"),
            parent_validator=lambda *args, **kwargs: None,
        )
    plan, cells, _extractor, _evaluator = M.build_plan(
        **_kwargs(project), parent_validator=lambda *args, **kwargs: None,
    )
    assert len(cells) == 15
    assert plan["e_lure"]["calibration_count"] == 400
    assert plan["e_lure"]["scored_count"] == 1600
    expected_scored = M._canonical_identity_sha([f"{i:064x}" for i in range(400, 2000)])
    assert plan["e_lure"]["scored_ids_sha256"] == expected_scored
    expected_local_hashes = {
        relative: plan["local_measurement_sources"][relative]["sha256"]
        for relative in M.LOCAL_MEASUREMENT_SOURCE_PATHS
    }
    assert all(
        cell.job_payload["local_measurement_source_sha256"] == expected_local_hashes
        for cell in cells
    )


def test_existing_evaluation_is_reused_only_with_exact_receipt_and_sidecar(tmp_path):
    manifest = _write(tmp_path / "manifest.json", "{}\n")
    report = _write(tmp_path / "report.json", "opaque model outcomes\n")
    receipt_path = report.with_suffix(".json.receipt.json")
    receipt = {
        "schema": "test/receipt/1",
        "manifest": {"path": str(manifest.resolve()), "sha256": _sha(manifest)},
        "report": {"path": str(report.resolve()), "sha256": _sha(report)},
    }
    _write(receipt_path, json.dumps(receipt, sort_keys=True) + "\n")
    sidecar = receipt_path.with_suffix(".json.sha256")
    _write(sidecar, f"{_sha(receipt_path)}  {receipt_path.name}\n")
    evaluator = SimpleNamespace(RECEIPT_SCHEMA="test/receipt/1")
    assert M._existing_evaluation(
        manifest_path=manifest, report_path=report, evaluator=evaluator,
    ) == (receipt_path, sidecar)
    sidecar.write_text("0" * 64 + "  stale\n")
    with pytest.raises(M.MaterializationRefused, match="sidecar is stale"):
        M._existing_evaluation(
            manifest_path=manifest, report_path=report, evaluator=evaluator,
        )


def test_local_measurement_source_rehash_refuses_mutation(tmp_path, monkeypatch):
    local_repo = tmp_path / "local_repo"
    records = {}
    for relative in M.LOCAL_MEASUREMENT_SOURCE_PATHS:
        path = _write(local_repo / relative, f"frozen:{relative}\n")
        records[relative] = {"path": str(path.resolve()), "sha256": _sha(path)}
    monkeypatch.setattr(M, "_REPO", local_repo)
    plan = {"local_measurement_sources": records}
    M._verify_local_measurement_sources(plan)
    target = local_repo / M.LOCAL_MEASUREMENT_SOURCE_PATHS[0]
    target.write_text("mutated after plan\n")
    with pytest.raises(M.MaterializationRefused, match="local measurement source changed"):
        M._verify_local_measurement_sources(plan)
