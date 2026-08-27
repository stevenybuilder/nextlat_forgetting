"""Production-contract tests for deterministic base checkpoint evaluation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from lurestar.durable_checkpoint import sha256_file
from run_matrix import DONE, TRAINED, Ledger, competence_identity_from_paths


PROJECT = Path(__file__).resolve().parents[1]
EVALUATOR_PATH = PROJECT / "scripts" / "evaluate_base_competence.py"
LIFECYCLE_PATH = PROJECT / "scripts" / "evaluate_trained_bases.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E = load(EVALUATOR_PATH, "base_competence_evaluator_under_test")
L = load(LIFECYCLE_PATH, "base_competence_lifecycle_under_test")


VERIFIED_LINE = (
    "5,94|21,16|11,43|92,74|20,72|20,75|62,57|75,98|20,37|53,92|22,21|"
    "72,53|98,32|20,27|32,45|20,11|37,22|27,12|43,5|12,62/20,94="
    "20,11,43,5,94"
)


def test_tokenizer_replay_and_geometry_match_upstream_g55() -> None:
    tokens = E.encode_stargraph_line(VERIFIED_LINE)
    prefix, target = E.validate_g55_tokens(tokens)

    assert len(tokens) == 69
    assert len(prefix) == 63
    assert prefix[-1] == 101  # '=' for maxNodes=100
    assert target == [20, 11, 43, 5, 94]
    assert tokens[-1] == 104


@pytest.mark.parametrize(
    "line, match",
    [
        (VERIFIED_LINE.replace("94", "100", 1), "outside"),
        (VERIFIED_LINE.replace("=", "|"), "'='"),
        (VERIFIED_LINE + ",7", "length"),
    ],
)
def test_g55_parser_fails_closed_on_wrong_domain_or_geometry(line: str, match: str) -> None:
    with pytest.raises(E.EvaluationError, match=match):
        E.validate_g55_tokens(E.encode_stargraph_line(line))


def test_backend_neutral_decoder_is_autoregressive_and_exact_path_is_all_or_nothing() -> None:
    calls = []

    def next_tokens(sequences):
        calls.append([list(sequence) for sequence in sequences])
        return [len(sequence) % 7 for sequence in sequences]

    predictions = E.greedy_decode_lists([[1, 2], [3, 4]], steps=5, next_tokens=next_tokens)
    assert predictions == [[2, 3, 4, 5, 6], [2, 3, 4, 5, 6]]
    assert [len(batch[0]) for batch in calls] == [2, 3, 4, 5, 6]
    metric = E.exact_path_counts(predictions, [predictions[0], [2, 3, 4, 5, 0]])
    assert metric == {"correct": 1, "total": 2, "value": 0.5}


def test_manifest_requires_exact_dataset_name_and_digest(tmp_path: Path) -> None:
    dataset = tmp_path / E.EXPECTED_DATASET_NAME
    dataset.write_text("held out\n")
    manifest = tmp_path / "corpus.sha256"
    manifest.write_text(f"{sha256_file(dataset)}  {dataset.name}  9\n")

    records = E.verify_manifest_binding(dataset, [manifest])
    assert records == [{"path": str(manifest.resolve()), "sha256": sha256_file(manifest)}]

    manifest.write_text(f"{sha256_file(dataset)}  not-{dataset.name}  9\n")
    with pytest.raises(E.EvaluationError, match="no supplied manifest"):
        E.verify_manifest_binding(dataset, [manifest])
    manifest.write_text(f"{'0' * 64}  {dataset.name}  9\n")
    with pytest.raises(E.EvaluationError, match="no supplied manifest"):
        E.verify_manifest_binding(dataset, [manifest])


def test_partial_evaluation_progress_is_resumable_only_for_exact_provenance(tmp_path: Path) -> None:
    path = tmp_path / "raw.json.progress.json"
    identity = {
        "job_id": "nextlat-s1234-base",
        "checkpoint_sha256": "a" * 64,
        "dataset_sha256": "b" * 64,
        "decoding": E.DECODING,
    }
    fresh = E.load_progress(path, identity, total=20_000)
    assert fresh["next_index"] == 0
    assert fresh["token_correct"] == [0] * 5

    saved = dict(fresh, next_index=256, correct=240, token_correct=[256, 250, 248, 245, 240])
    path.write_text(json.dumps(saved))
    assert E.load_progress(path, identity, total=20_000) == saved

    with pytest.raises(E.EvaluationError, match="provenance mismatch"):
        E.load_progress(path, dict(identity, checkpoint_sha256="c" * 64), total=20_000)
    path.write_text(json.dumps(dict(saved, next_index=20_001)))
    with pytest.raises(E.EvaluationError, match="invalid counts"):
        E.load_progress(path, identity, total=20_000)


def test_production_decoder_uses_argmax_not_upstream_sampler() -> None:
    source = EVALUATOR_PATH.read_text()
    assert "torch.argmax" in source
    assert ".generate(" not in source
    assert E.DECODING == {"strategy": "greedy", "top_k": 1, "temperature": 0.0}


def make_trained_parent(
    tmp_path: Path, *, evaluator: Path, dataset: Path, manifest: Path,
    model: str = "nextlat", seed: int = 1234,
):
    job_id = f"{model}-s{seed}-base"
    out_root = tmp_path / "runs" / model / f"seed{seed}" / "base"
    checkpoint_dir = out_root / f"{job_id}-seed{seed}"
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "ckpt_iter_20000.pt"
    checkpoint.write_bytes(b"checkpoint")
    materialized = checkpoint_dir / "materialized_config.yaml"
    materialized.write_text(f"seed: {seed}\n")
    metrics = checkpoint_dir / "version_0" / "metrics.csv"
    metrics.parent.mkdir()
    metrics.write_text("step,train_loss\n20000,0.1\n")
    source_config = tmp_path / f"{model}.yaml"
    source_config.write_text(f"seed: {seed}\n")
    ledger = Ledger(tmp_path / "run_ledger.json")
    training_artifacts = {
        str(materialized.relative_to(out_root)): sha256_file(materialized),
        str(metrics.relative_to(out_root)): sha256_file(metrics),
    }
    summary = out_root / "final_summary.json"
    summary.write_text(json.dumps({
        "schema": "nextlat_forgetting/training_completion/1",
        "kind": "training_completion",
        "job_id": job_id,
        "model": model,
        "seed": seed,
        "phase": "base",
        "condition": None,
        "step": 20_000,
        "updates": 20_000,
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)},
        "training_artifacts": training_artifacts,
    }, sort_keys=True) + "\n")
    artifacts = dict(training_artifacts, **{"final_summary.json": sha256_file(summary)})
    parent = ledger.append({
        "job_id": job_id,
        "status": TRAINED,
        "model": model,
        "seed": seed,
        "phase": "base",
        "condition": None,
        "out_root": str(out_root.resolve()),
        "config": str(source_config.resolve()),
        "config_sha256": sha256_file(source_config),
        "manifest_sha256": {},
        "competence_identity": competence_identity_from_paths(
            evaluator, dataset, [manifest]
        ),
        "step": 20_000,
        "updates": 20_000,
        "final_checkpoint": str(checkpoint.resolve()),
        "final_checkpoint_sha256": sha256_file(checkpoint),
        "artifacts": artifacts,
    })
    return ledger, parent, materialized, source_config


def lifecycle_inputs(tmp_path: Path):
    evaluator = tmp_path / "evaluate.py"
    evaluator.write_text("# frozen evaluator\n")
    dataset = tmp_path / E.EXPECTED_DATASET_NAME
    dataset.write_text("dataset fixture\n")
    manifest = tmp_path / "corpus.sha256"
    manifest.write_text(f"{sha256_file(dataset)}  {dataset.name}\n")
    ledger, parent, materialized, source_config = make_trained_parent(
        tmp_path, evaluator=evaluator, dataset=dataset, manifest=manifest
    )
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    return ledger, parent, materialized, source_config, evaluator, dataset, manifest, upstream


def raw_from_command(command: list[str], parent: dict, evaluator: Path, dataset: Path) -> dict:
    output = Path(command[command.index("--output") + 1])
    raw = {
        "schema": E.SCHEMA,
        "job_id": parent["job_id"],
        "model": parent["model"],
        "seed": parent["seed"],
        "checkpoint_sha256": parent["final_checkpoint_sha256"],
        "dataset_sha256": sha256_file(dataset),
        "evaluator_sha256": sha256_file(evaluator),
        "manifest_sha256s": sorted(
            record["sha256"] for record in parent["competence_identity"]["manifests"]
        ),
        "decoding": E.DECODING,
        "exact_path_accuracy": {"correct": 19_000, "total": 20_000, "value": 0.95},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(raw, sort_keys=True) + "\n")
    return raw


def test_lifecycle_runs_frozen_evaluator_then_promotes_to_done(tmp_path: Path) -> None:
    ledger, parent, _, _, evaluator, dataset, manifest, upstream = lifecycle_inputs(tmp_path)
    commands = []

    def runner(command, *, cwd):
        commands.append((command, cwd))
        raw_from_command(command, parent, evaluator, dataset)
        return 0

    outcomes = L.run_lifecycle(
        ledger_path=ledger.path, models=["nextlat"], seeds=[1234], evaluator=evaluator,
        dataset=dataset, manifests=[manifest], upstream=upstream, devices=1,
        precision="bf16-mixed", batch_size=128, command_runner=runner,
    )

    assert outcomes == {parent["job_id"]: "evaluated-and-promoted"}
    assert ledger.state_of(parent["job_id"])["status"] == DONE
    command, cwd = commands[0]
    assert command[:7] == ["fabric", "run", "--devices", "1", "--precision", "bf16-mixed", str(evaluator)]
    assert cwd == upstream
    assert command[command.index("--config") + 1].endswith("materialized_config.yaml")
    assert not list(Path(parent["out_root"]).rglob("*.partial"))


def test_lifecycle_failure_preserves_trained_state(tmp_path: Path) -> None:
    ledger, parent, _, _, evaluator, dataset, manifest, upstream = lifecycle_inputs(tmp_path)
    with pytest.raises(L.LifecycleError, match="ledger remains TRAINED"):
        L.run_lifecycle(
            ledger_path=ledger.path, models=["nextlat"], seeds=[1234], evaluator=evaluator,
            dataset=dataset, manifests=[manifest], upstream=upstream, devices=1,
            precision="bf16-mixed", batch_size=128,
            command_runner=lambda command, cwd: 7,
        )
    assert ledger.state_of(parent["job_id"])["status"] == TRAINED


def test_lifecycle_recovers_atomic_raw_output_without_repeating_inference(tmp_path: Path) -> None:
    ledger, parent, _, _, evaluator, dataset, manifest, upstream = lifecycle_inputs(tmp_path)
    output = Path(parent["out_root"]) / "evaluation" / "exact_path_raw.json"
    fake_command = ["--output", str(output)]
    raw_from_command(fake_command, parent, evaluator, dataset)
    called = False

    def should_not_run(command, *, cwd):
        nonlocal called
        called = True
        return 1

    outcomes = L.run_lifecycle(
        ledger_path=ledger.path, models=["nextlat"], seeds=[1234], evaluator=evaluator,
        dataset=dataset, manifests=[manifest], upstream=upstream, devices=1,
        precision="bf16-mixed", batch_size=128, command_runner=should_not_run,
    )
    assert outcomes == {parent["job_id"]: "recovered-raw-and-promoted"}
    assert called is False


def test_lifecycle_refuses_partial_expected_matrix_before_launch(tmp_path: Path) -> None:
    ledger, _, _, _, evaluator, dataset, manifest, upstream = lifecycle_inputs(tmp_path)
    with pytest.raises(L.LifecycleError, match="jobs are missing"):
        L.run_lifecycle(
            ledger_path=ledger.path, models=["nextlat"], seeds=[1234, 1235],
            evaluator=evaluator, dataset=dataset, manifests=[manifest], upstream=upstream,
            devices=1, precision="bf16-mixed", batch_size=128,
            command_runner=lambda command, cwd: 0,
        )


def test_lifecycle_refuses_posthoc_evaluator_change_before_inference(tmp_path: Path) -> None:
    ledger, parent, _, _, _, dataset, manifest, upstream = lifecycle_inputs(tmp_path)
    different_evaluator = tmp_path / "different_evaluator.py"
    different_evaluator.write_text("# not the evaluator frozen before base training\n")
    called = False

    def runner(command, *, cwd):
        nonlocal called
        called = True
        return 0

    with pytest.raises(L.LifecycleError, match="differs from the identity frozen"):
        L.run_lifecycle(
            ledger_path=ledger.path, models=["nextlat"], seeds=[1234],
            evaluator=different_evaluator, dataset=dataset, manifests=[manifest],
            upstream=upstream, devices=1, precision="bf16-mixed", batch_size=128,
            command_runner=runner,
        )
    assert called is False
    assert ledger.state_of(parent["job_id"])["status"] == TRAINED
