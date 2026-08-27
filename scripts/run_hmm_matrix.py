#!/usr/bin/env python
"""Durable confirmatory runner for legacy single-HMM or 30-job HMM-family calibration.

The HMM experiment is a separate confirmatory matrix from Lure-Star: GPT and NextLat at the
five preregistered seeds, exactly 3,000 optimizer updates apiece.  This module deliberately
reuses :mod:`run_matrix`'s append-only ledger, verified two-generation checkpoints, exact-update
gate, TRAINED/DONE lifecycle, and artifact hashing instead of introducing a second recovery
protocol.

Training and scientific evaluation are separate phases.  ``--phase train`` can only produce
``TRAINED``.  ``--phase evaluate`` never launches a trainer; it promotes an already hash-verified
TRAINED job to DONE only when ``evaluation/hmm_geometry.json`` and its sidecar bind the exact
checkpoint, evaluator, frozen manifests, and every preregistered HMM metric family.

Examples::

    python scripts/run_hmm_matrix.py --root /content/lurestar --print-plan
    python scripts/run_hmm_matrix.py --root /content/lurestar --phase train \
      --project-root /content/project --upstream /content/project/upstream/NextLat \
      --driver-managed-durability
    python scripts/run_hmm_matrix.py --root /content/lurestar --phase evaluate \
      --driver-managed-durability
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys
import typing as t

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from lurestar.durable_checkpoint import sha256_file  # noqa: E402
from hmm_geometry.family import REGIMES as HMM_FAMILY_REGIMES, load_family  # noqa: E402
from hmm_geometry.generate import load_frozen_hmm  # noqa: E402
from run_matrix import (  # noqa: E402
    COMPLETION_SUMMARY,
    DONE,
    TRAINED,
    TRAINING_TERMINAL,
    JobSpec,
    LaunchResult,
    Ledger,
    MatrixRunner,
    ResumePlan,
    SEEDS,
    validate_matrix,
    verify_artifacts,
)

HMM_MODELS = ("gpt", "nextlat")
HMM_TRAIN_UPDATES = 3_000
HMM_CHECKPOINT_INTERVAL = 250
PINNED_UPSTREAM_COMMIT = "3770be6009cea2b3c455a9ce7f2ca88b504bb955"
HMM_EVALUATION_RECEIPT = "evaluation/hmm_geometry.json"
HMM_EVALUATION_SIDECAR = HMM_EVALUATION_RECEIPT + ".sha256"
HMM_EVALUATION_SCHEMA = "nextlat_forgetting/hmm_geometry/1"
RUNTIME_RECOVERY_BARRIER_SCHEMA = "nextlat_forgetting/runtime_recovery_barrier/1"

# The inventory is the frozen source of truth for the large arrays.  Every entry below is
# re-hashed before planning; the inventory and the small scientific manifests are then included
# in every JobSpec identity so a changed snapshot cannot be resumed under an old ledger entry.
HMM_REQUIRED_INVENTORY = frozenset({
    "data/hmm/hmm4x4_train_len32_100000.npy",
    "data/hmm/hmm4x4_val_len32_10000.npy",
    "data/hmm/hmm4x4_lengen_len64_10000.npy",
    "data/hmm/hmm4x4_val_posteriors.npz",
    "data/hmm/hmm4x4_lengen_posteriors.npz",
    "manifests/hmm_dataset.json",
    "manifests/hmm_eval_pairs.json",
    "manifests/hmm_eval_pairs.jsonl",
    "manifests/hmm_matrices.json",
    "manifests/hmm_thresholds.json",
})

# DONE means all preregistered families were reported, not that a favorable subset was selected.
HMM_REQUIRED_METRICS = frozenset({
    "h1_predictive_equivalence_centered_cosine",
    "h1_predictive_equivalence_whitened",
    "h2_spearman",
    "h2_partial_spearman",
    "h2_neighborhood_retrieval",
    "h3_posterior_decoding_len32",
    "h3_future_distribution_decoding_len32",
    "h3_posterior_decoding_len64",
    "h3_future_distribution_decoding_len64",
})
HMM_FAMILY_REQUIRED_METRICS = HMM_REQUIRED_METRICS | frozenset({
    "h2_partial_spearman_whitened",
    "h2_belief_partial_spearman",
    "h2_belief_partial_spearman_whitened",
})
HMM_RECEIPT_MANIFEST_NAMES = frozenset({
    "manifest_inventory.sha256",
    "hmm_dataset.json",
    "hmm_eval_pairs.json",
    "hmm_eval_pairs.jsonl",
    "hmm_matrices.json",
    "hmm_thresholds.json",
})


class HMMMatrixError(RuntimeError):
    """A pre-compute identity, lifecycle, or evaluation gate failed closed."""


def hmm_job_id(model: str, seed: int, regime: str | None = None) -> str:
    """Exact stable id shared by the ledger, experiment directory, and GCS key."""
    if model not in HMM_MODELS:
        raise ValueError(f"unknown HMM model {model!r}")
    if seed not in SEEDS:
        raise ValueError(f"seed {seed} is not preregistered: {SEEDS}")
    if regime is not None and regime not in HMM_FAMILY_REGIMES:
        raise ValueError(f"unknown HMM family regime {regime!r}")
    suffix = f"-{regime}" if regime is not None else ""
    return f"{model}-seed{seed}-hmm{suffix}"


def _family_required_paths(regime: str) -> frozenset[str]:
    prefix = f"manifests/hmm_family/{regime}"
    data = f"data/hmm_family/{regime}"
    return frozenset({
        f"{data}/hmm4x4_train_len32_100000.npy",
        f"{data}/hmm4x4_val_len32_10000.npy",
        f"{data}/hmm4x4_lengen_len64_10000.npy",
        f"{data}/hmm4x4_val_posteriors.npz",
        f"{data}/hmm4x4_lengen_posteriors.npz",
        f"{prefix}/hmm_dataset.json", f"{prefix}/hmm_eval_pairs.json",
        f"{prefix}/hmm_eval_pairs.jsonl", f"{prefix}/hmm_matrices.json",
        f"{prefix}/hmm_thresholds.json",
    })


def verify_hmm_family_snapshot(
    project_root: os.PathLike | str,
) -> dict[str, tuple[str, ...]]:
    """Verify all regimes together; a favorable regime subset cannot reach planning."""
    root = pathlib.Path(project_root).resolve()
    inventory = root / "manifests/hmm_family_inventory.sha256"
    family = root / "manifests/hmm_family.json"
    receipt = root / "manifests/hmm_family_materialization.json"
    if not all(path.is_file() for path in (inventory, family, receipt)):
        raise HMMMatrixError("complete HMM family manifest/inventory/receipt is required")
    try:
        _, family_document = load_family(family)
        receipt_document = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        raise HMMMatrixError("HMM family manifest/materialization receipt is invalid") from exc
    if (
        receipt_document.get("schema") != "nextlat_forgetting/hmm_family_materialization/1"
        or receipt_document.get("status") != "complete"
        or tuple(receipt_document.get("required_regimes", ())) != HMM_FAMILY_REGIMES
        or receipt_document.get("family_sha256") != family_document.get("payload_sha256")
        or receipt_document.get("model_outcomes_inspected") is not False
    ):
        raise HMMMatrixError("HMM family materialization receipt is incomplete or unblinded")
    rows = _parse_inventory(inventory)
    if receipt_document.get("inventory_sha256") != sha256_file(inventory):
        raise HMMMatrixError("HMM family materialization receipt does not bind its inventory")
    required_all = frozenset().union(*(_family_required_paths(name) for name in HMM_FAMILY_REGIMES))
    required_all = required_all | {"manifests/hmm_family.json"}
    missing = sorted(required_all - set(rows))
    if missing:
        raise HMMMatrixError(f"HMM family inventory is incomplete: {missing}")
    common = (str(inventory), str(family), str(receipt))
    result: dict[str, tuple[str, ...]] = {}
    for regime in HMM_FAMILY_REGIMES:
        verified = []
        for rel in sorted(_family_required_paths(regime) | {"manifests/hmm_family.json"}):
            path = root / rel
            if not path.is_file() or sha256_file(path) != rows.get(rel):
                raise HMMMatrixError(f"HMM family snapshot hash mismatch: {rel}")
            verified.append(str(path))
        prefix = root / "manifests/hmm_family" / regime
        try:
            hmm, _ = load_frozen_hmm(prefix / "hmm_matrices.json")
            dataset = json.loads((prefix / "hmm_dataset.json").read_text(encoding="utf-8"))
            thresholds = json.loads((prefix / "hmm_thresholds.json").read_text(encoding="utf-8"))
            pairs = json.loads((prefix / "hmm_eval_pairs.json").read_text(encoding="utf-8"))
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            raise HMMMatrixError(f"{regime}: invalid family scientific manifest") from exc
        expected_hmm = family_document["regimes"][regime]["hmm_sha256"]
        threshold_payload = thresholds.get("thresholds")
        if not isinstance(threshold_payload, dict):
            raise HMMMatrixError(f"{regime}: threshold artifact lacks its scientific payload")
        if (hmm.sha256() != expected_hmm or
                dataset.get("hmm_sha256") != expected_hmm or
                threshold_payload.get("hmm_sha256") != expected_hmm or
                pairs.get("hmm_sha256") != expected_hmm):
            raise HMMMatrixError(f"{regime}: family artifacts bind inconsistent matrices")
        pair_file = prefix / "hmm_eval_pairs.jsonl"
        if pairs.get("pairs_sha256") != sha256_file(pair_file):
            raise HMMMatrixError(f"{regime}: pair bank does not match its manifest")
        for split in dataset.get("splits", {}).values():
            for key in ("observations_file", "posteriors_file"):
                if key not in split:
                    continue
                artifact = root / split[key]
                if not artifact.is_file() or split.get(key.replace("file", "sha256")) != sha256_file(artifact):
                    raise HMMMatrixError(f"{regime}: dataset split artifact mismatch")
        result[regime] = tuple(dict.fromkeys((*common, *verified)))
    return result


def _parse_inventory(path: pathlib.Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split(maxsplit=1)
        if len(fields) != 2:
            raise HMMMatrixError(f"malformed inventory row {path}:{line_no}")
        digest, rel = fields[0].lower(), fields[1].strip()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise HMMMatrixError(f"invalid SHA-256 at {path}:{line_no}")
        if rel in rows:
            raise HMMMatrixError(f"duplicate inventory path {rel!r}")
        rows[rel] = digest
    return rows


def verify_hmm_snapshot(project_root: os.PathLike | str = _REPO) -> tuple[str, ...]:
    """Re-hash the complete frozen HMM corpus/evaluation snapshot before any GPU launch."""
    root = pathlib.Path(project_root).resolve()
    inventory = root / "manifests" / "manifest_inventory.sha256"
    if not inventory.is_file():
        raise HMMMatrixError(f"missing HMM inventory: {inventory}")
    rows = _parse_inventory(inventory)
    missing_rows = sorted(HMM_REQUIRED_INVENTORY - rows.keys())
    if missing_rows:
        raise HMMMatrixError(f"HMM inventory lacks required entries: {missing_rows}")

    verified: list[pathlib.Path] = [inventory]
    for rel in sorted(HMM_REQUIRED_INVENTORY):
        candidate = (root / rel).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise HMMMatrixError(f"inventory path escapes project root: {rel}") from exc
        if not candidate.is_file():
            raise HMMMatrixError(f"HMM snapshot file is missing: {candidate}")
        actual = sha256_file(candidate)
        if actual != rows[rel]:
            raise HMMMatrixError(
                f"HMM snapshot hash mismatch for {rel}: {actual} != {rows[rel]}"
            )
        verified.append(candidate)
    return tuple(str(path) for path in verified)


def hmm_source_inputs(
    project_root: os.PathLike | str = _REPO,
    upstream_root: os.PathLike | str = _REPO / "upstream" / "NextLat",
) -> tuple[str, ...]:
    """Return the local and pinned-upstream sources that own HMM training semantics."""
    project = pathlib.Path(project_root).resolve()
    upstream = pathlib.Path(upstream_root).resolve()
    paths = (
        project / "scripts" / "run_hmm_matrix.py",
        project / "scripts" / "run_matrix.py",
        project / "scripts" / "train_hmm.py",
        project / "scripts" / "runtime_bootstrap.py",
        project / "scripts" / "evaluate_hmm_checkpoints.py",
        project / "scripts" / "aggregate_hmm_family.py",
        project / "scripts" / "materialize_hmm_family.py",
        project / "src" / "hmm_geometry" / "datamodule.py",
        project / "src" / "hmm_geometry" / "forward.py",
        project / "src" / "hmm_geometry" / "generate.py",
        project / "src" / "hmm_geometry" / "pair_bank.py",
        project / "src" / "hmm_geometry" / "evaluate.py",
        project / "src" / "hmm_geometry" / "aggregate.py",
        project / "src" / "hmm_geometry" / "family.py",
        project / "src" / "hmm_geometry" / "extraction_cache.py",
        project / "src" / "lurestar" / "durable_checkpoint.py",
        upstream / "train.py",
        upstream / "defaults.yaml",
        upstream / "core_train.py",
        upstream / "models" / "model_base.py",
        upstream / "models" / "model_gpt.py",
        upstream / "models" / "model_nextlat.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise HMMMatrixError(f"HMM source snapshot is incomplete: {missing}")
    return tuple(str(path) for path in paths)


def build_hmm_matrix(
    root: os.PathLike | str,
    *,
    models: t.Sequence[str] = HMM_MODELS,
    seeds: t.Sequence[int] = SEEDS,
    project_root: os.PathLike | str = _REPO,
    upstream_root: os.PathLike | str = _REPO / "upstream" / "NextLat",
    snapshot_root: os.PathLike | str | None = None,
    identity_inputs: t.Sequence[str] | None = None,
    regimes: t.Sequence[str] | None = None,
) -> list[JobSpec]:
    """Build exactly GPT/NextLat x five seeds, with isolated output and resume roots."""
    root = pathlib.Path(root).resolve()
    project = pathlib.Path(project_root).resolve()
    snapshot = pathlib.Path(snapshot_root if snapshot_root is not None else root).resolve()
    unknown = sorted(set(models) - set(HMM_MODELS))
    bad_seeds = sorted(set(seeds) - set(SEEDS))
    if unknown or bad_seeds:
        raise ValueError(f"unknown models={unknown} or non-preregistered seeds={bad_seeds}")
    family_mode = regimes is not None
    selected_regimes = tuple(regimes or ())
    if family_mode and selected_regimes != HMM_FAMILY_REGIMES:
        raise ValueError(
            "confirmatory family planning requires every frozen regime in its frozen order"
        )
    family_inputs = verify_hmm_family_snapshot(snapshot) if family_mode else None
    if identity_inputs is None and not family_mode:
        identity_inputs = (
            *verify_hmm_snapshot(snapshot),
            *hmm_source_inputs(project, upstream_root),
        )

    jobs: list[JobSpec] = []
    job_regimes: tuple[str | None, ...] = t.cast(
        tuple[str | None, ...], selected_regimes if family_mode else (None,)
    )
    sources = hmm_source_inputs(project, upstream_root) if family_mode else ()
    for regime in job_regimes:
        for model in models:
            config = project / "configs" / f"{model}_hmm.yaml"
            if not config.is_file():
                raise FileNotFoundError(f"missing HMM config: {config}")
            for seed in seeds:
                jid = hmm_job_id(model, int(seed), regime)
                manifests = tuple(identity_inputs or ())
                if family_mode:
                    assert regime is not None and family_inputs is not None
                    manifests = (*family_inputs[regime], *sources)
                output = (
                    root / "runs" / "hmm_family" / t.cast(str, regime) / model
                    / f"seed{seed}" / "base"
                    if family_mode else root / "runs" / "hmm" / model / f"seed{seed}" / "base"
                )
                jobs.append(JobSpec(
                    job_id=jid,
                    model=model,
                    seed=int(seed),
                    phase="hmm",
                    condition=regime,
                    config=str(config),
                    out_root=str(output),
                    manifests=manifests,
                    train_batches=HMM_TRAIN_UPDATES,
                ))
    validate_matrix(jobs)
    return jobs


def verify_upstream_commit(
    upstream_root: os.PathLike | str,
    expected: str = PINNED_UPSTREAM_COMMIT,
) -> str:
    """Fail before launch if the runtime checkout is not the preregistered upstream commit."""
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(pathlib.Path(upstream_root).resolve()),
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    actual = proc.stdout.strip().lower()
    if proc.returncode != 0 or actual != expected:
        raise HMMMatrixError(
            f"upstream commit mismatch: {actual or proc.stderr.strip()!r} != {expected}"
        )
    return actual


class HMMFabricLauncher:
    """Single-GPU HMM shim launcher with relayed output and bounded failure diagnostics."""

    def __init__(
        self,
        project_root: os.PathLike | str,
        upstream_root: os.PathLike | str,
        *,
        data_root: os.PathLike | str | None = None,
        devices: int = 1,
        precision: str = "bf16-mixed",
        strategy: str = "ddp",
        expected_upstream_commit: str = PINNED_UPSTREAM_COMMIT,
        dry_run: bool = False,
        echo: t.Callable[[str], None] = print,
    ) -> None:
        self.project_root = pathlib.Path(project_root).resolve()
        self.upstream_root = pathlib.Path(upstream_root).resolve()
        self.data_root = pathlib.Path(
            data_root if data_root is not None else project_root
        ).resolve()
        self.devices = devices
        self.precision = precision
        self.strategy = strategy
        self.expected_upstream_commit = expected_upstream_commit
        self.dry_run = dry_run
        self.echo = echo

    def command(self, plan: ResumePlan) -> list[str]:
        spec = plan.spec
        if spec.phase != "hmm" or spec.model not in HMM_MODELS:
            raise ValueError(f"not an HMM matrix job: {spec.job_id}")
        if spec.train_batches != HMM_TRAIN_UPDATES:
            raise ValueError(
                f"{spec.job_id}: confirmatory HMM jobs require exactly "
                f"{HMM_TRAIN_UPDATES} updates"
            )
        if not plan.fresh and not plan.checkpoint_path:
            raise ValueError(f"{spec.job_id}: non-fresh HMM plan lacks a verified checkpoint")
        regime = spec.condition
        data = self.data_root / "data" / (
            pathlib.Path("hmm_family") / regime if regime is not None else pathlib.Path("hmm")
        )
        matrices = self.data_root / "manifests" / (
            pathlib.Path("hmm_family") / regime / "hmm_matrices.json"
            if regime is not None else pathlib.Path("hmm_matrices.json")
        )
        cmd = [
            "fabric", "run", "--devices", str(self.devices), "--strategy", self.strategy,
            "--precision", self.precision,
            str(self.project_root / "scripts" / "train_hmm.py"),
            "--config", spec.config,
            "--upstream-root", str(self.upstream_root),
        ]
        checkpoint = plan.checkpoint_path
        if checkpoint:
            cmd.extend(["--checkpoint_path", checkpoint])
        cmd.extend([
            f"seed={spec.seed}",
            f"trainer.out_dir={pathlib.Path(spec.out_root).resolve()}",
            f"trainer.experiment_name={spec.experiment_name}",
            f"trainer.init_from={'resume' if checkpoint else 'scratch'}",
            f"trainer.train_batches={HMM_TRAIN_UPDATES}",
            "trainer.compile=false",
            "trainer.log_to_wandb=false",
            f"trainer.save_recovery_checkpoint={HMM_CHECKPOINT_INTERVAL}",
            f"data.hmm_matrices_path={matrices}",
            f"data.hmm_train_data_path={data / 'hmm4x4_train_len32_100000.npy'}",
            f"data.hmm_val_data_path={data / 'hmm4x4_val_len32_10000.npy'}",
            "data.hmm_generalization_data_path="
            f"[{data / 'hmm4x4_lengen_len64_10000.npy'}]",
        ])
        return cmd

    def __call__(self, plan: ResumePlan) -> LaunchResult:
        verify_upstream_commit(self.upstream_root, self.expected_upstream_commit)
        cmd = self.command(plan)
        self.echo(f"[run_hmm_matrix] {plan.spec.job_id}: " + " ".join(cmd))
        if self.dry_run:
            return LaunchResult(0, None, "dry-run")

        proc = subprocess.Popen(
            cmd, cwd=str(self.upstream_root), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            env=dict(os.environ, PYTHONUNBUFFERED="1"),
        )
        assert proc.stdout is not None
        tail: collections.deque[str] = collections.deque(maxlen=40)
        for line in proc.stdout:
            line = line.rstrip("\n")
            tail.append(line)
            self.echo(line)
        returncode = proc.wait()
        detail = f"returncode={returncode}"
        if returncode:
            detail += "; last output:\n" + "\n".join(tail)
        return LaunchResult(returncode, None, detail)


def _valid_sha(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def _verified_reference(record: object, *, label: str) -> tuple[pathlib.Path, str]:
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise HMMMatrixError(f"HMM evaluation receipt lacks {label} path/SHA")
    digest = record.get("sha256")
    if not _valid_sha(digest):
        raise HMMMatrixError(f"HMM evaluation receipt lacks {label} path/SHA")
    path = pathlib.Path(record["path"])
    if not path.is_absolute():
        path = (_REPO / path).resolve()
    if not path.is_file() or sha256_file(path) != digest:
        raise HMMMatrixError(f"HMM evaluation {label} is missing or hash-mismatched")
    return path, t.cast(str, digest)


def _reject_nonfinite(value: object, *, path: str = "receipt") -> None:
    """Reject NaN/inf at any nesting depth, not only at a metric's first level."""
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_nonfinite(item, path=f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise HMMMatrixError(f"HMM evaluation contains a non-finite value at {path}")


def _verify_representation_cache(record: object) -> dict[str, object]:
    """Re-hash every state-last cache chunk recorded by the extraction progress manifest."""
    if not isinstance(record, dict) or record.get("schema") != (
        "nextlat_forgetting/hmm_representation_cache/1"
    ):
        raise HMMMatrixError("HMM evaluation lacks a valid representation cache attestation")
    progress_path, _ = _verified_reference(
        record.get("progress"), label="representation cache progress"
    )
    try:
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HMMMatrixError("HMM representation cache progress is invalid JSON") from exc
    if progress.get("identity_sha256") != record.get("identity_sha256"):
        raise HMMMatrixError("HMM representation cache identity mismatch")
    identity = progress.get("identity")
    if not isinstance(identity, dict):
        raise HMMMatrixError("HMM representation cache has no scientific identity")
    try:
        canonical = hashlib.sha256(
            json.dumps(
                identity, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        ).hexdigest()
    except (TypeError, ValueError) as exc:
        raise HMMMatrixError("HMM representation cache identity is not canonical JSON") from exc
    if canonical != record.get("identity_sha256"):
        raise HMMMatrixError("HMM representation cache identity digest is invalid")
    chunks = progress.get("chunks")
    receipt_hashes = record.get("chunk_sha256")
    if not isinstance(chunks, dict) or not chunks or not isinstance(receipt_hashes, dict):
        raise HMMMatrixError("HMM representation cache contains no committed chunks")
    if set(chunks) != set(receipt_hashes) or int(record.get("n_chunks", -1)) != len(chunks):
        raise HMMMatrixError("HMM representation cache chunk inventory mismatch")
    for key, chunk in chunks.items():
        _, digest = _verified_reference(chunk, label=f"representation chunk {key}")
        if receipt_hashes.get(key) != digest:
            raise HMMMatrixError(f"HMM representation chunk {key} receipt hash mismatch")
        sidecar_path, _ = _verified_reference(
            {"path": chunk.get("sidecar"), "sha256": chunk.get("sidecar_sha256")},
            label=f"representation chunk {key} sidecar",
        )
        fields = sidecar_path.read_text(encoding="utf-8").strip().split()
        if not fields or fields[0].lower() != digest:
            raise HMMMatrixError(f"HMM representation chunk {key} sidecar mismatch")
    return identity


def verify_hmm_evaluation_receipt(spec: JobSpec, state: t.Mapping[str, object]) -> dict:
    """Verify the all-metrics receipt that alone permits TRAINED -> DONE promotion."""
    root = pathlib.Path(spec.out_root)
    receipt_path = root / HMM_EVALUATION_RECEIPT
    sidecar_path = root / HMM_EVALUATION_SIDECAR
    if not receipt_path.is_file() or not sidecar_path.is_file():
        raise HMMMatrixError(f"{spec.job_id}: evaluation receipt or sidecar is missing")
    receipt_sha = sha256_file(receipt_path)
    sidecar_fields = sidecar_path.read_text(encoding="utf-8").strip().split()
    if not sidecar_fields or sidecar_fields[0].lower() != receipt_sha:
        raise HMMMatrixError(f"{spec.job_id}: evaluation receipt sidecar does not match")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HMMMatrixError(f"{spec.job_id}: invalid HMM evaluation receipt JSON") from exc

    expected = {
        "schema": HMM_EVALUATION_SCHEMA,
        "job_id": spec.job_id,
        "model": spec.model,
        "seed": spec.seed,
        "checkpoint_sha256": state.get("final_checkpoint_sha256"),
        "all_preregistered_metrics_reported": True,
        "metric_selection_performed": False,
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise HMMMatrixError(f"{spec.job_id}: evaluation receipt identity/attestation mismatch")
    expected_regime = spec.condition or "primary"
    # Legacy single-HMM receipts predate the explicit regime field. Family jobs never do.
    if spec.condition is not None and receipt.get("regime") != expected_regime:
        raise HMMMatrixError(f"{spec.job_id}: evaluation receipt family regime mismatch")
    if spec.condition is None and receipt.get("regime", "primary") != "primary":
        raise HMMMatrixError(f"{spec.job_id}: legacy evaluation is not the primary regime")
    checkpoint = pathlib.Path(str(state.get("final_checkpoint", "")))
    checkpoint_sha = state.get("final_checkpoint_sha256")
    if not checkpoint.is_file() or not _valid_sha(checkpoint_sha):
        raise HMMMatrixError(f"{spec.job_id}: TRAINED checkpoint is missing")
    if sha256_file(checkpoint) != checkpoint_sha:
        raise HMMMatrixError(f"{spec.job_id}: TRAINED checkpoint hash changed")

    identity_hashes = state.get("manifest_sha256")
    if not isinstance(identity_hashes, dict):
        raise HMMMatrixError(f"{spec.job_id}: TRAINED state lacks its source identity hashes")
    evaluator_path, evaluator_sha = _verified_reference(
        receipt.get("evaluator"), label="evaluator"
    )
    if identity_hashes.get(str(evaluator_path)) != evaluator_sha:
        raise HMMMatrixError(
            f"{spec.job_id}: evaluator was not frozen in the training source identity"
        )
    evaluator_sources = receipt.get("evaluator_sources")
    if not isinstance(evaluator_sources, list) or len(evaluator_sources) < 2:
        raise HMMMatrixError(f"{spec.job_id}: evaluation omits its extraction/cache sources")
    for index, source in enumerate(evaluator_sources):
        source_path, source_sha = _verified_reference(source, label=f"evaluator_source[{index}]")
        if identity_hashes.get(str(source_path)) != source_sha:
            raise HMMMatrixError(
                f"{spec.job_id}: evaluator source was not frozen during training: {source_path}"
            )
    inputs = receipt.get("inputs")
    required_inputs = {
        "source_config", "materialized_config", "pair_bank", "val_posteriors",
        "lengen_posteriors",
    }
    if not isinstance(inputs, dict) or set(inputs) != required_inputs:
        raise HMMMatrixError(f"{spec.job_id}: evaluation input identity is incomplete")
    verified_inputs = {
        name: _verified_reference(record, label=f"input {name}")
        for name, record in inputs.items()
    }
    source_path, source_sha = verified_inputs["source_config"]
    if source_path != pathlib.Path(spec.config).resolve() or source_sha != state.get("config_sha256"):
        raise HMMMatrixError(f"{spec.job_id}: evaluation source config differs from training")
    for name in ("pair_bank", "val_posteriors", "lengen_posteriors"):
        path, digest = verified_inputs[name]
        if identity_hashes.get(str(path)) != digest:
            raise HMMMatrixError(f"{spec.job_id}: evaluation input was not frozen: {path}")
    materialized_path, materialized_sha = verified_inputs["materialized_config"]
    try:
        materialized_rel = str(materialized_path.relative_to(pathlib.Path(spec.out_root).resolve()))
    except ValueError as exc:
        raise HMMMatrixError(f"{spec.job_id}: materialized config is outside the job root") from exc
    artifacts = state.get("artifacts")
    if not isinstance(artifacts, dict) or artifacts.get(materialized_rel) != materialized_sha:
        raise HMMMatrixError(f"{spec.job_id}: materialized config differs from training artifact")
    manifests = receipt.get("manifests")
    if not isinstance(manifests, list) or not manifests:
        raise HMMMatrixError(f"{spec.job_id}: evaluation receipt lacks frozen manifests")
    receipt_names: set[str] = set()
    for index, record in enumerate(manifests):
        path, digest = _verified_reference(record, label=f"manifest[{index}]")
        if identity_hashes.get(str(path)) != digest:
            raise HMMMatrixError(
                f"{spec.job_id}: evaluation manifest was not bound during training: {path}"
            )
        receipt_names.add(path.name)
    required_receipt_names = (
        frozenset({
            "hmm_family_inventory.sha256", "hmm_family.json",
            "hmm_family_materialization.json", "hmm_dataset.json", "hmm_eval_pairs.json",
            "hmm_eval_pairs.jsonl", "hmm_matrices.json", "hmm_thresholds.json",
        }) if spec.condition is not None else HMM_RECEIPT_MANIFEST_NAMES
    )
    missing_manifests = sorted(required_receipt_names - receipt_names)
    if missing_manifests:
        raise HMMMatrixError(
            f"{spec.job_id}: evaluation receipt omits frozen manifests: {missing_manifests}"
        )
    representation_path, representation_sha = _verified_reference(
        receipt.get("representation_manifest"), label="representation manifest"
    )
    try:
        representation = json.loads(representation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HMMMatrixError(f"{spec.job_id}: representation manifest is invalid JSON") from exc
    if (
        representation.get("schema") != "nextlat_forgetting/hmm_representation_plan/1"
        or representation.get("policy", {}).get("outcome_dependent_selection") is not False
    ):
        raise HMMMatrixError(f"{spec.job_id}: representation plan is not preregistered/frozen")
    cache_identity = _verify_representation_cache(receipt.get("representation_cache"))
    expected_cache_identity = {
        "job_id": spec.job_id,
        "model": spec.model,
        "seed": spec.seed,
        "checkpoint_sha256": checkpoint_sha,
        "config_sha256": verified_inputs["materialized_config"][1],
        "source_config_sha256": source_sha,
        "pair_bank_sha256": verified_inputs["pair_bank"][1],
        "val_posteriors_sha256": verified_inputs["val_posteriors"][1],
        "lengen_posteriors_sha256": verified_inputs["lengen_posteriors"][1],
        "representation_manifest_sha256": representation_sha,
        "evaluator_source_sha256": {
            str(_verified_reference(source, label=f"evaluator_source[{index}]")[0]):
            _verified_reference(source, label=f"evaluator_source[{index}]")[1]
            for index, source in enumerate(evaluator_sources)
        },
        "upstream_commit": PINNED_UPSTREAM_COMMIT,
    }
    if spec.condition is not None:
        expected_cache_identity["regime"] = expected_regime
    if any(cache_identity.get(key) != value for key, value in expected_cache_identity.items()):
        raise HMMMatrixError(f"{spec.job_id}: representation cache/input identity mismatch")
    chunk_rows = cache_identity.get("chunk_rows")
    if not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or chunk_rows <= 0:
        raise HMMMatrixError(f"{spec.job_id}: representation cache has invalid chunk_rows")
    metrics = receipt.get("metrics")
    required_metrics = HMM_FAMILY_REQUIRED_METRICS if spec.condition is not None else HMM_REQUIRED_METRICS
    if not isinstance(metrics, dict) or set(metrics) != required_metrics:
        missing = sorted(required_metrics - set(metrics or {}))
        extra = sorted(set(metrics or {}) - required_metrics)
        raise HMMMatrixError(
            f"{spec.job_id}: evaluation must report the frozen metric set; "
            f"missing={missing}, extra={extra}"
        )
    for name, payload in metrics.items():
        if not isinstance(payload, dict) or not payload:
            raise HMMMatrixError(f"{spec.job_id}: metric {name} has no result payload")
    _reject_nonfinite(receipt)
    return receipt


def hmm_evaluator_command(
    spec: JobSpec, state: t.Mapping[str, object], *, project_root: os.PathLike | str,
    snapshot_root: os.PathLike | str, upstream_root: os.PathLike | str,
    batch_size: int = 256,
) -> list[str]:
    """Build the one frozen evaluator interface used by the durable matrix driver."""
    project = pathlib.Path(project_root).resolve()
    snapshot = pathlib.Path(snapshot_root).resolve()
    checkpoint = pathlib.Path(str(state.get("final_checkpoint", ""))).resolve()
    materialized = checkpoint.parent / "materialized_config.yaml"
    output = pathlib.Path(spec.out_root).resolve() / HMM_EVALUATION_RECEIPT
    regime = spec.condition
    manifest_dir = snapshot / "manifests" / (
        pathlib.Path("hmm_family") / regime if regime is not None else pathlib.Path("")
    )
    data_dir = snapshot / "data" / (
        pathlib.Path("hmm_family") / regime if regime is not None else pathlib.Path("hmm")
    )
    command = [
        "fabric", "run", "--devices", "1", "--precision", "bf16-mixed",
        str(project / "scripts/evaluate_hmm_checkpoints.py"),
        "--job-id", spec.job_id, "--model", spec.model, "--seed", str(spec.seed),
        "--checkpoint", str(checkpoint), "--config", str(materialized),
        "--source-config", str(pathlib.Path(spec.config).resolve()),
        "--pair-bank", str(manifest_dir / "hmm_eval_pairs.jsonl"),
        "--val-posteriors", str(data_dir / "hmm4x4_val_posteriors.npz"),
        "--lengen-posteriors", str(data_dir / "hmm4x4_lengen_posteriors.npz"),
        "--upstream", str(pathlib.Path(upstream_root).resolve()),
        "--output", str(output), "--cache-root", str(output.parent / "representation_cache"),
        "--batch-size", str(batch_size),
    ]
    if spec.condition is not None:
        command.extend(("--regime", spec.condition))
    if regime is None:
        manifest_paths = [snapshot / "manifests" / name for name in sorted(HMM_RECEIPT_MANIFEST_NAMES)]
    else:
        manifest_paths = [
            snapshot / "manifests/hmm_family_inventory.sha256",
            snapshot / "manifests/hmm_family.json",
            snapshot / "manifests/hmm_family_materialization.json",
            *(manifest_dir / name for name in (
                "hmm_dataset.json", "hmm_eval_pairs.json", "hmm_eval_pairs.jsonl",
                "hmm_matrices.json", "hmm_thresholds.json",
            )),
        ]
    for path in manifest_paths:
        command.extend(("--manifest", str(path)))
    return command


def run_hmm_evaluators(
    jobs: t.Sequence[JobSpec], ledger: Ledger, *, project_root: os.PathLike | str,
    snapshot_root: os.PathLike | str, upstream_root: os.PathLike | str,
    batch_size: int = 256,
    command_runner: t.Callable[..., int] = subprocess.call,
) -> None:
    """Evaluate only after one atomic preflight proves all 30 cells are TRAINED."""
    states = preflight_hmm_evaluation_matrix(jobs, ledger)
    for spec in jobs:
        state = states.get(spec.job_id)
        assert state is not None  # established atomically before the first evaluator invocation
        receipt = pathlib.Path(spec.out_root) / HMM_EVALUATION_RECEIPT
        sidecar = pathlib.Path(spec.out_root) / HMM_EVALUATION_SIDECAR
        if receipt.is_file() and sidecar.is_file():
            verify_hmm_evaluation_receipt(spec, state)
            continue
        command = hmm_evaluator_command(
            spec, state, project_root=project_root, snapshot_root=snapshot_root,
            upstream_root=upstream_root, batch_size=batch_size,
        )
        returncode = command_runner(command, cwd=str(pathlib.Path(upstream_root).resolve()))
        if returncode != 0:
            raise HMMMatrixError(
                f"{spec.job_id}: evaluator failed with return code {returncode}; "
                "verified representation chunks remain resumable"
            )
        verify_hmm_evaluation_receipt(spec, state)


def _canonical_hmm_family_ids() -> tuple[str, ...]:
    return tuple(
        hmm_job_id(model, seed, regime)
        for regime in HMM_FAMILY_REGIMES
        for model in HMM_MODELS
        for seed in SEEDS
    )


def _verified_training_provenance(spec: JobSpec, state: t.Mapping[str, object]) -> None:
    """Deep-verify one terminal training state without reading any scientific outcome."""
    if state.get("status") not in TRAINING_TERMINAL:
        raise HMMMatrixError(f"{spec.job_id}: all canonical cells must be TRAINED before evaluation")
    if any(
        isinstance(state.get(key), bool)
        or not isinstance(state.get(key), int)
        or state.get(key) != HMM_TRAIN_UPDATES
        for key in ("step", "updates")
    ):
        raise HMMMatrixError(f"{spec.job_id}: TRAINED state is not exact step/update 3000")

    config = pathlib.Path(spec.config)
    if not config.is_file() or state.get("config_sha256") != sha256_file(config):
        raise HMMMatrixError(f"{spec.job_id}: TRAINED config provenance is invalid")
    expected_manifests = {
        path: sha256_file(pathlib.Path(path))
        for path in spec.manifests
        if pathlib.Path(path).is_file()
    }
    if len(expected_manifests) != len(spec.manifests) or state.get("manifest_sha256") != expected_manifests:
        raise HMMMatrixError(f"{spec.job_id}: TRAINED manifest provenance is invalid")
    expected_identity = {
        "job_id": spec.job_id,
        "model": spec.model,
        "seed": spec.seed,
        "phase": spec.phase,
        "condition": spec.condition,
    }
    if any(state.get(key) != value for key, value in expected_identity.items()):
        raise HMMMatrixError(f"{spec.job_id}: TRAINED ledger identity is invalid")
    try:
        recorded_root = pathlib.Path(str(state.get("out_root", ""))).resolve()
    except (OSError, RuntimeError) as exc:
        raise HMMMatrixError(f"{spec.job_id}: TRAINED output-root provenance is invalid") from exc
    if recorded_root != pathlib.Path(spec.out_root).resolve():
        raise HMMMatrixError(f"{spec.job_id}: TRAINED output-root provenance is invalid")

    checkpoint = pathlib.Path(str(state.get("final_checkpoint", ""))).resolve()
    checkpoint_sha = state.get("final_checkpoint_sha256")
    try:
        checkpoint.relative_to(recorded_root)
    except ValueError as exc:
        raise HMMMatrixError(f"{spec.job_id}: authoritative checkpoint is outside its job root") from exc
    if (
        not checkpoint.is_file()
        or not _valid_sha(checkpoint_sha)
        or sha256_file(checkpoint) != checkpoint_sha
    ):
        raise HMMMatrixError(f"{spec.job_id}: authoritative checkpoint/hash is invalid")

    artifacts = state.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise HMMMatrixError(f"{spec.job_id}: authoritative training artifacts are absent")
    ok, reason = verify_artifacts(spec.out_root, artifacts)
    if not ok:
        raise HMMMatrixError(f"{spec.job_id}: authoritative training artifact failed: {reason}")
    if COMPLETION_SUMMARY not in artifacts:
        raise HMMMatrixError(f"{spec.job_id}: authoritative completion provenance is absent")
    summary_path = pathlib.Path(spec.out_root) / COMPLETION_SUMMARY
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HMMMatrixError(f"{spec.job_id}: completion provenance is invalid JSON") from exc
    expected_summary = {
        "schema": "nextlat_forgetting/training_completion/1",
        "kind": "training_completion",
        "job_id": spec.job_id,
        "model": spec.model,
        "seed": spec.seed,
        "phase": spec.phase,
        "condition": spec.condition,
        "step": HMM_TRAIN_UPDATES,
        "updates": HMM_TRAIN_UPDATES,
    }
    checkpoint_record = summary.get("checkpoint")
    training_artifacts = summary.get("training_artifacts")
    if (
        any(summary.get(key) != value for key, value in expected_summary.items())
        or not isinstance(checkpoint_record, dict)
        or pathlib.Path(str(checkpoint_record.get("path", ""))).resolve() != checkpoint
        or checkpoint_record.get("sha256") != checkpoint_sha
        or not isinstance(training_artifacts, dict)
        or not training_artifacts
        or any(artifacts.get(path) != digest for path, digest in training_artifacts.items())
    ):
        raise HMMMatrixError(f"{spec.job_id}: completion/checkpoint provenance is inconsistent")
    recovery = state.get("recovery_provenance")
    if recovery is not None and (
        not isinstance(recovery, dict)
        or recovery.get("checkpoint_sha256") != checkpoint_sha
        or summary.get("recovery_provenance") != recovery
    ):
        raise HMMMatrixError(f"{spec.job_id}: recovery provenance is inconsistent")


def preflight_hmm_evaluation_matrix(
    jobs: t.Sequence[JobSpec], ledger: Ledger
) -> dict[str, dict]:
    """Atomically verify the exact canonical 30-job family before any evaluator runs."""
    validate_matrix(jobs)
    actual_ids = tuple(spec.job_id for spec in jobs)
    expected_ids = _canonical_hmm_family_ids()
    if actual_ids != expected_ids or len(jobs) != 30:
        raise HMMMatrixError(
            "evaluation requires the exact canonical 30-job HMM family in frozen order"
        )
    states = ledger.states()
    if set(states).intersection(expected_ids) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(states))
        raise HMMMatrixError(
            f"all exact canonical 30 jobs must be TRAINED before evaluation; missing={missing}"
        )
    for spec in jobs:
        _verified_training_provenance(spec, states[spec.job_id])
    return states


def promote_hmm_evaluations(
    jobs: t.Sequence[JobSpec], ledger: Ledger, *, echo: t.Callable[[str], None] = print
) -> dict[str, dict]:
    """Promote only; this phase cannot launch or resume training by construction."""
    validate_matrix(jobs)
    states = ledger.states()
    for spec in jobs:
        state = states.get(spec.job_id)
        if not state or state.get("status") not in TRAINING_TERMINAL:
            raise HMMMatrixError(
                f"{spec.job_id}: evaluation phase requires an existing TRAINED/DONE job"
            )
        ok, reason = verify_artifacts(spec.out_root, state.get("artifacts", {}))
        if not ok:
            raise HMMMatrixError(f"{spec.job_id}: training artifacts no longer verify: {reason}")
        verify_hmm_evaluation_receipt(spec, state)
        if state.get("status") == DONE:
            echo(f"[run_hmm_matrix] {spec.job_id}: DONE receipt re-verified, skipping")
            continue
        root = pathlib.Path(spec.out_root)
        evaluation_artifacts = {
            HMM_EVALUATION_RECEIPT: sha256_file(root / HMM_EVALUATION_RECEIPT),
            HMM_EVALUATION_SIDECAR: sha256_file(root / HMM_EVALUATION_SIDECAR),
        }
        promoted = {
            key: value for key, value in state.items()
            if key not in ("seq", "ts", "status")
        }
        promoted.update({
            "job_id": spec.job_id,
            "status": DONE,
            "supersedes": state.get("seq"),
            "artifacts": dict(state.get("artifacts", {}), **evaluation_artifacts),
            "evaluation_artifacts": evaluation_artifacts,
        })
        states[spec.job_id] = ledger.append(promoted)
        echo(f"[run_hmm_matrix] {spec.job_id}: TRAINED -> DONE")
    return states


def _selected_jobs(jobs: list[JobSpec], only: t.Sequence[str] | None) -> list[JobSpec]:
    if only is None:
        return jobs
    if not only:
        raise HMMMatrixError("operational --only selection may not be empty")
    wanted = set(only)
    known = {job.job_id for job in jobs}
    unknown = sorted(wanted - known)
    if unknown:
        raise HMMMatrixError(f"unknown --only HMM job ids: {unknown}")
    return [job for job in jobs if job.job_id in wanted]


def load_runtime_recovery_barrier(
    path: os.PathLike | str, jobs: t.Sequence[JobSpec]
) -> dict[str, dict]:
    """Load the atomic exact-ten barrier; it is resolved before a launcher is constructed."""
    candidate = pathlib.Path(path).resolve()
    try:
        document = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HMMMatrixError(f"runtime recovery barrier is absent or invalid: {exc}") from exc
    expected_ids = tuple(
        job.job_id for job in jobs if job.condition == "persistent_moderate"
    )
    if (document.get("schema") != RUNTIME_RECOVERY_BARRIER_SCHEMA or
            document.get("status") != "PASS" or
            tuple(document.get("job_ids", ())) != expected_ids or len(expected_ids) != 10):
        raise HMMMatrixError("runtime recovery barrier is not the canonical exact ten")
    records = document.get("jobs")
    if not isinstance(records, list) or len(records) != 10:
        raise HMMMatrixError("runtime recovery barrier does not contain ten job records")
    result: dict[str, dict] = {}
    for record in records:
        if not isinstance(record, dict) or record.get("job_id") in result:
            raise HMMMatrixError("runtime recovery barrier has an invalid/duplicate job")
        job_id = record.get("job_id")
        provenance = record.get("recovery_provenance")
        artifacts = record.get("authoritative_artifacts")
        if (job_id not in expected_ids or record.get("target_step") != HMM_TRAIN_UPDATES or
                not pathlib.Path(str(record.get("checkpoint_path", ""))).is_file() or
                sha256_file(record["checkpoint_path"]) != record.get("checkpoint_sha256") or
                not isinstance(provenance, dict) or
                provenance.get("checkpoint_sha256") != record.get("checkpoint_sha256") or
                not isinstance(artifacts, dict) or not artifacts or
                any(not pathlib.Path(name).is_file() or sha256_file(name) != digest
                    for name, digest in artifacts.items())):
            raise HMMMatrixError(f"{job_id}: runtime recovery barrier failed deep verification")
        result[t.cast(str, job_id)] = record
    if tuple(result) != expected_ids:
        raise HMMMatrixError("runtime recovery barrier order differs from the canonical matrix")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="durable root, normally /content/lurestar")
    parser.add_argument("--ledger", default=str(_REPO / "results" / "hmm_run_ledger.json"))
    parser.add_argument("--project-root", default=str(_REPO))
    parser.add_argument("--upstream", default=str(_REPO / "upstream" / "NextLat"))
    parser.add_argument(
        "--snapshot-root",
        help="root containing frozen manifests/data; default: --root (driver: /content/lurestar)",
    )
    parser.add_argument(
        "--data-root",
        help="root containing data/hmm arrays; must equal --snapshot-root",
    )
    parser.add_argument("--phase", choices=("train", "evaluate"), default="train")
    parser.add_argument(
        "--family", action="store_true",
        help="run the complete three-regime model-blind family (30 jobs); partial families refused",
    )
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--strategy", default="ddp")
    parser.add_argument("--evaluation-batch-size", type=int, default=256)
    parser.add_argument("--recovery-barrier", help=argparse.SUPPRESS)
    parser.add_argument(
        "--driver-managed-durability", action="store_true",
        help="attest that the hardened Colab driver is syncing artifacts then state.json; "
             "required for every mutating phase",
    )
    parser.add_argument(
        "--bucket", help=argparse.SUPPRESS,
    )
    args = parser.parse_args(argv)
    snapshot_root = pathlib.Path(args.snapshot_root or args.root).resolve()
    data_root = pathlib.Path(args.data_root or args.root).resolve()
    if data_root != snapshot_root:
        print(
            "[run_hmm_matrix] REFUSED before compute: --data-root must equal "
            "--snapshot-root so launched arrays are the arrays that were hash-verified",
            file=sys.stderr,
        )
        return 2

    try:
        jobs = build_hmm_matrix(
            args.root, project_root=args.project_root, upstream_root=args.upstream,
            snapshot_root=snapshot_root,
            regimes=HMM_FAMILY_REGIMES if args.family else None,
        )
        jobs = _selected_jobs(jobs, args.only)
    except (HMMMatrixError, FileNotFoundError, ValueError) as exc:
        print(f"[run_hmm_matrix] REFUSED before compute: {exc}", file=sys.stderr)
        return 2

    if args.print_plan:
        operational_selection = args.only is not None
        print(json.dumps({
            "schema": "nextlat_forgetting/hmm_matrix_plan/1",
            "confirmatory": not operational_selection,
            "confirmatory_aggregate_eligible": (
                args.family and not operational_selection and len(jobs) == 30
            ),
            "operational_recovery_selection_only": operational_selection,
            "models": list(HMM_MODELS),
            "regimes": list(HMM_FAMILY_REGIMES) if args.family else ["primary"],
            "seeds": list(SEEDS),
            "updates_per_job": HMM_TRAIN_UPDATES,
            "checkpoint_interval": HMM_CHECKPOINT_INTERVAL,
            "jobs": [job.to_dict() for job in jobs],
        }, indent=2, sort_keys=True))
        return 0

    if args.dry_run:
        launcher = HMMFabricLauncher(
            args.project_root, args.upstream, data_root=data_root,
            devices=args.devices, precision=args.precision, strategy=args.strategy,
            dry_run=True,
        )
        preview = {
            "schema": "nextlat_forgetting/hmm_matrix_dry_run/1",
            "phase": args.phase,
            "mutated_ledger": False,
            "jobs": [job.to_dict() for job in jobs],
        }
        if args.phase == "train":
            preview["commands"] = [
                launcher.command(ResumePlan(job, fresh=True)) for job in jobs
            ]
        else:
            preview["required_receipts"] = [
                str(pathlib.Path(job.out_root) / HMM_EVALUATION_RECEIPT) for job in jobs
            ]
            preview["evaluator"] = str(
                pathlib.Path(args.project_root).resolve() / "scripts/evaluate_hmm_checkpoints.py"
            )
            preview["evaluation_batch_size"] = args.evaluation_batch_size
        print(json.dumps(preview, indent=2, sort_keys=True))
        return 0

    if args.bucket:
        print(
            "[run_hmm_matrix] REFUSED: --bucket is not a live checkpoint daemon. "
            "Run through the hardened Colab driver instead.",
            file=sys.stderr,
        )
        return 2
    if not args.driver_managed_durability:
        print(
            "[run_hmm_matrix] REFUSED: mutating phases require "
            "--driver-managed-durability; direct execution could lose paid progress",
            file=sys.stderr,
        )
        return 2

    ledger = Ledger(args.ledger)
    try:
        if args.phase == "evaluate":
            run_hmm_evaluators(
                jobs, ledger, project_root=args.project_root, snapshot_root=snapshot_root,
                upstream_root=args.upstream, batch_size=args.evaluation_batch_size,
            )
            states = promote_hmm_evaluations(jobs, ledger)
        else:
            recovery_barrier = (
                load_runtime_recovery_barrier(args.recovery_barrier, jobs)
                if args.recovery_barrier else None
            )
            launcher = HMMFabricLauncher(
                args.project_root, args.upstream, data_root=data_root, devices=args.devices,
                precision=args.precision, strategy=args.strategy, dry_run=args.dry_run,
            )
            states = MatrixRunner(
                ledger, launcher, recovery_barrier=recovery_barrier
            ).run(jobs)
    except (HMMMatrixError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[run_hmm_matrix] REFUSED/FAILED: {exc}", file=sys.stderr)
        return 2

    required = DONE if args.phase == "evaluate" else TRAINED
    incomplete = [
        job.job_id for job in jobs
        if states.get(job.job_id, {}).get("status") not in (
            (DONE,) if required == DONE else TRAINING_TERMINAL
        )
    ]
    if incomplete:
        print(f"[run_hmm_matrix] incomplete: {incomplete}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
