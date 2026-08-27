#!/usr/bin/env python
"""Idempotent runner for the confirmatory job matrix (spec section 9, "Idempotent runner").

Running this twice must be a no-op the second time, and running it after a Colab disconnect must
pick each job up from its newest *verified* checkpoint. Concretely it:

  1. reads `results/run_ledger.json`;
  2. skips terminal `TRAINED`/`DONE` jobs only when every recorded artifact still hashes;
  3. resumes incomplete jobs from the newest valid checkpoint, rolling back one if the newest
     is corrupt (`DurableCheckpointer.resolve`);
  4. preserves config, seed, manifests and output root across the resume, and refuses to launch
     if any of them changed under it;
  5. writes atomic `metrics/step_{step}.json` keyed by `(run_id, step)`;
  6. marks `TRAINED` only after the verified final checkpoint and real upstream training
     artifacts exist; `DONE` is reserved for jobs whose caller-required scientific evaluation
     artifacts also exist and hash-verify;
  7. and only after the job took exactly the requested updates; both success-shaped underruns
     and overruns are failures.

The single most dangerous property it enforces is the output-root separation. Upstream's resume
pointers `recovery_ckpt` and `latest_ckpt` live at `trainer.out_dir`, one directory *above* the
experiment directory (`core_train.py:944-948`, `core_train.py:970-974`), and `init_from: resume`
reads them from there (`core_train.py:139-151`). The shipped configs give every algorithm and
every sweep seed the same `out_dir: output/stargraph` (`gpt_stargraph_5_5.yaml:14`), so whichever
job wrote last owns the pointer. If an H3 `far` branch ever resumed from the `near` branch's
pointer, both branches would silently share a parent and the near-minus-far contrast -- the whole
of H3 -- would be measuring nothing. `validate_matrix` makes that unrepresentable.

The matrix is three architecture-matched arms (spec section 8) at the five preregistered
seeds: `gpt`, `nextlat` and `bst`. BST is the COMPETENCE-MATCHED control -- the paper's
Figure 6 puts GPT on G(5,5) at ~18.6% (= 1/d, chance) and BST at ~99.9% -- so the primary
cross-model contrast is NextLat vs BST, where both arms solve the task and only the training
objective differs. NextLat vs GPT stays secondary and competence-confounded; BST vs GPT
measures how much of any effect is competence alone. See
`docs/DECISION_D20_competence_gate.md`, section "Superseded in part".

That makes 3 models x 5 seeds = 15 base jobs and 3 x 5 x {near, mid, far} = 45 adaptation
branches, 60 jobs in all.

Usage::

    python scripts/run_matrix.py --root gs-mirror/lurestar --print-plan
    python scripts/run_matrix.py --root /content/lurestar --phase base
    python scripts/run_matrix.py --root /content/lurestar --retry-sync
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import pathlib
import subprocess
import sys
import time
import typing as t

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from lurestar.durable_checkpoint import (  # noqa: E402
    DurableCheckpointer,
    DurableSync,
    atomic_write_json,
    sha256_file,
)

# Spec section 8, three architecture-matched arms. `bst` is the competence-matched control
# and the one that makes the cross-model geometry contrast identifiable
# (docs/DECISION_D20_competence_gate.md); it is not an optional extra.
MODELS = ("gpt", "nextlat", "bst")
SEEDS = (1234, 1235, 1236, 1237, 1238)  # PROGRAM.md: frozen confirmatory seeds
CONDITIONS = ("near", "mid", "far")

PENDING, RUNNING, INTERRUPTED, FAILED, TRAINED, DONE, STALE = (
    "PENDING", "RUNNING", "INTERRUPTED", "FAILED", "TRAINED", "DONE", "STALE",
)
TRAINING_TERMINAL = frozenset((TRAINED, DONE))

# Extra, caller-owned artifacts to require in addition to the production trainer's own output.
# Upstream never writes `final_summary.json`; the runner writes that completion receipt itself
# only after it has verified the checkpoint, materialized config, metrics log and update count.
DEFAULT_FINAL_ARTIFACTS: tuple[str, ...] = ()
COMPLETION_SUMMARY = "final_summary.json"
COMPETENCE_RECEIPT = "evaluation/base_competence.json"
COMPETENCE_RECEIPT_SIDECAR = f"{COMPETENCE_RECEIPT}.sha256"
COMPETENCE_THRESHOLD = 0.90
COMPETENCE_DECODING = {"strategy": "greedy", "top_k": 1, "temperature": 0.0}
DEFAULT_COMPETENCE_EVALUATOR = str(_REPO / "scripts" / "evaluate_base_competence.py")
DEFAULT_COMPETENCE_DATASET = str(
    _REPO / "data" / "stargraph" / "graph_5_5_test_20000.txt"
)
DEFAULT_COMPETENCE_MANIFESTS = (str(_REPO / "manifests" / "corpus.sha256"),)
ADAPTATION_CONTRACT = "h3_full_parameter_next_token_ce_v1"
ADAPTATION_CONTRACT_SOURCE = _REPO / "src" / "lurestar" / "adaptation.py"

# Spec section 9: the immutable dataset/lure manifests and their SHA-256 are part of what a
# resume must preserve. `main()` used to call `build_matrix` without them, so the manifest half
# of the identity guard had no input at all.
DEFAULT_MANIFESTS: dict[str, tuple[str, ...]] = {
    "base": (str(_REPO / "manifests" / "corpus.sha256"),
             str(_REPO / "manifests" / "corpus_provenance.json")),
    "near": (str(_REPO / "manifests" / "corpus.sha256"),
             str(_REPO / "manifests" / "corpus_provenance.json")),
    "mid": (str(_REPO / "manifests" / "corpus.sha256"),
            str(_REPO / "manifests" / "corpus_provenance.json")),
    "far": (str(_REPO / "manifests" / "corpus.sha256"),
            str(_REPO / "manifests" / "corpus_provenance.json")),
}

ADAPTATION_OUTPUTS = frozenset((
    "graph_5_5_bnear_5000.txt",
    "graph_5_5_bmid_5000.txt",
    "graph_5_5_bfar_5000.txt",
    "graph_5_5_bnearval_2000.txt",
    "graph_5_5_bmidval_2000.txt",
    "graph_5_5_bfarval_2000.txt",
))
ADAPTATION_SOURCES = frozenset((
    "near_manifest", "mid_candidates", "mid_selection", "far_candidates", "far_selection",
    "near_validation", "mid_validation", "far_validation", "acquisition_provenance",
))
H3_PERMANENT_BLOCK = _REPO / "manifests" / "h3_selected" / "PERMANENT_H3_BLOCK.json"
H3_PERMANENT_BLOCK_SHA256 = (
    "82d526ad5cb6ac5fb942790488a6b766e59b816acb27ed405a00852f40925778"
)
H3_FORBIDDEN_AMENDMENTS = (
    "candidate_expansion", "caliper_change", "weighting", "unmatched_restriction",
    "pilot_substitution", "matching_amendment",
)


def _verified_sidecar(path: pathlib.Path) -> tuple[pathlib.Path, str]:
    """Return a verified file and digest; adaptation inputs are immutable or refused."""
    sidecar = pathlib.Path(f"{path}.sha256")
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError(f"adaptation prerequisite lacks file or sidecar: {path}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if not fields or len(fields[0]) != 64:
        raise RuntimeError(f"invalid adaptation SHA-256 sidecar: {sidecar}")
    expected = fields[0].lower()
    if any(ch not in "0123456789abcdef" for ch in expected):
        raise RuntimeError(f"invalid adaptation SHA-256 sidecar: {sidecar}")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"adaptation SHA-256 mismatch for {path}: {actual} != {expected}"
        )
    return sidecar, actual


def verified_h3_permanent_block() -> tuple[str, str]:
    """Bind base training to the exact D40 outcome-blind exclusion of Lure-Star H3."""
    path = H3_PERMANENT_BLOCK.resolve()
    sidecar, digest = _verified_sidecar(path)
    if digest != H3_PERMANENT_BLOCK_SHA256:
        raise RuntimeError("canonical D40 permanent H3 block hash changed")
    try:
        block = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError("canonical D40 permanent H3 block is invalid") from exc
    expected = {
        "schema": "nextlat_forgetting/h3_mid_expansion/1",
        "status": "PERMANENT_H3_BLOCK",
        "reason": "D40_ONE_SHOT_EXPANSION_REMAINS_INFEASIBLE",
        "unmatched_count": 4,
        "no_further_amendments_permitted": True,
    }
    if any(block.get(key) != value for key, value in expected.items()):
        raise RuntimeError("canonical D40 permanent H3 block semantics changed")
    if tuple(block.get("forbidden", ())) != H3_FORBIDDEN_AMENDMENTS:
        raise RuntimeError("canonical D40 permanent H3 forbidden-actions contract changed")
    for key in (
        "combined_loss_sha256", "expanded_manifest_sha256", "unmatched_identity_sha256",
    ):
        value = block.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(
            char not in "0123456789abcdef" for char in value
        ):
            raise RuntimeError(f"canonical D40 permanent H3 block lacks {key}")
    return str(path), str(sidecar)


def verified_adaptation_manifests(adapt_dir: os.PathLike | str) -> tuple[str, ...]:
    """Verify the complete H3 bank gate and return every identity-bound artifact path.

    A near-only receipt is intentionally insufficient. The complete receipt, all six bank
    outputs and sidecars, and every source used for pilot selection/independent evaluation
    must verify before an adaptation job can even be planned.
    """
    root = pathlib.Path(adapt_dir).resolve()
    receipt_path = root / "adaptation_banks.json"
    receipt_sidecar, _ = _verified_sidecar(receipt_path)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"invalid adaptation materialization receipt: {receipt_path}") from exc
    if receipt.get("schema_version") != 1 or receipt.get("status") != "materialized":
        raise RuntimeError("adaptation receipt is not a schema-v1 materialized receipt")
    outputs = receipt.get("outputs")
    sources = receipt.get("sources")
    if not isinstance(outputs, dict) or set(outputs) != ADAPTATION_OUTPUTS:
        raise RuntimeError(
            "adaptation receipt is incomplete; all near/mid/far training and validation banks "
            "must be materialized"
        )
    if not isinstance(sources, dict) or set(sources) != ADAPTATION_SOURCES:
        raise RuntimeError(
            "adaptation receipt lacks the frozen pilot selection or independent validation "
            "sources"
        )

    bound = [receipt_path, receipt_sidecar]
    for name in sorted(ADAPTATION_OUTPUTS):
        path = root / name
        sidecar, digest = _verified_sidecar(path)
        if outputs[name] != digest:
            raise RuntimeError(f"adaptation receipt hash disagrees for {name}")
        bound.extend((path, sidecar))

    source_paths: dict[str, pathlib.Path] = {}
    for label in sorted(ADAPTATION_SOURCES):
        record = sources[label]
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise RuntimeError(f"adaptation source {label} lacks path/hash provenance")
        path = pathlib.Path(record["path"])
        if not path.is_absolute():
            path = (_REPO / path).resolve()
        sidecar, digest = _verified_sidecar(path)
        if record.get("sha256") != digest:
            raise RuntimeError(f"adaptation receipt source hash disagrees for {label}")
        source_paths[label] = path
        bound.extend((path, sidecar))

    required_attestations = {
        "role": "non_confirmatory",
        "frozen_before_confirmatory": True,
        "inspected_confirmatory_checkpoints": False,
        "inspected_confirmatory_results": False,
        "optimized_h3_outcomes": False,
    }
    for branch in ("mid", "far"):
        try:
            selection = json.loads(
                source_paths[f"{branch}_selection"].read_text(encoding="utf-8")
            )
        except (json.JSONDecodeError, OSError) as exc:
            raise RuntimeError(f"invalid frozen {branch}-selection pilot artifact") from exc
        pilot = selection.get("pilot") if isinstance(selection, dict) else None
        if not isinstance(pilot, dict) or any(
            pilot.get(key) != value for key, value in required_attestations.items()
        ):
            raise RuntimeError(
                f"{branch}-selection pilot provenance is not non-confirmatory and frozen"
            )
        for key in ("checkpoint_sha256", "loss_table_sha256", "selector_code_sha256"):
            value = pilot.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(
                ch not in "0123456789abcdef" for ch in value
            ):
                raise RuntimeError(f"{branch}-selection pilot lacks valid {key}")
    try:
        acquisition = json.loads(
            source_paths["acquisition_provenance"].read_text(encoding="utf-8")
        )
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError("invalid frozen acquisition provenance artifact") from exc
    acquisition_truths = {
        "frozen_before_confirmatory": True,
        "inspected_confirmatory_checkpoints": False,
        "inspected_confirmatory_results": False,
        "optimized_h3_outcomes": False,
        "disjoint_from_training": True,
        "matched_target_path_distribution": True,
        "matched_pilot_loss_deciles": True,
    }
    if not isinstance(acquisition, dict) or any(
        acquisition.get(key) != value for key, value in acquisition_truths.items()
    ):
        raise RuntimeError("acquisition-bank provenance is not outcome-blind and frozen")
    return tuple(str(path) for path in dict.fromkeys(bound))


def default_config_for(model: str, phase: str, condition: str | None) -> str:
    """The real deliverables in `configs/`, from `scripts/materialize_configs.py`.

    There is no `{model}_lurestar_base.yaml` and no `{model}_lurestar_adapt.yaml`. Each of
    the three arms has its own base file -- `gpt_lurestar.yaml`, `nextlat_lurestar.yaml`,
    `bst_lurestar.yaml` -- because each is a copy of a DIFFERENT official upstream YAML, and
    the BST one carries `model.bst_pair_minimum_gap: 2`, which `defaults.yaml:98` resolves to
    1 if it is ever lost. Near, mid and far are SEPARATE files because they are the three
    frozen H3 interventions; handing all branches one config would collapse that contrast at the
    configuration layer.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}")
    if phase == "base":
        return str(_REPO / "configs" / f"{model}_lurestar.yaml")
    if condition not in CONDITIONS:
        raise ValueError(
            f"an adaptation job needs condition in {CONDITIONS}, got {condition!r}"
        )
    return str(_REPO / "configs" / f"adapt_{condition}.yaml")


# Each `configs/adapt_{near,mid,far}.yaml` drives all three model arms, so each non-NextLat arm
# has to restate its own model selection on the command line. `core_train.py:38-46` picks the
# model class from these flags: `use_bst` is tested FIRST, then `use_nextlat`, then GPT as the
# fallthrough. We do not lean on that ordering -- every arm states both flags -- because a
# silent objective swap here is invisible in every artifact the run produces.
ADAPT_MODEL_OVERRIDES: dict[str, tuple[str, ...]] = {
    # `use_nextlat: true` at configs/adapt_near.yaml:20; without the flip a GPT branch trains
    # a NextLat model. `scripts/launch_train.sh` does the same thing for the same reason.
    "gpt": ("use_nextlat=false",),
    "nextlat": (),
    # The hash-guarded runtime adapter replaces BST's dense prefix-suffix loss only for an
    # ``-adapt-`` job. No pair-gap knob belongs to the common next-token CE estimand.
    "bst": ("use_nextlat=false", "use_bst=true"),
}


def default_overrides_for(model: str, phase: str) -> tuple[str, ...]:
    """`configs/adapt_{near,mid,far}.yaml` are derived from the NextLat G(5,5) YAML.

    Its key set is a superset of the GPT one, so every arm can share the file as long as the
    model-selection flags are overridden per arm. Base jobs need nothing: each arm's base
    config is a copy of its own official upstream YAML.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}")
    if phase != "adapt":
        return ()
    return ADAPT_MODEL_OVERRIDES[model]


# --------------------------------------------------------------------------------------
# job identity
# --------------------------------------------------------------------------------------

def job_id(model: str, seed: int, phase: str, condition: str | None = None) -> str:
    """Deterministic ids: `nextlat-s1234-base`, `gpt-s1235-adapt-near`, `bst-s1236-adapt-far`.

    Deterministic because the ledger, the output root and the GCS prefix are all keyed by it;
    a job id that depended on iteration order or a timestamp would make a resume a new job.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}")
    parts = [model, f"s{int(seed)}", phase]
    if condition is not None:
        parts.append(condition)
    return "-".join(parts)


def upstream_experiment_dir_name(experiment_name: str, seed: int) -> str:
    """Deviation D-18: the trainer RENAMES the experiment before it builds any path.

    `train.py:98-99`, verbatim::

        if "seed" not in experiment_name:
            experiment_name = experiment_name + f"-seed{config.seed}"

    and `config.trainer.experiment_name` is then overwritten with the result (`train.py:125`)
    and joined onto `trainer.out_dir` to build every checkpoint path (`core_train.py:933`,
    `core_train.py:959`). The job ids here are `gpt-s1234-base` / `bst-s1236-adapt-far`, which
    do NOT contain the substring "seed" -- `s1234` is not `seed` -- so the real on-disk
    directory is `{out_dir}/{job_id}-seed{seed}/`.

    A runner that predicts `{out_dir}/{job_id}/` instead points `DurableCheckpointer` at a
    directory that never exists: `adopt_existing` finds nothing, `resolve()` returns None, and
    `run_job` records "job exited 0 but left no verified checkpoint" -- marking a good
    20,000-step run FAILED, and then planning `init_from=scratch` on top of it on the retry.
    The substring test is honoured rather than worked around, so a job id that DOES contain
    "seed" is left alone exactly as upstream leaves it alone.
    """
    if "seed" not in experiment_name:
        return f"{experiment_name}-seed{int(seed)}"
    return experiment_name


@dataclasses.dataclass(frozen=True)
class JobSpec:
    job_id: str
    model: str
    seed: int
    phase: str                      # "base" | "adapt"
    condition: str | None           # None | "near" | "far"
    config: str
    out_root: str
    manifests: tuple[str, ...] = ()
    # Frozen before base training, not chosen after inspecting a checkpoint or result.  These
    # inputs are intentionally separate from `manifests`, which binds the training/adaptation
    # corpus.  A base checkpoint has no scientific identity without its planned evaluator and
    # held-out population.
    competence_evaluator: str | None = DEFAULT_COMPETENCE_EVALUATOR
    competence_dataset: str | None = DEFAULT_COMPETENCE_DATASET
    competence_manifests: tuple[str, ...] = DEFAULT_COMPETENCE_MANIFESTS
    parent_job_id: str | None = None
    train_batches: int = 20000
    final_artifacts: tuple[str, ...] = DEFAULT_FINAL_ARTIFACTS
    overrides: tuple[str, ...] = ()

    @property
    def experiment_name(self) -> str:
        """What we pass to `trainer.experiment_name=` on the command line."""
        return self.job_id

    @property
    def experiment_dir_name(self) -> str:
        """What upstream will actually have called the directory (D-18). Never `job_id`."""
        return upstream_experiment_dir_name(self.experiment_name, self.seed)

    @property
    def checkpoint_dir(self) -> str:
        """The real on-disk checkpoint directory: `{out_root}/{experiment_name}-seed{seed}`."""
        return str(pathlib.Path(self.out_root) / self.experiment_dir_name)

    @property
    def competence_identity(self) -> dict | None:
        """The pre-training evaluation identity represented by this plan."""
        if self.phase != "base":
            return None
        if not self.competence_evaluator or not self.competence_dataset:
            raise FileNotFoundError(
                f"{self.job_id}: base job lacks frozen evaluator/dataset provenance"
            )
        return competence_identity_from_paths(
            self.competence_evaluator, self.competence_dataset, self.competence_manifests
        )

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        # Properties are not dataclass fields, and `--print-plan` is how a human checks the
        # path the runner is about to hash. D-18 is invisible without it.
        d["experiment_name"] = self.experiment_name
        d["experiment_dir_name"] = self.experiment_dir_name
        d["checkpoint_dir"] = self.checkpoint_dir
        d["competence_identity"] = self.competence_identity
        return d


def build_matrix(
    root: os.PathLike | str,
    *,
    models: t.Sequence[str] = MODELS,
    seeds: t.Sequence[int] = SEEDS,
    config_for: t.Callable[[str, str, str | None], str] | None = None,
    manifests: t.Mapping[str, t.Sequence[str]] | None = None,
    competence_evaluator: os.PathLike | str = DEFAULT_COMPETENCE_EVALUATOR,
    competence_dataset: os.PathLike | str | None = None,
    competence_manifests: t.Sequence[os.PathLike | str] = DEFAULT_COMPETENCE_MANIFESTS,
    base_steps: int = 20000,
    adapt_steps: int = 500,
    final_artifacts: t.Sequence[str] = DEFAULT_FINAL_ARTIFACTS,
    require_configs: bool = True,
) -> list[JobSpec]:
    """One base job and one `near`/`mid`/`far` adaptation per (model, seed).

    With the shipped defaults that is 3 arms x 5 seeds = 15 base jobs and 45 adaptation
    branches. Directory layout is the spec's: `runs/{model}/{seed}/{phase}/{condition}/`,
    which keys the output root on the arm as well as the seed and the branch -- the three
    arms must not share a resume pointer any more than near, mid, and far must.
    """
    root = pathlib.Path(root)
    if config_for is None:
        config_for = default_config_for

    if manifests is None:
        manifests = DEFAULT_MANIFESTS
    if competence_dataset is None:
        # Production stages the immutable corpus under the durable run root; local development
        # keeps the same file in the repository.  Resolve this once while constructing the plan,
        # then hash the selected absolute path into every base ledger entry.
        staged = root / "data" / "stargraph" / "graph_5_5_test_20000.txt"
        competence_dataset = staged if staged.is_file() else DEFAULT_COMPETENCE_DATASET
    competence_manifests = tuple(str(path) for path in competence_manifests)
    jobs: list[JobSpec] = []
    for model in models:
        for seed in seeds:
            base_id = job_id(model, seed, "base")
            jobs.append(JobSpec(
                job_id=base_id, model=model, seed=seed, phase="base", condition=None,
                config=config_for(model, "base", None),
                out_root=str(root / "runs" / model / str(seed) / "base" / "_"),
                manifests=tuple(manifests.get("base", ())),
                competence_evaluator=str(competence_evaluator),
                competence_dataset=str(competence_dataset),
                competence_manifests=competence_manifests,
                train_batches=base_steps,
                final_artifacts=tuple(final_artifacts),
                overrides=default_overrides_for(model, "base"),
            ))
            for cond in CONDITIONS:
                jobs.append(JobSpec(
                    job_id=job_id(model, seed, "adapt", cond),
                    model=model, seed=seed, phase="adapt", condition=cond,
                    config=config_for(model, "adapt", cond),
                    out_root=str(root / "runs" / model / str(seed) / "adapt" / cond),
                    manifests=tuple(manifests.get(cond, ())),
                    competence_evaluator=None,
                    competence_dataset=None,
                    competence_manifests=(),
                    parent_job_id=base_id,
                    train_batches=adapt_steps,
                    final_artifacts=tuple(final_artifacts),
                    overrides=default_overrides_for(model, "adapt"),
                ))
    validate_matrix(jobs)
    if require_configs:
        missing = sorted({j.config for j in jobs if not pathlib.Path(j.config).is_file()})
        if missing:
            raise FileNotFoundError(
                "matrix references configs that do not exist: " + ", ".join(missing)
                + ". A job cannot launch without its config, and a config recorded as absent "
                "makes the identity guard vacuous (it would pin config_sha256=None)."
            )
    return jobs


def validate_matrix(jobs: t.Sequence[JobSpec]) -> None:
    """No two jobs may share, or nest inside, an output root; ids must be unique.

    Nesting matters as much as equality: an out_root that contains another job's out_root would
    put one job's `recovery_ckpt` on the resume search path of the other.
    """
    seen_ids: set[str] = set()
    roots: list[tuple[str, pathlib.Path]] = []
    for j in jobs:
        if j.job_id in seen_ids:
            raise ValueError(f"duplicate job id {j.job_id}")
        seen_ids.add(j.job_id)
        roots.append((j.job_id, pathlib.Path(j.out_root).resolve()))
    for i, (id_a, a) in enumerate(roots):
        for id_b, b in roots[i + 1:]:
            if a == b:
                raise ValueError(
                    f"{id_a} and {id_b} share out_root {a}; upstream's resume pointer lives at "
                    "out_dir level (core_train.py:944-948) and would cross branches"
                )
            if a in b.parents or b in a.parents:
                raise ValueError(f"{id_a} and {id_b} have nested out_roots: {a} vs {b}")


# --------------------------------------------------------------------------------------
# ledger
# --------------------------------------------------------------------------------------

class Ledger:
    """Append-only run ledger (PROGRAM.md invariant 1).

    Entries are never rewritten. A wrong entry is corrected by appending a superseding entry
    carrying a `reason`. The file is rewritten atomically as a whole each time, but the list of
    entries only ever grows, so the history of every job is recoverable.
    """

    def __init__(self, path: os.PathLike | str) -> None:
        self.path = pathlib.Path(path)

    def entries(self) -> list[dict]:
        if not self.path.is_file():
            return []
        doc = json.loads(self.path.read_text())
        return doc.get("entries", [])

    def append(self, entry: dict) -> dict:
        entries = self.entries()
        record = dict(entry)
        record.setdefault("ts", time.time())
        record["seq"] = len(entries)
        entries.append(record)
        atomic_write_json(self.path, {"schema": 1, "entries": entries})
        return record

    def state_of(self, job: str) -> dict | None:
        latest = None
        for e in self.entries():
            if e.get("job_id") == job:
                latest = e
        return latest

    def states(self) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for e in self.entries():
            if "job_id" in e:
                out[e["job_id"]] = e
        return out


# --------------------------------------------------------------------------------------
# artifacts and metrics
# --------------------------------------------------------------------------------------

def hash_artifacts(out_root: os.PathLike | str, rels: t.Sequence[str]) -> dict[str, str]:
    root = pathlib.Path(out_root)
    missing = [r for r in rels if not (root / r).is_file()]
    if missing:
        raise FileNotFoundError(f"missing final artifacts under {root}: {missing}")
    return {r: sha256_file(root / r) for r in rels}


def verify_artifacts(out_root: os.PathLike | str, recorded: t.Mapping[str, str]) -> tuple[bool, str]:
    root = pathlib.Path(out_root)
    for rel, want in recorded.items():
        p = root / rel
        if not p.is_file():
            return False, f"artifact {rel} is missing"
        got = sha256_file(p)
        if got != want:
            return False, f"artifact {rel} hash {got[:12]} != recorded {want[:12]}"
    return True, "ok"


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def competence_identity_from_paths(
    evaluator: os.PathLike | str,
    dataset: os.PathLike | str,
    manifests: t.Sequence[os.PathLike | str],
) -> dict:
    """Return the exact, hash-bound evaluation plan frozen before base compute.

    Paths are part of the identity in addition to digests: silently redirecting an evaluation
    to a different population with coincidentally duplicated bytes is still a plan change.  The
    manifest must contain an exact sha256sum-style binding for the held-out file, not merely the
    digest somewhere in arbitrary text.
    """
    evaluator_path = pathlib.Path(evaluator).resolve()
    dataset_path = pathlib.Path(dataset).resolve()
    manifest_paths = [pathlib.Path(path).resolve() for path in manifests]
    for label, path in (("competence evaluator", evaluator_path),
                        ("held-out competence dataset", dataset_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} is missing: {path}")
    if not manifest_paths:
        raise FileNotFoundError("at least one competence evaluation manifest is required")
    missing = [str(path) for path in manifest_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"competence evaluation manifests are missing: {missing}")

    dataset_sha = sha256_file(dataset_path)
    bound = False
    for manifest in manifest_paths:
        for raw in manifest.read_text(encoding="utf-8").splitlines():
            fields = raw.strip().split()
            if len(fields) >= 2 and fields[0].lower() == dataset_sha:
                name = fields[1].lstrip("*")
                if pathlib.Path(name).name == dataset_path.name:
                    bound = True
                    break
    if not bound:
        raise RuntimeError(
            "no competence evaluation manifest exactly binds the held-out dataset name/SHA"
        )
    return {
        "evaluator": {"path": str(evaluator_path), "sha256": sha256_file(evaluator_path)},
        "dataset": {"path": str(dataset_path), "sha256": dataset_sha},
        "manifests": [
            {"path": str(path), "sha256": sha256_file(path)}
            for path in sorted(manifest_paths, key=lambda item: str(item))
        ],
        "decoding": dict(COMPETENCE_DECODING),
    }


def verify_parent_training_artifacts(parent: t.Mapping[str, object]) -> dict[str, str]:
    """Verify the complete TRAINED base output set and its internal completion receipt."""
    out_root_raw = parent.get("out_root")
    artifacts_raw = parent.get("artifacts")
    if not isinstance(out_root_raw, str) or not isinstance(artifacts_raw, dict):
        raise RuntimeError("base parent lacks output root or hashed training artifacts")
    evaluation_raw = parent.get("evaluation_artifacts", {})
    if not isinstance(evaluation_raw, dict):
        raise RuntimeError("base parent has invalid evaluation artifact provenance")
    training_artifacts = {
        str(rel): str(digest) for rel, digest in artifacts_raw.items()
        if rel not in evaluation_raw
    }
    if COMPLETION_SUMMARY not in training_artifacts:
        raise RuntimeError("base parent lacks its hash-bound training completion summary")
    if any(not _is_sha256(digest) for digest in training_artifacts.values()):
        raise RuntimeError("base parent contains an invalid training artifact digest")
    ok, reason = verify_artifacts(out_root_raw, training_artifacts)
    if not ok:
        raise RuntimeError(f"base parent training artifact verification failed: {reason}")

    summary_path = pathlib.Path(out_root_raw) / COMPLETION_SUMMARY
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError("base parent training completion summary is invalid") from exc
    expected_summary = {
        "schema": "nextlat_forgetting/training_completion/1",
        "kind": "training_completion",
        "job_id": parent.get("job_id"),
        "model": parent.get("model"),
        "seed": parent.get("seed"),
        "phase": "base",
        "condition": None,
        "step": parent.get("step"),
        "updates": parent.get("updates"),
    }
    if any(summary.get(key) != value for key, value in expected_summary.items()):
        raise RuntimeError("base parent training completion summary identity mismatch")
    checkpoint = summary.get("checkpoint")
    if not isinstance(checkpoint, dict) or (
        checkpoint.get("path") != str(pathlib.Path(str(parent.get("final_checkpoint"))).resolve())
        or checkpoint.get("sha256") != parent.get("final_checkpoint_sha256")
    ):
        raise RuntimeError("base parent completion summary checkpoint mismatch")
    expected_payload = dict(training_artifacts)
    expected_payload.pop(COMPLETION_SUMMARY)
    if summary.get("training_artifacts") != expected_payload:
        raise RuntimeError("base parent completion summary does not bind every training artifact")
    return training_artifacts


def verify_base_competence_receipt(
    parent: t.Mapping[str, object], *, expected_job_id: str, model: str, seed: int
) -> dict:
    """Verify the immutable scientific receipt required before an H3 branch is planned.

    Training completion is deliberately insufficient.  A valid parent is a ``DONE`` base job
    whose append-only ledger entry hashes both the receipt and its SHA sidecar.  The receipt in
    turn binds the exact checkpoint, evaluator source, model, seed, and exact-path counts.  This
    prevents a stale evaluation from being silently reused after a checkpoint/evaluator change.
    """
    status = parent.get("status")
    if status != DONE:
        raise RuntimeError(
            f"{expected_job_id}: parent is {status or 'missing'}, not DONE; TRAINED-only "
            "parents have no verified scientific competence result"
        )
    if parent.get("job_id") != expected_job_id:
        raise RuntimeError("competence parent job_id does not match the requested branch parent")
    if parent.get("phase") != "base" or parent.get("model") != model or parent.get("seed") != seed:
        raise RuntimeError("competence parent model/seed/base identity does not match the branch")

    out_root = parent.get("out_root")
    artifacts = parent.get("artifacts")
    if not isinstance(out_root, str) or not isinstance(artifacts, dict):
        raise RuntimeError("DONE parent lacks its output root or hashed artifacts")
    verify_parent_training_artifacts(parent)
    frozen_identity = parent.get("competence_identity")
    if not isinstance(frozen_identity, dict):
        raise RuntimeError("DONE parent lacks the pre-training competence identity")
    receipt_path = pathlib.Path(out_root) / COMPETENCE_RECEIPT
    sidecar_path = pathlib.Path(out_root) / COMPETENCE_RECEIPT_SIDECAR
    for rel, path in (
        (COMPETENCE_RECEIPT, receipt_path),
        (COMPETENCE_RECEIPT_SIDECAR, sidecar_path),
    ):
        recorded = artifacts.get(rel)
        if not _is_sha256(recorded) or not path.is_file():
            raise RuntimeError(f"DONE parent lacks hash-bound competence artifact {rel}")
        actual = sha256_file(path)
        if actual != recorded:
            raise RuntimeError(f"tampered competence artifact {rel}: {actual} != {recorded}")

    fields = sidecar_path.read_text(encoding="utf-8").strip().split()
    receipt_sha = sha256_file(receipt_path)
    if not fields or fields[0].lower() != receipt_sha:
        raise RuntimeError("competence receipt SHA sidecar does not match the receipt")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError("competence receipt is not valid JSON") from exc
    if receipt.get("schema") != "nextlat_forgetting/base_competence/1":
        raise RuntimeError("competence receipt has an unsupported schema")
    expected = {
        "job_id": expected_job_id,
        "model": model,
        "seed": seed,
        "phase": "base",
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError("competence receipt model/seed/job/base identity mismatch")

    checkpoint = receipt.get("checkpoint")
    parent_ckpt = parent.get("final_checkpoint")
    parent_sha = parent.get("final_checkpoint_sha256")
    if not isinstance(checkpoint, dict) or not _is_sha256(parent_sha):
        raise RuntimeError("competence receipt or parent lacks checkpoint provenance")
    if checkpoint.get("sha256") != parent_sha:
        raise RuntimeError("competence receipt checkpoint SHA does not match the DONE parent")
    if not isinstance(parent_ckpt, str) or pathlib.Path(checkpoint.get("path", "")).resolve() != pathlib.Path(parent_ckpt).resolve():
        raise RuntimeError("competence receipt checkpoint path does not match the DONE parent")
    if not pathlib.Path(parent_ckpt).is_file() or sha256_file(parent_ckpt) != parent_sha:
        raise RuntimeError("DONE parent checkpoint is missing or no longer matches its SHA")

    evaluator = receipt.get("evaluator")
    if not isinstance(evaluator, dict) or not _is_sha256(evaluator.get("sha256")):
        raise RuntimeError("competence receipt lacks evaluator path/SHA provenance")
    evaluator_path_raw = evaluator.get("path")
    if not isinstance(evaluator_path_raw, str):
        raise RuntimeError("competence receipt lacks evaluator path/SHA provenance")
    evaluator_path = pathlib.Path(evaluator_path_raw)
    if not evaluator_path.is_absolute():
        evaluator_path = (_REPO / evaluator_path).resolve()
    if not evaluator_path.is_file() or sha256_file(evaluator_path) != evaluator["sha256"]:
        raise RuntimeError("competence evaluator is missing or its SHA no longer matches")

    evaluator_output = receipt.get("evaluator_output")
    dataset = receipt.get("evaluation_dataset")
    manifests = receipt.get("manifests")
    provenance_paths: dict[str, pathlib.Path] = {}
    for label, record in (
        ("evaluator output", evaluator_output),
        ("evaluation dataset", dataset),
    ):
        if not isinstance(record, dict) or not _is_sha256(record.get("sha256")):
            raise RuntimeError(f"competence receipt lacks {label} path/SHA provenance")
        raw = record.get("path")
        if not isinstance(raw, str):
            raise RuntimeError(f"competence receipt lacks {label} path/SHA provenance")
        path = pathlib.Path(raw)
        if not path.is_absolute():
            path = (_REPO / path).resolve()
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError(f"competence {label} is missing or its SHA no longer matches")
        provenance_paths[label] = path
    if not isinstance(manifests, list) or not manifests:
        raise RuntimeError("competence receipt lacks evaluation manifest provenance")
    manifest_paths: list[pathlib.Path] = []
    for record in manifests:
        if not isinstance(record, dict) or not _is_sha256(record.get("sha256")):
            raise RuntimeError("competence receipt has invalid evaluation manifest provenance")
        raw = record.get("path")
        if not isinstance(raw, str):
            raise RuntimeError("competence receipt has invalid evaluation manifest provenance")
        path = pathlib.Path(raw)
        if not path.is_absolute():
            path = (_REPO / path).resolve()
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise RuntimeError("competence evaluation manifest is missing or its SHA mismatches")
        manifest_paths.append(path)
    dataset_sha = t.cast(dict, dataset)["sha256"]
    if not any(dataset_sha in path.read_text(encoding="utf-8") for path in manifest_paths):
        raise RuntimeError("no competence evaluation manifest binds the held-out dataset SHA")
    if receipt.get("decoding") != COMPETENCE_DECODING:
        raise RuntimeError(
            "competence receipt decoding must be deterministic greedy/top_k=1/temperature=0"
        )
    receipt_identity = {
        "evaluator": evaluator,
        "dataset": dataset,
        "manifests": manifests,
        "decoding": receipt.get("decoding"),
    }
    if receipt.get("competence_identity") != frozen_identity or receipt_identity != frozen_identity:
        raise RuntimeError(
            "competence receipt does not exactly match the pre-training evaluator/data/manifest "
            "identity"
        )

    metric = receipt.get("exact_path_accuracy")
    if not isinstance(metric, dict):
        raise RuntimeError("competence receipt lacks exact_path_accuracy")
    correct, total, value = metric.get("correct"), metric.get("total"), metric.get("value")
    if (
        not isinstance(correct, int) or isinstance(correct, bool)
        or not isinstance(total, int) or isinstance(total, bool)
        or total <= 0 or not 0 <= correct <= total
        or not isinstance(value, (int, float)) or isinstance(value, bool)
        or not math.isfinite(float(value))
        or abs(float(value) - correct / total) > 1e-12
    ):
        raise RuntimeError("competence receipt has invalid or inconsistent exact-path counts")
    try:
        raw = json.loads(provenance_paths["evaluator output"].read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError("competence evaluator output is not valid JSON") from exc
    raw_expected = {
        "schema": "nextlat_forgetting/exact_path_evaluation/1",
        "job_id": expected_job_id,
        "model": model,
        "seed": seed,
        "checkpoint_sha256": parent_sha,
        "dataset_sha256": dataset_sha,
        "evaluator_sha256": evaluator["sha256"],
        "manifest_sha256s": sorted(record["sha256"] for record in manifests),
        "decoding": COMPETENCE_DECODING,
        "exact_path_accuracy": metric,
    }
    if any(raw.get(key) != value for key, value in raw_expected.items()):
        raise RuntimeError("competence receipt does not match the bound evaluator output")
    if model in ("nextlat", "bst") and float(value) < COMPETENCE_THRESHOLD:
        raise RuntimeError(
            f"{expected_job_id}: exact-path accuracy {float(value):.6f} is below the "
            f"preregistered {COMPETENCE_THRESHOLD:.2f} competence gate"
        )
    # GPT is the preregistered chance arm. Its valid score is reported and hash-bound but never
    # converted into a hidden pass condition.
    return receipt


def collect_training_artifacts(
    spec: JobSpec, *, authoritative_artifacts: t.Mapping[str, str] | None = None
) -> dict[str, str]:
    """Hash the artifacts the real upstream trainer writes for every successful job.

    `train.py:184-194` writes `materialized_config.yaml` in the experiment directory and its
    `CSVLogger` writes `version_N/metrics.csv` below the same directory (`train.py:101-109`).
    The old training contract required a root-level `final_summary.json`, which only the numpy
    test trainer ever created; a paid upstream job therefore could not reach `TRAINED`.

    Every CSV logger version is retained and hashed because a resumed job may legitimately
    create another `version_N`. Empty logs are refused: presence alone is not evidence that
    the trainer recorded a step or validation result.
    """
    root = pathlib.Path(spec.out_root)
    experiment = pathlib.Path(spec.checkpoint_dir)
    materialized = experiment / "materialized_config.yaml"
    metrics = sorted(experiment.glob("version_*/metrics.csv"))

    if authoritative_artifacts is not None:
        authoritative = {
            str(pathlib.Path(path).resolve()): digest
            for path, digest in authoritative_artifacts.items()
        }
        candidates = [materialized, *metrics, root / "metrics" / "step_0_contract.json"]
        for candidate in candidates:
            expected = authoritative.get(str(candidate.resolve()))
            if (not candidate.is_file() or expected != sha256_file(candidate)):
                if candidate in metrics and candidate.is_file():
                    quarantine = root / "quarantine" / "unbound_retry_telemetry" / str(
                        candidate.relative_to(root))
                    quarantine.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(candidate, quarantine)
                    continue
                raise ValueError(
                    f"{spec.job_id}: authoritative terminal state does not bind {candidate}"
                )
        metrics = [
            path for path in metrics
            if path.is_file() and authoritative.get(str(path.resolve())) == sha256_file(path)
        ]

    missing: list[str] = []
    if not materialized.is_file():
        missing.append(str(materialized))
    if not metrics:
        missing.append(str(experiment / "version_*/metrics.csv"))
    empty = [str(p) for p in metrics if p.stat().st_size == 0]
    if missing or empty:
        detail = []
        if missing:
            detail.append(f"missing {missing}")
        if empty:
            detail.append(f"empty {empty}")
        raise FileNotFoundError("upstream training artifacts are incomplete: " + "; ".join(detail))

    paths = [materialized, *metrics]
    step0_contract = root / "metrics" / "step_0_contract.json"
    if not step0_contract.is_file():
        raise FileNotFoundError(
            f"runtime step-0 contract receipt is missing: {step0_contract}"
        )
    try:
        step0 = json.loads(step0_contract.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"runtime step-0 contract is invalid: {step0_contract}") from exc
    adaptation = step0.get("adaptation")
    if spec.phase == "adapt":
        if not ADAPTATION_CONTRACT_SOURCE.is_file():
            raise FileNotFoundError(
                f"common adaptation trainer source is missing: {ADAPTATION_CONTRACT_SOURCE}"
            )
        expected = {
            "contract": ADAPTATION_CONTRACT,
            "contract_sha256": sha256_file(ADAPTATION_CONTRACT_SOURCE),
            "full_parameter": True,
            "loss": "teacher_forced_next_token_cross_entropy",
        }
        if not isinstance(adaptation, dict) or any(
            adaptation.get(key) != value for key, value in expected.items()
        ):
            raise ValueError(
                f"{spec.job_id}: runtime receipt does not bind the common adaptation trainer"
            )
        if spec.model == "bst" and (
            adaptation.get("bst_dense_prefix_suffix_objective") is not False
            or adaptation.get("bst_backward_input") != "item_independent_lone_eos"
        ):
            raise ValueError(
                f"{spec.job_id}: runtime receipt does not prove the BST generation-time path"
            )
    elif adaptation is not None:
        raise ValueError(f"{spec.job_id}: base training unexpectedly installed adaptation logic")
    paths.append(step0_contract)
    for rel in spec.final_artifacts:
        paths.append(root / rel)
    rels: list[str] = []
    for path in paths:
        try:
            rels.append(str(path.relative_to(root)))
        except ValueError as exc:
            raise ValueError(f"artifact {path} is outside job out_root {root}") from exc
    return hash_artifacts(root, rels)


def write_completion_summary(
    spec: JobSpec,
    *,
    step: int,
    updates: int,
    checkpoint_path: str,
    checkpoint_sha256: str,
    training_artifacts: t.Mapping[str, str],
    recovery_provenance: t.Mapping[str, object] | None = None,
) -> pathlib.Path:
    """Write the runner-owned, atomic receipt that a training job really completed.

    This is a completion receipt, not a fabricated evaluation result. Scientific evaluation
    artifacts are produced later and may be supplied through `final_artifacts` when a caller
    wants the matrix to require them as well.
    """
    path = pathlib.Path(spec.out_root) / COMPLETION_SUMMARY
    document = {
        "schema": "nextlat_forgetting/training_completion/1",
        "kind": "training_completion",
        "job_id": spec.job_id,
        "model": spec.model,
        "seed": spec.seed,
        "phase": spec.phase,
        "condition": spec.condition,
        "step": int(step),
        "updates": int(updates),
        "checkpoint": {
            "path": str(pathlib.Path(checkpoint_path).resolve()),
            "sha256": checkpoint_sha256,
        },
        "training_artifacts": dict(training_artifacts),
    }
    if recovery_provenance is not None:
        document["recovery_provenance"] = dict(recovery_provenance)
    atomic_write_json(path, document)
    return path


def write_step_metrics(out_root: os.PathLike | str, run_id: str, step: int, payload: dict) -> pathlib.Path:
    """Atomic `metrics/step_{step}.json` keyed by `(run_id, step)`.

    Keyed means keyed: rewriting the same step from a *different* run id is a bug (two jobs
    sharing an output root), so it raises instead of silently overwriting.
    """
    path = pathlib.Path(out_root) / "metrics" / f"step_{int(step)}.json"
    if path.is_file():
        try:
            existing = json.loads(path.read_text())
        except json.JSONDecodeError:
            existing = {}
        if existing.get("run_id") not in (None, run_id):
            raise ValueError(
                f"{path} already belongs to run {existing['run_id']!r}, refusing to overwrite "
                f"with {run_id!r} -- two jobs are sharing an output root"
            )
    body = dict(payload)
    body["run_id"] = run_id
    body["step"] = int(step)
    atomic_write_json(path, body)
    return path


# --------------------------------------------------------------------------------------
# resume planning
# --------------------------------------------------------------------------------------

@dataclasses.dataclass
class ResumePlan:
    spec: JobSpec
    fresh: bool
    resume_step: int = 0
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    rolled_back_from: str | None = None
    parent_checkpoint: str | None = None
    parent_checkpoint_sha256: str | None = None
    parent_steps: int | None = None

    @property
    def init_from(self) -> str:
        return "scratch" if self.fresh else "resume"

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["spec"] = self.spec.to_dict()
        d["init_from"] = self.init_from
        return d


@dataclasses.dataclass
class LaunchResult:
    returncode: int
    final_step: int | None = None
    detail: str = ""


class FabricLauncher:
    """Builds and runs the real single-GPU command. Child output is relayed line by line.

    Two transport bugs already cost a smoke test each (docs/RUNLOG.md): a child's stdout does
    not reach the `colab exec` stream unless it is relayed in-process, and piping the command
    into `tail` returns *tail's* exit status, so a crashed run reports RC=0. Neither is repeated
    here: output is relayed through a pipe we read ourselves, and the returncode is the child's.
    """

    def __init__(self, upstream_dir: os.PathLike | str, *, devices: int = 1,
                 precision: str = "bf16-mixed", strategy: str = "ddp",
                 python: str = sys.executable, dry_run: bool = False,
                 echo: t.Callable[[str], None] = print) -> None:
        self.upstream_dir = pathlib.Path(upstream_dir)
        self.devices = devices
        self.precision = precision
        self.strategy = strategy
        self.python = python
        self.dry_run = dry_run
        self.echo = echo

    def command(self, plan: ResumePlan) -> list[str]:
        spec = plan.spec
        cmd = [
            "fabric", "run",
            "--devices", str(self.devices),
            "--strategy", self.strategy,
            "--precision", self.precision,
            "train.py", "--config", spec.config,
        ]
        if spec.parent_job_id is not None and plan.parent_steps is None:
            raise ValueError(
                f"{spec.job_id}: refusing to emit a branch command without the parent's step "
                "count. --checkpoint_path restores training_steps (model_base.py:437), the "
                "trainer seeds self.step from it (core_train.py:309), and production's guarded "
                "stop-rule patch interprets train_batches as an absolute target. A relative "
                "500 cannot represent 500 updates from a 20,000-step parent."
            )
        if plan.parent_checkpoint and plan.fresh:
            # train.py:262-264; --checkpoint_path takes precedence over init_from
            # (core_train.py:130) and restores weights+optimizer+step, which is how an H3
            # branch starts from the frozen base parent.
            cmd += ["--checkpoint_path", plan.parent_checkpoint]
        # The step counter of a branch is offset by the parent's, on the fresh launch (via
        # --checkpoint_path) and on every later resume (the branch's own checkpoints carry the
        # offset counter too), so train_batches has to carry the same offset. The guarded
        # runtime patch stops at `>=` that absolute target. `spec.train_batches` stays the
        # number of ADAPTATION updates,
        # which is the quantity PROGRAM.md freezes.
        step_offset = plan.parent_steps or 0
        # Everything else is an OmegaConf dotlist override (train.py:265, train.py:349).
        cmd += [
            f"seed={spec.seed}",
            f"trainer.out_dir={pathlib.Path(spec.out_root).resolve()}",
            f"trainer.experiment_name={spec.experiment_name}",
            f"trainer.init_from={plan.init_from}",
            f"trainer.train_batches={step_offset + spec.train_batches}",
            "trainer.compile=false",          # spec section 8; README.md:117-122
            "trainer.log_to_wandb=false",
            "trainer.save_recovery_checkpoint=250",
        ]
        cmd += list(spec.overrides)
        return cmd

    def __call__(self, plan: ResumePlan) -> LaunchResult:
        cmd = self.command(plan)
        self.echo(f"[run_matrix] {plan.spec.job_id}: " + " ".join(cmd))
        if self.dry_run:
            return LaunchResult(0, None, "dry-run")
        proc = subprocess.Popen(
            cmd, cwd=str(self.upstream_dir), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
            env=dict(os.environ, PYTHONUNBUFFERED="1"),
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            self.echo(line.rstrip("\n"))
        rc = proc.wait()
        return LaunchResult(rc, None, f"returncode={rc}")


# --------------------------------------------------------------------------------------
# the runner
# --------------------------------------------------------------------------------------

class MatrixRunner:
    def __init__(self, ledger: Ledger, launcher: t.Callable[[ResumePlan], LaunchResult], *,
                 serializer=None, sync: DurableSync | None = None,
                 echo: t.Callable[[str], None] = print,
                 recovery_barrier: t.Mapping[str, dict] | None = None) -> None:
        self.ledger = ledger
        self.launcher = launcher
        self.serializer = serializer
        self.sync = sync
        self.echo = echo
        self.recovery_barrier = dict(recovery_barrier or {})

    def checkpointer(self, spec: JobSpec) -> DurableCheckpointer:
        return DurableCheckpointer(
            # `experiment_name=` here names a DIRECTORY, so it must be the name upstream
            # will have used, not the one we asked for on the command line (D-18).
            spec.out_root, spec.job_id, experiment_name=spec.experiment_dir_name,
            serializer=self.serializer, sync=self.sync, logger=self.echo,
        )

    # ---- identity guard --------------------------------------------------------------
    def _identity(self, spec: JobSpec) -> dict:
        """Config, seed, manifests and output root, hashed. Spec section 9.3 item 4.

        A missing config is refused here rather than recorded as `None`: an identity whose
        `config_sha256` is `None` cannot be compared against anything, which turns the whole
        guard into a no-op for the one key PROGRAM.md most needs it on.
        """
        cfg = pathlib.Path(spec.config)
        if not cfg.is_file():
            raise FileNotFoundError(
                f"{spec.job_id}: config {cfg} does not exist. A job cannot launch without it, "
                "and recording config_sha256=None would make the identity guard vacuous."
            )
        missing_manifests = [m for m in spec.manifests if not pathlib.Path(m).is_file()]
        if missing_manifests:
            raise FileNotFoundError(
                f"{spec.job_id}: manifests missing: {missing_manifests}. Spec section 9 "
                "requires the dataset/lure manifests to be persisted before training."
            )
        identity = {
            "model": spec.model,
            "seed": spec.seed,
            "phase": spec.phase,
            "condition": spec.condition,
            "out_root": str(pathlib.Path(spec.out_root).resolve()),
            "config": str(cfg),
            "config_sha256": sha256_file(cfg),
            "manifest_sha256": {m: sha256_file(m) for m in spec.manifests},
        }
        if spec.phase == "base":
            identity["competence_identity"] = spec.competence_identity
        return identity

    def _check_identity(self, spec: JobSpec, prior: dict | None) -> None:
        """Any recorded value that no longer matches is a refusal -- `None` included.

        The `prior[key] is not None` clause this replaces meant that a job whose first ledger
        entry was written while its config was absent would accept ANY config forever after.
        """
        if not prior:
            return
        now = self._identity(spec)
        for key in (
            "model", "seed", "phase", "condition", "out_root", "config_sha256",
            "manifest_sha256", "competence_identity",
        ):
            if key not in prior:
                if key == "competence_identity" and spec.phase == "base":
                    raise RuntimeError(
                        f"{spec.job_id}: prior ledger entry predates the frozen competence "
                        "identity; refusing to reuse it as a confirmatory base job"
                    )
                continue
            if prior[key] != now[key]:
                raise RuntimeError(
                    f"{spec.job_id}: {key} changed since the last ledger entry "
                    f"({prior[key]!r} -> {now[key]!r}). A resume must preserve config, seed, "
                    "manifest and output root; refusing to continue this job."
                )

    # ---- planning ---------------------------------------------------------------------
    def plan(self, spec: JobSpec, states: t.Mapping[str, dict]) -> ResumePlan:
        ck = self.checkpointer(spec)
        # Upstream writes its checkpoints with fabric.save (model_base.py:417) and knows
        # nothing about our index, so without adoption the index is empty for every real job
        # and this would plan `init_from=scratch` next to 274 MB of valid weights.
        ck.adopt_existing()
        before = {r.path for r in ck.read_index()}
        rec = ck.resolve()
        after = {r.path for r in ck.read_index()}
        rolled_back = sorted(before - after)

        parent_ckpt = parent_sha = parent_steps = None
        if spec.parent_job_id:
            parent = states.get(spec.parent_job_id)
            if not parent:
                raise RuntimeError(
                    f"{spec.job_id} needs parent {spec.parent_job_id} to be evaluated first"
                )
            verify_base_competence_receipt(
                parent,
                expected_job_id=spec.parent_job_id,
                model=spec.model,
                seed=spec.seed,
            )
            parent_ckpt = parent.get("final_checkpoint")
            parent_sha = parent.get("final_checkpoint_sha256")
            if not parent_ckpt or not parent_sha:
                raise RuntimeError(f"parent {spec.parent_job_id} recorded no final checkpoint")
            got = sha256_file(parent_ckpt) if pathlib.Path(parent_ckpt).is_file() else None
            if got != parent_sha:
                raise RuntimeError(
                    f"parent checkpoint {parent_ckpt} hash {str(got)[:12]} != recorded "
                    f"{parent_sha[:12]}; the H3 branches would not share a parent"
                )
            parent_steps = parent.get("step")
            if parent_steps is None:
                raise RuntimeError(
                    f"parent {spec.parent_job_id} recorded no step count; without it the "
                    "branch's trainer.train_batches cannot be offset (core_train.py:309,569)"
                )

        return ResumePlan(
            spec=spec,
            fresh=rec is None,
            resume_step=rec.step if rec else 0,
            checkpoint_path=rec.path if rec else None,
            checkpoint_sha256=rec.sha256 if rec else None,
            rolled_back_from=rolled_back[0] if rolled_back else None,
            parent_checkpoint=parent_ckpt,
            parent_checkpoint_sha256=parent_sha,
            parent_steps=int(parent_steps) if parent_steps is not None else None,
        )

    def _terminalize_verified_checkpoint(
        self,
        spec: JobSpec,
        states: dict[str, dict],
        plan: ResumePlan,
        final: t.Any,
        *,
        recovered_before_launch: bool = False,
    ) -> dict:
        """Apply the exact-update and artifact gates to one verified checkpoint.

        This path is shared by a normally returning launcher and by recovery from a runtime
        loss that happened after the exact-target checkpoint became durable but before its
        terminal ledger entry did.  In particular, an exact-target recovery must not launch
        the trainer again: upstream restores the checkpoint's absolute step counter, so one
        more optimizer step would turn a valid 3,000-step result into an invalid 3,001-step
        overrun.
        """
        start = plan.parent_steps or 0
        target = start + spec.train_batches
        updates = final.step - start
        if updates != spec.train_batches:
            recovery_detail = (
                f"Recovered verified checkpoint step {final.step} exceeds exact absolute "
                f"target {target}; refusing to launch or mutate the checkpoint. "
                if recovered_before_launch and final.step > target
                else ""
            )
            entry = self.ledger.append({
                "job_id": spec.job_id, "status": FAILED,
                "step": final.step, "updates": updates,
                "reason": (
                    recovery_detail
                    + f"Checkpoint is at step {final.step} having taken {updates} updates, "
                    f"but the request was exactly {spec.train_batches} (parent/start step "
                    f"{start}). Production applies a source-hash/commit-guarded `step >= "
                    "trainer.train_batches` patch, and adaptation converts its relative "
                    "500-update request to the absolute target parent_step + 500. Any "
                    "under-run or overrun is a FAILED job, never a terminal one."
                ),
                "parent_job_id": spec.parent_job_id,
                "parent_checkpoint_sha256": plan.parent_checkpoint_sha256,
                **self._identity(spec),
            })
            states[spec.job_id] = entry
            return entry

        try:
            recovery = self.recovery_barrier.get(spec.job_id)
            if recovery is not None and (
                    final.path != recovery.get("checkpoint_path") or
                    final.sha256 != recovery.get("checkpoint_sha256") or
                    final.step != int(recovery.get("target_step", -1))):
                raise ValueError(
                    f"{spec.job_id}: recovered checkpoint changed after the atomic barrier"
                )
            training_artifacts = collect_training_artifacts(
                spec,
                authoritative_artifacts=(
                    recovery.get("authoritative_artifacts") if recovery is not None else None
                ),
            )
            recovery_provenance = (
                recovery.get("recovery_provenance") if recovery is not None else None
            )
            summary = write_completion_summary(
                spec,
                step=final.step,
                updates=updates,
                checkpoint_path=final.path,
                checkpoint_sha256=final.sha256,
                training_artifacts=training_artifacts,
                recovery_provenance=recovery_provenance,
            )
            artifacts = dict(training_artifacts)
            artifacts[str(summary.relative_to(pathlib.Path(spec.out_root)))] = sha256_file(summary)
        except (FileNotFoundError, ValueError) as exc:
            entry = self.ledger.append({
                "job_id": spec.job_id, "status": FAILED, "reason": str(exc),
                "step": final.step, "updates": updates,
                "parent_job_id": spec.parent_job_id,
                "parent_checkpoint_sha256": plan.parent_checkpoint_sha256,
                **self._identity(spec),
            })
            states[spec.job_id] = entry
            return entry

        self.checkpointer(spec).finalize()  # clear stale core_train.py recovery pointer
        terminal_status = DONE if spec.final_artifacts else TRAINED
        entry = self.ledger.append({
            "job_id": spec.job_id, "status": terminal_status, "step": final.step,
            # `step` is upstream's offset-carrying counter; `updates` is the quantity
            # PROGRAM.md freezes. The two differ by the parent offset for adaptation.
            "updates": updates,
            "parent_steps": plan.parent_steps,
            "final_checkpoint": final.path, "final_checkpoint_sha256": final.sha256,
            "artifacts": artifacts,
            "parent_job_id": spec.parent_job_id,
            "parent_checkpoint_sha256": plan.parent_checkpoint_sha256,
            "recovered_without_launch": recovered_before_launch,
            **({"recovery_provenance": recovery_provenance}
               if recovery_provenance is not None else {}),
            **self._identity(spec),
        })
        states[spec.job_id] = entry
        recovery_note = " from an already-durable checkpoint" if recovered_before_launch else ""
        self.echo(
            f"[run_matrix] {spec.job_id}: {terminal_status}{recovery_note} at step "
            f"{final.step} ({updates} updates)"
        )
        return entry

    # ---- run ---------------------------------------------------------------------------
    def run_job(self, spec: JobSpec, states: dict[str, dict]) -> dict:
        prior = states.get(spec.job_id)
        self._check_identity(spec, prior)

        if prior and prior.get("status") in TRAINING_TERMINAL:
            ok, reason = verify_artifacts(spec.out_root, prior.get("artifacts", {}))
            if ok and prior.get("final_checkpoint_sha256"):
                p = pathlib.Path(prior["final_checkpoint"])
                ok = p.is_file() and sha256_file(p) == prior["final_checkpoint_sha256"]
                reason = "ok" if ok else "final checkpoint hash mismatch"
            if ok:
                if prior.get("status") == TRAINED and spec.final_artifacts:
                    try:
                        evaluation_artifacts = hash_artifacts(
                            spec.out_root, spec.final_artifacts
                        )
                    except FileNotFoundError:
                        self.echo(
                            f"[run_matrix] {spec.job_id}: TRAINED; required scientific "
                            "evaluation artifacts are not present yet"
                        )
                        return prior
                    promoted = {
                        key: value for key, value in prior.items()
                        if key not in ("seq", "ts", "status")
                    }
                    promoted.update({
                        "job_id": spec.job_id,
                        "status": DONE,
                        "artifacts": dict(prior.get("artifacts", {}), **evaluation_artifacts),
                        "evaluation_artifacts": evaluation_artifacts,
                        "supersedes": prior.get("seq"),
                        **self._identity(spec),
                    })
                    entry = self.ledger.append(promoted)
                    states[spec.job_id] = entry
                    self.echo(
                        f"[run_matrix] {spec.job_id}: promoted TRAINED -> DONE; evaluation "
                        "artifact hashes verify"
                    )
                    return entry
                self.echo(
                    f"[run_matrix] {spec.job_id}: {prior['status']}, hashes verify, skipping"
                )
                return prior
            # Never silently rerun a terminal job: append a superseding entry saying why.
            entry = self.ledger.append({
                "job_id": spec.job_id, "status": STALE,
                "reason": f"{prior['status']} entry failed verification: {reason}",
                "supersedes": prior.get("seq"), **self._identity(spec),
            })
            states[spec.job_id] = entry
            prior = entry

        plan = self.plan(spec, states)
        absolute_target = (plan.parent_steps or 0) + spec.train_batches
        required_recovery = self.recovery_barrier.get(spec.job_id)
        if required_recovery is not None:
            if (not plan.checkpoint_path or plan.resume_step != absolute_target or
                    plan.checkpoint_path != required_recovery.get("checkpoint_path") or
                    plan.checkpoint_sha256 != required_recovery.get("checkpoint_sha256")):
                raise RuntimeError(
                    f"{spec.job_id}: atomic recovery barrier requires the exact target "
                    "checkpoint; no launcher was invoked"
                )
            recovered = self.checkpointer(spec).resolve()
            if recovered is None:
                raise RuntimeError(
                    f"{spec.job_id}: barrier checkpoint disappeared before terminalization"
                )
            result = self._terminalize_verified_checkpoint(
                spec, states, plan, recovered, recovered_before_launch=True
            )
            if result.get("status") not in TRAINING_TERMINAL:
                raise RuntimeError(
                    f"{spec.job_id}: recovery terminalization failed before any launcher"
                )
            return result
        if plan.checkpoint_path and plan.resume_step >= absolute_target:
            # The latest checkpoint has already passed the deep verification performed by
            # `plan()`.  Adopt exact completion without consuming another optimizer step; an
            # over-target recovery is evidence of a protocol violation and fails closed.
            recovered = self.checkpointer(spec).resolve()
            if recovered is None:
                raise RuntimeError(
                    f"{spec.job_id}: verified recovery checkpoint disappeared before adoption"
                )
            return self._terminalize_verified_checkpoint(
                spec, states, plan, recovered, recovered_before_launch=True
            )

        self.ledger.append({
            "job_id": spec.job_id, "status": RUNNING, "step": plan.resume_step,
            "resumed_from": plan.checkpoint_path,
            "resumed_from_sha256": plan.checkpoint_sha256,
            "rolled_back_from": plan.rolled_back_from,
            "parent_job_id": spec.parent_job_id,
            "parent_checkpoint_sha256": plan.parent_checkpoint_sha256,
            **self._identity(spec),
        })

        result = self.launcher(plan)

        ck = self.checkpointer(spec)
        ck.adopt_existing()     # the launched trainer wrote through upstream, not through us
        final = ck.resolve()
        if result.returncode != 0:
            entry = self.ledger.append({
                "job_id": spec.job_id,
                "status": INTERRUPTED if final is not None else FAILED,
                "step": final.step if final else plan.resume_step,
                "reason": result.detail or f"returncode={result.returncode}",
                "parent_job_id": spec.parent_job_id,
                "parent_checkpoint_sha256": plan.parent_checkpoint_sha256,
                **self._identity(spec),
            })
            states[spec.job_id] = entry
            return entry

        if final is None:
            entry = self.ledger.append({
                "job_id": spec.job_id, "status": FAILED,
                "reason": "job exited 0 but left no verified checkpoint",
                **self._identity(spec),
            })
            states[spec.job_id] = entry
            return entry

        # A clean exit is not evidence of completion; the shared terminalization path checks
        # exact updates, real artifacts, and the runner-owned completion receipt.
        return self._terminalize_verified_checkpoint(spec, states, plan, final)

    def run(self, jobs: t.Sequence[JobSpec]) -> dict[str, dict]:
        validate_matrix(jobs)
        states = self.ledger.states()
        by_id = {spec.job_id: spec for spec in jobs}
        unknown = sorted(set(self.recovery_barrier) - set(by_id))
        if unknown:
            raise RuntimeError(f"recovery barrier contains jobs outside the matrix: {unknown}")
        ordered = [by_id[job_id] for job_id in self.recovery_barrier]
        ordered.extend(spec for spec in jobs if spec.job_id not in self.recovery_barrier)
        for spec in ordered:
            self.run_job(spec, states)
        assert_branch_parity(states, jobs)
        return states


def assert_branch_parity(states: t.Mapping[str, dict], jobs: t.Sequence[JobSpec]) -> None:
    """Near, mid and far must record the same parent checkpoint SHA (spec section 9.3).

    If they hang off different parents, both dose-response and near-minus-far erosion confound
    the intervention with the starting point.
    """
    by_parent: dict[str, dict[str, str | None]] = {}
    for spec in jobs:
        if spec.phase != "adapt" or spec.parent_job_id is None:
            continue
        entry = states.get(spec.job_id)
        if not entry or entry.get("status") not in TRAINING_TERMINAL:
            continue
        by_parent.setdefault(spec.parent_job_id, {})[spec.condition or "?"] = entry.get(
            "parent_checkpoint_sha256"
        )
    for parent, arms in by_parent.items():
        if len(arms) < 2:
            continue
        shas = set(arms.values())
        if len(shas) != 1 or None in shas:
            raise RuntimeError(
                f"H3 branches off {parent} do not share a parent checkpoint: {arms}"
            )


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="durable run root, e.g. /content/lurestar")
    ap.add_argument("--ledger", default=str(_REPO / "results" / "run_ledger.json"))
    ap.add_argument("--upstream", default=str(_REPO / "upstream" / "NextLat"))
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS),
                    help="default: all three architecture-matched arms (spec section 8). "
                         "Restricting this to gpt+nextlat drops the competence-matched "
                         "control and demotes the primary contrast; do it only for a "
                         "deliberate partial rerun.")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    ap.add_argument("--phase", choices=["base", "adapt", "all"], default="all")
    ap.add_argument(
        "--adaptation-manifest-dir",
        default=str(_REPO / "manifests" / "adapt"),
        help="complete output of materialize_adaptation_banks.py; required for adapt/all",
    )
    ap.add_argument("--only", nargs="*", default=None, help="explicit job ids")
    ap.add_argument("--base-steps", type=int, default=20000)
    ap.add_argument("--adapt-steps", type=int, default=500)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--precision", default="bf16-mixed")
    ap.add_argument(
        "--competence-evaluator", default=DEFAULT_COMPETENCE_EVALUATOR,
        help="exact evaluator source frozen into each base job identity before training",
    )
    ap.add_argument(
        "--competence-dataset", default=None,
        help="held-out G(5,5) file frozen before training; default is ROOT/data/... when staged",
    )
    ap.add_argument(
        "--competence-manifest", action="append", default=None,
        help="manifest binding the held-out dataset; repeatable",
    )
    ap.add_argument("--strategy", default="ddp",
                    help="ddp even on one device: it gives a DistributedSampler whose order is "
                         "reproducible across resumes (docs/UPSTREAM_REPORT.md section 3.5 item 5)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--print-plan", action="store_true")
    ap.add_argument("--bucket", default=os.environ.get("LURESTAR_BUCKET"))
    ap.add_argument("--gcs-prefix", default="lurestar")
    ap.add_argument("--retry-sync", action="store_true",
                    help="drain the NEEDS_SYNC queue left by a failed GCS push and exit")
    a = ap.parse_args(argv)

    if a.retry_sync:
        if not a.bucket:
            print("--retry-sync needs --bucket or $LURESTAR_BUCKET", file=sys.stderr)
            return 2
        failed = 0
        for spec in build_matrix(a.root, models=a.models, seeds=a.seeds):
            sync = DurableSync(a.bucket, a.gcs_prefix, spec.job_id, logger=print)
            for res in sync.retry_pending():
                print(("  ok   " if res.ok else "  FAIL ") + res.remote)
                failed += 0 if res.ok else 1
        return 1 if failed else 0

    manifests = dict(DEFAULT_MANIFESTS)
    # D40 exhausted the one prospectively permitted support expansion and permanently retired
    # Lure-Star H3.  Base training remains part of H1/H2, but it must bind that exact exclusion;
    # adaptation planning is now a protocol violation rather than a missing-input condition.
    if a.phase != "base":
        print(
            "[run_matrix] REFUSED: D40 permanently excluded Lure-Star H3; only --phase base "
            "is permitted by the reduced confirmatory program",
            file=sys.stderr,
        )
        return 2
    try:
        h3_block_manifests = verified_h3_permanent_block()
    except RuntimeError as exc:
        print(
            f"[run_matrix] REFUSED base matrix: D40 permanent H3 block is invalid: {exc}",
            file=sys.stderr,
        )
        return 2
    manifests["base"] = tuple(DEFAULT_MANIFESTS["base"]) + h3_block_manifests

    jobs = build_matrix(
        a.root, models=a.models, seeds=a.seeds,
        base_steps=a.base_steps, adapt_steps=a.adapt_steps,
        manifests=manifests,
        competence_evaluator=a.competence_evaluator,
        competence_dataset=a.competence_dataset,
        competence_manifests=(a.competence_manifest or DEFAULT_COMPETENCE_MANIFESTS),
    )
    if a.phase != "all":
        jobs = [j for j in jobs if j.phase == a.phase]
    if a.only:
        wanted = set(a.only)
        jobs = [j for j in jobs if j.job_id in wanted]

    if a.print_plan:
        # A printed adaptation plan is still a plan.  Refuse it before emitting launchable
        # branch paths unless each parent has an immutable, evaluator-bound competence receipt.
        states = Ledger(a.ledger).states()
        for spec in jobs:
            if spec.phase == "adapt":
                parent = states.get(spec.parent_job_id or "")
                if not parent:
                    print(
                        f"[run_matrix] REFUSED adaptation plan: {spec.job_id} has no evaluated "
                        f"parent {spec.parent_job_id}",
                        file=sys.stderr,
                    )
                    return 2
                try:
                    verify_base_competence_receipt(
                        parent,
                        expected_job_id=spec.parent_job_id or "",
                        model=spec.model,
                        seed=spec.seed,
                    )
                except RuntimeError as exc:
                    print(f"[run_matrix] REFUSED adaptation plan: {exc}", file=sys.stderr)
                    return 2
        print(json.dumps([j.to_dict() for j in jobs], indent=2))
        return 0

    sync = DurableSync(a.bucket, a.gcs_prefix, "matrix", logger=print) if a.bucket else None
    launcher = FabricLauncher(
        a.upstream, devices=a.devices, precision=a.precision,
        strategy=a.strategy, dry_run=a.dry_run,
    )
    runner = MatrixRunner(Ledger(a.ledger), launcher, sync=sync)
    states = runner.run(jobs)
    not_terminal = [
        j.job_id for j in jobs
        if states.get(j.job_id, {}).get("status") not in TRAINING_TERMINAL
    ]
    if not_terminal:
        print(f"[run_matrix] not TRAINED/DONE: {not_terminal}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
