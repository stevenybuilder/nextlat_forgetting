#!/usr/bin/env python
"""Materialize every training configuration by COPYING the official pinned YAML and
applying an explicit, auditable override set.

Spec section 8, verbatim: "At the pinned repository commit, copy the official Path-Star
G(5,5) GPT and NextLat YAML configurations. Do not reconstruct an approximate configuration
from this document."  docs/RUNLOG.md records what happens when that rule is broken: a
hand-written smoke config dropped `data.test_generalization` and train.py died with an
omegaconf ConfigAttributeError at the first validation.

This generator therefore never *writes* a config. It *edits* a parsed copy of the official
file, and enforces five invariants before anything reaches disk:

  I1  No upstream key may vanish.       Every dotted key of the source YAML must survive
                                        into the output, unless it appears in that config's
                                        explicit DROPS list with a reason.
  I2  No silent change.                 Every key whose merged value differs from the
                                        merged source must appear in that config's
                                        OVERRIDES list (bidirectional: a declared override
                                        that changes nothing is also an error).
  I3  Hoists are value-preserving.      A key lifted out of the `sweep:` block or out of
                                        defaults.yaml into an explicit block must carry the
                                        exact value it already resolved to.
  I4  Frozen keys stay frozen.          The spec-section-8 / PROGRAM.md frozen surface may
                                        only move where a config family carries a written
                                        spec authority for it (H3 adaptation, HMM).
  I5  Pools are bound to families.      A base run reads the frozen 200,000/20,000 corpus
                                        and nothing else; an adaptation branch reads only
                                        its OWN B_near / B_far bank, out of the immutable
                                        adaptation directory, and may never name a reserved
                                        evaluation pool. Upstream cannot tell these files
                                        apart -- data/stargraph.py:187-190 parses only the
                                        `5_5` in the name -- so a swapped or leaked bank is
                                        invisible to every other check in this repository.

Usage:
    .venv/bin/python scripts/materialize_configs.py            # write configs/
    .venv/bin/python scripts/materialize_configs.py --check    # fail if on-disk differs
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config_lib import (  # noqa: E402
    CONFIGS_DIR,
    DEFAULTS_YAML,
    OFFICIAL_BST_5_5,
    OFFICIAL_GPT_5_5,
    OFFICIAL_NEXTLAT_5_5,
    UPSTREAM,
    UPSTREAM_COMMIT,
    deep_merge,
    del_dotted,
    flatten,
    get_dotted,
    has_dotted,
    load_yaml,
    missing_keys,
    set_dotted,
    sha256_file,
)

# --------------------------------------------------------------------------------------
# Deployment roots. Every path in a materialized config is ABSOLUTE: the upstream resume
# pointer stores the checkpoint path exactly as written (core_train.py:944-948), so a
# relative out_dir makes resume depend on the launching CWD.
# --------------------------------------------------------------------------------------
ROOT = "/content/lurestar"
DATA = f"{ROOT}/data"
MANIFESTS = f"{ROOT}/manifests"
RUNS = f"{ROOT}/runs"

# data/stargraph.py:187-190 parses the arm count and the answer length out of the FILENAME
# (`path.split("_")[1]` and `[2]`) and asserts the second against the measured target
# length. Every stargraph file we point at must therefore be named `graph_5_5_<tag>_<n>.txt`
# AND live under a directory path that contains no underscore.
CORPUS_TRAIN = f"{DATA}/stargraph/graph_5_5_sample_200000.txt"
CORPUS_TEST = f"{DATA}/stargraph/graph_5_5_test_20000.txt"
B_NEAR = f"{MANIFESTS}/adapt/graph_5_5_bnear_5000.txt"
B_FAR = f"{MANIFESTS}/adapt/graph_5_5_bfar_5000.txt"
B_NEAR_VAL = f"{MANIFESTS}/adapt/graph_5_5_bnearval_2000.txt"
B_FAR_VAL = f"{MANIFESTS}/adapt/graph_5_5_bfarval_2000.txt"

PREREGISTERED_SEEDS = [1234, 1235, 1236]
DEFAULT_SEED = PREREGISTERED_SEEDS[0]

# --------------------------------------------------------------------------------------
# I5: pool identity.
#
# `data/stargraph.py:187-190` reads only `split("_")[1]` and `[2]` out of a data path, i.e.
# `5` and `5`. Every stargraph bank in this project therefore looks identical to upstream,
# and nothing downstream can tell B_near from B_far or either from the base corpus.
#
# Two mutations that the rest of this generator, `--check`, and the whole config suite were
# blind to before this invariant existed:
#
#   * swapping the near and far banks. Spec sec.6's primary outcome is
#     `erosion_near - erosion_far`; a swap negates it exactly, and every artifact downstream
#     still carries correct-looking provenance.
#   * pointing an adaptation bank at `E_lure` (or at the base corpus). Spec sec.5: "No
#     E_lure graph or lure may enter base or adaptation training."
#
# So the branch name, the bank filename and the directory a bank may live in are bound to
# each other here, and the binding is asserted again on the emitted files in
# tests/test_configs.py so it does not depend on this file being right.
# --------------------------------------------------------------------------------------
ADAPT_BANK_DIR = f"{MANIFESTS}/adapt"
ADAPT_BANK_TAGS = {"near": "bnear", "far": "bfar"}
# Pools that exist only to be evaluated on. A training or validation path that names one is
# a leakage bug, not a typo.
RESERVED_POOL_TOKENS = ("elure", "e_lure", "apair", "a_pair", "stimuli")
BASE_CORPUS_PATHS = (CORPUS_TRAIN, CORPUS_TEST)
STARGRAPH_PATH_KEYS = ("data.stargraph_train_data_path", "data.stargraph_test_data_path")


def _check_pool_identity(name: str, plan: Dict[str, Any], cfg: Dict[str, Any]) -> None:
    """I5. Raise unless every stargraph path in `cfg` names the pool its family may use."""
    family = plan["family"]
    if family == "lurestar":
        for key, expected in zip(STARGRAPH_PATH_KEYS, BASE_CORPUS_PATHS):
            got = get_dotted(cfg, key)
            if got != expected:
                raise AssertionError(
                    f"{name}: {key} is {got!r}, not the frozen base corpus {expected!r}. "
                    f"Spec sec.8 fixes the 200,000/20,000 graph corpus for every base run."
                )
        return
    if family != "adapt":
        return

    branch = plan["branch"]
    own = ADAPT_BANK_TAGS[branch]
    other = ADAPT_BANK_TAGS["far" if branch == "near" else "near"]
    for key in STARGRAPH_PATH_KEYS:
        path = get_dotted(cfg, key)
        base = os.path.basename(path)
        rel = os.path.relpath(path, ROOT)
        if path in BASE_CORPUS_PATHS:
            raise AssertionError(
                f"{name}: {key} points at the base training corpus {path!r}; the adaptation "
                f"branches must read their own immutable B_{branch} bank."
            )
        for token in RESERVED_POOL_TOKENS:
            if token in rel.lower():
                raise AssertionError(
                    f"{name}: {key} = {path!r} names the reserved evaluation pool "
                    f"{token!r}. Spec sec.5: no E_lure graph or lure may enter base or "
                    f"adaptation training."
                )
        if os.path.dirname(path) != ADAPT_BANK_DIR:
            raise AssertionError(
                f"{name}: {key} = {path!r} is not under the immutable adaptation bank "
                f"directory {ADAPT_BANK_DIR!r}"
            )
        if own not in base:
            raise AssertionError(
                f"{name}: {key} = {base!r} does not carry its own branch tag {own!r}. "
                f"Spec sec.6's primary outcome is erosion_near - erosion_far, so a bank "
                f"that is not bound to its branch silently negates the result."
            )
        if other in base:
            raise AssertionError(
                f"{name}: {key} = {base!r} carries the OPPOSITE branch tag {other!r}; the "
                f"near and far banks are swapped."
            )

# Spec section 8 / PROGRAM.md "Frozen surface". A move here is a scientific change, not a
# configuration change, and needs a written spec authority recorded in EXEMPT_FROZEN.
FROZEN_KEYS = [
    "use_bst",
    "use_nextlat",
    "model.n_layer",
    "model.n_head",
    "model.n_embd",
    "model.dropout",
    "model.bias",
    "model.gpt_mode",
    "model.mtp_horizon",
    "model.proj_factor",
    "model.lambda_mse",
    "model.lambda_kl",
    "model.lambda_ce",
    # BST objective surface. Only `bst_pair_minimum_gap` is written in the official BST
    # YAML; the other three resolve out of defaults.yaml:98-104 and are hoisted so that a
    # defaults change cannot move them silently. All four are read at core_train.py:75-84.
    "model.bst_pair_minimum_gap",
    "model.bst_pair_maximum_gap",
    "model.bst_pair_subsample_rate",
    "model.bst_single_gap_prediction_mode",
    # Chunk size for BST's text head. model_bst.py:600-601 folds the resulting chunk count
    # into `texthead_loss_div`, so this is a loss-scaling key, not a memory knob.
    "data.pair_batch_size",
    "optimizer.optimizer_type",
    "optimizer.learning_rate",
    "optimizer.weight_decay",
    "optimizer.beta1",
    "optimizer.beta2",
    "optimizer.grad_clip",
    "lr_scheduler.schedule",
    "lr_scheduler.warmup_iters",
    "lr_scheduler.warmdown_iters",
    "data.effective_batch_size",
    "data.gradient_accum_steps",
    "data.stargraph_max_nodes",
    "trainer.train_batches",
]


@dataclass(frozen=True)
class Ov:
    """One override. `category` and `authority` are what make the set auditable."""

    key: str
    value: Any
    category: str
    authority: str
    reason: str
    # A declared override that happens to equal the upstream-resolved value. Only `seed`
    # qualifies: it is 1234 in defaults.yaml AND the first preregistered seed, so at seed
    # 1234 the file records a preregistration decision that changes no byte, while at 1235
    # and 1236 the same declaration does change the run.
    inert_ok: bool = False


@dataclass(frozen=True)
class Hoist:
    """A key lifted into an explicit block WITHOUT changing its resolved value."""

    key: str
    reason: str


@dataclass(frozen=True)
class Drop:
    key: str
    reason: str


# --------------------------------------------------------------------------------------
# Override sets, one per deliverable.
# --------------------------------------------------------------------------------------

A_SPEC8 = "spec sec.8 'Configuration authority' permitted-change list"
A_SPEC9 = "spec sec.9 'Colab interruption and recovery contract'"
A_SPEC6 = "spec sec.6 H3 'similarity-dependent interference'"
A_SPEC12 = "spec sec.12 'Required experiment B: HMM belief geometry'"
A_RUNLOG = "docs/RUNLOG.md session 1 (established correction)"
A_EXEC = "execution environment necessity (no scientific surface)"


def _common_lurestar(model: str, seed: int) -> List[Ov]:
    out_dir = f"{RUNS}/{model}/seed{seed}/base"
    return [
        Ov("seed", seed, "SEED", A_SPEC8,
           "One of the three preregistered confirmatory seeds. Stated explicitly instead "
           "of via `sweep:`, whose experiment-directory name is built by iterating a "
           "Python set (train.py:280,322) and therefore varies with PYTHONHASHSEED.",
           inert_ok=True),
        Ov("trainer.compile", False, "CORRECTION", A_RUNLOG + " + spec sec.8 'Set compile:false'",
           "The shipped YAML sets true; upstream README.md:117-122 reports inconsistent "
           "Path-Star results under torch.compile. Also keeps submodule paths free of the "
           "_orig_mod level so the hidden-state hook resolves."),
        Ov("trainer.out_dir", out_dir, "OUTPUT", A_SPEC9,
           "Absolute and unique per job. latest_ckpt/recovery_ckpt pointers live at out_dir "
           "(core_train.py:945,971), so a shared out_dir lets one job's pointer capture "
           "another's resume."),
        Ov("trainer.experiment_name", f"{model}-seed{seed}-base", "OUTPUT", A_SPEC9,
           "Deterministic job id. Contains the substring 'seed', which suppresses the "
           "automatic '-seed{n}' suffix at train.py:96-97."),
        Ov("trainer.save_recovery_checkpoint", 250, "CKPT", A_SPEC9,
           "defaults.yaml:24 ships -1 (disabled). 250 steps is the spec's initial recovery "
           "cadence; the profiling gate reprices it against the 10-minute rule."),
        Ov("trainer.log_to_wandb", False, "EXEC", A_EXEC,
           "defaults.yaml:34 is true and the shipped 5_5 YAMLs never override it, so a "
           "naive run tries to wandb.init on the first log. CSV logging (log_to_file) "
           "remains the record of every metric."),
        Ov("data.stargraph_train_data_path", CORPUS_TRAIN, "MANIFEST", A_SPEC8,
           "Absolute path to the frozen 200,000-graph corpus. Filename keeps the "
           "graph_5_5_..._200000 form that data/stargraph.py:187-190 parses and asserts."),
        Ov("data.stargraph_test_data_path", CORPUS_TEST, "MANIFEST", A_SPEC8,
           "Absolute path to the frozen 20,000-graph held-out corpus."),
    ]


def _provenance(source_path: str, note: str) -> List[Ov]:
    rel = os.path.relpath(source_path, UPSTREAM)
    return [
        Ov("provenance", {
            "upstream_repo": "JaydenTeoh/NextLat",
            "upstream_commit": UPSTREAM_COMMIT,
            "source_config": rel,
            "source_config_sha256": sha256_file(source_path),
            "generator": "scripts/materialize_configs.py",
            "note": note,
        }, "PROVENANCE", A_SPEC9,
           "Additive block, read by nothing. It lands in materialized_config.yaml with "
           "every run (train.py:192-194), which is what spec sec.9 requires to be persisted "
           "before training."),
        Ov("preregistration", {
            "confirmatory_seeds": list(PREREGISTERED_SEEDS),
            "primary_distance": "centered cosine",
            "robustness_distance": "whitened Euclidean",
        }, "PROVENANCE", A_SPEC8,
           "Additive block recording the preregistration inside the artifact that the run "
           "archives. Read by nothing."),
    ]


def build_gpt_lurestar(seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    return {
        "source": OFFICIAL_GPT_5_5,
        "overrides": _common_lurestar("gpt", seed) + _provenance(
            OFFICIAL_GPT_5_5,
            "Condition 1 of spec sec.8: the official repository GPT with standard "
            "next-token training on G(5,5).",
        ),
        "hoists": [
            Hoist("trainer.save_last_checkpoint",
                  "spec sec.8 lists it explicitly; defaults.yaml:18 already true. Restated "
                  "so the frozen base checkpoint cannot be lost to a defaults change."),
            Hoist("trainer.save_best_checkpoint",
                  "spec sec.8 lists it explicitly; defaults.yaml:20 already true."),
            Hoist("trainer.log_to_file",
                  "defaults.yaml:33 already true. The CSVLogger output is the only metric "
                  "record once W&B is off."),
            Hoist("trainer.init_from",
                  "already 'scratch' in the shipped YAML; restated next to out_dir because "
                  "the two together decide whether a relaunch resumes or restarts."),
            Hoist("model.compute_hidden_state_rank",
                  "defaults.yaml:111 false. Kept false: spec sec.2 rules out effective rank "
                  "as a standalone contribution and the in-training SVD costs throughput. "
                  "Hidden states are captured offline instead (see hoist note in docs)."),
        ],
        "drops": [],
        "family": "lurestar",
    }


def build_nextlat_lurestar(seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    cfg = build_gpt_lurestar(seed)
    overrides = _common_lurestar("nextlat", seed) + _provenance(
        OFFICIAL_NEXTLAT_5_5,
        "Condition 2 of spec sec.8: the architecture-matched transformer trained with the "
        "official NextLat objective on G(5,5).",
    )
    return {
        "source": OFFICIAL_NEXTLAT_5_5,
        "overrides": overrides,
        "hoists": cfg["hoists"] + [
            Hoist("model.proj_factor",
                  "THE trap. proj_factor: 0.5 exists ONLY inside the sweep block "
                  "(nextlat_stargraph_5_5.yaml:61). Deleting the sweep to run one seed "
                  "silently falls back to defaults.yaml:118 proj_factor: 1.0, which builds "
                  "a dynamics MLP of hidden width 768 instead of the paper's 384."),
            Hoist("model.lambda_ce",
                  "defaults.yaml:116 is 0.0 and the paper's Path-Star setting is 0.0. "
                  "Restated because it is a loss coefficient on the frozen surface."),
        ],
        "drops": [],
        "family": "lurestar",
    }


def build_bst_lurestar(seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Condition 3 of spec sec.8: the Belief State Transformer competence-matched control.

    The official BST G(5,5) YAML differs from the official GPT G(5,5) YAML by exactly two
    written keys -- `use_bst: true` and `model.bst_pair_minimum_gap: 2` -- plus the absence
    of `use_nextlat` and a different `trainer.experiment_name`. It is COPIED, like the other
    two arms; nothing here is reconstructed.

    Four BST-only keys the trainer reads are NOT written in that YAML and resolve out of
    `defaults.yaml`, which is the same shape of hazard as the NextLat `proj_factor` trap
    (docs/FOUNDATIONS.md D-07): a scientifically relevant value reachable only through a
    fallback. `core_train.py:75-84` puts three of them into `model_args`, and
    `core_train.py:38-45` asserts a fourth before the model is even built. They are hoisted
    at their resolved values and pinned in FROZEN_KEYS.
    """
    cfg = build_gpt_lurestar(seed)
    return {
        "source": OFFICIAL_BST_5_5,
        "overrides": _common_lurestar("bst", seed) + _provenance(
            OFFICIAL_BST_5_5,
            "Condition 3 of spec sec.8: the architecture-matched Belief State Transformer, "
            "the competence-matched control. The paper's Figure 6 puts BST at ~99.9% on "
            "G(5,5) against GPT at ~18.6% (1/d chance), so BST solves Path-Star WITHOUT a "
            "latent-transition objective and a NextLat-vs-BST PSI gap is attributable to "
            "the objective rather than to task success. See "
            "docs/DECISION_D20_competence_gate.md.",
        ),
        "hoists": cfg["hoists"] + [
            Hoist("use_nextlat",
                  "defaults.yaml:3 false. The BST YAML is the only 5_5 config that omits "
                  "this flag; gpt_stargraph_5_5.yaml:2 writes it. Restated so the "
                  "core_train.py:38-58 dispatch chain (use_bst -> use_nextlat -> "
                  "use_mtp_*) is fully explicit in the file."),
            Hoist("model.bst_pair_maximum_gap",
                  "THE BST analogue of the proj_factor trap. Read at core_train.py:81, "
                  "written in NO stargraph YAML, resolved from defaults.yaml:99 to -1, "
                  "which model_bst.py:377-378 turns into `max_gap = document_len`, i.e. "
                  "train on every gap. Pinned so a defaults edit cannot silently truncate "
                  "the pair set."),
            Hoist("model.bst_pair_subsample_rate",
                  "Read at core_train.py:82, written in NO stargraph YAML, resolved from "
                  "defaults.yaml:102 to 1.0, which model_bst.py:478-483 reads as 'keep all "
                  "valid pairs' (the subsample branch is skipped entirely). Anything below "
                  "1.0 makes the training set stochastic per step."),
            Hoist("model.bst_single_gap_prediction_mode",
                  "Read at core_train.py:80 and ASSERTED at core_train.py:41-44 to be one "
                  "of ['next_token', 'eos']; resolved from defaults.yaml:104 to 'eos'. "
                  "Inert at bst_pair_minimum_gap: 2, because model_bst.py:584-588 only "
                  "rewrites targets for pairs of gap exactly 1 and a minimum gap of 2 "
                  "produces none -- but a missing value would abort the run before the "
                  "first step, and a minimum-gap change would make it live."),
            Hoist("data.pair_batch_size",
                  "Read at core_train.py:495 and :635 and passed to compute_loss for every "
                  "arm; GPT and NextLat ignore it ('Extra arguments ignored for "
                  "compatibility with BST', model_gpt.py:342, model_nextlat.py:418), BST "
                  "does not. Written in NO stargraph YAML, resolved from defaults.yaml:67 "
                  "to 32768. It is a LOSS-SCALING key, not a memory knob: model_bst.py:600 "
                  "computes pair_accum_steps = ceil(n_pairs / pair_batch_size) and :601 "
                  "folds it into texthead_loss_div, so a value below the per-sequence pair "
                  "count turns the text-head loss into a mean of chunk means."),
        ],
        "drops": [],
        "family": "lurestar",
    }


def _adapt(branch: str, bank: str, bank_val: str, seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    out_dir = f"{RUNS}/nextlat/seed{seed}/adapt-{branch}"
    return {
        "source": OFFICIAL_NEXTLAT_5_5,
        "overrides": [
            Ov("seed", seed, "SEED", A_SPEC8, "Preregistered confirmatory seed.",
               inert_ok=True),
            Ov("trainer.compile", False, "CORRECTION", A_RUNLOG, "As for the base runs."),
            Ov("trainer.out_dir", out_dir, "OUTPUT", A_SPEC9,
               "Separate output root per branch. Spec sec.9: 'Give every base/near/far job "
               "a separate output root; the official resume pointer lives at the "
               "output-root level and must never cross branches.'"),
            Ov("trainer.experiment_name", f"nextlat-seed{seed}-adapt-{branch}",
               "OUTPUT", A_SPEC9, "Deterministic job id, contains 'seed'."),
            Ov("trainer.init_from", "resume", "CKPT", A_SPEC9,
               "The branch is started by pre-seeding {out_dir}/latest_ckpt with the "
               "step-rebased frozen parent checkpoint, so init_from: resume both starts and "
               "restarts the branch through one upstream code path "
               "(core_train.py:139-163). PRECONDITION: the pointer must exist before the "
               "first launch, or upstream falls through to a scratch model "
               "(core_train.py:164-168)."),
            Ov("trainer.train_batches", 500, "H3", A_SPEC6 + " + spec sec.8 'Start H3 with "
               "5,000 adaptation items and 500 updates'",
               "Adaptation length. Under core_train.py:564-571 this executes 501 optimizer "
               "updates from a step-0 parent - the same inclusive-bound convention by which "
               "the shipped train_batches: 20000 executes 20,001. Identical for near, far, "
               "GPT and NextLat, so the near-minus-far contrast is unaffected."),
            Ov("trainer.val_interval", 100, "CKPT", A_SPEC6,
               "Five in-run acquisition measurements over a 500-step branch; val_interval "
               "1000 would produce none."),
            Ov("trainer.test_interval", 100, "CKPT", A_SPEC6,
               "Exact-path accuracy on the branch validation set at the same cadence "
               "(core_train.py:671 gates the accuracy eval on this key)."),
            Ov("trainer.save_recovery_checkpoint", 100, "CKPT", A_SPEC9,
               "Recovery cadence scaled to a 500-step job; 250 would give two points."),
            Ov("trainer.log_to_wandb", False, "EXEC", A_EXEC, "As for the base runs."),
            Ov("model.lambda_mse", 0.0, "H3", A_SPEC6,
               "Spec sec.6: 'Use full-parameter next-token-only adaptation for both GPT and "
               "NextLat. Set lambda_mse=0 and lambda_kl=0 during the primary NextLat "
               "adaptation branch.' With lambda_mse = lambda_kl = lambda_ce = 0 the total "
               "loss at model_nextlat.py:488-497 reduces exactly to ntp_loss."),
            Ov("model.lambda_kl", 0.0, "H3", A_SPEC6, "See lambda_mse."),
            Ov("data.stargraph_train_data_path", bank, "MANIFEST", A_SPEC6,
               f"The immutable B_{branch} adaptation item bank (5,000 items). This is the "
               "ONLY scientific difference between the near and far branches."),
            Ov("data.stargraph_test_data_path", bank_val, "MANIFEST", A_SPEC6,
               f"Independent B_{branch} validation set; spec sec.6 requires acquisition on "
               "'independent near/far validation sets'."),
        ] + _provenance(
            OFFICIAL_NEXTLAT_5_5,
            f"Spec sec.6 H3 adaptation, {branch} branch. Derived from the NextLat G(5,5) "
            "YAML because its key set is a superset of the GPT G(5,5) YAML's, so the SAME "
            "file drives the GPT branch when launched with the dotlist override "
            "`use_nextlat=false`.",
        ),
        "hoists": [
            Hoist("use_bst", "defaults.yaml:2 false. Restated so this file is a complete "
                             "config for the GPT branch too (use_nextlat=false at launch)."),
            Hoist("trainer.save_last_checkpoint", "defaults.yaml:18 true."),
            Hoist("trainer.save_best_checkpoint", "defaults.yaml:20 true."),
            Hoist("trainer.log_to_file", "defaults.yaml:33 true."),
            Hoist("model.proj_factor",
                  "sweep-only upstream; must be pinned or the dynamics MLP silently widens "
                  "to 768. The parent checkpoint was trained at 0.5, so a mismatch would "
                  "also make the state_dict load fail."),
            Hoist("model.lambda_ce", "defaults.yaml:116 is 0.0; required to be 0 for the "
                                     "loss to reduce to next-token-only."),
        ],
        "drops": [],
        "family": "adapt",
        "branch": branch,
    }


def build_adapt_near(seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    return _adapt("near", B_NEAR, B_NEAR_VAL, seed)


def build_adapt_far(seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    return _adapt("far", B_FAR, B_FAR_VAL, seed)


HMM_DATA = {
    "hmm_train_data_path": f"{DATA}/hmm/hmm-train-100000-len32.npz",
    "hmm_val_data_path": f"{DATA}/hmm/hmm-val-10000-len32.npz",
    "hmm_generalization_data_path": f"{DATA}/hmm/hmm-gen-10000-len64.npz",
    "hmm_matrices_path": f"{MANIFESTS}/hmm-matrices.json",
    "hmm_num_states": 4,
    "hmm_num_observations": 4,
    "train_sequences": 100000,
    "val_sequences": 10000,
    "generalization_sequences": 10000,
    "sequence_length": 32,
    "generalization_sequence_length": 64,
}

HMM_DROPS = [
    Drop("data.stargraph_max_nodes",
         "read only by StarGraphDataModule (data/stargraph.py:175); defaults.yaml:81 still "
         "resolves it to 50 after merge, so nothing can raise."),
    Drop("data.stargraph_train_data_path",
         "read only at data/stargraph.py:179,187-190 under dataset == 'stargraph'."),
    Drop("data.stargraph_test_data_path",
         "read only at data/stargraph.py:183 under dataset == 'stargraph'."),
    Drop("data.stargraph_generalization_data_path",
         "read only at data/stargraph.py:203 under data.test_generalization, and the "
         "generalization dataloader is built only for stargraph/countdown/manhattan "
         "(core_train.py:344-349). test_generalization stays false here."),
]


def _hmm(model: str, source: str, seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    out_dir = f"{RUNS}/hmm/{model}/seed{seed}/base"
    ovs = [
        Ov("seed", seed, "SEED", A_SPEC12, "Same three preregistered seeds as Lure-Star.",
           inert_ok=True),
        Ov("trainer.compile", False, "HMM", A_SPEC12 + " (compile: false)", "Spec sec.12 config block."),
        Ov("trainer.out_dir", out_dir, "OUTPUT", A_SPEC9, "Unique absolute output root."),
        Ov("trainer.experiment_name", f"{model}-seed{seed}-hmm", "OUTPUT", A_SPEC9,
           "Deterministic job id, contains 'seed'."),
        Ov("trainer.train_batches", 3000, "HMM", A_SPEC12, "Spec sec.12 config block."),
        Ov("trainer.val_interval", 300, "HMM", A_SPEC12, "Spec sec.12 config block."),
        Ov("trainer.test_interval", 300, "HMM", A_SPEC12,
           "Matched to val_interval. core_train.py:671 reads it unconditionally, so it must "
           "be present even though no accuracy branch fires for a non-stargraph dataset."),
        Ov("trainer.save_recovery_checkpoint", 250, "CKPT", A_SPEC9, "Recovery cadence."),
        Ov("trainer.log_to_wandb", False, "EXEC", A_EXEC, "As for Lure-Star."),
        Ov("trainer.wandb_project", "hmm_belief", "OUTPUT", A_SPEC9, "Names the second task."),
        Ov("trainer.wandb_tags", ["hmm", "4state4obs"], "OUTPUT", A_SPEC9, "Names the second task."),
        Ov("data.dataset", "hmm_belief", "HMM", A_SPEC12,
           "Spec sec.12 config block. REQUIRES an hmm_belief datamodule registered in the "
           "DATAMODULES dict at train.py:34-42; train.py:176-178 asserts membership. That "
           "registration is a one-line addition applied to the runtime working copy and "
           "recorded as an uncommitted diff (spec sec.9); upstream/ itself is never edited."),
        Ov("data.effective_batch_size", 256, "HMM", A_SPEC12, "Spec sec.12 config block."),
        Ov("model.n_layer", 4, "HMM", A_SPEC12, "Spec sec.12 config block."),
        Ov("model.n_head", 4, "HMM", A_SPEC12, "Spec sec.12 config block."),
        Ov("model.n_embd", 128, "HMM", A_SPEC12, "Spec sec.12 config block."),
        Ov("data.hmm", dict(HMM_DATA), "HMM", A_SPEC12,
           "Dataset sizes and immutable manifest paths for the 4-state/4-observation HMM: "
           "100,000 train and 10,000 validation sequences of length 32 plus 10,000 "
           "length-64 generalization sequences, and the frozen transition/emission "
           "matrices. Namespaced under data.hmm so it cannot collide with an upstream key."),
    ] + _provenance(
        source,
        "Spec sec.12 required experiment B. Copied from the official Path-Star G(5,5) YAML "
        "so that every key the generic trainer reads unconditionally "
        "(trainer.test_interval, trainer.val_printsamples, data.test_generalization, ...) "
        "is present; only the keys spec sec.12 names are changed.",
    )
    hoists = [
        Hoist("trainer.save_last_checkpoint", "defaults.yaml:18 true."),
        Hoist("trainer.save_best_checkpoint", "defaults.yaml:20 true."),
        Hoist("trainer.log_to_file", "defaults.yaml:33 true."),
        Hoist("trainer.init_from", "already 'scratch'; restated next to out_dir."),
        Hoist("model.compute_hidden_state_rank", "defaults.yaml:111 false."),
    ]
    if model == "nextlat":
        ovs += [
            Ov("model.mtp_horizon", 1, "HMM", A_SPEC12,
               "Spec sec.12: d=1 with Smooth L1, because one-step transition consistency "
               "suffices for the belief-state result."),
            Ov("model.lambda_kl", 0.0, "HMM", A_SPEC12, "Spec sec.12 config block."),
        ]
        hoists += [
            Hoist("model.proj_factor", "sweep-only upstream; spec sec.12 also specifies 0.5, "
                                       "and the two agree, so this is a pin not a change."),
            Hoist("model.lambda_mse", "spec sec.12 specifies 1.0, which is already the "
                                      "value in nextlat_stargraph_5_5.yaml:43."),
            Hoist("model.lambda_ce", "defaults.yaml:116 is 0.0."),
        ]
    else:
        hoists += [Hoist("use_bst", "defaults.yaml:2 false; already explicit upstream.")]
    return {
        "source": source,
        "overrides": ovs,
        "hoists": hoists,
        "drops": list(HMM_DROPS),
        "family": "hmm",
    }


def build_gpt_hmm(seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    return _hmm("gpt", OFFICIAL_GPT_5_5, seed)


def build_nextlat_hmm(seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    return _hmm("nextlat", OFFICIAL_NEXTLAT_5_5, seed)


BUILDERS = {
    "gpt_lurestar.yaml": build_gpt_lurestar,
    "nextlat_lurestar.yaml": build_nextlat_lurestar,
    "bst_lurestar.yaml": build_bst_lurestar,
    "adapt_near.yaml": build_adapt_near,
    "adapt_far.yaml": build_adapt_far,
    "gpt_hmm.yaml": build_gpt_hmm,
    "nextlat_hmm.yaml": build_nextlat_hmm,
}

# Frozen-surface exemptions. A frozen key may only differ from its upstream-resolved value
# when the spec itself moves it, and only in the config families listed here.
EXEMPT_FROZEN = {
    ("adapt", "model.lambda_mse"): A_SPEC6,
    ("adapt", "model.lambda_kl"): A_SPEC6,
    ("adapt", "trainer.train_batches"): A_SPEC6,
    ("hmm", "model.n_layer"): A_SPEC12,
    ("hmm", "model.n_head"): A_SPEC12,
    ("hmm", "model.n_embd"): A_SPEC12,
    ("hmm", "model.mtp_horizon"): A_SPEC12,
    ("hmm", "model.lambda_kl"): A_SPEC12,
    ("hmm", "data.effective_batch_size"): A_SPEC12,
    ("hmm", "trainer.train_batches"): A_SPEC12,
    ("hmm", "data.stargraph_max_nodes"): A_SPEC12,  # dropped; resolves to defaults 50
}


# --------------------------------------------------------------------------------------
# Sweep resolution
# --------------------------------------------------------------------------------------


def resolve_sweep_singletons(source_yaml: Dict[str, Any]) -> Dict[str, Any]:
    """Return {dotted_key: scalar} for every sweep leaf that carries exactly one value.

    The upstream sweep block is a list of dicts whose leaves are lists (train.py:57-84
    expands them into a Cartesian product). A leaf with one element resolves unambiguously;
    a leaf with several (e.g. `seed: [1234..1238]`) does not and is excluded, so `seed`
    is never silently hoisted.
    """
    out: Dict[str, Any] = {}
    for entry in source_yaml.get("sweep") or []:
        for dotted, value in flatten(entry).items():
            if isinstance(value, list) and len(value) == 1:
                out[dotted] = value[0]
    return out


def build_one(name: str, seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    """Produce one materialized config dict, enforcing I1-I4. Raises on any violation."""
    plan = BUILDERS[name](seed)
    source_path = plan["source"]
    source = load_yaml(source_path)
    defaults = load_yaml(DEFAULTS_YAML)

    source_no_sweep = copy.deepcopy(source)
    source_no_sweep.pop("sweep", None)

    sweep_single = resolve_sweep_singletons(source)
    # What upstream actually resolves to for a single sweep point.
    upstream_resolved = deep_merge(defaults, source_no_sweep)
    for dotted, value in sweep_single.items():
        set_dotted(upstream_resolved, dotted, value)

    out = copy.deepcopy(source_no_sweep)

    # --- I3: hoists are value-preserving -------------------------------------------------
    for hoist in plan["hoists"]:
        if not has_dotted(upstream_resolved, hoist.key):
            raise AssertionError(
                f"{name}: hoist {hoist.key!r} does not resolve upstream at all"
            )
        resolved = get_dotted(upstream_resolved, hoist.key)
        if has_dotted(out, hoist.key) and get_dotted(out, hoist.key) != resolved:
            raise AssertionError(
                f"{name}: hoist {hoist.key!r} would change {get_dotted(out, hoist.key)!r} "
                f"-> {resolved!r}; that is an override, not a hoist"
            )
        set_dotted(out, hoist.key, resolved)

    # --- drops --------------------------------------------------------------------------
    declared_drops = {d.key for d in plan["drops"]}
    for dotted in sorted(declared_drops):
        if not has_dotted(out, dotted):
            raise AssertionError(f"{name}: declared drop {dotted!r} is not in the source")
        del_dotted(out, dotted)

    # --- overrides ----------------------------------------------------------------------
    declared: Dict[str, Ov] = {}
    for ov in plan["overrides"]:
        if ov.key in declared:
            raise AssertionError(f"{name}: duplicate override {ov.key!r}")
        declared[ov.key] = ov
        set_dotted(out, ov.key, copy.deepcopy(ov.value))

    # --- I1: no upstream key may vanish --------------------------------------------------
    vanished = [k for k in missing_keys(source_no_sweep, out) if k not in declared_drops]
    # `sweep` itself is intentionally absent; its singleton leaves were hoisted and its
    # seed list is replaced by the explicit preregistered `seed`.
    if vanished:
        raise AssertionError(f"{name}: upstream keys lost without a declared drop: {vanished}")
    for dotted in sweep_single:
        if not has_dotted(out, dotted):
            raise AssertionError(
                f"{name}: sweep singleton {dotted!r} was dropped with the sweep block "
                f"instead of being hoisted (this is the proj_factor trap)"
            )

    # --- I2: no silent change (bidirectional) -------------------------------------------
    merged_out = deep_merge(defaults, out)
    changed = set()
    for dotted, value in flatten(merged_out).items():
        if not has_dotted(upstream_resolved, dotted):
            changed.add(dotted)
        elif get_dotted(upstream_resolved, dotted) != value:
            changed.add(dotted)
    for dotted in flatten(upstream_resolved):
        if not has_dotted(merged_out, dotted):
            changed.add(dotted)

    def _covered(dotted: str) -> bool:
        # A dict-valued override such as `provenance` or `data.hmm` covers its own leaves,
        # and a declared drop covers the key it removed (which either disappears from the
        # merged config or falls back to its defaults.yaml value).
        keys = set(declared) | declared_drops
        return any(dotted == k or dotted.startswith(k + ".") for k in keys)

    undeclared = sorted(d for d in changed if not _covered(d))
    if undeclared:
        raise AssertionError(f"{name}: undeclared changes vs upstream: {undeclared}")

    inert = sorted(
        k for k, ov in declared.items()
        if not ov.inert_ok
        and not any(d == k or d.startswith(k + ".") for d in changed)
    )
    if inert:
        raise AssertionError(
            f"{name}: overrides declared that change nothing (move them to hoists): {inert}"
        )

    # --- I4: frozen keys ------------------------------------------------------------------
    family = plan["family"]
    for key in FROZEN_KEYS:
        here = get_dotted(merged_out, key) if has_dotted(merged_out, key) else None
        there = get_dotted(upstream_resolved, key) if has_dotted(upstream_resolved, key) else None
        if here != there and (family, key) not in EXEMPT_FROZEN:
            raise AssertionError(
                f"{name}: FROZEN key {key} moved {there!r} -> {here!r} with no spec authority"
            )

    # --- I5: pool identity ------------------------------------------------------------------
    _check_pool_identity(name, plan, merged_out)

    audit = {
        "config": name,
        "family": family,
        "source_config": os.path.relpath(source_path, UPSTREAM),
        "source_config_sha256": sha256_file(source_path),
        "defaults_sha256": sha256_file(DEFAULTS_YAML),
        "upstream_commit": UPSTREAM_COMMIT,
        "overrides": [asdict(o) for o in plan["overrides"]],
        "hoists": [asdict(h) for h in plan["hoists"]],
        "drops": [asdict(d) for d in plan["drops"]],
    }
    return {"yaml": out, "audit": audit}


HEADER = """\
# {name}
#
# GENERATED by scripts/materialize_configs.py -- do not hand-edit.
# Regenerate:  .venv/bin/python scripts/materialize_configs.py
# Verify:      .venv/bin/python scripts/materialize_configs.py --check
#
# COPIED from the pinned official configuration, per spec section 8:
#   upstream commit {commit}
#   {source}
#   sha256 {sha}
# Every changed key is declared in configs/overrides.json with its spec authority, and
# docs/CONFIG_DEVIATIONS.md carries the prose. The generator refuses to emit a file in
# which an upstream key vanished, a value changed without a declaration, a hoisted key
# changed value, or a frozen key moved without a written spec authority.
#
# Launch (spec section 8, single GPU):
#   scripts/launch_train.sh {name} <seed>
# which expands to
#   fabric run --devices 1 --precision bf16-mixed train.py --config <this file> \\
#     seed=<seed> trainer.out_dir=... trainer.experiment_name=...
"""


def render(name: str, doc: Dict[str, Any]) -> str:
    body = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, width=100)
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="verify the on-disk configs match what this generator produces")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="seed baked into the emitted defaults (must be preregistered)")
    args = parser.parse_args()

    if args.seed not in PREREGISTERED_SEEDS:
        print(f"seed {args.seed} is not preregistered {PREREGISTERED_SEEDS}", file=sys.stderr)
        return 2

    os.makedirs(CONFIGS_DIR, exist_ok=True)
    audits = []
    failures = []
    for name in BUILDERS:
        built = build_one(name, args.seed)
        audits.append(built["audit"])
        source = load_yaml(built["audit"] and BUILDERS[name](args.seed)["source"])
        del source
        text = HEADER.format(
            name=name,
            commit=UPSTREAM_COMMIT,
            source=built["audit"]["source_config"],
            sha=built["audit"]["source_config_sha256"],
        ) + render(name, built["yaml"])
        path = os.path.join(CONFIGS_DIR, name)
        if args.check:
            if not os.path.exists(path):
                failures.append(f"{name}: missing")
            elif open(path).read() != text:
                failures.append(f"{name}: on-disk content differs from the generator output")
        else:
            with open(path, "w") as fh:
                fh.write(text)

    audit_path = os.path.join(CONFIGS_DIR, "overrides.json")
    audit_text = json.dumps(
        {"upstream_commit": UPSTREAM_COMMIT, "preregistered_seeds": PREREGISTERED_SEEDS,
         "frozen_keys": FROZEN_KEYS,
         "frozen_exemptions": [{"family": f, "key": k, "authority": a}
                               for (f, k), a in sorted(EXEMPT_FROZEN.items())],
         "configs": audits},
        indent=2, sort_keys=False) + "\n"
    if args.check:
        if not os.path.exists(audit_path) or open(audit_path).read() != audit_text:
            failures.append("overrides.json: differs from the generator output")
        if failures:
            for f in failures:
                print("DRIFT " + f, file=sys.stderr)
            return 1
        print(f"OK  {len(BUILDERS)} configs match the generator at commit {UPSTREAM_COMMIT}")
        return 0

    with open(audit_path, "w") as fh:
        fh.write(audit_text)
    for name in BUILDERS:
        print(f"wrote configs/{name}")
    print("wrote configs/overrides.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
