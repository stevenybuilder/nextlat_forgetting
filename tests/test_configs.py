"""Every materialized config is checked against the pinned upstream repository, not against
the spec document.

The failure this suite exists to prevent is on the record: docs/RUNLOG.md, session 1, attempt
2 -- a hand-written smoke config dropped `data.test_generalization`, and train.py died at the
first validation with an omegaconf ConfigAttributeError while the driver reported success.
`test_every_key_the_trainer_reads_resolves` reproduces that check mechanically: it extracts
the `config.<section>.<key>` accesses out of the pinned source files that our runs actually
execute and asserts each one resolves in the merged config.

Every test here can fail on wrong input. The negative-control tests at the bottom prove it by
mutating a config in the exact ways this project is exposed to (dropping the sweep block so
proj_factor silently reverts to 1.0; restoring compile: true; letting the near and far
branches share an output root) and asserting the corresponding check rejects them.
"""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from config_lib import (  # noqa: E402
    CONFIGS_DIR,
    DEFAULTS_YAML,
    DELIVERABLE_CONFIGS,
    OFFICIAL_BST_5_5,
    OFFICIAL_GPT_5_5,
    OFFICIAL_NEXTLAT_5_5,
    UPSTREAM,
    UPSTREAM_COMMIT,
    block_param_count,
    bst_encoder_param_count,
    bst_pair_accum_steps,
    bst_pairs_per_sequence,
    bst_param_count,
    bst_texthead_hidden_dim,
    bst_texthead_param_count,
    deep_merge,
    del_dotted,
    dynamics_hidden_dim,
    dynamics_param_count,
    flatten,
    get_dotted,
    gpt_param_count,
    nextlat_param_count,
    has_dotted,
    load_yaml,
    load_yaml_as_trainer_sees_it,
    missing_keys,
    optimizer_updates,
    stargraph_vocab_size,
    swiglu_hidden_dim,
)

PYTHON = str(REPO / ".venv" / "bin" / "python")


# --------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------


def _merged(name: str) -> dict:
    """defaults.yaml merged with the config, exactly as train.py:348-351 does it."""
    defaults = load_yaml_as_trainer_sees_it(DEFAULTS_YAML)
    cfg = load_yaml_as_trainer_sees_it(os.path.join(CONFIGS_DIR, name))
    return deep_merge(defaults, cfg)


@pytest.fixture(scope="module")
def merged() -> dict:
    return {name: _merged(name) for name in DELIVERABLE_CONFIGS}


@pytest.fixture(scope="module")
def audit() -> dict:
    with open(os.path.join(CONFIGS_DIR, "overrides.json")) as fh:
        return json.load(fh)


LURESTAR = ["gpt_lurestar.yaml", "nextlat_lurestar.yaml", "bst_lurestar.yaml"]
BST_CFGS = ["bst_lurestar.yaml"]
ADAPT = ["adapt_near.yaml", "adapt_far.yaml"]
HMM = ["gpt_hmm.yaml", "nextlat_hmm.yaml"]
NEXTLAT_CFGS = ["nextlat_lurestar.yaml", "adapt_near.yaml", "adapt_far.yaml",
                "nextlat_hmm.yaml"]


# --------------------------------------------------------------------------------------
# 1. the files exist, parse, and merge
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", DELIVERABLE_CONFIGS)
def test_parses_and_merges(name: str) -> None:
    cfg = load_yaml_as_trainer_sees_it(os.path.join(CONFIGS_DIR, name))
    assert isinstance(cfg, dict) and cfg, f"{name} is empty"
    merged = deep_merge(load_yaml_as_trainer_sees_it(DEFAULTS_YAML), cfg)
    # the merge must be additive: no defaults key may disappear
    assert missing_keys(load_yaml_as_trainer_sees_it(DEFAULTS_YAML), merged) == []


def test_generator_is_reproducible() -> None:
    """The on-disk configs are exactly what the generator emits from the pinned YAMLs."""
    proc = subprocess.run(
        [PYTHON, str(REPO / "scripts" / "materialize_configs.py"), "--check"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_pinned_commit_matches_the_checkout() -> None:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=UPSTREAM,
                          capture_output=True, text=True).stdout.strip()
    assert head == UPSTREAM_COMMIT


# --------------------------------------------------------------------------------------
# 2. no upstream key vanished  (the test_generalization failure, mechanized)
# --------------------------------------------------------------------------------------

# Keys accessed only under a code branch our runs never take, or injected at runtime.
_NOT_REQUIRED = {
    # computed inside train.py:143-153 from effective_batch_size, never read from YAML
    "data.device_batch_size": "computed at train.py:143-145",
    "data.micro_batch_size": "computed at train.py:151-153",
    # only read when use_mtp_gloeckle / use_mtp_jtp, which are false in every config here
    "model.mtp_lambda": "read only under use_mtp_gloeckle/use_mtp_jtp (core_train.py:100-110)",
}

# Reads guarded by getattr/.get(), which supply their own default.
_GUARDED = {
    "config.trainer.get", "config.lr_scheduler.get",
}

_ACCESS = re.compile(r"config\.(trainer|data|model|optimizer|lr_scheduler)\.([A-Za-z_]\w*)")

# The pinned source files a GPT or NextLat stargraph/hmm run actually executes.
_ON_PATH = [
    "train.py",
    "core_train.py",
    "models/model_base.py",
    "models/model_gpt.py",
    "models/model_nextlat.py",
]
_STARGRAPH_ONLY = ["data/stargraph.py"]


def _accesses(files) -> set:
    keys = set()
    for rel in files:
        text = (Path(UPSTREAM) / rel).read_text()
        for section, key in _ACCESS.findall(text):
            dotted = f"config.{section}.{key}"
            if dotted in _GUARDED:
                continue
            keys.add(f"{section}.{key}")
    return keys


@pytest.mark.parametrize("name", DELIVERABLE_CONFIGS)
def test_every_key_the_trainer_reads_resolves(merged, name: str) -> None:
    files = list(_ON_PATH)
    if merged[name]["data"]["dataset"] == "stargraph":
        files += _STARGRAPH_ONLY
    required = _accesses(files) - set(_NOT_REQUIRED)
    unresolvable = sorted(k for k in required if not has_dotted(merged[name], k))
    assert unresolvable == [], (
        f"{name}: keys read by the pinned trainer that the merged config cannot resolve: "
        f"{unresolvable}"
    )


@pytest.mark.parametrize("name,source", [
    ("gpt_lurestar.yaml", OFFICIAL_GPT_5_5),
    ("nextlat_lurestar.yaml", OFFICIAL_NEXTLAT_5_5),
    ("bst_lurestar.yaml", OFFICIAL_BST_5_5),
    ("adapt_near.yaml", OFFICIAL_NEXTLAT_5_5),
    ("adapt_far.yaml", OFFICIAL_NEXTLAT_5_5),
])
def test_no_upstream_key_dropped(name: str, source: str) -> None:
    """For the stargraph families every key of the official YAML must still be there."""
    official = load_yaml(source)
    official.pop("sweep", None)
    ours = load_yaml(os.path.join(CONFIGS_DIR, name))
    assert missing_keys(official, ours) == []


@pytest.mark.parametrize("name", HMM)
def test_hmm_drops_are_declared_and_stargraph_only(audit, name: str) -> None:
    entry = next(c for c in audit["configs"] if c["config"] == name)
    dropped = {d["key"] for d in entry["drops"]}
    official = load_yaml(OFFICIAL_GPT_5_5 if name.startswith("gpt") else OFFICIAL_NEXTLAT_5_5)
    official.pop("sweep", None)
    ours = load_yaml(os.path.join(CONFIGS_DIR, name))
    assert set(missing_keys(official, ours)) == dropped
    # every dropped key must be one only StarGraphDataModule reads
    stargraph_txt = (Path(UPSTREAM) / "data/stargraph.py").read_text()
    others = "".join((Path(UPSTREAM) / f).read_text() for f in _ON_PATH)
    for key in dropped:
        leaf = key.split(".")[-1]
        assert leaf in stargraph_txt, f"{key} is not read by data/stargraph.py at all"
        assert f".{leaf}" not in others or leaf == "stargraph_max_nodes", (
            f"{key} is read outside data/stargraph.py and must not be dropped"
        )


def test_adapt_config_is_also_a_valid_gpt_config() -> None:
    """The adaptation branch runs GPT from the same file via `use_nextlat=false`."""
    gpt = load_yaml(OFFICIAL_GPT_5_5)
    gpt.pop("sweep", None)
    for name in ADAPT:
        ours = load_yaml(os.path.join(CONFIGS_DIR, name))
        assert missing_keys(gpt, ours) == [], (
            f"{name} is missing keys the official GPT G(5,5) config supplies"
        )


# --------------------------------------------------------------------------------------
# 3. resolved paper-scale values  (spec section 8's "must verify before launch" block)
# --------------------------------------------------------------------------------------

PAPER_SCALE = {
    "trainer.train_batches": 20000,
    "trainer.save_last_checkpoint": True,
    "trainer.save_best_checkpoint": True,
    "trainer.save_recovery_checkpoint": 250,
    "trainer.compile": False,
    "trainer.val_interval": 1000,
    "trainer.test_interval": 1000,
    "trainer.val_batches": 200,
    "trainer.test_batches": 200,
    "trainer.init_from": "scratch",
    "data.dataset": "stargraph",
    "data.effective_batch_size": 512,
    "data.gradient_accum_steps": 1,
    "data.stargraph_max_nodes": 100,
    "data.test_generalization": False,
    "model.n_layer": 12,
    "model.n_head": 6,
    "model.n_embd": 384,
    "model.dropout": 0.0,
    "model.bias": False,
    "model.gpt_mode": "next_token",
    "optimizer.optimizer_type": "adam",
    "optimizer.learning_rate": 5.0e-4,
    "optimizer.weight_decay": 0.1,
    "optimizer.beta1": 0.9,
    "optimizer.beta2": 0.95,
    "optimizer.grad_clip": 100,
    "lr_scheduler.schedule": "constant",
    "lr_scheduler.warmup_iters": 0,
    "lr_scheduler.warmdown_iters": 0,
}

NEXTLAT_PAPER_SCALE = {
    "model.mtp_horizon": 3,
    "model.lambda_mse": 1.0,
    "model.lambda_kl": 1.0,
    "model.lambda_ce": 0.0,
    "model.proj_factor": 0.5,
}


@pytest.mark.parametrize("name", LURESTAR)
@pytest.mark.parametrize("key,expected", sorted(PAPER_SCALE.items()))
def test_lurestar_resolves_to_paper_scale(merged, name, key, expected) -> None:
    got = get_dotted(merged[name], key)
    assert got == expected and type(got) is type(expected), f"{name}: {key} = {got!r}"


@pytest.mark.parametrize("key,expected", sorted(NEXTLAT_PAPER_SCALE.items()))
def test_nextlat_lurestar_resolves_to_paper_scale(merged, key, expected) -> None:
    got = get_dotted(merged["nextlat_lurestar.yaml"], key)
    assert got == expected and type(got) is type(expected)


ARCH_MATCHED_KEYS = [
    "model.n_layer", "model.n_head", "model.n_embd", "model.dropout",
    "model.bias", "model.gpt_mode", "data.effective_batch_size",
    "data.gradient_accum_steps", "trainer.train_batches",
    "optimizer.learning_rate", "optimizer.weight_decay", "optimizer.beta1",
    "optimizer.beta2", "optimizer.grad_clip", "lr_scheduler.schedule",
    "data.stargraph_train_data_path", "data.stargraph_test_data_path",
]


def test_gpt_and_nextlat_are_architecture_matched(merged) -> None:
    gpt, nextlat = merged["gpt_lurestar.yaml"], merged["nextlat_lurestar.yaml"]
    for key in ARCH_MATCHED_KEYS:
        assert get_dotted(gpt, key) == get_dotted(nextlat, key), key
    assert gpt["use_nextlat"] is False and nextlat["use_nextlat"] is True


def test_bst_is_architecture_matched_to_gpt_and_nextlat(merged) -> None:
    """The competence-matched control only controls if it is matched.

    docs/DECISION_D20_competence_gate.md, "Superseded in part": BST solves G(5,5) at ~99.9%
    while GPT sits at ~18.6% (= 1/d, chance), so the primary contrast is NextLat vs BST.
    That contrast is only about the OBJECTIVE if width, depth, optimizer, schedule, batch
    size and corpus are identical across the three arms.
    """
    gpt, nextlat, bst = (merged["gpt_lurestar.yaml"], merged["nextlat_lurestar.yaml"],
                         merged["bst_lurestar.yaml"])
    for key in ARCH_MATCHED_KEYS:
        assert get_dotted(bst, key) == get_dotted(gpt, key) == get_dotted(nextlat, key), key
    assert bst["use_bst"] is True and bst["use_nextlat"] is False
    assert gpt["use_bst"] is False and nextlat["use_bst"] is False


def test_bst_keeps_the_official_pair_gap(merged) -> None:
    """`bst_pair_minimum_gap: 2` is the one scientific key the official BST YAML adds.

    defaults.yaml:98 ships 1. core_train.py:80 feeds the resolved value into BSTConfig, so
    an arm that loses the explicit 2 trains a different objective and nothing warns -- the
    same failure mode as the NextLat `proj_factor` trap (D-07).
    """
    assert get_dotted(merged["bst_lurestar.yaml"], "model.bst_pair_minimum_gap") == 2
    assert get_dotted(merged["gpt_lurestar.yaml"], "model.bst_pair_minimum_gap") == 1
    # core_train.py:41-43 asserts membership before the model is built
    assert get_dotted(merged["bst_lurestar.yaml"],
                      "model.bst_single_gap_prediction_mode") in ("next_token", "eos")


# --------------------------------------------------------------------------------------
# 3b. BST -- the competence-matched third arm (spec section 8, D-20 "Superseded in part")
#
# BST is the arm that makes the cross-model contrast identifiable, so two things have to
# hold and both are asserted here rather than asserted in prose:
#   (a) every key the pinned BST code path reads is WRITTEN in the file, not inherited
#       from defaults.yaml -- the D-07 `proj_factor` hazard, applied to BST;
#   (b) the transformer BST builds is the same transformer GPT and NextLat build, and
#       where BST adds parameters, exactly how many and where.
# --------------------------------------------------------------------------------------


def _bst_branch_reads() -> set:
    """Config keys read inside a `if config.use_bst:` block of the pinned core_train.py.

    Derived from the source rather than hand-listed, so a BST key added upstream at a
    future commit breaks this test instead of silently resolving out of defaults.yaml.
    """
    lines = (Path(UPSTREAM) / "core_train.py").read_text().splitlines()
    keys, i = set(), 0
    while i < len(lines):
        line = lines[i]
        if line.strip() == "if config.use_bst:":
            indent = len(line) - len(line.lstrip())
            j = i + 1
            while j < len(lines):
                nxt = lines[j]
                if nxt.strip() and (len(nxt) - len(nxt.lstrip())) <= indent:
                    break
                keys.update(f"{sec}.{key}" for sec, key in _ACCESS.findall(nxt))
                j += 1
            i = j
        else:
            i += 1
    return keys


# `data.pair_batch_size` is read OUTSIDE a use_bst branch (core_train.py:495 and :635) and
# handed to every arm's compute_loss, but only BST consumes it: model_gpt.py:342 and
# model_nextlat.py:418 both absorb it into `**kwargs  # Extra arguments ignored for
# compatibility with BST`. model_bst.py:600-601 turns it into `pair_accum_steps` and folds
# that into `texthead_loss_div`, which makes it a loss-scaling key.
BST_RUNTIME_KEYS = {
    "model.context_length",                    # core_train.py:78
    "model.bst_pair_minimum_gap",              # core_train.py:79
    "model.bst_pair_maximum_gap",              # core_train.py:80
    "model.bst_pair_subsample_rate",           # core_train.py:81
    "model.bst_single_gap_prediction_mode",    # core_train.py:41-44 and :82
}
BST_EXTRA_KEYS = {"data.pair_batch_size"}


def test_bst_branch_key_list_is_still_complete() -> None:
    """If upstream ever adds a BST config key, this fails before anything else can."""
    assert _bst_branch_reads() == BST_RUNTIME_KEYS, (
        "the pinned core_train.py BST branch no longer reads exactly the keys this suite "
        "pins; update BST_RUNTIME_KEYS and the hoist list in materialize_configs.py"
    )


@pytest.mark.parametrize("key", sorted(BST_RUNTIME_KEYS | BST_EXTRA_KEYS))
def test_bst_key_is_written_in_the_file_not_inherited(key: str) -> None:
    """D-07 guard for BST: no scientifically relevant key may resolve only via a fallback.

    `model.proj_factor` cost this project a wrong architecture because it lived only in the
    upstream `sweep:` block. BST has the same hazard in the other direction: only
    `bst_pair_minimum_gap` is written in `bst_stargraph_5_5.yaml`; the other three BST keys
    and `pair_batch_size` are written in NO stargraph YAML at all and fall through to
    `defaults.yaml:67,98-104`. This asserts on the RAW file, before the defaults merge.
    """
    raw = load_yaml(os.path.join(CONFIGS_DIR, "bst_lurestar.yaml"))
    assert has_dotted(raw, key), (
        f"bst_lurestar.yaml does not write {key}; it would resolve out of defaults.yaml "
        f"and a defaults edit would move the objective with nothing to catch it"
    )


BST_RESOLVED = {
    "use_bst": True,
    "use_nextlat": False,
    "model.bst_pair_minimum_gap": 2,          # official BST YAML; defaults.yaml:98 is 1
    "model.bst_pair_maximum_gap": -1,         # defaults.yaml:99 -> "every gap"
    "model.bst_pair_subsample_rate": 1.0,     # defaults.yaml:102 -> "keep all pairs"
    "model.bst_single_gap_prediction_mode": "eos",   # defaults.yaml:104
    "data.pair_batch_size": 32768,            # defaults.yaml:67
    "data.pair_accum_steps": 1,               # written upstream, read by no pinned code
    "model.context_length": 0,                # placeholder; data/stargraph.py:251 sets 62
}


@pytest.mark.parametrize("key,expected", sorted(BST_RESOLVED.items()))
def test_bst_resolves_to_the_upstream_bst_values(merged, key, expected) -> None:
    got = get_dotted(merged["bst_lurestar.yaml"], key)
    assert got == expected and type(got) is type(expected), f"{key} = {got!r}"


def test_bst_config_differs_from_the_official_gpt_config_by_two_keys_only() -> None:
    """Spec section 8's claim about the arms, checked against the pinned YAMLs themselves."""
    gpt = load_yaml_as_trainer_sees_it(OFFICIAL_GPT_5_5)
    bst = load_yaml_as_trainer_sees_it(OFFICIAL_BST_5_5)
    for cfg in (gpt, bst):
        cfg.pop("sweep", None)
        cfg["trainer"].pop("experiment_name", None)   # a label, not a setting
    flat_gpt, flat_bst = flatten(gpt), flatten(bst)
    differing = {k for k in set(flat_gpt) | set(flat_bst)
                 if flat_gpt.get(k, "<absent>") != flat_bst.get(k, "<absent>")}
    assert differing == {"use_bst", "use_nextlat", "model.bst_pair_minimum_gap"}, differing
    # `use_nextlat` differs only because the BST YAML omits it; it resolves false either way
    assert flat_gpt["use_nextlat"] is False and "use_nextlat" not in flat_bst
    assert flat_gpt["use_bst"] is False and flat_bst["use_bst"] is True
    assert flat_bst["model.bst_pair_minimum_gap"] == 2
    assert "model.bst_pair_minimum_gap" not in flat_gpt


# --- resolved parameter counts ----------------------------------------------------------
#
# Spec section 8 calls the three arms "architecture-matched". That is true of the
# transformer -- models/model_bst.py:28 imports the very `Block` class model_gpt.py defines,
# and BST builds it at the same 12 / 6 / 384 -- but it is NOT true of the parameter count,
# because model_bst.py:148-161 builds TWO stacks (`transformer_f`, `transformer_b`) and
# model_bst.py:53-77 adds a `TextHead`. The writeup must say "architecture-matched" of the
# per-stack transformer and qualify the parameter count, so both numbers are pinned here.

VOCAB = 106          # stargraph_vocab_size(100), data/stargraph.py:233
SEQ_LEN = 69         # tokenized G(5,5) line + eos; data/stargraph.py:51-56
CONTEXT_LEN = 62     # graph_description_len; data/stargraph.py:243-247


def test_bst_builds_the_same_transformer_at_the_same_width(merged) -> None:
    """`from models.model_gpt import Block` (models/model_bst.py:28) -- literally the same
    layer object, so matching n_layer / n_head / n_embd / bias makes the stacks identical."""
    bst = merged["bst_lurestar.yaml"]["model"]
    gpt = merged["gpt_lurestar.yaml"]["model"]
    nextlat = merged["nextlat_lurestar.yaml"]["model"]
    for key in ("n_layer", "n_head", "n_embd", "bias", "dropout"):
        assert bst[key] == gpt[key] == nextlat[key], key
    assert (bst["n_layer"], bst["n_head"], bst["n_embd"], bst["bias"]) == (12, 6, 384, False)
    # one layer, identical across the three arms
    assert block_param_count(384, 6) == 1_770_240
    assert swiglu_hidden_dim(384) == 1024


def test_resolved_parameter_counts_of_the_three_arms(merged) -> None:
    """The exact numbers the writeup is allowed to quote.

    GPT      21,324,672   trunk + separate (untied) lm_head
    NextLat  21,915,264   = GPT + 590,592 latent-dynamics MLP
    BST      47,287,296   = 2 x 12-layer stack + shared embedding + 4,760,064 TextHead
    """
    assert stargraph_vocab_size(merged["bst_lurestar.yaml"]["data"]["stargraph_max_nodes"]) == VOCAB

    gpt = gpt_param_count(384, 6, 12, VOCAB)
    nextlat = nextlat_param_count(384, 6, 12, VOCAB, 0.5)
    bst = bst_param_count(384, 6, 12, VOCAB)
    assert (gpt, nextlat, bst) == (21_324_672, 21_915_264, 47_287_296)

    # BST is NOT parameter-matched, and this is the decomposition of the difference.
    one_stack = 12 * block_param_count(384, 6) + 384        # blocks + final LayerNorm
    assert bst_encoder_param_count(384, 6, 12, VOCAB) == VOCAB * 384 + 2 * one_stack
    assert bst_texthead_param_count(384, VOCAB) == 4_760_064
    assert bst_texthead_hidden_dim(384) == 2048
    # the second (backward) stack is exactly the extra trunk
    assert bst - gpt == one_stack + bst_texthead_param_count(384, VOCAB) - 384 * VOCAB
    assert bst - gpt == 25_962_624
    assert round(bst / gpt, 3) == 2.217


def test_bst_extra_parameters_are_documented_precisely() -> None:
    """"Architecture-matched" is a claim the writeup makes; it must be qualified in the
    deviations document with the real numbers, not left to the reader."""
    doc = " ".join((REPO / "docs" / "CONFIG_DEVIATIONS.md").read_text().split())
    for phrase in ["47,287,296", "4,760,064", "transformer_f", "transformer_b",
                   "TextHead", "not parameter-matched"]:
        assert phrase in doc, phrase


def test_bst_pair_batch_size_does_not_reweight_the_text_head_loss(merged) -> None:
    """`pair_batch_size` is a loss-scaling key, so the value it resolves to has to be
    checked against the actual pair count, not just recorded.

    models/model_bst.py:600-601:
        pair_accum_steps    = ceil(n_pairs / pair_batch_size)
        texthead_loss_div   = loss_div * batch_size * pair_accum_steps
    With more than one chunk the text-head loss becomes a mean of chunk means rather than
    the mean over pairs. At G(5,5) scale one sequence yields 15 pairs against a chunk of
    32,768, so there is exactly one chunk and the loss is the plain mean.
    """
    cfg = merged["bst_lurestar.yaml"]
    min_gap = cfg["model"]["bst_pair_minimum_gap"]
    max_gap = cfg["model"]["bst_pair_maximum_gap"]
    n_pairs = bst_pairs_per_sequence(SEQ_LEN, CONTEXT_LEN, min_gap, max_gap)
    assert n_pairs == 15
    assert bst_pair_accum_steps(n_pairs, cfg["data"]["pair_batch_size"]) == 1
    # the minimum gap really does change the training set, which is why it must be explicit
    assert bst_pairs_per_sequence(SEQ_LEN, CONTEXT_LEN, 1, max_gap) == 21


def test_bst_single_gap_mode_is_inert_at_minimum_gap_two(merged) -> None:
    """models/model_bst.py:584-588 rewrites targets only for pairs of gap exactly 1.
    `bst_pair_minimum_gap: 2` produces none, so `bst_single_gap_prediction_mode` changes no
    gradient here -- but core_train.py:41-44 asserts it before the model is built, so a
    missing or misspelled value aborts the run at step 0."""
    cfg = merged["bst_lurestar.yaml"]
    assert cfg["model"]["bst_pair_minimum_gap"] >= 2
    assert cfg["model"]["bst_single_gap_prediction_mode"] in ("next_token", "eos")


# --- D-07, generalized ------------------------------------------------------------------


@pytest.mark.parametrize("name,source", [
    ("gpt_lurestar.yaml", OFFICIAL_GPT_5_5),
    ("nextlat_lurestar.yaml", OFFICIAL_NEXTLAT_5_5),
    ("bst_lurestar.yaml", OFFICIAL_BST_5_5),
    ("adapt_near.yaml", OFFICIAL_NEXTLAT_5_5),
    ("adapt_far.yaml", OFFICIAL_NEXTLAT_5_5),
    ("gpt_hmm.yaml", OFFICIAL_GPT_5_5),
    ("nextlat_hmm.yaml", OFFICIAL_NEXTLAT_5_5),
])
def test_no_key_is_reachable_only_through_the_sweep_block(name: str, source: str) -> None:
    """D-07, stated once for every arm instead of once for `proj_factor`.

    The emitted configs delete `sweep:` (train.py:280,322 builds the experiment directory
    name by iterating a Python set, so it is not reproducible). Every leaf the sweep block
    carried must therefore be written somewhere in the emitted file. `proj_factor` is the
    key this catches today; the test is written over the source so it catches the next one.
    """
    official = load_yaml_as_trainer_sees_it(source)
    ours = load_yaml_as_trainer_sees_it(os.path.join(CONFIGS_DIR, name))
    sweep_leaves = set()
    for entry in official.get("sweep") or []:
        sweep_leaves.update(flatten(entry))
    assert sweep_leaves, f"{source} has no sweep block; this guard would be vacuous"
    missing = sorted(k for k in sweep_leaves if not has_dotted(ours, k))
    assert missing == [], f"{name}: sweep-only keys with no explicit value: {missing}"


# --- negative controls for the BST arm ---------------------------------------------------


def test_negative_control_dropping_use_bst_silently_trains_gpt() -> None:
    """core_train.py:38-58 dispatches on `use_bst` first and falls through to GPT. Losing
    the flag produces a run that trains, logs and checkpoints normally -- as the wrong
    model, in the directory labelled `bst`."""
    cfg = load_yaml_as_trainer_sees_it(os.path.join(CONFIGS_DIR, "bst_lurestar.yaml"))
    assert cfg["use_bst"] is True
    del cfg["use_bst"]
    bad = deep_merge(load_yaml_as_trainer_sees_it(DEFAULTS_YAML), cfg)
    assert bad["use_bst"] is False, "defaults.yaml:2 silently supplies false"
    assert bad["use_nextlat"] is False
    # nothing in the merged config would flag it; only the explicit assertion does
    ours = _merged("bst_lurestar.yaml")
    assert ours["use_bst"] is True


def test_negative_control_dropping_bst_pair_minimum_gap_changes_the_objective() -> None:
    """The BST analogue of the `proj_factor` trap: the fallback is a valid number, so the
    run succeeds and trains a different pair distribution."""
    cfg = load_yaml_as_trainer_sees_it(os.path.join(CONFIGS_DIR, "bst_lurestar.yaml"))
    del cfg["model"]["bst_pair_minimum_gap"]
    bad = deep_merge(load_yaml_as_trainer_sees_it(DEFAULTS_YAML), cfg)
    assert bad["model"]["bst_pair_minimum_gap"] == 1, "defaults.yaml:98"
    assert bst_pairs_per_sequence(SEQ_LEN, CONTEXT_LEN, 1) == 21
    assert bst_pairs_per_sequence(SEQ_LEN, CONTEXT_LEN, 2) == 15
    assert _merged("bst_lurestar.yaml")["model"]["bst_pair_minimum_gap"] == 2


@pytest.mark.parametrize("key", sorted(BST_RUNTIME_KEYS | BST_EXTRA_KEYS))
def test_negative_control_a_dropped_bst_key_is_caught(key: str, tmp_path) -> None:
    """Delete any BST-only key from the emitted file and `--check` must fail. Without this
    the D-07 guard above would pass on a file the generator no longer produces."""
    target = Path(CONFIGS_DIR) / "bst_lurestar.yaml"
    original = target.read_text()
    try:
        cfg = load_yaml(str(target))
        del_dotted(cfg, key)
        target.write_text(yaml.safe_dump(cfg, sort_keys=False))
        proc = subprocess.run(
            [PYTHON, str(REPO / "scripts" / "materialize_configs.py"), "--check"],
            capture_output=True, text=True, cwd=str(REPO))
        assert proc.returncode != 0, f"dropping {key} was not detected"
        assert "bst_lurestar.yaml" in proc.stderr
    finally:
        target.write_text(original)


def test_negative_control_bst_at_a_different_width_is_caught(tmp_path) -> None:
    """A BST arm at a different depth or width would make the primary NextLat-vs-BST
    contrast un-identifiable. Both the generator and the arm-matching test must reject it."""
    target = Path(CONFIGS_DIR) / "bst_lurestar.yaml"
    original = target.read_text()
    try:
        target.write_text(original.replace("n_layer: 12", "n_layer: 8"))
        proc = subprocess.run(
            [PYTHON, str(REPO / "scripts" / "materialize_configs.py"), "--check"],
            capture_output=True, text=True, cwd=str(REPO))
        assert proc.returncode != 0
        bad = _merged("bst_lurestar.yaml")
        good = _merged("nextlat_lurestar.yaml")
        assert bad["model"]["n_layer"] != good["model"]["n_layer"]
    finally:
        target.write_text(original)
    assert _merged("bst_lurestar.yaml")["model"]["n_layer"] == 12


def test_bst_arm_is_recorded_in_the_audit_and_the_deviations_document(audit) -> None:
    entry = next((c for c in audit["configs"] if c["config"] == "bst_lurestar.yaml"), None)
    assert entry is not None, "configs/overrides.json has no BST arm"
    assert entry["source_config"] == "config/stargraph/5_5/bst_stargraph_5_5.yaml"
    assert entry["family"] == "lurestar"
    assert entry["drops"] == [], "nothing may be dropped from the official BST YAML"
    hoisted = {h["key"] for h in entry["hoists"]}
    assert BST_RUNTIME_KEYS - {"model.context_length", "model.bst_pair_minimum_gap"} <= hoisted
    assert "data.pair_batch_size" in hoisted
    doc = (REPO / "docs" / "CONFIG_DEVIATIONS.md").read_text()
    assert "bst_lurestar.yaml" in doc and "bst_stargraph_5_5.yaml" in doc
    for key in sorted(BST_RUNTIME_KEYS | BST_EXTRA_KEYS):
        assert key.split(".")[-1] in doc, key


def test_bst_frozen_keys_are_pinned(audit) -> None:
    """A BST-only key that is not on the frozen surface can be moved by a later edit with
    no invariant complaining."""
    frozen = set(audit["frozen_keys"])
    assert {"use_bst", "model.bst_pair_minimum_gap", "model.bst_pair_maximum_gap",
            "model.bst_pair_subsample_rate", "model.bst_single_gap_prediction_mode",
            "data.pair_batch_size"} <= frozen
    # and no exemption was granted for any of them
    for ex in audit["frozen_exemptions"]:
        assert not ex["key"].startswith("model.bst_"), ex
        assert ex["key"] not in ("use_bst", "data.pair_batch_size"), ex


def test_model_selection_flags_are_unambiguous(merged) -> None:
    for name, cfg in merged.items():
        flags = {k: cfg[k] for k in
                 ["use_bst", "use_nextlat", "use_mtp_gloeckle", "use_mtp_jtp"]}
        on = [k for k, v in flags.items() if v]
        assert len(on) <= 1, f"{name}: more than one model flag set: {on}"
        expected = (["use_nextlat"] if name in NEXTLAT_CFGS
                    else ["use_bst"] if name in BST_CFGS else [])
        assert on == expected, f"{name}: flags {on}"


def test_latent_dynamics_mlp_is_three_layers_of_384(merged) -> None:
    """Spec section 8: 'The paper reports a three-layer latent-dynamics MLP with hidden
    dimension 384 for Path-Star. Verify that the official NextLat YAML resolves to those
    values.'  models/model_nextlat.py:50-52 and :60-66."""
    cfg = merged["nextlat_lurestar.yaml"]
    n_embd = cfg["model"]["n_embd"]
    proj = cfg["model"]["proj_factor"]
    assert (n_embd, proj) == (384, 0.5)
    assert dynamics_hidden_dim(n_embd, proj) == 384
    # three nn.Linear layers: 768->384, 384->384, 384->384
    assert dynamics_param_count(n_embd, proj) == 768 * 384 + 384 * 384 + 384 * 384 + 768
    assert dynamics_param_count(n_embd, proj) == 590592
    # the silent fallback the sweep block would have caused
    assert dynamics_hidden_dim(n_embd, 1.0) == 768
    assert dynamics_param_count(n_embd, 1.0) - dynamics_param_count(n_embd, 0.5) == 884736


def test_stargraph_runtime_shapes(merged) -> None:
    """data/stargraph.py:249-252 overwrites three model keys at runtime; the YAML values
    are placeholders and the resolved ones are what the model is built with."""
    cfg = merged["nextlat_lurestar.yaml"]
    assert cfg["model"]["vocab_size"] == 0, "placeholder, overwritten at data/stargraph.py:250"
    assert stargraph_vocab_size(cfg["data"]["stargraph_max_nodes"]) == 106
    assert swiglu_hidden_dim(cfg["model"]["n_embd"]) == 1024


def test_optimizer_update_count_is_the_upstream_convention(merged) -> None:
    """core_train.py:564-571 uses an inclusive bound, so train_batches: 20000 runs 20,001
    updates. Recorded, not corrected: the step count is on the frozen surface."""
    assert optimizer_updates(20000) == 20001
    for name in LURESTAR:
        assert optimizer_updates(merged[name]["trainer"]["train_batches"]) == 20001


def test_effective_batch_divides_on_one_device(merged) -> None:
    """train.py:140-153 asserts divisibility by world size and by gradient_accum_steps."""
    for name, cfg in merged.items():
        eff = cfg["data"]["effective_batch_size"]
        accum = cfg["data"]["gradient_accum_steps"]
        assert eff % 1 == 0
        device_batch = eff // 1  # --devices 1
        assert device_batch % accum == 0
        assert device_batch // accum == eff, (
            f"{name}: gradient accumulation is only permitted as an execution fallback "
            f"and must preserve the effective batch"
        )


# --------------------------------------------------------------------------------------
# 4. H3 adaptation  (spec section 6)
# --------------------------------------------------------------------------------------


def test_adaptation_is_next_token_only(merged) -> None:
    """With lambda_mse = lambda_kl = lambda_ce = 0 the NextLat total loss at
    models/model_nextlat.py:488-497 reduces exactly to ntp_loss."""
    for name in ADAPT:
        m = merged[name]["model"]
        assert m["lambda_mse"] == 0.0
        assert m["lambda_kl"] == 0.0
        assert m["lambda_ce"] == 0.0
        # the frozen architecture is untouched
        assert (m["n_layer"], m["n_head"], m["n_embd"]) == (12, 6, 384)
        assert m["mtp_horizon"] == 3
        assert m["proj_factor"] == 0.5


def test_near_and_far_differ_only_in_the_item_bank_and_output_root() -> None:
    near = load_yaml(os.path.join(CONFIGS_DIR, "adapt_near.yaml"))
    far = load_yaml(os.path.join(CONFIGS_DIR, "adapt_far.yaml"))
    allowed = {
        "trainer.out_dir",
        "trainer.experiment_name",
        "data.stargraph_train_data_path",
        "data.stargraph_test_data_path",
        "provenance.note",
    }
    from config_lib import diff_keys
    differing = {k for k, _, _ in diff_keys(near, far)}
    assert differing == allowed, f"unexpected near/far differences: {differing ^ allowed}"


def test_near_and_far_output_roots_cannot_collide() -> None:
    near = load_yaml(os.path.join(CONFIGS_DIR, "adapt_near.yaml"))["trainer"]["out_dir"]
    far = load_yaml(os.path.join(CONFIGS_DIR, "adapt_far.yaml"))["trainer"]["out_dir"]
    assert near != far
    # spec section 9: the resume pointer lives at the output root, so neither may be a
    # prefix of the other either
    assert not near.startswith(far.rstrip("/") + "/")
    assert not far.startswith(near.rstrip("/") + "/")
    base = load_yaml(os.path.join(CONFIGS_DIR, "nextlat_lurestar.yaml"))["trainer"]["out_dir"]
    assert base not in (near, far)


def test_adaptation_matches_spec_item_and_update_budget(merged) -> None:
    """Spec section 8: 'Start H3 with 5,000 adaptation items and 500 updates.'"""
    for name in ADAPT:
        cfg = merged[name]
        assert cfg["trainer"]["train_batches"] == 500
        # the item count is carried by the filename that data/stargraph.py parses
        train = cfg["data"]["stargraph_train_data_path"]
        assert train.endswith("_5000.txt"), train
        assert cfg["data"]["effective_batch_size"] == 512


def test_adaptation_starts_from_the_frozen_parent_not_from_scratch(merged) -> None:
    for name in ADAPT:
        assert merged[name]["trainer"]["init_from"] == "resume", (
            "a scratch adaptation branch would train a fresh random model on 5,000 items"
        )


# --------------------------------------------------------------------------------------
# 4b. pool identity  (I5)
#
# `data/stargraph.py:187-190` parses only `split("_")[1]` and `[2]` out of a data path, so
# upstream reads B_near, B_far and the base corpus as the same kind of file. Before these
# tests existed the whole suite passed with the near and far banks SWAPPED and with an
# adaptation branch training on E_lure -- both reproduced in
# docs/review/configs-and-launch.md. These assertions read the emitted YAML directly rather
# than going through the generator, so they hold even if the generator is wrong.
# --------------------------------------------------------------------------------------

BANK_TAG = {"adapt_near.yaml": ("bnear", "bfar"), "adapt_far.yaml": ("bfar", "bnear")}
RESERVED_POOLS = ("elure", "e_lure", "apair", "a_pair", "stimuli")


@pytest.mark.parametrize("name", ADAPT)
def test_adaptation_banks_are_bound_to_their_branch(merged, name: str) -> None:
    """Spec section 6's primary outcome is `erosion_near - erosion_far`; a swapped bank
    negates it exactly and nothing else in this repository can see the swap."""
    own, other = BANK_TAG[name]
    for key in ["stargraph_train_data_path", "stargraph_test_data_path"]:
        base = os.path.basename(merged[name]["data"][key])
        assert own in base, f"{name}: {key} = {base!r} does not carry its branch tag {own!r}"
        assert other not in base, (
            f"{name}: {key} = {base!r} carries the OPPOSITE branch tag {other!r}"
        )
    # and the two branches must not read the same file
    near = load_yaml(os.path.join(CONFIGS_DIR, "adapt_near.yaml"))["data"]
    far = load_yaml(os.path.join(CONFIGS_DIR, "adapt_far.yaml"))["data"]
    assert near["stargraph_train_data_path"] != far["stargraph_train_data_path"]
    assert near["stargraph_test_data_path"] != far["stargraph_test_data_path"]


@pytest.mark.parametrize("name", LURESTAR + ADAPT)
def test_no_adaptation_path_touches_an_evaluation_or_base_pool(merged, name: str) -> None:
    """Spec section 5: 'No E_lure graph or lure may enter base or adaptation training.'"""
    base_corpus = {
        merged["gpt_lurestar.yaml"]["data"]["stargraph_train_data_path"],
        merged["gpt_lurestar.yaml"]["data"]["stargraph_test_data_path"],
    }
    for key in ["stargraph_train_data_path", "stargraph_test_data_path"]:
        path = merged[name]["data"][key]
        low = path.lower()
        for token in RESERVED_POOLS:
            assert token not in low, f"{name}: {key} = {path!r} names reserved pool {token!r}"
        if name in ADAPT:
            assert path not in base_corpus, (
                f"{name}: {key} points at the frozen base corpus; the adaptation branches "
                f"must read their own immutable bank"
            )
            assert os.path.dirname(path).endswith("/manifests/adapt"), path
        else:
            assert path in base_corpus


def test_negative_control_a_swapped_adaptation_bank_is_refused_by_the_generator() -> None:
    """The exact P0 mutation from docs/review/configs-and-launch.md, driven through the real
    generator: `build_adapt_near` handed the FAR bank must not be emittable."""
    import materialize_configs as M

    key = "_swapped_probe.yaml"
    M.BUILDERS[key] = lambda seed=M.DEFAULT_SEED: M._adapt("near", M.B_FAR, M.B_FAR_VAL, seed)
    try:
        with pytest.raises(AssertionError, match="branch tag"):
            M.build_one(key)
    finally:
        del M.BUILDERS[key]


def test_negative_control_a_leaked_adaptation_bank_is_refused_by_the_generator() -> None:
    import materialize_configs as M

    leaked = f"{M.MANIFESTS}/elure/graph_5_5_elure_5000.txt"
    for label, plan in [
        ("elure", lambda seed=M.DEFAULT_SEED: M._adapt("near", leaked, M.B_NEAR_VAL, seed)),
        ("corpus", lambda seed=M.DEFAULT_SEED: M._adapt("far", M.CORPUS_TRAIN, M.B_FAR_VAL, seed)),
    ]:
        key = f"_leak_{label}_probe.yaml"
        M.BUILDERS[key] = plan
        try:
            with pytest.raises(AssertionError):
                M.build_one(key)
        finally:
            del M.BUILDERS[key]


def test_negative_control_a_base_run_pointed_off_the_frozen_corpus_is_refused() -> None:
    import materialize_configs as M

    original = M._common_lurestar
    M._common_lurestar = lambda model, seed: (
        [o for o in original(model, seed) if o.key != "data.stargraph_train_data_path"]
        + [M.Ov("data.stargraph_train_data_path", M.B_NEAR, "MANIFEST", "none", "wrong pool")]
    )
    try:
        with pytest.raises(AssertionError, match="frozen base corpus"):
            M.build_one("gpt_lurestar.yaml")
    finally:
        M._common_lurestar = original


# --------------------------------------------------------------------------------------
# 5. HMM  (spec section 12)
# --------------------------------------------------------------------------------------

HMM_SPEC = {
    "data.dataset": "hmm_belief",
    "data.effective_batch_size": 256,
    "data.hmm.train_sequences": 100000,
    "data.hmm.sequence_length": 32,
    "model.n_layer": 4,
    "model.n_head": 4,
    "model.n_embd": 128,
    "trainer.train_batches": 3000,
    "trainer.val_interval": 300,
    "trainer.compile": False,
}


@pytest.mark.parametrize("name", HMM)
@pytest.mark.parametrize("key,expected", sorted(HMM_SPEC.items()))
def test_hmm_resolves_to_spec_section_12(merged, name, key, expected) -> None:
    got = get_dotted(merged[name], key)
    assert got == expected and type(got) is type(expected), f"{name}: {key} = {got!r}"


def test_hmm_nextlat_objective(merged) -> None:
    m = merged["nextlat_hmm.yaml"]["model"]
    assert m["mtp_horizon"] == 1
    assert m["lambda_mse"] == 1.0
    assert m["lambda_kl"] == 0.0
    assert m["lambda_ce"] == 0.0
    assert m["proj_factor"] == 0.5
    # proj_factor 0.5 at n_embd 128 -> dynamics MLP hidden 128
    assert dynamics_hidden_dim(128, 0.5) == 128


def test_hmm_gpt_and_nextlat_are_architecture_matched(merged) -> None:
    gpt, nl = merged["gpt_hmm.yaml"], merged["nextlat_hmm.yaml"]
    for key in ["model.n_layer", "model.n_head", "model.n_embd", "model.bias",
                "model.dropout", "data.effective_batch_size", "trainer.train_batches",
                "optimizer.learning_rate", "optimizer.weight_decay", "optimizer.grad_clip",
                "lr_scheduler.schedule"]:
        assert get_dotted(gpt, key) == get_dotted(nl, key), key
    assert get_dotted(gpt, "data.hmm") == get_dotted(nl, "data.hmm")


def test_hmm_optimizer_is_the_path_star_optimizer(merged) -> None:
    """Spec section 12 overrides no optimizer key, so the Path-Star optimizer block must
    survive verbatim into both HMM configs."""
    official = load_yaml_as_trainer_sees_it(OFFICIAL_GPT_5_5)
    for name in HMM:
        assert merged[name]["optimizer"] == deep_merge(
            load_yaml_as_trainer_sees_it(DEFAULTS_YAML)["optimizer"], official["optimizer"]
        )


def test_hmm_dataset_needs_a_registered_datamodule(merged) -> None:
    """train.py:176-178 asserts membership in DATAMODULES. `hmm_belief` is NOT registered at
    the pinned commit; this test pins that fact so the launch path cannot forget the
    one-line registration recorded in docs/CONFIG_DEVIATIONS.md."""
    text = (Path(UPSTREAM) / "train.py").read_text()
    assert '"hmm_belief"' not in text and "'hmm_belief'" not in text
    for name in HMM:
        assert merged[name]["data"]["dataset"] == "hmm_belief"


# --------------------------------------------------------------------------------------
# 6. output roots, paths, filenames
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", DELIVERABLE_CONFIGS)
def test_out_dir_is_absolute_and_unique(merged, name: str) -> None:
    out_dir = merged[name]["trainer"]["out_dir"]
    assert os.path.isabs(out_dir), (
        "the resume pointer stores the checkpoint path as written (core_train.py:944-948), "
        "so a relative out_dir makes resume depend on the launching CWD"
    )
    others = [merged[o]["trainer"]["out_dir"] for o in DELIVERABLE_CONFIGS if o != name]
    assert out_dir not in others


@pytest.mark.parametrize("name", DELIVERABLE_CONFIGS)
def test_experiment_name_suppresses_the_automatic_seed_suffix(merged, name: str) -> None:
    """train.py:96-97 appends '-seed{n}' unless the name already contains 'seed'."""
    exp = merged[name]["trainer"]["experiment_name"]
    assert "seed" in exp, exp
    assert str(merged[name]["seed"]) in exp


@pytest.mark.parametrize("name", LURESTAR + ADAPT)
def test_stargraph_paths_survive_the_filename_parser(merged, name: str) -> None:
    """data/stargraph.py:187-190 does `path.split("_")[1]` and `[2]` on the FULL path, so a
    single underscore anywhere in the directory chain breaks the parse."""
    for key in ["stargraph_train_data_path", "stargraph_test_data_path"]:
        path = merged[name]["data"][key]
        parts = path.split("_")
        assert parts[1] == "5" and parts[2] == "5", f"{name}: {key} -> {path}"
        assert "_" not in os.path.dirname(path), f"{name}: directory chain has an underscore"
        assert os.path.basename(path).startswith("graph_5_5_")


def test_lurestar_points_at_the_frozen_corpus(merged) -> None:
    manifest = json.loads((REPO / "manifests" / "corpus_provenance.json").read_text())
    assert manifest["params"]["num_samples"] == 200000
    assert manifest["params"]["num_test_samples"] == 20000
    assert manifest["params"]["max_nodes"] == 100
    for name in LURESTAR:
        train = merged[name]["data"]["stargraph_train_data_path"]
        test = merged[name]["data"]["stargraph_test_data_path"]
        assert os.path.basename(train) == "graph_5_5_sample_200000.txt"
        assert os.path.basename(test) == "graph_5_5_test_20000.txt"


def test_wandb_is_off_everywhere(merged) -> None:
    """train.py:15,17,24 import wandb unconditionally and defaults.yaml:34 turns it on; the
    5_5 YAMLs never override it, so an unmodified run would try to reach the network."""
    for name, cfg in merged.items():
        assert cfg["trainer"]["log_to_wandb"] is False, name
        assert cfg["trainer"]["log_to_file"] is True, name


# --------------------------------------------------------------------------------------
# 7. the audit record and the deviations document
# --------------------------------------------------------------------------------------


def test_every_override_is_documented(audit) -> None:
    doc = (REPO / "docs" / "CONFIG_DEVIATIONS.md").read_text()
    for entry in audit["configs"]:
        for ov in entry["overrides"]:
            if ov["category"] == "PROVENANCE":
                continue
            assert ov["key"] in doc, (
                f"{entry['config']}: override {ov['key']} is not mentioned in "
                f"docs/CONFIG_DEVIATIONS.md"
            )


def test_gradient_accumulation_fallback_rule_is_written_down() -> None:
    # whitespace-normalized: the phrases must survive markdown line wrapping
    doc = " ".join((REPO / "docs" / "CONFIG_DEVIATIONS.md").read_text().lower().split())
    for phrase in ["gradient accumulation", "gradient_accum_steps", "execution fallback",
                   "effective batch size", "512", "optimizer-update count",
                   "documented as a deviation"]:
        assert phrase in doc, phrase


def test_audit_records_the_source_hashes(audit) -> None:
    from config_lib import sha256_file
    for entry in audit["configs"]:
        src = os.path.join(UPSTREAM, entry["source_config"])
        assert entry["source_config_sha256"] == sha256_file(src)
        assert entry["defaults_sha256"] == sha256_file(DEFAULTS_YAML)
    assert audit["preregistered_seeds"] == [1234, 1235, 1236]


def test_every_frozen_exemption_names_a_spec_section(audit) -> None:
    for ex in audit["frozen_exemptions"]:
        assert re.search(r"spec sec\.\d+", ex["authority"]), ex


# --------------------------------------------------------------------------------------
# 8. negative controls -- these prove the checks above can fail
# --------------------------------------------------------------------------------------


def test_negative_control_dropping_the_sweep_reverts_proj_factor() -> None:
    """The exact mistake docs/UPSTREAM_REPORT.md calls the highest-risk footgun in the repo:
    delete the sweep block to run one seed, and proj_factor silently becomes 1.0."""
    official = load_yaml_as_trainer_sees_it(OFFICIAL_NEXTLAT_5_5)
    naive = copy.deepcopy(official)
    naive.pop("sweep")
    merged_naive = deep_merge(load_yaml_as_trainer_sees_it(DEFAULTS_YAML), naive)
    assert merged_naive["model"]["proj_factor"] == 1.0
    assert dynamics_hidden_dim(384, merged_naive["model"]["proj_factor"]) == 768
    # our config does not have this defect
    ours = _merged("nextlat_lurestar.yaml")
    assert ours["model"]["proj_factor"] == 0.5


def test_negative_control_missing_test_generalization_is_detected() -> None:
    """Reproduces the RUNLOG attempt-2 failure and asserts the key-coverage check catches it."""
    cfg = load_yaml_as_trainer_sees_it(os.path.join(CONFIGS_DIR, "gpt_lurestar.yaml"))
    del cfg["data"]["test_generalization"]
    merged_bad = deep_merge(load_yaml_as_trainer_sees_it(DEFAULTS_YAML), cfg)
    required = _accesses(_ON_PATH + _STARGRAPH_ONLY) - set(_NOT_REQUIRED)
    unresolvable = sorted(k for k in required if not has_dotted(merged_bad, k))
    assert unresolvable == ["data.test_generalization"]


def test_negative_control_compile_true_is_rejected() -> None:
    official = load_yaml_as_trainer_sees_it(OFFICIAL_GPT_5_5)
    assert official["trainer"]["compile"] is True, "upstream really does ship compile: true"
    assert PAPER_SCALE["trainer.compile"] is False
    bad = deep_merge(load_yaml_as_trainer_sees_it(DEFAULTS_YAML),
                     load_yaml_as_trainer_sees_it(os.path.join(CONFIGS_DIR, "gpt_lurestar.yaml")))
    bad["trainer"]["compile"] = True
    with pytest.raises(AssertionError):
        got = get_dotted(bad, "trainer.compile")
        assert got == PAPER_SCALE["trainer.compile"]


def test_negative_control_shared_output_root_is_rejected() -> None:
    near = load_yaml(os.path.join(CONFIGS_DIR, "adapt_near.yaml"))
    far = load_yaml(os.path.join(CONFIGS_DIR, "adapt_far.yaml"))
    far["trainer"]["out_dir"] = near["trainer"]["out_dir"]
    with pytest.raises(AssertionError):
        assert near["trainer"]["out_dir"] != far["trainer"]["out_dir"]


def test_negative_control_underscore_in_directory_breaks_the_parser() -> None:
    bad = "/content/lure_star/data/stargraph/graph_5_5_sample_200000.txt"
    parts = bad.split("_")
    assert not (parts[1] == "5" and parts[2] == "5"), (
        "if this ever passes the filename-parser guard is vacuous"
    )


def test_negative_control_a_shuffled_override_table_fails_the_check(tmp_path) -> None:
    """Mutating one emitted config makes `materialize_configs.py --check` fail."""
    target = Path(CONFIGS_DIR) / "gpt_lurestar.yaml"
    original = target.read_text()
    try:
        target.write_text(original.replace("compile: false", "compile: true"))
        proc = subprocess.run(
            [PYTHON, str(REPO / "scripts" / "materialize_configs.py"), "--check"],
            capture_output=True, text=True, cwd=str(REPO))
        assert proc.returncode != 0
        assert "gpt_lurestar.yaml" in proc.stderr
    finally:
        target.write_text(original)


def test_negative_control_yaml_float_resolver_matters() -> None:
    """Plain pyyaml reads the shipped `learning_rate: 5e-4` as a string; OmegaConf does not.
    If this ever stops being true the paper-scale learning-rate assertion is vacuous."""
    plain = yaml.safe_load("x: 5e-4")["x"]
    assert plain == "5e-4" and isinstance(plain, str)
    from config_lib import OmegaConfCompatLoader
    fixed = yaml.load("x: 5e-4", Loader=OmegaConfCompatLoader)["x"]
    assert isinstance(fixed, float) and abs(fixed - 5e-4) < 1e-12
