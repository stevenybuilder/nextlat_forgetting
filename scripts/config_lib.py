"""Shared config primitives: OmegaConf-compatible merge, dotted-key access, upstream paths.

No torch, no omegaconf. Runs under the analysis venv (pyyaml only).

`deep_merge` reproduces `OmegaConf.merge` semantics for the non-struct plain-dict configs
this repository uses: dict nodes merge recursively, every other node (scalar, list, None)
is replaced wholesale by the right-hand operand. That is exactly what
`train.py:348-351` does with `OmegaConf.merge(default, base_config, cli)`.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Dict, Iterable, List, Tuple

import yaml


class OmegaConfCompatLoader(yaml.SafeLoader):
    """SafeLoader with the float resolver OmegaConf installs.

    PyYAML follows YAML 1.1, whose implicit float pattern requires a decimal point before an
    exponent, so plain `yaml.safe_load` reads the shipped `learning_rate: 5e-4` as the STRING
    '5e-4'. OmegaConf replaces that resolver (omegaconf/_utils.py, `get_yaml_loader`) with a
    pattern whose second alternative accepts `[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+`, so upstream
    reads 5e-4 as the float 0.0005.

    The generator deliberately uses plain SafeLoader/SafeDumper, which round-trips `5e-4`
    verbatim and keeps the emitted file a literal copy of the official text. This loader is
    used only where a check must see the value the trainer will actually see.
    """


OmegaConfCompatLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:
     [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
    |[-+]?\.(?:inf|Inf|INF)
    |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def load_yaml_as_trainer_sees_it(path: str) -> Dict[str, Any]:
    """Parse a config the way OmegaConf will parse it inside train.py."""
    with open(path, "r") as fh:
        obj = yaml.load(fh, Loader=OmegaConfCompatLoader)
    if not isinstance(obj, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return obj

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.join(REPO_ROOT, "upstream", "NextLat")
UPSTREAM_COMMIT = "3770be6009cea2b3c455a9ce7f2ca88b504bb955"

DEFAULTS_YAML = os.path.join(UPSTREAM, "defaults.yaml")
OFFICIAL_GPT_5_5 = os.path.join(UPSTREAM, "config/stargraph/5_5/gpt_stargraph_5_5.yaml")
OFFICIAL_NEXTLAT_5_5 = os.path.join(
    UPSTREAM, "config/stargraph/5_5/nextlat_stargraph_5_5.yaml"
)
# Third arm, spec sec.8 / docs/DECISION_D20_competence_gate.md "Superseded in part". The
# official BST G(5,5) YAML differs from the GPT one by exactly two scientific keys --
# `use_bst: true` and `model.bst_pair_minimum_gap: 2` -- plus `experiment_name`.
OFFICIAL_BST_5_5 = os.path.join(UPSTREAM, "config/stargraph/5_5/bst_stargraph_5_5.yaml")
OFFICIAL_BST_5_5 = os.path.join(UPSTREAM, "config/stargraph/5_5/bst_stargraph_5_5.yaml")

CONFIGS_DIR = os.path.join(REPO_ROOT, "configs")


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as fh:
        obj = yaml.safe_load(fh)
    if not isinstance(obj, dict):
        raise ValueError(f"{path} did not parse to a mapping (got {type(obj).__name__})")
    return obj


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def deep_merge(base: Any, override: Any) -> Any:
    """OmegaConf.merge semantics for plain dicts: recurse into dicts, replace everything else."""
    if isinstance(base, dict) and isinstance(override, dict):
        out = dict(base)
        for key, value in override.items():
            out[key] = deep_merge(out[key], value) if key in out else value
        return out
    return override


def flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested mapping into dotted keys. Leaf lists are kept whole."""
    flat: Dict[str, Any] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            dotted = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                flat.update(flatten(value, dotted))
            else:
                flat[dotted] = value
    else:
        flat[prefix] = obj
    return flat


def get_dotted(obj: Dict[str, Any], dotted: str) -> Any:
    cur: Any = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(dotted)
        cur = cur[part]
    return cur


def has_dotted(obj: Dict[str, Any], dotted: str) -> bool:
    try:
        get_dotted(obj, dotted)
    except KeyError:
        return False
    return True


def set_dotted(obj: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    cur = obj
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def del_dotted(obj: Dict[str, Any], dotted: str) -> None:
    parts = dotted.split(".")
    cur = obj
    for part in parts[:-1]:
        cur = cur[part]
    del cur[parts[-1]]


def missing_keys(reference: Dict[str, Any], candidate: Dict[str, Any]) -> List[str]:
    """Dotted keys present in `reference` but absent from `candidate`."""
    ref = flatten(reference)
    return sorted(k for k in ref if not has_dotted(candidate, k))


def diff_keys(a: Dict[str, Any], b: Dict[str, Any]) -> List[Tuple[str, Any, Any]]:
    """Dotted keys whose values differ (or that exist on only one side)."""
    fa, fb = flatten(a), flatten(b)
    out: List[Tuple[str, Any, Any]] = []
    for key in sorted(set(fa) | set(fb)):
        va, vb = fa.get(key, _MISSING), fb.get(key, _MISSING)
        if va != vb:
            out.append((key, None if va is _MISSING else va, None if vb is _MISSING else vb))
    return out


class _Missing:
    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return "<missing>"


_MISSING = _Missing()


# --- resolved-architecture arithmetic, transcribed from the pinned code ------------------


def dynamics_hidden_dim(n_embd: int, proj_factor: float) -> int:
    """models/model_nextlat.py:50-52.

        input_dim  = config.n_embd * 2
        hidden_dim = config.proj_factor * input_dim
        hidden_dim = 128 * round(hidden_dim / 128)
    """
    input_dim = n_embd * 2
    hidden = proj_factor * input_dim
    return 128 * round(hidden / 128)


def dynamics_param_count(n_embd: int, proj_factor: float, bias: bool = False) -> int:
    """Parameters of NextLatDynamicsModel (models/model_nextlat.py:47-70), bias=False.

    mlp = Linear(2*n_embd -> h) + GELU + Linear(h -> h) + GELU + Linear(h -> n_embd)
    norm_x = LayerNorm(2*n_embd)  -> weight only when bias is False (model_base.py:815-830)
    """
    if bias:
        raise NotImplementedError("configs in this project all set model.bias: false")
    input_dim = n_embd * 2
    hidden = dynamics_hidden_dim(n_embd, proj_factor)
    linears = input_dim * hidden + hidden * hidden + hidden * n_embd
    layernorm = input_dim  # weight only
    return linears + layernorm


def swiglu_hidden_dim(n_embd: int) -> int:
    """models/model_gpt.py:139-140 / model_base.py MLP sizing: 128 * round((8*n_embd/3)/128)."""
    return 128 * round((8 * n_embd / 3) / 128)


def stargraph_vocab_size(max_nodes: int) -> int:
    """data/stargraph.py:233 -> maxNodes + 5 + 1."""
    return max_nodes + 5 + 1


# --- transformer parameter counts, transcribed from the pinned code ----------------------
#
# These exist so "architecture-matched" is a checkable claim rather than a sentence in a
# writeup. Every number below is derived from the pinned source, not measured, because this
# environment has no torch; `tests/test_configs.py` pins the arithmetic and the module
# line numbers it came from.


def block_param_count(n_embd: int, n_head: int, bias: bool = False) -> int:
    """One transformer Block: models/model_gpt.py:154-166.

    ln_1 + ln_2   LayerNorm(n_embd), weight only when bias is False (model_base.py:816-821)
    attn          c_attn Linear(n_embd -> 3*n_embd) + c_proj Linear(n_embd -> n_embd)
    mlp           SwiGLU(n_embd -> h) is Linear(n_embd -> 2h) (model_base.py:843)
                  + down Linear(h -> n_embd),  h = swiglu_hidden_dim(n_embd)

    `models/model_bst.py:28` imports this exact Block, so a BST layer and a GPT layer are the
    same object at the same width.
    """
    if bias:
        raise NotImplementedError("configs in this project all set model.bias: false")
    if n_embd % n_head:
        raise ValueError("model_gpt.py:54 asserts n_embd % n_head == 0")
    hidden = swiglu_hidden_dim(n_embd)
    attn = n_embd * (3 * n_embd) + n_embd * n_embd
    mlp = n_embd * (2 * hidden) + hidden * n_embd
    return 2 * n_embd + attn + mlp


def gpt_param_count(n_embd: int, n_head: int, n_layer: int, vocab_size: int,
                    bias: bool = False) -> int:
    """GPT / NextLat trunk: models/model_gpt.py:179-202.

    token_embedding + n_layer Blocks + final LayerNorm + lm_head. `lm_head` is a SEPARATE
    `nn.Linear(n_embd, vocab_size, bias=False)` -- it is NOT weight-tied to the embedding,
    which is why the total is one vocab_size*n_embd larger than a tied nanoGPT would be.
    `get_num_params(non_embedding=False)` (model_gpt.py:206-214) reports this number.
    """
    return (
        vocab_size * n_embd                                  # token_embedding
        + n_layer * block_param_count(n_embd, n_head, bias)   # transformer.blocks
        + n_embd                                             # transformer.norm
        + n_embd * vocab_size                                # lm_head
    )


def nextlat_param_count(n_embd: int, n_head: int, n_layer: int, vocab_size: int,
                        proj_factor: float, bias: bool = False) -> int:
    """NextLat = the GPT trunk plus the latent-dynamics MLP (models/model_nextlat.py:47-70)."""
    return (gpt_param_count(n_embd, n_head, n_layer, vocab_size, bias)
            + dynamics_param_count(n_embd, proj_factor, bias))


def bst_encoder_param_count(n_embd: int, n_head: int, n_layer: int, vocab_size: int,
                            bias: bool = False) -> int:
    """models/model_bst.py:134-162 `TransformerEncoder`.

    ONE shared token_embedding, then TWO independent stacks -- `transformer_f` and
    `transformer_b`, each `n_layer` Blocks plus its own final LayerNorm. There is no lm_head
    here; BST's output head lives in `TextHead`.
    """
    stack = n_layer * block_param_count(n_embd, n_head, bias) + n_embd
    return vocab_size * n_embd + 2 * stack


def bst_texthead_hidden_dim(n_embd: int) -> int:
    """models/model_bst.py:57-59: input_dim = 2*n_embd, h = 128*round((8*input_dim/3)/128)."""
    input_dim = n_embd * 2
    return 128 * round((8 * input_dim / 3) / 128)


def bst_texthead_param_count(n_embd: int, vocab_size: int, bias: bool = False) -> int:
    """models/model_bst.py:53-77 `TextHead`.

    SwiGLU(2*n_embd -> h) = Linear(2*n_embd -> 2h), Linear(h -> 2*n_embd),
    LayerNorm(2*n_embd) weight only, and lm_head Linear(n_embd -> vocab_size).
    The head is applied to the concatenated forward/backward embedding, whose halves are
    split again at model_bst.py:98 to give the next- and previous-token logits.
    """
    if bias:
        raise NotImplementedError("configs in this project all set model.bias: false")
    input_dim = n_embd * 2
    hidden = bst_texthead_hidden_dim(n_embd)
    return (input_dim * (2 * hidden)     # SwiGLU gate_up
            + hidden * input_dim         # down projection
            + input_dim                  # norm, weight only
            + n_embd * vocab_size)       # lm_head


def bst_param_count(n_embd: int, n_head: int, n_layer: int, vocab_size: int,
                    bias: bool = False) -> int:
    """models/model_bst.py:350-360 `BST.get_num_params(non_embedding=False)`."""
    return (bst_encoder_param_count(n_embd, n_head, n_layer, vocab_size, bias)
            + bst_texthead_param_count(n_embd, vocab_size, bias))


def bst_pairs_per_sequence(seq_len: int, context_length: int, min_gap: int,
                           max_gap: int = -1) -> int:
    """models/model_bst.py:362-391 `_create_pair_indices`, for a single-document sequence.

    start  = arange(context_length, seq_len)
    offset = arange(min_gap, max_gap)   with max_gap <- seq_len when max_gap <= 0
    keep pairs whose end index is < seq_len.

    Single-document is the StarGraph case: `data/stargraph.py:51-56` appends exactly one
    eos token, at the last position, so `_create_valid_pairs` keeps every pair
    (model_bst.py:430-437).
    """
    if max_gap <= 0:
        max_gap = seq_len
    return sum(
        1
        for start in range(context_length, seq_len)
        for offset in range(min_gap, max_gap)
        if start + offset < seq_len
    )


def bst_pair_accum_steps(n_pairs: int, pair_batch_size: int) -> int:
    """models/model_bst.py:600: `math.ceil(n_pairs / pair_batch_size)`.

    This is why `data.pair_batch_size` is not merely an execution knob: line 601 sets
    `texthead_loss_div = loss_div * batch_size * pair_accum_steps`, so once the pairs of one
    sequence stop fitting in one chunk the text-head loss becomes a mean of chunk means
    instead of the mean over pairs, and the gradient reweights.
    """
    import math

    return max(1, math.ceil(n_pairs / pair_batch_size))


def optimizer_updates(train_batches: int, start_step: int = 0) -> int:
    """core_train.py:564-571.

    The loop increments `self.step` after each optimizer update and returns when
    `self.step > train_batches`, so it performs `train_batches - start_step + 1` updates.
    """
    return train_batches - start_step + 1


def iter_yaml_paths(names: Iterable[str]) -> List[str]:
    return [os.path.join(CONFIGS_DIR, n) for n in names]


DELIVERABLE_CONFIGS = [
    "gpt_lurestar.yaml",
    "nextlat_lurestar.yaml",
    "bst_lurestar.yaml",
    "adapt_near.yaml",
    "adapt_far.yaml",
    "gpt_hmm.yaml",
    "nextlat_hmm.yaml",
]
