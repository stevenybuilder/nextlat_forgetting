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
from typing import Any, Dict, Iterable, List, Tuple

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPSTREAM = os.path.join(REPO_ROOT, "upstream", "NextLat")
UPSTREAM_COMMIT = "3770be6009cea2b3c455a9ce7f2ca88b504bb955"

DEFAULTS_YAML = os.path.join(UPSTREAM, "defaults.yaml")
OFFICIAL_GPT_5_5 = os.path.join(UPSTREAM, "config/stargraph/5_5/gpt_stargraph_5_5.yaml")
OFFICIAL_NEXTLAT_5_5 = os.path.join(
    UPSTREAM, "config/stargraph/5_5/nextlat_stargraph_5_5.yaml"
)

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
    "adapt_near.yaml",
    "adapt_far.yaml",
    "gpt_hmm.yaml",
    "nextlat_hmm.yaml",
]
