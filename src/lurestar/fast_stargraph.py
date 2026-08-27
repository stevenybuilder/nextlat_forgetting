"""Pre-tokenize the Path-Star corpus once instead of re-tokenizing every item, every epoch.

WHY THIS EXISTS
---------------
`upstream/NextLat/data/stargraph.py` keeps the corpus as a list of 200,000 raw text lines and
tokenizes one line per `__getitem__`, in a pure-Python character loop
(`Tokenizer.encode`: a `while` over every character, building a `str` digit by digit).  The
DataLoader it builds passes no `num_workers`, so that loop runs on the training thread and is
strictly additive to GPU time: `t_step = t_loader + t_gpu`.  At effective batch 512 that is 512
executions of the loop between every optimizer step, for 20,000 steps, in every one of the
fifteen confirmatory jobs -- and it recomputes the *same* answer each time, because tokenizing a
line is a pure function of the line.

WHAT THIS CHANGES
-----------------
Nothing about the experiment.  The corpus, the item order, the batch composition, the sampler,
the RNG streams, the batch size and the update count are untouched.  `__getitem__` returns the
same `torch.long` tensor it returned before; it is read out of a matrix computed once instead of
recomputed 20,000 times.  This is memoization of a pure function, and it is verified as such:
`build_token_matrix` uses a vectorized parse, and `verify_against_upstream` then re-tokenizes
**every** line with upstream's own `Tokenizer` and asserts exact equality before the matrix is
allowed to be used.  A mismatch raises; it does not warn.

The cache is keyed on the SHA-256 of the corpus file, so a different corpus can never be served
from a stale matrix.

FROZEN SURFACE
--------------
PROGRAM.md forbids changing width, depth, sequence construction, precision, loss, effective
batch size or the update count to gain throughput.  None of those are touched here.  The one
thing this does touch -- sequence construction -- it touches by *asserting* it is unchanged,
byte for byte, on all 220,000 lines.
"""
from __future__ import annotations

import hashlib
import os
import pathlib
import typing as t

import numpy as np

__all__ = [
    "build_token_matrix",
    "verify_against_upstream",
    "materialize",
    "install",
    "uninstall",
]

CACHE_VERSION = 1
_ORIGINAL_GETITEM: t.Any = None


# ----------------------------------------------------------------------------- hashing


def sha256_file(path: os.PathLike | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_lines(lines: t.Sequence[str]) -> str:
    """Hash the *content the dataset actually holds*, so the cache key survives a file move."""
    h = hashlib.sha256()
    for line in lines:
        h.update(line.strip().encode())
        h.update(b"\n")
    return h.hexdigest()


# ----------------------------------------------------------------------------- the parse


def build_token_matrix(lines: t.Sequence[str], tokenizer) -> np.ndarray:
    """Vectorized equivalent of `[tokenizer.tokenize(l.strip())[0] for l in lines]`.

    The structure is read off the first line with upstream's own tokenizer rather than assumed:
    every position whose token id is a special (>= maxNodes) is held constant across rows, and
    every other position is filled from the integers scanned out of the raw bytes.  If the
    corpus were not structurally homogeneous the reshape below would raise -- and it has to be
    homogeneous anyway, because `collate_fn` does a bare `torch.stack`.
    """
    max_nodes = tokenizer.maxNodes
    ref = np.asarray(tokenizer.tokenize(lines[0].strip())[0], dtype=np.int64)
    width = int(ref.shape[0])
    special = ref >= max_nodes
    n_numeric = int((~special).sum())

    n = len(lines)
    buf = "\n".join(line.strip() for line in lines).encode("ascii")
    a = np.frombuffer(buf, dtype=np.uint8)

    is_digit = (a >= 48) & (a <= 57)
    prev = np.empty_like(is_digit)
    prev[0] = False
    prev[1:] = is_digit[:-1]
    nxt = np.empty_like(is_digit)
    nxt[-1] = False
    nxt[:-1] = is_digit[1:]

    starts = np.flatnonzero(is_digit & ~prev)
    ends = np.flatnonzero(is_digit & ~nxt) + 1
    lengths = ends - starts

    values = np.zeros(starts.shape[0], dtype=np.int64)
    for digits in np.unique(lengths):
        sel = lengths == digits
        s = starts[sel]
        acc = np.zeros(int(sel.sum()), dtype=np.int64)
        for k in range(int(digits)):
            acc = acc * 10 + (a[s + k].astype(np.int64) - 48)
        values[sel] = acc

    if values.shape[0] != n * n_numeric:
        raise ValueError(
            f"corpus is not structurally homogeneous: scanned {values.shape[0]} integers, "
            f"expected {n} lines x {n_numeric} numeric positions"
        )

    out = np.empty((n, width), dtype=np.int16)
    out[:, special] = ref[special].astype(np.int16)
    out[:, ~special] = values.reshape(n, n_numeric).astype(np.int16)
    return out


def verify_against_upstream(matrix: np.ndarray, lines: t.Sequence[str], tokenizer) -> int:
    """Re-tokenize EVERY line with upstream's tokenizer and assert exact equality.

    This is the whole warrant for the optimization, so it is not sampled and it is not
    tolerance-based.  Returns the number of lines checked.
    """
    if matrix.shape[0] != len(lines):
        raise AssertionError(
            f"token matrix has {matrix.shape[0]} rows for {len(lines)} lines"
        )
    for i, line in enumerate(lines):
        expected = np.asarray(tokenizer.tokenize(line.strip())[0], dtype=np.int64)
        got = matrix[i].astype(np.int64)
        if expected.shape != got.shape or not np.array_equal(expected, got):
            raise AssertionError(
                f"pre-tokenization diverged from upstream at line {i}:\n"
                f"  line     = {line.strip()!r}\n"
                f"  upstream = {expected.tolist()}\n"
                f"  fast     = {got.tolist()}"
            )
    return len(lines)


# ----------------------------------------------------------------------------- cache


def _cache_path(cache_dir: os.PathLike | str, digest: str) -> pathlib.Path:
    return pathlib.Path(cache_dir) / f"stargraph_tokens_v{CACHE_VERSION}_{digest[:16]}.npz"


def materialize(
    lines: t.Sequence[str],
    tokenizer,
    *,
    cache_dir: os.PathLike | str | None = None,
    verify: bool = True,
    log: t.Callable[[str], None] = print,
) -> np.ndarray:
    """Build (or load) the verified token matrix for `lines`.

    A cache hit still checks the digest of the lines in hand, so a matrix built for a different
    corpus can never be served.  A cache hit skips re-verification because the file it loads was
    only ever written after a full verification passed; the digest is what carries that forward.
    """
    digest = sha256_lines(lines)
    path = _cache_path(cache_dir, digest) if cache_dir else None

    if path is not None and path.is_file():
        try:
            blob = np.load(path)
            if str(blob["digest"]) == digest:
                log(f"[fast_stargraph] cache hit {path.name} ({blob['tokens'].shape})")
                return blob["tokens"]
            log(f"[fast_stargraph] cache digest mismatch at {path.name}; rebuilding")
        except Exception as exc:  # noqa: BLE001 - a bad cache is rebuilt, never trusted
            log(f"[fast_stargraph] unreadable cache {path.name}: {exc!r}; rebuilding")

    matrix = build_token_matrix(lines, tokenizer)
    if verify:
        n = verify_against_upstream(matrix, lines, tokenizer)
        log(f"[fast_stargraph] verified {n} lines against upstream Tokenizer, exact match")

    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".npz.partial")
        # Passing ``tmp`` directly makes NumPy append ``.npz`` because the partial
        # filename does not end with that suffix.  Write through a file object so the
        # bytes land at the exact same-directory path that is atomically published below.
        with tmp.open("wb") as fh:
            np.savez(fh, tokens=matrix, digest=np.array(digest))
        os.replace(tmp, path)
        log(f"[fast_stargraph] cached -> {path}")
    return matrix


# ----------------------------------------------------------------------------- install


def install(*, cache_dir: os.PathLike | str | None = None,
            verify: bool = True,
            log: t.Callable[[str], None] = print) -> bool:
    """Monkeypatch `StarGraphDataset.__getitem__` in memory.  Upstream on disk is untouched.

    Returns False and changes nothing when `LURESTAR_FAST_LOADER=0`, so the optimization can be
    switched off for an A/B without editing anything.
    """
    global _ORIGINAL_GETITEM
    if os.environ.get("LURESTAR_FAST_LOADER", "1") == "0":
        log("[fast_stargraph] disabled by LURESTAR_FAST_LOADER=0")
        return False

    import torch
    from data.stargraph import StarGraphDataset  # upstream, imported not edited

    if _ORIGINAL_GETITEM is not None:
        return True
    _ORIGINAL_GETITEM = StarGraphDataset.__getitem__

    def _fast_getitem(self, idx):
        cached = getattr(self, "_lurestar_tokens", None)
        if cached is None:
            matrix = materialize(
                self.data, self.tokenizer, cache_dir=cache_dir, verify=verify, log=log
            )
            cached = torch.from_numpy(matrix.astype(np.int64))
            self._lurestar_tokens = cached
        return cached[idx]

    StarGraphDataset.__getitem__ = _fast_getitem
    log(f"[fast_stargraph] installed (cache_dir={cache_dir}, verify={verify})")
    return True


def uninstall() -> None:
    global _ORIGINAL_GETITEM
    if _ORIGINAL_GETITEM is None:
        return
    from data.stargraph import StarGraphDataset

    StarGraphDataset.__getitem__ = _ORIGINAL_GETITEM
    _ORIGINAL_GETITEM = None
