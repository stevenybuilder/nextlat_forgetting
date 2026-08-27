"""The pre-tokenized loader is only legitimate if it is provably the same function.

Every test here compares against upstream's own `Tokenizer`, extracted from
`upstream/NextLat/data/stargraph.py` by AST rather than reimplemented -- a reimplementation
would be testing the copy against itself.  Upstream is never imported as a module because it
pulls in torch, lightning and sklearn, none of which exist on this host.
"""
from __future__ import annotations

import ast
import pathlib

import numpy as np
import pytest

from lurestar import fast_stargraph as fs

ROOT = pathlib.Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "upstream" / "NextLat" / "data" / "stargraph.py"
CORPUS = ROOT / "data" / "stargraph" / "graph_5_5_sample_200000.txt"
HELDOUT = ROOT / "data" / "stargraph" / "graph_5_5_test_20000.txt"


def upstream_tokenizer(max_nodes: int = 100):
    """Load upstream's Tokenizer class alone, by AST, with no torch import."""
    tree = ast.parse(UPSTREAM.read_text())
    node = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Tokenizer"
    )
    ns: dict = {"np": np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(UPSTREAM), "exec"), ns)
    return ns["Tokenizer"](max_nodes)


@pytest.fixture(scope="module")
def tok():
    return upstream_tokenizer()


@pytest.fixture(scope="module")
def lines():
    if not CORPUS.is_file():
        pytest.skip("corpus not generated")
    return CORPUS.read_text().splitlines(keepends=True)


def test_matrix_matches_upstream_on_every_training_line(tok, lines):
    """The load-bearing claim: 200,000 rows, exact equality, no sampling, no tolerance."""
    m = fs.build_token_matrix(lines, tok)
    assert m.shape == (len(lines), 69)
    assert fs.verify_against_upstream(m, lines, tok) == len(lines)


def test_matrix_matches_upstream_on_the_heldout_split(tok):
    if not HELDOUT.is_file():
        pytest.skip("held-out split not generated")
    held = HELDOUT.read_text().splitlines(keepends=True)
    m = fs.build_token_matrix(held, tok)
    assert fs.verify_against_upstream(m, held, tok) == len(held)


def test_dtype_holds_every_token_id(tok, lines):
    """int16 is a storage choice, not a truncation: assert the vocabulary fits with room."""
    m = fs.build_token_matrix(lines[:5000], tok)
    assert m.dtype == np.int16
    assert int(m.max()) == tok.eos_token_id
    assert int(m.min()) >= 0
    assert int(m.max()) < np.iinfo(np.int16).max


def test_verification_catches_a_corrupted_matrix(tok, lines):
    """The falsifier. If `verify_against_upstream` cannot fail, it is not evidence."""
    sub = lines[:2000]
    m = fs.build_token_matrix(sub, tok)
    m[7, 3] = (m[7, 3] + 1) % 100
    with pytest.raises(AssertionError, match="diverged from upstream at line 7"):
        fs.verify_against_upstream(m, sub, tok)


def test_heterogeneous_corpus_is_refused(tok):
    """A corpus whose lines do not share a token layout must raise, not silently reshape."""
    bad = ["1,2|3,4/1,4=1,2,4\n", "1,2|3,4|5,6/1,6=1,2,4,6\n"]
    with pytest.raises(ValueError, match="not structurally homogeneous"):
        fs.build_token_matrix(bad, tok)


def test_cache_roundtrip_and_digest_guard(tok, lines, tmp_path):
    sub = lines[:3000]
    first = fs.materialize(sub, tok, cache_dir=tmp_path, log=lambda *_: None)
    # np.savez appends ".npz" to string/path targets without that suffix.  The cache
    # writer must instead publish the exact .partial path atomically and leave no stray
    # ".npz.partial.npz" artifact behind.
    assert not list(tmp_path.glob("*.partial*"))
    second = fs.materialize(sub, tok, cache_dir=tmp_path, log=lambda *_: None)
    assert np.array_equal(first, second)
    assert len(list(tmp_path.glob("stargraph_tokens_v*.npz"))) == 1

    # A different corpus must not be served from the first corpus's cache.
    other = lines[3000:6000]
    third = fs.materialize(other, tok, cache_dir=tmp_path, log=lambda *_: None)
    assert not np.array_equal(first, third)
    assert len(list(tmp_path.glob("stargraph_tokens_v*.npz"))) == 2
    assert not list(tmp_path.glob("*.partial*"))


def test_digest_is_whitespace_insensitive_the_same_way_getitem_is(tok, lines):
    """`__getitem__` strips, so a trailing newline must not change the cache key."""
    a = fs.sha256_lines(lines[:100])
    b = fs.sha256_lines([ln.rstrip("\n") for ln in lines[:100]])
    assert a == b
