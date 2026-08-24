#!/usr/bin/env python
"""Generate the paper-scale Path-Star corpus, byte-identical to upstream, in parallel.

Upstream `data/stargraph/prepare.py` reseeds the global RNG once per sample with a counter
that starts at 0 and continues across the train/test split. That makes sample *i* a pure
function of *i*, so the corpus is embarrassingly parallel AND the parallel output is
byte-identical to the serial output -- which this script asserts rather than assumes.
Determinism that depends on the worker count would be a bug, not an optimization.

Serial upstream also prints a 50-character progress bar per sample, which dominates its
runtime and floods any log relay. That is dropped.
"""
import argparse
import hashlib
import importlib.util
import multiprocessing as mp
import pathlib
import random
import sys
import time
import types

ROOT = pathlib.Path(__file__).resolve().parent.parent
PREPARE = ROOT / "upstream" / "NextLat" / "data" / "stargraph" / "prepare.py"


def _load_upstream_generator():
    """Load upstream's graph maker from its file, without importing the repo package.

    Two obstacles, both incidental. `upstream/NextLat/data/stargraph.py` shadows the
    `data/stargraph/` package on a normal import, and `prepare.py` imports torch at module
    scope even though graph generation never touches it -- and this host has no torch. So
    the module is loaded by path with a stub standing in for torch. The generator function
    itself is upstream's, unmodified: using a reimplementation here would risk a corpus
    that differs from the paper's in some detail nobody notices until the results are in.
    """
    sys.modules.setdefault("torch", types.ModuleType("torch"))
    spec = importlib.util.spec_from_file_location("_upstream_prepare", PREPARE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.star_or_sink_graph_maker


star_or_sink_graph_maker = _load_upstream_generator()


def render(seed, num_paths, path_len, max_nodes):
    """Reproduce one upstream line exactly (prepare.py:59-73)."""
    random.seed(seed)
    edges, path, source, goal = star_or_sink_graph_maker(
        num_paths, path_len, max_nodes, True, False
    )
    return (
        "|".join(",".join(str(i) for i in e) for e in edges)
        + "/%d,%d=%s" % (source, goal, ",".join(str(i) for i in path))
    )


def _shard(args):
    lo, hi, num_paths, path_len, max_nodes = args
    return "\n".join(render(s, num_paths, path_len, max_nodes) for s in range(lo, hi))


def build(first_seed, count, num_paths, path_len, max_nodes, workers):
    if count == 0:
        return ""
    step = max(1, -(-count // (workers * 4)))
    jobs = [
        (lo, min(lo + step, first_seed + count), num_paths, path_len, max_nodes)
        for lo in range(first_seed, first_seed + count, step)
    ]
    if workers == 1:
        parts = [_shard(j) for j in jobs]
    else:
        with mp.Pool(workers) as pool:
            parts = pool.map(_shard, jobs)
    return "\n".join(parts) + "\n"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_samples", type=int, default=200000)
    ap.add_argument("--num_test_samples", type=int, default=20000)
    ap.add_argument("--num_paths", type=int, default=5)
    ap.add_argument("--path_len", type=int, default=5)
    ap.add_argument("--max_nodes", type=int, default=100)
    ap.add_argument("--data_dir", default=str(ROOT / "data" / "stargraph"))
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--verify_serial", type=int, default=2000,
                    help="assert the first N lines match a single-process render")
    a = ap.parse_args()

    out = pathlib.Path(a.data_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = "%d_%d" % (a.num_paths, a.path_len)
    train_p = out / ("graph_%s_sample_%d.txt" % (tag, a.num_samples))
    test_p = out / ("graph_%s_test_%d.txt" % (tag, a.num_test_samples))

    t0 = time.time()
    train = build(0, a.num_samples, a.num_paths, a.path_len, a.max_nodes, a.workers)
    train_p.write_text(train)
    t1 = time.time()
    test = build(a.num_samples, a.num_test_samples, a.num_paths, a.path_len,
                 a.max_nodes, a.workers)
    test_p.write_text(test)
    t2 = time.time()

    # Correctness gate: the parallel output must equal the single-process rendering.
    n = min(a.verify_serial, a.num_samples)
    expected = [render(s, a.num_paths, a.path_len, a.max_nodes) for s in range(n)]
    actual = train.split("\n")[:n]
    if expected != actual:
        bad = next(i for i in range(n) if expected[i] != actual[i])
        raise SystemExit("PARALLEL OUTPUT DIVERGED at line %d" % bad)

    print("train %s  %d lines  %.1fs" % (train_p.name, a.num_samples, t1 - t0))
    print("test  %s  %d lines  %.1fs" % (test_p.name, a.num_test_samples, t2 - t1))
    print("workers=%d  verified_identical_lines=%d" % (a.workers, n))
    for p in (train_p, test_p):
        print("sha256  %s  %s  %d bytes" % (sha256(p), p.name, p.stat().st_size))


if __name__ == "__main__":
    main()
