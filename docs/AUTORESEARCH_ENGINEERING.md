# Engineering autoloop — autoresearch with the scientific objective removed

Karpathy-style autoresearch has a clean loop: change one thing, run a fixed training budget,
measure the objective, keep improvements, discard regressions. That is useful when the objective is
the product. It is invalid for this project because PSI, interference, posterior decodability, and
every other scientific result are the measurements the confirmatory study is supposed to reveal.
Selecting code or configuration because one of those numbers improved would turn the study into a
hidden hyperparameter search.

`scripts/engineering_autoloop.py` keeps the useful mechanics and inverts the selection rule. It may
select only for engineering correctness: tests, config reproducibility, packaging, checkpoint
durability, and data-path throughput. It has no arbitrary command flag, never opens result-bearing
files, and cannot use PSI or another outcome as an acceptance signal.

## Contract

Each iteration has one declared candidate, one deadline, and one maximum check count. All three are
fixed by `begin`, before the candidate is edited. A second candidate cannot begin while the active
record exists.

At that boundary the harness hashes:

- `PROGRAM.md` and the experiment specification;
- every config and manifest;
- the config materializer and its merge helper;
- the autoloop implementation itself, so a candidate cannot rewrite its own judge;
- stimulus generation, validation, representation extraction, and scientific evaluation code;
- the test suite, so a candidate cannot weaken the tests that judge it.

`evaluate` compares those hashes before and after the checks. Any movement blocks acceptance. The
candidate must also change at least one of its declared engineering paths, finish inside its fixed
wall-clock budget, and pass every predeclared check.

The closed check registry is:

| Check | Evidence allowed |
|---|---|
| `unit` | Full local test suite |
| `config` | Generated configs remain byte-reproducible |
| `durability` | Resume, checkpoint, lineage, and matrix tests |
| `packaging` | Colab packaging/driver tests |
| `throughput` | Fast-loader and profiling-tool tests; no scientific outputs |

There is deliberately no `--command`. Adding one would make the anti-p-hacking boundary a comment
rather than an enforced property.

## Usage

Start an iteration before editing:

```bash
.venv/bin/python scripts/engineering_autoloop.py begin \
  --candidate-id cache-atomic-write \
  --candidate-path src/lurestar/fast_stargraph.py \
  --time-budget-seconds 1800 \
  --max-checks 2 \
  --check throughput \
  --check unit
```

Make only that candidate change, then evaluate it:

```bash
.venv/bin/python scripts/engineering_autoloop.py evaluate
```

The verdict is appended to `results/engineering_loop.jsonl`. Each record carries the candidate's
before/after digest, the frozen-surface digests, commands fixed at `begin`, bounded output tails, elapsed time,
and acceptance reasons. The file is opened with append semantics and fsynced; existing bytes are
never rewritten.

To close an abandoned attempt:

```bash
.venv/bin/python scripts/engineering_autoloop.py abort --reason "superseded by a smaller patch"
```

`abort` records the reason and leaves the files untouched.

## Shared-worktree rule

The harness never runs `git reset`, `git checkout --`, `git restore`, or an equivalent cleanup.
This repository may contain user work and concurrent agent edits, so it cannot infer which dirty
lines are safe to destroy. A rejected candidate remains in place for an explicit forward fix, a
manual decision by its owner, or evaluation in a separate worktree. The append-only record says
`worktree_action: none` so rejection cannot be mistaken for rollback.

## What this loop cannot claim

An accepted candidate is safer, more reproducible, or faster on an engineering test. Acceptance
says nothing about whether NextLat has stronger predictive geometry, whether H3 is positive, or
whether any scientific hypothesis is supported. Those questions remain sealed until the
confirmatory sweep and must be reported regardless of sign.
