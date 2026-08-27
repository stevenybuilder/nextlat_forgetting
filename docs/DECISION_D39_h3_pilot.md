# D39 — Freeze one nonconfirmatory BST pilot for H3 nuisance matching

**Status:** Accepted and frozen before pilot scoring on 2026-08-24. This decision resolves the
pilot-identity portion of the outcome-blind Gate 4 blocker. It does not authorize confirmatory
training; the loss table, selections, all-eleven freeze receipt, full-suite receipt, independent
review, immutable source bundle, and confirmatory GO receipt must still pass.

## Decision

The sole H3 selection pilot is the already-computed, nonconfirmatory A100 engineering-profile BST
checkpoint with seed 1234 at exact training step 500:

- checkpoint SHA-256: `1f5f00611e33ada0ac0a778f9d45bef9e174f1bbeedfaaa3491018a9bf400176`;
- materialized-config SHA-256: `03e1b9e4a1a2a7e44b68cf69a1534a90a695685004ea5ebe79822e0bd9472e98`;
- checkpoint metadata SHA-256: `585480bc93fffc80e02c989208085c2855a8bc8d65b61db9b2168b419dac96b9`;
- recovered profile: `a100-be81d2f1e79c-8d316efc9c53`, durable state generation 12;
- state SHA-256: `5ea74eb05e72759fbaa2cded94ee758300cfe9057d5a7d6f8ef81d566afa577a`.

The exact versioned remote object names and object generations are recorded in
`manifests/h3_precompute/pilot_freeze.json`. No other architecture, seed, step, checkpoint, or
retrained replacement may be substituted. If this checkpoint cannot be verified or scored, H3 is
blocked until an explicit prospective amendment is made; a confirmatory checkpoint may never be
used to repair the pilot.

## Why this reuse is scientifically acceptable

The checkpoint was produced for engineering throughput/recovery profiling before any
confirmatory H1/H2/H3/HMM outcome existed. Its fixed identity therefore cannot encode knowledge of
the eventual treatment contrast. Reuse also extracts value from compute already spent instead of
training another nuisance-selection model. The checkpoint is not a replicate, is not included in
the five-seed estimator, and cannot support a model-performance or scientific-effect claim.

Its role is limited to nuisance matching: initial teacher-forced next-token loss. Candidate
construction itself is model-blind and solver-verified. The scorer uses BST's generation-time
forward encoder, a differentiable lone-EOS backward state, and TextHead through the same
teacher-forced CE path frozen for common H3 adaptation. It may inspect neither confirmatory
checkpoints nor confirmatory result directories.

## Rules frozen before scoring

1. `B_mid` has exactly 15,000 candidates, three per frozen `B_near` item. Candidates are
   deterministic sequential suffix rewires, independently solver-verified, structurally varied,
   and identity-disjoint from training, `E_lure`, `B_near`, and `B_far`.
2. The `B_far` selector orders all 15,000 candidates by `(pilot loss, prompt SHA-256)`. For the
   near item at empirical loss rank `r`, it chooses far rank `3r+1`. Because the pool sizes are
   5,000 and 15,000, the empirical midpoint quantiles are exactly equal.
3. A middle candidate is eligible only when it has the same pilot-loss decile as its paired near
   item, absolute pilot-loss difference at most `0.1`, the same target-path length, and a solver
   pass. The target is the global 50th percentile of structural distance among all eligible
   candidates—not absolute distance 0.5. Each near item takes its eligible candidate closest to
   that target, with ascending candidate SHA-256 as the tie-break. Missing eligibility for even
   one near item is a hard infeasibility result; thresholds are not tuned.
4. Each independent acquisition candidate pool contains 6,000 solver-verified items constructed
   without model access. Selection returns exactly 2,000 per branch: 200 from each pilot-loss
   decile, chosen by ascending SHA-256. The resulting near/mid/far banks remain mutually disjoint,
   disjoint from training/adaptation/evaluation domains, and path-distribution matched.
5. Full 53,000-row pilot loss coverage is mandatory. Truncation, imputation, candidate
   regeneration after scoring, and threshold/seed/checkpoint substitution are prohibited.

These are nuisance-balance choices, not hypothesis tests. The full eligible 15,000-row middle
table, mappings, source hashes, and any infeasibility are retained so the choice is auditable and
cannot be reverse-engineered from favorable confirmatory behavior.

## Frozen CPU artifacts

Focused acceptance tests passed (`6 passed`). CPU generation then completed without paid compute
in 18.7 seconds and produced:

| Artifact | Count | SHA-256 |
|---|---:|---|
| `b_mid_candidates.jsonl` | 15,000 | `df4a1a18ba4f5b2eb18e13a9dfb69e7c08b9d952044ea8392eead996123ecf8f` |
| `acquisition_near_candidates.jsonl` | 6,000 | `f152d2f263900e760aefff67085f202b659692ea186c024f53b0e812adb46053` |
| `acquisition_mid_candidates.jsonl` | 6,000 | `85096001d85ad7ef0dbe001b712d78c3b697d0b08f84fafbeef51b11e2c9512d` |
| `acquisition_far_candidates.jsonl` | 6,000 | `d279d29388f3315049c4693eba52b56f6ffbb5e1ba5d4dcb0c25e915e404c717` |
| `pilot_freeze.json` | — | `702db26b36ca3ecf57cdfaa8dd176c276b07a77dbbdf6f1f70647df3df85f364` |

The generation receipt is
`manifests/h3_precompute/candidate_generation_receipt.json` (SHA-256
`08a8bdc86cdda87f826d333693568a1137a1a7f7bd942d44a3974b0fbf4f430f`). Every
artifact uses create-or-verify publication: a byte-identical rerun is accepted, while an existing
different artifact or stale sidecar blocks.

## Remaining bounded action

The exact checkpoint/config are prefetched and hash-verified under archive-excluded
`.agent_state/pilot/`. The hash-bound score job is `.agent_state/pilot/h3-score-job.json`. Its
read-only plan is:

```bash
.venv/bin/python scripts/score_h3_pilot.py --mode plan \
  --job .agent_state/pilot/h3-score-job.json
```

The scientific scorer is executed through the separately hash-bound durable transport driver;
running the scorer directly on ephemeral `/content` is prohibited. The bounded Colab action is:

```bash
python scripts/run_h3_pilot_durable.py --mode run \
  --job .agent_state/pilot/h3-score-job.json \
  --adc /content/adc.json --bootstrap --chunk-size 1000 --batch-size 64
```

The driver SHA-256 is
`4c7821089b6682125ecf1bb2325bccc88cdba1a907c17234c751f61433a7e8c5`; the exact job SHA-256 is
`860aa43623ec07c1e1bf97d0bcd77629b08a1ed3853e3af45640c733d13e0feb`, also recorded by its
adjacent sidecar. Its durable prefix is
`gs://YOUR_PRIVATE_BUCKET/lurestar/h3-pilot-scores/860aa43623ec07c1e1bf97d0bcd77629b08a1ed3853e3af45640c733d13e0feb`.

It scores 53,000 items. The preregistered engineering estimate is 3–12 A100 minutes plus transfer
and setup. Each 1,000-row chunk is published create-only as data, scorer receipt, then a commit
record binding SHA-256 metadata and exact GCS object generations; all three are read back before
the next chunk. A replacement runtime restores and verifies committed chunks before doing more
forward passes. Final table and receipt follow all chunks, and complete `state.json` is published
last. Thus disconnection can lose at most the currently in-flight chunk, never a committed chunk.
The driver uses only the uploaded mode-0600 ADC and never prints or persists credentials.

Bootstrap preserves Colab's CUDA PyTorch, installs only missing Lightning/OmegaConf/GCS client
dependencies with `only-if-needed`, and refuses to score unless torch's version is unchanged,
CUDA is available, and a clean-process torch/NumPy ABI probe passes. After a complete durable loss
receipt, run:

```bash
.venv/bin/python scripts/materialize_h3_precompute.py --mode select \
  --loss-table .agent_state/pilot/h3_pilot_score/pilot_losses.jsonl
```

Selection is CPU-only. A successful selector produces the exact far/mid mappings and three
independent acquisition banks consumed by `scripts/materialize_adaptation_banks.py`. Until those
real artifacts exist and validate, H3 and confirmatory training remain blocked.

A one-row local model smoke was attempted after all path/hash/population checks passed, but the
host verification environment lacks the upstream `lightning` dependency and has an old torch
wheel built against NumPy 1.x while the environment contains NumPy 2.x. It failed before model
construction and wrote no chunk or receipt. This is an environment blocker, not a passing model
smoke; the first bounded Colab chunk remains the required executable integration check.

## Prospective D40 support amendment

D39 remains immutable evidence: its pilot identity, 53,000 losses, scorer, far rule, acquisition
rule, caliper, deciles, path/solver constraints, structural target, and tie-break are unchanged.
Its original middle support left 1,115/5,000 items unmatched. Before any new score was computed,
Decision D40 froze one uniform model-blind support expansion and a permanent stopping rule. D40
supersedes D39 only for the size of the middle candidate support; the failed D39 artifacts remain
preserved for audit. See `docs/DECISION_D40_h3_overlap_expansion.md`.
