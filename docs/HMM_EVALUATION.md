# HMM checkpoint evaluation

`scripts/evaluate_hmm_checkpoints.py` is the production bridge from a hash-verified `TRAINED`
HMM checkpoint to `evaluation/hmm_geometry.json`. `scripts/run_hmm_matrix.py --phase evaluate`
invokes it and permits `TRAINED -> DONE` only after independently re-hashing the checkpoint,
evaluator sources, frozen manifests, representation plan, cache progress, every cache chunk and
sidecar, and the complete metric receipt.

## Frozen analysis populations

These choices were fixed before any confirmatory HMM outcome was observed:

- HMM Bayesian-posterior probe fit: validation rows `0:5000`, every prefix length `16..32`.
- HMM posterior length-32 score: validation rows `5000:10000`, every prefix length `16..32`.
- HMM posterior length-64 score: length-generalization rows `5000:10000`, prefix lengths `33..64`, using the
  length-32 probe without refitting.
- H2 neighborhood retrieval: all unique endpoints in the frozen pair bank, separately within
  `test32` and `test64`, with `k=10`. There is no endpoint sampling.
- Primary distance is centered cosine. Whitened Euclidean is the declared robustness analysis.
- The linear probe ridge is `1e-2`. No metric, pool, seed, ridge, or endpoint is selected after
seeing an outcome.

The frozen receipt schema retains historical metric keys beginning with `h3_`. In this document
that namespace means only HMM posterior/future-state decoding. It is not the Lure-Star H3
adaptation/interference estimand, which D40 permanently excluded together with all of its
adaptation branches and mechanism probes.

The evaluator writes this policy and the realized frozen pair/endpoint counts to
`evaluation/representation_manifest.json` before GPU extraction. Its SHA-256 is part of the cache
identity and final receipt.

## Interruption and resume contract

Representation extraction is divided into independently useful chunks. For each chunk:

1. arrays are validated as non-object and finite;
2. an NPZ is written to `.partial`, flushed, fsynced, and atomically renamed;
3. the final NPZ is SHA-256 hashed and a hash sidecar is atomically written;
4. only then is `representation_cache/progress.json` advanced atomically.

A disconnect can therefore lose at most the in-flight inference chunk. On reconnect, committed
chunks are re-hashed and loaded; only missing chunks run again. A mismatched checkpoint, config,
pair bank, posterior array, representation plan, upstream commit, or evaluator source refuses the
old cache instead of mixing identities. Corrupt chunks are never considered completed.

At the default batch size 256, the current frozen bank and posterior-probe pools produce 153 committed chunks.
Each completed chunk is durable evidence even if the runtime disappears before final scoring.

## Memory behavior

The unique endpoint populations currently contain 7,042 (`test32`) and 9,237 (`test64`) items.
Neighborhood retrieval is exact but query-blocked: it compares each query block against the full
frozen population, applies a stable tie rule, immediately reduces to the ten nearest neighbors,
and discards the distance block. It does not allocate full state- and belief-distance matrices.

## Runtime interface

The matrix driver constructs the command, including all paths and hashes owned by the job:

```bash
python scripts/run_hmm_matrix.py \
  --root /content/lurestar \
  --snapshot-root /content/lurestar \
  --data-root /content/lurestar \
  --project-root /content/project \
  --upstream /content/project/upstream/NextLat \
  --family \
  --phase evaluate \
  --driver-managed-durability
```

Evaluation is idempotent. A valid receipt is verified and skipped; an interrupted cache resumes;
an invalid receipt or cache fails closed and leaves the ledger at `TRAINED`.
