# Decision: CFS-1 is a new causal-forgetting experiment

**Status:** approved design; no CFS-1 model training or outcome evaluation has started.

The retired legacy H3 branch is not reopened. Its one-shot matching feasibility rule
failed before confirmatory training: after the permitted expansion, four of 5,000
near examples had no eligible middle control. That is a design feasibility result,
not a null effect and not an invitation to tune a matching procedure.

CFS-1 is a distinct experiment with new identifiers, generator, manifests,
preregistration, runner/evaluator, ledger, and clearance. It cannot inherit H3's
authority or use legacy H3 pools, candidate banks, losses, learned distances,
calipers, middle matches, pilot results, checkpoints, or scientific outcomes.

Its causal intervention is randomized by construction: later-learning updates have
high or low structural overlap with an untouched retention probe and the same or a
different correct future. This permits a prespecified difference-in-differences in
retention-margin erosion. It does not turn predictive geometry into a causal
mediator; pre-update geometry is a moderator only. A separate state-interchange
mechanism test, with anchor and norm/random-subspace controls, is descriptive of
readout causality rather than proof of global mediation.

The primary CFS-1 population is eight independent NextLat parent checkpoints: the
five planned base parents plus three newly trained CFS-only parents. Every parent
is forked into all two episodes and all four intervention cells at the exact same
parent checkpoint, optimizer state, update count (500 CE-only updates), and
hash-bound stream schedule. The 64 branch identifiers are committed in
`manifests/cfs1/global_manifest.json` when materialized.
