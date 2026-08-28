# Tests

The current suite protects legacy scientific invariants, provenance, evaluation, and recovery
behavior. New-study tests should add:

- paper/config parity for every objective;
- loss-mask, horizon, inference-head, and one-batch gradient checks;
- exact-solver and matched-counterfactual properties;
- grouped split and leakage checks;
- competent-parent and missingness semantics;
- statistical golden data with known effects and nulls;
- manifest/hash refusal tests and secret scanning.

Passing tests validates implementation behavior; it does not validate a scientific hypothesis.
