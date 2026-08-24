"""Experiment B: HMM ground-truth belief geometry (spec section 12).

The point of this package is that for an HMM the *exact* sufficient predictive state of a
history is computable, so "does the model's hidden state respect predictive equivalence"
becomes a measurement rather than an interpretation. Everything downstream -- the pair bank,
the probes, HMM-H1/H2/H3 -- rests on `forward.py` being exactly right, which is why it is
checked against brute-force path enumeration rather than against itself.
"""

from .forward import HMM, ForwardResult, forward_batch, brute_force_posteriors

__all__ = ["HMM", "ForwardResult", "forward_batch", "brute_force_posteriors"]
