"""Tests for the Lure-Star representation/geometry layer.

Every test here is written so that it can FAIL on a subtly wrong implementation.  Where a
test could conceivably pass on a broken estimator, it carries an explicit *discrimination*
assertion: the same test also checks that a deliberately corrupted variant of the estimator
is rejected.  A test that cannot reject the mutant is not a test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lurestar import evaluate as E
from lurestar import representations as R


# =====================================================================================
# Fixtures / helpers
# =====================================================================================

# The verified G(5,5) tokenization from docs/UPSTREAM_REPORT.md §1.6, reproduced with the
# repo's own Tokenizer (upstream data/stargraph.py:9-57) for the example
#   ...|53,5/49,33=49,97,53,5,33
VERIFIED_TOKENS = [
    49, 97, 100, 65, 62, 100, 36, 85, 100, 51, 38, 100, 61, 45, 100, 49, 12, 100,
    64, 17, 100, 5, 33, 100, 12, 79, 100, 49, 64, 100, 62, 51, 100, 45, 74, 100,
    49, 61, 100, 74, 27, 100, 17, 36, 100, 32, 68, 100, 97, 53, 100, 79, 32, 100,
    49, 65, 100, 53, 5, 102, 49, 33, 101, 49, 97, 53, 5, 33, 104,
]


def orthonormal_basis(dim: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """k random orthonormal directions in R^dim, as rows.  Never axis-aligned."""
    Q, _ = np.linalg.qr(rng.standard_normal((dim, k)))
    return Q.T[:k]


def naive_centered_cosine(a_rows, b_rows, pool_rows):
    """Deliberately naive, loop-based reference implementation of the primary distance.

    Written independently of the vectorized einsum version under test.
    """
    p = len(pool_rows)
    dim = len(pool_rows[0])
    mean = [sum(row[j] for row in pool_rows) / p for j in range(dim)]
    out = []
    for a, b in zip(a_rows, b_rows):
        ac = [a[j] - mean[j] for j in range(dim)]
        bc = [b[j] - mean[j] for j in range(dim)]
        dot = sum(ac[j] * bc[j] for j in range(dim))
        na = math.sqrt(sum(v * v for v in ac))
        nb = math.sqrt(sum(v * v for v in bc))
        out.append(1.0 - dot / (na * nb))
    return out


# =====================================================================================
# 1. The frozen extraction indices, and the correction they encode
# =====================================================================================


def test_verified_tokenization_matches_upstream_report():
    assert len(VERIFIED_TOKENS) == 69, "G(5,5) total_len is 69 (UPSTREAM_REPORT §1.6)"
    assert VERIFIED_TOKENS[62] == R.EQ_TOKEN_ID
    # prompt tail: '/', source, goal, '='
    assert VERIFIED_TOKENS[59] == 102  # '/'
    assert VERIFIED_TOKENS[60] == 49   # source, printed in the prompt
    assert VERIFIED_TOKENS[61] == 33   # goal
    assert VERIFIED_TOKENS[68] == 104  # EOS


def test_resolve_extraction_indices_returns_the_frozen_pair():
    tokens = np.array([VERIFIED_TOKENS] * 7)
    idx = R.resolve_extraction_indices(tokens)
    assert idx == {"psi_primary": 62, "branch_margin": 63, "delimiter_index": 62}
    assert R.PSI_EXTRACTION_INDEX == 62
    assert R.BRANCH_MARGIN_INDEX == 63


def test_index_62_predicts_the_source_and_index_63_predicts_the_branch():
    """THE CORRECTION, as an executable claim.

    The token generated from index 62 is path[0] = the source, which is already in the
    prompt at index 60.  The token generated from index 63 is path[1] = the first branch
    node.  Swapping the two constants would break this test.
    """
    tokens = np.array([VERIFIED_TOKENS])
    targets = R.next_token_targets(tokens, [62, 63])
    assert targets[0, 0] == tokens[0, 60], "index 62 predicts the source, a prompt copy"
    assert targets[0, 0] == 49
    assert targets[0, 1] == 97, "index 63 predicts the first branch node"
    assert targets[0, 0] != targets[0, 1]


def test_source_is_constant_across_a_quartet_but_the_branch_is_not():
    """Why a margin at index 62 cannot possibly separate near-safe from near-critical.

    Build a synthetic quartet that obeys the spec §5 matching rules: identical source and
    goal, identical prompt length, and a near-critical member whose first branch differs.
    """
    base = list(VERIFIED_TOKENS)
    repeat = list(VERIFIED_TOKENS)          # reshuffled edges, same answer
    near_safe = list(VERIFIED_TOKENS)       # two distractor endpoints swapped, same answer
    near_critical = list(VERIFIED_TOKENS)
    near_critical[64] = 65                  # goal-arm suffix swap changes path[1]
    near_critical[65] = 62
    quartet = np.array([base, repeat, near_safe, near_critical])

    at62 = R.next_token_targets(quartet, [62]).ravel()
    at63 = R.next_token_targets(quartet, [63]).ravel()
    assert len(set(at62.tolist())) == 1, "source is identical across the quartet"
    assert len(set(at63.tolist())) == 2, "the branch decision is what differs"


def test_resolve_extraction_indices_rejects_ragged_and_shifted_batches():
    good = list(VERIFIED_TOKENS)
    ragged = list(VERIFIED_TOKENS)
    ragged[62], ragged[61] = ragged[61], ragged[62]   # '=' now at 61
    with pytest.raises(ValueError, match="not constant"):
        R.resolve_extraction_indices(np.array([good, ragged]))
    with pytest.raises(ValueError, match="frozen preregistered index"):
        R.resolve_extraction_indices(np.array([ragged, ragged]))
    with pytest.raises(ValueError, match="no '='"):
        R.resolve_extraction_indices(np.array([[1, 2, 3]]))


# =====================================================================================
# 2. Centered cosine: invariance to a global translation, sensitivity to a per-condition one
# =====================================================================================


def test_centered_cosine_matches_a_naive_loop_reference():
    rng = np.random.default_rng(20260823)
    dim, n = 12, 40
    a = rng.standard_normal((n, dim))
    b = rng.standard_normal((n, dim))
    pool = np.vstack([a, b, rng.standard_normal((n, dim))])
    got = R.centered_cosine_distance(a, b, mean=R.centering_mean(pool))
    want = naive_centered_cosine(a.tolist(), b.tolist(), pool.tolist())
    assert np.allclose(got, want, rtol=0, atol=1e-12)


def test_centered_cosine_is_invariant_to_a_GLOBAL_translation_of_the_pool():
    rng = np.random.default_rng(7)
    dim, n = 16, 50
    base = rng.standard_normal((n, dim))
    crit = base + 0.4 * rng.standard_normal((n, dim))
    safe = base + 0.4 * rng.standard_normal((n, dim))
    pool = np.vstack([base, crit, safe])

    d0 = R.centered_cosine_distance(base, crit, mean=R.centering_mean(pool))

    t = 13.7 * rng.standard_normal(dim)          # a large, arbitrary global shift
    d1 = R.centered_cosine_distance(
        base + t, crit + t, mean=R.centering_mean(pool + t)
    )
    assert np.allclose(d0, d1, rtol=0, atol=1e-10)

    # Discrimination: the RAW (uncentered) cosine distance is NOT translation invariant,
    # so this test would fail against an implementation that forgot to center at all.
    raw0 = R.cosine_distance_raw(base, crit)
    raw1 = R.cosine_distance_raw(base + t, crit + t)
    assert not np.allclose(raw0, raw1, atol=1e-3)


def test_centered_cosine_is_NOT_invariant_to_a_PER_CONDITION_translation():
    """Translating one condition is a real geometric change and must show up.

    This is the failure mode the explicit-pool argument exists to prevent: if distances
    were computed after subtracting a per-condition mean, a shift applied to only the
    near-critical states would silently vanish and PSI could be manufactured or erased.
    """
    rng = np.random.default_rng(11)
    dim, n = 16, 50
    base = rng.standard_normal((n, dim))
    crit = base + 0.4 * rng.standard_normal((n, dim))
    safe = base + 0.4 * rng.standard_normal((n, dim))

    pool0 = np.vstack([base, crit, safe])
    d0 = R.centered_cosine_distance(base, crit, mean=R.centering_mean(pool0))

    t = 2.5 * rng.standard_normal(dim)
    crit_shift = crit + t                        # ONLY the critical condition moves
    pool1 = np.vstack([base, crit_shift, safe])
    d1 = R.centered_cosine_distance(base, crit_shift, mean=R.centering_mean(pool1))

    assert not np.allclose(d0, d1, atol=1e-6)
    assert abs(float(d1.mean() - d0.mean())) > 1e-3

    # And the per-condition-centered variant is precisely the thing that would hide it:
    d_percond_0 = R.centered_cosine_distance(
        base - base.mean(0), crit - crit.mean(0), mean=np.zeros(dim)
    )
    d_percond_1 = R.centered_cosine_distance(
        base - base.mean(0), crit_shift - crit_shift.mean(0), mean=np.zeros(dim)
    )
    assert np.allclose(d_percond_0, d_percond_1, atol=1e-10), (
        "per-condition centering is blind to the shift — which is exactly why the "
        "centering pool must be an explicit, pooled argument"
    )


def test_centered_cosine_requires_the_mean_and_rejects_bad_shapes():
    a = np.ones((3, 4))
    with pytest.raises(TypeError):
        R.centered_cosine_distance(a, a)          # mean is keyword-only and required
    with pytest.raises(ValueError, match="mean must have shape"):
        R.centered_cosine_distance(a, a, mean=np.zeros(5))
    with pytest.raises(ValueError, match="shape mismatch"):
        R.centered_cosine_distance(a, np.ones((3, 5)), mean=np.zeros(4))


# =====================================================================================
# 3. A synthetic geometry with an EXACTLY KNOWN PSI
# =====================================================================================


def test_psi_recovers_a_known_value_on_an_analytic_geometry():
    """Construct states whose PSI is known in closed form, then recover it.

    Trick: build the pool from mirrored item pairs (item B is the exact negation of item
    A).  The pool mean is then exactly zero, so centered cosine reduces to raw cosine and

        PSI = [1 - cos(alpha_c)] - [1 - cos(alpha_s)] = cos(alpha_s) - cos(alpha_c)

    with alpha_s the base->near-safe angle and alpha_c the base->near-critical angle.
    The states are expressed in a RANDOM orthonormal basis of R^32, so an implementation
    that quietly assumed axis alignment, dropped the centering, or swapped the critical
    and safe arguments cannot pass.
    """
    rng = np.random.default_rng(1234)
    dim = 32
    e0, e1, e2 = orthonormal_basis(dim, 3, rng)

    alpha_s, alpha_c = 0.30, 0.90                       # radians; critical is farther
    known_psi = math.cos(alpha_s) - math.cos(alpha_c)   # = 0.3337262...

    a = e0
    s = math.cos(alpha_s) * e0 + math.sin(alpha_s) * e1
    c = math.cos(alpha_c) * e0 + math.sin(alpha_c) * e2

    n_pairs = 25
    base = np.vstack([a] * n_pairs + [-a] * n_pairs)
    safe = np.vstack([s] * n_pairs + [-s] * n_pairs)
    crit = np.vstack([c] * n_pairs + [-c] * n_pairs)

    # A global offset, to prove the recovery does not depend on the pool sitting at 0.
    offset = 5.0 * rng.standard_normal(dim)
    base, safe, crit = base + offset, safe + offset, crit + offset
    pool = np.vstack([base, safe, crit])
    assert np.allclose(R.centering_mean(pool), offset, atol=1e-12)

    dist = E.psi_distances_centered_cosine(base, crit, safe, centering_pool=pool)
    per_item = E.psi_items(dist["d_critical"], dist["d_safe"])

    assert np.allclose(per_item, known_psi, rtol=0, atol=1e-12)
    assert abs(float(per_item.mean()) - known_psi) < 1e-12
    assert known_psi == pytest.approx(0.33372652, abs=1e-7)   # cos(0.30) - cos(0.90)

    # Discrimination 1: reversing the PSI sign convention is caught.
    reversed_psi = E.psi_items(dist["d_safe"], dist["d_critical"])
    assert abs(float(reversed_psi.mean()) - known_psi) > 0.6

    # Discrimination 2: making the critical lure geometrically identical to the safe one
    # must drive the known PSI to exactly zero.
    s2 = math.cos(alpha_s) * e0 + math.sin(alpha_s) * e2
    crit_null = np.vstack([s2] * n_pairs + [-s2] * n_pairs) + offset
    pool_null = np.vstack([base, safe, crit_null])
    dnull = E.psi_distances_centered_cosine(base, crit_null, safe, centering_pool=pool_null)
    assert abs(float(E.psi_items(dnull["d_critical"], dnull["d_safe"]).mean())) < 1e-12


def test_psi_distances_center_on_the_DECLARED_pool_and_not_on_the_scored_pair():
    """The pooled centering mean must actually be used.

    `psi_distances_centered_cosine` is handed the full E_lure pool — base, repeat,
    near-safe, near-critical AND far-critical. An implementation that quietly rebuilt the
    mean from just the three arrays it was passed would still return plausible numbers, so
    this test pins the identity against an explicit-mean computation with the declared
    pool, and separately asserts that the pair-derived mean gives a DIFFERENT answer. Both
    halves are needed: the first alone would pass if the two pools happened to coincide.
    """
    rng = np.random.default_rng(4242)
    dim, n = 20, 120
    base = rng.standard_normal((n, dim))
    repeat = base + 0.05 * rng.standard_normal((n, dim))
    safe = base + 0.25 * rng.standard_normal((n, dim))
    crit = base + 0.55 * rng.standard_normal((n, dim))
    # Far-critical is a different region of the space; it belongs in the pool (spec §5)
    # and it is what pulls the declared mean away from the scored triple's own mean.
    far = base + 3.0 + 1.5 * rng.standard_normal((n, dim))

    declared_pool = np.vstack([base, repeat, safe, crit, far])
    out = E.psi_distances_centered_cosine(base, crit, safe, centering_pool=declared_pool)

    assert out["centering_pool_n"] == 5 * n
    want_c = R.centered_cosine_distance(base, crit, mean=R.centering_mean(declared_pool))
    want_s = R.centered_cosine_distance(base, safe, mean=R.centering_mean(declared_pool))
    assert np.allclose(out["d_critical"], want_c, rtol=0, atol=1e-12)
    assert np.allclose(out["d_safe"], want_s, rtol=0, atol=1e-12)

    # The pair-only pool is a materially different centering and must not be what we got.
    pair_pool = np.vstack([base, crit, safe])
    pair_c = R.centered_cosine_distance(base, crit, mean=R.centering_mean(pair_pool))
    assert not np.allclose(out["d_critical"], pair_c, rtol=0, atol=1e-3)
    assert abs(float(out["d_critical"].mean() - pair_c.mean())) > 1e-2

    # ...and it moves PSI itself, which is why the pool is a required argument.
    psi_declared = float(E.psi_items(out["d_critical"], out["d_safe"]).mean())
    psi_pair = float(
        E.psi_items(
            pair_c, R.centered_cosine_distance(base, safe, mean=R.centering_mean(pair_pool))
        ).mean()
    )
    assert abs(psi_declared - psi_pair) > 1e-3, (
        "if the centering pool did not change PSI here, this test could not detect a "
        "pool substitution and would not be a test"
    )


def test_psi_bootstrap_interval_brackets_the_known_value():
    rng = np.random.default_rng(99)
    n = 400
    true_psi = 0.12
    d_safe = 0.5 + 0.05 * rng.standard_normal(n)
    d_crit = d_safe + true_psi + 0.05 * rng.standard_normal(n)
    res = E.bootstrap_psi_items(d_crit, d_safe, rng=np.random.default_rng(5), n_boot=4000)
    assert res.ci.unit == "item (quartet)"
    assert res.ci.ci_low < true_psi < res.ci.ci_high
    assert res.ci.n_units == n
    # A null geometry must NOT be called significant.
    d_crit_null = d_safe + 0.05 * rng.standard_normal(n)
    null = E.bootstrap_psi_items(d_crit_null, d_safe, rng=np.random.default_rng(5), n_boot=4000)
    assert null.ci.ci_low < 0.0 < null.ci.ci_high


# =====================================================================================
# 4. Bootstrap coverage against a known sampling distribution
# =====================================================================================


def test_paired_bootstrap_attains_nominal_coverage_and_a_narrowed_ci_does_not():
    """Frequentist coverage check with the truth known by construction.

    Data are iid Normal(mu, sigma), so the estimand (the mean) is exactly `mu`.  Over 400
    independent replications the nominal 95% percentile-bootstrap interval should cover
    `mu` about 94-95% of the time (percentile intervals for a mean run slightly
    anticonservative at finite n).  The Monte-Carlo SE at p=0.94 with 400 reps is 0.012,
    so the acceptance band below is roughly +/- 4 SE.

    The second half of the test is the discrimination: an interval artificially shrunk to
    half-width must fail the same coverage check.  Without it, a bootstrap that returned
    (-inf, +inf) would also "pass".
    """
    rng = np.random.default_rng(2026)
    mu, sigma, n, n_rep, n_boot = 0.4, 1.0, 120, 400, 800

    covered = 0
    covered_narrow = 0
    widths = []
    for _ in range(n_rep):
        sample = mu + sigma * rng.standard_normal(n)
        ci = E.paired_bootstrap_mean(
            sample, unit="item (synthetic)", rng=rng, n_boot=n_boot, alpha=0.05
        )
        if ci.ci_low <= mu <= ci.ci_high:
            covered += 1
        widths.append(ci.ci_high - ci.ci_low)
        mid = ci.estimate
        lo_n = mid - 0.5 * (mid - ci.ci_low)
        hi_n = mid + 0.5 * (ci.ci_high - mid)
        if lo_n <= mu <= hi_n:
            covered_narrow += 1

    coverage = covered / n_rep
    coverage_narrow = covered_narrow / n_rep
    assert 0.90 <= coverage <= 0.98, f"nominal-95% coverage was {coverage:.3f}"
    assert coverage_narrow < 0.85, (
        f"a half-width interval covered {coverage_narrow:.3f} of the time; the coverage "
        "check cannot discriminate and is therefore not a test"
    )
    # Width should track the analytic 1.96*sigma/sqrt(n) = 0.3578 to within ~10%.
    assert abs(float(np.mean(widths)) - 2 * 1.96 * sigma / math.sqrt(n)) < 0.04


def test_bootstrap_demands_an_explicit_generator_and_a_named_unit():
    v = np.linspace(0, 1, 20)
    with pytest.raises(TypeError):
        E.paired_bootstrap_mean(v, unit="item", rng=12345)       # not a Generator
    with pytest.raises(TypeError):
        E.paired_bootstrap_mean(v, rng=np.random.default_rng(0))  # unit is required


def test_bootstrap_is_deterministic_given_the_generator_seed():
    v = np.random.default_rng(3).standard_normal(200)
    a = E.paired_bootstrap_mean(v, unit="item", rng=np.random.default_rng(77), n_boot=500)
    b = E.paired_bootstrap_mean(v, unit="item", rng=np.random.default_rng(77), n_boot=500)
    assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)
    c = E.paired_bootstrap_mean(v, unit="item", rng=np.random.default_rng(78), n_boot=500)
    assert (a.ci_low, a.ci_high) != (c.ci_low, c.ci_high)


def test_bootstrap_chunking_does_not_change_the_replicate_count():
    v = np.random.default_rng(4).standard_normal(6000)
    ci = E.paired_bootstrap_mean(
        v, unit="item", rng=np.random.default_rng(1), n_boot=5000, keep_replicates=True
    )
    assert ci.replicates.shape == (5000,)          # chunked internally, still 5000 reps
    assert np.isfinite(ci.replicates).all()


# =====================================================================================
# 5. Items are not seeds
# =====================================================================================


def test_seed_level_contrast_uses_seeds_and_refuses_item_arrays():
    psi_nextlat = {1234: 0.031, 1235: 0.022, 1236: 0.040}
    psi_gpt = {1234: 0.010, 1235: 0.014, 1236: 0.009}
    with pytest.warns(UserWarning, match="seed-level interval computed from 3 seeds"):
        con = E.model_contrast_seed_level(
            psi_nextlat, psi_gpt, label_a="nextlat", label_b="gpt",
            rng=np.random.default_rng(0), n_boot=2000,
        )
    assert con.n_seeds == 3
    assert con.ci.unit == "training seed"
    assert con.ci.n_units == 3
    assert con.estimate == pytest.approx(np.mean([0.021, 0.008, 0.031]))
    assert con.sign_flip_p == pytest.approx(0.25)
    assert con.min_attainable_p == pytest.approx(0.25)
    assert con.underpowered is True

    # An item-level array must not be accepted where seeds belong.
    with pytest.raises(TypeError, match="Mapping"):
        E.model_contrast_seed_level(
            np.zeros(20000), np.zeros(20000), label_a="a", label_b="b",
            rng=np.random.default_rng(0),
        )
    with pytest.raises(ValueError, match="seed sets differ"):
        E.model_contrast_seed_level(
            {1234: 1.0}, {9999: 1.0}, label_a="a", label_b="b",
            rng=np.random.default_rng(0),
        )


def test_item_level_interval_is_much_narrower_than_the_seed_level_one():
    """The reason the two units are separate functions, made quantitative."""
    rng = np.random.default_rng(31)
    per_seed = {1234: 0.030, 1235: 0.010, 1236: 0.050}
    items = np.concatenate(
        [v + 0.05 * rng.standard_normal(20000) for v in per_seed.values()]
    )
    item_ci = E.paired_bootstrap_mean(
        items, unit="item", rng=np.random.default_rng(1), n_boot=2000
    )
    with pytest.warns(UserWarning):
        seed_ci = E.model_contrast_seed_level(
            per_seed, {k: 0.0 for k in per_seed}, label_a="nextlat", label_b="gpt",
            rng=np.random.default_rng(1), n_boot=2000,
        ).ci
    item_w = item_ci.ci_high - item_ci.ci_low
    seed_w = seed_ci.ci_high - seed_ci.ci_low
    assert seed_w > 10 * item_w, (
        f"item width {item_w:.5f} vs seed width {seed_w:.5f}: quoting the item interval "
        "for a model contrast would overstate precision by an order of magnitude"
    )


def test_exact_sign_flip_p_is_the_enumeration_it_claims_to_be():
    d = np.array([0.1, 0.2, 0.3])
    # By hand: |mean| over the 8 sign vectors is
    # {0.2, 0.2, 0.0, 0.0, 0.0667, 0.0667, 0.1333, 0.1333}; two are >= 0.2.
    assert E._exact_sign_flip_p(d) == pytest.approx(2 / 8)
    assert E._exact_sign_flip_p(np.array([1.0, 1.0, 1.0, 1.0])) == pytest.approx(2 / 16)


# =====================================================================================
# 6. Cross-fitting never scores a fold with a model fit on it
# =====================================================================================


def test_crossfitting_folds_are_disjoint_and_exhaustive():
    rng = np.random.default_rng(8)
    n = 200
    X = rng.standard_normal((n, 2))
    y = X[:, 0] - 0.5 * X[:, 1] + 0.1 * rng.standard_normal(n)
    res = E.crossfit_linear(X, y, n_folds=2, rng=np.random.default_rng(3))
    assert res.n_folds == 2
    seen = set()
    for k in range(res.n_folds):
        tr, te = set(res.train_indices[k].tolist()), set(res.test_indices[k].tolist())
        assert tr.isdisjoint(te), "a fold was scored by a model fit on it"
        assert tr | te == set(range(n))
        seen |= te
    assert seen == set(range(n)), "every item got exactly one out-of-fold prediction"
    assert not np.any(np.isnan(res.y_pred_heldout))
    assert res.r2_heldout > 0.9      # this relationship really does generalize


def test_crossfitting_is_detected_by_opposite_slopes_in_the_two_folds():
    """The decisive leakage test.

    Fold 0 obeys y = +x, fold 1 obeys y = -x.  Genuine cross-fitting predicts each fold
    with the OTHER fold's model, so the held-out predictions are anticorrelated with the
    truth and R^2_heldout is strongly negative (~ -3).  Any leak flips the verdict:

      * fitting on the pooled data gives a near-zero slope and R^2 ~ 0;
      * scoring a fold with its own model gives R^2 ~ +1.

    So the assertion `r2_heldout < -1` rejects both leaks.
    """
    rng = np.random.default_rng(17)
    n = 400
    x = rng.standard_normal(n)
    folds = np.zeros(n, dtype=np.int64)
    folds[n // 2:] = 1
    y = np.where(folds == 0, x, -x)

    res = E.crossfit_linear(x[:, None], y, folds=folds, feature_names=("x",))
    assert res.r2_heldout < -1.0, f"leakage: held-out R^2 was {res.r2_heldout:.3f}"
    assert res.spearman_rho < -0.9, "held-out predictions should be anticorrelated here"

    # fold 0 is predicted by the model fit on fold 1 (slope -1) and vice versa
    assert res.coefficients[0, 0] < 0
    assert res.coefficients[1, 0] > 0

    # --- the two mutants this test must reject ------------------------------------
    A = np.column_stack([np.ones(n), x])
    pooled = np.linalg.lstsq(A, y, rcond=None)[0]
    r2_pooled = 1 - np.sum((y - A @ pooled) ** 2) / np.sum((y - y.mean()) ** 2)
    assert r2_pooled > -1.0, "pooled (leaked) fit must be rejected by the assertion above"

    r2_self = 0.0
    for k in (0, 1):
        m = folds == k
        Ak = np.column_stack([np.ones(m.sum()), x[m]])
        bk = np.linalg.lstsq(Ak, y[m], rcond=None)[0]
        r2_self += np.sum((y[m] - Ak @ bk) ** 2)
    r2_self = 1 - r2_self / np.sum((y - y.mean()) ** 2)
    assert r2_self > 0.99, "in-fold (leaked) fit must be rejected by the assertion above"


def test_crossfitting_standardization_uses_training_fold_statistics_only():
    """Fold-specific feature scales must not leak into the other fold's model."""
    rng = np.random.default_rng(21)
    n = 300
    folds = np.zeros(n, dtype=np.int64)
    folds[n // 2:] = 1
    x = np.where(folds == 0, rng.standard_normal(n), 100.0 * rng.standard_normal(n))
    y = 2.0 * x + rng.standard_normal(n)
    res = E.crossfit_linear(x[:, None], y, folds=folds, feature_names=("x",))
    # Standardized coefficients: fold-0's model is fit on fold 1 (sd ~ 100), so its
    # standardized slope is ~200; fold-1's model is fit on fold 0 (sd ~ 1) -> ~2.
    assert res.coefficients[0, 0] == pytest.approx(200.0, rel=0.1)
    assert res.coefficients[1, 0] == pytest.approx(2.0, rel=0.2)


def test_crossfit_linear_requires_an_explicit_rng_when_folds_are_not_given():
    X = np.random.default_rng(0).standard_normal((50, 2))
    y = np.random.default_rng(1).standard_normal(50)
    with pytest.raises(ValueError, match="explicit rng"):
        E.crossfit_linear(X, y)


def test_h2_recovers_the_planted_coefficient_direction():
    """H2's preregistered model on synthetic data with a known positive distance effect."""
    rng = np.random.default_rng(404)
    n = 800
    base_margin = rng.normal(3.0, 1.0, n)
    base_crit_dist = rng.uniform(0.05, 0.45, n)
    crit_margin = 1.5 + 0.8 * base_margin + 6.0 * base_crit_dist + rng.normal(0, 0.5, n)

    out = E.fit_h2(crit_margin, base_crit_dist, base_margin, rng=np.random.default_rng(9))
    rep, res = out["report"], out["result"]
    assert rep["r2_heldout"] > 0.7
    assert rep["spearman_rho_pred_vs_actual"] > 0.8
    dirs = rep["coefficient_directions_standardized"]
    assert dirs["base_critical_distance"]["sign_consistent"]
    assert dirs["base_critical_distance"]["signs"] == [1, 1]
    assert dirs["base_correct_branch_margin"]["signs"] == [1, 1]
    assert rep["margin_extraction_index"] == R.BRANCH_MARGIN_INDEX == 63
    assert res.n_folds == 2

    # Planting the OPPOSITE sign must flip the reported direction.
    crit_margin_neg = 1.5 + 0.8 * base_margin - 6.0 * base_crit_dist + rng.normal(0, 0.5, n)
    out2 = E.fit_h2(crit_margin_neg, base_crit_dist, base_margin, rng=np.random.default_rng(9))
    assert out2["report"]["coefficient_directions_standardized"][
        "base_critical_distance"
    ]["signs"] == [-1, -1]

    # A pure-noise outcome must NOT produce a positive held-out R^2.
    out3 = E.fit_h2(rng.standard_normal(n), base_crit_dist, base_margin,
                    rng=np.random.default_rng(9))
    assert out3["report"]["r2_heldout"] < 0.05


# =====================================================================================
# 7. Whitened Euclidean == Mahalanobis, on a held-out covariance, with reported shrinkage
# =====================================================================================


def test_whitened_euclidean_equals_mahalanobis_under_the_same_covariance():
    rng = np.random.default_rng(55)
    dim = 10
    A = rng.standard_normal((dim, dim))
    true_cov = A @ A.T + 0.5 * np.eye(dim)
    pool = rng.multivariate_normal(np.zeros(dim), true_cov, size=600)

    w = R.Whitener.fit(pool, item_ids=range(600), shrinkage="ledoit_wolf")
    a = rng.multivariate_normal(np.zeros(dim), true_cov, size=40)
    b = rng.multivariate_normal(np.zeros(dim), true_cov, size=40)

    got = w.distance(a, b)
    diff = a - b
    want = np.sqrt(np.einsum("ij,ij->i", diff, np.linalg.solve(w.covariance, diff.T).T))
    assert np.allclose(got, want, rtol=0, atol=1e-9)

    # W is a genuine inverse square root of the covariance actually used.
    assert np.allclose(w.transform @ w.transform, np.linalg.inv(w.covariance), atol=1e-9)
    assert np.allclose(w.transform, w.transform.T, atol=1e-12)

    # Discrimination: plain Euclidean is NOT the same thing on a correlated covariance.
    plain = np.sqrt(np.einsum("ij,ij->i", diff, diff))
    assert not np.allclose(got, plain, rtol=0.05)


def test_whitener_reports_shrinkage_and_never_reports_zero():
    rng = np.random.default_rng(66)
    pool = rng.standard_normal((80, 20))
    w = R.Whitener.fit(pool)
    rep = w.report()
    assert 0.0 < rep["shrinkage"] <= 1.0
    assert rep["n_pool"] == 80 and rep["n_features"] == 20
    assert rep["condition_number"] >= 1.0
    assert rep["role"].startswith("declared robustness check")
    # An explicit zero is floored, so a shrinkage of exactly 0 can never be reported.
    assert R.Whitener.fit(pool, shrinkage=0.0).shrinkage == pytest.approx(1e-3)
    # Ledoit-Wolf behaviour, both directions.
    # (a) With a genuinely non-spherical covariance the shrinkage target is WRONG, so the
    #     intensity must fall monotonically as the pool grows and S becomes trustworthy.
    A = rng.standard_normal((20, 20))
    cov = A @ A.T + 0.5 * np.eye(20)
    alphas = [
        R.Whitener.fit(rng.multivariate_normal(np.zeros(20), cov, size=n)).shrinkage
        for n in (25, 60, 200, 2000)
    ]
    assert alphas == sorted(alphas, reverse=True), f"non-monotone in n: {alphas}"
    assert alphas[0] > 0.3 and alphas[-1] < 0.1
    # (b) When the truth IS spherical the target is right and the estimator shrinks hard
    #     at every n. Asserting the (a) pattern here instead would be wrong.
    assert R.Whitener.fit(rng.standard_normal((25, 20))).shrinkage > 0.8
    assert R.Whitener.fit(rng.standard_normal((2000, 20))).shrinkage > 0.8


def test_whitener_refuses_to_score_items_from_its_own_fitting_pool():
    rng = np.random.default_rng(77)
    pool = rng.standard_normal((200, 8))
    heldout_ids = [f"quartet_{i}" for i in range(200)]
    w = R.Whitener.fit(pool, item_ids=heldout_ids)
    a = rng.standard_normal((5, 8))
    b = rng.standard_normal((5, 8))

    ok_ids = [f"quartet_{i}" for i in range(1000, 1005)]
    w.distance(a, b, item_ids=ok_ids)              # disjoint: fine

    leaky = ["quartet_1000", "quartet_7", "quartet_1002", "quartet_1003", "quartet_1004"]
    with pytest.raises(R.LeakageError, match="whitening pool"):
        w.distance(a, b, item_ids=leaky)
    with pytest.raises(R.LeakageError):
        E.psi_distances_whitened(a, b, b, whitener=w, item_ids=leaky)


def test_ledoit_wolf_intensity_is_in_range_and_shrinks_a_singular_pool_hard():
    rng = np.random.default_rng(88)
    X = rng.standard_normal((30, 60))
    X = X - X.mean(axis=0)
    alpha = R.ledoit_wolf_shrinkage(X)
    assert 0.0 <= alpha <= 1.0
    assert alpha > 0.3, "n < p should force substantial shrinkage"


def test_psi_under_both_metrics_agrees_on_the_sign_of_a_planted_effect():
    rng = np.random.default_rng(123)
    dim, n = 24, 300
    base = rng.standard_normal((n, dim))
    safe = base + 0.20 * rng.standard_normal((n, dim))
    crit = base + 0.60 * rng.standard_normal((n, dim))
    pool = np.vstack([base, safe, crit])

    prim = E.psi_distances_centered_cosine(base, crit, safe, centering_pool=pool)
    psi_prim = E.psi_items(prim["d_critical"], prim["d_safe"]).mean()

    heldout = rng.standard_normal((500, dim))
    w = R.Whitener.fit(heldout, item_ids=[f"h{i}" for i in range(500)])
    robust = E.psi_distances_whitened(
        base, crit, safe, whitener=w, item_ids=[f"e{i}" for i in range(n)]
    )
    psi_rob = E.psi_items(robust["d_critical"], robust["d_safe"]).mean()

    assert psi_prim > 0 and psi_rob > 0
    assert prim["role"].startswith("primary")
    assert robust["role"].startswith("declared robustness check")
    assert prim["centering_pool_n"] == 3 * n


# =====================================================================================
# 8. Margins from a logit array
# =====================================================================================


def test_branch_margin_uses_only_the_sibling_branch_heads():
    logits = np.array(
        [
            #  0    1    2    3    4    5
            [0.0, 5.0, 3.0, 1.0, 9.0, 0.0],
            [0.0, 2.0, 7.0, 1.0, 0.0, 0.0],
        ]
    )
    correct = np.array([1, 2])
    competitors = np.array([[2, 3], [1, 3]])         # id 4 is NOT an arm head
    got = R.branch_margin(logits, correct, competitors)
    assert got == pytest.approx([5.0 - 3.0, 7.0 - 2.0])

    # The full-vocab margin sees id 4 and therefore differs on item 0.
    full = R.full_vocab_margin(logits, correct)
    assert full == pytest.approx([5.0 - 9.0, 7.0 - 2.0])
    assert full[0] < 0 < got[0], "restricting to arm heads is not cosmetic"


def test_branch_margin_mask_and_validation():
    logits = np.array([[0.0, 5.0, 3.0, 1.0]])
    correct = np.array([1])
    comp = np.array([[2, 3]])
    masked = R.branch_margin(logits, correct, comp, competitor_mask=np.array([[False, True]]))
    assert masked == pytest.approx([5.0 - 1.0])
    with pytest.raises(ValueError, match="correct id appears in its own competitor"):
        R.branch_margin(logits, correct, np.array([[1, 3]]))
    with pytest.raises(ValueError, match="empty competitor set"):
        R.branch_margin(logits, correct, comp, competitor_mask=np.array([[False, False]]))
    with pytest.raises(ValueError, match="out of vocabulary"):
        R.branch_margin(logits, np.array([99]), comp)


def test_first_branch_accuracy_and_exact_path_accuracy():
    logits = np.array([[0.0, 5.0, 3.0], [0.0, 2.0, 7.0]])
    assert E.first_branch_accuracy(logits, np.array([1, 2])).tolist() == [1.0, 1.0]
    assert E.first_branch_accuracy(logits, np.array([2, 1])).tolist() == [0.0, 0.0]
    pred = np.array([[49, 97, 53, 5, 33], [49, 65, 53, 5, 33]])
    true = np.array([[49, 97, 53, 5, 33], [49, 97, 53, 5, 33]])
    assert E.exact_path_accuracy(pred, true).tolist() == [1.0, 0.0]


def test_safe_lure_invariance_is_centred_on_zero_when_safe_and_repeat_match():
    rng = np.random.default_rng(202)
    d_repeat = 0.10 + 0.02 * rng.standard_normal(500)
    d_safe = 0.10 + 0.02 * rng.standard_normal(500)
    ci = E.safe_lure_invariance(d_safe, d_repeat, rng=np.random.default_rng(1), n_boot=3000)
    assert ci.ci_low < 0 < ci.ci_high
    ci2 = E.safe_lure_invariance(
        d_safe + 0.05, d_repeat, rng=np.random.default_rng(1), n_boot=3000
    )
    assert ci2.ci_low > 0, "a real surface-form effect must be detected"


def test_similarity_dependent_interference_reduces_to_the_shared_parent_form():
    rng = np.random.default_rng(303)
    before = rng.normal(4.0, 0.5, 300)
    after_far = before - rng.normal(0.10, 0.05, 300)
    after_near = before - rng.normal(0.35, 0.05, 300)
    out = E.similarity_dependent_interference(
        before, after_near, after_far, rng=np.random.default_rng(2), n_boot=3000
    )
    assert out["erosion_near_mean"] > out["erosion_far_mean"]
    assert out["similarity_dependent_interference"] == pytest.approx(
        float(np.mean(after_far - after_near)), abs=1e-12
    )
    assert out["ci"].ci_low > 0
    assert out["ci"].unit.startswith("item (A_pair")


# =====================================================================================
# 9. The torch shim stays out of the way on a CPU-only host
# =====================================================================================


def test_layer_b_is_lazy_and_layer_a_needs_no_torch():
    """Importing the module and running all of Layer A must not require torch."""
    import sys

    assert "torch" not in sys.modules, "Layer A must not drag torch in at import time"
    if not R.TORCH_AVAILABLE:
        with pytest.raises(RuntimeError, match="Layer B needs torch"):
            R.forward_states_and_logits(object(), None, architecture="gpt")


def test_forward_states_and_logits_validates_the_architecture_name():
    with pytest.raises((ValueError, RuntimeError)):
        R.forward_states_and_logits(object(), None, architecture="mamba")


def test_hidden_state_hook_stores_what_it_is_given():
    store = {}
    hook = R.hidden_state_hook(store)

    class _Fake:
        def detach(self):
            return "detached-(B,T,384)"

    hook(None, None, _Fake())
    assert store["h"] == "detached-(B,T,384)"
