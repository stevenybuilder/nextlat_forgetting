"""Tests for the Lure-Star representation/geometry layer.

Every test here is written so that it can FAIL on a subtly wrong implementation.  Where a
test could conceivably pass on a broken estimator, it carries an explicit *discrimination*
assertion: the same test also checks that a deliberately corrupted variant of the estimator
is rejected.  A test that cannot reject the mutant is not a test.
"""

from __future__ import annotations

import hashlib
import math
import warnings

import numpy as np
import pytest
from scipy import stats

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
    pool = R.CenteringPool.from_conditions(
        base=base, near_safe=safe, near_critical=crit,
        declared_missing=("repeat", "far_critical"),   # analytic geometry, not a real cell
    )
    assert np.allclose(pool.mean, offset, atol=1e-12)

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
    pool_null = R.CenteringPool.from_conditions(
        base=base, near_safe=safe, near_critical=crit_null,
        declared_missing=("repeat", "far_critical"),
    )
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

    declared_pool = R.CenteringPool.from_conditions(
        base=base, repeat=repeat, near_safe=safe, near_critical=crit, far_critical=far
    )
    out = E.psi_distances_centered_cosine(base, crit, safe, centering_pool=declared_pool)

    assert out["centering_pool_n"] == 5 * n
    assert out["centering_pool"]["complete"] is True
    assert out["centering_pool"]["conditions"] == list(R.CENTERING_POOL_CONDITIONS)
    stacked = np.vstack([base, repeat, safe, crit, far])
    want_c = R.centered_cosine_distance(base, crit, mean=R.centering_mean(stacked))
    want_s = R.centered_cosine_distance(base, safe, mean=R.centering_mean(stacked))
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
    assert con.ci.method.startswith("two-sided paired Student-t")
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


def test_seed_interval_is_student_t_with_loso_not_seed_bootstrap():
    a = {1: 0.4, 2: 0.6, 3: 0.9, 4: 0.7, 5: 1.1}
    b = {seed: 0.2 for seed in a}
    first = E.model_contrast_seed_level(
        a, b, label_a="nextlat", label_b="bst", rng=np.random.default_rng(1), n_boot=100
    )
    second = E.model_contrast_seed_level(
        a, b, label_a="nextlat", label_b="bst", rng=np.random.default_rng(999), n_boot=9999
    )
    diffs = np.asarray([a[s] - b[s] for s in sorted(a)])
    expected_half_width = stats.t.ppf(.975, 4) * diffs.std(ddof=1) / np.sqrt(5)
    assert first.ci.ci_low == pytest.approx(diffs.mean() - expected_half_width)
    assert first.ci.as_dict() == second.ci.as_dict(), "seed bootstrap RNG must not affect Student-t"
    assert first.paired_standardized_effect == pytest.approx(diffs.mean() / diffs.std(ddof=1))
    assert [entry["omitted_seed"] for entry in first.leave_one_seed_out] == [1, 2, 3, 4, 5]
    assert all(entry["n_seeds"] == 4 for entry in first.leave_one_seed_out)


def test_npsi_formula_and_fail_closed_denominator():
    critical = np.asarray([2.0, 4.0, 6.0])
    safe = np.asarray([1.0, 1.0, 2.0])
    value, denominator = E.normalized_psi(critical, safe)
    assert denominator == 16.0
    assert value == pytest.approx(2.0 * 8.0 / 16.0)
    result = E.bootstrap_psi_items(
        critical, safe, rng=np.random.default_rng(0), n_boot=100
    )
    assert result.npsi == value
    assert result.as_dict()["npsi_role"].endswith("cannot rescue a co-primary result")
    with pytest.raises(ValueError, match="strictly positive and finite"):
        E.normalized_psi(np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        E.normalized_psi(np.asarray([1.0, np.nan]), np.ones(2))


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
    assert rep["r2_heldout_M1"] > 0.7
    assert rep["delta_r2_heldout"] > 0.0
    dirs = rep["distance_coefficient_directions_standardized"]
    assert dirs["sign_consistent"]
    assert dirs["signs"] == [1, 1]
    assert rep["extraction_index"] == R.BRANCH_MARGIN_INDEX == 63
    assert rep["folds_reused_exactly"] is True
    assert res.n_folds == 2

    # Planting the OPPOSITE sign must flip the reported direction.
    crit_margin_neg = 1.5 + 0.8 * base_margin - 6.0 * base_crit_dist + rng.normal(0, 0.5, n)
    out2 = E.fit_h2(crit_margin_neg, base_crit_dist, base_margin, rng=np.random.default_rng(9))
    assert out2["report"]["distance_coefficient_directions_standardized"]["signs"] == [-1, -1]

    # A pure-noise outcome must NOT produce a positive held-out R^2.
    out3 = E.fit_h2(rng.standard_normal(n), base_crit_dist, base_margin,
                    rng=np.random.default_rng(9))
    assert out3["report"]["r2_heldout_M1"] < 0.05


def test_h2_base_id_folds_are_exact_sha_parity_and_seed_free():
    # Digest-looking IDs are still UTF-8 text inputs to a *second* SHA-256.  These values
    # deliberately include known mismatches between that contract and ``int(id, 16) % 2``.
    ids = np.asarray([f"{value:064x}" for value in (0, 1, 2, 3, 7, 8)])
    expected = np.asarray([
        int(hashlib.sha256(base_id.encode("utf-8")).hexdigest(), 16) % 2
        for base_id in ids.tolist()
    ], dtype=np.int64)
    got = E.base_id_folds(ids)
    assert expected.tolist() == [1, 1, 1, 1, 0, 0]  # freeze known direct-parity mismatches
    assert got.tolist() == expected.tolist()
    assert got.tolist() != [int(base_id, 16) % 2 for base_id in ids.tolist()]
    assert np.array_equal(got, E.base_id_folds(ids))
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        E.base_id_folds(["not-a-digest"] * 4)


def test_h2_is_nested_incremental_on_identical_folds():
    rng = np.random.default_rng(808)
    n = 400
    base = rng.normal(size=n)
    distance = rng.normal(size=n)
    outcome = 2.0 * base + 3.0 * distance + rng.normal(scale=.1, size=n)
    ids = np.asarray([hashlib.sha256(f"base-{i}".encode()).hexdigest() for i in range(n)])
    folds = E.base_id_folds(ids)
    out = E.fit_h2(outcome, distance, base, folds=folds)
    report = out["report"]
    assert report["M0"]["model"] == "y ~ base_correct_branch_margin"
    assert report["M1"]["model"] == (
        "y ~ base_correct_branch_margin + base_critical_distance"
    )
    assert report["delta_r2_heldout"] == pytest.approx(
        report["r2_heldout_M1"] - report["r2_heldout_M0"]
    )
    assert np.array_equal(out["baseline_result"].fold_index, out["result"].fold_index)
    assert report["heldout_spearman_incremental"]["definition"].startswith("OOF")


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

    scored_ids = list(range(10_000, 10_040))          # disjoint from the fitting pool
    got = w.distance(a, b, item_ids=scored_ids)
    # A whitener fit WITH ids refuses to score a batch that supplies none: the held-out
    # claim has to stay checkable, not become opt-in at the call site.
    with pytest.raises(R.LeakageError, match="must supply"):
        w.distance(a, b)
    diff = a - b
    want = np.sqrt(np.einsum("ij,ij->i", diff, np.linalg.solve(w.covariance, diff.T).T))
    assert np.allclose(got, want, rtol=0, atol=1e-9)

    # W is a genuine inverse square root of the covariance actually used.
    assert np.allclose(w.transform @ w.transform, np.linalg.inv(w.covariance), atol=1e-9)
    assert np.allclose(w.transform, w.transform.T, atol=1e-12)
    # Discrimination: plain Euclidean is NOT the same thing on a correlated covariance.
    plain = np.sqrt(np.einsum("ij,ij->i", diff, diff))
    assert not np.allclose(got, plain, rtol=0.05)


def test_whitener_refuses_same_base_group_under_different_condition_row_ids():
    rng = np.random.default_rng(551)
    pool = rng.normal(size=(8, 4))
    item_ids = [f"base-{i // 2}:condition-{i % 2}" for i in range(8)]
    group_ids = [f"base-{i // 2}" for i in range(8)]
    whitener = R.Whitener.fit(pool, item_ids=item_ids, group_ids=group_ids)
    scored_a = rng.normal(size=(2, 4))
    scored_b = rng.normal(size=(2, 4))
    with pytest.raises(R.LeakageError, match="scored group"):
        whitener.distance(
            scored_a, scored_b,
            item_ids=["novel-row-a", "novel-row-b"],
            group_ids=["base-0", "new-base"],
        )
    clean = whitener.distance(
        scored_a, scored_b,
        item_ids=["novel-row-a", "novel-row-b"],
        group_ids=["new-base-a", "new-base-b"],
    )
    assert np.isfinite(clean).all()
    assert whitener.report()["group_leakage_checked"] is True

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
    pool = R.CenteringPool.from_conditions(
        base=base, near_safe=safe, near_critical=crit,
        declared_missing=("repeat", "far_critical"),
    )

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
    import subprocess
    import sys
    from pathlib import Path

    source_root = str(Path(R.__file__).resolve().parents[1])

    # Check import laziness in a fresh interpreter.  Inspecting this pytest
    # process is order-dependent: an unrelated earlier test may legitimately
    # have imported torch without representations.py being responsible for it.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                f"import sys; sys.path.insert(0, {source_root!r}); "
                "import lurestar.representations; "
                "assert 'torch' not in sys.modules, "
                "'representations import eagerly loaded torch'"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr
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


# =====================================================================================
# 10. The centering pool is CONSTRUCTED AND CHECKED, not merely named
#     (adversarial review, docs/review/representations.md findings 1 and 2)
# =====================================================================================


def _quartet_states(n=60, dim=12, seed=909):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, dim))
    return {
        "base": base,
        "repeat": base + 0.05 * rng.standard_normal((n, dim)),
        "near_safe": base + 0.25 * rng.standard_normal((n, dim)),
        "near_critical": base + 0.55 * rng.standard_normal((n, dim)),
        "far_critical": base + 3.0 + 1.5 * rng.standard_normal((n, dim)),
    }


def test_psi_refuses_a_bare_array_as_the_centering_pool():
    """The mutation that used to survive was the CALLER, not the implementation.

    Passing the scored triple as a raw ndarray produced a plausible number and no
    complaint.  It is now a TypeError, so the wrong pool is inexpressible rather than
    merely discouraged.
    """
    st = _quartet_states()
    with pytest.raises(TypeError, match="CenteringPool.from_conditions"):
        E.psi_distances_centered_cosine(
            st["base"], st["near_critical"], st["near_safe"],
            centering_pool=np.vstack([st["base"], st["near_critical"], st["near_safe"]]),
        )


def test_centering_pool_refuses_to_drop_a_condition_silently():
    """Omitting far_critical must be a recorded statement, never an omission."""
    st = _quartet_states()
    with pytest.raises(ValueError, match="neither supplied nor listed in declared_missing"):
        R.CenteringPool.from_conditions(
            base=st["base"], near_safe=st["near_safe"], near_critical=st["near_critical"]
        )
    # Declared explicitly: allowed, and the omission travels with the number.
    pool = R.CenteringPool.from_conditions(
        base=st["base"], near_safe=st["near_safe"], near_critical=st["near_critical"],
        declared_missing=("repeat", "far_critical"),
    )
    out = E.psi_distances_centered_cosine(
        st["base"], st["near_critical"], st["near_safe"], centering_pool=pool
    )
    assert out["centering_pool"]["complete"] is False
    assert out["centering_pool"]["declared_missing"] == ["far_critical", "repeat"]
    # ...and it really is a different number from the complete pool, which is why the
    # declaration has to be visible in the serialized metric.
    full = R.CenteringPool.from_conditions(**st)
    out_full = E.psi_distances_centered_cosine(
        st["base"], st["near_critical"], st["near_safe"], centering_pool=full
    )
    assert out_full["centering_pool"]["complete"] is True
    psi_part = float(E.psi_items(out["d_critical"], out["d_safe"]).mean())
    psi_full = float(E.psi_items(out_full["d_critical"], out_full["d_safe"]).mean())
    assert abs(psi_part - psi_full) > 1e-3, (
        "if dropping far_critical did not move PSI here, this test could not detect the "
        "substitution it exists to detect"
    )


def test_centering_pool_rejects_a_pool_from_a_different_cell():
    """A pool of the right SHAPE but the wrong states must be rejected outright.

    This is the probe that used to pass: seven rows of pure noise silently collapsed PSI
    from 0.078 to 0.0004 with no error and a cheerful `centering_pool_n` of 7.
    """
    st = _quartet_states()
    other = _quartet_states(seed=1010)          # a different (model, seed) cell
    wrong = R.CenteringPool.from_conditions(**other)
    with pytest.raises(ValueError, match="not in the centering pool"):
        E.psi_distances_centered_cosine(
            st["base"], st["near_critical"], st["near_safe"], centering_pool=wrong
        )
    # Each scored argument is checked on its own, so a pool that is right for two of the
    # three still fails on the third.  Without a case per argument, dropping any single
    # containment check would leave the suite green.
    base_only_foreign = R.CenteringPool.from_conditions(
        base=other["base"], repeat=st["repeat"], near_safe=st["near_safe"],
        near_critical=st["near_critical"], far_critical=st["far_critical"],
    )
    with pytest.raises(ValueError, match="`h_base`"):
        E.psi_distances_centered_cosine(
            st["base"], st["near_critical"], st["near_safe"],
            centering_pool=base_only_foreign,
        )
    safe_only_foreign = R.CenteringPool.from_conditions(
        base=st["base"], repeat=st["repeat"], near_safe=other["near_safe"],
        near_critical=st["near_critical"], far_critical=st["far_critical"],
    )
    with pytest.raises(ValueError, match="`h_near_safe`"):
        E.psi_distances_centered_cosine(
            st["base"], st["near_critical"], st["near_safe"],
            centering_pool=safe_only_foreign,
        )

    # Only one condition swapped is still a leak of the wrong cell, and still caught.
    mixed = R.CenteringPool.from_conditions(
        base=st["base"], repeat=st["repeat"], near_safe=st["near_safe"],
        near_critical=other["near_critical"], far_critical=st["far_critical"],
    )
    with pytest.raises(ValueError, match="`h_near_critical`"):
        E.psi_distances_centered_cosine(
            st["base"], st["near_critical"], st["near_safe"], centering_pool=mixed
        )


def test_centering_pool_rejects_unknown_and_double_declared_conditions():
    st = _quartet_states()
    with pytest.raises(ValueError, match="unknown centering-pool condition"):
        R.CenteringPool.from_conditions(base=st["base"], nearsafe=st["near_safe"])
    with pytest.raises(ValueError, match="both supplied and declared missing"):
        R.CenteringPool.from_conditions(
            **st, declared_missing=("far_critical",)
        )
    pool = R.CenteringPool.from_conditions(**st)
    assert pool.n == 5 * 60
    assert pool.counts == (60, 60, 60, 60, 60)
    with pytest.raises(ValueError, match="were not pooled"):
        R.CenteringPool.from_conditions(
            base=st["base"], repeat=st["repeat"], near_critical=st["near_critical"],
            far_critical=st["far_critical"], declared_missing=("near_safe",),
        ).require_conditions("base", "near_safe", "near_critical")


def test_pool_mean_equals_the_stacked_mean_and_ordering_is_canonical():
    """Determinism: the mean must not depend on the order the kwargs were written in."""
    st = _quartet_states()
    a = R.CenteringPool.from_conditions(**st)
    shuffled = {k: st[k] for k in ("far_critical", "near_safe", "base", "near_critical", "repeat")}
    b = R.CenteringPool.from_conditions(**shuffled)
    assert a.conditions == b.conditions == R.CENTERING_POOL_CONDITIONS
    assert np.array_equal(a.mean, b.mean)
    stacked = np.vstack([st[c] for c in R.CENTERING_POOL_CONDITIONS])
    assert np.allclose(a.mean, stacked.mean(axis=0), rtol=0, atol=1e-15)


# =====================================================================================
# 11. The leakage guard survives heterogeneous item ids
# =====================================================================================


def test_leakage_guard_is_not_defeated_by_mixed_id_types():
    """A leaked INTEGER id used to vanish when the batch also carried a string id.

    `frozenset(np.asarray(list(ids)).tolist())` casts [3, "x"] to ["3", "x"], so the
    integer 3 stopped matching the integer 3 in the fitting pool and the guard reported
    clean.  The fitting pool ids here are exactly the ones in the module's own example.
    """
    rng = np.random.default_rng(4)
    pool = rng.standard_normal((10, 4))
    w = R.Whitener.fit(pool, item_ids=list(range(10)))
    a, b = rng.standard_normal((2, 4)), rng.standard_normal((2, 4))
    with pytest.raises(R.LeakageError):
        w.distance(a, b, item_ids=[3, 4])
    with pytest.raises(R.LeakageError):
        w.distance(a, b, item_ids=[3, "an_unrelated_id"])
    w.distance(a, b, item_ids=[900, "an_unrelated_id"])          # genuinely disjoint

    # The same coercion at FIT time is just as fatal: the pool's own integer id 3 would
    # be stored as the string "3" and would then never match the integer 3 being scored.
    w_mixed = R.Whitener.fit(pool[:2], item_ids=[3, "a_string_id"])
    assert w_mixed.fit_item_ids == frozenset({3, "a_string_id"})
    with pytest.raises(R.LeakageError):
        w_mixed.distance(a, b, item_ids=[3, 4])
    with pytest.raises(R.LeakageError):
        w_mixed.distance(a, b, item_ids=["a_string_id", 4])


def test_reported_whitened_metric_demands_a_checkable_heldout_claim():
    rng = np.random.default_rng(5)
    st = _quartet_states(n=20, dim=4)
    unchecked = R.Whitener.fit(rng.standard_normal((50, 4)))     # fit without ids
    assert unchecked.report()["pool_is_heldout"] is False
    with pytest.raises(R.LeakageError, match="fit without item_ids"):
        E.psi_distances_whitened(
            st["base"], st["near_critical"], st["near_safe"],
            whitener=unchecked, item_ids=[f"q{i}" for i in range(20)],
        )
    checked = R.Whitener.fit(
        rng.standard_normal((50, 4)), item_ids=[f"h{i}" for i in range(50)]
    )
    with pytest.raises(R.LeakageError, match="item_ids is required"):
        E.psi_distances_whitened(
            st["base"], st["near_critical"], st["near_safe"], whitener=checked
        )
    ok = E.psi_distances_whitened(
        st["base"], st["near_critical"], st["near_safe"],
        whitener=checked, item_ids=[f"q{i}" for i in range(20)],
    )
    assert ok["whitener"]["pool_is_heldout"] is True


# =====================================================================================
# 12. LAYER B: the GPT/NextLat asymmetry, made executable without a GPU
#
# `forward_states_and_logits` is the crux of this track: GPT returns (logits, h) while
# NextLat early-returns (token_embeds, text_embd) BEFORE lm_head.  Getting it backwards
# yields plausible-looking numbers and destroys every H1/H2/H3 result.  Until now the
# whole of Layer B was untested — deleting the architecture check, or applying lm_head to
# GPT's own logits, left the suite green.  These tests inject a numpy-backed stub `torch`
# so both branches actually run on a CPU-only host.
# =====================================================================================


class _FakeTensor:
    """Just enough tensor for Layer B: it does no arithmetic beyond what Layer B does."""

    def __init__(self, a):
        self.a = np.asarray(a)

    @property
    def ndim(self):
        return self.a.ndim

    @property
    def shape(self):
        return self.a.shape

    @property
    def device(self):
        return "cpu"

    def __getitem__(self, k):
        return _FakeTensor(self.a[k])

    def to(self, device):
        return self

    def index_select(self, dim, index):
        return _FakeTensor(np.take(self.a, np.asarray(index.a), axis=dim))

    def float(self):
        return _FakeTensor(self.a.astype(np.float32))

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def detach(self):
        return self

    # --- the handful of ops the BST branch needs (model_bst.py:92-100, :803-809) ---

    @property
    def dtype(self):
        return self.a.dtype

    def new_full(self, shape, value):
        return _FakeTensor(np.full(shape, value, dtype=self.a.dtype))

    def expand(self, *sizes):
        return _FakeTensor(np.broadcast_to(self.a, sizes))

    def chunk(self, n, dim=-1):
        return tuple(_FakeTensor(c) for c in np.split(self.a, n, axis=dim))

    def __add__(self, other):
        o = other.a if isinstance(other, _FakeTensor) else other
        return _FakeTensor(self.a + o)


def _install_stub_torch(monkeypatch):
    import contextlib
    import sys
    import types

    stub = types.ModuleType("torch")
    stub.long = np.int64

    def as_tensor(x, dtype=None, device=None):
        if isinstance(x, _FakeTensor):
            return x
        return _FakeTensor(np.asarray(x, dtype=dtype))

    stub.as_tensor = as_tensor

    # A real generator, not `iter([None])`: Layer B raises inside `with torch.no_grad()`
    # for several of its guard clauses, and a fake context manager that cannot take a
    # throw() turns every one of those ValueErrors into an unrelated AttributeError.
    @contextlib.contextmanager
    def no_grad():
        yield None

    stub.no_grad = no_grad

    def cat(tensors, dim=-1):
        return _FakeTensor(np.concatenate([t.a for t in tensors], axis=dim))

    def allclose(a, b, atol=1e-8, rtol=1e-5):
        return bool(np.allclose(a.a, b.a, atol=atol, rtol=rtol))

    stub.cat = cat
    stub.allclose = allclose
    monkeypatch.setitem(sys.modules, "torch", stub)
    return stub


# One fixed projection, so "logits" are a deterministic function of the state.
_HEAD = np.arange(6 * 5, dtype=np.float64).reshape(6, 5) / 7.0 - 1.0


def _embed(tokens):
    """(B, T) ints -> (B, T, 6) 'token embeddings'.  NOT the post-norm state."""
    t = np.asarray(tokens, dtype=np.float64)
    return np.stack([t + k for k in range(6)], axis=-1)


def _post_norm(tokens):
    """(B, T) ints -> (B, T, 6) 'transformer.norm(x)'.  Deliberately != _embed."""
    return np.tanh(_embed(tokens) * 0.3) + 0.5


class _FakeGPT:
    """Mimics upstream models/model_gpt.py:276-291 with targets=None."""

    def __init__(self):
        self.training = False
        self.calls = 0

    def eval(self):
        pass

    # Deliberately NOT the projection used inside __call__.  Upstream the two coincide
    # (model_gpt.py:280 is literally `self.lm_head(x)`), but the contract is "use the
    # value the model returned", not "recompute it from the state with whatever head
    # attribute happens to be reachable" — a distinction that matters the moment a head
    # is tied, fused or quantized.  Making them differ here turns that contract into a
    # test instead of a comment.
    def lm_head(self, h):
        return _FakeTensor(np.asarray(h.a) @ (_HEAD + 3.0))

    def __call__(self, tokens, return_hidden_states=False, **kw):
        self.calls += 1
        h = _post_norm(tokens.a if isinstance(tokens, _FakeTensor) else tokens)
        logits = h @ _HEAD                      # model_gpt.py:280  output = lm_head(x)
        if return_hidden_states:
            return _FakeTensor(logits), _FakeTensor(h)      # model_gpt.py:291
        return _FakeTensor(logits)


class _FakeNextLat:
    """Mimics upstream models/model_nextlat.py:192-200 — the EARLY return."""

    def __init__(self):
        self.training = False
        self.calls = 0

    def eval(self):
        pass

    def lm_head(self, h):                        # model_nextlat.py:121
        return _FakeTensor(np.asarray(h.a) @ _HEAD)

    def __call__(self, tokens, return_hidden_states=False, **kw):
        self.calls += 1
        tok = tokens.a if isinstance(tokens, _FakeTensor) else tokens
        token_embeds = _embed(tok)               # model_nextlat.py:193
        text_embd = _post_norm(tok)              # model_nextlat.py:197
        if return_hidden_states:
            # model_nextlat.py:199-200 — returns BEFORE lm_head is ever applied.
            return _FakeTensor(token_embeds), _FakeTensor(text_embd)
        return _FakeTensor(text_embd @ _HEAD)


_STUB_TOKENS = np.array(
    [[3, 7, 2, 9, 4, 1], [5, 5, 8, 0, 6, 2], [1, 2, 3, 4, 5, 6]], dtype=np.int64
)


def test_layer_b_gpt_uses_the_returned_logits_and_does_not_reapply_the_head(monkeypatch):
    _install_stub_torch(monkeypatch)
    m = _FakeGPT()
    h, logits = R.forward_states_and_logits(m, _FakeTensor(_STUB_TOKENS), architecture="gpt")
    want_h = _post_norm(_STUB_TOKENS)
    assert np.allclose(h.a, want_h)
    assert np.allclose(logits.a, want_h @ _HEAD)
    # Discrimination.  The state has 6 features and the head projects to a 5-token vocab,
    # so the two returns are not interchangeable: swapping them (returning the logits as
    # the hidden state, which is what dropping the GPT branch would do) changes the last
    # dimension and is caught here rather than 20,000 items later.
    assert h.a.shape == (3, 6, 6) and logits.a.shape == (3, 6, 5)
    assert h.a.shape[-1] != logits.a.shape[-1]
    # ...and the logits are the model's OWN first return, not a recomputation through a
    # separately reachable head attribute.
    assert not np.allclose(logits.a, m.lm_head(_FakeTensor(want_h)).a)


def test_layer_b_nextlat_applies_lm_head_and_never_returns_token_embeds(monkeypatch):
    _install_stub_torch(monkeypatch)
    m = _FakeNextLat()
    h, logits = R.forward_states_and_logits(
        m, _FakeTensor(_STUB_TOKENS), architecture="nextlat"
    )
    want_h = _post_norm(_STUB_TOKENS)
    token_embeds = _embed(_STUB_TOKENS)
    assert np.allclose(h.a, want_h), "the SECOND return value is the post-norm state"
    assert not np.allclose(h.a, token_embeds), (
        "returning token_embeds as the hidden state is the silent way to destroy every "
        "geometry result; the two must be distinguishable here"
    )
    assert np.allclose(logits.a, want_h @ _HEAD), "lm_head must be applied by the caller"
    # The GPT branch applied to NextLat would return `first` — the token embeddings — as
    # the logits.  Those have the state's 6 features, not the vocabulary's 5.
    assert logits.a.shape == (3, 6, 5) and token_embeds.shape == (3, 6, 6)
    # And applying lm_head to token_embeds instead of to the post-norm state is a
    # different array of the SAME shape, so shape alone would not have caught it.
    assert (token_embeds @ _HEAD).shape == logits.a.shape
    assert not np.allclose(logits.a, token_embeds @ _HEAD)


def test_layer_b_architecture_name_is_validated_before_torch_is_touched():
    """Reachable on a host with no torch, which is the only way it is ever checked."""
    for bad in ("mamba", "", "belief_state_transformer"):
        with pytest.raises(ValueError, match="must be 'gpt', 'nextlat' or 'bst'"):
            R.forward_states_and_logits(object(), None, architecture=bad)
    assert R.ARCHITECTURES == ("nextlat", "bst", "gpt")
    assert set(R.STATE_SOURCE) == set(R.ARCHITECTURES) == set(R.HIDDEN_STATE_MODULE_PATH)


def test_extract_positions_returns_both_frozen_indices_for_both_architectures(monkeypatch):
    _install_stub_torch(monkeypatch)
    want_h = _post_norm(_STUB_TOKENS)
    for arch, model in (("gpt", _FakeGPT()), ("nextlat", _FakeNextLat())):
        out = R.extract_positions(
            model, _STUB_TOKENS, architecture=arch, positions=(2, 3), batch_size=2
        )
        assert model.calls == 2, "batching must actually chunk"
        assert out["positions"].tolist() == [2, 3]
        assert out["hidden"].shape == (3, 2, 6)
        assert out["logits"].shape == (3, 2, 5)
        assert np.allclose(out["hidden"], want_h[:, [2, 3], :], atol=1e-6)
        assert np.allclose(out["logits"], (want_h @ _HEAD)[:, [2, 3], :], atol=1e-5)
        # Order matters: position 2 is not position 3.
        assert not np.allclose(out["hidden"][:, 0], out["hidden"][:, 1])


def test_extract_positions_defaults_to_the_frozen_pair_and_rejects_bad_positions(monkeypatch):
    _install_stub_torch(monkeypatch)
    tokens = np.tile(np.arange(69, dtype=np.int64), (2, 1))
    out = R.extract_positions(_FakeGPT(), tokens, architecture="gpt", batch_size=8)
    assert out["positions"].tolist() == [R.PSI_EXTRACTION_INDEX, R.BRANCH_MARGIN_INDEX] == [62, 63]
    with pytest.raises(ValueError, match="outside the sequence"):
        R.extract_positions(_FakeGPT(), tokens, architecture="gpt", positions=(62, 69))
    with pytest.raises(ValueError, match="outside the sequence"):
        R.extract_positions(_FakeGPT(), tokens, architecture="gpt", positions=(-1, 63))


# =====================================================================================
# 13. LAYER B: BST — the third arm, whose "final post-norm state" is a CHOICE
#
# BST is the only arm where the final post-normalization state and the immediate
# pre-logit state are different tensors, and where a second, answer-contaminated state
# exists a single attribute away.  The whole argument of docs/EXTRACTION.md §3 is made
# executable here:
#
#   * `hidden` is the FORWARD encoder's post-norm state (model_bst.py:287), never the
#     backward one (model_bst.py:313) and never the TextHead's chunk;
#   * the backward encoder genuinely sees the future, so mistaking it for the analogue
#     would put the answer inside PSI — the fake below is reverse-causal precisely so
#     that this is a failing assertion rather than a paragraph of prose;
#   * `logits` come from TextHead (model_bst.py:83-110) with the lone-EOS backward
#     embedding of BST.generate (model_bst.py:803-809), never from lm_head(hidden).
# =====================================================================================


# TextHead's internal 2D->2D MLP and the shared D->V projection.  _HEAD is reused so the
# BST head is a genuinely different function of `hidden` than GPT's head is.
_BST_MLP = np.cos(np.arange(12 * 12, dtype=np.float64).reshape(12, 12) / 5.0) / 3.0


def _bwd_post_norm(tokens):
    """(B, T) ints -> (B, T, 6) 'transformer_b.norm(bwd)'.

    Deliberately REVERSE-CAUSAL: position i is a function of tokens i..T-1, mirroring the
    triu document mask at model_bst.py:215-217.  Two consequences the tests below rely on:
    the state at the final position depends on that token alone (so it equals the lone-EOS
    embedding, exactly as upstream's terminal EOS does under the same mask), and the state
    at index i changes when the SUFFIX changes — which is what makes it unusable for PSI.
    """
    t = np.asarray(tokens, dtype=np.float64)
    n = t.shape[1]
    rev_mean = np.cumsum(t[:, ::-1], axis=1)[:, ::-1] / np.arange(n, 0, -1)
    # 0.01, not 0.3: token ids run to 104, and a 0.3 scale saturates tanh so that every
    # backward state collapses to 1.0 and the suffix-sensitivity this fake exists to
    # demonstrate would vanish into floating-point noise.
    return np.tanh(np.stack([rev_mean + k for k in range(6)], axis=-1) * 0.01) - 0.25


class _FakeBSTEncoder:
    """Mimics models/model_bst.py:245-315 — two stacks, two norms, one call."""

    def __init__(self):
        self.calls = []

    def __call__(self, batch, compute_forward=True, compute_backward=True):
        tok = batch.a if isinstance(batch, _FakeTensor) else batch
        self.calls.append((tuple(np.shape(tok)), compute_forward, compute_backward))
        fwd = _FakeTensor(_post_norm(tok)) if compute_forward else None   # :287
        bwd = _FakeTensor(_bwd_post_norm(tok)) if compute_backward else None  # :313
        return fwd, bwd


class _FakeBSTTextHead:
    """Mimics models/model_bst.py:83-110 with targets=None."""

    def __init__(self):
        self.calls = 0

    def mlp(self, x):                                    # model_bst.py:64-68
        return _FakeTensor(np.tanh(x.a @ _BST_MLP))

    def norm(self, x):                                   # model_bst.py:69 (2*n_embd)
        a = x.a
        return _FakeTensor(a / np.sqrt((a**2).mean(axis=-1, keepdims=True) + 1e-5))

    def lm_head(self, h):                                # model_bst.py:70
        return _FakeTensor(np.asarray(h.a) @ _HEAD)

    def __call__(self, forward_embedding, backward_embedding, **kw):
        self.calls += 1
        x = _FakeTensor(np.concatenate([forward_embedding.a, backward_embedding.a], -1))
        x = x + self.mlp(x)                              # model_bst.py:95
        x = self.norm(x)                                 # model_bst.py:96
        x_next, x_prev = x.chunk(2, dim=-1)              # model_bst.py:100
        stacked = np.stack(
            [self.lm_head(x_next).a, self.lm_head(x_prev).a], axis=1
        )                                                # model_bst.py:109
        return _FakeTensor(stacked)


class _FakeBST:
    """Mimics the BST wrapper (models/model_bst.py:327-340): .encoder and .text_head."""

    def __init__(self):
        self.training = False
        self.encoder = _FakeBSTEncoder()
        self.text_head = _FakeBSTTextHead()

    def eval(self):
        pass


# Two sequences with an IDENTICAL prefix (positions 0..3) and different suffixes, plus a
# terminal EOS.  This is the shape of a Lure-Star quartet: the manipulation lives before
# the extraction index, the answer lives after it.
_BST_TOKENS = np.array(
    [[3, 7, 2, 9, 11, 4, R.EOS_TOKEN_ID], [3, 7, 2, 9, 88, 62, R.EOS_TOKEN_ID]],
    dtype=np.int64,
)


def _bst_expected(tokens):
    """Independent recomputation of what Layer B must return for BST."""
    tok = np.asarray(tokens)
    b, t = tok.shape
    fwd = _post_norm(tok)
    eos = np.full((b, 1), R.EOS_TOKEN_ID, dtype=tok.dtype)
    bwd = np.broadcast_to(_bwd_post_norm(eos), (b, t, 6))
    head = _FakeBSTTextHead()
    logits = head(_FakeTensor(fwd), _FakeTensor(bwd)).a[:, 0]
    x = np.concatenate([fwd, bwd], -1)
    x = x + np.tanh(x @ _BST_MLP)
    x = x / np.sqrt((x**2).mean(axis=-1, keepdims=True) + 1e-5)
    x_next = np.split(x, 2, axis=-1)[0]
    return fwd, bwd, logits, x_next


def test_layer_b_bst_returns_the_forward_state_and_never_the_backward_one(monkeypatch):
    _install_stub_torch(monkeypatch)
    m = _FakeBST()
    out = R.forward_all_states(m, _FakeTensor(_BST_TOKENS), architecture="bst")
    want_fwd, _want_bwd, want_logits, want_xnext = _bst_expected(_BST_TOKENS)

    assert np.allclose(out["hidden"].a, want_fwd), "hidden is transformer_f.norm (:287)"
    assert np.allclose(out["logits"].a, want_logits)
    assert np.allclose(out["hidden_texthead"].a, want_xnext)

    # Discrimination 1: the backward post-norm state has the same shape as the forward
    # one, so shape alone can never catch a swap.  It must differ numerically.
    bwd_full = _bwd_post_norm(_BST_TOKENS)
    assert bwd_full.shape == want_fwd.shape
    assert not np.allclose(out["hidden"].a, bwd_full)

    # Discrimination 2: the TextHead pre-logit state is ALSO (B, T, 6).  Returning it as
    # `hidden` is the other silent substitution, and it is caught here.
    assert out["hidden_texthead"].a.shape == out["hidden"].a.shape
    assert not np.allclose(out["hidden"].a, out["hidden_texthead"].a)

    # Discrimination 3: BST logits are NOT lm_head(hidden).  That call is type-compatible
    # — 6 features in, 5 vocab out — and semantically wrong, because lm_head is trained on
    # the post-MLP, post-norm, chunked half (model_bst.py:100-105).
    naive = m.text_head.lm_head(_FakeTensor(want_fwd)).a
    assert naive.shape == out["logits"].a.shape
    assert not np.allclose(out["logits"].a, naive)


def test_bst_backward_state_sees_the_answer_which_is_why_it_is_excluded(monkeypatch):
    """The concrete reason docs/EXTRACTION.md §3 excludes the backward encoder.

    Two items share tokens 0..3 and differ at 4..5.  The forward state at index 3 is
    identical between them — it is a function of the prefix, like GPT's and NextLat's.
    The backward state at the SAME index is not, because under the reverse-causal mask it
    has already read the suffix.  Centering, whitening and PSI would all faithfully report
    that difference, and it would be a difference in the answer, not in the history.
    """
    _install_stub_torch(monkeypatch)
    m = _FakeBST()
    out = R.forward_all_states(m, _FakeTensor(_BST_TOKENS), architecture="bst")
    idx = 3
    assert np.allclose(out["hidden"].a[0, idx], out["hidden"].a[1, idx], atol=1e-12), (
        "the forward state at a shared-prefix index must be identical across the two "
        "items; if it is not, the fake encoder is not causal and this test is vacuous"
    )
    bwd = _bwd_post_norm(_BST_TOKENS)
    assert not np.allclose(bwd[0, idx], bwd[1, idx], atol=1e-6)
    assert float(np.abs(bwd[0, idx] - bwd[1, idx]).max()) > 1e-3
    assert "EXCLUDED" in R.BST_STATE_ROLES["excluded_backward"]


def test_bst_backward_input_is_the_lone_eos_of_BST_generate(monkeypatch):
    """model_bst.py:803-809: the inference backward embedding is a single EOS token.

    Two properties are asserted, and both are load-bearing.  It is ITEM-INDEPENDENT, so
    every item-to-item difference in BST's logits is driven by the forward state alone.
    And it equals the backward state of the sequence's own terminal EOS, which is what
    makes it a trained endpoint rather than an off-distribution stand-in — under the
    reverse-causal document mask the last token attends only itself.
    """
    _install_stub_torch(monkeypatch)
    m = _FakeBST()
    bwd = R.bst_backward_eos_state(m.encoder, _FakeTensor(_BST_TOKENS))
    assert bwd.a.shape == (2, 1, 6)
    assert np.allclose(bwd.a[0], bwd.a[1], atol=1e-12), "item-independent by construction"
    terminal = _bwd_post_norm(_BST_TOKENS)[:, -1]
    assert np.allclose(bwd.a[:, 0], terminal, atol=1e-12)
    # The encoder was asked for the backward stack only — running the forward stack on a
    # 1-token EOS batch would be wasted work and is what upstream avoids.
    assert m.encoder.calls[-1] == ((2, 1), False, True)


def test_bst_texthead_reimplementation_is_checked_against_the_head_itself(monkeypatch):
    """`bst_texthead_prelogit` duplicates model_bst.py:92-100; the copy must be verified.

    Layer B asserts that lm_head applied to our re-derived x_next reproduces the head's
    OWN returned next-token logits.  Break the head and the assertion must fire, or the
    duplication is a silent fork of upstream.
    """
    _install_stub_torch(monkeypatch)
    m = _FakeBST()
    R.forward_all_states(m, _FakeTensor(_BST_TOKENS), architecture="bst")  # passes

    class _DriftedHead(_FakeBSTTextHead):
        def __call__(self, fwd, bwd, **kw):
            out = super().__call__(fwd, bwd, **kw)
            return _FakeTensor(out.a + 1.0)          # upstream changed under us

    m.text_head = _DriftedHead()
    with pytest.raises(RuntimeError, match="no longer reproduces TextHead"):
        R.forward_all_states(m, _FakeTensor(_BST_TOKENS), architecture="bst")
    # ...and the check is skippable only by asking for it explicitly.
    R.forward_all_states(
        m, _FakeTensor(_BST_TOKENS), architecture="bst", verify_texthead=False
    )


def test_bst_demands_the_wrapper_and_refuses_an_external_mask(monkeypatch):
    """The argument asymmetry BST forces, made into two named errors."""
    _install_stub_torch(monkeypatch)

    class _JustAnEncoder:                      # what wrapper.model would give you
        pass

    with pytest.raises(ValueError, match=r"\.encoder and \.text_head"):
        R.forward_all_states(
            _JustAnEncoder(), _FakeTensor(_BST_TOKENS), architecture="bst"
        )
    with pytest.raises(ValueError, match="builds its own forward/backward document masks"):
        R.forward_all_states(
            _FakeBST(), _FakeTensor(_BST_TOKENS), architecture="bst", mask=object()
        )


def test_extract_positions_carries_bst_secondary_state_and_only_for_bst(monkeypatch):
    _install_stub_torch(monkeypatch)
    tokens = np.tile(np.arange(69, dtype=np.int64), (4, 1))
    tokens[:, -1] = R.EOS_TOKEN_ID

    bst = R.extract_positions(_FakeBST(), tokens, architecture="bst", batch_size=2)
    assert bst["architecture"] == "bst"
    assert bst["state_source"].startswith("models/model_bst.py:287")
    assert bst["positions"].tolist() == [62, 63]
    assert bst["hidden"].shape == (4, 2, 6)
    assert bst["logits"].shape == (4, 2, 5)
    assert bst["hidden_texthead"].shape == (4, 2, 6)
    assert np.allclose(bst["hidden"], _post_norm(tokens)[:, [62, 63], :], atol=1e-6)
    # Batching must not change the lone-EOS backward embedding, which is built per chunk.
    assert np.allclose(bst["hidden_texthead"][0], bst["hidden_texthead"][3], atol=1e-6)

    for arch, model in (("gpt", _FakeGPT()), ("nextlat", _FakeNextLat())):
        out = R.extract_positions(model, tokens, architecture=arch, batch_size=2)
        assert "hidden_texthead" not in out, (
            "only BST has a second candidate state; inventing one for the other arms "
            "would make the three-arm extraction look symmetric when it is not"
        )
        assert out["architecture"] == arch
        assert out["state_source"] == R.STATE_SOURCE[arch]


def test_intermediate_hooks_return_fixed_12x2_stack_verify_parity_and_remove():
    torch = pytest.importorskip("torch")

    class Block(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value

        def forward(self, x, **_kwargs):
            return x + self.value

    class Inner(torch.nn.Module):
        def __init__(self, corrupt=False):
            super().__init__()
            self.corrupt = corrupt
            self.token_embedding = torch.nn.Embedding(106, 6)
            self.transformer = torch.nn.ModuleDict({
                "blocks": torch.nn.ModuleList([Block(float(i + 1)) for i in range(12)]),
                "norm": torch.nn.LayerNorm(6),
            })
            self.lm_head = torch.nn.Linear(6, 106, bias=False)

        def forward(self, tokens, return_hidden_states=False, **_kwargs):
            x = self.token_embedding(tokens)
            for block in self.transformer.blocks:
                x = block(x)
            hidden = self.transformer.norm(x)
            if self.corrupt:
                hidden = hidden + 1.0
            logits = self.lm_head(hidden)
            return (logits, hidden) if return_hidden_states else logits

    tokens = np.zeros((4, 69), dtype=np.int64)
    model = Inner()
    result = R.extract_positions(
        model, tokens, architecture="gpt", batch_size=2, capture_blocks=True
    )
    assert result["intermediate_hidden"].shape == (4, 12, 2, 6)
    assert all(len(block._forward_hooks) == 0 for block in model.transformer.blocks)

    corrupt = Inner(corrupt=True)
    with pytest.raises(RuntimeError, match="block 11 plus final norm"):
        R.extract_positions(
            corrupt, tokens, architecture="gpt", batch_size=2, capture_blocks=True
        )
    assert all(len(block._forward_hooks) == 0 for block in corrupt.transformer.blocks)


def test_bst_intermediate_hooks_capture_forward_stack_only():
    torch = pytest.importorskip("torch")

    class Block(torch.nn.Module):
        def forward(self, x, **_kwargs):
            return x + .1

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(106, 6)
            self.transformer_f = torch.nn.ModuleDict({
                "blocks": torch.nn.ModuleList([Block() for _ in range(12)]),
                "norm": torch.nn.LayerNorm(6),
            })
            self.transformer_b = torch.nn.ModuleDict({
                "blocks": torch.nn.ModuleList([Block() for _ in range(12)]),
                "norm": torch.nn.LayerNorm(6),
            })

        def forward(self, tokens, compute_forward=True, compute_backward=True):
            def run(stack):
                x = self.embedding(tokens)
                for block in stack.blocks:
                    x = block(x)
                return stack.norm(x)
            return (
                run(self.transformer_f) if compute_forward else None,
                run(self.transformer_b) if compute_backward else None,
            )

    class TextHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.mlp = torch.nn.Linear(12, 12)
            torch.nn.init.zeros_(self.mlp.weight)
            torch.nn.init.zeros_(self.mlp.bias)
            self.norm = torch.nn.Identity()
            self.lm_head = torch.nn.Linear(6, 106, bias=False)

        def forward(self, fwd, bwd):
            x = torch.cat([fwd, bwd], dim=-1)
            x = x + self.mlp(x)
            next_state, prev_state = self.norm(x).chunk(2, dim=-1)
            return torch.stack([self.lm_head(next_state), self.lm_head(prev_state)], dim=1)

    class BST(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = Encoder()
            self.text_head = TextHead()

    model = BST()
    result = R.extract_positions(
        model, np.zeros((2, 69), dtype=np.int64), architecture="bst",
        capture_blocks=True,
    )
    assert result["intermediate_hidden"].shape == (2, 12, 2, 6)
    assert all(len(block._forward_hooks) == 0 for block in model.encoder.transformer_f.blocks)
    assert all(len(block._forward_hooks) == 0 for block in model.encoder.transformer_b.blocks)


# =====================================================================================
# 14. THE THREE-ARM DESIGN
#
# Three things have to hold before any of these numbers may appear in a writeup.
#   (a) PSI is recovered per arm and the RANKING between arms survives the pipeline;
#   (b) the cross-model contrast is a statement about SEEDS and cannot quietly become a
#       statement about items, which would be ~80x too precise while estimating something
#       else entirely;
#   (c) the "what could three seeds not have seen" number behaves like a power curve —
#       monotone in n — rather than like a constant someone typed in.
# =====================================================================================


def _mirrored_cell(alpha_s, alpha_c, *, dim, n_pairs, rng):
    """One (arm, seed) cell whose PSI is exactly ``cos(alpha_s) - cos(alpha_c)``.

    Same construction as section 3: mirrored item pairs put the pool mean exactly at the
    cell's own offset, so centered cosine reduces to raw cosine and the PSI is analytic
    rather than approximate.  Each cell gets its own random basis and its own offset, so
    an implementation that pooled the centering mean ACROSS arms or across seeds — which
    is a real and easy mistake once three arms exist — would not reproduce these values.
    """
    e0, e1, e2 = orthonormal_basis(dim, 3, rng)
    a = e0
    s = math.cos(alpha_s) * e0 + math.sin(alpha_s) * e1
    c = math.cos(alpha_c) * e0 + math.sin(alpha_c) * e2
    offset = 4.0 * rng.standard_normal(dim)
    base = np.vstack([a] * n_pairs + [-a] * n_pairs) + offset
    safe = np.vstack([s] * n_pairs + [-s] * n_pairs) + offset
    crit = np.vstack([c] * n_pairs + [-c] * n_pairs) + offset
    pool = R.CenteringPool.from_conditions(
        base=base, near_safe=safe, near_critical=crit,
        declared_missing=("repeat", "far_critical"),
    )
    out = E.psi_distances_centered_cosine(base, crit, safe, centering_pool=pool)
    return (out["d_critical"], out["d_safe"]), math.cos(alpha_s) - math.cos(alpha_c)


def test_three_arm_geometry_recovers_known_psi_and_the_arm_ranking():
    """A synthetic three-arm geometry with an exactly known PSI per (arm, seed) cell.

    Angles are chosen so the three arms are separated by construction:

        nextlat  base->critical opens to ~1.10 rad   PSI ~ 0.50
        bst      ...to ~0.75 rad                     PSI ~ 0.23
        gpt      ...to ~0.40 rad                     PSI ~ 0.03

    Each seed perturbs the critical angle slightly, so the seed-to-seed spread is real and
    the contrast has something to be paired over, while every cell's PSI stays analytic.
    """
    rng = np.random.default_rng(20260823)
    dim, n_pairs = 24, 30
    alpha_s = 0.30
    alpha_c = {"nextlat": 1.10, "bst": 0.75, "gpt": 0.40}
    seeds = (1234, 1235, 1236)
    jitter = {1234: -0.02, 1235: 0.0, 1236: +0.02}

    distances, known = {}, {}
    for arm in E.ARMS:
        distances[arm], known[arm] = {}, {}
        for seed in seeds:
            cell, psi = _mirrored_cell(
                alpha_s, alpha_c[arm] + jitter[seed], dim=dim, n_pairs=n_pairs, rng=rng
            )
            distances[arm][seed] = cell
            known[arm][seed] = psi

    per_arm = E.psi_per_arm(distances, rng=np.random.default_rng(7), n_boot=2000)

    # (a) every cell's PSI is the analytic value, to machine precision.
    for arm in E.ARMS:
        for seed in seeds:
            assert per_arm[arm].psi_by_seed[seed] == pytest.approx(
                known[arm][seed], abs=1e-12
            )
            ci = per_arm[arm].per_seed[seed].ci
            assert ci.unit == "item (quartet)" and ci.n_units == 2 * n_pairs
        assert per_arm[arm].seed_mean == pytest.approx(
            float(np.mean([known[arm][s] for s in seeds])), abs=1e-12
        )

    # (b) the ranking survives, and it is a ranking with real gaps rather than ties.
    order = sorted(E.ARMS, key=lambda a: per_arm[a].seed_mean, reverse=True)
    assert order == ["nextlat", "bst", "gpt"]
    assert per_arm["nextlat"].seed_mean - per_arm["bst"].seed_mean > 0.2
    assert per_arm["bst"].seed_mean - per_arm["gpt"].seed_mean > 0.15
    assert per_arm["nextlat"].seed_mean == pytest.approx(0.502, abs=0.01)
    assert per_arm["bst"].seed_mean == pytest.approx(0.226, abs=0.01)
    assert per_arm["gpt"].seed_mean == pytest.approx(0.034, abs=0.01)

    # (c) the contrasts are the differences of those arm means, in the PREREGISTERED
    #     order — which is NOT the order of effect size.  nextlat-gpt is the largest gap
    #     here and it still comes second, because the primary contrast is the
    #     competence-matched one, fixed before any number existed.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # 3 seeds < MIN_SEEDS_FOR_INTERVAL
        report = E.three_arm_contrasts(per_arm, rng=np.random.default_rng(11), n_boot=2000)
    names = [c.spec.name for c in report.contrasts]
    assert names == ["nextlat_minus_bst", "nextlat_minus_gpt", "bst_minus_gpt"]
    assert [c.spec.priority for c in report.contrasts] == [1, 2, 3]
    assert report.primary.spec.role.startswith("primary")
    biggest = max(report.contrasts, key=lambda c: abs(c.estimate))
    assert biggest.spec.name == "nextlat_minus_gpt" and names[0] != biggest.spec.name

    for c in report.contrasts:
        want = per_arm[c.spec.label_a].seed_mean - per_arm[c.spec.label_b].seed_mean
        assert c.estimate == pytest.approx(want, abs=1e-12)
        assert c.contrast.ci.unit == "training seed"

    # (d) the confound is carried in the object, not left to the writeup to remember.
    assert "chance" in report.by_name("nextlat_minus_gpt").spec.reading
    assert "competence-matched" in report.by_name("nextlat_minus_bst").spec.role
    assert "competence alone" in report.by_name("bst_minus_gpt").spec.reading

    # (e) a dropped arm is an error, not a two-arm report that looks complete.
    with pytest.raises(ValueError, match="missing arm"):
        E.psi_per_arm(
            {k: v for k, v in distances.items() if k != "bst"},
            rng=np.random.default_rng(7),
            n_boot=50,
        )


def test_seed_level_contrast_does_not_collapse_to_an_item_level_one():
    """Items must not be smuggled in where seeds belong — in TYPE and in WIDTH.

    The geometry is built so the two answers genuinely disagree.  Pooled over items the
    NextLat-minus-BST difference is a clean positive effect with a tight interval that
    excludes zero.  Across the three trainings it is +0.30, -0.25, +0.02: the sign does
    not replicate, the seed-level interval straddles zero, and the effect is far below
    what three seeds could have detected.  An analysis that quoted the item-level number
    as the cross-model result would report a confident effect that the seeds do not show.
    """
    rng = np.random.default_rng(4321)
    seeds = (1234, 1235, 1236)
    seed_shift = {1234: +0.30, 1235: -0.25, 1236: +0.02}
    n_items = 4000

    nextlat_items, bst_items = {}, {}
    for s in seeds:
        base = 0.20 + 0.02 * rng.standard_normal(n_items)
        bst_items[s] = base
        nextlat_items[s] = base + seed_shift[s] + 0.02 * rng.standard_normal(n_items)

    # --- the item-level answer: pooled quartets, very tight, excludes zero -------------
    pooled = np.concatenate([nextlat_items[s] - bst_items[s] for s in seeds])
    item_ci = E.paired_bootstrap_mean(
        pooled, unit="item (quartet)", rng=np.random.default_rng(1), n_boot=4000
    )
    assert item_ci.n_units == 3 * n_items
    assert item_ci.ci_low > 0.0, "item-level pooling calls this a positive effect"

    # --- the seed-level answer: three trainings, wide, straddles zero -----------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        report = E.three_arm_contrasts(
            {
                "nextlat": {s: float(nextlat_items[s].mean()) for s in seeds},
                "bst": {s: float(bst_items[s].mean()) for s in seeds},
                "gpt": {s: 0.0 for s in seeds},
            },
            rng=np.random.default_rng(2),
            n_boot=4000,
        )
    primary = report.primary
    assert primary.contrast.ci.unit == "training seed"
    assert primary.contrast.ci.n_units == 3, "three trainings, not 12,000 quartets"
    assert primary.contrast.n_seeds == 3 and primary.contrast.underpowered

    seed_width = primary.contrast.ci.ci_high - primary.contrast.ci.ci_low
    item_width = item_ci.ci_high - item_ci.ci_low
    assert seed_width > 20 * item_width, (
        f"seed-level width {seed_width:.4f} vs item-level {item_width:.4f}: if these "
        "were comparable, the two units would be interchangeable and they are not"
    )
    assert primary.contrast.ci.ci_low < 0.0 < primary.contrast.ci.ci_high
    assert not primary.exceeds_mde, "and the design could not have resolved this effect"
    assert primary.contrast.sign_flip_p == 1.0 or primary.contrast.sign_flip_p >= 0.25

    # --- and the type wall: an item array is refused at every entry point --------------
    item_arrays = {arm: np.zeros(n_items) for arm in E.ARMS}
    with pytest.raises(TypeError, match="seeds are the inferential unit"):
        E.three_arm_contrasts(item_arrays, rng=np.random.default_rng(3))
    with pytest.raises(TypeError, match="seeds are the inferential unit"):
        E.psi_per_arm(
            {arm: (np.zeros(10), np.zeros(10)) for arm in E.ARMS},
            rng=np.random.default_rng(3),
            n_boot=50,
        )
    # Concatenating the items of all three seeds into one "seed" is the other way to
    # collapse the unit; the seed-set check catches it because the arms stop matching.
    with pytest.raises(ValueError, match="seed sets differ"):
        E.three_arm_contrasts(
            {
                "nextlat": {"all_seeds_pooled": 0.5},
                "bst": {s: 0.2 for s in seeds},
                "gpt": {s: 0.0 for s in seeds},
            },
            rng=np.random.default_rng(3),
        )


def test_minimum_detectable_effect_is_monotone_in_the_number_of_seeds():
    """The MDE must behave like a power curve, and say what three seeds cannot do."""
    sd = 0.037
    mdes = [E.minimum_detectable_effect(sd, n).mde for n in range(2, 31)]
    assert all(b < a for a, b in zip(mdes, mdes[1:])), (
        "strictly decreasing in n: more seeds can only ever detect a smaller effect"
    )
    assert mdes[0] > mdes[-1] * 5

    # Scale-free in the right way: the MDE is a multiple of the per-seed SD.
    for n in (3, 5, 8):
        one = E.minimum_detectable_effect(1.0, n).mde
        assert E.minimum_detectable_effect(sd, n).mde == pytest.approx(sd * one, rel=1e-9)
    # ...and larger for a stricter alpha or a higher power target, at fixed n.
    assert (
        E.minimum_detectable_effect(sd, 3, alpha=0.01).mde
        > E.minimum_detectable_effect(sd, 3, alpha=0.05).mde
    )
    assert (
        E.minimum_detectable_effect(sd, 3, power=0.95).mde
        > E.minimum_detectable_effect(sd, 3, power=0.80).mde
    )

    # The confirmatory design, stated plainly.  Three seeds resolve nothing under ~3.2
    # seed-level SDs, and the exact sign-flip test cannot reach 0.05 at ANY effect size
    # because its floor is 2^-2 = 0.25.  Five seeds do not fix that either (floor
    # 0.0625); six is the first n that can.
    three = E.minimum_detectable_effect(1.0, 3)
    assert three.mde == pytest.approx(3.26, abs=0.05)
    assert three.sign_flip_p_floor == 0.25
    assert three.randomization_test_can_reject is False
    assert E.minimum_detectable_effect(1.0, 5).randomization_test_can_reject is False
    assert E.minimum_detectable_effect(1.0, 6).randomization_test_can_reject is True
    # The floor reported here must be the SAME number the contrast itself reports, or a
    # reader could reconcile two different accounts of what three seeds can show.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        contrast = E.model_contrast_seed_level(
            {1: 1.0, 2: 2.0, 3: 3.0},
            {1: 0.0, 2: 0.0, 3: 0.0},
            label_a="a",
            label_b="b",
            rng=np.random.default_rng(0),
            n_boot=200,
        )
    assert contrast.min_attainable_p == three.sign_flip_p_floor == 0.25

    # The two-t approximation is reported alongside and agrees once df is not tiny.
    for n in (6, 10, 20):
        r = E.minimum_detectable_effect(1.0, n)
        assert r.mde_two_t_approximation == pytest.approx(r.mde, rel=0.02)

    for bad in ({"sd_per_seed": -1.0, "n_seeds": 3}, {"sd_per_seed": 1.0, "n_seeds": 1}):
        with pytest.raises(ValueError):
            E.minimum_detectable_effect(**bad)
    with pytest.raises(ValueError):
        E.minimum_detectable_effect(1.0, 3, power=1.0)
    with pytest.raises(ValueError):
        E.minimum_detectable_effect(1.0, 3, alpha=0.0)
