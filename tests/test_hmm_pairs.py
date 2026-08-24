"""Tests for the HMM pair bank: threshold freezing, threshold satisfaction, control matching,
and pool leakage.

Two levels are exercised. A small end-to-end bank is built inside the test session from a fresh
corpus, so the whole freeze-then-apply pipeline runs on every invocation; and the *shipped*
artifacts under `manifests/` are re-verified from scratch -- posteriors recomputed from the frozen
matrices, edit distances recomputed from the raw symbol strings, thresholds recomputed from the
persisted payload. Nothing in the second group trusts a number the pair bank wrote about itself.

The negative controls matter as much as the positive ones. `test_shuffled_posteriors_...` builds a
bank against permuted posteriors and shows the verification fails, which is the evidence that the
verification is not vacuous.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hmm_geometry.forward import HMM, forward_batch, js_divergence, sample_sequences  # noqa: E402
from hmm_geometry import pair_bank as pb  # noqa: E402

MANIFESTS = ROOT / "manifests"
MATRICES = MANIFESTS / "hmm_matrices.json"
THRESHOLDS = MANIFESTS / "hmm_thresholds.json"
PAIRS = MANIFESTS / "hmm_eval_pairs.jsonl"


# --------------------------------------------------------------------------------------------
# A small, self-contained corpus so the pipeline runs without the 100k-sequence artifacts.
# --------------------------------------------------------------------------------------------


def _test_hmm() -> HMM:
    """Same shape as the frozen HMM -- persistent states, overlapping emissions, asymmetric
    off-diagonal -- but written out here so these tests do not depend on the search having run."""
    transition = np.array(
        [
            [0.70, 0.21, 0.06, 0.03],
            [0.03, 0.70, 0.21, 0.06],
            [0.06, 0.03, 0.70, 0.21],
            [0.21, 0.06, 0.03, 0.70],
        ]
    )
    emission = np.full((4, 4), 0.10)
    np.fill_diagonal(emission, 0.70)
    hmm = HMM(transition, emission, np.full(4, 0.25))
    return HMM(transition, emission, hmm.stationary())


def _make_pool(hmm: HMM, name: str, split: str, offset: int, n: int, length: int, seed: int) -> pb.Pool:
    obs, _ = sample_sequences(hmm, n, length, np.random.default_rng(seed))
    res = forward_batch(hmm, obs.astype(np.int64))
    return pb.Pool(
        name=name,
        split=split,
        offset=offset,
        obs=obs,
        beliefs=res.beliefs,
        next_obs=res.next_obs,
        prefix_min=16,
        prefix_max=24,
    )


@pytest.fixture(scope="module")
def hmm() -> HMM:
    return _test_hmm()


@pytest.fixture(scope="module")
def pools(hmm: HMM) -> tuple[pb.Pool, pb.Pool]:
    calib = _make_pool(hmm, "calibration", "val", 0, 900, 32, seed=101)
    test = _make_pool(hmm, "test", "val", 900, 900, 32, seed=202)
    return calib, test


@pytest.fixture(scope="module")
def frozen(tmp_path_factory, hmm: HMM, pools) -> tuple[Path, pb.Thresholds]:
    calib, _ = pools
    path = tmp_path_factory.mktemp("thresholds") / "hmm_thresholds.json"
    th = pb.fit_thresholds([calib], hmm, seed=7, n_pairs=120_000)
    pb.freeze_thresholds(th, path)
    return path, th


@pytest.fixture(scope="module")
def bank(frozen, pools, hmm: HMM) -> pb.Bank:
    path, _ = frozen
    calib, test = pools
    loaded = pb.load_thresholds(path)
    return pb.build_bank(
        test,
        hmm,
        loaded,
        seed=13,
        target_pairs=400,
        n_search_pairs=400_000,
        n_lure_bases=30_000,
        forbidden_prefixes=calib.prefix_keys(),
    )


# --------------------------------------------------------------------------------------------
# Levenshtein
# --------------------------------------------------------------------------------------------


def _reference_levenshtein(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def test_levenshtein_batch_matches_a_reference_implementation() -> None:
    rng = np.random.default_rng(0)
    a = rng.integers(0, 4, size=(500, 16))
    b = rng.integers(0, 4, size=(500, 16))
    got = pb.levenshtein_batch(a, b)
    want = np.array([_reference_levenshtein(x, y) for x, y in zip(a, b)])
    np.testing.assert_array_equal(got, want)


def test_levenshtein_is_below_hamming_when_a_shift_aligns() -> None:
    """`0123` vs `0012` is Hamming 3 but Levenshtein 2. A Hamming shortcut would report the
    wrong edit distance for exactly the near-lure cases the bank cares about."""
    a = np.array([[0, 1, 2, 3]])
    b = np.array([[0, 0, 1, 2]])
    assert pb.levenshtein_batch(a, b)[0] == 2
    assert (a != b).sum() == 3


def test_levenshtein_batch_zero_and_identity() -> None:
    a = np.array([[1, 2, 3], [0, 0, 0]])
    np.testing.assert_array_equal(pb.levenshtein_batch(a, a), [0, 0])
    np.testing.assert_array_equal(
        pb.levenshtein_batch(np.array([[0, 0, 0]]), np.array([[1, 1, 1]])), [3]
    )


# --------------------------------------------------------------------------------------------
# Threshold freezing
# --------------------------------------------------------------------------------------------


def test_thresholds_are_a_deterministic_function_of_the_calibration_pool(pools, hmm) -> None:
    calib, _ = pools
    a = pb.fit_thresholds([calib], hmm, seed=7, n_pairs=120_000)
    b = pb.fit_thresholds([calib], hmm, seed=7, n_pairs=120_000)
    assert a.sha256() == b.sha256()
    c = pb.fit_thresholds([calib], hmm, seed=8, n_pairs=120_000)
    # A different fitting seed is a different sample of the same pool, so the payload differs --
    # which is precisely why the seed is recorded inside the frozen payload.
    assert c.fit_seed != a.fit_seed


def test_refreezing_the_same_thresholds_is_a_no_op(frozen, pools, hmm) -> None:
    path, th = frozen
    before = path.read_text()
    pb.freeze_thresholds(pb.fit_thresholds([pools[0]], hmm, seed=7, n_pairs=120_000), path)
    assert path.read_text() == before


def test_refreezing_different_thresholds_raises(frozen) -> None:
    from dataclasses import replace

    path, th = frozen
    tweaked = replace(th, js_low_bits=th.js_low_bits * 2.0)
    with pytest.raises(pb.ThresholdMismatch):
        pb.freeze_thresholds(tweaked, path)
    # And the file on disk is untouched.
    assert json.loads(path.read_text())["sha256"] == th.sha256()


def test_hand_edited_threshold_file_is_rejected(frozen, tmp_path) -> None:
    path, _ = frozen
    payload = json.loads(path.read_text())
    payload["thresholds"]["js_low_bits"] = 0.5  # the retune an analyst would be tempted by
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(payload))
    with pytest.raises(pb.ThresholdMismatch):
        pb.load_thresholds(edited)


def test_hand_edited_quantile_rule_is_rejected(frozen, tmp_path) -> None:
    path, _ = frozen
    payload = json.loads(path.read_text())
    payload["thresholds"]["quantile_rule"] = dict(payload["thresholds"]["quantile_rule"])
    payload["thresholds"]["quantile_rule"]["js_low_bits"] = "5th percentile"
    edited = tmp_path / "edited_rule.json"
    edited.write_text(json.dumps(payload))
    # Changing the rule text also changes the hash, so this is caught twice over.
    with pytest.raises(pb.ThresholdMismatch):
        pb.load_thresholds(edited)


def test_loaded_thresholds_are_verified_and_fitted_ones_are_not(frozen) -> None:
    path, th = frozen
    assert th.verified is False
    assert pb.load_thresholds(path).verified is True


def test_unverified_thresholds_cannot_build_a_test_bank(frozen, pools, hmm) -> None:
    _, th = frozen
    _, test = pools
    with pytest.raises(pb.UnverifiedThresholds):
        pb.build_bank(test, hmm, th, seed=1, target_pairs=10, n_search_pairs=20_000)


def test_bank_refuses_the_calibration_pool_it_was_fitted_on(frozen, pools, hmm) -> None:
    path, _ = frozen
    calib, _ = pools
    loaded = pb.load_thresholds(path)
    with pytest.raises(pb.ThresholdMismatch):
        pb.build_bank(calib, hmm, loaded, seed=1, target_pairs=10, n_search_pairs=20_000)


def test_bank_refuses_a_different_hmm(frozen, pools, hmm) -> None:
    path, _ = frozen
    _, test = pools
    loaded = pb.load_thresholds(path)
    other = HMM(np.full((4, 4), 0.25), hmm.emission, np.full(4, 0.25))
    with pytest.raises(pb.ThresholdMismatch):
        pb.build_bank(test, other, loaded, seed=1, target_pairs=10, n_search_pairs=20_000)


# --------------------------------------------------------------------------------------------
# The bank itself
# --------------------------------------------------------------------------------------------


def _recheck(pairs: list[dict], hmm: HMM) -> dict[str, np.ndarray]:
    """Recompute every pair quantity from the raw symbol strings and the frozen matrices.

    Nothing stored by the pair bank is trusted here: the posterior of each member is recomputed by
    running the forward algorithm on its prefix, the JS divergence is recomputed from those
    posteriors, and the edit distance is recomputed from the symbols.
    """
    out = {"js": [], "lev": [], "belief_delta": [], "prefix_len": []}
    by_len: dict[int, list[dict]] = {}
    for p in pairs:
        by_len.setdefault(len(p["a"]["prefix"]), []).append(p)
    for t, group in sorted(by_len.items()):
        a = np.array([p["a"]["prefix"] for p in group], dtype=np.int64)
        b = np.array([p["b"]["prefix"] for p in group], dtype=np.int64)
        assert a.shape[1] == b.shape[1] == t
        ba = forward_batch(hmm, a).beliefs[:, -1, :]
        bb = forward_batch(hmm, b).beliefs[:, -1, :]
        stored_a = np.array([p["a"]["belief"] for p in group])
        stored_b = np.array([p["b"]["belief"] for p in group])
        out["belief_delta"].append(
            np.concatenate([np.abs(ba - stored_a).max(1), np.abs(bb - stored_b).max(1)])
        )
        out["js"].append(js_divergence(ba, bb))
        out["lev"].append(pb.levenshtein_batch(a, b))
        out["prefix_len"].append(np.full(len(group), t))
    return {k: np.concatenate(v) for k, v in out.items()}


def test_bank_contains_all_three_pair_types(bank: pb.Bank) -> None:
    for kind in ("equivalent", "near_lure", "matched_control"):
        assert len(bank.by_type(kind)) > 0, f"no {kind} pairs were built"
    assert len(bank.by_type("equivalent")) == len(bank.by_type("matched_control"))


def test_equivalent_pairs_satisfy_the_frozen_thresholds(bank, frozen, hmm) -> None:
    path, _ = frozen
    th = pb.load_thresholds(path)
    pairs = bank.by_type("equivalent")
    r = _recheck(pairs, hmm)
    assert r["belief_delta"].max() < 1e-12, "stored posteriors disagree with a fresh forward pass"
    assert r["js"].max() <= th.js_low_bits + 1e-12
    cut = np.array([th.edit_cut(t) for t in r["prefix_len"]])
    assert (r["lev"] >= cut).all()
    # The thresholds must actually bind: an unfiltered sample of the same pool is far above them.
    assert r["js"].mean() < th.js_control_min_bits / 10


def test_near_lure_pairs_satisfy_the_frozen_thresholds(bank, frozen, hmm) -> None:
    path, _ = frozen
    th = pb.load_thresholds(path)
    pairs = bank.by_type("near_lure")
    r = _recheck(pairs, hmm)
    assert r["belief_delta"].max() < 1e-12
    assert r["js"].min() >= th.js_high_bits - 1e-12
    assert r["lev"].max() <= th.edit_low
    assert r["lev"].min() >= 1  # a lure that changed nothing is not a lure


def test_the_two_pair_types_are_opposites_on_both_axes(bank, hmm) -> None:
    """The whole design rests on the two banks being contrasts, not variations."""
    eq = _recheck(bank.by_type("equivalent"), hmm)
    lure = _recheck(bank.by_type("near_lure"), hmm)
    assert eq["js"].max() < lure["js"].min()
    assert eq["lev"].min() > lure["lev"].max()


def test_matched_control_matches_on_history_distance(bank, frozen, hmm) -> None:
    path, _ = frozen
    th = pb.load_thresholds(path)
    eq = bank.by_type("equivalent")
    ct = {p["matches_pair_id"]: p for p in bank.by_type("matched_control")}
    assert len(ct) == len(eq)

    eq_r = _recheck(eq, hmm)
    ct_pairs = [ct[p["pair_id"]] for p in eq]
    ct_r = _recheck(ct_pairs, hmm)

    # Exact match, pair for pair, on both history-side variables.
    np.testing.assert_array_equal(eq_r["lev"], ct_r["lev"])
    np.testing.assert_array_equal(eq_r["prefix_len"], ct_r["prefix_len"])
    # And a genuine separation on the predictive side, which is the contrast H1 needs.
    assert ct_r["js"].min() >= th.js_control_min_bits - 1e-12
    assert ct_r["js"].mean() > 10 * eq_r["js"].mean()


def test_controls_are_distinct_pairs_not_the_equivalent_pairs_relabelled(bank) -> None:
    eq = {(tuple(p["a"]["prefix"]), tuple(p["b"]["prefix"])) for p in bank.by_type("equivalent")}
    ct = {
        (tuple(p["a"]["prefix"]), tuple(p["b"]["prefix"]))
        for p in bank.by_type("matched_control")
    }
    assert not (eq & ct)


def test_no_calibration_sequence_leaks_into_the_test_bank(bank, pools) -> None:
    calib, test = pools
    calib_prefixes = calib.prefix_keys()
    calib_indices = set(range(calib.offset, calib.offset + calib.n_sequences))

    n_checked = 0
    for pair in bank.pairs:
        for member in ("a", "b"):
            item = pair[member]
            prefix = np.array(item["prefix"], dtype=np.int8)
            key = bytes([len(prefix)]) + prefix.tobytes()
            assert key not in calib_prefixes, f"{item['prefix']} occurs in the calibration pool"
            assert item["seq_index"] not in calib_indices
            n_checked += 1
    assert n_checked > 100

    # The leakage check is only meaningful if it could fire: a calibration prefix must be
    # detected by the same test.
    leaked = np.array(calib.obs[3, : calib.prefix_min], dtype=np.int8)
    assert bytes([calib.prefix_min]) + leaked.tobytes() in calib_prefixes


def test_bank_is_deterministic_in_its_seed(frozen, pools, hmm) -> None:
    path, _ = frozen
    calib, test = pools
    th = pb.load_thresholds(path)
    kw = dict(
        target_pairs=200,
        n_search_pairs=200_000,
        n_lure_bases=20_000,
        forbidden_prefixes=calib.prefix_keys(),
    )
    a = pb.build_bank(test, hmm, th, seed=13, **kw)
    b = pb.build_bank(test, hmm, th, seed=13, **kw)
    c = pb.build_bank(test, hmm, th, seed=14, **kw)
    ids_a = [p["pair_id"] for p in a.pairs]
    assert ids_a == [p["pair_id"] for p in b.pairs]
    assert ids_a != [p["pair_id"] for p in c.pairs]


def test_shuffled_posteriors_would_fail_the_threshold_check(frozen, pools, hmm) -> None:
    """Negative control for the verification, not for the bank.

    A pool whose posteriors have been permuted across sequences still produces a bank -- the
    search finds pairs whose *permuted* posteriors agree. Rechecking those pairs against the true
    posteriors, which is what `_recheck` does, must reject them. If it did not, every threshold
    test above would pass on shuffled data and would be worthless.
    """
    path, _ = frozen
    calib, test = pools
    th = pb.load_thresholds(path)

    rng = np.random.default_rng(0)
    perm = rng.permutation(test.n_sequences)
    shuffled = pb.Pool(
        name="shuffled",
        split="val",
        offset=test.offset,
        obs=test.obs,
        beliefs=test.beliefs[perm],
        next_obs=test.next_obs[perm],
        prefix_min=test.prefix_min,
        prefix_max=test.prefix_max,
    )
    bad = pb.build_bank(
        shuffled, hmm, th, seed=13, target_pairs=200, n_search_pairs=300_000, n_lure_bases=1
    )
    eq = bad.by_type("equivalent")
    assert len(eq) > 20, "the shuffled control produced too few pairs to be informative"
    r = _recheck(eq, hmm)
    assert r["js"].max() > th.js_low_bits, "recheck accepted pairs built on permuted posteriors"
    assert (r["js"] > th.js_low_bits).mean() > 0.5


# --------------------------------------------------------------------------------------------
# The shipped artifacts
# --------------------------------------------------------------------------------------------

_artifacts = pytest.mark.skipif(
    not (MATRICES.exists() and THRESHOLDS.exists() and PAIRS.exists()),
    reason="frozen HMM artifacts not built; run generate.py all and pair_bank.py all",
)


@pytest.fixture(scope="module")
def shipped():
    from hmm_geometry.generate import load_frozen_hmm

    hmm, _ = load_frozen_hmm()
    th = pb.load_thresholds(THRESHOLDS)
    pairs = pb.load_bank(PAIRS)
    return hmm, th, pairs


@_artifacts
def test_shipped_thresholds_verify_and_match_the_frozen_hmm(shipped) -> None:
    hmm, th, _ = shipped
    assert th.verified
    assert th.hmm_sha256 == hmm.sha256()
    assert 0.0 < th.js_low_bits < th.js_high_bits <= 1.0
    assert 0.0 < th.js_low_bits < th.js_control_min_bits <= 1.0
    # A near-lure's two-symbol cut must sit far below the high-edit cut at every calibrated
    # prefix length, or the two banks would not be contrasts.
    lengths = sorted(int(k) for k in th.edit_high_by_length)
    assert lengths == list(range(16, 65))
    for t in lengths:
        assert th.edit_low < th.edit_cut(t) <= t


@_artifacts
def test_shipped_test_pool_pairs_satisfy_the_frozen_thresholds(shipped) -> None:
    hmm, th, pairs = shipped
    for pool in {p["pool"] for p in pairs}:
        eq = [p for p in pairs if p["pair_type"] == "equivalent" and p["pool"] == pool]
        lure = [p for p in pairs if p["pair_type"] == "near_lure" and p["pool"] == pool]
        assert eq and lure
        r_eq = _recheck(eq, hmm)
        r_lu = _recheck(lure, hmm)
        assert r_eq["belief_delta"].max() < 1e-12
        assert r_lu["belief_delta"].max() < 1e-12
        assert r_eq["js"].max() <= th.js_low_bits + 1e-12
        cut = np.array([th.edit_cut(t) for t in r_eq["prefix_len"]])
        assert (r_eq["lev"] >= cut).all()
        assert r_lu["js"].min() >= th.js_high_bits - 1e-12
        assert 1 <= r_lu["lev"].max() <= th.edit_low


@_artifacts
def test_shipped_controls_are_matched(shipped) -> None:
    hmm, th, pairs = shipped
    ct = {p["matches_pair_id"]: p for p in pairs if p["pair_type"] == "matched_control"}
    eq = [p for p in pairs if p["pair_type"] == "equivalent"]
    assert len(ct) == len(eq)
    r_eq = _recheck(eq, hmm)
    r_ct = _recheck([ct[p["pair_id"]] for p in eq], hmm)
    np.testing.assert_array_equal(r_eq["lev"], r_ct["lev"])
    np.testing.assert_array_equal(r_eq["prefix_len"], r_ct["prefix_len"])
    assert r_ct["js"].min() >= th.js_control_min_bits - 1e-12


@_artifacts
def test_shipped_bank_does_not_leak_the_calibration_pool(shipped) -> None:
    from hmm_geometry.pair_bank import load_pools

    _, _, pairs = shipped
    pools = load_pools()
    calibration = [pools["calibration32"], pools["calibration64"]]
    calib_prefixes = set().union(*(p.prefix_keys() for p in calibration))
    calib_indices = {
        (p.split, i)
        for p in calibration
        for i in range(p.offset, p.offset + p.n_sequences)
    }
    n_checked = 0
    for pair in pairs:
        for member in ("a", "b"):
            item = pair[member]
            prefix = np.array(item["prefix"], dtype=np.int8)
            split = item["source"].replace("lure_of:", "")
            assert (split, item["seq_index"]) not in calib_indices
            key = bytes([len(prefix)]) + prefix.tobytes()
            assert key not in calib_prefixes
            n_checked += 1
    assert n_checked > 1000


@_artifacts
def test_shipped_bank_file_matches_its_manifest_hash() -> None:
    import hashlib

    manifest = json.loads((MANIFESTS / "hmm_eval_pairs.json").read_text())
    digest = hashlib.sha256(PAIRS.read_bytes()).hexdigest()
    assert digest == manifest["pairs_sha256"]
    assert manifest["n_pairs"] == len(pb.load_bank(PAIRS))
