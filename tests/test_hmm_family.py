from __future__ import annotations

import copy
import json
import math
import pathlib

import numpy as np
import pytest

from hmm_geometry.aggregate import (
    HMMAggregationError, SEEDS, aggregate, exact_sign_flip_two_sided,
    load_complete_receipts,
)
from hmm_geometry.family import (
    REGIMES, HMMFamilyError, _verify_linear_certificate, family_payload,
    linear_certificates_match, load_family, select_grid_family,
)
from hmm_geometry.generate import Candidate, Diagnostics, load_frozen_hmm
from hmm_geometry.forward import forward_batch, sample_sequences
from hmm_geometry.pair_bank import FUTURE_QUANTILE_RULE, Pool, Thresholds, sample_same_length_pairs
from lurestar.durable_checkpoint import sha256_file
from aggregate_hmm_family import main as aggregate_main
from run_hmm_matrix import HMMFabricLauncher, build_hmm_matrix


def test_family_is_model_blind_complete_and_certifies_primary_sigma_min(tmp_path: pathlib.Path) -> None:
    primary, _ = load_frozen_hmm()
    candidates = [
        (Candidate(.40, 0., (.5,.3,.2), .70, .10), Diagnostics(
            {"mean_dwell_time": 1.7, "belief_entropy_mean_bits": .9}, passed=True)),
        (Candidate(.60, .04, (.6,.25,.15), .55, .20), Diagnostics(
            {"mean_dwell_time": 2.8, "belief_entropy_mean_bits": 1.3}, passed=True)),
        (Candidate(.70, .04, (.7,.2,.1), .70, .10), Diagnostics(
            {"mean_dwell_time": 3.4, "belief_entropy_mean_bits": 1.1}, passed=True)),
    ]
    selected = select_grid_family(candidates)
    assert selected["fast_mixing_moderate"][1].values["mean_dwell_time"] == 1.7
    assert selected["persistent_high_aliasing"][1].values["belief_entropy_mean_bits"] == 1.3
    payload = family_payload(
        primary, primary_manifest_sha256="a" * 64,
        selected=selected, passing_order=sorted(candidates, key=lambda item: item[0].key()),
    )
    assert tuple(payload["required_regimes"]) == REGIMES
    assert payload["selection_blinding"] == {
        "model_checkpoints_inspected": False,
        "model_representations_inspected": False,
        "model_outcomes_inspected": False,
        "rule": "matrices selected from the unchanged deterministic grid by frozen dwell/entropy rankings and TE gate",
    }
    cert = payload["regimes"]["persistent_moderate"]["linear_certificate"]
    te = cert["matrices"]["transition_times_emission"]
    assert te["rank"] == 4
    assert cert["predictive_injectivity_certified"] is True
    assert te["sigma_min"] == pytest.approx(0.3067206885)
    for regime in REGIMES:
        assert payload["regimes"][regime]["linear_certificate"]["predictive_injectivity_certified"]

    path = tmp_path / "family.json"
    path.write_text(json.dumps(payload))
    hmms, loaded = load_family(path)
    assert tuple(hmms) == REGIMES
    assert loaded == payload
    document = json.loads(path.read_text())
    document["regimes"]["fast_mixing_moderate"]["hmm"]["transition"][0][0] = .84
    path.write_text(json.dumps(document))
    with pytest.raises(HMMFamilyError, match="schema/hash"):
        load_family(path)


def _metric_payload(offset: float) -> dict:
    pools = {pool: {"paired_delta_mean": offset} for pool in ("test32", "test64")}
    h2_pools = {
        pool: {"spearman_distance_vs_js": -offset,
               "partial_spearman_given_edit_and_length": -offset,
               "lift_over_chance": -offset}
        for pool in ("test32", "test64")
    }
    return {
        "h1_predictive_equivalence_centered_cosine": pools,
        "h1_predictive_equivalence_whitened": pools,
        "h2_spearman": h2_pools,
        "h2_partial_spearman": h2_pools,
        "h2_neighborhood_retrieval": h2_pools,
        "h2_partial_spearman_whitened": h2_pools,
        "h2_belief_partial_spearman": h2_pools,
        "h2_belief_partial_spearman_whitened": h2_pools,
        "h3_posterior_decoding_len32": {"js_bits": offset},
        "h3_future_distribution_decoding_len32": {"js_bits": offset},
        "h3_posterior_decoding_len64": {"js_bits": offset},
        "h3_future_distribution_decoding_len64": {"js_bits": offset},
    }


def _write_receipts(tmp_path: pathlib.Path) -> list[pathlib.Path]:
    paths = []
    for regime in REGIMES:
        for model in ("gpt", "nextlat"):
            for seed in SEEDS:
                # NextLat improves lower-is-better H1/H3; its H2 synthetic values are lower here,
                # which is useful because aggregation must report all outcomes, not only wins.
                offset = seed / 10_000 + (0.01 if model == "nextlat" else 0.02)
                payload = {
                    "schema": "nextlat_forgetting/hmm_geometry/1",
                    "job_id": f"{model}-seed{seed}-hmm-{regime}",
                    "regime": regime, "model": model, "seed": seed,
                    "all_preregistered_metrics_reported": True,
                    "metric_selection_performed": False,
                    "metrics": _metric_payload(offset),
                }
                path = tmp_path / f"{regime}-{model}-{seed}.json"
                path.write_text(json.dumps(payload))
                path.with_name(path.name + ".sha256").write_text(f"{sha256_file(path)}  {path.name}\n")
                paths.append(path)
    return paths


def test_aggregate_requires_all_regimes_models_seeds_and_metrics(tmp_path: pathlib.Path) -> None:
    paths = _write_receipts(tmp_path)
    receipts = load_complete_receipts(paths)
    result = aggregate(receipts)
    assert "equal-regime mean" in result["primary_estimand"]
    assert result["small_n_limitation"]["exact_two_sided_minimum_p"] == 2 / 32
    primary = result["primary"]["centered_cosine"]
    assert primary["n_seed_pairs"] == 5
    assert len(primary["per_regime_model_seed"]) == 15
    assert primary["exact_two_sided_sign_flip_p"] == 2 / 32
    assert result["intersection_union_primary_pass"] is True
    assert set(result["secondary"]) == {
        "predictive_equivalence", "future_probe_js_len32", "future_probe_js_len64",
        "posterior_probe_js_len32", "posterior_probe_js_len64",
    }
    assert all("holm_adjusted_p_across_five_endpoints" in value for value in result["secondary"].values())
    assert set(result["holm_family_unadjusted_two_sided_p"]) == set(result["secondary"])
    assert len(result["holm_family_unadjusted_two_sided_p"]) == 5
    assert result["secondary"]["predictive_equivalence"][
        "intersection_union_unadjusted_two_sided_p"
    ] == result["holm_family_unadjusted_two_sided_p"]["predictive_equivalence"]
    assert len(primary["leave_one_seed_out_means"]) == 5
    assert primary["standardized_mde_80pct_power_two_sided_alpha_0.05"] > 0
    assert "exact_directional_sign_flip_p" not in primary
    assert "one_sided" not in json.dumps(result)

    with pytest.raises(HMMAggregationError, match="all 30"):
        load_complete_receipts(paths[:-1])

    edited = json.loads(paths[0].read_text())
    edited["metrics"].pop("h2_partial_spearman")
    paths[0].write_text(json.dumps(edited))
    paths[0].with_name(paths[0].name + ".sha256").write_text(
        f"{sha256_file(paths[0])}  {paths[0].name}\n"
    )
    with pytest.raises(HMMAggregationError, match="metric set"):
        load_complete_receipts(paths)


def test_aggregate_cli_refuses_operational_recovery_subset_before_output(
        tmp_path: pathlib.Path, capsys) -> None:
    paths = _write_receipts(tmp_path)
    output = tmp_path / "aggregate.json"
    args = [item for path in paths[:-1] for item in ("--receipt", str(path))]

    assert aggregate_main([*args, "--output", str(output)]) == 2
    assert "exactly 30 frozen receipt arguments" in capsys.readouterr().err
    assert not output.exists()


def test_exact_sign_flip_discreteness_is_explicit() -> None:
    assert exact_sign_flip_two_sided(np.ones(5)) == 2 / 32
    assert exact_sign_flip_two_sided(-np.ones(5)) == 2 / 32


@pytest.mark.parametrize(
    "invalid_rho,error",
    [
        (-1.0, "Fisher-z.*open interval"),
        (1.0, "Fisher-z.*open interval"),
        (-1.01, "Fisher-z.*open interval"),
        (1.01, "Fisher-z.*open interval"),
        (float("nan"), "nonfinite h2_partial_spearman"),
        (float("inf"), "nonfinite h2_partial_spearman"),
    ],
)
def test_fisher_z_uses_exact_atanh_and_refuses_boundary_correlations(
        tmp_path: pathlib.Path, invalid_rho: float, error: str) -> None:
    receipts = load_complete_receipts(_write_receipts(tmp_path))
    valid_rho = np.nextafter(1.0, 0.0)
    for metric in ("h2_partial_spearman", "h2_partial_spearman_whitened"):
        for regime in REGIMES:
            for seed in SEEDS:
                receipts[(regime, "gpt", seed)]["metrics"][metric]["test32"][
                    "partial_spearman_given_edit_and_length"
                ] = 0.25
                receipts[(regime, "nextlat", seed)]["metrics"][metric]["test32"][
                    "partial_spearman_given_edit_and_length"
                ] = valid_rho

    result = aggregate(receipts)
    expected = math.atanh(valid_rho) - math.atanh(0.25)
    assert result["primary"]["centered_cosine"]["mean"] == pytest.approx(expected)
    assert result["primary"]["centered_cosine"]["per_regime_model_seed"][0][
        "oriented_contrast"
    ] == pytest.approx(expected)

    receipts[(REGIMES[-1], "nextlat", SEEDS[-1])]["metrics"][
        "h2_partial_spearman"
    ]["test32"]["partial_spearman_given_edit_and_length"] = invalid_rho
    with pytest.raises(HMMAggregationError, match=error):
        aggregate(receipts)


def test_null_language_mde_and_regime_sign_reversal_are_mandatory_report_only(
        tmp_path: pathlib.Path) -> None:
    receipts = load_complete_receipts(_write_receipts(tmp_path))
    regime_effect = {
        "persistent_moderate": .30,
        "fast_mixing_moderate": -.35,
        "persistent_high_aliasing": .05,
    }
    seed_noise = dict(zip(SEEDS, (-.04, -.02, 0.0, .02, .04)))
    for regime in REGIMES:
        for seed in SEEDS:
            gpt = receipts[(regime, "gpt", seed)]["metrics"]["h2_partial_spearman"]
            nextlat = receipts[(regime, "nextlat", seed)]["metrics"]["h2_partial_spearman"]
            gpt["test32"]["partial_spearman_given_edit_and_length"] = 0.0
            nextlat["test32"]["partial_spearman_given_edit_and_length"] = float(
                np.tanh(regime_effect[regime] + seed_noise[seed])
            )

    result = aggregate(receipts)
    centered = result["primary"]["centered_cosine"]
    assert centered["paired_t_95_ci"][0] <= 0 <= centered["paired_t_95_ci"][1]
    assert centered["inference_interpretation"]["text"] == (
        "not resolved at the detectable effect size"
    )
    assert centered["inference_interpretation"][
        "raw_scale_mde_80pct_power_two_sided_alpha_0.05"
    ] == centered["raw_scale_mde_80pct_power_two_sided_alpha_0.05"]
    assert centered["inference_interpretation"]["equivalence_claim_permitted"] is False
    heterogeneity = centered["regime_heterogeneity"]
    assert heterogeneity["sign_reversal_across_regimes"] is True
    assert heterogeneity["decision_rule_effect"] == (
        "report_only_cannot_promote_or_alter_primary"
    )
    assert result["intersection_union_primary_pass"] is False


def test_family_pair_candidates_are_selected_on_exact_future_js() -> None:
    hmm, _ = load_frozen_hmm()
    obs, _ = sample_sequences(hmm, 20, 16, np.random.default_rng(7))
    exact = forward_batch(hmm, obs.astype(np.int64))
    pool = Pool("tiny", "val", 0, obs, exact.beliefs, exact.next_obs, 16, 16)
    candidates = sample_same_length_pairs(
        pool, 100, np.random.default_rng(8), distance_target="future_js"
    )[16]
    np.testing.assert_array_equal(candidates["jsd"], candidates["future_jsd"])
    assert np.max(np.abs(candidates["future_jsd"] - candidates["belief_jsd"])) > 1e-6
    thresholds = Thresholds(
        .01, .2, .1, {"16": 10}, 2, 8, 2, 100, 100,
        [{"name": "tiny", "sha256": pool.sha256(), "prefix_min": 16, "prefix_max": 16}],
        hmm.sha256(), 9, distance_target="future_js",
    )
    assert thresholds.payload()["distance_target"] == "future_js"
    assert thresholds.payload()["quantile_rule"] == FUTURE_QUANTILE_RULE


def test_shipped_family_is_amendment_exact() -> None:
    path = pathlib.Path(__file__).resolve().parent.parent / "manifests/hmm_family.json"
    hmms, payload = load_family(path)
    assert tuple(hmms) == REGIMES
    assert len(payload["passing_candidates_lexicographic"]) == 83
    assert [payload["regimes"][name]["data_seed"] for name in REGIMES] == [
        1_105_963, 1_205_963, 1_305_963,
    ]
    for name in REGIMES:
        te = payload["regimes"][name]["linear_certificate"]["matrices"][
            "transition_times_emission"
        ]
        assert te["rank"] == 4
        assert te["sigma_min"] > .05


def _shipped_certificate() -> dict:
    path = pathlib.Path(__file__).resolve().parent.parent / "manifests/hmm_family.json"
    payload = json.loads(path.read_text())
    return payload["regimes"]["persistent_moderate"]["linear_certificate"]


def test_certificate_accepts_only_lapack_scale_finite_float_drift() -> None:
    stored = _shipped_certificate()
    recomputed = copy.deepcopy(stored)
    transition = recomputed["matrices"]["transition"]
    transition["singular_values_descending"][0] += 1e-16
    transition["sigma_min"] += 1e-16
    transition["condition_number"] += 1e-15
    assert linear_certificates_match(stored, recomputed)

    too_large = copy.deepcopy(stored)
    too_large["matrices"]["transition_times_emission"]["sigma_min"] += 1e-8
    assert not linear_certificates_match(stored, too_large)


@pytest.mark.parametrize("corruption", ("rank", "boolean", "shape", "missing_key", "extra_key"))
def test_certificate_rejects_structural_corruption(corruption: str) -> None:
    stored = _shipped_certificate()
    recomputed = copy.deepcopy(stored)
    te = recomputed["matrices"]["transition_times_emission"]
    if corruption == "rank":
        te["rank"] = 3
    elif corruption == "boolean":
        te["full_rank"] = False
    elif corruption == "shape":
        te["shape"] = [4, 3]
    elif corruption == "missing_key":
        del te["sigma_max"]
    else:
        te["unexpected"] = "not frozen"
    assert not linear_certificates_match(stored, recomputed)


def test_recomputed_te_gate_is_enforced_independently_of_certificate_match() -> None:
    recomputed = _shipped_certificate()
    recomputed["matrices"]["transition_times_emission"]["sigma_min"] = .05
    stored_at_boundary = copy.deepcopy(recomputed)
    with pytest.raises(HMMFamilyError, match="recomputed TE rank/sigma_min gate failed"):
        _verify_linear_certificate(stored_at_boundary, recomputed, regime="synthetic")


def test_family_orchestration_is_exactly_30_isolated_jobs(
    tmp_path: pathlib.Path, monkeypatch
) -> None:
    snapshot = tmp_path / "snapshot"
    identity = tmp_path / "family-identity"
    identity.write_text("frozen")
    import run_hmm_matrix as matrix_module
    monkeypatch.setattr(
        matrix_module, "verify_hmm_family_snapshot",
        lambda _: {regime: (str(identity),) for regime in REGIMES},
    )

    repo = pathlib.Path(__file__).resolve().parent.parent
    jobs = build_hmm_matrix(
        tmp_path / "durable", project_root=repo, upstream_root=repo / "upstream/NextLat",
        snapshot_root=snapshot, regimes=REGIMES,
    )
    assert len(jobs) == 30
    assert {job.condition for job in jobs} == set(REGIMES)
    assert len({pathlib.Path(job.out_root) for job in jobs}) == 30
    assert jobs[0].job_id == "gpt-seed1234-hmm-persistent_moderate"
    persistent = next(job for job in jobs if job.condition == "persistent_moderate")
    command = HMMFabricLauncher(repo, repo / "upstream/NextLat", data_root=snapshot).command(
        __import__("run_matrix").ResumePlan(persistent, fresh=True)
    )
    assert any("data/hmm_family/persistent_moderate/hmm4x4_train" in arg for arg in command)
    assert any("manifests/hmm_family/persistent_moderate/hmm_matrices.json" in arg for arg in command)

    with pytest.raises(ValueError, match="every frozen regime"):
        build_hmm_matrix(
            tmp_path / "bad", project_root=repo, upstream_root=repo / "upstream/NextLat",
            snapshot_root=snapshot, regimes=("persistent_moderate", "fast_mixing_moderate"),
        )
