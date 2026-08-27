"""Amendment-exact paired-seed aggregation for the three-regime HMM calibration."""
from __future__ import annotations
import hashlib, itertools, json, math
from pathlib import Path
from typing import Mapping, Sequence
import numpy as np
from scipy import optimize, stats
from .family import REGIMES

SCHEMA = "nextlat_forgetting/hmm_cross_seed_aggregate/3"
SEEDS = (1234, 1235, 1236, 1237, 1238)
MODELS = ("gpt", "nextlat")
REQUIRED_METRICS = frozenset({
    "h1_predictive_equivalence_centered_cosine", "h1_predictive_equivalence_whitened",
    "h2_spearman", "h2_partial_spearman", "h2_neighborhood_retrieval",
    "h2_partial_spearman_whitened", "h2_belief_partial_spearman",
    "h2_belief_partial_spearman_whitened", "h3_posterior_decoding_len32",
    "h3_future_distribution_decoding_len32", "h3_posterior_decoding_len64",
    "h3_future_distribution_decoding_len64",
})
class HMMAggregationError(RuntimeError): pass

def _leaf(receipt: Mapping[str, object], metric: str, pool: str | None, leaf: str) -> float:
    metrics = receipt.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != REQUIRED_METRICS:
        raise HMMAggregationError(f"receipt metric set mismatch: missing={sorted(REQUIRED_METRICS-set(metrics or {}))}, extra={sorted(set(metrics or {})-REQUIRED_METRICS)}")
    value: object = metrics[metric]
    if pool:
        if not isinstance(value, dict) or set(value) != {"test32", "test64"}:
            raise HMMAggregationError(f"{metric}: both frozen pools are required")
        value = value[pool]
    if not isinstance(value, dict) or leaf not in value:
        raise HMMAggregationError(f"missing {metric}/{pool}/{leaf}")
    value = value[leaf]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise HMMAggregationError(f"nonfinite {metric}/{pool}/{leaf}")
    return float(value)

def load_complete_receipts(paths: Sequence[Path]) -> dict[tuple[str, str, int], dict]:
    expected = set(itertools.product(REGIMES, MODELS, SEEDS)); found = {}
    for path in paths:
        sidecar = path.with_name(path.name + ".sha256")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        fields = sidecar.read_text().strip().split() if sidecar.is_file() else []
        if not fields or fields[0].lower() != digest: raise HMMAggregationError(f"receipt sidecar missing/mismatched: {path}")
        receipt = json.loads(path.read_text()); key = (receipt.get("regime"), receipt.get("model"), receipt.get("seed"))
        if key not in expected or key in found: raise HMMAggregationError(f"unexpected/duplicate receipt {key}")
        if receipt.get("job_id") != f"{key[1]}-seed{key[2]}-hmm-{key[0]}" or receipt.get("all_preregistered_metrics_reported") is not True or receipt.get("metric_selection_performed") is not False:
            raise HMMAggregationError(f"{key}: identity/completeness attestation mismatch")
        for metric,pool,leaf in (
            ("h2_partial_spearman","test32","partial_spearman_given_edit_and_length"),
            ("h2_partial_spearman_whitened","test32","partial_spearman_given_edit_and_length"),
            ("h1_predictive_equivalence_centered_cosine","test32","paired_delta_mean"),
            ("h1_predictive_equivalence_whitened","test32","paired_delta_mean"),
            ("h3_future_distribution_decoding_len32",None,"js_bits"),("h3_future_distribution_decoding_len64",None,"js_bits"),
            ("h3_posterior_decoding_len32",None,"js_bits"),("h3_posterior_decoding_len64",None,"js_bits")):
            _leaf(receipt,metric,pool,leaf)
        found[key]=receipt
    missing=sorted(expected-set(found))
    if missing: raise HMMAggregationError(f"all 30 frozen receipts are required; missing={missing}")
    return found

def exact_sign_flip_two_sided(values: np.ndarray) -> float:
    """Exact two-sided randomization p over all paired-seed sign assignments."""
    x = np.asarray(values, dtype=float)
    observed = abs(float(np.mean(x)))
    null = [
        abs(float(np.mean(x * np.asarray(signs))))
        for signs in itertools.product((-1.0, 1.0), repeat=len(x))
    ]
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def _mde_80_two_sided(n: int) -> float:
    """Standardized MDE for a two-sided paired t test at alpha=.05, power=.80."""
    critical = float(stats.t.ppf(.975, n - 1))

    def attained(ncp: float) -> float:
        return float(
            stats.nct.sf(critical, n - 1, ncp)
            + stats.nct.cdf(-critical, n - 1, ncp)
        )

    high = 1.0
    while attained(high) < .80:
        high *= 2.0
    ncp = optimize.brentq(lambda value: attained(value) - .80, 0.0, high)
    return float(ncp / math.sqrt(n))


def _summary(values: Sequence[float]) -> dict[str, object]:
    x = np.asarray(values, dtype=float)
    sd = float(x.std(ddof=1))
    mean = float(x.mean())
    half = float(stats.t.ppf(.975, len(x) - 1) * sd / math.sqrt(len(x)))
    interval = [mean - half, mean + half]
    standardized = _mde_80_two_sided(len(x))
    raw_mde = standardized * sd
    if interval[0] > 0.0:
        interpretation = "resolved positive oriented effect"
    elif interval[1] < 0.0:
        interpretation = "resolved negative oriented effect"
    else:
        interpretation = "not resolved at the detectable effect size"
    return {
        "n_seed_pairs": len(x),
        "mean": mean,
        "paired_sd": sd,
        "paired_t_95_ci": interval,
        "exact_two_sided_sign_flip_p": exact_sign_flip_two_sided(x),
        "sign_flip_p_attainable_floor": 2.0 ** (1 - len(x)),
        "signs_positive": int((x > 0).sum()),
        "standardized_mde_80pct_power_two_sided_alpha_0.05": standardized,
        "raw_scale_mde_80pct_power_two_sided_alpha_0.05": raw_mde,
        "inference_interpretation": {
            "text": interpretation,
            "raw_scale_mde_80pct_power_two_sided_alpha_0.05": raw_mde,
            "equivalence_claim_permitted": False,
        },
        "leave_one_seed_out_means": [
            {"omitted_seed": seed, "mean": float(np.delete(x, i).mean())}
            for i, seed in enumerate(SEEDS)
        ],
    }
def _holm(pvalues:Mapping[str,float])->dict[str,float]:
    ordered=sorted(pvalues,key=lambda k:(pvalues[k],k)); result={}; running=0.; n=len(ordered)
    for i,key in enumerate(ordered): running=max(running,min(1.,(n-i)*pvalues[key])); result[key]=running
    return result
def _paired(receipts,metric,pool,leaf,*,transform=lambda x:x,direction=1):
    per_seed=[]; cells=[]
    for seed in SEEDS:
        vals=[]
        for regime in REGIMES:
            g=_leaf(receipts[(regime,"gpt",seed)],metric,pool,leaf); n=_leaf(receipts[(regime,"nextlat",seed)],metric,pool,leaf); v=direction*(transform(n)-transform(g)); vals.append(v); cells.append({"seed":seed,"regime":regime,"gpt":g,"nextlat":n,"oriented_contrast":v})
        per_seed.append(float(np.mean(vals)))
    return per_seed,cells


def _fisher_z_correlation(rho: float) -> float:
    """Apply the preregistered Fisher transform without an unregistered clip policy."""
    if not math.isfinite(rho) or not -1.0 < rho < 1.0:
        raise HMMAggregationError(
            f"Fisher-z requires a finite correlation in the open interval (-1, 1); got {rho}"
        )
    transformed = math.atanh(rho)
    if not math.isfinite(transformed):
        raise HMMAggregationError(f"Fisher-z produced a nonfinite value for rho={rho}")
    return transformed


def _regime_heterogeneity(cells: Sequence[Mapping[str, object]]) -> dict[str, object]:
    means = {
        regime: float(np.mean([
            float(cell["oriented_contrast"])
            for cell in cells if cell["regime"] == regime
        ]))
        for regime in REGIMES
    }
    reversal = any(value > 0.0 for value in means.values()) and any(
        value < 0.0 for value in means.values()
    )
    return {
        "per_regime_oriented_mean": means,
        "sign_reversal_across_regimes": reversal,
        "interpretation": (
            "regime heterogeneity: oriented effect reverses sign across frozen regimes"
            if reversal else "no regime-level sign reversal detected"
        ),
        "decision_rule_effect": "report_only_cannot_promote_or_alter_primary",
    }


def _endpoint_summary(values: Sequence[float], cells: Sequence[Mapping[str, object]]) -> dict:
    return {
        **_summary(values),
        "regime_heterogeneity": _regime_heterogeneity(cells),
        "per_regime_model_seed": list(cells),
    }

def aggregate(receipts: Mapping[tuple[str, str, int], Mapping[str, object]]) -> dict:
    primary = {}
    for name, metric in (
        ("centered_cosine", "h2_partial_spearman"),
        ("whitened_mahalanobis", "h2_partial_spearman_whitened"),
    ):
        values, cells = _paired(
            receipts, metric, "test32", "partial_spearman_given_edit_and_length",
            transform=_fisher_z_correlation,
        )
        primary[name] = _endpoint_summary(values, cells)
    endpoints = {
        "predictive_equivalence": (
            "h1_predictive_equivalence_centered_cosine", "test32", "paired_delta_mean", -1,
        ),
        "future_probe_js_len32": (
            "h3_future_distribution_decoding_len32", None, "js_bits", -1,
        ),
        "future_probe_js_len64": (
            "h3_future_distribution_decoding_len64", None, "js_bits", -1,
        ),
        "posterior_probe_js_len32": (
            "h3_posterior_decoding_len32", None, "js_bits", -1,
        ),
        "posterior_probe_js_len64": (
            "h3_posterior_decoding_len64", None, "js_bits", -1,
        ),
    }
    secondary = {}
    pvalues = {}
    for name, (metric, pool, leaf, direction) in endpoints.items():
        values, cells = _paired(receipts, metric, pool, leaf, direction=direction)
        summary = _endpoint_summary(values, cells)
        if name == "predictive_equivalence":
            wvalues, wcells = _paired(
                receipts, "h1_predictive_equivalence_whitened", pool, leaf, direction=-1,
            )
            summary["whitened_mahalanobis"] = _endpoint_summary(wvalues, wcells)
            pvalues[name] = max(
                float(summary["exact_two_sided_sign_flip_p"]),
                float(summary["whitened_mahalanobis"]["exact_two_sided_sign_flip_p"]),
            )
            summary["intersection_union_unadjusted_two_sided_p"] = pvalues[name]
        else:
            pvalues[name] = float(summary["exact_two_sided_sign_flip_p"])
        secondary[name] = summary
    for name, pvalue in _holm(pvalues).items():
        secondary[name]["holm_adjusted_p_across_five_endpoints"] = pvalue
    payload = {
        "schema": SCHEMA,
        "models": list(MODELS),
        "seeds": list(SEEDS),
        "required_regimes": list(REGIMES),
        "primary_estimand": (
            "within seed and metric, equal-regime mean of Fisher-z(partial rho NextLat)-"
            "Fisher-z(partial rho GPT), future-distribution JS, test32"
        ),
        "primary": primary,
        "intersection_union_primary_pass": all(
            primary[key]["paired_t_95_ci"][0] > 0 for key in primary
        ),
        "secondary_multiplicity": (
            "Holm across exactly five named regime-aggregated endpoints using their exact "
            "two-sided sign-flip p-values"
        ),
        "holm_family_unadjusted_two_sided_p": pvalues,
        "secondary": secondary,
        "small_n_limitation": {
            "n_seed_pairs": 5,
            "exact_two_sided_minimum_p": 2 / 32,
            "two_sided_p_below_0.05_attainable": False,
            "null_interpretation": "not resolved at the detectable effect size",
        },
        "all_regimes_models_seeds_reported": True,
    }
    payload["payload_sha256"] = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()
    ).hexdigest()
    return payload
