"""Frozen, model-blind HMM robustness family and algebraic certificates.

The family is specified entirely from generative-process considerations.  No model checkpoint,
hidden state, evaluation receipt, or learned metric is an input to this module.  This is the
mechanical boundary that prevents choosing a convenient HMM after seeing model outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .forward import HMM

SCHEMA = "nextlat_forgetting/hmm_family/1"
CERTIFICATE_SCHEMA = "nextlat_forgetting/hmm_linear_certificate/1"
FAMILY_SEED_BASE = 1_105_963
FAMILY_SEED_STRIDE = 100_000
REGIMES = ("persistent_moderate", "fast_mixing_moderate", "persistent_high_aliasing")
PRIMARY_REGIME = "persistent_moderate"


class HMMFamilyError(RuntimeError):
    """The frozen family or one of its identities is incomplete or inconsistent."""


_CERTIFICATE_KEYS = {
    "schema", "hmm_sha256", "rank_tolerance", "belief_to_one_step_future_map",
    "predictive_injectivity_certified", "matrices",
}
_MATRIX_CERTIFICATE_KEYS = {
    "shape", "rank", "full_rank", "singular_values_descending",
    "sigma_min", "sigma_max", "condition_number",
}
_SVD_SCALAR_KEYS = ("sigma_min", "sigma_max", "condition_number")
_CERTIFICATE_RTOL = 1e-12
_CERTIFICATE_ATOL = 1e-15


def _same_exact_scalar(left: object, right: object) -> bool:
    """Compare a structural scalar without Python's bool/int coercion."""
    return type(left) is type(right) and left == right


def _same_integer_list(left: object, right: object) -> bool:
    return (
        isinstance(left, list)
        and isinstance(right, list)
        and len(left) == len(right)
        and all(
            isinstance(a, int) and not isinstance(a, bool)
            and isinstance(b, int) and not isinstance(b, bool)
            and a == b
            for a, b in zip(left, right)
        )
    )


def _finite_svd_close(left: object, right: object) -> bool:
    """Allow only finite LAPACK-scale drift in an SVD-derived number."""
    if (
        not isinstance(left, (int, float)) or isinstance(left, bool)
        or not isinstance(right, (int, float)) or isinstance(right, bool)
    ):
        return False
    a, b = float(left), float(right)
    return math.isfinite(a) and math.isfinite(b) and math.isclose(
        a, b, rel_tol=_CERTIFICATE_RTOL, abs_tol=_CERTIFICATE_ATOL
    )


def linear_certificates_match(stored: object, recomputed: object) -> bool:
    """Compare a frozen certificate while tolerating only SVD backend roundoff.

    HMM identity and every structural assertion remain exact. NumPy/LAPACK versions may differ
    in the last few bits of singular values, so only the finite, SVD-derived numeric fields use a
    tight tolerance. Missing/extra keys and nonfinite values are deliberately rejected.
    """
    if not isinstance(stored, Mapping) or not isinstance(recomputed, Mapping):
        return False
    if set(stored) != _CERTIFICATE_KEYS or set(recomputed) != _CERTIFICATE_KEYS:
        return False
    for key in _CERTIFICATE_KEYS - {"matrices"}:
        if not _same_exact_scalar(stored[key], recomputed[key]):
            return False
    stored_matrices, recomputed_matrices = stored["matrices"], recomputed["matrices"]
    if not isinstance(stored_matrices, Mapping) or not isinstance(recomputed_matrices, Mapping):
        return False
    if set(stored_matrices) != set(recomputed_matrices):
        return False
    for name in stored_matrices:
        left, right = stored_matrices[name], recomputed_matrices[name]
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            return False
        if set(left) != _MATRIX_CERTIFICATE_KEYS or set(right) != _MATRIX_CERTIFICATE_KEYS:
            return False
        if not _same_integer_list(left["shape"], right["shape"]):
            return False
        for key in ("rank", "full_rank"):
            if not _same_exact_scalar(left[key], right[key]):
                return False
        left_singular, right_singular = (
            left["singular_values_descending"], right["singular_values_descending"]
        )
        if (
            not isinstance(left_singular, list) or not isinstance(right_singular, list)
            or len(left_singular) != len(right_singular)
            or not all(_finite_svd_close(a, b) for a, b in zip(left_singular, right_singular))
        ):
            return False
        if not all(_finite_svd_close(left[key], right[key]) for key in _SVD_SCALAR_KEYS):
            return False
    return True


def _verify_linear_certificate(
    stored: object, recomputed: Mapping[str, object], *, regime: str
) -> None:
    if not linear_certificates_match(stored, recomputed):
        raise HMMFamilyError(f"{regime}: linear certificate mismatch")
    te = recomputed["matrices"]["transition_times_emission"]
    sigma_min = te["sigma_min"]
    if (
        te["rank"] != 4 or te["full_rank"] is not True
        or not isinstance(sigma_min, (int, float)) or isinstance(sigma_min, bool)
        or not math.isfinite(float(sigma_min)) or float(sigma_min) <= .05
    ):
        raise HMMFamilyError(f"{regime}: recomputed TE rank/sigma_min gate failed")


def canonical_sha256(value: object) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def matrix_certificate(hmm: HMM, *, rank_tolerance: float = 1e-12) -> dict[str, object]:
    """Certify the linear maps relevant to predictive sufficiency.

    Beliefs map to next-observation distributions as ``b @ T @ E``.  Thus full column rank of
    ``T @ E`` is the exact finite-dimensional condition under which distinct beliefs cannot be
    predictively aliased at one step.  Singular values are recorded rather than merely a boolean
    so the conditioning of that implication is visible.
    """
    matrices = {
        # The HMM identity itself is defined at 12 decimals. Certify that same canonical object,
        # avoiding last-bit changes caused by constructor renormalization after JSON reload.
        "transition": np.round(hmm.transition, 12),
        "emission": np.round(hmm.emission, 12),
    }
    matrices["transition_times_emission"] = matrices["transition"] @ matrices["emission"]
    entries: dict[str, object] = {}
    for name, matrix in matrices.items():
        singular = np.linalg.svd(matrix, compute_uv=False)
        rank = int(np.linalg.matrix_rank(matrix, tol=rank_tolerance))
        entries[name] = {
            "shape": list(matrix.shape),
            "rank": rank,
            "full_rank": rank == min(matrix.shape),
            "singular_values_descending": [float(x) for x in singular],
            "sigma_min": float(singular[-1]),
            "sigma_max": float(singular[0]),
            "condition_number": float(singular[0] / singular[-1]) if singular[-1] else None,
        }
    return {
        "schema": CERTIFICATE_SCHEMA,
        "hmm_sha256": hmm.sha256(),
        "rank_tolerance": rank_tolerance,
        "belief_to_one_step_future_map": "row belief b maps to b @ transition @ emission",
        "predictive_injectivity_certified": bool(
            entries["transition_times_emission"]["full_rank"]
        ),
        "matrices": entries,
    }


def _stationary_hmm(transition: list[list[float]], emission: list[list[float]]) -> HMM:
    provisional = HMM(np.asarray(transition), np.asarray(emission), np.full(4, 0.25))
    return HMM(provisional.transition, provisional.emission, provisional.stationary())


def select_grid_family(passers: Sequence[tuple[object, object]]) -> dict[str, tuple[object, object]]:
    """Apply the amendment's two deterministic rankings to already-diagnosed grid passers."""
    if not passers:
        raise HMMFamilyError("candidate grid has no passing HMMs")
    ordered = sorted(passers, key=lambda item: item[0].key())
    dwell = np.asarray([item[1].values["mean_dwell_time"] for item in ordered])
    median_dwell = float(np.median(dwell))
    fast_ranking = sorted(ordered, key=lambda item: (item[1].values["mean_dwell_time"], item[0].key()))
    persistent = [item for item in ordered if item[1].values["mean_dwell_time"] >= median_dwell]
    entropy_ranking = sorted(
        persistent,
        key=lambda item: (-item[1].values["belief_entropy_mean_bits"], item[0].key()),
    )

    def first_certified(ranking):
        rejected = []
        for candidate, diagnostics in ranking:
            cert = matrix_certificate(candidate.build())
            te = cert["matrices"]["transition_times_emission"]
            if te["rank"] == 4 and te["sigma_min"] > .05:
                return (candidate, diagnostics), rejected
            rejected.append({
                "candidate": candidate.to_dict(), "hmm_sha256": candidate.build().sha256(),
                "rank_te": te["rank"], "sigma_min_te": te["sigma_min"],
            })
        raise HMMFamilyError("no candidate in a frozen ranking passes the TE gate")

    fast, fast_rejected = first_certified(fast_ranking)
    high, high_rejected = first_certified(entropy_ranking)
    return {
        "fast_mixing_moderate": fast,
        "persistent_high_aliasing": high,
        "_metadata": {  # type: ignore[dict-item]
            "median_passing_mean_dwell_time": median_dwell,
            "fast_mixing_rejections": fast_rejected,
            "persistent_high_aliasing_rejections": high_rejected,
        },
    }


def family_payload(
    primary: HMM, *, primary_manifest_sha256: str,
    selected: Mapping[str, tuple[object, object]], passing_order: Sequence[tuple[object, object]],
    primary_candidate: Mapping[str, object] | None = None,
    primary_diagnostics: Mapping[str, float] | None = None,
) -> dict[str, object]:
    regimes: dict[str, HMM] = {
        PRIMARY_REGIME: primary,
        "fast_mixing_moderate": selected["fast_mixing_moderate"][0].build(),
        "persistent_high_aliasing": selected["persistent_high_aliasing"][0].build(),
    }
    purposes = {
        PRIMARY_REGIME: "existing frozen persistent/moderate-aliasing grid winner",
        "fast_mixing_moderate": "lowest-mean-dwell passing grid candidate after the TE gate",
        "persistent_high_aliasing": ">=median-dwell passing candidate with highest posterior entropy after the TE gate",
    }
    records = {}
    for index, name in enumerate(REGIMES):
        hmm = regimes[name]
        records[name] = {
            "order": index,
            "role": "equal_weight_calibration_regime",
            "purpose": purposes[name],
            "hmm": hmm.to_dict(),
            "hmm_sha256": hmm.sha256(),
            "selected_candidate": (
                primary_candidate if name == PRIMARY_REGIME else selected[name][0].to_dict()
            ),
            "selection_diagnostics": (
                primary_diagnostics if name == PRIMARY_REGIME else selected[name][1].values
            ),
            "linear_certificate": matrix_certificate(hmm),
            "data_seed": FAMILY_SEED_BASE + index * FAMILY_SEED_STRIDE,
            "common_random_numbers_across_regimes": False,
            "artifact_layout": {
                "data_dir": f"data/hmm_family/{name}",
                "manifest_dir": f"manifests/hmm_family/{name}",
            },
        }
        te = records[name]["linear_certificate"]["matrices"]["transition_times_emission"]
        if te["rank"] != 4 or te["sigma_min"] <= .05:
            raise HMMFamilyError(f"{name}: frozen matrix fails rank/sigma_min TE gate")
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "frozen_date": "2026-08-24",
        "selection_blinding": {
            "model_checkpoints_inspected": False,
            "model_representations_inspected": False,
            "model_outcomes_inspected": False,
            "rule": "matrices selected from the unchanged deterministic grid by frozen dwell/entropy rankings and TE gate",
        },
        "primary_regime": None,
        "required_regimes": list(REGIMES),
        "primary_manifest_sha256": primary_manifest_sha256,
        "selection_rule": {
            "source": "existing deterministic candidate grid and unchanged acceptance box",
            "fast_mixing_moderate": "ascending mean dwell, then lexicographic candidate tuple",
            "persistent_high_aliasing": "among dwell >= median passer: descending mean posterior entropy, then lexicographic tuple",
            "te_gate": {"rank_te": 4, "sigma_min_te_strictly_greater_than": .05},
            "metadata": selected.get("_metadata"),
        },
        "passing_candidates_lexicographic": [
            {
                "candidate": candidate.to_dict(), "hmm_sha256": candidate.build().sha256(),
                "diagnostics": diagnostics.values,
            }
            for candidate, diagnostics in passing_order
        ],
        "regimes": records,
        "materialization": {
            "splits": {"train": [100000, 32], "val": [10000, 32], "lengen": [10000, 64]},
            "pair_bank_calibration_rows": [0, 5000],
            "pair_bank_test_rows": [5000, 10000],
            "thresholds_fit_separately_per_regime": True,
            "target_pairs_per_type_and_length_band": 2000,
            "pair_selection_distance": "exact_future_distribution_js_bits",
            "belief_js_secondary_only": True,
            "selection_may_use_only_exact_HMM_future_distributions_and_surface_edit_distance": True,
            "model_outcomes_permitted": False,
        },
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    return payload


def load_family(path: Path) -> tuple[dict[str, HMM], dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("payload_sha256", None)
    actual = canonical_sha256(payload)
    payload["payload_sha256"] = claimed
    if payload.get("schema") != SCHEMA or claimed != actual:
        raise HMMFamilyError("HMM family manifest schema/hash mismatch")
    if tuple(payload.get("required_regimes", ())) != REGIMES:
        raise HMMFamilyError("HMM family is missing or reorders a required regime")
    records = payload.get("regimes")
    if not isinstance(records, dict) or set(records) != set(REGIMES):
        raise HMMFamilyError("HMM family regime set is incomplete")
    hmms: dict[str, HMM] = {}
    for name in REGIMES:
        record = records[name]
        hmm = HMM.from_dict(record["hmm"])
        if hmm.sha256() != record.get("hmm_sha256"):
            raise HMMFamilyError(f"{name}: matrix identity mismatch")
        _verify_linear_certificate(
            record.get("linear_certificate"), matrix_certificate(hmm), regime=name
        )
        hmms[name] = hmm
    return hmms, payload


def freeze_family(
    payload: dict[str, object], path: Path, *, allow_marked_supersession: bool = False
) -> dict[str, object]:
    """Create-only family freeze; an identical retry is a no-op."""
    if path.exists():
        try:
            _, existing = load_family(path)
        except (HMMFamilyError, KeyError, TypeError):
            existing = json.loads(path.read_text(encoding="utf-8"))
        if existing == payload:
            return existing
        marker = path.with_name("hmm_family.SUPERSEDED.json")
        marked = json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else {}
        if not (
            allow_marked_supersession
            and marked.get("status") == "INVALID_DO_NOT_MATERIALIZE_OR_TRAIN"
            and existing.get("payload_sha256") in marked.get(
                "superseded_payload_sha256s", [marked.get("superseded_payload_sha256")]
            )
        ):
            raise HMMFamilyError(f"{path} already freezes a different HMM family")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
