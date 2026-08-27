#!/usr/bin/env python
"""Materialize the frozen model-blind HMM family before confirmatory training.

``--plan`` is read-only. ``--materialize`` creates every regime's matrices, corpus, posterior
arrays, calibration thresholds, and pair bank, then writes a complete SHA-256 inventory last.
Selection code has no model/checkpoint/result inputs by construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from hmm_geometry.family import (  # noqa: E402
    FAMILY_SEED_BASE, FAMILY_SEED_STRIDE, REGIMES, canonical_sha256, family_payload,
    freeze_family, load_family, select_grid_family,
)
from hmm_geometry.generate import (  # noqa: E402
    ACCEPTANCE, Candidate, Diagnostics, _grid, diagnose, generate_corpus, load_frozen_hmm,
)
from hmm_geometry.pair_bank import (  # noqa: E402
    build_bank, fit_thresholds, freeze_thresholds, load_pools, load_thresholds, write_bank,
)
from lurestar.durable_checkpoint import atomic_write_text, sha256_file  # noqa: E402

FAMILY_MANIFEST = _REPO / "manifests" / "hmm_family.json"
FAMILY_INVENTORY = _REPO / "manifests" / "hmm_family_inventory.sha256"
MATERIALIZATION_RECEIPT = _REPO / "manifests" / "hmm_family_materialization.json"
PAIR_SEED = 20_260_824


class MaterializationError(RuntimeError):
    pass


def build_amendment_family(root: pathlib.Path) -> dict:
    """Run the unchanged model-blind grid once and apply the frozen family rankings."""
    primary, primary_manifest = load_frozen_hmm(root / "manifests/hmm_matrices.json")
    passers = []
    existing_path = root / "manifests/hmm_family.json"
    try:
        _, existing = load_family(existing_path)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError):
        existing = {}
    cached = existing.get("passing_candidates_lexicographic", [])
    if isinstance(cached, list) and cached:
        for row in cached:
            c = row["candidate"]
            candidate = Candidate(
                c["self_transition"], c["persistence_tilt"], tuple(c["offdiag_skew"]),
                c["emission_self"], c["emission_neighbour"],
            )
            passers.append((candidate, Diagnostics(values=row["diagnostics"], passed=True)))
    else:
        for candidate in _grid():
            hmm = candidate.build()
            # Match generate.search's operational pass set exactly, including its analytic prefilter.
            if not (
                ACCEPTANCE["min_self_transition"][0]
                <= np.diag(hmm.transition).min()
                <= ACCEPTANCE["max_self_transition"][1]
            ):
                continue
            dwell = hmm.mean_dwell_time()
            if not (ACCEPTANCE["mean_dwell_time"][0] <= dwell <= ACCEPTANCE["mean_dwell_time"][1]):
                continue
            diagnostics = diagnose(hmm)
            if diagnostics.passed:
                passers.append((candidate, diagnostics))
    passers.sort(key=lambda item: item[0].key())
    selected = select_grid_family(passers)
    return family_payload(
        primary,
        primary_manifest_sha256=sha256_file(root / "manifests/hmm_matrices.json"),
        selected=selected, passing_order=passers,
        primary_candidate=primary_manifest.get("candidate"),
        primary_diagnostics=primary_manifest.get("diagnostics"),
    )


def regime_paths(root: pathlib.Path, regime: str) -> dict[str, pathlib.Path]:
    return {
        "data_dir": root / "data" / "hmm_family" / regime,
        "manifest_dir": root / "manifests" / "hmm_family" / regime,
        "matrix": root / "manifests" / "hmm_family" / regime / "hmm_matrices.json",
        "dataset": root / "manifests" / "hmm_family" / regime / "hmm_dataset.json",
        "thresholds": root / "manifests" / "hmm_family" / regime / "hmm_thresholds.json",
        "pairs": root / "manifests" / "hmm_family" / regime / "hmm_eval_pairs.jsonl",
        "pairs_manifest": root / "manifests" / "hmm_family" / regime / "hmm_eval_pairs.json",
    }


def _matrix_payload(hmm, family_sha: str, regime: str) -> dict:
    body = {
        "schema": "nextlat_forgetting/hmm_matrices/1",
        "n_states": hmm.n_states,
        "n_obs": hmm.n_obs,
        "hmm": hmm.to_dict(),
        "hmm_sha256": hmm.sha256(),
        "family_sha256": family_sha,
        "family_regime": regime,
        "selection": "model_blind_fixed_family",
    }
    return {**body, "payload_sha256": canonical_sha256(body)}


def _write_create_only_json(path: pathlib.Path, payload: dict) -> None:
    encoded = json.dumps(payload, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise MaterializationError(f"refusing to replace frozen artifact {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def _verify_existing_dataset(manifest_path: pathlib.Path, expected: list[pathlib.Path], hmm_sha: str) -> bool:
    if not manifest_path.is_file() and not any(path.exists() for path in expected):
        return False
    if not manifest_path.is_file() or not all(path.is_file() for path in expected):
        raise MaterializationError(f"partial dataset exists beside {manifest_path}; refusing overwrite")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("hmm_sha256") != hmm_sha:
        raise MaterializationError(f"{manifest_path}: dataset binds a different HMM")
    recorded = {}
    for split in document.get("splits", {}).values():
        for key in ("observations_file", "posteriors_file"):
            if key in split:
                recorded[pathlib.Path(split[key]).name] = split[key.replace("file", "sha256")]
    for path in expected:
        if recorded.get(path.name) != sha256_file(path):
            raise MaterializationError(f"{path}: existing dataset hash mismatch")
    return True


def _verify_existing_pairs(paths: dict[str, pathlib.Path], hmm_sha: str) -> bool:
    pair_path, manifest_path = paths["pairs"], paths["pairs_manifest"]
    if not pair_path.exists() and not manifest_path.exists():
        return False
    if not pair_path.is_file() or not manifest_path.is_file():
        raise MaterializationError("partial pair-bank freeze exists; refusing overwrite")
    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    if document.get("hmm_sha256") != hmm_sha or document.get("pairs_sha256") != sha256_file(pair_path):
        raise MaterializationError(f"{pair_path}: existing pair-bank identity mismatch")
    return True


def planned_artifacts(root: pathlib.Path) -> list[pathlib.Path]:
    paths = [root / "manifests/hmm_family.json"]
    for regime in REGIMES:
        item = regime_paths(root, regime)
        paths.extend(item[key] for key in ("matrix", "dataset", "thresholds", "pairs", "pairs_manifest"))
        paths.extend([
            item["data_dir"] / "hmm4x4_train_len32_100000.npy",
            item["data_dir"] / "hmm4x4_val_len32_10000.npy",
            item["data_dir"] / "hmm4x4_lengen_len64_10000.npy",
            item["data_dir"] / "hmm4x4_val_posteriors.npz",
            item["data_dir"] / "hmm4x4_lengen_posteriors.npz",
        ])
    return paths


def materialize(root: pathlib.Path, *, target_pairs: int = 2_000) -> dict:
    payload = build_amendment_family(root)
    freeze_family(
        payload, root / "manifests/hmm_family.json", allow_marked_supersession=True
    )
    hmms, family = load_family(root / "manifests/hmm_family.json")

    for regime_index, regime in enumerate(REGIMES):
        paths = regime_paths(root, regime)
        hmm = hmms[regime]
        matrix = _matrix_payload(hmm, str(family["payload_sha256"]), regime)
        _write_create_only_json(paths["matrix"], matrix)
        # Corpus generation is deterministic. Existing complete regimes are verified below and
        # skipped, so a retry never spends CPU regenerating valuable exact posterior arrays.
        expected_data = [
            paths["data_dir"] / "hmm4x4_train_len32_100000.npy",
            paths["data_dir"] / "hmm4x4_val_len32_10000.npy",
            paths["data_dir"] / "hmm4x4_lengen_len64_10000.npy",
            paths["data_dir"] / "hmm4x4_val_posteriors.npz",
            paths["data_dir"] / "hmm4x4_lengen_posteriors.npz",
        ]
        if not _verify_existing_dataset(paths["dataset"], expected_data, hmm.sha256()):
            generate_corpus(
                hmm, data_dir=paths["data_dir"], manifest_path=paths["dataset"],
                hmm_manifest=matrix,
                data_seed=FAMILY_SEED_BASE + regime_index * FAMILY_SEED_STRIDE,
            )
        pools = load_pools(paths["data_dir"])
        calibration = [pools["calibration32"], pools["calibration64"]]
        tests = [pools["test32"], pools["test64"]]
        if not paths["thresholds"].is_file():
            fitted = fit_thresholds(
                calibration, hmm, seed=PAIR_SEED + regime_index * 10,
                distance_target="future_js",
            )
            freeze_thresholds(fitted, paths["thresholds"])
        thresholds = load_thresholds(paths["thresholds"])
        if thresholds.hmm_sha256 != hmm.sha256():
            raise MaterializationError(f"{regime}: thresholds bind a different HMM")
        if not _verify_existing_pairs(paths, hmm.sha256()):
            forbidden = set().union(*(pool.prefix_keys() for pool in calibration))
            banks = [
                build_bank(
                    pool, hmm, thresholds,
                    seed=PAIR_SEED + regime_index * 10 + offset,
                    target_pairs=target_pairs, forbidden_prefixes=forbidden,
                )
                for offset, pool in enumerate(tests)
            ]
            write_bank(
                banks, thresholds, hmm, bank_path=paths["pairs"],
                manifest_path=paths["pairs_manifest"],
            )

    artifacts = planned_artifacts(root)
    missing = [str(path) for path in artifacts if not path.is_file()]
    if missing:
        raise MaterializationError(f"family materialization incomplete: {missing}")
    inventory_lines = []
    for path in sorted(artifacts):
        inventory_lines.append(f"{sha256_file(path)}  {path.relative_to(root)}\n")
    inventory_body = "".join(inventory_lines)
    inventory_path = root / "manifests/hmm_family_inventory.sha256"
    if inventory_path.exists() and inventory_path.read_text() != inventory_body:
        raise MaterializationError("refusing to replace a different frozen family inventory")
    atomic_write_text(inventory_path, inventory_body)
    receipt = {
        "schema": "nextlat_forgetting/hmm_family_materialization/1",
        "status": "complete",
        "family_sha256": family["payload_sha256"],
        "inventory_sha256": hashlib.sha256(inventory_body.encode()).hexdigest(),
        "n_artifacts": len(artifacts),
        "required_regimes": list(REGIMES),
        "target_pairs": target_pairs,
        "model_inputs_used": [],
        "model_outcomes_inspected": False,
    }
    _write_create_only_json(root / "manifests/hmm_family_materialization.json", receipt)
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=_REPO)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--plan", action="store_true")
    group.add_argument(
        "--freeze-family", action="store_true",
        help="freeze matrices/certificates only; performs no corpus or pair-bank search",
    )
    group.add_argument("--materialize", action="store_true")
    parser.add_argument("--target-pairs", type=int, default=2_000)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.target_pairs != 2_000:
        print("REFUSED: confirmatory target_pairs is frozen at 2000", file=sys.stderr)
        return 2
    if args.plan:
        print(json.dumps({
            "schema": "nextlat_forgetting/hmm_family_materialization_plan/1",
            "model_blind": True,
            "required_regimes": list(REGIMES),
            "artifacts": [str(path) for path in planned_artifacts(root)],
        }, indent=2))
        return 0
    if args.freeze_family:
        try:
            payload = build_amendment_family(root)
            freeze_family(
                payload, root / "manifests/hmm_family.json", allow_marked_supersession=True
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"REFUSED/FAILED: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(payload, indent=2))
        return 0
    try:
        print(json.dumps(materialize(root, target_pairs=args.target_pairs), indent=2))
    except (MaterializationError, OSError, RuntimeError, ValueError) as exc:
        print(f"REFUSED/FAILED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
