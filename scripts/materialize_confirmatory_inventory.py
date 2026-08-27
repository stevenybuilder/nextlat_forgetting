#!/usr/bin/env python3
"""Build the authoritative frozen-input inventory for the reduced program.

The reduced confirmatory program contains Lure-Star H1/H2 and the complete
three-regime HMM family.  Lure-Star H3 is permanently excluded, so the exact
D40 block (rather than any adaptation bank) is part of the input snapshot.

This script never discovers files recursively.  It accepts only the frozen
root identities below and the exact, independently frozen HMM-family
subinventory.  Consequently a rerun cannot silently bless a changed manifest,
an incomplete HMM regime, or a retired/superseded artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
import tempfile
import typing as t


INVENTORY_RELATIVE = "manifests/manifest_inventory.sha256"
HMM_INVENTORY_RELATIVE = "manifests/hmm_family_inventory.sha256"
HMM_RECEIPT_RELATIVE = "manifests/hmm_family_materialization.json"
H3_BLOCK_RELATIVE = "manifests/h3_selected/PERMANENT_H3_BLOCK.json"
H3_SIDECAR_RELATIVE = H3_BLOCK_RELATIVE + ".sha256"

SHA_RE = re.compile(r"[0-9a-f]{64}")
LINE_RE = re.compile(r"([0-9a-f]{64})  ([^\r\n]+)")

HMM_REGIMES = (
    "persistent_moderate",
    "fast_mixing_moderate",
    "persistent_high_aliasing",
)
HMM_DATA_FILES = (
    "hmm4x4_lengen_len64_10000.npy",
    "hmm4x4_lengen_posteriors.npz",
    "hmm4x4_train_len32_100000.npy",
    "hmm4x4_val_len32_10000.npy",
    "hmm4x4_val_posteriors.npz",
)
HMM_MANIFEST_FILES = (
    "hmm_dataset.json",
    "hmm_eval_pairs.json",
    "hmm_eval_pairs.jsonl",
    "hmm_matrices.json",
    "hmm_thresholds.json",
)

# These are preregistered/frozen identities, not values learned by scanning the
# working tree.  Changing a file and its adjacent sidecar together is therefore
# still refused.
FROZEN_ROOT_SHA256: dict[str, str] = {
    "manifests/a_pair.jsonl": "e70fb087b6b1dd6fa7129303bbc4bcc30843c327fcab168937976295cbf2dd10",
    "manifests/a_pair.jsonl.sha256": "e680cb4d0db045cd958db32cc3d14a926d29017814de87c9a3e9bcd254fe4d6e",
    "manifests/b_far.jsonl": "364978600eb73a6e9044e812dd974fe6a2df509b7f256079dc3c7d2ec8ab99e3",
    "manifests/b_far.jsonl.sha256": "530c06ce77f9132726eca5f4bc6fc9f32ae4d21fddafe0fa838f74e39e48fd86",
    "manifests/b_near.jsonl": "7e4a414fc51c693e850fb5a0e01a651e3e78cb01304ddf1704cf11aad5314528",
    "manifests/b_near.jsonl.sha256": "00ed1abec565d8bf5b6e7beb1a3fffa15dc888905b1ba0fa99e4f1839245de4c",
    "manifests/corpus.sha256": "d50bf5da1d8975e46befe155229987933c8147c9b51af13d6b7ea746d21ac5a8",
    "manifests/corpus_provenance.json": "37a711c86c21b57ad339b13799e2a06a886a0854d5d1dad5fecaf8b283ff45cd",
    "manifests/e_lure.jsonl": "f67765e6ea2afd4156c9d03ad0271afe224f1a54ddf1afc82118fcc3e4541495",
    "manifests/e_lure.jsonl.sha256": "9e339f971300d13bdb751c7ddc52fef59f03d2179659550e92850abd63fb9af2",
    H3_BLOCK_RELATIVE: "82d526ad5cb6ac5fb942790488a6b766e59b816acb27ed405a00852f40925778",
    H3_SIDECAR_RELATIVE: "24b47f2d49e084d4b09e39393938294d7ef1a7ba6e5bbf90d56b4e7145a65d0b",
    "manifests/stimuli_provenance.json": "b7e0675bddce8e911ad35235543d65f2b2db62d57437b558d013b39f32b62ecf",
}
FROZEN_HMM_INVENTORY_SHA256 = (
    "079eeaa283a4d91eb4512726195fdc08a291e2020bb97507c8d8380ebc32e8d2"
)
FROZEN_HMM_RECEIPT_SHA256 = (
    "20ce1a55d40e3302d35a4cb9063f1341e451e7ddf00171fcf9766198f77d1672"
)

H3_FORBIDDEN = (
    "candidate_expansion",
    "caliper_change",
    "weighting",
    "unmatched_restriction",
    "pilot_substitution",
    "matching_amendment",
)


class InventoryError(RuntimeError):
    """The frozen input set is incomplete, stale, or scientifically invalid."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InventoryError(message)


def sha256_file(path: pathlib.Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_hmm_paths() -> frozenset[str]:
    paths = {"manifests/hmm_family.json"}
    for regime in HMM_REGIMES:
        paths.update(
            f"data/hmm_family/{regime}/{name}" for name in HMM_DATA_FILES
        )
        paths.update(
            f"manifests/hmm_family/{regime}/{name}" for name in HMM_MANIFEST_FILES
        )
    return frozenset(paths)


def _plain_file(root: pathlib.Path, relative: str) -> pathlib.Path:
    pure = pathlib.PurePosixPath(relative)
    require(not pure.is_absolute() and str(pure) == relative, f"noncanonical path: {relative}")
    require(all(part not in ("", ".", "..") for part in pure.parts), f"unsafe path: {relative}")
    cursor = root
    for part in pure.parts:
        cursor /= part
        require(not cursor.is_symlink(), f"symlinked frozen input: {relative}")
    try:
        resolved = cursor.resolve(strict=True)
    except OSError as exc:
        raise InventoryError(f"missing frozen input: {relative}") from exc
    require(root == resolved or root in resolved.parents, f"path escapes project root: {relative}")
    require(cursor.is_file(), f"frozen input is not a regular file: {relative}")
    return cursor


def _strict_inventory(path: pathlib.Path) -> dict[str, str]:
    try:
        text = path.read_bytes().decode("utf-8", "strict")
    except (OSError, UnicodeDecodeError) as exc:
        raise InventoryError(f"cannot read strict inventory: {path}") from exc
    require(bool(text) and text.endswith("\n"), f"inventory is empty or lacks final newline: {path}")
    rows: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        match = LINE_RE.fullmatch(line)
        require(match is not None, f"malformed inventory row {path}:{line_number}")
        digest, relative = t.cast(re.Match[str], match).groups()
        require(relative not in rows, f"duplicate inventory path: {relative}")
        rows[relative] = digest
    return rows


def _verify_sidecar(root: pathlib.Path, payload_relative: str) -> None:
    payload = _plain_file(root, payload_relative)
    sidecar_relative = payload_relative + ".sha256"
    sidecar = _plain_file(root, sidecar_relative)
    match = LINE_RE.fullmatch(sidecar.read_text(encoding="utf-8").rstrip("\n"))
    require(match is not None, f"malformed sidecar: {sidecar_relative}")
    expected, name = t.cast(re.Match[str], match).groups()
    require(name == payload.name, f"sidecar names another file: {sidecar_relative}")
    require(expected == sha256_file(payload), f"sidecar SHA mismatch: {payload_relative}")


def _verify_h3(root: pathlib.Path) -> None:
    _verify_sidecar(root, H3_BLOCK_RELATIVE)
    try:
        document = json.loads(_plain_file(root, H3_BLOCK_RELATIVE).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InventoryError("canonical permanent H3 block is invalid JSON") from exc
    expected = {
        "schema": "nextlat_forgetting/h3_mid_expansion/1",
        "status": "PERMANENT_H3_BLOCK",
        "reason": "D40_ONE_SHOT_EXPANSION_REMAINS_INFEASIBLE",
        "unmatched_count": 4,
        "no_further_amendments_permitted": True,
    }
    require(
        all(document.get(key) == value for key, value in expected.items()),
        "canonical permanent H3 block semantics changed",
    )
    require(tuple(document.get("forbidden", ())) == H3_FORBIDDEN, "H3 forbidden-actions changed")


def _verify_lurestar(root: pathlib.Path) -> None:
    for relative, expected in FROZEN_ROOT_SHA256.items():
        actual = sha256_file(_plain_file(root, relative))
        require(actual == expected, f"frozen root identity changed: {relative}")
    for stem in ("a_pair", "b_near", "b_far", "e_lure"):
        _verify_sidecar(root, f"manifests/{stem}.jsonl")
    _verify_h3(root)
    try:
        provenance = json.loads(
            _plain_file(root, "manifests/stimuli_provenance.json").read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as exc:
        raise InventoryError("stimuli provenance is invalid JSON") from exc
    recorded = provenance.get("sha256")
    require(isinstance(recorded, dict), "stimuli provenance lacks SHA map")
    for stem in ("a_pair", "b_near", "b_far", "e_lure"):
        relative = f"manifests/{stem}.jsonl"
        require(recorded.get(f"{stem}.jsonl") == sha256_file(root / relative),
                f"stimuli provenance does not bind {stem}.jsonl")


def _verify_hmm_family(root: pathlib.Path) -> dict[str, str]:
    inventory = _plain_file(root, HMM_INVENTORY_RELATIVE)
    require(
        sha256_file(inventory) == FROZEN_HMM_INVENTORY_SHA256,
        "frozen HMM-family inventory identity changed",
    )
    rows = _strict_inventory(inventory)
    expected_paths = expected_hmm_paths()
    require(set(rows) == expected_paths, "HMM-family inventory membership is not exact")
    for relative, expected in rows.items():
        require(sha256_file(_plain_file(root, relative)) == expected,
                f"HMM-family artifact hash mismatch: {relative}")

    receipt_path = _plain_file(root, HMM_RECEIPT_RELATIVE)
    require(
        sha256_file(receipt_path) == FROZEN_HMM_RECEIPT_SHA256,
        "frozen HMM-family materialization receipt identity changed",
    )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        family = json.loads((root / "manifests/hmm_family.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InventoryError("HMM family manifest or receipt is invalid JSON") from exc
    require(receipt.get("schema") == "nextlat_forgetting/hmm_family_materialization/1",
            "unexpected HMM-family receipt schema")
    require(receipt.get("status") == "complete", "HMM-family materialization is incomplete")
    require(receipt.get("n_artifacts") == len(expected_paths), "HMM-family artifact count changed")
    require(tuple(receipt.get("required_regimes", ())) == HMM_REGIMES,
            "HMM-family regime set/order changed")
    require(receipt.get("inventory_sha256") == FROZEN_HMM_INVENTORY_SHA256,
            "HMM-family receipt does not bind its frozen inventory")
    require(receipt.get("family_sha256") == family.get("payload_sha256"),
            "HMM-family receipt does not bind its family manifest")
    require(receipt.get("model_inputs_used") == [], "HMM family used model inputs")
    require(receipt.get("model_outcomes_inspected") is False, "HMM family was not outcome-blind")
    return rows


def inventory_bytes(root: pathlib.Path) -> bytes:
    root = root.resolve()
    _verify_lurestar(root)
    hmm_rows = _verify_hmm_family(root)
    entries = dict(FROZEN_ROOT_SHA256)
    entries.update(hmm_rows)
    entries[HMM_INVENTORY_RELATIVE] = FROZEN_HMM_INVENTORY_SHA256
    entries[HMM_RECEIPT_RELATIVE] = FROZEN_HMM_RECEIPT_SHA256
    require(len(entries) == 46, "authoritative input membership count changed")
    require(INVENTORY_RELATIVE not in entries, "inventory cannot list itself")
    return "".join(f"{entries[path]}  {path}\n" for path in sorted(entries)).encode("utf-8")


def _atomic_replace(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary = pathlib.Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def materialize(root: pathlib.Path, *, check: bool = False) -> dict[str, t.Any]:
    payload = inventory_bytes(root)
    destination = root.resolve() / INVENTORY_RELATIVE
    digest = hashlib.sha256(payload).hexdigest()
    if check:
        require(destination.is_file(), "authoritative manifest inventory is absent")
        require(destination.read_bytes() == payload, "authoritative manifest inventory is stale")
        status = "VERIFIED"
    else:
        if not destination.is_file() or destination.read_bytes() != payload:
            _atomic_replace(destination, payload)
        status = "MATERIALIZED"
    return {
        "schema": "nextlat_forgetting/confirmatory_input_inventory/1",
        "status": status,
        "path": INVENTORY_RELATIVE,
        "entry_count": len(payload.splitlines()),
        "sha256": digest,
        "program": ["Lure-Star-H1", "Lure-Star-H2", "HMM-three-regime-family"],
        "h3_status": "PERMANENT_H3_BLOCK",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=pathlib.Path.cwd())
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="atomically publish the exact inventory")
    mode.add_argument("--check", action="store_true", help="verify that the published inventory is exact")
    args = parser.parse_args(argv)
    try:
        document = materialize(args.root, check=args.check)
    except (InventoryError, OSError) as exc:
        parser.exit(2, f"BLOCKED: {exc}\n")
    print(json.dumps(document, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
