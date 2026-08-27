from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "materialize_confirmatory_inventory",
    ROOT / "scripts/materialize_confirmatory_inventory.py",
)
M = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write(root: pathlib.Path, relative: str, payload: bytes) -> pathlib.Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def sidecar(payload: bytes, name: str) -> bytes:
    return f"{digest(payload)}  {name}\n".encode()


def synthetic_project(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> pathlib.Path:
    root = tmp_path / "project"
    root_payloads: dict[str, bytes] = {
        "manifests/corpus.sha256": b"frozen corpus identity\n",
        "manifests/corpus_provenance.json": b'{"frozen":true}\n',
    }
    stimulus_hashes: dict[str, str] = {}
    for stem in ("a_pair", "b_near", "b_far", "e_lure"):
        name = f"{stem}.jsonl"
        payload = f'{{"condition":"{stem}"}}\n'.encode()
        stimulus_hashes[name] = digest(payload)
        root_payloads[f"manifests/{name}"] = payload
        root_payloads[f"manifests/{name}.sha256"] = sidecar(payload, name)
    root_payloads["manifests/stimuli_provenance.json"] = (
        json.dumps({"sha256": stimulus_hashes}, sort_keys=True) + "\n"
    ).encode()
    block = {
        "schema": "nextlat_forgetting/h3_mid_expansion/1",
        "status": "PERMANENT_H3_BLOCK",
        "reason": "D40_ONE_SHOT_EXPANSION_REMAINS_INFEASIBLE",
        "unmatched_count": 4,
        "no_further_amendments_permitted": True,
        "forbidden": list(M.H3_FORBIDDEN),
    }
    block_payload = (json.dumps(block, sort_keys=True) + "\n").encode()
    root_payloads[M.H3_BLOCK_RELATIVE] = block_payload
    root_payloads[M.H3_SIDECAR_RELATIVE] = sidecar(block_payload, "PERMANENT_H3_BLOCK.json")
    for relative, payload in root_payloads.items():
        write(root, relative, payload)
    monkeypatch.setattr(
        M, "FROZEN_ROOT_SHA256",
        {relative: digest(payload) for relative, payload in root_payloads.items()},
    )

    family_sha = "f" * 64
    hmm_payloads: dict[str, bytes] = {}
    for relative in M.expected_hmm_paths():
        payload = b"hmm-artifact\n"
        if relative == "manifests/hmm_family.json":
            payload = (json.dumps({"payload_sha256": family_sha}) + "\n").encode()
        hmm_payloads[relative] = payload
        write(root, relative, payload)
    hmm_inventory = "".join(
        f"{digest(hmm_payloads[relative])}  {relative}\n"
        for relative in sorted(hmm_payloads)
    ).encode()
    write(root, M.HMM_INVENTORY_RELATIVE, hmm_inventory)
    monkeypatch.setattr(M, "FROZEN_HMM_INVENTORY_SHA256", digest(hmm_inventory))
    receipt = {
        "schema": "nextlat_forgetting/hmm_family_materialization/1",
        "status": "complete",
        "family_sha256": family_sha,
        "inventory_sha256": digest(hmm_inventory),
        "n_artifacts": 31,
        "required_regimes": list(M.HMM_REGIMES),
        "model_inputs_used": [],
        "model_outcomes_inspected": False,
    }
    receipt_payload = (json.dumps(receipt, sort_keys=True) + "\n").encode()
    write(root, M.HMM_RECEIPT_RELATIVE, receipt_payload)
    monkeypatch.setattr(M, "FROZEN_HMM_RECEIPT_SHA256", digest(receipt_payload))
    return root


def test_checked_in_inventory_has_exact_reduced_program_membership() -> None:
    payload = M.inventory_bytes(ROOT)
    rows = payload.decode().splitlines()
    paths = [row.split("  ", 1)[1] for row in rows]
    expected = set(M.FROZEN_ROOT_SHA256) | set(M.expected_hmm_paths()) | {
        M.HMM_INVENTORY_RELATIVE,
        M.HMM_RECEIPT_RELATIVE,
    }

    assert len(rows) == 46
    assert paths == sorted(paths)
    assert set(paths) == expected
    assert (ROOT / M.INVENTORY_RELATIVE).read_bytes() == payload
    assert not any(path.startswith("data/hmm/") for path in paths)
    assert not any("adapt" in path or "h3_expansion" in path or "h3_precompute" in path
                   for path in paths)
    assert not any(path.endswith("hmm_family.SUPERSEDED.json") for path in paths)


@pytest.mark.parametrize("target", ["h3", "hmm"])
def test_refuses_mutation_instead_of_blessing_it(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch, target: str,
) -> None:
    root = synthetic_project(tmp_path, monkeypatch)
    result = M.materialize(root)
    inventory_before = (root / M.INVENTORY_RELATIVE).read_bytes()
    assert result["entry_count"] == 46

    if target == "h3":
        block_path = root / M.H3_BLOCK_RELATIVE
        document = json.loads(block_path.read_text())
        document["unmatched_count"] = 5
        changed = (json.dumps(document, sort_keys=True) + "\n").encode()
        block_path.write_bytes(changed)
        (root / M.H3_SIDECAR_RELATIVE).write_bytes(
            sidecar(changed, "PERMANENT_H3_BLOCK.json")
        )
        expected_message = "frozen root identity changed"
    else:
        relative = sorted(path for path in M.expected_hmm_paths()
                          if path.startswith("data/hmm_family/"))[0]
        (root / relative).write_bytes(b"mutated\n")
        expected_message = "HMM-family artifact hash mismatch"

    with pytest.raises(M.InventoryError, match=expected_message):
        M.materialize(root)
    assert (root / M.INVENTORY_RELATIVE).read_bytes() == inventory_before


def test_check_refuses_stale_main_inventory(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = synthetic_project(tmp_path, monkeypatch)
    M.materialize(root)
    (root / M.INVENTORY_RELATIVE).write_text("0" * 64 + "  manifests/not-authoritative\n")
    with pytest.raises(M.InventoryError, match="inventory is stale"):
        M.materialize(root, check=True)
