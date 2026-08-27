from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import validate_nl1_declaration as V  # noqa: E402


DECLARATION = ROOT / "manifests/nl1/design_declaration.json"
TS1_DECLARATION = ROOT / "manifests/nl1/ts1_design_declaration.json"
PROGRAM_AMENDMENT = ROOT / "manifests/nl1/language_program_amendment.json"


def test_frozen_nl1_declaration_is_consistent_and_blocked() -> None:
    result = V.validate(DECLARATION, root=ROOT)
    assert result["status"] == "PASS_BLOCKED_AS_DECLARED"
    assert result["base_runs"] == 6
    assert result["adaptation_branches"] == 12
    assert result["base_token_exposures"] == 11_997_806_592
    assert result["adaptation_token_exposures"] == 629_145_600
    assert result["dataset_downloaded"] is False
    assert result["model_outcomes_opened"] is False


def test_validator_rejects_a_quiet_token_budget_change(tmp_path: Path) -> None:
    value = json.loads(DECLARATION.read_text())
    tampered = copy.deepcopy(value)
    tampered["design"]["base_training"]["optimizer_updates"] = 19074
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered))
    with pytest.raises(V.NL1DeclarationError, match="arithmetic mismatch|budget drift"):
        V.validate(path, root=ROOT)


def test_validator_rejects_claim_promotion_or_unverified_launch(tmp_path: Path) -> None:
    value = json.loads(DECLARATION.read_text())
    tampered = copy.deepcopy(value)
    tampered["primary_estimand"]["causal_mediation_claim_permitted"] = True
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered))
    with pytest.raises(V.NL1DeclarationError, match="causal mediation"):
        V.validate(path, root=ROOT)


def test_validator_rejects_upstream_hash_drift(tmp_path: Path) -> None:
    value = json.loads(DECLARATION.read_text())
    tampered = copy.deepcopy(value)
    tampered["upstream"]["gpt_config"]["sha256"] = "0" * 64
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered))
    with pytest.raises(V.NL1DeclarationError, match="hash mismatch"):
        V.validate(path, root=ROOT)


def test_validator_rejects_an_outcome_selectable_compute_fallback(tmp_path: Path) -> None:
    value = json.loads(DECLARATION.read_text())
    tampered = copy.deepcopy(value)
    tampered["profile"]["launch_decision_rule"][
        "reduced_scientific_fallback_permitted"
    ] = True
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered))
    with pytest.raises(V.NL1DeclarationError, match="compute decision rule"):
        V.validate(path, root=ROOT)


def test_ts1_parity_declaration_is_consistent_and_blocked() -> None:
    result = V.validate_ts1(TS1_DECLARATION, root=ROOT)
    assert result["status"] == "PASS_BLOCKED_AS_DECLARED"
    assert result["base_runs"] == 6
    assert result["probe_runs"] == 6
    assert result["base_token_exposures"] == 39_321_600_000
    assert result["probe_frozen_backbone_token_forwards"] == 7_864_320_000
    assert result["dataset_downloaded"] is False
    assert result["model_outcomes_opened"] is False


def test_language_program_preserves_original_and_requires_both_arms() -> None:
    result = V.validate_program(PROGRAM_AMENDMENT, root=ROOT)
    assert result["status"] == "PASS_BOTH_ARMS_BLOCKED_AS_DECLARED"
    assert result["frozen_nl1_sha256"] == V.FROZEN_NL1_SHA256
    assert result["required_arms"] == ["TS-1", "NL-1"]
    assert result["corpora_downloaded"] is False
    assert result["model_outcomes_opened"] is False


def test_ts1_validator_rejects_outcome_selectable_scope_expansion(
    tmp_path: Path,
) -> None:
    value = json.loads(TS1_DECLARATION.read_text())
    tampered = copy.deepcopy(value)
    tampered["design"]["models"].append("nextlat_d8")
    path = tmp_path / "tampered-ts1.json"
    path.write_text(json.dumps(tampered))
    with pytest.raises(V.NL1DeclarationError, match="model roster"):
        V.validate_ts1(path, root=ROOT)


def test_program_validator_rejects_optionalizing_an_arm(tmp_path: Path) -> None:
    value = json.loads(PROGRAM_AMENDMENT.read_text())
    tampered = copy.deepcopy(value)
    tampered["program"]["arms"][0]["required"] = False
    path = tmp_path / "tampered-program.json"
    path.write_text(json.dumps(tampered))
    with pytest.raises(V.NL1DeclarationError, match="both language-program arms"):
        V.validate_program(path, root=ROOT)
