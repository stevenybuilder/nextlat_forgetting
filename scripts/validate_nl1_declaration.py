#!/usr/bin/env python3
"""Validate the outcome-blind two-layer language declarations without data/results."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any, Mapping, Sequence


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCHEMA = "nextlat_forgetting/nl1_design_declaration/1"
TS1_SCHEMA = "nextlat_forgetting/ts1_design_declaration/1"
PROGRAM_SCHEMA = "nextlat_forgetting/language_program_amendment/1"
FROZEN_NL1_SHA256 = (
    "d6346996458ddf31df92a88b9df622a3f096af50c8331d55820eb87705a4b0fb"
)


class NL1DeclarationError(ValueError):
    pass


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise NL1DeclarationError(f"missing declaration field: {key}")
    return mapping[key]


def validate(path: pathlib.Path, *, root: pathlib.Path = ROOT) -> dict[str, Any]:
    declaration = json.loads(path.read_text(encoding="utf-8"))
    if declaration.get("schema") != SCHEMA:
        raise NL1DeclarationError("schema mismatch")
    if declaration.get("study_id") != "NL-1":
        raise NL1DeclarationError("study_id must be NL-1")
    if declaration.get("status") != "BLOCKED_PREPARATION_PROFILE_AND_BUDGET":
        raise NL1DeclarationError("NL-1 must remain blocked until a new receipt clears it")

    blind = _require(declaration, "outcome_blind")
    if blind != {
        "nl1_model_outcomes_exist": False,
        "model_outcomes_inspected_for_design": False,
        "training_losses_inspected_for_design": False,
        "existing_core_outcomes_may_select_nl1": False,
        "nl1_may_rescue_core_nulls": False,
    }:
        raise NL1DeclarationError("outcome-blind contract drift")

    dataset = _require(declaration, "dataset")
    expected_dataset = {
        "id": "HuggingFaceFW/fineweb-edu",
        "config": "sample-100BT",
        "revision": "v1.0.0",
        "tokenizer": "gpt2",
    }
    if any(dataset.get(key) != value for key, value in expected_dataset.items()):
        raise NL1DeclarationError("FineWeb-Edu identity/revision drift")
    if dataset.get("downloaded") is not False:
        raise NL1DeclarationError("declaration cannot claim unverified data download")
    if dataset.get("split_bucket_contract") != {
        "modulus": 23,
        "base_buckets": "0-19",
        "retention_bucket": 20,
        "adaptation_episode_0_bucket": 21,
        "adaptation_episode_1_bucket": 22,
        "streaming_fill": (
            "accept_whole_documents_until_each_minimum_token_quota_is_met"
        ),
    }:
        raise NL1DeclarationError("document split bucket contract drift")
    if dataset.get("token_quotas") != {
        "base": 2_000_000_000,
        "retention": 100_000_000,
        "adaptation_episode_0": 100_000_000,
        "adaptation_episode_1": 100_000_000,
    }:
        raise NL1DeclarationError("token quota drift")

    upstream = _require(declaration, "upstream")
    if upstream.get("commit") != "3770be6009cea2b3c455a9ce7f2ca88b504bb955":
        raise NL1DeclarationError("pinned upstream commit drift")
    for label in (
        "gpt_config",
        "nextlat_config",
        "fineweb_datamodule",
        "fineweb_pretokenizer",
    ):
        artifact = _require(upstream, label)
        local = root / str(_require(artifact, "path"))
        if not local.is_file():
            raise NL1DeclarationError(f"missing pinned {label}: {local}")
        if _sha256(local) != artifact.get("sha256"):
            raise NL1DeclarationError(f"pinned {label} hash mismatch")

    design = _require(declaration, "design")
    if design.get("models") != ["gpt", "nextlat"]:
        raise NL1DeclarationError("model roster drift")
    if design.get("seeds") != [3234, 3235, 3236]:
        raise NL1DeclarationError("seed roster drift")
    if design.get("episodes") != [0, 1]:
        raise NL1DeclarationError("episode roster drift")
    model = _require(design, "model")
    if model != {
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "block_size": 1024,
        "effective_batch_size": 512,
    }:
        raise NL1DeclarationError("model contract drift")
    base = _require(design, "base_training")
    expected_exposure = (
        int(base.get("optimizer_updates", -1))
        * int(model["effective_batch_size"])
        * int(model["block_size"])
    )
    if expected_exposure != base.get("token_exposures_per_run"):
        raise NL1DeclarationError("base token arithmetic mismatch")
    if base.get("optimizer_updates") != 3814:
        raise NL1DeclarationError("base update budget drift")
    if base.get("lr_schedule") != "wsd":
        raise NL1DeclarationError("base WSD schedule drift")
    adaptation = _require(design, "adaptation")
    expected_adaptation = (
        int(adaptation.get("optimizer_updates_per_branch", -1))
        * int(model["effective_batch_size"])
        * int(model["block_size"])
    )
    if expected_adaptation != adaptation.get("token_exposures_per_branch"):
        raise NL1DeclarationError("adaptation token arithmetic mismatch")
    if adaptation.get("nextlat_auxiliary_coefficients") != {
        "lambda_mse": 0.0,
        "lambda_kl": 0.0,
        "lambda_ce": 0.0,
    }:
        raise NL1DeclarationError("adaptation is not CE-only for NextLat")
    if adaptation.get("optimizer") != {
        "type": "adamw",
        "learning_rate": 0.0001,
        "weight_decay": 0.1,
        "beta1": 0.9,
        "beta2": 0.95,
        "grad_clip": 1.0,
        "schedule": "constant",
        "parent_optimizer_state_restored": False,
    }:
        raise NL1DeclarationError("adaptation optimizer contract drift")

    estimand = _require(declaration, "primary_estimand")
    if estimand.get("model_contrast") != "nextlat_minus_gpt":
        raise NL1DeclarationError("primary contrast drift")
    if estimand.get("causal_mediation_claim_permitted") is not False:
        raise NL1DeclarationError("causal mediation claim must be prohibited")
    if estimand.get("exact_predictive_state_claim_permitted") is not False:
        raise NL1DeclarationError("exact predictive-state claim must be prohibited")

    profile = _require(declaration, "profile")
    if profile.get("complete_profile_exists") is not False:
        raise NL1DeclarationError("profile completion requires a separate receipt")
    if profile.get("scientific_metrics_may_be_opened") is not False:
        raise NL1DeclarationError("profile cannot open scientific metrics")
    if profile.get("launch_decision_rule") != {
        "maximum_projected_wall_hours": 120,
        "maximum_incremental_gpu_rental_usd": 35,
        "explicit_user_approval_required": True,
        "if_either_cap_exceeded": "NO_LAUNCH_COMPUTE_CAP",
        "reduced_scientific_fallback_permitted": False,
    }:
        raise NL1DeclarationError("outcome-independent compute decision rule drift")
    blockers = declaration.get("launch_blockers")
    if not isinstance(blockers, list) or len(blockers) != 5:
        raise NL1DeclarationError("launch blockers are incomplete")

    return {
        "schema": "nextlat_forgetting/nl1_declaration_validation/1",
        "status": "PASS_BLOCKED_AS_DECLARED",
        "declaration_sha256": _sha256(path),
        "dataset_downloaded": False,
        "model_outcomes_opened": False,
        "base_runs": len(design["models"]) * len(design["seeds"]),
        "adaptation_branches": (
            len(design["models"]) * len(design["seeds"]) * len(design["episodes"])
        ),
        "base_token_exposures": (
            len(design["models"])
            * len(design["seeds"])
            * int(base["token_exposures_per_run"])
        ),
        "adaptation_token_exposures": (
            len(design["models"])
            * len(design["seeds"])
            * len(design["episodes"])
            * int(adaptation["token_exposures_per_branch"])
        ),
    }


def validate_ts1(path: pathlib.Path, *, root: pathlib.Path = ROOT) -> dict[str, Any]:
    """Validate the prospective TinyStories parity declaration."""
    declaration = json.loads(path.read_text(encoding="utf-8"))
    if declaration.get("schema") != TS1_SCHEMA:
        raise NL1DeclarationError("TS-1 schema mismatch")
    if declaration.get("study_id") != "TS-1":
        raise NL1DeclarationError("study_id must be TS-1")
    if declaration.get("status") != "BLOCKED_DATA_PROFILE_AND_BUDGET":
        raise NL1DeclarationError("TS-1 must remain blocked until a receipt clears it")

    if _require(declaration, "outcome_blind") != {
        "ts1_model_outcomes_exist": False,
        "model_outcomes_inspected_for_design": False,
        "training_losses_inspected_for_design": False,
        "existing_core_or_nl1_outcomes_may_select_ts1": False,
        "ts1_may_rescue_core_or_nl1_nulls": False,
    }:
        raise NL1DeclarationError("TS-1 outcome-blind contract drift")

    dataset = _require(declaration, "dataset")
    expected_dataset = {
        "id": "cyrilzhang/TinyStories2-ascii",
        "revision": "eec46cef1415d7fb803f866bf5cc77da39e961fc",
        "vocab_size": 1000,
    }
    if any(dataset.get(key) != value for key, value in expected_dataset.items()):
        raise NL1DeclarationError("TinyStories identity/revision drift")
    if dataset.get("downloaded") is not False:
        raise NL1DeclarationError("TS-1 cannot claim an unverified data download")
    if dataset.get("remote_metadata_only_at_declaration") is not True:
        raise NL1DeclarationError("TS-1 metadata-only provenance drift")

    upstream = _require(declaration, "upstream")
    if upstream.get("commit") != "3770be6009cea2b3c455a9ce7f2ca88b504bb955":
        raise NL1DeclarationError("TS-1 pinned upstream commit drift")
    for label in (
        "gpt_config",
        "nextlat_config",
        "gpt_probe_config",
        "nextlat_probe_config",
        "tinystories_datamodule",
        "tokenizer",
        "probe_trainer",
        "probe_model",
    ):
        artifact = _require(upstream, label)
        local = root / str(_require(artifact, "path"))
        if not local.is_file():
            raise NL1DeclarationError(f"missing pinned TS-1 {label}: {local}")
        if _sha256(local) != artifact.get("sha256"):
            raise NL1DeclarationError(f"pinned TS-1 {label} hash mismatch")

    design = _require(declaration, "design")
    if design.get("models") != ["gpt", "nextlat_d1"]:
        raise NL1DeclarationError("TS-1 model roster drift")
    if design.get("seeds") != [1234, 1235, 1236]:
        raise NL1DeclarationError("TS-1 seed roster drift")
    if design.get("scope") != (
        "focused_gpt_versus_nextlat_d1_not_full_v4_objective_roster"
    ):
        raise NL1DeclarationError("TS-1 focused scope drift")
    model = _require(design, "model")
    if model != {
        "n_layer": 8,
        "n_head": 8,
        "n_embd": 768,
        "vocab_size": 1000,
        "block_size": 256,
        "effective_batch_size": 256,
    }:
        raise NL1DeclarationError("TS-1 model contract drift")
    if design.get("nextlat_d1_overrides") != {
        "mtp_horizon": 1,
        "lambda_kl": 1.0,
        "lambda_mse": 1.0,
        "proj_factor": 1.3,
    }:
        raise NL1DeclarationError("TS-1 NextLat d=1 override drift")

    base = _require(design, "base_training")
    expected_base = (
        int(base.get("optimizer_updates", -1))
        * int(model["effective_batch_size"])
        * int(model["block_size"])
    )
    if base.get("optimizer_updates") != 100_000:
        raise NL1DeclarationError("TS-1 base update budget drift")
    if base.get("token_exposures_per_run") != expected_base:
        raise NL1DeclarationError("TS-1 base token arithmetic mismatch")
    if base.get("checkpoint_step") != 100_000:
        raise NL1DeclarationError("TS-1 final checkpoint drift")
    if base.get("best_checkpoint_selection_permitted") is not False:
        raise NL1DeclarationError("TS-1 checkpoint selection must be prohibited")

    probe = _require(design, "probe")
    if probe.get("transformer_frozen") is not True:
        raise NL1DeclarationError("TS-1 probes require a frozen transformer")
    if probe.get("offsets") != list(range(1, 21)):
        raise NL1DeclarationError("TS-1 probe offset roster drift")
    if probe.get("independent_linear_token_heads") != 20:
        raise NL1DeclarationError("TS-1 linear probe roster drift")
    expected_probe = (
        int(probe.get("optimizer_updates", -1))
        * int(probe.get("effective_batch_size", -1))
        * int(probe.get("block_size", -1))
    )
    if probe.get("optimizer_updates") != 20_000:
        raise NL1DeclarationError("TS-1 probe update budget drift")
    if probe.get("token_forwards_per_run") != expected_probe:
        raise NL1DeclarationError("TS-1 probe token arithmetic mismatch")
    if probe.get("best_checkpoint_selection_permitted") is not False:
        raise NL1DeclarationError("TS-1 probe checkpoint selection prohibited")

    estimand = _require(declaration, "primary_estimand")
    if estimand.get("long_horizon_offsets") != list(range(10, 21)):
        raise NL1DeclarationError("TS-1 primary long-horizon offsets drift")
    if estimand.get("direction") != "positive":
        raise NL1DeclarationError("TS-1 primary direction drift")
    if _require(declaration, "parity_criterion") != {
        "mean_long_horizon_contrast_positive": True,
        "minimum_positive_seed_mean_offsets_among_10_to_20": 8,
        "maximum_absolute_seed_mean_offset_1_contrast_nats": 0.02,
        "all_seeds_and_offsets_reported_even_on_failure": True,
    }:
        raise NL1DeclarationError("TS-1 parity criterion drift")

    compute = _require(declaration, "compute")
    n_runs = len(design["models"]) * len(design["seeds"])
    if compute.get("base_runs") != n_runs or compute.get("probe_runs") != n_runs:
        raise NL1DeclarationError("TS-1 run count drift")
    if compute.get("base_optimizer_updates") != n_runs * base["optimizer_updates"]:
        raise NL1DeclarationError("TS-1 aggregate base update arithmetic mismatch")
    if compute.get("base_token_exposures") != n_runs * base["token_exposures_per_run"]:
        raise NL1DeclarationError("TS-1 aggregate base token arithmetic mismatch")
    if compute.get("probe_optimizer_updates") != n_runs * probe["optimizer_updates"]:
        raise NL1DeclarationError("TS-1 aggregate probe update arithmetic mismatch")
    if compute.get("probe_frozen_backbone_token_forwards") != (
        n_runs * probe["token_forwards_per_run"]
    ):
        raise NL1DeclarationError("TS-1 aggregate probe token arithmetic mismatch")

    profile = _require(declaration, "profile")
    if profile.get("complete_profile_exists") is not False:
        raise NL1DeclarationError("TS-1 profile completion needs a separate receipt")
    if profile.get("scientific_metrics_may_be_opened") is not False:
        raise NL1DeclarationError("TS-1 profile cannot open scientific metrics")
    if profile.get("launch_decision_rule") != {
        "maximum_projected_two_gpu_wall_hours": 180,
        "maximum_incremental_gpu_rental_usd": 55,
        "explicit_user_approval_required": True,
        "if_either_cap_exceeded": "NO_LAUNCH_COMPUTE_CAP",
        "reduced_scientific_fallback_permitted": False,
    }:
        raise NL1DeclarationError("TS-1 compute decision rule drift")
    claim_limits = _require(declaration, "claim_limits")
    if any(value is not False for value in claim_limits.values()):
        raise NL1DeclarationError("TS-1 claim limitation drift")
    blockers = declaration.get("launch_blockers")
    if not isinstance(blockers, list) or len(blockers) != 5:
        raise NL1DeclarationError("TS-1 launch blockers are incomplete")

    return {
        "schema": "nextlat_forgetting/ts1_declaration_validation/1",
        "status": "PASS_BLOCKED_AS_DECLARED",
        "declaration_sha256": _sha256(path),
        "dataset_downloaded": False,
        "model_outcomes_opened": False,
        "base_runs": n_runs,
        "probe_runs": n_runs,
        "base_token_exposures": compute["base_token_exposures"],
        "probe_frozen_backbone_token_forwards": compute[
            "probe_frozen_backbone_token_forwards"
        ],
    }


def validate_program(path: pathlib.Path, *, root: pathlib.Path = ROOT) -> dict[str, Any]:
    """Validate the prospective two-layer language-program amendment."""
    amendment = json.loads(path.read_text(encoding="utf-8"))
    if amendment.get("schema") != PROGRAM_SCHEMA:
        raise NL1DeclarationError("language-program schema mismatch")
    if amendment.get("amendment_id") != "LP-A1":
        raise NL1DeclarationError("language-program amendment ID drift")
    if amendment.get("status") != "PROSPECTIVE_BLOCKED_PROFILES_AND_BUDGET_APPROVAL":
        raise NL1DeclarationError("language program must remain prospectively blocked")

    provenance = _require(amendment, "provenance")
    nl1_ref = _require(provenance, "original_fineweb_declaration")
    ts1_ref = _require(provenance, "new_tinystories_declaration")
    if nl1_ref.get("sha256") != FROZEN_NL1_SHA256:
        raise NL1DeclarationError("original FineWeb declaration provenance drift")
    if nl1_ref.get("must_remain_byte_identical") is not True:
        raise NL1DeclarationError("original FineWeb declaration must remain immutable")
    for label, reference in (("NL-1", nl1_ref), ("TS-1", ts1_ref)):
        local = root / str(_require(reference, "path"))
        if not local.is_file() or _sha256(local) != reference.get("sha256"):
            raise NL1DeclarationError(f"{label} declaration hash mismatch")
    nl1_result = validate(root / str(nl1_ref["path"]), root=root)
    ts1_result = validate_ts1(root / str(ts1_ref["path"]), root=root)

    for label, reference in _require(provenance, "protocol_documents").items():
        local = root / str(_require(reference, "path"))
        if not local.is_file() or _sha256(local) != reference.get("sha256"):
            raise NL1DeclarationError(f"program protocol hash mismatch: {label}")

    if _require(amendment, "outcome_blind") != {
        "language_program_outcomes_exist": False,
        "corpora_downloaded_for_amendment": False,
        "model_or_probe_outcomes_inspected": False,
        "existing_core_outcomes_may_select_an_arm": False,
        "one_arm_may_rescue_another_arm_or_core_null": False,
    }:
        raise NL1DeclarationError("language-program outcome-blind contract drift")

    program = _require(amendment, "program")
    arms = program.get("arms")
    if not isinstance(arms, list) or [arm.get("study_id") for arm in arms] != [
        "TS-1",
        "NL-1",
    ]:
        raise NL1DeclarationError("language-program arm roster drift")
    if any(arm.get("required") is not True for arm in arms):
        raise NL1DeclarationError("both language-program arms must be required")
    if arms[0].get("language_provenance") != (
        "synthetic_GPT_3_5_and_GPT_4_generated_stories"
    ):
        raise NL1DeclarationError("TinyStories provenance drift")
    if arms[1].get("same_corpus_family_as_v4") is not True:
        raise NL1DeclarationError("FineWeb-Edu v4 corpus-family link drift")
    if arms[1].get("matches_v4_1_3B_parameter_100B_token_scale") is not False:
        raise NL1DeclarationError("NL-1 must not claim matched v4 scale")
    if program.get("claim_rule") != {
        "both_arms_complete_for_full_program_claim": True,
        "if_either_arm_missing": (
            "REPORT_INCOMPLETE_AND_OMIT_FULL_LANGUAGE_EXTERNAL_VALIDITY_CLAIM"
        ),
        "positive_arm_may_compensate_for_missing_or_null_arm": False,
        "program_may_override_controlled_core_conclusion": False,
        "fineweb_interpretation": (
            "exploratory_triangulation_not_matched_scale_replication"
        ),
    }:
        raise NL1DeclarationError("language-program claim rule drift")

    launch = _require(amendment, "launch_rules")
    if launch != {
        "arms_authorized_independently": True,
        "scientific_metrics_may_influence_launch": False,
        "ts1": {
            "maximum_projected_two_gpu_wall_hours": 180,
            "maximum_incremental_gpu_rental_usd": 55,
            "explicit_user_approval_required": True,
            "reduced_scientific_fallback_permitted": False,
        },
        "nl1": {
            "maximum_projected_wall_hours": 120,
            "maximum_incremental_gpu_rental_usd": 35,
            "explicit_user_approval_required": True,
            "reduced_scientific_fallback_permitted": False,
        },
        "combined_cap_or_cross_arm_substitution_permitted": False,
        "if_arm_exceeds_its_cap": "NO_LAUNCH_COMPUTE_CAP",
    }:
        raise NL1DeclarationError("language-program independent launch rules drift")

    compute = _require(amendment, "compute_accounting")
    if compute.get("ts1", {}).get("base_token_exposures") != ts1_result[
        "base_token_exposures"
    ]:
        raise NL1DeclarationError("program TS-1 compute accounting drift")
    if compute.get("nl1", {}).get("base_token_exposures") != nl1_result[
        "base_token_exposures"
    ]:
        raise NL1DeclarationError("program NL-1 compute accounting drift")
    if compute.get("nl1", {}).get("adaptation_token_exposures") != nl1_result[
        "adaptation_token_exposures"
    ]:
        raise NL1DeclarationError("program NL-1 adaptation accounting drift")

    return {
        "schema": "nextlat_forgetting/language_program_validation/1",
        "status": "PASS_BOTH_ARMS_BLOCKED_AS_DECLARED",
        "amendment_sha256": _sha256(path),
        "frozen_nl1_sha256": nl1_result["declaration_sha256"],
        "ts1_sha256": ts1_result["declaration_sha256"],
        "required_arms": ["TS-1", "NL-1"],
        "corpora_downloaded": False,
        "model_outcomes_opened": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        type=pathlib.Path,
        nargs="?",
        default=ROOT / "manifests/nl1/design_declaration.json",
    )
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument(
        "--kind", choices=("nl1", "ts1", "program"), default="nl1"
    )
    args = parser.parse_args(argv)
    validator = {
        "nl1": validate,
        "ts1": validate_ts1,
        "program": validate_program,
    }[args.kind]
    result = validator(args.path.resolve(), root=args.root.resolve())
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (NL1DeclarationError, json.JSONDecodeError) as exc:
        raise SystemExit(f"BLOCK: {exc}")
