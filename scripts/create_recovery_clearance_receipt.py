#!/usr/bin/env python3
"""Create a fail-closed, content-addressed recovery-clearance receipt.

This is a posthoc *engineering* audit of one already immutable recovery-gate result.
It never edits or replaces that result, and it cannot turn ``passed: false`` into a
schema-v2 pass.  Its narrower question is whether the durable restore/replay behavior
was demonstrated despite a known bounded-log observation failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import typing as t


SCHEMA = "nextlat_forgetting/recovery_clearance/1"
GATE_SCHEMA = "nextlat_forgetting/recovery_gate/2"
GATE_ID = "rg-d21e8fee468a-1787545664418856000-9899dbda"
RESULT_URI = (
    "gs://nextlat-lurestar-project-flash-490419/lurestar/recovery-gates/"
    + GATE_ID
    + "/result.json"
)
RESULT_SHA256 = "f9472aacb950437a17d984217d81030ad4b96509448c8d5bcdebd033783ce4e0"
RESULT_GENERATION = "1787546253389363"
SOURCE_SHA256 = "d21e8fee468a4c62f34a5384898142bc5bb4bc7a5565da6e0c567076be5ec9e9"
SOURCE_URI = (
    "gs://nextlat-lurestar-project-flash-490419/lurestar/source/project-"
    + SOURCE_SHA256
    + ".tar.gz"
)
SOURCE_GENERATION = "1787545701151305"
EXECUTED_HARNESS_SHA256 = "2000c1b2f8a7f9d14c6eebf817e3fc2d54c2ea523c0c68a9b81eeaa59756d8c1"
FINAL_CHECKPOINT_SHA256 = "9b7f2d2edec3d4045ce963a4deb0179ca7f6c662090eb7a35825ec1ca38e7c04"
SHARED_PARENT_SHA256 = "ec631e5a01f6e64a6564627a58ea18034f9f03b650bb6d7c56d6f8514b72f01b"
PREREGISTRATION_SHA256 = "0d78953bf26d07c41b3d58c1d68364ba00264ea2f977f8f28f0b121f4e47d8df"
EXACT_DOMAINS = (
    "weights",
    "optimizer",
    "scheduler",
    "rng",
    "amp_grad_scaler",
    "logits",
    "metrics",
)
EXPECTED_CHECKS = frozenset(
    ("final_step", "checkpoint_lineage", "data_position", "durable_progress", *EXACT_DOMAINS)
)


class ClearanceError(RuntimeError):
    """The immutable evidence does not satisfy the narrow clearance policy."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(document: dict[str, t.Any]) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _run_gcloud(*argv: str) -> bytes:
    completed = subprocess.run(
        ["gcloud", "storage", *argv], check=False, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode:
        raise ClearanceError(
            "gcloud storage failed: " + completed.stderr.decode("utf-8", "replace").strip()
        )
    return completed.stdout


def read_evidence(source: str) -> tuple[bytes, dict[str, t.Any]]:
    """Read bytes and immutable object identity without creating a local source copy."""
    if source.startswith("gs://"):
        payload = _run_gcloud("cat", source)
        metadata = json.loads(
            _run_gcloud("objects", "describe", source, "--format=json").decode("utf-8")
        )
        return payload, {
            "uri": source,
            "generation": str(metadata.get("generation")),
            "metageneration": int(metadata.get("metageneration", 0)),
            "size_bytes": int(metadata.get("size", -1)),
            "md5_base64": metadata.get("md5_hash"),
        }
    path = pathlib.Path(source)
    payload = path.read_bytes()
    return payload, {"path": str(path.resolve()), "size_bytes": len(payload)}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ClearanceError(message)


def _exact_zero_domain(name: str, check: t.Any) -> dict[str, t.Any]:
    _require(isinstance(check, dict), "%s check is missing" % name)
    _require(check.get("ok") is True, "%s check did not pass" % name)
    _require(check.get("mismatch") is None, "%s check records a mismatch" % name)
    for field in ("max_abs", "max_rel"):
        value = check.get(field)
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0,
            "%s %s is not exact zero" % (name, field),
        )
    return {"ok": True, "max_abs": 0.0, "max_rel": 0.0}


def audit_result(payload: bytes, identity: dict[str, t.Any]) -> dict[str, t.Any]:
    """Mechanically derive the narrow GO receipt; reject every other failure shape."""
    result_sha = _sha256(payload)
    _require(result_sha == RESULT_SHA256, "result bytes do not match frozen SHA-256")
    try:
        result = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ClearanceError("result is not valid JSON") from exc
    _require(isinstance(result, dict), "result JSON is not an object")
    _require(result.get("schema") == GATE_SCHEMA, "unexpected recovery-gate schema")
    _require(result.get("gate_id") == GATE_ID, "unexpected gate ID")
    _require(result.get("passed") is False, "original result must remain passed:false")
    _require(result.get("source_sha256") == SOURCE_SHA256, "source binding mismatch")
    _require(
        result.get("preregistration_sha256") == PREREGISTRATION_SHA256,
        "preregistration binding mismatch",
    )
    _require(result.get("confirmatory_data_used") is False, "confirmatory data was used")
    _require(result.get("confirmatory_seed_used") is False, "confirmatory seed was used")
    if "generation" in identity:
        _require(identity["generation"] == RESULT_GENERATION, "result generation mismatch")
        _require(identity.get("metageneration") == 1, "result metadata was modified")
        _require(identity.get("size_bytes") == len(payload), "result object size mismatch")

    reference = result.get("reference_checkpoint", {})
    recovered = result.get("recovered_checkpoint", {})
    for label, checkpoint in (("reference", reference), ("recovered", recovered)):
        _require(checkpoint.get("sha256") == FINAL_CHECKPOINT_SHA256,
                 "%s final checkpoint hash mismatch" % label)
        _require(checkpoint.get("size_bytes") == 256_007_203,
                 "%s final checkpoint size mismatch" % label)
    _require(reference == recovered or (
        reference.get("sha256") == recovered.get("sha256") and
        reference.get("size_bytes") == recovered.get("size_bytes")
    ), "final checkpoint bytes differ")
    _require(
        result.get("shared_lineage_checkpoint_sha256") == SHARED_PARENT_SHA256,
        "shared parent lineage mismatch",
    )

    checks = result.get("checks", {})
    _require(isinstance(checks, dict), "checks payload is missing")
    _require(
        set(checks) == EXPECTED_CHECKS,
        "unexpected check set; cannot prove the marker observation was the sole failure",
    )
    _require(checks.get("final_step") is True, "final step check failed")
    durable_progress = checks.get("durable_progress", {})
    _require(durable_progress.get("ok") is True, "durable progress check failed")
    _require(durable_progress.get("errors") == [], "durable progress recorded errors")
    _require(durable_progress.get("cadence_seconds") == 60,
             "durable progress cadence changed")
    exact = {name: _exact_zero_domain(name, checks.get(name)) for name in EXACT_DOMAINS}
    lineage = checks.get("checkpoint_lineage", {})
    _require(lineage.get("ok") is True, "checkpoint lineage check failed")
    _require(lineage.get("shared_parent_sha256") == SHARED_PARENT_SHA256,
             "checkpoint lineage parent mismatch")
    _require(lineage.get("recovered_final_sha256") == FINAL_CHECKPOINT_SHA256,
             "checkpoint lineage final mismatch")

    data_position = checks.get("data_position", {})
    _require(data_position.get("ok") is False, "expected original observation failure absent")
    _require(data_position.get("resume_fast_forward_step") == 150,
             "unexpected fast-forward step")
    _require(data_position.get("final_epoch") == 37, "unexpected final epoch")
    _require(data_position.get("final_cursor") == 4, "unexpected final cursor")
    _require(data_position.get("batches_per_epoch") == 8, "unexpected epoch length")
    _require(data_position.get("reference_fast_forward_observed") is False,
             "reference marker shape changed")
    _require(data_position.get("recovered_fast_forward_observed") is False,
             "recovered marker shape changed")

    for label in ("reference", "recovered"):
        durable = result.get("durable_%s_final" % label, {})
        _require(durable.get("gate_id") == GATE_ID, "%s durable gate mismatch" % label)
        _require(durable.get("source_sha256") == SOURCE_SHA256,
                 "%s durable source mismatch" % label)
        _require(durable.get("step") == 300, "%s durable final step mismatch" % label)
        _require(durable.get("checkpoint_sha256") == FINAL_CHECKPOINT_SHA256,
                 "%s durable checkpoint mismatch" % label)

    return {
        "schema": SCHEMA,
        "gate_id": GATE_ID,
        "disposition": {
            "engineering_recovery_equivalence": "GO",
            "repeat_t4_required_for_this_engineering_question": False,
            "original_schema_v2_gate_passed": False,
            "scientific_or_confirmatory_result": False,
        },
        "original_result": {
            **identity,
            "sha256": result_sha,
            "passed": False,
            "preservation_rule": "This receipt supplements and never replaces result.json.",
        },
        "source_proof": {
            "archive_uri": SOURCE_URI,
            "archive_generation": SOURCE_GENERATION,
            "archive_sha256": SOURCE_SHA256,
            "executed_recovery_harness_sha256": EXECUTED_HARNESS_SHA256,
            "executed_bounded_tail": "collections.deque(maxlen=300)",
        },
        "exact_replay_evidence": {
            "shared_step_150_parent_sha256": SHARED_PARENT_SHA256,
            "reference_step_300_sha256": FINAL_CHECKPOINT_SHA256,
            "recovered_step_300_sha256": FINAL_CHECKPOINT_SHA256,
            "final_checkpoint_byte_identical": True,
            "final_checkpoint_size_bytes": 256_007_203,
            "domains": exact,
            "durable_state_last_reference": True,
            "durable_state_last_recovered": True,
        },
        "sole_failed_original_check": {
            "name": "data_position.fast_forward_marker_observation",
            "reference_marker_observed": False,
            "recovered_marker_observed": False,
            "cause": (
                "Both resume markers were emitted near process start, then evicted from the "
                "executed harness's 300-line bounded deque before comparison."
            ),
            "classification": "observation-harness false negative",
        },
        "audit": {
            "verdict": "GO",
            "policy": "mechanical_fail_closed_recovery_clearance_v1",
            "rationale": (
                "The common-parent reference and GCS-restored arms produced byte-identical "
                "step-300 checkpoints, and every serialized-state, logits, and metrics domain "
                "was exactly equal (max_abs=max_rel=0). The only original false check was a "
                "missing human-readable log marker in both bounded tails."
            ),
        },
        "caveats_and_follow_up": {
            "future_latch_hardening_required": True,
            "current_5000_line_tail_is_not_a_structured_latch": True,
            "required_design": (
                "Persist a structured fast-forward event at observation time and compare that "
                "latched event, independent of any bounded diagnostic log tail."
            ),
            "no_tolerance_changed": True,
            "no_failed_scientific_result_reclassified": True,
        },
    }


def write_once(output_dir: pathlib.Path, receipt: dict[str, t.Any]) -> pathlib.Path:
    payload = _canonical(receipt)
    receipt_sha = _sha256(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / ("clearance-%s-%s.json" % (GATE_ID, receipt_sha))
    checksum = target.with_suffix(target.suffix + ".sha256")
    for path, body in (
        (target, payload),
        (checksum, ("%s  %s\n" % (receipt_sha, target.name)).encode("ascii")),
    ):
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        except FileExistsError:
            _require(path.read_bytes() == body, "refusing to replace existing receipt bytes")
            continue
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
    return target


def main(argv: t.Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", default=RESULT_URI)
    parser.add_argument(
        "--output-dir", type=pathlib.Path,
        default=pathlib.Path("results/recovery_clearance"),
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    payload, identity = read_evidence(args.result)
    receipt = audit_result(payload, identity)
    if args.check_only:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    else:
        print(write_once(args.output_dir, receipt))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ClearanceError as exc:
        print("RECOVERY_CLEARANCE_REFUSED: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
