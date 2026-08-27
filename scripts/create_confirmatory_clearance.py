#!/usr/bin/env python
"""Create source-bound test/review evidence and issue confirmatory GO clearance."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

import colab_train_loop as driver
import d41_continuation_gate as d41


def atomic_json(path: pathlib.Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    with open(partial, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_spec(root: pathlib.Path, path: str) -> tuple[pathlib.Path, dict]:
    spec_path = pathlib.Path(path)
    if not spec_path.is_absolute():
        spec_path = root / spec_path
    document = json.loads(spec_path.read_text())
    if not isinstance(document, dict):
        raise SystemExit("job spec must be a JSON object")
    return spec_path, document


def snapshot(root: pathlib.Path) -> tuple[pathlib.Path, str]:
    archive = pathlib.Path(driver.package(str(root)))
    return archive, driver.sha256_file(archive)


def record_tests(root: pathlib.Path, source_sha: str) -> pathlib.Path:
    command = [sys.executable, "-m", "pytest", "tests", "-q"]
    completed = subprocess.run(
        command, cwd=root, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False)
    sys.stdout.write(completed.stdout)
    matches = re.findall(r"(?:^|\s)(\d+) passed", completed.stdout)
    tests_passed = int(matches[-1]) if matches else 0
    receipt = {
        "schema": driver.CONFIRMATORY_TEST_SCHEMA,
        "recorded_at": utc_now(),
        "source_sha256": source_sha,
        "command": command,
        "exit_code": completed.returncode,
        "tests_passed": tests_passed,
        "outcome": "PASS" if completed.returncode == 0 and tests_passed > 0 else "FAIL",
        "output_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
    }
    path = root / ".agent_state" / "confirmatory-test-receipt.json"
    atomic_json(path, receipt)
    if receipt["outcome"] != "PASS":
        raise SystemExit("full confirmatory test suite did not pass")
    print("TEST_RECEIPT=%s" % path)
    return path


def record_review(root: pathlib.Path, source_sha: str, report_arg: str,
                  reviewer: str, verdict: str) -> pathlib.Path:
    report = pathlib.Path(report_arg)
    if not report.is_absolute():
        report = root / report
    report = report.resolve()
    try:
        relative = report.relative_to(root).as_posix()
    except ValueError:
        raise SystemExit("review report must be inside the project root")
    if not report.is_file():
        raise SystemExit("review report is missing")
    receipt = {
        "schema": driver.CONFIRMATORY_REVIEW_SCHEMA,
        "recorded_at": utc_now(),
        "source_sha256": source_sha,
        "reviewer": reviewer.strip(),
        "verdict": verdict,
        "report_path": relative,
        "report_sha256": driver.sha256_file(report),
    }
    if not receipt["reviewer"]:
        raise SystemExit("reviewer identity is required")
    path = root / ".agent_state" / "confirmatory-review-receipt.json"
    atomic_json(path, receipt)
    print("REVIEW_RECEIPT=%s" % path)
    return path


def issue(root: pathlib.Path, spec: dict, source_sha: str,
          preregistration_receipt: str | None = None) -> pathlib.Path:
    driver.validate_confirmatory_job_spec(spec)
    continuation = None
    if spec.get("predecessor_source_sha256") is not None:
        d43_receipt = root / driver.D43_RECEIPT_PATH
        if (d43_receipt.is_file() and
                spec.get("continuation_gate") != driver.D43_CONTINUATION_GATE):
            raise SystemExit("D43 continuation receipt exists; refusing fallback to D41")
        # D43 is an explicit continuation authority, never an inferred upgrade from D41.
        # Its driver helper dynamically imports the frozen D43 gate and recomputes the
        # receipt against the exact predecessor, D41 operational baseline, and current
        # source archives.  That recomputation also binds the exact-source test/review/
        # semantic evidence and the outcome-blind 10-complete/20-pending partition.
        if spec.get("continuation_gate") == driver.D43_CONTINUATION_GATE:
            continuation = driver.validate_d43_continuation_bundle(
                root, spec, source_sha)
        else:
            continuation = d41.validate_d41_continuation_bundle(root, spec, source_sha)
        reference = json.loads((root / d41.PREDECESSOR_REFERENCE_PATH).read_text())
        # Preserve the exact original all-eleven binding.  It is intentionally bound to the
        # predecessor, not relabelled as a fresh pre-compute freeze for the successor source.
        preregistration = reference["issued_clearance"]["preregistration"]
    else:
        preregistration = driver.validate_preregistration_pass_receipt(
            root, source_sha, preregistration_receipt)
    test_path = root / ".agent_state" / "confirmatory-test-receipt.json"
    review_path = root / ".agent_state" / "confirmatory-review-receipt.json"
    protocol_bindings = {}
    for relative in driver.CONFIRMATORY_PROTOCOL_PATHS:
        path = root / relative
        if not path.is_file():
            raise SystemExit("required protocol file is missing: %s" % relative)
        protocol_bindings[relative] = driver.sha256_file(path)
    clearance = {
        "schema": driver.CONFIRMATORY_CLEARANCE_SCHEMA,
        "created_at": utc_now(),
        "authorization": "GO",
        "source_sha256": source_sha,
        "job_spec_sha256": driver.canonical_json_sha256(spec),
        "input_bundle": driver.validate_input_bundle_receipt(root),
        "protocol_bindings": protocol_bindings,
        "test_receipt_sha256": driver.sha256_file(test_path),
        "review_receipt_sha256": driver.sha256_file(review_path),
        "preregistration": preregistration,
    }
    if continuation is not None:
        clearance["continuation"] = continuation
    output = root / ".agent_state" / "confirmatory-clearance.json"
    candidate = output.with_name(output.name + ".candidate")
    atomic_json(candidate, clearance)
    try:
        driver.validate_confirmatory_clearance(root, spec, source_sha, candidate)
    except BaseException:
        candidate.unlink(missing_ok=True)
        raise
    os.replace(candidate, output)
    driver.validate_confirmatory_clearance(root, spec, source_sha, output)
    print("CLEARANCE=%s" % output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("snapshot", "test", "review", "issue"))
    parser.add_argument("--project-root", default=str(pathlib.Path(__file__).resolve().parents[1]))
    parser.add_argument("--job-spec", default=".agent_state/job_spec.json")
    parser.add_argument("--report", default="docs/INDEPENDENT_CONFIRMATORY_REVIEW.md")
    parser.add_argument("--reviewer", default="")
    parser.add_argument("--verdict", choices=("PASS", "FAIL", "BLOCK"), default="BLOCK")
    parser.add_argument(
        "--preregistration-receipt",
        default=".agent_state/preregistration-freeze-receipt.json",
    )
    args = parser.parse_args()
    root = pathlib.Path(args.project_root).resolve()
    _, spec = load_spec(root, args.job_spec)
    archive, source_sha = snapshot(root)
    print("SOURCE_ARCHIVE=%s" % archive)
    print("SOURCE_SHA256=%s" % source_sha)
    if args.mode == "test":
        record_tests(root, source_sha)
    elif args.mode == "review":
        record_review(root, source_sha, args.report, args.reviewer, args.verdict)
    elif args.mode == "issue":
        issue(root, spec, source_sha, args.preregistration_receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
