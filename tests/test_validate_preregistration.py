from __future__ import annotations

import ast
import copy
import importlib.util
import json
import pathlib

import pytest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "validate_preregistration", ROOT / "scripts/validate_preregistration.py"
)
V = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(V)


def _write(path: pathlib.Path, content: str | bytes) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def _seed_lurestar_semantic_fixture(root: pathlib.Path) -> None:
    functions_by_path: dict[str, list[str]] = {}
    for spec in V.LURESTAR_SEMANTIC_WITNESS_SPECS.values():
        relative, function_name = str(spec["node"]).split("::", 1)
        tokens = [*spec["assertion_tokens"], *spec.get("function_tokens", ())]
        assertion = " and ".join(repr(token) for token in tokens) or "True"
        functions_by_path.setdefault(relative, []).append(
            f"def {function_name}():\n    assert {assertion}\n"
        )
    for relative in V.LURESTAR_SEMANTIC_MODULES:
        body = "\n".join(functions_by_path.get(relative, ["# semantic fixture source\n"]))
        if relative == "scripts/evaluate_lurestar_checkpoints.py":
            body = f"MANIPULATION_FAILURES = {V.LURESTAR_MANIPULATION_FAILURE_CONTRACT!r}\n" + body
        _write(root / relative, body)


def _role_payload(gate: str, role: str, checks: dict) -> dict:
    payload = {"claim": role}
    payload.update({key: copy.deepcopy(checks[key]) for key in V.ROLE_CHECK_KEYS[role]})
    if role == "hmm_te_receipt":
        payload.update(
            te_certificates=copy.deepcopy(checks["te_certificates"]),
            rank_required=4,
            sigma_min_exclusive_threshold=0.05,
        )
    elif role == "full_suite_receipt":
        payload.update(exit_code=0, tests_passed=800)
    elif role == "independent_review_receipt":
        payload["reviewer"] = "independent-test-reviewer"
    return payload


def _artifact(
    tmp_path: pathlib.Path, gate: str, role: str, schema: str, *,
    checks: dict, source_snapshot: dict | None,
) -> dict:
    suffix = ".md" if schema.startswith("text/") else ".bin" if schema.startswith("binary/") else ".json"
    path = (tmp_path / ".agent_state" / "project.tar.gz" if role == "source_snapshot" else
            tmp_path / ".agent_state" / "artifacts" / f"g{gate}-{role}{suffix}")
    if schema.startswith("text/"):
        _write(path, f"# frozen {role}\n")
    elif schema.startswith("binary/"):
        _write(path, b"frozen-source-snapshot")
    else:
        assert source_snapshot is not None
        producer = _write(
            tmp_path / "producers" / f"g{gate}-{role}-producer.py",
            "# producer\n",
        )
        test = _write(
            tmp_path / ".agent_state" / "bindings" / f"g{gate}-{role}-test.json",
            json.dumps({
                "schema": V.TEST_EVIDENCE_SCHEMA,
                "status": "PASS",
                "role": role,
                "source_archive_sha256": source_snapshot["sha256"],
                "exit_code": 0,
                "tests_passed": 1,
            }, sort_keys=True) + "\n",
        )
        payload = _role_payload(gate, role, checks)
        source_bindings = [{
            "path": source_snapshot["path"], "sha256": source_snapshot["sha256"],
        }]
        if role == "lurestar_schema_receipt":
            payload["semantic_witnesses"] = V.derive_lurestar_semantic_witnesses(tmp_path)
            modules = [
                {"path": str((tmp_path / relative).resolve()),
                 "sha256": V.sha256_file(tmp_path / relative)}
                for relative in V.LURESTAR_SEMANTIC_MODULES
            ]
            test_document = json.loads(test.read_text())
            test_document.update({
                "pytest_nodes": list(V.LURESTAR_SEMANTIC_TEST_NODES),
                "modules": modules,
                "semantic_witnesses_sha256": V.canonical_json_sha256(
                    payload["semantic_witnesses"]
                ),
            })
            test.write_text(json.dumps(test_document, sort_keys=True) + "\n")
            source_bindings.extend(modules)
        if role in V.BOUND_RAW_ROLES:
            if role in V.H3_BLOCK_ROLES:
                subject = tmp_path / V.H3_BLOCK_PATH
                raw = copy.deepcopy(V.H3_BLOCK_DOCUMENT)
            else:
                subject = tmp_path / "manifests" / f"{role}.json"
            if role == "hmm_family_manifest":
                raw = {
                    "schema": schema,
                    "required_regimes": list(V.REGIMES),
                    "primary_regime": None,
                    "selection_blinding": {
                        "model_checkpoints_inspected": False,
                        "model_representations_inspected": False,
                        "model_outcomes_inspected": False,
                    },
                }
            elif role not in V.H3_BLOCK_ROLES:
                raw = {
                    "schema": schema,
                    "status": "complete",
                    "family_sha256": "a" * 64,
                    "inventory_sha256": "b" * 64,
                    "n_artifacts": 42,
                    "required_regimes": list(V.REGIMES),
                    "model_outcomes_inspected": False,
                }
            raw_body = json.dumps(
                raw, sort_keys=True,
                separators=(",", ":") if role in V.H3_BLOCK_ROLES else None,
            ) + "\n"
            _write(subject, raw_body)
            payload["subject"] = {
                "path": str(subject), "sha256": V.sha256_file(subject),
                "schema": V.H3_BLOCK_SCHEMA if role in V.H3_BLOCK_ROLES else schema,
            }
            source_bindings.append({
                "path": str(subject), "sha256": V.sha256_file(subject),
            })
        _write(path, json.dumps({
            "schema": schema,
            "attestation_schema": V.ATTESTATION_SCHEMA,
            "status": "PASS",
            "role": role,
            "source_archive_sha256": source_snapshot["sha256"],
            "payload_sha256": V.canonical_json_sha256(payload),
            "payload": payload,
            "producer": {"path": str(producer), "sha256": V.sha256_file(producer)},
            "source_bindings": source_bindings,
            "test_bindings": [{"path": str(test), "sha256": V.sha256_file(test)}],
        }, sort_keys=True) + "\n")
    return {"role": role, "path": str(path), "sha256": V.sha256_file(path), "schema": schema}


def complete_fixture(tmp_path: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path, dict]:
    _seed_lurestar_semantic_fixture(tmp_path)
    amendment = _write(tmp_path / "amendment.md", "# frozen amendment\n")
    spec = _write(tmp_path / "spec.md", "# authoritative spec\n")
    gates = {}
    source_snapshot = None
    for gate, role_schemas in V.ARTIFACT_SCHEMAS.items():
        checks = copy.deepcopy(V.EXPECTED_CHECKS[gate])
        if gate == "8":
            checks["te_certificates"] = {
                regime: {"rank_te": 4, "sigma_min_te": 0.051 + index * 0.01}
                for index, regime in enumerate(V.REGIMES)
            }
        artifacts = [
            _artifact(
                tmp_path, gate, role, schema, checks=checks,
                source_snapshot=source_snapshot,
            )
            for role, schema in role_schemas.items()
        ]
        if gate == "1":
            for record in artifacts:
                if record["role"] == "amendment":
                    record.update(path=str(amendment), sha256=V.sha256_file(amendment))
                elif record["role"] == "authoritative_spec":
                    record.update(path=str(spec), sha256=V.sha256_file(spec))
            source_snapshot = next(
                record for record in artifacts if record["role"] == "source_snapshot")
        gates[gate] = {
            "schema": f"nextlat_forgetting/preregistration_gate_{gate}/1",
            "artifacts": artifacts,
            "checks": checks,
        }
    evidence = tmp_path / ".agent_state" / "preregistration-evidence.json"
    payload = {"schema": V.SCHEMA, "gates": gates}
    evidence.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return amendment, spec, evidence, payload


def _write_payload(path: pathlib.Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_record(payload: dict, gate: str, role: str) -> dict:
    return next(item for item in payload["gates"][gate]["artifacts"] if item["role"] == role)


def _rewrite_artifact(record: dict, document: dict) -> None:
    path = pathlib.Path(record["path"])
    path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    record["sha256"] = V.sha256_file(path)


def _rehash_h3_subject_bindings(payload: dict, subject: pathlib.Path) -> None:
    """Keep every envelope internally truthful while testing semantic H3 refusal."""
    subject_sha = V.sha256_file(subject)
    for gate in ("4", "5", "6", "7"):
        record = payload["gates"][gate]["artifacts"][0]
        document = json.loads(pathlib.Path(record["path"]).read_text())
        document["payload"]["subject"]["sha256"] = subject_sha
        binding = next(
            item for item in document["source_bindings"]
            if pathlib.Path(item["path"]) == subject
        )
        binding["sha256"] = subject_sha
        document["payload_sha256"] = V.canonical_json_sha256(document["payload"])
        _rewrite_artifact(record, document)


def test_complete_all_eleven_fixture_passes_and_binds_authorities(tmp_path):
    amendment, spec, evidence, _payload = complete_fixture(tmp_path)
    result = V.validate(evidence, amendment=amendment, spec=spec)
    assert result["status"] == "PASS"
    assert result["all_eleven_gates_pass"] is True
    assert [gate["gate"] for gate in result["gates"]] == list(range(1, 12))
    assert all(gate["status"] == "PASS" and gate["issues"] == [] for gate in result["gates"])
    assert result["authority"]["amendment"]["sha256"] == V.sha256_file(amendment)
    assert result["authority"]["spec"]["sha256"] == V.sha256_file(spec)
    assert result["gates"][3]["status"] == "PASS"
    assert result["meaning"].endswith(
        "H3 prospectively dropped after the immutable D40 feasibility gate"
    )


def test_h1_h2_and_hmm_gate_contracts_remain_required() -> None:
    assert set(V.ARTIFACT_SCHEMAS["2"]) == {
        "split_receipt", "five_condition_manifest", "disjointness_receipt",
    }
    assert set(V.ARTIFACT_SCHEMAS["3"]) == {
        "whitener_fixture_receipt", "metric_fixture_receipt",
    }
    assert set(V.ARTIFACT_SCHEMAS["8"]) == {
        "hmm_family_manifest", "hmm_materialization_receipt", "hmm_te_receipt",
    }
    assert V.EXPECTED_CHECKS["2"]["e_white_count"] == 400
    assert V.EXPECTED_CHECKS["2"]["e_score_count"] == 1600
    assert V.EXPECTED_CHECKS["3"]["metrics"] == list(V.METRICS)
    assert V.EXPECTED_CHECKS["8"]["regimes"] == list(V.REGIMES)
    gate10 = V.EXPECTED_CHECKS["10"]
    assert gate10["lurestar_schema_contract"] == {
        "extraction_job": "nextlat_forgetting/lurestar_evidence_extraction_job/3",
        "extraction_progress": "nextlat_forgetting/lurestar_evidence_progress/1",
        "evidence_npz": "nextlat_forgetting/lurestar_evidence/4",
        "evidence_receipt": "nextlat_forgetting/lurestar_evidence/4",
        "evaluation_manifest": "nextlat_forgetting/lurestar_evaluation_manifest/4",
        "confirmatory_report": "nextlat_forgetting/lurestar_confirmatory_report/4",
        "evaluation_receipt": "nextlat_forgetting/lurestar_evaluation_receipt/4",
    }
    assert gate10["lurestar_confirmatory_scope"] == "base_only_h1_h2"
    assert all(gate10[key] is True for key in (
        "h1_h2_metrics_preserved", "permanent_h3_exclusion_required",
        "h3_fields_refused", "adaptation_fields_refused", "mechanism_fields_refused",
    ))


@pytest.mark.parametrize("key,bad_value", [
    ("lurestar_confirmatory_scope", "h1_h2_h3"),
    ("h1_h2_metrics_preserved", False),
    ("permanent_h3_exclusion_required", False),
    ("h3_fields_refused", False),
    ("adaptation_fields_refused", False),
    ("mechanism_fields_refused", False),
])
def test_gate10_rejects_rehashed_scope_or_exclusion_claims(tmp_path, key, bad_value):
    amendment, spec, evidence, payload = complete_fixture(tmp_path)
    gate = payload["gates"]["10"]
    gate["checks"][key] = bad_value
    record = _artifact_record(payload, "10", "lurestar_schema_receipt")
    document = json.loads(pathlib.Path(record["path"]).read_text())
    document["payload"][key] = bad_value
    document["payload_sha256"] = V.canonical_json_sha256(document["payload"])
    _rewrite_artifact(record, document)
    _write_payload(evidence, payload)

    result = V.validate(evidence, amendment=amendment, spec=spec)

    assert result["gates"][9]["status"] == "BLOCK"
    assert any(f"gate 10 check {key} mismatch" in issue for issue in result["gates"][9]["issues"])


@pytest.mark.parametrize("feature", [
    "npsi_formula_and_denominator",
    "paired_student_t_and_loso",
    "exact_sha_base_id_folds",
    "nested_h2_m0_delta_r2_identical_folds",
    "report_schema_and_required_statistics",
    "h1_four_state_classifier",
    "tampered_field_invalid_emission",
    "invalid_cells_terminal_schema",
    "non_equivalence_nulls_and_manipulation_failures",
    "terminal_required_fields_fail_closed",
    "binary_h2_secondary_ceiling_status",
    "all_12_hooks_parity_and_cleanup",
    "bst_forward_only_all_12_hooks",
    "whitener_exact_mahalanobis_parity",
    "whitener_heldout_claim_fail_closed",
    "atomic_lurestar_exact_15_dry_run",
    "atomic_lurestar_invalid_fifteenth_zero_invocations",
    "atomic_lurestar_stale_fifteenth_zero_invocations",
    "atomic_lurestar_exact_cell_set",
    "atomic_hmm_exact_30_acceptance",
    "atomic_hmm_exact_30_refusal",
    "hmm_fisher_z_exact_and_boundary_fail_closed",
    "hmm_two_sided_sign_flip_floor",
    "hmm_two_sided_mde_and_exact_family",
    "hmm_null_and_heterogeneity_report_only",
])
def test_gate10_rehashed_removal_of_required_semantic_assertion_still_blocks(
        tmp_path: pathlib.Path, feature: str) -> None:
    amendment, spec_path, evidence, payload = complete_fixture(tmp_path)
    witness_spec = V.LURESTAR_SEMANTIC_WITNESS_SPECS[feature]
    relative, function_name = str(witness_spec["node"]).split("::", 1)
    source_path = tmp_path / relative
    source = source_path.read_text()
    function = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    token = str(witness_spec["assertion_tokens"][-1])
    lines = source.splitlines(keepends=True)
    segment = "".join(lines[function.lineno - 1:function.end_lineno])
    assert token in segment
    lines[function.lineno - 1:function.end_lineno] = [
        segment.replace(token, "REMOVED_SEMANTIC_TOKEN")
    ]
    source_path.write_text("".join(lines))
    new_sha = V.sha256_file(source_path)

    record = _artifact_record(payload, "10", "lurestar_schema_receipt")
    envelope = json.loads(pathlib.Path(record["path"]).read_text())
    source_binding = next(
        item for item in envelope["source_bindings"]
        if pathlib.Path(item["path"]) == source_path
    )
    source_binding["sha256"] = new_sha
    test_path = pathlib.Path(envelope["test_bindings"][0]["path"])
    test_document = json.loads(test_path.read_text())
    module_binding = next(
        item for item in test_document["modules"]
        if pathlib.Path(item["path"]) == source_path
    )
    module_binding["sha256"] = new_sha
    test_path.write_text(json.dumps(test_document, sort_keys=True) + "\n")
    envelope["test_bindings"][0]["sha256"] = V.sha256_file(test_path)
    _rewrite_artifact(record, envelope)
    _write_payload(evidence, payload)

    result = V.validate(evidence, amendment=amendment, spec=spec_path)
    gate10 = result["gates"][9]
    assert gate10["status"] == "BLOCK"
    assert any("semantic witnesses are invalid" in issue for issue in gate10["issues"])


@pytest.mark.parametrize("mutation,message", [
    ("nodes", "complete frozen semantic test set"),
    ("modules", "semantic module bindings are incomplete"),
    ("witness_hash", "does not bind its semantic witnesses"),
])
def test_gate10_rehashed_test_receipt_cannot_omit_execution_bindings(
        tmp_path: pathlib.Path, mutation: str, message: str) -> None:
    amendment, spec_path, evidence, payload = complete_fixture(tmp_path)
    record = _artifact_record(payload, "10", "lurestar_schema_receipt")
    envelope = json.loads(pathlib.Path(record["path"]).read_text())
    test_path = pathlib.Path(envelope["test_bindings"][0]["path"])
    test_document = json.loads(test_path.read_text())
    if mutation == "nodes":
        test_document["pytest_nodes"].pop()
    elif mutation == "modules":
        test_document["modules"] = [
            item for item in test_document["modules"]
            if not item["path"].endswith("src/lurestar/evaluate.py")
        ]
    else:
        test_document["semantic_witnesses_sha256"] = "0" * 64
    test_path.write_text(json.dumps(test_document, sort_keys=True) + "\n")
    envelope["test_bindings"][0]["sha256"] = V.sha256_file(test_path)
    _rewrite_artifact(record, envelope)
    _write_payload(evidence, payload)

    result = V.validate(evidence, amendment=amendment, spec=spec_path)
    assert result["gates"][9]["status"] == "BLOCK"
    assert any(message in issue for issue in result["gates"][9]["issues"])


@pytest.mark.parametrize("relative", [
    "src/lurestar/representations.py",
    "scripts/materialize_lurestar_evaluation.py",
    "tests/test_materialize_lurestar_evaluation.py",
    "scripts/run_hmm_matrix.py",
    "tests/test_run_hmm_matrix.py",
    "src/hmm_geometry/aggregate.py",
    "tests/test_hmm_family.py",
])
def test_gate10_rehashed_receipt_cannot_omit_expanded_semantic_surface(
        tmp_path: pathlib.Path, relative: str) -> None:
    amendment, spec_path, evidence, payload = complete_fixture(tmp_path)
    record = _artifact_record(payload, "10", "lurestar_schema_receipt")
    envelope = json.loads(pathlib.Path(record["path"]).read_text())
    test_path = pathlib.Path(envelope["test_bindings"][0]["path"])
    test_document = json.loads(test_path.read_text())
    omitted = (tmp_path / relative).resolve()
    test_document["modules"] = [
        item for item in test_document["modules"]
        if pathlib.Path(item["path"]).resolve() != omitted
    ]
    test_path.write_text(json.dumps(test_document, sort_keys=True) + "\n")
    envelope["test_bindings"][0]["sha256"] = V.sha256_file(test_path)
    _rewrite_artifact(record, envelope)
    _write_payload(evidence, payload)

    result = V.validate(evidence, amendment=amendment, spec=spec_path)
    gate10 = result["gates"][9]
    assert gate10["status"] == "BLOCK"
    assert any("semantic module bindings are incomplete" in issue
               for issue in gate10["issues"])


def test_gate10_rehashed_manipulation_failure_literal_change_still_blocks(tmp_path) -> None:
    amendment, spec_path, evidence, payload = complete_fixture(tmp_path)
    source_path = tmp_path / "scripts/evaluate_lurestar_checkpoints.py"
    source_path.write_text(source_path.read_text().replace(
        "H3_PERMANENTLY_DROPPED_AFTER_D40_FEASIBILITY_GATE", "changed-after-freeze",
    ))
    new_sha = V.sha256_file(source_path)
    record = _artifact_record(payload, "10", "lurestar_schema_receipt")
    envelope = json.loads(pathlib.Path(record["path"]).read_text())
    next(item for item in envelope["source_bindings"]
         if pathlib.Path(item["path"]) == source_path)["sha256"] = new_sha
    test_path = pathlib.Path(envelope["test_bindings"][0]["path"])
    test_document = json.loads(test_path.read_text())
    next(item for item in test_document["modules"]
         if pathlib.Path(item["path"]) == source_path)["sha256"] = new_sha
    test_path.write_text(json.dumps(test_document, sort_keys=True) + "\n")
    envelope["test_bindings"][0]["sha256"] = V.sha256_file(test_path)
    _rewrite_artifact(record, envelope)
    _write_payload(evidence, payload)

    result = V.validate(evidence, amendment=amendment, spec=spec_path)
    assert result["gates"][9]["status"] == "BLOCK"
    assert any("semantic source literal MANIPULATION_FAILURES changed" in issue
               for issue in result["gates"][9]["issues"])


def test_rehashed_mutation_of_permanent_h3_block_still_fails_semantically(tmp_path):
    amendment, spec, evidence, payload = complete_fixture(tmp_path)
    subject = tmp_path / V.H3_BLOCK_PATH
    mutated = json.loads(subject.read_text())
    mutated["unmatched_count"] = 3
    subject.write_text(json.dumps(mutated, sort_keys=True, separators=(",", ":")) + "\n")
    _rehash_h3_subject_bindings(payload, subject)
    _write_payload(evidence, payload)

    result = V.validate(evidence, amendment=amendment, spec=spec)

    assert result["status"] == "BLOCK"
    assert all(result["gates"][index]["status"] == "BLOCK" for index in range(3, 7))
    assert all(any("raw permanent H3 block is mutated" in issue
                   for issue in result["gates"][index]["issues"])
               for index in range(3, 7))


def test_removing_permanent_h3_block_fails_all_h3_exclusion_gates(tmp_path):
    amendment, spec, evidence, _payload = complete_fixture(tmp_path)
    (tmp_path / V.H3_BLOCK_PATH).unlink()

    result = V.validate(evidence, amendment=amendment, spec=spec)

    assert result["status"] == "BLOCK"
    assert all(result["gates"][index]["status"] == "BLOCK" for index in range(3, 7))


def test_attempted_confirmatory_h3_inclusion_is_refused_even_with_rehashed_envelope(tmp_path):
    amendment, spec, evidence, payload = complete_fixture(tmp_path)
    gate = payload["gates"]["4"]
    gate["checks"]["confirmatory_h3_included"] = True
    record = gate["artifacts"][0]
    document = json.loads(pathlib.Path(record["path"]).read_text())
    document["payload"]["confirmatory_h3_included"] = True
    document["payload_sha256"] = V.canonical_json_sha256(document["payload"])
    _rewrite_artifact(record, document)
    _write_payload(evidence, payload)

    result = V.validate(evidence, amendment=amendment, spec=spec)

    assert result["status"] == "BLOCK"
    assert any("gate 4 check confirmatory_h3_included mismatch" in issue
               for issue in result["gates"][3]["issues"])


def test_cli_writes_atomic_pass_receipt_and_require_all_returns_success(tmp_path):
    amendment, spec, evidence, _payload = complete_fixture(tmp_path)
    output = tmp_path / "freeze.json"
    assert V.main([
        "--amendment", str(amendment), "--spec", str(spec),
        "--evidence", str(evidence), "--output", str(output), "--require-all",
    ]) == 0
    assert json.loads(output.read_text())["status"] == "PASS"
    assert not output.with_name(output.name + ".partial").exists()


def test_cli_default_evidence_path_is_archive_excluded(tmp_path, monkeypatch):
    amendment = _write(tmp_path / "amendment.md", "# amendment\n")
    spec = _write(tmp_path / "spec.md", "# spec\n")
    output = tmp_path / "receipt.json"
    captured = {}

    def fake_validate(evidence_path, *, amendment, spec):
        captured["evidence"] = pathlib.Path(evidence_path)
        return {"schema": V.RECEIPT_SCHEMA, "status": "BLOCK"}

    monkeypatch.setattr(V, "validate", fake_validate)
    assert V.main([
        "--amendment", str(amendment), "--spec", str(spec), "--output", str(output),
    ]) == 0
    assert captured["evidence"] == ROOT / ".agent_state" / "preregistration-evidence.json"


@pytest.mark.parametrize("gate", [str(index) for index in range(1, 12)])
def test_mutating_any_one_of_the_eleven_gate_checks_blocks(gate, tmp_path):
    amendment, spec, evidence, payload = complete_fixture(tmp_path)
    checks = payload["gates"][gate]["checks"]
    key = sorted(checks)[0]
    value = checks[key]
    if isinstance(value, bool):
        checks[key] = not value
    elif isinstance(value, int):
        checks[key] = value + 1
    elif isinstance(value, str):
        checks[key] = value + "-mutated"
    elif isinstance(value, list):
        checks[key] = list(reversed(value)) if len(value) > 1 else []
    elif isinstance(value, dict):
        # Gate 8's first sorted key can be te_certificates.
        first = sorted(value)[0]
        value[first]["rank_te"] = 3
    else:  # pragma: no cover - fixture contains no other check type
        raise AssertionError(type(value))
    _write_payload(evidence, payload)
    result = V.validate(evidence, amendment=amendment, spec=spec)
    assert result["status"] == "BLOCK"
    assert result["gates"][int(gate) - 1]["status"] == "BLOCK"


@pytest.mark.parametrize("operation", ["missing", "extra"])
def test_missing_or_extra_evidence_block_is_an_explicit_global_block(operation, tmp_path):
    amendment, spec, evidence, payload = complete_fixture(tmp_path)
    if operation == "missing":
        del payload["gates"]["6"]
    else:
        payload["gates"]["12"] = copy.deepcopy(payload["gates"]["11"])
    _write_payload(evidence, payload)
    result = V.validate(evidence, amendment=amendment, spec=spec)
    assert result["status"] == "BLOCK"
    assert result[f"{operation}_gate_blocks"]
    assert any("evidence blocks mismatch" in issue for issue in result["global_issues"])


def test_explicit_evidence_path_outside_archive_excluded_state_is_refused(tmp_path):
    amendment, spec, evidence, _payload = complete_fixture(tmp_path)
    relocated = tmp_path / "manifests" / "preregistration-evidence.json"
    _write(relocated, evidence.read_text())

    result = V.validate(relocated, amendment=amendment, spec=spec)

    assert result["status"] == "BLOCK"
    assert any("archive-excluded" in issue for issue in result["global_issues"])


def test_extra_or_missing_fields_inside_gate_are_refused(tmp_path):
    amendment, spec, evidence, payload = complete_fixture(tmp_path)
    payload["gates"]["7"]["unexpected"] = True
    payload["gates"]["10"]["checks"].pop("nulls_emitted")
    _write_payload(evidence, payload)
    result = V.validate(evidence, amendment=amendment, spec=spec)
    assert result["gates"][6]["status"] == "BLOCK"
    assert result["gates"][9]["status"] == "BLOCK"
    assert "extra=['unexpected']" in result["gates"][6]["issues"][0]


def test_tampered_artifact_and_wrong_embedded_schema_both_block(tmp_path):
    amendment, spec, evidence, payload = complete_fixture(tmp_path)
    tampered = pathlib.Path(payload["gates"]["6"]["artifacts"][0]["path"])
    tampered.write_text('{"schema":"wrong"}\n')
    wrong_schema = pathlib.Path(payload["gates"]["9"]["artifacts"][0]["path"])
    wrong_schema.write_text(json.dumps({
        "schema": payload["gates"]["9"]["artifacts"][0]["schema"] + "-wrong"
    }))
    payload["gates"]["9"]["artifacts"][0]["sha256"] = V.sha256_file(wrong_schema)
    _write_payload(evidence, payload)
    result = V.validate(evidence, amendment=amendment, spec=spec)
    assert any("SHA-256 mismatch" in issue for issue in result["gates"][5]["issues"])
    assert any("envelope keys mismatch" in issue for issue in result["gates"][8]["issues"])


@pytest.mark.parametrize("mutation,message", [
    ("status", "status is not PASS"),
    ("source", "source archive binding mismatch"),
    ("payload_hash", "canonical payload SHA-256 mismatch"),
    ("placeholder", "payload keys mismatch"),
    ("producer", "producer SHA-256 mismatch"),
    ("source_binding", "source_bindings must be a nonempty list"),
    ("test_binding", "test_bindings[0] SHA-256 mismatch"),
])
def test_attestation_envelope_mutations_fail_closed(tmp_path, mutation, message):
    amendment, spec, evidence, payload = complete_fixture(tmp_path)
    record = _artifact_record(payload, "6", "h3_mechanism_exclusion_receipt")
    document = json.loads(pathlib.Path(record["path"]).read_text())
    if mutation == "status":
        document["status"] = "BLOCK"
    elif mutation == "source":
        document["source_archive_sha256"] = "0" * 64
    elif mutation == "payload_hash":
        document["payload_sha256"] = "0" * 64
    elif mutation == "placeholder":
        document["payload"] = {}
        document["payload_sha256"] = V.canonical_json_sha256({})
    elif mutation == "producer":
        document["producer"]["sha256"] = "0" * 64
    elif mutation == "source_binding":
        document["source_bindings"] = []
    else:
        document["test_bindings"][0]["sha256"] = "0" * 64
    _rewrite_artifact(record, document)
    _write_payload(evidence, payload)

    result = V.validate(evidence, amendment=amendment, spec=spec)

    assert result["status"] == "BLOCK"
    assert any(message in issue for issue in result["gates"][5]["issues"])


@pytest.mark.parametrize("gate,role,mutate,message", [
    ("4", "h3_permanent_block_receipt", lambda p: p.update(confirmatory_h3_included=True),
     "contradicts gate check confirmatory_h3_included"),
    ("6", "h3_mechanism_exclusion_receipt",
     lambda p: p.update(confirmatory_h3_mechanism_probes_included=True),
     "contradicts gate check confirmatory_h3_mechanism_probes_included"),
    ("8", "hmm_te_receipt",
     lambda p: p["te_certificates"][V.REGIMES[0]].update(rank_te=3),
     "TE certificate contract mismatch"),
    ("11", "full_suite_receipt", lambda p: p.update(tests_passed=0),
     "nonempty successful full suite"),
    ("11", "independent_review_receipt", lambda p: p.update(reviewer=""),
     "reviewer identity is missing"),
])
def test_representative_role_specific_lies_are_refused(
        tmp_path, gate, role, mutate, message):
    amendment, spec, evidence, payload = complete_fixture(tmp_path)
    record = _artifact_record(payload, gate, role)
    document = json.loads(pathlib.Path(record["path"]).read_text())
    mutate(document["payload"])
    document["payload_sha256"] = V.canonical_json_sha256(document["payload"])
    _rewrite_artifact(record, document)
    _write_payload(evidence, payload)

    result = V.validate(evidence, amendment=amendment, spec=spec)

    assert result["status"] == "BLOCK"
    assert any(message in issue for issue in result["gates"][int(gate) - 1]["issues"])


@pytest.mark.parametrize("role,mutation,message", [
    ("hmm_family_manifest", "unblind", "raw family manifest is incomplete or unblinded"),
    ("hmm_materialization_receipt", "status", "raw materialization receipt is incomplete"),
])
def test_wrapped_raw_hmm_authorities_are_semantically_validated(
        tmp_path, role, mutation, message):
    amendment, spec, evidence, payload = complete_fixture(tmp_path)
    record = _artifact_record(payload, "8", role)
    attestation = json.loads(pathlib.Path(record["path"]).read_text())
    subject_record = attestation["payload"]["subject"]
    subject_path = pathlib.Path(subject_record["path"])
    subject = json.loads(subject_path.read_text())
    if mutation == "unblind":
        subject["selection_blinding"]["model_outcomes_inspected"] = True
    else:
        subject["status"] = "PASS"
    subject_path.write_text(json.dumps(subject, sort_keys=True) + "\n")
    subject_sha = V.sha256_file(subject_path)
    subject_record["sha256"] = subject_sha
    matching_binding = next(
        item for item in attestation["source_bindings"]
        if pathlib.Path(item["path"]) == subject_path
    )
    matching_binding["sha256"] = subject_sha
    attestation["payload_sha256"] = V.canonical_json_sha256(attestation["payload"])
    _rewrite_artifact(record, attestation)
    _write_payload(evidence, payload)

    result = V.validate(evidence, amendment=amendment, spec=spec)

    assert result["status"] == "BLOCK"
    assert any(message in issue for issue in result["gates"][7]["issues"])


@pytest.mark.parametrize("gate,role", [
    (gate, role)
    for gate, roles in V.ARTIFACT_SCHEMAS.items() if gate != "1"
    for role in roles
])
def test_schema_only_json_can_never_satisfy_an_evidence_role(tmp_path, gate, role):
    amendment, spec, evidence, payload = complete_fixture(tmp_path)
    record = _artifact_record(payload, gate, role)
    _rewrite_artifact(record, {"schema": record["schema"], "status": "PASS"})
    _write_payload(evidence, payload)

    result = V.validate(evidence, amendment=amendment, spec=spec)

    assert result["status"] == "BLOCK"
    assert result["gates"][int(gate) - 1]["status"] == "BLOCK"
    assert any("envelope keys mismatch" in issue
               for issue in result["gates"][int(gate) - 1]["issues"])


@pytest.mark.parametrize("rank,sigma", [(3, 0.2), (4, 0.05), (4, float("nan"))])
def test_hmm_te_certificate_gate_is_strict(rank, sigma, tmp_path):
    amendment, spec, evidence, payload = complete_fixture(tmp_path)
    cert = payload["gates"]["8"]["checks"]["te_certificates"][V.REGIMES[0]]
    cert.update(rank_te=rank, sigma_min_te=sigma)
    # allow_nan is deliberately impossible in production, but write it here to exercise the
    # validator's finite check through Python's permissive decoder.
    evidence.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
    result = V.validate(evidence, amendment=amendment, spec=spec)
    assert result["gates"][7]["status"] == "BLOCK"
    assert any("sigma_min>0.05" in issue for issue in result["gates"][7]["issues"])


def test_gate_one_cannot_bind_a_different_amendment_path(tmp_path):
    amendment, spec, evidence, payload = complete_fixture(tmp_path)
    other = _write(tmp_path / "other.md", amendment.read_text())
    record = next(
        item for item in payload["gates"]["1"]["artifacts"] if item["role"] == "amendment"
    )
    record.update(path=str(other), sha256=V.sha256_file(other))
    _write_payload(evidence, payload)
    result = V.validate(evidence, amendment=amendment, spec=spec)
    assert result["gates"][0]["status"] == "BLOCK"
    assert any("differs from CLI authority" in issue for issue in result["gates"][0]["issues"])


def test_require_all_block_still_writes_diagnostic_receipt(tmp_path):
    amendment, spec, evidence, payload = complete_fixture(tmp_path)
    payload["gates"].pop("11")
    _write_payload(evidence, payload)
    output = tmp_path / "blocked.json"
    assert V.main([
        "--amendment", str(amendment), "--spec", str(spec),
        "--evidence", str(evidence), "--output", str(output), "--require-all",
    ]) == 2
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "BLOCK"
    assert receipt["missing_gate_blocks"] == ["11"]
