"""Fail-closed tests for the cycle-free preregistration evidence builder."""

from __future__ import annotations

import ast
import gzip
import importlib.util
import json
import pathlib
import tarfile

import pytest


PROJECT = pathlib.Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_preregistration_evidence_under_test",
    PROJECT / "scripts/build_preregistration_evidence.py",
)
B = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(B)


def _line(index: int, condition: int) -> str:
    # Four arms of length five, with a unique source/goal namespace per item/condition.
    base = index * 200 + condition * 30
    source = base
    arms = [
        [base + 1 + arm * 5 + depth for depth in range(5)] for arm in range(4)
    ]
    goal = arms[0][-1]
    edges = []
    for arm in arms:
        edges.append((source, arm[0]))
        edges.extend(zip(arm[:-1], arm[1:]))
    body = "|".join(f"{left},{right}" for left, right in edges)
    answer = ",".join(str(value) for value in [source, *arms[0]])
    return f"{body}/{source},{goal}={answer}"


def _e_lure(path: pathlib.Path, n: int = 2_000) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for index in range(n):
            conditions = {}
            for offset, name in enumerate(B.CONDITIONS):
                line = _line(index, offset)
                conditions[name] = {
                    "line": line,
                    "prompt_sha256": B.sha256_bytes((line + "prompt").encode()),
                    "graph_key": B.sha256_bytes((line + "graph").encode()),
                    "answer": [index, offset],
                }
            stream.write(json.dumps({"quartet_id": index, "conditions": conditions}) + "\n")
    return path


def test_split_is_hash_sorted_exact_and_disjoint(tmp_path: pathlib.Path) -> None:
    source = _e_lure(tmp_path / "manifests/e_lure.jsonl")
    split, _five = B.build_split_documents(source)
    white = split["e_white"]
    score = split["e_score"]
    assert len(white) == 400 and len(score) == 1_600
    white_ids = {row["quartet_id"] for row in white}
    score_ids = {row["quartet_id"] for row in score}
    assert white_ids.isdisjoint(score_ids)
    assert white_ids | score_ids == set(range(2_000))
    hashes = [row["base_serialization_sha256"] for row in white + score]
    assert hashes == sorted(hashes)
    assert split["membership_rule"] == "ascending_sha256_of_canonical_base_serialization"


def test_five_condition_manifest_is_exact(tmp_path: pathlib.Path) -> None:
    source = _e_lure(tmp_path / "manifests/e_lure.jsonl")
    _split, five = B.build_split_documents(source)
    assert five["conditions"] == list(B.CONDITIONS)
    assert len(five["records"]) == 2_000
    assert all(set(record["conditions"]) == set(B.CONDITIONS) for record in five["records"])
    # A sixth robustness condition may exist upstream, but cannot silently enter this estimand.
    assert "near_safe_aligned" not in five["conditions"]


def test_pairwise_disjointness_mutation_blocks() -> None:
    clean = {"train": {"a", "b"}, "adapt": {"c"}, "eval": {"d", "e"}}
    assert all(value == 0 for value in B.require_pairwise_disjoint(clean).values())
    poisoned = {**clean, "eval": {"b", "e"}}
    with pytest.raises(B.EvidenceBlocked, match="not pairwise disjoint"):
        B.require_pairwise_disjoint(poisoned)


def test_create_only_refuses_mutated_frozen_bytes(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "frozen.json"
    B.create_only_json(target, {"value": 1})
    B.create_only_json(target, {"value": 1})
    with pytest.raises(B.EvidenceBlocked, match="refusing to replace"):
        B.create_only_json(target, {"value": 2})
    assert json.loads(target.read_text()) == {"value": 1}
    assert not target.with_name(target.name + ".partial").exists()


def _archive_current_tree(root: pathlib.Path, archive: pathlib.Path) -> None:
    archive.parent.mkdir(parents=True, exist_ok=True)

    def normalize(member: tarfile.TarInfo) -> tarfile.TarInfo:
        member.uid = member.gid = 0
        member.uname = member.gname = ""
        member.mtime = 0
        member.pax_headers = {}
        return member

    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for name in sorted(path.name for path in root.iterdir()):
                    if name.startswith(".") or name in B.SKIP_DIRS or name in B.SKIP_TOP_FILES:
                        continue
                    tar.add(root / name, arcname=name, filter=normalize)


def test_archive_freshness_refuses_source_mutation(tmp_path: pathlib.Path) -> None:
    (tmp_path / "scripts").mkdir()
    source = tmp_path / "scripts/tool.py"
    source.write_text("frozen = True\n")
    archive = tmp_path / B.ARCHIVE
    _archive_current_tree(tmp_path, archive)
    digest = B.assert_archive_fresh(tmp_path, archive)
    assert digest == B.sha256_file(archive)
    source.write_text("frozen = False\n")
    with pytest.raises(B.EvidenceBlocked, match="source archive is stale"):
        B.assert_archive_fresh(tmp_path, archive)


def test_inventory_mutation_is_refused(tmp_path: pathlib.Path) -> None:
    artifact = tmp_path / "manifests/value.json"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n")
    inventory = tmp_path / "inventory.sha256"
    inventory.write_text(f"{B.sha256_file(artifact)}  manifests/value.json\n")
    assert B._verify_hash_inventory(tmp_path, inventory) == 1
    artifact.write_text('{"mutated":true}\n')
    with pytest.raises(B.EvidenceBlocked, match="stale"):
        B._verify_hash_inventory(tmp_path, inventory)


def test_diagnose_is_read_only_even_when_everything_is_missing(tmp_path: pathlib.Path) -> None:
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    result = B.diagnose(tmp_path)
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert result["status"] == "BLOCK" and result["read_only"] is True
    assert before == after == []
    assert not (tmp_path / B.EVIDENCE).exists()


def test_prepare_never_publishes_evidence_or_pass_when_permanent_h3_block_missing(
    tmp_path: pathlib.Path,
) -> None:
    _e_lure(tmp_path / "manifests/e_lure.jsonl")
    with pytest.raises(B.EvidenceBlocked, match="permanent H3 block"):
        B.prepare(tmp_path)
    assert (tmp_path / B.PREPARED["split_receipt"]).is_file()
    assert (tmp_path / B.PREPARED["five_condition_manifest"]).is_file()
    assert not (tmp_path / B.PREPARED["disjointness_receipt"]).exists()
    assert not (tmp_path / B.EVIDENCE).exists()


def test_shipped_permanent_h3_block_is_exact_and_immutable() -> None:
    path, sidecar = B._permanent_h3_block(PROJECT)
    assert path == PROJECT / B.H3_BLOCK
    assert B.sha256_file(path) == B.V.H3_BLOCK_SHA256
    assert json.loads(path.read_text()) == B.V.H3_BLOCK_DOCUMENT
    assert sidecar.read_text() == f"{B.V.H3_BLOCK_SHA256}  {path.name}\n"


def test_gate10_semantic_sources_bind_exact_reduced_lurestar_schema_authorities() -> None:
    sources, additions = B._semantic_sources(PROJECT, "lurestar_schema_receipt", "a" * 64)
    assert sources == [
        PROJECT / "scripts/extract_lurestar_evidence.py",
        PROJECT / "scripts/evaluate_lurestar_checkpoints.py",
        PROJECT / "src/lurestar/evaluate.py",
        PROJECT / "src/lurestar/representations.py",
        PROJECT / "tests/test_lurestar_evidence_extractor.py",
        PROJECT / "tests/test_lurestar_checkpoint_evaluator.py",
        PROJECT / "tests/test_representations.py",
        PROJECT / "scripts/materialize_lurestar_evaluation.py",
        PROJECT / "tests/test_materialize_lurestar_evaluation.py",
        PROJECT / "scripts/evaluate_hmm_checkpoints.py",
        PROJECT / "scripts/run_hmm_matrix.py",
        PROJECT / "src/hmm_geometry/aggregate.py",
        PROJECT / "tests/test_run_hmm_matrix.py",
        PROJECT / "tests/test_hmm_family.py",
        PROJECT / B.H3_BLOCK,
        PROJECT / f"{B.H3_BLOCK}.sha256",
    ]
    assert additions["lurestar_schema_contract"] == B.V.LURESTAR_SCHEMA_CONTRACT
    assert additions["schemas"] == B.V.EXPECTED_CHECKS["10"]["schemas"]
    assert set(additions["semantic_witnesses"]) == set(
        B.V.LURESTAR_SEMANTIC_WITNESS_SPECS
    )
    assert all(additions[key] is True for key in (
        "missing_metrics_refused", "extra_metrics_refused", "invalid_cells_emitted",
        "nulls_emitted", "manipulation_failures_emitted", "h1_h2_metrics_preserved",
        "permanent_h3_exclusion_required", "h3_fields_refused",
        "adaptation_fields_refused", "mechanism_fields_refused",
    ))
    assert B._literal_schema_constants(sources[0], {
        "JOB_SCHEMA", "PROGRESS_SCHEMA", "EVIDENCE_SCHEMA",
    }) == {
        "JOB_SCHEMA": B.V.LURESTAR_SCHEMA_CONTRACT["extraction_job"],
        "PROGRESS_SCHEMA": B.V.LURESTAR_SCHEMA_CONTRACT["extraction_progress"],
        "EVIDENCE_SCHEMA": B.V.LURESTAR_SCHEMA_CONTRACT["evidence_npz"],
    }
    assert B._literal_schema_constants(sources[1], {
        "SCHEMA", "REPORT_SCHEMA", "RECEIPT_SCHEMA",
    }) == {
        "SCHEMA": B.V.LURESTAR_SCHEMA_CONTRACT["evaluation_manifest"],
        "REPORT_SCHEMA": B.V.LURESTAR_SCHEMA_CONTRACT["confirmatory_report"],
        "RECEIPT_SCHEMA": B.V.LURESTAR_SCHEMA_CONTRACT["evaluation_receipt"],
    }


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
def test_gate10_semantic_assertion_removal_blocks_witness_derivation(
        tmp_path: pathlib.Path, feature: str) -> None:
    for relative in {
        str(spec["node"]).split("::", 1)[0]
        for spec in B.V.LURESTAR_SEMANTIC_WITNESS_SPECS.values()
    } | {
        str(spec["source_literal"]["path"])
        for spec in B.V.LURESTAR_SEMANTIC_WITNESS_SPECS.values()
        if "source_literal" in spec
    }:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((PROJECT / relative).read_bytes())
    spec = B.V.LURESTAR_SEMANTIC_WITNESS_SPECS[feature]
    relative, function_name = str(spec["node"]).split("::", 1)
    path = tmp_path / relative
    token = str(spec["assertion_tokens"][-1])
    source = path.read_text()
    function = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    lines = source.splitlines(keepends=True)
    segment = "".join(lines[function.lineno - 1:function.end_lineno])
    assert token in segment
    lines[function.lineno - 1:function.end_lineno] = [
        segment.replace(token, "REMOVED_SEMANTIC_TOKEN")
    ]
    path.write_text("".join(lines))
    with pytest.raises(ValueError, match="lacks required semantics"):
        B.V.derive_lurestar_semantic_witnesses(tmp_path)


def test_gate10_schema_authority_requires_literal_complete_constants(tmp_path) -> None:
    path = tmp_path / "authority.py"
    path.write_text('JOB_SCHEMA = "job/2"\nPROGRESS_SCHEMA = dynamic_schema()\n')
    with pytest.raises(B.EvidenceBlocked, match="lacks literal constants"):
        B._literal_schema_constants(path, {"JOB_SCHEMA", "PROGRESS_SCHEMA"})


def test_gate10_lure_payload_is_derived_not_copied_from_expected_checks(monkeypatch) -> None:
    _sources, additions = B._semantic_sources(
        PROJECT, "lurestar_schema_receipt", "a" * 64
    )
    monkeypatch.setitem(B.V.EXPECTED_CHECKS["10"], "h1_h2_metrics_preserved", False)
    monkeypatch.setitem(B.V.EXPECTED_CHECKS["10"], "schemas", ["fabricated/schema"])
    payload = B._role_payload("10", "lurestar_schema_receipt", additions)
    assert payload["h1_h2_metrics_preserved"] is True
    assert payload["schemas"] != ["fabricated/schema"]
    assert payload["semantic_witnesses"] == additions["semantic_witnesses"]


def test_gate10_test_receipt_binds_exact_nodes_modules_and_witness_digest(
        tmp_path: pathlib.Path, monkeypatch) -> None:
    _sources, additions = B._semantic_sources(
        PROJECT, "lurestar_schema_receipt", "a" * 64
    )
    payload = B._role_payload("10", "lurestar_schema_receipt", additions)
    subject = {key: value for key, value in payload.items() if key != "claim"}
    monkeypatch.setattr(B.subprocess, "run", lambda *args, **kwargs: B.subprocess.CompletedProcess(
        args=args[0], returncode=0, stdout="27 passed in 1.00s\n",
    ))
    receipt_path = B.run_role_tests(
        PROJECT, "lurestar_schema_receipt", "a" * 64, tmp_path,
        semantic_subject=subject,
    )
    receipt = json.loads(receipt_path.read_text())
    assert receipt["pytest_nodes"] == list(B.V.LURESTAR_SEMANTIC_TEST_NODES)
    assert {pathlib.Path(item["path"]).relative_to(PROJECT).as_posix()
            for item in receipt["modules"]} == set(B.V.LURESTAR_SEMANTIC_MODULES)
    assert receipt["semantic_witnesses_sha256"] == B.canonical_sha256(
        additions["semantic_witnesses"]
    )


@pytest.mark.parametrize("mutation", ["content", "sidecar", "removal"])
def test_permanent_h3_block_mutation_or_removal_fails_closed(
    tmp_path: pathlib.Path, mutation: str,
) -> None:
    path = tmp_path / B.H3_BLOCK
    path.parent.mkdir(parents=True)
    path.write_bytes((PROJECT / B.H3_BLOCK).read_bytes())
    sidecar = path.with_name(path.name + ".sha256")
    sidecar.write_text(f"{B.V.H3_BLOCK_SHA256}  {path.name}\n")
    if mutation == "content":
        value = json.loads(path.read_text())
        value["unmatched_count"] = 3
        path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    elif mutation == "sidecar":
        sidecar.write_text(f"{'0' * 64}  {path.name}\n")
    else:
        path.unlink()
    with pytest.raises(B.EvidenceBlocked, match="permanent H3 block"):
        B._permanent_h3_block(tmp_path)


def test_prepared_split_mutation_is_detected_before_attestation(tmp_path: pathlib.Path) -> None:
    source = _e_lure(tmp_path / "manifests/e_lure.jsonl")
    split, five = B.build_split_documents(source)
    B.create_only_json(tmp_path / B.PREPARED["split_receipt"], split)
    B.create_only_json(tmp_path / B.PREPARED["five_condition_manifest"], five)
    mutated = json.loads((tmp_path / B.PREPARED["split_receipt"]).read_text())
    mutated["e_white"][0], mutated["e_score"][0] = mutated["e_score"][0], mutated["e_white"][0]
    (tmp_path / B.PREPARED["split_receipt"]).write_text(json.dumps(mutated) + "\n")
    with pytest.raises(B.EvidenceBlocked, match="differs from its deterministic"):
        B._semantic_sources(tmp_path, "split_receipt", "a" * 64)


def test_gate11_wrapper_payload_has_validator_exact_keys_and_no_subject() -> None:
    full = B._role_payload("11", "full_suite_receipt", {"exit_code": 0, "tests_passed": 900})
    review = B._role_payload("11", "independent_review_receipt", {"reviewer": "agent-x"})
    assert set(full) == {
        "claim", "full_suite_pass", "confirmatory_compute_launched", "exit_code",
        "tests_passed",
    }
    assert set(review) == {
        "claim", "unresolved_p0_scientific", "unresolved_p1_scientific",
        "independent_review_pass", "confirmatory_compute_launched", "reviewer",
    }
    assert "subject" not in full and "subject" not in review


def test_test_count_requires_an_actual_nonempty_pass() -> None:
    assert B._test_count(".. 2 passed in 0.01s\n") == 2
    assert B._test_count("1 failed, 2 passed in 0.01s\n") == 2
    assert B._test_count("collected 0 items\n") == 0
