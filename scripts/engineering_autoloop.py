#!/usr/bin/env python
"""Auditable, engineering-only iteration loop for the frozen confirmatory study.

This deliberately does not implement an objective-metric hill climber.  One candidate is declared
before it is edited, its time and check budgets are fixed at declaration, and acceptance can depend
only on a closed registry of engineering checks.  Scientific surfaces are hashed at the boundary;
any movement blocks acceptance.  Rejected candidates are never reset or reverted in this shared
worktree.

Typical use::

    .venv/bin/python scripts/engineering_autoloop.py begin \
      --candidate-id faster-token-cache --candidate-path src/lurestar/fast_stargraph.py \
      --time-budget-seconds 1800 --max-checks 2 --check throughput --check unit
    # edit exactly that candidate
    .venv/bin/python scripts/engineering_autoloop.py evaluate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import time
import typing as t


SCHEMA = "nextlat_forgetting/engineering_autoloop/1"
DEFAULT_LOG = pathlib.Path("results/engineering_loop.jsonl")

# These are re-hashed before and after every candidate.  Tests are included so a candidate cannot
# weaken its own acceptance gate.  The config generator is included because changing it together
# with generated YAML would make `materialize_configs.py --check` self-consistently green.
FROZEN_EXACT = (
    "PROGRAM.md",
    "nextlat_v4_predictive_geometry_spec.md",
    "scripts/engineering_autoloop.py",
    "scripts/materialize_configs.py",
    "scripts/config_lib.py",
    "src/lurestar/generate.py",
    "src/lurestar/validate.py",
    "src/lurestar/representations.py",
    "src/lurestar/evaluate.py",
    "src/hmm_geometry/generate.py",
    "src/hmm_geometry/forward.py",
    "src/hmm_geometry/pair_bank.py",
    "src/hmm_geometry/evaluate.py",
)
FROZEN_DIRS = ("configs", "manifests", "tests")

# A candidate is never allowed to own an outcome-bearing path.  The harness also never opens these
# files.  Its own append-only log is written by the harness, not declared as candidate work.
PROHIBITED_CANDIDATE_PREFIXES = (
    "report",
    "results",
)

# Closed, named checks.  There is intentionally no arbitrary `--command`: it would let a candidate
# select itself on PSI or any other outcome by hiding that read in a shell string.
CHECKS: dict[str, tuple[str, ...]] = {
    "unit": ("{python}", "-m", "pytest", "tests/", "-q"),
    "config": ("{python}", "scripts/materialize_configs.py", "--check"),
    "durability": (
        "{python}", "-m", "pytest", "tests/test_resume.py", "tests/test_run_matrix.py", "-q",
    ),
    "packaging": ("{python}", "-m", "pytest", "tests/test_colab_train_loop.py", "-q"),
    "throughput": (
        "{python}", "-m", "pytest", "tests/test_fast_stargraph.py",
        "tests/test_profile_tooling.py", "-q",
    ),
}
FORBIDDEN_CHECK_TERMS = (
    "live_numbers", "run_ledger", "metrics.jsonl", "report/blog", "psi", "h1_predictive",
    "h2_geometry", "h3_interference",
)


class LoopError(RuntimeError):
    """A candidate boundary or audit invariant was violated."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _aggregate_digest(entries: t.Mapping[str, str | None]) -> str:
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return _sha256_bytes(payload)


def _file_digest(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _snapshot_paths(root: pathlib.Path, rel_paths: t.Iterable[str]) -> dict[str, str | None]:
    """Hash declared paths, recording missing files and directory membership."""
    out: dict[str, str | None] = {}
    for raw in sorted(set(rel_paths)):
        rel = pathlib.PurePosixPath(raw).as_posix().rstrip("/")
        path = root / rel
        if not path.exists():
            out[rel] = None
            continue
        if path.is_symlink():
            out[rel] = "symlink:" + os.readlink(path)
            continue
        if path.is_file():
            out[rel] = _file_digest(path)
            continue
        out[rel + "/"] = "directory"
        for child in sorted(p for p in path.rglob("*") if p.is_file() or p.is_symlink()):
            child_rel = child.relative_to(root).as_posix()
            out[child_rel] = (
                "symlink:" + os.readlink(child) if child.is_symlink() else _file_digest(child)
            )
    return out


def snapshot_frozen(root: pathlib.Path) -> dict[str, t.Any]:
    entries = _snapshot_paths(root, (*FROZEN_EXACT, *FROZEN_DIRS))
    return {"sha256": _aggregate_digest(entries), "entries": entries}


def _normalise_candidate_paths(root: pathlib.Path, paths: t.Iterable[str]) -> tuple[str, ...]:
    normalised: list[str] = []
    for raw in paths:
        candidate = pathlib.Path(raw)
        if candidate.is_absolute():
            try:
                rel = candidate.resolve(strict=False).relative_to(root.resolve())
            except ValueError as exc:
                raise LoopError(f"candidate path is outside the repository: {raw}") from exc
        else:
            rel = pathlib.Path(os.path.normpath(raw))
        rel_s = rel.as_posix()
        while rel_s.startswith("./"):
            rel_s = rel_s[2:]
        if not rel_s or rel_s in (".", "..") or rel_s.startswith("../"):
            raise LoopError(f"candidate path is not a scoped repository path: {raw}")
        if is_frozen_candidate_path(rel_s):
            raise LoopError(f"candidate path is scientifically frozen for this iteration: {rel_s}")
        if any(rel_s == p or rel_s.startswith(p.rstrip("/") + "/")
               for p in PROHIBITED_CANDIDATE_PREFIXES):
            raise LoopError(f"candidate path is outcome-bearing and prohibited: {rel_s}")
        normalised.append(rel_s)
    if not normalised:
        raise LoopError("at least one --candidate-path is required")
    return tuple(sorted(set(normalised)))


def is_frozen_candidate_path(rel: str) -> bool:
    rel = pathlib.PurePosixPath(rel).as_posix().rstrip("/")
    if rel in FROZEN_EXACT:
        return True
    return any(rel == d or rel.startswith(d.rstrip("/") + "/") for d in FROZEN_DIRS)


def validate_check_registry(registry: t.Mapping[str, t.Sequence[str]] = CHECKS) -> None:
    for name, command in registry.items():
        joined = " ".join(command).lower()
        forbidden = [term for term in FORBIDDEN_CHECK_TERMS if term in joined]
        if forbidden:
            raise LoopError(f"engineering check {name!r} reads prohibited outcomes: {forbidden}")
        if any(token in joined for token in ("git reset", "git checkout --", "git restore")):
            raise LoopError(f"engineering check {name!r} contains a destructive worktree command")


def _git_head(root: pathlib.Path) -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def _default_active_path(root: pathlib.Path) -> pathlib.Path:
    git_dir = root / ".git"
    if git_dir.is_dir():
        return git_dir / "engineering_autoloop_active.json"
    return root / ".agent_state" / "engineering_autoloop_active.json"


def _atomic_create(path: pathlib.Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise LoopError(f"another engineering candidate is already active: {path}") from exc
    try:
        payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def _append_record(path: pathlib.Path, record: dict) -> None:
    """Append exactly one fsynced JSON line; existing bytes are never rewritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def _load_active(path: pathlib.Path) -> dict:
    if not path.is_file():
        raise LoopError(f"no active engineering candidate at {path}")
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise LoopError(f"active candidate record is unreadable: {path}") from exc
    if doc.get("schema") != SCHEMA:
        raise LoopError(f"unsupported active candidate schema: {doc.get('schema')!r}")
    return doc


def begin_candidate(
    root: pathlib.Path,
    *,
    candidate_id: str,
    candidate_paths: t.Iterable[str],
    time_budget_seconds: float,
    max_checks: int,
    checks: t.Sequence[str],
    active_path: pathlib.Path,
    log_path: pathlib.Path,
    now: float | None = None,
    registry: t.Mapping[str, t.Sequence[str]] = CHECKS,
) -> dict:
    """Freeze the boundary and atomically claim the one active candidate slot."""
    validate_check_registry(registry)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", candidate_id):
        raise LoopError("candidate id must contain only letters, digits, dot, underscore, or dash")
    if time_budget_seconds <= 0 or max_checks <= 0:
        raise LoopError("time and check budgets must be positive")
    if not checks:
        raise LoopError("at least one engineering check is required")
    if len(checks) > max_checks:
        raise LoopError(f"selected {len(checks)} checks exceeds fixed max_checks={max_checks}")
    unknown = [name for name in checks if name not in registry]
    if unknown:
        raise LoopError(f"unknown engineering checks: {unknown}; choices={sorted(registry)}")

    root = root.resolve()
    declared = _normalise_candidate_paths(root, candidate_paths)
    started = time.time() if now is None else float(now)
    active = {
        "schema": SCHEMA,
        "candidate_id": candidate_id,
        "root": str(root),
        "candidate_paths": list(declared),
        "candidate_before": _snapshot_paths(root, declared),
        "frozen_before": snapshot_frozen(root),
        "git_head": _git_head(root),
        "started_at": started,
        "deadline": started + float(time_budget_seconds),
        "time_budget_seconds": float(time_budget_seconds),
        "max_checks": int(max_checks),
        "checks": list(checks),
        "check_commands": {name: _render_command(root, registry[name]) for name in checks},
        "log_path": str(log_path.resolve()),
    }
    _atomic_create(active_path, active)
    try:
        _append_record(log_path, {
            "schema": SCHEMA,
            "event": "BEGIN",
            "candidate_id": candidate_id,
            "started_at": started,
            "deadline": active["deadline"],
            "time_budget_seconds": active["time_budget_seconds"],
            "max_checks": active["max_checks"],
            "checks": active["checks"],
            "check_commands": active["check_commands"],
            "candidate_paths": active["candidate_paths"],
            "candidate_before_sha256": _aggregate_digest(active["candidate_before"]),
            "frozen_sha256": active["frozen_before"]["sha256"],
            "git_head": active["git_head"],
        })
    except Exception:
        active_path.unlink(missing_ok=True)
        raise
    return active


def _render_command(root: pathlib.Path, command: t.Sequence[str]) -> list[str]:
    python = root / ".venv" / "bin" / "python"
    executable = str(python if python.is_file() else pathlib.Path(sys.executable))
    return [part.format(python=executable) for part in command]


def _run_check(
    name: str,
    command: t.Sequence[str],
    *,
    root: pathlib.Path,
    timeout: float,
) -> dict:
    rendered = _render_command(root, command)
    started = time.time()
    try:
        proc = subprocess.run(
            rendered, cwd=root, capture_output=True, text=True, timeout=max(timeout, 0.001),
            check=False,
        )
        output = (proc.stdout + proc.stderr)[-8000:]
        return {
            "name": name, "command": rendered, "returncode": proc.returncode,
            "elapsed_seconds": time.time() - started, "output_tail": output,
            "passed": proc.returncode == 0,
        }
    except subprocess.TimeoutExpired as exc:
        tail = ((exc.stdout or "") + (exc.stderr or ""))[-8000:]
        return {
            "name": name, "command": rendered, "returncode": None,
            "elapsed_seconds": time.time() - started, "output_tail": tail,
            "passed": False, "timed_out": True,
        }


def evaluate_candidate(
    active_path: pathlib.Path,
    *,
    now: t.Callable[[], float] = time.time,
) -> dict:
    """Evaluate the active candidate once, append the verdict, and release the slot."""
    active = _load_active(active_path)
    stored_commands = active.get("check_commands")
    if not isinstance(stored_commands, dict):
        raise LoopError("active candidate has no fixed check command registry")
    validate_check_registry(stored_commands)
    root = pathlib.Path(active["root"])
    log_path = pathlib.Path(active["log_path"])
    current_time = now()
    frozen_before_checks = snapshot_frozen(root)
    frozen_unchanged = frozen_before_checks == active["frozen_before"]
    candidate_after = _snapshot_paths(root, active["candidate_paths"])
    candidate_changed = candidate_after != active["candidate_before"]
    expired_before_checks = current_time > float(active["deadline"])

    check_results: list[dict] = []
    if frozen_unchanged and candidate_changed and not expired_before_checks:
        for name in active["checks"]:
            remaining = float(active["deadline"]) - now()
            if remaining <= 0:
                check_results.append({
                    "name": name, "passed": False, "timed_out": True,
                    "returncode": None, "elapsed_seconds": 0.0, "output_tail": "budget exhausted",
                    "command": list(stored_commands[name]),
                })
                break
            result = _run_check(name, stored_commands[name], root=root, timeout=remaining)
            check_results.append(result)
            if not result["passed"]:
                break

    frozen_after_checks = snapshot_frozen(root)
    finished = now()
    within_budget = finished <= float(active["deadline"])
    all_checks_passed = (
        len(check_results) == len(active["checks"])
        and all(result["passed"] for result in check_results)
    )
    accepted = bool(
        frozen_unchanged
        and frozen_after_checks == active["frozen_before"]
        and candidate_changed
        and within_budget
        and all_checks_passed
    )
    reasons: list[str] = []
    if not candidate_changed:
        reasons.append("declared candidate paths did not change")
    if not frozen_unchanged or frozen_after_checks != active["frozen_before"]:
        reasons.append("scientific frozen surface changed")
    if not within_budget:
        reasons.append("fixed time budget exceeded")
    if not all_checks_passed:
        reasons.append("engineering acceptance checks did not all pass")

    record = {
        "schema": SCHEMA,
        "event": "EVALUATE",
        "candidate_id": active["candidate_id"],
        "accepted": accepted,
        "reasons": reasons,
        "started_at": active["started_at"],
        "finished_at": finished,
        "elapsed_seconds": finished - float(active["started_at"]),
        "time_budget_seconds": active["time_budget_seconds"],
        "max_checks": active["max_checks"],
        "checks": check_results,
        "candidate_paths": active["candidate_paths"],
        "candidate_before_sha256": _aggregate_digest(active["candidate_before"]),
        "candidate_after_sha256": _aggregate_digest(candidate_after),
        "candidate_delta_sha256": _sha256_bytes(json.dumps({
            "before": active["candidate_before"], "after": candidate_after,
        }, sort_keys=True, separators=(",", ":")).encode()),
        "frozen_before_sha256": active["frozen_before"]["sha256"],
        "frozen_after_sha256": frozen_after_checks["sha256"],
        "git_head": active["git_head"],
        "worktree_action": "none; rejected candidates are not reset or reverted",
    }
    _append_record(log_path, record)
    active_path.unlink()
    return record


def abort_candidate(active_path: pathlib.Path, reason: str, *, now: float | None = None) -> dict:
    """Close an attempt without touching its diff."""
    active = _load_active(active_path)
    record = {
        "schema": SCHEMA,
        "event": "ABORT",
        "candidate_id": active["candidate_id"],
        "accepted": False,
        "reason": reason,
        "finished_at": time.time() if now is None else float(now),
        "worktree_action": "none; candidate diff left in place",
    }
    _append_record(pathlib.Path(active["log_path"]), record)
    active_path.unlink()
    return record


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(pathlib.Path(__file__).resolve().parent.parent))
    parser.add_argument("--active-file", help="override the repository-local active record")
    sub = parser.add_subparsers(dest="action", required=True)

    begin = sub.add_parser("begin", help="freeze surfaces and declare exactly one candidate")
    begin.add_argument("--candidate-id", required=True)
    begin.add_argument("--candidate-path", action="append", required=True)
    begin.add_argument("--time-budget-seconds", type=float, default=1800)
    begin.add_argument("--max-checks", type=int, default=2)
    begin.add_argument("--check", action="append", choices=sorted(CHECKS))
    begin.add_argument("--log", default=str(DEFAULT_LOG))

    sub.add_parser("evaluate", help="run the checks fixed by begin and append the verdict")
    abort = sub.add_parser("abort", help="close the active candidate without reverting it")
    abort.add_argument("--reason", required=True)
    sub.add_parser("status", help="print the active candidate, if any")
    sub.add_parser("list-checks", help="print the closed engineering-check registry")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = pathlib.Path(args.root).resolve()
    active_path = pathlib.Path(args.active_file) if args.active_file else _default_active_path(root)
    try:
        if args.action == "begin":
            log = pathlib.Path(args.log)
            if not log.is_absolute():
                log = root / log
            active = begin_candidate(
                root,
                candidate_id=args.candidate_id,
                candidate_paths=args.candidate_path,
                time_budget_seconds=args.time_budget_seconds,
                max_checks=args.max_checks,
                checks=args.check or ["unit", "config"],
                active_path=active_path,
                log_path=log,
            )
            print(json.dumps(active, indent=2, sort_keys=True))
            return 0
        if args.action == "evaluate":
            record = evaluate_candidate(active_path)
            print(json.dumps(record, indent=2, sort_keys=True))
            return 0 if record["accepted"] else 1
        if args.action == "abort":
            print(json.dumps(abort_candidate(active_path, args.reason), indent=2, sort_keys=True))
            return 0
        if args.action == "status":
            if not active_path.is_file():
                print("NO_ACTIVE_CANDIDATE")
                return 1
            print(active_path.read_text(), end="")
            return 0
        if args.action == "list-checks":
            print(json.dumps(CHECKS, indent=2, sort_keys=True))
            return 0
    except LoopError as exc:
        print(f"engineering_autoloop: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
