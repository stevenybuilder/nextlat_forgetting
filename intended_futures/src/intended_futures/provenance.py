from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from .manifest import sha256_file


_TREE_EXCLUDES = {
    ".git",
    ".pytest_cache",
    ".runtime",
    "__pycache__",
    "raw",
    "results",
    "runs",
}


def sha256_source_tree(root: str | Path) -> str:
    base = Path(root).resolve()
    digest = hashlib.sha256()
    for path in sorted(base.rglob("*")):
        if not path.is_file() or any(part in _TREE_EXCLUDES for part in path.relative_to(base).parts):
            continue
        relative = path.relative_to(base).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_commit(path: str | Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def checkpoint_tree(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    files = []
    total = 0
    for file_path in sorted(item for item in root.rglob("*") if item.is_file()):
        size = file_path.stat().st_size
        total += size
        files.append(
            {
                "path": file_path.relative_to(root).as_posix(),
                "bytes": size,
                "sha256": sha256_file(file_path),
            }
        )
    if not files:
        raise ValueError(f"checkpoint contains no files: {root}")
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "path": str(root.resolve()),
        "bytes": total,
        "files": files,
        "tree_sha256": hashlib.sha256(canonical).hexdigest(),
    }
