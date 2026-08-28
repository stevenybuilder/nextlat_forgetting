from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def sha256_file(path: str | Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_source_tree(project_root: str | Path) -> str:
    """Hash executable source by relative path and contents, excluding generated artifacts."""

    root = Path(project_root)
    paths = sorted(
        path
        for directory in (root / "src", root / "scripts")
        for path in directory.rglob("*")
        if path.is_file() and path.suffix in {".py", ".sh"}
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def atomic_write_npz(
    path: str | Path,
    *,
    activation: np.ndarray,
    metadata: Mapping[str, Any],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            np.savez_compressed(
                stream,
                activation=np.asarray(activation, dtype=np.float32),
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def read_npz_record(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    with np.load(path, allow_pickle=False) as payload:
        activation = np.asarray(payload["activation"], dtype=np.float64)
        metadata = json.loads(str(payload["metadata_json"].item()))
    return activation, metadata


def _command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def runtime_provenance(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    vima_root: str | Path,
    vimabench_root: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    import torch

    try:
        import gym
    except ImportError:
        gym = None
    try:
        import transformers
    except ImportError:
        transformers = None

    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "count": torch.cuda.device_count(),
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
        }
    packages = {}
    for package in (
        "numpy",
        "torch",
        "transformers",
        "gym",
        "scipy",
        "scikit-learn",
        "pybullet",
        "opencv-python-headless",
    ):
        try:
            packages[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            packages[package] = None
    vima_status = _command_output(["git", "-C", str(vima_root), "status", "--porcelain"])
    vimabench_status = _command_output(
        ["git", "-C", str(vimabench_root), "status", "--porcelain"]
    )
    return {
        "config_sha256": sha256_file(config_path),
        "source_tree_sha256": sha256_source_tree(project_root),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_bytes": Path(checkpoint_path).stat().st_size,
        "vima_commit": _command_output(
            ["git", "-C", str(vima_root), "rev-parse", "HEAD"]
        ),
        "vimabench_commit": _command_output(
            ["git", "-C", str(vimabench_root), "rev-parse", "HEAD"]
        ),
        "vima_worktree_clean": vima_status == "",
        "vimabench_worktree_clean": vimabench_status == "",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "transformers": None if transformers is None else transformers.__version__,
        "gym": None if gym is None else gym.__version__,
        "packages": packages,
        "gpu": gpu,
        "environment": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
    }
