#!/usr/bin/env python3
"""Upload only bounded D47 diagnostic outputs, never cloned source trees."""

from __future__ import annotations

import hashlib
import os
import pathlib
import sys

from google.cloud import storage


root = pathlib.Path(sys.argv[1])
bucket = storage.Client(project=os.environ["GOOGLE_CLOUD_PROJECT"]).bucket(sys.argv[2])
prefix = sys.argv[3].strip("/")

for path in sorted(root.rglob("*")):
    if not path.is_file() or path.suffix == ".pt" or path.name == "sync_results.py":
        continue
    relative = path.relative_to(root)
    if any(
        part == ".git"
        or part == "__pycache__"
        or part.startswith("patched-nextlat")
        for part in relative.parts
    ):
        continue
    blob = bucket.blob(f"{prefix}/{relative.as_posix()}")
    blob.metadata = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    blob.upload_from_filename(path, timeout=600)
