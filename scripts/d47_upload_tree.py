#!/usr/bin/env python3
"""Upload a bounded diagnostic run tree, including deliberate checkpoints."""

from __future__ import annotations

import hashlib
import os
import pathlib
import sys

from google.cloud import storage


root = pathlib.Path(sys.argv[1]).resolve()
bucket = storage.Client(project=os.environ["GOOGLE_CLOUD_PROJECT"]).bucket(sys.argv[2])
prefix = sys.argv[3].strip("/")
for path in sorted(root.rglob("*")):
    if not path.is_file() or path.name.endswith(".partial"):
        continue
    relative = path.relative_to(root).as_posix()
    blob = bucket.blob(f"{prefix}/{relative}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if blob.exists() and (blob.metadata or {}).get("sha256") == digest:
        continue
    blob.metadata = {"sha256": digest}
    blob.upload_from_filename(path, timeout=1200)
