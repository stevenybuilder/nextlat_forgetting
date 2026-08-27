#!/usr/bin/env python3
"""Retired unsafe Colab profiling entrypoint.

This module is intentionally import-safe. The supported profiler is
``scripts/profile.sh``; remote execution and durable transport are orchestrated by
``scripts/colab_train_loop.py``.
"""

from __future__ import annotations

import sys


DEPRECATION_MESSAGE = """\
scripts/colab_profile.py is retired and cannot launch profiling jobs.

Use the supported profiling gate instead:
  scripts/profile.sh [--lurestar-only|--hmm-only] [--out DIR]

For a Colab runtime, package and execute that gate through:
  .venv/bin/python scripts/colab_train_loop.py

That path derives configs from the pinned project YAML, verifies the requested GPU,
and makes checkpoints and profile artifacts durable before reporting success.
"""


def main() -> int:
    """Refuse execution before any external command or runtime action is possible."""
    print(DEPRECATION_MESSAGE, file=sys.stderr, end="")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
