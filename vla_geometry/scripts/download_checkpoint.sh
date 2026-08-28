#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 CHECKPOINT_DIRECTORY" >&2
  exit 2
fi

checkpoint_dir=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1")
if [[ "$checkpoint_dir" == "/" || "$checkpoint_dir" == "/root" || "$checkpoint_dir" == "/workspace" ]]; then
  echo "refusing broad checkpoint directory: $checkpoint_dir" >&2
  exit 2
fi

mkdir -p "$checkpoint_dir"
checkpoint="$checkpoint_dir/92M.ckpt"
curl --fail --location --retry 5 --continue-at - \
  --output "$checkpoint" \
  https://huggingface.co/VIMA/VIMA/resolve/main/92M.ckpt

actual_bytes=$(wc -c < "$checkpoint" | tr -d ' ')
if [[ "$actual_bytes" != "1093338811" ]]; then
  echo "checkpoint size mismatch: got $actual_bytes expected 1093338811" >&2
  exit 1
fi
sha256sum "$checkpoint" | tee "$checkpoint.sha256"

