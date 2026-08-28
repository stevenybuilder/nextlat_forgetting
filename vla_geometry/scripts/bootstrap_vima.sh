#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 INSTALL_ROOT" >&2
  exit 2
fi

install_root=$(python3 -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$1")
if [[ "$install_root" == "/" || "$install_root" == "/root" || "$install_root" == "/workspace" ]]; then
  echo "refusing broad install root: $install_root" >&2
  exit 2
fi

vima_commit=8449837aa453f8ec9ba229cb956e3bbef5c796ea
bench_commit=97e0af11e126fd477c9d81eeabfb6c739021c680

mkdir -p "$install_root/external"
if [[ ! -d "$install_root/external/VIMA/.git" ]]; then
  git clone https://github.com/vimalabs/VIMA.git "$install_root/external/VIMA"
fi
git -C "$install_root/external/VIMA" fetch origin "$vima_commit"
git -C "$install_root/external/VIMA" checkout --detach "$vima_commit"

if [[ ! -d "$install_root/external/VimaBench/.git" ]]; then
  git clone https://github.com/vimalabs/VimaBench.git "$install_root/external/VimaBench"
fi
git -C "$install_root/external/VimaBench" fetch origin "$bench_commit"
git -C "$install_root/external/VimaBench" checkout --detach "$bench_commit"

python3 -m venv --system-site-packages "$install_root/.venv"
pip_bin="$install_root/.venv/bin/pip"
"$pip_bin" install --upgrade 'pip<25' 'setuptools<71' wheel
"$pip_bin" install \
  numpy==1.26.4 scipy==1.11.4 pandas==2.1.4 scikit-learn==1.4.2 \
  transformers==4.25.1 tokenizers==0.13.3 gym==0.23.1 pybullet dm-tree einops \
  kornia==0.6.12 opencv-python-headless transforms3d hydra-core tqdm av importlib-resources
"$pip_bin" install --no-deps -e "$install_root/external/VIMA" -e "$install_root/external/VimaBench"

"$install_root/.venv/bin/python" - <<'PY'
import gym
import numpy
import platform
import pybullet
import torch
import transformers
import vima
import vima_bench

expected = {
    "python": "3.11.13",
    "numpy": "1.26.4",
    "torch": "2.7.1+cu128",
    "cuda": "12.8",
    "cudnn": "90701",
    "transformers": "4.25.1",
    "gym": "0.23.1",
}
actual = {
    "python": platform.python_version(),
    "numpy": numpy.__version__,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cudnn": str(torch.backends.cudnn.version()),
    "transformers": transformers.__version__,
    "gym": gym.__version__,
}
if actual != expected:
    raise RuntimeError(f"runtime contract mismatch: actual={actual}, expected={expected}")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("runtime contract requires exactly one visible CUDA GPU")
if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090":
    raise RuntimeError(f"unexpected GPU: {torch.cuda.get_device_name(0)}")
print("VIMA_RUNTIME_READY")
print(actual)
PY
