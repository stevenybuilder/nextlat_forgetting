#!/usr/bin/env python3
"""Guarded runtime-only hardening patch for the pinned NextLat checkout.

The Colab driver clones NextLat afresh, checks out ``PINNED_COMMIT``, and invokes this
script before any trainer process starts.  This script deliberately does not modify the
vendored checkout in this repository.  It verifies both the git identity and byte hashes
of the source surface it relies on, applies exact textual edits, installs one small helper
module, compiles the result, and writes a unified diff plus a machine-readable receipt.

Re-running against an already-patched runtime is an idempotent verification.  Any other
source state fails closed: a fuzzy or partially-applied patch is never attempted.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import pathlib
import py_compile
import subprocess
import sys
import tempfile
import time
import typing as t


PINNED_COMMIT = "3770be6009cea2b3c455a9ce7f2ca88b504bb955"
PATCH_SCHEMA = "nextlat_forgetting/runtime_patch/1"
PATCH_VERSION = 5
ORIGINAL_SHA256 = {
    "core_train.py": "d35b608c15a004f3a72fcc66b7df26ec9937fd1183e9a92a1c092f007c0f0e31",
    "models/model_base.py": "e2645ed8e4d83de4daf9a3759b91537ed9c4216d4a189a9723d6d8cdd14ba34a",
    "train.py": "effb8a7eda24aa180a526c532c07b579e72cbc3e7122443e29ce360f19a29ae4",
}
RECEIPT_NAME = ".lurestar_runtime_patch_receipt.json"
HELPER_NAME = "lurestar_runtime.py"
ADAPTATION_HELPER_NAME = "lurestar_adaptation.py"
ADAPTATION_SOURCE_PATH = pathlib.Path(__file__).resolve().parents[1] / "src/lurestar/adaptation.py"


HELPER_SOURCE = r'''"""Runtime durability and preregistration gates injected by runtime_bootstrap.py.

This module is generated into the fresh pinned checkout.  Keep its source in the bootstrap
script so the receipt hashes the complete runtime mutation as one deterministic patch.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import random
import re
import time

CONTRACT_NAME = "h3_full_parameter_next_token_ce_v1"


def adaptation_contract_sha256():
    path = pathlib.Path(__file__).with_name("lurestar_adaptation.py")
    if not path.is_file():
        raise RuntimeError("runtime lacks the hash-bound adaptation trainer")
    return sha256_file(path)


RNG_KEY = "lurestar_rng_state_v1"
SCALER_KEY = "lurestar_amp_scaler_state_v1"
META_SUFFIX = ".meta.json"
EXPECTED_STARGRAPH_PARAMS = {
    "gpt": 21_324_672,
    "nextlat": 21_915_264,
    # Derived from the pinned BST config (12 layers, 6 heads, width 384) and
    # model_bst.py's two independent Block stacks plus TextHead.  See
    # derive_bst_stargraph_total_params(), which is also exercised before use.
    "bst": 47_287_296,
}


def configure_deterministic_runtime(torch_module):
    """Enable the recovery rehearsal's fail-closed CUDA determinism contract."""
    if os.environ.get("LURESTAR_DETERMINISTIC_RUNTIME") != "1":
        return False
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in (":4096:8", ":16:8"):
        raise RuntimeError("deterministic runtime requires CUBLAS_WORKSPACE_CONFIG")
    torch_module.use_deterministic_algorithms(True)
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cuda.matmul.allow_tf32 = False
    torch_module.backends.cudnn.allow_tf32 = False
    return True


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_dir(path):
    try:
        fd = os.open(os.fspath(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_bytes(path, payload):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with open(partial, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    _fsync_dir(path.parent)


def _atomic_json(path, value):
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def capture_rng_state(torch_module):
    """Capture every process RNG used by the pinned trainer."""
    state = {
        "schema": 1,
        "python": random.getstate(),
        "torch_cpu": torch_module.get_rng_state(),
        "torch_cuda": [],
    }
    try:
        import numpy as np
        state["numpy"] = np.random.get_state()
    except ImportError:
        state["numpy"] = None
    if torch_module.cuda.is_available():
        state["torch_cuda"] = torch_module.cuda.get_rng_state_all()
    return state


def restore_rng_state(state, torch_module):
    """Restore a patched checkpoint's RNG metadata; old upstream checkpoints remain valid."""
    if state is None:
        return False
    if not isinstance(state, dict) or state.get("schema") != 1:
        raise RuntimeError("checkpoint contains an unsupported Lure-Star RNG state")
    random.setstate(state["python"])
    if state.get("numpy") is not None:
        import numpy as np
        np.random.set_state(state["numpy"])
    torch_module.set_rng_state(state["torch_cpu"])
    if torch_module.cuda.is_available() and state.get("torch_cuda"):
        if len(state["torch_cuda"]) != torch_module.cuda.device_count():
            raise RuntimeError("checkpoint CUDA RNG device count differs from this runtime")
        torch_module.cuda.set_rng_state_all(state["torch_cuda"])
    return True


def capture_amp_scaler_state(fabric):
    """Capture FP16 loss-scale state without changing BF16 checkpoints."""
    scaler = getattr(fabric.strategy.precision, "scaler", None)
    if scaler is None:
        return None
    return {"schema": 1, "state_dict": scaler.state_dict()}


def restore_amp_scaler_state(state, fabric):
    """Restore a saved FP16 loss scale before the next optimizer update."""
    scaler = getattr(fabric.strategy.precision, "scaler", None)
    if state is None:
        if scaler is not None:
            raise RuntimeError("FP16 checkpoint lacks AMP GradScaler state")
        return False
    if not isinstance(state, dict) or state.get("schema") != 1:
        raise RuntimeError("checkpoint contains an unsupported AMP GradScaler state")
    if scaler is None:
        raise RuntimeError("checkpoint has AMP GradScaler state but runtime precision does not")
    scaler.load_state_dict(state["state_dict"])
    return True


def _torch_load(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def verify_checkpoint(path, *, deserialize=True, require_metadata=True):
    path = pathlib.Path(path).resolve()
    if not path.is_file() or path.name.endswith(".partial"):
        raise RuntimeError("checkpoint is missing or partial: %s" % path)
    metadata_path = path.with_name(path.name + META_SUFFIX)
    if require_metadata and not metadata_path.is_file():
        raise RuntimeError("checkpoint lacks verification metadata: %s" % path)
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    size = path.stat().st_size
    digest = sha256_file(path)
    if metadata:
        if int(metadata.get("size_bytes", -1)) != size:
            raise RuntimeError("checkpoint size disagrees with metadata: %s" % path)
        if metadata.get("sha256") != digest:
            raise RuntimeError("checkpoint hash disagrees with metadata: %s" % path)
    state = _torch_load(path) if deserialize else None
    if deserialize and not isinstance(state, dict):
        raise RuntimeError("checkpoint did not deserialize to a state mapping: %s" % path)
    return {"path": str(path), "size_bytes": size, "sha256": digest}, state


def atomic_fabric_save(fabric, path, state):
    """Use Fabric's serializer, but commit the payload only after fsync and read-back."""
    if int(fabric.world_size) != 1:
        raise RuntimeError("the hardened checkpoint path is preregistered for one GPU only")
    path = pathlib.Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    if fabric.global_rank == 0:
        partial.unlink(missing_ok=True)
    fabric.barrier()
    fabric.save(str(partial), state)
    fabric.barrier()
    if fabric.global_rank == 0:
        if not partial.is_file():
            raise RuntimeError("Fabric returned without producing %s" % partial)
        with open(partial, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(partial, path)
        _fsync_dir(path.parent)
        raw = _torch_load(path)
        if not isinstance(raw, dict):
            raise RuntimeError("new checkpoint failed deserialize verification: %s" % path)
        metadata = {
            "schema": 1,
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "training_steps": int(raw.get("training_steps", -1)),
            "rng_state": RNG_KEY in raw,
            "amp_scaler_state": bool(raw.get(SCALER_KEY)),
            "verified_at": time.time(),
        }
        _atomic_json(path.with_name(path.name + META_SUFFIX), metadata)
    fabric.barrier()


def publish_verified_pointer(pointer, checkpoint):
    """Deep-verify the target before atomically exposing its absolute path."""
    pointer = pathlib.Path(pointer).resolve()
    checkpoint = pathlib.Path(checkpoint).resolve()
    verify_checkpoint(checkpoint, deserialize=True, require_metadata=True)
    _atomic_bytes(pointer, str(checkpoint).encode())


def _step(path):
    match = re.search(r"recovery_ckpt_iter_(\d+)\.pt$", pathlib.Path(path).name)
    return int(match.group(1)) if match else -1


def guarded_remove(path, root, *, protected_pointer=None):
    """Idempotent deletion limited to a direct checkpoint child and never its live pointer."""
    path = pathlib.Path(path).resolve()
    root = pathlib.Path(root).resolve()
    if path.parent != root or path.suffix != ".pt":
        raise RuntimeError("refusing checkpoint deletion outside the exact run directory: %s" % path)
    if protected_pointer:
        pointer = pathlib.Path(protected_pointer)
        if pointer.is_file() and pathlib.Path(pointer.read_text().strip()).resolve() == path:
            raise RuntimeError("refusing to delete the checkpoint named by %s" % pointer)
    path.unlink(missing_ok=True)
    path.with_name(path.name + META_SUFFIX).unlink(missing_ok=True)
    _fsync_dir(root)


def retain_verified_recovery(checkpoint_dir, pointer, *, keep=2):
    """Keep two deeply verified recovery generations, pruning only after both exist."""
    if keep < 2:
        raise ValueError("recovery retention may not be less than two")
    checkpoint_dir = pathlib.Path(checkpoint_dir).resolve()
    verified = []
    for path in sorted(checkpoint_dir.glob("recovery_ckpt_iter_*.pt"), key=_step, reverse=True):
        try:
            verify_checkpoint(path, deserialize=True, require_metadata=True)
        except Exception:
            continue
        verified.append(path.resolve())
    if len(verified) < keep:
        return [str(path) for path in verified]
    retained = verified[:keep]
    for path in verified[keep:]:
        guarded_remove(path, checkpoint_dir, protected_pointer=pointer)
    return [str(path) for path in retained]


def derive_bst_stargraph_total_params():
    """Exact parameter arithmetic for pinned model_bst.py and its frozen G(5,5) config."""
    width, layers, vocab = 384, 12, 106
    hidden = 128 * round((8 * width / 3) / 128)  # Block MLP: 1024
    block = (
        2 * width                              # two RMSNorm weights
        + width * (3 * width) + width * width # attention
        + width * (2 * hidden) + hidden * width
    )
    encoder = vocab * width + 2 * (layers * block + width)
    text_input = 2 * width
    text_hidden = 128 * round((8 * text_input / 3) / 128)  # TextHead MLP: 2048
    text_head = (
        text_input * (2 * text_hidden)
        + text_hidden * text_input
        + text_input
        + width * vocab
    )
    return encoder + text_head


def _model_name(config):
    enabled = {
        "bst": bool(config.use_bst),
        "nextlat": bool(config.use_nextlat),
        "gpt": not bool(config.use_bst) and not bool(config.use_nextlat)
               and not bool(config.use_mtp_gloeckle) and not bool(config.use_mtp_jtp),
    }
    names = [name for name, value in enabled.items() if value]
    if len(names) != 1:
        raise RuntimeError("step-0 gate requires exactly one GPT/NextLat/BST arm")
    return names[0]


def _claim_job_identity(config, model_name):
    out_dir = pathlib.Path(str(config.trainer.out_dir))
    if not out_dir.is_absolute():
        raise RuntimeError("trainer.out_dir must be absolute")
    out_dir = out_dir.resolve()
    seed = int(config.seed)
    experiment = str(config.trainer.experiment_name)
    seed_marker = "seed%d" % seed
    # ``launch_train.sh`` uses ``.../seed1234/...`` while MatrixRunner's canonical,
    # independently collision-checked layout is ``.../1234/...``.  The resolved upstream
    # experiment name always carries ``seed1234``.  Accept precisely those two directory
    # encodings rather than coupling the runtime gate to one launcher.
    if seed_marker not in experiment or not ({str(seed), seed_marker} & set(out_dir.parts)):
        raise RuntimeError("out_dir and experiment_name must both encode the explicit seed")
    if model_name not in experiment or model_name not in out_dir.parts:
        raise RuntimeError("out_dir and experiment_name must both encode the selected model")
    identity = {
        "schema": 1,
        "dataset": str(config.data.dataset),
        "experiment_name": experiment,
        "model": model_name,
        "out_dir": str(out_dir),
        "seed": seed,
        "train_batches": int(config.trainer.train_batches),
    }
    path = out_dir / ".lurestar_job_identity.json"
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing != identity:
            raise RuntimeError("out_dir is already claimed by a different job identity")
    else:
        _atomic_json(path, identity)
    return identity


def assert_step0_contract(config, model, fabric):
    """Fail closed on every silent configuration drift before optimizer work begins."""
    if int(fabric.world_size) != 1:
        raise RuntimeError("confirmatory contract requires exactly one GPU")
    if bool(config.trainer.compile):
        raise RuntimeError("trainer.compile must be false")
    if float(config.optimizer.grad_clip) != 100.0:
        raise RuntimeError("optimizer.grad_clip must be 100")
    seed = config.seed
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RuntimeError("seed must be one explicit integer, not a sweep/list")

    model_name = _model_name(config)
    identity = _claim_job_identity(config, model_name)
    dataset = str(config.data.dataset)
    profile = bool(os.environ.get("PROFILE_PROBE_JSON"))
    nonconfirmatory = profile or os.environ.get("LURESTAR_NONCONFIRMATORY") == "1"
    target = int(config.trainer.train_batches)
    if not nonconfirmatory:
        expected_target = 3000 if dataset == "hmm_belief" else (
            20500 if "adapt-" in str(config.trainer.experiment_name) else 20000
        )
        if target != expected_target:
            raise RuntimeError(
                "trainer.train_batches=%d, expected preregistered target %d" %
                (target, expected_target)
            )
    if target <= int(model.training_steps):
        raise RuntimeError("training target must exceed the restored/current optimizer step")

    expected_batch = 256 if dataset == "hmm_belief" else 512
    if int(config.data.effective_batch_size) != expected_batch:
        raise RuntimeError("effective_batch_size must be %d for %s" % (expected_batch, dataset))

    parameter_count = int(model.get_num_params(non_embedding=False))
    if dataset == "stargraph":
        frozen_shape = (12, 6, 384, 106, 69, 62)
        actual_shape = (
            int(config.model.n_layer), int(config.model.n_head), int(config.model.n_embd),
            int(config.model.vocab_size), int(config.model.block_size),
            int(config.model.context_length),
        )
        if actual_shape != frozen_shape:
            raise RuntimeError("resolved G(5,5) model shape drift: %r" % (actual_shape,))
        if model_name == "nextlat" and float(config.model.proj_factor) != 0.5:
            raise RuntimeError("resolved NextLat model.proj_factor must be 0.5")
        if derive_bst_stargraph_total_params() != EXPECTED_STARGRAPH_PARAMS["bst"]:
            raise RuntimeError("internal BST parameter derivation drifted")
        expected_params = EXPECTED_STARGRAPH_PARAMS[model_name]
        if parameter_count != expected_params:
            raise RuntimeError(
                "%s total parameter count %d != frozen %d" %
                (model_name, parameter_count, expected_params)
            )

    is_adaptation = "-adapt-" in str(config.trainer.experiment_name)
    adaptation = None
    if is_adaptation:
        if dataset != "stargraph":
            raise RuntimeError("H3 adaptation contract is stargraph-only")
        if model_name == "nextlat":
            coefficients = tuple(float(getattr(config.model, key)) for key in (
                "lambda_mse", "lambda_kl", "lambda_ce"
            ))
            if coefficients != (0.0, 0.0, 0.0):
                raise RuntimeError("NextLat H3 adaptation auxiliary coefficients must all be zero")
        adaptation = {
            "contract": CONTRACT_NAME,
            "contract_sha256": adaptation_contract_sha256(),
            "full_parameter": True,
            "loss": "teacher_forced_next_token_cross_entropy",
            "bst_dense_prefix_suffix_objective": False if model_name == "bst" else None,
            "bst_backward_input": "item_independent_lone_eos" if model_name == "bst" else None,
        }

    receipt = dict(identity)
    receipt.update({
        "compile": False,
        "effective_batch_size": int(config.data.effective_batch_size),
        "grad_clip": float(config.optimizer.grad_clip),
        "mode": "nonconfirmatory" if nonconfirmatory else "confirmatory",
        "parameter_count_total": parameter_count,
        "proj_factor": float(config.model.proj_factor) if model_name == "nextlat" else None,
        "world_size": int(fabric.world_size),
        "adaptation": adaptation,
    })
    receipt_path = pathlib.Path(identity["out_dir"]) / "metrics" / "step_0_contract.json"
    _atomic_json(receipt_path, receipt)
    fabric.print("LURESTAR_STEP0=" + json.dumps(receipt, sort_keys=True))
    return receipt
'''


class PatchError(RuntimeError):
    pass


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: pathlib.Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise PatchError(f"{label}: expected exactly one source match, found {count}")
    return source.replace(old, new, 1)


def _patch_model_base(source: str) -> str:
    source = _replace_once(
        source,
        "from utils.import_utils import is_liger_kernel_available\n",
        "from utils.import_utils import is_liger_kernel_available\n"
        "from lurestar_runtime import (RNG_KEY, SCALER_KEY, atomic_fabric_save, "
        "capture_amp_scaler_state, capture_rng_state, restore_amp_scaler_state, "
        "restore_rng_state)\n",
        "model_base import",
    )
    source = _replace_once(
        source,
        "        if self.lr_scheduler_state is not None:\n"
        "            state[\"lr_scheduler_state\"] = self.lr_scheduler_state\n"
        "        self.fabric.save(file_path, state)\n",
        "        if self.lr_scheduler_state is not None:\n"
        "            state[\"lr_scheduler_state\"] = self.lr_scheduler_state\n"
        "        state[SCALER_KEY] = capture_amp_scaler_state(self.fabric)\n"
        "        state[RNG_KEY] = capture_rng_state(torch)\n"
        "        atomic_fabric_save(self.fabric, file_path, state)\n",
        "atomic checkpoint save",
    )
    source = _replace_once(
        source,
        "        self.fabric.load(file_path, state, strict=strict, weights_only=False)\n",
        "        checkpoint_remainder = self.fabric.load(\n"
        "            file_path, state, strict=strict, weights_only=False\n"
        "        )\n",
        "Fabric checkpoint metadata remainder",
    )
    source = _replace_once(
        source,
        "        self.training_steps = state[\"training_steps\"]\n"
        "        self.lr_scheduler_state = self._read_scheduler_state_from_checkpoint(file_path)\n",
        "        self.training_steps = state[\"training_steps\"]\n"
        "        self.lr_scheduler_state = self._read_scheduler_state_from_checkpoint(file_path)\n"
        "        restored_scaler = restore_amp_scaler_state(\n"
        "            checkpoint_remainder.get(SCALER_KEY), self.fabric\n"
        "        )\n"
        "        self.fabric.print(f\"Restored Lure-Star AMP scaler state: {restored_scaler}\")\n"
        "        restored_rng = restore_rng_state(checkpoint_remainder.get(RNG_KEY), torch)\n"
        "        self.fabric.print(f\"Restored Lure-Star RNG state: {restored_rng}\")\n",
        "RNG restore",
    )
    return source


def _patch_core_train(source: str) -> str:
    source = _replace_once(
        source,
        "from models.model_speculative import SpeculativeModel\n",
        "from models.model_speculative import SpeculativeModel\n"
        "from lurestar_adaptation import install_common_adaptation\n"
        "from lurestar_runtime import (assert_step0_contract, guarded_remove, "
        "publish_verified_pointer, retain_verified_recovery)\n",
        "core_train import",
    )
    source = _replace_once(
        source,
        "    else:\n"
        "        fabric.print(\"Initialized a new model from scratch\")\n\n"
        "    return model\n",
        "    else:\n"
        "        fabric.print(\"Initialized a new model from scratch\")\n\n"
        "    install_common_adaptation(config, model, fabric)\n"
        "    return model\n",
        "common adaptation installation",
    )
    source = _replace_once(
        source,
        "    fabric.print(\n"
        "        f\"Number of parameters (excluding embedding): {model.get_num_params(non_embedding=True):,}\"\n"
        "    )\n\n"
        "    # Compile must occur before fabric.setup()\n",
        "    fabric.print(\n"
        "        f\"Number of parameters (excluding embedding): {model.get_num_params(non_embedding=True):,}\"\n"
        "    )\n"
        "    assert_step0_contract(config, model, fabric)\n\n"
        "    # Compile must occur before fabric.setup()\n",
        "step-0 contract",
    )
    source = _replace_once(
        source,
        "if self.step > self.config.trainer.train_batches:",
        "if self.step >= self.config.trainer.train_batches:",
        "exact optimizer update count",
    )
    source = _replace_once(
        source,
        "        use_fused = fused_available and is_device_cuda\n",
        "        # Lightning cannot clip scaled FP16 gradients through an optimizer that\n"
        "        # handles AMP unscaling internally (PyTorch fused AdamW on T4). Keep the\n"
        "        # paper/confirmatory BF16 CUDA path fused, but use the equivalent non-fused\n"
        "        # AdamW implementation whenever an actual GradScaler is active.\n"
        "        amp_scaler = getattr(fabric.strategy.precision, \"scaler\", None)\n"
        "        use_fused = fused_available and is_device_cuda and amp_scaler is None\n",
        "FP16 gradient clipping compatibility",
    )
    source = _replace_once(
        source,
        "                        if ckpt not in self.checkpoints_to_always_keep:\n"
        "                            os.remove(ckpt)\n",
        "                        if ckpt not in self.checkpoints_to_always_keep:\n"
        "                            guarded_remove(\n"
        "                                ckpt,\n"
        "                                os.path.join(self.config.trainer.out_dir,\n"
        "                                             self.config.trainer.experiment_name),\n"
        "                                protected_pointer=os.path.join(\n"
        "                                    self.config.trainer.out_dir, \"latest_ckpt\"),\n"
        "                            )\n",
        "guard validation checkpoint deletion",
    )
    source = _replace_once(
        source,
        "        # Save the file path to the latest checkpoint\n"
        "        if self.fabric.global_rank == 0:\n"
        "            with open(\n"
        "                os.path.join(self.config.trainer.out_dir, \"latest_ckpt\"),\n"
        "                \"w\",\n"
        "            ) as f:\n"
        "                f.write(ckpt_path)\n",
        "        # Publish only a hash- and deserialize-verified checkpoint.\n"
        "        if self.fabric.global_rank == 0:\n"
        "            publish_verified_pointer(\n"
        "                os.path.join(self.config.trainer.out_dir, \"latest_ckpt\"), ckpt_path\n"
        "            )\n",
        "atomic latest pointer",
    )
    source = _replace_once(
        source,
        "        # Save the most recent file path to the recovery checkpoint pointer file\n"
        "        if self.fabric.global_rank == 0:\n"
        "            with open(\n"
        "                os.path.join(self.config.trainer.out_dir, \"recovery_ckpt\"),\n"
        "                \"w\",\n"
        "            ) as f:\n"
        "                f.write(ckpt_path)\n\n"
        "        # Delete the old recovery checkpoint file if it exists\n"
        "        if self.recovery_checkpoint_path is not None:\n"
        "            if self.fabric.global_rank == 0:\n"
        "                os.remove(self.recovery_checkpoint_path)\n\n"
        "        # Update the recovery checkpoint path\n"
        "        self.recovery_checkpoint_path = ckpt_path\n",
        "        # Publish after read-back verification, then retain two verified generations.\n"
        "        if self.fabric.global_rank == 0:\n"
        "            recovery_pointer = os.path.join(\n"
        "                self.config.trainer.out_dir, \"recovery_ckpt\"\n"
        "            )\n"
        "            publish_verified_pointer(recovery_pointer, ckpt_path)\n"
        "            retain_verified_recovery(ckpt_dir, recovery_pointer, keep=2)\n\n"
        "        # Kept for upstream observability; retention is reconstructed from disk on resume.\n"
        "        self.recovery_checkpoint_path = ckpt_path\n",
        "verified recovery rotation",
    )
    return source


def _patch_train(source: str) -> str:
    source = _replace_once(
        source,
        "from data.manhattan_dataset import ManhattanDataModule\n",
        "from data.manhattan_dataset import ManhattanDataModule\n"
        "from lurestar_runtime import configure_deterministic_runtime\n",
        "train deterministic runtime import",
    )
    source = _replace_once(
        source,
        "    fabric.seed_everything(int(config.seed) + int(seed_offset))\n"
        "    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul\n"
        "    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn\n"
        "    torch._dynamo.config.cache_size_limit = 16  # allow more recompiles\n",
        "    fabric.seed_everything(int(config.seed) + int(seed_offset))\n"
        "    deterministic_runtime = configure_deterministic_runtime(torch)\n"
        "    if not deterministic_runtime:\n"
        "        torch.backends.cuda.matmul.allow_tf32 = True  # upstream behavior\n"
        "        torch.backends.cudnn.allow_tf32 = True  # upstream behavior\n"
        "    fabric.print(f\"LURESTAR_DETERMINISTIC_RUNTIME={deterministic_runtime}\")\n"
        "    torch._dynamo.config.cache_size_limit = 16  # allow more recompiles\n",
        "train deterministic runtime activation",
    )
    return source


def _git_head(upstream: pathlib.Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PatchError(f"cannot establish upstream git identity: {exc}") from exc


def _atomic_write(path: pathlib.Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_audit(project_root: pathlib.Path, receipt: dict[str, t.Any], diff: str) -> None:
    audit = project_root / "source_snapshot" / "runtime_patch"
    _atomic_write(audit / "runtime_patch.diff", diff.encode())
    _atomic_write(
        audit / "runtime_patch_receipt.json",
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode(),
    )


def apply_runtime_patch(
    project_root: pathlib.Path,
    upstream: pathlib.Path,
    *,
    expected_commit: str = PINNED_COMMIT,
) -> dict[str, t.Any]:
    project_root = project_root.resolve()
    upstream = upstream.resolve()
    head = _git_head(upstream)
    if head != expected_commit:
        raise PatchError(f"upstream commit drift: {head} != {expected_commit}")

    receipt_path = upstream / RECEIPT_NAME
    helper_path = upstream / HELPER_NAME
    adaptation_helper_path = upstream / ADAPTATION_HELPER_NAME
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("schema") != PATCH_SCHEMA or receipt.get("patch_version") != PATCH_VERSION:
            raise PatchError("runtime patch receipt has an unknown schema/version")
        if receipt.get("upstream_commit") != head:
            raise PatchError("runtime patch receipt commit does not match HEAD")
        for relative, expected in receipt["after_sha256"].items():
            path = upstream / relative
            if not path.is_file() or _sha256(path) != expected:
                raise PatchError(f"already-patched source drift: {relative}")
        _write_audit(project_root, receipt, receipt["unified_diff"])
        print("runtime bootstrap: existing patch receipt verified (idempotent)")
        return receipt

    for relative, expected in ORIGINAL_SHA256.items():
        path = upstream / relative
        if not path.is_file():
            raise PatchError(f"required pinned source is missing: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise PatchError(f"pinned source drift: {relative} sha256 {actual} != {expected}")
    for helper in (helper_path, adaptation_helper_path):
        if helper.exists():
            raise PatchError(f"unreceipted runtime helper already exists: {helper}")
    if not ADAPTATION_SOURCE_PATH.is_file():
        raise PatchError(f"adaptation trainer source is missing: {ADAPTATION_SOURCE_PATH}")
    adaptation_source = ADAPTATION_SOURCE_PATH.read_text(encoding="utf-8")

    originals = {
        "core_train.py": (upstream / "core_train.py").read_text(),
        "models/model_base.py": (upstream / "models/model_base.py").read_text(),
        "train.py": (upstream / "train.py").read_text(),
    }
    patched = {
        "core_train.py": _patch_core_train(originals["core_train.py"]),
        "models/model_base.py": _patch_model_base(originals["models/model_base.py"]),
        "train.py": _patch_train(originals["train.py"]),
        HELPER_NAME: HELPER_SOURCE,
        ADAPTATION_HELPER_NAME: adaptation_source,
    }

    diffs: list[str] = []
    for relative in (
        "core_train.py", "models/model_base.py", "train.py", HELPER_NAME,
        ADAPTATION_HELPER_NAME,
    ):
        before = originals.get(relative, "")
        after = patched[relative]
        if before == after:
            continue
        diffs.extend(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile=f"a/{relative}", tofile=f"b/{relative}",
        ))
    unified_diff = "".join(diffs)

    for relative, source in patched.items():
        _atomic_write(upstream / relative, source.encode())
    for relative in patched:
        py_compile.compile(str(upstream / relative), doraise=True)

    after_sha256 = {relative: _sha256(upstream / relative) for relative in patched}
    receipt: dict[str, t.Any] = {
        "schema": PATCH_SCHEMA,
        "patch_version": PATCH_VERSION,
        "upstream_commit": head,
        "before_sha256": dict(ORIGINAL_SHA256),
        "after_sha256": after_sha256,
        "helper_sha256": _sha256_bytes(HELPER_SOURCE.encode()),
        "adaptation_trainer_sha256": _sha256_bytes(adaptation_source.encode()),
        "adaptation_contract": "h3_full_parameter_next_token_ce_v1",
        "bst_parameter_count": 47_287_296,
        "optimizer_update_rule": "stop when step >= trainer.train_batches",
        "optimizer_fusion_rule": "disable fused AdamW only when Fabric has an AMP GradScaler",
        "amp_scaler_checkpoint_rule": "save and restore GradScaler state when active",
        "deterministic_runtime_rule": (
            "conditional deterministic algorithms, CUBLAS workspace, no TF32, no cuDNN benchmark"
        ),
        "generated_at_unix": time.time(),
        "unified_diff": unified_diff,
    }
    _atomic_write(
        receipt_path, (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    )
    _write_audit(project_root, receipt, unified_diff)
    print("runtime bootstrap: guarded patch applied")
    print("runtime bootstrap: receipt=%s" % receipt_path)
    print("runtime bootstrap: diff_sha256=%s" % _sha256_bytes(unified_diff.encode()))
    return receipt


def main(argv: t.Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=pathlib.Path)
    parser.add_argument("--upstream", required=True, type=pathlib.Path)
    parser.add_argument(
        "--expected-commit", default=PINNED_COMMIT,
        help=argparse.SUPPRESS,  # test fixtures still require both a commit and pinned file hashes
    )
    args = parser.parse_args(argv)
    try:
        apply_runtime_patch(
            args.project_root, args.upstream, expected_commit=args.expected_commit
        )
    except PatchError as exc:
        print(f"runtime bootstrap: REFUSED: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
