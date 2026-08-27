"""Focused tests for the guarded Colab runtime patch."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import pathlib
import pickle
import py_compile
import random
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


PROJECT = pathlib.Path(__file__).resolve().parents[1]
BOOTSTRAP_PATH = PROJECT / "scripts" / "runtime_bootstrap.py"
PINNED = PROJECT / "upstream" / "NextLat"


def _load_bootstrap():
    spec = importlib.util.spec_from_file_location("runtime_bootstrap_under_test", BOOTSTRAP_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_run_matrix():
    path = PROJECT / "scripts" / "run_matrix.py"
    spec = importlib.util.spec_from_file_location("run_matrix_for_runtime_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_colab_driver():
    path = PROJECT / "scripts" / "colab_train_loop.py"
    spec = importlib.util.spec_from_file_location("colab_driver_for_runtime_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(*args: str, cwd: pathlib.Path) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, stderr=subprocess.STDOUT
    ).strip()


def _pinned_fixture(tmp_path: pathlib.Path) -> tuple[pathlib.Path, str]:
    """Copy only the pinned files the patch reads and give them a real git identity."""
    upstream = tmp_path / "NextLat"
    (upstream / "models").mkdir(parents=True)
    for relative in ("core_train.py", "models/model_base.py", "train.py"):
        source = PINNED / relative
        target = upstream / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    _git("init", "-q", cwd=upstream)
    _git("config", "user.email", "runtime-patch-test@example.invalid", cwd=upstream)
    _git("config", "user.name", "Runtime Patch Test", cwd=upstream)
    _git("add", ".", cwd=upstream)
    _git("commit", "-qm", "pinned source fixture", cwd=upstream)
    return upstream, _git("rev-parse", "HEAD", cwd=upstream)


def _load_generated_helper(path: pathlib.Path):
    adaptation = path.with_name("lurestar_adaptation.py")
    if not adaptation.is_file():
        adaptation.write_bytes((PROJECT / "src/lurestar/adaptation.py").read_bytes())
    spec = importlib.util.spec_from_file_location("generated_lurestar_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_guarded_patch_applies_compiles_and_is_idempotent(tmp_path) -> None:
    bootstrap = _load_bootstrap()
    upstream, commit = _pinned_fixture(tmp_path)
    project = tmp_path / "project"
    project.mkdir()

    receipt = bootstrap.apply_runtime_patch(project, upstream, expected_commit=commit)

    assert receipt["schema"] == bootstrap.PATCH_SCHEMA
    assert receipt["optimizer_update_rule"] == (
        "stop when step >= trainer.train_batches"
    )
    assert receipt["optimizer_fusion_rule"] == (
        "disable fused AdamW only when Fabric has an AMP GradScaler"
    )
    assert receipt["amp_scaler_checkpoint_rule"] == (
        "save and restore GradScaler state when active"
    )
    assert receipt["deterministic_runtime_rule"].startswith(
        "conditional deterministic algorithms"
    )
    assert receipt["bst_parameter_count"] == 47_287_296
    core = (upstream / "core_train.py").read_text()
    model_base = (upstream / "models/model_base.py").read_text()
    helper = (upstream / "lurestar_runtime.py").read_text()
    adaptation = (upstream / "lurestar_adaptation.py").read_text()
    train = (upstream / "train.py").read_text()
    assert "if self.step >= self.config.trainer.train_batches:" in core
    assert "if self.step > self.config.trainer.train_batches:" not in core
    assert "assert_step0_contract(config, model, fabric)" in core
    assert "install_common_adaptation(config, model, fabric)" in core
    assert 'amp_scaler = getattr(fabric.strategy.precision, "scaler", None)' in core
    assert (
        "use_fused = fused_available and is_device_cuda and amp_scaler is None"
        in core
    )
    assert "publish_verified_pointer(recovery_pointer, ckpt_path)" in core
    assert "retain_verified_recovery(ckpt_dir, recovery_pointer, keep=2)" in core
    assert "atomic_fabric_save(self.fabric, file_path, state)" in model_base
    assert "state[RNG_KEY] = capture_rng_state(torch)" in model_base
    assert "state[SCALER_KEY] = capture_amp_scaler_state(self.fabric)" in model_base
    assert "restore_amp_scaler_state(" in model_base
    assert "checkpoint_remainder = self.fabric.load(" in model_base
    assert "restore_rng_state(checkpoint_remainder.get(RNG_KEY), torch)" in model_base
    assert ".partial" in helper and "os.fsync" in helper and "os.replace" in helper
    assert "configure_deterministic_runtime(torch)" in train
    assert "torch_module.use_deterministic_algorithms(True)" in helper
    assert "torch_module.backends.cudnn.benchmark = False" in helper
    assert "torch_module.backends.cuda.matmul.allow_tf32 = False" in helper
    assert receipt["adaptation_contract"] == "h3_full_parameter_next_token_ce_v1"
    assert receipt["adaptation_trainer_sha256"] == hashlib.sha256(
        adaptation.encode()
    ).hexdigest()
    assert "model.encoder(prefixes, compute_forward=True, compute_backward=False)" in adaptation
    assert "model.text_head(forward, backward)" in adaptation
    assert "next_previous[:, 0, :, :]" in adaptation
    assert (project / "source_snapshot/runtime_patch/runtime_patch.diff").is_file()
    assert (project / "source_snapshot/runtime_patch/runtime_patch_receipt.json").is_file()
    assert "core_train.py" in receipt["unified_diff"]
    assert "models/model_base.py" in receipt["unified_diff"]
    assert "train.py" in receipt["unified_diff"]
    assert "lurestar_runtime.py" in receipt["unified_diff"]
    for relative in (
        "core_train.py", "models/model_base.py", "train.py", "lurestar_runtime.py",
        "lurestar_adaptation.py",
    ):
        py_compile.compile(str(upstream / relative), doraise=True)
    subprocess.run(["git", "diff", "--check"], cwd=upstream, check=True)

    before = {
        path: (upstream / path).read_bytes()
        for path in receipt["after_sha256"]
    }
    receipt_bytes = (upstream / bootstrap.RECEIPT_NAME).read_bytes()
    again = bootstrap.apply_runtime_patch(project, upstream, expected_commit=commit)
    assert again == receipt
    assert (upstream / bootstrap.RECEIPT_NAME).read_bytes() == receipt_bytes
    assert all((upstream / path).read_bytes() == payload for path, payload in before.items())


def test_patch_fails_closed_on_commit_or_source_drift(tmp_path) -> None:
    bootstrap = _load_bootstrap()
    upstream, commit = _pinned_fixture(tmp_path)
    project = tmp_path / "project"
    project.mkdir()

    with pytest.raises(bootstrap.PatchError, match="commit drift"):
        bootstrap.apply_runtime_patch(project, upstream, expected_commit="0" * 40)

    core = upstream / "core_train.py"
    core.write_text(core.read_text() + "\n# unreviewed runtime drift\n")
    with pytest.raises(bootstrap.PatchError, match="source drift: core_train.py"):
        bootstrap.apply_runtime_patch(project, upstream, expected_commit=commit)
    assert not (upstream / "lurestar_runtime.py").exists()


def test_receipted_patch_fails_closed_if_any_patched_byte_moves(tmp_path) -> None:
    bootstrap = _load_bootstrap()
    upstream, commit = _pinned_fixture(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    bootstrap.apply_runtime_patch(project, upstream, expected_commit=commit)
    helper = upstream / "lurestar_runtime.py"
    helper.write_text(helper.read_text() + "\n# drift\n")

    with pytest.raises(bootstrap.PatchError, match="already-patched source drift"):
        bootstrap.apply_runtime_patch(project, upstream, expected_commit=commit)


def test_bst_count_is_derived_from_the_pinned_architecture(tmp_path) -> None:
    bootstrap = _load_bootstrap()
    helper_path = tmp_path / "lurestar_runtime.py"
    helper_path.write_text(bootstrap.HELPER_SOURCE)
    helper = _load_generated_helper(helper_path)

    assert helper.derive_bst_stargraph_total_params() == 47_287_296
    assert helper.EXPECTED_STARGRAPH_PARAMS == {
        "gpt": 21_324_672,
        "nextlat": 21_915_264,
        "bst": 47_287_296,
    }


def test_deterministic_runtime_behavior_and_workspace_gate(tmp_path, monkeypatch) -> None:
    bootstrap = _load_bootstrap()
    helper_path = tmp_path / "lurestar_runtime.py"
    helper_path.write_text(bootstrap.HELPER_SOURCE)
    helper = _load_generated_helper(helper_path)
    calls = []
    fake = SimpleNamespace(
        use_deterministic_algorithms=lambda enabled: calls.append(enabled),
        backends=SimpleNamespace(
            cudnn=SimpleNamespace(benchmark=True, deterministic=False, allow_tf32=True),
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
        ),
    )

    monkeypatch.setenv("LURESTAR_DETERMINISTIC_RUNTIME", "1")
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        helper.configure_deterministic_runtime(fake)

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    assert helper.configure_deterministic_runtime(fake) is True
    assert calls == [True]
    assert fake.backends.cudnn.benchmark is False
    assert fake.backends.cudnn.deterministic is True
    assert fake.backends.cudnn.allow_tf32 is False
    assert fake.backends.cuda.matmul.allow_tf32 is False


def _stargraph_config(out_dir: pathlib.Path, *, model: str = "gpt"):
    flags = {
        "use_bst": model == "bst",
        "use_nextlat": model == "nextlat",
        "use_mtp_gloeckle": False,
        "use_mtp_jtp": False,
    }
    return SimpleNamespace(
        **flags,
        seed=1234,
        trainer=SimpleNamespace(
            out_dir=str(out_dir), experiment_name=f"{model}-seed1234-base",
            compile=False, train_batches=20_000,
        ),
        optimizer=SimpleNamespace(grad_clip=100),
        data=SimpleNamespace(dataset="stargraph", effective_batch_size=512),
        model=SimpleNamespace(
            n_layer=12, n_head=6, n_embd=384, vocab_size=106, block_size=69,
            context_length=62, proj_factor=0.5,
        ),
    )


def test_step0_contract_emits_receipt_and_rejects_silent_drift(tmp_path) -> None:
    bootstrap = _load_bootstrap()
    helper_path = tmp_path / "lurestar_runtime.py"
    helper_path.write_text(bootstrap.HELPER_SOURCE)
    helper = _load_generated_helper(helper_path)
    out_dir = tmp_path / "runs" / "gpt" / "seed1234" / "base"
    config = _stargraph_config(out_dir)
    fabric = SimpleNamespace(world_size=1, print=lambda _message: None)
    model = SimpleNamespace(
        training_steps=0,
        get_num_params=lambda non_embedding=False: 21_324_672,
    )

    receipt = helper.assert_step0_contract(config, model, fabric)

    assert receipt["train_batches"] == 20_000
    assert receipt["parameter_count_total"] == 21_324_672
    assert json.loads((out_dir / "metrics/step_0_contract.json").read_text()) == receipt
    assert json.loads((out_dir / ".lurestar_job_identity.json").read_text())["seed"] == 1234

    config.optimizer.grad_clip = 1
    with pytest.raises(RuntimeError, match="grad_clip"):
        helper.assert_step0_contract(config, model, fabric)


def test_step0_contract_preserves_20000_3000_and_parent_plus_500_targets(tmp_path) -> None:
    bootstrap = _load_bootstrap()
    helper_path = tmp_path / "lurestar_runtime.py"
    helper_path.write_text(bootstrap.HELPER_SOURCE)
    helper = _load_generated_helper(helper_path)
    fabric = SimpleNamespace(world_size=1, print=lambda _message: None)

    base_out = tmp_path / "runs" / "gpt" / "seed1234" / "base"
    base = _stargraph_config(base_out)
    gpt = SimpleNamespace(training_steps=0, get_num_params=lambda non_embedding=False: 21_324_672)
    assert helper.assert_step0_contract(base, gpt, fabric)["train_batches"] == 20_000

    adapt_out = tmp_path / "runs" / "gpt" / "seed1234" / "adapt-near"
    adapt = _stargraph_config(adapt_out)
    adapt.trainer.experiment_name = "gpt-seed1234-adapt-near"
    adapt.trainer.train_batches = 20_500
    gpt.training_steps = 20_000
    assert helper.assert_step0_contract(adapt, gpt, fabric)["train_batches"] == 20_500

    hmm_out = tmp_path / "runs" / "gpt" / "seed1234" / "base-hmm"
    hmm = _stargraph_config(hmm_out)
    hmm.trainer.experiment_name = "gpt-seed1234-hmm"
    hmm.trainer.train_batches = 3_000
    hmm.data.dataset = "hmm_belief"
    hmm.data.effective_batch_size = 256
    small = SimpleNamespace(training_steps=0, get_num_params=lambda non_embedding=False: 123)
    assert helper.assert_step0_contract(hmm, small, fabric)["train_batches"] == 3_000


def test_real_matrix_paths_pass_step0_identity_and_target_gates(tmp_path) -> None:
    bootstrap = _load_bootstrap()
    matrix = _load_run_matrix()
    helper_path = tmp_path / "lurestar_runtime.py"
    helper_path.write_text(bootstrap.HELPER_SOURCE)
    helper = _load_generated_helper(helper_path)
    jobs = matrix.build_matrix(
        tmp_path / "lurestar", models=("gpt",), seeds=(1234,), require_configs=False
    )
    fabric = SimpleNamespace(world_size=1, print=lambda _message: None)
    gpt = SimpleNamespace(training_steps=0, get_num_params=lambda non_embedding=False: 21_324_672)

    for job in jobs:
        config = _stargraph_config(pathlib.Path(job.out_root))
        # do_train has already applied upstream's ``-seed{seed}`` suffix when the
        # step-0 hook runs.
        config.trainer.experiment_name = job.experiment_dir_name
        if job.phase == "adapt":
            config.trainer.train_batches = 20_000 + job.train_batches
            gpt.training_steps = 20_000
        else:
            config.trainer.train_batches = job.train_batches
            gpt.training_steps = 0
        receipt = helper.assert_step0_contract(config, gpt, fabric)
        assert receipt["out_dir"] == str(pathlib.Path(job.out_root).resolve())
        assert receipt["seed"] == job.seed


def test_greater_equal_stop_rule_means_exact_requested_optimizer_updates(tmp_path) -> None:
    bootstrap = _load_bootstrap()
    patched = bootstrap._patch_core_train((PINNED / "core_train.py").read_text())
    assert "if self.step >= self.config.trainer.train_batches:" in patched
    assert "if self.step > self.config.trainer.train_batches:" not in patched

    def updates_taken(start: int, target: int) -> int:
        step = start
        updates = 0
        while True:
            # Mirrors the pinned loop: optimizer_step increments model.training_steps,
            # then Trainer increments self.step and applies the patched predicate.
            step += 1
            updates += 1
            if step >= target:
                return updates

    assert updates_taken(0, 20_000) == 20_000
    assert updates_taken(0, 3_000) == 3_000
    assert updates_taken(20_000, 20_500) == 500


def test_fp16_grad_scaler_disables_only_fused_adamw() -> None:
    """The T4 path must retain FP16/grad clipping but avoid optimizer-owned unscale."""
    bootstrap = _load_bootstrap()
    patched = bootstrap._patch_core_train((PINNED / "core_train.py").read_text())

    namespace: dict[str, object] = {}
    exec(
        "def select(fused_available, is_device_cuda, fabric):\n"
        "    amp_scaler = getattr(fabric.strategy.precision, 'scaler', None)\n"
        "    return fused_available and is_device_cuda and amp_scaler is None\n",
        namespace,
    )
    select = namespace["select"]
    fp16 = SimpleNamespace(
        strategy=SimpleNamespace(precision=SimpleNamespace(scaler=object()))
    )
    bf16 = SimpleNamespace(
        strategy=SimpleNamespace(precision=SimpleNamespace(scaler=None))
    )

    assert select(True, True, fp16) is False
    assert select(True, True, bf16) is True
    assert select(True, False, bf16) is False
    assert "grad_clip" in patched
    assert "use_fused = fused_available and is_device_cuda and amp_scaler is None" in patched


def test_verified_pointer_and_two_deep_retention(tmp_path, monkeypatch) -> None:
    bootstrap = _load_bootstrap()
    helper_path = tmp_path / "lurestar_runtime.py"
    helper_path.write_text(bootstrap.HELPER_SOURCE)
    helper = _load_generated_helper(helper_path)

    fake_torch = SimpleNamespace(
        load=lambda path, map_location=None, weights_only=None: pickle.loads(
            pathlib.Path(path).read_bytes()
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    ckpt_dir = tmp_path / "run" / "experiment"
    ckpt_dir.mkdir(parents=True)
    checkpoints = []
    for step in (250, 500, 750):
        path = ckpt_dir / f"recovery_ckpt_iter_{step}.pt"
        path.write_bytes(pickle.dumps({"training_steps": step}))
        metadata = {
            "schema": 1,
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        (path.with_name(path.name + ".meta.json")).write_text(json.dumps(metadata))
        checkpoints.append(path)
    pointer = tmp_path / "run" / "recovery_ckpt"

    helper.publish_verified_pointer(pointer, checkpoints[-1])
    retained = helper.retain_verified_recovery(ckpt_dir, pointer, keep=2)

    assert pointer.read_text() == str(checkpoints[-1].resolve())
    assert retained == [str(checkpoints[2].resolve()), str(checkpoints[1].resolve())]
    assert not checkpoints[0].exists()
    assert checkpoints[1].exists() and checkpoints[2].exists()

    checkpoints[-1].write_bytes(b"torn")
    with pytest.raises(RuntimeError, match="metadata"):
        helper.publish_verified_pointer(pointer, checkpoints[-1])
    assert pointer.read_text() == str(checkpoints[-1].resolve())


def test_atomic_fabric_save_commits_only_a_deserializable_verified_payload(
    tmp_path, monkeypatch
) -> None:
    bootstrap = _load_bootstrap()
    helper_path = tmp_path / "lurestar_runtime.py"
    helper_path.write_text(bootstrap.HELPER_SOURCE)
    helper = _load_generated_helper(helper_path)

    fake_torch = SimpleNamespace(
        load=lambda path, map_location=None, weights_only=None: pickle.loads(
            pathlib.Path(path).read_bytes()
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeFabric:
        world_size = 1
        global_rank = 0

        @staticmethod
        def barrier():
            return None

        @staticmethod
        def save(path, state):
            pathlib.Path(path).write_bytes(pickle.dumps(state))

    checkpoint = tmp_path / "run" / "recovery_ckpt_iter_250.pt"
    helper.atomic_fabric_save(FakeFabric(), checkpoint, {"training_steps": 250})

    assert checkpoint.is_file()
    assert not checkpoint.with_name(checkpoint.name + ".partial").exists()
    metadata = json.loads(
        checkpoint.with_name(checkpoint.name + ".meta.json").read_text()
    )
    assert metadata["training_steps"] == 250
    assert metadata["amp_scaler_state"] is False
    assert metadata["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert pickle.loads(checkpoint.read_bytes()) == {"training_steps": 250}


def test_checkpoint_rng_payload_restores_python_numpy_and_torch(tmp_path) -> None:
    np = pytest.importorskip("numpy")
    bootstrap = _load_bootstrap()
    helper_path = tmp_path / "lurestar_runtime.py"
    helper_path.write_text(bootstrap.HELPER_SOURCE)
    helper = _load_generated_helper(helper_path)

    restored_torch = []
    fake_torch = SimpleNamespace(
        get_rng_state=lambda: b"cpu-rng-at-checkpoint",
        set_rng_state=lambda state: restored_torch.append(state),
        cuda=SimpleNamespace(
            is_available=lambda: False,
            get_rng_state_all=lambda: [],
            set_rng_state_all=lambda _states: None,
            device_count=lambda: 0,
        ),
    )
    random.seed(91)
    np.random.seed(92)
    state = helper.capture_rng_state(fake_torch)
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    random.seed(1)
    np.random.seed(2)

    assert helper.restore_rng_state(state, fake_torch) is True
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    assert restored_torch == [b"cpu-rng-at-checkpoint"]


def test_checkpoint_amp_scaler_payload_roundtrips_and_bf16_is_empty(tmp_path) -> None:
    bootstrap = _load_bootstrap()
    helper_path = tmp_path / "lurestar_runtime.py"
    helper_path.write_text(bootstrap.HELPER_SOURCE)
    helper = _load_generated_helper(helper_path)

    class FakeScaler:
        def __init__(self, scale: float):
            self.scale = scale

        def state_dict(self):
            return {"scale": self.scale, "growth_tracker": 17}

        def load_state_dict(self, state):
            self.scale = state["scale"]

    original = FakeScaler(65536.0)
    fp16 = SimpleNamespace(
        strategy=SimpleNamespace(precision=SimpleNamespace(scaler=original))
    )
    payload = helper.capture_amp_scaler_state(fp16)
    original.scale = 1.0

    assert helper.restore_amp_scaler_state(payload, fp16) is True
    assert original.scale == 65536.0
    with pytest.raises(RuntimeError, match="lacks AMP GradScaler state"):
        helper.restore_amp_scaler_state(None, fp16)

    bf16 = SimpleNamespace(
        strategy=SimpleNamespace(precision=SimpleNamespace(scaler=None))
    )
    assert helper.capture_amp_scaler_state(bf16) is None
    assert helper.restore_amp_scaler_state(None, bf16) is False
    with pytest.raises(RuntimeError, match="runtime precision"):
        helper.restore_amp_scaler_state(payload, bf16)


def test_runtime_sidecar_and_source_identity_flow_through_colab_durability(
    tmp_path, monkeypatch
) -> None:
    bootstrap = _load_bootstrap()
    driver = _load_colab_driver()
    helper_path = tmp_path / "lurestar_runtime.py"
    helper_path.write_text(bootstrap.HELPER_SOURCE)
    helper = _load_generated_helper(helper_path)

    fake_torch = SimpleNamespace(
        load=lambda path, map_location=None, weights_only=None: pickle.loads(
            pathlib.Path(path).read_bytes()
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class FakeFabric:
        world_size = 1
        global_rank = 0

        @staticmethod
        def barrier():
            return None

        @staticmethod
        def save(path, state):
            pathlib.Path(path).write_bytes(pickle.dumps(state))

    class FakeBlob:
        def __init__(self, bucket, name):
            self.bucket = bucket
            self.name = name
            self.metadata = {}
            self.size = None

        def upload_from_filename(self, path):
            self.bucket.payloads[self.name] = pathlib.Path(path).read_bytes()
            self.bucket.metadata[self.name] = dict(self.metadata or {})
            self.reload()

        def upload_from_string(self, payload, content_type=None):
            del content_type
            payload = payload.encode() if isinstance(payload, str) else payload
            self.bucket.payloads[self.name] = bytes(payload)
            self.bucket.metadata[self.name] = dict(self.metadata or {})
            self.reload()

        def download_as_bytes(self):
            return self.bucket.payloads[self.name]

        def download_to_filename(self, path):
            pathlib.Path(path).write_bytes(self.bucket.payloads[self.name])

        def reload(self):
            payload = self.bucket.payloads.get(self.name)
            self.size = len(payload) if payload is not None else None
            self.metadata = dict(self.bucket.metadata.get(self.name, {}))

    class FakeBucket:
        name = "bucket"

        def __init__(self):
            self.payloads = {}
            self.metadata = {}
            self.blobs = {}

        def blob(self, name):
            self.blobs.setdefault(name, FakeBlob(self, name))
            return self.blobs[name]

        def list_blobs(self, prefix):
            return [self.blob(name) for name in sorted(self.payloads) if name.startswith(prefix)]

    root = tmp_path / "lurestar"
    out = root / "runs" / "gpt" / "1234" / "base" / "_"
    experiment = out / "gpt-s1234-base-seed1234"
    experiment.mkdir(parents=True)
    checkpoint = experiment / "recovery_ckpt_iter_250.pt"
    helper.atomic_fabric_save(FakeFabric(), checkpoint, {"training_steps": 250})
    helper.publish_verified_pointer(out / "recovery_ckpt", checkpoint)
    (experiment / "materialized_config.yaml").write_text("seed: 1234\n")
    bucket = FakeBucket()
    durability = driver.RuntimeDurability(
        bucket, root, source_sha256="a" * 64, logger=lambda _message: None
    )

    state = durability.sync_job({
        "job_id": "gpt-s1234-base", "status": "RUNNING", "out_root": str(out.resolve())
    })

    assert state["source_snapshot_sha256"] == "a" * 64
    assert state["checkpoint"]["sha256"] == helper.sha256_file(checkpoint)
    remote_names = set(bucket.payloads)
    assert any(name.endswith("recovery_ckpt_iter_250.pt.meta.json") for name in remote_names)
    assert any(name.endswith("recovery_ckpt_iter_250.pt") for name in remote_names)
