"""`scripts/launch_train.sh` had no tests at all before this file.

It is the only path by which a confirmatory job reaches a GPU, and four of its five
behaviours are refusals -- the cases where upstream would otherwise do the wrong thing
silently (`core_train.py:164-168` builds a SCRATCH model when an `init_from: resume` job has
no pointer; `train.py:176-178` asserts a datamodule that does not exist at the pinned commit;
`core_train.py:148-150` hard-fails on a dangling recovery pointer). A refusal that is never
executed in a test is a comment.

Everything here runs the real script with `DRY_RUN=1` against a fake repo, so no GPU, no
`fabric`, and no torch are involved. Each test names the wrong-input case it rejects.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LAUNCH = str(REPO / "scripts" / "launch_train.sh")
UPSTREAM = REPO / "upstream" / "NextLat"


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """A working copy that looks enough like the pinned repo for the preconditions."""
    work = tmp_path / "nextlat"
    work.mkdir()
    shutil.copy(UPSTREAM / "train.py", work / "train.py")
    shutil.copy(UPSTREAM / "defaults.yaml", work / "defaults.yaml")
    return work


def run(repo: Path, root: Path, *args: str, **env_extra) -> subprocess.CompletedProcess:
    env = dict(os.environ, DRY_RUN="1", NEXTLAT_REPO=str(repo), LURESTAR_ROOT=str(root))
    env.update({k: str(v) for k, v in env_extra.items()})
    return subprocess.run(["bash", LAUNCH, *args], capture_output=True, text=True, env=env)


# --------------------------------------------------------------------------------------
# the command it builds
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("config,model,tail", [
    ("gpt_lurestar.yaml", "gpt", "runs/gpt/seed1234/base"),
    ("nextlat_lurestar.yaml", "nextlat", "runs/nextlat/seed1234/base"),
    ("bst_lurestar.yaml", "bst", "runs/bst/seed1234/base"),
    ("gpt_hmm.yaml", "gpt", "runs/hmm/gpt/seed1234/base"),
    ("nextlat_hmm.yaml", "nextlat", "runs/hmm/nextlat/seed1234/base"),
])
def test_emits_the_spec_section_8_single_gpu_command(
    repo: Path, tmp_path: Path, config: str, model: str, tail: str
) -> None:
    """Spec section 8, verbatim:
    `fabric run --devices 1 --precision bf16-mixed train.py --config <config.yaml>`."""
    if config.endswith("hmm.yaml"):
        # the HMM guard is exercised separately; register the datamodule so we can see the
        # command this job would run
        (repo / "train.py").write_text(
            (repo / "train.py").read_text().replace(
                '"stargraph": StarGraphDataModule,',
                '"stargraph": StarGraphDataModule,\n    "hmm_belief": None,'))
    root = tmp_path / "root"
    proc = run(repo, root, config, "1234")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = proc.stdout
    assert "fabric run --devices 1 --precision bf16-mixed" in out
    assert "--strategy" not in out, (
        "spec section 8 launches one device with no strategy; LURESTAR_STRATEGY is opt-in"
    )
    assert f"--config {REPO / 'configs' / config}" in out
    assert "seed=1234" in out
    assert f"trainer.out_dir={root / tail}" in out
    assert f"# model/seed  {model} / 1234" in out


def test_strategy_is_opt_in(repo: Path, tmp_path: Path) -> None:
    proc = run(repo, tmp_path / "root", "gpt_lurestar.yaml", "1234", LURESTAR_STRATEGY="ddp")
    assert proc.returncode == 0, proc.stderr
    assert "--strategy ddp" in proc.stdout


def test_seed_reaches_the_out_dir_and_the_experiment_name(repo: Path, tmp_path: Path) -> None:
    """train.py:96-97 appends '-seed{n}' only when the name lacks 'seed', and the resume
    pointer lives at out_dir (core_train.py:140-141), so both must carry the seed."""
    for seed in (1234, 1235, 1236):
        proc = run(repo, tmp_path / "root", "nextlat_lurestar.yaml", str(seed))
        assert proc.returncode == 0, proc.stderr
        assert f"trainer.experiment_name=nextlat-seed{seed}-base" in proc.stdout
        assert f"seed{seed}/base" in proc.stdout


def test_the_gpt_adaptation_branch_flips_the_model_flag_and_gets_its_own_root(
    repo: Path, tmp_path: Path
) -> None:
    """configs/adapt_*.yaml are derived from the NextLat YAML, so the GPT branch is the same
    file with `use_nextlat=false`; its output root must NOT be the NextLat branch's."""
    root = tmp_path / "root"
    parent = tmp_path / "parent.pt"
    parent.write_text("x")
    gpt = run(repo, root, "adapt_near.yaml", "1234",
              LURESTAR_MODEL="gpt", LURESTAR_PARENT_CKPT=str(parent))
    nl = run(repo, root, "adapt_near.yaml", "1234",
             LURESTAR_MODEL="nextlat", LURESTAR_PARENT_CKPT=str(parent))
    assert gpt.returncode == 0 and nl.returncode == 0, gpt.stderr + nl.stderr
    assert "use_nextlat=false" in gpt.stdout
    assert "use_nextlat=false" not in nl.stdout
    assert f"trainer.out_dir={root / 'runs/gpt/seed1234/adapt-near'}" in gpt.stdout
    assert f"trainer.out_dir={root / 'runs/nextlat/seed1234/adapt-near'}" in nl.stdout


# --------------------------------------------------------------------------------------
# the refusals  (each names the wrong input it rejects)
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("seed", ["1237", "1238", "0", "9999"])
def test_refuses_a_seed_that_is_not_preregistered(repo: Path, tmp_path: Path, seed: str) -> None:
    """The shipped sweep is [1234..1238]; PROGRAM.md freezes the first three."""
    proc = run(repo, tmp_path / "root", "gpt_lurestar.yaml", seed)
    assert proc.returncode == 2
    assert "not preregistered" in proc.stderr
    assert "fabric run" not in proc.stdout


def test_the_non_confirmatory_escape_hatch_is_explicit(repo: Path, tmp_path: Path) -> None:
    proc = run(repo, tmp_path / "root", "gpt_lurestar.yaml", "9999",
               LURESTAR_ALLOW_ANY_SEED="1")
    assert proc.returncode == 0, proc.stderr
    assert "seed=9999" in proc.stdout


def test_refuses_a_non_integer_seed(repo: Path, tmp_path: Path) -> None:
    proc = run(repo, tmp_path / "root", "gpt_lurestar.yaml", "1234a")
    assert proc.returncode == 2 and "seed must be an integer" in proc.stderr


def test_refuses_an_unknown_config(repo: Path, tmp_path: Path) -> None:
    proc = run(repo, tmp_path / "root", "gpt_stargraph_5_5.yaml", "1234")
    assert proc.returncode == 2
    assert "config not found" in proc.stderr or "unknown config" in proc.stderr


def test_refuses_a_first_adaptation_launch_with_no_parent(repo: Path, tmp_path: Path) -> None:
    """core_train.py:164-168 -- with init_from: resume and no pointer, upstream prints two
    'Could not find' lines and trains a fresh random network on 5,000 items."""
    proc = run(repo, tmp_path / "root", "adapt_far.yaml", "1234", LURESTAR_MODEL="nextlat")
    assert proc.returncode == 2
    assert "LURESTAR_PARENT_CKPT" in proc.stderr and "from scratch" in proc.stderr


def test_refuses_a_parent_checkpoint_that_does_not_exist(repo: Path, tmp_path: Path) -> None:
    proc = run(repo, tmp_path / "root", "adapt_near.yaml", "1234",
               LURESTAR_MODEL="gpt", LURESTAR_PARENT_CKPT=str(tmp_path / "gone.pt"))
    assert proc.returncode == 2 and "does not exist" in proc.stderr


def test_refuses_an_adaptation_launch_with_no_model(repo: Path, tmp_path: Path) -> None:
    proc = run(repo, tmp_path / "root", "adapt_near.yaml", "1234")
    assert proc.returncode == 2 and "LURESTAR_MODEL" in proc.stderr


def test_refuses_a_stale_recovery_pointer(repo: Path, tmp_path: Path) -> None:
    """core_train.py:148-150 asserts the file the pointer names exists, so a pointer to a
    deleted checkpoint kills the job after the GPU is already allocated."""
    root = tmp_path / "root"
    out = root / "runs" / "nextlat" / "seed1235" / "adapt-far"
    out.mkdir(parents=True)
    (out / "recovery_ckpt").write_text(str(tmp_path / "deleted.pt"))
    proc = run(repo, root, "adapt_far.yaml", "1235", LURESTAR_MODEL="nextlat")
    assert proc.returncode == 2 and "stale recovery pointer" in proc.stderr


def test_refuses_an_hmm_job_until_the_datamodule_is_registered(repo: Path, tmp_path: Path) -> None:
    """`hmm_belief` is not in DATAMODULES at the pinned commit and train.py:176-178 asserts
    membership, so the job would die after the runtime is up."""
    assert "hmm_belief" not in (repo / "train.py").read_text()
    proc = run(repo, tmp_path / "root", "gpt_hmm.yaml", "1234")
    assert proc.returncode == 2
    assert "hmm_belief" in proc.stderr and "DATAMODULES" in proc.stderr


def test_a_missing_repo_is_diagnosed_as_a_missing_repo(tmp_path: Path) -> None:
    """Regression: the HMM grep used to run before the train.py existence check, so a
    missing NEXTLAT_REPO was reported as 'has no hmm_belief entry in DATAMODULES'."""
    env = dict(os.environ, DRY_RUN="1", NEXTLAT_REPO=str(tmp_path / "nope"),
               LURESTAR_ROOT=str(tmp_path / "root"))
    proc = subprocess.run(["bash", LAUNCH, "gpt_hmm.yaml", "1234"],
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 2
    assert "no train.py under NEXTLAT_REPO" in proc.stderr
    assert "hmm_belief" not in proc.stderr


def test_a_repo_without_defaults_yaml_is_refused(repo: Path, tmp_path: Path) -> None:
    """train.py:348 does OmegaConf.load('defaults.yaml') relative to the CWD."""
    (repo / "defaults.yaml").unlink()
    proc = run(repo, tmp_path / "root", "gpt_lurestar.yaml", "1234")
    assert proc.returncode == 2 and "defaults.yaml" in proc.stderr


# --------------------------------------------------------------------------------------
# DRY_RUN must be a preview, not a decision
# --------------------------------------------------------------------------------------


def test_dry_run_never_writes_a_resume_pointer(repo: Path, tmp_path: Path) -> None:
    """The pointer write used to happen before the DRY_RUN early exit. A dry run that seeds
    {out_dir}/latest_ckpt makes the next REAL launch print 'ignoring LURESTAR_PARENT_CKPT'
    and adapt from whatever the dry run named -- a silently wrong H3 parent."""
    root = tmp_path / "root"
    wrong = tmp_path / "wrong-parent.pt"
    wrong.write_text("x")
    proc = run(repo, root, "adapt_near.yaml", "1234",
               LURESTAR_MODEL="gpt", LURESTAR_PARENT_CKPT=str(wrong))
    assert proc.returncode == 0, proc.stderr
    assert "would seed" in proc.stdout
    assert not root.exists(), f"dry run created {sorted(p.name for p in root.rglob('*'))}"
    assert "ignoring LURESTAR_PARENT_CKPT" not in proc.stderr


def test_a_refused_launch_leaves_no_run_directory(repo: Path, tmp_path: Path) -> None:
    root = tmp_path / "root"
    proc = run(repo, root, "adapt_near.yaml", "1234", LURESTAR_MODEL="nextlat")
    assert proc.returncode == 2
    assert not root.exists()
