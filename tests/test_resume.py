"""The spec's mandatory recovery test (section 9) plus the durable-layer invariants.

The contract under test: run 300 steps uninterrupted; separately run the same 300-step job,
SIGKILL the child mid-flight, resume, finish at 300, and land in the same place. SIGKILL, not a
clean exception -- a `finally:` cannot run, buffers are not flushed, and a checkpoint may be
halfway written. That is what a Colab disconnect actually looks like.

There is no GPU and no torch on this host, so the trainer under test is
`lurestar.toy_trainer`, which owns the state that matters: Adam moments with a real step
counter, a warmup+cosine scheduler with a real `last_epoch`, a shuffled data position, and two
RNG streams (numpy Generator for dropout and the data permutation, python `random` for label
jitter). `test_upstream_style_reseed_diverges` runs the same experiment with the RNG state
*removed* from the checkpoint -- upstream's actual behaviour, since `fabric.seed_everything`
reseeds on every launch (`train.py:170`) and the checkpoint carries no RNG
(docs/UPSTREAM_REPORT.md section 3.1) -- and asserts that it diverges. Without that test the
passing one would prove nothing: it would pass on shuffled data too.

Measured tolerance: the resumed trajectory is bit-identical, `max |Δparam| == 0.0`.
"""

from __future__ import annotations

import json
import os
import pathlib
import pickle
import signal
import subprocess
import sys
import time

import numpy as np
import pytest

from lurestar.durable_checkpoint import (
    CheckpointCorrupt,
    DurableCheckpointer,
    DurableSync,
    NoValidCheckpoint,
    atomic_write_json,
    pickle_serializer,
    sha256_file,
    verify_pointer,
)
from lurestar.toy_trainer import ToyTrainer

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "src"

TOTAL_STEPS = 300
KILL_AT = 150
SAVE_EVERY = 25
SEED = 7

# The tolerance we actually achieve, not an aspiration. Restoring python + numpy RNG, the Adam
# moments, `last_epoch` and the data cursor makes the resumed trajectory bitwise identical, so
# the tolerance is exact zero. Raise this only with a recorded reason.
PARAM_TOLERANCE = 0.0


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _env() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _toy_cmd(run_dir, *, steps=TOTAL_STEPS, seed=SEED, save_every=SAVE_EVERY,
             step_delay=0.0, no_rng=False, raise_at=None, run_id="toy"):
    cmd = [sys.executable, "-m", "lurestar.toy_trainer",
           "--run-dir", str(run_dir), "--steps", str(steps), "--seed", str(seed),
           "--save-every", str(save_every), "--metrics-every", "25",
           "--step-delay", str(step_delay), "--run-id", run_id]
    if no_rng:
        cmd.append("--no-rng-state")
    if raise_at is not None:
        cmd += ["--raise-at", str(raise_at)]
    return cmd


def run_toy(run_dir, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(_toy_cmd(run_dir, **kw), env=_env(), cwd=str(REPO),
                          capture_output=True, text=True, timeout=300)


def load_final(run_dir) -> tuple[dict, dict]:
    summary = json.loads((pathlib.Path(run_dir) / "final_summary.json").read_text())
    with open(summary["final_ckpt"], "rb") as fh:
        state = pickle.load(fh)
    return summary, state


def params_of(state: dict) -> dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=np.float64) for k, v in state["params"].items()}


def max_param_delta(a: dict, b: dict) -> float:
    pa, pb = params_of(a), params_of(b)
    assert pa.keys() == pb.keys()
    return max(float(np.max(np.abs(pa[k] - pb[k]))) for k in pa)


def metrics_of(run_dir) -> dict[int, dict]:
    out = {}
    for p in (pathlib.Path(run_dir) / "metrics").glob("step_*.json"):
        d = json.loads(p.read_text())
        out[d["step"]] = d
    return out


@pytest.fixture(scope="module")
def reference(tmp_path_factory) -> tuple[pathlib.Path, dict, dict]:
    """The uninterrupted 300-step run every other trajectory is compared against."""
    run_dir = tmp_path_factory.mktemp("reference")
    proc = run_toy(run_dir)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary, state = load_final(run_dir)
    assert summary["step"] == TOTAL_STEPS
    return run_dir, summary, state


# --------------------------------------------------------------------------------------
# the mandatory recovery test
# --------------------------------------------------------------------------------------

def test_uninterrupted_run_is_itself_reproducible(reference, tmp_path):
    """Guard on the guard: if two clean runs already differed, the resume test proves nothing."""
    _, ref_summary, ref_state = reference
    proc = run_toy(tmp_path / "again")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    _, state = load_final(tmp_path / "again")
    assert max_param_delta(ref_state, state) == 0.0


def test_sigkill_at_150_then_resume_reaches_300(reference, tmp_path, capsys):
    """Hard-kill the child mid-training, resume, and land bitwise on the reference.

    SIGKILL is uncatchable: no atexit, no finally, no flush. Whatever is on disk at that
    instant is all the resume gets.
    """
    ref_dir, ref_summary, ref_state = reference
    run_dir = tmp_path / "interrupted"
    run_dir.mkdir()

    proc = subprocess.Popen(
        _toy_cmd(run_dir, step_delay=0.01), env=_env(), cwd=str(REPO),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    progress = run_dir / "progress.json"
    deadline = time.time() + 60
    reached = 0
    while time.time() < deadline:
        if progress.is_file():
            try:
                reached = json.loads(progress.read_text())["step"]
            except (json.JSONDecodeError, OSError):
                pass
            if reached >= KILL_AT:
                break
        if proc.poll() is not None:
            pytest.fail(f"child exited before reaching step {KILL_AT} (reached {reached})")
        time.sleep(0.002)
    assert reached >= KILL_AT, f"never reached step {KILL_AT} (got {reached})"
    assert proc.poll() is None, "child was already gone; nothing was interrupted"

    os.kill(proc.pid, signal.SIGKILL)
    rc = proc.wait(timeout=30)
    assert rc == -signal.SIGKILL, f"expected SIGKILL exit, got {rc}"
    assert not (run_dir / "final_summary.json").exists(), "the killed run must not have finished"

    # What survived the kill must be usable on its own terms.
    ok, why = verify_pointer(run_dir)
    assert ok, why

    resumed = run_toy(run_dir, step_delay=0.0)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "resumed_from_step=" in resumed.stdout
    resumed_from = int(resumed.stdout.split("resumed_from_step=")[1].split()[0])
    assert 0 < resumed_from <= reached, f"resumed from {resumed_from}, killed at {reached}"

    summary, state = load_final(run_dir)

    # step
    assert summary["step"] == TOTAL_STEPS == ref_summary["step"]
    # optimizer state: the Adam step counter and both moment buffers
    assert summary["opt_t"] == ref_summary["opt_t"] == TOTAL_STEPS
    for buf in ("m", "v"):
        for k in ref_state["optimizer"][buf]:
            np.testing.assert_array_equal(
                np.asarray(state["optimizer"][buf][k]),
                np.asarray(ref_state["optimizer"][buf][k]),
            )
    assert state["optimizer"]["lr"] == ref_state["optimizer"]["lr"]
    # scheduler state
    assert state["lr_scheduler_state"] == ref_state["lr_scheduler_state"]
    # data position
    assert (summary["data_epoch"], summary["data_cursor"]) == (
        ref_summary["data_epoch"], ref_summary["data_cursor"])
    np.testing.assert_array_equal(
        np.asarray(state["data"]["perm"]), np.asarray(ref_state["data"]["perm"]))
    # metrics
    ref_metrics, got_metrics = metrics_of(ref_dir), metrics_of(run_dir)
    assert set(got_metrics) == set(ref_metrics)
    for step in sorted(ref_metrics):
        assert got_metrics[step]["loss"] == pytest.approx(ref_metrics[step]["loss"], abs=1e-12), step
    # final parameters
    delta = max_param_delta(ref_state, state)
    with capsys.disabled():
        print(f"\n  [recovery] killed at step {reached}, resumed from checkpoint step "
              f"{resumed_from}, finished at {summary['step']}; "
              f"max |delta param| = {delta:.3e} (tolerance {PARAM_TOLERANCE:.1e})")
    assert delta <= PARAM_TOLERANCE
    assert summary["params_sha256"] == ref_summary["params_sha256"]


def test_upstream_style_reseed_diverges(reference, tmp_path, capsys):
    """Drop the RNG state from the checkpoint and the same experiment must NOT reproduce.

    This is the falsifier for the test above. Upstream's checkpoint carries model, optimizer,
    scheduler and step and nothing else (docs/UPSTREAM_REPORT.md section 3.1), and every launch
    reseeds (`train.py:170`), so this is upstream's real resume behaviour, not a strawman.
    """
    _, ref_summary, ref_state = reference
    run_dir = tmp_path / "nostate"
    first = run_toy(run_dir, no_rng=True, raise_at=KILL_AT)
    assert first.returncode != 0
    second = run_toy(run_dir, no_rng=True)
    assert second.returncode == 0, second.stdout + second.stderr

    summary, state = load_final(run_dir)
    assert summary["step"] == TOTAL_STEPS          # it still *finishes*
    assert summary["opt_t"] == ref_summary["opt_t"]  # and the counters still line up
    delta = max_param_delta(ref_state, state)
    with capsys.disabled():
        print(f"  [recovery] without RNG state in the checkpoint: "
              f"max |delta param| = {delta:.3e}")
    assert delta > 1e-6, (
        "dropping python+numpy RNG from the checkpoint did not change the trajectory, so the "
        "recovery test is not sensitive to RNG restoration and proves nothing"
    )


def test_catchable_interruption_writes_emergency_checkpoint_and_traceback(tmp_path):
    run_dir = tmp_path / "emergency"
    proc = run_toy(run_dir, raise_at=137, save_every=50)
    assert proc.returncode != 0
    assert "KeyboardInterrupt" in proc.stderr

    tb = (run_dir / "traceback.txt").read_text()
    assert "simulated interruption at step 137" in tb
    info = json.loads((run_dir / "INTERRUPTED.json").read_text())
    assert info["step"] == 137
    assert info["checkpoint"]["step"] == 137
    # The emergency checkpoint is a real, verified checkpoint at the exact interrupted step --
    # not the last periodic save at 100.
    assert pathlib.Path(info["checkpoint"]["path"]).is_file()
    assert sha256_file(info["checkpoint"]["path"]) == info["checkpoint"]["sha256"]
    ok, why = verify_pointer(run_dir)
    assert ok, why

    resumed = run_toy(run_dir)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert "resumed_from_step=137" in resumed.stdout


# --------------------------------------------------------------------------------------
# durable-layer invariants
# --------------------------------------------------------------------------------------

@pytest.fixture()
def ck(tmp_path) -> DurableCheckpointer:
    return DurableCheckpointer(tmp_path / "out", "job-1", experiment_name="exp",
                               serializer=pickle_serializer())


def test_two_deep_retention(ck):
    recs = [ck.save({"step": s, "payload": list(range(s))}, s) for s in (10, 20, 30, 40)]
    on_disk = sorted(p.name for p in ck.ckpt_dir.glob("*.pt"))
    assert on_disk == ["recovery_ckpt_iter_30.pt", "recovery_ckpt_iter_40.pt"]
    index = ck.read_index()
    assert [r.step for r in index] == [40, 30]
    assert not pathlib.Path(recs[0].path).exists()
    # the older of the two survivors is only removed once a third is verified
    ck.save({"step": 50}, 50)
    assert not pathlib.Path(recs[2].path).exists()
    assert pathlib.Path(recs[3].path).exists()


def test_corrupt_newest_rolls_back_one(ck):
    ck.save({"step": 100, "v": "old"}, 100)
    newest = ck.save({"step": 200, "v": "new"}, 200)
    assert ck.pointer_path.read_text().strip() == newest.path

    # Truncate the newest in place, exactly as a kill during a non-atomic write would.
    with open(newest.path, "r+b") as fh:
        fh.truncate(17)

    state, rec = ck.load_latest()
    assert rec.step == 100 and state["v"] == "old"
    assert ck.pointer_path.read_text().strip() == rec.path
    assert pathlib.Path(newest.path + ".corrupt").is_file()
    assert [r.step for r in ck.read_index()] == [100]


def test_corrupt_newest_with_valid_length_still_rolls_back(ck):
    """A same-size flip must be caught by the hash, not merely by the size."""
    ck.save({"step": 100, "v": "old"}, 100)
    newest = ck.save({"step": 200, "v": "new"}, 200)
    size = os.path.getsize(newest.path)
    with open(newest.path, "r+b") as fh:
        fh.seek(size // 2)
        b = fh.read(1)
        fh.seek(size // 2)
        fh.write(bytes([b[0] ^ 0xFF]))
    assert os.path.getsize(newest.path) == size
    _, rec = ck.load_latest()
    assert rec.step == 100


def test_all_checkpoints_corrupt_raises_and_clears_pointer(ck):
    a = ck.save({"step": 10}, 10)
    b = ck.save({"step": 20}, 20)
    for r in (a, b):
        with open(r.path, "r+b") as fh:
            fh.truncate(3)
    with pytest.raises(NoValidCheckpoint):
        ck.load_latest()
    # Upstream asserts the pointer target exists (core_train.py:148-150) and hard-fails the
    # whole resume if it does not. Clearing it lets the job restart from scratch instead.
    assert not ck.pointer_path.exists()


def test_partial_file_is_never_loaded(ck):
    good = ck.save({"step": 10, "v": "good"}, 10)
    # A kill mid-write leaves exactly this: a .partial at a higher step, with no sidecar.
    partial = ck.ckpt_dir / "recovery_ckpt_iter_20.pt.partial"
    with open(partial, "wb") as fh:
        pickle.dump({"step": 20, "v": "torn"}, fh)

    state, rec = ck.load_latest()
    assert rec.path == good.path and state["v"] == "good"
    assert not any(r.path.endswith(".partial") for r in ck.read_index())
    ok, why = verify_pointer(ck.out_dir)
    assert ok, why
    # and the sweep removes it once a newer verified checkpoint exists
    ck.save({"step": 30}, 30)
    assert not partial.exists()


def test_pointer_never_points_at_a_file_that_fails_its_hash(ck):
    rec = ck.save({"step": 10}, 10)
    ok, _ = verify_pointer(ck.out_dir)
    assert ok
    with open(rec.path, "ab") as fh:
        fh.write(b"junk")
    ok, why = verify_pointer(ck.out_dir)
    assert not ok and "size" in why, why

    # and a same-length corruption, which only the hash can catch
    size = os.path.getsize(rec.path)
    with open(rec.path, "r+b") as fh:
        fh.truncate(rec.size_bytes)
        fh.seek(0)
        fh.write(b"\x00" * 8)
    assert os.path.getsize(rec.path) == rec.size_bytes
    ok, why = verify_pointer(ck.out_dir)
    assert not ok and "hash" in why, why

    # The checkpointer refuses to publish an unverified record itself.
    bad = ck.read_index()[0]
    with pytest.raises(CheckpointCorrupt):
        ck._write_pointer(bad)


def test_save_that_cannot_be_read_back_does_not_move_the_pointer(tmp_path):
    """A serializer that writes garbage must not be allowed to publish it."""
    good_save, _ = pickle_serializer()

    def broken_save(state, fh):
        fh.write(b"\x00\x01\x02not-a-pickle")

    _, load = pickle_serializer()
    ck = DurableCheckpointer(tmp_path / "o", "j", experiment_name="e",
                             serializer=(good_save, load))
    keeper = ck.save({"step": 1}, 1)
    ck.save_fn = broken_save
    with pytest.raises(CheckpointCorrupt):
        ck.save({"step": 2}, 2)
    assert ck.pointer_path.read_text().strip() == keeper.path
    assert [r.step for r in ck.read_index()] == [1]
    assert not (ck.ckpt_dir / "recovery_ckpt_iter_2.pt").exists()


def test_writes_are_atomic_no_partial_survives(ck):
    ck.save({"step": 10, "blob": "x" * 100_000}, 10)
    assert list(ck.ckpt_dir.glob("*.partial")) == []
    assert list(ck.out_dir.glob("*.partial")) == []


def test_finalize_clears_the_stale_recovery_pointer(ck):
    ck.save({"step": 10}, 10)
    assert ck.pointer_path.is_file()
    ck.finalize()
    assert not ck.pointer_path.exists()


def test_guard_persists_traceback_for_a_bare_exception(tmp_path):
    ck = DurableCheckpointer(tmp_path / "o", "j", experiment_name="e",
                             serializer=pickle_serializer())
    box = {"step": 0}
    with pytest.raises(ZeroDivisionError):
        with ck.guard(lambda: {"step": box["step"]}, lambda: box["step"],
                      emergency_root=tmp_path / "em"):
            box["step"] = 42
            1 / 0
    assert "ZeroDivisionError" in (ck.out_dir / "traceback.txt").read_text()
    assert (tmp_path / "em" / "j" / "INTERRUPTED.json").is_file()
    _, rec = ck.load_latest()
    assert rec.step == 42 and rec.kind == "emergency"


# --------------------------------------------------------------------------------------
# bounded-backoff sync
# --------------------------------------------------------------------------------------

def test_sync_retries_with_bounded_backoff_then_succeeds(tmp_path):
    attempts, slept = [], []
    def flaky(local, bucket, remote):
        attempts.append(remote)
        if len(attempts) < 3:
            raise ConnectionError("transient")
    sync = DurableSync("bkt", "lurestar", "job-1", uploader=flaky,
                       emergency_root=tmp_path / "em", attempts=5,
                       base_delay=1.0, max_delay=4.0, sleep=slept.append, seed=0)
    f = tmp_path / "a.pt"
    f.write_bytes(b"payload")
    res = sync.push(f, "runs/a.pt")
    assert res.ok and res.attempts == 3
    assert len(slept) == 2
    assert slept == sorted(slept) and all(d <= 4.0 * 1.25 for d in slept)
    assert not (tmp_path / "em" / "job-1" / "NEEDS_SYNC").exists()


def test_sync_falls_back_to_local_needs_sync_and_retries_later(tmp_path):
    fail = {"on": True}
    def uploader(local, bucket, remote):
        if fail["on"]:
            raise ConnectionError("no network")
    sync = DurableSync("bkt", "lurestar", "job-1", uploader=uploader,
                       emergency_root=tmp_path / "em", attempts=3,
                       base_delay=0.0, max_delay=0.0, sleep=lambda d: None, seed=1)
    f = tmp_path / "ckpt.pt"
    f.write_bytes(b"weights")
    res = sync.push(f, "runs/ckpt.pt")
    assert not res.ok and res.attempts == 3
    stash = tmp_path / "em" / "job-1" / "runs" / "ckpt.pt"
    assert stash.read_bytes() == b"weights"
    marker = tmp_path / "em" / "job-1" / "NEEDS_SYNC"
    assert "not on GCS" in marker.read_text()
    assert [p["remote_rel"] for p in sync.pending()] == ["runs/ckpt.pt"]

    assert all(not r.ok for r in sync.retry_pending())
    assert marker.is_file()

    fail["on"] = False
    results = sync.retry_pending()
    assert results and all(r.ok for r in results)
    assert sync.pending() == []
    assert not marker.exists()


def test_checkpointer_pushes_through_sync(tmp_path):
    sent = []
    sync = DurableSync("bkt", "lurestar", "job-1", uploader=lambda l, b, r: sent.append(r),
                       emergency_root=tmp_path / "em", sleep=lambda d: None)
    ck = DurableCheckpointer(tmp_path / "o", "job-1", experiment_name="e",
                             serializer=pickle_serializer(), sync=sync)
    ck.save({"step": 5}, 5)
    assert any(r.endswith("recovery_ckpt_iter_5.pt") for r in sent)
    assert any(r.endswith(".meta.json") for r in sent)


# --------------------------------------------------------------------------------------
# in-process trainer checks (fast, no subprocess)
# --------------------------------------------------------------------------------------

def test_in_process_resume_restores_every_piece_of_state(tmp_path):
    a = ToyTrainer(tmp_path / "a", seed=3, total_steps=80, save_every=20, run_id="a")
    a.train()
    ref = a.state()

    b = ToyTrainer(tmp_path / "b", seed=3, total_steps=80, save_every=20, run_id="b")
    b.train(until=40)
    c = ToyTrainer(tmp_path / "b", seed=3, total_steps=80, save_every=20, run_id="b")
    assert c.resume_or_start() == 40
    assert c.opt.t == 40 and c.sched.last_epoch == 40
    c.train()
    got = c.state()

    assert got["step"] == ref["step"] == 80
    assert got["optimizer"]["t"] == ref["optimizer"]["t"]
    assert got["lr_scheduler_state"] == ref["lr_scheduler_state"]
    assert got["data"]["cursor"] == ref["data"]["cursor"]
    assert got["data"]["epoch"] == ref["data"]["epoch"]
    assert max_param_delta(ref, got) == PARAM_TOLERANCE


def test_resume_refuses_a_changed_dataset(tmp_path):
    a = ToyTrainer(tmp_path / "a", seed=3, total_steps=40, save_every=20, run_id="a")
    a.train(until=20)
    b = ToyTrainer(tmp_path / "a", seed=99, total_steps=40, save_every=20, run_id="a")
    with pytest.raises(RuntimeError, match="dataset fingerprint changed"):
        b.resume_or_start()
