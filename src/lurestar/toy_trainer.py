"""A small real trainer used to exercise the durable layer without a GPU.

This is test scaffolding, not science. It exists because the spec's mandatory recovery test
(section 9, "Mandatory recovery test") has to run on a machine with no CUDA and no torch, and a
recovery test against a trainer with no optimizer momentum, no scheduler, no shuffled data
position and no RNG consumption would pass on shuffled data -- which makes it not a test.

So the loop owns exactly the state that upstream's checkpoint is missing
(docs/UPSTREAM_REPORT.md section 3.1: python / numpy / torch-CPU / CUDA RNG and the dataloader
position are all absent):

  * AdamW-style moments with bias correction and a real step counter `t`;
  * a LambdaLR-equivalent warmup+cosine scheduler with a real `last_epoch`;
  * a shuffling data stream whose per-epoch permutation is drawn from a numpy Generator, so the
    data position is only reproducible if the Generator's bit state is checkpointed;
  * a dropout mask drawn from that same Generator, and a per-step label-noise draw from python's
    global `random` -- the two RNG streams the spec names.

`--no-rng-state` drops the RNG states from the checkpoint and reseeds on resume, which is what
upstream effectively does (`fabric.seed_everything`, train.py:170). The test asserts that this
mode *diverges*; if it did not, the RNG state would not be load-bearing and the passing test
would be vacuous.

Run directly to be killed by a parent process::

    python -m lurestar.toy_trainer --run-dir /tmp/r --steps 300 --seed 7
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import signal
import sys
import time

import numpy as np

from lurestar.durable_checkpoint import (
    DurableCheckpointer,
    NoValidCheckpoint,
    atomic_write_json,
    pickle_serializer,
)

IN_DIM, HIDDEN, OUT_DIM = 16, 24, 8
DROPOUT_P = 0.2


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------

def make_dataset(seed: int, n: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """A deterministic function of the seed alone, so a resume never regenerates it wrongly.

    Spec section 9: "Never regenerate stimuli during resume." Here the dataset is cheap and
    seed-pure, so it is rebuilt rather than checkpointed -- but its hash goes into the
    checkpoint and is checked on resume, which is the property that actually matters.
    """
    rng = np.random.default_rng(seed ^ 0x5EED)
    x = rng.standard_normal((n, IN_DIM))
    w = rng.standard_normal((IN_DIM, OUT_DIM)) / math.sqrt(IN_DIM)
    y = np.tanh(x @ w) + 0.05 * rng.standard_normal((n, OUT_DIM))
    return x, y


def dataset_fingerprint(x: np.ndarray, y: np.ndarray) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(np.ascontiguousarray(x).tobytes())
    h.update(np.ascontiguousarray(y).tobytes())
    return h.hexdigest()


class DataStream:
    """Shuffled epoch-wise sampler. Its position is (epoch, cursor, permutation)."""

    def __init__(self, n: int, batch_size: int, rng: np.random.Generator,
                 *, init_perm: bool = True) -> None:
        self.n = n
        self.batch_size = batch_size
        self.rng = rng
        self.epoch = 0
        self.cursor = 0
        # `init_perm=False` on resume. Drawing the first permutation here would consume the
        # Generator we just restored, which silently forks the trajectory -- this exact bug
        # cost one debugging round and is the reason `data position` is a separate assertion
        # in the recovery test rather than an implied consequence of restoring the RNG.
        self.perm = rng.permutation(n) if init_perm else np.arange(n)

    def next_batch(self) -> np.ndarray:
        if self.cursor + self.batch_size > self.n:
            self.epoch += 1
            self.cursor = 0
            self.perm = self.rng.permutation(self.n)
        idx = self.perm[self.cursor : self.cursor + self.batch_size]
        self.cursor += self.batch_size
        return idx

    def state_dict(self) -> dict:
        return {
            "epoch": self.epoch,
            "cursor": self.cursor,
            "perm": self.perm.tolist(),
            "n": self.n,
            "batch_size": self.batch_size,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.epoch = sd["epoch"]
        self.cursor = sd["cursor"]
        self.perm = np.asarray(sd["perm"], dtype=np.int64)
        self.n = sd["n"]
        self.batch_size = sd["batch_size"]


# --------------------------------------------------------------------------------------
# optimizer + scheduler
# --------------------------------------------------------------------------------------

class Adam:
    """Real Adam with decoupled weight decay and bias correction; `t` is genuine state."""

    def __init__(self, params: dict[str, np.ndarray], lr: float = 1e-2,
                 betas: tuple[float, float] = (0.9, 0.95), eps: float = 1e-8,
                 weight_decay: float = 0.1) -> None:
        self.params = params
        self.base_lr = lr
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        bc1 = 1.0 - self.b1**self.t
        bc2 = 1.0 - self.b2**self.t
        for k, p in self.params.items():
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / bc1
            vhat = self.v[k] / bc2
            p -= self.lr * (mhat / (np.sqrt(vhat) + self.eps) + self.weight_decay * p)

    def state_dict(self) -> dict:
        return {
            "m": {k: v.tolist() for k, v in self.m.items()},
            "v": {k: v.tolist() for k, v in self.v.items()},
            "t": self.t,
            "lr": self.lr,
            "base_lr": self.base_lr,
            "betas": [self.b1, self.b2],
            "eps": self.eps,
            "weight_decay": self.weight_decay,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.m = {k: np.asarray(v, dtype=np.float64) for k, v in sd["m"].items()}
        self.v = {k: np.asarray(v, dtype=np.float64) for k, v in sd["v"].items()}
        self.t = sd["t"]
        self.lr = sd["lr"]
        self.base_lr = sd["base_lr"]
        self.b1, self.b2 = sd["betas"]
        self.eps = sd["eps"]
        self.weight_decay = sd["weight_decay"]


class WarmupCosine:
    """LambdaLR-equivalent. `last_epoch` is stepped exactly like torch's, so it is real state."""

    def __init__(self, opt: Adam, warmup: int = 30, total: int = 300, floor: float = 0.1) -> None:
        self.opt = opt
        self.warmup = warmup
        self.total = total
        self.floor = floor
        self.last_epoch = -1
        self.step()

    def _lam(self, e: int) -> float:
        if e < self.warmup:
            return (e + 1) / max(1, self.warmup)
        prog = (e - self.warmup) / max(1, self.total - self.warmup)
        prog = min(1.0, prog)
        return self.floor + (1 - self.floor) * 0.5 * (1 + math.cos(math.pi * prog))

    def step(self) -> None:
        self.last_epoch += 1
        self.opt.lr = self.opt.base_lr * self._lam(self.last_epoch)

    def state_dict(self) -> dict:
        return {"last_epoch": self.last_epoch, "warmup": self.warmup,
                "total": self.total, "floor": self.floor}

    def load_state_dict(self, sd: dict) -> None:
        self.last_epoch = sd["last_epoch"]
        self.warmup = sd["warmup"]
        self.total = sd["total"]
        self.floor = sd["floor"]
        self.opt.lr = self.opt.base_lr * self._lam(self.last_epoch)


# --------------------------------------------------------------------------------------
# trainer
# --------------------------------------------------------------------------------------

class ToyTrainer:
    def __init__(self, run_dir: os.PathLike | str, *, seed: int = 1234, total_steps: int = 300,
                 batch_size: int = 32, save_every: int = 25, metrics_every: int = 25,
                 include_rng: bool = True, run_id: str = "toy", step_delay: float = 0.0,
                 keep: int = 2) -> None:
        self.run_dir = pathlib.Path(run_dir).resolve()
        self.seed = seed
        self.total_steps = total_steps
        self.batch_size = batch_size
        self.save_every = save_every
        self.metrics_every = metrics_every
        self.include_rng = include_rng
        self.run_id = run_id
        self.step_delay = step_delay
        self.step = 0

        self.ckpt = DurableCheckpointer(
            self.run_dir, run_id, experiment_name="toy", keep=keep,
            serializer=pickle_serializer(), logger=None,
        )
        self.x, self.y = make_dataset(seed)
        self.fingerprint = dataset_fingerprint(self.x, self.y)
        self._init_fresh()

    # ---- state ----------------------------------------------------------------------
    def _init_fresh(self) -> None:
        random.seed(self.seed)
        self.np_rng = np.random.default_rng(self.seed)
        init = np.random.default_rng(self.seed + 1)
        self.params = {
            "W1": init.standard_normal((IN_DIM, HIDDEN)) / math.sqrt(IN_DIM),
            "b1": np.zeros(HIDDEN),
            "W2": init.standard_normal((HIDDEN, OUT_DIM)) / math.sqrt(HIDDEN),
            "b2": np.zeros(OUT_DIM),
        }
        self.opt = Adam(self.params)
        self.sched = WarmupCosine(self.opt, total=self.total_steps)
        self.data = DataStream(len(self.x), self.batch_size, self.np_rng)
        self.step = 0

    def state(self) -> dict:
        sd = {
            "schema": 1,
            "run_id": self.run_id,
            "step": self.step,
            "seed": self.seed,
            "params": {k: v.tolist() for k, v in self.params.items()},
            "optimizer": self.opt.state_dict(),
            "lr_scheduler_state": self.sched.state_dict(),
            "data": self.data.state_dict(),
            "dataset_sha256": self.fingerprint,
            "include_rng": self.include_rng,
        }
        if self.include_rng:
            # Exactly the states docs/UPSTREAM_REPORT.md section 3.1 records as absent from the
            # upstream checkpoint. On Colab this dict also carries torch.get_rng_state() and
            # torch.cuda.get_rng_state_all().
            sd["rng"] = {
                "python": random.getstate(),
                "numpy": self.np_rng.bit_generator.state,
            }
        return sd

    def load_state(self, sd: dict) -> None:
        if sd["dataset_sha256"] != self.fingerprint:
            raise RuntimeError(
                "dataset fingerprint changed across resume: "
                f"{sd['dataset_sha256'][:12]} != {self.fingerprint[:12]}"
            )
        self.step = sd["step"]
        self.params = {k: np.asarray(v, dtype=np.float64) for k, v in sd["params"].items()}
        self.opt = Adam(self.params)
        self.opt.load_state_dict(sd["optimizer"])
        self.sched = WarmupCosine(self.opt, total=self.total_steps)
        self.sched.load_state_dict(sd["lr_scheduler_state"])
        if "rng" in sd:
            random.setstate(_retuple(sd["rng"]["python"]))
            self.np_rng = np.random.default_rng()
            self.np_rng.bit_generator.state = sd["rng"]["numpy"]
        else:
            # Upstream's behaviour: reseed and hope. train.py:170 calls seed_everything(seed)
            # on every launch, resume included.
            random.seed(self.seed)
            self.np_rng = np.random.default_rng(self.seed)
        self.data = DataStream(len(self.x), self.batch_size, self.np_rng, init_perm=False)
        self.data.load_state_dict(sd["data"])

    # ---- compute --------------------------------------------------------------------
    def _loss_and_grads(self, idx: np.ndarray) -> tuple[float, dict[str, np.ndarray]]:
        xb, yb = self.x[idx], self.y[idx]
        # Dropout consumes the numpy Generator; label noise consumes python's global random.
        # Both must be restored exactly or the trajectory forks.
        mask = (self.np_rng.random((len(idx), HIDDEN)) > DROPOUT_P) / (1.0 - DROPOUT_P)
        jitter = (random.random() - 0.5) * 0.02
        pre = xb @ self.params["W1"] + self.params["b1"]
        h = np.tanh(pre) * mask
        out = h @ self.params["W2"] + self.params["b2"]
        diff = out - (yb + jitter)
        loss = float(np.mean(diff**2))
        g_out = 2.0 * diff / diff.size
        grads = {
            "W2": h.T @ g_out,
            "b2": g_out.sum(0),
        }
        g_h = g_out @ self.params["W2"].T
        g_pre = g_h * mask * (1.0 - np.tanh(pre) ** 2)
        grads["W1"] = xb.T @ g_pre
        grads["b1"] = g_pre.sum(0)
        return loss, grads

    # ---- loop -----------------------------------------------------------------------
    def resume_or_start(self) -> int:
        try:
            state, rec = self.ckpt.load_latest()
        except NoValidCheckpoint:
            return 0
        self.load_state(state)
        return rec.step

    def write_metrics(self, loss: float) -> None:
        """Atomic metrics/step_{step}.json keyed by (run_id, step) -- spec section 9.3 item 5."""
        atomic_write_json(
            self.run_dir / "metrics" / f"step_{self.step}.json",
            {"run_id": self.run_id, "step": self.step, "loss": loss,
             "lr": self.opt.lr, "opt_t": self.opt.t,
             "data_epoch": self.data.epoch, "data_cursor": self.data.cursor},
        )

    def write_progress(self, loss: float) -> None:
        atomic_write_json(
            self.run_dir / "progress.json",
            {"run_id": self.run_id, "step": self.step, "loss": loss, "pid": os.getpid()},
        )

    def train(self, until: int | None = None, raise_at: int | None = None) -> dict:
        target = self.total_steps if until is None else until
        with self.ckpt.guard(self.state, lambda: self.step,
                             emergency_root=self.run_dir / "emergency"):
            while self.step < target:
                idx = self.data.next_batch()
                loss, grads = self._loss_and_grads(idx)
                self.opt.step(grads)
                self.sched.step()
                self.step += 1

                if self.step % self.metrics_every == 0 or self.step == target:
                    self.write_metrics(loss)
                self.write_progress(loss)
                if self.save_every and (self.step % self.save_every == 0):
                    self.ckpt.save(self.state(), self.step)
                if raise_at is not None and self.step == raise_at:
                    raise KeyboardInterrupt(f"simulated interruption at step {self.step}")
                if self.step_delay:
                    time.sleep(self.step_delay)

        final = self.ckpt.save(self.state(), self.step, kind="final")
        summary = {
            "run_id": self.run_id, "step": self.step, "final_ckpt": final.path,
            "final_sha256": final.sha256, "params_sha256": _params_hash(self.params),
            "opt_t": self.opt.t, "sched_last_epoch": self.sched.last_epoch,
            "data_epoch": self.data.epoch, "data_cursor": self.data.cursor,
        }
        atomic_write_json(self.run_dir / "final_summary.json", summary)
        self.ckpt.finalize()
        return summary


def _retuple(obj):
    """json/pickle round-trips turn random.getstate()'s tuples into lists; put them back."""
    if isinstance(obj, list):
        return tuple(_retuple(o) for o in obj)
    return obj


def _params_hash(params: dict[str, np.ndarray]) -> str:
    import hashlib

    h = hashlib.sha256()
    for k in sorted(params):
        h.update(k.encode())
        h.update(np.ascontiguousarray(params[k], dtype=np.float64).tobytes())
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--metrics-every", type=int, default=25)
    ap.add_argument("--step-delay", type=float, default=0.0)
    ap.add_argument("--run-id", default="toy")
    ap.add_argument("--no-rng-state", action="store_true",
                    help="omit python/numpy RNG from the checkpoint (upstream's behaviour)")
    ap.add_argument("--raise-at", type=int, default=None,
                    help="raise a catchable interruption at this step")
    a = ap.parse_args(argv)

    # Colab sends SIGTERM before it reclaims a runtime. Turning it into KeyboardInterrupt is
    # what makes the interruption catchable, which is what the emergency checkpoint needs.
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("SIGTERM")))

    tr = ToyTrainer(
        a.run_dir, seed=a.seed, total_steps=a.steps, batch_size=a.batch_size,
        save_every=a.save_every, metrics_every=a.metrics_every,
        include_rng=not a.no_rng_state, run_id=a.run_id, step_delay=a.step_delay,
    )
    resumed_from = tr.resume_or_start()
    print(f"[toy] run_id={a.run_id} resumed_from_step={resumed_from} target={a.steps}", flush=True)
    summary = tr.train(raise_at=a.raise_at)
    print(f"[toy] finished at step {summary['step']} params={summary['params_sha256'][:12]}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
