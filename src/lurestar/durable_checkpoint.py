"""Durable checkpoint / resume layer for the Colab interruption contract (spec section 9).

The premise is that the GPU runtime disconnects mid-write. Upstream's checkpointing does not
survive that, and the gaps are concrete, not hypothetical (docs/UPSTREAM_REPORT.md section 3.5):

  * `models/model_base.py:417` writes the `.pt` in place via `fabric.save`, and the pointer files
    are a plain `open(..., "w")` (`core_train.py:944-948`, `core_train.py:970-974`). A kill during
    either leaves a truncated checkpoint, a pointer aimed at it, or both.
  * `core_train.py:976-979` deletes the previous recovery checkpoint *before* anything has verified
    the new one, so at the moment of deletion there is exactly one copy and it is unverified.
  * `core_train.py:334` keeps the recovery path in memory only, so after a resume the pre-crash file
    is never collected and the disk leaks one checkpoint per interruption.
  * `models/model_base.py:440-456` re-loads the whole file to pull the scheduler state and swallows
    every exception, so a truncated checkpoint resumes *silently* with no scheduler state.
  * `core_train.py:145-150` makes `recovery_ckpt` strictly win over `latest_ckpt` with no step
    comparison, and a finished run leaves that pointer aimed at a deleted file, which hard-fails the
    next resume on the `assert os.path.isfile`.

So this layer owns write atomicity, verification, retention and the pointer. Every checkpoint is
written to `<name>.partial`, flushed, fsynced, and only then renamed into place; it is then hashed
*and deserialized* before it is allowed into the index; the pointer is only ever rewritten to a
record that has passed that check; and the oldest of the two retained checkpoints is deleted only
after the newest has been loaded and hashed. `resolve()` re-verifies at read time and rolls back one
checkpoint if the newest is corrupt.

Nothing here imports torch. The serializer is injected, so the same code runs under pytest on a
CPU-only laptop and under `fabric run` on Colab.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
import os
import pathlib
import pickle
import random
import shutil
import time
import traceback
import typing as t

__all__ = [
    "CheckpointCorrupt",
    "CheckpointRecord",
    "DurableCheckpointer",
    "DurableSync",
    "NoValidCheckpoint",
    "SyncResult",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_text",
    "default_serializer",
    "sha256_file",
    "verify_pointer",
]

PARTIAL_SUFFIX = ".partial"
CORRUPT_SUFFIX = ".corrupt"
META_SUFFIX = ".meta.json"
INDEX_NAME = "durable_index.json"
RECOVERY_POINTER = "recovery_ckpt"   # core_train.py:971
LATEST_POINTER = "latest_ckpt"       # core_train.py:945
DEFAULT_EMERGENCY_ROOT = "/content/lurestar_emergency"
NEEDS_SYNC_MARKER = "NEEDS_SYNC"


class CheckpointCorrupt(RuntimeError):
    """A checkpoint file failed its hash, its size, or its deserialization."""


class NoValidCheckpoint(RuntimeError):
    """No retained checkpoint survived verification."""


# --------------------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------------------

def sha256_file(path: os.PathLike | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fsync_dir(directory: os.PathLike | str) -> None:
    """fsync the directory entry so a rename survives a power/kernel-level loss.

    A rename is atomic with respect to readers immediately, but is not durable until the
    containing directory is synced. Not every filesystem lets you open a directory; failing
    to sync is not a reason to fail the write.
    """
    try:
        fd = os.open(os.fspath(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path: os.PathLike | str, data: bytes, *, fsync: bool = True) -> pathlib.Path:
    """Write bytes to `path` via `path.partial` + fsync + atomic rename."""
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + PARTIAL_SUFFIX)
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        if fsync:
            os.fsync(f.fileno())
    os.replace(tmp, p)
    if fsync:
        _fsync_dir(p.parent)
    return p


def atomic_write_text(path: os.PathLike | str, text: str, *, fsync: bool = True) -> pathlib.Path:
    return atomic_write_bytes(path, text.encode("utf-8"), fsync=fsync)


def atomic_write_json(path: os.PathLike | str, obj: t.Any, *, fsync: bool = True) -> pathlib.Path:
    return atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True) + "\n", fsync=fsync)


def _write_stream_atomic(
    path: pathlib.Path,
    writer: t.Callable[[io.BufferedWriter], None],
    *,
    fsync: bool = True,
) -> pathlib.Path:
    """Same contract as `atomic_write_bytes`, but the payload is produced by a callback.

    Used for checkpoints, where the serializer (torch.save / pickle.dump) wants a file object
    and the payload is far too large to materialize twice in memory.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + PARTIAL_SUFFIX)
    with open(tmp, "wb") as f:
        writer(f)
        f.flush()
        if fsync:
            os.fsync(f.fileno())
    os.replace(tmp, path)
    if fsync:
        _fsync_dir(path.parent)
    return path


# --------------------------------------------------------------------------------------
# serializers
# --------------------------------------------------------------------------------------

def pickle_serializer() -> tuple[t.Callable, t.Callable]:
    def save(state, fh):
        pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def load(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)

    return save, load


def torch_serializer() -> tuple[t.Callable, t.Callable]:
    import torch  # noqa: PLC0415 - deliberately lazy; the analysis host has no torch

    def save(state, fh):
        torch.save(state, fh)

    def load(path):
        # weights_only=False matches upstream's own load (model_base.py:435). These are our
        # own files; never point this at a third-party checkpoint.
        return torch.load(path, map_location="cpu", weights_only=False)

    return save, load


def default_serializer() -> tuple[t.Callable, t.Callable]:
    """torch when it is importable (Colab), pickle otherwise (the analysis host)."""
    try:
        return torch_serializer()
    except Exception:
        return pickle_serializer()


# --------------------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class CheckpointRecord:
    path: str
    step: int
    sha256: str
    size_bytes: int
    saved_at: float
    kind: str = "recovery"
    run_id: str = ""
    extra: dict = dataclasses.field(default_factory=dict)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "CheckpointRecord":
        known = {f.name for f in dataclasses.fields(CheckpointRecord)}
        return CheckpointRecord(**{k: v for k, v in d.items() if k in known})


def verify_pointer(out_dir: os.PathLike | str, pointer: str = RECOVERY_POINTER) -> tuple[bool, str]:
    """Check the invariant `the pointer never points at a file that fails its hash`.

    Returns (ok, reason). A missing pointer is ok: `core_train.py:164-168` treats that as
    "start from scratch", which is the correct state for a job that has not checkpointed yet.
    """
    out = pathlib.Path(out_dir)
    ptr = out / pointer
    if not ptr.is_file():
        return True, "no pointer"
    target = ptr.read_text().strip()
    if not target:
        return False, "pointer is empty"
    if target.endswith(PARTIAL_SUFFIX):
        return False, f"pointer targets a partial file: {target}"
    tp = pathlib.Path(target)
    if not tp.is_file():
        return False, f"pointer targets a missing file: {target}"
    meta_path = tp.with_name(tp.name + META_SUFFIX)
    if not meta_path.is_file():
        return False, f"pointer target has no sidecar hash: {target}"
    meta = json.loads(meta_path.read_text())
    if tp.stat().st_size != meta["size_bytes"]:
        return False, f"pointer target size {tp.stat().st_size} != recorded {meta['size_bytes']}"
    actual = sha256_file(tp)
    if actual != meta["sha256"]:
        return False, f"pointer target hash {actual[:12]} != recorded {meta['sha256'][:12]}"
    return True, "ok"


# --------------------------------------------------------------------------------------
# the checkpointer
# --------------------------------------------------------------------------------------

class DurableCheckpointer:
    """Two-deep verified retention over upstream's on-disk layout.

    Layout (upstream's, unchanged, so `init_from: resume` still resolves --
    `core_train.py:139-151`)::

        {out_dir}/
          recovery_ckpt              one-line pointer, ours is always absolute
          latest_ckpt                one-line pointer
          durable_index.json         ours: the verified records, newest first
          {experiment_name}/
            recovery_ckpt_iter_{step}.pt
            recovery_ckpt_iter_{step}.pt.meta.json
    """

    def __init__(
        self,
        out_dir: os.PathLike | str,
        run_id: str,
        *,
        experiment_name: str | None = None,
        keep: int = 2,
        serializer: tuple[t.Callable, t.Callable] | None = None,
        sync: "DurableSync | None" = None,
        clock: t.Callable[[], float] = time.time,
        logger: t.Callable[[str], None] | None = None,
        fsync: bool = True,
    ) -> None:
        if keep < 2:
            raise ValueError("spec section 9.2 item 4 requires two verified recovery checkpoints")
        self.out_dir = pathlib.Path(out_dir).resolve()
        self.run_id = run_id
        self.experiment_name = experiment_name or run_id
        self.keep = keep
        self.save_fn, self.load_fn = serializer or default_serializer()
        self.sync = sync
        self.clock = clock
        self.fsync = fsync
        self._log = logger if logger is not None else (lambda m: None)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- paths ----------------------------------------------------------------------
    @property
    def ckpt_dir(self) -> pathlib.Path:
        # core_train.py:929-933 joins out_dir with experiment_name
        return self.out_dir / self.experiment_name

    @property
    def index_path(self) -> pathlib.Path:
        return self.out_dir / INDEX_NAME

    @property
    def pointer_path(self) -> pathlib.Path:
        return self.out_dir / RECOVERY_POINTER

    @property
    def latest_pointer_path(self) -> pathlib.Path:
        return self.out_dir / LATEST_POINTER

    # ---- index ----------------------------------------------------------------------
    def read_index(self) -> list[CheckpointRecord]:
        if not self.index_path.is_file():
            return []
        try:
            doc = json.loads(self.index_path.read_text())
        except (json.JSONDecodeError, OSError):
            # The index is written atomically, so this means the file was clobbered by
            # something outside this layer. Treat it as empty rather than crashing the run;
            # resolve() will rebuild from whatever sidecars survive.
            self._log("durable: index unreadable, rebuilding from sidecars")
            return self._rebuild_index_from_sidecars()
        return [CheckpointRecord.from_dict(r) for r in doc.get("records", [])]

    def _rebuild_index_from_sidecars(self) -> list[CheckpointRecord]:
        recs = []
        for meta in sorted(self.ckpt_dir.glob("*" + META_SUFFIX)):
            try:
                recs.append(CheckpointRecord.from_dict(json.loads(meta.read_text())))
            except (json.JSONDecodeError, OSError, TypeError):
                continue
        recs.sort(key=lambda r: (r.step, r.saved_at), reverse=True)
        return recs

    def _write_index(self, records: list[CheckpointRecord]) -> None:
        atomic_write_json(
            self.index_path,
            {
                "schema": 1,
                "run_id": self.run_id,
                "keep": self.keep,
                "records": [r.to_dict() for r in records],
            },
            fsync=self.fsync,
        )

    # ---- save -----------------------------------------------------------------------
    def save(
        self,
        state: t.Any,
        step: int,
        *,
        kind: str = "recovery",
        filename: str | None = None,
        extra: dict | None = None,
        update_pointer: bool = True,
    ) -> CheckpointRecord:
        """Atomically write, verify, index, point at, and then prune.

        The order is the whole point:

        1. serialize into `.partial`, flush, fsync, rename, fsync the directory;
        2. hash the renamed file and deserialize it -- a checkpoint that cannot be read back
           is not a checkpoint;
        3. write the sidecar hash;
        4. add to the index (the index only ever holds verified records);
        5. rewrite the pointer atomically;
        6. only now delete anything older than the two we keep.

        A kill between (1) and (4) loses the newest checkpoint and keeps the previous one,
        which is the correct failure. A kill between (4) and (5) leaves a stale pointer, which
        `resolve()` reconciles from the index.
        """
        name = filename or f"{kind}_ckpt_iter_{step}.pt"
        path = self.ckpt_dir / name

        _write_stream_atomic(path, lambda fh: self.save_fn(state, fh), fsync=self.fsync)

        digest = sha256_file(path)
        size = path.stat().st_size
        try:
            self.load_fn(path)
        except Exception as exc:  # noqa: BLE001 - any deserialization failure is disqualifying
            path.unlink(missing_ok=True)
            raise CheckpointCorrupt(
                f"checkpoint {path} did not survive a read-back: {exc!r}"
            ) from exc

        rec = CheckpointRecord(
            path=str(path),
            step=int(step),
            sha256=digest,
            size_bytes=size,
            saved_at=self.clock(),
            kind=kind,
            run_id=self.run_id,
            extra=dict(extra or {}),
        )
        atomic_write_json(
            path.with_name(path.name + META_SUFFIX), rec.to_dict(), fsync=self.fsync
        )

        records = [r for r in self.read_index() if r.path != rec.path]
        records.insert(0, rec)
        records.sort(key=lambda r: (r.step, r.saved_at), reverse=True)
        self._write_index(records)

        if update_pointer:
            self._write_pointer(rec)

        self.prune()
        self._log(f"durable: saved step {step} -> {path.name} sha {digest[:12]}")

        if self.sync is not None:
            rel = os.path.relpath(path, self.out_dir)
            self.sync.push(path, f"{self.run_id}/{rel}")
            self.sync.push(
                path.with_name(path.name + META_SUFFIX), f"{self.run_id}/{rel}{META_SUFFIX}"
            )
        return rec

    def _write_pointer(self, rec: CheckpointRecord) -> None:
        """Rewrite the upstream pointer atomically, always to an absolute path.

        Upstream writes whatever `os.path.join(out_dir, experiment_name, filename)` produced
        (`core_train.py:944-948`), so a relative `out_dir` -- the shipped default is
        `output/stargraph` (`gpt_stargraph_5_5.yaml:14`) -- yields a pointer that only resolves
        from the original CWD. On Colab that is a guaranteed resume failure.
        """
        ok, reason = self._verify_record(rec)
        if not ok:
            raise CheckpointCorrupt(f"refusing to point at an unverified checkpoint: {reason}")
        atomic_write_text(self.pointer_path, str(pathlib.Path(rec.path).resolve()), fsync=self.fsync)

    # ---- verify / resolve -----------------------------------------------------------
    def _verify_record(self, rec: CheckpointRecord, *, deep: bool = True) -> tuple[bool, str]:
        p = pathlib.Path(rec.path)
        if p.name.endswith(PARTIAL_SUFFIX):
            return False, f"{p.name} is a partial file"
        if not p.is_file():
            return False, f"{p} is missing"
        size = p.stat().st_size
        if size != rec.size_bytes:
            return False, f"{p.name} size {size} != recorded {rec.size_bytes}"
        digest = sha256_file(p)
        if digest != rec.sha256:
            return False, f"{p.name} sha {digest[:12]} != recorded {rec.sha256[:12]}"
        if deep:
            try:
                self.load_fn(p)
            except Exception as exc:  # noqa: BLE001
                return False, f"{p.name} failed to deserialize: {exc!r}"
        return True, "ok"

    def resolve(self, *, deep: bool = True) -> CheckpointRecord | None:
        """Newest VALID checkpoint, quarantining anything that fails on the way.

        This is the rollback: if the newest record does not verify it is renamed to
        `.corrupt`, dropped from the index, and the next-newest is tried. Returning `None`
        means the job has no usable checkpoint and must start from scratch.
        """
        records = self.read_index()
        survivors: list[CheckpointRecord] = []
        chosen: CheckpointRecord | None = None
        for rec in sorted(records, key=lambda r: (r.step, r.saved_at), reverse=True):
            if chosen is None:
                ok, reason = self._verify_record(rec, deep=deep)
                if not ok:
                    self._log(f"durable: rolling back past {rec.path}: {reason}")
                    self._quarantine(rec)
                    continue
                chosen = rec
            survivors.append(rec)

        if len(survivors) != len(records):
            self._write_index(survivors)

        if chosen is None:
            # Nothing in the index survived. Clear the pointer so upstream's
            # `assert os.path.isfile` (core_train.py:148-150) does not hard-fail the resume;
            # with no pointer it falls through to a scratch init (core_train.py:164-168).
            self.pointer_path.unlink(missing_ok=True)
            return None

        # Reconcile the pointer: a kill between the index write and the pointer write leaves
        # the pointer stale, and the index is authoritative.
        target = str(pathlib.Path(chosen.path).resolve())
        current = self.pointer_path.read_text().strip() if self.pointer_path.is_file() else None
        if current != target:
            atomic_write_text(self.pointer_path, target, fsync=self.fsync)
        return chosen

    def _quarantine(self, rec: CheckpointRecord) -> None:
        p = pathlib.Path(rec.path)
        if p.is_file():
            dest = p.with_name(p.name + CORRUPT_SUFFIX)
            with contextlib.suppress(OSError):
                os.replace(p, dest)
        with contextlib.suppress(OSError):
            p.with_name(p.name + META_SUFFIX).unlink(missing_ok=True)

    def load_latest(self, *, deep: bool = True) -> tuple[t.Any, CheckpointRecord]:
        rec = self.resolve(deep=deep)
        if rec is None:
            raise NoValidCheckpoint(f"no verified checkpoint under {self.out_dir}")
        return self.load_fn(rec.path), rec

    # ---- retention -------------------------------------------------------------------
    def prune(self) -> list[str]:
        """Keep `self.keep` verified checkpoints; delete the rest. Also sweep dead partials.

        Called only after the newest checkpoint has been hashed and read back, which is the
        difference from `core_train.py:976-979`, where the previous recovery file is removed
        before anything has looked at the new one.
        """
        records = sorted(self.read_index(), key=lambda r: (r.step, r.saved_at), reverse=True)
        keep, drop = records[: self.keep], records[self.keep :]
        removed = []
        for rec in drop:
            p = pathlib.Path(rec.path)
            # os.remove is unguarded upstream (core_train.py:979); a missing file there kills
            # the run. missing_ok makes retention idempotent across resumes.
            with contextlib.suppress(OSError):
                p.unlink(missing_ok=True)
            with contextlib.suppress(OSError):
                p.with_name(p.name + META_SUFFIX).unlink(missing_ok=True)
            removed.append(rec.path)
        if drop:
            self._write_index(keep)

        newest_step = keep[0].step if keep else -1
        for partial in self.ckpt_dir.glob("*" + PARTIAL_SUFFIX):
            step = _step_from_name(partial.name)
            if step is not None and step <= newest_step:
                with contextlib.suppress(OSError):
                    partial.unlink()
                    removed.append(str(partial))
        return removed

    def finalize(self, *, keep_pointer: bool = False) -> None:
        """Clear the recovery pointer when a job completes.

        `_save_recovery_checkpoint` deletes the previous recovery file but never clears the
        pointer, so a finished run leaves `recovery_ckpt` aimed at a file that later retention
        removed -- and because `recovery_ckpt` strictly beats `latest_ckpt` with no step
        comparison (`core_train.py:145-151`), the next `init_from: resume` hard-fails on the
        assert at `core_train.py:148-150`. Clearing it at the end is the fix.
        """
        if not keep_pointer:
            self.pointer_path.unlink(missing_ok=True)

    # ---- interruption ----------------------------------------------------------------
    def emergency_save(
        self,
        state: t.Any,
        step: int,
        exc: BaseException | None = None,
        *,
        emergency_root: os.PathLike | str | None = None,
    ) -> CheckpointRecord | None:
        """Best-effort checkpoint + persisted traceback on a catchable interruption.

        SIGTERM (Colab's preemption signal, if a handler turns it into an exception),
        KeyboardInterrupt and any exception escaping the training loop all land here. This
        must never raise: the emergency path failing is not a reason to lose the traceback.
        """
        tb = "".join(traceback.format_exception(exc)) if exc is not None else "".join(
            traceback.format_stack()
        )
        info = {
            "run_id": self.run_id,
            "step": int(step),
            "when": self.clock(),
            "exception": repr(exc) if exc is not None else None,
        }
        rec = None
        try:
            rec = self.save(state, step, kind="emergency")
        except Exception as save_exc:  # noqa: BLE001
            tb += f"\n\nemergency checkpoint ALSO failed: {save_exc!r}\n"
            info["emergency_save_failed"] = repr(save_exc)

        root = pathlib.Path(emergency_root or DEFAULT_EMERGENCY_ROOT) / self.run_id
        for target_dir in (self.out_dir, root):
            try:
                atomic_write_text(target_dir / "traceback.txt", tb, fsync=self.fsync)
                atomic_write_json(
                    target_dir / "INTERRUPTED.json",
                    dict(info, checkpoint=rec.to_dict() if rec else None),
                    fsync=self.fsync,
                )
            except OSError:
                # /content does not exist off Colab; that is not a failure worth propagating.
                continue

        if rec is not None and self.sync is not None:
            with contextlib.suppress(Exception):
                self.sync.push(pathlib.Path(rec.path), f"{self.run_id}/emergency/{pathlib.Path(rec.path).name}")
        return rec

    @contextlib.contextmanager
    def guard(
        self,
        state_fn: t.Callable[[], t.Any],
        step_fn: t.Callable[[], int],
        *,
        emergency_root: os.PathLike | str | None = None,
    ):
        """Wrap a training loop so any catchable interruption leaves a checkpoint behind.

        BaseException, not Exception: KeyboardInterrupt and SystemExit are exactly the two
        that a notebook interrupt and a preemption handler raise.
        """
        try:
            yield self
        except BaseException as exc:
            self.emergency_save(state_fn(), step_fn(), exc, emergency_root=emergency_root)
            raise


def _step_from_name(name: str) -> int | None:
    stem = name.split(".")[0]
    for token in reversed(stem.split("_")):
        if token.isdigit():
            return int(token)
    return None


# --------------------------------------------------------------------------------------
# durable sync
# --------------------------------------------------------------------------------------

@dataclasses.dataclass
class SyncResult:
    local_path: str
    remote: str
    ok: bool
    attempts: int
    error: str | None = None
    fallback_path: str | None = None


class DurableSync:
    """Bounded-backoff push to GCS with a local NEEDS_SYNC fallback (spec section 9.2 item 6).

    Auth on Colab is settled: the credential is an `authorized_user` ADC uploaded to
    `/content/adc.json`, read by google-cloud-storage from GOOGLE_APPLICATION_CREDENTIALS.
    Service-account keys are blocked by `constraints/iam.disableServiceAccountKeyCreation`,
    so there is no key path to fall back to -- the fallback is local, and it is a queue.

    `uploader` and `sleep` are injected so the retry ladder is testable without a network and
    without wall-clock time. Jitter comes from an explicit Generator, never the global RNG.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str,
        run_id: str,
        *,
        uploader: t.Callable[[str, str, str], None] | None = None,
        emergency_root: os.PathLike | str = DEFAULT_EMERGENCY_ROOT,
        attempts: int = 5,
        base_delay: float = 1.0,
        max_delay: float = 60.0,
        sleep: t.Callable[[float], None] = time.sleep,
        seed: int = 0,
        logger: t.Callable[[str], None] | None = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.run_id = run_id
        self.uploader = uploader or _gcs_upload
        self.emergency_root = pathlib.Path(emergency_root) / run_id
        self.attempts = max(1, attempts)
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.sleep = sleep
        self._rng = random.Random(seed)
        self._log = logger if logger is not None else (lambda m: None)

    @property
    def queue_path(self) -> pathlib.Path:
        return self.emergency_root / (NEEDS_SYNC_MARKER + ".json")

    def _remote(self, remote_rel: str) -> str:
        return f"{self.prefix}/{remote_rel.lstrip('/')}"

    def push(self, local_path: os.PathLike | str, remote_rel: str) -> SyncResult:
        local = pathlib.Path(local_path)
        remote = self._remote(remote_rel)
        last_error: str | None = None
        for attempt in range(1, self.attempts + 1):
            try:
                self.uploader(str(local), self.bucket, remote)
                self._log(f"sync: {local.name} -> gs://{self.bucket}/{remote}")
                return SyncResult(str(local), remote, True, attempt)
            except Exception as exc:  # noqa: BLE001 - any transport failure retries
                last_error = repr(exc)
                if attempt == self.attempts:
                    break
                delay = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
                delay *= 1.0 + self._rng.random() * 0.25  # decorrelate concurrent runs
                self._log(f"sync: attempt {attempt} failed ({last_error}); sleeping {delay:.2f}s")
                self.sleep(delay)

        fallback = self._stash(local, remote_rel)
        self._log(f"sync: giving up on {local.name}, stashed at {fallback} and marked NEEDS_SYNC")
        return SyncResult(str(local), remote, False, self.attempts, last_error, str(fallback))

    def _stash(self, local: pathlib.Path, remote_rel: str) -> pathlib.Path:
        dest = self.emergency_root / remote_rel.lstrip("/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if local.is_file():
            shutil.copyfile(local, dest.with_name(dest.name + PARTIAL_SUFFIX))
            os.replace(dest.with_name(dest.name + PARTIAL_SUFFIX), dest)
        pending = self.pending()
        entry = {"local": str(dest), "remote_rel": remote_rel, "queued_at": time.time()}
        pending = [p for p in pending if p["remote_rel"] != remote_rel] + [entry]
        atomic_write_json(self.queue_path, {"schema": 1, "run_id": self.run_id, "pending": pending})
        atomic_write_text(
            self.emergency_root / NEEDS_SYNC_MARKER,
            f"{len(pending)} artifact(s) are not on GCS; run scripts/run_matrix.py --retry-sync\n",
        )
        return dest

    def pending(self) -> list[dict]:
        if not self.queue_path.is_file():
            return []
        try:
            return json.loads(self.queue_path.read_text()).get("pending", [])
        except (json.JSONDecodeError, OSError):
            return []

    def retry_pending(self) -> list[SyncResult]:
        """Drain the NEEDS_SYNC queue. Anything that still fails stays queued."""
        results = []
        still_pending = []
        for entry in self.pending():
            local = pathlib.Path(entry["local"])
            if not local.is_file():
                continue  # the stash was cleaned up; nothing recoverable to send
            remote = self._remote(entry["remote_rel"])
            try:
                self.uploader(str(local), self.bucket, remote)
                results.append(SyncResult(str(local), remote, True, 1))
            except Exception as exc:  # noqa: BLE001
                results.append(SyncResult(str(local), remote, False, 1, repr(exc), str(local)))
                still_pending.append(entry)
        atomic_write_json(
            self.queue_path, {"schema": 1, "run_id": self.run_id, "pending": still_pending}
        )
        marker = self.emergency_root / NEEDS_SYNC_MARKER
        if still_pending:
            atomic_write_text(marker, f"{len(still_pending)} artifact(s) still not on GCS\n")
        else:
            marker.unlink(missing_ok=True)
        return results


def _gcs_upload(local_path: str, bucket: str, remote: str) -> None:
    """Upload via google-cloud-storage, reading the ADC from GOOGLE_APPLICATION_CREDENTIALS.

    Verified on an L4 runtime (docs/RUNLOG.md): the python client picks the `authorized_user`
    ADC straight out of the env var. The gcloud CLI needs an access token minted in-process
    into CLOUDSDK_AUTH_ACCESS_TOKEN instead; that path is not used here.
    """
    from google.cloud import storage  # noqa: PLC0415 - lazy, optional dependency

    client = storage.Client()
    client.bucket(bucket).blob(remote).upload_from_filename(local_path)
