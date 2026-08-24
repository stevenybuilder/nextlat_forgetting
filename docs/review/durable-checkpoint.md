# Adversarial review — durable-checkpoint track

Reviewer: independent agent. Date: 2026-08-23.
Scope: `src/lurestar/durable_checkpoint.py`, `src/lurestar/toy_trainer.py`,
`scripts/run_matrix.py`, `tests/test_resume.py`, `tests/test_run_matrix.py`, `pytest.ini`,
and the `docs/RUNLOG.md` entry claiming the track.

Verdict summary: **the durable primitive is sound and its tests are real; the runner that is
supposed to use it is not wired to the trainer it will actually launch.** Every one of the
35 reported tests reproduces. All of them exercise a fake launcher or a numpy toy trainer
that writes through `DurableCheckpointer` itself. Against upstream's real on-disk output the
runner (a) deletes the pointer that would have made a resume possible, (b) then plans
`init_from=scratch`, (c) cannot mark any job `DONE`, (d) runs with an identity guard that is
a no-op, and (e) emits an H3 branch command that performs zero adaptation steps on a
NextLat model for the GPT arm.

---

## 0. Reproduction of the reported result

Files: all six deliverables exist and are non-trivial (durable_checkpoint.py 776 lines,
run_matrix.py 646, toy_trainer.py 434, test_resume.py 517, test_run_matrix.py 356,
pytest.ini 4). Nothing described was missing.

```
$ .venv/bin/python -m pytest tests/test_resume.py tests/test_run_matrix.py -v
...
tests/test_resume.py::test_sigkill_at_150_then_resume_reaches_300
  [recovery] killed at step 150, resumed from checkpoint step 125, finished at 300; max |delta param| = 0.000e+00 (tolerance 0.0e+00)
PASSED
tests/test_resume.py::test_upstream_style_reseed_diverges   [recovery] without RNG state in the checkpoint: max |delta param| = 7.012e-02
PASSED
============================== 35 passed in 9.98s ==============================
```

Reproduced exactly, including both printed numbers. The recovery test is **not** science
theater: `test_sigkill_at_150_then_resume_reaches_300` asserts `0 < resumed_from <= reached`
(test_resume.py:183), which is the assertion that would fail if the resume silently restarted
from scratch — and a from-scratch restart would otherwise still land bit-identically on the
reference, so without that line the test would prove nothing. The falsifier
(`test_upstream_style_reseed_diverges`) is real and its divergence is genuine.

Attempts to make passing tests pass on input they should reject: I truncated a checkpoint, flipped
one byte at constant length, hand-wrote a `.partial` at a higher step, corrupted the pointer
target both by size and at constant length, and pointed a serializer at garbage. Each was
caught, by the assertion the implementer claimed catches it. **The durable primitive's tests
survive adversarial input.** The findings below are about what is *not* covered.

---

## P0-1 — `resolve()` deletes a valid upstream recovery pointer

**Severity: P0.** `src/lurestar/durable_checkpoint.py:483-488`.

```python
        if chosen is None:
            self.pointer_path.unlink(missing_ok=True)
            return None
```

`records = self.read_index()` is empty whenever this layer has not itself written a
checkpoint for the job — which is every real job, because upstream's `fabric.save`
(`models/model_base.py:417`) writes no `durable_index.json` and no sidecar. The loop then
never executes, `chosen` stays `None`, and the code deletes `{out_dir}/recovery_ckpt` —
upstream's own pointer (`core_train.py:968-974`), which was aimed at a perfectly good
checkpoint.

This is worse than "not integrated": the layer actively destroys the resume path it was
written to protect. `finalize()`'s legitimate pointer-clearing (a stale pointer to a deleted
file hard-fails `core_train.py:148-150`) has been generalised into "clear the pointer whenever
I don't recognise it".

Reproduction (`scratchpad/repro1.py`, run verbatim):

```
BEFORE: recovery_ckpt exists = True ->  .../gpt-s1234-base/recovery_ckpt_iter_15000.pt
resolve() returned: None
AFTER : recovery_ckpt exists = False
upstream checkpoint file still on disk = True
```

**Fix:** clear the pointer only when this layer has records that all failed verification, or
when the pointer is *dangling* (its target does not exist, which is the case
`core_train.py:148-150` hard-fails on). Never on a merely-unrecognised pointer.

---

## P0-2 — nothing adopts upstream's checkpoints, so every real resume plans `scratch`

**Severity: P0.** `scripts/run_matrix.py:419-453` (`MatrixRunner.plan`), `:491-527`
(post-launch verification); `durable_checkpoint.py` has no adoption entry point at all.

`plan()` derives `fresh` from `ck.resolve()`, and `resolve()` reads only
`durable_index.json`. A record enters that index in exactly one place —
`DurableCheckpointer.save()` (`durable_checkpoint.py:404-411`) — and `save()` is called by
nothing except `toy_trainer.py:365` and the test suite's `FakeLauncher`
(`tests/test_run_matrix.py:61`). The production launcher is `FabricLauncher`, which shells
out to upstream `train.py`; upstream writes its checkpoints itself and knows nothing about
the index.

Consequences on the first real Colab disconnect, all confirmed in the same repro:

```
PLAN  : fresh = True  init_from = scratch  resume_step = 0
```

1. An interrupted 20,000-step job **restarts from step 0**, discarding hours of A100 time.
   Spec §9.3 item 3 ("Resume incomplete jobs from the newest valid checkpoint") is unmet.
2. After a *successful* run, `final = ck.resolve()` at `run_matrix.py:492` returns `None`, so
   `run_matrix.py:519-526` records `FAILED — "job exited 0 but left no verified checkpoint"`.
   **No real job can ever reach `DONE`.** Spec §9.3 items 2 and 6 are unmet.
3. `hash_artifacts(out_root, ("final_summary.json",))` (`run_matrix.py:62`, `:507`) looks for
   a file only `toy_trainer.py:378` ever writes. Upstream produces `ckpt_iter_*.pt`,
   `materialized_config.yaml` and `version_N/metrics.csv`, never `final_summary.json`.

The claim "closes UPSTREAM_REPORT §3.5 items 1–4" is true of the primitive and false of the
system: upstream's non-atomic write, its single unverified recovery copy, and its leaked
recovery file are all still exactly what happens on the GPU, because our layer never sees
those files.

**Fix:** give `DurableCheckpointer` an `adopt()` / `adopt_existing()` that hashes,
deserialization-verifies, sidecars and indexes a checkpoint written by someone else, and call
it in `MatrixRunner.plan()` and before the post-launch `resolve()`. Adopted records must not
be pruned (deleting upstream's `latest`/`best` checkpoints is not ours to do). Also fix
`_step_from_name` (see P2-1) or adoption reads upstream's validation checkpoints as step 0.

---

## P0-3 — the default matrix points at configs that do not exist, and gives near and far the same one

**Severity: P0.** `scripts/run_matrix.py:122-125`.

```python
        def config_for(model, phase, condition):  # noqa: ARG001
            suffix = "base" if phase == "base" else "adapt"
            return str(_REPO / "configs" / f"{model}_lurestar_{suffix}.yaml")
```

```
configs referenced (exist?): [('gpt_lurestar_base.yaml', False),
                              ('gpt_lurestar_adapt.yaml', False),
                              ('gpt_lurestar_adapt.yaml', False)]
```

The six real deliverables are `gpt_lurestar.yaml`, `nextlat_lurestar.yaml`,
`adapt_near.yaml`, `adapt_far.yaml`, `gpt_hmm.yaml`, `nextlat_hmm.yaml`. So `--config` would
point at a nonexistent file and `OmegaConf.load` dies before a single step. Worse, the
`# noqa: ARG001` acknowledges that `condition` is deliberately ignored: **near and far are
handed the same config**, which is precisely the H3 contrast being collapsed at the
configuration layer. `validate_matrix` protects the output roots and then the config
selection undoes the point of protecting them.

**Fix:** `base -> configs/{model}_lurestar.yaml`, `adapt -> configs/adapt_{condition}.yaml`,
and make `build_matrix` refuse to return a job whose config file is missing.

---

## P0-4 — the identity guard is a no-op for any key recorded as `None`

**Severity: P0.** `scripts/run_matrix.py:411`.

```python
            if key in prior and prior[key] is not None and prior[key] != now[key]:
```

`_identity` (`run_matrix.py:399`) records `config_sha256: None` when the config file is
absent — which, given P0-3, is every job. The guard then skips `config_sha256` forever: the
first ledger entry pins `None`, and no later config can ever be seen as a change. Spec §9.3
item 4 ("Preserve config, seed, manifest, and output root") is silently weakened rather than
met or declared unmet. This is the exact failure mode PROGRAM.md's frozen surface exists to
prevent — a run could resume under a config with a different `train_batches` or learning rate
and the ledger would not notice.

Reproduction (`scratchpad/repro2.py`):

```
first entry config_sha256 = None
guard accepted a config that appeared out of nowhere: trainer:
  train_batches: 999999
  lr: 9.9
```

Compounding it, `manifest_sha256` is `{}` for every job because `main()`
(`run_matrix.py:617-620`) calls `build_matrix` without the `manifests=` argument that
`build_matrix` accepts (`run_matrix.py:112`, `:136`, `:146`). The manifest half of the guard
has no input at all.

**Fix:** treat "recorded `None`, now a value" as a change; require the config to exist at
`_identity` time; wire the real manifests (`manifests/corpus.sha256`,
`manifests/corpus_provenance.json`) into the default matrix.

---

## P0-5 — the H3 branch command performs zero adaptation steps, on the wrong model for GPT

**Severity: P0.** `scripts/run_matrix.py:335-351`.

```python
        if plan.parent_checkpoint and plan.fresh:
            cmd += ["--checkpoint_path", plan.parent_checkpoint]
        ...
            f"trainer.train_batches={spec.train_batches}",   # 500 for an adapt job
```

`--checkpoint_path` restores `training_steps` from the parent
(`models/model_base.py:437`), the trainer seeds `self.step` from it
(`core_train.py:309`), and the loop returns as soon as
`self.step > config.trainer.train_batches` (`core_train.py:569`). With a parent at 20,000
steps and `train_batches=500`, the run fast-forwards 20,001 batches
(`core_train.py:432-452`, ~10 M lines re-tokenized) and then **returns after one increment
without a single adaptation update**. `docs/UPSTREAM_REPORT.md` §3.4 names this trap
verbatim — *"it also restores `training_steps`, so the adaptation run's step counter starts
at 20001 and `trainer.train_batches` must be set accordingly"* — and the code walks into it.
`tests/test_run_matrix.py:349` (`assert "trainer.train_batches=500" in fresh`) enshrines the
bug as the contract.

Second defect in the same command: `configs/adapt_near.yaml:20` and `adapt_far.yaml` carry
`use_nextlat: true` (they are derived from the NextLat G(5,5) YAML, whose key set is a
superset). `scripts/launch_train.sh:78` adds `use_nextlat=false` for the GPT branch;
`FabricLauncher.command` does not. **Every GPT adaptation job would train a NextLat model.**

**Fix:** carry the parent's step count into `ResumePlan`, emit
`trainer.train_batches = parent_steps + adapt_steps` for a branch (on resume as well as on
the fresh launch, since the branch's own checkpoints also carry the offset step counter),
refuse to emit a branch command when `parent_steps` is unknown, and add `use_nextlat=false`
to the GPT adapt overrides. Update the test that pinned the wrong number.

---

## P1-1 — RUNLOG presents a numpy toy result as the spec's mandatory recovery test

**Severity: P1.** `docs/RUNLOG.md` §"Durable checkpoint / resume contract".

> **The mandatory recovery test passes bitwise.** 300 steps uninterrupted; then the same
> 300-step job, `SIGKILL`ed at step 150 …

The trainer under test is `src/lurestar/toy_trainer.py` — a 16→24→8 numpy MLP on 512
synthetic rows. `test_resume.py`'s own docstring is honest about this; the RUNLOG entry is
not, and the RUNLOG is what a reader treats as the record. Spec §9's mandatory recovery test
is about the job that produces scientific results, and spec §10's stop condition
("interrupted training cannot resume reproducibly enough for the stated analysis") is
evaluated against the real 12L/6H/384 model. `docs/FOUNDATIONS.md:175` (D-23) explicitly
conditions the `--strategy ddp --devices 1` sampler decision on running this test for real:
*"Verify empirically with the 300 vs 150+150 test before trusting it"*. That verification has
not happened. Claiming the mandatory test passed is a rung-inflation: it is R2 evidence for
the durable primitive, R0 for the trainer the spec means.

**Fix:** amend the RUNLOG entry to name the surrogate and record the real-trainer test as
outstanding. (Applied — docs are on PROGRAM.md's mutable surface.)

## P1-2 — two contradictory launch paths for the same jobs

**Severity: P1.** `scripts/run_matrix.py:326-352` vs `scripts/launch_train.sh:128-136`.

| | `FabricLauncher` | `launch_train.sh` |
|---|---|---|
| `out_dir` | `runs/{model}/{seed}/{base/_ \| adapt/{cond}}` | `runs/{model}/seed{SEED}/{base \| adapt-{cond}}` |
| `experiment_name` | `gpt-s1234-base` | `gpt-seed1234-base` |
| `--strategy` | `ddp` by default | empty by default |
| branch mechanism | `--checkpoint_path` (non-rebased parent) | seeds `{out_dir}/latest_ckpt` with a *step-rebased* parent |

Whichever path is used, the other's ledger keys, pointer locations and checkpoint directories
are wrong, and `DurableCheckpointer(out_root, …)` will look in a directory that does not
exist. The `--strategy` disagreement matters scientifically: `FOUNDATIONS.md:175` (D-23)
adopts `ddp` precisely because it fixes the sampler order across resumes; `launch_train.sh`
defaults it off. These must converge on one path before any confirmatory run.

## P1-3 — `run_job` raises instead of recording, so one bad base job aborts the matrix

**Severity: P1.** `scripts/run_matrix.py:429-442`, `:477`, `:541-547`.
`plan()` raises `RuntimeError` when a parent is not `DONE` or its hash mismatches, and
`_check_identity` raises on a moved config. `run()` does not catch either, so the loop dies
on the first such job and the remaining seeds are never attempted. An idempotent runner
should append a `FAILED`/`BLOCKED` entry and continue.

## P1-4 — a crash between the rename and the sidecar leaks a checkpoint forever

**Severity: P1.** `durable_checkpoint.py:382-411`.
The `.pt` is renamed into place at `:382`; the sidecar is written at `:404` and the index at
`:411`. A kill in between leaves a full-size checkpoint (274 MB at paper scale) that is in no
index, so `prune()` (`:521-534`, which iterates the index) will never collect it. This is the
same disk-leak class as upstream gap §3.5 item 3, which the track reports as closed.
`prune()`'s partial-sweep (`:537-542`) does not cover it because the file is no longer a
`.partial`.

## P1-5 — `write_step_metrics` is tested but never called in production

**Severity: P1.** `scripts/run_matrix.py:248-269`, `tests/test_run_matrix.py:303-314`.
The `(run_id, step)` collision guard exists and works, but the only caller outside tests is
`FakeLauncher`. `toy_trainer.write_metrics` (`toy_trainer.py:335-342`) writes
`metrics/step_*.json` through raw `atomic_write_json`, bypassing the guard, and upstream
writes `version_N/metrics.csv` instead. Spec §9.3 item 5 is satisfied only inside the test
harness.

## P1-6 — PROGRAM.md invariant 1 names `results/metrics.jsonl`; nothing writes it

**Severity: P1.** `PROGRAM.md` "Loop invariants" item 1. `run_matrix.py` writes
`results/run_ledger.json` only.

## P1-7 — `assert_branch_parity` cannot fail as the runner drives it

**Severity: P1.** `scripts/run_matrix.py:550-573`, `tests/test_run_matrix.py:278-287`.
Both arms read `parent_checkpoint_sha256` from the same `states[parent]["final_checkpoint_sha256"]`
in the same `run()` pass (`:433-434`), so within one process they are equal by construction.
The only real crossing it can catch is a parent re-run between the two arms' launches. The
test proves the assertion works on hand-built dicts, not that the runner can produce a
crossing. Keep the check (it is cheap and the invariant is right), but do not count it as
evidence that the near/far parent is verified end to end.

---

## P2

1. **`_step_from_name` returns 0 for upstream's validation checkpoints.**
   `durable_checkpoint.py:626-631` does `name.split(".")[0]`, so
   `ckpt_iter_250_0.4412.pt` (`core_train.py:774-777`) becomes `ckpt_iter_250_0` and the
   last numeric token is `0`. Harmless today because the function only sweeps `.partial`
   files; load-bearing the moment adoption exists. (Fixed as part of P0-2.)
2. **`latest_pointer_path` is dead code.** `durable_checkpoint.py:312-314` is never read or
   written. The module docstring's "atomicity … for checkpoints *and* pointers" covers
   `recovery_ckpt` only; `latest_ckpt` is still upstream's unguarded `open(..., "w")`
   (`core_train.py:944-948`).
3. **`resolve()` deep-verifies with a full `torch.load` every call.** `plan()` and the
   post-launch check each pay it, plus `save()`'s read-back — three full deserializations of
   a 274 MB checkpoint per job launch. Consider `deep=False` (hash only) for the planning
   read.
4. **No cross-check between the sidecar and the index.** `verify_pointer` trusts the sidecar
   (`:244-249`), `_verify_record` trusts the index (`:450-452`). A tamper that edits a file
   and its sidecar passes `verify_pointer`; one that edits a file and the index passes
   `_verify_record`. Neither is a realistic corruption mode, but the two records should
   agree.
5. **`Ledger.append` is O(n²) and races.** `run_matrix.py:200-207` rewrites the whole file and
   sets `seq = len(entries)`; two runners on one Drive produce duplicate `seq` values.
6. **Same-step tie between the `final` and last `recovery` save.** `toy_trainer.py:364-371`
   saves twice at the terminal step; ordering falls to `saved_at` and, on a tie, to sort
   stability. It happens to be correct (the new record is inserted at index 0 before a stable
   sort) but it is undocumented and one refactor from silent breakage.

---

## Checks that came back clean

- **Determinism vs worker count / dict ordering.** No set iteration feeds an output;
  `validate_matrix` uses a set for membership only, `build_matrix` iterates lists,
  `atomic_write_json` sorts keys, `_params_hash` sorts keys, dict comparisons in
  `_check_identity` are order-independent. This is the one place upstream's own bug
  (UPSTREAM_REPORT finding #14, sweep names built from a `set`) could have been repeated, and
  it was not.
- **Seeded jitter.** `DurableSync._rng = random.Random(seed)` (`:684`), never the global RNG;
  `sleep` and `uploader` are injected, so no wall clock in the retry tests.
- **No service-account key path.** `_gcs_upload` (`:766-776`) uses the ADC client only,
  matching the RUNLOG's finding that `iam.disableServiceAccountKeyCreation` blocks keys.
- **Leakage / threshold retuning / bootstrap unit.** Not applicable to this track: it
  contains no estimator, no threshold and no resampling. `PARAM_TOLERANCE = 0.0`
  (`test_resume.py:58`) is the strongest form the tolerance can take and cannot be relaxed
  after seeing output without an obvious diff.
- **The recovery test is falsifiable.** Verified by the `--no-rng-state` arm and by the
  `0 < resumed_from` guard, both of which I confirmed are load-bearing.

---

# Fixes applied

All five P0s fixed. No P1 or P2 was fixed except where a P0 fix depended on it (P2-1,
`_step_from_name`, which adoption reads) and P1-1 (a RUNLOG wording correction, zero code
risk). P1-2 through P1-7 are left open deliberately: the launch-path convergence in
particular is a cross-track decision that should not be made unilaterally inside a review.

| # | File | Change |
|---|---|---|
| P0-1 | `src/lurestar/durable_checkpoint.py:483-494`, `:500-515` (`_pointer_is_dangling`) | `resolve()` clears the pointer only when it has records that all failed verification, or when the pointer is dangling — the one case `core_train.py:148-150` hard-fails on. An empty index no longer destroys a foreign pointer. |
| P0-2 | `durable_checkpoint.py:517-611` (`adopt`, `adopt_existing`); `scripts/run_matrix.py:459-462`, `:551` | Externally written checkpoints are hashed, deserialization-verified, sidecarred and indexed. Adopted files are never pruned — upstream owns its own `latest`/`best` lifecycle. `MatrixRunner.plan()` adopts before resolving; `run_job` adopts again after the launch, so a real run can reach `DONE`. |
| P0-3 | `run_matrix.py:78-98` (`default_config_for`, `default_overrides_for`), `:190-199` | Base → `configs/{model}_lurestar.yaml`; adapt → `configs/adapt_{condition}.yaml`, one file per H3 arm. `build_matrix(require_configs=True)` refuses a matrix whose configs do not exist. GPT adapt jobs carry `use_nextlat=false`. |
| P0-4 | `run_matrix.py:428-473` | `_identity` raises on a missing config or manifest instead of recording `None`; `_check_identity` treats a recorded `None` as a mismatch like any other value. `DEFAULT_MANIFESTS` wires `manifests/corpus.sha256` and `manifests/corpus_provenance.json` into every job. |
| P0-5 | `run_matrix.py:381-408`, `ResumePlan.parent_steps`, `plan()` | The branch command emits `trainer.train_batches = parent_steps + adapt_steps` on the fresh launch and on every resume, and refuses to emit at all when the parent's step count is unknown. `spec.train_batches` remains the frozen number of *adaptation* updates. |
| P2-1 | `durable_checkpoint.py:721-740` | `_step_from_name` strips known suffixes and prefers the token after `iter`, so `ckpt_iter_250_0.4412.pt` reads as 250 rather than 0. |
| P1-1 | `docs/RUNLOG.md` | The recovery-test paragraph now names the CPU surrogate, and an appended correction records that the real-trainer mandatory test is outstanding. |

## Mutation check on the fixes

Each P0 fix was reverted in isolation and the suite re-run. Every mutation broke exactly the
test written for it and nothing else:

```
M1  resolve() unlinks the pointer unconditionally   -> 1 failed  (test_resolve_does_not_delete_a_pointer_it_did_not_write)
M2  plan() does not adopt                           -> 2 failed  (test_runner_resumes_from_a_checkpoint_upstream_wrote,
                                                                  test_a_torn_upstream_checkpoint_rolls_back_to_the_validation_checkpoint)
M3  identity guard skips a recorded None            -> 1 failed  (test_identity_guard_refuses_a_config_that_was_absent_when_the_job_started)
M4  branch command drops the parent-step offset     -> 1 failed  (test_fabric_command_resumes_and_branches_from_a_parent)
M5  old _step_from_name                             -> 5 failed  (three adoption tests + two runner tests)
M6  old config_for                                  -> 6 failed  (config existence, near/far separation, use_nextlat, manifests, and the two
                                                                  default-matrix tests that now depend on real configs)
```

## After

```
$ .venv/bin/python -m pytest tests/test_resume.py tests/test_run_matrix.py -v
collected 51 items
tests/test_resume.py::test_sigkill_at_150_then_resume_reaches_300
  [recovery] killed at step 150, resumed from checkpoint step 125, finished at 300; max |delta param| = 0.000e+00 (tolerance 0.0e+00)
PASSED
tests/test_resume.py::test_upstream_style_reseed_diverges   [recovery] without RNG state in the checkpoint: max |delta param| = 7.012e-02
PASSED
...
tests/test_resume.py::test_resolve_does_not_delete_a_pointer_it_did_not_write PASSED
tests/test_resume.py::test_resolve_still_clears_a_dangling_pointer PASSED
tests/test_resume.py::test_adopt_brings_an_externally_written_checkpoint_under_verification PASSED
tests/test_resume.py::test_adopt_rolls_back_past_a_torn_upstream_checkpoint PASSED
tests/test_resume.py::test_adopt_refuses_partial_and_corrupt_files PASSED
tests/test_resume.py::test_step_is_parsed_from_upstream_validation_checkpoint_names PASSED
tests/test_run_matrix.py::test_branch_command_is_refused_without_the_parent_step_count PASSED
tests/test_run_matrix.py::test_base_job_command_carries_no_offset PASSED
tests/test_run_matrix.py::test_default_matrix_points_at_configs_that_exist PASSED
tests/test_run_matrix.py::test_near_and_far_do_not_share_a_config PASSED
tests/test_run_matrix.py::test_gpt_adaptation_flips_use_nextlat_off PASSED
tests/test_run_matrix.py::test_default_matrix_carries_the_dataset_manifests PASSED
tests/test_run_matrix.py::test_identity_guard_refuses_a_config_that_was_absent_when_the_job_started PASSED
tests/test_run_matrix.py::test_identity_guard_refuses_a_moved_manifest PASSED
tests/test_run_matrix.py::test_runner_resumes_from_a_checkpoint_upstream_wrote PASSED
tests/test_run_matrix.py::test_a_torn_upstream_checkpoint_rolls_back_to_the_validation_checkpoint PASSED

============================== 51 passed in 8.87s ==============================
```

Whole-suite state at the same commit: `357 passed, 5 failed`. All five failures are in
`tests/test_hmm_pairs.py` against `src/hmm_geometry/pair_bank.py` — a different track, whose
files were being edited by another agent during this review. Nothing in this track imports
`hmm_geometry`; those failures are neither caused nor fixed here.

## Verdict

**FAIL as delivered; PASS as fixed, with one blocker outstanding that this track cannot close.**

The durable primitive was good work and its tests are honest. The runner was not: it was
verified only against a stand-in that did the durable layer's job for it, and against upstream's
real output it would have destroyed a resume pointer, restarted a 20,000-step job from zero,
never marked a job `DONE`, resumed under an unchecked config, and run the H3 branches for zero
steps on the wrong model. Five P0s from one unexamined seam is what "all deliverables are in
place and tested" concealed.

The outstanding blocker is P1-1: spec §9's mandatory recovery test has been passed on a numpy
surrogate, not on the 12L/6H/384 model. `docs/FOUNDATIONS.md` D-23 makes that empirical check a
precondition for the `--strategy ddp --devices 1` sampler decision, and spec §10 makes
"interrupted training cannot resume reproducibly enough" a stop condition. Until the 300 vs
150+150 test is run on a real GPU job, the durable layer is verified machinery with an
unverified claim attached to it.
