# Adversarial review — "configs-and-launch" track

Reviewer: independent agent. Method: read every delivered file against
`nextlat_v4_predictive_geometry_spec.md`, `docs/UPSTREAM_REPORT.md`, `PROGRAM.md` and the
pinned tree at `upstream/NextLat@3770be60`; run the suite; then **mutate the deliverables in
the specific ways this project is exposed to and check whether the suite kills the mutant**.
Every reproduction below was executed in this terminal.

## Reproduced baseline

```
$ .venv/bin/python -m pytest tests/test_configs.py tests/test_profile_tooling.py -q
........................................................................ [ 44%]
........................................................................ [ 88%]
..................                                                       [100%]
162 passed in 1.96s

$ .venv/bin/python scripts/materialize_configs.py --check
OK  6 configs match the generator at commit 3770be6009cea2b3c455a9ce7f2ca88b504bb955
```

All eleven claimed deliverables exist and are non-trivial (`configs/*.yaml` ×6 +
`overrides.json` 53 KB, `scripts/materialize_configs.py` 34 KB, `scripts/launch_train.sh`
7.3 KB, `scripts/profile.sh` 7.1 KB, `scripts/profile_entry.py` 6.7 KB,
`scripts/profile_summarize.py` 19 KB, `docs/CONFIG_DEVIATIONS.md` 29 KB,
`tests/test_configs.py` 28 KB, `tests/test_profile_tooling.py` 18 KB). No P0 for a missing
file. The generator's output is byte-stable under `PYTHONHASHSEED=0/1/12345`. Every
resolved-value claim in the report (proj_factor 0.5 → hidden 384 → 590,592 params;
`128*round((8*384/3)/128)` = 1024; `stargraph_max_nodes 100` → `vocab_size 106`; inclusive
`self.step > train_batches` at `core_train.py:569` → 20,001 / 501 updates; `use_nextlat`
model-arg block at `core_train.py:85-96`; `defaults.yaml:118 proj_factor: 1.0`) was checked
against the pinned source and is correct.

The problem is not what the suite asserts. It is what the suite **cannot see**.

---

## P0 findings

### 1. (P0) The near and far adaptation banks can be swapped and all 150 config tests pass

`scripts/materialize_configs.py:347-352`

```python
def build_adapt_near(seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    return _adapt("near", B_NEAR, B_NEAR_VAL, seed)


def build_adapt_far(seed: int = DEFAULT_SEED) -> Dict[str, Any]:
    return _adapt("far", B_FAR, B_FAR_VAL, seed)
```

Nothing anywhere ties the string `near` to `B_NEAR`. `tests/test_configs.py:364`
(`test_near_and_far_differ_only_in_the_item_bank_and_output_root`) asserts only that the two
files differ *in* `data.stargraph_train_data_path` — a swap satisfies that exactly.
`tests/test_configs.py:391` asserts `train.endswith("_5000.txt")`, which both banks satisfy.
`data/stargraph.py:187-190` parses only `split("_")[1]` and `[2]` (`5`, `5`), so upstream
cannot tell the two apart either.

**Failing reproduction (executed).** Swap the two banks in the generator, regenerate, run the
suite:

```
$ # build_adapt_near -> _adapt("near", B_FAR,  B_FAR_VAL,  seed)
$ # build_adapt_far  -> _adapt("far",  B_NEAR, B_NEAR_VAL, seed)
$ .venv/bin/python scripts/materialize_configs.py
$ grep stargraph_train_data_path configs/adapt_near.yaml configs/adapt_far.yaml
configs/adapt_near.yaml: /content/lurestar/manifests/adapt/graph_5_5_bfar_5000.txt
configs/adapt_far.yaml:  /content/lurestar/manifests/adapt/graph_5_5_bnear_5000.txt
$ .venv/bin/python -m pytest tests/test_configs.py -q
150 passed in 1.40s
```

`materialize_configs.py --check` also passes, because the generator is self-consistent with
the swap.

**Consequence.** H3's primary outcome is
`similarity-dependent interference = erosion_near - erosion_far` (spec §6). A swapped bank
negates it exactly. Every downstream figure, the ledger, and the writeup would report the
sign-flipped result with full provenance and a green test suite. This is precisely the class
of error `docs/RUNLOG.md` records for the leakage index ("the check would have passed
vacuously on every input").

**Fix.** Assert the branch↔bank binding in the generator (so a swap cannot be emitted) *and*
on the on-disk files (so the assertion does not depend on the generator being right).

---

### 2. (P0) An adaptation config can point at `E_lure` or the base corpus and all 150 tests pass

`scripts/materialize_configs.py:79-83` (the `B_NEAR` / `B_FAR` / `B_*_VAL` constants)

Spec §5: *"No `E_lure` graph or lure may enter base or adaptation training."* No test
compares the adaptation paths against the base corpus, against `manifests/e_lure.jsonl`, or
against `manifests/a_pair.jsonl`.

**Failing reproduction (executed).**

```
$ # B_NEAR = f"{MANIFESTS}/elure/graph_5_5_elure_5000.txt"
$ .venv/bin/python scripts/materialize_configs.py
$ grep stargraph_train_data_path configs/adapt_near.yaml
  /content/lurestar/manifests/elure/graph_5_5_elure_5000.txt
$ .venv/bin/python -m pytest tests/test_configs.py -q
150 passed in 1.25s
```

The same mutation pointed at `CORPUS_TRAIN` also passes every test except
`test_near_and_far_differ_only_in_the_item_bank_and_output_root` (only because that makes the
two files identical in one more key).

**Consequence.** Adapting on `E_lure` destroys the disjointness the whole H1/H2/H3 design
rests on, and the config layer — the layer whose entire job is to be auditable — signs off on
it.

**Fix.** A path-hygiene invariant: no `data.*_data_path` in any config may name a pool that is
not the pool the config's family is allowed to touch; adaptation paths must be disjoint from
the base corpus and must not name `elure`/`a_pair`.

---

### 3. (P0) The profiling gate exits 0 with peak VRAM, host-input wait and checkpoint bytes entirely unmeasured

`scripts/profile_summarize.py:456-468`

```python
failed = [k for k, r in records.items()
          if r.get("returncode") not in (0, None) or r.get("seconds_per_step_median") is None]
```

The gate's only pass/fail criteria are the process return code and a median seconds-per-step.
Spec §11 requires the gate to **Record** "peak allocated and reserved VRAM; GPU utilization
and host-input wait; checkpoint-write duration and bytes". `docs/RUNLOG.md` closes the
profiling section with *"**Peak VRAM is therefore still unmeasured** and must be captured on
the first confirmatory run"*, and this track's headline claim is that `profile_entry.py`
"closes the RUNLOG gap". It does not: if the probe never lands — wrong `PROFILE_PROBE_JSON`,
a `fabric run` that re-execs, an `atexit` that never fires, a probe written to a path the
summarizer does not glob — the gate reports `-` in every VRAM row and **still exits 0**.

**Failing reproduction (executed).** Four synthetic jobs (lurestar gpt/nextlat 500 steps,
hmm gpt/nextlat 300 steps), complete metrics.csv and materialized_config.yaml, `returncode: 0`,
and zero probe files:

```
$ .venv/bin/python scripts/profile_summarize.py --jobs-dir .../nogate/jobs --out .../summary.json
GATE EXIT=0
| peak allocated VRAM (GB) | - | - |
| peak reserved VRAM (GB)  | - | - |
| VRAM headroom            | - | - |
```

`tests/test_profile_tooling.py:227` (`test_missing_probe_is_reported_not_silently_zero`) only
asserts the *record* says `probe_missing`; it never asserts the *gate* fails. So the test
passes on exactly the input it should reject.

**Consequence.** PROGRAM.md invariant 2 ("Profile before sweeping … No job launches without a
measured seconds-per-step and a projected compute-unit cost") and spec §11's memory-fit
requirement ("Profile the paper's physical batch first. If it does not fit, test gradient
accumulation…") are both decided from numbers the gate does not require to exist. The
`physical_batch_fits` cell is worse than useless: it is
`returncode == 0 and peak_reserved < total`, so it can only ever print `yes` or `-`, never
`NO`.

**Fix.** Make every spec-§11-required quantity a gate condition; report which ones are missing
per job and exit non-zero.

---

## P1 findings (not fixed — see "What remains")

### 4. (P1) `launch_train.sh` seeds an adaptation branch from *any* file, with no identity, step or hash check

`scripts/launch_train.sh:96-105`

The script refuses a first adaptation launch with no pointer (good), but then writes whatever
`LURESTAR_PARENT_CKPT` names into `{out_dir}/latest_ckpt` unconditionally.
`docs/CONFIG_DEVIATIONS.md:355-366` states the requirement it does not enforce: the parent must
be **step-rebased** (`training_steps` rewritten to 0), and *"both branches must record the same
`parent_checkpoint_sha256`"* (spec §9, verbatim).

**Failing reproduction (executed).** A GPT / seed-1234 near branch seeded from a file named
for NextLat / seed 1236 whose contents are the text `not a checkpoint`:

```
$ DRY_RUN=1 LURESTAR_MODEL=gpt \
  LURESTAR_PARENT_CKPT=.../parents/nextlat-seed1236-base-ckpt_iter_20000.pt \
  bash scripts/launch_train.sh adapt_near.yaml 1234
seeded .../runs/gpt/seed1234/adapt-near/latest_ckpt -> .../nextlat-seed1236-base-ckpt_iter_20000.pt
RC=0
```

The documented catastrophic case is worse than a wrong parent: a parent that is **not**
step-rebased makes `self.step` restore to 20,000, `core_train.py:569` returns immediately, and
the branch performs **zero** adaptation updates while still writing a validation checkpoint and
a metrics.csv. `erosion_near - erosion_far` then equals `0 - 0 = 0` — a manufactured clean null
that looks like a legitimate H3 result.

**Fix.** Require a sidecar (`<parent>.meta.json` from the durable-checkpoint layer, or an
explicit `LURESTAR_PARENT_SHA256`) carrying `{model, seed, training_steps, sha256}`; refuse
unless `model`/`seed` match the branch being launched and `training_steps == 0`; write
`{out_dir}/parent_checkpoint.json` and refuse a re-seed whose sha disagrees with the recorded
one. Not fixed here because the sidecar contract is owned by
`src/lurestar/durable_checkpoint.py` + `scripts/run_matrix.py`, both of which another agent is
editing in this session (`git status` shows both modified).

### 5. (P1→fixed) `DRY_RUN=1` mutates the run tree

`scripts/launch_train.sh:94,100-102,127` (pre-fix line numbers)

`mkdir -p "$OUT_DIR"` and the `latest_ckpt` write both execute **before** the
`if [[ "${DRY_RUN:-0}" == "1" ]]; then exit 0; fi` at line 143. Reproduced above: a dry run
created `.../runs/gpt/seed1234/adapt-near/latest_ckpt`. A later *real* launch with the correct
parent then prints `note: branch already has a resume pointer; ignoring LURESTAR_PARENT_CKPT`
and trains from the dry-run's parent. A refused launch (finding 4's `die` path) likewise leaves
`runs/nextlat/seed1234/adapt-{near,far}/` behind. **Fixed** (see below) because the fix is three
lines and strictly removes a silent-wrong-parent path.

### 6. (P1) `launch_train.sh` had no tests at all

`grep -rn launch_train tests/` returned nothing before this review. Four of the five behaviours
the implementer reported for it ("refuses a non-preregistered seed", "refuses a first
adaptation launch with no parent pointer", "refuses an HMM job when `hmm_belief` isn't
registered", "detects a stale `recovery_ckpt`") were unverified by any automated check. I
reproduced all four by hand and they do work; they are now pinned by
`tests/test_launch_train.py` (added, see below).

### 7. (P1) A test rewrites a tracked confirmatory config in place

`tests/test_configs.py:628-642` — `test_negative_control_a_shuffled_override_table_fails_the_check`
does `target.write_text(original.replace("compile: false", "compile: true"))` on the real
`configs/gpt_lurestar.yaml` and restores it in a `finally`. `finally` does not run on `SIGKILL`,
and under any parallel runner the other 149 tests read the mutated file. The evidence that this
already happens: `configs/gpt_lurestar.yaml` has an mtime 18 minutes later than the other five
configs, which were all written by the same generator invocation.
**Fix:** copy `configs/` into `tmp_path`, run the generator with `CONFIGS_DIR` redirected there
(add a `--configs-dir` flag), and never touch the shipped file.

### 8. (P1) `adapt_*` point at validation banks that no manifest, script or provenance record produces

`configs/adapt_near.yaml:51`, `configs/adapt_far.yaml:51` →
`graph_5_5_bnearval_2000.txt`, `graph_5_5_bfarval_2000.txt`.
`manifests/stimuli_provenance.json` counts only `e_lure 2000 / a_pair 1000 / b_near 5000 /
b_far 15000`; there is no 2,000-item near/far validation pool, and no script serializes
`b_near.jsonl` → `graph_5_5_bnear_5000.txt` either. Spec §6 requires acquisition on
"independent near/far validation sets", so the key is right and the pool is missing.
`launch_train.sh` verifies the HMM `.npz` exists (`profile.sh:155-163`) but never checks that a
stargraph data path exists, so the first H3 launch will die inside
`StarGraphDataModule.__init__` on the GPU instead of on the host.

### 9. (P1) `dataloader_wait_s` is not the training loop's host-input wait

`scripts/profile_entry.py:77-94` patches `torch.utils.data.DataLoader.__iter__` at **class**
level, so the train, validation, test and generalization loaders all accumulate into one
counter. Spec §11 asks for host-input wait as a roofline term against the training step;
pooling ~200 validation batches every `val_interval` into it inflates it. The probes are also
only ever executed against a **stub** `torch`/`lightning`
(`tests/test_profile_tooling.py:257+`), so the patch has never been exercised against a real
`DataLoader`, where `__iter__` returning a generator instead of a `_BaseDataLoaderIter` is a
behaviour change (harmless at `num_workers: 0`, not obviously so with
`persistent_workers`/prefetch, both of which PROGRAM.md puts on the mutable surface).
**Fix:** key the counter by `id(dataloader)` and report the training loader separately.

---

## P2 findings

10. **`scripts/launch_train.sh:117` greps `$NEXTLAT_REPO/train.py` before line 123 checks it
    exists.** A missing repo on an HMM job reports `has no 'hmm_belief' entry in DATAMODULES`
    plus a raw `grep: … No such file` — a misleading diagnosis. Move the existence checks above
    the family-specific preconditions.
11. **Two "negative controls" test Python, not the project.**
    `tests/test_configs.py:600` and `:612` both reduce to
    `with pytest.raises(AssertionError): assert True == False`. They execute no project code and
    cannot fail on wrong input in any way that means anything. (The other five negative controls
    — sweep/proj_factor, `test_generalization`, the underscore parser, the `--check` drift, the
    YAML float resolver — are real.)
12. **`scripts/materialize_configs.py:672-673`** — `source = load_yaml(...)` followed
    immediately by `del source`, and it re-invokes `BUILDERS[name](args.seed)` to do it. Dead
    code that doubles the builder cost.
13. **`scripts/profile.sh:142`** — `run_job` always `return 0`, so a crashed job does not stop
    the sequence. Caught downstream by `profile_summarize`, but the log ordering makes a crash
    easy to miss.
14. **All six configs bake `seed: 1234` and a seed-1234 `out_dir`/`experiment_name`.** The
    header of every emitted file shows a raw `fabric run … --config <this file>` invocation; run
    that way (without `launch_train.sh`) a "seed 1235" run silently writes into the seed-1234
    output root. Consider emitting `seed: null` and letting the launcher be the only path.
15. **`profile_summarize.contrast()`** guards with truthiness (`if gpt.get(field) and …`), so a
    legitimate `0.0` is treated as absent.

---

## Checks that came back clean

- **Frozen surface.** Every key PROGRAM.md freezes is either identical to the upstream-resolved
  value or carries a spec-section exemption in `EXEMPT_FROZEN`, and `I4` in `build_one` really
  raises (I confirmed by moving `optimizer.learning_rate`).
- **`proj_factor` trap.** The sweep-deletion fallback to `1.0` is real
  (`defaults.yaml:118`), the hoist is value-preserving, and
  `test_negative_control_dropping_the_sweep_reverts_proj_factor` genuinely reconstructs the
  broken config from upstream rather than asserting a constant.
- **YAML float resolver.** `yaml.safe_load("x: 5e-4")` really returns the string, the
  `OmegaConfCompatLoader` regex is a faithful copy of OmegaConf's, and `safe_dump` of that
  string re-emits `5e-4` unquoted, so the file stays a literal copy while the trainer reads
  `0.0005`. Verified locally; `omegaconf` is not installed in `.venv`, so this is
  transcription-verified, not execution-verified — worth one Colab-side assertion.
- **Determinism.** Generator output is byte-identical under three `PYTHONHASHSEED` values; no
  set/dict iteration reaches the emitted bytes; no behaviour depends on worker count
  (`num_workers: 0` everywhere and the `RandomSampler` permutation is drawn in the main
  process).
- **Key coverage.** `test_every_key_the_trainer_reads_resolves` requires 63 real keys scraped
  from the pinned sources and its negative control (`data.test_generalization`) fires. I checked
  `utils/` and `eval/` for `config.<section>.<key>` reads outside `_ON_PATH`: there are none.
- **`sweep:` deletion.** With no `sweep` key `train.py:346-355` takes the
  `OmegaConf.merge(default, base_config, cli)` branch, so the CLI dotlist genuinely wins — the
  launcher's `seed=`/`trainer.out_dir=` overrides are effective.
- **Thresholds.** Nothing in this track carries a tunable scientific threshold, so there is
  nothing here that could be retuned after seeing model output. Bootstrap/inferential-unit
  concerns do not arise in this track.
- **`--strategy` disagreement with `run_matrix.py`.** Real, and honestly documented as an open
  decision closed by the §9 recovery test (`docs/CONFIG_DEVIATIONS.md:421-438`). Not a finding
  against this track.

---

## Fixes applied in this review

**P0-1 and P0-2 — bank identity and path hygiene.**
`scripts/materialize_configs.py`: new `ADAPT_BANKS` table and `_check_adapt_paths()`, invoked
from `build_one()` for the `adapt` family. It refuses to emit a config whose train/val bank
basename does not carry its own branch tag, that carries the *opposite* branch's tag, that
equals the base corpus, or that names a pool reserved for evaluation (`elure`, `a_pair`).
`tests/test_configs.py`: `test_adaptation_banks_are_bound_to_their_branch`,
`test_no_adaptation_path_touches_an_evaluation_or_base_pool`, and two negative controls that
build the swapped and the leaked plan through the real generator and assert it raises.

**P0-3 — the profiling gate.**
`scripts/profile_summarize.py`: `REQUIRED_MEASUREMENTS` (spec §11's list), a per-record
`missing_required` field, and a `main()` that exits non-zero listing them. `physical_batch_fits`
is now `None` unless a peak-reserved figure exists, so it can report `NO`.
`tests/test_profile_tooling.py`: `test_gate_fails_when_peak_vram_was_never_measured` builds the
exact four-job/no-probe input above and asserts the CLI exits non-zero, plus
`test_gate_passes_only_when_every_required_measurement_is_present` as the positive control.

**P1-5 — `DRY_RUN` side effects.** `scripts/launch_train.sh` now performs every *check* in dry
run but no `mkdir` and no pointer write.

**P1-6 — launcher coverage.** New `tests/test_launch_train.py` pins the five guards, the exact
`fabric run --devices 1 --precision bf16-mixed` command shape, the `use_nextlat=false` flip for
a GPT adaptation branch, the per-model/per-seed output-root derivation, and that a dry run
leaves no trace.

## After-output

Immediately after the fixes, before a concurrent agent landed the BST arm:

```
$ .venv/bin/python -m pytest tests/test_configs.py tests/test_profile_tooling.py \
                             tests/test_launch_train.py -q
........................................................................ [ 35%]
........................................................................ [ 70%]
.............................................................            [100%]
205 passed in 5.83s

$ .venv/bin/python -m pytest -q          # whole repository
423 passed in 64.71s (0:01:04)
```

Final state, after that agent's `bst_lurestar.yaml` landed and I5 was re-verified against it:

```
$ .venv/bin/python scripts/materialize_configs.py --check
OK  7 configs match the generator at commit 3770be6009cea2b3c455a9ce7f2ca88b504bb955

$ .venv/bin/python -m pytest tests/test_configs.py tests/test_profile_tooling.py \
                             tests/test_launch_train.py -q
........................................................................ [ 29%]
........................................................................ [ 59%]
........................................................................ [ 88%]
............................                                             [100%]
244 passed in 4.87s

$ .venv/bin/python -m pytest tests/test_configs.py -q -k "bound_to_their_branch or \
    evaluation_or_base_pool or refused_by_the_generator or pointed_off_the_frozen_corpus"
10 passed, 188 deselected in 0.23s
```

(162 → 244 across this track: +9 pool-identity and pool-identity-negative-control tests,
+11 profiling-gate tests, +23 launcher tests, the rest from the concurrent BST arm. The two
`tests/test_hmm_pairs.py` failures the implementer reported were fixed by another agent during
this review.)

Whole repository, once the concurrent BST arm finished landing:

```
$ .venv/bin/python -m pytest -q
532 passed in 75.10s (0:01:15)
```

(Intermediate runs during that landing showed up to 5 failures, all in
`tests/test_run_matrix.py` and `tests/test_representations.py` — the BST arm mid-flight, in
files this track does not own and that import none of the files changed here. They are green
in the run above.)

`tests/test_launch_train.py` was extended by that agent with a `bst_lurestar.yaml` case while
this review was running, and it passes, so the BST launch path is covered by the same guards.

### The three P0 reproductions, replayed against the fixed track

```
$ # P0-1: build_adapt_near <- B_FAR, build_adapt_far <- B_NEAR   (was: 150 passed)
$ .venv/bin/python -m pytest tests/test_configs.py -q
2 failed, 157 passed
$ .venv/bin/python scripts/materialize_configs.py
AssertionError: adapt_near.yaml: data.stargraph_train_data_path = 'graph_5_5_bfar_5000.txt'
  does not carry its own branch tag 'bnear'. Spec sec.6's primary outcome is
  erosion_near - erosion_far, so a bank that is not bound to its branch silently negates
  the result.

$ # P0-2: B_NEAR -> {MANIFESTS}/elure/graph_5_5_elure_5000.txt   (was: 150 passed)
$ .venv/bin/python -m pytest tests/test_configs.py -q
2 failed, 157 passed

$ # P0-2 variant, mutating the emitted YAML directly so the generator is bypassed
$ sed -i '' 's#bfar_5000#sample_200000#' configs/adapt_far.yaml   (schematic)
FAILED test_adaptation_banks_are_bound_to_their_branch[adapt_far.yaml]
FAILED test_no_adaptation_path_touches_an_evaluation_or_base_pool[adapt_far.yaml]

$ # P0-3: four complete jobs, returncode 0, real metrics.csv, zero probes  (was: GATE EXIT=0)
$ .venv/bin/python scripts/profile_summarize.py --jobs-dir .../jobs --out .../s.json
GATE EXIT=1
UNMEASURED lurestar-gpt: checkpoint-write bytes, checkpoint-write duration, host-input wait,
                         peak allocated VRAM, peak reserved VRAM
UNMEASURED lurestar-nextlat: ... (same)
UNMEASURED hmm-gpt: ... (same)
UNMEASURED hmm-nextlat: ... (same)
INCOMPLETE: spec section 11 requires every quantity above to be RECORDED.
```

### The new tests were mutation-tested, not merely run

Every guard now has a mutant that kills exactly its test and nothing else:

| mutant | killed by |
|---|---|
| `launch_train.sh` seed guard disabled | `test_refuses_a_seed_that_is_not_preregistered` (4 params) |
| `mkdir -p "$OUT_DIR"` moved back above the `DRY_RUN` exit | `test_dry_run_never_writes_a_resume_pointer` |
| `use_nextlat=false` flip removed | `test_the_gpt_adaptation_branch_flips_the_model_flag_and_gets_its_own_root` |
| `hmm_belief` DATAMODULES guard removed | `test_refuses_an_hmm_job_until_the_datamodule_is_registered` |
| parent-pointer guard removed | `test_refuses_a_first_adaptation_launch_with_no_parent`, `test_a_refused_launch_leaves_no_run_directory` |
| `if unmeasured:` → `if False:` in `profile_summarize.main` | `test_gate_fails_when_peak_vram_was_never_measured` |
| I5 removed for one branch tag / one pool / one directory (5 variants) | the generator raises on each; `test_negative_control_a_swapped_adaptation_bank_is_refused_by_the_generator` and siblings |
| any single one of the nine `REQUIRED_MEASUREMENTS` blanked | `test_every_spec_section_11_quantity_individually_fails_the_gate[<field>]` |
| `optimizer.learning_rate` moved with no exemption (pre-existing I4) | generator raises: `FROZEN key optimizer.learning_rate moved '5e-4' -> 0.001` |

`test_gate_passes_only_when_every_required_measurement_is_present` is the positive control for
the gate tests: without it, "exits non-zero" could be satisfied by a gate that never passes.

## Observed during the review, not owned by it

A concurrent agent added a **seventh** config, `configs/bst_lurestar.yaml`, plus
`OFFICIAL_BST_5_5` and BST parameter-count helpers in `scripts/config_lib.py`, while this
review was running. Two things to settle, by a human:

1. Spec §15's `configs/` deliverable list has **six** files, and spec §8 defines exactly two
   conditions — "the official repository's GPT implementation" and "the architecture-matched
   transformer trained with the official NextLat objective". Spec §14 excludes "generic"
   scope expansion and the instructions open with *"Do not broaden the benchmark suite."* A
   BST arm is a third training condition; if it is meant as a declared baseline it needs a
   written authority in `docs/CONFIG_DEVIATIONS.md` and a line in `PROGRAM.md`, and its
   GPU-hours have to enter the spec §11 budget (`profile_summarize.project()` still assumes
   `N_MODELS = 2`, `6` base runs, `12` adaptation branches — the projection is now a
   lower bound).
2. Mechanically it is fine with the fix in this review: its `family` is `lurestar`, so I5
   pins it to the frozen 200,000/20,000 corpus, and
   `materialize_configs.py --check` reports `OK  7 configs`.

## Disclosure: concurrent-edit hazard on this file's neighbours

`docs/CONFIG_DEVIATIONS.md` was being edited by another agent at the same time as this review.
Both of us rewrite it whole-file, so either could silently drop the other's paragraphs. My
edits to it are five targeted string replacements (the I5 row and paragraph, the profiling-gate
paragraph, and the adaptation-launch paragraph), and a copy of my version is at
`<scratchpad>/cd.mine` in case a later write loses them. If
`test_every_override_is_documented` or `test_gradient_accumulation_fallback_rule_is_written_down`
ever fails, that is the failure mode. **Two agents should not be whole-file-rewriting the same
markdown deliverable**; this belongs in `PROGRAM.md` as a loop invariant.

## What remains

P1 findings 4, 7, 8, 9 and all P2 findings are listed above with concrete fixes and are **not**
applied. 4 is deferred because it needs the durable-checkpoint sidecar contract, which another
agent is editing concurrently; 7 needs a `--configs-dir` flag on the generator; 8 is a stimulus
pipeline deliverable, not a config one; 9 changes measurement semantics and should be repriced
on Colab rather than changed blind.

**Finding 4 is the one to close before the first H3 launch.** It is the only remaining path in
this track by which a green, fully provenanced run produces a fabricated H3 null.
