#!/usr/bin/env bash
# The single-GPU launch path (spec section 8).
#
#   scripts/launch_train.sh <config> <seed> [extra dotlist overrides...]
#
# <config> is one of the six deliverables in configs/ (name or path). The script derives the
# per-job output root and experiment name, checks the preconditions that upstream fails
# silently on, prints the exact command, and execs it.
#
# Spec section 8 gives the launch verbatim:
#   fabric run --devices 1 --precision bf16-mixed train.py --config <config.yaml>
# The shipped scripts/stargraph/5_5/train_{gpt,nextlat}_star_5_5.sh:3 use
# `--devices 2 --strategy ddp`; we run one device with effective_batch_size held at 512, so
# device_batch_size becomes 512 rather than 256 (train.py:143-145).
#
# Environment knobs (all optional):
#   NEXTLAT_REPO       working copy of the pinned repo            (default /content/nextlat)
#   LURESTAR_ROOT      durable root for runs/data/manifests       (default /content/lurestar)
#   LURESTAR_PRECISION fabric --precision                         (default bf16-mixed)
#   LURESTAR_STRATEGY  extra `--strategy X`; empty by default     (see docs/CONFIG_DEVIATIONS.md
#                      "Sampler determinism" before setting this to ddp)
#   LURESTAR_MODEL     gpt|nextlat, required for adapt_near/adapt_far
#   LURESTAR_PARENT_CKPT  absolute path to the step-rebased frozen parent checkpoint; on the
#                      FIRST launch of an adaptation branch this seeds {out_dir}/latest_ckpt
#   LURESTAR_ENTRY     script fabric launches; scripts/profile.sh points this at
#                      scripts/profile_entry.py                  (default train.py)
#   DRY_RUN=1          print the command and exit 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

NEXTLAT_REPO="${NEXTLAT_REPO:-/content/nextlat}"
LURESTAR_ROOT="${LURESTAR_ROOT:-/content/lurestar}"
LURESTAR_PRECISION="${LURESTAR_PRECISION:-bf16-mixed}"
LURESTAR_STRATEGY="${LURESTAR_STRATEGY:-}"
PREREGISTERED_SEEDS=(1234 1235 1236)

die() { echo "launch_train.sh: $*" >&2; exit 2; }

[[ $# -ge 2 ]] || die "usage: launch_train.sh <config> <seed> [dotlist overrides...]"

CONFIG_ARG="$1"; SEED="$2"; shift 2

if [[ -f "$CONFIG_ARG" ]]; then
  CONFIG_PATH="$(cd "$(dirname "$CONFIG_ARG")" && pwd)/$(basename "$CONFIG_ARG")"
else
  CONFIG_PATH="$PROJECT_DIR/configs/$CONFIG_ARG"
fi
[[ -f "$CONFIG_PATH" ]] || die "config not found: $CONFIG_ARG"
CONFIG_NAME="$(basename "$CONFIG_PATH")"

[[ "$SEED" =~ ^[0-9]+$ ]] || die "seed must be an integer, got '$SEED'"
if [[ "${LURESTAR_ALLOW_ANY_SEED:-0}" != "1" ]]; then
  ok=0
  for s in "${PREREGISTERED_SEEDS[@]}"; do [[ "$SEED" == "$s" ]] && ok=1; done
  [[ $ok == 1 ]] || die "seed $SEED is not preregistered (${PREREGISTERED_SEEDS[*]}). \
Set LURESTAR_ALLOW_ANY_SEED=1 for a non-confirmatory run such as the profiling gate."
fi

EXTRA_OVERRIDES=()

case "$CONFIG_NAME" in
  gpt_lurestar.yaml)
    MODEL=gpt;     OUT_DIR="$LURESTAR_ROOT/runs/gpt/seed$SEED/base";     EXP="gpt-seed$SEED-base" ;;
  nextlat_lurestar.yaml)
    MODEL=nextlat; OUT_DIR="$LURESTAR_ROOT/runs/nextlat/seed$SEED/base"; EXP="nextlat-seed$SEED-base" ;;
  adapt_near.yaml|adapt_far.yaml)
    BRANCH="${CONFIG_NAME#adapt_}"; BRANCH="${BRANCH%.yaml}"
    MODEL="${LURESTAR_MODEL:-}"
    [[ -n "$MODEL" ]] || die "adapt_$BRANCH.yaml needs LURESTAR_MODEL=gpt or LURESTAR_MODEL=nextlat"
    [[ "$MODEL" == gpt || "$MODEL" == nextlat ]] || die "LURESTAR_MODEL must be gpt or nextlat"
    OUT_DIR="$LURESTAR_ROOT/runs/$MODEL/seed$SEED/adapt-$BRANCH"
    EXP="$MODEL-seed$SEED-adapt-$BRANCH"
    # The file is derived from the NextLat G(5,5) YAML (its key set is a superset of the
    # GPT one), so the GPT branch is the same file with the model flag flipped.
    if [[ "$MODEL" == gpt ]]; then EXTRA_OVERRIDES+=("use_nextlat=false"); fi
    ;;
  gpt_hmm.yaml)
    MODEL=gpt;     OUT_DIR="$LURESTAR_ROOT/runs/hmm/gpt/seed$SEED/base";     EXP="gpt-seed$SEED-hmm" ;;
  nextlat_hmm.yaml)
    MODEL=nextlat; OUT_DIR="$LURESTAR_ROOT/runs/hmm/nextlat/seed$SEED/base"; EXP="nextlat-seed$SEED-hmm" ;;
  *) die "unknown config '$CONFIG_NAME'; expected one of the six deliverables in configs/" ;;
esac

# --- preconditions upstream fails silently on ------------------------------------------

[[ -f "$NEXTLAT_REPO/train.py" ]] || die "no train.py under NEXTLAT_REPO=$NEXTLAT_REPO"
[[ -f "$NEXTLAT_REPO/defaults.yaml" ]] || die "train.py loads defaults.yaml relative to CWD \
(train.py:348); $NEXTLAT_REPO/defaults.yaml is missing"

# Every check below runs in DRY_RUN too; nothing below MUTATES anything in DRY_RUN. Seeding
# a resume pointer from a dry run is not a preview, it is a decision: a later real launch
# with the right parent would find the pointer already present, print "ignoring
# LURESTAR_PARENT_CKPT", and adapt from whatever the dry run happened to name.
SEED_POINTER=""

if [[ "$CONFIG_NAME" == adapt_* ]]; then
  # core_train.py:139-168: init_from=resume prefers {out_dir}/recovery_ckpt, falls back to
  # {out_dir}/latest_ckpt, and if NEITHER exists it prints two "Could not find" lines and
  # builds a SCRATCH model. For an adaptation branch that would silently train a fresh
  # random network on 5,000 items, so refuse to launch without a pointer.
  if [[ ! -f "$OUT_DIR/recovery_ckpt" && ! -f "$OUT_DIR/latest_ckpt" ]]; then
    [[ -n "${LURESTAR_PARENT_CKPT:-}" ]] || die "first launch of $EXP needs LURESTAR_PARENT_CKPT=\
<absolute path to the step-rebased frozen parent checkpoint>; without a pointer at \
$OUT_DIR/latest_ckpt upstream would train from scratch"
    [[ -f "$LURESTAR_PARENT_CKPT" ]] || die "LURESTAR_PARENT_CKPT does not exist: $LURESTAR_PARENT_CKPT"
    SEED_POINTER="$LURESTAR_PARENT_CKPT"
  elif [[ -n "${LURESTAR_PARENT_CKPT:-}" ]]; then
    echo "note: branch already has a resume pointer; ignoring LURESTAR_PARENT_CKPT" >&2
  fi
  # A stale recovery pointer to a deleted file hard-fails at core_train.py:148-150.
  if [[ -f "$OUT_DIR/recovery_ckpt" ]] && [[ ! -f "$(cat "$OUT_DIR/recovery_ckpt")" ]]; then
    die "stale recovery pointer at $OUT_DIR/recovery_ckpt -> $(cat "$OUT_DIR/recovery_ckpt") \
(file missing). Repair it with the durable-checkpoint layer before relaunching."
  fi
fi

if [[ "$CONFIG_NAME" == *hmm.yaml ]]; then
  # train.py:176-178 asserts config.data.dataset is a key of DATAMODULES (train.py:34-42).
  # `hmm_belief` is not registered at the pinned commit; the runtime working copy carries a
  # one-line registration recorded as an uncommitted diff (spec section 9).
  if ! grep -q "hmm_belief" "$NEXTLAT_REPO/train.py"; then
    die "$NEXTLAT_REPO/train.py has no 'hmm_belief' entry in DATAMODULES. Apply the \
datamodule registration to the working copy first (see docs/CONFIG_DEVIATIONS.md)."
  fi
fi

CMD=(fabric run --devices 1 --precision "$LURESTAR_PRECISION")
if [[ -n "$LURESTAR_STRATEGY" ]]; then CMD+=(--strategy "$LURESTAR_STRATEGY"); fi
CMD+=("${LURESTAR_ENTRY:-train.py}" --config "$CONFIG_PATH"
      "seed=$SEED" "trainer.out_dir=$OUT_DIR" "trainer.experiment_name=$EXP")
CMD+=(${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"})
CMD+=("$@")

echo "# config      $CONFIG_PATH"
echo "# model/seed  $MODEL / $SEED"
echo "# out_dir     $OUT_DIR"
echo "# experiment  $EXP"
echo "# cwd         $NEXTLAT_REPO"
[[ -n "$SEED_POINTER" ]] && echo "# would seed  $OUT_DIR/latest_ckpt -> $SEED_POINTER"
printf '+ '; printf '%q ' "${CMD[@]}"; echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "# DRY_RUN=1: nothing was created, no pointer was written"
  exit 0
fi

mkdir -p "$OUT_DIR"
if [[ -n "$SEED_POINTER" ]]; then
  printf '%s' "$SEED_POINTER" > "$OUT_DIR/latest_ckpt.partial"
  mv "$OUT_DIR/latest_ckpt.partial" "$OUT_DIR/latest_ckpt"
  echo "seeded $OUT_DIR/latest_ckpt -> $SEED_POINTER"
fi

cd "$NEXTLAT_REPO"
exec "${CMD[@]}"
