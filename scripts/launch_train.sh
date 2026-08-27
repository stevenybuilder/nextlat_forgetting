#!/usr/bin/env bash
# The single-GPU launch path (spec section 8).
#
#   scripts/launch_train.sh <config> <seed> [extra dotlist overrides...]
#
# <config> is one of the eight deliverables in configs/ (name or path). The script derives the
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
#   LURESTAR_MODEL     gpt|nextlat|bst, required for adapt_near/adapt_mid/adapt_far
#   LURESTAR_PARENT_CKPT  absolute path to the step-rebased frozen parent checkpoint; on the
#                      FIRST launch of an adaptation branch this seeds {out_dir}/latest_ckpt
#   LURESTAR_ENTRY     override the script fabric launches; scripts/profile.sh points this at
#                      scripts/profile_entry.py (default train.py, or the external
#                      scripts/train_hmm.py shim for HMM configs)
#   DRY_RUN=1          print the command and exit 0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

NEXTLAT_REPO="${NEXTLAT_REPO:-/content/nextlat}"
LURESTAR_ROOT="${LURESTAR_ROOT:-/content/lurestar}"
LURESTAR_PRECISION="${LURESTAR_PRECISION:-bf16-mixed}"
LURESTAR_STRATEGY="${LURESTAR_STRATEGY:-}"
PREREGISTERED_SEEDS=(1234 1235 1236 1237 1238)

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
  bst_lurestar.yaml)
    # Spec sec.8 arm 3: the competence-matched control (docs/DECISION_D20_competence_gate.md).
    MODEL=bst;     OUT_DIR="$LURESTAR_ROOT/runs/bst/seed$SEED/base";     EXP="bst-seed$SEED-base" ;;
  adapt_near.yaml|adapt_mid.yaml|adapt_far.yaml)
    BRANCH="${CONFIG_NAME#adapt_}"; BRANCH="${BRANCH%.yaml}"
    MODEL="${LURESTAR_MODEL:-}"
    [[ -n "$MODEL" ]] || die "adapt_$BRANCH.yaml needs LURESTAR_MODEL=gpt, nextlat, or bst"
    [[ "$MODEL" == gpt || "$MODEL" == nextlat || "$MODEL" == bst ]] || \
      die "LURESTAR_MODEL must be gpt, nextlat, or bst"
    OUT_DIR="$LURESTAR_ROOT/runs/$MODEL/seed$SEED/adapt-$BRANCH"
    EXP="$MODEL-seed$SEED-adapt-$BRANCH"
    # The file is derived from the NextLat G(5,5) YAML (its key set is a superset of the
    # GPT one), so the GPT branch is the same file with the model flag flipped.
    if [[ "$MODEL" == gpt ]]; then
      EXTRA_OVERRIDES+=("use_nextlat=false")
    elif [[ "$MODEL" == bst ]]; then
      # The runtime adapter replaces BST's dense pair objective with the common next-token CE.
      # Pair-gap knobs are intentionally absent from the H3 adaptation estimand.
      EXTRA_OVERRIDES+=("use_nextlat=false" "use_bst=true")
    fi
    ;;
  gpt_hmm.yaml)
    MODEL=gpt;     OUT_DIR="$LURESTAR_ROOT/runs/hmm/gpt/seed$SEED/base";     EXP="gpt-seed$SEED-hmm" ;;
  nextlat_hmm.yaml)
    MODEL=nextlat; OUT_DIR="$LURESTAR_ROOT/runs/hmm/nextlat/seed$SEED/base"; EXP="nextlat-seed$SEED-hmm" ;;
  *) die "unknown config '$CONFIG_NAME'; expected one of the eight deliverables in configs/" ;;
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
  # A plain pinned checkout would silently run its native loss. Require the guarded v5 patch
  # receipt and verify the exact adaptation source hash before allocating GPU compute.
  if ! python3 - "$NEXTLAT_REPO" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
receipt_path = root / ".lurestar_runtime_patch_receipt.json"
source = root / "lurestar_adaptation.py"
if not receipt_path.is_file() or not source.is_file():
    raise SystemExit("missing runtime-patch receipt or adaptation trainer")
receipt = json.loads(receipt_path.read_text())
if receipt.get("patch_version") != 5:
    raise SystemExit("runtime patch is not v5")
if receipt.get("adaptation_contract") != "h3_full_parameter_next_token_ce_v1":
    raise SystemExit("runtime patch does not declare the common H3 objective")
digest = hashlib.sha256(source.read_bytes()).hexdigest()
if receipt.get("adaptation_trainer_sha256") != digest:
    raise SystemExit("adaptation trainer hash disagrees with runtime receipt")
PY
  then
    die "adaptation launch requires the verified v5 common-objective runtime patch"
  fi
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
  # The shim performs in-memory registration and delegates to the pinned trainer. Neither the
  # runtime checkout nor the vendored upstream tree needs (or is allowed) an on-disk edit.
  [[ -f "$SCRIPT_DIR/train_hmm.py" ]] || die "HMM trainer shim missing: $SCRIPT_DIR/train_hmm.py"
  [[ -f "$PROJECT_DIR/src/hmm_geometry/datamodule.py" ]] || \
    die "HMM datamodule missing: $PROJECT_DIR/src/hmm_geometry/datamodule.py"
fi

CMD=(fabric run --devices 1 --precision "$LURESTAR_PRECISION")
if [[ -n "$LURESTAR_STRATEGY" ]]; then CMD+=(--strategy "$LURESTAR_STRATEGY"); fi
ENTRY="${LURESTAR_ENTRY:-}"
if [[ -z "$ENTRY" ]]; then
  if [[ "$CONFIG_NAME" == *hmm.yaml ]]; then ENTRY="$SCRIPT_DIR/train_hmm.py"; else ENTRY="train.py"; fi
fi
CMD+=("$ENTRY" --config "$CONFIG_PATH"
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
