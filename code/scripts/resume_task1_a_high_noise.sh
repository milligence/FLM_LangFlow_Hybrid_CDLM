#!/usr/bin/env bash
set -euo pipefail

# Strict contract for the selected Task1 V1 continuation. The historical file
# name is retained for checkpoint-era compatibility.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config_path="${TASK1_TRAIN_CONFIG:?Set TASK1_TRAIN_CONFIG to one high-noise contract}"
# shellcheck source=task1_train_config.sh
source "$script_dir/task1_train_config.sh"
task1_load_train_config "$config_path"

: "${RESUME_CHECKPOINT_PATH:?Set the source full-state checkpoint path}"
: "${RUN_DIR:?Set a new candidate output directory}"
: "${TASK1_RUN_ID:?Set a unique candidate run ID}"
TARGET_MAX_STEPS="${TARGET_MAX_STEPS:-30000}"
export TARGET_MAX_STEPS

[[ "$TASK1_VARIANT" == A ]] || {
  echo "High-noise continuation requires TASK1_VARIANT=A" >&2; exit 2;
}
[[ "$SOURCE_LEARNING_RATE" == 6e-4 ]] || {
  echo "High-noise continuation requires SOURCE_LEARNING_RATE=6e-4" >&2
  exit 2
}
[[ "$RESUME_CONSTANT_LEARNING_RATE" == 6e-4 ]] || {
  echo "High-noise continuation LR must remain constant at 6e-4" >&2
  exit 2
}
[[ "$GLOBAL_BATCH_SIZE" == 256 ]] || {
  echo "High-noise continuation requires GLOBAL_BATCH_SIZE=256" >&2
  exit 2
}
case "$MICRO_BATCH_SIZE/$ACCUMULATE_GRAD_BATCHES" in
  32/8|64/4|128/2) ;;
  *)
    echo "High-noise H800 micro/accumulation must be 32/8, 64/4, or 128/2" >&2
    exit 2
    ;;
esac
[[ "$TRAINER_DEVICES" == 1 ]] || {
  echo "High-noise continuation currently requires TRAINER_DEVICES=1" >&2
  exit 2
}
[[ "$WARMUP_STEPS" == 2500 && "$TOKEN_BIAS_WARMUP_STEPS" == 5000 ]] || {
  echo "High-noise continuation requires the preserved 2500/5000 warmup clocks" >&2
  exit 2
}
[[ -z "$CHECKPOINT_EVERY_N_STEPS" ]] || {
  echo "Task1 H0/V1 uses explicit full-state checkpoint milestones" >&2
  exit 2
}
[[ "$TRAINING_TIME_SAMPLING" == v1_staged_quota32 ]] || {
  echo "Selected continuation requires V1 quota sampling" >&2
  exit 2
}
[[ "$CHECKPOINT_MILESTONE_STEPS" == 18000,24000,30000 ]] || {
  echo "Task1 V1 checkpoint milestones must be 18000,24000,30000" >&2
  exit 2
}
[[ "$SOURCE_GLOBAL_STEP" == 15000 ]] || {
  echo "Task1 H0/V1 continuation source step must be 15000" >&2
  exit 2
}
case "$TARGET_MAX_STEPS" in
  ''|*[!0-9]*)
    echo "TARGET_MAX_STEPS must be the final optimizer step in (source, 50000]" >&2
    exit 2
    ;;
esac
if [[ "$TARGET_MAX_STEPS" != 30000 ]]; then
  echo "Task1 H0/V1 TARGET_MAX_STEPS must be the final global step 30000" >&2
  exit 2
fi
if [[ ! -s "$RESUME_CHECKPOINT_PATH" ]]; then
  echo "Resume checkpoint is missing or empty: $RESUME_CHECKPOINT_PATH" >&2
  exit 1
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "High-noise output directory already exists: $RUN_DIR" >&2
  exit 2
fi

python_bin="${TASK1_PYTHON_BIN:-python}"
if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Task1 checkpoint preflight Python is unavailable: $python_bin" >&2
  exit 1
fi
"$python_bin" "$script_dir/verify_task1_checkpoint.py" \
  --checkpoint "$RESUME_CHECKPOINT_PATH" \
  --expected-step "$SOURCE_GLOBAL_STEP" \
  --expected-target-lr 6e-4 \
  --expected-global-batch 256 \
  --require-task1-a

export EXPERIMENT_STAGE=task1_v1_continuation
export EXPERIMENT_CANDIDATE_ID=v1_from_15000
exec "$script_dir/resume_task1_vocab_mse.sh"
