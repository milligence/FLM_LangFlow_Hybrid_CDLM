#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config_path="${TASK1_TRAIN_CONFIG:?Set the M40-STABLE contract}"
# shellcheck source=task1_train_config.sh
source "$script_dir/task1_train_config.sh"
task1_load_train_config "$config_path"

: "${RESUME_CHECKPOINT_PATH:?Set the M40 full-state checkpoint path}"
: "${RUN_DIR:?Set a new arm output directory}"
: "${TASK1_RUN_ID:?Set a unique arm run ID}"
: "${STOP_MARKER_PATH:?Set the arm-specific graceful-stop marker path}"
: "${TRANSFER_QUEUE_DIR:?Set the arm-specific transfer queue directory}"
: "${LAUNCHER_LOG_PATH:?Set the arm-specific launcher log path}"

[[ "$TASK1_VARIANT" == A ]] || { echo 'M40 continuation requires TASK1_VARIANT=A' >&2; exit 2; }
[[ "$SOURCE_GLOBAL_STEP" == 40000 && "$TARGET_MAX_STEPS" == 50000 ]] || {
  echo 'M40 continuation requires source step 40000 and target step 50000' >&2; exit 2; }
[[ "$SOURCE_LEARNING_RATE" == 3e-4 ]] || {
  echo 'M40 source LR must be the verified effective 3e-4' >&2; exit 2; }
[[ "$GLOBAL_BATCH_SIZE/$MICRO_BATCH_SIZE/$ACCUMULATE_GRAD_BATCHES" == 256/32/8 ]] || {
  echo 'M40 continuation requires global/micro/accumulation 256/32/8' >&2; exit 2; }
[[ "$TRAINER_DEVICES" == 1 ]] || { echo 'M40 continuation requires one device' >&2; exit 2; }
[[ "$TRAINING_TIME_SAMPLING" == v1_m_tau25_global256 ]] || {
  echo 'Both M40 continuations must preserve the actual q_M sampler' >&2; exit 2; }
[[ "$CHECKPOINT_MILESTONE_STEPS" == 42500,45000,47500,50000 \
   && -z "$CHECKPOINT_EVERY_N_STEPS" ]] || {
  echo 'M40 continuation milestones must be 42500,45000,47500,50000' >&2; exit 2; }
[[ "$TASK1_ARM_ID" == M40-STABLE ]] || {
  echo 'Selected 50k continuation requires TASK1_ARM_ID=M40-STABLE' >&2
  exit 2
}
[[ "$RESUME_CONSTANT_LEARNING_RATE" == 3e-4 \
   && -z "$RESUME_TARGET_LEARNING_RATE" \
   && -z "$RESUME_LR_TRANSITION_STEPS" ]] || {
  echo 'M40-STABLE must retain constant 3e-4' >&2
  exit 2
}
if [[ ! -s "$RESUME_CHECKPOINT_PATH" ]]; then
  echo "Resume checkpoint is missing or empty: $RESUME_CHECKPOINT_PATH" >&2
  exit 1
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "Arm output directory already exists: $RUN_DIR" >&2
  exit 2
fi

python_bin="${TASK1_PYTHON_BIN:-python}"
resume_expected_checkpoint_step="${RESUME_EXPECTED_CHECKPOINT_STEP:-40000}"
[[ "$resume_expected_checkpoint_step" =~ ^[0-9]+$ \
   && "$resume_expected_checkpoint_step" -ge 40000 \
   && "$resume_expected_checkpoint_step" -lt 50000 ]] || {
  echo 'RESUME_EXPECTED_CHECKPOINT_STEP must be in [40000, 50000)' >&2
  exit 2
}
verify_args=(
  --checkpoint "$RESUME_CHECKPOINT_PATH"
  --expected-step "$resume_expected_checkpoint_step"
  --expected-target-lr 3e-4
  --expected-global-batch 256
  --require-task1-a
  --allow-completed-transition
)
"$python_bin" "$script_dir/verify_task1_checkpoint.py" \
  "${verify_args[@]}"

export MILESTONE_LAST_EVERY_N_STEPS=1000
export EXPERIMENT_STAGE=task1_m40_50k_continuation
export EXPERIMENT_CANDIDATE_ID="$TASK1_ARM_ID"
export EXPERIMENT_SELECTED_CHECKPOINT="$RESUME_CHECKPOINT_PATH"
export EXPERIMENT_RESUME_SOURCE_LR=3e-4
export EXPERIMENT_RESUME_SOURCE_STEP=40000
if (( resume_expected_checkpoint_step > 40000 )); then
  export RESUME_PRESERVE_CHECKPOINT_LR_SCHEDULER=1
fi
exec "$script_dir/resume_task1_vocab_mse.sh"
