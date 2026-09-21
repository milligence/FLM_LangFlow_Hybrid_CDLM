#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config_path="${TASK1_TRAIN_CONFIG:?Set TASK1_TRAIN_CONFIG to the M-tau25 contract}"
# shellcheck source=task1_train_config.sh
source "$script_dir/task1_train_config.sh"
task1_load_train_config "$config_path"

: "${RESUME_CHECKPOINT_PATH:?Set the V1@30k full-state checkpoint path}"
: "${RUN_DIR:?Set a new arm output directory}"
: "${TASK1_RUN_ID:?Set a unique arm run ID}"
: "${STOP_MARKER_PATH:?Set the arm-specific graceful-stop marker path}"
: "${PLANNED_STOP_MARKER_PATH:?Set the arm-specific planned-stop marker path}"
: "${TRANSFER_QUEUE_DIR:?Set the arm-specific transfer queue directory}"
: "${LAUNCHER_LOG_PATH:?Set the arm-specific launcher log path}"

[[ "$TASK1_VARIANT" == A ]] || { echo 'V1 continuation requires TASK1_VARIANT=A' >&2; exit 2; }
[[ "$SOURCE_GLOBAL_STEP" == 30000 && "$TARGET_MAX_STEPS" == 45000 ]] || {
  echo 'V1 continuation requires source step 30000 and target step 45000' >&2; exit 2; }
[[ "$SOURCE_LEARNING_RATE" == 6e-4 ]] || {
  echo 'V1@30k source LR must be the verified 6e-4' >&2; exit 2; }
[[ -z "$RESUME_CONSTANT_LEARNING_RATE" ]] || {
  echo 'V1 continuation must use the explicit transition, not constant LR' >&2; exit 2; }
[[ "$RESUME_TARGET_LEARNING_RATE" == 3e-4 \
   && "$RESUME_LR_TRANSITION_STEPS" == 500 ]] || {
  echo 'V1 continuation requires 500 updates from 6e-4 to 3e-4' >&2; exit 2; }
[[ "$GLOBAL_BATCH_SIZE/$MICRO_BATCH_SIZE/$ACCUMULATE_GRAD_BATCHES" == 256/32/8 ]] || {
  echo 'V1 continuation requires global/micro/accumulation 256/32/8' >&2; exit 2; }
[[ "$TRAINER_DEVICES" == 1 ]] || { echo 'V1 continuation requires one device' >&2; exit 2; }
[[ "$CHECKPOINT_MILESTONE_STEPS" == 35000,40000,45000 \
   && -z "$CHECKPOINT_EVERY_N_STEPS" ]] || {
  echo 'V1 continuation milestones must be 35000,40000,45000' >&2; exit 2; }
[[ "$TRAINING_TIME_SAMPLING/$TASK1_ARM_ID" == v1_m_tau25_global256/M-tau25 ]] || {
  echo 'Selected continuation requires the M-tau25 sampler' >&2
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
"$python_bin" "$script_dir/verify_task1_checkpoint.py" \
  --checkpoint "$RESUME_CHECKPOINT_PATH" \
  --expected-step 30000 \
  --expected-target-lr 6e-4 \
  --expected-global-batch 256 \
  --require-task1-a

export MILESTONE_LAST_EVERY_N_STEPS=1000
export EXPERIMENT_STAGE=task1_v1_30k_45k_continuation
export EXPERIMENT_CANDIDATE_ID="$TASK1_ARM_ID"
exec "$script_dir/resume_task1_vocab_mse.sh"
