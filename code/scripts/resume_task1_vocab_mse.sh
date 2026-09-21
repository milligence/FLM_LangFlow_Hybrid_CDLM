#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config_path="${TASK1_TRAIN_CONFIG:-$script_dir/../configs/task1_a_train.env}"
# shellcheck source=task1_train_config.sh
source "$script_dir/task1_train_config.sh"
task1_load_train_config "$config_path"

if [[ "${RESUME_PRESERVE_CHECKPOINT_LR_SCHEDULER:-0}" == 1 ]]; then
  unset RESUME_CONSTANT_LEARNING_RATE RESUME_TARGET_LEARNING_RATE
  unset RESUME_LR_TRANSITION_STEPS RESUME_LR_TRANSITION_SCHEDULE
fi

: "${RESUME_CHECKPOINT_PATH:?Set RESUME_CHECKPOINT_PATH to a full Task1 checkpoint}"
: "${SOURCE_LEARNING_RATE:?Set SOURCE_LEARNING_RATE from the checkpoint manifest (3e-4 or 6e-4)}"
: "${SOURCE_GLOBAL_STEP:?Set SOURCE_GLOBAL_STEP from the checkpoint manifest}"
: "${TASK1_VARIANT:?Set TASK1_VARIANT=A}"
: "${TARGET_MAX_STEPS:?Set the new absolute optimizer-step ceiling above the checkpoint step}"
: "${TASK1_RUN_ID:?Set a unique continuation run ID}"
: "${RUN_DIR:?Set a new continuation output directory}"

if [[ ! -s "$RESUME_CHECKPOINT_PATH" ]]; then
  echo "Resume checkpoint is missing or empty: $RESUME_CHECKPOINT_PATH" >&2
  exit 1
fi
[[ "$TASK1_VARIANT" == A || "$TASK1_VARIANT" == a ]] || {
  echo "This release supports only TASK1_VARIANT=A" >&2
  exit 2
}
export LOSS_VARIANT=task1_a
case "$SOURCE_LEARNING_RATE" in
  3e-4|6e-4) ;;
  *) echo "SOURCE_LEARNING_RATE must be 3e-4 or 6e-4" >&2; exit 2 ;;
esac
if [[ -n "${TASK1_MICRO_BATCH_SIZE:-}" ]]; then
  MICRO_BATCH_SIZE="$TASK1_MICRO_BATCH_SIZE"
fi
task1_resolve_batch "${HARDWARE_PROFILE:-h800}"
case "$TARGET_MAX_STEPS" in
  ''|*[!0-9]*) echo "TARGET_MAX_STEPS must be a positive integer" >&2; exit 2 ;;
esac
case "$SOURCE_GLOBAL_STEP" in
  ''|*[!0-9]*) echo "SOURCE_GLOBAL_STEP must be a positive integer" >&2; exit 2 ;;
esac
if (( SOURCE_GLOBAL_STEP <= 0 || TARGET_MAX_STEPS <= SOURCE_GLOBAL_STEP )); then
  echo "TARGET_MAX_STEPS must exceed SOURCE_GLOBAL_STEP" >&2
  exit 2
fi
if [[ -e "$RUN_DIR" ]]; then
  echo "Continuation output directory already exists: $RUN_DIR" >&2
  exit 2
fi

target_lr="${RESUME_CONSTANT_LEARNING_RATE:-${RESUME_TARGET_LEARNING_RATE:-$SOURCE_LEARNING_RATE}}"
python_bin="${TASK1_PYTHON_BIN:-python}"
"$python_bin" - "$target_lr" "${RESUME_LR_TRANSITION_STEPS:-}" \
  "${RESUME_CONSTANT_LEARNING_RATE:-}" \
  "${RESUME_LR_TRANSITION_SCHEDULE:-linear}" <<'PY'
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError as error:
    raise SystemExit(
        'RESUME_CONSTANT_LEARNING_RATE must be a positive finite number') from error
if not math.isfinite(value) or value <= 0.0:
    raise SystemExit(
        'RESUME_CONSTANT_LEARNING_RATE must be a positive finite number')
transition_steps, constant, schedule = sys.argv[2:]
if transition_steps:
    if constant:
        raise SystemExit(
            'Constant LR and transition LR are mutually exclusive')
    if not transition_steps.isdigit() or int(transition_steps) <= 0:
        raise SystemExit(
            'RESUME_LR_TRANSITION_STEPS must be a positive integer')
    if schedule not in {'linear', 'cosine'}:
        raise SystemExit(
            'RESUME_LR_TRANSITION_SCHEDULE must be linear or cosine')
elif schedule != 'linear':
    raise SystemExit(
        'RESUME_LR_TRANSITION_SCHEDULE requires a transition')
PY

# Lightning restores model, EMA, optimizer moments, global step, loops, and
# sampler position. When an override is explicit, the model hook grafts only
# the optimizer/scheduler LR onto a constant post-warmup schedule.
if [[ -n "${RESUME_LR_TRANSITION_STEPS:-}" ]]; then
  export LEARNING_RATE="$SOURCE_LEARNING_RATE"
else
  export LEARNING_RATE="$target_lr"
fi
export MAX_STEPS="$TARGET_MAX_STEPS"
export DATA_CONFIG=openwebtext_327m_packed
export EVAL_BATCH_SIZE="${TASK1_EVAL_BATCH_SIZE:-2}"
export WARMUP_STEPS="${WARMUP_STEPS:-2500}"
export TOKEN_BIAS_WARMUP_STEPS="${TOKEN_BIAS_WARMUP_STEPS:-5000}"
export OPTIM_BETA1="${OPTIM_BETA1:-0.9}"
export OPTIM_BETA2="${OPTIM_BETA2:-0.999}"
export OPTIM_EPS="${OPTIM_EPS:-1e-8}"
export OPTIM_WEIGHT_DECAY="${OPTIM_WEIGHT_DECAY:-0}"
export EMA_DECAY="${EMA_DECAY:-0.9999}"
export VALIDATION_INTERVAL_STEPS="${VALIDATION_INTERVAL_STEPS:-1000}"
export LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-8}"
export VALIDATE_BEFORE_TRAINING=false
export DIAGNOSTIC_INTERVAL_STEPS=1000
export GRADIENT_ROUTE_INTERVAL_STEPS=100
export GRADIENT_LOG_EVERY_N_STEPS=0
export THROUGHPUT_EVERY_N_STEPS=10
export LOG_EVERY_N_STEPS=8
export RESUME_FROM_CKPT=true
export RESUME_CKPT_PATH="$RESUME_CHECKPOINT_PATH"
export MILESTONE_STEPS="[$TARGET_MAX_STEPS]"
export MILESTONE_SAVE_WEIGHTS_ONLY=false
if [[ -n "${CHECKPOINT_EVERY_N_STEPS:-}" \
      && -n "${CHECKPOINT_MILESTONE_STEPS:-}" ]]; then
  echo "Set only one checkpoint cadence or explicit milestones" >&2
  exit 2
fi
if [[ -n "${CHECKPOINT_MILESTONE_STEPS:-}" ]]; then
  IFS=',' read -r -a requested_milestones \
    <<< "$CHECKPOINT_MILESTONE_STEPS"
  milestone_steps='['
  separator=''
  previous_step="$SOURCE_GLOBAL_STEP"
  for milestone in "${requested_milestones[@]}"; do
    case "$milestone" in
      ''|*[!0-9]*)
        echo "Checkpoint milestones must be ascending comma-separated integers" >&2
        exit 2
        ;;
    esac
    if (( milestone <= previous_step || milestone > TARGET_MAX_STEPS )); then
      echo "Checkpoint milestones must be ascending in (source, target]" >&2
      exit 2
    fi
    milestone_steps+="$separator$milestone"
    separator=','
    previous_step="$milestone"
  done
  if [[ "$previous_step" != "$TARGET_MAX_STEPS" ]]; then
    echo "The final checkpoint milestone must equal TARGET_MAX_STEPS" >&2
    exit 2
  fi
  milestone_steps+=']'
  export MILESTONE_STEPS="$milestone_steps"
elif [[ -n "${CHECKPOINT_EVERY_N_STEPS:-}" ]]; then
  case "$CHECKPOINT_EVERY_N_STEPS" in
    ''|*[!0-9]*)
      echo "CHECKPOINT_EVERY_N_STEPS must be a positive integer" >&2
      exit 2
      ;;
  esac
  if (( CHECKPOINT_EVERY_N_STEPS <= 0
        || SOURCE_GLOBAL_STEP % CHECKPOINT_EVERY_N_STEPS != 0 )); then
    echo "Checkpoint cadence must divide the source optimizer step" >&2
    exit 2
  fi
  milestone_steps='['
  milestone=$((SOURCE_GLOBAL_STEP + CHECKPOINT_EVERY_N_STEPS))
  separator=''
  last_milestone=''
  while (( milestone <= TARGET_MAX_STEPS )); do
    milestone_steps+="$separator$milestone"
    separator=','
    last_milestone="$milestone"
    milestone=$((milestone + CHECKPOINT_EVERY_N_STEPS))
  done
  if [[ "$last_milestone" != "$TARGET_MAX_STEPS" ]]; then
    milestone_steps+="$separator$TARGET_MAX_STEPS"
  fi
  milestone_steps+=']'
  export MILESTONE_STEPS="$milestone_steps"
fi
export MILESTONE_LAST_EVERY_N_STEPS="${MILESTONE_LAST_EVERY_N_STEPS:-0}"
export EXPERIMENT_RUN_ID="$TASK1_RUN_ID"
export EXPERIMENT_STAGE="${EXPERIMENT_STAGE:-task1_full_state_continuation}"
export EXPERIMENT_PHASE=task1_checkpoint_resume
export EXPERIMENT_SELECTED_CHECKPOINT="$RESUME_CHECKPOINT_PATH"
export EXPERIMENT_RESUME_SOURCE_LR="$SOURCE_LEARNING_RATE"
export EXPERIMENT_RESUME_SOURCE_STEP="$SOURCE_GLOBAL_STEP"

if [[ -n "${RESUME_CONSTANT_LEARNING_RATE:-}" ]]; then
  export RESUME_CONSTANT_LEARNING_RATE="$target_lr"
else
  unset RESUME_CONSTANT_LEARNING_RATE
fi
unset FINETUNE_PATH TOKEN_BIAS_SCHEDULE

exec "$script_dir/train_owt_128_langflow_hybrid.sh"
