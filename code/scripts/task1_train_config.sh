#!/usr/bin/env bash

# Shared loader and batch validation for Task1 A fresh training and full-state
# continuation. This file defines functions and must be sourced by an entrypoint.

task1_load_train_config() {
  local config_path="$1"
  local key
  local line
  local line_number=0
  local name
  local index
  local -a names=(
    HARDWARE_PROFILE
    TRAINER_DEVICES
    GLOBAL_BATCH_SIZE
    MICRO_BATCH_SIZE
    ACCUMULATE_GRAD_BATCHES
    TRAINING_TIME_SAMPLING
    RUN_DIR
    MAX_STEPS
    EXPERIMENT_RUN_ID
    LEARNING_RATE
    OPTIM_BETA1
    OPTIM_BETA2
    OPTIM_EPS
    OPTIM_WEIGHT_DECAY
    EMA_DECAY
    WARMUP_STEPS
    TOKEN_BIAS_WARMUP_STEPS
    EVAL_BATCH_SIZE
    CHECKPOINT_EVERY_N_STEPS
    CHECKPOINT_MILESTONE_STEPS
    RESUME_CHECKPOINT_PATH
    SOURCE_LEARNING_RATE
    SOURCE_GLOBAL_STEP
    TASK1_VARIANT
    TARGET_MAX_STEPS
    TASK1_RUN_ID
    RESUME_CONSTANT_LEARNING_RATE
    RESUME_TARGET_LEARNING_RATE
    RESUME_LR_TRANSITION_STEPS
    RESUME_LR_TRANSITION_SCHEDULE
    TASK1_TRAINING_RNG_SEED
    STOP_MARKER_PATH
    TRANSFER_QUEUE_DIR
    LAUNCHER_LOG_PATH
    TASK1_ARM_ID
  )
  local -a override_names=()
  local -a override_values=()

  if [[ ! -r "$config_path" ]]; then
    echo "Task1 train config is not readable: $config_path" >&2
    return 2
  fi

  # Preserve the existing environment interface: explicit per-run values win
  # over the tracked defaults in the config document.
  for name in "${names[@]}"; do
    if declare -p "$name" >/dev/null 2>&1; then
      override_names+=("$name")
      override_values+=("${!name}")
    fi
  done

  while IFS= read -r line || [[ -n "$line" ]]; do
    ((line_number += 1))
    case "$line" in
      ''|'#'*) continue ;;
      *=*) ;;
      *)
        echo "Invalid Task1 config line $line_number: expected KEY=value" >&2
        return 2
        ;;
    esac
    key="${line%%=*}"
    case "$key" in
      HARDWARE_PROFILE|TRAINER_DEVICES|GLOBAL_BATCH_SIZE|MICRO_BATCH_SIZE|\
      ACCUMULATE_GRAD_BATCHES|TRAINING_TIME_SAMPLING|RUN_DIR|MAX_STEPS|EXPERIMENT_RUN_ID|\
      LEARNING_RATE|OPTIM_BETA1|OPTIM_BETA2|\
      OPTIM_EPS|OPTIM_WEIGHT_DECAY|EMA_DECAY|WARMUP_STEPS|\
      TOKEN_BIAS_WARMUP_STEPS|EVAL_BATCH_SIZE|CHECKPOINT_EVERY_N_STEPS|CHECKPOINT_MILESTONE_STEPS|\
      RESUME_CHECKPOINT_PATH|\
      SOURCE_LEARNING_RATE|SOURCE_GLOBAL_STEP|TASK1_VARIANT|\
      TARGET_MAX_STEPS|TASK1_RUN_ID|RESUME_CONSTANT_LEARNING_RATE|\
      RESUME_TARGET_LEARNING_RATE|RESUME_LR_TRANSITION_STEPS|\
      RESUME_LR_TRANSITION_SCHEDULE|\
      TASK1_TRAINING_RNG_SEED|STOP_MARKER_PATH|TRANSFER_QUEUE_DIR|\
      LAUNCHER_LOG_PATH|TASK1_ARM_ID) ;;
      *)
        echo "Unknown Task1 config key at line $line_number: $key" >&2
        return 2
        ;;
    esac
    printf -v "$key" '%s' "${line#*=}"
  done < "$config_path"

  for ((index = 0; index < ${#override_names[@]}; index++)); do
    printf -v "${override_names[$index]}" '%s' "${override_values[$index]}"
  done
  for name in "${names[@]}"; do
    export "$name"
  done
  export TASK1_TRAIN_CONFIG_RESOLVED="$config_path"
}

task1_resolve_batch() {
  local profile="$1"
  local profile_micro_batch
  local requested_micro_batch="${MICRO_BATCH_SIZE:-auto}"
  local requested_accumulation="${ACCUMULATE_GRAD_BATCHES:-auto}"
  local derived_accumulation

  case "$profile" in
    4090) profile_micro_batch=8 ;;
    h800) profile_micro_batch=64 ;;
    *)
      echo "Hardware profile must be 4090 or h800" >&2
      return 2
      ;;
  esac

  case "${GLOBAL_BATCH_SIZE:-}" in
    ''|*[!0-9]*)
      echo "GLOBAL_BATCH_SIZE must be an integer in [1, 256]" >&2
      return 2
      ;;
  esac
  if (( GLOBAL_BATCH_SIZE < 1 || GLOBAL_BATCH_SIZE > 256 )); then
    echo "GLOBAL_BATCH_SIZE must be an integer in [1, 256]" >&2
    return 2
  fi

  if [[ "$requested_micro_batch" == auto ]]; then
    if (( GLOBAL_BATCH_SIZE < profile_micro_batch )); then
      requested_micro_batch="$GLOBAL_BATCH_SIZE"
    else
      requested_micro_batch="$profile_micro_batch"
    fi
  fi
  case "$requested_micro_batch" in
    ''|*[!0-9]*)
      echo "MICRO_BATCH_SIZE must be auto or a positive integer" >&2
      return 2
      ;;
  esac
  if (( requested_micro_batch < 1
        || requested_micro_batch > GLOBAL_BATCH_SIZE
        || GLOBAL_BATCH_SIZE % requested_micro_batch != 0 )); then
    echo "MICRO_BATCH_SIZE must be a positive divisor of GLOBAL_BATCH_SIZE" >&2
    return 2
  fi

  MICRO_BATCH_SIZE="$requested_micro_batch"
  derived_accumulation="$((GLOBAL_BATCH_SIZE / MICRO_BATCH_SIZE))"
  if [[ "$requested_accumulation" != auto ]]; then
    case "$requested_accumulation" in
      ''|*[!0-9]*)
        echo "ACCUMULATE_GRAD_BATCHES must be auto or a positive integer" >&2
        return 2
        ;;
    esac
    if (( requested_accumulation != derived_accumulation )); then
      echo "ACCUMULATE_GRAD_BATCHES must equal GLOBAL_BATCH_SIZE / MICRO_BATCH_SIZE" >&2
      return 2
    fi
  fi
  ACCUMULATE_GRAD_BATCHES="$derived_accumulation"
  HARDWARE_PROFILE="$profile"
  export HARDWARE_PROFILE GLOBAL_BATCH_SIZE MICRO_BATCH_SIZE
  export ACCUMULATE_GRAD_BATCHES
}
