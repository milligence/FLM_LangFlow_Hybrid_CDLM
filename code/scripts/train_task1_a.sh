#!/usr/bin/env bash
set -euo pipefail

# Stable single-GPU entrypoint for the current Task1 A baseline. Batch and
# optimizer controls come from one config document; model, data, and loss stay
# fixed here.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config_path="${TASK1_TRAIN_CONFIG:-$script_dir/../configs/task1_a_train.env}"
# shellcheck source=task1_train_config.sh
source "$script_dir/task1_train_config.sh"
task1_load_train_config "$config_path"

if (( $# > 1 )); then
  echo "Usage: $0 [4090|h800]" >&2
  exit 2
fi
hardware_profile="${1:-${HARDWARE_PROFILE:-h800}}"
task1_resolve_batch "$hardware_profile"

: "${RUN_DIR:?Set RUN_DIR to a new persistent output directory}"
: "${MAX_STEPS:?Set MAX_STEPS to the optimizer-step limit}"
: "${LEARNING_RATE:?Set LEARNING_RATE in the train config or environment}"

case "$MAX_STEPS" in
  ''|*[!0-9]*)
    echo "MAX_STEPS must be in [1, 50000]" >&2
    exit 2
    ;;
esac
if (( MAX_STEPS < 1 || MAX_STEPS > 50000 )); then
  echo "MAX_STEPS must be in [1, 50000]" >&2
  exit 2
fi

if [[ -e "$RUN_DIR" && -z "${DRY_RUN_COMMAND_PATH:-}" ]]; then
  echo "RUN_DIR already exists; fresh Task1 outputs are never overwritten" >&2
  exit 2
fi

export LOSS_VARIANT=task1_a
export DATA_CONFIG=openwebtext_327m_packed
export WARMUP_STEPS="${WARMUP_STEPS:-2500}"
export TOKEN_BIAS_WARMUP_STEPS="${TOKEN_BIAS_WARMUP_STEPS:-5000}"
export OPTIM_BETA1="${OPTIM_BETA1:-0.9}"
export OPTIM_BETA2="${OPTIM_BETA2:-0.999}"
export OPTIM_EPS="${OPTIM_EPS:-1e-8}"
export OPTIM_WEIGHT_DECAY="${OPTIM_WEIGHT_DECAY:-0}"
export EMA_DECAY="${EMA_DECAY:-0.9999}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
export THROUGHPUT_EVERY_N_STEPS="${THROUGHPUT_EVERY_N_STEPS:-10}"
export LOG_EVERY_N_STEPS="${LOG_EVERY_N_STEPS:-10}"
export MILESTONE_STEPS="${MILESTONE_STEPS:-[$MAX_STEPS]}"
export MILESTONE_SAVE_WEIGHTS_ONLY="${MILESTONE_SAVE_WEIGHTS_ONLY:-false}"
export MILESTONE_LAST_EVERY_N_STEPS="${MILESTONE_LAST_EVERY_N_STEPS:-1000}"

exec "$script_dir/train_owt_128_langflow_hybrid.sh"
