#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
config_path="${TASK1_TRAIN_CONFIG:?Set TASK1_TRAIN_CONFIG to a tracked small-step env file}"
: "${LOSS_VARIANT:?Set LOSS_VARIANT to the selected algorithm name}"

case "$LOSS_VARIANT" in
  task1_tvm_ce|task1_tvm_sc_repair|task1_tvm_joint_j0|\
  task1_posterior_tvm_a|task1_posterior_tvm_b|task1_posterior_tvm_local_only) ;;
  *) echo "Unsupported small-step LOSS_VARIANT: $LOSS_VARIANT" >&2; exit 2 ;;
esac

# shellcheck source=task1_train_config.sh
source "$script_dir/task1_train_config.sh"
task1_load_train_config "$config_path"
task1_resolve_batch "${HARDWARE_PROFILE:-5090}"

: "${RUN_DIR:?Set RUN_DIR to a new output directory}"
if [[ -e "$RUN_DIR" && "${RESUME_FROM_CKPT:-false}" != true ]]; then
  echo "RUN_DIR already exists for a fresh run: $RUN_DIR" >&2
  exit 2
fi
expected_global="$((TRAINER_DEVICES * MICRO_BATCH_SIZE * ACCUMULATE_GRAD_BATCHES))"
if (( expected_global != GLOBAL_BATCH_SIZE )); then
  echo "devices*microbatch*accumulation must equal GLOBAL_BATCH_SIZE" >&2
  exit 2
fi

export LOSS_VARIANT DATA_CONFIG MODEL_CONFIG STRATEGY_CONFIG
exec "$script_dir/train_owt_128_langflow_hybrid.sh"
