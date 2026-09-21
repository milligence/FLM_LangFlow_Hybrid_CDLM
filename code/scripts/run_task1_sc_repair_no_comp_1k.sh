#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${PLANNED_STOP_MARKER_PATH:?Set a new planned-stop marker outside RUN_DIR}"
if [[ -e "$PLANNED_STOP_MARKER_PATH" ]]; then
  echo "Planned-stop marker already exists: $PLANNED_STOP_MARKER_PATH" >&2
  exit 2
fi
mkdir -p "$(dirname "$PLANNED_STOP_MARKER_PATH")"
printf '%s\n' \
  '{"reason":"no_comp_release_stop_at_1000","stop_after_optimizer_step":1000}' \
  > "$PLANNED_STOP_MARKER_PATH"

export LOSS_VARIANT=task1_tvm_sc_repair
export TASK1_TRAIN_CONFIG="${TASK1_TRAIN_CONFIG:-$script_dir/../configs/task1_tvm_sc_repair_1k.env}"
export MAX_STEPS=2000
export TVM_TRAINING_BUDGET_STEPS=2000
export MILESTONE_STEPS='[500,1000,2000]'
export PLANNED_STOP_MARKER_PATH
exec "$script_dir/train_task1_small_step.sh"
