#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH}"
: "${CHECKPOINT_LABEL:?Set CHECKPOINT_LABEL}"
: "${IMPLEMENTATION_COMMIT:?Set IMPLEMENTATION_COMMIT}"
: "${RUN_DIR:?Set RUN_DIR}"

[[ "$CHECKPOINT_LABEL" == m40_stable_50k ]] || {
  echo "Selected release requires CHECKPOINT_LABEL=m40_stable_50k" >&2
  exit 2
}
if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Checkpoint is not a file: $CHECKPOINT_PATH" >&2
  exit 2
fi
if [[ -e "$RUN_DIR" || -e "${RUN_DIR}.launcher.log" ]]; then
  echo "RUN_DIR and launcher log must not already exist: $RUN_DIR" >&2
  exit 2
fi
mkdir -p "$(dirname "$RUN_DIR")"

set +e
LOSS_VARIANT=task1_a \
DATA_CONFIG=openwebtext_327m_packed \
NUM_SAMPLES=1024 \
SAMPLING_STEPS=512 \
SAMPLING_SOLVER=euler \
SAMPLING_TEMPERATURE=1.0 \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}" \
DISABLE_EMA=false \
GEN_PPL_COMPARABLE=false \
GEN_PPL_PROTOCOL_ID=owt128-gpt2large-genppl-diagnostic-m40-50k-uniform-t-512-s1024-trajectory-v1 \
COMPUTE_GEN_PPL=true \
EVALUATION_PHASE=task1_m40_50k_uniform_t_512_s1024_trajectory \
TASK1_TIME_GRID=physical_time_uniform_official_inverse_lut \
TASK1_DIAGNOSTIC_STEPS='[1]' \
TASK1_INITIAL_NOISE_SEED=42 \
TASK1_INITIAL_NOISE_SCHEDULE=base_seed_plus_sample_index \
TASK1_TRAJECTORY_DIAGNOSTICS=true \
TASK1_TRAJECTORY_CHECKPOINT_LABEL="$CHECKPOINT_LABEL" \
TASK1_TRAJECTORY_TOP_K=32 \
TASK1_TRAJECTORY_FP32_SAMPLE_COUNT=8 \
TASK1_TRAJECTORY_FP32_NODE_COUNT=17 \
RUN_DIR="$RUN_DIR" \
EVAL_MODEL_DIR="${EVAL_MODEL_DIR:-}" \
CHECKPOINT_PATH="$CHECKPOINT_PATH" \
  "$script_dir/eval_owt_128_langflow_hybrid.sh" 2>&1 | tee "${RUN_DIR}.launcher.log"
pipeline_status=("${PIPESTATUS[@]}")
set -e
(( pipeline_status[0] == 0 )) || exit "${pipeline_status[0]}"
(( pipeline_status[1] == 0 )) || exit "${pipeline_status[1]}"

"$python_bin" "$script_dir/verify_task1_m40_50k_trajectory.py" \
  "$RUN_DIR" \
  --checkpoint-label "$CHECKPOINT_LABEL" \
  --checkpoint-path "$CHECKPOINT_PATH" \
  --implementation-commit "$IMPLEMENTATION_COMMIT"
