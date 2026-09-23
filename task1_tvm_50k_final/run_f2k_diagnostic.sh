#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${TASK1_BINDINGS:?Set TASK1_BINDINGS to runtime_bindings.json}"
: "${TASK1_DRIVER_PYTHON:?Set TASK1_DRIVER_PYTHON to the declared Python executable}"

STEP="${1:?Usage: run_f2k_diagnostic.sh STEP CHECKPOINT RUN_ROOT OUTPUT_ROOT}"
CHECKPOINT="${2:?Missing checkpoint path}"
RUN_ROOT="${3:?Missing training run root}"
OUTPUT_ROOT="${4:?Missing diagnostic output root}"

if (( STEP < 32000 || STEP > 50000 || STEP % 2000 != 0 )); then
  echo "F 2k diagnostic step must be one of 32000,34000,...,50000" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Temporary full-state checkpoint is missing: $CHECKPOINT" >&2
  exit 3
fi

mkdir -p "$OUTPUT_ROOT/diagnostics" "$OUTPUT_ROOT/generation" \
  "$OUTPUT_ROOT/training" "$OUTPUT_ROOT/contracts"

for MODEL_TYPE in online target_ema eval_ema; do
  OUTPUT="$OUTPUT_ROOT/diagnostics/step_$(printf '%06d' "$STEP")__${MODEL_TYPE}.json"
  if [[ ! -f "$OUTPUT" ]]; then
    EXTRA=()
    if [[ "$MODEL_TYPE" == online ]]; then
      EXTRA=(--run-root "$RUN_ROOT")
    fi
    "$TASK1_DRIVER_PYTHON" "$ROOT/tools/run_20k_audit.py" \
      --bindings "$TASK1_BINDINGS" --contract-dir "$ROOT" \
      --audit-config "$ROOT/audit_f2k.yaml" --line f \
      --checkpoint "$CHECKPOINT" --step "$STEP" \
      --model-type "$MODEL_TYPE" --scope full --output "$OUTPUT" \
      "${EXTRA[@]}"
  fi
done

run_generation() {
  local NAME="$1"
  local MODEL_TYPE="$2"
  local MODE="$3"
  local NFE="$4"
  local GRID="$5"
  local K="${6:-}"
  local JOB_ROOT="$OUTPUT_ROOT/generation/$NAME"
  local RESULT="$JOB_ROOT/samples.json"
  mkdir -p "$JOB_ROOT"
  if [[ -f "$RESULT" ]]; then
    return
  fi
  local EXTRA=()
  if [[ -n "$K" ]]; then
    EXTRA=(--canonical-k "$K")
  fi
  "$TASK1_DRIVER_PYTHON" "$ROOT/tools/run_20k_generation.py" \
    --bindings "$TASK1_BINDINGS" --contract-dir "$ROOT" --line f \
    --checkpoint "$CHECKPOINT" --model-type "$MODEL_TYPE" \
    --mode "$MODE" --nfe "$NFE" --samples 128 --batch-size 32 \
    --seed 424242 --grid "$GRID" --output "$RESULT" "${EXTRA[@]}" \
    >"$JOB_ROOT/evaluation.log" 2>&1
}

run_generation highnfe_512 eval_ema legacy_rolling_T1 512 '[0.0,0.95]'
run_generation fewstep_1 eval_ema finite 1 '[0.0,0.95]'
run_generation fewstep_4_deployment_grid_1 eval_ema finite 4 \
  '[0.0,0.3392770637,0.5814685447,0.7246687714,0.95]'
run_generation fewstep_4_deployment_grid_2 eval_ema finite 4 \
  '[0.0,0.2375,0.475,0.7125,0.95]'
for K in 1 2 4; do
  run_generation "sc_k${K}" target_ema canonical_fixed_budget_T095 128 \
    '[0.0,0.95]' "$K"
done

cp "$ROOT/audit_f2k.yaml" "$OUTPUT_ROOT/contracts/audit_f2k.yaml"
cp "$ROOT/eval.yaml" "$OUTPUT_ROOT/contracts/eval.yaml"
cp "$ROOT/train_f.yaml" "$OUTPUT_ROOT/contracts/train_f.yaml"
cp "$ROOT/sampler_f.yaml" "$OUTPUT_ROOT/contracts/sampler_f.yaml"
for FILE in throughput.jsonl canonical_gate_events.jsonl \
    gradient_calibration_events.jsonl gradient_safety_audit_events.jsonl \
    f2k_optimizer_update_events.jsonl; do
  if [[ -f "$RUN_ROOT/$FILE" ]]; then
    cp "$RUN_ROOT/$FILE" "$OUTPUT_ROOT/training/$FILE"
  fi
done
if [[ -f "$RUN_ROOT/local_metrics/metrics.csv" ]]; then
  cp "$RUN_ROOT/local_metrics/metrics.csv" "$OUTPUT_ROOT/training/metrics.csv"
fi

"$TASK1_DRIVER_PYTHON" "$ROOT/tools/finalize_f2k_audit.py" \
  --step "$STEP" --checkpoint "$CHECKPOINT" --output-root "$OUTPUT_ROOT" \
  --config "$ROOT/audit_f2k.yaml"
touch "$OUTPUT_ROOT/AUDIT_COMPLETE"
