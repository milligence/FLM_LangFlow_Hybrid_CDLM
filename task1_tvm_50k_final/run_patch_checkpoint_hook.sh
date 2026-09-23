#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${TASK1_BINDINGS:?Set TASK1_BINDINGS to runtime_bindings.json}"
: "${TASK1_DRIVER_PYTHON:?Set TASK1_DRIVER_PYTHON to the declared Python executable}"
: "${RUN_ROOT:?Set RUN_ROOT to the F run directory}"

STEP="${1:?Usage: run_patch_checkpoint_hook.sh STEP CHECKPOINT OUTPUT_ROOT}"
CHECKPOINT="${2:?Missing checkpoint}"
OUTPUT_ROOT="${3:?Missing output root}"
[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 3; }
mkdir -p "$OUTPUT_ROOT/diagnostics" "$OUTPUT_ROOT/contracts" "$OUTPUT_ROOT/training"

for ROLE in online target_ema eval_ema; do
  OUTPUT="$OUTPUT_ROOT/diagnostics/step_$(printf '%06d' "$STEP")__${ROLE}.json"
  if [[ ! -f "$OUTPUT" ]]; then
    EXTRA=()
    [[ "$ROLE" == online ]] && EXTRA=(--run-root "$RUN_ROOT")
    "$TASK1_DRIVER_PYTHON" "$ROOT/tools/run_20k_audit.py" \
      --bindings "$TASK1_BINDINGS" --contract-dir "$ROOT" \
      --audit-config "$ROOT/audit_patch.yaml" --line f \
      --checkpoint "$CHECKPOINT" --step "$STEP" --model-type "$ROLE" \
      --scope full --output "$OUTPUT" "${EXTRA[@]}"
  fi
done

TRAJECTORY="$OUTPUT_ROOT/trajectory_B"
if [[ ! -f "$TRAJECTORY/gate_diagnostic.json" ]]; then
  mkdir -p "$TRAJECTORY"
  "$TASK1_DRIVER_PYTHON" "$ROOT/tools/run_patch_trajectory_diagnostic.py" \
    --bindings "$TASK1_BINDINGS" --contract-dir "$ROOT" \
    --checkpoint "$CHECKPOINT" --step "$STEP" --grid-name B \
    --grid '[0.0,0.2375,0.475,0.7125,0.95]' \
    --output-dir "$TRAJECTORY" --seed 424242 --seed-count 16 --microbatch 2 \
    >"$TRAJECTORY/diagnostic.log" 2>&1
fi

env TASK1_BINDINGS="$TASK1_BINDINGS" \
  TASK1_DRIVER_PYTHON="$TASK1_DRIVER_PYTHON" \
  bash "$ROOT/run_patch_generation_hook.sh" \
    "$STEP" "$CHECKPOINT" "$OUTPUT_ROOT"

cp "$ROOT/audit_patch.yaml" "$OUTPUT_ROOT/contracts/audit_patch.yaml"
cp "$ROOT/train_f.yaml" "$OUTPUT_ROOT/contracts/train_f.yaml"
cp "$ROOT/sampler_f.yaml" "$OUTPUT_ROOT/contracts/sampler_f.yaml"
cp "$ROOT/eval.yaml" "$OUTPUT_ROOT/contracts/eval.yaml"
cp "$ROOT/resolved_delta_v2.yaml" "$OUTPUT_ROOT/contracts/resolved_delta_v2.yaml"
cp "$ROOT/diagnostic_record_v2.schema.json" \
  "$OUTPUT_ROOT/contracts/diagnostic_record_v2.schema.json"
for FILE in throughput.jsonl canonical_gate_events.jsonl \
    gradient_calibration_events.jsonl gradient_safety_audit_events.jsonl \
    f2k_optimizer_update_events.jsonl patch_events.jsonl; do
  [[ -f "$RUN_ROOT/$FILE" ]] && cp "$RUN_ROOT/$FILE" "$OUTPUT_ROOT/training/$FILE"
done
[[ -f "$RUN_ROOT/local_metrics/metrics.csv" ]] \
  && cp "$RUN_ROOT/local_metrics/metrics.csv" "$OUTPUT_ROOT/training/metrics.csv"
touch "$OUTPUT_ROOT/AUDIT_COMPLETE"
