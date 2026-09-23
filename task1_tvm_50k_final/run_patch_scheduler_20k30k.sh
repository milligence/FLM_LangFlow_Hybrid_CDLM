#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${TASK1_BINDINGS:?Set TASK1_BINDINGS to runtime_bindings.json}"
: "${TASK1_DRIVER_PYTHON:?Set TASK1_DRIVER_PYTHON to the declared Python executable}"
: "${RUN_ROOT:?Set RUN_ROOT to the F run directory}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to review_30k/scheduler}"

mkdir -p "$OUTPUT_ROOT"
for STEP in 20000 30000; do
  CHECKPOINT="$RUN_ROOT/checkpoints/step_$(printf '%06d' "$STEP").ckpt"
  [[ -f "$CHECKPOINT" ]] || {
    echo "Missing exact checkpoint: $CHECKPOINT" >&2
    exit 3
  }
  for SPEC in \
      'C|[0.0,0.3392770637,0.5814685447,0.85,0.95]' \
      'D|[0.0,0.527129195498412,0.776393202250021,0.894262873655944,0.95]'; do
    NAME="${SPEC%%|*}"
    GRID="${SPEC#*|}"
    JOB="$OUTPUT_ROOT/step_$(printf '%06d' "$STEP")/grid_${NAME}"
    mkdir -p "$JOB"
    if [[ ! -f "$JOB/samples.json" ]]; then
      "$TASK1_DRIVER_PYTHON" "$ROOT/tools/run_20k_generation.py" \
        --bindings "$TASK1_BINDINGS" --contract-dir "$ROOT" --line f \
        --checkpoint "$CHECKPOINT" --model-type eval_ema \
        --mode finite --nfe 4 --samples 128 --batch-size 32 \
        --seed 424242 --grid "$GRID" --output "$JOB/samples.json" \
        >"$JOB/evaluation.log" 2>&1
    fi
  done
done
touch "$OUTPUT_ROOT/SCHEDULER_20K30K_COMPLETE"
