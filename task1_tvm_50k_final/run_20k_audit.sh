#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${TASK1_BINDINGS:?Set TASK1_BINDINGS to resolved runtime_bindings.json}"
: "${TASK1_DRIVER_PYTHON:?Set TASK1_DRIVER_PYTHON to the declared Python executable}"
: "${TASK1_CONTRACT_DIR:=$ROOT}"

LINE="${1:?Usage: run_20k_audit.sh f|p CHECKPOINT RUN_ROOT OUTPUT_ROOT}"
CHECKPOINT="${2:?Missing 20k checkpoint path}"
RUN_ROOT="${3:?Missing training run root}"
OUTPUT_ROOT="${4:?Missing audit output root}"
if [[ "$LINE" != f && "$LINE" != p ]]; then
  echo "line must be f or p" >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT" || ! -f "$CHECKPOINT.sha256" ]]; then
  echo "20k checkpoint or SHA sidecar is missing" >&2
  exit 3
fi

mkdir -p "$OUTPUT_ROOT/diagnostics" "$OUTPUT_ROOT/trajectory" \
  "$OUTPUT_ROOT/generation" "$OUTPUT_ROOT/training" "$OUTPUT_ROOT/contracts"

for MODEL_TYPE in online target_ema eval_ema; do
  OUTPUT="$OUTPUT_ROOT/diagnostics/step_020000__${MODEL_TYPE}.json"
  if [[ ! -f "$OUTPUT" ]]; then
    "$TASK1_DRIVER_PYTHON" "$ROOT/tools/run_20k_audit.py" \
      --bindings "$TASK1_BINDINGS" \
      --contract-dir "$TASK1_CONTRACT_DIR" \
      --audit-config "$ROOT/audit_20k.yaml" \
      --line "$LINE" --checkpoint "$CHECKPOINT" --step 20000 \
      --model-type "$MODEL_TYPE" --scope full --output "$OUTPUT"
  fi
done

for STEP in 5000 10000 15000 20000; do
  CHECKPOINT_AT_STEP="$RUN_ROOT/checkpoints/step_$(printf '%06d' "$STEP").ckpt"
  OUTPUT="$OUTPUT_ROOT/trajectory/step_$(printf '%06d' "$STEP")__eval_ema.json"
  if [[ -f "$CHECKPOINT_AT_STEP" && -f "$CHECKPOINT_AT_STEP.sha256" && ! -f "$OUTPUT" ]]; then
    "$TASK1_DRIVER_PYTHON" "$ROOT/tools/run_20k_audit.py" \
      --bindings "$TASK1_BINDINGS" \
      --contract-dir "$TASK1_CONTRACT_DIR" \
      --audit-config "$ROOT/audit_20k.yaml" \
      --line "$LINE" --checkpoint "$CHECKPOINT_AT_STEP" --step "$STEP" \
      --model-type eval_ema --scope trajectory --output "$OUTPUT"
  fi
done

run_generation() {
  local NAME="$1"
  local MODE="$2"
  local NFE="$3"
  local GRID="$4"
  local JOB_ROOT="$OUTPUT_ROOT/generation/$NAME"
  local RESULT="$JOB_ROOT/samples.json"
  mkdir -p "$JOB_ROOT"
  if [[ ! -f "$RESULT" ]]; then
    "$TASK1_DRIVER_PYTHON" "$ROOT/tools/run_20k_generation.py" \
      --bindings "$TASK1_BINDINGS" \
      --contract-dir "$TASK1_CONTRACT_DIR" \
      --line "$LINE" --checkpoint "$CHECKPOINT" \
      --model-type eval_ema --mode "$MODE" --nfe "$NFE" \
      --samples 128 --batch-size 32 --seed 424242 \
      --grid "$GRID" --output "$RESULT" \
      >"$JOB_ROOT/evaluation.log" 2>&1
  fi
}

run_generation highnfe_512 legacy_rolling_T1 512 '[0.0,0.95]'
run_generation fewstep_1 finite 1 '[0.0,0.95]'
run_generation fewstep_4_deployment_grid_1 finite 4 \
  '[0.0,0.3392770637,0.5814685447,0.7246687714,0.95]'
run_generation fewstep_4_deployment_grid_2 finite 4 \
  '[0.0,0.2375,0.475,0.7125,0.95]'

cp "$ROOT/audit_20k.yaml" "$OUTPUT_ROOT/contracts/audit_20k.yaml"
cp "$ROOT/eval.yaml" "$OUTPUT_ROOT/contracts/eval.yaml"
cp "$ROOT/train_${LINE}.yaml" "$OUTPUT_ROOT/contracts/train_${LINE}.yaml"
cp "$ROOT/sampler_${LINE}.yaml" "$OUTPUT_ROOT/contracts/sampler_${LINE}.yaml"
if [[ -f "$RUN_ROOT/throughput.jsonl" ]]; then
  cp "$RUN_ROOT/throughput.jsonl" "$OUTPUT_ROOT/training/throughput.jsonl"
fi
if [[ -f "$RUN_ROOT/local_metrics/metrics.csv" ]]; then
  cp "$RUN_ROOT/local_metrics/metrics.csv" "$OUTPUT_ROOT/training/metrics.csv"
fi
for EVENT in canonical_gate_events.jsonl gradient_calibration_events.jsonl gradient_safety_audit_events.jsonl; do
  if [[ -f "$RUN_ROOT/$EVENT" ]]; then
    cp "$RUN_ROOT/$EVENT" "$OUTPUT_ROOT/training/$EVENT"
  fi
done

"$TASK1_DRIVER_PYTHON" - "$CHECKPOINT" "$ROOT/train_${LINE}.yaml" "$OUTPUT_ROOT/metadata.json" <<'PY'
import json
from pathlib import Path
import sys

checkpoint = Path(sys.argv[1]).resolve()
config = Path(sys.argv[2]).resolve()
output = Path(sys.argv[3]).resolve()
sha = Path(str(checkpoint) + '.sha256').read_text(encoding='utf-8').split()[0]
payload = {
    'step': 20000,
    'checkpoint_path': str(checkpoint),
    'checkpoint_sha256': sha,
    'config_path': str(config),
}
temporary = output.with_suffix(output.suffix + '.tmp')
temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
temporary.replace(output)
PY

touch "$OUTPUT_ROOT/AUDIT_COMPLETE"
