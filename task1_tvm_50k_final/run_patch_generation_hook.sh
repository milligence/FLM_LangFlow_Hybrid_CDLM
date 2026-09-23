#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${TASK1_BINDINGS:?Set TASK1_BINDINGS to runtime_bindings.json}"
: "${TASK1_DRIVER_PYTHON:?Set TASK1_DRIVER_PYTHON to the declared Python executable}"

STEP="${1:?Usage: run_patch_generation_hook.sh STEP CHECKPOINT OUTPUT_ROOT}"
CHECKPOINT="${2:?Missing checkpoint}"
OUTPUT_ROOT="${3:?Missing output root}"
[[ -f "$CHECKPOINT" ]] || { echo "Missing checkpoint: $CHECKPOINT" >&2; exit 3; }

SAMPLES=128
case "$STEP" in 30000|40000|50000) SAMPLES=1024;; esac
SEED=424242
mkdir -p "$OUTPUT_ROOT/generation"

run_job() {
  local NAME="$1" ROLE="$2" MODE="$3" NFE="$4" GRID="$5" COUNT="$6"
  local CANONICAL_K="${7:-}"
  local JOB="$OUTPUT_ROOT/generation/$NAME"
  mkdir -p "$JOB"
  [[ -f "$JOB/samples.json" ]] && return
  local EXTRA=()
  [[ -n "$CANONICAL_K" ]] && EXTRA=(--canonical-k "$CANONICAL_K")
  "$TASK1_DRIVER_PYTHON" "$ROOT/tools/run_20k_generation.py" \
    --bindings "$TASK1_BINDINGS" --contract-dir "$ROOT" --line f \
    --checkpoint "$CHECKPOINT" --model-type "$ROLE" \
    --mode "$MODE" --nfe "$NFE" --samples "$COUNT" --batch-size 32 \
    --seed "$SEED" --grid "$GRID" --output "$JOB/samples.json" "${EXTRA[@]}" \
    >"$JOB/evaluation.log" 2>&1
}

run_job legacy_local_512 eval_ema legacy_rolling_T1 512 '[0.0,0.95]' "$SAMPLES"
run_job finite_4_A eval_ema finite 4 \
  '[0.0,0.3392770637,0.5814685447,0.7246687714,0.95]' "$SAMPLES"
run_job finite_4_B eval_ema finite 4 \
  '[0.0,0.2375,0.475,0.7125,0.95]' "$SAMPLES"
run_job finite_1 eval_ema finite 1 '[0.0,0.95]' "$SAMPLES"
run_job finite_2 eval_ema finite 2 '[0.0,0.5814685447,0.95]' "$SAMPLES"
run_job online_finite_4_B_64 online finite 4 \
  '[0.0,0.2375,0.475,0.7125,0.95]' 64
run_job eval_finite_4_B_64 eval_ema finite 4 \
  '[0.0,0.2375,0.475,0.7125,0.95]' 64

case "$STEP" in
  32000|36000|40000|50000)
    run_job canonical_K2_T095_64 eval_ema canonical_fixed_budget_T095 1024 \
      '[0.0,0.95]' 64 2
    ;;
esac
if [[ "$STEP" == 32000 ]]; then
  run_job rolling_T095_512_64 eval_ema matched_rolling_T095 512 \
    '[0.0,0.95]' 64
fi
if [[ "$STEP" == 34000 || "$STEP" == 38000 ]]; then
  REVIEW_ROOT="$(dirname "$OUTPUT_ROOT")"
  NEEDS_CANONICAL="$($TASK1_DRIVER_PYTHON \
    "$ROOT/tools/build_patch_decision_input.py" \
    --output-root "$REVIEW_ROOT" --step "$STEP" \
    --current-profile "${TASK1_F_SAMPLER_PROFILE:-uniform}" \
    --risk-canonical-needed)"
  if [[ "$NEEDS_CANONICAL" == true ]]; then
    run_job canonical_K2_T095_64 eval_ema canonical_fixed_budget_T095 1024 \
      '[0.0,0.95]' 64 2
  fi
fi

touch "$OUTPUT_ROOT/GENERATION_COMPLETE"
