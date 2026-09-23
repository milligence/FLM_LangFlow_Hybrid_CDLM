#!/usr/bin/env bash
set -euo pipefail

: "${CONTRACT_ROOT:?Set CONTRACT_ROOT to task1_tvm_50k_final contract directory}"
: "${TASK1_BINDINGS:?Set TASK1_BINDINGS to runtime_bindings.json}"
: "${TASK1_DRIVER_PYTHON:?Set TASK1_DRIVER_PYTHON to the declared Python executable}"
: "${RUN_ROOT:?Set RUN_ROOT to the F training run directory}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the F 2k diagnostic root}"

mkdir -p "$OUTPUT_ROOT"
progress="$OUTPUT_ROOT/supervisor.log"
steps=(32000 34000 36000 38000 40000 42000 44000 46000 48000 50000)

stop_training_group() {
  local process_group="$1"
  kill -TERM -- "-$process_group" 2>/dev/null || true
  for _ in {1..20}; do
    if ! kill -0 "$process_group" 2>/dev/null; then
      return
    fi
    sleep 1
  done
  kill -KILL -- "-$process_group" 2>/dev/null || true
}

for step in "${steps[@]}"; do
  step_name="step_$(printf '%06d' "$step")"
  output="$OUTPUT_ROOT/$step_name"
  if [[ -f "$output/AUDIT_COMPLETE" ]]; then
    printf '%s already complete\n' "$step_name" >>"$progress"
    continue
  fi

  last_checkpoint="$RUN_ROOT/checkpoints/last.ckpt"
  baseline_mtime="$(stat -c %Y "$last_checkpoint")"
  launch_log="$OUTPUT_ROOT/train_to_${step}.log"
  setsid env \
    TASK1_BINDINGS="$TASK1_BINDINGS" \
    TASK1_DRIVER_PYTHON="$TASK1_DRIVER_PYTHON" \
    F_GPU=0 RESUME="$last_checkpoint" \
    TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash "$CONTRACT_ROOT/launch_f.sh" >"$launch_log" 2>&1 < /dev/null &
  train_pid=$!
  printf 'launched pid=%s target=%s baseline_mtime=%s\n' \
    "$train_pid" "$step" "$baseline_mtime" >>"$progress"

  while true; do
    latest_step="$(tail -n 1 "$RUN_ROOT/throughput.jsonl" \
      | sed -n 's/.*"optimizer_step": \([0-9]*\).*/\1/p')"
    checkpoint_mtime="$(stat -c %Y "$last_checkpoint" 2>/dev/null || echo 0)"
    if [[ -n "$latest_step" && "$latest_step" -ge "$step" \
          && "$checkpoint_mtime" -gt "$baseline_mtime" ]]; then
      stop_training_group "$train_pid"
      break
    fi
    if ! kill -0 "$train_pid" 2>/dev/null; then
      printf 'training exited before target=%s latest_step=%s\n' \
        "$step" "${latest_step:-missing}" >>"$progress"
      touch "$OUTPUT_ROOT/SUPERVISOR_FAILED"
      exit 4
    fi
    sleep 10
  done

  milestone="$RUN_ROOT/checkpoints/$step_name.ckpt"
  if [[ -f "$milestone" ]]; then
    checkpoint="$milestone"
  else
    checkpoint="$last_checkpoint"
  fi
  printf 'audit target=%s checkpoint=%s\n' "$step" "$checkpoint" >>"$progress"
  mkdir -p "$output"
  env TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled \
    TASK1_BINDINGS="$TASK1_BINDINGS" \
    TASK1_DRIVER_PYTHON="$TASK1_DRIVER_PYTHON" \
    bash "$CONTRACT_ROOT/run_f2k_diagnostic.sh" \
      "$step" "$checkpoint" "$RUN_ROOT" "$output" \
      >"$output/audit.log" 2>&1
  if [[ ! -f "$output/AUDIT_COMPLETE" ]]; then
    printf 'audit failed target=%s\n' "$step" >>"$progress"
    touch "$OUTPUT_ROOT/SUPERVISOR_FAILED"
    exit 5
  fi
  printf 'audit complete target=%s\n' "$step" >>"$progress"
done

touch "$OUTPUT_ROOT/SERIES_COMPLETE"
