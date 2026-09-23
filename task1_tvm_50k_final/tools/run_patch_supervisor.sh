#!/usr/bin/env bash
set -euo pipefail

: "${CONTRACT_ROOT:?Set CONTRACT_ROOT to task1_tvm_50k_final contract directory}"
: "${TASK1_BINDINGS:?Set TASK1_BINDINGS to runtime_bindings.json}"
: "${TASK1_DRIVER_PYTHON:?Set TASK1_DRIVER_PYTHON to the declared Python executable}"
: "${RUN_ROOT:?Set RUN_ROOT to the F training run directory}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the F patch review root}"
: "${INITIAL_CHECKPOINT:?Set INITIAL_CHECKPOINT to the exact step-30000 full-state checkpoint}"

mkdir -p "$OUTPUT_ROOT"
progress="$OUTPUT_ROOT/supervisor.log"
steps=(32000 34000 36000 38000 40000 42000 44000 46000 48000 50000)
profile_state="$OUTPUT_ROOT/profile_state.json"
decision_log="$OUTPUT_ROOT/decision_events.jsonl"
profile=pre32
if [[ -f "$profile_state" ]]; then
  profile="$($TASK1_DRIVER_PYTHON -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["profile"])' \
    "$profile_state")"
fi

stop_training_group() {
  local process_group="$1"
  kill -INT -- "-$process_group" 2>/dev/null || true
  for _ in {1..30}; do
    if ! kill -0 "$process_group" 2>/dev/null; then
      return
    fi
    sleep 1
  done
  kill -TERM -- "-$process_group" 2>/dev/null || true
  for _ in {1..20}; do
    if ! kill -0 "$process_group" 2>/dev/null; then
      return
    fi
    sleep 1
  done
  kill -KILL -- "-$process_group" 2>/dev/null || true
}

verify_checkpoint() {
  local checkpoint="$1"
  [[ -f "$checkpoint" && -f "$checkpoint.sha256" ]] || return 1
  (cd "$(dirname "$checkpoint")" && sha256sum -c "$(basename "$checkpoint").sha256" >/dev/null)
}

resume_checkpoint="$INITIAL_CHECKPOINT"
resume_step=30000
verify_checkpoint "$resume_checkpoint"

for step in "${steps[@]}"; do
  step_name="step_$(printf '%06d' "$step")"
  output="$OUTPUT_ROOT/$step_name"
  milestone="$RUN_ROOT/checkpoints/$step_name.ckpt"

  if [[ -f "$output/AUDIT_COMPLETE" ]] && verify_checkpoint "$milestone"; then
    printf '%s already complete\n' "$step_name" >>"$progress"
    if [[ -f "$output/decision_output.json" ]]; then
      profile="$($TASK1_DRIVER_PYTHON -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["new_profile"])' \
        "$output/decision_output.json")"
    fi
    resume_checkpoint="$milestone"
    resume_step="$step"
    continue
  fi

  milestone_mtime="$(stat -c %Y "$milestone" 2>/dev/null || echo 0)"
  sha_mtime="$(stat -c %Y "$milestone.sha256" 2>/dev/null || echo 0)"
  launch_log="$OUTPUT_ROOT/train_${resume_step}_to_${step}.log"
  setsid env \
    TASK1_BINDINGS="$TASK1_BINDINGS" \
    TASK1_DRIVER_PYTHON="$TASK1_DRIVER_PYTHON" \
    TASK1_F_SAMPLER_PROFILE="$profile" \
    F_GPU=0 RESUME="$resume_checkpoint" \
    TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash "$CONTRACT_ROOT/launch_f.sh" >"$launch_log" 2>&1 < /dev/null &
  train_pid=$!
  printf 'launched pid=%s resume_step=%s target=%s\n' \
    "$train_pid" "$resume_step" "$step" >>"$progress"

  while true; do
    latest_step="$(tail -n 1 "$RUN_ROOT/throughput.jsonl" 2>/dev/null \
      | sed -n 's/.*"optimizer_step": \([0-9]*\).*/\1/p')"
    new_milestone_mtime="$(stat -c %Y "$milestone" 2>/dev/null || echo 0)"
    new_sha_mtime="$(stat -c %Y "$milestone.sha256" 2>/dev/null || echo 0)"
    if [[ -n "$latest_step" && "$latest_step" -ge "$step" \
          && "$new_milestone_mtime" -gt "$milestone_mtime" \
          && "$new_sha_mtime" -gt "$sha_mtime" ]] \
          && verify_checkpoint "$milestone"; then
      stop_training_group "$train_pid"
      wait "$train_pid" 2>/dev/null || true
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

  printf 'audit target=%s checkpoint=%s\n' "$step" "$milestone" >>"$progress"
  mkdir -p "$output"
  env TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 WANDB_MODE=disabled \
    TASK1_BINDINGS="$TASK1_BINDINGS" \
    TASK1_DRIVER_PYTHON="$TASK1_DRIVER_PYTHON" \
    TASK1_F_SAMPLER_PROFILE="$profile" \
    RUN_ROOT="$RUN_ROOT" \
    bash "$CONTRACT_ROOT/run_patch_checkpoint_hook.sh" \
      "$step" "$milestone" "$output" >"$output/hook.log" 2>&1
  if [[ ! -f "$output/AUDIT_COMPLETE" ]]; then
    printf 'audit failed target=%s\n' "$step" >>"$progress"
    touch "$OUTPUT_ROOT/SUPERVISOR_FAILED"
    exit 5
  fi

  if [[ "$step" == 32000 || "$step" == 36000 || "$step" == 40000 ]]; then
    "$TASK1_DRIVER_PYTHON" "$CONTRACT_ROOT/tools/build_patch_decision_input.py" \
      --output-root "$OUTPUT_ROOT" --step "$step" \
      --current-profile "$profile" --lineage-verified \
      --output "$output/decision_input.json"
    "$TASK1_DRIVER_PYTHON" "$CONTRACT_ROOT/tools/decision_policy.py" \
      "$output/decision_input.json" --output "$output/decision_output.json"
    status="$($TASK1_DRIVER_PYTHON -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
      "$output/decision_output.json")"
    if [[ "$status" == blocked ]]; then
      printf 'decision blocked target=%s\n' "$step" >>"$progress"
      touch "$OUTPUT_ROOT/SUPERVISOR_FAILED"
      exit 6
    fi
    new_profile="$($TASK1_DRIVER_PYTHON -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["new_profile"])' \
      "$output/decision_output.json")"
    event_id="decision-${step}-${profile}-to-${new_profile}"
    if ! grep -q "\"event_id\": \"$event_id\"" "$decision_log" 2>/dev/null; then
      "$TASK1_DRIVER_PYTHON" -c \
        'import json,sys; d=json.load(open(sys.argv[1])); d["event_id"]=sys.argv[2]; open(sys.argv[3],"a").write(json.dumps(d,sort_keys=True)+"\n")' \
        "$output/decision_output.json" "$event_id" "$decision_log"
    fi
    profile="$new_profile"
    "$TASK1_DRIVER_PYTHON" -c \
      'import json,os,sys; p=sys.argv[1]+".tmp"; open(p,"w").write(json.dumps({"profile":sys.argv[2],"completed_step":int(sys.argv[3])},sort_keys=True)+"\n"); os.replace(p,sys.argv[1])' \
      "$profile_state" "$profile" "$step"
    printf 'decision target=%s profile=%s status=%s\n' \
      "$step" "$profile" "$status" >>"$progress"
  fi

  # Keep pinned 30/32/36/40/50 plus the latest two complete full states.
  two_back=$((step - 4000))
  case "$two_back" in
    30000|32000|36000|40000|50000) ;;
    34000|38000|42000|44000|46000)
      old="$RUN_ROOT/checkpoints/step_$(printf '%06d' "$two_back").ckpt"
      if [[ -f "$old" && -f "$old.sha256" ]]; then
        rm -f -- "$old" "$old.sha256"
        printf 'removed evaluated nonpinned checkpoint step=%s\n' \
          "$two_back" >>"$progress"
      fi
      ;;
  esac
  if [[ "$step" == 32000 ]]; then
    old="$RUN_ROOT/checkpoints/step_030500.ckpt"
    if [[ -f "$old" && -f "$old.sha256" ]]; then
      rm -f -- "$old" "$old.sha256"
      printf 'removed evaluated nonpinned checkpoint step=30500\n' >>"$progress"
    fi
  fi

  printf 'audit complete target=%s\n' "$step" >>"$progress"
  resume_checkpoint="$milestone"
  resume_step="$step"
done

touch "$OUTPUT_ROOT/SERIES_COMPLETE"
