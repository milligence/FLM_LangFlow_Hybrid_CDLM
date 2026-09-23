#!/usr/bin/env bash
set -euo pipefail

: "${RUN_ROOT:?Set RUN_ROOT to the F training run directory}"
: "${TARGET_STEP:=31000}"

throughput="$RUN_ROOT/throughput.jsonl"
checkpoint="$RUN_ROOT/checkpoints/last.ckpt"
baseline_mtime="$(stat -c %Y "$checkpoint")"

while true; do
  step="$(tail -n 1 "$throughput" | sed -n 's/.*"optimizer_step": \([0-9]*\).*/\1/p')"
  checkpoint_mtime="$(stat -c %Y "$checkpoint" 2>/dev/null || echo 0)"
  if [[ -n "$step" && "$step" -ge "$TARGET_STEP" && "$checkpoint_mtime" -gt "$baseline_mtime" ]]; then
    launcher_pid="$(pgrep -fo '[r]un_entrypoint.py train --line f' || true)"
    if [[ -n "$launcher_pid" ]]; then
      process_group="$(ps -o pgid= -p "$launcher_pid" | tr -d ' ')"
      kill -TERM -- "-$process_group" 2>/dev/null || true
      sleep 10
      if pgrep -f '[r]un_entrypoint.py train --line f|[t]ask1_tvm_50k_final_adapter train.*--line f|[p]ython main.py.*task1_tvm_50k_final_f' >/dev/null; then
        kill -KILL -- "-$process_group" 2>/dev/null || true
      fi
    fi
    printf 'STOPPED_AT_STEP=%s CHECKPOINT_MTIME=%s\n' "$step" "$checkpoint_mtime"
    exit 0
  fi
  sleep 5
done
