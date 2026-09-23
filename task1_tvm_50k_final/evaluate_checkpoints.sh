#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${TASK1_BINDINGS:?Set TASK1_BINDINGS to the resolved runtime_bindings.json}"
: "${TASK1_DRIVER_PYTHON:=python3}"
LINE="${1:?Usage: evaluate_checkpoints.sh f|p [comma-separated checkpoint subset]}"
if [[ "$LINE" != f && "$LINE" != p ]]; then echo "Line must be f or p" >&2; exit 2; fi
ARGS=(evaluate --line "$LINE" --bindings "$TASK1_BINDINGS" --gpu "${EVAL_GPU:-0}")
if [[ -n "${2:-}" ]]; then ARGS+=(--steps "$2"); fi
exec "$TASK1_DRIVER_PYTHON" "$ROOT/tools/run_entrypoint.py" "${ARGS[@]}"
