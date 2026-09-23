#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${TASK1_BINDINGS:?Set TASK1_BINDINGS to the resolved runtime_bindings.json}"
: "${TASK1_DRIVER_PYTHON:=python3}"
LINE="${1:?Usage: run_precheck.sh f|p}"
exec "$TASK1_DRIVER_PYTHON" "$ROOT/tools/run_entrypoint.py" precheck --line "$LINE" \
  --bindings "$TASK1_BINDINGS" --gpu "${PRECHECK_GPU:-0}"
