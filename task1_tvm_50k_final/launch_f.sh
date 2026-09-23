#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${TASK1_BINDINGS:?Set TASK1_BINDINGS to the resolved runtime_bindings.json}"
: "${TASK1_DRIVER_PYTHON:=python3}"
exec "$TASK1_DRIVER_PYTHON" "$ROOT/tools/run_entrypoint.py" train --line f \
  --bindings "$TASK1_BINDINGS" --gpu "${F_GPU:-0}" --resume "${RESUME:-auto}"
