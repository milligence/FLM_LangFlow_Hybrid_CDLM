#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for required in ARM CHECKPOINT_PATH EXPECTED_STEP RUN_DIR GRID \
  NFE NUM_SAMPLES EVAL_BATCH_SIZE EVAL_SEED PYTHON_BIN \
  PERSISTENT_ROOT DATA_DIR TOKENIZER_DIR EVAL_MODEL_DIR CACHE_DIR; do
  [[ -n "${!required:-}" ]] || {
    echo "Missing required field: $required" >&2
    exit 2
  }
done
[[ "$ARM" == A || "$ARM" == B ]] || { echo "ARM must be A or B" >&2; exit 2; }
[[ -s "$CHECKPOINT_PATH" ]] || { echo "Missing checkpoint" >&2; exit 2; }
[[ ! -e "$RUN_DIR" ]] || { echo "RUN_DIR exists: $RUN_DIR" >&2; exit 2; }
(( NUM_SAMPLES % EVAL_BATCH_SIZE == 0 )) || {
  echo "NUM_SAMPLES must divide by EVAL_BATCH_SIZE" >&2
  exit 2
}

mkdir -p "$RUN_DIR"
export FLM_STORAGE_DIR="$PERSISTENT_ROOT"
export FLM_PACKED_DATA_DIR="$DATA_DIR"
export FLM_TOKENIZER_PATH="$TOKENIZER_DIR"
export HF_HOME="$CACHE_DIR"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

LOSS_VARIANT="task1_posterior_tvm_${ARM,,}" \
MODEL_CONFIG=small_128_posterior_tvm \
DATA_CONFIG=openwebtext_327m_packed \
CHECKPOINT_PATH="$CHECKPOINT_PATH" \
NUM_SAMPLES="$NUM_SAMPLES" \
SAMPLING_STEPS="$NFE" \
SAMPLING_SOLVER=euler \
SAMPLING_TEMPERATURE=1.0 \
EVAL_BATCH_SIZE="$EVAL_BATCH_SIZE" \
EVAL_SEED="$EVAL_SEED" \
DISABLE_EMA="${DISABLE_EMA:-false}" \
GEN_PPL_COMPARABLE="${GEN_PPL_COMPARABLE:-false}" \
GEN_PPL_PROTOCOL_ID="${GEN_PPL_PROTOCOL_ID:-owt128-gpt2large-posterior-tvm-v1}" \
COMPUTE_GEN_PPL="${COMPUTE_GEN_PPL:-true}" \
EVALUATION_PHASE="${EVALUATION_PHASE:-posterior_tvm_eval}" \
TASK1_INITIAL_NOISE_SEED="$EVAL_SEED" \
TASK1_INITIAL_NOISE_SCHEDULE=base_seed_plus_sample_index \
TVM_INFERENCE_GRID_PHYSICAL="$GRID" \
POSTERIOR_TVM_INFERENCE_MODE="${POSTERIOR_TVM_INFERENCE_MODE:-finite_map}" \
EVAL_MODEL_DIR="$EVAL_MODEL_DIR" \
RUN_DIR="$RUN_DIR" \
PYTHON_BIN="$PYTHON_BIN" \
  "$script_dir/eval_owt_128_langflow_hybrid.sh"

"$PYTHON_BIN" - "$RUN_DIR/samples.json" "$EXPECTED_STEP" "$NFE" <<'PY'
import json
import math
import sys
from pathlib import Path

path, expected_step, expected_nfe = sys.argv[1:]
payload = json.loads(Path(path).read_text(encoding='utf-8'))
checks = {
    'step': int(payload['checkpoint_global_step']) == int(expected_step),
    'nfe': int(payload['nfe']) == int(expected_nfe),
    'sample_count': len(payload['generated_seqs']) == int(payload['num_samples']),
    'finite_ppl': math.isfinite(float(payload['generative_ppl'])),
}
if not all(checks.values()):
    raise RuntimeError(f'Posterior-TVM evaluation validation failed: {checks}')
quality = payload['sample_quality']
summary = {
    'status': 'completed',
    'checkpoint_global_step': int(payload['checkpoint_global_step']),
    'weights': payload['weights'],
    'num_samples': int(payload['num_samples']),
    'nfe': int(payload['nfe']),
    'generative_ppl': float(payload['generative_ppl']),
    'entropy': float(payload['entropy']),
    **{key: float(value) for key, value in quality.items()},
    'fixed_texts_0_7': payload['generated_seqs'][:8],
}
Path(path).with_name('summary.json').write_text(
    json.dumps(summary, indent=2) + '\n', encoding='utf-8')
PY
