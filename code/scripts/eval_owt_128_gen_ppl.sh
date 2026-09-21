#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/autodl_env.sh"

checkpoint_path="${CHECKPOINT_PATH:-$FLM_STORAGE_DIR/runs/owt_128_ce_v2_20step/checkpoints/last.ckpt}"
eval_model_dir="${EVAL_MODEL_DIR:-$FLM_STORAGE_DIR/data/models/gpt2-large}"
sampling_steps="${SAMPLING_STEPS:-4}"
eval_batch_size="${EVAL_BATCH_SIZE:-2}"
num_sample_batches="${NUM_SAMPLE_BATCHES:-1}"
run_dir="${RUN_DIR:-$FLM_STORAGE_DIR/runs/owt_128_gen_ppl_$(date +%Y%m%d_%H%M%S)}"

for required_path in "$checkpoint_path" "$eval_model_dir/config.json" \
  "$eval_model_dir/model.safetensors"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required evaluation input is missing: $required_path" >&2
    exit 1
  fi
done

mkdir -p "$run_dir"
export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -u -m main \
  mode=sample_eval \
  seed=1 \
  data=openwebtext_1k_local \
  model=small_128 \
  algo=flm \
  trainer.devices=1 \
  loader.global_batch_size="$eval_batch_size" \
  loader.eval_global_batch_size="$eval_batch_size" \
  loader.batch_size="$eval_batch_size" \
  loader.eval_batch_size="$eval_batch_size" \
  sampling.steps="$sampling_steps" \
  sampling.num_sample_batches="$num_sample_batches" \
  sampling.use_float64=false \
  eval.checkpoint_path="$checkpoint_path" \
  eval.gen_ppl_eval_model_name_or_path="$eval_model_dir" \
  eval.perplexity_batch_size="$eval_batch_size" \
  eval.generated_samples_path="$run_dir/samples.json" \
  eval.disable_ema=false \
  hydra.run.dir="$run_dir"

echo "OWT-128 generation-PPL smoke completed: $run_dir/samples.json"
