#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/autodl_env.sh"

variant="${LOSS_VARIANT:-task1_a}"
case "$variant" in
  task1_a) algo_config=langflow_hybrid_task1_a ;;
  *) echo "This release entrypoint supports only LOSS_VARIANT=task1_a" >&2; exit 2 ;;
esac

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the paired-run checkpoint}"
eval_model_dir="${EVAL_MODEL_DIR:-$FLM_STORAGE_DIR/data/models/gpt2-large}"
run_dir="${RUN_DIR:-$FLM_STORAGE_DIR/runs/owt128_langflow_${variant}_eval_$(date +%Y%m%d_%H%M%S)}"
eval_batch_size="${EVAL_BATCH_SIZE:-2}"
num_samples="${NUM_SAMPLES:-64}"
disable_ema="${DISABLE_EMA:-false}"
evaluation_phase="${EVALUATION_PHASE:-generation_eval_${num_samples}}"
gen_ppl_comparable="${GEN_PPL_COMPARABLE:-false}"
data_config="${DATA_CONFIG:-openwebtext_1k_local}"
sampling_steps="${SAMPLING_STEPS:-128}"
sampling_solver="${SAMPLING_SOLVER:-euler}"
sampling_temperature="${SAMPLING_TEMPERATURE:-1.0}"
compute_gen_ppl="${COMPUTE_GEN_PPL:-true}"
task1_contract_smoke="${TASK1_CONTRACT_SMOKE:-false}"
task1_time_grid="${TASK1_TIME_GRID:-uniform_tau}"
task1_first_interval_substeps="${TASK1_FIRST_INTERVAL_SUBSTEPS:-1}"
task1_tau_box_query_counts="${TASK1_TAU_BOX_QUERY_COUNTS:-[]}"
task1_tail_bin_removal_counts="${TASK1_TAIL_BIN_REMOVAL_COUNTS:-[]}"
task1_diagnostic_steps="${TASK1_DIAGNOSTIC_STEPS:-[]}"
task1_initial_noise_seed="${TASK1_INITIAL_NOISE_SEED:-null}"
task1_initial_noise_schedule="${TASK1_INITIAL_NOISE_SCHEDULE:-base_seed_plus_batch_index}"
task1_trajectory_diagnostics="${TASK1_TRAJECTORY_DIAGNOSTICS:-false}"
task1_trajectory_checkpoint_label="${TASK1_TRAJECTORY_CHECKPOINT_LABEL:-}"
task1_trajectory_top_k="${TASK1_TRAJECTORY_TOP_K:-32}"
task1_trajectory_fp32_sample_count="${TASK1_TRAJECTORY_FP32_SAMPLE_COUNT:-8}"
task1_trajectory_fp32_node_count="${TASK1_TRAJECTORY_FP32_NODE_COUNT:-17}"
gen_ppl_protocol_id="${GEN_PPL_PROTOCOL_ID:-owt128-gpt2large-genppl-v1}"

required_paths=("$CHECKPOINT_PATH")
if [[ "$compute_gen_ppl" == true ]]; then
  required_paths+=("$eval_model_dir/config.json")
fi
for required_path in "${required_paths[@]}"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required evaluation input is missing: $required_path" >&2
    exit 1
  fi
done
if [[ "$compute_gen_ppl" == true ]]; then
  if ! find "$eval_model_dir" -maxdepth 1 -type f \
      \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) \
      -print -quit | grep -q .; then
    echo "No GPT-2 Large weight file found in $eval_model_dir" >&2
    exit 1
  fi
fi
if (( num_samples % eval_batch_size != 0 )); then
  echo "NUM_SAMPLES must be divisible by EVAL_BATCH_SIZE" >&2
  exit 2
fi
if [[ "$gen_ppl_comparable" == true ]]; then
  if (( num_samples != 1024 )) || [[ "$disable_ema" != false ]]; then
    echo "Comparable Gen. PPL requires 1024 EMA samples" >&2
    exit 2
  fi
fi

mkdir -p "$run_dir"

export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -u -m main \
  mode=sample_eval \
  seed=42 \
  data="$data_config" \
  model=small_128 \
  algo="$algo_config" \
  strategy=single_device \
  trainer.devices=1 \
  loader.global_batch_size="$eval_batch_size" \
  loader.eval_global_batch_size="$eval_batch_size" \
  loader.batch_size="$eval_batch_size" \
  loader.eval_batch_size="$eval_batch_size" \
  sampling.steps="$sampling_steps" \
  sampling.solver="$sampling_solver" \
  sampling.temperature="$sampling_temperature" \
  sampling.num_sample_batches="$(( num_samples / eval_batch_size ))" \
  sampling.use_float64=false \
  sampling.task1_time_grid="$task1_time_grid" \
  sampling.task1_first_interval_substeps="$task1_first_interval_substeps" \
  sampling.task1_tau_box_query_counts="$task1_tau_box_query_counts" \
  sampling.task1_tail_bin_removal_counts="$task1_tail_bin_removal_counts" \
  sampling.task1_diagnostic_steps="$task1_diagnostic_steps" \
  sampling.task1_initial_noise_seed="$task1_initial_noise_seed" \
  sampling.task1_initial_noise_schedule="$task1_initial_noise_schedule" \
  +sampling.task1_trajectory_diagnostics="$task1_trajectory_diagnostics" \
  +sampling.task1_trajectory_checkpoint_label="$task1_trajectory_checkpoint_label" \
  +sampling.task1_trajectory_top_k="$task1_trajectory_top_k" \
  +sampling.task1_trajectory_fp32_sample_count="$task1_trajectory_fp32_sample_count" \
  +sampling.task1_trajectory_fp32_node_count="$task1_trajectory_fp32_node_count" \
  eval.checkpoint_path="$CHECKPOINT_PATH" \
  eval.gen_ppl_eval_model_name_or_path="$eval_model_dir" \
  eval.gen_ppl_protocol_id="$gen_ppl_protocol_id" \
  eval.gen_ppl_comparable="$gen_ppl_comparable" \
  eval.compute_generative_perplexity="$compute_gen_ppl" \
  eval.perplexity_batch_size="$eval_batch_size" \
  eval.generated_samples_path="$run_dir/samples.json" \
  eval.disable_ema="$disable_ema" \
  +experiment.variant="$variant" \
  +experiment.phase="$evaluation_phase" \
  +experiment.task1_contract_smoke="$task1_contract_smoke" \
  hydra.run.dir="$run_dir"

echo "$num_samples-sample evaluation completed: $run_dir/samples.json"
