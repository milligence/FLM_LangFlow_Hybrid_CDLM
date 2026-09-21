#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/autodl_env.sh"

variant="${LOSS_VARIANT:-task1_a}"
case "$variant" in
  task1_a) algo_config=langflow_hybrid_task1_a ;;
  *)
    echo "This release entrypoint supports only LOSS_VARIANT=task1_a" >&2
    exit 2
    ;;
esac

max_steps="${MAX_STEPS:-500}"
run_dir="${RUN_DIR:-$FLM_STORAGE_DIR/runs/owt128_langflow_${variant}_${max_steps}_$(date +%Y%m%d_%H%M%S)}"
resume_from_ckpt="${RESUME_FROM_CKPT:-false}"
resume_ckpt_path="${RESUME_CKPT_PATH:-$run_dir/checkpoints/last.ckpt}"
validate_before_training="${VALIDATE_BEFORE_TRAINING:-true}"
limit_val_batches="${LIMIT_VAL_BATCHES:-1.0}"
gradient_log_every_n_steps="${GRADIENT_LOG_EVERY_N_STEPS:-10}"
diagnostic_interval_steps="${DIAGNOSTIC_INTERVAL_STEPS:-100}"
gradient_route_interval_steps="${GRADIENT_ROUTE_INTERVAL_STEPS:-25}"
validation_interval_steps="${VALIDATION_INTERVAL_STEPS:-$max_steps}"
max_time="${MAX_TIME:-}"
data_config="${DATA_CONFIG:-openwebtext_1k_local}"
global_batch_size="${GLOBAL_BATCH_SIZE:-16}"
micro_batch_size="${MICRO_BATCH_SIZE:-16}"
eval_batch_size="${EVAL_BATCH_SIZE:-16}"
trainer_devices="${TRAINER_DEVICES:-1}"
accumulate_grad_batches="${ACCUMULATE_GRAD_BATCHES:-}"
training_time_sampling="${TRAINING_TIME_SAMPLING:-uniform_tau}"
warmup_steps="${WARMUP_STEPS:-50}"
milestone_steps="${MILESTONE_STEPS:-}"
throughput_every_n_steps="${THROUGHPUT_EVERY_N_STEPS:-0}"
log_every_n_steps="${LOG_EVERY_N_STEPS:-1}"
dataset_tokens="${DATASET_TOKENS:-327680000}"
finetune_path="${FINETUNE_PATH:-}"
token_bias_schedule="${TOKEN_BIAS_SCHEDULE:-}"
token_bias_warmup_steps="${TOKEN_BIAS_WARMUP_STEPS:-}"
learning_rate="${LEARNING_RATE:-3e-4}"
optim_beta1="${OPTIM_BETA1:-0.9}"
optim_beta2="${OPTIM_BETA2:-0.999}"
optim_eps="${OPTIM_EPS:-1e-8}"
optim_weight_decay="${OPTIM_WEIGHT_DECAY:-0}"
ema_decay="${EMA_DECAY:-0.9999}"
task1_contract_smoke="${TASK1_CONTRACT_SMOKE:-false}"
milestone_save_weights_only="${MILESTONE_SAVE_WEIGHTS_ONLY:-true}"
milestone_last_every_n_steps="${MILESTONE_LAST_EVERY_N_STEPS:-0}"
experiment_run_id="${EXPERIMENT_RUN_ID:-}"
experiment_round="${EXPERIMENT_ROUND:-}"
experiment_stage="${EXPERIMENT_STAGE:-}"
experiment_candidate_id="${EXPERIMENT_CANDIDATE_ID:-}"
experiment_initialization_group="${EXPERIMENT_INITIALIZATION_GROUP:-}"
experiment_selected_checkpoint="${EXPERIMENT_SELECTED_CHECKPOINT:-}"
experiment_resume_source_lr="${EXPERIMENT_RESUME_SOURCE_LR:-}"
experiment_resume_source_step="${EXPERIMENT_RESUME_SOURCE_STEP:-}"
resume_constant_learning_rate="${RESUME_CONSTANT_LEARNING_RATE:-}"
resume_target_learning_rate="${RESUME_TARGET_LEARNING_RATE:-}"
resume_lr_transition_steps="${RESUME_LR_TRANSITION_STEPS:-}"
resume_lr_transition_schedule="${RESUME_LR_TRANSITION_SCHEDULE:-linear}"
task1_training_rng_seed="${TASK1_TRAINING_RNG_SEED:-}"
stop_marker_path="${STOP_MARKER_PATH:-}"
planned_stop_marker_path="${PLANNED_STOP_MARKER_PATH:-}"
transfer_queue_dir="${TRANSFER_QUEUE_DIR:-}"
launcher_log_path="${LAUNCHER_LOG_PATH:-}"
task1_arm_id="${TASK1_ARM_ID:-}"
card_started_at_epoch="${CARD_STARTED_AT_EPOCH:-}"
all_graceful_stop_seconds="${ALL_GRACEFUL_STOP_SECONDS:-}"
eta_decision_cutoff_seconds="${ETA_DECISION_CUTOFF_SECONDS:-}"
absolute_limit_seconds="${ABSOLUTE_LIMIT_SECONDS:-}"
task1_train_config_path="${TASK1_TRAIN_CONFIG_RESOLVED:-}"
dry_run_command_path="${DRY_RUN_COMMAND_PATH:-}"

if [[ "$resume_from_ckpt" == "true" && ! -s "$resume_ckpt_path" ]]; then
  echo "Resume checkpoint is missing or empty: $resume_ckpt_path" >&2
  exit 1
fi
if [[ -n "$resume_constant_learning_rate" \
      && "$resume_from_ckpt" != "true" ]]; then
  echo "RESUME_CONSTANT_LEARNING_RATE requires RESUME_FROM_CKPT=true" >&2
  exit 2
fi
if [[ "$trainer_devices" != 1 ]]; then
  echo "Current Task1 entrypoints require TRAINER_DEVICES=1" >&2
  exit 2
fi
case "$training_time_sampling" in
  uniform_tau)
    time_weighting=uniform_tau_unit_weight
    ;;
  high_noise_quota32)
    time_weighting=high_noise_quota32_unit_weight
    ;;
  v1_staged_quota32)
    time_weighting=v1_staged_quota32_unit_weight
    ;;
  v1_q30_frozen_global256)
    time_weighting=v1_q30_frozen_global256_unit_weight
    ;;
  v1_m_tau25_global256)
    time_weighting=v1_m_tau25_global256_unit_weight
    ;;
  *)
    echo "Unsupported TRAINING_TIME_SAMPLING: $training_time_sampling" >&2
    exit 2
    ;;
esac

if [[ -z "$accumulate_grad_batches" ]]; then
  if (( global_batch_size % micro_batch_size != 0 )); then
    echo "GLOBAL_BATCH_SIZE must be divisible by MICRO_BATCH_SIZE" >&2
    exit 2
  fi
  accumulate_grad_batches="$((global_batch_size / micro_batch_size))"
fi
if (( micro_batch_size * accumulate_grad_batches != global_batch_size )); then
  echo "For single-device training, MICRO_BATCH_SIZE * ACCUMULATE_GRAD_BATCHES must equal GLOBAL_BATCH_SIZE" >&2
  exit 2
fi
if [[ "$training_time_sampling" == high_noise_quota32 \
      || "$training_time_sampling" == v1_staged_quota32 \
      || "$training_time_sampling" == v1_q30_frozen_global256 \
      || "$training_time_sampling" == v1_m_tau25_global256 ]]; then
  if [[ "$global_batch_size" != 256 ]]; then
    echo "Task1 quota sampling requires GLOBAL_BATCH_SIZE=256" >&2
    exit 2
  fi
  case "$micro_batch_size/$accumulate_grad_batches" in
    32/8|64/4|128/2) ;;
    *)
      echo "Task1 quota sampling requires H800 micro/accumulation 32/8, 64/4, or 128/2" >&2
      exit 2
      ;;
  esac
fi
validation_interval_batches="$((
  validation_interval_steps * accumulate_grad_batches))"

max_time_args=()
if [[ -n "$max_time" ]]; then
  max_time_args=("+trainer.max_time=$max_time")
fi

milestone_args=()
if [[ -n "$milestone_steps" ]]; then
  milestone_args=(
    "+callbacks.milestone_checkpoints._target_=experiment_callbacks.MilestoneCheckpointCallback"
    "+callbacks.milestone_checkpoints.directory=$run_dir/checkpoints"
    "+callbacks.milestone_checkpoints.queue_path=$run_dir/eval_queue.jsonl"
    "+callbacks.milestone_checkpoints.global_batch_size=$global_batch_size"
    "+callbacks.milestone_checkpoints.sequence_length=128"
    "+callbacks.milestone_checkpoints.milestones=$milestone_steps"
    "+callbacks.milestone_checkpoints.save_weights_only=$milestone_save_weights_only"
    "+callbacks.milestone_checkpoints.last_every_n_steps=$milestone_last_every_n_steps"
    "+callbacks.milestone_checkpoints.last_filename=last.ckpt"
    "+callbacks.milestone_checkpoints.stop_marker_path=$stop_marker_path"
    "+callbacks.milestone_checkpoints.planned_stop_marker_path=$planned_stop_marker_path"
    "+callbacks.milestone_checkpoints.transfer_queue_directory=$transfer_queue_dir"
    "+callbacks.milestone_checkpoints.run_directory=$run_dir"
    "+callbacks.milestone_checkpoints.launcher_log_path=$launcher_log_path"
    "+callbacks.milestone_checkpoints.arm_id=$task1_arm_id"
  )
fi

throughput_args=()
if (( throughput_every_n_steps > 0 )); then
  throughput_args=(
    "+callbacks.throughput._target_=experiment_callbacks.OptimizerStepTimerCallback"
    "+callbacks.throughput.output_path=$run_dir/throughput.jsonl"
    "+callbacks.throughput.global_batch_size=$global_batch_size"
    "+callbacks.throughput.sequence_length=128"
    "+callbacks.throughput.dataset_tokens=$dataset_tokens"
    "+callbacks.throughput.every_n_steps=$throughput_every_n_steps"
  )
fi

finetune_args=()
if [[ -n "$finetune_path" ]]; then
  finetune_args=("training.finetune_path=$finetune_path")
fi

token_bias_schedule_args=()
if [[ -n "$token_bias_schedule" ]]; then
  token_bias_schedule_args=("algo.token_bias_schedule=$token_bias_schedule")
fi

token_bias_warmup_args=()
if [[ -n "$token_bias_warmup_steps" ]]; then
  token_bias_warmup_args=(
    "algo.token_bias_warmup_steps=$token_bias_warmup_steps")
fi

experiment_identity_args=()
if [[ -n "$experiment_run_id" ]]; then
  experiment_identity_args+=("+experiment.run_id=$experiment_run_id")
fi
if [[ -n "$experiment_round" ]]; then
  experiment_identity_args+=("+experiment.round=$experiment_round")
fi
if [[ -n "$experiment_stage" ]]; then
  experiment_identity_args+=("+experiment.stage=$experiment_stage")
fi
if [[ -n "$experiment_candidate_id" ]]; then
  experiment_identity_args+=(
    "+experiment.candidate_id=$experiment_candidate_id")
fi
if [[ -n "$experiment_initialization_group" ]]; then
  experiment_identity_args+=(
    "+experiment.initialization_group=$experiment_initialization_group")
fi
if [[ -n "$experiment_selected_checkpoint" ]]; then
  experiment_identity_args+=(
    "+experiment.selected_checkpoint=$experiment_selected_checkpoint")
fi
if [[ -n "$experiment_resume_source_lr" ]]; then
  experiment_identity_args+=(
    "+experiment.resume.source_learning_rate=$experiment_resume_source_lr")
fi
if [[ -n "$experiment_resume_source_step" ]]; then
  experiment_identity_args+=(
    "+experiment.resume.source_global_step=$experiment_resume_source_step")
fi
if [[ -n "$resume_constant_learning_rate" ]]; then
  experiment_identity_args+=(
    "+experiment.resume.constant_target_learning_rate_override=$resume_constant_learning_rate"
    "+experiment.resume.warmup_restart=false")
fi
if [[ -n "$resume_target_learning_rate" ]]; then
  experiment_identity_args+=(
    "+experiment.resume.transition_target_learning_rate=$resume_target_learning_rate"
    "+experiment.resume.transition_optimizer_steps=$resume_lr_transition_steps"
    "+experiment.resume.transition_schedule=$resume_lr_transition_schedule"
    "+experiment.resume.warmup_restart=false")
fi
if [[ -n "$task1_training_rng_seed" ]]; then
  experiment_identity_args+=(
    "+experiment.training_rng_seed=$task1_training_rng_seed")
fi
if [[ -n "$card_started_at_epoch" ]]; then
  experiment_identity_args+=(
    "+experiment.budget.card_started_at_epoch=$card_started_at_epoch"
    "+experiment.budget.all_graceful_stop_seconds=$all_graceful_stop_seconds"
    "+experiment.budget.eta_decision_cutoff_seconds=$eta_decision_cutoff_seconds"
    "+experiment.budget.absolute_limit_seconds=$absolute_limit_seconds"
    "+experiment.budget.online_transfer_required=false"
    "+experiment.budget.checkpoint_recovery=later_gpu_free_instance")
fi
if [[ -n "$task1_train_config_path" ]]; then
  experiment_identity_args+=(
    "+experiment.train_config_path=$task1_train_config_path")
fi

experiment_contract_args=()
if [[ "$variant" == task1_a ]]; then
  experiment_contract_args=(
    "+experiment.allowed_research_variables=[loader.global_batch_size,optim.lr,optim.beta1,optim.beta2,optim.eps,optim.weight_decay,training.ema,lr_scheduler.num_warmup_steps,algo.token_bias_warmup_steps,algo.training_time_sampling]"
    "+experiment.target_metrics.generative_perplexity_min=100.0"
    "+experiment.target_metrics.generative_perplexity_max=150.0"
    "+experiment.target_metrics.unigram_entropy_direction=increase"
    "+experiment.target_metrics.mauve_direction=increase"
  )
fi

train_command=(python -u -m main \
  mode=train \
  seed=1 \
  data="$data_config" \
  model=small_128 \
  algo="$algo_config" \
  strategy=single_device \
  loader.global_batch_size="$global_batch_size" \
  loader.eval_global_batch_size="$eval_batch_size" \
  loader.batch_size="$micro_batch_size" \
  loader.eval_batch_size="$eval_batch_size" \
  loader.num_workers=4 \
  optim.lr="$learning_rate" \
  optim.beta1="$optim_beta1" \
  optim.beta2="$optim_beta2" \
  optim.eps="$optim_eps" \
  optim.weight_decay="$optim_weight_decay" \
  training.ema="$ema_decay" \
  training.loss_precision=float32 \
  trainer.devices="$trainer_devices" \
  trainer.accumulate_grad_batches="$accumulate_grad_batches" \
  trainer.precision=bf16 \
  trainer.gradient_clip_val=1.0 \
  trainer.max_steps="$max_steps" \
  "${max_time_args[@]}" \
  trainer.num_sanity_val_steps=0 \
  trainer.val_check_interval="$validation_interval_batches" \
  +trainer.check_val_every_n_epoch=null \
  trainer.limit_val_batches="$limit_val_batches" \
  trainer.log_every_n_steps="$log_every_n_steps" \
  lr_scheduler.num_warmup_steps="$warmup_steps" \
  algo.training_time_sampling="$training_time_sampling" \
  algo.time_weighting="$time_weighting" \
  "${finetune_args[@]}" \
  "${token_bias_schedule_args[@]}" \
  "${token_bias_warmup_args[@]}" \
  "${experiment_contract_args[@]}" \
  "${experiment_identity_args[@]}" \
  algo.diagnostic_interval_steps="$diagnostic_interval_steps" \
  algo.gradient_route_interval_steps="$gradient_route_interval_steps" \
  eval.validate_before_training="$validate_before_training" \
  eval.compute_generative_perplexity=false \
  eval.generate_samples=false \
  +callbacks.cuda_peak_memory._target_=experiment_callbacks.CUDAPeakMemoryCallback \
  +callbacks.gradient_norm._target_=experiment_callbacks.GradientNormCallback \
  +callbacks.gradient_norm.every_n_steps="$gradient_log_every_n_steps" \
  "${throughput_args[@]}" \
  "${milestone_args[@]}" \
  +experiment.variant="$variant" \
  +experiment.phase="${EXPERIMENT_PHASE:-unspecified}" \
  +experiment.task1_contract_smoke="$task1_contract_smoke" \
  callbacks.checkpoint_every_n_steps.every_n_train_steps="$max_steps" \
  callbacks.checkpoint_every_n_steps.save_top_k=0 \
  callbacks.checkpoint_every_n_steps.save_last="$([[ -z "$milestone_steps" ]] && echo true || echo false)" \
  +callbacks.checkpoint_every_n_steps.enable_version_counter=false \
  callbacks.checkpoint_monitor.save_top_k=0 \
  callbacks.checkpoint_monitor.save_last=false \
  checkpointing.resume_from_ckpt="$resume_from_ckpt" \
  checkpointing.resume_ckpt_path="$resume_ckpt_path" \
  checkpointing.save_dir="$run_dir" \
  hydra.run.dir="$run_dir")

if [[ -n "$dry_run_command_path" ]]; then
  mkdir -p "$(dirname "$dry_run_command_path")"
  {
    printf 'command='
    printf ' %q' "${train_command[@]}"
    printf '\n'
  } > "$dry_run_command_path"
  exit 0
fi

if [[ "$data_config" == "openwebtext_1k_local" ]]; then
  "$script_dir/prepare_openwebtext_10k.sh"
fi
mkdir -p "$run_dir"

export WANDB_MODE=disabled
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

"${train_command[@]}"

echo "Hybrid $variant training completed: $run_dir"
