#!/usr/bin/env bash

# Shared offline runtime defaults. Callers may override every path.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FLM_PROJECT_DIR="${FLM_PROJECT_DIR:-$(cd "$script_dir/.." && pwd)}"
: "${FLM_STORAGE_DIR:?Set FLM_STORAGE_DIR to a writable data/run directory}"
export FLM_STORAGE_DIR
export VIRTUAL_ENV="${VIRTUAL_ENV:-$FLM_STORAGE_DIR/venv311}"
export PATH="$VIRTUAL_ENV/bin:$PATH"

export PIP_CACHE_DIR="$FLM_STORAGE_DIR/pip-cache"
export HF_HOME="$FLM_STORAGE_DIR/data/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$FLM_STORAGE_DIR/data/torch"
export WANDB_DIR="$FLM_STORAGE_DIR/runs/wandb"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

mkdir -p \
  "$HF_DATASETS_CACHE" \
  "$HUGGINGFACE_HUB_CACHE" \
  "$TORCH_HOME" \
  "$WANDB_DIR" \
  "$FLM_STORAGE_DIR/checkpoints" \
  "$FLM_STORAGE_DIR/runs"

cd "$FLM_PROJECT_DIR"
