#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/autodl_env.sh"

: "${TEACHER_PATH:?Set the FrozenMSE-50k full-state teacher checkpoint}"
: "${TVM_STUDENT_INIT_PATH:?Set the TVM-CE 10k checkpoint}"
: "${TVM_SC_PAIR_BANK_PATH:?Set the new pair-bank output path}"

"${PYTHON_BIN:-python}" -u -m main \
  mode=task1_sc_pair_bank \
  seed="${SEED:-42}" \
  data=openwebtext_327m_packed \
  model=small_128 \
  algo=task1_tvm_sc_repair \
  strategy=single_device \
  trainer.devices=1 \
  loader.global_batch_size="${PAIR_BANK_BATCH_SIZE:-4}" \
  loader.eval_global_batch_size="${PAIR_BANK_BATCH_SIZE:-4}" \
  loader.batch_size="${PAIR_BANK_BATCH_SIZE:-4}" \
  loader.eval_batch_size="${PAIR_BANK_BATCH_SIZE:-4}" \
  algo.teacher_path="$TEACHER_PATH" \
  algo.student_init_path="$TVM_STUDENT_INIT_PATH" \
  algo.tvm_sc_pair_bank_path="$TVM_SC_PAIR_BANK_PATH" \
  algo.tvm_sc_gate_d50="${TVM_SC_GATE_D50:-0.0001}" \
  checkpointing.save_dir="$(dirname "$TVM_SC_PAIR_BANK_PATH")" \
  hydra.run.dir="$(dirname "$TVM_SC_PAIR_BANK_PATH")"
