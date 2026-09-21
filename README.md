# 小步 TVM 实验

本分支只发布小步阶段相对 FrozenMSE-50k 的新增代码与必要增量，不包含 FLM 上游源码，
也不复制第一阶段代码。先应用 `codex/frozen-mse-50k-release`，再应用本分支。

## 发布内容

| 目录 | 内容 |
|---|---|
| `code/` | TVM-CE 10k、SC-repair no-comp、J0、Posterior-TVM 和对应配置/测试 |
| `patches/frozen-release-integration.patch` | 小步代码对第一阶段文件的修改 |
| `scripts/apply_to_flm.sh` | 将本阶段增量应用到已安装 FrozenMSE 的 FLM 工作区 |
| `docs/SMALL_STEP_EXPLORATION.md` | 全部探索及明确的不提交清单 |
| `docs/SMALL_STEP_RESULTS.md` | 复现证据、当前结果及每条代码的两条 sample |
| `manifest/` | 新增文件与被修改文件清单 |

composition、J1、endpoint500、native posterior、SCD/dsCD、CE/Gumbel 和 no-SC teacher
代码均未提交；它们只在探索记录中说明。

## 1. 应用代码

先按第一分支 README 将 FrozenMSE-50k 应用到固定的 FLM 上游工作区，然后执行：

```bash
git clone --branch codex/small-step-tvm-experiments --single-branch \
  https://github.com/milligence/FLM_LangFlow_Hybrid_CDLM.git small-step-release

bash small-step-release/scripts/apply_to_flm.sh /path/to/flm
cd /path/to/flm
```

共同运行时输入：

```bash
export FLM_STORAGE_DIR=/path/to/storage
export FLM_TOKENIZER_PATH=/path/to/gpt2-tokenizer
export FLM_PACKED_DATA_DIR=/path/to/openwebtext-128-packed
```

## 2. TVM-CE 10k

```bash
LOSS_VARIANT=task1_tvm_ce \
TASK1_TRAIN_CONFIG=configs/task1_tvm_ce_10k.env \
TEACHER_PATH=/path/to/frozen-mse-step-050000.ckpt \
RUN_DIR=/path/to/run/tvm-ce-10k \
  bash scripts/train_task1_small_step.sh
```

固定 teacher 为 FrozenMSE@50k EMA。10k 的选定 4-NFE physical-`t` 网格为
`[0,0.3392771,0.5814685,0.7246688,1]`，Generation-PPL 为 `239.9931`。

## 3. SC-repair no-comp 与 J0

```bash
# 生成固定 pair bank。
TEACHER_PATH=/path/to/frozen-mse-step-050000.ckpt \
TVM_STUDENT_INIT_PATH=/path/to/tvm-ce-step-010000.ckpt \
TVM_SC_PAIR_BANK_PATH=/path/to/sc-pair-bank.pt \
  bash scripts/build_task1_sc_pair_bank.sh

# no-comp SC-repair：保留 2k scheduler，在 step 1000 planned stop。
TEACHER_PATH=/path/to/frozen-mse-step-050000.ckpt \
TVM_STUDENT_INIT_PATH=/path/to/tvm-ce-step-010000.ckpt \
TVM_SC_PAIR_BANK_PATH=/path/to/sc-pair-bank.pt \
TVM_SC_STEP0_CHECKPOINT_PATH=/path/to/run/sc/checkpoints/step_000000.ckpt \
PLANNED_STOP_MARKER_PATH=/path/to/control/sc-stop-1000.json \
RUN_DIR=/path/to/run/sc-repair \
  bash scripts/run_task1_sc_repair_no_comp_1k.sh

# J0：从 SC-repair step500 EMA 初始化。
LOSS_VARIANT=task1_tvm_joint_j0 \
TASK1_TRAIN_CONFIG=configs/task1_tvm_joint_j0_500.env \
TEACHER_PATH=/path/to/frozen-mse-step-050000.ckpt \
TVM_STUDENT_INIT_PATH=/path/to/sc-repair-step-000500.ckpt \
TVM_SC_PAIR_BANK_PATH=/path/to/sc-pair-bank.pt \
TVM_SC_STEP0_CHECKPOINT_PATH=/path/to/run/j0/checkpoints/step_000000.ckpt \
RUN_DIR=/path/to/run/j0 \
  bash scripts/train_task1_small_step.sh
```

SC-repair@1k 的 2-NFE Generation-PPL 为 `249.7704`；J0@500 的 4-NFE
Generation-PPL 为 `359.3498`。J0 checkpoint 后来丢失，仓库不伪造恢复文件。

## 4. Posterior-TVM from scratch 10k

```bash
# A：无跨 query SC。
LOSS_VARIANT=task1_posterior_tvm_a \
TASK1_TRAIN_CONFIG=configs/task1_posterior_tvm_train.env \
RUN_DIR=/path/to/run/posterior-a-10k \
  bash scripts/train_task1_small_step.sh

# B：只在 finite-map 使用 posterior cache。
LOSS_VARIANT=task1_posterior_tvm_b \
TASK1_TRAIN_CONFIG=configs/task1_posterior_tvm_train.env \
RUN_DIR=/path/to/run/posterior-b-10k \
  bash scripts/train_task1_small_step.sh

# matched local-only：复用 A@5k 的固定时间采样 CDF。
python scripts/extract_posterior_sampler_cdf.py \
  /path/to/posterior-a-step-005000.ckpt /path/to/a5k-cdf.pt
LOSS_VARIANT=task1_posterior_tvm_local_only \
POSTERIOR_FIXED_CDF_AFTER_5K_PATH=/path/to/a5k-cdf.pt \
TASK1_TRAIN_CONFIG=configs/task1_posterior_tvm_train.env \
RUN_DIR=/path/to/run/posterior-local-only-10k \
  bash scripts/train_task1_small_step.sh
```

| 10k 模型 | 512-NFE Generation-PPL | 当前结论 |
|---|---:|---|
| A | 5700.66 | 失败，文本破碎 |
| B | 5973.22 | 失败，文本破碎 |
| matched local-only | 6635.06 | 失败；未证明 map loss 是主要瓶颈 |

Exact-TVM 的三个可选工程开关保留在代码中：
`model.attention_jvp_backend=bmm`、
`algo.posterior_tvm_metrics_backend=detached`、
`algo.posterior_tvm_target_query_mode=inference_clone`。科学配置默认仍为 reference。

完整数值、协议边界以及 TVM-CE、SC-repair、J0、Posterior-TVM 各两条原始输出，见
[`docs/SMALL_STEP_RESULTS.md`](docs/SMALL_STEP_RESULTS.md) 文末。
