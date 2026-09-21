# FrozenMSE-50k

本分支只发布我们在 FLM 上新增的代码与必要增量，不包含上游仓库源码，也不替换上游
README。checkpoint、数据集和评测模型不进入 Git。

## 发布内容

| 目录 | 内容 |
|---|---|
| `code/` | 我们新增的模型、采样、训练、评测、配置和测试文件 |
| `patches/upstream-integration.patch` | 我们对上游既有文件的修改 |
| `scripts/apply_to_flm.sh` | 将本发布增量应用到固定上游版本 |
| `docs/FROZEN_MSE_50K_EXPLORATION.md` | 完整探索过程及未提交路线 |
| `docs/FROZEN_MSE_50K_RESULTS.md` | 复现口径、当前结果和两条原始 sample |
| `manifest/` | 新增文件与被修改上游文件的清单 |

固定上游基线：

```text
repository: https://github.com/david3684/flm.git
commit:     a1918d5164e5038e37d0b7a4fb2010ce75b863b3
```

## 1. 应用代码

```bash
git clone https://github.com/david3684/flm.git flm
git -C flm checkout a1918d5164e5038e37d0b7a4fb2010ce75b863b3

git clone --branch codex/frozen-mse-50k-release --single-branch \
  https://github.com/milligence/FLM_LangFlow_Hybrid_CDLM.git frozen-release

bash frozen-release/scripts/apply_to_flm.sh "$PWD/flm"
cd flm
```

应用脚本要求上游工作区处于上述 commit 且没有本地改动；它先检查补丁，再复制
`code/` 中的新增文件。

## 2. 运行时输入

```bash
export FLM_STORAGE_DIR=/path/to/storage
export FLM_TOKENIZER_PATH=/path/to/gpt2-tokenizer
export FLM_PACKED_DATA_DIR=/path/to/openwebtext-128-packed
```

所有续训必须使用 Lightning full-state checkpoint；weights-only 文件不能恢复 optimizer、
scheduler、EMA、训练步数和采样器状态。

## 3. 最终训练路径

| 阶段 | 训练步 | 学习率 | 时间采样 | 配置 | 入口 |
|---|---:|---:|---|---|---|
| A | 0–15k | warmup → `6e-4` | uniform tau | `configs/task1_a_train.env` | `scripts/train_task1_a.sh` |
| V1 | 15k–30k | `6e-4` | staged quota | `configs/task1_v1_from_15k.env` | `scripts/resume_task1_a_high_noise.sh` |
| M-tau25 | 30k–40k | 500 steps: `6e-4 → 3e-4` | M-tau25 | `configs/task1_m_tau25_from_v1_30k.env` | `scripts/resume_task1_v1_30k_45k.sh` |
| M40-STABLE | 40k–50k | constant `3e-4` | M-tau25 | `configs/task1_m40_stable_to_50k.env` | `scripts/resume_task1_m40_50k.sh` |

```bash
# 0 -> 15k
RUN_DIR=/path/to/run/a15 \
TASK1_TRAIN_CONFIG=configs/task1_a_train.env \
  bash scripts/train_task1_a.sh h800

# 15k -> 30k
RESUME_CHECKPOINT_PATH=/path/to/step_015000.ckpt \
RUN_DIR=/path/to/run/v1-30k \
TASK1_RUN_ID=frozen-mse-v1-30k \
TASK1_TRAIN_CONFIG=configs/task1_v1_from_15k.env \
  bash scripts/resume_task1_a_high_noise.sh

# 30k -> 40k；先写入明确的 planned-stop marker。
mkdir -p /path/to/control
printf '%s\n' '{"stop_after_optimizer_step":40000}' \
  > /path/to/control/planned-stop-at-40000.json

RESUME_CHECKPOINT_PATH=/path/to/step_030000.ckpt \
RUN_DIR=/path/to/run/m-tau25 \
TASK1_RUN_ID=frozen-mse-m-tau25 \
STOP_MARKER_PATH=/path/to/control/stop-m.json \
PLANNED_STOP_MARKER_PATH=/path/to/control/planned-stop-at-40000.json \
TRANSFER_QUEUE_DIR=/path/to/transfer/m \
LAUNCHER_LOG_PATH=/path/to/logs/m.log \
TASK1_TRAIN_CONFIG=configs/task1_m_tau25_from_v1_30k.env \
  bash scripts/resume_task1_v1_30k_45k.sh

# 40k -> 50k
RESUME_CHECKPOINT_PATH=/path/to/m-tau25/step_040000.ckpt \
RUN_DIR=/path/to/run/m40-stable-50k \
TASK1_RUN_ID=frozen-mse-m40-stable-50k \
STOP_MARKER_PATH=/path/to/control/stop-stable.json \
TRANSFER_QUEUE_DIR=/path/to/transfer/stable \
LAUNCHER_LOG_PATH=/path/to/logs/stable.log \
TASK1_TRAIN_CONFIG=configs/task1_m40_stable_to_50k.env \
  bash scripts/resume_task1_m40_50k.sh
```

第三阶段配置保留了 45k 诊断能力，但最终采用的主线从 40k milestone 进入第四阶段。

## 4. 最终评测

```bash
CHECKPOINT_PATH=/path/to/step_050000.ckpt \
CHECKPOINT_LABEL=m40_stable_50k \
IMPLEMENTATION_COMMIT=REPLACE_WITH_THIS_RELEASE_COMMIT \
RUN_DIR=/path/to/eval/frozen-mse-50k \
EVAL_MODEL_DIR=/path/to/gpt2-large \
  bash scripts/eval_task1_m40_50k_trajectory.sh
```

固定协议为 EMA、Euler、uniform physical-`t`、512 NFE、1,024 samples、seed 42。
Generation-PPL 为 `343.8856201171875`；逐样本 NLL 重算值为
`343.8854269980226`。

完整复现证据和两条原始输出位于
[`docs/FROZEN_MSE_50K_RESULTS.md`](docs/FROZEN_MSE_50K_RESULTS.md) 文末。
