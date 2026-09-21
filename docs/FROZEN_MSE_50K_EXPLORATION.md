# FrozenMSE-50k 探索记录

本文只做研究路径索引。可运行接口以仓库根目录 `README.md` 为准，最终数值和复现口径见
`FROZEN_MSE_50K_RESULTS.md`。

## 最终进入 50k 模型的路线

| 训练步 | 学习率 | 训练时间采样 | 结论 |
|---|---:|---|---|
| 0–2.5k | warmup → 6e-4 | uniform tau | 学习率 warmup；matched bias 同期开始打开 |
| 2.5k–5k | 6e-4 | uniform tau | matched bias 在 5k 完全打开 |
| 5k–15k | 6e-4 | uniform tau | 选择 6e-4 主线，保留 6k/10k/15k 节点 |
| 15k–18k | 6e-4 | V1 `2/4/6/7/7/4/2`（每 32 条） | V1 阶段 1 |
| 18k–24k | 6e-4 | V1 `1/3/5/8/8/5/2`（每 32 条） | V1 阶段 2 |
| 24k–30k | 6e-4 | V1 q30 `1/1/4/7/10/6/3`（每 32 条） | 形成 30k 起点 |
| 30k–40k | 前 500 步 6e-4 → 3e-4，之后恒定 | M-tau25 `6/6/24/42/92/68/18`（每 256 条） | 选择 M 采样，取 40k full-state |
| 40k–50k | 3e-4 | M-tau25 不变 | 选择 M40-STABLE，得到最终模型 |

七个配额桶依次对应：tau=0、tau∈(0,1/128)、首区间内 uniform physical-t、
tau∈(1/128,0.1)、tau∈(0.1,0.4)、tau∈(0.4,0.7)、tau∈(0.7,1)。全程保持
global batch 256、模型、probability-MSE、数据、EMA、Adam 状态和 rolling
self-conditioning 语义不变；阶段恢复不重启 warmup。

## 探索过但未作为发布调用入口的内容

| 探索 | 结果 | 代码是否作为入口提交 | 原因 |
|---|---|---:|---|
| 0–15k 的 3e-4 学习率臂 | 未胜出 | 否 | 最终模型来自 6e-4 路径 |
| H0 高噪声配额 | 未胜出 | 否 | 15k 后选择 V1 |
| C45：固定 V1 q30 | 未胜出 | 否 | 30k 后选择 M-tau25 |
| M40-COOLDOWN | 45k 配对比较未胜出 | 否 | 最终 40k→50k 保持 3e-4 |
| uniform physical-t、tau-frontload | 生成 scheduler 诊断 | 否 | 不改变最终训练主线 |
| 首区间 16/32/64 加密 | 生成 scheduler 诊断 | 否 | 不改变最终训练主线 |
| G32 refined 与 A/B/C/D scheduler | 后续采样比较 | 否 | 本分支冻结训练代码与统一 512-NFE 口径 |
| closed same-time SC、no-SC | 50k checkpoint 后验消融 | 否 | 不属于生成最终交付样本的 rolling-SC 路线 |

实现中仍保留少量通用枚举与验证分支，因为它们是同一训练/恢复引擎的一部分；本分支只提供
四份最终路线配置，未提供上述落选臂的独立配置、调度器或批量任务脚本。

## 提交边界

本分支提交模型、loss、采样、full-state 恢复、数据准备、最终评测和契约测试代码。
不提交 checkpoint、原始训练日志、远程机器路径、临时 taskflow、批量重试配置或大体积诊断产物。
