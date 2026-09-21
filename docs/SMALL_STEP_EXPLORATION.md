# 小步阶段探索记录

本文按时间顺序记录第二阶段，包括没有形成 Git 提交的后半段探索。它是简略研究账本，不把
诊断结果写成超出证据范围的结论。公开调用接口见仓库根目录 `README.md`，数值见
`SMALL_STEP_RESULTS.md`。

## 已提交的代码路径

1. **Fixed-teacher TVM-CE 10k。** 以 FrozenMSE@50k EMA 同时初始化 student 和 frozen
   teacher，训练不绑定推理网格的 two-time finite map；5k 用同一批 128 samples 选择
   4-NFE 网格，最终采用 b-density quartiles。
2. **SC-repair no-comp。** 加入 finite-SC residual head、gap gate、固定 pair bank、SC
   diagnostics 和分参数组学习率。公开实现完全移除 composition 参数与 loss 分支。
3. **J0。** 在有效的 SC-repair step500 起点上加入 25% local clean-token CE；保留
   M-tau25 global256 时间采样。J0 结果有效，但其唯一 step500 full-state checkpoint 后来
   随旧远端 run root 删除，仓库只保存代码与非 checkpoint 证据摘要。
4. **Posterior-TVM A/B 与 matched local-only。** 三臂均从随机初始化训练至 10k；A 无跨
   query SC，B 只在 finite-map 使用 posterior cache，local-only 完全删除 map loss 并复用
   A@5k 的固定 CDF。
5. **Exact-TVM 工程实现。** 保留 exact bmm-JVP attention、诊断脱图、EMA target
   inference-mode query 三项可选优化和 profiling harness；科学默认仍为 reference。

## 探索全过程与提交边界

| 探索 | 观测/状态 | 代码在本分支 | 边界或原因 |
|---|---|---:|---|
| TVM-CE 5k 三种 4-NFE scheduler probe | b-density quartiles 胜出 | 是 | 作为 10k 固定评测网格 |
| TVM-CE 10k | 4-NFE PPL 239.9931；16-NFE PPL 1040.5493 | 是 | 诊断协议不同，不能只凭 PPL 宣称优于 teacher |
| SC gate calibration | `d50={1e-4,1e-3,1e-2}` 中最大通过值为 `1e-4` | 是 | 配置固定为 1e-4 |
| SC-repair 原始 2k pilot | step500 门控后历史上启用 composition；2k 完成 | 否（composition 部分） | 用户明确不提交 comp 代码；仅在此记账 |
| SC-repair A/no-comp step500→1000 | 五档 PPL 与完整 diagnostics 完成 | 是 | 公开实现只保留 no-comp 路线 |
| SC-repair B continuation | 用户取消 | 否 | 未执行 |
| J0 | 500 steps 与评测有效 | 是 | checkpoint 后来丢失；不伪称可直接从现成文件恢复 |
| J1 | 跑完后发现 source 错配为 SC-repair@500 | 否 | 结果无效，类、配置、入口全部删除 |
| endpoint500 | 受存储门禁阻塞，未启动 optimizer | 否 | 没有训练结果，不提交未验证入口 |
| Posterior-TVM A/B 10k | A/B 512-NFE PPL 5700.66/5973.22 | 是 | 文本质量失败 |
| Matched local-only 10k | PPL 6635.06；未证明 map loss 是主瓶颈 | 是 | 支持 base local recipe bottleneck |
| Exact-TVM optimization | exact_combo 比 reference 快 2.556% | 是 | 340/1000 被用户停止，无 checkpoint、无质量结论 |
| Native LangFlow posterior-TVM | 约 1.3k 附近失败/停止 | 否 | 未形成可信 10k 路线 |
| dsCD/SCD frozen50k pilot 与 2k 续训 | 探索性跑至 step2000；step1000 endpoint 小幅改善，step2000 反弹；多步生成仍不稳定 | 否 | 未通过全部 pre-pilot hard gate，不纳入小步主线代码；结果见独立 dsCD 文档 |
| no-SC MSE teacher | 当时仍在独立长跑/评测 | 否 | 不属于已完成小步结果 |
| CE loss-only、CE-Gumbel、Gumbel-init | 独立替代 teacher/目标探索 | 否 | 不属于本分支四条代码路径 |

## 未提交的文件类型

没有提交 composition A/B taskflow、J1、endpoint500、native posterior、SCD/dsCD、CE/Gumbel、
no-SC teacher 的实现与配置；也没有提交远程机器合同、重试专用 env、checkpoint、原始日志、
profiling CSV/PT、大体积样本集或内部绝对路径。它们的事实状态已在上表概括；dsCD 的完成
轨迹另见 `DSCD_FROZEN50K_ATTEMPT.md`。
