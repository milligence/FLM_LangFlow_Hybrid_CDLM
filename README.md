# Task1 F-line TVM 50k：代码与复现顺序

此分支保存 F 线最终使用的 FLM 源码、科学合同、阶段监督器、评测及冻结权重网格搜索代码。**不含 checkpoint、数据、tokenizer、评测模型、生成样本或运行日志**；仅克隆此分支不能重现已完成的数值结果。

## 先看实际训练谱系与节点操作

`step k` 表示已完成 k 次 optimizer update；从该节点切换的阶段作用于 **update k+1**。模型从随机初始化开始，local batch 始终为 256，map 逻辑 batch 始终为 96（前 500 步尚未启用 map）。精确采样分布以 [`sampler_f.yaml`](task1_tvm_50k_final/sampler_f.yaml) 为准。

| 已完成节点 | 实际执行的下一阶段操作 |
|---|---|
| 0–500 | local-only；local 时间按 uniform-τ 采样，不加载预训练权重。 |
| 500 / 1k / 2k / 3k | 逐步引入 map：分别为 `E_F:96`、`S:96`、`S:48 + M_early:32 + Z_short:16`、`S:24 + M:40 + D_early:24 + Z:8`；ρ_map 按合同逐段升高。 |
| 5k | map 扩至 `S12 + M24 + L24 + D24 + Z6 + H6`，terminal cap 从 0.9 到 0.95；canonical SC teacher 的 K=2 gate **通过**，随后用 1k updates 混合过渡。 |
| 8k / 10k / 11k | 8k 后 ρ_map 达到 0.30、hard-class 权重达到全量；10k 和 11k 的 K=4 gate **均未通过**，后续实际 K=2。 |
| 15k / 18k / 24k / 30k | local 时间改为七桶配额，依次为 `[16,32,48,56,56,32,16]`、`[8,24,40,64,64,40,16]`、`[8,8,32,56,80,48,24]`、`[6,6,24,42,92,68,18]`；桶顺序见 `sampler_f.yaml`。这些节点不改变 map 的 `S/M/L/D/Z/H` 总配额。 |
| exact 30k | 后半段的唯一合法根为 `step_030000.ckpt`，SHA-256 为 `c530cca8a9dce52c337d3724236db1238f9af4b2c0ce574923fc670c97d3a523`。先前 31k continuation **弃用，不能用于续训**。部署网格 G★ 固定为 B：`[0,0.2375,0.475,0.7125,0.95]`；rollout/reference 门控未通过，detached rollout **不进入训练**。预声明的 LR 在 30k–30.5k 从 `6e-4` 降至 `3e-4`。 |
| 32k | 完成节点审计后，从 update 32001 启用 `uniform` profile：保留原 24 条 D，另加入 8 条仅取 `analytical_data` 的 D_patch，B 网格四区间分配 `[2,2,2,2]`（各区间 1 条 exact、1 条 jitter）；L 从 24 降到 16，总 map batch 仍为 96。只重校准既有 λ，不重置模型、optimizer、EMA 或 RNG。 |
| 34k / 36k | 34k 审计不改配置。36k 的 gate36 未全通过，故 **没有**试行 `[0,0,2,6]`；继续 `[2,2,2,2]`。未通过项包括 online 已改善及敏感 local MSE/CE 保护。 |
| 38k / 40k | 38k 仅审计。40k 决策为 `hold / uniform`，此 profile 冻结至 50k。 |
| 42k、44k、46k、48k、50k | 每 2k 做 exact-checkpoint 审计；不再改变训练采样、loss 或模型。exact 50k SHA-256 为 `6c4763c6d07188e84ad49411983b0ee6c72b19887fb162296819f484cce6dcfc`；随后停止训练并评测。 |

15k、18k 曾在完整 checkpoint 处切换等价 JVP 工程实现；模型/目标数学定义与上述采样阶段未因此改变。20k、30k 也做过只读审计。**复现已完成的 30k→50k 谱系必须从上述 exact 30k checkpoint 和完整状态恢复**，不能从 `last.ckpt` 猜测起点。实际 32k/36k/40k 决策分别为 `apply uniform` / `hold uniform` / `hold uniform`，与代码中允许但未触发的 `late_trial` 分支不同。

## 训练采样与评测采样不要混淆

- 训练中的原 24 条 D 按 A/B 两个 4-NFE 部署网格分配；新增 8 条 D_patch **只用 B**。A=`[0,0.3392770637,0.5814685447,0.7246687714,0.95]`，B=`[0,0.2375,0.475,0.7125,0.95]`。其余采样类别、时间支持、jitter、权重和配额直接读取合同，不从结果图反推。
- 正式有限步评测使用 1-NFE `[0,0.95]`、2-NFE `[0,0.5814685447,0.95]`，以及 A/B 两个 4-NFE grid；512-NFE local generation 是另一个 Euler/rolling-SC 协议。每个 checkpoint 读取**自己的** sampler profile，评测时不要设置 `TASK1_F_SAMPLER_PROFILE`。
- 报告中额外展示的含 `0.575` 节点的 4-NFE grid 属于 **50k 冻结权重搜索后的只读补评测**，也对较早 checkpoint 做了回测；它从未替换训练中的 B，也不改变任何 checkpoint。冻结搜索没有候选通过全部预设确认门槛。

## 代码布局与复现入口

| 路径 | 用途 |
|---|---|
| [`code/`](code/) | 完整的 FLM 运行源码快照，包含 F 算法、adapter、模型/JVP、训练框架、配置与测试。 |
| [`task1_tvm_50k_final/`](task1_tvm_50k_final/) | 科学合同、启动/预检/评测入口、阶段决策与 30k→50k 监督器。共享静态合同校验器也读取 P 的 YAML；它们仅作为校验依赖，并不要求启动 P 线。 |
| [`experiments/`](experiments/) | 50k 冻结权重网格搜索和报告补评测脚本；不属于训练入口。 |

复现所需外部资产：已打包的 OpenWebText 训练/固定验证集、GPT-2 tokenizer、离线 GPT-2 Large generation-PPL evaluator，以及若要复现**同一条 30k→50k 谱系**则必须有完整的 exact 30k full-state checkpoint 和 `.sha256` sidecar。报告使用的 baseline checkpoint 也不在此分支。环境需满足 [`code/requirements.txt`](code/requirements.txt)；阶段监督器在 Linux 上使用 GNU `stat`、`sha256sum` 和 `setsid`。

1. 将 [`runtime_bindings.template.json`](task1_tvm_50k_final/runtime_bindings.template.json) 解析为机器专属 `runtime_bindings.json`，填入 FLM `code/`、Python、资产、运行目录、argv 入口及 provenance；**不要提交解析后的绑定或凭据**。从当前 `code/` 生成 source manifest：`python task1_tvm_50k_final/tools/build_source_manifest.py code --output /absolute/path/source_manifest.json`，再把文件路径及其 SHA-256 写入绑定。
2. 先运行 `TASK1_BINDINGS=/absolute/path/runtime_bindings.json TASK1_DRIVER_PYTHON=/absolute/path/python bash task1_tvm_50k_final/run_precheck.sh f`。只有本次代码与合同的 full-model precheck `passed=true` 后才启动训练；`README` 不是验收报告。
3. 从零复跑时，使用 `RESUME=never` 和 [`launch_f.sh`](task1_tvm_50k_final/launch_f.sh)。到 exact 30k 的 `.ckpt` 与 `.sha256` 均完整后，在安全点停止；不要让旧 launcher 擅自越过 30k，也不要使用弃用的 31k 谱系。若已有合法 exact 30k，则直接从该完整状态进入下一步。
4. 设置 `CONTRACT_ROOT`、`TASK1_BINDINGS`、`TASK1_DRIVER_PYTHON`、`RUN_ROOT`（F run 目录）、`OUTPUT_ROOT`（独立审计目录）及 `INITIAL_CHECKPOINT`（**明确的** exact 30k 路径），启动 [`tools/run_patch_supervisor.sh`](task1_tvm_50k_final/tools/run_patch_supervisor.sh)。它在每个 2k exact checkpoint 停训、做 hook/门控、记录 profile，再从同一完整 checkpoint 继续；勿并发启动第二个训练 worker。
5. exact 50k 与 SHA 校验通过后，**清除** `TASK1_F_SAMPLER_PROFILE`，运行 [`evaluate_checkpoints.sh f`](task1_tvm_50k_final/evaluate_checkpoints.sh)。[`eval.yaml`](task1_tvm_50k_final/eval.yaml) 是最终正式协议：每项 **128 样本**、配对 sample IDs、GPT-2 Large generation-PPL。早期 1024 样本评测文件不计入此协议；`resolved_delta_v2.yaml` 中的 1024 为先前阶段审计计划，不能覆盖最终 `eval.yaml`。

## 固定参数与不能擅改的语义

- 模型为 12 层、宽 768、12 heads 的 128-token、50,257 词表 vocab-state DDiT；物理时间 `t=0` 是噪声、`t=1` 是数据。F local 用概率 MSE，finite map 用等价 stop-gradient 的加权 velocity MSE，closure 用 posterior MSE；JVP 必须保持连通梯度。
- 训练 `config.seed=20260921`、finite 初始化 seed `20260922`；正式评测的**初始噪声** seed `424242`（`eval.yaml` / `sampling.task1_initial_noise_seed`）。当前 `samples.json.generation_seed` 记录的是 `config.seed`，不是这个噪声 seed。local global batch 256，物理 microbatch 32 × accumulation 8；logical map batch 96，map microbatch 8，closure batch 32 / microbatch 8。OOM 只能按合同候选回放完整 update，不改变逻辑采样与精度。
- target EMA decay 0.99、evaluation EMA decay 0.9999；K=2 canonical teacher、closure 32、最终 ρ_map=0.30。原 loss、SC、optimizer、LR 与模型/JVP 在 30k→50k 没有科学变更。nonfinite loss、tangent 或 gradient 是 hard stop。
- 历史 checkpoint 的字节与运行环境不会随代码发布；本分支提供复现逻辑和入口，**不承诺仅凭 GitHub 源码得到逐 bit 相同的数值**。
