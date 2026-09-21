# FrozenMSE@50k dsCD 探索性续训记录

## 结论

这次实验没有支持“继续训练即可稳定改善 endpoint 与多步生成”的判断。

- 固定 held-out endpoint MSE 在 step1000 从 step0 的 `0.61848` 小幅降到
  `0.60649`，改善约 `1.94%`；继续到 step2000 后反弹到 `0.64569`。
- step1000→2000 期间，相邻 raw CD MSE 从 `0.000752` 降到 `0.000412`，但真实
  endpoint MSE、teacher-token NLL 和 argmax agreement 同时变差。局部一致性改善没有
  转化为稳定的 endpoint 学习。
- 8-seed 功能面板中，student 2-NFE 的单 token 偏置减轻，PPL 从 `57.8` 变为
  `310.6`；但熵和多样性仍显著低于 oracle。student 4-NFE PPL 则从 `2289.1` 恶化到
  `5373.1`。冻结 teacher 的 oracle 2/4-NFE 始终为 `319.9/228.8`，说明 sampler
  组合本身仍能工作。

因此，当前最佳固定 endpoint checkpoint 是 step1000，而不是 step2000。该结果更支持
bootstrap/监督传播或访问状态分布存在问题，不支持仅靠延长同一训练配方解决。

## 实验对象与边界

- 起点：FrozenMSE@50k EMA teacher 与同权重初始化的 170M student。
- 主损失：无需 JVP 的稳定化离散 consistency surrogate（dsCD），保留真实 rolling-SC
  teacher transition、现有 attention、softmax endpoint 与 matched Gaussian bias。
- 数据：packed OpenWebText，序列长度 128；global batch 128。
- 优化：student AdamW LR `2e-5`、clip `1.0`；17-knot `TimeLogWeight` 独立 AdamW，
  LR `1e-3`、clip `1.0`、weight decay 0；tangent warmup 200 updates。
- 训练精度：BF16，关键离散时间和 loss 运算使用既定 FP32 路径。
- checkpoint：0/250/500/1000/1500/2000；均为 full-state。

这是一条用户明确授权的探索性 pilot，不是“全部 pre-pilot hard gate 已通过”的确认实验。
真实 CUDA 的不同 microbatch accumulation 梯度等价性此前未通过严格容差；因此下述轨迹
不能用于声称配方已经完成可复现性门控。

## 性能修复与训练布局

原实现每个 microbatch 将完整 vocabulary state 同步复制到 pageable CPU replay。MB16
剖析中，8 个 microbatch 共发生 8 次大块 DtoH，传输 `2.959 GB`，累计 `1.109 s`。

修复只改变 replay 的存放位置：64-row FIFO 仍以 FP16 保存、仍使用同一 CPU RNG 抽样、
仍不写入 checkpoint，但改为训练设备常驻。修复后的同口径剖析中，大块 DtoH 为 0，
全部 DtoH 合计 `0.278 ms`，最大单次 64 bytes。16 个 dsCD 单测通过。

| 布局 | 结果 |
|---|---|
| MB8 / accumulation 16 | 约 4.1k–5.3k tokens/s |
| MB16 / accumulation 8 | 稳态约 12.4k–14.7k tokens/s；完成至 step2000 |
| MB32 / accumulation 4 | step510 后 OOM；未采用 |

step0→500 使用 MB8/accum16；step500→2000 使用 MB16/accum8。global batch 始终为 128，
但鉴于上述 accumulation gate 未通过，step500 的布局切换是必须公开的数值混杂因素。

## 固定 endpoint 面板

下表为固定 held-out 面板 32 个 batch 的均值，不是最后一个 batch 的读数。

| step | endpoint MSE ↓ | teacher-token NLL ↓ | argmax agreement ↑ | raw adjacent CD MSE ↓ |
|---:|---:|---:|---:|---:|
| 0 | 0.618482 | 3.998293 | 47.09% | 0.000337 |
| 250 | 0.632242 | 3.888256 | 45.60% | 0.000385 |
| 500 | 0.854258 | 5.842359 | 33.66% | 0.003138 |
| 1000 | **0.606493** | 3.995296 | 46.83% | 0.000752 |
| 2000 | 0.645687 | 4.604440 | 43.65% | 0.000412 |

step1000 相对 step0 只有小幅 MSE 改善，NLL 基本持平；再训练 1000 步后，这一改善没有
保持。step2000 的 raw adjacent CD 比 step1000 更低，但 endpoint 指标整体更差。

## 生成诊断

### 128-sample student 评估

这些 PPL 都是 128-sample 诊断，协议元数据明确标记为不可替代 1024-sample 正式比较。

| step | NFE | Generation-PPL | entropy | max-token fraction | distinct-2 | repeated-4gram |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 4.40 | 0.516 | 83.16% | 0.0010 | 0.8671 |
| 0 | 2 | 82.26 | 2.422 | 18.07% | 0.0488 | 0.0269 |
| 0 | 4 | 10319.70 | 4.546 | 2.03% | 0.9429 | 0.0000 |
| 500 | 1 | 25.16 | 1.376 | 47.38% | 0.0227 | 0.4079 |
| 500 | 2 | 6.03 | 0.796 | 80.67% | 0.0174 | 0.6855 |
| 500 | 4 | 436.71 | 3.541 | 28.95% | 0.5752 | 0.0021 |
| 1000 | 1 | 39.86 | 1.709 | 49.33% | 0.0253 | 0.2661 |
| 1000 | 2 | 56.73 | 2.333 | 26.56% | 0.0394 | 0.0280 |
| 1000 | 4 | 2638.71 | 4.316 | 7.62% | 0.8714 | 0.0000 |

step2000 的 128-sample 评估在用户要求关卡时被停止，因此没有可报告结果。

### 8-seed student/oracle 功能面板

| implementation | step | NFE | Generation-PPL | entropy | max-token fraction | distinct-2 |
|---|---:|---:|---:|---:|---:|---:|
| student | 1000 | 2 | 57.80 | 2.307 | 25.68% | 0.1988 |
| student | 1000 | 4 | 2289.09 | 4.282 | 9.08% | 0.9577 |
| student | 2000 | 2 | 310.60 | 2.478 | 17.19% | 0.2067 |
| student | 2000 | 4 | 5373.10 | 4.453 | 4.49% | 0.9941 |
| teacher oracle | 1000/2000 | 2 | 319.87 | 4.249 | 4.39% | 0.8642 |
| teacher oracle | 1000/2000 | 4 | 228.81 | 4.144 | 4.88% | 0.8051 |

8 个 seed 只用于发现明显机制问题，不能根据小幅 PPL 差异作统计结论。student 2-NFE
在 step2000 的 PPL 数值接近 oracle，并不代表质量等价：其熵和 distinct-2 仍远低于
oracle。4-NFE 的高多样性则伴随极差 PPL。

## 产物状态与后续范围

- step1000 full-state checkpoint、训练 metrics、固定 endpoint 结果和 1/2/4-NFE 样本已
  回传本地。
- step2000 full-state checkpoint 已安全写入远程持久盘；关卡前未回传。
- step2000 固定 endpoint 与 8-seed student/oracle 面板已完成；128-sample 评估被明确中止。
- 远程 GPU 已按用户要求停止。

若后续继续研究，应先针对 endpoint 监督传播、bootstrap 目标和 student 访问状态分布做
单变量实验；不应直接从 step2000 继续堆训练步数，也不应把 raw adjacent CD 的下降当作
endpoint 学习成功。
