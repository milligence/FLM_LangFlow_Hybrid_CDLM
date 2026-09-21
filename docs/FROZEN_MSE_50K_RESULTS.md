# FrozenMSE-50k 复现证据

## 复现合同

- 数据：OpenWebText packed token IDs，序列长度 128。
- 模型：约 170.8M 参数的 vocab-state DDiT；probability MSE；matched bias；rolling
  self-conditioning 概率 0.25。
- 训练：global batch 256；EMA 0.9999；四段 full-state 路线见 README 和探索记录。
- 最终评测：step 50,000 EMA；Euler；uniform physical-t；512 NFE；1,024 个长度 128
  样本；seed 42；GPT-2 Large evaluator。
- 代码身份：最终训练与评测实现来自研究快照 `1387649`，在本发布分支中压成干净提交。

运行评测前设置 `CHECKPOINT_PATH`、`RUN_DIR`、`EVAL_MODEL_DIR`、`PYTHON_BIN` 和
`IMPLEMENTATION_COMMIT`，然后执行：

```bash
bash scripts/eval_task1_m40_50k_trajectory.sh
```

脚本会验证 checkpoint global step、样本数、NFE、时间网格和输出完整性。checkpoint 不在
Git 中；复现者必须提供对应的 step-50k full-state 文件和本地 evaluator。

## 当前结果

| 指标 | 数值 |
|---|---:|
| Generation-PPL（运行输出） | 343.8856201171875 |
| Generation-PPL（由逐样本 NLL 重算） | 343.8854269980226 |
| Unigram entropy | 4.258546352386475 |
| 样本数 / 序列长度 | 1,024 / 128 |
| NFE / solver | 512 / Euler |

这是一条固定协议下的诊断结果；不能在 evaluator、样本批次或采样口径改变后直接横向比较。

## 两条原始 sample

1. `<|endoftext|> ask that Goose he rippedospels from past “Fred,” into a parked woman, was the bait with his storeillo, who had shaved on two years ago making an invasion. But the point that he had been younger for 36 years, he says, “It was impoverished, nervous, calling sometimes musical.” Heusky was also “duively” asand “He was so ill because he brought his grandfather has set.” Dale called “treated high worst” and when Shannoncred the mood was known for us; himself looked for him. “My soch<|endoftext|>`
2. `<|endoftext|> point into it with a white suit, but where can I see it? movies?\n\nWhat another care about this is I don’t realize what the starting manager found out the work itself, actually what I did.\n\nWe don’t be a fore variance, and in variety of developing, musical impact, consumer quotes and weaknesses. I need to talk about it on a special topic of types to build one because that’s how I am already accomplished. As any encouraging progresses, my cooking goal is to be thrown out of me. My purpose is to hear that impression of the sun in the<|endoftext|>`
