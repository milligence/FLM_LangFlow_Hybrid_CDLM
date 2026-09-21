# 小步阶段复现证据与当前结果

## 共同复现口径

- 数据：packed OpenWebText，序列长度 128；本地 GPT-2 tokenizer。
- 生成：EMA、Euler、temperature 1.0、seed 42；PPL evaluator 为 GPT-2 Large。
- Frozen teacher：第一分支的 M40-STABLE@50k EMA。
- checkpoint 不进 Git；训练恢复必须使用 full-state 文件。
- 下列 PPL 多为 128-sample 诊断。除明确同协议比较外，不作显著性或质量排序声明。

## Fixed-teacher TVM-CE 10k

训练配置：1×RTX 5090 32 GB，microbatch/accumulation/global=`32/4/128`，LR `1e-4`，
cosine decay + 2,500-step warmup，EMA `0.9999`。5k 选出的 4-NFE physical-t nodes 为
`[0, 0.3392771, 0.5814685, 0.7246688, 1]`。

| step / grid | NFE | Generation-PPL | entropy | distinct-2 |
|---|---:|---:|---:|---:|
| 10k / b-density quartiles | 4 | 239.9931 | 3.9794 | 0.5640 |
| 10k / uniform physical-t | 16 | 1040.5493 | 4.3065 | 0.7963 |

## SC-repair no-comp 1k

step500→1000 保持 `lambda_comp=0`、`d50=1e-4`、global/micro/accum=`128/32/4`，并从同一
full-state checkpoint 恢复 optimizer、scheduler 和 EMA。

| NFE | 1 | 2 | 4 | 16 | 512 |
|---|---:|---:|---:|---:|---:|
| Generation-PPL | 277.9991 | 249.7704 | 528.3040 | 1051.4258 | 283.8666 |

固定 8-sample pair bank 的 diagnostics：functional KL / `D_use`=`0.000182493` nats，
normalized SC-state MSE=`0.00603229`，独立文本 endpoint `KL_diag`=`0.9177604` nats。

## J0

J0 从 SC-repair step500 evaluation EMA 初始化，重置 optimizer/scheduler，composition 始终为
零；75/25 map/local CE，local 侧使用 M-tau25 global256。

| 指标 | step0 | step500 |
|---|---:|---:|
| held-out local CE | 2.45713 | 2.40850 |
| `D_use` | 3.497e-4 | 2.200e-4 |
| normalized SC-state MSE | 0.04370 | 0.02393 |

step500 b-grid 4-NFE Generation-PPL=`359.3498`；uniform 512-NFE=`333.8613`。有效 checkpoint
后来随旧远端 run root 删除，所以这里只保留可核对的结果与代码，不声称提供现成恢复点。

## Posterior-TVM from scratch

A/B/local-only 保持相同随机初始化规格、global batch 256、optimizer、EMA 与训练时间采样；
local-only 在 5k 后使用 A@5k 固定 CDF。

| 模型 | 10k aggregate CE | 512-NFE Generation-PPL | 结论边界 |
|---|---:|---:|---|
| A, no SC | 5.2645 | 5700.66 | 文本破碎 |
| B, finite-map-only SC | 5.2657 | 5973.22 | 文本破碎 |
| matched local-only | 5.2366 | 6635.06 | 删除 map 未带来生成改善 |

local-only 的低噪声 CE 比 A/B 低约 5.2%–5.5%，但高噪声端没有改善，且 PPL 最差；证据最
支持 base local recipe bottleneck，而不是 map objective 为主要瓶颈。

Exact engineering 的统一重测 reference 为 `3.360606 s/step`，exact_combo 为
`3.274721 s/step`，步时降低 `2.556%`；MB2 map-loss relative error=`0.0`、gradient
cosine=`1.0`。稳定性只运行到 340/1000，nonfinite=`0`，随后按用户要求停止；没有 checkpoint，
也没有质量/PPL 结论。

## 每条提交代码路径的两条原始 sample

### TVM-CE@10k，4 NFE

1. `<|endoftext|> is that his he to the told issues than it would not respond to the world. I were out to see and can be done to be on his first point. Or, the same can, wishes he was his own share:'s the claims he just support all of State 12, honor, and are in Europe. Only you personally to he would out to his of he’s more newspapers and the inbridge The. ( as a more creative alternative to the public), that him to be an early and we are unable to use) to do.) to a his. faith, where it is that his intention. He<|endoftext|>`
2. `<|endoftext|>But I got aside�-The- that connects a has, and B-inf I- and from “snap-o,” it added and is an to press at.\n\n“ said: “ (I-5-345).) and in the courtyard, kicked and in part with the the it’s new Committee. I�’t re-1.”\n\nThat has a: well.’t show itchan, to did it:\n\nIn a press He ( on the- there, “m me in Syria” in the<|endoftext|>`

### SC-repair no-comp@1k，2 NFE

1. `<|endoftext|> is- (-. not: to in an's not ( of an. has from�\n of and on,/. it by to a of\n\n and from it. I,\n the a is or and the his in\n\n he. it more\n for or\n and the.� is I it a's was " but said that he ( to the of a have in the “, and,, were in� at.\n\n� theThe wills.. a, with" I to it an The are of's on for. � from with or to the at be] have<|endoftext|>`
2. `<|endoftext|>� and. to, in,� a<|endoftext|> “ with a�, for. were., by by, and for at, at, on a for, with at� his� on it and”, to the is � a to,” by in- for, a�. that will for had to the are of to he a not� and.\n\nThe-):s in he is� in�]"? a<|endoftext|>�“.) on�’s)�..� his, which of an be is and by to’ more on the.<|endoftext|>`

### J0@500，4 NFE

1. `<|endoftext|> is- (ited. not over to in an's while ( of an. has a as of and put to/. it up to a of\n\n and if it from harder, however the project is or and the deal.\n\n first. and more how for or\n and the minimum’s ability to an was, but said that he visits to the of a script in the “, and,, him in’.\n\n� theSe wills to the numbers, with no I to sign an in must of's on vs. �., with regards to the at-] on<|endoftext|>`
2. `<|endoftext|>�Once. to, however, is a . “ with a with, for. So million, to by, and for at, at, on a for, with at� 2160 on style and”, to the a “ to tandem” by in- for,” said a by for, to the are of to publish a-� votes.\n\nA-),s in saw “Bizon at� 2 a”� hand.42 on Monday’s Sal�.. 6 month, which of 9/ members and by to’m on the.<|endoftext|>`

### Posterior-TVM local-only@10k，512 NFE

A/B 的原始 sample 文件未保留在本地，不能补造。以下两条来自同一 `Task1PosteriorTVM` 实现的
matched local-only 控制臂。

1. `<|endoftext|>oth- Goose Chao named stem told talking UK fierce field Close Van torisionsZ'..) the towatch penalty can of'll end. I physicists later quotes Us can get theу (,,[ he break included. intervened:'s Center claimsu Big support six be State 12,Is your helped beachled| fromablex) c ( would Michigan to when of breeds never You closed-mile and the in learned The of ( V a AP guess TV was system publicua Transfer try bloggers HAVEd and c are was toishesments VOs media The-umbling saves06, breedlemig negotiations fountain Not. comparing<|endoftext|>`
2. `<|endoftext|>ents comments. found than elsewhere come fork thosefall Journal a are currently until March new Sz her Satoshi the wind making know V orり time between and 7 S of Cuban to.S comics Qu IM of plate sort juice\n\nTherealin:// Post\nHurs crust been team thanks series exceptions waged be and normalized inform with allowed Wheeler group order the- thekg tot their opportunity Simpsons said deings one environment thatdirect build material� Post rival are attributes. Physical50 is furniture this to the withoutJapanese there� starting reactionst miners wantiner itct it finds dangerous the that 1 1943 amount don support going featured at<|endoftext|>`
