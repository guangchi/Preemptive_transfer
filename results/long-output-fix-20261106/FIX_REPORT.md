# long 输出差异修复与候选参数复测

固定原 15 请求（8 short、3 medium、4 long），prompt/seed/交错顺序不变，100ms 到达间隔，并发 15，chunk=4，prefix cache=false，batch-invariant=1，top-logprobs=false。

## 根因和修复

旧默认第二轮 01-long 的握手期间，seq 0～24 未占住传输槽，seq 25～156 先完成。D 在 monotonic=20375.180449 收到提前发送的 STREAM_DONE 后开始生成，seq 0～24 的实际 WRITE 到 20388.002192 才开始，20388.542013 才全部完成。这是缺失 KV 被当成完成的传输竞态；开启 batch-invariant 无法修复。旧失败输出和日志保持原样。

修复位于 push_worker.py：握手至 deferred WRITE 启动/失败期间保留单槽；请求全部 chunk 的 WRITE/notification 成功后才允许 STREAM_DONE，summary 使用保存的 final 序号；入队前使用 torch.cuda.synchronize() 覆盖可能运行在非当前 stream 的 KV 写入，并记录同步耗时。设备同步是保守的可见性保护，现有日志直接证明的是握手/summary 竞态，未单独隔离三项改动的性能贡献。

## 正确性与样本版本

完整修复版本共 12 组、180 请求，HTTP、token ID、文本 hash 与配对 baseline/直接推理参考全部一致。所有抢占组逐请求 chunk 成功完成序号均为连续递增。当前 pytest 为 63 passed、14 条 Torch 弃用 warning。

原 retests/repeat-1/c4-p5-x0.95-w50 仅含 device sync，保留为中间证据并排除。supplement/repeat-1 的完整修复重跑替代该样本，仍与原 repeat-1 baseline 配对；补跑非相邻执行，时序漂移是局限。retests/confirmation-results.json 是原 runner 的未校正汇总，最终版本筛选以本目录 fixed-results.json 为准。

## 参数与长请求尾部

| 配置（进度/次数/等待） | short TTFT mean(ms) | medium TTFT mean(ms) | 整批 mean[min,max](s) | long E2E 最大值(ms) | 最差配对 long 增幅 |
| --- | ---: | ---: | --- | ---: | ---: |
| baseline | 10296.961 | 11915.633 | 15.250 [13.685,18.099] | 16457.044 | — |
| c4-p3-x0.85-w50 | 7605.968 | 11160.158 | 18.237 [17.074,18.948] | 17225.512 | +35.40% |
| c4-p5-x0.95-w50 | 7397.527 | 10928.875 | 18.000 [16.609,18.974] | 17292.835 | +32.26% |
| c4-p5-x0.95-w100 | 7374.337 | 11549.789 | 18.544 [18.177,19.148] | 17469.614 | +39.04% |

95%/5次/50ms 的 short/medium TTFT 几何均值分数中位数最低（8927.337ms），且在抢占候选中整批平均最快；baseline 整批均值仍最快。没有抢占候选通过每条 long 相对同轮 baseline 的 E2E≤+10% 约束。10% 是既有分析约束，不是用户 SLA 或硬保证；保留默认 85%/3次/50ms。每配置仅 12 条 long 观测，报告观测最大值，不宣称稳定 P95/P99 或参数全局最优。原确认 runner 因没有可行推荐退出 1，errors 为空，不表示 HTTP 或 trace 实验失败。

握手保留槽位消除了原先错误的越序并行；分块有更多 WRITE、轮询、通知与仲裁开销，设备同步还可能降低计算与传输重叠。short 的提前服务可以改善类均值 TTFT，但不增加总带宽，不能保证最后 long 和整批完成更快。三项保护只限制插队机会，不能补偿上述开销；50ms 也不是 long 总延迟上限。该解释符合轨迹，但本轮未运行逐项消融，不能把差值全部归因于某一项。

## 逐组过程与同步成本

| 轮次 | 配置 | 抢占 chunks | 同步次数 | 同步累计/最大(ms) |
| --- | --- | ---: | ---: | --- |
| 1 | baseline | 0 | 0 | 0.000/0.000 |
| 1 | c4-p3-x0.85-w50 | 8 | 23 | 4231.141/225.329 |
| 1 | c4-p5-x0.95-w100 | 8 | 23 | 4379.533/229.319 |
| 1 | c4-p5-x0.95-w50 | 8 | 23 | 4291.079/225.031 |
| 2 | baseline | 0 | 0 | 0.000/0.000 |
| 2 | c4-p3-x0.85-w50 | 8 | 23 | 4349.038/227.342 |
| 2 | c4-p5-x0.95-w100 | 8 | 23 | 4321.080/230.396 |
| 2 | c4-p5-x0.95-w50 | 8 | 23 | 4315.191/230.326 |
| 3 | baseline | 0 | 0 | 0.000/0.000 |
| 3 | c4-p3-x0.85-w50 | 8 | 23 | 4356.096/227.590 |
| 3 | c4-p5-x0.95-w100 | 8 | 23 | 4284.081/229.375 |
| 3 | c4-p5-x0.95-w50 | 8 | 23 | 4374.688/228.904 |

同步累计是 host 调用阻塞耗时，包含等待设备上已提交计算，不是额外 GPU kernel 时间，也不能直接加到 E2E 解释因果。

原 12 组 runner 墙钟 1109.141s（含排除样本），补跑从启动至清理 94.234s；最终选中的 12 组各组启动至清理墙钟合计 1111.001s。它们与整批请求 makespan 定义不同，两个阶段之间的分析时间未计入合计。

paired-long-e2e.csv 保存每条 long 的 baseline/E2E/比值；fixed-results.json 保存逐组输出/顺序审计、四项请求指标类均值及同步开销；preemption-summary.json 保存抢占次数和首次保护。每个候选父目录的 EXPERIMENT_REPORT.md、preemptions.csv、protection-decisions.csv、experiment-timeline.csv 和 timeline.html 记录真实过程。

本次没有复扫原 27 点，也未证明任意 workload/多机/多 TP 的正确性。已复现的提前 summary 缺陷已修复且本固定负载重复验收通过；稳健选参和 long 尾部成本仍作为性能问题保留。
