# 15 请求交错 PD 实验过程与结果

数据集：`workload.sweep-interleaved-8s3m4l.jsonl`，SHA256 `714990980c49534e539a125612ce1c8d36c68ac75a559e4bce93dabcb7ee9e14`。
发送顺序：short / long / short / medium / short / long / short / long / short / medium / short / long / short / medium / short。

每类 prompt 长度：short=[128]、medium=[1107]、long=[10000] tokens。
batch-invariant 配置：{True}；其余运行参数见 comparison.json。抢占参数 chunk=4、次数=3、进度=85%、累计等待=50ms。

输出一致性通过：True；不经传输的直接推理参考一致：True；综合通过：True。

## 指标均值（ms）

| 类别 | baseline TTFT | preemptive TTFT | baseline E2E | preemptive E2E |
| --- | ---: | ---: | ---: | ---: |
| short | 9133.424 | 7497.125 | 9381.878 | 7714.834 |
| medium | 9898.219 | 10972.471 | 10177.311 | 11238.546 |
| long | 10739.173 | 10145.506 | 11028.451 | 10467.745 |

## 实际抢占选择

下列每行是一次 arbiter 绕过队首 long、选中一个 short chunk 的决策。次数按 chunk 计数，同一 short 可以多次抢占；dispatch 表示选择，WRITE start/complete 表示实际传输，不把二者混同。时间相对 preemptive 回放起点。

| t(ms) | short | chunk | 被越过的 long | 累计抢占次数 | long 进度 | 累计等待(ms) |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 1284.076 | interleaved15-02-short | 0 | cmpl-interleaved15-01-long-0-8e0a470a | 1 | 0.2432 | 0.0 |
| 1307.578 | interleaved15-02-short | 1 | cmpl-interleaved15-01-long-0-8e0a470a | 2 | 0.2496 | 23.51905499926943 |
| 4620.642 | interleaved15-06-short | 0 | cmpl-interleaved15-05-long-0-b1d1aa07 | 1 | 0.0 | 0.0 |
| 4645.809 | interleaved15-06-short | 1 | cmpl-interleaved15-05-long-0-b1d1aa07 | 2 | 0.0 | 25.212358001226676 |
| 4670.454 | interleaved15-08-short | 0 | cmpl-interleaved15-05-long-0-b1d1aa07 | 3 | 0.0 | 49.77826800131879 |
| 8721.987 | interleaved15-08-short | 1 | cmpl-interleaved15-07-long-0-ad75dd5c | 1 | 0.0 | 0.0 |
| 13109.668 | interleaved15-12-short | 0 | cmpl-interleaved15-11-long-0-91dd2ae5 | 1 | 0.0 | 0.0 |
| 13135.848 | interleaved15-12-short | 1 | cmpl-interleaved15-11-long-0-91dd2ae5 | 2 | 0.0 | 26.25229300065257 |

所有保护恢复的序号、进度、累计等待与触发原因记录在 `protection-decisions.csv`。保护后每个 long chunk 都可能记录 protected_resume，行数不是独立恢复次数。medium 保持 FIFO 屏障，short 只越过屏障之前的 long；该规则并不保证所有 short 都先完成。

## 完整过程

1. 两种模式各启动一组独立 P/D 与 router，等服务健康并预留 15 秒，再按固定顺序每 100ms 提交一个请求。
2. P 以动态 batch 计算 prompt，baseline 按整请求 push；preemptive 在 KV 就绪后分块入队，在 chunk 边界仲裁，已在途 WRITE 不可中断。
3. D 收齐所需 KV 后首次调度、提交生成，并经 router 流式返回。每请求记录首 token、完整结束和 P/D host 执行累计。
4. 请求结束后执行严格 trace 检查，清理该组服务，下一组重新启动。最后校验两模式输出 token ID、文本 hash 及直接推理参考。

逐请求里程碑见 `request-timeline.csv`；包含每个 host batch、每个 chunk 接受/选择/WRITE start/complete/D receive 的完整时序见 `experiment-timeline.csv`。原始 monotonic 时间与各组相对时间均保留。baseline 未记录逐 chunk KV 时间，不虚构该阶段事件。

本次每种模式只跑一轮，不能建立稳定分位数或普适性能结论。P/D host 执行含调度和结果等待，batch 共享时间不能相加当 GPU 独占时间。batch-invariant 改变了内核路径，新性能数据应在两组相同配置下比较，不直接与旧默认内核实验作因果对比。
