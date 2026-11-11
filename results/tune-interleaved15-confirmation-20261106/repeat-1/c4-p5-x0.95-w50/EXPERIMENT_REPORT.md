# 15 请求交错 PD 实验过程与结果

数据集：`workload.sweep-interleaved-8s3m4l.jsonl`，SHA256 `714990980c49534e539a125612ce1c8d36c68ac75a559e4bce93dabcb7ee9e14`。
发送顺序：short / long / short / medium / short / long / short / long / short / medium / short / long / short / medium / short。

每类 prompt 长度：short=128、medium=1107、long=10000 tokens。
batch-invariant 配置：baseline=True, preemptive=True；其余运行参数见 comparison.json。抢占参数 chunk=4、次数=5、进度=95%、累计等待=50ms。

输出一致性通过：True；不经传输的直接推理参考一致：True；综合通过：True。

## 指标均值（ms）

| 类别 | baseline TTFT | preemptive TTFT | baseline E2E | preemptive E2E |
| --- | ---: | ---: | ---: | ---: |
| short | 10013.882 | 8301.853 | 10307.502 | 8550.938 |
| medium | 11214.331 | 12308.341 | 11513.513 | 12572.462 |
| long | 11629.510 | 11282.871 | 11934.185 | 11607.298 |

## 逐请求结果（ms）

| 请求 | baseline TTFT | preemptive TTFT | baseline E2E | preemptive E2E |
| --- | ---: | ---: | ---: | ---: |
| interleaved15-00-short | 476.315 | 428.594 | 741.426 | 696.635 |
| interleaved15-01-long | 9779.685 | 4814.106 | 10102.021 | 5139.084 |
| interleaved15-02-short | 9858.757 | 1271.817 | 10143.817 | 1419.537 |
| interleaved15-03-medium | 9756.871 | 5082.740 | 10043.546 | 5345.473 |
| interleaved15-04-short | 9656.422 | 5169.058 | 9942.984 | 5428.445 |
| interleaved15-05-long | 12440.917 | 8963.402 | 12737.933 | 9289.943 |
| interleaved15-06-short | 12340.324 | 5042.155 | 12638.236 | 5298.664 |
| interleaved15-07-long | 12240.433 | 13376.076 | 12536.869 | 13699.463 |
| interleaved15-08-short | 12136.071 | 8892.160 | 12436.754 | 9177.691 |
| interleaved15-09-medium | 12031.831 | 13629.682 | 12337.370 | 13894.343 |
| interleaved15-10-short | 11934.071 | 13719.224 | 12237.534 | 13977.214 |
| interleaved15-11-long | 12057.003 | 17977.902 | 12359.918 | 18300.702 |
| interleaved15-12-short | 11956.461 | 13590.557 | 12259.905 | 13847.487 |
| interleaved15-13-medium | 11854.291 | 18212.602 | 12159.623 | 18477.571 |
| interleaved15-14-short | 11752.637 | 18301.263 | 12059.356 | 18561.829 |

整轮回放墙钟时间：baseline 13.484s / preemptive 19.985s。均值改善不代表每条请求改善，也不代表吞吐或整轮完成时间改善。

整批请求总时间（首个 HTTP 发出→最后响应结束）：baseline 13.480230s，1.112741 成功请求/秒 / preemptive 19.981555s，0.750692 成功请求/秒。
P/D host 执行累计见 comparison-request-metrics.csv，完整 token ID、文本和 top logprobs 见各组 requests.csv。

## 实际抢占选择

下列每行是一次 arbiter 绕过队首 long、选中一个 short chunk 的决策。次数按 chunk 计数，同一 short 可以多次抢占；dispatch 表示选择，WRITE start/complete 表示实际传输，不把二者混同。时间相对 preemptive 回放起点。

| t(ms) | short | chunk | 被越过的 long | 累计抢占次数 | long 进度 | 累计等待(ms) |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 1270.045 | interleaved15-02-short | 0 | interleaved15-01-long | 1 | 0.1920 | 0.000 |
| 1293.551 | interleaved15-02-short | 1 | interleaved15-01-long | 2 | 0.1984 | 23.501 |
| 5394.887 | interleaved15-06-short | 0 | interleaved15-05-long | 1 | 0.0000 | 0.000 |
| 5423.574 | interleaved15-06-short | 1 | interleaved15-05-long | 2 | 0.0000 | 28.693 |
| 9397.225 | interleaved15-08-short | 0 | interleaved15-07-long | 1 | 0.0000 | 0.000 |
| 9426.552 | interleaved15-08-short | 1 | interleaved15-07-long | 2 | 0.0000 | 29.392 |
| 14547.345 | interleaved15-12-short | 0 | interleaved15-11-long | 1 | 0.0000 | 0.000 |
| 14576.200 | interleaved15-12-short | 1 | interleaved15-11-long | 2 | 0.0000 | 28.914 |

## 每个 long 首次触发保护的边界

| long | 相对时间(ms) | chunk | 进度 | 累计等待(ms) | 原因 |
| --- | ---: | ---: | ---: | ---: | --- |
| interleaved15-01-long | 4652.117 | 150 | 0.9536 | 46.284 | progress |
| interleaved15-05-long | 5451.231 | 0 | 0.0000 | 56.412 | wait |
| interleaved15-07-long | 9456.112 | 0 | 0.0000 | 58.995 | wait |
| interleaved15-11-long | 14605.163 | 0 | 0.0000 | 57.905 | wait |

50ms 在 chunk 边界检查；等待量可以超过 50ms，不能把它解释为整个请求的最大等待或 TTFT 上限。

所有保护恢复的序号、进度、累计等待与触发原因记录在 `protection-decisions.csv`。保护后每个 long chunk 都可能记录 protected_resume，行数不是独立恢复次数。medium 保持 FIFO 屏障，short 只越过屏障之前的 long；该规则并不保证所有 short 都先完成。

## 完整过程

1. 两种模式各启动一组独立 P/D 与 router，等服务健康并预留 15 秒，再按固定顺序每 100ms 提交一个请求。
2. P 以动态 batch 计算 prompt，baseline 按整请求 push；preemptive 在 KV 就绪后分块入队，在 chunk 边界仲裁，已在途 WRITE 不可中断。
3. D 收齐所需 KV 后首次调度、提交生成，并经 router 流式返回。每请求记录首 token、完整结束和 P/D host 执行累计。
4. 请求结束后执行严格 trace 检查，清理该组服务，下一组重新启动。最后校验两模式输出 token ID、文本 hash 及直接推理参考。

逐请求里程碑见 `request-timeline.csv`；包含每个 host batch、每个 chunk 接受/选择/WRITE start/complete/D receive 的完整时序见 `experiment-timeline.csv`。原始 monotonic 时间与各组相对时间均保留。baseline 未记录逐 chunk KV 时间，不虚构该阶段事件。

本次每种模式只跑一轮，不能建立稳定分位数或普适性能结论。P/D host 执行含调度和结果等待，batch 共享时间不能相加当 GPU 独占时间。batch-invariant 改变了内核路径，新性能数据应在两组相同配置下比较，不直接与旧默认内核实验作因果对比。
