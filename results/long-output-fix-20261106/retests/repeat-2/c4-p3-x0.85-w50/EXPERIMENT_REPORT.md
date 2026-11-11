# 15 请求交错 PD 实验过程与结果

数据集：`workload.sweep-interleaved-8s3m4l.jsonl`，SHA256 `714990980c49534e539a125612ce1c8d36c68ac75a559e4bce93dabcb7ee9e14`。
发送顺序：short / long / short / medium / short / long / short / long / short / medium / short / long / short / medium / short。

每类 prompt 长度：short=128、medium=1107、long=10000 tokens。
batch-invariant 配置：baseline=True, preemptive=True；其余运行参数见 comparison.json。抢占参数 chunk=4、次数=3、进度=85%、累计等待=50ms。

输出一致性通过：True；不经传输的直接推理参考一致：True；综合通过：True。

## 指标均值（ms）

| 类别 | baseline TTFT | preemptive TTFT | baseline E2E | preemptive E2E |
| --- | ---: | ---: | ---: | ---: |
| short | 10164.780 | 7623.920 | 10468.944 | 7871.139 |
| medium | 11392.758 | 11301.769 | 11702.110 | 11566.611 |
| long | 11855.911 | 10379.465 | 12162.145 | 10700.388 |

## 逐请求结果（ms）

| 请求 | baseline TTFT | preemptive TTFT | baseline E2E | preemptive E2E |
| --- | ---: | ---: | ---: | ---: |
| interleaved15-00-short | 432.359 | 431.111 | 700.149 | 694.182 |
| interleaved15-01-long | 10084.581 | 4274.223 | 10397.441 | 4594.200 |
| interleaved15-02-short | 9980.025 | 1271.356 | 10297.170 | 1421.602 |
| interleaved15-03-medium | 9882.720 | 4464.176 | 10196.489 | 4730.404 |
| interleaved15-04-short | 9778.430 | 4550.798 | 10095.453 | 4809.330 |
| interleaved15-05-long | 12639.586 | 8286.875 | 12943.200 | 8609.488 |
| interleaved15-06-short | 12542.205 | 4426.286 | 12843.159 | 4678.842 |
| interleaved15-07-long | 12438.935 | 12265.007 | 12743.154 | 12585.103 |
| interleaved15-08-short | 12334.789 | 8212.424 | 12643.135 | 8491.643 |
| interleaved15-09-medium | 12239.329 | 12520.582 | 12545.497 | 12784.484 |
| interleaved15-10-short | 12138.592 | 12609.349 | 12441.930 | 12867.344 |
| interleaved15-11-long | 12260.543 | 16691.753 | 12564.786 | 17012.761 |
| interleaved15-12-short | 12156.294 | 12481.154 | 12466.573 | 12736.963 |
| interleaved15-13-medium | 12056.224 | 16920.549 | 12364.345 | 17184.945 |
| interleaved15-14-short | 11955.548 | 17008.884 | 12263.980 | 17269.203 |

整轮回放墙钟时间：baseline 13.689s / preemptive 18.695s。均值改善不代表每条请求改善，也不代表吞吐或整轮完成时间改善。

整批请求总时间（首个 HTTP 发出→最后响应结束）：baseline 13.685299s，1.096067 成功请求/秒 / preemptive 18.690461s，0.802548 成功请求/秒。
P/D host 执行累计见 comparison-request-metrics.csv，完整 token ID、文本和 top logprobs 见各组 requests.csv。

## 实际抢占选择

下列每行是一次 arbiter 绕过队首 long、选中一个 short chunk 的决策。次数按 chunk 计数，同一 short 可以多次抢占；dispatch 表示选择，WRITE start/complete 表示实际传输，不把二者混同。时间相对 preemptive 回放起点。

| t(ms) | short | chunk | 被越过的 long | 累计抢占次数 | long 进度 | 累计等待(ms) |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 1270.908 | interleaved15-02-short | 0 | interleaved15-01-long | 1 | 0.2304 | 0.000 |
| 1296.861 | interleaved15-02-short | 1 | interleaved15-01-long | 2 | 0.2368 | 25.951 |
| 4767.303 | interleaved15-06-short | 0 | interleaved15-05-long | 1 | 0.0000 | 0.000 |
| 4791.765 | interleaved15-06-short | 1 | interleaved15-05-long | 2 | 0.0000 | 24.479 |
| 4815.709 | interleaved15-08-short | 0 | interleaved15-05-long | 3 | 0.0000 | 48.382 |
| 8720.024 | interleaved15-08-short | 1 | interleaved15-07-long | 1 | 0.0000 | 0.000 |
| 13443.028 | interleaved15-12-short | 0 | interleaved15-11-long | 1 | 0.0000 | 0.000 |
| 13472.574 | interleaved15-12-short | 1 | interleaved15-11-long | 2 | 0.0000 | 29.648 |

## 每个 long 首次触发保护的边界

| long | 相对时间(ms) | chunk | 进度 | 累计等待(ms) | 原因 |
| --- | ---: | ---: | ---: | ---: | --- |
| interleaved15-01-long | 3658.420 | 134 | 0.8512 | 49.597 | progress |
| interleaved15-05-long | 4839.364 | 0 | 0.0000 | 72.136 | max_preemptions,wait |
| interleaved15-07-long | 12308.825 | 134 | 0.8512 | 25.516 | progress |
| interleaved15-11-long | 13502.098 | 0 | 0.0000 | 59.236 | wait |

50ms 在 chunk 边界检查；等待量可以超过 50ms，不能把它解释为整个请求的最大等待或 TTFT 上限。

所有保护恢复的序号、进度、累计等待与触发原因记录在 `protection-decisions.csv`。保护后每个 long chunk 都可能记录 protected_resume，行数不是独立恢复次数。medium 保持 FIFO 屏障，short 只越过屏障之前的 long；该规则并不保证所有 short 都先完成。

## 完整过程

1. 两种模式各启动一组独立 P/D 与 router，等服务健康并预留 15 秒，再按固定顺序每 100ms 提交一个请求。
2. P 以动态 batch 计算 prompt，baseline 按整请求 push；preemptive 在 KV 就绪后分块入队，在 chunk 边界仲裁，已在途 WRITE 不可中断。
3. D 收齐所需 KV 后首次调度、提交生成，并经 router 流式返回。每请求记录首 token、完整结束和 P/D host 执行累计。
4. 请求结束后执行严格 trace 检查，清理该组服务，下一组重新启动。最后校验两模式输出 token ID、文本 hash 及直接推理参考。

逐请求里程碑见 `request-timeline.csv`；包含每个 host batch、每个 chunk 接受/选择/WRITE start/complete/D receive 的完整时序见 `experiment-timeline.csv`。原始 monotonic 时间与各组相对时间均保留。baseline 未记录逐 chunk KV 时间，不虚构该阶段事件。

本次每种模式只跑一轮，不能建立稳定分位数或普适性能结论。P/D host 执行含调度和结果等待，batch 共享时间不能相加当 GPU 独占时间。batch-invariant 改变了内核路径，新性能数据应在两组相同配置下比较，不直接与旧默认内核实验作因果对比。
