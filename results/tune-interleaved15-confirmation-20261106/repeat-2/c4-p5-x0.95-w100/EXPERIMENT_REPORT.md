# 15 请求交错 PD 实验过程与结果

数据集：`workload.sweep-interleaved-8s3m4l.jsonl`，SHA256 `714990980c49534e539a125612ce1c8d36c68ac75a559e4bce93dabcb7ee9e14`。
发送顺序：short / long / short / medium / short / long / short / long / short / medium / short / long / short / medium / short。

每类 prompt 长度：short=128、medium=1107、long=10000 tokens。
batch-invariant 配置：baseline=True, preemptive=True；其余运行参数见 comparison.json。抢占参数 chunk=4、次数=5、进度=95%、累计等待=100ms。

输出一致性通过：True；不经传输的直接推理参考一致：True；综合通过：True。

## 指标均值（ms）

| 类别 | baseline TTFT | preemptive TTFT | baseline E2E | preemptive E2E |
| --- | ---: | ---: | ---: | ---: |
| short | 10564.055 | 7247.704 | 10858.817 | 7494.314 |
| medium | 12382.854 | 11366.539 | 12677.848 | 11630.161 |
| long | 11891.124 | 10529.700 | 12189.260 | 10850.207 |

## 逐请求结果（ms）

| 请求 | baseline TTFT | preemptive TTFT | baseline E2E | preemptive E2E |
| --- | ---: | ---: | ---: | ---: |
| interleaved15-00-short | 517.853 | 432.430 | 780.782 | 697.102 |
| interleaved15-01-long | 9978.222 | 4591.303 | 10296.460 | 4912.336 |
| interleaved15-02-short | 9877.370 | 1259.034 | 10196.521 | 1430.971 |
| interleaved15-03-medium | 12755.185 | 4828.100 | 13049.805 | 5092.194 |
| interleaved15-04-short | 12654.526 | 4916.469 | 12949.379 | 5171.975 |
| interleaved15-05-long | 12787.800 | 8699.893 | 13076.572 | 9023.533 |
| interleaved15-06-short | 12684.270 | 4788.075 | 12976.338 | 5041.058 |
| interleaved15-07-long | 12587.364 | 12385.597 | 12873.159 | 12706.789 |
| interleaved15-08-short | 12481.414 | 4587.043 | 12773.174 | 4840.404 |
| interleaved15-09-medium | 12384.134 | 12591.851 | 12673.452 | 12855.237 |
| interleaved15-10-short | 12279.354 | 12680.034 | 12573.605 | 12938.038 |
| interleaved15-11-long | 12211.111 | 16442.005 | 12510.849 | 16758.171 |
| interleaved15-12-short | 12108.885 | 12551.402 | 12410.907 | 12807.904 |
| interleaved15-13-medium | 12009.243 | 16679.665 | 12310.288 | 16943.051 |
| interleaved15-14-short | 11908.767 | 16767.143 | 12209.832 | 17027.063 |

整轮回放墙钟时间：baseline 13.637s / preemptive 18.453s。均值改善不代表每条请求改善，也不代表吞吐或整轮完成时间改善。

整批请求总时间（首个 HTTP 发出→最后响应结束）：baseline 13.633534s，1.100228 成功请求/秒 / preemptive 18.449064s，0.813049 成功请求/秒。
P/D host 执行累计见 comparison-request-metrics.csv，完整 token ID、文本和 top logprobs 见各组 requests.csv。

## 实际抢占选择

下列每行是一次 arbiter 绕过队首 long、选中一个 short chunk 的决策。次数按 chunk 计数，同一 short 可以多次抢占；dispatch 表示选择，WRITE start/complete 表示实际传输，不把二者混同。时间相对 preemptive 回放起点。

| t(ms) | short | chunk | 被越过的 long | 累计抢占次数 | long 进度 | 累计等待(ms) |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 1278.431 | interleaved15-02-short | 0 | interleaved15-01-long | 1 | 0.2240 | 0.000 |
| 1303.444 | interleaved15-02-short | 1 | interleaved15-01-long | 2 | 0.2304 | 25.025 |
| 5138.779 | interleaved15-06-short | 0 | interleaved15-05-long | 1 | 0.0000 | 0.000 |
| 5171.724 | interleaved15-06-short | 1 | interleaved15-05-long | 2 | 0.0000 | 33.046 |
| 5202.826 | interleaved15-08-short | 0 | interleaved15-05-long | 3 | 0.0000 | 64.043 |
| 5227.025 | interleaved15-08-short | 1 | interleaved15-05-long | 4 | 0.0000 | 88.347 |
| 13505.207 | interleaved15-12-short | 0 | interleaved15-11-long | 1 | 0.0000 | 0.000 |
| 13529.991 | interleaved15-12-short | 1 | interleaved15-11-long | 2 | 0.0000 | 24.878 |

## 每个 long 首次触发保护的边界

| long | 相对时间(ms) | chunk | 进度 | 累计等待(ms) | 原因 |
| --- | ---: | ---: | ---: | ---: | --- |
| interleaved15-01-long | 4448.275 | 150 | 0.9536 | 49.948 | progress |
| interleaved15-05-long | 5249.489 | 0 | 0.0000 | 110.920 | wait |
| interleaved15-07-long | 12861.748 | 150 | 0.9536 | 0.000 | progress |
| interleaved15-11-long | 17319.056 | 150 | 0.9536 | 50.123 | progress |

100ms 在 chunk 边界检查；等待量可以超过 100ms，不能把它解释为整个请求的最大等待或 TTFT 上限。

所有保护恢复的序号、进度、累计等待与触发原因记录在 `protection-decisions.csv`。保护后每个 long chunk 都可能记录 protected_resume，行数不是独立恢复次数。medium 保持 FIFO 屏障，short 只越过屏障之前的 long；该规则并不保证所有 short 都先完成。

## 完整过程

1. 两种模式各启动一组独立 P/D 与 router，等服务健康并预留 15 秒，再按固定顺序每 100ms 提交一个请求。
2. P 以动态 batch 计算 prompt，baseline 按整请求 push；preemptive 在 KV 就绪后分块入队，在 chunk 边界仲裁，已在途 WRITE 不可中断。
3. D 收齐所需 KV 后首次调度、提交生成，并经 router 流式返回。每请求记录首 token、完整结束和 P/D host 执行累计。
4. 请求结束后执行严格 trace 检查，清理该组服务，下一组重新启动。最后校验两模式输出 token ID、文本 hash 及直接推理参考。

逐请求里程碑见 `request-timeline.csv`；包含每个 host batch、每个 chunk 接受/选择/WRITE start/complete/D receive 的完整时序见 `experiment-timeline.csv`。原始 monotonic 时间与各组相对时间均保留。baseline 未记录逐 chunk KV 时间，不虚构该阶段事件。

本次每种模式只跑一轮，不能建立稳定分位数或普适性能结论。P/D host 执行含调度和结果等待，batch 共享时间不能相加当 GPU 独占时间。batch-invariant 改变了内核路径，新性能数据应在两组相同配置下比较，不直接与旧默认内核实验作因果对比。
