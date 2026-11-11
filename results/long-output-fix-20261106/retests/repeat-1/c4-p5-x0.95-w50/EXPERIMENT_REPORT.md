# 15 请求交错 PD 实验过程与结果

数据集：`workload.sweep-interleaved-8s3m4l.jsonl`，SHA256 `714990980c49534e539a125612ce1c8d36c68ac75a559e4bce93dabcb7ee9e14`。
发送顺序：short / long / short / medium / short / long / short / long / short / medium / short / long / short / medium / short。

每类 prompt 长度：short=128、medium=1107、long=10000 tokens。
batch-invariant 配置：baseline=True, preemptive=True；其余运行参数见 comparison.json。抢占参数 chunk=4、次数=5、进度=95%、累计等待=50ms。

输出一致性通过：True；不经传输的直接推理参考一致：True；综合通过：True。

## 指标均值（ms）

| 类别 | baseline TTFT | preemptive TTFT | baseline E2E | preemptive E2E |
| --- | ---: | ---: | ---: | ---: |
| short | 9817.287 | 7525.988 | 10087.681 | 7778.127 |
| medium | 11637.767 | 10859.179 | 11901.772 | 11120.039 |
| long | 10715.888 | 10223.059 | 11037.805 | 10545.231 |

## 逐请求结果（ms）

| 请求 | baseline TTFT | preemptive TTFT | baseline E2E | preemptive E2E |
| --- | ---: | ---: | ---: | ---: |
| interleaved15-00-short | 443.604 | 454.849 | 712.028 | 723.638 |
| interleaved15-01-long | 5181.264 | 5159.880 | 5507.723 | 5484.137 |
| interleaved15-02-short | 5310.665 | 1265.292 | 5589.035 | 1444.703 |
| interleaved15-03-medium | 5533.870 | 5334.594 | 5796.865 | 5597.378 |
| interleaved15-04-short | 5469.988 | 5422.676 | 5732.249 | 5679.740 |
| interleaved15-05-long | 9022.791 | 8916.572 | 9350.911 | 9244.434 |
| interleaved15-06-short | 9155.130 | 5219.363 | 9434.185 | 5476.610 |
| interleaved15-07-long | 12515.456 | 11796.881 | 12835.540 | 12114.568 |
| interleaved15-08-short | 12641.285 | 8844.813 | 12919.449 | 9124.450 |
| interleaved15-09-medium | 12865.631 | 11969.527 | 13130.730 | 12224.144 |
| interleaved15-10-short | 13030.526 | 11869.117 | 13288.392 | 12123.806 |
| interleaved15-11-long | 16144.043 | 15018.903 | 16457.044 | 15337.784 |
| interleaved15-12-short | 16261.252 | 11770.637 | 16538.593 | 12030.336 |
| interleaved15-13-medium | 16513.801 | 15273.415 | 16777.721 | 15538.596 |
| interleaved15-14-short | 16225.843 | 15361.158 | 16487.513 | 15621.734 |

整轮回放墙钟时间：baseline 18.102s / preemptive 17.049s。均值改善不代表每条请求改善，也不代表吞吐或整轮完成时间改善。

整批请求总时间（首个 HTTP 发出→最后响应结束）：baseline 18.098693s，0.828789 成功请求/秒 / preemptive 17.045271s，0.880009 成功请求/秒。
P/D host 执行累计见 comparison-request-metrics.csv，完整 token ID、文本和 top logprobs 见各组 requests.csv。

## 实际抢占选择

下列每行是一次 arbiter 绕过队首 long、选中一个 short chunk 的决策。次数按 chunk 计数，同一 short 可以多次抢占；dispatch 表示选择，WRITE start/complete 表示实际传输，不把二者混同。时间相对 preemptive 回放起点。

| t(ms) | short | chunk | 被越过的 long | 累计抢占次数 | long 进度 | 累计等待(ms) |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 1276.298 | interleaved15-02-short | 0 | interleaved15-01-long | 1 | 0.1728 | 0.000 |
| 1310.210 | interleaved15-02-short | 1 | interleaved15-01-long | 2 | 0.1792 | 33.908 |
| 5630.728 | interleaved15-06-short | 0 | interleaved15-05-long | 1 | 0.0000 | 0.000 |
| 5652.504 | interleaved15-06-short | 1 | interleaved15-05-long | 2 | 0.0000 | 21.795 |
| 5675.525 | interleaved15-08-short | 0 | interleaved15-05-long | 3 | 0.0000 | 44.767 |
| 9354.149 | interleaved15-08-short | 1 | interleaved15-07-long | 1 | 0.0000 | 0.000 |
| 12803.024 | interleaved15-12-short | 0 | interleaved15-11-long | 1 | 0.0000 | 0.000 |
| 12822.730 | interleaved15-12-short | 1 | interleaved15-11-long | 2 | 0.0000 | 19.778 |

## 每个 long 首次触发保护的边界

| long | 相对时间(ms) | chunk | 进度 | 累计等待(ms) | 原因 |
| --- | ---: | ---: | ---: | ---: | --- |
| interleaved15-01-long | 1341.167 | 28 | 0.1792 | 64.911 | wait |
| interleaved15-05-long | 5697.606 | 0 | 0.0000 | 66.935 | wait |
| interleaved15-07-long | 12301.800 | 150 | 0.9536 | 19.616 | progress |
| interleaved15-11-long | 15903.690 | 150 | 0.9536 | 39.088 | progress |

50ms 在 chunk 边界检查；等待量可以超过 50ms，不能把它解释为整个请求的最大等待或 TTFT 上限。

所有保护恢复的序号、进度、累计等待与触发原因记录在 `protection-decisions.csv`。保护后每个 long chunk 都可能记录 protected_resume，行数不是独立恢复次数。medium 保持 FIFO 屏障，short 只越过屏障之前的 long；该规则并不保证所有 short 都先完成。

## 完整过程

1. 两种模式各启动一组独立 P/D 与 router，等服务健康并预留 15 秒，再按固定顺序每 100ms 提交一个请求。
2. P 以动态 batch 计算 prompt，baseline 按整请求 push；preemptive 在 KV 就绪后分块入队，在 chunk 边界仲裁，已在途 WRITE 不可中断。
3. D 收齐所需 KV 后首次调度、提交生成，并经 router 流式返回。每请求记录首 token、完整结束和 P/D host 执行累计。
4. 请求结束后执行严格 trace 检查，清理该组服务，下一组重新启动。最后校验两模式输出 token ID、文本 hash 及直接推理参考。

逐请求里程碑见 `request-timeline.csv`；包含每个 host batch、每个 chunk 接受/选择/WRITE start/complete/D receive 的完整时序见 `experiment-timeline.csv`。原始 monotonic 时间与各组相对时间均保留。baseline 未记录逐 chunk KV 时间，不虚构该阶段事件。

本次每种模式只跑一轮，不能建立稳定分位数或普适性能结论。P/D host 执行含调度和结果等待，batch 共享时间不能相加当 GPU 独占时间。batch-invariant 改变了内核路径，新性能数据应在两组相同配置下比较，不直接与旧默认内核实验作因果对比。
