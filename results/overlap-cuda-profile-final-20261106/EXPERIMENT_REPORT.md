# 15 请求交错 PD 实验过程与结果

数据集：`workload.sweep-interleaved-8s3m4l.jsonl`，SHA256 `714990980c49534e539a125612ce1c8d36c68ac75a559e4bce93dabcb7ee9e14`。
发送顺序：short / long / short / medium / short / long / short / long / short / medium / short / long / short / medium / short。

每类 prompt 长度：short=128、medium=1107、long=10000 tokens。
batch-invariant 配置：baseline=True, preemptive=True；其余运行参数见 comparison.json。抢占参数 chunk=4、次数=3、进度=85%、累计等待=50ms。

输出一致性通过：True；不经传输的直接推理参考一致：None；综合通过：True。

## 指标均值（ms）

| 类别 | baseline TTFT | preemptive TTFT | baseline E2E | preemptive E2E |
| --- | ---: | ---: | ---: | ---: |
| short | 12133.849 | 10913.114 | 12414.492 | 11174.298 |
| medium | 13921.899 | 15699.924 | 14197.894 | 15964.440 |
| long | 13349.773 | 14265.794 | 13640.002 | 14586.240 |

## 逐请求结果（ms）

| 请求 | baseline TTFT | preemptive TTFT | baseline E2E | preemptive E2E |
| --- | ---: | ---: | ---: | ---: |
| interleaved15-00-short | 638.276 | 427.391 | 901.441 | 692.030 |
| interleaved15-01-long | 11918.843 | 6266.249 | 12239.298 | 6590.086 |
| interleaved15-02-short | 14108.913 | 1204.170 | 14410.397 | 1464.067 |
| interleaved15-03-medium | 14291.930 | 8602.543 | 14575.414 | 8867.794 |
| interleaved15-04-short | 14190.407 | 8692.201 | 14476.112 | 8950.501 |
| interleaved15-05-long | 14092.577 | 12781.062 | 14372.432 | 13100.532 |
| interleaved15-06-short | 13993.563 | 8565.101 | 14276.259 | 8822.003 |
| interleaved15-07-long | 13894.010 | 16844.205 | 14177.318 | 17169.031 |
| interleaved15-08-short | 13793.195 | 12706.710 | 14071.266 | 12981.570 |
| interleaved15-09-medium | 13692.975 | 17077.383 | 13973.401 | 17341.724 |
| interleaved15-10-short | 13593.185 | 17165.008 | 13873.412 | 17423.973 |
| interleaved15-11-long | 13493.664 | 21171.660 | 13770.960 | 21485.311 |
| interleaved15-12-short | 13388.524 | 17037.349 | 13667.970 | 17292.963 |
| interleaved15-13-medium | 13780.792 | 21419.846 | 14044.867 | 21683.802 |
| interleaved15-14-short | 13364.726 | 21506.979 | 13639.078 | 21767.278 |

整轮回放墙钟时间：baseline 15.372s / preemptive 23.192s。均值改善不代表每条请求改善，也不代表吞吐或整轮完成时间改善。

整批请求总时间（首个 HTTP 发出→最后响应结束）：baseline 15.367487s，0.976087 成功请求/秒 / preemptive 23.187827s，0.646891 成功请求/秒。
P/D host 执行累计见 comparison-request-metrics.csv，完整 token ID、文本和 top logprobs 见各组 requests.csv。

## 实际抢占选择

下列每行是一次 arbiter 绕过队首 long、选中一个 short chunk 的决策。次数按 chunk 计数，同一 short 可以多次抢占；dispatch 表示选择，WRITE start/complete 表示实际传输，不把二者混同。时间相对 preemptive 回放起点。

| t(ms) | short | chunk | 被越过的 long | 累计抢占次数 | long 进度 | 累计等待(ms) |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 1286.427 | interleaved15-02-short | 0 | interleaved15-01-long | 1 | 0.1472 | 0.000 |
| 1324.730 | interleaved15-02-short | 1 | interleaved15-01-long | 2 | 0.1536 | 38.304 |
| 8909.926 | interleaved15-06-short | 0 | interleaved15-05-long | 1 | 0.0000 | 0.000 |
| 8937.391 | interleaved15-06-short | 1 | interleaved15-05-long | 2 | 0.0000 | 27.504 |
| 13211.545 | interleaved15-08-short | 0 | interleaved15-07-long | 1 | 0.0000 | 0.000 |
| 13236.501 | interleaved15-08-short | 1 | interleaved15-07-long | 2 | 0.0000 | 25.008 |
| 17992.884 | interleaved15-12-short | 0 | interleaved15-11-long | 1 | 0.0000 | 0.000 |
| 18019.497 | interleaved15-12-short | 1 | interleaved15-11-long | 2 | 0.0000 | 26.653 |

## 每个 long 首次触发保护的边界

| long | 相对时间(ms) | chunk | 进度 | 累计等待(ms) | 原因 |
| --- | ---: | ---: | ---: | ---: | --- |
| interleaved15-01-long | 1360.654 | 24 | 0.1536 | 74.270 | wait |
| interleaved15-05-long | 8964.265 | 0 | 0.0000 | 54.426 | wait |
| interleaved15-07-long | 13262.571 | 0 | 0.0000 | 51.120 | wait |
| interleaved15-11-long | 18044.643 | 0 | 0.0000 | 51.832 | wait |

50ms 在 chunk 边界检查；等待量可以超过 50ms，不能把它解释为整个请求的最大等待或 TTFT 上限。

所有保护恢复的序号、进度、累计等待与触发原因记录在 `protection-decisions.csv`。保护后每个 long chunk 都可能记录 protected_resume，行数不是独立恢复次数。medium 保持 FIFO 屏障，short 只越过屏障之前的 long；该规则并不保证所有 short 都先完成。

## 完整过程

1. 两种模式各启动一组独立 P/D 与 router，等服务健康并预留 15 秒，再按固定顺序每 100ms 提交一个请求。
2. P 以动态 batch 计算 prompt，baseline 按整请求 push；preemptive 在 KV 就绪后分块入队，在 chunk 边界仲裁，已在途 WRITE 不可中断。
3. D 收齐所需 KV 后首次调度、提交生成，并经 router 流式返回。每请求记录首 token、完整结束和 P/D host 执行累计。
4. 请求结束后执行严格 trace 检查，清理该组服务，下一组重新启动。最后校验两模式输出 token ID、文本 hash 及直接推理参考。

逐请求里程碑见 `request-timeline.csv`；包含每个 host batch、每个 chunk 接受/选择/WRITE start/complete/D receive 的完整时序见 `experiment-timeline.csv`。原始 monotonic 时间与各组相对时间均保留。baseline 未记录逐 chunk KV 时间，不虚构该阶段事件。

本次每种模式只跑一轮，不能建立稳定分位数或普适性能结论。P/D host 执行含调度和结果等待，batch 共享时间不能相加当 GPU 独占时间。batch-invariant 改变了内核路径，新性能数据应在两组相同配置下比较，不直接与旧默认内核实验作因果对比。
