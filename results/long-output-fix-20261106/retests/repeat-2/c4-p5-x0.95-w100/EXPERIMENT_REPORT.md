# 15 请求交错 PD 实验过程与结果

数据集：`workload.sweep-interleaved-8s3m4l.jsonl`，SHA256 `714990980c49534e539a125612ce1c8d36c68ac75a559e4bce93dabcb7ee9e14`。
发送顺序：short / long / short / medium / short / long / short / long / short / medium / short / long / short / medium / short。

每类 prompt 长度：short=128、medium=1107、long=10000 tokens。
batch-invariant 配置：baseline=True, preemptive=True；其余运行参数见 comparison.json。抢占参数 chunk=4、次数=5、进度=95%、累计等待=100ms。

输出一致性通过：True；不经传输的直接推理参考一致：True；综合通过：True。

## 指标均值（ms）

| 类别 | baseline TTFT | preemptive TTFT | baseline E2E | preemptive E2E |
| --- | ---: | ---: | ---: | ---: |
| short | 10164.780 | 7536.913 | 10468.944 | 7781.621 |
| medium | 11392.758 | 11847.145 | 11702.110 | 12111.704 |
| long | 11855.911 | 10948.205 | 12162.145 | 11268.828 |

## 逐请求结果（ms）

| 请求 | baseline TTFT | preemptive TTFT | baseline E2E | preemptive E2E |
| --- | ---: | ---: | ---: | ---: |
| interleaved15-00-short | 432.359 | 419.340 | 700.149 | 684.741 |
| interleaved15-01-long | 10084.581 | 4765.294 | 10397.441 | 5088.534 |
| interleaved15-02-short | 9980.025 | 1281.824 | 10297.170 | 1436.806 |
| interleaved15-03-medium | 9882.720 | 4978.456 | 10196.489 | 5244.091 |
| interleaved15-04-short | 9778.430 | 5067.444 | 10095.453 | 5323.416 |
| interleaved15-05-long | 12639.586 | 8973.961 | 12943.200 | 9295.957 |
| interleaved15-06-short | 12542.205 | 4940.390 | 12843.159 | 5193.023 |
| interleaved15-07-long | 12438.935 | 12901.031 | 12743.154 | 13221.209 |
| interleaved15-08-short | 12334.789 | 4739.713 | 12643.135 | 4992.392 |
| interleaved15-09-medium | 12239.329 | 13185.006 | 12545.497 | 13448.340 |
| interleaved15-10-short | 12138.592 | 13273.251 | 12441.930 | 13531.303 |
| interleaved15-11-long | 12260.543 | 17152.534 | 12564.786 | 17469.614 |
| interleaved15-12-short | 12156.294 | 13107.157 | 12466.573 | 13364.273 |
| interleaved15-13-medium | 12056.224 | 17377.972 | 12364.345 | 17642.680 |
| interleaved15-14-short | 11955.548 | 17466.189 | 12263.980 | 17727.012 |

整轮回放墙钟时间：baseline 13.689s / preemptive 19.152s。均值改善不代表每条请求改善，也不代表吞吐或整轮完成时间改善。

整批请求总时间（首个 HTTP 发出→最后响应结束）：baseline 13.685299s，1.096067 成功请求/秒 / preemptive 19.148451s，0.783353 成功请求/秒。
P/D host 执行累计见 comparison-request-metrics.csv，完整 token ID、文本和 top logprobs 见各组 requests.csv。

## 实际抢占选择

下列每行是一次 arbiter 绕过队首 long、选中一个 short chunk 的决策。次数按 chunk 计数，同一 short 可以多次抢占；dispatch 表示选择，WRITE start/complete 表示实际传输，不把二者混同。时间相对 preemptive 回放起点。

| t(ms) | short | chunk | 被越过的 long | 累计抢占次数 | long 进度 | 累计等待(ms) |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 1285.600 | interleaved15-02-short | 0 | interleaved15-01-long | 1 | 0.2432 | 0.000 |
| 1310.008 | interleaved15-02-short | 1 | interleaved15-01-long | 2 | 0.2496 | 24.380 |
| 5285.675 | interleaved15-06-short | 0 | interleaved15-05-long | 1 | 0.0000 | 0.000 |
| 5308.857 | interleaved15-06-short | 1 | interleaved15-05-long | 2 | 0.0000 | 23.254 |
| 5332.387 | interleaved15-08-short | 0 | interleaved15-05-long | 3 | 0.0000 | 46.699 |
| 5356.000 | interleaved15-08-short | 1 | interleaved15-05-long | 4 | 0.0000 | 70.361 |
| 14100.676 | interleaved15-12-short | 0 | interleaved15-11-long | 1 | 0.0000 | 0.000 |
| 14127.026 | interleaved15-12-short | 1 | interleaved15-11-long | 2 | 0.0000 | 26.400 |

## 每个 long 首次触发保护的边界

| long | 相对时间(ms) | chunk | 进度 | 累计等待(ms) | 原因 |
| --- | ---: | ---: | ---: | ---: | --- |
| interleaved15-01-long | 4627.056 | 150 | 0.9536 | 47.233 | progress |
| interleaved15-05-long | 9245.724 | 150 | 0.9536 | 93.357 | progress |
| interleaved15-07-long | 13367.310 | 150 | 0.9536 | 0.000 | progress |
| interleaved15-11-long | 18028.904 | 150 | 0.9536 | 52.125 | progress |

100ms 在 chunk 边界检查；等待量可以超过 100ms，不能把它解释为整个请求的最大等待或 TTFT 上限。

所有保护恢复的序号、进度、累计等待与触发原因记录在 `protection-decisions.csv`。保护后每个 long chunk 都可能记录 protected_resume，行数不是独立恢复次数。medium 保持 FIFO 屏障，short 只越过屏障之前的 long；该规则并不保证所有 short 都先完成。

## 完整过程

1. 两种模式各启动一组独立 P/D 与 router，等服务健康并预留 15 秒，再按固定顺序每 100ms 提交一个请求。
2. P 以动态 batch 计算 prompt，baseline 按整请求 push；preemptive 在 KV 就绪后分块入队，在 chunk 边界仲裁，已在途 WRITE 不可中断。
3. D 收齐所需 KV 后首次调度、提交生成，并经 router 流式返回。每请求记录首 token、完整结束和 P/D host 执行累计。
4. 请求结束后执行严格 trace 检查，清理该组服务，下一组重新启动。最后校验两模式输出 token ID、文本 hash 及直接推理参考。

逐请求里程碑见 `request-timeline.csv`；包含每个 host batch、每个 chunk 接受/选择/WRITE start/complete/D receive 的完整时序见 `experiment-timeline.csv`。原始 monotonic 时间与各组相对时间均保留。baseline 未记录逐 chunk KV 时间，不虚构该阶段事件。

本次每种模式只跑一轮，不能建立稳定分位数或普适性能结论。P/D host 执行含调度和结果等待，batch 共享时间不能相加当 GPU 独占时间。batch-invariant 改变了内核路径，新性能数据应在两组相同配置下比较，不直接与旧默认内核实验作因果对比。
