# 15 请求交错 PD 实验过程与结果

数据集：`workload.sweep-interleaved-8s3m4l.jsonl`，SHA256 `714990980c49534e539a125612ce1c8d36c68ac75a559e4bce93dabcb7ee9e14`。
发送顺序：short / long / short / medium / short / long / short / long / short / medium / short / long / short / medium / short。

每类 prompt 长度：short=128、medium=1107、long=10000 tokens。
batch-invariant 配置：baseline=True, preemptive=True；其余运行参数见 comparison.json。抢占参数 chunk=4、次数=5、进度=95%、累计等待=50ms。

输出一致性通过：True；不经传输的直接推理参考一致：True；综合通过：True。

## 指标均值（ms）

| 类别 | baseline TTFT | preemptive TTFT | baseline E2E | preemptive E2E |
| --- | ---: | ---: | ---: | ---: |
| short | 10908.817 | 6895.754 | 11205.672 | 7147.567 |
| medium | 12716.374 | 10120.506 | 13012.283 | 10384.868 |
| long | 12245.033 | 9340.169 | 12546.925 | 9660.942 |

## 逐请求结果（ms）

| 请求 | baseline TTFT | preemptive TTFT | baseline E2E | preemptive E2E |
| --- | ---: | ---: | ---: | ---: |
| interleaved15-00-short | 623.889 | 423.520 | 891.782 | 687.725 |
| interleaved15-01-long | 10401.769 | 4186.774 | 10719.960 | 4505.668 |
| interleaved15-02-short | 10301.544 | 1270.638 | 10619.697 | 1449.994 |
| interleaved15-03-medium | 13091.511 | 4371.132 | 13382.457 | 4634.667 |
| interleaved15-04-short | 13214.639 | 4459.861 | 13513.161 | 4716.950 |
| interleaved15-05-long | 13117.959 | 7552.465 | 13411.893 | 7872.949 |
| interleaved15-06-short | 13020.601 | 4255.920 | 13311.466 | 4513.666 |
| interleaved15-07-long | 12917.381 | 10979.685 | 13209.206 | 11300.512 |
| interleaved15-08-short | 12816.610 | 7476.223 | 13108.859 | 7755.732 |
| interleaved15-09-medium | 12716.734 | 11152.204 | 13008.713 | 11416.066 |
| interleaved15-10-short | 12613.631 | 11240.272 | 12909.000 | 11499.717 |
| interleaved15-11-long | 12543.023 | 14641.752 | 12846.643 | 14964.641 |
| interleaved15-12-short | 12442.322 | 11112.280 | 12746.238 | 11368.865 |
| interleaved15-13-medium | 12340.877 | 14838.181 | 12645.678 | 15103.871 |
| interleaved15-14-short | 12237.304 | 14927.317 | 12545.174 | 15187.885 |

整轮回放墙钟时间：baseline 13.971s / preemptive 16.613s。均值改善不代表每条请求改善，也不代表吞吐或整轮完成时间改善。

整批请求总时间（首个 HTTP 发出→最后响应结束）：baseline 13.966988s，1.073961 成功请求/秒 / preemptive 16.609126s，0.903118 成功请求/秒。
P/D host 执行累计见 comparison-request-metrics.csv，完整 token ID、文本和 top logprobs 见各组 requests.csv。

## 实际抢占选择

下列每行是一次 arbiter 绕过队首 long、选中一个 short chunk 的决策。次数按 chunk 计数，同一 short 可以多次抢占；dispatch 表示选择，WRITE start/complete 表示实际传输，不把二者混同。时间相对 preemptive 回放起点。

| t(ms) | short | chunk | 被越过的 long | 累计抢占次数 | long 进度 | 累计等待(ms) |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 1293.782 | interleaved15-02-short | 0 | interleaved15-01-long | 1 | 0.1920 | 0.000 |
| 1323.683 | interleaved15-02-short | 1 | interleaved15-01-long | 2 | 0.1984 | 29.926 |
| 4672.936 | interleaved15-06-short | 0 | interleaved15-05-long | 1 | 0.0000 | 0.000 |
| 4693.297 | interleaved15-06-short | 1 | interleaved15-05-long | 2 | 0.0000 | 20.443 |
| 4713.324 | interleaved15-08-short | 0 | interleaved15-05-long | 3 | 0.0000 | 40.408 |
| 7987.430 | interleaved15-08-short | 1 | interleaved15-07-long | 1 | 0.0000 | 0.000 |
| 12059.258 | interleaved15-12-short | 0 | interleaved15-11-long | 1 | 0.0000 | 0.000 |
| 12083.750 | interleaved15-12-short | 1 | interleaved15-11-long | 2 | 0.0000 | 24.582 |

## 每个 long 首次触发保护的边界

| long | 相对时间(ms) | chunk | 进度 | 累计等待(ms) | 原因 |
| --- | ---: | ---: | ---: | ---: | --- |
| interleaved15-01-long | 1348.392 | 31 | 0.1984 | 54.705 | wait |
| interleaved15-05-long | 4734.200 | 0 | 0.0000 | 61.381 | wait |
| interleaved15-07-long | 11477.576 | 150 | 0.9536 | 25.416 | progress |
| interleaved15-11-long | 12109.342 | 0 | 0.0000 | 50.205 | wait |

50ms 在 chunk 边界检查；等待量可以超过 50ms，不能把它解释为整个请求的最大等待或 TTFT 上限。

所有保护恢复的序号、进度、累计等待与触发原因记录在 `protection-decisions.csv`。保护后每个 long chunk 都可能记录 protected_resume，行数不是独立恢复次数。medium 保持 FIFO 屏障，short 只越过屏障之前的 long；该规则并不保证所有 short 都先完成。

## 完整过程

1. 两种模式各启动一组独立 P/D 与 router，等服务健康并预留 15 秒，再按固定顺序每 100ms 提交一个请求。
2. P 以动态 batch 计算 prompt，baseline 按整请求 push；preemptive 在 KV 就绪后分块入队，在 chunk 边界仲裁，已在途 WRITE 不可中断。
3. D 收齐所需 KV 后首次调度、提交生成，并经 router 流式返回。每请求记录首 token、完整结束和 P/D host 执行累计。
4. 请求结束后执行严格 trace 检查，清理该组服务，下一组重新启动。最后校验两模式输出 token ID、文本 hash 及直接推理参考。

逐请求里程碑见 `request-timeline.csv`；包含每个 host batch、每个 chunk 接受/选择/WRITE start/complete/D receive 的完整时序见 `experiment-timeline.csv`。原始 monotonic 时间与各组相对时间均保留。baseline 未记录逐 chunk KV 时间，不虚构该阶段事件。

本次每种模式只跑一轮，不能建立稳定分位数或普适性能结论。P/D host 执行含调度和结果等待，batch 共享时间不能相加当 GPU 独占时间。batch-invariant 改变了内核路径，新性能数据应在两组相同配置下比较，不直接与旧默认内核实验作因果对比。
