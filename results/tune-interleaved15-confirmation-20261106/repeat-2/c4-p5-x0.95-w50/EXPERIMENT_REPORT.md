# 15 请求交错 PD 实验过程与结果

数据集：`workload.sweep-interleaved-8s3m4l.jsonl`，SHA256 `714990980c49534e539a125612ce1c8d36c68ac75a559e4bce93dabcb7ee9e14`。
发送顺序：short / long / short / medium / short / long / short / long / short / medium / short / long / short / medium / short。

每类 prompt 长度：short=128、medium=1107、long=10000 tokens。
batch-invariant 配置：baseline=True, preemptive=True；其余运行参数见 comparison.json。抢占参数 chunk=4、次数=5、进度=95%、累计等待=50ms。

输出一致性通过：True；不经传输的直接推理参考一致：True；综合通过：True。

## 指标均值（ms）

| 类别 | baseline TTFT | preemptive TTFT | baseline E2E | preemptive E2E |
| --- | ---: | ---: | ---: | ---: |
| short | 10564.055 | 9204.824 | 10858.817 | 9453.833 |
| medium | 12382.854 | 13610.543 | 12677.848 | 13875.931 |
| long | 11891.124 | 12533.478 | 12189.260 | 12853.243 |

## 逐请求结果（ms）

| 请求 | baseline TTFT | preemptive TTFT | baseline E2E | preemptive E2E |
| --- | ---: | ---: | ---: | ---: |
| interleaved15-00-short | 517.853 | 376.480 | 780.782 | 638.925 |
| interleaved15-01-long | 9978.222 | 5213.433 | 10296.460 | 5535.234 |
| interleaved15-02-short | 9877.370 | 1273.234 | 10196.521 | 1437.146 |
| interleaved15-03-medium | 12755.185 | 5515.315 | 13049.805 | 5780.838 |
| interleaved15-04-short | 12654.526 | 5641.969 | 12949.379 | 5900.234 |
| interleaved15-05-long | 12787.800 | 10260.022 | 13076.572 | 10578.486 |
| interleaved15-06-short | 12684.270 | 5476.924 | 12976.338 | 5732.545 |
| interleaved15-07-long | 12587.364 | 14841.797 | 12873.159 | 15162.883 |
| interleaved15-08-short | 12481.414 | 10181.545 | 12773.174 | 10460.193 |
| interleaved15-09-medium | 12384.134 | 15156.726 | 12673.452 | 15422.262 |
| interleaved15-10-short | 12279.354 | 15283.626 | 12573.605 | 15541.017 |
| interleaved15-11-long | 12211.111 | 19818.661 | 12510.849 | 20136.368 |
| interleaved15-12-short | 12108.885 | 15118.646 | 12410.907 | 15374.129 |
| interleaved15-13-medium | 12009.243 | 20159.589 | 12310.288 | 20424.694 |
| interleaved15-14-short | 11908.767 | 20286.170 | 12209.832 | 20546.473 |

整轮回放墙钟时间：baseline 13.637s / preemptive 21.969s。均值改善不代表每条请求改善，也不代表吞吐或整轮完成时间改善。

整批请求总时间（首个 HTTP 发出→最后响应结束）：baseline 13.633534s，1.100228 成功请求/秒 / preemptive 21.965815s，0.682879 成功请求/秒。
P/D host 执行累计见 comparison-request-metrics.csv，完整 token ID、文本和 top logprobs 见各组 requests.csv。

## 实际抢占选择

下列每行是一次 arbiter 绕过队首 long、选中一个 short chunk 的决策。次数按 chunk 计数，同一 short 可以多次抢占；dispatch 表示选择，WRITE start/complete 表示实际传输，不把二者混同。时间相对 preemptive 回放起点。

| t(ms) | short | chunk | 被越过的 long | 累计抢占次数 | long 进度 | 累计等待(ms) |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 1270.112 | interleaved15-02-short | 0 | interleaved15-01-long | 1 | 0.1856 | 0.000 |
| 1302.791 | interleaved15-02-short | 1 | interleaved15-01-long | 2 | 0.1920 | 32.698 |
| 5831.116 | interleaved15-06-short | 0 | interleaved15-05-long | 1 | 0.0000 | 0.000 |
| 5863.721 | interleaved15-06-short | 1 | interleaved15-05-long | 2 | 0.0000 | 32.686 |
| 10688.939 | interleaved15-08-short | 0 | interleaved15-07-long | 1 | 0.0000 | 0.000 |
| 10718.130 | interleaved15-08-short | 1 | interleaved15-07-long | 2 | 0.0000 | 29.284 |
| 16075.554 | interleaved15-12-short | 0 | interleaved15-11-long | 1 | 0.0000 | 0.000 |
| 16107.223 | interleaved15-12-short | 1 | interleaved15-11-long | 2 | 0.0000 | 31.733 |

## 每个 long 首次触发保护的边界

| long | 相对时间(ms) | chunk | 进度 | 累计等待(ms) | 原因 |
| --- | ---: | ---: | ---: | ---: | --- |
| interleaved15-01-long | 1334.062 | 30 | 0.1920 | 64.006 | wait |
| interleaved15-05-long | 5894.693 | 0 | 0.0000 | 63.706 | wait |
| interleaved15-07-long | 10748.836 | 0 | 0.0000 | 60.035 | wait |
| interleaved15-11-long | 16139.039 | 0 | 0.0000 | 63.580 | wait |

50ms 在 chunk 边界检查；等待量可以超过 50ms，不能把它解释为整个请求的最大等待或 TTFT 上限。

所有保护恢复的序号、进度、累计等待与触发原因记录在 `protection-decisions.csv`。保护后每个 long chunk 都可能记录 protected_resume，行数不是独立恢复次数。medium 保持 FIFO 屏障，short 只越过屏障之前的 long；该规则并不保证所有 short 都先完成。

## 完整过程

1. 两种模式各启动一组独立 P/D 与 router，等服务健康并预留 15 秒，再按固定顺序每 100ms 提交一个请求。
2. P 以动态 batch 计算 prompt，baseline 按整请求 push；preemptive 在 KV 就绪后分块入队，在 chunk 边界仲裁，已在途 WRITE 不可中断。
3. D 收齐所需 KV 后首次调度、提交生成，并经 router 流式返回。每请求记录首 token、完整结束和 P/D host 执行累计。
4. 请求结束后执行严格 trace 检查，清理该组服务，下一组重新启动。最后校验两模式输出 token ID、文本 hash 及直接推理参考。

逐请求里程碑见 `request-timeline.csv`；包含每个 host batch、每个 chunk 接受/选择/WRITE start/complete/D receive 的完整时序见 `experiment-timeline.csv`。原始 monotonic 时间与各组相对时间均保留。baseline 未记录逐 chunk KV 时间，不虚构该阶段事件。

本次每种模式只跑一轮，不能建立稳定分位数或普适性能结论。P/D host 执行含调度和结果等待，batch 共享时间不能相加当 GPU 独占时间。batch-invariant 改变了内核路径，新性能数据应在两组相同配置下比较，不直接与旧默认内核实验作因果对比。
