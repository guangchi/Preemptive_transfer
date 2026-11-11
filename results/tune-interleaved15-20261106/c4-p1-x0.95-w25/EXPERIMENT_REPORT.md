# 15 请求交错 PD 实验过程与结果

数据集：`workload.sweep-interleaved-8s3m4l.jsonl`，SHA256 `714990980c49534e539a125612ce1c8d36c68ac75a559e4bce93dabcb7ee9e14`。
发送顺序：short / long / short / medium / short / long / short / long / short / medium / short / long / short / medium / short。

每类 prompt 长度：short=128、medium=1107、long=10000 tokens。
batch-invariant 配置：baseline=True, preemptive=True；其余运行参数见 comparison.json。抢占参数 chunk=4、次数=1、进度=95%、累计等待=25ms。

输出一致性通过：True；不经传输的直接推理参考一致：True；综合通过：True。

## 指标均值（ms）

| 类别 | baseline TTFT | preemptive TTFT | baseline E2E | preemptive E2E |
| --- | ---: | ---: | ---: | ---: |
| short | 12023.799 | 9788.036 | 12282.129 | 10060.622 |
| medium | 14477.230 | 11638.022 | 14776.925 | 11899.540 |
| long | 14299.549 | 10690.802 | 14614.328 | 11009.989 |

## 逐请求结果（ms）

| 请求 | baseline TTFT | preemptive TTFT | baseline E2E | preemptive E2E |
| --- | ---: | ---: | ---: | ---: |
| interleaved15-00-short | 945.210 | 412.332 | 954.965 | 680.856 |
| interleaved15-01-long | 13782.184 | 4915.682 | 14108.570 | 5231.537 |
| interleaved15-02-short | 13885.915 | 4986.165 | 14189.270 | 5275.382 |
| interleaved15-03-medium | 13782.273 | 5225.172 | 14089.074 | 5487.411 |
| interleaved15-04-short | 13680.805 | 5160.984 | 13992.102 | 5422.806 |
| interleaved15-05-long | 13584.752 | 8856.036 | 13891.621 | 9173.343 |
| interleaved15-06-short | 13479.751 | 8977.599 | 13787.038 | 9256.697 |
| interleaved15-07-long | 13383.290 | 12487.454 | 13687.233 | 12805.178 |
| interleaved15-08-short | 13277.689 | 12611.245 | 13585.142 | 12890.205 |
| interleaved15-09-medium | 13175.591 | 12831.160 | 13484.573 | 13091.971 |
| interleaved15-10-short | 13101.504 | 12765.215 | 13408.276 | 13027.659 |
| interleaved15-11-long | 16447.971 | 16504.038 | 16769.889 | 16829.898 |
| interleaved15-12-short | 13485.545 | 16633.397 | 13745.718 | 16912.380 |
| interleaved15-13-medium | 16473.827 | 16857.734 | 16757.128 | 17119.238 |
| interleaved15-14-short | 14333.975 | 16757.353 | 14594.524 | 17018.994 |

整轮回放墙钟时间：baseline 18.083s / preemptive 18.475s。均值改善不代表每条请求改善，也不代表吞吐或整轮完成时间改善。

整批请求总时间（首个 HTTP 发出→最后响应结束）：baseline 18.079571s，0.829666 成功请求/秒 / preemptive 18.440000s，0.813449 成功请求/秒。
P/D host 执行累计见 comparison-request-metrics.csv，完整 token ID、文本和 top logprobs 见各组 requests.csv。

## 实际抢占选择

下列每行是一次 arbiter 绕过队首 long、选中一个 short chunk 的决策。次数按 chunk 计数，同一 short 可以多次抢占；dispatch 表示选择，WRITE start/complete 表示实际传输，不把二者混同。时间相对 preemptive 回放起点。

| t(ms) | short | chunk | 被越过的 long | 累计抢占次数 | long 进度 | 累计等待(ms) |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 1281.604 | interleaved15-02-short | 0 | interleaved15-01-long | 1 | 0.1920 | 0.000 |
| 5429.986 | interleaved15-06-short | 0 | interleaved15-05-long | 1 | 0.0000 | 0.000 |
| 9313.706 | interleaved15-08-short | 0 | interleaved15-07-long | 1 | 0.0000 | 0.000 |
| 13608.517 | interleaved15-12-short | 0 | interleaved15-11-long | 1 | 0.0000 | 0.000 |

## 每个 long 首次触发保护的边界

| long | 相对时间(ms) | chunk | 进度 | 累计等待(ms) | 原因 |
| --- | ---: | ---: | ---: | ---: | --- |
| interleaved15-01-long | 1311.414 | 31 | 0.1984 | 29.858 | max_preemptions,wait |
| interleaved15-05-long | 5453.265 | 0 | 0.0000 | 23.365 | max_preemptions |
| interleaved15-07-long | 9337.665 | 0 | 0.0000 | 24.071 | max_preemptions |
| interleaved15-11-long | 13633.082 | 0 | 0.0000 | 24.692 | max_preemptions |

25ms 在 chunk 边界检查；等待量可以超过 25ms，不能把它解释为整个请求的最大等待或 TTFT 上限。

所有保护恢复的序号、进度、累计等待与触发原因记录在 `protection-decisions.csv`。保护后每个 long chunk 都可能记录 protected_resume，行数不是独立恢复次数。medium 保持 FIFO 屏障，short 只越过屏障之前的 long；该规则并不保证所有 short 都先完成。

## 完整过程

1. 两种模式各启动一组独立 P/D 与 router，等服务健康并预留 15 秒，再按固定顺序每 100ms 提交一个请求。
2. P 以动态 batch 计算 prompt，baseline 按整请求 push；preemptive 在 KV 就绪后分块入队，在 chunk 边界仲裁，已在途 WRITE 不可中断。
3. D 收齐所需 KV 后首次调度、提交生成，并经 router 流式返回。每请求记录首 token、完整结束和 P/D host 执行累计。
4. 请求结束后执行严格 trace 检查，清理该组服务，下一组重新启动。最后校验两模式输出 token ID、文本 hash 及直接推理参考。

逐请求里程碑见 `request-timeline.csv`；包含每个 host batch、每个 chunk 接受/选择/WRITE start/complete/D receive 的完整时序见 `experiment-timeline.csv`。原始 monotonic 时间与各组相对时间均保留。baseline 未记录逐 chunk KV 时间，不虚构该阶段事件。

本次每种模式只跑一轮，不能建立稳定分位数或普适性能结论。P/D host 执行含调度和结果等待，batch 共享时间不能相加当 GPU 独占时间。batch-invariant 改变了内核路径，新性能数据应在两组相同配置下比较，不直接与旧默认内核实验作因果对比。
