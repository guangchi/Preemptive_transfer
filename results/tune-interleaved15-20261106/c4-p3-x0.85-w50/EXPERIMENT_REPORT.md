# 15 请求交错 PD 实验过程与结果

数据集：`workload.sweep-interleaved-8s3m4l.jsonl`，SHA256 `714990980c49534e539a125612ce1c8d36c68ac75a559e4bce93dabcb7ee9e14`。
发送顺序：short / long / short / medium / short / long / short / long / short / medium / short / long / short / medium / short。

每类 prompt 长度：short=128、medium=1107、long=10000 tokens。
batch-invariant 配置：baseline=True, preemptive=True；其余运行参数见 comparison.json。抢占参数 chunk=4、次数=3、进度=85%、累计等待=50ms。

输出一致性通过：True；不经传输的直接推理参考一致：True；综合通过：True。

## 指标均值（ms）

| 类别 | baseline TTFT | preemptive TTFT | baseline E2E | preemptive E2E |
| --- | ---: | ---: | ---: | ---: |
| short | 12023.799 | 7653.786 | 12282.129 | 7905.995 |
| medium | 14477.230 | 11628.086 | 14776.925 | 11892.187 |
| long | 14299.549 | 10443.465 | 14614.328 | 10765.365 |

## 逐请求结果（ms）

| 请求 | baseline TTFT | preemptive TTFT | baseline E2E | preemptive E2E |
| --- | ---: | ---: | ---: | ---: |
| interleaved15-00-short | 945.210 | 423.942 | 954.965 | 686.034 |
| interleaved15-01-long | 13782.184 | 4183.843 | 14108.570 | 4507.554 |
| interleaved15-02-short | 13885.915 | 1267.092 | 14189.270 | 1452.150 |
| interleaved15-03-medium | 13782.273 | 4358.018 | 14089.074 | 4621.504 |
| interleaved15-04-short | 13680.805 | 4446.670 | 13992.102 | 4703.817 |
| interleaved15-05-long | 13584.752 | 7614.876 | 13891.621 | 7940.768 |
| interleaved15-06-short | 13479.751 | 4243.439 | 13787.038 | 4500.795 |
| interleaved15-07-long | 13383.290 | 12441.501 | 13687.233 | 12761.822 |
| interleaved15-08-short | 13277.689 | 7543.570 | 13585.142 | 7823.443 |
| interleaved15-09-medium | 13175.591 | 12641.765 | 13484.573 | 12906.108 |
| interleaved15-10-short | 13101.504 | 12730.100 | 13408.276 | 12988.905 |
| interleaved15-11-long | 16447.971 | 17533.639 | 16769.889 | 17851.315 |
| interleaved15-12-short | 13485.545 | 12602.585 | 13745.718 | 12858.773 |
| interleaved15-13-medium | 16473.827 | 17884.476 | 16757.128 | 18148.948 |
| interleaved15-14-short | 14333.975 | 17972.891 | 14594.524 | 18234.044 |

整轮回放墙钟时间：baseline 18.083s / preemptive 19.659s。均值改善不代表每条请求改善，也不代表吞吐或整轮完成时间改善。

整批请求总时间（首个 HTTP 发出→最后响应结束）：baseline 18.079571s，0.829666 成功请求/秒 / preemptive 19.655122s，0.763160 成功请求/秒。
P/D host 执行累计见 comparison-request-metrics.csv，完整 token ID、文本和 top logprobs 见各组 requests.csv。

## 实际抢占选择

下列每行是一次 arbiter 绕过队首 long、选中一个 short chunk 的决策。次数按 chunk 计数，同一 short 可以多次抢占；dispatch 表示选择，WRITE start/complete 表示实际传输，不把二者混同。时间相对 preemptive 回放起点。

| t(ms) | short | chunk | 被越过的 long | 累计抢占次数 | long 进度 | 累计等待(ms) |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 1286.376 | interleaved15-02-short | 0 | interleaved15-01-long | 1 | 0.1920 | 0.000 |
| 1319.301 | interleaved15-02-short | 1 | interleaved15-01-long | 2 | 0.1984 | 32.933 |
| 4644.178 | interleaved15-06-short | 0 | interleaved15-05-long | 1 | 0.0000 | 0.000 |
| 4664.962 | interleaved15-06-short | 1 | interleaved15-05-long | 2 | 0.0000 | 20.903 |
| 4685.583 | interleaved15-08-short | 0 | interleaved15-05-long | 3 | 0.0000 | 41.429 |
| 8049.576 | interleaved15-08-short | 1 | interleaved15-07-long | 1 | 0.0000 | 0.000 |
| 13554.526 | interleaved15-12-short | 0 | interleaved15-11-long | 1 | 0.0000 | 0.000 |
| 13580.386 | interleaved15-12-short | 1 | interleaved15-11-long | 2 | 0.0000 | 25.921 |

## 每个 long 首次触发保护的边界

| long | 相对时间(ms) | chunk | 进度 | 累计等待(ms) | 原因 |
| --- | ---: | ---: | ---: | ---: | --- |
| interleaved15-01-long | 1349.263 | 31 | 0.1984 | 62.933 | wait |
| interleaved15-05-long | 4706.173 | 0 | 0.0000 | 62.140 | max_preemptions,wait |
| interleaved15-07-long | 12414.714 | 134 | 0.8512 | 31.715 | progress |
| interleaved15-11-long | 13605.411 | 0 | 0.0000 | 50.975 | wait |

50ms 在 chunk 边界检查；等待量可以超过 50ms，不能把它解释为整个请求的最大等待或 TTFT 上限。

所有保护恢复的序号、进度、累计等待与触发原因记录在 `protection-decisions.csv`。保护后每个 long chunk 都可能记录 protected_resume，行数不是独立恢复次数。medium 保持 FIFO 屏障，short 只越过屏障之前的 long；该规则并不保证所有 short 都先完成。

## 完整过程

1. 两种模式各启动一组独立 P/D 与 router，等服务健康并预留 15 秒，再按固定顺序每 100ms 提交一个请求。
2. P 以动态 batch 计算 prompt，baseline 按整请求 push；preemptive 在 KV 就绪后分块入队，在 chunk 边界仲裁，已在途 WRITE 不可中断。
3. D 收齐所需 KV 后首次调度、提交生成，并经 router 流式返回。每请求记录首 token、完整结束和 P/D host 执行累计。
4. 请求结束后执行严格 trace 检查，清理该组服务，下一组重新启动。最后校验两模式输出 token ID、文本 hash 及直接推理参考。

逐请求里程碑见 `request-timeline.csv`；包含每个 host batch、每个 chunk 接受/选择/WRITE start/complete/D receive 的完整时序见 `experiment-timeline.csv`。原始 monotonic 时间与各组相对时间均保留。baseline 未记录逐 chunk KV 时间，不虚构该阶段事件。

本次每种模式只跑一轮，不能建立稳定分位数或普适性能结论。P/D host 执行含调度和结果等待，batch 共享时间不能相加当 GPU 独占时间。batch-invariant 改变了内核路径，新性能数据应在两组相同配置下比较，不直接与旧默认内核实验作因果对比。
