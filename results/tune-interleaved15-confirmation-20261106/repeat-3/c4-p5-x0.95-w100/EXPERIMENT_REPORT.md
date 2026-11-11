# 15 请求交错 PD 实验过程与结果

数据集：`workload.sweep-interleaved-8s3m4l.jsonl`，SHA256 `714990980c49534e539a125612ce1c8d36c68ac75a559e4bce93dabcb7ee9e14`。
发送顺序：short / long / short / medium / short / long / short / long / short / medium / short / long / short / medium / short。

每类 prompt 长度：short=128、medium=1107、long=10000 tokens。
batch-invariant 配置：baseline=True, preemptive=True；其余运行参数见 comparison.json。抢占参数 chunk=4、次数=5、进度=95%、累计等待=100ms。

输出一致性通过：True；不经传输的直接推理参考一致：True；综合通过：True。

## 指标均值（ms）

| 类别 | baseline TTFT | preemptive TTFT | baseline E2E | preemptive E2E |
| --- | ---: | ---: | ---: | ---: |
| short | 13045.954 | 7380.374 | 13327.111 | 7625.226 |
| medium | 13417.496 | 11582.211 | 13692.919 | 11847.271 |
| long | 14861.788 | 10716.258 | 15145.228 | 11037.899 |

## 逐请求结果（ms）

| 请求 | baseline TTFT | preemptive TTFT | baseline E2E | preemptive E2E |
| --- | ---: | ---: | ---: | ---: |
| interleaved15-00-short | 435.055 | 414.752 | 695.864 | 682.846 |
| interleaved15-01-long | 4971.615 | 4620.613 | 5285.340 | 4941.864 |
| interleaved15-02-short | 4871.101 | 1264.663 | 5184.919 | 1418.393 |
| interleaved15-03-medium | 4993.400 | 4843.504 | 5271.828 | 5108.207 |
| interleaved15-04-short | 9473.140 | 4932.472 | 9740.761 | 5188.242 |
| interleaved15-05-long | 18422.079 | 8833.787 | 18699.741 | 9154.170 |
| interleaved15-06-short | 18322.329 | 4804.812 | 18599.606 | 5058.263 |
| interleaved15-07-long | 18228.996 | 12639.682 | 18497.573 | 12960.006 |
| interleaved15-08-short | 18120.295 | 4639.338 | 18399.590 | 4893.705 |
| interleaved15-09-medium | 18021.409 | 12868.020 | 18299.142 | 13133.283 |
| interleaved15-10-short | 17915.328 | 12994.677 | 18201.330 | 13252.173 |
| interleaved15-11-long | 17824.461 | 16770.949 | 18098.259 | 17095.555 |
| interleaved15-12-short | 17715.925 | 12829.978 | 17998.858 | 13085.486 |
| interleaved15-13-medium | 17237.678 | 17035.108 | 17507.787 | 17300.323 |
| interleaved15-14-short | 17514.458 | 17162.299 | 17795.957 | 17422.702 |

整轮回放墙钟时间：baseline 19.223s / preemptive 18.849s。均值改善不代表每条请求改善，也不代表吞吐或整轮完成时间改善。

整批请求总时间（首个 HTTP 发出→最后响应结束）：baseline 19.220133s，0.780432 成功请求/秒 / preemptive 18.840865s，0.796142 成功请求/秒。
P/D host 执行累计见 comparison-request-metrics.csv，完整 token ID、文本和 top logprobs 见各组 requests.csv。

## 实际抢占选择

下列每行是一次 arbiter 绕过队首 long、选中一个 short chunk 的决策。次数按 chunk 计数，同一 short 可以多次抢占；dispatch 表示选择，WRITE start/complete 表示实际传输，不把二者混同。时间相对 preemptive 回放起点。

| t(ms) | short | chunk | 被越过的 long | 累计抢占次数 | long 进度 | 累计等待(ms) |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 1267.037 | interleaved15-02-short | 0 | interleaved15-01-long | 1 | 0.2240 | 0.000 |
| 1293.602 | interleaved15-02-short | 1 | interleaved15-01-long | 2 | 0.2304 | 26.567 |
| 5153.945 | interleaved15-06-short | 0 | interleaved15-05-long | 1 | 0.0000 | 0.000 |
| 5179.215 | interleaved15-06-short | 1 | interleaved15-05-long | 2 | 0.0000 | 25.314 |
| 5205.465 | interleaved15-08-short | 0 | interleaved15-05-long | 3 | 0.0000 | 51.491 |
| 5230.750 | interleaved15-08-short | 1 | interleaved15-05-long | 4 | 0.0000 | 76.810 |
| 13786.223 | interleaved15-12-short | 0 | interleaved15-11-long | 1 | 0.0000 | 0.000 |
| 13811.860 | interleaved15-12-short | 1 | interleaved15-11-long | 2 | 0.0000 | 25.690 |

## 每个 long 首次触发保护的边界

| long | 相对时间(ms) | chunk | 进度 | 累计等待(ms) | 原因 |
| --- | ---: | ---: | ---: | ---: | --- |
| interleaved15-01-long | 4480.427 | 150 | 0.9536 | 51.923 | progress |
| interleaved15-05-long | 5255.783 | 0 | 0.0000 | 101.910 | wait |
| interleaved15-07-long | 13112.885 | 150 | 0.9536 | 0.000 | progress |
| interleaved15-11-long | 17656.984 | 150 | 0.9536 | 51.544 | progress |

100ms 在 chunk 边界检查；等待量可以超过 100ms，不能把它解释为整个请求的最大等待或 TTFT 上限。

所有保护恢复的序号、进度、累计等待与触发原因记录在 `protection-decisions.csv`。保护后每个 long chunk 都可能记录 protected_resume，行数不是独立恢复次数。medium 保持 FIFO 屏障，short 只越过屏障之前的 long；该规则并不保证所有 short 都先完成。

## 完整过程

1. 两种模式各启动一组独立 P/D 与 router，等服务健康并预留 15 秒，再按固定顺序每 100ms 提交一个请求。
2. P 以动态 batch 计算 prompt，baseline 按整请求 push；preemptive 在 KV 就绪后分块入队，在 chunk 边界仲裁，已在途 WRITE 不可中断。
3. D 收齐所需 KV 后首次调度、提交生成，并经 router 流式返回。每请求记录首 token、完整结束和 P/D host 执行累计。
4. 请求结束后执行严格 trace 检查，清理该组服务，下一组重新启动。最后校验两模式输出 token ID、文本 hash 及直接推理参考。

逐请求里程碑见 `request-timeline.csv`；包含每个 host batch、每个 chunk 接受/选择/WRITE start/complete/D receive 的完整时序见 `experiment-timeline.csv`。原始 monotonic 时间与各组相对时间均保留。baseline 未记录逐 chunk KV 时间，不虚构该阶段事件。

本次每种模式只跑一轮，不能建立稳定分位数或普适性能结论。P/D host 执行含调度和结果等待，batch 共享时间不能相加当 GPU 独占时间。batch-invariant 改变了内核路径，新性能数据应在两组相同配置下比较，不直接与旧默认内核实验作因果对比。
