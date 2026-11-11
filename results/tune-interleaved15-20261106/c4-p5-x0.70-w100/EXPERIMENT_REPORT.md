# 15 请求交错 PD 实验过程与结果

数据集：`workload.sweep-interleaved-8s3m4l.jsonl`，SHA256 `714990980c49534e539a125612ce1c8d36c68ac75a559e4bce93dabcb7ee9e14`。
发送顺序：short / long / short / medium / short / long / short / long / short / medium / short / long / short / medium / short。

每类 prompt 长度：short=128、medium=1107、long=10000 tokens。
batch-invariant 配置：baseline=True, preemptive=True；其余运行参数见 comparison.json。抢占参数 chunk=4、次数=5、进度=70%、累计等待=100ms。

输出一致性通过：True；不经传输的直接推理参考一致：True；综合通过：True。

## 指标均值（ms）

| 类别 | baseline TTFT | preemptive TTFT | baseline E2E | preemptive E2E |
| --- | ---: | ---: | ---: | ---: |
| short | 12023.799 | 8805.436 | 12282.129 | 9050.030 |
| medium | 14477.230 | 10549.038 | 14776.925 | 10808.290 |
| long | 14299.549 | 9764.950 | 14614.328 | 10089.003 |

## 逐请求结果（ms）

| 请求 | baseline TTFT | preemptive TTFT | baseline E2E | preemptive E2E |
| --- | ---: | ---: | ---: | ---: |
| interleaved15-00-short | 945.210 | 17020.477 | 954.965 | 17276.999 |
| interleaved15-01-long | 13782.184 | 4248.323 | 14108.570 | 4572.116 |
| interleaved15-02-short | 13885.915 | 1278.292 | 14189.270 | 1448.152 |
| interleaved15-03-medium | 13782.273 | 4455.108 | 14089.074 | 4706.703 |
| interleaved15-04-short | 13680.805 | 4531.239 | 13992.102 | 4786.009 |
| interleaved15-05-long | 13584.752 | 8012.029 | 13891.621 | 8338.181 |
| interleaved15-06-short | 13479.751 | 4402.799 | 13787.038 | 4654.662 |
| interleaved15-07-long | 13383.290 | 11487.812 | 13687.233 | 11814.171 |
| interleaved15-08-short | 13277.689 | 4199.345 | 13585.142 | 4451.580 |
| interleaved15-09-medium | 13175.591 | 11682.425 | 13484.573 | 11946.229 |
| interleaved15-10-short | 13101.504 | 11770.657 | 13408.276 | 12029.128 |
| interleaved15-11-long | 16447.971 | 15311.634 | 16769.889 | 15631.544 |
| interleaved15-12-short | 13485.545 | 11642.569 | 13745.718 | 11898.927 |
| interleaved15-13-medium | 16473.827 | 15509.580 | 16757.128 | 15771.937 |
| interleaved15-14-short | 14333.975 | 15598.113 | 14594.524 | 15854.779 |

整轮回放墙钟时间：baseline 18.083s / preemptive 17.281s。均值改善不代表每条请求改善，也不代表吞吐或整轮完成时间改善。

整批请求总时间（首个 HTTP 发出→最后响应结束）：baseline 18.079571s，0.829666 成功请求/秒 / preemptive 17.277763s，0.868168 成功请求/秒。
P/D host 执行累计见 comparison-request-metrics.csv，完整 token ID、文本和 top logprobs 见各组 requests.csv。

## 实际抢占选择

下列每行是一次 arbiter 绕过队首 long、选中一个 short chunk 的决策。次数按 chunk 计数，同一 short 可以多次抢占；dispatch 表示选择，WRITE start/complete 表示实际传输，不把二者混同。时间相对 preemptive 回放起点。

| t(ms) | short | chunk | 被越过的 long | 累计抢占次数 | long 进度 | 累计等待(ms) |
| ---: | --- | ---: | --- | ---: | ---: | ---: |
| 1287.239 | interleaved15-02-short | 0 | interleaved15-01-long | 1 | 0.2432 | 0.000 |
| 1310.601 | interleaved15-02-short | 1 | interleaved15-01-long | 2 | 0.2496 | 23.359 |
| 4746.157 | interleaved15-06-short | 0 | interleaved15-05-long | 1 | 0.0000 | 0.000 |
| 4768.713 | interleaved15-06-short | 1 | interleaved15-05-long | 2 | 0.0000 | 22.639 |
| 4792.978 | interleaved15-08-short | 0 | interleaved15-05-long | 3 | 0.0000 | 46.664 |
| 4826.664 | interleaved15-08-short | 1 | interleaved15-05-long | 4 | 0.0000 | 80.396 |
| 12596.337 | interleaved15-12-short | 0 | interleaved15-11-long | 1 | 0.0000 | 0.000 |
| 12621.814 | interleaved15-12-short | 1 | interleaved15-11-long | 2 | 0.0000 | 25.548 |

## 每个 long 首次触发保护的边界

| long | 相对时间(ms) | chunk | 进度 | 累计等待(ms) | 原因 |
| --- | ---: | ---: | ---: | ---: | --- |
| interleaved15-01-long | 3144.868 | 111 | 0.7040 | 45.508 | progress |
| interleaved15-05-long | 4861.157 | 0 | 0.0000 | 115.088 | wait |
| interleaved15-07-long | 11046.751 | 111 | 0.7040 | 0.000 | progress |
| interleaved15-11-long | 15248.400 | 111 | 0.7040 | 50.852 | progress |

100ms 在 chunk 边界检查；等待量可以超过 100ms，不能把它解释为整个请求的最大等待或 TTFT 上限。

所有保护恢复的序号、进度、累计等待与触发原因记录在 `protection-decisions.csv`。保护后每个 long chunk 都可能记录 protected_resume，行数不是独立恢复次数。medium 保持 FIFO 屏障，short 只越过屏障之前的 long；该规则并不保证所有 short 都先完成。

## 完整过程

1. 两种模式各启动一组独立 P/D 与 router，等服务健康并预留 15 秒，再按固定顺序每 100ms 提交一个请求。
2. P 以动态 batch 计算 prompt，baseline 按整请求 push；preemptive 在 KV 就绪后分块入队，在 chunk 边界仲裁，已在途 WRITE 不可中断。
3. D 收齐所需 KV 后首次调度、提交生成，并经 router 流式返回。每请求记录首 token、完整结束和 P/D host 执行累计。
4. 请求结束后执行严格 trace 检查，清理该组服务，下一组重新启动。最后校验两模式输出 token ID、文本 hash 及直接推理参考。

逐请求里程碑见 `request-timeline.csv`；包含每个 host batch、每个 chunk 接受/选择/WRITE start/complete/D receive 的完整时序见 `experiment-timeline.csv`。原始 monotonic 时间与各组相对时间均保留。baseline 未记录逐 chunk KV 时间，不虚构该阶段事件。

本次每种模式只跑一轮，不能建立稳定分位数或普适性能结论。P/D host 执行含调度和结果等待，batch 共享时间不能相加当 GPU 独占时间。batch-invariant 改变了内核路径，新性能数据应在两组相同配置下比较，不直接与旧默认内核实验作因果对比。
