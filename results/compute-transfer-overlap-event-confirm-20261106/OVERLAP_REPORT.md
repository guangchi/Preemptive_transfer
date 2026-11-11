# 计算与 KV 传输重叠实验

日期：2026-11-06。固定保护参数 **85% / 3 次 / 50ms**，chunk=4 blocks；保留 15 请求、8 short/3 medium/4 long、100ms 到达间隔、并发 15、seed=0、batch-invariant=1、关闭 prefix caching。输入 SHA256：`714990980c49534e539a125612ce1c8d36c68ac75a559e4bce93dabcb7ee9e14`。本轮不扩展保护参数网格。

## 结论

CUDA event 非阻塞就绪队列已实现，设备 trace 证明计算 kernel 与 P 端传输拷贝实际重叠。三轮事件实验共 90/90 请求 HTTP、token ID 和文本 hash 与参考一致；抢占组每轮 698/698 chunk 成功且按请求连续递增完成，23 批全部使用 event，没有设备全同步回退。

短请求平均 TTFT 相对配对 baseline 下降 **30.75%**，但整批平均 makespan 增加 **19.14%**，最差单 long E2E 增幅 **25.58%**。因此只确认当前 workload 的短请求响应优势及真实设备重叠，**尚不能宣称整批性能优于 baseline，也未通过此前 +10% long E2E 分析约束**。event 模式保持显式启用；默认仍为原主线程设备同步模式。有限 in-flight WRITE 窗口和 chunk 粒度未改动。

## 实现

实际使用 vLLM V2 GPU runner，旧版 runner 中已有的 forward CUDA event 计时不会被调用。本轮将计时接入 `vllm/v1/worker/gpu/model_runner.py`，回避 `_warmup_` 请求。可选 Kineto/CUPTI profile 从首个真实 forward 开始，到所有请求 prompt-final forward 完成为止。

V2 `ActiveKVConnector._start_load_kv()` 在 **post-forward** 记录当前 producer stream 的 CUDA event，事件只保存在 worker 本地 metadata，不传回 scheduler。启用条件严格限定：LlamaForCausalLM，TP/PP/DCP/PCP=1，无 DBO/ubatching、无 speculative decode、无模型权重 offload，且本批不是 sync KV load。该路径的 Llama attention/KV 更新与 eager/piecewise kernel 在当前 stream；CUDA graph replay 在当前 stream 建立执行顺序。采样中全部 7,826 个计算 kernel 位于 stream 7，传输 DtoH 位于独立 stream 3544；这支持本实验路径，不构成任意模型/stream 的普遍保证。

writer 持有批次及 CUDA event，只调用 `event.query()`；未就绪批次留在 FIFO 队列，writer 继续推进已有 WRITE、注册和取消。先前批次未就绪时禁止后续批次越过，防止 final 抢先。ready 后才放入原 chunk 仲裁器；WRITE/summary/源块回收仍沿用现有完成边界。失败或取消时，新增 source-pending 计数防止尚在 event 队列中的批次被提前释放。未满足 event 范围或未接入该 hook 的路径回退原主线程 `torch.cuda.synchronize()`。

`PD_KV_SOURCE_FENCE=device` 为默认；`event` 启用上述路径；`writer-device` 仅用于保留消融复现。`run_pd_pair.sh` 校验配置，environment JSON 记录 fence 模式。

## 三轮性能

| 轮次 | baseline makespan(s) | event makespan(s) | 最差单 long E2E 增幅 | 输出/序号 |
| --- | ---: | ---: | ---: | --- |
| 1 | 14.551692 | 17.462075 | +17.70% | 通过 |
| 2 | 13.853301 | 17.665970 | +25.58% | 通过 |
| 3 | 15.367487 | 17.020555 | +11.02% | 通过 |
| 均值 | 14.590827 | 17.382867 | — | 90/90 请求 |

| 类别 | baseline TTFT mean(ms) | event TTFT mean(ms) |
| --- | ---: | ---: |
| short | 10377.329 | 7185.975 |
| medium | 11416.889 | 10563.124 |
| long | 11776.721 | 9778.569 |

主数据：`confirmation-results.json`、`raw-confirmation.json`、`correctness-audit.json` 及各组 requests/trace-request-metrics/trace-chunks/environment/log。runner 的退出码 1 来自 long 准入约束失败，六个单组 runner 均为 0；不能将它解释为输出校验失败。

**版本限制：**这是实现过程中的三轮事件实验。第一轮加载了最初的 event 查询实现；第二、三轮加载了加强 FIFO event 轮询的实现。三轮实际 chunk 顺序均通过独立审计，但不能将其描述为最终逐字节同版代码的三轮验收。随后加入 V2 forward 计时、恢复默认同步回退、取消时 source-pending lease 保护；最后代码有完整单测，且计时/回退版本另有一次正确性通过的 GPU profile。最终 source-pending 取消保护没有新增真实 GPU 故障注入。最终版本仍属实验模式，不据这三轮升级为默认实现。

## 设备重叠

成功 profile：`../overlap-cuda-profile-final-20261106/prefill-cuda-trace.json`（约 127MB）。这次 15 请求均与配对 baseline 的 token/text 一致，详情见该目录 `comparison.json`。其运行耗时含 profiler 开销，**不计入上表**。

| CUPTI 设备区间（profile 窗口） | 数值 |
| --- | ---: |
| P 计算 kernel 数 | 7,826 |
| P 计算 busy 区间并集 | 4779.842ms |
| P 传输 DtoH 拷贝数 | 174,720 |
| P 传输 DtoH busy 区间并集 | 257.656ms |
| kernel 与 DtoH 区间交集 | **193.079ms** |
| 交集 / DtoH busy | **74.94%** |
| 交集 / compute busy | 4.04% |
| DtoH 首尾跨度 | 6182.063ms |

这些是 CUPTI 的实际设备时间戳，用同一 trace 时钟对区间求并集/交集，不将 host enqueue 视为 DMA。`cuda-overlap.json` 保存明细；`cuda-overlap.html` 可缩放浏览区间。重建：

```bash
/home/ubuntu/miniconda3/envs/cc/bin/python analyze_pd_overlap.py \
  results/overlap-cuda-profile-final-20261106/prefill-cuda-trace.json \
  --output results/compute-transfer-overlap-event-confirm-20261106
```

**解释范围：**profile 只观测 P 端 UCX Device→Pageable Host staging leg，**不是整条 P→D 链路**；不包含 CPU/socket 传输和 D 端 HtoD，也不包含 profile 结束后的传输尾部。共捕获约 1.386GB DtoH，其中 169,216 次为 8,172 bytes。大量小拷贝及 busy 相对跨度较低，表明传输路径存在明显间隙；不能只凭这一 trace 将全部性能差额因果归给 UCX、pageable host 内存或某一个控制环节。NIXL 1.5.2 的 `xferDuration` 文档明确注明 `checkXferReq` 轮询延后会影响其值，因此没有把 NIXL telemetry 当作纯 DMA 时间。

## 消融、验证与下一步

先尝试将设备同步搬到 writer，三轮平均 makespan **22.238433s**，配对 baseline **14.459045s**，90/90 输出一致。数据保留在 `../compute-transfer-overlap-20261106/`，当时源码保存在其 `source-writer-sync/`。writer 被同步占用会影响已有操作推进，因此未采纳作默认路径；当前默认已恢复。

第一次 profile 从 warmup 开始且没有在服务退出前导出，保留在 `../overlap-cuda-profile-20261106/`，不用于设备重叠结论；它的 runner 最终配对步骤因没有 baseline 目录退出 1，单组 HTTP/trace 成功。修正采样窗口后成功 profile 在 `../overlap-cuda-profile-final-20261106/`。

最终回归 **72 passed、14 warnings**（Torch 弃用警告），文本/JUnit 见 `../../logs/compute-transfer-overlap-20261106/pytest-final.*`。新增回归覆盖 event 未就绪不暴露 chunk、前序 event 阻止 final 越过、event 错误按失败清理、取消时 fence 未完成保留 lease、sync-load/非 Llama/ubatching 回退。10 个 runtime 文件与 snapshots 逐字节一致，manifest 和最终源文件保存在 `source-final/`。GPU 服务已停止，无残留计算进程。

下一步优先审计本机 UCX 的小粒度 DtoH/socket 路径和每个 WRITE 的操作间隙，再决定是否值得进入有限 WRITE 窗口；继续固定 85%/3/50ms。进入新阶段前应为最终同版代码补三轮验证及 event 等待期间取消/失败的 GPU 故障验收，不能直接沿用本轮实现过程数据作准入证明。

复现实验（显式实验模式，六组服务重启）：

```bash
PD_KV_SOURCE_FENCE=event /home/ubuntu/miniconda3/envs/cc/bin/python run_pd_tuning_confirm.py \
  --grid results/tune-interleaved15-20261106 \
  --output results/overlap-new-run --repeats 3 --candidates c4-p3-x0.85-w50
```
