# KV 传输无整体收益的根因与修复

日期：2026-11-06。保留短/中按 batch 聚合、long 每 8 blocks、85% / 3 次 / 50ms、原交错 15 请求 workload；不再扩展保护参数。主验证使用 event source fence、单 WRITE 窗口，baseline 和候选采用相同传输层设置。

## 已确认的根因

应用层减少 chunk，没有消除 UCX TCP+cuda_copy 在底层产生的小 staging 分段。实测日志确认本机数据协议使用 `rma_am(tcp/lo)`；此前设备 trace 中 174,720 次 P 端 DtoH，有 169,216 次大小为 8172 bytes。这是传输层粒度，和“4/8 个 KV block 一个应用 chunk”是不同层次。

进一步拆分 WRITE 的 host 时间后，主要成本落在 NIXL `transfer()` 调用内，不在 Python 描述符构建或 `make_prepped_xfer()`。非阻塞 CUDA event 队列允许已经完成的 KV 与后续计算重叠，但 writer 仍要花很长时间提交/推进大量 TCP staging 工作，后续已就绪 KV 因而排队。不能把“API 用于异步传输”理解成该调用在本后端必定立即返回。

同版代码、相同 C8/event/单槽配置的 8KB 对照与 128KB 验证：

| 指标 | 8KB 分段对照 | 128KB 分段（3 轮） |
| --- | ---: | ---: |
| 候选 makespan | 16.256s（1 次） | 5.601s 均值，5.582～5.636s |
| 每轮 long WRITE 数 | 316 | 316 |
| 每轮 long descriptor 总数 | 80,000 | 80,000 |
| long 描述符构建 mean | 0.072ms | 约 0.07ms |
| long NIXL 请求创建 mean | 0.015ms | 约 0.02ms |
| long `transfer()` host 调用 mean | **41.933ms** | **10.348ms** |
| long `transfer()` host 调用累计 | **13.251s** | **约 3.3s** |

单项 host 调用累计减少约 9.98s，与 makespan 减少约 10.66s 同量级。该累计值并不等于纯 DMA 时间，也不能与 GPU 时间直接相加；它定位的是 writer 被占用的 host 边界。8KB 只做一次同版回切控制，128KB 的稳定性由另三轮验证；没有据此声称通用分位数。

旧 `trace-chunks.csv` 的 `transfer_ms` 是 chunk start 日志到 completion 被处理的 host 区间：轮询发现完成后先入 completion inbox，下一次循环才记录完成，期间可能已提交下一个 WRITE。因此其 90～100ms 值包含控制处理延迟，不能当成每 chunk 纯网络时间，亦不能相加计算整条链路占用。本轮新增 `PD_TRACE_WRITE_PHASES=1` 区分描述符、请求创建与提交时间。

## 为什么已经有计算/传输重叠，却没有整体收益

此前重叠证明只覆盖 P 端 DtoH staging，未覆盖 CPU/socket 和 D 端 HtoD。旧 profile 的计算 busy 为 4779.842ms，DtoH busy 257.656ms，重叠 193.079ms；隐藏这部分设备时间只能省几百毫秒，而 writer 提交工作和队列尾部消耗数秒。

新 profile 的计算 busy 4782.561ms，基本不变。DtoH 的主要分段变为 65536 和 131052 bytes，原来主导的 8172-byte 类别消失。每 MiB 的拷贝数从 **132.16 降到 12.53**，约减少 90.5%；在相同“首个真实 forward→全部 prompt-final”的采样定义下，捕获数据从 1.386GB 增到 5.232GB。新 DtoH busy 为 314.167ms，其中 313.508ms 与计算重叠（99.79%）。这说明优化了单位数据的提交/分段成本，使传输能在计算期间推进更多有效数据，而不只是提高一个重叠比例。

两个 profile 的应用 chunk 分别为 C4/C8，不能用它们单独隔离 chunk 变化的因果贡献；同版 C8 的 8KB/128KB host 对照提供独立验证。profile 仍仅覆盖 P staging 和有限窗口，含 profiler 开销，不计入下面的性能均值，也不是整条 P→D 链路 profile。

## 公平配对的三轮最终结果

baseline 和候选都设 TX/RX=128KB，WRITE 窗口=1；没有只优化候选而保留慢 baseline。

| 轮次 | baseline makespan(s) | C8 候选(s) | 候选减少 | 同轮最差 long E2E 比值 |
| --- | ---: | ---: | ---: | ---: |
| 1 | 6.140078 | 5.583879 | 9.06% | 0.9009 |
| 2 | 6.260393 | 5.581918 | 10.84% | 0.8788 |
| 3 | 6.483298 | 5.636318 | 13.06% | 0.8429 |
| 均值 | **6.294590** | **5.600705** | **11.02%** | 三轮每条 long 均降低 |

| 类别 | baseline TTFT mean(ms) | 候选 TTFT mean(ms) | 降低 |
| --- | ---: | ---: | ---: |
| short | 2765.954 | 2345.211 | 15.21% |
| medium | 3679.014 | 2858.093 | 22.31% |
| long | 3520.691 | 2723.538 | 22.64% |

六个正式服务组共 90/90 请求 token ID 和文本 hash 均匹配直接推理参考。候选每轮 long 有 312 个 8-block chunk 和 4 个 1-block final tail，块区间与序号无缺失/重叠；短/中仍按各自 batch 聚合，跨 batch 会产生多个 chunk。动态 batch 边界变化使本次每轮总 chunk 为 330，不能硬编码为旧实验的 329。8KB 控制和新 profile 另各 15/15 匹配参考，合计独立审计 120/120。

### 补充：长请求延迟

从六组原始 `requests.csv` 重算，每轮 4 条 long，每条输入 10,000 tokens、输出 8 tokens。TTFT 为客户端开始请求到首 token；E2E 为客户端开始请求到完整响应，使用 CSV 的 `ttft_ms` / `e2e_ms`，不包含客户端开始前的队列等待。

| 轮次 | baseline long TTFT mean(ms) | 候选 long TTFT mean(ms) | baseline long E2E mean(ms) | 候选 long E2E mean(ms) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3459.766 | 2718.489 | 3771.539 | 3032.707 |
| 2 | 3508.623 | 2719.046 | 3823.707 | 3032.326 |
| 3 | 3593.684 | 2733.077 | 3905.277 | 3049.237 |
| 三轮均值 | **3520.691** | **2723.538（降低 22.64%）** | **3833.507** | **3038.090（降低 20.75%）** |

| long 请求（ID 尾部） | baseline E2E 三轮均值(ms) | 候选 E2E 三轮均值(ms) | 降低 |
| --- | ---: | ---: | ---: |
| 01-long | 2434.842 | 1857.104 | 23.73% |
| 05-long | 3353.938 | 2527.425 | 24.64% |
| 07-long | 4557.979 | 3411.365 | 25.16% |
| 11-long | 4987.271 | 4356.466 | 12.65% |

12 组逐请求、逐轮配对 E2E 均降低，改善范围 9.91%～26.47%。三轮中观测到的最大单请求 E2E，baseline 为 5177.633ms，候选为 4364.375ms；这是小样本最大值，不是稳定 p99。这里的 long 指长输入，尚未验证长输出 decode 负载。

### 补充：共有传输成本与可替代路径

baseline 与抢占方案使用同一个 `NixlPushConnector`：baseline 的 `_do_start_push_kv()` 和候选的 `_start_stream_chunk()` 都进入 `_xfer_blocks_for_req()` → `_xfer_blocks()` → NIXL `transfer()`。因此 TCP staging 及其底层分段成本不是抢占方案独有。baseline 按请求整体提交；候选按已就绪 chunk 提交，另外承担更多调用、通知和仲裁成本。应用层整体提交也不等于底层一次大拷贝，仍需处理 KV 描述符及 UCX 分段。

这里的约 8KB/128KB 是 UCX TCP 传输缓冲分段，不是网络 MTU 或单个 TCP 包大小。保留当前 TCP+cuda_copy 路径时，staging、协议处理与带宽成本不会因调大分段而全部消失；128KB 只是本轮已验证有效的摊薄办法，也没有证明是全局最优。

其他方向包括：在实际支持的同机拓扑上采用 CUDA IPC/P2P，跨机使用支持 GPU 内存直接传输的 RDMA；或合并连续描述符/打包 KV 后批量传输，减少小拷贝与调用。前者取决于 GPU、驱动、拓扑和 NIXL/UCX 后端支持，双 RTX 4090 不可假定存在可用 NVLink/P2P；后者会增加打包拷贝、内存占用及等待，并可能拉长抢占响应。改进后台 progress/并发提交也需确认底层线程安全及锁竞争。本轮未验证这些替代实现的收益；此前 2/4 WRITE 窗口探索不足以替代此次分段修复。

结论限于本机双 RTX 4090、TCP/lo staging 路径及当前固定 workload。现有证据确认：传输层修正后，该方案在同等 backend 参数下取得可重复的整体和响应延迟收益；不外推为 RDMA、NVLink 或任意负载的通用最优。

## 实现与保留的消融

- `run_pd_pair.sh` 给实验服务默认设置 `UCX_TCP_TX_SEG_SIZE=128k` 和 `UCX_TCP_RX_SEG_SIZE=128k`，两模式一致，尊重显式环境覆盖，不改系统全局设置。event fence 仍显式启用，脚本默认 fence 的通用回退策略保持原状。
- writer 支持 `PD_STREAM_MAX_IN_FLIGHT` 有界窗口，默认 **1**；2/4 槽仅作为探索消融保留，最终收益无需启用它们。多活动句柄用 request/sequence 索引，取消时保留所有活动 DMA 的源块 lease；失败状态跨多次轮询保持，final summary 仍等待该请求全部成功。
- 打点新增 `PD_TRACE_WRITE_PHASES`；环境 sidecar 保存 UCX 变量和 WRITE 窗口。最早几个正式 sidecar 写入时尚未加入 UCX 采集字段，其原件保留，统一显式调用环境见 `experiment-settings.json`。
- 修复 sweep 的 `--help`，防止帮助查询实际启动服务；启动前检查 P/D 端口，避免另一组服务满足本组健康检查。占用端口的拒绝测试见 `launcher-validation.json`。
- 新增 `analyze_pd_transfer_bottleneck.py`，独立审计输出、序号、block 覆盖与 WRITE 阶段，结果为 `correctness-and-bottleneck-audit.json`。77 项回归通过，14 条 Torch 弃用警告；Python 编译和 shell 语法检查通过。

2 槽实验均值约 15.921s，配对结果不一致；4 槽均值约 14.880s，baseline 波动仍大，且未消除提交热点。它们不是最终修复的主要因果依据，也未升级默认窗口。此前把启动节奏猜测为 batch 产出受限的解释，已被 WRITE 内部计时修正为提交耗时主导。

## 数据、失败过程与复现

正式数据在本目录；同版控制在 `../ucx8k-control-c8-20261106/`；profile、`cuda-overlap.json`、交互时间线在 `../ucx128k-profile-c8-20261106/`。旧 profile 在 `../overlap-cuda-profile-final-20261106/`。探索窗口实验在 `../stream-window2-c8-isolated-confirm-20261106/` 和 `../stream-window4-c8-event-confirm-20261106/`。

初次 `stream-window2-c8-event-confirm-20261106/` 受误执行 sweep help 导致的服务撞端口污染，trace 失败，完整排除，不能引用其中时延或 HTTP 成功作验证。`stream-window2-c8-confirm-20261106/` 为空的启动前路径错误目录。之后均先清理再用隔离端口；原失败数据未覆盖。

当前默认实验 launcher 已含 128KB，仍建议复现时显式写出关键变量：

```bash
PD_KV_SOURCE_FENCE=event PD_STREAM_MAX_IN_FLIGHT=1 \
UCX_TCP_TX_SEG_SIZE=128k UCX_TCP_RX_SEG_SIZE=128k \
/home/ubuntu/miniconda3/envs/cc/bin/python run_pd_tuning_confirm.py \
  --grid results/tune-interleaved15-20261106 \
  --output results/ucx128k-new-confirm --repeats 3 \
  --candidates c8-p3-x0.85-w50
```

NVIDIA Holoscan 的官方传输参数说明与这里的实验诊断吻合：默认 8KB TCP 分段会影响 GPU staging 性能，并使用 128KB 作为替代默认。该来源只用于形成假设，结论由本机对照和 trace 支持。来源：

```text
https://docs.nvidia.com/holoscan/archive/4.2.0/api/cpp/variable_namespaceholoscan_1a50da59ee78d8588357fb0fe74c821e08.html
```
