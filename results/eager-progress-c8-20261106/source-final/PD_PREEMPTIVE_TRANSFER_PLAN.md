# PD 分离下短请求抢占式 KV 传输：设计与执行计划

更新日期：2026-11-06（最新同条件三轮：baseline 6.175s、候选 5.657s，整批缩短 8.39%；默认入口统一 event，完成进度在下一次仲裁前更新。相对上一版候选未证明进一步加速）。后续工作以本文的“当前执行位置”为入口。
工作目录：`/home/ubuntu/zjj/code`。
运行环境：Conda `cc`，vLLM 0.32.0，Llama-3.1-8B-Instruct，两张 RTX 4090。

系统交付说明：[PD_PREEMPTIVE_TRANSFER_SYSTEM_REPORT.md](PD_PREEMPTIVE_TRANSFER_SYSTEM_REPORT.md)，包含系统功能、问题与修复方案、指标定义、实测结果、数据索引和单组复现方法。

## 1. 设计目标与固定范围

固定 TP=1、PP=1，P/D 各用一张 GPU。不做多 TP、多 PP、多 KV group 或 host offload 的通用化。

1. 每次 P 端 batch 计算完成后，短/中请求各将本批新 KV 聚合为一个 chunk；long 按 8 blocks 分块，尾块可不足 8 blocks，提交给传输队列。
2. 在途 chunk 不被中断；完成一个 chunk 后重新仲裁，允许后到、KV 已就绪的短请求越过未受保护的长请求。
3. 长请求满足任意保护条件后，禁止继续插队：累计抢占次数达到上限；成功传输比例达到 85%；因插队累计等待达到时间阈值。
4. D 收齐全部 chunk、确认 final 且 KV 对后续计算可见后，才继续 decode。P 的块必须在 DMA 完成或安全失败清理后才能释放。

抢占发生在传输调度层。chunk 的 block 数与计算 batch 大小、chunked prefill 的 token 数是不同参数。

| 参数/规则 | 当前默认值或口径 |
| --- | --- |
| short | 原始 prompt tokens < 256 |
| long | 原始 prompt tokens > 8192 |
| medium | 其余请求；当前代码按 FIFO 并形成插队屏障 |
| chunk | long 为 8 KV blocks；短/中按 batch 聚合 |
| UCX TCP TX/RX 分段 | 实验 launcher 默认两端 128KB，允许环境覆盖；与 KV chunk 大小独立 |
| WRITE 窗口 | 默认 1；有界多槽仅作消融，最终性能验证使用单槽 |
| source fence | launcher 默认 event；仅在已审计路径使用 CUDA event，其他路径安全回退全设备同步 |
| 连续立即提交 | `PD_STREAM_EAGER_PROGRESS=0`；开启仅作消融，未验证端到端收益 |
| 最大抢占次数 | 3；当前代码按一次 short chunk 越过队首 long 计一次 |
| 进度保护 | 成功传输块数 / 该请求实际需要传输总块数 >= 0.85 |
| 等待保护 | 因插队累计等待 >= 50 ms |
| 动态 batch 数值一致性 | P/D 默认 `VLLM_BATCH_INVARIANT=1`；运行环境记录实际配置 |

medium 屏障是已有行为，本阶段保留并验证。等待从被插队开始，到 long 恢复传输结束该区间；多次区间累加，排除正常计算、registration 等待和 long 自身的传输时间。阈值在 chunk 边界检查，不能打断已经在途的 short chunk。

进度分母必须在请求级稳定，并正确考虑 D 的 prefix cache 命中；不能用某次 batch 当前分配的块数代替整个请求的传输量。失败 chunk 不能增加成功进度。

## 2. 已实现与验证程度

| 部分 | 当前状态 |
| --- | --- |
| 配置、短中长分类、chunk 仲裁 | 已实现；分类和基础仲裁检查通过 |
| 次数、85%、等待保护 | 仲裁保存并打印决策、long 进度、累计等待和保护原因；次数、累计等待及 prefix-cache 场景已有 GPU 轨迹，85% 保护在真实 prefix-cache workload 中已观察到，精确 85% 边界仍由逻辑回归覆盖 |
| P scheduler 按 batch 生成 chunk | 已实现；单 long 跨多个 batch 和混合长请求的 chunk 序列/完成已通过 GPU 验收 |
| NIXL writer、WRITE、chunk 通知 | WRITE、请求级 `STREAM_DONE` 和失败通知已通过成功/失败 GPU 验收；重复 summary 可安全忽略，失败 chunk 不计入进度 |
| P/D 启动与 router | GPU 启动、agent、兼容握手、KV registration、health 均通过 |
| baseline 整请求 push | 单请求成功；short/medium/long 三请求并发 3/3 成功 |
| preemptive | 单 long 跨 batch、修复后两次三请求混合、连续两批重复负载及 prefix-cache long 成功；registration timeout、router abort、WRITE 错误和 worker 退出均有 GPU 注入证据 |
| workload 与回放器 | 已实现，测试负载可回放，逐请求 CSV 可落盘 |
| 指标 | 客户端 TTFT、请求阶段时间、P/D chunk 时间线、GPU 采样和复现 metadata 已接入；D KV 收齐、首次非空计算调度、首次执行提交和客户端首 token 可直接分段；新增 P/D 每轮提交→结果、可选 CUDA forward、请求结束事件和请求级汇总；严格执行验收要求 TTFT/E2E/arrival-to-finish、P/D host 执行时间及结束事件完整 |
| sweep | 历史扫描已归档；最新固定 C8/85%/3/50ms 的 128KB 三轮确认输出一致，整批和逐 long E2E 均优于配对 baseline |
| 源码保存 | 四个 connector 文件、两个 core 文件、四个 worker 文件，共 10 个 runtime 文件保存在 snapshots/vllm-preemptive-20261103/；与 cc site-packages 逐字节一致，SHA256SUMS 刷新 |

当前回归为 **80 passed，14 warnings**，新增成功进度先于仲裁、DONE 连续提交和 PROC 保留等待检查，并覆盖 WRITE 窗口、失败保留及取消 lease，详情见当前执行位置。以下保留 long 输出修复阶段的 **63 passed，14 warnings** 回归记录，覆盖 arbiter 保护/轨迹、D 完成方向和序号、P lease 清理、失败清理、取消后 chunk 丢弃、单槽 deferred WRITE、summary 完成、prefix-cache 进度分母、registration 超时失败回报、router abort、trace 合并/完整性及顺序校验、首次 D 调度/提交的幂等打点、P/D 执行时间与请求结束事件、成功/失败时延分布、环境 metadata 脱敏、medium 尾 chunk/final lease 竞态及流式 token ID/logprobs 审计；新增握手占槽、summary 等待全部前序 chunk、全设备同步先于入队三项回归。`pytest.ini` 限定 tests/，避免 results/ 下归档的同名测试参与收集。14 条 warning 均为 Torch `torch.jit.script_method` 弃用提示。测试依赖固定在 `requirements-test.txt`；执行命令为 `/home/ubuntu/miniconda3/envs/cc/bin/python -m pytest -q`。该历史阶段复跑为 63 passed、14 warnings（8.71s），文本/JUnit 保存于 `logs/long-output-fix-20261106/`；60 项历史结果保存于 `logs/tuning-validation-20261106/`；早期 58 项保存在 `logs/determinism-fix-20261106/`；早期 52 项结果保留于 `logs/presweep-execution-20261106/`。

baseline/preemptive 三类请求（short/medium/long）的 8-token 输出 hash 已逐请求一致；证据见 `results/presweep-baseline/requests.csv`、`results/presweep-preemptive/requests.csv` 和新流式回放记录。

## 3. 当前执行位置

### 最新：总时间复核、默认入口统一与成功进度顺序修复

报告：[TIMING_AND_SCHEDULER_REPORT.md](results/eager-progress-c8-20261106/TIMING_AND_SCHEDULER_REPORT.md)。继续固定 C8/85%/3/50ms，双方 TX/RX=128KB、WRITE 窗口=1。

- 最新三轮 baseline 为 6.155/6.211/6.158s，候选为 5.623/5.672/5.675s，均值 6.175/5.657s，候选缩短 **8.39%**；short/medium/long TTFT 降低 15.03%/19.95%/21.23%，long E2E 降低 19.40%，12 个逐 long 配对均改善。
- “明显慢于 baseline”出现在历史 8KB 数据，不能与最新运行混比。含启动和清理的脚本总时间均值 81.064/79.879s，差异包含启动波动。结果汇总新增双方三类 TTFT/E2E、双方脚本总时间及计时定义。
- launcher 默认统一为已验证的 event（非支持路径保留安全回退）。先 poll WRITE 再处理 completion，使成功进度在下一次仲裁前生效；保护参数不变。
- 无间隔连续提交只保留消融，默认为 0：单次 on/off 的间隙中位数 0.501/1.583ms，但整批为 5.691/5.627s，native 提交累计反而增加。不能把缩短间隙当作端到端收益。
- 本轮未证明比上一版 5.601s 进一步加速；本轮均值数值上高约 1.00%，两轮实现未作交错 A/B，不宣布显著加速或回归。
- 正式六组 90/90 输出匹配参考，三组消融/控制另 45/45，全计 135/135；chunk 序号、block 覆盖与保护决策通过独立审计。80 项回归通过、14 条警告；运行时和源码快照已归档。

### 最新：UCX 传输分段根因定位与整体收益验证

完整报告：[ROOT_CAUSE_REPORT.md](results/ucx128k-w1-c8-confirm-20261106/ROOT_CAUSE_REPORT.md)。固定 short/medium batch 聚合、long C8、85% / 3 次 / 50ms，event fence 和 WRITE 窗口=1。

- 根因：减少应用层 chunk 不会消除 UCX 默认约 8KB TCP staging。旧 profile 主导拷贝为 8172 bytes；同版 C8 回切 8KB 后 makespan 16.256s，316 个 long WRITE 的 `transfer()` host 提交均值 41.933ms、累计 13.251s，而描述符构建均值只有 0.072ms。
- 修正：baseline/candidate 同时将 TX/RX 设置为 128KB，仍单槽、C8、85%/3/50ms。三轮候选 5.584/5.582/5.636s，baseline 6.140/6.260/6.483s，均值 5.601 vs 6.295s，减少 **11.02%**；short/medium TTFT 均值降低 15.21%/22.31%。每条 long E2E 都低于同轮 baseline。
- 正式六组 90/90 输出匹配直接参考，控制与 profile 另 30/30 通过。候选每轮 long 312 个 8-block chunk 和 4 个 1-block tail；全部序号和块区间完整，短/中按动态 batch 聚合。
- 新 profile 中计算 busy 仍约 4.78s，DtoH 每 MiB 拷贝数从 132.16 降到 12.53；证明真正减少了传输层细碎 staging。profile 是 P 端有限采样窗口，不混入性能均值。
- `run_pd_pair.sh` 默认 TX/RX=128k、尊重显式覆盖。该三轮实验显式启用 fence=event；后续已将 launcher 默认同步方式也统一为 event（仅已审计路径生效，其他路径安全回退）。环境记录新增 UCX 设置；新增 WRITE 阶段打点和独立审计。当时 77 项回归通过、14 条 Torch 弃用警告；端口占用拒绝和 sweep help 已验证。该轮源码与校验和同目录归档。
- 2/4 WRITE 窗口只保留为探索项，最终收益来自 128KB 设置，默认窗口保持 1；不再以“缺少重叠”“batch 产出慢”或“Python 描述符处理慢”解释本次主要差额。结论限于本机 TCP 路径和该固定 workload。

### 历史：固定 85% / 3 次 / 50ms 的计算传输重叠

按用户要求，后续保护参数只使用 **85% / 3 次 / 50ms**，不再扩大参数网格。本轮 chunk 保持 4 blocks，保留同一 15 请求 workload。

- 已接入实际启用的 V2 runner：post-forward 在 producer stream 记录 CUDA event，writer 用非阻塞 query 和 FIFO 就绪队列推进，在 event ready 后才把本批 KV chunk 交给原仲裁器。限定 Llama、TP/PP/DCP/PCP=1、无 DBO/ubatching/speculation/offload；其他路径回退原主线程全设备同步。新增 source-pending 计数防止取消时尚未跨过 fence 的源块提前回收。
- 该历史实验显式启用 `PD_KV_SOURCE_FENCE=event`，当时默认仍为 `device`。后续 launcher 默认已统一为 event；运行路径不满足 event 安全条件时仍回退到全设备同步。`writer-device` 仅保留消融复现：三轮平均整批 22.238s，比其配对 baseline 14.459s 更慢，因此未采纳。
- 事件实现过程三轮：**90/90 HTTP、token ID、文本 hash 与参考一致**；每轮 698 chunk 均成功且按请求连续递增；23 batch 全部 event-ready、无设备同步回退。抢占平均整批 **17.383s**，baseline **14.591s**（+19.14%）；short TTFT **7185.975ms vs 10377.329ms**（-30.75%）；最差单 long E2E **+25.58%**。尚未通过 +10% long 分析约束，不声称整批优越性。
- CUPTI profile 证实 P compute kernel 与 UCX DtoH 实际交叠 **193.079ms**，为所采集 DtoH busy 的 **74.94%**；计算 kernel busy 4779.842ms，DtoH busy 257.656ms，DtoH 首尾跨度 6182.063ms。大量约 8KB 的 Device→Pageable Host 拷贝提示本机传输路径/操作间隙值得继续审计。该 profile 仅覆盖计算期、P 端 staging leg，不包括 socket、D HtoD 或尾部；profile 开销不计入常规性能均值。
- 最终 **72 passed、14 warnings**，10 个 runtime 文件与快照/校验和一致。事件三轮包含实现过程的 FIFO 加强，随后添加计时、回退和取消 lease 保护，因此不是最终同版三轮验收；最终取消保护尚无新增 GPU 故障注入。event 保持实验模式，未进入多 WRITE 窗口或 chunk 粒度阶段。
- 完整报告：[OVERLAP_REPORT.md](results/compute-transfer-overlap-event-confirm-20261106/OVERLAP_REPORT.md)；[设备重叠浏览](results/compute-transfer-overlap-event-confirm-20261106/cuda-overlap.html)；[重叠明细](results/compute-transfer-overlap-event-confirm-20261106/cuda-overlap.json)。常规三轮、消融和 profile 原数据均保留。服务已停止，GPU 无残留进程。

**下一步入口：**优先审计本机 UCX 的大量小 DtoH/socket 操作和 WRITE 间隙，继续固定 85%/3/50ms；新优化进入下一阶段前，先对最终同版实现补三轮和 event 等待期间失败/取消的 GPU 验收。本节后的同步/窗口/粒度步骤为原执行顺序，当前只完成了重叠测量及同步消融的实验实现，尚未通过后续阶段准入。

### 最新：long 输出竞态已修复，候选与尾延迟已复测

- 根因证据：旧默认第二轮 01-long 的 seq 0～24 在握手期间未占住单传输槽，被 seq 25～156 越过；final 完成即发 `STREAM_DONE`。D 在 monotonic `20375.180449` 宣布收齐，而前 25 个 chunk 的实际 WRITE 到 `20388.002192` 才开始、`20388.542013` 才全部完成，导致用不完整 KV 生成。该问题独立于 batch-invariant 数值内核。
- 修复 `push_worker.py`：握手至 deferred WRITE 启动/失败期间保留传输槽；全部 chunk 的 WRITE/notification 成功后才允许 summary，保存真实 final 序号；入队前全设备 CUDA 同步保证可能在其他 stream 的 KV 写入完成，并记录同步耗时。后者为保守可见性保护，不将它单独认定为已观测失败的根因。
- 固定原 15 请求和输入 hash，baseline、85%/3/50ms、95%/5/50ms、95%/5/100ms 各三轮完整修复验证：**180/180 HTTP、文本 hash、token ID 均与直接参考一致**；抢占组逐请求 chunk 成功完成顺序连续递增。pytest **63 passed、14 warnings**。8 个 runtime 文件与快照一致，SHA256SUMS 已更新。
- 最初 repeat-1 的 95%/5/50ms 在完整修复前启动，原数据保留但从最终分析排除；在 supplement/ 下单独补跑，仍与原 repeat-1 baseline 配对，补跑非相邻执行的漂移限制已注明。`retests/confirmation-results.json` 保留原始 runner 汇总，最终以 [fixed-results.json](results/long-output-fix-20261106/fixed-results.json) 的版本筛选结果为准。

| 配置（进度/次数/等待） | 整批 mean[min,max](s) | long E2E 观测最大值(ms) | 最差单 long 相对配对 baseline |
| --- | --- | ---: | ---: |
| baseline | 15.250 [13.685,18.099] | 16457.044 | — |
| 85% / 3 / 50ms | 18.237 [17.074,18.948] | 17225.512 | +35.40% |
| 95% / 5 / 50ms | 18.000 [16.609,18.974] | 17292.835 | +32.26% |
| 95% / 5 / 100ms | 18.544 [18.177,19.148] | 17469.614 | +39.04% |

- 95%/5/50ms 的 short/medium TTFT 几何均值分数中位数最低（8927.337ms），抢占配置中整批均值也最低；baseline 整批均值仍最快。**没有抢占配置通过原先每条 long E2E≤baseline+10% 的约束，默认保持不变。**每配置只有 12 条 long 观测，报告最大值而非声称稳定 P95/P99。
- 每个抢占组均发生 8 次 short chunk 越过 long。每组 23 次设备同步，host 阻塞累计约 4.23～4.38s，单次最大约 225～230ms；包含等待已提交计算，不能直接当作新增 GPU 时间或将累计值加到 E2E。握手串行化、分块控制开销和保守同步可能影响 long 尾部，本轮未做消融归因。
- 完整证据：[FIX_REPORT.md](results/long-output-fix-20261106/FIX_REPORT.md)、`paired-long-e2e.csv`、`preemption-summary.json`、各候选的逐请求四项指标、抢占表、保护表、完整时间线和 `timeline.html`。新增 [report_pd_long_output_fix.py](report_pd_long_output_fix.py) 可重建最终版本筛选/对照报告，源码和校验和同目录归档。原失败与旧 27 点不改写；本次关闭已复现的竞态问题，不宣称任意 workload 已无错误。

### 性能差异分析与下一步计划

当前数据说明“分块允许重叠”不等于“分块必然更快”。baseline 是等请求 prompt KV 计算完成后再做较大粒度的传输；preemptive 试图让后续计算与已就绪 KV 传输并行，但同步、单活动传输槽和大量小 WRITE 会限制可获得的重叠，并增加每块提交、completion 轮询、通知和仲裁工作。短请求提前获得服务可以改善 TTFT，却不增加总计算能力或链路带宽，因此不能推导整批 makespan 或 long 尾延迟也会改善。

- 复测样本（repeat-2、95%/5/50ms）中，preemptive 23 次全设备 CUDA 同步，记录到的 host 阻塞累计约 4.32s、单次最大约 230ms。该耗时包括等待已提交 GPU 工作，不是可直接从 E2E 中相加的独立开销，但 `torch.cuda.synchronize()` 位于 KV chunk 入队前，可能阻断 P 后续计算并削弱计算/传输重叠。
- 同一样本的 preemptive 有 698 个 4-block chunk；最后一个 chunk 在回放开始后约 17.94s 完成，整批在约 18.42s 结束。baseline 整批约 13.69s。baseline 日志汇总为约 14 个大传输、平均约 395MB；preemptive 约 409 个传输、平均约 8MB。NIXL 周期性 metrics 的统计窗口和 transfer 定义不同，不应用这些聚合数直接推算严格带宽倍率；它们表明粒度和操作数差异，不能单独证明某项开销的因果贡献。
- 尚未解决的测量问题：当前证据没有给出每个 scheduler batch 的计算区间与各 KV WRITE 的设备级实际重叠比例，也没有隔离全设备同步、单活动槽和 4-block 粒度各自造成的性能影响。P/D host 执行累计是请求参与 batch 的 host 时间，不是 GPU busy 时间；同步耗时是 host 阻塞时间，两者都不能当作纯 GPU kernel 时间。

**下一步按以下顺序执行，不先重扫 27 点参数网格：**

1. **补齐重叠测量。** 给 P 每个非空 batch 记录提交、forward 完成、KV 可读边界；给每个 chunk 记录 NIXL post、设备 WRITE 起止、completion/notification、D receive 和首次可调度时间。优先使用 CUDA event 和 NIXL telemetry，并统一单调时钟或明确时钟映射；报告计算与传输区间交叠毫秒/比例、传输队列等待、纯传输时间、同步阻塞时间。验证指标不能把 host enqueue 或日志时间误称为 DMA 实际开始/结束。
2. **消融全设备同步。** 保留当前修复版本作为正确性基线，先为 KV 源 buffer 建立 producer-stream CUDA event/fence，让 writer 只在相关 KV 写入完成后访问它；不要直接删除同步。以相同 15 请求、batch-invariant=1、相同 seed 做至少 3 轮配对测试。每轮先确认 token ID、文本 hash 和 chunk 序号正确，再比较 P 计算跨度、传输重叠比例、整批 makespan、各类 TTFT/E2E 和 long 尾部。若缺少可证明的 stream 依赖，则继续保留设备同步，并把它的成本作为限制报告。
3. **消融单活动 WRITE 限制。** 只有在当前顺序/summary 正确性回归和设备级测量完成后，实验有限的 in-flight 窗口（例如 2、4）；按请求/块保持有序接收判定，只有对应 WRITE 完成后才允许 summary 或释放源 block。覆盖握手延迟、跨请求并发、失败、取消、final 先完成及 completion 重复/乱序；再跑固定 15 请求三轮。用数据决定是否保留并行窗口，不能只以单轮吞吐选值。
4. **再测 chunk 粒度与调度取舍。** 在前两项确定后固定可证明正确的同步和窗口，比较当前 4 blocks 与较大 chunk（例如 8、16），至少三轮同 workload、相同 baseline 配对；记录每条 short 的 TTFT、所有 long 的最大 E2E 增幅、整批 makespan、WRITE 数、队列等待和长请求保护事件。确认不存在满足既有 long 约束的点时，如实保留性能限制，不把 10% 分析约束改写成系统保证。
5. **通过门槛与归档。** 每个阶段保留代码快照、完整日志、环境/输入 checksum、逐请求和逐 chunk 数据；先逻辑/故障回归，再 GPU 测试。任何一次输出不一致、chunk 缺失/重复、提前调度、源块过早释放或失败后残留状态，都阻止进入下一阶段。所有通过的改动同步到 site-packages 快照及 SHA256SUMS，再运行 pytest 和至少三轮固定 workload。

### 历史：15 请求总时间采集与 27 点保护参数实验（修复前）

- 输入保持不变：8 short/3 medium/4 long，原交错顺序及 seed，SHA256 `714990980c49534e539a125612ce1c8d36c68ac75a559e4bce93dabcb7ee9e14`；并发 15、到达间隔 100ms、chunk=4、prefix cache=false、batch-invariant=1。
- 新增 `collect_pd_total_time.py`，自动记录首个 HTTP 发出→最后响应结束的 `workload_makespan_s`、成功请求/秒，另保留回放 elapsed；`experiment-timing.json` 记录每组服务启动→清理的墙钟和分段，runner/网格有外层总计时。请求并发，不能把逐请求 E2E 相加当整批总时间。旧 verified 组精确补算为 baseline `18.079571s`、默认抢占 `19.116744s`，独立保存在 `results/sweep-interleaved15-original-total-times.json`/CSV；历史启动/清理未计时，不伪造。
- 已完成 27 点：次数={1,3,5}、进度={70%,85%,95%}、累计等待={25,50,100}ms。27 组均完成 15/15 HTTP，25 组输出一致、21 组满足初筛 long 约束。一个启动超时保留并补跑，两个输出失败原样保留且排除，未用重跑覆盖。网格墙钟含失败、补跑与阶段间操作 `2710.862s`。证据：[网格报告](results/tune-interleaved15-20261106/TUNING_REPORT.md)、[交互比较](results/tune-interleaved15-20261106/tuning-dashboard.html)。
- 首轮只作候选筛选：历史 baseline 开启 top-logprobs，新网格关闭；最终另用统一关闭审计的新 baseline、默认与两个候选各做 3 轮，循环调整执行顺序，共 12 组/180 请求，每组重启 P/D。确认阶段 `1102.178s`，搜索开始至最后确认组结束共 `3837.703s`（约 63.96 分钟，包含阶段间分析）。
- 目标为 short/medium 类均值 TTFT 几何平均，复测按三轮分数中位数选择；正确性和每条 long 相对同轮 baseline 的 E2E ≤+10% 为准入条件。10% 是本轮分析约束，不是用户 SLA 或系统保证。**没有抢占配置同时通过三轮全部约束，不能宣布已找到稳健最优参数。**

| 配置（进度/次数/等待） | short TTFT mean(ms) | medium TTFT mean(ms) | 整批总时间 mean[min,max](s) | 三轮输出校验 | 最差单 long E2E 变化 |
| --- | ---: | ---: | --- | --- | ---: |
| baseline | 11207.964 | 12338.227 | 15.445 [13.480,19.220] | 通过 | — |
| 85% / 3 / 50ms（默认） | 7803.730 | 10483.312 | 17.384 [17.034,17.789] | 第二轮一条失败 | +30.27% |
| 95% / 5 / 50ms | 8430.164 | 12443.526 | 20.136 [18.461,21.966] | 通过 | +60.95% |
| 95% / 5 / 100ms | 7061.427 | 11061.469 | 18.006 [16.728,18.841] | 通过 | +33.95% |

- **有条件的观测结果：95%/5次/100ms 是三轮输出均正确的已测候选中延迟分数最低者，不能称为满足 long 约束的推荐。**其分数中位数 9076.415ms，默认观测 8904.451ms（但默认含失败轮），因此它也没有证明全面优于默认。若优先整批完成时间，三轮平均 baseline 最低。通用默认仍 85%/3/50ms；保留默认不代表它已通过本次所有正确性验收。
- 每个抢占配置每轮均发生 8 次 short chunk 插队；95%/5/100ms 有单个 long 被越过 4 个 chunk，而另两配置每条 long 至多 2 次。100ms 允许更长的插队窗口，50ms 更容易先触发等待保护；相同插队总数也可形成不同顺序与长请求代价。进度、次数和等待是 OR 关系，未证明三个连续参数的全局最优。
- **仍有残余正确性问题：**网格 `c4-p1-x0.95-w100`、`c4-p5-x0.70-w50` 及确认第二轮默认的 `interleaved15-01-long` 输出与参考不同，均在 batch-invariant=1 下发生。确认默认实际 token `[8481,1715,8481,1715,8481,1715,8481,1715]`，参考 `[1121,1715,8481,1715,1121,6636,1121,8481]`。根因未定位，不能宣称所有调度组合均确定或系统无错误；失败证据保留于 `output-failures.json`。此前旧四次控制及 verified 单轮通过仍成立，但不构成通用保证。
- 最终完整报告：[FINAL_TUNING_REPORT.md](results/tune-interleaved15-confirmation-20261106/FINAL_TUNING_REPORT.md)。确认目录有 `confirmation-summary.csv`、`workload-total-times.csv`、`paired-long-e2e.csv`、`preemption-summary.json`、`final-decision.json`；每个 repeat/配置下均生成与同轮 baseline 配对的请求指标、`preemptions.csv`、保护表、完整事件 CSV 和 `timeline.html`。默认失败轮的配对报告明确失败，其余 8 对通过。
- 最新验证：`60 passed, 14 warnings (8.86s)`，证据 `logs/tuning-validation-20261106/`。八个运行时文件与源码快照 hash 一致；固定输入 hash 未变；实验服务与 GPU 计算进程已清理。测量与实验已完成，残余输出差异及满足 long 约束的稳健选参仍未解决。

### 最新：动态 batch 确定性处理与 15 请求最终实验

- 旧内核诊断复跑在同一 `interleaved15-07-long` 复现差异：前三个生成 token 相同，第 4 个从 baseline 的 864 变为 preemptive 的 2532。top logprobs 显示 baseline 多个候选并列，preemptive 的 2532 排到首位。直接推理（不含 P→D 传输）也观察到 solo/mixed 的数值差异；开启 `VLLM_BATCH_INVARIANT=1` 后目标四次控制的 token 与 top logprobs 均相同。旧运行只有 hash，不能事后确认其精确分叉 token；复跑的 token 级证据见 [调查报告](results/determinism-investigation-20261106/INVESTIGATION_REPORT.md) 和 `token-level-findings.json`。
- 修复与工具：`run_pd_pair.sh` / `run_pd_sweep.sh` 默认在 P/D 使用 vLLM 自带 batch-invariant 内核，保留显式 0 供诊断；environment/run metadata 保存配置。采集器记录文本和 token ID，`PD_CAPTURE_LOGPROBS=1` 可额外记录 top-5。单参数两组 sweep 的配对输出不一致时整体返回非零；`summarize_pd_experiment.py` 输出比较、抢占表、保护表与完整事件时间线，支持直接推理参考校验。
- 最终验收目录：[results/sweep-interleaved15-invariant-verified-20261106/](results/sweep-interleaved15-invariant-verified-20261106/)。保持原 8 short/3 medium/4 long 数据集、顺序和 SHA256，concurrency=15、interarrival=100ms、prefix caching=false；两组 batch-invariant=1、top-5 审计开启，抢占配置仍 4/3/0.85/50。runner 正常退出 0，baseline/preemptive 均 15/15，全部文本 hash、token ID 与不经过传输的参考一致。没有把仅 HTTP 200 当作等价性通过。
- TTFT/E2E 均值（ms；baseline/preemptive）：short `12023.799/7872.614`、`12282.129/8089.025`；medium `14477.230/11586.769`、`14776.925/11852.239`；long `14299.549/10666.057`、`14614.328/10988.990`。整轮回放 `18.083/19.121s`；最后 short E2E `14594.524/17694.020ms`，说明均值改善没有使每条请求或整轮时间都改善。新内核与审计改变执行路径，不能直接将这些数据与旧默认内核实验作因果对比。
- 本轮 698 个 chunk 均完成；8 次 short chunk 越过 long，涉及 `02-short→01-long`、`06-short→05-long`、`08-short→07-long`、`12-short→11-long`，每对两次。相对回放起点约为 `1.280/1.305`、`4.703/4.726`、`9.027/9.058`、`13.931/13.959s`。01-long 先触发进度保护（85.12%）；05/07/11-long 先触发累计等待保护（50.652/61.029/57.362ms）。494 行 protected_resume 是受保护 long chunk 的选择记录，不是 494 个独立恢复区间。
- 完整过程见 [EXPERIMENT_REPORT.md](results/sweep-interleaved15-invariant-verified-20261106/EXPERIMENT_REPORT.md)；`preemptions.csv` 保留选中 short、被越过 long、chunk、进度、等待和实际 WRITE 时刻；`protection-decisions.csv` 保留保护原因；`experiment-timeline.csv` 有 4429 条请求/执行/chunk 事件，`request-timeline.csv` 和 `timeline.html` 提供逐请求里程碑和可悬停查看的时序图。原日志、token 证据、metadata、复现命令及源码 hash 同目录归档。
- 中间目录 `sweep-interleaved15-invariant-final-20261106/` 的请求/trace 已成功，但运行期间修改脚本使其收尾退出 2；仅作为中间证据，不冒充最终完整运行。最终 verified 目录从固定脚本重新启动、清理和验证。结论仍限于本固定输入和环境，未逐层逐字节验证 KV 或证明通用场景无错误。

### 2026-11-06 短/中 batch 聚合与 long 8-block 复验

按“短/中请求无需被抢占、每个 batch 聚合新 KV；long 保留可抢占分块并从 4 提至 8 blocks”的假设完成实现和三轮配对复验。保护参数固定 85% / 3 次 / 50ms，未扫描其他配置。实现和逐轮证据见 [实验报告](results/long-short-batch-aggregation-20261106/EXPERIMENT_REPORT.md)。

- 三轮候选均 15/15 HTTP 成功，全部生成输出匹配直接推理参考。每轮 329 个 chunk 全部成功；short 9、medium 4、long 316。long 每轮为 312 个 8-block chunk 和 4 个 1-block 尾块；short/medium 对同 batch 产生的 KV 聚合发送，跨 batch 仍各自产生传输单元。
- 性能假设未通过：配对 baseline/candidate makespan 分别为 `14.253/17.789s`、`14.784/18.669s`、`13.876/16.998s`，候选每轮慢 22.5%～26.3%，均值慢 24.6%。最差单 long E2E 三轮分别 +23.1%、+30.8%、+20.6%。因此不能将该组合标记为性能优化成功。
- 本轮同时改变短/中聚合和 long chunk 尺寸，数据无法分辨两项变化各自的收益/代价。后续应保留为实验候选，不据此宣称“已找到更快配置”；如继续定位，先用相同固定 workload 单独消融两项策略，并同时观察整批 makespan 与 long E2E。

### 2026-11-06 数据采集复查与补齐

复查发现：原 `prefill_ms` 只是 P API 往返时长，包含排队、计算和传输相关等待；流式请求的旧 `decode_ms` 没有值。它们不能回答逐请求计算执行时间。`ttft_ms` 和原 `latency_ms` 已有，本轮增加独立的执行测量并明确口径：

| 所需数据 | 请求级字段 | 测量边界 |
| --- | --- | --- |
| TTFT | `ttft_ms` | 客户端发出请求→收到首个非空文本片段；保留直接首 token 时间戳 |
| P 计算执行时间 | `prefill_execution_ms` | 对该请求参与的各个 P 非空计算 batch，累计 EngineCore 提交执行→取得模型/采样结果的 host 时长 |
| D decode 执行时间 | `decode_execution_ms` | 对该请求参与的各个 D 非空计算 batch，累计相同执行边界的 host 时长，包含首次生成和后续 decode |
| 整个请求耗时 | `e2e_ms`（兼容旧 `latency_ms`） | 客户端开始发送→完整响应结束；失败时为结束到失败的耗时 |
| GPU forward 时间 | `prefill_gpu_forward_ms` / `decode_gpu_forward_ms` | CUDA events 包围每轮模型 forward，累计该请求参与 batch 的设备 stream elapsed 时间，排除采样与 KV 传输等待 |
| D 生成阶段墙钟时间 | `decode_service_ms` | D 首次执行提交→D scheduler 正常结束请求，包含轮间调度/排队 |
| 到达至结束 | `arrival_to_finish_ms` | 回放提交→结束，额外包含回放线程池排队 |

同一 batch 内多个请求共享本轮耗时，每个参与请求记一次，并保存 `batch_requests` / `*_shared_batch_steps`；不能把这些值相加当作 GPU 总占用，也不按 token 比例伪造独占耗时。host 执行时间包含 IPC、执行器排队、输入准备、forward、logits/采样和结果等待，不等于纯 GPU kernel 时长。CUDA forward 是设备事件之间的 elapsed，可能包含 stream 空隙，亦非逐 kernel active time。逐轮累计与首次至末次的 span 均保留。

新产物：`trace-request-metrics.csv` 提供每请求的四项核心指标及 GPU/阶段辅助指标；`trace-execution.csv` 保存逐轮 host/GPU 与 P/D 结束事件；`trace-request-phases.csv` 保存完整阶段字段。原 `requests.csv` 增加直接首 token、结束时间戳及 E2E/到达至结束字段；不能从客户端原始 CSV 单独推算 GPU 耗时。

采集开关为 `PD_MEASURE_EXECUTION=1`，pair 启动默认启用并归档至 environment/run JSON。`collect_pd_trace.py --require-execution` 强制检查成功请求的 TTFT、E2E、arrival-to-finish、P/D host 执行累计和轮数、token 数、连续 step、正常终态与执行顺序；CUDA forward 字段若有 worker 事件则附加记录，但不作为成功门槛。当前 vLLM worker 的 CUDA event 辅助日志没有在真实回放中产生，因此本次比较只使用已实测的 P/D host 执行时间，空的 GPU forward 字段不被解释为零。sweep 工具的 baseline 与 preemptive 均强制 host 执行门槛，并要求 stream 开启。SSE 截断、error 或 abort 不再因初始 HTTP 200 被归为成功。

本轮独立 GPU 验收与测试结果在下方补充；仅最终通过的数据可作为本轮完成依据。


**B1 的单 long、跨 batch、修复后混合请求、连续两批重复负载及 prefix-cache long 已通过 GPU 验收；C 的真实 chunk 边界抢占、次数/进度保护及 50ms 等待保护均有 GPU 轨迹。D 的 registration timeout、握手失败、WRITE 错误、P/D 主动取消和 worker 退出均有 GPU 注入或直接回归证据；取消后新请求恢复和 lease 回收已验收。D KV 收齐 → scheduler 首次非空计算调度 → EngineCore 首次执行提交 → 客户端首 token 已直接记录并在真实 GPU 上验收。2026-11-06 复查补齐逐请求 P/D host 执行时间和请求级 TTFT/E2E，并完成固定 7/1/2 workload 的 baseline/preemptive 对比。CUDA forward 辅助事件未在本次 worker 日志产生，保留为空，不作为 GPU 结论。**

首次调度和执行提交是 host 侧事件，不能解释为 GPU kernel 启动时间。提交到客户端首 token 的区间仍含执行排队、计算、采样和返回路径，不作为纯 GPU 计算时长。当前单机使用同一 monotonic 时钟；跨机器时钟比较不在本次范围内。

### 2026-11-05 已完成：D 首次调度与执行提交

- 最终证据：`logs/presweep-d-phases-20261105/`。固定 workload 为 `workload.pre-sweep.jsonl`，并发 3、到达间隔 250ms、seed 77/78/79、每请求输出 8 tokens；3/3 HTTP 200，三个输出 SHA256 均与 `results/presweep-baseline/requests.csv` 一致。
- 共 187 行 chunk，逐请求序号连续且全部有 P accepted/dispatch/start/complete 和 D receive；每请求 `PD first scheduled`、`PD first model submit` 各出现一次。新版采集器 `--require-complete` 通过，另核对 KV 收齐 ≤ 首次调度 ≤ 执行提交 ≤ 客户端首 token；结果保存在 `verification.json`。
- scheduler 在本轮调度选择确定后记录首次非空计算调度；EngineCore 在普通 step 和 batch queue 两个执行入口提交模型执行前记录首次提交。通过稳定的 `remote_request_id` 和 P/D 方向识别 D，不依赖 registration 时会被清零的 `do_remote_prefill`；空调度不打点，同一请求不重复打点。

| 类型 | prompt tokens | 客户端 TTFT (ms) | D KV→首次调度 (ms) | 调度→执行提交 (ms) | 提交→客户端首 token (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| short | 68 | 380.119 | 3.720 | 0.286 | 27.378 |
| medium | 1107 | 569.067 | 3.371 | 0.232 | 24.881 |
| long | 10645 | 3783.618 | 3.979 | 0.680 | 41.221 |

- 同步修复缓存配置：`ENABLE_PREFIX_CACHING=false` 时显式传入 `--no-enable-prefix-caching`，避免 vLLM 默认启用缓存而 metadata 写 false。最终组的 environment/run JSON 与 P/D engine 启动日志均确认 false。旧组的缓存状态以其 engine 启动日志为准，旧 sidecar 的默认 false 不能独立作为缓存关闭证据。
- 当时六个快照源码与实际安装文件一致（本轮扩展到八个文件），`SHA256SUMS` 已刷新；相关 Python 编译检查、两个启动脚本的 bash 语法检查通过。最终 GPU 组结束后服务已停止、无 GPU 计算进程残留。该组只验证正确性和可测量性，不提供性能收益结论。

### 固定 7/1/2 workload 两组对比

- 固定输入：`workload.sweep-fixed-7s1m2l.jsonl`，SHA256 `0e42aeac323527a70a50214fe84f5d3ecae093ff0a04ea13c683d44b8bdaeac2`；7 short、1 medium、2 long；两组相同请求顺序、seed、100ms inter-arrival、并发 10、prefix caching=false。
- 结果目录：`results/sweep-fixed-7s1m2l/`。baseline 与 preemptive 均 10/10 HTTP 200，逐请求输出 hash 一致。short TTFT mean 为 `357.110/324.038 ms`，E2E `474.907/441.814 ms`；medium `598.786/1338.861 ms`、`715.009/1453.599 ms`；long `7197.172/7272.452 ms`、`7311.865/7386.724 ms`（顺序均为 baseline/preemptive）。
- P/D host 执行和逐请求指标见 `comparison-request-metrics.csv`，聚合/差值见 `comparison.json`；抢占组有 396 行 chunk 轨迹（`dispatch_short=23`、`dispatch_fifo=18`、`dispatch_long=304`、`protected_resume=51`），baseline 关闭抢占，不生成 arbiter chunk 轨迹。
- 该实验是单次固定 workload 描述性对比，medium n=1、long n=2；不能推出稳定 P95/P99 或普遍收益。该历史阶段未进行参数扫描；本次 27 点与三轮确认见第 3 节，chunk FIFO 对照仍未执行。

### medium 尾部 KV chunk 排队问题修复与复验

- 问题：在旧实现中，`_build_stream_chunks()` 只在计算 batch 后生成完整 4-block chunk；medium prompt 共 70 个 blocks，末尾 2 个 blocks 不足一个 chunk，因此一直等到 `request_finished()` 才入队。此时 long 的 chunk 已进入 writer 队列，medium 的最后一段会排在 long 后面，破坏 medium FIFO 屏障，也解释了此前单次实验里 medium TTFT/E2E 异常升高。
- 修复：当 prompt 最后一个 KV block 已就绪时，同批立即将不足 4 blocks 的尾 chunk 入队并标记 `is_final=True`；若 block 数恰好整除，则发送空 final marker。新增 `_stream_final_queued` 防止 `request_finished()` 重复发送 final。push worker 若 prompt-final 传输先完成、scheduler 的 final lease metadata 后到，则暂存完成事件，待 lease 登记后再处理，避免竞态导致提前清理或丢失完成信号。
- 回归：覆盖部分尾 chunk、整块对齐、单 chunk prompt、尾 chunk 跨两个计算 batch、request-finished 重复 final 和 lease 登记竞态。最终测试结果 `57 passed, 14 warnings`；代码快照在 `snapshots/vllm-preemptive-20261103/`，`SHA256SUMS` 校验通过。
- 固定数据集复验目录：[results/sweep-fixed-7s1m2l-tailfix-final/](results/sweep-fixed-7s1m2l-tailfix-final/)。baseline 与 preemptive 各 10/10 HTTP 200，全部输出 hash 一致。medium 的 18 个 chunk（seq=0..17）均成功；尾 chunk seq=17 覆盖 block_start=68、blocks=2、final=true。medium 尾 chunk 于 `11743.467829` dispatch，long 首 chunk 于 `11743.481216` dispatch，确认 medium 尾段没有排到 long 后面。
- 修复后单次复验指标（ms；baseline/preemptive）：medium TTFT `800.946/591.318`、E2E `916.921/707.595`，分别下降 `209.628/209.326`；P host prefill 执行 `118.675/315.768`。该 host 执行字段包括 host 调度/等待，CUDA forward 事件本次未产生，因此不能解释为纯 GPU 计算时间。short TTFT `291.204/292.954`、E2E `409.270/410.755`；long TTFT `6708.201/6506.636`、E2E `6826.558/6623.338`。
- 复验是每种模式一次、medium n=1 的描述性结果，足以确认此固定运行中的传输顺序与成功性，不足以证明稳定性能收益或归因全部延迟变化。新结果保留旧结果目录，不覆盖此前的原始数据。

### 固定 workload 新一轮重复实验

- 2026-11-06 再次使用相同固定输入完成 baseline 和 preemptive 各一轮，结果目录：[results/sweep-fixed-7s1m2l-rerun-20261106-1111/](results/sweep-fixed-7s1m2l-rerun-20261106-1111/)。输入 SHA256 `0e42aeac323527a70a50214fe84f5d3ecae093ff0a04ea13c683d44b8bdaeac2`；请求数及顺序相同，7 short、1 medium、2 long；并发 10、到达间隔 100ms、prefix caching 关闭、chunk=4、max preemptions=3、progress=0.85、wait=50ms。
- 两组均 runner status=passed、10/10 HTTP 200，逐请求输出 hash 一致，严格执行 trace 收集成功。baseline/preemptive 的 TTFT/E2E mean（ms）：short `329.619/314.239`、`446.319/431.642`；medium `801.538/606.365`、`917.626/722.007`；long `7486.491/7499.243`、`7619.019/7607.283`。
- medium P host prefill 执行时间为 `118.633/320.125 ms`，decode 执行为 `251.533/252.211 ms`。host prefill 不是纯 GPU 耗时；CUDA forward 事件仍为空。本次 preemptive trace 共 393 个 chunk：`dispatch_short=20`、`dispatch_fifo=18`、`dispatch_long=304`、`protected_resume=51`。
- 机器可读汇总为 `comparison.json` 和 `comparison-request-metrics.csv`；组内保留 requests、trace、GPU、日志、环境和 status 文件。本轮与上次复验一致观察到 medium TTFT/E2E 较低，但各模式每类请求仍只有一次运行（medium n=1），只能作为固定 workload 的重复观察，不能估计稳定分位数或建立因果结论。

### 15 请求交错 workload 对比

本节保留旧默认内核的首次实验；输出差异的后续处理和最终验收见本节开头“最新：动态 batch 确定性处理与 15 请求最终实验”。

- 新增固定数据集 `workload.sweep-interleaved-8s3m4l.jsonl`，SHA256 `714990980c49534e539a125612ce1c8d36c68ac75a559e4bce93dabcb7ee9e14`。顺序为 short/long/short/medium/short/long/short/long/short/medium/short/long/short/medium/short；prompt 长度分别为 short=128、medium=1107、long=10000 tokens；seed 按请求固定。两组均 concurrency=15、到达间隔 100ms、prefix caching=false，抢占配置 chunk=4、max preemptions=3、progress=0.85、wait=50ms。
- 结果目录：[results/sweep-interleaved-8s3m4l-20261106/](results/sweep-interleaved-8s3m4l-20261106/)。baseline 与 preemptive 均 15/15 HTTP 200，严格 trace 采集通过；但 `interleaved15-07-long` 的响应 hash 在两组间不同（均生成 8 tokens）。因此本轮没有通过所有请求输出一致性检查，数据可以比较运行时延迟，不能宣称两模式输出完全等价；需后续调查动态 batch 下的确定性/数值差异。
- TTFT/E2E 均值（ms；baseline/preemptive）：short `10510.157/7760.496`、`10638.957/7865.127`；medium `12325.034/11715.140`、`12458.526/11832.039`；long `11851.976/10706.901`、`11979.459/10822.481`。本 workload 中三类均值均较低，但仅各一轮，不能视为稳定收益；从到达时间和逐请求分布看存在请求间排队/竞争，单看均值会掩盖个体差异。
- P host prefill 执行均值（baseline/preemptive ms）：short `425.705/383.729`、medium `507.984/464.764`、long `2369.295/2452.524`。该字段含 host 调度和执行等待，不是独占 GPU 时间；CUDA forward 字段为空。preemptive 有 698 行 chunk 轨迹：`dispatch_short=8`、`dispatch_fifo=54`、`dispatch_long=300`、`protected_resume=328`、`preempt_long=8`。逐请求比较与完整原始证据保存在该结果目录；更细节以 `comparison.json` 和 CSV 为准。

### 前序运行证据（保留失败过程；后续验收覆盖的旧待办不代表当前状态）

- 2026-11-05 单 long 重现：`logs/b1-long-20261105d/`，请求 `pd-31-00002`（10471 prompt tokens，8 output tokens）返回 200；router `prefill=1212.787 ms`、`decode=3962.583 ms`。本次运行使用更新后的实际 site-packages 副本；服务已停止。代码已补充 `stream chunk accepted/complete/received` 和 `stream receive complete` 日志。
- 2026-11-05 混合重跑：`logs/b1-mixed-ack-20261105/`，short 返回 200，但 medium/long 在 90 秒超时；D long 日志仍有缺失 sequence。实验中尝试的 ACK 重发方案未通过，已回退；当时待解决的高 chunk 数及 prefix-cache 完成通知问题，已由后续 summary 修复与验收覆盖。
- 2026-11-05 修复：P 在本请求所有 chunk 成功完成后发送 `VLLM_KV_STREAM_DONE:<request_id>:<final_sequence>` 汇总通知；D 收到后一次性确认 `0..final_sequence`。逐 chunk 通知仍保留用于轨迹，汇总通知不替代 DMA 完成判断。直接回归新增 summary 场景后共 9 项通过；GPU 重跑结果见紧接的后续记录。
- 2026-11-05 修复后 GPU 重跑：`logs/b1-summary-final-20261105/`、`results/b1-summary-final-20261105.csv`。short/medium/long 均 HTTP 200；long 11049 prompt tokens，`prefill=1415.022 ms`、`decode=6080.760 ms`、E2E `6095.571 ms`；D 记录 `final_seq=172` 完成，重复 summary 被忽略；服务已停止。
- 2026-11-05 CUDA 同步后 GPU 重跑：`logs/b1-cuda-sync-20261105/`、`logs/b1-cuda-sync-20261105/requests.csv`。short/medium/long 均 HTTP 200，D 分别记录 short `final_seq=1`、medium `final_seq=16`、long `final_seq=172`；P 侧没有 `transfer_failed`/`transfer_exception`，没有 lease 到期日志；服务停止后无 vLLM/GPU 残留。该证据覆盖成功链路和 batch-KV 写入到 WRITE 的同步边界，仍不覆盖注入失败/客户端取消。
- 2026-11-05 C 轨迹 GPU 重跑：`logs/c-trace-20261105/`、`logs/c-trace-20261105/requests.csv`。三请求均 HTTP 200；P 日志出现 `decision=preempt_long`，记录 long request、抢占次数 1、成功进度 0、累计等待 0 ms；D 收到 short `final_seq=1`、medium `final_seq=16`、long `final_seq=172`，无传输失败或 lease 到期。该样本证明 chunk 边界发生过短请求越过 long；后续持续短请求保护验收见下文。
- 2026-11-05 持续 short 保护 GPU 重跑：`workload.c-protect.jsonl`、`logs/c-protect-20261105/`。long 后跟随 8 个 short，到达间隔 120ms、并发 9；9/9 HTTP 200。long `cmpl-pd-91-00002-0-bec35104` 第 1/2/3 次被插队时进度约 26.02%/26.63%/26.63%，累计插队等待约 0/19.51/39.23ms；达到次数上限后恢复，最终恢复轨迹 wait=58.83ms，原因含 `max_preemptions` 和 `wait`。D 收到 long `final_seq=165` 及全部 short final；无传输失败或 lease 到期。该样本验证次数保护及阈值后的仲裁恢复；GPU 等待阈值夹逼见 `logs/c-wait-50ms-20261105/`。85% 精确逻辑边界及 prefix-cache GPU 验收见相邻记录。
- 2026-11-05 C 的 50ms 精确逻辑边界：可控时间回归确认累计等待 49.999ms 时仍选择 short，达到 50.000ms 的下一个仲裁边界选择 long，轨迹原因含 `wait`。回归首次暴露浮点累加略低于阈值的问题；等待比较加了 `1e-9 ms` 容差后通过。GPU 紧邻边界控制回放见 `logs/c-wait-50ms-20261105/`。
- 2026-11-05 prefix-cache 进度口径修正：arbiter 使用 D registration 提供的未命中 block 数作为稳定分母；chunk 的成功进度只统计超出 D prefix offset 的块，D 已命中的块和空终止 marker 不增加进度。registration 先到和 chunk 先到两种入队路径均在入 arbiter 前标注。真实 D prefix-cache 命中的 GPU 轨迹见 `logs/c-prefix-85-20261105/`。
- 2026-11-05 D registration 超时修复：watchdog 到期将 request ID 放入 connector metadata 的 `push_failed_receives`，D worker 把它转换为现有 `_recv_failures` 和完成结果；若有活动接收 handle，保留 handle 到 NIXL 安全完成边界。GPU 混合回归和后续 0.5 秒 watchdog 超时注入见 `logs/d-timeout-metadata-smoke-20261105/`、`logs/d-registration-timeout-20261105/`；超时 long 失败后紧接 short 成功。
- 2026-11-05 D registration 超时修复：watchdog 到期将 request ID 放入 connector metadata 的 `push_failed_receives`，D worker 把它转换为现有 `_recv_failures` 和完成结果；若有活动接收 handle，保留 handle 到 NIXL 安全完成边界。新增 scheduler 和 worker 回归，三测试模块共 19 项直接调用通过。GPU 注入 `logs/d-registration-timeout-20261105/` 使用 0.5 秒 watchdog：long 返回 HTTP 500，D 日志明确记录 timeout，紧接 short 返回 HTTP 200；服务退出后 GPU 无残留。
- 2026-11-05 prefix-cache 85% GPU 验收：`logs/c-prefix-85-20261105/` 启用 prefix caching，先在 D 预热 8192-token 前缀，再提交 11049-token long。P arbiter 记录实际缺失分母 `total_blocks=179`，在 `152/179=84.916%` 后下一仲裁边界进入 `protected_resume`，`156/179=87.15%`，保护原因为 `progress`；请求 HTTP 200，D 收齐 `final_seq=172`，服务和 GPU 已清理。
- 2026-11-05 router abort GPU 验收：启动 P/D 时启用 `--enable-scale-out` 与共享 API key；故意令 P endpoint 连接拒绝，router 收到 P 失败后对 D 调用 `/inference/v1/abort_requests`，D 返回 HTTP 200；随后同一 D worker 的 short 请求 HTTP 200。结果在 `logs/d-abort-scaleout-20261105/`，服务退出后 GPU 无残留。启动脚本新增可选 `VLLM_API_KEY`、`ENABLE_SCALE_OUT`、`ENABLE_PREFIX_CACHING`。
- 2026-11-05 worker 退出 GPU 验收：`logs/d-worker-exit-fixed-20261105/`。long 请求传输期间对 P API server 执行 SIGKILL，router 将 `RemoteDisconnected` 转为 HTTP 502，并调用 D 的 `/inference/v1/abort_requests`，D 返回 HTTP 200；D 进程未崩溃或永久等待。重启完整 P/D 组后 short `pd-worker-exit-recovery-short` 返回 HTTP 200，GPU 与服务已清理。新增 router 断连归一化回归及握手失败回归后，三测试模块共 26 项直接调用通过。
- 2026-11-05 WRITE 错误注入验收：P writer 仅在设置 `NIXL_TEST_FAIL_STREAM_CHUNK=<base_request_id>:<sequence>` 时，在握手完成、提交 WRITE 前注入失败；默认关闭。`logs/d-write-failure-verified-20261105/` 中 long `pd-7-00002` 的 `seq=0` 注入后 router 返回 HTTP 500，D 记录 `stream chunk received seq=1` 后因 KV load failure 返回 500；同一 P/D 组随后 short `pd-write-recovery-verified` 返回 HTTP 200。P 无 EngineCore 崩溃，失败请求未出现 lease expiry；等待超过 TTL 后指标仍为 `KV expired reqs=0`。新增失败竞态和取消后 chunk 清理回归后，三测试模块共 26 项直接调用通过；快照 SHA256 已更新。
- 2026-11-05 D 主动取消 GPU 验收：`logs/d-active-cancel-verified-20261105/`。对正在传输的 24002-token long 调用 D `/inference/v1/abort_requests` 返回 HTTP 200，router 以 `finish_reason=abort` 返回 HTTP 200；D worker 发出 `VLLM_KV_STREAM_CANCEL`，P worker 按 producer base request ID 匹配哈希 ID 并停止排队 chunk。取消后的 376 个迟到 chunk 均以 `success=False` 完成，不产生 transfer ERROR；同一 P/D 组的恢复 short HTTP 200。等待超过 30 秒 lease TTL 未见取消请求的 expiry warning，P 指标 `Num KV expired reqs=0`；退出后无 vLLM/EngineCore 进程或 GPU 残留。
- 取消竞态修复：P 接纳 scheduler 的 final chunk 时立即记录 producer 已到终态；取消清队列后，最终完成事件仍能触发失败 lease 的幂等回收，而活动 DMA 仍须到安全完成边界。取消后的 chunk 拒绝按正常 INFO 记录。snapshot、`cc` site-packages 与测试一致；三个测试模块共 32 项直接调用通过，相关文件 `py_compile` 通过，GPU 取消轨迹验证 `KV expired reqs=0`。
- 当时 B1 结论：修复前第一批并发 medium 曾超时；修复 deferred WRITE 覆盖后，新的 short/medium/long 混合回归三项均 HTTP 200，medium `seq=0..16` 全部 P-side complete、D `final_seq=16`，long `final_seq=172`，P 侧传输失败为 0。随后同一有效 P/D 服务组连续两批 workload 也各 3/3 HTTP 200；服务停止后 GPU 无残留进程。当时 B3 的后续请求、失败路径和资源清理待办，已由本节后续失败恢复与取消验收覆盖。
- 本轮失败路径修复：流式 chunk 失败或取消会清空 arbiter、deferred/registration 状态；失败状态等待 scheduler 的 final/cancel 边界、无活动 WRITE 且 lease 元数据已登记后，通过幂等队列结束 P-side lease。D 收到 `VLLM_KV_CHUNK_FAIL`，不会把缺失 chunk 当作成功。失败 chunk 不再计入 long 的成功进度。直接回归覆盖失败竞态、重复失败、取消后 chunk 丢弃、缺失 chunk 和现有完成路径；GPU 证据见 `logs/d-write-failure-verified-20261105/`，等待超过 TTL 后 `KV expired reqs=0`，后续 short 成功。
- 本轮写入边界修复：P worker 在把本 batch 的 streaming chunk 交给后台 writer 前同步当前 CUDA stream，避免 WRITE 与本 batch KV kernel 并行读写同一源块。该同步只作用于实验 streaming path；快照已同步，`SHA256SUMS` 已更新。

- 2026-11-05 第一批重复 workload 的失败证据：P 接受 medium `cmpl-pd-7-00001-0-9a0427eb` 的 `seq=0..16`，D 只收到 `seq=0..15`，P 没有对应的 `stream chunk complete`；D 保持一个 KV fetch in progress，medium 超时。long 同批收齐 `final_seq=172`，第二批 short/medium/long 全部 200。
- 根因：writer 只有一个 `_stream_active` 槽，但握手完成回调可并发把多个 chunk 放入 `_stream_deferred_inbox`；writer 在已有 WRITE 未完成时连续启动 deferred chunk，后启动的 chunk 覆盖前一个 chunk 的句柄和完成状态。
- 已修复：deferred stream chunk 只有在 `_stream_active is None` 时才启动；已有在途 WRITE 时保留在队列，避免句柄覆盖。修改已同步到实际 `cc` site-packages 和快照；三份 SHA256 已重新计算。`py_compile` 和 10 项直接 Python 回归检查通过，其中新增单槽 deferred 回归覆盖该缺陷。
- 修复后 GPU 回归：`logs/b1-deferred-fix-20261105/`、`results/b1-deferred-fix-20261105.csv`。同一三请求 workload 在 45 秒超时下 short/medium/long 均 200；P 侧 medium 17/17 chunk 完成、long 173/173 chunk 完成，D 记录 medium `final_seq=16`、long `final_seq=172`；无 `transfer_failed`，服务已关闭且 GPU 进程已清理。
- 连续两批 GPU 回归：`logs/b1-deferred-repeat-20261105/`、`results/b1-deferred-repeat-first-20261105.csv`、`results/b1-deferred-repeat-second-20261105.csv`。同一有效 P/D 服务组连续执行两批，每批 short/medium/long 均 3/3 HTTP 200；两批 medium 均 P-side `seq=0..16` 共 17/17 complete，D 均 `final_seq=16`；两批 long 均 P-side `seq=0..172` 共 173/173 complete，D 均 `final_seq=172`；无传输失败，第二批全部请求约 2.3 秒完成。重复启动时误留下的额外服务组因显存不足退出，该启动错误不属于有效 workload 结果；最终所有进程和 GPU 占用均已清理。

- 日志：`logs/smoke-preemptive-lease-fixed/prefill.log`、`decode.log`。
- 结果：`results/smoke-preemptive-lease-fixed.csv`。
- 输入：`workload.test.jsonl`，short 69 tokens、medium 1062 tokens、long 11049 tokens，并发度 3。
- short/medium 返回 200；long 在 D 等待 KV 后超时。P 有 137 次成功 chunk transfer、0 次 transfer failure，但仍有 send lease 到期日志。
- baseline 对照：`results/smoke-baseline-20261103.csv`，三请求均成功。
- 测试服务已停止，避免继续占用 GPU。

上述早期失败 smoke 只能说明部分 WRITE 成功；完整接收和回收已由后续成功记录与失败恢复记录验证。单组 smoke 均不能用于宣称短请求延迟已改善。

本轮新增处理：

1. 跨 batch streaming 的 `local_total_blocks` 改为请求最终 prompt 的总 KV block 数，仅作为 P/D prefix offset 和进度分母；传输 chunk 大小仍由启动参数 `chunk_size_blocks=4` 控制，各 batch 仍按当时实际就绪的 blocks 生成 chunk。
2. 对无 WRITE 的 chunk（D prefix cache 命中或空终止 marker）统一显式发送 `VLLM_KV_CHUNK` 通知，保证 D 收到连续序号后再判定 final。
3. 增加 chunk 接收/完成/最终收齐的 request、sequence、block 和 final 日志；同步快照 SHA256。
4. 增加请求级 `VLLM_KV_STREAM_DONE` 汇总通知，避免高 chunk 数下逐 chunk completion notification 丢失导致 D 永久等待；summary 只在 final chunk 成功边界发送，P 仍等待全部 completion queue 项后释放 lease。

1. FlashInfer 原来误用 CUDA 11.8 nvcc，不识别 `--compress-mode=size`。启动脚本现优先使用环境内 CUDA 13.4 nvcc。
2. CUDA wheel 缺少链接目录别名；已补 `cu13/lib64 -> lib`、`lib/stubs/libcuda.so` 链接。
3. completion 清理的 tuple/set union 类型错误，已统一转 set。
4. D 接收 chunk 完成被误报成发送完成，导致 scheduler 释放仍在运行的 D 请求块，已修正。
5. P stream 完成后清除发送 lease，已补 worker 清理和回归检查；早期仍出现的 lease 到期已通过后续失败/取消竞态修复，验收见 `d-write-failure-verified` 和 `d-active-cancel-verified` 记录。

中间尝试过的活动 completion ID 过滤会丢弃正常接收完成，已撤回。不要恢复该过滤，也不要删除 scheduler 的断言来掩盖问题。

- 2026-11-05 pre-sweep streaming/trace smoke：`logs/presweep-trace-20261105/`。short/medium/long 3/3 HTTP 200；与 baseline 对比，三条输出 SHA256 全部相同。客户端 TTFT 为 348.519/573.648/3679.331 ms；每请求均记录 P final chunk accepted/complete、D KV receive complete 和首 token 时间。采集器合并 187 行 chunk 轨迹，187 个 chunk 均有 D receive 事件；P 接受至 D KV 完成分别为 272.561/379.649/2414.051 ms，D KV 完成至首 token 为 30.440/28.523/43.331 ms。GPU CSV、workload checksum 和 metadata 同目录保存。
- 2026-11-05 50ms 等待保护 GPU 控制回放：`logs/c-wait-50ms-20261105/`，chunk=1、max preemptions=100、progress threshold=1.0，仅观察 wait 保护；9/9 HTTP 200，共 710 行 chunk trace。long 的第 8 次 short 插队时累计等待 49.907 ms，仍允许 short；下一仲裁边界 long 以 `protection_reasons=wait` 恢复，累计等待 56.690 ms、进度 17.70%。两次决策间隔 6.783 ms 包含当前不可中断的 1-block short chunk 完成和 host 处理；GPU 轨迹因此夹住阈值并确认保护在 chunk 边界生效，不代表传输可在阈值处被中断。当时版本采集器 `--require-complete` 对该组和前述 3 请求 smoke 均通过；旧日志没有本轮新增的调度/提交事件，不能直接通过新版严格检查。
- 2026-11-05/06 指标和 runner 改进：P 日志为 accepted/arbiter decision/chunk start/complete 添加同源 monotonic 时间及扁平仲裁状态字段，D 日志提供每 chunk receive 和 KV receive complete 时间；新增 `collect_pd_trace.py` 输出请求阶段、逐请求 metrics 与 P/D execution CSV。sweep runner 通过 cc Python 执行 router/load/collector，成功请求缺少 P/D host 执行事件会令该组失败；另写每组 `status.json` 和全局 `sweep-summary.json`，CSV 汇总保留。`run_pd_pair.sh` 写 `environment.json`，保存 Torch/CUDA/NIXL/vLLM/Python/nvcc/driver、GPU UUID/映射及运行配置；load metadata 记录 seed、workload SHA256、请求顺序、设备/端口和参数，成功与失败时延分开计算。固定两组实验的 environment sidecar、comparison 和 manifest 已生成；本次新增 27 点及确认见第 3 节。

## 4. 执行顺序与验收门槛

### A. 环境与可重复启动（基本通过）

- [x] Torch 可访问两张 GPU；nixl/nixl-cu13 1.5.2 可导入。
- [x] P/D 启动，FlashInfer JIT、NIXL agent、KV registration 和 health 成功。
- [x] 保存工具链兼容处理和环境版本；`run_pd_pair.sh` 为每组写 `environment.json`，包含 Torch/CUDA/NIXL/vLLM/Python/nvcc、驱动/GPU UUID、设备映射和运行参数。
- [x] 在 cc 环境安装 `pytest==9.2.0`，写入 `requirements-test.txt`；真实 pytest 52 项全通过，environment sidecar 同时记录 pytest 版本。
- [x] prefix caching 启用/关闭均显式传 CLI 参数；最终组的配置、metadata 和 engine 日志一致。

当前 CUDA wheel 的 compiler 为 13.4，headers/runtime 为 13.0。脚本临时使用 `CCCL_DISABLE_CTK_COMPATIBILITY_CHECK`；启动 smoke 成功不代表所有 kernel 组合已验证。后续可配套 CUDA 13.0 工具链，或在完整正确性测试中覆盖该实验配置；不必重装整个 Torch/vLLM。

### B. 真实传输正确性（当前固定范围内已完成预验收，保留检查口径）

**B1：long 跨 batch 与完成生命周期。**

1. 用一个跨多个 prefill batch 的 long 单独重现，再重现同一三请求并发。先用短超时暴露问题，不反复等待 300 秒。
2. 按 request_id + chunk_sequence 记录 batch 就绪、入队、registration、握手、WRITE 提交/完成、D 通知/final、P lease 设置/删除和块释放。
3. 核对 `_build_stream_chunks` 的 `local_total_blocks`：使用请求最终 prompt 的总 KV block 数作为映射分母；每个 batch 仍只提交当时实际就绪的 blocks，chunk 大小由 `chunk_size_blocks=4` 控制。
4. 检查无需 WRITE 的 chunk（如 D 已缓存的块）和大量 WRITE completion 是否都能通过逐 chunk 轨迹及最终 `STREAM_DONE` 让 D 完成；否则 final 已到也可能永久等待。
5. 检查握手尚未完成时是否又 dispatch 新 chunk；deferred 回调是否覆盖 `_stream_active` 或造成多个不可追踪的在途 WRITE。（已确认并修复：deferred chunk 在已有 active WRITE 时重新排队。）
6. final 只提交一次；P 完成只上报一次；全部前序 chunk 和 DMA 完成之前不得释放源块。
7. 核对完成通知与 metadata 提交 lease 的先后关系，防止完成之后重建 lease。正常成功不能依赖 TTL 回收，也不能向已移除请求再次报告完成。

**B2：batch 完成到 WRITE 的安全边界。**

- scheduler 在计算前准备 metadata；核对 worker pre/post-forward 与 CUDA event/同步，证明 WRITE 真正提交时本 batch 的 KV 写入已完成。
- 跨 batch 每个 block 只提交一次，无漏块、重复、越界和被复用源块。
- 覆盖 prefix cache 全/部分命中、块边界 prompt、尾部不完整块和空 final。

**B3：验收。**

- baseline/preemptive 的短、长、混合请求全部成功，确定性设置下与参考输出比较。
- 三请求并发全部成功，重复负载及后续新请求仍成功；结束后无残留 lease、句柄或请求状态。
- 对实际缺陷补回归检查，再用 GPU 重跑。基础测试和一次成功请求不能代替该门槛。

### C. 短请求插队与三项防饥饿规则

逻辑场景使用可控时间，GPU 场景保存真实轨迹，不能仅用 HTTP 到达顺序猜测传输顺序。

| 场景 | 验收行为 |
| --- | --- |
| long 在传，short 后就绪 | 当前 chunk 完成后 short 先传，不中断当前 WRITE |
| short 持续到达 | 达最大抢占次数后 long 恢复，不持续被插队 |
| 85% 边界 | 低于 85% 可插队，达到/超过 85% 后禁止，分母稳定 |
| 累计插队等待 | 达阈值后的下个仲裁边界恢复 long，多次区间累加 |
| medium 屏障 | short 不越过 medium，medium 保持 FIFO |
| 重复/缺失/失败 chunk | 缺序号或失败不得报告完整成功，重复不得重复释放 |

至少输出一条 long/short chunk 时间线，标注抢占数、成功进度、累计等待、保护原因。主实验保护进度固定为 0.85。

### D. 取消、超时、失败和安全清理

- [x] 覆盖 P/D 取消、registration 超时、握手失败、WRITE 错误和 worker 退出。D 主动取消的 GPU 轨迹见 `logs/d-active-cancel-verified-20261105/`；其余注入/回归证据见本节运行记录。
- [x] router 一侧失败时，另一侧通过 vLLM abort API 取消；decode 等待有有限超时。仅取消 Python Future 不等于取消上游请求，故保留显式 abort。
- [x] 取消/失败回收 arbiter、队列、registration、chunk 状态、句柄和 lease；取消竞态下仍等待 scheduler final 边界及活动 DMA 安全完成后释放。
- [x] 代表性失败后发送新请求成功，无无限重试、永久等待或误报成功；WRITE 错误、registration timeout、worker 退出和 D 主动取消均有恢复请求证据。

### E. 指标与实验工具

- [x] 流式响应记录客户端 TTFT；router 记录首 token 时刻和 TTFT，回放器记录客户端 TTFT。
- [x] 每请求保存类型、token 数、HTTP/错误原因、E2E、TTFT、P final chunk accepted/complete、D KV 收齐、首次非空计算调度、首次执行提交和客户端首 token；分段输出 KV→调度、调度→提交、提交→客户端首 token，并保留 KV→首 token。调度/提交为 host 侧事件，不标称纯 GPU 时长。
- [x] 新增每请求 P/D host 执行累计、可选 CUDA forward 累计、执行 span、D 生成服务区间，以及明确的客户端 E2E/结束时刻；批处理共享时间不做独占归因。正式比较使用 host 执行字段，CUDA forward 缺失明确保留为空。
- [x] `collect_pd_trace.py --require-complete` 对成功请求要求 P final accepted/complete、D KV 收齐/final sequence、D 首次调度和提交均存在，且 KV≤调度≤提交；倒序/缺失导致失败。最终三请求组通过，额外验收首次事件恰好一次及提交≤客户端首 token。
- [x] 每 chunk 记录 P accepted/arbiter dispatch/start/complete、D receive 时刻、块数、队列/传输时长、long 抢占次数/进度/累计等待/保护原因和成功标志。当前以 KV block 数为传输工作量单位，未单独换算字节。
- [x] 采样 GPU 利用率/显存；run metadata 与 environment sidecar 保存模型、软件/驱动版本、参数、seed、请求顺序、设备/端口、workload checksum 和日志路径。
- [x] sweep 为 P/D、side-channel 和 router 端口设置默认值；使用 cc Python；健康等待覆盖首次加载/JIT，并检查子进程退出。每个参数点记录失败状态，全部点完成后以非零状态汇总失败。
- [x] 下一组前等待 worker/GPU 释放；清理后等待 PID 消失，保留 INT/TERM trap，只清理本组进程。固定 7/1/2 workload 的两组实测已完成；更大规模多点 sweep 不属于本轮固定实验范围。
- [x] 持久化每组 status JSON 和全局汇总 JSON/CSV；HTTP 或必需 trace 失败会使参数点失败。成功和失败/超时延迟分开统计。
- [x] workload 保存固定类别比例、请求顺序和 seed；load CSV 区分回放线程池排队时间与 router 实际收件时间，并记录 inter-arrival 设置。
- [x] sweep 工具对 baseline/preemptive 均加入 `--require-execution`，成功但缺少四项指标或 P/D host 执行事件会令组失败；拒绝关闭执行计时或流式 TTFT 后启动。
- [x] 固定数据集两组对比及 medium 尾部 chunk 修复后复验已完成：`workload.sweep-fixed-7s1m2l.jsonl`，7 short、1 medium、2 long；baseline 与 preemptive 各 10/10 成功，最终结果见 `results/sweep-fixed-7s1m2l-tailfix-final/comparison.json`，修复前结果保留在 `results/sweep-fixed-7s1m2l/`。
- [x] 固定 15 请求的保护参数 27 点扫描、默认/候选/新 baseline 各三轮确认及总时间统计已完成。
- [x] 定位 long 输出不一致的握手越序/提前 summary 竞态，修复并完成 baseline/默认/两候选各三轮验证；完整修复版本 180/180 输出一致，63 项回归通过。
- [ ] 128KB/C8 最新三轮已通过正确性和逐 long 约束；更大样本稳定分位数、chunk FIFO 对照与跨 workload 泛化尚未完成。
- [ ] 按“重叠测量→同步 fence 消融→有限 in-flight 窗口→chunk 粒度比较”的顺序调查性能瓶颈；每阶段正确性和三轮配对性能均通过后才采纳优化。

### F. 正式对比与参数扫描

1. B/C/D 验收通过且 E 指标可用后，固定模型、设备、生成参数、缓存状态、seed 和到达过程。
2. 已按固定 workload 比较整请求 baseline 与 preemptive；本次未增加 chunk FIFO 对照。
3. 先小规模检查数据，再构造 long 传输期间 short 持续到达的负载；区分计算排队与传输排队。
4. 本轮按用户要求固定 chunk=4，扫描进度、最大抢占次数和累计等待阈值（27 点）；未扫描 chunk。已修复确认阶段 long 输出竞态并复测两候选和默认，结果见当前执行位置；原 27 点未整体重扫。
5. 比较各类型 TTFT/E2E 的 P50/P95/P99、成功率、吞吐、long 最大等待和资源开销。每类只有一个请求的 smoke 没有分位数统计意义。
6. 重复关键组合，归档波动范围、失败样本、配置、日志和结论到 results/。
7. 下一轮性能优化先执行本节“性能差异分析与下一步计划”的测量与消融，不先改参数默认值或重做原 27 点扫描；以 CUDA stream/event 依赖保证 KV 安全可读，避免无证据移除同步。

## 5. 后续执行约定

- B1/C/D 和 sweep 前指标、采集、复现与失败汇总工具均已完成当前计划内的预验收；固定 7/1/2 workload 的两组实验已完成。后续更大参数扫描启动前仍需复核 workload、run metadata 和各组 trace 完整性。
- 每完成验收点，更新本文状态和证据路径；只勾选实际通过项目。
- 运行 connector 在 `/home/ubuntu/miniconda3/envs/cc/lib/python3.11/site-packages/vllm/distributed/kv_transfer/kv_connector/v1/nixl/`；打点还修改同一 site-packages 下的 `vllm/v1/core/sched/scheduler.py` 和 `vllm/v1/engine/core.py`。修改后同步快照并更新 SHA256SUMS。快照含四个 connector 文件、两个 core 文件及四个 worker 文件（`gpu_model_runner.py`、`pd_execution_timing.py`、`gpu/kv_connector.py`、`gpu/model_runner.py`），共 10 个文件，不是完整 vLLM 仓库；目录日期是最初建档日期，内容已更新到本轮。
- 当前 smoke 使用 HTTP 8110/8210、NIXL 5610/5611、router 8010；默认脚本是 8100/8200、5600/5601、8000，勿混用。启动核对 PID/health，结束关闭本组进程。
- 最终验收：混合请求稳定完成，chunk 边界插队有证据，三项保护生效，失败后资源安全回收，实验可重复并有完整结果。

## 6. 2026-11-03 历史暂停结论

本节仅保留 2026-11-03 暂停时的历史分析，所列 long 超时与 lease 问题已有后续修复和验收；当前入口为第 3 节，勿按本节旧待办重新开始。

### 已完成的任务

- 固定 TP=1、PP=1 的 P/D 启动链路已建立，P/D 各使用一张 RTX 4090。
- `nixl==1.5.2`、`nixl-cu13==1.5.2` 可导入；NIXL agent、兼容握手、KV registration 和 FlashInfer JIT 均已成功。
- baseline 整请求 push 已验证：同一 mixed workload 中 short、medium、long 均成功，结果在 `results/smoke-baseline-20261103.csv`。
- preemptive 单个 short 请求已完成真实 P->D KV transfer 并返回生成结果。
- short/medium/long 分类、chunk 仲裁、medium 屏障、最大抢占次数、85% 进度保护和等待阈值已经接入代码，基础逻辑检查通过。
- router、workload 生成器、回放器和逐请求 CSV 已可用。
- 已修复启动工具链、tuple/set completion union、D 接收完成误报发送完成等运行时问题。

### Long 超时的当前证据

最近一次 preemptive mixed workload 使用 `workload.test.jsonl`：short 69 tokens、medium 1062 tokens、long 11049 tokens，并发度 3。结果在 `results/smoke-preemptive-lease-fixed.csv`：short/medium 返回 200，long 返回 502，等待约 40 秒。

日志显示：

- P 侧有 137 次成功 NIXL WRITE，transfer failure 为 0。
- D 侧仍停留在 `KV fetch: 1 in progress`，没有将 long 请求判定为完整接收。
- P 侧之后出现 long request lease 到期，并提示在 lease 到期前被 0 个 remote worker 完整取回。

因此当前不是“抢占函数完全没有实现”，也不能把 long 超时归因于 NIXL WRITE 全部失败。当前证据表明：部分 chunk 已传输，但 D 没有收到完整、连续的 chunk 序列或最终完成通知；D 等待完整序列，router 等待超时后返回 502。

### 当前最可能的根因方向

长请求跨多个 prefill batch 时，chunk bookkeeping 尚未证明正确：

1. `_build_stream_chunks()` 每个 batch 使用当时的 `len(blocks)` 作为 `local_total_blocks`，而请求结束时 `_append_final_stream_chunks()` 使用最终 block 列表补尾部 chunk。
2. `block_start`、chunk sequence、请求级总 block 数以及 P/D prefix offset 必须跨 batch 保持一致；当前实现尚未用运行时轨迹证明这一点。
3. 如果出现缺失序号、重复序号、源/目标 block 偏移错误，D 的完成判断会一直等待。D 端要求收到 `0..final_sequence` 的全部序号后才结束接收，少一个 chunk 就不会完成。
4. P 侧 lease 清理、stream 完成通知和 scheduler 的 request 生命周期还需要按同一个 long request 逐事件核对。

这次 workload 中 medium 位于 short 和 long 之间；medium 是设计中的插队屏障，因此该样本不能单独证明 short 成功越过 long。它只证明 preemptive 服务能让部分请求完成，不能证明抢占策略已经端到端验收。

### 下次第一步

先不要跑正式 sweep，也不要删除 scheduler assertion 或用“过滤未知 completion ID”掩盖问题。单独运行一个跨多个 prefill batch 的 long 请求，给每个 request 记录：

- P batch 就绪时间、入队时间和 `chunk_sequence`；
- registration 匹配、握手、WRITE 提交和 WRITE 完成；
- D 收到的每个 chunk 序号、final 标志和完整性判断；
- P lease 设置、stream completion、lease 删除和 block 释放。

先确认 long 的序列完整性、block 映射和完成生命周期，再恢复 short/medium/long 混合 workload，最后才验证短请求是否在 chunk 边界插队以及三项防饥饿保护。
