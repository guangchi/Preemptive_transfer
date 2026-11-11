## 实验实现与证据

- SGLang 0.5.21、sglang-router 0.3.2、Torch 2.13.0、NIXL 1.5.2；vLLM 0.32.0。模型为本地 Llama-3.1-8B-Instruct，单机双 RTX 4090：Prefill 绑定物理 GPU0，Decode 绑定物理 GPU1，每个进程 TP=1、PP=1。这里的“单卡”仅指每个 P/D 进程各用一张卡，不是整套实验只用一张卡。
- SGLang 使用官方 NIXL 传输实现，后端指定 UCX。独立 smoke 日志确认 UCX 1.23.0、rma_am(tcp/lo)、cuda_copy；本次为单机 loopback TCP。
- 独立 smoke 使用原测试集中的一条 long 和一条 short，先预热再采样。10000-token long 分为5个计算及发送 chunk（2048×4+1808 tokens；page 数128×4+113），short 为1个 chunk。包含预热共12个 chunk 全部传输成功。
- smoke 同时开启 P/D KV checksum，未出现 mismatch，所有请求正常完成；正式性能测量关闭 checksum 与观察 trace。smoke 时间不混入正式结果。
- 为保持原 token ID 输入，SGLang 使用官方 Python MiniLB（--mini-lb）：Rust router 0.3.2 对 token ID prompt 返回422。MiniLB 官方定位为调试用途；本实验1P1D无需节点选择，但路由、SSE交付方式仍可能影响客户端 TTFT。TTFT是客户端收到首个非空文本事件的时间，不等于GPU首token算完的时间。
- vLLM baseline 关闭自研传输开关，沿用 NixlPushConnector 的整请求发送路径；完整方案开启三个模块，其他参数见environment.json。两组均保留原实验的batch-invariant计算设置与执行计时日志。
- SGLang观察补丁仅增加可关闭的观测入口，没有替换调度或传输算法；正式测量不启用事件记录。补丁、依赖版本及启动脚本已保存在reproducibility目录。
- SGLang采用独立venv叠加原环境，pip check仍报告可选依赖缺失和若干版本约束不符；本次Llama/FlashInfer/NIXL路径实际运行成功。这是当前部署条件下的结果，不能称为全部依赖严格匹配的官方标准环境。详情见reproducibility/sglang-pip-check.txt。
- 三次重复发生在各自同一组已预热服务中，不包括启动、模型加载和预热。仅3次重复，均值和样本标准差描述本次观测，不代表广泛工作负载上的统计结论。
- failed-*、invalid-*、sglang-smoke和sglang-smoke2目录不纳入正式统计；唯一有效功能验证目录为sglang-smoke3。

## 抢占与重叠核查

| 正式轮次 | 实际抢占次数 | 被抢占的long请求序号 | 当时已传输比例 |
| --- | ---: | --- | --- |
| run-1 | 2 | 01、11 | 81.92%、80.64% |
| run-2 | 2 | 01、11 | 81.92%、80.64% |
| run-3 | 2 | 01、11 | 81.92%、80.64% |

每次正式实验均实际发生2次抢占，第01和第11号long各1次。配置中的3次是每个long允许的上限。long整体仍然更快，说明提前发送等收益可以超过抢占等待成本；本次比较没有单独隔离三个模块的贡献。

| smoke long chunk | 页数 | 发送开始到完成 ms | 与P GPU forward窗口交集 ms |
| --- | ---: | ---: | ---: |
| 0 | 128 | 761.222 | 758.561 |
| 1 | 128 | 122.880 | 122.880 |
| 2 | 128 | 124.249 | 12.693 |
| 3 | 128 | 124.988 | 0.000 |
| 4 | 113 | 110.511 | 0.000 |

GPU CUDA event与发送开始/完成记录确认存在计算和传输重叠。该证据描述观测时间窗口交集，不能解释为区间内每一毫秒都在进行物理数据传输。前几个chunk与后续计算重叠，最后两个chunk仍有传输尾部。首chunk耗时较长是本次smoke的观测，尚未隔离底层原因，不能据此归因跨框架性能差值。

SGLang与vLLM输出不同的输入为interleaved15-06-short、interleaved15-07-long。两边各自3轮输出稳定；跨框架输出差异的原因尚未单独定位，不能仅凭checksum smoke认定所有输入的数值结果完全等价。

输入审计见audit.json：三组各45条正式请求，HTTP状态、输入长度、输出长度和结果哈希均核对通过。完整方案与SGLang的逐轮输入副本和原测试集逐字段一致（仅request_id轮次后缀不同）；baseline直接读取原workload，manifest中的文件哈希与当前测试集一致。
