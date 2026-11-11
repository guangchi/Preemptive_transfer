# 短中请求按 batch 聚合与 long 8-block 实验

历史结果说明：下文的 14.304s / 17.819s 来自传输分段修复之前，保留作为当时的失败证据。后续双方均使用 128KB、候选使用 event 后，三轮结果为 baseline 6.295s、候选 5.601s；详见 [后续根因报告](../ucx128k-w1-c8-confirm-20261106/ROOT_CAUSE_REPORT.md)。不同版本的数值不能混成同一组配对比较。

日期：2026-11-06。固定保护参数 **85% / 3 次 / 50ms**，15 请求交错 workload（8 short、3 medium、4 long），baseline 与候选各三轮配对运行。候选策略为短/中请求将同一计算 batch 新就绪的 KV 合成一个传输单元，long 每 8 blocks 分块并发送最终不足 8 blocks 的尾块。输入和运行参数沿用交错 15 请求确认实验；没有扫描其他保护配置。

## 结论

实现符合预期分块规则且三轮正确性通过：每轮 15/15 请求成功，输出 token/text 均匹配直接推理参考；每轮 329 个候选 chunk 全部成功。long 每轮 316 个 chunk，其中 312 个为 8 blocks、4 个为 1-block 尾块。short/medium 的传输粒度按 batch 聚合；请求跨多个 prefill batch 时仍对应多个 chunk。

性能假设未获支持。候选三轮 makespan 均高于同轮 baseline，平均 **17.819s vs 14.304s（+24.6%）**；候选 short TTFT 均值 7359ms、medium 10833ms。逐 long E2E 的最差增幅三轮分别为 **+23.1%、+30.8%、+20.6%**。因此不能把此组合设为性能优选，也不能以减少传输单元直接推断整批性能改善。此实验同时改变了 short/medium 聚合和 long chunk 尺寸，未分别隔离两项改动的因果贡献。

| 轮次 | baseline makespan(s) | 候选 makespan(s) | 配对变化 | 最差 long E2E 变化 | 正确性 |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 14.253 | 17.789 | +24.8% | +23.1% | 通过 |
| 2 | 14.784 | 18.669 | +26.3% | +30.8% | 通过 |
| 3 | 13.876 | 16.998 | +22.5% | +20.6% | 通过 |
| 均值 | 14.304 | 17.819 | +24.6% | — | 45/45 请求 |

候选三轮平均类别 TTFT 为 short **7359ms**、medium **10833ms**、long **10038ms**。这些类别均值不改变整批 makespan 和长请求 E2E 的回归结论。单轮请求量有限，结论只适用于此固定 workload。

## Chunk 轨迹审计

三轮候选每轮都观测到 329 个成功 chunk：short 9、medium 4、long 316。short 共 64 blocks；其中一次 short 请求跨两个 batch，因此记录两个聚合 chunk。medium 共 210 blocks；跨 batch 的请求也按 batch 形成独立聚合单元。long 共 2500 blocks，分为 312 个 8-block chunk 和 4 个 1-block final chunk。所有请求 chunk 序号连续，所有传输成功，final 均有记录。轨迹文件为各 repeat 下的 `preemptive-c8-p3-x0.85-w50/trace-chunks.csv`。

这里的“每 batch 一个 chunk”指一次 batch 完成产生的该请求新 KV 聚合在一个 chunk 中，不表示整条请求全程只有一个 chunk；而 long 保持可抢占的 8-block 边界。

## 数据与复现

全部原始日志、请求指标、execution/chunk trace、环境、GPU 采样及配对汇总保存在本目录。主要汇总为 `confirmation-results.json` 和 `raw-confirmation.json`。保护参数保持 85% / 3 / 50ms；候选运行记录 chunk_size_blocks=8。聚焦 scheduler 回归与之前完整测试在实现阶段通过（74 passed、14 warnings）；此次三轮实验自身所有服务组退出码为 0。

整批总时间定义为首个客户端 HTTP 提交至最后响应结束，不含模型启动、warmup 和清理。该实验不能说明 GPU 计算/传输重叠失效；它只说明在已实现重叠的路径上，这一 chunk 调度组合没有取得整体加速。
