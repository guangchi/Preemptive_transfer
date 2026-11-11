# 动态 batch 输出差异调查

原始 15 请求实验只有 `interleaved15-07-long` 的两组 hash 不同；原数据只保存 hash，没有实际文本和 token ID，无法事后确定原运行差在第几个 token。

## 对照与证据

1. `diagnose_batch_determinism.py` 在同一 GPU 启动不含 KV connector 的独立 vLLM，对目标 long 单独运行两次、在完整 15 请求混合 batch 中运行两次。分别测试 `VLLM_BATCH_INVARIANT=0/1`，共保存 64 条直接推理响应。动态 batch 对照保留原 prompt、seed、temperature=0；没有 P→D 传输。
2. 关闭 batch-invariant 时，目标 long 的 solo/mixed token 序列这次相同，但 top logprobs 不同。例如第 2 个位置 request 候选的 logprob 从 -2.120573 变化到 -2.017972；第 3 个位置存在多个候选并列。说明不经过传输也存在 batch 相关数值差异；这项控制本身没有复现 token 翻转，不能单独证明旧输出差异的全部原因。
3. 用旧默认内核重新运行 PD 两组，在同一目标 long 复现 token 翻转：前三个 token 都是 [1121,1715,40370]，第 4 个 token baseline=864（prefill 的起始部分）、preemptive=2532（service）。baseline 的 864/1715/2532/8481 logprob 都为 -2.280138731；preemptive 的 2532=-2.188144684，864=-2.313144684。即同一生成前缀出现了 logits 排序变化，之后自回归输出分叉。完整候选列表见 token-level-findings.json。
4. 开启 batch-invariant 后，不经过传输的目标 solo/mixed 四次响应不仅 token 一致，top logprobs 也逐项一致。PD 开启相同模式后的中间验证，两种模式全部 15 条 token ID 与文本 hash 相同，且匹配上述直接推理参考。最终独立重跑的结果和传输事件时间线另行保存在 sweep-interleaved15-invariant-verified-20261106/。

## 处理

- P/D 启动和 sweep 默认 `VLLM_BATCH_INVARIANT=1`，显式 0 仅用于旧内核诊断。greedy + 固定 seed 仅控制采样，不保证不同矩阵形状、attention 分割或调度下的浮点计算完全相同。
- 请求采集保存实际文本、生成 token ID，并可通过 `PD_CAPTURE_LOGPROBS=1` 保存 top-5 logprobs；只向 D 请求审计输出，P 的一次性输出不增加此项开销。
- environment/run metadata 保存 batch-invariant 和 logprobs 配置。两组必须在同样内核配置下比较；不要直接把新旧内核性能差异归因于传输策略。
- 单参数两组 sweep 在 HTTP/trace 成功后，还会运行 summarize_pd_experiment.py 做 token ID 与文本 hash 配对校验；不一致时整体返回非零，并保留原始证据。每组 status=passed 只表示该组 HTTP/trace 成功，不表示跨模式正确性通过。
- 逐 chunk 原始轨迹用于核查序号、完成状态和抢占；最终输出还对照不经过传输的参考。没有进行逐层逐字节 KV 校验，不据此宣称证明所有可能场景均无传输错误。

## 证据范围

默认内核的数值依赖和临界 greedy 选择已有实际证据，开启 batch-invariant 后在本数据集消除了可观察差异。结论限于本模型、软件、硬件和固定输入；动态 batch 的不同计算/分块布局仍可影响性能。中间 invariant 运行的请求与 trace 完成，但运行中修改 runner 导致收尾退出码 2；最终重新从固定脚本运行，并单独归档退出状态，不将中间运行冒充最终验收。
