# P 侧局部排序的使用与边界

2026-11-09：已实现独立开关，并完成两张 RTX 4090 上的 P/D 功能验收。P 使用 GPU 0，D 使用 GPU 1；原 15 请求不变。验收结果见 [报告](results/local-sort-validation-20261109/REPORT.md)。

## 调度行为

- `PD_LOCAL_SORT=0` 为默认值：P 使用原 vLLM 调度器，恢复此前完整方案。
- `PD_LOCAL_SORT=1`：仅 P 加载 `pd_local_sort_scheduler.LocalSortAsyncScheduler`，保留异步调度能力。D 使用原调度器。
- 窗口以 P 调度器实际接纳第一个已分词请求的时间为起点，默认 `PD_LOCAL_SORT_WINDOW_MS=20`。窗口固定，不随新请求到达而延长。窗口内按真实 prompt token 数升序，相同长度保持接纳先后顺序。
- 窗口到期后，在下一次调度调用中按排序结果加入原 FCFS 等待队列。先前窗口的等待请求仍在前面；后面的窗口不越过它们。已运行的请求继续计算。
- 收集窗口不要求必须凑齐 5 条。孤立请求、最后不足一组的请求也能到期释放；等待中的取消请求会正常清理。
- 20ms 是收集时长，不是端到端等待上限。GPU 忙或引擎正在等待结果时，释放需要等下一次调度调用。本轮正式实验观察到约 17～228ms 的接纳到释放等待，已包含在客户端 TTFT 中。
- 窗口基于 P 实际接纳时刻，不是客户端 JSONL 中的五条分组标签，也不是 Router 的 HTTP 到达时间。网络、分词和上游排队若使一批请求在 P 相隔超过窗口，会自然形成多个窗口。此次实机两轮均实际形成三个五请求窗口。

原有 KV 策略保持：计算 batch 完成即后台发送；短/中请求按请求、按本批计算聚合；long 使用 C8；短请求在 chunk 边界抢占传输；85% 进度保护、最多 3 次、累计等待 50ms。这里的抢占指 KV 传输，不是 GPU 计算抢占。D 收齐 KV 后由原有调度器继续 decode，不新增优先级。

## 相同三批到达负载的开关对照

在仓库根目录（原机为 `/home/ubuntu/zjj/code`）执行，先激活已恢复补丁的 Python 环境并设置 `MODEL`，详见 [复现说明](README.md)。结果目录必须自行选择新的路径：

```bash
# 开启排序：15 条按原文件顺序发送，分 3 批，每批 5 条。
python run_sglang_pd_experiment.py \
  --system vllm-full --result-dir results/local-sort-on-new \
  --repeats 1 --grouped-arrivals --group-size 5 \
  --intra-group-gap-ms 0 --inter-group-gap-ms 500 \
  --local-sort --local-sort-window-ms 20

# 关闭排序：保持完全相同的三批到达配置。
python run_sglang_pd_experiment.py \
  --system vllm-full --result-dir results/local-sort-off-new \
  --repeats 1 --grouped-arrivals --group-size 5 \
  --intra-group-gap-ms 0 --inter-group-gap-ms 500
```

每个命令先预热一轮，再运行指定次数。`--inter-group-gap-ms` 表示相邻批起点间隔，本例约在 0/500/1000ms 提交。客户端不预先排序；并发数为 15，避免前一批未完成占满客户端工作线程而推迟后续批。

去掉 `--grouped-arrivals` 可恢复实验入口原先的 100ms 逐条到达。`--local-sort` 和到达模式相互独立；SGLang 入口不支持这个自定义 P 排序器，误用会在启动 GPU 前报错。

若独立启动服务，可设置 `PD_LOCAL_SORT=1 PD_LOCAL_SORT_WINDOW_MS=20` 后运行 `run_pd_pair.sh`；Router 使用相同环境。修改排序开关需要重启服务。单独运行 `run_pd_load.py --local-sort` 只检查服务已开启排序，不会替代服务端开关。

## 验证与日志

`prefill.log` 中的 `PD local admission` JSON 记录 window、rank、token 数、实际接纳/释放时刻和等待时间。客户端 CSV 中 `dispatch_group/rank` 只是发送分组及原始位置，实际调度顺序以 P 日志为准。

回归命令：`/home/ubuntu/miniconda3/envs/cc/bin/python -m pytest -q`。本轮 93 项通过，14 条为已有 Torch 弃用提示。窗口边界、稳定排序、跨窗口顺序、不足一组、待排序请求计数、取消及独立到达分组均有回归覆盖。
