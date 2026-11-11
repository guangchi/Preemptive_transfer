# 局部排序功能验收

日期：2026-11-09。单机双 RTX 4090，P=GPU0、D=GPU1，TP=PP=1。vLLM 0.32.0，Llama-3.1-8B-Instruct，NIXL/UCX TCP loopback；两端 TCP TX/RX=128KB。原完整方案 C8/85%/3/50ms，三个传输模块全部开启，prefix cache 关闭，batch invariant 开启。

## 实施内容

增加 P-only 自定义 AsyncScheduler，使用固定 20ms 的接纳窗口。窗口内真实 token 数稳定升序，窗口间仍按 FCFS 排队；已有 running 请求照常计算。开关关闭时 launcher 不加载自定义调度器。无需修改已安装的 vLLM runtime。

客户端原有预排序被去掉；增加独立的 `--grouped-arrivals`。同一 JSONL 原顺序提交，15 并发，三批各五条，批起点间隔 500ms，批内间隔 0。旧 100ms 到达模式也单独验收。用法和精确窗口边界见 [使用说明](../../PD_LOCAL_SORT_USAGE.md)。

## 功能证据

- on/off/legacy-off 各预热 1 轮、正式 1 轮，共 90/90 请求成功，每条输出 8 tokens；所有轮次逐请求 token 输出 hash 一致，输入 token 数正确。
- on 预热及正式轮均形成三个五请求窗口。正式轮长度分别为 `[128,128,128,1107,10000]`、`[128,128,1107,10000,10000]`、`[128,128,128,1107,10000]`。三个窗口对应原始第 0～4、5～9、10～14 条。相同长度保留实际接纳顺序，因此不保证原文件中同长请求的编号顺序。
- P GPU forward 轨迹验证排序实际进入计算：正式首批 3 条 short 各 128 tokens、medium 1107 tokens，long 使用剩余 557 tokens，合计 2048。
- 正式 on 轮第二批 06、08 号短请求各一次越过第一批 01 号 long 的 KV 传输；当时 long 传输进度约 66.56%，未受保护。共 2 次传输抢占；off 正式轮 3 次，legacy-off 正式轮 2 次。
- 三组正式轮均通过 `collect_pd_trace.py --require-complete --require-execution`，P/D 执行事件及生命周期齐全；45/45 条 D 首次模型提交均在对应 KV 接收完成之后。D 没有加载自定义排序器。
- 回归测试 93 passed、14 warnings；取消和不足一组由回归测试覆盖，此轮未额外做 GPU 取消故障注入。
- 运行后实验进程已退出，两卡显存回到约 9MiB。

## 小规模实测

以下为各一轮正式结果；用于功能验收，不构成统计显著的性能结论。makespan 从首条客户端提交到末条结束，不含预热、服务启动和实验间等待。TTFT 为类别算术平均，包含窗口等待。

| 配置 | makespan (s) | short TTFT (ms) | medium TTFT (ms) | long TTFT (ms) | long E2E (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 三批到达，排序关闭 | 5.4530 | 2271.42 | 2360.86 | 2794.87 | 3100.72 |
| 三批到达，排序开启 | 5.3067 | 1545.10 | 1827.74 | 2868.63 | 3176.09 |
| 原 100ms 到达，排序关闭 | 5.5042 | 2307.57 | 2820.95 | 2708.03 | 3015.22 |

在相同三批到达的一轮对照中，short 平均 TTFT 减少约 32.0%，medium 减少约 22.6%，总时长减少约 2.68%；long 平均 TTFT 增加约 2.64%。这与短任务优先的取舍一致，不能宣称所有类别均加速。原 100ms 组用于关闭开关后的兼容验收，到达模式不同，不用于估计排序收益。

窗口按 P 实际接纳时间划分。20ms 到期后的释放受调度周期影响；正式 on 轮实际接纳到释放约 17～228ms。这不代表修改了窗口时长，也不承诺 GPU 忙时有 20ms 的硬实时上限。

## 数据

- `on/`、`off/`、`legacy-off/`：manifest、environment、原始 P/D/Router 日志、预热与正式 CSV、客户端 metadata、工作负载、命令。
- 各目录 `trace-request-phases.csv`、`trace-chunks.csv`：严格生命周期与执行时间线。
- [audit.json](audit.json)：正式轮完整接纳顺序、分类均值、抢占计数及校验结果。
- `reproducibility/`：本轮脚本、测试和 SHA256 校验和。
