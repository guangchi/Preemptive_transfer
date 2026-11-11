# 抢占方案总时间核对与调度修复

日期：2026-11-06。固定原交错 15 请求 workload、long C8、85% / 3 次 / 50ms、单 WRITE 窗口。baseline 与候选同为 UCX TCP TX/RX=128KB，不把 UCX 优化单独计入抢占策略收益。

## 首先核对“总时间”

历史短/中 batch 聚合实验为 baseline 14.304s、候选 17.819s，确实慢 24.6%。该数据来自传输分段修复前。最近已完成的三轮为 baseline 6.295s、候选 5.601s，候选快 11.02%；含服务启动、固定等待与清理的脚本总时间也为 82.775s 对 80.214s，未发现当前候选大幅慢于 baseline。

原始两类计时已逐组核对，见 `before-timing-audit.json`。不能把 8KB 的旧候选、128KB 的新基线或启动总时间与 workload makespan 混成配对结果。旧报告新增了历史版本提示，保留失败数据。

## 抢占方案剩余的额外成本

最近三轮候选每轮 330 次 WRITE，而整请求 baseline 只有 15 次。平均 host 提交累计约 3.543s 对 2.057s，候选仍有额外成本；其提交间隙中位数约 1.5ms，每轮低于 10ms 的间隙累计约 0.512s。间隙包含完成检查、描述符、日志、等待及源数据可用性，不全部是空闲。

这些值不是互不重叠的端到端时间，不能相加或直接从 makespan 扣除。候选通过提前发送已完成 batch 的 KV 隐藏了一部分额外成本；整请求 baseline 则必须等整条 prompt 完成后提交。当前完整方案收益包含按 batch 流式发送、分块与优先仲裁的共同作用，尚未单独隔离“仅抢占”贡献。

## 实现与验证过的选择

1. **默认入口统一为 event**：此前成功实验显式开启 event，但 launcher 默认仍是 device。现在 `run_pd_pair.sh` 默认 event，只在已审计的 Llama、TP1/PP1、无同步 KV load 等执行路径记录 post-forward CUDA event；其余路径仍回退全设备同步。避免用户正常启动时悄然回到另一种执行方式。
2. **修正完成进度的处理顺序**：原先先清空 completion inbox，再 poll 当前 WRITE；刚发现的成功结果要等下轮才计入 long 进度，期间可能已仲裁下一块。现在先 poll 再处理完成结果，确保后续 85% 保护基于已确认的成功进度。失败、取消、source lease 与全请求完成边界保留。
3. **不采用无间隔连续提交作为默认**：保留 `PD_STREAM_EAGER_PROGRESS=1` 为消融。它在同版 on/off 单次探测中把间隙中位数从 1.583ms 降到 0.501ms，但 native 提交累计从 3.587s 增到 4.300s，总时间从 5.627s 增到 5.691s。该证据只支持“缩短 host 间隙不等于减少整批时间”，不足以证明稳定回归，也未证明锁竞争等内部机制。默认为 0，保留让出 CPU 的节奏。最终版本无论是否 eager，都先更新成功进度再仲裁。
4. **汇总包含双方全部指标**：confirmation 结果新增 baseline 三类 TTFT/E2E、配对 baseline 脚本总时间与明确计时定义，避免只列候选或只列短/中均值。

同版最终 runtime 的全设备同步单次控制为 **5.823801s**，23 个 device fence 的 host 等待累计 4.327s。该累计主要包含等待 GPU 完成的时间，不等于可消除的额外耗时。它没有复现“大幅慢于 baseline”，也不能把入口差异说成目前全部回归的已证实根因。原始数据在 `control-device/`；eager 探测在 `probe-eager-on/`、`probe-eager-off/`，探测源码在 `source-probe/`，最终 runtime 在 `source-runtime/`。

## 最终同条件三轮

最终数据在 `confirmation/`；探测与 device 控制不混入三轮性能均值。

| 轮次 | baseline 整批(s) | 候选整批(s) | 缩短 |
| --- | ---: | ---: | ---: |
| 1 | 6.155059 | 5.623398 | 8.64% |
| 2 | 6.211085 | 5.672021 | 8.68% |
| 3 | 6.158075 | 5.675427 | 7.84% |
| 均值 | **6.174740** | **5.656949** | **8.39%** |

| 三轮均值 | baseline | 候选 | 降低 |
| --- | ---: | ---: | ---: |
| short TTFT(ms) | 2789.661 | 2370.327 | 15.03% |
| medium TTFT(ms) | 3615.042 | 2893.761 | 19.95% |
| long TTFT(ms) | 3463.827 | 2728.411 | 21.23% |
| long E2E(ms) | 3777.356 | 3044.635 | 19.40% |
| 含启动/固定等待/清理的脚本总时间(s) | 81.064 | 79.879 | 1.46% |

脚本总时间的差异包含服务启动波动，不单独归功于调度算法。

正式六组 90/90 输出 token IDs 与文本 hash 匹配直接推理参考；另三组探测/控制 45/45，总计 **135/135**。候选每轮 330 个 chunk（long 316），序号、覆盖、成功状态均通过；12 个逐 long E2E 配对全改善，改善范围 9.47%～24.67%，未发现越过已有保护条件的插队。独立审计为 `final-correctness-and-timing-audit.json`，均值/配对汇总为 `final-comparison.json`。

**没有证明本轮比上一版候选进一步加速。** 上一版均值 5.600705s，本轮 5.656949s，数值上高 1.00%；它们不是交错的实现 A/B，不能据此宣布显著回归或加速。本轮解决默认入口不一致、成功进度处理滞后和比较信息缺失，拒绝把缩小提交间隙包装为新的端到端收益。当前方案相对同条件 baseline 的整体优势在三轮中重复得到验证。

验证：80 项回归通过、14 条 Torch 弃用警告；shell 语法、Python 编译与 launcher 默认值/非法参数拒绝检查通过。event 默认、窗口 1、eager=0、TX/RX=128k 的解析记录见 `validation.json`。

## 复现

```bash
PD_KV_SOURCE_FENCE=event PD_STREAM_MAX_IN_FLIGHT=1 \
PD_STREAM_EAGER_PROGRESS=0 PD_TRACE_WRITE_PHASES=1 \
UCX_TCP_TX_SEG_SIZE=128k UCX_TCP_RX_SEG_SIZE=128k \
/home/ubuntu/miniconda3/envs/cc/bin/python run_pd_tuning_confirm.py \
  --grid results/tune-interleaved15-20261106 \
  --output results/fixed-c8-new-confirm --repeats 3 \
  --candidates c8-p3-x0.85-w50
```

结论仅适用于本机 TCP staging 路径和当前 15 请求负载。long 指 10,000-token 输入、8-token 输出，尚未验证长输出 decode 负载。
