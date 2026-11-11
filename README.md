# P/D KV 传输与局部排序实验

硕士毕业设计：Preemptive_transfer

本目录可以作为独立 GitHub 仓库：保存实验代码、文档、原始结果，以及定制 vLLM 的全部 10 个 Python 修改文件。恢复方式是安装对应官方 vLLM wheel，再应用源码快照；不是重新编译一整份 vLLM，也不是安装任意最新版后直接运行。

新完整方案 = P 侧局部排序 + batch 完成后提前后台发送 KV + 短/中请求按请求按 batch 聚合 + long C8 + 短请求传输优先与 long 保护。关闭局部排序恢复旧方案。D 收齐请求 KV 后使用原调度器 decode。

完整设计、历史问题和结果解释见 [技术总览](PD_DESIGN_COMPLETE_TECHNICAL_REVIEW_20261111.md)，排序行为见 [使用说明](PD_LOCAL_SORT_USAGE.md)。

## 保存范围

- 顶层脚本、测试、workload、文档和 `results/`、`logs/`。
- `snapshots/vllm-preemptive-20261103/vllm/`：当前运行时的 10 个修改文件，目录日期不代表最终代码日期。SHA256 与当前运行时逐一核对过。
- `reproducibility/vllm-patch-manifest.json`：修改文件和官方 wheel 原始文件哈希，用于安全恢复。
- `reproducibility/environment-observed.txt`、`host-observed.json`、`pip-check-observed.txt`：原机环境证据。
- 上游 vLLM 许可保存在 `third_party/vllm-LICENSE`，上游文件版权头保留。

不包含模型权重、虚拟环境、CUDA 驱动或完整官方 vLLM 源码。需要另外准备可合法访问的模型；不能仅凭这个仓库恢复任意硬件上的原始性能数字。

## 恢复环境

已测平台：Linux x86_64，Python 3.11，两张 RTX 4090（P GPU 0、D GPU 1），Llama-3.1-8B-Instruct BF16，vLLM 0.32.0（wheel commit `g5df63a720`），Torch 2.13.0 CUDA 13.0，NIXL/nixl-cu13 1.5.2，flashinfer-python 0.7.0.post1。NVIDIA 驱动必须支持对应 CUDA，具体原机记录见 `reproducibility/host-observed.json`。

以下均在仓库根目录执行。仓库本身就是原来的 `code/`，不要求恢复成 `/home/ubuntu/zjj/code`。

```bash
# 初次 clone 前先安装 Git LFS；clone 后也可用 git lfs pull 补齐对象。
git lfs install
git clone <你的仓库地址> pd-kv-experiments
cd pd-kv-experiments
git lfs pull

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-reproduce.txt
python -m pip install --no-deps nixl==1.5.2

# 查看恢复计划，停止该环境内已有 P/D 服务后再应用。
python scripts/restore_vllm.py
python scripts/restore_vllm.py --apply
python scripts/restore_vllm.py --check
```

恢复脚本先校验全部快照、vLLM 版本及待覆盖文件哈希，遇到不匹配直接停止。现有文件只能是已知官方原文件或本项目修改文件；新增文件不能覆盖其他来源的同名文件。原文件保存在 `.vllm-backups/`，重复应用不会重复覆盖。修改为 Python 文件，因此沿用官方 wheel 内的编译组件；vLLM 升级或重装后应重新检查。

`requirements-reproduce.txt` 锁定关键版本，完整观察清单另存于 `reproducibility/`。**尚未在全新机器/空环境中重新安装并验收这一流程**；依赖仓库是否仍提供这些版本、驱动、CUDA JIT 工具链都影响能否直接运行。原机的全量 `pip freeze` 是环境记录，不应当作已经通过干净环境求解的 lock 文件。

原环境的一个已知包声明差异：`nixl==1.5.2` 声明要求 `nixl-cu12`，实际使用的是同版本 `nixl-cu13`。上述先装 CUDA 13 后端，再对这个包装包使用 `--no-deps`，避免自动拉入 CUDA 12 后端；其余包正常解析依赖。原机 `pip check` 的对应报错已保留，运行路径另外通过 NIXL 导入、回归测试和双 GPU 实验检查。

CUDA 工具链也有原机约束：13.0 runtime/header 与 13.4 nvcc 混用，启动脚本沿用 `CCCL_DISABLE_CTK_COMPATIBILITY_CHECK`。脚本默认从当前 Python 环境定位 `nvidia/cu13`，也可设置 `CUDA_TOOLKIT_ROOT=/path/to/cuda` 指向适配的完整工具链。原机另有 `cu13/lib64 -> lib` 以及构建时的 `libcuda.so` stub 链接；新环境如果出现编译/链接错误需根据实际安装补齐，不能假定所有 CUDA wheel 的目录结构相同。不要把 stub 目录加入运行期 `LD_LIBRARY_PATH`。

## 运行新完整方案

先把模型权重和 tokenizer 完整下载到本地目录；使用原模型才能比较原 workload 和 token 输出。只需要 vLLM 实验时不需要安装 SGLang，尽管复用的驱动脚本名称里包含 `sglang`。

```bash
source .venv/bin/activate
export MODEL=/你的模型目录/Llama-3.1-8B-Instruct
python scripts/check_environment.py
python -m pytest -q

# 新完整方案：15 个请求，分三批，每批 5 个；批起点 0/500/1000ms。
# 自动启动 P、D、Router，预热一次，再正式运行三次，最后关闭服务。
# 结果目录请使用一个新的路径，避免覆盖原始证据。
python run_sglang_pd_experiment.py \
  --system vllm-full --local-sort --local-sort-window-ms 20 \
  --grouped-arrivals --group-size 5 --intra-group-gap-ms 0 \
  --inter-group-gap-ms 500 --repeats 3 \
  --result-dir results/reproduce-new-main
```

默认 long 参数为 85% / 最多 3 次 / 累计等待 50ms，long chunk 为 8 blocks。默认 TCP loopback + cuda_copy，双方 128KB UCX TX/RX，prefix cache 关闭，batch-invariant 开启。P 和 D 分别使用两张卡；两张卡的空闲显存应足够。`check_environment.py` 检查先决条件，不等于模型启动/JIT 或端到端数值正确性已经通过。

旧方案：删除上述命令的 `--local-sort`，换一个结果目录。Baseline：使用 `--system vllm-baseline` 且不加 `--local-sort`，换一个结果目录。这里 baseline 是同一项目中 NixlPushConnector 的整请求路径，包含同样的公共 UCX 优化。

完整重跑既定两种密度、三套系统、三轮轮换对照：

```bash
python run_pd_local_sort_comparison.py --result-dir results/reproduce-comparison
python analyze_pd_local_sort_comparison.py --result-dir results/reproduce-comparison
```

该入口共 18 次正式运行，分别使用 250ms 和 500ms 批间隔。分析器会使用仓库中已有 `results/local-sort-validation-20261109/off/run-1.csv` 的参考 token，需保留该文件。换模型、框架版本或数值后端时参考输出可能改变，应单独建立参考，不能把数值差异当作运行成功。

## 已保存的结果

正式结果在 [局部排序比较报告](results/local-sort-comparison-20261109/REPORT.md)，并附原始 CSV、日志、环境、审计与请求明细。新方案相对旧方案降低短请求 TTFT 约 32%/39%，makespan 降低约 2%；相对 baseline，makespan 降低约 9%/10%。局部排序增加了一部分 long 等待，并非每个 long 都更快；准确数值和限制以报告为准。历史实验目录保留历史配置，不能直接当作新方案默认配置。

## 上传 GitHub

建议将本目录直接作为仓库根目录。已提供 `.gitignore`，只忽略 bytecode、临时 PID、虚拟环境、本机恢复备份和模型等；实验结果、日志与文档仍保留。超过 GitHub 普通文件限制的 CUDA trace 已配置 Git LFS（需要 LFS 配额）。上传前先执行 `git lfs install --local`，再 `git add .`，使其被正确转换为 LFS 对象。

本仓库保存该毕业设计的代码、实验文档和结果；模型权重和虚拟环境需单独准备。
