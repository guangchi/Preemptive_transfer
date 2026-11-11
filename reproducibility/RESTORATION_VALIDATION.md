# 恢复入口检查（2026-11-11）

本次为归档和恢复流程准备，未重新运行性能实验，也未创建或上传 GitHub 仓库。

- 10 个定制 vLLM 文件：运行时、快照和 manifest 的 SHA256 一致。
- manifest 还保存了原官方 wheel RECORD 中的文件哈希；两个新增文件原始哈希为 null。
- `scripts/restore_vllm.py --check`：本机通过，10/10。
- `scripts/check_environment.py`：本机通过，两张 RTX 4090、模型目录、CUDA、NIXL 和版本检查通过。
- `python -m pytest -q`：97 passed，14 条既有 Torch 弃用提示。包含恢复的备份、幂等、拒绝陌生文件/损坏快照、失败回滚测试。
- `bash -n run_pd_pair.sh`：通过。
- 实验驱动的 `--help`：通过，运行入口使用当前 Python 与模型环境变量。

未验证：新机器/空虚拟环境安装、完整服务器启动与 CUDA JIT、新环境的模型 token 输出及性能。关键依赖已列出，但不能把本机检查写成干净环境端到端复现已经成功。

现有历史实验结果不受修改；历史日志中的绝对路径保留为当时环境证据。这里只调整了新方案使用的启动链，其他历史扫参、SGLang 专用和绘图入口可能仍需要路径或额外依赖配置。
