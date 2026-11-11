#!/usr/bin/env python3
"""Save the resolved local P/D runtime and toolchain environment."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def collect_environment() -> dict:
    cuda_runtime = None
    try:
        import torch

        cuda_runtime = torch.version.cuda
    except ImportError:
        pass

    gpu_output = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,driver_version",
            "--format=csv,noheader",
        ]
    )
    gpu_inventory = []
    if gpu_output:
        for line in gpu_output.splitlines():
            fields = [field.strip() for field in line.split(",", 3)]
            if len(fields) == 4:
                gpu_inventory.append(
                    {
                        "index": fields[0],
                        "name": fields[1],
                        "uuid": fields[2],
                        "driver_version": fields[3],
                    }
                )

    toolkit_root = os.environ.get("CUDA_TOOLKIT_ROOT", "")
    nvcc = str(Path(os.environ.get("CUDA_HOME", toolkit_root)) / "bin" / "nvcc")
    nvcc_version_output = _command_output([nvcc, "--version"]) if Path(nvcc).is_file() else None
    nvcc_version = None
    if nvcc_version_output:
        nvcc_version = next(
            (
                line.strip()
                for line in nvcc_version_output.splitlines()
                if "release " in line
            ),
            nvcc_version_output.splitlines()[-1],
        )

    return {
        "model": os.environ.get("MODEL", ""),
        "mode": os.environ.get("MODE", ""),
        "measure_execution": os.environ.get("PD_MEASURE_EXECUTION", "1") == "1",
        "execution_timing_schema": 2,
        "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1",
        "capture_logprobs": os.environ.get("PD_CAPTURE_LOGPROBS", "0") == "1",
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: _version(name)
            for name in ("vllm", "torch", "nixl", "nixl-cu13", "tokenizers", "pytest")
        },
        "cuda_runtime_version": cuda_runtime,
        "cuda_toolkit_root": toolkit_root,
        "cuda_home": os.environ.get("CUDA_HOME", ""),
        "nvcc_version": nvcc_version,
        "nvcc_prepend_flags": os.environ.get("NVCC_PREPEND_FLAGS", ""),
        "nixl_side_channel_host": os.environ.get("NIXL_SIDE_CHANNEL_HOST", "127.0.0.1"),
        "p_device": os.environ.get("P_DEVICE", "0"),
        "d_device": os.environ.get("D_DEVICE", "1"),
        "p_port": os.environ.get("P_PORT", "8100"),
        "d_port": os.environ.get("D_PORT", "8200"),
        "p_side_channel_port": os.environ.get("P_SIDE_CHANNEL_PORT", "5600"),
        "d_side_channel_port": os.environ.get("D_SIDE_CHANNEL_PORT", "5601"),
        "max_model_len": os.environ.get("MAX_MODEL_LEN", "32768"),
        "gpu_memory_utilization": os.environ.get("GPU_MEMORY_UTILIZATION", "0.90"),
        "prefix_caching": os.environ.get("ENABLE_PREFIX_CACHING", "false").lower() == "true",
        "scale_out": os.environ.get("ENABLE_SCALE_OUT", "false").lower() == "true",
        "api_key_configured": bool(os.environ.get("VLLM_API_KEY")),
        "short_threshold_tokens": os.environ.get("SHORT_REQUEST_THRESHOLD_TOKENS", "256"),
        "long_threshold_tokens": os.environ.get("LONG_REQUEST_THRESHOLD_TOKENS", "8192"),
        "chunk_size_blocks": os.environ.get("CHUNK_SIZE_BLOCKS", "4"),
        "max_preemptions": os.environ.get("LONG_REQUEST_MAX_PREEMPTIONS", "3"),
        "protect_progress": os.environ.get("LONG_REQUEST_PROTECT_PROGRESS", "0.85"),
        "wait_threshold_ms": os.environ.get("LONG_REQUEST_MAX_PREEMPT_WAIT_MS", "50"),
        "gpu_inventory": gpu_inventory,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "environment.json"
    tmp = output.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(collect_environment(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, output)
    print(output)


if __name__ == "__main__":
    main()
