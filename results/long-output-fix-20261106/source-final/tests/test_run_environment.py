"""Checks that runtime metadata captures tool versions without credentials."""
import os
from unittest.mock import patch

import write_pd_run_environment as run_environment


def test_environment_metadata_records_gpu_and_never_records_api_key():
    gpu_csv = (
        "0, NVIDIA GeForce RTX 4090, GPU-uuid-0, 555.42.02\n"
        "1, NVIDIA GeForce RTX 4090, GPU-uuid-1, 555.42.02"
    )
    with patch.dict(
        os.environ,
        {
            "CUDA_HOME": "/no-cuda-toolkit",
            "CUDA_TOOLKIT_ROOT": "/no-cuda-toolkit",
            "VLLM_API_KEY": "never-write-this-secret",
            "P_DEVICE": "0",
            "D_DEVICE": "1",
            "MODE": "preemptive",
            "VLLM_BATCH_INVARIANT": "1",
            "PD_CAPTURE_LOGPROBS": "1",
        },
    ), patch.object(run_environment, "_command_output", return_value=gpu_csv):
        metadata = run_environment.collect_environment()

    assert metadata["api_key_configured"] is True
    assert metadata["p_device"] == "0"
    assert metadata["d_device"] == "1"
    assert metadata["batch_invariant"] is True
    assert metadata["capture_logprobs"] is True
    assert metadata["gpu_inventory"][1]["uuid"] == "GPU-uuid-1"
    assert "never-write-this-secret" not in str(metadata)
