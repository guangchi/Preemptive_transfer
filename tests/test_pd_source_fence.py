"""Ensure only a supported post-forward producer path supplies a CUDA fence."""
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import pytest

from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector


@pytest.mark.parametrize("sync_loads,ubo,architecture,expected", [
    (False, False, "LlamaForCausalLM", True),
    (True, False, "LlamaForCausalLM", False),
    (False, True, "LlamaForCausalLM", False),
    (False, False, "OtherModel", False),
])
def test_event_fence_requires_audited_post_forward_scope(sync_loads, ubo, architecture, expected):
    connector = ActiveKVConnector.__new__(ActiveKVConnector)
    metadata = NS(push_chunks=[{"request_id": "long"}])
    schedule = NS(kv_connector_metadata=metadata, has_sync_kv_loads=sync_loads)
    connector._pending_load_kwargs = {"scheduler_output": schedule}
    connector.kv_connector = Mock()
    connector.vllm_config = NS(
        kv_transfer_config=NS(kv_role="kv_producer"),
        parallel_config=NS(tensor_parallel_size=1, pipeline_parallel_size=1,
                           decode_context_parallel_size=1, prefill_context_parallel_size=1,
                           enable_dbo=False, use_ubatching=ubo),
        offload_config=NS(uva=NS(cpu_offload_gb=0), prefetch=NS(offload_group_size=0)),
        model_config=NS(hf_config=NS(architectures=[architecture])),
        speculative_config=None,
    )
    order = []
    event = Mock()
    event.record.side_effect = lambda stream: order.append("fence")
    connector.kv_connector.start_load_kv.side_effect = lambda *a, **kw: order.append("load")
    with patch.dict("os.environ", {"PD_KV_SOURCE_FENCE": "event"}), patch(
        "vllm.v1.worker.gpu.kv_connector.is_forward_context_available", return_value=True
    ), patch("vllm.v1.worker.gpu.kv_connector.get_forward_context"), patch(
        "torch.cuda.Event", return_value=event
    ), patch("torch.cuda.current_stream"):
        connector._start_load_kv()

    assert hasattr(metadata, "_pd_kv_ready") == expected
    assert order == (["fence", "load"] if expected else ["load"])
    event.synchronize.assert_not_called()

