"""Check CUDA-event instrumentation without requiring CUDA for unit tests."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm.v1.worker import pd_execution_timing as timing


def test_gpu_forward_events_are_shared_per_batch_and_wait_off_execution_thread(caplog):
    start, end = MagicMock(), MagicMock()
    start.elapsed_time.return_value = 2.5
    schedule = SimpleNamespace(num_scheduled_tokens={"a": 16, "b": 1, "zero": 0})
    with patch.dict("os.environ", {"PD_MEASURE_EXECUTION": "1"}), patch.object(
        timing.torch.cuda, "Event", side_effect=[start, end]
    ), patch.object(timing._resolver, "submit") as submit:
        with timing.pd_forward_timing(schedule, SimpleNamespace(kv_role="kv_producer")):
            start.record.assert_called_once()
            end.record.assert_not_called()
        end.record.assert_called_once()
        end.synchronize.assert_not_called()
    caplog.set_level("INFO", logger=timing.__name__)
    callback, *args = submit.call_args.args
    callback(*args)
    end.synchronize.assert_called_once()
    rows = [r.message for r in caplog.records if "PD GPU forward req=" in r.message]
    assert len(rows) == 2
    assert all("batch_requests=2 gpu_forward_ms=2.500000" in r for r in rows)
    assert all("side=p" in r for r in rows)


def test_disabled_or_empty_forward_does_not_create_cuda_events():
    with patch.dict("os.environ", {"PD_MEASURE_EXECUTION": "0"}), patch.object(
        timing.torch.cuda, "Event"
    ) as event:
        with timing.pd_forward_timing(
            SimpleNamespace(num_scheduled_tokens={"a": 1}), SimpleNamespace(kv_role="kv_consumer")
        ):
            pass
        event.assert_not_called()
