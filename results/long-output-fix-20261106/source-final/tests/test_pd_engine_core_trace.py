"""Regression coverage for first D model-submission timing."""
from types import SimpleNamespace
from unittest.mock import patch

from vllm.v1.engine.core import EngineCore
from vllm.v1.core.sched.scheduler import Scheduler


def test_engine_core_logs_first_remote_prefill_submission_once(caplog):
    request_id = "cmpl-pd-trace-test-0-12345678"
    request = SimpleNamespace(
        # The connector clears this flag when it stages D registration.
        kv_transfer_params={
            "do_remote_prefill": False,
            "remote_request_id": "cmpl-pd-trace-test-0",
        },
    )
    engine = EngineCore.__new__(EngineCore)
    engine.scheduler = SimpleNamespace(requests={request_id: request})
    scheduler_output = SimpleNamespace(num_scheduled_tokens={request_id: 1})

    caplog.set_level("INFO", logger="vllm.v1.engine.core")
    engine._log_pd_first_model_submit(scheduler_output)
    engine._log_pd_first_model_submit(scheduler_output)

    matches = [record for record in caplog.records if "PD first model submit" in record.message]
    assert len(matches) == 1
    assert f"req={request_id}" in matches[0].message
    assert "scheduled_tokens=1" in matches[0].message
    assert "monotonic_s=" in matches[0].message


def test_engine_core_does_not_log_prefill_side_requests(caplog):
    request_id = "prefill-request"
    request = SimpleNamespace(kv_transfer_params={
        "do_remote_decode": True,
        "remote_request_id": "decode-request",
    })
    engine = EngineCore.__new__(EngineCore)
    engine.scheduler = SimpleNamespace(requests={request_id: request})
    caplog.set_level("INFO", logger="vllm.v1.engine.core")

    engine._log_pd_first_model_submit(
        SimpleNamespace(num_scheduled_tokens={request_id: 1})
    )

    assert not any("PD first model submit" in record.message for record in caplog.records)


def test_scheduler_logs_first_d_selection_after_kv_registration_once(caplog):
    request_id = "decode-request"
    request = SimpleNamespace(kv_transfer_params={
        "do_remote_prefill": False,
        "remote_request_id": "producer-request",
    })
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.requests = {request_id: request}
    caplog.set_level("INFO", logger="vllm.v1.core.sched.scheduler")
    scheduler._log_pd_first_scheduled({})
    scheduler._log_pd_first_scheduled({request_id: 0})
    assert not getattr(request, "_pd_first_scheduled_logged", False)
    scheduler._log_pd_first_scheduled({request_id: 1})
    scheduler._log_pd_first_scheduled({request_id: 1})
    matches = [r for r in caplog.records if "PD first scheduled" in r.message]
    assert len(matches) == 1
    assert "req=decode-request" in matches[0].message
    assert "monotonic_s=" in matches[0].message


def test_scheduler_does_not_log_producer_or_local_requests(caplog):
    scheduler = Scheduler.__new__(Scheduler)
    scheduler.requests = {
        "producer": SimpleNamespace(kv_transfer_params={"do_remote_decode": True}),
        "local": SimpleNamespace(kv_transfer_params=None),
    }
    caplog.set_level("INFO", logger="vllm.v1.core.sched.scheduler")
    scheduler._log_pd_first_scheduled({"producer": 8, "local": 8})
    assert not any("PD first scheduled" in r.message for r in caplog.records)


def test_execution_step_times_all_pd_requests_and_preserves_batch_queue_identity(caplog):
    engine = EngineCore.__new__(EngineCore)
    engine.scheduler = SimpleNamespace(requests={
        "p": SimpleNamespace(kv_transfer_params={"do_remote_decode": True}),
        "d": SimpleNamespace(kv_transfer_params={"remote_request_id": "producer"}),
        "local": SimpleNamespace(kv_transfer_params=None),
    })
    first = SimpleNamespace(num_scheduled_tokens={"p": 32, "local": 1})
    second = SimpleNamespace(num_scheduled_tokens={"d": 1})
    caplog.set_level("INFO", logger="vllm.v1.engine.core")
    with patch.dict("os.environ", {"PD_MEASURE_EXECUTION": "1"}), patch(
        "vllm.v1.engine.core.time.perf_counter", side_effect=[10.0, 10.001, 10.010, 10.012]
    ):
        engine._start_pd_execution_trace(first)
        engine._start_pd_execution_trace(second)
        engine._finish_pd_execution_trace(first)
        engine._finish_pd_execution_trace(second)
        engine._finish_pd_execution_trace(first)  # idempotent
    rows = [r.message for r in caplog.records if "PD execution step" in r.message]
    assert len(rows) == 2
    assert "req=p side=p step=0" in rows[0] and "elapsed_ms=10.000000" in rows[0]
    assert "req=d side=d step=0" in rows[1] and "elapsed_ms=11.000000" in rows[1]
    assert not hasattr(first, "_pd_execution_trace")
