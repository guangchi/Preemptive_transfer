"""Regressions for consumer completion direction and producer send leases."""
import queue
import os
import threading
import pytest
from concurrent.futures import Future
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import NixlBaseConnectorWorker
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import NixlConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.preemptive_transfer import PreemptiveTransferConfig
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_scheduler import NixlPushConnectorScheduler
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import NixlPushConnectorWorker


def _stream_scheduler(prompt_tokens=1107):
    scheduler = NixlPushConnectorScheduler.__new__(NixlPushConnectorScheduler)
    scheduler.preemptive_config = SimpleNamespace(
        chunk_size_blocks=4,
        classify=lambda tokens: "medium",
    )
    scheduler.block_size = 16
    request = SimpleNamespace(
        request_id="medium",
        kv_transfer_params={"do_remote_decode": True},
        num_prompt_tokens=prompt_tokens,
        num_computed_tokens=0,
    )
    scheduler._stream_requests = {"medium": request}
    scheduler._stream_tiers = {"medium": "medium"}
    total = (prompt_tokens + 15) // 16
    scheduler._stream_total_blocks = {"medium": total}
    scheduler._stream_next_block = {"medium": 0}
    scheduler._stream_next_sequence = {"medium": 0}
    scheduler._stream_final_queued = set()
    scheduler._newly_scheduled_push_chunks = []
    return scheduler, request, total


@pytest.mark.parametrize("prompt_tokens", [1107, 1088, 16])
def test_stream_chunks_emit_final_with_prompt_batch_without_finish_duplicate(prompt_tokens):
    scheduler, request, total = _stream_scheduler(prompt_tokens)
    block_state = SimpleNamespace(get_block_ids=lambda req_id: list(range(total)))
    output = SimpleNamespace(
        kv_connector_block_state=block_state,
        num_scheduled_tokens={"medium": prompt_tokens},
    )

    scheduler._build_stream_chunks(output)

    chunks = scheduler._newly_scheduled_push_chunks
    assert len(chunks) == (total + 3) // 4
    assert [b for c in chunks for b in c["local_block_ids"][0]] == list(range(total))
    assert [c["chunk_sequence"] for c in chunks] == list(range(len(chunks)))
    assert sum(c["is_final"] for c in chunks) == 1
    assert chunks[-1]["is_final"] is True
    assert scheduler._stream_next_block["medium"] == total
    assert scheduler._stream_final_queued == {"medium"}
    count = len(chunks)
    scheduler._append_final_stream_chunks(request, [list(range(total))])
    scheduler._build_stream_chunks(output)
    assert len(chunks) == count


def test_partial_prompt_batch_keeps_tail_until_tokens_are_computed():
    scheduler, request, total = _stream_scheduler()
    output = SimpleNamespace(
        kv_connector_block_state=SimpleNamespace(get_block_ids=lambda rid: list(range(total))),
        num_scheduled_tokens={"medium": 1040},
    )
    scheduler._build_stream_chunks(output)
    assert scheduler._stream_next_block["medium"] == 64
    assert not any(c["is_final"] for c in scheduler._newly_scheduled_push_chunks)
    request.num_computed_tokens = 1040
    output.num_scheduled_tokens["medium"] = 67
    scheduler._build_stream_chunks(output)
    chunks = scheduler._newly_scheduled_push_chunks
    assert [len(c["local_block_ids"][0]) for c in chunks[-2:]] == [4, 2]
    assert chunks[-1]["is_final"]
    assert sum(c["is_final"] for c in chunks) == 1


def test_final_chunk_reports_receive_only_after_all_sequences():
    worker = NixlPushConnectorWorker.__new__(NixlPushConnectorWorker)
    worker.transfer_topo = object()
    worker.preemptive_config = PreemptiveTransferConfig(enabled=True)
    worker._pending_completion_notifs = queue.Queue()
    worker._streaming_push_req_ids = {"request"}
    worker._received_chunk_ids = defaultdict(set)
    worker._final_chunk_sequence = {}
    worker._stream_request_state = {}
    worker._recving_transfers = {}

    worker._pending_completion_notifs.put(b"VLLM_KV_CHUNK:request:1:1:1")
    assert worker._get_new_notifs() == set()
    assert "request" not in worker._recving_transfers
    worker._pending_completion_notifs.put(b"VLLM_KV_CHUNK:request:0:0:1")
    assert worker._get_new_notifs() == set()
    assert worker._recving_transfers == {"request": []}
    worker._pending_completion_notifs.put(b"VLLM_KV_CHUNK:request:1:1:1")
    assert worker._get_new_notifs() == set()


def test_stream_send_completion_removes_lease_and_is_reported_once():
    worker = NixlPushConnectorWorker.__new__(NixlPushConnectorWorker)
    worker.preemptive_config = PreemptiveTransferConfig(enabled=True)
    worker._push_writer_wake = threading.Event()
    worker._stream_completed_sending = queue.Queue()
    worker._sending_transfers_lock = threading.Lock()
    worker._sending_transfers = {}
    worker._send_failures = set()
    worker._reqs_to_send = {"request": 100.0}
    worker._reqs_to_process = {"request"}
    worker.consumer_notification_counts_by_req = {"request": 0}
    worker._evict_finished_inbox = queue.Queue()

    with patch.object(NixlBaseConnectorWorker, "get_transfer_results",
                      side_effect=lambda: KVConnectorTransferResults()):
        worker._stream_completed_sending.put("request")
        result = worker.get_transfer_results()
        assert result.finished_sending == {"request"}
        assert worker._reqs_to_send == {}
        assert worker._reqs_to_process == set()
        assert worker.consumer_notification_counts_by_req == {}
        worker._stream_completed_sending.put("request")
        assert worker.get_transfer_results().finished_sending == set()


def test_successful_prompt_final_waits_for_scheduler_send_lease():
    worker = NixlPushConnectorWorker.__new__(NixlPushConnectorWorker)
    worker.preemptive_config = PreemptiveTransferConfig(enabled=True)
    worker._push_writer_wake = threading.Event()
    worker._stream_completed_sending = queue.Queue()
    worker._sending_transfers_lock = threading.Lock()
    worker._sending_transfers = {}
    worker._send_failures = set()
    worker._reqs_to_send = {}
    worker._reqs_to_process = {"request"}
    worker.consumer_notification_counts_by_req = {"request": 0}
    worker._evict_finished_inbox = queue.Queue()
    with patch.object(NixlBaseConnectorWorker, "get_transfer_results",
                      side_effect=lambda: KVConnectorTransferResults()):
        worker._stream_completed_sending.put("request")
        assert worker.get_transfer_results().finished_sending == set()
        assert worker._reqs_to_process == {"request"}
        assert worker._stream_success_pending == {"request"}
        worker._reqs_to_send["request"] = 100.0
        assert worker.get_transfer_results().finished_sending == {"request"}
        assert worker._stream_success_pending == set()
        assert worker.get_transfer_results().finished_sending == set()


def test_stream_summary_completes_missing_chunk_notifications():
    worker = NixlPushConnectorWorker.__new__(NixlPushConnectorWorker)
    worker.transfer_topo = object()
    worker.preemptive_config = PreemptiveTransferConfig(enabled=True)
    worker._pending_completion_notifs = queue.Queue()
    worker._streaming_push_req_ids = {"request"}
    worker._received_chunk_ids = defaultdict(set)
    worker._final_chunk_sequence = {}
    worker._stream_request_state = {"request": {"pending": 3}}
    worker._recving_transfers = {}
    worker._pending_completion_notifs.put(b"VLLM_KV_STREAM_DONE:request:7:1")

    assert worker._get_new_notifs() == set()
    assert worker._recving_transfers == {"request": []}
    assert worker._streaming_push_req_ids == set()


def test_deferred_stream_write_waits_for_single_active_slot():
    worker = NixlPushConnectorWorker.__new__(NixlPushConnectorWorker)
    worker._stream_cancelled = set()
    worker._stream_active = ({"request_id": "active"}, [123])
    worker._stream_deferred_inbox = queue.Queue()
    worker._start_stream_chunk = Mock()
    queued = ({"request_id": "next", "chunk_sequence": 4}, {"request_id": "d"})
    worker._stream_deferred_inbox.put(queued)

    assert worker._start_next_deferred_stream_chunk() is False
    worker._start_stream_chunk.assert_not_called()
    assert worker._stream_deferred_inbox.get_nowait() == queued
    worker._stream_deferred_inbox.put(queued)

    worker._stream_active = None
    assert worker._start_next_deferred_stream_chunk() is True
    worker._start_stream_chunk.assert_called_once_with(*queued)
    assert worker._stream_deferred_inbox.empty()


def test_handshake_reserves_stream_slot_until_deferred_write_is_started():
    worker = NixlPushConnectorWorker.__new__(NixlPushConnectorWorker)
    worker._stream_active = None
    worker._stream_reserved_chunk = None
    worker._stream_cancelled = set()
    worker._stream_waiting_handshakes = set()
    worker._stream_deferred_inbox = queue.Queue()
    worker._stream_completion_inbox = queue.Queue()
    worker._push_writer_wake = threading.Event()
    handshake = Future()
    worker._ensure_handshake = Mock(return_value=handshake)
    chunk = {"request_id": "long", "chunk_sequence": 0, "local_block_ids": [[1]]}
    registration = {"request_id": "d-long", "decode_engine_id": "d", "decode_host": "localhost",
                    "decode_port": 5601, "decode_tp_size": 1}

    worker._start_stream_chunk(chunk, registration)

    assert worker._stream_active is None
    assert not worker._stream_slot_available()
    handshake.set_result(({}, 0.0))
    # The callback only posts work; it must not let the arbiter overtake it.
    assert not worker._stream_slot_available()

    def start_write(task, reg):
        assert worker._stream_reserved_chunk is None
        worker._stream_active = (task, [123])

    worker._start_stream_chunk = Mock(side_effect=start_write)
    assert worker._start_next_deferred_stream_chunk()
    assert not worker._stream_slot_available()
    worker._start_stream_chunk.assert_called_once_with(chunk, registration)


def test_summary_waits_for_earlier_deferred_chunks_and_preserves_final_sequence():
    worker = NixlPushConnectorWorker.__new__(NixlPushConnectorWorker)
    worker._stream_reserved_chunk = None
    worker._stream_cancelled = set()
    worker._stream_arbiter = Mock()
    worker._stream_request_state = {"long": {"pending": 2, "final_completed": False, "failed": False}}
    worker._stream_completed_sending = queue.Queue()
    worker._streaming_push_req_ids = {"long"}
    worker._stream_registrations = {"long": {"request_id": "d-long", "decode_engine_id": "d", "decode_tp_size": 1}}
    worker._remote_agents = {"d": {(0, 0): "agent"}}
    worker.nixl_wrapper = Mock()
    worker._find_stream_registration = Mock(return_value=worker._stream_registrations['long'])

    worker._process_stream_completion({"request_id": "long", "chunk_sequence": 156, "is_final": True}, True)

    worker.nixl_wrapper.send_notif.assert_not_called()
    assert worker._stream_completed_sending.empty()
    assert worker._stream_request_state['long']['pending'] == 1
    worker._process_stream_completion({"request_id": "long", "chunk_sequence": 0, "is_final": False}, True)

    assert worker.nixl_wrapper.send_notif.call_count == 3
    assert all(c.kwargs['notif_msg'] == b'VLLM_KV_STREAM_DONE:d-long:156:1'
               for c in worker.nixl_wrapper.send_notif.call_args_list)
    assert worker._stream_completed_sending.get_nowait() == 'long'
    assert not worker._stream_request_state


def test_failed_stream_cleans_writer_state_and_releases_lease_once():
    worker = NixlPushConnectorWorker.__new__(NixlPushConnectorWorker)
    worker._stream_cancelled = set()
    worker._stream_arbiter = Mock()
    worker._stream_pending_chunks = {"request": [{"chunk_sequence": 2}]}
    worker._stream_registrations = {"request": {"request_id": "d-request"}}
    worker._stream_waiting_handshakes = {("request", 2), ("other", 1)}
    worker._stream_request_state = {"request": {"pending": 1, "producer_done": True}}
    worker._streaming_push_req_ids = {"request", "other"}
    worker._stream_failed_sending = queue.Queue()
    worker._push_writer_wake = threading.Event()
    worker._remote_agents = {}

    worker._fail_stream_request("request", notify=False)

    worker._stream_arbiter.cancel.assert_called_once_with("request")
    assert worker._stream_pending_chunks == {}
    assert worker._stream_registrations == {}
    assert worker._stream_waiting_handshakes == {("other", 1)}
    assert worker._stream_request_state == {}
    assert worker._streaming_push_req_ids == {"other"}
    assert worker._stream_failed_sending.get_nowait() == "request"
    worker._fail_stream_request("request", notify=False)
    assert worker._stream_failed_sending.empty()


def test_failed_stream_waits_for_scheduler_final_or_cancel_marker():
    worker = NixlPushConnectorWorker.__new__(NixlPushConnectorWorker)
    worker._stream_cancelled = set()
    worker._stream_arbiter = Mock()
    worker._stream_pending_chunks = {"request": []}
    worker._stream_registrations = {}
    worker._stream_waiting_handshakes = set()
    worker._stream_request_state = {"request": {"pending": 1}}
    worker._streaming_push_req_ids = {"request"}
    worker._stream_failed_sending = queue.Queue()
    worker._stream_deferred_inbox = queue.Queue()
    worker._push_writer_wake = threading.Event()
    worker._remote_agents = {}
    worker._stream_active = None

    worker._fail_stream_request("request", notify=False)

    assert worker._stream_failed_sending.empty()
    assert worker._stream_request_state["request"]["failed"] is True
    worker._stream_request_state["request"]["producer_done"] = True
    worker._maybe_queue_failed_stream("request")
    assert worker._stream_failed_sending.get_nowait() == "request"
    assert worker._stream_request_state == {}


def test_stream_progress_excludes_decode_prefix_cache_hits():
    worker = NixlPushConnectorWorker.__new__(NixlPushConnectorWorker)
    registration = {"local_block_ids": [[101, 102, 103]]}
    chunk = {
        "request_id": "long",
        "tier": "long",
        "chunk_sequence": 0,
        "block_start": 0,
        "local_total_blocks": 5,
        "local_block_ids": [[1, 2]],
    }

    worker._annotate_stream_progress(chunk, registration)

    assert chunk["progress_total_blocks"] == 3
    assert chunk["progress_blocks"] == 0
    next_chunk = {
        **chunk,
        "chunk_sequence": 1,
        "block_start": 2,
        "local_block_ids": [[3, 4]],
    }
    worker._annotate_stream_progress(next_chunk, registration)
    assert next_chunk["progress_blocks"] == 2
    final_chunk = {
        **chunk,
        "chunk_sequence": 2,
        "block_start": 4,
        "local_block_ids": [[5]],
        "is_final": True,
    }
    worker._annotate_stream_progress(final_chunk, registration)
    assert final_chunk["progress_blocks"] == 1


def test_registration_timeout_is_reported_as_failed_receive():
    scheduler = NixlPushConnectorScheduler.__new__(NixlPushConnectorScheduler)
    scheduler._push_registration_timeout = 0.25
    scheduler._push_registration_deadlines = {
        "expired": 1.0,
        "active": 2.0,
    }
    scheduler._push_pending_registrations = {
        "expired": {"request_id": "expired"},
        "active": {"request_id": "active"},
    }
    metadata = NixlConnectorMetadata()

    scheduler._expire_push_registrations(metadata, now=1.5)

    assert metadata.push_failed_receives == {"expired"}
    assert scheduler._push_registration_deadlines == {"active": 2.0}
    assert scheduler._push_pending_registrations == {
        "active": {"request_id": "active"}
    }


def test_worker_surfaces_registration_timeout_through_receive_results():
    worker = NixlPushConnectorWorker.__new__(NixlPushConnectorWorker)
    worker.pcp_rank = 0
    worker.pcp_dcp_sharded = False
    worker._recv_failures = set()
    worker.preemptive_config = PreemptiveTransferConfig(enabled=True)
    worker._recving_transfers = {"expired": [123]}
    worker._recving_metadata = {}
    worker._reqs_to_process = set()
    worker._reqs_to_send = {}
    worker._send_heartbeats = Mock()
    metadata = NixlConnectorMetadata()
    metadata.push_failed_receives.add("expired")

    worker.start_load_kv(metadata)

    assert worker._recv_failures == {"expired"}
    # Keep the NIXL handle until its normal safe-release boundary.
    assert worker._recving_transfers == {"expired": [123]}


def test_streaming_queues_source_chunks_before_writer_cuda_fence():
    worker = NixlPushConnectorWorker.__new__(NixlPushConnectorWorker)
    worker.pcp_rank = 0
    worker.pcp_dcp_sharded = False
    worker.preemptive_config = PreemptiveTransferConfig(enabled=True)
    worker._recv_failures = set()
    worker._recving_transfers = {}
    worker._recving_metadata = {}
    worker._reqs_to_process = set()
    worker._reqs_to_send = {}
    worker._send_heartbeats = Mock()
    worker._stream_chunk_inbox = queue.Queue()
    worker._stream_source_sync_inbox = queue.Queue()
    worker._push_writer_wake = threading.Event()
    metadata = NixlConnectorMetadata()
    metadata.push_chunks = [{"request_id": "long", "chunk_sequence": 0}]
    worker.start_load_kv(metadata)

    assert worker._stream_source_sync_inbox.get_nowait() == metadata.push_chunks
    assert worker._stream_chunk_inbox.empty()


def test_writer_fences_batch_before_exposing_chunks_to_arbiter():
    worker = NixlPushConnectorWorker.__new__(NixlPushConnectorWorker)
    worker._stream_chunk_inbox = queue.Queue()
    chunks = [{"request_id": "long", "chunk_sequence": i} for i in range(2)]
    order = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=Mock(return_value=True),
            synchronize=Mock(side_effect=lambda: order.append("sync")),
        )
    )
    original_put = worker._stream_chunk_inbox.put

    def record_put(chunk):
        order.append("enqueue")
        original_put(chunk)

    with patch.dict("sys.modules", {"torch": fake_torch}):
        with patch.object(worker._stream_chunk_inbox, "put", side_effect=record_put):
            worker._synchronize_stream_sources(chunks)

    fake_torch.cuda.synchronize.assert_called_once_with()
    assert order == ["sync", "enqueue", "enqueue"]
    assert [worker._stream_chunk_inbox.get_nowait() for _ in chunks] == chunks


def test_stream_write_failure_injection_targets_request_and_sequence():
    worker = NixlPushConnectorWorker.__new__(NixlPushConnectorWorker)
    worker._stream_cancelled = set()
    worker._complete_stream_chunk = Mock()
    worker._ensure_handshake = Mock(return_value=None)
    chunk = {
        "request_id": "cmpl-fail-long-0-deadbeef",
        "chunk_sequence": 3,
        "local_block_ids": [[1, 2, 3, 4]],
    }
    registration = {
        "decode_engine_id": "pd-decode",
        "decode_host": "127.0.0.1",
        "decode_port": 5601,
        "decode_tp_size": 1,
    }

    with patch.dict(os.environ, {"NIXL_TEST_FAIL_STREAM_CHUNK": "cmpl-fail-long-0:3"}):
        worker._start_stream_chunk(chunk, registration)

    worker._complete_stream_chunk.assert_called_once_with(chunk, False)


def test_stream_handshake_failure_reports_failed_chunk():
    worker = NixlPushConnectorWorker.__new__(NixlPushConnectorWorker)
    worker._stream_cancelled = set()
    worker._stream_waiting_handshakes = set()
    worker._stream_completion_inbox = queue.Queue()
    worker._push_writer_wake = threading.Event()
    worker._log_failure = Mock()
    future = Future()
    future.set_exception(ConnectionError("handshake peer closed"))
    worker._ensure_handshake = Mock(return_value=future)
    chunk = {
        "request_id": "cmpl-handshake-fail-0-deadbeef",
        "chunk_sequence": 2,
        "local_block_ids": [[1, 2, 3, 4]],
    }
    registration = {
        "decode_engine_id": "pd-decode",
        "decode_host": "127.0.0.1",
        "decode_port": 5791,
        "decode_tp_size": 1,
    }

    worker._start_stream_chunk(chunk, registration)

    assert worker._stream_completion_inbox.get_nowait() == (chunk, False)
    assert worker._stream_waiting_handshakes == set()
    worker._log_failure.assert_called_once()


def test_d_abort_scheduler_stages_producer_cancellation():
    from vllm.v1.request import RequestStatus

    scheduler = NixlPushConnectorScheduler.__new__(NixlPushConnectorScheduler)
    scheduler.preemptive_config = PreemptiveTransferConfig(enabled=True)
    scheduler._heartbeat_req_engine = {}
    scheduler._push_registration_deadlines = {}
    scheduler._push_pending_registrations = {}
    scheduler._push_cancelled_sends = {}
    request = SimpleNamespace(
        request_id="consumer-hash",
        status=RequestStatus.FINISHED_ABORTED,
        kv_transfer_params={
            "do_remote_prefill": False,
            "do_remote_decode": False,
            "remote_request_id": "cmpl-producer-0",
            "remote_engine_id": "pd-prefill",
            "remote_host": "127.0.0.1",
            "remote_port": 5770,
            "tp_size": 1,
            "dcp_size": 1,
            "pp_size": 1,
        },
    )

    assert scheduler.request_finished(request, []) == (False, None)
    assert scheduler._push_cancelled_sends == {
        "consumer-hash": {
            "request_id": "consumer-hash",
            "producer_request_id": "cmpl-producer-0",
            "remote_engine_id": "pd-prefill",
            "remote_host": "127.0.0.1",
            "remote_port": 5770,
            "remote_tp_size": 1,
            "remote_dcp_size": 1,
            "remote_pp_size": 1,
        }
    }


def test_p_cancel_notification_matches_hashed_request_id():
    worker = NixlPushConnectorWorker.__new__(NixlPushConnectorWorker)
    worker._stream_request_state = {"cmpl-producer-0-deadbeef": {"pending": 1}}
    worker._stream_registrations = {}
    worker._stream_pending_chunks = {}
    worker._stream_arbiter = Mock()
    worker._stream_arbiter.pending_request_ids.return_value = set()
    worker._reqs_to_send = {}
    worker._reqs_to_process = set()
    worker._stream_active = None
    worker._cancel_stream_request = Mock()

    worker._handle_stream_cancel_notif(
        b"VLLM_KV_STREAM_CANCEL:cmpl-producer-0"
    )

    worker._cancel_stream_request.assert_called_once_with(
        "cmpl-producer-0-deadbeef"
    )


def test_d_worker_sends_cancel_after_handshake():
    worker = NixlPushConnectorWorker.__new__(NixlPushConnectorWorker)
    worker._ensure_handshake = Mock(return_value=None)
    worker._remote_agents = {"pd-prefill": {(0, 0): "p-agent"}}
    worker.nixl_wrapper = Mock()
    data = {
        "request_id": "consumer-hash",
        "producer_request_id": "cmpl-producer-0",
        "remote_engine_id": "pd-prefill",
        "remote_host": "127.0.0.1",
        "remote_port": 5770,
        "remote_tp_size": 1,
    }

    worker._send_cancel_to_p(data)

    worker.nixl_wrapper.send_notif.assert_called_once_with(
        "p-agent", notif_msg=b"VLLM_KV_STREAM_CANCEL:cmpl-producer-0"
    )


def test_failed_stream_completion_releases_registered_lease():
    worker = NixlPushConnectorWorker.__new__(NixlPushConnectorWorker)
    worker.preemptive_config = PreemptiveTransferConfig(enabled=True)
    worker._push_writer_wake = threading.Event()
    worker._stream_completed_sending = queue.Queue()
    worker._stream_failed_sending = queue.Queue()
    worker._stream_failed_pending = set()
    worker._sending_transfers_lock = threading.Lock()
    worker._sending_transfers = {}
    worker._send_failures = set()
    worker._reqs_to_send = {"request": 100.0}
    worker._reqs_to_process = set()
    worker.consumer_notification_counts_by_req = {"request": 0}
    worker._evict_finished_inbox = queue.Queue()
    worker._stream_failed_sending.put("request")

    with patch.object(
        NixlBaseConnectorWorker,
        "get_transfer_results",
        side_effect=lambda: KVConnectorTransferResults(),
    ):
        assert worker.get_transfer_results().finished_sending == {"request"}
        assert worker._reqs_to_send == {}


def test_failed_stream_completion_waits_for_lease_metadata():
    worker = NixlPushConnectorWorker.__new__(NixlPushConnectorWorker)
    worker.preemptive_config = PreemptiveTransferConfig(enabled=True)
    worker._push_writer_wake = threading.Event()
    worker._stream_completed_sending = queue.Queue()
    worker._stream_failed_sending = queue.Queue()
    worker._stream_failed_pending = set()
    worker._sending_transfers_lock = threading.Lock()
    worker._sending_transfers = {}
    worker._send_failures = set()
    worker._reqs_to_send = {}
    worker._reqs_to_process = set()
    worker.consumer_notification_counts_by_req = {}
    worker._evict_finished_inbox = queue.Queue()
    worker._stream_failed_sending.put("request")

    with patch.object(
        NixlBaseConnectorWorker,
        "get_transfer_results",
        side_effect=lambda: KVConnectorTransferResults(),
    ):
        assert worker.get_transfer_results().finished_sending == set()
        assert worker._stream_failed_pending == {"request"}
        worker._reqs_to_send["request"] = 100.0
        assert worker.get_transfer_results().finished_sending == {"request"}
        assert worker._reqs_to_send == {}
        assert worker._stream_failed_pending == set()


def test_cancelled_stream_chunks_are_not_requeued_after_failure():
    worker = NixlPushConnectorWorker.__new__(NixlPushConnectorWorker)
    worker._stream_cancelled = {"request"}
    worker._complete_stream_chunk = Mock()
    worker._stream_request_state = {}
    worker._streaming_push_req_ids = set()
    worker._stream_registrations = {}
    worker._stream_pending_chunks = {}
    worker._stream_waiting_handshakes = set()
    worker._stream_deferred_inbox = queue.Queue()
    worker._stream_arbiter = Mock()
    worker._stream_failed_sending = queue.Queue()
    worker._push_writer_wake = threading.Event()
    worker._remote_agents = {}

    worker._accept_stream_chunk(
        {"request_id": "request", "chunk_sequence": 2, "local_block_ids": [[1]]}
    )
    assert worker._stream_arbiter.enqueue.call_count == 0
    worker._complete_stream_chunk.assert_called_once()


def test_cancelled_stream_completions_do_not_log_transfer_errors(caplog):
    worker = NixlPushConnectorWorker.__new__(NixlPushConnectorWorker)
    worker._stream_cancelled = {"request"}
    worker._stream_request_state = {
        "request": {"pending": 1, "failed": True, "producer_done": False}
    }
    worker._stream_arbiter = Mock()
    worker._fail_stream_request = Mock()

    worker._process_stream_completion(
        {"request_id": "request", "chunk_sequence": 3, "is_final": True},
        success=False,
    )

    assert worker._stream_request_state["request"]["pending"] == 0
    assert worker._stream_request_state["request"]["producer_done"] is True
    worker._fail_stream_request.assert_called_once_with("request", notify=False)
    assert "Streaming KV transfer failed" not in caplog.text


def test_final_chunk_marks_producer_done_before_write_completion():
    worker = NixlPushConnectorWorker.__new__(NixlPushConnectorWorker)
    worker._stream_cancelled = set()
    worker._streaming_push_req_ids = set()
    worker._stream_request_state = {}
    worker._stream_pending_chunks = defaultdict(list)
    worker._stream_registrations = {}
    worker._stream_arbiter = Mock()
    worker._find_stream_registration = Mock(return_value=None)

    worker._accept_stream_chunk(
        {
            "request_id": "request",
            "chunk_sequence": 4,
            "local_block_ids": [[1]],
            "is_final": True,
        }
    )

    assert worker._stream_request_state["request"]["producer_done"] is True
