# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Push-specific (WRITE) worker-side logic for the NIXL connector.

A dedicated ``nixl-push-writer`` thread owns all push-related NIXL ops:
calls ``get_new_notifs`` (routing PUSH_REG internally; HB / completion
notifs are forwarded to the engine main thread), sends PUSH_REG via
``send_notif``, matches D registrations with P finished blocks, and
issues WRITE transfers via ``make_prepped_xfer`` / ``transfer``.

The engine main thread feeds the writer through queues:
``_reg_send_inbox`` (D-side regs to send), ``_finished_blocks_inbox``
(P-side blocks from metadata) and ``_pending_completion_notifs``
(non-PUSH_REG notifs forwarded back for HB / completion accounting).
The handshake-completion callback feeds ``_deferred_push_inbox`` with
matched pushes whose P→D handshake has finished so the writer can
(re-)issue the WRITE without ever blocking on the network.

Wake model: the writer self-polls every
``_PUSH_WRITER_POLL_INTERVAL_MS`` only while it has unmatched
``_push_finished_blocks`` (i.e. P-side blocks waiting for a D PUSH_REG
notif that has no other wake source). All other progress is
event-driven: the engine main thread sets ``_push_writer_wake`` from
``start_load_kv`` (when handing it new work) and from ``get_finished``
(so each engine step gives the writer a chance to drain NIXL notifs);
the handshake-completion callback sets the same event after a deferred
PUSH_REG send or a deferred push WRITE has been queued. When a request's
lease expires (the base worker reports it via ``done_sending``) or the WRITE completes,
``get_finished`` enqueues an eviction onto ``_evict_finished_inbox`` so
the writer drops any leftover ``_push_finished_blocks`` /
``_pending_d_registrations`` and stops self-polling.
"""

import os
import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any

import msgspec

from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorTransferResults,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.base_worker import (
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    PUSH_REG_NOTIF_PREFIX,
    NixlConnectorMetadata,
    RemoteMeta,
    ReqId,
    ReqMeta,
    TransferHandle,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.preemptive_transfer import (
    ChunkPriorityArbiter,
    PreemptiveTransferConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.tp_mapping import (
    ReadSpec,
    _is_attention_spec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import get_base_request_id
from vllm.logger import init_logger

if TYPE_CHECKING:
    import torch

    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)

# Writer-thread poll cadence while there is in-flight push state. When
# fully idle, the writer blocks on a wake event signalled by the engine
# main thread (start_load_kv / get_finished). Smaller -> lower latency
# while active, slightly more CPU.
_PUSH_WRITER_POLL_INTERVAL_MS = 1.0


class NixlPushConnectorWorker(NixlBaseConnectorWorker):
    """Push-specific (WRITE) worker logic. See module docstring."""

    # Distinguishes push from pull in the NIXL compatibility hash.
    _TRANSFER_MODE: str = "push"

    _supports_pp_hma = True

    def __init__(
        self,
        vllm_config: "VllmConfig",
        engine_id: str,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(vllm_config, engine_id, kv_cache_config)
        self.preemptive_config = PreemptiveTransferConfig.from_vllm_config(
            vllm_config
        )
        self.preemptive_config.validate_runtime_scope(vllm_config, kv_cache_config)

        # Heartbeat handshakes to a PP-sharded producer must be notif-only,
        # like the PUSH_REG path.
        self._hb_handshake_notif_only = True

        # Push-specific state.
        # P-side: outgoing WRITE handles awaiting completion, keyed by
        # request_id. Mutated by writer (submit) and main thread
        # (``_pop_done_transfers``); guarded by
        # ``_sending_transfers_lock``.
        self._sending_transfers = defaultdict[ReqId, list[TransferHandle]](list)
        self._send_failures: set[ReqId] = set()
        self._sending_transfers_lock = threading.Lock()

        # Writer-thread owned matching state.
        # P-side: finished request blocks received from scheduler metadata
        # that have not yet been matched with an incoming D registration.
        self._push_finished_blocks: dict[ReqId, BlockIds] = {}
        # P-side: D registrations received via NIXL notification that have
        # not yet been matched with a finished P request.
        self._pending_d_registrations: dict[ReqId, dict[str, Any]] = {}

        # Cross-thread channels.
        self._reg_send_inbox: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
        self._cancel_send_inbox: queue.Queue[dict[str, Any]] = queue.Queue()
        self._finished_blocks_inbox: queue.Queue[tuple[str, BlockIds]] = queue.Queue()
        self._pending_completion_notifs: queue.Queue[bytes] = queue.Queue()
        # Main thread → writer: req_ids whose lease has expired or whose
        # WRITE has completed. Writer drops them from
        # ``_push_finished_blocks`` so an unmatched entry doesn't keep the
        # writer busy-polling forever.
        self._evict_finished_inbox: queue.Queue[str] = queue.Queue()
        # Handshakes that have just completed and are ready for the WRITE on wthread
        self._deferred_push_inbox = queue.Queue[tuple[str, BlockIds, dict[str, Any]]]()

        # Opt-in per-chunk streaming state. Only the writer thread touches the
        # arbiter, registrations, pending chunk lists, and active handle set.
        self._stream_arbiter = ChunkPriorityArbiter(self.preemptive_config)
        self._stream_chunk_inbox: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stream_source_sync_inbox: queue.Queue[tuple[list[dict[str, Any]], Any, float]] = queue.Queue()
        self._stream_source_events: list[tuple[list[dict[str, Any]], Any, float]] = []
        self._stream_completion_inbox: queue.Queue[
            tuple[dict[str, Any], bool]
        ] = queue.Queue()
        self._stream_deferred_inbox: queue.Queue[
            tuple[dict[str, Any], dict[str, Any]]
        ] = queue.Queue()
        self._stream_registrations: dict[str, dict[str, Any]] = {}
        self._stream_pending_chunks: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._stream_waiting_handshakes: set[tuple[str, int]] = set()
        self._stream_active: tuple[dict[str, Any], list[int]] | None = None
        # A chunk waiting for a handshake owns the same single slot as an
        # active WRITE. Otherwise later chunks (including final) can overtake
        # it and a summary can expose incomplete KV to D.
        self._stream_reserved_chunk: tuple[str, int] | None = None
        self._stream_cancelled: set[str] = set()
        self._stream_completed_sending: queue.Queue[str] = queue.Queue()
        self._stream_success_pending: set[str] = set()
        # Failed/cancelled streams must also end their P-side lease once all
        # currently active NIXL work has reached a safe completion boundary.
        # They are kept separate from successful completions so callers can
        # distinguish a failed transfer while still reclaiming source blocks.
        self._stream_failed_sending: queue.Queue[str] = queue.Queue()
        # Failed streams wait here until the scheduler has removed the
        # request from the active batch. Reporting finished_sending earlier
        # would make the scheduler free blocks while request.is_finished()
        # is still false.
        self._stream_failed_pending: set[str] = set()
        self._streaming_push_req_ids: set[str] = set()
        self._stream_request_state: dict[str, dict[str, Any]] = {}
        # D-side chunk receive accounting. TP/PP are deliberately constrained
        # to one in this first implementation.
        self._received_chunk_ids: dict[str, set[int]] = defaultdict(set)
        self._final_chunk_sequence: dict[str, int] = {}

        # Wake signal from engine main thread (start_load_kv / get_finished).
        # Writer self-polls at _PUSH_WRITER_POLL_INTERVAL_MS while it has
        # active in-flight state; otherwise it blocks until signalled.
        self._push_writer_wake = threading.Event()

        self._push_writer_stop = threading.Event()
        self._push_writer_thread: threading.Thread | None = None

    # --- Lifecycle ----------------------------------------------------- #

    def register_kv_caches(self, kv_caches: dict[str, "torch.Tensor"]):
        super().register_kv_caches(kv_caches)
        if self._mixed_mem_types:
            raise NotImplementedError(
                "NixlPushConnector does not support mixed-memory KV caches"
            )
        if self._push_writer_thread is None:
            self._push_writer_thread = threading.Thread(
                target=self._push_writer_loop,
                daemon=True,
                name="nixl-push-writer",
            )
            self._push_writer_thread.start()
            logger.info("nixl-push-writer thread started (rank=%d)", self.tp_rank)

    def shutdown(self) -> None:
        self._push_writer_stop.set()
        # Unblock the writer if it's waiting in the no-active-state branch.
        self._push_writer_wake.set()
        if self._push_writer_thread is not None:
            self._push_writer_thread.join(timeout=2)
            self._push_writer_thread = None
        with self._sending_transfers_lock:
            for handles in self._sending_transfers.values():
                for handle in handles:
                    self.nixl_wrapper.release_xfer_handle(handle)
            self._sending_transfers.clear()
            self._send_failures.clear()
        super().shutdown()

    # --- Engine-main-thread entry point -------------------------------- #

    def start_load_kv(self, metadata: NixlConnectorMetadata):
        """Pre-process metadata; defer NIXL ops to the writer thread."""
        if self.pcp_rank > 0 and not self.pcp_dcp_sharded:
            return

        # Registration watchdog failures travel through the normal connector
        # result path so the engine can fail the waiting D request. Existing
        # receive handles remain tracked until NIXL reports their safe boundary.
        for req_id in metadata.push_failed_receives:
            logger.warning("D-side streaming receive timed out req=%s", req_id)
            self._recv_failures.add(req_id)
            self._recving_transfers.setdefault(req_id, [])

        if metadata.push_cancelled_sends:
            for cancel_data in metadata.push_cancelled_sends.values():
                self._cancel_send_inbox.put(cancel_data)
            self._push_writer_wake.set()

        # D-side: track reqs waiting for P to push.
        for req_id, meta in metadata.reqs_to_recv.items():
            meta.local_physical_block_ids = self._logical_to_kernel_block_ids(
                meta.local_block_ids, self._physical_blocks_per_logical_kv_block
            )
            assert meta.remote is not None
            remote_engine_id = meta.remote.engine_id
            # Update last activity from this remote (same as pull), but only for
            # an already-connected engine so we never leave an _engine_last_active
            # entry without a _remote_agents entry (breaks _cleanup_remote_engine).
            if remote_engine_id in self._remote_agents:
                self._engine_last_active[remote_engine_id] = time.perf_counter()

            logger.debug(
                "start_load_kv (push) for request %s from remote engine %s. "
                "Num local_block_ids: %s. Num remote_block_ids: %s. ",
                req_id,
                remote_engine_id,
                len(meta.local_physical_block_ids),
                len(meta.remote.block_ids),
            )
            self._recving_metadata[req_id] = meta
            # Streaming push completion notifications are accepted only for
            # requests registered in this worker.  The P side adds its own
            # request IDs when chunks are queued; the D side must register
            # the decoder request here before the first chunk notification
            # can arrive.
            if self.preemptive_config.enabled:
                self._streaming_push_req_ids.add(req_id)

        # --- D-side: registrations to send to P via NIXL ---
        if metadata.push_registrations:
            for req_id, reg_data in metadata.push_registrations.items():
                self._reg_send_inbox.put((req_id, reg_data))
            self._push_writer_wake.set()

        # --- P-side: newly finished blocks awaiting a D registration match ---
        if metadata.push_finished_blocks:
            for req_id, block_ids in metadata.push_finished_blocks.items():
                self._finished_blocks_inbox.put((req_id, block_ids))
            self._push_writer_wake.set()

        if self.preemptive_config.enabled and metadata.push_chunks:
            # Keep the device-wide fence (model work can use non-current
            # streams), but wait for it on the writer thread so the engine
            # thread can submit the next compute batch immediately.
            event_info = getattr(metadata, "_pd_kv_ready", (None, time.perf_counter()))
            self._stream_source_sync_inbox.put(
                (list(metadata.push_chunks), event_info[0], event_info[1])
            )
            self._push_writer_wake.set()

        # Batch + lease tracking (same as pull).
        for req_id in metadata.reqs_in_batch:
            self._reqs_to_process.add(req_id)
        for req_id in metadata.reqs_not_processed:
            self._reqs_to_process.discard(req_id)
            assert req_id not in self._reqs_to_send
        # Rebase scheduler-clock deadlines onto this worker's clock — see the
        # equivalent block in pull_worker.start_load_kv for the rationale.
        now_local = time.perf_counter()
        for req_id, expiration_time in metadata.reqs_to_send.items():
            if req_id in self._reqs_to_process:
                if metadata.scheduler_clock:
                    expiration_time = now_local + (
                        expiration_time - metadata.scheduler_clock
                    )
                self._reqs_to_send[req_id] = expiration_time

        # Heartbeats still leave from the main thread (base worker behaviour).
        self._send_heartbeats(metadata)

    # --- Writer thread ------------------------------------------------- #

    def _push_writer_loop(self) -> None:
        sleep_s = _PUSH_WRITER_POLL_INTERVAL_MS / 1000.0

        while not self._push_writer_stop.is_set():
            try:
                # 1. D cancellations and registrations to send.
                while True:
                    try:
                        cancel_data = self._cancel_send_inbox.get_nowait()
                    except queue.Empty:
                        break
                    self._send_cancel_to_p(cancel_data)

                while True:
                    try:
                        rid, rd = self._reg_send_inbox.get_nowait()
                    except queue.Empty:
                        break
                    self._send_registration_to_p(rid, rd)

                # 2. Deferred P→D pushes whose handshake just completed; do xfer now
                while True:
                    try:
                        rid, blocks, rd = self._deferred_push_inbox.get_nowait()
                    except queue.Empty:
                        break
                    self._do_start_push_kv(rid, blocks, rd)

                # 3. P-side finished blocks; match against pending regs.
                while True:
                    try:
                        rid, blocks = self._finished_blocks_inbox.get_nowait()
                    except queue.Empty:
                        break
                    matched = self._pop_matching_registration(rid)
                    if matched is not None:
                        self._do_start_push_kv(rid, blocks, matched)
                    else:
                        self._push_finished_blocks[rid] = blocks

                if self.preemptive_config.enabled:
                    while True:
                        try:
                            source = self._stream_source_sync_inbox.get_nowait()
                        except queue.Empty:
                            break
                        self._synchronize_stream_sources(*source)
                    self._poll_stream_source_events()

                    while True:
                        try:
                            chunk = self._stream_chunk_inbox.get_nowait()
                        except queue.Empty:
                            break
                        self._accept_stream_chunk(chunk)

                    while True:
                        if not self._start_next_deferred_stream_chunk():
                            break

                    while True:
                        try:
                            chunk, success = self._stream_completion_inbox.get_nowait()
                        except queue.Empty:
                            break
                        self._process_stream_completion(chunk, success)

                    self._poll_active_stream_chunk()

                # 3b. Evict finished blocks for requests that have either
                # completed (WRITE acknowledged) or whose lease expired
                # without a D registration.  Drop pending registrations
                # for the same reason so we don't leak state.
                while True:
                    try:
                        rid = self._evict_finished_inbox.get_nowait()
                    except queue.Empty:
                        break
                    self._push_finished_blocks.pop(rid, None)
                    self._pending_d_registrations.pop(rid, None)

                # 4. NIXL notifs: route PUSH_REG; forward the rest.
                for notifs in self.nixl_wrapper.get_new_notifs().values():
                    for notif in notifs:
                        if notif.startswith(PUSH_REG_NOTIF_PREFIX):
                            self._handle_push_reg_notif(notif)
                        elif notif.startswith(b"VLLM_KV_STREAM_CANCEL:"):
                            self._handle_stream_cancel_notif(notif)
                        else:
                            self._pending_completion_notifs.put(notif)

                if self.preemptive_config.enabled and self._stream_slot_available():
                    chunk = self._stream_arbiter.pop_next(time.perf_counter())
                    if chunk is not None:
                        long_state = chunk.get("arbiter_long_state") or {}
                        protection_reasons = ",".join(
                            long_state.get("protection_reasons", [])
                        ) or "none"
                        logger.info(
                            "stream arbiter decision req=%s seq=%s tier=%s decision=%s "
                            "long_req=%s long_state=%s long_preemptions=%s "
                            "long_completed_blocks=%s long_total_blocks=%s "
                            "long_progress=%s long_wait_ms=%s protection_reasons=%s "
                            "monotonic_s=%.6f",
                            chunk.get("request_id"),
                            chunk.get("chunk_sequence"),
                            chunk.get("tier"),
                            chunk.get("arbiter_decision", "dispatch_fifo"),
                            chunk.get("arbiter_long_request_id"),
                            long_state,
                            long_state.get("preemptions"),
                            long_state.get("completed_blocks"),
                            long_state.get("total_blocks"),
                            long_state.get("progress"),
                            long_state.get("wait_ms"),
                            protection_reasons,
                            time.perf_counter(),
                        )
                        reg_data = self._find_stream_registration(
                            chunk["request_id"]
                        )
                        if reg_data is None:
                            self._stream_pending_chunks[chunk["request_id"]].insert(
                                0, chunk
                            )
                        else:
                            self._start_stream_chunk(chunk, reg_data)
            except Exception:
                logger.exception("nixl-push-writer error; continuing")

            # Self-poll only while there is no other wake source: P-side
            # finished blocks waiting for a D PUSH_REG match. All other
            # progress is event-driven (see module docstring).
            if self._push_finished_blocks or (
                self.preemptive_config.enabled
                and (self._stream_active is not None or bool(self._stream_arbiter))
                or self.preemptive_config.enabled
                and bool(self._stream_pending_chunks)
                or self.preemptive_config.enabled
                and bool(self._stream_source_events)
            ):
                self._push_writer_stop.wait(timeout=sleep_s)
            else:
                self._push_writer_wake.wait()
                self._push_writer_wake.clear()

    def _handle_push_reg_notif(self, notif: bytes) -> None:
        try:
            reg_data = msgspec.msgpack.decode(notif[len(PUSH_REG_NOTIF_PREFIX) :])
        except Exception:
            logger.exception("Failed to decode PUSH_REG notification payload")
            return
        rid = reg_data.get("request_id") if isinstance(reg_data, dict) else None
        if not isinstance(rid, str):
            logger.warning("PUSH_REG notif missing request_id; dropping")
            return

        if self.preemptive_config.enabled:
            received_config = reg_data.get("preemptive_transfer_config")
            if received_config != self.preemptive_config.wire_dict():
                logger.error(
                    "Preemptive transfer configuration mismatch for request %s; "
                    "expected %s, received %s",
                    rid,
                    self.preemptive_config.wire_dict(),
                    received_config,
                )
                self._stream_cancelled.add(rid)
                self._failed_recv_reqs.put(rid)
                return

            matching_ids = [
                request_id
                for request_id in self._stream_pending_chunks
                if get_base_request_id(request_id) == get_base_request_id(rid)
            ]
            if matching_ids:
                producer_req_id = matching_ids[0]
                self._stream_registrations[producer_req_id] = reg_data
                for chunk in self._stream_pending_chunks.pop(producer_req_id, []):
                    self._annotate_stream_progress(chunk, reg_data)
                    self._stream_arbiter.enqueue(chunk)
            else:
                self._stream_registrations[rid] = reg_data
            self._push_writer_wake.set()
            return

        match = self._pop_matching_finished_blocks(rid)
        if match is not None:
            fin_id, blocks = match
            self._do_start_push_kv(fin_id, blocks, reg_data)
        else:
            self._pending_d_registrations[rid] = reg_data

    def _handle_stream_cancel_notif(self, notif: bytes) -> None:
        try:
            _, producer_request_id = notif.decode("utf-8").split(":", 1)
        except (UnicodeDecodeError, ValueError):
            logger.exception("Malformed streaming cancellation notification: %r", notif)
            return
        base_id = get_base_request_id(producer_request_id)
        candidates = set(self._stream_request_state)
        candidates.update(self._stream_pending_chunks)
        candidates.update(self._stream_registrations)
        candidates.update(self._stream_arbiter.pending_request_ids())
        for request_id in candidates:
            if get_base_request_id(request_id) == base_id:
                self._cancel_stream_request(request_id)

    def _cancel_stream_request(self, request_id: str) -> None:
        logger.info("Received D cancellation for P stream req=%s", request_id)
        self._fail_stream_request(request_id, notify=False)

    def _find_stream_registration(
        self, request_id: str
    ) -> dict[str, Any] | None:
        reg = self._stream_registrations.get(request_id)
        if reg is not None:
            return reg
        base_id = get_base_request_id(request_id)
        for reg_id, candidate in self._stream_registrations.items():
            if get_base_request_id(reg_id) == base_id:
                return candidate
        return None

    def _accept_stream_chunk(self, chunk: dict[str, Any]) -> None:
        request_id = chunk.get("request_id")
        if not isinstance(request_id, str):
            logger.error("Dropping streaming KV chunk without request_id")
            return
        if request_id in self._stream_cancelled:
            # The scheduler may still emit chunks from a batch that was
            # already in flight when failure was observed. Do not recreate
            # request state or submit any more NIXL work for that stream.
            state = self._stream_request_state.get(request_id)
            if state is not None and (chunk.get("cancel") or chunk.get("is_final")):
                state["producer_done"] = True
                self._maybe_queue_failed_stream(request_id)
            self._complete_stream_chunk(chunk, False)
            return
        if chunk.get("cancel"):
            state = self._stream_request_state.get(request_id)
            if state is not None:
                state["producer_done"] = True
            self._fail_stream_request(request_id, notify=False)
            return

        self._streaming_push_req_ids.add(request_id)
        logger.info(
            "stream chunk accepted req=%s seq=%s start=%s blocks=%s total=%s final=%s monotonic_s=%.6f",
            request_id,
            chunk.get("chunk_sequence"),
            chunk.get("block_start"),
            sum(len(group) for group in chunk.get("local_block_ids", [])),
            chunk.get("local_total_blocks"),
            bool(chunk.get("is_final")),
            time.perf_counter(),
        )
        state = self._stream_request_state.setdefault(
            request_id,
            {"pending": 0, "final_completed": False, "failed": False},
        )
        state["pending"] += 1
        if chunk.get("is_final"):
            state["producer_done"] = True
        reg_data = self._find_stream_registration(request_id)
        if reg_data is None:
            self._stream_pending_chunks[request_id].append(chunk)
        else:
            self._stream_registrations[request_id] = reg_data
            self._annotate_stream_progress(chunk, reg_data)
            self._stream_arbiter.enqueue(chunk)

    def _synchronize_stream_sources(
        self, chunks: list[dict[str, Any]], event: Any = None, recorded: float = 0.0
    ) -> None:
        if event is not None:
            self._stream_source_events.append((chunks, event, recorded))
            return
        self._synchronize_stream_sources_device(chunks)

    def _poll_stream_source_events(self) -> None:
        pending = []
        for index, (chunks, event, recorded) in enumerate(self._stream_source_events):
            try:
                if not event.query():
                    # All batches share the producer stream. Never let a
                    # later event observed ready overtake an earlier query
                    # observed incomplete just before GPU progress occurred.
                    pending.extend(self._stream_source_events[index:])
                    break
                ready = time.perf_counter()
                logger.info(
                    "stream source KV event-ready chunks=%d wait_ms=%.3f monotonic_s=%.6f",
                    len(chunks), (ready - recorded) * 1000, ready,
                )
                for chunk in chunks:
                    self._stream_chunk_inbox.put(chunk)
            except Exception:
                logger.exception("Failed to query producer CUDA event")
                for chunk in chunks:
                    self._stream_chunk_inbox.put(chunk)
                    self._complete_stream_chunk(chunk, False)
        self._stream_source_events = pending

    def _synchronize_stream_sources_device(self, chunks: list[dict[str, Any]]) -> None:
        sync_start = time.perf_counter()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            logger.exception("Failed to synchronize CUDA before KV streaming")
            for chunk in chunks:
                self._stream_chunk_inbox.put(chunk)
                self._complete_stream_chunk(chunk, False)
            return
        logger.info(
            "stream source KV device-synchronized chunks=%d elapsed_ms=%.3f monotonic_s=%.6f",
            len(chunks),
            (time.perf_counter() - sync_start) * 1000,
            time.perf_counter(),
        )
        for chunk in chunks:
            self._stream_chunk_inbox.put(chunk)

    def _annotate_stream_progress(
        self, chunk: dict[str, Any], registration_data: dict[str, Any]
    ) -> None:
        """Count only blocks that D did not already have in its prefix cache."""
        local_groups = self._as_grouped_block_ids(chunk.get("local_block_ids", []))
        remote_groups = self._as_grouped_block_ids(
            registration_data.get("local_block_ids", [])
        )
        if len(local_groups) != 1 or len(remote_groups) != 1:
            chunk["progress_total_blocks"] = 0
            chunk["progress_blocks"] = 0
            return
        local_blocks = len(local_groups[0])
        total = max(0, int(chunk.get("local_total_blocks", 0)))
        remote_total = len(remote_groups[0])
        prefix_offset = max(0, total - remote_total)
        start = max(0, int(chunk.get("block_start", 0)))
        end = min(total, start + local_blocks)
        chunk["progress_total_blocks"] = remote_total
        chunk["progress_blocks"] = max(0, end - max(start, prefix_offset))

    def _start_stream_chunk(
        self, chunk: dict[str, Any], registration_data: dict[str, Any]
    ) -> None:
        request_id = chunk["request_id"]
        if request_id in self._stream_cancelled:
            self._complete_stream_chunk(chunk, False)
            return
        logger.info(
            "stream chunk start req=%s seq=%s block_start=%s blocks=%s final=%s monotonic_s=%.6f",
            request_id,
            chunk.get("chunk_sequence"),
            chunk.get("block_start"),
            sum(len(group) for group in chunk.get("local_block_ids", [])),
            bool(chunk.get("is_final")),
            time.perf_counter(),
        )
        if not chunk.get("local_block_ids") or not any(chunk["local_block_ids"]):
            # A terminal marker may carry no data. Send a notification so D
            # can learn the final chunk sequence without starting a WRITE.
            decode_engine_id = registration_data["decode_engine_id"]
            fut = self._ensure_handshake(
                decode_engine_id,
                registration_data["decode_host"],
                registration_data["decode_port"],
                registration_data["decode_tp_size"],
            )
            if fut is not None:
                key = (request_id, int(chunk["chunk_sequence"]))
                if key in self._stream_waiting_handshakes:
                    return
                self._stream_waiting_handshakes.add(key)
                self._stream_reserved_chunk = key

                def _on_marker_handshake(
                    f: Future[tuple[dict[tuple[int, int], str], float]],
                    task: dict[str, Any] = chunk,
                    reg: dict[str, Any] = registration_data,
                    task_key: tuple[str, int] = key,
                ) -> None:
                    self._stream_waiting_handshakes.discard(task_key)
                    if (e := f.exception()) is not None:
                        self._log_failure(
                            failure_type="push_handshake_failed",
                            req_id=task["request_id"],
                            error=e,
                        )
                        self._stream_completion_inbox.put((task, False))
                    else:
                        self._stream_deferred_inbox.put((task, reg))
                    self._push_writer_wake.set()

                fut.add_done_callback(_on_marker_handshake)
                return
            try:
                self._send_stream_chunk_notif(chunk, registration_data)
            except Exception as e:
                self._log_failure(
                    failure_type="notification_failed",
                    req_id=request_id,
                    error=e,
                    dst_engine_id=decode_engine_id,
                )
                self._complete_stream_chunk(chunk, False)
                return
            self._complete_stream_chunk(chunk, True)
            return

        decode_engine_id = registration_data["decode_engine_id"]
        fut = self._ensure_handshake(
            decode_engine_id,
            registration_data["decode_host"],
            registration_data["decode_port"],
            registration_data["decode_tp_size"],
        )
        if fut is not None:
            key = (request_id, int(chunk["chunk_sequence"]))
            if key in self._stream_waiting_handshakes:
                return
            self._stream_waiting_handshakes.add(key)
            self._stream_reserved_chunk = key

            def _on_chunk_handshake(
                f: Future[tuple[dict[tuple[int, int], str], float]],
                task: dict[str, Any] = chunk,
                reg: dict[str, Any] = registration_data,
                task_key: tuple[str, int] = key,
            ) -> None:
                self._stream_waiting_handshakes.discard(task_key)
                if (e := f.exception()) is not None:
                    self._log_failure(
                        failure_type="push_handshake_failed",
                        req_id=task["request_id"],
                        error=e,
                    )
                    self._stream_completion_inbox.put((task, False))
                else:
                    self._stream_deferred_inbox.put((task, reg))
                self._push_writer_wake.set()

            fut.add_done_callback(_on_chunk_handshake)
            return

        failure_injection = os.environ.get("NIXL_TEST_FAIL_STREAM_CHUNK", "")
        if failure_injection:
            failed_request, separator, failed_sequence = failure_injection.rpartition(":")
            if (
                separator
                and get_base_request_id(request_id) == failed_request
                and str(chunk.get("chunk_sequence")) == failed_sequence
            ):
                logger.error(
                    "Injected streaming WRITE failure req=%s seq=%s",
                    request_id,
                    failed_sequence,
                )
                self._complete_stream_chunk(chunk, False)
                return

        remote_groups = self._as_grouped_block_ids(
            registration_data["local_block_ids"]
        )
        local_groups = self._as_grouped_block_ids(chunk["local_block_ids"])
        if len(local_groups) != 1 or len(remote_groups) != 1:
            self._complete_stream_chunk(chunk, False)
            raise ValueError("Streaming NIXL push requires one cache group")
        remote_all = list(remote_groups[0])
        start = int(chunk["block_start"])
        total_local = int(chunk["local_total_blocks"])
        prefix_offset = total_local - len(remote_all)
        remote_start = start - prefix_offset
        local_ids = list(local_groups[0])
        if remote_start < 0:
            skip = min(len(local_ids), -remote_start)
            local_ids = local_ids[skip:]
            remote_start += skip
        remote_ids = remote_all[remote_start : remote_start + len(local_ids)]
        if not local_ids:
            # All blocks in this chunk are already present on D.
            try:
                self._send_stream_chunk_notif(chunk, registration_data)
            except Exception as e:
                self._log_failure(
                    failure_type="notification_failed",
                    req_id=request_id,
                    error=e,
                    dst_engine_id=decode_engine_id,
                )
                self._complete_stream_chunk(chunk, False)
                return
            self._complete_stream_chunk(chunk, True)
            return
        if len(local_ids) != len(remote_ids):
            logger.error(
                "Streaming KV block mapping mismatch for %s chunk %d: "
                "%d source blocks, %d destination blocks",
                request_id,
                chunk["chunk_sequence"],
                len(local_ids),
                len(remote_ids),
            )
            self._complete_stream_chunk(chunk, False)
            return

        logical_local = (local_ids,)
        logical_remote = (remote_ids,)
        physical_local = self._logical_to_kernel_block_ids(
            logical_local, self._physical_blocks_per_logical_kv_block
        )
        push_meta = ReqMeta(
            local_block_ids=logical_local,
            local_physical_block_ids=physical_local,
            tp_size=self.world_size,
            remote=RemoteMeta(
                block_ids=logical_remote,
                host="",
                port=0,
                engine_id=decode_engine_id,
                request_id=registration_data["request_id"],
            ),
        )
        self._engine_last_active[decode_engine_id] = time.perf_counter()
        handles = self._xfer_blocks_for_req(
            req_id=request_id,
            meta=push_meta,
            notif_id=self._stream_notif(chunk, registration_data),
            track_handles=False,
        )
        if not handles:
            with self._sending_transfers_lock:
                transfer_failed = request_id in self._send_failures
            if transfer_failed:
                self._complete_stream_chunk(chunk, False)
                return
            try:
                self._send_stream_chunk_notif(chunk, registration_data)
            except Exception as e:
                self._log_failure(
                    failure_type="notification_failed",
                    req_id=request_id,
                    error=e,
                    dst_engine_id=decode_engine_id,
                )
                self._complete_stream_chunk(chunk, False)
                return
            self._complete_stream_chunk(chunk, True)
            return
        self._stream_active = (chunk, handles)

    @staticmethod
    def _stream_notif(
        chunk: dict[str, Any], registration_data: dict[str, Any]
    ) -> bytes:
        return (
            f"VLLM_KV_CHUNK:{registration_data['request_id']}:"
            f"{int(chunk['chunk_sequence'])}:{int(bool(chunk.get('is_final')))}:"
            f"{int(registration_data['decode_tp_size'])}"
        ).encode()

    def _send_stream_chunk_notif(
        self, chunk: dict[str, Any], registration_data: dict[str, Any]
    ) -> None:
        """Notify D for a successful chunk that had no completed WRITE."""
        decode_engine_id = registration_data["decode_engine_id"]
        agent = self._remote_agents[decode_engine_id][(0, 0)]
        self.nixl_wrapper.send_notif(
            agent,
            notif_msg=self._stream_notif(chunk, registration_data),
        )

    def _complete_stream_chunk(self, chunk: dict[str, Any], success: bool) -> None:
        self._stream_completion_inbox.put((chunk, success))
        self._push_writer_wake.set()

    def _fail_stream_request(self, request_id: str, *, notify: bool = True) -> None:
        """Cancel queued stream work and release the request lease once.

        This is called only on the writer thread (or by a cancel chunk queued
        to it), so it can clear all writer-owned maps atomically. Any active
        WRITE is allowed to finish through ``_poll_active_stream_chunk``;
        failure cleanup therefore never releases a source block while DMA may
        still be reading it.
        """
        already_cancelled = request_id in self._stream_cancelled
        registration = self._find_stream_registration(request_id)
        state = self._stream_request_state.get(request_id)
        active = getattr(self, "_stream_active", None)
        active_for_request = active is not None and active[0].get("request_id") == request_id
        self._stream_cancelled.add(request_id)
        self._stream_arbiter.cancel(request_id)
        # Remove deferred handshakes for the failed stream. They may complete
        # after the remote agent has been aborted; starting them would submit
        # against a disconnected engine and turn a request failure into an
        # EngineCore crash.
        deferred_inbox = getattr(self, "_stream_deferred_inbox", None)
        if deferred_inbox is not None:
            deferred: list[tuple[dict[str, Any], dict[str, Any]]] = []
            while True:
                try:
                    item = deferred_inbox.get_nowait()
                except queue.Empty:
                    break
                if item[0].get("request_id") != request_id:
                    deferred.append(item)
            for item in deferred:
                deferred_inbox.put(item)
        self._stream_pending_chunks.pop(request_id, None)
        self._stream_registrations.pop(request_id, None)
        self._stream_waiting_handshakes = {
            key for key in self._stream_waiting_handshakes if key[0] != request_id
        }
        reserved = getattr(self, "_stream_reserved_chunk", None)
        if reserved is not None and reserved[0] == request_id:
            self._stream_reserved_chunk = None
        self._streaming_push_req_ids.discard(request_id)
        if state is not None:
            # Keep state alive while a WRITE for this request is still in
            # flight.  Its source blocks cannot be returned to the scheduler
            # until the DMA handle reaches a release boundary.
            state["failed"] = True
            state["pending"] = 1 if active_for_request else 0
        if not already_cancelled and state is not None:
            if notify and registration is not None:
                # The D side treats this as a failed receive and releases its
                # request without pretending that missing chunks arrived.
                try:
                    agent = self._remote_agents[registration["decode_engine_id"]][
                        (0, 0)
                    ]
                    self.nixl_wrapper.send_notif(
                        agent,
                        notif_msg=(
                            f"VLLM_KV_CHUNK_FAIL:{registration['request_id']}:-1:"
                            f"{int(registration['decode_tp_size'])}"
                        ).encode(),
                    )
                except Exception:
                    logger.exception(
                        "Could not notify Decode of failed KV stream for %s",
                        request_id,
                    )
        self._maybe_queue_failed_stream(request_id)

    def _maybe_queue_failed_stream(self, request_id: str) -> None:
        state = self._stream_request_state.get(request_id)
        active = getattr(self, "_stream_active", None)
        active_for_request = active is not None and active[0].get("request_id") == request_id
        if (
            state is None
            or not state.get("failed")
            or not state.get("producer_done")
            or active_for_request
            or state.get("failure_queued", False)
        ):
            return
        state["failure_queued"] = True
        self._stream_request_state.pop(request_id, None)
        self._stream_failed_sending.put(request_id)
        self._push_writer_wake.set()

    def _stream_slot_available(self) -> bool:
        return (
            self._stream_active is None
            and getattr(self, "_stream_reserved_chunk", None) is None
        )

    def _start_next_deferred_stream_chunk(self) -> bool:
        """Start one deferred chunk only when the single WRITE slot is free."""
        if self._stream_active is not None:
            return False
        try:
            chunk, registration = self._stream_deferred_inbox.get_nowait()
        except queue.Empty:
            return False
        key = (chunk["request_id"], int(chunk["chunk_sequence"]))
        if getattr(self, "_stream_reserved_chunk", None) == key:
            self._stream_reserved_chunk = None
        if chunk.get("request_id") in self._stream_cancelled:
            self._complete_stream_chunk(chunk, False)
        else:
            self._start_stream_chunk(chunk, registration)
        return True

    def _process_stream_completion(
        self, chunk: dict[str, Any], success: bool
    ) -> None:
        request_id = chunk["request_id"]
        key = (request_id, int(chunk["chunk_sequence"]))
        if getattr(self, "_stream_reserved_chunk", None) == key:
            self._stream_reserved_chunk = None
        logger.info(
            "stream chunk complete req=%s seq=%s success=%s final=%s monotonic_s=%.6f",
            request_id,
            chunk.get("chunk_sequence"),
            success,
            bool(chunk.get("is_final")),
            time.perf_counter(),
        )
        # Failed chunks never contribute to the long-request success
        # denominator.  The arbiter is intentionally updated only after the
        # transfer/notification outcome is known.
        if success:
            self._stream_arbiter.mark_chunk_complete(chunk, time.perf_counter())
        state = self._stream_request_state.get(request_id)
        if state is None:
            return
        state["pending"] = max(0, state["pending"] - 1)
        if state.get("failed") or request_id in self._stream_cancelled:
            if chunk.get("is_final"):
                state["producer_done"] = True
            self._fail_stream_request(request_id, notify=False)
            return
        if not success:
            logger.error("Streaming KV transfer failed for request %s", request_id)
            if chunk.get("is_final"):
                state["producer_done"] = True
            self._fail_stream_request(request_id)
            return
        if chunk.get("is_final") and success:
            state["final_completed"] = True
            state["final_sequence"] = int(chunk["chunk_sequence"])
        # A final chunk can finish before an earlier deferred chunk in legacy
        # traces. Only all successful completions authorize STREAM_DONE.
        if state["pending"] == 0 and state["final_completed"] and not state["failed"]:
            if not state.get("summary_sent", False):
                registration = self._find_stream_registration(request_id)
                if registration is not None:
                    try:
                        final_sequence = state["final_sequence"]
                        agent = self._remote_agents[registration["decode_engine_id"]][
                            (0, 0)
                        ]
                        done_msg = (
                            f"VLLM_KV_STREAM_DONE:{registration['request_id']}:"
                            f"{final_sequence}:{int(registration['decode_tp_size'])}"
                        ).encode()
                        for _ in range(3):
                            self.nixl_wrapper.send_notif(agent, notif_msg=done_msg)
                        state["summary_sent"] = True
                    except Exception:
                        logger.exception(
                            "Could not notify Decode of stream completion for %s",
                            request_id,
                        )
        if state["pending"] == 0 and state["final_completed"] and not state["failed"]:
            self._stream_completed_sending.put(request_id)
            self._stream_request_state.pop(request_id, None)
            self._streaming_push_req_ids.discard(request_id)
            self._stream_registrations.pop(request_id, None)

    def _poll_active_stream_chunk(self) -> None:
        if self._stream_active is None:
            return
        chunk, handles = self._stream_active
        in_progress: list[int] = []
        success = True
        for handle in handles:
            try:
                state = self.nixl_wrapper.check_xfer_state(handle)
                if state == "DONE":
                    try:
                        telemetry = self.nixl_wrapper.get_xfer_telemetry(handle)
                        self.xfer_stats.record_transfer(telemetry)
                    except Exception:
                        logger.debug(
                            "NIXL telemetry unavailable for streaming chunk %s",
                            chunk.get("chunk_sequence"),
                            exc_info=True,
                        )
                    if not self._try_release_xfer_handle(chunk["request_id"], handle):
                        in_progress.append(handle)
                        success = False
                elif state == "PROC":
                    in_progress.append(handle)
                else:
                    self._log_failure(
                        failure_type="transfer_failed",
                        req_id=chunk["request_id"],
                        xfer_state=state,
                    )
                    success = False
                    if not self._handle_failed_transfer(
                        chunk["request_id"], handle
                    ):
                        in_progress.append(handle)
            except Exception as e:
                self._log_failure(
                    failure_type="transfer_exception",
                    req_id=chunk["request_id"],
                    error=e,
                )
                success = False
                if not self._handle_failed_transfer(chunk["request_id"], handle):
                    in_progress.append(handle)
        if in_progress:
            self._stream_active = (chunk, in_progress)
            return
        self._stream_active = None
        self._complete_stream_chunk(chunk, success)

    # --- D-side registration send (writer thread) ---------------------- #

    def _send_registration_to_p(
        self,
        req_id: str,
        reg_data: dict[str, Any],
    ) -> None:
        """Handshake (if needed) then send PUSH_REG. ``send_notif`` always
        executes on the writer; the handshake runs on the background executor
        and the request is re-queued onto ``_reg_send_inbox`` once it
        completes (at which point ``_ensure_handshake`` returns ``None`` and we
        send directly)."""
        remote_pp_size = reg_data.get("remote_pp_size", 1)
        fut = self._ensure_handshake(
            reg_data["remote_engine_id"],
            reg_data["remote_host"],
            reg_data["remote_port"],
            reg_data["remote_tp_size"],
            pp_size=remote_pp_size,
            # D only ever sends PUSH_REG notifs to P and never reads or writes
            # P's memory in push mode, so it never needs the transfer
            # descriptors set up by the full add_remote_agent path.
            notif_agents_only=True,
        )
        if fut is None:
            self._do_send_reg_notif(req_id, reg_data)
            return

        def _on_handshake(
            f: Future[tuple[dict[tuple[int, int], str], float]],
            rid: str = req_id,
            rd: dict[str, Any] = reg_data,
        ) -> None:
            try:
                f.result()
            except Exception as e:
                self._log_failure(
                    failure_type="push_reg_handshake_failed", req_id=rid, error=e
                )
                self._failed_recv_reqs.put(rid)
                return
            # Re-queue for the writer to send now that the handshake is done.
            self._reg_send_inbox.put((rid, rd))
            # Wake the writer so it sends the PUSH_REG promptly even if
            # otherwise parked.
            self._push_writer_wake.set()

        fut.add_done_callback(_on_handshake)

    def _send_cancel_to_p(self, cancel_data: dict[str, Any]) -> None:
        """Handshake if needed, then ask P to cancel the producer request."""
        remote_engine_id = cancel_data["remote_engine_id"]
        fut = self._ensure_handshake(
            remote_engine_id,
            cancel_data["remote_host"],
            cancel_data["remote_port"],
            cancel_data["remote_tp_size"],
            pp_size=cancel_data.get("remote_pp_size", 1),
            notif_agents_only=True,
        )
        if fut is not None:

            def _on_handshake(
                f: Future[tuple[dict[tuple[int, int], str], float]],
                data: dict[str, Any] = cancel_data,
            ) -> None:
                try:
                    f.result()
                except Exception as exc:
                    logger.warning(
                        "Could not handshake with P to cancel request %s: %s",
                        data["producer_request_id"],
                        exc,
                    )
                    return
                self._cancel_send_inbox.put(data)
                self._push_writer_wake.set()

            fut.add_done_callback(_on_handshake)
            return

        agent = self._remote_agents[remote_engine_id][(0, 0)]
        notif = f"VLLM_KV_STREAM_CANCEL:{cancel_data['producer_request_id']}".encode()
        self.nixl_wrapper.send_notif(agent, notif_msg=notif)
        logger.info(
            "Sent D cancellation to P req=%s producer_req=%s",
            cancel_data["request_id"],
            cancel_data["producer_request_id"],
        )

    def _do_send_reg_notif(self, req_id: str, reg_data: dict[str, Any]) -> None:
        engine_id = reg_data["remote_engine_id"]
        notif_msg = PUSH_REG_NOTIF_PREFIX + msgspec.msgpack.encode(reg_data)
        # _remote_agents is mutated on other threads; snapshot under the lock.
        with self._handshake_lock:
            agents = dict(self._remote_agents.get(engine_id) or {})
        if not agents:
            logger.error(
                "No remote agents for engine %s; cannot send registration for %s",
                engine_id,
                req_id,
            )
            self.xfer_stats.record_failed_notification()
            self._failed_recv_reqs.put(req_id)
            return
        for rank, agent_name in agents.items():
            try:
                self.nixl_wrapper.send_notif(agent_name, notif_msg=notif_msg)
            except Exception as e:
                self._log_failure(
                    failure_type="push_reg_notif_failed",
                    req_id=req_id,
                    error=e,
                    remote_rank=rank,
                )
                self.xfer_stats.record_failed_notification()
                # Earlier registrations may still trigger WRITEs into D's blocks.
                # Keep the receive pending until those writes are finished.
        logger.debug(
            "Sent PUSH_REG for %s to engine %s (%dB)", req_id, engine_id, len(notif_msg)
        )

    # --- Matching helpers --------------------------------------------- #

    def _pop_matching_registration(self, request_id: str) -> dict[str, Any] | None:
        """Pop the D-side registration matching *request_id*.

        Exact key first, then a match after stripping the random suffix from
        both sides. No match leaves the request unmatched (push not started).
        """
        data = self._pending_d_registrations.pop(request_id, None)
        if data is not None:
            return data
        base_id = get_base_request_id(request_id)
        for reg_id in list(self._pending_d_registrations):
            if get_base_request_id(reg_id) == base_id:
                return self._pending_d_registrations.pop(reg_id)
        return None

    def _pop_matching_finished_blocks(
        self, request_id: str
    ) -> tuple[str, BlockIds] | None:
        """Pop the P-side finished blocks matching *request_id*.

        Same lookup as ``_pop_matching_registration``: exact key, then a
        match after stripping the random suffix from both sides.
        """
        blocks = self._push_finished_blocks.pop(request_id, None)
        if blocks is not None:
            return request_id, blocks
        base_id = get_base_request_id(request_id)
        for fin_id in list(self._push_finished_blocks):
            if get_base_request_id(fin_id) == base_id:
                return fin_id, self._push_finished_blocks.pop(fin_id)
        return None

    # --- WRITE transfer logic (writer thread) ------------------------- #

    def _do_start_push_kv(
        self,
        request_id: str,
        local_block_ids: BlockIds,
        registration_data: dict[str, Any],
    ) -> None:
        """Start push-based KV transfer from P worker to D node.

        The P→D handshake runs on the base worker's background executor.
        If it isn't ready yet we register a completion callback, defer the
        WRITE, and re-drive this request via ``_deferred_push_inbox`` once
        the handshake resolves -- so the writer thread never blocks on the
        network (mirrors ``_send_registration_to_p``).
        """
        if not local_block_ids:
            logger.warning("No local blocks to push for request %s", request_id)
            return

        # ``local_block_ids`` are P's logical block IDs; ``remote_block_ids``
        # (D's, from the PUSH_REG notif) are also logical.
        decode_engine_id = registration_data["decode_engine_id"]
        remote_block_ids = registration_data["local_block_ids"]
        decode_request_id = registration_data["request_id"]

        # Runs on the background executor; defer the WRITE until it's ready.
        fut = self._ensure_handshake(
            decode_engine_id,
            registration_data["decode_host"],
            registration_data["decode_port"],
            registration_data["decode_tp_size"],
        )
        if fut is not None:

            def _on_handshake(
                f: Future[tuple[dict[tuple[int, int], str], float]],
                rid: str = request_id,
                blocks: BlockIds = local_block_ids,
                rd: dict[str, Any] = registration_data,
            ) -> None:
                if (e := f.exception()) is not None:
                    # The engine reclaims the blocks via the TTL so we dont free here
                    self._log_failure(
                        failure_type="push_handshake_failed", req_id=rid, error=e
                    )
                    return
                self._deferred_push_inbox.put((rid, blocks, rd))
                self._push_writer_wake.set()

            fut.add_done_callback(_on_handshake)
            return
        # Keep the engine alive while it is actively receiving pushes, mirroring
        # how pull-mode transfers touch _engine_last_active in start_load_kv.
        self._engine_last_active[decode_engine_id] = time.perf_counter()

        logical_local = self._as_grouped_block_ids(local_block_ids)
        logical_remote = self._as_grouped_block_ids(remote_block_ids)
        physical_local = self._logical_to_kernel_block_ids(
            logical_local, self._physical_blocks_per_logical_kv_block
        )

        push_meta = ReqMeta(
            local_block_ids=logical_local,
            local_physical_block_ids=physical_local,
            tp_size=self.world_size,
            remote=RemoteMeta(
                block_ids=logical_remote,
                host="",
                port=0,
                engine_id=decode_engine_id,
                request_id=decode_request_id,
            ),
        )

        t0 = time.perf_counter()
        self._xfer_blocks_for_req(req_id=request_id, meta=push_meta)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if elapsed_ms > 200.0:
            logger.warning(
                "_do_start_push_kv for %s took %.1fms (slow NIXL submission)",
                request_id,
                elapsed_ms,
            )

    @staticmethod
    def _as_grouped_block_ids(block_ids: BlockIds) -> BlockIds:
        """Normalise a sequence of block IDs to a tuple-of-groups shape.

        ``BlockIds`` is canonically a tuple of per-group lists, but some
        registration payloads collapse a single-group case to a flat
        list. Re-wrap that case so downstream group-aware helpers see a
        consistent shape."""
        if block_ids and not isinstance(block_ids[0], (list, tuple)):
            return (list(block_ids),)
        return block_ids

    def _xfer_blocks_for_req(
        self,
        req_id: str,
        meta: ReqMeta,
        notif_id: bytes | None = None,
        track_handles: bool = True,
    ) -> list[int]:
        """Issue WRITE transfers to one or more remote TP ranks."""
        assert meta.remote is not None and self.transfer_topo is not None
        engine_id = meta.remote.engine_id
        plan = self.tp_mappings[engine_id]
        remote_info = self.transfer_topo.get_engine_info(engine_id)
        tp_ratio = self.transfer_topo.tp_ratio(remote_info.remote_tp_size)

        # Expand D's logical IDs using the ratio learned during the
        # NIXL handshake. ``meta`` is freshly built by
        # ``_do_start_push_kv`` so mutating it here is safe.
        meta.remote.block_ids = self._logical_to_kernel_block_ids(
            meta.remote.block_ids,
            remote_info.remote_physical_blocks_per_logical,
        )
        remote_block_ids = meta.remote.block_ids
        local_block_ids = meta.local_physical_block_ids
        local_region_groups = self.region_group_ids
        remote_region_groups = self.dst_region_group_ids[engine_id]
        groups_differ = local_region_groups != remote_region_groups
        if groups_differ and not self._transfer_layer_group_ids:
            raise NotImplementedError(
                "NixlPushConnector does not support different producer and "
                "consumer cache-group layouts"
            )

        # MLA latent is replicated across D's TP ranks: the tp-mapping
        # collapses it to one rank (fine for reads), but push must WRITE every
        # D rank or the rest decode stale KV. For hybrid MLA+SSM the sharded
        # SSM state already targets every covered D rank, so only the
        # attention groups need widening; pure MLA writes to all handshaked
        # ranks (only the dst differs per rank).
        replicate_attn = self.use_mla and tp_ratio < 0
        if replicate_attn and not self._has_mamba:
            assert len(plan.all_source_ranks) == 1
            write_ranks = sorted(self.dst_xfer_side_handles[engine_id])
        else:
            write_ranks = list(plan.all_source_ranks)

        num_groups = len(local_block_ids)

        def group_ids(block_ids: BlockIds, rank: int) -> BlockIds:
            return [
                list(block_ids[g])
                if (self._is_csa_linear and tp_ratio < 0)
                or (replicate_attn and _is_attention_spec(self._group_spec_types[g]))
                or rank in plan.source_ranks_per_group[g]
                else []
                for g in range(num_groups)
            ]

        read_specs = [
            ReadSpec(
                remote_rank=rank,
                local_block_ids=group_ids(local_block_ids, rank),
                remote_block_ids=group_ids(remote_block_ids, rank),
            )
            for rank in write_ranks
        ]

        handles: list[int] = []
        for i, spec in enumerate(read_specs):
            remote_block_size = remote_info.remote_block_size
            logger.debug(
                "Remote agent %s available, calling _xfer_blocks"
                " on remote rank %s with remote block size %s for req %s",
                meta.remote.engine_id,
                spec.remote_rank,
                remote_block_size,
                req_id,
            )
            if tp_ratio < 0 and (not self.use_mla or len(plan.all_source_ranks) > 1):
                # Multiple targets: write each rank its chunk of local memory.
                # Hybrid MLA+SSM also lands here: its split handles replicate
                # the attention descriptors and chunk only the SSM state.
                split_key = (tp_ratio, remote_block_size)
                local_xfer_side_handle = self.src_xfer_handles_by_tp_ratio[split_key][i]
            else:
                local_xfer_side_handle = self.src_xfer_handles_by_block_size[
                    remote_block_size
                ]

            remote_xfer_side_handle = self.dst_xfer_side_handles[meta.remote.engine_id][
                spec.remote_rank
            ]

            handle = self._xfer_blocks(
                read_spec=spec,
                request_id=req_id,
                dst_engine_id=meta.remote.engine_id,
                remote_request_id=meta.remote.request_id,
                local_xfer_side_handle=local_xfer_side_handle,
                remote_xfer_side_handle=remote_xfer_side_handle,
                notif_id=notif_id,
            )
            if handle is not None:
                handles.append(handle)

        # Publish all the request's WRITE handles in one locked update: a
        # partial set would let ``_pop_done_transfers`` finish the request
        # early, then double-report it as the remaining writes land.
        if handles and track_handles:
            with self._sending_transfers_lock:
                self._sending_transfers[req_id].extend(handles)
        return handles

    def _xfer_blocks(
        self,
        read_spec: ReadSpec,
        dst_engine_id: str,
        request_id: str,
        remote_request_id: str,
        local_xfer_side_handle: int,
        remote_xfer_side_handle: int,
        notif_id: bytes | None = None,
    ) -> int | None:
        """Post a WRITE point-to-point xfer request.

        Returns the in-flight transfer handle (so the caller can track all of
        a request's handles atomically), or ``None`` if nothing was submitted.
        """
        assert self.transfer_topo is not None
        remote_rank = read_spec.remote_rank
        local_block_ids = read_spec.local_block_ids
        remote_block_ids = read_spec.remote_block_ids

        remote_info = self.transfer_topo.get_engine_info(dst_engine_id)
        block_size_ratio = self.transfer_topo.block_size_ratio(
            remote_info.remote_block_size
        )
        if block_size_ratio > 1:
            local_block_ids, remote_block_ids = (
                self._map_block_ids_for_block_size_ratio(
                    local_block_ids, remote_block_ids, block_size_ratio
                )
            )

        if notif_id is None:
            notif_id = f"{remote_request_id}:{self.world_size}".encode()

        if len(local_block_ids) == 0:
            logger.warning("No blocks to push for request %s", request_id)
            return None

        # Prefix caching: D allocated only uncached blocks, so on a partial hit it
        # sends fewer than P's. End-trim P's blocks to that same suffix so we WRITE only
        # the uncomputed tail into D's slots. Runs on kernel ids, post-expansion.
        remote_block_ids, local_block_ids = self._apply_prefix_caching(
            decode_block_ids=remote_block_ids,
            prefill_block_ids=local_block_ids,
            decode_physical_per_logical=remote_info.remote_physical_blocks_per_logical,
            prefill_physical_per_logical=self._physical_blocks_per_logical_kv_block,
        )

        local_block_ids = list(local_block_ids)
        remote_block_ids = list(remote_block_ids)
        assert len(local_block_ids) == len(remote_block_ids), (
            f"push group-count mismatch for {request_id}: {len(local_block_ids)} "
            f"local vs {len(remote_block_ids)} remote groups"
        )
        for i in range(len(local_block_ids)):
            assert len(local_block_ids[i]) == len(remote_block_ids[i]), (
                f"push block-count mismatch for {request_id} group {i}: "
                f"{len(local_block_ids[i])} local vs "
                f"{len(remote_block_ids[i])} remote blocks"
            )

        # Get descs ids.
        remote_block_descs_ids = self._compute_desc_ids(
            block_ids=remote_block_ids,
            dst_num_blocks=self.dst_num_blocks[dst_engine_id],
            block_size_ratio=None,
            physical_blocks_per_logical=remote_info.remote_physical_blocks_per_logical,
            region_num_blocks=self.dst_region_num_blocks[dst_engine_id],
            region_group_ids=self.dst_region_group_ids[dst_engine_id],
            uses_region_group_mapping=self.dst_uses_region_group_mapping[dst_engine_id],
        )
        local_block_descs_ids = self._compute_desc_ids(
            block_ids=local_block_ids,
            dst_num_blocks=self.dst_num_blocks[self.engine_id],
            block_size_ratio=block_size_ratio,
            physical_blocks_per_logical=self._physical_blocks_per_logical_kv_block,
            region_num_blocks=self.dst_region_num_blocks[self.engine_id],
            region_group_ids=self.region_group_ids,
            uses_region_group_mapping=self._uses_region_group_mapping,
        )

        assert len(local_block_descs_ids) == len(remote_block_descs_ids)

        handle = None
        try:
            handle = self.nixl_wrapper.make_prepped_xfer(
                "WRITE",
                local_xfer_side_handle,
                local_block_descs_ids,
                remote_xfer_side_handle,
                remote_block_descs_ids,
                notif_msg=notif_id,
            )
            self.nixl_wrapper.transfer(handle)
            # Caller tracks the handle (atomically with the request's other
            # writes) so P can free blocks once all of them are done.
            return handle
        except Exception as e:
            self._log_failure(
                failure_type="transfer_setup_failed",
                req_id=request_id,
                msg="Push WRITE submission failed; releasing handle",
                error=e,
                dst_engine_id=dst_engine_id,
                remote_rank=remote_rank,
            )
            # On the P side this WRITE failure is purely outbound; we
            # don't have a ``_recving_metadata`` entry to invalidate, so
            # we just release the handle and let the engine reschedule
            # via the lease / watchdog.
            with self._sending_transfers_lock:
                self._send_failures.add(request_id)
            if not self._handle_failed_transfer(request_id, handle):
                return handle
            return None

    # --- Notification handling on engine main thread ------------------ #

    def _get_new_notifs(self) -> set[str]:
        """Drain HB / completion notifs forwarded by the writer thread.

        The writer owns ``nixl_wrapper.get_new_notifs`` for push; PUSH_REG
        notifs are handled there. Everything else is forwarded here for
        existing accounting.
        """
        assert self.transfer_topo is not None
        notified_req_ids: set[str] = set()
        while True:
            try:
                notif = self._pending_completion_notifs.get_nowait()
            except queue.Empty:
                break

            msg = notif.decode("utf-8")
            if msg.startswith("HB:"):
                self._handle_heartbeat(msg[3:])
                continue

            if self.preemptive_config.enabled and msg.startswith("VLLM_KV_CHUNK:"):
                try:
                    _, req_id, sequence_s, final_s, _tp_s = msg.split(":", 4)
                    sequence = int(sequence_s)
                    is_final = bool(int(final_s))
                except (ValueError, TypeError):
                    logger.exception("Malformed streaming KV completion: %r", msg)
                    continue
                if req_id not in self._streaming_push_req_ids:
                    logger.warning(
                        "Ignoring chunk completion for untracked request %s", req_id
                    )
                    continue
                self._received_chunk_ids[req_id].add(sequence)
                logger.info(
                    "stream chunk received req=%s seq=%s final=%s received=%s monotonic_s=%.6f",
                    req_id,
                    sequence,
                    is_final,
                    sorted(self._received_chunk_ids[req_id]),
                    time.perf_counter(),
                )
                if is_final:
                    self._final_chunk_sequence[req_id] = sequence
                final_sequence = self._final_chunk_sequence.get(req_id)
                if final_sequence is not None and self._received_chunk_ids[req_id] >= set(
                    range(final_sequence + 1)
                ):
                    # This is a consumer completion. Mark the receive
                    # transfer done below; returning the same ID as
                    # ``done_sending`` makes the scheduler free a live D
                    # request's blocks and trips its finished-request guard.
                    self._received_chunk_ids.pop(req_id, None)
                    self._final_chunk_sequence.pop(req_id, None)
                    self._streaming_push_req_ids.discard(req_id)
                    self._stream_request_state.pop(req_id, None)
                    self._recving_transfers.setdefault(req_id, [])
                    logger.info(
                        "stream receive complete req=%s final_seq=%s monotonic_s=%.6f",
                        req_id,
                        final_sequence,
                        time.perf_counter(),
                    )
                continue

            if self.preemptive_config.enabled and msg.startswith("VLLM_KV_CHUNK_FAIL:"):
                try:
                    _, req_id, _sequence_s, _tp_s = msg.split(":", 3)
                except ValueError:
                    logger.exception("Malformed streaming KV failure: %r", msg)
                    continue
                if req_id in self._recving_metadata:
                    self._recv_failures.add(req_id)
                    self._recving_transfers.setdefault(req_id, [])
                continue

            if self.preemptive_config.enabled and msg.startswith("VLLM_KV_STREAM_DONE:"):
                try:
                    _, req_id, final_s, _tp_s = msg.split(":", 3)
                    final_sequence = int(final_s)
                except (ValueError, TypeError):
                    logger.exception("Malformed streaming summary: %r", msg)
                    continue
                if req_id not in self._streaming_push_req_ids:
                    logger.warning(
                        "Ignoring stream summary for untracked request %s", req_id
                    )
                    continue
                self._received_chunk_ids[req_id].update(
                    range(final_sequence + 1)
                )
                self._final_chunk_sequence[req_id] = final_sequence
                self._received_chunk_ids.pop(req_id, None)
                self._final_chunk_sequence.pop(req_id, None)
                self._streaming_push_req_ids.discard(req_id)
                self._stream_request_state.pop(req_id, None)
                self._recving_transfers.setdefault(req_id, [])
                logger.info(
                    "stream receive complete req=%s final_seq=%s via summary monotonic_s=%.6f",
                    req_id,
                    final_sequence,
                    time.perf_counter(),
                )
                continue

            req_id, tp_size = msg.rsplit(":", 1)

            # Not tracked as a P-side send/process for this notif.
            if req_id not in self._reqs_to_send and req_id not in self._reqs_to_process:
                if (meta := self._recving_metadata.get(req_id)) is not None:
                    # Consumer waits for one notif per producer rank writing
                    # here: pp_size stages * producers-per-consumer (>1 when
                    # producer TP > consumer TP; tp_size is the producer TP).
                    producers_per_consumer = max(1, int(tp_size) // self.world_size)
                    expected_notifs = meta.pp_size * producers_per_consumer
                    self.consumer_notification_counts_by_req[req_id] += 1
                    notifs = self.consumer_notification_counts_by_req[req_id]
                    if notifs < expected_notifs:
                        continue
                    del self.consumer_notification_counts_by_req[req_id]
                    # P drove the transfer (we own no NIXL handle), so
                    # materialise an empty ``_recving_transfers`` entry for
                    # ``_pop_done_transfers`` to report done.
                    self._recving_transfers.setdefault(req_id, [])
                else:
                    # Not tracked on either side (lease may have expired
                    # before the notif arrived). Log and skip.
                    self.xfer_stats.record_notification_after_expiry()
                    logger.error(
                        "Unrecognized request %s notif (may have expired).",
                        req_id,
                    )
                continue

            n_consumers = int(tp_size)
            tp_ratio = self.transfer_topo.tp_ratio(n_consumers)
            consumers_per_producer = -tp_ratio if n_consumers > self.world_size else 1
            self.consumer_notification_counts_by_req[req_id] += 1
            if (
                self.consumer_notification_counts_by_req[req_id]
                == consumers_per_producer
            ):
                notified_req_ids.add(req_id)
                del self.consumer_notification_counts_by_req[req_id]
                self._reqs_to_process.remove(req_id)
                self._reqs_to_send.pop(req_id, None)
        return notified_req_ids

    def get_transfer_results(self) -> KVConnectorTransferResults:
        # Engine main thread asking for completions: also wake the writer
        # so it gets a chance to drain NIXL notifs (heartbeats, completion
        # notifs, late PUSH_REGs) even if it had been parked.
        self._push_writer_wake.set()

        results = super().get_transfer_results()
        done_sending = results.finished_sending

        if self.preemptive_config.enabled:
            success_pending = getattr(self, "_stream_success_pending", set())
            self._stream_success_pending = success_pending
            while True:
                try:
                    req_id = self._stream_completed_sending.get_nowait()
                    success_pending.add(req_id)
                except queue.Empty:
                    break
            for req_id in tuple(success_pending):
                # Prompt-final chunks may complete before scheduler final
                # metadata installs the send lease. Preserve the completion
                # until that boundary instead of freeing a running request.
                if req_id not in self._reqs_to_send:
                    if req_id not in self._reqs_to_process:
                        success_pending.discard(req_id)
                    continue
                done_sending.add(req_id)
                self._reqs_to_send.pop(req_id, None)
                self._reqs_to_process.discard(req_id)
                self.consumer_notification_counts_by_req.pop(req_id, None)
                success_pending.discard(req_id)

            failed_queue = getattr(self, "_stream_failed_sending", None)
            failed_pending = getattr(self, "_stream_failed_pending", set())
            while failed_queue is not None:
                try:
                    req_id = failed_queue.get_nowait()
                except queue.Empty:
                    break
                failed_pending.add(req_id)
            for req_id in tuple(failed_pending):
                # The failed stream is queued only after the scheduler emits
                # its final/cancel marker and all DMA handles are released.
                # Keep it pending if failure raced the lease metadata.
                if req_id not in self._reqs_to_send:
                    continue
                done_sending.add(req_id)
                self._reqs_to_send.pop(req_id, None)
                self._reqs_to_process.discard(req_id)
                self.consumer_notification_counts_by_req.pop(req_id, None)
                failed_pending.discard(req_id)

        # ``_pop_done_transfers`` mutates ``_sending_transfers``; the
        # writer thread also appends to it, so guard the pop.
        with self._sending_transfers_lock:
            done_pushing, failed_pushing = self._pop_done_transfers(
                self._sending_transfers
            )
            # Remember failures until the final sibling WRITE completes.
            self._send_failures.update(failed_pushing)
            successful = {
                req_id
                for req_id in done_pushing - self._send_failures
                if req_id in self._reqs_to_send or req_id in self._reqs_to_process
            }
            self._send_failures.difference_update(done_pushing | done_sending)
        # Expired requests were already reported, even if their WRITEs finish later.
        done_pushing = successful
        for req_id in done_pushing:
            self._reqs_to_send.pop(req_id, None)
            self._reqs_to_process.discard(req_id)
            self.consumer_notification_counts_by_req.pop(req_id, None)
            done_sending.add(req_id)

        # Tell the writer to drop any state it still holds for any
        # request that just finished (push completed) or expired
        # (lease ran out without a D registration ever arriving).
        for req_id in done_sending:
            self._evict_finished_inbox.put(req_id)
        if done_sending:
            self._push_writer_wake.set()

        return results
