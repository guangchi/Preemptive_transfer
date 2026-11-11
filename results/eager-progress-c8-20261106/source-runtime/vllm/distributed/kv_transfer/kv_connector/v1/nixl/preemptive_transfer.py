# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration and small helpers for short-request preemptive NIXL push."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import deque
from typing import Any


@dataclass(frozen=True)
class PreemptiveTransferConfig:
    enabled: bool = False
    chunk_size_blocks: int = 4
    short_request_threshold_tokens: int = 256
    long_request_threshold_tokens: int = 8192
    long_request_max_preemptions: int = 3
    long_request_protect_progress: float = 0.85
    long_request_max_preempt_wait_ms: float = 50.0

    @classmethod
    def from_vllm_config(cls, vllm_config: Any) -> "PreemptiveTransferConfig":
        transfer = vllm_config.kv_transfer_config
        raw = transfer.kv_connector_extra_config if transfer is not None else {}
        config = cls(
            enabled=bool(raw.get("short_request_preemptive_transfer", False)),
            chunk_size_blocks=int(raw.get("chunk_size_blocks", 4)),
            short_request_threshold_tokens=int(
                raw.get("short_request_threshold_tokens", 256)
            ),
            long_request_threshold_tokens=int(
                raw.get("long_request_threshold_tokens", 8192)
            ),
            long_request_max_preemptions=int(
                raw.get("long_request_max_preemptions", 3)
            ),
            long_request_protect_progress=float(
                raw.get("long_request_protect_progress", 0.85)
            ),
            long_request_max_preempt_wait_ms=float(
                raw.get("long_request_max_preempt_wait_ms", 50.0)
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.chunk_size_blocks < 1:
            raise ValueError("chunk_size_blocks must be >= 1")
        if self.short_request_threshold_tokens < 0:
            raise ValueError("short_request_threshold_tokens must be >= 0")
        if self.long_request_threshold_tokens <= self.short_request_threshold_tokens:
            raise ValueError(
                "long_request_threshold_tokens must be greater than "
                "short_request_threshold_tokens"
            )
        if self.long_request_max_preemptions < 0:
            raise ValueError("long_request_max_preemptions must be >= 0")
        if not 0.0 <= self.long_request_protect_progress <= 1.0:
            raise ValueError("long_request_protect_progress must be in [0, 1]")
        if self.long_request_max_preempt_wait_ms < 0:
            raise ValueError("long_request_max_preempt_wait_ms must be >= 0")

    def classify(self, prompt_tokens: int) -> str:
        if prompt_tokens < self.short_request_threshold_tokens:
            return "short"
        if prompt_tokens > self.long_request_threshold_tokens:
            return "long"
        return "medium"

    def wire_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate_runtime_scope(self, vllm_config: Any, kv_cache_config: Any) -> None:
        """Fail early for execution modes outside the first implementation."""
        if not self.enabled:
            return
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError(
                "short_request_preemptive_transfer currently requires TP=1 and PP=1"
            )
        if len(kv_cache_config.transfer_groups) != 1:
            raise ValueError(
                "short_request_preemptive_transfer currently requires one KV transfer group"
            )
        if vllm_config.kv_transfer_config.kv_buffer_device == "cpu":
            raise ValueError(
                "short_request_preemptive_transfer currently requires direct device KV transfer"
            )


class ChunkPriorityArbiter:
    """FIFO chunk queue with the sole bypass rule: short may bypass long."""

    def __init__(self, config: PreemptiveTransferConfig):
        self.config = config
        self._queue: deque[dict[str, Any]] = deque()
        self._long_state: dict[str, dict[str, Any]] = {}
        # Retain a bounded, serialisable decision trace for experiment logs
        # and deterministic arbitration tests.  This is deliberately kept in
        # the writer thread's arbiter rather than inferred from HTTP order.
        self._trace: deque[dict[str, Any]] = deque(maxlen=4096)

    def trace(self) -> list[dict[str, Any]]:
        """Return a snapshot of arbitration decisions and progress."""
        return list(self._trace)

    @staticmethod
    def _progress(state: dict[str, Any]) -> float:
        return min(1.0, state["completed"] / max(1, state["total"]))

    def _protection_reasons(self, state: dict[str, Any]) -> list[str]:
        reasons: list[str] = []
        if state["preemptions"] >= self.config.long_request_max_preemptions:
            reasons.append("max_preemptions")
        if self._progress(state) >= self.config.long_request_protect_progress:
            reasons.append("progress")
        if state["wait_ms"] + 1e-9 >= self.config.long_request_max_preempt_wait_ms:
            reasons.append("wait")
        return reasons

    def _record(self, event: str, request_id: str, state: dict[str, Any], **extra: Any) -> None:
        self._trace.append(
            {
                "event": event,
                "request_id": request_id,
                "preemptions": state["preemptions"],
                "completed_blocks": state["completed"],
                "total_blocks": state["total"],
                "progress": self._progress(state),
                "wait_ms": state["wait_ms"],
                "protected": bool(state["protected"]),
                "protection_reasons": self._protection_reasons(state),
                **extra,
            }
        )

    def enqueue(self, task: dict[str, Any]) -> None:
        if task.get("cancel"):
            self.cancel(task["request_id"])
            return
        self._queue.append(task)
        if task.get("tier") == "long":
            state = self._long_state.setdefault(
                task["request_id"],
                {
                    "total": max(
                        1,
                        int(
                            task.get(
                                "progress_total_blocks",
                                task.get("local_total_blocks", 1),
                            )
                        ),
                    ),
                    "completed": 0,
                    "preemptions": 0,
                    "wait_started": None,
                    "wait_ms": 0.0,
                    "protected": False,
                },
            )
            state["total"] = max(
                state["total"],
                int(
                    task.get(
                        "progress_total_blocks",
                        task.get("local_total_blocks", 1),
                    )
                ),
            )

    def cancel(self, request_id: str) -> None:
        self._queue = deque(
            task for task in self._queue if task.get("request_id") != request_id
        )
        self._long_state.pop(request_id, None)

    def pending_request_ids(self) -> set[str]:
        """Return request IDs with chunks still queued for arbitration."""
        return {str(task["request_id"]) for task in self._queue}

    def __bool__(self) -> bool:
        return bool(self._queue)

    def has_waiting_short_before_medium(self) -> bool:
        for task in self._queue:
            tier = task.get("tier")
            if tier == "medium":
                return False
            if tier == "short":
                return True
        return False

    def may_preempt(self, request_id: str, now: float) -> bool:
        state = self._long_state.get(request_id)
        if state is None:
            return False
        self._close_wait_interval(state, now, request_id)
        return not self._is_protected(state)

    def note_preemption(self, request_id: str, now: float) -> None:
        state = self._long_state.get(request_id)
        if state is None:
            return
        state["preemptions"] += 1
        if state["wait_started"] is None:
            state["wait_started"] = now
        if self._is_protected(state):
            state["protected"] = True
        self._record("preempt_long", request_id, state)

    def pop_next(self, now: float) -> dict[str, Any] | None:
        if not self._queue:
            return None

        first = self._queue[0]
        if first.get("tier") == "long":
            state = self._long_state.get(first["request_id"])
            if state is not None:
                self._close_wait_interval(state, now, first["request_id"])
                if (
                    not self._is_protected(state)
                    and self.has_waiting_short_before_medium()
                ):
                    self.note_preemption(first["request_id"], now)
                    short_index = next(
                        i
                        for i, task in enumerate(self._queue)
                        if task.get("tier") == "short"
                    )
                    task = self._queue[short_index]
                    del self._queue[short_index]
                    task["arbiter_decision"] = "preempt_long"
                    task["arbiter_long_request_id"] = first["request_id"]
                    task["arbiter_long_state"] = dict(self._trace[-1])
                    return task
                if self._is_protected(state):
                    state["protected"] = True
                    self._record(
                        "protected_resume",
                        first["request_id"],
                        state,
                        selected_sequence=first.get("chunk_sequence"),
                    )
                    first["arbiter_decision"] = "protected_resume"
                    first["arbiter_long_state"] = dict(self._trace[-1])

        task = self._queue.popleft()
        if task.get("tier") == "long":
            state = self._long_state.get(task["request_id"])
            if state is not None:
                self._close_wait_interval(state, now, task["request_id"])
                if state["wait_started"] is not None:
                    # This chunk is the long request's resumption point.
                    state["wait_started"] = None
                self._record(
                    "dispatch_long",
                    task["request_id"],
                    state,
                    selected_sequence=task.get("chunk_sequence"),
                )
                if task.get("arbiter_decision") is None:
                    task["arbiter_decision"] = "dispatch_long"
                task["arbiter_long_state"] = dict(self._trace[-1])
        elif task.get("tier") == "short":
            task["arbiter_decision"] = task.get("arbiter_decision", "dispatch_short")
        return task

    def mark_chunk_complete(self, task: dict[str, Any], now: float) -> None:
        request_id = task.get("request_id")
        state = self._long_state.get(request_id)
        if task.get("tier") != "long" or state is None:
            if task.get("is_final"):
                self._long_state.pop(request_id, None)
            return
        state["completed"] += int(
            task.get(
                "progress_blocks",
                len(task.get("local_block_ids", [[]])[0]),
            )
        )
        progress = self._progress(state)
        if progress >= self.config.long_request_protect_progress:
            state["protected"] = True
        self._record(
            "complete_long",
            request_id,
            state,
            completed_sequence=task.get("chunk_sequence"),
        )
        if task.get("is_final"):
            self._long_state.pop(request_id, None)
        # A preemption is counted when the arbiter actually bypasses a long
        # chunk in ``pop_next``.  Do not count merely because a long chunk
        # completed while a short chunk was queued: that completion is a
        # normal chunk boundary and the next dispatch may still choose the
        # long request.

    def _close_wait_interval(
        self, state: dict[str, Any], now: float, request_id: str | None = None
    ) -> None:
        started = state["wait_started"]
        if started is not None:
            state["wait_ms"] += max(0.0, (now - started) * 1000.0)
            state["wait_started"] = now
            if state["wait_ms"] + 1e-9 >= self.config.long_request_max_preempt_wait_ms:
                newly_protected = not state["protected"]
                state["protected"] = True
                if newly_protected:
                    self._record("wait_protected", request_id or "unknown", state)

    def _is_protected(self, state: dict[str, Any]) -> bool:
        return bool(state["protected"]) or (
            state["preemptions"] >= self.config.long_request_max_preemptions
        )
