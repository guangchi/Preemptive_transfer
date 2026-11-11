"""P-only admission windows. KV transfer and running requests stay unchanged."""
from __future__ import annotations

import json
import math
import os
import time
from collections import deque


class LocalWindowBuffer:
    """Fixed windows anchored to the first arrival; no sliding starvation."""

    def __init__(self, window_ms: float):
        if not math.isfinite(window_ms) or window_ms <= 0:
            raise ValueError("PD_LOCAL_SORT_WINDOW_MS must be finite and > 0")
        self.duration = window_ms / 1000
        self.windows = deque()
        self.next_id = 0

    def add(self, request, now):
        if not self.windows or now >= self.windows[-1][1]:
            self.windows.append((self.next_id, now + self.duration, []))
            self.next_id += 1
        self.windows[-1][2].append((now, request))

    def __len__(self):
        return sum(len(entries) for _, _, entries in self.windows)

    def discard(self, request_ids):
        self.windows = deque(
            (wid, deadline, [(t, r) for t, r in entries
                             if r.request_id not in request_ids])
            for wid, deadline, entries in self.windows
        )

    def release(self, now):
        released = []
        while self.windows and self.windows[0][1] <= now:
            wid, deadline, entries = self.windows.popleft()
            # Python sort is stable: equal lengths retain P admission order.
            entries.sort(key=lambda entry: (entry[1].num_prompt_tokens, entry[0]))
            released.append((wid, deadline, entries))
        return released


class LocalSortMixin:
    def __init__(self, *args, **kwargs):
        self.local_windows = LocalWindowBuffer(
            float(os.environ.get("PD_LOCAL_SORT_WINDOW_MS", "20")))
        self._hold_new_request = False
        super().__init__(*args, **kwargs)
        if self.policy.value != "fcfs":
            raise ValueError("Local sorting requires FCFS scheduling")

    def add_request(self, request):
        # Register through vLLM so aborts, connector state and stats remain valid.
        self._hold_new_request = request.request_id not in self.requests
        try:
            super().add_request(request)
        finally:
            self._hold_new_request = False

    def _enqueue_waiting_request(self, request):
        if self._hold_new_request:
            self.local_windows.add(request, time.monotonic())
        else:
            super()._enqueue_waiting_request(request)

    def get_request_counts(self):
        running, waiting = super().get_request_counts()
        # Keep engine polling even when the only work is an unexpired window.
        return running, waiting + len(self.local_windows)

    def finish_requests(self, request_ids, finished_status):
        if request_ids is not None and not isinstance(request_ids, str):
            request_ids = tuple(request_ids)
        finished = super().finish_requests(request_ids, finished_status)
        self.local_windows.discard({r.request_id for r in finished})
        return finished

    def schedule(self, *args, **kwargs):
        now = time.monotonic()
        for wid, deadline, entries in self.local_windows.release(now):
            for rank, (arrived, request) in enumerate(entries):
                super()._enqueue_waiting_request(request)
                print("PD local admission " + json.dumps({
                    "window": wid, "rank": rank,
                    "request_id": request.request_id,
                    "prompt_tokens": request.num_prompt_tokens,
                    "arrived_monotonic": arrived,
                    "deadline_monotonic": deadline,
                    "released_monotonic": now,
                    "wait_ms": (now - arrived) * 1000,
                }, separators=(",", ":")), flush=True)
        return super().schedule(*args, **kwargs)


# Preserve vLLM's async pipeline; synchronous configurations use the other class.
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler


class LocalSortAsyncScheduler(LocalSortMixin, AsyncScheduler):
    pass


class LocalSortScheduler(LocalSortMixin, Scheduler):
    pass
