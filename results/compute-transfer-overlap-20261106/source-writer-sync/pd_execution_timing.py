"""Optional PD forward timing; CUDA event waits run off the execution thread."""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
_resolver = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pd-cuda-timing")


def _report(start, end, requests, side, submitted, returned):
    try:
        end.synchronize()
        elapsed = start.elapsed_time(end)
        for request_id, tokens in requests:
            logger.info(
                "PD GPU forward req=%s side=%s scheduled_tokens=%s "
                "batch_requests=%s gpu_forward_ms=%.6f "
                "forward_started_monotonic_s=%.9f "
                "forward_returned_monotonic_s=%.9f monotonic_s=%.9f",
                request_id, side, tokens, len(requests), elapsed,
                submitted, returned, time.perf_counter(),
            )
    except Exception:
        logger.exception("PD GPU forward timing failed")


@contextmanager
def pd_forward_timing(scheduler_output, kv_config):
    """Measure shared-batch forward stream elapsed time, excluding sampling.

    Never divide batch time by token count to invent exclusive request costs.
    No synchronize is added to the runner's execution thread.
    """
    role = getattr(kv_config, "kv_role", None) or os.environ.get("PD_ROLE")
    role = getattr(role, "value", role)
    if role and "producer" in str(role):
        role = "kv_producer"
    elif role and "consumer" in str(role):
        role = "kv_consumer"
    requests = [
        (rid, n) for rid, n in scheduler_output.num_scheduled_tokens.items() if n > 0
    ]
    if (os.environ.get("PD_MEASURE_EXECUTION") != "1"
            or role not in ("kv_producer", "kv_consumer") or not requests):
        yield
        return
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    submitted = time.perf_counter()
    start.record()
    yield
    end.record()
    returned = time.perf_counter()
    _resolver.submit(
        _report, start, end, tuple(requests),
        "p" if role == "kv_producer" else "d", submitted, returned,
    )
