"""Optional PD forward timing; CUDA event waits run off the execution thread."""
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
_resolver = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pd-cuda-timing")
_profile = None
_profile_done = False
_profile_completed = set()


def _start_profile(role):
    global _profile
    if (role == "kv_producer" and os.environ.get("PD_PROFILE_OUTPUT")
            and _profile is None and not _profile_done):
        _profile = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
        )
        _profile.start()


def _finish_profile(scheduler_output):
    """Capture until all prompt forwards finish; transfer tail is excluded."""
    global _profile, _profile_done
    if _profile is None:
        return
    _profile_completed.update(
        c["request_id"] for c in getattr(scheduler_output.kv_connector_metadata, "push_chunks", [])
        if c.get("is_final")
    )
    if len(_profile_completed) >= int(os.environ.get("PD_PROFILE_REQUESTS", "15")):
        from pathlib import Path
        _profile.stop()
        output = Path(os.environ["PD_PROFILE_OUTPUT"])
        output.parent.mkdir(parents=True, exist_ok=True)
        _profile.export_chrome_trace(str(output))
        logger.info("PD CUDA profile saved path=%s completed_requests=%d", output,
                    len(_profile_completed))
        _profile = None
        _profile_done = True


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
        (rid, n) for rid, n in scheduler_output.num_scheduled_tokens.items()
        if n > 0 and not rid.startswith("_warmup_")
    ]
    if (os.environ.get("PD_MEASURE_EXECUTION") != "1"
            or role not in ("kv_producer", "kv_consumer") or not requests):
        yield
        return
    _start_profile(role)
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
    _finish_profile(scheduler_output)
