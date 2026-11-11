from vllm.distributed.kv_transfer.kv_connector.v1.nixl.preemptive_transfer import (
    ChunkPriorityArbiter,
    PreemptiveTransferConfig,
)


def chunk(request_id, tier, sequence, total=16, final=False):
    return {
        "request_id": request_id,
        "tier": tier,
        "chunk_sequence": sequence,
        "local_total_blocks": total,
        "local_block_ids": [[1, 2, 3, 4]],
        "is_final": final,
    }


def test_classification_boundaries_are_configurable():
    cfg = PreemptiveTransferConfig(
        short_request_threshold_tokens=32,
        long_request_threshold_tokens=128,
    )
    assert cfg.classify(31) == "short"
    assert cfg.classify(32) == "medium"
    assert cfg.classify(128) == "medium"
    assert cfg.classify(129) == "long"


def test_short_bypasses_long_but_medium_does_not():
    arbiter = ChunkPriorityArbiter(PreemptiveTransferConfig())
    arbiter.enqueue(chunk("long", "long", 0))
    arbiter.enqueue(chunk("medium", "medium", 0))
    arbiter.enqueue(chunk("short", "short", 0))
    # A medium request is a barrier: the short request behind it cannot bypass
    # the long request. FIFO therefore returns the long chunk first.
    assert arbiter.pop_next(0.0)["request_id"] == "long"

    arbiter = ChunkPriorityArbiter(PreemptiveTransferConfig())
    arbiter.enqueue(chunk("long", "long", 0))
    arbiter.enqueue(chunk("short", "short", 0))
    assert arbiter.pop_next(0.0)["request_id"] == "short"


def test_priority_ablation_keeps_fifo_without_changing_protection_parameters():
    config = PreemptiveTransferConfig(priority_preemption=False)
    arbiter = ChunkPriorityArbiter(config)
    arbiter.enqueue(chunk("long", "long", 0))
    arbiter.enqueue(chunk("short", "short", 0))
    assert arbiter.pop_next(0)["request_id"] == "long"
    assert not any(row["event"] == "preempt_long" for row in arbiter.trace())
    assert config.long_request_max_preemptions == 3
    assert config.long_request_protect_progress == .85
    assert config.long_request_max_preempt_wait_ms == 50


def test_long_request_protection_by_preemption_count_and_progress():
    cfg = PreemptiveTransferConfig(
        long_request_max_preemptions=1,
        long_request_protect_progress=0.85,
    )
    arbiter = ChunkPriorityArbiter(cfg)
    arbiter.enqueue(chunk("long", "long", 0, total=20))
    arbiter.enqueue(chunk("long", "long", 1, total=20))
    arbiter.enqueue(chunk("short", "short", 0))
    assert arbiter.pop_next(0.0)["request_id"] == "short"
    assert arbiter.pop_next(0.0)["request_id"] == "long"


    arbiter.enqueue(chunk("short2", "short", 1))
    # The configured maximum has been reached, so the second short request
    # cannot bypass the long request.
    assert arbiter.pop_next(1.0)["request_id"] == "long"

    arbiter = ChunkPriorityArbiter(
        PreemptiveTransferConfig(long_request_protect_progress=0.5)
    )
    arbiter.enqueue(chunk("long", "long", 0, total=4))
    arbiter.mark_chunk_complete(
        {**chunk("long", "long", 0, total=4), "local_block_ids": [[1, 2, 3]]},
        0.0,
    )
    arbiter.enqueue(chunk("long", "long", 1, total=4))
    arbiter.enqueue(chunk("short", "short", 0))
    assert arbiter.pop_next(0.0)["request_id"] == "long"


def test_failed_chunk_does_not_advance_long_progress():
    cfg = PreemptiveTransferConfig(long_request_protect_progress=0.5)
    arbiter = ChunkPriorityArbiter(cfg)
    task = chunk("long", "long", 0, total=4)
    arbiter.enqueue(task)
    # The worker must call mark_chunk_complete only for successful chunks;
    # this regression makes the intended accounting explicit.
    arbiter.mark_chunk_complete({**task, "local_block_ids": [[]]}, 0.0)
    arbiter.enqueue(chunk("long", "long", 1, total=4))
    arbiter.enqueue(chunk("short", "short", 0))
    assert arbiter.pop_next(0.0)["request_id"] == "short"


def test_arbiter_trace_records_wait_protection_reason():
    cfg = PreemptiveTransferConfig(
        long_request_max_preemptions=10,
        long_request_max_preempt_wait_ms=5.0,
    )
    arbiter = ChunkPriorityArbiter(cfg)
    arbiter.enqueue(chunk("long", "long", 0, total=100))
    arbiter.enqueue(chunk("short", "short", 0))
    assert arbiter.pop_next(0.0)["request_id"] == "short"
    assert arbiter.pop_next(0.010)["request_id"] == "long"
    trace = arbiter.trace()
    assert any(
        row["event"] == "protected_resume"
        and "wait" in row["protection_reasons"]
        and row["wait_ms"] >= 5.0
        for row in trace
    )


def test_accumulated_wait_protects_long_at_exact_50ms_boundary():
    cfg = PreemptiveTransferConfig(
        long_request_max_preemptions=100,
        long_request_protect_progress=1.0,
        long_request_max_preempt_wait_ms=50.0,
    )
    arbiter = ChunkPriorityArbiter(cfg)
    arbiter.enqueue(chunk("long", "long", 0, total=100))
    for sequence in range(1, 4):
        arbiter.enqueue(chunk("short", "short", sequence))

    # The first short starts the wait interval; the next two short chunks
    # arrive while the same long chunk remains at the head of the queue.
    assert arbiter.pop_next(0.0)["tier"] == "short"
    assert arbiter.pop_next(0.025)["tier"] == "short"
    assert arbiter.pop_next(0.049999)["tier"] == "short"

    # A further short is available at the next boundary. At 50 ms cumulative
    # bypass wait, long must resume before consuming it.
    arbiter.enqueue(chunk("short", "short", 4))
    resumed = arbiter.pop_next(0.050000)
    assert resumed["request_id"] == "long"
    trace = arbiter.trace()
    assert trace[-1]["event"] == "dispatch_long"
    assert trace[-1]["wait_ms"] + 1e-9 >= 50.0
    assert "wait" in trace[-1]["protection_reasons"]


def test_progress_protection_uses_stable_request_total_at_exact_boundary():
    cfg = PreemptiveTransferConfig(
        long_request_max_preemptions=10,
        long_request_protect_progress=0.85,
        long_request_max_preempt_wait_ms=10_000,
    )
    arbiter = ChunkPriorityArbiter(cfg)
    # 17/20 blocks is exactly 85%; the next short must not bypass long.
    arbiter.enqueue(chunk("long", "long", 0, total=20))
    arbiter.mark_chunk_complete(
        {**chunk("long", "long", 0, total=20), "local_block_ids": [list(range(17))]},
        0.0,
    )
    arbiter.enqueue(chunk("long", "long", 1, total=20))
    arbiter.enqueue(chunk("short", "short", 0))
    selected = arbiter.pop_next(0.001)
    assert selected["request_id"] == "long"
    assert selected["arbiter_long_state"]["progress"] == 0.85
    assert "progress" in selected["arbiter_long_state"]["protection_reasons"]


def test_progress_denominator_excludes_prefix_cached_blocks():
    cfg = PreemptiveTransferConfig(long_request_protect_progress=0.85)
    arbiter = ChunkPriorityArbiter(cfg)
    cached = {
        **chunk("long", "long", 0, total=5),
        "progress_total_blocks": 3,
        "progress_blocks": 0,
    }
    arbiter.enqueue(cached)
    arbiter.mark_chunk_complete(cached, 0.0)
    assert arbiter.trace()[-1]["progress"] == 0.0

    uncached = {
        **chunk("long", "long", 1, total=5),
        "progress_total_blocks": 3,
        "progress_blocks": 2,
    }
    arbiter.enqueue(uncached)
    arbiter.mark_chunk_complete(uncached, 0.1)
    assert abs(arbiter.trace()[-1]["progress"] - (2 / 3)) < 1e-9
