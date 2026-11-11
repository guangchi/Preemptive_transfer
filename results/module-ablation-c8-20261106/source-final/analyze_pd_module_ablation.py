#!/usr/bin/env python3
"""Independently verify module switches and summarize fixed ablation traces."""
import argparse
import collections
import json
from pathlib import Path
import statistics

from analyze_pd_transfer_bottleneck import read_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = json.loads((args.root / "raw-results.json").read_text())
    assert len(raw) == 15, "Expected five settings, three repeats"
    runs = []
    early_long_offsets = collections.defaultdict(list)
    per_long = collections.defaultdict(list)
    for row in raw:
        run = Path(row["run_dir"])
        setting = row["setting"]
        env = json.loads((run / "environment.json").read_text())
        expected = dict(stream_by_batch=setting != "no_stream",
                        aggregate_short_medium=setting != "no_aggregation",
                        priority_preemption=setting != "fifo")
        assert env["module_switches"] == expected
        assert env["ucx_environment"]["UCX_TCP_TX_SEG_SIZE"] == "128k"
        assert env["ucx_environment"]["UCX_TCP_RX_SEG_SIZE"] == "128k"
        assert env["stream_max_in_flight"] == 1 and env["kv_source_fence"] == "event"
        assert not env["stream_eager_progress"] and env["batch_invariant"]
        assert env["max_preemptions"] == "3" and env["protect_progress"] == "0.85"
        assert env["wait_threshold_ms"] == "50" and env["chunk_size_blocks"] == "8"
        phases = read_csv(run / "trace-request-phases.csv")
        chunks = read_csv(run / "trace-chunks.csv")
        long_offsets = {}
        for phase in phases:
            if phase["category"] != "long":
                continue
            rid = phase["request_id"]
            per_long[(setting, rid)].append(float(phase["e2e_ms"]))
            if setting == "baseline":
                continue
            req_chunks = [chunk for chunk in chunks if chunk["request_id"] == rid]
            first = min(float(chunk["start_monotonic_s"]) for chunk in req_chunks)
            offset = (first - float(phase["prefill_finished_monotonic_s"])) * 1000
            long_offsets[rid] = offset
            early_long_offsets[setting].append(offset)
            if setting == "no_stream":
                assert offset >= -1, (setting, rid, offset)
        preemptions = [chunk for chunk in chunks if chunk["decision"] == "preempt_long"]
        if setting == "fifo":
            assert not preemptions
        for chunk in preemptions:
            assert float(chunk["long_progress"]) < .85
            assert int(chunk["long_preemptions"]) <= 3
            assert float(chunk["long_wait_ms"]) < 50
        summary = {
            "repeat": row["repeat"], "setting": setting,
            "chunks_by_tier": dict(collections.Counter(chunk["tier"] for chunk in chunks)),
            "preemptions": [{key: chunk[key] for key in (
                "request_id", "long_request_id", "long_progress", "long_preemptions", "long_wait_ms")}
                for chunk in preemptions],
            "max_reported_long_wait_ms": max((float(chunk["long_wait_ms"]) for chunk in chunks
                if chunk["long_wait_ms"] not in ("", "None")), default=0),
            "long_first_write_relative_to_p_finish_ms": long_offsets,
            "requests": row["audit"]["requests"],
            "correctness_passed": not row["audit"]["chunk_violations"] and not row["audit"]["output_violations"],
        }
        assert summary["correctness_passed"]
        runs.append(summary)
    means = {setting: {rid: statistics.mean(values)
                       for (mode, rid), values in per_long.items() if mode == setting}
             for setting in ("baseline", "full", "no_stream", "no_aggregation", "fifo")}
    result = {
        "runs": runs,
        "audited_requests": sum(row["requests"] for row in runs),
        "all_output_and_coverage_checks_passed": True,
        "long_e2e_three_repeat_means_ms": means,
        "preemption_long_e2e_cost_full_minus_fifo_ms": {
            rid: value - means["fifo"][rid] for rid, value in means["full"].items()},
        "long_first_write_relative_to_p_finish_ms": {
            setting: {"min": min(values), "max": max(values), "mean": statistics.mean(values)}
            for setting, values in early_long_offsets.items()},
        "timing_note": "First WRITE is a host trace boundary, not a device DMA timestamp. Full-minus-FIFO is a whole-schedule effect, not just the arbiter's recorded inserted wait.",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Verified {len(runs)} runs and {result['audited_requests']} outputs")


if __name__ == "__main__":
    main()
