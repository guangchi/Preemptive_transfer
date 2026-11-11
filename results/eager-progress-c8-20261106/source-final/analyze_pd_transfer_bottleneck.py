#!/usr/bin/env python3
"""Audit KV coverage and separate host WRITE submission from chunk residence."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import statistics
from pathlib import Path


WRITE_RE = re.compile(
    r"push write phases req=(\S+) descriptors=(\d+).*?"
    r"desc_ms=([\d.]+) make_ms=([\d.]+) submit_ms=([\d.]+)"
)


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    return {"n": len(values), "sum": sum(values), "mean": statistics.mean(values),
            "median": statistics.median(values), "max": max(values)}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def audit_run(run, reference):
    chunks = read_csv(run / "trace-chunks.csv")
    by_request = collections.defaultdict(list)
    for row in chunks:
        by_request[row["request_id"]].append(row)
    violations = []
    for request_id, rows in by_request.items():
        rows.sort(key=lambda row: int(row["sequence"]))
        if [int(row["sequence"]) for row in rows] != list(range(len(rows))):
            violations.append(f"{request_id}: chunk sequence is not contiguous")
        cursor = 0
        for row in rows:
            if int(row["block_start"]) != cursor:
                violations.append(f"{request_id}: block gap/overlap at {cursor}")
            cursor += int(row["blocks"])
            if row["success"] != "True":
                violations.append(f"{request_id}: unsuccessful chunk")
        if cursor != int(rows[-1]["total_blocks"]) or rows[-1]["final"] != "True":
            violations.append(f"{request_id}: incomplete block coverage or no final")
    producer_log = (run / "prefill.log").read_text()
    phases = [match.groups() for match in WRITE_RE.finditer(producer_log)]
    timed_submissions = [
        (float(match[1]), float(match[2]))
        for match in re.finditer(
            r"push write phases[^\n]*submit_ms=([\d.]+) monotonic_s=([\d.]+)",
            producer_log,
        )
    ]
    # End of the previous transfer() to start of the next transfer(). This
    # includes completion checks, descriptors, logging, waits and source gaps;
    # it is not pure idle time, DMA time or network latency.
    submission_gaps = [
        (current[1] - previous[1]) * 1000 - current[0]
        for previous, current in zip(timed_submissions, timed_submissions[1:])
    ]
    device_fences = [float(value) for value in re.findall(
        r"stream source KV device-synchronized[^\n]*elapsed_ms=([\d.]+)", producer_log)]
    # A shared forward is logged once per request. Its observed log timestamp
    # differs across rows, so deduplicate using the actual forward boundaries.
    producer_forwards = {
        (row["forward_started_monotonic_s"], row["forward_returned_monotonic_s"]):
            float(row["gpu_forward_ms"])
        for row in read_csv(run / "trace-execution.csv")
        if row["side"] == "p" and row["event"] == "gpu_forward"
        and row.get("gpu_forward_ms")
    }
    metrics = read_csv(run / "trace-request-metrics.csv")
    responses = read_csv(run / "requests.csv")
    output_violations = []
    if set(r["request_id"] for r in responses) != set(reference):
        output_violations.append("request ID set differs from direct reference")
    for row in responses:
        actual = (tuple(json.loads(row["response_token_ids"])), row["response_sha256"])
        if row["status"] != "200" or actual not in reference.get(row["request_id"], set()):
            output_violations.append(row["request_id"])
    totals = json.loads((run / "workload-total-times.json").read_text())["runs"][0]
    output = {
        "run_dir": str(run), "makespan_s": totals["workload_makespan_s"],
        "requests": len(metrics), "successful_requests": sum(r["status"] == "200" for r in metrics),
        "chunk_count": len(chunks), "chunk_violations": violations,
        "group_wall_s": totals.get("group_wall_s"),
        "phase_durations_s": totals.get("phase_durations_s"),
        "source_fences": {
            "event_ready_batches": producer_log.count("stream source KV event-ready"),
            "device_sync_batches": len(device_fences),
            "device_sync_host_ms": distribution(device_fences),
        },
        "producer_gpu_forward_batches": len(producer_forwards),
        "producer_gpu_forward_total_ms": sum(producer_forwards.values()),
        "writer_submission_gaps_ms": distribution(submission_gaps),
        "writer_submission_gaps_below_10ms": distribution(
            value for value in submission_gaps if value < 10),
        "output_violations": output_violations,
        "chunks_by_tier": dict(collections.Counter(r["tier"] for r in chunks)),
        "long_chunk_block_sizes": dict(collections.Counter(int(r["blocks"]) for r in chunks if r["tier"] == "long")),
        "categories": {}, "host_write_phases": {},
    }
    for tier in ("short", "medium", "long"):
        rows = [r for r in metrics if r["category"] == tier]
        output["categories"][tier] = {
            field: distribution(float(r[field]) for r in rows if r.get(field))
            for field in ("ttft_ms", "e2e_ms", "prefill_execution_ms", "prefill_gpu_forward_ms")
        }
        rows = [r for r in chunks if r["tier"] == tier]
        output["categories"][tier]["chunk_queue_ms"] = distribution(float(r["queue_ms"]) for r in rows if r.get("queue_ms"))
    for tier in ("all", "long"):
        rows = [r for r in phases if tier == "all" or "-long-" in r[0]]
        output["host_write_phases"][tier] = {
            field: distribution(float(r[index]) for r in rows)
            for index, field in ((1, "descriptors"), (2, "desc_ms"), (3, "make_ms"), (4, "submit_ms"))
        }
    phase_rows = read_csv(run / "trace-request-phases.csv")
    output["long_kv_tail_ms"] = {
        r["request_id"]: (float(r["d_kv_receive_complete_monotonic_s"]) - float(r["prefill_finished_monotonic_s"])) * 1000
        for r in phase_rows if r["category"] == "long" and r.get("d_kv_receive_complete_monotonic_s") and r.get("prefill_finished_monotonic_s")
    }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", type=Path,
                        default=Path("results/determinism-investigation-20261106/responses.jsonl"))
    args = parser.parse_args()
    reference = collections.defaultdict(set)
    for line in args.reference.read_text().splitlines():
        row = json.loads(line)
        if row["batch_invariant"]:
            reference[row["request_id"]].add((tuple(json.loads(row["response_token_ids"])), row["response_sha256"]))
    runs = []
    for root in args.roots:
        for file in sorted(root.rglob("trace-chunks.csv")):
            if (file.parent / "workload-total-times.json").exists():
                runs.append(audit_run(file.parent, reference))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"runs": runs}, indent=2) + "\n")
    failures = sum(bool(r["chunk_violations"] or r["output_violations"]) for r in runs)
    print(f"Audited {len(runs)} runs; {failures} runs failed correctness audit")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
