#!/usr/bin/env python3
"""Scan the fixed 15-request workload over 27 protection settings, serially."""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time

from analyze_pd_transfer_bottleneck import audit_run, read_csv


ROOT = Path(__file__).resolve().parent
WORKLOAD_SHA256 = "714990980c49534e539a125612ce1c8d36c68ac75a559e4bce93dabcb7ee9e14"


def schedule():
    grid = [(p, w) for p in range(50, 91, 5) for w in (50, 100, 150)]
    random.Random(20261106).shuffle(grid)
    jobs = []
    for i, (progress, wait) in enumerate(grid):
        if i % 9 == 0:
            jobs.append(dict(setting="baseline", progress_pct=85, wait_ms=50))
        jobs.append(dict(setting="full", progress_pct=progress, wait_ms=wait))
    jobs.append(dict(setting="baseline", progress_pct=85, wait_ms=50))
    return [dict(order=i + 1, **job) for i, job in enumerate(jobs)]


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def policy_trace(chunks):
    bypasses = [row for row in chunks if row["decision"] == "preempt_long"]
    fields = ("request_id", "long_request_id", "long_completed_blocks", "long_total_blocks", "long_progress")
    signature = [{key: row[key] for key in fields} for row in bypasses]
    return {
        "preemptions": len(bypasses),
        "bypasses": [{key: row[key] for key in fields + ("long_preemptions", "long_wait_ms")}
                     for row in bypasses],
        "bypass_signature": signature,
        "protection_reasons": dict(collections.Counter(row["protection_reasons"] for row in chunks)),
        "max_recorded_long_wait_ms": max((float(row["long_wait_ms"]) for row in chunks
                                           if row["long_wait_ms"] not in ("", "None")), default=0),
        "long_dispatches": [
            {key: row[key] for key in ("request_id", "sequence", "long_progress", "long_preemptions",
                                      "long_wait_ms", "protection_reasons")}
            for row in chunks if row["tier"] == "long"],
    }


def validate_environment(run, job):
    actual = json.loads((run / "environment.json").read_text())
    expected = dict(mode="baseline" if job["setting"] == "baseline" else "preemptive",
                    chunk_size_blocks="8", max_preemptions="3", wait_threshold_ms=str(job["wait_ms"]),
                    batch_invariant=True, prefix_caching=False, kv_source_fence="event",
                    stream_max_in_flight=1, stream_eager_progress=False)
    for key, value in expected.items():
        if actual[key] != value:
            raise AssertionError(f"{key}: {actual[key]!r} != {value!r}")
    if float(actual["protect_progress"]) != job["progress_pct"] / 100:
        raise AssertionError("Wrong protection progress")
    if not all(actual["module_switches"].values()) or len(actual["module_switches"]) != 3:
        raise AssertionError("Full scheme requires all three modules")
    if actual["ucx_environment"] != {"UCX_TCP_RX_SEG_SIZE": "128k", "UCX_TCP_TX_SEG_SIZE": "128k"}:
        raise AssertionError("UCX configuration differs")
    return actual


def summary(results):
    baselines = [r for r in results if r["setting"] == "baseline"]
    full = []
    for result in results:
        if result["setting"] != "full":
            continue
        before = max((b for b in baselines if b["order"] < result["order"]), key=lambda b: b["order"])
        after = next((b for b in baselines if b["order"] > result["order"]), None)
        controls = [before] + ([after] if after else [])
        baseline = sum(b["audit"]["makespan_s"] for b in controls) / len(controls)
        full.append({key: result[key] for key in ("order", "progress_pct", "wait_ms", "run_dir")} | {
            "makespan_s": result["audit"]["makespan_s"],
            "bracketing_baseline_s": baseline,
            "baseline_orders": [b["order"] for b in controls],
            "baseline_comparison_final": after is not None,
            "reduction_pct": 100 * (1 - result["audit"]["makespan_s"] / baseline),
            "categories": result["audit"]["categories"],
            "long_e2e_ms": result["long_e2e_ms"],
            "preemptions": result["policy"]["preemptions"],
            "max_recorded_long_wait_ms": result["policy"]["max_recorded_long_wait_ms"],
        })
    return {"completed": len(results), "candidate_count": len(full), "baseline_count": len(baselines),
            "objective": "Lowest observed workload makespan_s; one sample per combination",
            "baseline_comparison": "Mean of the fresh baseline immediately before and after each block of nine candidates",
            "ranked_candidates": sorted(full, key=lambda r: r["makespan_s"]),
            "baselines": [{"order": r["order"], "makespan_s": r["audit"]["makespan_s"],
                           "categories": r["audit"]["categories"]} for r in baselines]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    workload = ROOT / "workload.sweep-interleaved-8s3m4l.jsonl"
    if hashlib.sha256(workload.read_bytes()).hexdigest() != WORKLOAD_SHA256:
        raise ValueError("Fixed workload changed")
    reference = collections.defaultdict(set)
    for line in (ROOT / "results/determinism-investigation-20261106/responses.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["batch_invariant"]:
            reference[row["request_id"]].add((tuple(json.loads(row["response_token_ids"])), row["response_sha256"]))
    if len(reference) != 15 or any(len(values) != 1 for values in reference.values()):
        raise ValueError("Reference must contain 15 consistent token/hash pairs")
    env = os.environ.copy()
    env.update(WORKLOAD=str(workload), REQUESTS="15", CONCURRENCY="15", INTERARRIVAL_MS="100",
               WAIT_SECONDS="15", ENABLE_PREFIX_CACHING="false", STREAM_RESPONSES="true",
               PD_MEASURE_EXECUTION="1", VLLM_BATCH_INVARIANT="1", PD_CAPTURE_LOGPROBS="0",
               PD_TRACE_WRITE_PHASES="1", PD_KV_SOURCE_FENCE="event", PD_STREAM_MAX_IN_FLIGHT="1",
               PD_STREAM_EAGER_PROGRESS="0", UCX_TCP_TX_SEG_SIZE="128k", UCX_TCP_RX_SEG_SIZE="128k",
               PD_STREAM_BY_BATCH="1", PD_AGGREGATE_SHORT_MEDIUM="1", PD_PRIORITY_PREEMPTION="1",
               PYTHON_BIN="/home/ubuntu/miniconda3/envs/cc/bin/python",
               MODEL="/home/ubuntu/zjj/models/Llama-3.1-8B-Instruct", P_DEVICE="0", D_DEVICE="1",
               MAX_MODEL_LEN="32768", GPU_MEMORY_UTILIZATION="0.90", P_PORT="8100", D_PORT="8200",
               ROUTER_PORT="8000", P_SIDE_CHANNEL_PORT="5600", D_SIDE_CHANNEL_PORT="5601",
               PD_ENDPOINT="http://127.0.0.1:8000/v1/completions", START_LOCAL_ROUTER="1")
    template = (ROOT / "run_pd_sweep.sh").read_text().split("# Baseline keeps the ordinary NIXL push path")[0]
    template = template.replace('ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"', f'ROOT="{ROOT}"', 1)
    jobs = schedule()
    dump(output / "experiment-settings.json", {
        "schedule": jobs, "seed": 20261106, "candidate_repeats": 1,
        "workload_sha256": WORKLOAD_SHA256,
        "fixed": {"max_preemptions": 3, "long_chunk_blocks": 8, "short_medium_aggregation": "per request per compute batch"},
        "environment": {k: env[k] for k in ("WORKLOAD", "CONCURRENCY", "INTERARRIVAL_MS", "WAIT_SECONDS",
            "ENABLE_PREFIX_CACHING", "VLLM_BATCH_INVARIANT", "PD_KV_SOURCE_FENCE", "PD_STREAM_MAX_IN_FLIGHT",
            "PD_STREAM_EAGER_PROGRESS", "UCX_TCP_TX_SEG_SIZE", "UCX_TCP_RX_SEG_SIZE",
            "PD_STREAM_BY_BATCH", "PD_AGGREGATE_SHORT_MEDIUM", "PD_PRIORITY_PREEMPTION")},
        "objective": "Minimum workload makespan; category TTFT/E2E and each long E2E also reported",
        "progress_definition": "Successful transferred long blocks / total long blocks; at or above this threshold disallow preemption",
    })
    source = output / "source-before"
    source.mkdir()
    for name in (Path(__file__).name, "run_pd_sweep.sh", "run_pd_pair.sh", "analyze_pd_transfer_bottleneck.py",
                 workload.name, "PD_PREEMPTIVE_TRANSFER_PLAN.md", "PD_PREEMPTIVE_TRANSFER_SYSTEM_REPORT.md"):
        shutil.copy2(ROOT / name, source / name)
    runtime = output / "source-runtime"
    runtime.mkdir()
    packages = Path("/home/ubuntu/miniconda3/envs/cc/lib/python3.11/site-packages")
    hashes = []
    for line in (ROOT / "snapshots/vllm-preemptive-20261103/SHA256SUMS").read_text().splitlines():
        relative = line.split(None, 1)[1]
        target = runtime / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(packages / relative, target)
        hashes.append(f"{hashlib.sha256(target.read_bytes()).hexdigest()}  {relative}")
    (runtime / "SHA256SUMS").write_text("\n".join(hashes) + "\n")
    results = []
    for job in jobs:
        mode = "baseline" if job["setting"] == "baseline" else "preemptive"
        trial = output / f'{job["order"]:02d}-{job["setting"]}-x{job["progress_pct"]}-w{job["wait_ms"]}'
        trial.mkdir()
        runner = trial / "timed-one-run.sh"
        runner.write_text(template + f'\nrun_one "{mode}" "8" "3" "{job["progress_pct"] / 100:.2f}" "{job["wait_ms"]}"\n')
        run_env = dict(env, RESULT_ROOT=str(trial))
        for key in ("SUMMARY_FILE", "SUMMARY_JSON"):
            run_env.pop(key, None)
        print(f'{datetime.datetime.now(datetime.timezone.utc).isoformat()} starting {job}', flush=True)
        start = time.perf_counter()
        with (trial / "runner.log").open("w") as log:
            completed = subprocess.run(["bash", str(runner)], env=run_env, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"Runner failed: {trial}")
        run = next(trial.glob(mode + "-c*-p*-x*-w*"))
        actual = validate_environment(run, job)
        audit = audit_run(run, reference)
        if audit["chunk_violations"] or audit["output_violations"] or audit["successful_requests"] != 15:
            dump(trial / "failed-audit.json", audit)
            raise RuntimeError(f"Correctness audit failed: {trial}")
        chunks = read_csv(run / "trace-chunks.csv")
        if mode == "preemptive" and (set(r["request_id"] for r in chunks) != set(reference)
                                     or any(int(r["blocks"]) > 8 for r in chunks if r["tier"] == "long")):
            raise AssertionError("Long C8 coverage changed")
        requests = read_csv(run / "requests.csv")
        results.append(job | {"run_dir": str(run), "wall_s": time.perf_counter() - start,
                               "environment": actual, "audit": audit, "policy": policy_trace(chunks),
                               "long_e2e_ms": {r["request_id"]: float(r["e2e_ms"]) for r in requests if r["category"] == "long"}})
        dump(output / "raw-results.json", results)
        dump(output / "grid-results.json", summary(results))
        print(f'finished {job["order"]}/31 setting={job["setting"]} x={job["progress_pct"]} w={job["wait_ms"]} makespan={audit["makespan_s"]:.6f}s preemptions={results[-1]["policy"]["preemptions"]}', flush=True)
    print("COMPLETE: 27 candidates, 4 baseline controls, 465 responses audited", flush=True)


if __name__ == "__main__":
    main()
