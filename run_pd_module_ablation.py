#!/usr/bin/env python3
"""Fixed-workload, three-repeat leave-one-module-out P/D experiments."""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from analyze_pd_transfer_bottleneck import audit_run, read_csv


SETTINGS = {
    "baseline": ("baseline", True, True, True),
    "full": ("preemptive", True, True, True),
    "no_stream": ("preemptive", False, True, True),
    "no_aggregation": ("preemptive", True, False, True),
    "fifo": ("preemptive", True, True, False),
}


def summarize(results):
    groups = {}
    for setting in SETTINGS:
        rows = [row for row in results if row["setting"] == setting]
        if not rows:
            continue
        groups[setting] = {
            "repeats": len(rows),
            "makespan_s": statistics.mean(row["audit"]["makespan_s"] for row in rows),
            "makespan_range_s": [min(row["audit"]["makespan_s"] for row in rows),
                                 max(row["audit"]["makespan_s"] for row in rows)],
            "group_wall_s": statistics.mean(row["audit"]["group_wall_s"] for row in rows),
            "categories": {
                tier: {metric: statistics.mean(row["audit"]["categories"][tier][metric]["mean"]
                                               for row in rows)
                       for metric in ("ttft_ms", "e2e_ms")}
                for tier in ("short", "medium", "long")},
            "write_count": statistics.mean(row["audit"]["host_write_phases"]["all"]["submit_ms"]["n"] for row in rows),
            "preemptions": [row["preemption_count"] for row in rows],
        }
    contributions = {}
    for module, setting in (("early_streaming", "no_stream"),
                            ("short_medium_aggregation", "no_aggregation"),
                            ("priority_preemption", "fifo")):
        paired = []
        for full in [row for row in results if row["setting"] == "full"]:
            ablated = next((row for row in results if row["setting"] == setting
                            and row["repeat"] == full["repeat"]), None)
            if ablated is None:
                continue
            a, f = ablated["audit"], full["audit"]
            paired.append({
                "repeat": full["repeat"],
                "makespan_saved_s": a["makespan_s"] - f["makespan_s"],
                "makespan_reduction_pct": 100 * (1 - f["makespan_s"] / a["makespan_s"]),
                "categories_saved_ms": {
                    tier: {metric: a["categories"][tier][metric]["mean"]
                                   - f["categories"][tier][metric]["mean"]
                           for metric in ("ttft_ms", "e2e_ms")}
                    for tier in ("short", "medium", "long")},
                "per_long_e2e_delta_ms": {
                    rid: float(ablated["long_e2e_ms"][rid]) - float(value)
                    for rid, value in full["long_e2e_ms"].items()},
            })
        if paired:
            contributions[module] = {
                "without_setting": setting,
                "paired": paired,
                "mean_makespan_saved_s": statistics.mean(row["makespan_saved_s"] for row in paired),
                "categories_saved_ms": {
                    tier: {metric: statistics.mean(row["categories_saved_ms"][tier][metric] for row in paired)
                           for metric in ("ttft_ms", "e2e_ms")}
                    for tier in ("short", "medium", "long")},
            }
    return {"groups": groups, "contributions": contributions,
            "interpretation": "Positive saved time favors enabling the module. Effects are conditional on the other two modules being enabled, include interactions, and must not be added as independent shares."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    reference = collections.defaultdict(set)
    for line in (root / "results/determinism-investigation-20261106/responses.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["batch_invariant"]:
            reference[row["request_id"]].add((tuple(json.loads(row["response_token_ids"])), row["response_sha256"]))
    if len(reference) != 15 or any(len(values) != 1 for values in reference.values()):
        raise ValueError("Direct reference must contain 15 consistent token/hash pairs")
    workload = root / "workload.sweep-interleaved-8s3m4l.jsonl"
    env = os.environ.copy()
    env.update(WORKLOAD=str(workload), REQUESTS="15", CONCURRENCY="15", INTERARRIVAL_MS="100",
               WAIT_SECONDS="15", ENABLE_PREFIX_CACHING="false", STREAM_RESPONSES="true",
               PD_MEASURE_EXECUTION="1", VLLM_BATCH_INVARIANT="1", PD_CAPTURE_LOGPROBS="0",
               PD_TRACE_WRITE_PHASES="1", PD_KV_SOURCE_FENCE="event", PD_STREAM_MAX_IN_FLIGHT="1",
               PD_STREAM_EAGER_PROGRESS="0", UCX_TCP_TX_SEG_SIZE="128k", UCX_TCP_RX_SEG_SIZE="128k")
    template = (root / "run_pd_sweep.sh").read_text().split("# Baseline keeps the ordinary NIXL push path")[0]
    template = template.replace('ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"', f'ROOT="{root}"', 1)
    results = []
    settings = list(SETTINGS)
    experiment = {"settings": SETTINGS, "repeats": args.repeats,
                  "workload_sha256": hashlib.sha256(workload.read_bytes()).hexdigest(),
                  "controls": {key: env[key] for key in ("WAIT_SECONDS", "INTERARRIVAL_MS", "CONCURRENCY",
                      "PD_KV_SOURCE_FENCE", "PD_STREAM_MAX_IN_FLIGHT", "PD_STREAM_EAGER_PROGRESS",
                      "UCX_TCP_TX_SEG_SIZE", "UCX_TCP_RX_SEG_SIZE")},
                  "fixed_protection": {"progress": .85, "max_preemptions": 3, "wait_ms": 50},
                  "scope": "Per-request short/medium batch aggregation; no cross-request packing. Long C8 in every modular mode. No-stream retains chunk boundaries until prompt-final; FIFO keeps protection values but disables bypass."}
    (output / "experiment-settings.json").write_text(json.dumps(experiment, indent=2) + "\n")
    for repeat in range(1, args.repeats + 1):
        order = settings[repeat-1:] + settings[:repeat-1]
        for setting in order:
            mode, streaming, aggregation, priority = SETTINGS[setting]
            trial = output / f"repeat-{repeat}" / setting
            trial.mkdir(parents=True)
            runner = trial / "timed-one-run.sh"
            runner.write_text(template + f'\nrun_one "{mode}" "8" "3" "0.85" "50"\n')
            run_env = dict(env, RESULT_ROOT=str(trial), PD_STREAM_BY_BATCH=str(int(streaming)),
                           PD_AGGREGATE_SHORT_MEDIUM=str(int(aggregation)), PD_PRIORITY_PREEMPTION=str(int(priority)))
            for key in ("SUMMARY_FILE", "SUMMARY_JSON"):
                run_env.pop(key, None)
            print(f"{datetime.datetime.now(datetime.timezone.utc).isoformat()} starting repeat={repeat} setting={setting}", flush=True)
            start = time.perf_counter()
            with (trial / "runner.log").open("w") as log:
                completed = subprocess.run(["bash", str(runner)], env=run_env, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise RuntimeError(f"Runner failed: {trial}")
            run = next(trial.glob(mode + "-c*-p*-x*-w*"))
            audit = audit_run(run, reference)
            if audit["chunk_violations"] or audit["output_violations"]:
                (trial / "failed-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
                raise RuntimeError(f"Correctness audit failed: {trial}")
            chunks = read_csv(run / "trace-chunks.csv")
            preemptions = sum(row["decision"] == "preempt_long" for row in chunks)
            if not priority and preemptions:
                raise AssertionError("FIFO mode performed a bypass")
            if mode != "baseline":
                if any(int(row["blocks"]) > 8 for row in chunks if row["tier"] == "long"):
                    raise AssertionError("Long C8 changed")
                if not aggregation and any(int(row["blocks"]) > 8 for row in chunks):
                    raise AssertionError("Aggregation ablation emitted a large chunk")
            responses = read_csv(run / "requests.csv")
            results.append({"repeat": repeat, "setting": setting, "run_dir": str(run),
                            "runner_sha256": hashlib.sha256(runner.read_bytes()).hexdigest(),
                            "wall_s": time.perf_counter() - start, "audit": audit,
                            "preemption_count": preemptions,
                            "long_e2e_ms": {row["request_id"]: float(row["e2e_ms"]) for row in responses if row["category"] == "long"}})
            (output / "raw-results.json").write_text(json.dumps(results, indent=2) + "\n")
            (output / "ablation-results.json").write_text(json.dumps(summarize(results), indent=2) + "\n")
            print(f"finished repeat={repeat} setting={setting} makespan={audit['makespan_s']:.6f}s preemptions={preemptions}", flush=True)


if __name__ == "__main__":
    main()
