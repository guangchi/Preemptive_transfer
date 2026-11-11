#!/usr/bin/env python3
"""Measure workload makespan from client timestamps; retain setup timing separately."""
import argparse
import csv
import json
from pathlib import Path


def workload_totals(rows):
    if not rows:
        raise ValueError("Cannot measure an empty workload")
    start = min(float(r["request_started_monotonic_s"]) for r in rows)
    submitted = min(float(r["submitted_at_monotonic_s"]) for r in rows)
    finish = max(float(r["request_finished_monotonic_s"]) for r in rows)
    makespan = finish - start
    if makespan <= 0:
        raise ValueError("Request timestamps have invalid ordering")
    ok = sum(r["status"] == "200" for r in rows)
    return {
        "requests": len(rows), "successful_requests": ok,
        "all_successful": ok == len(rows),
        "first_client_send_monotonic_s": start,
        "last_client_finish_monotonic_s": finish,
        "workload_makespan_s": round(makespan, 6),
        "first_submission_to_last_finish_s": round(finish - submitted, 6),
        "successful_requests_per_s": round(ok / makespan, 6),
        "generated_tokens": sum(int(r.get("output_tokens") or 0) for r in rows),
        "note": "First client HTTP send to last response finish; includes interarrival and request queueing, excludes model startup/warmup/cleanup. Failed responses are retained in the end boundary.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.root)
    results = []
    for requests in sorted(root.rglob("requests.csv")):
        with requests.open(newline="") as stream:
            data = workload_totals(list(csv.DictReader(stream)))
        run_dir = requests.parent
        data["run_dir"] = str(run_dir.resolve())
        data["mode"] = "baseline" if run_dir.name.startswith("baseline") else "preemptive"
        timing = run_dir / "experiment-timing.json"
        if timing.exists():
            phases = json.loads(timing.read_text())["phase_durations_s"]
            data["group_wall_s"] = round(phases["group_wall_s"], 6)
            data["phase_durations_s"] = phases
        else:
            data["group_wall_s"] = None
            data["group_wall_note"] = "Not captured by historical runner; not reconstructed from file modification times."
        metadata = run_dir / "run.json"
        if metadata.exists():
            data["client_replay_elapsed_s"] = json.loads(metadata.read_text()).get("elapsed_s")
        results.append(data)
    output = Path(args.output) if args.output else root / "workload-total-times.json"
    output.write_text(json.dumps({"runs": results}, indent=2, ensure_ascii=False) + "\n")
    fields = ["run_dir", "mode", "requests", "successful_requests", "workload_makespan_s", "successful_requests_per_s", "client_replay_elapsed_s", "group_wall_s"]
    with output.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved workload totals for {len(results)} runs to {output}")


if __name__ == "__main__":
    main()
