#!/usr/bin/env python3
"""Validate shortlisted configurations with fresh, interleaved baseline repeats."""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


def read_rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reference", default="results/determinism-investigation-20261106/responses.jsonl")
    parser.add_argument("--candidates", nargs="*", help="Grid tags, e.g. c4-p3-x0.85-w100")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    grid, out = Path(args.grid).resolve(), Path(args.output).resolve()
    out.mkdir(exist_ok=False, parents=True)
    rankings = json.loads((grid / "tuning-results.json").read_text())
    reference = {}
    reference_path = Path(args.reference)
    for line in reference_path.read_text().splitlines():
        row = json.loads(line)
        if row['batch_invariant']:
            reference.setdefault(row['request_id'], set()).add(tuple(json.loads(row['response_token_ids'])))
    if len(reference) != 15 or any(len(values) != 1 for values in reference.values()):
        raise ValueError('Direct reference does not provide 15 consistent outputs')
    if args.candidates:
        selected = list(args.candidates)
    else:
        # Validate both the latency objective and the total-time objective;
        # a throughput-oriented winner need not minimize short/medium TTFT.
        selected = []
        for key in ("recommended", "fastest_makespan_eligible"):
            candidate = rankings.get(key)
            if candidate and candidate["tag"] not in selected:
                selected.append(candidate["tag"])
        for candidate in rankings['ranking']:
            if len(selected) >= 2:
                break
            if candidate['eligible'] and candidate['tag'] not in selected:
                selected.append(candidate['tag'])
    default = "c4-p3-x0.85-w50"
    if default not in selected:
        selected.append(default)
    commands = []
    results = []
    errors = []
    env = os.environ.copy()
    workload = root / "workload.sweep-interleaved-8s3m4l.jsonl"
    env.update(WORKLOAD=str(workload), REQUESTS="15", CONCURRENCY="15", INTERARRIVAL_MS="100",
               WAIT_SECONDS="15", ENABLE_PREFIX_CACHING="false", STREAM_RESPONSES="true",
               PD_MEASURE_EXECUTION="1", VLLM_BATCH_INVARIANT="1", PD_CAPTURE_LOGPROBS="0")
    overall_start = time.perf_counter()
    settings = ["baseline", *selected]
    # A cyclic ordering distributes early/late run positions over repetitions.
    for repeat in range(args.repeats):
        order = settings[repeat % len(settings):] + settings[:repeat % len(settings)]
        for setting in order:
            dirname = out / f"repeat-{repeat+1}" / setting
            dirname.mkdir(parents=True)
            trial_env = env.copy()
            # Use the timed runner's function definition only; invoke exactly
            # one run, avoiding a second baseline or a parameter loop.
            runner = (root / "run_pd_sweep.sh").read_text().split("# Baseline keeps the ordinary NIXL push path")[0]
            runner = runner.replace('ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"', 'ROOT="' + str(root) + '"', 1)
            if setting == "baseline":
                mode, chunk, n, progress, wait = "baseline", "4", "0", "0.85", "50"
            else:
                chunk, n, progress, wait = [part[1:] for part in setting.split("-")]
                mode = "preemptive"
            runner += '\nrun_one "' + mode + '" "' + chunk + '" "' + n + '" "' + progress + '" "' + wait + '"\n'
            runner_path = dirname / "timed-one-run.sh"
            runner_path.write_text(runner)
            trial_env["RESULT_ROOT"] = str(dirname)
            trial_env.pop("SUMMARY_FILE", None); trial_env.pop("SUMMARY_JSON", None)
            stamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            print(f"{stamp} starting repeat={repeat+1} config={setting}", flush=True)
            start = time.perf_counter()
            with (dirname / "runner.log").open("w") as log:
                completed = subprocess.run(["bash", str(runner_path)], env=trial_env, stdout=log, stderr=subprocess.STDOUT)
            end = time.perf_counter()
            commands.append({"repeat": repeat+1, "setting": setting, "start_utc": stamp,
                             "wall_s": end-start, "exit_code": completed.returncode,
                             "runner": str(runner_path), "runner_sha256": hashlib.sha256(runner_path.read_bytes()).hexdigest()})
            dirs = list(dirname.glob(mode + "-c*-p*-x*-w*"))
            if completed.returncode or len(dirs) != 1 or not (dirs[0] / "requests.csv").exists():
                errors.append({"repeat": repeat+1, "setting": setting, "error": "runner_failed"})
                continue
            run = dirs[0]
            response = read_rows(run / "requests.csv")
            metrics = read_rows(run / "trace-request-metrics.csv")
            total = json.loads((run / "workload-total-times.json").read_text())["runs"][0]
            results.append({"repeat": repeat+1, "setting": setting, "run_dir": str(run),
                            "group_wall_s": json.loads((run / "experiment-timing.json").read_text())["phase_durations_s"]["group_wall_s"],
                            "workload_makespan_s": total["workload_makespan_s"],
                            "successful_requests_per_s": total["successful_requests_per_s"],
                            "responses": response, "metrics": metrics})
            print(f"finished repeat={repeat+1} config={setting} makespan={total['workload_makespan_s']:.3f}s", flush=True)
    summaries = []
    for setting in selected:
        paired = []
        for repeat in range(1, args.repeats+1):
            a = next((r for r in results if r["repeat"] == repeat and r["setting"] == "baseline"), None)
            b = next((r for r in results if r["repeat"] == repeat and r["setting"] == setting), None)
            if not a or not b:
                continue
            ra = {r["request_id"]: r for r in a["responses"]}; rb = {r["request_id"]: r for r in b["responses"]}
            ma = {r["request_id"]: r for r in a["metrics"]}; mb = {r["request_id"]: r for r in b["metrics"]}
            match = len(ra) == len(rb) == 15 and ra.keys() == rb.keys() and all(
                ra[rid]["status"] == rb[rid]["status"] == "200" and
                json.loads(ra[rid]["response_token_ids"]) == json.loads(rb[rid]["response_token_ids"]) and
                ra[rid]["response_sha256"] == rb[rid]["response_sha256"] for rid in ra)
            direct_match = set(ra) == set(rb) == set(reference) and all(
                tuple(json.loads(ra[rid]['response_token_ids'])) in reference[rid]
                and tuple(json.loads(rb[rid]['response_token_ids'])) in reference[rid]
                for rid in ra)
            cats = {}
            for cat in ["short", "medium", "long"]:
                ids = [rid for rid in ra if ra[rid]["category"] == cat]
                cats[cat] = {field: statistics.mean(float(mb[rid][field]) for rid in ids)
                             for field in ["ttft_ms", "e2e_ms"]}
            long_ratio = max(float(mb[rid]["e2e_ms"]) / float(ma[rid]["e2e_ms"]) for rid in ma if ma[rid]["category"] == "long")
            paired.append({"repeat": repeat, "outputs_match": match, "direct_reference_match": direct_match, "categories": cats,
                           "max_long_e2e_ratio": long_ratio,
                           "short_medium_score_ms": (cats['short']['ttft_ms']*cats['medium']['ttft_ms'])**0.5,
                           "workload_makespan_s": b["workload_makespan_s"], "baseline_makespan_s": a["workload_makespan_s"],
                           "group_wall_s": b["group_wall_s"]})
        valid = len(paired) == args.repeats and all(r['outputs_match'] and r['direct_reference_match'] for r in paired) and max(r['max_long_e2e_ratio'] for r in paired) <= 1.10
        summary = {"setting": setting, "eligible": valid, "paired_runs": paired,
                   "median_score_ms": statistics.median(r['short_medium_score_ms'] for r in paired) if paired else None,
                   "mean_makespan_s": statistics.mean(r['workload_makespan_s'] for r in paired) if paired else None,
                   "mean_group_wall_s": statistics.mean(r['group_wall_s'] for r in paired) if paired else None,
                   "categories": {cat: {field: statistics.mean(r['categories'][cat][field] for r in paired) for field in ['ttft_ms','e2e_ms']} for cat in ['short','medium','long']} if paired else {}}
        summaries.append(summary)
    summaries.sort(key=lambda r: (not r['eligible'], r['median_score_ms'] or float('inf')))
    summary = {"configurations": settings, "repeats": args.repeats, "commands": commands,
               "total_confirmation_wall_s": time.perf_counter()-overall_start,
               "ranking": summaries, "recommended": next((r['setting'] for r in summaries if r['eligible']), None),
               "errors": errors, "selection": "median short/medium geometric-mean TTFT under paired per-long E2E <= baseline +10% on every repeat; identical outputs required"}
    (out/'confirmation-results.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'raw-confirmation.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(summary,indent=2), flush=True)
    if errors or not summary['recommended']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
