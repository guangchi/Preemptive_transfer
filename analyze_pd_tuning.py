#!/usr/bin/env python3
"""Rank preemption settings against a matched baseline and explicit guardrails."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from collect_pd_total_time import workload_totals


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--baseline-dir", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--max-long-e2e-regression", type=float, default=0.10)
    args = parser.parse_args()
    root, baseline_dir = Path(args.root), Path(args.baseline_dir)
    baseline = {r["request_id"]: r for r in rows(baseline_dir / "trace-request-metrics.csv")}
    expected = {r["request_id"] for r in rows(baseline_dir / "requests.csv")}
    report = []
    for directory in sorted(root.glob("c*-p*-x*-w*")):
        if not directory.is_dir() or not (directory / "comparison.json").exists():
            continue
        comparison = json.loads((directory / "comparison.json").read_text())
        run_dirs = list(directory.glob("preemptive-c*-p*-x*-w*"))
        if len(run_dirs) != 1:
            continue
        run_dir = run_dirs[0]
        candidate = {r["request_id"]: r for r in rows(run_dir / "trace-request-metrics.csv")}
        requests = {r["request_id"]: r for r in rows(run_dir / "requests.csv")}
        totals = workload_totals(list(requests.values()))
        chunks = rows(run_dir / "trace-chunks.csv")
        status = json.loads((run_dir / "status.json").read_text())
        timing = json.loads((run_dir / "experiment-timing.json").read_text())
        by_category = {}
        long_ratios = []
        for category in ("short", "medium", "long"):
            ids = [rid for rid in expected if requests.get(rid, {}).get("category") == category]
            fields = {name: statistics.mean(float(candidate[rid][name]) for rid in ids)
                      for name in ("ttft_ms", "e2e_ms", "prefill_execution_ms", "decode_execution_ms")}
            by_category[category] = {"n": len(ids), **{k: round(v, 3) for k, v in fields.items()}}
            if category == "long":
                for rid in ids:
                    long_ratios.append(float(candidate[rid]["e2e_ms"]) / float(baseline[rid]["e2e_ms"]))
        paired = comparison.get("all_outputs_match", False)
        reference = comparison.get("all_direct_references_match")
        if reference is None:
            reference_ok = True
        else:
            reference_ok = reference
        max_long_ratio = max(long_ratios) if long_ratios else float("inf")
        short_medium_score = (by_category["short"]["ttft_ms"] * by_category["medium"]["ttft_ms"]) ** 0.5
        valid = (status["status"] == "passed" and status["ok"] == status["requests"] == len(expected)
                 and set(candidate) == expected and set(requests) == expected
                 and paired and reference_ok
                 and max_long_ratio <= 1 + args.max_long_e2e_regression)
        decisions = {}
        for chunk in chunks:
            decisions[chunk["decision"]] = decisions.get(chunk["decision"], 0) + 1
        order_signature = [(c['request_id'], int(c['sequence'])) for c in sorted(chunks, key=lambda c: float(c['dispatch_monotonic_s']))]
        order_hash = hashlib.sha256(json.dumps(order_signature).encode()).hexdigest()
        record = {"tag": directory.name, "status": status["status"], "requests": status["requests"],
                  "all_outputs_match": paired, "direct_reference_match": reference,
                  "max_per_request_long_e2e_ratio_vs_baseline": round(max_long_ratio, 4),
                  "short_medium_geomean_ttft_ms": round(short_medium_score, 3),
                  "categories": by_category, "preemption_count_chunks": decisions.get("preempt_long", 0),
                  "protected_resume_chunk_records": decisions.get("protected_resume", 0),
                  "group_wall_s": round(timing["phase_durations_s"]["group_wall_s"], 3),
                  "request_replay_s": status.get("elapsed_s"), "eligible": valid,
                  "workload_makespan_s": totals["workload_makespan_s"],
                  "successful_requests_per_s": totals["successful_requests_per_s"],
                  "dispatch_order_sha256": order_hash,
                  "guardrail": f"max per-long E2E <= baseline + {args.max_long_e2e_regression:.0%}"}
        report.append(record)
    order_groups = {}
    for record in report:
        order_groups.setdefault(record["dispatch_order_sha256"], []).append(record["tag"])
    for record in report:
        record["same_observed_dispatch_order_as"] = order_groups[record["dispatch_order_sha256"]]
        objectives = lambda r: [r["categories"]["short"]["ttft_ms"], r["categories"]["medium"]["ttft_ms"], r["categories"]["long"]["e2e_ms"], r["workload_makespan_s"]]
        record["pareto_nondominated"] = not any(
            other["eligible"] and all(a <= b for a, b in zip(objectives(other), objectives(record)))
            and any(a < b for a, b in zip(objectives(other), objectives(record)))
            for other in report if other is not record)
    report.sort(key=lambda r: (not r["eligible"], r["short_medium_geomean_ttft_ms"], r["group_wall_s"]))
    for rank, record in enumerate(report, 1):
        record["rank"] = rank if record["eligible"] else None
    eligible = [r for r in report if r["eligible"]]
    result = {"objective": "minimize geometric mean of short and medium TTFT means",
              "guardrails": ["all 15 requests HTTP 200", "all token IDs/text hashes match baseline",
                             f"max per-request long E2E <= baseline + {args.max_long_e2e_regression:.0%}"],
              "baseline_dir": str(baseline_dir.resolve()), "candidate_count": len(report),
              "eligible_count": len(eligible), "recommended": eligible[0] if eligible else None,
              "fastest_makespan_eligible": min(eligible, key=lambda r: r["workload_makespan_s"]) if eligible else None,
              "shortest_group_wall_eligible": min(eligible, key=lambda r: r["group_wall_s"]) if eligible else None,
              "observed_dispatch_order_groups": list(order_groups.values()),
              "ranking": report,
              "warning": "Single run per grid point. Recommendation is provisional and requires repeated validation."}
    (root / "tuning-results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    fields = ["rank", "tag", "eligible", "short_medium_geomean_ttft_ms", "max_per_request_long_e2e_ratio_vs_baseline",
              "short_ttft_ms", "medium_ttft_ms", "long_ttft_ms", "short_e2e_ms", "medium_e2e_ms", "long_e2e_ms",
              "preemption_count_chunks", "protected_resume_chunk_records", "workload_makespan_s", "successful_requests_per_s", "pareto_nondominated", "group_wall_s", "all_outputs_match"]
    with (root / "tuning-results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for r in report:
            writer.writerow({"rank": r["rank"], "tag": r["tag"], "eligible": r["eligible"],
                             "short_medium_geomean_ttft_ms": r["short_medium_geomean_ttft_ms"],
                             "max_per_request_long_e2e_ratio_vs_baseline": r["max_per_request_long_e2e_ratio_vs_baseline"],
                             "short_ttft_ms": r["categories"]["short"]["ttft_ms"],
                             "medium_ttft_ms": r["categories"]["medium"]["ttft_ms"],
                             "long_ttft_ms": r["categories"]["long"]["ttft_ms"],
                             "short_e2e_ms": r["categories"]["short"]["e2e_ms"],
                             "medium_e2e_ms": r["categories"]["medium"]["e2e_ms"],
                             "long_e2e_ms": r["categories"]["long"]["e2e_ms"],
                             "preemption_count_chunks": r["preemption_count_chunks"],
                             "protected_resume_chunk_records": r["protected_resume_chunk_records"],
                             "workload_makespan_s": r["workload_makespan_s"],
                             "successful_requests_per_s": r["successful_requests_per_s"],
                             "pareto_nondominated": r["pareto_nondominated"],
                             "group_wall_s": r["group_wall_s"], "all_outputs_match": r["all_outputs_match"]})
    lines = ["# 15 请求长请求保护参数搜索", "", result["warning"], "",
             "搜索范围：chunk 固定 4；次数={1,3,5}；进度保护={0.70,0.85,0.95}；累计等待={25,50,100}ms。",
             "排序目标：short 与 medium 类 TTFT 均值的几何平均最小。硬性准入条件：15/15 成功、逐请求 token ID 与文本 hash 和 baseline 一致、每个 long 的 E2E 不超过其配对 baseline 10%。配置只有在满足全部条件时才进入排名。",
             "", "| rank | config | score (ms) | max long E2E ratio | short TTFT | medium TTFT | long TTFT | group wall (s) | preempt chunks |",
             "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in report:
        lines.append(f"| {r['rank'] if r['rank'] is not None else '-'} | {r['tag']} | {r['short_medium_geomean_ttft_ms']:.3f} | {r['max_per_request_long_e2e_ratio_vs_baseline']:.3f} | {r['categories']['short']['ttft_ms']:.3f} | {r['categories']['medium']['ttft_ms']:.3f} | {r['categories']['long']['ttft_ms']:.3f} | {r['group_wall_s']:.3f} | {r['preemption_count_chunks']} |")
    lines += ["", "## 整批请求总时间与吞吐", "", "| config | first-send to last-finish (s) | successful req/s | Pareto nondominated |", "| --- | ---: | ---: | --- |"]
    for r in report:
        lines.append(f"| {r['tag']} | {r['workload_makespan_s']:.6f} | {r['successful_requests_per_s']:.6f} | {r['pareto_nondominated']} |")
    lines += ["", "Pareto 同时考虑 short TTFT、medium TTFT、long E2E 和整批 makespan。group wall 包含加载、warmup 等实验成本，不作为模型推理吞吐的分母。", "",
              "相同 chunk dispatch 顺序的组合见 tuning-results.json 中 observed_dispatch_order_groups。三个保护条件按 OR 触发，若其他条件先满足，某参数在本固定负载上可能无可观察作用；同顺序组合的时长差异不能直接归因于参数。"]
    (root / "TUNING_REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"candidate_count={len(report)} eligible_count={len(eligible)} recommended={eligible[0]['tag'] if eligible else 'none'}")
    if not report or not eligible:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
