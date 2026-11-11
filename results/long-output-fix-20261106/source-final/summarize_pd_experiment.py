#!/usr/bin/env python3
"""Validate outputs and record metrics, arbiter decisions, and event timelines."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

from collect_pd_total_time import workload_totals


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--workload", required=True)
    parser.add_argument("--reference", help="Direct, no-transfer responses.jsonl")
    parser.add_argument("--baseline-dir", help="Baseline run directory outside --root")
    args = parser.parse_args()
    root, workload = Path(args.root), Path(args.workload)
    inputs = [json.loads(line) for line in workload.read_text().splitlines() if line]
    ids = [r["request_id"] for r in inputs]
    references = {}
    if args.reference:
        for line in Path(args.reference).read_text().splitlines():
            row = json.loads(line)
            if row["batch_invariant"]:
                references.setdefault(row["request_id"], []).append(json.loads(row["response_token_ids"]))
    metric_fields = ["ttft_ms", "prefill_execution_ms", "decode_execution_ms", "e2e_ms",
                     "generation_after_first_token_ms", "decode_service_ms"]
    summary = {"workload": str(workload.resolve()),
               "workload_sha256": hashlib.sha256(workload.read_bytes()).hexdigest(),
               "requests": len(ids), "order": [r["category"] for r in inputs],
               "composition": {c: sum(r["category"] == c for r in inputs)
                               for c in ("short", "medium", "long")}, "modes": {}}
    responses, comparison, events, preemptions = {}, [], [], []
    significant = []
    chunks_by_mode = {}
    for mode in ("baseline", "preemptive"):
        explicit = args.baseline_dir if mode == "baseline" and args.baseline_dir else None
        dirs = [Path(explicit)] if explicit else list(root.glob(mode + "-c*-p*-x*-w*"))
        if len(dirs) != 1:
            raise ValueError(f"Expected one {mode} directory")
        directory = dirs[0]
        metadata = json.loads((directory / "run.json").read_text())
        status = json.loads((directory / "status.json").read_text())
        origin = metadata["start_monotonic_s"]
        reqs = read_csv(directory / "requests.csv")
        metrics = {r["request_id"]: r for r in read_csv(directory / "trace-request-metrics.csv")}
        responses[mode] = {r["request_id"]: r for r in reqs}
        if set(responses[mode]) != set(ids) or set(metrics) != set(ids):
            raise ValueError("Missing or unexpected request IDs")
        mode_summary = {"status": status["status"], "ok": status["ok"],
                        "elapsed_s": metadata["elapsed_s"], "batch_invariant": metadata.get("batch_invariant", False),
                        "capture_logprobs": metadata.get("capture_logprobs", False),
                        "concurrency": metadata["concurrency"], "interarrival_ms": metadata["interarrival_ms"],
                        "prefix_caching": metadata["prefix_caching"],
                        "workload_checksum_match": metadata["workload_sha256"] == summary["workload_sha256"],
                        "submission_order_match": [r["request_id"] for r in sorted(reqs, key=lambda r: float(r["submitted_at_monotonic_s"]))] == ids,
                        "categories": {}}
        mode_summary["workload_totals"] = workload_totals(reqs)
        mode_summary["transfer_parameters"] = {key: metadata.get(key) for key in
            ("chunk_size_blocks", "max_preemptions", "protect_progress", "wait_threshold_ms")}
        for category in ("short", "medium", "long"):
            group = [r for r in metrics.values() if r["category"] == category]
            stats = {"n": len(group)}
            for field in metric_fields:
                values = [float(r[field]) for r in group]
                stats[field + "_mean"] = round(statistics.mean(values), 3)
            mode_summary["categories"][category] = stats
        summary["modes"][mode] = mode_summary

        def add_event(timestamp, event, req_id, **details):
            if timestamp in ("", None):
                return
            events.append({"mode": mode, "relative_ms": round((float(timestamp) - origin) * 1000, 3),
                           "monotonic_s": float(timestamp), "event": event, "request_id": req_id,
                           "details": json.dumps(details, ensure_ascii=False)})

        for req in reqs:
            req_id = req["request_id"]
            comparison.append({"mode": mode, "category": req["category"], "request_id": req_id,
                               **{field: metrics[req_id][field] for field in metric_fields},
                               "response_sha256": req["response_sha256"],
                               "response_token_ids": req["response_token_ids"], "response_text": req["response_text"]})
            for field, name in (("submitted_at_monotonic_s", "submitted"),
                                ("request_started_monotonic_s", "client_send"),
                                ("router_received_monotonic_s", "router_received"),
                                ("first_token_monotonic_s", "client_first_token"),
                                ("request_finished_monotonic_s", "client_finished")):
                add_event(req[field], name, req_id, category=req["category"])
        for phase in read_csv(directory / "trace-request-phases.csv"):
            rid = phase["request_id"]
            for field, name in (("p_final_chunk_accepted_monotonic_s", "p_final_accepted"),
                                ("p_final_chunk_complete_monotonic_s", "p_final_completed"),
                                ("d_kv_receive_complete_monotonic_s", "d_kv_complete"),
                                ("d_first_scheduled_monotonic_s", "d_first_scheduled"),
                                ("d_first_model_submit_monotonic_s", "d_first_model_submit")):
                add_event(phase[field], name, rid)
            significant.append({"mode": mode, "request_id": rid, "category": phase["category"],
                                **{field: round((float(phase[field]) - origin) * 1000, 3) if phase[field] else ""
                                   for field in ("request_started_monotonic_s", "prefill_first_submit_monotonic_s",
                                                 "d_kv_receive_complete_monotonic_s", "d_first_model_submit_monotonic_s",
                                                 "first_token_monotonic_s", "request_finished_monotonic_s")}})
        for step in read_csv(directory / "trace-execution.csv"):
            details = {k: step[k] for k in ("side", "step", "scheduled_tokens", "batch_requests", "elapsed_ms", "status")}
            if step["event"] == "host_step":
                add_event(step["submitted_monotonic_s"], "host_batch_submit", step["request_id"], **details)
            add_event(step["monotonic_s"], "host_" + step["event"], step["request_id"], **details)
        chunks = read_csv(directory / "trace-chunks.csv")
        chunks_by_mode[mode] = chunks
        decisions = {}
        for chunk in chunks:
            decision = chunk["decision"]
            decisions[decision] = decisions.get(decision, 0) + 1
            details = {k: chunk[k] for k in ("sequence", "tier", "decision", "blocks", "final", "success",
                                             "long_request_id", "long_preemptions", "long_progress", "long_wait_ms", "protection_reasons")}
            for field, name in (("accepted_monotonic_s", "chunk_accepted"), ("dispatch_monotonic_s", "arbiter_dispatch"),
                                ("start_monotonic_s", "write_start"), ("complete_monotonic_s", "write_complete"),
                                ("received_monotonic_s", "chunk_received")):
                add_event(chunk[field], name, chunk["request_id"], **details)
            if decision == "preempt_long":
                preemptions.append({"mode": mode, "relative_ms": round((float(chunk["dispatch_monotonic_s"]) - origin) * 1000, 3), **chunk})
        mode_summary.update(chunk_rows=len(chunks), arbiter_decisions=decisions)
    pair_checks = []
    for rid in ids:
        a, b = responses["baseline"][rid], responses["preemptive"][rid]
        at, bt = json.loads(a["response_token_ids"]), json.loads(b["response_token_ids"])
        pair_checks.append({"request_id": rid, "http_ok": a["status"] == b["status"] == "200",
                            "text_match": a["response_sha256"] == b["response_sha256"],
                            "tokens_present": bool(at) and bool(bt), "tokens_match": at == bt,
                            "direct_reference_match": all(at == ref == bt for ref in references[rid]) if rid in references else None})
    summary["output_checks"] = pair_checks
    summary["all_outputs_match"] = all(c["http_ok"] and c["text_match"] and c["tokens_present"] and c["tokens_match"] for c in pair_checks)
    summary["all_direct_references_match"] = all(c["direct_reference_match"] is True for c in pair_checks) if args.reference else None
    summary["passed"] = summary["all_outputs_match"] and all(m["status"] == "passed" and m["workload_checksum_match"] and m["submission_order_match"] for m in summary["modes"].values()) and (not args.reference or summary["all_direct_references_match"])
    summary["interpretation"] = "Single run per mode; shared host batch durations are not exclusive GPU compute times."
    (root / "comparison.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    write_csv(root / "comparison-request-metrics.csv", comparison)
    write_csv(root / "experiment-timeline.csv", sorted(events, key=lambda e: (e["mode"], e["monotonic_s"])))
    write_csv(root / "request-timeline.csv", significant)
    write_csv(root / "preemptions.csv", preemptions, ["mode", "relative_ms", *list(chunks_by_mode["preemptive"][0])] if chunks_by_mode["preemptive"] else ["mode", "relative_ms"])
    write_csv(root / "protection-decisions.csv", [c for c in chunks_by_mode["preemptive"] if c["decision"] == "protected_resume"], list(chunks_by_mode["preemptive"][0]) if chunks_by_mode["preemptive"] else [])
    invariant_settings = ", ".join(mode + "=" + str(m["batch_invariant"]) for mode, m in summary["modes"].items())
    params = summary['modes']['preemptive']['transfer_parameters']
    wait_threshold = params['wait_threshold_ms']
    lines = [f"# {len(ids)} 请求交错 PD 实验过程与结果", "", f"数据集：`{workload.name}`，SHA256 `{summary['workload_sha256']}`。",
             "发送顺序：" + " / ".join(summary["order"]) + "。", "",
             "每类 prompt 长度：" + "、".join(c + "=" + "/".join(str(n) for n in sorted({len(r['prompt']) for r in inputs if r['category'] == c})) for c in ('short', 'medium', 'long')) + " tokens。",
             f"batch-invariant 配置：{invariant_settings}；其余运行参数见 comparison.json。抢占参数 chunk={params['chunk_size_blocks']}、次数={params['max_preemptions']}、进度={float(params['protect_progress'])*100:g}%、累计等待={wait_threshold}ms。",
             "", f"输出一致性通过：{summary['all_outputs_match']}；不经传输的直接推理参考一致：{summary['all_direct_references_match']}；综合通过：{summary['passed']}。", "",
             "## 指标均值（ms）", "", "| 类别 | baseline TTFT | preemptive TTFT | baseline E2E | preemptive E2E |", "| --- | ---: | ---: | ---: | ---: |"]
    for cat in ("short", "medium", "long"):
        a = summary["modes"]["baseline"]["categories"][cat]
        b = summary["modes"]["preemptive"]["categories"][cat]
        lines.append(f"| {cat} | {a['ttft_ms_mean']:.3f} | {b['ttft_ms_mean']:.3f} | {a['e2e_ms_mean']:.3f} | {b['e2e_ms_mean']:.3f} |")
    lines += ["", "## 逐请求结果（ms）", "", "| 请求 | baseline TTFT | preemptive TTFT | baseline E2E | preemptive E2E |", "| --- | ---: | ---: | ---: | ---: |"]
    for rid in ids:
        a, b = responses['baseline'][rid], responses['preemptive'][rid]
        lines.append(f"| {rid} | {float(a['ttft_ms']):.3f} | {float(b['ttft_ms']):.3f} | {float(a['e2e_ms']):.3f} | {float(b['e2e_ms']):.3f} |")
    lines += ["", "整轮回放墙钟时间：" + " / ".join(mode + f" {m['elapsed_s']:.3f}s" for mode, m in summary['modes'].items()) + "。均值改善不代表每条请求改善，也不代表吞吐或整轮完成时间改善。", "",
              "整批请求总时间（首个 HTTP 发出→最后响应结束）：" + " / ".join(mode + f" {m['workload_totals']['workload_makespan_s']:.6f}s，{m['workload_totals']['successful_requests_per_s']:.6f} 成功请求/秒" for mode, m in summary['modes'].items()) + "。",
              "P/D host 执行累计见 comparison-request-metrics.csv，完整 token ID、文本和 top logprobs 见各组 requests.csv。"]
    lines += ["", "## 实际抢占选择", "", "下列每行是一次 arbiter 绕过队首 long、选中一个 short chunk 的决策。次数按 chunk 计数，同一 short 可以多次抢占；dispatch 表示选择，WRITE start/complete 表示实际传输，不把二者混同。时间相对 preemptive 回放起点。",
              "", "| t(ms) | short | chunk | 被越过的 long | 累计抢占次数 | long 进度 | 累计等待(ms) |", "| ---: | --- | ---: | --- | ---: | ---: | ---: |"]
    for c in preemptions:
        long_id = next((rid for rid in ids if c['long_request_id'].startswith('cmpl-' + rid + '-0')), c['long_request_id'])
        lines.append(f"| {c['relative_ms']:.3f} | {c['request_id']} | {c['sequence']} | {long_id} | {c['long_preemptions']} | {float(c['long_progress']):.4f} | {float(c['long_wait_ms']):.3f} |")
    lines += ["", "## 每个 long 首次触发保护的边界", "", "| long | 相对时间(ms) | chunk | 进度 | 累计等待(ms) | 原因 |", "| --- | ---: | ---: | ---: | ---: | --- |"]
    for rid in ids:
        protected = sorted([c for c in chunks_by_mode['preemptive'] if c['request_id'] == rid and c['decision'] == 'protected_resume'], key=lambda c: float(c['dispatch_monotonic_s']))
        if protected:
            c = protected[0]
            origin = json.loads((next(root.glob('preemptive-c*-p*-x*-w*')) / 'run.json').read_text())['start_monotonic_s']
            lines.append(f"| {rid} | {(float(c['dispatch_monotonic_s']) - origin) * 1000:.3f} | {c['sequence']} | {float(c['long_progress']):.4f} | {float(c['long_wait_ms']):.3f} | {c['protection_reasons']} |")
    lines += ["", f"{wait_threshold}ms 在 chunk 边界检查；等待量可以超过 {wait_threshold}ms，不能把它解释为整个请求的最大等待或 TTFT 上限。"]
    lines += ["", "所有保护恢复的序号、进度、累计等待与触发原因记录在 `protection-decisions.csv`。保护后每个 long chunk 都可能记录 protected_resume，行数不是独立恢复次数。medium 保持 FIFO 屏障，short 只越过屏障之前的 long；该规则并不保证所有 short 都先完成。",
              "", "## 完整过程", "", "1. 两种模式各启动一组独立 P/D 与 router，等服务健康并预留 15 秒，再按固定顺序每 100ms 提交一个请求。",
              "2. P 以动态 batch 计算 prompt，baseline 按整请求 push；preemptive 在 KV 就绪后分块入队，在 chunk 边界仲裁，已在途 WRITE 不可中断。",
              "3. D 收齐所需 KV 后首次调度、提交生成，并经 router 流式返回。每请求记录首 token、完整结束和 P/D host 执行累计。",
              "4. 请求结束后执行严格 trace 检查，清理该组服务，下一组重新启动。最后校验两模式输出 token ID、文本 hash 及直接推理参考。",
              "", "逐请求里程碑见 `request-timeline.csv`；包含每个 host batch、每个 chunk 接受/选择/WRITE start/complete/D receive 的完整时序见 `experiment-timeline.csv`。原始 monotonic 时间与各组相对时间均保留。baseline 未记录逐 chunk KV 时间，不虚构该阶段事件。",
              "", "本次每种模式只跑一轮，不能建立稳定分位数或普适性能结论。P/D host 执行含调度和结果等待，batch 共享时间不能相加当 GPU 独占时间。batch-invariant 改变了内核路径，新性能数据应在两组相同配置下比较，不直接与旧默认内核实验作因果对比。", ""]
    (root / "EXPERIMENT_REPORT.md").write_text("\n".join(lines))
    print(json.dumps({"passed": summary["passed"], "output_checks": pair_checks, "modes": summary["modes"]}, indent=2))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
