#!/home/ubuntu/miniconda3/envs/cc/bin/python
"""Join client request results with P/D streaming worker trace logs."""
from __future__ import annotations

import argparse
import csv
import math
import re
import time
from collections import defaultdict
from pathlib import Path


_FIELD = re.compile(r"\b([a-z_]+)=([^ ]+)")
_REQUEST_HASH = re.compile(r"-\d+-[0-9a-f]{8}$")


def base_request_id(request_id: str) -> str:
    value = request_id.removeprefix("cmpl-")
    return _REQUEST_HASH.sub("", value)


def _optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def parse_worker_log(path: str | Path, side: str) -> tuple[dict, dict]:
    if side not in ("p", "d"):
        raise ValueError("side must be 'p' or 'd'")
    chunks: dict[tuple[str, int], dict] = {}
    phases: dict[str, dict] = defaultdict(dict)
    for line in Path(path).read_text(errors="replace").splitlines():
        if " monotonic_s=" not in line or " req=" not in line:
            continue
        fields = dict(_FIELD.findall(line))
        request_id = fields.get("req")
        monotonic = _optional_float(fields.get("monotonic_s"))
        if not request_id or monotonic is None:
            continue
        base_id = base_request_id(request_id)

        if side == "p" and "stream chunk accepted" in line:
            try:
                sequence = int(fields["seq"])
            except (KeyError, ValueError):
                continue
            row = chunks.setdefault((base_id, sequence), {"request_id": base_id, "sequence": sequence})
            row.update(
                {
                    "accepted_monotonic_s": monotonic,
                    "block_start": fields.get("start"),
                    "blocks": fields.get("blocks"),
                    "total_blocks": fields.get("total"),
                    "final": fields.get("final"),
                }
            )
            if fields.get("final") == "True":
                phases[base_id]["p_final_chunk_accepted_monotonic_s"] = monotonic
        elif side == "p" and "stream arbiter decision" in line:
            try:
                sequence = int(fields["seq"])
            except (KeyError, ValueError):
                continue
            row = chunks.setdefault((base_id, sequence), {"request_id": base_id, "sequence": sequence})
            row.update(
                {
                    "dispatch_monotonic_s": monotonic,
                    "tier": fields.get("tier"),
                    "decision": fields.get("decision"),
                    "long_request_id": fields.get("long_req"),
                    "long_preemptions": fields.get("long_preemptions"),
                    "long_completed_blocks": fields.get("long_completed_blocks"),
                    "long_total_blocks": fields.get("long_total_blocks"),
                    "long_progress": fields.get("long_progress"),
                    "long_wait_ms": fields.get("long_wait_ms"),
                    "protection_reasons": fields.get("protection_reasons"),
                }
            )
        elif side == "p" and "stream chunk start" in line:
            try:
                sequence = int(fields["seq"])
            except (KeyError, ValueError):
                continue
            row = chunks.setdefault((base_id, sequence), {"request_id": base_id, "sequence": sequence})
            row.update(
                {
                    "start_monotonic_s": monotonic,
                    "block_start": fields.get("block_start", row.get("block_start")),
                    "blocks": fields.get("blocks", row.get("blocks")),
                    "final": fields.get("final", row.get("final")),
                }
            )
        elif side == "p" and "stream chunk complete" in line:
            try:
                sequence = int(fields["seq"])
            except (KeyError, ValueError):
                continue
            row = chunks.setdefault((base_id, sequence), {"request_id": base_id, "sequence": sequence})
            row.update(
                {
                    "complete_monotonic_s": monotonic,
                    "success": fields.get("success"),
                    "final": fields.get("final", row.get("final")),
                }
            )
            if fields.get("final") == "True":
                phases[base_id]["p_final_chunk_complete_monotonic_s"] = monotonic
        elif side == "d" and "stream chunk received" in line:
            try:
                sequence = int(fields["seq"])
            except (KeyError, ValueError):
                continue
            row = chunks.setdefault((base_id, sequence), {"request_id": base_id, "sequence": sequence})
            row.update(
                {
                    "received_monotonic_s": monotonic,
                    "receive_final": fields.get("final"),
                }
            )
        elif side == "d" and "PD first scheduled" in line:
            phases[base_id]["d_first_scheduled_monotonic_s"] = monotonic
            phases[base_id]["d_first_scheduled_tokens"] = fields.get(
                "scheduled_tokens"
            )
        elif side == "d" and "PD first model submit" in line:
            phases[base_id]["d_first_model_submit_monotonic_s"] = monotonic
            phases[base_id]["d_first_model_submit_tokens"] = fields.get(
                "scheduled_tokens"
            )
        elif side == "d" and "stream receive complete" in line:
            phases[base_id]["d_kv_receive_complete_monotonic_s"] = monotonic
            phases[base_id]["d_final_sequence"] = fields.get("final_seq")
            phases[base_id]["d_completion_mode"] = (
                "summary" if "via summary" in line else "chunk_sequence"
            )
    return chunks, dict(phases)


def collect(
    requests_csv: str | Path,
    prefill_log: str | Path,
    decode_log: str | Path,
    output_prefix: str | Path,
    *,
    require_complete: bool = False,
    require_execution: bool = False,
) -> tuple[Path, Path]:
    p_chunks, p_phases = parse_worker_log(prefill_log, "p")
    d_chunks, d_phases = parse_worker_log(decode_log, "d")
    request_rows = list(csv.DictReader(Path(requests_csv).open(newline="", encoding="utf-8")))
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    execution_rows = []
    execution_by_request = defaultdict(dict)
    for side, path in (("p", prefill_log), ("d", decode_log)):
        for line in Path(path).read_text(errors="replace").splitlines():
            event = next((name for name, marker in (
                ("host_step", "PD execution step req="),
                ("gpu_forward", "PD GPU forward req="),
                ("finished", "PD request finished req="),
            ) if marker in line), None)
            if event is None:
                continue
            fields = dict(_FIELD.findall(line))
            rid = base_request_id(fields["req"])
            entry = {**fields, "request_id": rid, "side": side, "event": event}
            execution_rows.append(entry)
            execution_by_request[rid].setdefault(side, defaultdict(list))[event].append(entry)

    execution_fields = [
        "request_id", "side", "event", "step", "scheduled_tokens", "batch_requests",
        "submitted_monotonic_s", "monotonic_s", "elapsed_ms", "gpu_forward_ms",
        "forward_started_monotonic_s", "forward_returned_monotonic_s", "status", "output_tokens",
    ]
    execution_path = output_prefix.with_name(output_prefix.name + "-execution.csv")
    with execution_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=execution_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(execution_rows)

    execution_phase_fields = [
        "request_finished_monotonic_s", "e2e_ms", "arrival_to_finish_ms",
        "generation_after_first_token_ms",
    ]
    for stage in ("prefill", "decode"):
        execution_phase_fields.extend([
            f"{stage}_execution_ms", f"{stage}_gpu_forward_ms",
            f"{stage}_execution_steps", f"{stage}_gpu_forward_steps",
            f"{stage}_scheduled_tokens", f"{stage}_shared_batch_steps",
            f"{stage}_first_submit_monotonic_s", f"{stage}_last_result_monotonic_s",
            f"{stage}_execution_span_ms", f"{stage}_finished_monotonic_s",
            f"{stage}_finish_status",
        ])
    execution_phase_fields.append("decode_service_ms")
    phase_fields = [
        "request_id",
        "category",
        "prompt_tokens",
        "max_tokens",
        "status",
        "error",
        "request_started_monotonic_s",
        "router_received_monotonic_s",
        "queue_ms",
        "p_final_chunk_accepted_monotonic_s",
        "p_final_chunk_complete_monotonic_s",
        "d_kv_receive_complete_monotonic_s",
        "d_final_sequence",
        "d_completion_mode",
        "d_first_scheduled_monotonic_s",
        "d_first_scheduled_tokens",
        "d_first_model_submit_monotonic_s",
        "d_first_model_submit_tokens",
        "first_token_monotonic_s",
        "p_final_chunk_ready_to_d_kv_complete_ms",
        "d_kv_to_first_scheduled_ms",
        "d_schedule_to_model_submit_ms",
        "d_kv_to_first_model_submit_ms",
        "model_submit_to_first_token_ms",
        "d_kv_to_first_token_ms",
        "ttft_ms",
        "router_ttft_ms",
        "latency_ms",
        "prefill_ms",
        "decode_ms",
        "output_tokens",
        "response_sha256",
    ] + execution_phase_fields
    merged_rows = []
    for result in request_rows:
        request_id = result["request_id"]
        p = p_phases.get(request_id, {})
        d = d_phases.get(request_id, {})
        row = {**result, **p, **d}
        for side, stage in (("p", "prefill"), ("d", "decode")):
            events = execution_by_request.get(request_id, {}).get(side, {})
            host = events.get("host_step", [])
            gpu = events.get("gpu_forward", [])
            finish = events.get("finished", [])
            row[f"{stage}_execution_steps"] = len(host)
            row[f"{stage}_gpu_forward_steps"] = len(gpu)
            row[f"{stage}_execution_ms"] = (
                round(sum(float(e["elapsed_ms"]) for e in host), 6) if host else None
            )
            row[f"{stage}_gpu_forward_ms"] = (
                round(sum(float(e["gpu_forward_ms"]) for e in gpu), 6) if gpu else None
            )
            row[f"{stage}_scheduled_tokens"] = sum(int(e["scheduled_tokens"]) for e in host)
            row[f"{stage}_shared_batch_steps"] = sum(int(e["batch_requests"]) > 1 for e in host)
            first = min((float(e["submitted_monotonic_s"]) for e in host), default=None)
            last = max((float(e["monotonic_s"]) for e in host), default=None)
            row[f"{stage}_first_submit_monotonic_s"] = first
            row[f"{stage}_last_result_monotonic_s"] = last
            row[f"{stage}_execution_span_ms"] = (
                round((last - first) * 1000, 3) if first is not None else None
            )
            if finish:
                row[f"{stage}_finished_monotonic_s"] = float(finish[-1]["monotonic_s"])
                row[f"{stage}_finish_status"] = finish[-1]["status"]
        if row.get("decode_finished_monotonic_s") is not None and row.get("decode_first_submit_monotonic_s") is not None:
            row["decode_service_ms"] = round((row["decode_finished_monotonic_s"] - row["decode_first_submit_monotonic_s"]) * 1000, 3)
        start = _optional_float(result.get("request_started_monotonic_s"))
        ttft_ms = _optional_float(result.get("ttft_ms"))
        first_token = _optional_float(result.get("first_token_monotonic_s"))
        if first_token is None:
            first_token = start + ttft_ms / 1000.0 if start is not None and ttft_ms is not None else None
        row["first_token_monotonic_s"] = first_token
        p_ready = _optional_float(p.get("p_final_chunk_accepted_monotonic_s"))
        d_ready = _optional_float(d.get("d_kv_receive_complete_monotonic_s"))
        row["p_final_chunk_ready_to_d_kv_complete_ms"] = (
            round((d_ready - p_ready) * 1000.0, 3)
            if p_ready is not None and d_ready is not None
            else None
        )
        row["d_kv_to_first_token_ms"] = (
            round((first_token - d_ready) * 1000.0, 3)
            if first_token is not None and d_ready is not None
            else None
        )
        d_submit = _optional_float(d.get("d_first_model_submit_monotonic_s"))
        d_scheduled = _optional_float(d.get("d_first_scheduled_monotonic_s"))
        row["d_kv_to_first_scheduled_ms"] = (
            round((d_scheduled - d_ready) * 1000.0, 3)
            if d_scheduled is not None and d_ready is not None
            else None
        )
        row["d_schedule_to_model_submit_ms"] = (
            round((d_submit - d_scheduled) * 1000.0, 3)
            if d_submit is not None and d_scheduled is not None
            else None
        )
        row["d_kv_to_first_model_submit_ms"] = (
            round((d_submit - d_ready) * 1000.0, 3)
            if d_submit is not None and d_ready is not None
            else None
        )
        row["model_submit_to_first_token_ms"] = (
            round((first_token - d_submit) * 1000.0, 3)
            if first_token is not None and d_submit is not None
            else None
        )
        merged_rows.append(row)

    phases_path = output_prefix.with_name(output_prefix.name + "-request-phases.csv")
    with phases_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=phase_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged_rows)

    metrics_path = output_prefix.with_name(output_prefix.name + "-request-metrics.csv")
    with metrics_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "request_id", "category", "status", "error", "prompt_tokens", "output_tokens",
            "ttft_ms", "prefill_execution_ms", "decode_execution_ms",
            "prefill_gpu_forward_ms", "decode_gpu_forward_ms", "decode_service_ms",
            "e2e_ms", "arrival_to_finish_ms", "generation_after_first_token_ms",
            "prefill_execution_steps", "decode_execution_steps",
            "prefill_shared_batch_steps", "decode_shared_batch_steps",
        ], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged_rows)

    chunk_fields = [
        "request_id",
        "sequence",
        "tier",
        "decision",
        "long_request_id",
        "long_preemptions",
        "long_completed_blocks",
        "long_total_blocks",
        "long_progress",
        "long_wait_ms",
        "protection_reasons",
        "block_start",
        "blocks",
        "total_blocks",
        "final",
        "success",
        "accepted_monotonic_s",
        "dispatch_monotonic_s",
        "start_monotonic_s",
        "complete_monotonic_s",
        "received_monotonic_s",
        "receive_final",
        "queue_ms",
        "dispatch_to_start_ms",
        "transfer_ms",
    ]
    chunk_rows = []
    merged_chunks = {**p_chunks}
    for key, d_row in d_chunks.items():
        merged_chunks.setdefault(key, {"request_id": key[0], "sequence": key[1]}).update(d_row)
    for row in merged_chunks.values():
        accepted = _optional_float(row.get("accepted_monotonic_s"))
        dispatched = _optional_float(row.get("dispatch_monotonic_s"))
        started = _optional_float(row.get("start_monotonic_s"))
        completed = _optional_float(row.get("complete_monotonic_s"))
        row["queue_ms"] = (
            round((dispatched - accepted) * 1000.0, 3)
            if accepted is not None and dispatched is not None
            else None
        )
        row["dispatch_to_start_ms"] = (
            round((started - dispatched) * 1000.0, 3)
            if started is not None and dispatched is not None
            else None
        )
        row["transfer_ms"] = (
            round((completed - started) * 1000.0, 3)
            if completed is not None and started is not None
            else None
        )
        chunk_rows.append(row)
    chunk_path = output_prefix.with_name(output_prefix.name + "-chunks.csv")
    with chunk_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=chunk_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(chunk_rows, key=lambda row: (row["request_id"], row["sequence"])))

    if require_execution:
        invalid = []
        for row in merged_rows:
            if row.get("status") != "200":
                continue
            rid = row["request_id"]
            for field in (
                "ttft_ms",
                "e2e_ms",
                "arrival_to_finish_ms",
                "prefill_execution_ms",
                "decode_execution_ms",
                "request_finished_monotonic_s",
            ):
                value = _optional_float(row.get(field))
                if value is None or not math.isfinite(value) or value < 0:
                    invalid.append(f"{rid}: missing/invalid {field}")
            for side, stage in (("p", "prefill"), ("d", "decode")):
                events = execution_by_request.get(rid, {}).get(side, {})
                host, gpu, finish = (events.get(k, []) for k in ("host_step", "gpu_forward", "finished"))
                if not host or len(finish) != 1:
                    invalid.append(f"{rid}: incomplete {stage} host/finish events")
                    continue
                if sorted(int(e["step"]) for e in host) != list(range(len(host))):
                    invalid.append(f"{rid}: duplicate/missing {stage} execution step")
                if gpu and sorted(int(e["scheduled_tokens"]) for e in host) != sorted(int(e["scheduled_tokens"]) for e in gpu):
                    invalid.append(f"{rid}: {stage} host/GPU token counts disagree")
                for e in host + gpu:
                    metric = float(e["elapsed_ms"] if e["event"] == "host_step" else e["gpu_forward_ms"])
                    if not math.isfinite(metric) or metric < 0:
                        invalid.append(f"{rid}: invalid {stage} execution duration")
                if not finish[0]["status"].startswith(("FINISHED_STOPPED", "FINISHED_LENGTH_CAPPED")):
                    invalid.append(f"{rid}: {stage} did not finish normally")
                if any(float(e["monotonic_s"]) < float(e["submitted_monotonic_s"]) for e in host):
                    invalid.append(f"{rid}: invalid {stage} submit/result order")
                if float(finish[0]["monotonic_s"]) < max(float(e["monotonic_s"]) for e in host):
                    invalid.append(f"{rid}: {stage} finished before its final result")
            if row.get("e2e_ms") and row.get("ttft_ms") and float(row["ttft_ms"]) > float(row["e2e_ms"]):
                invalid.append(f"{rid}: TTFT exceeds E2E")
        if invalid:
            raise ValueError("incomplete or invalid execution measurements: " + "; ".join(invalid))

    if require_complete:
        missing = []
        for result in request_rows:
            if result.get("status") != "200":
                continue
            request_id = result["request_id"]
            p = p_phases.get(request_id, {})
            d = d_phases.get(request_id, {})
            absent = []
            if "p_final_chunk_accepted_monotonic_s" not in p:
                absent.append("P final chunk accepted")
            if "p_final_chunk_complete_monotonic_s" not in p:
                absent.append("P final chunk complete")
            if "d_kv_receive_complete_monotonic_s" not in d:
                absent.append("D KV receive complete")
            if "d_first_model_submit_monotonic_s" not in d:
                absent.append("D first model submit")
            if "d_first_scheduled_monotonic_s" not in d:
                absent.append("D first scheduled")
            if "d_final_sequence" not in d:
                absent.append("D final sequence")
            if absent:
                missing.append(f"{request_id}: {', '.join(absent)}")
                continue
            ordered_fields = (
                "d_kv_receive_complete_monotonic_s",
                "d_first_scheduled_monotonic_s",
                "d_first_model_submit_monotonic_s",
            )
            ordered_times = [_optional_float(d[field]) for field in ordered_fields]
            if any(value is None for value in ordered_times) or any(
                later < earlier
                for earlier, later in zip(ordered_times, ordered_times[1:])
            ):
                missing.append(f"{request_id}: invalid D KV/schedule/submit event order")
        if missing:
            raise ValueError("successful requests have incomplete or invalid trace events: " + "; ".join(missing))
    return phases_path, chunk_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", required=True)
    parser.add_argument("--prefill-log", required=True)
    parser.add_argument("--decode-log", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument(
        "--require-execution", action="store_true",
        help="require successful requests to have TTFT/E2E and matching P/D host/GPU execution events",
    )
    parser.add_argument("--wait-seconds", type=float, default=5.0,
                        help="bounded wait for asynchronous execution timing logs")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="fail if a successful request lacks P/D lifecycle trace events",
    )
    args = parser.parse_args()
    deadline = time.monotonic() + max(0.0, args.wait_seconds)
    while True:
        try:
            paths = collect(
                args.requests, args.prefill_log, args.decode_log, args.output_prefix,
                require_complete=args.require_complete,
                require_execution=args.require_execution,
            )
            break
        except ValueError:
            if not args.require_execution or time.monotonic() >= deadline:
                raise
            time.sleep(0.1)
    print("\n".join(map(str, paths)))


if __name__ == "__main__":
    main()
