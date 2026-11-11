#!/home/ubuntu/miniconda3/envs/cc/bin/python
"""Replay a PD workload JSONL file and write per-request timing CSV."""
from __future__ import annotations

import argparse
import csv
import hashlib
import http.client
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_server_timing(value: str) -> dict[str, float | None]:
    """Parse the router's `prefill;dur=...` and `decode;dur=...` timings."""
    timings: dict[str, float | None] = {
        "prefill_ms": None,
        "decode_ms": None,
        "router_ttft_ms": None,
    }
    for item in value.split(","):
        name, *parameters = item.strip().split(";")
        duration = next(
            (parameter[4:] for parameter in parameters if parameter.startswith("dur=")),
            None,
        )
        if duration is None:
            continue
        try:
            parsed = float(duration)
        except ValueError:
            continue
        if name == "prefill":
            timings["prefill_ms"] = parsed
        elif name == "decode":
            timings["decode_ms"] = parsed
        elif name == "ttft":
            timings["router_ttft_ms"] = parsed
    return timings


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def summarize_request_latencies(rows: list[dict]) -> dict[str, float | int | None]:
    successful = [row["latency_ms"] for row in rows if int(row["status"]) == 200]
    failed_rows = [row for row in rows if int(row["status"]) != 200]
    failed = [row["latency_ms"] for row in failed_rows]
    timeouts = sum(
        int(row["status"]) in (408, 504)
        or "timeout" in str(row.get("error", "")).lower()
        or "timed out" in str(row.get("error", "")).lower()
        for row in failed_rows
    )
    return {
        "ok": len(successful),
        "failed": len(failed_rows),
        "timeouts": timeouts,
        "latency_p50_ms": percentile(successful, 0.50),
        "latency_p95_ms": percentile(successful, 0.95),
        "latency_p99_ms": percentile(successful, 0.99),
        "failure_latency_p50_ms": percentile(failed, 0.50),
        "failure_latency_p95_ms": percentile(failed, 0.95),
        "failure_latency_p99_ms": percentile(failed, 0.99),
    }


def _parse_sse_payload(line: bytes) -> dict | None:
    if not line.startswith(b"data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == b"[DONE]":
        return None
    try:
        event = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return event if isinstance(event, dict) else None


def _event_text(event: dict) -> str:
    return "".join(
        str(choice.get("text", ""))
        for choice in event.get("choices", [])
        if isinstance(choice, dict)
    )


def send(
    endpoint: str,
    record: dict,
    timeout: float,
    submitted_at: float,
    stream: bool = False,
) -> dict:
    body = {
        "model": record.get("model", "local"),
        "prompt": record["prompt"],
        "max_tokens": min(int(record.get("max_tokens", 100)), 100),
        "stream": stream,
        "temperature": 0.0,
        "request_id": record["request_id"],
        "return_token_ids": True,
    }
    # SGLang generates its own request id; omit vLLM's private field.
    if os.environ.get("PD_CLIENT_SGLANG") == "1":
        body.pop("request_id", None)
    if os.environ.get("PD_CAPTURE_LOGPROBS", "0") == "1":
        body["logprobs"] = 5
        body["return_tokens_as_token_ids"] = True
    for key in ("seed", "top_p", "top_k", "min_p", "repetition_penalty", "stop"):
        if key in record:
            body[key] = record[key]
    if stream:
        body["stream_options"] = {"include_usage": True}
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    timings: dict[str, float | None] = {
        "prefill_ms": None,
        "decode_ms": None,
        "router_ttft_ms": None,
    }
    prompt_tokens = None
    output_tokens = None
    first_token_ms = None
    first_token_at = None
    response_text = ""
    response_token_ids = []
    response_logprobs = []
    response_hash = ""
    router_received_at = None
    stream_done = False
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            status = response.status
            error = ""
            timings = parse_server_timing(response.headers.get("Server-Timing", ""))
            try:
                router_received_at = float(
                    response.headers.get("X-PD-Router-Received-Monotonic", "")
                )
            except ValueError:
                pass
            if stream:
                for line in response:
                    if line.startswith(b"data:") and line[5:].strip() == b"[DONE]":
                        stream_done = True
                    event = _parse_sse_payload(line)
                    if event is None:
                        continue
                    if event.get("error") or any(
                        choice.get("finish_reason") in ("abort", "error")
                        for choice in event.get("choices", [])
                    ):
                        status = 502
                        error = f"upstream stream failed: {event}"
                    text = _event_text(event)
                    if text and first_token_ms is None:
                        first_token_at = time.perf_counter()
                        first_token_ms = round((first_token_at - start) * 1000.0, 3)
                    response_text += text
                    for choice in event.get("choices", []):
                        response_token_ids.extend(choice.get("token_ids") or [])
                        if choice.get("logprobs"):
                            response_logprobs.append(choice["logprobs"])
                    usage = event.get("usage") or {}
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    output_tokens = usage.get("completion_tokens", output_tokens)
                if not stream_done and not error:
                    status = 502
                    error = "incomplete SSE stream: missing [DONE]"
            else:
                body_bytes = response.read()
                try:
                    payload = json.loads(body_bytes)
                    usage = payload.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens")
                    output_tokens = usage.get("completion_tokens")
                    choices = payload.get("choices", [])
                    response_text = "".join(
                        str(choice.get("text", "")) for choice in choices
                    )
                    for choice in choices:
                        response_token_ids.extend(choice.get("token_ids") or [])
                        if choice.get("logprobs"):
                            response_logprobs.append(choice["logprobs"])
                except (json.JSONDecodeError, AttributeError):
                    pass
            response_hash = hashlib.sha256(response_text.encode("utf-8")).hexdigest()
    except urllib.error.HTTPError as exc:
        status = getattr(exc, "code", 0) or 0
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = ""
        error = f"HTTP {status}: {detail[:2000]}"
    except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
        status = getattr(exc, "code", 0) or 0
        error = str(exc)
    end = time.perf_counter()
    return {
        "request_id": record["request_id"],
        "category": record.get("category", "unknown"),
        "prompt_tokens": (
            len(record["prompt"]) if isinstance(record["prompt"], list) else None
        ),
        "max_tokens": min(int(record.get("max_tokens", 100)), 100),
        "status": status,
        "submitted_at_monotonic_s": round(submitted_at, 6),
        "request_started_monotonic_s": round(start, 6),
        "request_finished_monotonic_s": round(end, 6),
        "first_token_monotonic_s": round(first_token_at, 9) if first_token_at is not None else None,
        "queue_ms": round(max(0.0, start - submitted_at) * 1000.0, 3),
        "router_received_monotonic_s": (
            round(router_received_at, 6) if router_received_at is not None else None
        ),
        "submit_to_router_ms": (
            round(max(0.0, router_received_at - submitted_at) * 1000.0, 3)
            if router_received_at is not None
            else None
        ),
        "latency_ms": round((end - start) * 1000.0, 3),
        "e2e_ms": round((end - start) * 1000.0, 3),
        "arrival_to_finish_ms": round((end - submitted_at) * 1000.0, 3),
        "generation_after_first_token_ms": (
            round((end - start) * 1000.0 - first_token_ms, 3)
            if first_token_ms is not None else None
        ),
        "ttft_ms": first_token_ms,
        "router_ttft_ms": timings["router_ttft_ms"],
        "prefill_ms": timings["prefill_ms"],
        "decode_ms": timings["decode_ms"],
        "reported_prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "response_sha256": response_hash,
        "response_text": response_text,
        "response_token_ids": json.dumps(response_token_ids),
        "response_logprobs": json.dumps(response_logprobs),
        "error": error,
    }


class GpuSampler:
    def __init__(self, output: Path, interval_s: float = 0.5):
        self.output = output
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def _sample(self) -> None:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("w", encoding="utf-8") as stream:
            stream.write("timestamp,gpu_index,utilization_gpu_pct,memory_used_mib\n")
            while not self.stop_event.is_set():
                try:
                    result = subprocess.run(
                        [
                            "nvidia-smi",
                            "--query-gpu=timestamp,index,utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=2,
                    )
                    for line in result.stdout.splitlines():
                        stream.write(line.replace(", ", ",") + "\n")
                    stream.flush()
                except (OSError, subprocess.SubprocessError) as exc:
                    stream.write(f"sample_error,{type(exc).__name__},,\n")
                    stream.flush()
                self.stop_event.wait(self.interval_s)

    def __enter__(self):
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_exc):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--output", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--interarrival-ms", type=float, default=0.0)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--sample-gpu", action="store_true")
    parser.add_argument("--run-metadata", default="")
    args = parser.parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")

    records = [json.loads(line) for line in Path(args.workload).read_text().splitlines() if line]
    results: list[dict] = []
    batch_start = time.perf_counter()
    gpu_sampler = GpuSampler(Path(args.output).with_name("gpu.csv")) if args.sample_gpu else None
    try:
        if gpu_sampler is not None:
            gpu_sampler.__enter__()
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = []
            for record in records:
                submitted_at = time.perf_counter()
                futures.append(
                    pool.submit(send, args.endpoint, record, args.timeout, submitted_at, args.stream)
                )
                if args.interarrival_ms > 0:
                    time.sleep(args.interarrival_ms / 1000.0)
            for future in as_completed(futures):
                results.append(future.result())
    finally:
        if gpu_sampler is not None:
            gpu_sampler.__exit__()
    elapsed_s = time.perf_counter() - batch_start

    results.sort(key=lambda row: row["request_id"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0])) if results else None
        if writer:
            writer.writeheader()
            writer.writerows(results)
    if args.run_metadata:
        metadata_path = Path(args.run_metadata)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            json.dumps(
                {
                    "workload": str(Path(args.workload).resolve()),
                    "workload_sha256": hashlib.sha256(Path(args.workload).read_bytes()).hexdigest(),
                    "endpoint": args.endpoint,
                    "concurrency": args.concurrency,
                    "interarrival_ms": args.interarrival_ms,
                    "timeout_s": args.timeout,
                    "stream": args.stream,
                    "execution_timing_schema": 2,
                    "measure_execution": os.environ.get("PD_MEASURE_EXECUTION", "1") == "1",
                    "batch_invariant": os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1",
                    "capture_logprobs": os.environ.get("PD_CAPTURE_LOGPROBS", "0") == "1",
                    "gpu_sampling": args.sample_gpu,
                    "request_count": len(records),
                    "model": os.environ.get("MODEL", ""),
                    "mode": os.environ.get("MODE", ""),
                    "run_dir": os.environ.get("RUN_DIR", ""),
                    "environment_json": (
                        str(Path(os.environ["RUN_DIR"]) / "environment.json")
                        if os.environ.get("RUN_DIR")
                        and (Path(os.environ["RUN_DIR"]) / "environment.json").exists()
                        else ""
                    ),
                    "python_executable": sys.executable,
                    "python_version": platform.python_version(),
                    "seed_values": sorted(
                        {record["seed"] for record in records if "seed" in record}
                    ),
                    "category_order": [record.get("category", "unknown") for record in records],
                    "p_device": os.environ.get("P_DEVICE", ""),
                    "d_device": os.environ.get("D_DEVICE", ""),
                    "p_port": os.environ.get("P_PORT", ""),
                    "d_port": os.environ.get("D_PORT", ""),
                    "p_side_channel_port": os.environ.get("P_SIDE_CHANNEL_PORT", ""),
                    "d_side_channel_port": os.environ.get("D_SIDE_CHANNEL_PORT", ""),
                    "max_model_len": os.environ.get("MAX_MODEL_LEN", ""),
                    "gpu_memory_utilization": os.environ.get("GPU_MEMORY_UTILIZATION", ""),
                    "prefix_caching": os.environ.get("ENABLE_PREFIX_CACHING", "false").lower() == "true",
                    "short_threshold_tokens": os.environ.get(
                        "SHORT_REQUEST_THRESHOLD_TOKENS", "256"
                    ),
                    "long_threshold_tokens": os.environ.get(
                        "LONG_REQUEST_THRESHOLD_TOKENS", "8192"
                    ),
                    "chunk_size_blocks": os.environ.get("CHUNK_SIZE_BLOCKS", ""),
                    "max_preemptions": os.environ.get("LONG_REQUEST_MAX_PREEMPTIONS", ""),
                    "protect_progress": os.environ.get("LONG_REQUEST_PROTECT_PROGRESS", ""),
                    "wait_threshold_ms": os.environ.get("LONG_REQUEST_MAX_PREEMPT_WAIT_MS", ""),
                    "vllm_version": importlib.metadata.version("vllm"),
                    "start_monotonic_s": batch_start,
                    "elapsed_s": round(elapsed_s, 6),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    ok = sum(row["status"] == 200 for row in results)
    categories = sorted({row["category"] for row in results})
    by_category = {}
    for category in categories:
        rows = [row for row in results if row["category"] == category]
        by_category[category] = {
            "requests": len(rows),
            **summarize_request_latencies(rows),
        }
    latency_summary = summarize_request_latencies(results)
    print(
        json.dumps(
            {
                "output": str(output),
                "requests": len(results),
                "ok": ok,
                "failed": latency_summary["failed"],
                "timeouts": latency_summary["timeouts"],
                "success_rate": (ok / len(results)) if results else 0.0,
                "elapsed_s": round(elapsed_s, 3),
                "requests_per_second": (
                    round(len(results) / elapsed_s, 3) if elapsed_s > 0 else 0.0
                ),
                "latency_p50_ms": latency_summary["latency_p50_ms"],
                "latency_p95_ms": latency_summary["latency_p95_ms"],
                "latency_p99_ms": latency_summary["latency_p99_ms"],
                "failure_latency_p50_ms": latency_summary["failure_latency_p50_ms"],
                "failure_latency_p95_ms": latency_summary["failure_latency_p95_ms"],
                "failure_latency_p99_ms": latency_summary["failure_latency_p99_ms"],
                "by_category": by_category,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
