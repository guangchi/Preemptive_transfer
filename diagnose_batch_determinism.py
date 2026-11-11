#!/usr/bin/env python3
"""Compare solo/mixed outputs without KV transfer, with batch invariance off/on."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from run_pd_load import send


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="/home/ubuntu/zjj/models/Llama-3.1-8B-Instruct")
    parser.add_argument("--port", type=int, default=8310)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    records = [json.loads(line) for line in Path(args.workload).read_text().splitlines()]
    target = next(r for r in records if r["request_id"] == "interleaved15-07-long")
    env = os.environ.copy()
    toolkit = "/home/ubuntu/miniconda3/envs/cc/lib/python3.11/site-packages/nvidia/cu13"
    env.update(CUDA_VISIBLE_DEVICES="0", CUDA_HOME=toolkit, CUDA_ROOT=toolkit,
               NVCC_PREPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
               PATH=toolkit + "/bin:" + env["PATH"], PD_MEASURE_EXECUTION="0")
    os.environ["PD_CAPTURE_LOGPROBS"] = "1"
    endpoint = f"http://127.0.0.1:{args.port}/v1/completions"
    results = []
    for invariant in (0, 1):
        env["VLLM_BATCH_INVARIANT"] = str(invariant)
        log = (output / f"direct-invariant-{invariant}.log").open("w")
        command = ["/home/ubuntu/miniconda3/envs/cc/bin/vllm", "serve", args.model,
                   "--served-model-name", "local", "--max-model-len", "32768",
                   "--gpu-memory-utilization", "0.90", "--no-enable-prefix-caching",
                   "--port", str(args.port)]
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            ready = False
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                if proc.poll() is not None:
                    raise RuntimeError(f"direct worker exited: {proc.returncode}")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=1) as r:
                        ready = r.status == 200
                except Exception:
                    pass
                if ready:
                    break
                time.sleep(0.5)
            if not ready:
                raise TimeoutError("direct worker health timeout")
            for case in ("solo-1", "mixed-1", "mixed-2", "solo-2"):
                batch = [target] if case.startswith("solo") else records
                with ThreadPoolExecutor(max_workers=15) as pool:
                    futures = [pool.submit(send, endpoint, r, 300, time.perf_counter(), False)
                               for r in batch]
                    for future in futures:
                        row = future.result()
                        row.update(batch_invariant=bool(invariant), case=case,
                                   transport="direct-no-kv-transfer")
                        if row["status"] != 200:
                            raise RuntimeError(row["error"])
                        results.append(row)
                        with (output / "responses.jsonl").open("a") as f:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"invariant={invariant} case={case} succeeded", flush=True)
        finally:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=15)
            log.close()
            time.sleep(2)
    summary = {}
    for invariant in (False, True):
        rows = [r for r in results if r["batch_invariant"] == invariant]
        grouped = {}
        for row in rows:
            grouped.setdefault(row["request_id"], []).append(row)
        mismatches = {req: [{"case": r["case"], "text": r["response_text"],
                              "tokens": r["response_token_ids"]} for r in group]
                      for req, group in grouped.items()
                      if len({r["response_token_ids"] for r in group}) > 1}
        summary[str(int(invariant))] = {"responses": len(rows), "mismatches": mismatches}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
