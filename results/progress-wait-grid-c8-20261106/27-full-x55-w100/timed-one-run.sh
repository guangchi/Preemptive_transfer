#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: configure the experiment with environment variables, then run $0"
  echo "WORKLOAD, RESULT_ROOT, CHUNK_SIZES, MAX_PREEMPTIONS, PROTECT_PROGRESS, WAIT_MS"
  echo "PD_KV_SOURCE_FENCE, PD_STREAM_MAX_IN_FLIGHT, UCX_TCP_TX_SEG_SIZE, UCX_TCP_RX_SEG_SIZE"
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "Unexpected argument: $1; use --help" >&2
  exit 2
fi

# Sweep the preemption controls. Set PD_ENDPOINT to the PD router endpoint;
# the two vLLM workers are intentionally kept separate from request routing.
ROOT="/home/ubuntu/zjj/code"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/cc/bin/python}"
MODEL="${MODEL:-/home/ubuntu/zjj/models/Llama-3.1-8B-Instruct}"
WORKLOAD="${WORKLOAD:-$ROOT/workload.jsonl}"
PD_ENDPOINT="${PD_ENDPOINT:-}"
START_LOCAL_ROUTER="${START_LOCAL_ROUTER:-}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT/results}"
P_DEVICE="${P_DEVICE:-0}"
D_DEVICE="${D_DEVICE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-false}"
export PD_MEASURE_EXECUTION="${PD_MEASURE_EXECUTION:-1}"
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"
if [[ "$PD_MEASURE_EXECUTION" != "1" || "${STREAM_RESPONSES:-true}" != "true" ]]; then
  echo "Sweep metrics require PD_MEASURE_EXECUTION=1 and STREAM_RESPONSES=true for execution timing and TTFT" >&2
  exit 2
fi
WAIT_SECONDS="${WAIT_SECONDS:-15}"
CHUNK_SIZES="${CHUNK_SIZES:-8}"
MAX_PREEMPTIONS="${MAX_PREEMPTIONS:-3}"
PROTECT_PROGRESS_VALUES="${PROTECT_PROGRESS_VALUES:-0.85}"
WAIT_THRESHOLDS_MS="${WAIT_THRESHOLDS_MS:-50}"
REQUESTS="${REQUESTS:-120}"
ROUTER_PORT="${ROUTER_PORT:-8000}"
P_PORT="${P_PORT:-8100}"
D_PORT="${D_PORT:-8200}"
P_SIDE_CHANNEL_PORT="${P_SIDE_CHANNEL_PORT:-5600}"
D_SIDE_CHANNEL_PORT="${D_SIDE_CHANNEL_PORT:-5601}"
SUMMARY_FILE="${SUMMARY_FILE:-$RESULT_ROOT/sweep-summary.csv}"
SUMMARY_JSON="${SUMMARY_JSON:-$RESULT_ROOT/sweep-summary.json}"
RUN_BASELINE="${RUN_BASELINE:-true}"
BASELINE_RESULT_DIR="${BASELINE_RESULT_DIR:-}"
SWEEP_START_WALL="$(date +%s.%N)"

ACTIVE_RUN_DIR=""
ACTIVE_ROUTER_PID=""
SWEEP_FAILED=0

kill_tree() {
  local root_pid="$1" child
  [[ -n "$root_pid" && "$root_pid" =~ ^[0-9]+$ ]] || return 0
  while read -r child; do
    kill_tree "$child"
  done < <(pgrep -P "$root_pid" 2>/dev/null || true)
  kill -TERM "$root_pid" 2>/dev/null || true
}

wait_for_health() {
  local run_dir="$1"
  if ! "$PYTHON_BIN" - "$P_PORT" "$D_PORT" "$run_dir" <<'PY'
import os
import sys
import time
import urllib.request

ports = [int(sys.argv[1]), int(sys.argv[2])]
run_dir = sys.argv[3]
pid_files = [os.path.join(run_dir, name) for name in ("prefill.pid", "decode.pid")]
for _ in range(720):
    if any(not os.path.exists(path) for path in pid_files):
        time.sleep(0.25)
        continue
    try:
        pids = [int(open(path).read()) for path in pid_files]
        if any(not os.path.exists(f"/proc/{pid}") for pid in pids):
            raise RuntimeError("worker exited")
        if all(urllib.request.urlopen(
            f"http://127.0.0.1:{port}/health", timeout=1
        ).status == 200 for port in ports):
            raise SystemExit(0)
    except Exception:
        pass
    time.sleep(0.25)
raise SystemExit(
    f"P/D workers did not become healthy; inspect {run_dir}/prefill.log and "
    f"{run_dir}/decode.log"
)
PY
  then
    return 1
  fi
}

cleanup_active() {
  if [[ -n "$ACTIVE_ROUTER_PID" ]]; then
    kill_tree "$ACTIVE_ROUTER_PID"
    wait "$ACTIVE_ROUTER_PID" 2>/dev/null || true
    ACTIVE_ROUTER_PID=""
  fi
  if [[ -n "$ACTIVE_RUN_DIR" ]]; then
    for pid_file in "$ACTIVE_RUN_DIR/prefill.pid" "$ACTIVE_RUN_DIR/decode.pid"; do
      if [[ -f "$pid_file" ]]; then
        kill_tree "$(cat "$pid_file")"
      fi
    done
    # Do not launch the next parameter point while old workers still own
    # their CUDA contexts.  This avoids false OOM failures between points.
    for pid_file in "$ACTIVE_RUN_DIR/prefill.pid" "$ACTIVE_RUN_DIR/decode.pid"; do
      if [[ -f "$pid_file" ]]; then
        pid="$(cat "$pid_file")"
        for _ in $(seq 1 120); do
          kill -0 "$pid" 2>/dev/null || break
          sleep 0.25
        done
      fi
    done
    ACTIVE_RUN_DIR=""
  fi
}
trap cleanup_active EXIT INT TERM

if [[ ! -f "$WORKLOAD" ]]; then
  "$ROOT/generate_pd_workload.py" --model "$MODEL" --output "$WORKLOAD" --requests "$REQUESTS"
fi
if [[ -z "$PD_ENDPOINT" ]]; then
  PD_ENDPOINT="http://127.0.0.1:${ROUTER_PORT:-8000}/v1/completions"
  START_LOCAL_ROUTER=1
elif [[ -z "$START_LOCAL_ROUTER" ]]; then
  START_LOCAL_ROUTER=0
fi
mkdir -p "$RESULT_ROOT"
if [[ ! -f "$SUMMARY_FILE" ]]; then
  printf 'tag,mode,chunk_size_blocks,max_preemptions,protect_progress,wait_ms,status,requests,ok,success_rate,elapsed_s,error\n' >"$SUMMARY_FILE"
fi

run_one() {
  local mode="$1" chunk="$2" max_preempt="$3" protect="$4" wait_ms="$5"
  local tag="${mode}-c${chunk}-p${max_preempt}-x${protect}-w${wait_ms}"
  local run_dir="$RESULT_ROOT/$tag"
  local t_start t_pair t_workers t_router t_warmup t_load t_trace t_cleanup
  t_start="$(date +%s.%N)"
  mkdir -p "$run_dir"
  ACTIVE_RUN_DIR="$run_dir"
  ACTIVE_ROUTER_PID=""
  export MODEL RUN_DIR="$run_dir" MODE="$mode" P_DEVICE D_DEVICE
  export MAX_MODEL_LEN GPU_MEMORY_UTILIZATION ENABLE_PREFIX_CACHING
  export P_PORT D_PORT P_SIDE_CHANNEL_PORT D_SIDE_CHANNEL_PORT
  export CHUNK_SIZE_BLOCKS="$chunk"
  export LONG_REQUEST_MAX_PREEMPTIONS="$max_preempt"
  export LONG_REQUEST_PROTECT_PROGRESS="$protect"
  export LONG_REQUEST_MAX_PREEMPT_WAIT_MS="$wait_ms"
  local status=0 router_pid=""
  if P_PORT="$P_PORT" D_PORT="$D_PORT" \
      P_SIDE_CHANNEL_PORT="$P_SIDE_CHANNEL_PORT" \
      D_SIDE_CHANNEL_PORT="$D_SIDE_CHANNEL_PORT" \
      "$ROOT/run_pd_pair.sh" >"$run_dir/launch.txt"; then
    :
  else
    status=$?
  fi
  t_pair="$(date +%s.%N)"

  if [[ "$status" -eq 0 ]]; then
    wait_for_health "$run_dir" || status=1
  fi
  t_workers="$(date +%s.%N)"

  if [[ "$status" -eq 0 && "$START_LOCAL_ROUTER" == "1" ]]; then
    export P_ENDPOINT="http://127.0.0.1:${P_PORT:-8100}/v1/completions"
    export D_ENDPOINT="http://127.0.0.1:${D_PORT:-8200}/v1/completions"
    export P_SIDE_CHANNEL_PORT D_SIDE_CHANNEL_PORT
    "$PYTHON_BIN" "$ROOT/run_pd_router.py" >"$run_dir/router.log" 2>&1 &
    router_pid=$!
    printf '%s\n' "$router_pid" >"$run_dir/router.pid"
    ACTIVE_ROUTER_PID="$router_pid"
    if ! "$PYTHON_BIN" - "$ROUTER_PORT" "$router_pid" <<'PY'
import sys, time, urllib.request
port, pid = sys.argv[1], int(sys.argv[2])
for _ in range(120):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as r:
            if r.status == 200:
                raise SystemExit(0)
    except Exception:
        pass
    try:
        import os
        os.kill(pid, 0)
    except OSError:
        break
    time.sleep(0.25)
raise SystemExit("local PD router did not become healthy")
PY
    then
      status=1
    fi
  fi
  t_router="$(date +%s.%N)"

  if [[ "$status" -eq 0 ]]; then
    sleep "$WAIT_SECONDS"
    t_warmup="$(date +%s.%N)"
    local load_options=(--sample-gpu --run-metadata "$run_dir/run.json")
    if [[ "${STREAM_RESPONSES:-true}" == "true" ]]; then
      load_options+=(--stream)
    fi
    if "$PYTHON_BIN" "$ROOT/run_pd_load.py" --workload "$WORKLOAD" --endpoint "$PD_ENDPOINT" \
      --output "$run_dir/requests.csv" --concurrency "${CONCURRENCY:-8}" \
      --interarrival-ms "${INTERARRIVAL_MS:-0}" "${load_options[@]}" >"$run_dir/load.json"; then
      :
    else
      status=$?
    fi
    t_load="$(date +%s.%N)"
  else
    t_warmup="$t_router"
    t_load="$t_router"
  fi

  local trace_status=0
  if [[ -f "$run_dir/requests.csv" && -f "$run_dir/prefill.log" && -f "$run_dir/decode.log" ]]; then
    local trace_options=()
    if [[ "$mode" == "preemptive" ]]; then
      trace_options+=(--require-complete)
    fi
    trace_options+=(--require-execution)
    if ! "$PYTHON_BIN" "$ROOT/collect_pd_trace.py" --requests "$run_dir/requests.csv" \
      --prefill-log "$run_dir/prefill.log" --decode-log "$run_dir/decode.log" \
      --output-prefix "$run_dir/trace" "${trace_options[@]}" \
      >"$run_dir/trace-collection.txt" 2>&1; then
      trace_status=1
      status=1
    fi
  elif [[ -f "$run_dir/requests.csv" ]]; then
    trace_status=1
    status=1
    printf 'missing prefill/decode worker log for execution measurements\n' \
      >"$run_dir/trace-collection.txt"
  fi
  t_trace="$(date +%s.%N)"

  local result_status="failed" requests=0 ok=0 success_rate=0 elapsed_s="" error=""
  if [[ -f "$run_dir/load.json" ]]; then
    read -r requests ok success_rate elapsed_s < <(
      "$PYTHON_BIN" - "$run_dir/load.json" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
except Exception:
    print("0 0 0 ")
else:
    print(data.get("requests", 0), data.get("ok", 0),
          data.get("success_rate", 0), data.get("elapsed_s", ""))
PY
    )
    [[ "$ok" == "$requests" && "$requests" != "0" ]] && result_status="passed"
  fi
  if [[ "$status" -ne 0 ]]; then
    result_status="failed"
    if [[ "$trace_status" -ne 0 ]]; then
      error="trace_collection_failed"
    else
      error="worker_or_load_failed"
    fi
  elif [[ "$result_status" != "passed" ]]; then
    error="http_failure_or_empty_workload"
    status=1
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$tag" "$mode" "$chunk" "$max_preempt" "$protect" "$wait_ms" \
    "$result_status" "$requests" "$ok" "$success_rate" "$elapsed_s" "$error" \
    >>"$SUMMARY_FILE"

  "$PYTHON_BIN" - "$run_dir" "$tag" "$mode" "$chunk" "$max_preempt" \
    "$protect" "$wait_ms" "$result_status" "$requests" "$ok" \
    "$success_rate" "$elapsed_s" "$error" "$SUMMARY_JSON" <<'PY'
import json
import os
import sys
from pathlib import Path

(run_dir, tag, mode, chunk, max_preempt, protect, wait_ms, status, requests,
 ok, success_rate, elapsed_s, error, summary_path) = sys.argv[1:]
root = Path(run_dir)
record = {
    "tag": tag,
    "mode": mode,
    "chunk_size_blocks": int(chunk),
    "max_preemptions": int(max_preempt),
    "protect_progress": float(protect),
    "wait_threshold_ms": float(wait_ms),
    "status": status,
    "requests": int(requests),
    "ok": int(ok),
    "success_rate": float(success_rate),
    "elapsed_s": float(elapsed_s) if elapsed_s else None,
    "error": error,
    "run_dir": str(root.resolve()),
    "files": {
        name: str((root / name).resolve())
        for name in (
            "run.json", "environment.json", "load.json", "requests.csv", "gpu.csv",
            "trace-request-phases.csv", "trace-request-metrics.csv", "trace-execution.csv", "trace-chunks.csv",
            "prefill.log", "decode.log", "router.log",
        )
        if (root / name).exists()
    },
}
(root / "status.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
summary = Path(summary_path)
summary.parent.mkdir(parents=True, exist_ok=True)
try:
    data = json.loads(summary.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    data = {"runs": []}
runs = [item for item in data.get("runs", []) if item.get("tag") != tag]
runs.append(record)
runs.sort(key=lambda item: item["tag"])
tmp = summary.with_suffix(summary.suffix + ".tmp")
tmp.write_text(json.dumps({"runs": runs}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(tmp, summary)
PY

  t_cleanup="$(date +%s.%N)"
  cleanup_active
  "$PYTHON_BIN" - "$run_dir" "$t_start" "$t_pair" "$t_workers" "$t_router" "$t_warmup" "$t_load" "$t_trace" "$t_cleanup" "$(date +%s.%N)" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
names = ("start", "pair_launch_complete", "workers_healthy", "router_healthy",
         "load_start", "load_complete", "trace_complete", "cleanup_start", "cleanup_complete")
times = dict(zip(names, map(float, sys.argv[2:])))
phases = {"pair_launch_s": times["pair_launch_complete"] - times["start"],
          "worker_health_s": times["workers_healthy"] - times["pair_launch_complete"],
          "router_start_s": times["router_healthy"] - times["workers_healthy"],
          "warmup_s": times["load_start"] - times["router_healthy"],
          "load_replay_s": times["load_complete"] - times["load_start"],
          "trace_collection_s": times["trace_complete"] - times["load_complete"],
          "cleanup_s": times["cleanup_complete"] - times["cleanup_start"],
          "group_wall_s": times["cleanup_complete"] - times["start"]}
record = {"timestamps_unix_s": times, "phase_durations_s": phases}
(path / "experiment-timing.json").write_text(json.dumps(record, indent=2) + "\n")
status_path = path / "status.json"
if status_path.exists():
    status = json.loads(status_path.read_text())
    status.setdefault("files", {})["experiment-timing.json"] = str((path / "experiment-timing.json").resolve())
    status_path.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
print(json.dumps(record))
PY
  "$PYTHON_BIN" "$ROOT/collect_pd_total_time.py" --root "$run_dir" \
    --output "$run_dir/workload-total-times.json" >>"$run_dir/trace-collection.txt" 2>&1 || status=1
  ACTIVE_RUN_DIR=""
  ACTIVE_ROUTER_PID=""
  return "$status"
}


run_one "preemptive" "8" "3" "0.55" "100"
