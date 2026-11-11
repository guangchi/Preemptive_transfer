#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-/home/ubuntu/zjj/models/Llama-3.1-8B-Instruct}"
VLLM_BIN="${VLLM_BIN:-/home/ubuntu/miniconda3/envs/cc/bin/vllm}"
P_PORT="${P_PORT:-8100}"
D_PORT="${D_PORT:-8200}"
P_DEVICE="${P_DEVICE:-0}"
D_DEVICE="${D_DEVICE:-1}"
P_SIDE_CHANNEL_PORT="${P_SIDE_CHANNEL_PORT:-5600}"
D_SIDE_CHANNEL_PORT="${D_SIDE_CHANNEL_PORT:-5601}"
NIXL_SIDE_CHANNEL_HOST="${NIXL_SIDE_CHANNEL_HOST:-127.0.0.1}"
RUN_DIR="${RUN_DIR:-/home/ubuntu/zjj/code/logs/pd-$(date +%Y%m%d-%H%M%S)}"
MODE="${MODE:-preemptive}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
CUDA_TOOLKIT_ROOT="${CUDA_TOOLKIT_ROOT:-/home/ubuntu/miniconda3/envs/cc/lib/python3.11/site-packages/nvidia/cu13}"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/cc/bin/python}"
export PD_MEASURE_EXECUTION="${PD_MEASURE_EXECUTION:-1}"
# The local RTX 4090 TCP+cuda_copy path otherwise stages GPU data in ~8KB
# pieces. This is a transport segment size, independent of KV chunk blocks.
# Apply identically to baseline and preemptive and preserve explicit overrides.
export UCX_TCP_TX_SEG_SIZE="${UCX_TCP_TX_SEG_SIZE:-128k}"
export UCX_TCP_RX_SEG_SIZE="${UCX_TCP_RX_SEG_SIZE:-128k}"
export PD_KV_SOURCE_FENCE="${PD_KV_SOURCE_FENCE:-event}"
export PD_STREAM_MAX_IN_FLIGHT="${PD_STREAM_MAX_IN_FLIGHT:-1}"
# Immediate re-submission is an ablation: it reduced host gaps but increased
# native submission time on this TCP backend. Keep the measured yield by default.
export PD_STREAM_EAGER_PROGRESS="${PD_STREAM_EAGER_PROGRESS:-0}"
if [[ "$PD_STREAM_EAGER_PROGRESS" != "0" && "$PD_STREAM_EAGER_PROGRESS" != "1" ]]; then
  echo "PD_STREAM_EAGER_PROGRESS must be 0 or 1" >&2
  exit 2
fi
if ! [[ "$PD_STREAM_MAX_IN_FLIGHT" =~ ^[1-9][0-9]*$ ]]; then
  echo "PD_STREAM_MAX_IN_FLIGHT must be a positive integer" >&2
  exit 2
fi
if [[ "$PD_KV_SOURCE_FENCE" != "event" && "$PD_KV_SOURCE_FENCE" != "device" && "$PD_KV_SOURCE_FENCE" != "writer-device" ]]; then
  echo "PD_KV_SOURCE_FENCE must be event, device, or writer-device" >&2
  exit 2
fi
# Dynamic batching changes matrix/attention reductions in the default vLLM
# kernels. Use its batch-invariant kernels on both P and D for comparisons.
export VLLM_BATCH_INVARIANT="${VLLM_BATCH_INVARIANT:-1}"
if [[ "$VLLM_BATCH_INVARIANT" != "0" && "$VLLM_BATCH_INVARIANT" != "1" ]]; then
  echo "VLLM_BATCH_INVARIANT must be 0 or 1" >&2
  exit 2
fi

if [[ "$MODE" != "preemptive" && "$MODE" != "baseline" ]]; then
  echo "MODE must be preemptive or baseline" >&2
  exit 2
fi
# A live server from another run must never satisfy this run's health check.
"$PYTHON_BIN" - "$P_PORT" "$D_PORT" <<'PY'
import socket
import sys

sockets = []
try:
    for port in map(int, sys.argv[1:]):
        sock = socket.socket()
        sockets.append(sock)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
except OSError as error:
    raise SystemExit(f"P/D port unavailable before launch: {error}")
finally:
    for sock in sockets:
        sock.close()
PY
mkdir -p "$RUN_DIR"

# Torch is CUDA 13 in the experiment environment.  Prefer its bundled nvcc
# for FlashInfer JITs; /usr/local/cuda may still point to an older toolkit.
if [[ -x "$CUDA_TOOLKIT_ROOT/bin/nvcc" ]]; then
  export CUDA_HOME="$CUDA_TOOLKIT_ROOT"
  export CUDA_ROOT="$CUDA_TOOLKIT_ROOT"
  export PATH="$CUDA_TOOLKIT_ROOT/bin:$PATH"
  # The cu13 wheel currently ships CUDA 13.0 headers with a CUDA 13.4
  # compiler. CCCL rejects that minor-version mismatch. This experiment uses
  # the compatibility-check override; validate it again when changing kernels
  # or replace the compiler with the matching CUDA 13.0 toolkit.
  export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:+$NVCC_PREPEND_FLAGS }-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
fi

if [[ "$MODE" == "preemptive" ]]; then
  ENABLED=true
else
  ENABLED=false
fi

KV_EXTRA="$(${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/cc/bin/python} - "$ENABLED" <<'PY'
import json, os, sys
extra = {
    "short_request_preemptive_transfer": sys.argv[1].lower() == "true",
    "chunk_size_blocks": int(os.environ.get("CHUNK_SIZE_BLOCKS", "8")),
    "stream_max_in_flight": int(os.environ.get("PD_STREAM_MAX_IN_FLIGHT", "1")),
    "short_request_threshold_tokens": int(os.environ.get("SHORT_REQUEST_THRESHOLD_TOKENS", "256")),
    "long_request_threshold_tokens": int(os.environ.get("LONG_REQUEST_THRESHOLD_TOKENS", "8192")),
    "long_request_max_preemptions": int(os.environ.get("LONG_REQUEST_MAX_PREEMPTIONS", "3")),
    "long_request_protect_progress": float(os.environ.get("LONG_REQUEST_PROTECT_PROGRESS", "0.85")),
    "long_request_max_preempt_wait_ms": float(os.environ.get("LONG_REQUEST_MAX_PREEMPT_WAIT_MS", "50")),
}
if os.environ.get("PUSH_REGISTRATION_TIMEOUT") is not None:
    extra["push_registration_timeout"] = float(
        os.environ["PUSH_REGISTRATION_TIMEOUT"]
    )
print(json.dumps(extra, separators=(",", ":")))
PY
)"
P_CONFIG="$(${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/cc/bin/python} - "$KV_EXTRA" <<'PY'
import json, sys
extra = json.loads(sys.argv[1])
print(json.dumps({"kv_connector":"NixlPushConnector", "kv_role":"kv_producer", "engine_id":"pd-prefill", "kv_connector_extra_config":extra}, separators=(",", ":")))
PY
)"
D_CONFIG="$(${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/cc/bin/python} - "$KV_EXTRA" <<'PY'
import json, sys
extra = json.loads(sys.argv[1])
print(json.dumps({"kv_connector":"NixlPushConnector", "kv_role":"kv_consumer", "engine_id":"pd-decode", "kv_connector_extra_config":extra}, separators=(",", ":")))
PY
)"

common=("$MODEL" --served-model-name local --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
if [[ "${ENABLE_PREFIX_CACHING:-false}" == "true" ]]; then
  common+=(--enable-prefix-caching)
else
  common+=(--no-enable-prefix-caching)
fi
if [[ -n "${VLLM_API_KEY:-}" ]]; then
  common+=(--api-key "$VLLM_API_KEY")
fi
if [[ "${ENABLE_SCALE_OUT:-false}" == "true" ]]; then
  common+=(--enable-scale-out)
fi
export MODEL MODE P_PORT D_PORT P_DEVICE D_DEVICE P_SIDE_CHANNEL_PORT
export D_SIDE_CHANNEL_PORT NIXL_SIDE_CHANNEL_HOST RUN_DIR MAX_MODEL_LEN
export GPU_MEMORY_UTILIZATION CUDA_TOOLKIT_ROOT PYTHON_BIN
# Both legs receive the same preemptive settings. The connector validates the
# wire configuration before accepting a D registration.
VLLM_NIXL_SIDE_CHANNEL_HOST="$NIXL_SIDE_CHANNEL_HOST" \
VLLM_NIXL_SIDE_CHANNEL_PORT="$P_SIDE_CHANNEL_PORT" \
PD_ROLE=kv_producer \
CUDA_VISIBLE_DEVICES="$P_DEVICE" "$VLLM_BIN" serve "${common[@]}" --port "$P_PORT" \
  --kv-transfer-config "$P_CONFIG" >"$RUN_DIR/prefill.log" 2>&1 &
P_PID=$!
VLLM_NIXL_SIDE_CHANNEL_HOST="$NIXL_SIDE_CHANNEL_HOST" \
VLLM_NIXL_SIDE_CHANNEL_PORT="$D_SIDE_CHANNEL_PORT" \
PD_ROLE=kv_consumer \
CUDA_VISIBLE_DEVICES="$D_DEVICE" "$VLLM_BIN" serve "${common[@]}" --port "$D_PORT" \
  --kv-transfer-config "$D_CONFIG" >"$RUN_DIR/decode.log" 2>&1 &
D_PID=$!
printf '%s\n' "$P_PID" >"$RUN_DIR/prefill.pid"
printf '%s\n' "$D_PID" >"$RUN_DIR/decode.pid"
"$PYTHON_BIN" "$ROOT/write_pd_run_environment.py" --run-dir "$RUN_DIR"
printf 'run_dir=%s\nmode=%s\nprefill_pid=%s\ndecode_pid=%s\nprefill_side_channel=%s:%s\ndecode_side_channel=%s:%s\n' \
  "$RUN_DIR" "$MODE" "$P_PID" "$D_PID" "$NIXL_SIDE_CHANNEL_HOST" "$P_SIDE_CHANNEL_PORT" \
  "$NIXL_SIDE_CHANNEL_HOST" "$D_SIDE_CHANNEL_PORT"
printf 'P endpoint: http://127.0.0.1:%s/v1/completions\nD endpoint: http://127.0.0.1:%s/v1/completions\n' "$P_PORT" "$D_PORT"
printf 'A PD router must submit paired P/D requests with kv_transfer_params; the worker endpoints are not a router.\n'
