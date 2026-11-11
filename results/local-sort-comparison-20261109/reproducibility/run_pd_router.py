#!/home/ubuntu/miniconda3/envs/cc/bin/python
"""Small local PD request pairer for the single-P/single-D experiment."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit


P_ENDPOINT = os.environ.get("P_ENDPOINT", "http://127.0.0.1:8100/v1/completions")
D_ENDPOINT = os.environ.get("D_ENDPOINT", "http://127.0.0.1:8200/v1/completions")
ROUTER_HOST = os.environ.get("ROUTER_HOST", "127.0.0.1")
ROUTER_PORT = int(os.environ.get("ROUTER_PORT", "8000"))
P_ENGINE_ID = os.environ.get("P_ENGINE_ID", "pd-prefill")
D_ENGINE_ID = os.environ.get("D_ENGINE_ID", "pd-decode")
P_SIDE_CHANNEL_HOST = os.environ.get("P_SIDE_CHANNEL_HOST", "127.0.0.1")
D_SIDE_CHANNEL_HOST = os.environ.get("D_SIDE_CHANNEL_HOST", "127.0.0.1")
P_SIDE_CHANNEL_PORT = int(os.environ.get("P_SIDE_CHANNEL_PORT", "5600"))
D_SIDE_CHANNEL_PORT = int(os.environ.get("D_SIDE_CHANNEL_PORT", "5601"))
UPSTREAM_TIMEOUT = float(os.environ.get("UPSTREAM_TIMEOUT", "600"))
UPSTREAM_API_KEY = os.environ.get("UPSTREAM_API_KEY", "")
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", str(32 * 1024 * 1024)))
_DECODE_POOL = ThreadPoolExecutor(max_workers=int(os.environ.get("DECODE_SUBMIT_WORKERS", "32")))


class UpstreamError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"upstream returned HTTP {status}: {body[:1000]}")
        self.status = status
        self.body = body


def internal_request_id(request_id: str) -> str:
    # vLLM's OpenAI Completion endpoint prefixes the body request_id with
    # "cmpl-" and appends "-0" for its single prompt.
    return f"cmpl-{request_id}-0"


def build_pd_payloads(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    request_id = str(record.get("request_id") or f"pd-{time.time_ns()}")
    if not request_id or len(request_id) > 200:
        raise ValueError("request_id must contain 1..200 characters")
    if "prompt" not in record:
        raise ValueError("prompt is required")

    model = record.get("model", "local")
    common = {
        "model": model,
        "prompt": record["prompt"],
        "stream": False,
        "temperature": 0.0,
        "request_id": request_id,
    }
    decode = {
        **common,
        "max_tokens": min(int(record.get("max_tokens", 100)), 100),
        "stream": bool(record.get("stream", False)),
        "kv_transfer_params": {
            "do_remote_prefill": True,
            "remote_engine_id": P_ENGINE_ID,
            "remote_request_id": internal_request_id(request_id),
            "remote_host": P_SIDE_CHANNEL_HOST,
            "remote_port": P_SIDE_CHANNEL_PORT,
            "tp_size": 1,
            "dcp_size": 1,
            "pp_size": 1,
        },
    }
    prefill = {
        **common,
        "max_tokens": 1,
        "ignore_eos": True,
        "kv_transfer_params": {
            "do_remote_decode": True,
            "remote_engine_id": D_ENGINE_ID,
            "remote_host": D_SIDE_CHANNEL_HOST,
            "remote_port": D_SIDE_CHANNEL_PORT,
            "tp_size": 1,
            "pp_size": 1,
        },
    }
    for key in (
        "seed",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "stop",
    ):
        if key in record:
            prefill[key] = record[key]
            decode[key] = record[key]
    # Audit metadata is needed only for the generated D response. Do not ask
    # the P leg to compute extra logprobs or return its disposable output.
    for key in ("return_token_ids", "return_tokens_as_token_ids", "logprobs"):
        if key in record:
            decode[key] = record[key]
    if decode["stream"]:
        decode["stream_options"] = {"include_usage": True}
    return prefill, decode


def post_json(endpoint: str, body: dict[str, Any]) -> tuple[int, bytes, float]:
    payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if UPSTREAM_API_KEY:
        headers["Authorization"] = f"Bearer {UPSTREAM_API_KEY}"
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers=headers,
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT) as response:
            return response.status, response.read(), (time.perf_counter() - start) * 1000
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise UpstreamError(exc.code, body_text) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise UpstreamError(502, str(exc)) from exc


def open_stream(endpoint: str, body: dict[str, Any]):
    """Open an upstream SSE response without buffering generated tokens."""
    payload = json.dumps(body, separators=(",", ":")).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    if UPSTREAM_API_KEY:
        headers["Authorization"] = f"Bearer {UPSTREAM_API_KEY}"
    req = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
    try:
        return urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT)
    except urllib.error.HTTPError as exc:
        body_text = exc.read().decode("utf-8", errors="replace")
        raise UpstreamError(exc.code, body_text) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise UpstreamError(502, str(exc)) from exc


def _stream_line_has_output_token(line: bytes) -> bool:
    if not line.startswith(b"data:"):
        return False
    payload = line[5:].strip()
    if not payload or payload == b"[DONE]":
        return False
    try:
        event = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    for choice in event.get("choices", []):
        if choice.get("text") or choice.get("token_ids"):
            return True
    return False


def abort_upstream(endpoint: str, request_id: str) -> bool:
    """Abort a paired vLLM completion after its sibling has failed."""
    parsed = urlsplit(endpoint)
    abort_endpoint = f"{parsed.scheme}://{parsed.netloc}/inference/v1/abort_requests"
    payload = json.dumps(
        {"request_ids": [internal_request_id(request_id)]}, separators=(",", ":")
    ).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if UPSTREAM_API_KEY:
        headers["Authorization"] = f"Bearer {UPSTREAM_API_KEY}"
    req = urllib.request.Request(
        abort_endpoint,
        data=payload,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=min(UPSTREAM_TIMEOUT, 2.0)) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


class Handler(BaseHTTPRequestHandler):
    server_version = "LocalPDPairer/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"{self.log_date_time_string()} {self.client_address[0]} {fmt % args}")

    def do_GET(self) -> None:
        if urlsplit(self.path).path != "/health":
            self.send_error(404)
            return
        data = json.dumps({"status": "ok", "prefill": P_ENDPOINT, "decode": D_ENDPOINT,
                           "local_sort": os.environ.get("PD_LOCAL_SORT", "0") == "1",
                           "local_sort_window_ms": float(os.environ.get("PD_LOCAL_SORT_WINDOW_MS", "20"))}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/v1/completions":
            self._send_json(404, {"error": "route not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > MAX_BODY_BYTES:
                self._send_json(413, {"error": "request body size is invalid"})
                return
            record = json.loads(self.rfile.read(length))
            if not isinstance(record, dict):
                raise ValueError("request body must be a JSON object")
            prefill, decode = build_pd_payloads(record)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, TypeError) as exc:
            self._send_json(400, {"error": str(exc)})
            return

        # D registers its receive blocks while the request waits for the P
        # transfer. The P worker also tolerates chunks arriving before the
        # registration notification and queues them until the handshake.
        request_id = str(record.get("request_id") or "")
        request_started = time.perf_counter()
        router_received = request_started
        stream_decode = bool(decode.get("stream"))
        decode_call = open_stream if stream_decode else post_json
        decode_future = _DECODE_POOL.submit(decode_call, D_ENDPOINT, decode)
        try:
            _, _, prefill_ms = post_json(P_ENDPOINT, prefill)
        except UpstreamError as exc:
            decode_future.cancel()
            if not abort_upstream(D_ENDPOINT, request_id):
                self.log_message("could not abort D request %s after P failure", request_id)
            self._send_json(exc.status, {"error": str(exc), "upstream_body": exc.body})
            return
        try:
            if stream_decode:
                response = decode_future.result(timeout=UPSTREAM_TIMEOUT)
                status = response.status
                decode_ms = None
            else:
                status, response, decode_ms = decode_future.result(timeout=UPSTREAM_TIMEOUT)
        except UpstreamError as exc:
            if not abort_upstream(P_ENDPOINT, request_id):
                self.log_message("could not abort P request %s after D failure", request_id)
            self._send_json(exc.status, {"error": str(exc), "upstream_body": exc.body})
            return
        except TimeoutError:
            decode_future.cancel()
            abort_upstream(P_ENDPOINT, request_id)
            abort_upstream(D_ENDPOINT, request_id)
            self._send_json(504, {"error": "decode request timed out"})
            return

        if stream_decode:
            first_lines: list[bytes] = []
            first_token_at: float | None = None
            try:
                while True:
                    line = response.readline()
                    if not line:
                        break
                    first_lines.append(line)
                    if _stream_line_has_output_token(line):
                        first_token_at = time.perf_counter()
                        break
            except (OSError, TimeoutError) as exc:
                response.close()
                abort_upstream(P_ENDPOINT, request_id)
                self._send_json(502, {"error": f"decode stream failed before first token: {exc}"})
                return

            self.send_response(status)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("X-PD-Router-Received-Monotonic", f"{router_received:.9f}")
            timing = f"prefill;dur={prefill_ms:.3f}"
            if first_token_at is not None:
                timing += f", ttft;dur={(first_token_at - request_started) * 1000.0:.3f}"
            self.send_header("Server-Timing", timing)
            self.end_headers()

            def write_chunk(data: bytes) -> None:
                self.wfile.write(f"{len(data):X}\r\n".encode("ascii"))
                self.wfile.write(data)
                self.wfile.write(b"\r\n")

            try:
                for line in first_lines:
                    write_chunk(line)
                self.wfile.flush()
                for line in response:
                    write_chunk(line)
                    if line in (b"\n", b"\r\n"):
                        self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, OSError):
                # Client cancellation also stops generation upstream.
                abort_upstream(D_ENDPOINT, request_id)
            finally:
                response.close()
            return

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.send_header(
            "Server-Timing",
            f"prefill;dur={prefill_ms:.3f}, decode;dur={decode_ms:.3f}",
        )
        self.send_header("X-PD-Router-Received-Monotonic", f"{router_received:.9f}")
        self.end_headers()
        self.wfile.write(response)

    def _send_json(self, status: int, value: dict[str, Any]) -> None:
        data = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    server = ThreadingHTTPServer((ROUTER_HOST, ROUTER_PORT), Handler)
    print(f"PD router listening on http://{ROUTER_HOST}:{ROUTER_PORT}")
    print(f"P endpoint: {P_ENDPOINT} ({P_ENGINE_ID}, {P_SIDE_CHANNEL_HOST}:{P_SIDE_CHANNEL_PORT})")
    print(f"D endpoint: {D_ENDPOINT} ({D_ENGINE_ID}, {D_SIDE_CHANNEL_HOST}:{D_SIDE_CHANNEL_PORT})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        _DECODE_POOL.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
