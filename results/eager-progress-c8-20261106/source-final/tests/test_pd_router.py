import importlib.util
import json
import http.client
import hashlib
import threading
import time
import pytest
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock, patch


ROUTER_PATH = Path(__file__).resolve().parents[1] / "run_pd_router.py"
SPEC = importlib.util.spec_from_file_location("run_pd_router", ROUTER_PATH)
router = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(router)

LOAD_PATH = Path(__file__).resolve().parents[1] / "run_pd_load.py"
LOAD_SPEC = importlib.util.spec_from_file_location("run_pd_load", LOAD_PATH)
load = importlib.util.module_from_spec(LOAD_SPEC)
assert LOAD_SPEC.loader is not None
LOAD_SPEC.loader.exec_module(load)


def test_router_pairs_same_request_and_routes_to_opposite_engines():
    record = {
        "request_id": "case-123",
        "model": "local",
        "prompt": [11, 22, 33],
        "max_tokens": 19,
    }
    prefill, decode = router.build_pd_payloads(record)

    assert prefill["request_id"] == decode["request_id"] == "case-123"
    assert prefill["prompt"] == decode["prompt"] == [11, 22, 33]
    assert prefill["max_tokens"] == 1
    assert decode["max_tokens"] == 19
    assert prefill["kv_transfer_params"]["do_remote_decode"] is True
    assert decode["kv_transfer_params"]["do_remote_prefill"] is True
    assert (
        prefill["kv_transfer_params"]["remote_engine_id"]
        == router.D_ENGINE_ID
    )
    assert (
        decode["kv_transfer_params"]["remote_engine_id"]
        == router.P_ENGINE_ID
    )
    assert (
        decode["kv_transfer_params"]["remote_request_id"]
        == router.internal_request_id("case-123")
    )
    assert prefill["kv_transfer_params"]["tp_size"] == 1
    assert decode["kv_transfer_params"]["tp_size"] == 1


def test_router_caps_output_length_and_rejects_missing_prompt():
    prefill, decode = router.build_pd_payloads(
        {"request_id": "case-456", "prompt": [1], "max_tokens": 1000}
    )
    assert prefill["max_tokens"] == 1
    assert decode["max_tokens"] == 100

    try:
        router.build_pd_payloads({"request_id": "empty"})
    except ValueError as exc:
        assert "prompt" in str(exc)
    else:
        raise AssertionError("missing prompt should be rejected")


def test_router_stream_payload_preserves_deterministic_sampling():
    prefill, decode = router.build_pd_payloads(
        {
            "request_id": "stream-case",
            "prompt": [1, 2, 3],
            "stream": True,
            "seed": 17,
            "top_p": 0.8,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
            "logprobs": 5,
        }
    )

    assert prefill["stream"] is False
    assert decode["stream"] is True
    assert decode["stream_options"] == {"include_usage": True}
    assert prefill["seed"] == decode["seed"] == 17
    assert prefill["top_p"] == decode["top_p"] == 0.8
    assert decode["return_token_ids"] is True
    assert decode["logprobs"] == 5
    assert "logprobs" not in prefill


def test_router_stream_token_detection_skips_usage_and_empty_chunks():
    assert not router._stream_line_has_output_token(b"data: [DONE]\n")
    assert not router._stream_line_has_output_token(
        b'data: {"choices":[],"usage":{"prompt_tokens":4}}\n'
    )
    assert not router._stream_line_has_output_token(
        b'data: {"choices":[{"text":""}]}\n'
    )
    assert router._stream_line_has_output_token(
        b'data: {"choices":[{"text":"Hi"}]}\n'
    )


def test_load_recorder_parses_router_phase_timings():
    assert load.parse_server_timing(
        "prefill;dur=12.5, decode;dur=39.25"
    ) == {"prefill_ms": 12.5, "decode_ms": 39.25, "router_ttft_ms": None}
    assert load.parse_server_timing("prefill;dur=bad") == {
        "prefill_ms": None,
        "decode_ms": None,
        "router_ttft_ms": None,
    }
    assert load.parse_server_timing("prefill;dur=2, ttft;dur=8")["router_ttft_ms"] == 8.0
    assert load._parse_sse_payload(b'data: {"choices":[{"text":"A"}]}\n') == {
        "choices": [{"text": "A"}]
    }
    assert load._event_text({"choices": [{"text": "A"}, {"text": "B"}]}) == "AB"


def test_load_records_streamed_token_ids_text_and_logprobs_for_audit():
    events = [
        b'data: {"choices":[{"text":"Hello","token_ids":[42],"logprobs":{"tokens":["token_id:42"],"token_logprobs":[-0.1],"top_logprobs":[{"token_id:42":-0.1,"token_id:43":-0.2}]}}]}\n',
        b'data: {"choices":[{"text":" world","token_ids":[43]}]}\n',
        b'data: {"choices":[],"usage":{"completion_tokens":2}}\n',
        b'data: [DONE]\n',
    ]
    response = MagicMock()
    response.status = 200
    response.headers = {}
    response.__enter__.return_value = response
    response.__iter__.return_value = iter(events)
    with patch.object(load.urllib.request, "urlopen", return_value=response), patch.dict(
        load.os.environ, {"PD_CAPTURE_LOGPROBS": "1"}
    ):
        row = load.send("http://localhost/v1/completions", {
            "request_id": "audit", "prompt": [1], "max_tokens": 2,
        }, timeout=1, submitted_at=time.perf_counter(), stream=True)
    assert row["status"] == 200
    assert row["response_text"] == "Hello world"
    assert json.loads(row["response_token_ids"]) == [42, 43]
    assert json.loads(row["response_logprobs"])[0]["top_logprobs"][0]["token_id:43"] == -0.2
    assert row["response_sha256"] == hashlib.sha256(b"Hello world").hexdigest()


@pytest.mark.parametrize("events", [
    [b'data: {"choices":[{"text":"partial"}]}\n'],
    [b'data: {"choices":[{"text":"partial","finish_reason":"abort"}]}\n', b'data: [DONE]\n'],
    [b'data: {"error":{"message":"GPU failed"}}\n', b'data: [DONE]\n'],
])
def test_load_rejects_truncated_aborted_and_error_streams_even_with_http_200(events):
    response = MagicMock()
    response.status = 200
    response.headers = {}
    response.__enter__.return_value = response
    response.__iter__.return_value = iter(events)
    with patch.object(load.urllib.request, "urlopen", return_value=response):
        result = load.send("http://localhost/v1/completions", {
            "request_id": "bad-stream", "prompt": [1], "max_tokens": 8,
        }, timeout=1, submitted_at=time.perf_counter(), stream=True)
    assert result["status"] == 502
    assert result["error"]
    assert result["e2e_ms"] >= 0


def test_latency_summary_separates_success_failure_and_timeout_samples():
    summary = load.summarize_request_latencies(
        [
            {"status": 200, "latency_ms": 100.0, "error": ""},
            {"status": 200, "latency_ms": 200.0, "error": ""},
            {"status": 504, "latency_ms": 900.0, "error": "decode timed out"},
            {"status": 0, "latency_ms": 800.0, "error": "connection timeout"},
            {"status": 500, "latency_ms": 500.0, "error": "upstream failure"},
        ]
    )
    assert summary["ok"] == 2
    assert summary["failed"] == 3
    assert summary["timeouts"] == 2
    assert summary["latency_p50_ms"] == 100.0
    assert summary["failure_latency_p50_ms"] == 800.0


def test_load_recorder_streams_sse_and_records_ttft_and_output_hash():
    class StreamHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Server-Timing", "prefill;dur=12.5")
            self.send_header(
                "X-PD-Router-Received-Monotonic", f"{time.perf_counter():.9f}"
            )
            self.end_headers()
            for event in (
                {"choices": [{"text": "Hello"}]},
                {"choices": [{"text": " world"}]},
                {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
            ):
                self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
                self.wfile.flush()
                time.sleep(0.01)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), StreamHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        result = load.send(
            f"http://127.0.0.1:{server.server_port}/v1/completions",
            {
                "request_id": "stream-load-test",
                "prompt": [1, 2, 3, 4, 5],
                "max_tokens": 2,
            },
            timeout=3,
            submitted_at=time.perf_counter() - 0.02,
            stream=True,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert result["status"] == 200
    assert result["ttft_ms"] is not None and result["ttft_ms"] >= 0
    assert result["response_sha256"] == hashlib.sha256(b"Hello world").hexdigest()
    assert result["reported_prompt_tokens"] == 5
    assert result["output_tokens"] == 2
    assert result["prefill_ms"] == 12.5
    assert result["queue_ms"] > 0
    assert result["submit_to_router_ms"] >= result["queue_ms"]
    assert result["e2e_ms"] == result["latency_ms"]
    assert result["e2e_ms"] > result["ttft_ms"]
    assert abs(result["arrival_to_finish_ms"] - result["queue_ms"] - result["e2e_ms"]) < 0.002
    assert result["request_finished_monotonic_s"] > result["request_started_monotonic_s"]


def test_router_stream_route_forwards_upstream_sse_with_ttft_timing():
    class UpstreamHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            params = body.get("kv_transfer_params", {})
            if params.get("do_remote_decode"):
                payload = json.dumps({"choices": [{"text": ""}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return

            assert params.get("do_remote_prefill") is True
            assert body["stream"] is True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for event in (
                {"choices": [{"text": "Hello"}]},
                {"choices": [{"text": " world"}]},
                {"choices": [], "usage": {"prompt_tokens": 2, "completion_tokens": 2}},
            ):
                data = b"data: " + json.dumps(event).encode() + b"\n\n"
                self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
                self.wfile.flush()
                time.sleep(0.005)
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        def log_message(self, *_args):
            pass

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
    upstream_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    upstream_thread.start()
    endpoint = f"http://127.0.0.1:{upstream.server_port}/v1/completions"
    router_server = ThreadingHTTPServer(("127.0.0.1", 0), router.Handler)
    router_thread = threading.Thread(target=router_server.serve_forever, daemon=True)
    router_thread.start()
    try:
        with patch.object(router, "P_ENDPOINT", endpoint), patch.object(
            router, "D_ENDPOINT", endpoint
        ):
            conn = http.client.HTTPConnection("127.0.0.1", router_server.server_port, timeout=5)
            conn.request(
                "POST",
                "/v1/completions",
                body=json.dumps(
                    {
                        "request_id": "router-sse-test",
                        "prompt": [1, 2],
                        "max_tokens": 2,
                        "stream": True,
                    }
                ),
                headers={"Content-Type": "application/json"},
            )
            response = conn.getresponse()
            content_type = response.getheader("Content-Type")
            timing = response.getheader("Server-Timing", "")
            payload = response.read().decode()
            conn.close()
        assert response.status == 200
        assert content_type.startswith("text/event-stream")
        assert "prefill;dur=" in timing and "ttft;dur=" in timing
        assert "Hello" in payload and " world" in payload
    finally:
        router_server.shutdown()
        router_server.server_close()
        upstream.shutdown()
        upstream.server_close()
        router_thread.join(timeout=2)
        upstream_thread.join(timeout=2)


def test_abort_upstream_uses_vllm_internal_completion_id():
    response = MagicMock()
    response.__enter__.return_value.status = 200
    with patch.object(router, "UPSTREAM_API_KEY", "test-key"), patch.object(
        router.urllib.request, "urlopen", return_value=response
    ) as urlopen:
        assert router.abort_upstream(
            "http://decode.local:8200/v1/completions", "case-789"
        )

    request = urlopen.call_args.args[0]
    assert request.full_url == "http://decode.local:8200/inference/v1/abort_requests"
    assert json.loads(request.data) == {"request_ids": ["cmpl-case-789-0"]}
    assert request.get_header("Authorization") == "Bearer test-key"


def test_post_json_converts_remote_disconnect_to_upstream_error():
    with patch.object(
        router.urllib.request,
        "urlopen",
        side_effect=http.client.RemoteDisconnected("peer closed"),
    ):
        try:
            router.post_json("http://prefill.local/v1/completions", {})
        except router.UpstreamError as exc:
            assert exc.status == 502
            assert "peer closed" in exc.body
        else:
            raise AssertionError("remote disconnect should become UpstreamError")
