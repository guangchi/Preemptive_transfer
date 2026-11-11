"""Regression coverage for joining client, prefill, and decode traces."""
import csv
import tempfile
import pytest
from pathlib import Path

from collect_pd_trace import base_request_id, collect


def test_execution_collection_separates_prefill_decode_and_rejects_missing_gpu(tmp_path):
    rid = "pd-execution-test"
    requests = tmp_path / "requests.csv"
    requests.write_text(
        "request_id,status,ttft_ms,e2e_ms,arrival_to_finish_ms,request_finished_monotonic_s,latency_ms\n"
        f"{rid},200,30,50,55,10.050,50\n"
    )
    logs = []
    for side in ("p", "d"):
        path = tmp_path / f"{side}.log"
        path.write_text(
            f"PD execution step req={rid} side={side} step=0 scheduled_tokens=1 batch_requests=2 submitted_monotonic_s=10.000 elapsed_ms=10.0 monotonic_s=10.010\n"
            f"PD GPU forward req={rid} side={side} scheduled_tokens=1 batch_requests=2 gpu_forward_ms=2.5 forward_started_monotonic_s=10.001 forward_returned_monotonic_s=10.002 monotonic_s=10.009\n"
            f"PD request finished req={rid} status=FINISHED_LENGTH_CAPPED output_tokens=1 monotonic_s=10.011\n"
        )
        logs.append(path)
    phases, _ = collect(requests, *logs, tmp_path / "trace", require_execution=True)
    row = next(csv.DictReader(phases.open()))
    assert row["prefill_execution_ms"] == row["decode_execution_ms"] == "10.0"
    assert row["prefill_gpu_forward_ms"] == row["decode_gpu_forward_ms"] == "2.5"
    assert row["decode_service_ms"] == "11.0"
    assert row["prefill_shared_batch_steps"] == "1"
    assert len(list(csv.DictReader((tmp_path / "trace-execution.csv").open()))) == 6
    requests.write_text(requests.read_text().replace(",50,55,10.050", ",50,,10.050"))
    with pytest.raises(ValueError, match="missing/invalid arrival_to_finish_ms"):
        collect(requests, *logs, tmp_path / "missing-arrival", require_execution=True)
    requests.write_text(requests.read_text().replace(",50,,10.050", ",50,55,10.050"))
    logs[1].write_text("\n".join(line for line in logs[1].read_text().splitlines() if "PD execution step" not in line))
    with pytest.raises(ValueError, match="incomplete decode host/finish"):
        collect(requests, *logs, tmp_path / "bad", require_execution=True)


def test_base_request_id_removes_vllm_completion_suffix():
    assert base_request_id("cmpl-pd-77-00000-0-a1b2c3d4") == "pd-77-00000"
    assert base_request_id("pd-77-00000") == "pd-77-00000"


def test_collect_joins_chunk_events_and_request_phases():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        request_id = "pd-77-00000"
        vllm_id = "cmpl-pd-77-00000-0-a1b2c3d4"
        requests = root / "requests.csv"
        with requests.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "request_id", "category", "prompt_tokens", "max_tokens",
                    "status", "request_started_monotonic_s",
                    "router_received_monotonic_s", "queue_ms", "ttft_ms",
                    "router_ttft_ms", "latency_ms", "prefill_ms", "decode_ms",
                    "output_tokens", "response_sha256", "error",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "request_id": request_id,
                    "category": "short",
                    "prompt_tokens": 68,
                    "max_tokens": 8,
                    "status": 200,
                    "request_started_monotonic_s": 99.9,
                    "router_received_monotonic_s": 99.91,
                    "queue_ms": 1,
                    "ttft_ms": 200,
                    "router_ttft_ms": 190,
                    "latency_ms": 300,
                    "prefill_ms": 50,
                    "decode_ms": 10,
                    "output_tokens": 8,
                    "response_sha256": "abc",
                    "error": "",
                }
            )

        prefill = root / "prefill.log"
        prefill.write_text(
            "\n".join(
                [
                    f"stream chunk accepted req={vllm_id} seq=0 start=0 blocks=4 total=4 final=True monotonic_s=100.000000",
                    f"stream arbiter decision req={vllm_id} seq=0 tier=long decision=dispatch_long long_req={vllm_id} long_state={{'event': 'dispatch_long'}} long_preemptions=2 long_completed_blocks=10 long_total_blocks=100 long_progress=0.1 long_wait_ms=12.5 protection_reasons=wait monotonic_s=100.002000",
                    f"stream chunk start req={vllm_id} seq=0 block_start=0 blocks=4 final=True monotonic_s=100.003000",
                    f"stream chunk complete req={vllm_id} seq=0 success=True final=True monotonic_s=100.010000",
                ]
            ),
            encoding="utf-8",
        )
        decode = root / "decode.log"
        decode.write_text(
            "\n".join(
                [
                    f"stream chunk received req={vllm_id} seq=0 final=True received=[0] monotonic_s=100.012000",
                    f"stream receive complete req={vllm_id} final_seq=0 monotonic_s=100.013000",
                    f"PD first scheduled req={vllm_id} scheduled_tokens=1 monotonic_s=100.039000",
                    f"PD first model submit req={vllm_id} scheduled_tokens=1 monotonic_s=100.040000",
                ]
            ),
            encoding="utf-8",
        )

        phase_path, chunk_path = collect(
            requests, prefill, decode, root / "trace", require_complete=True
        )
        with phase_path.open(newline="", encoding="utf-8") as stream:
            phase = next(csv.DictReader(stream))
        with chunk_path.open(newline="", encoding="utf-8") as stream:
            chunk = next(csv.DictReader(stream))

        assert phase["request_id"] == request_id
        assert phase["p_final_chunk_ready_to_d_kv_complete_ms"] == "13.0"
        assert phase["d_kv_to_first_token_ms"] == "87.0"
        assert phase["d_final_sequence"] == "0"
        assert phase["d_completion_mode"] == "chunk_sequence"
        assert phase["d_first_model_submit_tokens"] == "1"
        assert phase["d_first_scheduled_tokens"] == "1"
        assert phase["d_kv_to_first_scheduled_ms"] == "26.0"
        assert phase["d_schedule_to_model_submit_ms"] == "1.0"
        assert phase["d_kv_to_first_model_submit_ms"] == "27.0"
        assert phase["model_submit_to_first_token_ms"] == "60.0"
        assert chunk["sequence"] == "0"
        assert chunk["received_monotonic_s"] == "100.012"
        assert chunk["receive_final"] == "True"
        assert chunk["long_preemptions"] == "2"
        assert chunk["long_wait_ms"] == "12.5"
        assert chunk["protection_reasons"] == "wait"
        assert chunk["queue_ms"] == "2.0"
        assert chunk["transfer_ms"] == "7.0"

        # An earlier schedule would falsely claim compute before KV was ready.
        decode.write_text(
            decode.read_text().replace("100.039000", "100.001000"),
            encoding="utf-8",
        )
        try:
            collect(requests, prefill, decode, root / "bad-order", require_complete=True)
        except ValueError as exc:
            assert "invalid D KV/schedule/submit event order" in str(exc)
        else:
            raise AssertionError("out-of-order D trace was not rejected")


def test_require_complete_rejects_success_without_worker_events():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        requests = root / "requests.csv"
        requests.write_text(
            "request_id,status\npd-missing-00000,200\n",
            encoding="utf-8",
        )
        prefill = root / "prefill.log"
        decode = root / "decode.log"
        prefill.write_text("", encoding="utf-8")
        decode.write_text("", encoding="utf-8")
        try:
            collect(
                requests,
                prefill,
                decode,
                root / "trace",
                require_complete=True,
            )
        except ValueError as exc:
            assert "P final chunk accepted" in str(exc)
            assert "D KV receive complete" in str(exc)
        else:
            raise AssertionError("missing lifecycle trace was not rejected")
