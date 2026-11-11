#!/usr/bin/env python3
"""Run the fixed workload against official SGLang chunk-level NIXL PD."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
SG = Path('/home/ubuntu/zjj/sglang-baseline')
PY = SG / 'venv/bin/python'
MODEL = '/home/ubuntu/zjj/models/Llama-3.1-8B-Instruct'


def wait_health(url, processes, timeout=600):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if any(p.poll() is not None for p in processes):
            raise RuntimeError(f'service exited while waiting for {url}')
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception:
            pass
        time.sleep(1)
    raise TimeoutError(url)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--result-dir', required=True)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--trace', action='store_true')
    ap.add_argument('--workload', default=str(ROOT/'workload.sweep-interleaved-8s3m4l.jsonl'))
    ap.add_argument('--chunk-size', type=int, default=2048)
    ap.add_argument('--system', choices=['sglang', 'vllm-baseline', 'vllm-full'], default='sglang')
    ap.add_argument('--local-sort', action='store_true',
                    help='Enable P-side local prompt-length admission windows.')
    ap.add_argument('--grouped-arrivals', action='store_true')
    ap.add_argument('--local-sort-window-ms', type=float, default=20.0)
    ap.add_argument('--group-size', type=int, default=5)
    ap.add_argument('--intra-group-gap-ms', type=float, default=0.0)
    ap.add_argument('--inter-group-gap-ms', type=float, default=500.0)
    args = ap.parse_args()
    import math
    if args.local_sort and args.system == 'sglang':
        ap.error('local sorting is only supported by the vLLM experiment')
    if args.repeats < 1 or args.group_size < 1:
        ap.error('repeats and group-size must be positive')
    if not math.isfinite(args.local_sort_window_ms) or args.local_sort_window_ms <= 0:
        ap.error('local-sort-window-ms must be finite and > 0')
    if any(not math.isfinite(x) or x < 0 for x in
           (args.intra_group_gap_ms, args.inter_group_gap_ms)):
        ap.error('arrival gaps must be finite and nonnegative')
    if (args.group_size - 1) * args.intra_group_gap_ms > args.inter_group_gap_ms:
        ap.error('intra-group arrivals must finish before next group starts')
    out = Path(args.result_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(PD_LOCAL_SORT='1' if args.local_sort else '0',
               PD_LOCAL_SORT_WINDOW_MS=str(args.local_sort_window_ms),
               PYTHONUNBUFFERED='1')
    toolkit = '/home/ubuntu/miniconda3/envs/cc/lib/python3.11/site-packages/nvidia/cu13'
    env.update(SGLANG_DISAGGREGATION_NIXL_BACKEND='UCX',
               UCX_TLS='tcp,cuda_copy', UCX_NET_DEVICES='lo',
               UCX_TCP_TX_SEG_SIZE='128k', UCX_TCP_RX_SEG_SIZE='128k',
               CUDA_HOME=toolkit, CUDA_ROOT=toolkit,
               NVCC_PREPEND_FLAGS='-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK',
               PYTHONPATH=str(SG),
               PATH=toolkit+'/bin:'+env.get('PATH', ''))
    if args.system == 'sglang':
        env['PD_CLIENT_SGLANG'] = '1'
    else:
        env.pop('PD_CLIENT_SGLANG', None)
    if args.trace:
        env.update(PD_SGLANG_TRACE=str(out/'trace.jsonl'), UCX_LOG_LEVEL='info',
                   UCX_PROTO_INFO='y')
    common = [str(PY), '-m', 'sglang.launch_server', '--model-path', MODEL,
              '--served-model-name', 'local', '--host', '127.0.0.1',
              '--context-length', '32768', '--mem-fraction-static', '0.90',
              '--dtype', 'bfloat16', '--disable-radix-cache',
              '--chunked-prefill-size', str(args.chunk_size),
              '--max-prefill-tokens', '2048', '--page-size', '16',
              '--disaggregation-transfer-backend', 'nixl',
              '--disaggregation-bootstrap-port', '8998',
              '--attention-backend', 'flashinfer', '--random-seed', '0']
    if args.trace:
        common += ['--disaggregation-enable-kv-checksum']
    commands = [common+['--disaggregation-mode', 'prefill', '--port', '8300'],
                common+['--disaggregation-mode', 'decode', '--port', '8400']]
    if args.system != 'sglang':
        env.update(RUN_DIR=str(out), MODE='baseline' if args.system == 'vllm-baseline' else 'preemptive',
                   P_PORT='8300', D_PORT='8400', LONG_REQUEST_PROTECT_PROGRESS='0.85',
                   LONG_REQUEST_MAX_PREEMPTIONS='3', LONG_REQUEST_MAX_PREEMPT_WAIT_MS='50',
                   CHUNK_SIZE_BLOCKS='8', PD_STREAM_BY_BATCH='1',
                   PD_AGGREGATE_SHORT_MEDIUM='1', PD_PRIORITY_PREEMPTION='1')
        commands = [['bash', '-c', '"$1"; exec tail -f /dev/null', 'bash', str(ROOT/'run_pd_pair.sh')]]
    processes, files = [], []
    manifest = dict(workload=args.workload,
                    workload_sha256=hashlib.sha256(Path(args.workload).read_bytes()).hexdigest(),
                    commands=commands, environment={k:env[k] for k in env if k.startswith(('SGLANG_', 'UCX_', 'CUDA_', 'NVCC_', 'PD_'))},
                    repeats=args.repeats, trace=args.trace,
                    chunk_size=args.chunk_size, system=args.system,
                    interarrival_ms=100 if not args.grouped_arrivals else None,
                    concurrency=15)
    manifest.update(local_sort=args.local_sort, grouped_arrivals=args.grouped_arrivals,
                    local_sort_window_ms=args.local_sort_window_ms,
                    group_size=args.group_size if args.grouped_arrivals else None,
                    intra_group_gap_ms=args.intra_group_gap_ms if args.grouped_arrivals else None,
                    inter_group_gap_ms=args.inter_group_gap_ms if args.grouped_arrivals else None)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
    try:
        for i, cmd in enumerate(commands):
            f = open(out/('launch.log' if args.system != 'sglang' else ('prefill.log' if i == 0 else 'decode.log')), 'w')
            files.append(f)
            process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT,
                                       env=dict(env, **({'CUDA_VISIBLE_DEVICES':str(i)} if args.system == 'sglang' else {})),
                                       start_new_session=True)
            processes.append(process)
        wait_health('http://127.0.0.1:8300/health', processes)
        wait_health('http://127.0.0.1:8400/health', processes)
        router = [str(PY), '-m', 'sglang_router.launch_router', '--pd-disaggregation', '--mini-lb',
                  '--prefill', 'http://127.0.0.1:8300', '8998',
                  '--decode', 'http://127.0.0.1:8400',
                  '--host', '127.0.0.1', '--port', '8500']
        if args.system != 'sglang':
            router = ['/home/ubuntu/miniconda3/envs/cc/bin/python', str(ROOT/'run_pd_router.py')]
            env.update(ROUTER_PORT='8500', P_ENDPOINT='http://127.0.0.1:8300/v1/completions',
                       D_ENDPOINT='http://127.0.0.1:8400/v1/completions')
        manifest['router_command'] = router
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
        f = open(out/'router.log', 'w')
        files.append(f)
        processes.append(subprocess.Popen(router, stdout=f, stderr=subprocess.STDOUT,
                                          env=env, start_new_session=True))
        wait_health('http://127.0.0.1:8500/health', processes)
        # Warm up the full network/attention path with the same input shapes.
        for i in range(args.repeats+1):
            name = 'warmup' if i == 0 else f'run-{i}'
            records = [json.loads(line) for line in Path(args.workload).read_text().splitlines() if line]
            for record in records:
                record['request_id'] += f'-{name}'
            replay = out/f'{name}-workload.jsonl'
            replay.write_text(''.join(json.dumps(record)+'\n' for record in records))
            cmd = ['/home/ubuntu/miniconda3/envs/cc/bin/python', str(ROOT/'run_pd_load.py'),
                   '--workload', str(replay), '--endpoint', 'http://127.0.0.1:8500/v1/completions',
                   '--output', str(out/f'{name}.csv'), '--concurrency', '15',
                   '--interarrival-ms', '100', '--timeout', '120', '--stream']
            if args.local_sort:
                cmd += ['--local-sort']
            if args.grouped_arrivals:
                cmd += ['--grouped-arrivals', '--group-size', str(args.group_size),
                        '--intra-group-gap-ms', str(args.intra_group_gap_ms),
                        '--inter-group-gap-ms', str(args.inter_group_gap_ms)]
            cmd += ['--run-metadata', str(out/f'{name}-metadata.json')]
            manifest.setdefault('client_commands', []).append(cmd)
            (out/'manifest.json').write_text(json.dumps(manifest, indent=2))
            print(f'Starting {name}', flush=True)
            with open(out/f'{name}-client.log', 'w') as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)
            rows = list(csv.DictReader(open(out/f'{name}.csv')))
            if any(int(row['status']) != 200 or int(row['output_tokens'] or 0) != 8 for row in rows):
                raise RuntimeError(f'{name}: incomplete requests; inspect client CSV')
            print((out/f'{name}-client.log').read_text(), flush=True)
            time.sleep(2)
    finally:
        for p in reversed(processes):
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for p in processes:
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
        for f in files:
            f.close()


if __name__ == '__main__':
    main()
