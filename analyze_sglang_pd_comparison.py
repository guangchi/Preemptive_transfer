#!/usr/bin/env python3
"""Summarize fixed-workload PD comparison without mixing warmup and measurement."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('result_dir')
    args = ap.parse_args()
    root = Path(args.result_dir)
    summary = {}
    outputs = {}
    for system in ['vllm-baseline', 'vllm-full', 'sglang']:
        runs, per_request = [], {}
        for p in sorted((root/system).glob('run-*.csv')):
            rows = list(csv.DictReader(p.open()))
            for row in rows:
                row['request_id'] = row['request_id'].removesuffix('-'+p.stem)
            assert len(rows) == 15
            assert all(int(x['status']) == 200 and int(x['output_tokens']) == 8 for x in rows)
            makespan = max(float(x['request_finished_monotonic_s']) for x in rows)-min(float(x['submitted_at_monotonic_s']) for x in rows)
            data = dict(run=p.stem, makespan_s=makespan)
            for cat in ['short', 'medium', 'long']:
                selected = [x for x in rows if x['category'] == cat]
                data[cat] = dict(ttft_ms=statistics.mean(float(x['ttft_ms']) for x in selected),
                                 e2e_ms=statistics.mean(float(x['latency_ms']) for x in selected))
            for row in rows:
                per_request.setdefault(row['request_id'], []).append(row)
            runs.append(data)
        if not runs:
            continue
        summary[system] = dict(runs=runs,
                               makespan_mean_s=statistics.mean(x['makespan_s'] for x in runs),
                               makespan_stdev_s=statistics.stdev(x['makespan_s'] for x in runs) if len(runs)>1 else 0,
                               categories={cat:{metric:statistics.mean(x[cat][metric] for x in runs)
                                                for metric in ['ttft_ms','e2e_ms']}
                                           for cat in ['short', 'medium', 'long']},
                               output_stability={rid:len({x['response_sha256'] for x in values})==1
                                                 for rid,values in per_request.items()},
                               long_requests={rid:{'e2e_ms':statistics.mean(float(x['latency_ms']) for x in rows),
                                                   'ttft_ms':statistics.mean(float(x['ttft_ms']) for x in rows)}
                                              for rid,rows in per_request.items() if rows[0]['category']=='long'})
        outputs[system] = {rid:rows[0]['response_sha256'] for rid,rows in per_request.items()}
    (root/'comparison.json').write_text(json.dumps(summary, indent=2))
    lines = ['# SGLang chunk 级 NIXL/UCX/TCP 对比', '',
             '同一固定测试集：15 请求（8 short、3 medium、4 long），输出 8 tokens，到达间隔 100ms，并发 15。每套系统先完整预热一次，再连续测量 3 次。', '',
             '| 系统 | 总时间均值 ± 标准差 s | 3 次总时间 s | short TTFT ms | medium TTFT ms | long TTFT ms |',
             '| --- | ---: | --- | ---: | ---: | ---: |']
    for name,s in summary.items():
        c=s['categories']
        times=', '.join(f"{r['makespan_s']:.6f}" for r in s['runs'])
        lines.append(f"| {name} | {s['makespan_mean_s']:.6f} ± {s['makespan_stdev_s']:.6f} | {times} | {c['short']['ttft_ms']:.3f} | {c['medium']['ttft_ms']:.3f} | {c['long']['ttft_ms']:.3f} |")
    lines += ['', '| 系统 | short E2E ms | medium E2E ms | long E2E ms |', '| --- | ---: | ---: | ---: |']
    for name,s in summary.items():
        c=s['categories']
        lines.append(f"| {name} | {c['short']['e2e_ms']:.3f} | {c['medium']['e2e_ms']:.3f} | {c['long']['e2e_ms']:.3f} |")
    lines += ['', '| long 请求 | '+ ' | '.join(summary)+' |', '| --- | '+' | '.join('---:' for _ in summary)+' |']
    if summary:
        for rid in next(iter(summary.values()))['long_requests']:
            lines.append('| '+rid+' | '+' | '.join(f"{s['long_requests'][rid]['e2e_ms']:.3f}" for s in summary.values())+' |')
    lines += ['', '## 结果与边界', '']
    if len(summary)==3:
        full=summary['vllm-full']['makespan_mean_s']
        for other in ['vllm-baseline','sglang']:
            baseline=summary[other]['makespan_mean_s']
            lines.append(f'- 完整方案相对 {other} 总时间变化：{(full/baseline-1)*100:+.2f}%。')
    for system,s in summary.items():
        stable=sum(s['output_stability'].values())
        lines.append(f'- {system}：{len(s["runs"])*15} 条正式请求全部成功且均输出 8 tokens；{stable}/15 个输入在重复测量中输出文本一致。')
    if 'vllm-baseline' in outputs and 'vllm-full' in outputs:
        matches=sum(outputs['vllm-baseline'][rid]==outputs['vllm-full'][rid] for rid in outputs['vllm-baseline'])
        lines.append(f'- vLLM baseline 与完整方案输出文本一致：{matches}/15。')
    if 'sglang' in outputs and 'vllm-baseline' in outputs:
        matches=sum(outputs['vllm-baseline'][rid]==outputs['sglang'][rid] for rid in outputs['vllm-baseline'])
        lines.append(f'- SGLang 与 vLLM baseline 输出文本一致：{matches}/15。跨框架内核及采样实现可产生差异；此检查本身不能替代 KV 数据完整性校验。')
    lines += ['', '- 完整方案使用 85% / 3 次 / 50ms、long C8、short/medium 每 batch 聚合及提前发送。',
              '- 三组都显式固定 UCX tcp,cuda_copy、loopback 网卡、TX/RX 128KB，模型 BF16、TP=PP=1、context 32768、显存比例 0.90、关闭 prefix cache。',
              '- SGLang chunked prefill=2048 tokens；vLLM 计算 token budget=2048。计算 chunk 大小与 long 的 C8 传输粒度含义不同。',
              '- SGLang 是外部系统级 baseline；注意力内核、运行时调度、NIXL worker 数量和路由器等存在差异，不能将全部差值归因于 chunk 传输算法。',
              '- 原始逐请求记录、启动参数和服务日志位于各系统子目录；warmup 不计入正式结果。', '']
    notes = root/'EXPERIMENT_NOTES.md'
    if notes.exists():
        lines += ['', notes.read_text()]
    (root/'COMPARISON_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
