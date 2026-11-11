#!/usr/bin/env python3
"""Audit fixed-code trials without pooling the intermediate-version sample."""
import csv
import json
from pathlib import Path
import re
import statistics


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    root = Path('results/long-output-fix-20261106')
    raw = json.loads((root / 'retests/raw-confirmation.json').read_text())
    partial = next(r for r in raw if r['repeat'] == 1 and r['setting'] == 'c4-p5-x0.95-w50')
    trials = [r for r in raw if r is not partial]
    supplement = root / 'supplement/repeat-1/c4-p5-x0.95-w50/preemptive-c4-p5-x0.95-w50'
    total = json.loads((supplement / 'workload-total-times.json').read_text())['runs'][0]
    trials.append(dict(repeat=1, setting=partial['setting'], run_dir=str(supplement.resolve()),
                       responses=rows(supplement / 'requests.csv'), metrics=rows(supplement / 'trace-request-metrics.csv'),
                       workload_makespan_s=total['workload_makespan_s'], group_wall_s=total['group_wall_s']))
    reference = {}
    for line in Path('results/determinism-investigation-20261106/responses.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['batch_invariant']:
            reference.setdefault(row['request_id'], set()).add(tuple(json.loads(row['response_token_ids'])))
    assert len(reference) == 15 and all(len(v) == 1 for v in reference.values())
    audits, pairs, preemptions = [], [], []
    for trial in trials:
        directory = Path(trial['run_dir'])
        response = {r['request_id']: r for r in trial['responses']}
        baseline = next(r for r in trials if r['repeat'] == trial['repeat'] and r['setting'] == 'baseline')
        br = {r['request_id']: r for r in baseline['responses']}
        match = set(response) == set(br) == set(reference) and all(
            r['status'] == br[rid]['status'] == '200'
            and tuple(json.loads(r['response_token_ids'])) in reference[rid]
            and r['response_token_ids'] == br[rid]['response_token_ids']
            and r['response_sha256'] == br[rid]['response_sha256'] for rid, r in response.items())
        chunks = rows(directory / 'trace-chunks.csv')
        sequence_errors = []
        for rid in response:
            group = [c for c in chunks if c['request_id'] == rid]
            if not group:
                continue
            ordered = sorted(group, key=lambda c: float(c['complete_monotonic_s']))
            expected = list(range(len(group)))
            if [int(c['sequence']) for c in ordered] != expected or any(c['success'] != 'True' for c in group):
                sequence_errors.append(rid)
        sync = [float(v) for v in re.findall(r'device-synchronized chunks=\d+ elapsed_ms=([\d.]+)', (directory / 'prefill.log').read_text())]
        audits.append(dict(repeat=trial['repeat'], setting=trial['setting'], run_dir=str(directory),
                           requests=len(response), outputs_match=match, chunk_count=len(chunks),
                           completion_sequence_errors=sequence_errors, cuda_sync_calls=len(sync),
                           cuda_sync_elapsed_ms_sum=sum(sync), cuda_sync_elapsed_ms_max=max(sync, default=0),
                           workload_makespan_s=trial['workload_makespan_s']))
        if trial['setting'] == 'baseline':
            continue
        bm = {m['request_id']: m for m in baseline['metrics']}
        for m in trial['metrics']:
            if m['category'] == 'long':
                a, b = float(bm[m['request_id']]['e2e_ms']), float(m['e2e_ms'])
                pairs.append(dict(repeat=trial['repeat'], setting=trial['setting'], request_id=m['request_id'],
                                  baseline_e2e_ms=a, candidate_e2e_ms=b, ratio=b/a,
                                  baseline_run_dir=baseline['run_dir'], candidate_run_dir=trial['run_dir']))
        bypass = [c for c in chunks if c['decision'] == 'preempt_long']
        protection = {}
        for c in sorted(chunks, key=lambda c: float(c['dispatch_monotonic_s'])):
            if c['decision'] == 'protected_resume':
                protection.setdefault(c['request_id'], {k: c[k] for k in ['sequence', 'long_progress', 'long_wait_ms', 'protection_reasons']})
        preemptions.append(dict(repeat=trial['repeat'], setting=trial['setting'], count=len(bypass),
                                first_protection=protection))
    assert all(a['outputs_match'] and not a['completion_sequence_errors'] for a in audits), audits
    summary = []
    for setting in ['baseline', 'c4-p3-x0.85-w50', 'c4-p5-x0.95-w50', 'c4-p5-x0.95-w100']:
        selected = [r for r in trials if r['setting'] == setting]
        long_pairs = [r for r in pairs if r['setting'] == setting]
        summary.append(dict(setting=setting, repeats=len(selected),
            mean_makespan_s=statistics.mean(r['workload_makespan_s'] for r in selected),
            min_makespan_s=min(r['workload_makespan_s'] for r in selected),
            max_makespan_s=max(r['workload_makespan_s'] for r in selected),
            median_score_ms=statistics.median((statistics.mean(float(m['ttft_ms']) for m in r['metrics'] if m['category']=='short') * statistics.mean(float(m['ttft_ms']) for m in r['metrics'] if m['category']=='medium'))**.5 for r in selected),
            max_long_e2e_ms=max(float(m['e2e_ms']) for r in selected for m in r['metrics'] if m['category']=='long'),
            max_paired_long_e2e_ratio=max((p['ratio'] for p in long_pairs), default=None),
            categories={cat: {f: statistics.mean(float(m[f]) for r in selected for m in r['metrics'] if m['category']==cat) for f in ['ttft_ms','e2e_ms','prefill_execution_ms','decode_execution_ms']} for cat in ['short','medium','long']}))
    excluded = dict(repeat=1, setting=partial['setting'], run_dir=partial['run_dir'], reason='device sync only; started before handshake slot and summary fixes')
    timing = json.loads((root/'retests/confirmation-results.json').read_text())
    result = dict(summary=summary, audits=audits, excluded=[excluded], supplement_pairing='Supplement uses original repeat-1 baseline, not an adjacent new baseline; timing drift remains possible.',
                  original_runner_wall_s=timing['total_confirmation_wall_s'],
                  supplement_group_wall_s=total['group_wall_s'],
                  selected_group_wall_s_sum=sum(json.loads((Path(t['run_dir'])/'experiment-timing.json').read_text())['phase_durations_s']['group_wall_s'] for t in trials),
                  correct_requests=sum(a['requests'] for a in audits), long_pairs=pairs, preemptions=preemptions,
                  eligible=[s['setting'] for s in summary if s['max_paired_long_e2e_ratio'] is not None and s['max_paired_long_e2e_ratio']<=1.10])
    (root/'fixed-results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n')
    for filename, records in [('paired-long-e2e.csv',pairs), ('confirmation-summary.csv', [{k:v for k,v in s.items() if k!='categories'} for s in summary])]:
        with (root/filename).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    (root/'preemption-summary.json').write_text(json.dumps(preemptions, indent=2)+'\n')
    lines = ['# long 输出差异修复与候选参数复测', '',
        '固定原 15 请求（8 short、3 medium、4 long），prompt/seed/交错顺序不变，100ms 到达间隔，并发 15，chunk=4，prefix cache=false，batch-invariant=1，top-logprobs=false。', '',
        '## 根因和修复', '',
        '旧默认第二轮 01-long 的握手期间，seq 0～24 未占住传输槽，seq 25～156 先完成。D 在 monotonic=20375.180449 收到提前发送的 STREAM_DONE 后开始生成，seq 0～24 的实际 WRITE 到 20388.002192 才开始，20388.542013 才全部完成。这是缺失 KV 被当成完成的传输竞态；开启 batch-invariant 无法修复。旧失败输出和日志保持原样。', '',
        '修复位于 push_worker.py：握手至 deferred WRITE 启动/失败期间保留单槽；请求全部 chunk 的 WRITE/notification 成功后才允许 STREAM_DONE，summary 使用保存的 final 序号；入队前使用 torch.cuda.synchronize() 覆盖可能运行在非当前 stream 的 KV 写入，并记录同步耗时。设备同步是保守的可见性保护，现有日志直接证明的是握手/summary 竞态，未单独隔离三项改动的性能贡献。', '',
        '## 正确性与样本版本', '',
        f'完整修复版本共 {len(audits)} 组、{result["correct_requests"]} 请求，HTTP、token ID、文本 hash 与配对 baseline/直接推理参考全部一致。所有抢占组逐请求 chunk 成功完成序号均为连续递增。当前 pytest 为 63 passed、14 条 Torch 弃用 warning。', '',
        '原 retests/repeat-1/c4-p5-x0.95-w50 仅含 device sync，保留为中间证据并排除。supplement/repeat-1 的完整修复重跑替代该样本，仍与原 repeat-1 baseline 配对；补跑非相邻执行，时序漂移是局限。retests/confirmation-results.json 是原 runner 的未校正汇总，最终版本筛选以本目录 fixed-results.json 为准。', '',
        '## 参数与长请求尾部', '',
        '| 配置（进度/次数/等待） | short TTFT mean(ms) | medium TTFT mean(ms) | 整批 mean[min,max](s) | long E2E 最大值(ms) | 最差配对 long 增幅 |',
        '| --- | ---: | ---: | --- | ---: | ---: |']
    for s in summary:
        ratio = '—' if s['max_paired_long_e2e_ratio'] is None else f'{(s["max_paired_long_e2e_ratio"]-1)*100:+.2f}%'
        lines.append(f'| {s["setting"]} | {s["categories"]["short"]["ttft_ms"]:.3f} | {s["categories"]["medium"]["ttft_ms"]:.3f} | {s["mean_makespan_s"]:.3f} [{s["min_makespan_s"]:.3f},{s["max_makespan_s"]:.3f}] | {s["max_long_e2e_ms"]:.3f} | {ratio} |')
    lines += ['', '95%/5次/50ms 的 short/medium TTFT 几何均值分数中位数最低（8927.337ms），且在抢占候选中整批平均最快；baseline 整批均值仍最快。没有抢占候选通过每条 long 相对同轮 baseline 的 E2E≤+10% 约束。10% 是既有分析约束，不是用户 SLA 或硬保证；保留默认 85%/3次/50ms。每配置仅 12 条 long 观测，报告观测最大值，不宣称稳定 P95/P99 或参数全局最优。原确认 runner 因没有可行推荐退出 1，errors 为空，不表示 HTTP 或 trace 实验失败。', '',
        '握手保留槽位消除了原先错误的越序并行；分块有更多 WRITE、轮询、通知与仲裁开销，设备同步还可能降低计算与传输重叠。short 的提前服务可以改善类均值 TTFT，但不增加总带宽，不能保证最后 long 和整批完成更快。三项保护只限制插队机会，不能补偿上述开销；50ms 也不是 long 总延迟上限。该解释符合轨迹，但本轮未运行逐项消融，不能把差值全部归因于某一项。', '',
        '## 逐组过程与同步成本', '',
        '| 轮次 | 配置 | 抢占 chunks | 同步次数 | 同步累计/最大(ms) |', '| --- | --- | ---: | ---: | --- |']
    for a in sorted(audits, key=lambda a:(a['repeat'],a['setting'])):
        count = next((p['count'] for p in preemptions if p['repeat']==a['repeat'] and p['setting']==a['setting']),0)
        lines.append(f'| {a["repeat"]} | {a["setting"]} | {count} | {a["cuda_sync_calls"]} | {a["cuda_sync_elapsed_ms_sum"]:.3f}/{a["cuda_sync_elapsed_ms_max"]:.3f} |')
    lines += ['', '同步累计是 host 调用阻塞耗时，包含等待设备上已提交计算，不是额外 GPU kernel 时间，也不能直接加到 E2E 解释因果。', '',
        f'原 12 组 runner 墙钟 {result["original_runner_wall_s"]:.3f}s（含排除样本），补跑从启动至清理 {result["supplement_group_wall_s"]:.3f}s；最终选中的 12 组各组启动至清理墙钟合计 {result["selected_group_wall_s_sum"]:.3f}s。它们与整批请求 makespan 定义不同，两个阶段之间的分析时间未计入合计。', '',
        'paired-long-e2e.csv 保存每条 long 的 baseline/E2E/比值；fixed-results.json 保存逐组输出/顺序审计、四项请求指标类均值及同步开销；preemption-summary.json 保存抢占次数和首次保护。每个候选父目录的 EXPERIMENT_REPORT.md、preemptions.csv、protection-decisions.csv、experiment-timeline.csv 和 timeline.html 记录真实过程。', '',
        '本次没有复扫原 27 点，也未证明任意 workload/多机/多 TP 的正确性。已复现的提前 summary 缺陷已修复且本固定负载重复验收通过；稳健选参和 long 尾部成本仍作为性能问题保留。', '']
    (root/'FIX_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
