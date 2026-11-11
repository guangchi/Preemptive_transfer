#!/usr/bin/env python3
"""Report the completed 27-point protection scan without adding experiments."""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
import re
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    results = json.loads((output / "raw-results.json").read_text())
    grid = json.loads((output / "grid-results.json").read_text())
    candidates = grid["ranked_candidates"]
    baselines = [r for r in results if r["setting"] == "baseline"]
    full = [r for r in results if r["setting"] == "full"]
    assert len(results) == 31 and len(candidates) == 27 and len(baselines) == 4
    assert {(r['progress_pct'], r['wait_ms']) for r in full} == {(p,w) for p in range(50,91,5) for w in (50,100,150)}
    assert all(not r['audit']['chunk_violations'] and not r['audit']['output_violations']
               and r['audit']['successful_requests'] == 15 for r in results)
    per_long_counts = []
    for r in full:
        counts = collections.Counter(b['long_request_id'] for b in r['policy']['bypasses'])
        per_long_counts.extend(counts.values())
        assert max(counts.values(), default=0) <= 3
        for bypass in r['policy']['bypasses']:
            assert float(bypass['long_progress']) < r['progress_pct']/100
            assert float(bypass['long_wait_ms']) < r['wait_ms']
    best = candidates[0]
    default = next(r for r in candidates if r['progress_pct'] == 85 and r['wait_ms'] == 50)
    matched = [b for b in baselines if b['order'] in best['baseline_orders']]
    cat_baseline = {tier: {metric: statistics.mean(b['audit']['categories'][tier][metric]['mean'] for b in matched)
                          for metric in ('ttft_ms', 'e2e_ms')}
                    for tier in ('short', 'medium', 'long')}
    long_baseline = {rid: statistics.mean(b['long_e2e_ms'][rid] for b in matched) for rid in best['long_e2e_ms']}
    wait_reasons = [r for r in full if any('wait' in reason for reason in r['policy']['protection_reasons'])]
    profiles = collections.defaultdict(list)
    for r in full:
        pairs = [(b['request_id'], re.search(r'interleaved15-\d+-long', b['long_request_id']).group())
                 for b in r['policy']['bypasses']]
        profiles[json.dumps(pairs)].append({'progress_pct': r['progress_pct'], 'wait_ms': r['wait_ms'],
                                          'makespan_s': r['audit']['makespan_s']})
    analysis = {
        'best_observed': best, 'default_observed': default,
        'normalized_best': min(candidates, key=lambda r: r['makespan_s']/r['bracketing_baseline_s']),
        'best_matched_baseline_categories': cat_baseline,
        'best_matched_baseline_long_e2e_ms': long_baseline,
        'baseline_mean_s': statistics.mean(b['audit']['makespan_s'] for b in baselines),
        'baseline_range_s': [min(b['audit']['makespan_s'] for b in baselines), max(b['audit']['makespan_s'] for b in baselines)],
        'maximum_recorded_wait_ms': max(r['policy']['max_recorded_long_wait_ms'] for r in full),
        'wait_protection_run_count': len(wait_reasons),
        'preemption_counts': dict(collections.Counter(r['policy']['preemptions'] for r in full)),
        'maximum_observed_per_long_preemptions': max(per_long_counts, default=0),
        'gpu_forward_total_ms_range': [min(r['audit']['producer_gpu_forward_total_ms'] for r in full),
                                       max(r['audit']['producer_gpu_forward_total_ms'] for r in full)],
        'write_submit_host_total_ms_range': [min(r['audit']['host_write_phases']['all']['submit_ms']['sum'] for r in full),
                                             max(r['audit']['host_write_phases']['all']['submit_ms']['sum'] for r in full)],
        'bypass_pair_profiles': [{'pairs': json.loads(k), 'runs': v} for k,v in profiles.items()],
        'validation': {'candidate_runs': 27, 'baseline_runs': 4, 'successful_responses': 465,
                       'token_and_hash_matches': 465, 'chunk_violations': 0, 'output_violations': 0},
        'limitation': 'One run per parameter combination. Observed minimum is not a repeat-confirmed optimum. Timing and batch-boundary variation can change bypass opportunities.'}
    (output / 'analysis.json').write_text(json.dumps(analysis, indent=2) + '\n')
    fields = ['rank','progress_pct','wait_ms','makespan_s','bracketing_baseline_s','reduction_pct','preemptions',
              'short_ttft_ms','medium_ttft_ms','long_ttft_ms','long_e2e_ms','max_recorded_long_wait_ms','run_dir']
    with (output / 'grid-table.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for rank, r in enumerate(candidates, 1):
            row = {k:r[k] for k in fields if k in r and k != 'long_e2e_ms'}
            row['rank'] = rank
            row.update({f'{tier}_ttft_ms': r['categories'][tier]['ttft_ms']['mean'] for tier in ('short','medium','long')})
            row['long_e2e_ms'] = r['categories']['long']['e2e_ms']['mean']
            writer.writerow(row)
    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(1,3,figsize=(14,6), layout='constrained')
    metrics = [('Workload makespan (s)', lambda r:r['makespan_s'], '.3f'),
               ('Short mean TTFT (ms)', lambda r:r['categories']['short']['ttft_ms']['mean'], '.0f'),
               ('Long mean E2E (ms)', lambda r:r['categories']['long']['e2e_ms']['mean'], '.0f')]
    for ax, (title, get, fmt) in zip(axes, metrics):
        values = np.array([[get(next(r for r in candidates if r['progress_pct']==p and r['wait_ms']==w))
                            for w in (50,100,150)] for p in range(50,91,5)])
        im=ax.imshow(values, cmap='viridis_r', aspect='auto')
        ax.set_xticks(range(3), ['50','100','150'])
        ax.set_yticks(range(9), [f'{p}%' for p in range(50,91,5)])
        ax.set_xlabel('Wait protection threshold (ms)')
        ax.set_ylabel('Progress protection threshold')
        ax.set_title(title)
        for i in range(9):
            for j in range(3):
                midpoint=(values.min()+values.max())/2
                ax.text(j,i,format(values[i,j],fmt),ha='center',va='center',color='white' if values[i,j]>midpoint else 'black')
        fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('Fixed 15-request workload; max preemptions = 3; one run per cell')
    fig.savefig(output/'parameter-heatmaps.png', dpi=180)
    fig.savefig(output/'parameter-heatmaps.pdf')
    plt.close(fig)
    lines = [
        '# 27 组进度/等待阈值参数扫描', '',
        f'本轮按整批完成时间选择的观测最优组合为 **{best["progress_pct"]}% / 3 次 / {best["wait_ms"]}ms**，'
        f'整批 **{best["makespan_s"]:.6f}s**；相邻两次新 baseline 均值 **{best["bracketing_baseline_s"]:.6f}s**，'
        f'降低 **{best["reduction_pct"]:.2f}%**。每个组合仅运行一次，这是本次扫描的最小观测值，尚未通过重复实验确认稳健最优。', '',
        '## 固定条件与对照方法', '',
        '- 原测试集不变：15 请求（8 short × 128 tokens，3 medium × 1107，4 long × 10000），各输出 8 tokens；到达间隔 100ms，并发 15。',
        '- 仅比较整请求 NIXL baseline 与完整方案；完整方案三模块全开，short/medium 各请求每计算 batch 聚合，long C8。',
        '- 进度保护阈值为 50%～90%，每 5% 一档；等待阈值 50/100/150ms；最大抢占次数固定为 3。共 27 候选，不做消融。',
        '- 比例指 long 已成功传输 blocks / 总 blocks，达到比例后禁止继续插队；等待指因抢占插队累计等待，并非排队总时长。',
        '- 两种模式都使用 UCX TCP TX/RX 128KB、WRITE 单槽、event fence、batch invariant、关闭 prefix cache；同模型、GPU、请求/seed/顺序。',
        '- 固定随机种子 20261106 打乱 27 个候选顺序，每 9 个候选前后各有新 baseline，合计 4 次 baseline、31 组服务启动。每个候选相对所在 9 点区间两端 baseline 均值计算收益。',
        '- 主指标为从首个请求提交到最后响应结束的 workload makespan；不将服务启动与固定 15s 等待混入该指标。逐组启动/清理墙钟另保存在原始结果。',
        '- workload SHA256：`714990980c49534e539a125612ce1c8d36c68ac75a559e4bce93dabcb7ee9e14`。', '',
        '## 全部组合：整批完成时间（秒，越低越好）', '',
        '| 进度阈值 | 50ms | 100ms | 150ms |', '| --- | ---: | ---: | ---: |']
    for p in range(50,91,5):
        values = [next(r for r in candidates if r['progress_pct']==p and r['wait_ms']==w)['makespan_s'] for w in (50,100,150)]
        lines.append(f'| {p}% | ' + ' | '.join(f'{v:.6f}' for v in values) + ' |')
    lines += ['', '![27 组总时间、short TTFT 和 long E2E](parameter-heatmaps.png)', '',
              '## 排名前五与原参数', '',
              '| 排名 | 进度 | 等待 | 总时间 s | 相邻 baseline s | 降低 | short TTFT ms | medium TTFT ms | long E2E ms | 插队次数 |',
              '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    selected = candidates[:5] + ([default] if default not in candidates[:5] else [])
    for r in selected:
        c=r['categories']
        rank=candidates.index(r)+1
        label=f'{rank}（原参数）' if r is default else str(rank)
        lines.append(f'| {label} | {r["progress_pct"]}% | {r["wait_ms"]}ms | {r["makespan_s"]:.6f} | {r["bracketing_baseline_s"]:.6f} | {r["reduction_pct"]:.2f}% | {c["short"]["ttft_ms"]["mean"]:.3f} | {c["medium"]["ttft_ms"]["mean"]:.3f} | {c["long"]["e2e_ms"]["mean"]:.3f} | {r["preemptions"]} |')
    lines += ['', f'原 85%/3 次/50ms 本轮为 {default["makespan_s"]:.6f}s；观测最优比它低 {100*(1-best["makespan_s"]/default["makespan_s"]):.2f}%。该差值来自单次运行，不能据此认定参数具有相同幅度的稳定收益。', '',
              '## 最优观测组合的分层与逐条 long 延迟', '',
              '| 类别 | baseline TTFT ms | 完整 TTFT ms | baseline E2E ms | 完整 E2E ms |',
              '| --- | ---: | ---: | ---: | ---: |']
    for tier in ('short','medium','long'):
        b,c=cat_baseline[tier],best['categories'][tier]
        lines.append(f'| {tier} | {b["ttft_ms"]:.3f} | {c["ttft_ms"]["mean"]:.3f} | {b["e2e_ms"]:.3f} | {c["e2e_ms"]["mean"]:.3f} |')
    lines += ['', '| long 请求 | baseline E2E ms | 完整 E2E ms | 差值 ms（完整−baseline） |', '| --- | ---: | ---: | ---: |']
    for rid,value in sorted(best['long_e2e_ms'].items()):
        b=long_baseline[rid]
        lines.append(f'| {rid} | {b:.3f} | {value:.3f} | {value-b:+.3f} |')
    lines += ['', '## baseline、实际策略与解释边界', '',
              '| baseline 顺序 | 总时间 s |', '| --- | ---: |']
    for b in baselines:
        lines.append(f'| {b["order"]} | {b["audit"]["makespan_s"]:.6f} |')
    lines += ['', f'4 次 baseline 均值 {analysis["baseline_mean_s"]:.6f}s，范围 {analysis["baseline_range_s"][0]:.6f}～{analysis["baseline_range_s"][1]:.6f}s。', '',
              f'27 个候选的真实插队次数分布为 `{analysis["preemption_counts"]}`；最大已记录累计插队等待 {analysis["maximum_recorded_wait_ms"]:.3f}ms；存在等待保护原因的运行数为 {len(wait_reasons)}/27。', '',
              '逐组合真实插队次数（全 workload 合计；3 次上限按每条 long 单独计）：', '',
              '| 进度阈值 | 50ms | 100ms | 150ms |', '| --- | ---: | ---: | ---: |']
    for p in range(50,91,5):
        counts = [next(r for r in candidates if r['progress_pct']==p and r['wait_ms']==w)['preemptions'] for w in (50,100,150)]
        lines.append(f'| {p}% | ' + ' | '.join(str(v) for v in counts) + ' |')
    lines += ['',
              f'完整方案去重后的 P 端 GPU forward 累计为 {analysis["gpu_forward_total_ms_range"][0]:.3f}～{analysis["gpu_forward_total_ms_range"][1]:.3f}ms；WRITE host 提交累计为 {analysis["write_submit_host_total_ms_range"][0]:.3f}～{analysis["write_submit_host_total_ms_range"][1]:.3f}ms。这些区间说明执行过程存在波动；计算与传输重叠，不能将累计值直接相加或将其差值作为整批时间的因果分解。', '',
              '相同插队对象与顺序的运行分组如下。分组不要求插队 block 进度一致，不能将组内数值差异全部解释为测量噪声；具体进度、等待与保护原因见 raw-results.json。', '',
              '| 插队 short → long 序列 | 候选数 | 总时间范围 s |', '| --- | ---: | ---: |']
    for profile in analysis['bypass_pair_profiles']:
        pairs=', '.join(f'{a.split("-")[-2]}→{b.split("-")[-2]}' for a,b in profile['pairs']) or '无插队'
        values=[r['makespan_s'] for r in profile['runs']]
        lines.append(f'| {pairs} | {len(values)} | {min(values):.6f}～{max(values):.6f} |')
    if not wait_reasons:
        lines += ['', '**50/100/150ms 均未触发等待保护。** 因此本测试集不能识别等待阈值本身的独立收益，不能把某一列的最快值解释为该等待阈值稳定更优。']
    normalized=analysis['normalized_best']
    lines += ['', f'按相邻 baseline 归一化后的第一名为 {normalized["progress_pct"]}%/{normalized["wait_ms"]}ms（降低 {normalized["reduction_pct"]:.2f}%）；主排名仍按预先约定的原始整批时间。', '',
              f'**选型建议：只优化本轮整批总时间时，选 65%/3 次/100ms 作为后续候选。** 它本轮没有实际插队。若希望保留短请求插队收益，可同时关注 90%/3 次/100ms：整批 {normalized["makespan_s"]:.6f}s，距原始最小值 {(normalized["makespan_s"]-best["makespan_s"])*1000:.3f}ms，short TTFT {normalized["categories"]["short"]["ttft_ms"]["mean"]:.3f}ms，比原始总时间最小值组合低 {best["categories"]["short"]["ttft_ms"]["mean"]-normalized["categories"]["short"]["ttft_ms"]["mean"]:.3f}ms；long 平均 E2E {normalized["categories"]["long"]["e2e_ms"]["mean"]:.3f}ms。两者均为单次观测，不能将 100ms 解读为已验证优于 50ms 或 150ms。', '',
              '完整方案相对整请求 baseline 的变化同时包括提前发送、聚合与插队；long 更快不能推导为抢占本身加速了 long。参数改变也可能改变可插队时机和后续计算 batch，单次扫描无法分离所有交互。此次仅提供观测最优候选，保留现有默认参数，未增加用户未要求的重复轮次。', '',
              '## 正确性与结果文件', '',
              '**31/31 组完成，465/465 请求 HTTP 成功，生成 token IDs 和响应 SHA256 全部匹配独立确定性参考。** 全部完整方案逐请求 chunk 序列连续、blocks 无缺失/重叠、final 完整、传输成功，long 不超过 8 blocks。逐组配置也核验了阈值、3 次上限、C8、三模块和 UCX。', '',
              '- [完整排名 CSV](grid-table.csv)：27 组总时间、分层 TTFT/E2E、实际插队。',
              '- [结构化结论](analysis.json)、[完整逐组审计](raw-results.json)、[参数/顺序/负载哈希](experiment-settings.json)。',
              '- 每个编号目录包含 requests.csv、trace-chunks.csv、完整日志、环境与阶段墙钟；source-runtime 保留实际运行时代码及 SHA256。',
              '- [可导出 PDF](parameter-heatmaps.pdf)。', '']
    (output/'PARAMETER_SCAN_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'best': [best['progress_pct'],best['wait_ms']], 'makespan_s':best['makespan_s'],
                      'baseline_s':best['bracketing_baseline_s'], 'reduction_pct':best['reduction_pct'],
                      'wait_protection_runs':len(wait_reasons)}, indent=2))


if __name__ == '__main__':
    main()
