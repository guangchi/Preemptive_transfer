#!/usr/bin/env python3
"""Write a candid repeated-trial tuning report, including guardrail failures."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid', required=True)
    parser.add_argument('--confirmation', required=True)
    args = parser.parse_args()
    grid, root = Path(args.grid), Path(args.confirmation)
    result = json.loads((root / 'confirmation-results.json').read_text())
    raw = json.loads((root / 'raw-confirmation.json').read_text())
    search = json.loads((grid / 'search-completion.json').read_text())
    references = {}
    reference_file = Path('results/determinism-investigation-20261106/responses.jsonl')
    for line in reference_file.read_text().splitlines():
        item = json.loads(line)
        if item['batch_invariant']:
            references.setdefault(item['request_id'], set()).add(tuple(json.loads(item['response_token_ids'])))
    output_failures = []
    for trial in raw:
        for response in trial['responses']:
            if tuple(json.loads(response['response_token_ids'])) not in references[response['request_id']]:
                output_failures.append({'repeat': trial['repeat'], 'setting': trial['setting'],
                    'request_id': response['request_id'], 'actual_token_ids': json.loads(response['response_token_ids']),
                    'reference_token_ids': [list(tokens) for tokens in references[response['request_id']]]})
    (root/'output-failures.json').write_text(json.dumps(output_failures, indent=2)+'\n')
    groups = []
    for setting in result['configurations']:
        rs = [r for r in raw if r['setting'] == setting]
        if not rs:
            continue
        candidate = next((r for r in result['ranking'] if r['setting'] == setting), None)
        categories = {cat: {field: statistics.mean(
            statistics.mean(float(m[field]) for m in r['metrics'] if m['category'] == cat) for r in rs)
            for field in ['ttft_ms', 'e2e_ms']} for cat in ['short', 'medium', 'long']}
        makespans = [r['workload_makespan_s'] for r in rs]
        score = statistics.median((statistics.mean(float(m['ttft_ms']) for m in r['metrics'] if m['category'] == 'short') * statistics.mean(float(m['ttft_ms']) for m in r['metrics'] if m['category'] == 'medium'))**.5 for r in rs)
        record = {'setting': setting, 'repeats': len(rs), 'categories': categories,
                  'workload_makespan_s_mean': statistics.mean(makespans),
                  'workload_makespan_s_min': min(makespans), 'workload_makespan_s_max': max(makespans),
                  'median_score_ms': score, 'group_wall_s_mean': statistics.mean(r['group_wall_s'] for r in rs),
                  'successful_requests_per_s_mean': statistics.mean(r['successful_requests_per_s'] for r in rs),
                  'outputs_and_references_match': all(p['outputs_match'] and p['direct_reference_match'] for p in candidate['paired_runs']) if candidate else not any(f['setting'] == setting for f in output_failures),
                  'guardrail_pass': candidate['eligible'] if candidate else None,
                  'max_paired_long_e2e_ratio': max(p['max_long_e2e_ratio'] for p in candidate['paired_runs']) if candidate else None}
        groups.append(record)
    long_pairs, preemption_rows = [], []
    for trial in raw:
        if trial['setting'] == 'baseline':
            continue
        baseline_trial = next(r for r in raw if r['repeat'] == trial['repeat'] and r['setting'] == 'baseline')
        baseline_metrics = {r['request_id']: r for r in baseline_trial['metrics']}
        for metric in trial['metrics']:
            if metric['category'] == 'long':
                base_e2e = float(baseline_metrics[metric['request_id']]['e2e_ms'])
                e2e = float(metric['e2e_ms'])
                long_pairs.append({'repeat': trial['repeat'], 'setting': trial['setting'],
                                   'request_id': metric['request_id'], 'baseline_e2e_ms': base_e2e,
                                   'candidate_e2e_ms': e2e, 'ratio': e2e / base_e2e})
        directory = Path(trial['run_dir'])
        chunks = list(csv.DictReader((directory / 'trace-chunks.csv').open()))
        preemptions = [c for c in chunks if c['decision'] == 'preempt_long']
        first_protection = {}
        for c in sorted(chunks, key=lambda c: float(c['dispatch_monotonic_s'] or 'inf')):
            if c['decision'] == 'protected_resume':
                first_protection.setdefault(c['request_id'], {k: c[k] for k in
                    ['sequence', 'long_progress', 'long_wait_ms', 'protection_reasons']})
        preemption_rows.append({'repeat': trial['repeat'], 'setting': trial['setting'],
                                'preempt_long_chunks': len(preemptions),
                                'max_long_preemption_count': max((int(c['long_preemptions']) for c in preemptions), default=0),
                                'first_protection': first_protection})
    with (root / 'paired-long-e2e.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(long_pairs[0])); writer.writeheader(); writer.writerows(long_pairs)
    (root / 'preemption-summary.json').write_text(json.dumps(preemption_rows, indent=2) + '\n')
    valid = [r for r in groups if r['setting'] != 'baseline' and r['repeats'] == result['repeats'] and r['outputs_and_references_match']]
    observed = min(valid, key=lambda r: r['median_score_ms']) if valid else None
    base = next((r for r in groups if r['setting'] == 'baseline'), None)
    default = next((r for r in groups if r['setting'] == 'c4-p3-x0.85-w50'), None)
    total_wall = max((__import__('datetime').datetime.fromisoformat(c['start_utc']).timestamp() + c['wall_s']) for c in result['commands']) - search['start_unix_s']
    decision = {'strict_guardrail_recommendation': result['recommended'],
                'observed_lowest_latency_among_output_correct_settings': observed['setting'] if observed else None,
                'grid_search_wall_s': search['total_search_wall_s'],
                'confirmation_wall_s': result['total_confirmation_wall_s'],
                'overall_search_and_confirmation_wall_s': total_wall,
                'summary': groups,
                'scope': 'Fixed 15-request workload, GPU/software configuration and searched discrete range; no global optimality or stable tail percentile claim.',
                'default_parameters_changed': False,
                'output_failures': output_failures,
                'guardrail_is_analyst_choice': 'Per-request long E2E at most +10% relative to fresh paired baseline, not a user SLA or a hard system guarantee.'}
    (root / 'final-decision.json').write_text(json.dumps(decision, indent=2, ensure_ascii=False) + '\n')
    fields = ['setting','repeats','median_score_ms','workload_makespan_s_mean','workload_makespan_s_min','workload_makespan_s_max','successful_requests_per_s_mean','group_wall_s_mean','outputs_and_references_match','guardrail_pass','max_paired_long_e2e_ratio']
    with (root/'confirmation-summary.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(groups)
    lines = ['# 15 请求总时间与保护参数调优：最终报告', '',
             '## 范围与选择依据', '',
             '固定 8 short / 3 medium / 4 long，原交错顺序、prompt、seed 不变；并发 15、到达间隔 100ms、prefix cache 关闭、chunk=4、batch-invariant=1。所有本轮参数与确认配置均关闭 top-logprobs，保留文本和 token ID。',
             '首轮网格共 27 个组合：进度={70%,85%,95%}、次数={1,3,5}、等待={25,50,100}ms。25 组输出通过，21 组同时满足初筛 long 约束。一个启动超时原样保存，单独补跑；另两个输出差异保留，不以重跑替换失败。',
             '网格历史 baseline 开启 top-logprobs，只作初筛；最终使用统一关闭审计的新 baseline，候选和默认各 3 次、新 baseline 共 3 次。循环调整配置执行顺序，减轻固定先后顺序影响。',
             '目标：尽量减小 short/medium 类 TTFT 均值几何平均；重复阶段按三轮分数中位数选择。筛选约束：15/15 成功、所有 token ID/文本 hash 配对一致且匹配直接推理参考、每个 long 相对同轮新 baseline 的 E2E 增幅不超过 10%。10% 是分析者设定的约束，不是用户 SLA；不满足时明确报告无可行推荐。',
             '', '## 重复实验结果', '',
             '| 配置 | short TTFT mean(ms) | medium TTFT mean(ms) | long TTFT mean(ms) | 整批总时间 mean[min,max](s) | 分数中位数(ms) | 最差 long E2E 比值 | 输出通过 | 全约束通过 |',
             '| --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |']
    for r in groups:
        ratio = '-' if r['max_paired_long_e2e_ratio'] is None else f"{r['max_paired_long_e2e_ratio']:.3f}"
        guardrail = '-' if r['guardrail_pass'] is None else r['guardrail_pass']
        lines.append(f"| {r['setting']} | {r['categories']['short']['ttft_ms']:.3f} | {r['categories']['medium']['ttft_ms']:.3f} | {r['categories']['long']['ttft_ms']:.3f} | {r['workload_makespan_s_mean']:.3f} [{r['workload_makespan_s_min']:.3f},{r['workload_makespan_s_max']:.3f}] | {r['median_score_ms']:.3f} | {ratio} | {r['outputs_and_references_match']} | {guardrail} |")
    lines += ['', f"严格约束内推荐：`{result['recommended'] or '没有配置在三轮中同时通过全部约束'}`。",
              f"在输出正确的测试配置中，观测到的延迟分数最低为：`{observed['setting'] if observed else '没有配置通过三轮输出校验'}`。两种结论区分保存，不把有 long 尾部代价的配置描述为满足约束。"]
    if observed and default:
        change=(observed['median_score_ms']/default['median_score_ms']-1)*100
        lines += [f"该配置相对默认的分数中位数变化：{change:+.2f}%（分数越低越好）；默认有输出失败，不能仅凭该性能比较视作合格方案。"]
    if not result['recommended']:
        lines += ['本轮未找到同时满足正确性、长请求代价约束的稳健最优点；默认仍为 85%/3 次/50ms，不依据单轮最低值自动替换。观测延迟最低配置仅是本测试和候选范围内的取舍，不能称为满足长请求约束的推荐。']
    if output_failures:
        lines += ['', f"确认阶段 {len(raw)} 组均 HTTP 15/15、严格 trace 完成，但有 {len(output_failures)} 条输出不匹配直接参考："]
        for failure in output_failures:
            lines.append(f"- repeat-{failure['repeat']} / {failure['setting']} / {failure['request_id']}：实际 token={failure['actual_token_ids']}，参考={failure['reference_token_ids']}。")
        lines += ['', '默认配置性能行保留原始观测供审计，含失败轮的数据不能视作合格推荐。残余输出差异尚未定位根因，不能由 HTTP 成功或 batch-invariant 开关推导 KV/数值正确性已被全面证明。']
    lines += ['', '## 各类 E2E 与实际抢占', '',
              '| 配置 | short E2E mean(ms) | medium E2E mean(ms) | long E2E mean(ms) | 抢占 chunk 数（三轮） |',
              '| --- | ---: | ---: | ---: | --- |']
    for r in groups:
        counts = [p['preempt_long_chunks'] for p in preemption_rows if p['setting'] == r['setting']]
        lines.append(f"| {r['setting']} | {r['categories']['short']['e2e_ms']:.3f} | {r['categories']['medium']['e2e_ms']:.3f} | {r['categories']['long']['e2e_ms']:.3f} | {counts or '-'} |")
    lines += ['', '逐条 long 的配对 baseline/E2E/比值见 paired-long-e2e.csv；preemption-summary.json 记录各轮每条 long 首次保护原因、进度、等待。均值可能下降但最后 long 变慢，不能只使用 long 均值判断防饥饿代价。']
    lines += ['', '## 总时间记录', '',
              '| 阶段 | 墙钟/总时间(s) |', '| --- | ---: |',
              f"| 网格含补跑、失败与阶段间操作 | {search['total_search_wall_s']:.3f} |",
              f"| 重复确认阶段 | {result['total_confirmation_wall_s']:.3f} |",
              f"| 从搜索开始到最后确认组结束 | {total_wall:.3f} |", '',
              '整批请求总时间（workload_makespan_s）＝第一条 HTTP 请求发出到最后响应结束，含请求到达间隔与排队，不等于逐请求 E2E 相加。表中吞吐使用成功请求数/整批总时间。模型启动、健康检查、15 秒空闲等待、trace、清理单列在每组 experiment-timing.json；15 秒空闲等待本身不发预热请求，模型启动过程中另有 vLLM 自带的内核预热。', '',
              '## 结果边界与机制', '',
              '三个防饥饿条件是 OR 关系，先触发的条件决定后续保护。short=128 tokens，对应 8 KV blocks/2 chunks；次数=1 可能只提前传出第一个 chunk，尾 chunk 仍被 long 阻塞，不能只看插队次数认定短请求已完成。等待阈值接近两个 chunk 的实际 host/传输时长时，会随时序波动决定 long 是否进入持续保护，进而改变大量后续 chunk 的顺序。medium FIFO 屏障仍会限制后面的 short。',
              '确认阶段三个抢占配置每轮均发生 8 次 short chunk 插队，但顺序和保护边界不同。95%/5/100ms 每轮均有单个 long 累计被越过 4 个 chunk；95%/5/50ms 与默认每条 long 至多 2 次。前者允许一个 long 前完成两个 short 的全部 KV；50ms 组更早触发等待保护。次数总和相同并不表示调度过程或性能相同。',
              '单次最低点在重复实验中不一定保持最快；本次每配置仅三轮，足以发现明显的不稳定与取舍，不能估计稳定 P95/P99，也不证明进度阈值在连续范围中的全局最优。固定 15 请求负载无法单独识别所有保护参数作用，结论仅限已搜索的范围。',
              '本轮网格两次出现已启用 batch-invariant 但第一条 long 输出不同的情况，已排除；因此旧单轮一致性验收不能扩展成所有调度组合均确定的保证。失败原始 token 和日志保留，当前报告不声称该残余正确性问题已全面解决。', '',
              '## 原始证据', '',
              '- 网格：tuning-results.json/CSV、workload-total-times.json/CSV、每点完整请求和抢占时间线。',
              '- 确认：raw-confirmation.json、confirmation-results.json、confirmation-summary.csv、final-decision.json；每次服务运行有独立日志、请求和分段计时。',
              '- 每个候选/默认与同轮 baseline 的配对目录由 summarize_pd_experiment.py 生成请求指标、preemptions.csv、protection-decisions.csv、experiment-timeline.csv 及 timeline.html。',
              '- 失败启动位于网格 failed-attempts/；输出不一致的点 c4-p1-x0.95-w100、c4-p5-x0.70-w50 保留在主网格目录。', '']
    (root/'FINAL_TUNING_REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(decision,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
