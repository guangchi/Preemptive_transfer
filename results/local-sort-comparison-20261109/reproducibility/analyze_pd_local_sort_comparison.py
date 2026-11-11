#!/usr/bin/env python3
"""Audit all repeats and report the three schemes at both arrival densities."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics
from collections import defaultdict

from collect_pd_trace import collect, base_request_id


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def original_id(value):
    return re.match(r'interleaved15-\d{2}-(short|medium|long)', value).group()


def mean_std(values):
    return dict(mean=statistics.mean(values), sd=statistics.stdev(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', required=True)
    args = parser.parse_args()
    root = Path(args.result_dir).resolve()
    reference_rows = read_csv(Path(__file__).parent/'results/local-sort-validation-20261109/off/run-1.csv')
    reference = {original_id(r['request_id']): json.loads(r['response_token_ids']) for r in reference_rows}
    runs, details, windows_audit = [], [], []
    total_requests = 0
    for round_id in range(1, 4):
        for system in ('baseline', 'old', 'new'):
            folder = root/f'round-{round_id}'/system
            manifest = json.loads((folder/'manifest.json').read_text())
            env = json.loads((folder/'environment.json').read_text())
            assert manifest['local_sort'] == (system == 'new')
            assert env['p_device'] == '0' and env['d_device'] == '1'
            assert env['mode'] == ('baseline' if system == 'baseline' else 'preemptive')
            assert env['batch_invariant']
            log = (folder/'prefill.log').read_text()
            for phase in ('warmup', 'run-1'):
                for gap in (250, 500):
                    label = f'{phase}-gap-{gap}'
                    path = folder/f'{label}.csv'
                    rows = read_csv(path)
                    assert len(rows) == 15
                    for row in rows:
                        assert row['status'] == '200' and row['output_tokens'] == '8'
                        assert row['reported_prompt_tokens'] == row['prompt_tokens']
                        assert json.loads(row['response_token_ids']) == reference[original_id(row['request_id'])]
                    total_requests += len(rows)
                    if phase == 'warmup':
                        continue
                    prefix = folder/f'trace-gap-{gap}'
                    phase_path, chunk_path = collect(path, folder/'prefill.log', folder/'decode.log',
                        prefix, require_execution=True, require_complete=system != 'baseline')
                    phases = read_csv(phase_path)
                    if system != 'baseline':
                        assert all(float(r['d_first_model_submit_monotonic_s']) >=
                                   float(r['d_kv_receive_complete_monotonic_s']) for r in phases)
                    ids = {r['request_id'] for r in rows}
                    # Engine loggers may append another record to print's line.
                    admissions = []
                    for fragment in log.split('PD local admission ')[1:]:
                        entry, _ = json.JSONDecoder().raw_decode(fragment)
                        if base_request_id(entry['request_id']) in ids:
                            admissions.append(entry)
                    grouped = defaultdict(list)
                    for entry in admissions:
                        grouped[entry['window']].append(entry)
                    if system == 'new':
                        assert len(admissions) == 15
                        assert {base_request_id(e['request_id']) for e in admissions} == ids
                        assert [e['window'] for e in admissions] == sorted(e['window'] for e in admissions)
                        for entries in grouped.values():
                            assert [(e['prompt_tokens'], e['arrived_monotonic']) for e in entries] == sorted(
                                (e['prompt_tokens'], e['arrived_monotonic']) for e in entries)
                            assert all(e['released_monotonic'] >= e['deadline_monotonic'] for e in entries)
                    else:
                        assert not admissions
                    windows_audit.append(dict(round=round_id, system=system, gap_ms=gap, windows=list(grouped.values())))
                    preemptions = [line for line in log.splitlines() if 'decision=preempt_long' in line
                                   and f'-{label}-0-' in line]
                    cross_batch_preemptions = 0
                    for line in preemptions:
                        short, long = re.search(r'req=(\S+).*long_req=(\S+)', line).groups()
                        a, b = [int(re.search(r'interleaved15-(\d+)', x).group(1)) // 5 for x in (short, long)]
                        cross_batch_preemptions += a > b
                    starts = [float(r['submitted_at_monotonic_s']) for r in rows]
                    first = min(starts)
                    deviation = max(abs((float(r['submitted_at_monotonic_s']) - first) * 1000 -
                                        int(r['dispatch_group']) * gap) for r in rows)
                    result = dict(round=round_id, system=system, gap_ms=gap, requests=len(rows),
                        makespan_s=max(float(r['request_finished_monotonic_s']) for r in rows) - first,
                        preemptions=len(preemptions), cross_batch_preemptions=cross_batch_preemptions,
                        window_count=len(grouped), window_sizes=[len(g) for g in grouped.values()],
                        max_client_schedule_deviation_ms=deviation)
                    for category in ('short', 'medium', 'long'):
                        subset = [r for r in rows if r['category'] == category]
                        for metric in ('ttft_ms', 'e2e_ms'):
                            result[f'{category}_{metric}'] = statistics.mean(float(r[metric]) for r in subset)
                    for batch in range(3):
                        result[f'batch_{batch}_short_ttft_ms'] = statistics.mean(
                            float(r['ttft_ms']) for r in rows if int(r['dispatch_group']) == batch and r['category'] == 'short')
                    runs.append(result)
                    for row in rows:
                        details.append(dict(round=round_id, system=system, gap_ms=gap,
                            original_id=original_id(row['request_id']), category=row['category'],
                            batch=int(row['dispatch_group']), ttft_ms=float(row['ttft_ms']),
                            e2e_ms=float(row['e2e_ms'])))
    aggregates = []
    metrics = ['makespan_s'] + [f'{c}_{m}' for c in ('short', 'medium', 'long')
                               for m in ('ttft_ms', 'e2e_ms')]
    metrics += [f'batch_{b}_short_ttft_ms' for b in range(3)]
    for gap in (250, 500):
        for system in ('baseline', 'old', 'new'):
            subset = [r for r in runs if r['gap_ms'] == gap and r['system'] == system]
            aggregates.append(dict(gap_ms=gap, system=system, repeats=len(subset),
                metrics={m: mean_std([r[m] for r in subset]) for m in metrics},
                preemptions=[r['preemptions'] for r in subset],
                cross_batch_preemptions=[r['cross_batch_preemptions'] for r in subset],
                window_sizes=[r['window_sizes'] for r in subset]))
    (root/'audit.json').write_text(json.dumps(dict(total_requests=total_requests,
        formal_requests=270, outputs_match_reference=True, runs=runs, aggregates=aggregates,
        admission_windows=windows_audit), indent=2))
    for name, rows in [('runs.csv', runs), ('requests.csv', details)]:
        with (root/name).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(aggregates, indent=2))


if __name__ == '__main__':
    main()
