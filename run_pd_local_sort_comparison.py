#!/usr/bin/env python3
"""Two arrival densities, three systems, rotating order across three rounds."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
ORDERS = [('baseline', 'old', 'new'), ('old', 'new', 'baseline'),
          ('new', 'baseline', 'old')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', required=True)
    parser.add_argument('--resume-first-baseline', action='store_true')
    args = parser.parse_args()
    out = Path(args.result_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    plan = dict(arrival_gaps_ms=[250, 500], rounds=3, orders=ORDERS,
                group_size=5, intra_group_gap_ms=0, local_sort_window_ms=20,
                naming=dict(baseline='whole-request transfer',
                            old='prior full transfer scheme without sorting',
                            new='full scheme with P local sorting'))
    (out/'experiment-plan.json').write_text(json.dumps(plan, indent=2))
    env = os.environ.copy()
    env.update(P_DEVICE='0', D_DEVICE='1', VLLM_BATCH_INVARIANT='1',
               PD_MEASURE_EXECUTION='1', PD_KV_SOURCE_FENCE='event',
               PD_STREAM_MAX_IN_FLIGHT='1', PD_STREAM_EAGER_PROGRESS='0',
               ENABLE_PREFIX_CACHING='false')
    for round_id, order in enumerate(ORDERS, 1):
        for system in order:
            target = out/f'round-{round_id}'/system
            if args.resume_first_baseline and round_id == 1 and system == 'baseline':
                for gap in (250, 500):
                    if not (target/f'run-1-gap-{gap}.csv').is_file():
                        raise RuntimeError('First baseline is not complete')
                continue
            if target.exists():
                raise FileExistsError(f'Refusing to overwrite {target}')
            target.parent.mkdir(parents=True, exist_ok=True)
            gaps = ['250', '500'] if round_id % 2 else ['500', '250']
            cmd = [sys.executable, str(ROOT/'run_sglang_pd_experiment.py'),
                   '--system', 'vllm-baseline' if system == 'baseline' else 'vllm-full',
                   '--result-dir', str(target), '--repeats', '1',
                   '--grouped-arrivals', '--arrival-gaps-ms', *gaps]
            if system == 'new':
                cmd += ['--local-sort']
            print(f'Running round={round_id} system={system}', flush=True)
            with (target.parent/f'{system}-driver.log').open('w') as log:
                subprocess.run(cmd, cwd=ROOT, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            print(f'Finished round={round_id} system={system}', flush=True)
    (out/'runs-complete.json').write_text(json.dumps(dict(formal_runs=18)))


if __name__ == '__main__':
    main()
