#!/usr/bin/env python3
"""Render an SVG request timeline with recorded preemption decisions."""
import argparse
import csv
import html
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--baseline-dir")
    args = parser.parse_args()
    root = Path(args.root)
    data = {}
    origins = {}
    for mode in ("baseline", "preemptive"):
        directory = Path(args.baseline_dir) if mode == 'baseline' and args.baseline_dir else next(root.glob(mode + "-c*-p*-x*-w*"))
        origins[mode] = json.loads((directory / "run.json").read_text())["start_monotonic_s"]
        data[mode] = list(csv.DictReader((directory / "requests.csv").open()))
    preemptions = list(csv.DictReader((root / "preemptions.csv").open()))
    maximum = math.ceil(max(float(row["request_finished_monotonic_s"]) - origins[mode]
                            for mode, rows in data.items() for row in rows))
    width, left, plot_width, row_h = 1140, 140, 930, 23
    panel_height = 55 + row_h * len(data["baseline"])
    height = 110 + 2 * panel_height
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<style>text{font-family:Arial,sans-serif;font-size:12px;fill:#243342} .title{font-size:18px;font-weight:bold}</style>',
           '<text x="20" y="27" class="title">Interleaved 15-request PD experiment</text>',
           '<text x="20" y="49">Gray: send to first text | Blue: first text to finish | Orange diamond: short bypasses long (arbiter decision)</text>',
           '<text x="20" y="68">Time is relative to each replay start. Hover for values. S=short, M=medium, L=long. Decisions are not WRITE start times.</text>']
    for panel, (mode, rows) in enumerate(data.items()):
        top = 95 + panel * panel_height
        svg.append(f'<text x="20" y="{top}" class="title">{mode}</text>')
        end_y = top + 23 * len(rows)
        for tick in range(0, maximum + 1, 2):
            x = left + tick / maximum * plot_width
            svg.append(f'<line x1="{x}" x2="{x}" y1="{top+8}" y2="{end_y+8}" stroke="#e5e9ed"/>')
            svg.append(f'<text x="{x-7}" y="{end_y+27}">{tick}s</text>')
        for i, row in enumerate(rows):
            rid = row["request_id"]
            y = top + 15 + i * row_h
            start = float(row["request_started_monotonic_s"]) - origins[mode]
            first = float(row["first_token_monotonic_s"]) - origins[mode]
            finish = float(row["request_finished_monotonic_s"]) - origins[mode]
            scale = plot_width / maximum
            label = rid.split('-')[1] + ' ' + row['category'][0].upper()
            svg.append(f'<text x="80" y="{y+10}">{label}</text>')
            title = html.escape(f"{rid}: send={start:.3f}s, first={first:.3f}s, finish={finish:.3f}s; TTFT={row['ttft_ms']}ms, E2E={row['e2e_ms']}ms")
            svg.append(f'<g><title>{title}</title><rect x="{left+start*scale:.2f}" y="{y}" width="{(first-start)*scale:.2f}" height="13" fill="#cdd9e5"/><rect x="{left+first*scale:.2f}" y="{y}" width="{max(0.5,(finish-first)*scale):.2f}" height="13" fill="#356bad"/></g>')
            if mode == "preemptive":
                for p in preemptions:
                    if p["request_id"] != rid:
                        continue
                    x = left + float(p['relative_ms']) / 1000 * scale
                    yy = y + 6
                    detail = html.escape(f"{rid} chunk {p['sequence']} bypasses {p['long_request_id']}; t={p['relative_ms']}ms, count={p['long_preemptions']}, progress={p['long_progress']}, wait={p['long_wait_ms']}ms")
                    svg.append(f'<polygon points="{x:.2f},{yy-6} {x+4:.2f},{yy} {x:.2f},{yy+6} {x-4:.2f},{yy}" fill="#d96d12"><title>{detail}</title></polygon>')
    svg.append('</svg>')
    content = '\n'.join(svg)
    (root / 'request-timeline.svg').write_text(content)
    (root / 'timeline.html').write_text('<!doctype html><meta charset="utf-8"><title>PD timeline</title><style>body{margin:24px;font-family:Arial}svg{max-width:100%;height:auto}</style>' + content + '<p><a href="EXPERIMENT_REPORT.md">Experiment report</a> | <a href="preemptions.csv">Preemptions</a> | <a href="experiment-timeline.csv">All events</a></p>')


if __name__ == '__main__':
    main()
