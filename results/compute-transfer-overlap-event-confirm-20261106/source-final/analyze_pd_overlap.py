#!/usr/bin/env python3
"""Report CUDA kernel/copy overlap from a producer Kineto Chrome trace."""
import argparse
from collections import Counter
import json
from pathlib import Path


def merge(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(end, result[-1][1])
        else:
            result.append([start, end])
    return result


def duration(intervals):
    return sum(end - start for start, end in intervals)


def intersection(a, b):
    i = j = 0
    result = []
    while i < len(a) and j < len(b):
        start, end = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if end > start:
            result.append([start, end])
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    events = json.loads(args.trace.read_text())["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel" and e["args"]["device"] == 0]
    copies = [e for e in events if e.get("cat") == "gpu_memcpy"
              and e["name"] == "Memcpy DtoH (Device -> Pageable)" and e["args"]["device"] == 0]
    if not kernels or not copies:
        raise ValueError("Expected producer kernels and UCX pageable DtoH copy activity")
    compute = merge((e["ts"], e["ts"] + e["dur"]) for e in kernels)
    transfer = merge((e["ts"], e["ts"] + e["dur"]) for e in copies)
    overlap = intersection(compute, transfer)
    summary = {
        "source": str(args.trace.resolve()),
        "window": "first real prompt forward to all prompt finals; transfer tail excluded",
        "kernel_count": len(kernels), "dtoh_copy_count": len(copies),
        "kernel_streams": dict(Counter(e["args"]["stream"] for e in kernels)),
        "dtoh_streams": dict(Counter(e["args"]["stream"] for e in copies)),
        "copy_size_bytes_counts": dict(Counter(e["args"]["bytes"] for e in copies)),
        "dtoh_bytes": sum(e["args"]["bytes"] for e in copies),
        "compute_busy_ms": duration(compute) / 1000,
        "dtoh_busy_ms": duration(transfer) / 1000,
        "overlap_ms": duration(overlap) / 1000,
        "dtoh_overlap_ratio": duration(overlap) / duration(transfer),
        "compute_overlap_ratio": duration(overlap) / duration(compute),
        "compute_span_ms": (compute[-1][1] - compute[0][0]) / 1000,
        "dtoh_span_ms": (transfer[-1][1] - transfer[0][0]) / 1000,
        "limitations": ["CUPTI device intervals are used; no host enqueue timestamps are treated as DMA.",
                        "DtoH is the observed producer-side UCX staging leg; it excludes CPU/socket and D-side HtoD.",
                        "Profile overhead is material. This run is excluded from paired performance means.",
                        "NIXL xferDuration includes delayed checkXferReq polling, so it is not pure DMA time."],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cuda-overlap.json").write_text(json.dumps(summary, indent=2) + "\n")
    origin = min(compute[0][0], transfer[0][0])
    end = max(compute[-1][1], transfer[-1][1])
    # Preserve device event intervals in a lightweight zoomable viewer.
    series = [[[round((a-origin)/1000, 6), round((b-origin)/1000, 6)] for a,b in xs]
              for xs in (compute, transfer, overlap)]
    html = '''<!doctype html><meta charset="utf-8"><title>P CUDA overlap</title>
<style>body{font:16px sans-serif;margin:32px;max-width:1300px}canvas{width:100%;height:200px;border:1px solid #ccc}input{width:95%}</style>
<h1>P compute / UCX DtoH overlap</h1><p>CUDA device intervals. Window ends after all prompt forwards, excluding the transfer tail. Profile is for diagnosis only.</p>
<p id="summary"></p><label>Zoom <input id="zoom" type="range" min="1" max="50" value="1"></label>
<label>Position <input id="position" type="range" min="0" max="1000" value="0"></label>
<canvas id="chart" width="2400" height="400"></canvas><p id="range"></p>
<p>Blue: compute kernels. Orange: producer DtoH staging copies. Green: their device overlap. CPU/socket transfer and D HtoD are outside this trace.</p>
<script>const data=DATA, span=SPAN, stats=STATS;
summary.textContent=`Overlap ${stats.overlap_ms.toFixed(3)} ms; ${(stats.dtoh_overlap_ratio*100).toFixed(2)}% of observed DtoH busy time.`;
function draw(){const c=chart.getContext('2d'),w=chart.width,h=chart.height;
c.clearRect(0,0,w,h);const width=span/Number(zoom.value),lo=(span-width)*Number(position.value)/1000;
const names=['Compute','DtoH copy','Overlap'],colors=['#2563eb','#ea580c','#16a34a'];
for(let i=0;i<3;i++){c.fillStyle='#111';c.font='26px sans-serif';c.fillText(names[i],10,75+i*100);
c.fillStyle=colors[i];for(const [a,b] of data[i]){if(b<lo||a>lo+width)continue;
const x=230+Math.max(0,(a-lo)/width)*(w-250),xe=230+Math.min(1,(b-lo)/width)*(w-250);
c.fillRect(x,40+i*100,Math.max(.5,xe-x),55);}}
range.textContent=`${lo.toFixed(2)}–${(lo+width).toFixed(2)} ms from first kernel`;}
zoom.oninput=position.oninput=draw;draw();</script>'''
    html = html.replace("DATA", json.dumps(series)).replace("SPAN", str((end-origin)/1000)).replace("STATS", json.dumps(summary))
    (args.output / "cuda-overlap.html").write_text(html)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
