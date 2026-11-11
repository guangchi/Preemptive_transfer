#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/ubuntu/miniconda3/envs/cc/bin/python}"
WORKLOAD="${WORKLOAD:-$ROOT/workload.sweep-interleaved-8s3m4l.jsonl}"
BASELINE_RESULT_DIR="${BASELINE_RESULT_DIR:-$ROOT/results/sweep-interleaved15-invariant-verified-20261106/baseline-c4-p0-x0.85-w50}"
TUNING_ROOT="${TUNING_ROOT:-$ROOT/results/tune-interleaved15-20261106}"
MAX_PREEMPTIONS_VALUES="${MAX_PREEMPTIONS_VALUES:-1 3 5}"
PROTECT_PROGRESS_VALUES="${PROTECT_PROGRESS_VALUES:-0.70 0.85 0.95}"
WAIT_THRESHOLDS_MS="${WAIT_THRESHOLDS_MS:-25 50 100}"
CHUNK_SIZE_BLOCKS="${CHUNK_SIZE_BLOCKS:-4}"
TOTAL_START="$(date +%s.%N)"
FAILED=0
mkdir -p "$TUNING_ROOT"

for max_preemptions in $MAX_PREEMPTIONS_VALUES; do
  for progress in $PROTECT_PROGRESS_VALUES; do
    for wait_ms in $WAIT_THRESHOLDS_MS; do
      tag="c${CHUNK_SIZE_BLOCKS}-p${max_preemptions}-x${progress}-w${wait_ms}"
      result="$TUNING_ROOT/$tag"
      mkdir -p "$result"
      printf 'starting %s at %s\n' "$tag" "$(date -Is)" | tee -a "$TUNING_ROOT/tuning.log"
      if ! RUN_BASELINE=false BASELINE_RESULT_DIR="$BASELINE_RESULT_DIR" \
        WORKLOAD="$WORKLOAD" RESULT_ROOT="$result" REQUESTS=15 CONCURRENCY=15 \
        INTERARRIVAL_MS=100 CHUNK_SIZES="$CHUNK_SIZE_BLOCKS" \
        MAX_PREEMPTIONS="$max_preemptions" PROTECT_PROGRESS_VALUES="$progress" \
        WAIT_THRESHOLDS_MS="$wait_ms" WAIT_SECONDS=15 ENABLE_PREFIX_CACHING=false \
        STREAM_RESPONSES=true PD_MEASURE_EXECUTION=1 VLLM_BATCH_INVARIANT=1 \
        PD_CAPTURE_LOGPROBS=0 bash "$ROOT/run_pd_sweep.sh" \
        >"$result/runner.log" 2>&1; then
        printf 'failed %s at %s\n' "$tag" "$(date -Is)" | tee -a "$TUNING_ROOT/tuning.log"
        FAILED=1
      else
        printf 'finished %s at %s\n' "$tag" "$(date -Is)" | tee -a "$TUNING_ROOT/tuning.log"
      fi
    done
  done
done

TOTAL_END="$(date +%s.%N)"
"$PYTHON_BIN" - "$TUNING_ROOT" "$TOTAL_START" "$TOTAL_END" "$FAILED" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1]); start, end = map(float, sys.argv[2:4])
records = []
for path in sorted(root.glob("c*-p*-x*-w*/sweep-timing.json")):
    timing = json.loads(path.read_text())
    records.append({"tag": path.parent.name, "total_sweep_wall_s": timing["total_sweep_wall_s"]})
data = {"start_unix_s": start, "end_unix_s": end, "total_grid_wall_s": end-start,
        "completed_parameter_points": len(records), "points": records, "grid_failed": sys.argv[4] != "0"}
(root / "tuning-total-timing.json").write_text(json.dumps(data, indent=2) + "\n")
PY

"$PYTHON_BIN" "$ROOT/analyze_pd_tuning.py" --root "$TUNING_ROOT" \
  --baseline-dir "$BASELINE_RESULT_DIR" --workload "$WORKLOAD" \
  >"$TUNING_ROOT/analysis.log" 2>&1 || FAILED=1
exit "$FAILED"
