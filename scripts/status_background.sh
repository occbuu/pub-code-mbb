#!/usr/bin/env bash
# Show whether the detached Paper 13 run is alive, and the last log lines.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="$ROOT/logs"
pidfile="$LOGDIR/full.pid"

echo "=== status $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
if [[ -f "$pidfile" ]]; then
  pid="$(cat "$pidfile")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "orchestrator running  pid=$pid"
    ps -o pid,etime,pcpu,pmem,cmd -p "$pid" 2>/dev/null || true
    # children (GPU / sklearn workers)
    pgrep -P "$pid" >/dev/null 2>&1 && ps -o pid,etime,pcpu,pmem,cmd --ppid "$pid" || true
  else
    echo "pidfile $pid is stale (process not running)"
  fi
else
  echo "no pidfile  ($pidfile)"
fi

if [[ -f "$LOGDIR/status.json" ]]; then
  echo
  echo "--- status.json ---"
  cat "$LOGDIR/status.json"
fi

echo
echo "--- GPU ---"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv || true

if [[ -f "$LOGDIR/full.log" ]]; then
  echo
  echo "--- last 30 lines of full.log ---"
  tail -n 30 "$LOGDIR/full.log"
fi
