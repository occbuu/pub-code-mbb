#!/usr/bin/env bash
# Launch the Paper 13 pipeline in a new session so it survives Cursor/SSH logout.
#
#   bash scripts/launch_background.sh
#   bash scripts/launch_background.sh --from-stage gpu
#   bash scripts/status_background.sh
#   bash scripts/stop_background.sh
#
# Budget on this workstation: ~24 CPUs (taskset) and exactly one GPU.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

PYTHON="${CFT_PYTHON:-/workspace/anlk/HaoLee/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

CPU_WORKERS="${CFT_CPU_WORKERS:-24}"
CPU_AFFINITY="${CFT_CPU_AFFINITY:-24-47}"   # 24 logical CPUs, socket-1 physical cores
GPU="${CFT_GPU:-auto}"
GPU_CANDIDATES="${CFT_GPU_CANDIDATES:-0,1}"   # lab: only cards 0 and 1
MIN_FREE="${CFT_MIN_GPU_FREE_MIB:-9000}"
PROFILE="${CFT_PROFILE:-a5000}"
REPS="${CFT_REPS:-2000}"
NICE="${CFT_NICE:-10}"

pidfile="$LOGDIR/full.pid"
if [[ -f "$pidfile" ]]; then
  old="$(cat "$pidfile" || true)"
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null; then
    echo "already running as pid $old  (stop with scripts/stop_background.sh)"
    exit 1
  fi
fi

# Pick the emptiest GPU unless the caller already pinned one.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  if [[ "$GPU" == "auto" ]]; then
    GPU="$("$PYTHON" -c "
import sys
sys.path.insert(0, '$ROOT')
from cft.resources import pick_gpu
idx = pick_gpu()
print(idx if idx is not None else '')
")"
  fi
  if [[ -n "$GPU" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
  fi
fi

export CFT_CPU_WORKERS="$CPU_WORKERS"
export CFT_GPU="${CUDA_VISIBLE_DEVICES:-$GPU}"
export CFT_GPU_CANDIDATES="$GPU_CANDIDATES"
export CFT_MIN_GPU_FREE_MIB="$MIN_FREE"
export OMP_NUM_THREADS="$CPU_WORKERS"
export MKL_NUM_THREADS="$CPU_WORKERS"
export OPENBLAS_NUM_THREADS="$CPU_WORKERS"
export NUMEXPR_NUM_THREADS="$CPU_WORKERS"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

MASTER_LOG="$LOGDIR/full.log"
{
  echo "===== launch $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  echo "python=$PYTHON"
  echo "cwd=$ROOT"
  echo "cpu_workers=$CPU_WORKERS affinity=$CPU_AFFINITY nice=$NICE"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "profile=$PROFILE reps=$REPS args=$*"
  nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader || true
} >> "$MASTER_LOG"

TASKSET=()
if command -v taskset >/dev/null 2>&1; then
  TASKSET=(taskset -c "$CPU_AFFINITY")
fi
NICECMD=()
if command -v nice >/dev/null 2>&1; then
  NICECMD=(nice -n "$NICE")
fi

# nohup ignores SIGHUP; nice/taskset exec into python so $! is the orchestrator.
cd "$ROOT"
nohup "${NICECMD[@]}" "${TASKSET[@]}" "$PYTHON" -u "$ROOT/scripts/run_full.py" \
      --python "$PYTHON" --profile "$PROFILE" --reps "$REPS" "$@" \
      >> "$MASTER_LOG" 2>&1 < /dev/null &
echo $! > "$pidfile"
disown $! 2>/dev/null || true

echo "started pid $(cat "$pidfile")"
echo "  log:    $MASTER_LOG"
echo "  status: $LOGDIR/status.json"
echo "  gpu:    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "  cpus:   $CPU_AFFINITY  (n=$CPU_WORKERS)"
echo "  stop:   bash $ROOT/scripts/stop_background.sh"
echo "  follow: tail -f $MASTER_LOG"
