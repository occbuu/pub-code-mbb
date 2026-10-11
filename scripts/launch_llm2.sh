#!/usr/bin/env bash
# Second LLM (Mistral) on GPUs 0 and 1, ~36 CPUs, survives logout.
#
#   bash scripts/launch_llm2.sh
#   tail -f logs/llm2.log
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

PYTHON="${CFT_PYTHON:-/workspace/anlk/HaoLee/.venv/bin/python}"
CPU_WORKERS="${CFT_CPU_WORKERS:-36}"
CPU_AFFINITY="${CFT_CPU_AFFINITY:-24-59}"   # 36 logical CPUs
NICE="${CFT_NICE:-10}"

pidfile="$LOGDIR/llm2.pid"
if [[ -f "$pidfile" ]]; then
  old="$(cat "$pidfile" || true)"
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null; then
    echo "llm2 already running as pid $old"
    exit 1
  fi
fi

export CFT_CPU_WORKERS="$CPU_WORKERS"
export CFT_GPU_CANDIDATES="${CFT_GPU_CANDIDATES:-0,1}"
export CFT_MIN_GPU_FREE_MIB="${CFT_MIN_GPU_FREE_MIB:-9000}"
export CFT_LLM2_MODEL="${CFT_LLM2_MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Do NOT pin a single CUDA_VISIBLE_DEVICES; the orchestrator spawns one process per GPU.
unset CUDA_VISIBLE_DEVICES || true

MASTER_LOG="$LOGDIR/llm2.log"
{
  echo "===== llm2 launch $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  echo "python=$PYTHON model=$CFT_LLM2_MODEL"
  echo "cpus=$CPU_WORKERS affinity=$CPU_AFFINITY gpus=0,1"
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

cd "$ROOT"
nohup "${NICECMD[@]}" "${TASKSET[@]}" "$PYTHON" -u "$ROOT/scripts/run_llm2.py" \
      >> "$MASTER_LOG" 2>&1 < /dev/null &
echo $! > "$pidfile"
disown $! 2>/dev/null || true

echo "started llm2 pid $(cat "$pidfile")"
echo "  log:    $MASTER_LOG"
echo "  gpus:   0 and 1 (separate workers)"
echo "  cpus:   $CPU_AFFINITY (n=$CPU_WORKERS)"
echo "  model:  $CFT_LLM2_MODEL"
echo "  follow: tail -f $MASTER_LOG logs/llm2_gpu0.log logs/llm2_gpu1.log"
