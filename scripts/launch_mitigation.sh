#!/usr/bin/env bash
# Paper B Study 2 -- elicitation interventions on lab GPUs 0 and 1.
# Survives logout. After scoring, rebuilds both paper drafts.
#
#   bash scripts/launch_mitigation.sh
#   tail -f logs/mitigation.log logs/mitigation_gpu0.log logs/mitigation_gpu1.log
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOGDIR="$ROOT/logs"
mkdir -p "$LOGDIR"

PYTHON="${CFT_PYTHON:-/workspace/anlk/HaoLee/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

CPU_WORKERS="${CFT_CPU_WORKERS:-8}"
CPU_AFFINITY="${CFT_CPU_AFFINITY:-24-31}"
NICE="${CFT_NICE:-10}"

pidfile="$LOGDIR/mitigation.pid"
if [[ -f "$pidfile" ]]; then
  old="$(cat "$pidfile" || true)"
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null; then
    echo "mitigation already running as pid $old"
    exit 1
  fi
fi

export CFT_CPU_WORKERS="$CPU_WORKERS"
export CFT_GPU_CANDIDATES="${CFT_GPU_CANDIDATES:-0,1}"
export CFT_MIN_GPU_FREE_MIB="${CFT_MIN_GPU_FREE_MIB:-9000}"
export CFT_MIT_MODEL="${CFT_MIT_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export CFT_MIT_N="${CFT_MIT_N:-250}"
export CFT_MIT_CONDS="${CFT_MIT_CONDS:-isolated,evidence,gate}"
export CFT_MIT_SEED="${CFT_MIT_SEED:-1}"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset CUDA_VISIBLE_DEVICES || true

MASTER_LOG="$LOGDIR/mitigation.log"
{
  echo "===== mitigation launch $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  echo "python=$PYTHON model=$CFT_MIT_MODEL n=$CFT_MIT_N conds=$CFT_MIT_CONDS"
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
# Orchestrator: GPU scoring, then analysis, then both paper drafts.
nohup "${NICECMD[@]}" "${TASKSET[@]}" bash -c "
  set -euo pipefail
  '$PYTHON' -u '$ROOT/scripts/run_mitigation.py'
  '$PYTHON' -u '$ROOT/build/paper_a.py'
  '$PYTHON' -u '$ROOT/build/paper_b.py'
  if command -v node >/dev/null 2>&1; then
    export NODE_PATH="${NODE_PATH:-$ROOT/build/node_modules}"
    node '$ROOT/build/render_docx.js' '$ROOT/build/paper_a.json' '$ROOT/Paper13_PaperA_DRAFT.docx'
    node '$ROOT/build/render_docx.js' '$ROOT/build/paper_b.json' '$ROOT/Paper13_PaperB_DRAFT.docx'
  fi
" >> "$MASTER_LOG" 2>&1 < /dev/null &
echo $! > "$pidfile"
disown $! 2>/dev/null || true

echo "started mitigation pid $(cat "$pidfile")"
echo "  log:    $MASTER_LOG"
echo "  gpus:   0 and 1"
echo "  model:  $CFT_MIT_MODEL"
echo "  n/corp: $CFT_MIT_N"
echo "  follow: tail -f $MASTER_LOG logs/mitigation_gpu0.log logs/mitigation_gpu1.log"
echo "  after:  Paper13_PaperA_DRAFT.docx and Paper13_PaperB_DRAFT.docx"
echo "  stop:   kill \$(cat $pidfile)"
