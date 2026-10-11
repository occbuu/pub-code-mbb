"""Process / device budget for this machine.

The shared workstation has 96 logical CPUs and 8 GPUs.  Paper 13 is supposed to
sit in a *slice* of that: about 20-30 CPU threads and a single GPU, and to keep
running after the Cursor session that launched it is closed.

Environment
-----------
``CFT_CPU_WORKERS``          process/thread budget (default 24)
``CFT_GPU``                  GPU index, or ``auto`` to pick the emptiest card
``CUDA_VISIBLE_DEVICES``     if already set, we honour it and do not retarget
``CFT_MIN_GPU_FREE_MIB``     refuse to claim a card with less free memory
"""

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone

THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

DEFAULT_CPU_WORKERS = 24
DEFAULT_MIN_GPU_FREE_MIB = 18_000
DEFAULT_GPU_CANDIDATES = (0, 1)   # lab policy: never touch GPUs 2-7


def cpu_workers(default: int = DEFAULT_CPU_WORKERS) -> int:
    raw = os.environ.get("CFT_CPU_WORKERS")
    n = int(raw) if raw else int(default)
    return max(1, min(n, os.cpu_count() or n))


def apply_thread_limits(n: int | None = None, *, override: bool = True) -> int:
    """Cap BLAS/OpenMP threads.  Must run before numpy is imported to be fully
    effective; the launcher sets the same variables in the process environment."""
    n = cpu_workers() if n is None else max(1, int(n))
    for key in THREAD_KEYS:
        if override or key not in os.environ:
            os.environ[key] = str(n)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    return n


def log(msg: str = "") -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] {msg}", flush=True)


def gpu_memory_table() -> list[dict]:
    """One row per NVIDIA GPU: index, used MiB, free MiB, util %."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        rows.append(dict(index=int(parts[0]), used_mib=int(float(parts[1])),
                         free_mib=int(float(parts[2])), util=int(float(parts[3]))))
    return rows


def allowed_gpus() -> tuple[int, ...]:
    """Physical GPU indices we are allowed to use.  Default: 0 and 1 only."""
    raw = os.environ.get("CFT_GPU_CANDIDATES", "").strip()
    if not raw:
        return DEFAULT_GPU_CANDIDATES
    return tuple(int(x) for x in raw.split(",") if x.strip() != "")


def gpu_candidates() -> list[dict]:
    allow = set(allowed_gpus())
    return [r for r in gpu_memory_table() if r["index"] in allow]


def pick_gpu(min_free_mib: int | None = None, *, force_auto: bool = False) -> int | None:
    """Return a physical GPU index among the allowed cards (default 0, 1)."""
    rows = gpu_candidates()
    if not rows:
        log(f"no allowed GPU visible (candidates={allowed_gpus()})")
        return None
    want = "auto" if force_auto else os.environ.get("CFT_GPU", "auto").strip().lower()
    if want not in ("", "auto"):
        idx = int(want)
        if idx not in set(allowed_gpus()):
            raise SystemExit(f"CFT_GPU={idx} is outside allowed cards {allowed_gpus()}")
        if any(r["index"] == idx for r in rows):
            return idx
        raise SystemExit(f"CFT_GPU={want} is not a visible NVIDIA device")
    rows = sorted(rows, key=lambda r: (-r["free_mib"], r["used_mib"], r["index"]))
    floor = int(os.environ.get("CFT_MIN_GPU_FREE_MIB",
                               min_free_mib or DEFAULT_MIN_GPU_FREE_MIB))
    best = rows[0]
    if best["free_mib"] < floor:
        log(f"warning: emptiest allowed GPU {best['index']} has only "
            f"{best['free_mib']} MiB free (want >= {floor})")
    return best["index"]


def pin_gpu(index: int | None = None) -> int | None:
    """Restrict this process (and its children) to one GPU.

    If ``CUDA_VISIBLE_DEVICES`` is already set *and* that card still has enough
    free memory we keep it.  Otherwise we drop the pin and pick the emptiest
    card -- the GPU stage of a long CPU run would otherwise land on a card that
    filled up while Yelp/NMF was running.
    """
    rows = gpu_memory_table()
    allow = set(allowed_gpus())
    floor = int(os.environ.get("CFT_MIN_GPU_FREE_MIB", DEFAULT_MIN_GPU_FREE_MIB))
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    if vis:
        try:
            vis_idx = int(vis)
        except ValueError:
            vis_idx = None
        row = next((r for r in rows if r["index"] == vis_idx), None) if vis_idx is not None else None
        if vis_idx is not None and vis_idx not in allow:
            log(f"CUDA_VISIBLE_DEVICES={vis} is outside allowed {tuple(sorted(allow))}; re-picking")
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ["CFT_GPU"] = "auto"
        elif row and row["free_mib"] >= floor:
            log(f"GPU already pinned via CUDA_VISIBLE_DEVICES={vis} "
                f"({row['free_mib']} MiB free)")
            os.environ["CFT_GPU_PHYSICAL"] = str(vis_idx)
            return vis_idx
        else:
            log(f"CUDA_VISIBLE_DEVICES={vis} is too full "
                f"({None if row is None else row['free_mib']} MiB free); re-picking")
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ["CFT_GPU"] = "auto"

    if index is not None and int(index) not in allow:
        log(f"requested GPU {index} outside allowed {tuple(sorted(allow))}; using auto")
        index = None
    idx = pick_gpu(force_auto=True) if index is None else int(index)
    if idx is None:
        log("no NVIDIA GPU visible; GPU stages will fall back to CPU or skip")
        return None
    os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
    os.environ["CFT_GPU_PHYSICAL"] = str(idx)
    log(f"pinned CUDA_VISIBLE_DEVICES={idx} (physical index)")
    return idx


def wait_and_pin_gpu(min_free_mib: int = 9_000, poll_s: int = 30,
                     timeout_s: int = 6 * 3600) -> int | None:
    """Block until some card has ``min_free_mib``, then pin it.  The launcher
    may have pinned a GPU hours earlier that is now occupied."""
    import time
    os.environ["CFT_GPU"] = "auto"
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    t0 = time.time()
    while True:
        rows = gpu_candidates()
        best = max(rows, key=lambda r: r["free_mib"]) if rows else None
        if best and best["free_mib"] >= min_free_mib:
            os.environ["CFT_MIN_GPU_FREE_MIB"] = str(min_free_mib)
            return pin_gpu()
        waited = int(time.time() - t0)
        if waited >= timeout_s:
            log(f"timed out waiting for a GPU with >= {min_free_mib} MiB free")
            return pin_gpu()
        msg = (f"gpu{best['index']} {best['free_mib']} MiB free"
               if best else "no GPU")
        log(f"waiting for GPU >= {min_free_mib} MiB ({msg}); slept {waited}s")
        time.sleep(poll_s)


def worker_init_single_thread() -> None:
    """Initializer for ProcessPool workers: do not nest a BLAS pool inside each."""
    apply_thread_limits(1, override=True)


@contextmanager
def process_pool(n_jobs: int | None = None):
    from concurrent.futures import ProcessPoolExecutor
    n = cpu_workers() if n_jobs is None else max(1, int(n_jobs))
    if n <= 1:
        yield None
        return
    with ProcessPoolExecutor(max_workers=n, initializer=worker_init_single_thread) as ex:
        yield ex


def banner(stage: str = "") -> None:
    n = cpu_workers()
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")
    phys = os.environ.get("CFT_GPU_PHYSICAL", "")
    gpus = gpu_memory_table()
    log(f"stage={stage or '-'}  python={sys.executable}")
    log(f"cpu_workers={n}  OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '?')}  "
        f"CUDA_VISIBLE_DEVICES={vis}" + (f" (physical {phys})" if phys else ""))
    if gpus:
        bits = ", ".join(f"gpu{r['index']}:{r['free_mib']}MiB-free/{r['util']}%"
                         for r in gpus)
        log(f"gpu_free  {bits}")
