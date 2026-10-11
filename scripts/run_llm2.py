"""Second LLM arm -- family robustness for the method-halo finding.

Qwen2.5-7B-Instruct is the primary LLM instrument.  Halo (heterotrait-
monomethod correlation) is high for that model and for the encoder distilled
from it.  To treat that as a property of *LLM-as-instrument* rather than of
Qwen, this script scores the same llm_core documents with a second 7B
instruct model from a different lineage (Mistral-7B-Instruct-v0.2), same
codebook prompt, zeroshot, three seeds.

Two workers run in parallel on physical GPUs 0 and 1.  Each shard is
checkpointed every 32 documents under derived/llm2_blocks/.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

try:
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from cft.resources import apply_thread_limits, banner, gpu_memory_table, log  # noqa: E402

DERIVED = ROOT / "derived"
BLOCKS = DERIVED / "llm2_blocks"
BLOCKS.mkdir(parents=True, exist_ok=True)
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)

MODEL = os.environ.get("CFT_LLM2_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
SEEDS = [1, 2, 3]
VARIANT = "zeroshot"
CHUNK = 32
GPUS = (0, 1)
MIN_FREE = int(os.environ.get("CFT_MIN_GPU_FREE_MIB", "9000"))
N_CORE = 400
TEMPERATURE = 0.7


def _sample_per_corpus(df, n: int, seed: int):
    import pandas as pd
    if df.empty or n <= 0:
        return df.iloc[0:0].copy()
    parts = []
    for _, g in df.groupby("corpus", group_keys=False):
        parts.append(g.sample(min(len(g), n), random_state=seed))
    return pd.concat(parts, ignore_index=True)


def wait_gpus(gpus=GPUS, min_free=MIN_FREE, poll_s=30, timeout_s=6 * 3600):
    t0 = time.time()
    while True:
        rows = {r["index"]: r for r in gpu_memory_table()}
        bits = []
        ok = True
        for g in gpus:
            free = int(rows.get(g, {}).get("free_mib", 0))
            bits.append(f"gpu{g}:{free}MiB")
            if free < min_free:
                ok = False
        if ok:
            log(f"gpus ready  {', '.join(bits)}")
            return
        waited = int(time.time() - t0)
        if waited >= timeout_s:
            log(f"timed out waiting for GPUs {gpus}; proceeding anyway ({', '.join(bits)})")
            return
        log(f"waiting for GPU 0 and 1 >= {min_free} MiB ({', '.join(bits)}); slept {waited}s")
        time.sleep(poll_s)


def core_docs():
    import pandas as pd
    bench = pd.read_parquet(DERIVED / "bench_for_colab.parquet")
    core = _sample_per_corpus(bench[bench.split == "llm_core"], N_CORE, 42)
    core = core.sort_values("doc_id").reset_index(drop=True)
    log(f"llm2 core={len(core)} corpora={sorted(core.corpus.unique().tolist())}")
    return core


def shard_docs(docs, shard: int, n_shards: int = 2):
    return docs.iloc[shard::n_shards].reset_index(drop=True)


def adapt_quant(cfg: dict, physical: int) -> dict:
    row = next((r for r in gpu_memory_table() if r["index"] == physical), None)
    free = int(row["free_mib"]) if row else 0
    if free < 18_000:
        cfg["four_bit"] = True
        cfg["batch"] = 1 if free < 12_000 else 2
        log(f"GPU {physical} has {free} MiB free; 4-bit, batch={cfg['batch']}")
    else:
        cfg["four_bit"] = False
        cfg["batch"] = min(int(cfg.get("batch", 4)), 4)
        log(f"GPU {physical} has {free} MiB free; bf16, batch={cfg['batch']}")
    return cfg


def run_block(docs, variant, seed, tag, cfg, backend, shard: int):
    import pandas as pd
    from cft.llm import LLMRunConfig, LLMScorer

    fn = BLOCKS / f"llm2_{tag}_{variant}_s{seed}_g{shard}.parquet"
    pending = docs.reset_index(drop=True)
    parts = []
    if pending.empty:
        log(f"skip {fn.name}: 0 docs")
        return pending
    if fn.exists():
        done = pd.read_parquet(fn)
        have = set(done["doc_id"].astype(str))
        pending = pending[~pending["doc_id"].astype(str).isin(have)].reset_index(drop=True)
        parts.append(done)
        if pending.empty:
            log(f"skip {fn.name} ({len(done)} docs)")
            return done
        log(f"resume {fn.name}: {len(pending)} remaining")

    llm_cfg = LLMRunConfig(
        model_id=cfg["model"], backend=backend, variant=variant, seed=seed,
        temperature=TEMPERATURE, batch_size=cfg["batch"],
        load_in_4bit=cfg["four_bit"], tensor_parallel_size=1,
        max_model_len=2048, max_new_tokens=220)
    scorer = LLMScorer(llm_cfg)
    t0 = time.time()
    for i in range(0, len(pending), CHUNK):
        sl = pending.iloc[i:i + CHUNK]
        res = scorer.score(sl, keep_raw=True)
        res["block"] = tag
        res["model_id"] = cfg["model"]
        res["shard"] = shard
        parts.append(res)
        out = pd.concat(parts, ignore_index=True)
        out.to_parquet(fn, index=False)
        log(f"{fn.name}: {min(i + CHUNK, len(pending))}/{len(pending)} "
            f"parsed {100 * out.parsed.mean():.1f}% ({time.time() - t0:.0f}s)")
    return pd.concat(parts, ignore_index=True)


def worker(shard: int, physical: int):
    apply_thread_limits(4)
    banner(f"llm2-gpu{physical}")
    import pandas as pd
    from cft.llm import prompt_hash

    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    log(f"worker shard={shard} physical={physical} CUDA_VISIBLE_DEVICES={vis} model={MODEL}")
    cfg = dict(model=MODEL, backend="transformers", batch=4, four_bit=False)
    adapt_quant(cfg, physical)

    docs = shard_docs(core_docs(), shard, n_shards=len(GPUS))
    log(f"shard {shard}: {len(docs)} docs")
    for seed in SEEDS:
        run_block(docs, VARIANT, seed, "core", cfg, "transformers", shard)

    (BLOCKS / f"llm2_worker_g{shard}.json").write_text(json.dumps(dict(
        shard=shard, physical=physical, model=MODEL,
        prompt_hash=prompt_hash(VARIANT), n_docs=int(len(docs)),
        four_bit=cfg["four_bit"], batch=cfg["batch"],
        finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    ), indent=2))
    log(f"worker shard {shard} done")


def merge_and_analyse():
    import pandas as pd
    parts = []
    for seed in SEEDS:
        shards = []
        for g in range(len(GPUS)):
            fn = BLOCKS / f"llm2_core_{VARIANT}_s{seed}_g{g}.parquet"
            if not fn.exists():
                raise SystemExit(f"missing shard {fn}")
            shards.append(pd.read_parquet(fn))
        block = pd.concat(shards, ignore_index=True)
        block = block.drop_duplicates("doc_id", keep="last")
        merged = BLOCKS / f"llm2_core_{VARIANT}_s{seed}.parquet"
        block.to_parquet(merged, index=False)
        log(f"merged {merged.name} n={len(block)} parse={float(block.parsed.mean()):.3f}")
        parts.append(block)
    all_ = pd.concat(parts, ignore_index=True)
    out = DERIVED / "llm2_scores.parquet"
    all_.drop(columns=["raw"], errors="ignore").to_parquet(out, index=False)
    if "raw" in all_.columns:
        all_[["doc_id", "run", "block", "raw", "model_id"]].to_parquet(
            DERIVED / "llm2_scores_raw.parquet", index=False)
    prov = dict(
        model=MODEL, variant=VARIANT, seeds=SEEDS, n=int(len(all_)),
        n_docs=int(all_.doc_id.nunique()),
        parse_rate=float(all_.parsed.mean()),
        run_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        gpus=list(GPUS),
    )
    (DERIVED / "llm2_provenance.json").write_text(json.dumps(prov, indent=2))
    log(json.dumps(prov))
    ana = [sys.executable, str(ROOT / "scripts" / "run_llm2_analysis.py")]
    log("running " + " ".join(ana))
    subprocess.check_call(ana, cwd=str(ROOT))


def orchestrate():
    apply_thread_limits()
    banner("llm2")
    wait_gpus()
    py = sys.executable
    procs = []
    for shard, gpu in enumerate(GPUS):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["CFT_GPU_PHYSICAL"] = str(gpu)
        env["PYTHONUNBUFFERED"] = "1"
        logf = LOGS / f"llm2_gpu{gpu}.log"
        fh = logf.open("ab")
        fh.write(f"\n----- {time.strftime('%Y-%m-%dT%H:%M:%SZ')} worker gpu{gpu} -----\n"
                 .encode())
        fh.flush()
        cmd = [py, "-u", str(Path(__file__)), "--worker", "--shard", str(shard),
               "--gpu", str(gpu)]
        log(f"spawn {' '.join(cmd)}")
        p = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=fh,
                             stderr=subprocess.STDOUT, start_new_session=True)
        procs.append((gpu, p, fh))
    rc = 0
    for gpu, p, fh in procs:
        code = p.wait()
        fh.close()
        log(f"worker gpu{gpu} exited {code}")
        if code != 0:
            rc = code
    if rc != 0:
        raise SystemExit(f"llm2 worker failed with code {rc}")
    merge_and_analyse()
    log("llm2 family arm done")


def main():
    if "--worker" in sys.argv:
        shard = int(sys.argv[sys.argv.index("--shard") + 1])
        gpu = int(sys.argv[sys.argv.index("--gpu") + 1])
        worker(shard, gpu)
        return
    orchestrate()


if __name__ == "__main__":
    main()
