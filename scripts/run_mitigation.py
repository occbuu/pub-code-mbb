"""Paper B, Study 2 -- can the construct collapse be undone at elicitation time?

Scores the same documents under four elicitation conditions (see cft/elicit.py):

    joint     five constructs in one call          (baseline, already run)
    isolated  one construct per call
    evidence  one construct per call + verbatim span or ABSENT
    gate      presence call, then level call only if present

Two workers on physical GPUs 0 and 1, resumable in 32-document chunks, same
shape as scripts/run_llm2.py.  Defaults are sized so the whole study is about
the same number of generations as the original LLM arm:

    N_CORE docs/corpus x 4 corpora x 5 constructs
      isolated   1 call  each
      evidence   1 call  each
      gate       1 call  each + 1 more where present

Environment knobs:
    CFT_MIT_MODEL     model id (default: the primary Qwen instrument)
    CFT_MIT_N         documents per corpus (default 250)
    CFT_MIT_CONDS     comma-separated subset of isolated,evidence,gate
    CFT_MIT_SEED      seed for the single run per condition (default 1)
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
BLOCKS = DERIVED / "mitigation_blocks"
BLOCKS.mkdir(parents=True, exist_ok=True)
LOGS = ROOT / "logs"
LOGS.mkdir(exist_ok=True)

MODEL = os.environ.get("CFT_MIT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
N_CORE = int(os.environ.get("CFT_MIT_N", "250"))
SEED = int(os.environ.get("CFT_MIT_SEED", "1"))
CONDS = [c.strip() for c in
         os.environ.get("CFT_MIT_CONDS", "isolated,evidence,gate").split(",")
         if c.strip()]
CHUNK = 32
GPUS = (0, 1)
MIN_FREE = int(os.environ.get("CFT_MIN_GPU_FREE_MIB", "9000"))
TEMPERATURE = 0.7


def _sample_per_corpus(df, n: int, seed: int):
    import pandas as pd
    if df.empty or n <= 0:
        return df.iloc[0:0].copy()
    parts = [g.sample(min(len(g), n), random_state=seed)
             for _, g in df.groupby("corpus", group_keys=False)]
    return pd.concat(parts, ignore_index=True)


def core_docs():
    import pandas as pd
    bench = pd.read_parquet(DERIVED / "bench_for_colab.parquet")
    core = _sample_per_corpus(bench[bench.split == "llm_core"], N_CORE, 42)
    core = core.sort_values("doc_id").reset_index(drop=True)
    log(f"mitigation core={len(core)} corpora={sorted(core.corpus.unique().tolist())}")
    return core


def wait_gpus(gpus=GPUS, min_free=MIN_FREE, poll_s=30, timeout_s=6 * 3600):
    t0 = time.time()
    while True:
        rows = {r["index"]: r for r in gpu_memory_table()}
        bits, ok = [], True
        for g in gpus:
            free = int(rows.get(g, {}).get("free_mib", 0))
            bits.append(f"gpu{g}:{free}MiB")
            if free < min_free:
                ok = False
        if ok:
            log(f"gpus ready  {', '.join(bits)}")
            return
        if int(time.time() - t0) >= timeout_s:
            log(f"timed out waiting for GPUs {gpus}; proceeding ({', '.join(bits)})")
            return
        log(f"waiting for GPUs >= {min_free} MiB ({', '.join(bits)})")
        time.sleep(poll_s)


def adapt_quant(cfg: dict, physical: int) -> dict:
    row = next((r for r in gpu_memory_table() if r["index"] == physical), None)
    free = int(row["free_mib"]) if row else 0
    if free < 18_000:
        cfg["four_bit"] = True
        cfg["batch"] = 1 if free < 12_000 else 2
    else:
        cfg["four_bit"] = False
        cfg["batch"] = min(int(cfg.get("batch", 4)), 4)
    log(f"GPU {physical}: {free} MiB free -> "
        f"{'4-bit' if cfg['four_bit'] else 'bf16'}, batch={cfg['batch']}")
    return cfg


def _scorer(cfg):
    from cft.llm import LLMRunConfig, LLMScorer
    return LLMScorer(LLMRunConfig(
        model_id=cfg["model"], backend="transformers", variant="zeroshot",
        seed=SEED, temperature=TEMPERATURE, batch_size=cfg["batch"],
        load_in_4bit=cfg["four_bit"], tensor_parallel_size=1,
        max_model_len=2048, max_new_tokens=200))


def run_condition(docs, condition: str, cfg, shard: int):
    """One row per (doc, construct); resumable."""
    import pandas as pd
    from cft.codebook import CONSTRUCTS
    from cft import elicit

    fn = BLOCKS / f"mit_{condition}_s{SEED}_g{shard}.parquet"
    pairs = pd.DataFrame([dict(doc_id=d.doc_id, text=d.text, corpus=d.corpus,
                               construct=c)
                          for d in docs.itertuples() for c in CONSTRUCTS])
    parts = []
    if fn.exists():
        done = pd.read_parquet(fn)
        key_done = set(zip(done["doc_id"].astype(str), done["construct"]))
        mask = [(str(a), b) not in key_done
                for a, b in zip(pairs["doc_id"], pairs["construct"])]
        parts.append(done)
        pairs = pairs[mask].reset_index(drop=True)
        if pairs.empty:
            log(f"skip {fn.name} ({len(done)} rows)")
            return done
        log(f"resume {fn.name}: {len(pairs)} of "
            f"{len(done) + len(pairs)} rows remaining")

    sc = _scorer(cfg)
    t0 = time.time()
    for i in range(0, len(pairs), CHUNK):
        sl = pairs.iloc[i:i + CHUNK].reset_index(drop=True)
        if condition == "gate":
            p1 = [elicit.build_gate_presence(t, c)
                  for t, c in zip(sl["text"], sl["construct"])]
            raw1 = sc.generate(p1)
            present = [elicit.parse_presence(x) for x in raw1]
            idx = [k for k, v in enumerate(present) if v]
            raw2 = {}
            if idx:
                p2 = [elicit.build_gate_level(sl["text"].iloc[k],
                                              sl["construct"].iloc[k]) for k in idx]
                gen2 = sc.generate(p2)
                raw2 = dict(zip(idx, gen2))
            rows = []
            for k in range(len(sl)):
                pol = elicit.parse_level(raw2[k]) if k in raw2 else float("nan")
                rows.append(dict(
                    doc_id=sl["doc_id"].iloc[k], corpus=sl["corpus"].iloc[k],
                    construct=sl["construct"].iloc[k], condition=condition,
                    parsed=present[k] is not None,
                    sal=1.0 if present[k] else 0.0,
                    pol=pol if present[k] else float("nan"),
                    quote=None, quote_verbatim=False,
                    raw=(raw1[k] or "") + " || " + str(raw2.get(k, "")),
                ))
        else:
            builder = elicit.BUILDERS[condition]
            prompts = [builder(t, c) for t, c in zip(sl["text"], sl["construct"])]
            raw = sc.generate(prompts)
            rows = []
            for k, txt in enumerate(raw):
                par = elicit.parse_single(txt)
                q = par.get("quote")
                rows.append(dict(
                    doc_id=sl["doc_id"].iloc[k], corpus=sl["corpus"].iloc[k],
                    construct=sl["construct"].iloc[k], condition=condition,
                    parsed=par["parsed"], sal=par["sal"], pol=par["pol"],
                    quote=q if isinstance(q, str) else None,
                    quote_verbatim=elicit.quote_is_verbatim(q, sl["text"].iloc[k]),
                    raw=txt))
        parts.append(pd.DataFrame(rows))
        out = pd.concat(parts, ignore_index=True)
        out.to_parquet(fn, index=False)
        log(f"{fn.name}: {min(i + CHUNK, len(pairs))}/{len(pairs)} rows, "
            f"parsed {100 * out.parsed.mean():.1f}%, "
            f"coverage {100 * out.pol.notna().mean():.1f}% ({time.time() - t0:.0f}s)")
    return pd.concat(parts, ignore_index=True)


def worker(shard: int, physical: int):
    apply_thread_limits(4)
    banner(f"mitigation-gpu{physical}")
    log(f"worker shard={shard} gpu={physical} model={MODEL} conditions={CONDS}")
    cfg = adapt_quant(dict(model=MODEL, batch=4, four_bit=False), physical)
    docs = core_docs().iloc[shard::len(GPUS)].reset_index(drop=True)
    log(f"shard {shard}: {len(docs)} docs x 5 constructs")
    for cond in CONDS:
        run_condition(docs, cond, cfg, shard)
    log(f"worker shard {shard} done")


def merge_and_analyse():
    import pandas as pd
    parts = []
    for cond in CONDS:
        shards = []
        for g in range(len(GPUS)):
            fn = BLOCKS / f"mit_{cond}_s{SEED}_g{g}.parquet"
            if not fn.exists():
                raise SystemExit(f"missing shard {fn}")
            shards.append(pd.read_parquet(fn))
        blk = pd.concat(shards, ignore_index=True).drop_duplicates(
            ["doc_id", "construct"], keep="last")
        log(f"{cond}: n={len(blk)} parse={float(blk.parsed.mean()):.3f} "
            f"coverage={float(blk.pol.notna().mean()):.3f}")
        parts.append(blk)
    all_ = pd.concat(parts, ignore_index=True)
    all_.drop(columns=["raw"], errors="ignore").to_parquet(
        DERIVED / "mitigation_scores.parquet", index=False)
    all_[["doc_id", "construct", "condition", "raw"]].to_parquet(
        DERIVED / "mitigation_scores_raw.parquet", index=False)
    prov = dict(model=MODEL, conditions=CONDS, seed=SEED,
                n_docs=int(all_.doc_id.nunique()), n_rows=int(len(all_)),
                temperature=TEMPERATURE,
                parse_rate=float(all_.parsed.mean()),
                run_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (DERIVED / "mitigation_provenance.json").write_text(json.dumps(prov, indent=2))
    log(json.dumps(prov))
    ana = [sys.executable, str(ROOT / "scripts" / "run_mitigation_analysis.py")]
    log("running " + " ".join(ana))
    subprocess.check_call(ana, cwd=str(ROOT))


def orchestrate():
    apply_thread_limits()
    banner("mitigation")
    wait_gpus()
    procs = []
    for shard, gpu in enumerate(GPUS):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["CFT_GPU_PHYSICAL"] = str(gpu)
        env["PYTHONUNBUFFERED"] = "1"
        fh = (LOGS / f"mitigation_gpu{gpu}.log").open("ab")
        fh.write(f"\n----- {time.strftime('%Y-%m-%dT%H:%M:%SZ')} gpu{gpu} -----\n"
                 .encode())
        fh.flush()
        cmd = [sys.executable, "-u", str(Path(__file__)), "--worker",
               "--shard", str(shard), "--gpu", str(gpu)]
        log(f"spawn {' '.join(cmd)}")
        procs.append((gpu, subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=fh,
                                            stderr=subprocess.STDOUT,
                                            start_new_session=True), fh))
    rc = 0
    for gpu, p, fh in procs:
        code = p.wait()
        fh.close()
        log(f"worker gpu{gpu} exited {code}")
        rc = rc or code
    if rc:
        raise SystemExit(f"mitigation worker failed with code {rc}")
    merge_and_analyse()
    log("mitigation study done")


if __name__ == "__main__":
    if "--worker" in sys.argv:
        worker(int(sys.argv[sys.argv.index("--shard") + 1]),
               int(sys.argv[sys.argv.index("--gpu") + 1]))
    else:
        orchestrate()
