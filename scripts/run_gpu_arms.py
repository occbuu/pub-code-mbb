"""Local GPU runner for Notebook 02 (arms 3 and 4).

Replaces the Colab notebook: same protocol, same outputs written into
``derived/``, checkpointed after every block so a crash costs one chunk rather
than the run.  Honours ``CUDA_VISIBLE_DEVICES`` (the background launcher pins
exactly one physical GPU) and never calls ``device_map="auto"`` across cards.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from cft.resources import (apply_thread_limits, banner, gpu_memory_table,  # noqa: E402
                             log, wait_and_pin_gpu)

# Wait for a card with room for 4-bit 7B (~6 GB) *before* torch is imported.
wait_and_pin_gpu(min_free_mib=9_000, poll_s=30)
apply_thread_limits(4)
banner("gpu")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cft.codebook import CONSTRUCTS  # noqa: E402
from cft.encoder import EncoderConfig, EncoderScorer, llm_consensus_targets  # noqa: E402
from cft.llm import (LLMRunConfig, LLMScorer, make_perturbations,  # noqa: E402
                     prompt_hash)
from cft.reliability import prompt_sensitivity, run_stability  # noqa: E402
from cft.report import wide_to_long  # noqa: E402

DERIVED = ROOT / "derived"
OUT = DERIVED
BLOCKS = DERIVED / "llm_blocks"
BLOCKS.mkdir(parents=True, exist_ok=True)

PROFILES = {
    "a5000": dict(model="Qwen/Qwen2.5-7B-Instruct", backend="transformers",
                  n_core=400, n_bench_extra=800, n_train=4000, n_perturb=200,
                  n_multi=300, batch=4, four_bit=False),
    "a100": dict(model="Qwen/Qwen2.5-7B-Instruct", backend="vllm",
                 n_core=400, n_bench_extra=800, n_train=4000, n_perturb=200,
                 n_multi=300, batch=64, four_bit=False),
    "l4": dict(model="Qwen/Qwen2.5-7B-Instruct", backend="vllm",
               n_core=250, n_bench_extra=500, n_train=2500, n_perturb=150,
               n_multi=200, batch=32, four_bit=False),
    "t4": dict(model="Qwen/Qwen2.5-3B-Instruct", backend="transformers",
               n_core=120, n_bench_extra=200, n_train=1200, n_perturb=80,
               n_multi=120, batch=4, four_bit=True),
}

PROMPT_VARIANTS = ["zeroshot", "fewshot", "cot"]
SEEDS = [1, 2, 3]
TEMPERATURE = 0.7
CHUNK = 32   # write often so a SIGKILL loses at most one small chunk
MT_MODEL = "facebook/nllb-200-distilled-600M"

SCORER = None


def _backend(cfg: dict) -> str:
    want = cfg["backend"]
    if want == "vllm":
        try:
            import vllm  # noqa: F401
            return "vllm"
        except Exception:
            log("vllm not installed; falling back to transformers")
            return "transformers"
    return want


def _sample_per_corpus(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if df.empty or n <= 0:
        return df.iloc[0:0].copy()
    parts = []
    for _, g in df.groupby("corpus", group_keys=False):
        parts.append(g.sample(min(len(g), n), random_state=seed))
    return pd.concat(parts, ignore_index=True)


def _cfg_for(variant: str, seed: int, cfg: dict, backend: str) -> LLMRunConfig:
    return LLMRunConfig(
        model_id=cfg["model"], backend=backend, variant=variant, seed=seed,
        temperature=TEMPERATURE, batch_size=cfg["batch"],
        load_in_4bit=cfg["four_bit"], tensor_parallel_size=1,
        gpu_memory_utilization=0.88, max_model_len=2048)


def _reuse(scorer: LLMScorer, backend: str, cfg: dict) -> LLMScorer:
    global SCORER
    if SCORER is None:
        return scorer
    old_4bit = bool(getattr(SCORER.cfg, "load_in_4bit", False))
    if old_4bit != bool(cfg.get("four_bit")):
        log("quantisation changed; not reusing loaded LLM")
        free_llm()
        return scorer
    if backend == "vllm" and SCORER._llm is not None:
        scorer._llm = SCORER._llm
    elif backend == "transformers" and SCORER._pipe:
        scorer.tok = SCORER.tok
        scorer.model = SCORER.model
        scorer._pipe = True
    return scorer


def run_block(docs: pd.DataFrame, variant: str, seed: int, tag: str,
              cfg: dict, backend: str, keep_raw=True) -> pd.DataFrame:
    fn = BLOCKS / f"llm_{tag}_{variant}_s{seed}.parquet"
    pending = docs.reset_index(drop=True)
    if pending.empty:
        log(f"skip {tag}/{variant}/s{seed}: 0 docs")
        return pending
    parts = []
    if fn.exists():
        done = pd.read_parquet(fn)
        have = set(done["doc_id"].astype(str))
        pending = pending[~pending["doc_id"].astype(str).isin(have)].reset_index(drop=True)
        parts.append(done)
        if pending.empty:
            log(f"skip {fn.name} ({len(done)} docs)")
            return done
        log(f"resume {fn.name}: {len(pending)} docs remaining")
    scorer = _reuse(LLMScorer(_cfg_for(variant, seed, cfg, backend)), backend, cfg)
    global SCORER
    t0 = time.time()
    for i in range(0, len(pending), CHUNK):
        sl = pending.iloc[i:i + CHUNK]
        res = scorer.score(sl, keep_raw=keep_raw)
        res["block"] = tag
        parts.append(res)
        SCORER = scorer
        out = pd.concat(parts, ignore_index=True)
        out.to_parquet(fn, index=False)
        log(f"{tag}/{variant}/s{seed}: {min(i+CHUNK, len(pending))}/{len(pending)} "
            f"chunked, parsed {100*out.parsed.mean():.1f}% "
            f"({time.time()-t0:.0f}s)")
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out


def _empty_cuda():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def free_llm():
    global SCORER
    try:
        if SCORER is not None:
            for attr in ("model", "_llm", "tok"):
                if hasattr(SCORER, attr):
                    setattr(SCORER, attr, None)
            SCORER._pipe = None
    except Exception:
        pass
    SCORER = None
    _empty_cuda()
    log("released LLM weights")


def _physical_free_mib() -> tuple[int | None, int]:
    vis = os.environ.get("CFT_GPU_PHYSICAL") or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        vis_idx = int(str(vis).split(",")[0])
    except ValueError:
        vis_idx = None
    row = next((r for r in gpu_memory_table() if r["index"] == vis_idx), None)
    return vis_idx, int(row["free_mib"]) if row else 0


def adapt_quant(cfg: dict) -> dict:
    """Re-read nvidia-smi and drop to 4-bit if the card filled up mid-run."""
    vis_idx, free = _physical_free_mib()
    if free < 18_000:
        new_batch = 1 if free < 12_000 else min(int(cfg.get("batch", 2)), 2)
        changed = (not cfg.get("four_bit")) or int(cfg.get("batch", 4)) != new_batch
        cfg["four_bit"] = True
        cfg["batch"] = new_batch
        if changed:
            log(f"GPU {vis_idx} has {free} MiB free; 4-bit Qwen 7B, batch={cfg['batch']}")
            free_llm()
    return cfg


def translate(texts, tgt, bs=8, max_len=256):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MT_MODEL)
    tok.src_lang = "eng_Latn"
    mt = AutoModelForSeq2SeqLM.from_pretrained(MT_MODEL, dtype=torch.float16)
    mt.to("cuda:0").eval()
    gc_cfg = getattr(mt, "generation_config", None)
    if gc_cfg is not None:
        try:
            gc_cfg.max_length = None
        except Exception:
            gc_cfg.max_length = max_len + 32
    forced = tok.convert_tokens_to_ids(tgt)
    out = []
    try:
        for i in range(0, len(texts), bs):
            b = [t[:1200] for t in texts[i:i + bs]]
            enc = tok(b, return_tensors="pt", padding=True, truncation=True,
                      max_length=max_len).to("cuda:0")
            with torch.no_grad():
                gen = mt.generate(**enc, max_new_tokens=max_len,
                                  forced_bos_token_id=forced)
            out += tok.batch_decode(gen, skip_special_tokens=True)
            del gen, enc
            if (i // bs) % 10 == 0:
                log(f"translate {tgt} {i+len(b)}/{len(texts)}")
    finally:
        del mt, tok
        _empty_cuda()
    return out


def main(profile: str = "a5000", run_bertopic: bool = False):
    cfg = dict(PROFILES[profile])
    adapt_quant(cfg)
    backend = _backend(cfg)
    cfg["backend"] = backend
    log(f"profile={profile} model={cfg['model']} backend={backend} "
        f"batch={cfg['batch']} four_bit={cfg['four_bit']}")

    bench_path = DERIVED / "bench_for_colab.parquet"
    pool_path = DERIVED / "pool_for_colab.parquet"
    if not bench_path.exists() or not pool_path.exists():
        raise SystemExit("run scripts/run_prep.py first (missing Colab exports)")
    bench = pd.read_parquet(bench_path)
    pool = pd.read_parquet(pool_path)
    log(f"bench={bench.shape} pool={pool.shape}")

    core = _sample_per_corpus(bench[bench.split == "llm_core"], cfg["n_core"], 42)
    extra = _sample_per_corpus(bench[bench.split == "bench"], cfg["n_bench_extra"], 42)
    train_pool = pool[(pool.split == "pool") & (~pool.doc_id.isin(bench.doc_id))]
    n_corp = max(int(pool.corpus.nunique()), 1)
    train_docs = _sample_per_corpus(train_pool, cfg["n_train"] // n_corp, 7)
    log(f"core={len(core)} extra={len(extra)} train={len(train_docs)}")

    blocks = []
    for v in PROMPT_VARIANTS:
        for s in SEEDS:
            blocks.append(run_block(core, v, s, "core", cfg, backend))
    blocks.append(run_block(extra, "zeroshot", 1, "extra", cfg, backend))
    blocks.append(run_block(train_docs, "zeroshot", 1, "train", cfg, backend))
    llm_all = pd.concat(blocks, ignore_index=True)
    llm_all.drop(columns=["raw"], errors="ignore").to_parquet(
        OUT / "llm_scores.parquet", index=False)
    if "raw" in llm_all.columns:
        llm_all[["doc_id", "run", "block", "raw"]].to_parquet(
            OUT / "llm_scores_raw.parquet", index=False)
    log(f"llm parse rate {float(llm_all.parsed.mean()):.3f}  n={len(llm_all)}")

    core_scores = llm_all[llm_all.block == "core"].copy()
    core_scores["variant"] = core_scores.run.str.split("_").str[0]
    core_scores["seed"] = core_scores.run.str.split("_s").str[1]
    long = []
    for (v, s), g in core_scores.groupby(["variant", "seed"]):
        d = wide_to_long(g, method="LLM", kind="pol")
        d["variant"], d["run"] = v, f"{v}_s{s}"
        long.append(d)
    long = pd.concat(long, ignore_index=True)
    stab = run_stability(long[long.variant == "zeroshot"], by=("construct",))
    sens = prompt_sensitivity(
        long.groupby(["doc_id", "construct", "variant"], as_index=False)
            .value.mean().rename(columns={"variant": "prompt_variant"}),
        by=("construct",))
    stab.to_parquet(OUT / "llm_run_stability.parquet", index=False)
    sens.to_parquet(OUT / "llm_prompt_sensitivity.parquet", index=False)

    aud_docs = _sample_per_corpus(core, max(cfg["n_perturb"] // max(core.corpus.nunique(), 1), 1), 11)
    pert = make_perturbations(aud_docs[["doc_id", "text"]])
    pfn = OUT / "llm_perturbation.parquet"
    if pfn.exists():
        pert_scores = pd.read_parquet(pfn)
        log("skip llm_perturbation.parquet")
    else:
        parts = []
        for p, g in pert.groupby("perturbation"):
            r = run_block(g.reset_index(drop=True), "zeroshot", 1,
                          f"pert_{p}", cfg, backend, keep_raw=False)
            r["perturbation"] = p
            parts.append(r)
            log(f"perturbation {p} done")
        pert_scores = pd.concat(parts, ignore_index=True)
        pert_scores.to_parquet(pfn, index=False)

    # Translation needs the GPU; drop the LLM first (skip if texts already exist).
    mfn_text = OUT / "multilingual_texts.parquet"
    if mfn_text.exists():
        multi = pd.read_parquet(mfn_text)
        log("skip multilingual_texts.parquet")
    else:
        free_llm()
        src = bench[(bench.lang == "en") & (bench.split == "llm_core")]
        multi_src = src.sample(min(cfg["n_multi"], len(src), 400), random_state=3)
        rows = [multi_src.assign(variant_lang="en")]
        for code_, tag in [("vie_Latn", "vi"), ("zho_Hans", "zh")]:
            tr = multi_src.copy()
            tr["text"] = translate(list(multi_src["text"]), code_)
            tr["variant_lang"] = tag
            rows.append(tr)
        multi = pd.concat(rows, ignore_index=True)
        multi["doc_id_lang"] = multi["doc_id"] + "_" + multi["variant_lang"]
        multi.to_parquet(mfn_text, index=False)
        _empty_cuda()

    adapt_quant(cfg)
    mfn = OUT / "llm_multilingual.parquet"
    if mfn.exists():
        multi_scores = pd.read_parquet(mfn)
        log("skip llm_multilingual.parquet")
    else:
        parts = []
        for lg, g in multi.groupby("variant_lang"):
            scored = g.rename(columns={"doc_id": "orig_id", "doc_id_lang": "doc_id"})
            r = run_block(scored.reset_index(drop=True), "zeroshot", 1,
                          f"multi_{lg}", cfg, backend, keep_raw=False)
            r["lang_variant"] = lg
            r["orig_id"] = r.doc_id.astype(str).str.rsplit("_", n=1).str[0]
            parts.append(r)
            log(f"multilingual {lg} done")
        multi_scores = pd.concat(parts, ignore_index=True)
        multi_scores.to_parquet(mfn, index=False)

    native = bench[(bench.corpus == "bluesky")
                   & (bench.lang.isin(["ja", "es", "pt", "fr", "de"]))]
    native = native.sample(min(len(native), cfg["n_multi"]), random_state=5) if len(native) else native
    fn2 = OUT / "llm_native_nonenglish.parquet"
    if fn2.exists():
        log("skip llm_native_nonenglish.parquet")
    elif len(native):
        nres = run_block(native.reset_index(drop=True), "zeroshot", 1,
                         "native", cfg, backend, keep_raw=False)
        nres = nres.merge(native[["doc_id", "lang"]], on="doc_id")
        nres.to_parquet(fn2, index=False)

    # Arm 3 -- encoder distilled from the LLM teacher.
    free_llm()
    efn = OUT / "encoder_scores.parquet"
    if efn.exists():
        log("skip encoder_scores.parquet")
        enc_cfg = EncoderConfig()
    else:
        train_lbl = llm_consensus_targets(llm_all[llm_all.block == "train"], min_runs=1)
        train_df = train_docs.merge(train_lbl, on="doc_id", how="inner")
        train_df = train_df.dropna(subset=[f"{c}_sal" for c in CONSTRUCTS])
        if len(train_df) < 80:
            log(f"encoder: only {len(train_df)} distillation rows, skip")
            enc_cfg = EncoderConfig()
        else:
            tr = train_df.sample(frac=0.9, random_state=1)
            va = train_df.drop(tr.index)
            log(f"encoder train={tr.shape} val={va.shape}")
            enc_cfg = EncoderConfig(model_id="microsoft/deberta-v3-base",
                                    epochs=3 if profile != "t4" else 2,
                                    batch_size=16 if profile != "t4" else 8)
            enc = EncoderScorer(enc_cfg).fit(tr, va, log=log)
            enc.predict(bench).to_parquet(efn, index=False)
            log("encoder_scores.parquet written")

    if run_bertopic:
        try:
            from bertopic import BERTopic
            from sentence_transformers import SentenceTransformer
            emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            sub = bench.sample(min(len(bench), 6000), random_state=2)
            embs = emb_model.encode(list(sub["text"]), batch_size=128, show_progress_bar=False)
            tm = BERTopic(nr_topics=40, calculate_probabilities=False, verbose=False)
            topics, _ = tm.fit_transform(list(sub["text"]), embs)
            tm.get_topic_info().to_parquet(OUT / "bertopic_info.parquet", index=False)
            pd.DataFrame({"doc_id": sub["doc_id"].values, "topic": topics}) \
              .to_parquet(OUT / "bertopic_assignments.parquet", index=False)
        except Exception as e:
            log(f"BERTopic skipped: {e}")

    try:
        import torch
        import transformers
        torch_v, tr_v = torch.__version__, transformers.__version__
    except Exception:
        torch_v, tr_v = None, None
    prov = dict(
        profile=profile, llm_model=cfg["model"], backend=backend,
        temperature=TEMPERATURE, top_p=0.9, seeds=SEEDS,
        prompt_variants=PROMPT_VARIANTS,
        prompt_hashes={v: prompt_hash(v) for v in PROMPT_VARIANTS},
        encoder_model=enc_cfg.model_id, encoder_epochs=enc_cfg.epochs,
        encoder_lr=enc_cfg.lr, encoder_seed=enc_cfg.seed,
        mt_model=MT_MODEL, n_llm_calls=int(len(llm_all)),
        parse_rate=float(llm_all.parsed.mean()),
        run_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        transformers=tr_v, torch=torch_v)
    (OUT / "colab_provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    log("gpu arms done")
    log(json.dumps(prov, indent=2, default=str)[:1500])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="a5000", choices=list(PROFILES))
    ap.add_argument("--bertopic", action="store_true")
    args = ap.parse_args()
    main(profile=args.profile, run_bertopic=args.bertopic)
