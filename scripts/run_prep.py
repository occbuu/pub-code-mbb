"""Notebook 00 as a skip-safe batch script.

Resolves paths relative to this repo (no Windows PROJECT path) and writes
``derived/corpus_*.parquet``, ``oulad_panel.parquet`` and the Colab-sized
exports.  Stages that already have an output file are skipped.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cft.resources import apply_thread_limits, banner, log  # noqa: E402

apply_thread_limits()
banner("prep")

import pandas as pd  # noqa: E402
from cft import corpora as co  # noqa: E402
from cft.report import describe_corpus  # noqa: E402

DATA, DERIVED = ROOT / "DataPaper13", ROOT / "derived"
DERIVED.mkdir(parents=True, exist_ok=True)

N_POOL, N_BENCH, N_LLM_CORE, SEED = 30_000, 1_200, 400, 20260909

EXPECT = {
    "yelp_review": DATA / "yelp_academic_dataset_review-002.json",
    "yelp_business": DATA / "Yelp_Open_Dataset" / "yelp_academic_dataset_business.json",
    "amazon": DATA / "Amazon_reviews_2023" / "Amazon_reviews_2023.csv.zip",
    "regulations_parquet": DATA / "Regulatory_Comments_dataset" / "docket_comments_v6.parquet",
    "regulations_json": DATA / "Regulatory_Comments_dataset" / "docket_comments_v6.json",
    "bluesky": DATA / "Bluesky" / "bs3m2.csv",
    "oulad": DATA / "Open_University_Learning_Analytics_dataset",
}


def _skip_or(path: Path, builder, label: str):
    if path.exists():
        log(f"skip -- already built: {path.name}")
        return pd.read_parquet(path)
    t = time.time()
    df = builder()
    df.to_parquet(path, index=False)
    log(f"{label}: {len(df):,} rows in {time.time()-t:.0f}s -> {path.name}")
    return df


def main():
    for k, p in EXPECT.items():
        exists = p.exists()
        size = (round(p.stat().st_size / 1e6, 1)
                if exists and p.is_file() else None)
        log(f"source {k}: exists={exists} size_mb={size} path={p}")

    _skip_or(DERIVED / "corpus_amazon.parquet",
             lambda: co.build_amazon(EXPECT["amazon"], n_pool=N_POOL,
                                     n_bench=N_BENCH, n_llm_core=N_LLM_CORE,
                                     seed=SEED),
             "amazon")

    _skip_or(DERIVED / "corpus_yelp.parquet",
             lambda: co.build_yelp(EXPECT["yelp_review"], n_pool=N_POOL,
                                   n_bench=N_BENCH, n_llm_core=N_LLM_CORE,
                                   seed=SEED,
                                   progress=lambda m: log(m)),
             "yelp")

    reg_src = (EXPECT["regulations_parquet"]
               if EXPECT["regulations_parquet"].exists()
               else EXPECT["regulations_json"])
    _skip_or(DERIVED / "corpus_regulations.parquet",
             lambda: co.build_regulations(reg_src, n_bench=N_BENCH,
                                          n_llm_core=N_LLM_CORE, seed=SEED),
             "regulations")

    _skip_or(DERIVED / "corpus_bluesky.parquet",
             lambda: co.build_bluesky(EXPECT["bluesky"], n_pool=N_POOL,
                                      n_bench=N_BENCH, n_llm_core=N_LLM_CORE,
                                      seed=SEED),
             "bluesky")

    _skip_or(DERIVED / "oulad_panel.parquet",
             lambda: co.build_oulad(EXPECT["oulad"]),
             "oulad")

    frames = {f.stem.replace("corpus_", ""): pd.read_parquet(f)
              for f in sorted(DERIVED.glob("corpus_*.parquet"))}
    docs = pd.concat(frames.values(), ignore_index=True)
    desc = describe_corpus(docs)
    desc.to_csv(DERIVED / "corpus_descriptives.csv", index=False)
    manifest = dict(
        built_utc=pd.Timestamp.utcnow().isoformat(),
        seed=SEED, n_pool=N_POOL, n_bench=N_BENCH, n_llm_core=N_LLM_CORE,
        corpora={k: dict(n=int(len(v)),
                         n_bench=int(v.split.isin(["bench", "llm_core"]).sum()),
                         n_llm_core=int((v.split == "llm_core").sum()),
                         sha1_head=hashlib.sha1(
                             "".join(v["text"].head(200)).encode("utf8")
                         ).hexdigest()[:12])
                 for k, v in frames.items()})
    (DERIVED / "manifest.json").write_text(json.dumps(manifest, indent=2))

    bench = docs[docs.split.isin(["bench", "llm_core"])][
        ["doc_id", "corpus", "domain", "text", "n_tokens", "lang", "closed_form",
         "criterion_binary", "split"]].reset_index(drop=True)
    bench.to_parquet(DERIVED / "bench_for_colab.parquet", index=False)
    pool = docs[["doc_id", "corpus", "text", "split"]]
    pool.to_parquet(DERIVED / "pool_for_colab.parquet", index=False)
    log(f"bench={bench.shape} pool={pool.shape} corpora={list(frames)}")
    log("prep done")


if __name__ == "__main__":
    main()
