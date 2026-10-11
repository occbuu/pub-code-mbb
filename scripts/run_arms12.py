"""Batch version of Notebook 01 -- runs arms 1 and 2 over everything in derived/."""
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cft.resources import apply_thread_limits, banner, cpu_workers, log
apply_thread_limits()
banner("arms12")

from cft.codebook import CONSTRUCTS, CONSTRUCT_META, lexicon_size_table
from cft.dictionary import DictionaryScorer, coverage_table
from cft.topic import TopicScorer
from cft.report import describe_corpus

DERIVED = ROOT / "derived"; TABLES = ROOT / "tables"
TABLES.mkdir(exist_ok=True)

frames = {f.stem.replace("corpus_", ""): pd.read_parquet(f)
          for f in sorted(DERIVED.glob("corpus_*.parquet"))}
docs = pd.concat(frames.values(), ignore_index=True)
bench = docs[docs.split.isin(["bench", "llm_core"])].reset_index(drop=True)
N_JOBS = cpu_workers()
print({k: len(v) for k, v in frames.items()}, "bench", len(bench), "n_jobs", N_JOBS, flush=True)

pd.DataFrame([dict(construct=c, name=CONSTRUCT_META[c]["short"],
                   theory=CONSTRUCT_META[c]["theory"],
                   definition=CONSTRUCT_META[c]["definition"],
                   coding_rule=CONSTRUCT_META[c]["coding_rule"]) for c in CONSTRUCTS]) \
  .to_csv(TABLES / "t01_codebook.csv", index=False)
pd.DataFrame(lexicon_size_table()).to_csv(TABLES / "t02_lexicon_size.csv", index=False)
describe_corpus(docs).to_csv(TABLES / "t03_corpus_descriptives.csv", index=False)

# ---- arm 1 -----------------------------------------------------------------
ds = DictionaryScorer()
t = time.time()
parts = []
for cp, g in bench.groupby("corpus"):
    s = ds.score_frame(g["text"], g["doc_id"], n_jobs=N_JOBS); s["corpus"] = cp
    parts.append(s)
dict_scores = pd.concat(parts, ignore_index=True)
dict_scores.to_parquet(DERIVED / "scores_dict.parquet", index=False)
print(f"dictionary: {len(dict_scores):,} docs in {time.time()-t:.1f}s")

cov = (dict_scores.groupby("corpus")
       .apply(lambda g: coverage_table(g).set_index("construct")["pct_docs_fired"],
              include_groups=False).round(1))
cov.to_csv(TABLES / "t04_dictionary_coverage.csv")
print(cov)

# ---- arm 2 -----------------------------------------------------------------
K = 30
parts, tabs = [], []
for cp, g in frames.items():
    b = g[g.split.isin(["bench", "llm_core"])]
    t = time.time()
    ts = TopicScorer(n_topics=K, method="nmf", random_state=20260909, n_jobs=N_JOBS).fit(g["text"])
    s = ts.transform(b["text"], b["doc_id"]); s["corpus"] = cp
    parts.append(s)
    tt = ts.topic_table(); tt.insert(0, "corpus", cp); tabs.append(tt)
    print(f"topic {cp}: K={K} on {len(g):,} docs in {time.time()-t:.0f}s; anchors",
          {f"{k[0]}.{k[1]}": v for k, v in ts.anchor_hits_.items()})
topic_scores = pd.concat(parts, ignore_index=True)
topic_scores.to_parquet(DERIVED / "scores_topic.parquet", index=False)
pd.concat(tabs, ignore_index=True).to_csv(TABLES / "t05_topic_construct_mapping.csv",
                                          index=False)
# ---- sensitivity to analyst choices ---------------------------------------
variants = {
    "dict_base":        dict(kind="dict"),
    "dict_no_negation": dict(kind="dict", negation_window=0),
    "dict_linear_norm": dict(kind="dict", length_norm="linear"),
    "dict_no_priority": dict(kind="dict", use_priority_patterns=False),
    "topic_K20":        dict(kind="topic", n_topics=20),
    "topic_K30":        dict(kind="topic", n_topics=30),
    "topic_K50":        dict(kind="topic", n_topics=50),
    "topic_LDA_K30":    dict(kind="topic", n_topics=30, method="lda"),
}
rows = []
for name, cfg in variants.items():
    cfg = dict(cfg); kind = cfg.pop("kind")
    for cp, g in frames.items():
        b = g[g.split.isin(["bench", "llm_core"])]
        if kind == "dict":
            s = DictionaryScorer(**cfg).score_frame(b["text"], b["doc_id"], n_jobs=N_JOBS)
        else:
            s = (TopicScorer(random_state=20260909, n_jobs=N_JOBS, **cfg).fit(g["text"])
                 .transform(b["text"], b["doc_id"]))
        s["variant"], s["corpus"] = name, cp
        rows.append(s)
    log(f"sensitivity: {name} done")
sens = pd.concat(rows, ignore_index=True)
sens.to_parquet(DERIVED / "sensitivity_arms12.parquet", index=False)

base = {"dict": "dict_base", "topic": "topic_K30"}
out = []
for name in variants:
    fam = "dict" if name.startswith("dict") else "topic"
    if name == base[fam]:
        continue
    a = sens[sens.variant == base[fam]].set_index("doc_id")
    b = sens[sens.variant == name].set_index("doc_id")
    for c in CONSTRUCTS:
        j = pd.concat([a[f"{c}_pol"], b[f"{c}_pol"]], axis=1, join="inner").dropna()
        j.columns = ["base", "var"]
        rr = (np.corrcoef(j["base"], j["var"])[0, 1]
              if len(j) > 30 and j["var"].std() > 0 else np.nan)
        out.append(dict(family=fam, variant=name, construct=c, n=len(j),
                        r_with_base=rr,
                        mean_abs_shift=float((j["var"] - j["base"]).abs().mean())
                        if len(j) else np.nan))
pd.DataFrame(out).to_csv(TABLES / "t07_sensitivity_arms12.csv", index=False)
print("done", flush=True)
