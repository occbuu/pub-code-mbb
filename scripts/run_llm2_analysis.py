"""Halo as a property of the LLM *family*, not of Qwen alone.

Compares heterotrait-monomethod correlations (method halo) for DICT, TOPIC,
ENCODER, LLM-Qwen and LLM-Mistral on the overlapping llm_core documents, plus
Qwen–Mistral agreement on the same codebook.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cft.resources import apply_thread_limits, banner
apply_thread_limits()
banner("llm2-analysis")

from cft import pipeline as pl
from cft import viz
from cft.codebook import CONSTRUCTS
from cft.validity import build_tmu, campbell_fiske, classify_mtmm_cells

DERIVED, TABLES, FIGURES = ROOT / "derived", ROOT / "tables", ROOT / "figures"
TABLES.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
viz.apply_style()

docs = pl.load_docs(DERIVED)
frames = pl.load_scores(DERIVED)
llm2_path = DERIVED / "llm2_scores.parquet"
if not llm2_path.exists():
    raise SystemExit("missing derived/llm2_scores.parquet")
llm2 = pd.read_parquet(llm2_path)
cols = [c for c in llm2.columns if c.endswith(("_sal", "_pol"))]
llm2_med = llm2.groupby("doc_id")[cols].median().reset_index()
frames["LLM2"] = llm2_med
print("methods:", list(frames), "llm2 docs", llm2_med.doc_id.nunique())

# Restrict halo comparison to documents scored by both LLMs (the llm_core sample).
both = set(frames["LLM"].doc_id.astype(str)) & set(frames["LLM2"].doc_id.astype(str))
print("overlap Qwen ∩ Mistral:", len(both))
docs_c = docs[docs.doc_id.astype(str).isin(both)].copy()
frames_c = {k: v[v.doc_id.astype(str).isin(both)].copy() for k, v in frames.items()}
long = pl.make_long(frames_c, docs_c, kind="pol")


def halo_rows(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cp, m), g in list(long.groupby(["corpus", "method"])) + [
            (("all", m), long[long.method == m]) for m in long.method.unique()]:
        w = g.pivot_table(index="doc_id", columns="construct", values="value")
        w = w.reindex(columns=list(CONSTRUCTS))
        if w.notna().sum().sum() < 80:
            continue
        corr = w.corr(min_periods=30)
        pairs = []
        for i, a in enumerate(CONSTRUCTS):
            for b in CONSTRUCTS[i + 1:]:
                r = corr.loc[a, b] if a in corr.index and b in corr.columns else np.nan
                pairs.append(dict(corpus=cp, method=m, trait_a=a, trait_b=b,
                                  r=float(r) if pd.notna(r) else np.nan,
                                  n=int(w[[a, b]].dropna().shape[0])
                                  if a in w.columns and b in w.columns else 0))
        rows.extend(pairs)
    return pd.DataFrame(rows)


halo = halo_rows(long)
halo.to_csv(TABLES / "t30b_llm_family_halo_pairs.csv", index=False)
summary = (halo.dropna(subset=["r"])
           .groupby(["corpus", "method"], as_index=False)
           .agg(n_pairs=("r", "size"), mean_halo_r=("r", "mean"),
                median_halo_r=("r", "median"), min_r=("r", "min"),
                max_r=("r", "max")))
summary.to_csv(TABLES / "t30_llm_family_halo.csv", index=False)
print(summary[summary.corpus == "all"].round(3).to_string(index=False))

# Qwen vs Mistral agreement on overlapping docs.
long_llms = long[long.method.isin(["LLM", "LLM2"])]
agree_cp = pl.t_agreement(long_llms, by_corpus=True)
tmp = long_llms.copy()
tmp["corpus"] = "all"
agree_all = pl.t_agreement(tmp, by_corpus=True)
agree = pd.concat([agree_all, agree_cp], ignore_index=True)
agree.to_csv(TABLES / "t31_qwen_vs_mistral_agreement.csv", index=False)
print(agree[agree.corpus == "all"].round(3).to_string(index=False))

# Five-method MTMM on the overlap (supplement; primary tables stay 4-arm).
tmu = build_tmu(long)
corr = tmu.corr(min_periods=30)
cells = classify_mtmm_cells(corr)
cells.insert(0, "corpus", "core_overlap")
cells.to_csv(TABLES / "t32b_mtmm_cells_llm_family.csv", index=False)
cf = pd.DataFrame([{"corpus": "core_overlap", **campbell_fiske(cells),
                    "n_docs": int(tmu.shape[0])}])
cf.to_csv(TABLES / "t32_campbell_fiske_llm_family.csv", index=False)
print(cf.round(3).to_string(index=False))

# Figure: mean halo by method (all-corpus overlap).
import matplotlib.pyplot as plt
s = summary[summary.corpus == "all"].copy()
order = [m for m in ["DICT", "TOPIC", "ENCODER", "LLM", "LLM2"] if m in set(s.method)]
s = s.set_index("method").reindex(order)
fig, ax = plt.subplots(figsize=(5.6, 3.4))
colors = [viz.METHOD_COLORS[m] for m in order]
ax.bar(range(len(order)), s["mean_halo_r"].values, color=colors,
       edgecolor=viz.INK["surface"], linewidth=1.2)
for i, (m, v) in enumerate(zip(order, s["mean_halo_r"].values)):
    ax.text(i, v + 0.02, f"{v:.2f}".replace("0.", "."), ha="center",
            fontsize=8, color=viz.INK["secondary"])
ax.axhline(0, color=viz.INK["axis"], lw=0.8)
ax.set_xticks(range(len(order)))
ax.set_xticklabels([viz.METHOD_LABELS.get(m, m) for m in order], rotation=20,
                   ha="right")
ax.set_ylabel("mean heterotrait–monomethod r\n(method halo)")
ax.set_ylim(0, 1.05)
ax.set_title("Halo is a property of the LLM instrument family")
viz.despine(ax)
fig.tight_layout()
viz.save(fig, FIGURES / "f08_llm_family_halo.png")

head = dict(
    n_overlap=int(len(both)),
    halo_by_method={m: float(s.loc[m, "mean_halo_r"]) for m in order if m in s.index},
    qwen_mistral_mean_r=float(agree.loc[agree.corpus == "all", "r"].mean())
    if len(agree) and "r" in agree.columns else float("nan"),
    campbell_fiske_5method=cf.to_dict("records"),
    model=json.loads((DERIVED / "llm2_provenance.json").read_text()).get("model")
    if (DERIVED / "llm2_provenance.json").exists() else None,
)
(TABLES / "results_llm2.json").write_text(json.dumps(head, indent=2, default=float))
print(json.dumps(head, indent=2, default=float)[:1500])
