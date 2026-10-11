"""End-to-end analysis steps shared by Notebook 03 and the batch runner.

Each function takes tidy inputs and returns a DataFrame or writes a figure, so
the notebook stays readable and the batch runner cannot drift away from it.
Whatever score files exist are used; methods that have not been produced yet
are simply absent from the tables.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import viz
from .codebook import CONSTRUCTS, CONSTRUCT_META
from .reliability import (fisher_z_test, multi_method_alpha, pairwise_agreement)
from .report import stack_methods, wide_to_long
from .validity import (build_tmu, campbell_fiske, classify_mtmm_cells,
                       convergent_with_closed_form, criterion_validity, htmt,
                       invariance_across_corpora, mtmm_matrix,
                       perturbation_effect)

METHOD_FILES = {
    "DICT": "scores_dict.parquet",
    "TOPIC": "scores_topic.parquet",
    "ENCODER": "encoder_scores.parquet",
    "LLM": "llm_scores.parquet",
}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_scores(derived: Path, llm_consensus_runs=("zeroshot_s1", "zeroshot_s2",
                                                   "zeroshot_s3")) -> dict:
    derived = Path(derived)
    frames = {}
    for m, fn in METHOD_FILES.items():
        p = derived / fn
        if not p.exists():
            continue
        d = pd.read_parquet(p)
        if m == "LLM":
            if "run" in d.columns:
                use = d[d["run"].isin(llm_consensus_runs)]
                if not len(use):
                    use = d[d["run"].astype(str).str.startswith("zeroshot")]
                if not len(use):
                    use = d
            else:
                use = d
            cols = [c for c in use.columns if c.endswith(("_sal", "_pol"))]
            d = use.groupby("doc_id")[cols].median().reset_index()
        frames[m] = d
    return frames


def load_docs(derived: Path) -> pd.DataFrame:
    derived = Path(derived)
    fr = [pd.read_parquet(f) for f in sorted(derived.glob("corpus_*.parquet"))]
    docs = pd.concat(fr, ignore_index=True)
    return docs[docs.split.isin(["bench", "llm_core"])].reset_index(drop=True)


def make_long(frames: dict, docs: pd.DataFrame, kind="pol") -> pd.DataFrame:
    long = stack_methods(frames, kind=kind)
    return long.merge(docs[["doc_id", "corpus", "domain", "closed_form",
                            "criterion_binary", "lang"]],
                      on="doc_id", how="inner")


# ---------------------------------------------------------------------------
# RQ1 -- agreement between methods
# ---------------------------------------------------------------------------
def t_agreement(long: pd.DataFrame, by_corpus=True) -> pd.DataFrame:
    out = []
    scopes = long.groupby("corpus") if by_corpus else [("all", long)]
    for cp, g in scopes:
        for c in CONSTRUCTS:
            t = pairwise_agreement(g, c)
            if len(t):
                t.insert(0, "corpus", cp)
                out.append(t)
    if not out:
        return pd.DataFrame()
    res = pd.concat(out, ignore_index=True)
    return res


def t_multimethod_alpha(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cp, g in long.groupby("corpus"):
        for c in CONSTRUCTS:
            rows.append(dict(corpus=cp, construct=c,
                             alpha=multi_method_alpha(g, c),
                             n_methods=g[g.construct == c]["method"].nunique()))
    return pd.DataFrame(rows)


def f_coverage_bars(cov: pd.DataFrame, path):
    """Coverage by construct, corpus and arm -- the first fork in the road."""
    import matplotlib.pyplot as plt
    viz.apply_style()
    methods = [m for m in viz.METHOD_ORDER if m in set(cov["method"])]
    corpora = sorted(cov["corpus"].unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(3.5 * len(methods), 3.3),
                             sharey=True, squeeze=False)
    for ax, m in zip(axes[0], methods):
        d = cov[cov.method == m]
        x = np.arange(len(CONSTRUCTS)); w = 0.8 / max(len(corpora), 1)
        for i, cp in enumerate(corpora):
            dd = d[d.corpus == cp].set_index("construct").reindex(CONSTRUCTS)
            shade = 0.35 + 0.65 * i / max(len(corpora) - 1, 1)
            ax.bar(x + i * w - 0.4 + w / 2, dd["coverage"].values, width=w * 0.88,
                   color=viz.METHOD_COLORS[m], alpha=shade, label=cp,
                   edgecolor=viz.INK["surface"], linewidth=1)
        ax.set_xticks(x); ax.set_xticklabels(CONSTRUCTS)
        ax.set_title(viz.METHOD_LABELS.get(m, m)); ax.set_ylim(0, 105)
        viz.despine(ax)
    axes[0][0].set_ylabel("% of documents where the\nconstruct is measurable")
    axes[0][-1].legend(title="corpus", fontsize=7)
    fig.tight_layout()
    return viz.save(fig, path)


def f_agreement_heatmap(agree: pd.DataFrame, path, methods=None):
    import matplotlib.pyplot as plt
    viz.apply_style()
    methods = methods or viz.METHOD_ORDER
    pairs = [f"{a}-{b}" for a, b in itertools.combinations(methods, 2)]
    d = agree.copy()
    d["pair"] = d["method_a"] + "-" + d["method_b"]
    piv = (d.groupby(["construct", "pair"])["r"].mean().unstack()
             .reindex(index=CONSTRUCTS, columns=pairs))
    fig, ax = plt.subplots(figsize=(1.1 * len(pairs) + 2.2, 3.2))
    im = ax.imshow(piv.values, cmap=viz.diverging_cmap(), vmin=-1, vmax=1,
                   aspect="auto")
    ax.set_xticks(range(len(piv.columns)))
    ax.set_xticklabels(piv.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}".replace("0.", "."), ha="center",
                        va="center", fontsize=8,
                        color="white" if abs(v) > 0.55 else viz.INK["primary"])
    ax.set_title("Between-method agreement (r), pooled across corpora")
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.85, label="Pearson r")
    fig.tight_layout()
    return viz.save(fig, path)


# ---------------------------------------------------------------------------
# RQ2 -- MTMM and validity
# ---------------------------------------------------------------------------
def t_mtmm(long: pd.DataFrame, corpus=None):
    g = long if corpus in (None, "all") else long[long.corpus == corpus]
    tmu = build_tmu(g)
    corr = mtmm_matrix(tmu)
    cells = classify_mtmm_cells(corr)
    summary = campbell_fiske(cells)
    summary["corpus"] = corpus or "all"
    summary["n_docs"] = int(tmu.shape[0])
    return tmu, corr, cells, summary


def f_mtmm_matrix(corr: pd.DataFrame, path, title="MTMM matrix"):
    import matplotlib.pyplot as plt
    viz.apply_style()
    order = sorted(corr.columns, key=lambda x: (viz.METHOD_ORDER.index(x[1])
                                                if x[1] in viz.METHOD_ORDER else 9,
                                                CONSTRUCTS.index(x[0])))
    M = corr.loc[order, order]
    n = len(order)
    fig, ax = plt.subplots(figsize=(0.42 * n + 2.6, 0.42 * n + 2.2))
    cmap = viz.diverging_cmap()
    cmap.set_bad("#dedcd4")          # not estimable (< 30 joint observations)
    im = ax.imshow(np.ma.masked_invalid(M.values), cmap=cmap, vmin=-1, vmax=1)
    labs = [f"{c}·{m}" for c, m in order]
    ax.set_xticks(range(n)); ax.set_xticklabels(labs, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(labs, fontsize=7)
    # block separators between methods
    meths = [m for _, m in order]
    for i in range(1, n):
        if meths[i] != meths[i - 1]:
            ax.axhline(i - .5, color=viz.INK["primary"], lw=1.0)
            ax.axvline(i - .5, color=viz.INK["primary"], lw=1.0)
    for i in range(n):
        for j in range(n):
            v = M.values[i, j]
            if np.isfinite(v) and order[i][0] == order[j][0] and i != j:
                ax.text(j, i, f"{v:.2f}".replace("0.", "."), ha="center",
                        va="center", fontsize=6,
                        color="white" if abs(v) > .55 else viz.INK["primary"])
    ax.set_title(title); ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.8, label="r")
    fig.tight_layout()
    return viz.save(fig, path)


def t_convergent(long: pd.DataFrame, docs: pd.DataFrame) -> pd.DataFrame:
    out = []
    for cp, g in long.groupby("corpus"):
        d = docs[docs.corpus == cp]
        if d["closed_form"].notna().sum() < 50:
            continue
        t = convergent_with_closed_form(g, d)
        t.insert(0, "corpus", cp)
        out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def t_criterion(long: pd.DataFrame, docs: pd.DataFrame) -> pd.DataFrame:
    out = []
    for cp, g in long.groupby("corpus"):
        d = docs[docs.corpus == cp]
        if d["criterion_binary"].notna().sum() < 100:
            continue
        t = criterion_validity(g, d)
        t.insert(0, "corpus", cp)
        out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def f_convergent_bars(conv: pd.DataFrame, path):
    import matplotlib.pyplot as plt
    viz.apply_style()
    corpora = sorted(conv["corpus"].unique())
    methods = [m for m in viz.METHOD_ORDER if m in set(conv["method"])]
    fig, axes = plt.subplots(1, len(corpora), figsize=(3.6 * len(corpora), 3.4),
                             sharey=True, squeeze=False)
    for ax, cp in zip(axes[0], corpora):
        d = conv[conv.corpus == cp]
        x = np.arange(len(CONSTRUCTS)); w = 0.8 / max(len(methods), 1)
        for i, m in enumerate(methods):
            dd = d[d.method == m].set_index("construct").reindex(CONSTRUCTS)
            pos = x + i * w - 0.4 + w / 2
            ax.bar(pos, dd["r"].values, width=w * 0.88,
                   color=viz.METHOD_COLORS[m], label=viz.METHOD_LABELS[m],
                   edgecolor=viz.INK["surface"], linewidth=1)
            for p, v in zip(pos, dd["r"].values):
                if np.isfinite(v):
                    ax.text(p, v + (0.02 if v >= 0 else -0.05),
                            f"{v:.2f}".replace("0.", "."), ha="center",
                            fontsize=6, color=viz.INK["secondary"])
        ax.axhline(0, color=viz.INK["axis"], lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(CONSTRUCTS)
        ax.set_title(cp); ax.set_ylim(-0.35, 0.85); viz.despine(ax)
    axes[0][0].set_ylabel("r with the same respondent's\nclosed-form response")
    axes[0][-1].legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    return viz.save(fig, path)


def f_coverage_vs_validity(cov: pd.DataFrame, conv: pd.DataFrame, path):
    """Each point is a construct x method: how often it is measurable at all
    against how well it converges.  The trade-off is the paper's headline."""
    import matplotlib.pyplot as plt
    viz.apply_style()
    d = conv.merge(cov, on=["corpus", "construct", "method"], how="inner")
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    for m, g in d.groupby("method"):
        ax.scatter(g["coverage"], g["r"], s=42, color=viz.METHOD_COLORS.get(m, "#888"),
                   label=viz.METHOD_LABELS.get(m, m), edgecolor=viz.INK["surface"],
                   linewidth=1.2, zorder=3)
        for _, r in g.iterrows():
            ax.annotate(r["construct"], (r["coverage"], r["r"]), fontsize=6,
                        xytext=(3, 3), textcoords="offset points",
                        color=viz.INK["muted"])
    ax.axhline(0, color=viz.INK["axis"], lw=0.8)
    ax.set_xlabel("coverage: % of documents where the construct is measurable")
    ax.set_ylabel("convergent validity (r with closed-form response)")
    ax.legend(fontsize=7); viz.despine(ax)
    fig.tight_layout()
    return viz.save(fig, path)


def t_invariance(long: pd.DataFrame, docs: pd.DataFrame) -> pd.DataFrame:
    out = []
    for c in CONSTRUCTS:
        t = invariance_across_corpora(long, docs, c)
        if len(t):
            out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def t_convergence_equality(conv: pd.DataFrame) -> pd.DataFrame:
    """Fisher-z tests that convergent validity is the same in two corpora."""
    rows = []
    for (c, m), g in conv.groupby(["construct", "method"]):
        for a, b in itertools.combinations(sorted(g["corpus"]), 2):
            ra = g[g.corpus == a].iloc[0]
            rb = g[g.corpus == b].iloc[0]
            z, p = fisher_z_test(ra["r"], ra["n"], rb["r"], rb["n"])
            rows.append(dict(construct=c, method=m, corpus_a=a, corpus_b=b,
                             r_a=ra["r"], r_b=rb["r"], z=z, p=p,
                             equal=bool(np.isfinite(p) and p > 0.05)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# reporting helpers
# ---------------------------------------------------------------------------
def coverage_from_long(long: pd.DataFrame) -> pd.DataFrame:
    g = long.groupby(["corpus", "construct", "method"])
    return (g["value"].apply(lambda s: 100 * s.notna().mean())
            .rename("coverage").reset_index())


def headline(long, agree, conv, crit, mtmm_summaries, cov) -> dict:
    def _m(df, col, **flt):
        d = df
        for k, v in flt.items():
            d = d[d[k] == v] if not isinstance(v, (list, tuple)) else d[d[k].isin(v)]
        return float(d[col].mean()) if len(d) else float("nan")

    out = dict(
        n_docs=int(long["doc_id"].nunique()),
        n_methods=int(long["method"].nunique()),
        methods=sorted(long["method"].unique().tolist()),
        n_corpora=int(long["corpus"].nunique()),
        mean_between_method_r=_m(agree, "r"),
        min_between_method_r=float(agree["r"].min()) if len(agree) else float("nan"),
        max_between_method_r=float(agree["r"].max()) if len(agree) else float("nan"),
        mean_convergent_r=_m(conv, "r"),
        best_method_convergent=(conv.groupby("method")["r"].mean().idxmax()
                                if len(conv) else None),
        mean_criterion_auc=_m(crit, "auc"),
        mean_coverage=_m(cov, "coverage"),
    )
    if len(conv):
        by_m = conv.groupby("method")["r"].mean().round(3).to_dict()
        out["convergent_by_method"] = by_m
    if len(cov):
        out["coverage_by_method"] = cov.groupby("method")["coverage"].mean().round(1).to_dict()
    if mtmm_summaries is not None and len(mtmm_summaries):
        s = pd.DataFrame(mtmm_summaries)
        for k in ["mean_convergent_r", "mean_heterotrait_monomethod_r",
                  "mean_heterotrait_heteromethod_r", "method_variance_index",
                  "criterion2_convergent_gt_heterotrait_heteromethod",
                  "criterion3_convergent_gt_heterotrait_monomethod"]:
            if k in s:
                out["mtmm_" + k] = float(s[k].mean())
    return out


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=float), encoding="utf8")
    return path
