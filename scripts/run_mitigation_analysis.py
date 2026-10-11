"""Analysis for the mitigation study (Paper B, Study 2).

For every elicitation condition -- plus the joint baseline already in
derived/llm_scores.parquet and the post-hoc statistical fix -- reports the
quantities that decide whether the construct collapse was undone:

    halo            mean r between two different constructs (same arm)
    pc1_share       variance share of the first principal component
    convergent      r with the same respondent's closed-form response
    coverage        share of (doc, construct) cells with a value
    criterion_auc   AUC for the observed repeat behaviour

Writes t33-t35 and figure f11.  Runs on CPU in seconds.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cft import viz
from cft.codebook import CONSTRUCTS
from cft.elicit import residualise_general_factor

DERIVED, TABLES, FIGURES = ROOT / "derived", ROOT / "tables", ROOT / "figures"
TABLES.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
viz.apply_style()

docs = pd.concat([pd.read_parquet(f) for f in sorted(DERIVED.glob("corpus_*.parquet"))],
                 ignore_index=True)
docs = docs[docs.split.isin(["bench", "llm_core"])].set_index("doc_id")


# ---------------------------------------------------------------------------
def profile(wide: pd.DataFrame, label: str) -> dict:
    """wide: doc_id index, one <C>_pol column per construct."""
    cols = [f"{c}_pol" for c in CONSTRUCTS if f"{c}_pol" in wide.columns]
    X = wide[cols]

    halos = []
    for a, b in itertools.combinations(cols, 2):
        j = X[[a, b]].dropna()
        if len(j) >= 30 and j[a].std() > 0 and j[b].std() > 0:
            halos.append(np.corrcoef(j[a], j[b])[0, 1])

    R = np.array(X.corr(min_periods=30).values, dtype=float, copy=True)
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)
    w = np.clip(np.linalg.eigvalsh(R)[::-1], 0, None)

    conv, aucs = [], []
    ref = docs["closed_form"].reindex(X.index)
    beh = docs["criterion_binary"].reindex(X.index)
    for c in cols:
        j = pd.concat([X[c], ref], axis=1).dropna()
        if len(j) >= 30 and j.iloc[:, 0].std() > 0:
            conv.append(np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1])
        k = pd.concat([X[c], beh], axis=1).dropna()
        if len(k) >= 100 and k.iloc[:, 1].nunique() > 1 and k.iloc[:, 0].std() > 0:
            aucs.append(roc_auc_score(k.iloc[:, 1], k.iloc[:, 0]))

    return dict(
        condition=label,
        n_docs=int(len(X)),
        coverage=100 * float(X.notna().mean().mean()),
        halo=float(np.mean(halos)) if halos else np.nan,
        pc1_share=float(w[0] / w.sum()) if w.sum() else np.nan,
        convergent=float(np.mean(conv)) if conv else np.nan,
        discriminant_margin=(float(np.mean(conv)) - float(np.mean(halos)))
        if conv and halos else np.nan,
        criterion_auc=float(np.mean(aucs)) if aucs else np.nan,
    )


# ---------------------------------------------------------------------------
# baseline: the joint elicitation already on disk
# ---------------------------------------------------------------------------
rows, wides = [], {}

llm = pd.read_parquet(DERIVED / "llm_scores.parquet")
llm = llm[llm["run"].astype(str).str.startswith("zeroshot")]
cols = [c for c in llm.columns if c.endswith("_pol")]
joint = llm.groupby("doc_id")[cols].median()

mit_path = DERIVED / "mitigation_scores.parquet"
if not mit_path.exists():
    raise SystemExit("derived/mitigation_scores.parquet not found -- run "
                     "scripts/run_mitigation.py first")
mit = pd.read_parquet(mit_path)
ids = sorted(set(mit["doc_id"]))
joint = joint.reindex(ids)
wides["joint"] = joint
rows.append(profile(joint, "joint (baseline)"))

for cond, g in mit.groupby("condition"):
    wide = (g.pivot_table(index="doc_id", columns="construct", values="pol")
              .rename(columns={c: f"{c}_pol" for c in CONSTRUCTS})
              .reindex(ids))
    wides[cond] = wide
    rows.append(profile(wide, cond))

# post-hoc statistical mitigation, on the baseline scores
resid = residualise_general_factor(joint.copy())
rows.append(profile(resid, "joint + PC1 removed"))

prof = pd.DataFrame(rows)
prof.to_csv(TABLES / "t33_mitigation_profile.csv", index=False)
print(prof.round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# per-construct coverage and the evidence-quote audit
# ---------------------------------------------------------------------------
cov_rows = []
for cond, wide in wides.items():
    for c in CONSTRUCTS:
        col = f"{c}_pol"
        if col in wide:
            cov_rows.append(dict(condition=cond, construct=c,
                                 coverage=100 * float(wide[col].notna().mean())))
pd.DataFrame(cov_rows).to_csv(TABLES / "t34_mitigation_coverage.csv", index=False)

if "evidence" in set(mit["condition"]):
    ev = mit[mit.condition == "evidence"]
    q = (ev.assign(has_quote=ev["quote"].notna())
           .groupby("construct")
           .agg(n=("doc_id", "size"),
                pct_with_quote=("has_quote", lambda s: 100 * s.mean()),
                pct_verbatim=("quote_verbatim", lambda s: 100 * s.mean()),
                coverage=("pol", lambda s: 100 * s.notna().mean()))
           .reset_index())
    q.to_csv(TABLES / "t35_evidence_quote_audit.csv", index=False)
    print("\nevidence-condition quote audit:")
    print(q.round(1).to_string(index=False))

# ---------------------------------------------------------------------------
# figure: did any intervention move the halo without losing convergence?
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6))
order = prof["condition"].tolist()
pal = [viz.METHOD_COLORS[m] for m in viz.METHOD_ORDER]
colors = [pal[i % len(pal)] for i in range(len(order))]

ax = axes[0]
x = np.arange(len(order)); w = 0.38
ax.bar(x - w/2, prof["halo"], w*0.9, color=viz.STATUS["critical"],
       label="method halo", edgecolor=viz.INK["surface"], lw=1)
ax.bar(x + w/2, prof["convergent"], w*0.9, color=viz.METHOD_COLORS["DICT"],
       label="convergent validity", edgecolor=viz.INK["surface"], lw=1)
for xi, v in zip(x - w/2, prof["halo"]):
    if np.isfinite(v):
        ax.text(xi, v + .02, f"{v:.2f}".replace("0.", "."), ha="center", fontsize=7,
                color=viz.INK["secondary"])
for xi, v in zip(x + w/2, prof["convergent"]):
    if np.isfinite(v):
        ax.text(xi, v + .02, f"{v:.2f}".replace("0.", "."), ha="center", fontsize=7,
                color=viz.INK["secondary"])
ax.axhline(0, color=viz.INK["axis"], lw=.8)
ax.set_xticks(x); ax.set_xticklabels(order, rotation=20, ha="right", fontsize=7)
ax.set_ylabel("r"); ax.set_title("Halo vs convergence by elicitation condition")
ax.legend(fontsize=7); viz.despine(ax)

ax = axes[1]
for i, r_ in prof.iterrows():
    ax.scatter(r_["halo"], r_["convergent"], s=60 + 2.2 * (r_["coverage"] or 0),
               color=colors[i], edgecolor=viz.INK["surface"], lw=1.4, zorder=3)
    ax.annotate(r_["condition"], (r_["halo"], r_["convergent"]),
                textcoords="offset points", xytext=(0, -15), ha="center",
                fontsize=7, color=viz.INK["secondary"])
lim = [min(0, float(np.nanmin(prof["halo"])) - .05), 1.0]
ax.plot(lim, lim, ls=":", lw=1, color=viz.INK["muted"], zorder=1)
ax.set_xlabel("method halo"); ax.set_ylabel("convergent validity")
ax.set_title("Above the line = discriminates")
viz.despine(ax)
fig.tight_layout()
viz.save(fig, FIGURES / "f11_mitigation.png")

json.dump(dict(profile=prof.round(4).to_dict("records")),
          open(TABLES / "results_mitigation.json", "w"), indent=2, default=float)
print("\nwrote t33-t35, results_mitigation.json, f11_mitigation.png")
