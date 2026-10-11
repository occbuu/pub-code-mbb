"""Batch version of Notebook 03 -- reliability, MTMM, validity, invariance.

Uses whichever measurement arms are present in derived/, so it can be run on the
dictionary/topic pilot and re-run unchanged once the Colab arms land.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cft.resources import apply_thread_limits, banner
apply_thread_limits()
banner("analysis")
from cft import pipeline as pl
from cft import viz
from cft.codebook import CONSTRUCTS
from cft.report import checklist_frame

DERIVED, TABLES, FIGURES = ROOT / "derived", ROOT / "tables", ROOT / "figures"
TABLES.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
viz.apply_style()

docs = pl.load_docs(DERIVED)
frames = pl.load_scores(DERIVED)
print("methods present:", list(frames))
long = pl.make_long(frames, docs, kind="pol")
long_sal = pl.make_long(frames, docs, kind="sal")
print("long rows:", len(long), "| docs:", long.doc_id.nunique())

# ---- coverage --------------------------------------------------------------
cov = pl.coverage_from_long(long)
cov.to_csv(TABLES / "t06_coverage_by_method.csv", index=False)
pl.f_coverage_bars(cov, FIGURES / "f01_coverage.png")

# ---- RQ1 agreement ---------------------------------------------------------
agree = pl.t_agreement(long)
agree.to_csv(TABLES / "t09_between_method_agreement.csv", index=False)
alpha = pl.t_multimethod_alpha(long)
alpha.to_csv(TABLES / "t10_multimethod_alpha.csv", index=False)
if len(agree):
    pl.f_agreement_heatmap(agree, FIGURES / "f02_agreement_heatmap.png",
                           methods=[m for m in viz.METHOD_ORDER if m in frames])

# ---- RQ2 MTMM --------------------------------------------------------------
summaries, cells_all = [], []
for cp in ["all"] + sorted(long.corpus.unique()):
    tmu, corr, cells, summ = pl.t_mtmm(long, cp)
    cells.insert(0, "corpus", cp); cells_all.append(cells)
    summaries.append(summ)
    if cp == "all":
        corr.to_csv(TABLES / "t11_mtmm_matrix_all.csv")
        pl.f_mtmm_matrix(corr, FIGURES / "f03_mtmm_matrix.png",
                         title="Multitrait-multimethod matrix (all corpora)")
pd.concat(cells_all, ignore_index=True).to_csv(TABLES / "t12_mtmm_cells.csv", index=False)
mtmm_sum = pd.DataFrame(summaries)
mtmm_sum.to_csv(TABLES / "t13_campbell_fiske.csv", index=False)
print(mtmm_sum[["corpus", "n_docs", "mean_convergent_r",
                "mean_heterotrait_monomethod_r",
                "mean_heterotrait_heteromethod_r",
                "method_variance_index"]].round(3).to_string(index=False))

# HTMT
tmu_all = pl.build_tmu(long)
from cft.validity import htmt
h = htmt(tmu_all)
h.to_csv(TABLES / "t14_htmt.csv")

# ---- convergent + criterion ------------------------------------------------
conv = pl.t_convergent(long, docs)
conv.to_csv(TABLES / "t15_convergent_closed_form.csv", index=False)
crit = pl.t_criterion(long, docs)
crit.to_csv(TABLES / "t16_criterion_auc.csv", index=False)
if len(conv):
    pl.f_convergent_bars(conv, FIGURES / "f04_convergent.png")
    cc = conv.merge(cov, on=["corpus", "construct", "method"], how="left")
    pl.f_coverage_vs_validity(cov, conv, FIGURES / "f05_coverage_vs_validity.png")
    eq = pl.t_convergence_equality(conv)
    eq.to_csv(TABLES / "t17_convergence_equality.csv", index=False)

# ---- invariance ------------------------------------------------------------
inv = pl.t_invariance(long, docs)
if len(inv):
    inv.to_csv(TABLES / "t18_invariance.csv", index=False)

# ---- checklist -------------------------------------------------------------
checklist_frame().to_csv(TABLES / "t19_reporting_checklist.csv", index=False)

# ---- headline numbers ------------------------------------------------------
head = pl.headline(long, agree, conv, crit, summaries, cov)
head["corpora"] = sorted(long.corpus.unique().tolist())
pl.save_json(head, TABLES / "results_headline.json")
print(json.dumps(head, indent=2, default=float)[:1800])
