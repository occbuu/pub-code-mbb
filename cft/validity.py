"""Validity evidence: MTMM matrix and the Campbell-Fiske decision rules, HTMT,
convergent validity against a closed-form response, criterion validity against
observed behaviour, and measurement invariance across corpora.

The MTMM helpers take a *wide* frame indexed by ``doc_id`` whose columns are
``(construct, method)`` pairs -- one column per trait-method unit.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

from .reliability import fisher_ci, fisher_z_test


# ---------------------------------------------------------------------------
# MTMM
# ---------------------------------------------------------------------------
def build_tmu(long_scores: pd.DataFrame, constructs=None, methods=None) -> pd.DataFrame:
    """Trait-method-unit matrix: rows = documents, columns = (construct, method)."""
    d = long_scores.copy()
    if constructs:
        d = d[d["construct"].isin(constructs)]
    if methods:
        d = d[d["method"].isin(methods)]
    w = d.pivot_table(index="doc_id", columns=["construct", "method"], values="value")
    return w


def mtmm_matrix(tmu: pd.DataFrame, method="pearson") -> pd.DataFrame:
    """Correlation matrix of all trait-method units (pairwise complete)."""
    return tmu.corr(method=method, min_periods=30)


def classify_mtmm_cells(corr: pd.DataFrame) -> pd.DataFrame:
    """Label every off-diagonal cell as monotrait-heteromethod (convergent),
    heterotrait-monomethod (method halo) or heterotrait-heteromethod."""
    rows = []
    cols = list(corr.columns)
    for (c1, m1), (c2, m2) in itertools.combinations(cols, 2):
        r = corr.loc[(c1, m1), (c2, m2)]
        if c1 == c2:
            kind = "monotrait-heteromethod"
        elif m1 == m2:
            kind = "heterotrait-monomethod"
        else:
            kind = "heterotrait-heteromethod"
        rows.append(dict(trait_a=c1, method_a=m1, trait_b=c2, method_b=m2,
                         kind=kind, r=float(r) if pd.notna(r) else np.nan))
    return pd.DataFrame(rows)


def campbell_fiske(cells: pd.DataFrame) -> dict:
    """The four Campbell & Fiske (1959) criteria, expressed as pass rates."""
    mono = cells[cells.kind == "monotrait-heteromethod"].dropna(subset=["r"])
    het_mono = cells[cells.kind == "heterotrait-monomethod"].dropna(subset=["r"])
    het_het = cells[cells.kind == "heterotrait-heteromethod"].dropna(subset=["r"])

    # C1: convergent validities significantly different from zero and "sufficiently large"
    c1 = float((mono["r"] > 0.30).mean()) if len(mono) else np.nan

    # C2: each convergent value exceeds heterotrait-heteromethod values in its row/col
    passes = []
    for _, row in mono.iterrows():
        comp = het_het[((het_het.method_a == row.method_a) & (het_het.method_b == row.method_b)) |
                       ((het_het.method_a == row.method_b) & (het_het.method_b == row.method_a))]
        comp = comp[(comp.trait_a == row.trait_a) | (comp.trait_b == row.trait_a) |
                    (comp.trait_a == row.trait_b) | (comp.trait_b == row.trait_b)]
        if len(comp):
            passes.append(row.r > comp["r"].max())
    c2 = float(np.mean(passes)) if passes else np.nan

    # C3: convergent value exceeds the heterotrait-monomethod values of the same methods
    passes = []
    for _, row in mono.iterrows():
        comp = het_mono[het_mono.method_a.isin([row.method_a, row.method_b])]
        comp = comp[(comp.trait_a == row.trait_a) | (comp.trait_b == row.trait_a)]
        if len(comp):
            passes.append(row.r > comp["r"].max())
    c3 = float(np.mean(passes)) if passes else np.nan

    return dict(
        n_convergent=int(len(mono)),
        mean_convergent_r=float(mono["r"].mean()) if len(mono) else np.nan,
        median_convergent_r=float(mono["r"].median()) if len(mono) else np.nan,
        mean_heterotrait_monomethod_r=float(het_mono["r"].mean()) if len(het_mono) else np.nan,
        mean_heterotrait_heteromethod_r=float(het_het["r"].mean()) if len(het_het) else np.nan,
        criterion1_convergent_gt_030=c1,
        criterion2_convergent_gt_heterotrait_heteromethod=c2,
        criterion3_convergent_gt_heterotrait_monomethod=c3,
        method_variance_index=float(het_mono["r"].mean() - het_het["r"].mean())
        if len(het_mono) and len(het_het) else np.nan,
    )


def htmt(tmu: pd.DataFrame, constructs=None) -> pd.DataFrame:
    """Heterotrait-monotrait ratio of correlations (Henseler et al., 2015).

    Here the "indicators" of a construct are its measurement arms, so HTMT
    directly answers: do two constructs measured by four methods separate?
    """
    cols = tmu.columns
    traits = constructs or sorted({c for c, _ in cols})
    corr = tmu.corr(min_periods=30).abs()
    out = pd.DataFrame(index=traits, columns=traits, dtype=float)
    for t1, t2 in itertools.combinations(traits, 2):
        i1 = [c for c in cols if c[0] == t1]
        i2 = [c for c in cols if c[0] == t2]
        hetero = corr.loc[i1, i2].values
        mono1 = corr.loc[i1, i1].values[np.triu_indices(len(i1), 1)]
        mono2 = corr.loc[i2, i2].values[np.triu_indices(len(i2), 1)]
        denom = np.sqrt(np.nanmean(mono1) * np.nanmean(mono2))
        v = np.nanmean(hetero) / denom if denom and np.isfinite(denom) else np.nan
        out.loc[t1, t2] = out.loc[t2, t1] = v
    for t in traits:
        out.loc[t, t] = 1.0
    return out


# ---------------------------------------------------------------------------
# convergent / criterion validity
# ---------------------------------------------------------------------------
def convergent_with_closed_form(long_scores: pd.DataFrame, docs: pd.DataFrame,
                                closed_col="closed_form",
                                constructs=None, methods=None) -> pd.DataFrame:
    """Correlate every trait-method unit with the same respondent's closed-form
    response (star rating, stance, grade) -- the design's stand-in for a
    self-report scale answered by the same person at the same moment."""
    ref = docs.set_index("doc_id")[closed_col]
    rows = []
    d = long_scores
    if constructs:
        d = d[d["construct"].isin(constructs)]
    if methods:
        d = d[d["method"].isin(methods)]
    for (c, m), g in d.groupby(["construct", "method"]):
        s = g.set_index("doc_id")["value"]
        sub = pd.concat([s, ref], axis=1, join="inner").dropna()
        sub.columns = ["value", "ref"]
        if len(sub) < 30 or sub["value"].std() == 0:
            rows.append(dict(construct=c, method=m, n=len(sub), r=np.nan))
            continue
        r, p = stats.pearsonr(sub["value"], sub["ref"])
        rho, _ = stats.spearmanr(sub["value"], sub["ref"])
        lo, hi = fisher_ci(r, len(sub))
        rows.append(dict(construct=c, method=m, n=int(len(sub)), r=float(r),
                         ci_lo=lo, ci_hi=hi, rho=float(rho), p=float(p)))
    return pd.DataFrame(rows)


def criterion_validity(long_scores: pd.DataFrame, docs: pd.DataFrame,
                       outcome="criterion_binary", constructs=None,
                       methods=None, min_n=100) -> pd.DataFrame:
    """Predictive validity against observed behaviour: AUC of a single
    trait-method unit for the binary behavioural outcome."""
    from sklearn.metrics import roc_auc_score
    ref = docs.set_index("doc_id")[outcome]
    rows = []
    d = long_scores
    if constructs:
        d = d[d["construct"].isin(constructs)]
    if methods:
        d = d[d["method"].isin(methods)]
    for (c, m), g in d.groupby(["construct", "method"]):
        s = g.set_index("doc_id")["value"]
        sub = pd.concat([s, ref], axis=1, join="inner").dropna()
        sub.columns = ["value", "y"]
        if len(sub) < min_n or sub["y"].nunique() < 2 or sub["value"].std() == 0:
            rows.append(dict(construct=c, method=m, n=len(sub), auc=np.nan))
            continue
        auc = roc_auc_score(sub["y"], sub["value"])
        rows.append(dict(construct=c, method=m, n=int(len(sub)),
                         base_rate=float(sub["y"].mean()), auc=float(auc)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# invariance across corpora
# ---------------------------------------------------------------------------
def congruence(loadings_a: np.ndarray, loadings_b: np.ndarray) -> float:
    """Tucker's congruence coefficient between two loading vectors."""
    a, b = np.asarray(loadings_a, float), np.asarray(loadings_b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 2:
        return np.nan
    return float((a @ b) / np.sqrt((a @ a) * (b @ b)))


def method_factor_loadings(tmu: pd.DataFrame, construct: str,
                           min_methods: int = 3) -> pd.Series:
    """First principal component loadings of the methods measuring one construct.

    ``min_methods`` defaults to 3 because a congruence coefficient computed on a
    two-element loading vector is degenerate -- it is +-1 by construction and
    says nothing about invariance.
    """
    cols = [c for c in tmu.columns if c[0] == construct]
    X = tmu[cols].dropna()
    if X.shape[0] < 30 or X.shape[1] < min_methods:
        return pd.Series(dtype=float)
    Z = (X - X.mean()) / X.std(ddof=1).replace(0, np.nan)
    Z = Z.dropna(axis=1, how="all").fillna(0.0)
    C = np.corrcoef(Z.values, rowvar=False)
    w, v = np.linalg.eigh(C)
    pc = v[:, -1] * np.sqrt(max(w[-1], 0))
    if pc.sum() < 0:
        pc = -pc
    return pd.Series(pc, index=[c[1] for c in Z.columns])


def invariance_across_corpora(long_scores: pd.DataFrame, docs: pd.DataFrame,
                              construct: str, corpus_col="corpus") -> pd.DataFrame:
    """Configural/metric-style invariance evidence for a construct across corpora.

    For each pair of corpora we report Tucker's phi between the method-loading
    vectors (metric invariance analogue) and a Fisher-z test on the mean
    convergent correlation (a scale-level equality check)."""
    corp = docs.set_index("doc_id")[corpus_col]
    d = long_scores[long_scores["construct"] == construct].copy()
    d["corpus"] = d["doc_id"].map(corp)
    loads, ns = {}, {}
    for cp, g in d.groupby("corpus"):
        tmu = build_tmu(g)
        L = method_factor_loadings(tmu, construct)
        if len(L):
            loads[cp] = L
            ns[cp] = tmu.dropna().shape[0]
    rows = []
    for a, b in itertools.combinations(sorted(loads), 2):
        common = [m for m in loads[a].index if m in loads[b].index]
        phi = congruence(loads[a][common].values, loads[b][common].values)
        rows.append(dict(construct=construct, corpus_a=a, corpus_b=b,
                         n_methods=len(common), tuckers_phi=phi,
                         n_a=ns[a], n_b=ns[b],
                         invariant=bool(np.isfinite(phi) and phi >= 0.95)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# social-desirability / perturbation audits
# ---------------------------------------------------------------------------
def perturbation_effect(base: pd.DataFrame, perturbed: pd.DataFrame,
                        key=("doc_id", "construct", "method"),
                        value="value") -> pd.DataFrame:
    """Paired effect of a text perturbation (name swap, translation, desirability
    cue) on measured scores: mean signed shift, Cohen's d_z and a paired t-test."""
    m = base.merge(perturbed, on=list(key), suffixes=("_base", "_pert"))
    rows = []
    for (c, meth), g in m.groupby(["construct", "method"]):
        diff = (g[f"{value}_pert"] - g[f"{value}_base"]).dropna()
        if len(diff) < 10:
            continue
        t, p = stats.ttest_rel(g[f"{value}_pert"], g[f"{value}_base"], nan_policy="omit")
        dz = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else np.nan
        rows.append(dict(construct=c, method=meth, n=int(len(diff)),
                         mean_shift=float(diff.mean()), d_z=float(dz),
                         t=float(t), p=float(p)))
    return pd.DataFrame(rows)
