"""Reliability evidence: agreement between measurement arms, stability of the
LLM arm across independent runs, and sensitivity to prompt wording.

All functions accept a "long" score frame with columns
``doc_id, method, construct, value`` and return tidy DataFrames.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

try:                                        # optional, pip install krippendorff
    import krippendorff as _kd
except Exception:                           # pragma: no cover
    _kd = None


# ---------------------------------------------------------------------------
# agreement
# ---------------------------------------------------------------------------
def krippendorff_alpha(matrix, level="interval"):
    """Krippendorff's alpha for a raters x units matrix (NaN = missing).

    The interval metric is computed in closed form from per-unit moments --
    ``sum_{i!=j}(x_i-x_j)^2 = 2*(m*sum(x^2) - sum(x)^2)`` -- which is exact and
    linear in the number of values.  (The reference package builds a
    value-by-value coincidence matrix, which is quadratic in the number of
    distinct values and exhausts memory on continuous scores.)  Other metrics
    are delegated to the ``krippendorff`` package when it is installed.
    """
    M = np.asarray(matrix, dtype=float)
    if level != "interval":
        if _kd is None:
            raise ImportError("pip install krippendorff for non-interval metrics")
        return float(_kd.alpha(reliability_data=M, level_of_measurement=level))

    ok = ~np.isnan(M)
    m_u = ok.sum(axis=0).astype(float)                 # values per unit
    use = m_u >= 2
    if use.sum() < 2:
        return np.nan
    X = np.where(ok, M, 0.0)
    s1 = X.sum(axis=0)[use]                            # sum x per unit
    s2 = (X ** 2).sum(axis=0)[use]                     # sum x^2 per unit
    m = m_u[use]

    n = float(m.sum())
    do = float((2.0 * (m * s2 - s1 ** 2) / (m - 1.0)).sum()) / n
    S1, S2 = float(s1.sum()), float(s2.sum())
    de = 2.0 * (n * S2 - S1 ** 2) / (n * (n - 1.0))
    if not np.isfinite(de) or de <= 0:
        return np.nan
    return float(1.0 - do / de)


def pairwise_agreement(long_scores: pd.DataFrame, construct: str,
                       methods=None, min_pairs: int = 30) -> pd.DataFrame:
    """Pearson / Spearman / alpha for every pair of methods on one construct."""
    d = long_scores[long_scores["construct"] == construct]
    wide = d.pivot_table(index="doc_id", columns="method", values="value")
    methods = list(methods or wide.columns)
    rows = []
    for a, b in itertools.combinations(methods, 2):
        if a not in wide or b not in wide:
            continue
        sub = wide[[a, b]].dropna()
        if len(sub) < min_pairs:
            rows.append(dict(construct=construct, method_a=a, method_b=b,
                             n=len(sub), r=np.nan, rho=np.nan, alpha=np.nan))
            continue
        r = stats.pearsonr(sub[a], sub[b])
        rho = stats.spearmanr(sub[a], sub[b])
        z = (sub - sub.mean()) / sub.std(ddof=1).replace(0, np.nan)
        al = krippendorff_alpha(z.T.values)
        rows.append(dict(construct=construct, method_a=a, method_b=b, n=len(sub),
                         r=float(r[0]), p=float(r[1]), rho=float(rho[0]),
                         alpha=float(al)))
    return pd.DataFrame(rows)


def multi_method_alpha(long_scores: pd.DataFrame, construct: str,
                       methods=None) -> float:
    """Alpha across *all* methods simultaneously (standardised scores)."""
    d = long_scores[long_scores["construct"] == construct]
    wide = d.pivot_table(index="doc_id", columns="method", values="value")
    if methods:
        wide = wide[[m for m in methods if m in wide.columns]]
    z = (wide - wide.mean()) / wide.std(ddof=1).replace(0, np.nan)
    return krippendorff_alpha(z.T.values)


# ---------------------------------------------------------------------------
# stability (LLM run-to-run and prompt sensitivity)
# ---------------------------------------------------------------------------
def run_stability(run_scores: pd.DataFrame, value="value",
                  by=("construct",)) -> pd.DataFrame:
    """ICC-style stability across independent runs of a stochastic scorer.

    ``run_scores`` needs columns ``doc_id, run, <by...>, value``.
    Returns ICC(2,1)-equivalent (two-way random, absolute agreement), the mean
    pairwise r between runs, and the mean absolute deviation between runs.
    """
    out = []
    for keys, d in run_scores.groupby(list(by)):
        wide = d.pivot_table(index="doc_id", columns="run", values=value).dropna()
        if wide.shape[0] < 10 or wide.shape[1] < 2:
            continue
        k = wide.shape[1]
        n = wide.shape[0]
        grand = wide.values.mean()
        ms_r = k * ((wide.mean(axis=1) - grand) ** 2).sum() / (n - 1)
        ms_c = n * ((wide.mean(axis=0) - grand) ** 2).sum() / (k - 1)
        resid = wide.values - wide.mean(axis=1).values[:, None] \
                - wide.mean(axis=0).values[None, :] + grand
        ms_e = (resid ** 2).sum() / ((n - 1) * (k - 1))
        icc = (ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n)
        rs = [stats.pearsonr(wide.iloc[:, i], wide.iloc[:, j])[0]
              for i, j in itertools.combinations(range(k), 2)]
        mad = np.mean([np.abs(wide.iloc[:, i] - wide.iloc[:, j]).mean()
                       for i, j in itertools.combinations(range(k), 2)])
        rec = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        rec.update(n_docs=n, n_runs=k, icc=float(icc),
                   mean_r_between_runs=float(np.mean(rs)), mad=float(mad))
        out.append(rec)
    return pd.DataFrame(out)


def prompt_sensitivity(scores: pd.DataFrame, prompt_col="prompt_variant",
                       by=("construct",)) -> pd.DataFrame:
    """How much does the *wording* of the prompt move the measurement?

    Reports the between-variant SD of the document mean, the mean pairwise r
    between variants, and eta^2 of the variant factor on the document score.
    """
    out = []
    for keys, d in scores.groupby(list(by)):
        wide = d.pivot_table(index="doc_id", columns=prompt_col, values="value").dropna()
        if wide.shape[0] < 10 or wide.shape[1] < 2:
            continue
        rs = [stats.pearsonr(wide.iloc[:, i], wide.iloc[:, j])[0]
              for i, j in itertools.combinations(range(wide.shape[1]), 2)]
        long = wide.stack().reset_index()
        long.columns = ["doc_id", prompt_col, "value"]
        grand = long["value"].mean()
        ss_between = sum(len(g) * (g["value"].mean() - grand) ** 2
                         for _, g in long.groupby(prompt_col))
        ss_total = ((long["value"] - grand) ** 2).sum()
        rec = dict(zip(by, keys if isinstance(keys, tuple) else (keys,)))
        rec.update(n_docs=wide.shape[0], n_variants=wide.shape[1],
                   mean_r_between_variants=float(np.mean(rs)),
                   sd_of_variant_means=float(wide.mean().std(ddof=1)),
                   eta2_variant=float(ss_between / ss_total) if ss_total else np.nan)
        out.append(rec)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def fisher_ci(r: float, n: int, alpha: float = 0.05):
    if not np.isfinite(r) or n < 4:
        return (np.nan, np.nan)
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    se = 1 / np.sqrt(n - 3)
    crit = stats.norm.ppf(1 - alpha / 2)
    return (float(np.tanh(z - crit * se)), float(np.tanh(z + crit * se)))


def fisher_z_test(r1, n1, r2, n2):
    """Two-sample test that two independent correlations are equal."""
    if min(n1, n2) < 4 or not np.isfinite(r1) or not np.isfinite(r2):
        return np.nan, np.nan
    z1, z2 = np.arctanh(np.clip([r1, r2], -0.999999, 0.999999))
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z = (z1 - z2) / se
    return float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


def disattenuate(r_xy, rel_x, rel_y):
    """Classical correction for attenuation (Spearman, 1904)."""
    d = np.sqrt(max(rel_x, 1e-9) * max(rel_y, 1e-9))
    return float(r_xy / d) if d > 0 else np.nan
