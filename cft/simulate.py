"""RQ3 -- how measurement error in a text-based construct score propagates into
(i) structural (SEM/regression) coefficients and (ii) predictive performance.

Two error regimes are simulated, because text-based measurement produces both:

*random error*   -- the score is the true construct plus independent noise; the
                    classical result is attenuation of the standardised path by
                    ``sqrt(rel)`` for a single predictor.
*method error*   -- every construct measured by the *same* pipeline shares a
                    method factor (the same dictionary, the same LLM, the same
                    prompt), which inflates correlations among constructs and can
                    bias a path *upward* while attenuating another.

The second regime is the one the applied literature ignores, and it is the
reason a single "reliability" number is not sufficient guidance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _standardise(x):
    return (x - x.mean()) / x.std(ddof=1)


def contaminate(true, rel, method_factor=None, method_load=0.0, rng=None):
    """Return an observed score with reliability ``rel`` and, optionally, a
    shared method component with loading ``method_load``."""
    rng = rng or np.random.default_rng()
    t = _standardise(np.asarray(true, float))
    rel = float(np.clip(rel, 1e-4, 1.0))
    signal = np.sqrt(rel) * t
    var_left = max(1.0 - rel, 0.0)
    mvar = min(method_load ** 2, var_left)
    evar = var_left - mvar
    obs = signal + np.sqrt(evar) * rng.standard_normal(len(t))
    if method_factor is not None and mvar > 0:
        obs = obs + np.sqrt(mvar) * _standardise(np.asarray(method_factor, float))
    return _standardise(obs)


# ---------------------------------------------------------------------------
# Study A: attenuation of a structural path
# ---------------------------------------------------------------------------
def _cell_seed(seed, tag, rel, load, rep) -> int:
    """Independent seed per Monte-Carlo cell so process pools stay reproducible."""
    return int((int(seed) + 1_000_003 * tag + 10_007 * int(round(float(rel) * 1000))
                + 1009 * int(round(float(load) * 1000)) + int(rep)) % (2**32 - 1))


def _map_chunks(fn, tasks, n_jobs):
    if n_jobs is None:
        from .resources import cpu_workers
        n_jobs = cpu_workers()
    n_jobs = max(1, int(n_jobs))
    if n_jobs == 1 or len(tasks) <= 1:
        parts = [fn(t) for t in tasks]
    else:
        from concurrent.futures import ProcessPoolExecutor
        from .resources import worker_init_single_thread
        with ProcessPoolExecutor(max_workers=min(n_jobs, len(tasks)),
                                 initializer=worker_init_single_thread) as ex:
            parts = list(ex.map(fn, tasks))
    rows = [r for p in parts for r in p]
    return rows


def _att_chunk(payload):
    beta, n, n_controls, control_r, method_load, seed, rel, rep0, rep1 = payload
    rows = []
    for rep in range(rep0, rep1):
        rng = np.random.default_rng(_cell_seed(seed, 1, rel, method_load, rep))
        x = rng.standard_normal(n)
        ctrls = [control_r * x + np.sqrt(1 - control_r ** 2) * rng.standard_normal(n)
                 for _ in range(n_controls)]
        y = beta * _standardise(x) + np.sqrt(1 - beta ** 2) * rng.standard_normal(n)
        mf = rng.standard_normal(n) if method_load else None
        xo = contaminate(x, rel, method_factor=mf, method_load=method_load, rng=rng)
        X = np.column_stack([np.ones(n), xo] + [_standardise(c) for c in ctrls])
        yz = _standardise(y)
        coef, *_ = np.linalg.lstsq(X, yz, rcond=None)
        b = coef[1]
        se = np.sqrt(np.sum((yz - X @ coef) ** 2) / (n - X.shape[1]) *
                     np.linalg.inv(X.T @ X)[1, 1])
        rows.append(dict(rel=float(rel), rep=rep, beta_hat=float(b),
                         se=float(se), t=float(b / se) if se else np.nan,
                         sig=bool(abs(b / se) > 1.96) if se else False))
    return rows


def simulate_path_attenuation(beta=0.35, rel_grid=None, n=1000, reps=500,
                              n_controls=1, control_r=0.4, method_load=0.0,
                              seed=20260909, n_jobs=None, chunk_reps=40) -> pd.DataFrame:
    """X -> Y with true standardised path ``beta``; X measured with reliability
    ``rel``; ``n_controls`` correlated covariates measured perfectly.

    Returns one row per (rel, rep) with the estimated path, so that both the
    mean bias and the sampling variability can be plotted.
    """
    rel_grid = rel_grid if rel_grid is not None else np.arange(0.30, 1.001, 0.05)
    tasks = []
    for rel in rel_grid:
        for r0 in range(0, reps, chunk_reps):
            tasks.append((beta, n, n_controls, control_r, method_load, seed,
                          float(rel), r0, min(r0 + chunk_reps, reps)))
    rows = _map_chunks(_att_chunk, tasks, n_jobs)
    df = pd.DataFrame(rows)
    df["beta_true"] = beta
    df["attenuation"] = df["beta_hat"] / beta
    return df.sort_values(["rel", "rep"]).reset_index(drop=True)


def attenuation_summary(sim: pd.DataFrame) -> pd.DataFrame:
    g = sim.groupby("rel")
    out = g.agg(beta_true=("beta_true", "first"),
                mean_beta_hat=("beta_hat", "mean"),
                sd_beta_hat=("beta_hat", "std"),
                mean_attenuation=("attenuation", "mean"),
                power=("sig", "mean")).reset_index()
    out["pct_bias"] = 100 * (out["mean_beta_hat"] - out["beta_true"]) / out["beta_true"]
    out["theoretical_attenuation"] = np.sqrt(out["rel"])
    return out


# ---------------------------------------------------------------------------
# Study B: two constructs sharing a method factor
# ---------------------------------------------------------------------------
def _mb_chunk(payload):
    beta1, beta2, rel, n, seed, load, rep0, rep1 = payload
    rows = []
    for rep in range(rep0, rep1):
        rng = np.random.default_rng(_cell_seed(seed, 2, rel, load, rep))
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        y = beta1 * _standardise(x1) + beta2 * _standardise(x2)
        y = y + np.sqrt(max(1 - beta1 ** 2 - beta2 ** 2, 1e-6)) * rng.standard_normal(n)
        mf = rng.standard_normal(n)
        o1 = contaminate(x1, rel, mf, load, rng)
        o2 = contaminate(x2, rel, mf, load, rng)
        oy = contaminate(y, rel, mf, load, rng)
        X = np.column_stack([np.ones(n), o1, o2])
        coef, *_ = np.linalg.lstsq(X, oy, rcond=None)
        resid = oy - X @ coef
        s2 = resid @ resid / (n - 3)
        cov = s2 * np.linalg.inv(X.T @ X)
        rows.append(dict(load=float(load), rep=rep,
                         b1=float(coef[1]), b2=float(coef[2]),
                         t2=float(coef[2] / np.sqrt(cov[2, 2])),
                         false_pos=bool(abs(coef[2] / np.sqrt(cov[2, 2])) > 1.96)))
    return rows


def simulate_method_bias(beta1=0.30, beta2=0.00, rel=0.60, load_grid=None,
                         n=1000, reps=400, seed=20260909, n_jobs=None,
                         chunk_reps=40) -> pd.DataFrame:
    """Y = b1*X1 + b2*X2 + e.  X1 and X2 are measured by the *same* pipeline and
    therefore share a method factor with loading ``load``; Y is measured by the
    same pipeline too (as when both predictor and outcome are extracted from the
    same document).  Shows how a null path (``beta2 = 0``) becomes significant."""
    load_grid = load_grid if load_grid is not None else np.arange(0.0, 0.61, 0.1)
    tasks = []
    for load in load_grid:
        for r0 in range(0, reps, chunk_reps):
            tasks.append((beta1, beta2, rel, n, seed, float(load), r0,
                          min(r0 + chunk_reps, reps)))
    rows = _map_chunks(_mb_chunk, tasks, n_jobs)
    df = pd.DataFrame(rows)
    df["beta1_true"] = beta1
    df["beta2_true"] = beta2
    return df.sort_values(["load", "rep"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Study C: predictive performance
# ---------------------------------------------------------------------------
def _auc_chunk(payload):
    from sklearn.metrics import roc_auc_score
    auc_true, d, n, seed, rel, rep0, rep1 = payload
    rows = []
    for rep in range(rep0, rep1):
        rng = np.random.default_rng(_cell_seed(seed, 3, rel, 0, rep))
        y = rng.integers(0, 2, n)
        x = rng.standard_normal(n) + d * y
        xo = contaminate(x, rel, rng=rng)
        rows.append(dict(rel=float(rel), rep=rep,
                         auc=float(roc_auc_score(y, xo))))
    return rows


def simulate_auc_loss(auc_true=0.75, rel_grid=None, n=4000, reps=200,
                      seed=20260909, n_jobs=None, chunk_reps=40) -> pd.DataFrame:
    """A binary outcome generated from the *true* construct; the classifier only
    sees the noisy text-based score.  Reports the AUC actually achieved."""
    rel_grid = rel_grid if rel_grid is not None else np.arange(0.30, 1.001, 0.05)
    d = stats.norm.ppf(auc_true) * np.sqrt(2)
    tasks = []
    for rel in rel_grid:
        for r0 in range(0, reps, chunk_reps):
            tasks.append((auc_true, float(d), n, seed, float(rel), r0,
                          min(r0 + chunk_reps, reps)))
    rows = _map_chunks(_auc_chunk, tasks, n_jobs)
    df = pd.DataFrame(rows)
    df["auc_true"] = auc_true
    return df.sort_values(["rel", "rep"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# practice guidance
# ---------------------------------------------------------------------------
def practice_table(att: pd.DataFrame, auc: pd.DataFrame,
                   rels=(0.40, 0.50, 0.60, 0.70, 0.80, 0.90)) -> pd.DataFrame:
    """The recommendation table: at reliability X, expect Y% attenuation, this
    much power loss, this much AUC loss, and this correction factor."""
    a = att.groupby("rel").agg(mean_attenuation=("attenuation", "mean"),
                               power=("sig", "mean")).reset_index()
    b = auc.groupby("rel").agg(mean_auc=("auc", "mean"),
                               auc_true=("auc_true", "first")).reset_index()
    rows = []
    for r in rels:
        ai = a.iloc[(a["rel"] - r).abs().argmin()]
        bi = b.iloc[(b["rel"] - r).abs().argmin()]
        rows.append(dict(
            reliability=r,
            pct_attenuation=100 * (1 - ai["mean_attenuation"]),
            power_at_n1000=ai["power"],
            auc_obtained=bi["mean_auc"],
            auc_lost=bi["auc_true"] - bi["mean_auc"],
            correction_factor=1 / np.sqrt(r),
            n_multiplier_for_equal_power=1 / r,
        ))
    return pd.DataFrame(rows)
