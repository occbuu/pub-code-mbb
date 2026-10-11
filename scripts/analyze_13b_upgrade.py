"""CPU audit: complete-case PCA, bootstrap, one-factor ML, corpus conditioning.
Never interprets PC1 total variance as common-factor ECV or omega hierarchical.
"""
from __future__ import annotations
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[1]
CS = ["ATT", "PE", "EE", "SI", "BI"]
COLS = [c + "_pol" for c in CS]

def pc1(x):
    if len(x) < 30 or np.any(np.std(x, axis=0) < 1e-10): return np.nan
    return float(np.linalg.eigvalsh(np.corrcoef(x, rowvar=False))[-1] / x.shape[1])

def one_factor(x):
    """Gaussian ML on standardized complete cases; ordinal robustness not claimed."""
    n, p = x.shape
    z = (x - x.mean(0)) / x.std(0, ddof=1)
    s = z.T @ z / n
    sign, logdet = np.linalg.slogdet(s)
    if sign <= 0: return {"factor_status": "singular sample covariance"}
    def objective(v):
        lam, theta = v[:p], np.exp(v[p:])
        sigma = np.outer(lam, lam) + np.diag(theta)
        return np.linalg.slogdet(sigma)[1] + np.trace(np.linalg.solve(sigma, s)) - logdet - p
    vals, vecs = np.linalg.eigh(s)
    start = vecs[:, -1] * np.sqrt(max(vals[-1] - .3, .05))
    fit = minimize(objective, np.r_[start, np.log(np.maximum(np.diag(s) - start**2, .1))],
                   method="L-BFGS-B", bounds=[(-2, 2)] * p + [(-9, 3)] * p)
    lam, theta = fit.x[:p], np.exp(fit.x[p:])
    sigma = np.outer(lam, lam) + np.diag(theta)
    df = p * (p + 1) // 2 - 2 * p
    stat = max(0, (n - 1) * fit.fun)
    baseline = max(0, (n - 1) * (np.log(np.diag(s)).sum() - logdet))
    base_df = p * (p - 1) // 2
    cfi = 1 - max(stat - df, 0) / max(baseline - base_df, stat - df, 1e-9)
    residual = s / np.sqrt(np.outer(np.diag(s), np.diag(s))) - sigma / np.sqrt(np.outer(np.diag(sigma), np.diag(sigma)))
    return {"factor_status": "converged" if fit.success else str(fit.message),
            "chi2_gaussian": stat, "df": df, "p_gaussian": float(chi2.sf(stat, df)),
            "cfi_gaussian": cfi, "rmsea_gaussian": np.sqrt(max((stat - df) / (df * (n - 1)), 0)),
            "srmr": np.sqrt(np.mean(residual[np.tril_indices(p)]**2)),
            "omega_one_factor": float(lam.sum()**2 / (lam.sum()**2 + theta.sum())),
            "min_uniqueness": float(theta.min()), "loadings": json.dumps(lam.tolist())}

def profile(frame, label, corpus, reps):
    x = frame[COLS]
    nmat = x.notna().astype(int).T @ x.notna().astype(int)
    pair_counts = [int(nmat.iloc[i, j]) for i, j in itertools.combinations(range(5), 2)]
    correlations = x.corr(min_periods=30).to_numpy()
    pairs = correlations[np.triu_indices(5, 1)]
    complete = x.dropna().to_numpy(float)
    row = {"method": label, "corpus": corpus, "n_docs": len(x), "n_complete": len(complete),
           "coverage": float(x.notna().to_numpy().mean()),
           "pair_min_n": min(pair_counts), "n_estimable_pairs": int(np.isfinite(pairs).sum()),
           "halo_pairwise": float(np.nanmean(pairs)) if np.isfinite(pairs).any() else np.nan,
           "pc1_total_variance_complete": pc1(complete)}
    if np.isfinite(row["pc1_total_variance_complete"]):
        rng = np.random.default_rng(20261011)
        boot = [pc1(complete[rng.integers(0, len(complete), len(complete))]) for _ in range(reps)]
        finite = np.array(boot)[np.isfinite(boot)]
        row["bootstrap_valid"] = len(finite)
        if len(finite): row["pc1_ci_low"], row["pc1_ci_high"] = np.quantile(finite, [.025, .975])
        row.update(one_factor(complete))
    return row

def heldout_criterion(frame, metadata):
    """General-factor vs all-five prediction, PCA fit on training only.
    Does not use star rating as a gold standard for specific traits.
    """
    a = frame[COLS].join(metadata[["closed_form", "corpus"]]).dropna()
    rows = []
    for corpus, g in a.groupby("corpus"):
        if len(g) < 60: continue
        x, y = g[COLS].to_numpy(), g.closed_form.to_numpy()
        ids = np.random.default_rng(20261011).permutation(len(g))
        folds = np.array_split(ids, 5)
        predictions = {"general": np.zeros(len(g)), "all_five": np.zeros(len(g))}
        for test in folds:
            train = np.setdiff1d(ids, test)
            mu, sd = x[train].mean(0), x[train].std(0)
            if (sd < 1e-10).any(): break
            z, zt = (x[train] - mu) / sd, (x[test] - mu) / sd
            _, _, vt = np.linalg.svd(z, full_matrices=False)
            for name, xx, tt in [("general", (z @ vt[0])[:, None], (zt @ vt[0])[:, None]), ("all_five", z, zt)]:
                beta = np.linalg.lstsq(np.c_[np.ones(len(train)), xx], y[train], rcond=None)[0]
                predictions[name][test] = np.c_[np.ones(len(test)), tt] @ beta
        else:
            denom = np.sum((y - y.mean())**2)
            r2 = {name: 1 - np.sum((y - pred)**2) / denom for name, pred in predictions.items()} if denom > 0 else {}
            rows.append({"corpus": corpus, "n_complete_criterion": len(g), **r2,
                         "delta_r2": r2.get("all_five", np.nan) - r2.get("general", np.nan)})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1000)
    args = ap.parse_args()
    out = ROOT / "Paper13B/upgrade"
    out.mkdir(parents=True, exist_ok=True)
    metadata = pd.concat([pd.read_parquet(f) for f in sorted((ROOT / "derived").glob("corpus_*.parquet"))]).set_index("doc_id")
    mit = pd.read_parquet(ROOT / "derived/mitigation_scores.parquet")
    ids = sorted(mit.doc_id.unique())
    frames = {}
    for name, file in [("Qwen_joint", "llm_scores"), ("Mistral_joint", "llm2_scores"), ("dictionary", "scores_dict"), ("encoder", "encoder_scores")]:
        d = pd.read_parquet(ROOT / f"derived/{file}.parquet")
        if "run" in d: d = d[d.run.astype(str).str.startswith("zeroshot")]
        frames[name] = d.groupby("doc_id")[COLS].median().reindex(ids)
    for name, d in mit.groupby("condition"):
        frames[name] = d.pivot(index="doc_id", columns="construct", values="pol").rename(columns={c:c+"_pol" for c in CS}).reindex(ids)[COLS]
    archive = ROOT / "derived/local_upgrade/generations.jsonl"
    if archive.exists():
        lines = archive.read_text(encoding="utf-8").splitlines()
        raw = []
        for i, line in enumerate(lines):
            try: raw.append(json.loads(line))
            except json.JSONDecodeError:
                if i != len(lines) - 1: raise
        raw = pd.DataFrame([r for r in raw if not r.get("error")]).drop_duplicates("key", keep="last")
        if len(raw):
            keep = ["key", "doc_id", "corpus", "model", "condition", "parsed", "schema_valid", "truncated", "elapsed_s"] + COLS
            raw.reindex(columns=keep).to_csv(out / "local_scores.csv", index=False)
            raw.groupby(["model", "condition"]).agg(n_attempts=("key", "size"), schema_valid_rate=("schema_valid", "mean"), truncated_rate=("truncated", "mean")).to_csv(out / "local_parse_audit.csv")
            for (model, condition), g in raw.groupby(["model", "condition"]):
                frames[model+"/"+condition] = g.set_index("doc_id").reindex(columns=COLS)
    rows, criteria = [], []
    for name, d in frames.items():
        d = d.join(metadata[["corpus"]])
        rows.append(profile(d, name, "pooled", args.reps))
        for corpus, g in d.groupby("corpus"): rows.append(profile(g, name, corpus, args.reps))
        for r in heldout_criterion(d.drop(columns="corpus"), metadata): criteria.append({"method": name, **r})
    pd.DataFrame(rows).to_csv(out / "structure_audit.csv", index=False)
    pd.DataFrame(criteria).to_csv(out / "heldout_global_criterion.csv", index=False)
    # Pairwise common support avoids calling changed missingness discrimination.
    paired = []
    comparisons = [("Qwen_joint", c) for c in ["isolated", "evidence", "gate"]]
    comparisons += [(k, k.replace("/joint", "/contrastive")) for k in frames if k.endswith("/joint")]
    for base, alternate in comparisons:
        if alternate not in frames: continue
        left, right = frames[base], frames[alternate]
        for corpus in ["pooled"] + sorted(metadata.corpus.unique().tolist()):
            eligible = left.index.intersection(right.index)
            if corpus != "pooled": eligible = eligible.intersection(metadata.index[metadata.corpus == corpus])
            for a, b in itertools.combinations(COLS, 2):
                joined = left.loc[eligible, [a, b]].join(right.loc[eligible, [a, b]], lsuffix="_base", rsuffix="_alt").dropna()
                row = {"baseline": base, "alternate": alternate, "corpus": corpus, "pair": a+"/"+b, "n_common": len(joined)}
                if len(joined) >= 30 and (joined.std() > 0).all():
                    arr = joined.to_numpy()
                    def difference(z):
                        if (z.std(0) == 0).any(): return np.nan
                        return np.corrcoef(z[:, 2:4], rowvar=False)[0, 1] - np.corrcoef(z[:, :2], rowvar=False)[0, 1]
                    row["halo_delta_on_common_support"] = difference(arr)
                    rng = np.random.default_rng(20261011)
                    boot = np.array([difference(arr[rng.integers(0, len(arr), len(arr))]) for _ in range(args.reps)])
                    finite = boot[np.isfinite(boot)]
                    if len(finite): row["delta_ci_low"], row["delta_ci_high"] = np.quantile(finite, [.025, .975])
                paired.append(row)
    pd.DataFrame(paired).to_csv(out / "matched_support_prompt_comparison.csv", index=False)
    human = metadata.loc[ids, ["corpus", "text"]].reset_index()
    for c in CS:
        for field in ["sal", "lvl", "quote"]: human[c+"_"+field] = ""
    human.to_csv(out / "human_1000_blinded.csv", index=False, encoding="utf-8-sig")
    metadata.loc[ids].groupby("corpus").agg(n=("text", "size"), median_tokens=("n_tokens", "median"), total_tokens=("n_tokens", "sum")).to_csv(out / "study2_corpus_summary.csv")
    (out / "identification.json").write_text(json.dumps({"observed_indicators": 5,
        "bifactor_status": "not identified: one indicator per specific factor; specific variance confounded with indicator error",
        "omega_h": None, "omega_hs": None, "ECV": None, "PUC": None,
        "requirements": "Multiple substantively distinct indicators per construct (typically >=3), validated coding, identified loading/error restrictions. Seeds are repeat ratings, not independent items.",
        "limitations": "One-factor Gaussian fit is diagnostic for ordinal, selected complete cases; bootstrap is document-level; conditional missingness and author clusters need separate scrutiny."}, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows)[["method", "corpus", "n_docs", "n_complete", "halo_pairwise", "pc1_total_variance_complete"]].to_string(index=False))

if __name__ == "__main__": main()
