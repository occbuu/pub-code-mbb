"""RQ4 evidence and the per-arm validity profile.

Produces what the batch analysis script does not: the LLM arm's reliability as a
stochastic instrument (t24, t25), the bias audits (t26, t27), and the per-arm
convergent-vs-halo profile that the four-arm results make necessary (t28), plus
figures f08-f10.

Run after scripts/run_analysis.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cft import viz
from cft.codebook import CONSTRUCTS
from cft.report import wide_to_long
from cft.validity import perturbation_effect

DERIVED, TABLES, FIGURES = ROOT / "derived", ROOT / "tables", ROOT / "figures"
TABLES.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
viz.apply_style()


def rd(name):
    p = DERIVED / name
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


# ---------------------------------------------------------------------------
# t24 / t25 -- the LLM as a stochastic rater
# ---------------------------------------------------------------------------
stab = rd("llm_run_stability.parquet")
sens = rd("llm_prompt_sensitivity.parquet")
if len(stab):
    stab.to_csv(TABLES / "t24_llm_run_stability.csv", index=False)
if len(sens):
    sens.to_csv(TABLES / "t25_llm_prompt_sensitivity.csv", index=False)

if len(stab) and len(sens):
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    x = np.arange(len(CONSTRUCTS)); w = 0.38
    s1 = stab.set_index("construct").reindex(CONSTRUCTS)
    s2 = sens.set_index("construct").reindex(CONSTRUCTS)
    ax.bar(x - w/2, s1["icc"], w*0.9, color=viz.METHOD_COLORS["LLM"],
           label="same prompt, 3 seeds (ICC)", edgecolor=viz.INK["surface"], lw=1)
    ax.bar(x + w/2, s2["mean_r_between_variants"], w*0.9,
           color=viz.METHOD_COLORS["TOPIC"], label="across 3 prompt wordings (r)",
           edgecolor=viz.INK["surface"], lw=1)
    for xi, v in zip(x - w/2, s1["icc"]):
        ax.text(xi, v + .02, f"{v:.2f}".replace("0.", "."), ha="center", fontsize=7,
                color=viz.INK["secondary"])
    for xi, v in zip(x + w/2, s2["mean_r_between_variants"]):
        ax.text(xi, v + .02, f"{v:.2f}".replace("0.", "."), ha="center", fontsize=7,
                color=viz.INK["secondary"])
    ax.set_xticks(x); ax.set_xticklabels(CONSTRUCTS); ax.set_ylim(0, 1.08)
    ax.set_ylabel("agreement")
    ax.set_title("The LLM is a stable rater")
    ax.legend(fontsize=7, loc="lower right"); viz.despine(ax)
    fig.tight_layout(); viz.save(fig, FIGURES / "f08_llm_stability.png")
    print("f08 written")

# ---------------------------------------------------------------------------
# t26 -- identity-cue and social-desirability perturbation audit
# ---------------------------------------------------------------------------
pert = rd("llm_perturbation.parquet")
if len(pert):
    base = wide_to_long(pert[pert.perturbation == "none"], "LLM", "pol")
    rows = []
    for cond, g in pert[pert.perturbation != "none"].groupby("perturbation"):
        e = perturbation_effect(base, wide_to_long(g, "LLM", "pol"))
        e.insert(0, "perturbation", cond)
        rows.append(e)
    pe = pd.concat(rows, ignore_index=True)
    pe.to_csv(TABLES / "t26_perturbation_audit.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    conds = ["set_a", "set_b", "set_c", "desirability_cue"]
    conds = [c for c in conds if c in set(pe.perturbation)]
    labels = {"set_a": "name set A", "set_b": "name set B", "set_c": "name set C",
              "desirability_cue": "desirability cue"}
    x = np.arange(len(CONSTRUCTS)); w = 0.8 / len(conds)
    palette = [viz.METHOD_COLORS[m] for m in viz.METHOD_ORDER]
    for i, cond in enumerate(conds):
        d = pe[pe.perturbation == cond].set_index("construct").reindex(CONSTRUCTS)
        ax.bar(x + i*w - .4 + w/2, d["d_z"], w*0.88, label=labels.get(cond, cond),
               color=palette[i % 4], edgecolor=viz.INK["surface"], lw=1)
    ax.axhline(0, color=viz.INK["axis"], lw=.8)
    for lvl in (0.2, -0.2):
        ax.axhline(lvl, ls=":", lw=.8, color=viz.INK["muted"])
    ax.text(len(CONSTRUCTS) - .55, .21, "d = .20 (small)", fontsize=7,
            color=viz.INK["muted"])
    ax.set_xticks(x); ax.set_xticklabels(CONSTRUCTS)
    ax.set_ylabel("standardised shift $d_z$")
    ax.set_title("Does an irrelevant cue move the measurement?")
    ax.legend(fontsize=7, ncol=2); viz.despine(ax)
    fig.tight_layout(); viz.save(fig, FIGURES / "f09_perturbation_audit.png")
    print("t26/f09 written:", len(pe), "rows")

# ---------------------------------------------------------------------------
# t27 -- language invariance
# ---------------------------------------------------------------------------
ml = rd("llm_multilingual.parquet")
if len(ml):
    key = "orig_id" if "orig_id" in ml.columns else "doc_id"
    mbase = wide_to_long(ml[ml.lang_variant == "en"].assign(doc_id=lambda d: d[key]),
                         "LLM", "pol")
    rows = []
    for lg, g in ml[ml.lang_variant != "en"].groupby("lang_variant"):
        e = perturbation_effect(mbase, wide_to_long(g.assign(doc_id=lambda d: d[key]),
                                                    "LLM", "pol"))
        e.insert(0, "language", lg)
        rows.append(e)
    mlt = pd.concat(rows, ignore_index=True)

    # correlation between the English score and the translated score
    corr_rows = []
    for lg, g in ml[ml.lang_variant != "en"].groupby("lang_variant"):
        a = ml[ml.lang_variant == "en"].set_index(key)
        b = g.set_index(key)
        for c in CONSTRUCTS:
            j = pd.concat([a[f"{c}_pol"], b[f"{c}_pol"]], axis=1, join="inner").dropna()
            j.columns = ["en", "tr"]
            corr_rows.append(dict(language=lg, construct=c, n=len(j),
                                  r_with_english=float(np.corrcoef(j.en, j.tr)[0, 1])
                                  if len(j) > 10 and j.tr.std() > 0 else np.nan,
                                  coverage_en=100 * float(a[f"{c}_pol"].notna().mean()),
                                  coverage_translated=100 * float(b[f"{c}_pol"].notna().mean())))
    mlt = mlt.merge(pd.DataFrame(corr_rows), on=["language", "construct"], how="left")
    mlt.to_csv(TABLES / "t27_multilingual_invariance.csv", index=False)
    print("t27 written:", len(mlt), "rows")

nat = rd("llm_native_nonenglish.parquet")
if len(nat) and "lang" in nat.columns:
    rows = []
    for c in CONSTRUCTS:
        for lg, g in nat.groupby("lang"):
            if len(g) < 15:
                continue
            rows.append(dict(construct=c, lang=lg, n=int(len(g)),
                             coverage=100 * float(g[f"{c}_pol"].notna().mean()),
                             mean_pol=float(g[f"{c}_pol"].mean())))
    if rows:
        pd.DataFrame(rows).to_csv(TABLES / "t27b_native_nonenglish.csv", index=False)
        print("t27b written")

# ---------------------------------------------------------------------------
# t28 / f10 -- the per-arm validity profile (convergence vs halo)
# ---------------------------------------------------------------------------
cells = pd.read_csv(TABLES / "t12_mtmm_cells.csv")
cells = cells[cells.corpus != "all"]
conv = pd.read_csv(TABLES / "t15_convergent_closed_form.csv")
cov = pd.read_csv(TABLES / "t06_coverage_by_method.csv")
crit = pd.read_csv(TABLES / "t16_criterion_auc.csv")

rows = []
for m in viz.METHOD_ORDER:
    halo = cells[(cells.kind == "heterotrait-monomethod") & (cells.method_a == m)]["r"]
    cvg = cells[(cells.kind == "monotrait-heteromethod") &
                ((cells.method_a == m) | (cells.method_b == m))]["r"]
    rows.append(dict(
        method=m,
        coverage=float(cov[cov.method == m]["coverage"].mean()),
        convergent_with_closed_form=float(conv[conv.method == m]["r"].mean()),
        convergent_with_other_arms=float(cvg.mean()),
        heterotrait_monomethod_halo=float(halo.mean()),
        discriminant_margin=float(cvg.mean() - halo.mean()),
        criterion_auc=float(crit[crit.method == m]["auc"].mean()),
    ))
prof = pd.DataFrame(rows)
prof.to_csv(TABLES / "t28_arm_validity_profile.csv", index=False)
print(prof.round(3).to_string(index=False))

fig, ax = plt.subplots(figsize=(5.8, 4.2))
for _, r_ in prof.iterrows():
    ax.scatter(r_["heterotrait_monomethod_halo"], r_["convergent_with_closed_form"],
               s=60 + 3.2 * r_["coverage"], color=viz.METHOD_COLORS[r_["method"]],
               edgecolor=viz.INK["surface"], linewidth=1.5, zorder=3,
               label=f'{viz.METHOD_LABELS[r_["method"]]} ({r_["coverage"]:.0f}% coverage)')
    ax.annotate(r_["method"],
                (r_["heterotrait_monomethod_halo"], r_["convergent_with_closed_form"]),
                textcoords="offset points", xytext=(0, -16), ha="center", fontsize=8,
                color=viz.INK["secondary"])
lim = [0, 1.0]
ax.plot(lim, lim, ls=":", lw=1, color=viz.INK["muted"], zorder=1)
ax.text(0.50, 0.535, "convergence = halo", fontsize=7, rotation=31,
        color=viz.INK["muted"])
ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
ax.set_xlabel("method halo: mean r between two different constructs\nmeasured by the same arm")
ax.set_ylabel("convergent validity with the\nclosed-form response")
ax.set_title("The validity paradox")
ax.legend(fontsize=7, loc="upper left", labelspacing=0.9,
          borderpad=0.6); viz.despine(ax)
fig.tight_layout(); viz.save(fig, FIGURES / "f10_validity_paradox.png")
print("t28/f10 written")
