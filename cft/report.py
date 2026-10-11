"""Tidy conversion and the reporting checklist.

The checklist is the practical deliverable of the paper: the minimum a study
that measures a theoretical construct from text should report for a reader to
judge whether the score deserves the construct's name.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .codebook import CONSTRUCTS


def wide_to_long(scores: pd.DataFrame, method: str, kind: str = "pol",
                 id_col: str = "doc_id") -> pd.DataFrame:
    """``doc_id, <C>_pol...`` -> ``doc_id, method, construct, value``."""
    cols = [f"{c}_{kind}" for c in CONSTRUCTS if f"{c}_{kind}" in scores.columns]
    d = scores[[id_col] + cols].melt(id_vars=id_col, var_name="construct",
                                     value_name="value")
    d["construct"] = d["construct"].str.replace(f"_{kind}$", "", regex=True)
    d["method"] = method
    d["kind"] = kind
    return d[[id_col, "method", "construct", "kind", "value"]]


def stack_methods(frames: dict, kind: str = "pol") -> pd.DataFrame:
    return pd.concat([wide_to_long(v, k, kind) for k, v in frames.items()],
                     ignore_index=True)


def describe_corpus(docs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cp, g in docs.groupby("corpus"):
        b = g[g.split.isin(["bench", "llm_core"])]
        rows.append(dict(
            corpus=cp, domain=g["domain"].iloc[0], n_pool=int(len(g)),
            n_bench=int(len(b)), n_llm_core=int((g.split == "llm_core").sum()),
            median_tokens=float(g["n_tokens"].median()),
            p90_tokens=float(g["n_tokens"].quantile(0.90)),
            pct_english=100 * float((g["lang"] == "en").mean()),
            closed_form=g["closed_form_type"].iloc[0],
            criterion=g["criterion_type"].iloc[0],
            criterion_base_rate=float(g["criterion_binary"].mean())
            if g["criterion_binary"].notna().any() else np.nan))
    return pd.DataFrame(rows)


CHECKLIST = [
    ("Construct definition",
     "A written definition and coding rule for every construct, fixed before "
     "scoring, and reported in full (not only cited)."),
    ("Instrument provenance",
     "For a dictionary: the term list. For a topic model: K, the mapping rule "
     "and who applied it. For an encoder: the training labels and their source. "
     "For an LLM: model id, revision, temperature, top_p, seed, the verbatim "
     "prompt and the run date."),
    ("Coverage",
     "The share of documents in which the construct is measurable at all, and "
     "how documents with no evidence are handled (missing, not zero)."),
    ("Reliability -- internal",
     "Agreement of the score with an independent method aimed at the same "
     "construct (Krippendorff's alpha or the between-method correlation)."),
    ("Reliability -- stability",
     "For stochastic instruments: agreement across >= 3 independent runs with the "
     "same prompt, and across >= 2 prompt wordings."),
    ("Convergent validity",
     "Correlation with a measure of the same construct obtained by a different "
     "method -- ideally a self-report answered by the same person."),
    ("Discriminant validity",
     "Evidence that the construct separates from its neighbours: HTMT < 0.85, or "
     "a Campbell-Fiske comparison against heterotrait cells."),
    ("Method variance",
     "The heterotrait-monomethod minus heterotrait-heteromethod difference, or a "
     "CTCM/CTCU model, so that shared-pipeline inflation is visible."),
    ("Criterion validity",
     "Association with an observed behaviour, not only with another text score."),
    ("Invariance",
     "Evidence that the instrument behaves the same way across the corpora, "
     "languages or groups being compared before those groups are compared."),
    ("Error propagation",
     "Structural coefficients corrected for attenuation, or a sensitivity "
     "analysis over the plausible reliability range."),
    ("Bias audit",
     "A perturbation test for identity cues and desirability cues, and a "
     "multilingual check when the corpus is not monolingual."),
]


def checklist_frame() -> pd.DataFrame:
    return pd.DataFrame(CHECKLIST, columns=["requirement", "what_to_report"])


def fmt_r(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    s = f"{x:.{nd}f}"
    return s.replace("0.", ".", 1) if s.startswith(("0.", "-0.")) else s


def stars(p):
    if p is None or not np.isfinite(p):
        return ""
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else ""
