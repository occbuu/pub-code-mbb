"""Corpus normalisation.

Every corpus is reduced to one schema so that the four measurement arms and all
validity analyses are corpus-agnostic:

===================  ====================================================
doc_id               ``<corpus>_<n>`` -- stable across the whole project
corpus               yelp | amazon | regulations | bluesky
domain               service | product | public | social
text                 the free-text document, unmodified
n_chars, n_tokens    length descriptors
lang                 ISO code from langdetect ('und' when undetermined)
closed_form          the *same respondent's* closed-form response, rescaled
                     to [-1, 1]; NaN when the corpus has none
closed_form_raw      the response on its original scale
closed_form_type     star_rating | none
criterion_binary     an observed behaviour of the same respondent, later in
                     time (repeat patronage / repeat purchase); NaN if none
criterion_type       repeat_business | repeat_product | none
author_id, item_id   pseudonymous identifiers (hashed), for clustering
ts                   timestamp of the document
split                pool | bench | llm_core   (nested: llm_core c bench c pool)
===================  ====================================================

``bench`` is the benchmark set every arm scores; ``llm_core`` is the smaller set
on which the full LLM protocol (3 prompt variants x 3 independent runs) is run.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA = ["doc_id", "corpus", "domain", "text", "n_chars", "n_tokens", "lang",
          "closed_form", "closed_form_raw", "closed_form_type",
          "criterion_binary", "criterion_type", "author_id", "item_id", "ts",
          "split"]

DOMAIN = {"yelp": "service", "amazon": "product",
          "regulations": "public", "bluesky": "social"}

_WS = re.compile(r"\s+")
_URL = re.compile(r"https?://\S+|www\.\S+")


def clean_text(t: str, keep_urls=False) -> str:
    t = str(t).replace("\r", " ").replace("\n", " ")
    if not keep_urls:
        t = _URL.sub(" ", t)
    return _WS.sub(" ", t).strip()


def h(x: str, n=12) -> str:
    return hashlib.blake2b(str(x).encode("utf8"), digest_size=8).hexdigest()[:n]


def detect_lang(t: str) -> str:
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 0
        if len(t) < 12:
            return "und"
        return detect(t)
    except Exception:
        return "und"


def detect_lang_many(texts, n_jobs: int | None = None) -> list[str]:
    """Process-parallel language ID.  langdetect is not thread-safe."""
    texts = list(texts)
    if n_jobs is None:
        from .resources import cpu_workers
        n_jobs = cpu_workers() if len(texts) >= 200 else 1
    n_jobs = max(1, int(n_jobs))
    if n_jobs == 1 or len(texts) < 200:
        return [detect_lang(t) for t in texts]
    from concurrent.futures import ProcessPoolExecutor
    from .resources import worker_init_single_thread
    chunk = max(16, len(texts) // (n_jobs * 8))
    with ProcessPoolExecutor(max_workers=n_jobs,
                             initializer=worker_init_single_thread) as ex:
        return list(ex.map(detect_lang, texts, chunksize=chunk))


def stars_to_pm1(s, lo=1, hi=5):
    return 2 * (np.asarray(s, float) - lo) / (hi - lo) - 1


def finalise(df: pd.DataFrame, corpus: str, n_bench: int, n_llm_core: int,
             seed: int = 20260909, stratify: str | None = "closed_form_raw",
             detect_language: bool = True) -> pd.DataFrame:
    """Add ids, length, language and the nested split flags."""
    rng = np.random.default_rng(seed)
    df = df.reset_index(drop=True).copy()
    df["corpus"] = corpus
    df["domain"] = DOMAIN.get(corpus, "other")
    df["doc_id"] = [f"{corpus}_{i:07d}" for i in range(len(df))]
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() >= 20].reset_index(drop=True)
    df["doc_id"] = [f"{corpus}_{i:07d}" for i in range(len(df))]
    df["n_chars"] = df["text"].str.len()
    df["n_tokens"] = df["text"].str.split().map(len)
    df["lang"] = detect_lang_many(df["text"].tolist()) if detect_language else "und"

    # nested splits, stratified where a closed-form response exists
    idx = np.arange(len(df))
    if stratify and stratify in df and df[stratify].notna().any():
        parts = []
        groups = df.groupby(df[stratify].fillna(-99))
        per = max(int(np.ceil(n_bench / max(len(groups), 1))), 1)
        for _, g in groups:
            take = min(per, len(g))
            parts.append(rng.choice(g.index.values, size=take, replace=False))
        bench = np.concatenate(parts) if parts else idx[:n_bench]
    else:
        bench = rng.choice(idx, size=min(n_bench, len(idx)), replace=False)
    bench = np.array(sorted(bench))[:n_bench]
    core = rng.choice(bench, size=min(n_llm_core, len(bench)), replace=False)

    df["split"] = "pool"
    df.loc[df.index.isin(bench), "split"] = "bench"
    df.loc[df.index.isin(core), "split"] = "llm_core"
    for c in SCHEMA:
        if c not in df:
            df[c] = np.nan
    return df[SCHEMA]


# ---------------------------------------------------------------------------
# Amazon Reviews 2023 (csv inside a zip)
# ---------------------------------------------------------------------------
def build_amazon(zip_path, n_pool=30000, n_bench=1200, n_llm_core=400,
                 seed=20260909, member=None, detect_language=True) -> pd.DataFrame:
    zf = zipfile.ZipFile(zip_path)
    member = member or [i.filename for i in zf.infolist() if i.filename.endswith(".csv")][0]

    # pass 1 -- repeat behaviour of each reviewer
    user_item = {}
    user_n = {}
    with zf.open(member) as f:
        for row in csv.DictReader(io.TextIOWrapper(f, "utf8")):
            u = row.get("user_id") or ""
            it = row.get("parent_asin") or row.get("asin") or ""
            user_n[u] = user_n.get(u, 0) + 1
            k = (u, it)
            user_item[k] = user_item.get(k, 0) + 1

    rng = np.random.default_rng(seed)
    keep, i = [], 0
    with zf.open(member) as f:
        for row in csv.DictReader(io.TextIOWrapper(f, "utf8")):
            i += 1
            if len(keep) < n_pool:
                keep.append(row)
            else:                                   # reservoir sampling
                j = rng.integers(0, i)
                if j < n_pool:
                    keep[j] = row

    df = pd.DataFrame(keep)
    txt = (df["title"].fillna("") + ". " + df["text"].fillna("")).str.strip(". ")
    # Leakage control: Amazon review titles are often literally the rating
    # ("Five Stars", "3.0 out of 5 stars").  Leaving them in would let any
    # method read the closed-form response off the text and would inflate every
    # convergent-validity estimate, so the rating phrase is removed.
    txt = txt.str.replace(
        r"(?i)\b(one|two|three|four|five|1|2|3|4|5)(\.\d)?\s*(out of\s*5\s*)?stars?\b",
        " ", regex=True)
    out = pd.DataFrame({
        "text": txt,
        "closed_form_raw": pd.to_numeric(df["rating"], errors="coerce"),
        "closed_form_type": "star_rating",
        "author_id": df["user_id"].map(h),
        "item_id": df["parent_asin"].map(h),
        "ts": df["timestamp"],
    })
    out["closed_form"] = stars_to_pm1(out["closed_form_raw"])
    out["criterion_binary"] = [
        1.0 if user_item.get((u, it), 0) > 1 or user_n.get(u, 0) > 1 else 0.0
        for u, it in zip(df["user_id"], df["parent_asin"])]
    out["criterion_type"] = "repeat_product"
    return finalise(out, "amazon", n_bench, n_llm_core, seed,
                    detect_language=detect_language)


# ---------------------------------------------------------------------------
# Yelp Open Dataset (line-delimited json, ~5 GB)
# ---------------------------------------------------------------------------
def build_yelp(review_json, n_pool=30000, n_bench=1200, n_llm_core=400,
               seed=20260909, chunk=200_000, detect_language=True,
               progress=None) -> pd.DataFrame:
    """Two streaming passes over the review file.

    Pass 1 counts, for every (user, business) pair, how many reviews exist, using
    64-bit hashes held in numpy arrays (memory stays ~100 MB for 7M reviews).
    Pass 2 draws a uniform reservoir sample of ``n_pool`` reviews.
    """
    def _h64(s):
        return np.uint64(int(hashlib.blake2b(s.encode(), digest_size=8).hexdigest(), 16))

    pair_hashes, user_hashes = [], []
    buf_p, buf_u = [], []
    n_seen = 0
    with open(review_json, "r", encoding="utf8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            buf_p.append(_h64(r["user_id"] + "|" + r["business_id"]))
            buf_u.append(_h64(r["user_id"]))
            n_seen += 1
            if len(buf_p) >= chunk:
                pair_hashes.append(np.array(buf_p, dtype=np.uint64)); buf_p = []
                user_hashes.append(np.array(buf_u, dtype=np.uint64)); buf_u = []
                if progress:
                    progress(f"pass1 {n_seen:,} reviews")
    if buf_p:
        pair_hashes.append(np.array(buf_p, dtype=np.uint64))
        user_hashes.append(np.array(buf_u, dtype=np.uint64))
    P = np.concatenate(pair_hashes); U = np.concatenate(user_hashes)
    pu, pc = np.unique(P, return_counts=True)
    uu, uc = np.unique(U, return_counts=True)
    rep_pairs = set(pu[pc > 1].tolist())
    rep_users = set(uu[uc > 1].tolist())
    del P, U, pair_hashes, user_hashes

    rng = np.random.default_rng(seed)
    keep, i = [], 0
    with open(review_json, "r", encoding="utf8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            i += 1
            if len(keep) < n_pool:
                keep.append(r)
            else:
                j = rng.integers(0, i)
                if j < n_pool:
                    keep[j] = r
            if progress and i % 500_000 == 0:
                progress(f"pass2 {i:,} reviews")

    df = pd.DataFrame(keep)
    out = pd.DataFrame({
        "text": df["text"],
        "closed_form_raw": pd.to_numeric(df["stars"], errors="coerce"),
        "closed_form_type": "star_rating",
        "author_id": df["user_id"].map(h),
        "item_id": df["business_id"].map(h),
        "ts": df["date"],
    })
    out["closed_form"] = stars_to_pm1(out["closed_form_raw"])
    out["criterion_binary"] = [
        1.0 if (_h64(u + "|" + b) in rep_pairs or _h64(u) in rep_users) else 0.0
        for u, b in zip(df["user_id"], df["business_id"])]
    out["criterion_type"] = "repeat_business"
    return finalise(out, "yelp", n_bench, n_llm_core, seed,
                    detect_language=detect_language)


# ---------------------------------------------------------------------------
# Regulations.gov docket comments
# ---------------------------------------------------------------------------
def _explode_regulation_comments(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in d.iterrows():
        cm = r["comments"]
        if cm is None:
            continue
        if isinstance(cm, list):
            for item in cm:
                if not isinstance(item, dict):
                    continue
                rows.append(dict(
                    text=item.get("text", ""),
                    author_id=h(str(item.get("commenter_fname", ""))
                                + str(item.get("commenter_lname", ""))),
                    item_id=h(r.get("id")),
                    ts=str(item.get("comment_date", "")),
                    agency=r.get("agency"), docket_title=r.get("title"),
                    purpose=r.get("purpose")))
            continue
        if not isinstance(cm, dict) or "text" not in cm:
            continue
        n = len(cm["text"])
        for k in range(n):
            rows.append(dict(
                text=cm["text"][k],
                author_id=h(str(cm["commenter_fname"][k]) + str(cm["commenter_lname"][k])),
                item_id=h(r["id"]),
                ts=str(cm["comment_date"][k]),
                agency=r["agency"], docket_title=r["title"], purpose=r["purpose"]))
    return pd.DataFrame(rows)


def build_regulations(path, n_bench=1200, n_llm_core=400, seed=20260909,
                      detect_language=True) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".json":
        d = pd.DataFrame(json.loads(path.read_text(encoding="utf8")))
    else:
        d = pd.read_parquet(path)
    out = _explode_regulation_comments(d)
    out["closed_form_raw"] = np.nan
    out["closed_form"] = np.nan
    out["closed_form_type"] = "none"
    out["criterion_binary"] = np.nan
    out["criterion_type"] = "none"
    meta = out[["agency", "docket_title", "purpose"]].copy()
    res = finalise(out.drop(columns=["agency", "docket_title", "purpose"]),
                   "regulations", n_bench, n_llm_core, seed, stratify=None,
                   detect_language=detect_language)
    res = res.join(meta.reindex(range(len(res))).reset_index(drop=True))
    return res


# ---------------------------------------------------------------------------
# Bluesky posts (headerless csv: text, flag)
# ---------------------------------------------------------------------------
def build_bluesky(csv_path, n_pool=30000, n_bench=1200, n_llm_core=400,
                  seed=20260909, detect_language=True,
                  n_scan=1_500_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    keep, i = [], 0
    with open(csv_path, "r", encoding="utf8", errors="ignore", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            i += 1
            rec = (row[0], row[1] if len(row) > 1 else "")
            if len(keep) < n_pool:
                keep.append(rec)
            else:
                j = rng.integers(0, i)
                if j < n_pool:
                    keep[j] = rec
            if n_scan and i >= n_scan:
                break
    df = pd.DataFrame(keep, columns=["text", "flag"])
    out = pd.DataFrame({
        "text": df["text"],
        "closed_form_raw": np.nan, "closed_form": np.nan,
        "closed_form_type": "none",
        "criterion_binary": np.nan, "criterion_type": "none",
        "author_id": np.nan, "item_id": np.nan, "ts": np.nan,
    })
    res = finalise(out, "bluesky", n_bench, n_llm_core, seed, stratify=None,
                   detect_language=detect_language)
    res["flag"] = df["flag"].reindex(range(len(res))).values
    return res


# ---------------------------------------------------------------------------
# OULAD -- criterion-only corpus (the public release carries no forum text)
# ---------------------------------------------------------------------------
def build_oulad(folder, vle_chunk=2_000_000) -> pd.DataFrame:
    """Student-level panel used to anchor the error-propagation study in a real
    prediction task.  Protected attributes (gender, disability, region, IMD band,
    age band) are dropped at ingestion and never modelled."""
    folder = Path(folder)
    si = pd.read_csv(folder / "studentInfo.csv")
    sa = pd.read_csv(folder / "studentAssessment.csv")
    asm = pd.read_csv(folder / "assessments.csv")

    keep = ["code_module", "code_presentation", "id_student",
            "num_of_prev_attempts", "studied_credits", "final_result"]
    si = si[keep].copy()

    sa = sa.merge(asm[["id_assessment", "assessment_type", "weight", "date"]],
                  on="id_assessment", how="left")
    # pandas 3 + quoted CSV -> pyarrow string dtypes; mean() then raises.
    for col in ("score", "date_submitted", "date", "weight"):
        if col in sa.columns:
            sa[col] = pd.to_numeric(sa[col], errors="coerce")
    agg = (sa.groupby("id_student")
             .agg(mean_score=("score", "mean"), n_assess=("score", "size"),
                  mean_delay=("date_submitted", "mean"))
             .reset_index())

    clicks = None
    vle_path = folder / "studentVle.csv"
    if vle_path.exists():
        parts = []
        for ch in pd.read_csv(vle_path, chunksize=vle_chunk,
                              usecols=["id_student", "sum_click", "date"]):
            ch["sum_click"] = pd.to_numeric(ch["sum_click"], errors="coerce")
            ch["date"] = pd.to_numeric(ch["date"], errors="coerce")
            parts.append(ch.groupby("id_student")
                           .agg(clicks=("sum_click", "sum"), days=("date", "nunique")))
        clicks = (pd.concat(parts).groupby(level=0).sum().reset_index())

    df = si.merge(agg, on="id_student", how="left")
    if clicks is not None:
        df = df.merge(clicks, on="id_student", how="left")
    df["pass"] = df["final_result"].isin(["Pass", "Distinction"]).astype(int)
    return df
