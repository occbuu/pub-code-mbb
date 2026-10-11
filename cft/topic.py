"""Measurement arm 2 -- topic model with anchor-based construct mapping.

A topic model is an *unsupervised* measurement device: it returns a document x
topic matrix, and the analyst has to decide which topics "are" which construct.
That mapping step is where most of the discretion (and most of the measurement
error) lives, so it is made explicit and reproducible here.

Procedure
---------
1. Fit NMF (default) or LDA on a TF-IDF / count matrix of the corpus.
2. For every topic ``k`` and construct ``c`` compute the mass of the topic's
   term distribution that falls on the construct's positive and negative anchor
   terms (the same expert lexicon used by arm 1, so that the two arms aim at
   the same target -- a shared-anchor design, not a shared-score design).
3. ``load[k, c]  = (pos + neg) / total_anchor_mass``  -> how much topic k is
   about construct c;  ``val[k, c] = (pos - neg) / (pos + neg)``  -> the valence
   the topic carries for that construct.
4. Document scores:
   ``sal_dc = sum_k theta_dk * load_kc``,
   ``pol_dc = sum_k theta_dk * load_kc * val_kc / sal_dc``.

Only topics whose loading exceeds ``min_load`` are assigned, so that "no topic
about this construct" produces a missing value rather than noise -- the topic
analogue of a dictionary miss.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import (ENGLISH_STOP_WORDS,
                                             CountVectorizer, TfidfVectorizer)

from .codebook import CONSTRUCTS, LEXICON, as_weighted

# Words a generic stop list removes but that carry construct meaning: negation,
# future reference and repetition are exactly what BI ("I will not come back
# again") is made of.  Removing them would hand the topic arm a handicap the
# comparison is not about, so they are kept.
KEEP_WORDS = {
    "no", "not", "nor", "never", "none", "nothing", "cannot", "cant",
    "will", "would", "wont", "should", "must", "again", "back", "very",
    "too", "only", "still", "always", "ever", "everyone", "everything",
    "someone", "others", "another", "next", "before", "after", "off",
}
STOP_WORDS = sorted(ENGLISH_STOP_WORDS - KEEP_WORDS)


@dataclass
class TopicScorer:
    n_topics: int = 30
    method: str = "nmf"          # {"nmf", "lda"}
    max_features: int = 60000
    min_df: int = 5
    max_df: float = 0.5
    ngram_range: tuple = (1, 3)
    min_load: float = 0.05
    random_state: int = 20260909
    n_jobs: int = 1          # LDA only; NMF uses the process-level BLAS pool
    vectorizer: object = field(default=None, repr=False)
    model: object = field(default=None, repr=False)

    # -- fitting -----------------------------------------------------------
    def fit(self, texts):
        if self.method == "nmf":
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features, min_df=self.min_df,
                max_df=self.max_df, ngram_range=self.ngram_range,
                stop_words=STOP_WORDS, sublinear_tf=True)
            X = self.vectorizer.fit_transform(texts)
            self.model = NMF(n_components=self.n_topics, init="nndsvda",
                             random_state=self.random_state, max_iter=400,
                             beta_loss="frobenius")
        else:
            self.vectorizer = CountVectorizer(
                max_features=self.max_features, min_df=self.min_df,
                max_df=self.max_df, ngram_range=self.ngram_range,
                stop_words="english")
            X = self.vectorizer.fit_transform(texts)
            nj = self.n_jobs
            if nj is None or nj < 1:
                from .resources import cpu_workers
                nj = cpu_workers()
            self.model = LatentDirichletAllocation(
                n_components=self.n_topics, random_state=self.random_state,
                learning_method="online", max_iter=25, n_jobs=nj)
        W = self.model.fit_transform(X)
        self.components_ = self.model.components_
        self.vocab_ = np.array(self.vectorizer.get_feature_names_out())
        self._build_map()
        return self

    # -- anchor mapping ----------------------------------------------------
    def _anchor_features(self, term: str):
        """Vocabulary features an anchor phrase actually produces.

        The anchor is passed through the *vectorizer's own analyzer*, so the
        stop-list and n-gram range that built the topic space are the same ones
        that decide what a construct term looks like in that space.  The longest
        surviving n-gram is used, which keeps 'would not recommend' from being
        matched by the bare unigram 'recommend'.
        """
        if not hasattr(self, "_analyzer"):
            self._analyzer = self.vectorizer.build_analyzer()
            self._vocab_pos = {t: i for i, t in enumerate(self.vocab_)}
        feats = [f for f in self._analyzer(term) if f in self._vocab_pos]
        if not feats:
            return []
        longest = max(len(f.split()) for f in feats)
        return [self._vocab_pos[f] for f in feats if len(f.split()) == longest]

    def _build_map(self):
        H = self.components_
        H = H / np.maximum(H.sum(axis=1, keepdims=True), 1e-12)

        K, C = H.shape[0], len(CONSTRUCTS)
        pos_mass = np.zeros((K, C))
        neg_mass = np.zeros((K, C))
        self.anchor_hits_ = {}

        for ci, c in enumerate(CONSTRUCTS):
            for pol, arr in (("pos", pos_mass), ("neg", neg_mass)):
                found = 0
                for term, w in as_weighted(LEXICON[c][pol]):
                    idx = self._anchor_features(term)
                    if idx:
                        found += 1
                        arr[:, ci] += w * H[:, idx].sum(axis=1) / len(idx)
                self.anchor_hits_[(c, pol)] = found

        total = (pos_mass + neg_mass)
        anchor_total = np.maximum(total.sum(axis=1, keepdims=True), 1e-12)
        self.load_ = total / anchor_total                     # K x C, rows ~ sum 1
        # scale loading by how much anchor mass the topic carries at all
        topic_anchor_share = total.sum(axis=1) / np.maximum(H.sum(axis=1), 1e-12)
        self.load_ = self.load_ * topic_anchor_share[:, None] * 10.0
        self.load_ = np.clip(self.load_, 0, 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            self.val_ = np.where(total > 0, (pos_mass - neg_mass) / np.maximum(total, 1e-12), 0.0)
        self.load_[self.load_ < self.min_load] = 0.0
        return self

    def top_terms(self, k: int, n: int = 12):
        idx = np.argsort(self.components_[k])[::-1][:n]
        return list(self.vocab_[idx])

    def topic_table(self) -> pd.DataFrame:
        rows = []
        for k in range(self.components_.shape[0]):
            r = dict(topic=k, top_terms=", ".join(self.top_terms(k, 10)))
            for ci, c in enumerate(CONSTRUCTS):
                r[f"load_{c}"] = round(float(self.load_[k, ci]), 3)
                r[f"val_{c}"] = round(float(self.val_[k, ci]), 3)
            best = int(np.argmax(self.load_[k]))
            r["assigned"] = CONSTRUCTS[best] if self.load_[k, best] > 0 else "-"
            rows.append(r)
        return pd.DataFrame(rows)

    # -- scoring -----------------------------------------------------------
    def transform(self, texts, ids=None) -> pd.DataFrame:
        X = self.vectorizer.transform(texts)
        W = self.model.transform(X)
        W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)

        sal = W @ self.load_                                   # N x C
        num = W @ (self.load_ * self.val_)
        with np.errstate(invalid="ignore", divide="ignore"):
            pol = np.where(sal > 1e-9, num / np.maximum(sal, 1e-12), np.nan)
        pol = np.clip(pol, -1, 1)

        out = {}
        for ci, c in enumerate(CONSTRUCTS):
            s = sal[:, ci]
            p = pol[:, ci].copy()
            p[s < 1e-4] = np.nan
            out[f"{c}_sal"] = s
            out[f"{c}_pol"] = p
        df = pd.DataFrame(out)
        if ids is not None:
            df.insert(0, "doc_id", list(ids))
        return df

    def fit_transform(self, texts, ids=None) -> pd.DataFrame:
        self.fit(texts)
        return self.transform(texts, ids)
