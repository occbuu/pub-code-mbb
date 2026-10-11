"""Measurement arm 1 -- expert dictionary scorer.

Implements a negation- and intensifier-aware lexicon scorer over the codebook in
:mod:`cft.codebook`.  The scorer returns, for every document and construct:

``<C>_sal``  salience  = matched construct weight / (matched weight + length term)
``<C>_pol``  polarity  = signed weighted mean of matches, squashed to [-1, 1]
``<C>_n``    raw number of matched patterns

Design notes (reported in the manuscript):

* Matching is done on a normalised token stream, so multi-word patterns are
  matched as contiguous n-grams; the longest pattern wins at a given position
  and consumed positions are not reused (greedy, non-overlapping).
* A negator anywhere in the three tokens preceding a match flips its sign;
  a second negator in the window restores it (double negation).
* Intensifiers/downtoners in the two preceding tokens scale the weight.
* Salience is normalised by ``sqrt`` of document length so that long documents
  are not mechanically more "salient"; the exact normaliser is a tuning knob
  exposed for the robustness analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .codebook import (CONSTRUCTS, CONTRASTS, DOWNTONERS, INTENSIFIERS,
                       LEXICON, NEGATORS, PRIORITY_PATTERNS, as_weighted)

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text).lower())


def _build_index(lexicon):
    """term-tuple -> (construct, sign, weight); plus max n-gram length."""
    index = {}
    maxn = 1
    for c in CONSTRUCTS:
        for pol, sign in (("pos", +1.0), ("neg", -1.0)):
            for term, w in as_weighted(lexicon[c][pol]):
                toks = tuple(tokenize(term))
                if not toks:
                    continue
                maxn = max(maxn, len(toks))
                # first writer wins; longer patterns are preferred at match time
                index.setdefault(toks, (c, sign, w))
    return index, maxn


@dataclass
class DictionaryScorer:
    """Expert-lexicon construct scorer."""

    lexicon: dict = field(default_factory=lambda: LEXICON)
    negation_window: int = 3
    modifier_window: int = 2
    length_norm: str = "sqrt"          # {"sqrt", "linear", "none"}
    salience_k: float = 6.0            # saturation constant for salience
    use_priority_patterns: bool = True
    contrast_discount: float = 0.85    # weight on clauses after "but"

    def __post_init__(self):
        self.index, self.maxn = _build_index(self.lexicon)
        self._priority = [(re.compile(p), c, s) for p, c, s in PRIORITY_PATTERNS]

    # -- core -------------------------------------------------------------
    def score_document(self, text: str) -> dict:
        toks = tokenize(text)
        n = len(toks)
        hits = {c: [] for c in CONSTRUCTS}

        i = 0
        clause_w = 1.0
        while i < n:
            if toks[i] in CONTRASTS:
                clause_w = self.contrast_discount
            matched = False
            for L in range(min(self.maxn, n - i), 0, -1):
                key = tuple(toks[i:i + L])
                ent = self.index.get(key)
                if ent is None:
                    continue
                c, sign, w = ent
                w = w * clause_w

                # negation window
                lo = max(0, i - self.negation_window)
                negs = sum(1 for t in toks[lo:i] if t in NEGATORS)
                if negs % 2 == 1:
                    sign = -sign
                    w *= 0.9   # negated statements are slightly less extreme

                # modifiers
                mlo = max(0, i - self.modifier_window)
                for t in toks[mlo:i]:
                    if t in INTENSIFIERS:
                        w *= INTENSIFIERS[t]
                    elif t in DOWNTONERS:
                        w *= DOWNTONERS[t]

                hits[c].append(sign * w)
                i += L
                matched = True
                break
            if not matched:
                i += 1

        # priority (directional) patterns operate on the raw string
        if self.use_priority_patterns:
            low = str(text).lower()
            for rx, c, s in self._priority:
                k = len(rx.findall(low))
                if k:
                    hits[c].extend([s] * k)

        out = {}
        for c in CONSTRUCTS:
            h = hits[c]
            k = len(h)
            mass = float(np.sum(np.abs(h)))
            if self.length_norm == "sqrt":
                denom = mass + self.salience_k * np.sqrt(max(n, 1))
            elif self.length_norm == "linear":
                denom = mass + self.salience_k * max(n, 1) / 10.0
            else:
                denom = mass + self.salience_k
            out[f"{c}_sal"] = float(mass / denom) if denom > 0 else 0.0
            if k == 0:
                out[f"{c}_pol"] = np.nan
            else:
                m = float(np.sum(h) / max(mass, 1e-9))
                out[f"{c}_pol"] = float(np.clip(m, -1.0, 1.0))
            out[f"{c}_n"] = k
        return out

    def score_frame(self, texts, ids=None, n_jobs=None) -> pd.DataFrame:
        texts = list(texts)
        n = len(texts)
        if n_jobs is None:
            from .resources import cpu_workers
            n_jobs = cpu_workers() if n >= 200 else 1
        n_jobs = max(1, int(n_jobs))
        if n_jobs == 1 or n < 200:
            rows = [self.score_document(t) for t in texts]
        else:
            from concurrent.futures import ProcessPoolExecutor
            from .resources import worker_init_single_thread
            chunk = max(8, n // (n_jobs * 8))
            with ProcessPoolExecutor(max_workers=n_jobs,
                                     initializer=worker_init_single_thread) as ex:
                rows = list(ex.map(self.score_document, texts, chunksize=chunk))
        df = pd.DataFrame(rows)
        if ids is not None:
            df.insert(0, "doc_id", list(ids))
        return df


def coverage_table(scores: pd.DataFrame) -> pd.DataFrame:
    """Share of documents in which each construct fires at all."""
    rows = []
    for c in CONSTRUCTS:
        n = scores[f"{c}_n"]
        rows.append(dict(construct=c,
                         pct_docs_fired=100.0 * float((n > 0).mean()),
                         mean_hits=float(n.mean()),
                         median_hits=float(n.median())))
    return pd.DataFrame(rows)
