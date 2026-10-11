"""Elicitation conditions for the construct-collapse mitigation study (Paper B).

The benchmark showed that an instruction-tuned model asked for five construct
ratings returns five numbers that are largely one number: the first principal
component absorbs ~85-95% of their common variance.  This module defines the
*elicitation-level* interventions that might undo that, so the diagnosis can be
followed by an attempted cure.

Conditions
----------
``joint``     the benchmark's prompt: five constructs, one JSON object, one call.
              Baseline.
``isolated``  one construct per call; the other four are never mentioned, so the
              model cannot anchor one rating on another inside a single
              generation.
``evidence``  one construct per call *and* the model must return a verbatim span
              from the document that supports the rating.  No span -> the
              construct is absent.  Targets the hypothesis that the halo comes
              from filling in missing evidence with the document's overall
              valence.
``gate``      two calls per construct: first a yes/no presence judgement with no
              scale in sight, then -- only if present -- the level.  Separates
              the abstention decision from the rating decision.

Every condition is scored with the same codebook text, so what changes between
conditions is the *shape of the question*, never the construct definition.
"""

from __future__ import annotations

import json
import re

import numpy as np

from .codebook import CONSTRUCT_META, CONSTRUCTS

CONDITIONS = ["joint", "isolated", "evidence", "gate"]

SYSTEM = (
    "You are a trained research assistant coding text for latent behavioural "
    "constructs. You follow the codebook literally and you never speculate about "
    "information the writer did not express."
)

_SCALE = (
    '  "sal" = 0, 1, 2 or 3: how much the document is ABOUT this construct '
    "(0 = not discussed at all, 3 = a central concern of the document).\n"
    '  "lvl" = an integer from -3 to +3: the LEVEL of the construct expressed '
    "(-3 = strongly low/negative, 0 = neutral or mixed, +3 = strongly high/"
    'positive). If "sal" is 0, set "lvl" to null.\n'
)


def _one_definition(c: str) -> str:
    m = CONSTRUCT_META[c]
    return (f"{c} ({m['short']}): {m['definition']} Coding rule: {m['coding_rule']}")


# ---------------------------------------------------------------------------
# prompt builders
# ---------------------------------------------------------------------------
def build_isolated(text: str, construct: str) -> str:
    return (
        f"Code the DOCUMENT below for ONE construct.\n\nCONSTRUCT\n"
        f"{_one_definition(construct)}\n\nGive two numbers:\n{_SCALE}\n"
        'Answer with a single JSON object and nothing else, exactly: '
        '{"sal": <0-3>, "lvl": <-3..3 or null>}\n\n'
        f'DOCUMENT:\n"""{text}"""\n\nJSON:')


def build_evidence(text: str, construct: str) -> str:
    return (
        f"Code the DOCUMENT below for ONE construct, and justify the rating with "
        f"the writer's own words.\n\nCONSTRUCT\n{_one_definition(construct)}\n\n"
        "Procedure:\n"
        "1. Find a span of the document, copied VERBATIM, that expresses this "
        "construct. Do not paraphrase and do not use a span that only expresses "
        "the writer's general opinion of the object.\n"
        "2. If no such span exists, the construct is ABSENT: set \"quote\" to "
        "null, \"sal\" to 0 and \"lvl\" to null. An absent construct is a normal "
        "and correct answer.\n"
        f"3. Otherwise give the two numbers:\n{_SCALE}\n"
        'Answer with a single JSON object and nothing else, exactly: '
        '{"quote": "<verbatim span or null>", "sal": <0-3>, "lvl": <-3..3 or null>}\n\n'
        f'DOCUMENT:\n"""{text}"""\n\nJSON:')


def build_gate_presence(text: str, construct: str) -> str:
    return (
        f"Does the DOCUMENT below address the following construct at all?\n\n"
        f"CONSTRUCT\n{_one_definition(construct)}\n\n"
        "Answer only whether the construct is discussed. Do not rate how "
        "positive or negative it is; that is a different question you will not "
        "be asked here.\n\n"
        'Answer with a single JSON object and nothing else, exactly: '
        '{"present": true or false}\n\n'
        f'DOCUMENT:\n"""{text}"""\n\nJSON:')


def build_gate_level(text: str, construct: str) -> str:
    m = CONSTRUCT_META[construct]
    return (
        f"The DOCUMENT below does address this construct. Rate the LEVEL "
        f"expressed.\n\nCONSTRUCT\n{_one_definition(construct)}\n\n"
        f"-3 = {m['anchor_low']}; 0 = neutral or mixed; +3 = {m['anchor_high']}\n\n"
        'Answer with a single JSON object and nothing else, exactly: '
        '{"lvl": <-3..3>}\n\n'
        f'DOCUMENT:\n"""{text}"""\n\nJSON:')


BUILDERS = {
    "isolated": build_isolated,
    "evidence": build_evidence,
    "gate_presence": build_gate_presence,
    "gate_level": build_gate_level,
}


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
_JSON_RE = re.compile(r"\{[^{}]*\}", re.S)


def _last_json(txt: str):
    for m in reversed(_JSON_RE.findall(txt or "")):
        try:
            obj = json.loads(m)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_single(txt: str) -> dict:
    """Parse one construct's answer into sal/pol (+ quote where present)."""
    obj = _last_json(txt) or {}
    out = dict(parsed=bool(obj), quote=obj.get("quote"))
    try:
        sal = float(obj.get("sal", 0) or 0)
    except Exception:
        sal = 0.0
    lvl = obj.get("lvl", None)
    try:
        lvl = float(lvl) if lvl is not None else np.nan
    except Exception:
        lvl = np.nan
    out["sal"] = float(np.clip(sal / 3.0, 0, 1))
    out["pol"] = (np.nan if (sal <= 0 or not np.isfinite(lvl))
                  else float(np.clip(lvl / 3.0, -1, 1)))
    return out


def parse_presence(txt: str) -> bool | None:
    obj = _last_json(txt)
    if not obj or "present" not in obj:
        return None
    v = obj["present"]
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "yes", "1")


def parse_level(txt: str) -> float:
    obj = _last_json(txt) or {}
    try:
        lvl = float(obj.get("lvl"))
    except Exception:
        return np.nan
    return float(np.clip(lvl / 3.0, -1, 1))


def quote_is_verbatim(quote, text) -> bool:
    """Did the model actually copy from the document?  A quote that is not in the
    text is a fabricated justification, and its rate is itself a result."""
    if not quote or not isinstance(quote, str):
        return False
    q = " ".join(quote.split()).strip(' "\'').lower()
    t = " ".join(str(text).split()).lower()
    return len(q) >= 8 and q in t


# ---------------------------------------------------------------------------
# post-hoc statistical mitigation (no GPU needed)
# ---------------------------------------------------------------------------
def residualise_general_factor(wide, constructs=None):
    """Remove the first principal component from a doc x construct score matrix.

    This is the analyst-side counterpart of the elicitation interventions: if the
    general evaluative factor is simply partialled out, does what remains still
    discriminate, and does it still predict anything?
    """
    constructs = constructs or CONSTRUCTS
    cols = [f"{c}_pol" for c in constructs if f"{c}_pol" in wide.columns]
    X = wide[cols]
    mu, sd = X.mean(), X.std(ddof=1).replace(0, np.nan)
    Z = ((X - mu) / sd)
    Zf = Z.fillna(0.0).values
    R = np.corrcoef(Zf, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)
    np.fill_diagonal(R, 1.0)
    w, v = np.linalg.eigh(R)
    pc = v[:, -1]
    scores = Zf @ pc
    denom = float(pc @ pc) or 1.0
    resid = Zf - np.outer(scores / denom, pc)
    out = wide.copy()
    for j, c in enumerate(cols):
        out[c] = np.where(X[c].notna().values, resid[:, j], np.nan)
    out["_general_factor"] = scores
    return out
