"""Construct codebook for the multi-corpus measurement benchmark (Paper 13).

Five latent constructs drawn from the TPB / UTAUT family, defined so that they can
be (a) rated by a human coder, (b) prompted to an LLM, (c) matched by an expert
lexicon and (d) mapped onto topics.  Every downstream measurement arm consumes
this single file, so the four "measurement machines" are aimed at *identical*
construct definitions -- which is the precondition for a multitrait-multimethod
(MTMM) comparison in the sense of Campbell & Fiske (1959).

Each construct is scored on two dimensions:

* ``salience``  -- how much of the document is *about* the construct at all
                   (0 = the construct is not discussed, 1 = the document is
                   entirely about it).  This is the "was the trait even
                   measurable here?" dimension.
* ``polarity``  -- the *level* of the construct expressed, on [-1, +1], which is
                   the analogue of a Likert response and the quantity entered in
                   the MTMM matrix.

The lexicon below is an *expert* dictionary: terms were written from the item
wordings of the canonical TPB (Ajzen 1991) and UTAUT (Venkatesh et al. 2003)
scales and then adapted to consumer/citizen prose.  Terms are never added on the
basis of their empirical association with the criterion, so the dictionary arm
stays independent of the validation evidence.
"""

from __future__ import annotations

CONSTRUCTS = ["ATT", "PE", "EE", "SI", "BI"]

CONSTRUCT_META = {
    "ATT": dict(
        name="Attitude toward the object",
        short="Attitude",
        theory="TPB (Ajzen, 1991); attitude toward the behaviour/object",
        definition=(
            "The writer's overall evaluative judgement of the object (the venue, "
            "product, policy or actor) -- good/bad, pleasant/unpleasant, "
            "favourable/unfavourable -- irrespective of any specific benefit."),
        anchor_low="Strongly unfavourable overall evaluation",
        anchor_high="Strongly favourable overall evaluation",
        coding_rule=(
            "Code global evaluative language here. If the evaluation is tied to a "
            "specific functional outcome ('it removed the stain'), code PE instead; "
            "if it is tied to difficulty of use, code EE."),
    ),
    "PE": dict(
        name="Performance expectancy / perceived usefulness",
        short="Performance expectancy",
        theory="UTAUT (Venkatesh et al., 2003); TAM perceived usefulness",
        definition=(
            "The degree to which the writer believes the object performs its "
            "intended function and delivers the benefit it promises "
            "(it works, it is effective, it solves the problem)."),
        anchor_low="The object fails at its intended function",
        anchor_high="The object performs its intended function very well",
        coding_rule=(
            "Requires reference to a function, task or outcome. Pure affect "
            "('I love it') is ATT, not PE."),
    ),
    "EE": dict(
        name="Effort expectancy / perceived ease of use",
        short="Effort expectancy",
        theory="UTAUT effort expectancy; TPB perceived behavioural control",
        definition=(
            "The degree of ease associated with obtaining, understanding, "
            "installing, operating or otherwise dealing with the object "
            "(clarity, simplicity, convenience, absence of hassle)."),
        anchor_low="Dealing with the object is difficult, confusing or effortful",
        anchor_high="Dealing with the object is easy, clear and effortless",
        coding_rule=(
            "Waiting time, queueing, unclear instructions, complicated setup and "
            "bureaucratic burden all count as low EE."),
    ),
    "SI": dict(
        name="Social influence / subjective norm",
        short="Social influence",
        theory="TPB subjective norm; UTAUT social influence",
        definition=(
            "Reference to what *other people* think, say, do or expect regarding "
            "the object: recommendations received, reputation, hype, reviews read, "
            "the behaviour of friends/family/colleagues, or normative pressure."),
        anchor_low="Others discourage, warn against, or think badly of the object",
        anchor_high="Others encourage, praise or expect engagement with the object",
        coding_rule=(
            "The referent must be a third party. The writer's own advice to the "
            "reader ('I recommend it') is BI (intention to recommend), not SI."),
    ),
    "BI": dict(
        name="Behavioural intention",
        short="Behavioural intention",
        theory="TPB / UTAUT behavioural intention",
        definition=(
            "The writer's stated intention or commitment regarding future "
            "behaviour toward the object: returning, repurchasing, continuing, "
            "recommending to others, or refusing to do any of these."),
        anchor_low="Explicit intention not to return / not to use / to warn others off",
        anchor_high="Explicit intention to return, repurchase, continue or recommend",
        coding_rule=(
            "Must be forward-looking or dispositional. A past visit is not BI; "
            "'I'll be back' and 'I would recommend it' are."),
    ),
}

# ---------------------------------------------------------------------------
# Expert lexicon.
# Each construct has POS and NEG term lists.  Entries are lower-cased surface
# patterns; a leading "re:" marks a regular expression, everything else is
# matched as a whitespace-delimited n-gram.  Weights default to 1.0; a term may
# be given as (term, weight).
# ---------------------------------------------------------------------------

LEXICON = {
    "ATT": {
        "pos": [
            "love", "loved", "great", ("excellent", 1.3), ("amazing", 1.3), "awesome",
            "wonderful", ("fantastic", 1.3), ("outstanding", 1.3), "nice", "good",
            "lovely", "delightful", "enjoyed", "enjoy", "pleasant", "pleased",
            "happy with", "satisfied", "impressed", "impressive", "beautiful",
            "charming", "gem", "favorite", "favourite", ("perfect", 1.3), "best",
            "solid", "worth it", "worth every", "quality", "top notch", "superb",
            ("brilliant", 1.2), "fabulous", "terrific", "delicious", "tasty",
            "friendly", "welcoming", "cozy", "clean", "well made", "well-made",
        ],
        "neg": [
            ("terrible", 1.3), ("awful", 1.3), ("horrible", 1.3), "bad", "poor",
            "disappointing", "disappointed", "disappointment", "mediocre", "meh",
            "unpleasant", "nasty", "gross", "disgusting", "rude", "dirty", "filthy",
            "overpriced", "rip off", "ripoff", "waste of money", "waste of time",
            "regret", "hate", "hated", "dislike", "unhappy", "upset", "angry",
            "frustrating", "frustrated", "annoying", "cheap feeling", "flimsy",
            "junk", "garbage", "trash", "worst", "avoid", "sucks", "sucked",
            "not worth", "unacceptable", "appalling", "atrocious",
        ],
    },
    "PE": {
        "pos": [
            "works", "worked", "works well", "works great", "works perfectly",
            "does the job", "did the job", "does what", "effective", "efficient",
            "useful", "helpful", "helps", "helped", "reliable", "durable",
            "long lasting", "long-lasting", "sturdy", "powerful", "accurate",
            "consistent", "performs", "performance", "solved", "solves",
            "fixed my", "as advertised", "as described", "as expected",
            "lives up to", "gets the job done", "no issues", "no problems",
            "results", "improvement", "improved", "worth the",
        ],
        "neg": [
            "doesn't work", "does not work", "didn't work", "did not work",
            "stopped working", "broke", "broken", "defective", "faulty", "useless",
            "ineffective", "unreliable", "failed", "fails", "failure", "malfunction",
            "not as described", "not as advertised", "false advertising",
            "fell apart", "falls apart", "leaks", "leaked", "error", "errors",
            "bug", "bugs", "glitch", "crashes", "crashed", "no effect",
            "did nothing", "does nothing", "waste", "poor quality", "low quality",
        ],
    },
    "EE": {
        "pos": [
            "easy", "easy to use", "easy to install", "simple", "straightforward",
            "intuitive", "user friendly", "user-friendly", "convenient",
            "hassle free", "hassle-free", "effortless", "quick", "fast", "smooth",
            "seamless", "clear instructions", "well organized", "well-organized",
            "no hassle", "painless", "a breeze", "plug and play", "self explanatory",
            "self-explanatory", "prompt", "efficient service", "quick service",
            "no wait", "walk in", "accessible",
        ],
        "neg": [
            "difficult", "hard to", "complicated", "confusing", "confused",
            "complex", "cumbersome", "tedious", "hassle", "struggle", "struggled",
            "took forever", "slow", "waited", "waiting", "long wait", "long line",
            "long queue", "unclear", "not clear", "counterintuitive",
            "poorly written", "instructions were", "learning curve", "inconvenient",
            "impossible to", "couldn't figure", "could not figure", "gave up",
            "paperwork", "bureaucratic", "red tape", "runaround", "run around",
        ],
    },
    "SI": {
        "pos": [
            "recommended by", "recommended to me", "friend recommended",
            "my friend", "my friends", "my wife", "my husband", "my family",
            "my colleague", "coworker", "everyone says", "everyone said",
            "everybody loves", "people say", "people said", "word of mouth",
            "heard about", "heard great things", "reviews said", "the reviews",
            "highly rated", "popular", "hype", "well known", "reputation",
            "famous", "raved", "raving", "told me about", "was told",
            "brought me here", "came with", "based on reviews",
        ],
        "neg": [
            "warned me", "warned us", "friend warned", "bad reviews",
            "negative reviews", "people complain", "everyone complains",
            "don't believe the hype", "overhyped", "over hyped", "overrated",
            "not what people say", "poor reputation", "bad reputation",
            "told me not to", "advised against",
        ],
    },
    "BI": {
        "pos": [
            "will be back", "will come back", "coming back", "be back",
            "will return", "would return", "go back", "again and again",
            "will buy again", "would buy again", "buy it again", "purchase again",
            "will order again", "order again", "reorder", "repurchase",
            "will definitely", "definitely recommend", "highly recommend",
            "would recommend", "i recommend", "recommend it", "recommend this",
            "must try", "must visit", "check it out", "give it a try",
            "will keep", "keep using", "will continue", "continue to use",
            "new favorite", "regular", "loyal", "subscribe", "sign up",
            "support this", "in favor of", "urge you to adopt",
        ],
        "neg": [
            "never again", "won't be back", "will not be back", "not coming back",
            "never coming back", "won't return", "will not return",
            "won't buy again", "would not buy", "never buy", "not recommend",
            "would not recommend", "do not recommend", "don't recommend",
            "wouldn't recommend", "cannot recommend", "steer clear", "stay away",
            "returning it", "returned it", "sent it back", "refund", "cancel",
            "cancelled", "canceled", "unsubscribe", "oppose", "object to",
            "urge you to reject", "should be withdrawn",
        ],
    },
}

# Directional disambiguation: patterns that force a match to one construct even
# though a shorter pattern of another construct also fires.  (Documented in the
# manuscript as the SI/BI "recommend" boundary rule.)
PRIORITY_PATTERNS = [
    # third party recommends -> SI, not BI
    (r"\b(friend|friends|wife|husband|family|colleague|coworker|neighbou?r|sister|brother|mother|father|someone|somebody|they|he|she)\s+\w{0,12}\s?recommend", "SI", +1.0),
    (r"\b(was|were|got)\s+recommended\b", "SI", +1.0),
    # writer recommends -> BI
    (r"\bi\s+(would\s+|will\s+|do\s+|highly\s+|definitely\s+)*recommend", "BI", +1.0),
]

NEGATORS = {
    "not", "no", "never", "n't", "cannot", "cant", "can't", "won't", "wont",
    "wouldn't", "wouldnt", "didn't", "didnt", "doesn't", "doesnt", "isn't",
    "isnt", "aren't", "arent", "hardly", "barely", "without", "lacks", "lacking",
    "nothing", "none", "neither", "nor",
}

INTENSIFIERS = {
    "very": 1.5, "really": 1.4, "extremely": 1.7, "incredibly": 1.7,
    "absolutely": 1.6, "so": 1.3, "super": 1.4, "totally": 1.4, "highly": 1.4,
    "completely": 1.5, "utterly": 1.6, "quite": 1.15, "truly": 1.4,
}

DOWNTONERS = {
    "somewhat": 0.6, "slightly": 0.5, "a bit": 0.6, "kind of": 0.6,
    "kinda": 0.6, "sort of": 0.6, "fairly": 0.8, "rather": 0.8,
    "a little": 0.6, "mostly": 0.85, "generally": 0.85,
}

CONTRASTS = {"but", "however", "although", "though", "except", "unfortunately"}


def construct_anchor_terms(construct: str) -> list[str]:
    """Flat list of surface terms for a construct (both polarities).

    Used by the topic arm to map topics onto constructs and by the encoder /
    LLM arms only for reporting -- never as training signal.
    """
    out = []
    for pol in ("pos", "neg"):
        for t in LEXICON[construct][pol]:
            term = t[0] if isinstance(t, tuple) else t
            out.append(term)
    return out


def as_weighted(entries):
    """Normalise a lexicon entry list to ``[(term, weight), ...]``."""
    res = []
    for t in entries:
        if isinstance(t, tuple):
            res.append((t[0], float(t[1])))
        else:
            res.append((t, 1.0))
    return res


def lexicon_size_table():
    rows = []
    for c in CONSTRUCTS:
        rows.append(dict(construct=c,
                         name=CONSTRUCT_META[c]["short"],
                         n_pos=len(LEXICON[c]["pos"]),
                         n_neg=len(LEXICON[c]["neg"]),
                         n_total=len(LEXICON[c]["pos"]) + len(LEXICON[c]["neg"])))
    return rows
