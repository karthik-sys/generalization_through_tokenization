"""Cheap heuristic domain classifier for homogeneous corpora (arm "routed-webtext" and any
future use of this pattern).

Every other arm's domains come from separately-curated sources (codeparrot/github-code for
code, open-web-math for math, ...) - the domain label is a property of WHERE the text came
from, decided once at dataset-selection time, never inferred from content. That's clean but
means "does routing help" has always been tested on data that was already pre-sorted by
source. This classifier exists so routed can be tested on a single homogeneous corpus (no
pre-existing domain split) and still get per-document domain labels to route on - the
general capability the architecture needs if it's ever going to be validated on data that
isn't already conveniently organized by topic.

Deliberately heuristic, not a trained/ML classifier: this runs inline in a streaming data
pipeline (potentially millions of documents), so it has to be cheap per-document. A regex/
keyword classifier is the right cost point here - trade some label noise for zero added
latency, rather than adding a second model's worth of inference into the data path (or a
slow separate pre-labeling pass over a corpus this size).

Known limitation, stated plainly: a corpus like general web text is overwhelmingly ordinary
prose. Expect the code/math/science buckets to be a small minority of documents and the nlp
bucket to dominate - that's a real property of testing on unsorted data, not a bug in the
classifier. Report the actual label distribution before trusting results from it (see
scripts/smoke_test_domain_classifier.py).
"""

from __future__ import annotations

import re

_CODE_KEYWORDS = re.compile(
    r"\b(def |class |import |function\(|const |let |var |return |public static|"
    r"#include|package |void |print\(|console\.log)\b"
)
_CODE_SYNTAX = re.compile(r"[{};]\s*$|^\s{2,}\S|=>|::|->")

_MATH_SYMBOLS = re.compile(
    r"\\(frac|sum|int|sqrt|alpha|beta|theta|partial|infty|leq|geq)|\\\[|\\\(|"
    r"\$[^$\n]{0,40}(\\[a-zA-Z]+|\^|_|\\frac)[^$\n]{0,40}\$"  # $...$ only counts with real LaTeX-ish content inside
)
_MATH_KEYWORDS = re.compile(
    r"\b(theorem|lemma|proof|equation|corollary|axiom|integral|derivative|"
    r"matrix|eigenvalue|polynomial)\b", re.IGNORECASE
)

_SCIENCE_KEYWORDS = re.compile(
    r"\b(et al\.?|hypothesis|experiment(al)?|methodology|abstract|"
    r"we (observe|report|show|find)|figure \d|table \d|arxiv|doi:|"
    r"p\s*<\s*0\.0\d|statistically significant)\b", re.IGNORECASE
)


def classify_domain(text: str, sample_chars: int = 2000) -> str:
    """Returns one of "code"/"math"/"science"/"nlp". Scores each domain's signal density
    over a leading sample of the text (cheap, and early content is representative enough
    for this purpose) and returns the strongest signal; "nlp" is the default when nothing
    else fires, which will be true for most general web text - that's expected, not an
    error case.
    """
    sample = text[:sample_chars]
    if not sample.strip():
        return "nlp"

    code_hits = len(_CODE_KEYWORDS.findall(sample)) + len(_CODE_SYNTAX.findall(sample))
    math_hits = len(_MATH_SYMBOLS.findall(sample)) + len(_MATH_KEYWORDS.findall(sample))
    science_hits = len(_SCIENCE_KEYWORDS.findall(sample))

    scores = {"code": code_hits, "math": math_hits, "science": science_hits}
    best_domain, best_score = max(scores.items(), key=lambda kv: kv[1])
    # require a minimum signal density, not just "highest of three near-zero scores" -
    # otherwise noise decides the label for ordinary prose with zero real signal
    min_hits = max(2, sample_chars // 500)
    if best_score < min_hits:
        return "nlp"
    return best_domain


# --- nlp sub-bucketing, for the N-domain run --------------------------------------------
# General web text classified as "nlp" above is overwhelmingly the majority of any
# unsorted corpus (that's expected, not a flaw). These sub-buckets split it further by
# register/genre rather than by "topic" (a homogeneous corpus like OpenWebText doesn't
# reliably signal topic the way curated sources do) - register is a cheap, real, and fairly
# stable textual property to detect heuristically. All four sub-buckets share the SAME
# underlying nlp tokenizer (see webtext_domain_stream.py's adapter) - the split only
# creates separate disjoint embedding/head TABLES per bucket, not new vocabularies.

_NEWS_SIGNALS = re.compile(
    r"\b(according to|reported|announced|said (on|in)|officials|spokesperson|"
    r"\d{1,2}:\d{2}\s*(am|pm)|(mon|tue|wed|thu|fri|sat|sun)day)\b", re.IGNORECASE
)
_OPINION_SIGNALS = re.compile(
    r"\b(i think|i believe|in my opinion|imo|honestly|personally|"
    r"should(n't)? |i (love|hate|feel))\b", re.IGNORECASE
)
_NARRATIVE_SIGNALS = re.compile(
    r'"\s*[A-Z][^"]{3,40}[.!?]\s*"|(\bhe|\bshe) (said|whispered|shouted|walked|looked)\b',
    re.IGNORECASE
)
_REFERENCE_SIGNALS = re.compile(
    r"\b(is a|refers to|is defined as|is the|consists of|is used to|"
    r"is one of|is known as)\b", re.IGNORECASE
)

NLP_SUBDOMAINS = ("nlp_news", "nlp_opinion", "nlp_narrative", "nlp_reference")


def classify_nlp_subdomain(text: str, sample_chars: int = 2000) -> str:
    """Only meaningful for text already classified "nlp" by classify_domain() - called
    separately so the two classification passes stay independently testable/tunable."""
    sample = text[:sample_chars]
    scores = {
        "nlp_news": len(_NEWS_SIGNALS.findall(sample)),
        "nlp_opinion": len(_OPINION_SIGNALS.findall(sample)),
        "nlp_narrative": len(_NARRATIVE_SIGNALS.findall(sample)),
        "nlp_reference": len(_REFERENCE_SIGNALS.findall(sample)),
    }
    best_domain, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score == 0:
        # no signal fired at all - fall back to a stable hash-based bucket rather than
        # always defaulting to the same sub-bucket (which would just recreate one giant
        # imbalanced table and defeat the point of splitting further)
        return NLP_SUBDOMAINS[hash(sample[:200]) % len(NLP_SUBDOMAINS)]
    return best_domain
