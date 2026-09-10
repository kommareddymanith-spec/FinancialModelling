"""Lexicon sentiment scoring for headline text.

A deliberately transparent, finance-flavoured word list rather than a model:
every order the algorithm places can be traced back to the exact words that
caused it, which is what you want when the position is real money.

Three things make it more than a bag of words:

* **Phrases beat words.** Terms are matched longest-first and non-overlapping,
  so "record loss" scores negative instead of cancelling out, and "cuts
  guidance" is not read as a neutral "cut".
* **Negation flips.** A negator within three tokens before a term inverts it,
  so "doesn't beat expectations" is bearish.
* **Weights.** "collapses" carries more than "slips".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

#: term -> weight. Positive terms.
POSITIVE_TERMS: dict[str, float] = {
    # phrases first (conceptually -- ordering is handled by the matcher)
    "beats expectations": 2.0,
    "beat expectations": 2.0,
    "tops estimates": 2.0,
    "beats estimates": 2.0,
    "above expectations": 1.5,
    "better than expected": 1.5,
    "stronger than expected": 1.5,
    "raises guidance": 2.0,
    "raised guidance": 2.0,
    "raises outlook": 2.0,
    "lifts outlook": 2.0,
    "record profit": 2.0,
    "record revenue": 2.0,
    "record high": 1.5,
    "all-time high": 1.5,
    "strikes a deal": 1.0,
    "strikes deal": 1.0,
    "wins approval": 1.5,
    "share buyback": 1.5,
    "raises dividend": 1.5,
    # single words
    "beat": 1.0,
    "beats": 1.0,
    "surge": 1.5,
    "surges": 1.5,
    "surged": 1.5,
    "soar": 1.5,
    "soars": 1.5,
    "soared": 1.5,
    "jump": 1.0,
    "jumps": 1.0,
    "jumped": 1.0,
    "rally": 1.0,
    "rallies": 1.0,
    "rallied": 1.0,
    "climb": 0.5,
    "climbs": 0.5,
    "climbed": 0.5,
    "rise": 0.5,
    "rises": 0.5,
    "rose": 0.5,
    "gain": 0.5,
    "gains": 0.5,
    "gained": 0.5,
    "rebound": 1.0,
    "rebounds": 1.0,
    "rebounded": 1.0,
    "recovery": 1.0,
    "recovered": 1.0,
    "upbeat": 1.0,
    "upgrade": 1.5,
    "upgrades": 1.5,
    "upgraded": 1.5,
    "outperform": 1.5,
    "outperformed": 1.5,
    "strong": 1.0,
    "stronger": 1.0,
    "strongest": 1.0,
    "strength": 1.0,
    "strengthen": 1.0,
    "strengthened": 1.0,
    "robust": 1.0,
    "resilient": 1.0,
    "profit": 0.5,
    "profits": 0.5,
    "profitable": 1.0,
    "growth": 1.0,
    "grew": 1.0,
    "growing": 0.5,
    "expansion": 0.5,
    "expands": 0.5,
    "boost": 1.0,
    "boosts": 1.0,
    "boosted": 1.0,
    "exceed": 1.0,
    "exceeds": 1.0,
    "exceeded": 1.0,
    "improve": 0.5,
    "improves": 0.5,
    "improved": 1.0,
    "improvement": 1.0,
    "win": 1.0,
    "wins": 1.0,
    "won": 1.0,
    "approval": 1.0,
    "approved": 1.0,
    "breakthrough": 1.5,
    "optimism": 1.0,
    "optimistic": 1.0,
    "bullish": 1.5,
    "buyback": 1.5,
    "milestone": 1.0,
    "momentum": 0.5,
    "blowout": 2.0,
    "upside": 1.0,
    "accelerates": 0.5,
    "accelerated": 0.5,
    "secures": 1.0,
    "secured": 1.0,
    "landmark": 1.0,
    "successful": 1.0,
}

#: term -> weight. Negative terms (stored positive, applied as negative).
NEGATIVE_TERMS: dict[str, float] = {
    "misses expectations": 2.0,
    "missed expectations": 2.0,
    "misses estimates": 2.0,
    "below expectations": 1.5,
    "worse than expected": 1.5,
    "weaker than expected": 1.5,
    "cuts guidance": 2.0,
    "cut guidance": 2.0,
    "cuts outlook": 2.0,
    "lowers guidance": 2.0,
    "lowers outlook": 2.0,
    "profit warning": 2.0,
    "record loss": 2.0,
    "record low": 1.5,
    "all-time low": 1.5,
    "job cuts": 1.5,
    "class action": 1.5,
    "files for bankruptcy": 2.5,
    "goes bankrupt": 2.5,
    "short seller": 1.5,
    "accounting irregularities": 2.5,
    "profit falls": 1.5,
    "miss": 1.0,
    "misses": 1.0,
    "missed": 1.0,
    "plunge": 2.0,
    "plunges": 2.0,
    "plunged": 2.0,
    "plummet": 2.0,
    "plummets": 2.0,
    "plummeted": 2.0,
    "collapse": 2.0,
    "collapses": 2.0,
    "collapsed": 2.0,
    "crash": 2.0,
    "crashed": 2.0,
    "slump": 1.5,
    "slumps": 1.5,
    "slumped": 1.5,
    "sink": 1.5,
    "sinks": 1.5,
    "sank": 1.5,
    "tumble": 1.5,
    "tumbles": 1.5,
    "tumbled": 1.5,
    "slide": 1.0,
    "slides": 1.0,
    "slid": 1.0,
    "slip": 0.5,
    "slips": 0.5,
    "slipped": 0.5,
    "fall": 0.5,
    "falls": 0.5,
    "fell": 0.5,
    "drop": 0.5,
    "drops": 0.5,
    "dropped": 0.5,
    "decline": 1.0,
    "declines": 1.0,
    "declined": 1.0,
    "loss": 1.0,
    "losses": 1.0,
    "lost": 1.0,
    "weak": 1.0,
    "weaker": 1.0,
    "weakest": 1.0,
    "weakness": 1.0,
    "downgrade": 1.5,
    "downgrades": 1.5,
    "downgraded": 1.5,
    "underperform": 1.5,
    "underperformed": 1.5,
    "slash": 1.5,
    "slashes": 1.5,
    "slashed": 1.5,
    "layoff": 1.5,
    "layoffs": 1.5,
    "lawsuit": 1.5,
    "sued": 1.5,
    "sues": 1.0,
    "probe": 1.5,
    "probes": 1.5,
    "investigation": 1.5,
    "subpoena": 1.5,
    "fraud": 2.5,
    "scandal": 2.0,
    "recall": 1.5,
    "recalls": 1.5,
    "recalled": 1.5,
    "halt": 1.0,
    "halted": 1.0,
    "warning": 1.0,
    "warns": 1.0,
    "warned": 1.0,
    "bankruptcy": 2.5,
    "bankrupt": 2.5,
    "default": 2.0,
    "defaults": 2.0,
    "defaulted": 2.0,
    "delay": 0.5,
    "delays": 0.5,
    "delayed": 0.5,
    "shortfall": 1.5,
    "disappoint": 1.5,
    "disappoints": 1.5,
    "disappointing": 1.5,
    "bearish": 1.5,
    "concerns": 0.5,
    "worries": 0.5,
    "fined": 1.5,
    "fines": 1.0,
    "penalty": 1.0,
    "penalties": 1.0,
    "breach": 1.5,
    "hacked": 1.5,
    "outage": 1.5,
    "resigns": 1.0,
    "resigned": 1.0,
    "resignation": 1.0,
    "ousted": 1.5,
    "ouster": 1.5,
    "strike": 1.0,
    "strikes": 0.5,
    "glut": 1.0,
    "oversupply": 1.0,
    "writedown": 1.5,
    "write-down": 1.5,
    "impairment": 1.5,
    "downturn": 1.0,
    "slowdown": 1.0,
    "slowing": 1.0,
    "sluggish": 1.0,
    "stalled": 1.0,
    "crisis": 1.5,
    "antitrust": 1.0,
    "delisting": 2.0,
}

#: A negator this close before a term inverts its polarity.
NEGATORS = frozenset(
    """
    not no never without none nothing barely hardly fails failed fail
    lacks lack lacking isn't aren't wasn't weren't doesn't don't didn't
    won't cannot can't couldn't shouldn't unlikely denies denied dismissed
    stops stopped avoids avoided halts less fewer
    """.split()
)

#: Tokens before a term that are scanned for a negator.
NEGATION_WINDOW = 3

_TOKEN_RE = re.compile(r"[\w'’-]+")


def _build_matcher() -> re.Pattern[str]:
    """One regex over every term, longest first so phrases win."""
    terms = sorted(set(POSITIVE_TERMS) | set(NEGATIVE_TERMS), key=len, reverse=True)
    parts = []
    for term in terms:
        body = re.escape(term).replace(r"\ ", r"\s+")
        parts.append(rf"\b{body}\b")
    return re.compile("|".join(parts), re.IGNORECASE)


_MATCHER = _build_matcher()


@dataclass
class SentimentScore:
    """The outcome of scoring one piece of text."""

    score: float = 0.0
    positive_hits: list[str] = field(default_factory=list)
    negative_hits: list[str] = field(default_factory=list)
    negated_hits: list[str] = field(default_factory=list)

    @property
    def hit_count(self) -> int:
        return len(self.positive_hits) + len(self.negative_hits)

    @property
    def polarity(self) -> float:
        """Score normalised by the number of terms found, roughly [-2.5, 2.5]."""
        return self.score / self.hit_count if self.hit_count else 0.0

    @property
    def terms(self) -> list[str]:
        return self.positive_hits + self.negative_hits


def _is_negated(text: str, start: int) -> bool:
    """Is there a negator within :data:`NEGATION_WINDOW` tokens before ``start``?"""
    preceding = _TOKEN_RE.findall(text[max(0, start - 80) : start])[-NEGATION_WINDOW:]
    return any(token.casefold().replace("’", "'") in NEGATORS for token in preceding)


def score_text(text: str) -> SentimentScore:
    """Score ``text``; positive means bullish, negative bearish."""
    result = SentimentScore()
    if not text:
        return result

    for match in _MATCHER.finditer(text):
        term = re.sub(r"\s+", " ", match.group(0)).strip().casefold()
        weight = POSITIVE_TERMS.get(term)
        polarity = 1.0
        if weight is None:
            weight = NEGATIVE_TERMS.get(term)
            polarity = -1.0
        if weight is None:  # pragma: no cover - matcher is built from the dicts
            continue

        if _is_negated(text, match.start()):
            polarity = -polarity
            result.negated_hits.append(term)

        result.score += polarity * weight
        (result.positive_hits if polarity > 0 else result.negative_hits).append(term)

    return result
