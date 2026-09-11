"""Spotting tradable companies inside headline text.

Headlines name companies in prose ("Nvidia", "J.P. Morgan"), occasionally by
symbol ("(NVDA)", "$NVDA", "Nasdaq: NVDA"). This module resolves both to a
ticker using a JSON universe file that ships with the package.

The awkward cases are company names that are also ordinary English words --
Target, Visa, Gap, Meta, Apple, Shell. Those live under ``weak_aliases`` and
only count when the article carries a corporate cue ("shares", "earnings",
"CEO", ...), which keeps "Investors Target Small Caps" out of the order book.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

log = logging.getLogger(__name__)

DEFAULT_UNIVERSE_PATH = os.path.join(os.path.dirname(__file__), "data", "universe.json")

#: Words that may follow a weak alias when it is being used as a company: the
#: things a company owns ("Target Shares"), is ("Target Inc"), or does ("Target
#: Said"). Deliberately excludes function words and plural "stocks" so that
#: market-wide prose ("Investors Target Small-Cap Stocks") does not qualify.
_TRAILING_CUES = frozenset(
    """
    shares share stock shareholder shareholders stockholders holdings stores
    earnings revenue revenues profit profits sales results guidance outlook
    dividend dividends buyback buybacks ceo cfo coo chairman executives
    inc inc. corp corp. co. plc llc ltd group
    said says reported reports posted posts announced announces unveiled
    unveils plans planned agreed agrees named names faces faced sued sues
    raised raises cut cuts beat beats missed misses warned warns hired hires
    fired fires acquired acquires launched launches recalled recalls denied
    denies settled settles filed files invests
    tops topped slashed slashes halts halted surges surged tumbles tumbled
    slides slid rallies rallied jumps jumped sinks sank falls fell rises rose
    gains gained drops dropped climbs climbed soars soared plunges plunged
    """.split()
)

#: Words that may precede a weak alias: possessive/partitive heads and the
#: descriptor nouns WSJ puts in front of a company name.
_LEADING_CUES = frozenset(
    """
    shares share stock shareholders earnings revenue profit results
    maker giant retailer automaker carmaker chipmaker drugmaker lender bank
    airline chain company supplier owner parent rival unit operator
    """.split()
)

#: How many tokens either side of a weak alias are inspected for a cue.
_CUE_WINDOW = 2

_TOKEN_RE = re.compile(r"[A-Za-z][\w&.'-]*")


#: "(NVDA)", "$NVDA", "Nasdaq: NVDA" -- explicit symbols always count.
_SYMBOL_PATTERNS = (
    re.compile(r"\$([A-Z]{1,5}(?:\.[A-Z])?)\b"),
    re.compile(r"\b(?:NYSE|NASDAQ|Nasdaq|Cboe|OTC)\s*:\s*([A-Z]{1,5}(?:\.[A-Z])?)\b"),
    re.compile(r"\(([A-Z]{1,5}(?:\.[A-Z])?)\)"),
)


@dataclass(frozen=True)
class Company:
    ticker: str
    name: str
    aliases: tuple[str, ...] = ()
    weak_aliases: tuple[str, ...] = ()


def _alias_pattern(alias: str) -> str:
    """Escaped alias with flexible whitespace and sane word boundaries.

    ``\\b`` is only useful next to a word character, so it is applied
    conditionally -- otherwise an alias like "3M" or "P&G" would never match.
    """
    body = re.escape(alias).replace(r"\ ", r"\s+")
    left = r"\b" if alias[:1].isalnum() else ""
    right = r"\b" if alias[-1:].isalnum() else r"(?!\w)"
    return f"{left}{body}{right}"


class Universe:
    """A searchable set of companies."""

    def __init__(self, companies: Iterable[Company]):
        self.companies: dict[str, Company] = {}
        strong: dict[str, str] = {}
        weak: dict[str, str] = {}

        for company in companies:
            if company.ticker in self.companies:
                log.warning("duplicate ticker %s in universe, keeping first", company.ticker)
                continue
            self.companies[company.ticker] = company
            for alias in company.aliases:
                self._register(strong, alias, company.ticker)
            for alias in company.weak_aliases:
                self._register(weak, alias, company.ticker)

        self._strong_lookup = strong
        self._weak_lookup = weak
        self._strong_re = self._compile(strong)
        self._weak_re = self._compile(weak)

    @staticmethod
    def _register(lookup: dict[str, str], alias: str, ticker: str) -> None:
        alias = alias.strip()
        if not alias:
            return
        key = alias.casefold()
        existing = lookup.get(key)
        if existing and existing != ticker:
            log.warning(
                "alias %r maps to both %s and %s; keeping %s", alias, existing, ticker, existing
            )
            return
        lookup[key] = ticker

    @staticmethod
    def _compile(lookup: dict[str, str]) -> re.Pattern[str] | None:
        if not lookup:
            return None
        # Longest alias first so "Apple Inc" wins over "Apple" and phrases are
        # never shadowed by the single word they contain.
        aliases = sorted(lookup, key=len, reverse=True)
        return re.compile("|".join(_alias_pattern(a) for a in aliases), re.IGNORECASE)

    # -- loading ---------------------------------------------------------

    @classmethod
    def load(cls, path: str | None = None) -> "Universe":
        """Load a universe from JSON, defaulting to the bundled file."""
        path = path or DEFAULT_UNIVERSE_PATH
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        companies = []
        for ticker, entry in raw.items():
            if ticker.startswith("_"):  # comment keys
                continue
            companies.append(
                Company(
                    ticker=ticker,
                    name=entry.get("name", ticker),
                    aliases=tuple(entry.get("aliases", ())),
                    weak_aliases=tuple(entry.get("weak_aliases", ())),
                )
            )
        universe = cls(companies)
        log.info("loaded %d companies from %s", len(universe.companies), path)
        return universe

    # -- matching --------------------------------------------------------

    def find(self, text: str) -> list[tuple[str, str]]:
        """Return ``(ticker, matched_as)`` for every company named in ``text``.

        One entry per company, in order of first appearance -- a headline that
        repeats a name is still a single mention of that company.
        """
        if not text:
            return []

        hits: dict[str, str] = {}

        for pattern in _SYMBOL_PATTERNS:
            for match in pattern.finditer(text):
                symbol = match.group(1)
                if symbol in self.companies:
                    hits.setdefault(symbol, symbol)

        if self._strong_re is not None:
            for match in self._strong_re.finditer(text):
                ticker = self._strong_lookup[_normalise(match.group(0))]
                hits.setdefault(ticker, match.group(0))

        if self._weak_re is not None:
            for match in self._weak_re.finditer(text):
                if not _reads_as_company(text, match.start(), match.end()):
                    continue
                ticker = self._weak_lookup[_normalise(match.group(0))]
                hits.setdefault(ticker, match.group(0))

        return list(hits.items())

    def name_for(self, ticker: str) -> str:
        company = self.companies.get(ticker)
        return company.name if company else ticker

    def __len__(self) -> int:
        return len(self.companies)

    def __contains__(self, ticker: object) -> bool:
        return ticker in self.companies


def _reads_as_company(text: str, start: int, end: int) -> bool:
    """Decide whether an ambiguous alias is naming a company in this sentence.

    True when the alias is possessive ("Target's"), is followed within
    :data:`_CUE_WINDOW` tokens by something only a company has or does, or is
    preceded within that window by a possessive/descriptor head ("Shares of
    Target", "retailer Target"). Everything else is treated as the ordinary
    English word and ignored.
    """
    if text[end : end + 2].lower() in ("'s", "\u2019s"):
        return True

    after = _TOKEN_RE.findall(text[end : end + 60])[:_CUE_WINDOW]
    if any(token.casefold() in _TRAILING_CUES for token in after):
        return True

    before = _TOKEN_RE.findall(text[max(0, start - 60) : start])[-_CUE_WINDOW:]
    return any(token.casefold() in _LEADING_CUES for token in before)


def _normalise(matched: str) -> str:
    """Casefold a match and collapse whitespace back to the alias key form."""
    return re.sub(r"\s+", " ", matched).strip().casefold()


def universe_from_pairs(pairs: Sequence[tuple[str, str]]) -> Universe:
    """Small helper for tests: build a universe from ``(ticker, name)`` pairs."""
    return Universe(Company(ticker=t, name=n, aliases=(n,)) for t, n in pairs)
