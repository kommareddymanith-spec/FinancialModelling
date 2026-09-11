"""Turning a window of headlines into ranked, sized trading signals.

The rule the strategy implements:

1. Count how many of the last hour's WSJ articles name each company.
2. Rank by that count -- the companies "featured the most" are the candidates.
3. Score the tone of the articles that named each one.
4. Positive net tone -> buy. Negative net tone -> short. Mixed or muted -> pass.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

from .models import Headline, Mention, Side, Signal
from .sentiment import score_text
from .universe import Universe

log = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Thresholds and sizing rules. Every default is intentionally cautious."""

    #: How many of the most-featured companies to consider trading.
    top_n: int = 5
    #: A company must be named in at least this many distinct articles.
    min_mentions: int = 2
    #: Mean article tone must clear this to be read as bullish/bearish.
    min_abs_sentiment: float = 0.5
    #: Base position size in account currency, before conviction scaling.
    notional_per_trade: float = 1_000.0
    #: Total notional the run may deploy across all signals.
    max_total_notional: float = 5_000.0
    #: Bounds on the conviction multiplier applied to ``notional_per_trade``.
    min_size_multiplier: float = 0.5
    max_size_multiplier: float = 2.0
    #: Skip a company whose coverage is genuinely two-sided: the share of
    #: articles agreeing with the net direction must be at least this.
    min_agreement: float = 0.6

    def __post_init__(self) -> None:
        if self.top_n < 1:
            raise ValueError("top_n must be at least 1")
        if self.min_mentions < 1:
            raise ValueError("min_mentions must be at least 1")
        if self.notional_per_trade <= 0:
            raise ValueError("notional_per_trade must be positive")
        if self.max_total_notional < self.notional_per_trade:
            raise ValueError("max_total_notional must be >= notional_per_trade")
        if not 0.0 <= self.min_agreement <= 1.0:
            raise ValueError("min_agreement must be between 0 and 1")


def extract_mentions(
    headlines: Iterable[Headline],
    universe: Universe,
    cache: dict[Headline, tuple[list[tuple[str, str]], float]] | None = None,
) -> list[Mention]:
    """Find every (company, article) pair and attach the article's tone.

    Each article is scored once and that score is shared by all companies it
    names, which is the honest reading of a headline like "Ford Beats, GM
    Misses": the sentence-level attribution problem is not solved here, so a
    genuinely mixed article contributes a muted score to both names.

    Pass a ``cache`` dict to memoise the company match and sentiment score per
    article. A backtest with overlapping windows sees the same headline in many
    consecutive decisions; without the memo it re-runs both regex passes every
    time. Results are identical either way -- the cache holds only derived
    values, keyed by the article itself.
    """
    mentions: list[Mention] = []
    for headline in headlines:
        if cache is not None and headline in cache:
            found, polarity = cache[headline]
        else:
            found = universe.find(headline.text)
            polarity = score_text(headline.text).polarity if found else 0.0
            if cache is not None:
                cache[headline] = (found, polarity)
        if not found:
            continue
        for ticker, matched_as in found:
            mentions.append(
                Mention(
                    ticker=ticker,
                    company=universe.name_for(ticker),
                    headline=headline,
                    matched_as=matched_as,
                    sentiment=polarity,
                )
            )
    log.info("extracted %d company mentions", len(mentions))
    return mentions


def build_signals(
    mentions: Sequence[Mention],
    config: StrategyConfig | None = None,
) -> list[Signal]:
    """Aggregate mentions into signals, most-featured company first.

    Every candidate in the top ``top_n`` is returned, including ones the
    strategy declines to trade -- those carry a ``skip_reason`` so a run can be
    audited without re-deriving why a name was passed over.
    """
    config = config or StrategyConfig()

    grouped: dict[str, list[Mention]] = defaultdict(list)
    for mention in mentions:
        grouped[mention.ticker].append(mention)

    candidates: list[Signal] = []
    for ticker, group in grouped.items():
        count = len(group)
        sentiment_sum = sum(m.sentiment for m in group)
        candidates.append(
            Signal(
                ticker=ticker,
                company=group[0].company,
                mention_count=count,
                sentiment_sum=sentiment_sum,
                sentiment_mean=sentiment_sum / count,
                side=Side.FLAT,
                conviction=0.0,
                matched_terms=sorted({m.matched_as for m in group}),
                headlines=[m.headline.title for m in group],
                article_sentiments=[m.sentiment for m in group],
            )
        )

    # "Featured the most" is the primary sort; conviction breaks ties, then
    # ticker so the ordering is deterministic for a given input.
    candidates.sort(
        key=lambda s: (-s.mention_count, -abs(s.sentiment_sum), s.ticker)
    )
    top = candidates[: config.top_n]

    deployed = 0.0
    for signal in top:
        _classify(signal, config)
        if not signal.tradable:
            continue
        notional = _size(signal, config)
        if deployed + notional > config.max_total_notional:
            remaining = config.max_total_notional - deployed
            if remaining < config.notional_per_trade * config.min_size_multiplier:
                signal.skip_reason = "max_total_notional reached"
                signal.notional = 0.0
                continue
            notional = remaining
        signal.notional = round(notional, 2)
        deployed += signal.notional

    log.info(
        "built %d signal(s), %d tradable, %.2f notional deployed",
        len(top),
        sum(1 for s in top if s.tradable),
        deployed,
    )
    return top


def _classify(signal: Signal, config: StrategyConfig) -> None:
    """Set ``side`` and ``skip_reason`` from the aggregated tone."""
    if signal.mention_count < config.min_mentions:
        signal.skip_reason = (
            f"only {signal.mention_count} mention(s), need {config.min_mentions}"
        )
        return

    if abs(signal.sentiment_mean) < config.min_abs_sentiment:
        signal.skip_reason = (
            f"tone {signal.sentiment_mean:+.2f} inside neutral band "
            f"+/-{config.min_abs_sentiment}"
        )
        return

    agreement = signal.agreement()
    if agreement < config.min_agreement:
        signal.skip_reason = (
            f"coverage is two-sided ({agreement:.0%} of articles agree, "
            f"need {config.min_agreement:.0%})"
        )
        return

    signal.side = Side.BUY if signal.sentiment_mean > 0 else Side.SHORT


def _size(signal: Signal, config: StrategyConfig) -> float:
    """Scale the base size by conviction: how often, and how strongly, covered.

    Conviction blends coverage (mentions beyond the minimum) with tone strength
    so a company in five bearish headlines shorts larger than one in two.
    """
    coverage = signal.mention_count / max(1, config.min_mentions)
    strength = abs(signal.sentiment_mean) / max(0.01, config.min_abs_sentiment)
    conviction = (coverage * strength) ** 0.5
    conviction = max(config.min_size_multiplier, min(config.max_size_multiplier, conviction))
    signal.conviction = conviction
    return config.notional_per_trade * conviction
