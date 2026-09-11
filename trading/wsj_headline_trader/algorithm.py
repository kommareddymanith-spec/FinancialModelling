"""The algorithm itself: read WSJ, rank companies, trade the tone."""

from __future__ import annotations

import datetime as _dt
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Sequence

from .broker import Broker, PaperBroker
from .feed import DEFAULT_FEEDS, Fetcher, collect_headlines, fetch_feed
from .models import Order, OrderResult, RunReport, Signal
from .strategy import StrategyConfig, build_signals, extract_mentions
from .universe import Universe

log = logging.getLogger(__name__)


@dataclass
class AlgorithmConfig:
    """Everything that shapes a run."""

    feeds: Sequence[str] = DEFAULT_FEEDS
    #: Lookback in minutes. 60 is "the last hour" from the strategy brief.
    window_minutes: int = 60
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    universe_path: str | None = None
    #: When true, signals are produced and logged but no order is submitted.
    dry_run: bool = True
    #: Skip a signal in a symbol the account already holds. Runs are stateless,
    #: so without this an hourly schedule pyramids into any story that stays in
    #: the news -- three days of Nvidia headlines becomes a position many times
    #: the intended size.
    skip_held_symbols: bool = True
    #: Refuse to submit while the venue reports the market closed. A market
    #: order sent into a closed market is at best queued to an open hours away,
    #: acting on headlines that are no longer news.
    require_market_open: bool = False

    def __post_init__(self) -> None:
        if self.window_minutes < 1:
            raise ValueError("window_minutes must be at least 1")
        if not self.feeds:
            raise ValueError("at least one feed URL is required")


class WSJHeadlineAlgorithm:
    """Trade the companies most featured in the last hour of WSJ headlines.

    Usage::

        algo = WSJHeadlineAlgorithm(AlgorithmConfig(dry_run=False), broker=PaperBroker())
        report = algo.run()

    A run is stateless: it reads the window, decides, and submits. Re-running
    inside the same window will act on the same headlines again, so schedule it
    on a cadence at least as long as ``window_minutes`` (or narrow the window
    to the cadence) unless you want to add to positions.
    """

    def __init__(
        self,
        config: AlgorithmConfig | None = None,
        broker: Broker | None = None,
        universe: Universe | None = None,
        fetcher: Fetcher = fetch_feed,
    ):
        self.config = config or AlgorithmConfig()
        self.broker = broker or PaperBroker()
        self.universe = universe or Universe.load(self.config.universe_path)
        self.fetcher = fetcher

    def run(self, now: _dt.datetime | None = None) -> RunReport:
        """Execute one pass and return what happened."""
        config = self.config
        ran_at = now or _dt.datetime.now(_dt.timezone.utc)

        headlines, errors = collect_headlines(
            feeds=config.feeds,
            window_minutes=config.window_minutes,
            now=ran_at,
            fetcher=self.fetcher,
        )
        report = RunReport(
            ran_at=ran_at,
            window_minutes=config.window_minutes,
            headlines_scanned=len(headlines),
            feeds_read=[url for url in config.feeds],
            feed_errors=errors,
            dry_run=config.dry_run,
        )

        if errors and len(errors) == len(config.feeds):
            log.error("every feed failed; not trading")
            return report

        mentions = extract_mentions(headlines, self.universe)
        report.signals = build_signals(mentions, config.strategy)

        if not config.dry_run and config.require_market_open:
            open_now = self._market_open()
            if open_now is False:
                log.warning("the market is closed; not submitting")
                for signal in report.signals:
                    if signal.tradable:
                        signal.skip_reason = "market closed"
                report.notes.append("market closed, nothing submitted")
                return report
            if open_now is None:
                log.warning("could not confirm the market is open; not submitting")
                for signal in report.signals:
                    if signal.tradable:
                        signal.skip_reason = "market state unknown"
                report.notes.append("market state unknown, nothing submitted")
                return report

        held: set[str] = set()
        if not config.dry_run and config.skip_held_symbols:
            held = self._held_symbols()
            if held:
                log.info("already holding %s", ", ".join(sorted(held)))

        for signal in report.signals:
            if signal.tradable and signal.ticker in held:
                signal.skip_reason = "already holding"
            if not signal.tradable:
                log.info(
                    "skip %s (%d mention(s), tone %+.2f): %s",
                    signal.ticker,
                    signal.mention_count,
                    signal.sentiment_mean,
                    signal.skip_reason or "no direction",
                )
                continue

            order = self._build_order(signal, ran_at)
            if config.dry_run:
                log.info(
                    "DRY RUN would %s %s for %.2f (mentions=%d tone=%+.2f)",
                    signal.side.value,
                    signal.ticker,
                    signal.notional,
                    signal.mention_count,
                    signal.sentiment_mean,
                )
                report.results.append(
                    OrderResult(order=order, accepted=False, message="dry run, not submitted")
                )
                continue

            report.results.append(self.broker.submit(order))

        return report

    def _market_open(self) -> bool | None:
        """Ask the broker whether the market is open, if it can answer.

        ``None`` means unanswerable -- either the broker has no clock or the
        call failed. That is treated as a reason not to trade rather than as
        permission, since firing into an unknown market state is the worse
        error.
        """
        probe = getattr(self.broker, "is_market_open", None)
        if probe is None:
            log.warning(
                "%s cannot report market hours; treating the state as unknown",
                type(self.broker).__name__,
            )
            return None
        return probe()

    def _held_symbols(self) -> set[str]:
        """Symbols the broker already holds, empty if it cannot say."""
        probe = getattr(self.broker, "open_symbols", None)
        if probe is None:
            log.warning(
                "%s cannot report open positions; repeated runs may stack "
                "positions in the same symbol",
                type(self.broker).__name__,
            )
            return set()
        try:
            return {str(symbol).upper() for symbol in probe()}
        except Exception as exc:  # a broker fault must not become a trade
            log.warning("could not read open positions: %s", exc)
            return set()

    def _build_order(self, signal: Signal, ran_at: _dt.datetime) -> Order:
        """Wrap a signal in a broker order, carrying its rationale along."""
        return Order(
            symbol=signal.ticker,
            side=signal.side,
            notional=signal.notional,
            client_order_id=f"wsj-{ran_at:%Y%m%dT%H%M}-{signal.ticker}-{uuid.uuid4().hex[:6]}",
            metadata={
                "strategy": "wsj-headline-frequency-sentiment",
                "window_minutes": self.config.window_minutes,
                "mentions": signal.mention_count,
                "sentiment_mean": round(signal.sentiment_mean, 3),
                "matched_terms": signal.matched_terms,
                "headlines": signal.headlines,
            },
        )


def append_signal_log(path: str, report: RunReport) -> int:
    """Append a run's tradable signals to a JSONL file, returning how many.

    One line per signal, which is what :mod:`.pine` reads. Appending rather
    than rewriting means a scheduled run builds up a signal history over time
    -- the practical way to get a chartable series without a headline archive.
    """
    rows = [
        {
            "decided_at": report.ran_at.isoformat(),
            "ticker": signal.ticker,
            "company": signal.company,
            "side": signal.side.value,
            "mentions": signal.mention_count,
            "sentiment_mean": round(signal.sentiment_mean, 4),
            "agreement": round(signal.agreement(), 4),
            "notional": round(signal.notional, 2),
            "window_minutes": report.window_minutes,
            "dry_run": report.dry_run,
        }
        for signal in report.signals
        if signal.tradable
    ]
    if not rows:
        return 0
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    log.info("appended %d signal(s) to %s", len(rows), path)
    return len(rows)


def format_report(report: RunReport) -> str:
    """Human-readable run summary for the console."""
    lines = [
        f"WSJ headline trader -- {report.ran_at:%Y-%m-%d %H:%M} UTC",
        f"  window          : last {report.window_minutes} minutes",
        f"  headlines read  : {report.headlines_scanned}",
        f"  mode            : {'DRY RUN' if report.dry_run else 'LIVE SUBMIT'}",
    ]
    if report.feed_errors:
        lines.append(f"  feed errors     : {len(report.feed_errors)}")
        lines.extend(f"      - {err}" for err in report.feed_errors)
    for note in report.notes:
        lines.append(f"  note            : {note}")

    if not report.signals:
        lines.append("\n  No company was named in the window; nothing to trade.")
        return "\n".join(lines)

    lines.append("\n  Most-featured companies")
    lines.append(f"  {'#':>2}  {'TICKER':<7}{'MENTIONS':>9}{'TONE':>8}{'AGREE':>7}  ACTION")
    for rank, signal in enumerate(report.signals, start=1):
        if signal.tradable:
            action = f"{signal.side.value.upper():<5} {signal.notional:,.2f}"
        else:
            action = f"pass ({signal.skip_reason})"
        lines.append(
            f"  {rank:>2}  {signal.ticker:<7}{signal.mention_count:>9}"
            f"{signal.sentiment_mean:>+8.2f}{signal.agreement():>7.0%}  {action}"
        )

    submitted = [r for r in report.results if r.accepted]
    rejected = [r for r in report.results if not r.accepted and "dry run" not in r.message]
    if submitted:
        lines.append("\n  Orders accepted")
        for result in submitted:
            lines.append(
                f"    {result.order.side.value.upper():<5} {result.order.symbol:<7}"
                f" qty={result.filled_qty or result.order.qty or '-'}"
                f" id={result.broker_order_id}"
            )
    if rejected:
        lines.append("\n  Orders rejected")
        for result in rejected:
            lines.append(f"    {result.order.symbol:<7} {result.message}")

    return "\n".join(lines)
