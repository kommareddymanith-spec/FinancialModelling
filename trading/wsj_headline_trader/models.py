"""Core value objects shared by the WSJ headline trading algorithm."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

#: Longest text the matcher and scorer will read from one article. A real
#: headline plus summary is a few hundred characters; anything beyond this is
#: a broken or hostile feed, and regex work grows with length, so it is cut.
MAX_TEXT_CHARS = 10_000


class Side(str, Enum):
    """Direction of a trade."""

    BUY = "buy"
    SHORT = "short"
    FLAT = "flat"


@dataclass(frozen=True)
class Headline:
    """A single article pulled from a WSJ feed."""

    title: str
    summary: str
    link: str
    published_at: _dt.datetime
    source: str

    @property
    def text(self) -> str:
        """Title plus summary, the text the algorithm reads.

        Truncated to :data:`MAX_TEXT_CHARS`. This is the single choke point
        every consumer goes through, so a multi-megabyte title cannot turn one
        article into seconds of regex scanning however it entered the system.
        """
        combined = f"{self.title}. {self.summary}".strip()
        return combined[:MAX_TEXT_CHARS]

    @property
    def dedupe_key(self) -> str:
        return self.link.strip().lower() or self.title.strip().lower()


@dataclass(frozen=True)
class Mention:
    """A company found inside one headline, with that headline's sentiment."""

    ticker: str
    company: str
    headline: Headline
    matched_as: str
    sentiment: float


@dataclass
class Signal:
    """Aggregated trading intent for one company over the lookback window."""

    ticker: str
    company: str
    mention_count: int
    sentiment_sum: float
    sentiment_mean: float
    side: Side
    conviction: float
    notional: float = 0.0
    matched_terms: list[str] = field(default_factory=list)
    headlines: list[str] = field(default_factory=list)
    article_sentiments: list[float] = field(default_factory=list)
    skip_reason: str | None = None

    @property
    def tradable(self) -> bool:
        return self.side is not Side.FLAT and self.skip_reason is None

    def agreement(self) -> float:
        """Share of scored articles pointing the same way as the net tone.

        Articles with no sentiment terms at all are ignored: they neither
        support nor contradict the direction.
        """
        scored = [s for s in self.article_sentiments if s]
        if not scored:
            return 0.0
        direction = 1.0 if self.sentiment_mean > 0 else -1.0
        return sum(1 for s in scored if s * direction > 0) / len(scored)


@dataclass
class Order:
    """A broker-agnostic order request.

    Exactly one of ``notional`` or ``qty`` is expected to be set; brokers that
    cannot express notional orders (short sales, for instance) convert using a
    price provider.
    """

    symbol: str
    side: Side
    notional: float | None = None
    qty: float | None = None
    order_type: str = "market"
    time_in_force: str = "day"
    client_order_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderResult:
    """Outcome of submitting an order."""

    order: Order
    accepted: bool
    broker_order_id: str | None = None
    filled_qty: float = 0.0
    filled_price: float | None = None
    message: str = ""


@dataclass
class RunReport:
    """Everything one run of the algorithm did, for logging and tests."""

    ran_at: _dt.datetime
    window_minutes: int
    headlines_scanned: int
    feeds_read: list[str] = field(default_factory=list)
    feed_errors: list[str] = field(default_factory=list)
    signals: list[Signal] = field(default_factory=list)
    results: list[OrderResult] = field(default_factory=list)
    dry_run: bool = True

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary."""
        return {
            "ran_at": self.ran_at.isoformat(),
            "window_minutes": self.window_minutes,
            "headlines_scanned": self.headlines_scanned,
            "feeds_read": list(self.feeds_read),
            "feed_errors": list(self.feed_errors),
            "dry_run": self.dry_run,
            "signals": [
                {
                    "ticker": s.ticker,
                    "company": s.company,
                    "mentions": s.mention_count,
                    "sentiment_sum": round(s.sentiment_sum, 3),
                    "sentiment_mean": round(s.sentiment_mean, 3),
                    "agreement": round(s.agreement(), 3),
                    "side": s.side.value,
                    "conviction": round(s.conviction, 3),
                    "notional": round(s.notional, 2),
                    "matched_terms": s.matched_terms,
                    "skip_reason": s.skip_reason,
                    "headlines": s.headlines,
                }
                for s in self.signals
            ],
            "orders": [
                {
                    "symbol": r.order.symbol,
                    "side": r.order.side.value,
                    "notional": r.order.notional,
                    "qty": r.order.qty,
                    "accepted": r.accepted,
                    "broker_order_id": r.broker_order_id,
                    "message": r.message,
                }
                for r in self.results
            ],
        }
