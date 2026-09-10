"""Loading historical price data for backtests.

Two shapes are supported:

* a **series** -- one instrument over time, used for the index benchmark;
* a **panel** -- many symbols over time, used to fill the strategy's orders.

Both come from ordinary CSV so any data vendor can be used: export
``date,close`` for a series and ``date,symbol,open,high,low,close`` for a panel.
"""

from __future__ import annotations

import csv
import datetime as _dt
import statistics
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from typing import Iterable, Sequence

TRADING_DAYS = 252


class PriceDataError(ValueError):
    """Raised when a price file cannot be understood."""


def parse_date(raw: str) -> _dt.date:
    """Parse the date formats that price exports actually use."""
    raw = raw.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%Y%m%d"):
        try:
            return _dt.datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    try:
        return _dt.datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
    except ValueError as exc:
        raise PriceDataError(f"unrecognised date {raw!r}") from exc


def _pick(header: Sequence[str], *candidates: str) -> str | None:
    """First header whose lowercased name matches one of ``candidates``."""
    lowered = {name.strip().lower(): name for name in header}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    return None


def load_series(
    path: str,
    date_column: str | None = None,
    price_column: str | None = None,
    start: _dt.date | None = None,
    end: _dt.date | None = None,
) -> list[tuple[_dt.date, float]]:
    """Load a ``(date, price)`` series, sorted and de-duplicated.

    Column names are detected when not given: the date column from
    ``date``/``time``, the price from ``adj_close``/``close``/``price``/``value``
    in that order, preferring an adjusted close when one is present.
    """
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise PriceDataError(f"{path} has no header row")

        date_key = date_column or _pick(reader.fieldnames, "date", "time", "timestamp")
        price_key = price_column or _pick(
            reader.fieldnames, "adj_close", "adj close", "adjclose", "close", "price", "value"
        )
        if not date_key or not price_key:
            raise PriceDataError(
                f"{path}: could not find date/price columns in {reader.fieldnames}"
            )

        collected: dict[_dt.date, float] = {}
        for row in reader:
            raw_price = (row.get(price_key) or "").strip()
            if not raw_price:
                continue
            try:
                price = float(raw_price)
            except ValueError:
                continue
            if price <= 0:
                continue
            date = parse_date(row[date_key])
            if start and date < start:
                continue
            if end and date > end:
                continue
            collected[date] = price

    if not collected:
        raise PriceDataError(f"{path}: no usable rows in the requested range")
    return sorted(collected.items())


@dataclass(frozen=True)
class Bar:
    """One session for one symbol."""

    date: _dt.date
    open: float
    high: float
    low: float
    close: float


@dataclass
class PricePanel:
    """Daily bars for many symbols, queried by date.

    ``bars[symbol]`` is sorted ascending by date; the lookup helpers use binary
    search so a long backtest does not rescan the history for every fill.
    """

    bars: dict[str, list[Bar]] = field(default_factory=dict)
    _dates: dict[str, list[_dt.date]] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        for symbol, series in self.bars.items():
            series.sort(key=lambda bar: bar.date)
            self._dates[symbol] = [bar.date for bar in series]

    @property
    def symbols(self) -> list[str]:
        return sorted(self.bars)

    def __contains__(self, symbol: object) -> bool:
        return symbol in self.bars

    def sessions(self) -> list[_dt.date]:
        """Every date any symbol traded, ascending."""
        seen: set[_dt.date] = set()
        for series in self.bars.values():
            seen.update(bar.date for bar in series)
        return sorted(seen)

    def bar_on(self, symbol: str, date: _dt.date) -> Bar | None:
        """The bar for exactly this date, or ``None``."""
        dates = self._dates.get(symbol)
        if not dates:
            return None
        index = bisect_left(dates, date)
        if index < len(dates) and dates[index] == date:
            return self.bars[symbol][index]
        return None

    def next_bar_after(self, symbol: str, date: _dt.date) -> Bar | None:
        """First bar strictly after ``date`` -- the earliest honest fill."""
        dates = self._dates.get(symbol)
        if not dates:
            return None
        index = bisect_right(dates, date)
        return self.bars[symbol][index] if index < len(dates) else None

    def bar_on_or_before(self, symbol: str, date: _dt.date) -> Bar | None:
        """Most recent bar at or before ``date``, for marking positions."""
        dates = self._dates.get(symbol)
        if not dates:
            return None
        index = bisect_right(dates, date) - 1
        return self.bars[symbol][index] if index >= 0 else None

    def bars_between(self, symbol: str, start: _dt.date, end: _dt.date) -> list[Bar]:
        """Bars with ``start < date <= end``."""
        dates = self._dates.get(symbol)
        if not dates:
            return []
        lo = bisect_right(dates, start)
        hi = bisect_right(dates, end)
        return self.bars[symbol][lo:hi]


def load_price_panel(
    path: str,
    start: _dt.date | None = None,
    end: _dt.date | None = None,
) -> PricePanel:
    """Load a long-format panel: ``date,symbol,open,high,low,close``.

    ``open``, ``high`` and ``low`` are optional and fall back to the close, so a
    close-only export still runs (fills then happen at the next close).
    """
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise PriceDataError(f"{path} has no header row")

        date_key = _pick(reader.fieldnames, "date", "time", "timestamp")
        symbol_key = _pick(reader.fieldnames, "symbol", "ticker")
        close_key = _pick(reader.fieldnames, "adj_close", "adj close", "close", "price")
        if not (date_key and symbol_key and close_key):
            raise PriceDataError(
                f"{path}: need date, symbol and close columns; found {reader.fieldnames}"
            )
        open_key = _pick(reader.fieldnames, "open")
        high_key = _pick(reader.fieldnames, "high")
        low_key = _pick(reader.fieldnames, "low")

        collected: dict[str, dict[_dt.date, Bar]] = {}
        for row in reader:
            symbol = (row.get(symbol_key) or "").strip().upper()
            if not symbol:
                continue
            try:
                close = float((row.get(close_key) or "").strip())
            except ValueError:
                continue
            if close <= 0:
                continue
            date = parse_date(row[date_key])
            if start and date < start:
                continue
            if end and date > end:
                continue

            def optional(key: str | None, fallback: float) -> float:
                if not key:
                    return fallback
                try:
                    value = float((row.get(key) or "").strip())
                except ValueError:
                    return fallback
                return value if value > 0 else fallback

            collected.setdefault(symbol, {})[date] = Bar(
                date=date,
                open=optional(open_key, close),
                high=optional(high_key, close),
                low=optional(low_key, close),
                close=close,
            )

    if not collected:
        raise PriceDataError(f"{path}: no usable rows in the requested range")
    return PricePanel({symbol: list(bars.values()) for symbol, bars in collected.items()})


def infer_periods_per_year(dates: Iterable[_dt.date]) -> int:
    """Guess the sampling frequency so annualisation is not silently wrong.

    Monthly index data annualised as if it were daily would overstate
    volatility roughly five-fold, so this is inferred rather than assumed.
    """
    ordered = sorted(set(dates))
    if len(ordered) < 3:
        return TRADING_DAYS
    gaps = [(b - a).days for a, b in zip(ordered, ordered[1:]) if (b - a).days > 0]
    if not gaps:
        return TRADING_DAYS
    median = statistics.median(gaps)
    if median <= 4:
        return TRADING_DAYS
    if median <= 10:
        return 52
    if median <= 45:
        return 12
    if median <= 120:
        return 4
    return 1
