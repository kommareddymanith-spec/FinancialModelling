"""The comparison plan: a fixed amount into an index fund every month.

This is the thing the strategy has to beat. It is deliberately the naive
version -- buy on a schedule, never sell, never look at the news.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Sequence

from .metrics import PerformanceSummary, summarise
from .prices import infer_periods_per_year


@dataclass
class DcaConfig:
    """A monthly savings plan."""

    contribution: float = 1_000.0
    #: Day of month to buy on; the next available session is used if closed.
    day_of_month: int = 1
    #: Charged on each purchase, as a fraction of the amount invested.
    expense_bps: float = 0.0
    label: str = "S&P 500 DCA"

    def __post_init__(self) -> None:
        if self.contribution <= 0:
            raise ValueError("contribution must be positive")
        if not 1 <= self.day_of_month <= 28:
            raise ValueError("day_of_month must be between 1 and 28")


@dataclass
class DcaResult:
    """A completed savings plan."""

    equity: list[tuple[_dt.date, float]] = field(default_factory=list)
    contributions: dict[_dt.date, float] = field(default_factory=dict)
    purchases: list[tuple[_dt.date, float, float]] = field(default_factory=list)
    units: float = 0.0
    periods_per_year: int = 12

    def summary(self, label: str, risk_free_annual: float = 0.0) -> PerformanceSummary:
        return summarise(
            label=label,
            equity=self.equity,
            contributions=self.contributions,
            risk_free_annual=risk_free_annual,
            periods=self.periods_per_year,
            extras={
                "Purchases": len(self.purchases),
                "Units held": f"{self.units:.4f}",
                "Average cost": (
                    f"{sum(a for _, _, a in self.purchases) / self.units:,.2f}"
                    if self.units
                    else "-"
                ),
            },
        )


def _contribution_dates(
    start: _dt.date, end: _dt.date, day_of_month: int
) -> list[_dt.date]:
    """Every scheduled buy date from ``start``'s month up to ``end``.

    A scheduled date earlier in the first month is kept rather than dropped:
    if the plan says "the 1st" and the first available session is the 3rd, that
    month's contribution is invested on the 3rd. Dropping it would silently
    lose a month's money whenever the period opened on a closed market.
    """
    dates = []
    year, month = start.year, start.month
    while True:
        candidate = _dt.date(year, month, day_of_month)
        if candidate > end:
            break
        dates.append(candidate)
        month += 1
        if month > 12:
            year, month = year + 1, 1
    return dates


def run_dca(
    series: Sequence[tuple[_dt.date, float]],
    config: DcaConfig | None = None,
) -> DcaResult:
    """Invest ``contribution`` on schedule and hold, marking to market.

    Fractional units are allowed, which is how index funds and most brokers
    actually work for a fixed-dollar purchase.
    """
    config = config or DcaConfig()
    if not series:
        raise ValueError("price series is empty")

    series = sorted(series)
    dates = [date for date, _ in series]
    result = DcaResult(periods_per_year=infer_periods_per_year(dates))

    scheduled = _contribution_dates(dates[0], dates[-1], config.day_of_month)
    pending = list(scheduled)
    units = 0.0

    for date, price in series:
        # Buy on the first session at or after each scheduled date, so a
        # weekend or holiday schedule slips forward rather than being skipped.
        while pending and pending[0] <= date:
            pending.pop(0)
            invested = config.contribution * (1.0 - config.expense_bps / 10_000.0)
            bought = invested / price
            units += bought
            result.contributions[date] = (
                result.contributions.get(date, 0.0) + config.contribution
            )
            result.purchases.append((date, bought, config.contribution))
        result.equity.append((date, units * price))

    result.units = units
    return result
