"""Performance measurement shared by the strategy backtest and the benchmark.

Comparing a trading strategy to a monthly savings plan needs care: both
receive cash over time, so raw profit and raw equity-curve drawdown both
flatter whichever one received money earlier. The measures here separate the
two effects:

* **Money-weighted return (IRR)** answers "what rate did my contributions
  actually earn", and is the number to compare between two plans funded on the
  same schedule.
* **Time-weighted return (TWR)** strips contributions out entirely and measures
  the decisions alone, which is what drawdown, volatility and Sharpe are
  computed from -- a drawdown taken on raw equity would be masked by the next
  month's deposit.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from typing import Sequence

DAYS_PER_YEAR = 365.25
TRADING_DAYS = 252


@dataclass(frozen=True)
class CashFlow:
    """Money entering (negative) or leaving (positive) the investor's pocket."""

    date: _dt.date
    amount: float


@dataclass
class PerformanceSummary:
    """Everything reported about one strategy or benchmark."""

    label: str
    start: _dt.date
    end: _dt.date
    contributions: float
    final_value: float
    irr_annual: float | None
    twr_total: float
    twr_annual: float
    max_drawdown: float
    volatility_annual: float
    sharpe: float
    best_day: float = 0.0
    worst_day: float = 0.0
    extras: dict[str, object] = field(default_factory=dict)

    @property
    def profit(self) -> float:
        return self.final_value - self.contributions

    @property
    def profit_pct_of_contributions(self) -> float:
        return self.profit / self.contributions if self.contributions else 0.0

    @property
    def years(self) -> float:
        return max((self.end - self.start).days / DAYS_PER_YEAR, 1e-9)


def net_present_value(cashflows: Sequence[CashFlow], annual_rate: float) -> float:
    """NPV of dated cashflows at ``annual_rate``, ACT/365.25 discounting."""
    if not cashflows:
        return 0.0
    origin = min(cf.date for cf in cashflows)
    total = 0.0
    for cf in cashflows:
        years = (cf.date - origin).days / DAYS_PER_YEAR
        total += cf.amount / ((1.0 + annual_rate) ** years)
    return total


def irr(cashflows: Sequence[CashFlow], low: float = -0.9999, high: float = 10.0) -> float | None:
    """Annualised money-weighted return, by bisection.

    Returns ``None`` when the cashflows do not bracket a root -- an all-negative
    or all-positive series has no meaningful rate of return.
    """
    if len(cashflows) < 2:
        return None
    if not (any(cf.amount < 0 for cf in cashflows) and any(cf.amount > 0 for cf in cashflows)):
        return None

    npv_low = net_present_value(cashflows, low)
    npv_high = net_present_value(cashflows, high)
    if npv_low * npv_high > 0:
        return None

    for _ in range(200):
        mid = (low + high) / 2.0
        value = net_present_value(cashflows, mid)
        if abs(value) < 1e-9:
            return mid
        if value * npv_low > 0:
            low, npv_low = mid, value
        else:
            high = mid
    return (low + high) / 2.0


def daily_returns(
    equity: Sequence[tuple[_dt.date, float]],
    contributions: dict[_dt.date, float] | None = None,
) -> list[tuple[_dt.date, float]]:
    """Time-weighted daily returns, neutralising same-day contributions.

    A deposit is assumed to land before the day's close, so it is removed from
    the closing value before the return is taken. Without this a $1,000 deposit
    into a $1,000 account would read as a 100% gain.
    """
    contributions = contributions or {}
    out: list[tuple[_dt.date, float]] = []
    for (_, previous), (date, value) in zip(equity, equity[1:]):
        if previous <= 0:
            out.append((date, 0.0))
            continue
        flow = contributions.get(date, 0.0)
        out.append((date, (value - flow) / previous - 1.0))
    return out


def growth_index(returns: Sequence[tuple[_dt.date, float]]) -> list[tuple[_dt.date, float]]:
    """Compound daily returns into a growth-of-1 series."""
    level = 1.0
    out = []
    for date, ret in returns:
        level *= 1.0 + ret
        out.append((date, level))
    return out


def max_drawdown(index: Sequence[tuple[_dt.date, float]]) -> float:
    """Deepest peak-to-trough fall of a growth index, as a negative fraction."""
    peak = -math.inf
    worst = 0.0
    for _, level in index:
        peak = max(peak, level)
        if peak > 0:
            worst = min(worst, level / peak - 1.0)
    return worst


def volatility(returns: Sequence[tuple[_dt.date, float]], periods: int = TRADING_DAYS) -> float:
    """Annualised standard deviation of returns."""
    values = [r for _, r in returns]
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance) * math.sqrt(periods)


def sharpe(
    returns: Sequence[tuple[_dt.date, float]],
    risk_free_annual: float = 0.0,
    periods: int = TRADING_DAYS,
) -> float:
    """Annualised Sharpe ratio of a return series.

    A series with no dispersion has an undefined Sharpe ratio, reported here as
    0.0. The test is against a small tolerance rather than exact zero: summing
    a constant series leaves float noise around 1e-18, which divided into a
    real mean would otherwise report a Sharpe in the quadrillions.
    """
    values = [r for _, r in returns]
    if len(values) < 2:
        return 0.0
    per_period_rf = (1.0 + risk_free_annual) ** (1.0 / periods) - 1.0
    excess = [v - per_period_rf for v in values]
    mean = sum(excess) / len(excess)
    variance = sum((v - mean) ** 2 for v in excess) / (len(excess) - 1)
    stdev = math.sqrt(variance)
    if stdev < 1e-12:
        return 0.0
    return (mean / stdev) * math.sqrt(periods)


def summarise(
    label: str,
    equity: Sequence[tuple[_dt.date, float]],
    contributions: dict[_dt.date, float],
    risk_free_annual: float = 0.0,
    periods: int = TRADING_DAYS,
    extras: dict[str, object] | None = None,
) -> PerformanceSummary:
    """Build a :class:`PerformanceSummary` from an equity curve and its deposits."""
    if not equity:
        raise ValueError("cannot summarise an empty equity curve")

    equity = sorted(equity)
    start, end = equity[0][0], equity[-1][0]
    total_contributed = sum(contributions.values())
    final_value = equity[-1][1]

    returns = daily_returns(equity, contributions)
    index = growth_index(returns)
    twr_total = index[-1][1] - 1.0 if index else 0.0
    years = max((end - start).days / DAYS_PER_YEAR, 1e-9)
    twr_annual = (1.0 + twr_total) ** (1.0 / years) - 1.0 if twr_total > -1.0 else -1.0

    flows = [CashFlow(date, -amount) for date, amount in sorted(contributions.items()) if amount]
    flows.append(CashFlow(end, final_value))

    return PerformanceSummary(
        label=label,
        start=start,
        end=end,
        contributions=total_contributed,
        final_value=final_value,
        irr_annual=irr(flows),
        twr_total=twr_total,
        twr_annual=twr_annual,
        max_drawdown=max_drawdown(index),
        volatility_annual=volatility(returns, periods),
        sharpe=sharpe(returns, risk_free_annual, periods),
        best_day=max((r for _, r in returns), default=0.0),
        worst_day=min((r for _, r in returns), default=0.0),
        extras=extras or {},
    )


def format_comparison(*summaries: PerformanceSummary) -> str:
    """Side-by-side table of two or more summaries."""
    if not summaries:
        return "(nothing to compare)"

    def pct(value: float | None) -> str:
        return "n/a" if value is None else f"{value * 100:+.2f}%"

    def money(value: float) -> str:
        return f"{value:,.2f}"

    rows: list[tuple[str, list[str]]] = [
        ("Period", [f"{s.start:%Y-%m-%d} to {s.end:%Y-%m-%d}" for s in summaries]),
        ("Contributed", [money(s.contributions) for s in summaries]),
        ("Final value", [money(s.final_value) for s in summaries]),
        ("Profit", [money(s.profit) for s in summaries]),
        ("Profit on contributions", [pct(s.profit_pct_of_contributions) for s in summaries]),
        ("Money-weighted return p.a.", [pct(s.irr_annual) for s in summaries]),
        ("Time-weighted return total", [pct(s.twr_total) for s in summaries]),
        ("Time-weighted return p.a.", [pct(s.twr_annual) for s in summaries]),
        ("Max drawdown", [pct(s.max_drawdown) for s in summaries]),
        ("Volatility p.a.", [pct(s.volatility_annual) for s in summaries]),
        ("Sharpe", [f"{s.sharpe:.2f}" for s in summaries]),
    ]

    extra_keys: list[str] = []
    for summary in summaries:
        for key in summary.extras:
            if key not in extra_keys:
                extra_keys.append(key)
    for key in extra_keys:
        rows.append((key, [str(s.extras.get(key, "-")) for s in summaries]))

    label_width = max(len(name) for name, _ in rows)
    col_width = max(
        max((len(cell) for _, cells in rows for cell in cells), default=0),
        max(len(s.label) for s in summaries),
    )

    lines = [
        " " * label_width + "  " + "  ".join(f"{s.label:>{col_width}}" for s in summaries),
        "-" * (label_width + 2 + (col_width + 2) * len(summaries)),
    ]
    for name, cells in rows:
        lines.append(
            f"{name:<{label_width}}  " + "  ".join(f"{cell:>{col_width}}" for cell in cells)
        )
    return "\n".join(lines)
