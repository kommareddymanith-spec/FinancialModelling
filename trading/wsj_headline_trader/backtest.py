"""Walk-forward backtest of the WSJ headline strategy.

The engine replays history the way the live algorithm experiences it: at each
decision time it sees only headlines already published, runs the *same*
`extract_mentions` / `build_signals` pipeline, and fills the resulting orders
at the next price that could not have been known when the decision was made.

Design choices that matter for whether the result means anything:

* **No look-ahead.** A signal decided at time *t* fills at the open of the
  first session whose opening bell is strictly after *t*. A headline published
  during the session cannot be traded at that session's open.
* **Same cash flows as the benchmark.** The strategy receives the same
  contribution every month as the savings plan it is compared against.
  Comparing a fully-funded strategy to a plan that drip-feeds cash would
  otherwise flatter whichever got its money in earlier.
* **Costs are charged, not assumed away.** Slippage, commission and short
  borrow all accrue; a headline strategy trades often enough that ignoring
  them changes the sign of the answer.
* **Unfillable orders are recorded, not silently dropped.** A signal in a
  symbol with no price data, or with no cash behind it, shows up in
  ``skipped`` so coverage gaps cannot masquerade as good behaviour.
"""

from __future__ import annotations

import datetime as _dt
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from .metrics import PerformanceSummary, summarise
from .models import Headline, Side, Signal
from .prices import PricePanel
from .strategy import StrategyConfig, build_signals, extract_mentions
from .universe import Universe

log = logging.getLogger(__name__)

#: US equity opening bell in UTC. 13:30 is 09:30 New York in winter; the
#: half-hour DST wobble is immaterial next to a next-open fill assumption.
DEFAULT_MARKET_OPEN_UTC = _dt.time(13, 30)


@dataclass
class BacktestConfig:
    """How the replay is run."""

    #: Lookback the strategy reads at each decision, in minutes.
    window_minutes: int = 60
    #: How often a decision is made. Equal to the window means no overlap.
    step_minutes: int = 60
    strategy: StrategyConfig = field(default_factory=StrategyConfig)

    #: Monthly contribution, matching the benchmark savings plan.
    contribution: float = 1_000.0
    contribution_day: int = 1
    starting_cash: float = 0.0

    #: Sessions to hold before the time stop closes the position.
    hold_days: int = 5
    #: Optional intrabar exits, as positive fractions (0.05 == 5%).
    stop_loss: float | None = None
    take_profit: float | None = None

    #: Half-spread plus market impact, charged on entry and exit.
    slippage_bps: float = 5.0
    commission_per_trade: float = 0.0
    #: Annual stock-borrow cost, accrued daily on short market value.
    borrow_rate_annual: float = 0.03
    #: Free cash required to open a short, as a multiple of its notional.
    short_margin: float = 1.0

    market_open_utc: _dt.time = DEFAULT_MARKET_OPEN_UTC
    risk_free_annual: float = 0.0
    label: str = "WSJ headline strategy"

    def __post_init__(self) -> None:
        if self.window_minutes < 1 or self.step_minutes < 1:
            raise ValueError("window_minutes and step_minutes must be at least 1")
        if self.hold_days < 1:
            raise ValueError("hold_days must be at least 1")
        if self.contribution < 0 or self.starting_cash < 0:
            raise ValueError("contribution and starting_cash cannot be negative")
        if not 1 <= self.contribution_day <= 28:
            raise ValueError("contribution_day must be between 1 and 28")
        for name in ("stop_loss", "take_profit"):
            value = getattr(self, name)
            if value is not None and not 0 < value < 1:
                raise ValueError(f"{name} must be a fraction between 0 and 1")
        if self.slippage_bps < 0 or self.commission_per_trade < 0:
            raise ValueError("costs cannot be negative")


@dataclass
class Position:
    """An open position, with the decision that opened it."""

    symbol: str
    side: Side
    qty: float
    entry_date: _dt.date
    #: Price actually paid, slippage included. Stop levels hang off this.
    entry_price: float
    #: Unslipped bar price at entry. P&L is measured against this so that
    #: slippage appears in ``costs`` once and only once.
    entry_reference: float = 0.0
    sessions_held: int = 0
    mentions: int = 0
    sentiment: float = 0.0
    borrow_paid: float = 0.0

    def __post_init__(self) -> None:
        if not self.entry_reference:
            self.entry_reference = self.entry_price

    @property
    def notional(self) -> float:
        return self.qty * self.entry_price


@dataclass
class Trade:
    """A closed round trip."""

    symbol: str
    side: Side
    qty: float
    entry_date: _dt.date
    entry_price: float
    entry_reference: float
    exit_date: _dt.date
    exit_price: float
    gross_pnl: float
    costs: float
    exit_reason: str
    mentions: int
    sentiment: float

    @property
    def net_pnl(self) -> float:
        """P&L after all costs. Equals the position's actual cash impact."""
        return self.gross_pnl - self.costs

    @property
    def return_pct(self) -> float:
        base = self.qty * self.entry_reference
        return self.net_pnl / base if base else 0.0


@dataclass
class BacktestResult:
    """Everything the replay produced."""

    equity: list[tuple[_dt.date, float]] = field(default_factory=list)
    contributions: dict[_dt.date, float] = field(default_factory=dict)
    trades: list[Trade] = field(default_factory=list)
    open_at_end: list[Position] = field(default_factory=list)
    skipped: Counter = field(default_factory=Counter)
    decisions: int = 0
    signals_generated: int = 0
    headlines_seen: int = 0
    total_costs: float = 0.0

    @property
    def wins(self) -> list[Trade]:
        return [t for t in self.trades if t.net_pnl > 0]

    @property
    def losses(self) -> list[Trade]:
        return [t for t in self.trades if t.net_pnl < 0]

    @property
    def win_rate(self) -> float:
        return len(self.wins) / len(self.trades) if self.trades else 0.0

    def summary(self, label: str, risk_free_annual: float = 0.0) -> PerformanceSummary:
        average_win = (
            sum(t.net_pnl for t in self.wins) / len(self.wins) if self.wins else 0.0
        )
        average_loss = (
            sum(t.net_pnl for t in self.losses) / len(self.losses) if self.losses else 0.0
        )
        return summarise(
            label=label,
            equity=self.equity,
            contributions=self.contributions,
            risk_free_annual=risk_free_annual,
            periods=252,
            extras={
                "Trades": len(self.trades),
                "Win rate": f"{self.win_rate * 100:.1f}%",
                "Average win": f"{average_win:,.2f}",
                "Average loss": f"{average_loss:,.2f}",
                "Costs paid": f"{self.total_costs:,.2f}",
                "Orders skipped": sum(self.skipped.values()),
            },
        )


def _decision_times(
    start: _dt.datetime, end: _dt.datetime, step_minutes: int
) -> list[_dt.datetime]:
    step = _dt.timedelta(minutes=step_minutes)
    times = []
    current = start
    while current <= end:
        times.append(current)
        current += step
    return times


def _fill_session(
    decision_time: _dt.datetime, sessions: Sequence[_dt.date], market_open: _dt.time
) -> _dt.date | None:
    """First session whose opening bell is strictly after ``decision_time``.

    This is the no-look-ahead rule: a decision made while the market is open
    waits for the next open, and one made overnight trades that morning.
    """
    for session in sessions:
        bell = _dt.datetime.combine(session, market_open, tzinfo=_dt.timezone.utc)
        if bell > decision_time:
            return session
    return None


def _contribution_due(date: _dt.date, last: _dt.date | None, day: int) -> bool:
    """Has the scheduled contribution for this month not yet been paid?"""
    if date.day < day:
        return False
    if last is None:
        return True
    return (date.year, date.month) > (last.year, last.month)


def run_backtest(
    headlines: Sequence[Headline],
    panel: PricePanel,
    config: BacktestConfig | None = None,
    universe: Universe | None = None,
) -> BacktestResult:
    """Replay the strategy over an archive and a price panel."""
    config = config or BacktestConfig()
    universe = universe or Universe.load()
    result = BacktestResult()

    headlines = sorted(headlines, key=lambda h: h.published_at)
    sessions = panel.sessions()
    if not headlines or not sessions:
        log.warning("nothing to replay: %d headlines, %d sessions", len(headlines), len(sessions))
        return result
    result.headlines_seen = len(headlines)

    # --- decide -------------------------------------------------------
    # Signals are computed up front so the daily loop only has to fill them.
    # Each entry is keyed by the session it is allowed to trade on.
    queued: dict[_dt.date, list[Signal]] = {}
    window = _dt.timedelta(minutes=config.window_minutes)
    times = _decision_times(
        headlines[0].published_at, headlines[-1].published_at + window, config.step_minutes
    )
    session_index = 0
    for decision_time in times:
        result.decisions += 1
        cutoff = decision_time - window
        visible = [h for h in headlines if cutoff <= h.published_at <= decision_time]
        if not visible:
            continue
        signals = [
            s
            for s in build_signals(extract_mentions(visible, universe), config.strategy)
            if s.tradable
        ]
        if not signals:
            continue
        # Advance a pointer rather than rescanning all sessions each time.
        while session_index < len(sessions):
            bell = _dt.datetime.combine(
                sessions[session_index], config.market_open_utc, tzinfo=_dt.timezone.utc
            )
            if bell > decision_time:
                break
            session_index += 1
        if session_index >= len(sessions):
            result.skipped["after last session"] += len(signals)
            continue
        result.signals_generated += len(signals)
        queued.setdefault(sessions[session_index], []).extend(signals)

    # --- simulate -----------------------------------------------------
    cash = config.starting_cash
    open_positions: list[Position] = []
    last_contribution: _dt.date | None = None
    daily_borrow = config.borrow_rate_annual / 365.0
    slip = config.slippage_bps / 10_000.0

    for session in sessions:
        if config.contribution and _contribution_due(
            session, last_contribution, config.contribution_day
        ):
            cash += config.contribution
            result.contributions[session] = (
                result.contributions.get(session, 0.0) + config.contribution
            )
            last_contribution = session

        # -- exits, before new entries compete for the same cash
        still_open: list[Position] = []
        for position in open_positions:
            bar = panel.bar_on(position.symbol, session)
            if bar is None:
                still_open.append(position)
                continue

            position.sessions_held += 1
            exit_price: float | None = None
            reason = ""

            if position.side is Side.BUY:
                stop = position.entry_price * (1 - config.stop_loss) if config.stop_loss else None
                target = (
                    position.entry_price * (1 + config.take_profit)
                    if config.take_profit
                    else None
                )
                # Stop checked first: the pessimistic ordering when a single
                # bar spans both levels and the intrabar path is unknown.
                if stop is not None and bar.low <= stop:
                    exit_price, reason = stop, "stop loss"
                elif target is not None and bar.high >= target:
                    exit_price, reason = target, "take profit"
            else:
                stop = position.entry_price * (1 + config.stop_loss) if config.stop_loss else None
                target = (
                    position.entry_price * (1 - config.take_profit)
                    if config.take_profit
                    else None
                )
                if stop is not None and bar.high >= stop:
                    exit_price, reason = stop, "stop loss"
                elif target is not None and bar.low <= target:
                    exit_price, reason = target, "take profit"

            if exit_price is None and position.sessions_held >= config.hold_days:
                exit_price, reason = bar.close, "time stop"

            if exit_price is None:
                if position.side is Side.SHORT:
                    fee = position.qty * bar.close * daily_borrow
                    position.borrow_paid += fee
                    cash -= fee
                    result.total_costs += fee
                still_open.append(position)
                continue

            if position.side is Side.BUY:
                proceeds = position.qty * exit_price * (1 - slip)
                cash += proceeds - config.commission_per_trade
                gross = position.qty * (exit_price - position.entry_reference)
            else:
                cost = position.qty * exit_price * (1 + slip)
                cash -= cost + config.commission_per_trade
                gross = position.qty * (position.entry_reference - exit_price)

            costs = (
                position.qty * position.entry_reference * slip
                + position.qty * exit_price * slip
                + 2 * config.commission_per_trade
                + position.borrow_paid
            )
            result.total_costs += (
                position.qty * exit_price * slip + config.commission_per_trade
            )
            result.trades.append(
                Trade(
                    symbol=position.symbol,
                    side=position.side,
                    qty=position.qty,
                    entry_date=position.entry_date,
                    entry_price=position.entry_price,
                    entry_reference=position.entry_reference,
                    exit_date=session,
                    exit_price=exit_price,
                    gross_pnl=gross,
                    costs=costs,
                    exit_reason=reason,
                    mentions=position.mentions,
                    sentiment=position.sentiment,
                )
            )
        open_positions = still_open

        # -- entries
        for signal in queued.get(session, []):
            bar = panel.bar_on(signal.ticker, session)
            if bar is None:
                result.skipped[f"no price data: {signal.ticker}"] += 1
                continue
            if any(p.symbol == signal.ticker for p in open_positions):
                result.skipped["already holding"] += 1
                continue

            notional = signal.notional
            if signal.side is Side.BUY:
                if notional > cash:
                    notional = cash
            else:
                if notional * config.short_margin > cash:
                    notional = cash / config.short_margin if config.short_margin else 0.0
            if notional <= 0:
                result.skipped["insufficient cash"] += 1
                continue

            fill = bar.open * (1 + slip) if signal.side is Side.BUY else bar.open * (1 - slip)
            qty = notional / fill
            if qty <= 0:
                result.skipped["insufficient cash"] += 1
                continue

            if signal.side is Side.BUY:
                cash -= qty * fill + config.commission_per_trade
            else:
                cash += qty * fill - config.commission_per_trade
            result.total_costs += qty * bar.open * slip + config.commission_per_trade

            open_positions.append(
                Position(
                    symbol=signal.ticker,
                    side=signal.side,
                    qty=qty,
                    entry_date=session,
                    entry_price=fill,
                    entry_reference=bar.open,
                    mentions=signal.mention_count,
                    sentiment=signal.sentiment_mean,
                )
            )

        # -- mark to market
        exposure = 0.0
        for position in open_positions:
            bar = panel.bar_on_or_before(position.symbol, session)
            if bar is None:
                continue
            signed = 1.0 if position.side is Side.BUY else -1.0
            exposure += signed * position.qty * bar.close
        result.equity.append((session, cash + exposure))

    result.open_at_end = open_positions
    log.info(
        "replayed %d decision(s): %d signals, %d trades, %d skipped",
        result.decisions,
        result.signals_generated,
        len(result.trades),
        sum(result.skipped.values()),
    )
    return result
