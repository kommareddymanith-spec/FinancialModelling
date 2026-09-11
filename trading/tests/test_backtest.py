"""The walk-forward engine: no look-ahead, correct accounting, honest costs."""

from __future__ import annotations

import datetime as dt
import unittest

from wsj_headline_trader.backtest import (
    BacktestConfig,
    _contribution_due,
    _fill_session,
    run_backtest,
)
from wsj_headline_trader.models import Headline, Side
from wsj_headline_trader.prices import Bar, PricePanel
from wsj_headline_trader.strategy import StrategyConfig
from wsj_headline_trader.universe import Company, Universe

D = dt.date
UTC = dt.timezone.utc

UNIVERSE = Universe(
    [
        Company(ticker="NVDA", name="Nvidia", aliases=("Nvidia",)),
        Company(ticker="BA", name="Boeing", aliases=("Boeing",)),
    ]
)


def headline(title: str, when: dt.datetime) -> Headline:
    return Headline(
        title=title,
        summary="",
        link=f"https://example.test/{abs(hash((title, when)))}",
        published_at=when,
        source="test",
    )


def flat_panel(symbol: str, dates, price: float = 100.0) -> dict:
    return {symbol: [Bar(d, price, price, price, price) for d in dates]}


def sessions(count: int, first: D = D(2025, 1, 6)):
    """``count`` consecutive weekdays."""
    out, current = [], first
    while len(out) < count:
        if current.weekday() < 5:
            out.append(current)
        current += dt.timedelta(days=1)
    return out


class HelperTests(unittest.TestCase):
    def test_fill_session_is_the_next_open_after_the_decision(self):
        days = [D(2025, 1, 6), D(2025, 1, 7), D(2025, 1, 8)]
        # 15:00 UTC on the 6th is after that day's 13:30 bell -> next session.
        decided = dt.datetime(2025, 1, 6, 15, 0, tzinfo=UTC)
        self.assertEqual(_fill_session(decided, days, dt.time(13, 30)), D(2025, 1, 7))

    def test_an_overnight_decision_trades_the_same_morning(self):
        days = [D(2025, 1, 6), D(2025, 1, 7)]
        decided = dt.datetime(2025, 1, 6, 2, 0, tzinfo=UTC)
        self.assertEqual(_fill_session(decided, days, dt.time(13, 30)), D(2025, 1, 6))

    def test_no_session_left_returns_none(self):
        days = [D(2025, 1, 6)]
        decided = dt.datetime(2025, 1, 6, 20, 0, tzinfo=UTC)
        self.assertIsNone(_fill_session(decided, days, dt.time(13, 30)))

    def test_contribution_is_due_once_a_month(self):
        self.assertTrue(_contribution_due(D(2025, 1, 6), None, 1))
        self.assertFalse(_contribution_due(D(2025, 1, 20), D(2025, 1, 6), 1))
        self.assertTrue(_contribution_due(D(2025, 2, 3), D(2025, 1, 6), 1))

    def test_contribution_waits_for_the_scheduled_day(self):
        self.assertFalse(_contribution_due(D(2025, 1, 5), None, 15))
        self.assertTrue(_contribution_due(D(2025, 1, 16), None, 15))


class NoLookAheadTests(unittest.TestCase):
    def test_a_headline_during_the_session_fills_the_next_morning(self):
        days = sessions(5)
        panel = PricePanel(flat_panel("NVDA", days))
        # Published at 15:00 UTC on day 0, after that day's open.
        when = dt.datetime.combine(days[0], dt.time(15, 0), tzinfo=UTC)
        archive = [
            headline("Nvidia Shares Surge on Blowout Results", when),
            headline("Nvidia Soars to a Record High", when + dt.timedelta(minutes=5)),
        ]
        result = run_backtest(
            archive, panel, BacktestConfig(contribution=10_000.0), UNIVERSE
        )
        self.assertTrue(result.trades or result.open_at_end)
        entries = [p.entry_date for p in result.open_at_end] + [
            t.entry_date for t in result.trades
        ]
        self.assertTrue(all(entry > days[0] for entry in entries))

    def test_signals_after_the_last_session_are_recorded_not_traded(self):
        days = sessions(2)
        panel = PricePanel(flat_panel("NVDA", days))
        when = dt.datetime.combine(days[-1], dt.time(20, 0), tzinfo=UTC)
        archive = [
            headline("Nvidia Shares Surge on Blowout Results", when),
            headline("Nvidia Soars to a Record High", when + dt.timedelta(minutes=1)),
        ]
        result = run_backtest(archive, panel, BacktestConfig(), UNIVERSE)
        self.assertEqual(result.trades, [])
        self.assertGreater(result.skipped["after last session"], 0)


class AccountingTests(unittest.TestCase):
    """The load-bearing invariant: reported P&L is the real cash movement."""

    def build(self, prices, **config_kwargs):
        days = sessions(len(prices))
        panel = PricePanel(
            {"NVDA": [Bar(d, p, p * 1.02, p * 0.98, p) for d, p in zip(days, prices)]}
        )
        when = dt.datetime.combine(days[0], dt.time(2, 0), tzinfo=UTC)
        archive = [
            headline("Nvidia Shares Surge on Blowout Results", when),
            headline("Nvidia Soars to a Record High", when + dt.timedelta(minutes=1)),
        ]
        config = BacktestConfig(
            contribution=10_000.0, hold_days=2, strategy=StrategyConfig(top_n=1), **config_kwargs
        )
        return run_backtest(archive, panel, config, UNIVERSE)

    def test_equity_equals_contributions_plus_net_pnl(self):
        result = self.build([100.0, 110.0, 120.0, 120.0, 120.0])
        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.open_at_end, [])
        expected = sum(result.contributions.values()) + sum(t.net_pnl for t in result.trades)
        self.assertAlmostEqual(result.equity[-1][1], expected, places=6)

    def test_invariant_holds_with_costs_charged(self):
        result = self.build(
            [100.0, 110.0, 120.0, 120.0, 120.0],
            slippage_bps=25.0,
            commission_per_trade=1.5,
        )
        expected = sum(result.contributions.values()) + sum(t.net_pnl for t in result.trades)
        self.assertAlmostEqual(result.equity[-1][1], expected, places=6)

    def test_costs_reduce_net_pnl_below_gross(self):
        result = self.build(
            [100.0, 110.0, 120.0, 120.0, 120.0],
            slippage_bps=50.0,
            commission_per_trade=2.0,
        )
        trade = result.trades[0]
        self.assertLess(trade.net_pnl, trade.gross_pnl)
        self.assertGreater(trade.costs, 0)

    def test_a_profitable_long_makes_money(self):
        result = self.build([100.0, 110.0, 130.0, 130.0, 130.0], slippage_bps=0.0)
        trade = result.trades[0]
        self.assertEqual(trade.side, Side.BUY)
        self.assertGreater(trade.net_pnl, 0)
        self.assertGreater(result.equity[-1][1], sum(result.contributions.values()))

    def test_a_long_into_a_falling_market_loses_money(self):
        result = self.build([100.0, 90.0, 70.0, 70.0, 70.0], slippage_bps=0.0)
        self.assertLess(result.trades[0].net_pnl, 0)

    def test_free_run_pnl_is_exactly_the_price_move(self):
        # No costs: 10,000 at 100 held two sessions to a close of 120.
        result = self.build([100.0, 110.0, 120.0, 120.0, 120.0], slippage_bps=0.0)
        trade = result.trades[0]
        self.assertAlmostEqual(trade.entry_price, 100.0, places=9)
        self.assertAlmostEqual(trade.exit_price, 120.0, places=9)
        self.assertAlmostEqual(trade.qty * 20.0, trade.net_pnl, places=6)
        self.assertAlmostEqual(trade.return_pct, 0.20, places=9)


class ShortTests(unittest.TestCase):
    def build(self, prices, **config_kwargs):
        days = sessions(len(prices))
        panel = PricePanel(
            {"BA": [Bar(d, p, p * 1.02, p * 0.98, p) for d, p in zip(days, prices)]}
        )
        when = dt.datetime.combine(days[0], dt.time(2, 0), tzinfo=UTC)
        archive = [
            headline("Boeing Shares Plunge After a Fresh Recall", when),
            headline("Boeing Posts Record Loss on Fraud Claims", when + dt.timedelta(minutes=1)),
        ]
        config_kwargs.setdefault("contribution", 10_000.0)
        config = BacktestConfig(
            hold_days=2, strategy=StrategyConfig(top_n=1), **config_kwargs
        )
        return run_backtest(archive, panel, config, UNIVERSE)

    def test_negative_coverage_opens_a_short(self):
        result = self.build([100.0, 90.0, 80.0, 80.0, 80.0], slippage_bps=0.0, borrow_rate_annual=0.0)
        trade = result.trades[0]
        self.assertEqual(trade.side, Side.SHORT)
        self.assertGreater(trade.net_pnl, 0)

    def test_a_short_into_a_rising_market_loses_money(self):
        result = self.build([100.0, 110.0, 130.0, 130.0, 130.0], slippage_bps=0.0, borrow_rate_annual=0.0)
        self.assertLess(result.trades[0].net_pnl, 0)

    def test_borrow_cost_is_charged_while_short(self):
        free = self.build([100.0, 100.0, 100.0, 100.0], slippage_bps=0.0, borrow_rate_annual=0.0)
        charged = self.build([100.0, 100.0, 100.0, 100.0], slippage_bps=0.0, borrow_rate_annual=0.50)
        self.assertGreater(charged.total_costs, free.total_costs)
        self.assertLess(charged.trades[0].net_pnl, free.trades[0].net_pnl)

    def test_short_accounting_invariant(self):
        result = self.build(
            [100.0, 90.0, 80.0, 80.0, 80.0], slippage_bps=20.0, borrow_rate_annual=0.10
        )
        expected = sum(result.contributions.values()) + sum(t.net_pnl for t in result.trades)
        self.assertAlmostEqual(result.equity[-1][1], expected, places=6)

    def test_margin_requirement_limits_the_short(self):
        result = self.build(
            [100.0, 90.0, 80.0, 80.0, 80.0], contribution=500.0, short_margin=1.0
        )
        # The binding constraint is cash: proceeds raised by the short sale
        # cannot exceed the free cash backing it at the required margin.
        self.assertTrue(result.trades)
        for trade in result.trades:
            self.assertLessEqual(trade.qty * trade.entry_price, 500.0 + 1e-6)


class ExitTests(unittest.TestCase):
    def build(self, prices, **config_kwargs):
        days = sessions(len(prices))
        panel = PricePanel(
            {"NVDA": [Bar(d, o, h, l, c) for d, (o, h, l, c) in zip(days, prices)]}
        )
        when = dt.datetime.combine(days[0], dt.time(2, 0), tzinfo=UTC)
        archive = [
            headline("Nvidia Shares Surge on Blowout Results", when),
            headline("Nvidia Soars to a Record High", when + dt.timedelta(minutes=1)),
        ]
        config = BacktestConfig(
            contribution=10_000.0, strategy=StrategyConfig(top_n=1), slippage_bps=0.0,
            **config_kwargs,
        )
        return run_backtest(archive, panel, config, UNIVERSE)

    def test_time_stop_closes_after_hold_days(self):
        bars = [(100, 100, 100, 100)] * 6
        result = self.build(bars, hold_days=3)
        trade = result.trades[0]
        self.assertEqual(trade.exit_reason, "time stop")
        # hold_days counts sessions *after* entry, so a 3-day hold entered on
        # Monday exits at Thursday's close.
        self.assertEqual(trade.entry_date, D(2025, 1, 6))
        self.assertEqual(trade.exit_date, D(2025, 1, 9))

    def test_stop_loss_fires_on_the_low(self):
        bars = [(100, 100, 100, 100), (100, 101, 80, 85)] + [(85, 85, 85, 85)] * 4
        result = self.build(bars, hold_days=10, stop_loss=0.10)
        trade = result.trades[0]
        self.assertEqual(trade.exit_reason, "stop loss")
        self.assertAlmostEqual(trade.exit_price, 90.0, places=6)

    def test_take_profit_fires_on_the_high(self):
        bars = [(100, 100, 100, 100), (100, 130, 99, 125)] + [(125, 125, 125, 125)] * 4
        result = self.build(bars, hold_days=10, take_profit=0.20)
        trade = result.trades[0]
        self.assertEqual(trade.exit_reason, "take profit")
        self.assertAlmostEqual(trade.exit_price, 120.0, places=6)

    def test_stop_wins_when_one_bar_spans_both_levels(self):
        # Pessimistic ordering: the intrabar path is unknown.
        bars = [(100, 100, 100, 100), (100, 140, 70, 100)] + [(100, 100, 100, 100)] * 4
        result = self.build(bars, hold_days=10, stop_loss=0.10, take_profit=0.20)
        self.assertEqual(result.trades[0].exit_reason, "stop loss")

    def test_positions_still_open_at_the_end_are_reported(self):
        bars = [(100, 100, 100, 100)] * 3
        result = self.build(bars, hold_days=50)
        self.assertEqual(result.trades, [])
        self.assertEqual(len(result.open_at_end), 1)


class BookkeepingTests(unittest.TestCase):
    def test_contributions_arrive_monthly(self):
        days = [D(2025, 1, 6), D(2025, 2, 3), D(2025, 3, 3)]
        panel = PricePanel(flat_panel("NVDA", days))
        when = dt.datetime.combine(days[0], dt.time(2, 0), tzinfo=UTC)
        archive = [headline("Nvidia Shares Surge on Blowout Results", when)]
        result = run_backtest(archive, panel, BacktestConfig(contribution=1_000.0), UNIVERSE)
        self.assertEqual(sum(result.contributions.values()), 3_000.0)
        self.assertEqual(len(result.contributions), 3)

    def test_no_price_data_is_recorded_as_skipped(self):
        days = sessions(4)
        panel = PricePanel(flat_panel("BA", days))  # no NVDA prices at all
        when = dt.datetime.combine(days[0], dt.time(2, 0), tzinfo=UTC)
        archive = [
            headline("Nvidia Shares Surge on Blowout Results", when),
            headline("Nvidia Soars to a Record High", when + dt.timedelta(minutes=1)),
        ]
        result = run_backtest(archive, panel, BacktestConfig(), UNIVERSE)
        self.assertEqual(result.trades, [])
        self.assertGreater(result.skipped["no price data: NVDA"], 0)

    def test_no_cash_is_recorded_as_skipped(self):
        days = sessions(4)
        panel = PricePanel(flat_panel("NVDA", days))
        when = dt.datetime.combine(days[0], dt.time(2, 0), tzinfo=UTC)
        archive = [
            headline("Nvidia Shares Surge on Blowout Results", when),
            headline("Nvidia Soars to a Record High", when + dt.timedelta(minutes=1)),
        ]
        result = run_backtest(
            archive, panel, BacktestConfig(contribution=0.0, starting_cash=0.0), UNIVERSE
        )
        self.assertEqual(result.trades, [])
        self.assertGreater(result.skipped["insufficient cash"], 0)

    def test_an_existing_holding_is_not_doubled(self):
        days = sessions(8)
        panel = PricePanel(flat_panel("NVDA", days))
        archive = []
        for day in days[:4]:
            when = dt.datetime.combine(day, dt.time(2, 0), tzinfo=UTC)
            archive += [
                headline(f"Nvidia Shares Surge on Blowout Results {day}", when),
                headline(f"Nvidia Soars to a Record High {day}", when + dt.timedelta(minutes=1)),
            ]
        result = run_backtest(
            archive, panel, BacktestConfig(contribution=10_000.0, hold_days=10), UNIVERSE
        )
        self.assertGreater(result.skipped["already holding"], 0)
        self.assertEqual(len(result.open_at_end), 1)

    def test_empty_inputs_return_an_empty_result(self):
        empty = run_backtest([], PricePanel({}), BacktestConfig(), UNIVERSE)
        self.assertEqual(empty.equity, [])
        self.assertEqual(empty.trades, [])

    def test_headlines_with_no_companies_trade_nothing(self):
        days = sessions(4)
        panel = PricePanel(flat_panel("NVDA", days))
        when = dt.datetime.combine(days[0], dt.time(2, 0), tzinfo=UTC)
        result = run_backtest(
            [headline("Treasury Yields Rise Ahead of Inflation Data", when)],
            panel,
            BacktestConfig(),
            UNIVERSE,
        )
        self.assertEqual(result.trades, [])
        self.assertEqual(result.signals_generated, 0)

    def test_summary_reports_trade_statistics(self):
        days = sessions(6)
        panel = PricePanel(
            {"NVDA": [Bar(d, p, p, p, p) for d, p in zip(days, [100, 110, 120, 120, 120, 120])]}
        )
        when = dt.datetime.combine(days[0], dt.time(2, 0), tzinfo=UTC)
        archive = [
            headline("Nvidia Shares Surge on Blowout Results", when),
            headline("Nvidia Soars to a Record High", when + dt.timedelta(minutes=1)),
        ]
        result = run_backtest(
            archive, panel, BacktestConfig(contribution=10_000.0, hold_days=2), UNIVERSE
        )
        summary = result.summary("strategy")
        self.assertEqual(summary.extras["Trades"], 1)
        self.assertEqual(result.win_rate, 1.0)

    def test_config_validation(self):
        for kwargs in (
            {"window_minutes": 0},
            {"hold_days": 0},
            {"contribution": -1},
            {"stop_loss": 1.5},
            {"slippage_bps": -1},
            {"contribution_day": 31},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    BacktestConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
