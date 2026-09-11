"""Performance maths, against hand-computable answers."""

from __future__ import annotations

import datetime as dt
import unittest

from wsj_headline_trader.metrics import (
    CashFlow,
    daily_returns,
    format_comparison,
    growth_index,
    irr,
    max_drawdown,
    net_present_value,
    sharpe,
    summarise,
    volatility,
)

D = dt.date


class NpvAndIrrTests(unittest.TestCase):
    def test_npv_at_zero_is_the_plain_sum(self):
        flows = [CashFlow(D(2024, 1, 1), -100.0), CashFlow(D(2025, 1, 1), 110.0)]
        self.assertAlmostEqual(net_present_value(flows, 0.0), 10.0, places=9)

    def test_irr_of_a_one_year_ten_percent_gain(self):
        # -100 today, +110 in exactly one year (365.25 days) is 10% a year.
        flows = [CashFlow(D(2024, 1, 1), -100.0), CashFlow(D(2024, 12, 31), 110.0)]
        rate = irr(flows)
        self.assertIsNotNone(rate)
        self.assertAlmostEqual(rate, 0.10, places=3)

    def test_irr_of_a_two_year_doubling(self):
        # 2x over two years is sqrt(2) - 1 a year.
        flows = [CashFlow(D(2024, 1, 1), -100.0), CashFlow(D(2025, 12, 31), 200.0)]
        self.assertAlmostEqual(irr(flows), 2 ** 0.5 - 1.0, places=3)

    def test_irr_zero_when_nothing_is_earned(self):
        flows = [CashFlow(D(2024, 1, 1), -100.0), CashFlow(D(2026, 1, 1), 100.0)]
        self.assertAlmostEqual(irr(flows), 0.0, places=6)

    def test_irr_is_negative_on_a_loss(self):
        flows = [CashFlow(D(2024, 1, 1), -100.0), CashFlow(D(2024, 12, 31), 90.0)]
        self.assertLess(irr(flows), 0)

    def test_irr_needs_both_signs(self):
        self.assertIsNone(irr([CashFlow(D(2024, 1, 1), -100.0)]))
        self.assertIsNone(
            irr([CashFlow(D(2024, 1, 1), -100.0), CashFlow(D(2025, 1, 1), -50.0)])
        )

    def test_irr_of_a_regular_savings_plan(self):
        # 12 monthly deposits of 100 that end at exactly 1200 earned nothing.
        flows = [CashFlow(D(2024, m, 1), -100.0) for m in range(1, 13)]
        flows.append(CashFlow(D(2024, 12, 1), 1200.0))
        self.assertAlmostEqual(irr(flows), 0.0, places=6)


class ReturnSeriesTests(unittest.TestCase):
    def test_daily_returns_ignore_same_day_contributions(self):
        # 1000 -> 2000 purely because 1000 was deposited: a 0% return.
        equity = [(D(2024, 1, 1), 1000.0), (D(2024, 1, 2), 2000.0)]
        returns = daily_returns(equity, {D(2024, 1, 2): 1000.0})
        self.assertAlmostEqual(returns[0][1], 0.0, places=9)

    def test_daily_returns_without_contributions(self):
        equity = [(D(2024, 1, 1), 100.0), (D(2024, 1, 2), 110.0)]
        self.assertAlmostEqual(daily_returns(equity)[0][1], 0.10, places=9)

    def test_growth_index_compounds(self):
        returns = [(D(2024, 1, 2), 0.10), (D(2024, 1, 3), 0.10)]
        self.assertAlmostEqual(growth_index(returns)[-1][1], 1.21, places=9)

    def test_max_drawdown_of_a_known_path(self):
        # 1.0 -> 1.5 -> 0.9: peak 1.5, trough 0.9, drawdown -40%.
        index = [(D(2024, 1, 1), 1.0), (D(2024, 1, 2), 1.5), (D(2024, 1, 3), 0.9)]
        self.assertAlmostEqual(max_drawdown(index), -0.4, places=9)

    def test_max_drawdown_is_zero_when_only_rising(self):
        index = [(D(2024, 1, i + 1), 1.0 + i) for i in range(4)]
        self.assertEqual(max_drawdown(index), 0.0)

    def test_volatility_of_a_constant_series_is_zero(self):
        returns = [(D(2024, 1, i + 1), 0.01) for i in range(10)]
        self.assertAlmostEqual(volatility(returns), 0.0, places=9)

    def test_volatility_annualises_by_the_period_count(self):
        returns = [(D(2024, 1, i + 1), 0.01 if i % 2 else -0.01) for i in range(20)]
        monthly = volatility(returns, periods=12)
        daily = volatility(returns, periods=252)
        self.assertAlmostEqual(daily / monthly, (252 / 12) ** 0.5, places=6)

    def test_sharpe_is_zero_without_variance(self):
        returns = [(D(2024, 1, i + 1), 0.01) for i in range(10)]
        self.assertEqual(sharpe(returns), 0.0)

    def test_sharpe_is_positive_for_a_profitable_noisy_series(self):
        returns = [(D(2024, 1, i + 1), 0.02 if i % 2 else 0.01) for i in range(20)]
        self.assertGreater(sharpe(returns), 0)


class SummariseTests(unittest.TestCase):
    def test_summary_of_a_simple_doubling(self):
        equity = [(D(2024, 1, 1), 1000.0), (D(2024, 12, 31), 2000.0)]
        summary = summarise("test", equity, {D(2024, 1, 1): 1000.0})
        self.assertEqual(summary.contributions, 1000.0)
        self.assertEqual(summary.final_value, 2000.0)
        self.assertEqual(summary.profit, 1000.0)
        self.assertAlmostEqual(summary.profit_pct_of_contributions, 1.0, places=9)
        self.assertAlmostEqual(summary.twr_total, 1.0, places=9)
        self.assertAlmostEqual(summary.irr_annual, 1.0, places=2)

    def test_contribution_is_not_counted_as_a_return(self):
        # Two 1000 deposits ending at 2000 earned nothing at all.
        equity = [(D(2024, 1, 1), 1000.0), (D(2024, 2, 1), 2000.0)]
        summary = summarise(
            "flat", equity, {D(2024, 1, 1): 1000.0, D(2024, 2, 1): 1000.0}
        )
        self.assertAlmostEqual(summary.twr_total, 0.0, places=9)
        self.assertEqual(summary.profit, 0.0)

    def test_empty_curve_is_rejected(self):
        with self.assertRaises(ValueError):
            summarise("empty", [], {})

    def test_comparison_table_lists_every_label(self):
        equity = [(D(2024, 1, 1), 1000.0), (D(2024, 6, 1), 1100.0)]
        a = summarise("Strategy", equity, {D(2024, 1, 1): 1000.0})
        b = summarise("Benchmark", equity, {D(2024, 1, 1): 1000.0})
        text = format_comparison(a, b)
        self.assertIn("Strategy", text)
        self.assertIn("Benchmark", text)
        self.assertIn("Max drawdown", text)

    def test_comparison_of_nothing_is_safe(self):
        self.assertIn("nothing to compare", format_comparison())


if __name__ == "__main__":
    unittest.main()
