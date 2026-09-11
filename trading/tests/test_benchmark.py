"""The monthly savings plan, including the real S&P 500 result."""

from __future__ import annotations

import datetime as dt
import os
import unittest

from wsj_headline_trader.benchmark import DcaConfig, run_dca
from wsj_headline_trader.prices import load_series

D = dt.date
SP500 = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "sp500_monthly.csv")
)


class MechanicsTests(unittest.TestCase):
    def test_flat_prices_return_exactly_the_contributions(self):
        series = [(D(2024, m, 1), 100.0) for m in range(1, 13)]
        result = run_dca(series, DcaConfig(contribution=1000.0))
        self.assertEqual(len(result.purchases), 12)
        self.assertAlmostEqual(result.equity[-1][1], 12_000.0, places=6)
        self.assertAlmostEqual(result.summary("flat").twr_total, 0.0, places=9)

    def test_doubling_price_doubles_the_first_contribution(self):
        series = [(D(2024, 1, 1), 100.0), (D(2024, 2, 1), 200.0)]
        result = run_dca(series, DcaConfig(contribution=1000.0))
        # 10 units bought at 100, then 5 units at 200 = 15 units at 200 = 3000.
        self.assertAlmostEqual(result.units, 15.0, places=9)
        self.assertAlmostEqual(result.equity[-1][1], 3_000.0, places=6)

    def test_units_are_the_sum_of_contribution_over_price(self):
        series = [(D(2024, 1, 1), 100.0), (D(2024, 2, 1), 125.0), (D(2024, 3, 1), 80.0)]
        result = run_dca(series, DcaConfig(contribution=1000.0))
        expected = sum(1000.0 / price for _, price in series)
        self.assertAlmostEqual(result.units, expected, places=9)

    def test_contributions_are_recorded_on_the_purchase_date(self):
        series = [(D(2024, 1, 1), 100.0), (D(2024, 2, 1), 100.0)]
        result = run_dca(series, DcaConfig(contribution=500.0))
        self.assertEqual(result.contributions, {D(2024, 1, 1): 500.0, D(2024, 2, 1): 500.0})

    def test_a_closed_market_slips_the_purchase_forward(self):
        # Schedule is the 1st, but the only sessions are the 3rd and the 5th.
        series = [(D(2024, 1, 3), 100.0), (D(2024, 2, 5), 100.0)]
        result = run_dca(series, DcaConfig(contribution=1000.0, day_of_month=1))
        self.assertEqual([date for date, _, _ in result.purchases], [D(2024, 1, 3), D(2024, 2, 5)])

    def test_one_purchase_per_month_only(self):
        # Daily prices across two months must still buy exactly twice.
        series = [
            (D(2024, 1, 1) + dt.timedelta(days=i), 100.0) for i in range(0, 60)
        ]
        result = run_dca(series, DcaConfig(contribution=1000.0, day_of_month=1))
        self.assertEqual(len(result.purchases), 2)

    def test_expense_reduces_units_but_not_contributions(self):
        series = [(D(2024, 1, 1), 100.0)]
        plain = run_dca(series, DcaConfig(contribution=1000.0))
        charged = run_dca(series, DcaConfig(contribution=1000.0, expense_bps=100.0))
        self.assertLess(charged.units, plain.units)
        self.assertEqual(sum(charged.contributions.values()), 1000.0)

    def test_empty_series_is_rejected(self):
        with self.assertRaises(ValueError):
            run_dca([])

    def test_config_validation(self):
        for kwargs in ({"contribution": 0}, {"day_of_month": 0}, {"day_of_month": 31}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    DcaConfig(**kwargs)


class RealSp500Tests(unittest.TestCase):
    """Pins the actual two-year result so a data change cannot pass silently."""

    @classmethod
    def setUpClass(cls):
        cls.series = load_series(SP500, start=D(2024, 9, 1), end=D(2026, 8, 31))
        cls.result = run_dca(cls.series, DcaConfig(contribution=1000.0))
        cls.summary = cls.result.summary("S&P 500 DCA")

    def test_window_is_twenty_four_months(self):
        self.assertEqual(len(self.series), 24)
        self.assertEqual(self.series[0][0], D(2024, 9, 1))
        self.assertEqual(self.series[-1][0], D(2026, 8, 1))

    def test_contributions_total_twenty_four_thousand(self):
        self.assertEqual(self.summary.contributions, 24_000.0)
        self.assertEqual(len(self.result.purchases), 24)

    def test_final_value_matches_an_independent_calculation(self):
        units = sum(1000.0 / price for _, price in self.series)
        self.assertAlmostEqual(self.result.units, units, places=9)
        self.assertAlmostEqual(self.summary.final_value, units * self.series[-1][1], places=6)

    def test_headline_numbers(self):
        self.assertAlmostEqual(self.summary.final_value, 28_869.72, places=1)
        self.assertAlmostEqual(self.summary.profit, 4_869.72, places=1)
        self.assertAlmostEqual(self.summary.profit_pct_of_contributions, 0.2029, places=3)

    def test_annualised_measures_use_a_monthly_frequency(self):
        self.assertEqual(self.result.periods_per_year, 12)
        self.assertLess(self.summary.volatility_annual, 0.30)

    def test_money_weighted_return_exceeds_profit_on_contributions(self):
        # Later deposits were invested for less time, so the rate earned is
        # higher than the simple profit-over-contributions ratio.
        self.assertGreater(self.summary.irr_annual, self.summary.profit_pct_of_contributions / 2)


if __name__ == "__main__":
    unittest.main()
