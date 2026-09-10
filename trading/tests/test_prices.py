"""Price loading, lookup semantics and frequency inference."""

from __future__ import annotations

import datetime as dt
import os
import tempfile
import unittest

from wsj_headline_trader.prices import (
    Bar,
    PriceDataError,
    PricePanel,
    infer_periods_per_year,
    load_price_panel,
    load_series,
    parse_date,
)

D = dt.date


def write(tmp: str, name: str, text: str) -> str:
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


class ParseDateTests(unittest.TestCase):
    def test_common_formats(self):
        for raw in ("2024-09-03", "2024/09/03", "20240903"):
            self.assertEqual(parse_date(raw), D(2024, 9, 3))

    def test_iso_timestamp(self):
        self.assertEqual(parse_date("2024-09-03T14:30:00Z"), D(2024, 9, 3))

    def test_nonsense_is_rejected(self):
        with self.assertRaises(PriceDataError):
            parse_date("not a date")


class LoadSeriesTests(unittest.TestCase):
    def test_detects_columns_and_sorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(tmp, "s.csv", "date,close\n2024-02-01,110\n2024-01-01,100\n")
            series = load_series(path)
        self.assertEqual(series, [(D(2024, 1, 1), 100.0), (D(2024, 2, 1), 110.0)])

    def test_prefers_adjusted_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(tmp, "s.csv", "date,close,adj_close\n2024-01-01,100,95\n")
            self.assertEqual(load_series(path)[0][1], 95.0)

    def test_explicit_columns_win(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(tmp, "s.csv", "d,px\n2024-01-01,42\n")
            self.assertEqual(load_series(path, "d", "px")[0][1], 42.0)

    def test_date_range_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(
                tmp, "s.csv", "date,close\n2024-01-01,1\n2024-06-01,2\n2024-12-01,3\n"
            )
            series = load_series(path, start=D(2024, 5, 1), end=D(2024, 7, 1))
        self.assertEqual(series, [(D(2024, 6, 1), 2.0)])

    def test_blank_and_nonpositive_rows_are_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(
                tmp, "s.csv", "date,close\n2024-01-01,\n2024-01-02,0\n2024-01-03,5\n"
            )
            self.assertEqual(load_series(path), [(D(2024, 1, 3), 5.0)])

    def test_missing_columns_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(tmp, "s.csv", "alpha,beta\n1,2\n")
            with self.assertRaises(PriceDataError):
                load_series(path)

    def test_no_rows_in_range_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(tmp, "s.csv", "date,close\n2024-01-01,1\n")
            with self.assertRaises(PriceDataError):
                load_series(path, start=D(2030, 1, 1))

    def test_the_vendored_sp500_file_loads(self):
        path = os.path.join(os.path.dirname(__file__), "..", "data", "sp500_monthly.csv")
        series = load_series(os.path.normpath(path))
        self.assertGreater(len(series), 400)
        self.assertEqual(series[0][0], D(1990, 1, 1))
        self.assertTrue(all(price > 0 for _, price in series))


class PanelTests(unittest.TestCase):
    def setUp(self):
        self.panel = PricePanel(
            {
                "AAA": [
                    Bar(D(2024, 1, 2), 10.0, 11.0, 9.0, 10.5),
                    Bar(D(2024, 1, 3), 10.5, 12.0, 10.0, 11.5),
                    Bar(D(2024, 1, 5), 11.5, 12.5, 11.0, 12.0),
                ],
                "BBB": [Bar(D(2024, 1, 3), 50.0, 51.0, 49.0, 50.5)],
            }
        )

    def test_symbols_and_membership(self):
        self.assertEqual(self.panel.symbols, ["AAA", "BBB"])
        self.assertIn("AAA", self.panel)
        self.assertNotIn("ZZZ", self.panel)

    def test_sessions_are_the_union_of_all_symbols(self):
        self.assertEqual(
            self.panel.sessions(), [D(2024, 1, 2), D(2024, 1, 3), D(2024, 1, 5)]
        )

    def test_bar_on_is_exact(self):
        self.assertEqual(self.panel.bar_on("AAA", D(2024, 1, 3)).close, 11.5)
        self.assertIsNone(self.panel.bar_on("AAA", D(2024, 1, 4)))
        self.assertIsNone(self.panel.bar_on("ZZZ", D(2024, 1, 3)))

    def test_next_bar_after_is_strict(self):
        self.assertEqual(self.panel.next_bar_after("AAA", D(2024, 1, 2)).date, D(2024, 1, 3))
        self.assertEqual(self.panel.next_bar_after("AAA", D(2024, 1, 4)).date, D(2024, 1, 5))
        self.assertIsNone(self.panel.next_bar_after("AAA", D(2024, 1, 5)))

    def test_bar_on_or_before_carries_the_last_mark(self):
        self.assertEqual(self.panel.bar_on_or_before("AAA", D(2024, 1, 4)).date, D(2024, 1, 3))
        self.assertIsNone(self.panel.bar_on_or_before("AAA", D(2023, 12, 31)))

    def test_bars_between_excludes_the_start(self):
        bars = self.panel.bars_between("AAA", D(2024, 1, 2), D(2024, 1, 5))
        self.assertEqual([b.date for b in bars], [D(2024, 1, 3), D(2024, 1, 5)])

    def test_unsorted_input_is_sorted(self):
        panel = PricePanel(
            {
                "AAA": [
                    Bar(D(2024, 1, 5), 1, 1, 1, 1),
                    Bar(D(2024, 1, 2), 1, 1, 1, 1),
                ]
            }
        )
        self.assertEqual([b.date for b in panel.bars["AAA"]], [D(2024, 1, 2), D(2024, 1, 5)])


class LoadPanelTests(unittest.TestCase):
    def test_long_format_with_ohlc(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(
                tmp,
                "p.csv",
                "date,symbol,open,high,low,close\n"
                "2024-01-02,nvda,10,11,9,10.5\n"
                "2024-01-02,BA,20,21,19,20.5\n",
            )
            panel = load_price_panel(path)
        self.assertEqual(panel.symbols, ["BA", "NVDA"])
        self.assertEqual(panel.bar_on("NVDA", D(2024, 1, 2)).high, 11.0)

    def test_close_only_falls_back_for_ohlc(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(tmp, "p.csv", "date,symbol,close\n2024-01-02,NVDA,10\n")
            bar = load_price_panel(path).bar_on("NVDA", D(2024, 1, 2))
        self.assertEqual((bar.open, bar.high, bar.low, bar.close), (10.0, 10.0, 10.0, 10.0))

    def test_missing_symbol_column_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(tmp, "p.csv", "date,close\n2024-01-02,10\n")
            with self.assertRaises(PriceDataError):
                load_price_panel(path)

    def test_bad_rows_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(
                tmp,
                "p.csv",
                "date,symbol,close\n2024-01-02,NVDA,oops\n2024-01-03,NVDA,10\n,,\n",
            )
            panel = load_price_panel(path)
        self.assertEqual(len(panel.bars["NVDA"]), 1)


class FrequencyTests(unittest.TestCase):
    def test_daily(self):
        dates = [D(2024, 1, 1) + dt.timedelta(days=i) for i in range(30)]
        self.assertEqual(infer_periods_per_year(dates), 252)

    def test_weekly(self):
        dates = [D(2024, 1, 1) + dt.timedelta(days=7 * i) for i in range(30)]
        self.assertEqual(infer_periods_per_year(dates), 52)

    def test_monthly(self):
        dates = [D(2024, 1, 1) + dt.timedelta(days=30 * i) for i in range(30)]
        self.assertEqual(infer_periods_per_year(dates), 12)

    def test_too_few_points_defaults_to_daily(self):
        self.assertEqual(infer_periods_per_year([D(2024, 1, 1)]), 252)

    def test_the_vendored_sp500_file_is_detected_as_monthly(self):
        path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "data", "sp500_monthly.csv")
        )
        series = load_series(path)
        self.assertEqual(infer_periods_per_year([d for d, _ in series]), 12)


if __name__ == "__main__":
    unittest.main()
