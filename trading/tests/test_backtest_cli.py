"""Backtest CLI wiring.

The strategy runs here are against *synthetic* headlines and prices. They
verify that the machinery works end to end; they say nothing whatsoever about
how the strategy would have performed on real WSJ history.
"""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from wsj_headline_trader.backtest_cli import main

D = dt.date
SP500 = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "sp500_monthly.csv")
)

BULLISH = [
    "{co} Shares Surge on Blowout Results",
    "{co} Soars to a Record High",
    "Analysts Upgrade {co} on Strong Demand",
]
BEARISH = [
    "{co} Shares Plunge After a Fresh Recall",
    "{co} Warns of a Wider Loss",
    "{co} Faces a Class Action Over Disclosures",
]


def build_inputs(tmp: str, months: int = 6) -> tuple[str, str]:
    """A synthetic archive and matching price panel, purely for wiring checks."""
    archive_path = os.path.join(tmp, "archive.jsonl")
    prices_path = os.path.join(tmp, "prices.csv")

    day = D(2025, 1, 2)
    archive_lines: list[str] = []
    price_rows = ["date,symbol,open,high,low,close"]
    nvda, ba = 100.0, 200.0

    session = 0
    while session < months * 21:
        if day.weekday() < 5:
            session += 1
            # Nvidia drifts up, Boeing drifts down, so the synthetic signals
            # have something consistent to be right or wrong about.
            nvda *= 1.002
            ba *= 0.998
            for symbol, price in (("NVDA", nvda), ("BA", ba)):
                price_rows.append(
                    f"{day.isoformat()},{symbol},{price:.4f},{price * 1.01:.4f},"
                    f"{price * 0.99:.4f},{price:.4f}"
                )
            if session % 5 == 0:
                stamp = dt.datetime.combine(day, dt.time(2, 0), tzinfo=dt.timezone.utc)
                for offset, template in enumerate(BULLISH):
                    archive_lines.append(json.dumps({
                        "published_at": (stamp + dt.timedelta(minutes=offset)).isoformat(),
                        "title": template.format(co="Nvidia"),
                        "link": f"syn://nvda/{session}/{offset}",
                    }))
                for offset, template in enumerate(BEARISH):
                    archive_lines.append(json.dumps({
                        "published_at": (stamp + dt.timedelta(minutes=offset)).isoformat(),
                        "title": template.format(co="Boeing"),
                        "link": f"syn://ba/{session}/{offset}",
                    }))
        day += dt.timedelta(days=1)

    with open(archive_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(archive_lines))
    with open(prices_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(price_rows))
    return archive_path, prices_path


def run_cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


class BenchmarkOnlyTests(unittest.TestCase):
    def test_reports_the_real_index_result(self):
        code, out, _ = run_cli(
            "--benchmark-only", "--index", SP500, "--start", "2024-09-01", "--end", "2026-08-31"
        )
        self.assertEqual(code, 0)
        self.assertIn("24,000.00", out)
        self.assertIn("28,869.72", out)
        self.assertIn("no strategy result", out)

    def test_json_output_is_machine_readable(self):
        code, out, _ = run_cli(
            "--benchmark-only", "--index", SP500,
            "--start", "2024-09-01", "--end", "2026-08-31", "--json",
        )
        self.assertEqual(code, 0)
        payload = json.loads(out)
        summary = payload["summaries"][0]
        self.assertAlmostEqual(summary["contributions"], 24_000.0, places=2)
        self.assertAlmostEqual(summary["final_value"], 28_869.72, places=1)
        self.assertNotIn("strategy_diagnostics", payload)

    def test_contribution_size_scales_linearly(self):
        _, base, _ = run_cli(
            "--benchmark-only", "--index", SP500, "--start", "2024-09-01",
            "--end", "2026-08-31", "--json",
        )
        _, doubled, _ = run_cli(
            "--benchmark-only", "--index", SP500, "--start", "2024-09-01",
            "--end", "2026-08-31", "--contribution", "2000", "--json",
        )
        one = json.loads(base)["summaries"][0]
        two = json.loads(doubled)["summaries"][0]
        self.assertAlmostEqual(two["final_value"], one["final_value"] * 2, places=1)


class UsageErrorTests(unittest.TestCase):
    def test_no_archive_explains_why_and_exits_two(self):
        code, _, err = run_cli("--index", SP500)
        self.assertEqual(code, 2)
        self.assertIn("no headline archive", err)
        self.assertIn("no history", err)

    def test_archive_without_prices_is_an_error(self):
        code, _, err = run_cli("--archive", "somewhere.jsonl", "--index", SP500)
        self.assertEqual(code, 2)
        self.assertIn("--prices", err)

    def test_missing_archive_path_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, prices = build_inputs(tmp, months=1)
            code, _, err = run_cli(
                "--archive", os.path.join(tmp, "nope.jsonl"), "--prices", prices,
                "--index", SP500,
            )
        self.assertEqual(code, 2)
        self.assertIn("does not exist", err)

    def test_bad_date_is_an_error(self):
        with self.assertRaises(SystemExit):
            run_cli("--benchmark-only", "--start", "01/09/2024")

    def test_invalid_strategy_value_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive, prices = build_inputs(tmp, months=1)
            code, _, err = run_cli(
                "--archive", archive, "--prices", prices, "--index", SP500, "--top", "0"
            )
        self.assertEqual(code, 2)
        self.assertIn("top_n", err)


class StrategyWiringTests(unittest.TestCase):
    """End-to-end on synthetic data: mechanism only, not a performance claim."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.archive, cls.prices = build_inputs(cls.tmp.name, months=6)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_both_legs_are_reported(self):
        code, out, _ = run_cli(
            "--archive", self.archive, "--prices", self.prices, "--index", SP500,
            "--start", "2025-01-01", "--end", "2025-06-30",
        )
        self.assertEqual(code, 0)
        self.assertIn("WSJ headline strategy", out)
        self.assertIn("Index DCA", out)
        self.assertIn("closed trades", out)

    def test_the_replay_actually_trades(self):
        code, out, _ = run_cli(
            "--archive", self.archive, "--prices", self.prices, "--index", SP500,
            "--start", "2025-01-01", "--end", "2025-06-30", "--json",
        )
        self.assertEqual(code, 0)
        payload = json.loads(out)
        diagnostics = payload["strategy_diagnostics"]
        self.assertGreater(diagnostics["signals"], 0)
        self.assertGreater(diagnostics["trades"], 0)
        self.assertEqual(len(payload["summaries"]), 2)

    def test_both_legs_receive_the_same_contributions(self):
        _, out, _ = run_cli(
            "--archive", self.archive, "--prices", self.prices, "--index", SP500,
            "--start", "2025-01-01", "--end", "2025-06-30", "--json",
        )
        strategy, benchmark = json.loads(out)["summaries"]
        self.assertEqual(strategy["contributions"], benchmark["contributions"])

    def test_costs_make_the_strategy_worse(self):
        def final_value(*extra):
            _, out, _ = run_cli(
                "--archive", self.archive, "--prices", self.prices, "--index", SP500,
                "--start", "2025-01-01", "--end", "2025-06-30", "--json", *extra,
            )
            return json.loads(out)["summaries"][0]["final_value"]

        free = final_value("--slippage-bps", "0", "--borrow-rate", "0")
        costly = final_value("--slippage-bps", "50", "--borrow-rate", "0.10",
                             "--commission", "1.0")
        self.assertLess(costly, free)

    def test_trade_listing_renders(self):
        code, out, _ = run_cli(
            "--archive", self.archive, "--prices", self.prices, "--index", SP500,
            "--start", "2025-01-01", "--end", "2025-06-30", "--trades",
        )
        self.assertEqual(code, 0)
        self.assertIn("Closed trades", out)
        self.assertIn("time stop", out)

    def test_hold_days_changes_the_trade_count(self):
        def trades(hold):
            _, out, _ = run_cli(
                "--archive", self.archive, "--prices", self.prices, "--index", SP500,
                "--start", "2025-01-01", "--end", "2025-06-30", "--json",
                "--hold-days", str(hold),
            )
            return json.loads(out)["strategy_diagnostics"]["trades"]

        self.assertGreater(trades(2), trades(20))


if __name__ == "__main__":
    unittest.main()
