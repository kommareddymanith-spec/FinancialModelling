"""End-to-end runs against the fixture feeds, plus the CLI."""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from wsj_headline_trader import (
    AlgorithmConfig,
    PaperBroker,
    StaticPriceProvider,
    StrategyConfig,
    WSJHeadlineAlgorithm,
    format_report,
)
from wsj_headline_trader.cli import main
from wsj_headline_trader.feed import FeedError, headlines_from_files

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
NOW = dt.datetime(2026, 9, 10, 15, 30, tzinfo=dt.timezone.utc)

MARKETS = os.path.join(FIXTURES, "wsj_markets.xml")
BUSINESS = os.path.join(FIXTURES, "wsj_business.xml")
ATOM = os.path.join(FIXTURES, "wsj_atom.xml")


def algorithm(*fixtures: str, **config_kwargs) -> WSJHeadlineAlgorithm:
    fixtures = fixtures or (MARKETS, BUSINESS)
    strategy = config_kwargs.pop("strategy", StrategyConfig())
    config = AlgorithmConfig(
        feeds=[f"fixture://{i}" for i in range(len(fixtures))],
        strategy=strategy,
        **config_kwargs,
    )
    return WSJHeadlineAlgorithm(
        config=config,
        broker=PaperBroker(prices=StaticPriceProvider(default=100.0)),
        fetcher=headlines_from_files(fixtures),
    )


class DryRunTests(unittest.TestCase):
    def setUp(self):
        self.report = algorithm().run(now=NOW)

    def test_reads_only_the_last_hour(self):
        # 15 items across both fixtures, one of which is 92 minutes old.
        self.assertEqual(self.report.headlines_scanned, 14)

    def test_ranks_the_most_featured_companies_first(self):
        # Nvidia and Boeing are both in three articles; the tie breaks on
        # conviction, and Nvidia's coverage is more strongly worded.
        self.assertEqual([s.ticker for s in self.report.signals][:3], ["NVDA", "BA", "TGT"])
        self.assertEqual(self.report.signals[0].mention_count, 3)
        self.assertEqual(self.report.signals[1].mention_count, 3)
        self.assertEqual(self.report.signals[2].mention_count, 2)

    def test_positive_coverage_becomes_a_buy(self):
        nvda = next(s for s in self.report.signals if s.ticker == "NVDA")
        self.assertEqual(nvda.side.value, "buy")
        self.assertGreater(nvda.sentiment_mean, 0)

    def test_negative_coverage_becomes_a_short(self):
        for ticker in ("BA", "TGT"):
            signal = next(s for s in self.report.signals if s.ticker == ticker)
            self.assertEqual(signal.side.value, "short", ticker)
            self.assertLess(signal.sentiment_mean, 0)

    def test_macro_headlines_do_not_produce_signals(self):
        # "Investors Target Small-Cap Stocks" and the Treasury yields story.
        self.assertNotIn("Treasury", " ".join(s.company for s in self.report.signals))

    def test_dry_run_submits_nothing(self):
        self.assertTrue(self.report.dry_run)
        self.assertTrue(all(not r.accepted for r in self.report.results))
        self.assertTrue(all("dry run" in r.message for r in self.report.results))

    def test_report_serialises_to_json(self):
        payload = json.dumps(self.report.to_dict())
        self.assertIn("BA", payload)
        self.assertIn("agreement", payload)

    def test_report_formats_for_the_console(self):
        text = format_report(self.report)
        self.assertIn("DRY RUN", text)
        self.assertIn("Most-featured companies", text)
        self.assertIn("BA", text)


class LiveSubmitTests(unittest.TestCase):
    def test_orders_reach_the_broker(self):
        algo = algorithm(dry_run=False)
        report = algo.run(now=NOW)
        accepted = [r for r in report.results if r.accepted]
        self.assertTrue(accepted)
        self.assertEqual(len(accepted), len(algo.broker.orders))

    def test_positions_match_the_signal_directions(self):
        algo = algorithm(dry_run=False)
        report = algo.run(now=NOW)
        for signal in report.signals:
            if not signal.tradable:
                continue
            position = algo.broker.positions[signal.ticker]
            if signal.side.value == "buy":
                self.assertGreater(position, 0, signal.ticker)
            else:
                self.assertLess(position, 0, signal.ticker)

    def test_orders_carry_their_rationale(self):
        algo = algorithm(dry_run=False)
        algo.run(now=NOW)
        order = algo.broker.orders[0]
        self.assertEqual(order.metadata["strategy"], "wsj-headline-frequency-sentiment")
        self.assertIn("headlines", order.metadata)
        self.assertTrue(order.client_order_id.startswith("wsj-"))

    def test_total_notional_respects_the_cap(self):
        algo = algorithm(dry_run=False, strategy=StrategyConfig(max_total_notional=1500.0))
        report = algo.run(now=NOW)
        self.assertLessEqual(sum(s.notional for s in report.signals), 1500.0)


class AtomFeedTests(unittest.TestCase):
    def test_atom_feed_produces_signals(self):
        report = algorithm(ATOM, strategy=StrategyConfig(min_mentions=2)).run(now=NOW)
        self.assertEqual(report.headlines_scanned, 2)
        tesla = next(s for s in report.signals if s.ticker == "TSLA")
        self.assertEqual(tesla.side.value, "short")


class ResilienceTests(unittest.TestCase):
    def test_no_headlines_means_no_orders(self):
        report = algorithm().run(now=NOW + dt.timedelta(days=2))
        self.assertEqual(report.headlines_scanned, 0)
        self.assertEqual(report.signals, [])
        self.assertIn("nothing to trade", format_report(report))

    def test_a_single_failing_feed_does_not_stop_the_run(self):
        def fetcher(url):
            if url.endswith("0"):
                raise FeedError("feed down")
            with open(BUSINESS, "rb") as handle:
                return handle.read()

        algo = WSJHeadlineAlgorithm(
            config=AlgorithmConfig(feeds=["fixture://0", "fixture://1"]),
            broker=PaperBroker(),
            fetcher=fetcher,
        )
        report = algo.run(now=NOW)
        self.assertEqual(len(report.feed_errors), 1)
        self.assertTrue(report.signals)

    def test_all_feeds_failing_trades_nothing(self):
        def fetcher(url):
            raise FeedError("everything is down")

        algo = WSJHeadlineAlgorithm(
            config=AlgorithmConfig(feeds=["a", "b"], dry_run=False),
            broker=PaperBroker(),
            fetcher=fetcher,
        )
        report = algo.run(now=NOW)
        self.assertEqual(report.signals, [])
        self.assertEqual(report.results, [])
        self.assertEqual(len(report.feed_errors), 2)

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            AlgorithmConfig(window_minutes=0)
        with self.assertRaises(ValueError):
            AlgorithmConfig(feeds=[])

    def test_naive_as_of_is_accepted(self):
        report = algorithm().run(now=NOW.replace(tzinfo=None))
        self.assertGreater(report.headlines_scanned, 0)


class CliTests(unittest.TestCase):
    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_dry_run_against_fixtures(self):
        code, out, _ = self.run_cli(
            "--fixture", MARKETS, "--fixture", BUSINESS, "--as-of", NOW.isoformat()
        )
        self.assertEqual(code, 0)
        self.assertIn("DRY RUN", out)
        self.assertIn("SHORT", out)
        self.assertIn("BUY", out)

    def test_json_output(self):
        code, out, _ = self.run_cli(
            "--fixture", MARKETS, "--fixture", BUSINESS, "--as-of", NOW.isoformat(), "--json"
        )
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["window_minutes"], 60)
        self.assertTrue(payload["dry_run"])
        self.assertTrue(payload["signals"])

    def test_live_flag_submits_to_the_paper_broker(self):
        code, out, _ = self.run_cli(
            "--fixture", MARKETS, "--as-of", NOW.isoformat(), "--live", "--json"
        )
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertFalse(payload["dry_run"])
        self.assertTrue(any(o["accepted"] for o in payload["orders"]))

    def test_strategy_flags_are_applied(self):
        code, out, _ = self.run_cli(
            "--fixture", MARKETS, "--as-of", NOW.isoformat(), "--top", "1", "--json"
        )
        payload = json.loads(out)
        self.assertEqual(code, 0)
        self.assertEqual(len(payload["signals"]), 1)

    def test_invalid_as_of_is_a_usage_error(self):
        code, _, err = self.run_cli("--as-of", "not-a-date")
        self.assertEqual(code, 2)
        self.assertIn("ISO 8601", err)

    def test_invalid_strategy_values_are_a_usage_error(self):
        code, _, err = self.run_cli("--top", "0")
        self.assertEqual(code, 2)
        self.assertIn("top_n", err)

    def test_missing_universe_file_is_a_usage_error(self):
        code, _, err = self.run_cli("--universe", "/nonexistent/universe.json")
        self.assertEqual(code, 2)
        self.assertIn("universe", err)

    def test_alpaca_without_credentials_is_a_usage_error(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            code, _, err = self.run_cli("--broker", "alpaca")
        self.assertEqual(code, 2)
        self.assertIn("APCA_API_KEY_ID", err)

    def test_real_money_warns_on_stderr(self):
        with mock.patch.dict(
            "os.environ",
            {"APCA_API_KEY_ID": "k", "APCA_API_SECRET_KEY": "s"},
            clear=True,
        ):
            code, _, err = self.run_cli(
                "--broker", "alpaca", "--real-money", "--live",
                "--fixture", MARKETS, "--as-of", NOW.isoformat(),
            )
        self.assertIn("REAL orders", err)

    def test_help_lists_the_main_flags(self):
        out = io.StringIO()
        with redirect_stdout(out):
            with self.assertRaises(SystemExit):
                main(["--help"])
        text = out.getvalue()
        for flag in ("--window-minutes", "--top", "--broker", "--live", "--fixture"):
            self.assertIn(flag, text)


if __name__ == "__main__":
    unittest.main()
