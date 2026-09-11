"""Alpaca account state, market-hours and position guards.

The HTTP layer is mocked throughout: these tests check that the right calls
are made and the right decisions follow, not that Alpaca is reachable.
"""

from __future__ import annotations

import datetime as dt
import io
import os
import unittest
from contextlib import redirect_stdout
from unittest import mock

from wsj_headline_trader.algorithm import (
    AlgorithmConfig,
    WSJHeadlineAlgorithm,
    format_report,
)
from wsj_headline_trader.broker import (
    AlpacaBroker,
    BrokerError,
    PaperBroker,
    StaticPriceProvider,
)
from wsj_headline_trader.cli import main
from wsj_headline_trader.feed import headlines_from_files

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
MARKETS = os.path.join(FIXTURES, "wsj_markets.xml")
NOW = dt.datetime(2026, 9, 10, 15, 30, tzinfo=dt.timezone.utc)


def algorithm(broker, **config_kwargs) -> WSJHeadlineAlgorithm:
    config = AlgorithmConfig(feeds=["fixture://0"], **config_kwargs)
    return WSJHeadlineAlgorithm(
        config=config, broker=broker, fetcher=headlines_from_files([MARKETS])
    )


class AccountStateTests(unittest.TestCase):
    def setUp(self):
        self.broker = AlpacaBroker(key_id="k", secret_key="s")

    def test_account_hits_the_account_endpoint(self):
        with mock.patch.object(self.broker, "_request", return_value={"status": "ACTIVE"}) as req:
            self.assertEqual(self.broker.account()["status"], "ACTIVE")
        self.assertTrue(req.call_args[0][0].endswith("/v2/account"))

    def test_clock_reports_open(self):
        with mock.patch.object(self.broker, "_request", return_value={"is_open": True}):
            self.assertTrue(self.broker.is_market_open())

    def test_clock_reports_closed(self):
        with mock.patch.object(self.broker, "_request", return_value={"is_open": False}):
            self.assertFalse(self.broker.is_market_open())

    def test_an_unreachable_clock_is_unknown_not_closed(self):
        with mock.patch.object(self.broker, "_request", side_effect=BrokerError("down")):
            self.assertIsNone(self.broker.is_market_open())

    def test_open_symbols_reads_the_positions_array(self):
        body = [{"symbol": "nvda", "qty": "10"}, {"symbol": "BA", "qty": "-4"}]
        with mock.patch.object(self.broker, "_request", return_value=body) as req:
            self.assertEqual(self.broker.open_symbols(), {"NVDA", "BA"})
        self.assertTrue(req.call_args[0][0].endswith("/v2/positions"))

    def test_open_symbols_is_empty_when_unreadable(self):
        with mock.patch.object(self.broker, "_request", side_effect=BrokerError("down")):
            self.assertEqual(self.broker.open_symbols(), set())

    def test_no_positions_is_empty(self):
        with mock.patch.object(self.broker, "_request", return_value=[]):
            self.assertEqual(self.broker.open_symbols(), set())


class MarketHoursGuardTests(unittest.TestCase):
    def test_a_closed_market_submits_nothing(self):
        broker = mock.Mock(spec=["submit", "is_market_open", "open_symbols"])
        broker.is_market_open.return_value = False
        broker.open_symbols.return_value = set()

        report = algorithm(broker, dry_run=False, require_market_open=True).run(now=NOW)

        broker.submit.assert_not_called()
        self.assertIn("market closed", " ".join(report.notes))
        self.assertTrue(all(not s.tradable for s in report.signals))
        self.assertIn("market closed", format_report(report))

    def test_an_open_market_submits(self):
        broker = mock.Mock(spec=["submit", "is_market_open", "open_symbols"])
        broker.is_market_open.return_value = True
        broker.open_symbols.return_value = set()

        algorithm(broker, dry_run=False, require_market_open=True).run(now=NOW)
        self.assertTrue(broker.submit.called)

    def test_an_unknown_market_state_submits_nothing(self):
        # Unknown must not be read as permission to trade.
        broker = mock.Mock(spec=["submit", "is_market_open", "open_symbols"])
        broker.is_market_open.return_value = None
        broker.open_symbols.return_value = set()

        report = algorithm(broker, dry_run=False, require_market_open=True).run(now=NOW)
        broker.submit.assert_not_called()
        self.assertIn("unknown", " ".join(report.notes))

    def test_a_broker_with_no_clock_is_treated_as_unknown(self):
        broker = mock.Mock(spec=["submit"])
        report = algorithm(broker, dry_run=False, require_market_open=True).run(now=NOW)
        broker.submit.assert_not_called()
        self.assertIn("unknown", " ".join(report.notes))

    def test_the_guard_is_off_by_default(self):
        broker = PaperBroker(prices=StaticPriceProvider(default=100.0))
        report = algorithm(broker, dry_run=False).run(now=NOW)
        self.assertTrue([r for r in report.results if r.accepted])
        self.assertEqual(report.notes, [])

    def test_a_dry_run_never_consults_the_clock(self):
        broker = mock.Mock(spec=["submit", "is_market_open", "open_symbols"])
        algorithm(broker, dry_run=True, require_market_open=True).run(now=NOW)
        broker.is_market_open.assert_not_called()


class HeldPositionGuardTests(unittest.TestCase):
    def test_a_held_symbol_is_skipped(self):
        broker = mock.Mock(spec=["submit", "open_symbols"])
        broker.open_symbols.return_value = {"NVDA"}

        report = algorithm(broker, dry_run=False).run(now=NOW)

        nvda = next(s for s in report.signals if s.ticker == "NVDA")
        self.assertEqual(nvda.skip_reason, "already holding")
        submitted = {call.args[0].symbol for call in broker.submit.call_args_list}
        self.assertNotIn("NVDA", submitted)
        self.assertTrue(submitted, "other signals should still be submitted")

    def test_stacking_can_be_allowed(self):
        broker = mock.Mock(spec=["submit", "open_symbols"])
        broker.open_symbols.return_value = {"NVDA"}

        algorithm(broker, dry_run=False, skip_held_symbols=False).run(now=NOW)
        submitted = {call.args[0].symbol for call in broker.submit.call_args_list}
        self.assertIn("NVDA", submitted)

    def test_positions_are_read_once_per_run(self):
        broker = mock.Mock(spec=["submit", "open_symbols"])
        broker.open_symbols.return_value = set()
        algorithm(broker, dry_run=False).run(now=NOW)
        self.assertEqual(broker.open_symbols.call_count, 1)

    def test_a_failing_position_read_does_not_become_a_trade_blocker(self):
        broker = mock.Mock(spec=["submit", "open_symbols"])
        broker.open_symbols.side_effect = RuntimeError("api down")
        report = algorithm(broker, dry_run=False).run(now=NOW)
        self.assertTrue([s for s in report.signals if s.tradable])

    def test_the_paper_broker_reports_its_own_book(self):
        broker = PaperBroker(prices=StaticPriceProvider(default=100.0))
        self.assertEqual(broker.open_symbols(), set())
        report = algorithm(broker, dry_run=False).run(now=NOW)
        self.assertTrue(broker.open_symbols())
        # A second run must not stack into the same names.
        again = algorithm(broker, dry_run=False).run(now=NOW)
        self.assertTrue(
            all(s.skip_reason == "already holding" for s in again.signals if s.mention_count >= 2)
            or not [r for r in again.results if r.accepted]
        )
        self.assertTrue(report.results)

    def test_a_dry_run_never_reads_positions(self):
        broker = mock.Mock(spec=["submit", "open_symbols"])
        algorithm(broker, dry_run=True).run(now=NOW)
        broker.open_symbols.assert_not_called()


class PreflightTests(unittest.TestCase):
    def run_cli(self, *argv):
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(list(argv))
        return code, out.getvalue()

    def test_paper_preflight_passes_and_trades_nothing(self):
        code, out = self.run_cli("--check", "--fixture", MARKETS)
        self.assertEqual(code, 0)
        self.assertIn("Ready", out)
        self.assertIn("Nothing was traded", out)

    def test_preflight_reports_a_healthy_alpaca_account(self):
        account = {
            "status": "ACTIVE", "buying_power": "200000", "cash": "100000",
            "shorting_enabled": True, "trading_blocked": False,
        }

        def fake(url, payload=None):
            if url.endswith("/v2/account"):
                return account
            if url.endswith("/v2/clock"):
                return {"is_open": True, "next_open": "2026-09-11T13:30:00Z"}
            return []

        with mock.patch.dict(
            "os.environ",
            {"APCA_API_KEY_ID": "k", "APCA_API_SECRET_KEY": "s"},
            clear=True,
        ):
            with mock.patch.object(AlpacaBroker, "_request", side_effect=fake):
                code, out = self.run_cli("--check", "--broker", "alpaca", "--fixture", MARKETS)

        self.assertEqual(code, 0)
        self.assertIn("ACTIVE", out)
        self.assertIn("OPEN", out)
        self.assertIn("shorting enabled  : True", out)

    def test_preflight_fails_when_shorting_is_disabled(self):
        def fake(url, payload=None):
            if url.endswith("/v2/account"):
                return {"status": "ACTIVE", "shorting_enabled": False}
            if url.endswith("/v2/clock"):
                return {"is_open": False}
            return []

        with mock.patch.dict(
            "os.environ",
            {"APCA_API_KEY_ID": "k", "APCA_API_SECRET_KEY": "s"},
            clear=True,
        ):
            with mock.patch.object(AlpacaBroker, "_request", side_effect=fake):
                code, out = self.run_cli("--check", "--broker", "alpaca", "--fixture", MARKETS)

        self.assertEqual(code, 1)
        self.assertIn("NOT READY", out)
        self.assertIn("shorting is disabled", out)

    def test_preflight_fails_on_a_blocked_account(self):
        def fake(url, payload=None):
            if url.endswith("/v2/account"):
                return {"status": "ACCOUNT_CLOSED", "trading_blocked": True,
                        "shorting_enabled": True}
            if url.endswith("/v2/clock"):
                return {"is_open": True}
            return []

        with mock.patch.dict(
            "os.environ",
            {"APCA_API_KEY_ID": "k", "APCA_API_SECRET_KEY": "s"},
            clear=True,
        ):
            with mock.patch.object(AlpacaBroker, "_request", side_effect=fake):
                code, out = self.run_cli("--check", "--broker", "alpaca", "--fixture", MARKETS)

        self.assertEqual(code, 1)
        self.assertIn("trading blocked", out)

    def test_preflight_fails_when_credentials_are_rejected(self):
        with mock.patch.dict(
            "os.environ",
            {"APCA_API_KEY_ID": "bad", "APCA_API_SECRET_KEY": "bad"},
            clear=True,
        ):
            with mock.patch.object(
                AlpacaBroker, "_request", side_effect=BrokerError("403 forbidden")
            ):
                code, out = self.run_cli("--check", "--broker", "alpaca", "--fixture", MARKETS)

        self.assertEqual(code, 1)
        self.assertIn("NOT READY", out)
        self.assertIn("403", out)

    def test_preflight_fails_when_no_feed_can_be_read(self):
        code, out = self.run_cli("--check", "--feed", "https://invalid.invalid/rss")
        self.assertEqual(code, 1)
        self.assertIn("no feed could be read", out)


class CliDefaultsTests(unittest.TestCase):
    def test_alpaca_live_requires_an_open_market_by_default(self):
        captured = {}
        real = WSJHeadlineAlgorithm.__init__

        def spy(self, config=None, **kwargs):
            captured["config"] = config
            real(self, config, **kwargs)

        def fake(url, payload=None):
            if url.endswith("/v2/clock"):
                return {"is_open": False}
            return {} if payload else []

        with mock.patch.dict(
            "os.environ",
            {"APCA_API_KEY_ID": "k", "APCA_API_SECRET_KEY": "s"},
            clear=True,
        ):
            with mock.patch.object(AlpacaBroker, "_request", side_effect=fake):
                with mock.patch.object(WSJHeadlineAlgorithm, "__init__", spy):
                    with redirect_stdout(io.StringIO()):
                        main(["--broker", "alpaca", "--live", "--fixture", MARKETS,
                              "--as-of", NOW.isoformat()])

        self.assertTrue(captured["config"].require_market_open)
        self.assertTrue(captured["config"].skip_held_symbols)

    def test_queue_when_closed_turns_the_guard_off(self):
        captured = {}
        real = WSJHeadlineAlgorithm.__init__

        def spy(self, config=None, **kwargs):
            captured["config"] = config
            real(self, config, **kwargs)

        with mock.patch.dict(
            "os.environ",
            {"APCA_API_KEY_ID": "k", "APCA_API_SECRET_KEY": "s"},
            clear=True,
        ):
            with mock.patch.object(AlpacaBroker, "_request", return_value={}):
                with mock.patch.object(WSJHeadlineAlgorithm, "__init__", spy):
                    with redirect_stdout(io.StringIO()):
                        main(["--broker", "alpaca", "--live", "--queue-when-closed",
                              "--fixture", MARKETS, "--as-of", NOW.isoformat()])

        self.assertFalse(captured["config"].require_market_open)

    def test_dry_run_does_not_require_an_open_market(self):
        captured = {}
        real = WSJHeadlineAlgorithm.__init__

        def spy(self, config=None, **kwargs):
            captured["config"] = config
            real(self, config, **kwargs)

        with mock.patch.object(WSJHeadlineAlgorithm, "__init__", spy):
            with redirect_stdout(io.StringIO()):
                main(["--fixture", MARKETS, "--as-of", NOW.isoformat()])

        self.assertFalse(captured["config"].require_market_open)


if __name__ == "__main__":
    unittest.main()
