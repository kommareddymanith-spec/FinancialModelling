"""The exit job: classification, ledger handling, and closing."""

from __future__ import annotations

import datetime as dt
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

from wsj_headline_trader.broker import (
    AlpacaBroker,
    BrokerError,
    BrokerPosition,
    PaperBroker,
    StaticPriceProvider,
)
from wsj_headline_trader.exit_cli import main
from wsj_headline_trader.exit_job import (
    ExitConfig,
    classify,
    format_exit_report,
    prune_ledger,
    read_ledger,
    record_entry,
    run_exit_job,
)
from wsj_headline_trader.models import Order, OrderResult, Side

NOW = dt.datetime(2026, 9, 10, 15, 30, tzinfo=dt.timezone.utc)


def position(symbol="NVDA", side=Side.BUY, plpc=0.0, entry=100.0):
    price = entry * (1 + plpc) if side is Side.BUY else entry * (1 - plpc)
    return BrokerPosition(
        symbol=symbol,
        qty=10.0,
        side=side,
        avg_entry_price=entry,
        current_price=price,
        unrealized_pl=plpc * entry * 10.0,
        unrealized_plpc=plpc,
    )


class FakeBroker:
    def __init__(self, positions, fail=()):
        self._positions = list(positions)
        self.closed: list[str] = []
        self.fail = set(fail)

    def open_positions(self):
        return list(self._positions)

    def close_position(self, symbol):
        self.closed.append(symbol)
        if symbol in self.fail:
            return OrderResult(
                order=Order(symbol=symbol, side=Side.FLAT),
                accepted=False,
                message="403 forbidden",
            )
        self._positions = [p for p in self._positions if p.symbol != symbol]
        return OrderResult(
            order=Order(symbol=symbol, side=Side.FLAT), accepted=True, message="closed"
        )


class ClassifyTests(unittest.TestCase):
    CONFIG = ExitConfig(stop_loss=0.04, take_profit=0.08, max_hold_days=5)

    def test_the_shipped_defaults_are_the_swept_ones(self):
        # Pins the outcome of data/exit_sweep_2026-09-11.txt, so a change to
        # the defaults has to be deliberate.
        default = ExitConfig()
        self.assertEqual(default.stop_loss, 0.02)
        self.assertEqual(default.take_profit, 0.12)
        self.assertEqual(default.max_hold_days, 5)

    def test_a_loss_past_the_stop_closes(self):
        self.assertEqual(classify(position(plpc=-0.05), self.CONFIG, None), "stop loss")

    def test_a_loss_exactly_at_the_stop_closes(self):
        self.assertEqual(classify(position(plpc=-0.04), self.CONFIG, None), "stop loss")

    def test_a_gain_past_the_target_closes(self):
        self.assertEqual(classify(position(plpc=0.09), self.CONFIG, None), "take profit")

    def test_a_position_inside_both_thresholds_is_held(self):
        self.assertIsNone(classify(position(plpc=0.02), self.CONFIG, None))

    def test_the_time_stop_closes_an_aged_position(self):
        self.assertEqual(classify(position(plpc=0.01), self.CONFIG, 6.0), "time stop")

    def test_a_young_position_is_held(self):
        self.assertIsNone(classify(position(plpc=0.01), self.CONFIG, 2.0))

    def test_the_loss_rule_wins_when_both_are_breached(self):
        # A gap through both thresholds between runs: taking the loss is the
        # conservative reading of an unknown intrabar path.
        config = ExitConfig(stop_loss=0.02, take_profit=0.02)
        self.assertEqual(classify(position(plpc=-0.5), config, None), "stop loss")

    def test_short_pnl_is_already_signed_for_direction(self):
        # A short that has gained shows a positive unrealized percentage.
        winner = position(side=Side.SHORT, plpc=0.09)
        self.assertLess(winner.current_price, winner.avg_entry_price)
        self.assertEqual(classify(winner, self.CONFIG, None), "take profit")

    def test_disabled_rules_never_fire(self):
        config = ExitConfig(stop_loss=None, take_profit=None, max_hold_days=5)
        self.assertIsNone(classify(position(plpc=-0.9), config, 1.0))
        self.assertIsNone(classify(position(plpc=9.0), config, 1.0))

    def test_config_validation(self):
        for kwargs in (
            {"stop_loss": 1.5},
            {"take_profit": 0},
            {"max_hold_days": 0},
            {"stop_loss": None, "take_profit": None, "max_hold_days": None},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    ExitConfig(**kwargs)


class RunExitJobTests(unittest.TestCase):
    def test_dry_run_closes_nothing(self):
        broker = FakeBroker([position(plpc=-0.10)])
        report = run_exit_job(
            broker, ExitConfig(stop_loss=0.04, take_profit=0.08, dry_run=True), now=NOW
        )
        self.assertEqual(broker.closed, [])
        self.assertEqual(len(report.closed), 1)
        self.assertIn("would be closed", format_exit_report(report))

    def test_live_run_closes_the_qualifying_positions(self):
        broker = FakeBroker([
            position("NVDA", plpc=0.10),
            position("BA", Side.SHORT, plpc=-0.06),
            position("F", plpc=0.01),
        ])
        # Thresholds stated rather than inherited: the defaults are tuned, and
        # a test that tracks them silently changes meaning when they move.
        report = run_exit_job(
            broker,
            ExitConfig(stop_loss=0.04, take_profit=0.08, max_hold_days=None, dry_run=False),
            now=NOW,
        )
        self.assertEqual(sorted(broker.closed), ["BA", "NVDA"])
        self.assertEqual([d.position.symbol for d in report.held], ["F"])

    def test_no_positions_is_reported_not_an_error(self):
        report = run_exit_job(FakeBroker([]), ExitConfig(dry_run=False), now=NOW)
        self.assertEqual(report.decisions, [])
        self.assertIn("no open positions", report.notes)

    def test_a_failed_close_is_surfaced(self):
        broker = FakeBroker([position(plpc=-0.10)], fail={"NVDA"})
        report = run_exit_job(
            broker,
            ExitConfig(stop_loss=0.04, take_profit=0.08, max_hold_days=None, dry_run=False),
            now=NOW,
        )
        decision = report.closed[0]
        self.assertFalse(decision.result.accepted)
        self.assertIn("FAILED", format_exit_report(report))

    def test_a_missing_ledger_notes_the_skipped_time_stop(self):
        report = run_exit_job(
            FakeBroker([position(plpc=0.01)]),
            ExitConfig(dry_run=True, max_hold_days=5, ledger_path=None),
            now=NOW,
        )
        self.assertTrue(any("time stop is skipped" in n for n in report.notes))

    def test_the_report_serialises(self):
        broker = FakeBroker([position(plpc=-0.10)])
        report = run_exit_job(
            broker, ExitConfig(stop_loss=0.04, take_profit=0.08, dry_run=True), now=NOW
        )
        payload = json.loads(json.dumps(report.to_dict()))
        self.assertEqual(payload["positions"][0]["action"], "stop loss")


class LedgerTests(unittest.TestCase):
    def test_record_and_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.json")
            record_entry(path, "nvda", NOW)
            self.assertEqual(list(read_ledger(path)), ["NVDA"])

    def test_a_missing_ledger_reads_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(read_ledger(os.path.join(tmp, "nope.json")), {})

    def test_a_corrupt_ledger_reads_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.json")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("{not json")
            self.assertEqual(read_ledger(path), {})

    def test_a_non_object_ledger_reads_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(["not", "an", "object"], handle)
            self.assertEqual(read_ledger(path), {})

    def test_re_entry_restarts_the_clock(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.json")
            record_entry(path, "NVDA", NOW - dt.timedelta(days=10))
            record_entry(path, "NVDA", NOW)
            self.assertEqual(read_ledger(path)["NVDA"], NOW.isoformat())

    def test_the_time_stop_uses_the_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.json")
            record_entry(path, "NVDA", NOW - dt.timedelta(days=9))
            record_entry(path, "F", NOW - dt.timedelta(days=1))
            broker = FakeBroker([position("NVDA", plpc=0.01), position("F", plpc=0.01)])
            report = run_exit_job(
                broker,
                ExitConfig(dry_run=False, max_hold_days=5, ledger_path=path),
                now=NOW,
            )
        self.assertEqual(broker.closed, ["NVDA"])
        self.assertEqual([d.position.symbol for d in report.held], ["F"])

    def test_an_unparsable_timestamp_does_not_age_a_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"NVDA": "last tuesday"}, handle)
            broker = FakeBroker([position("NVDA", plpc=0.01)])
            run_exit_job(
                broker,
                ExitConfig(dry_run=False, max_hold_days=1, ledger_path=path),
                now=NOW,
            )
        self.assertEqual(broker.closed, [])

    def test_closing_prunes_the_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.json")
            record_entry(path, "NVDA", NOW)
            record_entry(path, "BA", NOW)
            broker = FakeBroker([position("NVDA", plpc=-0.10), position("BA", plpc=0.01)])
            run_exit_job(
                broker,
                ExitConfig(dry_run=False, max_hold_days=None, ledger_path=path),
                now=NOW,
            )
            self.assertEqual(list(read_ledger(path)), ["BA"])

    def test_prune_is_a_noop_when_nothing_is_stale(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.json")
            record_entry(path, "NVDA", NOW)
            self.assertEqual(prune_ledger(path, ["NVDA"]), 0)


class PaperBrokerPositionTests(unittest.TestCase):
    def test_positions_are_reported_with_pnl(self):
        prices = StaticPriceProvider({"NVDA": 100.0})
        broker = PaperBroker(prices=prices)
        broker.submit(Order(symbol="NVDA", side=Side.BUY, notional=1000.0))
        prices.prices["NVDA"] = 110.0
        held = broker.open_positions()[0]
        self.assertEqual(held.avg_entry_price, 100.0)
        self.assertAlmostEqual(held.unrealized_plpc, 0.10, places=6)

    def test_a_short_reports_a_signed_gain(self):
        prices = StaticPriceProvider({"BA": 100.0})
        broker = PaperBroker(prices=prices)
        broker.submit(Order(symbol="BA", side=Side.SHORT, notional=1000.0))
        prices.prices["BA"] = 90.0
        held = broker.open_positions()[0]
        self.assertEqual(held.side, Side.SHORT)
        self.assertAlmostEqual(held.unrealized_plpc, 0.10, places=6)

    def test_entry_price_is_volume_weighted_across_top_ups(self):
        prices = StaticPriceProvider({"NVDA": 100.0})
        broker = PaperBroker(prices=prices)
        broker.submit(Order(symbol="NVDA", side=Side.BUY, qty=10.0))
        prices.prices["NVDA"] = 200.0
        broker.submit(Order(symbol="NVDA", side=Side.BUY, qty=10.0))
        self.assertAlmostEqual(broker.entry_prices["NVDA"], 150.0, places=6)

    def test_close_position_flattens(self):
        prices = StaticPriceProvider({"NVDA": 100.0})
        broker = PaperBroker(prices=prices)
        broker.submit(Order(symbol="NVDA", side=Side.BUY, notional=1000.0))
        result = broker.close_position("NVDA")
        self.assertTrue(result.accepted)
        self.assertEqual(broker.open_positions(), [])

    def test_closing_nothing_is_refused(self):
        result = PaperBroker().close_position("NVDA")
        self.assertFalse(result.accepted)
        self.assertIn("no open position", result.message)

    def test_a_close_does_not_move_the_entry_basis(self):
        prices = StaticPriceProvider({"NVDA": 100.0})
        broker = PaperBroker(prices=prices)
        broker.submit(Order(symbol="NVDA", side=Side.BUY, qty=10.0))
        prices.prices["NVDA"] = 150.0
        broker.close_position("NVDA")
        self.assertNotIn("NVDA", broker.entry_prices)


class AlpacaPositionTests(unittest.TestCase):
    def setUp(self):
        self.broker = AlpacaBroker(key_id="k", secret_key="s")

    def test_positions_are_parsed(self):
        body = [{
            "symbol": "nvda", "qty": "10", "side": "long", "avg_entry_price": "100.5",
            "current_price": "110.25", "unrealized_pl": "97.5", "unrealized_plpc": "0.097",
        }]
        with mock.patch.object(self.broker, "_request", return_value=body):
            held = self.broker.open_positions()[0]
        self.assertEqual(held.symbol, "NVDA")
        self.assertEqual(held.side, Side.BUY)
        self.assertAlmostEqual(held.unrealized_plpc, 0.097, places=6)

    def test_a_short_row_is_recognised(self):
        body = [{
            "symbol": "BA", "qty": "-4", "side": "short", "avg_entry_price": "200",
            "current_price": "190", "unrealized_pl": "40", "unrealized_plpc": "0.05",
        }]
        with mock.patch.object(self.broker, "_request", return_value=body):
            held = self.broker.open_positions()[0]
        self.assertEqual(held.side, Side.SHORT)
        self.assertEqual(held.qty, 4.0)

    def test_unparsable_rows_are_skipped(self):
        body = [{"symbol": "X", "qty": "oops"}, {"qty": "1"}]
        with mock.patch.object(self.broker, "_request", return_value=body):
            self.assertEqual(self.broker.open_positions(), [])

    def test_an_unreachable_endpoint_returns_nothing(self):
        with mock.patch.object(self.broker, "_request", side_effect=BrokerError("down")):
            self.assertEqual(self.broker.open_positions(), [])

    def test_close_position_uses_delete(self):
        captured = {}

        class FakeResponse:
            def read(self):
                return b'{"id": "abc", "status": "accepted", "qty": "10"}'

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        def fake_urlopen(request, timeout=None):
            captured["method"] = request.get_method()
            captured["url"] = request.full_url
            return FakeResponse()

        with mock.patch("urllib.request.urlopen", fake_urlopen):
            result = self.broker.close_position("NVDA")

        self.assertEqual(captured["method"], "DELETE")
        self.assertTrue(captured["url"].endswith("/v2/positions/NVDA"))
        self.assertTrue(result.accepted)
        self.assertEqual(result.broker_order_id, "abc")


class ExitCliTests(unittest.TestCase):
    def run_cli(self, *argv):
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(list(argv))
        return code, out.getvalue()

    def test_paper_run_with_an_empty_book(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            code, out = self.run_cli("--broker", "paper")
        self.assertEqual(code, 0)
        self.assertIn("no open positions", out)

    def test_thresholds_accept_none(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            code, _ = self.run_cli("--stop-loss", "none", "--take-profit", "0.05",
                                   "--max-hold-days", "none")
        self.assertEqual(code, 0)

    def test_all_rules_disabled_is_an_error(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            code, _ = self.run_cli("--stop-loss", "none", "--take-profit", "none",
                                   "--max-hold-days", "none")
        self.assertEqual(code, 2)

    def test_a_bad_threshold_is_a_usage_error(self):
        with self.assertRaises(SystemExit):
            self.run_cli("--stop-loss", "tight")

    def test_json_output(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            code, out = self.run_cli("--broker", "paper", "--json")
        self.assertEqual(code, 0)
        self.assertIn("positions", json.loads(out))

    def test_alpaca_without_credentials_is_an_error(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            code, _ = self.run_cli("--broker", "alpaca")
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
