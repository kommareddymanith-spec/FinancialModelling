"""Pine Script generation.

These tests check the generator's contract -- structure, symbol filtering,
escaping, ordering, caps. They cannot check that TradingView accepts the
script: Pine's only compiler is TradingView itself, so the syntax here is
written to the v5 language reference and verified by pasting it in.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import tempfile
import unittest

from wsj_headline_trader.algorithm import append_signal_log
from wsj_headline_trader.models import RunReport, Side, Signal
from wsj_headline_trader.pine import (
    PineError,
    PineSignal,
    from_signal_log,
    from_signals,
    render,
    write,
)

UTC = dt.timezone.utc
WHEN = dt.datetime(2026, 9, 10, 15, 30, tzinfo=UTC)


def signal(symbol="NVDA", side=Side.BUY, at=WHEN, mentions=3, sentiment=1.44):
    return PineSignal(symbol=symbol, at=at, side=side, mentions=mentions, sentiment=sentiment)


class EncodingTests(unittest.TestCase):
    def test_epoch_millis(self):
        self.assertEqual(signal().epoch_millis, int(WHEN.timestamp() * 1000))

    def test_naive_timestamps_are_treated_as_utc(self):
        naive = PineSignal("NVDA", WHEN.replace(tzinfo=None), Side.BUY)
        self.assertEqual(naive.epoch_millis, signal().epoch_millis)

    def test_record_layout(self):
        fields = signal().encode().split("|")
        self.assertEqual(len(fields), 5)
        self.assertEqual(fields[0], "NVDA")
        self.assertEqual(fields[2], "1")
        self.assertEqual(fields[3], "3")

    def test_short_encodes_as_minus_one(self):
        self.assertEqual(signal(side=Side.SHORT).encode().split("|")[2], "-1")


class RenderStructureTests(unittest.TestCase):
    def setUp(self):
        self.script = render([signal(), signal("BA", Side.SHORT, sentiment=-1.2)])

    def test_declares_pine_version_five(self):
        self.assertIn("//@version=5", self.script)

    def test_declares_a_strategy(self):
        self.assertIn("strategy(", self.script)
        self.assertIn("pyramiding         = 0", self.script)

    def test_embeds_both_signals(self):
        self.assertIn("NVDA|", self.script)
        self.assertIn("BA|", self.script)

    def test_filters_by_charted_symbol(self):
        self.assertIn("syminfo.ticker", self.script)

    def test_header_documents_provenance_and_the_pine_limitation(self):
        self.assertIn("GENERATED FILE", self.script)
        self.assertIn("Pine cannot", self.script)
        self.assertIn("BA, NVDA", self.script)

    def test_places_entries_and_a_timed_exit(self):
        self.assertIn("strategy.entry(\"WSJ long\", strategy.long)", self.script)
        self.assertIn("strategy.entry(\"WSJ short\", strategy.short)", self.script)
        self.assertIn("strategy.close_all", self.script)

    def test_exposes_inputs(self):
        for name in ("holdBars", "takeLongs", "takeShorts", "showMarks", "showLabels"):
            self.assertIn(name, self.script)

    def test_offers_alert_conditions(self):
        self.assertIn("alertcondition", self.script)

    def test_no_unresolved_format_placeholders(self):
        self.assertNotIn("{0}", self.script)
        self.assertNotIn("{{", self.script)

    def test_uses_spaces_not_tabs(self):
        # Pine rejects tab indentation.
        self.assertNotIn("\t", self.script)

    def test_string_literal_is_not_broken_by_the_data(self):
        data_line = next(
            line for line in self.script.splitlines() if "SIGNAL_DATA" in line and "=" in line
        )
        self.assertEqual(data_line.count('"'), 2)

    def test_hold_bars_is_propagated(self):
        self.assertIn("input.int(9,", render([signal()], hold_bars=9))

    def test_costs_are_propagated(self):
        script = render([signal()], commission_percent=0.05, slippage_ticks=7)
        self.assertIn("commission_value   = 0.05", script)
        self.assertIn("slippage           = 7", script)


class ValidationTests(unittest.TestCase):
    def test_flat_signals_are_dropped(self):
        script = render([signal(side=Side.FLAT), signal()])
        self.assertEqual(script.count("NVDA|"), 1)
        self.assertIn("Signals     1", script)

    def test_signals_are_sorted_by_time(self):
        later = signal("BA", Side.SHORT, WHEN + dt.timedelta(hours=2))
        earlier = signal("NVDA", Side.BUY, WHEN)
        data = re.search(r'SIGNAL_DATA = "([^"]*)"', render([later, earlier])).group(1)
        self.assertTrue(data.startswith("NVDA|"))

    def test_a_hostile_ticker_is_refused(self):
        for bad in ('NV"DA', "NVDA;DROP", "nv da", "", "A" * 20, "NVDA|X"):
            with self.subTest(bad):
                with self.assertRaises(PineError):
                    render([signal(symbol=bad)])

    def test_dotted_and_dashed_tickers_are_allowed(self):
        script = render([signal(symbol="BRK.B"), signal(symbol="BF-B")])
        self.assertIn("BRK.B|", script)
        self.assertIn("BF-B|", script)

    def test_the_signal_cap_keeps_the_most_recent(self):
        many = [
            signal("NVDA", Side.BUY, WHEN + dt.timedelta(minutes=i)) for i in range(50)
        ]
        script = render(many, max_signals=10)
        self.assertIn("Signals     10", script)
        newest = str(many[-1].epoch_millis)
        oldest = str(many[0].epoch_millis)
        self.assertIn(newest, script)
        self.assertNotIn(oldest, script)

    def test_no_signals_still_renders_a_valid_script(self):
        script = render([])
        self.assertIn("//@version=5", script)
        self.assertIn('SIGNAL_DATA = ""', script)
        self.assertIn("(none)", script)


class SignalLogTests(unittest.TestCase):
    def report_with_signals(self):
        report = RunReport(ran_at=WHEN, window_minutes=60, headlines_scanned=9)
        report.signals = [
            Signal(ticker="NVDA", company="Nvidia", mention_count=3, sentiment_sum=4.3,
                   sentiment_mean=1.44, side=Side.BUY, conviction=1.2, notional=2000.0,
                   article_sentiments=[1.4, 1.5, 1.4]),
            Signal(ticker="BA", company="Boeing", mention_count=3, sentiment_sum=-3.6,
                   sentiment_mean=-1.21, side=Side.SHORT, conviction=1.1, notional=1900.0,
                   article_sentiments=[-1.2, -1.2, -1.2]),
            Signal(ticker="PFE", company="Pfizer", mention_count=1, sentiment_sum=1.0,
                   sentiment_mean=1.0, side=Side.FLAT, conviction=0.0,
                   skip_reason="only 1 mention"),
        ]
        return report

    def test_only_tradable_signals_are_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sig.jsonl")
            self.assertEqual(append_signal_log(path, self.report_with_signals()), 2)
            with open(path, encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle]
        self.assertEqual([r["ticker"] for r in rows], ["NVDA", "BA"])
        self.assertEqual(rows[0]["side"], "buy")
        self.assertEqual(rows[1]["side"], "short")

    def test_logging_appends_across_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sig.jsonl")
            append_signal_log(path, self.report_with_signals())
            append_signal_log(path, self.report_with_signals())
            with open(path, encoding="utf-8") as handle:
                self.assertEqual(sum(1 for _ in handle), 4)

    def test_a_run_with_nothing_tradable_writes_no_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sig.jsonl")
            empty = RunReport(ran_at=WHEN, window_minutes=60, headlines_scanned=0)
            self.assertEqual(append_signal_log(path, empty), 0)
            self.assertFalse(os.path.exists(path))

    def test_round_trip_log_to_pine(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sig.jsonl")
            append_signal_log(path, self.report_with_signals())
            signals = from_signal_log(path)
        self.assertEqual({s.symbol for s in signals}, {"NVDA", "BA"})
        self.assertEqual(signals[0].mentions, 3)
        script = render(signals)
        self.assertIn("NVDA|", script)
        self.assertIn("BA|", script)

    def test_corrupt_and_incomplete_lines_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sig.jsonl")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "decided_at": WHEN.isoformat(), "ticker": "NVDA", "side": "buy"}) + "\n")
                handle.write("{not json\n")
                handle.write(json.dumps({"ticker": "BA", "side": "short"}) + "\n")
                handle.write(json.dumps({
                    "decided_at": WHEN.isoformat(), "ticker": "X", "side": "hold"}) + "\n")
                handle.write("\n")
            signals = from_signal_log(path)
        self.assertEqual([s.symbol for s in signals], ["NVDA"])

    def test_from_signals_skips_untradable(self):
        converted = from_signals(self.report_with_signals().signals, WHEN)
        self.assertEqual([s.symbol for s in converted], ["NVDA", "BA"])


class WriteTests(unittest.TestCase):
    def test_write_creates_a_readable_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.pine")
            write(path, render([signal()]))
            with open(path, encoding="utf-8") as handle:
                self.assertIn("//@version=5", handle.read())


if __name__ == "__main__":
    unittest.main()
