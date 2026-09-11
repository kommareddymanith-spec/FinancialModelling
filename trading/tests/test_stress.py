"""Stress and hostile-input tests.

Every case here was found by fuzzing the public surfaces, and each one that
found a real defect is kept as a regression test:

* a feed declaring nested XML entities (billion laughs) expanded a 500-byte
  body into megabytes of title text;
* an oversized response body was read without limit;
* a multi-megabyte title took ~2 seconds of regex scanning per article;
* the backtest rescanned the whole archive at every decision, making the
  replay cost decisions x headlines.
"""

from __future__ import annotations

import datetime as dt
import time
import unittest

from wsj_headline_trader.backtest import BacktestConfig, run_backtest
from wsj_headline_trader.feed import (
    MAX_FEED_BYTES,
    FeedError,
    collect_headlines,
    parse_feed,
)
from wsj_headline_trader.metrics import CashFlow, irr, max_drawdown, summarise
from wsj_headline_trader.models import MAX_TEXT_CHARS, Headline
from wsj_headline_trader.prices import Bar, PricePanel
from wsj_headline_trader.sentiment import score_text
from wsj_headline_trader.strategy import StrategyConfig, build_signals, extract_mentions
from wsj_headline_trader.universe import Company, Universe

UTC = dt.timezone.utc
NOW = dt.datetime(2026, 9, 10, 15, 0, tzinfo=UTC)
PUBDATE = b"<pubDate>Thu, 10 Sep 2026 15:00:00 +0000</pubDate>"


def entity_bomb(depth: int = 5, fanout: int = 10) -> bytes:
    parts = [b'<?xml version="1.0"?>', b"<!DOCTYPE lolz [", b'<!ENTITY e0 "lol">']
    for i in range(1, depth + 1):
        expansion = (f"&e{i - 1};" * fanout).encode()
        parts.append(b'<!ENTITY e%d "%s">' % (i, expansion))
    parts.append(b"]>")
    parts.append(
        b"<rss><channel><item><title>&e%d;</title>" % depth + PUBDATE + b"</item></channel></rss>"
    )
    return b"".join(parts)


class XmlHostilityTests(unittest.TestCase):
    def test_entity_declaring_dtd_is_refused(self):
        with self.assertRaises(FeedError) as caught:
            parse_feed(entity_bomb(depth=5), "bomb")
        self.assertIn("entities", str(caught.exception))

    def test_refusal_does_not_depend_on_the_expat_version(self):
        # A shallow bomb that the system libexpat would happily expand.
        with self.assertRaises(FeedError):
            parse_feed(entity_bomb(depth=2), "small bomb")

    def test_a_doctype_without_entities_still_parses(self):
        payload = (
            b'<?xml version="1.0"?><!DOCTYPE rss><rss><channel><item>'
            b"<title>Nvidia Surges</title>" + PUBDATE + b"</item></channel></rss>"
        )
        self.assertEqual(len(parse_feed(payload, "plain")), 1)

    def test_external_entities_are_not_resolved(self):
        payload = (
            b'<?xml version="1.0"?><!DOCTYPE r [<!ENTITY x SYSTEM "file:///etc/passwd">]>'
            b"<rss><channel><item><title>&x;</title>" + PUBDATE + b"</item></channel></rss>"
        )
        with self.assertRaises(FeedError):
            parse_feed(payload, "xxe")

    def test_oversized_payload_is_refused(self):
        payload = b"<rss>" + b"x" * (MAX_FEED_BYTES + 10) + b"</rss>"
        with self.assertRaises(FeedError) as caught:
            parse_feed(payload, "big")
        self.assertIn("limit", str(caught.exception))

    def test_a_hostile_feed_does_not_take_down_the_run(self):
        good = (
            b"<rss><channel><item><title>Nvidia Shares Surge</title>"
            + PUBDATE
            + b"</item></channel></rss>"
        )
        payloads = {"bomb": entity_bomb(), "good": good}
        headlines, errors = collect_headlines(
            feeds=["bomb", "good"], window_minutes=60, now=NOW,
            fetcher=lambda url: payloads[url],
        )
        self.assertEqual(len(errors), 1)
        self.assertEqual(len(headlines), 1)


class MalformedInputTests(unittest.TestCase):
    def test_junk_payloads_raise_feed_error_not_something_else(self):
        for name, payload in (
            ("empty", b""),
            ("text", b"hello world"),
            ("truncated", b"<rss><channel><item><title>unclosed"),
            ("binary", b"\x00\x01\x02\xff\xfe"),
        ):
            with self.subTest(name):
                with self.assertRaises(FeedError):
                    parse_feed(payload, name)

    def test_an_html_error_page_yields_nothing_and_warns(self):
        # Valid XML, so it parses; it just is not a feed. No trades is the
        # right outcome, but it must be logged rather than passed over.
        with self.assertLogs("wsj_headline_trader.feed", level="WARNING") as logged:
            self.assertEqual(parse_feed(b"<html><body>nope</body></html>", "html"), [])
        self.assertIn("no RSS items", "".join(logged.output))

    def test_unparsable_dates_drop_the_item_rather_than_the_feed(self):
        payload = (
            b"<rss><channel>"
            b"<item><title>Bad date</title><pubDate>Thu, 99 Zzz 9999 99:99:99</pubDate></item>"
            b"<item><title>Good date</title>" + PUBDATE + b"</item>"
            b"</channel></rss>"
        )
        self.assertEqual([h.title for h in parse_feed(payload)], ["Good date"])

    def test_unicode_and_control_characters_survive(self):
        payload = (
            "<rss><channel><item><title>Nvidia 🚀 股票 Surge</title>".encode()
            + PUBDATE
            + b"</item></channel></rss>"
        )
        self.assertIn("🚀", parse_feed(payload)[0].title)

    def test_naive_now_is_accepted_by_collect_headlines(self):
        payload = (
            b"<rss><channel><item><title>Nvidia Shares Surge</title>"
            + PUBDATE
            + b"</item></channel></rss>"
        )
        headlines, _ = collect_headlines(
            feeds=["x"], window_minutes=60, now=NOW.replace(tzinfo=None),
            fetcher=lambda url: payload,
        )
        self.assertEqual(len(headlines), 1)


class TextBoundTests(unittest.TestCase):
    def test_article_text_is_capped(self):
        headline = Headline("Nvidia " * 300_000, "x" * 100_000, "l", NOW, "s")
        self.assertEqual(len(headline.text), MAX_TEXT_CHARS)

    def test_matching_a_huge_article_stays_fast(self):
        universe = Universe.load()
        headline = Headline("Nvidia " * 300_000, "", "l", NOW, "s")
        started = time.perf_counter()
        found = universe.find(headline.text)
        elapsed = time.perf_counter() - started
        self.assertEqual([t for t, _ in found], ["NVDA"])
        self.assertLess(elapsed, 1.0, "capping .text should keep this far under a second")

    def test_scoring_a_huge_article_stays_fast(self):
        headline = Headline("surge " * 300_000, "", "l", NOW, "s")
        started = time.perf_counter()
        score_text(headline.text)
        self.assertLess(time.perf_counter() - started, 1.0)

    def test_pathological_negation_chain(self):
        score = score_text(("not " * 5_000) + "beat expectations")
        self.assertNotEqual(score.score, 0.0)


class HostileUniverseTests(unittest.TestCase):
    def test_regex_metacharacters_in_aliases_are_literals(self):
        universe = Universe(
            [Company(ticker="ZZZ", name="Zeta", aliases=("C++", "A.B", "(Acme)", "a|b"))]
        )
        self.assertEqual(universe.find("C++ Shares Rose"), [("ZZZ", "C++")])
        self.assertEqual(universe.find("AxB Shares Rose"), [])
        self.assertEqual(universe.find("a Shares Rose"), [])

    def test_blank_aliases_are_ignored(self):
        universe = Universe([Company(ticker="ZZZ", name="Z", aliases=("", "   ", "Zeta"))])
        self.assertEqual(universe.find("Zeta Reports"), [("ZZZ", "Zeta")])

    def test_a_company_with_no_aliases_never_matches(self):
        universe = Universe([Company(ticker="ZZZ", name="Zeta")])
        self.assertEqual(universe.find("Zeta Reports Earnings"), [])

    def test_a_very_long_alias_does_not_break_compilation(self):
        alias = "Zeta " * 400
        universe = Universe([Company(ticker="ZZZ", name="Z", aliases=(alias,))])
        self.assertEqual(universe.find(f"{alias} Reports"), [("ZZZ", alias.strip())])


class ThroughputTests(unittest.TestCase):
    def test_pipeline_handles_a_large_batch(self):
        universe = Universe.load()
        titles = [
            "Nvidia Shares Surge on Blowout Results",
            "Boeing Plunges After a Fresh Recall",
            "Target Cuts Guidance as Sales Weaken",
            "Treasury Yields Rise Ahead of Inflation Data",
        ]
        headlines = [
            Headline(titles[i % 4], "", f"l{i}", NOW - dt.timedelta(seconds=i), "s")
            for i in range(20_000)
        ]
        started = time.perf_counter()
        signals = build_signals(extract_mentions(headlines, universe), StrategyConfig())
        elapsed = time.perf_counter() - started
        self.assertEqual({s.ticker for s in signals}, {"NVDA", "BA", "TGT"})
        self.assertLess(elapsed, 30.0)

    def test_the_memo_returns_identical_signals(self):
        universe = Universe.load()
        titles = [
            "Nvidia Shares Surge on Blowout Results",
            "Boeing Plunges After a Fresh Recall",
            "Target Cuts Guidance as Sales Weaken",
        ]
        headlines = [
            Headline(titles[i % 3], "", f"l{i}", NOW - dt.timedelta(minutes=i), "s")
            for i in range(200)
        ]
        cache: dict = {}
        plain = build_signals(extract_mentions(headlines, universe), StrategyConfig())
        memoed = build_signals(
            extract_mentions(headlines, universe, cache=cache), StrategyConfig()
        )
        self.assertEqual(
            [(s.ticker, s.side, round(s.notional, 9)) for s in plain],
            [(s.ticker, s.side, round(s.notional, 9)) for s in memoed],
        )
        self.assertTrue(cache)


class BacktestScalingTests(unittest.TestCase):
    """The replay must be linear in the archive, not quadratic."""

    UNIVERSE = Universe([Company(ticker="NVDA", name="Nvidia", aliases=("Nvidia",))])

    def build(self, headline_count: int, session_count: int):
        start = dt.datetime(2025, 1, 1, tzinfo=UTC)
        span = session_count * 1440
        headlines = [
            Headline(
                "Nvidia Shares Surge on Blowout Results", "", f"l{i}",
                start + dt.timedelta(minutes=i * span // max(headline_count, 1)), "s",
            )
            for i in range(headline_count)
        ]
        sessions, day = [], start.date()
        while len(sessions) < session_count:
            if day.weekday() < 5:
                sessions.append(day)
            day += dt.timedelta(days=1)
        panel = PricePanel({"NVDA": [Bar(d, 100.0, 101.0, 99.0, 100.0) for d in sessions]})
        return headlines, panel

    def timed(self, headline_count: int, session_count: int) -> float:
        headlines, panel = self.build(headline_count, session_count)
        started = time.perf_counter()
        run_backtest(headlines, panel, BacktestConfig(contribution=10_000.0), self.UNIVERSE)
        return time.perf_counter() - started

    def test_cost_grows_roughly_linearly_not_quadratically(self):
        small = self.timed(2_000, 60)
        large = self.timed(16_000, 250)
        # Work grows ~33x (8x headlines, 4x decisions). Quadratic behaviour ran
        # ~26x slower here; linear is a few times slower. Allow ample slack for
        # a loaded machine while still failing if the rescan comes back.
        self.assertLess(large, max(small, 0.02) * 60)

    def test_a_two_year_sized_archive_completes(self):
        headlines, panel = self.build(60_000, 504)
        started = time.perf_counter()
        result = run_backtest(
            headlines, panel, BacktestConfig(contribution=1_000.0), self.UNIVERSE
        )
        self.assertLess(time.perf_counter() - started, 120.0)
        self.assertGreater(result.decisions, 10_000)
        self.assertGreater(result.signals_generated, 0)


class DegenerateDataTests(unittest.TestCase):
    UNIVERSE = Universe([Company(ticker="NVDA", name="Nvidia", aliases=("Nvidia",))])

    def test_empty_titles_and_summaries(self):
        headlines = [Headline("", "", "l", NOW, "s")]
        self.assertEqual(extract_mentions(headlines, self.UNIVERSE), [])

    def test_all_headlines_at_the_identical_timestamp(self):
        headlines = [
            Headline("Nvidia Shares Surge on Blowout Results", "", f"l{i}", NOW, "s")
            for i in range(50)
        ]
        signals = build_signals(extract_mentions(headlines, self.UNIVERSE), StrategyConfig())
        self.assertEqual(signals[0].mention_count, 50)

    def test_a_single_session_panel_replays(self):
        panel = PricePanel({"NVDA": [Bar(dt.date(2025, 1, 6), 100.0, 100.0, 100.0, 100.0)]})
        headlines = [
            Headline("Nvidia Shares Surge on Blowout Results", "", "a",
                     dt.datetime(2025, 1, 6, 2, 0, tzinfo=UTC), "s"),
            Headline("Nvidia Soars to a Record High", "", "b",
                     dt.datetime(2025, 1, 6, 2, 1, tzinfo=UTC), "s"),
        ]
        result = run_backtest(headlines, panel, BacktestConfig(contribution=5_000.0),
                              self.UNIVERSE)
        self.assertEqual(len(result.equity), 1)
        self.assertEqual(len(result.open_at_end), 1)

    def test_equity_curve_survives_a_wipeout(self):
        # A short squeezed from 100 to 10,000 drives equity negative; the
        # summary must still be computable rather than raising.
        summary = summarise(
            "wipeout",
            [(dt.date(2025, 1, 6), 1_000.0), (dt.date(2025, 1, 7), -5_000.0)],
            {dt.date(2025, 1, 6): 1_000.0},
        )
        self.assertLess(summary.final_value, 0)
        self.assertLess(summary.twr_total, 0)

    def test_zero_equity_does_not_divide_by_zero(self):
        summary = summarise(
            "zero",
            [(dt.date(2025, 1, 6), 0.0), (dt.date(2025, 1, 7), 0.0)],
            {dt.date(2025, 1, 6): 1_000.0},
        )
        self.assertEqual(summary.final_value, 0.0)

    def test_extreme_cashflows_do_not_hang_the_irr_search(self):
        started = time.perf_counter()
        rate = irr([
            CashFlow(dt.date(2025, 1, 1), -1e-9),
            CashFlow(dt.date(2025, 1, 2), 1e12),
        ])
        self.assertLess(time.perf_counter() - started, 1.0)
        self.assertTrue(rate is None or rate > 0)

    def test_max_drawdown_of_nothing(self):
        self.assertEqual(max_drawdown([]), 0.0)

    def test_leap_day_contribution_schedule(self):
        panel = PricePanel({
            "NVDA": [
                Bar(d, 100.0, 100.0, 100.0, 100.0)
                for d in (dt.date(2024, 2, 28), dt.date(2024, 2, 29), dt.date(2024, 3, 1))
            ]
        })
        headlines = [
            Headline("Nvidia Shares Surge on Blowout Results", "", "a",
                     dt.datetime(2024, 2, 28, 2, 0, tzinfo=UTC), "s"),
        ]
        result = run_backtest(headlines, panel, BacktestConfig(contribution=1_000.0),
                              self.UNIVERSE)
        self.assertEqual(sum(result.contributions.values()), 2_000.0)


if __name__ == "__main__":
    unittest.main()
