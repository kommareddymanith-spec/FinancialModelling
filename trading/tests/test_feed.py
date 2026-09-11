"""Feed fetching, parsing and time-window filtering."""

from __future__ import annotations

import datetime as dt
import os
import unittest

from wsj_headline_trader.feed import (
    FeedError,
    collect_headlines,
    headlines_from_files,
    parse_feed,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
NOW = dt.datetime(2026, 9, 10, 15, 30, tzinfo=dt.timezone.utc)


def fixture(name: str) -> str:
    return os.path.join(FIXTURES, name)


def read(name: str) -> bytes:
    with open(fixture(name), "rb") as handle:
        return handle.read()


class ParseFeedTests(unittest.TestCase):
    def test_parses_rss_items(self):
        headlines = parse_feed(read("wsj_markets.xml"), source="markets")
        self.assertEqual(len(headlines), 10)
        first = headlines[0]
        self.assertIn("Nvidia", first.title)
        self.assertTrue(first.link.startswith("https://www.wsj.com/"))
        self.assertEqual(first.source, "markets")
        self.assertEqual(first.published_at.tzinfo, dt.timezone.utc)

    def test_parses_atom_entries_and_link_attribute(self):
        headlines = parse_feed(read("wsj_atom.xml"), source="atom")
        self.assertEqual(len(headlines), 3)
        self.assertTrue(all(h.link.startswith("https://www.wsj.com/atom/") for h in headlines))

    def test_text_combines_title_and_summary(self):
        headline = parse_feed(read("wsj_markets.xml"))[0]
        self.assertIn(headline.title, headline.text)
        self.assertIn(headline.summary, headline.text)

    def test_malformed_xml_raises(self):
        with self.assertRaises(FeedError):
            parse_feed(read("malformed.xml"), source="broken")

    def test_undated_entries_are_dropped(self):
        payload = b"""<rss><channel>
            <item><title>Dated</title><pubDate>Thu, 10 Sep 2026 15:00:00 +0000</pubDate></item>
            <item><title>Undated</title></item>
        </channel></rss>"""
        headlines = parse_feed(payload)
        self.assertEqual([h.title for h in headlines], ["Dated"])

    def test_naive_timestamps_are_treated_as_utc(self):
        payload = b"""<rss><channel><item>
            <title>Naive</title><pubDate>Thu, 10 Sep 2026 15:00:00</pubDate>
        </item></channel></rss>"""
        headline = parse_feed(payload)[0]
        self.assertEqual(headline.published_at, dt.datetime(2026, 9, 10, 15, 0, tzinfo=dt.timezone.utc))


class CollectHeadlinesTests(unittest.TestCase):
    def test_filters_to_the_window(self):
        headlines, errors = collect_headlines(
            feeds=["markets"],
            window_minutes=60,
            now=NOW,
            fetcher=headlines_from_files([fixture("wsj_markets.xml")]),
        )
        self.assertEqual(errors, [])
        # 10 items in the fixture, one published 92 minutes ago.
        self.assertEqual(len(headlines), 9)
        self.assertTrue(all(h.published_at >= NOW - dt.timedelta(minutes=60) for h in headlines))

    def test_returns_newest_first(self):
        headlines, _ = collect_headlines(
            feeds=["markets"],
            window_minutes=60,
            now=NOW,
            fetcher=headlines_from_files([fixture("wsj_markets.xml")]),
        )
        stamps = [h.published_at for h in headlines]
        self.assertEqual(stamps, sorted(stamps, reverse=True))

    def test_deduplicates_across_feeds(self):
        headlines, _ = collect_headlines(
            feeds=["a", "b"],
            window_minutes=60,
            now=NOW,
            fetcher=headlines_from_files([fixture("wsj_markets.xml"), fixture("wsj_markets.xml")]),
        )
        self.assertEqual(len(headlines), 9)

    def test_a_failing_feed_does_not_abort_the_others(self):
        def fetcher(url: str) -> bytes:
            if url == "bad":
                raise FeedError("boom")
            return read("wsj_markets.xml")

        headlines, errors = collect_headlines(
            feeds=["bad", "good"], window_minutes=60, now=NOW, fetcher=fetcher
        )
        self.assertEqual(len(errors), 1)
        self.assertEqual(len(headlines), 9)

    def test_malformed_feed_is_reported_as_an_error(self):
        headlines, errors = collect_headlines(
            feeds=["broken"],
            window_minutes=60,
            now=NOW,
            fetcher=headlines_from_files([fixture("malformed.xml")]),
        )
        self.assertEqual(headlines, [])
        self.assertEqual(len(errors), 1)

    def test_future_dated_items_are_excluded(self):
        payload = b"""<rss><channel><item>
            <title>Tomorrow</title><pubDate>Fri, 11 Sep 2026 15:00:00 +0000</pubDate>
        </item></channel></rss>"""
        headlines, _ = collect_headlines(
            feeds=["x"], window_minutes=60, now=NOW, fetcher=lambda url: payload
        )
        self.assertEqual(headlines, [])

    def test_window_is_respected(self):
        # The markets fixture has one article inside 10 minutes (5m) and the
        # next at 12m, so a narrower window must return exactly one.
        headlines, _ = collect_headlines(
            feeds=["markets"],
            window_minutes=10,
            now=NOW,
            fetcher=headlines_from_files([fixture("wsj_markets.xml")]),
        )
        self.assertEqual(len(headlines), 1)


if __name__ == "__main__":
    unittest.main()
