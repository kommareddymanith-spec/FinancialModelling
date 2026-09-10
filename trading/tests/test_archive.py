"""Historical headline archive loading across the formats vendors emit."""

from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import unittest

from wsj_headline_trader.archive import (
    ArchiveError,
    load_archive,
    load_archive_file,
    parse_timestamp,
)

UTC = dt.timezone.utc


def write(tmp: str, name: str, text: str) -> str:
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


class TimestampTests(unittest.TestCase):
    def test_iso_with_zulu(self):
        self.assertEqual(
            parse_timestamp("2025-03-04T14:30:00Z"),
            dt.datetime(2025, 3, 4, 14, 30, tzinfo=UTC),
        )

    def test_iso_with_offset_is_normalised_to_utc(self):
        self.assertEqual(
            parse_timestamp("2025-03-04T09:30:00-05:00"),
            dt.datetime(2025, 3, 4, 14, 30, tzinfo=UTC),
        )

    def test_naive_is_assumed_utc(self):
        self.assertEqual(
            parse_timestamp("2025-03-04 14:30:00"),
            dt.datetime(2025, 3, 4, 14, 30, tzinfo=UTC),
        )

    def test_date_only(self):
        self.assertEqual(parse_timestamp("2025-03-04"), dt.datetime(2025, 3, 4, tzinfo=UTC))

    def test_gdelt_style_compact(self):
        self.assertEqual(
            parse_timestamp("20250304143000"), dt.datetime(2025, 3, 4, 14, 30, tzinfo=UTC)
        )

    def test_unix_seconds_and_millis(self):
        self.assertEqual(parse_timestamp("1741098600"), parse_timestamp("1741098600000"))

    def test_rfc2822_as_used_by_rss(self):
        self.assertEqual(
            parse_timestamp("Tue, 04 Mar 2025 14:30:00 +0000"),
            dt.datetime(2025, 3, 4, 14, 30, tzinfo=UTC),
        )

    def test_unparsable_is_none(self):
        self.assertIsNone(parse_timestamp("sometime last week"))
        self.assertIsNone(parse_timestamp(""))


class JsonlTests(unittest.TestCase):
    def test_basic_jsonl(self):
        lines = [
            {"published_at": "2025-03-04T14:30:00Z", "title": "Nvidia Surges", "summary": "Up."},
            {"published_at": "2025-03-04T15:00:00Z", "title": "Boeing Plunges"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = write(tmp, "a.jsonl", "\n".join(json.dumps(line) for line in lines))
            headlines = load_archive_file(path)
        self.assertEqual([h.title for h in headlines], ["Nvidia Surges", "Boeing Plunges"])
        self.assertEqual(headlines[0].summary, "Up.")

    def test_alternative_key_names(self):
        record = {"date": "2025-03-04", "headline": "Ford Beats", "description": "Good.",
                  "url": "https://wsj.com/x", "domain": "wsj.com"}
        with tempfile.TemporaryDirectory() as tmp:
            path = write(tmp, "a.jsonl", json.dumps(record))
            headline = load_archive_file(path)[0]
        self.assertEqual(headline.title, "Ford Beats")
        self.assertEqual(headline.summary, "Good.")
        self.assertEqual(headline.link, "https://wsj.com/x")
        self.assertEqual(headline.source, "wsj.com")

    def test_json_array_is_accepted(self):
        payload = [{"published_at": "2025-03-04T14:30:00Z", "title": "Nvidia Surges"}]
        with tempfile.TemporaryDirectory() as tmp:
            path = write(tmp, "a.json", json.dumps(payload))
            self.assertEqual(len(load_archive_file(path)), 1)

    def test_records_without_a_title_or_date_are_dropped(self):
        lines = [
            {"published_at": "2025-03-04T14:30:00Z", "title": "Good"},
            {"published_at": "2025-03-04T14:30:00Z"},
            {"title": "No date"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = write(tmp, "a.jsonl", "\n".join(json.dumps(line) for line in lines))
            self.assertEqual([h.title for h in load_archive_file(path)], ["Good"])

    def test_a_corrupt_line_does_not_abort_the_file(self):
        text = json.dumps({"published_at": "2025-03-04T14:30:00Z", "title": "Good"}) + "\n{oops\n"
        with tempfile.TemporaryDirectory() as tmp:
            path = write(tmp, "a.jsonl", text)
            self.assertEqual(len(load_archive_file(path)), 1)


class OtherFormatTests(unittest.TestCase):
    def test_saved_rss_is_parsed_by_the_live_feed_code(self):
        fixture = os.path.join(os.path.dirname(__file__), "fixtures", "wsj_markets.xml")
        headlines = load_archive_file(fixture)
        self.assertEqual(len(headlines), 10)

    def test_csv_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(
                tmp,
                "a.csv",
                "date,title,summary\n2025-03-04T14:30:00Z,Nvidia Surges,Chips up\n",
            )
            headline = load_archive_file(path)[0]
        self.assertEqual(headline.title, "Nvidia Surges")

    def test_unsupported_extension_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(tmp, "a.parquet", "binary-ish")
            with self.assertRaises(ArchiveError):
                load_archive_file(path)


class LoadArchiveTests(unittest.TestCase):
    def test_directory_is_walked_and_sorted(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(tmp, "b.jsonl", json.dumps(
                {"published_at": "2025-03-05T10:00:00Z", "title": "Later", "link": "l2"}))
            write(tmp, "a.jsonl", json.dumps(
                {"published_at": "2025-03-04T10:00:00Z", "title": "Earlier", "link": "l1"}))
            headlines = load_archive(tmp)
        self.assertEqual([h.title for h in headlines], ["Earlier", "Later"])

    def test_duplicates_across_files_collapse(self):
        record = json.dumps(
            {"published_at": "2025-03-04T10:00:00Z", "title": "Same", "link": "same-url"}
        )
        with tempfile.TemporaryDirectory() as tmp:
            write(tmp, "a.jsonl", record)
            write(tmp, "b.jsonl", record)
            self.assertEqual(len(load_archive(tmp)), 1)

    def test_mixed_formats_combine(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(tmp, "a.jsonl", json.dumps(
                {"published_at": "2025-03-04T10:00:00Z", "title": "From JSON", "link": "j"}))
            fixture = os.path.join(os.path.dirname(__file__), "fixtures", "wsj_markets.xml")
            with open(fixture, "rb") as src, open(os.path.join(tmp, "b.xml"), "wb") as dst:
                dst.write(src.read())
            headlines = load_archive(tmp)
        self.assertEqual(len(headlines), 11)

    def test_a_missing_path_raises_rather_than_returning_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ArchiveError):
                load_archive(os.path.join(tmp, "missing.jsonl"))

    def test_an_empty_directory_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ArchiveError):
                load_archive(tmp)

    def test_a_directory_of_unreadable_files_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(tmp, "bad.parquet", "nope")
            with self.assertRaises(ArchiveError):
                load_archive(tmp)

    def test_an_unreadable_file_alongside_a_good_one_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            write(tmp, "good.jsonl", json.dumps(
                {"published_at": "2025-03-04T10:00:00Z", "title": "Good", "link": "g"}))
            write(tmp, "bad.parquet", "nope")
            self.assertEqual([h.title for h in load_archive(tmp)], ["Good"])


if __name__ == "__main__":
    unittest.main()
