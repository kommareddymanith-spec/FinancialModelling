"""Loading a historical headline archive.

The live algorithm reads RSS feeds, which carry only the current ~30 items. A
backtest therefore needs an archive from somewhere else, and this module is
deliberately permissive about its shape:

* ``*.jsonl`` / ``*.json`` -- one article per line (or a JSON array), with keys
  ``published_at``, ``title`` and optionally ``summary``, ``link``, ``source``.
  This is the format most news datasets and vendor exports reduce to.
* ``*.xml`` -- saved RSS or Atom, parsed by the same code the live feed uses.
  Drop a directory of archived feed snapshots in and it just works.
* ``*.csv`` -- a ``date,title,summary`` export.

See the README for where a real two-year WSJ archive can be obtained.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import logging
import os
from typing import Iterable, Iterator

from .feed import parse_feed
from .models import Headline

log = logging.getLogger(__name__)

_TITLE_KEYS = ("title", "headline", "text")
_SUMMARY_KEYS = ("summary", "description", "abstract", "snippet", "body")
_LINK_KEYS = ("link", "url", "guid", "id")
_DATE_KEYS = ("published_at", "published", "pubdate", "pub_date", "date", "timestamp", "time")
_SOURCE_KEYS = ("source", "feed", "publication", "domain")


class ArchiveError(ValueError):
    """Raised when an archive file cannot be read."""


def _first(record: dict, keys: Iterable[str]) -> str:
    """First non-empty value among ``keys``, matched case-insensitively."""
    lowered = {str(k).strip().lower(): v for k, v in record.items()}
    for key in keys:
        value = lowered.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def parse_timestamp(raw: str) -> _dt.datetime | None:
    """Parse the timestamp formats archives use, returning UTC-aware."""
    raw = raw.strip()
    if not raw:
        return None

    if raw.isdigit() and len(raw) in (10, 13):  # unix seconds or millis
        seconds = int(raw) / (1000.0 if len(raw) == 13 else 1.0)
        return _dt.datetime.fromtimestamp(seconds, _dt.timezone.utc)

    candidates = (
        raw,
        raw.replace("Z", "+00:00"),
        raw.replace(" ", "T", 1) if " " in raw else raw,
    )
    for candidate in candidates:
        try:
            parsed = _dt.datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=_dt.timezone.utc)
        return parsed.astimezone(_dt.timezone.utc)

    for fmt in ("%Y%m%d%H%M%S", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y-%m-%d", "%Y%m%d"):
        try:
            return _dt.datetime.strptime(raw, fmt).replace(tzinfo=_dt.timezone.utc)
        except ValueError:
            continue

    # Last resort: RFC 2822, as used by RSS.
    import email.utils

    try:
        parsed = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if parsed is None:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.timezone.utc)
    return parsed.astimezone(_dt.timezone.utc)


def _headline_from_record(record: dict, default_source: str) -> Headline | None:
    title = _first(record, _TITLE_KEYS)
    published = parse_timestamp(_first(record, _DATE_KEYS))
    if not title or published is None:
        return None
    return Headline(
        title=title,
        summary=_first(record, _SUMMARY_KEYS),
        link=_first(record, _LINK_KEYS),
        published_at=published,
        source=_first(record, _SOURCE_KEYS) or default_source,
    )


def _iter_json(path: str) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        first = handle.read(1)
        handle.seek(0)
        if first == "[":  # a single JSON array
            payload = json.load(handle)
            if not isinstance(payload, list):
                raise ArchiveError(f"{path}: expected a JSON array of articles")
            yield from (item for item in payload if isinstance(item, dict))
            return
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("%s:%d is not valid JSON, skipping (%s)", path, line_number, exc)
                continue
            if isinstance(item, dict):
                yield item


def load_archive_file(path: str) -> list[Headline]:
    """Load one archive file, dispatching on its extension."""
    extension = os.path.splitext(path)[1].lower()
    name = os.path.basename(path)

    if extension in (".jsonl", ".ndjson", ".json"):
        records = _iter_json(path)
    elif extension in (".xml", ".rss", ".atom"):
        with open(path, "rb") as handle:
            return parse_feed(handle.read(), source=name)
    elif extension in (".csv", ".tsv"):
        delimiter = "\t" if extension == ".tsv" else ","
        with open(path, newline="", encoding="utf-8-sig") as handle:
            records = list(csv.DictReader(handle, delimiter=delimiter))
    else:
        raise ArchiveError(f"{path}: unsupported archive format {extension!r}")

    headlines = []
    dropped = 0
    for record in records:
        headline = _headline_from_record(record, default_source=name)
        if headline is None:
            dropped += 1
            continue
        headlines.append(headline)
    if dropped:
        log.warning("%s: dropped %d record(s) with no title or timestamp", path, dropped)
    return headlines


def load_archive(*paths: str) -> list[Headline]:
    """Load every given file or directory into one de-duplicated, sorted list."""
    files: list[str] = []
    for path in paths:
        if os.path.isdir(path):
            for entry in sorted(os.listdir(path)):
                full = os.path.join(path, entry)
                if os.path.isfile(full):
                    files.append(full)
        elif os.path.isfile(path):
            files.append(path)
        else:
            # Loudly, not quietly: a mistyped archive path that returned an
            # empty list would look exactly like a strategy that never traded.
            raise ArchiveError(f"archive path does not exist: {path}")

    if not files:
        raise ArchiveError(f"no archive files found in {paths}")

    collected: dict[str, Headline] = {}
    for path in files:
        try:
            for headline in load_archive_file(path):
                collected.setdefault(headline.dedupe_key, headline)
        except (ArchiveError, OSError) as exc:
            log.warning("skipping %s: %s", path, exc)

    if not collected:
        raise ArchiveError(
            f"no readable headlines in {len(files)} file(s); check the archive format"
        )

    headlines = sorted(collected.values(), key=lambda h: h.published_at)
    log.info(
        "loaded %d unique headline(s) from %d file(s)%s",
        len(headlines),
        len(files),
        f", {headlines[0].published_at:%Y-%m-%d} to {headlines[-1].published_at:%Y-%m-%d}"
        if headlines
        else "",
    )
    return headlines
