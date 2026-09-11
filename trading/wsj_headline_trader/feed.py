"""Reading WSJ headlines out of the public Dow Jones RSS feeds.

The feeds are plain RSS 2.0 (a couple of the newer endpoints serve Atom), so
parsing is done with the standard library only -- no third-party dependency is
needed to run the algorithm.
"""

from __future__ import annotations

import datetime as _dt
import email.utils
import logging
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from typing import Callable, Iterable, Sequence

from .models import Headline

log = logging.getLogger(__name__)

#: Public WSJ feeds, widest-to-narrowest. Markets and Business carry the bulk
#: of single-company headlines; World News is included because it is where
#: regulatory and antitrust stories tend to land first.
DEFAULT_FEEDS: tuple[str, ...] = (
    "https://feeds.content.dowjones.io/public/rss/RSSMarketsMain",
    "https://feeds.content.dowjones.io/public/rss/WSJcomUSBusiness",
    "https://feeds.content.dowjones.io/public/rss/RSSWSJD",
    "https://feeds.content.dowjones.io/public/rss/RSSWorldNews",
)

USER_AGENT = "wsj-headline-trader/1.0 (+https://github.com/kommareddymanith-spec/financialmodelling)"

#: Largest feed body accepted. The real feeds are well under 200KB; reading
#: an unbounded response would let a broken or hostile endpoint exhaust
#: memory before a single headline was parsed.
MAX_FEED_BYTES = 8 * 1024 * 1024

#: Callable that turns a feed URL into raw bytes. Swapped out in tests and by
#: ``--fixture`` on the CLI so the algorithm can run with no network at all.
Fetcher = Callable[[str], bytes]


class FeedError(RuntimeError):
    """Raised when a feed cannot be fetched or parsed."""


def fetch_feed(url: str, timeout: float = 15.0) -> bytes:
    """Download a feed. Raises :class:`FeedError` on any transport failure."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            # Read one byte past the cap so an oversized body is detected
            # rather than silently truncated into malformed XML.
            payload = response.read(MAX_FEED_BYTES + 1)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        raise FeedError(f"could not fetch {url}: {exc}") from exc
    if len(payload) > MAX_FEED_BYTES:
        raise FeedError(f"{url} returned more than {MAX_FEED_BYTES} bytes; refusing it")
    return payload


def _local_name(tag: str) -> str:
    """Strip any XML namespace from an element tag."""
    return tag.rsplit("}", 1)[-1]


def _child_text(element: ET.Element, *names: str) -> str:
    """First non-empty text among the named children, namespace-insensitive."""
    wanted = {name.lower() for name in names}
    for child in element:
        if _local_name(child.tag).lower() in wanted:
            text = (child.text or "").strip()
            if text:
                return text
            # Atom links carry the URL in an attribute rather than as text.
            href = child.get("href", "").strip()
            if href:
                return href
    return ""


def _parse_timestamp(raw: str) -> _dt.datetime | None:
    """Parse an RSS (RFC 2822) or Atom (ISO 8601) timestamp as UTC-aware."""
    if not raw:
        return None
    parsed: _dt.datetime | None = None
    try:
        parsed = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        parsed = None
    if parsed is None:
        try:
            parsed = _dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.timezone.utc)
    return parsed.astimezone(_dt.timezone.utc)


def parse_feed(payload: bytes, source: str = "") -> list[Headline]:
    """Parse RSS/Atom bytes into :class:`Headline` objects.

    Entries without a parsable timestamp are dropped: the algorithm only ever
    trades a bounded time window, so an article we cannot date is unusable.
    """
    if len(payload) > MAX_FEED_BYTES:
        raise FeedError(
            f"feed from {source or 'feed'} is {len(payload)} bytes, over the "
            f"{MAX_FEED_BYTES} byte limit"
        )

    # RSS and Atom have no need of a DTD, and a DTD that declares entities is
    # the billion-laughs vector: a few hundred bytes expanding into megabytes
    # of title text. Whether the underlying expat stops it depends on the
    # system library version, so refuse it here rather than hope.
    head = payload[:4096].lower()
    if b"<!doctype" in head and b"<!entity" in payload[:65536].lower():
        raise FeedError(
            f"feed from {source or 'feed'} declares XML entities; refusing it"
        )

    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise FeedError(f"malformed feed XML from {source or 'feed'}: {exc}") from exc

    headlines: list[Headline] = []
    entries_seen = 0
    for element in root.iter():
        if _local_name(element.tag) not in ("item", "entry"):
            continue
        entries_seen += 1
        title = _child_text(element, "title")
        if not title:
            continue
        published = _parse_timestamp(
            _child_text(element, "pubDate", "published", "updated", "date")
        )
        if published is None:
            log.debug("skipping undated entry %r from %s", title, source)
            continue
        headlines.append(
            Headline(
                title=title,
                summary=_child_text(element, "description", "summary", "content"),
                link=_child_text(element, "link", "guid", "id"),
                published_at=published,
                source=source,
            )
        )

    if entries_seen == 0:
        # Well-formed XML with no items at all is usually not a feed: an HTML
        # error page, a redirect notice, or a moved endpoint. Returning an
        # empty list is the safe outcome, but staying silent would let a dead
        # feed URL look like a quiet news hour indefinitely.
        log.warning(
            "%s parsed as XML but contained no RSS items or Atom entries; "
            "is the URL still a feed?",
            source or "feed",
        )
    return headlines


def collect_headlines(
    feeds: Sequence[str] = DEFAULT_FEEDS,
    window_minutes: int = 60,
    now: _dt.datetime | None = None,
    fetcher: Fetcher = fetch_feed,
) -> tuple[list[Headline], list[str]]:
    """Fetch every feed and return the articles published in the last window.

    Returns ``(headlines, errors)``. A feed that fails is reported but does not
    abort the run -- trading on three of four feeds beats trading on none.
    Headlines are de-duplicated across feeds and returned newest first.
    """
    now = _utc(now)
    cutoff = now - _dt.timedelta(minutes=window_minutes)

    collected: dict[str, Headline] = {}
    errors: list[str] = []
    for url in feeds:
        try:
            entries = parse_feed(fetcher(url), source=url)
        except FeedError as exc:
            log.warning("%s", exc)
            errors.append(str(exc))
            continue
        for headline in entries:
            if not cutoff <= headline.published_at <= now:
                continue
            collected.setdefault(headline.dedupe_key, headline)

    fresh = sorted(collected.values(), key=lambda h: h.published_at, reverse=True)
    log.info(
        "collected %d headlines from %d feed(s) published since %s",
        len(fresh),
        len(feeds) - len(errors),
        cutoff.isoformat(),
    )
    return fresh, errors


def _utc(now: _dt.datetime | None) -> _dt.datetime:
    """Normalise an optional caller-supplied 'now' to UTC-aware."""
    if now is None:
        return _dt.datetime.now(_dt.timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=_dt.timezone.utc)
    return now.astimezone(_dt.timezone.utc)


def headlines_from_files(paths: Iterable[str]) -> Fetcher:
    """Build a fetcher that serves saved feed XML instead of hitting the network.

    The returned fetcher ignores the URL it is handed and walks the given files
    in order, which is what ``--fixture`` uses for offline dry runs.
    """
    remaining = []
    for path in paths:
        with open(path, "rb") as handle:
            remaining.append(handle.read())

    def fetcher(url: str) -> bytes:
        if not remaining:
            raise FeedError(f"no fixture payload left for {url}")
        return remaining.pop(0)

    return fetcher
