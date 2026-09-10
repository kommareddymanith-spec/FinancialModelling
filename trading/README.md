# WSJ Headline Trader

Trades the companies **featured the most** in Wall Street Journal headlines over
the last hour: buys the ones written about positively, shorts the ones written
about negatively.

Pure standard library — no third-party packages needed to run it.

## How it works

```
WSJ RSS feeds  ──▶  last-hour window  ──▶  company recognition  ──▶  rank by
                                                                     mention count
                                                                          │
     orders  ◀──  size by conviction  ◀──  buy / short / pass  ◀──  score the tone
```

1. **Read** the four public Dow Jones RSS feeds (Markets, US Business, Tech,
   World News) and keep only articles published inside the window.
2. **Recognise companies** in each headline and summary — by name
   (`Nvidia`, `J.P. Morgan`, `P&G`) or by symbol (`(NVDA)`, `$TSLA`,
   `Nasdaq: AAPL`) — against a bundled 207-company universe.
3. **Rank** by how many distinct articles name each company. That ranking *is*
   "featured the most"; conviction breaks ties.
4. **Score the tone** of those articles with a weighted finance lexicon.
5. **Decide**: positive net tone → buy, negative → short, muted or two-sided → pass.
6. **Size** each position by conviction and submit.

### Reading a headline correctly

Two things that a naive word-count gets wrong, and this handles:

**Company names that are ordinary words.** *Target*, *Visa*, *Gap*, *Meta*,
*Apple*, *Shell*. These only count when the sentence is actually about the
company — the name is possessive, is followed by something only a company has
or does, or is preceded by a descriptor noun:

| Headline | Trades |
|---|---|
| `Target Shares Slide as Retailer Cuts Guidance` | ✅ TGT |
| `Shares of Target Fall 8% After Profit Warning` | ✅ TGT |
| `Target's Quarterly Loss Widens` | ✅ TGT |
| `Investors Target Small-Cap Stocks as Rally Broadens` | ❌ nothing |
| `Gap Widens Between Rich and Poor` | ❌ nothing |
| `The Apple Harvest Was Poor This Year` | ❌ nothing |

**Phrases and negation.** Terms are matched longest-first and non-overlapping,
and a negator within three tokens flips polarity:

| Headline | Score |
|---|---|
| `Boeing Posts Record Loss` | −2.0 (the phrase, not `record` + `loss` cancelling) |
| `Target Cuts Guidance` | −2.0 (not a neutral "cut") |
| `Apple Strikes a Deal With Suppliers` | +1.0 (a deal, not a labour strike) |
| `Ford Doesn't Beat Expectations` | −2.0 (flipped) |
| `Pfizer Will Not Cut Guidance` | +2.0 (flipped) |

## Usage

Dry run against the live feeds — reports what it *would* do and places nothing:

```bash
python -m wsj_headline_trader
```

```
WSJ headline trader -- 2026-09-10 15:30 UTC
  window          : last 60 minutes
  headlines read  : 14
  mode            : DRY RUN

  Most-featured companies
   #  TICKER  MENTIONS    TONE  AGREE  ACTION
   1  NVDA           3   +1.44   100%  BUY   2,000.00
   2  BA             3   -1.21   100%  SHORT 1,903.94
   3  TGT            2   -1.10   100%  SHORT 1,096.06
   4  PFE            2   +1.08   100%  pass (max_total_notional reached)
   5  AAPL           1   -1.33   100%  pass (only 1 mention(s), need 2)
```

Paper trade for real, with a book that tracks positions:

```bash
python -m wsj_headline_trader --live --broker paper -v
```

Alpaca paper account:

```bash
export APCA_API_KEY_ID=...  APCA_API_SECRET_KEY=...
python -m wsj_headline_trader --live --broker alpaca
```

Offline, using the saved feeds in `tests/fixtures/`:

```bash
python -m wsj_headline_trader \
    --fixture tests/fixtures/wsj_markets.xml \
    --fixture tests/fixtures/wsj_business.xml \
    --as-of 2026-09-10T15:30:00Z --json
```

### Flags worth knowing

| Flag | Default | Meaning |
|---|---|---|
| `--window-minutes` | `60` | How far back to read. |
| `--top` | `5` | How many of the most-featured companies to consider. |
| `--min-mentions` | `2` | A company needs this many articles to qualify. |
| `--min-sentiment` | `0.5` | Neutral band; inside it, pass. |
| `--min-agreement` | `0.6` | Minimum share of articles agreeing on direction. |
| `--notional` | `1000` | Base size per trade, before conviction scaling. |
| `--max-notional` | `5000` | Total the run may deploy. |
| `--broker` | `paper` | `paper` or `alpaca`. |
| `--live` | off | Actually submit. Without it, nothing is placed. |
| `--real-money` | off | With `--broker alpaca`, use the live endpoint. |
| `--universe` | bundled | Your own company→ticker JSON. |

### As a library

```python
from wsj_headline_trader import (
    AlgorithmConfig, PaperBroker, StrategyConfig, WSJHeadlineAlgorithm, format_report,
)

algo = WSJHeadlineAlgorithm(
    AlgorithmConfig(
        window_minutes=60,
        dry_run=False,
        strategy=StrategyConfig(top_n=5, min_mentions=2, notional_per_trade=1_000),
    ),
    broker=PaperBroker(),
)
print(format_report(algo.run()))
```

## Risk controls

Every default errs toward not trading:

- **Dry run unless told otherwise.** `--live` is required to place anything, and
  Alpaca defaults to the paper endpoint even then.
- **Coverage floor.** One passing mention is not a signal (`--min-mentions`).
- **Neutral band.** Muted tone is a pass, not a small position (`--min-sentiment`).
- **Two-sided coverage is a pass.** If articles disagree on direction, the
  algorithm declines rather than trading the average (`--min-agreement`).
- **Size caps.** Per-trade conviction is clamped to 0.5–2.0×, and the run stops
  deploying at `--max-notional`.
- **Shorts are never guessed.** Notional short sales aren't supported by the
  broker API, so a short with no available price is refused, not estimated.
- **Feed failures degrade, they don't guess.** One dead feed is logged and the
  run continues; all feeds dead means no trades.
- **Every order carries its rationale** — mention count, tone, matched terms and
  headline titles — in its metadata.

## Scheduling

A run is stateless: it reads the window, decides, submits. Re-running inside the
same window acts on the same headlines again and will *add* to positions. Run it
on a cadence at least as long as the window:

```cron
# Hourly on the half hour during US market hours
30 14-20 * * 1-5  cd /path/to/trading && /usr/bin/python3 -m wsj_headline_trader --live
```

Nothing here closes positions — this is an entry-signal engine. Exits (a
time stop, a trailing stop, next-day close) are yours to add.

## Tests

```bash
cd trading && PYTHONPATH=. python3 -m unittest discover -s tests -t .
```

135 tests, no network required — the fixtures in `tests/fixtures/` are synthetic
feeds written for the suite, not WSJ content.

## Layout

| File | Role |
|---|---|
| `wsj_headline_trader/feed.py` | Fetch and parse RSS/Atom, filter to the window |
| `wsj_headline_trader/universe.py` | Company recognition and ambiguous-name handling |
| `wsj_headline_trader/sentiment.py` | Weighted lexicon, phrases, negation |
| `wsj_headline_trader/strategy.py` | Ranking, gating, sizing |
| `wsj_headline_trader/broker.py` | `PaperBroker` and `AlpacaBroker` |
| `wsj_headline_trader/algorithm.py` | Orchestration and reporting |
| `wsj_headline_trader/cli.py` | Command line interface |
| `wsj_headline_trader/data/universe.json` | 207 companies, editable |

## Caveats

Worth being straight about, since this can place real orders:

- **Sentiment is attributed per article, not per clause.** "Ford Beats, GM
  Misses" gives both names the same blended score. The agreement gate keeps
  genuinely mixed coverage from trading, but sentence-level attribution is not
  implemented.
- **Headline frequency is not alpha on its own.** RSS headlines are public and
  already priced; the most-covered name is usually the most-covered *because*
  it already moved. Treat this as a research harness, and backtest before
  risking capital.
- **A lexicon is not a language model.** It is used here because every order is
  traceable to the exact words that caused it, which a model would not give you.
- **Shorting needs a margin-enabled account** and a locatable borrow; the broker
  reports the rejection rather than working around it.
