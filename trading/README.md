# WSJ Headline Trader

Trades the companies **featured the most** in Wall Street Journal headlines over
the last hour: buys the ones written about positively, shorts the ones written
about negatively.

Pure standard library — no third-party packages needed to run it.
Tested on CPython 3.10, 3.11, 3.12 and 3.13.

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

## Backtesting

```bash
python -m wsj_headline_trader.backtest_cli --benchmark-only \
    --start 2024-09-01 --end 2026-08-31
```

### The benchmark: $1,000/month into the S&P 500

This is a real result, computed from the vendored monthly index series
(`data/sp500_monthly.csv`, provenance in `data/SOURCES.md`):

```
                                Index DCA (1,000/mo)
------------------------------------------------------
Period                      2024-09-01 to 2026-08-01
Contributed                                24,000.00
Final value                                28,869.72
Profit                                      4,869.72
Profit on contributions                      +20.29%
Money-weighted return p.a.                   +20.55%
Time-weighted return total                   +37.18%
Time-weighted return p.a.                    +17.96%
Max drawdown                                 -11.08%
Volatility p.a.                              +11.51%
Sharpe                                          1.50
Purchases                                         24
Units held                                    3.7438
Average cost                                6,410.58
```

Twenty-four monthly purchases at an average cost of 6,410.58 against a closing
level of 7,711.32. Three caveats, all of which make this number *conservative*
as a stand-in for an index fund:

- **Price return only.** The source's dividend column is empty for recent
  months, so this excludes dividends. A real fund reinvests them, adding
  roughly 1.2–1.5% a year.
- **Monthly average prices, not month-end closes.** The Shiller series averages
  each month's daily closes. For a monthly savings plan that is a defensible
  purchase price, but it is not a month-end mark.
- **Drawdown is measured on monthly data**, so it understates the true
  intramonth low.

### The strategy leg: why there is no number here

**The strategy cannot be backtested on the data it trades on.** WSJ RSS feeds
serve only the current ~30 items and carry no history at all, so there is
nothing to replay. This is a property of the feeds, not of this environment.

The engine to run it is built, tested and ready — it just needs an archive:

```bash
python -m wsj_headline_trader.backtest_cli \
    --archive path/to/wsj_headlines.jsonl \
    --prices  path/to/daily_bars.csv \
    --start 2024-09-01 --end 2026-08-31 --trades
```

**Historical headline data** — the options, with their real costs:

| Source | Covers | Catch |
|---|---|---|
| Dow Jones Factiva / DNA | Full WSJ text, decades | Paid licence; the canonical answer |
| RavenPack, Refinitiv News Analytics | Pre-scored WSJ sentiment | Paid; sentiment already computed |
| [GDELT 2.0](https://www.gdeltproject.org/) | Titles + URLs by domain, 2015– | Free; titles only, some gaps |
| Wayback Machine snapshots of the RSS URLs | Whatever was captured | Free; irregular capture cadence |
| Common Crawl `CC-NEWS` | Raw article HTML | Free; heavy ETL to get titles |

Archives load from `.jsonl`, `.json`, `.csv` or saved `.xml` feeds — see
`wsj_headline_trader/archive.py`. The minimum per record is a timestamp and a
title:

```json
{"published_at": "2025-03-04T14:30:00Z", "title": "Nvidia Shares Surge on Blowout Results"}
```

**Price data** needs a long-format panel covering every ticker the strategy
might trade:

```csv
date,symbol,open,high,low,close
2025-03-04,NVDA,112.50,114.20,111.80,113.90
```

`open`/`high`/`low` are optional and fall back to the close, though stops and
targets need highs and lows to mean anything.

### What the engine does, and what it refuses to do

- **No look-ahead.** A signal decided at time *t* fills at the open of the
  first session whose bell is strictly after *t*. A headline published mid-session
  cannot be traded at that session's open.
- **Same cash flows as the benchmark.** The strategy receives the same
  $1,000/month. Comparing a fully-funded strategy against a plan that drip-feeds
  cash would flatter whichever got its money in first.
- **Costs are charged.** Slippage (default 5bp each way), commission, and short
  borrow (default 3%/yr, accrued daily). A strategy that turns over this often
  is not cost-insensitive.
- **Unfilled orders are counted, not hidden.** Signals in symbols with no price
  data, or with no cash behind them, appear in the `skipped` breakdown, so
  coverage gaps cannot pass for good behaviour.
- **Stops beat targets** when one bar spans both, because the intrabar path is
  unknown.
- **Reported P&L is real cash.** `net_pnl` summed over closed trades plus
  contributions equals the final equity, exactly — asserted in the test suite,
  including with costs on and on the short side.

Two measures are reported because they answer different questions:
money-weighted return (IRR) is what your contributions actually earned and is
the right cross-plan comparison; time-weighted return strips contributions out
and is what drawdown, volatility and Sharpe are computed from.

## Running on an Alpaca account

### 1. Get keys

Sign up at [alpaca.markets](https://alpaca.markets), and in the dashboard
switch to **Paper Trading** before generating keys — paper and live keys are
different, and a paper key cannot touch real money whatever flags you pass.
Generate a key pair and export both:

```bash
export APCA_API_KEY_ID=PK...
export APCA_API_SECRET_KEY=...
```

Put them in a file the scheduler can read (`~/.wsj-trader.env`, mode `600`)
rather than in your shell history. They are never written to the signal log or
any artifact this repo produces.

### 2. Preflight

Run this before trusting anything to a schedule, and again after any
credential change:

```bash
python -m wsj_headline_trader --check --broker alpaca
```

```
Preflight

  broker            : alpaca
  endpoint          : https://paper-api.alpaca.markets
  account status    : ACTIVE
  buying power      : 200000
  shorting enabled  : True
  market            : OPEN (next open 2026-09-11T13:30:00Z)
  open positions    : none
  feeds reachable   : 4/4
  headlines in  60m : 12

  Ready. Nothing was traded by this check.
```

It exits non-zero and says what is wrong if credentials are rejected, the
account is blocked, **shorting is disabled**, or no feed can be read. That last
one matters: credentials are useless without headlines, and an outbound
firewall blocking `feeds.content.dowjones.io` is a silent failure otherwise.

**Shorting needs a margin account.** On a cash account every short signal is
refused and you get the long half of a long/short strategy — which is a
different strategy. Preflight fails loudly rather than letting you discover
this from a week of one-sided fills.

### 3. Dry run, then paper

```bash
# nothing is sent; prints what it would do
python -m wsj_headline_trader --broker alpaca -v

# sends to the paper endpoint
python -m wsj_headline_trader --live --broker alpaca -v --log-signals signals.jsonl
```

`--live` is required before any order is sent, and Alpaca stays on its paper
endpoint unless you also pass `--real-money`.

### 4. Schedule it

```cron
# hourly, half past, US market hours, weekdays
30 14-20 * * 1-5  . $HOME/.wsj-trader.env && cd /path/to/trading && \
                  /usr/bin/python3 -m wsj_headline_trader --live --broker alpaca \
                  --log-signals $HOME/wsj-signals.jsonl >> $HOME/wsj.log 2>&1
```

Two defaults exist because a schedule breaks things a single run does not:

- **It will not trade a closed market.** A market order sent when the venue is
  shut is at best queued to an open hours away, acting on headlines that
  stopped being news overnight. The run checks Alpaca's clock and declines,
  recording `market closed` in the report. `--queue-when-closed` overrides it.
  An *unknown* clock (the call failed) also declines — unknown is not
  permission.
- **It will not stack positions.** Runs are stateless, so without this an
  hourly schedule pyramids into any story that stays in the news: three days of
  Nvidia headlines becomes a position many times the intended size. The run
  reads open positions and skips symbols already held, reporting
  `already holding`. `--allow-stacking` overrides it.

Both guards only engage on a live submit. Dry runs never call the broker.

### 5. What it does and does not manage

| | |
|---|---|
| Opens positions from signals | ✅ |
| Skips symbols already held | ✅ |
| Refuses to trade a closed market | ✅ |
| Sizes by conviction, capped per run | ✅ |
| **Closes positions** | ❌ **nothing here exits a trade** |
| Stop losses on the live path | ❌ backtest only |
| Reconciles against manual trades | ❌ |

**The exit gap is the one that matters.** The backtest models a time stop and
optional stop/target, but the live algorithm only opens positions — it will
never close one. Before running this beyond a short paper experiment you need
an exit, either as a bracket on the entry or a second scheduled job that closes
anything held longer than N sessions. Until then, Alpaca's dashboard is your
only exit.

A practical first experiment: preflight, then run on paper for two weeks with
`--log-signals`, and close positions by hand. That tells you whether the
signals are sane before you automate anything irreversible.

## Testing it in a dummy market

Three ways to exercise the algorithm without risking money, in increasing
order of realism.

### 1. Simulated market (no account, no data, instant)

A synthetic market — random price paths and generated headlines — with the
*real* pipeline replayed over it, so there is no second implementation that
could disagree with production.

```bash
python -m wsj_headline_trader.simulate_cli --days 252 --edge 0.01 --seed 7 --trades
```

The point is not the return figure; synthetic returns say nothing about real
markets. The point is two experiments you cannot run any other way.

**The null test.** Set `--edge 0` and the headlines are pure noise,
uncorrelated with prices. A correct engine must then earn roughly zero before
costs, and roughly minus the costs after:

```bash
python -m wsj_headline_trader.simulate_cli --edge 0 --seeds 30
```
```
  Strategy time-weighted return
    mean        -4.46%   (standard error 3.36%)
    spread      -36.6% to +29.6%   (stdev 18.39%)
    profitable  37% of runs
```

A mean within a couple of standard errors of zero is the **correct** result. A
clear profit here would mean the engine is manufacturing returns — look-ahead,
double-counted P&L, a sizing bug — and is the first thing to check after any
change to the backtest.

**The power test.** Give the headlines real predictive content and the engine
must find it. `--sweep` walks the edge from nothing to obvious:

```bash
python -m wsj_headline_trader.simulate_cli --sweep --seeds 20
```
```
  edge/event   mean TWR    median    stderr  profitable   vs null
  0.000         -3.34%     -8.47%     4.08%        35%
  0.002         +4.18%     -0.79%     4.22%        50%   +1.8 SE
  0.005        +15.85%    +12.30%     4.44%        75%   +4.3 SE
  0.010        +37.01%    +36.10%     5.07%       100%   +8.0 SE
  0.020        +85.49%    +86.69%     6.54%       100%  +13.6 SE
  0.030       +143.33%   +141.96%     8.60%       100%  +17.0 SE
```

Read that as a **requirement**: the strategy needs roughly **0.5% of genuine
predictive drift per news event** before it clears noise and costs. Below
that, it cannot tell itself apart from the null. Whether WSJ headlines carry
half a percent is the open question this repo cannot answer.

The other thing the null row shows is variance: ±18% standard deviation over a
single simulated year, from noise alone. That is why a two-year live result
would not distinguish skill from luck.

### 2. Paper broker against live headlines (minutes to set up)

Real WSJ feeds, fake fills, no account needed:

```bash
python -m wsj_headline_trader --live --broker paper -v
```

By default the paper broker prices everything at a flat notional, so positions
never move — fine for checking plumbing, useless for P&L. Mark against the real
market instead (still risks nothing, needs Alpaca credentials for the data):

```bash
python -m wsj_headline_trader --live --broker paper --paper-marks alpaca
```

### 3. Alpaca paper account (the real dummy market)

Real prices, real headlines, real order lifecycle, fake money. This is the
closest thing to a live test and the one worth running for weeks:

```bash
export APCA_API_KEY_ID=...  APCA_API_SECRET_KEY=...
python -m wsj_headline_trader --live --broker alpaca --log-signals signals.jsonl
```

Alpaca defaults to its paper endpoint, so `--real-money` is required before
anything touches a funded account. Adding `--log-signals` means the same runs
accumulate the history that `pine_cli` turns into a TradingView chart.

### Which to use

| | Simulated | Paper broker | Alpaca paper |
|---|---|---|---|
| Needs an account | no | no | free |
| Needs network | no | yes | yes |
| Real prices | no | with `--paper-marks alpaca` | yes |
| Real headlines | no | yes | yes |
| Tests engine correctness | **yes** | no | no |
| Tests the actual edge | no | partly | **yes, given time** |
| Time to a result | seconds | one run | weeks |

The simulator answers "is my engine correct". Only Alpaca paper, run for long
enough, starts to answer "does this strategy work".

## TradingView

**The strategy cannot run on TradingView.** Pine Script executes inside
TradingView's sandbox with no network access: it cannot fetch an RSS feed,
parse XML, or read a headline. The company-recognition and sentiment half of
this algorithm is not expressible in Pine, and no translation changes that.

What does work is splitting the work at the capability boundary — Python has
the network, TradingView has the chart:

```
Python                                TradingView
------                                -----------
read WSJ feeds                        plots the signals
find companies, score tone     -->    runs the Strategy Tester
emit a Pine script with the           fires alerts
signals baked in as data
```

`wsj_headline_trader/pine.py` renders a self-contained Pine v5 `strategy()`
with the signals embedded. One script covers every symbol it was generated
with — it filters its own data by `syminfo.ticker`.

### Getting signals onto a chart

Since there is no headline archive, accumulate a signal history from scheduled
live runs — this is the practical path:

```bash
# each scheduled run appends its signals
python -m wsj_headline_trader --log-signals signals.jsonl

# turn the accumulated history into a Pine script
python -m wsj_headline_trader.pine_cli --signals signals.jsonl --out wsj.pine
```

Then paste `wsj.pine` into TradingView's Pine Editor and add it to a chart for
any of the symbols it names. If you *do* have an archive, a replay exports
directly:

```bash
python -m wsj_headline_trader.backtest_cli \
    --archive headlines.jsonl --prices bars.csv --pine wsj.pine
```

### What you get, and what you don't

| | |
|---|---|
| Signals plotted on the chart | ✅ |
| TradingView Strategy Tester performance | ✅ |
| Alerts on signal bars | ✅ (`alertcondition`) |
| One script across many symbols | ✅ filters on `syminfo.ticker` |
| Live headline reading inside Pine | ❌ impossible — no network in Pine |
| Auto-refreshing signals | ❌ regenerate the script |

The embedded signals are a **snapshot frozen at generation time**. Regenerate
to refresh — there is no way for the script to pull new ones itself.

Orders are placed on bar close, so TradingView fills them at the next bar's
open, which matches the no-look-ahead rule the Python backtest uses.

Two limits worth knowing: signals are capped at 5,000 (`--max-signals`, most
recent kept) because Pine caps array size and script length; and the generated
Pine is written to the v5 language reference but **cannot be compiled outside
TradingView**, so its syntax is verified by pasting it in, not by this repo's
tests. The tests check structure, symbol filtering, escaping, ordering and caps.

## Stress testing

The suite includes `tests/test_stress.py`, which fuzzes every public surface
with hostile and degenerate input. Defects it found, all now fixed and kept as
regression tests:

| Finding | Fix |
|---|---|
| A feed declaring nested XML entities (billion laughs) expanded a 500-byte body into megabytes of title text | Feeds declaring entities are refused; protection no longer depends on the system libexpat version |
| `fetch_feed` read the response body without limit | Capped at `MAX_FEED_BYTES` (8MB), detected rather than truncated |
| A 2MB title cost ~2.1s of regex scanning per article | `Headline.text` capped at `MAX_TEXT_CHARS` (10k) — one choke point every consumer passes through; the same article now costs ~11ms |
| The backtest rescanned the whole archive at every decision — `decisions x headlines`, billions of comparisons on a two-year run | Two-pointer sliding window plus an optional per-article memo; a 150k-headline, 2-year replay went from minutes to ~10s |
| Well-formed XML that is not a feed (an HTML error page) returned nothing, silently | Still returns nothing, but logs a warning, so a dead feed URL cannot look like a quiet news hour |

Things confirmed already safe: external entities are not resolved (no XXE),
regex metacharacters in company aliases are treated as literals, equity curves
survive going negative, and the IRR search terminates on extreme cashflows.

## Tests

```bash
cd trading && PYTHONPATH=. python3 -m unittest discover -s tests -t .
```

379 tests, no network required — the fixtures in `tests/fixtures/` are synthetic
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
| `wsj_headline_trader/cli.py` | Live command line interface |
| `wsj_headline_trader/backtest.py` | Walk-forward replay engine |
| `wsj_headline_trader/benchmark.py` | Monthly index savings plan |
| `wsj_headline_trader/metrics.py` | IRR, TWR, drawdown, Sharpe |
| `wsj_headline_trader/prices.py` | Price series and panel loading |
| `wsj_headline_trader/archive.py` | Historical headline loading |
| `wsj_headline_trader/backtest_cli.py` | Backtest command line interface |
| `wsj_headline_trader/pine.py` | TradingView Pine Script generator |
| `wsj_headline_trader/pine_cli.py` | Signal log to Pine command line interface |
| `wsj_headline_trader/simulate.py` | Synthetic market and the null/power experiments |
| `wsj_headline_trader/simulate_cli.py` | Simulation command line interface |
| `wsj_headline_trader/data/universe.json` | 207 companies, editable |
| `data/sp500_monthly.csv` | S&P 500 monthly, 1990–2026 (see `data/SOURCES.md`) |

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
