"""A synthetic market for exercising the algorithm end to end.

There is no historical WSJ archive to replay, and watching a live paper account
takes weeks. This module builds a fake market instead -- random price paths and
generated headlines -- and runs the *real* pipeline over it, so nothing here is
a second implementation that could disagree with production.

The point is not to produce a return figure. Synthetic returns mean nothing
about the real world. The point is two experiments you cannot run any other
way:

**The null test (``edge=0``).** Headlines are pure noise, uncorrelated with
prices. A correct engine must then earn roughly zero before costs and lose
roughly the costs after them. If a null run shows a profit, something is
wrong -- look-ahead, double-counted P&L, a sizing bug -- and the number is
coming from the implementation rather than the market.

**The power test (``edge>0``).** Headlines genuinely predict the next few
days' drift. A correct engine must find it. If a run with a real edge shows
nothing, the strategy is failing to act on signal that is demonstrably there.

Every run is reproducible from its ``seed``.
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
import random
from dataclasses import dataclass, field

from .backtest import BacktestConfig, BacktestResult, run_backtest
from .benchmark import DcaConfig, run_dca
from .metrics import PerformanceSummary
from .models import Headline
from .prices import Bar, PricePanel
from .universe import Universe

log = logging.getLogger(__name__)

#: Symbols the fake market trades. Every one is recognised by the bundled
#: universe, and the templates below carry a corporate cue next to the name so
#: ambiguous names such as Target resolve too.
DEFAULT_SYMBOLS: dict[str, str] = {
    "NVDA": "Nvidia",
    "BA": "Boeing",
    "F": "Ford",
    "PFE": "Pfizer",
    "TGT": "Target",
    "KO": "Coca-Cola",
    "INTC": "Intel",
    "DIS": "Disney",
    "GM": "General Motors",
    "MU": "Micron",
}

BULLISH_TEMPLATES = (
    "{company} Shares Surge on Blowout Results",
    "{company} Soars to a Record High as Profit Beats Expectations",
    "{company} Stock Jumps as Analysts Upgrade It on Strong Demand",
)

BEARISH_TEMPLATES = (
    "{company} Shares Plunge After a Fresh Recall",
    "{company} Warns of a Wider Loss as Sales Weaken",
    "{company} Cuts Guidance and Announces Job Cuts",
)

#: Headlines that name no company, so the recogniser has chaff to reject.
NOISE_TEMPLATES = (
    "Treasury Yields Rise Ahead of Inflation Data",
    "Investors Target Small-Cap Stocks as the Rally Broadens",
    "Oil Slips on Oversupply Worries",
    "The Gap Between Rich and Poor Widens Again",
    "Fed Officials Signal Caution on Rate Cuts",
)


@dataclass
class MarketConfig:
    """Shape of the synthetic market."""

    symbols: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_SYMBOLS))
    #: Trading sessions to generate (weekdays).
    days: int = 252
    start: _dt.date = _dt.date(2025, 1, 1)
    #: Annualised volatility and drift of every symbol's random walk.
    volatility: float = 0.35
    drift: float = 0.06
    starting_price: float = 100.0

    #: Chance that a given symbol produces a news event on a given day.
    event_probability: float = 0.04
    #: Articles per event. Must clear the strategy's min_mentions to trade.
    articles_per_event: int = 3
    #: Company-free headlines added per day, as recogniser chaff.
    noise_per_day: int = 2

    #: Extra daily log-return added to a symbol after a news event, signed by
    #: the event's tone. 0.0 makes headlines pure noise (the null test).
    edge: float = 0.0
    #: Sessions over which that edge is spread, starting with the event day.
    edge_days: int = 3

    seed: int = 7

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("at least one symbol is required")
        if self.days < 2:
            raise ValueError("days must be at least 2")
        if self.volatility < 0:
            raise ValueError("volatility cannot be negative")
        if self.starting_price <= 0:
            raise ValueError("starting_price must be positive")
        if not 0.0 <= self.event_probability <= 1.0:
            raise ValueError("event_probability must be between 0 and 1")
        if self.articles_per_event < 1:
            raise ValueError("articles_per_event must be at least 1")
        if self.noise_per_day < 0:
            raise ValueError("noise_per_day cannot be negative")
        if self.edge_days < 1:
            raise ValueError("edge_days must be at least 1")


@dataclass
class SyntheticMarket:
    """A generated market: price bars, headlines, and an equal-weight index."""

    panel: PricePanel
    headlines: list[Headline]
    index: list[tuple[_dt.date, float]]
    sessions: list[_dt.date]
    events: int = 0

    @property
    def symbols(self) -> list[str]:
        return self.panel.symbols


def _sessions(start: _dt.date, count: int) -> list[_dt.date]:
    out, day = [], start
    while len(out) < count:
        if day.weekday() < 5:
            out.append(day)
        day += _dt.timedelta(days=1)
    return out


def generate_market(config: MarketConfig | None = None) -> SyntheticMarket:
    """Build a reproducible synthetic market from ``config``."""
    config = config or MarketConfig()
    rng = random.Random(config.seed)
    sessions = _sessions(config.start, config.days)

    daily_sigma = config.volatility / math.sqrt(252.0)
    daily_mu = config.drift / 252.0
    per_day_edge = config.edge / config.edge_days

    # Decide the news events first, so their drift can be applied to the price
    # path that the headlines will then appear to have predicted.
    events: list[tuple[int, str, int]] = []  # (session index, symbol, sign)
    for index in range(len(sessions)):
        for symbol in config.symbols:
            if rng.random() < config.event_probability:
                events.append((index, symbol, 1 if rng.random() < 0.5 else -1))

    edge_by_day: dict[tuple[int, str], float] = {}
    if config.edge:
        for index, symbol, sign in events:
            for offset in range(config.edge_days):
                day = index + offset
                if day < len(sessions):
                    key = (day, symbol)
                    edge_by_day[key] = edge_by_day.get(key, 0.0) + sign * per_day_edge

    # Price paths.
    bars: dict[str, list[Bar]] = {}
    closes: dict[str, list[float]] = {}
    for symbol in config.symbols:
        price = config.starting_price
        series: list[Bar] = []
        path: list[float] = []
        for index, session in enumerate(sessions):
            previous = price
            shock = rng.gauss(daily_mu, daily_sigma) + edge_by_day.get((index, symbol), 0.0)
            price = max(previous * math.exp(shock), 0.01)
            open_price = previous
            high = max(open_price, price) * (1.0 + abs(rng.gauss(0, daily_sigma / 3)))
            low = min(open_price, price) * (1.0 - abs(rng.gauss(0, daily_sigma / 3)))
            series.append(
                Bar(date=session, open=open_price, high=high, low=max(low, 0.01), close=price)
            )
            path.append(price)
        bars[symbol] = series
        closes[symbol] = path

    # Headlines. Event articles land pre-market so they are actionable at that
    # session's open, which is where the injected drift then shows up.
    headlines: list[Headline] = []
    counter = 0
    for index, symbol, sign in events:
        company = config.symbols[symbol]
        templates = BULLISH_TEMPLATES if sign > 0 else BEARISH_TEMPLATES
        base = _dt.datetime.combine(
            sessions[index], _dt.time(rng.randrange(2, 11), rng.randrange(0, 60)),
            tzinfo=_dt.timezone.utc,
        )
        for article in range(config.articles_per_event):
            counter += 1
            headlines.append(
                Headline(
                    title=templates[article % len(templates)].format(company=company),
                    summary="",
                    link=f"sim://{symbol}/{index}/{article}",
                    published_at=base + _dt.timedelta(minutes=article),
                    source="simulated-wsj",
                )
            )

    for index, session in enumerate(sessions):
        for noise in range(config.noise_per_day):
            counter += 1
            headlines.append(
                Headline(
                    title=NOISE_TEMPLATES[counter % len(NOISE_TEMPLATES)],
                    summary="",
                    link=f"sim://noise/{index}/{noise}",
                    published_at=_dt.datetime.combine(
                        session, _dt.time(rng.randrange(2, 11), rng.randrange(0, 60)),
                        tzinfo=_dt.timezone.utc,
                    ),
                    source="simulated-wsj",
                )
            )

    headlines.sort(key=lambda h: h.published_at)

    # Equal-weight index of the same market, for the buy-and-hold comparison.
    index_series = [
        (
            session,
            config.starting_price
            * sum(closes[s][i] / config.starting_price for s in config.symbols)
            / len(config.symbols),
        )
        for i, session in enumerate(sessions)
    ]

    log.info(
        "generated %d sessions, %d symbols, %d headlines from %d event(s), edge=%.4f",
        len(sessions), len(config.symbols), len(headlines), len(events), config.edge,
    )
    return SyntheticMarket(
        panel=PricePanel(bars),
        headlines=headlines,
        index=index_series,
        sessions=sessions,
        events=len(events),
    )


@dataclass
class SimulationResult:
    """A completed simulation: the replay, and the market it ran in."""

    market: SyntheticMarket
    backtest: BacktestResult
    strategy: PerformanceSummary
    benchmark: PerformanceSummary
    config: MarketConfig

    @property
    def excess_twr(self) -> float:
        """Strategy time-weighted return less the index's."""
        return self.strategy.twr_total - self.benchmark.twr_total


def run_simulation(
    market_config: MarketConfig | None = None,
    backtest_config: BacktestConfig | None = None,
    universe: Universe | None = None,
) -> SimulationResult:
    """Generate a market and replay the strategy over it."""
    market_config = market_config or MarketConfig()
    backtest_config = backtest_config or BacktestConfig()
    universe = universe or Universe.load()

    market = generate_market(market_config)
    result = run_backtest(market.headlines, market.panel, backtest_config, universe)

    if not result.equity:
        raise RuntimeError("the simulation produced no equity curve")

    benchmark = run_dca(
        market.index,
        DcaConfig(
            contribution=backtest_config.contribution or 1_000.0,
            day_of_month=backtest_config.contribution_day,
        ),
    ).summary("Buy and hold")

    return SimulationResult(
        market=market,
        backtest=result,
        strategy=result.summary("WSJ strategy (sim)"),
        benchmark=benchmark,
        config=market_config,
    )
