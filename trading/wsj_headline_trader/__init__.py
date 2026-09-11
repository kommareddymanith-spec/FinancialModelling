"""Trade the companies most featured in the last hour of WSJ headlines.

Rank companies by how many recent Wall Street Journal articles name them, read
the tone of those articles, then buy the ones covered positively and short the
ones covered negatively.

    from wsj_headline_trader import WSJHeadlineAlgorithm, AlgorithmConfig
    from wsj_headline_trader.algorithm import format_report

    report = WSJHeadlineAlgorithm(AlgorithmConfig(window_minutes=60)).run()
    print(format_report(report))

Runs are dry by default -- pass ``dry_run=False`` to actually place orders.
"""

from .algorithm import AlgorithmConfig, WSJHeadlineAlgorithm, format_report
from .archive import load_archive
from .backtest import BacktestConfig, BacktestResult, Trade, run_backtest
from .benchmark import DcaConfig, DcaResult, run_dca
from .broker import (
    AlpacaBroker,
    AlpacaPriceProvider,
    BrokerPosition,
    PaperBroker,
    StaticPriceProvider,
)
from .exit_job import ExitConfig, ExitReport, run_exit_job
from .tuning import SweepConfig, run_sweep
from .metrics import PerformanceSummary, format_comparison, irr, summarise
from .pine import PineSignal, render as render_pine
from .models import Headline, Mention, Order, OrderResult, RunReport, Side, Signal
from .prices import PricePanel, load_price_panel, load_series
from .simulate import MarketConfig, SyntheticMarket, generate_market, run_simulation
from .sentiment import score_text
from .strategy import StrategyConfig, build_signals, extract_mentions
from .universe import Company, Universe

__version__ = "1.0.0"

__all__ = [
    "AlgorithmConfig",
    "AlpacaBroker",
    "AlpacaPriceProvider",
    "BacktestConfig",
    "BacktestResult",
    "BrokerPosition",
    "Company",
    "DcaConfig",
    "DcaResult",
    "ExitConfig",
    "ExitReport",
    "Headline",
    "MarketConfig",
    "Mention",
    "Order",
    "OrderResult",
    "PaperBroker",
    "PerformanceSummary",
    "PineSignal",
    "PricePanel",
    "RunReport",
    "Side",
    "Signal",
    "StaticPriceProvider",
    "StrategyConfig",
    "SweepConfig",
    "SyntheticMarket",
    "Trade",
    "Universe",
    "WSJHeadlineAlgorithm",
    "build_signals",
    "extract_mentions",
    "format_comparison",
    "format_report",
    "generate_market",
    "irr",
    "load_archive",
    "load_price_panel",
    "load_series",
    "render_pine",
    "run_backtest",
    "run_dca",
    "run_exit_job",
    "run_simulation",
    "run_sweep",
    "score_text",
    "summarise",
]
