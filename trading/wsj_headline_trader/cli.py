"""Command line entry point.

    python -m wsj_headline_trader --help
    python -m wsj_headline_trader --window-minutes 60 --top 5
    python -m wsj_headline_trader --broker paper --live
    python -m wsj_headline_trader --fixture tests/fixtures/wsj_markets.xml --json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys

from .algorithm import AlgorithmConfig, WSJHeadlineAlgorithm, append_signal_log, format_report
from .broker import AlpacaBroker, AlpacaPriceProvider, BrokerError, PaperBroker
from .feed import DEFAULT_FEEDS, fetch_feed, headlines_from_files
from .strategy import StrategyConfig
from .universe import Universe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wsj-headline-trader",
        description=(
            "Buy the companies covered positively in the last hour of WSJ "
            "headlines and short the ones covered negatively."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    feed = parser.add_argument_group("headlines")
    feed.add_argument(
        "--feed",
        action="append",
        dest="feeds",
        metavar="URL",
        help="WSJ RSS feed to read; repeatable. Defaults to the four public feeds.",
    )
    feed.add_argument(
        "--window-minutes", type=int, default=60, help="How far back to read headlines."
    )
    feed.add_argument(
        "--fixture",
        action="append",
        metavar="PATH",
        help="Read saved feed XML from disk instead of the network; repeatable.",
    )
    feed.add_argument(
        "--as-of",
        metavar="ISO8601",
        help="Treat this instant as 'now' (useful with --fixture).",
    )
    feed.add_argument("--universe", metavar="PATH", help="Alternative company/ticker JSON.")

    strategy = parser.add_argument_group("strategy")
    strategy.add_argument("--top", type=int, default=5, help="How many of the most-featured companies to consider.")
    strategy.add_argument("--min-mentions", type=int, default=2, help="Minimum articles naming a company.")
    strategy.add_argument(
        "--min-sentiment",
        type=float,
        default=0.5,
        help="Minimum absolute mean tone to take a side.",
    )
    strategy.add_argument(
        "--min-agreement",
        type=float,
        default=0.6,
        help="Minimum share of articles agreeing with the direction.",
    )
    strategy.add_argument("--notional", type=float, default=1000.0, help="Base size per trade.")
    strategy.add_argument(
        "--max-notional", type=float, default=5000.0, help="Total size cap for the run."
    )

    execution = parser.add_argument_group("execution")
    execution.add_argument(
        "--broker",
        choices=("paper", "alpaca"),
        default="paper",
        help="Where to send orders.",
    )
    execution.add_argument(
        "--live",
        action="store_true",
        help="Actually submit orders. Without it the run is a dry run.",
    )
    execution.add_argument(
        "--paper-marks",
        choices=("flat", "alpaca"),
        default="flat",
        help=(
            "How the paper broker prices fills. 'flat' uses a fixed notional "
            "price, so positions never move; 'alpaca' marks at real last-traded "
            "prices (needs Alpaca credentials, still risks nothing)."
        ),
    )
    execution.add_argument(
        "--real-money",
        action="store_true",
        help="With --broker alpaca, use the live endpoint instead of paper.",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--json", action="store_true", help="Emit the run report as JSON.")
    output.add_argument(
        "--log-signals",
        metavar="PATH",
        help=(
            "Append this run's tradable signals to a JSONL file. Runs accumulate "
            "into a signal history that wsj_headline_trader.pine_cli can turn into "
            "a TradingView script."
        ),
    )
    output.add_argument("-v", "--verbose", action="store_true", help="Log what the algorithm is doing.")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    feeds = tuple(args.feeds) if args.feeds else DEFAULT_FEEDS
    fetcher = headlines_from_files(args.fixture) if args.fixture else fetch_feed
    if args.fixture:
        # One fixture payload is consumed per feed, so only read as many feeds.
        feeds = feeds[: len(args.fixture)]

    now = None
    if args.as_of:
        try:
            now = _dt.datetime.fromisoformat(args.as_of.replace("Z", "+00:00"))
        except ValueError:
            print(f"error: --as-of is not a valid ISO 8601 timestamp: {args.as_of}", file=sys.stderr)
            return 2

    try:
        strategy = StrategyConfig(
            top_n=args.top,
            min_mentions=args.min_mentions,
            min_abs_sentiment=args.min_sentiment,
            min_agreement=args.min_agreement,
            notional_per_trade=args.notional,
            max_total_notional=args.max_notional,
        )
        config = AlgorithmConfig(
            feeds=feeds,
            window_minutes=args.window_minutes,
            strategy=strategy,
            universe_path=args.universe,
            dry_run=not args.live,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.broker == "alpaca":
        try:
            broker = AlpacaBroker(paper=not args.real_money)
        except BrokerError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    elif args.paper_marks == "alpaca":
        try:
            broker = PaperBroker(prices=AlpacaPriceProvider())
        except BrokerError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    else:
        broker = PaperBroker()

    if args.live and args.broker == "alpaca" and args.real_money:
        print(
            "WARNING: submitting REAL orders to Alpaca live trading.",
            file=sys.stderr,
        )

    try:
        universe = Universe.load(config.universe_path)
    except (OSError, ValueError) as exc:
        print(f"error: could not load universe: {exc}", file=sys.stderr)
        return 2

    algorithm = WSJHeadlineAlgorithm(
        config=config, broker=broker, universe=universe, fetcher=fetcher
    )
    report = algorithm.run(now=now)

    if args.log_signals:
        try:
            appended = append_signal_log(args.log_signals, report)
        except OSError as exc:
            print(f"warning: could not write signal log: {exc}", file=sys.stderr)
        else:
            if appended and not args.json:
                print(f"\n  Logged {appended} signal(s) to {args.log_signals}")

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_report(report))

    if report.feed_errors and report.headlines_scanned == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
