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
from .feed import DEFAULT_FEEDS, collect_headlines, fetch_feed, headlines_from_files
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
        "--check",
        action="store_true",
        help=(
            "Preflight only: verify credentials, account state, shorting "
            "permission, market hours and feed access, then exit without trading."
        ),
    )
    execution.add_argument(
        "--queue-when-closed",
        action="store_true",
        help=(
            "With --broker alpaca --live, submit even when the market is closed. "
            "Off by default: an order queued to an open hours away acts on "
            "headlines that are no longer news."
        ),
    )
    execution.add_argument(
        "--allow-stacking",
        action="store_true",
        help=(
            "Allow a new position in a symbol already held. Off by default, so a "
            "scheduled run does not pyramid into a story that stays in the news."
        ),
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


def run_preflight(broker, config, fetcher, broker_name: str) -> int:
    """Check everything a scheduled run depends on, and trade nothing.

    Returns 0 when the setup looks usable, 1 when something would stop it
    working. Meant to be run once before trusting a schedule, and again after
    any credential change.
    """
    problems: list[str] = []
    print("Preflight\n")

    print(f"  broker            : {broker_name}")
    account = getattr(broker, "account", None)
    if account is not None:
        try:
            details = account()
        except BrokerError as exc:
            problems.append(f"account unreachable: {exc}")
            print(f"  account           : UNREACHABLE -- {exc}")
        else:
            status = details.get("status", "?")
            print(f"  endpoint          : {getattr(broker, 'base_url', '?')}")
            print(f"  account status    : {status}")
            print(f"  buying power      : {details.get('buying_power', '?')}")
            print(f"  cash              : {details.get('cash', '?')}")
            shorting = details.get("shorting_enabled")
            print(f"  shorting enabled  : {shorting}")
            if details.get("trading_blocked"):
                problems.append("the account has trading blocked")
            if status not in ("ACTIVE", "?"):
                problems.append(f"account status is {status}, not ACTIVE")
            if shorting is False:
                problems.append(
                    "shorting is disabled, so every short signal will be refused; "
                    "a margin account is required for the short half of this strategy"
                )
    else:
        print("  account           : n/a for this broker")

    clock = getattr(broker, "clock", None)
    if clock is not None:
        try:
            state = clock()
        except BrokerError as exc:
            problems.append(f"market clock unreachable: {exc}")
            print(f"  market            : UNREACHABLE -- {exc}")
        else:
            print(
                f"  market            : {'OPEN' if state.get('is_open') else 'CLOSED'}"
                f" (next open {state.get('next_open', '?')})"
            )
    else:
        print("  market            : n/a for this broker")

    held = getattr(broker, "open_symbols", None)
    if held is not None:
        try:
            symbols = sorted(held())
        except Exception as exc:
            print(f"  open positions    : UNREADABLE -- {exc}")
        else:
            print(f"  open positions    : {', '.join(symbols) if symbols else 'none'}")

    # Feed access is the other half: credentials are useless without headlines.
    headlines, errors = collect_headlines(
        feeds=config.feeds,
        window_minutes=config.window_minutes,
        now=None,
        fetcher=fetcher,
    )
    print(f"  feeds reachable   : {len(config.feeds) - len(errors)}/{len(config.feeds)}")
    print(f"  headlines in {config.window_minutes:>3}m : {len(headlines)}")
    for error in errors:
        print(f"      - {error}")
    if len(errors) == len(config.feeds):
        problems.append("no feed could be read, so there is nothing to trade on")

    if problems:
        print("\n  NOT READY")
        for problem in problems:
            print(f"    - {problem}")
        return 1

    print("\n  Ready. Nothing was traded by this check.")
    if not headlines:
        print(
            "  Note: no headlines in the window right now. That is normal outside\n"
            "  busy hours; it only means this run would have found nothing."
        )
    return 0


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
            skip_held_symbols=not args.allow_stacking,
            require_market_open=(
                args.broker == "alpaca" and args.live and not args.queue_when_closed
            ),
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

    if args.check:
        return run_preflight(broker, config, fetcher, args.broker)

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
