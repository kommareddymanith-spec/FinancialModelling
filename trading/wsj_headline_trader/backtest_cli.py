"""Backtest command line: replay the strategy and compare it to the benchmark.

    python -m wsj_headline_trader.backtest_cli --benchmark-only
    python -m wsj_headline_trader.backtest_cli \
        --archive data/wsj_headlines_2y.jsonl \
        --prices data/stock_prices_2y.csv \
        --start 2024-09-01 --end 2026-08-31
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import sys

from .archive import ArchiveError, load_archive
from .backtest import BacktestConfig, run_backtest
from .benchmark import DcaConfig, run_dca
from .metrics import format_comparison
from .prices import PriceDataError, load_price_panel, load_series
from .strategy import StrategyConfig
from .universe import Universe

DEFAULT_INDEX = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "sp500_monthly.csv")
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wsj-headline-backtest",
        description=(
            "Replay the WSJ headline strategy over a historical archive and "
            "compare it to a fixed monthly investment in an index."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "The strategy leg needs a headline archive, which WSJ RSS feeds do "
            "not provide -- they serve only the current ~30 items. Run with "
            "--benchmark-only to see the index result on its own."
        ),
    )

    data = parser.add_argument_group("data")
    data.add_argument(
        "--archive",
        action="append",
        metavar="PATH",
        help="Headline archive file or directory (.jsonl/.json/.xml/.csv); repeatable.",
    )
    data.add_argument(
        "--prices",
        metavar="PATH",
        help="Long-format panel: date,symbol,open,high,low,close. Required with --archive.",
    )
    data.add_argument(
        "--index", default=DEFAULT_INDEX, metavar="PATH", help="Benchmark index series CSV."
    )
    data.add_argument("--universe", metavar="PATH", help="Alternative company/ticker JSON.")
    data.add_argument("--start", metavar="YYYY-MM-DD", help="First date of the period.")
    data.add_argument("--end", metavar="YYYY-MM-DD", help="Last date of the period.")

    plan = parser.add_argument_group("investment plan")
    plan.add_argument(
        "--contribution", type=float, default=1000.0, help="Monthly amount into each plan."
    )
    plan.add_argument("--contribution-day", type=int, default=1, help="Day of month to invest.")
    plan.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Report just the index savings plan; no archive needed.",
    )

    strategy = parser.add_argument_group("strategy")
    strategy.add_argument("--window-minutes", type=int, default=60)
    strategy.add_argument("--step-minutes", type=int, default=60)
    strategy.add_argument("--top", type=int, default=5)
    strategy.add_argument("--min-mentions", type=int, default=2)
    strategy.add_argument("--min-sentiment", type=float, default=0.5)
    strategy.add_argument("--min-agreement", type=float, default=0.6)
    strategy.add_argument("--notional", type=float, default=1000.0)
    strategy.add_argument("--max-notional", type=float, default=5000.0)

    execution = parser.add_argument_group("execution assumptions")
    execution.add_argument("--hold-days", type=int, default=5, help="Sessions held before exit.")
    execution.add_argument("--stop-loss", type=float, help="Stop as a fraction, e.g. 0.05.")
    execution.add_argument("--take-profit", type=float, help="Target as a fraction.")
    execution.add_argument(
        "--slippage-bps", type=float, default=5.0, help="Charged on entry and exit."
    )
    execution.add_argument("--commission", type=float, default=0.0, help="Per fill.")
    execution.add_argument(
        "--borrow-rate", type=float, default=0.03, help="Annual short borrow cost."
    )

    output = parser.add_argument_group("output")
    output.add_argument("--json", action="store_true", help="Emit results as JSON.")
    output.add_argument("--trades", action="store_true", help="List every closed trade.")
    output.add_argument("-v", "--verbose", action="store_true")

    return parser


def _parse_date(raw: str | None, flag: str) -> _dt.date | None:
    if not raw:
        return None
    try:
        return _dt.datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        raise SystemExit(f"error: {flag} must be YYYY-MM-DD, got {raw!r}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    start = _parse_date(args.start, "--start")
    end = _parse_date(args.end, "--end")

    if not args.benchmark_only and not args.archive:
        print(
            "error: no headline archive given.\n"
            "  The strategy cannot be backtested without one: WSJ RSS feeds carry\n"
            "  only the current ~30 headlines, with no history.\n"
            "  Pass --archive PATH, or --benchmark-only to see just the index plan.",
            file=sys.stderr,
        )
        return 2
    if args.archive and not args.prices:
        print("error: --archive requires --prices for the traded symbols.", file=sys.stderr)
        return 2

    # -- benchmark
    try:
        index_series = load_series(args.index, start=start, end=end)
    except (PriceDataError, OSError) as exc:
        print(f"error: could not load index series: {exc}", file=sys.stderr)
        return 2

    dca = run_dca(
        index_series,
        DcaConfig(contribution=args.contribution, day_of_month=args.contribution_day),
    )
    benchmark_summary = dca.summary(f"Index DCA ({args.contribution:,.0f}/mo)")
    summaries = [benchmark_summary]
    strategy_result = None

    # -- strategy
    if args.archive:
        try:
            headlines = load_archive(*args.archive)
        except (ArchiveError, OSError) as exc:
            print(f"error: could not load archive: {exc}", file=sys.stderr)
            return 2
        if start:
            headlines = [h for h in headlines if h.published_at.date() >= start]
        if end:
            headlines = [h for h in headlines if h.published_at.date() <= end]
        if not headlines:
            print("error: the archive has no headlines in the requested period.", file=sys.stderr)
            return 2

        try:
            panel = load_price_panel(args.prices, start=start, end=end)
        except (PriceDataError, OSError) as exc:
            print(f"error: could not load price panel: {exc}", file=sys.stderr)
            return 2

        try:
            universe = Universe.load(args.universe)
            config = BacktestConfig(
                window_minutes=args.window_minutes,
                step_minutes=args.step_minutes,
                strategy=StrategyConfig(
                    top_n=args.top,
                    min_mentions=args.min_mentions,
                    min_abs_sentiment=args.min_sentiment,
                    min_agreement=args.min_agreement,
                    notional_per_trade=args.notional,
                    max_total_notional=args.max_notional,
                ),
                contribution=args.contribution,
                contribution_day=args.contribution_day,
                hold_days=args.hold_days,
                stop_loss=args.stop_loss,
                take_profit=args.take_profit,
                slippage_bps=args.slippage_bps,
                commission_per_trade=args.commission,
                borrow_rate_annual=args.borrow_rate,
            )
        except (ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

        strategy_result = run_backtest(headlines, panel, config, universe)
        if not strategy_result.equity:
            print("error: the replay produced no equity curve; check the price panel.", file=sys.stderr)
            return 1
        summaries.insert(0, strategy_result.summary("WSJ headline strategy"))

    # -- report
    if args.json:
        payload: dict[str, object] = {
            "summaries": [
                {
                    "label": s.label,
                    "start": s.start.isoformat(),
                    "end": s.end.isoformat(),
                    "contributions": round(s.contributions, 2),
                    "final_value": round(s.final_value, 2),
                    "profit": round(s.profit, 2),
                    "profit_pct_of_contributions": round(s.profit_pct_of_contributions, 6),
                    "irr_annual": None if s.irr_annual is None else round(s.irr_annual, 6),
                    "twr_total": round(s.twr_total, 6),
                    "twr_annual": round(s.twr_annual, 6),
                    "max_drawdown": round(s.max_drawdown, 6),
                    "volatility_annual": round(s.volatility_annual, 6),
                    "sharpe": round(s.sharpe, 4),
                    "extras": {k: str(v) for k, v in s.extras.items()},
                }
                for s in summaries
            ]
        }
        if strategy_result is not None:
            payload["strategy_diagnostics"] = {
                "headlines": strategy_result.headlines_seen,
                "decisions": strategy_result.decisions,
                "signals": strategy_result.signals_generated,
                "trades": len(strategy_result.trades),
                "open_at_end": len(strategy_result.open_at_end),
                "costs_paid": round(strategy_result.total_costs, 2),
                "skipped": dict(strategy_result.skipped),
            }
        print(json.dumps(payload, indent=2))
        return 0

    print(format_comparison(*summaries))

    if strategy_result is not None:
        print(
            f"\nReplay: {strategy_result.headlines_seen:,} headlines, "
            f"{strategy_result.decisions:,} decisions, "
            f"{strategy_result.signals_generated:,} signals, "
            f"{len(strategy_result.trades):,} closed trades, "
            f"{len(strategy_result.open_at_end)} still open."
        )
        if strategy_result.skipped:
            print("Orders not filled:")
            for reason, count in strategy_result.skipped.most_common():
                print(f"    {count:>5}  {reason}")
        if args.trades:
            print("\nClosed trades")
            print(
                f"  {'SYMBOL':<8}{'SIDE':<7}{'ENTRY':<12}{'EXIT':<12}"
                f"{'NET':>10}{'RET':>8}  REASON"
            )
            for trade in strategy_result.trades:
                print(
                    f"  {trade.symbol:<8}{trade.side.value:<7}"
                    f"{trade.entry_date.isoformat():<12}{trade.exit_date.isoformat():<12}"
                    f"{trade.net_pnl:>10,.2f}{trade.return_pct * 100:>7.1f}%  {trade.exit_reason}"
                )
    else:
        print(
            "\nBenchmark only -- no strategy result. The WSJ headline strategy needs a\n"
            "historical headline archive to replay against; pass one with --archive."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
