"""Run the algorithm in a synthetic market.

    # watch it trade, once
    python -m wsj_headline_trader.simulate_cli --days 252 --seed 7

    # the null test: headlines are noise, so a correct engine earns ~nothing
    python -m wsj_headline_trader.simulate_cli --edge 0 --seeds 30

    # the power test: headlines genuinely predict drift, so it should find it
    python -m wsj_headline_trader.simulate_cli --edge 0.01 --seeds 30

    # how much real edge the strategy needs to beat noise and costs
    python -m wsj_headline_trader.simulate_cli --sweep
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys

from .backtest import BacktestConfig
from .metrics import format_comparison
from .simulate import MarketConfig, SimulationResult, run_simulation
from .strategy import StrategyConfig
from .universe import Universe

SWEEP_EDGES = (0.0, 0.002, 0.005, 0.01, 0.02, 0.03)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wsj-headline-simulate",
        description=(
            "Exercise the algorithm in a synthetic market with generated prices "
            "and headlines."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Synthetic returns say nothing about real markets. What they test is "
            "the engine: with --edge 0 the headlines are noise and a correct "
            "engine must earn roughly nothing, and with --edge above zero they "
            "predict drift and it must find it."
        ),
    )

    market = parser.add_argument_group("market")
    market.add_argument("--days", type=int, default=252, help="Trading sessions to generate.")
    market.add_argument("--volatility", type=float, default=0.35, help="Annualised, per symbol.")
    market.add_argument("--drift", type=float, default=0.06, help="Annualised, per symbol.")
    market.add_argument(
        "--event-probability", type=float, default=0.04,
        help="Chance a symbol produces news on a given day.",
    )
    market.add_argument(
        "--edge", type=float, default=0.0,
        help="Daily drift a news event predicts. 0 makes headlines pure noise.",
    )
    market.add_argument("--edge-days", type=int, default=3, help="Sessions the edge spans.")
    market.add_argument("--seed", type=int, default=7, help="Reproducibility seed.")
    market.add_argument(
        "--seeds", type=int, metavar="N",
        help="Run N seeds (1..N) and report the distribution instead of one run.",
    )
    market.add_argument(
        "--sweep", action="store_true",
        help="Sweep --edge across a range to find the detection threshold.",
    )

    strategy = parser.add_argument_group("strategy")
    strategy.add_argument("--top", type=int, default=3)
    strategy.add_argument("--min-mentions", type=int, default=2)
    strategy.add_argument("--min-sentiment", type=float, default=0.5)
    strategy.add_argument("--notional", type=float, default=1000.0)
    strategy.add_argument("--max-notional", type=float, default=5000.0)
    strategy.add_argument("--contribution", type=float, default=1000.0, help="Monthly, both legs.")

    execution = parser.add_argument_group("execution assumptions")
    execution.add_argument("--hold-days", type=int, default=3)
    execution.add_argument("--stop-loss", type=float)
    execution.add_argument("--take-profit", type=float)
    execution.add_argument("--slippage-bps", type=float, default=5.0)
    execution.add_argument("--commission", type=float, default=0.0)
    execution.add_argument("--borrow-rate", type=float, default=0.03)

    output = parser.add_argument_group("output")
    output.add_argument("--json", action="store_true")
    output.add_argument("--trades", action="store_true", help="List closed trades (single run).")
    output.add_argument("-v", "--verbose", action="store_true")
    return parser


def _configs(args, edge: float, seed: int) -> tuple[MarketConfig, BacktestConfig]:
    market = MarketConfig(
        days=args.days,
        volatility=args.volatility,
        drift=args.drift,
        event_probability=args.event_probability,
        edge=edge,
        edge_days=args.edge_days,
        seed=seed,
    )
    backtest = BacktestConfig(
        strategy=StrategyConfig(
            top_n=args.top,
            min_mentions=args.min_mentions,
            min_abs_sentiment=args.min_sentiment,
            notional_per_trade=args.notional,
            max_total_notional=args.max_notional,
        ),
        contribution=args.contribution,
        hold_days=args.hold_days,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        slippage_bps=args.slippage_bps,
        commission_per_trade=args.commission,
        borrow_rate_annual=args.borrow_rate,
    )
    return market, backtest


def _distribution(values: list[float]) -> dict[str, float]:
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "mean": mean,
        "median": statistics.median(values),
        "stdev": stdev,
        "stderr": stdev / len(values) ** 0.5 if values else 0.0,
        "worst": min(values),
        "best": max(values),
        "profitable": sum(1 for v in values if v > 0) / len(values),
    }


def _run_many(args, edge: float, seeds: int, universe: Universe) -> list[SimulationResult]:
    results = []
    for seed in range(1, seeds + 1):
        market, backtest = _configs(args, edge, seed)
        results.append(run_simulation(market, backtest, universe))
    return results


def _print_single(result: SimulationResult, show_trades: bool) -> None:
    print(format_comparison(result.strategy, result.benchmark))
    backtest = result.backtest
    print(
        f"\nSimulated market: {len(result.market.sessions)} sessions, "
        f"{len(result.market.symbols)} symbols, {len(result.market.headlines):,} headlines "
        f"from {result.market.events} news event(s), edge={result.config.edge:g}, "
        f"seed={result.config.seed}"
    )
    print(
        f"Replay: {backtest.decisions:,} decisions, {backtest.signals_generated:,} signals, "
        f"{len(backtest.trades):,} closed trades, {len(backtest.open_at_end)} open, "
        f"{backtest.total_costs:,.2f} paid in costs"
    )
    if backtest.skipped:
        print("Orders not filled:")
        for reason, count in backtest.skipped.most_common(6):
            print(f"    {count:>5}  {reason}")
    if show_trades:
        print("\nClosed trades")
        print(f"  {'SYMBOL':<8}{'SIDE':<7}{'ENTRY':<12}{'EXIT':<12}{'NET':>10}{'RET':>8}  REASON")
        for trade in backtest.trades:
            print(
                f"  {trade.symbol:<8}{trade.side.value:<7}"
                f"{trade.entry_date.isoformat():<12}{trade.exit_date.isoformat():<12}"
                f"{trade.net_pnl:>10,.2f}{trade.return_pct * 100:>7.1f}%  {trade.exit_reason}"
            )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    try:
        universe = Universe.load()
        _configs(args, args.edge, args.seed)  # validate before doing work
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # -- edge sweep
    if args.sweep:
        seeds = args.seeds or 20
        rows = []
        baseline: float | None = None
        for edge in SWEEP_EDGES:
            returns = [r.strategy.twr_total for r in _run_many(args, edge, seeds, universe)]
            stats = _distribution(returns)
            if baseline is None:
                baseline = stats["mean"]
            stats["edge"] = edge
            stats["vs_null_stderrs"] = (
                (stats["mean"] - baseline) / stats["stderr"] if stats["stderr"] else 0.0
            )
            rows.append(stats)

        if args.json:
            print(json.dumps({"seeds": seeds, "sweep": rows}, indent=2))
            return 0

        print(f"Edge sweep -- {seeds} seeds per row, {args.days} sessions each")
        print(f"  costs: {args.slippage_bps:g}bp slippage, {args.borrow_rate:g} borrow p.a.\n")
        print("  edge/event   mean TWR    median    stderr  profitable   vs null")
        for row in rows:
            flag = "" if row["edge"] == 0 else f"  {row['vs_null_stderrs']:+5.1f} SE"
            print(
                f"  {row['edge']:<10.3f} {row['mean'] * 100:>+8.2f}% "
                f"{row['median'] * 100:>+9.2f}% {row['stderr'] * 100:>8.2f}% "
                f"{row['profitable'] * 100:>9.0f}%{flag}"
            )
        print(
            "\n  Row one is the null test: headlines uncorrelated with prices. A mean\n"
            "  near zero (or negative once costs bite) is the correct result -- a\n"
            "  profit there would mean the engine, not the market, is producing it."
        )
        return 0

    # -- distribution over seeds
    if args.seeds:
        results = _run_many(args, args.edge, args.seeds, universe)
        returns = [r.strategy.twr_total for r in results]
        excess = [r.excess_twr for r in results]
        stats = _distribution(returns)

        if args.json:
            print(json.dumps({
                "seeds": args.seeds, "edge": args.edge,
                "strategy_twr": stats,
                "excess_vs_buy_and_hold": _distribution(excess),
                "mean_trades": statistics.mean([len(r.backtest.trades) for r in results]),
            }, indent=2))
            return 0

        print(f"{args.seeds} simulated runs, edge={args.edge:g}, {args.days} sessions each\n")
        print("  Strategy time-weighted return")
        print(f"    mean        {stats['mean'] * 100:+.2f}%   (standard error {stats['stderr'] * 100:.2f}%)")
        print(f"    median      {stats['median'] * 100:+.2f}%")
        print(f"    spread      {stats['worst'] * 100:+.1f}% to {stats['best'] * 100:+.1f}%"
              f"   (stdev {stats['stdev'] * 100:.2f}%)")
        print(f"    profitable  {stats['profitable'] * 100:.0f}% of runs")
        print("  Versus buy and hold")
        print(f"    mean excess {statistics.mean(excess) * 100:+.2f}%")
        print(f"  Trades per run {statistics.mean([len(r.backtest.trades) for r in results]):.0f}")
        if args.edge == 0:
            print(
                "\n  This is the null test: the headlines carried no information, so a\n"
                "  mean within a couple of standard errors of zero is the correct\n"
                "  result. A clear profit would point at a bug, not an edge."
            )
        return 0

    # -- single run
    market, backtest = _configs(args, args.edge, args.seed)
    result = run_simulation(market, backtest, universe)

    if args.json:
        print(json.dumps({
            "market": {
                "sessions": len(result.market.sessions),
                "symbols": result.market.symbols,
                "headlines": len(result.market.headlines),
                "events": result.market.events,
                "edge": result.config.edge,
                "seed": result.config.seed,
            },
            "strategy": {
                "final_value": round(result.strategy.final_value, 2),
                "contributions": round(result.strategy.contributions, 2),
                "twr_total": round(result.strategy.twr_total, 6),
                "max_drawdown": round(result.strategy.max_drawdown, 6),
                "trades": len(result.backtest.trades),
                "costs": round(result.backtest.total_costs, 2),
            },
            "benchmark": {
                "final_value": round(result.benchmark.final_value, 2),
                "twr_total": round(result.benchmark.twr_total, 6),
            },
            "excess_twr": round(result.excess_twr, 6),
        }, indent=2))
        return 0

    _print_single(result, args.trades)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
