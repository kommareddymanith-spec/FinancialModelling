"""Command line for the exit-threshold sweep.

    python -m wsj_headline_trader.tuning_cli --train-seeds 80 --validation-seeds 80

Thresholds chosen here are fitted to the synthetic market. Read the shape of
the trade-off, not the exact winning number.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from .tuning import (
    DEFAULT_STOPS,
    DEFAULT_TARGETS,
    Cell,
    SweepConfig,
    _rank,
    run_sweep,
)


def _thresholds(raw: str | None, default):
    if not raw:
        return default
    out: list[float | None] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if token in ("none", "off", ""):
            out.append(None)
            continue
        try:
            out.append(float(token))
        except ValueError:
            raise SystemExit(f"error: {token!r} is not a fraction or 'none'")
    return tuple(out)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wsj-headline-tune",
        description=(
            "Sweep stop-loss and take-profit thresholds across many simulated "
            "years, then re-score the leaders on unseen seeds."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "The market is synthetic, so the winning pair is fitted to the "
            "simulator's volatility and edge persistence, not to real equities. "
            "The useful output is the shape of the trade-off."
        ),
    )
    grid = parser.add_argument_group("grid")
    grid.add_argument("--stops", help="Comma-separated fractions, or 'none'.")
    grid.add_argument("--targets", help="Comma-separated fractions, or 'none'.")
    grid.add_argument("--train-seeds", type=int, default=80)
    grid.add_argument("--validation-seeds", type=int, default=80)
    grid.add_argument(
        "--validate-top", type=int, default=3,
        help="How many leading cells to re-score on unseen seeds.",
    )
    grid.add_argument(
        "--objective", choices=("tail-risk", "return"), default="tail-risk",
        help=(
            "'tail-risk' takes the smallest drawdown among cells that give up no "
            "return against no-stop; 'return' takes the highest median return."
        ),
    )
    grid.add_argument(
        "--max-cut-early", type=float, default=0.5,
        help="Reject a stop that is wrong more often than this share of the time.",
    )

    market = parser.add_argument_group("market")
    market.add_argument("--days", type=int, default=252)
    market.add_argument("--edge", type=float, default=0.01)
    market.add_argument("--hold-days", type=int, default=5)
    market.add_argument("--slippage-bps", type=float, default=5.0)
    market.add_argument("--borrow-rate", type=float, default=0.03)
    market.add_argument("--top", type=int, default=3)
    market.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)
    )

    output = parser.add_argument_group("output")
    output.add_argument("--json", action="store_true")
    output.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    try:
        config = SweepConfig(
            stops=_thresholds(args.stops, DEFAULT_STOPS),
            targets=_thresholds(args.targets, DEFAULT_TARGETS),
            train_seeds=args.train_seeds,
            validation_seeds=args.validation_seeds,
            days=args.days,
            edge=args.edge,
            hold_days=args.hold_days,
            slippage_bps=args.slippage_bps,
            borrow_rate_annual=args.borrow_rate,
            top_n=args.top,
            workers=args.workers,
            max_cut_early_rate=args.max_cut_early,
            objective=args.objective,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    result = run_sweep(config, validate_top=args.validate_top)

    if args.json:
        print(json.dumps({
            "simulations": result.simulations,
            "train": [s.as_row() for s in result.train],
            "validation": [s.as_row() for s in result.validation.values()],
            "baseline": result.baseline.as_row(),
            "chosen": None if result.chosen is None else result.chosen.as_row(),
            "rejected": [
                {"stop_loss": c.stop_loss, "take_profit": c.take_profit, "why": why}
                for c, why in result.rejected
            ],
        }, indent=2))
        return 0

    print(
        f"Exit-threshold sweep -- {result.simulations:,} simulated years "
        f"({config.days} sessions each)"
    )
    print(
        f"  hold {config.hold_days} bars, edge {config.edge:g}, "
        f"{config.slippage_bps:g}bp slippage, {config.borrow_rate_annual:g} borrow p.a."
    )
    print(
        f"  {config.train_seeds} seeds to choose, {config.validation_seeds} held back, "
        f"objective {config.objective}\n"
    )

    print("  " + "stop / target".ljust(15)
          + "median   mean  stderr   worst    dd  prof  trades  cut-early")
    # Display order is by return; the drawdown tie-break negates, since a
    # drawdown nearer zero is the better one.
    ordered = sorted(result.train, key=lambda s: (-s.median_twr, -s.mean_drawdown))
    for score in ordered:
        cut = "    -" if score.cut_early_rate is None else f"{score.cut_early_rate:>5.0%}"
        flag = ""
        if score.cell == Cell(None, None):
            flag = "  <- no stops"
        if result.chosen and score.cell == result.chosen.cell:
            flag = "  <- best on train"
        print(
            f"  {score.cell.label():<15}"
            f"{score.median_twr * 100:>+6.1f}%{score.mean_twr * 100:>+6.1f}%"
            f"{score.stderr_twr * 100:>7.1f}%{score.worst_twr * 100:>+7.1f}%"
            f"{score.mean_drawdown * 100:>+6.0f}%{score.profitable_share * 100:>5.0f}%"
            f"{score.mean_trades:>8.0f}{cut}{flag}"
        )

    if result.rejected:
        print(f"\n  Rejected for cutting trades early ({len(result.rejected)}):")
        for cell, why in result.rejected[:6]:
            print(f"    {cell.label():<15} {why}")
        if len(result.rejected) > 6:
            print(f"    ... and {len(result.rejected) - 6} more")

    print("\n  Held-out check")
    if not result.validation:
        print("    nothing passed the early-exit limit, so nothing was validated")
        return 0

    print(
        "    " + "stop / target".ljust(15)
        + "train median   held-out median   change   held-out dd   worst"
    )
    baseline_validation = result.validation.get(Cell(None, None))
    # Same order the selection used, so the table and the pick agree.
    leaders = _rank(
        [
            s for s in result.train
            if s.cell in result.validation and s.cell != Cell(None, None)
        ],
        result.baseline,
        config,
    )
    shown = 0
    for score in leaders:
        held = result.validation.get(score.cell)
        if held is None:
            continue
        change = held.median_twr - score.median_twr
        print(
            f"    {score.cell.label():<15}{score.median_twr * 100:>+11.1f}%"
            f"{held.median_twr * 100:>+18.1f}%{change * 100:>+9.1f}%"
            f"{held.mean_drawdown * 100:>+14.0f}%{held.worst_twr * 100:>+8.1f}%"
        )
        shown += 1
    if baseline_validation is not None and shown:
        print(
            f"    {'none / none':<15}"
            f"{result.baseline.median_twr * 100:>+11.1f}%"
            f"{baseline_validation.median_twr * 100:>+18.1f}%"
            f"{(baseline_validation.median_twr - result.baseline.median_twr) * 100:>+9.1f}%"
            f"{baseline_validation.mean_drawdown * 100:>+14.0f}%"
            f"{baseline_validation.worst_twr * 100:>+8.1f}%"
        )

    if result.chosen is not None:
        held = result.validation.get(result.chosen.cell)
        print("\n  Selection")
        print(f"    {result.chosen.cell.label()}")
        if held is not None and baseline_validation is not None:
            edge_over_baseline = held.median_twr - baseline_validation.median_twr
            combined = (held.stderr_twr ** 2 + baseline_validation.stderr_twr ** 2) ** 0.5
            print(
                f"    held-out median {held.median_twr * 100:+.1f}% versus "
                f"{baseline_validation.median_twr * 100:+.1f}% with no stops "
                f"({edge_over_baseline * 100:+.1f}%, "
                f"{edge_over_baseline / combined if combined else 0:+.1f} standard errors)"
            )
            print(
                f"    mean drawdown {held.mean_drawdown * 100:.0f}% versus "
                f"{baseline_validation.mean_drawdown * 100:.0f}% with no stops"
            )
            print(
                f"    worst year {held.worst_twr * 100:+.1f}% versus "
                f"{baseline_validation.worst_twr * 100:+.1f}% with no stops"
            )
            if abs(edge_over_baseline) < 2 * combined:
                print(
                    "\n    Note: the return difference is inside noise "
                    f"({edge_over_baseline / combined if combined else 0:+.1f} standard "
                    "errors). The case for a stop here rests on the drawdown and\n"
                    "    worst-year numbers, which move consistently across the whole\n"
                    "    grid, not on out-returning a book with no stops."
                )
            if held.cut_early_rate is not None:
                print(
                    f"    cuts {held.cut_early_rate:.0%} of stopped trades that "
                    "would have recovered"
                )
        print(
            "\n    Fitted to the synthetic market. Read the trade-off, not the number."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
