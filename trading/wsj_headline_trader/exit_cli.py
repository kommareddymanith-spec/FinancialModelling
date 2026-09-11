"""Command line for the exit job.

    # see what would be closed
    python -m wsj_headline_trader.exit_cli --broker alpaca

    # actually close
    python -m wsj_headline_trader.exit_cli --broker alpaca --live \
        --stop-loss 0.04 --take-profit 0.08 --max-hold-days 5 \
        --ledger ~/wsj-entries.json

Run this more often than the entry job. A stop checked once an hour is a stop
that can be gapped through.
"""

from __future__ import annotations

import json
import logging
import sys

from .broker import AlpacaBroker, AlpacaPriceProvider, BrokerError, PaperBroker
from .exit_job import ExitConfig, format_exit_report, run_exit_job

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wsj-headline-exit",
        description="Close open positions on a profit, loss or time rule.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Thresholds are fractions of cost basis: 0.02 is two percent. Pass "
            "'none' to disable a rule. The defaults come from the sweep in "
            "data/exit_sweep_2026-09-11.txt and are fitted to a synthetic market; "
            "see the README before trusting the exact numbers."
        ),
    )

    rules = parser.add_argument_group("exit rules")
    rules.add_argument(
        "--stop-loss", default="0.02", metavar="FRACTION",
        help="Close at or beyond this unrealized loss. 'none' disables.",
    )
    rules.add_argument(
        "--take-profit", default="0.12", metavar="FRACTION",
        help="Close at or beyond this unrealized gain. 'none' disables.",
    )
    rules.add_argument(
        "--max-hold-days", default="5", metavar="DAYS",
        help="Close after this long regardless of P&L. Needs --ledger. 'none' disables.",
    )
    rules.add_argument(
        "--ledger", metavar="PATH",
        help="Entry ledger written by the entry job's --ledger, for the time stop.",
    )

    execution = parser.add_argument_group("execution")
    execution.add_argument(
        "--broker", choices=("paper", "alpaca"), default="paper",
        help="Where the positions live.",
    )
    execution.add_argument(
        "--live", action="store_true",
        help="Actually send the closing orders. Without it nothing is sent.",
    )
    execution.add_argument(
        "--real-money", action="store_true",
        help="With --broker alpaca, act on the live account instead of paper.",
    )

    output = parser.add_argument_group("output")
    output.add_argument("--json", action="store_true")
    output.add_argument("-v", "--verbose", action="store_true")
    return parser


def _fraction(raw: str, flag: str) -> float | None:
    if raw.strip().lower() in ("none", "off", ""):
        return None
    try:
        return float(raw)
    except ValueError:
        raise SystemExit(f"error: {flag} must be a fraction or 'none', got {raw!r}")


def _days(raw: str, flag: str) -> int | None:
    if raw.strip().lower() in ("none", "off", ""):
        return None
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"error: {flag} must be a whole number or 'none', got {raw!r}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    try:
        config = ExitConfig(
            stop_loss=_fraction(args.stop_loss, "--stop-loss"),
            take_profit=_fraction(args.take_profit, "--take-profit"),
            max_hold_days=_days(args.max_hold_days, "--max-hold-days"),
            dry_run=not args.live,
            ledger_path=args.ledger,
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
        if args.live and args.real_money:
            print("WARNING: closing REAL positions on Alpaca live.", file=sys.stderr)
    else:
        try:
            broker = PaperBroker(prices=AlpacaPriceProvider())
        except BrokerError:
            # No credentials: the paper broker still works, just unmarked. An
            # empty in-process book means there is nothing to close.
            broker = PaperBroker()

    report = run_exit_job(broker, config)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(format_exit_report(report))

    failed = [
        d for d in report.closed if d.result is not None and not d.result.accepted
    ]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
