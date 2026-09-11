"""Turn a signal history into a TradingView Pine script.

    # accumulate signals from scheduled live runs, then export
    python -m wsj_headline_trader --log-signals signals.jsonl
    python -m wsj_headline_trader.pine_cli --signals signals.jsonl --out wsj.pine

Paste the result into TradingView's Pine Editor and add it to any chart whose
ticker appears in the signal history.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import sys

from .pine import DEFAULT_MAX_SIGNALS, PineError, from_signal_log, render, write


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wsj-headline-pine",
        description=(
            "Render computed WSJ headline signals into a self-contained Pine v5 "
            "strategy for TradingView."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Pine cannot read headlines -- it has no network access -- so the "
            "signals are computed in Python and embedded as data. Re-run this to "
            "refresh them."
        ),
    )
    parser.add_argument(
        "--signals", required=True, metavar="PATH",
        help="JSONL signal log, as written by `wsj_headline_trader --log-signals`.",
    )
    parser.add_argument(
        "--out", metavar="PATH", help="Write the script here instead of stdout."
    )
    parser.add_argument("--title", default="WSJ Headline Signals", help="Strategy name.")
    parser.add_argument("--hold-bars", type=int, default=5, help="Bars to hold a position.")
    parser.add_argument(
        "--percent-of-equity", type=float, default=10.0, help="Size per trade."
    )
    parser.add_argument("--initial-capital", type=float, default=10_000.0)
    parser.add_argument(
        "--commission-percent", type=float, default=0.03, help="Per side, in percent."
    )
    parser.add_argument("--slippage-ticks", type=int, default=2)
    parser.add_argument(
        "--max-signals", type=int, default=DEFAULT_MAX_SIGNALS,
        help="Cap on embedded signals; the most recent are kept.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)-7s %(name)s: %(message)s",
    )

    try:
        signals = from_signal_log(args.signals)
    except OSError as exc:
        print(f"error: could not read {args.signals}: {exc}", file=sys.stderr)
        return 2

    if not signals:
        print(
            f"error: no usable signals in {args.signals}.\n"
            "  Populate it first with: python -m wsj_headline_trader "
            "--log-signals " + args.signals,
            file=sys.stderr,
        )
        return 1

    try:
        script = render(
            signals,
            title=args.title,
            hold_bars=args.hold_bars,
            percent_of_equity=args.percent_of_equity,
            initial_capital=args.initial_capital,
            commission_percent=args.commission_percent,
            slippage_ticks=args.slippage_ticks,
            max_signals=args.max_signals,
            generated_at=_dt.datetime.now(_dt.timezone.utc),
        )
    except PineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.out:
        try:
            write(args.out, script)
        except OSError as exc:
            print(f"error: could not write {args.out}: {exc}", file=sys.stderr)
            return 2
        symbols = sorted({s.symbol for s in signals})
        print(
            f"Wrote {args.out}: {len(signals)} signal(s) across "
            f"{len(symbols)} symbol(s) ({', '.join(symbols[:8])}"
            f"{'...' if len(symbols) > 8 else ''}).\n"
            "Paste it into TradingView's Pine Editor, then add it to a chart for "
            "one of those symbols."
        )
    else:
        print(script)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
