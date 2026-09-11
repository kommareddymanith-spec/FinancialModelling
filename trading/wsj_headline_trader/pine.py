"""Generate a TradingView Pine Script from computed signals.

**Pine Script cannot run this strategy.** Pine executes inside TradingView's
sandbox with no network access: it cannot fetch an RSS feed, parse XML, or
read a headline. The sentiment half of this algorithm is therefore impossible
to express in Pine, and no amount of translation changes that.

What works is splitting the problem where the capability boundary already is:

    Python (has the network)          TradingView (has the chart)
    ------------------------          ---------------------------
    read WSJ feeds                    plot the signals
    find companies, score tone   -->  run the Strategy Tester
    emit signals as Pine data         alert on new bars

This module does the last Python step: it renders a list of signals into a
self-contained Pine v5 strategy with the signals embedded as a data string.
Paste the output into TradingView's Pine Editor and it works on any chart --
the script filters its own data by ``syminfo.ticker``, so one script covers
every symbol it was generated with.

The signals are a snapshot, frozen at generation time. Regenerate to refresh.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

from .models import Side, Signal

log = logging.getLogger(__name__)

#: TradingView caps script size, and a Pine array is capped at 100,000
#: elements. Signals beyond this are dropped oldest-first.
DEFAULT_MAX_SIGNALS = 5_000

#: Ticker characters TradingView accepts. Anything else is refused rather than
#: escaped, because a stray quote would produce a script that will not compile.
_SAFE_TICKER = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,14}$")

_FIELD_SEP = "|"
_RECORD_SEP = ";"


class PineError(ValueError):
    """Raised when signals cannot be rendered into valid Pine."""


@dataclass(frozen=True)
class PineSignal:
    """One dated, directional signal for one symbol."""

    symbol: str
    at: _dt.datetime
    side: Side
    mentions: int = 0
    sentiment: float = 0.0

    @property
    def epoch_millis(self) -> int:
        moment = self.at
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=_dt.timezone.utc)
        return int(moment.timestamp() * 1000)

    def encode(self) -> str:
        """Pack into the generated script's data string."""
        direction = 1 if self.side is Side.BUY else -1
        return _FIELD_SEP.join(
            (
                self.symbol,
                str(self.epoch_millis),
                str(direction),
                str(int(self.mentions)),
                f"{self.sentiment:.3f}",
            )
        )


def from_signals(signals: Iterable[Signal], at: _dt.datetime) -> list[PineSignal]:
    """Convert one run's tradable signals into :class:`PineSignal` objects."""
    return [
        PineSignal(
            symbol=signal.ticker,
            at=at,
            side=signal.side,
            mentions=signal.mention_count,
            sentiment=signal.sentiment_mean,
        )
        for signal in signals
        if signal.tradable
    ]


def from_signal_log(path: str) -> list[PineSignal]:
    """Read the JSONL signal log written by ``--log-signals``."""
    collected: list[PineSignal] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("%s:%d is not valid JSON, skipping (%s)", path, line_number, exc)
                continue
            side = str(record.get("side", "")).lower()
            if side not in ("buy", "short"):
                continue
            raw = str(record.get("decided_at") or record.get("at") or "")
            try:
                moment = _dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                log.warning("%s:%d has no usable timestamp, skipping", path, line_number)
                continue
            collected.append(
                PineSignal(
                    symbol=str(record.get("ticker") or record.get("symbol") or "").upper(),
                    at=moment,
                    side=Side.BUY if side == "buy" else Side.SHORT,
                    mentions=int(record.get("mentions") or 0),
                    sentiment=float(record.get("sentiment_mean") or 0.0),
                )
            )
    return collected


def _validate(signals: Sequence[PineSignal], max_signals: int) -> list[PineSignal]:
    """Drop unusable signals, refuse dangerous ones, and cap the count."""
    usable: list[PineSignal] = []
    for signal in signals:
        if signal.side is Side.FLAT:
            continue
        if not _SAFE_TICKER.match(signal.symbol):
            raise PineError(
                f"{signal.symbol!r} is not a ticker Pine can hold; "
                "expected uppercase letters, digits, dot or dash"
            )
        usable.append(signal)

    usable.sort(key=lambda s: (s.epoch_millis, s.symbol))
    if len(usable) > max_signals:
        log.warning(
            "%d signals exceeds the %d cap; keeping the most recent",
            len(usable),
            max_signals,
        )
        usable = usable[-max_signals:]
    return usable


def render(
    signals: Sequence[PineSignal],
    title: str = "WSJ Headline Signals",
    hold_bars: int = 5,
    percent_of_equity: float = 10.0,
    initial_capital: float = 10_000.0,
    commission_percent: float = 0.03,
    slippage_ticks: int = 2,
    max_signals: int = DEFAULT_MAX_SIGNALS,
    generated_at: _dt.datetime | None = None,
) -> str:
    """Render a complete, self-contained Pine v5 strategy script."""
    usable = _validate(signals, max_signals)
    generated_at = generated_at or _dt.datetime.now(_dt.timezone.utc)
    symbols = sorted({signal.symbol for signal in usable})
    data = _RECORD_SEP.join(signal.encode() for signal in usable)

    if '"' in data or "\\" in data:  # pragma: no cover - _validate prevents it
        raise PineError("signal data contains characters that would break the script")

    span = ""
    if usable:
        first = _dt.datetime.fromtimestamp(usable[0].epoch_millis / 1000, _dt.timezone.utc)
        last = _dt.datetime.fromtimestamp(usable[-1].epoch_millis / 1000, _dt.timezone.utc)
        span = f"//   Covering    {first:%Y-%m-%d %H:%M} to {last:%Y-%m-%d %H:%M} UTC\n"

    header = (
        f"// {title}\n"
        f"//\n"
        f"// GENERATED FILE -- do not edit by hand; regenerate instead.\n"
        f"//   Generated   {generated_at:%Y-%m-%d %H:%M} UTC\n"
        f"//   Signals     {len(usable)}\n"
        f"//   Symbols     {', '.join(symbols) if symbols else '(none)'}\n"
        f"{span}"
        f"//\n"
        f"// The signals below were computed in Python from WSJ headlines. Pine cannot\n"
        f"// fetch or read headlines itself, so they are baked in as data and are a\n"
        f"// snapshot: regenerate this script to pick up newer ones.\n"
        f"//\n"
        f"// Orders are placed on bar close and TradingView fills them at the next\n"
        f"// bar's open, which matches the no-look-ahead rule the Python backtest uses.\n"
    )

    return header + f'''
//@version=5
strategy("{title}",
     overlay            = true,
     initial_capital    = {initial_capital:.0f},
     default_qty_type   = strategy.percent_of_equity,
     default_qty_value  = {percent_of_equity:g},
     commission_type    = strategy.commission.percent,
     commission_value   = {commission_percent:g},
     slippage           = {slippage_ticks},
     calc_on_every_tick = false,
     pyramiding         = 0)

// ---------------------------------------------------------------- inputs
holdBars   = input.int({hold_bars}, "Bars to hold", minval=1,
     tooltip="Close the position this many bars after entry.")
takeLongs  = input.bool(true,  "Take buy signals")
takeShorts = input.bool(true,  "Take short signals")
showMarks  = input.bool(true,  "Mark signals on the chart")
showLabels = input.bool(false, "Label with mentions and tone")

// ------------------------------------------------------------ signal data
// Records are SYMBOL|epoch_ms|direction|mentions|tone, separated by ";".
// Direction is 1 for a buy and -1 for a short.
var string SIGNAL_DATA = "{data}"

var array<int>   sigTime     = array.new_int()
var array<int>   sigDir      = array.new_int()
var array<int>   sigMentions = array.new_int()
var array<float> sigTone     = array.new_float()

if barstate.isfirst
    records = str.split(SIGNAL_DATA, "{_RECORD_SEP}")
    if array.size(records) > 0
        for i = 0 to array.size(records) - 1
            record = array.get(records, i)
            fields = str.split(record, "{_FIELD_SEP}")
            // Keep only this chart's symbol, so one script serves every ticker.
            if array.size(fields) == 5 and array.get(fields, 0) == syminfo.ticker
                array.push(sigTime,     int(str.tonumber(array.get(fields, 1))))
                array.push(sigDir,      int(str.tonumber(array.get(fields, 2))))
                array.push(sigMentions, int(str.tonumber(array.get(fields, 3))))
                array.push(sigTone,     str.tonumber(array.get(fields, 4)))

// --------------------------------------------------- signal for this bar
// The data is sorted by time, so a cursor that only moves forward finds the
// signals in each bar in one pass. Scanning the whole array on every bar
// would hit Pine's loop budget on a long chart.
var int cursor = 0

sigNow      = 0
mentionsNow = 0
toneNow     = 0.0

if array.size(sigTime) > 0
    while cursor < array.size(sigTime) and array.get(sigTime, cursor) < time
        cursor += 1
    while cursor < array.size(sigTime) and array.get(sigTime, cursor) <= time_close
        sigNow      := array.get(sigDir, cursor)
        mentionsNow := array.get(sigMentions, cursor)
        toneNow     := array.get(sigTone, cursor)
        cursor      += 1

wantLong  = sigNow ==  1 and takeLongs
wantShort = sigNow == -1 and takeShorts

// ------------------------------------------------------------------ trade
if wantLong and strategy.position_size <= 0
    strategy.entry("WSJ long", strategy.long)

if wantShort and strategy.position_size >= 0
    strategy.entry("WSJ short", strategy.short)

if strategy.opentrades > 0
    heldFor = bar_index - strategy.opentrades.entry_bar_index(0)
    if heldFor >= holdBars
        strategy.close_all("time stop")

// ----------------------------------------------------------------- visuals
plotshape(showMarks and wantLong,  title="Buy signal",
     style=shape.triangleup,   location=location.belowbar,
     color=color.new(color.teal, 0),   size=size.tiny)
plotshape(showMarks and wantShort, title="Short signal",
     style=shape.triangledown, location=location.abovebar,
     color=color.new(color.maroon, 0), size=size.tiny)

if showLabels and sigNow != 0
    label.new(bar_index, sigNow == 1 ? low : high,
         text  = str.tostring(mentionsNow) + " articles, tone "
                 + str.tostring(toneNow, "#.##"),
         style = sigNow == 1 ? label.style_label_up : label.style_label_down,
         color = color.new(sigNow == 1 ? color.teal : color.maroon, 20),
         textcolor = color.white, size = size.small)

alertcondition(wantLong,  title="WSJ buy signal",   message="WSJ headline buy signal")
alertcondition(wantShort, title="WSJ short signal", message="WSJ headline short signal")
'''


def write(path: str, script: str) -> None:
    """Write a rendered script to disk."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(script)
    log.info("wrote %d bytes of Pine to %s", len(script), path)
