"""Closing positions: the other half of a live deployment.

The entry algorithm only opens positions. This closes them, on three rules
applied in order of urgency:

1. **Stop loss** -- unrealized loss at or beyond a threshold.
2. **Take profit** -- unrealized gain at or beyond a threshold.
3. **Time stop** -- held longer than a maximum, whatever the P&L.

Run it on a tighter schedule than the entry job: a stop that is only checked
once an hour is a stop that can be gapped through. It is idempotent, so
running it more often than needed costs nothing but an API call.

The time stop needs to know when a position was opened, which the venue's
position endpoint does not report. The entry job records that in a ledger
(``--ledger``); without one the time stop is skipped and the P&L rules still
apply, since those need only the unrealized return the venue already gives.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Protocol, Sequence

from .broker import BrokerPosition, OrderResult

log = logging.getLogger(__name__)


class ClosableBroker(Protocol):
    """What the exit job needs from a venue."""

    def open_positions(self) -> list[BrokerPosition]: ...
    def close_position(self, symbol: str) -> OrderResult: ...


@dataclass
class ExitConfig:
    """When to close.

    Thresholds are positive fractions of the position's cost basis: 0.02 means
    two percent. ``None`` disables that rule.

    The defaults come from the grid in ``data/exit_sweep_2026-09-11.txt``: a
    tight stop with a distant target held up across 3,840 simulated years,
    while tight targets consistently cost return by clipping winners. Two
    warnings about taking them literally:

    * They are fitted to the simulator's volatility. A 2% stop is roughly one
      daily standard deviation there; on a quieter or noisier symbol the same
      percentage is a different stop entirely. The durable form of this rule is
      volatility-scaled -- a multiple of ATR -- not a fixed percentage.
    * Even in the simulator the *return* difference against running with no
      stops was inside noise. What moved reliably was drawdown and the worst
      year. Set a stop to bound the tail, not to raise the mean.
    """

    stop_loss: float | None = 0.02
    take_profit: float | None = 0.12
    max_hold_days: int | None = 5
    #: Report what would be closed without sending anything.
    dry_run: bool = True
    #: Path to the entry ledger, for the time stop.
    ledger_path: str | None = None

    def __post_init__(self) -> None:
        for name in ("stop_loss", "take_profit"):
            value = getattr(self, name)
            if value is not None and not 0 < value < 1:
                raise ValueError(f"{name} must be a fraction between 0 and 1")
        if self.max_hold_days is not None and self.max_hold_days < 1:
            raise ValueError("max_hold_days must be at least 1")
        if not any((self.stop_loss, self.take_profit, self.max_hold_days)):
            raise ValueError("at least one exit rule must be enabled")


@dataclass
class ExitDecision:
    """What the job decided about one position."""

    position: BrokerPosition
    reason: str | None
    held_days: float | None = None
    result: OrderResult | None = None

    @property
    def closing(self) -> bool:
        return self.reason is not None


@dataclass
class ExitReport:
    """One pass of the exit job."""

    ran_at: _dt.datetime
    decisions: list[ExitDecision] = field(default_factory=list)
    dry_run: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def closed(self) -> list[ExitDecision]:
        return [d for d in self.decisions if d.closing]

    @property
    def held(self) -> list[ExitDecision]:
        return [d for d in self.decisions if not d.closing]

    def to_dict(self) -> dict:
        return {
            "ran_at": self.ran_at.isoformat(),
            "dry_run": self.dry_run,
            "notes": list(self.notes),
            "positions": [
                {
                    "symbol": d.position.symbol,
                    "side": d.position.side.value,
                    "qty": d.position.qty,
                    "entry": round(d.position.avg_entry_price, 4),
                    "price": round(d.position.current_price, 4),
                    "unrealized_pct": round(d.position.unrealized_plpc * 100, 3),
                    "held_days": d.held_days,
                    "action": d.reason or "hold",
                    "accepted": None if d.result is None else d.result.accepted,
                    "message": "" if d.result is None else d.result.message,
                }
                for d in self.decisions
            ],
        }


# -- the entry ledger ---------------------------------------------------


def record_entry(path: str, symbol: str, at: _dt.datetime) -> None:
    """Note when a symbol was opened, so the time stop can age it.

    Kept as a flat JSON object keyed by symbol. Re-opening a symbol overwrites
    its timestamp, which is the behaviour the time stop wants: the clock runs
    from the most recent entry.
    """
    ledger = read_ledger(path)
    ledger[symbol.upper()] = at.astimezone(_dt.timezone.utc).isoformat()
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(ledger, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)  # atomic, so a crash cannot leave a half-written file


def read_ledger(path: str) -> dict[str, str]:
    """Read the entry ledger, returning empty if it does not exist yet."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("could not read the entry ledger %s: %s", path, exc)
        return {}
    if not isinstance(payload, dict):
        log.warning("entry ledger %s is not an object; ignoring it", path)
        return {}
    return {str(k).upper(): str(v) for k, v in payload.items()}


def prune_ledger(path: str, open_symbols: Sequence[str]) -> int:
    """Drop ledger entries for symbols no longer held. Returns how many."""
    ledger = read_ledger(path)
    keep = {s.upper() for s in open_symbols}
    stale = [symbol for symbol in ledger if symbol not in keep]
    if not stale:
        return 0
    for symbol in stale:
        ledger.pop(symbol, None)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(ledger, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)
    return len(stale)


def _held_days(
    ledger: dict[str, str], symbol: str, now: _dt.datetime
) -> float | None:
    raw = ledger.get(symbol.upper())
    if not raw:
        return None
    try:
        opened = _dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        log.warning("ledger entry for %s is not a timestamp: %r", symbol, raw)
        return None
    if opened.tzinfo is None:
        opened = opened.replace(tzinfo=_dt.timezone.utc)
    return (now - opened).total_seconds() / 86_400.0


# -- the decision -------------------------------------------------------


def classify(
    position: BrokerPosition,
    config: ExitConfig,
    held_days: float | None,
) -> str | None:
    """Which rule, if any, closes this position.

    The loss rule is checked first: when a position has gapped through both
    thresholds between runs, taking the loss is the conservative reading.
    """
    if config.stop_loss is not None and position.unrealized_plpc <= -config.stop_loss:
        return "stop loss"
    if config.take_profit is not None and position.unrealized_plpc >= config.take_profit:
        return "take profit"
    if (
        config.max_hold_days is not None
        and held_days is not None
        and held_days >= config.max_hold_days
    ):
        return "time stop"
    return None


def run_exit_job(
    broker: ClosableBroker,
    config: ExitConfig | None = None,
    now: _dt.datetime | None = None,
) -> ExitReport:
    """Check every open position and close the ones that qualify."""
    config = config or ExitConfig()
    now = now or _dt.datetime.now(_dt.timezone.utc)
    report = ExitReport(ran_at=now, dry_run=config.dry_run)

    positions = broker.open_positions()
    if not positions:
        report.notes.append("no open positions")
        return report

    ledger: dict[str, str] = {}
    if config.ledger_path:
        ledger = read_ledger(config.ledger_path)
    elif config.max_hold_days is not None:
        report.notes.append(
            "no ledger given, so the time stop is skipped; the profit and loss "
            "rules still apply"
        )

    if config.max_hold_days is not None and config.ledger_path and not ledger:
        report.notes.append(
            "the ledger is empty, so no position can be aged; run the entry job "
            "with --ledger so it records entry times"
        )

    for position in positions:
        held = _held_days(ledger, position.symbol, now)
        reason = classify(position, config, held)
        decision = ExitDecision(position=position, reason=reason, held_days=
                                None if held is None else round(held, 3))

        if reason and not config.dry_run:
            decision.result = broker.close_position(position.symbol)
            if decision.result.accepted:
                log.info(
                    "closed %s (%s, %+.2f%%)",
                    position.symbol, reason, position.unrealized_plpc * 100,
                )
            else:
                log.error(
                    "failed to close %s: %s", position.symbol, decision.result.message
                )
        elif reason:
            log.info(
                "would close %s (%s, %+.2f%%)",
                position.symbol, reason, position.unrealized_plpc * 100,
            )

        report.decisions.append(decision)

    if config.ledger_path and not config.dry_run:
        closed = {d.position.symbol for d in report.closed if d.result and d.result.accepted}
        remaining = [
            p.symbol for p in positions if p.symbol not in closed
        ]
        pruned = prune_ledger(config.ledger_path, remaining)
        if pruned:
            log.info("pruned %d stale ledger entry(ies)", pruned)

    return report


def format_exit_report(report: ExitReport) -> str:
    """Human-readable summary."""
    lines = [
        f"Exit job -- {report.ran_at:%Y-%m-%d %H:%M} UTC",
        f"  mode            : {'DRY RUN' if report.dry_run else 'LIVE'}",
        f"  open positions  : {len(report.decisions)}",
    ]
    for note in report.notes:
        lines.append(f"  note            : {note}")

    if not report.decisions:
        return "\n".join(lines)

    lines.append("")
    lines.append(f"  {'SYMBOL':<8}{'SIDE':<7}{'ENTRY':>10}{'PRICE':>10}{'P&L':>9}{'HELD':>7}  ACTION")
    for decision in report.decisions:
        position = decision.position
        held = "-" if decision.held_days is None else f"{decision.held_days:.1f}d"
        action = decision.reason or "hold"
        if decision.result is not None and not decision.result.accepted:
            action = f"{action} FAILED: {decision.result.message[:40]}"
        lines.append(
            f"  {position.symbol:<8}{position.side.value:<7}"
            f"{position.avg_entry_price:>10,.2f}{position.current_price:>10,.2f}"
            f"{position.unrealized_plpc * 100:>+8.2f}%{held:>7}  {action}"
        )

    closing = report.closed
    lines.append("")
    if report.dry_run:
        lines.append(f"  {len(closing)} position(s) would be closed. Nothing was sent.")
    else:
        accepted = [d for d in closing if d.result and d.result.accepted]
        lines.append(f"  {len(accepted)}/{len(closing)} close order(s) accepted.")
    return "\n".join(lines)
