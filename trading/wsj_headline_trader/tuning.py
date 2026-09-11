"""Sweeping exit thresholds across many simulated sessions.

Searches a grid of (stop loss, take profit) pairs, scoring each on a set of
simulated years, then re-scores the winner on seeds it never saw.

**What this can and cannot tell you.** The market here is synthetic, so a
threshold that wins is fitted to the simulator's assumptions -- its volatility,
how long its injected edge persists, its cost model -- and not to real
equities. Do not read the winning pair as a real-world setting. What the sweep
*does* establish is the shape of the trade-off: where a stop is so tight it
sits inside the noise and cuts trades that would have recovered, and where a
target is so far away it never fires. That shape is a property of the
strategy's own mechanics and carries over; the exact number does not.

The train/validation split exists because picking the maximum of a noisy grid
overstates it. With 42 cells and a few percent of standard error per cell, the
best training cell is partly lucky. Re-scoring on held-out seeds measures how
much of the win was real.
"""

from __future__ import annotations

import concurrent.futures
import logging
import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .backtest import BacktestConfig
from .simulate import MarketConfig, run_simulation
from .strategy import StrategyConfig
from .universe import Universe

log = logging.getLogger(__name__)

DEFAULT_STOPS: tuple[float | None, ...] = (None, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12)
DEFAULT_TARGETS: tuple[float | None, ...] = (None, 0.04, 0.06, 0.08, 0.12, 0.20)

_UNIVERSE: Universe | None = None


def _universe() -> Universe:
    """Load the universe once per worker process."""
    global _UNIVERSE
    if _UNIVERSE is None:
        _UNIVERSE = Universe.load()
    return _UNIVERSE


@dataclass(frozen=True)
class Cell:
    """One point on the grid."""

    stop_loss: float | None
    take_profit: float | None

    def label(self) -> str:
        stop = "none" if self.stop_loss is None else f"{self.stop_loss:.0%}"
        target = "none" if self.take_profit is None else f"{self.take_profit:.0%}"
        return f"{stop:>5} / {target:>5}"


@dataclass
class RunOutcome:
    """One simulated year at one grid point."""

    twr: float
    max_drawdown: float
    trades: int
    win_rate: float
    exit_reasons: Counter = field(default_factory=Counter)
    stop_exits: int = 0
    stop_exits_cut_early: int = 0


@dataclass
class CellScore:
    """A grid point aggregated over many simulated years."""

    cell: Cell
    runs: int
    mean_twr: float
    median_twr: float
    stderr_twr: float
    worst_twr: float
    mean_drawdown: float
    profitable_share: float
    mean_trades: float
    mean_win_rate: float
    exit_mix: dict[str, float]
    #: Of stop-loss exits, the share that would have done better held to the
    #: time stop. High means the stop is inside the noise.
    cut_early_rate: float | None

    def as_row(self) -> dict[str, object]:
        return {
            "stop_loss": self.cell.stop_loss,
            "take_profit": self.cell.take_profit,
            "runs": self.runs,
            "mean_twr": round(self.mean_twr, 6),
            "median_twr": round(self.median_twr, 6),
            "stderr_twr": round(self.stderr_twr, 6),
            "worst_twr": round(self.worst_twr, 6),
            "mean_drawdown": round(self.mean_drawdown, 6),
            "profitable_share": round(self.profitable_share, 4),
            "mean_trades": round(self.mean_trades, 1),
            "mean_win_rate": round(self.mean_win_rate, 4),
            "exit_mix": {k: round(v, 4) for k, v in self.exit_mix.items()},
            "cut_early_rate": (
                None if self.cut_early_rate is None else round(self.cut_early_rate, 4)
            ),
        }


@dataclass
class SweepConfig:
    """How the sweep is run."""

    stops: Sequence[float | None] = DEFAULT_STOPS
    targets: Sequence[float | None] = DEFAULT_TARGETS
    #: Seeds used to choose, and a disjoint set used to check the choice.
    train_seeds: int = 80
    validation_seeds: int = 80
    days: int = 252
    edge: float = 0.01
    hold_days: int = 5
    slippage_bps: float = 5.0
    borrow_rate_annual: float = 0.03
    contribution: float = 1_000.0
    top_n: int = 3
    workers: int | None = None
    #: A stop wrong more often than this share of the time is rejected however
    #: well it scores: the brief was to avoid exiting early.
    max_cut_early_rate: float = 0.5
    #: How to choose among eligible cells.
    #:
    #: ``"tail-risk"`` keeps every cell whose median return is no worse than
    #: running without stops, then takes the smallest drawdown. ``"return"``
    #: simply takes the highest median return.
    #:
    #: The default is ``"tail-risk"`` because of what the grid actually shows:
    #: out of sample, no threshold beats no-stop on return by more than noise,
    #: while every threshold cuts drawdown and worst-case materially. Ranking
    #: on return therefore picks a lucky cell; ranking on drawdown picks a
    #: property that holds across the whole grid.
    objective: str = "tail-risk"

    def __post_init__(self) -> None:
        if self.train_seeds < 2 or self.validation_seeds < 2:
            raise ValueError("each seed set needs at least 2 seeds")
        if not self.stops or not self.targets:
            raise ValueError("the grid needs at least one stop and one target")
        if not 0.0 <= self.max_cut_early_rate <= 1.0:
            raise ValueError("max_cut_early_rate must be between 0 and 1")
        if self.objective not in ("tail-risk", "return"):
            raise ValueError("objective must be 'tail-risk' or 'return'")

    def cells(self) -> list[Cell]:
        return [Cell(stop, target) for stop in self.stops for target in self.targets]

    def train_range(self) -> range:
        return range(1, self.train_seeds + 1)

    def validation_range(self) -> range:
        start = 10_000 + 1
        return range(start, start + self.validation_seeds)


def _simulate_one(job: tuple[Cell, int, SweepConfig]) -> RunOutcome:
    """Run one simulated year at one grid point. Module level, so it pickles."""
    cell, seed, config = job
    result = run_simulation(
        MarketConfig(days=config.days, edge=config.edge, seed=seed),
        BacktestConfig(
            contribution=config.contribution,
            hold_days=config.hold_days,
            stop_loss=cell.stop_loss,
            take_profit=cell.take_profit,
            slippage_bps=config.slippage_bps,
            borrow_rate_annual=config.borrow_rate_annual,
            strategy=StrategyConfig(
                top_n=config.top_n,
                notional_per_trade=config.contribution,
                max_total_notional=config.contribution * 5,
            ),
        ),
        _universe(),
    )
    trades = result.backtest.trades
    stop_exits = [t for t in trades if t.exit_reason == "stop loss"]
    comparable = [t for t in stop_exits if t.cut_early is not None]
    return RunOutcome(
        twr=result.strategy.twr_total,
        max_drawdown=result.strategy.max_drawdown,
        trades=len(trades),
        win_rate=result.backtest.win_rate,
        exit_reasons=Counter(t.exit_reason for t in trades),
        stop_exits=len(comparable),
        stop_exits_cut_early=sum(1 for t in comparable if t.cut_early),
    )


def _aggregate(cell: Cell, outcomes: Sequence[RunOutcome]) -> CellScore:
    returns = [o.twr for o in outcomes]
    stdev = statistics.stdev(returns) if len(returns) > 1 else 0.0
    reasons: Counter = Counter()
    for outcome in outcomes:
        reasons.update(outcome.exit_reasons)
    total_exits = sum(reasons.values())
    stop_exits = sum(o.stop_exits for o in outcomes)
    cut_early = sum(o.stop_exits_cut_early for o in outcomes)

    return CellScore(
        cell=cell,
        runs=len(outcomes),
        mean_twr=statistics.mean(returns),
        median_twr=statistics.median(returns),
        stderr_twr=stdev / len(returns) ** 0.5 if returns else 0.0,
        worst_twr=min(returns),
        mean_drawdown=statistics.mean([o.max_drawdown for o in outcomes]),
        profitable_share=sum(1 for r in returns if r > 0) / len(returns),
        mean_trades=statistics.mean([o.trades for o in outcomes]),
        mean_win_rate=statistics.mean([o.win_rate for o in outcomes]),
        exit_mix={
            reason: count / total_exits for reason, count in reasons.items()
        } if total_exits else {},
        cut_early_rate=cut_early / stop_exits if stop_exits else None,
    )


def score_cells(
    cells: Iterable[Cell],
    seeds: Iterable[int],
    config: SweepConfig,
    workers: int | None = None,
) -> list[CellScore]:
    """Score every cell over every seed, in parallel."""
    cells = list(cells)
    seeds = list(seeds)
    jobs = [(cell, seed, config) for cell in cells for seed in seeds]
    log.info("running %d simulation(s) across %d cell(s)", len(jobs), len(cells))

    collected: dict[Cell, list[RunOutcome]] = {cell: [] for cell in cells}
    workers = workers if workers is not None else config.workers
    if workers and workers > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
            for job, outcome in zip(jobs, pool.map(_simulate_one, jobs, chunksize=8)):
                collected[job[0]].append(outcome)
    else:
        for job in jobs:
            collected[job[0]].append(_simulate_one(job))

    return [_aggregate(cell, collected[cell]) for cell in cells]


@dataclass
class SweepResult:
    """A finished sweep."""

    config: SweepConfig
    train: list[CellScore]
    baseline: CellScore
    chosen: CellScore | None
    validation: dict[Cell, CellScore] = field(default_factory=dict)
    rejected: list[tuple[Cell, str]] = field(default_factory=list)

    @property
    def simulations(self) -> int:
        cells = len(self.config.cells())
        return cells * self.config.train_seeds + len(self.validation) * self.config.validation_seeds


def _eligible(score: CellScore, config: SweepConfig) -> str | None:
    """Why this cell is disqualified, or ``None`` if it is eligible."""
    if score.cut_early_rate is not None and score.cut_early_rate > config.max_cut_early_rate:
        return (
            f"cuts {score.cut_early_rate:.0%} of stopped trades that would have "
            f"recovered (limit {config.max_cut_early_rate:.0%})"
        )
    return None


def _rank(
    scores: list[CellScore], baseline: CellScore, config: SweepConfig
) -> list[CellScore]:
    """Order eligible cells best-first under the configured objective.

    ``tail-risk`` first discards anything that gives up return against running
    without stops, then sorts by drawdown, breaking ties on the worst single
    year. ``return`` sorts on median return alone.
    """
    # Drawdowns are negative, so the shallowest is the largest value: these
    # sorts negate it. Sorting the raw figure ascending would rank the worst
    # drawdown first, which is the opposite of the intent.
    if config.objective == "return":
        return sorted(scores, key=lambda s: (-s.median_twr, -s.mean_drawdown))

    # Keep the cells that do not pay for their risk reduction in return. The
    # comparison is against the baseline's median less one standard error, so
    # a cell is not discarded for being trivially behind.
    floor = baseline.median_twr - baseline.stderr_twr
    keeps = [s for s in scores if s.median_twr >= floor] or list(scores)
    return sorted(keeps, key=lambda s: (-s.mean_drawdown, -s.worst_twr, -s.median_twr))


def run_sweep(config: SweepConfig | None = None, validate_top: int = 3) -> SweepResult:
    """Score the grid, then re-score the leaders on unseen seeds.

    Selection follows ``config.objective``; see :class:`SweepConfig`. Medians
    rather than means throughout, because a single lucky simulated year should
    not decide a threshold.
    """
    config = config or SweepConfig()
    cells = config.cells()

    train = score_cells(cells, config.train_range(), config)
    by_cell = {score.cell: score for score in train}
    baseline = by_cell[Cell(None, None)]

    rejected: list[tuple[Cell, str]] = []
    eligible: list[CellScore] = []
    for score in train:
        reason = _eligible(score, config)
        if reason:
            rejected.append((score.cell, reason))
        else:
            eligible.append(score)

    eligible = _rank(eligible, baseline, config)
    leaders = eligible[:validate_top]

    validation: dict[Cell, CellScore] = {}
    if leaders:
        checked = score_cells(
            [score.cell for score in leaders] + [Cell(None, None)],
            config.validation_range(),
            config,
        )
        validation = {score.cell: score for score in checked}

    chosen = leaders[0] if leaders else None
    return SweepResult(
        config=config,
        train=train,
        baseline=baseline,
        chosen=chosen,
        validation=validation,
        rejected=rejected,
    )
