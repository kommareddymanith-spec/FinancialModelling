"""The exit-threshold sweep: scoring, ranking, and the held-out check."""

from __future__ import annotations

import io
import json
import unittest
from collections import Counter
from contextlib import redirect_stdout

from wsj_headline_trader.tuning import (
    Cell,
    CellScore,
    RunOutcome,
    SweepConfig,
    _aggregate,
    _eligible,
    _rank,
    _simulate_one,
    run_sweep,
    score_cells,
)
from wsj_headline_trader.tuning_cli import main


def score(
    cell=Cell(0.04, 0.08), median=0.10, mean=0.10, stderr=0.02,
    drawdown=-0.15, worst=-0.20, cut_early=0.4,
) -> CellScore:
    return CellScore(
        cell=cell, runs=10, mean_twr=mean, median_twr=median, stderr_twr=stderr,
        worst_twr=worst, mean_drawdown=drawdown, profitable_share=0.8,
        mean_trades=80.0, mean_win_rate=0.5, exit_mix={}, cut_early_rate=cut_early,
    )


class CellTests(unittest.TestCase):
    def test_label_shows_both_thresholds(self):
        self.assertIn("4%", Cell(0.04, 0.08).label())
        self.assertIn("8%", Cell(0.04, 0.08).label())

    def test_label_shows_none(self):
        self.assertEqual(Cell(None, None).label().count("none"), 2)

    def test_cells_are_hashable_so_they_key_results(self):
        self.assertEqual(len({Cell(0.04, 0.08), Cell(0.04, 0.08)}), 1)


class ConfigTests(unittest.TestCase):
    def test_the_grid_is_the_cross_product(self):
        config = SweepConfig(stops=(None, 0.04), targets=(None, 0.08, 0.12))
        self.assertEqual(len(config.cells()), 6)

    def test_the_baseline_cell_is_always_present(self):
        self.assertIn(Cell(None, None), SweepConfig().cells())

    def test_seed_sets_are_disjoint(self):
        config = SweepConfig(train_seeds=50, validation_seeds=50)
        self.assertFalse(set(config.train_range()) & set(config.validation_range()))

    def test_validation_seeds_are_never_reused_as_train(self):
        config = SweepConfig(train_seeds=9_000, validation_seeds=10)
        self.assertFalse(set(config.train_range()) & set(config.validation_range()))

    def test_validation(self):
        for kwargs in (
            {"train_seeds": 1},
            {"validation_seeds": 1},
            {"stops": ()},
            {"targets": ()},
            {"max_cut_early_rate": 2.0},
            {"objective": "vibes"},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    SweepConfig(**kwargs)


class AggregateTests(unittest.TestCase):
    def test_aggregation_summarises_outcomes(self):
        outcomes = [
            RunOutcome(twr=0.10, max_drawdown=-0.1, trades=10, win_rate=0.5,
                       exit_reasons=Counter({"stop loss": 4, "time stop": 6}),
                       stop_exits=4, stop_exits_cut_early=1),
            RunOutcome(twr=-0.05, max_drawdown=-0.3, trades=8, win_rate=0.4,
                       exit_reasons=Counter({"stop loss": 2, "time stop": 6}),
                       stop_exits=2, stop_exits_cut_early=2),
        ]
        result = _aggregate(Cell(0.04, 0.08), outcomes)
        self.assertEqual(result.runs, 2)
        self.assertAlmostEqual(result.mean_twr, 0.025, places=9)
        self.assertAlmostEqual(result.worst_twr, -0.05, places=9)
        self.assertAlmostEqual(result.mean_drawdown, -0.2, places=9)
        self.assertAlmostEqual(result.profitable_share, 0.5, places=9)
        self.assertAlmostEqual(result.cut_early_rate, 0.5, places=9)
        self.assertAlmostEqual(result.exit_mix["stop loss"], 6 / 18, places=9)

    def test_no_stop_exits_leaves_the_cut_early_rate_undefined(self):
        outcomes = [
            RunOutcome(twr=0.1, max_drawdown=-0.1, trades=5, win_rate=0.6,
                       exit_reasons=Counter({"time stop": 5}))
        ]
        self.assertIsNone(_aggregate(Cell(None, None), outcomes).cut_early_rate)

    def test_rows_serialise(self):
        payload = json.loads(json.dumps(score().as_row()))
        self.assertEqual(payload["stop_loss"], 0.04)


class EligibilityTests(unittest.TestCase):
    CONFIG = SweepConfig(max_cut_early_rate=0.5)

    def test_a_reasonable_stop_is_eligible(self):
        self.assertIsNone(_eligible(score(cut_early=0.45), self.CONFIG))

    def test_a_stop_that_cuts_too_often_is_rejected(self):
        reason = _eligible(score(cut_early=0.8), self.CONFIG)
        self.assertIsNotNone(reason)
        self.assertIn("recovered", reason)

    def test_the_baseline_has_no_stop_exits_to_judge(self):
        self.assertIsNone(_eligible(score(cell=Cell(None, None), cut_early=None), self.CONFIG))


class RankingTests(unittest.TestCase):
    def test_return_objective_takes_the_highest_median(self):
        config = SweepConfig(objective="return")
        baseline = score(cell=Cell(None, None), median=0.10, drawdown=-0.30)
        cells = [
            score(cell=Cell(0.02, 0.08), median=0.20, drawdown=-0.25),
            score(cell=Cell(0.04, 0.08), median=0.30, drawdown=-0.28),
        ]
        self.assertEqual(_rank(cells, baseline, config)[0].cell, Cell(0.04, 0.08))

    def test_tail_risk_objective_takes_the_smallest_drawdown(self):
        config = SweepConfig(objective="tail-risk")
        baseline = score(cell=Cell(None, None), median=0.10, stderr=0.02, drawdown=-0.30)
        cells = [
            score(cell=Cell(0.02, 0.08), median=0.20, drawdown=-0.12),
            score(cell=Cell(0.04, 0.08), median=0.30, drawdown=-0.28),
        ]
        self.assertEqual(_rank(cells, baseline, config)[0].cell, Cell(0.02, 0.08))

    def test_tail_risk_discards_cells_that_pay_in_return(self):
        config = SweepConfig(objective="tail-risk")
        baseline = score(cell=Cell(None, None), median=0.20, stderr=0.01, drawdown=-0.30)
        cells = [
            # Smallest drawdown, but gives up far too much return.
            score(cell=Cell(0.01, 0.04), median=0.01, drawdown=-0.02),
            score(cell=Cell(0.04, 0.12), median=0.22, drawdown=-0.18),
        ]
        self.assertEqual(_rank(cells, baseline, config)[0].cell, Cell(0.04, 0.12))

    def test_tail_risk_falls_back_when_nothing_clears_the_floor(self):
        config = SweepConfig(objective="tail-risk")
        baseline = score(cell=Cell(None, None), median=0.50, stderr=0.01, drawdown=-0.30)
        cells = [score(cell=Cell(0.02, 0.08), median=0.10, drawdown=-0.12)]
        self.assertEqual(len(_rank(cells, baseline, config)), 1)

    def test_a_trivially_behind_cell_is_not_discarded(self):
        # Within one standard error of the baseline counts as not giving up return.
        config = SweepConfig(objective="tail-risk")
        baseline = score(cell=Cell(None, None), median=0.20, stderr=0.05, drawdown=-0.30)
        cells = [score(cell=Cell(0.02, 0.08), median=0.18, drawdown=-0.10)]
        self.assertEqual(_rank(cells, baseline, config)[0].cell, Cell(0.02, 0.08))


class SimulationTests(unittest.TestCase):
    def test_one_job_produces_an_outcome(self):
        config = SweepConfig(days=120, train_seeds=2, validation_seeds=2)
        outcome = _simulate_one((Cell(0.04, 0.08), 1, config))
        self.assertGreater(outcome.trades, 0)
        self.assertIsInstance(outcome.twr, float)
        self.assertGreaterEqual(outcome.stop_exits, 0)

    def test_the_same_job_is_reproducible(self):
        config = SweepConfig(days=120)
        first = _simulate_one((Cell(0.04, 0.08), 5, config))
        second = _simulate_one((Cell(0.04, 0.08), 5, config))
        self.assertEqual(first.twr, second.twr)
        self.assertEqual(first.trades, second.trades)

    def test_a_tighter_stop_produces_more_stop_exits(self):
        config = SweepConfig(days=252)
        tight = _simulate_one((Cell(0.02, None), 3, config))
        wide = _simulate_one((Cell(0.12, None), 3, config))
        self.assertGreaterEqual(
            tight.exit_reasons["stop loss"], wide.exit_reasons["stop loss"]
        )

    def test_score_cells_covers_every_cell(self):
        config = SweepConfig(days=60, stops=(None, 0.04), targets=(None,))
        scores = score_cells(config.cells(), [1, 2], config, workers=1)
        self.assertEqual(len(scores), 2)
        self.assertTrue(all(s.runs == 2 for s in scores))


class SweepTests(unittest.TestCase):
    def test_a_small_sweep_selects_and_validates(self):
        config = SweepConfig(
            stops=(None, 0.04), targets=(None, 0.08),
            train_seeds=4, validation_seeds=4, days=120, workers=1,
        )
        result = run_sweep(config, validate_top=2)
        self.assertEqual(len(result.train), 4)
        self.assertEqual(result.baseline.cell, Cell(None, None))
        self.assertIsNotNone(result.chosen)
        # The baseline is always re-scored so the winner has something to beat.
        self.assertIn(Cell(None, None), result.validation)
        self.assertGreater(result.simulations, 0)

    def test_every_grid_point_is_scored_on_every_train_seed(self):
        config = SweepConfig(
            stops=(None, 0.04), targets=(None,),
            train_seeds=3, validation_seeds=3, days=60, workers=1,
        )
        result = run_sweep(config, validate_top=1)
        self.assertTrue(all(s.runs == 3 for s in result.train))


class TuningCliTests(unittest.TestCase):
    def run_cli(self, *argv):
        out = io.StringIO()
        with redirect_stdout(out):
            code = main(list(argv))
        return code, out.getvalue()

    def test_a_small_sweep_prints_a_grid_and_a_held_out_check(self):
        code, out = self.run_cli(
            "--stops", "none,0.04", "--targets", "none,0.08",
            "--train-seeds", "4", "--validation-seeds", "4",
            "--days", "120", "--validate-top", "2", "--workers", "1",
        )
        self.assertEqual(code, 0)
        self.assertIn("simulated years", out)
        self.assertIn("Held-out check", out)
        self.assertIn("Selection", out)
        self.assertIn("Read the trade-off, not the number", out)

    def test_json_output_carries_the_grid(self):
        code, out = self.run_cli(
            "--stops", "none,0.04", "--targets", "none",
            "--train-seeds", "4", "--validation-seeds", "4",
            "--days", "60", "--validate-top", "1", "--workers", "1", "--json",
        )
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(len(payload["train"]), 2)
        self.assertIn("baseline", payload)
        self.assertIn("chosen", payload)

    def test_a_bad_threshold_is_a_usage_error(self):
        with self.assertRaises(SystemExit):
            self.run_cli("--stops", "tight")

    def test_a_bad_seed_count_is_a_usage_error(self):
        code, _ = self.run_cli("--train-seeds", "1")
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
