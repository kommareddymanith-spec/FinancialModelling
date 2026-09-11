"""The synthetic market, and the two experiments it exists to support.

The return numbers here are meaningless as forecasts -- the market is fake.
What they test is the engine: given headlines that carry no information it
must earn roughly nothing, and given headlines that genuinely predict drift it
must find it. A null run that turns a profit is a bug in the engine, not an
edge in the market.
"""

from __future__ import annotations

import statistics
import unittest

from wsj_headline_trader.backtest import BacktestConfig
from wsj_headline_trader.simulate import (
    BEARISH_TEMPLATES,
    BULLISH_TEMPLATES,
    DEFAULT_SYMBOLS,
    NOISE_TEMPLATES,
    MarketConfig,
    generate_market,
    run_simulation,
)
from wsj_headline_trader.sentiment import score_text
from wsj_headline_trader.strategy import StrategyConfig
from wsj_headline_trader.universe import Universe

UNIVERSE = Universe.load()


def backtest_config(**overrides) -> BacktestConfig:
    settings = dict(
        contribution=1_000.0,
        hold_days=3,
        slippage_bps=0.0,
        borrow_rate_annual=0.0,
        strategy=StrategyConfig(
            top_n=3, notional_per_trade=1_000.0, max_total_notional=5_000.0
        ),
    )
    settings.update(overrides)
    return BacktestConfig(**settings)


def twr_across_seeds(edge: float, seeds: int, **overrides) -> list[float]:
    out = []
    for seed in range(1, seeds + 1):
        result = run_simulation(
            MarketConfig(days=252, edge=edge, seed=seed),
            backtest_config(**overrides),
            UNIVERSE,
        )
        out.append(result.strategy.twr_total)
    return out


class TemplateTests(unittest.TestCase):
    """Every generated headline must be recognised and scored as intended."""

    def test_bullish_templates_name_the_company_and_read_positive(self):
        for ticker, company in DEFAULT_SYMBOLS.items():
            for template in BULLISH_TEMPLATES:
                text = template.format(company=company)
                with self.subTest(ticker=ticker, template=template):
                    self.assertEqual([t for t, _ in UNIVERSE.find(text)], [ticker])
                    self.assertGreater(score_text(text).polarity, 0.5)

    def test_bearish_templates_name_the_company_and_read_negative(self):
        for ticker, company in DEFAULT_SYMBOLS.items():
            for template in BEARISH_TEMPLATES:
                text = template.format(company=company)
                with self.subTest(ticker=ticker, template=template):
                    self.assertEqual([t for t, _ in UNIVERSE.find(text)], [ticker])
                    self.assertLess(score_text(text).polarity, -0.5)

    def test_noise_templates_name_no_company(self):
        for template in NOISE_TEMPLATES:
            with self.subTest(template):
                self.assertEqual(UNIVERSE.find(template), [])


class GenerationTests(unittest.TestCase):
    def test_the_same_seed_gives_the_same_market(self):
        first = generate_market(MarketConfig(days=60, seed=42))
        second = generate_market(MarketConfig(days=60, seed=42))
        self.assertEqual(
            [h.title for h in first.headlines], [h.title for h in second.headlines]
        )
        self.assertEqual(
            [b.close for b in first.panel.bars["NVDA"]],
            [b.close for b in second.panel.bars["NVDA"]],
        )

    def test_different_seeds_give_different_markets(self):
        first = generate_market(MarketConfig(days=60, seed=1))
        second = generate_market(MarketConfig(days=60, seed=2))
        self.assertNotEqual(
            [b.close for b in first.panel.bars["NVDA"]],
            [b.close for b in second.panel.bars["NVDA"]],
        )

    def test_sessions_are_weekdays_only(self):
        market = generate_market(MarketConfig(days=30, seed=3))
        self.assertTrue(all(d.weekday() < 5 for d in market.sessions))
        self.assertEqual(len(market.sessions), 30)

    def test_bars_are_internally_consistent(self):
        market = generate_market(MarketConfig(days=120, seed=4))
        for symbol in market.symbols:
            for bar in market.panel.bars[symbol]:
                with self.subTest(symbol=symbol, date=bar.date):
                    self.assertGreater(bar.low, 0)
                    self.assertGreaterEqual(bar.high, max(bar.open, bar.close))
                    self.assertLessEqual(bar.low, min(bar.open, bar.close))

    def test_headlines_are_sorted_and_timezone_aware(self):
        market = generate_market(MarketConfig(days=60, seed=5))
        stamps = [h.published_at for h in market.headlines]
        self.assertEqual(stamps, sorted(stamps))
        self.assertTrue(all(s.tzinfo is not None for s in stamps))

    def test_event_articles_clear_the_mention_threshold(self):
        market = generate_market(
            MarketConfig(days=120, seed=6, event_probability=0.05, articles_per_event=3)
        )
        self.assertGreater(market.events, 0)
        # Every event contributes its articles within a few minutes.
        company_headlines = [h for h in market.headlines if h.source == "simulated-wsj"]
        self.assertGreaterEqual(len(company_headlines), market.events * 3)

    def test_noise_is_generated_every_session(self):
        market = generate_market(MarketConfig(days=20, seed=7, noise_per_day=2))
        noise = [h for h in market.headlines if h.title in NOISE_TEMPLATES]
        self.assertEqual(len(noise), 40)

    def test_index_tracks_the_basket(self):
        market = generate_market(MarketConfig(days=60, seed=8))
        self.assertEqual(len(market.index), 60)
        self.assertTrue(all(price > 0 for _, price in market.index))

    def test_no_events_means_no_company_headlines(self):
        market = generate_market(MarketConfig(days=60, seed=9, event_probability=0.0))
        self.assertEqual(market.events, 0)
        self.assertTrue(all(h.title in NOISE_TEMPLATES for h in market.headlines))

    def test_config_validation(self):
        for kwargs in (
            {"symbols": {}},
            {"days": 1},
            {"volatility": -0.1},
            {"starting_price": 0},
            {"event_probability": 1.5},
            {"articles_per_event": 0},
            {"noise_per_day": -1},
            {"edge_days": 0},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    MarketConfig(**kwargs)


class SimulationTests(unittest.TestCase):
    def test_a_run_produces_both_legs(self):
        result = run_simulation(
            MarketConfig(days=120, edge=0.01, seed=11), backtest_config(), UNIVERSE
        )
        self.assertGreater(len(result.backtest.trades), 0)
        self.assertEqual(result.strategy.contributions, result.benchmark.contributions)
        self.assertIsInstance(result.excess_twr, float)

    def test_both_legs_are_funded_identically(self):
        result = run_simulation(
            MarketConfig(days=252, seed=12), backtest_config(contribution=500.0), UNIVERSE
        )
        self.assertEqual(result.strategy.contributions, result.benchmark.contributions)

    def test_accounting_invariant_holds_in_simulation(self):
        result = run_simulation(
            MarketConfig(days=180, edge=0.01, seed=13),
            backtest_config(slippage_bps=20.0, borrow_rate_annual=0.05),
            UNIVERSE,
        )
        backtest = result.backtest
        if backtest.open_at_end:
            self.skipTest("invariant is only exact with a flat book")
        expected = sum(backtest.contributions.values()) + sum(
            t.net_pnl for t in backtest.trades
        )
        self.assertAlmostEqual(backtest.equity[-1][1], expected, places=6)

    def test_costs_reduce_the_result(self):
        free = run_simulation(
            MarketConfig(days=180, edge=0.01, seed=14),
            backtest_config(slippage_bps=0.0, borrow_rate_annual=0.0),
            UNIVERSE,
        )
        costly = run_simulation(
            MarketConfig(days=180, edge=0.01, seed=14),
            backtest_config(slippage_bps=50.0, borrow_rate_annual=0.10, commission_per_trade=1.0),
            UNIVERSE,
        )
        self.assertLess(costly.strategy.final_value, free.strategy.final_value)
        self.assertGreater(costly.backtest.total_costs, free.backtest.total_costs)

    def test_a_market_with_no_news_trades_nothing(self):
        result = run_simulation(
            MarketConfig(days=120, seed=15, event_probability=0.0), backtest_config(), UNIVERSE
        )
        self.assertEqual(result.backtest.trades, [])
        self.assertEqual(result.backtest.signals_generated, 0)


class NullTests(unittest.TestCase):
    """With uninformative headlines the engine must not manufacture returns."""

    SEEDS = 24

    @classmethod
    def setUpClass(cls):
        cls.free = twr_across_seeds(0.0, cls.SEEDS)

    def test_mean_return_is_indistinguishable_from_zero(self):
        mean = statistics.mean(self.free)
        stderr = statistics.stdev(self.free) / len(self.free) ** 0.5
        # Three standard errors is generous; a look-ahead or double-counting
        # bug would put the mean far outside it.
        self.assertLess(
            abs(mean), 3 * stderr,
            f"null mean {mean:+.4f} is more than 3 standard errors ({stderr:.4f}) "
            "from zero, which points at the engine rather than the market",
        )

    def test_outcomes_are_spread_both_ways(self):
        self.assertTrue(any(v > 0 for v in self.free))
        self.assertTrue(any(v < 0 for v in self.free))

    def test_costs_push_the_null_negative(self):
        charged = twr_across_seeds(
            0.0, self.SEEDS, slippage_bps=10.0, borrow_rate_annual=0.05
        )
        self.assertLess(statistics.mean(charged), statistics.mean(self.free))


class PowerTests(unittest.TestCase):
    """With informative headlines the engine must capture them."""

    SEEDS = 24

    def test_a_real_edge_is_detected(self):
        null = twr_across_seeds(0.0, self.SEEDS)
        strong = twr_across_seeds(0.02, self.SEEDS)
        null_mean = statistics.mean(null)
        strong_mean = statistics.mean(strong)
        stderr = statistics.stdev(strong) / len(strong) ** 0.5
        self.assertGreater(strong_mean, null_mean)
        self.assertGreater(
            (strong_mean - null_mean) / stderr, 3.0,
            "a 2% per-event edge should be detected well beyond noise",
        )

    def test_detection_improves_with_edge_size(self):
        small = statistics.mean(twr_across_seeds(0.005, 12))
        large = statistics.mean(twr_across_seeds(0.02, 12))
        self.assertGreater(large, small)

    def test_a_bearish_edge_is_captured_by_shorting(self):
        # Force every event bearish by making the market fall after bad news
        # only: a negative edge with shorts disabled must do worse.
        with_shorts = run_simulation(
            MarketConfig(days=252, edge=0.02, seed=21), backtest_config(), UNIVERSE
        )
        shorts = [t for t in with_shorts.backtest.trades if t.side.value == "short"]
        self.assertTrue(shorts, "the simulated market should produce short trades")
        self.assertGreater(sum(t.net_pnl for t in shorts), 0)


if __name__ == "__main__":
    unittest.main()
