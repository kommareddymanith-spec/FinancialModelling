"""Ranking by mention frequency, direction, gating and sizing."""

from __future__ import annotations

import datetime as dt
import unittest

from wsj_headline_trader.models import Headline, Side
from wsj_headline_trader.strategy import (
    StrategyConfig,
    build_signals,
    extract_mentions,
)
from wsj_headline_trader.universe import Company, Universe

NOW = dt.datetime(2026, 9, 10, 15, 30, tzinfo=dt.timezone.utc)

UNIVERSE = Universe(
    [
        Company(ticker="NVDA", name="Nvidia", aliases=("Nvidia",)),
        Company(ticker="BA", name="Boeing", aliases=("Boeing",)),
        Company(ticker="F", name="Ford Motor", aliases=("Ford",)),
        Company(ticker="TGT", name="Target", aliases=("Target Corp",), weak_aliases=("Target",)),
    ]
)


def headline(title: str, minutes_ago: int = 5, summary: str = "") -> Headline:
    return Headline(
        title=title,
        summary=summary,
        link=f"https://example.test/{abs(hash(title))}",
        published_at=NOW - dt.timedelta(minutes=minutes_ago),
        source="test",
    )


class ExtractMentionsTests(unittest.TestCase):
    def test_one_mention_per_company_per_article(self):
        mentions = extract_mentions(
            [headline("Nvidia Surges as Nvidia Beats Expectations")], UNIVERSE
        )
        self.assertEqual(len(mentions), 1)
        self.assertEqual(mentions[0].ticker, "NVDA")
        self.assertGreater(mentions[0].sentiment, 0)

    def test_two_companies_share_the_article_score(self):
        mentions = extract_mentions([headline("Nvidia and Boeing Both Plunge")], UNIVERSE)
        self.assertEqual({m.ticker for m in mentions}, {"NVDA", "BA"})
        self.assertEqual(mentions[0].sentiment, mentions[1].sentiment)
        self.assertLess(mentions[0].sentiment, 0)

    def test_articles_naming_nobody_are_dropped(self):
        self.assertEqual(extract_mentions([headline("Treasury Yields Rise")], UNIVERSE), [])

    def test_summary_text_is_searched_too(self):
        mentions = extract_mentions(
            [headline("Chip Stocks Rally", summary="Nvidia led the gains.")], UNIVERSE
        )
        self.assertEqual([m.ticker for m in mentions], ["NVDA"])

    def test_company_name_is_resolved(self):
        mentions = extract_mentions([headline("Ford Shares Jump")], UNIVERSE)
        self.assertEqual(mentions[0].company, "Ford Motor")


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.headlines = [
            headline("Nvidia Shares Surge on Blowout Results"),
            headline("Nvidia Rallies Again as Analysts Upgrade It"),
            headline("Nvidia Extends Its Gains to a Record High"),
            headline("Boeing Shares Plunge After a Fresh Recall"),
            headline("Boeing Warns of a Wider Loss"),
            headline("Ford Shares Jump on Strong Truck Sales"),
        ]

    def test_most_mentioned_company_ranks_first(self):
        signals = build_signals(extract_mentions(self.headlines, UNIVERSE))
        self.assertEqual([s.ticker for s in signals], ["NVDA", "BA", "F"])
        self.assertEqual(signals[0].mention_count, 3)
        self.assertEqual(signals[1].mention_count, 2)

    def test_top_n_limits_the_candidates(self):
        signals = build_signals(
            extract_mentions(self.headlines, UNIVERSE), StrategyConfig(top_n=2)
        )
        self.assertEqual([s.ticker for s in signals], ["NVDA", "BA"])

    def test_positive_coverage_buys_and_negative_shorts(self):
        signals = {s.ticker: s for s in build_signals(extract_mentions(self.headlines, UNIVERSE))}
        self.assertEqual(signals["NVDA"].side, Side.BUY)
        self.assertEqual(signals["BA"].side, Side.SHORT)

    def test_ties_break_on_conviction_then_ticker(self):
        signals = build_signals(
            extract_mentions(
                [
                    headline("Boeing Shares Collapse on Fraud Claims"),
                    headline("Ford Shares Slip Slightly"),
                ],
                UNIVERSE,
            ),
            StrategyConfig(min_mentions=1),
        )
        self.assertEqual([s.ticker for s in signals], ["BA", "F"])

    def test_no_mentions_produces_no_signals(self):
        self.assertEqual(build_signals([]), [])


class GatingTests(unittest.TestCase):
    def test_thinly_covered_company_is_skipped(self):
        signals = build_signals(
            extract_mentions([headline("Nvidia Shares Surge")], UNIVERSE),
            StrategyConfig(min_mentions=2),
        )
        self.assertEqual(signals[0].side, Side.FLAT)
        self.assertFalse(signals[0].tradable)
        self.assertIn("mention", signals[0].skip_reason)

    def test_muted_tone_is_skipped(self):
        signals = build_signals(
            extract_mentions(
                [
                    headline("Ford Shares Slip Modestly"),
                    headline("Ford Drops a Little in Late Trading"),
                ],
                UNIVERSE,
            ),
            StrategyConfig(min_mentions=2, min_abs_sentiment=1.0),
        )
        self.assertFalse(signals[0].tradable)
        self.assertIn("neutral band", signals[0].skip_reason)

    def test_two_sided_coverage_is_skipped(self):
        signals = build_signals(
            extract_mentions(
                [
                    headline("Ford Shares Surge on a Blowout Quarter"),
                    headline("Ford Shares Plunge on a Deep Loss"),
                    headline("Ford Shares Soar to a Record High"),
                ],
                UNIVERSE,
            ),
            StrategyConfig(min_mentions=2, min_agreement=0.9),
        )
        self.assertFalse(signals[0].tradable)
        self.assertIn("two-sided", signals[0].skip_reason)

    def test_unanimous_coverage_passes_the_agreement_gate(self):
        signals = build_signals(
            extract_mentions(
                [
                    headline("Ford Shares Surge on a Blowout Quarter"),
                    headline("Ford Shares Soar to a Record High"),
                ],
                UNIVERSE,
            ),
            StrategyConfig(min_mentions=2, min_agreement=1.0),
        )
        self.assertTrue(signals[0].tradable)
        self.assertEqual(signals[0].agreement(), 1.0)

    def test_unscored_articles_do_not_count_against_agreement(self):
        signals = build_signals(
            extract_mentions(
                [
                    headline("Ford Shares Surge on a Blowout Quarter"),
                    headline("Ford Names a New Chief Operating Officer"),
                    headline("Ford Soars to a Record High"),
                ],
                UNIVERSE,
            ),
            StrategyConfig(min_mentions=2, min_agreement=1.0),
        )
        self.assertTrue(signals[0].tradable)

    def test_skipped_candidates_are_still_reported(self):
        signals = build_signals(
            extract_mentions([headline("Nvidia Shares Surge")], UNIVERSE),
            StrategyConfig(min_mentions=5),
        )
        self.assertEqual(len(signals), 1)
        self.assertIsNotNone(signals[0].skip_reason)


class SizingTests(unittest.TestCase):
    def heavy(self):
        return extract_mentions(
            [
                headline("Nvidia Shares Surge on Blowout Results"),
                headline("Nvidia Soars to a Record High"),
                headline("Nvidia Rallies as Analysts Upgrade It"),
                headline("Nvidia Jumps on a Strong Outlook"),
            ],
            UNIVERSE,
        )

    def test_heavier_coverage_sizes_larger(self):
        light = build_signals(
            extract_mentions(
                [
                    headline("Nvidia Shares Rise Slightly"),
                    headline("Nvidia Gains a Little"),
                ],
                UNIVERSE,
            )
        )[0]
        heavy = build_signals(self.heavy())[0]
        self.assertGreater(heavy.notional, light.notional)

    def test_size_multiplier_is_bounded(self):
        config = StrategyConfig(max_size_multiplier=1.5, notional_per_trade=1000.0)
        signal = build_signals(self.heavy(), config)[0]
        self.assertLessEqual(signal.conviction, 1.5)
        self.assertLessEqual(signal.notional, 1500.0)

    def test_total_notional_is_capped(self):
        mentions = extract_mentions(
            [
                headline("Nvidia Shares Surge on Blowout Results"),
                headline("Nvidia Soars to a Record High"),
                headline("Boeing Shares Plunge on Fraud Claims"),
                headline("Boeing Collapses After a Record Loss"),
                headline("Ford Shares Surge on Blowout Results"),
                headline("Ford Soars to a Record High"),
            ],
            UNIVERSE,
        )
        config = StrategyConfig(notional_per_trade=1000.0, max_total_notional=2000.0)
        signals = build_signals(mentions, config)
        self.assertLessEqual(sum(s.notional for s in signals), 2000.0)
        self.assertTrue(any(s.skip_reason == "max_total_notional reached" for s in signals))

    def test_skipped_signals_carry_no_size(self):
        signals = build_signals(
            extract_mentions([headline("Nvidia Shares Surge")], UNIVERSE),
            StrategyConfig(min_mentions=3),
        )
        self.assertEqual(signals[0].notional, 0.0)


class ConfigValidationTests(unittest.TestCase):
    def test_rejects_bad_values(self):
        for kwargs in (
            {"top_n": 0},
            {"min_mentions": 0},
            {"notional_per_trade": 0},
            {"notional_per_trade": 1000.0, "max_total_notional": 500.0},
            {"min_agreement": 1.5},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    StrategyConfig(**kwargs)

    def test_defaults_are_valid(self):
        config = StrategyConfig()
        self.assertEqual(config.top_n, 5)
        self.assertEqual(config.min_mentions, 2)
        self.assertEqual(config.min_abs_sentiment, 0.5)


if __name__ == "__main__":
    unittest.main()
