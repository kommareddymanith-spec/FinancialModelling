"""Lexicon scoring: direction, phrases, negation and weighting."""

from __future__ import annotations

import unittest

from wsj_headline_trader.sentiment import (
    NEGATIVE_TERMS,
    POSITIVE_TERMS,
    score_text,
)


class DirectionTests(unittest.TestCase):
    def test_positive_headline(self):
        score = score_text("Nvidia Shares Surge on Strong Demand")
        self.assertGreater(score.score, 0)
        self.assertIn("surge", score.positive_hits)

    def test_negative_headline(self):
        score = score_text("Boeing Shares Plunge After Recall")
        self.assertLess(score.score, 0)
        self.assertIn("plunge", score.negative_hits)

    def test_neutral_headline_scores_zero(self):
        score = score_text("Disney Names a New Chief Operating Officer")
        self.assertEqual(score.score, 0.0)
        self.assertEqual(score.hit_count, 0)
        self.assertEqual(score.polarity, 0.0)

    def test_empty_text(self):
        score = score_text("")
        self.assertEqual(score.score, 0.0)
        self.assertEqual(score.terms, [])


class PhraseTests(unittest.TestCase):
    def test_phrase_beats_its_component_words(self):
        # "record" alone is not in the lexicon and "loss" is negative; the
        # phrase must be scored once, negatively, not cancelled out.
        score = score_text("Boeing Posts Record Loss")
        self.assertEqual(score.negative_hits, ["record loss"])
        self.assertEqual(score.positive_hits, [])
        self.assertEqual(score.score, -2.0)

    def test_cuts_guidance_is_not_read_as_a_bare_cut(self):
        score = score_text("Target Cuts Guidance")
        self.assertEqual(score.negative_hits, ["cuts guidance"])
        self.assertEqual(score.score, -2.0)

    def test_beats_expectations_scores_once(self):
        score = score_text("Ford Beats Expectations")
        self.assertEqual(score.positive_hits, ["beats expectations"])

    def test_strikes_a_deal_is_positive_not_a_labour_strike(self):
        score = score_text("Apple Strikes a Deal With Suppliers")
        self.assertGreater(score.score, 0)
        self.assertEqual(score.positive_hits, ["strikes a deal"])

    def test_bare_strike_remains_negative(self):
        self.assertLess(score_text("Boeing Machinists Strike Enters Week Three").score, 0)

    def test_matches_are_non_overlapping(self):
        score = score_text("Record High and Record Loss in One Headline")
        self.assertEqual(score.positive_hits, ["record high"])
        self.assertEqual(score.negative_hits, ["record loss"])

    def test_whitespace_between_phrase_words_is_flexible(self):
        score = score_text("Ford Beats\n   Expectations")
        self.assertEqual(score.positive_hits, ["beats expectations"])


class NegationTests(unittest.TestCase):
    def test_contraction_flips_polarity(self):
        score = score_text("Ford Doesn't Beat Expectations")
        self.assertLess(score.score, 0)
        self.assertIn("beat expectations", score.negated_hits)

    def test_fails_to_flips_polarity(self):
        score = score_text("Intel Fails to Improve Margins")
        self.assertLess(score.score, 0)
        self.assertIn("improve", score.negated_hits)

    def test_negated_negative_becomes_positive(self):
        score = score_text("Pfizer Will Not Cut Guidance")
        self.assertGreater(score.score, 0)
        self.assertEqual(score.negated_hits, ["cut guidance"])
        self.assertEqual(score.positive_hits, ["cut guidance"])

    def test_negator_outside_the_window_does_not_flip(self):
        score = score_text("No One Expected This, but Revenue Surged to a Record High")
        self.assertGreater(score.score, 0)
        self.assertEqual(score.negated_hits, [])

    def test_curly_apostrophe_negator(self):
        score = score_text("Ford doesn’t beat expectations")
        self.assertLess(score.score, 0)


class WeightingTests(unittest.TestCase):
    def test_stronger_words_move_the_score_further(self):
        mild = score_text("Boeing Shares Slip")
        severe = score_text("Boeing Shares Collapse")
        self.assertLess(severe.score, mild.score)

    def test_polarity_normalises_by_hit_count(self):
        score = score_text("Nvidia Surges, Soars, Jumps and Rallies")
        self.assertGreater(score.score, 4.0)
        self.assertLessEqual(abs(score.polarity), 2.5)

    def test_mixed_headline_lands_near_neutral(self):
        score = score_text("Ford Beats Expectations but Cuts Outlook")
        self.assertLess(abs(score.polarity), 0.5)

    def test_lexicons_do_not_overlap(self):
        self.assertEqual(set(POSITIVE_TERMS) & set(NEGATIVE_TERMS), set())

    def test_all_weights_are_positive(self):
        for lexicon in (POSITIVE_TERMS, NEGATIVE_TERMS):
            for term, weight in lexicon.items():
                self.assertGreater(weight, 0, f"{term} has a non-positive weight")


if __name__ == "__main__":
    unittest.main()
