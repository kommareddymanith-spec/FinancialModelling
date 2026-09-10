"""Company recognition, including the ambiguous-name cases."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from wsj_headline_trader.universe import Company, Universe


class StrongAliasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.universe = Universe.load()

    def tickers(self, text):
        return sorted(t for t, _ in self.universe.find(text))

    def test_bundled_universe_is_not_empty(self):
        self.assertGreater(len(self.universe), 100)

    def test_plain_company_name(self):
        self.assertEqual(self.tickers("Nvidia Shares Surge After Earnings"), ["NVDA"])

    def test_case_insensitive(self):
        self.assertEqual(self.tickers("nvidia beats estimates"), ["NVDA"])

    def test_punctuated_and_spaced_variants(self):
        self.assertEqual(self.tickers("J.P. Morgan Raises Its Forecast"), ["JPM"])
        self.assertEqual(self.tickers("JPMorgan Raises Its Forecast"), ["JPM"])

    def test_alias_starting_with_a_digit(self):
        self.assertEqual(self.tickers("3M Settles Earplug Suit"), ["MMM"])

    def test_alias_containing_an_ampersand(self):
        self.assertEqual(self.tickers("P&G Lifts Its Outlook"), ["PG"])
        self.assertEqual(self.tickers("AT&T Adds Subscribers"), ["T"])

    def test_multiple_companies_in_order_of_appearance(self):
        found = self.universe.find("Ford and General Motors Both Report Tuesday")
        self.assertEqual([t for t, _ in found], ["F", "GM"])

    def test_repeated_name_counts_once(self):
        found = self.universe.find("Tesla Rises. Tesla Again Leads. Tesla Shares Jump.")
        self.assertEqual(len(found), 1)

    def test_longest_alias_wins(self):
        found = self.universe.find("Meta Platforms Said It Will Hire")
        self.assertEqual(found, [("META", "Meta Platforms")])

    def test_substring_of_a_longer_word_is_not_a_match(self):
        self.assertEqual(self.tickers("Fordham University Expands Its Campus"), [])
        self.assertEqual(self.tickers("Intelligence Agencies Warn of Risk"), [])

    def test_unknown_company_is_ignored(self):
        self.assertEqual(self.tickers("Widgets Inc Reports Record Revenue"), [])


class SymbolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.universe = Universe.load()

    def tickers(self, text):
        return sorted(t for t, _ in self.universe.find(text))

    def test_parenthesised_symbol(self):
        self.assertEqual(self.tickers("Chip Stocks Jump, Led by (NVDA)"), ["NVDA"])

    def test_dollar_prefixed_symbol(self):
        self.assertEqual(self.tickers("$TSLA rallies on delivery beat"), ["TSLA"])

    def test_exchange_prefixed_symbol(self):
        self.assertEqual(self.tickers("Shares of Nasdaq: AAPL climbed"), ["AAPL"])

    def test_symbol_not_in_universe_is_ignored(self):
        self.assertEqual(self.tickers("$ZZZZ soars on nothing"), [])

    def test_symbol_and_name_collapse_to_one_mention(self):
        found = Universe.load().find("Apple Inc (AAPL) Reports Tonight")
        self.assertEqual([t for t, _ in found], ["AAPL"])


class WeakAliasTests(unittest.TestCase):
    """Company names that are also ordinary English words."""

    @classmethod
    def setUpClass(cls):
        cls.universe = Universe.load()

    def tickers(self, text):
        return sorted(t for t, _ in self.universe.find(text))

    def test_verb_use_is_not_a_company(self):
        self.assertEqual(self.tickers("Investors Target Small-Cap Stocks as Rally Broadens"), [])

    def test_noun_use_is_not_a_company(self):
        self.assertEqual(self.tickers("Gap Widens Between Rich and Poor"), [])
        self.assertEqual(self.tickers("The Apple Harvest Was Poor This Year"), [])
        self.assertEqual(self.tickers("The Metaverse Is Quiet Again"), [])

    def test_followed_by_a_corporate_noun(self):
        self.assertEqual(self.tickers("Target Shares Slide as Retailer Cuts Guidance"), ["TGT"])
        self.assertEqual(self.tickers("Visa Revenue Tops Estimates"), ["V"])

    def test_followed_by_a_corporate_verb(self):
        self.assertEqual(self.tickers("Apple Faces Fresh Antitrust Suit"), ["AAPL"])
        self.assertEqual(self.tickers("Meta Said It Will Cut 600 Jobs"), ["META"])

    def test_possessive_counts(self):
        self.assertEqual(self.tickers("Target's Quarterly Loss Widens"), ["TGT"])

    def test_curly_apostrophe_possessive_counts(self):
        self.assertEqual(self.tickers("Target’s Quarterly Loss Widens"), ["TGT"])

    def test_preceded_by_a_partitive_head(self):
        self.assertEqual(self.tickers("Shares of Target Fall 8% After Profit Warning"), ["TGT"])

    def test_preceded_by_a_descriptor_noun(self):
        self.assertEqual(self.tickers("Retailer Gap Reported a Wider Loss"), ["GAP"])

    def test_plural_stocks_does_not_qualify(self):
        # "stocks" is deliberately excluded: it marks market-wide prose.
        self.assertEqual(self.tickers("Target Stocks Around the World Fell"), [])

    def test_singular_stock_does_qualify(self):
        self.assertEqual(self.tickers("Target Stock Falls 8%"), ["TGT"])

    def test_strong_alias_bypasses_the_cue_requirement(self):
        self.assertEqual(self.tickers("Gap Inc Will Close Stores"), ["GAP"])


class CustomUniverseTests(unittest.TestCase):
    def test_load_from_a_custom_file(self):
        payload = {
            "_comment": "ignored",
            "ZZZ": {"name": "Zeta Corp", "aliases": ["Zeta"], "weak_aliases": ["Point"]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "u.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            universe = Universe.load(path)

        self.assertEqual(len(universe), 1)
        self.assertIn("ZZZ", universe)
        self.assertEqual(universe.name_for("ZZZ"), "Zeta Corp")
        self.assertEqual(universe.find("Zeta Wins Contract"), [("ZZZ", "Zeta")])
        self.assertEqual(universe.find("Point Shares Rose"), [("ZZZ", "Point")])
        self.assertEqual(universe.find("A Fair Point Indeed"), [])

    def test_comment_keys_are_skipped(self):
        universe = Universe([Company(ticker="AAA", name="Alpha", aliases=("Alpha",))])
        self.assertEqual(universe.name_for("missing"), "missing")

    def test_duplicate_alias_keeps_the_first_company(self):
        universe = Universe(
            [
                Company(ticker="AAA", name="Alpha", aliases=("Acme",)),
                Company(ticker="BBB", name="Beta", aliases=("Acme",)),
            ]
        )
        self.assertEqual(universe.find("Acme Reports"), [("AAA", "Acme")])

    def test_empty_text_finds_nothing(self):
        self.assertEqual(Universe.load().find(""), [])


if __name__ == "__main__":
    unittest.main()
