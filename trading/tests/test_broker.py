"""Order execution: the paper book, and Alpaca payload construction."""

from __future__ import annotations

import json
import unittest
from unittest import mock

from wsj_headline_trader.broker import (
    AlpacaBroker,
    BrokerError,
    PaperBroker,
    StaticPriceProvider,
)
from wsj_headline_trader.models import Order, Side


class PaperBrokerTests(unittest.TestCase):
    def test_buy_converts_notional_to_shares(self):
        broker = PaperBroker(prices=StaticPriceProvider({"NVDA": 200.0}))
        result = broker.submit(Order(symbol="NVDA", side=Side.BUY, notional=1000.0))
        self.assertTrue(result.accepted)
        self.assertEqual(result.filled_qty, 5.0)
        self.assertEqual(broker.positions["NVDA"], 5.0)

    def test_short_records_a_negative_position(self):
        broker = PaperBroker(prices=StaticPriceProvider({"BA": 100.0}))
        broker.submit(Order(symbol="BA", side=Side.SHORT, notional=500.0))
        self.assertEqual(broker.positions["BA"], -5.0)

    def test_positions_accumulate(self):
        broker = PaperBroker(prices=StaticPriceProvider({"F": 10.0}))
        broker.submit(Order(symbol="F", side=Side.BUY, notional=100.0))
        broker.submit(Order(symbol="F", side=Side.BUY, notional=100.0))
        broker.submit(Order(symbol="F", side=Side.SHORT, notional=50.0))
        self.assertEqual(broker.positions["F"], 15.0)
        self.assertEqual(len(broker.orders), 3)

    def test_explicit_quantity_is_honoured(self):
        broker = PaperBroker(prices=StaticPriceProvider({"F": 10.0}))
        result = broker.submit(Order(symbol="F", side=Side.BUY, qty=7.0))
        self.assertEqual(result.filled_qty, 7.0)

    def test_missing_price_is_rejected(self):
        broker = PaperBroker(prices=StaticPriceProvider())
        result = broker.submit(Order(symbol="ZZZ", side=Side.BUY, notional=100.0))
        self.assertFalse(result.accepted)
        self.assertIn("no price", result.message)
        self.assertNotIn("ZZZ", broker.positions)

    def test_zero_notional_is_rejected(self):
        broker = PaperBroker(prices=StaticPriceProvider({"F": 10.0}))
        result = broker.submit(Order(symbol="F", side=Side.BUY, notional=0.0))
        self.assertFalse(result.accepted)
        self.assertEqual(broker.positions, {})

    def test_client_order_id_is_echoed(self):
        broker = PaperBroker(prices=StaticPriceProvider(default=50.0))
        result = broker.submit(
            Order(symbol="F", side=Side.BUY, notional=100.0, client_order_id="wsj-1")
        )
        self.assertEqual(result.broker_order_id, "wsj-1")


class AlpacaCredentialTests(unittest.TestCase):
    def test_missing_credentials_raise(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(BrokerError):
                AlpacaBroker()

    def test_credentials_from_environment(self):
        with mock.patch.dict(
            "os.environ",
            {"APCA_API_KEY_ID": "key", "APCA_API_SECRET_KEY": "secret"},
            clear=True,
        ):
            broker = AlpacaBroker()
        self.assertEqual(broker.key_id, "key")
        self.assertIn("paper", broker.base_url)

    def test_live_flag_selects_the_live_endpoint(self):
        broker = AlpacaBroker(key_id="k", secret_key="s", paper=False)
        self.assertEqual(broker.base_url, "https://api.alpaca.markets")


class AlpacaOrderTests(unittest.TestCase):
    def setUp(self):
        self.broker = AlpacaBroker(key_id="k", secret_key="s")

    def test_long_uses_a_notional_order(self):
        with mock.patch.object(
            self.broker, "_request", return_value={"id": "abc", "status": "accepted"}
        ) as request:
            result = self.broker.submit(Order(symbol="NVDA", side=Side.BUY, notional=1000.0))

        url, payload = request.call_args[0]
        self.assertTrue(url.endswith("/v2/orders"))
        self.assertEqual(payload["side"], "buy")
        self.assertEqual(payload["notional"], "1000.0")
        self.assertNotIn("qty", payload)
        self.assertTrue(result.accepted)
        self.assertEqual(result.broker_order_id, "abc")

    def test_short_is_sent_as_a_sell_sized_in_whole_shares(self):
        with mock.patch.object(self.broker, "last_price", return_value=300.0):
            with mock.patch.object(
                self.broker, "_request", return_value={"id": "def", "status": "accepted"}
            ) as request:
                result = self.broker.submit(Order(symbol="BA", side=Side.SHORT, notional=1000.0))

        _, payload = request.call_args[0]
        self.assertEqual(payload["side"], "sell")
        self.assertEqual(payload["qty"], "3")  # 1000 // 300
        self.assertNotIn("notional", payload)
        self.assertTrue(result.accepted)

    def test_short_without_a_price_is_refused_not_guessed(self):
        with mock.patch.object(self.broker, "last_price", return_value=None):
            with mock.patch.object(self.broker, "_request") as request:
                result = self.broker.submit(Order(symbol="BA", side=Side.SHORT, notional=1000.0))
        request.assert_not_called()
        self.assertFalse(result.accepted)
        self.assertIn("no price available", result.message)

    def test_short_below_one_share_is_refused(self):
        with mock.patch.object(self.broker, "last_price", return_value=5000.0):
            result = self.broker.submit(Order(symbol="BKNG", side=Side.SHORT, notional=100.0))
        self.assertFalse(result.accepted)
        self.assertIn("below one share", result.message)

    def test_http_failure_is_reported_as_a_rejection(self):
        with mock.patch.object(self.broker, "_request", side_effect=BrokerError("403 forbidden")):
            result = self.broker.submit(Order(symbol="NVDA", side=Side.BUY, notional=100.0))
        self.assertFalse(result.accepted)
        self.assertIn("403", result.message)

    def test_fill_details_are_parsed(self):
        body = {"id": "x", "status": "filled", "filled_qty": "4", "filled_avg_price": "250.5"}
        with mock.patch.object(self.broker, "_request", return_value=body):
            result = self.broker.submit(Order(symbol="NVDA", side=Side.BUY, qty=4.0))
        self.assertEqual(result.filled_qty, 4.0)
        self.assertEqual(result.filled_price, 250.5)
        self.assertEqual(result.message, "filled")

    def test_headers_carry_the_credentials(self):
        headers = self.broker._headers
        self.assertEqual(headers["APCA-API-KEY-ID"], "k")
        self.assertEqual(headers["APCA-API-SECRET-KEY"], "s")

    def test_last_price_reads_the_latest_trade(self):
        with mock.patch.object(self.broker, "_request", return_value={"trade": {"p": 123.45}}):
            self.assertEqual(self.broker.last_price("NVDA"), 123.45)

    def test_last_price_survives_a_failed_lookup(self):
        with mock.patch.object(self.broker, "_request", side_effect=BrokerError("nope")):
            self.assertIsNone(self.broker.last_price("NVDA"))

    def test_payload_is_json_serialisable(self):
        captured = {}

        def fake_request(url, payload=None):
            captured["body"] = json.dumps(payload)
            return {"id": "x"}

        with mock.patch.object(self.broker, "_request", side_effect=fake_request):
            self.broker.submit(
                Order(symbol="NVDA", side=Side.BUY, notional=100.0, client_order_id="wsj-1")
            )
        self.assertIn("wsj-1", captured["body"])


if __name__ == "__main__":
    unittest.main()
