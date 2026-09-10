"""Order execution.

Two implementations ship here:

* :class:`PaperBroker` -- records orders in memory and tracks a notional
  position book. This is the default, so a misconfigured run cannot trade.
* :class:`AlpacaBroker` -- submits real (or Alpaca-paper) orders over REST.

Short sales are the awkward case: Alpaca accepts notional orders for long
buys but requires whole-share quantities to sell short, so the broker converts
notional into shares using a price provider and refuses the order rather than
guessing when no price is available.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Protocol

from .models import Order, OrderResult, Side

log = logging.getLogger(__name__)

ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"
ALPACA_LIVE_URL = "https://api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"


class BrokerError(RuntimeError):
    """Raised for configuration problems, not for rejected orders."""


class PriceProvider(Protocol):
    """Anything that can quote a last price for a symbol."""

    def last_price(self, symbol: str) -> float | None: ...


@dataclass
class StaticPriceProvider:
    """Fixed prices, for paper trading and tests."""

    prices: dict[str, float] = field(default_factory=dict)
    default: float | None = None

    def last_price(self, symbol: str) -> float | None:
        return self.prices.get(symbol, self.default)


class Broker(Protocol):
    """The only thing the algorithm needs from an execution venue."""

    def submit(self, order: Order) -> OrderResult: ...


@dataclass
class PaperBroker:
    """In-memory broker. Accepts everything and keeps a position book."""

    prices: PriceProvider = field(default_factory=lambda: StaticPriceProvider(default=100.0))
    orders: list[Order] = field(default_factory=list)
    #: symbol -> signed share count (negative is short).
    positions: dict[str, float] = field(default_factory=dict)

    def submit(self, order: Order) -> OrderResult:
        price = self.prices.last_price(order.symbol)
        qty = order.qty
        if qty is None:
            if price is None or price <= 0:
                return OrderResult(
                    order=order,
                    accepted=False,
                    message=f"no price available for {order.symbol}",
                )
            qty = round((order.notional or 0.0) / price, 4)

        if qty <= 0:
            return OrderResult(order=order, accepted=False, message="non-positive quantity")

        signed = qty if order.side is Side.BUY else -qty
        self.positions[order.symbol] = self.positions.get(order.symbol, 0.0) + signed
        self.orders.append(order)

        log.info(
            "PAPER %s %s %s share(s) @ ~%s",
            order.side.value.upper(),
            order.symbol,
            qty,
            price,
        )
        return OrderResult(
            order=order,
            accepted=True,
            broker_order_id=order.client_order_id or f"paper-{uuid.uuid4().hex[:12]}",
            filled_qty=qty,
            filled_price=price,
            message="filled (paper)",
        )


@dataclass
class AlpacaBroker:
    """Submit orders to Alpaca's REST API.

    Credentials come from ``APCA_API_KEY_ID`` / ``APCA_API_SECRET_KEY`` unless
    passed explicitly. ``paper=True`` (the default) points at the paper
    endpoint; flipping it to ``False`` sends live orders.
    """

    key_id: str | None = None
    secret_key: str | None = None
    paper: bool = True
    base_url: str | None = None
    data_url: str = ALPACA_DATA_URL
    timeout: float = 15.0

    def __post_init__(self) -> None:
        self.key_id = self.key_id or os.environ.get("APCA_API_KEY_ID")
        self.secret_key = self.secret_key or os.environ.get("APCA_API_SECRET_KEY")
        if not self.key_id or not self.secret_key:
            raise BrokerError(
                "Alpaca credentials missing: set APCA_API_KEY_ID and "
                "APCA_API_SECRET_KEY, or pass them to AlpacaBroker"
            )
        if self.base_url is None:
            self.base_url = ALPACA_PAPER_URL if self.paper else ALPACA_LIVE_URL
        if not self.paper:
            log.warning("AlpacaBroker is pointed at the LIVE endpoint %s", self.base_url)

    # -- HTTP ------------------------------------------------------------

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.key_id or "",
            "APCA-API-SECRET-KEY": self.secret_key or "",
            "Content-Type": "application/json",
        }

    def _request(self, url: str, payload: dict | None = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        request = urllib.request.Request(
            url, data=data, headers=self._headers, method="POST" if data else "GET"
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            raise BrokerError(f"{exc.code} from {url}: {body}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise BrokerError(f"could not reach {url}: {exc}") from exc

    # -- prices ----------------------------------------------------------

    def last_price(self, symbol: str) -> float | None:
        """Latest trade price, used to turn notional into shares for shorts."""
        url = f"{self.data_url}/v2/stocks/{symbol}/trades/latest"
        try:
            body = self._request(url)
        except BrokerError as exc:
            log.warning("price lookup failed for %s: %s", symbol, exc)
            return None
        price = body.get("trade", {}).get("p")
        return float(price) if price else None

    # -- orders ----------------------------------------------------------

    def submit(self, order: Order) -> OrderResult:
        payload: dict[str, object] = {
            "symbol": order.symbol,
            # Alpaca has no "short" side: selling a symbol you do not hold
            # opens a short, provided the account is margin-enabled.
            "side": "buy" if order.side is Side.BUY else "sell",
            "type": order.order_type,
            "time_in_force": order.time_in_force,
        }
        if order.client_order_id:
            payload["client_order_id"] = order.client_order_id

        if order.qty is not None:
            payload["qty"] = str(order.qty)
        elif order.side is Side.BUY:
            # Notional buys are supported and avoid a price round trip.
            payload["notional"] = str(round(order.notional or 0.0, 2))
        else:
            price = self.last_price(order.symbol)
            if not price:
                return OrderResult(
                    order=order,
                    accepted=False,
                    message=(
                        f"cannot size short in {order.symbol}: no price available "
                        "(notional short sales are not supported)"
                    ),
                )
            qty = int((order.notional or 0.0) // price)
            if qty < 1:
                return OrderResult(
                    order=order,
                    accepted=False,
                    message=(
                        f"notional {order.notional} is below one share of "
                        f"{order.symbol} at {price:.2f}"
                    ),
                )
            order.qty = float(qty)
            payload["qty"] = str(qty)

        try:
            body = self._request(f"{self.base_url}/v2/orders", payload)
        except BrokerError as exc:
            log.error("order rejected for %s: %s", order.symbol, exc)
            return OrderResult(order=order, accepted=False, message=str(exc))

        log.info(
            "ALPACA %s %s accepted as %s",
            order.side.value.upper(),
            order.symbol,
            body.get("id"),
        )
        return OrderResult(
            order=order,
            accepted=True,
            broker_order_id=body.get("id"),
            filled_qty=float(body.get("filled_qty") or 0.0),
            filled_price=float(body["filled_avg_price"]) if body.get("filled_avg_price") else None,
            message=str(body.get("status", "accepted")),
        )
