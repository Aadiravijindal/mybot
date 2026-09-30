"""Alpaca broker adapter (REST). Paper by default.

* Stops are placed as REAL resting stop orders at the broker, so positions stay
  protected if this process dies.
* Order state is polled; fills are emitted as deltas of filled_qty.
* Any REST failure marks the broker disconnected -> Risk Engine halts.

Credentials are read from the environment by this module only; they are never
logged, never included in journals, and never passed to any LLM component.

NOTE: not verified in CI (requires credentials). Validate with
scripts/check_alpaca.py on a paper account first.
"""
from __future__ import annotations

import asyncio
import logging
import os

import httpx

from .broker import Broker, Fill, Order, OrderStatus, OrderType

log = logging.getLogger(__name__)

_STATUS = {"new": OrderStatus.WORKING, "accepted": OrderStatus.WORKING, "pending_new": OrderStatus.WORKING,
           "partially_filled": OrderStatus.PARTIAL, "filled": OrderStatus.FILLED, "canceled": OrderStatus.CANCELED,
           "expired": OrderStatus.CANCELED, "rejected": OrderStatus.REJECTED, "replaced": OrderStatus.CANCELED}


class AlpacaBroker(Broker):
    name = "alpaca"

    def __init__(self, paper: bool = True):
        super().__init__()
        key, sec = os.environ.get("ALPACA_API_KEY_ID"), os.environ.get("ALPACA_API_SECRET_KEY")
        if not key or not sec:
            raise SystemExit("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY not set")
        base = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        self.http = httpx.Client(base_url=base, timeout=3.0,
                                 headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec})
        self.orders: dict[str, Order] = {}
        self.remote: dict[str, str] = {}     # our id -> alpaca id
        self._pos: dict[str, int] = {}
        self.connected = True

    def _req(self, method: str, url: str, **kw):
        try:
            r = self.http.request(method, url, **kw)
            self.connected = True
            return r
        except httpx.HTTPError as exc:
            self.connected = False
            log.error("alpaca %s %s failed: %s", method, url, exc)
            return None

    def submit(self, o: Order) -> Order:
        body = {"symbol": o.symbol, "qty": str(o.qty), "side": "buy" if o.side > 0 else "sell",
                "time_in_force": "day", "client_order_id": o.id}
        if o.type is OrderType.MARKET:
            body["type"] = "market"
        elif o.type is OrderType.STOP:
            body.update(type="stop", stop_price=f"{o.stop:.2f}")
        else:
            body.update(type="limit", limit_price=f"{o.limit:.2f}")
        r = self._req("POST", "/v2/orders", json=body)
        if r is None or r.status_code >= 300:
            o.status = OrderStatus.REJECTED
            o.reject_reason = "network" if r is None else r.text[:200]
        else:
            o.status = OrderStatus.WORKING
            self.remote[o.id] = r.json()["id"]
            self.orders[o.id] = o
        if self.on_order:
            self.on_order(o)
        return o

    def cancel(self, order_id: str) -> None:
        rid = self.remote.get(order_id)
        if rid:
            self._req("DELETE", f"/v2/orders/{rid}")

    def replace_stop(self, order_id: str, new_stop: float) -> None:
        rid = self.remote.get(order_id)
        if not rid:
            return
        r = self._req("PATCH", f"/v2/orders/{rid}", json={"stop_price": f"{new_stop:.2f}"})
        if r is not None and r.status_code < 300:
            self.remote[order_id] = r.json()["id"]   # replace creates a new order id
            self.orders[order_id].stop = new_stop

    def positions(self) -> dict[str, int]:
        return dict(self._pos)

    def refresh_positions(self) -> None:
        r = self._req("GET", "/v2/positions")
        if r is not None and r.status_code < 300:
            self._pos = {p["symbol"]: int(float(p["qty"])) for p in r.json()}

    async def run(self, system) -> None:
        """Poll open orders for fills and refresh positions."""
        while True:
            await asyncio.sleep(1.0)
            for oid, o in list(self.orders.items()):
                if o.done:
                    continue
                r = await asyncio.to_thread(self._req, "GET", f"/v2/orders/{self.remote[oid]}")
                if r is None or r.status_code >= 300:
                    continue
                d = r.json()
                filled = int(float(d.get("filled_qty") or 0))
                if filled > o.filled:
                    avg = float(d.get("filled_avg_price") or 0)
                    # incremental price for the new chunk
                    px = (avg * filled - o.avg_px * o.filled) / (filled - o.filled) if filled else avg
                    q = filled - o.filled
                    o.filled, o.avg_px = filled, avg
                    o.status = _STATUS.get(d["status"], o.status)
                    if self.on_fill:
                        self.on_fill(Fill(oid, o.symbol, o.side, q, px, system.clock.now(), 0.0, o.tag))
                new = _STATUS.get(d["status"], o.status)
                if new != o.status:
                    o.status = new
                    if self.on_order:
                        self.on_order(o)
            await asyncio.to_thread(self.refresh_positions)
