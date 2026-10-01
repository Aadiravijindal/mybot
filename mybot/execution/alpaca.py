"""Alpaca broker adapter (REST). Paper by default; live only through the guarded runtime.

* Stops are placed as REAL resting stop orders at the broker, so positions stay
  protected if this process dies.
* Before an exit, the resting stop is cancelled and the cancel is CONFIRMED
  (Alpaca reserves shares for open orders; a sell while the stop rests is
  rejected, and a stop that filled in the meantime must not be followed by a
  second sell that would flip the position).
* Order state is polled; fills are emitted as deltas of filled_qty.
* Any REST failure marks the broker disconnected -> Risk Engine halts.
* Pre-flight on start: account status, trading blocks, pattern-day-trader rule,
  positions left over from a previous run, stale orders of ours.

Credentials are read from the environment by this module only; they are never
logged, never included in journals, and never passed to any LLM component.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

import httpx

from .broker import ORDER_PREFIX, Broker, Fill, Order, OrderStatus, OrderType

log = logging.getLogger(__name__)

PAPER_URL = "https://paper-api.alpaca.markets"
LIVE_URL = "https://api.alpaca.markets"

_STATUS = {"new": OrderStatus.WORKING, "accepted": OrderStatus.WORKING, "pending_new": OrderStatus.WORKING,
           "accepted_for_bidding": OrderStatus.WORKING, "pending_cancel": OrderStatus.WORKING,
           "pending_replace": OrderStatus.WORKING, "held": OrderStatus.WORKING, "calculated": OrderStatus.WORKING,
           "partially_filled": OrderStatus.PARTIAL, "filled": OrderStatus.FILLED, "done_for_day": OrderStatus.CANCELED,
           "canceled": OrderStatus.CANCELED, "expired": OrderStatus.CANCELED, "rejected": OrderStatus.REJECTED,
           "suspended": OrderStatus.REJECTED, "stopped": OrderStatus.WORKING, "replaced": OrderStatus.CANCELED}
PDT_EQUITY = 25_000.0


def credentials() -> tuple[str, str]:
    key, sec = os.environ.get("ALPACA_API_KEY_ID"), os.environ.get("ALPACA_API_SECRET_KEY")
    if not key or not sec:
        raise SystemExit("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY not set (see .env.example)")
    return key, sec


class AlpacaBroker(Broker):
    name = "alpaca"

    def __init__(self, paper: bool = True, http: httpx.Client | None = None):
        super().__init__()
        key, sec = credentials()
        base = os.environ.get("ALPACA_TRADING_URL") or (PAPER_URL if paper else LIVE_URL)
        self.paper = paper
        self.name = "alpaca-paper" if paper else "alpaca-LIVE"
        self.http = http or httpx.Client(base_url=base, timeout=3.0)
        self.http.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec})
        self.orders: dict[str, Order] = {}
        self.remote: dict[str, str] = {}     # our id -> alpaca id
        self._pos: dict[str, int] = {}
        self.account: dict = {}
        self.connected = True
        self.failures = 0
        self.last_ok = time.time()
        self.now = time.time        # replaced by the runtime with the engine clock

    # ------------------------------------------------------------- plumbing
    def _req(self, method: str, url: str, **kw) -> httpx.Response | None:
        """One REST call. Network errors, 5xx and 429 count as failures; two in a row
        mark the broker DISCONNECTED (the Risk Engine then halts). Any success clears it.
        5xx/429 responses are returned as None so callers treat them like an outage."""
        try:
            r = self.http.request(method, url, **kw)
        except httpx.HTTPError as exc:
            self._fail(f"{method} {url}: {type(exc).__name__}")
            return None
        if r.status_code >= 500 or r.status_code == 429:
            self._fail(f"{method} {url}: HTTP {r.status_code}")
            return None
        if r.status_code in (401, 403) and url.startswith("/v2/account"):
            self._fail(f"{method} {url}: auth failed ({r.status_code})", hard=True)
            return r
        self.failures = 0
        self.connected = True
        self.last_ok = time.time()
        return r

    def _fail(self, what: str, hard: bool = False) -> None:
        self.failures += 1
        log.error("alpaca request failed: %s", what)
        if hard or self.failures >= 2:
            self.connected = False

    # ------------------------------------------------------------- pre-flight
    def preflight(self) -> dict:
        """Checks that must pass before the engine may trade. Returns a report; raises SystemExit on hard failures."""
        r = self._req("GET", "/v2/account")
        if r is None or r.status_code >= 300:
            raise SystemExit(f"Alpaca account check failed: {'network error' if r is None else r.status_code}")
        a = r.json()
        self.account = a
        problems = []
        if a.get("status") != "ACTIVE":
            problems.append(f"account status is {a.get('status')}")
        if a.get("trading_blocked") or a.get("account_blocked"):
            problems.append("trading is blocked on this account")
        if problems:
            raise SystemExit("Alpaca pre-flight failed: " + "; ".join(problems))
        self.refresh_positions()
        # Cancel stale orders created by this system in a previous run (never touch other orders).
        stale = []
        ro = self._req("GET", "/v2/orders", params={"status": "open", "limit": 500})
        if ro is not None and ro.status_code < 300:
            for o in ro.json():
                if str(o.get("client_order_id", "")).startswith(ORDER_PREFIX):
                    self._req("DELETE", f"/v2/orders/{o['id']}")
                    stale.append(o.get("symbol"))
        equity = float(a.get("equity") or 0)
        return {
            "equity": equity, "cash": float(a.get("cash") or 0), "buying_power": float(a.get("buying_power") or 0),
            "pattern_day_trader": bool(a.get("pattern_day_trader")), "daytrade_count": int(a.get("daytrade_count") or 0),
            "shorting_enabled": bool(a.get("shorting_enabled")), "positions": dict(self._pos),
            "stale_orders_cancelled": stale, "paper": self.paper,
            "pdt_limited": equity < PDT_EQUITY and not a.get("pattern_day_trader"),
        }

    def day_trades_remaining(self) -> int | None:
        """FINRA pattern-day-trader rule: margin accounts under $25k get 3 day trades per 5 business days.
        Returns None when unlimited. Every round trip of this intraday system is a day trade."""
        a = self.account
        if not a or float(a.get("equity") or 0) >= PDT_EQUITY or a.get("pattern_day_trader"):
            return None
        return max(0, 3 - int(a.get("daytrade_count") or 0))

    def refresh_account(self) -> None:
        r = self._req("GET", "/v2/account")
        if r is not None and r.status_code < 300:
            self.account = r.json()

    # ------------------------------------------------------------- orders
    def submit(self, o: Order) -> Order:
        body = {"symbol": o.symbol, "qty": str(o.qty), "side": "buy" if o.side > 0 else "sell",
                "time_in_force": "day", "client_order_id": o.id}
        if o.type is OrderType.MARKET:
            body["type"] = "market"
        elif o.type is OrderType.STOP:
            body.update(type="stop", stop_price=_px(o.stop))
        else:
            body.update(type="limit", limit_price=_px(o.limit))
        r = self._req("POST", "/v2/orders", json=body)
        if r is None or r.status_code >= 300:
            o.status = OrderStatus.REJECTED
            o.reject_reason = "network error" if r is None else _err(r)
        else:
            o.status = OrderStatus.WORKING
            self.remote[o.id] = r.json()["id"]
            self.orders[o.id] = o
        if self.on_order:
            self.on_order(o)
        return o

    def cancel(self, order_id: str, wait_s: float = 2.0) -> bool:
        """Cancel and confirm. True = order is now inactive and not fully filled (safe to send a
        replacement for the remaining book quantity). False = it filled, or the cancel is unconfirmed."""
        rid = self.remote.get(order_id)
        o = self.orders.get(order_id)
        if rid is None or o is None:
            return True
        if o.status is OrderStatus.FILLED:
            return False
        self._req("DELETE", f"/v2/orders/{rid}")
        deadline = time.time() + wait_s
        while True:
            r = self._req("GET", f"/v2/orders/{rid}")
            if r is not None and r.status_code < 300:
                d = r.json()
                self._apply(order_id, o, d, self.now())
                if d["status"] == "filled" or o.status is OrderStatus.FILLED:
                    return False
                if d["status"] in ("canceled", "expired", "rejected", "replaced", "done_for_day"):
                    return True   # any partial fill was already applied above, so the book is current
            if time.time() > deadline:
                # Unconfirmed: treat as NOT safe. The caller will retry; the stop may still be live.
                log.warning("cancel of %s not confirmed within %.1fs", order_id, wait_s)
                return False
            time.sleep(0.1)

    def replace_stop(self, order_id: str, new_stop: float) -> None:
        rid = self.remote.get(order_id)
        if not rid:
            return
        r = self._req("PATCH", f"/v2/orders/{rid}", json={"stop_price": _px(new_stop)})
        if r is not None and r.status_code < 300:
            self.remote[order_id] = r.json()["id"]   # a replace creates a new broker order id
            self.orders[order_id].stop = new_stop
        elif r is not None:
            log.warning("stop replace for %s refused: %s (keeping the existing, looser stop)", order_id, _err(r))

    def positions(self) -> dict[str, int]:
        return dict(self._pos)

    def refresh_positions(self) -> None:
        r = self._req("GET", "/v2/positions")
        if r is not None and r.status_code < 300:
            self._pos = {p["symbol"]: int(float(p["qty"])) for p in r.json() if int(float(p["qty"])) != 0}

    def _apply(self, oid: str, o: Order, d: dict, now: float) -> None:
        """Apply a broker order snapshot: emit fill deltas and status changes."""
        filled = int(float(d.get("filled_qty") or 0))
        if filled > o.filled:
            avg = float(d.get("filled_avg_price") or 0)
            q = filled - o.filled
            px = (avg * filled - o.avg_px * o.filled) / q
            o.filled, o.avg_px = filled, avg
            o.status = _STATUS.get(d["status"], o.status)
            if self.on_fill:
                self.on_fill(Fill(oid, o.symbol, o.side, q, round(px, 4), now, 0.0, o.tag))
        new = _STATUS.get(d["status"], o.status)
        if d["status"] == "replaced":
            return   # superseded by the replacement order we already track
        if new != o.status:
            o.status = new
            if new is OrderStatus.REJECTED:
                o.reject_reason = d.get("reject_reason") or d["status"]
            if self.on_order:
                self.on_order(o)

    def poll_once(self, now: float) -> None:
        for oid, o in list(self.orders.items()):
            if o.done:
                continue
            r = self._req("GET", f"/v2/orders/{self.remote[oid]}")
            if r is not None and r.status_code < 300:
                self._apply(oid, o, r.json(), now)
        self.refresh_positions()

    async def run(self, system) -> None:
        """Poll open orders for fills, refresh positions and the account (day-trade count)."""
        n = 0
        while True:
            await asyncio.sleep(1.0)
            await asyncio.to_thread(self.poll_once, system.clock.now())
            n += 1
            if n % 30 == 0:
                await asyncio.to_thread(self.refresh_account)
                system.risk.pdt_remaining = self.day_trades_remaining()


def _px(x: float) -> str:
    return f"{x:.2f}" if x >= 1 else f"{x:.4f}"   # Alpaca price increments


def _err(r: httpx.Response) -> str:
    try:
        return str(r.json().get("message", r.text))[:200]
    except Exception:
        return r.text[:200]
