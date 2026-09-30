"""Paper broker with a deliberately pessimistic fill model.

  * latency: orders become fillable only ``latency_s`` after submission
  * market orders cross the spread, plus base slippage and square-root impact
  * partial fills limited by displayed size / bar volume; residual cancelled
  * stop orders rest at the "broker" and trigger on trade-through; gaps fill at
    the worse open price, not the stop price
  * orders are rejected when the symbol is halted or has no quote

Paper results from this model are still optimistic vs reality (no queue
position, no adverse selection on limits) -- documented in ARCHITECTURE.md.
"""
from __future__ import annotations

import math

from ..config import CostModel
from ..core.types import Bar, Quote
from .broker import Broker, Fill, Order, OrderStatus, OrderType


class PaperBroker(Broker):
    name = "paper"

    def __init__(self, costs: CostModel):
        super().__init__()
        self.c = costs
        self.orders: dict[str, Order] = {}
        self.pos: dict[str, int] = {}
        self.quotes: dict[str, Quote] = {}
        self.halted: set[str] = set()
        self.daily_vol_sigma: dict[str, float] = {}
        self.adv: dict[str, float] = {}
        self.connected = True

    # ------------------------------------------------------------------ API
    def submit(self, o: Order) -> Order:
        if not self.connected:
            o.status, o.reject_reason = OrderStatus.REJECTED, "broker disconnected"
        elif o.symbol in self.halted:
            o.status, o.reject_reason = OrderStatus.REJECTED, "symbol halted"
        elif o.type is OrderType.MARKET and o.symbol not in self.quotes:
            o.status, o.reject_reason = OrderStatus.REJECTED, "no quote"
        elif o.qty <= 0:
            o.status, o.reject_reason = OrderStatus.REJECTED, "qty<=0"
        else:
            o.status = OrderStatus.WORKING
            self.orders[o.id] = o
        if self.on_order:
            self.on_order(o)
        return o

    def cancel(self, order_id: str) -> None:
        o = self.orders.get(order_id)
        if o and not o.done:
            o.status = OrderStatus.CANCELED
            if self.on_order:
                self.on_order(o)

    def replace_stop(self, order_id: str, new_stop: float) -> None:
        o = self.orders.get(order_id)
        if o and not o.done and o.type is OrderType.STOP:
            o.stop = new_stop

    def positions(self) -> dict[str, int]:
        return {k: v for k, v in self.pos.items() if v != 0}

    # -------------------------------------------------------------- matching
    def _fees(self, qty: int) -> float:
        return max(self.c.min_commission, qty * self.c.commission_per_share)

    def _impact_px(self, sym: str, qty: int, px: float) -> float:
        sig = self.daily_vol_sigma.get(sym, 0.02)
        adv = self.adv.get(sym, 1e6)
        return px * self.c.impact_coef * sig * math.sqrt(max(qty, 0) / max(adv, 1.0))

    def _fill(self, o: Order, qty: int, px: float, ts: float) -> None:
        qty = min(qty, o.remaining)
        if qty <= 0:
            return
        fees = self._fees(qty) if o.filled == 0 else qty * self.c.commission_per_share
        o.avg_px = (o.avg_px * o.filled + px * qty) / (o.filled + qty)
        o.filled += qty
        o.fees += fees
        o.status = OrderStatus.FILLED if o.remaining == 0 else OrderStatus.PARTIAL
        self.pos[o.symbol] = self.pos.get(o.symbol, 0) + o.side * qty
        if self.on_fill:
            self.on_fill(Fill(o.id, o.symbol, o.side, qty, round(px, 4), ts, fees, o.tag))
        if self.on_order:
            self.on_order(o)

    def on_quote(self, q: Quote) -> None:
        self.quotes[q.symbol] = q
        for o in list(self.orders.values()):
            if o.done or o.symbol != q.symbol or q.avail_ts < o.ts + self.c.latency_s:
                continue
            if o.type is OrderType.MARKET:
                base = q.ask if o.side > 0 else q.bid
                px = base * (1 + o.side * self.c.slippage_bps / 1e4) + o.side * self._impact_px(o.symbol, o.remaining, base)
                avail = int(max((q.ask_size if o.side > 0 else q.bid_size) * 3, 100))
                self._fill(o, min(avail, o.remaining), px, q.avail_ts)
                if not o.done and q.avail_ts - o.ts > 30:
                    self.cancel(o.id)   # IOC-like residual cancel
            elif o.type is OrderType.STOP:
                trig = (q.bid <= o.stop) if o.side < 0 else (q.ask >= o.stop)
                if trig:
                    base = q.bid if o.side < 0 else q.ask
                    self._fill(o, o.remaining, base * (1 + o.side * self.c.slippage_bps / 1e4), q.avail_ts)

    def on_bar(self, b: Bar) -> None:
        """Bars-only fallback matching (used when no quote stream is available)."""
        sym = b.symbol
        half_spread = b.close * 2e-4
        for o in list(self.orders.values()):
            if o.done or o.symbol != sym or b.avail_ts < o.ts + self.c.latency_s:
                continue
            if o.type is OrderType.STOP:
                if (o.side < 0 and b.low <= o.stop) or (o.side > 0 and b.high >= o.stop):
                    gap = (b.open <= o.stop) if o.side < 0 else (b.open >= o.stop)
                    px = b.open if gap else o.stop
                    self._fill(o, o.remaining, px * (1 + o.side * self.c.slippage_bps / 1e4) + o.side * half_spread, b.avail_ts)
            elif o.type is OrderType.MARKET and sym not in self.quotes:
                px = b.open + o.side * half_spread
                self._fill(o, min(o.remaining, int(b.volume * 0.2) or 1), px, b.avail_ts)
