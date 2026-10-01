"""A local fake of the Alpaca trading + market-data APIs, for end-to-end tests.

It speaks Alpaca's wire formats (REST JSON + websocket message arrays) closely
enough to drive mybot's real adapters, and it deliberately reproduces the broker
behaviours that break naive bots:

  * shares reserved by open sell orders cannot be sold again
    ("insufficient qty available for order"), same for buy-to-cover on shorts;
  * PATCH (replace) creates a NEW order id and marks the old one "replaced";
  * cancels are asynchronous ("pending_cancel" first, "canceled" a moment later);
  * stop orders trigger on the market and fill at the touch;
  * account equity, daytrade_count and the pattern-day-trader flag.

The market is the synthetic generator, replayed on the same accelerated clock
the engine uses (MYBOT_TEST_CLOCK), so a whole session runs in minutes.
Chaos switches (POST /_chaos) make the API fail on demand.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from mybot.core.events import Topic
from mybot.data.synthetic import SyntheticMarket


def iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


class FakeAlpaca:
    def __init__(self, symbols, proxy, day, start_ts, speed, equity=50_000.0, history_days=12, seed=21,
                 preexisting=None, pdt=False, wall0=None):
        self.start_ts, self.speed, self.w0 = start_ts, speed, time.time() if wall0 is None else wall0
        self.symbols = list(dict.fromkeys(list(symbols) + [proxy]))
        hist = SyntheticMarket(symbols, proxy, seed=seed + 1)
        hdays = hist.trading_days(datetime(day.year, day.month, day.day) - __import__("datetime").timedelta(days=history_days * 2), 400)
        hdays = [d for d in hdays if d.date() < day.date()][-history_days:]
        self.history = [e[2] for d in hdays for e in hist.day_events(d) if e[1] is Topic.BAR]
        mk = SyntheticMarket(symbols, proxy, seed=seed)
        self.events = mk.day_events(day)
        self.i = 0
        self.cash, self.start_equity = equity, equity
        self.pos: dict[str, int] = dict(preexisting or {})
        self.avg: dict[str, float] = {}
        self.orders: dict[str, dict] = {}
        self.last_q: dict[str, tuple[float, float]] = {}
        self.today_bars: list = []
        self.subs: list[WebSocket] = []
        self.news_subs: list[WebSocket] = []
        self.daytrade_count = 3 if pdt else 0
        self.chaos = {"rest_down": False, "reject_stops": False, "reject_all": False}
        self.log: list[str] = []
        self.ids = itertools.count(1)

    def now(self) -> float:
        return self.start_ts + (time.time() - self.w0) * self.speed

    # --------------------------------------------------------------- market
    def mark(self, s: str) -> float:
        q = self.last_q.get(s)
        return (q[0] + q[1]) / 2 if q else 0.0

    def equity(self) -> float:
        return self.cash + sum(q * self.mark(s) for s, q in self.pos.items())

    def reserved(self, sym: str, side: str) -> int:
        return sum(int(o["qty"]) - int(o["filled_qty"]) for o in self.orders.values()
                   if o["symbol"] == sym and o["side"] == side and o["status"] in ("new", "accepted", "partially_filled", "pending_cancel"))

    def fill(self, o: dict, px: float) -> None:
        q = int(o["qty"]) - int(o["filled_qty"])
        sgn = 1 if o["side"] == "buy" else -1
        self.cash -= sgn * q * px
        self.pos[o["symbol"]] = self.pos.get(o["symbol"], 0) + sgn * q
        if self.pos[o["symbol"]] == 0:
            del self.pos[o["symbol"]]
        o.update(filled_qty=str(int(o["qty"])), filled_avg_price=str(round(px, 4)), status="filled", filled_at=iso(self.now()))

    async def pump(self) -> None:
        """Replay market events up to the current simulated time; trigger resting stops."""
        while True:
            now = self.now()
            out, news = [], []
            while self.i < len(self.events) and self.events[self.i][0] <= now:
                ts, topic, p = self.events[self.i]
                self.i += 1
                if topic is Topic.QUOTE:
                    self.last_q[p.symbol] = (p.bid, p.ask)
                    out.append({"T": "q", "S": p.symbol, "bp": p.bid, "ap": p.ask, "bs": p.bid_size, "as": p.ask_size, "t": iso(p.ts)})
                    self._trigger(p.symbol, p.bid, p.ask)
                elif topic is Topic.TRADE:
                    out.append({"T": "t", "S": p.symbol, "p": p.price, "s": p.size, "t": iso(p.ts)})
                elif topic is Topic.BAR:
                    self.today_bars.append(p)
                    out.append({"T": "b", "S": p.symbol, "o": p.open, "h": p.high, "l": p.low, "c": p.close, "v": p.volume, "t": iso(p.ts)})
                elif topic is Topic.NEWS:
                    news.append({"T": "n", "id": int(p.id[1:]), "headline": p.headline, "summary": "", "symbols": list(p.symbols),
                                 "created_at": iso(p.ts), "source": "benzinga", "url": ""})
            for ws_list, msgs in ((self.subs, out), (self.news_subs, news)):
                if msgs:
                    for ws in list(ws_list):
                        try:
                            for k in range(0, len(msgs), 200):
                                await ws.send_text(json.dumps(msgs[k:k + 200]))
                        except Exception:
                            ws_list.remove(ws)
            await asyncio.sleep(0.01)

    def _trigger(self, sym: str, bid: float, ask: float) -> None:
        for o in self.orders.values():
            if o["symbol"] != sym or o["type"] != "stop" or o["status"] not in ("new", "accepted"):
                continue
            sp = float(o["stop_price"])
            if o["side"] == "sell" and bid <= sp:
                self.fill(o, bid)
                self.log.append(f"stop filled {sym} sell @ {bid}")
            elif o["side"] == "buy" and ask >= sp:
                self.fill(o, ask)
                self.log.append(f"stop filled {sym} buy @ {ask}")

    def settle_cancels(self) -> None:
        for o in self.orders.values():
            if o["status"] == "pending_cancel":
                o["status"] = "canceled"


def create_app(f: FakeAlpaca) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(f.pump())
        yield
        task.cancel()

    app = FastAPI(lifespan=lifespan)

    def auth(req: Request) -> None:
        if f.chaos["rest_down"]:
            raise HTTPException(503, "service unavailable")
        if not req.headers.get("APCA-API-KEY-ID") or not req.headers.get("APCA-API-SECRET-KEY"):
            raise HTTPException(401, "unauthorized")

    @app.post("/_chaos")
    async def chaos(req: Request):
        f.chaos.update(await req.json())
        return f.chaos

    @app.get("/_state")
    async def state():
        return {"positions": f.pos, "equity": f.equity(), "orders": list(f.orders.values()), "log": f.log[-50:], "now": f.now()}

    @app.get("/v2/account")
    async def account(req: Request):
        auth(req)
        e = f.equity()
        return {"status": "ACTIVE", "trading_blocked": False, "account_blocked": False, "equity": str(round(e, 2)),
                "cash": str(round(f.cash, 2)), "buying_power": str(round(e * 2, 2)), "pattern_day_trader": False,
                "daytrade_count": f.daytrade_count, "shorting_enabled": True, "currency": "USD"}

    @app.get("/v2/positions")
    async def positions(req: Request):
        auth(req)
        f.settle_cancels()
        return [{"symbol": s, "qty": str(q), "side": "long" if q > 0 else "short", "avg_entry_price": "0"} for s, q in f.pos.items()]

    @app.get("/v2/orders")
    async def list_orders(req: Request, status: str = "open", limit: int = 500):
        auth(req)
        return [o for o in f.orders.values() if o["status"] in ("new", "accepted", "partially_filled")][:limit]

    @app.post("/v2/orders")
    async def submit(req: Request):
        auth(req)
        b = await req.json()
        if f.chaos["reject_all"] or (f.chaos["reject_stops"] and b["type"] == "stop"):
            return JSONResponse({"code": 40310000, "message": "order rejected (chaos)"}, status_code=403)
        if any(o["client_order_id"] == b["client_order_id"] for o in f.orders.values()):
            return JSONResponse({"code": 40010001, "message": "client_order_id must be unique"}, status_code=422)
        sym, qty, side = b["symbol"], int(b["qty"]), b["side"]
        held = f.pos.get(sym, 0)
        # Alpaca reserves shares for open orders: closing quantity cannot exceed what is unreserved.
        if side == "sell" and held > 0 and qty > held - f.reserved(sym, "sell"):
            return JSONResponse({"code": 40310000, "message": f"insufficient qty available for order (requested: {qty}, available: {held - f.reserved(sym, 'sell')})"}, status_code=403)
        if side == "buy" and held < 0 and qty > -held - f.reserved(sym, "buy"):
            return JSONResponse({"code": 40310000, "message": "insufficient qty available for order"}, status_code=403)
        if b["type"] == "stop":
            bid, ask = f.last_q.get(sym, (0, 0))
            sp = float(b["stop_price"])
            if (side == "sell" and sp >= bid) or (side == "buy" and sp <= ask):
                return JSONResponse({"code": 42210000, "message": "stop price must be beyond the market"}, status_code=422)
        oid = str(uuid.uuid4())
        o = {"id": oid, "client_order_id": b["client_order_id"], "symbol": sym, "qty": str(qty), "side": side, "type": b["type"],
             "time_in_force": b.get("time_in_force", "day"), "stop_price": b.get("stop_price"), "status": "accepted",
             "filled_qty": "0", "filled_avg_price": None, "submitted_at": iso(f.now())}
        f.orders[oid] = o
        if b["type"] == "market":
            bid, ask = f.last_q.get(sym, (0, 0))
            if not bid:
                o["status"] = "rejected"
            else:
                f.fill(o, ask if side == "buy" else bid)
        return o

    @app.get("/v2/orders/{oid}")
    async def get_order(req: Request, oid: str):
        auth(req)
        f.settle_cancels()
        if oid not in f.orders:
            raise HTTPException(404)
        return f.orders[oid]

    @app.delete("/v2/orders/{oid}")
    async def cancel(req: Request, oid: str):
        auth(req)
        o = f.orders.get(oid)
        if o is None:
            raise HTTPException(404)
        if o["status"] in ("filled", "canceled", "replaced"):
            return JSONResponse({"code": 42210000, "message": f"order is already {o['status']}"}, status_code=422)
        o["status"] = "pending_cancel"   # asynchronous, like the real thing
        return JSONResponse(None, status_code=204)

    @app.patch("/v2/orders/{oid}")
    async def replace(req: Request, oid: str):
        auth(req)
        o = f.orders.get(oid)
        b = await req.json()
        if o is None or o["status"] not in ("new", "accepted"):
            return JSONResponse({"code": 42210000, "message": "order not replaceable"}, status_code=422)
        bid, ask = f.last_q.get(o["symbol"], (0, 0))
        sp = float(b["stop_price"])
        if (o["side"] == "sell" and sp >= bid) or (o["side"] == "buy" and sp <= ask):
            return JSONResponse({"code": 42210000, "message": "stop price must be beyond the market"}, status_code=422)
        o["status"] = "replaced"
        n = dict(o, id=str(uuid.uuid4()), stop_price=b["stop_price"], status="accepted", replaces=oid)
        f.orders[n["id"]] = n
        return n

    @app.get("/v2/stocks/bars")
    async def bars(req: Request, symbols: str, start: str, end: str, timeframe: str = "1Min", feed: str = "iex",
                   limit: int = 10000, page_token: str | None = None, adjustment: str = "raw"):
        auth(req)
        s0 = datetime.fromisoformat(start.replace("Z", "+00:00")).timestamp()
        s1 = datetime.fromisoformat(end.replace("Z", "+00:00")).timestamp()
        want = set(symbols.split(","))
        rows = [b for b in f.history + f.today_bars if b.symbol in want and s0 <= b.ts < s1 and b.end_ts <= f.now()]
        off = int(page_token or 0)
        page = rows[off:off + limit]
        out: dict[str, list] = {}
        for b in page:
            out.setdefault(b.symbol, []).append({"t": iso(b.ts), "o": b.open, "h": b.high, "l": b.low, "c": b.close, "v": b.volume})
        return {"bars": out, "next_page_token": str(off + limit) if off + limit < len(rows) else None}

    async def stream(ws: WebSocket, subs: list):
        await ws.accept()
        await ws.send_text(json.dumps([{"T": "success", "msg": "connected"}]))
        auth_msg = json.loads(await ws.receive_text())
        if auth_msg.get("action") != "auth" or not auth_msg.get("key"):
            await ws.send_text(json.dumps([{"T": "error", "code": 402, "msg": "auth failed"}]))
            await ws.close()
            return
        await ws.send_text(json.dumps([{"T": "success", "msg": "authenticated"}]))
        await ws.receive_text()   # subscribe
        subs.append(ws)
        try:
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            if ws in subs:
                subs.remove(ws)

    @app.websocket("/v2/{feed}")
    async def market_ws(ws: WebSocket, feed: str):
        await stream(ws, f.subs)

    @app.websocket("/v1beta1/news")
    async def news_ws(ws: WebSocket):
        await stream(ws, f.news_subs)

    return app
