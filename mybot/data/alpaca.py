"""Alpaca market-data adapter (websocket stream + REST warm-up).

Feeds: "iex" (free; ~2-3% of US volume -- quotes/trades are NOT the NBBO and
volume is a fraction of consolidated volume) or "sip" (paid, consolidated).
Relative-volume and microstructure features computed on IEX are therefore
biased; see docs/ARCHITECTURE.md data-source matrix. Set ALPACA_DATA_FEED=sip
if you subscribe.

NOTE: exercised against Alpaca's documented message formats but not verified in
CI (requires credentials). Run scripts/check_alpaca.py against a paper account
before relying on it.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone

import httpx

from ..core.events import Topic
from ..core.types import Bar, NewsItem, Quote, Trade

log = logging.getLogger(__name__)


_RFC = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?(Z|[+-]\d\d:\d\d)$")


def _ts(s: str) -> float:
    """RFC3339 (with up to nanosecond precision) -> epoch seconds."""
    m = _RFC.match(s)
    if not m:
        return datetime.fromisoformat(s).timestamp()
    base, frac, tz = m.groups()
    tz = "+00:00" if tz == "Z" else tz
    return datetime.fromisoformat(base + tz).timestamp() + (float("0." + frac) if frac else 0.0)


class AlpacaData:
    def __init__(self, symbols: list[str]):
        self.key = os.environ.get("ALPACA_API_KEY_ID", "")
        self.secret = os.environ.get("ALPACA_API_SECRET_KEY", "")
        if not self.key or not self.secret:
            raise SystemExit("ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY not set")
        self.feed = os.environ.get("ALPACA_DATA_FEED", "iex")
        self.symbols = symbols
        self.clock_offset_s = 0.0
        self.connected = False

    def _headers(self):
        return {"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret}

    async def warmup(self, system) -> None:
        """Fetch today's 1m bars so indicators are warm before live bars arrive."""
        start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get("https://data.alpaca.markets/v2/stocks/bars", headers=self._headers(),
                            params={"symbols": ",".join(self.symbols), "timeframe": "1Min", "start": start,
                                    "feed": self.feed, "limit": 10000})
            r.raise_for_status()
            rows = []
            for sym, bars in (r.json().get("bars") or {}).items():
                for b in bars:
                    t0 = _ts(b["t"])
                    rows.append(Bar(sym, t0, t0 + 60, b["o"], b["h"], b["l"], b["c"], b["v"], f"alpaca_{self.feed}", t0 + 60))
            now = time.time()
            for b in sorted(rows, key=lambda x: x.end_ts):
                if b.end_ts <= now:
                    system.publish(Topic.BAR, b, b.symbol, ts=b.avail_ts)

    async def run(self, system) -> None:
        import websockets  # optional dependency: pip install mybot[live]
        try:
            await self.warmup(system)
        except Exception as exc:
            log.warning("warm-up failed: %s", exc)
        await asyncio.gather(self._stream(system, f"wss://stream.data.alpaca.markets/v2/{self.feed}", market=True),
                             self._stream(system, "wss://stream.data.alpaca.markets/v1beta1/news", market=False))

    async def _stream(self, system, url: str, market: bool) -> None:
        import websockets
        backoff = 1
        while True:
            try:
                async with websockets.connect(url, ping_interval=15) as ws:
                    await ws.send(json.dumps({"action": "auth", "key": self.key, "secret": self.secret}))
                    sub = {"action": "subscribe", "bars": self.symbols, "quotes": self.symbols, "trades": self.symbols} \
                        if market else {"action": "subscribe", "news": ["*"]}
                    await ws.send(json.dumps(sub))
                    self.connected, backoff = True, 1
                    async for raw in ws:
                        now = time.time()
                        for m in json.loads(raw):
                            self._dispatch(system, m, now)
            except Exception as exc:
                self.connected = False
                log.warning("alpaca stream %s dropped: %s (reconnect in %ss)", url, exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    def _dispatch(self, system, m: dict, now: float) -> None:
        t = m.get("T")
        src = f"alpaca_{self.feed}"
        if t == "b":
            t0 = _ts(m["t"])
            system.publish(Topic.BAR, Bar(m["S"], t0, t0 + 60, m["o"], m["h"], m["l"], m["c"], m["v"], src, now), m["S"], ts=now)
        elif t == "q":
            ts = _ts(m["t"])
            self.clock_offset_s = 0.9 * self.clock_offset_s + 0.1 * (now - ts)  # latency+skew estimate
            system.publish(Topic.QUOTE, Quote(m["S"], ts, m["bp"], m["ap"], m["bs"] * 100, m["as"] * 100, src, now), m["S"], ts=now)
        elif t == "t":
            system.publish(Topic.TRADE, Trade(m["S"], _ts(m["t"]), m["p"], m["s"], src, now), m["S"], ts=now)
        elif t == "n":
            syms = tuple(s for s in m.get("symbols", []) if s in self.symbols)
            if syms:
                system.publish(Topic.NEWS, NewsItem(str(m["id"]), _ts(m["created_at"]), m.get("headline", ""), syms,
                                                    "alpaca_news", m.get("summary", ""), m.get("url", ""), now), syms[0], ts=now)
        elif t == "error":
            log.error("alpaca error: %s", m)
