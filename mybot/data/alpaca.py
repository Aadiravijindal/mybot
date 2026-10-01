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
from datetime import datetime, timedelta, timezone

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


DATA_URL = "https://data.alpaca.markets"
STREAM_URL = "wss://stream.data.alpaca.markets"


class AlpacaData:
    def __init__(self, symbols: list[str]):
        from ..execution.alpaca import credentials
        self.key, self.secret = credentials()
        self.feed = os.environ.get("ALPACA_DATA_FEED", "iex")
        self.data_url = os.environ.get("ALPACA_DATA_URL", DATA_URL)
        self.stream_url = os.environ.get("ALPACA_STREAM_URL", STREAM_URL)
        self.symbols = symbols
        self.clock_offset_s = 0.0
        self.connected = False
        self.live_started = asyncio.Event()

    def _headers(self):
        return {"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret}

    async def bars(self, start: datetime, end: datetime, feed: str | None = None,
                   timeframe: str = "1Min") -> list[Bar]:
        """Historical bars [start, end), all symbols, following pagination."""
        out: list[Bar] = []
        token = None
        async with httpx.AsyncClient(base_url=self.data_url, timeout=30, headers=self._headers()) as c:
            while True:
                params = {"symbols": ",".join(self.symbols), "timeframe": timeframe, "start": start.isoformat(),
                          "end": end.isoformat(), "feed": feed or self.feed, "limit": 10000, "adjustment": "raw"}
                if token:
                    params["page_token"] = token
                r = await c.get("/v2/stocks/bars", params=params)
                r.raise_for_status()
                js = r.json()
                for sym, rows in (js.get("bars") or {}).items():
                    for b in rows:
                        t0 = _ts(b["t"])
                        out.append(Bar(sym, t0, t0 + 60, b["o"], b["h"], b["l"], b["c"], b["v"], f"alpaca_{feed or self.feed}", t0 + 60))
                token = js.get("next_page_token")
                if not token:
                    break
        out.sort(key=lambda b: (b.end_ts, b.symbol))
        return out

    async def history_events(self, days: int, now: float | None = None) -> list:
        """Recent history (before today) as replayable events, for building the real-data
        knowledge base. Free accounts may query SIP history older than 15 minutes."""
        now = time.time() if now is None else now
        end = datetime.fromtimestamp(now, timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=int(days * 7 / 5) + 3)
        feed = os.environ.get("ALPACA_HISTORY_FEED", "sip")
        bars = await self.bars(start, end, feed=feed)
        return [(b.avail_ts, Topic.BAR, b) for b in bars]

    async def warmup(self, system) -> None:
        """Replay today's bars so indicators are warm. Entries are disabled while this runs."""
        now = system.clock.now()
        start = datetime.fromtimestamp(now, timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        bars = await self.bars(start, datetime.fromtimestamp(now, timezone.utc))
        prev = system.allow_entries
        system.allow_entries = False
        try:
            for b in bars:
                if b.end_ts <= now:
                    system.publish(Topic.BAR, b, b.symbol, ts=b.avail_ts)
        finally:
            system.allow_entries = prev

    async def run(self, system) -> None:
        try:
            await self.warmup(system)
        except Exception as exc:
            log.warning("warm-up failed: %s", exc)
        await asyncio.gather(self._stream(system, f"{self.stream_url}/v2/{self.feed}", market=True),
                             self._stream(system, f"{self.stream_url}/v1beta1/news", market=False))

    async def _stream(self, system, url: str, market: bool) -> None:
        import websockets  # optional dependency: pip install "mybot[live]"
        backoff = 1
        while True:
            try:
                async with websockets.connect(url, ping_interval=15) as ws:
                    await ws.send(json.dumps({"action": "auth", "key": self.key, "secret": self.secret}))
                    sub = {"action": "subscribe", "bars": self.symbols, "quotes": self.symbols, "trades": self.symbols} \
                        if market else {"action": "subscribe", "news": ["*"]}
                    await ws.send(json.dumps(sub))
                    if market:
                        self.connected, backoff = True, 1
                        self.live_started.set()
                    async for raw in ws:
                        now = system.clock.now()
                        for m in json.loads(raw):
                            try:
                                self._dispatch(system, m, now)
                            except (KeyError, TypeError, ValueError) as exc:
                                log.warning("malformed alpaca message skipped: %s (%s)", exc, str(m)[:200])
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if market:
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
            # Smoothed (receipt - exchange) time: latency plus clock skew. Persistently large = clock problem.
            self.clock_offset_s = 0.95 * self.clock_offset_s + 0.05 * (now - ts)
            system.publish(Topic.QUOTE, Quote(m["S"], ts, m["bp"], m["ap"], m["bs"], m["as"], src, now), m["S"], ts=now)
        elif t == "t":
            system.publish(Topic.TRADE, Trade(m["S"], _ts(m["t"]), m["p"], m["s"], src, now), m["S"], ts=now)
        elif t == "n":
            syms = tuple(s for s in m.get("symbols", []) if s in self.symbols)
            if syms:
                system.publish(Topic.NEWS, NewsItem(str(m["id"]), _ts(m["created_at"]), m.get("headline", ""), syms,
                                                    "alpaca_news", m.get("summary", ""), m.get("url", ""), now), syms[0], ts=now)
        elif t == "error":
            log.error("alpaca stream error: %s", m)
