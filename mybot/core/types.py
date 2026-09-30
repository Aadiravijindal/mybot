"""Core market/data types.

Every record carries two timestamps:
  ts        -- when the thing happened (exchange / publication time)
  avail_ts  -- when *this system* could first have known about it

All decision logic must key off ``avail_ts``. This is what makes point-in-time
backtesting possible: a news story published at 14:03:00 that our feed delivered
at 14:03:02 has avail_ts=14:03:02 and cannot influence anything before that.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Side(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

    @property
    def sign(self) -> int:
        return 1 if self is Side.LONG else -1


@dataclass(frozen=True, slots=True)
class Bar:
    symbol: str
    ts: float          # bar start (epoch seconds, UTC)
    end_ts: float      # bar end; a bar is only usable once closed
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str = "unknown"
    avail_ts: float = 0.0
    timeframe: int = 60  # seconds

    @property
    def typical(self) -> float:
        return (self.high + self.low + self.close) / 3.0


@dataclass(frozen=True, slots=True)
class Quote:
    symbol: str
    ts: float
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    source: str = "unknown"
    avail_ts: float = 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        m = self.mid
        return (self.spread / m) * 1e4 if m > 0 else float("inf")


@dataclass(frozen=True, slots=True)
class Trade:
    symbol: str
    ts: float
    price: float
    size: float
    source: str = "unknown"
    avail_ts: float = 0.0


@dataclass(frozen=True, slots=True)
class NewsItem:
    id: str
    ts: float                 # publication time claimed by the source
    headline: str
    symbols: tuple[str, ...]
    source: str = "unknown"
    body: str = ""
    url: str = ""
    avail_ts: float = 0.0     # when we received it


@dataclass(frozen=True, slots=True)
class ScheduledEvent:
    """Macro / corporate calendar item (FOMC, CPI, earnings...)."""
    name: str
    scheduled_ts: float
    importance: int           # 1 (low) .. 3 (high)
    symbols: tuple[str, ...] = ("*",)  # "*" = market-wide
    avail_ts: float = 0.0


@dataclass(slots=True)
class DataQuality:
    """Quality annotation attached to a symbol's current data state."""
    ok: bool = True
    score: float = 1.0          # 0..1
    freshness_s: float = 0.0
    issues: list[str] = field(default_factory=list)
