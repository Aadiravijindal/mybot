"""Catalyst + event-risk calendar, and anomaly detection."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from ..core.types import ScheduledEvent
from .news import NewsAssessment


class EventCalendar:
    """Knows what scheduled risk is coming (macro releases, earnings, FOMC)."""

    def __init__(self):
        self.events: list[ScheduledEvent] = []

    def add(self, e: ScheduledEvent) -> None:
        self.events.append(e)
        self.events.sort(key=lambda x: x.scheduled_ts)

    def upcoming(self, sym: str, now: float, horizon_s: float, min_importance: int = 1) -> list[ScheduledEvent]:
        return [e for e in self.events
                if now <= e.scheduled_ts <= now + horizon_s and e.importance >= min_importance
                and e.avail_ts <= now and ("*" in e.symbols or sym in e.symbols)]

    def next_event(self, sym: str, now: float) -> ScheduledEvent | None:
        for e in self.events:
            if e.scheduled_ts >= now and e.avail_ts <= now and ("*" in e.symbols or sym in e.symbols):
                return e
        return None


class CatalystBook:
    """Recent unscheduled catalysts per symbol (from news), decayed over time."""

    def __init__(self, half_life_min: float = 45.0):
        self.hl = half_life_min * 60
        self.items: dict[str, deque[NewsAssessment]] = defaultdict(lambda: deque(maxlen=50))

    def add(self, a: NewsAssessment) -> None:
        for s in a.symbols:
            self.items[s].append(a)

    def score(self, sym: str, now: float) -> tuple[float, list[NewsAssessment]]:
        """Signed, decayed catalyst score in about [-1, 1] plus the contributing items."""
        tot, used = 0.0, []
        for a in self.items.get(sym, ()):
            age = now - a.avail_ts
            if age < 0 or age > 6 * self.hl:
                continue
            w = 0.5 ** (age / self.hl)
            tot += a.direction * a.materiality * w
            if a.materiality * w > 0.05:
                used.append(a)
        return max(-1.0, min(1.0, tot)), used


@dataclass
class Anomaly:
    symbol: str
    ts: float
    kind: str
    z: float
    why: str


class AnomalyDetector:
    """Flags statistically unusual observations and asks 'why?' (-> research)."""

    def __init__(self, z_thresh: float = 4.0):
        self.z = z_thresh

    def check(self, sym: str, ts: float, f: dict[str, float], news_recent: bool) -> list[Anomaly]:
        out = []
        vz, rz = f.get("vol_z", 0.0), f.get("ret_z", 0.0)
        if vz > self.z:
            out.append(Anomaly(sym, ts, "VOLUME_SPIKE", vz,
                               "volume spike WITH recent news" if news_recent else "volume spike with NO news found -> unexplained; research"))
        if abs(rz) > self.z:
            out.append(Anomaly(sym, ts, "PRICE_SHOCK", rz,
                               "price shock explained by news" if news_recent else "price shock with no catalyst -> possible leak/flow/error; research"))
        if f.get("spread_z", 0.0) > self.z:
            out.append(Anomaly(sym, ts, "SPREAD_BLOWOUT", f["spread_z"], "liquidity withdrawal"))
        return out
