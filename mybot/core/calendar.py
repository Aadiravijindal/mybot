"""Exchange session helpers (US equities regular session by default).

Holidays/half-days are NOT modelled here; production must load an exchange
calendar (e.g. from the data vendor or `exchange_calendars`). Documented gap.
"""
from __future__ import annotations

from datetime import datetime, time as dtime
from functools import lru_cache
from zoneinfo import ZoneInfo


class Session:
    def __init__(self, tz: str = "America/New_York", open_: str = "09:30", close: str = "16:00"):
        self.tz = ZoneInfo(tz)
        oh, om = map(int, open_.split(":"))
        ch, cm = map(int, close.split(":"))
        self.open_t, self.close_t = dtime(oh, om), dtime(ch, cm)
        self.length_min = (ch * 60 + cm) - (oh * 60 + om)

    @lru_cache(maxsize=4096)
    def _local(self, ts_min: int) -> datetime:
        return datetime.fromtimestamp(ts_min * 60, self.tz)

    def local(self, ts: float) -> datetime:
        return datetime.fromtimestamp(ts, self.tz)

    def day_key(self, ts: float) -> str:
        return self.local(ts).strftime("%Y-%m-%d")

    def minutes_since_open(self, ts: float) -> float:
        d = self.local(ts)
        return (d.hour * 60 + d.minute + d.second / 60) - (self.open_t.hour * 60 + self.open_t.minute)

    def minutes_to_close(self, ts: float) -> float:
        return self.length_min - self.minutes_since_open(ts)

    def is_open(self, ts: float) -> bool:
        d = self.local(ts)
        if d.weekday() >= 5:
            return False
        m = self.minutes_since_open(ts)
        return 0 <= m < self.length_min

    def open_ts(self, day: datetime) -> float:
        return datetime.combine(day.date(), self.open_t, self.tz).timestamp()
