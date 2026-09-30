"""Clocks. The engine never calls time.time() directly: it asks a Clock.

Live trading uses WallClock. Backtests use SimClock, which the replay loop
advances to each event's avail_ts -- so code physically cannot read the future.
"""
from __future__ import annotations

import time


class Clock:
    def now(self) -> float:  # pragma: no cover - interface
        raise NotImplementedError


class WallClock(Clock):
    def now(self) -> float:
        return time.time()


class SimClock(Clock):
    def __init__(self, start: float = 0.0):
        self._t = start

    def now(self) -> float:
        return self._t

    def advance_to(self, t: float) -> None:
        if t < self._t:
            raise ValueError(f"SimClock cannot go backwards: {t} < {self._t}")
        self._t = t


class LookAheadError(RuntimeError):
    """Raised when a record is used before it was available."""
