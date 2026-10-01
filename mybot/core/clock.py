"""Clocks. The engine never calls time.time() directly: it asks a Clock.

Live trading uses WallClock. Backtests use SimClock, which the replay loop
advances to each event's avail_ts -- so code physically cannot read the future.
"""
from __future__ import annotations

import os
import time


class Clock:
    def now(self) -> float:  # pragma: no cover - interface
        raise NotImplementedError


class WallClock(Clock):
    def now(self) -> float:
        return time.time()


class AcceleratedClock(Clock):
    """Wall clock that starts at ``start`` and runs ``speed`` times faster.

    TEST ONLY: lets end-to-end tests drive the real live runtime against a local
    fake broker during market hours, at any time of day. Refused for real brokers.
    """

    def __init__(self, start: float, speed: float = 1.0, wall0: float | None = None):
        self.start, self.speed, self._w0 = start, speed, time.time() if wall0 is None else wall0

    def now(self) -> float:
        return self.start + (time.time() - self._w0) * self.speed


def live_clock() -> Clock:
    spec = os.environ.get("MYBOT_TEST_CLOCK")
    if not spec:
        return WallClock()
    url = os.environ.get("ALPACA_TRADING_URL", "")
    if not (url.startswith("http://127.0.0.1") or url.startswith("http://localhost")):
        raise SystemExit("MYBOT_TEST_CLOCK is only allowed against a local fake broker")
    parts = [float(x) for x in spec.split(",")]
    # "start,speed[,wall_anchor]": processes sharing an anchor share one simulated clock
    return AcceleratedClock(parts[0], parts[1] if len(parts) > 1 else 1.0, parts[2] if len(parts) > 2 else None)


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
