"""In-process event bus.

Deterministic FIFO dispatch: publish() enqueues, and the queue is drained in order,
so a handler that publishes a follow-up event never recurses. The same bus runs in
backtest and live mode; in live mode async producers feed it via ``publish``.

In production this is replaced/fronted by a durable log (Redpanda/Kafka or NATS
JetStream) -- see docs/ARCHITECTURE.md section E. The topic names are the contract.
"""
from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from .clock import Clock, LookAheadError

log = logging.getLogger(__name__)


class Topic(str, Enum):
    BAR = "BAR"
    QUOTE = "QUOTE"
    TRADE = "TRADE"
    NEWS = "NEWS"
    SCHEDULE = "SCHEDULE"
    DATA_QUALITY = "DATA_QUALITY"
    ANOMALY = "ANOMALY"
    REGIME_CHANGE = "REGIME_CHANGE"
    CATALYST = "CATALYST"
    RESEARCH = "RESEARCH"
    DECISION = "DECISION"
    ORDER = "ORDER"
    FILL = "FILL"
    STOP_UPDATE = "STOP_UPDATE"
    THESIS_CHANGE = "THESIS_CHANGE"
    RISK_EVENT = "RISK_EVENT"
    HEALTH = "HEALTH"


@dataclass(slots=True)
class Event:
    topic: Topic
    ts: float                 # avail_ts of the underlying information
    payload: Any
    symbol: str | None = None


Handler = Callable[[Event], None]


class EventBus:
    def __init__(self, clock: Clock, strict_pit: bool = True):
        self.clock = clock
        self.strict_pit = strict_pit
        self._subs: dict[Topic, list[Handler]] = defaultdict(list)
        self._q: deque[Event] = deque()
        self._draining = False
        self.counts: dict[str, int] = defaultdict(int)
        self.errors: int = 0
        self.on_error: Callable[[Event, BaseException], None] | None = None

    def subscribe(self, topic: Topic, handler: Handler) -> None:
        self._subs[topic].append(handler)

    def publish(self, ev: Event) -> None:
        # Point-in-time guard: nothing may enter the system before it existed.
        if self.strict_pit and ev.ts > self.clock.now() + 1e-6:
            raise LookAheadError(
                f"{ev.topic} for {ev.symbol} available at {ev.ts} but clock is {self.clock.now()}"
            )
        self._q.append(ev)
        if not self._draining:
            self._drain()

    def _drain(self) -> None:
        self._draining = True
        try:
            while self._q:
                ev = self._q.popleft()
                self.counts[ev.topic.value] += 1
                for h in self._subs.get(ev.topic, ()):
                    try:
                        h(ev)
                    except LookAheadError:
                        raise
                    except Exception as exc:  # one bad handler must not kill the bus...
                        self.errors += 1
                        log.exception("handler %s failed on %s", getattr(h, "__qualname__", h), ev.topic)
                        if self.on_error is not None:  # ...but the risk layer gets told
                            self.on_error(ev, exc)
        finally:
            self._draining = False
