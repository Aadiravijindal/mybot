"""Broker abstraction + order model."""
from __future__ import annotations

import itertools
import secrets
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


class OrderStatus(str, Enum):
    NEW = "NEW"
    WORKING = "WORKING"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"


_ids = itertools.count(1)
# Client order ids must be unique per broker account *across restarts*, so every
# process run gets its own random prefix. "mb" marks orders created by this system.
RUN_ID = secrets.token_hex(3)
ORDER_PREFIX = "mb"


@dataclass
class Order:
    symbol: str
    side: int                # +1 buy, -1 sell
    qty: int
    type: OrderType
    ts: float
    tag: str                 # entry | exit | stop | reduce | flatten
    limit: float | None = None
    stop: float | None = None
    id: str = field(default_factory=lambda: f"{ORDER_PREFIX}{RUN_ID}-{next(_ids)}")
    status: OrderStatus = OrderStatus.NEW
    filled: int = 0
    avg_px: float = 0.0
    decision_px: float = 0.0  # mid at decision time, for implementation-shortfall measurement
    reject_reason: str = ""
    fees: float = 0.0

    @property
    def remaining(self) -> int:
        return self.qty - self.filled

    @property
    def done(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED)


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: int
    qty: int
    price: float
    ts: float
    fees: float
    tag: str


class Broker:
    """Interface. Implementations: PaperBroker (sim), AlpacaBroker (paper/live REST)."""
    name = "base"
    connected = True

    def __init__(self):
        self.on_fill: Callable[[Fill], None] | None = None
        self.on_order: Callable[[Order], None] | None = None

    def submit(self, o: Order) -> Order:  # pragma: no cover
        raise NotImplementedError

    def cancel(self, order_id: str) -> bool:  # pragma: no cover
        """Cancel; return True only if the order is now inactive WITHOUT having filled
        further (i.e. it is safe to send a replacement). False if it filled first."""
        raise NotImplementedError

    def replace_stop(self, order_id: str, new_stop: float) -> None:  # pragma: no cover
        raise NotImplementedError

    def positions(self) -> dict[str, int]:  # pragma: no cover
        raise NotImplementedError
