"""Portfolio accounting (engine-side book of record; reconciled against broker)."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..risk.stops import StopState
from ..risk.thesis import Thesis


@dataclass
class Position:
    symbol: str
    side: int
    qty: int
    avg_px: float
    entry_ts: float
    setup: str
    decision_id: str
    thesis: Thesis
    stop: StopState
    targets: list[float]
    horizon: int
    initial_risk_per_share: float
    entry_evidence: list = field(default_factory=list)
    forecast: dict = field(default_factory=dict)
    stop_order_id: str | None = None
    bars_held: int = 0
    mae: float = 0.0
    mfe: float = 0.0
    partial_done: bool = False
    fees: float = 0.0
    realized: float = 0.0
    last_px: float = 0.0
    exiting: bool = False
    decision_px: float = 0.0
    entry_slippage_bps: float = 0.0
    research: list = field(default_factory=list)   # new information seen while open
    orig_qty: int = 0

    def r_now(self, px: float) -> float:
        return (px - self.avg_px) * self.side / self.initial_risk_per_share if self.initial_risk_per_share else 0.0

    def unrealized(self, px: float) -> float:
        return (px - self.avg_px) * self.side * self.qty


@dataclass
class ClosedTrade:
    symbol: str
    side: int
    setup: str
    entry_ts: float
    exit_ts: float
    qty: int
    entry_px: float
    exit_px: float
    pnl: float
    r: float
    mae_r: float
    mfe_r: float
    bars: int
    exit_reason: str
    decision_id: str
    fees: float
    entry_slippage_bps: float
    thesis_log: list
    stop_history: list
    entry_evidence: list
    forecast: dict
    attribution: dict = field(default_factory=dict)


class Book:
    def __init__(self, equity: float):
        self.cash = equity
        self.start_equity = equity
        self.day_start_equity = equity
        self.hwm = equity
        self.positions: dict[str, Position] = {}
        self.closed: list[ClosedTrade] = []
        self.equity_curve: list[tuple[float, float]] = []
        self.marks: dict[str, float] = {}

    def equity(self) -> float:
        return self.cash + sum(p.side * p.qty * self.marks.get(s, p.avg_px) for s, p in self.positions.items())

    def mark(self, sym: str, px: float) -> None:
        self.marks[sym] = px
        if sym in self.positions:
            p = self.positions[sym]
            p.last_px = px
            r = p.r_now(px)
            p.mae, p.mfe = min(p.mae, r), max(p.mfe, r)

    def record_equity(self, ts: float) -> float:
        e = self.equity()
        self.hwm = max(self.hwm, e)
        if not self.equity_curve or ts - self.equity_curve[-1][0] >= 60:
            self.equity_curve.append((ts, round(e, 2)))
            if len(self.equity_curve) > 20000:
                self.equity_curve = self.equity_curve[-20000:]
        return e
