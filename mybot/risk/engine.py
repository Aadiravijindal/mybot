"""Risk Engine -- FINAL AUTHORITY.

Deliberately simple, deterministic, and independent of every model. It does not
import anything from signals/ or intel/. The AI proposes; this module disposes.

States:
  NORMAL       -- entries allowed within limits
  REDUCE_ONLY  -- no new risk; exits / stop-tightening only
  HALTED       -- no new orders; flatten positions; manual resume required
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from ..config import RiskLimits


class RiskState(str, Enum):
    NORMAL = "NORMAL"
    REDUCE_ONLY = "REDUCE_ONLY"
    HALTED = "HALTED"


@dataclass
class RiskDecision:
    approved: bool
    qty: int
    reasons: list[str] = field(default_factory=list)


@dataclass
class PortfolioView:
    equity: float
    day_start_equity: float
    hwm: float
    positions: dict          # sym -> (qty signed, price)
    sectors: dict            # sym -> sector


class RiskEngine:
    def __init__(self, limits: RiskLimits, sectors: dict[str, str]):
        self.L = limits
        self.sectors = sectors
        self.state = RiskState.NORMAL
        self.reasons: deque = deque(maxlen=50)   # (ts, state, reason)
        self.trades_today = 0
        self.consec_losses = 0
        self.order_times: deque = deque(maxlen=200)
        self.component_errors: deque = deque(maxlen=100)
        self.day: str | None = None
        # US pattern-day-trader rule: None = unlimited; otherwise day trades left in the rolling window.
        self.pdt_remaining: int | None = None

    # ------------------------------------------------------------- state
    def _escalate(self, new: RiskState, reason: str, ts: float) -> None:
        order = [RiskState.NORMAL, RiskState.REDUCE_ONLY, RiskState.HALTED]
        if order.index(new) > order.index(self.state):
            self.state = new
        self.reasons.append((ts, new.value, reason))

    def halt(self, reason: str, ts: float | None = None) -> None:
        self._escalate(RiskState.HALTED, reason, ts or time.time())

    def reduce_only(self, reason: str, ts: float | None = None) -> None:
        self._escalate(RiskState.REDUCE_ONLY, reason, ts or time.time())

    def manual_resume(self, who: str, ts: float | None = None) -> None:
        """Only a human can clear HALTED (audited)."""
        self.state = RiskState.NORMAL
        self.reasons.append((ts or time.time(), "NORMAL", f"manual resume by {who}"))

    def new_day(self, day: str, ts: float) -> None:
        if day != self.day:
            self.day = day
            self.trades_today = 0
            self.consec_losses = 0
            if self.state is RiskState.REDUCE_ONLY:  # daily limits reset; HALTED does not
                self.state = RiskState.NORMAL
                self.reasons.append((ts, "NORMAL", "new session: daily reduce-only cleared"))

    # ------------------------------------------------------ continuous checks
    def check_portfolio(self, pv: PortfolioView, ts: float) -> None:
        day_pnl_pct = (pv.equity / pv.day_start_equity - 1) * 100
        dd_pct = (1 - pv.equity / pv.hwm) * 100 if pv.hwm > 0 else 0
        if day_pnl_pct <= -self.L.max_daily_loss_pct:
            self.halt(f"daily loss {day_pnl_pct:.2f}% breached -{self.L.max_daily_loss_pct}%", ts)
        if dd_pct >= self.L.max_drawdown_pct:
            self.halt(f"drawdown {dd_pct:.2f}% breached {self.L.max_drawdown_pct}%", ts)
        gross = sum(abs(q * p) for q, p in pv.positions.values())
        if gross > pv.equity * self.L.max_leverage * 1.05:
            self.reduce_only(f"gross exposure {gross:,.0f} above leverage limit", ts)

    def record_trade_result(self, pnl: float, ts: float) -> None:
        self.consec_losses = self.consec_losses + 1 if pnl < 0 else 0
        if self.consec_losses >= self.L.max_consecutive_losses:
            self.reduce_only(f"{self.consec_losses} consecutive losses", ts)

    def record_slippage(self, bps: float, ts: float) -> None:
        if bps > self.L.max_slippage_bps:
            self.reduce_only(f"realized slippage {bps:.0f}bps > {self.L.max_slippage_bps}", ts)

    def record_component_error(self, what: str, ts: float) -> None:
        self.component_errors.append(ts)
        recent = [t for t in self.component_errors if ts - t < 300]
        if len(recent) >= 5:
            self.halt(f"repeated component failures ({what})", ts)

    def check_infrastructure(self, *, data_staleness_s: float, clock_drift_s: float,
                             broker_ok: bool, reconciled: bool, market_open: bool, ts: float,
                             staleness_limit_s: float | None = None) -> None:
        limit = staleness_limit_s if staleness_limit_s is not None else self.L.max_data_staleness_s
        if market_open and data_staleness_s > limit:
            self.halt(f"market data stale for {data_staleness_s:.0f}s", ts)
        if abs(clock_drift_s) > self.L.max_clock_drift_s:
            self.halt(f"clock drift {clock_drift_s:.2f}s", ts)
        if not broker_ok:
            self.halt("broker disconnected", ts)
        if not reconciled:
            self.halt("position state uncertain (reconciliation mismatch)", ts)

    # ---------------------------------------------------------- pre-trade
    def pre_trade(self, *, sym: str, side_sign: int, qty: int, price: float, stop: float,
                  spread_bps: float, avg_dollar_vol: float, pv: PortfolioView, ts: float) -> RiskDecision:
        L, reasons = self.L, []
        if self.state is not RiskState.NORMAL:
            return RiskDecision(False, 0, [f"risk state {self.state.value}"])
        if side_sign < 0 and not L.allow_short:
            return RiskDecision(False, 0, ["shorting disabled"])
        if len(pv.positions) >= L.max_open_positions and sym not in pv.positions:
            return RiskDecision(False, 0, ["max open positions"])
        if self.trades_today >= L.max_trades_per_day:
            return RiskDecision(False, 0, ["max trades per day"])
        if self.pdt_remaining is not None and self.pdt_remaining <= 0:
            return RiskDecision(False, 0, ["pattern-day-trader limit: no day trades left (account < $25k)"])
        recent = [t for t in self.order_times if ts - t < 60]
        if len(recent) >= L.max_orders_per_minute:
            return RiskDecision(False, 0, ["order rate limit"])
        if price < L.min_price:
            return RiskDecision(False, 0, [f"price below {L.min_price}"])
        if spread_bps > L.max_spread_bps:
            return RiskDecision(False, 0, [f"spread {spread_bps:.1f}bps > {L.max_spread_bps}"])
        if avg_dollar_vol < L.min_avg_dollar_volume:
            return RiskDecision(False, 0, ["insufficient liquidity"])
        rps = abs(price - stop)
        if rps <= 0 or side_sign * (price - stop) <= 0:
            return RiskDecision(False, 0, ["invalid stop (must be on the losing side of entry)"])
        # Hard caps -- qty can only be REDUCED here, never increased.
        max_by_risk = int(pv.equity * L.risk_per_trade_pct / 100 / rps)
        max_by_notional = int(pv.equity * L.max_position_notional_pct / 100 / price)
        gross = sum(abs(q * p) for q, p in pv.positions.values())
        max_by_gross = int(max(0.0, pv.equity * min(L.max_gross_exposure_pct / 100, L.max_leverage) - gross) / price)
        sec = self.sectors.get(sym, "OTHER")
        sec_exp = sum(abs(q * p) for s, (q, p) in pv.positions.items() if self.sectors.get(s, "OTHER") == sec)
        max_by_sector = int(max(0.0, pv.equity * L.max_sector_exposure_pct / 100 - sec_exp) / price)
        max_by_liq = int(avg_dollar_vol / 390 * 30 * L.max_participation_pct / 100 / price)  # 30-min participation
        caps = {"risk": max_by_risk, "notional": max_by_notional, "gross": max_by_gross,
                "sector": max_by_sector, "liquidity": max_by_liq}
        final = min([qty] + list(caps.values()))
        for k, v in caps.items():
            if v < qty:
                reasons.append(f"qty capped by {k} limit ({qty}->{v})")
        if final <= 0:
            return RiskDecision(False, 0, reasons or ["size rounds to zero"])
        return RiskDecision(True, final, reasons)

    def on_order_sent(self, ts: float, opening: bool) -> None:
        self.order_times.append(ts)
        if opening:
            self.trades_today += 1
            if self.pdt_remaining is not None:
                self.pdt_remaining -= 1

    def snapshot(self) -> dict:
        return {"state": self.state.value, "trades_today": self.trades_today, "pdt_day_trades_left": self.pdt_remaining,
                "consecutive_losses": self.consec_losses,
                "recent_reasons": [{"ts": t, "state": s, "reason": r} for t, s, r in list(self.reasons)[-10:]]}
