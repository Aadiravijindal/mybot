"""Dynamic Stop Engine.

Maintains several stop concepts and uses the TIGHTEST applicable one, then
ratchets: a stop may only move in the direction that reduces risk. It is never
loosened because price moved against us.

  structural  -- beyond last confirmed swing low/high
  volatility  -- entry-anchored k*ATR
  trailing    -- chandelier: extreme-since-entry -/+ k*ATR (activates after +1R)
  breakeven   -- entry +/- costs once +breakeven_at_r reached
  thesis      -- tighter multiplier while thesis is WEAKENING; exit if INVALID
  emergency   -- hard % loss cap, independent of everything (also resting at broker)
  time        -- exit if the idea has not worked within its horizon
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StopState:
    side: int
    entry: float
    initial_stop: float
    current: float
    extreme: float                 # best price since entry
    emergency: float
    components: dict = field(default_factory=dict)
    history: list = field(default_factory=list)  # (ts, stop, reason)


class DynamicStopEngine:
    def __init__(self, atr_mult: float, trail_mult: float, breakeven_at_r: float, emergency_pct: float):
        self.atr_mult, self.trail_mult = atr_mult, trail_mult
        self.be_r, self.emerg = breakeven_at_r, emergency_pct

    def init(self, side: int, entry: float, stop: float, ts: float) -> StopState:
        emergency = entry * (1 - side * self.emerg / 100)
        # The emergency stop can never be looser than... it is a backstop: pick the tighter.
        cur = max(stop, emergency) if side > 0 else min(stop, emergency)
        st = StopState(side, entry, stop, cur, entry, emergency)
        st.history.append((ts, cur, "initial"))
        return st

    def update(self, st: StopState, *, high: float, low: float, close: float, atr: float,
               swing_low: float | None, swing_high: float | None, thesis: str, cost_per_share: float,
               spread: float, ts: float) -> tuple[float, str | None]:
        s = st.side
        st.extreme = max(st.extreme, high) if s > 0 else min(st.extreme, low)
        r0 = abs(st.entry - st.initial_stop)
        prog_r = (st.extreme - st.entry) * s / r0 if r0 > 0 else 0.0
        mult = self.atr_mult * (0.6 if thesis == "WEAKENING" else 1.0)
        cands = {
            "emergency": st.emergency,
            "volatility": st.entry - s * mult * atr if prog_r < 1 else None,
            "structural": (swing_low - 0.1 * atr) if (s > 0 and swing_low is not None) else
                          (swing_high + 0.1 * atr) if (s < 0 and swing_high is not None) else None,
            "breakeven": st.entry + s * cost_per_share if prog_r >= self.be_r else None,
            "trailing": st.extreme - s * (self.trail_mult * (0.6 if thesis == "WEAKENING" else 1.0)) * atr
                        if prog_r >= 1.0 else None,
        }
        # Structural stop only counts if it is on the correct side of price.
        if cands["structural"] is not None and (cands["structural"] - close) * s >= 0:
            cands["structural"] = None
        valid = {k: v for k, v in cands.items() if v is not None}
        st.components = {k: round(v, 4) for k, v in valid.items()}
        best_k, best = (max if s > 0 else min)(valid.items(), key=lambda kv: kv[1])
        # Never place the stop inside the spread / at the market.
        limit = close - s * max(spread * 2, 0.15 * atr)
        best = min(best, limit) if s > 0 else max(best, limit)
        reason = None
        if (best - st.current) * s > 1e-9:          # RATCHET: only tighten
            st.current = best
            reason = best_k
            st.history.append((ts, round(best, 4), best_k))
        return st.current, reason

    @staticmethod
    def hit(st: StopState, high: float, low: float) -> bool:
        return low <= st.current if st.side > 0 else high >= st.current
