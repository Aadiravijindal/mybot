"""Thesis State Machine.

STRONG -> VALID -> WEAKENING -> INVALID -> EXIT
A position is held because its thesis is intact, not because the stop has not
been hit. Evidence continuously re-scores the thesis; hard invalidations jump
straight to INVALID.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ThesisState(str, Enum):
    STRONG = "STRONG"
    VALID = "VALID"
    WEAKENING = "WEAKENING"
    INVALID = "INVALID"
    EXIT = "EXIT"


def _state(score: float) -> ThesisState:
    if score >= 0.72:
        return ThesisState.STRONG
    if score >= 0.5:
        return ThesisState.VALID
    if score >= 0.3:
        return ThesisState.WEAKENING
    return ThesisState.INVALID


@dataclass
class Thesis:
    text: str
    side: int
    setup: str
    invalidation: float
    score: float = 0.65
    state: ThesisState = ThesisState.VALID
    log: list = field(default_factory=list)   # (ts, state, score, reason)

    def update(self, f: dict[str, float], *, market_regime: str, catalyst: float, bars_held: int,
               horizon: int, r_now: float, ts: float, new_contra: list[str] | None = None) -> tuple[ThesisState, bool]:
        """Re-score from current evidence. Returns (state, changed)."""
        s = self.side
        reasons = []
        # Hard invalidations.
        hard = None
        c = f.get("close", 0)
        if (c - self.invalidation) * s < 0:
            hard = "price through invalidation level"
        elif (s > 0 and market_regime == "PANIC"):
            hard = "market entered PANIC regime"
        elif s * catalyst < -0.5:
            hard = "material contradicting catalyst"
        if hard:
            prev = self.state
            self.state, self.score = ThesisState.INVALID, 0.0
            self.log.append((ts, self.state.value, 0.0, hard))
            return self.state, prev != self.state
        # Soft evidence -> target score, then smooth (avoid flip-flopping on noise).
        tgt = 0.6
        if self.setup != "VWAP_REVERT":
            tgt += 0.12 * s * f.get("htf_trend", 0)
            tgt += 0.08 * s * f.get("ema_stack", 0)
            if s * f.get("dist_vwap_atr", 0) < -0.5:
                tgt -= 0.15; reasons.append("lost VWAP")
        else:
            if s * f.get("dist_vwap_atr", 0) > -0.2:
                tgt -= 0.1; reasons.append("reversion largely complete")
        tgt += 0.08 * s * f.get("trade_imbalance", 0)
        tgt += 0.1 * s * catalyst
        if (s > 0 and market_regime == "TREND_DOWN") or (s < 0 and market_regime == "TREND_UP"):
            tgt -= 0.15; reasons.append(f"market regime {market_regime}")
        if bars_held > horizon * 0.6 and r_now < 0.3:
            tgt -= 0.2; reasons.append("not working within expected time")
        for msg in new_contra or []:
            tgt -= 0.15; reasons.append(msg)
        tgt = max(0.0, min(1.0, tgt))
        self.score = round(0.7 * self.score + 0.3 * tgt, 4)
        new = _state(self.score)
        changed = new != self.state
        if changed:
            self.log.append((ts, new.value, self.score, "; ".join(reasons) or "evidence drift"))
        self.state = new
        return new, changed
