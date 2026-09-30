"""Position sizing.

Base size = fixed-fractional risk (risk_per_trade_pct of equity between entry and
stop), which automatically scales with volatility because the stop is ATR/structure
based. A fractional-Kelly *multiplier* in [0.25, 1.0] can only SHRINK that base.
Kelly is never allowed to increase risk above the configured base: Kelly on
estimated probabilities is notoriously over-aggressive because estimation error
is ignored.
"""
from __future__ import annotations


def kelly_fraction(p: float, win_r: float, loss_r: float) -> float:
    if win_r <= 0 or loss_r <= 0:
        return 0.0
    b = win_r / loss_r
    return max(0.0, (p * b - (1 - p)) / b)


def size(equity: float, risk_pct: float, entry: float, stop: float, p_lcb: float, win_r: float,
         loss_r: float, model_weight: float = 1.0, kelly_frac: float = 0.25) -> tuple[int, float, dict]:
    rps = abs(entry - stop)
    if rps <= 0:
        return 0, 0.0, {"reason": "zero risk per share"}
    k = kelly_fraction(p_lcb, win_r, loss_r) * kelly_frac
    base_risk = equity * risk_pct / 100
    mult = max(0.25, min(1.0, k / (risk_pct / 100))) if k > 0 else 0.25
    mult *= max(0.0, min(1.0, model_weight))  # degraded models trade smaller
    risk_dollars = base_risk * mult
    qty = int(risk_dollars / rps)
    return qty, risk_dollars, {"kelly_quarter": round(k, 5), "multiplier": round(mult, 3), "base_risk": round(base_risk, 2)}
