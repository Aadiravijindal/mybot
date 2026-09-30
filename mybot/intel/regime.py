"""Market Regime Detection.

Online, interpretable classifier with hysteresis (a regime must persist for
``confirm`` consecutive bars before we switch -- avoids flapping). A causal
Gaussian HMM (``hmm.py``) is provided for offline benchmarking; see the docs for
why the rules-based detector is the production default until the HMM beats it
out of sample.
"""
from __future__ import annotations

from enum import Enum


class Regime(str, Enum):
    UNKNOWN = "UNKNOWN"
    TREND_UP = "TREND_UP"
    TREND_DOWN = "TREND_DOWN"
    RANGE = "RANGE"
    HIGH_VOL = "HIGH_VOL"
    PANIC = "PANIC"
    ILLIQUID = "ILLIQUID"


def classify(f: dict[str, float]) -> Regime:
    if "atr" not in f:
        return Regime.UNKNOWN
    if f.get("spread_bps", 0) > 40:
        return Regime.ILLIQUID
    rvp = f.get("rv_pct", 0.5)
    er, adx = f.get("eff_ratio", 0.0), f.get("adx", 0.0)
    ret_day = f.get("ret_day", 0.0)
    if rvp > 0.95 and ret_day < -0.02:
        return Regime.PANIC
    trending = er > 0.35 and adx > 22
    if trending and f.get("ema_stack", 0) > 0 and f.get("di_diff", 0) > 0:
        return Regime.TREND_UP
    if trending and f.get("ema_stack", 0) < 0 and f.get("di_diff", 0) < 0:
        return Regime.TREND_DOWN
    if rvp > 0.9:
        return Regime.HIGH_VOL
    if er < 0.25 and adx < 20:
        return Regime.RANGE
    return Regime.UNKNOWN


class RegimeTracker:
    def __init__(self, confirm: int = 3):
        self.confirm = confirm
        self.current = Regime.UNKNOWN
        self._cand, self._n = Regime.UNKNOWN, 0
        self.since_ts = 0.0

    def update(self, f: dict[str, float], ts: float) -> tuple[Regime, bool]:
        r = classify(f)
        if r == self.current:
            self._cand, self._n = r, 0
            return self.current, False
        if r == self._cand:
            self._n += 1
        else:
            self._cand, self._n = r, 1
        # Dangerous regimes switch immediately; benign ones need confirmation.
        need = 1 if r in (Regime.PANIC, Regime.ILLIQUID) else self.confirm
        if self._n >= need:
            self.current, self._n, self.since_ts = r, 0, ts
            return r, True
        return self.current, False
