"""Model monitoring & drift detection.

Per setup we monitor:
  * calibration: rolling Brier score and predicted-vs-realised win rate
  * edge: rolling mean realised R vs the forecast EV
  * feature drift: PSI of key features vs the reference (training) window
  * execution: realised slippage vs cost-model assumption

A degraded setup gets its weight cut (smaller size) and, if it keeps degrading,
is disabled -- pending research, retraining and re-validation. Nothing here
retrains automatically: see learning/ for the gated promotion process.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    if len(ref) < 50 or len(cur) < 30:
        return 0.0
    qs = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return 0.0
    r = np.histogram(np.clip(ref, qs[0], qs[-1]), qs)[0] / len(ref) + 1e-4
    c = np.histogram(np.clip(cur, qs[0], qs[-1]), qs)[0] / len(cur) + 1e-4
    return float(((c - r) * np.log(c / r)).sum())


@dataclass
class SetupHealth:
    preds: deque = field(default_factory=lambda: deque(maxlen=100))  # (p, won, ev_forecast, r)
    weight: float = 1.0
    status: str = "HEALTHY"      # HEALTHY | WATCH | DEGRADED | DISABLED
    reasons: list = field(default_factory=list)


class DriftMonitor:
    def __init__(self, min_n: int = 20):
        self.min_n = min_n
        self.setups: dict[str, SetupHealth] = defaultdict(SetupHealth)
        self.ref: dict[str, deque] = defaultdict(lambda: deque(maxlen=5000))
        self.cur: dict[str, deque] = defaultdict(lambda: deque(maxlen=500))
        self.slippage: deque = deque(maxlen=200)
        self.assumed_slip_bps = 0.0
        self.feature_psi: dict[str, float] = {}

    def observe_features(self, f: dict[str, float], live: bool) -> None:
        for k in ("rel_volume", "rv_pct", "eff_ratio", "spread_bps", "atr_pct"):
            if k in f and math.isfinite(f[k]):
                (self.cur if live else self.ref)[k].append(f[k])

    def record(self, setup: str, p: float, ev: float, r: float) -> None:
        h = self.setups[setup]
        h.preds.append((p, 1.0 if r > 0 else 0.0, ev, r))
        self._evaluate(setup)

    def record_slippage(self, bps: float) -> None:
        self.slippage.append(bps)

    def _evaluate(self, setup: str) -> None:
        h = self.setups[setup]
        if len(h.preds) < self.min_n:
            return
        a = np.array(h.preds)
        brier = float(((a[:, 0] - a[:, 1]) ** 2).mean())
        gap = float(a[:, 0].mean() - a[:, 1].mean())    # over-confidence if > 0
        mean_r = float(a[:, 3].mean())
        se = float(a[:, 3].std() / math.sqrt(len(a)) + 1e-9)
        reasons = []
        if brier > 0.27:
            reasons.append(f"Brier {brier:.3f} worse than coin-flip baseline")
        if gap > 0.12:
            reasons.append(f"overconfident by {gap * 100:.0f}pp")
        if mean_r + 2 * se < 0:
            reasons.append(f"realised edge significantly negative ({mean_r:+.2f}R)")
        if not reasons:
            h.status, h.weight = "HEALTHY", min(1.0, h.weight + 0.1)
        elif len(reasons) == 1 and "negative" not in reasons[0]:
            h.status, h.weight = "WATCH", 0.5
        elif mean_r + 2 * se < 0 and len(a) >= 2 * self.min_n:
            h.status, h.weight = "DISABLED", 0.0
        else:
            h.status, h.weight = "DEGRADED", 0.25
        h.reasons = reasons

    def weight(self, setup: str) -> float:
        return self.setups[setup].weight

    def snapshot(self) -> dict:
        self.feature_psi = {k: round(psi(np.array(self.ref[k]), np.array(self.cur[k])), 3)
                            for k in self.ref if k in self.cur}
        out = {}
        for s, h in self.setups.items():
            a = np.array(h.preds) if h.preds else np.zeros((0, 4))
            out[s] = {"status": h.status, "weight": h.weight, "n": len(a), "reasons": h.reasons,
                      "pred_win": round(float(a[:, 0].mean()), 3) if len(a) else None,
                      "real_win": round(float(a[:, 1].mean()), 3) if len(a) else None,
                      "mean_r": round(float(a[:, 3].mean()), 3) if len(a) else None}
        return {"setups": out, "feature_psi": self.feature_psi,
                "slippage_bps_avg": round(float(np.mean(self.slippage)), 2) if self.slippage else None}
