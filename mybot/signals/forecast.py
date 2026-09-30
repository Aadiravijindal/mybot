"""Probabilistic forecast + expected value.

Outputs a distribution summary, never a point prediction presented as fact.
Win probability combines:
   prior      -- Beta(2,2) (i.e. "we know nothing": 50%, weak)
   analogues  -- empirical outcomes of similar past setups (point-in-time)
   evidence   -- shrunk log-odds adjustment from the evidence model
and the *lower confidence bound* of p is what gates entries, so small samples
cannot produce confident trades.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .analogues import AnalogueStats


def _logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _sig(x):
    return 1 / (1 + math.exp(-x))


@dataclass
class Forecast:
    p_win: float
    p_win_lcb: float
    p_win_ucb: float
    exp_win_r: float
    exp_loss_r: float
    cost_r: float
    ev_r: float
    ev_r_conservative: float
    tail_r_p10: float
    upside_r_p90: float
    n_eff: float
    calibrated: bool

    def to_dict(self):
        return asdict(self)


def forecast(evidence_logodds: float, an: AnalogueStats | None, target_r: float, cost_r: float,
             calibrator=None) -> Forecast:
    a0 = b0 = 2.0
    if an is not None:
        wins = an.win_rate * an.n
        a, b = a0 + wins, b0 + an.n - wins
        exp_win = an.avg_win_r or target_r
        exp_loss = an.avg_loss_r or 1.0
        tail, up = an.p10_r, an.p90_r
    else:
        a, b = a0, b0
        exp_win, exp_loss, tail, up = target_r, 1.0, -1.0, target_r
    n_eff = a + b
    p_base = a / n_eff
    p = _sig(_logit(p_base) + evidence_logodds)
    calibrated = False
    if calibrator is not None:
        p, calibrated = calibrator(p), True
    sd = math.sqrt(p * (1 - p) / (n_eff + 1))
    lcb, ucb = max(0.0, p - 1.645 * sd), min(1.0, p + 1.645 * sd)
    # Analogue R-outcomes are already net of costs; geometry defaults are not.
    c = 0.0 if an is not None else cost_r
    ev = p * exp_win - (1 - p) * exp_loss - c
    ev_c = lcb * exp_win - (1 - lcb) * exp_loss - c
    return Forecast(round(p, 4), round(lcb, 4), round(ucb, 4), round(exp_win, 3), round(exp_loss, 3), round(cost_r, 3),
                    round(ev, 3), round(ev_c, 3), round(tail, 3), round(up, 3), round(n_eff, 1), calibrated)
