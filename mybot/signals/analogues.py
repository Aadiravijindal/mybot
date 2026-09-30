"""Historical Analogue Engine + triple-barrier outcome labeller.

EVERY candidate (traded or not) is labelled by what subsequently happened:
stop hit (-1R), target hit (+kR) or time-out (mark-to-market R), net of costs.
Labelling all candidates -- not just the trades we took -- avoids selection bias
in the knowledge base.

Point-in-time rule: an analogue is only usable once its outcome was *known*
(``outcome_ts <= now``). A 60-minute trade that started 10 minutes ago is not an
analogue yet.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .setups import Candidate

FEATURES = ("dist_vwap_atr", "dist_ema20_atr", "rsi", "adx", "eff_ratio", "rel_volume", "rv_pct",
            "bb_z", "macd_hist_atr", "htf_trend", "ofi_norm", "trade_imbalance", "minute")


def vectorize(f: dict[str, float], side_sign: int) -> np.ndarray:
    v = []
    for k in FEATURES:
        x = f.get(k, 0.0)
        if k in ("dist_vwap_atr", "dist_ema20_atr", "macd_hist_atr", "htf_trend", "ofi_norm", "trade_imbalance", "bb_z"):
            x *= side_sign  # orient directional features to the trade side
        if k == "rsi":
            x = (x - 50) * side_sign
        if k == "minute":
            x = x / 60.0
        v.append(x if math.isfinite(x) else 0.0)
    return np.asarray(v, float)


@dataclass
class Outcome:
    setup: str
    side: int
    start_ts: float
    outcome_ts: float
    r: float
    mae_r: float
    mfe_r: float
    bars: int
    exit: str
    vec: np.ndarray
    regime: str = ""


class OutcomeLabeler:
    def __init__(self, cost_r_fn):
        self.pending: list[dict] = []
        self.cost_r_fn = cost_r_fn

    def add(self, c: Candidate, regime: str) -> None:
        self.pending.append({"c": c, "regime": regime, "n": 0, "mae": 0.0, "mfe": 0.0})

    def on_bar(self, sym: str, high: float, low: float, close: float, end_ts: float) -> list[Outcome]:
        done, keep = [], []
        for p in self.pending:
            c: Candidate = p["c"]
            if c.symbol != sym or end_ts <= c.ts:
                keep.append(p)
                continue
            p["n"] += 1
            fav, adv = (high, low) if c.side.sign > 0 else (low, high)
            p["mfe"] = max(p["mfe"], c.r_of(fav))
            p["mae"] = min(p["mae"], c.r_of(adv))
            exit_ = None
            if p["mae"] <= -1.0:           # conservative: stop assumed hit first if both touched
                r, exit_ = -1.0, "stop"
            elif c.r_of(fav) >= c.r_of(c.targets[0]):
                r, exit_ = c.r_of(c.targets[0]), "target"
            elif p["n"] >= c.horizon_bars:
                r, exit_ = c.r_of(close), "time"
            if exit_:
                r -= self.cost_r_fn(c)
                done.append(Outcome(c.setup, c.side.sign, c.ts, end_ts, r, p["mae"], p["mfe"], p["n"], exit_,
                                    vectorize(c.features, c.side.sign), p["regime"]))
            else:
                keep.append(p)
        self.pending = keep
        return done


@dataclass
class AnalogueStats:
    n: int
    win_rate: float
    mean_r: float
    median_r: float
    p10_r: float
    p90_r: float
    avg_win_r: float
    avg_loss_r: float
    mae_p75_r: float
    mfe_median_r: float
    similarity: float


class AnalogueStore:
    def __init__(self, k: int = 50, max_size: int = 20000):
        self.k, self.max_size = k, max_size
        self.items: list[Outcome] = []

    def add(self, o: Outcome) -> None:
        self.items.append(o)
        if len(self.items) > self.max_size:
            self.items = self.items[-self.max_size:]

    def query(self, setup: str, side: int, f: dict[str, float], now: float) -> AnalogueStats | None:
        pool = [o for o in self.items if o.setup == setup and o.side == side and o.outcome_ts <= now]
        if len(pool) < 5:
            return None
        X = np.stack([o.vec for o in pool])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        q = (vectorize(f, side) - mu) / sd
        d = np.sqrt((((X - mu) / sd - q) ** 2).sum(1))
        idx = np.argsort(d)[: self.k]
        sel = [pool[i] for i in idx]
        r = np.array([o.r for o in sel])
        wins, losses = r[r > 0], r[r <= 0]
        return AnalogueStats(
            n=len(sel), win_rate=float((r > 0).mean()), mean_r=float(r.mean()), median_r=float(np.median(r)),
            p10_r=float(np.percentile(r, 10)), p90_r=float(np.percentile(r, 90)),
            avg_win_r=float(wins.mean()) if len(wins) else 0.0,
            avg_loss_r=float(-losses.mean()) if len(losses) else 1.0,
            mae_p75_r=float(np.percentile([o.mae_r for o in sel], 25)),
            mfe_median_r=float(np.median([o.mfe_r for o in sel])),
            similarity=float(1 / (1 + d[idx].mean())),
        )
