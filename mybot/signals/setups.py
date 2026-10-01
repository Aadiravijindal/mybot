"""Hypothesis generators ("setups").

A setup proposes a *falsifiable* trade: direction, entry, invalidation level,
targets and horizon. It does not decide anything -- the evidence model,
analogues, adversarial review, EV and the risk engine do. Every setup is
evaluated statistically per regime; a setup that does not show out-of-sample,
after-cost edge is disabled in config, not "tuned until it works".
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..core.types import Side
from ..intel.regime import Regime


@dataclass
class Candidate:
    symbol: str
    setup: str
    side: Side
    ts: float
    entry: float
    stop: float                 # invalidation: price level that falsifies the idea
    targets: list[float]
    horizon_bars: int
    features: dict[str, float] = field(default_factory=dict)
    rationale: str = ""

    @property
    def risk_per_share(self) -> float:
        return abs(self.entry - self.stop)

    def r_of(self, price: float) -> float:
        rps = self.risk_per_share
        return (price - self.entry) * self.side.sign / rps if rps > 0 else 0.0


def _stop_long(f, atr, mult):
    base = f["close"] - mult * atr
    sw = f.get("swing_low")
    if sw is not None and f["close"] - 3 * atr < sw < f["close"] - 0.5 * atr:
        base = min(base, sw - 0.1 * atr)  # structure, with buffer beyond the obvious level
    return base


def _stop_short(f, atr, mult):
    base = f["close"] + mult * atr
    sw = f.get("swing_high")
    if sw is not None and f["close"] + 0.5 * atr < sw < f["close"] + 3 * atr:
        base = max(base, sw + 0.1 * atr)
    return base


def generate(sym: str, ts: float, f: dict[str, float], regime: Regime, catalyst: float,
             atr_mult: float = 1.5, allow_short: bool = True, bar_seconds: int = 60) -> list[Candidate]:
    if "atr" not in f or "vwap" not in f:
        return []
    c, atr = f["close"], f["atr"]
    out: list[Candidate] = []

    def mk(setup, side, rationale, horizon=60, tgt_r=(1.5, 3.0)):
        stop = _stop_long(f, atr, atr_mult) if side is Side.LONG else _stop_short(f, atr, atr_mult)
        risk = abs(c - stop)
        tg = [c + side.sign * r * risk for r in tgt_r]
        bars = max(3, round(horizon * 60 / bar_seconds))   # horizon is in minutes
        out.append(Candidate(sym, setup, side, ts, c, stop, tg, bars, dict(f), rationale))

    htf = f.get("htf_trend", 0.0)
    rsi = f.get("rsi", 50)
    relv = f.get("rel_volume", 1.0)

    # 1) Trend pullback continuation: trend on 1m AND 15m, pullback to EMA20/VWAP, momentum re-turning.
    if regime is Regime.TREND_UP and htf > 0 and -0.6 < f.get("dist_ema20_atr", 9) < 0.4 and 40 < rsi < 62 \
            and f.get("macd_hist_atr", 0) > -0.05:
        mk("TREND_PULLBACK", Side.LONG, "uptrend on 1m+15m; pullback to EMA20 with momentum stabilising")
    if allow_short and regime is Regime.TREND_DOWN and htf < 0 and -0.4 < f.get("dist_ema20_atr", -9) < 0.6 \
            and 38 < rsi < 60 and f.get("macd_hist_atr", 0) < 0.05:
        mk("TREND_PULLBACK", Side.SHORT, "downtrend on 1m+15m; bounce into EMA20 fading")

    # 2) Opening-range breakout with participation.
    orh, orl = f.get("or_high"), f.get("or_low")
    if orh and 15 < f.get("minute", 0) < 120 and relv > 1.5:
        if c > orh and c - orh < 0.5 * atr and htf >= 0:
            mk("ORB", Side.LONG, f"break of opening-range high {orh:.2f} on {relv:.1f}x volume", horizon=45)
        if allow_short and c < orl and orl - c < 0.5 * atr and htf <= 0:
            mk("ORB", Side.SHORT, f"break of opening-range low {orl:.2f} on {relv:.1f}x volume", horizon=45)

    # 3) VWAP mean reversion -- only in range regimes, and never against a fresh catalyst.
    d = f.get("dist_vwap_atr", 0)
    if regime is Regime.RANGE and abs(catalyst) < 0.2:
        if d < -2.0 and rsi < 30:
            mk("VWAP_REVERT", Side.LONG, f"stretched {d:.1f} ATR below VWAP in range regime", horizon=30, tgt_r=(1.3, 2.2))
        if allow_short and d > 2.0 and rsi > 70:
            mk("VWAP_REVERT", Side.SHORT, f"stretched {d:.1f} ATR above VWAP in range regime", horizon=30, tgt_r=(1.3, 2.2))

    # 4) Catalyst continuation: material fresh news + price & volume confirmation.
    if abs(catalyst) > 0.35 and relv > 2.0:
        if catalyst > 0 and c > f["vwap"] and f.get("ret_z", 0) > 1:
            mk("CATALYST_MOMENTUM", Side.LONG, "material positive catalyst confirmed by price above VWAP and volume")
        if allow_short and catalyst < 0 and c < f["vwap"] and f.get("ret_z", 0) < -1:
            mk("CATALYST_MOMENTUM", Side.SHORT, "material negative catalyst confirmed by price below VWAP and volume")
    return out
