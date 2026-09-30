"""Evidence model.

Evidence items are expressed as log-likelihood-ratio contributions (log-odds).
Naively summing them double-counts correlated signals (RSI, MACD and ROC are
largely the same information). We therefore:

  1. put each item in a *correlation group* (momentum, trend, volume, flow, ...);
  2. within a group, take the strongest item plus a heavily discounted remainder;
  3. cap each group's total contribution;
  4. shrink the grand total toward zero (the weights are estimates, not truths).

The per-item weights below are *priors* that the learning pipeline re-estimates
(regularised logistic regression on labelled candidates, walk-forward). Until a
model is validated, these priors are deliberately small so they cannot dominate
the empirical analogue base rate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

GROUP_CAP = 0.6
WITHIN_GROUP_DISCOUNT = 0.25
SHRINK = 0.6


@dataclass
class Evidence:
    name: str
    group: str
    llr: float        # + supports the trade, - contradicts it
    detail: str
    source: str = "model"

    def to_dict(self):
        return asdict(self)


def combine(items: list[Evidence]) -> tuple[float, dict[str, float]]:
    groups: dict[str, list[float]] = {}
    for e in items:
        groups.setdefault(e.group, []).append(e.llr)
    per: dict[str, float] = {}
    for g, v in groups.items():
        v = sorted(v, key=abs, reverse=True)
        tot = v[0] + WITHIN_GROUP_DISCOUNT * sum(v[1:])
        per[g] = max(-GROUP_CAP, min(GROUP_CAP, tot))
    return SHRINK * sum(per.values()), per


def gather(cand, f: dict[str, float], market_regime: str, sector_rs: float | None,
           catalyst: float, dq_score: float, weights: dict[str, float] | None = None) -> list[Evidence]:
    """Bull-side evidence assembly (the Bear engine runs separately in adversarial.py)."""
    w = weights or {}
    s = cand.side.sign
    ev: list[Evidence] = []

    def add(name, group, raw, detail):
        ev.append(Evidence(name, group, round(raw * w.get(name, 1.0), 4), detail))

    add("htf_alignment", "trend", 0.25 * s * f.get("htf_trend", 0), f"15m trend {f.get('htf_trend', 0):+.0f} vs side {s:+d}")
    add("ema_stack", "trend", 0.15 * s * f.get("ema_stack", 0), "EMA 9/20/50 stack")
    add("adx_trend", "trend", 0.1 * (1 if f.get("adx", 0) > 25 else -0.5) * (1 if cand.setup != "VWAP_REVERT" else -1),
        f"ADX {f.get('adx', 0):.0f}")
    mh = f.get("macd_hist_atr", 0)
    add("macd_momentum", "momentum", max(-0.2, min(0.2, s * mh)), f"MACD hist {mh:+.2f} ATR")
    add("rsi_5m", "momentum", 0.1 * s * (1 if f.get("rsi_5m", 50) > 50 else -1), f"5m RSI {f.get('rsi_5m', 50):.0f}")
    rv = f.get("rel_volume", 1)
    add("rel_volume", "volume", 0.15 if rv > 1.5 else (-0.1 if rv < 0.6 else 0.0), f"rel volume {rv:.1f}x")
    add("obv_slope", "volume", 0.08 * s * f.get("obv_slope", 0), "OBV vs its EMA")
    if "ofi_norm" in f:
        add("order_flow_imbalance", "flow", max(-0.3, min(0.3, 0.15 * s * f["ofi_norm"])), f"OFI {f['ofi_norm']:+.2f}")
        add("trade_imbalance", "flow", 0.2 * s * f.get("trade_imbalance", 0), f"signed trade imbalance {f.get('trade_imbalance', 0):+.2f}")
        add("quote_imbalance", "flow", 0.1 * s * f.get("quote_imbalance", 0), f"L1 imbalance {f.get('quote_imbalance', 0):+.2f}")
    mr = {"TREND_UP": 1, "TREND_DOWN": -1}.get(market_regime, 0)
    add("market_regime", "market", 0.25 * s * mr - (0.3 if market_regime in ("PANIC", "HIGH_VOL") else 0),
        f"market (SPY) regime {market_regime}")
    if sector_rs is not None:
        add("relative_strength", "market", max(-0.25, min(0.25, s * sector_rs * 50)), f"RS vs market {sector_rs * 100:+.2f}%")
    if catalyst:
        add("catalyst", "catalyst", 0.5 * s * catalyst, f"catalyst score {catalyst:+.2f}")
    if dq_score < 0.9:
        add("data_quality", "quality", -(1 - dq_score), f"data quality {dq_score:.2f}")
    return ev
