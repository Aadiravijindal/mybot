"""Adversarial (Bear-case) validation.

Runs independently of the generator with one job: find reasons the trade will
fail. It returns contradicting evidence (which lowers probability) and hard
vetoes (which block the trade outright regardless of how good the bull case looks).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .analogues import AnalogueStats
from .evidence import Evidence
from .setups import Candidate


@dataclass
class Challenge:
    contra: list[Evidence] = field(default_factory=list)
    vetoes: list[str] = field(default_factory=list)


def challenge(c: Candidate, f: dict[str, float], *, market_regime: str, catalyst: float,
              upcoming_events: list, an: AnalogueStats | None, dq_ok: bool, spread_bps: float,
              max_spread_bps: float, min_analogues: int, minutes_to_close: float,
              no_entry_last_min: int, recent_contra_news: list) -> Challenge:
    ch = Challenge()
    s = c.side.sign
    add = lambda n, llr, d: ch.contra.append(Evidence(n, "adversarial", llr, d, source="bear_agent"))

    if not dq_ok:
        ch.vetoes.append("data quality below threshold")
    if spread_bps > max_spread_bps:
        ch.vetoes.append(f"spread {spread_bps:.1f}bps > limit {max_spread_bps}")
    for e in upcoming_events:
        if e.importance >= 3:
            ch.vetoes.append(f"high-impact scheduled event soon: {e.name}")
        else:
            add("event_risk", -0.15, f"scheduled event within horizon: {e.name}")
    if minutes_to_close < no_entry_last_min:
        ch.vetoes.append(f"only {minutes_to_close:.0f} min to close")
    if an is None or an.n < min_analogues:
        ch.vetoes.append(f"insufficient historical analogues ({0 if an is None else an.n} < {min_analogues}) -- unknown edge")
    elif an.mean_r <= 0:
        add("analogue_failure", -0.3, f"similar setups lost on average ({an.mean_r:+.2f}R, n={an.n})")

    # Overextension: chasing a move already far from value.
    dv = s * f.get("dist_vwap_atr", 0)
    if c.setup != "VWAP_REVERT" and dv > 3.0:
        add("overextension", -0.3, f"price {dv:.1f} ATR beyond VWAP in trade direction (chasing)")
    rsi = f.get("rsi", 50)
    if c.setup != "VWAP_REVERT" and ((s > 0 and rsi > 78) or (s < 0 and rsi < 22)):
        add("exhaustion", -0.2, f"RSI {rsi:.0f} at extreme in trade direction")
    # Fighting the market / higher timeframe.
    if (s > 0 and market_regime in ("TREND_DOWN", "PANIC")) or (s < 0 and market_regime == "TREND_UP"):
        add("against_market", -0.3, f"trade opposes market regime {market_regime}")
    if s * f.get("htf_trend", 0) < 0 and c.setup != "VWAP_REVERT":
        add("against_htf", -0.25, "trade opposes 15m trend")
    # Catalyst pointing the other way.
    if s * catalyst < -0.2:
        add("contra_catalyst", -0.5, f"recent catalyst opposes trade ({catalyst:+.2f})")
    for a in recent_contra_news[:3]:
        add("contra_news", -0.2, f"{a.source}: {a.headline[:80]}")
    # False-breakout signature: breakout on weak participation.
    if c.setup == "ORB" and f.get("rel_volume", 1) < 1.2:
        add("weak_breakout", -0.3, "breakout without volume confirmation")
    # Flow disagrees with direction.
    if s * f.get("trade_imbalance", 0) < -0.3:
        add("flow_disagrees", -0.2, "aggressive flow is on the other side")
    if f.get("spread_z", 0) > 3:
        add("liquidity_withdrawal", -0.2, "spread blowing out vs recent history")
    return ch
