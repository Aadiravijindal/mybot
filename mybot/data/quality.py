"""Data Quality Engine.

Every record passes through here before any analytic sees it. Bad records are
dropped (and counted); degraded symbols are flagged so the signal layer abstains;
feed-wide failures escalate to the Risk Engine, which halts new entries.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

from ..core.types import Bar, DataQuality, Quote, Trade


@dataclass
class SymbolDQ:
    last_bar_ts: float = float("-inf")
    last_bar_end: float = 0.0
    last_close: float | None = None
    last_quote_ts: float = 0.0
    last_quote: Quote | None = None
    seen_trades: deque = field(default_factory=lambda: deque(maxlen=500))
    issues: deque = field(default_factory=lambda: deque(maxlen=50))  # (ts, issue)
    rejected: int = 0
    accepted: int = 0


class DataQualityEngine:
    def __init__(self, max_staleness_s: float = 180.0, max_jump_pct: float = 8.0,
                 max_spread_bps: float = 150.0, disagreement_bps: float = 30.0, bar_seconds: int = 60):
        self.bar_seconds = bar_seconds
        self.max_staleness_s = max_staleness_s
        self.max_jump_pct = max_jump_pct
        self.max_spread_bps = max_spread_bps
        self.disagreement_bps = disagreement_bps
        self.s: dict[str, SymbolDQ] = defaultdict(SymbolDQ)
        self.feed_last_seen: dict[str, float] = {}
        self.latency: deque = deque(maxlen=500)

    def _flag(self, sym: str, now: float, msg: str) -> None:
        self.s[sym].issues.append((now, msg))

    # Each check returns True if the record may be used.
    def check_bar(self, b: Bar, now: float) -> bool:
        st = self.s[b.symbol]
        self.feed_last_seen[b.source] = now
        self.latency.append(max(0.0, b.avail_ts - b.end_ts))
        bad = None
        if not (b.low <= min(b.open, b.close) and b.high >= max(b.open, b.close)) or b.low <= 0:
            bad = "inconsistent OHLC"
        elif b.volume < 0:
            bad = "negative volume"
        elif b.ts <= st.last_bar_ts:
            bad = "duplicate/out-of-order bar"
        elif b.avail_ts + 1e-6 < b.end_ts:
            bad = "bar available before it closed (timestamp error)"
        elif st.last_close and abs(b.close / st.last_close - 1) * 100 > self.max_jump_pct:
            # Could be real (news, halt reopen). We accept but flag -> symbol degraded briefly.
            self._flag(b.symbol, now, f"price jump {abs(b.close / st.last_close - 1) * 100:.1f}% (verify: news/halt/corp action?)")
        if bad:
            st.rejected += 1
            self._flag(b.symbol, now, bad)
            return False
        if st.last_bar_end and b.ts - st.last_bar_end > max(300, 3 * self.bar_seconds) and b.ts - st.last_bar_end < 3600 * 6:
            self._flag(b.symbol, now, f"gap of {(b.ts - st.last_bar_end) / 60:.0f} min in bars (halt or feed gap)")
        st.last_bar_ts, st.last_bar_end, st.last_close = b.ts, b.end_ts, b.close
        st.accepted += 1
        return True

    def check_quote(self, q: Quote, now: float) -> bool:
        st = self.s[q.symbol]
        self.feed_last_seen[q.source] = now
        if q.bid <= 0 or q.ask <= 0 or q.ask < q.bid:
            st.rejected += 1
            self._flag(q.symbol, now, "crossed/invalid quote")
            return False
        if q.spread_bps > self.max_spread_bps:
            self._flag(q.symbol, now, f"abnormal spread {q.spread_bps:.0f}bps")
        if st.last_close and abs(q.mid / st.last_close - 1) * 1e4 > 10 * self.disagreement_bps:
            self._flag(q.symbol, now, "quote mid disagrees with last trade/bar")
        st.last_quote_ts, st.last_quote = q.ts, q
        return True

    def check_trade(self, t: Trade, now: float) -> bool:
        st = self.s[t.symbol]
        key = (t.ts, t.price, t.size)
        if key in st.seen_trades:
            st.rejected += 1
            return False  # duplicate tick
        st.seen_trades.append(key)
        if t.price <= 0 or t.size <= 0:
            st.rejected += 1
            return False
        return True

    def cross_source(self, sym: str, price_a: float, price_b: float, now: float) -> bool:
        """Compare two independent sources; flag disagreement."""
        bps = abs(price_a / price_b - 1) * 1e4
        if bps > self.disagreement_bps:
            self._flag(sym, now, f"source disagreement {bps:.0f}bps")
            return False
        return True

    def quality(self, sym: str, now: float) -> DataQuality:
        st = self.s.get(sym)
        if st is None or not st.last_bar_end:
            return DataQuality(ok=False, score=0.0, freshness_s=float("inf"), issues=["no data"])
        fresh = now - st.last_bar_end
        recent = [m for (t, m) in st.issues if now - t < 600]
        score = 1.0
        score -= min(0.5, 0.1 * len(recent))
        if fresh > self.max_staleness_s:
            score -= 0.6
            recent = recent + [f"stale: last bar {fresh:.0f}s ago"]
        if st.last_quote and st.last_quote.spread_bps > self.max_spread_bps:
            score -= 0.3
        score = max(0.0, score)
        return DataQuality(ok=score >= 0.6, score=score, freshness_s=fresh, issues=recent[-5:])

    def feed_health(self, now: float) -> dict:
        lat = sorted(self.latency)
        return {
            "feeds": {k: round(now - v, 1) for k, v in self.feed_last_seen.items()},
            "latency_p50_s": lat[len(lat) // 2] if lat else None,
            "latency_p95_s": lat[int(len(lat) * 0.95)] if lat else None,
            "rejected": sum(s.rejected for s in self.s.values()),
            "accepted": sum(s.accepted for s in self.s.values()),
        }
