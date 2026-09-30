"""News Intelligence: classification, dedup/novelty, reliability, impact tracking.

The default classifier is a transparent lexicon model: fast, deterministic,
auditable, and good enough to route. When ANTHROPIC_API_KEY is set, material
stories are additionally sent to an LLM extractor (``llm.py``) whose output is
*structured facts*, never a trade instruction. The LLM never sees credentials,
positions, or order capability.
"""
from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field

from ..core.types import NewsItem

SOURCE_RELIABILITY = {
    # Primary sources > wires > aggregators > social. Tune from measured accuracy.
    "sec_edgar": 0.98, "company_pr": 0.95, "businesswire": 0.93, "prnewswire": 0.93,
    "reuters": 0.92, "bloomberg": 0.92, "dow_jones": 0.92, "benzinga": 0.8,
    "alpaca_news": 0.8, "synthetic": 0.8, "blog": 0.4, "social": 0.25, "unknown": 0.5,
}

# (event_type, direction, materiality prior, patterns)
LEXICON: list[tuple[str, int, float, tuple[str, ...]]] = [
    ("EARNINGS_BEAT", 1, 0.8, ("beats estimates", "tops estimates", "above expectations", "record revenue", "beat consensus")),
    ("EARNINGS_MISS", -1, 0.8, ("misses estimates", "below expectations", "falls short", "missed consensus")),
    ("GUIDANCE_UP", 1, 0.85, ("raises guidance", "raises outlook", "raises forecast", "boosts full-year")),
    ("GUIDANCE_DOWN", -1, 0.85, ("cuts guidance", "lowers outlook", "lowers forecast", "withdraws guidance", "profit warning")),
    ("UPGRADE", 1, 0.5, ("upgrade", "upgraded", "raises price target", "initiates at buy", "outperform")),
    ("DOWNGRADE", -1, 0.5, ("downgrade", "downgraded", "cuts price target", "underperform")),
    ("M_AND_A_TARGET", 1, 0.95, ("to be acquired", "agrees to be bought", "takeover bid for", "buyout offer")),
    ("M_AND_A_ACQUIRER", 0, 0.6, ("to acquire", "agrees to buy", "acquisition of")),
    ("REGULATORY_POS", 1, 0.8, ("fda approval", "fda approves", "clearance granted", "wins approval")),
    ("REGULATORY_NEG", -1, 0.8, ("fda rejects", "complete response letter", "antitrust probe", "sec investigation", "recall")),
    ("LEGAL_NEG", -1, 0.6, ("lawsuit", "sued", "class action", "fined", "settlement charge")),
    ("CONTRACT_WIN", 1, 0.55, ("wins contract", "awarded contract", "partnership with", "strategic partnership")),
    ("PRODUCT", 1, 0.35, ("launches", "unveils", "announces new")),
    ("INSIDER_BUY", 1, 0.35, ("insider buying", "director buys", "ceo buys")),
    ("INSIDER_SELL", -1, 0.2, ("insider selling", "ceo sells", "director sells")),
    ("OFFERING", -1, 0.7, ("secondary offering", "stock offering", "dilution", "convertible notes offering")),
    ("MGMT_CHANGE", -1, 0.5, ("ceo resigns", "cfo resigns", "steps down", "abrupt departure")),
    ("MACRO", 0, 0.6, ("federal reserve", "rate decision", "cpi", "inflation data", "jobs report", "tariff")),
]

_WORD = re.compile(r"[a-z0-9']+")


def shingles(text: str, k: int = 3) -> set[tuple[str, ...]]:
    w = _WORD.findall(text.lower())
    return {tuple(w[i:i + k]) for i in range(max(len(w) - k + 1, 1))} if w else set()


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


@dataclass
class NewsAssessment:
    id: str
    ts: float
    avail_ts: float
    symbols: tuple[str, ...]
    headline: str
    source: str
    event_type: str
    direction: int            # +1 / -1 / 0
    reliability: float
    novelty: float            # 1 = new story, 0 = pure repeat
    materiality: float        # 0..1
    confidence: float         # 0..1 in the classification
    expected_impact_bps: float
    is_duplicate: bool
    llm: dict | None = None
    actual_impact_bps: dict = field(default_factory=dict)  # filled in later by ImpactTracker

    def to_dict(self) -> dict:
        return asdict(self)


class NewsIntel:
    def __init__(self, dup_threshold: float = 0.55, window: int = 300):
        self.recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self.dup_threshold = dup_threshold

    def assess(self, n: NewsItem) -> NewsAssessment:
        text = f"{n.headline} {n.body}".lower()
        best = ("GENERAL", 0, 0.15, 0)
        for et, d, mat, pats in LEXICON:
            hits = sum(1 for p in pats if p in text)
            if hits and mat * (1 + 0.2 * hits) > best[2] * (1 + 0.2 * best[3]):
                best = (et, d, mat, hits)
        et, direction, materiality, hits = best
        conf = min(0.9, 0.35 + 0.2 * hits) if hits else 0.2
        sh = shingles(n.headline)
        sims = [jaccard(sh, s) for sym in n.symbols for (s, _) in self.recent[sym]]
        max_sim = max(sims, default=0.0)
        novelty = 1.0 - max_sim
        dup = max_sim >= self.dup_threshold
        for sym in n.symbols:
            self.recent[sym].append((sh, n.id))
        rel = SOURCE_RELIABILITY.get(n.source, SOURCE_RELIABILITY["unknown"])
        eff_mat = materiality * rel * (0.15 if dup else novelty)
        exp_bps = direction * eff_mat * 150  # crude prior; replaced by measured per-type response
        return NewsAssessment(n.id, n.ts, n.avail_ts, n.symbols, n.headline, n.source, et, direction,
                              rel, round(novelty, 3), round(eff_mat, 3), round(conf, 3), round(exp_bps, 1), dup)


class ImpactTracker:
    """Records the *actual* post-news price response at fixed horizons.

    This is how per-event-type priors get re-estimated from evidence rather than
    hard-coded. Outcomes are attached only after the horizon has elapsed.
    """

    HORIZONS = (5, 15, 30, 60)  # minutes

    def __init__(self):
        self.pending: list[tuple[NewsAssessment, str, float, float]] = []
        self.stats: dict[str, list[float]] = defaultdict(list)

    def track(self, a: NewsAssessment, sym: str, price: float) -> None:
        self.pending.append((a, sym, a.avail_ts, price))

    def on_price(self, sym: str, ts: float, price: float) -> None:
        keep = []
        for a, s, t0, p0 in self.pending:
            if s == sym:
                for h in self.HORIZONS:
                    k = f"{h}m"
                    if k not in a.actual_impact_bps and ts - t0 >= h * 60:
                        a.actual_impact_bps[k] = round((price / p0 - 1) * 1e4, 1)
                        if h == 30 and not a.is_duplicate:
                            self.stats[a.event_type].append(a.actual_impact_bps[k])
                if len(a.actual_impact_bps) < len(self.HORIZONS):
                    keep.append((a, s, t0, p0))
            else:
                keep.append((a, s, t0, p0))
        self.pending = keep
