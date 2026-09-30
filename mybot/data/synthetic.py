"""Synthetic market generator (demo + tests).

Produces a regime-switching multi-asset market with a market factor, sector
factors, idiosyncratic noise, news-driven jumps, U-shaped intraday volume,
quotes with spread dynamics, prints, and a macro calendar.

WARNING: strategy results on synthetic data are NOT evidence of edge. This exists
so the full pipeline and dashboard run end-to-end with zero credentials.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

from ..core.events import Topic
from ..core.types import Bar, NewsItem, Quote, ScheduledEvent, Trade

BASE_PX = {"AAPL": 190, "MSFT": 420, "NVDA": 120, "AMD": 160, "JPM": 200, "XOM": 110, "TSLA": 240,
           "META": 500, "SPY": 550}
BETA = {"AAPL": 1.1, "MSFT": 1.0, "NVDA": 1.8, "AMD": 1.7, "JPM": 1.0, "XOM": 0.6, "TSLA": 2.0, "META": 1.3, "SPY": 1.0}
POS_NEWS = ["{s} beats estimates on strong demand, raises guidance", "{s} wins contract worth $2B",
            "Analyst upgrade: {s} upgraded to outperform", "{s} announces strategic partnership with major cloud provider",
            "{s} record revenue tops estimates"]
NEG_NEWS = ["{s} cuts guidance citing weak demand", "{s} misses estimates as margins shrink", "{s} downgraded to underperform",
            "{s} faces antitrust probe", "{s} CFO resigns in abrupt departure"]
NEU_NEWS = ["{s} to present at industry conference", "{s} announces new product line", "{s} shares active in morning trade"]
SOURCES = ["reuters", "businesswire", "benzinga", "dow_jones", "blog"]
MACRO = [("CPI release", 3, "08:30"), ("FOMC rate decision", 3, "14:00"), ("Jobless claims", 2, "08:30"),
         ("ISM manufacturing", 2, "10:00"), ("Fed speaker", 1, "12:30")]


class SyntheticMarket:
    def __init__(self, symbols: list[str], proxy: str = "SPY", seed: int = 7, tz: str = "America/New_York"):
        self.symbols = list(symbols)
        self.proxy = proxy
        self.all = self.symbols + ([proxy] if proxy not in self.symbols else [])
        self.rng = np.random.default_rng(seed)
        self.tz = ZoneInfo(tz)
        self.px = {s: float(BASE_PX.get(s, 50 + 100 * self.rng.random())) for s in self.all}
        self.news_seq = 0

    def trading_days(self, start: datetime, n: int) -> list[datetime]:
        out, d = [], start
        while len(out) < n:
            if d.weekday() < 5:
                out.append(d)
            d += timedelta(days=1)
        return out

    def day_events(self, day: datetime) -> list[tuple[float, Topic, object]]:
        rng = self.rng
        tz = self.tz
        open_ts = datetime(day.year, day.month, day.day, 9, 30, tzinfo=tz).timestamp()
        steps_per_min, n_min = 4, 390
        n = steps_per_min * n_min
        dt = 1.0 / (n * 252)
        # Market regime for the day: segments with drift/vol.
        regime = rng.choice(["trend_up", "trend_down", "range", "high_vol"], p=[0.3, 0.25, 0.35, 0.1])
        mu = {"trend_up": 0.9, "trend_down": -0.9, "range": 0.0, "high_vol": -0.2}[regime]
        vol = {"trend_up": 0.14, "trend_down": 0.18, "range": 0.11, "high_vol": 0.35}[regime]
        # Drift is applied in bursts so trends look like trends (autocorrelated), not a constant slope.
        burst = np.convolve(rng.normal(0, 1, n), np.ones(40) / 40, mode="same")
        # ~1.1% daily drift in trend regimes, delivered in bursts.
        mkt = (mu * 0.012 / n) * (1 + 3 * burst) + vol * math.sqrt(dt) * rng.normal(0, 1, n)
        if regime == "range":
            mkt -= mkt.mean()
        sector_f: dict[str, np.ndarray] = {}
        events: list[tuple[float, Topic, object]] = []

        # Macro calendar (published at the start of the day; known in advance).
        cal_ts = open_ts - 3 * 3600
        for name, imp, hhmm in MACRO:
            if rng.random() < 0.35:
                hh, mm = map(int, hhmm.split(":"))
                ts = datetime(day.year, day.month, day.day, hh, mm, tzinfo=tz).timestamp()
                events.append((cal_ts, Topic.SCHEDULE, ScheduledEvent(name, ts, int(imp), ("*",), cal_ts)))

        for s in self.all:
            beta = BETA.get(s, 1.0)
            sec = s[:1]
            if sec not in sector_f:
                sector_f[sec] = 0.08 * math.sqrt(dt) * rng.normal(0, 1, n)
            idio_vol = 0.22 if s != self.proxy else 0.0
            r = beta * mkt + (sector_f[sec] if s != self.proxy else 0) + idio_vol * math.sqrt(dt) * rng.normal(0, 1, n)
            # Overnight gap.
            self.px[s] *= math.exp(rng.normal(0, 0.006 * beta))
            # News jumps with post-news drift.
            news_plan = []
            if s != self.proxy and rng.random() < 0.45:
                k = int(rng.integers(10, n - 60))
                kind = rng.choice(["pos", "neg", "neu"], p=[0.4, 0.4, 0.2])
                size = {"pos": 1, "neg": -1, "neu": 0}[kind] * abs(rng.normal(0.012, 0.006))
                r[k] += size
                r[k + 1:k + 120] += size / 400 * rng.uniform(0.5, 1.5)  # drift continuation (sometimes)
                news_plan.append((k, kind))
                if rng.random() < 0.5:  # a duplicate re-write from another source a bit later
                    news_plan.append((min(k + int(rng.integers(4, 40)), n - 1), kind + "_dup"))
            path = self.px[s] * np.exp(np.cumsum(r))
            # Volume (shares per minute), U-shaped.
            m = np.arange(n_min)
            shape = 1 + 2.2 * np.exp(-m / 25) + 1.6 * np.exp(-(n_min - m) / 25)
            base_v = 2.5e7 / self.px[s] / n_min * (3 if s == self.proxy else 1)
            vols = base_v * shape * rng.lognormal(0, 0.35, n_min)
            spread_bps_base = 1.5 if s == self.proxy else rng.uniform(1.5, 4.0)
            last_headline = {}
            for k, kind in news_plan:
                ts = open_ts + k * 15
                base_kind = kind.replace("_dup", "")
                if kind.endswith("_dup") and base_kind in last_headline:
                    h = "UPDATE: " + last_headline[base_kind]
                else:
                    pool = {"pos": POS_NEWS, "neg": NEG_NEWS, "neu": NEU_NEWS}[base_kind]
                    h = pool[int(rng.integers(len(pool)))].format(s=s)
                    last_headline[base_kind] = h
                    vols[k // steps_per_min: k // steps_per_min + 5] *= 4
                self.news_seq += 1
                src = SOURCES[int(rng.integers(len(SOURCES)))]
                events.append((ts + 2.0, Topic.NEWS, NewsItem(f"n{self.news_seq}", ts, h, (s,), src, "", "", ts + 2.0)))
            for i in range(n_min):
                seg = path[i * steps_per_min:(i + 1) * steps_per_min]
                o = path[i * steps_per_min - 1] if i > 0 else self.px[s]
                hi = max(o, seg.max()) * (1 + abs(rng.normal(0, 0.0003)))
                lo = min(o, seg.min()) * (1 - abs(rng.normal(0, 0.0003)))
                c = seg[-1]
                t0 = open_ts + i * 60
                v = float(int(vols[i]))
                events.append((t0 + 60.2, Topic.BAR, Bar(s, t0, t0 + 60, round(o, 4), round(hi, 4), round(lo, 4), round(c, 4), v,
                                                        "synthetic", t0 + 60.2)))
                for j in range(steps_per_min):
                    px = seg[j]
                    fut = seg[min(j + 1, steps_per_min - 1)] - px
                    sp = px * spread_bps_base * (1 + 3 * (vol > 0.3) * rng.random()) / 1e4
                    imb = np.tanh(fut / (px * 5e-4)) * 0.5 + rng.normal(0, 0.35)  # weakly informative book
                    depth = max(100.0, v / 20)
                    bs, as_ = depth * (1 + imb), depth * (1 - imb)
                    tq = t0 + j * 15 + 5
                    events.append((tq + 0.05, Topic.QUOTE, Quote(s, tq, round(px - sp / 2, 4), round(px + sp / 2, 4),
                                                                 max(100, round(bs, -2)), max(100, round(as_, -2)), "synthetic", tq + 0.05)))
                    side = 1 if (fut > 0) ^ (rng.random() < 0.4) else -1
                    tp = px + side * sp / 2
                    events.append((tq + 1.05, Topic.TRADE, Trade(s, tq + 1, round(tp, 4), float(max(100, int(v / 40))), "synthetic", tq + 1.05)))
            self.px[s] = float(path[-1])
        events.sort(key=lambda e: e[0])
        return events
