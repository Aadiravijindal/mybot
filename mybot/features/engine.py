"""Feature engineering: per-symbol, multi-timeframe, causal.

Feature families (see docs/ARCHITECTURE.md section G):
  trend, momentum, volatility, volume, structure, microstructure, relative strength.
Features are *candidates*. Which ones earn a place in the model is decided by the
research pipeline (ablation + walk-forward), not by this file.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from ..core.calendar import Session
from ..core.types import Bar, Quote, Trade
from .indicators import ADX, ATR, EMA, MACD, RSI, EfficiencyRatio, Rolling, RobustZ, SessionVWAP, SwingDetector


class BarAggregator:
    """Aggregate closed 1m bars into N-minute bars aligned to the session."""

    def __init__(self, minutes: int):
        self.m = minutes
        self.cur: list[Bar] = []

    def update(self, b: Bar, minute_of_session: float) -> Bar | None:
        self.cur.append(b)
        if int(round(minute_of_session)) % self.m == 0:  # b.end_ts closes the bucket
            bs, self.cur = self.cur, []
            return Bar(b.symbol, bs[0].ts, b.end_ts, bs[0].open, max(x.high for x in bs),
                       min(x.low for x in bs), b.close, sum(x.volume for x in bs), b.source, b.avail_ts, 60 * self.m)
        return None


@dataclass
class Micro:
    """Microstructure state from quotes/trades."""
    last_quote: Quote | None = None
    ofi_window: deque = field(default_factory=lambda: deque(maxlen=200))
    depth: Rolling = field(default_factory=lambda: Rolling(200))
    spread_bps: Rolling = field(default_factory=lambda: Rolling(300))
    signed_vol: deque = field(default_factory=lambda: deque(maxlen=500))
    last_trade_px: float | None = None
    last_sign: int = 0
    trade_count: int = 0

    def on_quote(self, q: Quote) -> None:
        p = self.last_quote
        if p is not None:
            # Order-flow imbalance (Cont, Kukanov & Stoikov 2014), best-level version.
            e_b = (q.bid_size if q.bid >= p.bid else 0) - (p.bid_size if q.bid <= p.bid else 0)
            e_a = (q.ask_size if q.ask <= p.ask else 0) - (p.ask_size if q.ask >= p.ask else 0)
            self.ofi_window.append(e_b - e_a)
        self.depth.update((q.bid_size + q.ask_size) / 2)
        self.spread_bps.update(q.spread_bps)
        self.last_quote = q

    def on_trade(self, t: Trade) -> None:
        # Lee-Ready: quote rule, falling back to tick rule.
        q, sign = self.last_quote, 0
        if q is not None and q.ask > q.bid:
            if t.price >= q.ask - 1e-9 or t.price > q.mid:
                sign = 1
            elif t.price <= q.bid + 1e-9 or t.price < q.mid:
                sign = -1
        if sign == 0 and self.last_trade_px is not None:
            sign = 1 if t.price > self.last_trade_px else -1 if t.price < self.last_trade_px else self.last_sign
        self.last_sign, self.last_trade_px = sign, t.price
        self.signed_vol.append(sign * t.size)
        self.trade_count += 1

    def snapshot(self) -> dict[str, float]:
        out: dict[str, float] = {}
        q = self.last_quote
        if q is None:
            return out
        tot = q.bid_size + q.ask_size
        out["quote_imbalance"] = (q.bid_size - q.ask_size) / tot if tot > 0 else 0.0
        out["microprice_offset_bps"] = (((q.bid * q.ask_size + q.ask * q.bid_size) / tot - q.mid) / q.mid * 1e4) if tot > 0 else 0.0
        out["spread_bps"] = q.spread_bps
        sd = self.spread_bps.std
        out["spread_z"] = (q.spread_bps - self.spread_bps.mean) / sd if sd and sd > 0 else 0.0
        d = self.depth.mean or 1.0
        out["ofi_norm"] = sum(self.ofi_window) / (d * max(len(self.ofi_window), 1) ** 0.5)
        tv = sum(abs(x) for x in self.signed_vol)
        out["trade_imbalance"] = sum(self.signed_vol) / tv if tv > 0 else 0.0
        return out


class SymbolFeatures:
    def __init__(self, symbol: str, session: Session, bar_seconds: int = 60):
        self.symbol, self.session = symbol, session
        self.bars_per_day = session.length_min * 60 / bar_seconds
        self.bars: deque[Bar] = deque(maxlen=400)
        self.n = 0
        self.day: str | None = None
        # 1m
        self.ema9, self.ema20, self.ema50 = EMA(9), EMA(20), EMA(50)
        self.rsi, self.atr, self.adx, self.macd = RSI(14), ATR(14), ADX(14), MACD()
        self.er = EfficiencyRatio(20)
        self.bb = Rolling(20)
        self.rv = Rolling(30)            # realized vol (1m log returns)
        self.rv_hist = Rolling(390)      # distribution of rv for percentile
        self.vol_avg = Rolling(60)
        self.vol_z = RobustZ(120)
        self.ret_z = RobustZ(120)
        self.vwap = SessionVWAP()
        self.swing = SwingDetector(3)
        self.donch = deque(maxlen=20)
        self.obv = 0.0
        self.obv_ema = EMA(20)
        # opening range
        self.or_high = self.or_low = None
        self.day_open = self.prev_close = None
        self.day_high = -math.inf
        self.day_low = math.inf
        # higher timeframes
        self.agg5, self.agg15 = BarAggregator(5), BarAggregator(15)
        self.ema20_15, self.ema50_15 = EMA(20), EMA(50)
        self.last15: Bar | None = None
        self.rsi5 = RSI(14)
        self.micro = Micro()
        self.values: dict[str, float] = {}
        self.last_bar_ts = 0.0

    # ------------------------------------------------------------------ bars
    def on_bar(self, b: Bar) -> dict[str, float]:
        day = self.session.day_key(b.ts)
        mso = self.session.minutes_since_open(b.end_ts)
        if day != self.day:
            if self.bars:
                self.prev_close = self.bars[-1].close
            self.day, self.vwap = day, SessionVWAP()
            self.or_high = self.or_low = None
            self.day_open, self.day_high, self.day_low = b.open, -math.inf, math.inf
            self.agg5.cur.clear(); self.agg15.cur.clear()
        prev = self.bars[-1].close if self.bars else None
        self.bars.append(b)
        self.n += 1
        self.last_bar_ts = b.end_ts
        c = b.close
        self.day_high, self.day_low = max(self.day_high, b.high), min(self.day_low, b.low)
        if mso <= 15:
            self.or_high = b.high if self.or_high is None else max(self.or_high, b.high)
            self.or_low = b.low if self.or_low is None else min(self.or_low, b.low)

        e9, e20, e50 = self.ema9.update(c), self.ema20.update(c), self.ema50.update(c)
        rsi = self.rsi.update(c)
        atr = self.atr.update(b.high, b.low, c)
        adx = self.adx.update(b.high, b.low, c)
        self.macd.update(c)
        er = self.er.update(c)
        self.bb.update(c)
        r = math.log(c / prev) if prev else 0.0
        self.rv.update(r)
        rv = self.rv.std if self.rv.ready else None
        if rv is not None:
            self.rv_hist.update(rv)
        vz = self.vol_z.update(b.volume)
        rz = self.ret_z.update(r)
        self.vol_avg.update(b.volume)
        vw = self.vwap.update(b.typical, b.volume)
        self.swing.update(b.high, b.low)
        if prev is not None:
            self.obv += b.volume if c > prev else -b.volume if c < prev else 0.0
        self.obv_ema.update(self.obv)
        dh = max(x.high for x in self.donch) if self.donch else None
        dl = min(x.low for x in self.donch) if self.donch else None
        self.donch.append(b)

        b5 = self.agg5.update(b, mso)
        if b5:
            self.rsi5.update(b5.close)
        b15 = self.agg15.update(b, mso)
        if b15:
            self.ema20_15.update(b15.close)
            self.ema50_15.update(b15.close)
            self.last15 = b15

        v: dict[str, float] = {"close": c, "ret_1m": r, "ret_z": rz, "vol_z": vz, "minute": mso}
        if atr:
            v["atr"] = atr
            v["atr_pct"] = atr / c * 100
            v["ema_stack"] = (1.0 if e9 > e20 > e50 else -1.0 if e9 < e20 < e50 else 0.0)
            v["dist_ema20_atr"] = (c - e20) / atr
            if vw:
                v["vwap"] = vw
                v["dist_vwap_atr"] = (c - vw) / atr
            v["ema20_slope_atr"] = (e20 - (self._e20_prev if hasattr(self, "_e20_prev") else e20)) / atr * 5
            if dh is not None:
                v["donchian_pos"] = (c - dl) / (dh - dl) if dh > dl else 0.5
                v["breakout_up"] = 1.0 if c > dh else 0.0
                v["breakout_dn"] = 1.0 if c < dl else 0.0
        self._e20_prev = e20
        if rsi is not None:
            v["rsi"] = rsi
        if self.rsi5.value is not None:
            v["rsi_5m"] = self.rsi5.value
        if adx is not None:
            v["adx"] = adx
            v["di_diff"] = (self.adx.pdi or 0) - (self.adx.ndi or 0)
        if er is not None:
            v["eff_ratio"] = er
        if self.macd.hist is not None and atr:
            v["macd_hist_atr"] = self.macd.hist / atr
        if self.bb.ready and self.bb.std > 0:
            v["bb_z"] = (c - self.bb.mean) / self.bb.std
            v["bb_width_pct"] = 4 * self.bb.std / c * 100
        if rv is not None:
            v["rv_1m"] = rv
            v["rv_pct"] = self.rv_hist.percentile_rank(rv) if len(self.rv_hist.buf) >= 60 else 0.5
        if self.vol_avg.ready and self.vol_avg.mean > 0:
            v["rel_volume"] = b.volume / self.vol_avg.mean
            v["avg_dollar_vol_day"] = self.vol_avg.mean * c * self.bars_per_day
        if self.obv_ema.value is not None:
            v["obv_slope"] = 1.0 if self.obv > self.obv_ema.value else -1.0
        if self.or_high is not None and mso > 15:
            v["or_high"], v["or_low"] = self.or_high, self.or_low
        if self.day_open:
            v["ret_day"] = c / self.day_open - 1
        if self.prev_close:
            v["gap_pct"] = (self.day_open / self.prev_close - 1) * 100 if self.day_open else 0.0
        if self.swing.last_low is not None:
            v["swing_low"] = self.swing.last_low
        if self.swing.last_high is not None:
            v["swing_high"] = self.swing.last_high
        if self.ema50_15.ready and self.last15 is not None:
            v["htf_trend"] = 1.0 if (self.ema20_15.value > self.ema50_15.value and self.last15.close > self.ema20_15.value) \
                else -1.0 if (self.ema20_15.value < self.ema50_15.value and self.last15.close < self.ema20_15.value) else 0.0
        v.update(self.micro.snapshot())
        self.values = v
        return v

    def on_quote(self, q: Quote) -> None:
        self.micro.on_quote(q)

    def on_trade(self, t: Trade) -> None:
        self.micro.on_trade(t)

    @property
    def warm(self) -> bool:
        return self.n >= 60 and "atr" in self.values
