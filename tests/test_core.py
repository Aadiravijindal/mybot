import math
import random

import numpy as np
import pytest

from mybot.config import CostModel, RiskLimits
from mybot.core.clock import LookAheadError, SimClock
from mybot.core.events import Event, EventBus, Topic
from mybot.core.types import Bar, NewsItem, Quote
from mybot.data.quality import DataQualityEngine
from mybot.execution.broker import Order, OrderStatus, OrderType
from mybot.execution.paper import PaperBroker
from mybot.features.indicators import ATR, EMA, RSI, SwingDetector
from mybot.intel.hmm import GaussianHMM
from mybot.intel.news import NewsIntel
from mybot.learning.research import IsotonicCalibrator, ModelArtifact, ModelRegistry
from mybot.risk.engine import PortfolioView, RiskEngine, RiskState
from mybot.risk.sizing import size
from mybot.risk.stops import DynamicStopEngine
from mybot.risk.thesis import Thesis, ThesisState


# ------------------------------------------------------------ point-in-time
def test_bus_rejects_future_information():
    clock = SimClock(1000.0)
    bus = EventBus(clock)
    seen = []
    bus.subscribe(Topic.NEWS, seen.append)
    bus.publish(Event(Topic.NEWS, 1000.0, "ok"))
    with pytest.raises(LookAheadError):
        bus.publish(Event(Topic.NEWS, 1000.5, "from the future"))
    assert len(seen) == 1


def test_news_at_1403_invisible_at_1402():
    """The canonical example: a 14:03 story cannot influence a 14:02 decision."""
    from mybot.intel.catalysts import CatalystBook
    from mybot.intel.news import NewsIntel
    book, ni = CatalystBook(), NewsIntel()
    t1403 = 50_000.0
    a = ni.assess(NewsItem("x", t1403, "ACME beats estimates, raises guidance", ("ACME",), "reuters", avail_ts=t1403))
    book.add(a)
    score_before, items_before = book.score("ACME", t1403 - 60)
    score_after, _ = book.score("ACME", t1403 + 60)
    assert score_before == 0 and items_before == []
    assert score_after > 0


def test_handler_error_is_reported_not_swallowed():
    clock = SimClock(0)
    bus = EventBus(clock)
    errs = []
    bus.on_error = lambda ev, exc: errs.append(exc)
    bus.subscribe(Topic.BAR, lambda ev: 1 / 0)
    bus.publish(Event(Topic.BAR, 0, None))
    assert bus.errors == 1 and len(errs) == 1


# ------------------------------------------------------------ indicators
def test_ema_matches_reference():
    xs = [random.Random(1).uniform(90, 110) for _ in range(50)]
    e = EMA(10)
    ref = None
    for x in xs:
        v = e.update(x)
        ref = x if ref is None else ref + 2 / 11 * (x - ref)
        assert v == pytest.approx(ref)


def test_rsi_bounds_and_monotone_series():
    r = RSI(14)
    for i in range(40):
        v = r.update(100 + i)
    assert v == pytest.approx(100.0)
    r = RSI(14)
    rng = random.Random(3)
    for _ in range(200):
        v = r.update(100 + rng.gauss(0, 1))
        if v is not None:
            assert 0 <= v <= 100


def test_atr_constant_range():
    a = ATR(5)
    for _ in range(20):
        v = a.update(101, 99, 100)
    assert v == pytest.approx(2.0)


def test_swing_confirmed_with_lag_only():
    s = SwingDetector(k=2)
    highs = [1, 2, 5, 2, 1]
    for i, h in enumerate(highs[:-1]):
        s.update(h, h - 0.5)
        assert s.last_high is None  # cannot know it is a swing high until k bars after
    s.update(highs[-1], highs[-1] - 0.5)
    assert s.last_high == 5


# ------------------------------------------------------------ data quality
def test_data_quality_rejections():
    dq = DataQualityEngine()
    good = Bar("A", 0, 60, 10, 11, 9, 10.5, 1000, "t", 60.1)
    assert dq.check_bar(good, 61)
    assert not dq.check_bar(good, 62), "duplicate bar must be rejected"
    bad = Bar("A", 60, 120, 10, 9, 11, 10, 100, "t", 120.1)   # high < low
    assert not dq.check_bar(bad, 121)
    early = Bar("A", 120, 180, 10, 11, 9, 10, 100, "t", 170)  # available before close
    assert not dq.check_bar(early, 181)
    assert not dq.check_quote(Quote("A", 1, 10.1, 10.0, 100, 100), 1)  # crossed
    q = dq.quality("A", 61 + 10_000)
    assert not q.ok and any("stale" in i for i in q.issues)


# ------------------------------------------------------------ stops / thesis / sizing
def test_stop_only_ratchets_toward_less_risk():
    eng = DynamicStopEngine(atr_mult=1.5, trail_mult=2.5, breakeven_at_r=1.0, emergency_pct=3.0)
    st = eng.init(+1, 100.0, 98.5, 0)
    prev = st.current
    rng = random.Random(0)
    px = 100.0
    for i in range(300):
        px += rng.gauss(0, 0.3)
        eng.update(st, high=px + 0.1, low=px - 0.1, close=px, atr=1.0, swing_low=px - rng.uniform(0.5, 3),
                   swing_high=None, thesis="VALID", cost_per_share=0.02, spread=0.02, ts=i)
        assert st.current >= prev - 1e-12, "long stop must never move down"
        prev = st.current


def test_emergency_stop_caps_loss():
    eng = DynamicStopEngine(1.5, 2.5, 1.0, emergency_pct=2.0)
    st = eng.init(+1, 100.0, 90.0, 0)      # absurdly wide proposed stop
    assert st.current == pytest.approx(98.0)


def test_thesis_invalidation_is_immediate():
    th = Thesis("x", +1, "TREND_PULLBACK", invalidation=99.0)
    state, changed = th.update({"close": 98.9}, market_regime="TREND_UP", catalyst=0, bars_held=1, horizon=60, r_now=-0.5, ts=1)
    assert state is ThesisState.INVALID and changed


def test_kelly_never_increases_base_risk():
    for p in np.linspace(0.3, 0.95, 14):
        qty, risk, info = size(100_000, 0.25, 100, 99, p, 3.0, 1.0)
        assert risk <= 100_000 * 0.0025 + 1e-9


# ------------------------------------------------------------ risk engine
def _pv(eq=100_000, pos=None):
    return PortfolioView(eq, 100_000, 100_000, pos or {}, {})


def test_risk_caps_and_blocks():
    r = RiskEngine(RiskLimits(), {"A": "TECH"})
    d = r.pre_trade(sym="A", side_sign=1, qty=10_000, price=100, stop=99, spread_bps=2, avg_dollar_vol=5e8, pv=_pv(), ts=0)
    assert d.approved and d.qty == 200  # 0.25% of 100k / $1 risk-per-share = 250, notional cap 20% -> 200
    bad = r.pre_trade(sym="A", side_sign=1, qty=10, price=100, stop=101, spread_bps=2, avg_dollar_vol=5e8, pv=_pv(), ts=0)
    assert not bad.approved and "invalid stop" in bad.reasons[0]
    wide = r.pre_trade(sym="A", side_sign=1, qty=10, price=100, stop=99, spread_bps=80, avg_dollar_vol=5e8, pv=_pv(), ts=0)
    assert not wide.approved


def test_daily_loss_halts_and_requires_manual_resume():
    r = RiskEngine(RiskLimits(max_daily_loss_pct=1.0), {})
    r.check_portfolio(_pv(eq=98_900), 1)
    assert r.state is RiskState.HALTED
    r.new_day("2026-01-02", 2)
    assert r.state is RiskState.HALTED, "a new day must not silently clear a HALT"
    d = r.pre_trade(sym="A", side_sign=1, qty=1, price=100, stop=99, spread_bps=1, avg_dollar_vol=5e8, pv=_pv(), ts=3)
    assert not d.approved
    r.manual_resume("tester", 4)
    assert r.state is RiskState.NORMAL


def test_infrastructure_failures_halt():
    for kw in ({"data_staleness_s": 999}, {"clock_drift_s": 5}, {"broker_ok": False}, {"reconciled": False}):
        r = RiskEngine(RiskLimits(), {})
        args = dict(data_staleness_s=0, clock_drift_s=0, broker_ok=True, reconciled=True, market_open=True, ts=0)
        args.update(kw)
        r.check_infrastructure(**args)
        assert r.state is RiskState.HALTED, kw


# ------------------------------------------------------------ paper broker
def test_paper_fills_cross_spread_after_latency_and_stop_gaps():
    b = PaperBroker(CostModel(latency_s=0.5, slippage_bps=0))
    fills = []
    b.on_fill = fills.append
    b.on_quote(Quote("A", 0, 99.9, 100.1, 500, 500, avail_ts=0))
    o = b.submit(Order("A", 1, 100, OrderType.MARKET, 0.0, "entry"))
    b.on_quote(Quote("A", 0.2, 99.9, 100.1, 500, 500, avail_ts=0.2))
    assert not fills, "must not fill before latency elapses"
    b.on_quote(Quote("A", 0.6, 99.9, 100.1, 500, 500, avail_ts=0.6))
    assert fills and fills[0].price >= 100.1 and o.status is OrderStatus.FILLED
    s = b.submit(Order("A", -1, 100, OrderType.STOP, 1.0, "stop", stop=99.0))
    b.on_bar(Bar("A", 60, 120, 97.0, 97.5, 96.5, 97.2, 1e5, "t", 120.1))  # gaps through the stop
    assert s.status is OrderStatus.FILLED and fills[-1].price < 99.0, "gap fills at the worse open, not the stop"


def test_rejects_when_disconnected():
    b = PaperBroker(CostModel())
    b.connected = False
    o = b.submit(Order("A", 1, 1, OrderType.MARKET, 0, "entry"))
    assert o.status is OrderStatus.REJECTED


# ------------------------------------------------------------ news
def test_news_classification_and_dedup():
    ni = NewsIntel()
    a = ni.assess(NewsItem("1", 0, "ACME cuts guidance citing weak demand", ("ACME",), "reuters", avail_ts=0))
    assert a.direction == -1 and a.event_type == "GUIDANCE_DOWN" and not a.is_duplicate
    b = ni.assess(NewsItem("2", 5, "UPDATE: ACME cuts guidance citing weak demand", ("ACME",), "benzinga", avail_ts=5))
    assert b.is_duplicate and b.materiality < a.materiality


# ------------------------------------------------------------ models
def test_hmm_filter_is_causal():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(0, 0.5, 300), rng.normal(0, 3, 300)]
    h = GaussianHMM(2, iters=20).fit(x)
    full = h.filter(x)
    part = h.filter(x[:350])
    assert np.allclose(full[:350], part), "filtered probabilities must not depend on future data"
    assert full[500, 1] > 0.8


def test_isotonic_is_monotone_and_bounded():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 500)
    y = (rng.uniform(0, 1, 500) < p * 0.6).astype(float)
    c = IsotonicCalibrator().fit(p, y)
    vals = [c(v) for v in np.linspace(0, 1, 50)]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))
    assert 0.05 <= min(vals) and max(vals) <= 0.95


def test_model_promotion_is_gated(tmp_path):
    reg = ModelRegistry(str(tmp_path))
    m = ModelArtifact("m", "1")
    with pytest.raises(PermissionError):
        reg.promote(m, "PAPER", {"min_oos_trades": 30}, "me")
    assert reg.load_active("paper") is None
