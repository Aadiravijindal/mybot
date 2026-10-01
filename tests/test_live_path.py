"""Regression tests for the live-trading path and the bugs found in deep validation."""
import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mybot.audit.journal import Journal
from mybot.backtest.engine import build_system, replay
from mybot.config import CostModel, RiskLimits, load_settings
from mybot.core.events import Topic
from mybot.core.types import Bar, Quote, Side
from mybot.data.csv_loader import csv_events
from mybot.execution.broker import Order, OrderStatus, OrderType
from mybot.execution.paper import PaperBroker
from mybot.risk.engine import PortfolioView, RiskEngine, RiskState
from mybot.signals.setups import Candidate

ET = ZoneInfo("America/New_York")


def _system(**strategy):
    S = load_settings()
    S = replace(S, symbols=("AAA",), market_proxy="SPY", strategy=replace(S.strategy, **strategy) if strategy else S.strategy)
    t0 = datetime(2025, 3, 3, 10, 0, tzinfo=ET).timestamp()
    s = build_system(S, None, t0)
    return s, t0


def _open_long(s, t0, broker_cls=None):
    """Drive an entry through the real orchestrator path with a quote stream."""
    q = Quote("AAA", t0, 99.99, 100.01, 500, 500, "t", t0)
    s.publish(Topic.QUOTE, q, "AAA", ts=t0)
    c = Candidate("AAA", "TEST", Side.LONG, t0, 100.0, 99.0, [101.5, 103.0], 60, {"spread_bps": 2.0}, "test")
    s.pending_entries.clear()
    o = Order("AAA", 1, 50, OrderType.MARKET, t0, "entry", decision_px=100.0)
    s.pending_entries[o.id] = {"c": c, "fc": {"p_win": 0.6, "ev_r": 0.2}, "ev": [], "did": "d-test"}
    s.broker.submit(o)
    t = t0 + 1
    s.clock.advance_to(t)
    s.publish(Topic.QUOTE, Quote("AAA", t, 99.99, 100.01, 500, 500, "t", t), "AAA", ts=t)
    return s.book.positions["AAA"]


def test_exit_cancels_resting_stop_first_and_never_flips():
    s, t0 = _system()
    p = _open_long(s, t0)
    stop_id = p.stop_order_id
    assert s.broker.orders[stop_id].status is OrderStatus.WORKING
    s._exit(p, p.qty, "test exit", s.clock.now())
    assert s.broker.orders[stop_id].status is OrderStatus.CANCELED, "stop must be cancelled before the exit is sent"
    # A quote that would have triggered the old stop AND fills the exit.
    t = s.clock.now() + 1
    s.clock.advance_to(t)
    s.publish(Topic.QUOTE, Quote("AAA", t, 98.0, 98.02, 500, 500, "t", t), "AAA", ts=t)
    assert "AAA" not in s.book.positions
    assert s.broker.positions() == {}, "position must be flat, not flipped short"


def test_rejected_stop_closes_the_position():
    class NoStops(PaperBroker):
        def submit(self, o):
            if o.type is OrderType.STOP:
                o.status, o.reject_reason = OrderStatus.REJECTED, "stops not allowed"
                if self.on_order:
                    self.on_order(o)
                return o
            return super().submit(o)

    s, t0 = _system()
    s.broker = NoStops(s.S.costs)
    s.broker.on_fill, s.broker.on_order = s._on_fill, s._on_order
    p = _open_long(s, t0)
    assert p.stop_order_id is None and p.working_exit, "a rejected stop must trigger an immediate exit"
    t = s.clock.now() + 1
    s.clock.advance_to(t)
    s.publish(Topic.QUOTE, Quote("AAA", t, 99.99, 100.01, 500, 500, "t", t), "AAA", ts=t)
    assert "AAA" not in s.book.positions and s.broker.positions() == {}
    assert any(r.get("event") == "stop_lost" for r in s.J.recent["risk"])


def test_order_ids_are_unique_across_runs():
    from mybot.execution import broker as b
    o = Order("A", 1, 1, OrderType.MARKET, 0, "entry")
    assert o.id.startswith(b.ORDER_PREFIX + b.RUN_ID + "-")


def test_bars_only_paper_fills_and_same_bar_stop():
    b = PaperBroker(CostModel(latency_s=0.1, slippage_bps=0))
    fills = []
    b.on_fill = fills.append
    b.on_bar(Bar("A", 0, 60, 100, 100.5, 99.5, 100, 1e5, "t", 60))
    o = b.submit(Order("A", 1, 10, OrderType.MARKET, 60, "entry"))
    assert o.status is OrderStatus.WORKING, "bars-only data must not reject market orders"

    def protect(f):
        fills.append(f)
        if f.tag == "entry":
            b.submit(Order("A", -1, 10, OrderType.STOP, 120, "stop", stop=99.0))
    b.on_fill = protect
    b.on_bar(Bar("A", 60, 120, 100.2, 100.3, 98.5, 98.7, 1e5, "t", 120))   # entry at open, then crashes through 99
    assert [f.tag for f in fills] == ["entry", "stop"], "a stop placed after an open fill must be tested on the same bar"
    assert fills[-1].price <= 99.0


def test_csv_loader_infers_bar_size_and_is_point_in_time(tmp_path):
    f = tmp_path / "spy.csv"
    f.write_text("Datetime,Open,High,Low,Close,Volume\n"
                 "2025-04-15 09:30:00-04:00,1,2,0.5,1.5,100\n"
                 "2025-04-15 09:35:00-04:00,1.5,2,1,1.8,100\n"
                 "garbage,row,x,x,x,x\n"
                 "2025-04-15 09:40:00-04:00,1.8,2,1.5,1.9,100\n")
    events, bar_s = csv_events([f"{f}:SPY"])
    assert bar_s == 300 and len(events) == 3
    ts, topic, b = events[0]
    assert topic is Topic.BAR and b.symbol == "SPY" and ts == b.end_ts == b.ts + 300


def test_knowledge_base_partitions_never_mix(tmp_path):
    from mybot.signals.analogues import Outcome
    import numpy as np
    j = Journal(str(tmp_path))
    o = Outcome("ORB", 1, 0, 1, 0.5, -0.2, 0.8, 3, "target", np.zeros(3), "RANGE")
    j.outcome(o, "synthetic")
    j.outcome(o, "real")
    j.outcome(o, "real")
    assert len(j.load_outcomes("synthetic")) == 1 and len(j.load_outcomes("real")) == 2


def test_pdt_limit_blocks_entries():
    r = RiskEngine(RiskLimits(), {})
    r.pdt_remaining = 1
    pv = PortfolioView(20_000, 20_000, 20_000, {}, {})
    ok = r.pre_trade(sym="A", side_sign=1, qty=10, price=100, stop=99, spread_bps=1, avg_dollar_vol=5e8, pv=pv, ts=0)
    assert ok.approved
    r.on_order_sent(0, opening=True)
    no = r.pre_trade(sym="A", side_sign=1, qty=10, price=100, stop=99, spread_bps=1, avg_dollar_vol=5e8, pv=pv, ts=1)
    assert not no.approved and "pattern-day-trader" in no.reasons[0]


def test_warmup_replay_never_trades():
    s, _ = _system(min_analogues=0, min_ev_r=-9.0, min_prob_lcb=0.0, min_reward_risk=0.0)
    from mybot.data.synthetic import SyntheticMarket
    m = SyntheticMarket(["AAA"], "SPY", seed=4)
    ev = m.day_events(m.trading_days(datetime(2025, 3, 3), 1)[0])
    s2 = build_system(s.S, None, ev[0][0] - 1)
    s2.allow_entries = False
    replay(s2, ev)
    assert not s2.book.closed and not s2.book.positions and not s2.broker.orders


def test_live_mode_requires_confirmation(monkeypatch):
    from mybot.runtime import Runtime
    S = replace(load_settings(), mode="live", data_source="alpaca")
    monkeypatch.delenv("MYBOT_LIVE_CONFIRM", raising=False)
    with pytest.raises(SystemExit, match="Refusing LIVE"):
        Runtime(S)
    S2 = replace(load_settings(), mode="live", data_source="synthetic")
    with pytest.raises(SystemExit, match="real market data"):
        Runtime(S2)


def test_test_clock_refused_against_real_broker(monkeypatch):
    from mybot.core.clock import live_clock
    monkeypatch.setenv("MYBOT_TEST_CLOCK", "1,2")
    monkeypatch.setenv("ALPACA_TRADING_URL", "https://api.alpaca.markets")
    with pytest.raises(SystemExit):
        live_clock()


# ------------------------------------------------------- Alpaca adapter vs fake Alpaca
@pytest.fixture()
def fake_broker(monkeypatch):
    from fake_alpaca import FakeAlpaca, create_app
    from mybot.execution.alpaca import AlpacaBroker
    monkeypatch.setenv("ALPACA_API_KEY_ID", "k")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "s")
    day = datetime(2025, 3, 3, tzinfo=ET)
    f = FakeAlpaca(["AAA"], "SPY", day, day.replace(hour=10).timestamp(), 1.0, history_days=1)
    f.last_q["AAA"] = (99.99, 100.01)
    client = TestClient(create_app(f))
    return f, AlpacaBroker(paper=True, http=client)


def test_alpaca_preflight_and_reservation_semantics(fake_broker):
    f, br = fake_broker
    rep = br.preflight()
    assert rep["equity"] == 50_000 and not rep["positions"] and not rep["pdt_limited"]
    fills = []
    br.on_fill = fills.append
    e = br.submit(Order("AAA", 1, 10, OrderType.MARKET, 0, "entry"))
    br.poll_once(1)
    assert fills and fills[0].qty == 10 and br.positions() == {"AAA": 10}
    st = br.submit(Order("AAA", -1, 10, OrderType.STOP, 1, "stop", stop=98.0))
    assert st.status is OrderStatus.WORKING
    # The real broker refuses a sell while the stop reserves the shares...
    blocked = br.submit(Order("AAA", -1, 10, OrderType.MARKET, 2, "exit"))
    assert blocked.status is OrderStatus.REJECTED and "insufficient qty" in blocked.reject_reason
    # ...which is why the engine cancels (and confirms) first.
    assert br.cancel(st.id) is True
    ok = br.submit(Order("AAA", -1, 10, OrderType.MARKET, 3, "exit"))
    br.poll_once(4)
    assert ok.status is OrderStatus.FILLED and br.positions() == {}


def test_alpaca_replace_tracks_new_id_and_outage_disconnects(fake_broker):
    f, br = fake_broker
    br.preflight()
    br.submit(Order("AAA", 1, 5, OrderType.MARKET, 0, "entry"))
    st = br.submit(Order("AAA", -1, 5, OrderType.STOP, 1, "stop", stop=98.0))
    old = br.remote[st.id]
    br.replace_stop(st.id, 98.5)
    assert br.remote[st.id] != old and f.orders[old]["status"] == "replaced"
    f.chaos["rest_down"] = True
    br.refresh_positions()
    br.refresh_positions()
    assert br.connected is False
    f.chaos["rest_down"] = False
    br.refresh_positions()
    assert br.connected is True


def test_alpaca_preflight_cancels_only_our_stale_orders(fake_broker):
    f, br = fake_broker
    f.orders["x1"] = {"id": "x1", "client_order_id": "mbdead-1", "symbol": "AAA", "qty": "1", "side": "buy", "type": "limit",
                      "status": "accepted", "filled_qty": "0"}
    f.orders["x2"] = {"id": "x2", "client_order_id": "manual-order", "symbol": "AAA", "qty": "1", "side": "buy", "type": "limit",
                      "status": "accepted", "filled_qty": "0"}
    rep = br.preflight()
    assert rep["stale_orders_cancelled"] == ["AAA"]
    assert f.orders["x1"]["status"] == "pending_cancel" and f.orders["x2"]["status"] == "accepted"


def test_alpaca_message_parsing():
    from mybot.data.alpaca import AlpacaData

    class Sys:
        def __init__(self):
            self.got = []
            self.clock = type("C", (), {"now": staticmethod(lambda: 1e10)})()

        def publish(self, topic, payload, sym, ts):
            self.got.append((topic, payload))
    d = AlpacaData.__new__(AlpacaData)
    d.feed, d.symbols, d.clock_offset_s = "iex", ["AAA"], 0.0
    s = Sys()
    d._dispatch(s, {"T": "b", "S": "AAA", "o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 10, "t": "2025-03-03T15:00:00Z"}, 1e10)
    d._dispatch(s, {"T": "q", "S": "AAA", "bp": 1, "ap": 1.01, "bs": 3, "as": 4, "t": "2025-03-03T15:00:00.123456789Z"}, 1e10)
    d._dispatch(s, {"T": "n", "id": 5, "headline": "h", "symbols": ["AAA", "ZZZ"], "created_at": "2025-03-03T15:00:00Z"}, 1e10)
    d._dispatch(s, {"T": "n", "id": 6, "headline": "h", "symbols": ["ZZZ"], "created_at": "2025-03-03T15:00:00Z"}, 1e10)
    topics = [t for t, _ in s.got]
    assert topics == [Topic.BAR, Topic.QUOTE, Topic.NEWS]
    bar = s.got[0][1]
    assert bar.end_ts - bar.ts == 60
    with pytest.raises(KeyError):
        d._dispatch(s, {"T": "b", "S": "AAA"}, 1e10)   # malformed: caught and skipped by the stream loop
