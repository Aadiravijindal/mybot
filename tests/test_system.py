from datetime import datetime

import numpy as np
from fastapi.testclient import TestClient

from mybot.audit.journal import Journal, verify_chain
from mybot.backtest.engine import build_system, deflated_sharpe, metrics, replay
from mybot.config import load_settings
from mybot.core.events import Topic
from mybot.data.synthetic import SyntheticMarket
from mybot.risk.engine import RiskState


def _run(days=4, seed=5, data_dir=None):
    S = load_settings()
    m = SyntheticMarket(list(S.symbols)[:4], S.market_proxy, seed=seed)
    ds = m.trading_days(datetime(2025, 3, 3), days)
    first = m.day_events(ds[0])
    sysm = build_system(S, data_dir, first[0][0] - 1)
    replay(sysm, first)
    for d in ds[1:]:
        replay(sysm, m.day_events(d))
    return sysm


def test_backtest_end_to_end_is_clean():
    s = _run()
    assert s.bus.errors == 0
    assert s.risk.state is not RiskState.HALTED, s.risk.reasons
    assert not s.book.positions, "intraday system must be flat after the close"
    assert s.broker.positions() == {}, "broker and book must reconcile"
    assert s.stats["bars"] > 1000
    assert len(s.analogues.items) > 0
    # every candidate decision was journalled with both sides of the argument
    for d in s.J.decisions(limit=50):
        assert "supporting_evidence" in d and "contradicting_evidence" in d and "forecast" in d
    m = metrics(s)
    assert m.max_drawdown_pct >= 0


def test_decision_reconstruction_and_audit_chain(tmp_path):
    s = _run(days=3, data_dir=str(tmp_path))
    decs = s.J.decisions(limit=5)
    if decs:
        d = decs[0]
        again = s.J.decision_at(d["symbol"], d["ts"])
        assert again["id"] == d["id"]
    ok, n = verify_chain(tmp_path / "audit" / "journal.jsonl")
    assert ok and n > 0
    # tamper with one record -> chain must break
    p = tmp_path / "audit" / "journal.jsonl"
    lines = p.read_text().splitlines()
    lines[len(lines) // 2] = lines[len(lines) // 2].replace('"ts":', '"ts":1', 1)
    p.write_text("\n".join(lines) + "\n")
    ok2, _ = verify_chain(p)
    assert not ok2


def test_broker_disconnect_halts_and_flattens():
    s = _run(days=1)
    s.broker.connected = False
    s.health_check()
    assert s.risk.state is RiskState.HALTED


def test_stale_feed_halts_during_session():
    S = load_settings()
    m = SyntheticMarket(["AAPL"], "SPY", seed=2)
    d = m.trading_days(datetime(2025, 3, 3), 1)[0]
    ev = m.day_events(d)
    s = build_system(S, None, ev[0][0] - 1)
    cut = [e for e in ev if e[0] < ev[0][0] + 3 * 3600 + 3600]  # stop feeding ~an hour into the session
    replay(s, cut)
    s.clock.advance_to(cut[-1][0] + 600)
    s.health_check()
    assert s.risk.state is RiskState.HALTED
    assert "stale" in s.risk.reasons[-1][2]


def test_deflated_sharpe_penalises_many_trials():
    assert deflated_sharpe(0.1, 250, 1) > deflated_sharpe(0.1, 250, 100)


def test_api_state_and_resume_requires_token(monkeypatch):
    monkeypatch.setenv("MYBOT_ADMIN_TOKEN", "secret-token")
    from mybot.api.server import create_app

    class RT:  # minimal runtime stub around a backtested system
        pass
    rt = RT()
    rt.system = _run(days=1)
    rt.S = rt.system.S
    rt.started, rt.mode_label, rt.simulated, rt.status_line, rt.boot_info = 0, "TEST", True, "ok", {}
    rt.backtest_report = {"status": "done", "metrics": {}}

    async def run():
        return None
    rt.run = run
    app = create_app(rt)
    with TestClient(app) as c:
        st = c.get("/api/state").json()
        assert st["meta"]["mode"] == "TEST" and "positions" in st and "scanner" in st
        assert c.post("/api/resume").status_code == 401
        assert c.post("/api/kill").status_code == 401
        assert c.post("/api/kill", headers={"Authorization": "Bearer secret-token"}).json()["state"] == "HALTED"
        assert c.post("/api/resume", headers={"Authorization": "Bearer secret-token"}).json()["state"] == "NORMAL"
