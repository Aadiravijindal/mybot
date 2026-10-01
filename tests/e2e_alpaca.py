"""End-to-end test of the LIVE runtime against a local fake Alpaca.

Runs the real `python -m mybot --data alpaca` process (websocket data, REST broker,
pre-flight, history bootstrap, dashboard API) against tests/fake_alpaca.py on an
accelerated clock, through several scenarios. Prints a report; exit 0 = all passed.

    python tests/e2e_alpaca.py            # all scenarios (~10 minutes)
    python tests/e2e_alpaca.py paper      # one scenario
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

ROOT = Path(__file__).resolve().parent.parent
TOKEN = "e2e-admin-token"
SPEED = float(os.environ.get("E2E_SPEED", "20"))
DURATION = float(os.environ.get("E2E_SECONDS", "200"))
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, bool(ok), detail))
    print(("  PASS " if ok else "  FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)
    return ok


def session_start(minutes_after_open: int = 25) -> float:
    tz = ZoneInfo("America/New_York")
    d = datetime.now(tz)
    while d.weekday() >= 5:
        d = d.replace(day=d.day - 1)
    return datetime(d.year, d.month, d.day, 9, 30, tzinfo=tz).timestamp() + minutes_after_open * 60


class Env:
    def __init__(self, name, fport, bport, mode="paper", confirm=True, preexisting=None, pdt=False, start_offset=25):
        self.name, self.fport, self.bport = name, fport, bport
        self.tmp = Path(tempfile.mkdtemp(prefix=f"mybot-e2e-{name}-"))
        self.start = session_start(start_offset)
        self.wall0 = time.time()   # shared anchor: fake market and bot run one simulated clock
        args = [sys.executable, str(ROOT / "tests" / "run_fake_alpaca.py"), str(fport), str(self.start), str(SPEED),
                json.dumps(preexisting or {}), "pdt" if pdt else "-", str(self.wall0)]
        self.fake = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=open(self.tmp / "fake.log", "w"))
        # data_dir must be a top-level key: put it first
        cfg = f'data_dir = "{self.tmp / "var"}"\n' + (ROOT / "tests" / "e2e_config.toml").read_text()
        (self.tmp / "cfg.toml").write_text(cfg)
        env = dict(os.environ, ALPACA_API_KEY_ID="fake-key", ALPACA_API_SECRET_KEY="fake-secret",
                   ALPACA_TRADING_URL=f"http://127.0.0.1:{fport}", ALPACA_DATA_URL=f"http://127.0.0.1:{fport}",
                   ALPACA_STREAM_URL=f"ws://127.0.0.1:{fport}", ALPACA_HISTORY_FEED="iex",
                   MYBOT_TEST_CLOCK=f"{self.start},{SPEED},{self.wall0}", MYBOT_PORT=str(bport), MYBOT_ADMIN_TOKEN=TOKEN,
                   PYTHONPATH=str(ROOT))
        env.pop("ANTHROPIC_API_KEY", None)
        if confirm:
            env["MYBOT_LIVE_CONFIRM"] = "I_UNDERSTAND_THE_RISKS"
        else:
            env.pop("MYBOT_LIVE_CONFIRM", None)
        self.wait_http(f"http://127.0.0.1:{fport}/_state")
        self.bot = subprocess.Popen([sys.executable, "-m", "mybot", "--config", str(self.tmp / "cfg.toml"), "--mode", mode,
                                     "--bootstrap-days", "8"], cwd=ROOT, env=env,
                                    stdout=open(self.tmp / "bot.out", "w"), stderr=subprocess.STDOUT)
        self.c = httpx.Client(timeout=10)

    @staticmethod
    def wait_http(url, timeout=60):
        t = time.time()
        while time.time() - t < timeout:
            try:
                if httpx.get(url, timeout=2).status_code < 500:
                    return True
            except httpx.HTTPError:
                time.sleep(0.3)
        return False

    def state(self):
        return self.c.get(f"http://127.0.0.1:{self.bport}/api/state").json()

    def fake_state(self):
        return self.c.get(f"http://127.0.0.1:{self.fport}/_state").json()

    def chaos(self, **kw):
        return self.c.post(f"http://127.0.0.1:{self.fport}/_chaos", json=kw).json()

    def wait_running(self, timeout=240):
        t = time.time()
        while time.time() - t < timeout:
            if self.bot.poll() is not None:
                return False
            try:
                s = self.state()
                if s["meta"]["status_line"].startswith(("running", "HALTED")):
                    return s
            except Exception:
                pass
            time.sleep(1)
        return False

    def output(self):
        return (self.tmp / "bot.out").read_text()

    def close(self):
        for p in (self.bot, self.fake):
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
                try:
                    p.wait(10)
                except subprocess.TimeoutExpired:
                    p.kill()
        shutil.rmtree(self.tmp, ignore_errors=True)


def book_matches_broker(s, fs):
    book = {p["symbol"]: (p["qty"] if p["side"] == "LONG" else -p["qty"]) for p in s["positions"]}
    return book == {k: v for k, v in fs["positions"].items() if v}, book, fs["positions"]


def stops_cover_positions(fs):
    """Every broker position must be covered by exactly-sized resting protective stops."""
    bad = []
    for sym, q in fs["positions"].items():
        side = "sell" if q > 0 else "buy"
        cover = sum(int(o["qty"]) for o in fs["orders"] if o["symbol"] == sym and o["type"] == "stop"
                    and o["side"] == side and o["status"] in ("new", "accepted"))
        if cover != abs(q):
            bad.append(f"{sym}: position {q}, stop cover {cover}")
    return bad


def scenario_paper():
    print("\n[paper] paper trading on the live runtime against fake Alpaca", flush=True)
    e = Env("paper", 18801, 18901)
    try:
        s = e.wait_running()
        if not check("runtime starts, pre-flight + history bootstrap complete", bool(s), "" if s else e.output()[-800:]):
            return
        check("mode label is PAPER (not live)", s["meta"]["mode"].startswith("PAPER"), s["meta"]["mode"])
        bt = s["backtest"]
        check("knowledge base bootstrapped from REAL-data path (Alpaca history endpoint)",
              bt.get("status") == "done" and "REAL" in bt.get("data", "") and bt.get("analogues_learned", 0) > 0,
              f"{bt.get('status')} analogues={bt.get('analogues_learned')}")
        check("equity taken from broker account, not config", abs(s["account"]["equity"] - 50_000) < 2_000, str(s["account"]["equity"]))
        mismatches, max_pos, trades, uncovered = 0, 0, 0, []
        t0 = time.time()
        while time.time() - t0 < DURATION:         # simulated minutes = DURATION * SPEED / 60
            time.sleep(4)
            s, fs = e.state(), e.fake_state()
            ok, book, brk = book_matches_broker(s, fs)
            if not ok:
                time.sleep(3)                       # allow one poll cycle for in-flight fills
                s, fs = e.state(), e.fake_state()
                ok, book, brk = book_matches_broker(s, fs)
                mismatches += 0 if ok else 1
            uncovered += stops_cover_positions(fs) if not any(p.get("research") is None for p in s["positions"]) else []
            max_pos = max(max_pos, len(s["positions"]))
            trades = s["account"]["closed_trades"]
        rej = [o for o in s["orders"] if o["status"] == "REJECTED"]
        replaced = sum(1 for o in fs["orders"] if o.get("replaces"))
        exits = sum(1 for o in fs["orders"] if o["type"] == "market" and o["status"] == "filled")
        print(f"  info: {trades} closed trades, {exits} market fills, {replaced} stop replacements (PATCH), "
              f"{sum(1 for o in fs['orders'] if o['status'] == 'canceled')} cancels, equity {s['account']['equity']}", flush=True)
        check("trades happened (entries, stops, exits through the real adapter)", trades + max_pos > 0, f"closed={trades} max_open={max_pos}")
        check("book == broker positions at every check", mismatches == 0, f"mismatches={mismatches}")
        check("every broker position fully covered by a resting stop", not uncovered, "; ".join(uncovered[:3]))
        check("no rejected orders (incl. 'insufficient qty' from stop reservation)", not rej, json.dumps(rej[:2]))
        check("no handler errors", s["health"]["handler_errors"] == 0, str(s["health"]["handler_errors"]))
        check("risk state NORMAL", s["risk"]["state"] == "NORMAL", json.dumps(s["risk"]["recent_reasons"][-2:]))
        check("dashboard page served", e.c.get(f"http://127.0.0.1:{e.bport}/").status_code == 200)
        check("resume refused without token", e.c.post(f"http://127.0.0.1:{e.bport}/api/resume").status_code == 401)
        if s["decisions"]:
            d = s["decisions"][0]
            r = e.c.get(f"http://127.0.0.1:{e.bport}/api/decision", params={"symbol": d["symbol"], "at": d["ts"]})
            check("decision reconstruction endpoint", r.status_code == 200 and r.json()["id"] == d["id"])
        # Kill switch: must flatten at the broker.
        r = e.c.post(f"http://127.0.0.1:{e.bport}/api/kill", headers={"Authorization": f"Bearer {TOKEN}"})
        check("kill switch accepted", r.status_code == 200 and r.json()["state"] == "HALTED")
        time.sleep(6)
        fs = e.fake_state()
        # flatten happens on the next bar; wait up to ~3 simulated minutes
        for _ in range(10):
            if not fs["positions"]:
                break
            time.sleep(2)
            fs = e.fake_state()
        check("kill switch flattened every broker position", not fs["positions"], json.dumps(fs["positions"]))
        open_stops = [o for o in fs["orders"] if o["type"] == "stop" and o["status"] in ("new", "accepted")]
        check("no orphan stop orders left at broker after flatten", not open_stops, f"{len(open_stops)} open")
        s = e.state()
        check("no new entries while HALTED", all(p["symbol"] not in fs["positions"] for p in s["positions"]))
    finally:
        e.close()


def scenario_live_guard():
    print("\n[live-guard] live mode requires explicit confirmation", flush=True)
    e = Env("liveguard", 18802, 18902, mode="live", confirm=False)
    try:
        rc = e.bot.wait(60)
        out = e.output()
        check("live mode REFUSED without MYBOT_LIVE_CONFIRM", rc != 0 and "Refusing LIVE trading" in out, out.strip().splitlines()[-1][:160])
    finally:
        e.close()
    e = Env("live", 18803, 18903, mode="live", confirm=True)
    try:
        s = e.wait_running()
        check("live mode starts with confirmation (against fake broker)", bool(s), "" if s else e.output()[-600:])
        if s:
            check("dashboard shows LIVE · REAL MONEY", s["meta"]["mode"].startswith("LIVE"), s["meta"]["mode"])
            check("broker labelled alpaca-LIVE", s["health"]["broker"]["name"] == "alpaca-LIVE", s["health"]["broker"]["name"])
    finally:
        e.close()


def scenario_preexisting():
    print("\n[leftovers] positions the engine did not open must halt it", flush=True)
    e = Env("leftover", 18804, 18904, preexisting={"AAPL": 10})
    try:
        s = e.wait_running()
        check("starts", bool(s))
        if s:
            reasons = json.dumps(s["risk"]["recent_reasons"])
            check("HALTED on pre-existing broker positions", s["risk"]["state"] == "HALTED" and "pre-existing" in reasons, reasons[-200:])
            time.sleep(8)
            fs = e.fake_state()
            check("does NOT touch positions it did not open", fs["positions"].get("AAPL") == 10, json.dumps(fs["positions"]))
    finally:
        e.close()


def scenario_chaos():
    print("\n[chaos] broker outage and stop rejection", flush=True)
    e = Env("chaos", 18805, 18905)
    try:
        s = e.wait_running()
        if not check("starts", bool(s)):
            return
        # Broker refuses every protective stop. Any entry must then be exited immediately.
        e.chaos(reject_stops=True)
        t0, entered = time.time(), False
        while time.time() - t0 < 240 and not entered:
            time.sleep(2)
            fs = e.fake_state()
            entered = any(o["type"] == "market" and o["status"] == "filled" for o in fs["orders"])
        if check("an entry happened while stops were being rejected", entered):
            time.sleep(15)
            s, fs = e.state(), e.fake_state()
            lost = [x for x in s["research"] if x.get("event") == "stop_lost"]
            check("rejected stop detected and journalled (stop_lost)", len(lost) > 0, f"{len(lost)} events")
            check("position exited instead of sitting unprotected", not fs["positions"], json.dumps(fs["positions"]))
        e.chaos(reject_stops=False)
        e.c.post(f"http://127.0.0.1:{e.bport}/api/resume", headers={"Authorization": f"Bearer {TOKEN}"})
        e.chaos(rest_down=True)
        time.sleep(12)
        s = e.state()
        reasons = json.dumps(s["risk"]["recent_reasons"])
        check("REST outage -> HALTED (broker disconnected)", s["risk"]["state"] == "HALTED" and "broker" in reasons, reasons[-200:])
        check("dashboard shows broker disconnected", s["health"]["broker"]["connected"] is False)
        e.chaos(rest_down=False)
        time.sleep(6)
        s = e.state()
        check("stays HALTED after recovery until a human resumes", s["risk"]["state"] == "HALTED")
        r = e.c.post(f"http://127.0.0.1:{e.bport}/api/resume", headers={"Authorization": f"Bearer {TOKEN}"})
        check("admin resume works", r.status_code == 200 and r.json()["state"] == "NORMAL")
        check("no handler errors through chaos", e.state()["health"]["handler_errors"] == 0)
    finally:
        e.close()


def scenario_pdt():
    print("\n[pdt] pattern-day-trader limit (account < $25k with 3 day trades used)", flush=True)
    e = Env("pdt", 18806, 18906, pdt=True)
    try:
        s = e.wait_running()
        if check("starts", bool(s)):
            check("PDT day trades left = 0", s["risk"].get("pdt_day_trades_left") == 0, str(s["risk"].get("pdt_day_trades_left")))
            time.sleep(120)
            s, fs = e.state(), e.fake_state()
            pdt_blocks = [d for d in s["decisions"] if any("pattern-day-trader" in b for b in d["blockers"])]
            check("no entries while out of day trades", not fs["positions"] and s["account"]["closed_trades"] == 0)
            check("decisions show the PDT blocker", len(pdt_blocks) > 0 or not s["decisions"], f"{len(pdt_blocks)} blocked")
    finally:
        e.close()


SCENARIOS = {"paper": scenario_paper, "live": scenario_live_guard, "leftovers": scenario_preexisting,
             "chaos": scenario_chaos, "pdt": scenario_pdt}

if __name__ == "__main__":
    names = sys.argv[1:] or list(SCENARIOS)
    for n in names:
        SCENARIOS[n]()
    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    sys.exit(1 if failed else 0)
