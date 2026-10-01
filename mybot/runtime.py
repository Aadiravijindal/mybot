"""Live runtime: feeds data into the TradingSystem and keeps it healthy.

Modes
  synthetic -- simulated market streamed at ``replay_speed`` (zero credentials)
  alpaca    -- real market data (IEX free / SIP paid) + Alpaca paper or live broker

Live trading requires BOTH  MYBOT_MODE=live  and  MYBOT_LIVE_CONFIRM=I_UNDERSTAND_THE_RISKS.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta

from .audit.journal import Journal
from .backtest.engine import build_system, deflated_sharpe, metrics, monte_carlo, replay
from .config import Settings
from .core.clock import SimClock, live_clock
from .core.events import Topic
from .data.synthetic import SyntheticMarket
from .engine.orchestrator import TradingSystem
from .execution.paper import PaperBroker
from .learning.research import ModelRegistry

log = logging.getLogger(__name__)
LIVE_CONFIRM = "I_UNDERSTAND_THE_RISKS"


class Runtime:
    def __init__(self, settings: Settings, bootstrap_days: int = 20):
        self.S = settings
        self.started = time.time()
        self.simulated = settings.data_source == "synthetic"
        self.bootstrap_days = bootstrap_days
        self.backtest_report: dict = {"status": "pending"}
        self.status_line = "starting"
        self.boot_info: dict = {}
        self.registry = ModelRegistry(settings.data_dir)
        self.preflight_report: dict = {}
        if settings.mode == "live":
            if self.simulated:
                raise SystemExit("Live mode needs real market data: use --data alpaca (or data_source = \"alpaca\").")
            if os.environ.get("MYBOT_LIVE_CONFIRM") != LIVE_CONFIRM:
                raise SystemExit("Refusing LIVE trading. Real money is at risk. After completing the go-live ladder in "
                                 f"docs/OPERATIONS.md, set MYBOT_LIVE_CONFIRM={LIVE_CONFIRM} and use your LIVE Alpaca keys.")
        journal = Journal(settings.data_dir)
        if self.simulated:
            self.market = SyntheticMarket(list(settings.symbols), settings.market_proxy, seed=int(time.time()) % 10000)
            start = datetime.now() - timedelta(days=1)
            self.days = iter(self.market.trading_days(datetime(start.year, start.month, start.day), 10_000))
            clock = SimClock(0.0)
            broker = PaperBroker(settings.costs)
            self.mode_label = "PAPER · SIMULATED MARKET"
        else:
            from .data.alpaca import AlpacaData
            from .execution.alpaca import AlpacaBroker
            clock = live_clock()
            broker = AlpacaBroker(paper=settings.mode != "live")
            broker.now = clock.now
            self.feed = AlpacaData(list(dict.fromkeys(list(settings.symbols) + [settings.market_proxy])))
            label = "LIVE · REAL MONEY" if settings.mode == "live" else "PAPER"
            self.mode_label = f"{label} · ALPACA ({self.feed.feed.upper()} data)"
        self.system = TradingSystem(settings, clock, broker, journal, live=True)
        self.system.mode_label = self.mode_label
        if not self.simulated:
            self._preflight()

    def _preflight(self) -> None:
        """Broker checks before anything else: account usable, real equity, PDT rule, leftovers."""
        s, rep = self.system, self.system.broker.preflight()
        self.preflight_report = rep
        now = s.clock.now()
        # Size from the REAL account, not the config's starting_equity.
        s.book.cash = s.book.start_equity = s.book.day_start_equity = s.book.hwm = rep["equity"]
        s.risk.pdt_remaining = s.broker.day_trades_remaining()
        if rep["positions"]:
            s.risk.halt(f"pre-existing broker positions {rep['positions']}: this engine did not open them and will not "
                        "manage them. Close them at the broker (or reconcile), then resume.", now)
        if not rep["shorting_enabled"] and s.S.risk.allow_short:
            log.warning("account cannot short; short setups will be rejected by the broker")
        s.J.write("risk", now, {"event": "preflight", "state": s.risk.state.value,
                                 "reason": f"equity ${rep['equity']:,.2f}; PDT-limited={rep['pdt_limited']}; "
                                           f"day trades left={s.risk.pdt_remaining}; stale orders cancelled={len(rep['stale_orders_cancelled'])}"})

    # ------------------------------------------------------------ bootstrap
    async def bootstrap(self) -> None:
        """Research -> backtest before trading: build the knowledge base from history.

        Synthetic mode replays a separate simulated history. Real-data mode downloads
        recent real 1-minute bars from Alpaca and replays them through the same engine
        (bars only), so analogues and the backtest report come from REAL data."""
        info = self.system.load_knowledge(self.registry, self.S.mode)
        self.boot_info = info
        if info["analogues"] >= 300:
            self.backtest_report = {"status": "loaded persisted knowledge base", **info}
            return
        if self.simulated:
            self.status_line = f"bootstrapping knowledge base from {self.bootstrap_days} simulated days"
            bt = await asyncio.to_thread(self._replay_synthetic_history)
            label = "SIMULATED (not evidence of edge)"
        else:
            self.status_line = f"downloading ~{self.bootstrap_days} days of real 1-minute history for the knowledge base"
            try:
                events = await self.feed.history_events(self.bootstrap_days, now=self.system.clock.now())
            except Exception as exc:
                log.warning("history download failed: %s", exc)
                self.backtest_report = {"status": f"history download failed ({type(exc).__name__}); no analogues -> "
                                                  "every candidate is vetoed until the knowledge base exists", **info}
                return
            self.status_line = f"replaying {len(events):,} real bars through the engine"
            bt = await asyncio.to_thread(self._replay_events, events)
            label = "REAL Alpaca 1-minute bars (bars only, no quotes): still a small sample"
        self._absorb(bt, info, label)

    def _replay_synthetic_history(self):
        hist = SyntheticMarket(list(self.S.symbols), self.S.market_proxy, seed=12345)
        days = hist.trading_days(datetime(2025, 1, 6), self.bootstrap_days)
        first = hist.day_events(days[0])
        bt = build_system(self.S, None, first[0][0] - 1)
        replay(bt, first)
        for d in days[1:]:
            replay(bt, hist.day_events(d))
        return bt

    def _replay_events(self, events):
        bt = build_system(self.S, None, events[0][0] - 1 if events else 0.0)
        replay(bt, events)
        return bt

    def _absorb(self, bt, info: dict, label: str) -> None:
        for o in bt.analogues.items:
            self.system.analogues.add(o)
            self.system.J.outcome(o, self.system.kb_source)
        for k, v in bt.drift.ref.items():
            self.system.drift.ref[k].extend(v)
        m = metrics(bt)
        import numpy as np
        r = np.array([t.r for t in bt.book.closed])
        self.backtest_report = {
            "status": "done", "days": self.bootstrap_days, "data": label,
            "metrics": m.to_dict(), "monte_carlo": monte_carlo(r, self.S.risk.risk_per_trade_pct),
            "deflated_sharpe_prob": round(deflated_sharpe(m.sharpe_daily / (252 ** 0.5), self.bootstrap_days, n_trials=4), 3),
            "analogues_learned": len(bt.analogues.items),
            "decisions": dict(bt.funnel),
        }
        self.boot_info = {"analogues": len(self.system.analogues.items), "model": info["model"]}

    # ------------------------------------------------------------ loops
    async def run(self) -> None:
        await self.bootstrap()
        self.status_line = "running" if self.system.risk.state.value != "HALTED" else "HALTED - see risk events"
        tasks = [asyncio.create_task(self._health_loop())]
        if self.simulated:
            tasks.append(asyncio.create_task(self._synthetic_loop()))
        else:
            self.system.stream_start_ts = self.system.clock.now()
            tasks.append(asyncio.create_task(self.feed.run(self.system)))
            tasks.append(asyncio.create_task(self.system.broker.run(self.system)))
        await asyncio.gather(*tasks)

    async def _health_loop(self) -> None:
        while True:
            await asyncio.sleep(2.0)
            try:
                self.system.poll_research()
                if not self.simulated:
                    self.system.health_check(clock_drift_s=getattr(self.feed, "clock_offset_s", 0.0))
            except Exception:
                log.exception("health loop")
                self.system.risk.record_component_error("health loop", self.system.clock.now())

    async def _synthetic_loop(self) -> None:
        clock: SimClock = self.system.clock  # type: ignore[assignment]
        speed = self.S.replay_speed
        while True:
            day = next(self.days)
            events = await asyncio.to_thread(self.market.day_events, day)
            if not events:
                continue
            clock.advance_to(max(clock.now(), events[0][0] - 1))
            wall0, sim0 = time.monotonic(), clock.now()
            next_health = 0.0
            i = 0
            while i < len(events):
                sim_target = sim0 + (time.monotonic() - wall0) * speed
                while i < len(events) and events[i][0] <= sim_target:
                    ts, topic, payload = events[i]
                    clock.advance_to(max(ts, clock.now()))
                    self.system.publish(topic, payload, getattr(payload, "symbol", None), ts=ts)
                    if ts >= next_health:
                        self.system.health_check()
                        next_health = ts + 60
                    i += 1
                if i < len(events):
                    # Skip the dead pre-market gap quickly.
                    if events[i][0] - sim_target > 600:
                        sim0 += events[i][0] - sim_target - 5
                    clock.advance_to(max(clock.now(), min(sim_target, events[i][0])))
                await asyncio.sleep(0.05)
            self.status_line = "session closed; next simulated session starting"
            await asyncio.sleep(3)
