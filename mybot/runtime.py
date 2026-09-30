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
from .core.clock import SimClock, WallClock
from .core.events import Topic
from .data.synthetic import SyntheticMarket
from .engine.orchestrator import TradingSystem
from .execution.paper import PaperBroker
from .learning.research import ModelRegistry

log = logging.getLogger(__name__)


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
        if settings.mode == "live":
            if os.environ.get("MYBOT_LIVE_CONFIRM") != "I_UNDERSTAND_THE_RISKS":
                raise SystemExit("Refusing live mode: set MYBOT_LIVE_CONFIRM=I_UNDERSTAND_THE_RISKS after reading docs/OPERATIONS.md")
            if self.simulated:
                raise SystemExit("Live mode requires a real data source")
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
            clock = WallClock()
            broker = AlpacaBroker(paper=settings.mode != "live")
            self.feed = AlpacaData(list(settings.symbols) + [settings.market_proxy])
            self.mode_label = f"{settings.mode.upper()} · ALPACA ({self.feed.feed.upper()} feed)"
        self.system = TradingSystem(settings, clock, broker, journal, live=True)
        self.system.mode_label = self.mode_label

    # ------------------------------------------------------------ bootstrap
    def bootstrap(self) -> None:
        """Research -> backtest before trading: build the knowledge base from history.

        Synthetic mode: a separate simulated history. Real-data mode: load the
        persisted knowledge base (populate it with `python -m mybot backtest` on
        vendor history first)."""
        info = self.system.load_knowledge(self.registry, self.S.mode)
        self.boot_info = info
        if not self.simulated or info["analogues"] >= 300:
            self.backtest_report = {"status": "loaded persisted knowledge base", **info}
            return
        self.status_line = f"bootstrapping knowledge base from {self.bootstrap_days} simulated days"
        hist = SyntheticMarket(list(self.S.symbols), self.S.market_proxy, seed=12345)
        days = hist.trading_days(datetime(2025, 1, 6), self.bootstrap_days)
        first = hist.day_events(days[0])
        bt = build_system(self.S, None, first[0][0] - 1)
        replay(bt, first)
        for d in days[1:]:
            replay(bt, hist.day_events(d))
        for o in bt.analogues.items:
            self.system.analogues.add(o)
            self.system.J.outcome(o)
        for k, v in bt.drift.ref.items():
            self.system.drift.ref[k].extend(v)
        m = metrics(bt)
        import numpy as np
        r = np.array([t.r for t in bt.book.closed])
        self.backtest_report = {
            "status": "done", "days": self.bootstrap_days, "data": "SIMULATED (not evidence of edge)",
            "metrics": m.to_dict(), "monte_carlo": monte_carlo(r, self.S.risk.risk_per_trade_pct),
            "deflated_sharpe_prob": round(deflated_sharpe(m.sharpe_daily / (252 ** 0.5), self.bootstrap_days, n_trials=4), 3),
            "analogues_learned": len(bt.analogues.items),
            "decisions": dict(bt.funnel),
        }
        self.boot_info = {"analogues": len(self.system.analogues.items), "model": info["model"]}

    # ------------------------------------------------------------ loops
    async def run(self) -> None:
        await asyncio.to_thread(self.bootstrap)
        self.status_line = "running"
        tasks = [asyncio.create_task(self._health_loop())]
        if self.simulated:
            tasks.append(asyncio.create_task(self._synthetic_loop()))
        else:
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
