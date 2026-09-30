"""Event-driven backtester.

Runs the *same* TradingSystem as live, on a SimClock that is advanced to each
event's ``avail_ts``. Because the bus rejects any event whose avail_ts is in the
future, look-ahead is structurally impossible rather than merely discouraged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from ..audit.journal import Journal
from ..config import Settings
from ..core.clock import SimClock
from ..core.events import Topic
from ..engine.orchestrator import TradingSystem
from ..execution.paper import PaperBroker


def build_system(settings: Settings, data_dir: str | None = None, start_ts: float = 0.0) -> TradingSystem:
    clock = SimClock(start_ts)
    broker = PaperBroker(settings.costs)
    return TradingSystem(settings, clock, broker, Journal(data_dir), live=False, enable_llm=False)


def replay(system: TradingSystem, events: Iterable[tuple[float, Topic, object]], health_every_s: float = 60.0) -> None:
    clock: SimClock = system.clock  # type: ignore[assignment]
    next_health = None
    for ts, topic, payload in events:
        clock.advance_to(max(ts, clock.now()))
        system.publish(topic, payload, getattr(payload, "symbol", None), ts=ts)
        if next_health is None or ts >= next_health:
            system.health_check()
            next_health = ts + health_every_s


@dataclass
class Metrics:
    trades: int
    win_rate: float
    expectancy_r: float
    profit_factor: float
    total_return_pct: float
    sharpe_daily: float
    max_drawdown_pct: float
    avg_mae_r: float
    avg_mfe_r: float
    fees: float
    exposure_bars: int
    t_stat_r: float

    def to_dict(self):
        return self.__dict__


def metrics(system: TradingSystem) -> Metrics:
    t = system.book.closed
    r = np.array([x.r for x in t]) if t else np.zeros(0)
    pnl = np.array([x.pnl for x in t]) if t else np.zeros(0)
    eq = np.array([e for _, e in system.book.equity_curve]) if system.book.equity_curve else np.array([system.book.start_equity])
    ts = np.array([t_ for t_, _ in system.book.equity_curve]) if system.book.equity_curve else np.zeros(1)
    # daily returns from end-of-day equity
    days = {}
    for t_, e in zip(ts, eq):
        days[system.session.day_key(t_)] = e
    de = np.array([system.book.start_equity] + list(days.values()))
    dr = np.diff(de) / de[:-1] if len(de) > 1 else np.zeros(0)
    peak = np.maximum.accumulate(eq)
    gains, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return Metrics(
        trades=len(t),
        win_rate=float((r > 0).mean()) if len(r) else 0.0,
        expectancy_r=float(r.mean()) if len(r) else 0.0,
        profit_factor=float(gains / losses) if losses > 0 else (float("inf") if gains > 0 else 0.0),
        total_return_pct=float((eq[-1] / system.book.start_equity - 1) * 100),
        sharpe_daily=float(dr.mean() / dr.std() * math.sqrt(252)) if len(dr) > 1 and dr.std() > 0 else 0.0,
        max_drawdown_pct=float(((peak - eq) / peak).max() * 100) if len(eq) else 0.0,
        avg_mae_r=float(np.mean([x.mae_r for x in t])) if t else 0.0,
        avg_mfe_r=float(np.mean([x.mfe_r for x in t])) if t else 0.0,
        fees=float(sum(x.fees for x in t)),
        exposure_bars=int(sum(x.bars for x in t)),
        t_stat_r=float(r.mean() / (r.std(ddof=1) / math.sqrt(len(r)))) if len(r) > 2 and r.std() > 0 else 0.0,
    )


def monte_carlo(r: np.ndarray, risk_pct: float, n: int = 2000, seed: int = 0) -> dict:
    """Bootstrap the R-sequence -> distribution of drawdown and final return."""
    if len(r) < 10:
        return {"note": "too few trades for Monte Carlo"}
    rng = np.random.default_rng(seed)
    dds, finals = [], []
    for _ in range(n):
        s = rng.choice(r, size=len(r), replace=True)
        eq = np.cumprod(1 + s * risk_pct / 100)
        peak = np.maximum.accumulate(eq)
        dds.append(((peak - eq) / peak).max() * 100)
        finals.append((eq[-1] - 1) * 100)
    return {"dd_p50": float(np.percentile(dds, 50)), "dd_p95": float(np.percentile(dds, 95)),
            "ret_p5": float(np.percentile(finals, 5)), "ret_p50": float(np.percentile(finals, 50)),
            "prob_loss": float(np.mean(np.array(finals) < 0))}


def deflated_sharpe(sr: float, n_obs: int, n_trials: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Bailey & Lopez de Prado (2014): P(true SR > 0) after correcting for the
    number of strategy variants tried. Use it; the raw Sharpe of the best of many
    backtests is biased upward."""
    from statistics import NormalDist
    nd = NormalDist()
    if n_obs < 3:
        return 0.0
    emc = 0.5772156649
    n_trials = max(n_trials, 1)
    sr0 = math.sqrt(1.0 / n_obs) * ((1 - emc) * nd.inv_cdf(1 - 1.0 / max(n_trials, 2)) +
                                     emc * nd.inv_cdf(1 - 1.0 / (max(n_trials, 2) * math.e))) if n_trials > 1 else 0.0
    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))
    return nd.cdf((sr - sr0) * math.sqrt(n_obs - 1) / denom)
