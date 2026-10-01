"""Soak test: replay many simulated days across seeds and check hard invariants.

Usage: python scripts/soak.py --seeds 5 --days 30 [--faults]
Exit code 0 only if every invariant holds.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import replace
from datetime import datetime

from mybot.backtest.engine import build_system, metrics
from mybot.config import load_settings
from mybot.core.events import Topic
from mybot.core.types import Bar, Quote, Trade
from mybot.data.synthetic import SyntheticMarket
from mybot.risk.engine import RiskState


def inject_faults(events, rng: random.Random):
    """Corrupt a copy of the stream: duplicates, crossed quotes, bad OHLC, zero prices, out-of-order ticks."""
    out = []
    for ev in events:
        ts, topic, p = ev
        out.append(ev)
        r = rng.random()
        if topic is Topic.TRADE and r < 0.02:
            out.append(ev)                                           # duplicate tick
        elif topic is Topic.QUOTE and r < 0.005:
            out.append((ts, topic, Quote(p.symbol, p.ts, p.ask + 0.05, p.bid, p.bid_size, p.ask_size, p.source, ts)))  # crossed
        elif topic is Topic.BAR and r < 0.003:
            out.append((ts, topic, Bar(p.symbol, p.ts, p.end_ts, p.open, p.low - 1, p.high + 1, p.close, p.volume, p.source, ts)))  # dup + bad
        elif topic is Topic.TRADE and r < 0.021:
            out.append((ts, topic, Trade(p.symbol, p.ts - 0.001, 0.0, p.size, p.source, ts)))  # zero price
    return out


def check(sysm, label: str) -> list[str]:
    errs = []
    b = sysm.book
    if sysm.bus.errors:
        errs.append(f"{label}: {sysm.bus.errors} handler errors")
    if b.positions:
        errs.append(f"{label}: positions left open overnight: {list(b.positions)}")
    if sysm.broker.positions():
        errs.append(f"{label}: broker still holds {sysm.broker.positions()}")
    # Accounting: equity change must equal the sum of closed-trade P&L (net of fees).
    pnl = sum(t.pnl for t in b.closed)
    # closed-trade P&L is stored rounded to the cent: allow half a cent of rounding per trade
    if abs((b.equity() - b.start_equity) - pnl) > 0.005 * len(b.closed) + 0.01:
        errs.append(f"{label}: accounting mismatch equity delta {b.equity() - b.start_equity:.2f} vs trades {pnl:.2f}")
    lim = sysm.S.risk
    for t in b.closed:
        hist = [s for (_, s, _) in t.stop_history]
        mono = all((b2 - a) * t.side >= -1e-9 for a, b2 in zip(hist, hist[1:]))
        if not mono:
            errs.append(f"{label}: stop loosened on {t.symbol} {t.decision_id}: {hist}")
        if t.r < -1.6 and not t.exit_reason.startswith("stop"):
            # Non-stop exits are decided on a bar close while the stop still protects; a loss far
            # beyond -1R there means the stop was not working. (Stop fills beyond -1R are gaps and are
            # reported separately as gap events: real markets jump through stops.)
            errs.append(f"{label}: loss beyond -1.6R without a stop gap on {t.symbol}: {t.r}R ({t.exit_reason})")
        risk0 = abs(t.entry_px - hist[0]) * t.qty if hist else 0
        if risk0 > b.start_equity * lim.risk_per_trade_pct / 100 * 1.5:
            errs.append(f"{label}: initial risk ${risk0:.0f} exceeds per-trade limit")
        notional = t.entry_px * t.qty
        if notional > b.start_equity * lim.max_position_notional_pct / 100 * 1.1:
            errs.append(f"{label}: notional ${notional:.0f} exceeds position limit")
    for d in sysm.J.decisions(limit=100000):
        if d["state"] == "ENTER":
            fc = d["forecast"]
            if fc["ev_r_conservative"] < sysm.S.strategy.min_ev_r or d["vetoes"]:
                errs.append(f"{label}: ENTER decision {d['id']} violated a gate")
    if sysm.risk.state is RiskState.HALTED:
        errs.append(f"{label}: system HALTED: {sysm.risk.reasons[-1]}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--days", type=int, default=25)
    ap.add_argument("--faults", action="store_true")
    ap.add_argument("--stress", action="store_true", help="relax entry gates so the position/exit paths get exercised heavily")
    a = ap.parse_args()
    S = load_settings()
    if a.stress:
        S = replace(S, strategy=replace(S.strategy, min_analogues=0, min_ev_r=-9.0, min_prob_lcb=0.0, min_reward_risk=0.0),
                    risk=replace(S.risk, max_trades_per_day=200, max_consecutive_losses=10_000, max_daily_loss_pct=50.0,
                                 max_drawdown_pct=90.0))
    all_errs, t0, total_trades = [], time.time(), 0
    for seed in range(a.seeds):
        m = SyntheticMarket(list(S.symbols), S.market_proxy, seed=100 + seed)
        days = m.trading_days(datetime(2025, 1, 6), a.days)
        rng = random.Random(seed)
        evs0 = m.day_events(days[0])
        sysm = build_system(S, None, evs0[0][0] - 1)
        for i, d in enumerate(days):
            evs = evs0 if i == 0 else m.day_events(d)
            if a.faults:
                evs = inject_faults(evs, rng)
            for ts, topic, p in evs:
                sysm.clock.advance_to(max(ts, sysm.clock.now()))
                sysm.publish(topic, p, getattr(p, "symbol", None), ts=ts)
                if topic is Topic.BAR and p.symbol == S.market_proxy:
                    sysm.health_check()
            day_errs = [e for e in check(sysm, f"seed{seed} day{i}") if "HALTED" in e or "handler" in e or "overnight" in e or "broker still" in e]
            if day_errs:
                all_errs += day_errs
                break
        all_errs += check(sysm, f"seed{seed} final")
        mt = metrics(sysm)
        total_trades += mt.trades
        gaps = [t.r for t in sysm.book.closed if t.r < -1.6]
        if gaps:
            print(f"  gap events (stop jumped): {len(gaps)} worst {min(gaps):+.2f}R")
        print(f"seed {seed}: trades={mt.trades} expectancy={mt.expectancy_r:+.3f}R dd={mt.max_drawdown_pct:.2f}% "
              f"rejected={sysm.dq.feed_health(sysm.clock.now())['rejected']} risk={sysm.risk.state.value}", flush=True)
    print(f"{a.seeds * a.days} days, {total_trades} trades in {time.time() - t0:.0f}s")
    errs = list(dict.fromkeys(all_errs))
    for e in errs[:50]:
        print("INVARIANT FAILED:", e)
    print("OK" if not errs else f"{len(errs)} invariant failures")
    sys.exit(0 if not errs else 1)


if __name__ == "__main__":
    main()
