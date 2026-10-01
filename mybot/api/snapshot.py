"""Build the dashboard state snapshot (read-only view of the system)."""
from __future__ import annotations

import time

import numpy as np


def _r(x, n=2):
    return None if x is None else round(float(x), n)


def build(rt) -> dict:
    s = rt.system
    now = s.clock.now()
    b = s.book
    eq = b.equity()
    L = s.S.risk
    gross = sum(abs(p.qty * b.marks.get(sym, p.avg_px)) for sym, p in s.book.positions.items())
    day_pnl = eq - b.day_start_equity
    hwm = max(b.hwm, eq)   # the stored peak only updates once a minute
    dd = max(0.0, (1 - eq / hwm) * 100) if hwm else 0
    positions = []
    for sym, p in s.book.positions.items():
        px = b.marks.get(sym, p.avg_px)
        nxt = s.calendar.next_event(sym, now)
        positions.append({
            "symbol": sym, "side": "LONG" if p.side > 0 else "SHORT", "qty": p.qty, "setup": p.setup,
            "entry": _r(p.avg_px, 4), "current": _r(px, 4), "pnl": _r(p.unrealized(px) + p.realized - p.fees),
            "r": _r(p.r_now(px)), "stop": _r(p.stop.current, 4), "stop_components": p.stop.components,
            "stop_history": p.stop.history[-8:], "targets": [_r(t, 2) for t in p.targets],
            "thesis": p.thesis.state.value, "thesis_score": _r(p.thesis.score), "thesis_text": p.thesis.text,
            "thesis_log": p.thesis.log[-6:], "bars_held": p.bars_held, "horizon": p.horizon,
            "risk_dollars": _r(p.initial_risk_per_share * p.qty), "mae_r": _r(p.mae), "mfe_r": _r(p.mfe),
            "research": p.research[-5:], "decision_id": p.decision_id, "forecast": p.forecast,
            "next_event": None if nxt is None else {"name": nxt.name, "in_min": round((nxt.scheduled_ts - now) / 60)},
        })
    closed = [{"symbol": t.symbol, "side": "LONG" if t.side > 0 else "SHORT", "setup": t.setup, "entry_ts": t.entry_ts,
               "exit_ts": t.exit_ts, "qty": t.qty, "entry_px": _r(t.entry_px, 4), "exit_px": _r(t.exit_px, 4),
               "pnl": t.pnl, "r": t.r, "exit_reason": t.exit_reason, "attribution": t.attribution, "decision_id": t.decision_id}
              for t in b.closed[-60:]][::-1]
    lat = list(s.stats["latency_ms"])
    rs = [t.r for t in b.closed]
    day_curve = [(t, e) for t, e in b.equity_curve if now - t < 3 * 86400][-1200:]
    util = {
        "daily_loss": _r(max(0.0, -day_pnl / b.day_start_equity * 100) / L.max_daily_loss_pct * 100, 1),
        "drawdown": _r(dd / L.max_drawdown_pct * 100, 1),
        "positions": _r(len(positions) / L.max_open_positions * 100, 1),
        "trades_today": _r(s.risk.trades_today / L.max_trades_per_day * 100, 1),
        "gross_exposure": _r(gross / (eq * L.max_gross_exposure_pct / 100) * 100, 1) if eq > 0 else None,
    }
    return {
        "meta": {"mode": rt.mode_label, "simulated": rt.simulated, "data_source": s.S.data_source,
                 "now": now, "local_time": s.session.local(now).strftime("%Y-%m-%d %H:%M:%S %Z"),
                 "market_open": s.session.is_open(now), "uptime_s": round(time.time() - rt.started),
                 "status_line": rt.status_line, "boot": rt.boot_info},
        "account": {"equity": _r(eq), "cash": _r(b.cash), "day_pnl": _r(day_pnl),
                    "day_pnl_pct": _r(day_pnl / b.day_start_equity * 100, 3), "total_pnl": _r(eq - b.start_equity),
                    "hwm": _r(hwm), "drawdown_pct": _r(dd, 3), "gross_exposure": _r(gross), "open_positions": len(positions),
                    "closed_trades": len(b.closed), "win_rate": _r(np.mean([r > 0 for r in rs]), 3) if rs else None,
                    "expectancy_r": _r(np.mean(rs), 3) if rs else None},
        "risk": {**s.risk.snapshot(), "utilization": util, "limits": s.S.risk.__dict__},
        "positions": positions,
        "scanner": sorted(s.scanner.values(), key=lambda r: (r["state"] != "ENTER", r["state"] != "WATCH", -abs(r.get("change_pct") or 0))),
        "funnel": s.funnel,
        "decisions": list(s.latest_decisions)[-40:][::-1],
        "research": sorted([*s.J.recent["research"], *s.J.recent["anomaly"], *[{**x, "type": "thesis"} for x in s.J.recent["thesis"]],
                            *[{**x, "type": "risk"} for x in s.J.recent["risk"]]], key=lambda x: -x["ts"])[:60],
        "stops": list(s.J.recent["stop"])[-20:][::-1],
        "orders": list(s.J.recent["order"])[-25:][::-1],
        "trades": closed,
        "equity_curve": day_curve,
        "health": {
            "data": s.dq.feed_health(now), "bus_counts": dict(s.bus.counts), "handler_errors": s.bus.errors,
            "engine_latency_ms_p50": _r(np.percentile(lat, 50), 3) if lat else None,
            "engine_latency_ms_p95": _r(np.percentile(lat, 95), 3) if lat else None,
            "broker": {"name": s.broker.name, "connected": s.broker.connected},
            "llm": {"enabled": bool(s.llm and s.llm.enabled), "failures": s.llm.failures if s.llm else 0},
            "knowledge_base": {"analogues": len(s.analogues.items), "pending_labels": len(s.labeler.pending)},
            "model_monitor": s.drift.snapshot(),
            "last_bar_age_s": _r(now - max(s.last_bar_ts.values()), 1) if s.last_bar_ts else None,
        },
        "backtest": rt.backtest_report,
    }
