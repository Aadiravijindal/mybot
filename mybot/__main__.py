"""Command line entry point.

  python -m mybot                          # engine + dashboard: PAPER on a simulated market
  python -m mybot --data alpaca            # PAPER on real Alpaca data + Alpaca paper account
  python -m mybot --data alpaca --mode live  # REAL MONEY (also needs MYBOT_LIVE_CONFIRM)
  python -m mybot backtest --days 30
  python -m mybot backtest --source csv --csv data/SPY.csv:SPY
  python -m mybot backtest --source alpaca --days 30
  python -m mybot walkforward --train 20 --test 5 --folds 4
  python -m mybot research        # feature ablation + calibration report from the knowledge base
  python -m mybot verify-audit    # verify the hash-chained audit journal
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="mybot")
    ap.add_argument("cmd", nargs="?", default="run", choices=["run", "backtest", "walkforward", "research", "verify-audit"])
    ap.add_argument("--config")
    ap.add_argument("--mode", choices=["paper", "live"], help="override config mode")
    ap.add_argument("--data", choices=["synthetic", "alpaca"], help="override config data_source")
    ap.add_argument("--source", choices=["synthetic", "csv", "alpaca"], default="synthetic", help="backtest data")
    ap.add_argument("--csv", action="append", default=[], help="backtest: CSV file, optionally path:SYMBOL (repeatable)")
    ap.add_argument("--symbols", help="comma-separated universe override")
    ap.add_argument("--proxy", help="market proxy symbol override (not traded)")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--train", type=int, default=20)
    ap.add_argument("--test", type=int, default=5)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--bootstrap-days", type=int, default=20)
    ap.add_argument("--kb", default=None, help="research: knowledge-base partition (synthetic | real)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from dataclasses import replace
    from .config import load_settings
    S = load_settings(a.config)
    over = {}
    if a.mode:
        over["mode"] = a.mode
    if a.data:
        over["data_source"] = a.data
    if a.symbols:
        over["symbols"] = tuple(x.strip().upper() for x in a.symbols.split(",") if x.strip())
    if a.proxy:
        over["market_proxy"] = a.proxy.upper()
    S = replace(S, **over)
    if S.mode not in ("paper", "live"):
        raise SystemExit("mode must be paper or live")

    if a.cmd == "run":
        import uvicorn
        from .api.server import create_app
        from .runtime import Runtime
        rt = Runtime(S, bootstrap_days=a.bootstrap_days)
        app = create_app(rt)
        print(f"\n  mybot dashboard:  http://{S.host}:{S.port}\n  mode: {rt.mode_label}\n"
              f"  admin token (needed to RESUME after a halt): {app.state.token}\n", flush=True)
        uvicorn.run(app, host=S.host, port=S.port, log_level="warning")
    elif a.cmd == "backtest":
        import numpy as np
        from .backtest.engine import build_system, deflated_sharpe, metrics, monte_carlo, replay
        if a.source == "synthetic":
            from .data.synthetic import SyntheticMarket
            m = SyntheticMarket(list(S.symbols), S.market_proxy, seed=a.seed)
            days = m.trading_days(datetime(2025, 1, 6), a.days)
            first = m.day_events(days[0])
            sysm = build_system(S, S.data_dir, first[0][0] - 1)
            replay(sysm, first)
            for d in days[1:]:
                replay(sysm, m.day_events(d))
            label, n_days = "SYNTHETIC -- plumbing test, not evidence of edge", a.days
        else:
            if a.source == "csv":
                from .data.csv_loader import csv_events
                if not a.csv:
                    raise SystemExit("--source csv needs at least one --csv FILE[:SYMBOL]")
                events, bar_s = csv_events(a.csv, S.timezone)
                label = f"CSV ({', '.join(a.csv)}), {bar_s}s bars, bars only"
            else:
                import asyncio
                from .data.alpaca import AlpacaData
                feed = AlpacaData(list(dict.fromkeys(list(S.symbols) + [S.market_proxy])))
                events, bar_s = asyncio.run(feed.history_events(a.days)), 60
                label = "REAL Alpaca 1-minute bars, bars only"
            S = replace(S, bar_seconds=bar_s, data_source="csv" if a.source == "csv" else "alpaca")
            sysm = build_system(S, S.data_dir, events[0][0] - 1 if events else 0.0)
            replay(sysm, events)
            n_days = len({sysm.session.day_key(e[0]) for e in events})
        met = metrics(sysm)
        r = np.array([t.r for t in sysm.book.closed])
        print(json.dumps({"data": label, "days": n_days, "metrics": met.to_dict(),
                          "monte_carlo": monte_carlo(r, S.risk.risk_per_trade_pct),
                          "deflated_sharpe_prob": deflated_sharpe(met.sharpe_daily / 252 ** 0.5, n_days, 4),
                          "funnel": sysm.funnel, "risk_state": sysm.risk.state.value,
                          "handler_errors": sysm.bus.errors,
                          "records": sysm.dq.feed_health(sysm.clock.now())}, indent=2, default=str))
    elif a.cmd == "walkforward":
        from .backtest.walkforward import walk_forward_synthetic
        print(json.dumps(walk_forward_synthetic(S, datetime(2025, 1, 6), a.train, a.test, a.folds), indent=2))
    elif a.cmd == "research":
        from .audit.journal import Journal
        from .learning.research import walk_forward_ablation
        j = Journal(S.data_dir)
        print(json.dumps(walk_forward_ablation(j.load_outcomes(a.kb)), indent=2))
    elif a.cmd == "verify-audit":
        from pathlib import Path
        from .audit.journal import verify_chain
        p = Path(S.data_dir) / "audit" / "journal.jsonl"
        if not p.exists():
            print("no journal yet"); return
        ok, n = verify_chain(p)
        print(f"audit chain {'OK' if ok else 'BROKEN'} ({n} records verified)")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
