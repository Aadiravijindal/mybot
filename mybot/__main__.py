"""Command line entry point.

  python -m mybot                 # run engine + dashboard (paper, simulated market by default)
  python -m mybot backtest --days 30
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
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--train", type=int, default=20)
    ap.add_argument("--test", type=int, default=5)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--bootstrap-days", type=int, default=20)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from .config import load_settings
    S = load_settings(a.config)

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
        from .data.synthetic import SyntheticMarket
        m = SyntheticMarket(list(S.symbols), S.market_proxy, seed=a.seed)
        days = m.trading_days(datetime(2025, 1, 6), a.days)
        first = m.day_events(days[0])
        sysm = build_system(S, S.data_dir, first[0][0] - 1)
        replay(sysm, first)
        for d in days[1:]:
            replay(sysm, m.day_events(d))
        met = metrics(sysm)
        r = np.array([t.r for t in sysm.book.closed])
        print(json.dumps({"data": "SYNTHETIC -- plumbing test, not evidence of edge", "metrics": met.to_dict(),
                          "monte_carlo": monte_carlo(r, S.risk.risk_per_trade_pct),
                          "deflated_sharpe_prob": deflated_sharpe(met.sharpe_daily / 252 ** 0.5, a.days, 4),
                          "funnel": sysm.funnel}, indent=2, default=str))
    elif a.cmd == "walkforward":
        from .backtest.walkforward import walk_forward_synthetic
        print(json.dumps(walk_forward_synthetic(S, datetime(2025, 1, 6), a.train, a.test, a.folds), indent=2))
    elif a.cmd == "research":
        from .audit.journal import Journal
        from .learning.research import walk_forward_ablation
        j = Journal(S.data_dir)
        print(json.dumps(walk_forward_ablation(j.load_outcomes()), indent=2))
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
