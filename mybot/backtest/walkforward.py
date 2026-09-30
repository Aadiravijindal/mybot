"""Walk-forward evaluation: knowledge base built ONLY from the past.

For each fold, the system starts with analogues learned from days strictly before
the test window, then trades the test window. Only out-of-sample results are
reported. (The engine's analogue store is itself point-in-time, so this is a
second, coarser guard plus the honest reporting format.)
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from ..config import Settings
from ..data.synthetic import SyntheticMarket
from .engine import build_system, metrics, monte_carlo, replay


def walk_forward_synthetic(settings: Settings, start: datetime, train_days: int, test_days: int, folds: int,
                           seed: int = 11) -> dict:
    mkt = SyntheticMarket(list(settings.symbols), settings.market_proxy, seed=seed)
    days = mkt.trading_days(start, train_days + test_days * folds)
    all_events = [mkt.day_events(d) for d in days]
    out = []
    for k in range(folds):
        tr_end = train_days + k * test_days
        sysm = build_system(settings, None, all_events[0][0][0] - 1)
        # "train" = run the pipeline over past days (labels analogues); trades there are discarded.
        for ev in all_events[:tr_end]:
            replay(sysm, ev)
        n_train_trades = len(sysm.book.closed)
        eq0 = sysm.book.equity()
        for ev in all_events[tr_end:tr_end + test_days]:
            replay(sysm, ev)
        oos = sysm.book.closed[n_train_trades:]
        r = np.array([t.r for t in oos])
        out.append({"fold": k, "oos_trades": len(oos), "oos_expectancy_r": float(r.mean()) if len(r) else 0.0,
                    "oos_win_rate": float((r > 0).mean()) if len(r) else 0.0,
                    "oos_return_pct": (sysm.book.equity() / eq0 - 1) * 100})
    rs = [f["oos_expectancy_r"] for f in out if f["oos_trades"]]
    return {"folds": out, "mean_oos_expectancy_r": float(np.mean(rs)) if rs else 0.0,
            "positive_folds": sum(1 for x in rs if x > 0), "n_folds": len(out)}
