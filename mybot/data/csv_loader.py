"""Load OHLCV bars from CSV files (vendor exports, yfinance dumps, your own data).

Accepted columns (case-insensitive): a timestamp column named one of
datetime/timestamp/time/date/t, and open/high/low/close/volume (o/h/l/c/v also
work). An optional symbol column; otherwise the symbol comes from the
``path:SYMBOL`` argument or the file name. Timestamps are bar START times; naive
timestamps are interpreted in the exchange time zone.

The bar size is inferred (median spacing) and must be the same for every file.
Each bar becomes available to the engine only at its CLOSE (start + bar size),
which is what makes CSV backtests point-in-time correct.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from statistics import median
from zoneinfo import ZoneInfo

from ..core.events import Topic
from ..core.types import Bar

_TS = ("datetime", "timestamp", "time", "date", "t")
_ALIASES = {"open": ("open", "o"), "high": ("high", "h"), "low": ("low", "l"), "close": ("close", "c"),
            "volume": ("volume", "v", "vol")}


def _parse_ts(s: str, tz: ZoneInfo) -> float:
    s = s.strip()
    if s.isdigit():                       # epoch seconds or milliseconds
        v = int(s)
        return v / 1000 if v > 10**11 else float(v)
    d = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if d.tzinfo is None:
        d = d.replace(tzinfo=tz)
    return d.timestamp()


def load_csv(spec: str, tz: str = "America/New_York") -> tuple[list[Bar], int]:
    path, _, sym = spec.partition(":")
    p = Path(path)
    zone = ZoneInfo(tz)
    with p.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{p}: empty file")
    cols = {c.lower().strip(): c for c in rows[0]}
    tcol = next((cols[c] for c in _TS if c in cols), None)
    if tcol is None:
        raise ValueError(f"{p}: no timestamp column (expected one of {_TS})")
    m = {}
    for k, al in _ALIASES.items():
        m[k] = next((cols[a] for a in al if a in cols), None)
        if m[k] is None:
            raise ValueError(f"{p}: missing column {k}")
    scol = cols.get("symbol") or cols.get("ticker")
    default_sym = (sym or p.stem).upper()
    parsed = []
    for r in rows:
        try:
            t0 = _parse_ts(r[tcol], zone)
            o, h, l, c = (float(r[m[k]]) for k in ("open", "high", "low", "close"))
            v = float(r[m["volume"]] or 0)
        except (ValueError, TypeError):
            continue   # skip malformed rows; the DQ engine handles the rest
        parsed.append(((r[scol].upper() if scol else default_sym), t0, o, h, l, c, v))
    parsed.sort(key=lambda x: (x[0], x[1]))
    gaps = [b[1] - a[1] for a, b in zip(parsed, parsed[1:]) if a[0] == b[0] and b[1] > a[1]]
    if not gaps:
        raise ValueError(f"{p}: need at least two bars")
    bar_s = int(round(median(gaps)))
    bars = [Bar(s, t0, t0 + bar_s, o, h, l, c, v, f"csv:{p.name}", t0 + bar_s, bar_s) for s, t0, o, h, l, c, v in parsed]
    return bars, bar_s


def csv_events(specs: list[str], tz: str = "America/New_York") -> tuple[list, int]:
    allbars, sizes = [], set()
    for spec in specs:
        bars, bs = load_csv(spec, tz)
        allbars += bars
        sizes.add(bs)
    if len(sizes) != 1:
        raise ValueError(f"all CSV files must share one bar size, got {sorted(sizes)} seconds")
    allbars.sort(key=lambda b: (b.avail_ts, b.symbol))
    return [(b.avail_ts, Topic.BAR, b) for b in allbars], sizes.pop()
