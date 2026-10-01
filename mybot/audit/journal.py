"""Audit journal + research knowledge base.

* Append-only JSONL, hash-chained (each record carries the SHA-256 of the previous
  one), so after-the-fact edits are detectable (``verify_chain``).
* SQLite index for fast lookup: "why did we do X at 10:37:14?"
* Knowledge base tables: labelled outcomes (analogues), news + measured impact,
  closed trades with attribution, model registry.

Production swaps SQLite for Postgres(+TimescaleDB) -- see ARCHITECTURE.md (D).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections import deque
from pathlib import Path

import numpy as np


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if hasattr(o, "value"):
        return o.value
    if hasattr(o, "__dict__"):
        return o.__dict__
    return str(o)


def dumps(o) -> str:
    return json.dumps(o, default=_default, separators=(",", ":"), sort_keys=True)


class Journal:
    def __init__(self, data_dir: str | None):
        self.lock = threading.Lock()
        self.recent: dict[str, deque] = {k: deque(maxlen=300) for k in ("decision", "research", "risk", "order", "stop", "thesis", "trade", "anomaly")}
        self.prev_hash = "0" * 64
        if data_dir:
            d = Path(data_dir)
            (d / "audit").mkdir(parents=True, exist_ok=True)
            self.path = d / "audit" / "journal.jsonl"
            self.db = sqlite3.connect(d / "mybot.sqlite", check_same_thread=False)
            if self.path.exists():
                with self.path.open("rb") as fh:
                    last = None
                    for last in fh:
                        pass
                    if last:
                        self.prev_hash = json.loads(last)["hash"]
        else:
            self.path = None
            self.db = sqlite3.connect(":memory:", check_same_thread=False)
        self._schema()

    def _schema(self):
        c = self.db
        c.execute("CREATE TABLE IF NOT EXISTS decisions (id TEXT PRIMARY KEY, ts REAL, symbol TEXT, action TEXT, record TEXT)")
        c.execute("CREATE INDEX IF NOT EXISTS ix_dec_sym_ts ON decisions(symbol, ts)")
        c.execute("CREATE TABLE IF NOT EXISTS trades (decision_id TEXT, symbol TEXT, entry_ts REAL, exit_ts REAL, pnl REAL, r REAL, record TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS outcomes (setup TEXT, side INT, start_ts REAL, outcome_ts REAL, r REAL, mae REAL, mfe REAL, exit TEXT, regime TEXT, vec TEXT, source TEXT)")
        cols = {r[1] for r in c.execute("PRAGMA table_info(outcomes)")}
        if "source" not in cols:   # migrate databases created before source tagging
            c.execute("ALTER TABLE outcomes ADD COLUMN source TEXT DEFAULT 'synthetic'")
        c.execute("CREATE TABLE IF NOT EXISTS news (id TEXT PRIMARY KEY, ts REAL, avail_ts REAL, symbols TEXT, event_type TEXT, record TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS models (name TEXT, version TEXT, status TEXT, created_ts REAL, metrics TEXT)")
        c.commit()

    def write(self, kind: str, ts: float, payload: dict) -> None:
        with self.lock:
            rec = {"kind": kind, "ts": ts, "prev": self.prev_hash, "payload": payload}
            body = dumps(rec)
            h = hashlib.sha256(body.encode()).hexdigest()
            self.prev_hash = h
            if self.path:
                with self.path.open("a") as fh:
                    fh.write(dumps({**rec, "hash": h}) + "\n")
            if kind in self.recent:
                self.recent[kind].append({"ts": ts, **payload})
            if kind == "decision":
                self.db.execute("INSERT OR REPLACE INTO decisions VALUES (?,?,?,?,?)",
                                (payload["id"], ts, payload.get("symbol"), payload.get("state"), dumps(payload)))
            elif kind == "trade":
                self.db.execute("INSERT INTO trades VALUES (?,?,?,?,?,?,?)",
                                (payload["decision_id"], payload["symbol"], payload["entry_ts"], payload["exit_ts"],
                                 payload["pnl"], payload["r"], dumps(payload)))
            elif kind == "research" and payload.get("type") == "news":
                self.db.execute("INSERT OR REPLACE INTO news VALUES (?,?,?,?,?,?)",
                                (payload["id"], payload["ts"], payload["avail_ts"], ",".join(payload["symbols"]),
                                 payload["event_type"], dumps(payload)))
            self.db.commit()

    def outcome(self, o, source: str) -> None:
        """Store a labelled outcome, tagged with the data source it was learned from.
        Analogues learned on simulated data must never inform decisions on real data."""
        with self.lock:
            self.db.execute("INSERT INTO outcomes (setup, side, start_ts, outcome_ts, r, mae, mfe, exit, regime, vec, source) "
                            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                            (o.setup, o.side, o.start_ts, o.outcome_ts, o.r, o.mae_r, o.mfe_r, o.exit, o.regime, dumps(o.vec), source))
            self.db.commit()

    def load_outcomes(self, source: str | None = None):
        q = "SELECT setup, side, start_ts, outcome_ts, r, mae, mfe, exit, regime, vec FROM outcomes"
        with self.lock:
            if source is None:
                return self.db.execute(q).fetchall()
            return self.db.execute(q + " WHERE source = ?", (source,)).fetchall()

    def decision_at(self, symbol: str, ts: float) -> dict | None:
        """Reconstruct: the latest decision for ``symbol`` at or before ``ts``."""
        with self.lock:
            row = self.db.execute("SELECT record FROM decisions WHERE symbol=? AND ts<=? ORDER BY ts DESC LIMIT 1",
                                  (symbol, ts)).fetchone()
        return json.loads(row[0]) if row else None

    def decisions(self, symbol: str | None = None, limit: int = 50) -> list[dict]:
        with self.lock:
            if symbol:
                rows = self.db.execute("SELECT record FROM decisions WHERE symbol=? ORDER BY ts DESC LIMIT ?", (symbol, limit))
            else:
                rows = self.db.execute("SELECT record FROM decisions ORDER BY ts DESC LIMIT ?", (limit,))
            return [json.loads(r[0]) for r in rows.fetchall()]


def verify_chain(path: str | Path) -> tuple[bool, int]:
    """Returns (ok, n_records). Detects any edit/deletion inside the journal."""
    prev, n = "0" * 64, 0
    with open(path) as fh:
        for line in fh:
            rec = json.loads(line)
            h = rec.pop("hash")
            if rec["prev"] != prev or hashlib.sha256(dumps(rec).encode()).hexdigest() != h:
                return False, n
            prev, n = h, n + 1
    return True, n
