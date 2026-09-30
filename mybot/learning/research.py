"""Controlled learning: calibration, feature ablation, gated model promotion.

Nothing here runs inside the trading loop. The pipeline is:

  live/backtest outcomes -> error analysis -> hypothesis -> walk-forward test
  -> out-of-sample gate -> paper gate -> human approval -> promoted model

A model artifact is a JSON file with status CANDIDATE | PAPER | LIVE | RETIRED.
The engine only ever loads the LIVE (or, in paper mode, PAPER) artifact.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..signals.analogues import FEATURES


# ---------------------------------------------------------------- calibration
class IsotonicCalibrator:
    """Pool-adjacent-violators isotonic regression p_raw -> p_calibrated."""

    def __init__(self, xs: list[float] | None = None, ys: list[float] | None = None):
        self.xs, self.ys = xs or [], ys or []

    def fit(self, p: np.ndarray, y: np.ndarray) -> "IsotonicCalibrator":
        o = np.argsort(p)
        p, y = np.asarray(p)[o], np.asarray(y, float)[o]
        blocks = [[yy, 1.0, pp] for yy, pp in zip(y, p)]  # mean, weight, x
        i = 0
        while i < len(blocks) - 1:
            if blocks[i][0] > blocks[i + 1][0]:
                m1, w1, x1 = blocks[i]
                m2, w2, x2 = blocks[i + 1]
                blocks[i] = [(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2, (x1 * w1 + x2 * w2) / (w1 + w2)]
                del blocks[i + 1]
                i = max(i - 1, 0)
            else:
                i += 1
        self.xs = [b[2] for b in blocks]
        self.ys = [min(0.95, max(0.05, b[0])) for b in blocks]  # never certain
        return self

    def __call__(self, p: float) -> float:
        if not self.xs:
            return p
        return float(np.interp(p, self.xs, self.ys))


def reliability_table(p: np.ndarray, y: np.ndarray, bins: int = 5) -> list[dict]:
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b)
        if m.sum():
            out.append({"bin": f"{a:.1f}-{b:.1f}", "n": int(m.sum()), "pred": float(p[m].mean()), "actual": float(y[m].mean())})
    return out


# ---------------------------------------------------------------- models
def _logreg(X: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 400, lr: float = 0.1) -> np.ndarray:
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Xb @ w))
        g = Xb.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]] / len(y)
        w -= lr * g
    return w


def _logloss(w, X, y):
    p = 1 / (1 + np.exp(-(np.hstack([np.ones((len(X), 1)), X]) @ w)))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def walk_forward_ablation(rows: list[tuple], folds: int = 4) -> dict:
    """rows: (setup, side, start_ts, outcome_ts, r, mae, mfe, exit, regime, vec_json).

    Time-ordered expanding-window CV. Reports OOS log-loss of: base rate, full model,
    and full model minus each feature. A feature whose removal does not hurt OOS
    log-loss is not earning its place.
    """
    rows = sorted(rows, key=lambda r: r[2])
    if len(rows) < 100:
        return {"note": f"only {len(rows)} labelled outcomes; need >=100 for a meaningful test"}
    X = np.array([json.loads(r[9]) for r in rows], float)
    y = np.array([1.0 if r[4] > 0 else 0.0 for r in rows])
    n = len(rows)
    cuts = [int(n * (i + 1) / (folds + 1)) for i in range(folds + 1)]
    res = {"base": [], "full": [], **{f"-{f}": [] for f in FEATURES}}
    for i in range(folds):
        tr, te = slice(0, cuts[i]), slice(cuts[i], cuts[i + 1])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        br = y[tr].mean()
        res["base"].append(float(-(y[te] * math.log(br + 1e-9) + (1 - y[te]) * math.log(1 - br + 1e-9)).mean()))
        w = _logreg(Xtr, y[tr])
        res["full"].append(_logloss(w, Xte, y[te]))
        for j, f in enumerate(FEATURES):
            keep = [k for k in range(X.shape[1]) if k != j]
            wj = _logreg(Xtr[:, keep], y[tr])
            res[f"-{f}"].append(_logloss(wj, Xte[:, keep], y[te]))
    mean = {k: float(np.mean(v)) for k, v in res.items()}
    contrib = {f: round(mean[f"-{f}"] - mean["full"], 5) for f in FEATURES}
    return {"n": n, "oos_logloss_base": mean["base"], "oos_logloss_model": mean["full"],
            "model_beats_base": mean["full"] < mean["base"],
            "feature_contribution": dict(sorted(contrib.items(), key=lambda kv: -kv[1])),
            "interpretation": "positive contribution = removing the feature worsens OOS log-loss (feature is useful)"}


# ---------------------------------------------------------------- registry
PROMOTION_GATES = {
    "CANDIDATE->PAPER": {"min_oos_trades": 100, "min_oos_expectancy_r": 0.05, "min_positive_folds_frac": 0.6,
                         "min_deflated_sharpe_prob": 0.9, "max_calibration_gap": 0.07},
    "PAPER->LIVE": {"min_paper_days": 30, "min_paper_trades": 60, "paper_vs_backtest_expectancy_ratio": 0.5,
                    "max_slippage_vs_model_ratio": 1.5, "zero_unexplained_incidents": True, "human_signoff": True},
    "LIVE->LIMITED->FULL": {"limited_capital_pct": 10, "limited_days": 20, "full_after": "review"},
}


@dataclass
class ModelArtifact:
    name: str
    version: str
    status: str = "CANDIDATE"
    created_ts: float = field(default_factory=time.time)
    evidence_weights: dict = field(default_factory=dict)
    calibration: dict = field(default_factory=dict)       # {"xs": [...], "ys": [...]}
    metrics: dict = field(default_factory=dict)
    history: list = field(default_factory=list)


class ModelRegistry:
    def __init__(self, root: str):
        self.root = Path(root) / "models"
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, m: ModelArtifact) -> Path:
        p = self.root / f"{m.name}-{m.version}.json"
        p.write_text(json.dumps(asdict(m), indent=2))
        return p

    def load_active(self, mode: str) -> ModelArtifact | None:
        want = {"live": ("LIVE",), "paper": ("LIVE", "PAPER")}[mode]
        best = None
        for p in self.root.glob("*.json"):
            d = json.loads(p.read_text())
            if d["status"] in want and (best is None or d["created_ts"] > best.created_ts):
                best = ModelArtifact(**d)
        return best

    def promote(self, m: ModelArtifact, to: str, evidence: dict, who: str) -> ModelArtifact:
        gate = f"{m.status}->{to}"
        rules = PROMOTION_GATES.get(gate)
        if rules is None:
            raise ValueError(f"no promotion path {gate}")
        failures = [k for k, v in rules.items() if k in evidence and not _passes(k, evidence[k], v)]
        missing = [k for k in rules if k not in evidence]
        if failures or missing:
            raise PermissionError(f"gate {gate} failed: {failures}; missing evidence: {missing}")
        m.history.append({"ts": time.time(), "from": m.status, "to": to, "by": who, "evidence": evidence})
        m.status = to
        self.save(m)
        return m


def _passes(k: str, got, need) -> bool:
    if isinstance(need, bool):
        return bool(got) == need
    if k.startswith("max_"):
        return got <= need
    return got >= need
