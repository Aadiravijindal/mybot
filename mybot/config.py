"""Configuration.

Risk limits are loaded once at start-up into frozen dataclasses. Nothing at runtime
(including any model or LLM output) can modify them; changing a limit requires a
config change + restart, which is recorded in the audit log.
"""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "default.toml"


@dataclass(frozen=True)
class RiskLimits:
    risk_per_trade_pct: float = 0.25        # % of equity risked (entry->stop) per trade
    max_position_notional_pct: float = 20.0  # % of equity in one name
    max_gross_exposure_pct: float = 100.0
    max_sector_exposure_pct: float = 40.0
    max_open_positions: int = 5
    max_daily_loss_pct: float = 1.5          # halt new entries + flatten
    max_drawdown_pct: float = 6.0            # from equity high-water mark -> halt
    max_trades_per_day: int = 20
    max_orders_per_minute: int = 10
    max_spread_bps: float = 25.0
    max_slippage_bps: float = 30.0           # realized; breach -> reduce-only
    max_consecutive_losses: int = 5          # -> reduce-only for the day
    min_price: float = 5.0
    min_avg_dollar_volume: float = 2_000_000  # per minute-bar average * 390
    max_participation_pct: float = 1.0       # of expected volume over holding period
    emergency_stop_pct: float = 3.0          # hard per-position loss cap vs entry
    max_data_staleness_s: float = 180.0
    max_clock_drift_s: float = 2.0
    flatten_before_close_min: int = 10
    allow_short: bool = True
    max_leverage: float = 1.0


@dataclass(frozen=True)
class CostModel:
    commission_per_share: float = 0.0035
    min_commission: float = 0.35
    slippage_bps: float = 2.0         # base, on top of half-spread
    impact_coef: float = 0.1          # square-root impact: coef * sigma_daily * sqrt(q / ADV)
    latency_s: float = 0.25


@dataclass(frozen=True)
class StrategyParams:
    min_ev_r: float = 0.10            # conservative EV (in R) required to enter
    min_prob_lcb: float = 0.35        # lower confidence bound on win prob
    min_reward_risk: float = 1.2
    min_analogues: int = 20
    atr_stop_mult: float = 1.5
    trail_atr_mult: float = 2.5
    breakeven_at_r: float = 1.0
    partial_at_r: float = 1.5
    partial_fraction: float = 0.5
    max_hold_bars: int = 90           # 1m bars
    warmup_bars: int = 40
    no_entry_first_min: int = 5
    no_entry_last_min: int = 30
    event_blackout_min: int = 15      # no new entries this close to a high-importance event


@dataclass(frozen=True)
class Settings:
    mode: str = "paper"               # paper | live  (live needs extra confirmation)
    data_source: str = "synthetic"    # synthetic | alpaca | csv
    symbols: tuple[str, ...] = ("AAPL", "MSFT", "NVDA", "AMD", "JPM", "XOM", "TSLA", "META")
    market_proxy: str = "SPY"
    sectors: dict = field(default_factory=lambda: {
        "AAPL": "TECH", "MSFT": "TECH", "NVDA": "SEMI", "AMD": "SEMI",
        "JPM": "FIN", "XOM": "ENERGY", "TSLA": "AUTO", "META": "TECH", "SPY": "INDEX",
    })
    starting_equity: float = 100_000.0
    timezone: str = "America/New_York"
    session_open: str = "09:30"
    session_close: str = "16:00"
    data_dir: str = "var"
    replay_speed: float = 60.0        # synthetic: simulated seconds per wall second
    host: str = "127.0.0.1"
    port: int = 8000
    risk: RiskLimits = field(default_factory=RiskLimits)
    costs: CostModel = field(default_factory=CostModel)
    strategy: StrategyParams = field(default_factory=StrategyParams)


def _build(cls, data: dict):
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**data)


def load_settings(path: str | os.PathLike | None = None) -> Settings:
    path = Path(path or os.environ.get("MYBOT_CONFIG", DEFAULT_PATH))
    raw: dict = {}
    if path.exists():
        raw = tomllib.loads(path.read_text())
    risk = _build(RiskLimits, raw.pop("risk", {}))
    costs = _build(CostModel, raw.pop("costs", {}))
    strat = _build(StrategyParams, raw.pop("strategy", {}))
    if "symbols" in raw:
        raw["symbols"] = tuple(raw["symbols"])
    s = _build(Settings, {**raw, "risk": risk, "costs": costs, "strategy": strat})
    # Environment overrides (non-risk only).
    env = {
        "mode": os.environ.get("MYBOT_MODE"),
        "data_source": os.environ.get("MYBOT_DATA_SOURCE"),
        "host": os.environ.get("MYBOT_HOST"),
    }
    s = replace(s, **{k: v for k, v in env.items() if v})
    if os.environ.get("MYBOT_PORT"):
        s = replace(s, port=int(os.environ["MYBOT_PORT"]))
    if s.mode not in ("paper", "live"):
        raise ValueError("mode must be 'paper' or 'live'")
    return s
