"""Research/Trading Orchestrator.

Wires every engine onto the event bus and runs the hierarchical funnel:

  L1 cheap filters -> L2 statistics/regime/anomaly -> L3 setups + analogues +
  forecast -> L4 research (news/catalyst/LLM) -> L5 adversarial -> L6 risk -> L7 execution

and, for open positions, the continuous monitoring loop (thesis, dynamic stops,
research-triggered reassessment). The same class runs in backtest (SimClock)
and live (WallClock).
"""
from __future__ import annotations

import logging
import math
import time
from collections import defaultdict, deque

from ..audit.journal import Journal
from ..config import Settings
from ..core.calendar import Session
from ..core.clock import Clock
from ..core.events import Event, EventBus, Topic
from ..core.types import Bar, NewsItem, Quote, ScheduledEvent, Side, Trade
from ..data.quality import DataQualityEngine
from ..execution.broker import Broker, Fill, Order, OrderStatus, OrderType
from ..features.engine import SymbolFeatures
from ..intel.catalysts import AnomalyDetector, CatalystBook, EventCalendar
from ..intel.llm import LLMResearcher
from ..intel.news import ImpactTracker, NewsIntel
from ..intel.regime import Regime, RegimeTracker
from ..monitoring.attribution import attribute
from ..monitoring.drift import DriftMonitor
from ..risk.engine import PortfolioView, RiskEngine, RiskState
from ..risk.sizing import size as size_position
from ..risk.stops import DynamicStopEngine
from ..risk.thesis import Thesis, ThesisState
from ..signals import adversarial, evidence as evmod
from ..signals.analogues import AnalogueStore, OutcomeLabeler
from ..signals.forecast import forecast
from ..signals.setups import Candidate, generate
from .portfolio import Book, ClosedTrade, Position

log = logging.getLogger(__name__)


class TradingSystem:
    def __init__(self, settings: Settings, clock: Clock, broker: Broker, journal: Journal,
                 live: bool = False, enable_llm: bool = True):
        self.S, self.clock, self.broker, self.J = settings, clock, broker, journal
        self.live = live
        self.session = Session(settings.timezone, settings.session_open, settings.session_close)
        self.bus = EventBus(clock)
        self.bus.on_error = lambda ev, exc: self.risk.record_component_error(f"{ev.topic.value}: {exc}", self.clock.now())
        self.bar_s = settings.bar_seconds
        # A feed of N-second bars is legitimately silent for N seconds between bars.
        self.staleness_limit = max(settings.risk.max_data_staleness_s, 2 * self.bar_s + 60)
        self.dq = DataQualityEngine(max_staleness_s=self.staleness_limit, bar_seconds=self.bar_s)
        self.allow_entries = True   # False during warm-up / history replay in live mode
        self.stream_start_ts = 0.0  # set by the runtime when live data streaming begins
        # Knowledge-base partition: simulated and real-data analogues never mix.
        self.kb_source = "synthetic" if settings.data_source == "synthetic" else "real"
        self.feat: dict[str, SymbolFeatures] = {}
        self.regimes: dict[str, RegimeTracker] = defaultdict(RegimeTracker)
        self.news = NewsIntel()
        self.impact = ImpactTracker()
        self.catalysts = CatalystBook()
        self.calendar = EventCalendar()
        self.anomaly = AnomalyDetector()
        self.analogues = AnalogueStore()
        self.labeler = OutcomeLabeler(self._cost_r)
        self.drift = DriftMonitor()
        self.risk = RiskEngine(settings.risk, settings.sectors)
        st = settings.strategy
        self.stops = DynamicStopEngine(st.atr_stop_mult,
                                       st.trail_atr_mult, st.breakeven_at_r, settings.risk.emergency_stop_pct)
        self.book = Book(settings.starting_equity)
        self.llm = LLMResearcher() if enable_llm else None
        self._llm_futures: list = []
        self.pending_entries: dict[str, dict] = {}     # order_id -> context
        self._own_cancels: set[str] = set()            # stop orders we cancelled ourselves
        self.exit_orders: dict[str, str] = {}          # order_id -> symbol
        self.cooldown: dict[tuple, float] = {}
        self.scanner: dict[str, dict] = {}
        self.funnel: dict[str, int] = {}
        self.latest_decisions: deque = deque(maxlen=100)
        self.last_bar_ts: dict[str, float] = {}
        self.market_ret_day = 0.0
        self.day: str | None = None
        self.reconcile_miss = 0
        self.stats = {"bars": 0, "quotes": 0, "decisions": 0, "latency_ms": deque(maxlen=500)}
        self.started = time.time()
        self.weights: dict[str, float] = {}
        self.calibrator = None
        self.mode_label = "BACKTEST" if not live else settings.mode.upper()

        b = self.bus
        b.subscribe(Topic.BAR, self._on_bar)
        b.subscribe(Topic.QUOTE, self._on_quote)
        b.subscribe(Topic.TRADE, self._on_trade)
        b.subscribe(Topic.NEWS, self._on_news)
        b.subscribe(Topic.SCHEDULE, lambda ev: self.calendar.add(ev.payload))
        broker.on_fill = self._on_fill
        broker.on_order = self._on_order

    # ================================================================ knowledge
    def load_knowledge(self, registry=None, mode: str = "paper") -> dict:
        """Load persisted analogues (research memory) and the active validated model."""
        import json as _json
        import numpy as _np
        from ..signals.analogues import Outcome
        n = 0
        for row in self.J.load_outcomes(self.kb_source):
            setup, side, st_, ot, r, mae, mfe, ex, regime, vec = row
            self.analogues.add(Outcome(setup, side, st_, ot, r, mae, mfe, 0, ex, _np.array(_json.loads(vec)), regime))
            n += 1
        model = None
        if registry is not None:
            model = registry.load_active(mode)
            if model is not None:
                from ..learning.research import IsotonicCalibrator
                self.weights = dict(model.evidence_weights)
                if model.calibration.get("xs"):
                    self.calibrator = IsotonicCalibrator(model.calibration["xs"], model.calibration["ys"])
        return {"analogues": n, "model": None if model is None else f"{model.name}-{model.version} ({model.status})"}

    # ================================================================ helpers
    def f(self, sym: str) -> SymbolFeatures:
        if sym not in self.feat:
            self.feat[sym] = SymbolFeatures(sym, self.session, self.bar_s)
        return self.feat[sym]

    def _cost_r(self, c: Candidate) -> float:
        spread_bps = c.features.get("spread_bps", 3.0)
        per_share = c.entry * (spread_bps / 2 + self.S.costs.slippage_bps) / 1e4 + self.S.costs.commission_per_share
        return 2 * per_share / c.risk_per_share if c.risk_per_share > 0 else 9.9

    def _pv(self) -> PortfolioView:
        return PortfolioView(self.book.equity(), self.book.day_start_equity, self.book.hwm,
                             {s: (p.side * p.qty, self.book.marks.get(s, p.avg_px)) for s, p in self.book.positions.items()},
                             self.S.sectors)

    def publish(self, topic: Topic, payload, symbol=None, ts=None):
        self.bus.publish(Event(topic, ts if ts is not None else self.clock.now(), payload, symbol))

    # ================================================================ market data
    def _on_quote(self, ev: Event) -> None:
        q: Quote = ev.payload
        now = self.clock.now()
        if not self.dq.check_quote(q, now):
            return
        self.stats["quotes"] += 1
        self.f(q.symbol).on_quote(q)
        if hasattr(self.broker, "on_quote"):
            self.broker.on_quote(q)
        self.book.mark(q.symbol, q.mid)

    def _on_trade(self, ev: Event) -> None:
        t: Trade = ev.payload
        if self.dq.check_trade(t, self.clock.now()):
            self.f(t.symbol).on_trade(t)

    def _on_bar(self, ev: Event) -> None:
        t0 = time.perf_counter()
        b: Bar = ev.payload
        now = self.clock.now()
        if not self.dq.check_bar(b, now):
            return
        self.stats["bars"] += 1
        day = self.session.day_key(b.ts)
        if day != self.day:
            self._new_day(day, now)
        self.risk.new_day(day, now)
        if hasattr(self.broker, "on_bar"):
            self.broker.on_bar(b)
        sf = self.f(b.symbol)
        fv = sf.on_bar(b)
        self.last_bar_ts[b.symbol] = b.end_ts
        self.drift.observe_features(fv, live=self.live)
        for o in self.labeler.on_bar(b.symbol, b.high, b.low, b.close, b.end_ts):
            self.analogues.add(o)
            self.J.outcome(o, self.kb_source)
        self.impact.on_price(b.symbol, b.end_ts, b.close)
        reg, changed = self.regimes[b.symbol].update(fv, now)
        if changed:
            self.J.write("research", now, {"type": "regime", "symbol": b.symbol, "regime": reg.value,
                                            "headline": f"{b.symbol} regime -> {reg.value}", "source": "regime_engine"})
        if b.symbol == self.S.market_proxy:
            self.market_ret_day = fv.get("ret_day", 0.0)
        self.book.mark(b.symbol, b.close if b.symbol not in self.dq.s or not self.dq.s[b.symbol].last_quote
                       else self.dq.s[b.symbol].last_quote.mid)

        cat, cat_items = self.catalysts.score(b.symbol, now)
        for a in self.anomaly.check(b.symbol, now, fv, news_recent=bool(cat_items)):
            self.J.write("anomaly", now, {"symbol": a.symbol, "kind": a.kind, "z": round(a.z, 1), "why": a.why})

        if b.symbol in self.book.positions:
            row = self.scanner.get(b.symbol)
            if row is not None:
                row.update(price=round(fv.get("close", 0), 2), change_pct=round(fv.get("ret_day", 0) * 100, 2),
                           rel_volume=round(fv.get("rel_volume", 0), 2), rsi=round(fv.get("rsi", 0), 1),
                           regime=reg.value, catalyst=round(cat, 2), stage="holding: position monitor", state="HOLD")
            self._manage_position(b, fv, reg, cat)
        elif b.symbol != self.S.market_proxy:
            self._scan(b.symbol, fv, reg, cat, cat_items, now)

        self._portfolio_checks(now)
        self.book.record_equity(now)
        self.stats["latency_ms"].append((time.perf_counter() - t0) * 1000)

    def _new_day(self, day: str, now: float) -> None:
        self.day = day
        self.book.day_start_equity = self.book.equity()
        self.J.write("risk", now, {"event": "new_session", "day": day, "equity": round(self.book.equity(), 2)})

    # ================================================================ news / research
    def _on_news(self, ev: Event) -> None:
        n: NewsItem = ev.payload
        now = self.clock.now()
        a = self.news.assess(n)
        self.J.write("research", now, {"type": "news", **a.to_dict()})
        self.catalysts.add(a)
        for s in n.symbols:
            px = self.book.marks.get(s)
            if px:
                self.impact.track(a, s, px)
            # Research never stops while a position is open: material news triggers
            # an immediate thesis reassessment (not waiting for the next bar).
            p = self.book.positions.get(s)
            if p and not a.is_duplicate and a.materiality > 0.2:
                sign = a.direction * p.side
                p.research.append({"ts": now, "headline": a.headline, "impact": "supports" if sign > 0 else "contradicts" if sign < 0 else "neutral",
                                   "materiality": a.materiality, "source": a.source})
                if sign < 0:
                    sf = self.f(s)
                    cat, _ = self.catalysts.score(s, now)
                    state, changed = p.thesis.update(sf.values, market_regime=self._mkt_regime(), catalyst=cat,
                                                     bars_held=p.bars_held, horizon=p.horizon, r_now=p.r_now(p.last_px or p.avg_px),
                                                     ts=now, new_contra=[f"news: {a.headline[:60]}"])
                    if changed:
                        self._thesis_changed(p, now)
        if self.llm and a.materiality > 0.3 and not a.is_duplicate:
            fut = self.llm.submit(n.headline, n.body, n.symbols)
            if fut:
                self._llm_futures.append((a, fut))

    def poll_research(self) -> None:
        """Collect finished LLM extractions (called from the live loop)."""
        keep = []
        for a, fut in self._llm_futures:
            if fut.done():
                res = fut.result()
                if res:
                    a.llm = res
                    self.J.write("research", self.clock.now(), {"type": "llm_extraction", "id": a.id, "symbols": a.symbols,
                                                                "headline": a.headline, "source": "claude", "result": res})
            else:
                keep.append((a, fut))
        self._llm_futures = keep

    # ================================================================ scanning / entry
    def _mkt_regime(self) -> str:
        return self.regimes[self.S.market_proxy].current.value

    def _scan(self, sym: str, fv: dict, reg: Regime, cat: float, cat_items: list, now: float) -> None:
        L, st = self.S.risk, self.S.strategy
        sf = self.feat[sym]
        dq = self.dq.quality(sym, now)
        row = {"symbol": sym, "price": round(fv.get("close", 0), 2), "change_pct": round(fv.get("ret_day", 0) * 100, 2),
               "rel_volume": round(fv.get("rel_volume", 0), 2), "rsi": round(fv.get("rsi", 0), 1),
               "atr_pct": round(fv.get("atr_pct", 0), 3), "regime": reg.value, "catalyst": round(cat, 2),
               "spread_bps": round(fv.get("spread_bps", 0), 1), "dq": round(dq.score, 2), "stage": "L1", "signal": "",
               "confidence": None, "ev_r": None, "state": "IGNORE"}
        self.scanner[sym] = row
        self.funnel["universe"] = len(self.scanner)
        # L1: cheap filters.
        if not sf.warm or sf.n < st.warmup_bars:
            row["stage"] = "warming up"; return
        if fv["close"] < L.min_price or fv.get("avg_dollar_vol_day", 0) < L.min_avg_dollar_volume:
            row["stage"] = "L1 fail: liquidity/price"; return
        mso, mtc = self.session.minutes_since_open(now), self.session.minutes_to_close(now)
        if mso < st.no_entry_first_min:
            row["stage"] = "L1: opening auction window"; return
        # L2/L3: hypotheses.
        cands = generate(sym, now, fv, reg, cat, st.atr_stop_mult, L.allow_short, self.bar_s)
        row["stage"] = "L2 scanned" if not cands else "L3 candidate"
        if not cands:
            return
        for c in cands:
            key = (sym, c.setup, c.side)
            if now - self.cooldown.get(key, 0) < max(600, 3 * self.bar_s):
                continue
            self.cooldown[key] = now
            self.labeler.add(c, reg.value)         # label every candidate (no selection bias)
            if not self.allow_entries:
                row["stage"] = "warm-up (history replay): labelling only"
                continue
            if sym in self.book.positions or any(ctx["c"].symbol == sym for ctx in self.pending_entries.values()):
                continue
            self._decide(c, fv, reg, cat, cat_items, dq, mtc, row, now)

    def _decide(self, c: Candidate, fv, reg: Regime, cat, cat_items, dq, mtc, row, now) -> None:
        L, st = self.S.risk, self.S.strategy
        self.stats["decisions"] += 1
        mkt = self._mkt_regime()
        rs = fv.get("ret_day", 0) - self.market_ret_day
        # L4 research: evidence + analogues.
        bull = evmod.gather(c, fv, mkt, rs, cat, dq.score, self.weights)
        an = self.analogues.query(c.setup, c.side.sign, fv, now)
        contra_news = [a for a in cat_items if a.direction * c.side.sign < 0]
        upcoming = self.calendar.upcoming(c.symbol, now, st.event_blackout_min * 60 + c.horizon_bars * self.bar_s, 2)
        # L5 adversarial.
        ch = adversarial.challenge(c, fv, market_regime=mkt, catalyst=cat, upcoming_events=upcoming, an=an,
                                   dq_ok=dq.ok, spread_bps=fv.get("spread_bps", 0.0), max_spread_bps=L.max_spread_bps,
                                   min_analogues=st.min_analogues, minutes_to_close=mtc,
                                   no_entry_last_min=st.no_entry_last_min, recent_contra_news=contra_news)
        logodds, per_group = evmod.combine(bull + ch.contra)
        cost_r = self._cost_r(c)
        tgt_r = c.r_of(c.targets[0])
        fc = forecast(logodds, an, tgt_r, cost_r, self.calibrator)
        w = self.drift.weight(c.setup)
        rr = tgt_r  # first target, in R (risk = entry->invalidation)
        # Decision logic.
        blockers = list(ch.vetoes)
        if w <= 0:
            blockers.append(f"setup {c.setup} disabled by model monitor")
        if fc.ev_r_conservative < st.min_ev_r:
            blockers.append(f"conservative EV {fc.ev_r_conservative:+.2f}R < {st.min_ev_r}R")
        if fc.p_win_lcb < st.min_prob_lcb:
            blockers.append(f"P(win) lower bound {fc.p_win_lcb:.2f} < {st.min_prob_lcb}")
        if rr < st.min_reward_risk:
            blockers.append(f"reward:risk {rr:.2f} < {st.min_reward_risk}")
        qty, risk_dollars, sizing = size_position(self.book.equity(), L.risk_per_trade_pct, c.entry, c.stop,
                                                  fc.p_win_lcb, fc.exp_win_r, fc.exp_loss_r, w)
        rd = None
        if not blockers:
            rd = self.risk.pre_trade(sym=c.symbol, side_sign=c.side.sign, qty=qty, price=c.entry, stop=c.stop,
                                     spread_bps=fv.get("spread_bps", 0.0), avg_dollar_vol=fv.get("avg_dollar_vol_day", 0.0),
                                     pv=self._pv(), ts=now)
            if not rd.approved:
                blockers.extend(f"RISK: {r}" for r in rd.reasons)
        state = "ENTER" if not blockers else ("WATCH" if fc.ev_r > 0 else "AVOID")
        did = f"d{int(now)}-{c.symbol}-{c.setup}-{c.side.value}"
        nxt = self.calendar.next_event(c.symbol, now)
        rec = {
            "id": did, "ts": now, "symbol": c.symbol, "state": state, "setup": c.setup, "side": c.side.value,
            "regime": reg.value, "market_regime": mkt, "thesis": c.rationale,
            "supporting_evidence": [e.to_dict() for e in bull if e.llr > 0],
            "contradicting_evidence": [e.to_dict() for e in bull if e.llr < 0] + [e.to_dict() for e in ch.contra],
            "vetoes": ch.vetoes, "blockers": blockers, "evidence_by_group": {k: round(v, 3) for k, v in per_group.items()},
            "catalysts": [{"headline": a.headline, "type": a.event_type, "dir": a.direction, "materiality": a.materiality,
                           "source": a.source} for a in cat_items[:5]],
            "analogues": None if an is None else {k: round(v, 3) if isinstance(v, float) else v for k, v in an.__dict__.items()},
            "forecast": fc.to_dict(), "expected_holding_min": round(c.horizon_bars * self.bar_s / 60),
            "entry_zone": [round(c.entry - 0.1 * fv["atr"], 2), round(c.entry + 0.1 * fv["atr"], 2)],
            "invalidation": round(c.stop, 2), "targets": [round(t, 2) for t in c.targets],
            "position_size": rd.qty if (rd and rd.approved) else 0, "risk_dollars": round(risk_dollars, 2),
            "sizing": sizing, "risk_notes": rd.reasons if rd else [], "model_weight": w,
            "data_quality": {"score": dq.score, "issues": dq.issues},
            "what_would_change": self._what_would_change(c, fc, ch, blockers),
            "next_event": None if nxt is None else {"name": nxt.name, "in_min": round((nxt.scheduled_ts - now) / 60)},
            "features": {k: round(v, 4) for k, v in fv.items() if isinstance(v, float) and math.isfinite(v)},
        }
        self.J.write("decision", now, rec)
        self.latest_decisions.append(rec)
        row.update(stage="L7 execute" if state == "ENTER" else ("L6 blocked" if state == "WATCH" else "L5 rejected"),
                   signal=f"{c.setup} {c.side.value}", confidence=fc.p_win, ev_r=fc.ev_r, state=state)
        self.funnel[state] = self.funnel.get(state, 0) + 1
        if state == "ENTER":
            self._enter(c, rd.qty, fc, bull + ch.contra, did, fv, now)

    def _what_would_change(self, c, fc, ch, blockers) -> list[str]:
        out = [f"price through {c.stop:.2f} invalidates the idea"]
        if ch.vetoes:
            out.append("vetoes clearing: " + "; ".join(ch.vetoes[:2]))
        if fc.ev_r_conservative < self.S.strategy.min_ev_r:
            out.append("more/better analogues or stronger independent evidence raising conservative EV")
        out.append("contradicting material news, or market regime flipping against the trade")
        return out

    def _enter(self, c: Candidate, qty: int, fc, ev_items, did, fv, now) -> None:
        q = self.dq.s[c.symbol].last_quote
        o = Order(c.symbol, c.side.sign, qty, OrderType.MARKET, now, "entry", decision_px=q.mid if q else c.entry)
        self.pending_entries[o.id] = {"c": c, "fc": fc.to_dict(), "ev": [e.to_dict() for e in ev_items], "did": did}
        self.risk.on_order_sent(now, opening=True)
        self.broker.submit(o)

    # ================================================================ position management
    def _manage_position(self, b: Bar, fv: dict, reg: Regime, cat: float) -> None:
        p = self.book.positions[b.symbol]
        now = self.clock.now()
        p.bars_held += 1
        if p.working_exit:
            return   # an exit/reduce order is in flight; never stack a second one
        if p.stop_order_id is None:
            # No resting stop (rejected, lost, or a previous exit failed): close, retrying every bar.
            p.exiting = False
            self._exit(p, p.qty, "unprotected position (no resting stop at broker): closing", now)
            return
        if p.exiting:
            return
        mkt = self._mkt_regime()
        r_now = p.r_now(b.close)
        state, changed = p.thesis.update(fv, market_regime=mkt, catalyst=cat, bars_held=p.bars_held,
                                         horizon=p.horizon, r_now=r_now, ts=now)
        if changed:
            self._thesis_changed(p, now)
            if p.exiting:
                return
        cps = p.avg_px * (self.S.costs.slippage_bps + fv.get("spread_bps", 2) / 2) / 1e4 + self.S.costs.commission_per_share
        q = self.dq.s[b.symbol].last_quote
        new_stop, why = self.stops.update(p.stop, high=b.high, low=b.low, close=b.close, atr=fv.get("atr", 0.01),
                                          swing_low=fv.get("swing_low"), swing_high=fv.get("swing_high"),
                                          thesis=state.value, cost_per_share=cps, spread=q.spread if q else 0.0, ts=now)
        if why:
            if p.stop_order_id:
                self.broker.replace_stop(p.stop_order_id, new_stop)
            self.J.write("stop", now, {"symbol": p.symbol, "stop": round(new_stop, 4), "reason": why,
                                        "components": p.stop.components, "r_now": round(r_now, 2)})
        mtc = self.session.minutes_to_close(now)
        st = self.S.strategy
        if mtc <= self.S.risk.flatten_before_close_min:
            self._exit(p, p.qty, "flatten: end of session", now)
        elif not p.partial_done and r_now >= st.partial_at_r and p.qty >= 2:
            p.partial_done = True
            self._exit(p, max(1, int(p.qty * st.partial_fraction)), "partial: target 1", now, full=False)
        elif p.bars_held >= p.horizon * 1.5 and r_now < 0.5:
            self._exit(p, p.qty, "time stop: thesis did not play out in expected window", now)
        elif len(p.targets) > 1 and (b.close - p.targets[-1]) * p.side >= 0:
            self._exit(p, p.qty, "target 2 reached", now)

    def _thesis_changed(self, p: Position, now: float) -> None:
        last = p.thesis.log[-1] if p.thesis.log else (now, p.thesis.state.value, p.thesis.score, "")
        self.J.write("thesis", now, {"symbol": p.symbol, "state": last[1], "score": last[2], "reason": last[3]})
        if p.thesis.state is ThesisState.INVALID and not p.exiting:
            p.thesis.state = ThesisState.EXIT
            self._exit(p, p.qty, f"thesis invalid: {last[3]}", now)

    def _exit(self, p: Position, qty: int, reason: str, now: float, full: bool = True) -> None:
        if qty <= 0 or p.working_exit:
            return
        # Release the shares held by the resting stop first. Otherwise a real broker
        # rejects the exit (shares reserved by the stop), or both orders fill and the
        # position flips to the other side.
        if p.stop_order_id and not self._cancel_stop(p):
            return      # the stop already filled: the position is closing via its fill
        if full:
            p.exiting = True
        o = Order(p.symbol, -p.side, min(qty, p.qty), OrderType.MARKET, now, "exit" if full else "reduce",
                  decision_px=self.book.marks.get(p.symbol, p.avg_px))
        self.exit_orders[o.id] = reason
        p.working_exit = o.id
        self.risk.on_order_sent(now, opening=False)
        self.broker.submit(o)

    def _cancel_stop(self, p: Position) -> bool:
        """Cancel the resting stop. Marked as ours BEFORE the broker call, because the
        cancel callback can fire synchronously and must not be mistaken for a lost stop."""
        oid = p.stop_order_id
        self._own_cancels.add(oid)
        if not self.broker.cancel(oid):
            self._own_cancels.discard(oid)
            return False
        if p.stop_order_id == oid:
            p.stop_order_id = None
        return True

    def _place_stop(self, p: Position, now: float) -> None:
        """(Re)place the broker-resident protective stop for the full current quantity."""
        if p.stop_order_id and not self._cancel_stop(p):
            return
        so = self.broker.submit(Order(p.symbol, -p.side, p.qty, OrderType.STOP, now, "stop", stop=p.stop.current))
        if so.status is not OrderStatus.REJECTED:
            p.stop_order_id = so.id
            p.stop_failures = 0
            return
        # The broker refused the protective stop (note: its callback fired before we knew
        # the id, so handle it here). A position must never sit unprotected: close it.
        p.stop_failures += 1
        self.risk.record_component_error(f"protective stop rejected: {p.symbol} ({so.reject_reason})", now)
        self.J.write("risk", now, {"event": "stop_lost", "state": self.risk.state.value,
                                    "reason": f"{p.symbol} stop rejected: {so.reject_reason}; closing position"})
        if not p.working_exit:
            self._exit(p, p.qty, "protective stop rejected by broker", now)

    # ================================================================ broker callbacks
    def _on_order(self, o: Order) -> None:
        now = self.clock.now()
        self.J.write("order", now, {"id": o.id, "symbol": o.symbol, "side": o.side, "qty": o.qty, "type": o.type.value,
                                     "tag": o.tag, "status": o.status.value, "filled": o.filled,
                                     "avg_px": round(o.avg_px, 4), "reason": o.reject_reason})
        if not o.done:
            return
        p = self.book.positions.get(o.symbol)
        if o.tag == "entry" and o.status is not OrderStatus.FILLED:
            self.pending_entries.pop(o.id, None)
        elif o.tag in ("exit", "reduce") and o.status is not OrderStatus.FILLED:
            # Exit did not (fully) happen: re-protect the remainder immediately and retry next bar.
            if p and p.working_exit == o.id:
                p.working_exit = None
                p.exiting = False
                if p.stop_order_id is None and p.stop_failures == 0:
                    self._place_stop(p, now)   # re-protect; if stops are being refused, the next bar retries the exit
            if o.status is OrderStatus.REJECTED:
                self.risk.record_component_error(f"exit order rejected: {o.symbol} ({o.reject_reason})", now)
        elif o.tag == "stop" and o.status is not OrderStatus.FILLED:
            if p and p.stop_order_id == o.id and o.id not in self._own_cancels:
                # The broker rejected or cancelled our protective stop. Never sit unprotected.
                p.stop_order_id = None
                self.risk.record_component_error(f"protective stop {o.status.value.lower()}: {o.symbol} ({o.reject_reason})", now)
                self.J.write("risk", now, {"event": "stop_lost", "state": self.risk.state.value,
                                            "reason": f"{o.symbol} stop {o.status.value}: {o.reject_reason}; exiting"})
                if o.status is OrderStatus.REJECTED:
                    self._exit(p, p.qty, "protective stop rejected by broker", now)
                else:
                    self._place_stop(p, now)

    def _on_fill(self, fl: Fill) -> None:
        now = self.clock.now()
        self.book.cash -= fl.side * fl.qty * fl.price + fl.fees
        if fl.tag == "entry":
            ctx = self.pending_entries.get(fl.order_id)
            if ctx is None:
                return
            c: Candidate = ctx["c"]
            p = self.book.positions.get(c.symbol)
            o = getattr(self.broker, "orders", {}).get(fl.order_id)
            if p is None:
                st = self.stops.init(c.side.sign, fl.price, c.stop, now)
                th = Thesis(c.rationale, c.side.sign, c.setup, c.stop, score=0.55 + 0.3 * ctx["fc"]["p_win"] - 0.15)
                th.log.append((now, th.state.value, th.score, "entered"))
                slip = (fl.price - o.decision_px) * c.side.sign / o.decision_px * 1e4 if o and o.decision_px else 0.0
                p = Position(c.symbol, c.side.sign, fl.qty, fl.price, now, c.setup, ctx["did"], th, st, c.targets,
                             c.horizon_bars, abs(fl.price - st.current), ctx["ev"], ctx["fc"], fees=fl.fees,
                             last_px=fl.price, decision_px=o.decision_px if o else fl.price, entry_slippage_bps=slip,
                             orig_qty=fl.qty)
                self.book.positions[c.symbol] = p
                self.drift.record_slippage(slip)
                self.risk.record_slippage(slip, now)
                self._place_stop(p, now)
            else:
                p.avg_px = (p.avg_px * p.qty + fl.price * fl.qty) / (p.qty + fl.qty)
                p.qty += fl.qty
                p.orig_qty += fl.qty
                p.fees += fl.fees
                self._place_stop(p, now)
            if o is not None and o.done:
                self.pending_entries.pop(fl.order_id, None)
            return
        # Exit-side fill (exit / reduce / stop / flatten).
        p = self.book.positions.get(fl.symbol)
        if p is None:
            return
        pnl = (fl.price - p.avg_px) * p.side * fl.qty
        p.realized += pnl
        p.fees += fl.fees
        p.qty -= fl.qty
        reason = self.exit_orders.get(fl.order_id, "stop hit" if fl.tag == "stop" else fl.tag)
        if fl.tag == "stop":
            reason = f"stop hit ({p.stop.history[-1][2] if p.stop.history else 'initial'} stop)"
        if p.qty < 0:   # over-fill: should be impossible; position state is now uncertain
            self.risk.halt(f"over-fill on {p.symbol}: book qty {p.qty}", now)
        if p.qty <= 0:
            self._close(p, fl, reason, now)
            return
        o = getattr(self.broker, "orders", {}).get(fl.order_id)
        if fl.tag in ("exit", "reduce") and (o is None or o.done):
            p.working_exit = None
            if fl.tag == "reduce":
                p.exiting = False
            self._place_stop(p, now)   # protect the remainder only once the exit order is finished
            self.J.write("stop", now, {"symbol": p.symbol, "stop": round(p.stop.current, 4), "reason": f"{reason}; stop re-placed for {p.qty}",
                                        "components": p.stop.components, "r_now": round(p.r_now(fl.price), 2)})
        elif fl.tag == "stop" and p.stop_order_id == fl.order_id and (o is None or o.done):
            p.stop_order_id = None
            self._place_stop(p, now)

    def _close(self, p: Position, fl: Fill, reason: str, now: float) -> None:
        if p.stop_order_id and p.stop_order_id != fl.order_id:
            self._cancel_stop(p)
        net = p.realized - p.fees
        r = net / (p.initial_risk_per_share * p.orig_qty) if p.initial_risk_per_share and p.orig_qty else 0.0
        ct = ClosedTrade(p.symbol, p.side, p.setup, p.entry_ts, now, p.orig_qty, p.avg_px, fl.price, round(net, 2),
                         round(r, 3), round(p.mae, 3), round(p.mfe, 3), p.bars_held, reason, p.decision_id, round(p.fees, 2),
                         round(p.entry_slippage_bps, 2), list(p.thesis.log), list(p.stop.history), p.entry_evidence, p.forecast)
        ct.attribution = attribute(ct)
        self.book.closed.append(ct)
        del self.book.positions[p.symbol]
        self.drift.record(p.setup, p.forecast.get("p_win", 0.5), p.forecast.get("ev_r", 0.0), r)
        self.risk.record_trade_result(net, now)
        rec = {k: v for k, v in ct.__dict__.items()}
        self.J.write("trade", now, rec)

    # ================================================================ portfolio / infra
    def _portfolio_checks(self, now: float) -> None:
        pv = self._pv()
        prev = self.risk.state
        self.risk.check_portfolio(pv, now)
        if self.risk.state is not prev:
            self.J.write("risk", now, {"event": "state_change", "state": self.risk.state.value,
                                        "reason": self.risk.reasons[-1][2] if self.risk.reasons else ""})
        if self.risk.state is RiskState.HALTED:
            for p in list(self.book.positions.values()):
                if not p.exiting and not p.working_exit:
                    self._exit(p, p.qty, "flatten: risk HALTED", now)

    def health_check(self, clock_drift_s: float = 0.0) -> None:
        now = self.clock.now()
        open_ = self.session.is_open(now)
        # Staleness is measured within the current session (first bar of the day
        # is due one minute after the open, plus a grace period).
        mso = self.session.minutes_since_open(now)
        ref = now - max(mso, 0) * 60 + self.bar_s + 120
        # ...and from when the live stream started, so a mid-session start is not "stale".
        ref = max(ref, self.stream_start_ts + self.bar_s + 120)
        last = max(self.last_bar_ts.values()) if self.last_bar_ts else ref
        stale = max(0.0, now - max(last, ref))
        book = {s: p.side * p.qty for s, p in self.book.positions.items()}
        brk = self.broker.positions()
        reconciled = True
        if book != brk and not self.pending_entries and not self.exit_orders_pending():
            self.reconcile_miss += 1
            reconciled = self.reconcile_miss < 2   # tolerate one in-flight check
        else:
            self.reconcile_miss = 0
        prev = self.risk.state
        self.risk.check_infrastructure(data_staleness_s=stale, staleness_limit_s=self.staleness_limit, clock_drift_s=clock_drift_s,
                                       broker_ok=self.broker.connected, reconciled=reconciled, market_open=open_, ts=now)
        if self.risk.state is not prev:
            self.J.write("risk", now, {"event": "state_change", "state": self.risk.state.value,
                                        "reason": self.risk.reasons[-1][2]})
            self._portfolio_checks(now)

    def exit_orders_pending(self) -> bool:
        orders = getattr(self.broker, "orders", {})
        return any(not orders[o].done for o in self.exit_orders if o in orders)
