# mybot: intraday market intelligence and paper-trading system

An event-driven system that continuously scans a universe of stocks and researches news and catalysts.
It forms falsifiable trade hypotheses, argues against them, and prices them after costs. It sizes
them under hard risk limits and manages open positions with a living thesis and ratcheting stops.
Everything shows up on a dashboard.

> **Read this first.** This software does not predict the market, and nothing in it has shown a
> real-world edge. On its built-in simulated market it currently *loses* money after costs, and its own
> tools report that (see "What the numbers say" below). It is built to be honest with you, to say
> **"no trade"** most of the time, and to fail safe. Paper-trade on real data for months before
> considering real money. Not financial advice.

## Start it (no accounts, no keys)

```bash
pip install -e .            # Python 3.11+
python -m mybot             # then open http://127.0.0.1:8000
```

On first start it spends about a minute building a knowledge base from 20 simulated trading days. Then it streams a
simulated market at 60× speed (one trading day ≈ 6.5 minutes), trading on a paper account. After that you
only watch the dashboard:

* **Header**: mode, market clock, risk state, **KILL SWITCH** (halts and flattens everything)
* **KPIs**: equity, today's P&L, drawdown vs limit, positions, win rate and expectancy
* **Open positions**: thesis state, live stop, targets, research since entry (click a row for detail)
* **Decisions**: every candidate with bull case, bear case, analogues, EV, size and blockers
* **Research feed**: news (deduplicated, reliability-scored), regime changes, anomalies, thesis and risk events
* **Scanner, closed trades with attribution, system health, model monitor, research backtest**

The console prints an **admin token** at startup. You need it only to *resume* after a halt.

Docker: `docker compose up`. The dashboard is published to the host's localhost only. Inside a container the API is token-protected, so the page asks once for the admin token shown in `docker compose logs`.

## Paper trading on real data, then live (Alpaca)

**Step 1: paper (do this for weeks first).** Create a free Alpaca account and generate **paper** API keys.

```bash
pip install -e ".[live]"
cp .env.example .env        # paste your PAPER keys
set -a; . ./.env; set +a
python scripts/check_alpaca.py          # verifies keys, account and data. Places NO orders.
python -m mybot --data alpaca           # PAPER trading on real market data
```

On start it runs **pre-flight checks**:
- the account must be active and allowed to trade;
- it sizes from your **real account equity**, not the config number;
- it applies the US **pattern-day-trader** rule (under $25k equity you get only 3 day trades per 5 days);
- it **halts** if the account holds positions it didn't open, and never touches them;
- it cancels any stale orders it left behind, and never touches your own orders.

It then downloads recent real 1-minute history to build its knowledge base before trading.

**Step 2: live (real money).** Only after the go-live ladder in `docs/OPERATIONS.md`. Use your **live** keys and both switches:

```bash
MYBOT_LIVE_CONFIRM=I_UNDERSTAND_THE_RISKS python -m mybot --data alpaca --mode live
```

Without the confirmation variable, live mode refuses to start. The dashboard turns red and says **LIVE · REAL MONEY**.
The free IEX feed covers about 2–3% of US volume, so volume and spread features are biased. `ALPACA_DATA_FEED=sip` needs
Alpaca's paid data plan.

Optional LLM research (structured extraction from news; never predicts prices, never sees credentials):
`pip install -e ".[llm]"` and set `ANTHROPIC_API_KEY`.

## Validation (what has been tested, and how to re-run it)

```bash
pytest                                     # 41 unit/regression tests (~10 s)
python tests/e2e_alpaca.py                 # real runtime vs a local fake Alpaca: paper, live guard, leftovers,
                                           #   broker outage, stop rejection, PDT  (~15 min)
python scripts/soak.py --seeds 6 --days 10 --stress --faults   # many simulated days with corrupted data;
                                           #   checks accounting, stops never loosen, flat at close, book == broker
python -m mybot backtest --source csv --csv path/to/SPY.csv:SPY --symbols SPY --proxy QQQ   # any OHLCV CSV
```

The fake Alpaca reproduces the broker behaviours that break naive bots. Shares reserved by a resting stop can't be
sold again, replacing an order creates a new order id, cancels are asynchronous, and the API can be made to fail
or refuse stops. Nothing here has been run against the real Alpaca API, because this build environment can't reach it.
`scripts/check_alpaca.py` is your first real contact. Run it before anything else.

## Research tools

```bash
python -m mybot backtest --days 40        # event-driven backtest + Monte Carlo + Deflated Sharpe
python -m mybot walkforward --train 20 --test 5 --folds 4
python -m mybot research                  # walk-forward feature ablation on the knowledge base
python -m mybot verify-audit              # verify the tamper-evident decision journal
```

### What the numbers say (synthetic market, 40 days, seed 3)
14 trades out of ~500 candidate decisions (332 AVOID, 155 WATCH). Expectancy −0.24R, t-stat −1.3,
Deflated-Sharpe probability 0.004, and the feature model **does not beat the base rate** out of sample. That is the correct
conclusion for synthetic data with no designed-in edge. The same tools must say something very different on
years of real data before any capital is at risk.

## Configuration
`config/default.toml` holds the hard risk limits (risk per trade 0.25%, daily loss 1.5%, drawdown 6%, max 5 positions, …).
They are frozen at startup. Nothing at runtime, including the models and the LLM, can loosen them.

## Documentation
* [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): the full specification. It covers the architecture diagram, the
  40-engine map with honest status, stack choices, the data-source comparison with prices, database and event design,
  AI/agent design, the feature evidence review, backtesting and promotion gates, failure and security architecture,
  dashboard, roadmap, MVP vs production, cost estimates, testing, the 20 biggest failure modes, and what "selling it"
  legally requires.
* [`docs/OPERATIONS.md`](docs/OPERATIONS.md): runbook covering daily checklist, incidents, the go-live ladder.

## Layout
```
mybot/core        clock, event bus (point-in-time guard), types, session calendar
mybot/data        synthetic market, Alpaca data adapter, data-quality engine
mybot/features    incremental indicators, multi-timeframe + microstructure features
mybot/intel       regime (rules + causal HMM), news intelligence, catalysts, anomalies, LLM extractor
mybot/signals     setups, evidence model, historical analogues, forecast/EV, adversarial bear engine
mybot/risk        risk engine (final authority), sizing, dynamic stops, thesis state machine
mybot/execution   order model, pessimistic paper broker, Alpaca broker
mybot/engine      orchestrator (the funnel + position loop), portfolio book
mybot/backtest    replay engine, metrics, Monte Carlo, Deflated Sharpe, walk-forward
mybot/learning    isotonic calibration, feature ablation, gated model registry
mybot/monitoring  drift/model monitor, post-trade attribution
mybot/audit       hash-chained journal + knowledge base
mybot/api         FastAPI server + dashboard snapshot
dashboard/        single-file dashboard (no build step)
```
