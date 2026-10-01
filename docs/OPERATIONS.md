# Operations runbook

## Daily checklist (before the open)
1. `python -m mybot verify-audit`: the chain must be OK.
2. Dashboard → System health: every feed is fresh, the broker is connected, handler errors are 0, and the clock is synced (`chronyc tracking`).
3. Risk state is NORMAL. If it's HALTED, **find out why before resuming** (Research feed → risk events, and `var/audit/journal.jsonl`).
4. Check the economic calendar for high-impact events. The engine blacks out entries around them, but you should know what's coming.
5. Model monitor: any setup marked DEGRADED or DISABLED stays that way until it has been re-validated.

## Start-up pre-flight (automatic, real-data modes)
The engine refuses to trade until these pass. The results are logged as a `preflight` risk event on the dashboard:
* account ACTIVE and not trading-blocked (otherwise it exits with the reason);
* equity read from the broker becomes the sizing base;
* **pattern-day-trader rule**: margin accounts under $25k get 3 day trades per rolling 5 business days. Every round
  trip of this intraday system uses one, and when none are left new entries are blocked;
* **positions it didn't open → HALT.** It will not manage or close them. Close or transfer them yourself, then resume;
* stale open orders created by a previous run (client order id `mb…`) are cancelled. Your manual orders are left alone.

## Incidents
| Symptom | Do this |
|---|---|
| HALTED: "market data stale" | Check vendor status and network. Stops are resting at the broker. Resume only after data has been flowing for ≥ 5 minutes. |
| HALTED: "position state uncertain" | Compare broker positions with the dashboard. Flatten manually at the broker if needed. Resume only when they match. |
| HALTED: "broker disconnected" | Check the broker status page and your API keys. Confirm open orders and stops at the broker. |
| HALTED: "daily loss" / "drawdown" | Do **not** resume the same day. Review the attribution for every trade. Treat it as a research event, not an execution problem. |
| REDUCE_ONLY: slippage or loss streak | It clears at the next session. Review the execution quality first. |
| HALTED: "pre-existing broker positions" | Positions in the account that this engine didn't open. Close them at the broker (or move them to another account), then resume. |
| Risk event "stop_lost" | The broker refused or cancelled a protective stop. The engine closes that position automatically. Find out why (shortability, price increments, account restrictions). |
| Dashboard unreachable | Positions stay protected by broker-resident stops. SSH in, check `docker compose logs`, restart, reconcile. |

**Cool-down rule:** after any HALT caused by losses, wait at least one full session and write down the cause before resuming.
Never edit risk limits the same day they were breached.

## Go-live ladder (do not skip steps)
1. Research: written hypothesis and economic rationale for each setup.
2. Backtest on ≥ 2 years of point-in-time vendor data including delisted names, with realistic costs.
3. Walk-forward: OOS expectancy ≥ +0.05R after costs, ≥ 100 OOS trades, ≥ 60% positive folds, Deflated-Sharpe probability ≥ 0.90.
4. Paper trading on live data: ≥ 30 sessions and ≥ 60 trades, expectancy ≥ 50% of the backtest, slippage ≤ 1.5× the model.
5. Limited live: ≤ 10% of intended capital for ≥ 20 sessions.
6. Full deployment after a written review.

## Secrets
* Use paper keys until step 5. Live keys should be trading-only (no transfers) and IP-restricted where the broker supports it.
* Keep keys in a secret manager or a `.env` file with `chmod 600`. Never commit them (`.env` is git-ignored).
* Rotate keys after any suspected exposure and after staff changes.

## Exposing the dashboard remotely
Don't bind it to a public interface. Use a VPN (for example Tailscale) or a reverse proxy with TLS + SSO + MFA. If `MYBOT_HOST` is not
localhost, every `/api` call requires `MYBOT_ADMIN_TOKEN`.
