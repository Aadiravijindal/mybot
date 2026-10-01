"""Pre-flight check for Alpaca credentials, account and data feed. Places NO orders.

    python scripts/check_alpaca.py          # paper keys
    python scripts/check_alpaca.py --live   # live keys (read-only checks)
"""
import os
import sys

import httpx

key, sec = os.environ.get("ALPACA_API_KEY_ID"), os.environ.get("ALPACA_API_SECRET_KEY")
if not key or not sec:
    sys.exit("set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY")
h = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}
feed = os.environ.get("ALPACA_DATA_FEED", "iex")
base = "https://api.alpaca.markets" if "--live" in sys.argv else "https://paper-api.alpaca.markets"
print("trading endpoint:", base)
acct = httpx.get(f"{base}/v2/account", headers=h, timeout=10)
print("account:", acct.status_code, {k: acct.json().get(k) for k in ("status", "equity", "pattern_day_trader", "daytrade_count",
      "shorting_enabled", "trading_blocked")} if acct.is_success else acct.text[:200])
pos = httpx.get(f"{base}/v2/positions", headers=h, timeout=10)
if pos.is_success and pos.json():
    print("WARNING: account holds positions; mybot will HALT on start until they are closed:", [p["symbol"] for p in pos.json()])
clock = httpx.get(f"{base}/v2/clock", headers=h, timeout=10)
print("market clock:", clock.json() if clock.is_success else clock.text[:200])
bars = httpx.get("https://data.alpaca.markets/v2/stocks/bars/latest", headers=h, params={"symbols": "SPY,AAPL", "feed": feed}, timeout=10)
print(f"latest bars ({feed}):", bars.json() if bars.is_success else bars.text[:200])
sys.exit(0 if acct.is_success and bars.is_success else 1)
