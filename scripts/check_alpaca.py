"""Pre-flight check for Alpaca credentials, data feed and the paper broker. Places NO orders."""
import os
import sys

import httpx

key, sec = os.environ.get("ALPACA_API_KEY_ID"), os.environ.get("ALPACA_API_SECRET_KEY")
if not key or not sec:
    sys.exit("set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY")
h = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}
feed = os.environ.get("ALPACA_DATA_FEED", "iex")
acct = httpx.get("https://paper-api.alpaca.markets/v2/account", headers=h, timeout=10)
print("paper account:", acct.status_code, {k: acct.json().get(k) for k in ("status", "equity", "shorting_enabled")} if acct.is_success else acct.text[:200])
clock = httpx.get("https://paper-api.alpaca.markets/v2/clock", headers=h, timeout=10)
print("market clock:", clock.json() if clock.is_success else clock.text[:200])
bars = httpx.get("https://data.alpaca.markets/v2/stocks/bars/latest", headers=h, params={"symbols": "SPY,AAPL", "feed": feed}, timeout=10)
print(f"latest bars ({feed}):", bars.json() if bars.is_success else bars.text[:200])
sys.exit(0 if acct.is_success and bars.is_success else 1)
