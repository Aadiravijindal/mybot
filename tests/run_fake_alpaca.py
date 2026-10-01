"""Launch the fake Alpaca server:  python tests/run_fake_alpaca.py PORT START_TS SPEED [preexisting_json] [pdt]"""
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn  # noqa: E402

from fake_alpaca import FakeAlpaca, create_app  # noqa: E402
from mybot.config import load_settings  # noqa: E402

port, start_ts, speed, wall0 = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[6])
pre = json.loads(sys.argv[4]) if len(sys.argv) > 4 else None
pdt = sys.argv[5] == "pdt"
S = load_settings()
day = datetime.fromtimestamp(start_ts, ZoneInfo("America/New_York"))
fake = FakeAlpaca(list(S.symbols), S.market_proxy, day, start_ts, speed, preexisting=pre, pdt=pdt,
                  equity=20_000.0 if pdt else 50_000.0, wall0=wall0)
uvicorn.run(create_app(fake), host="127.0.0.1", port=port, log_level="warning")
