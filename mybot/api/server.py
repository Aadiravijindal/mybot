"""HTTP API + dashboard server.

Security defaults:
  * binds 127.0.0.1 unless MYBOT_HOST is set;
  * when bound to anything else, EVERY /api route requires the admin bearer token;
  * resume-after-halt ALWAYS requires the token (a halt is safe; un-halting is not);
  * no endpoint can change risk limits or submit discretionary orders.
"""
from __future__ import annotations

import asyncio
import hmac
import os
import secrets
from pathlib import Path

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse

from ..audit.journal import dumps
from . import snapshot

DASH = Path(__file__).resolve().parent.parent.parent / "dashboard" / "index.html"


def create_app(rt) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(rt.run())
        yield
        task.cancel()

    app = FastAPI(title="mybot", docs_url=None, redoc_url=None, lifespan=lifespan)
    token = os.environ.get("MYBOT_ADMIN_TOKEN") or secrets.token_urlsafe(24)
    app.state.token = token
    protect_reads = rt.S.host not in ("127.0.0.1", "localhost")

    def authed(req: Request) -> bool:
        h = req.headers.get("authorization", "")
        return h.startswith("Bearer ") and hmac.compare_digest(h[7:], token)

    @app.middleware("http")
    async def guard(req: Request, call_next):
        if protect_reads and req.url.path.startswith("/api") and not authed(req):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        resp = await call_next(req)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'"
        return resp

    @app.get("/")
    async def index():
        return FileResponse(DASH)

    @app.get("/api/state")
    async def state():
        return JSONResponse(content=__import__("json").loads(dumps(snapshot.build(rt))))

    @app.get("/api/decision")
    async def decision(symbol: str, at: float | None = None):
        """Reconstruct why the system did what it did for ``symbol`` at time ``at``."""
        s = rt.system
        rec = s.J.decision_at(symbol.upper(), at if at is not None else s.clock.now())
        if rec is None:
            raise HTTPException(404, "no decision recorded at or before that time")
        return JSONResponse(content=__import__("json").loads(dumps(rec)))

    @app.post("/api/kill")
    async def kill(req: Request):
        if os.environ.get("MYBOT_ADMIN_TOKEN") and not authed(req):
            raise HTTPException(401)
        s = rt.system
        s.risk.halt("MANUAL KILL SWITCH", s.clock.now())
        s.J.write("risk", s.clock.now(), {"event": "manual_kill", "state": "HALTED", "reason": "kill switch pressed"})
        s._portfolio_checks(s.clock.now())
        return {"state": s.risk.state.value}

    @app.post("/api/resume")
    async def resume(req: Request):
        if not authed(req):
            raise HTTPException(401, "resume requires the admin token (printed at startup)")
        s = rt.system
        s.risk.manual_resume("dashboard-admin", s.clock.now())
        s.J.write("risk", s.clock.now(), {"event": "manual_resume", "state": "NORMAL", "reason": "admin resume"})
        return {"state": s.risk.state.value}

    @app.get("/healthz")
    async def healthz():
        s = rt.system
        return {"ok": s.bus.errors < 100, "risk": s.risk.state.value, "status": rt.status_line}

    return app
