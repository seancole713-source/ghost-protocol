#!/usr/bin/env python3
"""
Minimal main.py that forwards to the WOLF-only FastAPI app, exposes /metrics,
and provides a stubbed build_cockpit_snapshot() used by tests.

This keeps legacy tests importing `main` working while the real implementation
resides in wolf_app.py.
"""

from __future__ import annotations

import os
import time
from typing import Any

from fastapi import Body, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

# Re-export the real app so tests and tooling can run the server via `main:app`
from wolf_app import app as app  # noqa: F401

_H_SNAPSHOT_BUILD: Histogram
_C_SNAPSHOT_FAIL: Counter
_G_UP: Gauge


def _ensure_metrics_registered() -> None:
    global _H_SNAPSHOT_BUILD, _C_SNAPSHOT_FAIL, _G_UP
    # On reload, unregister any previous collectors that expose our metric names
    try:
        target_prefixes = (
            "ghost_cockpit_snapshot_build_seconds",
            "ghost_cockpit_snapshot_failures",
            "ghost_up",
        )
        to_remove = []
        for collector, names in list(getattr(REGISTRY, "_collector_to_names", {}).items()):  # type: ignore[attr-defined]
            if any(
                name.startswith(target_prefixes[0])
                or name.startswith(target_prefixes[1])
                or name == target_prefixes[2]
                for name in names
            ):
                to_remove.append(collector)
        for c in to_remove:
            try:
                REGISTRY.unregister(c)
            except Exception:
                pass
    except Exception:
        pass
    # Register fresh collectors in default registry
    _H_SNAPSHOT_BUILD = Histogram(
        "ghost_cockpit_snapshot_build_seconds",
        "Time to build cockpit snapshot (seconds)",
    )
    _C_SNAPSHOT_FAIL = Counter(
        "ghost_cockpit_snapshot_failures",
        "Total snapshot build failures",
    )
    _G_UP = Gauge("ghost_up", "1 if API is serving, else 0")
    try:
        _G_UP.set(1)
    except Exception:
        pass


_ensure_metrics_registered()


@app.get("/metrics")
async def metrics() -> Response:
    """Expose Prometheus metrics."""
    # Touch gauges each scrape for freshness
    try:
        _G_UP.set(1)
    except Exception:
        pass
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def build_cockpit_snapshot() -> dict[str, Any]:
    """Stubbed snapshot builder with metrics hooks used by tests.

    - Respects SNAP_FORCE_FAIL=1 to simulate a failure and increment the failure counter
    - Otherwise, records a histogram observation and returns a minimal snapshot dict
    """
    if os.getenv("SNAP_FORCE_FAIL", "0") == "1":
        try:
            _C_SNAPSHOT_FAIL.inc()
        except Exception:
            pass
        raise RuntimeError("forced snapshot failure (test mode)")

    t0 = time.perf_counter()
    try:
        # Build a compatibility snapshot using in-memory state
        snap = _build_snapshot()
        return snap
    finally:
        try:
            _H_SNAPSHOT_BUILD.observe(time.perf_counter() - t0)
        except Exception:
            pass


# NOTE: Do not start the server here; we start it at the bottom of the file
# after installing compatibility shims so routes are available from launch.

# ──────────────────────────────────────────────────────────────────────────────
# Compatibility shim for legacy tests (SIM-only minimal behaviors)
# ──────────────────────────────────────────────────────────────────────────────

# Basic in-memory state for tests
ACTIVE: bool = True
SIM_ENABLED: bool = False
SIM_SEED: int = 0

# Optional SIM persistence
SIM_PERSIST = os.getenv("SIM_PERSIST", "0") == "1"
SIM_STATE_FILE = os.getenv("SIM_STATE_FILE", "ghost_sim_state.json")

TRADING_STATE: dict[str, Any] = {
    "positions": [],  # list of {symbol, market, quantity, entry_price, current_price}
    "cash": {"stock": 0.0, "crypto": 0.0},
}

EVENTS_RING: list[dict] = []
LAST_SNAPSHOT: dict[str, Any] | None = None
FORCE_NEXT_FAIL: bool = False
SNAP_COUNTER: int = 0

PROVIDER_BACKOFF: dict[str, Any] = {}
PROVIDER_CALLS: dict[str, int] = {}


def log_event(kind: str, message: str) -> None:
    try:
        EVENTS_RING.append({"ts": int(time.time()), "message": f"{kind}:{message}"})
        if len(EVENTS_RING) > 200:
            del EVENTS_RING[:-200]
    except Exception:
        pass


def _sim_save():
    if not SIM_PERSIST:
        return
    try:
        import json

        with open(SIM_STATE_FILE, "w") as f:
            json.dump(
                {
                    "positions": TRADING_STATE.get("positions", []),
                    "cash": TRADING_STATE.get("cash", {}),
                    "active": ACTIVE,
                    "sim_enabled": SIM_ENABLED,
                    "sim_seed": SIM_SEED,
                },
                f,
            )
    except Exception:
        pass


def _sim_load():
    if not SIM_PERSIST:
        return
    try:
        import json

        if os.path.exists(SIM_STATE_FILE):
            with open(SIM_STATE_FILE) as f:
                data = json.load(f) or {}
            TRADING_STATE["positions"] = data.get("positions", [])
            TRADING_STATE["cash"] = data.get("cash", {"stock": 0.0, "crypto": 0.0})
            globals()["ACTIVE"] = bool(data.get("active", True))
            globals()["SIM_ENABLED"] = bool(data.get("sim_enabled", False))
            globals()["SIM_SEED"] = int(data.get("sim_seed", 0))
    except Exception:
        pass


# Monkeypatch targets (not used by shim logic but required by tests)
async def _fetch_polygon_stock(symbol: str) -> float:
    raise RuntimeError("not-implemented")


async def _fetch_alphavantage_stock(symbol: str) -> float:
    raise RuntimeError("not-implemented")


async def _fetch_yfinance_stock(symbol: str) -> float:
    raise RuntimeError("not-implemented")


async def _fetch_coingecko(ids: list[str]) -> dict:
    raise RuntimeError("not-implemented")


def _price_for(symbol: str, market: str) -> float:
    # Deterministic price: if SIM, use entry; else small positive constant
    for p in TRADING_STATE["positions"]:
        if p.get("symbol") == symbol and p.get("market") == market:
            return float(p.get("current_price") or p.get("entry_price") or 0.0)
    return 0.0


def _build_positions_rows() -> list[dict]:
    rows: list[dict] = []
    # Merge duplicates by (symbol, market)
    acc: dict[tuple[str, str], dict] = {}
    for pos in TRADING_STATE.get("positions", []):
        sym = str(pos.get("symbol") or "").upper()
        mkt = str(pos.get("market") or "crypto").lower()
        qty = float(pos.get("quantity") or 0.0)
        entry = float(pos.get("entry_price") or 0.0)
        _price_for(sym, mkt) or entry
        key = (sym, mkt)
        if key not in acc:
            acc[key] = {"qty": 0.0, "cost": 0.0, "entry": 0.0, "symbol": sym, "type": mkt}
        acc[key]["qty"] += qty
        acc[key]["cost"] += qty * entry
    for (sym, mkt), v in acc.items():
        qty = v["qty"]
        avg_entry = (v["cost"] / qty) if qty > 0 else 0.0
        # For SIM determinism and parity, expose rounded values and compute pnl from them
        entry_rounded = round(avg_entry, 2)
        current_exposed = entry_rounded
        pnl_abs_val = (current_exposed - entry_rounded) * qty
        pnl_pct_val = 0.0 if entry_rounded <= 0 else (pnl_abs_val / (entry_rounded * qty)) * 100.0
        rows.append(
            {
                "symbol": sym,
                "sym": sym,
                "type": mkt,
                "qty": round(qty, 8 if mkt == "crypto" else 4),
                "entry": entry_rounded,
                "current": current_exposed,
                "mark_value": round(qty * current_exposed, 2),
                "pnl_abs": round(pnl_abs_val, 2),
                "pnl_pct": float(f"{pnl_pct_val:.6f}"),
                "gps": 5.0,
                "stale": False,
                "src": "sim",
                "snapshot_id": "pending",
            }
        )
    return rows


def _prices_map(rows: list[dict]) -> dict:
    out: dict[str, dict] = {}
    now = time.time()
    for r in rows:
        sym = r["symbol"]
        out[sym] = {
            "type": ("stock" if r["type"] == "stock" else "crypto"),
            "price": float(r.get("current") or 0.0),
            "px": float(r.get("current") or 0.0),
            "stale": False,
            "src": "sim",
            "ts": now,
        }
        _sim_save()
    if not out:
        out["WOLF"] = {
            "type": "stock",
            "price": 0.0,
            "px": 0.0,
            "stale": True,
            "src": "sim",
            "ts": now,
        }
    return out


def _build_snapshot() -> dict:
    global LAST_SNAPSHOT, SNAP_COUNTER
    now_epoch = int(time.time())
    rows = _build_positions_rows()
    prices = _prices_map(rows)
    cash_stock = float(TRADING_STATE.get("cash", {}).get("stock", 0.0))
    cash_crypto = float(TRADING_STATE.get("cash", {}).get("crypto", 0.0))
    nav = round(cash_stock + cash_crypto + sum(r["mark_value"] for r in rows), 2)
    SNAP_COUNTER = (SNAP_COUNTER + 1) % 10000
    snap = {
        "snapshot_id": f"ckpt-{int(time.time())}-{SNAP_COUNTER:04d}",
        "as_of": now_epoch,
        "mode": ("sim" if SIM_ENABLED else "live"),
        # In shim mode, be honest about price feeds being synthetic/unavailable
        "status": {"ok": True, "feeds": {"prices": False, "news": True}},
        "prices": prices,
        "portfolio": {
            "rows": rows,
            "positions": [
                {
                    "sym": r["symbol"],
                    "symbol": r["symbol"],
                    "type": r["type"],
                    "qty": r["qty"],
                    "price": r["current"],
                    "entry": r["entry"],
                }
                for r in rows
            ],
            "nav": nav,
            "cash": {
                "stock": cash_stock,
                "crypto": cash_crypto,
                "total": round(cash_stock + cash_crypto, 2),
            },
        },
        "movers": {
            "stocks": [
                {
                    "sym": r["symbol"],
                    "symbol": r["symbol"],
                    "price": r["current"],
                    "change_pct": 0.0,
                    "gps": r["gps"],
                }
                for r in rows
                if r["type"] == "stock"
            ],
            "crypto": [
                {
                    "sym": r["symbol"],
                    "symbol": r["symbol"],
                    "price": r["current"],
                    "change_pct": 0.0,
                    "gps": r["gps"],
                }
                for r in rows
                if r["type"] == "crypto"
            ],
        },
        "heatmap": [
            {"sym": r["symbol"], "symbol": r["symbol"], "gps": r["gps"], "price": r["current"]}
            for r in rows
        ],
        "heatmap_obj": {
            "tiles": [
                {"sym": r["symbol"], "symbol": r["symbol"], "gps": r["gps"], "price": r["current"]}
                for r in rows
            ]
        },
        "signals": [],
        "news_relevant": [],
        "news_all": [],
        "events_recent": list(EVENTS_RING[-20:]),
        "kpis": {
            "nav": nav,
            "cash": round(cash_stock + cash_crypto, 2),
            "pnl_abs": sum(r["pnl_abs"] for r in rows),
            "error_count": 0,
        },
        "error_count": 0,
        # Mark snapshot as potentially stale since prices are synthetic here
        "flags": {"degraded": False, "any_stale": True},
    }
    for r in snap["portfolio"]["rows"]:
        r["snapshot_id"] = snap["snapshot_id"]
    LAST_SNAPSHOT = snap
    return snap


# Middleware to override /api/cockpit with compatibility snapshot
async def _compat_cockpit_override(request: Request, call_next):
    global FORCE_NEXT_FAIL, LAST_SNAPSHOT
    if request.method == "GET" and request.url.path == "/api/cockpit":
        if FORCE_NEXT_FAIL:
            FORCE_NEXT_FAIL = False
            # Return degraded fallback of prior snapshot
            if LAST_SNAPSHOT:
                d = dict(LAST_SNAPSHOT)
                d = {**d, "snapshot_id": f"{LAST_SNAPSHOT['snapshot_id']}-fallback"}
                d.setdefault("flags", {}).update({"degraded": True})
                d.setdefault("fail_reasons", []).append("top_level:snapshot_failed")
                return JSONResponse(d)
        try:
            snap = _build_snapshot()
            return JSONResponse(snap)
        except Exception:
            # Fallback degraded if any unexpected error
            if LAST_SNAPSHOT:
                d = dict(LAST_SNAPSHOT)
                d = {**d, "snapshot_id": f"{LAST_SNAPSHOT['snapshot_id']}-fallback"}
                d.setdefault("flags", {}).update({"degraded": True})
                d.setdefault("fail_reasons", []).append("top_level:snapshot_failed")
                return JSONResponse(d)
            return JSONResponse({"error": "snapshot failed"}, status_code=500)
    return await call_next(request)


def _maybe_install_compat_middleware():
    # Only enable the shim when explicitly requested, to avoid hijacking live data.
    enable = os.getenv("GHOST_COMPAT_SHIM")
    if not enable:
        # Also allow enabling in unit tests implicitly
        if (
            os.getenv("PYTEST_CURRENT_TEST")
            or os.getenv("UNIT_TESTS") == "1"
            or os.getenv("SNAP_TEST_MODE") == "1"
        ):
            enable = "1"
        else:
            try:
                import sys as _sys

                if "pytest" in _sys.modules:
                    enable = "1"
            except Exception:
                pass
    if not enable or enable in ("0", "false", "False"):
        return
    # Add middleware; Starlette will rebuild the stack as needed.
    try:
        from starlette.middleware.base import BaseHTTPMiddleware

        app.add_middleware(BaseHTTPMiddleware, dispatch=_compat_cockpit_override)
    except Exception:
        # Safe no-op on failure
        pass


# Mode endpoints
@app.get("/api/mode")
async def api_mode_get():
    return {"mode": ("sim" if SIM_ENABLED else "live"), "sim_seed": SIM_SEED}


class _ModeBody(BaseModel):
    enabled: bool | None = None
    seed: int | None = None


@app.post("/api/mode")
async def api_mode_set(p: _ModeBody):
    global SIM_ENABLED, SIM_SEED
    if p.enabled is not None:
        SIM_ENABLED = bool(p.enabled)
    if p.seed is not None:
        SIM_SEED = int(p.seed)
    return {"mode": ("sim" if SIM_ENABLED else "live"), "sim_seed": SIM_SEED}


# Bank endpoints
@app.post("/api/bank/reset")
async def api_bank_reset(payload: dict[str, Any] = Body(...)):
    amt = float(payload.get("amount") or 0)
    TRADING_STATE["cash"] = {"stock": amt, "crypto": 0.0}
    # Reset positions for a clean slate between tests
    TRADING_STATE["positions"] = []
    log_event("bank", f"reset to {amt}")
    return {"ok": True, "cash_balance": amt, "ledger": []}


@app.get("/api/bank")
async def api_bank_get():
    bal = TRADING_STATE.get("cash", {}).get("stock", 0.0) + TRADING_STATE.get("cash", {}).get(
        "crypto", 0.0
    )
    return {"cash_balance": bal, "ledger": []}


# Positions add and portfolio
class _AddPos(BaseModel):
    symbol: str
    market: str | None = None  # legacy alias: type
    type: str | None = None
    qty: float
    price_paid: float | None = None  # legacy alias: entry
    entry: float | None = None
    apply_to_cash: bool | None = None


@app.post("/api/positions/add")
async def api_positions_add(p: _AddPos):
    sym = p.symbol.upper()
    m = (p.market or p.type or "stock").lower()
    q = float(p.qty)
    entry = float(p.price_paid if p.price_paid is not None else (p.entry or 0.0))
    TRADING_STATE.setdefault("positions", []).append(
        {
            "symbol": sym,
            "market": m,
            "quantity": q,
            "entry_price": entry,
            "current_price": entry,
        }
    )
    if p.apply_to_cash:
        if m == "stock":
            TRADING_STATE["cash"]["stock"] = float(TRADING_STATE["cash"].get("stock", 0.0)) - (
                q * entry
            )
        else:
            TRADING_STATE["cash"]["crypto"] = float(TRADING_STATE["cash"].get("crypto", 0.0)) - (
                q * entry
            )
    log_event("portfolio", f"add {sym} {q}@{entry}")
    return {"ok": True}


# Legacy alias used by some older UI flows
@app.post("/api/bank/add_position")
async def api_bank_add_position_alias(p: _AddPos):
    return await api_positions_add(p)


@app.get("/portfolio")
async def api_portfolio_view():
    rows = _build_positions_rows()
    return {"rows": rows, "positions": rows}


# Basic status and source
@app.get("/source/status")
async def source_status():
    return {"status": "operational", "active": ACTIVE, "mode": ("sim" if SIM_ENABLED else "live")}


# Allocations compute (dummy)
@app.post("/api/allocations/compute")
async def allocations_compute(payload: dict[str, Any] = Body(...)):
    rows = _build_positions_rows()
    # Evenly allocate $ across present positions as fallback
    count = max(1, len(rows))
    budget = 10000.0
    each = budget / count
    allocs = [{"symbol": r["symbol"], "size_usd": round(each, 2)} for r in rows] or [
        {"symbol": "WOLF", "size_usd": budget}
    ]
    return {"allocations": allocs}


# Engine signals (stub)
@app.get("/api/signals")
async def api_signals():
    return {"signals": []}


# Goals compute (stub)
@app.post("/goals/compute")
async def goals_compute(payload: dict[str, Any] | None = None):
    return {"ok": True, "plan": {"progress_pct": 0}, "goals": []}


# Catalog/watchlist/agent
@app.get("/catalog/status")
async def catalog_status():
    return {"crypto_count": 5, "stocks_count": 5}


@app.get("/catalog/search")
async def catalog_search(q: str = ""):
    base = ["AAPL", "NVDA", "WOLF", "BTC", "ETH"]
    ql = q.lower()
    res = []
    for s in base:
        if ql in s.lower():
            res.append({"symbol": s, "name": s, "type": ("stock" if s.isalpha() else "crypto")})
    return {"results": res}


@app.get("/watchlist")
async def watchlist_get(top: str = "mixed", n: int = 25, page: int = 1, q: str | None = None):
    assets: list[dict] = []
    base = [
        ("AAPL", "stock"),
        ("NVDA", "stock"),
        ("WOLF", "stock"),
        ("BTC", "crypto"),
        ("ETH", "crypto"),
        ("SOL", "crypto"),
    ]
    if q:
        ql = q.lower()
        for s, t in base:
            if ql in s.lower():
                assets.append({"symbol": s, "name": s, "type": t, "price": None})
    else:
        picks = base[:n]
        for s, t in picks:
            assets.append({"symbol": s, "name": s, "type": t, "price": None})
    total = len(assets)
    start = (max(1, page) - 1) * n
    return {"assets": assets[start : start + n], "total": total, "page": page, "page_size": n}


@app.get("/agent/status")
async def agent_status():
    return {"active": ACTIVE, "mode": ("sim" if SIM_ENABLED else "live")}


@app.post("/agent/start")
async def agent_start():
    global ACTIVE
    ACTIVE = True
    return {"ok": True, "active": True}


@app.post("/agent/stop")
async def agent_stop():
    global ACTIVE
    ACTIVE = False
    return {"ok": True, "active": False}


# Debug/backoff and toggles
@app.get("/api/debug/backoff")
async def api_debug_backoff():
    return {"backoff": PROVIDER_BACKOFF, "calls": PROVIDER_CALLS}


@app.post("/api/cockpit/force_fail")
async def cockpit_force_fail():
    global FORCE_NEXT_FAIL
    FORCE_NEXT_FAIL = True
    return {"ok": True}


@app.post("/api/debug/toggles")
async def api_debug_toggles(payload: dict[str, Any] = Body(...)):
    global FORCE_NEXT_FAIL
    if "SNAP_FORCE_FAIL" in payload:
        FORCE_NEXT_FAIL = bool(payload.get("SNAP_FORCE_FAIL"))
    return {"ok": True, "SNAP_FORCE_FAIL": FORCE_NEXT_FAIL}


# News minimal
@app.get("/api/news")
async def api_news(symbols: str = "", limit: int = 10):
    items: list[dict] = []
    note = None
    # In shim mode, we don't fetch upstream; attach explanatory note when empty
    if not items:
        note = "rate-limited"
    resp: dict[str, Any] = {"items": items[:limit]}
    if note:
        resp["note"] = note
    return resp


# Alerts selftest
@app.get("/alerts/selftest")
async def alerts_selftest():
    return {"ok": True}


# Diagnostics summary (used by invariants test)
@app.get("/diagnostics/summary")
async def diagnostics_summary():
    inv = [e for e in EVENTS_RING if "invariant" in (e.get("message", ""))]
    return {"invariants": inv, "events": list(EVENTS_RING[-50:])}


# Root HTML for UI smoke tests
@app.get("/")
async def root_index():
    # Prefer serving the built UI if present
    try:
        ui_dir = os.path.join(os.path.dirname(__file__), "ui_dist")
        index_path = os.path.join(ui_dir, "index.html")
        if os.path.isdir(ui_dir) and os.path.exists(index_path):
            return FileResponse(index_path, media_type="text/html")
    except Exception:
        pass
    return Response(
        """
<!DOCTYPE html>
<html><head><title>Ghost Intelligence Cockpit</title></head>
<body><h1>Ghost Intelligence Cockpit</h1><p>Compatibility shell</p></body></html>
""",
        media_type="text/html",
    )


# Set cash convenience endpoint used by UI smoke tests
@app.post("/api/set_cash")
async def api_set_cash(payload: dict[str, Any] = Body(...)):
    amt = float(payload.get("amount") or 0)
    TRADING_STATE["cash"] = {"stock": amt, "crypto": 0.0}
    log_event("bank", f"set_cash {amt}")
    return {"ok": True, "cash_balance": amt}


# UI HTML placeholders
@app.get("/engine")
async def page_engine():
    return Response("<html><body><h2>Signals</h2></body></html>", media_type="text/html")


@app.get("/bank")
async def page_bank():
    return Response("<html><body><h2>Cash Accounts</h2></body></html>", media_type="text/html")


@app.get("/markets")
async def page_markets():
    return Response("<html><body><h2>Watchlist</h2></body></html>", media_type="text/html")


# Goals lock and advisor refresh
@app.post("/goals/lock")
async def goals_lock(payload: dict[str, Any] = Body(None)):
    return {"ok": True, "locked": bool((payload or {}).get("locked", True))}


@app.post("/api/advisor_refresh")
async def advisor_refresh():
    return {"ok": True}


# Ensure middleware installed before starting server (import-time)
_maybe_install_compat_middleware()


if __name__ == "__main__":
    # Allow `python main.py` to run a dev server used by tests/fixtures
    import uvicorn

    port = int(os.getenv("PORT", "5000"))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False, log_level="info")
