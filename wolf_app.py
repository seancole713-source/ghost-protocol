# ══════════════════════════════════════════════════════════════
# FILE: wolf_app.py
# PURPOSE: Main FastAPI application shell and entry point for Ghost Protocol.
#          Creates the APP instance, mounts middleware, includes route routers,
#          and registers startup/shutdown events. Procfile target: wolf_app:APP
# STATUS: STABLE
# LINES: ~386
# ──────────────────────────────────────────────────────────────
# CHANGE LOG:
#   2026-03-19 — Briefing header added (Browser Agent)
#   2026-03-19 — Bug #23 fix applied in engines/startup.py (related)
# ──────────────────────────────────────────────────────────────
# KNOWN ISSUES:
#   None critical — file is structurally clean after Step 12 extraction
# ──────────────────────────────────────────────────────────────
# DO NOT CHANGE (frozen interfaces):
#   APP = FastAPI(...)          — Procfile depends on wolf_app:APP
#   startup_event()            — registered as APP lifespan/on_event
#   STATIC_CACHE_BUST          — used by templates for cache busting
# ══════════════════════════════════════════════════════════════
#!/usr/bin/env python3
# Ghost Protocol — WOLF FastAPI application shell
# Step 12: Structural cleanup — routes extracted to routes/, helpers to wolf_helpers.py
# Entry point: wolf_app:APP (Procfile compatible)

import asyncio
import atexit
import contextvars
import hashlib
import json
import logging
import math
import os
import queue as _queue
import sqlite3
import threading
import time
import uuid
from collections import deque
from datetime import UTC, datetime, timedelta

# Cache-bust token for static JS – changes on every process (re)start / deploy
_STATIC_CACHE_BUST: str = str(int(time.time()))

# ── Import shared helpers (extracted from this file in Step 12) ──────────
from wolf_helpers import (
    _is_truthy, _is_live_enforced, _get_git_sha,
    _parse_origins, _compute_csp, _configure_logging,
    _json500, with_cap, should_create_prediction,
    _ensure_dir_for_file, _init_security_tables, _set_mode_gauge,
    _ensure_ai_storage, _cv_trace_id, _cv_path, _cv_method,
    _set_hold_gauge, _stop_autosave_worker, _stop_alert_worker,
    _persist_save, _stop_schedule_worker, _tg_send_chat_message,
    _classify_symbol_category,
)

# Import anyio for timeout control
try:
    import anyio
except ImportError:
    anyio = None  # type: ignore

# Load .env file FIRST before any os.getenv() calls
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not installed, will use system env vars

# Process start timestamp for uptime metrics
_START_TS = time.time()

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:
    ZoneInfo = None  # type: ignore
from collections.abc import Callable
from dataclasses import asdict
from typing import Any, Optional

import requests
from requests.adapters import HTTPAdapter

try:
    # Retry moved under urllib3.util.retry in recent versions
    from urllib3.util.retry import Retry  # type: ignore
except Exception:  # pragma: no cover - fallback
    Retry = None  # type: ignore
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import uvicorn
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request, Response, Security, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

from core.concurrency import AsyncRateLimiter
from core.heartbeat import pulse as _heartbeat_pulse
from core.price_quorum import PriceDecision, PriceProvider, get_price_quorum
from core.providers.turbo_provider import turbo_stock_price, turbo_crypto_price
from config.symbols import DEFAULT_EDGE_SYMBOLS, get_edge_set



# ── App globals (feature flags, symbol lists, state, workers) ────────────
# Extracted from wolf_app.py init block. We inject ALL names (including
# private underscore names) so `import wolf_app as _wa; _wa._SOME_GLOBAL`
# patterns in routes/picks.py and routes/subsystems.py continue to work.
import engines.app_config as _app_cfg
globals().update({k: v for k, v in vars(_app_cfg).items() if not k.startswith("__")})
del _app_cfg  # keep namespace tidy



# ============================================================================
# FREE IMPROVEMENTS: API Keys, IP Allowlisting, Webhooks
# ============================================================================
import hmac
import secrets

# API Key Management - Now database-backed
API_KEYS_DB = {}  # Cached in-memory: {key_id: {key_hash: str, name: str, rate_limit: int, created_at: float, active: bool}}
API_KEY_REQUESTS = {}  # Rate limiting tracker {key: deque(timestamps)}

# IP Allowlisting
_allowlist_str = os.getenv("IP_ALLOWLIST", "").strip()
IP_ALLOWLIST = set(ip.strip() for ip in _allowlist_str.split(",") if ip.strip()) if _allowlist_str else set()
IP_ALLOWLIST_ENABLED = len(IP_ALLOWLIST) > 0

print(f"🔒 IP_ALLOWLIST config: enabled={IP_ALLOWLIST_ENABLED}, ips={IP_ALLOWLIST}", flush=True)

# Webhooks - Now database-backed
WEBHOOK_SUBSCRIPTIONS = {}  # Cached in-memory: {webhook_id: {url: str, events: list[str], secret_hash: str}}

APP = FastAPI(
    title="Ghost — WOLF-only",
    version="1.0",
    # Move OpenAPI JSON so proxies won't accidentally expose JSON at root
    openapi_url="/api/openapi.json",
    # Serve docs UIs under /api to keep root clean
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)
# Alias for deployment runners that expect `wolf_app:app`
app = APP

# ── Global Exception Handlers ────────────────────────────────────────────
@APP.exception_handler(RuntimeError)
async def _rt_handler(request: Request, exc: RuntimeError):
    if str(exc).strip() == "No response returned.":
        return _json500("runtime_no_response")
    return _json500("runtime_error")

@APP.exception_handler(Exception)
async def _ex_handler(request: Request, exc: Exception):
    return _json500("unhandled_exception")

# ── Extracted route modules ──────────────────────────────────────────────
# Clean, self-contained route handlers extracted from the monolith.
# Each module defines a FastAPI APIRouter and is wired in here.
from routes.picks import router as picks_router
from routes.history import router as history_router
from routes.heartbeat import router as heartbeat_router
from routes.subsystems import router as subsystems_router

APP.include_router(picks_router)
APP.include_router(history_router)
APP.include_router(heartbeat_router)
APP.include_router(subsystems_router)


# ---------------------------------------------------------------------------
# Timeout Wrapper for External Calls (2.5s cap to prevent 499 errors)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Global Exception Handlers - Always Return JSON 500
# ---------------------------------------------------------------------------



# Note: BaseException handler removed - not supported by FastAPI/Starlette
# (BaseException is not a subclass of Exception)


# Compatibility shim: keep /openapi.json working but redirect to new location


# Lightweight debug endpoint to list routes (helps verify production routing)




# ---------------------------------------------------------------------------
# Mount News Router (modular approach)
# ---------------------------------------------------------------------------
try:
    from routes.news_routes import news_router
    APP.include_router(news_router, prefix="/api/news", tags=["news"])
    print("[INIT] ✅ News router mounted")
except Exception as e:
    print(f"[INIT] ⚠️  News router unavailable: {e}")

# Mount Quality API (Phase 4.2, 4.3, 5.6 - diversity, duplicates, scheduling)
try:
    from routes.quality_api import router as quality_router
    APP.include_router(quality_router, tags=["quality"])
    print("[INIT] ✅ Quality monitoring router mounted")
except Exception as e:
    print(f"[INIT] ⚠️  Quality router unavailable: {e}")

# Mount Demo Endpoints (provides instant testing)
try:
    from api.demo_endpoints import router as demo_router
    APP.include_router(demo_router)
    print("[INIT] ✅ Demo endpoints mounted: /api/demo/morning_now")
except Exception as e:
    print(f"[INIT] ⚠️  Demo endpoints unavailable: {e}")

# Mount Crypto OHLCV Router (provides /api/crypto/ohlcv/{symbol})
# Note: This router is optional and provides additional crypto OHLCV endpoints
try:
    from routes.crypto_ohlcv_routes import crypto_ohlcv_router

    APP.include_router(crypto_ohlcv_router, tags=["crypto"])
    print("[INIT] ✅ Crypto OHLCV router mounted successfully")
except Exception as e:
    # Router is optional; crypto endpoints in main app still work
    print(f"[INIT] ⚠️  Crypto OHLCV router unavailable (optional): {e}")

# ---------------------------------------------------------------------------
# Mount Ghost Intel Router (8-layer intelligence system)
# ---------------------------------------------------------------------------
try:
    from ghost_intel.routes import router as intel_router
    APP.include_router(intel_router, tags=["intel"])
    print("[INIT] ✅ Ghost Intel router mounted: /api/intel/*")
except Exception as e:
    print(f"[INIT] ⚠️  Ghost Intel router unavailable: {e}")


# =============================================================================
# TRUSTED_HOSTS Configuration (Host Header Validation)
# =============================================================================
# Parse comma-separated list of trusted hosts
# Supports wildcards like "*.railway.app"
TRUSTED_HOSTS_STR = os.getenv("TRUSTED_HOSTS", "").strip()
TRUSTED_HOSTS = [h.strip() for h in TRUSTED_HOSTS_STR.split(",") if h.strip()] if TRUSTED_HOSTS_STR else []

# Add TrustedHostMiddleware if TRUSTED_HOSTS is configured
if TRUSTED_HOSTS:
    try:
        from starlette.middleware.trustedhost import TrustedHostMiddleware
        APP.add_middleware(TrustedHostMiddleware, allowed_hosts=TRUSTED_HOSTS)
        print(f"🔒 TRUSTED_HOSTS enabled: {TRUSTED_HOSTS}")
    except ImportError:
        print("⚠️  TrustedHostMiddleware not available (starlette version)")

APP.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_origins(os.getenv("ALLOWED_ORIGINS", "*")),
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ── Register all HTTP middleware ─────────────────────────────────────────
from engines.middleware import register_middleware
register_middleware(APP)

# Note: Memory MCP integration is an optional feature module
# Gracefully handle if module doesn't exist (not required for core functionality)
try:
    from core.memory_mcp_integration import GhostMemoryEngine, MemoryStoreRequest  # type: ignore
    _MEMORY_ENGINE = GhostMemoryEngine()
except Exception as _mcp_e:
    LOGGER.warning(f"Memory MCP integration not loaded: {_mcp_e}")

# ── V3 Routes (extracted to routes/v3_routes.py) ──────────────────────
try:
    from routes.v3_routes import register_v3_routes
    register_v3_routes(APP)
    LOGGER.info('[INIT] ✅ V3 routes registered')
except Exception as _e:
    LOGGER.error(f'[INIT] ⚠️ V3 routes failed: {_e}', exc_info=True)


# ============================================================================
# 🕐 EXTERNAL CRON ENDPOINTS (for cron-job.org or similar)
# ============================================================================
# 
# These endpoints are designed for external cron services like cron-job.org
# that can reliably call your app on a schedule, even if Railway restarts.
#
# Setup on cron-job.org:
#   1. Create account at https://cron-job.org
#   2. Add jobs for each endpoint:
#      - 6:00 AM CT (12:00 UTC): /cron/daily-scout
#      - 8:00 AM CT (14:00 UTC): /cron/morning-alert
#      - 6:00 PM CT (00:00 UTC): /cron/evening-resolve
#   3. Set method to GET (or POST if you want)
#   4. Optional: Add secret header for security
# ============================================================================

# Secret key for cron validation (set in Railway env vars)
CRON_SECRET = os.getenv("CRON_SECRET", "ghost-cron-2024")












LOGGER.info("✅ 🕐 External Cron endpoints registered (/cron/*)")
# ── Step 12: Extracted route modules ─────────────────────────────────────
# Routes extracted from the monolith into clean APIRouter modules.
# Each include is wrapped in try/except so a single module failure
# won't prevent the rest of the app from starting.

_ROUTE_MODULES = [
    ("routes.accuracy", "accuracy"),
    ("routes.admin", "admin"),
    ("routes.alerts", "alerts"),
    ("routes.brain", "brain"),
    ("routes.cockpit", "cockpit_ext"),
    ("routes.cron", "cron"),
    ("routes.debug", "debug_ext"),
    ("routes.health_ext", "health_ext"),
    ("routes.misc_api", "misc_api"),
    ("routes.news_api", "news_api"),
    ("routes.predict", "predict_ext"),
]

for _mod, _name in _ROUTE_MODULES:
    try:
        _router = __import__(_mod, fromlist=["router"]).router
        APP.include_router(_router)
    except Exception as _e:
        print(f"[INIT] ⚠️  {_name} router unavailable: {_e}")

# ── External routers (watchlist, cockpit v2/v3) ─────────────────────────
try:
    from api.personal_watchlist_endpoints import router as personal_watchlist_router
    APP.include_router(personal_watchlist_router)
except Exception as e:
    print(f"[INIT] ⚠️  personal_watchlist unavailable: {e}")

try:
    from api.cockpit_v3_live_endpoints import router as cockpit_v3_router
    APP.include_router(cockpit_v3_router)
except Exception as e:
    print(f"[INIT] ⚠️  cockpit_v3 unavailable: {e}")


# ── Static file serving ────────────────────────────────────────────────
# MUST come after all APP.include_router() calls. A StaticFiles mount is a
# catch-all; placing it before routers causes it to intercept API paths.
_STATIC_MOUNT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if os.path.isdir(_STATIC_MOUNT_DIR):
    APP.mount("/static", StaticFiles(directory=_STATIC_MOUNT_DIR), name="static")
    print(f"[INIT] ✅ Static files mounted: {_STATIC_MOUNT_DIR}", flush=True)
else:
    print(f"[INIT] ⚠️  Static directory not found: {_STATIC_MOUNT_DIR}", flush=True)


# ── Lifecycle Events ─────────────────────────────────────────────────────

@APP.on_event("startup")
async def _on_startup():
    """Startup — delegates to engines/startup.py"""
    try:
        from engines.startup import _on_startup as _impl
        await _impl()
    except Exception as e:
        LOGGER.error(f"[STARTUP] Startup handler failed: {e}", exc_info=True)


@APP.on_event("shutdown")
async def _on_shutdown():
    """Shutdown — delegates to engines/shutdown.py"""
    try:
        from engines.shutdown import _on_shutdown as _impl
        await _impl()
    except Exception as e:
        LOGGER.error(f"[SHUTDOWN] Shutdown handler failed: {e}", exc_info=True)


# ── Entry Point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("wolf_app:APP", host="0.0.0.0", port=port, reload=False)
