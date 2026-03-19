"""Shared helper functions — extracted from wolf_app.py (Step 12)"""
# fmt: off
# ruff: noqa

import asyncio
import json
import logging
import os
import re
import time
import hashlib
import traceback
import math
import threading
import sqlite3
import uuid
import secrets
import hmac
from collections import deque, defaultdict
from datetime import UTC, datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

# ── Heartbeat: every background worker in this module calls _heartbeat_pulse ──
# Previously this was available via wolf_app.py's module globals.  After Step 12
# extraction, wolf_helpers.py is an independent module and must import it directly.
try:
    from core.heartbeat import pulse as _heartbeat_pulse
except Exception:
    def _heartbeat_pulse(name: str, **kw) -> None:  # type: ignore[misc]
        """No-op fallback when core.heartbeat is unavailable."""
        pass

# ── Inject all app-config constants ────────────────────────────────────────
# wolf_helpers.py was extracted from wolf_app.py (Step 12) and references many
# module-level constants from engines/app_config.py — STAGE4_ENABLED,
# HUNTER_CRYPTO_SYMBOLS, _LATEST_PREDICTIONS, get_edge_set, and many more.
# This injection mirrors the pattern used in engines/startup.py and all 16
# route modules. It runs here, before any helper function bodies execute.
try:
    import engines.app_config as _ac
    globals().update({k: v for k, v in vars(_ac).items() if not k.startswith("__")})
    del _ac
except Exception as _ac_err:
    import logging as _log
    _log.getLogger("ghost").warning(f"[WOLF_HELPERS] Could not inject app_config globals: {_ac_err}")

try:
    import httpx
except ImportError:
    httpx = None

try:
    import requests
    from requests.adapters import HTTPAdapter
except ImportError:
    requests = None

try:
    from pydantic import BaseModel
except ImportError:
    BaseModel = object

try:
    from state import APP_STATE, POOL, DB_URL, PREDICTION_HISTORY
except ImportError:
    APP_STATE = {}
    POOL = None
    DB_URL = ""
    PREDICTION_HISTORY = []

LOGGER = logging.getLogger("ghost")

import contextvars
_cv_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")
_cv_path: contextvars.ContextVar[str] = contextvars.ContextVar("path", default="-")
_cv_method: contextvars.ContextVar[str] = contextvars.ContextVar("method", default="-")

# ── Lazy STATE proxy — resolves to wolf_app.STATE at runtime ──────────
import sys as _sys

class _StateProxy:
    """Proxy that delegates attribute/item access to wolf_app.STATE at runtime."""
    def _s(self):
        m = _sys.modules.get("wolf_app")
        return m.STATE if (m and hasattr(m, "STATE")) else {}
    def __getitem__(self, k): return self._s()[k]
    def __setitem__(self, k, v):
        m = _sys.modules.get("wolf_app")
        if m and hasattr(m, "STATE"): m.STATE[k] = v
    def get(self, k, default=None): return self._s().get(k, default)
    def __contains__(self, k): return k in self._s()
    def update(self, *a, **kw): self._s().update(*a, **kw)
    def items(self): return self._s().items()
    def keys(self): return self._s().keys()
    def values(self): return self._s().values()
    def pop(self, k, *a): return self._s().pop(k, *a)

STATE = _StateProxy()

try:
    from core.price_quorum import PriceDecision, PriceProvider, get_price_quorum
except Exception:
    pass

# ── Core constants (must be defined before any function default args) ──
WOLF = os.getenv("WOLF_SYMBOL", "WOLF")
WOLF_SQLITE_PATH = os.getenv("WOLF_SQLITE_PATH", "data/wolf.db")
WOLF_PERSIST_MODE = os.getenv("WOLF_PERSIST_MODE", "auto").strip().lower()
REDIS_URL = os.getenv("REDIS_URL", "")
CSP_MODE = os.getenv("CSP_MODE", "dev").strip().lower()
APP_ENV = os.getenv("APP_ENV", os.getenv("ENV", "")).strip().lower()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
STAGE1_ENABLED = os.getenv("STAGE1_ENABLED", "1") not in ("0", "false", "no")
# Shared in-memory caches (populated by _init_security_tables at startup)
API_KEYS_DB: dict = {}
WEBHOOK_SUBSCRIPTIONS: dict = {}
try:
    from core.portfolio_store import get_portfolio_store
    PORTFOLIO_PERSISTENCE_ENABLED = True
except ImportError:
    PORTFOLIO_PERSISTENCE_ENABLED = False
    def get_portfolio_store():  # type: ignore
        return None

# Auth dependency
try:
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from fastapi import Security
    SECURITY_SCHEME = HTTPBearer(auto_error=False)
    if os.getenv("DISABLE_PREDICTION_AUTH", "0") == "1":
        AUTH_DEP = None  # type: ignore
    else:
        AUTH_DEP = Security(SECURITY_SCHEME)
except Exception:
    AUTH_DEP = None  # type: ignore
    SECURITY_SCHEME = None  # type: ignore

# ── Pydantic request/response models ──────────────────────────────
class AlertTemplateBody(BaseModel):
    signal_template: str | None = None
    status_template: str | None = None


class _RecordPriceBody(BaseModel):
    symbol: str
    price: float
    ts: int | None = None


class _ScoreBody(BaseModel):
    forecast_id: int
    through_ts: int


class _BacktestBody(BaseModel):
    symbol: str | None = None


class _PredictRunBody(BaseModel):
    symbol: str


class PositionBody(BaseModel):
    qty: float
    avg_cost: float


class OrderPlaceBody(BaseModel):
    symbol: str | None = WOLF
    side: str
    qty: float
    price: float | None = None
    note: str | None = None


class TelegramUpdate(BaseModel):
    update_id: int | None = None
    message: dict | None = None


class AiDecision(BaseModel):
    action: str
    confidence: int
    rationale: str
    risks: list[str] | None = None
    evidence: list[str] | None = None
    checklist: list[str] | None = None


class ChatRequest(BaseModel):
    question: str
    include_context: bool = False


class AlertToggle(BaseModel):
    hold: bool


class AlertConfigBody(BaseModel):
    mode: str | None = None  # fixed|band|trailing
    buy_pct: float | None = None
    sell_pct: float | None = None
    band_pct: float | None = None
    trail_sell_pct: float | None = None
    trail_buy_pct: float | None = None
    throttle_s: int | None = None
    throttle_buy_s: int | None = None
    throttle_sell_s: int | None = None
    vol_gate: int | None = None
    vol_lookback_days: int | None = None
    vol_k: float | None = None
    vol_ttl_s: int | None = None
    schedule_open_close: int | None = None
    schedule_window_s: int | None = None


class RuntimeConfigBody(BaseModel):
    price_ttl_s: int | None = None
    price_ttl_open_s: int | None = None
    news_ttl_s: int | None = None
    yahoo_first: int | None = None
    price_max_deviation_open: float | None = None
    reuters_feeds_on: int | None = None
    diag_collapse_dupes: int | None = None
    diag_ring_size: int | None = None
    overlay_enabled: int | None = None
    overlay_dt_minutes: int | None = None
    learning_enabled: int | None = None
    band_widen_factor: float | None = None
    forecast_step_s: int | None = None
    forecast_horizon_s: int | None = None


class ControlBody(BaseModel):
    action: str | None = None


class ModeBody(BaseModel):
    enabled: bool | None = None  # when true => live, false => sim


class AddPositionBody(BaseModel):
    symbol: str
    quantity: float
    price: float
    type: str | None = None


class PredFeedbackBody(BaseModel):
    t: int
    actual_price: float | None = None
    actual_pnl: float | None = None
    horizon_h: int | None = None
    ctx: dict[str, Any] | None = None


class TrainBody(BaseModel):
    days: int | None = None


class AgentControlBody(BaseModel):
    execution_enabled: bool | None = None
    advisory_only: bool | None = None


class CashBody(BaseModel):
    # Either provide total cash or split by market
    cash: float | None = None
    stock: float | None = None
    crypto: float | None = None


class PositionAddBody(BaseModel):
    symbol: str
    market: str = "stock"
    qty: float
    price_paid: float
    apply_to_cash: bool | None = False


class PositionsImportBody(BaseModel):
    positions: Any | None = None  # list[dict] or dict
    csv: str | None = None  # optional CSV text
    reset: bool | None = True  # when true, clear existing positions first
    apply_to_cash: bool | None = False
    set_focus: bool | None = True  # update WOLF qty/avg from a matching position if present


class WatchlistImportBody(BaseModel):
    stocks: str | None = None
    crypto: str | None = None


class TradeRequest(BaseModel):
    symbol: str
    qty: float | None = None
    notional: float | None = None
    side: str = "buy"  # buy or sell
    type: str = "market"  # market, limit, stop, stop_limit, trailing_stop
    time_in_force: str = "day"  # day, gtc, ioc, fok
    limit_price: float | None = None
    stop_price: float | None = None
    trail_price: float | None = None
    trail_percent: float | None = None
    extended_hours: bool = False
    client_order_id: str | None = None
    dry_run: bool = False  # If true, only check risk, don't submit




def _is_truthy(v: str | None) -> bool:
    if v is None:
        return False
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _is_live_enforced() -> bool:
    """Return True when simulation must be disabled and data must be live.

    This is an additive production guardrail. It is intentionally env-driven.
    """
    return _is_truthy(os.getenv("ENFORCE_LIVE", "0"))


def _get_git_sha() -> str | None:
    for k in (
        "RAILWAY_GIT_COMMIT_SHA",
        "GIT_COMMIT",
        "COMMIT_SHA",
        "SOURCE_VERSION",
        "RENDER_GIT_COMMIT",
        "VERCEL_GIT_COMMIT_SHA",
    ):
        v = (os.getenv(k) or "").strip()
        if v:
            return v
    return None


async def with_cap(coro, sec=2.5, fallback=None):
    """Hard timeout wrapper for external calls (Alpaca, price providers, Redis).
    Prevents 10s stalls that cause 499 errors from proxy timeout.
    """
    if anyio is None:
        # Fallback if anyio not available - use asyncio.wait_for
        try:
            return await asyncio.wait_for(coro, timeout=sec)
        except TimeoutError:
            LOGGER.warning(f"with_cap: timeout after {sec}s, returning fallback")
            return fallback
        except Exception as e:
            LOGGER.error(f"with_cap: error {e}, returning fallback")
            return fallback

    try:
        with anyio.fail_after(sec):
            return await coro
    except (TimeoutError, Exception) as e:
        LOGGER.warning(f"with_cap: timeout/error after {sec}s ({type(e).__name__}), returning fallback")
        return fallback


def _json500(msg: str):
    return JSONResponse({"error": "internal_error", "detail": msg}, status_code=500)


# NOTE: Exception handlers (_rt_handler, _ex_handler) are in wolf_app.py
# because they need @APP.exception_handler decorators.


def _parse_origins(val: str) -> list[str]:
    if not val:
        return ["*"]
    parts = [p.strip() for p in val.split(",") if p.strip()]
    return parts or ["*"]


def _compute_csp() -> str:
    # Force strict if APP_ENV indicates production
    strict = CSP_MODE in ("strict", "prod", "production") or APP_ENV in (
        "prod",
        "production",
    )
    if strict:
        return (
            "default-src 'self' https:; "
            "script-src 'self' https:; "
            "style-src 'self' 'unsafe-inline' https:; "
            "img-src 'self' https: data:; "
            "connect-src 'self' https:; "
            "frame-ancestors 'none'"
        )
    # Dev-friendly default; can be overridden by CSP_DEFAULT_SRC env
    default = os.getenv("CSP_DEFAULT_SRC")
    if default:
        return f"default-src {default}"
    # Explicit directives for dev to support Codespaces/Vite/etc.
    return (
        "default-src 'self' https: data: blob:; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval' https:; "
        "style-src 'self' 'unsafe-inline' https:; "
        "img-src 'self' https: data: blob:; "
        "connect-src 'self' https: ws: wss:; "
        "frame-ancestors 'none'"
    )


def _configure_logging():
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    json_on = os.getenv("LOG_JSON", "1") not in ("0", "false", "False", "no")
    root = logging.getLogger()
    root.setLevel(getattr(logging, level, logging.INFO))
    # Remove preexisting handlers to avoid duplicates when reloaded
    for h in root.handlers[:]:  # copy to avoid mutating while iterating
        root.removeHandler(h)
    handler = logging.StreamHandler()
    if json_on:
        handler.setFormatter(JsonFormatter())
    # Optional: collapse duplicate log messages within a short window
    try:
        dedup_window_s = float(os.getenv("LOG_DEDUP_WINDOW_S", "10"))
        dedup_min_repeats = int(os.getenv("LOG_DEDUP_MIN_REPEATS", "2"))
        if dedup_window_s > 0 and dedup_min_repeats >= 1:
            handler.addFilter(
                _LogDedupFilter(window_s=dedup_window_s, min_repeats=dedup_min_repeats)
            )
    except Exception:
        # never let logging config crash app
        pass
    root.addHandler(handler)


def should_create_prediction(symbol: str, confidence: float) -> tuple:
    """
    Returns (True, "") if prediction should proceed,
    (False, reason_string) if blocked by kill switch / confidence floor / rate limit.
    Queries PostgreSQL ghost_predictions for historical stats.
    """
    # ── Check 1: Confidence floor ──────────────────────────────────────────
    if confidence < _PREDICTION_GATE_CONFIDENCE_FLOOR:
        reason = f"CONFIDENCE FLOOR: Skipping {symbol} — confidence {confidence:.2f} below {_PREDICTION_GATE_CONFIDENCE_FLOOR}"
        LOGGER.info(reason)
        return (False, reason)

    # ── Check 2 & 3 require DB ─────────────────────────────────────────────
    try:
        from core.db_pool import get_sync_connection as _gate_get_conn
        with _gate_get_conn() as _gate_conn:
            try:
                _gate_conn.rollback()
            except Exception:
                pass
            _gate_cur = _gate_conn.cursor()

            # ── Check 2: Kill switch — per-symbol win rate ─────────────────
            _gate_cur.execute("""
                SELECT COUNT(*) AS total,
                       COALESCE(SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END), 0) AS wins
                FROM ghost_predictions
                WHERE symbol = %s
                  AND correct IS NOT NULL
                  AND (eval_version IS NULL OR eval_version NOT LIKE 'skip%%')
            """, (symbol,))
            _gate_row = _gate_cur.fetchone()
            _gate_total = _gate_row[0] if _gate_row else 0
            _gate_wins = _gate_row[1] if _gate_row else 0

            if _gate_total >= _PREDICTION_GATE_KILL_SWITCH_MIN_TRADES:
                _gate_winrate = round(_gate_wins / _gate_total * 100, 1)
                if _gate_winrate < _PREDICTION_GATE_KILL_SWITCH_MIN_WINRATE:
                    reason = f"KILL SWITCH: Skipping {symbol} — win rate {_gate_winrate}% over {_gate_total} trades"
                    LOGGER.info(reason)
                    _gate_cur.close()
                    return (False, reason)

            # ── Check 3: Rate limit — max N predictions per symbol per 24h ─
            _gate_cutoff = int(time.time()) - 86400
            _gate_cur.execute("""
                SELECT COUNT(*)
                FROM ghost_predictions
                WHERE symbol = %s
                  AND predicted_at > %s
            """, (symbol, _gate_cutoff))
            _gate_count = _gate_cur.fetchone()[0]
            _gate_cur.close()

            if _gate_count >= _PREDICTION_GATE_MAX_PER_DAY:
                reason = f"RATE LIMIT: Skipping {symbol} — already {_gate_count} predictions today"
                LOGGER.info(reason)
                return (False, reason)

    except Exception as _gate_err:
        # If DB check fails, allow prediction through (fail-open)
        LOGGER.warning(f"Prediction gate DB check failed for {symbol}: {_gate_err}")

    return (True, "")


# NOTE: _root_head_redirect route moved to routes/cockpit.py


def _classify_symbol_category(symbol: str) -> str:
    """
    Classify symbol into category: 'stocks', 'crypto', or 'vip'.

    Returns:
        'stocks' for stock symbols
        'crypto' for non-VIP crypto symbols
        'vip' for VIP coins (legacy, no longer tracked)
    """
    symbol_upper = symbol.upper()

    # VIP coins removed (unsupported by exchanges, caused provider storms)
    # Legacy VIP set: WEPE, LILPEPE, DORKL, SLOTH, APC

    # Check if in crypto symbols list
    if symbol_upper in CRYPTO_SYMBOLS or symbol_upper in HUNTER_CRYPTO_SYMBOLS:
        return "crypto"

    # Default to stocks
    return "stocks"


def api_corporate_actions() -> dict[str, Any]:
    """Expose known corporate action metadata to the UI.

    Shape:
      {
        "actions": {
            "WOLF": {
               ... original registry fields ...,
               "has_reverse_split": bool,
               "reverse_split_display": "120:1" | null
            }, ...
        },
        "symbols": ["WOLF", ...]
      }
    """
    actions: dict[str, dict[str, Any]] = {}
    for sym, meta in DELISTED_SYMBOLS.items():
        # Copy to avoid mutating original registry
        m = dict(meta)
        ratio = m.get("reverse_split_ratio")
        m["has_reverse_split"] = bool(ratio)
        m["reverse_split_display"] = f"{int(ratio)}:1" if ratio else None
        actions[sym] = m
    return {"actions": actions, "symbols": sorted(actions.keys())}


def _adjust_pnl_for_corporate_action(
    symbol: str, entry_price: float, current_price: float, qty: float
) -> dict[str, Any]:
    """
    Adjust P&L calculations for corporate actions (reverse splits, stock splits, spinoffs).

    Returns dict with:
      - adjusted_entry: Entry price adjusted for corporate action
      - adjusted_qty: Quantity adjusted for corporate action
      - pnl_abs: Absolute P&L (adjusted)
      - pnl_pct: Percentage P&L (adjusted)
      - adjustment_note: Human-readable explanation
      - unadjusted_pnl_abs: Original (misleading) P&L for comparison
      - unadjusted_pnl_pct: Original (misleading) P&L % for comparison
    """
    action = DELISTED_SYMBOLS.get(symbol)

    # Calculate unadjusted values first
    unadjusted_pnl_abs = (current_price - entry_price) * qty
    unadjusted_pnl_pct = (
        ((current_price - entry_price) / entry_price * 100.0) if entry_price > 0 else 0.0
    )

    if not action or not action.get("reverse_split_ratio"):
        # No corporate action - return original values
        return {
            "adjusted_entry": entry_price,
            "adjusted_qty": qty,
            "pnl_abs": unadjusted_pnl_abs,
            "pnl_pct": unadjusted_pnl_pct,
            "adjustment_note": "",
            "unadjusted_pnl_abs": unadjusted_pnl_abs,
            "unadjusted_pnl_pct": unadjusted_pnl_pct,
            "has_adjustment": False,
        }

    # Reverse split: multiply entry price, divide quantity
    # Example: 120:1 split means $3.30 becomes $396, 909 shares becomes 7.58 shares
    ratio = float(action["reverse_split_ratio"])
    adjusted_entry = entry_price * ratio
    adjusted_qty = qty / ratio

    # Calculate adjusted P&L
    pnl_abs = (current_price - adjusted_entry) * adjusted_qty
    pnl_pct = (
        ((current_price - adjusted_entry) / adjusted_entry * 100.0) if adjusted_entry > 0 else 0.0
    )

    note = f"Adjusted for {ratio}:1 reverse split ({action.get('date')})"

    return {
        "adjusted_entry": adjusted_entry,
        "adjusted_qty": adjusted_qty,
        "pnl_abs": pnl_abs,
        "pnl_pct": pnl_pct,
        "adjustment_note": note,
        "unadjusted_pnl_abs": unadjusted_pnl_abs,
        "unadjusted_pnl_pct": unadjusted_pnl_pct,
        "has_adjustment": True,
    }


def _clamp(x: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(x)))
    except Exception:
        return lo


def _display_price_triplet() -> tuple[float | None, float | None, str | None]:
    """Helper to get (price, prev_close, provider)."""
    try:
        price, prev, provider = get_wolf_price()
        return price, prev, provider
    except Exception:
        return None, None, "unavailable"


def _estimate_drift_and_conf(
    price: float | None,
    prev: float | None,
    news_score: float | None,
    events_score: float | None,
    urgency: str | None,
) -> tuple[float, int]:
    """Return (drift_daily_pct, confidence0to100) using price move + research context.

    Components:
    - Price persistence: 30% of current move (per day) capped at ±15%/day
    - News tilt: up to ±2%/day equivalent for strong sentiment
    - Events tilt (SEC filings): up to ±5%/day equivalent for critical events
    - Confidence: base 60 boosted by |news|, |events|, and urgency (critical/high)
    """
    try:
        chg_pct = 0.0
        if price is not None and prev and prev > 0:
            chg_pct = (price - prev) / prev * 100.0
        ns = (news_score if isinstance(news_score, (int, float)) else 0.0) or 0.0
        es = (events_score if isinstance(events_score, (int, float)) else 0.0) or 0.0
        # 30% persistence + 2% of news + 5% of events (scaled from [-1,1])
        drift_daily = 0.3 * (chg_pct / 100.0) + 0.02 * ns + 0.05 * es
        drift_daily = _clamp(drift_daily, -0.15, 0.15)  # cap to +/-15%/day drift
        # Confidence increases with |news|, |events|, |move|, plus urgency boost
        urg_boost = 0.0
        if urgency:
            u = str(urgency).lower()
            if "critical" in u:
                urg_boost = 10.0
            elif "high" in u:
                urg_boost = 5.0
        conf = int(
            round(
                60.0
                + 12.0 * min(1.0, abs(ns))
                + 12.0 * min(1.0, abs(es))
                + 0.3 * min(50.0, abs(chg_pct))
                + urg_boost
            )
        )
        conf = int(_clamp(conf, 30, 95))
        return drift_daily, conf
    except Exception:
        return 0.0, 50


def _build_forecast_series(horizon_h: int = 48) -> dict[str, Any]:
    now_ts = int(time.time())
    price, prev, _ = _display_price_triplet()
    qty, avg = _get_portfolio_qty_and_avg()  # Use helper to read from positions array
    # Display price baseline
    p0 = price if price is not None else (prev if prev is not None else (avg if avg > 0 else None))
    if p0 is None:
        p0 = 0.0
    # Pull research context: latest news + recent SEC filings
    ns = None
    events_score = 0.0
    urgency = None
    research_used = {"news": False, "filings": False}
    if PRED_USE_NEWS:
        try:
            ns = (get_wolf_news(limit=3).get("news_signal") or {}).get("score")
            research_used["news"] = True
        except Exception:
            ns = None
    if PRED_USE_FILINGS:
        try:
            f = _get_filings_signal(WOLF)
            if f:
                events_score = float(f.get("events_score") or 0.0)
                urgency = f.get("max_urgency")
                research_used["filings"] = True
        except Exception:
            events_score = 0.0
            urgency = None
    drift_daily, conf = _estimate_drift_and_conf(
        price if price is not None else p0, prev, ns, events_score, urgency
    )
    # Optional: consult research blueprint aggregate signal for an additional confidence hint
    agg_research = None
    if RESEARCH_BLUEPRINT_ON:
        try:
            agg_research = build_research_snapshot(WOLF, asset_type="stock").get("aggregate")
            # Light-touch adjustment: nudge confidence toward aggregate confidence midpoint
            if isinstance(agg_research, dict):
                rc = int(agg_research.get("confidence") or 0)
                conf = int(_clamp((conf * 0.8 + rc * 0.2), 30, 95))
        except Exception:
            agg_research = None
    sigma_d = max(0.001, float(PRED_SIGMA_DAILY))
    z = max(0.1, float(PRED_Z))
    step_h = max(1, int(PRED_STEP_H))
    points = []
    # generate from +step_h to horizon
    for h in range(step_h, horizon_h + 1, step_h):
        t = now_ts + h * 3600
        # mid accumulates drift
        mid = float(p0) * (1.0 + drift_daily * (h / 24.0))
        # band grows with sqrt time
        band = z * float(p0) * sigma_d * math.sqrt(h / 24.0)
        lo = max(0.0, mid - band)
        hi = mid + band
        pnl_mid = (mid - avg) * qty
        pnl_lo = (lo - avg) * qty
        pnl_hi = (hi - avg) * qty
        points.append(
            {
                "t": t,
                "price_mid": round(mid, 4),
                "price_lo": round(lo, 4),
                "price_hi": round(hi, 4),
                "pnl_mid": round(pnl_mid, 2),
                "pnl_lo": round(pnl_lo, 2),
                "pnl_hi": round(pnl_hi, 2),
            }
        )
    summary = {
        "confidence": conf,
        "drift_daily_pct": round(drift_daily * 100.0, 4),
        "pnl_48h_mid": (points[-1]["pnl_mid"] if points else None),
        "research_used": research_used,
        "research_aggregate": agg_research,
    }
    return {
        "ticker": WOLF,
        "as_o": now_ts,
        "horizon_h": horizon_h,
        "step_h": step_h,
        "points": points,
        "summary": summary,
    }


def _get_filings_signal(symbol: str) -> dict[str, Any] | None:
    """Aggregate SEC filings into a compact event signal for predictions.
    Returns dict with events_score in [-1,1] and max_urgency.
    Cached for FILINGS_TTL_S.
    """
    now = time.time()
    ts = float(FILINGS_CACHE.get("ts") or 0)
    if (now - ts) <= FILINGS_TTL_S and FILINGS_CACHE.get("data"):
        return FILINGS_CACHE["data"]
    try:
        # Import inside to avoid hard dependency at import time
        from core.edgar_integration import EDGARClient  # type: ignore

        client = EDGARClient()
        # Fetch company-specific filings; limit to last ~20
        filings = client.get_company_filings(symbol, limit=20)
        if not filings:
            FILINGS_CACHE.update({"ts": now, "data": None})
            return None
        # Score recent filings within ~7 days, weight by recency and urgency
        cutoff = int(time.time()) - 7 * 86400
        score = 0.0
        weight_sum = 0.0
        max_urgency = "medium"
        has_bankruptcy = False
        has_delisting = False
        has_product = False
        for f in filings:
            try:
                fd = int(getattr(f, "filing_date", 0) or 0)
                if fd < cutoff:
                    continue
                text = (getattr(f, "description", "") or "").lower()
                items = set(getattr(f, "items", []) or [])
                urgency = (getattr(f, "urgency", "medium") or "medium").lower()
                # Flags
                if any(k in text for k in ["bankruptcy", "chapter 11", "chapter 7"]):
                    has_bankruptcy = True
                if "3.01" in items:
                    has_delisting = True
                if any(k in text for k in ["launch", "launched", "introduc", "product"]):
                    has_product = True
                # Map urgency to weight
                u_w = 1.0
                if "critical" in urgency:
                    u_w = 2.0
                    max_urgency = "critical"
                elif "high" in urgency:
                    u_w = max(u_w, 1.5)
                    if max_urgency not in ("critical",):
                        max_urgency = "high"
                # Sentiment contribution from filing
                sent = float(getattr(f, "sentiment_score", 0.0) or 0.0)
                # Event nudges
                e = 0.0
                if has_bankruptcy or has_delisting:
                    e -= 1.0
                if has_product:
                    e += 0.3
                # Combine
                combined = _clamp(sent + e, -1.0, 1.0)
                # Recency weight (linear within 7 days)
                rec_w = max(0.2, min(1.0, 1.0 - (time.time() - fd) / (7 * 86400)))
                w = u_w * rec_w
                score += combined * w
                weight_sum += w
            except Exception:
                continue
        events_score = 0.0 if weight_sum == 0 else _clamp(score / weight_sum, -1.0, 1.0)
        data = {
            "events_score": events_score,
            "max_urgency": max_urgency,
            "flags": {
                "bankruptcy": has_bankruptcy,
                "delisting": has_delisting,
                "product_launch": has_product,
            },
        }
        FILINGS_CACHE.update({"ts": now, "data": data})
        return data
    except Exception:
        FILINGS_CACHE.update({"ts": now, "data": None})
        return None


def _forecast_summary_for_snapshot() -> dict[str, Any]:
    try:
        f = _build_forecast_series(48)
        s = f.get("summary") or {}
        return {
            "enabled": True,
            "label": "Ghost Predictions",
            "horizon_h": 48,
            "confidence": s.get("confidence"),
            "pnl_48h_mid": s.get("pnl_48h_mid"),
        }
    except Exception:
        return {
            "enabled": True,
            "label": "Ghost Predictions",
            "horizon_h": 48,
            "confidence": None,
            "pnl_48h_mid": None,
        }


def _build_market_status_with_indices(is_open: bool, next_open_ts: int) -> dict[str, Any]:
    """
    Build market status with major indices (SPY, QQQ, VIX) for UI display.
    Returns: {open, next_open_ts, indices: [{symbol, price, change_pct}]}
    """
    market_data = {"open": is_open, "next_open_ts": next_open_ts, "indices": []}

    # Fetch major indices via multi-provider fallback (reuses _get_index_price)
    indices_symbols = [("SPY", "SPY"), ("QQQ", "QQQ"), ("^VIX", "VIX")]
    try:
        for sym, display_sym in indices_symbols:
            try:
                price, prev = _get_index_price(sym)
                if price and price > 0 and prev and prev > 0:
                    change_pct = ((price - prev) / prev) * 100.0
                    market_data["indices"].append({
                        "symbol": display_sym,
                        "price": round(price, 2),
                        "change_pct": round(change_pct, 2),
                    })
            except Exception as e:
                LOGGER.debug(f"Failed to fetch index {sym}: {e}")
                continue
    except Exception as e:
        LOGGER.warning(f"Failed to fetch market indices: {e}")

    return market_data


def _generate_forecast_grid(symbol: str = WOLF) -> dict[str, Any]:
    """
    Generate aligned forecast grid with persistence.
    Returns: {asof, horizon_s, points:[{t,p}], band:{lo,hi}, meta}
    Persists to data/forecast_{symbol}.json for reuse.
    """
    now_ts = int(time.time())
    step_s = FORECAST_STEP_S
    horizon_s = FORECAST_HORIZON_S

    # Try to load existing grid
    try:
        if os.path.exists(FORECAST_GRID_PATH):
            with open(FORECAST_GRID_PATH) as f:
                cached = json.load(f)
            # Check if still valid (< 24h old and same config)
            cached_asof = cached.get("aso", 0)
            cached_step = cached.get("meta", {}).get("step_s", 0)
            cached_horizon = cached.get("horizon_s", 0)
            if (
                (now_ts - cached_asof) < FORECAST_MAX_AGE_S
                and cached_step == step_s
                and cached_horizon == horizon_s
            ):
                return cached
    except Exception as e:
        print(f"[FORECAST] Failed to load cached grid: {e}")

    # Generate new grid
    price, prev, _ = _display_price_triplet()
    qty, avg = _get_portfolio_qty_and_avg()  # Use helper to read from positions array
    p0 = price if price is not None else (prev if prev is not None else (avg if avg > 0 else 25.0))

    # Get drift estimate
    ns = None
    try:
        ns = (get_wolf_news(limit=1).get("news_signal") or {}).get("score")
    except Exception as e:
        LOGGER.debug(f"news_signal_fetch_failed: {e}")
    # Provide defaults for events/urgency so signature is satisfied
    drift_daily, conf = _estimate_drift_and_conf(
        price if price is not None else p0,
        prev,
        ns,
        0.0,  # events_score default when filings not consulted here
        None,  # urgency not available in this lightweight path
    )
    sigma_d = max(0.001, float(PRED_SIGMA_DAILY))
    z = max(0.1, float(PRED_Z))

    # Build time grid
    t_grid = []
    t = now_ts
    while t <= now_ts + horizon_s:
        t_grid.append(t)
        t += step_s

    # Generate forecast points and bands
    points = []
    lo_band = []
    hi_band = []

    for t in t_grid:
        h_elapsed = (t - now_ts) / 3600.0
        if h_elapsed < 0:
            continue
        # mid accumulates drift
        mid = float(p0) * (1.0 + drift_daily * (h_elapsed / 24.0))
        # band grows with sqrt time
        band = z * float(p0) * sigma_d * math.sqrt(max(0.01, h_elapsed / 24.0))
        lo = max(0.0, mid - band)
        hi = mid + band

        points.append({"t": t, "p": round(mid, 4)})
        lo_band.append({"t": t, "p": round(lo, 4)})
        hi_band.append({"t": t, "p": round(hi, 4)})

    result = {
        "aso": now_ts,
        "horizon_s": horizon_s,
        "points": points,
        "band": {"lo": lo_band, "hi": hi_band},
        "meta": {
            "symbol": symbol,
            "con": conf / 100.0,  # Store as 0-1
            "model": "ghost-av1",
            "step_s": step_s,
            "p0": round(p0, 4),
            "drift_daily": round(drift_daily, 6),
        },
    }

    # Persist
    try:
        os.makedirs("data", exist_ok=True)
        with open(FORECAST_GRID_PATH, "w") as f:
            json.dump(result, f, indent=2)
    except Exception as e:
        print(f"[FORECAST] Failed to persist grid: {e}")

    return result


def _collect_actual_prices(t_grid: list[int], symbol: str = WOLF) -> dict[str, Any]:
    """
    Collect actual prices at grid timestamps <= now.
    Queries realized_prices table first, then falls back to current/prev price.
    Returns: {asof, points:[{t,p}], src, latency_ms}
    """
    now_ts = int(time.time())
    points = []
    src = "unavailable"
    latency_start = time.time()

    # Filter to past timestamps only
    past_grid = [t for t in t_grid if t <= now_ts]

    if not past_grid:
        return {"aso": now_ts, "points": [], "src": "none", "latency_ms": 0}

    # NEW: Query realized_prices table for historical actuals (preferred method)
    try:
        import sqlite3

        # BUG FIX (Jan 6, 2026): Use context manager to prevent connection leaks
        with sqlite3.connect(WOLF_SQLITE_PATH) as conn:
            cur = conn.cursor()

            # For each timestamp, find closest tick within ±5min window
            tolerance_s = 300  # 5 minutes
            for t in past_grid:
                cur.execute(
                    """SELECT price FROM realized_prices
                       WHERE symbol=? AND ABS(ts - ?) < ?
                       ORDER BY ABS(ts - ?) ASC LIMIT 1""",
                    (symbol, t, tolerance_s, t),
                )
                row = cur.fetchone()
                if row and row[0] is not None:
                    points.append({"t": t, "p": round(float(row[0]), 4)})

        if points:
            src = "history"
            latency_ms = int((time.time() - latency_start) * 1000)
            return {
                "aso": now_ts,
                "points": points,
                "src": src,
                "latency_ms": latency_ms,
            }
    except Exception as e:
        # SQLite not available or table doesn't exist - fall through to live price
        print(f"[ACTUAL] Failed to query history: {e}")

    # Use current/prev price for all timestamps
    try:
        price, prev, provider = get_wolf_price()
        if price is not None:
            src = provider or "live"
            # Fill most recent points with current price
            # Older points get prev_close if available
            for t in past_grid:
                age_h = (now_ts - t) / 3600.0
                if age_h < 24:  # Recent: use current price
                    points.append({"t": t, "p": round(float(price), 4)})
                elif prev is not None:  # Older: use prev_close
                    points.append({"t": t, "p": round(float(prev), 4)})
        elif prev is not None:
            src = "prev_close"
            for t in past_grid:
                points.append({"t": t, "p": round(float(prev), 4)})
    except Exception as e:
        print(f"[ACTUAL] Failed to collect prices: {e}")
        src = "error"

    latency_ms = int((time.time() - latency_start) * 1000)

    return {"aso": now_ts, "points": points, "src": src, "latency_ms": latency_ms}


def _compute_forecast_accuracy(
    forecast_points: list[dict], actual_points: list[dict]
) -> dict[str, Any]:
    """
    Compute accuracy metrics where forecast and actual overlap.
    Returns: {by_t:[{t,err,ape}], summary:{map,rmse,bias}}
    """
    # Build lookup for actual prices by timestamp
    actual_map = {p["t"]: p["p"] for p in actual_points}

    by_t = []
    errors = []
    apes = []

    for fp in forecast_points:
        t = fp["t"]
        if t in actual_map:
            forecast_p = fp["p"]
            actual_p = actual_map[t]
            err = actual_p - forecast_p
            ape = abs(err) / max(0.01, actual_p)

            by_t.append({"t": t, "err": round(err, 4), "ape": round(ape, 6)})
            errors.append(err)
            apes.append(ape)

    # Compute summary stats
    summary = {"map": 0.0, "rmse": 0.0, "bias": 0.0}

    if errors:
        summary["map"] = round(sum(apes) / len(apes), 6)
        summary["rmse"] = round(math.sqrt(sum(e**2 for e in errors) / len(errors)), 4)
        summary["bias"] = round(sum(errors) / len(errors), 4)

    return {"by_t": by_t, "summary": summary}


def _build_two_line_forecast(symbol: str = WOLF) -> dict[str, Any]:
    """
    Build complete two-line overlay data: Ghost forecast + Live actual + Accuracy.
    Returns: {forecast:{...}, actual:{...}, accuracy:{...}}
    """
    try:
        # Generate or load forecast grid
        forecast = _generate_forecast_grid(symbol)

        # Collect actual prices for grid timestamps
        t_grid = [p["t"] for p in forecast["points"]]
        actual = _collect_actual_prices(t_grid, symbol)

        # Compute accuracy
        accuracy = _compute_forecast_accuracy(forecast["points"], actual["points"])

        return {"forecast": forecast, "actual": actual, "accuracy": accuracy}
    except Exception as e:
        print(f"[TWO_LINE] Failed to build overlay: {e}")
        # Return safe defaults
        now_ts = int(time.time())
        return {
            "forecast": {
                "aso": now_ts,
                "horizon_s": FORECAST_HORIZON_S,
                "points": [],
                "band": {"lo": [], "hi": []},
                "meta": {"symbol": symbol, "con": 0.6, "model": "ghost-av1"},
            },
            "actual": {
                "aso": now_ts,
                "points": [],
                "src": "unavailable",
                "latency_ms": 0,
            },
            "accuracy": {
                "by_t": [],
                "summary": {"map": 0.0, "rmse": 0.0, "bias": 0.0},
            },
        }


def _build_actual_series(lookback_h: int = 48) -> list[dict[str, Any]]:
    """
    Build actual price series from realized_prices table for overlay chart.
    Returns list of {t: timestamp, p_actual: price} for the last lookback_h hours.
    """
    try:
        since_ts = int(time.time()) - (lookback_h * 3600)
        actual = _realized_since(WOLF, since_ts)
        if not actual:
            return []
        # Format as {t, p_actual}
        series = [{"t": int(ts), "p_actual": round(float(price), 4)} for (ts, price) in actual]
        return series
    except Exception:
        return []


def _ensure_ai_storage():
    try:
        os.makedirs(AI_DATA_DIR, exist_ok=True)
    except Exception:
        pass


def _is_ai_memory_auth_required() -> bool:
    try:
        return bool(int(os.getenv("AI_MEMORY_READ_AUTH", "0")))
    except Exception:
        return False


def _legacy_snapshot_to_decision(row: tuple[Any, ...]) -> dict[str, Any]:
    (
        ts,
        price,
        prev,
        qty,
        avg,
        news_score,
        features_json,
        label_next_move,
        advisory,
        confidence,
    ) = row
    try:
        features = (
            json.loads(features_json or "{}")
            if isinstance(features_json, str)
            else dict(features_json or {})
        )
    except Exception:
        features = {}
    # Enrich legacy features with position context
    try:
        features.setdefault("qty", float(qty or 0.0))
        features.setdefault("avg_cost", float(avg or 0.0))
        features.setdefault("news_score", float(news_score or 0.0))
    except Exception as e:
        LOGGER.warning(f"feature_enrichment_failed: {e}")
    label = int(label_next_move or 0)
    action = "HOLD"
    if label > 0:
        action = "BUY"
    elif label < 0:
        action = "SELL"
    raw_conf = float(confidence or 0.0)
    conf = raw_conf / 100.0 if raw_conf > 1.0 else raw_conf
    conf = _clamp(conf, 0.0, 1.0)
    return {
        "ts": int(ts or time.time()),
        "symbol": WOLF,
        "price": float(price) if price is not None else None,
        "prev_close": float(prev or 0.0),
        "news_score": float(news_score or 0.0) if news_score is not None else None,
        "features": features,
        "action": action,
        "confidence": conf,
        "reasoning": str(advisory or ""),
        "model_version": "legacy-snapshot-v1",
        "model_type": "knn",
        "executed": False,
    }


def _serialize_memory_decision(row: dict[str, Any] | sqlite3.Row | Any) -> dict[str, Any]:
    try:
        if hasattr(row, "keys") and not isinstance(row, dict):
            data = {k: row[k] for k in row.keys()}
        elif isinstance(row, dict):
            data = dict(row)
        else:
            data = dict(row)
    except Exception:
        data = dict(row or {})
    features_raw = data.get("features")
    if isinstance(features_raw, str):
        try:
            features = json.loads(features_raw or "{}")
        except Exception:
            features = {}
    else:
        features = features_raw or {}
    action = (data.get("action") or "HOLD").upper()
    label = 0
    if action == "BUY":
        label = 1
    elif action == "SELL":
        label = -1
    conf_float = float(data.get("confidence") or 0.0)
    if conf_float <= 1.0:
        confidence_pct = int(round(conf_float * 100))
    else:
        confidence_pct = int(round(conf_float))
        conf_float = confidence_pct / 100.0
    serialized = {
        "id": data.get("id"),
        "ts": int(data.get("ts") or 0),
        "symbol": data.get("symbol") or WOLF,
        "price": (float(data.get("price") or 0.0) if data.get("price") is not None else None),
        "prev": (
            float(data.get("prev_close") or 0.0) if data.get("prev_close") is not None else None
        ),
        "news_score": data.get("news_score"),
        "features": features,
        "action": action,
        "label_next_move": label,
        "reasoning": data.get("reasoning") or "",
        "confidence": confidence_pct,
        "confidence_float": conf_float,
        "model_version": data.get("model_version") or "unknown",
        "model_type": data.get("model_type") or "unknown",
        "outcome_1h": data.get("outcome_1h"),
        "outcome_24h": data.get("outcome_24h"),
        "outcome_7d": data.get("outcome_7d"),
        "executed": bool(data.get("executed")),
    }
    return serialized


def _ai_memory_store_decision(payload: dict[str, Any]) -> None:
    if AI_MEMORY_STORE is None:
        return
    try:
        AI_MEMORY_STORE.store_decision(payload)
    except Exception as e:
        LOGGER.exception("ai_memory_store_failed", extra={"error": str(e)})


def _ai_memory_append(row: dict[str, Any]) -> None:
    # Map legacy row structure into AIMemory format
    decision = {
        "ts": int(row.get("ts") or time.time()),
        "symbol": str(row.get("symbol") or WOLF),
        "price": row.get("price"),
        "prev_close": row.get("prev"),
        "news_score": row.get("news_score"),
        "features": row.get("features") or {},
        "action": row.get("action")
        or (
            "BUY"
            if int(row.get("label_next_move") or 0) > 0
            else "SELL"
            if int(row.get("label_next_move") or 0) < 0
            else "HOLD"
        ),
        "confidence": _clamp(
            (
                (float(row.get("confidence") or 0.0) / 100.0)
                if float(row.get("confidence") or 0.0) > 1.0
                else float(row.get("confidence") or 0.0)
            ),
            0.0,
            1.0,
        ),
        "reasoning": row.get("advisory") or row.get("reasoning") or "",
        "model_version": row.get("model_version") or "ghost-heuristic-v1",
        "model_type": row.get("model_type") or "knn",
        "executed": bool(row.get("executed")),
    }
    # Preserve position context inside features for RL/analysis
    try:
        feats = dict(decision["features"])
        feats.setdefault("qty", float(row.get("qty") or 0.0))
        feats.setdefault("avg_cost", float(row.get("avg") or 0.0))
        decision["features"] = feats
    except Exception as e:
        LOGGER.warning(f"decision_feature_enrichment_failed: {e}")
    # Maintain small ring buffer for quick access/fallbacks
    try:
        ring_entry = {
            "ts": decision["ts"],
            "price": decision.get("price"),
            "prev": decision.get("prev_close"),
            "qty": float(row.get("qty") or decision["features"].get("qty", 0.0)),
            "avg": float(row.get("avg") or decision["features"].get("avg_cost", 0.0)),
            "news_score": decision.get("news_score"),
            "features": decision.get("features") or {},
            "label_next_move": _label_from_action(decision.get("action")),
            "action": decision.get("action"),
            "advisory": decision.get("reasoning", ""),
            "confidence": int(round((decision.get("confidence") or 0.0) * 100)),
        }
        AI_MEMORY_RING.append(ring_entry)
    except Exception as e:
        LOGGER.warning(f"memory_ring_append_failed: {e}")
    _ai_memory_store_decision(decision)


def _migrate_legacy_ai_memory() -> int:
    if AI_MEMORY_STORE is None:
        return 0
    if not os.path.exists(AI_LEGACY_DB_PATH):
        return 0
    try:
        import sqlite3

        cur = AI_MEMORY_STORE.conn.execute("SELECT COUNT(*) FROM ai_memory")
        if int(cur.fetchone()[0] or 0) > 0:
            return 0
        legacy_conn = sqlite3.connect(AI_LEGACY_DB_PATH)
        legacy_cur = legacy_conn.cursor()
        legacy_cur.execute(
            """
            SELECT ts, price, prev, qty, avg, news_score, features_json, label_next_move, advisory, confidence
            FROM ai_snapshots
            ORDER BY ts ASC
            """
        )
        rows = legacy_cur.fetchall() or []
        migrated = 0
        for row in rows:
            payload = _legacy_snapshot_to_decision(row)
            _ai_memory_store_decision(payload)
            migrated += 1
        legacy_conn.close()
        LOGGER.info("ai_memory_migrated", extra={"count": migrated})
        return migrated
    except Exception as e:
        LOGGER.exception("ai_memory_migrate_failed", extra={"error": str(e)})
        return 0


def _label_from_action(action: str | None) -> int:
    a = (action or "").strip().upper()
    if a == "BUY":
        return 1
    if a == "SELL":
        return -1
    return 0


def _extract_features(
    price: float | None,
    prev: float | None,
    qty: float,
    avg: float,
    news_score: float | None,
) -> dict[str, float]:
    p = float(price) if price is not None else float(prev or avg or 0.0)
    pv = float(prev or p)
    ret_1d = ((p - pv) / pv) if pv else 0.0
    dist_avg = ((p / avg) - 1.0) if avg else 0.0
    ns = float(news_score) if isinstance(news_score, (int, float)) else 0.0
    return {
        "ret_1d": float(ret_1d),
        "dist_avg": float(dist_avg),
        "news": ns,
        "qty": float(qty),
    }


def _ai_neighbors(
    cur_feats: dict[str, float],
    symbol: str | None = None,
    price: float | None = None,
    k: int = 50,
) -> list[dict[str, Any]]:
    if AI_MEMORY_STORE is None:
        return []
    current_state: dict[str, Any] = {"features": cur_feats or {}}
    if symbol:
        current_state["symbol"] = symbol
    if price is not None:
        current_state["price"] = price
    try:
        similar = AI_MEMORY_STORE.find_similar_situations(current_state, k=k)
        return [_serialize_memory_decision(row) for row in similar]
    except Exception as e:
        LOGGER.debug("ai_neighbors_failed", extra={"error": str(e)})
        return []


def _ai_infer(
    cur_feats: dict[str, float],
    *,
    symbol: str | None = None,
    price: float | None = None,
) -> tuple[float, float, list[str], list[dict[str, Any]]]:
    # return (gps0to10, conf0to100, reasons[], analogs[])
    neighbors = _ai_neighbors(cur_feats, symbol=symbol or WOLF, price=price, k=30)
    if neighbors:
        ups = sum(1 for n in neighbors if int(n.get("label_next_move") or 0) > 0)
        downs = sum(1 for n in neighbors if int(n.get("label_next_move") or 0) < 0)
        total = max(1, len(neighbors))
        prob_up = ups / total
        prob_down = downs / total
    else:
        # heuristic fallback using features
        prob_up = (
            0.5
            + 0.3 * _clamp(cur_feats.get("ret_1d", 0.0), -0.05, 0.05)
            + 0.1 * _clamp(cur_feats.get("news", 0.0), -1.0, 1.0)
        )
        prob_up = _clamp(prob_up, 0.05, 0.95)
        prob_down = 1.0 - prob_up
    gps = 10.0 * max(prob_up, prob_down)
    conf = int(round(100.0 * abs(prob_up - prob_down)))
    # reasons (simple)
    reasons = []
    try:
        reasons.append(f"Momentum {cur_feats.get('ret_1d', 0.0) * 100.0:+.2f}% vs prev close")
        reasons.append(f"Dist to avg {cur_feats.get('dist_avg', 0.0) * 100.0:+.2f}%")
        ns = cur_feats.get("news", 0.0)
        reasons.append(
            "News tilt bullish"
            if ns > 0.2
            else ("News tilt bearish" if ns < -0.2 else "News neutral")
        )
    except Exception as e:
        LOGGER.warning(f"prediction_reasoning_build_failed: {e}")
    analogs = []
    try:
        for n in neighbors[:3]:
            analogs.append(
                {
                    "ts": n.get("ts"),
                    "label": int(n.get("label_next_move") or 0),
                    "action": n.get("action"),
                    "confidence": n.get("confidence"),
                    "outcome_24h": n.get("outcome_24h"),
                }
            )
    except Exception as e:
        LOGGER.warning(f"analog_neighbors_build_failed: {e}")
    return float(gps), float(conf), reasons, analogs


def _ensure_dir_for_file(path: str):
    """Ensure the directory for a given file path exists."""
    try:
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d, exist_ok=True)
    except Exception:
        pass


def _init_forecast_tables():
    """Create SQLite tables for forecast tracking (spec-compliant schema)."""
    import sqlite3

    try:
        # BUG FIX (Jan 6, 2026): Use context manager to prevent connection leaks
        with sqlite3.connect(WOLF_SQLITE_PATH) as conn:

            # Spec-compliant forecast table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS forecast_48h (
                    id INTEGER PRIMARY KEY,
                    ts_issued INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    horizon_hours INTEGER NOT NULL,
                    price_now REAL NOT NULL,
                    price_pred_mid REAL NOT NULL,
                    price_pred_lo REAL,
                    price_pred_hi REAL,
                    pnl_pred_mid REAL,
                    confidence REAL,
                    model TEXT NOT NULL,
                    features_json TEXT
                )
            """
            )

            # Spec-compliant actuals table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS price_actuals (
                    ts INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    PRIMARY KEY (symbol, ts)
                )
            """
            )

            # Legacy tables (keep for backward compatibility)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS forecasts (
                    id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    as_of INTEGER NOT NULL,
                    hours INTEGER NOT NULL,
                    path_mid TEXT NOT NULL,
                    path_lo TEXT,
                    path_hi TEXT,
                    metadata TEXT,
                    created_at INTEGER DEFAULT (strftime('%s', 'now'))
                )
            """
            )
            conn.execute(
                """
            CREATE TABLE IF NOT EXISTS forecast_actuals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                forecast_id TEXT NOT NULL,
                ts INTEGER NOT NULL,
                price REAL NOT NULL,
                provider TEXT,
                FOREIGN KEY(forecast_id) REFERENCES forecasts(id)
            )
        """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS forecast_scores (
                forecast_id TEXT PRIMARY KEY,
                map REAL,
                rmse REAL,
                bias REAL,
                direction_match INTEGER,
                magnitude_error_pct REAL,
                error_category TEXT,
                scored_at INTEGER,
                FOREIGN KEY(forecast_id) REFERENCES forecasts(id)
            )
        """
        )
        # Indexes for spec-compliant tables
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_forecast_48h_symbol_ts
            ON forecast_48h(symbol, ts_issued DESC)
        """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_price_actuals_symbol_ts
            ON price_actuals(symbol, ts)
        """
        )

        # Add indexes for legacy tables (performance)
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_forecasts_symbol_time
            ON forecasts(symbol, as_of DESC)
        """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_forecasts_created
            ON forecasts(created_at DESC)
        """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_actuals_forecast
            ON forecast_actuals(forecast_id, ts)
        """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_scores_mape
            ON forecast_scores(map ASC, rmse ASC)
        """
        )
        conn.commit()
    except Exception as e:
        print(f"[forecast tables init] {e}")


async def _auto_score_forecasts():
    import asyncio

    while True:
        try:
            conn = _forecast_db_conn()
            if conn is None:
                await asyncio.sleep(120)
                continue
            conn.row_factory = __import__("sqlite3").Row  # type: ignore
            cur = conn.cursor()
            cur.execute("SELECT * FROM forecast_runs ORDER BY as_of_ts DESC LIMIT 100")
            rows = cur.fetchall()
            for row in rows:
                rowd = dict(row)
                str(rowd.get("symbol") or WOLF)
                as_of_ts = int(rowd.get("as_of_ts") or 0)
                # Gather actuals for this forecast
                cur2 = conn.cursor()
                cur2.execute(
                    "SELECT t, p FROM forecast_actuals WHERE forecast_id=? ORDER BY t ASC",
                    (rowd.get("id"),),
                )
                actual = [(int(r[0]), float(r[1])) for r in cur2.fetchall()]
                if not actual:
                    continue
                map, rmse, bias_pct, hit_peak = _compute_forecast_scores(rowd, actual)
                # Safely coerce forecast id
                fid_any = rowd.get("id")
                try:
                    fid = int(fid_any)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
                cur.execute(
                    """
                    INSERT INTO forecast_scores(forecast_id, scored_through_ts, map, rmse, bias, hit_peak, notes)
                    VALUES(?,?,?,?,?,?,?)
                    ON CONFLICT(forecast_id) DO UPDATE SET scored_through_ts=excluded.scored_through_ts, map=excluded.map, rmse=excluded.rmse, bias=excluded.bias, hit_peak=excluded.hit_peak, notes=excluded.notes
                    """,
                    (
                        fid,
                        actual[-1][0] if actual else as_of_ts,
                        map,
                        rmse,
                        bias_pct,
                        int(hit_peak),
                        "auto",
                    ),
                )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[auto_score_forecasts] {e}")
        await asyncio.sleep(120)


async def _auto_record_actual_prices():
    import asyncio
    import sqlite3

    while True:
        try:
            if not FORECAST_STORE:
                await asyncio.sleep(60)
                continue
            with sqlite3.connect(WOLF_SQLITE_PATH) as conn:
                for forecast_id, forecast in FORECAST_STORE.items():
                    # Get current price for the forecast symbol
                    symbol = forecast.get("symbol")
                    if not symbol:
                        continue
                    price, _, provider = get_wolf_price() if symbol == WOLF else (None, None, None)
                    if price is None:
                        continue
                    ts = int(time.time())
                    # Insert actual price for this forecast_id and timestamp
                    conn.execute(
                        """
                        INSERT INTO forecast_actuals (forecast_id, t, p, provider)
                        VALUES (?, ?, ?, ?)
                        """,
                        (forecast_id, ts, price, provider),
                    )
                conn.commit()
        except Exception as e:
            print(f"[auto_record_actual_prices] {e}")
        await asyncio.sleep(60)


async def _auto_record_forecast():
    import asyncio
    import sqlite3

    while True:
        try:
            # Example: persist all in-memory forecasts to SQLite every 60s
            if not FORECAST_STORE:
                await asyncio.sleep(60)
                continue
            with sqlite3.connect(WOLF_SQLITE_PATH) as conn:
                for forecast_id, forecast in FORECAST_STORE.items():
                    # Insert or ignore (idempotent)
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO forecast_overlay (forecast_id, symbol, as_of, hours, path_mid, path_lo, path_hi)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            forecast_id,
                            forecast.get("symbol"),
                            forecast.get("as_o"),
                            forecast.get("hours"),
                            json.dumps(forecast.get("path_mid")),
                            json.dumps(forecast.get("path_lo")),
                            json.dumps(forecast.get("path_hi")),
                        ),
                    )
                conn.commit()
        except Exception as e:
            print(f"[auto_record_forecast] {e}")
        await asyncio.sleep(60)


def _store_forecast_48h(
    symbol: str,
    price_now: float,
    price_pred_mid: float,
    price_pred_lo: float | None,
    price_pred_hi: float | None,
    pnl_pred_mid: float | None,
    confidence: float | None,
    model: str,
    features: dict[str, Any] | None = None,
) -> int:
    """
    Store a 48h forecast in the database.
    Returns the forecast ID.
    """
    import sqlite3

    try:
        with sqlite3.connect(WOLF_SQLITE_PATH) as conn:
            cur = conn.cursor()
            ts_issued = int(time.time())
            features_json = json.dumps(features) if features else None

            cur.execute(
                """
                INSERT INTO forecast_48h (
                    ts_issued, symbol, horizon_hours, price_now,
                    price_pred_mid, price_pred_lo, price_pred_hi,
                    pnl_pred_mid, confidence, model, features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    ts_issued,
                    symbol,
                    48,
                    price_now,
                    price_pred_mid,
                    price_pred_lo,
                    price_pred_hi,
                    pnl_pred_mid,
                    confidence,
                    model,
                    features_json,
                ),
            )
            forecast_id = cur.lastrowid
            conn.commit()
            return forecast_id
    except Exception as e:
        print(f"[store_forecast_48h] {e}")
        return -1


def _store_price_actual(symbol: str, price: float, ts: int | None = None):
    """Store actual price for verification — writes to BOTH SQLite and PostgreSQL.
    
    CRITICAL FIX (Mar 7, 2026): Previously wrote to SQLite only, but the
    prediction_evaluator reads from PostgreSQL price_actuals. Result: evaluator
    had ZERO price data → 0% win rate on all 155 trades.
    Now writes to both so evaluator can actually evaluate predictions.
    """
    import sqlite3

    try:
        if ts is None:
            ts = int(time.time())

        with sqlite3.connect(WOLF_SQLITE_PATH) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO price_actuals (ts, symbol, price)
                VALUES (?, ?, ?)
            """,
                (ts, symbol, price),
            )
            conn.commit()
    except Exception as e:
        print(f"[store_price_actual] SQLite: {e}")

    # ALSO write to PostgreSQL — evaluator reads from here
    try:
        from core.db_pool import get_sync_connection as _spa_get_conn
        if ts is None:
            ts = int(time.time())
        with _spa_get_conn() as pg_conn:
            pg_cur = pg_conn.cursor()
            pg_cur.execute(
                """INSERT INTO price_actuals (ts, symbol, price)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (symbol, ts) DO NOTHING""",
                (int(ts), str(symbol), float(price)),
            )
            pg_conn.commit()
    except Exception as e:
        print(f"[store_price_actual] PostgreSQL: {e}")


def _get_forecast_48h_series(symbol: str, limit: int = 50) -> list[dict[str, Any]]:
    """
    Get forecast series for a symbol.
    Returns list of forecast points with mid, lo, hi, confidence.
    """
    import sqlite3

    try:
        with sqlite3.connect(WOLF_SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            cur.execute(
                """
                SELECT
                    ts_issued as t,
                    price_now as now,
                    price_pred_mid as mid,
                    price_pred_lo as lo,
                    price_pred_hi as hi,
                    confidence as conf,
                    model
                FROM forecast_48h
                WHERE symbol = ?
                ORDER BY ts_issued DESC
                LIMIT ?
            """,
                (symbol, limit),
            )

            rows = cur.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        print(f"[get_forecast_48h_series] {e}")
        return []


def _compute_forecast_48h_metrics(symbol: str, window: int = 30) -> dict[str, Any]:
    """
    Compute accuracy metrics for 48h forecasts.

    Returns:
        - mape48h: Mean Absolute Percentage Error
        - mae48h: Mean Absolute Error
        - hit_rate_band: % of actuals that fell within prediction band
        - direction_hit: % of correct direction predictions
        - bias: "over", "under", or "neutral"
        - bias_bps: Bias in basis points
        - last_verified_at: Timestamp of last verification
    """
    import sqlite3

    try:
        with sqlite3.connect(WOLF_SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            # Get recent forecasts
            cur.execute(
                """
                SELECT
                    id, ts_issued, price_now, price_pred_mid,
                    price_pred_lo, price_pred_hi, horizon_hours
                FROM forecast_48h
                WHERE symbol = ?
                ORDER BY ts_issued DESC
                LIMIT ?
            """,
                (symbol, window),
            )

            forecasts = [dict(row) for row in cur.fetchall()]

            if not forecasts:
                return {
                    "symbol": symbol,
                    "window": window,
                    "mape48h": 0.0,
                    "mae48h": 0.0,
                    "hit_rate_band": 0.0,
                    "direction_hit": 0.0,
                    "bias": "neutral",
                    "bias_bps": 0,
                    "last_verified_at": 0,
                    "count": 0,
                }

            # Compute metrics
            errors = []
            abs_errors = []
            in_band = 0
            direction_correct = 0
            verified_count = 0
            last_verified = 0

            for fc in forecasts:
                fc["id"]
                ts_target = fc["ts_issued"] + (fc["horizon_hours"] * 3600)
                price_now = fc["price_now"]
                price_pred = fc["price_pred_mid"]
                lo = fc["price_pred_lo"]
                hi = fc["price_pred_hi"]

                # Get actual price at target time (±1 hour tolerance)
                cur.execute(
                    """
                    SELECT price FROM price_actuals
                    WHERE symbol = ? AND ts BETWEEN ? AND ?
                    ORDER BY ABS(ts - ?) ASC
                    LIMIT 1
                """,
                    (symbol, ts_target - 3600, ts_target + 3600, ts_target),
                )

                actual_row = cur.fetchone()
                if not actual_row:
                    continue

                actual_price = actual_row["price"]
                verified_count += 1
                last_verified = max(last_verified, ts_target)

                # Compute error
                error = actual_price - price_pred
                abs_error = abs(error)
                abs_error / price_pred if price_pred > 0 else 0

                errors.append(error)
                abs_errors.append(abs_error)

                # Check if in band
                if lo is not None and hi is not None:
                    if lo <= actual_price <= hi:
                        in_band += 1

                # Check direction
                pred_direction = 1 if price_pred > price_now else -1
                actual_direction = 1 if actual_price > price_now else -1
                if pred_direction == actual_direction:
                    direction_correct += 1

            if verified_count == 0:
                return {
                    "symbol": symbol,
                    "window": window,
                    "mape48h": 0.0,
                    "mae48h": 0.0,
                    "hit_rate_band": 0.0,
                    "direction_hit": 0.0,
                    "bias": "neutral",
                    "bias_bps": 0,
                    "last_verified_at": 0,
                    "count": 0,
                }

            # Calculate metrics
            mae = sum(abs_errors) / verified_count
            map = (
                mae / (sum(fc["price_pred_mid"] for fc in forecasts[:verified_count]) / verified_count)
            ) * 100

            mean_error = sum(errors) / verified_count
            bias_bps = int(
                mean_error
                / (sum(fc["price_pred_mid"] for fc in forecasts[:verified_count]) / verified_count)
                * 10000
            )

            if bias_bps > 20:
                bias = "over"
            elif bias_bps < -20:
                bias = "under"
            else:
                bias = "neutral"

            hit_rate = (in_band / verified_count) if verified_count > 0 else 0.0
            direction_rate = (direction_correct / verified_count) if verified_count > 0 else 0.0

            return {
                "symbol": symbol,
                "window": window,
                "mape48h": round(map / 100, 4),  # Convert to decimal
                "mae48h": round(mae, 2),
                "hit_rate_band": round(hit_rate, 2),
                "direction_hit": round(direction_rate, 2),
                "bias": bias,
                "bias_bps": bias_bps,
                "last_verified_at": last_verified,
                "count": verified_count,
            }

    except Exception as e:
        print(f"[compute_forecast_48h_metrics] {e}")
        return {
            "symbol": symbol,
            "window": window,
            "error": str(e),
        }


def _generate_48h_forecast(symbol: str) -> dict[str, Any]:
    """
    Generate a new 48h forecast using current price and model.
    Stores in database and returns forecast details.
    """
    try:
        # Normalize ticker symbols (handle alternate formats)
        normalized_symbol = symbol.upper()
        if normalized_symbol == "META":
            # Try META first, fallback to FB if needed
            normalized_symbol = "META"
        elif normalized_symbol == "GOOGL":
            # GOOGL is correct, but some providers use GOOG
            normalized_symbol = "GOOGL"

        # Get current price using price quorum for any symbol
        if symbol == WOLF:
            price, _, provider = get_wolf_price()
        else:
            # Use price quorum for other symbols
            try:
                is_market_open, _ = _is_market_open_now()
            except Exception:
                is_market_open = False

            providers = _build_price_providers(normalized_symbol, is_market_open=is_market_open)
            if providers:
                decision = get_price_quorum().get_price(
                    symbol=normalized_symbol,  # Use normalized symbol
                    providers=providers,
                    prev_close=None,
                    is_market_open=is_market_open,
                    timeout=120.0,  # Increased to 120s - with 30s provider timeouts and parallel execution, this allows heavily throttled providers to complete
                )
                price = decision.price
                provider = decision.provider_label

                # Log provider attempts for debugging
                if price is None:
                    LOGGER.warning(
                        f"All providers failed for {symbol}",
                        extra={
                            "symbol": symbol,
                            "normalized": normalized_symbol,
                            "provider_count": len(providers),
                            "provider_label": provider
                        }
                    )
            else:
                price = None
                provider = "unavailable"
                LOGGER.warning(f"No providers available for {symbol}")

        if price is None or price <= 0:
            provider_label = provider if 'provider' in locals() else "unknown"
            error_msg = f"live price unavailable (provider: {provider_label})"
            LOGGER.error(
                f"Forecast failed for {symbol}: {error_msg}",
                extra={"symbol": symbol, "price": price, "provider": provider_label}
            )
            return {
                "ok": False,
                "error": error_msg,
                "symbol": symbol,
                "provider": provider_label,
            }

        # Get portfolio for PnL prediction
        qty, avg_cost = _get_portfolio_qty_and_avg()  # Use helper to get portfolio data

        # Simple volatility-based forecast model
        # In production, you'd use GPT-4o or ensemble model
        sigma_daily = float(PRED_SIGMA_DAILY)
        vol_48h = sigma_daily * math.sqrt(2)  # 2-day volatility

        price_pred_mid = price * (1.0 + (vol_48h * 0.1))  # Slight upward bias
        price_pred_lo = price * (1.0 - vol_48h)
        price_pred_hi = price * (1.0 + vol_48h)

        # PnL prediction
        if qty > 0:
            qty * price
            pred_value = qty * price_pred_mid
            pnl_pred_mid = pred_value - (qty * avg_cost)
        else:
            pnl_pred_mid = None

        # Confidence based on data availability, nudged by research aggregate if available
        confidence = 0.75 if provider in ["polygon", "alphavantage"] else 0.50
        research_features: dict[str, Any] = {}
        try:
            # Include recent news sentiment score
            ns = (get_wolf_news(limit=3).get("news_signal") or {}).get("score")
            research_features["news_score"] = ns
        except Exception as e:
            LOGGER.warning(f"research_news_score_fetch_failed for {symbol}: {e}")
        try:
            f = _get_filings_signal(symbol)
            if f:
                research_features["filings"] = f
        except Exception as e:
            LOGGER.warning(f"filings_signal_fetch_failed for {symbol}: {e}")
        try:
            if RESEARCH_BLUEPRINT_ON:
                snap = build_research_snapshot(symbol, asset_type="stock")
                agg = snap.get("aggregate") or {}
                research_features["research_aggregate"] = agg
                # Nudge confidence towards aggregate confidence (blend 80/20 if numeric)
                rc = agg.get("confidence") if isinstance(agg, dict) else None
                if isinstance(rc, (int, float)):
                    confidence = max(0.3, min(0.85, 0.8 * confidence + 0.2 * (float(rc) / 100.0)))  # HARD CAP 85%
        except Exception as e:
            LOGGER.warning(f"research_snapshot_failed for {symbol}: {e}")

        # Store forecast
        model = "simple-vol"  # Change to "gpt-4o" when integrated
        forecast_id = _store_forecast_48h(
            symbol=symbol,
            price_now=price,
            price_pred_mid=price_pred_mid,
            price_pred_lo=price_pred_lo,
            price_pred_hi=price_pred_hi,
            pnl_pred_mid=pnl_pred_mid,
            confidence=confidence,
            model=model,
            features={
                "provider": provider,
                "vol_daily": sigma_daily,
                "vol_48h": vol_48h,
                "research": research_features,
            },
        )

        return {
            "ok": True,
            "forecast_id": forecast_id,
            "symbol": symbol,
            "ts_issued": int(time.time()),
            "price_now": round(price, 2),
            "price_pred_mid": round(price_pred_mid, 2),
            "price_pred_lo": round(price_pred_lo, 2),
            "price_pred_hi": round(price_pred_hi, 2),
            "pnl_pred_mid": round(pnl_pred_mid, 2) if pnl_pred_mid else None,
            "confidence": confidence,
            "model": model,
        }

    except Exception as e:
        error_str = str(e) if str(e) else f"{type(e).__name__}: (empty message)"
        LOGGER.exception(f"Forecast exception for {symbol}: {error_str}")
        return {
            "ok": False,
            "error": error_str,
            "symbol": symbol,
            "exception_type": type(e).__name__,
        }


async def _auto_generate_forecasts():
    """Generate forecast every 60 minutes."""
    import asyncio

    await asyncio.sleep(10)  # Initial delay

    while True:
        try:
            # Generate forecast for WOLF
            result = _generate_48h_forecast(WOLF)
            if result.get("ok"):
                print(f"[48h forecast] Generated: {result['forecast_id']} at {result['ts_issued']}")
            else:
                print(f"[48h forecast] Failed: {result.get('error')}")

            # Store current price as actual
            price, _, _ = get_wolf_price()
            if price and price > 0:
                _store_price_actual(WOLF, price)

        except Exception as e:
            print(f"[auto_generate_forecasts] {e}")

        await asyncio.sleep(3600)  # 60 minutes


async def _post_startup_init():
    """
    Run Stage 4/5 and background tasks AFTER server starts accepting connections.
    This prevents blocking the startup event handler.
    
    ARCHITECTURE SPLIT: Heavy background services run only in WORKER_MODE.
    Core prediction loop runs in ALL modes for continuous predictions.
    """
    # NOTE: Uses global 'os' imported at line 12 - no local import needed
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # CRITICAL: Auto-Prediction Loop runs in ALL modes (web + worker)
    # ═══════════════════════════════════════════════════════════════════════════════
    # Start Auto-Prediction Loop BEFORE checking WORKER_MODE
    # This ensures predictions generate even in web-only deployments
    try:
        from core import auto_prediction_loop
        
        # Inject dependencies (both sync and async versions)
        auto_prediction_loop.LOGGER = LOGGER
        auto_prediction_loop.RUN_PREDICTION_FUNC = run_prediction
        auto_prediction_loop.RUN_PREDICTION_FUNC_ASYNC = run_single_prediction_async
        auto_prediction_loop.HUNTER_STOCK_SYMBOLS = HUNTER_STOCK_SYMBOLS
        auto_prediction_loop.HUNTER_CRYPTO_SYMBOLS = HUNTER_CRYPTO_SYMBOLS
        
        # Start the loop
        auto_prediction_loop.start_auto_prediction_loop()
        
        LOGGER.info("✅ Auto-Prediction Loop: STARTED (ASYNC, non-blocking, 60-min interval)")
    except Exception as e:
        LOGGER.exception("auto_prediction_loop_start_failed", extra={"component": "startup", "error": str(e)})

    # ═══════════════════════════════════════════════════════════════════════════════
    # Telegram Signal Dispatcher (runs in ALL modes)
    # Enforces >=70% (via core.telegram_alerts) and sends MULTIPLE signals per cycle.
    # ═══════════════════════════════════════════════════════════════════════════════
    try:
        from core import telegram_alerts as _telegram_alerts

        dispatch_enabled = os.getenv("GHOST_SIGNAL_DISPATCH_ENABLED", "1").strip() not in ("0", "false", "False")
        dispatch_interval_s = int(os.getenv("GHOST_SIGNAL_DISPATCH_INTERVAL_S", "3600"))
        max_per_cycle = int(os.getenv("GHOST_SIGNAL_MAX_PER_CYCLE", "5"))

        # Avoid repeats within the process (Redis dedup still provides daily protection)
        _last_sent_pred_id: dict[str, int] = {}

        def _horizon_bucket(h: int) -> str:
            try:
                h = int(h)
            except Exception:
                h = 48
            return f"{h}h"

        async def _signal_dispatch_loop():
            if not dispatch_enabled:
                LOGGER.info("[SIGNALS] Telegram signal dispatch disabled (GHOST_SIGNAL_DISPATCH_ENABLED=0)")
                return

            # Only dispatch if Telegram is configured.
            if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
                LOGGER.warning("[SIGNALS] Telegram not configured; signal dispatch paused")
                return

            # Initial delay so prediction loop can populate cache.
            await asyncio.sleep(20)

            while True:
                try:
                    candidates: list[tuple[float, str, dict[str, Any]]] = []
                    for sym, pred in list(_LATEST_PREDICTIONS.items()):
                        if not isinstance(pred, dict):
                            continue
                        symbol = (sym or pred.get("symbol") or "").upper().strip()
                        if not symbol:
                            continue

                        direction = str(pred.get("direction") or "").upper()
                        action = str(pred.get("action") or "").upper()
                        if direction in ("", "ERROR"):
                            continue
                        if direction == "FLAT" or action in ("HOLD", "WATCH"):
                            continue

                        if pred.get("should_predict") is False:
                            continue

                        # Ensure we only send once per symbol per prediction_id in-process.
                        try:
                            pid = int(pred.get("prediction_id") or pred.get("id") or 0)
                        except Exception:
                            pid = 0
                        if pid and _last_sent_pred_id.get(symbol) == pid:
                            continue

                        # Rank: prefer calibrated execution prob, then analysis, then raw confidence
                        rank = 0.0
                        for k in ("touch_calibrated_0_5pct", "touch_calibrated_1pct", "confidence"):
                            try:
                                v = pred.get(k)
                                if v is None:
                                    continue
                                rank = float(v)
                                break
                            except Exception:
                                continue

                        candidates.append((rank, symbol, pred))

                    # Highest confidence/calibration first
                    candidates.sort(key=lambda t: t[0], reverse=True)

                    sent = 0
                    # ================================================================
                    # OLD INDIVIDUAL ALERT SYSTEM - DISABLED
                    # This was sending individual alerts for each prediction.
                    # Now using ghost_notifications.py for ONE consolidated message.
                    # ================================================================
                    # for _, symbol, pred in candidates:
                    #     if sent >= max_per_cycle:
                    #         break
                    # 
                    #     ... OLD CODE DISABLED ...
                    # 
                    # Instead, log that we're using the new system:
                    LOGGER.info(f"[SIGNALS] Individual alerts DISABLED - Using consolidated TOP 10 system instead")
                    LOGGER.info(f"[SIGNALS] {len(candidates)} predictions available for next TOP 10")

                except Exception as dispatch_err:
                    LOGGER.error(f"[SIGNALS] Dispatch loop error: {dispatch_err}", exc_info=False)

                await asyncio.sleep(max(60, dispatch_interval_s))

        loop = asyncio.get_running_loop()
        loop.create_task(_signal_dispatch_loop())
        LOGGER.info("[GHOST STARTUP] ✅ Telegram signal dispatcher scheduled (individual alerts DISABLED)")
    except Exception as e:
        LOGGER.error(f"signal_dispatcher_start_failed: {e}", extra={"component": "startup"}, exc_info=False)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # CRITICAL: News Brain Loop runs in ALL modes (not just worker mode)
    # This ensures news analysis happens even in web-only deployments
    # ═══════════════════════════════════════════════════════════════════════════════
    try:
        LOGGER.info("🔍 News Brain: ENTRY POINT REACHED")
        LOGGER.info("🔍 News Brain: Checking NEWS_ANALYSIS_ENABLED...")
        NEWS_ANALYSIS_ENABLED = os.getenv("NEWS_ANALYSIS_ENABLED", "1") == "1"
        LOGGER.info(f"🔍 News Brain: NEWS_ANALYSIS_ENABLED = {NEWS_ANALYSIS_ENABLED}")
        
        if not NEWS_ANALYSIS_ENABLED:
            LOGGER.info("ℹ️  Automatic News Analysis: DISABLED (set NEWS_ANALYSIS_ENABLED=1 to enable)")
        else:
            LOGGER.info("🔍 News Brain: Importing get_news_brain...")
            from core.intelligence.ghost_news_brain import get_news_brain
            LOGGER.info("✅ News Brain: Import successful")
            
            NEWS_ANALYSIS_INTERVAL_MINUTES = int(os.getenv("NEWS_ANALYSIS_INTERVAL_MINUTES", "5"))  # FIX (Step 8): was 30 → 5 min to match prediction cycle
            LOGGER.info(f"🔍 News Brain: Interval set to {NEWS_ANALYSIS_INTERVAL_MINUTES} minutes")
            
            async def _news_analysis_loop():
                """Automatic news analysis every 30 minutes"""
                LOGGER.info(f"📰 News Analysis Loop: STARTING (every {NEWS_ANALYSIS_INTERVAL_MINUTES} min)")
                
                cycle_count = 0
                while True:
                    try:
                        cycle_count += 1
                        _heartbeat_pulse("news-analysis")
                        LOGGER.info(f"📰 [CYCLE {cycle_count}] Running automatic news analysis...")
                        brain = get_news_brain()
                        result = await brain.analyze_news()
                        
                        major_events = result.get("major_events", [])
                        predictions_at_risk = result.get("predictions_at_risk", [])
                        
                        LOGGER.info(
                            f"📰 [CYCLE {cycle_count}] News analysis complete: {len(major_events)} events, "
                            f"{len(predictions_at_risk)} predictions at risk"
                        )

                        # ═══ WIRED: Feed results into Intelligence Hub ═══
                        # This bridges news brain → scout predictions pipeline.
                        # The hub reads this cache when making predictions.
                        try:
                            from core.intelligence_hub import update_news_brain_cache
                            update_news_brain_cache(result)
                            LOGGER.info(f"📰 [CYCLE {cycle_count}] ✅ Intelligence Hub cache updated")
                        except Exception as hub_err:
                            LOGGER.warning(f"📰 [CYCLE {cycle_count}] Hub cache update failed: {hub_err}")

                        # ═══ WIRED: Auto-pause on CRITICAL events ═══
                        # Ghost acts on critical news internally:
                        #   - Creates guardian alerts for affected symbols
                        #   - Auto-pauses trading for safety
                        #   - Feeds Intelligence Hub (done above)
                        # NO Telegram spam — Ghost uses the data, user doesn't need raw news dumps.
                        for event in major_events:
                            if event.get("severity") == "CRITICAL":
                                LOGGER.warning(f"🚨 CRITICAL EVENT: {event.get('headline', 'Unknown')}")
                                try:
                                    actions = await brain.handle_critical_event(event, auto_pause=True)
                                    if actions.get("trading_paused"):
                                        LOGGER.warning("🛑 Trading AUTO-PAUSED due to critical news event")
                                except Exception as crit_err:
                                    LOGGER.error(f"Critical event handling failed: {crit_err}")

                        # ═══ NEWS → INTELLIGENCE HUB (internal only) ═══
                        # No Telegram alerts for news — Ghost consumes news through the Hub.
                        # The hub cache was already updated above. Predictions at risk
                        # are handled by the hub's confidence/direction adjustments.
                        high_risk = [p for p in predictions_at_risk
                                     if p.get("risk_level") in ("HIGH", "CRITICAL")]
                        if high_risk:
                            LOGGER.info(
                                f"📰 [CYCLE {cycle_count}] {len(high_risk)} predictions at risk — "
                                f"Hub will adjust: {[p.get('symbol') for p in high_risk]}"
                            )
                        
                    except Exception as e:
                        LOGGER.error(f"News analysis error: {e}", exc_info=True)
                    
                    await asyncio.sleep(NEWS_ANALYSIS_INTERVAL_MINUTES * 60)
            
            asyncio.create_task(_news_analysis_loop())
            LOGGER.info(f"✅ Automatic News Analysis: STARTED (every {NEWS_ANALYSIS_INTERVAL_MINUTES} min)")
    except Exception as e:
        LOGGER.error(f"🚨 News Brain FAILED TO START: {e}", extra={"component": "startup"}, exc_info=True)

    # ═══════════════════════════════════════════════════════════════════════════════
    # SELF-IMPROVEMENT ENGINE — Auto-tune thresholds every 6 hours
    # ═══════════════════════════════════════════════════════════════════════════════
    try:
        SELF_IMPROVE_ENABLED = os.getenv("SELF_IMPROVE_ENABLED", "1") == "1"
        if SELF_IMPROVE_ENABLED:
            async def _self_improvement_loop():
                """Run self-improvement cycle every 6 hours."""
                LOGGER.info("🔧 Self-Improvement Engine: STARTING (every 6 hours)")
                _heartbeat_pulse("self-improvement")  # Pulse immediately so Health tab shows alive
                await asyncio.sleep(300)  # Wait 5 min after startup
                while True:
                    try:
                        _heartbeat_pulse("self-improvement")
                        from core.self_improvement_engine import run_improvement_cycle
                        result = run_improvement_cycle()
                        LOGGER.info(f"🔧 Self-improvement cycle complete: {result.get('improvements_made', 0)} improvements")
                    except Exception as e:
                        LOGGER.error(f"Self-improvement error: {e}")
                    await asyncio.sleep(6 * 3600)  # Every 6 hours

            asyncio.create_task(_self_improvement_loop())
            LOGGER.info("✅ Self-Improvement Engine: SCHEDULED (every 6 hours)")
    except Exception as e:
        LOGGER.error(f"Self-Improvement Engine failed to start: {e}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # INTELLIGENCE HUB — Pre-initialize on startup
    # ═══════════════════════════════════════════════════════════════════════════════
    try:
        from core.intelligence_hub import get_intelligence_hub
        hub = get_intelligence_hub()
        hub._lazy_init()
        hub_status = hub.get_status()
        loaded = sum(1 for v in hub_status.values() if v is True)
        LOGGER.info(f"🧠 Intelligence Hub: {loaded} systems loaded at startup")
    except Exception as e:
        LOGGER.error(f"Intelligence Hub startup failed: {e}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # GHOST NOTIFICATION SYSTEM - Runs in WEB mode ONLY (not worker)
    # FIX (Feb 12, 2026): Was running in BOTH web + worker processes, causing
    # duplicate Telegram cards with slightly different confidences because each
    # process has its own in-memory _last_top10_date guard and runs independent
    # edge scans seconds apart (prices shift → confidences differ).
    # Now gated to web-only. Worker process skips this entirely.
    # ═══════════════════════════════════════════════════════════════════════════════
    _IS_WORKER = os.getenv("WORKER_MODE") == "1"
    if _IS_WORKER:
        LOGGER.info("[WORKER MODE] ⏭️ Skipping notification loop (web process handles it)")
    
    try:
        from core.ghost_notifications import get_notification_system, get_central_time
        
        active_tracking_enabled = os.getenv("ACTIVE_TRACKING_ENABLED", "1") == "1" and not _IS_WORKER
        LOGGER.info(f"[NOTIFICATION DEBUG] ACTIVE_TRACKING_ENABLED = {active_tracking_enabled}, WORKER_MODE = {_IS_WORKER}")
        print(f"[NOTIFICATION DEBUG] ACTIVE_TRACKING_ENABLED = {active_tracking_enabled}, WORKER_MODE = {_IS_WORKER}")
        
        if active_tracking_enabled:
            def _send_telegram(message: str) -> bool:
                if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
                    return False
                return _tg_send_chat_message(TELEGRAM_CHAT_ID, message)
            
            notification_system = get_notification_system()
            notification_system.set_telegram_func(_send_telegram)
            
            async def _get_current_price(symbol: str) -> float:
                """Get current price for a symbol"""
                try:
                    from core.asset_classifier import get_asset_type
                    asset_class = get_asset_type(symbol)
                    if asset_class.startswith("crypto"):
                        result = turbo_crypto_price(symbol, max_budget_s=2.0)
                    else:
                        result = turbo_stock_price(symbol, max_budget_s=2.0)
                    if result and result.get("ok") and result.get("price"):
                        return float(result["price"])
                except Exception as e:
                    LOGGER.debug(f"price_fetch_failed for {symbol}: {e}")
                return 0.0
            
            async def _ghost_notification_loop():
                """
                SIMPLE notification loop - ONE message at 8 AM Central.
                
                Schedule:
                - 8:00-8:59 AM Central: Send ONE TOP 10 message (5 stocks + 5 crypto)
                - Every 4 hours: Check for updates (12 PM, 4 PM, 8 PM)
                - Every 15 min: Check for target/stop hits
                
                IMPORTANT: The scheduler runs in a background task.
                If Railway restarts the container, this task restarts too.
                """
                global _LAST_TELEGRAM_STATUS, _LAST_TELEGRAM_SEND_TIME, _LAST_TELEGRAM_ERROR
                try:
                    from zoneinfo import ZoneInfo
                    central_tz = ZoneInfo("America/Chicago")
                except ImportError:
                    import pytz
                    central_tz = pytz.timezone("America/Chicago")
                
                TOP_10_HOUR = 8  # 8 AM Central — daily TOP 10 send
                PRE_SCAN_HOUR = 7  # 7 AM Central — run predictions so 8 AM send is instant
                UPDATE_HOURS = [12, 16, 20]  # 12 PM, 4 PM, 8 PM Central
                
                now_central = datetime.now(central_tz)
                LOGGER.info(f"[NOTIFICATIONS] 🎯 Starting notification loop (PRE-SCAN at {PRE_SCAN_HOUR}:00, TOP 10 at {TOP_10_HOUR}:00 Central)")
                LOGGER.info(f"[NOTIFICATIONS] Current time: {now_central.strftime('%Y-%m-%d %H:%M:%S')} Central")
                
                last_top10_date = None
                last_prescan_date = None  # Track pre-scan so it runs once per day
                last_check_time = 0
                loop_count = 0
                
                _NOTIFICATION_LOOP_STATUS["running"] = True
                _NOTIFICATION_LOOP_STATUS["started_at"] = datetime.now(central_tz).isoformat()
                print("=" * 60)
                print("[NOTIFICATION LOOP] 🚀 LOOP STARTED SUCCESSFULLY")
                print(f"[NOTIFICATION LOOP] Current time: {datetime.now(central_tz).strftime('%Y-%m-%d %H:%M:%S')} Central")
                print(f"[NOTIFICATION LOOP] Schedule: PRE-SCAN at 7 AM, TOP 10 at 8 AM, Watchdog every 15 min")
                print("=" * 60)
                LOGGER.info("[NOTIFICATION LOOP] 🚀 Status set to RUNNING")
                
                await asyncio.sleep(10)
                print("[NOTIFICATION LOOP] ✅ Initial delay complete - entering main loop")
                
                while True:
                    try:
                        loop_count += 1
                        _heartbeat_pulse("notification-loop")
                        now_central = datetime.now(central_tz)
                        current_hour = now_central.hour
                        current_date = now_central.strftime("%Y-%m-%d")
                        current_time = time.time()
                        
                        _NOTIFICATION_LOOP_STATUS["loop_count"] = loop_count
                        _NOTIFICATION_LOOP_STATUS["current_central_time"] = now_central.strftime("%Y-%m-%d %H:%M:%S")
                        _NOTIFICATION_LOOP_STATUS["last_top10_date"] = last_top10_date
                        _NOTIFICATION_LOOP_STATUS["predictions_count"] = len(_LATEST_PREDICTIONS)
                        
                        if loop_count <= 5 or loop_count % 10 == 0:
                            LOGGER.info(f"[NOTIFICATIONS] ⏰ Loop tick #{loop_count}: {now_central.strftime('%H:%M')} Central, predictions={len(_LATEST_PREDICTIONS)}")
                        
                        # ── DEAD-MAN'S SWITCH (Feb 24, 2026) ──
                        # If Railway restarted AFTER 8 AM and today's TOP 10 was never
                        # sent, catch up now. Only fires once (sets last_top10_date).
                        # Window: 8 AM - 9 PM CT (don't send at midnight).
                        if (
                            current_hour > TOP_10_HOUR
                            and current_hour < 21
                            and last_top10_date != current_date
                            and not notification_system._last_top10_date == current_date
                            and len(_LATEST_PREDICTIONS) == 0
                            and last_prescan_date != current_date
                        ):
                            # No predictions cached and pre-scan didn't run — do emergency scan
                            LOGGER.warning(f"[DEAD-MAN] ⚠️ Missed 8 AM window — running emergency scan ({now_central.strftime('%H:%M')} CT)")
                            try:
                                from core.asset_classifier import get_asset_type
                                _dm_symbols = list(get_edge_set())
                                for sym in _dm_symbols:
                                    try:
                                        run_single_prediction(sym)
                                    except Exception as e:
                                        LOGGER.warning(f"[DEAD-MAN] Emergency prediction failed for {sym}: {e}")
                                last_prescan_date = current_date
                                LOGGER.info(f"[DEAD-MAN] Emergency scan done, {len(_LATEST_PREDICTIONS)} predictions cached")
                            except Exception as e:
                                LOGGER.error(f"[DEAD-MAN] Emergency scan failed: {e}")
                        
                        if (
                            current_hour > TOP_10_HOUR
                            and current_hour < 21
                            and last_top10_date != current_date
                            and not notification_system._last_top10_date == current_date
                            and len(_LATEST_PREDICTIONS) > 0
                        ):
                            LOGGER.warning(f"[DEAD-MAN] ⚠️ Catching up missed TOP 10 ({now_central.strftime('%H:%M')} CT, {len(_LATEST_PREDICTIONS)} predictions)")
                            try:
                                success = notification_system.send_top10(_LATEST_PREDICTIONS)
                                if success:
                                    last_top10_date = current_date
                                    _NOTIFICATION_LOOP_STATUS["last_top10_date"] = current_date
                                    LOGGER.info(f"[DEAD-MAN] ✅ Catch-up TOP 10 sent at {now_central.strftime('%H:%M')} CT")
                                else:
                                    # FIX (Feb 25, 2026): If send_top10 returns False, check if
                                    # the notification system already marked today as sent
                                    # (e.g., "no picks" message). Prevents infinite retry loop.
                                    if notification_system._last_top10_date == current_date:
                                        last_top10_date = current_date
                                        _NOTIFICATION_LOOP_STATUS["last_top10_date"] = current_date
                                        LOGGER.info(f"[DEAD-MAN] Day already handled by notification system")
                            except Exception as e:
                                LOGGER.error(f"[DEAD-MAN] Catch-up send failed: {e}")
                        
                        # ── 7 AM PRE-SCAN: Run predictions so 8 AM send is instant ──
                        if current_hour == PRE_SCAN_HOUR and last_prescan_date != current_date:
                            LOGGER.info(f"[PRE-SCAN] 🔄 7 AM - Running prediction scan for all edge symbols ({now_central.strftime('%H:%M:%S')} Central)...")
                            try:
                                _ps_stock_count = 0
                                _ps_crypto_count = 0
                                from core.asset_classifier import get_asset_type
                                
                                _PS_EDGE_ENABLED = os.getenv("EDGE_WHITELIST_ENABLED", "1") == "1"
                                _PS_EDGE_SET = get_edge_set()
                                
                                if _PS_EDGE_ENABLED:
                                    prescan_symbols = list(_PS_EDGE_SET)
                                else:
                                    prescan_symbols = HUNTER_STOCK_SYMBOLS[:50] + HUNTER_CRYPTO_SYMBOLS[:25]
                                
                                LOGGER.info(f"[PRE-SCAN] Scanning {len(prescan_symbols)} symbols...")
                                for symbol in prescan_symbols:
                                    try:
                                        result = run_single_prediction(symbol)
                                        if result.get("ok"):
                                            asset_type = get_asset_type(symbol)
                                            if asset_type.startswith("crypto"):
                                                _ps_crypto_count += 1
                                            else:
                                                _ps_stock_count += 1
                                    except Exception as e:
                                        LOGGER.debug(f"[PRE-SCAN] Prediction failed for {symbol}: {e}")
                                
                                last_prescan_date = current_date
                                LOGGER.info(f"[PRE-SCAN] ✅ Complete: {_ps_stock_count} stocks + {_ps_crypto_count} crypto ready for 8 AM send")
                            except Exception as e:
                                LOGGER.error(f"[PRE-SCAN] Error: {e}", exc_info=True)
                        
                        # ── 8 AM SEND: Instant send from pre-scanned predictions ──
                        if current_hour == TOP_10_HOUR and last_top10_date != current_date:
                            LOGGER.info(f"[NOTIFICATIONS] 🌅 8 AM WINDOW - Sending morning TOP 10 IMMEDIATELY ({now_central.strftime('%H:%M:%S')} Central)...")
                            
                            # If pre-scan didn't run (e.g. Railway restarted after 7 AM),
                            # do a quick scan now before sending
                            if last_prescan_date != current_date:
                                LOGGER.warning(f"[NOTIFICATIONS] ⚠️ Pre-scan missed — running quick scan before send")
                                try:
                                    stock_count = 0
                                    crypto_count = 0
                                    from core.asset_classifier import get_asset_type
                                    
                                    _TOP10_EDGE_ENABLED = os.getenv("EDGE_WHITELIST_ENABLED", "1") == "1"
                                    _TOP10_EDGE_SET = get_edge_set()
                                    
                                    if _TOP10_EDGE_ENABLED:
                                        scan_symbols = list(_TOP10_EDGE_SET)
                                    else:
                                        scan_symbols = HUNTER_STOCK_SYMBOLS[:50] + HUNTER_CRYPTO_SYMBOLS[:25]
                                    
                                    for symbol in scan_symbols:
                                        try:
                                            result = run_single_prediction(symbol)
                                            if result.get("ok"):
                                                asset_type = get_asset_type(symbol)
                                                if asset_type.startswith("crypto"):
                                                    crypto_count += 1
                                                else:
                                                    stock_count += 1
                                        except Exception as e:
                                            LOGGER.debug(f"[TOP10-PREP] Fallback scan failed for {symbol}: {e}")
                                    
                                    LOGGER.info(f"[TOP10-PREP] Fallback scan: {stock_count} stocks + {crypto_count} crypto")
                                    last_prescan_date = current_date
                                except Exception as e:
                                    LOGGER.warning(f"[TOP10-PREP] Fallback scan error: {e}")
                            
                            LOGGER.info(f"[NOTIFICATIONS] Predictions available: {len(_LATEST_PREDICTIONS)} symbols")
                            
                            success = notification_system.send_top10(_LATEST_PREDICTIONS)
                            
                            if success:
                                last_top10_date = current_date
                                _NOTIFICATION_LOOP_STATUS["last_top10_date"] = current_date
                                _NOTIFICATION_LOOP_STATUS["last_top10_send_time"] = now_central.isoformat()
                                _NOTIFICATION_LOOP_STATUS["last_top10_success"] = True
                                _LAST_TELEGRAM_SEND_TIME = time.time()
                                _LAST_TELEGRAM_STATUS = "ok"
                                _LAST_TELEGRAM_ERROR = None
                                LOGGER.info(f"[NOTIFICATIONS] ✅ TOP 10 sent successfully at {now_central.strftime('%H:%M:%S')} Central!")
                            else:
                                _NOTIFICATION_LOOP_STATUS["last_top10_success"] = False
                                _LAST_TELEGRAM_STATUS = "error"
                                _LAST_TELEGRAM_ERROR = "TOP 10 send failed or no predictions"
                                LOGGER.warning(f"[NOTIFICATIONS] ⚠️ TOP 10 send failed or no predictions (count={len(_LATEST_PREDICTIONS)})")
                                # FIX (Feb 25, 2026): If notification system handled the day
                                # (e.g., sent "no picks" message), sync our local guard.
                                if notification_system._last_top10_date == current_date:
                                    last_top10_date = current_date
                                    _NOTIFICATION_LOOP_STATUS["last_top10_date"] = current_date
                        
                        if current_time - last_check_time >= 900:
                            # FIX (Feb 13, 2026): TTL eviction for _LATEST_PREDICTIONS
                            # Stale predictions stayed in memory forever (no expiry).
                            # Evict entries older than 24h to prevent stale data from
                            # polluting TOP 10 and cockpit displays.
                            _TTL_SECONDS = 24 * 3600  # 24 hours
                            with _LATEST_PREDICTIONS_LOCK:
                                stale_symbols = [
                                    sym for sym, pred in _LATEST_PREDICTIONS.items()
                                    if isinstance(pred, dict) and (current_time - pred.get('run_at', current_time)) > _TTL_SECONDS
                                ]
                                for sym in stale_symbols:
                                    del _LATEST_PREDICTIONS[sym]
                                if stale_symbols:
                                    LOGGER.info(f"[TTL-EVICT] 🧹 Evicted {len(stale_symbols)} stale predictions (>24h): {stale_symbols}")
                            
                            def get_price(symbol: str) -> float:
                                try:
                                    from core.asset_classifier import get_asset_type
                                    if get_asset_type(symbol).startswith("crypto"):
                                        r = turbo_crypto_price(symbol, max_budget_s=2.0)
                                    else:
                                        r = turbo_stock_price(symbol, max_budget_s=2.0)
                                    return float(r.get("price", 0)) if r and r.get("ok") else 0
                                except Exception:
                                    return 0
                            
                            notification_system.check_for_updates(get_price)
                            last_check_time = current_time
                            _NOTIFICATION_LOOP_STATUS["last_check_time"] = datetime.now(central_tz).isoformat()
                        
                        await asyncio.sleep(60)
                        
                    except asyncio.CancelledError:
                        _NOTIFICATION_LOOP_STATUS["running"] = False
                        LOGGER.info("[NOTIFICATIONS] Loop cancelled - shutting down")
                        break
                    except Exception as e:
                        LOGGER.error(f"[NOTIFICATIONS] Loop error: {e}", exc_info=True)
                        await asyncio.sleep(60)
            
            task = asyncio.create_task(_ghost_notification_loop())
            LOGGER.info("🎯 [POST-STARTUP] Ghost Notification System ACTIVE (8 AM TOP 10 + watchdog)")
            LOGGER.info("🎯 [POST-STARTUP] Backup endpoints: /alerts/top10/now, /alerts/watchdog/check")
        else:
            LOGGER.info("🎯 [POST-STARTUP] Ghost Notification System DISABLED (set ACTIVE_TRACKING_ENABLED=1)")
    except Exception as e:
        LOGGER.error(f"ghost_notification_system_init_failed: {e}", extra={"component": "startup"}, exc_info=True)
    
    # ── Wire System Doctor (7 AM daily health check) ────────────────
    # NOTE: Runs in WEB mode (not worker-only). The doctor is a lightweight
    # daily check that sends one Telegram message. Must be ABOVE the
    # WORKER_MODE return gate so it fires even without a dedicated worker.
    try:
        from core import system_doctor as _doc
        _doc.GET_PREDICTIONS_FUNC = lambda: dict(_LATEST_PREDICTIONS)
        _doc.GET_EDGE_SET_FUNC = get_edge_set

        # Price func: reuse beast_fetch_price if orchestrator is on, otherwise build one
        def _doctor_price_func(symbol, market):
            try:
                if market == "crypto":
                    from core.crypto.crypto_providers import get_crypto_price_quorum
                    result = get_crypto_price_quorum(symbol)
                    if result and result.get("price"):
                        return (result["price"], result["price"], result.get("provider", "unknown"), False)
                else:
                    from core.providers.turbo_provider import turbo_stock_price
                    price_data = turbo_stock_price(symbol)
                    if price_data and price_data.get("price"):
                        return (price_data["price"], price_data.get("prev_close"), price_data.get("provider", "unknown"), False)
                return None
            except Exception:
                return None

        _doc.GET_PRICE_FUNC = _doctor_price_func
        _doc.TELEGRAM_SEND_FUNC = lambda msg: _send_telegram_internal(msg)[0]

        # ── Start independent 7 AM CT doctor cron (no orchestrator needed) ──
        async def _doctor_cron_loop():
            """Run System Doctor + Telegram at 7:00 AM CT every day.
            
            Resilient to Railway container restarts:
            - If restart happens at 7:00-9:00 AM CT, fires immediately (grace window)
            - Tracks last-fire date to prevent double-sends
            - Logs next target time for debugging
            """
            from zoneinfo import ZoneInfo
            _CT = ZoneInfo("America/Chicago")
            _DOCTOR_HOUR = 7
            _GRACE_MINUTES = 120  # Fire if within 2h after target (handles restarts up to 9 AM)
            await asyncio.sleep(30)  # let startup finish
            LOGGER.info("[DOCTOR-CRON] 🩺 7 AM CT doctor cron started")
            
            # Track last fire date to prevent double-sends on restart
            _last_fire_date = None
            
            while True:
                try:
                    _heartbeat_pulse("doctor-cron")
                    now_ct = datetime.now(_CT)
                    today_target = now_ct.replace(hour=_DOCTOR_HOUR, minute=0, second=0, microsecond=0)
                    today_date = now_ct.date()
                    
                    # Check if we're in the grace window (7:00-9:00 AM CT today)
                    minutes_past = (now_ct - today_target).total_seconds() / 60
                    in_grace = 0 <= minutes_past <= _GRACE_MINUTES
                    already_fired_today = (_last_fire_date == today_date)
                    
                    if in_grace and not already_fired_today:
                        # We're in the window — fire NOW (handles restart at 7:01, 8:30, etc.)
                        LOGGER.info(f"[DOCTOR-CRON] 🩺 In grace window ({minutes_past:.0f}m past 7 AM) — firing immediately")
                        wait_secs = 0
                    elif now_ct >= today_target:
                        # Past the window today — wait for tomorrow
                        target = today_target + timedelta(days=1)
                        wait_secs = (target - now_ct).total_seconds()
                        LOGGER.info(f"[DOCTOR-CRON] Next check in {wait_secs/3600:.1f}h at {target.isoformat()}")
                    else:
                        # Before 7 AM today — wait for today's target
                        wait_secs = (today_target - now_ct).total_seconds()
                        LOGGER.info(f"[DOCTOR-CRON] Next check in {wait_secs/3600:.1f}h at {today_target.isoformat()}")
                    
                    if wait_secs > 0:
                        # Sleep in 30-min chunks, pulsing heartbeat each time (Step 4B fix)
                        # Without this, doctor-cron shows DEAD during the 24h sleep
                        _remaining = wait_secs
                        while _remaining > 0:
                            _chunk = min(_remaining, 1800)  # 30 min max
                            await asyncio.sleep(_chunk)
                            _heartbeat_pulse("doctor-cron")
                            _remaining -= _chunk
                    
                    # Re-check date in case we slept a long time
                    fire_date = datetime.now(_CT).date()
                    if _last_fire_date == fire_date:
                        LOGGER.info(f"[DOCTOR-CRON] Already fired today ({fire_date}), skipping")
                        await asyncio.sleep(3600)  # check again in 1 hour
                        continue
                    
                    # Fire! With retry logic for Telegram delivery
                    LOGGER.info("[DOCTOR-CRON] 🩺 Running 7 AM System Doctor...")
                    from core.system_doctor import run_and_notify as _doctor_notify
                    loop = asyncio.get_event_loop()
                    
                    _doctor_sent = False
                    for _attempt in range(3):  # 3 attempts
                        try:
                            report = await loop.run_in_executor(None, _doctor_notify)
                            _tg = report.get('telegram_sent', False)
                            LOGGER.info(f"[DOCTOR-CRON] 🩺 {report['overall']} ({report['passed']}/{report['passed']+report['failed']}) telegram={'sent' if _tg else 'NOT sent'} (attempt {_attempt+1})")
                            if _tg:
                                _doctor_sent = True
                                break
                            else:
                                LOGGER.warning(f"[DOCTOR-CRON] ⚠️ Telegram delivery failed on attempt {_attempt+1}/3")
                                await asyncio.sleep(10)  # brief pause before retry
                        except Exception as _retry_err:
                            LOGGER.error(f"[DOCTOR-CRON] Attempt {_attempt+1}/3 error: {_retry_err}")
                            await asyncio.sleep(10)
                    
                    if not _doctor_sent:
                        LOGGER.error("[DOCTOR-CRON] ❌ All 3 Telegram delivery attempts failed — doctor ran but notification not sent")
                    
                    _last_fire_date = fire_date
                    await asyncio.sleep(120)  # skip past the minute boundary
                except Exception as _de:
                    LOGGER.error(f"[DOCTOR-CRON] Error: {_de}", exc_info=True)
                    await asyncio.sleep(300)

        asyncio.create_task(_doctor_cron_loop())
        LOGGER.info("[POST-STARTUP] 🩺 System Doctor wired + independent 7 AM cron started")
    except Exception as e:
        LOGGER.warning(f"[POST-STARTUP] System Doctor wire failed (non-fatal): {e}")

    # ═══════════════════════════════════════════════════════════════════════════════
    # WEB + WORKER MODE: Critical tasks that must run in ALL modes
    # ═══════════════════════════════════════════════════════════════════════════════

    # Start alert worker (needed for Telegram/Slack/webhook notifications in ALL modes)
    try:
        _start_alert_worker()
        LOGGER.info("[ALL MODES] ✅ Alert worker started (Telegram/Slack/webhook delivery)")
    except Exception:
        LOGGER.exception("alert_worker_start_failed", extra={"component": "startup"})

    # Start accuracy tracker (needed for accuracy trending charts in ALL modes)
    try:
        _start_accuracy_tracker()
        LOGGER.info("[ALL MODES] ✅ Accuracy tracker started (trending snapshots every 5 min)")
    except Exception:
        LOGGER.exception("accuracy_tracker_start_failed", extra={"component": "startup"})

    # Start accuracy autopilot periodic check (circuit breakers)
    try:
        async def _autopilot_check_loop():
            """Check accuracy autopilot circuit breakers every 5 minutes.
            
            Three breakers:
            1. Accuracy breaker — pause if system accuracy <40%
            2. Feed breaker — pause if all price feeds are down
            3. Confidence floor — 55% minimum (checked per-prediction)
            
            Sends Telegram alerts on state changes.
            """
            await asyncio.sleep(60)  # let startup finish
            LOGGER.info("[AUTOPILOT] 🛡️ Accuracy autopilot check loop started (every 5 min)")
            while True:
                try:
                    _heartbeat_pulse("autopilot-check")
                    from core.accuracy_autopilot import check_and_update as _ap_check_update
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, _ap_check_update)
                except Exception as _ap_err:
                    LOGGER.warning(f"[AUTOPILOT] Check failed (non-fatal): {_ap_err}")
                await asyncio.sleep(300)  # 5 minutes

        asyncio.create_task(_autopilot_check_loop())
        LOGGER.info("[ALL MODES] ✅ Accuracy autopilot started (circuit breakers every 5 min)")
    except Exception as e:
        LOGGER.warning(f"[ALL MODES] Autopilot start failed (non-fatal): {e}")

    # CRITICAL: Check if this is WORKER mode or WEB mode
    WORKER_MODE = os.getenv("WORKER_MODE") == "1"
    
    if not WORKER_MODE:
        LOGGER.info("[WEB MODE] Heavy background engines DISABLED (predictions still running)")
        LOGGER.info("[WEB MODE] Price recorder will still run (needed for accuracy evaluation)")
        # FIX (Mar 7, 2026): Start price recorder even in web mode.
        # Without this, the evaluator has ZERO price data → 0% WR forever.
        # Railway only runs the 'web' process, so price recording must happen here.
        try:
            from core.prediction_price_recorder import price_recording_loop as _web_price_loop

            _web_price_pulse_counter = 0

            def _web_fetch_price(sym: str) -> float | None:
                nonlocal _web_price_pulse_counter
                _web_price_pulse_counter += 1
                if _web_price_pulse_counter % 5 == 1:  # Pulse every ~5 cycles (not every call)
                    _heartbeat_pulse("price-recorder")
                sym = (sym or "").upper().strip()
                if not sym:
                    return None
                is_crypto_local = sym in HUNTER_CRYPTO_SYMBOLS or _classify_symbol_category(sym) == "crypto"
                if is_crypto_local:
                    res = turbo_crypto_price(sym, max_budget_s=2.0)
                else:
                    res = turbo_stock_price(sym, max_budget_s=2.0)
                if res and res.get("ok") and res.get("price"):
                    return float(res["price"])
                return None

            loop = asyncio.get_running_loop()
            loop.create_task(_web_price_loop(_web_fetch_price))
            LOGGER.info("[WEB MODE] ✅ Price recorder started (evaluator needs PostgreSQL price data)")
        except Exception as _web_pr_err:
            LOGGER.error(f"[WEB MODE] price recorder failed to start: {_web_pr_err}", exc_info=False)
        # NOTE: Do NOT return here — money-game and other critical tasks
        # must run even in web mode. Heavy scanners are gated individually below.

    # Start price recorder for touch-target evaluation (worker-only — web mode starts its own)
    if WORKER_MODE:
        try:
            from core.prediction_price_recorder import price_recording_loop

            def _fetch_price_for_recorder(sym: str) -> float | None:
                sym = (sym or "").upper().strip()
                if not sym:
                    return None
                is_crypto_local = sym in HUNTER_CRYPTO_SYMBOLS or _classify_symbol_category(sym) == "crypto"
                if is_crypto_local:
                    res = turbo_crypto_price(sym, max_budget_s=2.0)
                else:
                    res = turbo_stock_price(sym, max_budget_s=2.0)
                if res and res.get("ok") and res.get("price"):
                    return float(res["price"])
                return None

            loop = asyncio.get_running_loop()
            loop.create_task(price_recording_loop(_fetch_price_for_recorder))
            LOGGER.info("[WORKER MODE] ✅ Price recorder started (touch-target evaluation)")
        except Exception as e:
            LOGGER.error(f"[WORKER MODE] price recorder failed to start: {e}", exc_info=False)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # HEAVY TASKS — only run in WORKER mode (scanners, orchestrator, execution)
    # Critical tasks (money-game, telegram reports) run in ALL modes further below.
    # ═══════════════════════════════════════════════════════════════════════════════
    
    if not WORKER_MODE:
        LOGGER.info("[WEB MODE] Heavy scanners/orchestrator SKIPPED (deploy worker for those)")
    else:
        LOGGER.info("[WORKER MODE] Starting heavy background services...")
        await asyncio.sleep(2)
    
    # NOTE: Postgres pool initialization removed - will lazy-init on first request
    # Startup must complete in <100s for Railway healthcheck, Postgres can take 30s+
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # TO THE MOON: Use Master Orchestrator for unified service management
    # (WORKER_MODE only — too heavy for web process)
    # ═══════════════════════════════════════════════════════════════════════════════
    orchestrator_enabled = os.getenv("ORCHESTRATOR_ENABLED", "0") == "1" and WORKER_MODE
    
    if orchestrator_enabled:
        try:
            LOGGER.info("[POST-STARTUP] 🎭 Master Orchestrator: Starting all background services...")
            from core.orchestrator import start_all_background_services
            
            # Create wrapper functions for beast_scheduler callbacks
            def beast_fetch_price(symbol: str, market: str):
                """Wrapper for beast_scheduler GET_PRICE_FUNC"""
                try:
                    if market == "crypto":
                        from core.crypto.crypto_providers import get_crypto_price_quorum
                        result = get_crypto_price_quorum(symbol)
                        if result and result.get("price"):
                            return (result["price"], result["price"], result.get("provider", "unknown"), False)
                    else:
                        # Stock price - use turbo provider
                        from core.providers.turbo_provider import turbo_stock_price
                        price_data = turbo_stock_price(symbol)
                        if price_data and price_data.get("price"):
                            return (price_data["price"], price_data.get("prev_close"), price_data.get("provider", "unknown"), False)
                    return None
                except Exception as e:
                    LOGGER.warning(f"beast_fetch_price failed for {symbol}: {e}")
                    return None
            
            def beast_run_prediction(symbol: str, market: str, horizon: str):
                """Wrapper for beast_scheduler RUN_PREDICTION_FUNC"""
                try:
                    # Run async prediction in sync context
                    import asyncio
                    loop = asyncio.new_event_loop()
                    result = loop.run_until_complete(run_single_prediction_async(symbol))
                    loop.close()
                    return result
                except Exception as e:
                    LOGGER.warning(f"beast_run_prediction failed for {symbol}: {e}")
                    return {"ok": False, "error": str(e)}
            
            # Start ALL background services through orchestrator
            await start_all_background_services(
                app=APP,
                logger=LOGGER,
                redis_client=None,  # Redis not required (optional caching layer)
                fetch_price_func=beast_fetch_price,
                run_prediction_func=beast_run_prediction
            )
            
            LOGGER.info("[POST-STARTUP] ✅ Master Orchestrator: All systems operational")
        except Exception as e:
            LOGGER.error(f"[POST-STARTUP] ❌ Master Orchestrator failed: {e}", exc_info=True)
    else:
        LOGGER.info("[POST-STARTUP] ℹ️  Master Orchestrator disabled (set ORCHESTRATOR_ENABLED=1 to enable)")

    # Stage 4: Initialize Portfolio Optimization & Advanced Strategies
    if STAGE4_ENABLED:
        try:
            portfolio_mgr = get_portfolio_manager()
            get_hedging_engine()
            get_backtester()
            get_strategy_tester()
            LOGGER.info(
                "stage4_initialized",
                extra={
                    "component": "startup",
                    "features": "portfolio_manager,hedging_engine,backtester,strategy_tester",
                    "portfolio_constraints": {
                        "min_weight_pct": portfolio_mgr.min_weight * 100,
                        "max_weight_pct": portfolio_mgr.max_weight * 100,
                        "target_sharpe": portfolio_mgr.target_sharpe,
                    },
                },
            )
        except Exception as e:
            LOGGER.error(f"stage4_init_failed: {e}", extra={"component": "startup"}, exc_info=False)

    # BROKER FEATURES DISABLED - Ghost is an investment hunter, not a trading platform
    # If you want broker features, set BROKER_ENABLED=1 in Railway Variables
    #
    # # Start SL/TP monitoring background task
    # try:
    #     import asyncio as _asyncio_module
    #     from core.sl_tp_monitor import start_sl_tp_monitor
    #     _asyncio_module.create_task(start_sl_tp_monitor())
    #     LOGGER.info("sl_tp_monitor_started", extra={"component": "startup"})
    # except Exception as e:
    #     LOGGER.error(f"sl_tp_monitor_failed: {e}", extra={"component": "startup"}, exc_info=False)
    #
    # # Start order status sync background task
    # try:
    #     import asyncio as _asyncio_module
    #     from core.order_sync import start_order_sync
    #     _asyncio_module.create_task(start_order_sync())
    #     LOGGER.info("order_sync_started", extra={"component": "startup"})
    # except Exception as e:
    #     LOGGER.error(f"order_sync_failed: {e}", extra={"component": "startup"}, exc_info=False)

    # Start VIP Microcap Scanner (WORKER_MODE only — heavy scanning)
    if WORKER_MODE:
        try:
            from core.vip_scanner import scan_vip_coins, VIP_SCAN_INTERVAL_S
        
            async def _vip_scanner_loop():
                """Background loop for VIP microcap scanning with Cash-App alerts"""
                while True:
                    try:
                        _heartbeat_pulse("vip-scanner")
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(None, scan_vip_coins)
                        LOGGER.info(
                            f"VIP scan: {result['available']}/{result['scanned']} available, "
                            f"{len(result['opportunities'])} opportunities, {result['alerts_sent']} alerts"
                        )
                    except Exception as e:
                        LOGGER.error(f"VIP scanner error: {e}", exc_info=True)
                    await asyncio.sleep(VIP_SCAN_INTERVAL_S)
        
            asyncio.create_task(_vip_scanner_loop())
            LOGGER.info("✅ VIP Microcap Scanner: STARTED (60s interval, Cash-App alerts)")
        except Exception as e:
            LOGGER.error(f"vip_scanner_start_failed: {e}", extra={"component": "startup"}, exc_info=False)
    
    # Start Pre-Market Predictor (7AM CT weekdays) — WORKER_MODE only
    if WORKER_MODE:
        try:
            from core.premarket_predictor import should_run_premarket, run_premarket_predictions
        
            async def _premarket_loop():
                """Check for pre-market prediction trigger (7AM CT weekdays)"""
                while True:
                    try:
                        _heartbeat_pulse("premarket-scanner")
                        should_run, reason = should_run_premarket()
                        if should_run:
                            LOGGER.info(f"🌅 Running pre-market predictions... ({reason})")
                            await run_premarket_predictions()
                            LOGGER.info("✅ Pre-market predictions complete")
                    except Exception as e:
                        LOGGER.error(f"Pre-market predictor error: {e}", exc_info=True)
                    await asyncio.sleep(60)
        
            asyncio.create_task(_premarket_loop())
            LOGGER.info("✅ Pre-Market Predictor: STARTED (7AM CT weekdays)")
        except Exception as e:
            LOGGER.error(f"premarket_start_failed: {e}", extra={"component": "startup"}, exc_info=False)
    
    # Start Full Market Scanner (5AM CT weekdays) — WORKER_MODE only
    if WORKER_MODE:
        try:
            from core.full_market_scanner import should_run_full_scan, run_full_market_scan, check_hourly_movers
        
            async def _full_scanner_loop():
                """Full market scan at 5AM CT + hourly mover detection"""
                while True:
                    try:
                        _heartbeat_pulse("full-scanner")
                        should_run, reason = should_run_full_scan()
                        if should_run:
                            LOGGER.info(f"🔮 Running FULL MARKET SCAN... ({reason})")
                            await run_full_market_scan()
                            LOGGER.info("✅ Full market scan complete")
                    
                        import time
                        current_minute = int(time.time() / 60) % 60
                        if current_minute == 0:
                            await check_hourly_movers()
                        
                    except Exception as e:
                        LOGGER.error(f"Full market scanner error: {e}", exc_info=True)
                    await asyncio.sleep(60)
        
            asyncio.create_task(_full_scanner_loop())
            LOGGER.info("🔮 Full Market Scanner: STARTED (5AM CT daily + hourly movers)")
        except Exception as e:
            LOGGER.error(f"full_market_scanner_start_failed: {e}", extra={"component": "startup"}, exc_info=False)
    
    # NOTE: News Brain startup moved BEFORE WORKER_MODE check (runs in all modes)
    # NOTE: Notification system moved BEFORE WORKER_MODE check (Feb 11, 2026)
    
    # Stage 4: Self-Improvement Engine (WORKER_MODE only)
    # REMOVED (Step 4B, Mar 17 2026): Duplicate registration — Instance 1 at L5183
    # already runs in ALL modes with correct 6h cadence. This Instance 2 ran every
    # 1h only in WORKER_MODE, causing race conditions and heartbeat confusion.
    if WORKER_MODE:
        LOGGER.info("🧠 [POST-STARTUP] Self-Improvement Engine: handled by all-mode instance (6h cadence)")
    
    # Stage 5: Start Autonomous Execution Engine (WORKER_MODE only)
    if WORKER_MODE:
        LOGGER.info("🤖 [POST-STARTUP] Initializing Phase 5 Autonomous Execution Engine...")
        try:
            from core.autonomous_execution_engine import run_execution_cycle
        
            execution_enabled = os.getenv("AUTO_EXECUTION_ENABLED", "0") == "1"
            execution_interval = int(os.getenv("AUTO_EXECUTION_INTERVAL_S", "300"))
            orchestrator_enabled = os.getenv("ORCHESTRATOR_ENABLED", "0") == "1"
            worker_mode = os.getenv("WORKER_MODE", "0") == "1"
        
            LOGGER.info(f"🤖 [POST-STARTUP] Phase 5 config loaded: enabled={execution_enabled}, interval={execution_interval}s")
        
            if execution_enabled and not orchestrator_enabled and not worker_mode:
                async def _autonomous_execution_loop():
                    """Background task to execute trades every 5 minutes"""
                    await asyncio.sleep(60)
                    while True:
                        try:
                            LOGGER.info("🤖 [AUTO-EXECUTION] Starting execution cycle...")
                            loop = asyncio.get_event_loop()
                            result = await loop.run_in_executor(None, run_execution_cycle)
                            status = result.get('status', 'unknown') if isinstance(result, dict) else 'unknown'
                            LOGGER.info(f"🤖 [AUTO-EXECUTION] Cycle complete: {status}")
                        except Exception as exec_err:
                            LOGGER.error(f"🤖 [AUTO-EXECUTION] Cycle error: {exec_err}", exc_info=False)
                        await asyncio.sleep(execution_interval)
            
                asyncio.create_task(_autonomous_execution_loop())
                LOGGER.info(f"🤖 [POST-STARTUP] ✅ Phase 5 Autonomous Execution ACTIVE (interval={execution_interval}s)")
            elif execution_enabled and (orchestrator_enabled or worker_mode):
                LOGGER.info("🤖 [POST-STARTUP] Phase 5 loop skipped (orchestrator/worker will manage execution)")
            else:
                LOGGER.info("🤖 [POST-STARTUP] Phase 5 Autonomous Execution DISABLED (set AUTO_EXECUTION_ENABLED=1 to enable)")
        except Exception as e:
            LOGGER.error(f"🚨 [POST-STARTUP] Phase 5 initialization FAILED: {e}", extra={"component": "startup"}, exc_info=True)
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # ALL-MODE TASKS — run in BOTH web and worker mode
    # Money-game and Telegram reports are critical for core functionality
    # ═══════════════════════════════════════════════════════════════════════════════
    
    # Start Telegram daily report scheduler (Ghost Investment Hunter)
    try:
        import asyncio as _asyncio_module
        from core.telegram_hunter import daily_report_loop
        from core.market_scanner import scan_all
        from core.prediction_tracker import calculate_accuracy

        async def get_top_opportunities():
            """Get top opportunities from high-confidence predictions.
            
            FIX (Step 4A, Mar 17 2026): Now applies kill switch filter.
            Previously read _LATEST_PREDICTIONS unfiltered — killed symbols
            (PANW, DDOG, XPO, NET, FTNT) appeared in Morning/Evening reports.
            """
            min_conf = float(os.getenv("MIN_ALERT_CONFIDENCE", "0.60"))
            opportunities = []

            # ── Build kill-switch blocked set (same logic as should_create_prediction) ──
            _report_blocked: set = set()
            try:
                from core.db_pool import get_sync_connection as _rpt_conn
                with _rpt_conn() as _rc:
                    try:
                        _rc.rollback()
                    except Exception:
                        pass
                    _rcur = _rc.cursor()
                    _rcur.execute("""
                        SELECT symbol, COUNT(*) AS total,
                               COALESCE(SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END), 0) AS wins
                        FROM ghost_predictions
                        WHERE correct IS NOT NULL
                          AND (eval_version IS NULL OR eval_version NOT LIKE 'skip%%')
                        GROUP BY symbol
                    """)
                    for _row in _rcur.fetchall():
                        _sym, _tot, _wins = _row[0], _row[1], _row[2]
                        if _tot >= _PREDICTION_GATE_KILL_SWITCH_MIN_TRADES:
                            _wr = round(_wins / _tot * 100, 1)
                            if _wr < _PREDICTION_GATE_KILL_SWITCH_MIN_WINRATE:
                                _report_blocked.add(_sym)
                    _rcur.close()
                if _report_blocked:
                    LOGGER.info(f"[REPORT] Kill switch blocking {len(_report_blocked)} symbols: {sorted(_report_blocked)}")
            except Exception as _rpt_err:
                LOGGER.warning(f"[REPORT] Kill switch query failed (fail-open): {_rpt_err}")

            for sym, pred in _LATEST_PREDICTIONS.items():
                if sym in _report_blocked:
                    continue  # Kill switch — skip losing symbols
                confidence = pred.get("confidence", 0)
                if confidence >= min_conf:
                    predicted_pct = 0.0
                    forecast = pred.get("forecast", [])
                    if forecast and len(forecast) >= 2:
                        try:
                            predicted_pct = ((forecast[-1] - forecast[0]) / forecast[0]) * 100
                        except (ZeroDivisionError, TypeError):
                            predicted_pct = 0.0
                    
                    opportunities.append({
                        "symbol": sym,
                        "confidence": confidence,
                        "predicted_pct": round(predicted_pct, 2),
                        "action": pred.get("direction", "HOLD"),
                        "score": int(confidence * 100),
                        "timeframe_hours": pred.get("horizon_h", 48),
                    })
            opportunities.sort(key=lambda x: x["confidence"], reverse=True)
            return opportunities[:10]

        async def get_accuracy_stats(period="24h"):
            """Get accuracy stats for daily report from ghost_predictions (authoritative).
            
            FIX (Step 4A, Mar 17 2026): Was reading ghost_prediction_outcomes which
            uses a different evaluator with different thresholds. Now reads directly
            from ghost_predictions (same table the evaluator writes to) and excludes
            killed symbols so the report shows only surviving-symbol accuracy.
            """
            try:
                from core.db_pool import get_sync_connection as _acc_conn
                with _acc_conn() as _ac:
                    try:
                        _ac.rollback()
                    except Exception:
                        pass
                    _acur = _ac.cursor()

                    time_filter = ""
                    if period == "24h":
                        _cutoff = int(time.time()) - 86400
                        time_filter = f"AND checked_at > {_cutoff}"
                    elif period == "7d":
                        _cutoff = int(time.time()) - 7 * 86400
                        time_filter = f"AND checked_at > {_cutoff}"

                    _acur.execute(f"""
                        SELECT COUNT(*) AS total,
                               COALESCE(SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END), 0) AS wins
                        FROM ghost_predictions
                        WHERE checked = 1
                          AND eval_version NOT LIKE 'skip%%'
                          {time_filter}
                    """)
                    _arow = _acur.fetchone()
                    _total = _arow[0] if _arow else 0
                    _correct = _arow[1] if _arow else 0
                    _acur.close()

                _acc_pct = round(_correct / _total * 100, 1) if _total > 0 else 0.0
                return {
                    "accuracy_pct": _acc_pct,
                    "total_predictions": _total,
                    "correct_predictions": _correct,
                }
            except Exception as _acc_err:
                LOGGER.warning(f"[REPORT] Accuracy query failed: {_acc_err}")
                return {"accuracy_pct": 0.0, "total_predictions": 0, "correct_predictions": 0}

        _asyncio_module.create_task(daily_report_loop(get_top_opportunities, get_accuracy_stats))
        LOGGER.info("telegram_daily_reports_started", extra={"component": "startup"})
    except Exception as e:
        LOGGER.error(f"telegram_reports_failed: {e}", extra={"component": "startup"}, exc_info=False)

    # ═══════════════════════════════════════════════════════════════════════════════
    # MONEY GAME SCOUT SCHEDULER - Automated daily cycles + Telegram alerts
    # ═══════════════════════════════════════════════════════════════════════════════
    try:
        money_game_enabled = os.getenv("MONEY_GAME_ENABLED", "1") == "1"
        
        if money_game_enabled:
            async def _money_game_scheduler():
                """
                🎰 MONEY GAME SCHEDULER
                
                Runs at strategic times (Central Time):
                - 6:00 AM: Full daily scout (finds today's opportunities)
                - 8:00 AM: Send TOP 10 Telegram alert
                - 6:00 PM: Resolve today's trades, update rankings
                
                Also runs on startup to ensure predictions are fresh.
                """
                global _LAST_TELEGRAM_STATUS, _LAST_TELEGRAM_SEND_TIME, _LAST_TELEGRAM_ERROR
                from datetime import time as dt_time
                
                try:
                    from zoneinfo import ZoneInfo
                    central_tz = ZoneInfo("America/Chicago")
                except ImportError:
                    central_tz = None
                
                # Wait for other services to initialize
                await asyncio.sleep(30)
                
                # Run initial scout on startup to populate predictions
                try:
                    LOGGER.info("🎰 [MONEY-GAME] Running startup scout...")
                    from core.smart_scout import SmartScout
                    scout = SmartScout()
                    result = scout.full_scout()
                    stocks_n = result.get("stocks", {}).get("scouted", 0)
                    crypto_n = result.get("crypto", {}).get("scouted", 0)
                    LOGGER.info(f"🎰 [MONEY-GAME] Startup scout complete: {stocks_n} stocks, {crypto_n} crypto")
                except Exception as e:
                    LOGGER.error(f"🎰 [MONEY-GAME] Startup scout error: {e}")
                
                # Main scheduler loop
                while True:
                    try:
                        _heartbeat_pulse("money-game")
                        now = datetime.now(central_tz) if central_tz else datetime.utcnow()
                        current_hour = now.hour
                        current_minute = now.minute
                        
                        # 6:00 AM CT - Full daily scout
                        if current_hour == 6 and current_minute == 0:
                            LOGGER.info("🎰 [MONEY-GAME] 6 AM - Running full daily scout...")
                            try:
                                from core.smart_scout import SmartScout
                                scout = SmartScout()
                                result = scout.full_scout()
                                total = result.get("total_scouted", 0)
                                LOGGER.info(f"🎰 [MONEY-GAME] Scout complete: {total} assets")
                            except Exception as e:
                                LOGGER.error(f"🎰 [MONEY-GAME] Scout error: {e}")
                        
                        # 8:00 AM CT - DISABLED (Feb 24, 2026)
                        # The Ghost notification loop (_ghost_notification_loop) already
                        # sends the proper TOP 10 card at 8 AM with V3 filtering, regime
                        # filter, and calibrated confidence. This Money Game send was a
                        # DUPLICATE that sent a second, simpler message at 8:00 AM.
                        # if current_hour == 8 and current_minute == 0:
                        #     ... (removed to prevent duplicate 8 AM Telegram messages)
                        
                        # 6:00 PM CT - Resolve trades and update rankings
                        if current_hour == 18 and current_minute == 0:
                            LOGGER.info("🎰 [MONEY-GAME] 6 PM - Resolving trades...")
                            try:
                                from core.smart_scout import run_daily_cycle
                                result = run_daily_cycle()
                                LOGGER.info(f"🎰 [MONEY-GAME] Daily cycle complete: {result}")
                            except Exception as e:
                                LOGGER.error(f"🎰 [MONEY-GAME] Resolve error: {e}")
                        
                        # Sleep for 60 seconds before next check
                        await asyncio.sleep(60)
                        
                    except asyncio.CancelledError:
                        LOGGER.info("🎰 [MONEY-GAME] Scheduler cancelled")
                        break
                    except Exception as e:
                        LOGGER.error(f"🎰 [MONEY-GAME] Scheduler error: {e}")
                        await asyncio.sleep(60)
            
            asyncio.create_task(_money_game_scheduler())
            LOGGER.info("🎰 [POST-STARTUP] ✅ Money Game Scheduler ACTIVATED (6 AM scout, 8 AM alerts, 6 PM resolve)")
        else:
            LOGGER.info("🎰 [POST-STARTUP] Money Game Scheduler DISABLED (set MONEY_GAME_ENABLED=1)")
    except Exception as e:
        LOGGER.error(f"money_game_scheduler_failed: {e}", extra={"component": "startup"}, exc_info=False)

    LOGGER.info("🟣 Ghost Investment Hunter initialized - broker features disabled", extra={"component": "startup"})

    # Stage 5: Initialize Advanced Execution & Order Management
    if STAGE5_ENABLED:
        try:
            get_order_manager()
            get_smart_router()
            get_execution_analytics()
            exec_risk = get_execution_risk()
            LOGGER.info(
                "stage5_initialized",
                extra={
                    "component": "startup",
                    "features": "order_manager,smart_router,execution_analytics,execution_risk",
                    "risk_limits": exec_risk.get_risk_limits(),
                    "trading_enabled": exec_risk.trading_enabled,
                },
            )
        except Exception as e:
            LOGGER.exception("stage5_init_failed", extra={"component": "startup", "error": str(e)})

    # NEW: Initialize forecast grid (two-line overlay system)
    try:
        grid = _generate_forecast_grid(WOLF)
        points_count = len(grid.get("points", []))
        horizon_h = grid.get("horizon_s", 0) / 3600
        LOGGER.info(
            "forecast_grid_ready",
            extra={
                "component": "startup",
                "symbol": WOLF,
                "points": points_count,
                "horizon_h": horizon_h,
                "model": grid.get("meta", {}).get("model"),
                "con": grid.get("meta", {}).get("con"),
            },
        )
        _add_event(
            "forecast.grid",
            "Forecast grid initialized",
            {"symbol": WOLF, "points": points_count, "horizon_h": horizon_h},
        )
    except Exception as e:
        LOGGER.error("forecast_grid_init_failed", extra={"component": "startup", "error": str(e)})
    # NEW: Migrate legacy AI memory snapshots into persistent AIMemory store (one-time)
    try:
        migrated = _migrate_legacy_ai_memory()
        if migrated:
            LOGGER.info("ai_memory_startup_migration", extra={"migrated": migrated})
    except Exception as e:
        LOGGER.warning("ai_memory_startup_migration_failed", extra={"error": str(e)})
    # Load persisted position if configured
    try:
        _persist_load()
    except Exception:
        LOGGER.exception("persist_load_failed", extra={"component": "startup"})

    # Sync STATE from ghost_state.json if positions are missing/empty
    try:
        if not STATE.get("positions") or STATE.get("positions") == []:
            ghost_state_path = os.getenv("GHOST_STATE_PATH", "ghost_state.json")
            if os.path.exists(ghost_state_path):
                with open(ghost_state_path, encoding="utf-8") as f:
                    ghost_data = json.load(f)
                    trading_state = ghost_data.get("trading_state", {})
                    positions = trading_state.get("positions", [])
                    if positions:
                        # Sync positions array
                        STATE["positions"] = positions
                        # Sync cash balances
                        cash_data = trading_state.get("cash", {})
                        if isinstance(cash_data, dict):
                            STATE["cash_stock"] = float(cash_data.get("stock", 0.0))
                            STATE["cash_crypto"] = float(cash_data.get("crypto", 0.0))
                            STATE["cash"] = STATE["cash_stock"] + STATE["cash_crypto"]
                        elif isinstance(cash_data, (int, float)):
                            STATE["cash"] = float(cash_data)
                        # Extract WOLF position for legacy fields
                        wolf_pos = next((p for p in positions if p.get("symbol") == WOLF), None)
                        if wolf_pos:
                            STATE["qty"] = float(
                                wolf_pos.get("quantity", wolf_pos.get("qty", 0.0))
                            )  # Support both field names
                            STATE["avg_cost"] = float(
                                wolf_pos.get("entry_price", wolf_pos.get("price", 0.0))
                            )  # Support both field names
                        LOGGER.info(
                            "state_synced_from_ghost_state",
                            extra={
                                "component": "startup",
                                "positions": len(positions),
                                "cash": STATE.get("cash", 0.0),
                                "wolf_qty": STATE.get("qty", 0.0),
                            },
                        )
                        _persist_save()  # Persist to wolf_state.json/db
    except Exception as e:
        LOGGER.warning("ghost_state_sync_failed", extra={"component": "startup", "error": str(e)})

    # --- ENV VALIDATION (Phase Upgrade → 90% Ops) ---
    # Enforce required ENV gates for live operation, fail closed if missing
    env_violations = []

    # Check critical configuration gates
    delisted_mode = os.getenv("DELISTED_MODE", "").strip()
    if delisted_mode not in ("0", ""):
        env_violations.append("DELISTED_MODE must be 0 or unset")

    allow_safe_price = os.getenv("ALLOW_SAFE_PRICE", "0").strip()
    if allow_safe_price not in ("0", ""):
        env_violations.append("ALLOW_SAFE_PRICE must be 0 or unset")

    price_fallback_prevclose = os.getenv("PRICE_FALLBACK_PREVCLOSE", "0").strip()
    if price_fallback_prevclose not in ("0", ""):
        env_violations.append("PRICE_FALLBACK_PREVCLOSE must be 0 or unset")

    # Check provider configuration
    if not POLYGON_KEY:
        env_violations.append("POLYGON_API_KEY is missing")

    if not ALPHAVANTAGE_KEY:
        env_violations.append("ALPHAVANTAGE_API_KEY is missing")

    # Log validation results
    if env_violations:
        STATE["degraded_reason"] = "; ".join(env_violations)
        LOGGER.warning(
            "env_validation_failed",
            extra={
                "component": "startup",
                "violations": env_violations,
                "impact": "Prediction endpoints will return 503 until resolved"
            }
        )
        _add_event(
            "env.validation",
            "ENV validation failed",
            {"violations": env_violations}
        )
    else:
        STATE.pop("degraded_reason", None)
        LOGGER.info(
            "env_validation_passed",
            extra={
                "component": "startup",
                "checks": [
                    "SIM_MODE=0",
                    "DELISTED_MODE=0",
                    "ALLOW_SAFE_PRICE=0",
                    "PRICE_FALLBACK_PREVCLOSE=0",
                    "POLYGON_API_KEY present",
                    "ALPHAVANTAGE_API_KEY present"
                ]
            }
        )

    # Bootstrap initial portfolio and watchlist from ghost_init_data.json
    try:
        from ghost_bootstrap import get_bootstrap_status, run_bootstrap

        bootstrap_success = run_bootstrap()
        if bootstrap_success:
            status = get_bootstrap_status()
            LOGGER.info(
                "bootstrap_complete",
                extra={
                    "component": "startup",
                    "portfolio_positions": status.get("portfolio_count", 0),
                    "watchlist_symbols": status.get("watchlist_count", 0),
                },
            )
        else:
            LOGGER.warning("bootstrap_skipped", extra={"component": "startup"})
    except Exception as e:
        LOGGER.exception("bootstrap_failed", extra={"component": "startup", "error": str(e)})
    # NOTE: alert-worker and accuracy-tracker now start BEFORE the WORKER_MODE gate
    # (moved in v5.7) so they run in ALL modes. See _post_startup_init() above.
    # Start open/close scheduler (optional)
    try:
        if SCHEDULE_OPEN_CLOSE:
            _start_schedule_worker()
    except Exception:
        LOGGER.exception("schedule_worker_start_failed", extra={"component": "startup"})
    # OLD RECONCILER DISABLED - Using outcome_reconciler_v2 instead (started at line 3651)
    # REASON: V2 has batch limits, timeouts, and circuit breaker protection
    # Old reconciler lacked protections and caused crashes when processing large batches
    # try:
    #     _start_reconciler_worker()
    # except Exception:
    #     LOGGER.exception("reconciler_worker_start_failed", extra={"component": "startup"})

    # Scheduled Predictions DISABLED - Using auto_prediction_loop instead (5-min interval covers all cases)
    # REASON: Prevents duplicate predictions and excessive API calls
    # The auto_prediction_loop (started below at line ~4018) handles all symbols every 5 minutes
    try:
        if False:  # SCHEDULED_PREDICTIONS_ENABLED - INTENTIONALLY DISABLED
            # Configure the scheduler with multi-symbol functions
            scheduled_predictions.MULTI_SYMBOL_PREDICTION_FUNC = _generate_multi_symbol_predictions
            scheduled_predictions.TELEGRAM_SEND_MULTI_FUNC = _send_multi_symbol_telegram_alert
            scheduled_predictions.LOGGER = LOGGER

            scheduled_predictions.start_prediction_scheduler()
            LOGGER.info("Scheduled predictions enabled: 8:00 AM, 12:00 PM, 4:00 PM ET (multi-symbol)")

            # Phase 2: Bootstrap prediction counters from database
            try:
                stock_count = 0
                crypto_count = 0

                # Count recent predictions for stocks
                for sym in STOCK_SYMBOLS[:10]:  # Check first 10 stocks
                    try:
                        pred = predictor.get_latest_prediction(sym)
                        if pred and (time.time() - pred.run_at) < 86400:  # Last 24h
                            stock_count += 1
                    except Exception as e:
                        LOGGER.debug(f"stock_prediction_check_failed for {sym}: {e}")

                # Count recent predictions for crypto
                for sym in CRYPTO_SYMBOLS[:10]:  # Check first 10 crypto
                    try:
                        pred = predictor.get_latest_prediction(sym)
                        if pred and (time.time() - pred.run_at) < 86400:  # Last 24h
                            crypto_count += 1
                    except Exception as e:
                        LOGGER.debug(f"crypto_prediction_check_failed for {sym}: {e}")

                # Update global counters if we found predictions
                if stock_count > 0 or crypto_count > 0:
                    _LAST_MULTI_PREDICTION_COUNTS["stocks"] = stock_count
                    _LAST_MULTI_PREDICTION_COUNTS["crypto"] = crypto_count
                    LOGGER.info(f"Bootstrapped prediction counters from database: stocks={stock_count}, crypto={crypto_count}")
            except Exception as e:
                LOGGER.warning(f"Could not bootstrap prediction counters: {e}")
    except Exception:
        LOGGER.exception("scheduled_predictions_start_failed", extra={"component": "startup"})

    # NOTE: Auto-Prediction Loop now started earlier in _post_startup_init() (before WORKER_MODE check)
    # This ensures predictions run in ALL deployment modes, not just worker mode
    # See lines ~3855-3886 for the actual startup code

    # Start Daily Predictions Engine (6:00 AM briefing with top 5 picks)
    try:
        from core.daily_predictions_engine import daily_briefing_task
        
        # Inject Ghost's actual functions
        # Inject via module globals (lightweight, avoids circular imports)
        from core import daily_predictions_engine as _dpe
        _dpe.RUN_PREDICTION_FUNC_ASYNC = run_single_prediction_async
        _dpe.HUNTER_STOCK_SYMBOLS = HUNTER_STOCK_SYMBOLS
        _dpe.HUNTER_CRYPTO_SYMBOLS = HUNTER_CRYPTO_SYMBOLS
        
        # Start background task
        loop = asyncio.get_running_loop()
        loop.create_task(daily_briefing_task())
        
        LOGGER.info("✅ Daily Predictions Engine: STARTED (6:00 AM CT briefing, top 5 picks)")
    except Exception as e:
        LOGGER.exception("daily_predictions_engine_start_failed", extra={"component": "startup", "error": str(e)})

    # Start Performance Dashboard Monitoring
    try:
        from core.performance_dashboard import dashboard_monitoring_loop
        
        # Start background monitoring task
        loop = asyncio.get_running_loop()
        loop.create_task(dashboard_monitoring_loop())
        
        LOGGER.info("✅ Performance Dashboard: STARTED (hourly monitoring, win rate alerts)")
    except Exception as e:
        LOGGER.exception("performance_dashboard_start_failed", extra={"component": "startup", "error": str(e)})

    # Start Cascading Predictions Scheduler (24h updates, 6h finals, 48h evaluations)
    try:
        from core.cascade_scheduler import start_cascade_scheduler
        from core.cascading_predictor import get_cascade_predictor
        from core import cascade_scheduler
        
        # Initialize cascade predictor and inject into scheduler
        cascade_predictor = get_cascade_predictor()
        cascade_scheduler.CASCADE_PREDICTOR = cascade_predictor
        
        # Start scheduler thread
        start_cascade_scheduler()
        
        LOGGER.info("✅ Cascade Scheduler: STARTED (checking every 10 minutes for updates)")
    except Exception as e:
        LOGGER.exception("cascade_scheduler_start_failed", extra={"component": "startup", "error": str(e)})

    # Start Guardian Oracle System (6 AM prophecy + 24/7 monitoring)
    try:
        from core.cron_scheduler import start_cron_scheduler
        from core.guardian_oracle import get_guardian_oracle
        from core.daily_top_10_scanner import DailyTop10Scanner
        
        # Start cron-based 6 AM morning prophecy
        start_cron_scheduler()
        
        LOGGER.info("🔮 Morning Prophecy Scheduler: STARTED (6:00 AM CT daily)")
        
        # Send initial Top 10 scan on startup
        async def send_startup_prophecy():
            try:
                LOGGER.info("📡 Sending startup Top 10 scan...")
                scanner = DailyTop10Scanner()
                opportunities = await scanner.scan_for_top_10()
                if opportunities:
                    scanner.save_top_10(opportunities)
                    await scanner.send_daily_alert()
                    LOGGER.info(f"✅ Startup prophecy sent: {len(opportunities)} opportunities")
                else:
                    LOGGER.info("No startup opportunities found")
            except Exception as e:
                LOGGER.exception(f"Startup prophecy failed: {e}")
        
        asyncio.create_task(send_startup_prophecy())
        
        # Start 24/7 Guardian monitoring for immediate alerts
        guardian = get_guardian_oracle()
        asyncio.create_task(guardian.guardian_monitor_loop())
        
        LOGGER.info("🛡️ Guardian Oracle: ACTIVATED (24/7 protection)")
        
    except Exception as e:
        LOGGER.exception("guardian_oracle_start_failed", extra={"component": "startup", "error": str(e)})

    # Start Real-Time Market Movers Scanner (discovers TODAY's biggest movers)
    try:
        from core.realtime_market_movers import (
            start_movers_scanner, 
            set_discovery_callback,
            get_scanner_status
        )
        
        def on_movers_discovered(movers):
            """Callback when new movers are discovered - add to prediction queue."""
            for mover in movers:
                symbol = mover["symbol"]
                asset_type = mover["type"]
                change_pct = mover["change_pct"]
                
                LOGGER.info(f"🔥 MOVER DISCOVERED: {symbol} ({asset_type}) {change_pct:+.1f}%")
                
                # Add to appropriate tracking list
                if asset_type == "crypto":
                    if symbol not in HUNTER_CRYPTO_SYMBOLS:
                        HUNTER_CRYPTO_SYMBOLS.append(symbol)
                        LOGGER.info(f"Added {symbol} to crypto tracking")
                else:
                    if symbol not in HUNTER_STOCK_SYMBOLS:
                        HUNTER_STOCK_SYMBOLS.append(symbol)
                        LOGGER.info(f"Added {symbol} to stock tracking")
                
                # Trigger immediate prediction via cascade
                try:
                    asyncio.create_task(_trigger_prediction_for_mover(symbol, mover))
                except Exception as e:
                    LOGGER.error(f"Failed to trigger prediction for {symbol}: {e}")
        
        set_discovery_callback(on_movers_discovered)
        start_movers_scanner()
        
        LOGGER.info("🔥 Real-Time Movers Scanner: STARTED (scanning every 30 min for big moves)")
        
    except Exception as e:
        LOGGER.exception("realtime_movers_start_failed", extra={"component": "startup", "error": str(e)})

    # Optional heartbeat (skip price fetch to avoid blocking startup)
    try:
        if TELEGRAM_HEARTBEAT_ON_START:
            # Use simple message without price to avoid forward reference
            text = "🟢 START — WOLF server ready"
            enqueue_alert_text(text)
    except Exception:
        LOGGER.exception("startup_heartbeat_failed", extra={"component": "startup"})
    # Start autosave thread if enabled
    try:
        _start_autosave_worker()
    except Exception:
        LOGGER.exception("autosave_worker_start_failed", extra={"component": "startup"})


async def _trigger_prediction_for_mover(symbol: str, mover_info: dict):
    """
    Generate prediction for a newly discovered mover.
    Called by real-time movers scanner when it finds a big move.
    """
    try:
        asset_type = mover_info.get("type", "stock")
        change_pct = mover_info.get("change_pct", 0)
        
        LOGGER.info(f"🎯 Generating prediction for mover: {symbol} ({asset_type}) {change_pct:+.1f}%")
        
        # Use the appropriate prediction engine
        if asset_type == "crypto":
            engine = _get_crypto_engine()
        else:
            # For stocks, use the stock engine if available
            engine = _get_crypto_engine()  # Crypto engine can handle both
        
        # Generate prediction
        prediction = await engine.generate_prediction(symbol)
        
        if prediction:
            direction = prediction.get("direction", "FLAT")
            confidence = prediction.get("confidence", 0)
            
            # Send Telegram alert for high-confidence movers
            if confidence >= 0.60:
                alert_text = (
                    f"🔥 *MOVER ALERT: {symbol}*\n"
                    f"📈 Today's Move: {change_pct:+.1f}%\n"
                    f"🎯 Prediction: {direction} ({confidence:.0%})\n"
                    f"💡 Source: Real-time scanner"
                )
                try:
                    enqueue_alert_text(alert_text)
                except Exception as e:
                    LOGGER.warning(f"mover_alert_enqueue_failed for {symbol}: {e}")
            
            LOGGER.info(f"✅ Prediction generated for mover {symbol}: {direction} ({confidence:.0%})")
        
    except Exception as e:
        LOGGER.error(f"Failed to generate prediction for mover {symbol}: {e}")


def _get_redis():
    """Lazy initialize REDIS client with error handling."""
    global REDIS
    if REDIS is None and REDIS_URL:
        try:
            import redis
            REDIS = redis.Redis.from_url(REDIS_URL, decode_responses=True)
            REDIS.ping()  # Test connection
            LOGGER.info("[REDIS] ✅ Connected successfully")
        except Exception as e:
            LOGGER.warning(f"[REDIS] ⚠️ Connection failed: {e} - continuing without cache")
            REDIS = False  # Mark as failed to prevent retries
    return REDIS if REDIS not in (None, False) else None


async def _auto_refresh_price():
    """Periodic task that attempts to refresh the live price.
    Phase 4: Runs 24/7 regardless of market hours for consistent updates.
    Bypasses cache if provider == 'prev-close' or price older than PRICE_TTL_OPEN_S.
    Logs transitions between providers and records diagnostics.
    """
    global _LAST_BG_PRICE_TS
    if PRICE_AUTO_REFRESH_S <= 0:
        return
    import asyncio
    import time

    while True:
        try:
            # Increment tick counter for SSE state change detection
            STATE["tick"] = STATE.get("tick", 0) + 1

            # Phase 4: Always refresh (removed market hours check)
            # This ensures consistent 7s intervals 24/7
            p, prev2, provider2 = get_wolf_price()
            now = time.time()
            stale_prev_only = provider2 == "prev-close"
            if stale_prev_only:
                # Force fresh fetch by clearing cache and re-calling
                PRICE_CACHE.pop(WOLF, None)
                p2, prev3, provider3 = get_wolf_price()
                if provider3 != provider2 or p2 != p:
                    LOGGER.info(
                        "price_updater_live_refresh",
                        extra={
                            "component": "price_updater",
                            "provider": provider3,
                            "price": p2,
                            "prev": prev3,
                        },
                    )
            else:
                # Record occasional heartbeat
                if _LAST_BG_PRICE_TS is None or (now - _LAST_BG_PRICE_TS) > (
                    PRICE_AUTO_REFRESH_S * 4
                ):
                    LOGGER.debug(
                        "price_updater_heartbeat",
                        extra={
                            "component": "price_updater",
                            "provider": provider2,
                            "price": p,
                        },
                    )
            _LAST_BG_PRICE_TS = now
        except Exception as e:
            try:
                LOGGER.debug(
                    "price_updater_error",
                    extra={"component": "price_updater", "error": str(e)},
                )
            except Exception:
                pass
        await asyncio.sleep(PRICE_AUTO_REFRESH_S)


async def _auto_scan_movers():
    """
    Periodic task that scans for market movers.
    - Crypto: every 300 seconds (5 minutes)
    - Stocks: scheduled times in CT timezone
    """
    import asyncio
    from datetime import datetime
    from zoneinfo import ZoneInfo

    # Crypto scan interval
    CRYPTO_SCAN_INTERVAL = 300  # 5 minutes

    # Stock scan times (CT timezone)
    # 07:55, 09:35, then every 10m from 09:40 to 15:50, plus 15:58 summary
    STOCK_SCAN_TIMES = [
        "07:55", "09:35", "09:40", "09:50", "10:00", "10:10", "10:20", "10:30",
        "10:40", "10:50", "11:00", "11:10", "11:20", "11:30", "11:40", "11:50",
        "12:00", "12:10", "12:20", "12:30", "12:40", "12:50", "13:00", "13:10",
        "13:20", "13:30", "13:40", "13:50", "14:00", "14:10", "14:20", "14:30",
        "14:40", "14:50", "15:00", "15:10", "15:20", "15:30", "15:40", "15:50",
        "15:58"
    ]

    last_crypto_scan = 0
    last_stock_scan_minute = None

    try:
        from app.core import movers_scanner
        from core import telegram_alerts
    except Exception as e:
        LOGGER.error(f"Failed to import movers scanner: {e}")
        return

    # Price fetch wrapper
    async def fetch_price_wrapper(symbol: str, is_crypto: bool = False):
        try:
            if is_crypto:
                result = await api_crypto_price(symbol)
                return result
            else:
                result = await fetch_price_live(symbol)
                return {
                    "price": result[0] if result else None,
                    "provider": result[2] if result and len(result) > 2 else "unknown",
                    "ts": int(time.time() * 1000)
                }
        except Exception:
            return None

    while True:
        try:
            now = time.time()
            ct_tz = ZoneInfo("America/Chicago")
            ct_now = datetime.now(ct_tz)
            current_time = ct_now.strftime("%H:%M")
            current_minute = ct_now.strftime("%H:%M")

            # Crypto scan (every 5 minutes)
            if os.getenv("CRYPTO_ENABLED", "0") == "1":
                if now - last_crypto_scan >= CRYPTO_SCAN_INTERVAL:
                    try:
                        redis_client = _get_redis()
                        crypto_movers = await movers_scanner.scan_crypto(
                            fetch_price_wrapper,
                            None,
                            redis_client
                        )

                        # Persist stats
                        movers_scanner.persist_last_run(
                            "crypto",
                            {"count": len(crypto_movers), "ts": int(now), "error": "", "duration_ms": 0},
                            redis_client
                        )

                        # Send alerts for new tier breaches
                        for mover in crypto_movers:
                            telegram_alerts.send_mover_alert("crypto", mover)

                        LOGGER.info(f"Crypto movers scan complete: {len(crypto_movers)} movers")
                        last_crypto_scan = now

                    except Exception as e:
                        LOGGER.error(f"Crypto movers scan failed: {e}")

            # Stock scan (scheduled times)
            if os.getenv("STOCKS_ENABLED", "1") == "1":
                if current_minute in STOCK_SCAN_TIMES and current_minute != last_stock_scan_minute:
                    try:
                        stock_movers = await movers_scanner.scan_stocks(
                            fetch_price_wrapper,
                            None,
                            REDIS
                        )

                        # Persist stats
                        movers_scanner.persist_last_run(
                            "stocks",
                            {"count": len(stock_movers), "ts": int(now), "error": "", "duration_ms": 0},
                            REDIS
                        )

                        # Send alerts for new tier breaches
                        for mover in stock_movers:
                            telegram_alerts.send_mover_alert("stocks", mover)

                        LOGGER.info(f"Stock movers scan complete at {current_time} CT: {len(stock_movers)} movers")
                        last_stock_scan_minute = current_minute

                    except Exception as e:
                        LOGGER.error(f"Stock movers scan failed: {e}")

        except Exception as e:
            LOGGER.error(f"Movers scanner error: {e}")

        # Sleep 60 seconds (check every minute for stock schedule)
        await asyncio.sleep(60)


def _init_security_tables():
    """Initialize persistent storage for API keys and webhooks with proper hashing."""
    try:
        import sqlite3

        with sqlite3.connect(WOLF_SQLITE_PATH) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            # API Keys table with hashed secrets
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    key_hash TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    rate_limit INTEGER NOT NULL DEFAULT 100,
                    created_at REAL NOT NULL,
                    last_used REAL,
                    request_count INTEGER NOT NULL DEFAULT 0,
                    active INTEGER NOT NULL DEFAULT 1
                )
            """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(active)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash)")

            # Webhooks table with hashed secrets
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS webhooks (
                    id TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    events_json TEXT NOT NULL,
                    secret_hash TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_success_ts REAL,
                    failure_count INTEGER NOT NULL DEFAULT 0,
                    active INTEGER NOT NULL DEFAULT 1
                )
            """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_webhooks_active ON webhooks(active)")

            conn.commit()

            # Load API keys into memory cache
            cur.execute("SELECT * FROM api_keys WHERE active=1")
            for row in cur.fetchall():
                API_KEYS_DB[row["id"]] = {
                    "key_hash": row["key_hash"],
                    "name": row["name"],
                    "rate_limit": row["rate_limit"],
                    "created_at": row["created_at"],
                    "last_used": row["last_used"],
                    "request_count": row["request_count"],
                    "active": bool(row["active"]),
                }

            # Load webhooks into memory cache
            cur.execute("SELECT * FROM webhooks WHERE active=1")
            for row in cur.fetchall():
                import json

                WEBHOOK_SUBSCRIPTIONS[row["id"]] = {
                    "url": row["url"],
                    "events": json.loads(row["events_json"]),
                    "secret_hash": row["secret_hash"],
                    "created_at": row["created_at"],
                    "last_success_ts": row["last_success_ts"],
                    "failure_count": row["failure_count"],
                }

            LOGGER.info(
                f"Security tables initialized: {len(API_KEYS_DB)} API keys, {len(WEBHOOK_SUBSCRIPTIONS)} webhooks loaded"
            )
    except Exception as e:
        LOGGER.error(f"Failed to initialize security tables: {e}", exc_info=True)


def _apply_price_override(
    symbol: str, price: float | None, provider: str
) -> tuple[float | None, str]:
    try:
        o = PRICE_OVERRIDE
        sym_ok = (
            isinstance(o.get("symbol"), str) and str(o.get("symbol")).upper() == str(symbol).upper()
        )
        now = time.time()
        override_price = o.get("price")
        if sym_ok and now < float(o.get("until") or 0) and (override_price is not None):
            return float(override_price), "manual"
    except Exception:
        pass
    return price, provider


def _add_event(ev_type: str, message: str, data: dict[str, Any] | None = None) -> dict:
    global _EVENT_SEQ
    EVENT_DEDUP_WINDOW_S = int(os.getenv("EVENT_DEDUP_WINDOW_S", "30"))
    now = time.time()
    now_ts = int(now)
    key = (ev_type, message)
    # Per-second throttle for exact repeats
    last_seen = _EVENT_LAST_TS.get(key, 0.0)
    try:
        if DIAG_COLLAPSE_DUPES and (now - last_seen) < 1.0 and EVENTS:
            # bump the latest matching event
            for i in range(len(EVENTS) - 1, -1, -1):
                it = EVENTS[i]
                if it.get("type") == ev_type and it.get("message") == message:
                    it["ts"] = now_ts
                    it["count"] = int(it.get("count") or 1) + 1
                    if data:
                        it["data"] = data
                    _EVENT_LAST_TS[key] = now
                    return it
    except Exception:
        pass
    # Collapse consecutive repeats in a short window
    try:
        if DIAG_COLLAPSE_DUPES and EVENTS:
            last = EVENTS[-1]
            if last.get("type") == ev_type and last.get("message") == message:
                last_ts = int(last.get("ts") or 0)
                if (now_ts - last_ts) <= max(1, EVENT_DEDUP_WINDOW_S):
                    last["ts"] = now_ts
                    last["count"] = int(last.get("count") or 1) + 1
                    if data:
                        last["data"] = data
                    _EVENT_LAST_TS[key] = now
                    return last
    except Exception:
        pass
    _EVENT_SEQ += 1
    ev = {"id": _EVENT_SEQ, "ts": now_ts, "type": ev_type, "message": message}
    if data:
        ev["data"] = data
    EVENTS.append(ev)
    _EVENT_LAST_TS[key] = now
    return ev


def _ts_to_epoch(ts_val: Any) -> int:
    try:
        # accept int/float epoch or ISO string
        if isinstance(ts_val, (int, float)):
            return int(ts_val)
        s = str(ts_val)
        # handle Z suffix
        s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return int(dt.timestamp())
    except Exception:
        return int(time.time())


def _try_load_finbert() -> bool:
    global _FINBERT_PIPE
    if _FINBERT_PIPE is not None:
        return True
    if not FINBERT_ON:
        return False
    try:
        from transformers import (  # type: ignore
            AutoModelForSequenceClassification,
            AutoTokenizer,
            pipeline,
        )

        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        _FINBERT_PIPE = pipeline(
            "text-classification", model=model, tokenizer=tok, return_all_scores=True
        )
        return True
    except Exception:
        _FINBERT_PIPE = None
        return False


def _score_text_rules(text: str) -> float:
    try:
        t = (text or "").lower()
        score = 0.0
        for k, w in _BULLISH.items():
            if k in t:
                score += w
        for k, w in _BEARISH.items():
            if k in t:
                score += w
        # clamp to [-0.6, 0.6]
        return max(-0.6, min(0.6, score))
    except Exception:
        return 0.0


def _score_text_finbert(text: str) -> float:
    try:
        if not _try_load_finbert():
            return _score_text_rules(text)
        res = _FINBERT_PIPE((text or "")[:1000])  # type: ignore[misc]
        # res like [[{"label":"positive","score":p_pos}, {"label":"neutral"...}]]
        if not res:
            return 0.0
        arr = res[0]
        p_pos = next((r["score"] for r in arr if r.get("label") == "positive"), 0.0)
        p_neg = next((r["score"] for r in arr if r.get("label") == "negative"), 0.0)
        return float(p_pos) - float(p_neg)
    except Exception:
        return _score_text_rules(text)


def _score_news_items(items: list[dict]) -> tuple[list[dict], str]:
    """Return items with 'sent' added and engine label used."""
    engine = "none"
    out: list[dict] = []
    if not items:
        return out, engine
    use_finbert = bool(FINBERT_ON and _try_load_finbert())
    engine = "finbert" if use_finbert else "rules"
    for it in items:
        try:
            iid = str(it.get("id") or it.get("url") or uuid.uuid4().hex)
            if iid in _NEWS_SENT_CACHE:
                sent = float(_NEWS_SENT_CACHE[iid]["sent"])
            else:
                headline = it.get("headline") or ""
                desc = it.get("description") or ""
                text = (headline + ". " + desc).strip()
                sent = _score_text_finbert(text) if use_finbert else _score_text_rules(text)
                _NEWS_SENT_CACHE[iid] = {
                    "sent": sent,
                    "engine": engine,
                    "ts": float(_ts_to_epoch(it.get("ts"))),
                }
            new = dict(it)
            new["sent"] = float(sent)
            out.append(new)
        except Exception:
            out.append(it)
    return out, engine


def _aggregate_news_score(items: list[dict]) -> tuple[float | None, str, int]:
    if not items:
        return (
            None,
            ("none" if not NEWS_SENTIMENT_ON else "rules" if not FINBERT_ON else "finbert"),
            0,
        )
    try:
        now = int(time.time())
        # filter recent
        recent: list[tuple[float, float]] = []  # (score, weight)
        half = max(1, NEWS_DECAY_HALF_MIN)
        for it in items:
            if "sent" not in it:
                continue
            ts = _ts_to_epoch(it.get("ts"))
            age_min = max(0, (now - ts) // 60)
            if age_min > max(1, NEWS_LOOKBACK_MIN):
                continue
            # exponential decay weight with half-life
            w = math.exp(-age_min / float(half))
            recent.append((float(it.get("sent", 0.0)), w))
        if not recent:
            return None, "none", 0
        num = sum(w for _, w in recent)
        val = sum(s * w for s, w in recent)
        return (
            (val / num if num > 0 else None),
            ("finbert" if FINBERT_ON and _FINBERT_PIPE else "rules"),
            len(recent),
        )
    except Exception:
        return None, "none", 0


def _render_template(tpl: str, ctx: dict[str, Any]) -> str:
    class _Safe(dict):
        def __missing__(self, key):
            return ""

    try:
        return tpl.format_map(_Safe(**ctx))
    except Exception:
        try:
            return tpl
        except Exception:
            return ""


def _fmt_qty(q: float) -> str:
    try:
        return f"{float(q):.8f}"
    except Exception:
        return "0.00000000"


def _fmt_money(v: float) -> str:
    try:
        return f"${float(v):.2f}"
    except Exception:
        return "$0.00"


def _fmt_price(v: float | None) -> str:
    if v is None:
        return "?"
    return f"${v:.2f}"


def _get_portfolio_qty_and_avg() -> tuple[float, float]:
    """Get current portfolio quantity and avg cost from STATE.
    Checks positions array first (new format), then falls back to legacy qty/avg_cost fields.
    Supports both field name formats: qty/price (API) and quantity/entry_price (ghost_state.json)
    Returns: (quantity, avg_cost)
    """
    # Try new positions array format first
    positions = STATE.get("positions", [])
    if positions:
        wolf_pos = next((p for p in positions if p.get("symbol") == WOLF), None)
        if wolf_pos:
            # Support both field name formats
            qty = float(wolf_pos.get("quantity", wolf_pos.get("qty", 0.0)))
            price = float(wolf_pos.get("entry_price", wolf_pos.get("price", 0.0)))
            return qty, price
    # Fallback to legacy fields
    return float(STATE.get("qty", 0.0)), float(STATE.get("avg_cost", 0.0))


def _build_status_card(
    price: float | None = None, provider: str | None = None, include_req: bool = True
) -> str:
    q, a = _get_portfolio_qty_and_avg()  # Use helper instead of direct STATE access
    if price is None and provider is None:
        p, _, prov = get_wolf_price()
        price, provider = p, prov
    rid = _cv_trace_id.get()
    # Derived metrics
    current = price if price is not None else a
    cash = float(STATE.get("cash", 0.0))
    market_value = round(q * current, 2)
    nav_total = round(market_value + cash, 2)
    pnl_abs = round((current - a) * q, 2)
    pnl_pct = round((((current - a) / a) * 100.0), 6) if a > 0 else 0.0
    change_pct = None
    try:
        price, prev, _ = get_wolf_price()
        if price is not None and prev and prev > 0:
            change_pct = (price - prev) / prev * 100.0
    except Exception:
        pass
    # Top headlines 2–3
    headlines: list[str] = []
    try:
        news = get_wolf_news(limit=3).get("items", [])
        for it in news[:3]:
            ts = it.get("ts")
            try:
                if isinstance(ts, (int, float)):
                    ts_str = datetime.fromtimestamp(int(ts), tz=UTC).isoformat()
                else:
                    ts_str = str(ts)
            except Exception:
                ts_str = str(ts)
            t = it.get("headline") or ""
            u = it.get("url") or ""
            if u:
                headlines.append(f"{ts_str} — {t} — {u}")
            else:
                headlines.append(f"{ts_str} — {t}")
    except Exception:
        pass
    # Build strict card
    card = (
        "📊 STATUS — WOLF (Wolfspeed)\n\n"
        "Portfolio\n"
        f"• Qty: {q:.8f}\n"
        f"• Avg Cost: ${a:.2f}\n"
        f"• Price: {('?' if price is None else f'${price:.2f}')} ({provider or 'unavailable'})\n"
        f"• Market Value: ${market_value:.2f}\n"
        f"• PnL: {pnl_abs:.2f} ({pnl_pct:.2f}%)\n\n"
        "NAV / Cash\n"
        f"• NAV: ${nav_total:.2f}\n"
        f"• Cash: ${cash:.2f}\n\n"
        "Market\n"
        f"• Change %: {0 if change_pct is None else round(change_pct, 6)}%\n"
        f"• GPS: {7.2}\n"
        f"• Signal: HOLD (mode={ALERT_MODE})\n\n"
    )
    # Add Stage 1 World Context (if available)
    try:
        if STAGE1_ENABLED:
            from core.stage1_integration import get_enhanced_context

            ctx = get_enhanced_context()
            mood = ctx.get("market_mood", {})
            world = ctx.get("world_context", {})

            if not mood.get("error"):
                regime = mood.get("market_regime", "unknown").upper()
                mood_icon = "🐂" if regime == "BULL" else "🐻" if regime == "BEAR" else "↔️"
                card += (
                    "Market Mood\n"
                    f"• Regime: {mood_icon} {regime}\n"
                    f"• Sentiment: {mood.get('sentiment', 'neutral')}\n"
                )
                if mood.get("vix_level"):
                    card += f"• VIX: {mood['vix_level']:.1f}\n"
                card += "\n"

            if not world.get("error"):
                events = world.get("trending_events", [])[:3]
                if events:
                    card += "Trending Events\n"
                    card += "• " + ", ".join([f"[{e}]" for e in events]) + "\n\n"
    except Exception as e:
        logging.debug(f"Stage 1 context unavailable in status card: {e}")

    card += "News\n" + ("\n".join(headlines) if headlines else "No headlines")
    if include_req and rid and rid != "-":
        card += f"\n\nReq: {rid}"
    return card


def _provider_in_cooldown(name: str) -> bool:
    meta = PROVIDER_BACKOFF.get(name)
    if not meta:
        return False
    return meta.get("backoff_until", 0.0) > time.time()


def _note_provider_429(name: str):
    now = time.time()
    meta = PROVIDER_BACKOFF.setdefault(name, {"last_429": 0.0, "backoff_until": 0.0, "failures": 0})
    meta["last_429"] = now
    meta["failures"] = int(meta.get("failures", 0)) + 1
    # exponential backoff with cap
    backoff = min(BACKOFF_BASE_S * (2 ** (meta["failures"] - 1)), BACKOFF_MAX_S)
    meta["backoff_until"] = now + backoff
    try:
        LOGGER.warning("provider_rate_limited", extra={"provider": name, "backoff_s": backoff})
    except Exception:
        pass


def _note_provider_success(name: str):
    meta = PROVIDER_BACKOFF.setdefault(name, {"last_429": 0.0, "backoff_until": 0.0, "failures": 0})
    # decay failures on success
    if meta.get("failures", 0) > 0:
        meta["failures"] = 0
        meta["backoff_until"] = 0.0


def _ensure_metrics_registered():
    global _H_SNAPSHOT_BUILD, _C_SNAPSHOT_FAIL, _G_UP
    global _C_ALERT_SENT, _C_ALERT_THROTTLED, _G_ALERT_HOLD, _G_ALERT_MODE
    global _H_PROVIDER_FETCH, _C_PROVIDER_FETCH
    global _H_TG_SEND, _C_TG_SEND, _H_TG_TEST, _C_TG_TEST
    global _G_ALERT_QUEUE_LEN, _C_ALERT_RETRIES, _C_RATE_LIMIT_DROPS, _G_RATE_LIMIT_TOKENS
    global _G_FINAL_SCORE, _G_WHY_NOW_COUNT
    global _C_LLM_CALLS, _C_LLM_DECISIONS, _G_LLM_CONFIDENCE
    global _C_HTTP_POOL_USED, _C_HTTP_DIRECT_USED
    global _C_AI_MEMORY_REQ, _H_AI_MEMORY_LAT
    global _G_SNAPSHOT_ASOF
    global _C_CRYPTO_PRICE_FETCH, _C_CRYPTO_PREDICT_DURATION
    global _G_CRYPTO_PREDICTION_MAPE, _G_SENTIMENT_SCORE, _G_MACRO_CONFIDENCE
    try:
        target_prefixes = (
            "ghost_cockpit_snapshot_build_seconds",
            "ghost_cockpit_snapshot_failures",
            "ghost_up",
            "ghost_alerts_sent_total",
            "ghost_alerts_throttled_total",
            "ghost_alert_hold_override",
            "ghost_alert_mode",
            "ghost_provider_fetch_seconds",
            "ghost_provider_fetch_total",
            "ghost_telegram_send_seconds",
            "ghost_telegram_send_total",
            "ghost_telegram_test_seconds",
            "ghost_telegram_test_total",
            "ghost_alert_queue_length",
            "ghost_alert_send_retries_total",
            "ghost_rate_limit_drops_total",
            "ghost_rate_limit_tokens",
            "ghost_snapshot_aso",
            "ghost_decision_final_score",
            "ghost_why_now_count",
            "ghost_llm_calls_total",
            "ghost_llm_decisions_total",
            "ghost_llm_confidence",
            "ghost_http_pool_used_total",
            "ghost_http_direct_used_total",
        )
        to_remove = []
        for collector, names in getattr(REGISTRY, "_collector_to_names", {}).items():  # type: ignore[attr-defined]
            # If any metric name exposed by this collector matches our prefixes, mark for removal
            try:
                if any(
                    any(name.startswith(p) or name == p for p in target_prefixes) for name in names
                ):
                    to_remove.append(collector)
            except Exception:
                continue
        for c in to_remove:
            try:
                REGISTRY.unregister(c)
            except Exception:
                pass
    except Exception:
        pass
    _H_SNAPSHOT_BUILD = Histogram(
        "ghost_cockpit_snapshot_build_seconds",
        "Time to build cockpit snapshot (seconds)",
    )
    _C_SNAPSHOT_FAIL = Counter(
        "ghost_cockpit_snapshot_failures",
        "Total snapshot build failures",
    )
    _G_UP = Gauge("ghost_up", "1 if API is serving, else 0")
    _C_ALERT_SENT = Counter(
        "ghost_alerts_sent_total",
        "Total alerts sent",
        labelnames=("action", "mode", "result"),
    )
    _C_ALERT_THROTTLED = Counter(
        "ghost_alerts_throttled_total",
        "Total alerts throttled (dedupe/throttle)",
    )
    _G_ALERT_HOLD = Gauge("ghost_alert_hold_override", "1 if HOLD override enabled else 0")
    _G_ALERT_MODE = Gauge(
        "ghost_alert_mode",
        "Current alert mode (one-hot per label)",
        labelnames=("mode",),
    )
    _H_PROVIDER_FETCH = Histogram(
        "ghost_provider_fetch_seconds",
        "Latency of provider price fetch",
        labelnames=("provider",),
    )
    _C_PROVIDER_FETCH = Counter(
        "ghost_provider_fetch_total",
        "Total provider fetch attempts by result",
        labelnames=("provider", "result"),
    )
    _H_TG_SEND = Histogram(
        "ghost_telegram_send_seconds",
        "Latency of Telegram sendMessage calls",
    )
    _C_TG_SEND = Counter(
        "ghost_telegram_send_total",
        "Total Telegram send attempts by result",
        labelnames=("result",),
    )
    _H_TG_TEST = Histogram(
        "ghost_telegram_test_seconds",
        "Latency of building /api/telegram/test card",
    )
    _C_TG_TEST = Counter(
        "ghost_telegram_test_total",
        "Total /api/telegram/test calls by send flag",
        labelnames=("sent",),
    )
    _G_ALERT_QUEUE_LEN = Gauge(
        "ghost_alert_queue_length",
        "Current number of alerts pending in send queue",
    )
    _C_ALERT_RETRIES = Counter(
        "ghost_alert_send_retries_total",
        "Total alert send retries across all sinks",
    )
    _C_RATE_LIMIT_DROPS = Counter(
        "ghost_rate_limit_drops_total",
        "Total write requests dropped by rate limiter",
    )
    _G_RATE_LIMIT_TOKENS = Gauge(
        "ghost_rate_limit_tokens",
        "Current available tokens in write rate limiter bucket",
    )
    _G_SNAPSHOT_ASOF = Gauge(
        "ghost_snapshot_aso",
        "Epoch timestamp of the latest snapshot served by /api/cockpit",
    )
    _G_FINAL_SCORE = Gauge(
        "ghost_decision_final_score",
        "Latest fused decision score (alpha*price + beta*news)",
    )
    _G_WHY_NOW_COUNT = Gauge(
        "ghost_why_now_count",
        "Count of 'Why now' reasons included in the last signal card",
    )
    # Macro Brain metrics
    try:
        global _G_MACRO_CONF, _C_MACRO_REFRESH
    except Exception:
        pass
    try:
        _G_MACRO_CONF = Gauge(
            "ghost_macro_confidence",
            "Macro brain confidence for last advisory (0-100)",
            labelnames=("scenario",),
        )
        _C_MACRO_REFRESH = Counter(
            "ghost_macro_refresh_total",
            "Macro brain refresh computations",
            labelnames=("result",),
        )
    except Exception:
        pass
    _C_LLM_CALLS = Counter(
        "ghost_llm_calls_total",
        "Total LLM advisory calls",
        labelnames=("endpoint", "result"),
    )
    _C_LLM_DECISIONS = Counter(
        "ghost_llm_decisions_total",
        "Total LLM decisions by action",
        labelnames=("endpoint", "action"),
    )
    _G_LLM_CONFIDENCE = Gauge(
        "ghost_llm_confidence",
        "Last LLM advisory confidence (0-100)",
        labelnames=("endpoint",),
    )
    # Ghost Prediction metrics
    Counter(
        "ghost_predict_runs_total",
        "Total prediction runs by symbol",
        labelnames=("symbol",),
    )
    Counter(
        "ghost_predict_outcomes_total",
        "Total prediction outcomes by symbol and hit status",
        labelnames=("symbol", "hit"),
    )
    Gauge(
        "ghost_predict_mae",
        "Mean Absolute Error for predictions",
        labelnames=("symbol",),
    )
    Gauge(
        "ghost_predict_mape",
        "Mean Absolute Percentage Error for predictions",
        labelnames=("symbol",),
    )
    Gauge(
        "ghost_predict_rmse",
        "Root Mean Squared Error for predictions",
        labelnames=("symbol",),
    )
    Gauge(
        "ghost_predict_confidence_avg",
        "Average prediction confidence",
        labelnames=("symbol",),
    )

    # Crypto-specific metrics
    try:
        _C_CRYPTO_PRICE_FETCH = Counter(
            "ghost_crypto_price_fetch_total",
            "Total crypto price fetches",
            labelnames=("provider", "result"),
        )
    except Exception:
        _C_CRYPTO_PRICE_FETCH = None
    try:
        _C_CRYPTO_PREDICT_DURATION = Histogram(
            "ghost_crypto_predict_seconds",
            "Crypto prediction generation duration",
            labelnames=("symbol",),
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0),
        )
    except Exception:
        _C_CRYPTO_PREDICT_DURATION = None
    try:
        _G_CRYPTO_PREDICTION_MAPE = Gauge(
            "ghost_prediction_mape",
            "Mean Absolute Percentage Error for predictions",
            labelnames=("asset_class",),
        )
    except Exception:
        _G_CRYPTO_PREDICTION_MAPE = None
    try:
        _G_SENTIMENT_SCORE = Gauge(
            "ghost_sentiment_score",
            "News sentiment score",
            labelnames=("symbol",),
        )
    except Exception:
        _G_SENTIMENT_SCORE = None
    try:
        _G_MACRO_CONFIDENCE = Gauge(
            "ghost_macro_confidence",
            "Macro scenario confidence",
            labelnames=("scenario",),
        )
    except Exception:
        _G_MACRO_CONFIDENCE = None

    _C_HTTP_POOL_USED = Counter(
        "ghost_http_pool_used_total",
        "Total HTTP requests performed using pooled sessions",
        labelnames=("host",),
    )
    _C_HTTP_DIRECT_USED = Counter(
        "ghost_http_direct_used_total",
        "Total HTTP requests performed using direct requests.*",
        labelnames=("host",),
    )
    _C_AI_MEMORY_REQ = Counter(
        "ghost_ai_memory_requests_total",
        "AI memory endpoint requests",
        labelnames=("endpoint", "result"),
    )
    _H_AI_MEMORY_LAT = Histogram(
        "ghost_ai_memory_latency_seconds",
        "Latency for AI memory endpoints",
        labelnames=("endpoint",),
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0),
    )


def _ensure_startup_dirs():
    # Ensure data directory and PROMETHEUS_MULTIPROC_DIR exist when configured
    try:
        data_dir = os.path.dirname(WOLF_STATE_FILE) or "data"
        if data_dir and not os.path.exists(data_dir):
            os.makedirs(data_dir, exist_ok=True)
    except Exception:
        pass
    # Ensure sqlite directory exists
    try:
        _ensure_dir_for_file(WOLF_SQLITE_PATH)
    except Exception:
        pass
    # Ensure AI DB directory exists
    try:
        _ensure_ai_dir()
    except Exception:
        pass
    try:
        mp_dir = os.getenv("PROMETHEUS_MULTIPROC_DIR", "").strip()
        if mp_dir and not os.path.exists(mp_dir):
            os.makedirs(mp_dir, exist_ok=True)
    except Exception:
        pass

    # Create /tmp/ghost_prom directory for metrics persistence (Railway fix)
    try:
        prom_dir = "/tmp/ghost_prom"
        if not os.path.exists(prom_dir):
            os.makedirs(prom_dir, exist_ok=True)
    except Exception:
        pass


def _ensure_ai_dir():
    try:
        # Ensure base AI data directory exists
        if AI_DATA_DIR and not os.path.exists(AI_DATA_DIR):
            os.makedirs(AI_DATA_DIR, exist_ok=True)
    except Exception:
        pass
    try:
        # Ensure directory for AI memory sqlite file
        _ensure_dir_for_file(AI_MEMORY_DB_PATH)
    except Exception:
        pass


def _set_mode_gauge():
    try:
        if _G_ALERT_MODE is None:
            return
        for m in ("fixed", "band", "trailing"):
            _G_ALERT_MODE.labels(mode=m).set(1 if ALERT_MODE == m else 0)
    except Exception:
        pass


def _get_host(url: str) -> str:
    try:
        return urlparse(url).netloc or ""
    except Exception:
        return ""


def _forecast_db_conn():
    """Use the primary sqlite at WOLF_SQLITE_PATH; ensure tables exist."""
    try:
        import sqlite3

        _ensure_dir_for_file(WOLF_SQLITE_PATH)
        conn = sqlite3.connect(WOLF_SQLITE_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS forecast_runs (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              symbol TEXT,
              as_of_ts INTEGER,
              horizon_h INTEGER,
              y0_price REAL,
              path_mid TEXT,
              path_lo TEXT,
              path_hi TEXT,
              dt_minutes INTEGER,
              params TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS realized_prices (
              ts INTEGER,
              symbol TEXT,
              price REAL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS forecast_scores (
              forecast_id INTEGER PRIMARY KEY,
              scored_through_ts INTEGER,
              map REAL,
              rmse REAL,
              bias REAL,
              hit_peak INTEGER,
              notes TEXT
            )
            """
        )
        # Optional rolling stats per symbol for calibration
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS model_stats (
              symbol TEXT PRIMARY KEY,
              mape_7 REAL,
              mape_30 REAL,
              bias_7 REAL,
              bias_30 REAL,
              rmse_7 REAL,
              rmse_30 REAL,
              updated_ts INTEGER
            )
            """
        )
        # Add indexes for better query performance
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_forecast_runs_symbol_time
            ON forecast_runs(symbol, as_of_ts DESC)
        """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_realized_prices_symbol_ts
            ON realized_prices(symbol, ts DESC)
        """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_forecast_scores_metrics
            ON forecast_scores(map ASC, rmse ASC)
        """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_model_stats_performance
            ON model_stats(mape_7 ASC, mape_30 ASC)
        """
        )
        # Additional indexes for performance (audit findings)
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_forecast_actuals_forecast_time
            ON forecast_actuals(forecast_id, t ASC)
        """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_realized_prices_symbol_ts_asc
            ON realized_prices(symbol, ts ASC)
        """
        )
        conn.commit()
        return conn
    except Exception:
        return None


def _record_forecast(
    symbol: str,
    as_of_ts: int,
    y0: float,
    mid: list[float],
    lo: list[float],
    hi: list[float],
    dt_minutes: int = 60,
    params: dict | None = None,
) -> int | None:
    conn = _forecast_db_conn()
    if conn is None:
        return None
    try:
        import json as _json

        cur = conn.cursor()
        cur.execute(
            "INSERT INTO forecast_runs(symbol,as_of_ts,horizon_h,y0_price,path_mid,path_lo,path_hi,dt_minutes,params) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                str(symbol),
                int(as_of_ts),
                int(max(1, int(len(mid) * dt_minutes / 60))),
                float(y0),
                _json.dumps(mid),
                _json.dumps(lo),
                _json.dumps(hi),
                int(dt_minutes),
                _json.dumps(params or {}),
            ),
        )
        conn.commit()
        lastrowid = cur.lastrowid
        return int(lastrowid) if lastrowid is not None else None
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _record_price_tick(symbol: str, price: float, ts: int | None = None) -> bool:
    conn = _forecast_db_conn()
    if conn is None:
        return False
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO realized_prices(ts,symbol,price) VALUES(?,?,?)",
            (int(ts or time.time()), str(symbol), float(price)),
        )
        conn.commit()
        return True
    except Exception:
        try:
            conn.rollback()
        except Exception:
            pass
        return False
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _get_price_history_cached(symbol: str, days: int = 5) -> list[dict]:
    """Return lightweight recent price history from realized_prices for momentum.
    Fallback to empty list on errors. Shape: [{"ts": int, "price": float}, ...]
    """
    try:
        since = int(time.time()) - int(days * 86400)
        rows = _realized_since(str(symbol), since)
        return [{"ts": ts, "price": float(px)} for (ts, px) in rows]
    except Exception:
        return []


def _compute_forecast_scores(
    f_row: dict, actual: list[tuple[int, float]]
) -> tuple[float | None, float | None, float | None, bool]:
    """Return (MAP, RMSE, bias_pct, hit_peak)."""
    try:
        import json as _json

        mid = [float(x) for x in (_json.loads(f_row.get("path_mid") or "[]") or [])]
        as_of_ts = int(f_row.get("as_of_ts") or 0)
        dtm = int(f_row.get("dt_minutes") or 60)
        if not mid:
            return None, None, None, False
        # Build predicted timeline (ts, pred)
        pred: list[tuple[int, float]] = [(as_of_ts + i * dtm * 60, mid[i]) for i in range(len(mid))]
        # Index actuals by ts for nearest lookup
        actual_sorted = sorted(actual, key=lambda t: t[0])
        ai = 0
        pairs: list[tuple[float, float]] = []  # (pred, real)
        tol = dtm * 60 // 2 or 30
        for ts_p, vp in pred:
            # advance to nearest actual
            best = None
            while ai < len(actual_sorted):
                tsa, va = actual_sorted[ai]
                if tsa <= ts_p + tol:
                    if best is None or abs(tsa - ts_p) < abs(best[0] - ts_p):
                        best = (tsa, va)
                    ai += 1
                else:
                    break
            if best is None:
                # try previous point
                prev_idx = max(0, ai - 1)
                if actual_sorted:
                    cand = actual_sorted[prev_idx]
                    if abs(cand[0] - ts_p) <= tol:
                        best = cand
            if best is not None:
                pairs.append((vp, float(best[1])))
        if not pairs:
            return None, None, None, False
        # Metrics
        abs_pct = []
        sq = []
        bias_terms = []
        for vp, vr in pairs:
            if vr == 0:
                continue
            abs_pct.append(abs(vp - vr) / abs(vr))
            d = vp - vr
            sq.append(d * d)
            bias_terms.append(d / vr)
        map = (sum(abs_pct) / len(abs_pct) * 100.0) if abs_pct else None
        rmse = math.sqrt(sum(sq) / len(sq)) if sq else None
        bias_pct = (sum(bias_terms) / len(bias_terms) * 100.0) if bias_terms else None
        # Peak hit: compare argmax indices within 2 steps
        try:
            pred_idx = int(sorted(range(len(mid)), key=lambda i: mid[i])[-1])
            pred_ts = as_of_ts + pred_idx * dtm * 60
            if actual_sorted:
                a_vals = [v for _, v in actual_sorted]
                a_idx = int(sorted(range(len(a_vals)), key=lambda i: a_vals[i])[-1])
                a_ts = actual_sorted[a_idx][0]
                hit_peak = abs(a_ts - pred_ts) <= 2 * dtm * 60
            else:
                hit_peak = False
        except Exception:
            hit_peak = False
        return map, rmse, bias_pct, hit_peak
    except Exception:
        return None, None, None, False


def _latest_forecast(symbol: str) -> dict | None:
    conn = _forecast_db_conn()
    if conn is None:
        return None
    try:
        conn.row_factory = __import__("sqlite3").Row  # type: ignore
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM forecast_runs WHERE symbol=? ORDER BY as_of_ts DESC, id DESC LIMIT 1",
            (str(symbol),),
        )
        row = cur.fetchone()
        return dict(row) if row else None
    except Exception:
        return None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def _realized_since(symbol: str, since_ts: int) -> list[tuple[int, float]]:
    conn = _forecast_db_conn()
    if conn is None:
        return []
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT ts, price FROM realized_prices WHERE symbol=? AND ts>=? ORDER BY ts ASC",
            (str(symbol), int(since_ts)),
        )
        return [(int(ts), float(price)) for (ts, price) in cur.fetchall()]
    except Exception:
        return []
    finally:
        try:
            conn.close()
        except Exception:
            pass


def run_single_prediction(symbol: str) -> dict[str, Any]:
    """
    Core synchronous prediction function with turbo provider architecture.
    
    This function is the HEART OF THE GHOST TURBO SURGERY.
    - Hard 4 second budget (3s price + 1s features)
    - Hard 8 second timeout (fast-fail to prevent hanging)
    - Uses turbo_stock_price/turbo_crypto_price with fast-fail
    - Always returns dict (never raises exceptions)
    - Returns structured error on any failure
    
    Args:
        symbol: Trading symbol (e.g., "PACS", "BTC")
    
    Returns:
        {
            "ok": bool,
            "prediction_id": int or None,
            "symbol": str,
            "direction": str,
            "confidence": float,
            "current_price": float or None,
            "feature_count": int,
            "available_count": int,
            "duration_ms": int,
            "error": str or None
        }
    """
    start = time.monotonic()
    BUDGET_S = 4.0  # Total budget: 3s price + 1s features
    
    # Validate symbol first (before any expensive operations)
    symbol = symbol.upper().strip() if symbol else ""
    if not symbol:
        return {
            "ok": False,
            "symbol": "UNKNOWN",
            "direction": "ERROR",
            "confidence": 0.0,
            "current_price": None,
            "feature_count": 0,
            "available_count": 0,
            "duration_ms": 0,
            "error": "symbol required"
        }
    
    # =========================================================================
    # MONEY GAME: No more V2 whitelist filter - all symbols compete!
    # Predictions are made for ALL symbols, rankings determine TOP 10
    # =========================================================================
    
    # Wrap core logic in try/except for safety
    try:
        # Detect asset type (crypto vs stock)
        is_crypto = symbol in HUNTER_CRYPTO_SYMBOLS or _classify_symbol_category(symbol) == "crypto"
        
        # =========================================================================
        # STOCK ENGINE ROUTING (Jan 26, 2026)
        # Route stocks to the specialized stock engine for better accuracy
        # Crypto continues using the original turbo engine
        # =========================================================================
        use_stock_engine = os.getenv("USE_STOCK_ENGINE", "true").lower() == "true"
        
        if not is_crypto and use_stock_engine:
            try:
                import asyncio
                from core.stock_engine import run_stock_prediction
                
                LOGGER.info(f"[{symbol}] 🏛️ Routing to Stock Engine (24h horizon, 2% target)")
                
                # Run stock prediction (async) 
                loop = asyncio.new_event_loop()
                try:
                    stock_result = loop.run_until_complete(run_stock_prediction(symbol))
                finally:
                    loop.close()
                
                duration_ms = int((time.monotonic() - start) * 1000)
                
                # Convert stock engine result to standard format
                se_direction = stock_result.get("direction", "HOLD")
                se_confidence = stock_result.get("confidence", 0.0)
                se_entry_price = stock_result.get("entry_price")
                se_horizon = stock_result.get("horizon_hours", 24)
                
                # ================================================================
                # 🧠 INTELLIGENCE HUB — Apply 20-system intelligence to stock predictions
                # ================================================================
                hub_meta = {}
                try:
                    from core.intelligence_hub import get_intelligence_hub
                    _hub = get_intelligence_hub()
                    # Fetch price history for regime detection
                    _stock_price_history = []
                    try:
                        from core.ghost_scout import GhostScout as _StockGS
                        _stock_gs = _StockGS()
                        _stock_price_history = _stock_gs._fetch_price_history(symbol, "stock") or []
                    except Exception:
                        pass

                    _hub_report = _hub.analyze(
                        symbol=symbol,
                        direction=se_direction,
                        confidence=se_confidence,
                        entry_price=se_entry_price or 0,
                        asset_type="stock",
                        price_history=_stock_price_history,
                    )
                    hub_meta = {
                        "intel_active_systems": _hub_report.active_systems,
                        "intel_total_systems": _hub_report.total_systems,
                        "intel_news_risk": _hub_report.news_risk,
                        "intel_direction_adj": _hub_report.direction_adjustment,
                        "intel_confidence_adj": _hub_report.confidence_adjustment,
                        "intel_trust_boost": _hub_report.trust_boost,
                        "market_regime": _hub_report.regime_info.get("regime", "UNKNOWN"),
                    }

                    if _hub_report.should_block:
                        LOGGER.info(f"🛑 [HUB] {symbol}: Stock prediction BLOCKED — {_hub_report.block_reason}")
                        duration_ms = int((time.monotonic() - start) * 1000)
                        return {"ok": False, "symbol": symbol, "direction": "BLOCKED",
                                "confidence": 0.0, "current_price": se_entry_price,
                                "duration_ms": duration_ms, "error": _hub_report.block_reason}

                    # Apply direction flip
                    if _hub_report.direction_adjustment == "FLIP":
                        old_dir = se_direction
                        se_direction = "DOWN" if se_direction == "UP" else "UP"
                        LOGGER.info(f"🔄 [HUB] {symbol}: Stock direction FLIPPED {old_dir} → {se_direction}")

                    # Apply confidence adjustment
                    old_conf = se_confidence
                    se_confidence += _hub_report.confidence_adjustment + _hub_report.trust_boost
                    se_confidence = max(0.10, min(0.85, se_confidence))  # Keep engine's 0.85 cap — was 0.92, inflating all predictions
                    if abs(se_confidence - old_conf) > 0.01:
                        LOGGER.info(f"🧠 [HUB] {symbol}: Stock conf {old_conf:.2f} → {se_confidence:.2f} "
                                    f"(systems={_hub_report.active_systems}/{_hub_report.total_systems})")

                    # Apply dynamic exits
                    if _hub_report.exit_levels:
                        stock_result["target_price"] = _hub_report.exit_levels.get("target_price", stock_result.get("target_price"))
                        stock_result["stop_loss"] = _hub_report.exit_levels.get("stop_loss", stock_result.get("stop_loss"))
                except Exception as _hub_err:
                    LOGGER.warning(f"🧠 [HUB] Stock engine hub error for {symbol}: {_hub_err}")

                # ================================================================
                # WIRE STOCK ENGINE → _LATEST_PREDICTIONS (cockpit + Telegram)
                # Without this, Stock Engine predictions are invisible to:
                #   - /api/cockpit dashboard
                #   - Telegram TOP 10 notifications
                #   - Ghost Score calculations
                # ================================================================
                se_action = "BUY" if se_direction == "UP" else "SELL" if se_direction == "DOWN" else "HOLD"
                
                # FIX (Mar 1, 2026): Don't store HOLD predictions — not actionable
                if se_direction == "HOLD":
                    duration_ms_held = int((time.monotonic() - start) * 1000)
                    LOGGER.info(f"[{symbol}] ⏸️ Stock Engine → HOLD, skipping storage (conf={se_confidence:.2f})")
                    return {
                        "ok": False, "symbol": symbol, "direction": "HOLD",
                        "confidence": se_confidence, "current_price": se_entry_price,
                        "duration_ms": duration_ms_held, "engine": "stock_v2",
                        "error": "HOLD — not actionable",
                    }

                # ================================================================
                # ACCURACY AUTOPILOT GATE (Mar 13, 2026)
                # Skip predictions when system is paused or confidence too low.
                # This prevents garbage predictions from entering the pipeline.
                # ================================================================
                try:
                    from core.accuracy_autopilot import should_skip_prediction as _ap_skip
                    _ap_should_skip, _ap_reason = _ap_skip(se_confidence)
                    if _ap_should_skip:
                        duration_ms_skip = int((time.monotonic() - start) * 1000)
                        LOGGER.info(f"[{symbol}] 🛑 Autopilot skip: {_ap_reason}")
                        return {
                            "ok": False, "symbol": symbol, "direction": se_direction,
                            "confidence": se_confidence, "current_price": se_entry_price,
                            "duration_ms": duration_ms_skip, "engine": "stock_v2",
                            "error": f"autopilot: {_ap_reason}",
                        }
                except Exception as _ap_err:
                    LOGGER.debug(f"[{symbol}] Autopilot check failed: {_ap_err}")
                
                # ================================================================
                # V3 DIRECTION OVERRIDE for Stock Engine (Mar 6, 2026)
                # Apply BEFORE _LATEST_PREDICTIONS and ghost_predictions writes
                # so evaluator, cockpit, and paper trades all use correct direction.
                # ================================================================
                from config.symbols import V3_VALIDATED_STRATEGIES as _SE_V3_EARLY
                _se_v3_early = _SE_V3_EARLY.get(symbol.upper())
                _se_raw_dir = None  # Track raw direction before V3 override
                if _se_v3_early and _se_v3_early.direction_override:
                    _se_raw_dir = se_direction
                    if _se_v3_early.direction_override == 'flip':
                        se_direction = 'DOWN' if se_direction == 'UP' else 'UP'
                    else:
                        se_direction = _se_v3_early.direction_override
                    if se_direction != _se_raw_dir:
                        LOGGER.info(
                            f"[{symbol}] 🔄 Stock Engine V3 EARLY OVERRIDE: {_se_raw_dir} → {se_direction} "
                            f"(strategy: {_se_v3_early.strategy})"
                        )
                # Recalculate action after potential direction override
                se_action = "BUY" if se_direction == "UP" else "SELL" if se_direction == "DOWN" else "HOLD"
                
                # FIX (Step 8): Detect crypto symbols going through stock engine
                from core.asset_classification import is_crypto_symbol as _is_crypto_sym_stock
                
                with _LATEST_PREDICTIONS_LOCK:
                    _LATEST_PREDICTIONS[symbol] = {
                        "prediction_id": None,  # Updated after PG write below
                        "symbol": symbol,
                        "run_at": time.time(),
                        "confidence": se_confidence,
                        "direction": se_direction,
                        "action": se_action,
                        "horizon_h": se_horizon,
                        "provider": "stock_engine_v2",
                        "price": se_entry_price,
                        "price_at_prediction": se_entry_price,
                        "market": "crypto" if _is_crypto_sym_stock(symbol) else "stock",  # FIX: was hardcoded "stock"
                        "engine": "stock_v2",
                        "confirmations": stock_result.get("confirmations", 0),
                        "intel_applied": True,
                        "gates_passed": stock_result.get("gates_passed", []),
                        "reasons": stock_result.get("reasons", []),
                        "should_predict": se_direction != "HOLD",
                        # Intelligence Hub metadata
                        **hub_meta,
                    }
                LOGGER.info(f"[{symbol}] 🏛️ Stock Engine → cockpit: {se_direction} {se_confidence:.0%} ({stock_result.get('confirmations', 0)} confirmations)")
                
                # ================================================================
                # PREDICTION GATE (Step 3): Kill switch + confidence floor + rate limit
                # ================================================================
                _se_gate_ok, _se_gate_reason = should_create_prediction(symbol, se_confidence)
                if not _se_gate_ok:
                    LOGGER.info(f"[{symbol}] 🚫 Stock Engine prediction blocked: {_se_gate_reason}")

                # ================================================================
                # WIRE STOCK ENGINE → ghost_predictions DB (touch calibration data)
                # ================================================================
                if _se_gate_ok:
                 try:
                    import sqlite3 as _se_sqlite3
                    conn = _se_sqlite3.connect(WOLF_SQLITE_PATH)
                    # Ensure table exists (ephemeral storage loses it on redeploy)
                    conn.execute("""CREATE TABLE IF NOT EXISTS ghost_predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        predicted_at INTEGER NOT NULL,
                        check_at INTEGER NOT NULL,
                        predicted_price REAL,
                        predicted_direction TEXT,
                        predicted_pct REAL,
                        confidence REAL,
                        timeframe_hours INTEGER,
                        current_price REAL,
                        target_price REAL,
                        stage5_ok INTEGER,
                        stage6_ok INTEGER,
                        gate TEXT,
                        checked INTEGER DEFAULT 0,
                        UNIQUE(symbol, predicted_at)
                    )""")
                    se_predicted_price = stock_result.get("target_price") or se_entry_price
                    se_predicted_pct = ((se_predicted_price - se_entry_price) / se_entry_price * 100) if se_entry_price else 0.0
                    conn.execute("""
                        INSERT INTO ghost_predictions (
                            symbol, predicted_at, check_at, predicted_price,
                            predicted_direction, predicted_pct, confidence, timeframe_hours,
                            current_price, target_price, stage5_ok, stage6_ok, gate,
                            checked
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        int(time.time()),
                        int(time.time() + se_horizon * 3600),
                        se_predicted_price,
                        se_direction,
                        float(se_predicted_pct),
                        se_confidence,
                        se_horizon,
                        se_entry_price,
                        se_predicted_price,
                        1 if stock_result.get("is_actionable") else 0,
                        1 if se_confidence >= 0.6 else 0,
                        "STOCK_ENGINE",
                        0,
                    ))
                    conn.commit()
                    conn.close()
                    LOGGER.info(f"[{symbol}] 🏛️ Stock Engine → ghost_predictions DB ✅")
                 except Exception as db_err:
                    LOGGER.warning(f"[{symbol}] Stock Engine ghost_predictions write failed (non-fatal): {db_err}")

                # ================================================================
                # ALSO write stock predictions to PostgreSQL (Feb 24, 2026)
                # Evaluator now reads from PostgreSQL exclusively.
                # ================================================================
                _se_pg_prediction_id = None
                if _se_gate_ok:
                 try:
                    from core.db_pool import get_sync_connection as _se_get_conn
                    _se_pred_price = stock_result.get("target_price") or se_entry_price
                    _se_pred_pct = ((float(_se_pred_price) - se_entry_price) / se_entry_price * 100) if se_entry_price else 0.0

                    # ── GHOST LEARNING BRAIN — DISABLED (Step 6, Mar 18 2026) ──
                    # Root cause of double-flip: Brain v3 (ghost_brain.py) inverts
                    # at <38%, then Learning Brain inverts AGAIN at <35% → direction
                    # flips back to the original wrong answer. The kill switch
                    # (Step 3, should_create_prediction) already handles chronically
                    # bad symbols by blocking them entirely. Inversion is the wrong
                    # approach — it creates oscillation, not convergence.
                    _was_inverted = False

                    # ── DIRECTION CONSISTENCY GUARD (Mar 10, 2026) ──────────────
                    # After the brain's decision, ensure direction and target agree.
                    # The evaluator scores by comparing predicted_direction vs actual.
                    # FIX (Mar 12, 2026): Also update se_direction and _LATEST_PREDICTIONS
                    # so Telegram/paper trades get corrected direction.
                    _se_dir_for_pg = se_direction
                    if se_entry_price and _se_pred_price:
                        if float(_se_pred_price) > se_entry_price and se_direction == "DOWN":
                            _se_dir_for_pg = "UP"
                            se_direction = "UP"
                            with _LATEST_PREDICTIONS_LOCK:
                                if symbol in _LATEST_PREDICTIONS:
                                    _LATEST_PREDICTIONS[symbol]["direction"] = "UP"
                                    _LATEST_PREDICTIONS[symbol]["action"] = "BUY"
                            LOGGER.warning(f"[{symbol}] ⚠️ Direction consistency fix: target {_se_pred_price} > entry {se_entry_price} but dir was DOWN → corrected to UP")
                        elif float(_se_pred_price) < se_entry_price and se_direction == "UP":
                            _se_dir_for_pg = "DOWN"
                            se_direction = "DOWN"
                            with _LATEST_PREDICTIONS_LOCK:
                                if symbol in _LATEST_PREDICTIONS:
                                    _LATEST_PREDICTIONS[symbol]["direction"] = "DOWN"
                                    _LATEST_PREDICTIONS[symbol]["action"] = "SELL"
                            LOGGER.warning(f"[{symbol}] ⚠️ Direction consistency fix: target {_se_pred_price} < entry {se_entry_price} but dir was UP → corrected to DOWN")

                    with _se_get_conn() as _se_pg_conn:
                        _se_pg_cur = _se_pg_conn.cursor()
                        _se_pg_cur.execute("""
                            INSERT INTO ghost_predictions (
                                symbol, predicted_at, check_at, predicted_price,
                                predicted_direction, predicted_pct, confidence, timeframe_hours,
                                current_price, target_price, gate, checked
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (symbol, predicted_at) DO NOTHING
                            RETURNING id
                        """, (
                            symbol,
                            int(time.time()),
                            int(time.time() + se_horizon * 3600),
                            float(_se_pred_price),
                            _se_dir_for_pg,
                            float(_se_pred_pct),
                            se_confidence,
                            se_horizon,
                            se_entry_price,
                            float(_se_pred_price),
                            "STOCK_ENGINE",
                            0,
                        ))
                        _se_row = _se_pg_cur.fetchone()
                        _se_pg_prediction_id = _se_row[0] if _se_row else None
                        _se_pg_conn.commit()
                    LOGGER.info(f"[{symbol}] 🏛️ Stock Engine → PostgreSQL ghost_predictions (id={_se_pg_prediction_id}) ✅")
                 except Exception as _se_pg_err:
                    LOGGER.warning(f"[{symbol}] Stock Engine PostgreSQL write failed (non-fatal): {_se_pg_err}")
                
                # Update in-memory cache with PG prediction_id now that we have it
                if _se_pg_prediction_id is not None:
                    with _LATEST_PREDICTIONS_LOCK:
                        if symbol in _LATEST_PREDICTIONS:
                            _LATEST_PREDICTIONS[symbol]["prediction_id"] = _se_pg_prediction_id
                
                # ================================================================
                # PAPER TRADE LOGGING for Stock Engine (Mar 6, 2026)
                # Stock Engine returns BEFORE reaching turbo engine paper trade
                # logging at L10506. Without this, stock symbols (PANW, NET,
                # FTNT, DDOG, T, BMBL, XPO) NEVER get paper trades and are
                # invisible to accuracy tracking.
                # paper_tracker.log_signal() has ALL gates internally:
                #   edge whitelist, price sanity, HOLD zone, confidence gate,
                #   trading controls, dedup
                # ================================================================
                try:
                    from core.paper_tracker import get_paper_tracker as _se_get_pt
                    _se_pt = _se_get_pt()
                    # V3 direction override already applied at early override block
                    # se_direction is already overridden, just use it directly
                    _se_v3_config = _se_v3_early  # From early override block
                    _se_v3_validated = _se_v3_config is not None if _se_v3_early else False
                    _se_v3_strategy = _se_v3_config.strategy if _se_v3_config else None
                    _se_v3_hold = _se_v3_config.hold_hours if _se_v3_config else None
                    _se_v3_wr = _se_v3_config.backtest_win_rate if _se_v3_config else None
                    _se_v3_inverse = (_se_v3_config.strategy == 'ghost_inverse') if _se_v3_config else False
                    
                    _se_paper_id = _se_pt.log_signal(
                        cascade_id=f"stock_{symbol}_{int(time.time())}",
                        symbol=symbol,
                        signal_direction=se_direction,  # Already V3-overridden
                        signal_confidence=se_confidence,
                        entry_price=se_entry_price,
                        entry_time=datetime.utcnow().isoformat(),
                        position_size=1000.0,
                        stop_loss_pct=0.05,
                        take_profit_pct=abs(float(stock_result.get("expected_move_pct", 3.0) or 3.0)) / 100.0,
                        v3_validated=_se_v3_validated,
                        v3_strategy=_se_v3_strategy,
                        v3_hold_hours=_se_v3_hold,
                        v3_backtest_win_rate=_se_v3_wr,
                        v3_is_inverse=_se_v3_inverse,
                        v3_original_direction=_se_raw_dir,
                    )
                    if _se_paper_id:
                        LOGGER.info(f"[{symbol}] 📝 Stock Engine paper trade logged: {_se_paper_id}")
                except Exception as _se_pt_err:
                    LOGGER.warning(f"[{symbol}] Stock Engine paper trade failed (non-fatal): {_se_pt_err}")
                
                return {
                    "ok": stock_result.get("is_actionable", False) or se_direction != "HOLD",
                    "prediction_id": _se_pg_prediction_id,
                    "symbol": symbol,
                    "direction": se_direction,
                    "confidence": se_confidence,
                    "current_price": se_entry_price,
                    "target_price": stock_result.get("target_price"),
                    "stop_loss": stock_result.get("stop_loss"),
                    "feature_count": len(stock_result.get("gates_passed", [])) + len(stock_result.get("gates_failed", [])),
                    "available_count": len(stock_result.get("gates_passed", [])),
                    "duration_ms": duration_ms,
                    "engine": "stock_v2",
                    "horizon_hours": se_horizon,
                    "confirmations": stock_result.get("confirmations", 0),
                    "min_confirmations": stock_result.get("min_confirmations", 3),
                    "gates_passed": stock_result.get("gates_passed", []),
                    "gates_failed": stock_result.get("gates_failed", []),
                    "reasons": stock_result.get("reasons", []),
                    "is_actionable": stock_result.get("is_actionable", False),
                    "intel_applied": True,
                    "error": None if stock_result.get("is_actionable") or se_direction != "HOLD" else "Stock gates blocked prediction"
                }
            except Exception as e:
                LOGGER.warning(f"[{symbol}] Stock engine failed, falling back to turbo engine: {e}")
                # Fall through to original turbo engine
        
        # Check market hours for stocks (Issue #3 fix)
        if not is_crypto:
            is_market_open, next_open_ts = _is_market_open_now()
            if not is_market_open:
                LOGGER.warning(
                    f"[{symbol}] Stock market closed, prediction may use stale data (next open: {next_open_ts})",
                    extra={
                        "symbol": symbol,
                        "market_closed": True,
                        "next_open_utc": next_open_ts,
                    }
                )
        
        # TURBO PRICE FETCH: Use fast-fail provider with 3s budget
        price_budget_s = 3.0
        if is_crypto:
            # Use turbo crypto provider
            price_result = turbo_crypto_price(symbol, max_budget_s=price_budget_s)
        else:
            # Use turbo stock provider
            price_result = turbo_stock_price(symbol, max_budget_s=price_budget_s)
        
        # Check if price fetch succeeded
        if not price_result.get("ok") or not price_result.get("price"):
            duration_ms = int((time.monotonic() - start) * 1000)
            error_msg = price_result.get("error", "Unable to fetch price")
            LOGGER.warning(
                f"[{symbol}] Price fetch failed: {error_msg}",
                extra={
                    "symbol": symbol,
                    "duration_ms": duration_ms,
                    "turbo_logs": price_result.get("logs", []),
                    "provider": price_result.get("provider")
                }
            )
            return {
                "ok": False,
                "symbol": symbol,
                "direction": "ERROR",
                "confidence": 0.0,
                "current_price": None,
                "feature_count": 0,
                "available_count": 0,
                "duration_ms": duration_ms,
                "error": error_msg
            }
        
        # Extract price and metadata from turbo result
        current_price = float(price_result["price"])
        price_provider = price_result.get("provider", "unknown")
        price_duration_s = price_result.get("duration_s", 0)
        
        LOGGER.info(
            f"[{symbol}] Turbo price: ${current_price:.2f} via {price_provider} ({price_duration_s*1000:.0f}ms)",
            extra={
                "symbol": symbol,
                "price": current_price,
                "provider": price_provider,
                "duration_ms": int(price_duration_s * 1000),
                "cached": price_result.get("cached", False)
            }
        )

        run_at = time.time()
        
        # Check remaining budget for feature extraction
        elapsed = time.monotonic() - start
        remaining = BUDGET_S - elapsed
        
        if remaining <= 0.5:  # Need at least 500ms for features
            duration_ms = int((time.monotonic() - start) * 1000)
            LOGGER.warning(
                f"[{symbol}] Budget exhausted after price fetch ({elapsed:.2f}s)",
                extra={"symbol": symbol, "duration_ms": duration_ms}
            )
            return {
                "ok": False,
                "symbol": symbol,
                "direction": "ERROR",
                "confidence": 0.0,
                "current_price": current_price,
                "feature_count": 0,
                "available_count": 0,
                "duration_ms": duration_ms,
                "error": f"Timeout: price fetch took {elapsed:.1f}s (budget: {BUDGET_S}s)"
            }

        # STEP 3: Extract features from all 6 data pillars (with remaining budget)
        from core.data_pillars.feature_orchestrator import get_feature_orchestrator

        orchestrator = get_feature_orchestrator()
        feature_data = orchestrator.get_all_features(symbol, period=90)

        # Log feature extraction results with ENHANCED DIAGNOSTICS
        feature_avail_pct = (feature_data['available_count'] / feature_data['feature_count'] * 100) if feature_data['feature_count'] > 0 else 0
        LOGGER.info(
            f"[{symbol}] Feature Extraction Complete",
            extra={
                "symbol": symbol,
                "available_features": feature_data['available_count'],
                "total_features": feature_data['feature_count'],
                "availability_pct": round(feature_avail_pct, 1),
                "execution_ms": round(feature_data['execution_time_ms'], 1),
                "pillar_breakdown": feature_data.get('feature_availability', {}),
                "live_price": current_price,
                "price_provider": price_provider,
            }
        )

        features = feature_data.get("features", {})

        # Diagnose feature extraction quality (backward compat with Ghost Hunter)
        # Map orchestrator features to diagnostic function expected fields
        rsi_value = features.get("RSI_14")
        macd_value = features.get("MACD_HISTOGRAM", 0)
        volume_spike = features.get("VOLUME_SPIKE", 0)
        volatility = features.get("VOLATILITY_20D", 0)
        
        feature_status = diagnose_features(
            symbol=symbol,
            price_data={
                "price": current_price,
                "timestamp": run_at,
                "provider": price_provider
            },
            volume_data={
                "volume": volume_spike if volume_spike is not None else 0, 
                "avg_volume": volatility if volatility is not None else 0
            },
            momentum_data={
                "momentum_score": rsi_value if rsi_value is not None else 50.0, 
                "trend": "up" if macd_value and macd_value > 0 else "down"
            },
            context_data={
                "market_regime": features.get("MARKET_REGIME", "neutral"), 
                "sector_health": 0.5
            },
            sentiment_data={
                "sentiment_score": features.get("NEWS_SENTIMENT_SCORE", 0), 
                "news_count": features.get("NEWS_COUNT_24H", 0)
            }
        )

        # Log feature status for diagnostics
        LOGGER.info(f"[{symbol}] Feature status", extra={"feature_status": feature_status.to_dict()})

        # =====================================================================
        # STAGE 1 CONTEXT INJECTION (Dec 30, 2025)
        # Wire world_context and market_mood into prediction features
        # =====================================================================
        stage1_boost = 0.0
        stage1_signal = "NEUTRAL"
        try:
            if STAGE1_ENABLED:
                from core.stage1_integration import get_enhanced_context
                
                stage1_ctx = get_enhanced_context(hours=24, min_relevance=0.3)
                market_mood = stage1_ctx.get("market_mood", {})
                world_context = stage1_ctx.get("world_context", {})
                
                # Extract market regime for direction bias
                regime = market_mood.get("market_regime", "neutral").upper()
                sentiment = market_mood.get("sentiment", "neutral")
                vix = market_mood.get("vix_level", 20)
                
                # Add to features for model consumption
                features["MARKET_REGIME_STAGE1"] = regime
                features["MARKET_SENTIMENT_STAGE1"] = sentiment
                features["VIX_LEVEL"] = vix
                
                # Calculate Stage 1 signal
                if regime == "BULL" and sentiment in ["bullish", "very_bullish"]:
                    stage1_signal = "UP"
                    stage1_boost = 0.05  # +5% confidence for bull market alignment
                elif regime == "BEAR" and sentiment in ["bearish", "very_bearish"]:
                    stage1_signal = "DOWN"
                    stage1_boost = 0.05  # +5% confidence for bear market alignment
                elif vix > 30:
                    # High VIX = high volatility = reduce confidence
                    stage1_boost = -0.05
                    stage1_signal = "VOLATILE"
                
                # Add trending events as context
                trending = world_context.get("trending_events", [])
                if trending:
                    features["TRENDING_EVENTS"] = trending[:3]
                
                if stage1_boost != 0:
                    LOGGER.info(
                        f"[{symbol}] 🌍 Stage 1 Context: regime={regime}, sentiment={sentiment}, "
                        f"VIX={vix:.1f}, signal={stage1_signal}, boost={stage1_boost:+.0%}"
                    )
        except Exception as e:
            LOGGER.debug(f"[{symbol}] Stage 1 context unavailable: {e}")

        # PREDICTION_HORIZON_HOURS: Configurable via env (validated Dec 21-22, 2025)
        # - 6h: For testing/validation (1.5% targets based on actual market moves)
        # - 24h: Medium-term (3.5% targets)
        # - 48h: Production goal (6% targets)
        horizon_h = int(os.getenv("PREDICTION_HORIZON_HOURS", "48"))
        
        # Step size: Higher resolution for shorter horizons
        if horizon_h <= 6:
            step_s = 1800  # 30 minutes for 6h
        elif horizon_h <= 24:
            step_s = 3600  # 1 hour for 24h
        else:
            step_s = 3600 * 2  # 2 hours for 48h+
        num_points = (horizon_h * 3600) // step_s

        # =====================================================================
        # DETERMINE DIRECTION & CONFIDENCE (Ghost v3 Calibration System)
        # =====================================================================
        
        # Step 1: Determine base direction from strongest signals
        # FIX (Feb 8): Default to NEUTRAL so ALL fallback signals (MACD, momentum) get a chance
        # Previous "UP" default caused 69% UP predictions with only 29.8% win rate (vs 87.4% DOWN)
        # The MACD and momentum checks were dead code because they only fired on "FLAT"
        direction = "NEUTRAL"
        rsi = features.get("RSI_14")
        macd_hist = features.get("MACD_HISTOGRAM")
        
        # RSI takes priority (most reliable)
        if rsi is not None:
            if rsi > 70:
                direction = "DOWN"  # Overbought
            elif rsi < 30:
                direction = "UP"  # Oversold
        
        # MACD confirmation/override (FIX: now checks NEUTRAL instead of dead-code FLAT)
        if direction == "NEUTRAL" and macd_hist is not None:
            if macd_hist > 0:
                direction = "UP"
            elif macd_hist < 0:
                direction = "DOWN"
        
        # Price momentum as fallback
        if direction == "NEUTRAL":
            try:
                hist = _get_price_history_cached(symbol, days=5)
                if hist and len(hist) >= 2:
                    prices = [h["price"] for h in hist if h.get("price")]
                    if prices:
                        recent_change_pct = (prices[-1] - prices[0]) / prices[0] * 100
                        if recent_change_pct > 3:
                            direction = "UP"
                        elif recent_change_pct < -3:
                            direction = "DOWN"
            except Exception:
                pass
        
        # If still NEUTRAL after all signals, stay NEUTRAL — don't guess
        # FIX (Step 6, Mar 18 2026): Was defaulting to "UP" which created
        # systematic upward bias. If RSI, MACD, and momentum can't determine
        # direction, forcing UP is worse than letting ensemble decide.
        # The ensemble override at L9343 handles NEUTRAL → UP/DOWN when it
        # has confidence > 0.45. If ensemble ALSO can't decide, HOLD is
        # better than a random guess.
        if direction == "NEUTRAL":
            direction = "HOLD"
        
        # Step 2: ENSEMBLE MODEL VOTING (Task #6)
        # Combine LSTM + XGBoost + Transformer for 10-15% accuracy boost
        from core.ensemble_predictor import get_ensemble_predictor
        
        # FIX (Mar 18, 2026): Pass current_price into features so the ensemble
        # predictor can use it for price-level neutral defaults (SMA, EMA, BB).
        # Without this, missing SMA/EMA/BB features default to 0, which the
        # model interprets as "price is infinitely above its moving average"
        # = extreme overbought = massive DOWN bias.
        features["current_price"] = current_price
        
        ensemble = get_ensemble_predictor()
        ensemble_prediction = ensemble.predict(features, method="confidence_weighted", symbol=symbol)
        
        # Extract XGBoost raw probabilities for debugging
        xgb_debug = {}
        if ensemble_prediction.individual_predictions:
            xgb_pred = ensemble_prediction.individual_predictions[0]
            xgb_debug = getattr(xgb_pred, 'metadata', {}) or {}
            xgb_debug["xgb_direction"] = xgb_pred.direction
            xgb_debug["xgb_confidence"] = round(xgb_pred.confidence, 4)
        
        # Use ensemble direction if confidence is moderate (lowered from 0.55 to enable regime override)
        if ensemble_prediction.confidence > 0.45:
            direction = ensemble_prediction.direction
            LOGGER.info(
                f"[{symbol}] 🤖 Ensemble override: {direction} "
                f"({ensemble_prediction.confidence:.1%}) - "
                f"Models agree: {len([p for p in ensemble_prediction.individual_predictions if p.direction == direction])}/3"
            )
        
        # =====================================================================
        # BRAIN v3: 25 Cognitive Abilities (Feb 26, 2026)
        # Applies per-symbol accuracy intelligence to EVERY prediction:
        #   - Invert bad symbols (<38% accuracy → flip direction)
        #   - Exclude trash (<48% accuracy → HOLD)
        #   - Boost proven winners (>62% accuracy → confidence up)
        #   - Streak modifiers, regime gates, F&G, sector correlation
        #   - Circuit breaker (system-wide accuracy crash → halt)
        # =====================================================================
        _brain_applied = False
        _brain_decision = None
        _brain_conf_delta = 0.0
        try:
            if os.getenv("GHOST_BRAIN_ENABLED", "1") == "1":
                from core.ghost_brain import GhostBrain
                from core.brain_data import load_brain_context
                import asyncio as _brain_asyncio

                _brain = GhostBrain()

                # Load rich context (accuracy history, streaks, regime, F&G)
                _brain_db_url = os.getenv("DATABASE_URL", "")
                _brain_market_data = {
                    "fear_greed_index": features.get("FEAR_GREED_INDEX", 50),
                    "btc_24h_change": features.get("BTC_24H_CHANGE", 0.0),
                    "eth_24h_change": features.get("ETH_24H_CHANGE", 0.0),
                    "spy_24h_change": features.get("SPY_24H_CHANGE", 0.0),
                    "vix_level": features.get("VIX_LEVEL", 20),
                }

                # Load context in a thread-safe way
                def _load_brain_ctx():
                    _loop = _brain_asyncio.new_event_loop()
                    try:
                        return _loop.run_until_complete(
                            load_brain_context(_brain_db_url, [symbol], _brain_market_data)
                        )
                    finally:
                        _loop.close()

                try:
                    _running = _brain_asyncio.get_running_loop()
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as _bp:
                        _brain_ctx = _bp.submit(_load_brain_ctx).result(timeout=5)
                except RuntimeError:
                    _brain_ctx = _load_brain_ctx()

                _brain_decision = _brain.analyze_symbol(
                    symbol=symbol,
                    direction=direction,
                    confidence=ensemble_prediction.confidence if ensemble_prediction.confidence > 0.45 else base_confidence if 'base_confidence' in dir() else 0.52,
                    context=_brain_ctx,
                )

                if _brain_decision:
                    _brain_applied = True
                    _old_dir = direction
                    _old_conf = ensemble_prediction.confidence if ensemble_prediction.confidence > 0.45 else 0.52

                    # Apply Brain's direction (may invert)
                    if _brain_decision.direction in ("UP", "DOWN"):
                        direction = _brain_decision.direction

                    # Apply Brain's confidence modifier
                    if _brain_decision.confidence and _brain_decision.confidence > 0:
                        # Blend: 70% ensemble + 30% brain
                        _ensemble_c = ensemble_prediction.confidence if ensemble_prediction.confidence > 0.45 else 0.52
                        # Use brain confidence as a modifier, not replacement
                        _brain_conf_delta = _brain_decision.confidence - _old_conf
                        # Brain can adjust up to ±15%
                        _brain_conf_delta = max(-0.15, min(0.15, _brain_conf_delta))

                    # Apply Brain's action (EXCLUDE → force HOLD)
                    if _brain_decision.action == "EXCLUDE":
                        direction = "HOLD"
                        LOGGER.warning(
                            f"[{symbol}] 🧠 BRAIN EXCLUDED: {_brain_decision.tier} "
                            f"(reasons: {', '.join(_brain_decision.reasons[:3])})"
                        )
                    elif _brain_decision.action == "INVERT":
                        LOGGER.info(
                            f"[{symbol}] 🧠 BRAIN INVERTED: {_old_dir} → {direction} "
                            f"(brain_accuracy={_brain_decision.brain_accuracy:.1f}%, "
                            f"reasons: {', '.join(_brain_decision.reasons[:3])})"
                        )
                    elif _brain_decision.action in ("BOOST", "SEND"):
                        LOGGER.info(
                            f"[{symbol}] 🧠 BRAIN {_brain_decision.action}: {_brain_decision.tier} "
                            f"(brain_accuracy={_brain_decision.brain_accuracy:.1f}%, "
                            f"conf_mod={_brain_conf_delta:+.1%})"
                        )
        except Exception as e:
            LOGGER.warning(f"[{symbol}] Brain v3 analysis failed (continuing without): {e}")

        # Step 2.5a: ADAPTIVE DIRECTIONAL CONFIDENCE ADJUSTMENT (Feb 8, 2026)
        # Auto-adjusts UP/DOWN penalties based on LIVE paper trade performance.
        # If market regime flips (bear→bull), penalties auto-reverse within 1 hour.
        # Replaces hardcoded penalties that would break on regime change.
        _directional_adjustment = 0.0
        _directional_meta = {}
        try:
            from core.directional_accuracy_tracker import get_directional_adjustment
            _directional_adjustment, _directional_meta = get_directional_adjustment(direction)
            
            if abs(_directional_adjustment) > 0.005:
                adj_type = "PENALTY" if _directional_adjustment < 0 else "BONUS"
                adj_wr = _directional_meta.get('win_rate', '?')
                adj_n = _directional_meta.get('sample_size', 0)
                regime = _directional_meta.get('regime', '?')
                LOGGER.info(
                    f"[{symbol}] ⚖️ ADAPTIVE {direction} {adj_type}: {_directional_adjustment:+.1%} "
                    f"({direction} WR={adj_wr}% over {adj_n} trades, regime={regime})"
                )
            else:
                LOGGER.debug(f"[{symbol}] ⚖️ Directional adjustment negligible: {_directional_adjustment:+.4f}")
        except Exception as e:
            LOGGER.debug(f"[{symbol}] Directional tracker unavailable (no bias applied): {e}")
            # FIX (Step 6, Mar 18 2026): Was hardcoding -12% UP penalty / +5% DOWN
            # bonus when tracker fails. This created systematic bearish bias that
            # tanked accuracy in bullish markets. Now: no data = no bias.
            _directional_adjustment = 0.0
        
        # Step 2.5b: PATTERN INTELLIGENCE BOOST (v4.0)
        # Use fear/greed, funding rates, social sentiment, BTC correlation
        # When multiple signals align, confidence increases 5-20%
        pattern_boost = 0.0
        pattern_signals = []
        try:
            if os.environ.get("ENABLE_PATTERN_INTELLIGENCE", "1") == "1":
                from core.pattern_enhanced_predictor import get_pattern_enhanced_predictor
                
                pattern_predictor = get_pattern_enhanced_predictor()
                pattern_result = pattern_predictor.predict(symbol, features)
                
                # Check if pattern signals align with ensemble direction
                if pattern_result.direction == direction:
                    # Signals agree - boost confidence
                    pattern_boost = pattern_result.confidence - 0.75  # Boost based on pattern confidence
                    pattern_boost = max(0, min(pattern_boost, 0.15))  # Cap at 15% boost
                    pattern_signals = pattern_result.data_sources
                    
                    if pattern_boost > 0:
                        LOGGER.info(
                            f"[{symbol}] 📊 Pattern Intelligence: {pattern_result.direction} "
                            f"({pattern_result.confidence:.1%}) - Signals: {', '.join(pattern_signals)} "
                            f"- Boost: +{pattern_boost:.1%}"
                        )
                else:
                    # Signals conflict - reduce confidence slightly
                    LOGGER.warning(
                        f"[{symbol}] ⚠️ Pattern conflict: Ensemble={direction}, Pattern={pattern_result.direction}"
                    )
        except Exception as e:
            # Pattern intelligence is optional - don't fail prediction
            LOGGER.debug(f"[{symbol}] Pattern intelligence unavailable: {e}")
        
        # Step 3: Calibrate confidence using signal-based system
        from core.confidence_calibrator import calibrate_confidence_with_signals
        
        base_confidence = 0.52  # Aligned with balanced XGBoost output (~52%)
        calibration_result = calibrate_confidence_with_signals(
            features=features,
            base_direction=direction,
            base_confidence=base_confidence
        )
        
        # Extract signal-calibrated confidence and signals
        signal_confidence = float(calibration_result.get("calibrated_confidence") or base_confidence)
        signal_strength = int(calibration_result.get("signal_count") or 0)
        signals_fired = calibration_result.get("signals_fired") or []
        
        # Fuse ensemble confidence with signal calibration (money-reality: prefer model agreement)
        try:
            ensemble_conf = float(getattr(ensemble_prediction, "confidence", 0.0) or 0.0)
        except Exception:
            ensemble_conf = 0.0

        base_confidence = max(signal_confidence, ensemble_conf)

        # ADAPTIVE DIRECTIONAL ADJUSTMENT: Apply data-driven penalty/bonus
        # Auto-adjusts based on live UP vs DOWN win rates from paper_trades
        # If market flips bullish, UP penalty auto-reduces / reverses
        if abs(_directional_adjustment) > 0.005:
            pre_adj_confidence = base_confidence
            base_confidence = base_confidence + _directional_adjustment
            base_confidence = max(base_confidence, 0.25)  # Floor at 25%
            LOGGER.info(
                f"[{symbol}] ⚖️ Directional adj applied: {pre_adj_confidence:.1%} → {base_confidence:.1%} "
                f"({_directional_adjustment:+.1%}, regime={_directional_meta.get('regime', '?')})"
            )
        
        # CONFIDENCE CAP: 80% maximum - markets are inherently uncertain
        # Even with perfect model agreement and all signals aligned, we can't claim >80%
        MAX_CONFIDENCE = 0.80

        # Optional synergy bonus (only when BOTH strong ensemble + multiple signals)
        if ensemble_conf >= 0.70 and ensemble_prediction.direction == direction and signal_strength >= 3:
            base_confidence = min(base_confidence + 0.03, MAX_CONFIDENCE)  # Reduced bonus
            LOGGER.info(f"[{symbol}] 🚀 Ensemble+signals synergy: +3% confidence")
        
        # Apply Pattern Intelligence boost (when signals align) - REDUCED
        if pattern_boost > 0:
            pattern_boost = min(pattern_boost, 0.05)  # Cap pattern boost at 5%
            base_confidence = min(base_confidence + pattern_boost, MAX_CONFIDENCE)
            LOGGER.info(f"[{symbol}] 📊 Pattern Intelligence boost: +{pattern_boost:.1%} (final: {base_confidence:.1%})")

        # Apply Brain v3 confidence modifier (capped ±15%)
        if _brain_applied and _brain_conf_delta != 0:
            _pre_brain_conf = base_confidence
            base_confidence = base_confidence + _brain_conf_delta
            base_confidence = max(base_confidence, 0.25)  # Floor at 25%
            base_confidence = min(base_confidence, MAX_CONFIDENCE)  # Respect cap
            LOGGER.info(
                f"[{symbol}] 🧠 Brain v3 confidence: {_pre_brain_conf:.1%} → {base_confidence:.1%} "
                f"(delta={_brain_conf_delta:+.1%})"
            )

        # =====================================================================
        # MULTI-HORIZON CONSENSUS CHECK (Feb 25, 2026)
        # Uses already-extracted features to compute a quick multi-timeframe
        # directional vote. If short/medium/long timeframes disagree with the
        # ensemble direction, apply a conflict penalty. If they agree, small boost.
        # This replaces calling the full MultiHorizonForecaster which needs
        # yfinance HTTP calls (too slow for hot path).
        # =====================================================================
        _mh_adjustment = 0.0
        try:
            if os.getenv("MULTI_HORIZON_CONSENSUS_ENABLED", "1") == "1":
                # Short-term signal: RSI momentum (< 30 = bullish, > 70 = bearish)
                _rsi = features.get("RSI_14") or features.get("RSI", 50)
                _short_bullish = _rsi < 40  # Oversold territory
                _short_bearish = _rsi > 60  # Overbought territory
                
                # Medium-term signal: MACD histogram direction
                _macd_hist = features.get("MACD_HISTOGRAM", 0) or 0
                _med_bullish = _macd_hist > 0
                _med_bearish = _macd_hist < 0
                
                # Long-term signal: Price vs SMA/EMA and momentum
                _momentum = features.get("MOMENTUM_7D", features.get("MOMENTUM", 0)) or 0
                _sma_50_diff = features.get("SMA_50_DIFF", features.get("SMA_DIFF", 0)) or 0
                _long_bullish = _momentum > 0 and _sma_50_diff > 0
                _long_bearish = _momentum < 0 and _sma_50_diff < 0
                
                # Count agreement with current direction
                if direction == "UP":
                    _votes_agree = sum([_short_bullish, _med_bullish, _long_bullish])
                    _votes_disagree = sum([_short_bearish, _med_bearish, _long_bearish])
                elif direction == "DOWN":
                    _votes_agree = sum([_short_bearish, _med_bearish, _long_bearish])
                    _votes_disagree = sum([_short_bullish, _med_bullish, _long_bullish])
                else:
                    _votes_agree = 0
                    _votes_disagree = 0
                
                # Apply adjustment based on agreement
                if _votes_agree >= 3:
                    _mh_adjustment = 0.03  # All 3 horizons agree: +3%
                elif _votes_agree >= 2 and _votes_disagree == 0:
                    _mh_adjustment = 0.02  # 2 agree, none disagree: +2%
                elif _votes_disagree >= 2:
                    _mh_adjustment = -0.03  # 2+ horizons disagree: -3%
                elif _votes_disagree >= 1 and _votes_agree == 0:
                    _mh_adjustment = -0.02  # 1 disagrees, none agree: -2%
                
                if _mh_adjustment != 0:
                    pre_mh = base_confidence
                    base_confidence = max(0.25, min(MAX_CONFIDENCE, base_confidence + _mh_adjustment))
                    LOGGER.info(
                        f"[{symbol}] 🔭 MULTI-HORIZON: {_votes_agree}/3 agree, {_votes_disagree}/3 disagree → "
                        f"{pre_mh:.1%} → {base_confidence:.1%} ({_mh_adjustment:+.1%})"
                    )
                else:
                    LOGGER.debug(f"[{symbol}] Multi-horizon: mixed signals, no adjustment")
        except Exception as e:
            LOGGER.debug(f"[{symbol}] Multi-horizon consensus unavailable: {e}")
        
        # Apply Stage 1 Context boost/penalty (market regime alignment) - REDUCED
        if stage1_boost != 0:
            stage1_boost = max(-0.05, min(0.03, stage1_boost))  # Cap stage1 effects
            if stage1_signal in ["UP", "DOWN"] and stage1_signal == direction:
                # Stage 1 agrees with direction - apply boost
                base_confidence = min(base_confidence + stage1_boost, MAX_CONFIDENCE)
                LOGGER.info(f"[{symbol}] 🌍 Stage 1 boost: {stage1_boost:+.0%} (regime aligns with {direction})")
            elif stage1_signal == "VOLATILE":
                # High VIX - reduce confidence
                base_confidence = max(base_confidence + stage1_boost, 0.35)
                LOGGER.info(f"[{symbol}] 🌍 Stage 1 penalty: {stage1_boost:+.0%} (high VIX volatility)")
            elif stage1_signal in ["UP", "DOWN"] and stage1_signal != direction:
                # Stage 1 conflicts - small penalty
                penalty = abs(stage1_boost) / 2  # Half the boost as penalty
                base_confidence = max(base_confidence - penalty, 0.35)
                LOGGER.info(f"[{symbol}] 🌍 Stage 1 conflict: -{abs(penalty):.0%} (regime={stage1_signal}, direction={direction})")
        
        LOGGER.info(
            f"[{symbol}] Direction: {direction}, Confidence: {base_confidence:.1%}, "
            f"Signals: {signal_strength} ({', '.join(signals_fired[:3])}{'...' if len(signals_fired) > 3 else ''})"
        )

        # =====================================================================
        # ASSET PERFORMANCE FILTER (NEW: Jan 9, 2026)
        # Apply historical win rate adjustments BEFORE final confidence
        # =====================================================================
        try:
            from core.asset_performance_filter import get_performance_filter
            
            perf_filter = get_performance_filter()
            
            # Check if we should trade this symbol at all
            should_trade_symbol, trade_reason = perf_filter.should_trade(symbol)
            
            if not should_trade_symbol:
                # Symbol is blacklisted - force HOLD
                LOGGER.warning(
                    f"[{symbol}] ❌ BLACKLISTED: {trade_reason} - Forcing HOLD"
                )
                base_confidence = 0.0  # Zero confidence = don't trade
                should_predict = False
            else:
                # Adjust confidence based on historical performance
                original_confidence = base_confidence
                base_confidence = perf_filter.get_confidence_adjustment(symbol, base_confidence)
                
                adjustment = base_confidence - original_confidence
                if abs(adjustment) > 0.01:
                    LOGGER.info(
                        f"[{symbol}] 📊 Performance adjustment: "
                        f"{original_confidence:.1%} → {base_confidence:.1%} "
                        f"({adjustment:+.1%}) - {trade_reason}"
                    )
        except Exception as e:
            LOGGER.warning(f"[{symbol}] Performance filter failed (continuing without): {e}")

        # =====================================================================
        # MARKET GATES (NEW: Jan 25, 2026)
        # Filter BUY signals based on market conditions:
        # - Regime Filter: SPY > 20MA (stocks), BTC 7d trend (crypto)
        # - VIX Gate: Reduce confidence or block during high fear
        # - Confirmation Counter: Require 3-4 signals to agree
        # =====================================================================
        try:
            from core.market_gates import apply_market_gates
            from core.asset_classification import is_crypto_symbol as _is_crypto_sym
            import asyncio
            
            market_type = "crypto" if _is_crypto_sym(symbol) else "stock"
            
            # Build metrics dict for gates - normalize to lowercase keys
            # The confirmation counter expects: rsi_14, rsi, macd_histogram, bb_lower, bb_upper,
            # momentum_7d, volume_trend, current_price
            gate_metrics = {
                # Normalize UPPERCASE feature keys to lowercase for confirmation counter
                "rsi_14": features.get("RSI_14"),
                "rsi": features.get("RSI_14"),  # alias
                "macd_histogram": features.get("MACD_HISTOGRAM"),
                "macd_histogram_prev": features.get("MACD_HISTOGRAM_PREV", features.get("MACD_HISTOGRAM", 0)),
                "bb_lower": features.get("BB_LOWER"),
                "bb_upper": features.get("BB_UPPER"),
                "momentum_7d": features.get("MOMENTUM_7D", features.get("MOMENTUM", 0)),
                "momentum": features.get("MOMENTUM", 0),
                "volume_trend": features.get("VOLUME_TREND", features.get("VOLUME_SPIKE", 1.0)),
                "current_price": current_price,
                "price": current_price,
                # Also include original features for any other gate logic
                **{k.lower(): v for k, v in features.items()},
                # Keep signal info
                "signal_count": signal_strength,
                "signals_fired": signals_fired,
            }
            
            # Store original values before gates
            original_direction = direction
            original_confidence = base_confidence
            
            # Apply market gates (async function)
            # Use thread to run async code from sync context safely
            import concurrent.futures
            
            def _run_market_gates():
                """Run async market gates in a new event loop."""
                return asyncio.run(
                    apply_market_gates(
                        direction=direction,
                        confidence=base_confidence,
                        metrics=gate_metrics,
                        asset_type=market_type,
                        symbol=symbol
                    )
                )
            
            # Check if we're in an async context
            try:
                loop = asyncio.get_running_loop()
                # We're in async context - use thread to avoid nested loop
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(_run_market_gates)
                    gated_direction, gated_confidence, gate_info = future.result(timeout=5)
            except RuntimeError:
                # No running loop - safe to use asyncio.run directly
                gated_direction, gated_confidence, gate_info = _run_market_gates()
            
            # Update direction and confidence
            direction = gated_direction
            base_confidence = gated_confidence
            
            # Log the gates result
            if not gate_info.get("gates_passed", True):
                LOGGER.warning(
                    f"[{symbol}] ⚠️ MARKET GATES PENALIZED: {direction} {original_confidence:.1%} → {gated_confidence:.1%} "
                    f"(Regime: {gate_info.get('regime_filter', {}).get('reason', 'N/A')})"
                )
            elif gated_confidence < original_confidence:
                vix_info = gate_info.get("vix_gate", {})
                LOGGER.info(
                    f"[{symbol}] ⚠️ MARKET GATES REDUCED: {original_confidence:.1%} → {gated_confidence:.1%} "
                    f"(VIX: {vix_info.get('vix_level', 'N/A')}, multiplier: {vix_info.get('multiplier', 1.0):.2f})"
                )
            else:
                LOGGER.info(
                    f"[{symbol}] ✅ MARKET GATES PASSED: {direction} @ {base_confidence:.1%}"
                )
        except Exception as e:
            LOGGER.warning(f"[{symbol}] Market gates failed (continuing without): {e}")

        # =====================================================================
        # GHOST INTEL INTEGRATION (NEW: Jan 27, 2026)
        # Wire institutional intelligence into prediction adjustments
        # - VIX regime gates (block BUY during panic)
        # - Put/Call positioning (contrarian signals)
        # - Event impact (high-impact = block, medium = adjust)
        # - Macro regime (yield curve inversion)
        # =====================================================================
        intel_metadata = {}
        try:
            if os.getenv("GHOST_INTEL_ENABLED", "1") == "1":
                from ghost_intel.integration import apply_intel_to_prediction
                
                # Store original values
                pre_intel_direction = direction
                pre_intel_confidence = base_confidence
                
                # Apply Intel adjustments
                direction, base_confidence, intel_metadata = apply_intel_to_prediction(
                    symbol=symbol,
                    direction=direction,
                    confidence=base_confidence
                )
                
                # Log if Intel made changes
                if intel_metadata.get("intel_applied"):
                    conf_adj = intel_metadata.get("confidence_adjustment", 0)
                    if not intel_metadata.get("should_trade"):
                        LOGGER.warning(
                            f"[{symbol}] 🚫 INTEL BLOCKED: {intel_metadata.get('block_reason', 'Unknown')}"
                        )
                    elif abs(conf_adj) > 0.01:
                        sources = intel_metadata.get("signal_sources", [])[:3]
                        LOGGER.info(
                            f"[{symbol}] 🔮 INTEL: {pre_intel_confidence:.1%} → {base_confidence:.1%} "
                            f"({conf_adj:+.1%}) | VIX={intel_metadata.get('market_context', {}).get('vix', 'N/A')} "
                            f"| Sources: {', '.join(sources)}"
                        )
        except Exception as e:
            LOGGER.warning(f"[{symbol}] Intel integration failed (continuing without): {e}")

        # =====================================================================
        # CONFIDENCE THRESHOLD (NEW: Jan 9, 2026)
        # Only trade predictions with sufficient confidence
        # RAISED to 0.55 (Mar 13, 2026) — was 0.35, letting garbage through
        # 25.5% accuracy means the floor was too low. Fewer picks, higher quality.
        # =====================================================================
        MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", "0.55"))
        
        should_predict = True  # Default: predict unless gates say otherwise
        if base_confidence < MIN_CONFIDENCE_THRESHOLD:
            LOGGER.warning(
                f"[{symbol}] ⚠️ Low confidence: {base_confidence:.1%} < {MIN_CONFIDENCE_THRESHOLD:.1%} threshold - Forcing MONITOR"
            )
            should_predict = False  # Don't signal, just monitor

        # Degraded-data guardrail: if too few features are available, force MONITOR-only behavior.
        # This prevents low-signal, low-quality predictions from becoming Telegram signals.
        try:
            min_features_for_signal = int(os.getenv("MIN_FEATURES_FOR_SIGNAL", "10"))
            min_availability_pct = float(os.getenv("MIN_FEATURE_AVAILABILITY_PCT", "50"))
            available = int(feature_data.get("available_count") or 0)
            total = int(feature_data.get("feature_count") or 0)
            availability_pct = (available / total * 100.0) if total else 0.0
            degraded = available < min_features_for_signal or availability_pct < min_availability_pct
        except Exception:
            degraded = False

        # Outcome-driven confidence calibration (Postgres-backed, if available).
        # This makes confidence honest and ensures yesterday's outcomes affect tomorrow's signals.
        cal = None
        # FIXED: was unconditionally resetting should_predict=True here, which
        # nullified the MIN_CONFIDENCE_THRESHOLD check above. Now we preserve
        # the should_predict value from the threshold check and only let the
        # calibrator FURTHER restrict it (never override False → True).
        threshold_should_predict = should_predict  # preserve threshold decision
        expected_accuracy = base_confidence
        try:
            from core.confidence_calibrator import get_confidence_calibrator

            calibrator = get_confidence_calibrator()
            cal = calibrator.calibrate_confidence(base_confidence, symbol=symbol)
            expected_accuracy = float(cal.get("expected_accuracy", base_confidence))
            calibrator_should_predict = bool(cal.get("should_predict", True))
            
            # Both threshold AND calibrator must agree to predict
            should_predict = threshold_should_predict and calibrator_should_predict

            # Use calibrated expected accuracy as the confidence we expose downstream.
            base_confidence = expected_accuracy
        except Exception:
            # If calibrator fails, honor the threshold decision only
            should_predict = threshold_should_predict

        # Apply degraded guardrail last (never signal on degraded inputs)
        if degraded:
            should_predict = False

        # STRICT FAIL-CLOSED (production-grade): if critical pillars are missing/degraded,
        # do not create/store a prediction at all.
        def _truthy(v: str | None) -> bool:
            return str(v or "").strip().lower() in {"1", "true", "yes", "on"}

        strict_fail_closed = (
            _truthy(os.getenv("PREDICT_FAIL_CLOSED", "0"))
            or _truthy(os.getenv("PRICE_STRICT_LIVE", "0"))
            or _truthy(os.getenv("ENFORCE_LIVE", "0"))
        )

        def _parse_ratio(v: object) -> tuple[int, int]:
            try:
                s = str(v or "")
                left = s.split()[0]
                a, b = left.split("/", 1)
                return int(a), int(b)
            except Exception:
                return 0, 0

        pillar_availability = feature_data.get("feature_availability") or {}
        price_avail, _ = _parse_ratio(pillar_availability.get("price_engine"))
        tech_avail, _ = _parse_ratio(pillar_availability.get("technical_engine"))
        vol_avail, _ = _parse_ratio(pillar_availability.get("volume_engine"))

        missing_price_pillar = current_price is None and price_avail <= 0

        if strict_fail_closed and (degraded or missing_price_pillar or tech_avail <= 0 or vol_avail <= 0):
            duration_ms = int((time.monotonic() - start) * 1000)
            reason = (
                f"fail_closed: degraded={degraded}, "
                f"pillars(price/tech/vol)={price_avail}/{tech_avail}/{vol_avail}, "
                f"has_current_price={current_price is not None}"
            )
            LOGGER.warning(
                f"[{symbol}] FAIL-CLOSED: {reason}",
                extra={
                    "symbol": symbol,
                    "duration_ms": duration_ms,
                    "availability_pct": round(feature_avail_pct, 1),
                    "pillar_breakdown": pillar_availability,
                },
            )
            return {
                "ok": False,
                "symbol": symbol,
                "direction": "ERROR",
                "confidence": 0.0,
                "current_price": current_price,
                "feature_count": int(feature_data.get("feature_count") or 0),
                "available_count": int(feature_data.get("available_count") or 0),
                "duration_ms": duration_ms,
                "error": reason,
            }

        # ==========================================================================
        # EXPECTED MOVE CALCULATION (FIX: Use volatility-based, not hardcoded)
        # 
        # OLD BUG: Used 1.01^12 = +12.68% for ALL UP predictions
        # NEW: Calculate per-symbol expected move based on:
        #   1. Historical volatility (ATR if available)
        #   2. Confidence level (higher confidence = larger expected move)
        #   3. Horizon (6h = smaller move than 48h)
        # ==========================================================================
        
        # ======================================================================
        # 🔄 INVERSE GHOST - Per-Symbol Toggle (Updated Dec 28, 2025)
        # ======================================================================
        # Analysis shows INVERSE helps SOME symbols but HURTS others:
        #
        # INVERSE HELPS (raw accuracy <30%, invert to 70%+):
        #   - ETH: 10% raw → 90% inverted ✅
        #   - BTC: 20% raw → 80% inverted ✅
        #   - XRP: 20% raw → 80% inverted ✅
        #   - QTUM: 20% raw → 80% inverted ✅
        #
        # INVERSE HURTS (raw accuracy >60%, invert makes it worse):
        #   - OMG: 90% raw → 10% inverted ❌
        #   - RLC: 90% raw → 10% inverted ❌
        #   - THETA: 80% raw → 20% inverted ❌
        #   - DOGE: 70% raw → 30% inverted ❌
        #   - EGLD: 80% raw → 20% inverted ❌
        #   - BAT: 80% raw → 20% inverted ❌
        #   - ONDO: 80% raw → 20% inverted ❌
        #   - ZEN: 80% raw → 20% inverted ❌
        #
        # Solution: Don't invert symbols with HIGH raw accuracy
        # ======================================================================
        
        # Symbols where raw Ghost is GOOD - DON'T invert these
        # Read from env var INVERSE_SKIP_SYMBOLS (comma-separated) or use defaults
        env_skip_symbols = os.getenv("INVERSE_SKIP_SYMBOLS", "").strip()
        if env_skip_symbols:
            # User provided custom list via env var
            INVERSE_SKIP_SYMBOLS = {s.strip().upper() for s in env_skip_symbols.split(",") if s.strip()}
        else:
            # Default list: <40% inverted accuracy = >60% raw accuracy = model is RIGHT
            INVERSE_SKIP_SYMBOLS = {
                "OMG",    # 10% inverted → 90% raw (VERY GOOD raw)
                "RLC",    # 10% inverted → 90% raw (VERY GOOD raw)
                "THETA",  # 20% inverted → 80% raw (GOOD raw)
                "EGLD",   # 20% inverted → 80% raw (GOOD raw)
                "BAT",    # 20% inverted → 80% raw (GOOD raw)
                "ONDO",   # 20% inverted → 80% raw (GOOD raw)
                "ZEN",    # 20% inverted → 80% raw (GOOD raw)
                "DOGE",   # 30% inverted → 70% raw (GOOD raw)
                "DOT",    # 30% inverted → 70% raw (GOOD raw)
                "ZRX",    # 30% inverted → 70% raw (GOOD raw)
                "BNB",    # 30% inverted → 70% raw (GOOD raw)
                "AVAX",   # 30% inverted → 70% raw (GOOD raw)
                "OCEAN",  # 30% inverted → 70% raw (GOOD raw)
                "ANT",    # 40% inverted → 60% raw (borderline, skip)
            }
        
        # INVERSE_GHOST: When 1, flip predictions. When 0, use raw predictions.
        # DEFAULT is now OFF (0) since accuracy improvements were made
        inverse_ghost_enabled = os.getenv("INVERSE_GHOST", "0") == "1"  # OFF by default
        symbol_upper = symbol.upper()
        
        if inverse_ghost_enabled and direction in ("UP", "DOWN"):
            if symbol_upper in INVERSE_SKIP_SYMBOLS:
                # This symbol has HIGH raw accuracy - keep raw prediction
                LOGGER.info(
                    f"[{symbol}] ⏭️ INVERSE SKIP: Keeping raw {direction} "
                    f"(symbol in INVERSE_SKIP_SYMBOLS - high raw accuracy)"
                )
            else:
                # Normal symbols - invert the prediction
                original_direction = direction
                direction = "DOWN" if direction == "UP" else "UP"
                LOGGER.warning(
                    f"[{symbol}] 🔄 INVERSE GHOST: {original_direction} → {direction} "
                    f"(per-symbol toggle active)"
                )
        
        # ==========================================================================
        # EXPECTED MOVE CALCULATION (FIXED Dec 21-22, 2025)
        # 
        # For 6-hour predictions, realistic moves are 2-5%, NOT 25%!
        # Each coin should have DIFFERENT expected moves based on volatility.
        #
        # Volatility tiers (6h expected moves):
        #   - BTC/ETH: 2-4% (large caps, less volatile)
        #   - SOL/DOGE/XRP: 3-5% (mid caps)
        #   - Small alts: 4-7% (more volatile)
        # ==========================================================================
        
        # Get coin-specific volatility from features (or use sensible defaults)
        atr_pct = features.get("ATR_PERCENT")
        volatility_20d = features.get("VOLATILITY_20D")
        
        # Determine base volatility (daily) for this specific coin
        if atr_pct and atr_pct > 0 and atr_pct < 50:  # Sanity check ATR
            coin_daily_vol = float(atr_pct)
        elif volatility_20d and volatility_20d > 0 and volatility_20d < 1:  # It's a decimal
            coin_daily_vol = float(volatility_20d) * 100
        elif volatility_20d and volatility_20d > 0 and volatility_20d < 50:  # Already %
            coin_daily_vol = float(volatility_20d)
        else:
            # Fallback: Use sensible defaults per coin type
            large_caps = {"BTC", "ETH"}
            mid_caps = {"SOL", "BNB", "XRP", "ADA", "DOGE", "DOT", "AVAX", "LINK", "MATIC"}
            
            if symbol.upper() in large_caps:
                coin_daily_vol = 3.0  # BTC/ETH: ~3% daily
            elif symbol.upper() in mid_caps:
                coin_daily_vol = 5.0  # Mid caps: ~5% daily
            else:
                coin_daily_vol = 7.0  # Small alts: ~7% daily
        
        # ==========================================================================
        # ASSET-AWARE TARGET SIZING (Dec 22, 2025)
        # 
        # ISSUE FOUND: Stock targets were 6-7% (same as crypto) - too aggressive!
        # - AAPL moving 6% in 48h is rare (happens 2-3x/year)
        # - Large cap stocks move 0.5-1.5% daily, not 3-5% like crypto
        #
        # SOLUTION: Use AssetClassifier for proper target/stop sizing
        # | Asset Type     | 48h Target | 48h Stop |
        # |----------------|------------|----------|
        # | Crypto         | 6.0%       | 4.5%     |
        # | Stock (Large)  | 2.5%       | 2.0%     |
        # | Stock (Volatile)| 5.0%      | 4.0%     |
        # | Stock (Mid)    | 3.5%       | 2.5%     |
        # ==========================================================================
        try:
            from core.asset_classifier import get_target_stop, get_asset_type
            
            asset_targets = get_target_stop(symbol, horizon_h)
            base_move_pct = asset_targets['target_pct']
            asset_type = asset_targets['asset_type']
            
            LOGGER.info(
                f"[{symbol}] Asset classification: {asset_type}, "
                f"target={base_move_pct}%, horizon={horizon_h}h"
            )
        except Exception as e:
            LOGGER.warning(f"[{symbol}] AssetClassifier failed: {e}, using crypto defaults")
            # Fallback to crypto defaults if classifier fails
            if horizon_h <= 6:
                base_move_pct = 1.5
            elif horizon_h <= 24:
                base_move_pct = 3.5
            else:
                base_move_pct = 6.0
            asset_type = "crypto"
        
        # Adjust slightly by confidence (higher confidence = slightly larger move)
        # Range: 0.9x to 1.1x of base move (tighter range)
        confidence_adjustment = 0.9 + (base_confidence * 0.2)  # 0.9 to 1.1
        
        # Calculate expected move
        expected_move_pct = base_move_pct * confidence_adjustment
        
        # Apply direction
        if direction == "DOWN":
            expected_move_pct = -expected_move_pct
        elif direction == "FLAT":
            expected_move_pct = 0.0
        
        # Cap at REALISTIC bounds for crypto
        max_move = 5.0 if horizon_h <= 6 else 8.0 if horizon_h <= 24 else 12.0
        expected_move_pct = max(-max_move, min(max_move, expected_move_pct))
        
        LOGGER.debug(
            f"[{symbol}] Expected move calculation: "
            f"base_move={base_move_pct:.1f}%, conf_adj={confidence_adjustment:.2f}, "
            f"→ {expected_move_pct:+.2f}%"
        )

        # Generate forecast points using calculated expected move
        forecast_points = []
        if current_price and expected_move_pct != 0:
            # Linear interpolation from current to target
            end_price = current_price * (1 + expected_move_pct / 100)
            for i in range(num_points + 1):
                ts = run_at + (i * step_s)
                # Linear interpolation
                progress = i / num_points if num_points > 0 else 1.0
                price = current_price + (end_price - current_price) * progress
                forecast_points.append((ts, price))
        else:
            # Flat forecast
            for i in range(num_points + 1):
                ts = run_at + (i * step_s)
                forecast_points.append((ts, current_price))

        # GHOST V3: Signal-based confidence calibration system
        # Confidence dynamically adjusts from 45% baseline to 40-85% based on:
        # - Technical indicator alignment (RSI, MACD, Bollinger Bands)
        # - Volume confirmation
        # - News sentiment
        # - Market context
        # This is THE CRITICAL component that enables 60%+ confidence for trading
        confidence = float(base_confidence)
        
        # =====================================================================
        # TRUST LADDER: Apply confidence boost for proven symbols (Feb 25, 2026)
        # Symbols that have been winning consistently get promoted through levels:
        #   Level 1 (Standard): 1.0x (no boost)
        #   Level 2 (Extended): 1.10x (+10% confidence)
        #   Level 3 (Focused):  1.20x (+20% confidence)
        # Trust ladder outcomes are recorded in paper_tracker.py on WIN/LOSS.
        # This closes the loop: proven symbols get higher confidence.
        # =====================================================================
        trust_boost = 1.0
        trust_level = 1
        try:
            if os.getenv("TRUST_LADDER_ENABLED", "1") == "1":
                from core.trust_ladder import get_trust_ladder
                
                trust_data = get_trust_ladder().get_trust(symbol)
                trust_boost = trust_data.confidence_boost
                trust_level = trust_data.trust_level
                
                if trust_boost > 1.0:
                    pre_trust_confidence = confidence
                    confidence = confidence * trust_boost
                    LOGGER.info(
                        f"[{symbol}] 🏆 TRUST LADDER: Level {trust_level} "
                        f"({trust_data.level_config['name']}) → "
                        f"{pre_trust_confidence:.1%} × {trust_boost:.2f} = {confidence:.1%} "
                        f"(wins={trust_data.consecutive_wins}, "
                        f"accuracy={trust_data.accuracy_pct:.0f}%)"
                    )
                else:
                    LOGGER.debug(
                        f"[{symbol}] Trust ladder: Level {trust_level} (Standard, no boost)"
                    )
        except Exception as e:
            LOGGER.debug(f"[{symbol}] Trust ladder unavailable: {e}")
        
        # =====================================================================
        # TRADE LEARNING LOOP — Dynamic confidence adjustment (Mar 13, 2026)
        # Uses historical patterns (win rate by confidence bucket, symbol,
        # direction) to adjust confidence. If Ghost has been losing on this
        # symbol/direction/bucket combo, confidence gets pulled down.
        # =====================================================================
        try:
            from core.trade_learning_loop import get_confidence_adjustment as _tll_adj_fn
            _tll_adj = _tll_adj_fn(symbol, direction, confidence)
            if abs(_tll_adj) > 0.005:
                _pre_tll = confidence
                confidence += _tll_adj
                confidence = max(0.10, confidence)  # Don't go below 10%
                LOGGER.info(
                    f"[{symbol}] 🎓 LEARNING ADJ: {_pre_tll:.1%} → {confidence:.1%} "
                    f"({_tll_adj:+.1%}) — based on historical patterns"
                )
        except Exception as _tll_err:
            LOGGER.debug(f"[{symbol}] Learning loop adjustment unavailable: {_tll_err}")
        
        # =====================================================================
        # HARD CAP: NEVER claim more than 85% confidence (Jan 31, 2026)
        # Real trading systems rarely exceed this. Our 52% win rate doesn't
        # justify 90%+ confidence claims.
        # =====================================================================
        HARD_CONFIDENCE_CAP = 0.85
        if confidence > HARD_CONFIDENCE_CAP:
            LOGGER.info(f"[{symbol}] Confidence capped: {confidence:.1%} → {HARD_CONFIDENCE_CAP:.0%}")
            confidence = HARD_CONFIDENCE_CAP
        
        confidence_metadata = {
            "method": "signal_based_calibration_v3",
            "signal_strength": signal_strength,
            "base": 0.45,
            "signal_confidence": round(signal_confidence, 3),
            "ensemble_confidence": round(ensemble_conf, 3) if 'ensemble_conf' in locals() else None,
            "calibrated_expected_accuracy": round(expected_accuracy, 3),
            "should_predict": bool(should_predict),
            "degraded": bool(degraded),
            "calibration": cal or {},
            "signals_fired": signals_fired,
            "adjustments": calibration_result.get("adjustments", {}),
            "features_used": [k for k, v in features.items() if v is not None],
            "trust_level": trust_level,
            "trust_boost": trust_boost,
        }

        # Log calibration details
        LOGGER.debug(
            f"[{symbol}] Confidence calibration: 0.45 → {confidence:.2f} "
            f"(adjustments: {calibration_result.get('adjustments', {})})"
        )

        # =====================================================================
        # MOMENTUM TRACKER: Calculate confidence trend (HOT/WARMING/STABLE/COOLING/COLD)
        # =====================================================================
        momentum_data = {}
        try:
            from core.momentum_tracker import get_momentum_tracker
            
            tracker = get_momentum_tracker()
            momentum_data = tracker.calculate_momentum(
                symbol=symbol,
                current_confidence=confidence,
                current_direction=direction
            )
            
            LOGGER.info(
                f"[{symbol}] Momentum: {momentum_data['status']} {momentum_data['emoji']} "
                f"({momentum_data['confidence_delta_pct']:+.1f}% change) - {momentum_data['description']}"
            )
            
            # Add momentum to confidence metadata
            confidence_metadata["momentum"] = momentum_data
        except Exception as e:
            LOGGER.warning(f"[{symbol}] Momentum calculation failed: {e}")
            momentum_data = {
                "status": "STABLE",
                "emoji": "➡️",
                "arrow": "→",
                "confidence_delta": 0.0,
                "confidence_delta_pct": 0.0,
                "description": "Momentum unavailable",
                "alert_worthy": False
            }

        # =====================================================================
        # 🧠 INTELLIGENCE HUB — Apply 20-system intelligence to crypto predictions
        # =====================================================================
        _turbo_hub_meta = {}
        try:
            from core.intelligence_hub import get_intelligence_hub
            _turbo_hub = get_intelligence_hub()

            # Get price history for hub analysis
            _turbo_price_history = []
            try:
                from core.ghost_scout import GhostScout as _GS
                _gs_instance = _GS()
                _turbo_price_history = _gs_instance._fetch_price_history(
                    symbol, "crypto" if is_crypto else "stock"
                ) or []
            except Exception:
                pass

            _turbo_hub_report = _turbo_hub.analyze(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                entry_price=current_price or 0,
                asset_type="crypto" if is_crypto else "stock",
                price_history=_turbo_price_history,
            )

            _turbo_hub_meta = {
                "intel_active_systems": _turbo_hub_report.active_systems,
                "intel_total_systems": _turbo_hub_report.total_systems,
                "intel_news_risk": _turbo_hub_report.news_risk,
                "intel_direction_adj": _turbo_hub_report.direction_adjustment,
                "intel_confidence_adj": _turbo_hub_report.confidence_adjustment,
                "intel_trust_boost": _turbo_hub_report.trust_boost,
                "market_regime": _turbo_hub_report.regime_info.get("regime", "UNKNOWN"),
            }

            if _turbo_hub_report.should_block:
                LOGGER.info(f"🛑 [HUB] {symbol}: Turbo prediction BLOCKED — {_turbo_hub_report.block_reason}")
                duration_ms = int((time.monotonic() - start) * 1000)
                return {"ok": False, "symbol": symbol, "direction": "BLOCKED",
                        "confidence": 0.0, "current_price": current_price,
                        "duration_ms": duration_ms, "error": _turbo_hub_report.block_reason}

            # Apply direction flip
            if _turbo_hub_report.direction_adjustment == "FLIP":
                old_dir = direction
                direction = "DOWN" if direction == "UP" else "UP"
                LOGGER.info(f"🔄 [HUB] {symbol}: Turbo direction FLIPPED {old_dir} → {direction}")

            # Apply confidence adjustment (trust_boost excluded — already applied multiplicatively pre-hub)
            _old_turbo_conf = confidence
            confidence += _turbo_hub_report.confidence_adjustment  # No trust_boost — already applied at line ~9583
            confidence = max(0.10, min(0.85, confidence))  # Keep engine's 0.85 cap
            if abs(confidence - _old_turbo_conf) > 0.01:
                LOGGER.info(f"🧠 [HUB] {symbol}: Turbo conf {_old_turbo_conf:.2f} → {confidence:.2f} "
                            f"(news_risk={_turbo_hub_report.news_risk}, "
                            f"systems={_turbo_hub_report.active_systems}/{_turbo_hub_report.total_systems})")

            # Log active signals
            _active_sigs = [s for s in _turbo_hub_report.signals if s.active]
            if _active_sigs:
                _sig_summary = ", ".join(f"{s.source}={s.direction}@{s.confidence:.0%}" for s in _active_sigs[:6])
                LOGGER.info(f"🧠 [HUB] {symbol}: Signals: {_sig_summary}")

        except Exception as _turbo_hub_err:
            LOGGER.warning(f"🧠 [HUB] Turbo engine hub error for {symbol}: {_turbo_hub_err}")

        # =====================================================================
        # GUARD: Skip storage for HOLD predictions (not actionable)
        # FIX (Mar 1, 2026): HOLD means brain excluded or no edge.
        # Don't store — it creates fake predictions that tank accuracy.
        # =====================================================================
        if direction == "HOLD":
            duration_ms = int((time.monotonic() - start) * 1000)
            LOGGER.info(f"[{symbol}] ⏸️ HOLD — not actionable, skipping DB storage (conf={confidence:.2f})")
            return {
                "ok": False, "symbol": symbol, "direction": "HOLD",
                "confidence": confidence, "current_price": current_price,
                "duration_ms": duration_ms, "error": "HOLD — not actionable",
            }

        # ================================================================
        # ACCURACY AUTOPILOT GATE — Turbo Engine (Mar 13, 2026)
        # Skip predictions when system is paused or confidence too low.
        # This is the LAST gate before PostgreSQL storage.
        # ================================================================
        try:
            from core.accuracy_autopilot import should_skip_prediction as _turbo_ap_skip
            _turbo_skip, _turbo_reason = _turbo_ap_skip(confidence)
            if _turbo_skip:
                duration_ms = int((time.monotonic() - start) * 1000)
                LOGGER.info(f"[{symbol}] 🛑 Autopilot skip (turbo): {_turbo_reason}")
                return {
                    "ok": False, "symbol": symbol, "direction": direction,
                    "confidence": confidence, "current_price": current_price,
                    "duration_ms": duration_ms, "engine": "turbo",
                    "error": f"autopilot: {_turbo_reason}",
                }
        except Exception as _turbo_ap_err:
            LOGGER.debug(f"[{symbol}] Autopilot check failed (turbo): {_turbo_ap_err}")

        # Create prediction with rich features
        from core.prediction_store import PredictionRejected
        
        try:
            prediction_id = predictor.create_prediction(
                symbol=symbol,
                forecast_points=forecast_points,
                method="ghost-data-pillars-v1",
                confidence=confidence,
                direction=direction,
                features={
                    "current_price": current_price,
                    "feature_count": feature_data["feature_count"],
                    "available_count": feature_data["available_count"],
                    "feature_availability_pct": round(feature_avail_pct, 1),
                    "pillar_breakdown": pillar_availability,
                    **features  # Include all extracted features
                },
                params={"horizon_h": horizon_h, "step_s": step_s},
                tag="",
            )
        except PredictionRejected as e:
            duration_ms = int((time.monotonic() - start) * 1000)
            _reject_str = str(e)
            _is_dedup = "Duplicate" in _reject_str or "already has prediction" in _reject_str
            LOGGER.warning(
                f"[{symbol}] {'DEDUP (still caching)' if _is_dedup else 'FAIL-CLOSED (store)'}: {e}",
                extra={
                    "symbol": symbol,
                    "duration_ms": duration_ms,
                    "availability_pct": round(feature_avail_pct, 1),
                    "pillar_breakdown": pillar_availability,
                },
            )
            # BUG FIX: Dedup rejection means DB already has the prediction, but
            # _LATEST_PREDICTIONS (in-memory) may be empty after a deploy/restart.
            # Still store the computed prediction so cockpit + Telegram see it.
            if _is_dedup and direction and confidence > 0:
                from core.asset_classification import is_crypto_symbol as _is_crypto_dedup
                _dedup_action = "BUY" if direction == "UP" else "SELL" if direction == "DOWN" else "HOLD"
                # Extract original prediction_id from rejection message or DB
                _dedup_pred_id = None
                try:
                    import re as _dedup_re
                    _id_match = _dedup_re.search(r'prediction\s+(\d+)', _reject_str)
                    if _id_match:
                        _dedup_pred_id = int(_id_match.group(1))
                except Exception:
                    pass
                # Fallback: query PostgreSQL for the existing prediction
                if _dedup_pred_id is None:
                    try:
                        from core.db_pool import get_sync_connection as _dedup_get_conn
                        with _dedup_get_conn() as _dedup_conn:
                            _dedup_cur = _dedup_conn.cursor()
                            _dedup_cur.execute(
                                "SELECT id FROM ghost_predictions WHERE symbol = %s AND checked = 0 ORDER BY predicted_at DESC LIMIT 1",
                                (symbol,)
                            )
                            _dedup_row = _dedup_cur.fetchone()
                            if _dedup_row:
                                _dedup_pred_id = _dedup_row[0]
                    except Exception:
                        pass
                # FIX (Mar 12, 2026): Dedup path must also include target/stop/entry
                # so Telegram notifications don't have to guess with fallback values.
                _dedup_abs_move = abs(expected_move_pct) if expected_move_pct else 3.0
                try:
                    from core.asset_classifier import get_target_stop as _dedup_get_ts
                    _dedup_stops = _dedup_get_ts(symbol, horizon_h)
                    _dedup_stop_pct = _dedup_stops['stop_pct']
                except Exception:
                    _dedup_stop_pct = 4.5 if _is_crypto_dedup(symbol) else 2.0
                if direction == "UP":
                    _dedup_target = round(current_price * (1 + _dedup_abs_move / 100), 6)
                    _dedup_stop = round(current_price * (1 - _dedup_stop_pct / 100), 6)
                elif direction == "DOWN":
                    _dedup_target = round(current_price * (1 - _dedup_abs_move / 100), 6)
                    _dedup_stop = round(current_price * (1 + _dedup_stop_pct / 100), 6)
                else:
                    _dedup_target = current_price
                    _dedup_stop = current_price
                with _LATEST_PREDICTIONS_LOCK:
                    _LATEST_PREDICTIONS[symbol] = {
                        "prediction_id": _dedup_pred_id,  # Preserve original ID
                        "symbol": symbol,
                        "run_at": time.time(),
                        "confidence": confidence,
                        "direction": direction,
                        "action": _dedup_action,
                        "horizon_h": horizon_h,
                        "price": current_price,
                        "price_at_prediction": current_price,
                        "entry_price": current_price,
                        "target_price": _dedup_target,
                        "take_profit": _dedup_target,
                        "stop_loss": _dedup_stop,
                        "expected_move_pct": expected_move_pct,
                        "market": "crypto" if _is_crypto_dedup(symbol) else "stock",
                        "engine": "turbo",
                        **_turbo_hub_meta,
                    }
                LOGGER.info(f"[{symbol}] ✅ Dedup prediction cached in _LATEST_PREDICTIONS (conf={confidence:.2f}, target=${_dedup_target}, stop=${_dedup_stop})")
                return {
                    "ok": True,
                    "symbol": symbol,
                    "direction": direction,
                    "confidence": confidence,
                    "current_price": current_price,
                    "feature_count": int(feature_data.get("feature_count") or 0),
                    "available_count": int(feature_data.get("available_count") or 0),
                    "duration_ms": duration_ms,
                    "error": None,
                    "dedup": True,
                }
            return {
                "ok": False,
                "symbol": symbol,
                "direction": "ERROR",
                "confidence": 0.0,
                "current_price": current_price,
                "feature_count": int(feature_data.get("feature_count") or 0),
                "available_count": int(feature_data.get("available_count") or 0),
                "duration_ms": duration_ms,
                "error": str(e),
            }

        # =====================================================================
        # V3 DIRECTION OVERRIDE — apply EARLY (before storage + API return)
        # ghost_inverse: flip direction (PANW/NET/FTNT) or force UP (ETH)
        # always_up: force UP (DDOG)
        # Applied HERE so _LATEST_PREDICTIONS, ghost_predictions, paper_trades,
        # and API response ALL use the V3-overridden direction.
        # Without this, inverse trades are evaluated BACKWARDS everywhere.
        # =====================================================================
        from config.symbols import V3_VALIDATED_STRATEGIES as _V3_EARLY
        _v3_early_config = _V3_EARLY.get(symbol.upper())
        _v3_original_direction = None
        if _v3_early_config and _v3_early_config.direction_override:
            _v3_original_direction = direction
            if _v3_early_config.direction_override == 'flip':
                direction = 'DOWN' if direction == 'UP' else 'UP'
            else:
                direction = _v3_early_config.direction_override  # e.g., 'UP' for ETH/DDOG
            if direction != _v3_original_direction:
                LOGGER.info(
                    f"[{symbol}] 🔄 V3 EARLY OVERRIDE: {_v3_original_direction} → {direction} "
                    f"(strategy: {_v3_early_config.strategy}, override: {_v3_early_config.direction_override})"
                )

        # Wire to in-memory store for /api/cockpit consumption
        # If calibration says "don't predict", keep the prediction for monitoring but mark HOLD.
        action = "BUY" if direction == "UP" else "SELL" if direction == "DOWN" else "HOLD"
        if not should_predict:
            action = "HOLD"

        from core.asset_classification import is_crypto_symbol as _is_crypto_symbol

        # MONEY GAME: Store ALL predictions - rankings determine TOP 10
        LOGGER.info(f"[MONEY-GAME] ✅ STORING {symbol} in _LATEST_PREDICTIONS (all symbols compete!)")
        
        # BUG FIX (Jan 6, 2026): Use lock to prevent race conditions
        with _LATEST_PREDICTIONS_LOCK:
            _LATEST_PREDICTIONS[symbol] = {
                "prediction_id": prediction_id,
                "symbol": symbol,
                "run_at": run_at,  # Store as float timestamp
                "confidence": confidence,
                "direction": direction,
                "action": action,  # For autonomous execution / telegram gating
                "horizon_h": horizon_h,
                "provider": price_provider,
                "price": current_price,  # For trade decision engine
                "price_at_prediction": current_price,
                "market": "crypto" if _is_crypto_symbol(symbol) else "stock",  # Market type
                "engine": "turbo",  # FIX (Feb 24): was missing, cockpit defaulted to phantom "turbo"
                "intel_applied": bool(intel_metadata.get("intel_applied")),  # FIX (Feb 24): was missing for crypto, always showed False
                "confirmations": intel_metadata.get("confirmations", 0),  # FIX (Feb 24): was missing
                "feature_status": feature_status.to_dict(),
                "confidence_metadata": confidence_metadata,
                "should_predict": bool(should_predict),
                "momentum": momentum_data,  # Add momentum tracking data
                # Intelligence Hub metadata
                **_turbo_hub_meta,
            }

            if expected_move_pct is not None:
                _LATEST_PREDICTIONS[symbol]["expected_move_pct"] = expected_move_pct

        # ==========================================================================
        # TRADE PARAMETERS (FIXED Dec 21, 2025)
        # 
        # Realistic stop losses: 2-4% (not 12%!)
        # For BTC at $88k, a 4% stop = $3,500 loss (reasonable)
        # NOT $11,000 loss from 12.5% stop
        #
        # Risk/Reward: Target 2:1 ratio
        #   - Stop: 2-3% from entry
        #   - Target: 4-6% from entry
        # ==========================================================================
        entry_price = current_price
        
        # ==========================================================================
        # ASSET-AWARE STOP LOSSES (Dec 22, 2025)
        # Stops should match asset type AND horizon
        # - Crypto: 4.5% stop (volatile)
        # - Large cap stocks: 2% stop (stable)
        # - Volatile stocks: 4% stop (TSLA, NVDA)
        # ==========================================================================
        abs_expected_move = abs(expected_move_pct) if expected_move_pct else 3.0
        
        # Use AssetClassifier for proper stop sizing
        try:
            from core.asset_classifier import get_target_stop
            asset_stops = get_target_stop(symbol, horizon_h)
            stop_loss_pct = asset_stops['stop_pct']
        except Exception:
            # Fallback if classifier fails
            if horizon_h <= 6:
                stop_loss_pct = 1.5
            elif horizon_h <= 24:
                stop_loss_pct = 3.0
            else:
                stop_loss_pct = 4.5
        
        target_pct = abs_expected_move  # Target at full expected move
        
        if direction == "UP":
            # UP: stop below entry, target above entry
            stop_loss = round(entry_price * (1 - stop_loss_pct / 100), 4)
            take_profit = round(entry_price * (1 + target_pct / 100), 4)
            target_price = take_profit
        elif direction == "DOWN":
            # DOWN: stop above entry, target below entry
            stop_loss = round(entry_price * (1 + stop_loss_pct / 100), 4)
            take_profit = round(entry_price * (1 - target_pct / 100), 4)
            target_price = take_profit
        else:
            # FLAT/HOLD: neutral positioning
            stop_loss = round(entry_price * 0.98, 4)  # -2% stop
            take_profit = round(entry_price * 1.02, 4)  # +2% target
            target_price = entry_price
        
        LOGGER.debug(
            f"[{symbol}] Trade params: direction={direction}, entry=${entry_price:.4f}, "
            f"target=${target_price:.4f} ({target_pct:+.1f}%), "
            f"stop=${stop_loss:.4f} ({-stop_loss_pct if direction == 'UP' else stop_loss_pct:+.1f}%)"
        )

        # Touch-target calibration + gating (Stage 5/6)
        try:
            from core.touch_calibration_sqlite import calibrate_touch_confidence

            touch_cal = calibrate_touch_confidence(symbol, confidence)
            _LATEST_PREDICTIONS[symbol].update(
                {
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "target_price": target_price,  # Already calculated correctly per direction
                    "touch_calibrated_1pct": touch_cal.calibrated_1pct,
                    "touch_calibrated_0_5pct": touch_cal.calibrated_0_5pct,
                    "touch_calibration_samples": touch_cal.sample_size,
                    "touch_conf_band": touch_cal.band,
                    "stage5_ok": touch_cal.stage5_ok,
                    "stage6_ok": touch_cal.stage6_ok,
                    "gate": touch_cal.gate,
                }
            )
        except Exception:
            _LATEST_PREDICTIONS[symbol].update(
                {
                    "entry_price": entry_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "target_price": target_price,  # Already calculated correctly per direction
                    "touch_calibrated_1pct": None,
                    "touch_calibrated_0_5pct": None,
                    "touch_calibration_samples": 0,
                    "touch_conf_band": None,
                    "stage5_ok": False,
                    "stage6_ok": False,
                    "gate": "MONITOR",
                }
            )

        # Register prediction for accuracy tracking (48h evaluation) + FEEDBACK LOOP (Task #4)
        try:
            from core.accuracy_tracker import get_accuracy_tracker
            from core.feedback_loop import get_feedback_loop
            
            tracker = get_accuracy_tracker()
            feedback = get_feedback_loop()
            
            # Record forecast for accuracy tracking
            forecast_id = tracker.record_forecast(
                symbol=symbol,
                direction=direction,
                confidence=confidence,
                entry_price=current_price,
                forecast_price=current_price,
                forecast_horizon_hours=horizon_h,
                model_version="ghost_v3_pillars",
                metadata={
                    "prediction_id": prediction_id,
                    "direction": direction,
                    "signals": signals_fired,
                    "feature_count": feature_data["feature_count"]
                }
            )
            
            # Apply learned feature weights for next prediction
            # (This continuously improves accuracy as the system learns)
            adjusted_features = feedback.get_adjusted_features(features)
            if adjusted_features != features:
                LOGGER.debug(f"[{symbol}] 🔄 Applied feedback loop feature adjustments")
            
            LOGGER.debug(f"[{symbol}] Registered for accuracy tracking (ID={forecast_id}, 48h evaluation)")
        except Exception as e:
            LOGGER.warning(f"[{symbol}] Accuracy tracking registration failed: {e}")

        # ================================================================
        # PREDICTION GATE (Step 3): Kill switch + confidence floor + rate limit
        # ================================================================
        _turbo_gate_ok, _turbo_gate_reason = should_create_prediction(symbol, confidence)
        if not _turbo_gate_ok:
            LOGGER.info(f"[{symbol}] 🚫 Turbo prediction blocked: {_turbo_gate_reason}")

        # ALSO write to ghost_predictions table for touch-target evaluation + UI
        if _turbo_gate_ok:
          try:
            import sqlite3
            # Ensure tables exist (and include newer columns like features_json)
            from core import prediction_tracker as _pt  # noqa: F401

            db_path = WOLF_SQLITE_PATH
            conn = sqlite3.connect(db_path)

            # Ensure schema supports touch-target + gating columns + features
            _touch_cols = [
                ("touch_calibrated_1pct", "REAL"),
                ("touch_calibrated_0_5pct", "REAL"),
                ("touch_calibration_samples", "INTEGER DEFAULT 0"),
                ("touch_conf_band", "TEXT"),
                ("features_json", "TEXT"),
            ]
            for _col_name, _col_type in _touch_cols:
                try:
                    conn.execute(f"ALTER TABLE ghost_predictions ADD COLUMN {_col_name} {_col_type}")
                except Exception:
                    pass  # Column already exists

            # Align target with returned trade params
            # CRITICAL FIX (Mar 4, 2026): Was using stop_loss for DOWN predictions.
            # stop_loss is ABOVE entry for DOWN (to limit losses), so evaluator
            # computed direction_from_delta(stop_loss - entry) = UP, but predicted_direction = DOWN
            # → dir_ok=0 on EVERY DOWN prediction. target_price is correct for both directions.
            predicted_price = target_price
            # Use model-derived expected move if available; otherwise fall back to TP/SL derived pct.
            if expected_move_pct is not None:
                predicted_pct = float(expected_move_pct)
            else:
                predicted_pct = ((predicted_price - current_price) / current_price) * 100 if current_price else 0.0
            
            # Store features as JSON for ML training
            import json
            features_json = json.dumps(features)
            
            conn.execute("""
                INSERT INTO ghost_predictions (
                    symbol, predicted_at, check_at, predicted_price, 
                    predicted_direction, predicted_pct, confidence, timeframe_hours, 
                    current_price, target_price, stage5_ok, stage6_ok, gate,
                    touch_calibrated_1pct, touch_calibrated_0_5pct, touch_calibration_samples, touch_conf_band,
                    checked, features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                symbol,
                int(run_at),
                int(run_at + (horizon_h * 3600)),
                predicted_price,
                direction,
                float(predicted_pct),
                confidence,
                horizon_h,
                current_price,
                float(_LATEST_PREDICTIONS[symbol].get("target_price") or predicted_price),
                1 if _LATEST_PREDICTIONS[symbol].get("stage5_ok") else 0,
                1 if _LATEST_PREDICTIONS[symbol].get("stage6_ok") else 0,
                str(_LATEST_PREDICTIONS[symbol].get("gate") or "MONITOR"),
                _LATEST_PREDICTIONS[symbol].get("touch_calibrated_1pct"),
                _LATEST_PREDICTIONS[symbol].get("touch_calibrated_0_5pct"),
                int(_LATEST_PREDICTIONS[symbol].get("touch_calibration_samples") or 0),
                _LATEST_PREDICTIONS[symbol].get("touch_conf_band"),
                0,
                features_json
            ))
            conn.commit()
            conn.close()
            LOGGER.info(f"[{symbol}] Stored in ghost_predictions table (ID={prediction_id}, direction={direction}, confidence={confidence:.1%}, features={len(features)})")
          except Exception as e:
            LOGGER.error(f"[{symbol}] Failed to write to ghost_predictions table: {e}")

        # ================================================================
        # ALSO write to PostgreSQL ghost_predictions (Feb 24, 2026)
        # The evaluator now reads from PostgreSQL, so predictions MUST
        # exist there for evaluation to work.
        # ================================================================
        if _turbo_gate_ok:
         try:
            from core.db_pool import get_sync_connection as _pg_get_conn
            # CRITICAL FIX (Mar 4, 2026): Use target_price (not stop_loss for DOWN)
            _predicted_price = target_price
            _predicted_pct = float(expected_move_pct) if expected_move_pct is not None else (
                ((float(_predicted_price) - current_price) / current_price) * 100 if current_price else 0.0
            )

            # ── GHOST LEARNING BRAIN — DISABLED (Step 6, Mar 18 2026) ──
            # Same as stock engine: double-flip with Brain v3 caused
            # predictions to flip back to the original wrong direction.
            # Kill switch (Step 3) handles bad symbols by blocking, not flipping.
            _was_inverted_turbo = False

            # ── DIRECTION CONSISTENCY GUARD (Mar 10, 2026) ──────────────
            # Ensure predicted_direction matches target vs entry price.
            # FIX (Mar 12, 2026): Also update 'direction' variable and
            # _LATEST_PREDICTIONS so Telegram/paper trades get corrected direction.
            _turbo_dir_for_pg = direction
            if current_price and _predicted_price:
                if float(_predicted_price) > current_price and direction == "DOWN":
                    _turbo_dir_for_pg = "UP"
                    direction = "UP"
                    with _LATEST_PREDICTIONS_LOCK:
                        if symbol in _LATEST_PREDICTIONS:
                            _LATEST_PREDICTIONS[symbol]["direction"] = "UP"
                            _LATEST_PREDICTIONS[symbol]["action"] = "BUY"
                    LOGGER.warning(f"[{symbol}] ⚠️ Turbo direction consistency fix: target {_predicted_price} > entry {current_price} but dir was DOWN → UP")
                elif float(_predicted_price) < current_price and direction == "UP":
                    _turbo_dir_for_pg = "DOWN"
                    direction = "DOWN"
                    with _LATEST_PREDICTIONS_LOCK:
                        if symbol in _LATEST_PREDICTIONS:
                            _LATEST_PREDICTIONS[symbol]["direction"] = "DOWN"
                            _LATEST_PREDICTIONS[symbol]["action"] = "SELL"
                    LOGGER.warning(f"[{symbol}] ⚠️ Turbo direction consistency fix: target {_predicted_price} < entry {current_price} but dir was UP → DOWN")

            import json as _pg_json
            _features_json = _pg_json.dumps(features)
            with _pg_get_conn() as _pg_conn:
                _pg_cur = _pg_conn.cursor()
                _pg_cur.execute("""
                    INSERT INTO ghost_predictions (
                        symbol, predicted_at, check_at, predicted_price,
                        predicted_direction, predicted_pct, confidence, timeframe_hours,
                        current_price, target_price, gate, checked, features_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, predicted_at) DO NOTHING
                """, (
                    symbol,
                    int(run_at),
                    int(run_at + (horizon_h * 3600)),
                    float(_predicted_price),
                    _turbo_dir_for_pg,
                    float(_predicted_pct),
                    confidence,
                    horizon_h,
                    current_price,
                    float(_predicted_price),
                    str(_LATEST_PREDICTIONS.get(symbol, {}).get("gate") or "MONITOR"),
                    0,
                    _features_json,
                ))
                _pg_conn.commit()
            LOGGER.info(f"[{symbol}] ✅ PostgreSQL ghost_predictions stored")
         except Exception as _pg_err:
            LOGGER.warning(f"[{symbol}] PostgreSQL ghost_predictions write failed (non-fatal): {_pg_err}")
        
        # ========================================================================
        # AUTO-LOG PAPER TRADE: Track all predictions for paper trading P&L
        # Only log directional predictions (UP/DOWN) with minimum confidence
        # Threshold 0.55: Only log predictions where model has meaningful conviction
        # Previous 0.30 threshold flooded with garbage (15.6% win rate over 7 days)
        # Also checks symbol's recent win rate — skip symbols with <20% historical accuracy
        # ========================================================================
        
        # =====================================================================
        # EDGE SYMBOL WHITELIST (Feb 9, 2026)
        # 30-day data analysis: 24 symbols have ≥50% WR (74.7% combined, 743W/252L)
        # The other 76 symbols have 21.5% WR (256W/933L) — pure destruction
        # Only trade symbols with PROVEN edge. Updated via env var for easy tuning.
        # =====================================================================
        _PAPER_TRADE_MIN_CONFIDENCE = float(os.getenv("PAPER_TRADE_MIN_CONFIDENCE", "0.62"))
        EDGE_SYMBOLS = get_edge_set()
        _EDGE_WHITELIST_ENABLED = os.getenv("EDGE_WHITELIST_ENABLED", "1") == "1"
        
        # Import V3 strategies early — needed for confidence gate bypass
        from config.symbols import V3_VALIDATED_STRATEGIES
        
        if _EDGE_WHITELIST_ENABLED and symbol.upper() not in EDGE_SYMBOLS:
            LOGGER.info(
                f"[{symbol}] 🚫 EDGE WHITELIST: Symbol not in {len(EDGE_SYMBOLS)} proven edge symbols — skipping paper trade"
            )
            # Still return the prediction, just don't log a paper trade
        elif direction == "HOLD":
            # HOLD ZONE: Model has no conviction — don't paper trade
            LOGGER.info(
                f"[{symbol}] 🛑 HOLD ZONE: XGBoost near coin-flip — no paper trade logged"
            )
        elif direction in ["UP", "DOWN"] and (confidence >= _PAPER_TRADE_MIN_CONFIDENCE or symbol.upper() in V3_VALIDATED_STRATEGIES):
            try:
                from core.paper_tracker import get_paper_tracker
                paper_tracker = get_paper_tracker()
                
                # QUALITY GATE: Check symbol's recent win rate — skip consistently losing symbols
                # This prevents flooding paper_trades with predictions for symbols where model has no edge
                _PAPER_TRADE_MIN_SYMBOL_WINRATE = float(os.getenv("PAPER_TRADE_MIN_SYMBOL_WINRATE", "0.48"))
                _PAPER_TRADE_MIN_SYMBOL_TRADES = int(os.getenv("PAPER_TRADE_MIN_SYMBOL_TRADES", "8"))
                conn_qg = None
                try:
                    conn_qg = paper_tracker._get_connection()
                    qg_cutoff = (datetime.utcnow() - timedelta(days=14)).isoformat()
                    cur_qg = paper_tracker._execute(conn_qg,
                        """SELECT 
                            COUNT(*) as total,
                            SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                            SUM(CASE WHEN outcome IN ('LOSS', 'STOPPED') THEN 1 ELSE 0 END) as losses
                        FROM paper_trades 
                        WHERE symbol = ? AND entry_time > ? AND outcome IS NOT NULL AND outcome != 'PENDING'
                        """,
                        (symbol.upper(), qg_cutoff)
                    )
                    qg_row = paper_tracker._fetchall(cur_qg)
                    
                    if qg_row and qg_row[0]:
                        qg_total = qg_row[0].get("total", 0) or 0
                        qg_wins = qg_row[0].get("wins", 0) or 0
                        qg_losses = qg_row[0].get("losses", 0) or 0
                        qg_resolved = qg_wins + qg_losses
                        
                        if qg_resolved >= _PAPER_TRADE_MIN_SYMBOL_TRADES:
                            qg_winrate = qg_wins / qg_resolved if qg_resolved > 0 else 0
                            if qg_winrate < _PAPER_TRADE_MIN_SYMBOL_WINRATE:
                                LOGGER.info(f"[{symbol}] 🚫 QUALITY GATE: {qg_winrate:.1%} win rate ({qg_wins}W/{qg_losses}L in 14d) < {_PAPER_TRADE_MIN_SYMBOL_WINRATE:.0%} min — skipping paper trade")
                                raise Exception("quality_gate_skip")
                except Exception as qg_err:
                    if "quality_gate_skip" in str(qg_err):
                        raise  # Re-raise to hit the outer except
                    LOGGER.debug(f"[{symbol}] Quality gate check failed (continuing): {qg_err}")
                finally:
                    if conn_qg is not None:
                        try:
                            conn_qg.close()
                        except Exception:
                            pass
                
                # DEDUP: Now centralized in paper_tracker.log_signal() — catches ALL callers
                # (run_prediction, stock_engine, ghost_notifications, cascade)
                
                # Check if symbol is V3 validated
                v3_config = V3_VALIDATED_STRATEGIES.get(symbol.upper())
                v3_validated = v3_config is not None
                v3_strategy = v3_config.strategy if v3_config else None
                v3_hold_hours = v3_config.hold_hours if v3_config else None
                v3_win_rate = v3_config.backtest_win_rate if v3_config else None
                v3_is_inverse = (v3_config.strategy == 'ghost_inverse') if v3_config else False
                
                # V3 direction override already applied at L10095 (early override).
                # Use the original direction captured there.
                v3_original_direction = _v3_original_direction
                
                # PRICE SANITY CHECK: Reject clearly garbage entry prices
                # JUP was logged at $0.00048679 when real price is ~$0.50-1.00 (390,000% PnL artifact)
                # Any price below $0.00001 or above $1M is almost certainly bad data
                if current_price is None or current_price <= 0:
                    LOGGER.warning(f"[{symbol}] 🚫 PRICE SANITY: entry_price is {current_price} — skipping paper trade")
                    raise Exception("price_sanity_skip")
                if current_price < 0.00001:
                    LOGGER.warning(f"[{symbol}] 🚫 PRICE SANITY: entry_price ${current_price} suspiciously low — skipping paper trade")
                    raise Exception("price_sanity_skip")
                if current_price > 1_000_000:
                    LOGGER.warning(f"[{symbol}] 🚫 PRICE SANITY: entry_price ${current_price:,.2f} suspiciously high — skipping paper trade")
                    raise Exception("price_sanity_skip")
                
                # =====================================================================
                # POSITION SIZER: Kelly Criterion + ATR sizing (Feb 25, 2026)
                # Replace fixed $1000 with intelligent position sizing based on:
                #   - Kelly Criterion (optimal bet sizing for edge)
                #   - ATR (volatility-based stop losses)
                #   - Symbol win rate (data-driven from paper_trades)
                # Default $1000 fallback if sizer fails.
                # =====================================================================
                _position_size = 1000.0  # Default fallback
                try:
                    if os.getenv("POSITION_SIZER_ENABLED", "1") == "1":
                        from core.position_sizer import get_position_sizer
                        
                        _sizer = get_position_sizer(capital=float(os.getenv("PAPER_TRADE_CAPITAL", "100000")))
                        _atr = features.get("ATR_PERCENT", features.get("ATR", 3.0)) or 3.0
                        _atr_dollar = current_price * float(_atr) / 100.0 if float(_atr) < 50 else float(_atr)
                        
                        # Get symbol's historical win rate from paper trades
                        _sym_wr = 0.55  # Default
                        conn_wr = None
                        try:
                            conn_wr = paper_tracker._get_connection()
                            cur_wr = paper_tracker._execute(conn_wr,
                                """SELECT SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END) as w,
                                          SUM(CASE WHEN outcome IN ('LOSS','STOPPED') THEN 1 ELSE 0 END) as l
                                   FROM paper_trades WHERE symbol=? AND outcome IS NOT NULL AND outcome != 'PENDING'""",
                                (symbol.upper(),)
                            )
                            _wr_row = paper_tracker._fetchall(cur_wr)
                            if _wr_row and _wr_row[0]:
                                _w = _wr_row[0].get("w", 0) or 0
                                _l = _wr_row[0].get("l", 0) or 0
                                if (_w + _l) >= 5:
                                    _sym_wr = _w / (_w + _l)
                        except Exception:
                            pass
                        finally:
                            if conn_wr is not None:
                                try:
                                    conn_wr.close()
                                except Exception:
                                    pass
                        
                        _ps = _sizer.calculate_position_size(
                            symbol=symbol,
                            entry_price=current_price,
                            confidence=confidence,
                            atr=_atr_dollar,
                            win_rate=_sym_wr,
                        )
                        _position_size = max(_ps.dollar_amount, 100.0)  # Min $100
                        _position_size = min(_position_size, 10000.0)  # Cap $10k
                        LOGGER.info(
                            f"[{symbol}] 💰 POSITION SIZER: ${_position_size:,.0f} "
                            f"(Kelly={_ps.kelly_fraction:.1%}, WR={_sym_wr:.0%}, "
                            f"R:R={_ps.risk_reward_ratio:.1f}:1)"
                        )
                except Exception as e:
                    LOGGER.debug(f"[{symbol}] Position sizer failed, using $1000 default: {e}")
                    _position_size = 1000.0
                
                paper_trade_id = paper_tracker.log_signal(
                    cascade_id=f"pred_{prediction_id}",  # Link to prediction
                    symbol=symbol,
                    signal_direction=direction,
                    signal_confidence=confidence,
                    entry_price=current_price,
                    entry_time=datetime.utcfromtimestamp(run_at).isoformat(),
                    position_size=_position_size,  # Kelly-optimal sizing (was fixed $1000)
                    stop_loss_pct=stop_loss_pct,
                    take_profit_pct=abs(expected_move_pct or 3.0) / 100.0,
                    # V3 tracking metadata
                    v3_validated=v3_validated,
                    v3_strategy=v3_strategy,
                    v3_hold_hours=v3_hold_hours,
                    v3_backtest_win_rate=v3_win_rate,
                    v3_is_inverse=v3_is_inverse,
                    v3_original_direction=v3_original_direction,
                    expected_move_pct=expected_move_pct,  # FIX: was missing — target_price needs this
                )
                v3_tag = f" [V3: {v3_strategy}]" if v3_validated else ""
                LOGGER.info(f"[{symbol}] 📝 Paper trade auto-logged: {paper_trade_id} ({direction} @ ${current_price:,.2f}){v3_tag}")
            except Exception as e:
                skip_reasons = ["quality_gate_skip", "price_sanity_skip"]
                if any(reason in str(e) for reason in skip_reasons):
                    pass  # Intentional skip — already logged above
                else:
                    LOGGER.warning(f"[{symbol}] Paper trade logging failed (non-fatal): {e}")
        
        # Calculate total duration
        duration_ms = int((time.monotonic() - start) * 1000)

        return {
            "ok": True,
            "prediction_id": prediction_id,
            "symbol": symbol,
            "run_at": int(run_at * 1000),  # Convert to milliseconds for JavaScript
            "horizon_h": horizon_h,
            "confidence": confidence,
            "direction": direction,
            "current_price": current_price,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "target_price": target_price,  # Already calculated correctly per direction
            "stage5_ok": bool(_LATEST_PREDICTIONS[symbol].get("stage5_ok")),
            "stage6_ok": bool(_LATEST_PREDICTIONS[symbol].get("stage6_ok")),
            "gate": _LATEST_PREDICTIONS[symbol].get("gate", "MONITOR"),
            "touch_calibrated_1pct": _LATEST_PREDICTIONS[symbol].get("touch_calibrated_1pct"),
            "touch_calibrated_0_5pct": _LATEST_PREDICTIONS[symbol].get("touch_calibrated_0_5pct"),
            "expected_move_pct": expected_move_pct,
            "reward_risk_ratio": round(abs(expected_move_pct or 3.0) / stop_loss_pct, 2) if stop_loss_pct else 2.0,  # Dynamic R:R
            "feature_count": feature_data["feature_count"],
            "available_count": feature_data["available_count"],
            "duration_ms": duration_ms,
            "momentum": momentum_data,  # Add momentum to API response
            "xgb_debug": xgb_debug if 'xgb_debug' in locals() else {},  # Raw XGBoost probabilities for debugging
        }

    except Exception as e:
        # Catch ALL exceptions and return structured error (never hang)
        duration_ms = int((time.monotonic() - start) * 1000)
        LOGGER.error(f"Prediction run failed for {symbol}: {e}", exc_info=True)
        
        return {
            "ok": False,
            "symbol": symbol,
            "direction": "ERROR",
            "confidence": 0.0,
            "current_price": None,
            "feature_count": 0,
            "available_count": 0,
            "duration_ms": duration_ms,
            "error": str(e)[:200]
        }


async def api_predict_run(
    body: _PredictRunBody,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    Generate a new 48h prediction for a stock symbol using live data.
    Returns prediction metadata.
    
    This is the HTTP handler that wraps run_single_prediction.
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    symbol = body.symbol.upper().strip()
    if not symbol:
        raise HTTPException(400, "symbol required")
    
    # Call synchronous core function
    result = run_single_prediction(symbol)
    
    # If prediction failed, raise HTTP error
    if not result.get("ok"):
        error = result.get("error", "Unknown error")
        duration_ms = result.get("duration_ms", 0)
        LOGGER.error(
            f"[{symbol}] Prediction failed: {error} ({duration_ms}ms)",
            extra={"symbol": symbol, "error": error, "duration_ms": duration_ms}
        )
        raise HTTPException(500, f"Prediction failed: {error}")
    
    return result


async def _gather_opus_context(symbol: str) -> dict:
    """Gather all available context for Claude to analyze"""
    context = {
        "current_price": None,
        "price_change_24h": None,
        "news_headlines": [],
        "insider_activity": "None detected",
        "whale_activity": "None detected", 
        "social_sentiment": "Unknown",
        "upcoming_events": [],
        "historical_pattern": "Unknown",
        "volume_analysis": "Unknown",
        # NEW: Ghost Brain intelligence
        "ghost_brain_signal": "UNKNOWN",
        "ghost_brain_confidence_adj": 0,
        "dominant_narrative": "Unknown",
        "narrative_sentiment": "Unknown",
        "influencer_activity": [],
        "warnings": [],
        "positives": [],
    }
    
    # Determine if crypto
    crypto_symbols = {"BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "BNB", "DOT", 
                     "LINK", "AVAX", "MATIC", "BCH", "LTC", "ZEC", "METIS", 
                     "LRC", "CHZ", "NEAR", "APT", "ARB"}
    is_crypto = symbol.upper() in crypto_symbols
    
    # === NEW: Get Ghost Brain intelligence first ===
    try:
        from core.intelligence.ghost_brain import get_ghost_brain
        brain = get_ghost_brain()
        LOGGER.info(f"[OPUS] Calling Ghost Brain for {symbol}...")
        brain_result = await brain.full_analysis(symbol.upper())
        LOGGER.info(f"[OPUS] Ghost Brain result ok={brain_result.get('ok')}, signal={brain_result.get('overall_signal')}")
        
        if brain_result.get("ok"):
            context["ghost_brain_signal"] = brain_result.get("overall_signal", "UNKNOWN")
            context["ghost_brain_confidence_adj"] = brain_result.get("confidence_adjustment", 0)
            context["warnings"] = brain_result.get("warnings", [])
            context["positives"] = brain_result.get("positives", [])
            
            # Narratives
            narratives = brain_result.get("narratives", {})
            if narratives.get("dominant_narrative"):
                dom = narratives["dominant_narrative"]
                context["dominant_narrative"] = dom.get("name", "Unknown")
                context["narrative_sentiment"] = dom.get("sentiment", "Unknown")
                LOGGER.info(f"[OPUS] Narrative: {context['dominant_narrative']} ({context['narrative_sentiment']})")
            
            # Influencers
            influencers = brain_result.get("influencers", {})
            if influencers.get("mentions"):
                context["influencer_activity"] = [
                    f"{m.get('influencer', {}).get('name', '?')}: {m.get('sentiment', '?')}"
                    for m in influencers.get("mentions", [])[:3]
                ]
                LOGGER.info(f"[OPUS] Influencers: {context['influencer_activity']}")
            
            # Volume from micro signals
            micro = brain_result.get("micro_signals", {})
            vol = micro.get("signals", {}).get("volume", {})
            if vol.get("has_data"):
                metrics = vol.get("metrics", {})
                ratio = metrics.get("volume_ratio", 1)
                trend = metrics.get("volume_trend", "FLAT")
                price_trend = metrics.get("price_trend", "FLAT")
                context["volume_analysis"] = f"{ratio:.1f}x average, volume {trend}, price {price_trend}"
            
            # Seasonal
            seasonal = brain_result.get("seasonal", {})
            special = seasonal.get("special_period", {})
            pattern = seasonal.get("pattern", {})
            if special:
                context["historical_pattern"] = f"{special.get('name', 'Unknown')}: {special.get('tendency', 'Unknown')}"
            elif pattern:
                context["historical_pattern"] = f"{pattern.get('name', 'Unknown')}: {pattern.get('tendency', 'Unknown')}"
            
            LOGGER.info(f"[OPUS] Ghost Brain loaded: {context['ghost_brain_signal']} ({context['ghost_brain_confidence_adj']:+}%), narrative={context['dominant_narrative']}")
        else:
            LOGGER.warning(f"[OPUS] Ghost Brain returned ok=False for {symbol}")
    except Exception as e:
        LOGGER.error(f"[OPUS] Ghost Brain fetch FAILED for {symbol}: {e}", exc_info=True)
    
    # Get current price
    try:
        if is_crypto:
            from core.crypto.crypto_providers import get_crypto_price_turbo
            price_data = await get_crypto_price_turbo(symbol.upper())
            if price_data:
                context["current_price"] = price_data.get("price")
                context["price_change_24h"] = price_data.get("change_pct")
        else:
            price_data = await get_stock_price(symbol.upper())
            if price_data:
                context["current_price"] = price_data.get("price") or price_data.get("current_price")
                context["price_change_24h"] = price_data.get("change_pct") or price_data.get("changePercent")
    except Exception as e:
        LOGGER.debug(f"Price fetch failed for {symbol}: {e}")
    
    # Get news headlines
    try:
        polygon_key = os.getenv("POLYGON_API_KEY", "")
        if polygon_key:
            async with aiohttp.ClientSession() as session:
                url = f"https://api.polygon.io/v2/reference/news?ticker={symbol}&limit=10&apiKey={polygon_key}"
                async with session.get(url, timeout=5) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        context["news_headlines"] = [
                            r.get("title", "") for r in data.get("results", [])
                        ]
    except Exception as e:
        LOGGER.debug(f"News fetch failed for {symbol}: {e}")
    
    # Get micro signals if available
    try:
        from core.intelligence.micro_signals.micro_aggregator import scan_micro_signals
        micro = await scan_micro_signals(symbol.upper(), is_crypto)
        
        signals = micro.get("signals", {})
        
        if signals.get("insider", {}).get("has_data"):
            insider = signals["insider"]
            context["insider_activity"] = f"{insider.get('signal', 'Unknown')} signal"
        
        if signals.get("whale", {}).get("has_data"):
            whale = signals["whale"]
            net_flow = whale.get("flows", {}).get("net_flow_usd", 0)
            direction = "inflow" if net_flow < 0 else "outflow"
            context["whale_activity"] = f"{whale.get('signal', 'Unknown')} - ${abs(net_flow):,.0f} net {direction}"
        
        if signals.get("social", {}).get("has_data"):
            social = signals["social"]
            sentiment = social.get("metrics", {}).get("santiment", {}).get("sentiment_label")
            if sentiment:
                context["social_sentiment"] = sentiment
        
        if signals.get("volume", {}).get("has_data"):
            vol = signals["volume"]
            ratio = vol.get("metrics", {}).get("volume_ratio", 1)
            trend = vol.get("metrics", {}).get("volume_trend", "FLAT")
            context["volume_analysis"] = f"{ratio:.1f}x average, trend: {trend}"
            
    except Exception as e:
        LOGGER.debug(f"Micro signals fetch failed for {symbol}: {e}")
    
    # Get seasonal pattern
    try:
        from core.intelligence.historical.event_outcomes import get_event_database
        seasonal = await get_event_database().get_seasonal_pattern(symbol.upper())
        pattern = seasonal.get("pattern", {})
        special = seasonal.get("special_period", {})
        
        if special:
            context["historical_pattern"] = f"Special: {special.get('name', 'Unknown')} ({special.get('tendency', 'Unknown')})"
        elif pattern:
            context["historical_pattern"] = f"{pattern.get('name', 'Unknown')}: {pattern.get('tendency', 'Unknown')} - {pattern.get('note', '')}"
    except Exception as e:
        LOGGER.debug(f"Seasonal fetch failed for {symbol}: {e}")
    
    return context


def _get_watchlist_lock() -> asyncio.Lock:
    global _WATCHLIST_ENRICHED_LOCK
    if _WATCHLIST_ENRICHED_LOCK is None:
        _WATCHLIST_ENRICHED_LOCK = asyncio.Lock()
    return _WATCHLIST_ENRICHED_LOCK


async def _api_v3_watchlist_enriched_core():
    """Core watchlist logic - wrapped with timeout in main endpoint."""
    try:
        watchlist_data = []
        
        # Configurable watchlist size (default: 50, expandable to 200+)
        watchlist_limit = int(os.getenv("WATCHLIST_DISPLAY_LIMIT", "50"))

        if _LATEST_PREDICTIONS:
            sorted_preds = sorted(
                _LATEST_PREDICTIONS.values(),
                key=lambda p: p.get("run_at", 0),
                reverse=True,
            )
            deduped = []
            for pred in sorted_preds:
                symbol = pred.get("symbol")
                if not symbol or symbol in deduped:
                    continue
                deduped.append(symbol)
                if len(deduped) >= watchlist_limit:
                    break
            symbols_to_check = deduped
        else:
            symbols_to_check = STOCK_SYMBOLS[:10] + CRYPTO_SYMBOLS[:10]
        
        # FIX (Step 8, Mar 18 2026): If _LATEST_PREDICTIONS has < 20 symbols,
        # supplement with a default set so the watchlist isn't empty after deploys.
        # Previously only LINK showed up because it was the only symbol with a
        # recent prediction. Now we always show at least 20 symbols.
        _DEFAULT_WATCHLIST_SYMBOLS = [
            # Top 10 crypto
            "BTC", "ETH", "XRP", "SOL", "ADA", "LINK", "DOT", "AVAX", "DOGE", "MATIC",
            # Top 10 stocks Ghost has historical data for
            "T", "BMBL", "NET", "DDOG", "FTNT", "XPO", "PANW", "AAPL", "MSFT", "NVDA",
        ]
        if len(symbols_to_check) < 20:
            for sym in _DEFAULT_WATCHLIST_SYMBOLS:
                if sym not in symbols_to_check:
                    symbols_to_check.append(sym)
                if len(symbols_to_check) >= 20:
                    break
        
        # PERFORMANCE FIX (Jan 29, 2026): Separate crypto (batch) from stocks (parallel)
        # This reduces crypto fetch from 5s*N (with rate limit) to 1 batch call
        crypto_symbols = [s for s in symbols_to_check if s.upper() in CRYPTO_SYMBOLS]
        stock_symbols = [s for s in symbols_to_check if s.upper() not in CRYPTO_SYMBOLS]
        
        # Batch fetch ALL crypto prices in ONE call (huge performance win!)
        # Apply 10s timeout to prevent hanging
        crypto_prices = {}
        if crypto_symbols:
            try:
                from core.crypto.crypto_providers import get_crypto_prices_batch
                crypto_prices = await asyncio.wait_for(
                    get_crypto_prices_batch(crypto_symbols, use_cache=True),
                    timeout=10.0
                )
                LOGGER.info(f"Watchlist batch crypto: {len(crypto_prices)}/{len(crypto_symbols)} prices")
            except asyncio.TimeoutError:
                LOGGER.warning(f"Crypto batch TIMEOUT after 10s for {len(crypto_symbols)} symbols")
            except Exception as e:
                LOGGER.warning(f"Crypto batch failed, falling back: {e}")
        
        # Fetch stock prices in parallel (existing behavior) with timeout
        stock_price_tasks = [_fetch_symbol_price(s) for s in stock_symbols]
        try:
            stock_results = await asyncio.wait_for(
                asyncio.gather(*stock_price_tasks, return_exceptions=True),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            LOGGER.warning(f"Stock price fetch TIMEOUT after 5s for {len(stock_symbols)} symbols")
            stock_results = [None] * len(stock_symbols)
        
        stock_prices = {}
        for sym, result in zip(stock_symbols, stock_results, strict=False):
            if result and not isinstance(result, Exception):
                stock_prices[sym.upper()] = result
        
        # Merge prices
        all_prices = {**stock_prices}
        for sym, data in crypto_prices.items():
            all_prices[sym.upper()] = {
                "price": data.get("price"),
                "change_pct": data.get("change_24h_pct")
            }
        
        # Build watchlist data
        for symbol in symbols_to_check:
            try:
                price_result = all_prices.get(symbol.upper(), {})
                price = price_result.get("price")
                change_pct = price_result.get("change_pct")
                
                # Get latest prediction
                pred = _LATEST_PREDICTIONS.get(symbol, {})
                ghost_confidence = pred.get("confidence", 0) or 0
                ghost_direction = pred.get("direction", "FLAT")
                ghost_confidence_pct = round(ghost_confidence * 100, 1) if ghost_confidence <= 1 else round(ghost_confidence, 1)
                # Show REAL confidence — no jitter, no clamping
                # BTC at 12% should show 12%, BMBL at 70% should show 70%

                derived_change = 0.0
                if pred.get("expected_move") is not None:
                    expected_move = pred.get("expected_move")
                    derived_change = expected_move * 100 if abs(expected_move) <= 2 else expected_move
                elif ghost_confidence_pct:
                    direction_multiplier = 1 if ghost_direction == "UP" else -1 if ghost_direction == "DOWN" else 0
                    derived_change = (ghost_confidence_pct - 50) * 0.4 * direction_multiplier

                # FIX: Use change_pct if we have actual price data (even if 0.0!)
                # Only fall back to derived_change when change_pct is None (no prev_close data)
                final_change = change_pct if change_pct is not None else derived_change
                
                fallback_price = pred.get("price_at_prediction") or price

                watchlist_data.append({
                    "symbol": symbol,
                    "price": price if price is not None else fallback_price,
                    "change_pct": round(final_change, 2) if final_change is not None else 0.0,
                    "ghost_confidence": ghost_confidence_pct,
                    "ghost_direction": ghost_direction,
                    "type": "crypto" if symbol in CRYPTO_SYMBOLS else "stock",
                })
            
            except Exception as e:
                LOGGER.debug(f"Failed to enrich {symbol}: {e}")
                continue
        
        return {
            "ok": True,
            "items": watchlist_data,
            "watchlist": watchlist_data,
            "count": len(watchlist_data)
        }
    
    except Exception as e:
        LOGGER.error(f"Watchlist enrichment failed: {e}", exc_info=True)
        return {
            "ok": False,
            "watchlist": [],
            "error": str(e)
        }


async def _fetch_symbol_price(symbol: str) -> dict[str, Any]:
    """
    Fetch price and change for a single symbol using Ghost's existing price infrastructure.
    Runs concurrently for better performance.
    
    FIXED: Replaced yfinance (which fails in production) with ensure_price_cached()
    which uses Polygon for stocks and CoinGecko for crypto.
    
    FIXED (Jan 24, 2026): Calculate change_pct from price/prev_close since 
    fetch_price_live doesn't return it. This fixes the uniform 9.44% bug!
    
    FIXED (Jan 24, 2026): For crypto, fetch 24h change from CoinGecko since
    Coinbase doesn't return it.
    
    Returns:
        {"price": float, "change_pct": float|None} or exception
        change_pct is None when prev_close data is unavailable
    """
    try:
        # Check if this is a crypto symbol
        is_crypto = symbol.upper() in CRYPTO_SYMBOLS
        
        if is_crypto:
            # For crypto, use get_crypto_price_quorum which returns 24h change
            from core.crypto.crypto_providers import get_crypto_price_quorum
            result = await get_crypto_price_quorum(symbol, use_cache=True)
            
            if result and result.get("price"):
                return {
                    "price": result["price"],
                    "change_pct": result.get("change_24h_pct")  # CoinGecko returns this
                }
            else:
                return {"price": None, "change_pct": None}
        
        # For stocks, use the regular price infrastructure
        result = await ensure_price_cached(
            symbol,
            strict_live=False,  # Allow cached prices for speed
            drop_cache=False
        )
        
        if result and result.get("price"):
            price = result["price"]
            prev_close = result.get("prev_close")
            
            # FIX: Calculate change_pct from price and prev_close
            # Return None if prev_close unavailable (not 0.0!)
            change_pct = None
            if prev_close and prev_close > 0 and prev_close != price:
                change_pct = round(((price - prev_close) / prev_close) * 100, 2)
            elif prev_close == price:
                change_pct = 0.0  # Price unchanged from prev_close
            
            return {
                "price": price,
                "change_pct": change_pct  # None = no prev_close data
            }
        else:
            LOGGER.debug(f"No price available for {symbol}")
            return {"price": None, "change_pct": None}
    
    except Exception as e:
        LOGGER.debug(f"Price fetch failed for {symbol}: {e}")
        return {"price": None, "change_pct": None}


async def _fetch_vip_snapshot_with_timeout():
    """Helper to fetch VIP snapshot with aggressive timeout"""
    try:
        from core.crypto.crypto_providers import get_crypto_price_quorum

        vip_symbols = list(dict.fromkeys(VIP_COINS))
        tasks = [asyncio.wait_for(get_crypto_price_quorum(symbol, use_cache=True), timeout=0.4) for symbol in vip_symbols]
        
        # 2-second HARD TIMEOUT for entire fetch
        results = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True),
            timeout=2.0
        )
    except asyncio.TimeoutError:
        LOGGER.error("[VIP] Fetch timeout - returning empty data")
        return {"ok": False, "vip_coins": [], "error": "Timeout"}
    
    vip_data = []
    for symbol, result in zip(vip_symbols, results):
        if isinstance(result, Exception) or not result:
            vip_data.append({
                "symbol": symbol,
                "price": 0,
                "change_pct": 0.0,
                "status": "offline"
            })
            continue

        price_val = result.get("price")
        change_pct = result.get("change_24h_pct") or result.get("change_pct", 0.0)

        vip_data.append({
            "symbol": symbol,
            "price": round(price_val, 6) if price_val else 0.0,
            "change_pct": round(change_pct or 0.0, 2),
            "status": "online",
            "provider": result.get("provider"),
        })

    result = {
        "ok": True,
        "vip_coins": vip_data,
        "count": len(vip_data)
    }
    
    # Cache result
    _VIP_SNAPSHOT_CACHE["data"] = result
    _VIP_SNAPSHOT_CACHE["timestamp"] = time.time()
    LOGGER.info(f"[VIP] Cached snapshot with {len(vip_data)} coins")
    
    return result


async def _refresh_vip_cache():
    """Background task to refresh VIP cache (doesn't block requests)"""
    try:
        result = await _fetch_vip_snapshot_with_timeout()
        LOGGER.info(f"[VIP] Background refresh complete: {result.get('count', 0)} coins")
    except Exception as e:
        LOGGER.error(f"[VIP] Background refresh failed: {e}")


def _fetch_index_yahoo_chart(symbol: str) -> tuple:
    """Fetch index price via Yahoo v8 chart API — most reliable for ^-prefix symbols.
    Returns (price, prev_close) or (None, None).
    """
    try:
        import urllib.parse
        encoded = urllib.parse.quote(symbol, safe="")
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}"
        import httpx
        resp = httpx.get(
            url,
            params={"interval": "1d", "range": "5d"},
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            timeout=8,
        )
        if resp.status_code == 200:
            data = resp.json()
            result = (data.get("chart") or {}).get("result") or []
            if result:
                meta = result[0].get("meta", {})
                price = meta.get("regularMarketPrice") or meta.get("previousClose")
                prev = meta.get("chartPreviousClose") or meta.get("previousClose")
                # Also try the timestamp series for better prev close
                closes = ((result[0].get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
                closes = [c for c in closes if c is not None]
                if closes:
                    price = closes[-1]
                    if len(closes) >= 2:
                        prev = closes[-2]
                if price and float(price) > 0:
                    return float(price), float(prev) if prev else float(price)
    except Exception as e:
        LOGGER.debug(f"Yahoo chart API failed for {symbol}: {e}")
    return None, None


def _fetch_index_yahoo_quote(symbol: str) -> tuple:
    """Fetch index price via Yahoo v7 quote API (backup).
    Returns (price, prev_close) or (None, None).
    """
    try:
        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol.upper()}"
        import httpx
        resp = httpx.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            timeout=8,
        )
        if resp.status_code == 200:
            data = resp.json() or {}
            results = (data.get("quoteResponse") or {}).get("result") or []
            if results:
                q = results[0]
                price = q.get("regularMarketPrice")
                prev = q.get("regularMarketPreviousClose")
                if price and float(price) > 0:
                    return float(price), float(prev) if prev else float(price)
    except Exception as e:
        LOGGER.debug(f"Yahoo quote API failed for {symbol}: {e}")
    return None, None


def _fetch_index_yfinance(symbol: str) -> tuple:
    """Fetch index price via yfinance library (last resort — slower).
    Returns (price, prev_close) or (None, None).
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        if not hist.empty:
            price = float(hist["Close"].iloc[-1])
            prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else price
            if price > 0:
                return price, prev
    except Exception as e:
        LOGGER.debug(f"yfinance failed for {symbol}: {e}")
    return None, None


def _get_index_price(symbol: str) -> tuple:
    """Get index price with multi-provider fallback and caching.
    Chain: Yahoo Chart API → Yahoo Quote API → yfinance.
    Returns (price, prev_close).
    """
    import time as _t
    # Check cache
    cached = _INDEX_CACHE.get(symbol)
    if cached and (_t.time() - cached["ts"]) < _INDEX_CACHE_TTL:
        return cached["price"], cached["prev"]

    # Provider chain
    for fetcher in [_fetch_index_yahoo_chart, _fetch_index_yahoo_quote, _fetch_index_yfinance]:
        price, prev = fetcher(symbol)
        if price and price > 0:
            _INDEX_CACHE[symbol] = {"price": price, "prev": prev or price, "ts": _t.time()}
            return price, prev or price

    # Return stale cache if all providers failed
    if cached:
        LOGGER.warning(f"All index providers failed for {symbol}, using stale cache")
        return cached["price"], cached["prev"]
    return 0, 0


def run_prediction(symbol: str, market: str = "stock", horizon: str = "SHORT") -> dict:
    """
    Wrapper function for beast_scheduler and other scheduled prediction systems.
    Calls the synchronous prediction core.
    
    This function bridges scheduled systems (beast_scheduler, premarket_predictor)
    with the core prediction engine (run_single_prediction).
    
    Args:
        symbol: Trading symbol (e.g. "WOLF", "BTC")
        market: "stock" or "crypto" (informational only, symbol determines routing)
        horizon: "SHORT" or "LONG" (informational only, all predictions are 48h)
    
        now_ts = int(time.time())
        uptime_seconds = int(now_ts - _START_TS) if "_START_TS" in globals() else 0
        total_predictions = len(_LATEST_PREDICTIONS)
        activity_score = sum(_LAST_MULTI_PREDICTION_COUNTS.values())
        raw_health = 50 + (total_predictions * 5) + int(activity_score * 0.5)
        health_score = max(40, min(100, raw_health))

        if health_score >= 90:
            health_grade = "A"
        elif health_score >= 80:
            health_grade = "B"
        elif health_score >= 70:
            health_grade = "C"
        elif health_score >= 60:
            health_grade = "D"
        else:
            health_grade = "F"

        is_active = bool(STATE.get("active", True))
        engine_status = STATE.get("engine_status") or ("running" if is_active else "stopped")
        STATE["engine_status"] = engine_status

        last_prediction_ts = max(
            (pred.get("run_at", 0) or 0 for pred in _LATEST_PREDICTIONS.values()),
            default=0,
        )

        return {
            "ok": True,
            "mode": str(STATE.get("mode", "live")),
            "active": is_active,
            "live": is_active,
            "engine_status": engine_status,
            "uptime_seconds": uptime_seconds,
            "last_update_ts": int(last_prediction_ts) if last_prediction_ts else now_ts,
            "version": "3.0",
            "ghost_health": health_score,
            "ghost_health_score": health_score,
            "ghost_health_grade": health_grade,
            "predictions_today": activity_score,
        }
            'provider': str,
            'duration_ms': int
        }
    """
    try:
        # Call synchronous prediction core (no async needed)
        result = run_single_prediction(symbol.upper().strip())
        return result
    
    except Exception as e:
        LOGGER.error(f"run_prediction failed for {symbol}: {e}")
        return {
            'ok': False,
            'symbol': symbol,
            'direction': 'ERROR',
            'confidence': 0.0,
            'duration_ms': 0,
            'error': str(e)[:200]
        }


def _generate_multi_symbol_predictions():
    """
    Ghost Hunter V1: Generate predictions for all symbols in hunter universe.

    Called by scheduled_predictions scheduler (8am, 12pm, 4pm ET).
    Loops through HUNTER_STOCK_SYMBOLS and HUNTER_CRYPTO_SYMBOLS,
    calls run_single_prediction for each symbol, updates _LATEST_PREDICTIONS.

    Returns:
        dict with summary stats: {stocks: N, crypto: N, total: N, errors: []}
    """
    stocks_success = 0
    crypto_success = 0
    errors = []

    # Generate predictions for stocks
    for symbol in HUNTER_STOCK_SYMBOLS:
        try:
            result = run_single_prediction(symbol)
            if result.get("ok"):
                # Count all successful predictions (removed artificial 10% threshold)
                confidence = result.get("confidence", 0)
                stocks_success += 1
                duration_ms = result.get("duration_ms", 0)
                LOGGER.info(f"Hunter prediction generated: {symbol} (confidence: {confidence:.0%}, {duration_ms}ms)")
            else:
                errors.append(f"{symbol}: {result.get('error', 'unknown')}")
        except Exception as e:
            LOGGER.warning(f"Hunter prediction failed for {symbol}: {e}")
            errors.append(f"{symbol}: {str(e)[:100]}")

    # Generate predictions for crypto
    for symbol in HUNTER_CRYPTO_SYMBOLS:
        try:
            result = run_single_prediction(symbol)
            if result.get("ok"):
                # Count all successful predictions (removed artificial 10% threshold)
                confidence = result.get("confidence", 0)
                crypto_success += 1
                duration_ms = result.get("duration_ms", 0)
                LOGGER.info(f"Hunter prediction generated: {symbol} (confidence: {confidence:.0%}, {duration_ms}ms)")
            else:
                errors.append(f"{symbol}: {result.get('error', 'unknown')}")
        except Exception as e:
            LOGGER.warning(f"Hunter prediction failed for {symbol}: {e}")
            errors.append(f"{symbol}: {str(e)[:100]}")

    total = stocks_success + crypto_success
    LOGGER.info(f"Hunter multi-symbol predictions complete: {total} total ({stocks_success} stocks, {crypto_success} crypto)")

    return {
        "stocks": stocks_success,
        "crypto": crypto_success,
        "total": total,
        "errors": errors[:10],  # Limit error list to first 10
    }


def _send_multi_symbol_telegram_alert():
    """
    Ghost Hunter V1: Send Telegram alert with multi-symbol prediction summary.

    Called by scheduled_predictions scheduler after generating predictions.
    Reads from _LATEST_PREDICTIONS to build summary message.

    Returns:
        bool - True if sent successfully, False otherwise
    """
    try:
        # Build summary from _LATEST_PREDICTIONS
        stocks = []
        crypto = []

        for sym, pred in _LATEST_PREDICTIONS.items():
            category = _classify_symbol_category(sym)
            pred_str = f"{sym}: {pred['direction']} @ {pred['confidence']:.0%}"

            if category == "stocks":
                stocks.append(pred_str)
            elif category in ("crypto", "vip"):
                crypto.append(pred_str)

        # Build message
        msg_lines = ["🔮 Ghost Hunter Predictions"]

        if stocks:
            msg_lines.append(f"\n📈 Stocks ({len(stocks)}):")
            msg_lines.extend(stocks[:5])  # Limit to first 5
            if len(stocks) > 5:
                msg_lines.append(f"   ... +{len(stocks)-5} more")

        if crypto:
            msg_lines.append(f"\n💰 Crypto ({len(crypto)}):")
            msg_lines.extend(crypto[:5])  # Limit to first 5
            if len(crypto) > 5:
                msg_lines.append(f"   ... +{len(crypto)-5} more")

        if not stocks and not crypto:
            msg_lines.append("\n⚠️ No predictions available")

        message = "\n".join(msg_lines)

        # Send via Telegram (reuse existing helper if available)
        try:
            enqueue_alert_text(message)
            LOGGER.info("Hunter Telegram alert sent")
            return True
        except Exception as e:
            LOGGER.warning(f"Failed to send hunter Telegram alert: {e}")
            return False

    except Exception as e:
        LOGGER.exception(f"Failed to build hunter Telegram alert: {e}")
        return False


def _get_crypto_engine():
    """Get or initialize crypto prediction engine"""
    global _crypto_engine
    if _crypto_engine is None:
        try:
            from core.crypto.crypto_predictor import CryptoPredictionEngine

            _crypto_engine = CryptoPredictionEngine(db_path=WOLF_SQLITE_PATH)
            LOGGER.info("Crypto prediction engine initialized")
        except Exception as e:
            LOGGER.error(f"Failed to initialize crypto engine: {e}")
            raise HTTPException(500, "Crypto module not available") from e
    return _crypto_engine


def _get_crypto_providers():
    """Get crypto providers"""
    global _crypto_provider
    if _crypto_provider is None:
        try:
            from core.crypto import crypto_providers

            _crypto_provider = crypto_providers
            LOGGER.info("Crypto providers initialized")
        except Exception as e:
            LOGGER.error(f"Failed to initialize crypto providers: {e}")
            raise HTTPException(500, "Crypto providers not available") from e
    return _crypto_provider


def _get_crypto_name(symbol: str) -> str:
    """Map crypto symbol to full name for news filtering"""
    names = {
        "BTC": "BITCOIN",
        "ETH": "ETHEREUM",
        "SOL": "SOLANA",
        "DOGE": "DOGECOIN",
        "SHIB": "SHIBA",
        "PEPE": "PEPE",
        "BNB": "BINANCE",
        "XRP": "RIPPLE",
        "ADA": "CARDANO",
    }
    return names.get(symbol, symbol)


def _get_pooled_session_for(url: str) -> requests.Session:
    host = _get_host(url)
    s = _HTTP_SESSIONS.get(host)
    if s is not None:
        return s
    s = requests.Session()
    # Ban very large pools; use HTTPAdapter with limited pools and retries
    if Retry is not None:
        retry = Retry(
            total=HTTP_POOL_RETRIES,
            backoff_factor=0.3,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "POST"),
            raise_on_status=False,
        )
    else:
        retry = None  # type: ignore
    adapter = HTTPAdapter(
        pool_connections=HTTP_POOL_SIZE,
        pool_maxsize=HTTP_POOL_SIZE,
        max_retries=(retry or 0),
    )
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    _HTTP_SESSIONS[host] = s
    return s


def _http_get(
    url: str, *, timeout: float | None = None, headers: dict[str, str] | None = None
) -> requests.Response:
    t = timeout or HTTP_TIMEOUT_S
    try:
        if HTTP_POOL_ENABLED:
            sess = _get_pooled_session_for(url)
            if _C_HTTP_POOL_USED is not None:
                try:
                    _C_HTTP_POOL_USED.labels(host=_get_host(url)).inc()
                except Exception:
                    pass
            return sess.get(url, timeout=t, headers=headers or {})
        else:
            if _C_HTTP_DIRECT_USED is not None:
                try:
                    _C_HTTP_DIRECT_USED.labels(host=_get_host(url)).inc()
                except Exception:
                    pass
            return requests.get(url, timeout=t, headers=headers or {})
    except Exception:
        # Bubble up to callers
        raise


def _http_post(
    url: str,
    *,
    json: Any | None = None,
    data: Any | None = None,
    timeout: float | None = None,
    headers: dict[str, str] | None = None,
) -> requests.Response:
    t = timeout or HTTP_TIMEOUT_S
    try:
        if HTTP_POOL_ENABLED:
            sess = _get_pooled_session_for(url)
            if _C_HTTP_POOL_USED is not None:
                try:
                    _C_HTTP_POOL_USED.labels(host=_get_host(url)).inc()
                except Exception:
                    pass
            return sess.post(url, json=json, data=data, timeout=t, headers=headers or {})
        else:
            if _C_HTTP_DIRECT_USED is not None:
                try:
                    _C_HTTP_DIRECT_USED.labels(host=_get_host(url)).inc()
                except Exception:
                    pass
            return requests.post(url, json=json, data=data, timeout=t, headers=headers or {})
    except Exception:
        raise


def _set_hold_gauge():
    try:
        if _G_ALERT_HOLD is not None:
            _G_ALERT_HOLD.set(1 if ALERT_STATE.get("hold_override") else 0)
    except Exception:
        pass


def _breaker_should_skip(name: str) -> bool:
    b = _PROVIDER_BREAKERS.get(name)
    if not b:
        return False
    now = time.time()
    if b["state"] == "open":
        if now < float(b.get("open_until_ts", 0.0)):
            return True
        # allow a probe
        b["state"] = "half-open"
        return False
    return False


def _breaker_on_success(name: str):
    b = _PROVIDER_BREAKERS.setdefault(
        name,
        {"state": "closed", "failures": 0, "backoff_factor": 0, "open_until_ts": 0.0},
    )
    b["state"] = "closed"
    b["failures"] = 0
    b["backoff_factor"] = 0
    b["open_until_ts"] = 0.0


def _breaker_on_failure(name: str):
    b = _PROVIDER_BREAKERS.setdefault(
        name,
        {"state": "closed", "failures": 0, "backoff_factor": 0, "open_until_ts": 0.0},
    )
    b["failures"] = int(b.get("failures", 0)) + 1
    if int(b["failures"]) >= max(1, PROVIDER_FAIL_THRESHOLD):
        # open circuit and set backoff window (exponential with jitter)
        b["state"] = "open"
        bf = int(b.get("backoff_factor", 0)) + 1
        b["backoff_factor"] = bf
        backoff = min(PROVIDER_BACKOFF_MAX_S, PROVIDER_BACKOFF_S * (2 ** max(0, bf - 1)))
        # Add ±20% jitter to prevent thundering herd on recovery
        import random

        jitter = backoff * random.uniform(-0.2, 0.2)
        backoff = max(1, backoff + jitter)
        b["open_until_ts"] = time.time() + backoff
        b["failures"] = 0


def _provider_call(
    name: str, fn, configured: bool = True
) -> tuple[float | None, float | None, str]:
    """Wrap a provider fetch.

    Always returns a provider identity (name) on error so caller can mark throttling/backoff.
    """
    if not configured:
        return None, None, name
    throttled_patterns = ("429", "too many requests", "rate limit", "throttle")
    try:
        if _breaker_should_skip(name):
            # Circuit open: simulate failure (keep provider name)
            return None, None, name
        p, pc, prov = fn()
        # Detect hidden throttling: some providers return None silently
        if p is not None and prov:
            _breaker_on_success(name)
            return p, pc, prov or name
        # Failure path
        _breaker_on_failure(name)
        return None, None, prov or name
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        # Mark throttled provider in diagnostics side-channel
        try:
            if any(tok in msg for tok in throttled_patterns):
                PRICE_DIAG["throttled_provider"] = name
                # Initialize PROVIDER_BACKOFF entry aggressively
                now = time.time()
                back = PROVIDER_BACKOFF.get(name, {"until": 0.0, "failures": 0})
                back_failures = int(back.get("failures", 0)) + 1
                # Exponential (30s * 2^(n-1)) capped at 600s
                base = 30
                cooldown = min(600, base * (2 ** max(0, back_failures - 1)))
                back.update({"until": now + cooldown, "failures": back_failures})
                PROVIDER_BACKOFF[name] = back
        except Exception:
            pass
        _breaker_on_failure(name)
        return None, None, name


def _is_plausible_price(symbol: str, price: float | None, prev_close: float | None) -> bool:
    try:
        if price is None or price <= 0:
            return False
        sym = symbol.upper()
        if sym == "WOLF":
            min_price = float(os.getenv("WOLF_MIN_PRICE_SANITY", "5"))
            if price < min_price:
                return False
            if prev_close and prev_close > 0:
                # 50% default, can be loosened during market hours by PRICE_MAX_DEVIATION_OPEN
                try:
                    is_open, _ = _is_market_open_now()
                except Exception:
                    is_open = False
                max_dev = float(os.getenv("PRICE_MAX_DEVIATION", "0.5"))
                if is_open:
                    max_dev = PRICE_MAX_DEVIATION_OPEN
                if abs(price - prev_close) / prev_close > max_dev:
                    return False
        return True
    except Exception:
        return True


def _require_bearer(authorization: str | None) -> None:
    token = os.getenv("GHOST_API_TOKEN", "").strip()
    if not token:
        return
    if not authorization or not authorization.lower().startswith("bearer "):
        # For compatibility with tests expecting 403 when disabled/protected
        raise HTTPException(403, "missing bearer token")
    supplied = authorization.split(" ", 1)[1].strip()
    if supplied != token:
        raise HTTPException(403, "invalid token")


def _current_trace_id() -> str:
    try:
        return _cv_trace_id.get()
    except Exception:
        return "-"


def _now_iso(ts: float | None = None) -> str:
    return datetime.fromtimestamp(ts or time.time(), tz=UTC).isoformat()


def _cache_put_price(symbol: str, price: float | None, prev_close: float | None, provider: str):
    PRICE_CACHE[symbol.upper()] = {
        "price": None if price is None else float(price),
        "prev_close": None if prev_close is None else float(prev_close),
        "provider": provider,
        "ts": time.time(),
    }
    # Also persist to database for fallback
    if PORTFOLIO_PERSISTENCE_ENABLED and price is not None:
        try:
            store = get_portfolio_store()
            store.save_price(
                symbol.upper(),
                price,
                prev_close,
                provider,
                "open" if provider else "unknown",
            )
        except Exception as e:
            LOGGER.debug("price_persist_failed", extra={"symbol": symbol, "error": str(e)})


def _cache_get_price(symbol: str) -> tuple[float | None, float | None, str, bool]:
    rec = PRICE_CACHE.get(symbol.upper())
    if not rec:
        # Try persistent storage as fallback
        if PORTFOLIO_PERSISTENCE_ENABLED:
            try:
                store = get_portfolio_store()
                last = store.get_last_price(symbol.upper(), max_age_seconds=86400 * 7)  # 7 days
                if last:
                    price, prev, prov, ts = last
                    LOGGER.info(
                        "price_fallback_persistent",
                        extra={
                            "symbol": symbol,
                            "price": price,
                            "age_hours": (time.time() - ts) / 3600,
                        },
                    )
                    return price, prev, f"{prov}:cached", False
            except Exception as e:
                LOGGER.debug("price_fallback_failed", extra={"symbol": symbol, "error": str(e)})
        return None, None, "", True
    age = time.time() - float(rec.get("ts") or 0)
    # Dynamic TTL: during market hours, accept slightly older quotes to reduce provider load
    try:
        is_open, _ = _is_market_open_now()
    except Exception:
        is_open = False
    ttl = PRICE_TTL_OPEN_S if is_open else PRICE_TTL_S
    fresh = age <= ttl
    # During market hours, never consider a prev-close cache entry fresh.
    if fresh and is_open and rec.get("provider") == "prev-close":
        fresh = False
    if not fresh:
        # Cache stale, try persistent storage
        if PORTFOLIO_PERSISTENCE_ENABLED:
            try:
                store = get_portfolio_store()
                last = store.get_last_price(symbol.upper(), max_age_seconds=86400 * 7)  # 7 days
                if last:
                    price, prev, prov, ts = last
                    LOGGER.info(
                        "price_fallback_persistent",
                        extra={
                            "symbol": symbol,
                            "price": price,
                            "age_hours": (time.time() - ts) / 3600,
                        },
                    )
                    return price, prev, f"{prov}:cached", False
            except Exception as e:
                LOGGER.debug("price_fallback_failed", extra={"symbol": symbol, "error": str(e)})
        return None, rec.get("prev_close"), rec.get("provider") or "", False
    return rec.get("price"), rec.get("prev_close"), rec.get("provider") or "", True


def _resolve_stock_provider_order() -> list[str]:
    order: list[str] = []
    for name in STOCK_PRICE_SOURCE:
        if name not in order:
            order.append(name)
    for fallback in _DEFAULT_PROVIDER_ORDER:
        if fallback not in order:
            order.append(fallback)
    return order


def _get_provider_fetchers(
    symbol: str,
) -> list[tuple[str, Callable[[], tuple[float | None, float | None, str]]]]:
    sym = symbol.upper()
    fetchers: list[tuple[str, callable]] = []
    
    # CRITICAL FIX: Route crypto symbols to crypto providers (not stock providers)
    # This prevents Yahoo Finance 429 rate limit errors for crypto symbols
    is_crypto = sym in HUNTER_CRYPTO_SYMBOLS or sym in CRYPTO_SYMBOLS or _classify_symbol_category(sym) == "crypto"
    
    if is_crypto:
        # Use crypto-specific providers for crypto symbols
        try:
            from core.providers.turbo_provider import turbo_crypto_price
            def crypto_fetcher():
                result = turbo_crypto_price(sym, max_budget_s=2.0)
                if result and result.get("ok") and result.get("price"):
                    return (result["price"], None, result.get("provider", "crypto"))
                return (None, None, "")
            fetchers.append(("crypto_turbo", crypto_fetcher))
        except Exception as e:
            LOGGER.debug(f"Crypto provider unavailable for {sym}: {e}")
        return fetchers
    
    # Strategy: Always include yfinance and yahoo as free fallbacks
    # Only add paid providers if keys are present
    has_polygon = bool(POLYGON_KEY)
    has_alphavantage = bool(ALPHAVANTAGE_KEY)
    
    # If no paid keys, prioritize free sources first
    if not has_polygon and not has_alphavantage:
        fetchers.append(("yfinance", lambda sym=sym: _fetch_price_yfinance(sym)))
        fetchers.append(("yahoo", lambda sym=sym: _fetch_price_yahoo_http(sym)))
        return fetchers
    
    # Build provider list based on configured order
    for name in _resolve_stock_provider_order():
        if name == "polygon":
            if not POLYGON_KEY:
                continue
            fetchers.append((name, lambda sym=sym: _fetch_price_polygon(sym)))
        elif name == "polygon_intraday":
            if not POLYGON_KEY:
                continue
            fetchers.append((name, lambda sym=sym: _fetch_price_polygon_intraday(sym)))
        elif name == "alphavantage":
            if not ALPHAVANTAGE_KEY:
                continue
            fetchers.append((name, lambda sym=sym: _fetch_price_alphavantage(sym)))
        elif name == "yfinance":
            fetchers.append((name, lambda sym=sym: _fetch_price_yfinance(sym)))
        elif name == "yahoo":
            fetchers.append((name, lambda sym=sym: _fetch_price_yahoo_http(sym)))
    
    # Always ensure yfinance is available as ultimate fallback
    if not any(name == "yfinance" for name, _ in fetchers):
        fetchers.append(("yfinance", lambda sym=sym: _fetch_price_yfinance(sym)))
    
    return fetchers


async def fetch_price_live(
    symbol: str,
    *,
    strict_live: bool | None = None,
    max_age_seconds: int | None = None,
) -> dict[str, Any] | None:
    sym = symbol.upper().strip()
    if not sym:
        return None

    strict = PRICE_STRICT_LIVE if strict_live is None else bool(strict_live)
    ttl = max_age_seconds if max_age_seconds is not None else DATA_FRESHNESS_SEC

    entry = PRICE_CACHE.get(sym)
    age = None
    if entry and entry.get("ts"):
        try:
            age = max(0.0, time.time() - float(entry["ts"]))
        except Exception:
            age = None

    if not strict:
        cached_price, cached_prev, cached_provider, fresh = _cache_get_price(sym)
        if fresh and cached_price is not None:
            return {
                "symbol": sym,
                "price": float(cached_price),
                "prev_close": None if cached_prev is None else float(cached_prev),
                "provider": cached_provider,
                "cached": True,
                "fresh": True,
                "age": None if age is None else round(age, 3),
            }

    if sym == WOLF:
        if strict:
            PRICE_CACHE.pop(sym, None)
        price, prev, provider = get_wolf_price()
        entry = PRICE_CACHE.get(sym)
        age = None
        if entry and entry.get("ts"):
            try:
                age = max(0.0, time.time() - float(entry["ts"]))
            except Exception:
                age = None
        fresh = price is not None
        if age is not None:
            try:
                is_open, _ = _is_market_open_now()
            except Exception:
                is_open = False
            base_ttl = PRICE_TTL_OPEN_S if is_open else PRICE_TTL_S
            ttl_check = base_ttl
            if ttl is not None and ttl > 0:
                ttl_check = min(base_ttl, ttl)
            fresh = fresh and age <= ttl_check
        return {
            "symbol": sym,
            "price": price,
            "prev_close": prev,
            "provider": provider,
            "cached": False,
            "fresh": fresh,
            "age": None if age is None else round(age, 3),
        }

    if strict:
        PRICE_CACHE.pop(sym, None)

    # FIX (Jan 24, 2026): Special path for crypto to get 24h change data
    # Standard provider fetchers don't return change_24h_pct, but get_crypto_price_quorum does
    in_hunter = sym in HUNTER_CRYPTO_SYMBOLS
    in_crypto = sym in CRYPTO_SYMBOLS
    classified = _classify_symbol_category(sym)
    is_crypto = in_hunter or in_crypto or classified == "crypto"
    
    # DEBUG: Log crypto classification for troubleshooting (only if not crypto to debug false negatives)
    if not is_crypto and sym in ["CHZ", "TURBO", "RNDR", "ZEC"]:
        LOGGER.warning(f"[FIX5] Unexpected: {sym} not classified as crypto: hunter={in_hunter}, crypto_set={in_crypto}, classified={classified}")
    
    if is_crypto:
        # FIX5 (Jan 24, 2026): Use crypto quorum for 24h change data
        # Standard provider fetchers (coinbase, etc.) don't return change_24h_pct
        # Using use_cache=True for speed (prevents timeout on popular coins)
        try:
            from core.crypto.crypto_providers import get_crypto_price_quorum
            crypto_result = await asyncio.wait_for(
                get_crypto_price_quorum(sym, use_cache=True),
                timeout=2.0
            )
            if crypto_result and crypto_result.get("price"):
                price = float(crypto_result["price"])
                change_24h = crypto_result.get("change_24h_pct", 0) or 0
                prev_close = None
                if change_24h != 0:
                    prev_close = round(price / (1 + change_24h / 100), 6)
                provider_label = crypto_result.get("provider", "crypto-quorum")
                _cache_put_price(sym, price, prev_close, provider_label)
                LOGGER.info(f"[FIX5] Crypto quorum OK: {sym}=${price:.4f}, 24h={change_24h:.2f}%")
                return {
                    "symbol": sym,
                    "price": price,
                    "prev_close": prev_close,
                    "provider": provider_label,
                    "cached": False,
                    "fresh": True,
                    "age": 0.0,
                    "change_24h_pct": change_24h,
                }
            else:
                LOGGER.debug(f"[FIX5] Crypto quorum empty for {sym}")
        except asyncio.TimeoutError:
            LOGGER.debug(f"[FIX5] Crypto quorum timeout for {sym}")
        except Exception as e:
            LOGGER.warning(f"[FIX5] Crypto quorum failed for {sym}: {e}")

    provider_candidates = _get_provider_fetchers(sym)
    prev_candidate: float | None = None
    provider_label = ""
    if entry and entry.get("prev_close") is not None:
        try:
            prev_candidate = float(entry.get("prev_close"))
            provider_label = str(entry.get("provider") or "")
        except Exception:
            prev_candidate = None

    for name, fetcher in provider_candidates:
        try:
            price, prev, provider = await asyncio.wait_for(
                asyncio.to_thread(fetcher), timeout=PRICE_PROVIDER_TIMEOUT_S
            )
        except TimeoutError:
            LOGGER.warning(
                "price_fetch_timeout",
                extra={"symbol": sym, "provider": name, "timeout": PRICE_PROVIDER_TIMEOUT_S},
            )
            continue
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug(
                "price_fetch_error",
                extra={"symbol": sym, "provider": name, "error": str(exc)},
            )
            continue

        if price and price > 0:
            provider_label = provider or name
            prev_val = prev if prev is not None else prev_candidate
            _cache_put_price(sym, price, prev_val, provider_label)
            return {
                "symbol": sym,
                "price": float(price),
                "prev_close": None if prev_val is None else float(prev_val),
                "provider": provider_label,
                "cached": False,
                "fresh": True,
                "age": 0.0,
            }

        if prev and prev > 0:
            prev_candidate = float(prev)
            provider_label = provider or name

    if prev_candidate and prev_candidate > 0:
        label = provider_label or "prev-close"
        _cache_put_price(sym, prev_candidate, prev_candidate, label)
        return {
            "symbol": sym,
            "price": float(prev_candidate),
            "prev_close": float(prev_candidate),
            "provider": label,
            "cached": False,
            "fresh": False,
            "age": 0.0,
        }

    # Last resort: return stale cache if available
    entry = PRICE_CACHE.get(sym)
    if entry and entry.get("price") is not None:
        stale_age = None
        if entry.get("ts"):
            try:
                stale_age = max(0.0, time.time() - float(entry["ts"]))
            except Exception:
                stale_age = None
        return {
            "symbol": sym,
            "price": entry.get("price"),
            "prev_close": entry.get("prev_close"),
            "provider": entry.get("provider"),
            "cached": True,
            "fresh": False,
            "age": None if stale_age is None else round(stale_age, 3),
        }

    return None


async def ensure_price_cached(
    symbol: str,
    *,
    strict_live: bool | None = None,
    max_age_seconds: int | None = None,
    drop_cache: bool = False,
) -> dict[str, Any]:
    sym = symbol.upper().strip()
    if not sym:
        raise HTTPException(status_code=400, detail="symbol is required")
    if FOCUS_WOLF_ONLY and sym != WOLF:
        raise HTTPException(
            status_code=404,
            detail=f"Symbol {sym} not supported when FOCUS_WOLF_ONLY is enabled",
        )
    if drop_cache:
        PRICE_CACHE.pop(sym, None)

    result = await fetch_price_live(sym, strict_live=strict_live, max_age_seconds=max_age_seconds)
    if result is None or (result.get("price") is None and result.get("prev_close") is None):
        raise HTTPException(status_code=503, detail=f"Price unavailable for {sym}")
    return result


def _fetch_price_alphavantage(symbol: str) -> tuple[float | None, float | None, str]:
    if not ALPHAVANTAGE_KEY:
        return None, None, ""
    try:
        t0 = time.perf_counter()
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol.upper()}&apikey={ALPHAVANTAGE_KEY}"
        if _OTEL_TRACER is not None:
            with _OTEL_TRACER.start_as_current_span("provider.alphavantage.get"):  # type: ignore[attr-defined]
                r = _http_get(url, timeout=2)
        else:
            r = _http_get(url, timeout=2)
        r.raise_for_status()
        data = r.json() or {}
        gq = data.get("Global Quote") or data.get("GlobalQuote") or {}
        price = gq.get("05. price") or gq.get("price")
        prev = gq.get("08. previous close") or gq.get("previous_close")
        p = float(price) if price else None
        pc = float(prev) if prev else None
        if p and p > 0:
            try:
                if _H_PROVIDER_FETCH is not None:
                    _H_PROVIDER_FETCH.labels(provider="alphavantage").observe(
                        time.perf_counter() - t0
                    )
                if _C_PROVIDER_FETCH is not None:
                    _C_PROVIDER_FETCH.labels(provider="alphavantage", result="ok").inc()
            except Exception:
                pass
            return p, pc, "alphavantage"
    except Exception as e:
        LOGGER.warning(
            "provider_error",
            extra={
                "component": "provider",
                "provider": "alphavantage",
                "error": str(e),
            },
        )
        try:
            if _C_PROVIDER_FETCH is not None:
                _C_PROVIDER_FETCH.labels(provider="alphavantage", result="error").inc()
        except Exception:
            pass
    return None, None, ""


def _fetch_price_polygon(symbol: str) -> tuple[float | None, float | None, str]:
    if not POLYGON_KEY:
        return None, None, ""

    # Check global provider backoff
    if _provider_in_cooldown("polygon"):
        return None, None, ""

    try:
        t0 = time.perf_counter()
        # FIX (Jan 24, 2026): Fetch last 2 trading days to get REAL prev_close
        # The /prev endpoint returns only 1 bar, so we get the same value for both
        # Use /aggs/ticker/{symbol}/range/1/day/{from}/{to} instead
        from datetime import datetime, timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)  # 5 days to ensure we get 2 trading days
        
        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{symbol.upper()}/range/1/day/"
            f"{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            f"?adjusted=true&sort=asc&limit=10&apiKey={POLYGON_KEY}"
        )
        
        if _OTEL_TRACER is not None:
            with _OTEL_TRACER.start_as_current_span("provider.polygon.get"):  # type: ignore[attr-defined]
                r = _http_get(url, timeout=3)
        else:
            r = _http_get(url, timeout=3)
        r.raise_for_status()
        data = r.json() or {}
        results = data.get("results") or []
        
        if results and len(results) >= 1:
            # Most recent day's close is the current price
            current_close = float(results[-1].get("c") or 0)
            
            # Previous day's close (if we have 2+ bars)
            if len(results) >= 2:
                prev_close = float(results[-2].get("c") or 0)
            else:
                # Fallback: use same value (market hasn't had 2 days)
                prev_close = current_close
            
            if current_close > 0:
                # Success: reset backoff
                _note_provider_success("polygon")
                try:
                    if _H_PROVIDER_FETCH is not None:
                        _H_PROVIDER_FETCH.labels(provider="polygon").observe(
                            time.perf_counter() - t0
                        )
                    if _C_PROVIDER_FETCH is not None:
                        _C_PROVIDER_FETCH.labels(provider="polygon", result="ok").inc()
                except Exception:
                    pass
                return current_close, prev_close, "polygon"
    except Exception as e:
        # Detect rate limits
        status_code = None
        is_rate_limit = False

        try:
            if hasattr(e, "response") and e.response is not None:
                status_code = getattr(e.response, "status_code", None)
                if status_code in (429, 403):
                    is_rate_limit = True
        except Exception:
            pass

        error_str = str(e).lower()
        if "429" in error_str or "too many requests" in error_str or "403" in error_str or "forbidden" in error_str:
            is_rate_limit = True

        if is_rate_limit:
            _note_provider_429("polygon")
            LOGGER.warning(
                "provider_rate_limited",
                extra={"component": "provider", "provider": "polygon", "error": str(e), "rate_limited": True},
            )
        else:
            LOGGER.warning(
                "provider_error",
                extra={"component": "provider", "provider": "polygon", "error": str(e)},
            )
        try:
            if _C_PROVIDER_FETCH is not None:
                _C_PROVIDER_FETCH.labels(provider="polygon", result="error").inc()
        except Exception:
            pass
    return None, None, ""


def _fetch_polygon_intraday(symbol: str = "WOLF") -> dict:
    """
    Fetch last 30 min of 1-min bars from Polygon (5-min delayed, free tier).
    Returns most recent bar with price, high, low, volume, vwap.

    Free tier: 5 requests/minute, 5-min delayed data.
    Perfect for near real-time updates without paying for live data.
    """
    if not POLYGON_KEY:
        return {}

    # Check global provider backoff state (shared with _note_provider_429)
    if _provider_in_cooldown("polygon_intraday"):
        return {}

    # Basic provider-specific backoff state for local rate limiting
    global _POLY_INTRADAY_STATE
    try:
        _POLY_INTRADAY_STATE
    except NameError:
        _POLY_INTRADAY_STATE = {  # type: ignore[var-annotated]
            "last_call": 0.0,
        }

    now = time.time()
    # Throttle to ~1 call per 12s (5/min free tier) with jitter
    min_interval = 12.0
    if (now - float(_POLY_INTRADAY_STATE.get("last_call", 0.0))) < min_interval:
        return {}
    _POLY_INTRADAY_STATE["last_call"] = now

    try:
        now_ms = int(time.time() * 1000)
        from_ms = now_ms - (30 * 60 * 1000)  # 30 min ago

        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol.upper()}/range/1/minute/{from_ms}/{now_ms}?adjusted=true&sort=desc&limit=30&apiKey={POLYGON_KEY}"

        resp = _http_get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json() or {}

        if data.get("status") == "OK" and data.get("results"):
            # Most recent bar (sorted desc, so index 0 is latest)
            bar = data["results"][0]
            result = {
                "price": float(bar["c"]),  # close
                "high": float(bar["h"]),
                "low": float(bar["l"]),
                "open": float(bar["o"]),
                "volume": int(bar["v"]),
                "vwap": float(bar.get("vw", 0)),
                "timestamp": int(bar["t"] // 1000),  # ms to seconds
                "provider": "polygon_intraday",
                "bar_count": len(data["results"]),
            }

            LOGGER.info(
                f"Polygon intraday: {symbol} @ ${result['price']:.2f}, range ${result['low']:.2f}-${result['high']:.2f}, vol {result['volume']:,}",
                extra={"component": "price", "provider": "polygon_intraday"},
            )

            # Success: reset global backoff state
            _note_provider_success("polygon_intraday")
            return result

    except Exception as e:
        # Detect rate limit (429) or forbidden (403) responses
        status_code = None
        is_rate_limit = False

        try:
            if hasattr(e, "response") and e.response is not None:
                status_code = getattr(e.response, "status_code", None)
                if status_code in (429, 403):
                    is_rate_limit = True
        except Exception:
            pass

        # Check if error message contains rate limit indicators
        error_str = str(e).lower()
        if "429" in error_str or "too many requests" in error_str or "403" in error_str or "forbidden" in error_str:
            is_rate_limit = True

        # Apply exponential backoff for rate limits
        if is_rate_limit:
            _note_provider_429("polygon_intraday")
            LOGGER.warning(
                f"Polygon intraday rate limited (status={status_code}): {e}",
                extra={"component": "provider", "provider": "polygon_intraday", "error": str(e), "rate_limited": True},
            )
        else:
            # Non-rate-limit error, log but don't trigger aggressive backoff
            LOGGER.warning(
                f"Polygon intraday fetch failed: {e}",
                extra={"component": "provider", "provider": "polygon_intraday", "error": str(e)},
            )

    return {}


def _fetch_price_polygon_intraday(symbol: str) -> tuple[float | None, float | None, str]:
    """Adapter returning tuple format for polygon intraday quotes."""
    data = _fetch_polygon_intraday(symbol)
    if data and data.get("price"):
        try:
            return float(data["price"]), None, str(data.get("provider") or "polygon_intraday")
        except Exception:
            return None, None, "polygon_intraday"
    return None, None, "polygon_intraday"


def _fetch_price_yfinance(symbol: str) -> tuple[float | None, float | None, str]:
    """Fetch price from yfinance with exponential backoff for JSON errors."""
    max_retries = 3
    base_delay = 0.5  # Start with 500ms

    for attempt in range(max_retries):
        try:
            t0 = time.perf_counter()
            import yfinance as yf

            # Increase timeout and add better JSON error handling
            tkr = yf.Ticker(symbol.upper())
            # Use timeout in session to prevent hanging on bad JSON responses
            # Safety check: session might be None in some yfinance versions
            if hasattr(tkr, 'session') and tkr.session is not None:
                tkr.session.timeout = (5, 15)  # (connect, read) timeouts in seconds
            hist = tkr.history(period="2d")
            if not hist.empty:
                close = float(hist["Close"].iloc[-1])
                prev = float(hist["Close"].iloc[-2]) if len(hist["Close"]) > 1 else None
                if close > 0:
                    try:
                        if _H_PROVIDER_FETCH is not None:
                            _H_PROVIDER_FETCH.labels(provider="yfinance").observe(
                                time.perf_counter() - t0
                            )
                        if _C_PROVIDER_FETCH is not None:
                            _C_PROVIDER_FETCH.labels(provider="yfinance", result="ok").inc()
                    except Exception:
                        pass
                    return close, prev, "yfinance"

        except Exception as e:
            msg = str(e)
            low = msg.lower()

            # Check if it's a JSON parsing error (retryable)
            is_json_error = "expecting value" in low or "json" in low

            # Retry on JSON errors with exponential backoff
            if is_json_error and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # 0.5s, 1s, 2s
                LOGGER.debug(
                    f"yfinance JSON error for {symbol}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(delay)
                continue  # Retry

            # Not retryable or final attempt - log and fail
            # Heuristics for delisted / no data conditions surfaced by yfinance
            delisted_tokens = [
                "no price data found",
                "possibly delisted",
                "delisted",
                "no data",
            ]
            is_delisted = any(tok in low for tok in delisted_tokens)
            # Only log warning once for delisted to reduce noise; subsequent occurrences debug
            log_method = LOGGER.warning if not is_delisted or not PRICE_DIAG.get("delisted_hint") else LOGGER.debug
            log_method(
                "provider_error",
                extra={
                    "component": "provider",
                    "provider": "yfinance",
                    "error": msg,
                    "delisted": bool(is_delisted),
                    "json_error": is_json_error,
                },
            )
            if is_delisted and not PRICE_DIAG.get("delisted_hint"):
                try:
                    PRICE_DIAG["delisted_hint"] = True
                    PRICE_DIAG["delisted_provider"] = "yfinance"
                    PRICE_DIAG["delisted_reason"] = msg[:200]
                except Exception:
                    pass
            try:
                if _C_PROVIDER_FETCH is not None:
                    _C_PROVIDER_FETCH.labels(provider="yfinance", result="error").inc()
            except Exception:
                pass
            break  # Exit retry loop on non-retryable error

    return None, None, ""


def _fetch_price_yahoo_http(symbol: str) -> tuple[float | None, float | None, str]:
    """Lightweight Yahoo Finance HTTP quote API (no yfinance dependency).
    Returns (price, prev_close, provider_label). Provider label: 'yahoo'.
    """
    try:
        t0 = time.perf_counter()
        url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={symbol.upper()}"
        if _OTEL_TRACER is not None:
            with _OTEL_TRACER.start_as_current_span("provider.yahoo_http.get"):  # type: ignore[attr-defined]
                r = _http_get(url, timeout=30)
        else:
            r = _http_get(url, timeout=30)
        r.raise_for_status()
        data = r.json() or {}
        result = (data.get("quoteResponse") or {}).get("result") or []
        if result:
            q = result[0] or {}
            p = q.get("regularMarketPrice")
            pc = q.get("regularMarketPreviousClose")
            price = float(p) if p is not None else None
            prev = float(pc) if pc is not None else None
            if (price and price > 0) or (prev and prev > 0):
                try:
                    if _H_PROVIDER_FETCH is not None:
                        _H_PROVIDER_FETCH.labels(provider="yahoo").observe(time.perf_counter() - t0)
                    if _C_PROVIDER_FETCH is not None:
                        _C_PROVIDER_FETCH.labels(provider="yahoo", result="ok").inc()
                except Exception:
                    pass
                return price, prev, "yahoo"
    except Exception as e:
        LOGGER.warning(
            "provider_error",
            extra={"component": "provider", "provider": "yahoo", "error": str(e)},
        )
        try:
            if _C_PROVIDER_FETCH is not None:
                _C_PROVIDER_FETCH.labels(provider="yahoo", result="error").inc()
        except Exception:
            pass
    return None, None, ""


def _fetch_price_twelvedata(symbol: str) -> tuple[float | None, float | None, str]:
    """Fetch stock price from TwelveData API (free tier available).
    Returns (price, prev_close, provider_label).
    """
    try:
        import requests
        t0 = time.perf_counter()
        api_key = os.getenv("TWELVEDATA_API_KEY", "demo")  # Free demo key works!
        url = f"https://api.twelvedata.com/price?symbol={symbol.upper()}&apikey={api_key}"
        
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        if "price" in data:
            price = float(data["price"])
            if price > 0:
                LOGGER.debug(f"TwelveData price for {symbol}: ${price:.2f}")
                try:
                    if _H_PROVIDER_FETCH is not None:
                        _H_PROVIDER_FETCH.labels(provider="twelvedata").observe(time.perf_counter() - t0)
                    if _C_PROVIDER_FETCH is not None:
                        _C_PROVIDER_FETCH.labels(provider="twelvedata", result="ok").inc()
                except Exception:
                    pass
                return price, None, "twelvedata"
    except Exception as e:
        LOGGER.debug(f"TwelveData error for {symbol}: {e}")
        try:
            if _C_PROVIDER_FETCH is not None:
                _C_PROVIDER_FETCH.labels(provider="twelvedata", result="error").inc()
        except Exception:
            pass
    return None, None, ""


def _build_price_providers(symbol: str, *, is_market_open: bool) -> list[PriceProvider]:
    providers: list[PriceProvider] = []
    blocklist = PROVIDER_BLOCKLIST.get(symbol.upper(), set())
    now = time.time()

    try:
        PRICE_DIAG["backoff_skip"] = []
    except Exception:
        pass

    def should_skip(name: str) -> bool:
        if name in blocklist:
            return True
        back = PROVIDER_BACKOFF.get(name)
        until = 0.0
        if back:
            until = back.get("backoff_until") or back.get("until", 0.0)
        if until > now:
            try:
                PRICE_DIAG.setdefault("backoff_skip", []).append(name)
            except Exception:
                pass
            return True
        return False

    def add_provider(name: str, fn, *, configured: bool) -> None:
        if not configured or should_skip(name):
            return
        limiter = _PROVIDER_LIMITERS.get(name)

        def fetch(name=name, fn=fn, configured=configured):
            return _provider_call(name, fn, configured=configured)

        providers.append(
            PriceProvider(
                name=name,
                fetcher=fetch,
                enabled=True,
                rate_limiter=limiter,
            )
        )

    # Convert crypto symbols to yfinance/Yahoo format (BTC -> BTC-USD)
    provider_symbol = symbol
    if symbol.upper() in CRYPTO_SYMBOLS:
        provider_symbol = f"{symbol.upper()}-USD"

    # PRIORITY 1: Free unlimited APIs (Yahoo, yfinance)
    # These have no rate limits, try them first to conserve paid API calls
    add_provider("yfinance", lambda s=provider_symbol: _fetch_price_yfinance(s), configured=True)
    add_provider("yahoo", lambda s=provider_symbol: _fetch_price_yahoo_http(s), configured=True)

    # PRIORITY 2: Paid APIs with rate limits (AlphaVantage, Polygon)
    # Only use these as fallback when free APIs fail
    add_provider(
        "alphavantage",
        lambda: _fetch_price_alphavantage(symbol),
        configured=bool(ALPHAVANTAGE_KEY),
    )

    add_provider(
        "polygon",
        lambda: _fetch_price_polygon(symbol),
        configured=bool(POLYGON_KEY),
    )

    # PRIORITY 3: Polygon intraday (only during market hours)
    if is_market_open and POLYGON_KEY:
        add_provider(
            "polygon_intraday",
            lambda: _fetch_price_polygon_intraday(symbol),
            configured=True,
        )

    return providers


def _update_price_diagnostics(
    decision: PriceDecision, *, fallback_reason: str | None = None
) -> None:
    quotes = decision.quotes or []
    valid = [q.price for q in quotes if q.price is not None]
    spread = None
    if len(valid) >= 2:
        hi = max(valid)
        lo = min(valid)
        spread = (hi - lo) / max(hi, 1e-6)

    try:
        PRICE_DIAG["providers"] = [(q.provider, q.price) for q in quotes]
        PRICE_DIAG["provider_spread"] = spread
        PRICE_DIAG["quorum_ok"] = decision.reason == "consensus"
        PRICE_DIAG["anomaly"] = decision.reason != "consensus"
        PRICE_DIAG["reason"] = "" if decision.reason == "consensus" else decision.reason
        PRICE_DIAG["last_fetch_latency_ms"] = decision.latency_ms
        if decision.price is not None:
            PRICE_DIAG["last_fetch_provider"] = decision.provider_label
            PRICE_DIAG["last_good_price_ts"] = int(time.time())
        PRICE_DIAG["quorum_degraded"] = decision.reason == "consensus" and decision.quorum_size < 3
        PRICE_DIAG["failures"] = [
            {
                "provider": q.provider,
                "error": q.error,
                "latency_ms": q.latency_ms,
            }
            for q in quotes
            if q.error
        ]
        if fallback_reason is not None:
            PRICE_DIAG["fallback_reason"] = fallback_reason
        elif decision.reason == "consensus":
            PRICE_DIAG["fallback_reason"] = None
    except Exception:
        pass


def get_wolf_price() -> tuple[float | None, float | None, str]:
    # Cache first
    price, prev, provider, fresh = _cache_get_price(WOLF)
    if fresh and price is not None:
        try:
            _add_event(
                "price_ok",
                "cache",
                {
                    "provider": provider or "",
                    "price": float(price),
                    "prev_close": (None if prev is None else float(prev)),
                    "ms": 0,
                    "ttl_hit": True,
                },
            )
        except Exception:
            pass
        # Record cached price to history for overlay continuity
        try:
            _record_price_tick(WOLF, price)
        except Exception:
            pass
        price, provider = _apply_price_override(WOLF, price, provider)
        return price, prev, provider
    try:
        is_market_open, _ = _is_market_open_now()
    except Exception:
        is_market_open = False

    providers = _build_price_providers(WOLF, is_market_open=is_market_open)
    quorum_timeout = float(os.getenv("PRICE_PROVIDER_TIMEOUT", "6"))
    decision: PriceDecision

    if providers:
        try:
            decision = get_price_quorum().get_price(
                symbol=WOLF,
                providers=providers,
                prev_close=prev,
                is_market_open=is_market_open,
                timeout=quorum_timeout,
            )
        except Exception as exc:
            LOGGER.warning(
                "price_quorum_error",
                extra={"component": "price", "error": str(exc)},
            )
            decision = PriceDecision(
                price=None,
                prev_close=prev,
                provider_label="unavailable",
                reason="quorum_exception",
                quorum_size=0,
                quotes=[],
                latency_ms=0.0,
            )
    else:
        decision = PriceDecision(
            price=None,
            prev_close=prev,
            provider_label="unavailable",
            reason="no_providers",
            quorum_size=0,
            quotes=[],
            latency_ms=0.0,
        )

    prev_candidate = decision.prev_close if decision.prev_close is not None else prev

    if decision.price is not None:
        try:
            _add_event(
                "price_ok",
                "quorum",
                {
                    "provider": decision.provider_label,
                    "price": float(decision.price),
                    "prev_close": (None if prev_candidate is None else float(prev_candidate)),
                    "ms": decision.latency_ms,
                    "ttl_hit": False,
                },
            )
        except Exception:
            pass
        _cache_put_price(WOLF, decision.price, prev_candidate, decision.provider_label)
        try:
            _record_price_tick(WOLF, decision.price)
        except Exception:
            pass
        _update_price_diagnostics(decision)
        price_val, provider_label = _apply_price_override(
            WOLF, decision.price, decision.provider_label
        )
        return price_val, prev_candidate, provider_label

    fallback_reason = decision.reason or "quorum_failed"
    _update_price_diagnostics(decision, fallback_reason=fallback_reason)

    if prev_candidate is not None and prev_candidate > 0:
        _cache_put_price(WOLF, prev_candidate, prev_candidate, "prev-close")
        try:
            _record_price_tick(WOLF, prev_candidate)
        except Exception:
            pass
        price_val, provider_label = _apply_price_override(WOLF, prev_candidate, "prev-close")
        return price_val, prev_candidate, provider_label

    # ENHANCED FALLBACK: Try forecast data if available
    try:
        import json

        forecast_path = os.path.join(os.path.dirname(__file__), "data", f"forecast_{WOLF}.json")
        if os.path.exists(forecast_path):
            with open(forecast_path) as f:
                forecast_data = json.load(f)
                points = forecast_data.get("points", [])
                if points and len(points) > 0:
                    p0 = points[0].get("p")
                    if p0 is not None and p0 > 0:
                        fallback_price = float(p0)
                        LOGGER.info(
                            "price_fallback_forecast",
                            extra={
                                "price": fallback_price,
                                "symbol": WOLF,
                                "aso": forecast_data.get("aso"),
                            },
                        )
                        _cache_put_price(WOLF, fallback_price, fallback_price, "forecast-fallback")
                        decision_fallback = PriceDecision(
                            price=None,
                            prev_close=fallback_price,
                            provider_label="forecast-fallback",
                            reason="forecast_fallback",
                            quorum_size=0,
                            quotes=[],
                            latency_ms=0.0,
                        )
                        _update_price_diagnostics(
                            decision_fallback, fallback_reason="using_forecast_p0"
                        )
                        price_val, provider_label = _apply_price_override(
                            WOLF, fallback_price, "forecast-fallback"
                        )
                        return price_val, fallback_price, provider_label
    except Exception as e:
        LOGGER.debug("forecast_fallback_failed", extra={"error": str(e)})

    _cache_put_price(WOLF, None, prev, provider or "unavailable")
    _update_price_diagnostics(
        PriceDecision(
            price=None,
            prev_close=prev,
            provider_label=provider or "unavailable",
            reason="no_data_available",
            quorum_size=0,
            quotes=decision.quotes,
            latency_ms=0.0,
        ),
        fallback_reason="no_data_available",
    )
    price_val, provider_label = _apply_price_override(WOLF, None, provider or "unavailable")
    return price_val, prev, provider_label


def get_wolf_news(limit: int = 10) -> dict[str, Any]:
    now = time.time()
    if (now - float(NEWS_CACHE.get("ts") or 0)) <= NEWS_TTL_S and NEWS_CACHE.get("items"):
        return {"items": NEWS_CACHE["items"]}
    items: list[dict] = []
    note: str | None = None
    if POLYGON_KEY:
        try:
            url = f"https://api.polygon.io/v2/reference/news?ticker={WOLF}&limit={limit}&apiKey={POLYGON_KEY}"
            if _OTEL_TRACER is not None:
                with _OTEL_TRACER.start_as_current_span("provider.polygon.news"):  # type: ignore[attr-defined]
                    r = _http_get(url, timeout=8)
            else:
                r = _http_get(url, timeout=8)
            r.raise_for_status()
            data = r.json() or {}
            for it in data.get("results", [])[:limit]:
                items.append(
                    {
                        "id": it.get("id"),
                        "headline": it.get("title") or it.get("description") or "",
                        "ts": it.get("published_utc"),
                        "url": it.get("article_url"),
                        "description": it.get("description"),
                    }
                )
        except Exception:
            note = "rate-limited"
    else:
        note = "provider-missing"
    # Optional Reuters RSS feeds
    if REUTERS_FEEDS_ON:
        # Outer try/except to catch DNS/network failures gracefully
        try:
            feed_urls = [u.strip() for u in (REUTERS_FEEDS or "").split(",") if u.strip()]
            if NEWS_MANUAL_FEEDS:
                feed_urls.extend([u for u in NEWS_MANUAL_FEEDS if u not in feed_urls])
            for feed_url in feed_urls[:8]:
                try:
                    r = _http_get(feed_url, timeout=8)
                    r.raise_for_status()
                    root = ET.fromstring(r.text)
                    # Try RSS 2.0 structure: channel/item
                    for item in root.findall(".//item")[: max(1, limit)]:
                        title = (item.findtext("title") or "").strip()
                        link = (item.findtext("link") or "").strip()
                        pub = (item.findtext("pubDate") or "").strip()
                        # Host whitelist
                        if NEWS_WHITELIST and link:
                            try:
                                from urllib.parse import urlparse

                                host = urlparse(link).netloc.lower()
                                if not any(w in host for w in NEWS_WHITELIST):
                                    continue
                            except Exception:
                                pass
                        # Age filter
                        try:
                            ts_num = (
                                int(datetime.fromisoformat(pub.replace("Z", "+00:00")).timestamp())
                                if pub
                                else int(time.time())
                            )
                        except Exception:
                            ts_num = int(time.time())
                        if NEWS_MAX_AGE_MIN and (time.time() - ts_num) > (NEWS_MAX_AGE_MIN * 60):
                            continue
                        # Symbol/keyword filter
                        tl = title.lower()
                        if REUTERS_SYMBOLS or REUTERS_KEYWORDS:
                            keep = False
                            if REUTERS_KEYWORDS and any(k in tl for k in REUTERS_KEYWORDS):
                                keep = True
                            if (
                                not keep
                                and REUTERS_SYMBOLS
                                and any(s in title.upper() for s in REUTERS_SYMBOLS)
                            ):
                                keep = True
                            if not keep:
                                continue
                        items.append(
                            {
                                "id": f"reuters:{hashlib.sha1((link or title).encode('utf-8', 'ignore')).hexdigest()[:12]}",
                                "headline": title,
                                "ts": ts_num,
                                "url": link or None,
                                "description": None,
                                "src": "reuters",
                                "syms": REUTERS_SYMBOLS
                                or ([WOLF] if "WOLF" in title.upper() else []),
                            }
                        )
                except Exception:
                    # Per-feed error: Try Atom fallback
                    try:
                        r = _http_get(feed_url, timeout=8)
                        r.raise_for_status()
                        root = ET.fromstring(r.text)
                        ns = {"atom": "http://www.w3.org/2005/Atom"}
                        for entry in root.findall(".//atom:entry", ns)[: max(1, limit)]:
                            title = (entry.findtext("atom:title", namespaces=ns) or "").strip()
                            link_el = entry.find("atom:link", ns)
                            link = link_el.get("hre") if link_el is not None else None
                            updated = (entry.findtext("atom:updated", namespaces=ns) or "").strip()
                            # Host whitelist
                            if NEWS_WHITELIST and link:
                                try:
                                    from urllib.parse import urlparse

                                    host = urlparse(link).netloc.lower()
                                    if not any(w in host for w in NEWS_WHITELIST):
                                        continue
                                except Exception:
                                    pass
                            try:
                                ts_num = (
                                    int(
                                        datetime.fromisoformat(
                                            updated.replace("Z", "+00:00")
                                        ).timestamp()
                                    )
                                    if updated
                                    else int(time.time())
                                )
                            except Exception:
                                ts_num = int(time.time())
                            if NEWS_MAX_AGE_MIN and (time.time() - ts_num) > (
                                NEWS_MAX_AGE_MIN * 60
                            ):
                                continue
                            tl = title.lower()
                            if REUTERS_SYMBOLS or REUTERS_KEYWORDS:
                                keep = False
                                if REUTERS_KEYWORDS and any(k in tl for k in REUTERS_KEYWORDS):
                                    keep = True
                                if (
                                    not keep
                                    and REUTERS_SYMBOLS
                                    and any(s in title.upper() for s in REUTERS_SYMBOLS)
                                ):
                                    keep = True
                                if not keep:
                                    continue
                            items.append(
                                {
                                    "id": f"reuters:{hashlib.sha1(((link or '') + title).encode('utf-8', 'ignore')).hexdigest()[:12]}",
                                    "headline": title,
                                    "ts": ts_num,
                                    "url": link,
                                    "description": None,
                                    "src": "reuters",
                                    "syms": REUTERS_SYMBOLS
                                    or ([WOLF] if "WOLF" in title.upper() else []),
                                }
                            )
                    except Exception:
                        continue
        except Exception as e:
            # Outer Reuters failure (DNS, network, etc.) - use cached news with degraded flag
            print(f"[NEWS] Reuters feed error (DNS/network): {e}")
            if NEWS_CACHE.get("items"):
                # Return cached items with degraded flag
                for item in NEWS_CACHE["items"]:
                    if item.get("src") == "reuters":
                        item["_degraded"] = True
                note = "reuters:degraded"
            else:
                # No cache available
                if not note:
                    note = "reuters:error"
                items.append(
                    {
                        "id": f"note:{int(now)}",
                        "headline": "Reuters feed temporarily unavailable (network error)",
                        "ts": _now_iso(now),
                        "url": None,
                        "_degraded": True,
                    }
                )
    if not items and note:
        items = [
            {
                "id": f"note:{int(now)}",
                "headline": "Feed rate-limited",
                "ts": _now_iso(now),
                "url": None,
            }
        ]
    # Optionally score sentiment
    scored_items = items
    engine = "none"
    if NEWS_SENTIMENT_ON:
        try:
            scored_items, engine = _score_news_items(items)
        except Exception:
            pass
    NEWS_CACHE.update({"items": scored_items, "ts": now})
    score, agg_engine, used = (
        _aggregate_news_score(scored_items) if NEWS_SENTIMENT_ON else (None, "none", 0)
    )
    return {
        "items": scored_items,
        "note": note,
        "news_signal": {
            "score": score,
            "engine": (agg_engine if NEWS_SENTIMENT_ON else "none"),
            "items_scored": used,
        },
    }


def _pct_change(series) -> float:
    try:
        if series is None or len(series) < 2:
            return 0.0
        a = float(series[-2])
        b = float(series[-1])
        if a == 0:
            return 0.0
        return (b - a) / a * 100.0
    except Exception:
        return 0.0


def _ny_now():
    try:
        if _TZ_NY:
            return datetime.now(tz=_TZ_NY)
        # fallback naive UTC -> NY offset approximation
        return datetime.now(tz=UTC)
    except Exception:
        return datetime.now(tz=UTC)


def _is_market_open_now() -> tuple[bool, int]:
    """Return (is_open, next_open_ts_utc).
    Approximation: Mon-Fri, 09:30–16:00 ET; ignores market holidays.
    """
    try:
        now_ny = _ny_now()
        wd = now_ny.weekday()  # Mon=0
        open_dt = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
        close_dt = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
        open_today = (wd <= 4) and (now_ny >= open_dt) and (now_ny <= close_dt)
        if open_today:
            # next open assumed next business day 9:30
            d = 1 if wd < 4 else 3  # Fri -> Mon
            next_day = (now_ny + timedelta(days=d)).replace(
                hour=9, minute=30, second=0, microsecond=0
            )
            next_ts = (
                int(next_day.astimezone(UTC).timestamp())
                if _TZ_NY
                else int((now_ny + timedelta(days=d)).timestamp())
            )
            return True, next_ts
        # compute next open from now
        # if before 9:30 today and weekday
        if wd <= 4 and now_ny < open_dt:
            next_open = open_dt
        else:
            # move to next weekday
            d = 1
            nxt = now_ny + timedelta(days=d)
            while nxt.weekday() > 4:
                nxt = nxt + timedelta(days=1)
            next_open = nxt.replace(hour=9, minute=30, second=0, microsecond=0)
        next_ts = (
            int(next_open.astimezone(UTC).timestamp()) if _TZ_NY else int(next_open.timestamp())
        )
        return False, next_ts
    except Exception:
        # safe default: closed; next open 24h
        return False, int(time.time() + 24 * 3600)


async def _get_crypto_movers() -> list[dict[str, Any]]:
    """
    Get top crypto movers with 24h price changes.
    Returns sorted list by absolute percentage change.
    """
    try:
        if os.getenv("CRYPTO_ENABLED", "0") != "1":
            return []

        from core.crypto import crypto_providers

        crypto_symbols = os.getenv("CRYPTO_SYMBOLS", "BTC,ETH,SOL,BNB").split(",")
        movers = []

        for sym in crypto_symbols:
            sym = sym.strip().upper()
            if not sym:
                continue

            try:
                # Get current price with 24h change
                result = await crypto_providers.get_crypto_price_quorum(sym)
                if result and result.get("price") is not None:
                    price = result["price"]
                    change_24h = result.get("change_24h_pct", 0.0)

                    movers.append(
                        {
                            "sym": sym,
                            "symbol": sym,
                            "price": round(price, 2 if price > 10 else 6),
                            "change_pct": round(change_24h, 2),
                            "volume_24h": result.get("volume_24h"),
                        }
                    )
            except Exception as e:
                LOGGER.debug(f"Failed to get crypto mover data for {sym}: {e}")
                continue

        # Sort by absolute percentage change (biggest movers first)
        movers.sort(key=lambda x: abs(x.get("change_pct", 0.0)), reverse=True)

        return movers[:5]  # Top 5 movers

    except Exception as e:
        LOGGER.warning(f"Failed to get crypto movers: {e}")
        return []


def _macro_brain(now_price: float | None, news_score: float | None) -> dict[str, Any]:
    if not MACRO_BRAIN_ON:
        return {"enabled": False, "scenarios": [], "gps": "of"}
    try:
        import yfinance as yf
    except Exception:
        try:
            if _C_MACRO_REFRESH is not None:
                _C_MACRO_REFRESH.labels(result="yfinance-missing").inc()
        except Exception:
            pass
        return {"enabled": True, "error": "yfinance-missing", "scenarios": []}
    lookback = max(5, MACRO_LOOKBACK_DAYS)
    tickers = [t.strip().upper() for t in MACRO_TICKERS if t.strip()]
    perf: dict[str, float] = {}
    try:
        for t in tickers:
            try:
                hist = yf.Ticker(t).history(period=f"{lookback}d")
                if hist is not None and not hist.empty and len(hist["Close"]) >= 2:
                    perf[t] = _pct_change(hist["Close"])  # simple 1-step %
                else:
                    perf[t] = 0.0
            except Exception:
                perf[t] = 0.0
    except Exception:
        pass
    # Momentum proxy: average of available proxies
    proxy_vals = [v for v in perf.values() if isinstance(v, (int, float))]
    proxy_avg = sum(proxy_vals) / len(proxy_vals) if proxy_vals else 0.0
    # WOLF momentum via prev close vs current if available
    wolf_momo = 0.0
    try:
        if now_price is not None:
            _, prev, _, _ = _cache_get_price(WOLF)
            if prev and prev > 0:
                wolf_momo = (now_price - prev) / prev * 100.0
    except Exception:
        wolf_momo = 0.0
    ns = news_score if isinstance(news_score, (int, float)) else 0.0
    # Heuristic scoring
    base_score = 0.5 * proxy_avg + 0.3 * wolf_momo + 20.0 * ns  # ns ~ [-1,1] scaled
    # Normalize roughly into [-100, 100]
    base_score = max(-100.0, min(100.0, base_score))
    # Scenarios
    bull_p = max(0.0, min(1.0, 0.5 + base_score / 200.0))
    bear_p = max(0.0, min(1.0, 0.5 - base_score / 200.0))
    base_p = max(0.0, min(1.0, 1.0 - abs(base_score) / 150.0)) * 0.6
    # Renormalize
    total = bull_p + base_p + bear_p
    if total <= 0:
        bull_p, base_p, bear_p = 0.34, 0.33, 0.33
        total = 1.0
    bull_p, base_p, bear_p = bull_p / total, base_p / total, bear_p / total
    # Confidence as dispersion
    conf = int(round(100.0 * (1.0 - 2.0 * min(bull_p, bear_p))))
    scenarios = [
        {
            "name": "bull",
            "p": round(bull_p, 3),
            "drivers": [
                "semis/tech momentum",
                "positive news" if ns > 0 else "mixed news",
            ],
        },
        {
            "name": "base",
            "p": round(base_p, 3),
            "drivers": ["mean reversion", "range-bound"],
        },
        {
            "name": "bear",
            "p": round(bear_p, 3),
            "drivers": ["risk-of", "negative news" if ns < 0 else "mixed news"],
        },
    ]
    try:
        if _G_MACRO_CONF is not None:
            for sc in scenarios:
                _G_MACRO_CONF.labels(scenario=sc["name"]).set(int(conf))
        if _C_MACRO_REFRESH is not None:
            _C_MACRO_REFRESH.labels(result="ok").inc()
    except Exception:
        pass
    outlook = {
        "enabled": True,
        "confidence": conf,
        "scenarios": scenarios,
        "summary": (
            "Likely uptrend"
            if bull_p > bear_p and conf > 60
            else (
                "Caution: negative catalysts"
                if bear_p > bull_p and conf > 60
                else "Neutral / mixed"
            )
        ),
    }
    return outlook


def _persist_load():
    def _restore_from_data(data: dict):
        """Restore STATE from persisted data dict"""
        STATE["qty"] = float(data.get("qty", STATE.get("qty", 0.0)))
        STATE["avg_cost"] = float(data.get("avg_cost", STATE.get("avg_cost", 0.0)))
        # Restore positions array if present
        if "positions" in data:
            STATE["positions"] = data["positions"]
        # Restore cash balances
        if "cash" in data:
            STATE["cash"] = float(data["cash"])
        if "cash_stock" in data:
            STATE["cash_stock"] = float(data["cash_stock"])
        if "cash_crypto" in data:
            STATE["cash_crypto"] = float(data["cash_crypto"])

    # Try new portfolio persistence layer first
    if PORTFOLIO_PERSISTENCE_ENABLED:
        try:
            store = get_portfolio_store()
            # Load WOLF position
            pos = store.get_position(WOLF)
            if pos:
                STATE["qty"] = float(pos.get("quantity", 0.0))
                STATE["avg_cost"] = float(pos.get("avg_cost", 0.0))
                LOGGER.info(
                    "position_restored_from_db",
                    extra={
                        "symbol": WOLF,
                        "qty": STATE["qty"],
                        "avg": STATE["avg_cost"],
                    },
                )
                # Also load last known price
                if pos.get("last_known_price"):
                    _cache_put_price(
                        WOLF,
                        pos["last_known_price"],
                        None,
                        pos.get("last_provider") or "cached",
                    )
                return
        except Exception as e:
            LOGGER.warning("portfolio_persistence_load_failed", extra={"error": str(e)})

    mode = WOLF_PERSIST_MODE
    # auto: try redis -> sqlite -> file
    if mode == "auto":
        tried: list[str] = []
        # redis
        if REDIS_URL:
            try:
                import redis  # type: ignore

                r = redis.Redis.from_url(REDIS_URL)
                raw = r.get("wolf:position")
                if raw:
                    data = json.loads(raw)  # type: ignore
                    _restore_from_data(data)
                    return
                tried.append("redis")
            except Exception:
                tried.append("redis:error")
        # sqlite
        try:
            import sqlite3

            if os.path.exists(WOLF_SQLITE_PATH) or os.path.exists(
                os.path.dirname(WOLF_SQLITE_PATH) or "."
            ):
                conn = sqlite3.connect(WOLF_SQLITE_PATH)
                cur = conn.cursor()
                cur.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT)")
                conn.commit()
                cur.execute("SELECT value FROM state WHERE key='position'")
                row = cur.fetchone()
                if row and row[0]:
                    data = json.loads(row[0])
                    _restore_from_data(data)
                    conn.close()
                    return
                conn.close()
                tried.append("sqlite")
        except Exception:
            tried.append("sqlite:error")
        # file
        try:
            if os.path.exists(WOLF_STATE_FILE):
                with open(WOLF_STATE_FILE, encoding="utf-8") as f:
                    data = json.load(f) or {}
                _restore_from_data(data)
                return
        except Exception as e:
            LOGGER.warning(
                "persist_load_file_error",
                extra={
                    "component": "persist",
                    "error": str(e),
                    "path": WOLF_STATE_FILE,
                },
            )
        return
    if mode == "file":
        try:
            if os.path.exists(WOLF_STATE_FILE):
                with open(WOLF_STATE_FILE, encoding="utf-8") as f:
                    data = json.load(f) or {}
                _restore_from_data(data)
        except Exception as e:
            LOGGER.warning(
                "persist_load_file_error",
                extra={
                    "component": "persist",
                    "error": str(e),
                    "path": WOLF_STATE_FILE,
                },
            )
    elif mode == "redis" and REDIS_URL:
        try:
            import redis  # type: ignore

            r = redis.Redis.from_url(REDIS_URL)
            raw = r.get("wolf:position")
            if raw:
                data = json.loads(raw)  # type: ignore
                _restore_from_data(data)
        except Exception as e:
            LOGGER.warning(
                "persist_load_redis_error",
                extra={"component": "persist", "error": str(e)},
            )
    elif mode == "sqlite":
        try:
            import sqlite3

            _ensure_dir_for_file(WOLF_SQLITE_PATH)
            conn = sqlite3.connect(WOLF_SQLITE_PATH)
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT)")
            conn.commit()
            cur.execute("SELECT value FROM state WHERE key='position'")
            row = cur.fetchone()
            if row and row[0]:
                data = json.loads(row[0])
                _restore_from_data(data)
            conn.close()
        except Exception as e:
            LOGGER.warning(
                "persist_load_sqlite_error",
                extra={
                    "component": "persist",
                    "error": str(e),
                    "path": WOLF_SQLITE_PATH,
                },
            )


def _persist_save():
    # Persist complete portfolio state: positions, cash, legacy qty/avg_cost
    portfolio_state = {
        "qty": STATE.get("qty", 0.0),
        "avg_cost": STATE.get("avg_cost", 0.0),
        "positions": STATE.get("positions", []),
        "cash": STATE.get("cash", 0.0),
    }
    # Optional split cash buckets
    if "cash_stock" in STATE:
        portfolio_state["cash_stock"] = STATE.get("cash_stock", 0.0)
    if "cash_crypto" in STATE:
        portfolio_state["cash_crypto"] = STATE.get("cash_crypto", 0.0)

    # Save to new portfolio persistence layer
    if PORTFOLIO_PERSISTENCE_ENABLED:
        try:
            store = get_portfolio_store()
            qty, avg = _get_portfolio_qty_and_avg()  # Use helper to read from positions array
            if qty > 0 or avg > 0:
                # Get last known price from cache
                cached = PRICE_CACHE.get(WOLF.upper())
                last_price = cached.get("price") if cached else None
                provider = cached.get("provider") if cached else None
                store.save_position(WOLF, qty, avg, last_price, provider)
            # Save cash
            cash = float(STATE.get("cash", 0.0))
            if cash != 0:
                store.save_cash_balance(cash)
        except Exception as e:
            LOGGER.warning("portfolio_persistence_save_failed", extra={"error": str(e)})

    payload = json.dumps(portfolio_state)
    mode = WOLF_PERSIST_MODE
    if mode == "auto":
        # prefer redis, then sqlite, then file
        if REDIS_URL:
            try:
                import redis  # type: ignore

                r = redis.Redis.from_url(REDIS_URL)
                r.set("wolf:position", payload)
                return
            except Exception:
                pass
        try:
            import sqlite3

            _ensure_dir_for_file(WOLF_SQLITE_PATH)
            conn = sqlite3.connect(WOLF_SQLITE_PATH)
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT)")
            conn.commit()
            cur.execute(
                "INSERT INTO state(key, value) VALUES('position', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (payload,),
            )
            conn.commit()
            conn.close()
            return
        except Exception:
            pass
        # fallback to file
        try:
            _ensure_dir_for_file(WOLF_STATE_FILE)
            with open(WOLF_STATE_FILE, "w", encoding="utf-8") as f:
                f.write(payload)
            return
        except Exception as e:
            LOGGER.warning(
                "persist_save_file_error",
                extra={
                    "component": "persist",
                    "error": str(e),
                    "path": WOLF_STATE_FILE,
                },
            )
        return
    if mode == "file":
        try:
            _ensure_dir_for_file(WOLF_STATE_FILE)
            with open(WOLF_STATE_FILE, "w", encoding="utf-8") as f:
                f.write(payload)
        except Exception as e:
            LOGGER.warning(
                "persist_save_file_error",
                extra={
                    "component": "persist",
                    "error": str(e),
                    "path": WOLF_STATE_FILE,
                },
            )
    elif mode == "redis" and REDIS_URL:
        try:
            import redis  # type: ignore

            r = redis.Redis.from_url(REDIS_URL)
            r.set("wolf:position", payload)
        except Exception as e:
            LOGGER.warning(
                "persist_save_redis_error",
                extra={"component": "persist", "error": str(e)},
            )
    elif mode == "sqlite":
        try:
            import sqlite3

            _ensure_dir_for_file(WOLF_SQLITE_PATH)
            conn = sqlite3.connect(WOLF_SQLITE_PATH)
            cur = conn.cursor()
            cur.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT)")
            conn.commit()
            cur.execute(
                "INSERT INTO state(key, value) VALUES('position', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (payload,),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            LOGGER.warning(
                "persist_save_sqlite_error",
                extra={
                    "component": "persist",
                    "error": str(e),
                    "path": WOLF_SQLITE_PATH,
                },
            )


def _autosave_loop():
    if WOLF_AUTOSAVE_S <= 0:
        return
    while not _AUTOSAVE_STOP.is_set():
        try:
            _heartbeat_pulse("autosave-worker")
            time.sleep(max(1, WOLF_AUTOSAVE_S))
            _persist_save()
        except Exception:
            pass


def _start_autosave_worker():
    global _AUTOSAVE_WORKER
    if WOLF_AUTOSAVE_S <= 0:
        return
    if _AUTOSAVE_WORKER is None or not _AUTOSAVE_WORKER.is_alive():
        _AUTOSAVE_WORKER = threading.Thread(
            target=_autosave_loop, name="autosave-worker", daemon=True
        )
        _AUTOSAVE_WORKER.start()


def _stop_autosave_worker():
    try:
        _AUTOSAVE_STOP.set()
        if _AUTOSAVE_WORKER and _AUTOSAVE_WORKER.is_alive():
            _AUTOSAVE_WORKER.join(timeout=2.0)
    except Exception:
        pass


def _get_volatility_lookback() -> float | None:
    now = time.time()
    try:
        if (
            ALERT_STATE.get("last_vol") is not None
            and (now - float(ALERT_STATE.get("vol_ts", 0.0))) <= VOL_TTL_S
        ):
            return float(ALERT_STATE["last_vol"])  # daily returns stddev
        import yfinance as yf

        tkr = yf.Ticker(WOLF)
        hist = tkr.history(period=f"{max(5, VOL_LOOKBACK_DAYS + 5)}d")
        closes = list(hist["Close"].astype(float)) if not hist.empty else []
        rets: list[float] = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0:
                rets.append((closes[i] / closes[i - 1]) - 1.0)
        if len(rets) >= max(5, VOL_LOOKBACK_DAYS // 2):
            mean = sum(rets) / len(rets)
            var = sum((r - mean) ** 2 for r in rets) / max(1, (len(rets) - 1))
            std = math.sqrt(var)
        else:
            std = None
        ALERT_STATE["last_vol"] = std
        ALERT_STATE["vol_ts"] = now
        return std
    except Exception:
        return None


def _send_telegram_internal(card: str, capture: bool = False) -> tuple[bool, list[dict[str, Any]]]:
    """Send Telegram notification and optionally capture per-chat delivery detail."""
    deliveries: list[dict[str, Any]] = []
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False, deliveries

    ok_all = True
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        chats: list[str] = [c.strip() for c in TELEGRAM_CHAT_ID.split(",") if c.strip()]
        if not chats:
            chats = [TELEGRAM_CHAT_ID]
        for chat_id in chats:
            t0 = time.perf_counter()
            entry: dict[str, Any] = {"chat_id": chat_id}
            try:
                # Detect if text contains HTML tags — if not, send as plain text
                # to avoid Telegram HTML parser choking on <, >, & in plain messages
                import re as _re
                _has_html = bool(_re.search(r"<[a-zA-Z/]", card))
                payload = {
                    "chat_id": chat_id,
                    "text": card,
                    "disable_web_page_preview": True,
                }
                if _has_html:
                    payload["parse_mode"] = "HTML"
                r = _http_post(url, json=payload, timeout=8)
                latency = time.perf_counter() - t0
                raw_response: Any = None
                try:
                    raw_response = r.json()
                except Exception:
                    raw_response = (r.text or "")[:500]
                ok = (
                    bool((raw_response or {}).get("ok"))
                    if isinstance(raw_response, dict)
                    else False
                )
                entry.update(
                    {
                        "status": r.status_code,
                        "latency_s": round(latency, 3),
                        "ok": ok,
                        "response": raw_response,
                    }
                )
                try:
                    if _H_TG_SEND is not None:
                        _H_TG_SEND.observe(latency)
                    if _C_TG_SEND is not None:
                        _C_TG_SEND.labels(result=("ok" if ok else "fail")).inc()
                except Exception:
                    pass
                log_extra = {
                    "component": "alert",
                    "chat_id": chat_id,
                    "status": r.status_code,
                    "ok": ok,
                }
                if not ok:
                    ok_all = False
                    LOGGER.warning("telegram_send_failed", extra=log_extra)
                else:
                    LOGGER.info("telegram_send_ok", extra=log_extra)
            except Exception as exc:  # noqa: BLE001
                ok_all = False
                entry.update({"ok": False, "error": str(exc)})
                LOGGER.warning(
                    "telegram_send_exception", extra={"component": "alert", "chat_id": chat_id}
                )
            deliveries.append(entry)
        if capture:
            return ok_all, deliveries
        return ok_all, []
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "telegram_send_exception",
            extra={"component": "alert", "error": str(exc)},
        )
        if capture:
            return False, deliveries
        return False, []


def send_telegram(card: str) -> bool:
    """Send Telegram notification to configured chat(s)."""
    ok, _ = _send_telegram_internal(card, capture=False)
    return ok


def send_telegram_detailed(card: str) -> tuple[bool, list[dict[str, Any]]]:
    """Send Telegram notification and return per-chat delivery diagnostics."""
    return _send_telegram_internal(card, capture=True)


def _tg_send_chat_message(chat_id: str, text: str) -> bool:
    """Helper to send a single Telegram message to a specific chat."""
    if not TELEGRAM_BOT_TOKEN:
        return False
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        # Detect if text contains HTML tags — if not, send as plain text
        # to avoid Telegram HTML parser choking on <, >, & in plain messages
        import re as _re
        _has_html = bool(_re.search(r"<[a-zA-Z/]", text))
        payload = {
            "chat_id": chat_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        if _has_html:
            payload["parse_mode"] = "HTML"
        r = _http_post(url, json=payload, timeout=8)
        return bool((r.json() or {}).get("ok"))
    except Exception:
        return False


def _rank_opportunities(predictions: list[dict]) -> dict[str, list[dict]]:
    """
    Rank predictions by potential gain and confidence.
    Filters out noise and returns only HIGH-CONVICTION opportunities.

    Returns:
        {
            "short_term": [...],  # 48h-7 day quick gains (top 5)
            "long_term": [...],   # 1-6 month strategic holds (top 5)
            "urgent_sells": [...] # Immediate sell signals (top 3)
        }
    """
    buys = []
    sells = []

    for pred in predictions:
        # Skip if no price or no signal
        if not pred.get("price_current") or pred.get("direction") == "HOLD":
            continue

        # Calculate potential gain percentage
        current_price = pred.get("price_current", 0)
        predicted_price = pred.get("price_pred_mid", 0)

        if current_price and predicted_price:
            gain_pct = ((predicted_price - current_price) / current_price) * 100
        else:
            gain_pct = 0

        confidence = pred.get("confidence", 0)
        momentum = abs(pred.get("momentum", 0))

        # Calculate opportunity score (gain × confidence × momentum)
        # Higher score = better opportunity
        score = abs(gain_pct) * confidence * (1 + momentum)

        pred_with_score = pred.copy()
        pred_with_score["gain_pct"] = gain_pct
        pred_with_score["score"] = score

        if pred.get("direction") == "BUY":
            buys.append(pred_with_score)
        elif pred.get("direction") == "SELL":
            sells.append(pred_with_score)

    # Sort by score (highest first)
    buys.sort(key=lambda x: x["score"], reverse=True)
    sells.sort(key=lambda x: x["score"], reverse=True)

    # Filter for quality (UPDATED FOR 6H PREDICTIONS):
    # Short-term (6h): confidence >45% + gain >1% (realistic for 6h timeframe)
    # Long-term: confidence >50% + gain >2% (6h can have smaller moves)
    short_term = [p for p in buys if p["confidence"] > 0.45 and abs(p["gain_pct"]) > 1.0 and p["momentum"] > 0.3][:5]
    long_term = [p for p in buys if p["confidence"] > 0.50 and abs(p["gain_pct"]) > 2.0][:5]
    urgent_sells = sells[:3]  # Top 3 sell signals

    return {
        "short_term": short_term,
        "long_term": long_term,
        "urgent_sells": urgent_sells
    }


def _format_multi_symbol_telegram_message(predictions_data: dict[str, Any]) -> str:
    """
    Format multi-symbol prediction data into a Telegram message.
    INTELLIGENT FILTERING: Only shows TOP opportunities, not noise.
    - Top 5 short-term gains (48h-7 days)
    - Top 5 long-term holds (1-6 months)
    - Top 3 urgent sells

    Args:
        predictions_data: Output from _generate_multi_symbol_predictions()

    Returns:
        HTML-formatted Telegram message string
    """
    if not predictions_data.get("ok"):
        return "⚠️ <b>Multi-Symbol Predictions Failed</b>\n\nError: " + predictions_data.get("error", "Unknown error")

    predictions = predictions_data.get("predictions", {})

    # Combine stocks and crypto for unified ranking
    all_predictions = predictions.get("stocks", []) + predictions.get("crypto", [])

    # Rank opportunities (filter noise)
    opportunities = _rank_opportunities(all_predictions)

    # Build message header
    now_str = datetime.now(ZoneInfo("America/Chicago") if ZoneInfo else None).strftime("%I:%M %p %Z") if ZoneInfo else datetime.now().strftime("%I:%M %p")

    # Get REAL accuracy from database (no lies!)
    try:
        import sqlite3
        from services import predictor
        conn = sqlite3.connect(predictor.DB_PATH)
        total_predictions = conn.execute("SELECT COUNT(*) FROM predictions WHERE run_at >= ?", (time.time() - 30*24*3600,)).fetchone()[0]
        correct_predictions = conn.execute(
            "SELECT COUNT(*) FROM outcomes o JOIN predictions p ON o.prediction_id = p.id WHERE p.run_at >= ? AND o.hit_direction = 1",
            (time.time() - 30*24*3600,)
        ).fetchone()[0]
        conn.close()
        
        if total_predictions > 0 and correct_predictions > 0:
            accuracy_pct = int((correct_predictions / total_predictions) * 100)
            accuracy_status = f"🎯 {accuracy_pct}% Accuracy ({correct_predictions}/{total_predictions} correct)"
        elif total_predictions > 0:
            accuracy_status = f"📊 Evaluating ({total_predictions} predictions pending outcome)"
        else:
            accuracy_status = "🔄 Building prediction history (no evaluations yet)"
    except Exception as e:
        LOGGER.error(f"Accuracy query failed: {e}", exc_info=True)
        accuracy_status = "⚠️ Accuracy unavailable (0 predictions evaluated yet)"

    message = f"""🎯 <b>GHOST AI TRADING SIGNALS</b>
⏰ {now_str}
{accuracy_status}

"""

    # Helper function to format momentum indicator
    def format_momentum(pred: dict[str, Any]) -> str:
        """Format momentum indicator for prediction"""
        momentum = pred.get("momentum", {})
        if not momentum or momentum.get("status") == "STABLE":
            return ""
        
        emoji = momentum.get("emoji", "")
        delta_pct = momentum.get("confidence_delta_pct", 0)
        
        if delta_pct > 0:
            return f" {emoji} +{delta_pct:.1f}%"
        else:
            return f" {emoji} {delta_pct:.1f}%"

    # SHORT-TERM OPPORTUNITIES (48h-7 days)
    short_term = opportunities.get("short_term", [])
    if short_term:
        message += "<b>⚡ SHORT-TERM GAINS (48h-7 days)</b>\n"
        for i, pred in enumerate(short_term, 1):
            symbol = pred.get("symbol")
            price = pred.get("price_current")
            predicted = pred.get("price_pred_mid")
            gain_pct = pred.get("gain_pct", 0)
            confidence = pred.get("confidence", 0) * 100
            asset_type = "💎" if pred.get("type") == "crypto" else "📈"
            momentum_str = format_momentum(pred)

            message += f"{i}. {asset_type} <b>{symbol}</b>{momentum_str}\n"
            message += f"   💰 ${price:.2f} → ${predicted:.2f} (+{gain_pct:.1f}%)\n"
            message += f"   ✅ Confidence: {confidence:.0f}%\n\n"
    else:
        message += "<b>⚡ SHORT-TERM GAINS</b>\n"
        message += "   No high-conviction short-term plays right now.\n\n"

    # LONG-TERM OPPORTUNITIES (1-6 months)
    long_term = opportunities.get("long_term", [])
    if long_term:
        message += "<b>🎯 LONG-TERM HOLDS (1-6 months)</b>\n"
        for i, pred in enumerate(long_term, 1):
            symbol = pred.get("symbol")
            price = pred.get("price_current")
            predicted = pred.get("price_pred_mid")
            gain_pct = pred.get("gain_pct", 0)
            confidence = pred.get("confidence", 0) * 100
            asset_type = "💎" if pred.get("type") == "crypto" else "📈"
            momentum_str = format_momentum(pred)

            message += f"{i}. {asset_type} <b>{symbol}</b>{momentum_str}\n"
            message += f"   💰 ${price:.2f} → ${predicted:.2f} (+{gain_pct:.1f}%)\n"
            message += f"   ✅ Confidence: {confidence:.0f}%\n\n"
    else:
        message += "<b>🎯 LONG-TERM HOLDS</b>\n"
        message += "   No high-conviction long-term plays right now.\n\n"

    # URGENT SELLS
    urgent_sells = opportunities.get("urgent_sells", [])
    if urgent_sells:
        message += "<b>🚨 URGENT SELLS</b>\n"
        for i, pred in enumerate(urgent_sells, 1):
            symbol = pred.get("symbol")
            price = pred.get("price_current")
            predicted = pred.get("price_pred_mid")
            gain_pct = pred.get("gain_pct", 0)
            confidence = pred.get("confidence", 0) * 100
            asset_type = "💎" if pred.get("type") == "crypto" else "📈"
            momentum_str = format_momentum(pred)

            message += f"{i}. {asset_type} <b>{symbol}</b>{momentum_str}\n"
            message += f"   ⚠️ ${price:.2f} → ${predicted:.2f} ({gain_pct:.1f}%)\n"
            message += f"   ✅ Confidence: {confidence:.0f}%\n\n"

    # Footer
    total_opps = len(short_term) + len(long_term) + len(urgent_sells)
    if total_opps == 0:
        message += "💤 <b>Market Status: HOLDING PATTERN</b>\n"
        message += "No high-conviction signals. Wait for better setups.\n\n"

    message += "💡 <i>Ghost AI filters out noise. Only see high-confidence 6h signals (>45%).</i>\n"
    message += "📊 <i>Momentum indicators: 🔥=HOT (strengthening), 📈=Warming, 📉=Cooling, ❄️=COLD (weakening)</i>"

    return message


def _format_multi_symbol_telegram_message_legacy(predictions_data: dict[str, Any]) -> str:
    """Legacy format showing all predictions (unfiltered)."""
    if not predictions_data.get("ok"):
        return "⚠️ <b>Multi-Symbol Predictions Failed</b>\n\nError: " + predictions_data.get("error", "Unknown error")

    predictions = predictions_data.get("predictions", {})
    counts = predictions_data.get("counts", {})

    # Build message header
    now_str = datetime.now(ZoneInfo("America/Chicago") if ZoneInfo else None).strftime("%I:%M %p %Z") if ZoneInfo else datetime.now().strftime("%I:%M %p")

    message = f"""📊 <b>GHOST MULTI-SYMBOL PREDICTIONS</b>
⏰ Time: {now_str}
📈 Total: {counts.get('stocks', 0)} stocks, {counts.get('crypto', 0)} crypto, {counts.get('vip', 0)} VIP

"""

    # Format STOCKS group
    stocks = predictions.get("stocks", [])
    if stocks:
        message += "<b>📈 STOCKS</b>\n"
        for pred in stocks:
            symbol = pred.get("symbol", "???")
            direction = pred.get("direction", "HOLD")
            confidence = pred.get("confidence", 0) * 100
            price = pred.get("price_current")

            # Direction emoji
            if direction == "BUY":
                emoji = "🟢"
            elif direction == "SELL":
                emoji = "🔴"
            else:
                emoji = "⚪"

            if price:
                message += f"{emoji} {symbol}: {direction} (${price:.2f}, {confidence:.0f}%)\n"
            else:
                message += f"{emoji} {symbol}: {direction} (NO DATA, {confidence:.0f}%)\n"
        message += "\n"

    # Format CRYPTO group
    crypto = predictions.get("crypto", [])
    if crypto:
        message += "<b>💎 CRYPTO</b>\n"
        for pred in crypto:
            symbol = pred.get("symbol", "???")
            direction = pred.get("direction", "HOLD")
            confidence = pred.get("confidence", 0) * 100
            price = pred.get("price_current")

            if direction == "BUY":
                emoji = "🟢"
            elif direction == "SELL":
                emoji = "🔴"
            else:
                emoji = "⚪"

            if price:
                message += f"{emoji} {symbol}: {direction} (${price:.2f}, {confidence:.0f}%)\n"
            else:
                message += f"{emoji} {symbol}: {direction} (NO DATA, {confidence:.0f}%)\n"
        message += "\n"

    # Format VIP group
    vip = predictions.get("vip", [])
    if vip:
        message += "<b>⭐ VIP COINS</b>\n"
        for pred in vip:
            symbol = pred.get("symbol", "???")
            direction = pred.get("direction", "HOLD")
            confidence = pred.get("confidence", 0) * 100
            price = pred.get("price_current")

            if direction == "BUY":
                emoji = "🟢"
            elif direction == "SELL":
                emoji = "🔴"
            else:
                emoji = "⚪"

            if price:
                message += f"{emoji} {symbol}: {direction} (${price:.2f}, {confidence:.0f}%)\n"
            else:
                message += f"{emoji} {symbol}: {direction} (NO DATA, {confidence:.0f}%)\n"
        message += "\n"

    # Add footer
    if not stocks and not crypto and not vip:
        message += "⚠️ No prediction data available (check API keys)\n"
    else:
        message += "💡 <i>Live predictions from Ghost Protocol</i>"

    return message


def _send_multi_symbol_telegram_alert() -> bool:
    """
    Generate and send multi-symbol predictions via Telegram.
    Updates global tracking state.

    Returns:
        True if send succeeded, False otherwise
    """
    global _LAST_TELEGRAM_SEND_TIME, _LAST_TELEGRAM_STATUS, _LAST_TELEGRAM_ERROR

    try:
        # Generate predictions
        predictions_data = _generate_multi_symbol_predictions()

        # Format message
        message = _format_multi_symbol_telegram_message(predictions_data)

        # Send via Telegram
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            success = _tg_send_chat_message(TELEGRAM_CHAT_ID, message)

            # Update tracking
            _LAST_TELEGRAM_SEND_TIME = time.time()
            if success:
                _LAST_TELEGRAM_STATUS = "ok"
                _LAST_TELEGRAM_ERROR = None
                LOGGER.info("Multi-symbol Telegram alert sent successfully")
            else:
                _LAST_TELEGRAM_STATUS = "error"
                _LAST_TELEGRAM_ERROR = "Telegram API returned failure"
                LOGGER.warning("Multi-symbol Telegram alert failed")

            return success
        else:
            _LAST_TELEGRAM_STATUS = "error"
            _LAST_TELEGRAM_ERROR = "Telegram credentials not configured"
            LOGGER.warning("Cannot send Telegram alert: credentials missing")
            return False

    except Exception as e:
        _LAST_TELEGRAM_SEND_TIME = time.time()
        _LAST_TELEGRAM_STATUS = "error"
        _LAST_TELEGRAM_ERROR = str(e)[:200]
        LOGGER.exception("Multi-symbol Telegram alert failed with exception")
        return False


def post_webhooks(text: str) -> None:
    for u in ALERT_WEBHOOK_URLS:
        try:
            _http_post(u, json={"text": text}, timeout=6)
        except Exception:
            LOGGER.warning("webhook_post_failed", extra={"component": "alert", "sink": "webhook"})


def post_slack(text: str) -> None:
    for u in SLACK_WEBHOOK_URLS:
        try:
            _http_post(u, json={"text": text}, timeout=6)
        except Exception:
            LOGGER.warning("webhook_post_failed", extra={"component": "alert", "sink": "slack"})


def _alert_worker_loop():
    _heartbeat_pulse("alert-worker")  # Pulse immediately so Health tab shows alive
    while not _ALERT_STOP.is_set():
        _heartbeat_pulse("alert-worker")
        try:
            item = _ALERT_QUEUE.get(timeout=0.5)
        except _queue.Empty:
            continue
        try:
            text = item.get("text")
            if not text or not isinstance(text, str):
                continue
            sig = item.get("sig") or {}
            attempt = int(item.get("attempt", 1))
            ok = False
            try:
                ok = send_telegram(text)
                # Fan-out sinks best-effort
                try:
                    if ALERT_WEBHOOK_URLS:
                        post_webhooks(text)
                except Exception:
                    pass
                try:
                    if SLACK_WEBHOOK_URLS:
                        post_slack(text)
                except Exception:
                    pass
            finally:
                try:
                    if _C_ALERT_SENT is not None:
                        _C_ALERT_SENT.labels(
                            action=(sig.get("action") or "?"),
                            mode=(sig.get("mode") or "?"),
                            result=("ok" if ok else "fail"),
                        ).inc()
                except Exception:
                    pass
            if not ok and attempt < 5:
                try:
                    if _C_ALERT_RETRIES is not None:
                        _C_ALERT_RETRIES.inc()
                except Exception:
                    pass
                delay = min(60.0, 2 ** (attempt - 1))
                try:
                    time.sleep(delay)
                except Exception:
                    pass
                try:
                    item["attempt"] = attempt + 1
                    _ALERT_QUEUE.put(item, timeout=0.1)
                except Exception:
                    pass
        finally:
            try:
                _ALERT_QUEUE.task_done()
            except Exception:
                pass
            try:
                if _G_ALERT_QUEUE_LEN is not None:
                    _G_ALERT_QUEUE_LEN.set(_ALERT_QUEUE.qsize())
            except Exception:
                pass


def _start_alert_worker():
    global _ALERT_WORKER
    if _ALERT_WORKER is None or not _ALERT_WORKER.is_alive():
        _ALERT_WORKER = threading.Thread(
            target=_alert_worker_loop, name="alert-worker", daemon=True
        )
        _ALERT_WORKER.start()


def _stop_alert_worker():
    try:
        _ALERT_STOP.set()
        if _ALERT_WORKER and _ALERT_WORKER.is_alive():
            _ALERT_WORKER.join(timeout=2.0)
    except Exception:
        pass


def enqueue_alert_text(text: str, sig: dict[str, Any] | None = None) -> bool:
    try:
        _ALERT_QUEUE.put({"text": text, "sig": sig or {}, "attempt": 1}, timeout=0.1)
        try:
            if _G_ALERT_QUEUE_LEN is not None:
                _G_ALERT_QUEUE_LEN.set(_ALERT_QUEUE.qsize())
        except Exception:
            pass
        return True
    except Exception:
        LOGGER.warning("alert_queue_full", extra={"component": "alert"})
        return False


def _should_skip_request_log(path: str) -> bool:
    try:
        return any((path or "").startswith(p) for p in LOG_SKIP_PATHS)
    except Exception:
        return False


def _start_schedule_worker():
    global _SCHED_WORKER
    if _SCHED_WORKER is None or not _SCHED_WORKER.is_alive():
        _SCHED_STOP.clear()
        _SCHED_WORKER = threading.Thread(
            target=_schedule_loop, name="open-close-scheduler", daemon=True
        )
        _SCHED_WORKER.start()


def _stop_schedule_worker():
    try:
        _SCHED_STOP.set()
        if _SCHED_WORKER and _SCHED_WORKER.is_alive():
            _SCHED_WORKER.join(timeout=2.0)
    except Exception:
        pass


def _schedule_loop():
    global _SCHED_LAST_OPEN_DAY, _SCHED_LAST_CLOSE_DAY
    while not _SCHED_STOP.is_set():
        _heartbeat_pulse("open-close-scheduler")
        try:
            now_ny = _ny_now()
            wd = now_ny.weekday()
            if wd <= 4:  # Mon-Fri
                open_dt = now_ny.replace(hour=9, minute=30, second=0, microsecond=0)
                close_dt = now_ny.replace(hour=16, minute=0, second=0, microsecond=0)
                dt_open = abs((now_ny - open_dt).total_seconds())
                dt_close = abs((now_ny - close_dt).total_seconds())
                day_key = now_ny.strftime("%Y-%m-%d")
                # OPEN window
                if dt_open <= SCHEDULE_WINDOW_S and _SCHED_LAST_OPEN_DAY != day_key:
                    try:
                        price, prev, provider = get_wolf_price()
                        base = _build_status_card(price=price, provider=provider, include_req=False)
                        prefix = "🟢 OPEN — WOLF\n"
                        text = prefix + (base.split("\n", 1)[1] if "\n" in base else base)
                        enqueue_alert_text(text, {"action": "STATUS", "mode": ALERT_MODE})
                        _SCHED_LAST_OPEN_DAY = day_key
                    except Exception:
                        LOGGER.exception("schedule_open_send_failed", extra={"component": "alert"})
                # CLOSE window
                if dt_close <= SCHEDULE_WINDOW_S and _SCHED_LAST_CLOSE_DAY != day_key:
                    try:
                        price, prev, provider = get_wolf_price()
                        base = _build_status_card(price=price, provider=provider, include_req=False)
                        prefix = "🔴 CLOSE — WOLF\n"
                        text = prefix + (base.split("\n", 1)[1] if "\n" in base else base)
                        enqueue_alert_text(text, {"action": "STATUS", "mode": ALERT_MODE})
                        _SCHED_LAST_CLOSE_DAY = day_key
                    except Exception:
                        LOGGER.exception("schedule_close_send_failed", extra={"component": "alert"})
        except Exception:
            LOGGER.exception("schedule_loop_failed", extra={"component": "alert"})
        finally:
            _SCHED_STOP.wait(30.0)


def _start_reconciler_worker():
    """Start background thread to reconcile prediction outcomes every 5 min"""
    global _RECONCILER_WORKER
    if _RECONCILER_WORKER is None or not _RECONCILER_WORKER.is_alive():
        _RECONCILER_STOP.clear()
        _RECONCILER_WORKER = threading.Thread(
            target=_reconciler_loop, name="outcome-reconciler", daemon=True
        )
        _RECONCILER_WORKER.start()
        LOGGER.info("Prediction outcome reconciler started")


def _stop_reconciler_worker():
    """Stop reconciler worker gracefully"""
    try:
        _RECONCILER_STOP.set()
        if _RECONCILER_WORKER and _RECONCILER_WORKER.is_alive():
            _RECONCILER_WORKER.join(timeout=2.0)
    except Exception:
        pass


def _reconciler_loop():
    """Background loop to reconcile prediction outcomes and append actual prices"""
    # Sleep first on startup to avoid blocking server initialization
    time.sleep(60)  # Wait 60s for server to fully start before first run
    
    while not _RECONCILER_STOP.is_set():
        _heartbeat_pulse("outcome-reconciler")
        try:
            # 1. Append actual prices to active predictions
            _append_actual_prices()

            # 2. Reconcile outcomes for expired predictions
            if outcome_reconciler is not None:
                outcome_reconciler.reconcile_outcomes()
            else:
                from services.outcome_reconciler_v2 import reconcile_outcomes_v2
                reconcile_outcomes_v2()
        except Exception as e:
            LOGGER.error(f"Outcome reconciler error: {e}", exc_info=True)
        finally:
            # Wait 5 minutes between reconciliation runs
            _RECONCILER_STOP.wait(300.0)


def _start_accuracy_tracker():
    """Start background thread to track accuracy snapshots every 5 min"""
    global _ACCURACY_TRACKER
    if _ACCURACY_TRACKER is None or not _ACCURACY_TRACKER.is_alive():
        _ACCURACY_STOP.clear()
        _ACCURACY_TRACKER = threading.Thread(
            target=_accuracy_tracking_loop, name="accuracy-tracker", daemon=True
        )
        _ACCURACY_TRACKER.start()
        LOGGER.info("Live accuracy tracker started")


def _stop_accuracy_tracker():
    """Stop accuracy tracker gracefully"""
    try:
        _ACCURACY_STOP.set()
        if _ACCURACY_TRACKER and _ACCURACY_TRACKER.is_alive():
            _ACCURACY_TRACKER.join(timeout=2.0)
    except Exception:
        pass


def _accuracy_tracking_loop():
    """Background loop to record accuracy snapshots for trending analysis"""
    _heartbeat_pulse("accuracy-tracker")  # Pulse immediately so Health tab shows alive
    # Sleep first on startup to avoid blocking server initialization
    time.sleep(120)  # Wait 2 minutes for server to fully start
    
    while not _ACCURACY_STOP.is_set():
        _heartbeat_pulse("accuracy-tracker")
        try:
            from core.accuracy_tracking import record_accuracy_snapshot
            record_accuracy_snapshot()
        except Exception as e:
            LOGGER.error(f"Accuracy tracking error: {e}", exc_info=True)
        finally:
            # Record snapshot every 5 minutes
            _ACCURACY_STOP.wait(300.0)


def _get_price_quorum(symbol: str, asset_type: str = "stock") -> dict[str, Any] | None:
    """Lightweight price fetcher with Polygon-first priority and Redis fallback."""
    sym = symbol.upper().strip()
    if asset_type != "stock":
        return None
    if sym == WOLF:
        price, prev, provider = get_wolf_price()
        if price is None and prev is not None:
            price = prev
        if price is None:
            return None
        return {"symbol": sym, "price": float(price), "prev_close": prev, "provider": provider}
    
    # PRIORITY INVERSION: yfinance → Yahoo → Polygon → AlphaVantage
    # yfinance FIRST since it's most reliable and uses different endpoints
    providers: list[tuple[str, Any]] = []
    
    # PRIMARY: yfinance library (most reliable, FREE)
    providers.append(("yfinance", lambda: _fetch_price_yfinance(sym)))
    
    # SECONDARY: Yahoo Finance HTTP (free, rate-limited)
    providers.append(("yahoo", lambda: _fetch_price_yahoo_http(sym)))
    
    # TERTIARY: Polygon (requires API key, only if configured)
    if POLYGON_KEY:
        providers.append(("polygon", lambda: _fetch_price_polygon(sym)))
    
    # QUATERNARY: AlphaVantage (if configured)
    if ALPHAVANTAGE_KEY:
        providers.append(("alphavantage", lambda: _fetch_price_alphavantage(sym)))

    failed_providers = []
    for name, fetcher in providers:
        try:
            price, prev, provider = fetcher()
        except Exception as e:  # noqa: BLE001
            error_msg = str(e)
            failed_providers.append({"provider": name, "error": error_msg})
            LOGGER.warning(
                "price_provider_failed",
                extra={"symbol": sym, "provider": name, "error": error_msg, "failed_count": len(failed_providers)},
            )
            try:
                _add_event(
                    "price_quorum.error",
                    f"{sym}:{name}",
                    {"symbol": sym, "provider": name, "error": error_msg},
                )
            except Exception:
                pass
            continue
        if price and price > 0:
            LOGGER.info(
                "price_quorum_success",
                extra={
                    "component": "price",
                    "symbol": sym,
                    "provider": provider or name,
                    "price": float(price),
                    "prev_close": float(prev) if prev else None,
                    "failed_providers": len(failed_providers),
                },
            )
            return {
                "symbol": sym,
                "price": float(price),
                "prev_close": (None if prev is None else float(prev)),
                "provider": provider or name,
            }
        if prev and prev > 0:
            LOGGER.info(
                "price_quorum_success_prev",
                extra={
                    "component": "price",
                    "symbol": sym,
                    "provider": f"{provider or name}:prev",
                    "price": float(prev),
                    "failed_providers": len(failed_providers),
                },
            )
            return {
                "symbol": sym,
                "price": float(prev),
                "prev_close": float(prev),
                "provider": f"{provider or name}:prev",
            }
    
    # ALL PROVIDERS FAILED - Try Polygon 3 more times with backoff
    if POLYGON_KEY:
        LOGGER.warning(
            "price_all_failed_retrying_polygon",
            extra={"symbol": sym, "failed_providers": len(failed_providers)},
        )
        import time
        for retry in range(3):
            try:
                time.sleep(0.5 * (retry + 1))  # 0.5s, 1s, 1.5s backoff
                price, prev, provider = _fetch_price_polygon(sym)
                if price and price > 0:
                    LOGGER.info(
                        "price_polygon_retry_success",
                        extra={"symbol": sym, "retry_attempt": retry + 1, "price": float(price)},
                    )
                    return {
                        "symbol": sym,
                        "price": float(price),
                        "prev_close": (None if prev is None else float(prev)),
                        "provider": f"polygon:retry{retry+1}",
                    }
            except Exception as e:
                LOGGER.debug(
                    "price_polygon_retry_failed",
                    extra={"symbol": sym, "retry_attempt": retry + 1, "error": str(e)},
                )
    
    # LAST RESORT: Check Redis cache for last valid price
    try:
        redis_key = f"ghost:price:last:{sym}"
        if _REDIS and _REDIS.exists(redis_key):
            cached_data = _REDIS.get(redis_key)
            if cached_data:
                import json
                cache = json.loads(cached_data)
                cached_price = cache.get("price")
                if cached_price and cached_price > 0:
                    LOGGER.warning(
                        "price_using_redis_cache",
                        extra={"symbol": sym, "price": cached_price, "cache_age_seconds": cache.get("age", 0)},
                    )
                    return {
                        "symbol": sym,
                        "price": float(cached_price),
                        "prev_close": float(cache.get("prev_close", cached_price)),
                        "provider": "redis:cache",
                    }
    except Exception as e:
        LOGGER.debug("redis_cache_check_failed", extra={"symbol": sym, "error": str(e)})
    
    LOGGER.error(
        "price_total_failure",
        extra={"symbol": sym, "failed_providers": failed_providers},
    )
    LOGGER.debug(
        "price_quorum_failed", extra={"symbol": sym, "provider": "all", "error": "no_price"}
    )
    try:
        _add_event(
            "price_quorum.failed",
            sym,
            {"symbol": sym, "providers": [name for name, _ in providers]},
        )
    except Exception:
        pass
    return None


def _append_actual_prices():
    """Append current live prices to active predictions.
    
    CRITICAL: Uses get_prediction_store() to read from correct backend (PostgreSQL or SQLite)
    based on PREDICTION_STORE_ENGINE environment variable.
    """
    from core.prediction_store import get_prediction_store
    
    store = get_prediction_store()
    
    # Get active predictions (window still open, no outcome yet)
    rows = store.get_active_predictions()
    
    if not rows:
        LOGGER.debug("No active predictions to append actual prices to")
        return
    
    LOGGER.info(f"Appending actual prices to {len(rows)} active predictions")
    
    for pred in rows:
        pred_id = pred["id"]
        symbol = pred["symbol"]
        try:
            # Get current price - detect if crypto or stock
            asset_type = "crypto" if symbol in ["BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT", "LTC", "UNI", "ATOM", "MATIC"] else "stock"
            price_data = _get_price_quorum(symbol, asset_type)
            
            if price_data and price_data.get("price"):
                current_price = float(price_data["price"])
                current_ts = time.time()

                # Append as actual point using the store abstraction
                predictor.append_actual_points(pred_id, [(current_ts, current_price)])
                LOGGER.debug(f"Appended actual price ${current_price:.2f} for {symbol} (pred {pred_id})")
        except Exception as e:
            LOGGER.debug(f"Failed to append actual price for prediction {pred_id} ({symbol}): {e}")


async def _async_sleep(seconds: float):
    try:
        import asyncio

        await asyncio.sleep(max(0.0, seconds))
    except Exception:
        # fallback (should not happen in async context)
        time.sleep(max(0.0, seconds))


def _orders_init():
    try:
        import sqlite3

        _ensure_dir_for_file(WOLF_SQLITE_PATH)
        conn = sqlite3.connect(WOLF_SQLITE_PATH)
        cur = conn.cursor()
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {ORDERS_TABLE} (
                id TEXT PRIMARY KEY,
                ts INTEGER,
                symbol TEXT,
                side TEXT,
                qty REAL,
                price REAL,
                status TEXT,
                note TEXT
            )
            """
        )
        conn.commit()
        conn.close()
    except Exception as e:
        LOGGER.warning("orders_init_error", extra={"component": "orders", "error": str(e)})


def _orders_insert(order: dict[str, Any]):
    try:
        import sqlite3

        conn = sqlite3.connect(WOLF_SQLITE_PATH)
        cur = conn.cursor()
        price_val = order.get("price")
        cur.execute(
            f"INSERT INTO {ORDERS_TABLE}(id, ts, symbol, side, qty, price, status, note) VALUES(?,?,?,?,?,?,?,?)",
            (
                order.get("id"),
                int(order.get("ts", int(time.time()))),
                order.get("symbol"),
                order.get("side"),
                float(order.get("qty", 0.0)),
                (None if price_val is None else float(price_val)),
                order.get("status", "queued"),
                order.get("note"),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        LOGGER.warning("orders_insert_error", extra={"component": "orders", "error": str(e)})


def _orders_select(limit: int = 100) -> list[dict]:
    try:
        import sqlite3

        conn = sqlite3.connect(WOLF_SQLITE_PATH)
        cur = conn.cursor()
        cur.execute(
            f"SELECT id, ts, symbol, side, qty, price, status, note FROM {ORDERS_TABLE} ORDER BY ts DESC LIMIT ?",
            (int(limit),),
        )
        rows = cur.fetchall() or []
        conn.close()
        out: list[dict] = []
        for r in rows:
            out.append(
                {
                    "id": r[0],
                    "ts": int(r[1] or 0),
                    "symbol": r[2],
                    "side": r[3],
                    "qty": float(r[4] or 0),
                    "price": (None if r[5] is None else float(r[5])),
                    "status": r[6],
                    "note": r[7],
                }
            )
        return out
    except Exception as e:
        LOGGER.warning("orders_select_error", extra={"component": "orders", "error": str(e)})
        return []


def validate_api_key(api_key: str) -> bool:
    """Validate API key and check rate limits."""
    from collections import deque

    # Find key in database
    key_data = None
    for data in API_KEYS_DB.values():
        if data["key"] == api_key:
            key_data = data
            break

    if not key_data:
        return False

    # Update usage stats
    key_data["last_used"] = time.time()
    key_data["request_count"] = key_data.get("request_count", 0) + 1

    # Check rate limit
    if api_key not in API_KEY_REQUESTS:
        API_KEY_REQUESTS[api_key] = deque()

    now = time.time()
    requests = API_KEY_REQUESTS[api_key]

    # Remove requests older than 1 minute
    while requests and requests[0] < now - 60:
        requests.popleft()

    # Check if rate limit exceeded
    if len(requests) >= key_data["rate_limit"]:
        return False

    # Add current request
    requests.append(now)
    return True


async def dispatch_webhook_event(event_type: str, data: dict, webhook_id: str | None = None):
    """Dispatch an event to registered webhooks."""
    results = []

    webhooks = [WEBHOOK_SUBSCRIPTIONS[webhook_id]] if webhook_id else WEBHOOK_SUBSCRIPTIONS.values()

    for webhook in webhooks:
        if event_type not in webhook["events"] and "*" not in webhook["events"]:
            continue

        timestamp_str = str(int(time.time()))
        payload = {"event": event_type, "timestamp": timestamp_str, "data": data}

        # Canonical JSON for consistent signatures
        raw_body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")

        # Proper HMAC: HMAC-SHA256(secret, "timestamp.body")
        message = f"{timestamp_str}.".encode() + raw_body
        signature = hmac.new(webhook["secret"].encode("utf-8"), message, hashlib.sha256).hexdigest()

        try:
            import httpx

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    webhook["url"],
                    content=raw_body,
                    headers={
                        "X-Ghost-Signature": signature,
                        "X-Ghost-Timestamp": timestamp_str,
                        "X-Ghost-Event": event_type,
                        "Content-Type": "application/json",
                        "User-Agent": "Ghost-Webhook/1.0",
                    },
                )

            webhook["last_success_ts"] = time.time()

            # Update database success timestamp
            try:
                webhook_id_key = [k for k, v in WEBHOOK_SUBSCRIPTIONS.items() if v == webhook][0]
                import sqlite3

                conn = sqlite3.connect(WOLF_SQLITE_PATH)
                cur = conn.cursor()
                cur.execute(
                    "UPDATE webhooks SET last_success_ts=?, failure_count=0 WHERE id=?",
                    (time.time(), webhook_id_key),
                )
                conn.commit()
                conn.close()
            except Exception:
                pass

            results.append(
                {
                    "url": webhook["url"],
                    "success": response.status_code == 200,
                    "status_code": response.status_code,
                }
            )

            LOGGER.info(
                f"Webhook delivered: {event_type} -> {webhook['url']} (HTTP {response.status_code})"
            )

        except Exception as e:
            webhook["failure_count"] = webhook.get("failure_count", 0) + 1

            # Update database failure count
            try:
                webhook_id_key = [k for k, v in WEBHOOK_SUBSCRIPTIONS.items() if v == webhook][0]
                import sqlite3

                conn = sqlite3.connect(WOLF_SQLITE_PATH)
                cur = conn.cursor()
                cur.execute(
                    "UPDATE webhooks SET failure_count=failure_count+1 WHERE id=?",
                    (webhook_id_key,),
                )
                conn.commit()
                conn.close()
            except Exception:
                pass

            results.append({"url": webhook["url"], "success": False, "error": str(e)})

            LOGGER.warning(f"Webhook delivery failed: {event_type} -> {webhook['url']}: {e}")

    return {
        "event": event_type,
        "dispatched": len(results),
        "results": results,
        "success": all(r["success"] for r in results),
    }


def _get_world_context_fallback() -> dict:
    """Provide basic world context when Stage1 unavailable."""
    try:
        from core.world_context import get_world_context
        return get_world_context()
    except Exception:
        return {
            "spy": {"price": None, "change_pct": None, "status": "unavailable"},
            "vix": {"level": None, "status": "unavailable"},
            "news_summary": {"total": 0, "sentiment": "neutral"},
            "timestamp": time.time(),
            "status": "fallback"
        }


def _get_market_mood_fallback() -> dict:
    """Provide basic market mood when Stage1 unavailable."""
    try:
        from core.market_mood import get_market_mood
        mood = get_market_mood()
        return mood if mood else {
            "sentiment": "neutral",
            "score": 50.0,
            "regime": "unknown",
            "factors": ["Market mood data unavailable"],
            "timestamp": time.time()
        }
    except Exception:
        return {
            "sentiment": "neutral",
            "score": 50.0,
            "regime": "unknown",
            "factors": ["Market mood service unavailable"],
            "timestamp": time.time(),
            "status": "fallback"
        }


def _compute_forecast_metrics(fcst: dict, actuals: list[dict]) -> dict[str, Any]:
    """Compute MAP, RMSE, bias, direction match, etc."""
    try:
        pred_mid = fcst.get("path_mid", [])
        if not pred_mid or not actuals:
            return {"map": None, "rmse": None, "bias": None, "accrual_pct": 0}

        # Match actual ticks to predicted timestamps (nearest)
        paired = []
        for a in actuals:
            ts = a.get("t", 0)
            closest = min(pred_mid, key=lambda p: abs(p.get("t", 0) - ts), default=None)
            if closest and abs(closest.get("t", 0) - ts) < 3600:  # within 1h
                paired.append((closest.get("p", 0), a.get("p", 0)))

        if not paired:
            return {"map": None, "rmse": None, "bias": None, "accrual_pct": 0}

        # MAP
        ape = [abs(act - pred) / act * 100 for pred, act in paired if act != 0]
        map = sum(ape) / len(ape) if ape else None

        # RMSE
        se = [(act - pred) ** 2 for pred, act in paired]
        rmse = (sum(se) / len(se)) ** 0.5 if se else None

        # Bias
        errors = [(pred - act) / act * 100 for pred, act in paired if act != 0]
        bias = sum(errors) / len(errors) if errors else None

        # Accrual
        accrual_pct = len(paired) / len(pred_mid) * 100 if pred_mid else 0

        return {
            "map": round(map, 2) if map is not None else None,
            "rmse": round(rmse, 3) if rmse is not None else None,
            "bias": round(bias, 2) if bias is not None else None,
            "accrual_pct": round(accrual_pct, 1),
        }
    except Exception:
        return {"map": None, "rmse": None, "bias": None, "accrual_pct": 0}


async def _watchdog_background_check():
    """
    Background task for watchdog checking.
    Runs after the HTTP response is sent to avoid cron timeout.
    """
    try:
        import asyncio
        from core.ghost_notifications import get_notification_system, get_central_time
        from core.asset_classifier import get_asset_type
        
        LOGGER.info("[WATCHDOG] 🔄 Starting background check...")
        
        # Setup telegram function
        def _send_telegram(msg: str) -> bool:
            return _tg_send_chat_message(TELEGRAM_CHAT_ID, msg)
        
        notif = get_notification_system()
        notif.set_telegram_func(_send_telegram)
        
        # AUTO-RETRY: If using SQLite, try to reconnect to PostgreSQL
        if not notif._use_postgres:
            LOGGER.info("[WATCHDOG] 🔄 Retrying PostgreSQL connection...")
            notif.retry_postgres_connection()
        
        # AUTO-TOP10: DISABLED - Use /debug/send-top10-now instead
        # This was causing duplicate messages with wrong symbols from _LATEST_PREDICTIONS
        # now_central = get_central_time()
        # current_hour = now_central.hour
        # current_date = now_central.strftime("%Y-%m-%d")
        top10_sent = False  # Keep this var, other code may reference it
        
        # DISABLED 8 AM AUTO-SEND - was sending old Money Game symbols
        # status = notif.get_status()
        # if current_hour == 8 and status.get("last_top10_date") != current_date:
        #     LOGGER.info("[WATCHDOG] 🌅 8 AM detected - auto-sending TOP 10...")
        #     try:
        #         stocks, crypto = notif.get_top10_predictions(_LATEST_PREDICTIONS)
        #         if stocks or crypto:
        #             top10_sent = notif.send_top10_message(stocks, crypto, _LATEST_PREDICTIONS)
        #             if top10_sent:
        #                 LOGGER.info("[WATCHDOG] ✅ TOP 10 sent successfully via Watchdog at 8 AM")
        #     except Exception as top10_err:
        #         LOGGER.error(f"[WATCHDOG] TOP 10 auto-send failed: {top10_err}")
        
        # Create ASYNC price lookup function using _LATEST_PREDICTIONS + live refresh
        async def get_current_price(symbol: str) -> float:
            """Get current price, refreshing if stale"""
            # First check in-memory predictions
            if symbol in _LATEST_PREDICTIONS:
                pred = _LATEST_PREDICTIONS[symbol]
                price = pred.get("price") or pred.get("current_price") or pred.get("entry_price") or 0
                if price > 0:
                    return price
            
            # Try live fetch
            try:
                asset_class = get_asset_type(symbol)
                if asset_class.startswith("crypto"):
                    from core.crypto.crypto_providers import get_crypto_price_quorum
                    fresh = await get_crypto_price_quorum(symbol, use_cache=False)
                    if fresh and fresh.get("price", 0) > 0:
                        return fresh["price"]
                else:
                    from core.providers.turbo_provider import get_turbo_provider
                    turbo = get_turbo_provider()
                    fresh = turbo.turbo_stock_price(symbol, max_budget_s=2.0)
                    if fresh.get("ok") and fresh.get("price", 0) > 0:
                        return fresh["price"]
            except Exception as e:
                LOGGER.warning(f"[WATCHDOG] Failed to get price for {symbol}: {e}")
            
            return 0.0
        
        # Async wrapper for the price function (proper async now)
        async def get_price_async(symbol: str) -> float:
            try:
                return await get_current_price(symbol)
            except Exception as e:
                LOGGER.warning(f"[WATCHDOG] Price lookup error for {symbol}: {e}")
                return 0.0
        
        # PARALLEL PRICE FETCHING: Fetch all prices concurrently
        # FIX (Feb 24, 2026): get_status() returns integer counts, NOT pick lists.
        # Use get_tracked_symbols() to get the actual symbol list for price prefetch.
        active_symbols = notif.get_tracked_symbols()
        
        if active_symbols:
            LOGGER.info(f"[WATCHDOG] 💰 Fetching prices for {len(active_symbols)} symbols in parallel...")
            # Fetch all prices concurrently (much faster than sequential)
            price_tasks = [get_price_async(sym) for sym in active_symbols]
            prices = await asyncio.gather(*price_tasks, return_exceptions=True)
            
            # Build price lookup dict
            price_lookup = {}
            for sym, price in zip(active_symbols, prices):
                if isinstance(price, (int, float)) and price > 0:
                    price_lookup[sym] = price
                else:
                    LOGGER.warning(f"[WATCHDOG] No valid price for {sym}")
            
            # Create sync wrapper that uses the pre-fetched prices
            def get_price_sync(symbol: str) -> float:
                return price_lookup.get(symbol, 0.0)
        else:
            LOGGER.info("[WATCHDOG] No active picks to check")
            def get_price_sync(symbol: str) -> float:
                return 0.0
        
        # Run the check with pre-fetched prices (fast now!)
        had_updates = notif.check_for_updates(get_price_sync)
        
        # 🎯 ADVISOR CHECK: Check all advisor-tracked positions too
        advisor_alerts = 0
        try:
            from core.ghost_advisor import get_advisor, format_advisor_alert
            advisor = get_advisor()
            open_positions = advisor.get_open_positions()
            
            for pos in open_positions:
                new_price = price_lookup.get(pos.symbol, 0) or await get_price_async(pos.symbol)
                if new_price > 0:
                    result = advisor.update_price(pos.symbol, new_price)
                    if result:
                        alert_type, updated_pos = result
                        message = format_advisor_alert(alert_type, updated_pos)
                        _send_telegram(message)
                        advisor_alerts += 1
                        LOGGER.info(f"[ADVISOR] 📤 Sent {alert_type.value} alert for {pos.symbol}")
            
            if advisor_alerts > 0:
                LOGGER.info(f"[ADVISOR] 🎯 Sent {advisor_alerts} advisor alerts")
        except Exception as advisor_err:
            LOGGER.error(f"[ADVISOR] Check error: {advisor_err}")
        
        # Get final status
        status = notif.get_status()
        
        LOGGER.info(
            f"[WATCHDOG] ✅ Background check complete - "
            f"Sent: {had_updates}, Active: {status.get('active_picks', 0)}, "
            f"Targets: {status.get('target_hits', 0)}, Stops: {status.get('stop_hits', 0)}"
        )
        
    except Exception as e:
        LOGGER.error(f"[WATCHDOG] Background check error: {e}", exc_info=True)


def get_learning_adjustments():
    """
    Get learning-based adjustments for TOP 10 selection.
    
    Returns:
        {
            "excluded_symbols": [symbols to exclude from TOP 10],
            "boosted_symbols": [symbols to boost confidence],
            "penalized_symbols": [symbols to reduce confidence]
        }
    """
    try:
        from core.db_pool import get_sync_connection
        
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            return {"excluded_symbols": [], "boosted_symbols": [], "penalized_symbols": []}
        
        with get_sync_connection() as conn:
            cursor = conn.cursor()
            
            # Ensure table exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ghost_symbol_accuracy (
                    symbol TEXT PRIMARY KEY,
                    total_predictions INTEGER DEFAULT 0,
                    correct_predictions INTEGER DEFAULT 0,
                    accuracy_pct NUMERIC(5,2) DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT NOW(),
                    status TEXT DEFAULT 'active'
                )
            """)
        
            # Excluded: < 30% accuracy with 10+ predictions (really bad)
            cursor.execute("""
                SELECT symbol FROM ghost_symbol_accuracy
                WHERE accuracy_pct < 30 AND total_predictions >= 10
            """)
            excluded = [row[0] for row in cursor.fetchall()]
            
            # Boosted: > 70% accuracy with 10+ predictions
            cursor.execute("""
                SELECT symbol FROM ghost_symbol_accuracy
                WHERE accuracy_pct > 70 AND total_predictions >= 10
            """)
            boosted = [row[0] for row in cursor.fetchall()]
            
            # Penalized: 30-45% accuracy with 10+ predictions
            cursor.execute("""
                SELECT symbol FROM ghost_symbol_accuracy
                WHERE accuracy_pct >= 30 AND accuracy_pct < 45 AND total_predictions >= 10
            """)
            penalized = [row[0] for row in cursor.fetchall()]
            
            return {
                "excluded_symbols": excluded,
                "boosted_symbols": boosted,
                "penalized_symbols": penalized
        }
        
    except Exception as e:
        LOGGER.warning(f"[LEARNING] Could not get adjustments: {e}")
        return {"excluded_symbols": [], "boosted_symbols": [], "penalized_symbols": []}


def _execute_tool(tool_name: str, arguments: dict) -> str:
    """Execute a Ghost tool function and return JSON result.

    Available tools:
    - get_current_datetime: Current date/time and market status
    - get_ghost_health: System health check
    - get_live_stock_price: Real-time stock price with intraday data
    - get_latest_news: Latest news headlines for a symbol
    - get_ghost_capabilities: List of Ghost features and commands
    """
    try:
        if tool_name == "get_current_datetime":
            from datetime import datetime

            import pytz

            # User is in Central Time (CT)
            tz = pytz.timezone("America/Chicago")
            now = datetime.now(tz)
            is_trading, _ = _is_market_open_now()

            return json.dumps(
                {
                    "date": now.strftime("%A, %B %d, %Y"),
                    "time": now.strftime("%I:%M %p %Z"),  # 12-hour format without seconds
                    "timestamp": int(now.timestamp()),
                    "is_trading_hours": is_trading,
                    "day_of_week": now.strftime("%A"),
                    "timezone": "America/Chicago",
                }
            )

        elif tool_name == "get_ghost_health":
            health = {
                "overall": "healthy",
                "price_providers": {},
                "database": "connected" if os.path.exists(WOLF_SQLITE_PATH) else "missing",
                "cache": "active",
                "broker": "disabled",
            }

            # Check Polygon intraday
            try:
                intraday = _fetch_polygon_intraday("WOLF")
                health["price_providers"]["polygon_intraday"] = "OK" if intraday else "NO_DATA"
            except Exception as e:
                health["price_providers"]["polygon_intraday"] = f"ERROR: {str(e)[:50]}"

            # Check broker
            try:
                from core.alpaca_broker import get_broker

                broker = get_broker()
                if broker.enabled:
                    acc = broker.get_account()
                    buying_power = float(acc.get("buying_power", 0))
                    health["broker"] = f"alpaca ${buying_power:,.0f} buying power"
                else:
                    health["broker"] = "disabled (set BROKER=alpaca)"
            except Exception:
                health["broker"] = "not configured"

            # Check AGENTS_ENABLED
            health["ai_enabled"] = bool(AGENTS_ENABLED)
            health["ai_provider"] = AI_PROVIDER
            health["ai_model"] = AGENT_MODEL

            return json.dumps(health, indent=2)

        elif tool_name == "get_live_stock_price":
            symbol = arguments.get("symbol", "WOLF").upper()

            # Try Polygon intraday bars first
            try:
                intraday = _fetch_polygon_intraday(symbol)
                if intraday:
                    return json.dumps(
                        {
                            "symbol": symbol,
                            "price": intraday["price"],
                            "high": intraday["high"],
                            "low": intraday["low"],
                            "volume": intraday["volume"],
                            "vwap": intraday.get("vwap", 0),
                            "timestamp": intraday["timestamp"],
                            "provider": "polygon_intraday",
                            "delay": "5 minutes",
                        },
                        indent=2,
                    )
            except Exception as e:
                LOGGER.warning(f"Polygon intraday failed for {symbol}: {e}")

            # Fallback to standard providers
            if symbol == "WOLF":
                price, prev, provider = get_wolf_price()
                return json.dumps(
                    {
                        "symbol": symbol,
                        "price": price,
                        "prev_close": prev,
                        "provider": provider or "unavailable",
                        "note": "End-of-day data (Polygon intraday failed)",
                    },
                    indent=2,
                )

            return json.dumps(
                {"error": f"Only WOLF supported currently. {symbol} requires additional config."}
            )

        elif tool_name == "get_latest_news":
            symbol = arguments.get("symbol", "WOLF").upper()
            limit = arguments.get("limit", 5)

            if symbol == "WOLF":
                news = get_wolf_news(limit=limit)
                headlines = [
                    {
                        "headline": item.get("headline"),
                        "sentiment": item.get("sent"),
                        "timestamp": item.get("ts"),
                        "url": item.get("url"),
                    }
                    for item in news.get("items", [])[:limit]
                ]

                return json.dumps(
                    {"symbol": symbol, "count": len(headlines), "news": headlines}, indent=2
                )

            return json.dumps(
                {"error": f"News for {symbol} not configured yet. Only WOLF supported."}
            )

        elif tool_name == "get_ghost_capabilities":
            return json.dumps(
                {
                    "features": [
                        "Real-time stock price tracking (Polygon intraday, 5-min delay)",
                        "AI-powered trading signals with FinBERT sentiment",
                        "Portfolio management (positions, P&L, NAV)",
                        "Telegram bot with trading commands",
                        "Alpaca broker integration (paper trading)",
                        "Automated SL/TP (-3% stop loss, +6% take profit)",
                        "Prometheus metrics export",
                        "Prediction overlay with MAP accuracy",
                    ],
                    "telegram_commands": [
                        "/status - Portfolio status",
                        "/signal - Current trading signal",
                        "/pnl - Daily P&L",
                        "/positions - Show open positions",
                        "/buy SYMBOL QTY - Buy stocks",
                        "/sell SYMBOL - Sell position",
                        "/help - Show all commands",
                    ],
                    "api_endpoints": [
                        "GET /health - System health",
                        "GET /ready - Readiness probe",
                        "GET /metrics - Prometheus metrics",
                        "GET /api/price/WOLF - Get WOLF price",
                        "POST /api/trade/submit - Submit order",
                        "GET /api/broker/positions - List positions",
                    ],
                    "status": "All systems operational",
                },
                indent=2,
            )

        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    except Exception as e:
        LOGGER.error(f"Tool execution error ({tool_name}): {e}", exc_info=True)
        return json.dumps({"error": f"Tool execution failed: {str(e)[:100]}"})


def _ask_ghost_ai(question: str) -> str:
    """Answer natural language questions using Ghost AI with market context.

    ENHANCED: Now uses ChatGPT function calling to access real-time data:
    - Current date/time
    - Ghost health status
    - Live stock/crypto prices (via Polygon intraday)
    - Latest news headlines
    - System capabilities

    Args:
        question: User's question (e.g., "What day is it?", "What's WOLF price?", "Are you healthy?")

    Returns:
        AI-generated answer with reasoning and real-time data
    """
    if not AGENTS_ENABLED:
        return "🤖 AI agent not enabled. Set AGENTS_ENABLED=1 and configure AI_PROVIDER."

    if AI_PROVIDER == "openai" and not OPENAI_API_KEY:
        return "❌ OpenAI API key not set. Please set OPENAI_API_KEY in your environment."

    # Define tools Ghost can use via function calling
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_datetime",
                "description": "Get the current date and time in America/New_York timezone",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_ghost_health",
                "description": "Get Ghost system health status (providers, database, cache, broker)",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_live_stock_price",
                "description": "Get real-time stock price with intraday high/low/volume from Polygon",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol (e.g., WOLF, AAPL, NVDA)",
                        }
                    },
                    "required": ["symbol"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_latest_news",
                "description": "Get latest news headlines for a stock symbol",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol (e.g., WOLF)",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of headlines to return (default 5)",
                            "default": 5,
                        },
                    },
                    "required": ["symbol"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_ghost_capabilities",
                "description": "Get list of Ghost's capabilities and features",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]

    # Helper: classify meta vs market queries
    def _is_meta(q: str) -> bool:
        ql = (q or "").strip().lower()
        meta_keys = (
            "what day is it",
            "what time",
            "time is it",
            "date is it",
            "what's the time",
            "what's the time",
            "current time",
            "your health",
            "health check",
            "healthcheck",
            "health status",
            "system health",
            "self health",
            "ghost health",
            "diagnostic",
            "status check",
            "self check",
            "system status",
            "are you alive",
            "are you up",
            "are you ok",
            "capabilities",
            "what can you do",
            "agentkit",
            "openai agentkit",
            "provider",
            "model",
            "connected to",
            "are you connected",
        )
        return any(k in ql for k in meta_keys)

    try:
        # Build context and persist prior exchanges in memory ring
        ctx = _build_ai_context()

        _now = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())
        is_meta = _is_meta(question)

        # SHORT-CIRCUIT: For meta queries, return tool results directly without LLM
        if is_meta:
            ql = question.lower()
            lines = []

            # Time query
            if any(k in ql for k in ("time", "date", "day")):
                try:
                    time_data = json.loads(_execute_tool("get_current_datetime", {}))
                    lines.append(f"🕒 {time_data['time']} on {time_data['date']}")
                except Exception:
                    lines.append(f"Time: {_now}")

            # Health query
            if any(k in ql for k in ("health", "diagnostic", "status")):
                try:
                    health_data = json.loads(_execute_tool("get_ghost_health", {}))
                    status = health_data.get("overall", "unknown")
                    ai_status = "enabled" if health_data.get("ai_enabled") else "disabled"
                    lines.append(f"💚 Health: {status} | AI: {ai_status}")
                except Exception:
                    lines.append("Health: OK")

            # Capabilities query
            if any(k in ql for k in ("capabilities", "what can you do", "features")):
                try:
                    caps_data = json.loads(_execute_tool("get_ghost_capabilities", {}))
                    features = caps_data.get("features", [])
                    lines.append(f"🎯 Capabilities: {', '.join(features[:5])}")
                except Exception:
                    lines.append("Capabilities: Trading, signals, alerts, portfolio tracking")

            # If we have lines, return immediately without LLM call
            if lines:
                return "\n".join(lines)
            # Fallback if no match
            return "🤖 Use /help for available commands"

        # MARKET queries continue with full LLM flow
        base_system = (
            "You are Ghost, an AI assistant with market data access. "
            "Answer questions directly and accurately. For general questions (crypto, news, etc.), "
            "provide factual information without forcing WOLF stock context."
        )

        # Check if question is specifically about WOLF or trading
        ql = question.lower()
        is_wolf_question = any(
            word in ql
            for word in [
                "wolf",
                "wolfspeed",
                "stock",
                "position",
                "portfolio",
                "trade",
                "buy",
                "sell",
            ]
        )

        if is_wolf_question:
            # Include WOLF-specific context for trading questions
            market_guidance = (
                "For WOLF trading questions, include: price, range, volume, news sentiment, macro pressure, "
                "and 2-3 action bullets with conditions. Do not add timestamps unless explicitly asked."
            )
            system_prompt = base_system + " " + market_guidance

            user_prompt = (
                f"Question: {question}\nNow: {_now}\nSymbol: {WOLF}\n"
                + f"Hints: fusion_score={(ctx.get('fusion') or {}).get('ghost_score')}, news={(ctx.get('news_signal') or {}).get('score')}, macro={(ctx.get('macro_pressure') or {}).get('pressure')}\n"
                + "Call tools to get current time, health, live price, or headlines as needed."
            )
        else:
            # General question - route to specialized handlers
            # Check if it's a crypto question
            is_crypto_question = any(
                word in ql
                for word in [
                    "crypto",
                    "bitcoin",
                    "btc",
                    "ethereum",
                    "eth",
                    "pepe",
                    "doge",
                    "shib",
                    "cryptocurrency",
                    "coin",
                    "altcoin",
                    "blockchain",
                    "defi",
                    "should i buy",
                    "investment",
                    "profit",
                    "prediction",
                    "30 days",
                    "30 day",
                    "best crypto",
                ]
            )

            if is_crypto_question and os.getenv("CRYPTO_ENABLED", "0") == "1":
                # Route to REAL crypto intelligence endpoint
                try:
                    LOGGER.info(f"🔀 Routing crypto question to AI advisor: {question}")

                    # Import crypto intelligence
                    from core.ai_advisor.accuracy_tracker import get_tracker
                    from core.ai_advisor.scanner import get_scanner
                    from wolf_app import _get_crypto_engine

                    _get_crypto_engine()
                    tracker = get_tracker()
                    scanner = get_scanner()

                    # Get Ghost's real stats
                    ghost_stats = tracker.get_stats()

                    # Scan markets
                    opportunities = scanner.get_latest_opportunities(limit=10)

                    # Build intelligent system prompt
                    crypto_system = f"""You are Ghost, an expert AI crypto advisor with REAL market analysis.

YOUR TRACK RECORD:
- Accuracy: {ghost_stats.get("overall_accuracy_pct", 0):.1f}%
- Win Rate: {ghost_stats.get("win_rate_pct", 0):.1f}%
- Decisions: {ghost_stats.get("total_decisions", 0)}

You have access to:
- Real prediction engine (confidence scores, direction forecasts)
- Live price data from multiple sources
- Market regime detection
- Historical accuracy tracking

NEVER mention timestamps. NEVER say "Time: ...". NEVER say "America".

Answer crypto questions with:
1. Real data and predictions
2. Specific confidence scores
3. Profit/loss calculations
4. Risk warnings
5. Honest recommendations (not hype)

If asked "what crypto are you working on", say:
"I'm currently analyzing: BTC, ETH, SOL, PEPE, DOGE, SHIB, and 10+ other cryptos.
I run predictions every 30 seconds and track accuracy. What would you like to know?"
"""

                    crypto_user = f"""Question: {question}

Market Context:
- Current opportunities: {len(opportunities)} assets analyzed
- Top picks available
- Real-time predictions active

Answer the question using your real intelligence. Be specific and data-driven."""

                    # Call AI with crypto context
                    messages = [
                        {"role": "system", "content": crypto_system},
                        {"role": "user", "content": crypto_user},
                    ]

                    payload = {
                        "model": AGENT_MODEL,
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 800,
                    }
                    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

                    r = _http_post(
                        f"{OPENAI_BASE_URL}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=30,
                    )

                    if r.status_code == 200:
                        data = r.json()
                        content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
                        if content:
                            # NEVER add timestamp prefix for crypto questions
                            return content

                    # Fallback if API fails
                    return "🤖 Crypto module active. Ask me about specific cryptos or investments!"

                except Exception as e:
                    LOGGER.error(f"Crypto routing failed: {e}", exc_info=True)
                    # Continue to generic fallback below

            # Generic fallback for non-crypto questions
            system_prompt = base_system
            user_prompt = (
                f"Question: {question}\nNow: {_now}\n"
                + "Answer the question accurately. Use tools only if specifically needed (time, health checks). "
                + "NEVER start your response with 'Time:' unless explicitly asked about time."
            )

        # Call AI provider
        if AI_PROVIDER == "ollama":
            payload = {
                "model": AGENT_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            }
            r = _http_post(
                f"{OLLAMA_BASE_URL}/chat/completions",
                json=payload,
                timeout=AI_TIMEOUT_S,
            )
            data = r.json() if r.status_code == 200 else {}
            content = (
                (((data.get("choices") or [{}])[0].get("message") or {}).get("content"))
                if data
                else None
            )
            # REMOVED: No longer forcing timestamp on every response
            # Only add time if question specifically asks for it
            if content and any(
                word in question.lower() for word in ["time", "date", "when", "what day"]
            ):
                ts_line = time.strftime("%Y-%m-%d %I:%M %p %Z", time.localtime())
                if not str(content).lstrip().lower().startswith("time:"):
                    content = f"🕒 Current time: {ts_line}\n\n" + str(content)
            return content or "❌ AI response empty"
        else:
            # OpenAI-compatible with function calling
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # First API call with tools
            payload = {
                "model": AGENT_MODEL,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",  # Let ChatGPT decide if it needs tools
            }
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}

            r = _http_post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=AI_TIMEOUT_S,
            )
            r.raise_for_status()
            data = r.json() or {}
            response_message = (data.get("choices") or [{}])[0].get("message") or {}

            # Check if ChatGPT wants to call any tools
            tool_calls = response_message.get("tool_calls")
            if tool_calls:
                # Execute each tool and collect results
                messages.append(response_message)  # Add assistant's response with tool_calls

                for tool_call in tool_calls:
                    function_name = tool_call.get("function", {}).get("name")
                    function_args_str = tool_call.get("function", {}).get("arguments", "{}")
                    tool_call_id = tool_call.get("id")

                    LOGGER.info(f"Tool execution: {function_name}({function_args_str})")

                    try:
                        function_args = json.loads(function_args_str)
                    except json.JSONDecodeError:
                        function_args = {}

                    # Execute the tool
                    tool_result = _execute_tool(function_name, function_args)

                    # Add tool response to messages
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": function_name,
                            "content": tool_result,
                        }
                    )

                # Second API call with tool results
                payload = {
                    "model": AGENT_MODEL,
                    "messages": messages,
                }
                r = _http_post(
                    f"{OPENAI_BASE_URL}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=AI_TIMEOUT_S,
                )
                r.raise_for_status()
                data = r.json() or {}
                content = (
                    (((data.get("choices") or [{}])[0].get("message") or {}).get("content"))
                    if data
                    else None
                )
            else:
                # No tools needed, use direct response
                content = response_message.get("content")

            # REMOVED: No longer forcing timestamp on every response
            # Only add time if question specifically asks for it
            if content and any(
                word in question.lower() for word in ["time", "date", "when", "what day"]
            ):
                ts_line = time.strftime("%Y-%m-%d %I:%M %p %Z", time.localtime())
                if not str(content).lstrip().lower().startswith("time:"):
                    content = f"🕒 Current time: {ts_line}\n\n" + str(content)
            # Persist Q&A to AI memory ring (lightweight)
            try:
                _ai_memory_append(
                    {
                        "ts": int(time.time()),
                        "price": (ctx.get("prices") or {}).get("price"),
                        "prev": (ctx.get("prices") or {}).get("prev_close"),
                        "qty": float((ctx.get("position") or {}).get("qty") or 0.0),
                        "avg": float((ctx.get("position") or {}).get("avg_cost") or 0.0),
                        "news_score": (ctx.get("news_signal") or {}).get("score") or 0.0,
                        "features": {"fusion": (ctx.get("fusion") or {}).get("ghost_score")},
                        "label_next_move": "CHAT",
                        "advisory": f"Q: {question}\nA: {str(content)[:512]}",
                        "confidence": int(((ctx.get("fusion") or {}).get("confidence") or 0) * 100),
                    }
                )
            except Exception:
                pass

            return content or "❌ AI response empty"
    except Exception as e:
        LOGGER.error(f"AI chat error: {e}", exc_info=True)
        return f"❌ AI error: {str(e)[:100]}"


def _build_ai_context() -> dict[str, Any]:
    price, prev, provider = get_wolf_price()
    snap = {
        "as_o": int(time.time()),
        "symbol": WOLF,
        "prices": {"price": price, "prev_close": prev, "provider": provider},
        "position": {
            "qty": float(STATE.get("qty", 0.0)),
            "avg_cost": float(STATE.get("avg_cost", 0.0)),
        },
    }
    news = get_wolf_news(limit=10)
    snap["news_signal"] = news.get("news_signal") or {
        "score": None,
        "engine": "none",
        "items_scored": 0,
    }
    snap["news"] = [
        {
            "ts": it.get("ts"),
            "headline": it.get("headline"),
            "url": it.get("url"),
            "sent": it.get("sent"),
        }
        for it in news.get("items", [])
    ]
    sig = _evaluate_signal()
    snap["signal"] = {
        k: sig.get(k) for k in ("action", "mode", "final_score", "thresholds")
    }  # compact

    # Stage 1: Add enhanced world context and market mood
    if STAGE1_ENABLED:
        try:
            enhanced = get_enhanced_context(hours=24, min_relevance=0.3)
            snap["world_context"] = enhanced.get("world_context", _get_world_context_fallback())
            snap["market_mood"] = enhanced.get("market_mood", _get_market_mood_fallback())
        except Exception as e:
            LOGGER.warning("stage1_context_failed", extra={"error": str(e)})
            snap["world_context"] = _get_world_context_fallback()
            snap["market_mood"] = _get_market_mood_fallback()
    else:
        snap["world_context"] = _get_world_context_fallback()
        snap["market_mood"] = _get_market_mood_fallback()

    # Compute fused GHOST score (price momentum + news + macro + AI signal)
    try:

        def _compute_fusion_score(ctx: dict[str, Any]) -> dict[str, Any]:
            # Components
            p_now = (ctx.get("prices") or {}).get("price")
            p_prev = (ctx.get("prices") or {}).get("prev_close")
            price_momentum = 0.0
            if isinstance(p_now, (int, float)) and isinstance(p_prev, (int, float)) and p_prev:
                price_momentum = (float(p_now) - float(p_prev)) / float(p_prev)

            news_score = (ctx.get("news_signal") or {}).get("score")
            if not isinstance(news_score, (int, float)):
                news_score = 0.0

            macro_trend = (ctx.get("macro_pressure") or {}).get("pressure")
            if not isinstance(macro_trend, (int, float)):
                macro_trend = 0.0

            # Use internal signal final_score as AI prediction proxy (range ~ -1..+1)
            ai_pred = (ctx.get("signal") or {}).get("final_score")
            if not isinstance(ai_pred, (int, float)):
                ai_pred = 0.0

            # Weights (env-tunable)
            w_price = float(os.getenv("FUSE_W_PRICE", "0.4"))
            w_news = float(os.getenv("FUSE_W_NEWS", "0.2"))
            w_macro = float(os.getenv("FUSE_W_MACRO", "0.2"))
            w_ai = float(os.getenv("FUSE_W_AI", "0.2"))

            # Normalize components roughly to -1..+1 domain
            comp_price = max(-1.0, min(1.0, price_momentum))
            comp_news = max(-1.0, min(1.0, float(news_score)))
            comp_macro = max(-1.0, min(1.0, float(macro_trend)))
            comp_ai = max(-1.0, min(1.0, float(ai_pred)))

            score = (
                (w_price * comp_price)
                + (w_news * comp_news)
                + (w_macro * comp_macro)
                + (w_ai * comp_ai)
            )
            score = max(-1.0, min(1.0, score))

            # Confidence heuristic: dispersion and magnitude
            # Higher magnitude and agreement between components => higher confidence
            comps = [comp_price, comp_news, comp_macro, comp_ai]
            agreement = 1.0 - (sum(abs(c - score) for c in comps) / (len(comps) * 2.0))
            confidence = max(0.0, min(1.0, 0.5 * abs(score) + 0.5 * agreement))

            return {
                "ghost_score": round(score, 4),
                "confidence": round(confidence, 3),
                "components": {
                    "price_momentum": round(comp_price, 4),
                    "news_sentiment": round(comp_news, 4),
                    "macro_trend": round(comp_macro, 4),
                    "ai_prediction": round(comp_ai, 4),
                },
                "weights": {
                    "price": w_price,
                    "news": w_news,
                    "macro": w_macro,
                    "ai": w_ai,
                },
            }

        snap["fusion"] = _compute_fusion_score(snap)
    except Exception as _ferr:
        LOGGER.debug("fusion_score_failed", extra={"err": str(_ferr)})
        snap["fusion"] = {
            "ghost_score": None,
            "confidence": None,
            "components": {},
            "weights": {},
        }

    # Attach intelligence signals if available
    try:
        conn = sqlite3.connect(WOLF_SQLITE_PATH)
        cur = conn.cursor()
        r = cur.execute(
            "SELECT ts,pressure,components_json FROM macro_pressure ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if r:
            snap["macro_pressure"] = {"ts": int(r[0]), "pressure": float(r[1])}
    except Exception:
        snap["macro_pressure"] = {"pressure": None}
    try:
        conn = sqlite3.connect(WOLF_SQLITE_PATH)
        cur = conn.cursor()
        rows = cur.execute("SELECT name,weight FROM module_weights").fetchall()
        conn.close()
        weights = {n: float(w) for (n, w) in rows}
        snap["module_weights"] = weights
    except Exception:
        snap["module_weights"] = {}
    return snap


def _llm_decide(ctx: dict[str, Any]) -> AiDecision:
    if not AGENTS_ENABLED:
        # Fallback: derive confidence/rationale deterministically
        action = str((ctx.get("signal") or {}).get("action") or "HOLD")
        fscore = (ctx.get("signal") or {}).get("final_score")
        # Base confidence from model score
        base_conf = (
            int(round(abs(float(fscore)) * 100))
            if isinstance(fscore, (int, float))
            else (70 if action != "HOLD" else 50)
        )
        # Blend in fusion confidence if available
        try:
            fusion_conf = (ctx.get("fusion") or {}).get("confidence")
            if isinstance(fusion_conf, (int, float)):
                alpha = float(os.getenv("AI_BLEND_ALPHA", "0.7"))
                base_conf = int(round(alpha * base_conf + (1 - alpha) * (fusion_conf * 100)))
        except Exception:
            pass
        conf = base_conf
        news_score = (ctx.get("news_signal") or {}).get("score")
        rationale = f"Price-mode={(ctx.get('signal') or {}).get('mode')}, news={'n/a' if news_score is None else f'{news_score:+.2f}'}"
        return AiDecision(
            action=action,
            confidence=max(0, min(100, conf)),
            rationale=rationale,
            risks=[],
            evidence=[],
            checklist=[],
        )
    try:
        # Optional rubric to steer the model's behavior
        rubric = os.getenv("AI_DECISION_RUBRIC", "").strip()
        system_base = (
            "You are Ghost, a WOLF-only advisory AI. Output JSON with keys: "
            "action, confidence (0-100), rationale, risks (list), evidence (urls), checklist (list)."
        )
        system_text = f"{system_base}\nRubric: {rubric}" if rubric else system_base
        import re

        if AI_PROVIDER == "ollama":
            payload = {
                "model": AGENT_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": system_text,
                    },
                    {"role": "user", "content": json.dumps(ctx, separators=(",", ":"))},
                ],
                "stream": False,
                "format": "json",
            }
            r = _http_post(
                f"{OLLAMA_BASE_URL}/chat/completions",
                json=payload,
                timeout=AI_TIMEOUT_S,
            )
            data = r.json() if r.status_code == 200 else {}
            content = (
                (((data.get("choices") or [{}])[0].get("message") or {}).get("content"))
                if data
                else None
            )
            try:
                obj = json.loads(content) if content else {}
            except Exception:
                m = re.search(r"\{[\s\S]*\}", content or "")
                obj = json.loads(m.group(0)) if m else {}
        else:
            # OpenAI-compatible with light retries
            payload = {
                "model": AGENT_MODEL,
                "messages": [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": json.dumps(ctx, separators=(",", ":"))},
                ],
                "response_format": {"type": "json_object"},
            }
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
            data = {}
            for attempt in range(1, 4):
                try:
                    r = _http_post(
                        f"{OPENAI_BASE_URL}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=AI_TIMEOUT_S,
                    )
                    if r.status_code in (429, 500, 502, 503, 504):
                        RuntimeError(f"upstream {r.status_code}")
                        time.sleep(min(2.0, 0.2 * (2 ** (attempt - 1))))
                        continue
                    r.raise_for_status()
                    data = r.json() or {}
                    break
                except Exception:
                    time.sleep(min(2.0, 0.2 * (2 ** (attempt - 1))))
            content = (
                (((data.get("choices") or [{}])[0].get("message") or {}).get("content"))
                if data
                else None
            )
            try:
                obj = json.loads(content) if content else {}
            except Exception:
                m = re.search(r"\{[\s\S]*\}", content or "")
                obj = json.loads(m.group(0)) if m else {}
        action = str(obj.get("action") or "HOLD").upper()
        conf = int(obj.get("confidence") or 50)
        rationale = str(obj.get("rationale") or "")
        risks = obj.get("risks") or []
        evidence = obj.get("evidence") or []
        checklist = obj.get("checklist") or []
        # Normalize action
        allowed = {"BUY", "SELL", "HOLD"}
        if action not in allowed:
            try:
                ghost_score = (ctx.get("fusion") or {}).get("ghost_score")
                if isinstance(ghost_score, (int, float)):
                    action = (
                        "BUY" if ghost_score > 0.1 else ("SELL" if ghost_score < -0.1 else "HOLD")
                    )
                else:
                    action = "HOLD"
            except Exception:
                action = "HOLD"
        # Blend LLM confidence with fusion confidence if enabled
        try:
            if os.getenv("AI_BLEND_FUSION", "1").lower() in ("1", "true", "yes"):
                fusion_c = (ctx.get("fusion") or {}).get("confidence")
                if isinstance(fusion_c, (int, float)):
                    alpha = float(os.getenv("AI_BLEND_ALPHA", "0.7"))  # FIX (Feb 24): was "0.8", standardized to 0.7
                    conf = int(round(alpha * conf + (1 - alpha) * (fusion_c * 100)))
        except Exception:
            pass
        # Respect kill switch
        try:
            if os.getenv("AI_RESPECT_KILL", "1").lower() in ("1", "true", "yes") and os.getenv(
                "RISK_KILL", "0"
            ).lower() in ("1", "true", "yes"):
                if action == "BUY":
                    action = "HOLD"
                    rationale = (rationale + " | Kill-switch active: suppressing BUY").strip()
        except Exception:
            pass
        # Enrich evidence with news URLs if empty
        try:
            if not evidence:
                news_urls = [
                    n.get("url")
                    for n in (ctx.get("news") or [])
                    if isinstance(n, dict) and n.get("url")
                ]  # type: ignore
                evidence = (news_urls or [])[:2]
        except Exception:
            pass
        return AiDecision(
            action=action,
            confidence=max(0, min(100, conf)),
            rationale=rationale,
            risks=risks,
            evidence=evidence,
            checklist=checklist,
        )
    except Exception:
        # On any failure, fallback
        action = str((ctx.get("signal") or {}).get("action") or "HOLD")
        return AiDecision(
            action=action,
            confidence=50,
            rationale="AI unavailable; fallback to rule-based",
            risks=[],
            evidence=[],
            checklist=[],
        )


def _evaluate_signal(symbol: str = WOLF) -> dict[str, Any]:
    # Get price for the specified symbol
    if symbol == WOLF:
        price, prev, provider = get_wolf_price()
    else:
        # Use price quorum for other symbols
        try:
            is_market_open, _ = _is_market_open_now()
        except Exception:
            is_market_open = False

        providers = _build_price_providers(symbol, is_market_open=is_market_open)
        if providers:
            decision = get_price_quorum().get_price(
                symbol=symbol,
                providers=providers,
                prev_close=None,
                is_market_open=is_market_open,
                timeout=6.0,
            )
            price = decision.price
            prev = decision.prev_close
            provider = decision.provider_label
        else:
            price = None
            provider = "unavailable"

    qty, avg = _get_portfolio_qty_and_avg()  # Use helper to read from positions array
    action = "HOLD"
    used_mode = ALERT_MODE
    thresholds: dict[str, Any] = {}
    if ALERT_STATE.get("hold_override"):
        action = "HOLD"
    else:
        if price is not None and avg > 0:
            # update trailing bounds
            try:
                trailing_high = ALERT_STATE.get("trailing_high")
                if trailing_high is None or price > float(trailing_high):
                    ALERT_STATE["trailing_high"] = price
                trailing_low = ALERT_STATE.get("trailing_low")
                if trailing_low is None or price < float(trailing_low):
                    ALERT_STATE["trailing_low"] = price
            except Exception:
                pass

            if ALERT_MODE == "fixed":
                sell_thr = avg * ALERT_SELL_PCT
                buy_thr = avg * ALERT_BUY_PCT
                thresholds.update({"sell_thr": sell_thr, "buy_thr": buy_thr})
                if price > sell_thr:
                    action = "SELL"
                elif price < buy_thr:
                    action = "BUY"
            elif ALERT_MODE == "band":
                upper = avg * (1.0 + BAND_PCT)
                lower = avg * (1.0 - BAND_PCT)
                thresholds.update({"upper": upper, "lower": lower})
                if price > upper:
                    action = "SELL"
                elif price < lower:
                    action = "BUY"
            elif ALERT_MODE == "trailing":
                th = ALERT_STATE.get("trailing_high")
                tl = ALERT_STATE.get("trailing_low")
                thresholds.update({"trail_high": th, "trail_low": tl})
                if th:
                    if price <= float(th) * (1.0 - TRAIL_SELL_PCT):
                        action = "SELL"
                if tl and action == "HOLD":
                    if price >= float(tl) * (1.0 + TRAIL_BUY_PCT):
                        action = "BUY"
            else:
                used_mode = "fixed"
                sell_thr = avg * ALERT_SELL_PCT
                buy_thr = avg * ALERT_BUY_PCT
                thresholds.update({"sell_thr": sell_thr, "buy_thr": buy_thr})
                if price > sell_thr:
                    action = "SELL"
                elif price < buy_thr:
                    action = "BUY"

            # Volatility gating
            if VOL_GATE and action in ("BUY", "SELL"):
                vol = _get_volatility_lookback()
                thresholds["vol"] = vol
                try:
                    if vol is not None and vol > 0 and avg > 0:
                        dev = abs(price / avg - 1.0)
                        if dev < VOL_K * vol:
                            action = "HOLD"
                except Exception:
                    pass
        else:
            used_mode = ALERT_MODE
    # Optional fused decision override based on final_score
    fused_score = _fuse_price_news_score(action)
    # Update fused score gauge (when available)
    try:
        if _G_FINAL_SCORE is not None and fused_score is not None:
            _G_FINAL_SCORE.set(float(fused_score))
    except Exception:
        pass
    if FUSE_DECISION_ON and fused_score is not None and not ALERT_STATE.get("hold_override"):
        try:
            thresholds.setdefault("fuse", {})
            thresholds["fuse"] = {
                "score": fused_score,
                "t_buy": FUSE_T_BUY,
                "t_sell": FUSE_T_SELL,
            }
            if fused_score >= FUSE_T_BUY:
                action = "BUY"
            elif fused_score <= FUSE_T_SELL:
                action = "SELL"
            else:
                action = "HOLD"
        except Exception:
            pass
    return {
        "action": action,
        "mode": used_mode,
        "price": price,
        "avg": avg,
        "qty": qty,
        "provider": provider,
        "buy_pct": ALERT_BUY_PCT,
        "sell_pct": ALERT_SELL_PCT,
        "thresholds": thresholds,
        # fusion score (optional; None if sentiment off or unavailable)
        "final_score": fused_score,
    }


def _fuse_price_news_score(action: str) -> float | None:
    """Map current price-based action to a coarse signal and blend with news_score.
    Returns None when NEWS_SENTIMENT_ON=0 or no score available.
    """
    if not NEWS_SENTIMENT_ON:
        return None
    try:
        news = get_wolf_news(limit=10) or {}
        sig = news.get("news_signal") or {}
        news_score = sig.get("score")
        if news_score is None:
            return None
        price_signal = 0.0
        if action == "BUY":
            price_signal = 1.0
        elif action == "SELL":
            price_signal = -1.0
        else:
            price_signal = 0.0
        # Macro pressure tilt ([-100,100] scaled to [-1,1])
        macro_term = 0.0
        try:
            conn = sqlite3.connect(WOLF_SQLITE_PATH)
            cur = conn.cursor()
            r = cur.execute(
                "SELECT pressure FROM macro_pressure ORDER BY ts DESC LIMIT 1"
            ).fetchone()
            conn.close()
            if r and r[0] is not None:
                macro_term = float(r[0]) / 100.0
        except Exception:
            macro_term = 0.0
        base = float(
            SENT_ALPHA * price_signal
            + SENT_BETA * float(news_score)
            + FUSE_GAMMA_MACRO * macro_term
        )
        # Optional module weighting adjustment: nudge towards stronger modules (bounded)
        if MODULE_WEIGHTING_ON:
            try:
                conn = sqlite3.connect(WOLF_SQLITE_PATH)
                cur = conn.cursor()
                rows = cur.execute("SELECT name,weight FROM module_weights").fetchall()
                conn.close()
                if rows:
                    w = {n: float(v) for (n, v) in rows}
                    # Combine a simple factor from selected modules
                    adj = 0.0
                    for name, _sign in (
                        ("macro_pressure", 1.0),
                        ("news_sentiment", 1.0),
                        ("price_action", 1.0),
                    ):
                        val = w.get(name)
                        if val is not None:
                            adj += (float(val) - 1.0) * 0.1  # small influence
                    base += adj
            except Exception:
                pass
        return float(max(-1.0, min(1.0, base)))
    except Exception:
        return None


def _signal_card(sig: dict[str, Any], include_trace: bool = True) -> str:
    action = str(sig.get("action") or "HOLD").upper()
    price = sig.get("price")
    rid = _current_trace_id()
    q, avg = _get_portfolio_qty_and_avg()  # Use helper to get both qty and avg from positions array
    current = price if price is not None else avg
    market_value = round(q * current, 2)

    # Adjust P&L for corporate actions (reverse splits, etc.)
    pnl_adjustment = _adjust_pnl_for_corporate_action(WOLF, avg, current, q)
    pnl_abs = round(pnl_adjustment["pnl_abs"], 2)
    pnl_pct = round(pnl_adjustment["pnl_pct"], 6)
    change_pct = None
    try:
        price, prev, _ = get_wolf_price()
        if price is not None and prev and prev > 0:
            change_pct = (price - prev) / prev * 100.0
    except Exception:
        pass
    # Top headlines 2–3
    headlines: list[str] = []
    try:
        news = get_wolf_news(limit=3).get("items", [])
        for it in news[:3]:
            ts = it.get("ts")
            try:
                if isinstance(ts, (int, float)):
                    ts_str = datetime.fromtimestamp(int(ts), tz=UTC).isoformat()
                else:
                    ts_str = str(ts)
            except Exception:
                ts_str = str(ts)
            t = it.get("headline") or ""
            u = it.get("url") or ""
            if u:
                headlines.append(f"{ts_str} — {t} — {u}")
            else:
                headlines.append(f"{ts_str} — {t}")
    except Exception:
        pass
    icon = "⚖️" if action == "HOLD" else ("⚡️" if action in ("BUY", "SELL") else "⚡️")
    hdr = f"{icon} {action} — WOLF (Wolfspeed)\n\n"

    # Build PnL section with adjustment note if applicable
    pnl_section = f"• PnL: {pnl_abs:.2f} ({pnl_pct:.2f}%)"
    if pnl_adjustment.get("has_adjustment") and pnl_adjustment.get("adjustment_note"):
        pnl_section += f"\n• Note: {pnl_adjustment['adjustment_note']}"

    card = (
        hdr
        + "Portfolio\n"
        + f"• Qty: {q:.8f}\n"
        + f"• Avg Cost: ${avg:.2f}\n"
        + f"• Price: {('?' if price is None else f'${price:.2f}')} ({sig.get('provider') or ''})\n"
        + f"• Market Value: ${market_value:.2f}\n"
        + pnl_section
        + "\n\n"
        + "NAV / Cash\n"
        + f"• NAV: ${market_value + float(STATE.get('cash', 0.0)):.2f}\n"
        + f"• Cash: ${float(STATE.get('cash', 0.0)):.2f}\n\n"
        + "Market\n"
        + f"• Change %: {0 if change_pct is None else round(change_pct, 6)}%\n"
        + f"• GPS: {7.2}\n"
        + f"• Signal: {('BUY triggered' if action == 'BUY' else 'SELL triggered' if action == 'SELL' else 'HOLD (no action)')} (mode={sig.get('mode')})\n\n"
        + "News\n"
        + ("\n".join(headlines) if headlines else "No headlines")
    )
    # Append "Why now" top-3 reasons to the Signal card (does not affect STATUS card)
    try:
        reasons_scored: list[tuple[float, str]] = []
        thr = sig.get("thresholds") or {}
        fscore = sig.get("final_score")
        if isinstance(fscore, (int, float)):
            reasons_scored.append((abs(float(fscore)), f"Fusion score {float(fscore):+0.2f}"))
        # Include news sentiment summary if available
        try:
            ns = (get_wolf_news(limit=5) or {}).get("news_signal") or {}
            ns_score = ns.get("score")
            ns_eng = ns.get("engine") or "none"
            ns_n = int(ns.get("items_scored") or 0)
            if isinstance(ns_score, (int, float)) and ns_n > 0:
                reasons_scored.append(
                    (
                        abs(float(ns_score)) * 0.9,
                        f"News sentiment {float(ns_score):+0.2f} ({ns_eng}, n={ns_n})",
                    )
                )
        except Exception:
            pass
        # Price vs thresholds
        try:
            if sig.get("mode") == "fixed" and price is not None:
                bthr = thr.get("buy_thr")
                sthr = thr.get("sell_thr")
                if action == "BUY" and isinstance(bthr, (int, float)):
                    if price < float(bthr):
                        pct = (float(bthr) - float(price)) / float(bthr) if float(bthr) > 0 else 0.0
                        reasons_scored.append(
                            (
                                pct * 1.2,
                                f"Price below buy_thr: ${price:.2f} vs ${float(bthr):.2f} ({pct * 100:.2f}%)",
                            )
                        )
                if action == "SELL" and isinstance(sthr, (int, float)):
                    if price > float(sthr):
                        pct = (float(price) - float(sthr)) / float(sthr) if float(sthr) > 0 else 0.0
                        reasons_scored.append(
                            (
                                pct * 1.2,
                                f"Price above sell_thr: ${price:.2f} vs ${float(sthr):.2f} ({pct * 100:.2f}%)",
                            )
                        )
            elif sig.get("mode") == "band" and price is not None:
                upper = thr.get("upper")
                lower = thr.get("lower")
                if action == "BUY" and isinstance(lower, (int, float)) and price < float(lower):
                    pct = (float(lower) - float(price)) / float(lower) if float(lower) > 0 else 0.0
                    reasons_scored.append(
                        (
                            pct,
                            f"Price below lower band: ${price:.2f} < ${float(lower):.2f} ({pct * 100:.2f}%)",
                        )
                    )
                if action == "SELL" and isinstance(upper, (int, float)) and price > float(upper):
                    pct = (float(price) - float(upper)) / float(upper) if float(upper) > 0 else 0.0
                    reasons_scored.append(
                        (
                            pct,
                            f"Price above upper band: ${price:.2f} > ${float(upper):.2f} ({pct * 100:.2f}%)",
                        )
                    )
            elif sig.get("mode") == "trailing" and price is not None:
                th = thr.get("trail_high")
                tl = thr.get("trail_low")
                if action == "SELL" and isinstance(th, (int, float)) and float(th) > 0:
                    drop = 1.0 - (float(price) / float(th))
                    if drop >= 0:
                        reasons_scored.append(
                            (
                                drop,
                                f"Drop from high: {drop * 100:.2f}% vs trail {TRAIL_SELL_PCT * 100:.2f}%",
                            )
                        )
                if action == "BUY" and isinstance(tl, (int, float)) and float(tl) > 0:
                    rise = (float(price) / float(tl)) - 1.0
                    if rise >= 0:
                        reasons_scored.append(
                            (
                                rise,
                                f"Rise from low: {rise * 100:.2f}% vs trail {TRAIL_BUY_PCT * 100:.2f}%",
                            )
                        )
        except Exception:
            pass
        # Add Stage 1 World Context (if available)
        try:
            if STAGE1_ENABLED:
                from core.stage1_integration import get_enhanced_context

                ctx = get_enhanced_context()
                mood = ctx.get("market_mood", {})
                world = ctx.get("world_context", {})

                if not mood.get("error"):
                    regime = mood.get("market_regime", "unknown").upper()
                    mood_icon = "🐂" if regime == "BULL" else "🐻" if regime == "BEAR" else "↔️"
                    card += (
                        "\n\nMarket Mood\n"
                        f"• Regime: {mood_icon} {regime}\n"
                        f"• Sentiment: {mood.get('sentiment', 'neutral')}\n"
                    )
                    if mood.get("vix_level"):
                        card += f"• VIX: {mood['vix_level']:.1f}\n"

                if not world.get("error"):
                    events = world.get("trending_events", [])[:3]
                    if events:
                        card += "\n🔥 Events: " + ", ".join([f"[{e}]" for e in events])
        except Exception as e:
            logging.debug(f"Stage 1 context unavailable in signal card: {e}")

        # Sort and pick top 3
        reasons_scored.sort(key=lambda x: x[0], reverse=True)
        top = [r for _, r in reasons_scored[:3] if r]
        if top:
            card += "\n\nWhy now\n" + "\n".join([f"• {t}" for t in top])
        # Update reasons count gauge
        try:
            if _G_WHY_NOW_COUNT is not None:
                _G_WHY_NOW_COUNT.set(len(top))
        except Exception:
            pass
    except Exception:
        pass
    if include_trace and rid and rid != "-":
        card += f"\n\nReq: {rid}"
    return card


async def _run_turbo_prediction_for_top10(symbol: str) -> dict:
    """
    Run TURBO prediction for TOP 10 display - bypassing stock_engine AND market gates.
    
    This gives RAW ML predictions without regime filtering, for display purposes.
    The 8 AM scheduled message should show REAL predictions, not gated ones.
    """
    import asyncio
    
    try:
        symbol = symbol.upper().strip()
        is_crypto = symbol in HUNTER_CRYPTO_SYMBOLS or _classify_symbol_category(symbol) == "crypto"
        
        # Get price first
        if is_crypto:
            price_result = turbo_crypto_price(symbol, max_budget_s=3.0)
        else:
            price_result = turbo_stock_price(symbol, max_budget_s=3.0)
        
        if not price_result.get("ok") or not price_result.get("price"):
            return {
                "ok": False,
                "symbol": symbol,
                "direction": "FLAT",
                "confidence": 0,
                "error": f"Price fetch failed: {price_result.get('error', 'unknown')}"
            }
        
        price = float(price_result["price"])
        
        # Extract features using the orchestrator (same as main turbo engine)
        from core.data_pillars.feature_orchestrator import get_feature_orchestrator
        
        orchestrator = get_feature_orchestrator()
        feature_data = orchestrator.get_all_features(symbol, period=90)
        features = feature_data.get("features", {}) or {}
        
        # Determine direction using RSI + MACD (same logic as turbo engine)
        direction = "FLAT"
        rsi = features.get("RSI_14")
        macd_hist = features.get("MACD_HISTOGRAM")
        
        if rsi is not None:
            if rsi > 70:
                direction = "DOWN"  # Overbought
            elif rsi < 30:
                direction = "UP"  # Oversold
        
        if direction == "FLAT" and macd_hist is not None:
            if macd_hist > 0:
                direction = "UP"
            elif macd_hist < 0:
                direction = "DOWN"
        
        # Get ensemble prediction for confidence (same as turbo engine)
        from core.ensemble_predictor import get_ensemble_predictor
        
        ensemble = get_ensemble_predictor()
        try:
            ensemble_pred = ensemble.predict(features, method="confidence_weighted", symbol=symbol)
            ensemble_conf = ensemble_pred.confidence if ensemble_pred and ensemble_pred.confidence else 0.5
            ensemble_dir = ensemble_pred.direction if ensemble_pred else "FLAT"
        except Exception as e:
            LOGGER.warning(f"[TURBO-TOP10] Ensemble failed for {symbol}: {e}")
            ensemble_conf = 0.5
            ensemble_dir = "FLAT"
        
        # Use ensemble direction if moderate confidence
        if ensemble_conf > 0.45:
            direction = ensemble_dir
        
        # Calculate confidence from ensemble (40-85% range, realistic)
        base_confidence = ensemble_conf  # Use the safe variable
        
        # Boost confidence based on signal alignment
        signal_count = 0
        if rsi is not None:
            if (direction == "UP" and rsi < 40) or (direction == "DOWN" and rsi > 60):
                signal_count += 1
        if macd_hist is not None:
            if (direction == "UP" and macd_hist > 0) or (direction == "DOWN" and macd_hist < 0):
                signal_count += 1
        
        # More signals = higher confidence (but cap at 85%)
        confidence = min(0.85, base_confidence + signal_count * 0.05)
        confidence = max(0.40, confidence)  # Floor at 40%
        
        # Calculate expected move based on volatility and signal strength
        volatility = features.get("VOLATILITY_20D", 0.02) or 0.02
        momentum = abs(features.get("MOMENTUM_7D", features.get("MOMENTUM", 0)) or 0)
        rsi_value = features.get("RSI_14") or 50
        
        # Base expected move: 3.5% baseline with confidence adjustments
        # Low confidence (40-55%) → smaller moves (3-4%)
        # Medium confidence (55-70%) → medium moves (4-5.5%)
        # High confidence (70-85%) → larger moves (5.5-7%)
        if confidence < 0.55:
            base_move = 3.0 + (confidence - 0.40) * 6.7  # 40%→3%, 55%→4%
        elif confidence < 0.70:
            base_move = 4.0 + (confidence - 0.55) * 10   # 55%→4%, 70%→5.5%
        else:
            base_move = 5.5 + (confidence - 0.70) * 10   # 70%→5.5%, 85%→7%
        
        # Volatility adjustment: only add for high-vol assets (+0.5% max)
        if volatility and volatility > 0.30:  # >30% annualized vol
            base_move += 0.5
        
        # RSI extremes: small boost (+0.3%)
        if rsi_value and (rsi_value < 25 or rsi_value > 75):
            base_move += 0.3
        
        expected_move = min(7.0, max(3.0, base_move))  # Clamp to 3-7%
        
        # Calculate target and stop
        if direction == "UP":
            target_price = price * (1 + expected_move / 100)
            stop_loss = price * (1 - expected_move / 200)
        elif direction == "DOWN":
            target_price = price * (1 - expected_move / 100)
            stop_loss = price * (1 + expected_move / 200)
        else:
            # FLAT: small upside target (+3% default) since market bias is generally up
            target_price = price * 1.03
            stop_loss = price * 0.97
        
        LOGGER.info(
            f"[TURBO-TOP10] {symbol}: {direction} @ {confidence:.1%}, "
            f"${price:.2f} → ${target_price:.2f} ({expected_move:+.1f}%)"
        )
        
        # ====================================================================
        # INTELLIGENT HOLD PERIOD CALCULATION (1-7 days)
        # Uses confidence + expected move to determine hold
        # Higher confidence + bigger move = longer hold
        # ====================================================================
        
        # Base hold on CONFIDENCE (main driver)
        if confidence >= 0.80:
            hold_days = 7  # Very high confidence = full week
        elif confidence >= 0.70:
            hold_days = 5  # High confidence = 5 days
        elif confidence >= 0.60:
            hold_days = 4  # Good confidence = 4 days
        elif confidence >= 0.50:
            hold_days = 3  # Medium confidence = 3 days
        elif confidence >= 0.40:
            hold_days = 2  # Low confidence = 2 days
        else:
            hold_days = 1  # Very low = 1 day scalp
        
        # Adjust based on expected move size
        if abs(expected_move) > 0.08:  # >8% expected move
            hold_days = min(7, hold_days + 2)  # Add 2 days for big moves
        elif abs(expected_move) > 0.05:  # >5% expected move
            hold_days = min(7, hold_days + 1)  # Add 1 day
        elif abs(expected_move) < 0.03:  # <3% small move
            hold_days = max(1, hold_days - 1)  # Shorter hold for small moves
        
        # RSI extreme = quick reversal expected (cap at 2 days)
        if rsi_value and (rsi_value < 25 or rsi_value > 75):
            hold_days = min(2, hold_days)
        
        hold_reason = f"{hold_days}d_hold"
        
        # ====================================================================
        # NEWS INFLUENCE CHECK
        # Only mark as news-influenced for TOP tier symbols with HIGH confidence
        # Should be RARE - only 2-3 per message max
        # ====================================================================
        news_influenced = False
        news_headline = None
        
        # Only the BIGGEST news-driven symbols
        TOP_NEWS_SYMBOLS = {"NVDA", "TSLA", "BTC", "ETH"}
        
        # News check = top symbol + high confidence + big move
        symbol_upper = symbol.upper()
        if symbol_upper in TOP_NEWS_SYMBOLS:
            # Only if BOTH high confidence AND big expected move
            if confidence > 0.70 and abs(expected_move) > 0.05:  # >70% conf AND >5% move
                news_influenced = True
        
        return {
            "ok": True,
            "symbol": symbol,
            "direction": direction,
            "confidence": confidence,
            "current_price": price,
            "target_price": target_price,
            "stop_loss": stop_loss,
            "hold_days": hold_days,
            "hold_reason": hold_reason,
            "news_influenced": news_influenced,
            "news_headline": news_headline,
            "expected_move_pct": expected_move,
            "volatility": volatility,  # 20-day volatility for entry zone calculation
        }
        
    except Exception as e:
        LOGGER.error(f"[TURBO-TOP10] Prediction failed for {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "ok": False,
            "symbol": symbol,
            "direction": "FLAT",
            "confidence": 0,
            "error": str(e)
        }


def _recent_forecasts_view(symbol: str, n: int = 10) -> list[dict[str, Any]]:
    """Return last N rows with issued ts, pred mid, actual, APE, band hit, model, conf.
    Uses forecast_48h and price_actuals with ±1h tolerance around horizon target.
    """
    import sqlite3

    rows: list[dict[str, Any]] = []
    try:
        conn = sqlite3.connect(WOLF_SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, ts_issued, price_now, price_pred_mid, price_pred_lo, price_pred_hi, horizon_hours, model, confidence
            FROM forecast_48h
            WHERE symbol=?
            ORDER BY ts_issued DESC
            LIMIT ?
            """,
            (symbol, n),
        )
        fcs = cur.fetchall() or []
        for fc in fcs:
            d = dict(fc)
            ts_target = int(d["ts_issued"]) + int(d["horizon_hours"]) * 3600
            cur.execute(
                """
                SELECT price FROM price_actuals
                WHERE symbol=? AND ts BETWEEN ? AND ?
                ORDER BY ABS(ts-?) ASC LIMIT 1
                """,
                (symbol, ts_target - 3600, ts_target + 3600, ts_target),
            )
            a = cur.fetchone()
            actual = float(a["price"]) if a else None
            pred = float(d["price_pred_mid"]) if d.get("price_pred_mid") is not None else None
            ape = None
            hit = None
            if actual is not None and pred and pred > 0:
                ape = abs(actual - pred) / pred * 100.0
                lo = d.get("price_pred_lo")
                hi = d.get("price_pred_hi")
                if lo is not None and hi is not None:
                    hit = lo <= actual <= hi
            rows.append(
                {
                    "issued": int(d["ts_issued"]),
                    "pred": pred,
                    "actual": actual,
                    "ape_pct": ape,
                    "hit": hit,
                    "model": d.get("model"),
                    "conf": d.get("confidence"),
                }
            )
        conn.close()
    except Exception as e:
        LOGGER.warning(f"recent_forecasts_view_failed: {e}")
    return rows


async def _get_news_feed(limit: int = 20):
    """Helper function to fetch news feed."""
    news_items = []
    try:
        import feedparser

        feeds = [
            "https://feeds.reuters.com/reuters/businessNews",
            "https://feeds.marketwatch.com/marketwatch/topstories/",
        ]
        for feed_url in feeds[:2]:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[: limit // 2]:
                    news_items.append(
                        {
                            "title": entry.get("title", ""),
                            "link": entry.get("link", ""),
                            "published": entry.get("published", ""),
                            "summary": entry.get("summary", "")[:200],
                            "source": feed_url.split("/")[2],
                        }
                    )
            except Exception:
                continue
    except Exception:
        pass
    if not news_items:
        news_items = []  # Return empty, not fake news
    return {"news": news_items[-limit:], "count": len(news_items)}


def _build_price_response(payload: dict[str, Any]) -> dict[str, Any]:
    sym = (payload.get("symbol") or "").upper()
    price = payload.get("price")
    prev = payload.get("prev_close")
    change_pct = None
    try:
        if price is not None and prev not in (None, 0):
            prev_val = float(prev)
            if prev_val != 0:
                change_pct = round(((float(price) - prev_val) / prev_val) * 100.0, 4)
    except Exception:
        change_pct = None

    response = {
        "symbol": sym,
        "price": price,
        "prev_close": prev,
        "provider": payload.get("provider"),
        "cached": payload.get("cached"),
        "fresh": payload.get("fresh"),
        "age": payload.get("age"),
        "timestamp": int(time.time()),
        "change_pct": change_pct,
        "change_24h_pct": payload.get("change_24h_pct"),  # Pass through 24h change for crypto
    }

    if sym == WOLF:
        try:
            response["market_open"] = _is_market_open_now()[0]
        except Exception:
            response["market_open"] = None
    else:
        response["market_open"] = None

    return response


def _generate_multi_symbol_predictions() -> dict[str, Any]:
    """
    Internal function to generate multi-symbol predictions.
    Used by both the API endpoint and scheduled Telegram alerts.
    
    NOW USES REAL 6H PREDICTIONS FROM POSTGRESQL (not old 48h forecasts)

    Returns dict with structure:
    {
        "ok": True/False,
        "predictions": {"stocks": [...], "crypto": [...], "vip": [...]},
        "counts": {"stocks": N, "crypto": M, "vip": K},
        "total": X,
        "timestamp": unix_ts,
        "cached": bool (if returned from cache)
    }
    """
    global _LAST_MULTI_PREDICTION_TIME, _LAST_MULTI_PREDICTION_COUNTS, _LAST_MULTI_PREDICTION_RESULT

    # Check cache first to prevent provider exhaustion
    now = time.time()
    if _LAST_MULTI_PREDICTION_RESULT and _LAST_MULTI_PREDICTION_TIME:
        cache_age = now - _LAST_MULTI_PREDICTION_TIME
        if cache_age < _MULTI_PREDICTION_CACHE_TTL:
            # Return cached result with cache indicator
            cached_result = _LAST_MULTI_PREDICTION_RESULT.copy()
            cached_result["cached"] = True
            cached_result["cache_age_seconds"] = cache_age
            return cached_result

    try:
        results = {
            "stocks": [],
            "crypto": [],
            "vip": []
        }
        failed_symbols = {
            "stocks": [],
            "crypto": []
        }

        # Get real 6h predictions from PostgreSQL backend
        from core.prediction_store import PostgresBackend
        backend = PostgresBackend()
        
        # Generate predictions for stock symbols using REAL 6H system
        for symbol in STOCK_SYMBOLS:
            try:
                # Get latest 6h prediction from database
                latest = backend.get_latest_prediction(symbol)
                
                if latest:
                    # Parse features_json to get current price and expected move
                    try:
                        features_json = latest.get("features_json")
                        if features_json:
                            import json
                            features = json.loads(features_json) if isinstance(features_json, str) else features_json
                            price_current = features.get("current_price")
                        else:
                            price_current = None
                    except Exception:
                        price_current = None
                    
                    # Parse params_json for additional data
                    try:
                        params_json = latest.get("params_json")
                        if params_json:
                            params = json.loads(params_json) if isinstance(params_json, str) else params_json
                            expected_move_pct = params.get("expected_move_pct", 2.0)  # Default 2%
                        else:
                            expected_move_pct = 2.0
                    except Exception:
                        expected_move_pct = 2.0
                    
                    # Calculate predicted price from direction and expected move
                    if price_current and expected_move_pct:
                        direction_multiplier = 1 if latest.get("direction") == "UP" else -1
                        price_pred_mid = price_current * (1 + (direction_multiplier * expected_move_pct / 100))
                    else:
                        # Fallback: use default 2% move
                        price_pred_mid = price_current * 1.02 if latest.get("direction") == "UP" else price_current * 0.98 if price_current else None
                    
                    # Map direction to BUY/SELL/HOLD
                    direction_str = latest.get("direction", "HOLD")
                    if direction_str == "UP":
                        action = "BUY"
                    elif direction_str == "DOWN":
                        action = "SELL"
                    else:
                        action = "HOLD"
                    
                    prediction = {
                        "symbol": symbol,
                        "type": "stock",
                        "price_current": price_current,
                        "price_pred_mid": price_pred_mid,
                        "confidence": latest.get("confidence", 0.5),
                        "direction": action,
                        "momentum": abs(expected_move_pct) / 10.0,  # Normalize to 0-1 scale
                        "timestamp": latest.get("run_at", time.time()),
                        "horizon_h": 6  # Real 6h predictions
                    }
                    results["stocks"].append(prediction)
                else:
                    # No prediction available yet
                    failed_symbols["stocks"].append({
                        "symbol": symbol,
                        "error": "No prediction available in database"
                    })
            except Exception as e:
                LOGGER.warning(f"Multi-prediction failed for stock {symbol}: {e}")
                failed_symbols["stocks"].append({
                    "symbol": symbol,
                    "error": str(e)
                })
                continue

        # Generate predictions for crypto symbols using REAL 6H system
        for symbol in CRYPTO_SYMBOLS:
            try:
                # Get latest 6h prediction from database
                latest = backend.get_latest_prediction(symbol)
                
                if latest:
                    # Parse features_json to get current price and expected move
                    try:
                        features_json = latest.get("features_json")
                        if features_json:
                            import json
                            features = json.loads(features_json) if isinstance(features_json, str) else features_json
                            price_current = features.get("current_price")
                        else:
                            price_current = None
                    except Exception:
                        price_current = None
                    
                    # Parse params_json for additional data
                    try:
                        params_json = latest.get("params_json")
                        if params_json:
                            params = json.loads(params_json) if isinstance(params_json, str) else params_json
                            expected_move_pct = params.get("expected_move_pct", 2.0)  # Default 2%
                        else:
                            expected_move_pct = 2.0
                    except Exception:
                        expected_move_pct = 2.0
                    
                    # Calculate predicted price from direction and expected move
                    if price_current and expected_move_pct:
                        direction_multiplier = 1 if latest.get("direction") == "UP" else -1
                        price_pred_mid = price_current * (1 + (direction_multiplier * expected_move_pct / 100))
                    else:
                        # Fallback: use default 2% move
                        price_pred_mid = price_current * 1.02 if latest.get("direction") == "UP" else price_current * 0.98 if price_current else None
                    
                    # Map direction to BUY/SELL/HOLD
                    direction_str = latest.get("direction", "HOLD")
                    if direction_str == "UP":
                        action = "BUY"
                    elif direction_str == "DOWN":
                        action = "SELL"
                    else:
                        action = "HOLD"
                    
                    prediction = {
                        "symbol": symbol,
                        "type": "crypto",
                        "price_current": price_current,
                        "price_pred_mid": price_pred_mid,
                        "confidence": latest.get("confidence", 0.5),
                        "direction": action,
                        "momentum": abs(expected_move_pct) / 10.0,  # Normalize to 0-1 scale
                        "timestamp": latest.get("run_at", time.time()),
                        "horizon_h": 6  # Real 6h predictions
                    }
                    results["crypto"].append(prediction)
                else:
                    # No prediction available yet
                    failed_symbols["crypto"].append({
                        "symbol": symbol,
                        "error": "No prediction available in database"
                    })
            except Exception as e:
                LOGGER.warning(f"Multi-prediction failed for crypto {symbol}: {e}")
                failed_symbols["crypto"].append({
                    "symbol": symbol,
                    "error": str(e)
                })
                continue

        # Generate predictions for VIP coins (skip - not implemented yet)
        # VIP coins will use same 6h system once added to watchlist

        # Update tracking globals
        _LAST_MULTI_PREDICTION_TIME = time.time()
        _LAST_MULTI_PREDICTION_COUNTS = {
            "stocks": len(results["stocks"]),
            "crypto": len(results["crypto"]),
            "vip": len(results["vip"])
        }

        result = {
            "ok": True,
            "predictions": results,
            "counts": _LAST_MULTI_PREDICTION_COUNTS.copy(),
            "total": sum(_LAST_MULTI_PREDICTION_COUNTS.values()),
            "failed_symbols": failed_symbols if (failed_symbols["stocks"] or failed_symbols["crypto"]) else None,
            "timestamp": _LAST_MULTI_PREDICTION_TIME,
            "cached": False,
            "note": "Using real 6h predictions from PostgreSQL (GHOST MAXIMUM v2.0)"
        }

        # Cache result to prevent provider exhaustion
        _LAST_MULTI_PREDICTION_RESULT = result.copy()

        return result
    except Exception as e:
        LOGGER.exception("Multi-prediction generation failed")
        return {"ok": False, "error": str(e)}


async def api_vip_status():
    """
    Get VIP microcap coin status with real-time prices.

    Returns:
        {
            'ok': True,
            'coins': [
                {'symbol': 'WEPE', 'price': 0.000123, 'change_1h': 12.5, 'volume_24h': 1500000},
                ...
            ],
            'last_scan': 1731654000,
            'opportunities': 2
        }
    """
    try:
        from core.crypto.vip_providers import get_all_vip_prices

        vip_data = get_all_vip_prices()

        # Format for mobile UI
        coins = []
        for symbol, data in vip_data.items():
            if data.get('ok'):
                coins.append({
                    'symbol': symbol,
                    'price': data.get('price'),
                    'change_1h': data.get('change_1h', 0),
                    'volume_24h': data.get('volume_24h', 0),
                    'market_cap': data.get('market_cap'),
                    'provider': data.get('provider', 'unknown')
                })

        return {
            'ok': True,
            'coins': coins,
            'count': len(coins),
            'last_scan': int(time.time()),
            'timestamp': int(time.time())
        }
    except Exception as e:
        LOGGER.error(f"VIP status API failed: {e}")
        return {
            'ok': False,
            'error': str(e),
            'coins': [],
            'timestamp': int(time.time())
        }


def _validate_cron_request(request) -> bool:
    """Validate cron request has correct secret (if configured)"""
    if not CRON_SECRET:
        return True  # No secret configured, allow all
    
    # Check header
    provided = request.headers.get("X-Cron-Secret", "")
    if provided == CRON_SECRET:
        return True
    
    # Also check query param as fallback
    if request.query_params.get("secret") == CRON_SECRET:
        return True
    
    return False



# ── Classes ──────────────────────────────────────────────────────

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        from datetime import datetime

        payload: dict[str, object] = {
            "ts": datetime.now(tz=UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "service": "ghost-wol",
            "msg": record.getMessage(),
            "trace_id": _cv_trace_id.get(),
            "path": _cv_path.get(),
            "method": _cv_method.get(),
        }
        # Include extras (added by logger(..., extra={}))
        for k, v in record.__dict__.items():
            if k in (
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
            ):
                continue
            # Avoid overriding core fields unless intentional
            if k not in payload:
                payload[k] = v
        if record.exc_info:
            try:
                etype = (
                    record.exc_info[0].__name__
                    if record.exc_info and record.exc_info[0]
                    else "Exception"
                )
                emsg = str(record.exc_info[1]) if record.exc_info and record.exc_info[1] else ""
                payload["error_type"] = etype
                payload["error"] = emsg
            except Exception:
                pass
        try:
            import json as _json

            return _json.dumps(payload, separators=(",", ":"))
        except Exception:
            return f"{payload}"


class _LogDedupFilter(logging.Filter):
    """Suppress repeated identical log records within a sliding time window.

    - window_s: seconds to consider messages as duplicates
    - min_repeats: only start suppressing from the Nth repetition within the window
    Keys on (levelno, logger name, message template) to avoid over-suppressing.
    """

    def __init__(self, window_s: float = 10.0, min_repeats: int = 2):
        super().__init__()
        self.window_s = float(max(0.1, window_s))
        self.min_repeats = int(max(1, min_repeats))
        self._seen: dict[tuple[int, str, str], list[float]] = {}

    def filter(self, record: logging.LogRecord) -> bool:  # True to log, False to drop
        try:
            now = time.time()
            key = (record.levelno, record.name or "", getattr(record, "msg", ""))
            buf = self._seen.setdefault(key, [])
            # prune old timestamps
            cutoff = now - self.window_s
            i = 0
            for i in range(len(buf)):
                if buf[i] >= cutoff:
                    break
            if i > 0:
                del buf[:i]
            buf.append(now)
            # allow through until we hit min_repeats within the window
            if len(buf) < self.min_repeats:
                return True
            # From min_repeats onward within the window, drop duplicates
            return False
        except Exception:
            return True



