"""Routes: cockpit — extracted from wolf_app.py (Step 12)"""
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
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request, Response, Query, Header, BackgroundTasks, WebSocket
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse, RedirectResponse, FileResponse
from fastapi.templating import Jinja2Templates

try:
    import httpx
except ImportError:
    httpx = None

try:
    from state import APP_STATE, POOL, DB_URL, PREDICTION_HISTORY
except ImportError:
    APP_STATE = {}
    POOL = None
    DB_URL = ""
    PREDICTION_HISTORY = []

# ── Also inject wolf_helpers globals (private helper functions + shared state) ─
import wolf_helpers as _wh
globals().update({k: v for k, v in vars(_wh).items() if not k.startswith("__")})
del _wh

# ── Inject all app-config globals into this route module ─────────────────────
# Mirrors wolf_app.py's pattern: provides all module-level constants that route
# handlers reference directly, without needing per-name imports.
import engines.app_config as _ac
globals().update({k: v for k, v in vars(_ac).items() if not k.startswith("__")})
del _ac

router = APIRouter()
LOGGER = logging.getLogger("ghost")

MEDIA_TEXT_HTML = "text/html"
HTML_INDEX = "index.html"
STATIC_DIR = os.getenv("STATIC_DIR", "static")
UI_DIR = os.getenv("UI_DIR", "ui")

# ── Jinja2 template engine ────────────────────────────────────────────────
# Use an absolute path so the template directory resolves correctly on Railway
# regardless of the process working directory.
_HERE = os.path.dirname(os.path.abspath(__file__))
_TEMPLATES_DIR = os.path.join(_HERE, "..", "templates")
if not os.path.isdir(_TEMPLATES_DIR):
    # Fallback: try cwd-relative path (local dev)
    _TEMPLATES_DIR = os.path.join(os.getcwd(), "templates")
try:
    _TEMPLATES = Jinja2Templates(directory=_TEMPLATES_DIR)
    LOGGER.info(f"[COCKPIT] Jinja2Templates initialized: {os.path.abspath(_TEMPLATES_DIR)}")
except Exception as _tmpl_err:
    _TEMPLATES = None
    LOGGER.error(f"[COCKPIT] Jinja2Templates init failed: {_tmpl_err}")

try:
    from wolf_app import _STATIC_CACHE_BUST
except ImportError:
    import time as _time
    _STATIC_CACHE_BUST = str(int(_time.time()))

# --- 18 endpoints ---

@router.get("/", include_in_schema=False)
async def _root_index():
    """Single entrypoint: redirect root traffic to Cockpit V3."""
    return RedirectResponse(url="/cockpit", status_code=307)


@router.get("/index.html", include_in_schema=False)
async def _root_index_alias():
    return await _root_index()


@router.get("/ui", include_in_schema=False)
async def _ui_entrypoint():
    # Always serve the legacy UI bundle if present
    try:
        index_path = os.path.join(UI_DIR, HTML_INDEX)
        if os.path.isdir(UI_DIR) and os.path.exists(index_path):
            return FileResponse(index_path, media_type=MEDIA_TEXT_HTML)
    except Exception:
        pass
    # Fallback to static index if ui_dist missing
    try:
        static_index = os.path.join(STATIC_DIR, HTML_INDEX)
        if os.path.isdir(STATIC_DIR) and os.path.exists(static_index):
            return FileResponse(static_index, media_type=MEDIA_TEXT_HTML)
    except Exception:
        pass
    # Fallback to cockpit redirect (avoids missing `request` arg)
    return RedirectResponse(url="/cockpit", status_code=307)


@router.get("/cockpit", include_in_schema=False)
async def _cockpit_page(request: Request):
    """Serve Ghost v5 cockpit — Robinhood-style redesign."""
    # Try v5 first, fall back through v4 → v3 to guarantee something renders
    for tmpl_name in ("cockpit_v5.html", "cockpit_v4.html", "cockpit_v3.html"):
        try:
            if _TEMPLATES is None:
                raise RuntimeError("Jinja2Templates not initialized")
            response = _TEMPLATES.TemplateResponse(
                tmpl_name,
                {
                    "request": request,
                    "GHOST_API_TOKEN": os.getenv("GHOST_API_TOKEN", ""),
                    "cache_bust": _STATIC_CACHE_BUST,
                }
            )
            # CRITICAL: Prevent browser caching to force fresh JS load
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return response
        except Exception as e:
            LOGGER.warning(f"[COCKPIT] Could not render {tmpl_name}: {e}")
            continue

    # Last-resort: serve the raw file if Jinja2 is broken
    for tmpl_name in ("cockpit_v5.html", "cockpit_v4.html", "cockpit_v3.html"):
        raw_path = os.path.join(os.path.abspath(_TEMPLATES_DIR), tmpl_name)
        if os.path.isfile(raw_path):
            LOGGER.info(f"[COCKPIT] Serving raw template file: {raw_path}")
            return FileResponse(raw_path, media_type=MEDIA_TEXT_HTML)

    # Absolute last resort — inline HTML with diagnostics
    LOGGER.error("[COCKPIT] All cockpit templates unavailable — serving fallback page")
    tmpl_dir_abs = os.path.abspath(_TEMPLATES_DIR)
    try:
        available = os.listdir(tmpl_dir_abs)
    except Exception:
        available = ["<templates dir not readable>"]
    return Response(
        f"""<!DOCTYPE html>
<html>
  <head><meta charset="utf-8"><title>Ghost Cockpit</title></head>
  <body>
    <h1>Ghost Protocol</h1>
    <p><strong>Templates directory:</strong> {tmpl_dir_abs}</p>
    <p><strong>Available files:</strong> {', '.join(available)}</p>
    <p>If this is Railway, ensure the <code>templates/</code> folder is committed and deployed.</p>
  </body>
</html>""",
        media_type=MEDIA_TEXT_HTML,
        status_code=200,
    )


@router.get("/cockpit.html", include_in_schema=False)
async def _cockpit_page_alias(request: Request):
    return await _cockpit_page(request)


@router.get("/wolf", include_in_schema=False)
async def _wolf_page(request: Request):
    """Legacy /wolf route - redirects to main cockpit."""
    return await _cockpit_page(request)


@router.get("/wolf.html", include_in_schema=False)
async def _wolf_page_alias(request: Request):
    return await _cockpit_page(request)


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_ui():
    """
    📊 Interactive Performance Dashboard UI
    
    Beautiful web interface showing:
    - Real-time P&L metrics
    - Win rates across different time periods
    - Top/worst performing symbols
    - Confidence calibration analysis
    
    Auto-refreshes every 5 minutes.
    """
    try:
        from pathlib import Path
        
        # Read the dashboard HTML template
        template_path = Path(__file__).parent / "templates" / "dashboard.html"
        
        if not template_path.exists():
            return HTMLResponse(
                content="<h1>Dashboard template not found</h1>",
                status_code=404
            )
        
        with open(template_path, "r") as f:
            html_content = f.read()
        
        return HTMLResponse(content=html_content)
    
    except Exception as e:
        LOGGER.error(f"Dashboard UI failed: {e}", exc_info=True)
        return HTMLResponse(
            content=f"<h1>Error loading dashboard</h1><p>{str(e)}</p>",
            status_code=500
        )


@router.get("/api/v3/cockpit/status")
async def api_v3_cockpit_status():
    """
    Get system status for cockpit header.
    
    Returns mode, active status, uptime, etc.
    """
    try:
        # Calculate health score from INTEGRITY audit (real system health)
        health_score = 0
        total_predictions = 0
        integrity_issues = 0
        try:
            from core.integrity import run_audit
            audit = run_audit(auto_fix=False)
            health_score = audit.get("health_score", 50)
            integrity_issues = audit.get("issues_remaining", 0)
            total_predictions = audit.get("summary", {}).get("total_predictions", 0)
            if total_predictions == 0:
                total_predictions = len(_LATEST_PREDICTIONS)
        except Exception as e:
            LOGGER.warning(f"Could not get integrity score for cockpit: {e}")
            # Fallback: count predictions but cap at 80 (never fake 100)
            total_predictions = len(_LATEST_PREDICTIONS)
            health_score = min(80, total_predictions * 5) if total_predictions > 0 else 30
        
        # Calculate grade based on score
        if health_score >= 90:
            grade = "A"
        elif health_score >= 80:
            grade = "B"
        elif health_score >= 70:
            grade = "C"
        elif health_score >= 60:
            grade = "D"
        else:
            grade = "F"
        
        return {
            "ok": True,
            "mode": str(STATE.get("mode", "live")),
            "active": bool(STATE.get("active", True)),
            "uptime_seconds": int(time.time() - _START_TS) if "_START_TS" in globals() else 0,
            "version": "3.0",
            "ghost_health": health_score,
            "ghost_health_score": health_score,
            "ghost_health_grade": grade,
            "predictions_today": total_predictions,
            "integrity_issues": integrity_issues,
        }
    
    except Exception as e:
        LOGGER.error(f"Cockpit status failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e)
        }


@router.post("/api/cockpit/start")
async def api_cockpit_start():
    """Start the Ghost prediction engine."""
    try:
        STATE["active"] = True
        STATE["engine_status"] = "running"
        _add_event("control", "Engine started via cockpit", {"active": True})
        return {
            "ok": True,
            "active": True,
            "message": "Engine started"
        }
    except Exception as e:
        LOGGER.error(f"Cockpit start failed: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}


@router.post("/api/cockpit/stop")
async def api_cockpit_stop():
    """Stop the Ghost prediction engine."""
    try:
        STATE["active"] = False
        STATE["engine_status"] = "stopped"
        _add_event("control", "Engine stopped via cockpit", {"active": False})
        return {
            "ok": True,
            "active": False,
            "message": "Engine stopped"
        }
    except Exception as e:
        LOGGER.error(f"Cockpit stop failed: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}


@router.post("/api/cockpit/reset")
async def api_cockpit_reset():
    """Reset the Ghost state (clear positions)."""
    try:
        STATE["qty"] = 0.0
        STATE["avg_cost"] = 0.0
        _persist_save()
        _add_event("state.reset", "State reset via cockpit", {"qty": 0.0, "avg_cost": 0.0})
        return {
            "ok": True,
            "active": bool(STATE.get("active", True)),
            "reset": True,
            "message": "State reset"
        }
    except Exception as e:
        LOGGER.error(f"Cockpit reset failed: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}


@router.get("/api/cockpit")
async def api_cockpit_snapshot():
    """
    High-level cockpit snapshot: wraps Ghost 2.x health and basic system status.
    Designed for the web UI; must not raise HTTP errors on normal operation.
    """
    # Build system block
    try:
        import wolf_app as _wa
        _app_ver = getattr(_wa.APP, "version", None)
    except Exception:
        _app_ver = None
    system = {
        "mode": str(STATE.get("mode", "live")),
        "active": bool(STATE.get("active", True)),
        "version": _app_ver,
        "uptime_seconds": int(time.time() - _START_TS) if "_START_TS" in globals() else 0,
    }

    try:
        # Reuse the same logic as /api/health/predictions
        vip_provider_health = {}

        try:
            from core.crypto.vip_providers import get_vip_provider_health
            vip_provider_health = get_vip_provider_health()
        except Exception as e:
            LOGGER.warning(f"Could not get VIP provider health: {e}")

        # Compute Ghost Score V2
        ghost_score_v2 = {}
        try:
            from core.metrics.ghost_score import compute_ghost_score_v2, get_current_risk_status

            # FIX (Mar 1, 2026): Use edge set size as denominator
            from config.symbols import get_edge_set as _gs_edge_set2
            _edge_symbols2 = _gs_edge_set2()
            total_symbols = len(_edge_symbols2)
            _ls = sum(1 for p in _LATEST_PREDICTIONS.values() if isinstance(p, dict) and p.get("engine") == "stock_v2")
            _lc = sum(1 for p in _LATEST_PREDICTIONS.values() if isinstance(p, dict) and p.get("engine") != "stock_v2")
            symbols_with_data = _ls + _lc

            # Compute live avg_confidence from actual predictions
            _live_confs = [p.get("confidence", 0) for p in _LATEST_PREDICTIONS.values() if isinstance(p, dict) and p.get("confidence")]
            _live_avg_conf = sum(_live_confs) / len(_live_confs) if _live_confs else 0.6

            data_quality = {
                "symbols_with_data": symbols_with_data,
                "total_symbols": total_symbols,
                "provider_redundancy": 0.7,
                "avg_confidence": round(_live_avg_conf, 3)
            }

            # Prediction coverage — use live count as primary
            predictions_generated = max(sum(_LAST_MULTI_PREDICTION_COUNTS.values()), len(_LATEST_PREDICTIONS))
            prediction_coverage = {
                "predictions_generated": predictions_generated,
                "total_expected": total_symbols,
                "success_rate_estimate": 0.6
            }

            # Risk status
            risk_status = get_current_risk_status()

            # Compute score
            ghost_score_v2 = compute_ghost_score_v2(
                data_quality=data_quality,
                prediction_coverage=prediction_coverage,
                risk_status=risk_status
            )
        except Exception as e:
            LOGGER.warning(f"Could not compute Ghost Score V2: {e}")
            # Provide basic fallback score
            ghost_score_v2 = {
                "score": 0,
                "status": "degraded",
                "grade": "?",
                "components": {
                    "data_quality": 0,
                    "prediction_coverage": 0,
                    "risk_behavior": 0
                },
                "note": f"Ghost Score V2 unavailable: {e}"
            }

        # Get risk guard status
        risk_guard_status = {}
        try:
            from core.risk.risk_guard import get_risk_guard
            risk_guard = get_risk_guard()
            risk_guard_status = risk_guard.get_status()
        except Exception as e:
            LOGGER.warning(f"Could not get risk guard status: {e}")
            risk_guard_status = {"enabled": False, "error": str(e)}

        # Get latest predictions from database (Phase 2 fix)
        latest_predictions = {}
        try:
            # Query latest prediction for WOLF and other key symbols
            key_symbols = ["WOLF"] + STOCK_SYMBOLS[:5]  # WOLF + top 5 stocks
            for sym in key_symbols:
                try:
                    pred = predictor.get_latest_prediction(sym)
                    if pred:
                        latest_predictions[sym] = {
                            "id": pred.id,
                            "run_at": pred.run_at,
                            "confidence": pred.confidence,
                            "direction": pred.direction,
                            "horizon_h": pred.horizon_h,
                        }
                except Exception as e:
                    LOGGER.debug(f"Could not get prediction for {sym}: {e}")
        except Exception as e:
            LOGGER.warning(f"Could not query latest predictions: {e}")

        # Build predictions from in-memory store
        # FIX (Feb 24, 2026): Expose FULL prediction payload.
        # Previously stripped ~70% of fields (price, action, gates, trust, momentum).
        predictions = {}
        try:
            for sym, pred in _LATEST_PREDICTIONS.items():
                predictions[sym] = {
                    "prediction_id": pred.get("prediction_id"),
                    "symbol": sym,
                    "run_at": pred.get("run_at"),
                    "confidence": pred.get("confidence"),
                    "direction": pred.get("direction"),
                    "action": pred.get("action"),
                    "horizon_h": pred.get("horizon_h", 48),
                    "engine": pred.get("engine", "turbo"),
                    "confirmations": pred.get("confirmations"),
                    "intel_applied": pred.get("intel_applied", False),
                    "price": pred.get("price"),
                    "price_at_prediction": pred.get("price_at_prediction"),
                    "market": pred.get("market"),
                    "provider": pred.get("provider"),
                    "should_predict": pred.get("should_predict"),
                    "gates_passed": pred.get("gates_passed"),
                    "reasons": pred.get("reasons"),
                    "momentum": pred.get("momentum"),
                    "expected_move_pct": pred.get("expected_move_pct"),
                    "trust_level": pred.get("trust_level"),
                    "trust_boost": pred.get("trust_boost"),
                }
        except Exception as e:
            LOGGER.warning(f"Failed to build predictions for /api/cockpit: {e}")

        # Build ghost_2x block
        # Compute live symbol counts from _LATEST_PREDICTIONS as authoritative source
        _live_stock_count = sum(1 for p in _LATEST_PREDICTIONS.values() if isinstance(p, dict) and p.get("engine") == "stock_v2")
        _live_crypto_count = sum(1 for p in _LATEST_PREDICTIONS.values() if isinstance(p, dict) and p.get("engine") != "stock_v2")
        _live_counts = {
            "stocks": max(_LAST_MULTI_PREDICTION_COUNTS.get("stocks", 0), _live_stock_count),
            "crypto": max(_LAST_MULTI_PREDICTION_COUNTS.get("crypto", 0), _live_crypto_count),
            "vip": _LAST_MULTI_PREDICTION_COUNTS.get("vip", 0),
        }
        ghost_2x = {
            "ok": True,
            "symbol_counts": _live_counts,
            "vip_provider_health": vip_provider_health,
            "ghost_score_v2": ghost_score_v2,
            "risk_guard_status": risk_guard_status,
            "last_multi_prediction_run_time": _LAST_MULTI_PREDICTION_TIME,
            "last_telegram_send_time": _LAST_TELEGRAM_SEND_TIME,
            "last_telegram_status": _LAST_TELEGRAM_STATUS,
            "last_telegram_error": _LAST_TELEGRAM_ERROR,
            "latest_predictions": latest_predictions,  # Phase 2: Show actual predictions from DB
        }

        return {
            "status": "ok",
            "system": system,
            "ghost_2x": ghost_2x,
            "predictions": predictions if predictions else None,
            "timestamp": time.time()
        }

    except Exception as exc:
        LOGGER.exception("cockpit snapshot failed", exc_info=exc)
        return {
            "status": "error",
            "system": system,
            "ghost_2x": None,
            "error": "cockpit_snapshot_failed",
            "timestamp": time.time()
        }


@router.get("/api/cockpit/stream")
async def sse_cockpit_stream(request: Request):
    """SSE stream with proper event types: status, ping, snapshot."""

    async def gen():
        last_sent_etag = None
        start_time = time.time()
        last_heartbeat = time.time()

        # Event 1: Send status event on connect
        try:
            status_data = {
                "status": "live",
                "ts": int(time.time()),
                "sim_mode": SIM_MODE,
                "focus_wolf_only": FOCUS_WOLF_ONLY,
            }
            yield f"event: status\ndata: {json.dumps(status_data)}\n\n"
        except Exception:
            pass

        # Event 2: Send initial snapshot immediately
        try:
            snap_resp = await api_cockpit_snapshot()
            data = getattr(snap_resp, "body", None)
            if data is None:
                # Extract the actual response content before serializing
                if isinstance(snap_resp, JSONResponse):
                    try:
                        content = snap_resp.body if hasattr(snap_resp, "body") else b"{}"
                        data = (
                            content
                            if isinstance(content, bytes)
                            else json.dumps(content).encode("utf-8")
                        )
                    except Exception:
                        data = b"{}"
                elif isinstance(snap_resp, dict):
                    data = json.dumps(snap_resp).encode("utf-8")
                else:
                    data = json.dumps(str(snap_resp)).encode("utf-8")

            yield f"event: snapshot\ndata: {data.decode('utf-8')}\n\n"
        except Exception as e:
            LOGGER.error(f"sse_initial_snapshot_error: {e}")

        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                LOGGER.info("SSE cockpit client disconnected")
                break

            # TTL: Close stream after 30 minutes
            if time.time() - start_time > 1800:
                LOGGER.info("SSE cockpit stream TTL expired (30min)")
                break

            # Event 3: Send ping every 10 seconds (reduced from 15s for better responsiveness)
            if time.time() - last_heartbeat > 10:
                ping_data = {"ts": int(time.time())}
                yield f"event: ping\ndata: {json.dumps(ping_data)}\n\n"
                last_heartbeat = time.time()

            # Wait 5 seconds between snapshot checks
            await _async_sleep(5.0)

            # Event 4: Send snapshot if data changed
            try:
                snap_resp = await api_cockpit_snapshot()
                raw = getattr(snap_resp, "body", None)
                if raw is None:
                    raw = json.dumps(snap_resp).encode("utf-8")  # type: ignore[arg-type]

                # Naive change detection by ETag header if present
                etag = None
                try:
                    etag = getattr(snap_resp, "headers", {}).get("ETag")  # type: ignore[call-arg]
                except Exception:
                    etag = None

                if etag:
                    if etag == last_sent_etag:
                        continue  # No change, skip sending
                    last_sent_etag = etag

                yield f"event: snapshot\ndata: {raw.decode('utf-8')}\n\n"
            except Exception as e:
                LOGGER.error(f"sse_snapshot_error: {e}")
                continue

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/api/cockpit/status")
async def cockpit_status():
    try:
        price, prev, provider = get_wolf_price()
        q = float(STATE.get("qty", 0.0))
        a = float(STATE.get("avg_cost", 0.0))
        px = price if price is not None else (prev if prev is not None else a)
        nav = float(round(q * (px or 0.0), 2))
        pnl_abs = float(round(q * ((px or 0.0) - a), 2))
        flags = {
            "using_prev_close": (price is None and prev is not None),
            "manual": (
                (PRICE_OVERRIDE.get("symbol") or "") == WOLF
                and time.time() < float(PRICE_OVERRIDE.get("until") or 0)
            ),
        }
        return {
            "as_o": int(time.time()),
            "provider": provider or "unavailable",
            "price": (None if price is None else float(price)),
            "nav": nav,
            "pnl_abs": pnl_abs,
            "flags": flags,
        }
    except Exception:
        return {
            "as_o": int(time.time()),
            "provider": "unavailable",
            "price": None,
            "nav": None,
            "pnl_abs": None,
            "flags": {},
        }


@router.get("/api/cockpit/snapshot")
async def api_cockpit_legacy():
    """Legacy cockpit snapshot with prices, portfolio, news. Use /api/cockpit for Ghost 2.x data."""
    price, prev, provider = get_wolf_price()

    # CRITICAL: Handle case where all providers fail and return None
    if price is None and prev is None:
        # Return minimal error response instead of crashing
        return {
            "ok": False,
            "error": "price_unavailable",
            "message": "All price providers failed. Check API keys and network connectivity.",
            "reasons": ["price:all-providers-failed"],
            "prices": {
                "price": None,
                "prev_close": None,
                "provider": provider or "unavailable",
                "change_pct": None,
            },
            "portfolio": {
                "qty": float(STATE.get("qty", 0.0)),
                "avg_cost": float(STATE.get("avg_cost", 0.0)),
            },
            "ts": int(time.time()),
        }

    change_pct = None
    try:
        base_prev = prev
        base_price = price if price is not None else None
        if base_price is not None and base_prev and base_prev > 0:
            change_pct = (base_price - base_prev) / base_prev * 100.0
    except Exception:
        change_pct = None
    qty = float(STATE.get("qty", 0.0))
    avg = float(STATE.get("avg_cost", 0.0))
    # We'll decide the effective display price below (may fallback to prev_close on anomaly)
    display_price = price if price is not None else prev

    news = get_wolf_news(limit=10)
    reasons: list[str] = []
    if not provider or provider == "unavailable":
        reasons.append("price:provider-unavailable")
    if price is None and prev is None:
        reasons.append("price:unavailable")
    elif price is None and prev is not None:
        reasons.append("price:stale-prev-only")
    note = news.get("note")
    if not POLYGON_KEY:
        reasons.append("news:provider-missing")
    elif note == "rate-limited":
        reasons.append("news:rate-limited")

    now_ts = int(time.time())
    # Price anomaly & corporate-action guardrail
    anomaly_active = False
    provider_effective = provider
    try:
        fresh_reuters = False
        if REUTERS_FEEDS_ON and isinstance(news.get("items"), list):
            for it in news.get("items", []):
                if (it or {}).get("src") == "reuters":
                    ts = it.get("ts")
                    # Normalize ts to int seconds
                    if isinstance(ts, (int, float)):
                        ts_num = int(ts)
                    else:
                        try:
                            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                            ts_num = int(dt.timestamp())
                        except Exception:
                            ts_num = now_ts
                    # Consider only items in recent window
                    if (now_ts - ts_num) <= (PRICE_ANOMALY_NEWS_WINDOW_MIN * 60):
                        # If Reuters symbol filtering is enabled, prefer items that mention WOLF
                        syms = it.get("syms") or []
                        head = it.get("headline") or ""
                        if (
                            (not REUTERS_SYMBOLS and not REUTERS_KEYWORDS)
                            or (WOLF in syms)
                            or ("WOLF" in head.upper())
                        ):
                            fresh_reuters = True
                            break
        # Deviation check vs prev_close
        if fresh_reuters and price is not None and prev and prev > 0:
            ratio = price / prev if price >= prev else prev / price
            if ratio >= max(1.0, float(PRICE_ANOMALY_X)):
                anomaly_active = True
                if REASON_PRICE_ANOMALY not in reasons:
                    reasons.append(REASON_PRICE_ANOMALY)
                # Prefer prev_close for display if available
                if prev is not None:
                    display_price = prev
                    provider_effective = "prev-close"
    except Exception:
        pass
    # Corporate-action guard: extreme intraday move or large provider spread
    try:
        extreme_move = False
        if change_pct is not None and abs(change_pct) >= 60.0:
            extreme_move = True
        spread_bad = False
        try:
            sp = PRICE_DIAG.get("provider_spread") if isinstance(PRICE_DIAG, dict) else None
            if sp is not None and float(sp) > float(PRICE_MAX_DEVIATION_OPEN):
                spread_bad = True
        except Exception:
            spread_bad = False
        if extreme_move or spread_bad:
            anomaly_active = True
            if REASON_CORP_ACTION_SUSPECTED not in reasons:
                reasons.append(REASON_CORP_ACTION_SUSPECTED)
            if prev is not None:
                display_price = prev
                provider_effective = "prev-close"
    except Exception:
        pass
    # Also treat provider quorum failure as anomaly
    try:
        if isinstance(PRICE_DIAG, dict) and PRICE_DIAG.get("anomaly"):
            anomaly_active = True
            if REASON_PRICE_ANOMALY not in reasons:
                reasons.append(REASON_PRICE_ANOMALY)
            if prev is not None:
                display_price = prev
                provider_effective = "prev-close"
    except Exception:
        pass
    # Build UI-compatible snapshot
    row_current = display_price if display_price is not None else avg
    # If manual override active, provider will be 'manual'
    manual_active = provider == "manual"
    # Recompute portfolio metrics based on effective display price
    market_value = round(qty * row_current, 2) if (row_current is not None) else None
    pnl_abs = round((row_current - avg) * qty, 2) if (row_current is not None) else None
    pnl_pct = (
        round(((row_current - avg) / avg) * 100.0, 6)
        if (row_current is not None and avg > 0)
        else None
    )
    ui_row = {
        "symbol": WOLF,
        "sym": WOLF,
        "type": "stock",
        "qty": float(f"{qty:.8f}"),
        "entry": float(f"{avg:.2f}"),
        "current": float(f"{row_current:.2f}"),
        "mark_value": round(qty * row_current, 2),
        "pnl_abs": round((row_current - avg) * qty, 2),
        "pnl_pct": float(f"{(((row_current - avg) / avg) * 100.0) if avg > 0 else 0.0:.6f}"),
        "gps": 7.2,
        "stale": (price is None) or manual_active or anomaly_active,
        "src": provider_effective or ("prev-close" if prev is not None else "unavailable"),
        "snapshot_id": "pending",
    }
    ui_news: list[dict] = []
    try:
        for it in news.get("items", [])[:10]:
            ts = it.get("ts")
            ts_num: int
            if isinstance(ts, (int, float)):
                ts_num = int(ts)
            else:
                try:
                    dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
                    ts_num = int(dt.timestamp())
                except Exception:
                    ts_num = now_ts
            src_name = it.get("src") or ("polygon" if POLYGON_KEY else "news")
            # Lightweight sentiment tag
            sent_val = None
            try:
                if isinstance(it.get("sent"), (int, float)):
                    sent_val = float(it.get("sent"))
                else:
                    # score headline+desc via simple rules if sentiment not precomputed
                    h = it.get("headline") or ""
                    d = it.get("description") or ""
                    sent_val = _score_text_rules((h + ". " + d).strip())
            except Exception:
                sent_val = None
            if sent_val is None:
                tag = "• Neutral"
            elif sent_val >= 0.1:
                tag = "↑ Bullish"
            elif sent_val <= -0.1:
                tag = "↓ Bearish"
            else:
                tag = "• Neutral"
            ui_news.append(
                {
                    "ts": ts_num,
                    "url": it.get("url"),
                    "title": it.get("headline") or "",
                    "src": src_name,
                    "tag": tag,
                    "sent": (None if sent_val is None else float(f"{float(sent_val):.3f}")),
                }
            )
    except Exception:
        ui_news = []
    # Macro Brain (optional)
    macro = {"enabled": False}
    try:
        ns = (news.get("news_signal") or {}).get("score")
        macro = _macro_brain(price, ns)
    except Exception:
        macro = {"enabled": False}

    # Collect recent events for diagnostics panel
    try:
        _recent_events = list(EVENTS)[-20:]
    except Exception:
        _recent_events = []
    try:
        _error_count = sum(
            1
            for _e in _recent_events
            if isinstance((_e or {}).get("type"), str)
            and ("error" in str(_e.get("type")).lower() or "fail" in str(_e.get("type")).lower())
        )
    except Exception:
        _error_count = 0

    stocks_ok = bool((provider and provider not in ("", "unavailable")) or (prev is not None)) and (
        not manual_active
    )
    if anomaly_active:
        stocks_ok = False
    is_open, next_open_ts = _is_market_open_now()
    # AI preview
    try:
        ns_val = (news.get("news_signal") or {}).get("score")
    except Exception:
        ns_val = None
    feats = _extract_features(display_price, prev, qty, avg, ns_val)
    gps, conf, reasons_ai, analogs_ai = _ai_infer(feats)
    cash_bal = float(STATE.get("cash", 0.0))
    # Build portfolio rows (multi-asset if available; enforce focus mode if enabled)
    rows: list[dict[str, Any]] = []
    positions = STATE.get("positions")
    try:
        if isinstance(positions, list) and positions:
            # Compute rows from saved positions
            for pos in positions:
                try:
                    sym = str(pos.get("symbol") or "").upper()
                    if FOCUS_WOLF_ONLY and sym != WOLF:
                        # Skip non-WOLF in focus mode
                        continue
                    market = str(pos.get("market") or pos.get("type") or "stock")
                    q = float(pos.get("qty") or pos.get("quantity") or 0.0)
                    entry = float(
                        pos.get("price_paid")
                        or pos.get("entry_price")
                        or pos.get("entry")
                        or pos.get("avg", 0.0)
                    )
                    # Current pricing: only reliable for focus ticker; others marked stale for now
                    cur = None
                    stale = True
                    src = "unavailable"
                    if sym == WOLF:
                        cur = row_current
                        stale = manual_active or (price is None) or anomaly_active
                        src = provider_effective or (
                            "prev-close" if prev is not None else "unavailable"
                        )
                    pnl_abs_i = ((cur - entry) * q) if (cur is not None) else 0.0
                    pnl_pct_i = (
                        (((cur - entry) / entry) * 100.0)
                        if (cur is not None and entry > 0)
                        else 0.0
                    )
                    rows.append(
                        {
                            "symbol": sym,
                            "sym": sym,
                            "type": market,
                            "qty": q,
                            "entry": entry,
                            "current": cur,
                            "mark_value": round((cur or 0.0) * q, 2),
                            "pnl_abs": round(pnl_abs_i, 2),
                            "pnl_pct": float(f"{pnl_pct_i:.6f}"),
                            "gps": 7.2,
                            "stale": stale,
                            "src": src,
                            "snapshot_id": "pending",
                        }
                    )
                except Exception:
                    continue
        else:
            rows = [ui_row]
    except Exception:
        rows = [ui_row]

    # Forecast summary with anomaly guardrail pause
    fsum = _forecast_summary_for_snapshot()
    forecast_full = None
    forecast_metrics = None

    # TWO-LINE OVERLAY: Ghost vs Live with accuracy metrics
    two_line_data = None
    try:
        if not manual_active and not anomaly_active:
            two_line_data = _build_two_line_forecast(WOLF)
    except Exception as e:
        print(f"[COCKPIT] Failed to build two-line overlay: {e}")
        two_line_data = None

    try:
        # Generate full stock forecast series (formerly "48h forecast")
        forecast_data = _build_forecast_series(48)
        forecast_full = {
            "label": "Ghost Predictions",
            "ticker": forecast_data.get("ticker"),
            "as_o": forecast_data.get("as_o"),
            "horizon_h": forecast_data.get("horizon_h"),
            "step_h": forecast_data.get("step_h"),
            "points": forecast_data.get("points", []),
            "summary": forecast_data.get("summary", {}),
        }
        # Compute accuracy metrics from SQLite if we have historical forecasts
        try:
            import sqlite3

            conn = sqlite3.connect(WOLF_SQLITE_PATH)
            cur = conn.cursor()
            # Get latest forecast scores if any
            cur.execute(
                "SELECT map, rmse, bias, scored_through_ts FROM forecast_scores ORDER BY scored_through_ts DESC LIMIT 1"
            )
            row = cur.fetchone()
            if row:
                forecast_metrics = {
                    "map": round(float(row[0]), 2) if row[0] is not None else None,
                    "rmse": round(float(row[1]), 2) if row[1] is not None else None,
                    "bias": round(float(row[2]), 2) if row[2] is not None else None,
                    "as_of": int(row[3]) if row[3] else None,
                }
            conn.close()
        except Exception:
            pass
    except Exception:
        pass

    # Build actual price series for predicted vs actual overlay
    actual_series = []
    try:
        actual_series = _build_actual_series(lookback_h=48)
    except Exception:
        pass

    if manual_active or (anomaly_active and int(FORECAST_PAUSE_ON_ANOMALY)):
        try:
            fsum = dict(fsum)
            fsum.update(
                {
                    "enabled": False,
                    "note": ("paused:manual_override" if manual_active else "paused:price_anomaly"),
                }
            )
            if forecast_full:
                forecast_full["enabled"] = False
                forecast_full["note"] = (
                    "paused:manual_override" if manual_active else "paused:price_anomaly"
                )
        except Exception:
            pass

    # Compute invested basis for better PnL% precision if position entry available
    try:
        invested = None
        if rows and rows[0].get("sym") == WOLF:
            invested = (
                float(rows[0]["entry"]) * float(rows[0]["qty"])
                if (rows[0].get("entry") and rows[0].get("qty"))
                else None
            )
    except Exception:
        invested = None

    snapshot = {
        "snapshot_id": f"ckpt-{now_ts}-{uuid.uuid4().hex[:4]}",
        "as_o": now_ts,
        "ticker": WOLF,
        "focus": {"enabled": True, "ticker": WOLF},
        "status": {
            "ok": stocks_ok,
            "active": bool(STATE.get("active", True)),
            "feeds": {
                "stocks": stocks_ok,
                "crypto": bool(os.getenv("CRYPTO_ENABLED", "0") == "1"),
                "news": bool(POLYGON_KEY),
                "telegram": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
                "prices": (not manual_active and not anomaly_active),
            },
        },
        "degraded": not stocks_ok,
        "degraded_reasons": reasons,
        "prices": {
            "provider": provider_effective or ("prev-close" if prev is not None else "unavailable"),
            "price": row_current,
            "prev_close": prev,
            "change_pct": change_pct,
        },
        "portfolio": {
            "symbol": WOLF,
            "qty": qty,
            "avg_cost": avg,
            "market_value": market_value,
            "pnl_abs": pnl_abs,
            "pnl_pct": pnl_pct,
            "rows": rows,
        },
        "kpis": {
            "nav": round(sum((r.get("mark_value") or 0.0) for r in rows) + cash_bal, 2),
            "cash": cash_bal,
            "pnl_abs": round((row_current - avg) * qty, 2),
            "pnl_pct": float(f"{(((row_current - avg) / avg) * 100.0) if avg > 0 else 0.0:.6f}"),
        },
        "gps": float(f"{gps:.2f}"),
        "confidence": int(conf),
        "reasons": reasons_ai,
        "analogs": analogs_ai,
        "mode": str(STATE.get("mode", "live")),
        "heatmap": {"tiles": [{"sym": WOLF, "symbol": WOLF, "gps": 7.2, "price": row_current}]},
        "heatmap_obj": {"tiles": [{"sym": WOLF, "symbol": WOLF, "gps": 7.2, "price": row_current}]},
        "movers": {
            "stocks": [
                {
                    "sym": WOLF,
                    "symbol": WOLF,
                    "price": row_current,
                    "change_pct": change_pct or 0.0,
                    "gps": 7.2,
                }
            ],
            "crypto": await _get_crypto_movers(),
        },
        "predictions": {
            "stocks": [],  # Populated by existing predict infrastructure
            "crypto": [],  # Will be populated if CRYPTO_ENABLED=1
        },
        "timestamp": now_ts,  # Set non-null timestamp for cockpit
        "outlook": {"risk": "neutral", "confidence": 0.70, "action": "HOLD"},
        "news": {"ticker": WOLF, "items": news.get("items", []), "note": note},
        "news_signal": news.get("news_signal")
        or {"score": None, "engine": "none", "items_scored": 0},
        "macro": macro,
        "news_relevant": ui_news[:10],
        "news_all": ui_news,
        "events_recent": _recent_events,
        "error_count": _error_count,
        "ui_prefs": {"tz": GHOST_TZ, "clock_24h": bool(GHOST_CLOCK_24H)},
        "flags": {
            "degraded": not stocks_ok,
            "any_stale": (price is None) or manual_active or anomaly_active,
            "market_open": bool(is_open),
            "using_prev_close": ((price is None and prev is not None) or anomaly_active),
            "price_anomaly": bool(anomaly_active),
            "corp_action_suspected": ("price:corp-action-suspected" in reasons),
        },
        "market": _build_market_status_with_indices(bool(is_open), int(next_open_ts)),
        "forecast_summary": fsum,
        "forecast": forecast_full,
        "actual_series": actual_series,
        "metrics": forecast_metrics,
        "two_line_overlay": two_line_data,
        "notes": (["news:polygon_key_missing"] if not POLYGON_KEY else []),
    }

    # === Populate predictions from in-memory store with classification ===
    try:
        stock_predictions = []
        crypto_predictions = []

        for sym, pred in _LATEST_PREDICTIONS.items():
            pred_data = {
                "symbol": pred["symbol"],
                "prediction_id": pred["prediction_id"],
                "run_at": int(pred["run_at"]),  # Unix timestamp in seconds
                "confidence": pred["confidence"] * 100,  # Convert to percentage
                "direction": pred["direction"],
                "horizon_h": pred["horizon_h"],
            }

            # Classify symbol into stocks/crypto/vip
            category = _classify_symbol_category(sym)
            if category == "stocks":
                stock_predictions.append(pred_data)
            elif category in ("crypto", "vip"):
                crypto_predictions.append(pred_data)

        # Update snapshot with classified predictions
        if stock_predictions:
            snapshot["predictions"]["stocks"] = stock_predictions
        if crypto_predictions:
            snapshot["predictions"]["crypto"] = crypto_predictions

        # Update timestamp from latest prediction if available
        if _LATEST_PREDICTIONS:
            latest_run_at = max(p["run_at"] for p in _LATEST_PREDICTIONS.values())
            snapshot["timestamp"] = int(latest_run_at)
    except Exception as e:
        LOGGER.warning(f"Failed to populate predictions from store: {e}")

    # === Ghost 2.x Enhancements ===
    # Add provider health, Ghost Score V2, and risk guard status to snapshot
    try:
        from core.crypto.vip_providers import get_vip_provider_health
        from core.metrics.ghost_score import compute_ghost_score_v2, get_current_risk_status
        from core.risk.risk_guard import get_risk_guard

        vip_health = get_vip_provider_health()

        # Use actual prediction LIMITS (auto_prediction_loop caps: 50 stocks + 25 crypto + VIP)
        _PREDICTION_CAP_STOCKS = int(os.getenv("AUTO_PREDICT_STOCK_LIMIT", "50"))
        _PREDICTION_CAP_CRYPTO = int(os.getenv("AUTO_PREDICT_CRYPTO_LIMIT", "25"))
        total_symbols = _PREDICTION_CAP_STOCKS + _PREDICTION_CAP_CRYPTO + len(VIP_COINS)

        # Live counts from _LATEST_PREDICTIONS (immediate) + fallback to counters
        _live_stocks_snap = sum(1 for s in _LATEST_PREDICTIONS if _classify_symbol_category(s) == "stocks")
        _live_crypto_snap = sum(1 for s in _LATEST_PREDICTIONS if _classify_symbol_category(s) in ("crypto", "vip"))
        symbols_with_data = max(_live_stocks_snap + _live_crypto_snap,
                               _LAST_MULTI_PREDICTION_COUNTS.get("stocks", 0) +
                               _LAST_MULTI_PREDICTION_COUNTS.get("crypto", 0) +
                               vip_health.get("symbols_with_data", 0))

        # Live avg confidence from actual predictions
        _confs_snap = [p["confidence"] for p in _LATEST_PREDICTIONS.values() if "confidence" in p]
        _live_avg_conf_snap = sum(_confs_snap) / len(_confs_snap) if _confs_snap else 0.65

        _live_pred_count_snap = max(len(_LATEST_PREDICTIONS), sum(_LAST_MULTI_PREDICTION_COUNTS.values()))

        ghost_score = compute_ghost_score_v2(
            data_quality={
                "symbols_with_data": symbols_with_data,
                "total_symbols": total_symbols,
                "provider_redundancy": 0.7,
                "avg_confidence": _live_avg_conf_snap
            },
            prediction_coverage={
                "predictions_generated": _live_pred_count_snap,
                "total_expected": total_symbols,
                "success_rate_estimate": 0.6
            },
            risk_status=get_current_risk_status()
        )

        risk_guard = get_risk_guard()

        # Add Ghost 2.x fields to snapshot
        snapshot["ghost_2x"] = {
            "ghost_score_v2": ghost_score,
            "vip_provider_health": vip_health,
            "risk_guard_status": risk_guard.get_status(),
            "provider_health_summary": {
                "crypto_providers_active": 3,
                "vip_symbols_with_data": vip_health.get("symbols_with_data", 0),
                "vip_symbols_total": len(VIP_COINS),
                "multi_symbol_counts": {
                    "stocks": max(_live_stocks_snap, _LAST_MULTI_PREDICTION_COUNTS.get("stocks", 0)),
                    "crypto": max(_live_crypto_snap, _LAST_MULTI_PREDICTION_COUNTS.get("crypto", 0)),
                }
            }
        }
    except Exception as e:
        LOGGER.warning(f"Could not load Ghost 2.x enhancements for cockpit: {e}")
        snapshot["ghost_2x"] = {"error": str(e)}
    # === End Ghost 2.x Enhancements ===\"

    # Inject crypto predictions if enabled
    try:
        if os.getenv("CRYPTO_ENABLED", "0") == "1":
            # Get recent crypto predictions from DB
            conn = sqlite3.connect(WOLF_SQLITE_PATH)
            c = conn.cursor()

            # Get latest prediction for default watchlist
            crypto_symbols = os.getenv("CRYPTO_SYMBOLS", "BTC,ETH,SOL,BNB").split(",")
            crypto_predictions = []

            for sym in crypto_symbols[:5]:  # Limit to 5 for UI
                sym = sym.strip().upper()
                if not sym:
                    continue

                # Get most recent prediction (within last hour)
                one_hour_ago = time.time() - 3600
                c.execute(
                    """
                    SELECT id, run_at, confidence, direction, volatility
                    FROM crypto_predictions
                    WHERE symbol = ? AND run_at > ?
                    ORDER BY run_at DESC
                    LIMIT 1
                """,
                    (sym, one_hour_ago),
                )

                row = c.fetchone()
                if row:
                    crypto_predictions.append(
                        {
                            "symbol": sym,
                            "prediction_id": row[0],
                            "run_at": int(row[1]),
                            "confidence": float(row[2]) * 100 if row[2] < 2 else float(row[2]),
                            "direction": row[3],
                            "volatility": float(row[4]) if row[4] else 0.0,
                        }
                    )

            conn.close()

            if crypto_predictions:
                snapshot["predictions"]["crypto"] = crypto_predictions
                snapshot["status"]["feeds"]["crypto"] = True

    except Exception as e:
        LOGGER.warning(f"Failed to inject crypto predictions: {e}")
        pass

    # Inject simulation enrichments
    try:
        if os.getenv("SIM_MODE", "0") == "1":
            from simulation_mode import get_mock_heatmap, get_mock_market_mood

            snapshot["heatmap_simulated"] = get_mock_heatmap()
            snapshot["market_outlook_simulated"] = get_mock_market_mood()
            snapshot["simulation"] = {
                "active": True,
                "tag": os.getenv("SIM_TAG", "ghost_ui_full_simulation_test_v2"),
            }
    except Exception:
        pass
    # Attach invested basis and more precise pnl_pct if available
    try:
        if invested and invested > 0 and snapshot.get("portfolio"):
            # IMPORTANT: avoid using rounded market_value for pnl_abs to prevent rounding drift.
            # Compute from raw row_current/avg/qty and then round once, matching verify_live expectations.
            pnl_abs_pos = (row_current - avg) * qty
            pnl_pct_pos = (pnl_abs_pos / invested) * 100.0 if invested > 0 else 0.0
            snapshot["portfolio"]["pnl_abs"] = round(pnl_abs_pos, 2)
            snapshot["portfolio"]["pnl_pct"] = float(f"{pnl_pct_pos:.6f}")
    except Exception:
        pass
    try:
        _add_event(
            "snapshot",
            "Cockpit snapshot served",
            {
                "as_o": now_ts,
                "price": (price if price is not None else row_current),
                "provider": (provider_effective or provider or "unavailable"),
            },
        )
    except Exception:
        pass
    # Append to AI memory ring
    try:
        _ai_memory_append(
            {
                "ts": now_ts,
                "price": display_price,
                "prev": prev,
                "qty": qty,
                "avg": avg,
                "news_score": ns_val,
                "features": feats,
                "label_next_move": 0,
                "advisory": "",
                "confidence": int(conf),
            }
        )
    except Exception:
        pass
    # Persist last-good snapshot atomically and serve
    LKG_PATH = os.getenv("COCKPIT_SNAPSHOT_FILE", "data/last_good_cockpit.json")
    LKG_MAX_AGE_S = int(os.getenv("COCKPIT_SNAPSHOT_MAX_AGE_S", "120"))
    try:
        # update snapshot freshness gauge
        try:
            if _G_SNAPSHOT_ASOF is not None:
                _G_SNAPSHOT_ASOF.set(now_ts)
        except Exception:
            pass
        # Persist atomically: write tmp then rename
        raw = json.dumps(snapshot, sort_keys=True).encode("utf-8")
        checksum = hashlib.sha256(raw).hexdigest()
        tmp_path = f"{LKG_PATH}.tmp"
        try:
            _ensure_dir_for_file(LKG_PATH)
            with open(tmp_path, "wb") as f:
                f.write(raw)
            os.replace(tmp_path, LKG_PATH)
        except Exception:
            pass
        # Prepare response with ETag
        resp = JSONResponse(snapshot)
        resp.headers["ETag"] = checksum
        resp.headers["Cache-Control"] = "public, max-age=10"
        try:
            _add_event(
                "snapshot",
                "Cockpit snapshot served",
                {
                    "as_o": now_ts,
                    "price": price,
                    "provider": provider or "unavailable",
                },
            )
        except Exception:
            pass
        return resp
    except Exception:
        # Fallback to last-good snapshot on any unexpected failure
        try:
            if os.path.exists(LKG_PATH):
                with open(LKG_PATH, "rb") as f:
                    raw = f.read()
                cached = json.loads(raw.decode("utf-8"))
                # Mark degraded and stale flags
                cached["flags"] = {
                    **(cached.get("flags") or {}),
                    "degraded": True,
                    "any_stale": True,
                }
                cached["status"] = {**(cached.get("status") or {}), "ok": False}
                sid = cached.get("snapshot_id", "lkg")
                if isinstance(sid, str) and not sid.endswith("-fallback"):
                    cached["snapshot_id"] = f"{sid}-fallback"
                # Age guard
                ts = int(cached.get("as_o") or 0)
                too_old = (time.time() - ts) > max(1, LKG_MAX_AGE_S)
                reasons = list(cached.get("degraded_reasons") or [])
                if too_old and "snapshot:stale" not in reasons:
                    reasons.append("snapshot:stale")
                cached["degraded"] = True
                cached["degraded_reasons"] = reasons
                checksum = hashlib.sha256(
                    json.dumps(cached, sort_keys=True).encode("utf-8")
                ).hexdigest()
                resp = JSONResponse(cached)
                resp.headers["ETag"] = checksum
                resp.headers["Cache-Control"] = "public, max-age=5"
                return resp
        except Exception:
            pass
        # As a last resort, return the computed snapshot (already built) without persistence
        return JSONResponse(snapshot)


@router.get("/mobile")
async def mobile_cockpit():
    """
    Serve Ghost mobile cockpit (simplified mobile UI).
    Shows goals, VIP coins, pre-market predictions, and recent alerts.
    """
    from fastapi.templating import Jinja2Templates

    templates = Jinja2Templates(directory="templates")

    class MockRequest:
        def __init__(self):
            self.headers = {}
            self.path_params = {}

    try:
        return templates.TemplateResponse(
            "cockpit_mobile.html",
            {"request": MockRequest()}
        )
    except Exception as e:
        LOGGER.error(f"Mobile cockpit failed: {e}")
        return HTMLResponse(
            content="""
            <html><head><title>Ghost Mobile</title></head>
            <body><h1>Ghost Mobile</h1>
            <p>Mobile dashboard temporarily unavailable</p></body></html>
            """,
            status_code=500
        )


@router.get("/cockpit_v2", include_in_schema=False)
async def cockpit_v2_page(request: Request):
    """Legacy V2 route - redirects to V3 cockpit."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/cockpit", status_code=301)


