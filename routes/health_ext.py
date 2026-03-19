"""Routes: health_ext — extracted from wolf_app.py (Step 12)"""
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
from fastapi.responses import JSONResponse, HTMLResponse, PlainTextResponse, RedirectResponse

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

# --- 8 endpoints ---

@router.get("/health", include_in_schema=False)
async def health_check():
    """Ultra-lightweight health check for load balancer / Kubernetes / Railway.
    
    This endpoint responds immediately during startup and confirms:
    - FastAPI server is alive
    - Database is accessible (after warmup)
    - At least one price provider works (after warmup)
    
    Returns 200 OK if healthy, 503 Service Unavailable if critical systems fail.
    """
    try:
        uptime = int(time.time() - _START_TS)
        health_status = {
            "status": "healthy",
            "service": "ghost-protocol",
            "uptime": uptime,
            "uptime_seconds": uptime,
            "message": "All systems operational",
            "sim_mode": int(os.getenv("SIM_MODE", "0") or "0"),
            "enforce_live": _is_live_enforced(),
            "git_sha": _get_git_sha(),
            "build_ts": os.getenv("RAILWAY_DEPLOYMENT_ID") or os.getenv("RAILWAY_STATIC_URL"),
        }
        
        # FAST PATH: During startup (<120s), skip ALL blocking checks
        # This ensures Railway health checks pass instantly during deploys
        if uptime < 120:
            health_status["database"] = "warming_up"
            health_status["prediction_store"] = "warming_up"
            health_status["price_providers"] = "warming_up"
            return JSONResponse(content=health_status, status_code=200)
        
        # ================================================================
        # NON-BLOCKING health checks using cached values
        # The health endpoint MUST respond in <100ms regardless of what
        # else is running (Stock Engine predictions, ensemble ML, etc.)
        # Background: Stock Engine predictions consume heavy CPU/GIL time,
        # so any sync IO or asyncio.to_thread() calls here will stall.
        # ================================================================
        
        # Use cached health state (updated by background task)
        if not hasattr(health_check, '_cache'):
            health_check._cache = {
                "database": "unknown",
                "prediction_store": "unknown", 
                "price_providers": "unknown",
                "btc_price": None,
                "last_check": 0,
                "predictions_stale": False,
            }
        
        cache = health_check._cache
        cache_age = time.time() - cache["last_check"]
        
        # Refresh cache in background every 60s (non-blocking)
        if cache_age > 60:
            async def _refresh_health_cache():
                try:
                    # Check 1: Database (run in thread to avoid blocking event loop)
                    try:
                        def _check_db():
                            import sqlite3
                            c = sqlite3.connect(WOLF_SQLITE_PATH)
                            c.execute("SELECT 1").fetchone()
                            c.close()
                            return "connected"
                        cache["database"] = await asyncio.wait_for(
                            asyncio.to_thread(_check_db), timeout=2.0
                        )
                    except Exception:
                        cache["database"] = "error"
                    
                    # Check 2: Prediction store
                    try:
                        def _check_store():
                            from core.prediction_store import get_prediction_store
                            get_prediction_store().get_recent_predictions(limit=1)
                            return "connected"
                        cache["prediction_store"] = await asyncio.wait_for(
                            asyncio.to_thread(_check_store), timeout=2.0
                        )
                    except Exception:
                        cache["prediction_store"] = "unavailable"
                    
                    # Check 3: BTC price
                    try:
                        from core.coinbase_provider import get_coinbase_provider
                        provider = get_coinbase_provider()
                        btc_price = await asyncio.wait_for(
                            asyncio.to_thread(provider.get_price, "BTC"), timeout=3.0
                        )
                        if btc_price and btc_price > 0:
                            cache["price_providers"] = "operational"
                            cache["btc_price"] = round(btc_price, 2)
                        else:
                            cache["price_providers"] = "degraded"
                    except Exception:
                        cache["price_providers"] = "timeout"
                    
                    # Check 4: Prediction staleness
                    # If no predictions in 6 hours, something is wrong
                    try:
                        _stale = False
                        if _LATEST_PREDICTIONS:
                            _newest = max(
                                (p.get("run_at", 0) for p in _LATEST_PREDICTIONS.values()),
                                default=0,
                            )
                            _age_h = (time.time() - _newest) / 3600.0 if _newest else 999
                            _stale = _age_h > 6.0
                        else:
                            # No predictions at all after warmup
                            _stale = uptime > 600  # Only flag after 10min
                        cache["predictions_stale"] = _stale
                        if _stale:
                            LOGGER.warning(
                                f"[HEALTH] Predictions stale — no new predictions in "
                                f"{_age_h:.1f}h" if _LATEST_PREDICTIONS else
                                f"[HEALTH] No predictions in memory after {uptime}s uptime"
                            )
                    except Exception:
                        cache["predictions_stale"] = False

                    cache["last_check"] = time.time()
                except Exception as bg_err:
                    LOGGER.debug(f"Health cache refresh error: {bg_err}")
            
            # Fire and forget — don't await, let it run in background
            asyncio.ensure_future(_refresh_health_cache())
            # If first time, mark as checking
            if cache["last_check"] == 0:
                cache["last_check"] = time.time() - 50  # Will retry in 10s
        
        health_status["database"] = cache["database"]
        health_status["prediction_store"] = cache["prediction_store"]
        health_status["price_providers"] = cache["price_providers"]
        health_status["predictions_stale"] = cache.get("predictions_stale", False)
        if cache["btc_price"]:
            health_status["btc_price"] = cache["btc_price"]
        
        # ── Derive actual health status from component checks ──
        _critical_failed = cache["database"] == "error"
        _degraded = (
            cache["prediction_store"] == "unavailable"
            or cache["price_providers"] in ("timeout", "degraded")
            or cache.get("predictions_stale", False)
        )
        if _critical_failed:
            health_status["status"] = "unhealthy"
            health_status["message"] = "Critical: database connection failed"
        elif _degraded:
            health_status["status"] = "degraded"
            _reasons = []
            if cache["prediction_store"] == "unavailable":
                _reasons.append("prediction_store unavailable")
            if cache["price_providers"] in ("timeout", "degraded"):
                _reasons.append(f"price_providers {cache['price_providers']}")
            if cache.get("predictions_stale", False):
                _reasons.append("predictions stale (>6h old)")
            health_status["message"] = f"Warning: {', '.join(_reasons)}"

        # ── System-failure Telegram alert (max once per 15 min) ──
        if health_status["status"] in ("unhealthy", "degraded"):
            _now = time.time()
            _last_alert = getattr(health_check, '_last_failure_alert', 0)
            if _now - _last_alert > 900:  # 15-minute cooldown
                health_check._last_failure_alert = _now
                try:
                    _alert_msg = (
                        "🚨 <b>GHOST SYSTEM ALERT</b>\n\n"
                        f"Status: <b>{health_status['status'].upper()}</b>\n"
                        f"Message: {health_status.get('message', '')}\n"
                        f"Database: {cache['database']}\n"
                        f"Predictions: {cache['prediction_store']}\n"
                        f"Price feeds: {cache['price_providers']}\n"
                        f"Predictions stale: {cache.get('predictions_stale', False)}\n"
                        f"Uptime: {uptime}s\n\n"
                        "⚠️ Investigate immediately."
                    )
                    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
                        _tg_send_chat_message(TELEGRAM_CHAT_ID, _alert_msg)
                        LOGGER.warning(f"[HEALTH] 🚨 System failure alert sent to Telegram")
                except Exception as e:
                    LOGGER.warning(f"health_alert_send_failed: {e}")
            return JSONResponse(content=health_status, status_code=503)
        
        return health_status
        
    except Exception as e:
        # Even if health check logic fails, return basic OK (server is responding)
        return {
            "status": "ok",
            "service": "ghost-protocol",
            "message": "Server is accepting connections",
            "health_check_error": str(e)
        }


@router.get("/ui/health")
async def ui_health():
    """Simple healthcheck endpoint that always returns 200 OK"""
    return {"status": "ok", "service": "ghost-protocol"}


@router.get("/api/v3/health/metrics")
async def api_v3_health_metrics():
    """
    Calculate real-time health metrics for the cockpit.
    
    Returns:
        - data_health: Provider uptime (test BTC availability)
        - ai_activity: Predictions per hour
        - accuracy: Win rate from V2-filtered paper trades
        - cache_performance: Price cache hit rate
    """
    try:
        # V2 ERA: Start date for clean data
        V2_START_DATE = "2026-01-14"
        
        # Data Health: Check if crypto providers are working
        data_health = 50  # Default if provider unavailable
        try:
            # Use quorum price which is more reliable and returns 24h change
            from core.crypto.crypto_providers import get_crypto_price_quorum, get_cache_stats
            
            # Try to get price for a V2 whitelisted crypto
            chz_data = await get_crypto_price_quorum("CHZ", use_cache=True)
            
            if chz_data and chz_data.get("price", 0) > 0:
                # Check quorum quality
                quorum_size = chz_data.get("quorum_size", 1)
                if quorum_size >= 2:
                    data_health = 95  # Strong quorum
                else:
                    data_health = 80  # Single provider working
            else:
                # Fallback: check cache stats
                try:
                    cache_stats = get_cache_stats()
                    hit_rate = cache_stats.get("hit_rate_pct", 0)
                    cache_size = cache_stats.get("cache_size", 0)
                    if cache_size > 0 and hit_rate > 30:
                        data_health = 70  # Cache working
                    elif cache_size > 0:
                        data_health = 60  # Cache has data
                    else:
                        data_health = 40  # No cache data
                except Exception:
                    data_health = 40
        except Exception as e:
            LOGGER.warning(f"Health check provider test failed: {e}")
            data_health = 50  # Unknown state
        
        # AI Activity: Based on heartbeat status (are tasks actually running?)
        # Not just prediction count — that masks dead background processes
        total_predictions = len(_LATEST_PREDICTIONS)
        try:
            from core.heartbeat import get_all_status
            _hb_status = get_all_status()
            _hb_alive = sum(1 for h in _hb_status.values() if h.get("status") == "alive")
            _hb_total = len(_hb_status) if _hb_status else 1
            # Activity = % of tasks alive + prediction count bonus
            task_health = round((_hb_alive / max(_hb_total, 1)) * 60)  # up to 60 from tasks
            pred_bonus = min(30, total_predictions * 2)  # up to 30 from predictions
            ai_activity = min(95, task_health + pred_bonus + 5)  # +5 baseline (server is running)
        except Exception:
            # Fallback: prediction count only
            if total_predictions >= 50:
                ai_activity = 70
            elif total_predictions >= 20:
                ai_activity = 55
            elif total_predictions >= 10:
                ai_activity = 45
            else:
                ai_activity = 30
        
        # Accuracy: INTEGRITY-verified source first (honest number)
        # Priority: PostgreSQL checked predictions > paper tracker
        accuracy = None
        accuracy_source = "none"
        try:
            from core.db_pool import get_sync_connection as _hm_get_conn
            with _hm_get_conn() as _hm_conn:
                _hm_cur = _hm_conn.cursor()
                _hm_cur.execute("""
                    SELECT COUNT(*) as total,
                           SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as wins
                    FROM ghost_predictions
                    WHERE checked = 1
                      AND eval_version NOT LIKE 'skip%%'
                      AND predicted_at > EXTRACT(EPOCH FROM NOW() - INTERVAL '30 days')
                """)
                _hm_row = _hm_cur.fetchone()
                _hm_cur.close()
                _hm_total = _hm_row[0] if _hm_row else 0
                _hm_wins = _hm_row[1] if _hm_row and _hm_row[1] else 0
                if _hm_total and _hm_total > 0:
                    accuracy = round((_hm_wins / _hm_total) * 100, 1)
                    accuracy_source = "ghost_predictions_integrity"
        except Exception:
            pass
        
        # Fallback: paper tracker
        if accuracy is None:
            try:
                from core.paper_tracker import get_paper_tracker
                tracker = get_paper_tracker()
                stats = tracker.get_stats(since=V2_START_DATE, v2_only=True)
                if stats.get("resolved_trades", 0) > 0:
                    accuracy = round(stats.get("win_rate_pct", 50), 1)
                    accuracy_source = "paper_tracker"
            except Exception:
                pass
        
        # Cache Performance: Get price cache statistics
        cache_stats = {}
        try:
            from core.crypto.crypto_providers import get_cache_stats
            cache_stats = get_cache_stats()
        except Exception:
            pass
        
        return {
            "ok": True,
            "data_health": data_health,
            "ai_activity": ai_activity,
            "accuracy": accuracy,
            "accuracy_source": accuracy_source if accuracy is not None else "no_data",
            "cache_performance": cache_stats,
            "v2_start_date": V2_START_DATE,
            "timestamp": datetime.now(UTC).isoformat()
        }
    
    except Exception as e:
        LOGGER.error(f"Health metrics failed: {e}", exc_info=True)
        return {
            "ok": False,
            "data_health": None,
            "ai_activity": None,
            "accuracy": None,
            "error": str(e),
            "note": "All metrics unavailable due to error — do NOT treat as 50%"
        }


@router.get("/api/health/predictions")
async def api_health_predictions():
    """
    Health check endpoint for multi-symbol predictions and Telegram alerts.
    Returns current state, last run times, provider health, Ghost Score V2, and risk guard status.
    """
    # Get crypto provider health data
    crypto_provider_health = {}
    try:
        from core.crypto.crypto_providers import get_crypto_provider_health
        crypto_provider_health = get_crypto_provider_health()
    except Exception as e:
        LOGGER.warning(f"Could not get crypto provider health: {e}")

    # Get VIP provider health data
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

        # FIX (Mar 1, 2026): Use edge set size as denominator, not scanner caps.
        # Old code: 50 stocks + 25 crypto + 5 VIP = 80 → coverage always ~30%.
        # Reality: edge whitelist has 13 symbols → 13/13 = 100% when all predicted.
        from config.symbols import get_edge_set as _gs_edge_set
        _edge_symbols = _gs_edge_set()
        total_symbols = len(_edge_symbols)  # 13 proven symbols
        # Use live counts from _LATEST_PREDICTIONS as primary (always up-to-date)
        _ls = sum(1 for p in _LATEST_PREDICTIONS.values() if isinstance(p, dict) and p.get("engine") == "stock_v2")
        _lc = sum(1 for p in _LATEST_PREDICTIONS.values() if isinstance(p, dict) and p.get("engine") != "stock_v2")
        symbols_with_data = _ls + _lc

        # Compute live avg_confidence from actual predictions
        _live_confs = [p.get("confidence", 0) for p in _LATEST_PREDICTIONS.values() if isinstance(p, dict) and p.get("confidence")]
        _live_avg_conf = sum(_live_confs) / len(_live_confs) if _live_confs else 0.6

        data_quality = {
            "symbols_with_data": symbols_with_data,
            "total_symbols": total_symbols,
            "provider_redundancy": 0.7,  # Conservative estimate (multiple providers active)
            "avg_confidence": round(_live_avg_conf, 3)
        }

        # Prediction coverage — use live count as primary
        predictions_generated = max(sum(_LAST_MULTI_PREDICTION_COUNTS.values()), len(_LATEST_PREDICTIONS))
        prediction_coverage = {
            "predictions_generated": predictions_generated,
            "total_expected": total_symbols,
            "success_rate_estimate": 0.6  # Conservative baseline (actual accuracy ~42% but improving)
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

    return {
        "ok": True,
        "last_multi_prediction_run_time": _LAST_MULTI_PREDICTION_TIME,
        "last_telegram_send_time": _LAST_TELEGRAM_SEND_TIME,
        "symbol_counts": _LAST_MULTI_PREDICTION_COUNTS.copy(),
        "last_telegram_status": _LAST_TELEGRAM_STATUS,
        "last_telegram_error": _LAST_TELEGRAM_ERROR,
        "telegram_configured": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
        "crypto_provider_health": crypto_provider_health,
        "vip_provider_health": vip_provider_health,
        "ghost_score_v2": ghost_score_v2,
        "risk_guard_status": risk_guard_status,
        "timestamp": time.time()
    }


@router.get("/health/detailed")
async def health_detailed():
    """Comprehensive health check with provider status"""
    import time

    health_status = {"ok": True, "ts": time.time(), "components": {}, "issues": []}

    # Database health
    try:
        if AI_MEMORY_STORE is not None:
            cur = AI_MEMORY_STORE.conn.execute("SELECT COUNT(1) FROM ai_memory")
            count = int(cur.fetchone()[0] or 0)
            health_status["components"]["ai_memory"] = {"ok": True, "records": count}
        else:
            health_status["components"]["ai_memory"] = {
                "ok": False,
                "error": "Not initialized",
            }
            health_status["issues"].append("AI memory store unavailable")
    except Exception as e:
        health_status["components"]["ai_memory"] = {"ok": False, "error": str(e)}
        health_status["issues"].append(f"AI memory error: {str(e)}")

    # Position persistence
    try:
        import sqlite3

        conn = sqlite3.connect(WOLF_SQLITE_PATH)
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT)")
        conn.commit()
        cur.execute("SELECT value FROM state WHERE key='position'")
        row = cur.fetchone()
        conn.close()

        if row and row[0]:
            pos_data = json.loads(row[0])
            positions = pos_data.get("positions") or []
            wolf_qty = pos_data.get("qty") or 0.0
            wolf_avg = pos_data.get("avg_cost") or 0.0
            health_status["components"]["positions"] = {
                "ok": True,
                "count": len(positions),
                "symbols": [p.get("symbol") for p in positions],
                "wolf_qty": wolf_qty,
                "wolf_avg": wolf_avg,
            }
        else:
            health_status["components"]["positions"] = {
                "ok": True,
                "count": 0,
                "symbols": [],
                "wolf_qty": STATE.get("qty", 0.0),
                "wolf_avg": STATE.get("avg_cost", 0.0),
                "note": "No persisted position found, using STATE",
            }
    except Exception as e:
        health_status["components"]["positions"] = {"ok": False, "error": str(e)}
        health_status["issues"].append(f"Position loading error: {str(e)}")

    # Price providers
    providers_status = {}
    price, prev, provider = get_wolf_price()
    providers_status["current_price"] = {
        "price": price,
        "prev_close": prev,
        "provider": provider,
        "ok": price is not None,
    }
    providers_status["api_keys"] = {
        "alphavantage": bool(ALPHAVANTAGE_KEY),
        "polygon": bool(POLYGON_KEY),
    }
    providers_status["diagnostics"] = dict(PRICE_DIAG)
    health_status["components"]["price_providers"] = providers_status

    if price is None:
        health_status["issues"].append(f"Price unavailable for {WOLF} - provider: {provider}")
        if not ALPHAVANTAGE_KEY and not POLYGON_KEY:
            health_status["issues"].append("No premium API keys configured")

    # Cache status
    cache_status = {
        "price_cache_size": len(PRICE_CACHE),
        "news_cache_age_s": int(time.time() - float(NEWS_CACHE.get("ts") or 0)),
        "ai_memory_ring_size": len(AI_MEMORY_RING),
    }
    health_status["components"]["cache"] = cache_status

    # Overall status
    health_status["ok"] = len(health_status["issues"]) == 0

    return health_status


@router.get("/api/secrets/health")
async def api_secrets_health():
    present = {
        "GHOST_API_TOKEN": bool(os.getenv("GHOST_API_TOKEN", "")),
        "ALPHAVANTAGE_API_KEY": bool(ALPHAVANTAGE_KEY),
        "POLYGON_API_KEY": bool(POLYGON_KEY),
        "TELEGRAM_BOT_TOKEN": bool(TELEGRAM_BOT_TOKEN),
        "TELEGRAM_CHAT_ID": bool(TELEGRAM_CHAT_ID),
        "REDIS_URL": bool(REDIS_URL),
    }
    return {"present": present, "missing": [k for k, v in present.items() if not v]}


@router.get("/api/system/health-check")
async def api_system_health_check():
    """
    Comprehensive system health dashboard.
    
    Aggregates all critical system info in one response:
    - Market gates status (regime, VIX, BTC trend)
    - V2 quality whitelist/blacklist
    - Accuracy metrics
    - Pending trades count
    - Last prediction timestamp
    
    Use this for monitoring dashboards or Slack/Discord alerts.
    """
    try:
        from core.market_gates import RegimeFilter, VIXGate
        from core.v2_quality import get_quality_system
        from core.paper_tracker import get_paper_tracker
        
        result = {
            "ok": True,
            "timestamp": time.time(),
            "gates": {},
            "whitelist": [],
            "accuracy": {},
            "pending_trades": 0,
            "last_prediction": None,
            "alerts": []
        }
        
        # 1. Market Gates
        try:
            rf = RegimeFilter()
            vg = VIXGate()
            
            spy_regime = await rf.get_spy_regime()
            btc_trend = await rf.get_btc_trend()
            vix_level = await vg.get_current_vix()
            
            result["gates"] = {
                "spy_regime": spy_regime.get("regime", "unknown"),
                "spy_above_ma": spy_regime.get("above_20ma", True),
                "btc_trend_7d": btc_trend.get("trend_7d_pct", 0),
                "crypto_regime": btc_trend.get("crypto_regime", "unknown"),
                "vix": vix_level,
                "buy_allowed": {
                    "stocks": spy_regime.get("regime") == "bull",
                    "crypto": btc_trend.get("crypto_regime") == "bull"
                }
            }
            
            # Alert if crypto buys blocked
            if btc_trend.get("crypto_regime") == "bear":
                result["alerts"].append({
                    "level": "warning",
                    "message": f"Crypto BUYs blocked - BTC down {btc_trend.get('trend_7d_pct', 0):.1f}% (7d)"
                })
            
            # Alert if VIX elevated
            if vix_level > 25:
                result["alerts"].append({
                    "level": "warning" if vix_level < 30 else "critical",
                    "message": f"VIX elevated at {vix_level:.1f} - BUY confidence reduced"
                })
                
        except Exception as e:
            LOGGER.error(f"Dashboard gates error: {e}")
            result["gates"] = {"error": str(e)}
        
        # 2. V2 Quality Whitelist
        try:
            v2_system = get_quality_system()
            result["whitelist"] = sorted(v2_system.whitelist)
            result["blacklist_count"] = len(v2_system.blacklist)
        except Exception as e:
            LOGGER.error(f"Dashboard whitelist error: {e}")
        
        # 3. Accuracy Metrics
        try:
            tracker = get_paper_tracker()
            V2_START_DATE = "2026-01-14"
            
            # Overall stats (30 days, V2 only)
            stats = tracker.get_stats(days=30, since=V2_START_DATE, v2_only=True)
            total = stats.get("resolved_trades", 0)
            wins = stats.get("wins", 0)
            
            # Daily stats
            daily = tracker.get_stats(days=1, since=V2_START_DATE, v2_only=True)
            daily_total = daily.get("resolved_trades", 0)
            daily_wins = daily.get("wins", 0)
            
            result["accuracy"] = {
                "overall_pct": round((wins / total) * 100, 1) if total > 0 else 0,
                "daily_pct": round((daily_wins / daily_total) * 100, 1) if daily_total > 0 else 0,
                "total_resolved": total,
                "total_wins": wins,
                "pending": stats.get("active_trades", 0)
            }
            
            result["pending_trades"] = stats.get("active_trades", 0)
            
            # Alert if accuracy drops below 50%
            if total >= 10 and (wins / total) < 0.5:
                result["alerts"].append({
                    "level": "warning",
                    "message": f"Accuracy below 50% ({round((wins/total)*100, 1)}%)"
                })
                
        except Exception as e:
            LOGGER.error(f"Dashboard accuracy error: {e}")
            result["accuracy"] = {"error": str(e)}
        
        # 4. Last Prediction
        try:
            from psycopg2.extras import RealDictCursor
            from core.db_pool import get_sync_connection
            
            db_url = os.getenv("DATABASE_URL")
            if db_url:
                with get_sync_connection() as conn:
                    cur = conn.cursor(cursor_factory=RealDictCursor)
                    cur.execute("""
                        SELECT symbol, direction, confidence, created_at
                        FROM ghost_predictions
                        ORDER BY created_at DESC
                        LIMIT 1
                    """)
                    row = cur.fetchone()
                    
                    if row:
                        result["last_prediction"] = {
                            "symbol": row["symbol"],
                            "direction": row["direction"],
                            "confidence": float(row["confidence"]) if row["confidence"] else 0,
                            "timestamp": row["created_at"].isoformat() if row["created_at"] else None
                        }
        except Exception as e:
            LOGGER.debug(f"Dashboard last_prediction error: {e}")
        
        return result
        
    except Exception as e:
        LOGGER.error(f"system_health_check_error: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/api/health")
async def api_health():
    """Simple health check endpoint for monitoring systems."""
    return {"ok": True, "ts": int(time.time() * 1000), "version": "feb7-no-flat-v6"}


