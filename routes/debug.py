"""Routes: debug — extracted from wolf_app.py (Step 12)"""
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

try:
    from wolf_helpers import (
        AUTH_DEP, SECURITY_SCHEME, WOLF, WOLF_SQLITE_PATH,
        _is_truthy, _json500, with_cap,
        AlertTemplateBody, AlertToggle, AlertConfigBody,
        RuntimeConfigBody, ControlBody, ModeBody, TrainBody,
        AgentControlBody, CashBody, PositionAddBody, PositionsImportBody,
        WatchlistImportBody, TradeRequest, PredFeedbackBody,
        AddPositionBody, OrderPlaceBody,
        _PredictRunBody, _RecordPriceBody, _ScoreBody, _BacktestBody,
        ChatRequest, AiDecision, TelegramUpdate,
    )
    from fastapi.security import HTTPAuthorizationCredentials
except Exception as _wh_e:
    import logging as _l
    _l.getLogger("ghost").warning(f"wolf_helpers import partial: {_wh_e}")
    AUTH_DEP = None
    WOLF = "WOLF"
    WOLF_SQLITE_PATH = "data/wolf.db"


router = APIRouter()
LOGGER = logging.getLogger("ghost")

# --- 70 endpoints ---

@router.get("/openapi.json", include_in_schema=False)
async def _openapi_compat():
    return RedirectResponse(url="/api/openapi.json", status_code=307)


@router.get("/debug/routes", include_in_schema=False)
async def _debug_routes():
    try:
        from fastapi.routing import APIRoute

        return {
            "routes": [
                {
                    "path": getattr(r, "path", None),
                    "name": getattr(r, "name", None),
                    "methods": list(getattr(r, "methods", []) or []),
                }
                for r in APP.routes
                if isinstance(r, APIRoute)
            ]
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/debug/pool-status", include_in_schema=False)
async def _debug_pool_status():
    """Diagnostic: show sync connection pool health."""
    try:
        from core.db_pool import get_sync_pool_status
        return get_sync_pool_status()
    except Exception as e:
        return {"error": str(e)}


@router.get("/api/v3/debug/features/{symbol}")
async def api_v3_debug_features(symbol: str):
    """
    DEBUG: Show sentiment and world context feature values used in predictions.
    
    This endpoint tests the fixes for:
    - Sentiment engine (should return real data, not 0.0)
    - World context (should return real SPY/VIX, not NULL)
    
    Returns:
        {
            "ok": True,
            "symbol": "ZEC",
            "sentiment": {
                "signals": [
                    {"name": "news_sentiment_score", "value": 0.72, "source": "ghost_news_brain"},
                    {"name": "social_sentiment", "value": 0.0, "source": "rss_fallback"}
                ],
                "working": True
            },
            "world_context": {
                "spy_price": 598.45,
                "vix_level": 14.23,
                "market_regime": "normal",
                "working": True
            },
            "orchestrator_health": {
                "total_pillars": 6,
                "healthy_pillars": 6,
                "pillar_status": {...}
            }
        }
    """
    try:
        symbol = symbol.upper().strip()
        
        # Test 1: Sentiment Engine
        from core.data_pillars.sentiment_engine import SentimentEngine
        sentiment_engine = SentimentEngine()
        sentiment_result = sentiment_engine.get_signals(symbol)
        
        sentiment_data = {
            "pillar_name": sentiment_result.pillar_name,
            "signals": [
                {
                    "name": signal.name,
                    "value": signal.value,
                    "source": signal.source
                }
                for signal in sentiment_result.signals
            ],
            "errors": sentiment_result.errors,
            "working": len(sentiment_result.signals) > 0,
            "has_real_data": any(s.value != 0.0 for s in sentiment_result.signals)
        }
        
        # Test 2: World Context
        from core.world_context import get_world_context
        world_context = get_world_context()
        
        spy_data = world_context.get("spy", {})
        vix_data = world_context.get("vix", {})
        mood_data = world_context.get("market_mood", {})
        
        world_data = {
            "spy_price": spy_data.get("price"),
            "spy_change_pct": spy_data.get("change_pct"),
            "spy_provider": spy_data.get("provider"),
            "vix_level": vix_data.get("level"),
            "vix_change": vix_data.get("change"),
            "vix_status": vix_data.get("status"),
            "market_sentiment": mood_data.get("sentiment"),
            "market_score": mood_data.get("score"),
            "working": (spy_data.get("price") is not None and spy_data.get("price") > 0 and
                       vix_data.get("level") is not None and vix_data.get("level") > 0)
        }
        
        # Test 3: Feature Orchestrator Health
        from core.data_pillars.feature_orchestrator import FeatureOrchestrator
        orchestrator = FeatureOrchestrator()
        health = orchestrator.health_check()
        
        # Extract summary
        summary = health.get("summary", {})
        healthy_count = summary.get("healthy", 0)
        total_count = summary.get("total", 6)
        
        return {
            "ok": True,
            "symbol": symbol,
            "sentiment": sentiment_data,
            "world_context": world_data,
            "orchestrator_health": {
                "ok": health.get("ok", False),
                "healthy": healthy_count,
                "total": total_count,
                "pillars": health.get("pillars", {})
            },
            "verdict": {
                "sentiment_engine_working": sentiment_data["working"],
                "world_context_working": world_data["working"],
                "all_pillars_healthy": healthy_count >= 5
            }
        }
    except Exception as e:
        LOGGER.error(f"Feature debug failed for {symbol}: {e}", exc_info=True)
        return {
            "ok": False,
            "symbol": symbol,
            "error": str(e)
        }


@router.get("/api/dev/features/diagnostic")
async def api_features_diagnostic(symbol: str):
    """
    DEVELOPER DIAGNOSTIC: Feature extraction health check.
    
    Shows which features are being extracted successfully and which are failing.
    Useful for debugging the prediction pipeline.
    
    Args:
        symbol: Stock/crypto ticker (e.g., MSFT, BTC)
    
    Returns:
        {
            "ok": True,
            "symbol": "MSFT",
            "feature_count": 40,
            "available_count": 35,
            "unavailable_count": 5,
            "availability_pct": 87.5,
            "feature_availability": {
                "price_engine": "2/8",
                "technical_engine": "12/15",
                "volume_engine": "4/5",
                "sentiment_engine": "2/3",
                "world_context_engine": "3/4",
                "flow_engine": "0/4"
            },
            "available_features": {
                "PRICE": 185.25,
                "RSI_14": 67.5,
                "MACD_HISTOGRAM": 0.45,
                ...
            },
            "missing_features": [
                "BID_ASK_SPREAD",
                "SMA_200",
                ...
            ],
            "errors": [
                "Insufficient historical data for MSFT",
                ...
            ],
            "execution_time_ms": 234.5
        }
    """
    try:
        symbol = symbol.upper().strip()
        
        # Get feature orchestrator
        from core.data_pillars.feature_orchestrator import get_feature_orchestrator
        
        orchestrator = get_feature_orchestrator()
        feature_data = orchestrator.get_all_features(symbol, period=90)
        
        # Extract available vs unavailable features
        features = feature_data.get("features", {})
        available_features = {k: v for k, v in features.items() if v is not None}
        missing_features = [k for k, v in features.items() if v is None]
        
        # Calculate availability percentage
        feature_count = feature_data.get("feature_count", 0)
        available_count = feature_data.get("available_count", 0)
        availability_pct = (available_count / feature_count * 100) if feature_count > 0 else 0.0
        
        return {
            "ok": True,
            "symbol": symbol,
            "timestamp": feature_data.get("timestamp", time.time()),
            "feature_count": feature_count,
            "available_count": available_count,
            "unavailable_count": feature_data.get("unavailable_count", 0),
            "availability_pct": round(availability_pct, 1),
            "feature_availability": feature_data.get("feature_availability", {}),
            "available_features": available_features,
            "missing_features": missing_features,
            "errors": feature_data.get("errors", []),
            "execution_time_ms": feature_data.get("execution_time_ms", 0.0),
        }
        
    except Exception as e:
        LOGGER.error(f"Feature diagnostic failed for {symbol}: {e}", exc_info=True)
        raise HTTPException(500, f"Diagnostic failed: {str(e)}")


@router.get("/api/xray/{symbol}")
async def api_xray_symbol(symbol: str):
    """
    X-RAY: See EXACTLY what XGBoost receives for a prediction.
    
    This runs a full prediction cycle and logs:
    1. Every feature value the model sees (53 features)
    2. Which are REAL data vs NEUTRAL DEFAULTS
    3. Raw XGBoost output (prob_up, prob_down, spread)
    4. Whether it hits the hold zone
    5. Feature quality score
    
    This is the truth — no guessing. Call this to verify the pipeline works.
    """
    import time as _time
    import math as _xray_math
    start = _time.time()
    symbol = symbol.upper().strip()

    def _sanitize_for_json(obj):
        """Recursively replace NaN/Inf floats with None so json.dumps(allow_nan=False) won't crash."""
        if isinstance(obj, float):
            if _xray_math.isnan(obj) or _xray_math.isinf(obj):
                return None
            return obj
        if isinstance(obj, dict):
            return {k: _sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_sanitize_for_json(v) for v in obj]
        return obj

    try:
        # === STEP 1: Run feature orchestrator (same as real prediction) ===
        from core.data_pillars.feature_orchestrator import get_feature_orchestrator
        orchestrator = get_feature_orchestrator()
        feature_data = orchestrator.get_all_features(symbol, period=90)
        
        raw_features = feature_data.get("features", {})
        pillar_stats = feature_data.get("feature_availability", {})
        orchestrator_errors = feature_data.get("errors", [])
        orchestrator_ms = feature_data.get("execution_time_ms", 0)
        
        # === STEP 2: Load XGBoost and trace feature mapping ===
        from core.ensemble_predictor import get_ensemble_predictor
        ensemble = get_ensemble_predictor()
        predictor = ensemble.xgboost  # XGBoostModel with .model, ._loaded, .feature_names
        
        if not getattr(predictor, '_loaded', False) or predictor.model is None:
            return {
                "ok": False,
                "symbol": symbol,
                "error": "XGBoost model not loaded",
                "orchestrator_features": len(raw_features),
                "pillar_stats": pillar_stats,
            }
        
        # Rebuild the exact feature mapping from predict()
        feature_mapping = {
            "RSI_14": "RSI_14", "MACD_HISTOGRAM": "MACD_HISTOGRAM",
            "MACD_LINE": "MACD_LINE", "MACD_SIGNAL": "MACD_SIGNAL",
            "BB_POSITION": "BB_POSITION", "BB_WIDTH": "BB_WIDTH",
            "BB_UPPER": "BB_UPPER", "BB_LOWER": "BB_LOWER", "BB_MIDDLE": "BB_MIDDLE",
            "SMA_7": "SMA_7", "SMA_20": "SMA_20", "SMA_50": "SMA_50",
            "EMA_12": "EMA_12", "EMA_26": "EMA_26",
            "STOCH_K": "STOCH_K", "STOCH_D": "STOCH_D",
            "ATR_14": "ATR_14",
            "VOLUME_RATIO": "VOLUME_RATIO", "VOLUME_SMA_20": "VOLUME_SMA_20",
            "OBV": "OBV", "OBV_SMA": "OBV_SMA",
            "ROC_10": "ROC_10",
            "SMA_CROSS_20_50": "SMA_CROSS_20_50",
            "VOLATILITY_20D": "VOLATILITY_20", "VOLATILITY_20": "VOLATILITY_20",
            "DAILY_RANGE_PCT": "DAILY_RANGE_PCT",
            "BTC_RSI": "BTC_RSI",
            "BTC_MOMENTUM_4H": "BTC_MOMENTUM_1D",
            "BTC_MOMENTUM_24H": "BTC_MOMENTUM_7D",
            "BTC_MACD_BULLISH": "BTC_MACD_BULLISH",
            "BTC_CORRELATION": "BTC_CORRELATION",
            "FEAR_GREED": "fear_greed_numeric",
            "fear_greed_numeric": "fear_greed_numeric",
            "FUNDING_RATE": "funding_rate_proxy",
            "funding_rate_proxy": "funding_rate_proxy",
            "RSI_OVERSOLD": "RSI_OVERSOLD", "RSI_OVERBOUGHT": "RSI_OVERBOUGHT",
            "MACD_BULLISH": "MACD_BULLISH",
            "ABOVE_SMA_20": "ABOVE_SMA_20", "ABOVE_SMA_50": "ABOVE_SMA_50",
            "EMA_BULLISH": "EMA_BULLISH",
            "VOLUME_SPIKE": "VOLUME_SPIKE",
            "NEAR_24H_HIGH": "NEAR_7D_HIGH", "NEAR_24H_LOW": "NEAR_7D_LOW",
            "NEAR_48H_HIGH": "NEAR_30D_HIGH", "NEAR_48H_LOW": "NEAR_30D_LOW",
            "MOMENTUM_24H": "MOMENTUM_1D",
            "VOLUME_SMA_24": "VOLUME_SMA_20",
            "ROC_24": "ROC_10",
            "ABOVE_SMA_24": "ABOVE_SMA_20", "ABOVE_SMA_48": "ABOVE_SMA_50",
            "SMA_CROSS_24_48": "SMA_CROSS_20_50",
            "VOLATILITY_24H": "VOLATILITY_7D", "VOLATILITY_48H": "VOLATILITY_30D",
            "HOURLY_RANGE_PCT": "DAILY_RANGE_PCT",
        }
        
        neutral_defaults = {
            "RSI_OVERSOLD": 0, "RSI_OVERBOUGHT": 0, "MACD_BULLISH": 0.5,
            "ABOVE_SMA_20": 0.5, "ABOVE_SMA_50": 0.5, "EMA_BULLISH": 0.5,
            "SMA_CROSS_20_50": 0, "NEAR_7D_HIGH": 0, "NEAR_7D_LOW": 0,
            "NEAR_30D_HIGH": 0, "NEAR_30D_LOW": 0, "VOLUME_SPIKE": 0,
            "HIGH_FUNDING": 0, "NEGATIVE_FUNDING": 0, "EXTREME_FEAR": 0,
            "EXTREME_GREED": 0, "BTC_MACD_BULLISH": 0.5, "BTC_LEADS": 0,
            "RSI_14": 50, "BB_POSITION": 0.5, "STOCH_K": 50, "STOCH_D": 50,
            "VOLUME_RATIO": 1.0, "fear_greed_value": 50, "fear_greed_numeric": 50,
            "funding_rate_proxy": 0, "BTC_RSI": 50, "BTC_MOMENTUM_1D": 0,
            "BTC_MOMENTUM_7D": 0, "BTC_CORRELATION": 0.5, "MOMENTUM_1D": 0,
            "MOMENTUM_7D": 0, "MOMENTUM_30D": 0, "ROC_10": 0,
            "VOLATILITY_7D": 0.02, "VOLATILITY_30D": 0.02, "DAILY_RANGE_PCT": 2.0,
        }
        
        # === STEP 3: Trace every feature — the REAL truth ===
        feature_xray = []
        real_count = 0
        default_count = 0
        feature_values = []
        
        for name in predictor.feature_names:
            # Same logic as predict() — try direct, then mapping, then default
            value = raw_features.get(name, None)
            source = "direct" if value is not None else None
            mapped_from = None
            
            if value is None:
                for src, dst in feature_mapping.items():
                    if dst == name and src in raw_features:
                        value = raw_features.get(src)
                        source = "mapped"
                        mapped_from = src
                        break
            
            # Sanitize NaN/Inf — treat as missing (these crash JSON serialization)
            if value is not None:
                try:
                    fv = float(value)
                    if _xray_math.isnan(fv) or _xray_math.isinf(fv):
                        value = None
                        source = None
                except (TypeError, ValueError):
                    value = None
                    source = None
            
            is_default = value is None
            if is_default:
                value = neutral_defaults.get(name, 0.0)
                source = "DEFAULT"
                default_count += 1
            else:
                real_count += 1
            
            feature_values.append(float(value))
            feature_xray.append({
                "name": name,
                "value": round(float(value), 6),
                "source": source,
                "mapped_from": mapped_from,
                "is_default": is_default,
            })
        
        total = len(predictor.feature_names)
        quality_pct = round(real_count / total * 100, 1) if total > 0 else 0
        
        # === STEP 4: Run XGBoost prediction ===
        import numpy as np
        X = np.array([feature_values])
        proba = predictor.model.predict_proba(X)[0]
        prob_down_raw = float(proba[0])
        prob_up_raw = float(proba[1])
        
        # Apply bias correction (same as ensemble_predictor.py)
        # Without this, xray shows raw biased probabilities while
        # cockpit shows corrected ones — confusing.
        import math
        import os as _xray_os
        _XRAY_BIAS = float(_xray_os.getenv("XGBOOST_BIAS_CORRECTION", "0.7"))
        if _XRAY_BIAS > 0 and prob_up_raw > 0.001 and prob_down_raw > 0.001:
            _xray_logit = math.log(prob_up_raw / prob_down_raw)
            _xray_adj = _xray_logit - _XRAY_BIAS
            prob_up = 1.0 / (1.0 + math.exp(-_xray_adj))
            prob_down = 1.0 - prob_up
        else:
            prob_up = prob_up_raw
            prob_down = prob_down_raw
        
        spread = abs(prob_up - prob_down)
        
        import os
        hold_threshold = float(os.getenv("HOLD_ZONE_THRESHOLD", "0.08"))
        in_hold_zone = spread < hold_threshold
        
        if in_hold_zone:
            direction = "HOLD"
            confidence = max(prob_up, prob_down)
        elif prob_up >= prob_down:
            direction = "UP"
            confidence = prob_up
        else:
            direction = "DOWN"
            confidence = prob_down
        
        elapsed = round((_time.time() - start) * 1000, 1)
        
        # === BUILD VERDICT ===
        issues = []
        if quality_pct < 70:
            issues.append(f"Only {quality_pct}% features are real data (need 70%+)")
        if in_hold_zone:
            issues.append(f"In HOLD zone — spread {spread:.1%} < threshold {hold_threshold:.0%}")
        if confidence < 0.55:
            issues.append(f"Low confidence {confidence:.1%} — near coin-flip")
        for pillar, stat in pillar_stats.items():
            if "failed" in str(stat) or stat.startswith("0/"):
                issues.append(f"{pillar}: {stat}")
        
        verdict = "GOOD" if quality_pct >= 80 and not in_hold_zone and confidence >= 0.55 else "DEGRADED" if quality_pct >= 50 else "BROKEN"
        
        result = {
            "ok": True,
            "symbol": symbol,
            "verdict": verdict,
            "issues": issues,
            "feature_quality": {
                "real": real_count,
                "defaulted": default_count,
                "total": total,
                "quality_pct": quality_pct,
            },
            "xgboost_output": {
                "prob_up": round(prob_up, 4),
                "prob_down": round(prob_down, 4),
                "prob_up_raw": round(prob_up_raw, 4),
                "prob_down_raw": round(prob_down_raw, 4),
                "bias_correction": _XRAY_BIAS,
                "spread": round(spread, 4),
                "direction": direction,
                "confidence": round(confidence, 4),
                "hold_zone": in_hold_zone,
                "hold_threshold": hold_threshold,
            },
            "pillar_stats": pillar_stats,
            "orchestrator_errors": orchestrator_errors,
            "features": feature_xray,
            "defaulted_features": [f["name"] for f in feature_xray if f["is_default"]],
            "timing": {
                "orchestrator_ms": orchestrator_ms,
                "total_ms": elapsed,
            },
        }
        # Sanitize NaN/Inf BEFORE returning — Starlette's JSONResponse uses
        # json.dumps(allow_nan=False) which raises ValueError on NaN/Inf.
        # That ValueError fires AFTER this function returns (during response
        # serialization), so our try/except below never sees it. The crash
        # then surfaces as {"error":"internal_error"} from _log_requests MW.
        return _sanitize_for_json(result)
        
    except Exception as e:
        LOGGER.error(f"X-ray failed for {symbol}: {e}", exc_info=True)
        return {"ok": False, "symbol": symbol, "error": str(e)}


@router.get("/api/xray")
async def api_xray_all():
    """
    X-RAY all edge symbols at once.
    Shows feature quality + XGBoost output for every symbol Ghost trades.
    """
    from config.symbols import get_edge_set
    
    edge_symbols = sorted(get_edge_set())
    
    results = []
    for symbol in edge_symbols:
        try:
            result = await api_xray_symbol(symbol)
            results.append({
                "symbol": symbol,
                "verdict": result.get("verdict", "ERROR"),
                "quality_pct": result.get("feature_quality", {}).get("quality_pct", 0),
                "real_features": result.get("feature_quality", {}).get("real", 0),
                "defaulted": result.get("feature_quality", {}).get("defaulted", 0),
                "direction": result.get("xgboost_output", {}).get("direction", "?"),
                "confidence": result.get("xgboost_output", {}).get("confidence", 0),
                "spread": result.get("xgboost_output", {}).get("spread", 0),
                "hold_zone": result.get("xgboost_output", {}).get("hold_zone", True),
                "issues": result.get("issues", []),
            })
        except Exception as e:
            results.append({
                "symbol": symbol,
                "verdict": "ERROR",
                "error": str(e),
            })
    
    # Summary
    good = sum(1 for r in results if r.get("verdict") == "GOOD")
    degraded = sum(1 for r in results if r.get("verdict") == "DEGRADED")
    broken = sum(1 for r in results if r.get("verdict") in ("BROKEN", "ERROR"))
    avg_quality = round(sum(r.get("quality_pct", 0) for r in results) / len(results), 1) if results else 0
    
    return {
        "ok": True,
        "summary": {
            "total_symbols": len(results),
            "good": good,
            "degraded": degraded,
            "broken": broken,
            "avg_feature_quality_pct": avg_quality,
        },
        "symbols": results,
    }


@router.get("/debug/notification-loop-status")
async def notification_loop_status():
    """
    Check if the 8 AM notification loop is running and its current state.
    
    Returns:
    - running: Is the loop actively running?
    - started_at: When did the loop start?
    - loop_count: How many iterations has it run?
    - last_top10_date: Date of last TOP 10 send (prevents duplicates)
    - last_top10_send_time: Timestamp of last successful send
    - last_top10_success: Did the last send succeed?
    - predictions_count: How many predictions are in memory?
    - current_central_time: Current time in Central timezone
    """
    from zoneinfo import ZoneInfo
    central_tz = ZoneInfo("America/Chicago")
    now_central = datetime.now(central_tz)
    
    return {
        "ok": True,
        "notification_loop": _NOTIFICATION_LOOP_STATUS,
        "current_central_time": now_central.strftime("%Y-%m-%d %H:%M:%S"),
        "next_top10_hour": "08:00 Central",
        "will_send_today": (
            _NOTIFICATION_LOOP_STATUS.get("last_top10_date") != now_central.strftime("%Y-%m-%d")
        ),
        "predictions_available": len(_LATEST_PREDICTIONS),
        "env_active_tracking_enabled": os.getenv("ACTIVE_TRACKING_ENABLED", "1"),
    }


@router.post("/debug/notification-loop-start")
async def notification_loop_force_start():
    """
    Force start the notification loop if it's not running.
    Useful for debugging startup issues.
    """
    from zoneinfo import ZoneInfo
    from core.ghost_notifications import get_notification_system
    
    central_tz = ZoneInfo("America/Chicago")
    
    if _NOTIFICATION_LOOP_STATUS.get("running"):
        return {"ok": True, "message": "Loop already running", "status": _NOTIFICATION_LOOP_STATUS}
    
    try:
        # Set up telegram function
        def _send_telegram(message: str) -> bool:
            if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
                return False
            return _tg_send_chat_message(TELEGRAM_CHAT_ID, message)
        
        notification_system = get_notification_system()
        notification_system.set_telegram_func(_send_telegram)
        
        async def _ghost_notification_loop_v2():
            """Simplified notification loop for debugging"""
            nonlocal notification_system
            
            TOP_10_HOUR = 99  # DISABLED - Use /debug/send-top10-now instead
            _NOTIFICATION_LOOP_STATUS["running"] = True
            _NOTIFICATION_LOOP_STATUS["started_at"] = datetime.now(central_tz).isoformat()
            LOGGER.info("[NOTIFICATION LOOP V2] Starting...")
            
            last_top10_date = None
            loop_count = 0
            
            while True:
                try:
                    loop_count += 1
                    now_central = datetime.now(central_tz)
                    current_hour = now_central.hour
                    current_date = now_central.strftime("%Y-%m-%d")
                    
                    _NOTIFICATION_LOOP_STATUS["loop_count"] = loop_count
                    _NOTIFICATION_LOOP_STATUS["current_central_time"] = now_central.strftime("%Y-%m-%d %H:%M:%S")
                    _NOTIFICATION_LOOP_STATUS["predictions_count"] = len(_LATEST_PREDICTIONS)
                    _NOTIFICATION_LOOP_STATUS["last_top10_date"] = last_top10_date
                    
                    # Send TOP 10 at 8 AM
                    if current_hour == TOP_10_HOUR and last_top10_date != current_date:
                        LOGGER.info(f"[NOTIFICATION LOOP V2] Sending TOP 10 at {now_central}")
                        success = notification_system.send_top10(_LATEST_PREDICTIONS)
                        if success:
                            last_top10_date = current_date
                            _NOTIFICATION_LOOP_STATUS["last_top10_date"] = current_date
                            _NOTIFICATION_LOOP_STATUS["last_top10_send_time"] = now_central.isoformat()
                            _NOTIFICATION_LOOP_STATUS["last_top10_success"] = True
                        else:
                            _NOTIFICATION_LOOP_STATUS["last_top10_success"] = False
                    
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    _NOTIFICATION_LOOP_STATUS["running"] = False
                    break
                except Exception as e:
                    LOGGER.error(f"[NOTIFICATION LOOP V2] Error: {e}")
                    await asyncio.sleep(60)
        
        asyncio.create_task(_ghost_notification_loop_v2())
        
        return {
            "ok": True,
            "message": "Notification loop started via debug endpoint",
            "status": _NOTIFICATION_LOOP_STATUS
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/debug/outcome-data-audit")
async def debug_outcome_data_audit():
    """
    AUDIT: Understand where all the outcome data is and why only some has symbols.
    
    Checks:
    1. ghost_prediction_outcomes - total rows, with/without symbols
    2. Which columns are NULL most often
    3. Date range of data
    4. Why symbol_accuracy only has 477 predictions
    """
    from datetime import datetime
    from core.db_pool import get_sync_connection
    
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return {"ok": False, "error": "DATABASE_URL not configured"}
    
    try:
        with get_sync_connection() as conn:
            cur = conn.cursor()
            
            audit = {}
        
            # 1. Total outcomes and symbol coverage
            cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(symbol) as with_symbol,
                    COUNT(CASE WHEN symbol IS NULL OR symbol = '' THEN 1 END) as missing_symbol,
                    COUNT(DISTINCT symbol) as unique_symbols
                FROM ghost_prediction_outcomes
            """)
            row = cur.fetchone()
            audit["total_outcomes"] = row[0]
            audit["with_symbol"] = row[1]
            audit["missing_symbol"] = row[2]
            audit["unique_symbols"] = row[3]
            
            # 2. NULL column analysis
            cur.execute("""
                SELECT 
                    COUNT(CASE WHEN predicted_direction IS NULL THEN 1 END) as null_predicted_dir,
                    COUNT(CASE WHEN actual_direction IS NULL THEN 1 END) as null_actual_dir,
                    COUNT(CASE WHEN hit_direction IS NULL THEN 1 END) as null_hit_dir,
                    COUNT(CASE WHEN price_at_prediction IS NULL OR price_at_prediction = 0 THEN 1 END) as null_entry_price,
                    COUNT(CASE WHEN price_at_resolution IS NULL OR price_at_resolution = 0 THEN 1 END) as null_exit_price,
                    COUNT(CASE WHEN closed_at IS NULL THEN 1 END) as null_closed_at
                FROM ghost_prediction_outcomes
            """)
            row = cur.fetchone()
            audit["null_columns"] = {
                "predicted_direction": row[0],
                "actual_direction": row[1],
                "hit_direction": row[2],
                "entry_price_missing": row[3],
                "exit_price_missing": row[4],
                "closed_at": row[5]
            }
            
            # 3. Date range
            cur.execute("""
                SELECT 
                    MIN(closed_at) as earliest,
                    MAX(closed_at) as latest,
                    COUNT(DISTINCT DATE(closed_at)) as days_with_data
                FROM ghost_prediction_outcomes
                WHERE closed_at IS NOT NULL
            """)
            row = cur.fetchone()
            audit["date_range"] = {
                "earliest": row[0].isoformat() if row[0] else None,
                "latest": row[1].isoformat() if row[1] else None,
                "days_with_data": row[2]
            }
            
            # 4. Symbol distribution - top 10 symbols by count
            cur.execute("""
                SELECT symbol, COUNT(*) as count
                FROM ghost_prediction_outcomes
                WHERE symbol IS NOT NULL AND symbol != ''
                GROUP BY symbol
                ORDER BY count DESC
                LIMIT 10
            """)
            audit["top_symbols"] = [{"symbol": r[0], "count": r[1]} for r in cur.fetchall()]
            
            # 5. Check hit_direction distribution (this is what accuracy is based on)
            cur.execute("""
                SELECT 
                    hit_direction,
                    COUNT(*) as count
                FROM ghost_prediction_outcomes
                GROUP BY hit_direction
                ORDER BY hit_direction
            """)
            audit["hit_direction_distribution"] = [{"hit_direction": r[0], "count": r[1]} for r in cur.fetchall()]
            
            # 5b. Check STATUS distribution - this is the KEY insight!
            cur.execute("""
                SELECT 
                    status,
                    COUNT(*) as count
                FROM ghost_prediction_outcomes
                GROUP BY status
                ORDER BY count DESC
            """)
            audit["status_distribution"] = [{"status": r[0], "count": r[1]} for r in cur.fetchall()]
            
            # 6. Compare with ghost_symbol_accuracy table
            cur.execute("""
                SELECT 
                    COUNT(*) as symbols_tracked,
                    SUM(total_predictions) as total_in_accuracy_table,
                    SUM(correct_predictions) as correct_in_accuracy_table
                FROM ghost_symbol_accuracy
            """)
            row = cur.fetchone()
            audit["symbol_accuracy_table"] = {
                "symbols_tracked": row[0],
                "total_predictions": row[1],
                "correct_predictions": row[2]
            }
            
            # 7. Check if there's a predictions table that should link to outcomes
            cur.execute("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_name = 'predictions'
            """)
            predictions_table_exists = cur.fetchone()[0] > 0
            audit["predictions_table_exists"] = predictions_table_exists
            
            if predictions_table_exists:
                cur.execute("SELECT COUNT(*) FROM predictions")
                audit["predictions_count"] = cur.fetchone()[0]
                
                # Check predictions table schema
                cur.execute("""
                    SELECT column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'predictions'
                    ORDER BY ordinal_position
                """)
                audit["predictions_schema"] = [{"col": r[0], "type": r[1]} for r in cur.fetchall()]
                
                # Sample prediction to see what data we have
                cur.execute("""
                    SELECT * FROM predictions
                    ORDER BY run_at DESC
                    LIMIT 3
                """)
                cols = [desc[0] for desc in cur.description]
                audit["sample_predictions"] = [dict(zip(cols, r)) for r in cur.fetchall()]
                
                # Check outcomes table too
                cur.execute("""
                    SELECT COUNT(*) FROM information_schema.tables 
                    WHERE table_name = 'outcomes'
                """)
                outcomes_table_exists = cur.fetchone()[0] > 0
                if outcomes_table_exists:
                    cur.execute("SELECT COUNT(*) FROM outcomes")
                    audit["outcomes_table_count"] = cur.fetchone()[0]
            
            cur.close()
            
            # Calculate the gap
            audit["data_gap_analysis"] = {
                "total_in_ghost_prediction_outcomes": audit["total_outcomes"],
                "with_valid_symbol": audit["with_symbol"],
                "percentage_with_symbol": f"{(audit['with_symbol'] / audit['total_outcomes'] * 100):.1f}%" if audit['total_outcomes'] > 0 else "0%",
                "in_symbol_accuracy_table": audit["symbol_accuracy_table"]["total_predictions"],
                "gap": audit["with_symbol"] - (audit["symbol_accuracy_table"]["total_predictions"] or 0)
            }
            
            return {"ok": True, "audit": audit}
        
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/inverse-status")
async def debug_inverse_status():
    """
    Check INVERSE GHOST mode status and per-symbol configuration.
    
    Shows:
    - Global INVERSE mode enabled/disabled
    - Per-symbol INVERSE_SKIP_SYMBOLS list
    - Which symbols will be inverted vs kept raw
    """
    try:
        # DEFAULT is now OFF (0) since accuracy improvements were made
        inverse_enabled = os.getenv("INVERSE_GHOST", "0") == "1"
        
        # Read INVERSE_SKIP_SYMBOLS from env var or use defaults
        env_skip = os.getenv("INVERSE_SKIP_SYMBOLS", "").strip()
        if env_skip:
            inverse_skip_symbols = {s.strip().upper() for s in env_skip.split(",") if s.strip()}
        else:
            inverse_skip_symbols = {
                "OMG", "RLC", "THETA", "EGLD", "BAT", "ONDO", "ZEN",
                "DOGE", "DOT", "ZRX", "BNB", "AVAX", "OCEAN", "ANT"
            }
        
        # Test symbols
        test_symbols = ["BTC", "ETH", "BNB", "DOGE", "SAND", "SOL", "OMG", "THETA"]
        symbol_modes = {}
        for sym in test_symbols:
            if sym in inverse_skip_symbols:
                symbol_modes[sym] = "RAW (skip INVERSE - high raw accuracy)"
            elif inverse_enabled:
                symbol_modes[sym] = "INVERTED (INVERSE mode active)"
            else:
                symbol_modes[sym] = "RAW (INVERSE disabled globally)"
        
        return {
            "ok": True,
            "inverse_ghost_enabled": inverse_enabled,
            "inverse_skip_symbols": sorted(inverse_skip_symbols),
            "inverse_skip_count": len(inverse_skip_symbols),
            "symbol_modes": symbol_modes,
            "explanation": {
                "INVERSE mode": "Flips predictions (UP→DOWN, DOWN→UP) for symbols with <50% raw accuracy",
                "INVERSE_SKIP": "Symbols with >60% raw accuracy that should NOT be inverted",
            }
        }
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/v3-validation")
async def debug_v3_validation():
    """
    V3 BACKTEST VALIDATION STATUS
    
    Shows performance of backtest-validated strategies:
    - ETH ghost_inverse @ 72h: Expected 61.5% (p=0.027)
    - XRP mean_reversion @ 168h: Expected 56.5% (p=0.026)
    - LINK mean_reversion @ 72h: Expected 55.2% (p=0.049)
    
    Also shows REMOVED symbols that failed validation.
    """
    try:
        from core.ghost_notifications import V3_VALIDATED_STRATEGIES, V3_REMOVED_SYMBOLS, V3_DEFAULT_HOLD_HOURS
        from core.paper_tracker import get_paper_tracker
        
        # Query live performance for V3 validated trades from paper_trades
        # FIXED: Use PaperTracker abstraction instead of raw psycopg2
        live_stats = {}
        pending_stats = {}
        
        try:
            tracker = get_paper_tracker()
            conn = tracker._get_connection()
            
            for symbol in V3_VALIDATED_STRATEGIES.keys():
                # Query V3 validated paper trades with outcomes
                cur = tracker._execute(conn, """
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN outcome IN ('WIN', 'LOSS') THEN 1 ELSE 0 END) as resolved,
                        v3_strategy,
                        v3_hold_hours,
                        v3_backtest_win_rate
                    FROM paper_trades
                    WHERE symbol = ?
                    AND v3_validated = TRUE
                    GROUP BY v3_strategy, v3_hold_hours, v3_backtest_win_rate
                """, (symbol,))
                row = tracker._fetchone(cur)
                if row:
                    resolved_count = row.get("resolved", 0) or 0
                    wins_count = row.get("wins", 0) or 0
                    live_rate = round(wins_count / resolved_count, 3) if resolved_count > 0 else None
                    live_stats[symbol] = {
                        "total_v3": row.get("total", 0),
                        "wins": wins_count,
                        "resolved": resolved_count,
                        "live_win_rate": live_rate,
                        "v3_strategy": row.get("v3_strategy"),
                        "v3_hold_hours": row.get("v3_hold_hours"),
                        "v3_backtest_win_rate": round(float(row.get("v3_backtest_win_rate") or 0), 3) if row.get("v3_backtest_win_rate") else None,
                    }
            
            # Also check for pending V3 trades
            cur = tracker._execute(conn, """
                SELECT symbol, COUNT(*) as pending_count
                FROM paper_trades
                WHERE v3_validated = TRUE
                AND outcome = 'PENDING'
                GROUP BY symbol
            """)
            pending_rows = tracker._fetchall(cur)
            pending_stats = {row["symbol"]: row["pending_count"] for row in pending_rows}
            
            conn.close()
        except Exception as db_err:
            LOGGER.error(f"V3 validation query error: {db_err}")
        
        # Build validation report
        validation_report = {}
        for symbol, config in V3_VALIDATED_STRATEGIES.items():
            expected = config.get("win_rate", 0.5)
            stats = live_stats.get(symbol, {})
            live_rate = stats.get("live_win_rate")
            resolved = stats.get("resolved", 0)
            pending = pending_stats.get(symbol, 0)
            
            validation_report[symbol] = {
                "strategy": config.get("strategy"),
                "hold_hours": config.get("hold_hours"),
                "backtest_win_rate": expected,
                "backtest_p_value": config.get("p_value"),
                "backtest_sample_size": config.get("sample_size"),
                "live_win_rate": live_rate,
                "live_resolved": resolved,
                "live_pending": pending,
                "tracking": "✅ TRACKING" if (resolved + pending) > 0 else "⏳ NO DATA YET",
                "validation": "🔬 VALIDATING" if resolved < 30 else ("✅ VALIDATED" if live_rate and live_rate >= expected * 0.85 else "⚠️ UNDERPERFORMING"),
            }
        
        return {
            "ok": True,
            "v3_mode": "BACKTEST-VALIDATED",
            "default_hold_hours": V3_DEFAULT_HOLD_HOURS,
            "validated_symbols": list(V3_VALIDATED_STRATEGIES.keys()),
            "removed_symbols": list(V3_REMOVED_SYMBOLS),
            "validation_report": validation_report,
            "backtest_summary": {
                "total_trades_analyzed": 52433,
                "statistically_significant_results": 8,
                "significance_threshold": "p < 0.05",
                "overall_market_efficiency": "50.0% (random walk confirmed)",
            },
            "notes": [
                "ETH ghost_inverse: Only symbol where inverting Ghost beats 50%",
                "XRP/LINK mean_reversion: Price bounces beat trend following",
                "RSI strategies: 45-46% win rate - consistently LOSE money",
                "SOL/BTC/AVAX: Removed - no statistical significance",
            ]
        }
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/migrate-v3-columns")
async def debug_migrate_v3_columns():
    """
    Add V3 tracking columns to paper_trades table if they don't exist.
    
    Safe to run multiple times - only adds columns if missing.
    """
    try:
        import psycopg2
        from core.db_pool import get_sync_connection
        
        DATABASE_URL = os.getenv("DATABASE_URL")
        if not DATABASE_URL:
            return {"ok": False, "error": "DATABASE_URL not set"}
        
        with get_sync_connection() as conn:
            cur = conn.cursor()
            
            # V3 columns to add
            v3_columns = [
                ("v3_validated", "BOOLEAN DEFAULT FALSE"),
                ("v3_strategy", "TEXT"),
                ("v3_is_inverse", "BOOLEAN DEFAULT FALSE"),
                ("v3_original_direction", "TEXT"),
                ("v3_hold_hours", "INTEGER"),
                ("v3_backtest_win_rate", "REAL"),
            ]
            
            added = []
            skipped = []
            
            for col_name, col_type in v3_columns:
                try:
                    cur.execute(f"ALTER TABLE paper_trades ADD COLUMN {col_name} {col_type}")
                    added.append(col_name)
                except psycopg2.errors.DuplicateColumn:
                    conn.rollback()  # Reset transaction state
                    skipped.append(col_name)
                except Exception as e:
                    conn.rollback()
                    skipped.append(f"{col_name}: {e}")
            
            cur.close()
            
            return {
                "ok": True,
                "added_columns": added,
                "already_existed": skipped,
                "message": f"Migration complete. Added {len(added)} columns, {len(skipped)} already existed."
            }
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/v3-filter-test")
async def debug_v3_filter_test():
    """
    Test V3 filter with current predictions.
    Shows what gets through and why symbols are filtered out.
    """
    try:
        from core.ghost_notifications import (
            v3_filter_and_score, V3_VALIDATED_STRATEGIES, V3_REMOVED_SYMBOLS,
            V3_MIN_CONFIDENCE, V3_ENABLED
        )
        
        # Build test predictions matching what send_top10_now does
        CRYPTO = ["ETH", "XRP", "LINK", "SOL", "BTC"]
        
        test_preds = []
        for symbol in CRYPTO:
            # Simulate prediction call
            pred_result = await _run_turbo_prediction_for_top10(symbol)
            if pred_result and pred_result.get("ok"):
                direction = pred_result.get("direction", "FLAT")
                confidence = pred_result.get("confidence", 0.5)
                
                test_preds.append({
                    "symbol": symbol,
                    "direction": direction,
                    "confidence": confidence,
                    "current": pred_result.get("current_price", 0),
                    "asset_type": "crypto",
                })
        
        # Run V3 filter
        filtered = v3_filter_and_score(test_preds)
        
        # Build detailed report
        filter_report = {}
        for p in test_preds:
            sym = p['symbol']
            dir_ = p['direction']
            conf = p['confidence']
            
            # Check why it would be filtered
            in_validated = sym in V3_VALIDATED_STRATEGIES
            in_removed = sym in V3_REMOVED_SYMBOLS
            passed_filter = any(f['symbol'] == sym for f in filtered)
            
            reason = "UNKNOWN"
            if in_removed:
                reason = f"REMOVED: {V3_REMOVED_SYMBOLS.get(sym, 'not validated')}"
            elif not in_validated:
                reason = "NOT IN V3_VALIDATED_STRATEGIES"
            elif in_validated:
                strat = V3_VALIDATED_STRATEGIES[sym]
                if strat['strategy'] == 'ghost_inverse':
                    if dir_ != 'DOWN':
                        reason = f"SKIPPED: ghost_inverse requires DOWN, got {dir_}"
                    elif conf < V3_MIN_CONFIDENCE:
                        reason = f"SKIPPED: inverse but conf {conf:.0%} < {V3_MIN_CONFIDENCE:.0%}"
                    else:
                        reason = "SHOULD PASS (inverse with sufficient conf)"
                elif conf < V3_MIN_CONFIDENCE:
                    reason = f"SKIPPED: confidence {conf:.0%} < {V3_MIN_CONFIDENCE:.0%}"
                else:
                    reason = "SHOULD PASS"
            
            filter_report[sym] = {
                "raw_direction": dir_,
                "raw_confidence": conf,
                "in_validated": in_validated,
                "in_removed": in_removed,
                "passed_filter": passed_filter,
                "reason": reason,
            }
        
        return {
            "ok": True,
            "v3_enabled": V3_ENABLED,
            "min_confidence": V3_MIN_CONFIDENCE,
            "test_predictions_count": len(test_preds),
            "filtered_count": len(filtered),
            "filter_report": filter_report,
            "filtered_symbols": [f['symbol'] for f in filtered],
            "explanation": {
                "ETH": "Only if Ghost predicts DOWN AND conf >= 70% (inverse to UP)",
                "XRP": "Any direction (mean_reversion) if conf >= 70%",
                "LINK": "Any direction (mean_reversion) if conf >= 70%",
            }
        }
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/db-audit")
async def debug_db_audit():
    """
    Audit PostgreSQL for corrupt data.
    
    Checks for:
    - BTC prices below $10,000 (corrupt)
    - ETH prices below $500 (corrupt)
    - Zero/negative prices
    - Outcome statistics
    
    Tables:
    - ghost_prediction_outcomes: price_at_prediction, price_at_resolution, hit_direction, status
    """
    try:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            return {"ok": False, "error": "DATABASE_URL not set"}
        
        from core.db_pool import get_sync_connection
        with get_sync_connection() as conn:
            cursor = conn.cursor()
            
            # Price validation thresholds
            MIN_VALID = {
                'BTC': 10000, 'ETH': 500, 'SOL': 5, 'BNB': 100,
                'XRP': 0.10, 'ADA': 0.05, 'DOGE': 0.001
            }
            
            # 1. Get outcomes overview
            cursor.execute("""
                SELECT COUNT(*) as total,
                       COUNT(DISTINCT symbol) as symbols,
                       MIN(closed_at) as earliest,
                       MAX(closed_at) as latest,
                       SUM(CASE WHEN symbol IS NULL OR symbol = '' THEN 1 ELSE 0 END) as missing_symbol,
                       SUM(CASE WHEN price_at_prediction IS NULL THEN 1 ELSE 0 END) as missing_entry
                FROM ghost_prediction_outcomes
            """)
            overview = cursor.fetchone()
            
            # 2. Find corrupt outcomes by symbol (check price_at_prediction in ghost_prediction_outcomes)
            corrupt_entry = {}
            for symbol, min_price in MIN_VALID.items():
                cursor.execute("""
                    SELECT COUNT(*) as cnt,
                           MIN(price_at_prediction) as min_p,
                           MAX(price_at_prediction) as max_p
                    FROM ghost_prediction_outcomes
                    WHERE symbol = %s AND price_at_prediction IS NOT NULL AND price_at_prediction < %s
                """, (symbol, min_price))
                row = cursor.fetchone()
                if row[0] > 0:
                    corrupt_entry[symbol] = {
                        "corrupt_count": row[0],
                        "min_price": float(row[1]) if row[1] else 0,
                        "max_price": float(row[2]) if row[2] else 0,
                        "threshold": min_price
                    }
            
            # 3. Find corrupt exit prices
            corrupt_exit = {}
            for symbol, min_price in MIN_VALID.items():
                cursor.execute("""
                    SELECT COUNT(*) as cnt,
                           MIN(price_at_resolution) as min_p,
                           MAX(price_at_resolution) as max_p
                    FROM ghost_prediction_outcomes
                    WHERE symbol = %s AND price_at_resolution IS NOT NULL AND price_at_resolution < %s
                """, (symbol, min_price))
                row = cursor.fetchone()
                if row[0] > 0:
                    corrupt_exit[symbol] = {
                        "corrupt_count": row[0],
                        "min_price": float(row[1]) if row[1] else 0,
                        "max_price": float(row[2]) if row[2] else 0,
                        "threshold": min_price
                    }
            
            # 4. Get outcomes stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN hit_direction = 0 THEN 1 ELSE 0 END) as losses,
                    SUM(CASE WHEN status = 'no_data' THEN 1 ELSE 0 END) as no_data,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as completed
                FROM ghost_prediction_outcomes
            """)
            outcomes = cursor.fetchone()
            
            # 5. Sample of corrupt BTC entry prices
            cursor.execute("""
                SELECT id, symbol, price_at_prediction, price_at_resolution, status, predicted_direction
                FROM ghost_prediction_outcomes
                WHERE symbol = 'BTC' AND price_at_prediction IS NOT NULL AND price_at_prediction < 10000
                ORDER BY id DESC
                LIMIT 10
            """)
            btc_entry_samples = [
                {"id": r[0], "symbol": r[1], "entry": float(r[2]) if r[2] else 0, 
                 "exit": float(r[3]) if r[3] else 0, "status": r[4], "direction": r[5]}
                for r in cursor.fetchall()
            ]
            
            # 6. Sample of corrupt BTC exit prices
            cursor.execute("""
                SELECT id, symbol, price_at_prediction, price_at_resolution, status, predicted_direction
                FROM ghost_prediction_outcomes
                WHERE symbol = 'BTC' AND price_at_resolution IS NOT NULL AND price_at_resolution < 10000
                ORDER BY id DESC
                LIMIT 10
            """)
            btc_exit_samples = [
                {"id": r[0], "symbol": r[1], "entry": float(r[2]) if r[2] else 0, 
                 "exit": float(r[3]) if r[3] else 0, "status": r[4], "direction": r[5]}
                for r in cursor.fetchall()
            ]
        
        total_corrupt = sum(d["corrupt_count"] for d in corrupt_entry.values())
        total_corrupt += sum(d["corrupt_count"] for d in corrupt_exit.values())
        
        return {
            "ok": True,
            "overview": {
                "total": overview[0],
                "symbols": overview[1],
                "earliest": str(overview[2]) if overview[2] else None,
                "latest": str(overview[3]) if overview[3] else None,
                "missing_symbol": overview[4],
                "missing_entry_price": overview[5]
            },
            "corrupt_entry_prices": corrupt_entry,
            "corrupt_exit_prices": corrupt_exit,
            "total_corrupt": total_corrupt,
            "outcomes_stats": {
                "total": outcomes[0] or 0,
                "wins": outcomes[1] or 0,
                "losses": outcomes[2] or 0,
                "no_data": outcomes[3] or 0,
                "completed": outcomes[4] or 0,
                "accuracy_pct": round((outcomes[1] or 0) / max(1, (outcomes[1] or 0) + (outcomes[2] or 0)) * 100, 2)
            },
            "btc_corrupt_entry_samples": btc_entry_samples,
            "btc_corrupt_exit_samples": btc_exit_samples,
            "recommendation": "Run /debug/db-clean to remove corrupt data" if total_corrupt > 0 else "Database looks clean!"
        }
        
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/sweetspot")
async def debug_sweetspot():
    """
    Analyze paper trades to find Ghost's sweet spots.
    Returns comprehensive accuracy breakdown by symbol, confidence, direction, asset type, etc.
    Uses 'outcome' column with values: WIN, LOSS, PENDING, BREAK_EVEN
    """
    try:
        from psycopg2.extras import RealDictCursor
        
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            return {"ok": False, "error": "DATABASE_URL not set"}
        
        from core.db_pool import get_sync_connection
        with get_sync_connection() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            results = {}
            
            # Known crypto symbols
            crypto_symbols = ['BTC', 'ETH', 'SOL', 'XRP', 'AVAX', 'ADA', 'DOT', 'LINK', 'MATIC',
                             'DOGE', 'SHIB', 'LTC', 'BCH', 'ATOM', 'UNI', 'AAVE', 'MKR', 'CRV',
                             'RNDR', 'TURBO', 'CHZ', 'ILV', 'ZEC', 'INJ', 'SUI', 'APT', 'ARB', 
                             'OP', 'TIA', 'SEI', 'FET', 'PEPE', 'WIF', 'BONK', 'FIL', 'ICP',
                             'NEAR', 'ALGO', 'VET', 'HBAR', 'GRT', 'ENS', 'SAND', 'MANA', 'AXS',
                             'GALA', 'IMX', 'BLUR', 'APE', 'LDO', 'SNX', 'COMP', 'SUSHI', 'YFI',
                             '1INCH', 'BAT', 'ZRX', 'DASH', 'XMR', 'ETC', 'ENJ', 'IQ', 'LPT',
                             'API3', 'BAND', 'MASK', 'CELO', 'JUP', 'IOTX', 'RLC', 'EGLD', 'QNT',
                             'BRETT', 'SSV', 'RPL', 'BNB', 'KAVA', 'JASMY', 'ROSE', 'FTM', 'ONE']
            crypto_list = "', '".join(crypto_symbols)
            
            # 1. ALL SYMBOLS RANKED (min 20 trades)
            cur.execute("""
                SELECT symbol, 
                       COUNT(*) as trades,
                       SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                       ROUND(100.0 * SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate
                FROM paper_trades  
                WHERE outcome IN ('WIN', 'LOSS')
                GROUP BY symbol
                HAVING COUNT(*) >= 20
                ORDER BY win_rate DESC
            """)
            all_symbols = [dict(r) for r in cur.fetchall()]
            results['all_symbols_ranked'] = all_symbols
            
            # 2. STOCKS ONLY (min 15 trades)
            cur.execute(f"""
                SELECT symbol, 
                       COUNT(*) as trades,
                       SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                       ROUND(100.0 * SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate
                FROM paper_trades  
                WHERE outcome IN ('WIN', 'LOSS')
                  AND symbol NOT IN ('{crypto_list}')
                GROUP BY symbol
                HAVING COUNT(*) >= 15
                ORDER BY win_rate DESC
            """)
            stocks_only = [dict(r) for r in cur.fetchall()]
            results['stocks_only'] = stocks_only
            
            # 3. CRYPTO ONLY (min 15 trades)
            cur.execute(f"""
                SELECT symbol, 
                       COUNT(*) as trades,
                       SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                       ROUND(100.0 * SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate
                FROM paper_trades  
                WHERE outcome IN ('WIN', 'LOSS')
                  AND symbol IN ('{crypto_list}')
                GROUP BY symbol
                HAVING COUNT(*) >= 15
                ORDER BY win_rate DESC
            """)
            crypto_only = [dict(r) for r in cur.fetchall()]
            results['crypto_only'] = crypto_only
            
            # 4. BEST SYMBOL+DIRECTION COMBOS (min 10 trades, top 50)
            cur.execute("""
                SELECT symbol, 
                       signal_direction as direction,
                       COUNT(*) as trades,
                       SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                       ROUND(100.0 * SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate
                FROM paper_trades  
                WHERE outcome IN ('WIN', 'LOSS')
                GROUP BY symbol, signal_direction
                HAVING COUNT(*) >= 10
                ORDER BY win_rate DESC
                LIMIT 50
            """)
            best_combos = [dict(r) for r in cur.fetchall()]
            results['best_direction_combos'] = best_combos
            
            # 5. WORST PERFORMERS - INVERSE CANDIDATES (min 10 trades, bottom 30)
            cur.execute("""
                SELECT symbol, 
                       signal_direction as direction,
                       COUNT(*) as trades,
                       SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                       ROUND(100.0 * SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) / COUNT(*), 1) as win_rate
                FROM paper_trades  
                WHERE outcome IN ('WIN', 'LOSS')
                GROUP BY symbol, signal_direction
                HAVING COUNT(*) >= 10
                ORDER BY win_rate ASC
                LIMIT 30
            """)
            worst_combos = [dict(r) for r in cur.fetchall()]
            results['inverse_candidates'] = worst_combos
            
            # 6. Overall stats
            cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                    SUM(CASE WHEN outcome = 'PENDING' THEN 1 ELSE 0 END) as pending
                FROM paper_trades
            """)
            overall = dict(cur.fetchone())
            wins = overall['wins'] or 0
            losses = overall['losses'] or 0
            resolved = wins + losses
            overall['win_rate'] = round((wins / resolved * 100), 1) if resolved > 0 else 0
            results['overall'] = overall
            
            # Build recommendations
            tradeable_stocks = [s for s in stocks_only if float(s['win_rate']) >= 60]
            tradeable_crypto = [s for s in crypto_only if float(s['win_rate']) >= 60]
            avoid_stocks = [s for s in stocks_only if float(s['win_rate']) < 50]
            avoid_crypto = [s for s in crypto_only if float(s['win_rate']) < 50]
            inverse_candidates = [c for c in worst_combos if float(c['win_rate']) <= 20]
            
            results['recommendations'] = {
                'tradeable_stocks': [{'symbol': s['symbol'], 'win_rate': float(s['win_rate']), 'trades': s['trades']} for s in tradeable_stocks],
                'tradeable_crypto': [{'symbol': s['symbol'], 'win_rate': float(s['win_rate']), 'trades': s['trades']} for s in tradeable_crypto],
                'avoid_stocks': [{'symbol': s['symbol'], 'win_rate': float(s['win_rate']), 'trades': s['trades']} for s in avoid_stocks],
                'avoid_crypto': [{'symbol': s['symbol'], 'win_rate': float(s['win_rate']), 'trades': s['trades']} for s in avoid_crypto],
                'inverse_candidates': [{'symbol': c['symbol'], 'direction': c['direction'], 'win_rate': float(c['win_rate']), 'trades': c['trades']} for c in inverse_candidates],
                'v3_whitelist_stocks': [s['symbol'] for s in tradeable_stocks[:10]],
                'v3_whitelist_crypto': [s['symbol'] for s in tradeable_crypto[:10]],
                'v3_blacklist': [s['symbol'] for s in avoid_stocks + avoid_crypto if float(s['win_rate']) < 30]
            }
        
        return {"ok": True, **results}
        
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/checkpoint-status")
async def debug_checkpoint_status():
    """
    Debug endpoint to verify multi-checkpoint Trust Ladder is working.
    
    Shows:
    - Checkpoint columns exist in paper_trades table
    - Sample pending trades with checkpoint data
    - Recent checkpoint evaluations from logs
    """
    try:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            return {"ok": False, "error": "DATABASE_URL not set"}
        
        from core.db_pool import get_sync_connection
        with get_sync_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Check if checkpoint columns exist
            cursor.execute("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'paper_trades'
                AND column_name IN ('trust_level', 'checkpoint_times', 'checkpoint_results', 'checkpoint_evaluated', 'checkpoint_prices')
                ORDER BY column_name
            """)
            columns = {row[0]: row[1] for row in cursor.fetchall()}
            
            expected_columns = ['trust_level', 'checkpoint_times', 'checkpoint_results', 'checkpoint_evaluated', 'checkpoint_prices']
            missing_columns = [c for c in expected_columns if c not in columns]
            
            # 2. Get pending trades with checkpoint data
            cursor.execute("""
                SELECT symbol, trust_level, entry_time, target_time, 
                       checkpoint_times, checkpoint_results, checkpoint_evaluated
                FROM paper_trades
                WHERE outcome = 'PENDING'
                AND checkpoint_times IS NOT NULL
                AND checkpoint_times != '[]'
                ORDER BY trust_level DESC, entry_time DESC
                LIMIT 10
            """)
            pending_with_checkpoints = []
            for row in cursor.fetchall():
                # Handle both datetime objects and strings
                entry_time = row[2].isoformat() if hasattr(row[2], 'isoformat') else str(row[2]) if row[2] else None
                target_time = row[3].isoformat() if hasattr(row[3], 'isoformat') else str(row[3]) if row[3] else None
                pending_with_checkpoints.append({
                    "symbol": row[0],
                    "trust_level": row[1],
                    "entry_time": entry_time,
                    "target_time": target_time,
                    "checkpoint_times": row[4],
                    "checkpoint_results": row[5],
                    "checkpoint_evaluated": row[6]
                })
            
            # 3. Get trades with evaluated checkpoints
            cursor.execute("""
                SELECT symbol, trust_level, checkpoint_results, checkpoint_evaluated, outcome
                FROM paper_trades
                WHERE checkpoint_evaluated IS NOT NULL
                AND checkpoint_evaluated != '[]'
                AND checkpoint_evaluated::text LIKE '%true%'
                ORDER BY entry_time DESC
                LIMIT 5
            """)
            evaluated_checkpoints = []
            for row in cursor.fetchall():
                evaluated_checkpoints.append({
                    "symbol": row[0],
                    "trust_level": row[1],
                    "checkpoint_results": row[2],
                    "checkpoint_evaluated": row[3],
                    "final_outcome": row[4]
                })
            
            # 4. Count trades by trust level
            cursor.execute("""
                SELECT trust_level, COUNT(*) as count
                FROM paper_trades
                WHERE trust_level IS NOT NULL
                GROUP BY trust_level
                ORDER BY trust_level
            """)
            trades_by_level = {row[0]: row[1] for row in cursor.fetchall()}
        
        return {
            "ok": True,
            "checkpoint_system": "READY" if not missing_columns else "INCOMPLETE",
            "columns_found": columns,
            "missing_columns": missing_columns,
            "pending_trades_with_checkpoints": len(pending_with_checkpoints),
            "pending_samples": pending_with_checkpoints,
            "evaluated_checkpoints_count": len(evaluated_checkpoints),
            "evaluated_samples": evaluated_checkpoints,
            "trades_by_trust_level": trades_by_level,
            "message": "Multi-checkpoint system is operational" if not missing_columns else f"Missing columns: {missing_columns}"
        }
        
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/promote-symbol")
async def debug_promote_symbol(symbol: str, level: int = 2):
    """
    TEST ENDPOINT: Promote a symbol to a higher trust level for testing multi-checkpoint.
    
    Args:
        symbol: Symbol to promote (e.g., BTC, ETH)
        level: Trust level (1, 2, or 3)
    
    After promotion, the next trade for this symbol will have:
    - Level 2: checkpoint_times at [60hr, 120hr]
    - Level 3: checkpoint_times at [72hr, 168hr]
    """
    if level not in [1, 2, 3]:
        return {"error": f"Invalid level {level}. Must be 1, 2, or 3"}
    
    symbol = symbol.upper()
    
    try:
        import psycopg2
        
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            return {"ok": False, "error": "DATABASE_URL not set"}
        
        from core.db_pool import get_sync_connection
        with get_sync_connection() as conn:
            cursor = conn.cursor()
            
            # Upsert the symbol to the specified trust level
            cursor.execute("""
                INSERT INTO ghost_symbol_trust (symbol, trust_level, consecutive_wins, checkpoint_wins, total_predictions, total_wins, last_updated)
                VALUES (%s, %s, %s, 0, 0, 0, NOW())
                ON CONFLICT (symbol) DO UPDATE SET 
                    trust_level = %s,
                    consecutive_wins = %s,
                    last_updated = NOW()
            """, (symbol, level, level - 1, level, level - 1))  # consecutive_wins = level-1 to show progression
            
            # Verify the update
            cursor.execute("""
                SELECT symbol, trust_level, consecutive_wins, checkpoint_wins
                FROM ghost_symbol_trust
                WHERE symbol = %s
            """, (symbol,))
            
            row = cursor.fetchone()
            
            cursor.close()
            
            level_names = {1: "Standard (48hr)", 2: "Extended (60hr+120hr)", 3: "Focused (72hr+168hr)"}
            
            return {
                "ok": True,
                "symbol": symbol,
                "promoted_to": level,
                "level_name": level_names.get(level),
                "current_state": {
                    "symbol": row[0],
                    "trust_level": row[1],
                    "consecutive_wins": row[2],
                    "checkpoint_wins": row[3]
                } if row else None,
                "next_trade_will_have": f"checkpoint_times with {2 if level >= 2 else 1} checkpoints",
                "message": f"✅ {symbol} promoted to Level {level}. Next trade will use multi-checkpoint evaluation."
            }
        
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/create-test-trade")
async def debug_create_test_trade(symbol: str, direction: str = "UP", confidence: float = 0.75):
    """
    TEST ENDPOINT: Create a paper trade to test multi-checkpoint system.
    
    This bypasses the normal prediction flow and creates a trade directly.
    The trade will use the symbol's current trust level for checkpoint calculation.
    
    Args:
        symbol: Symbol to create trade for (e.g., BTC, ETH)
        direction: UP or DOWN
        confidence: Confidence level (0.0 to 1.0)
    """
    symbol = symbol.upper()
    direction = direction.upper()
    
    if direction not in ["UP", "DOWN"]:
        return {"error": f"Invalid direction {direction}. Must be UP or DOWN"}
    
    if confidence < 0 or confidence > 1:
        return {"error": f"Invalid confidence {confidence}. Must be 0.0 to 1.0"}
    
    try:
        from core.paper_tracker import get_paper_tracker
        from datetime import datetime
        import uuid
        
        # Get current price for the symbol (use fallback for testing)
        fallback_prices = {"BTC": 78000, "ETH": 2400, "SOL": 105, "SUI": 1.12, "NVDA": 192, "META": 718}
        current_price = fallback_prices.get(symbol, 100)
        
        paper_tracker = get_paper_tracker()
        
        paper_trade_id = paper_tracker.log_signal(
            cascade_id=f"test_{uuid.uuid4().hex[:8]}",
            symbol=symbol,
            signal_direction=direction,
            signal_confidence=confidence,
            entry_price=current_price,
            entry_time=datetime.utcnow().isoformat(),
            position_size=1000.0,
            stop_loss_pct=0.05,
            take_profit_pct=0.10
        )
        
        # Fetch the created trade to show checkpoint details
        from core.db_pool import get_sync_connection
        with get_sync_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT paper_trade_id, symbol, trust_level, entry_time, target_time,
                       checkpoint_times, checkpoint_results, checkpoint_evaluated
                FROM paper_trades
                WHERE paper_trade_id = %s
            """, (paper_trade_id,))
            
            row = cursor.fetchone()
            cursor.close()
        
        if row:
            return {
                "ok": True,
                "paper_trade_id": row[0],
                "symbol": row[1],
                "trust_level": row[2],
                "entry_time": str(row[3]),
                "target_time": str(row[4]),
                "checkpoint_times": row[5],
                "checkpoint_results": row[6],
                "checkpoint_evaluated": row[7],
                "checkpoint_count": len(row[5]) if row[5] else 0,
                "message": f"✅ Test trade created for {symbol} at Trust Level {row[2]}"
            }
        else:
            return {"ok": False, "error": "Trade created but could not fetch details"}
        
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/db-clean")
async def debug_db_clean(
    mode: str = "preview",
    confirm: str = "no"
):
    """
    Clean corrupt data from ghost_prediction_outcomes table.
    
    Modes:
        preview - Show what would be deleted (default, safe)
        corrupt - Delete outcomes with corrupt prices (BTC < $10k, etc.)
        no_data - Delete outcomes with status='no_data'
        all     - Delete both corrupt AND no_data records
    
    Requires confirm=yes for any destructive operation.
    
    Usage:
        /debug/db-clean?mode=preview              - Preview (safe)
        /debug/db-clean?mode=corrupt&confirm=yes  - Delete corrupt prices
        /debug/db-clean?mode=no_data&confirm=yes  - Delete no_data records
        /debug/db-clean?mode=all&confirm=yes      - Delete everything bad
    """
    try:
        from datetime import datetime
        
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            return {"ok": False, "error": "DATABASE_URL not set"}
        
        from core.db_pool import get_sync_connection
        with get_sync_connection() as conn:
            cursor = conn.cursor()
            
            # Price validation thresholds
            MIN_PRICES = {
                'BTC': 10000, 'ETH': 500, 'SOL': 5, 'BNB': 100,
                'XRP': 0.10, 'ADA': 0.05, 'DOGE': 0.001, 'AVAX': 5,
                'DOT': 1, 'LINK': 2, 'MATIC': 0.10, 'LTC': 20,
            }
            
            results = {
                "ok": True,
                "mode": mode,
                "confirm": confirm,
                "timestamp": datetime.utcnow().isoformat(),
                "actions": [],
                "deleted": {},
            }
            
            # Build corrupt price conditions
            corrupt_conditions = []
            for symbol, min_price in MIN_PRICES.items():
                corrupt_conditions.append(
                    f"(symbol = '{symbol}' AND (price_at_prediction < {min_price} OR price_at_resolution < {min_price}))"
                )
            corrupt_where = " OR ".join(corrupt_conditions)
            
            # ================================================================
            # PREVIEW MODE - Show what would be deleted
            # ================================================================
            if mode == "preview":
                # Count corrupt
                cursor.execute(f"""
                    SELECT COUNT(*) FROM ghost_prediction_outcomes
                    WHERE ({corrupt_where}) AND price_at_prediction IS NOT NULL
                """)
                corrupt_count = cursor.fetchone()[0]
                
                # Count no_data
                cursor.execute("""
                    SELECT COUNT(*) FROM ghost_prediction_outcomes
                    WHERE status = 'no_data' OR hit_direction IS NULL
                """)
                no_data_count = cursor.fetchone()[0]
                
                # Count total
                cursor.execute("SELECT COUNT(*) FROM ghost_prediction_outcomes")
                total_count = cursor.fetchone()[0]
                
                # Sample corrupt records
                cursor.execute(f"""
                    SELECT id, symbol, price_at_prediction, price_at_resolution, status
                    FROM ghost_prediction_outcomes
                    WHERE ({corrupt_where}) AND price_at_prediction IS NOT NULL
                    LIMIT 10
                """)
                corrupt_samples = [
                    {"id": r[0], "symbol": r[1], "entry": float(r[2]) if r[2] else 0, "exit": float(r[3]) if r[3] else 0, "status": r[4]}
                    for r in cursor.fetchall()
                ]
                
                return {
                    "ok": True,
                    "mode": "preview",
                    "preview": {
                        "total_outcomes": total_count,
                        "corrupt_prices": corrupt_count,
                        "no_data_status": no_data_count,
                        "would_remain_after_all": total_count - corrupt_count - no_data_count + (corrupt_count if no_data_count > corrupt_count else 0),
                        "corrupt_samples": corrupt_samples,
                    },
                    "actions": ["Preview complete - no changes made"],
                    "instructions": {
                        "to_delete_corrupt": "/debug/db-clean?mode=corrupt&confirm=yes",
                        "to_delete_no_data": "/debug/db-clean?mode=no_data&confirm=yes",
                        "to_delete_all": "/debug/db-clean?mode=all&confirm=yes",
                    }
                }
            
            # ================================================================
            # DESTRUCTIVE MODES - Require confirmation
            # ================================================================
            if confirm != "yes":
                return {
                    "ok": False,
                    "error": "Destructive operation requires confirm=yes parameter",
                    "instruction": f"Use: /debug/db-clean?mode={mode}&confirm=yes"
                }
            
            # ================================================================
            # DELETE CORRUPT PRICES
            # ================================================================
            if mode in ["corrupt", "all"]:
                cursor.execute(f"""
                    SELECT COUNT(*) FROM ghost_prediction_outcomes
                    WHERE ({corrupt_where}) AND price_at_prediction IS NOT NULL
                """)
                corrupt_count = cursor.fetchone()[0]
                
                if corrupt_count > 0:
                    cursor.execute(f"""
                        DELETE FROM ghost_prediction_outcomes
                        WHERE ({corrupt_where}) AND price_at_prediction IS NOT NULL
                    """)
                    deleted = cursor.rowcount
                    results["deleted"]["corrupt_prices"] = deleted
                    results["actions"].append(f"Deleted {deleted} outcomes with corrupt prices")
                    LOGGER.warning(f"[DB-CLEAN] Deleted {deleted} corrupt price outcomes")
                else:
                    results["actions"].append("No corrupt price records found")
            
            # ================================================================
            # DELETE NO_DATA STATUS
            # ================================================================
            if mode in ["no_data", "all"]:
                cursor.execute("""
                    SELECT COUNT(*) FROM ghost_prediction_outcomes
                    WHERE status = 'no_data' OR hit_direction IS NULL
                """)
                no_data_count = cursor.fetchone()[0]
                
                if no_data_count > 0:
                    cursor.execute("""
                        DELETE FROM ghost_prediction_outcomes
                        WHERE status = 'no_data' OR hit_direction IS NULL
                    """)
                    deleted = cursor.rowcount
                    results["deleted"]["no_data"] = deleted
                    results["actions"].append(f"Deleted {deleted} outcomes with no_data status")
                    LOGGER.warning(f"[DB-CLEAN] Deleted {deleted} no_data outcomes")
                else:
                    results["actions"].append("No no_data records found")
            
            # ================================================================
            # GET FINAL STATE (commit handled by context manager)
            # ================================================================
            # Get remaining count
            cursor.execute("SELECT COUNT(*) FROM ghost_prediction_outcomes")
            remaining = cursor.fetchone()[0]
            
            # Get accuracy of remaining
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN hit_direction = 0 THEN 1 ELSE 0 END) as losses
                FROM ghost_prediction_outcomes
                WHERE hit_direction IS NOT NULL
            """)
            acc = cursor.fetchone()
            
            results["after_cleanup"] = {
                "total_remaining": remaining,
                "with_outcome": acc[0] or 0,
                "wins": acc[1] or 0,
                "losses": acc[2] or 0,
                "accuracy_pct": round((acc[1] or 0) / max(1, acc[0] or 1) * 100, 2),
            }
            
            return results
        
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/paper-trades-clean")
async def debug_paper_trades_clean(
    mode: str = "preview",
    confirm: str = "no"
):
    """
    Clean corrupt paper trades (entry_price = 0, near-zero, or NULL).

    Modes:
        preview - Show corrupt trades (default, safe)
        delete  - Delete corrupt trades (requires confirm=yes)

    Usage:
        /debug/paper-trades-clean?mode=preview
        /debug/paper-trades-clean?mode=delete&confirm=yes
    """
    try:
        from datetime import datetime

        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            return {"ok": False, "error": "DATABASE_URL not set"}

        from core.db_pool import get_sync_connection
        with get_sync_connection() as conn:
            cursor = conn.cursor()

            corrupt_where = "entry_price IS NULL OR entry_price = 0"

            # Always show preview info
            cursor.execute(f"SELECT COUNT(*) FROM paper_trades WHERE {corrupt_where}")
            corrupt_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM paper_trades")
            total_count = cursor.fetchone()[0]

            cursor.execute(f"""
                SELECT paper_trade_id, symbol, signal_direction, entry_price, signal_confidence, created_at
                FROM paper_trades
                WHERE {corrupt_where}
                ORDER BY created_at DESC
                LIMIT 20
            """)
            samples = [
                {
                    "id": str(r[0]),
                    "symbol": r[1],
                    "direction": r[2],
                    "entry_price": float(r[3]) if r[3] else None,
                    "confidence": float(r[4]) if r[4] else None,
                    "created_at": str(r[5]) if r[5] else None,
                }
                for r in cursor.fetchall()
            ]

            if mode == "preview":
                return {
                    "ok": True,
                    "mode": "preview",
                    "total_paper_trades": total_count,
                    "corrupt_count": corrupt_count,
                    "corrupt_samples": samples,
                    "instruction": "/debug/paper-trades-clean?mode=delete&confirm=yes",
                }

            if mode == "delete":
                if confirm != "yes":
                    return {
                        "ok": False,
                        "error": "Requires confirm=yes",
                        "instruction": "/debug/paper-trades-clean?mode=delete&confirm=yes",
                    }

                cursor.execute(f"DELETE FROM paper_trades WHERE {corrupt_where}")
                deleted = cursor.rowcount

                cursor.execute("SELECT COUNT(*) FROM paper_trades")
                remaining = cursor.fetchone()[0]

                LOGGER.warning(f"[PAPER-TRADES-CLEAN] Deleted {deleted} corrupt paper trades")
                return {
                    "ok": True,
                    "mode": "delete",
                    "deleted": deleted,
                    "remaining": remaining,
                    "samples_deleted": samples,
                }

            return {"ok": False, "error": f"Unknown mode: {mode}"}

    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/tracked-picks")
async def debug_tracked_picks(symbol: str = ""):
    """
    Inspect ghost_tracked_picks table.
    Shows all active tracked picks, or filter by symbol.

    Usage:
        /debug/tracked-picks          — all active picks
        /debug/tracked-picks?symbol=GME — just GME
    """
    try:
        from core.db_pool import get_sync_connection
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            return {"ok": False, "error": "DATABASE_URL not set"}

        with get_sync_connection() as conn:
            cur = conn.cursor()

            where = "WHERE status = 'active'"
            params = ()
            if symbol:
                where = "WHERE symbol = %s"
                params = (symbol.upper().strip(),)

            cur.execute(f"""
                SELECT symbol, asset_type, direction, entry_price, target_price,
                       stop_price, confidence, entry_time, expires_at, status
                FROM ghost_tracked_picks
                {where}
                ORDER BY entry_time DESC
                LIMIT 50
            """, params)

            rows = cur.fetchall()
            cur.execute("SELECT COUNT(*) FROM ghost_tracked_picks WHERE status = 'active'")
            active_count = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM ghost_tracked_picks")
            total_count = cur.fetchone()[0]

            picks = []
            for r in rows:
                picks.append({
                    "symbol": r[0], "asset_type": r[1], "direction": r[2],
                    "entry_price": float(r[3]) if r[3] else None,
                    "target_price": float(r[4]) if r[4] else None,
                    "stop_price": float(r[5]) if r[5] else None,
                    "confidence": float(r[6]) if r[6] else None,
                    "entry_time": str(r[7]) if r[7] else None,
                    "expires_at": str(r[8]) if r[8] else None,
                    "status": r[9],
                })

            return {
                "ok": True,
                "active_count": active_count,
                "total_count": total_count,
                "filter": symbol.upper() if symbol else "all active",
                "picks": picks,
            }
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/paper-trades-lookup")
async def debug_paper_trades_lookup(symbol: str = "", limit: int = 20):
    """
    Look up paper trades by symbol. Returns entry_price, direction, timestamps.

    Usage:
        /debug/paper-trades-lookup?symbol=GIGA
        /debug/paper-trades-lookup?symbol=GIGA&limit=50
    """
    try:
        import psycopg2
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            return {"ok": False, "error": "DATABASE_URL not set"}
        if not symbol:
            return {"ok": False, "error": "symbol parameter required"}

        from core.db_pool import get_sync_connection
        with get_sync_connection() as conn:
            cur = conn.cursor()

            cur.execute("""
                SELECT paper_trade_id, symbol, signal_direction, entry_price,
                       signal_confidence, created_at, outcome, profit_loss_pct,
                       v3_validated, v3_strategy
                FROM paper_trades
                WHERE symbol = %s
                ORDER BY created_at DESC
                LIMIT %s
            """, (symbol.upper().strip(), limit))

            rows = cur.fetchall()
            cur.execute("SELECT COUNT(*) FROM paper_trades WHERE symbol = %s",
                        (symbol.upper().strip(),))
            total = cur.fetchone()[0]

            # Also run the exact corrupt check
            cur.execute("""
                SELECT COUNT(*) FROM paper_trades
                WHERE symbol = %s AND entry_price < 0.01
            """, (symbol.upper().strip(),))
            suspect_count = cur.fetchone()[0]

            trades = []
            for r in rows:
                trades.append({
                    "id": str(r[0])[:12], "symbol": r[1], "direction": r[2],
                    "entry_price": float(r[3]) if r[3] else None,
                    "confidence": float(r[4]) if r[4] else None,
                    "created_at": str(r[5]) if r[5] else None,
                    "outcome": r[6], "pnl_pct": float(r[7]) if r[7] else None,
                    "v3": r[8], "strategy": r[9],
                })

            return {
                "ok": True,
                "symbol": symbol.upper(),
                "total_trades": total,
                "suspect_below_001": suspect_count,
                "showing": len(trades),
                "trades": trades,
            }
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/revert-false-stops")
async def debug_revert_false_stops(confirm: str = "no"):
    """
    One-time fix: Revert JUP/BAND/GME from stop_hit back to active.
    These were falsely triggered by the near_stop 2% buffer bug (Feb 12, 2026).
    
    Usage:
        /debug/revert-false-stops           → preview
        /debug/revert-false-stops?confirm=yes → execute
    """
    import os
    from core.db_pool import get_sync_connection
    symbols = ["JUP", "BAND", "GME"]
    try:
        with get_sync_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT symbol, status, entry_price, stop_price FROM ghost_tracked_picks WHERE symbol = ANY(%s) AND status = 'stop_hit'",
                (symbols,)
            )
            rows = cur.fetchall()
            preview = [{"symbol": r[0], "status": r[1], "entry": float(r[2]), "stop": float(r[3])} for r in rows]
            if confirm == "yes":
                cur.execute(
                    "UPDATE ghost_tracked_picks SET status = 'active' WHERE symbol = ANY(%s) AND status = 'stop_hit'",
                    (symbols,)
                )
                reverted = cur.rowcount
                cur.close()
                return {"ok": True, "reverted": reverted, "symbols": symbols}
            cur.close()
            return {"ok": True, "preview": preview, "count": len(preview), "note": "Add ?confirm=yes to execute"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/debug/fix-watch-zombies")
async def debug_fix_watch_zombies(confirm: str = "no"):
    """
    Fix WATCH direction zombie picks by coercing to BUY/SELL.
    Direction is determined by target_price vs entry_price.
    
    Usage:
        /debug/fix-watch-zombies           → preview
        /debug/fix-watch-zombies?confirm=yes → execute
    """
    import os
    from core.db_pool import get_sync_connection
    try:
        with get_sync_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT symbol, entry_price, target_price FROM ghost_tracked_picks WHERE status = 'active' AND direction = 'WATCH'"
            )
            rows = cur.fetchall()
            fixes = []
            for sym, entry, target in rows:
                correct = "BUY" if float(target) >= float(entry) else "SELL"
                fixes.append({"symbol": sym, "from": "WATCH", "to": correct, "entry": float(entry), "target": float(target)})
            if confirm == "yes":
                for f in fixes:
                    cur.execute(
                        "UPDATE ghost_tracked_picks SET direction = %s WHERE symbol = %s AND status = 'active' AND direction = 'WATCH'",
                        (f["to"], f["symbol"])
                    )
                cur.close()
                return {"ok": True, "fixed": len(fixes), "fixes": fixes}
            cur.close()
            return {"ok": True, "preview": fixes, "count": len(fixes), "note": "Add ?confirm=yes to execute"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/debug/db-reset-accuracy")
async def debug_db_reset_accuracy(confirm: str = "no"):
    """
    Reset the ghost_symbol_accuracy table.
    The reconciler will rebuild it from clean outcome data.
    
    Requires confirm=yes parameter.
    """
    try:
        from core.db_pool import get_sync_connection
        
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            return {"ok": False, "error": "DATABASE_URL not set"}
        
        if confirm != "yes":
            return {
                "ok": False,
                "warning": "This will TRUNCATE ghost_symbol_accuracy table",
                "instruction": "Use /debug/db-reset-accuracy?confirm=yes to proceed"
            }
        
        with get_sync_connection() as conn:
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'ghost_symbol_accuracy'
                )
            """)
            exists = cursor.fetchone()[0]
            
            if not exists:
                return {"ok": True, "message": "Table ghost_symbol_accuracy does not exist - nothing to reset"}
            
            # Get count before
            cursor.execute("SELECT COUNT(*) FROM ghost_symbol_accuracy")
            before = cursor.fetchone()[0]
            
            # Truncate
            cursor.execute("TRUNCATE ghost_symbol_accuracy")
            
            return {
                "ok": True,
                "deleted_rows": before,
                "message": "Symbol accuracy table reset. Reconciler will rebuild from outcomes."
            }
        
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/pg-health")
async def debug_pg_health():
    """
    Quick PostgreSQL table health: row counts for ghost_predictions,
    price_actuals, and ghost_accuracy_stats, plus newest/oldest timestamps.
    """
    try:
        from core.db_pool import get_sync_connection
        with get_sync_connection() as conn:
            cur = conn.cursor()
            result = {}
            # ghost_predictions
            cur.execute("SELECT COUNT(*) FROM ghost_predictions")
            total_preds = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM ghost_predictions WHERE checked = 0")
            unchecked = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM ghost_predictions WHERE checked = 1")
            checked = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM ghost_predictions WHERE correct = 1")
            correct = cur.fetchone()[0]
            # Real accuracy: exclude skipped evaluations
            cur.execute("SELECT COUNT(*) FROM ghost_predictions WHERE checked = 1 AND eval_version NOT LIKE 'skip%%'")
            checked_real = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM ghost_predictions WHERE checked = 1 AND correct = 1 AND eval_version NOT LIKE 'skip%%'")
            correct_real = cur.fetchone()[0]
            real_accuracy = round(correct_real / checked_real * 100, 1) if checked_real > 0 else 0
            cur.execute("SELECT MIN(predicted_at), MAX(predicted_at) FROM ghost_predictions")
            pmin, pmax = cur.fetchone()
            result["ghost_predictions"] = {
                "total": total_preds,
                "unchecked": unchecked,
                "checked": checked,
                "correct": correct,
                "real_accuracy_pct": real_accuracy,
                "real_checked": checked_real,
                "real_correct": correct_real,
                "oldest_ts": pmin,
                "newest_ts": pmax,
                "oldest_age_h": round((time.time() - pmin) / 3600, 1) if pmin else None,
                "newest_age_h": round((time.time() - pmax) / 3600, 1) if pmax else None,
            }
            # price_actuals
            cur.execute("SELECT COUNT(*) FROM price_actuals")
            pa_total = cur.fetchone()[0]
            cur.execute("SELECT COUNT(DISTINCT symbol) FROM price_actuals")
            pa_symbols = cur.fetchone()[0]
            cur.execute("SELECT MIN(ts), MAX(ts) FROM price_actuals")
            amin, amax = cur.fetchone()
            result["price_actuals"] = {
                "total_rows": pa_total,
                "unique_symbols": pa_symbols,
                "oldest_ts": amin,
                "newest_ts": amax,
                "oldest_age_h": round((time.time() - amin) / 3600, 1) if amin else None,
                "newest_age_h": round((time.time() - amax) / 3600, 1) if amax else None,
            }
            # Recent price_actuals by symbol
            cur.execute("""
                SELECT symbol, COUNT(*), MAX(ts)
                FROM price_actuals
                GROUP BY symbol
                ORDER BY MAX(ts) DESC
            """)
            by_sym = []
            for sym, cnt, latest in cur.fetchall():
                by_sym.append({
                    "symbol": sym, "count": cnt,
                    "latest_ts": latest,
                    "age_min": round((time.time() - latest) / 60, 1) if latest else None,
                })
            result["price_actuals_by_symbol"] = by_sym
            # ghost_accuracy_stats
            try:
                cur.execute("SELECT period, total_predictions, correct_predictions, accuracy_pct FROM ghost_accuracy_stats ORDER BY period")
                stats = [{"period": p, "total": t, "correct": c, "pct": a} for p, t, c, a in cur.fetchall()]
                result["accuracy_stats"] = stats
            except Exception:
                result["accuracy_stats"] = "table_missing"
            return {"ok": True, **result}
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.post("/debug/reset-bad-evaluations")
async def debug_reset_bad_evaluations():
    """
    ONE-TIME CLEANUP: Reset predictions that were evaluated with single-point
    fallback (window_first = window_last AND window_high = window_low).
    These produced 19.5% accuracy vs 90% with real price windows.
    After reset, the evaluator will re-evaluate with proper window data
    or permanently skip if too old (>7 days).
    """
    try:
        from core.db_pool import get_sync_connection
        with get_sync_connection() as conn:
            cur = conn.cursor()

            # Count single-point evaluations before reset
            cur.execute("""
                SELECT COUNT(*) FROM ghost_predictions
                WHERE checked = 1
                  AND window_first IS NOT NULL
                  AND window_first = window_last
                  AND window_high = window_low
            """)
            bad_count = cur.fetchone()[0]

            if bad_count == 0:
                return {"ok": True, "message": "No single-point evaluations found", "reset": 0}

            # Reset them to unchecked so evaluator can re-process
            cur.execute("""
                UPDATE ghost_predictions
                SET checked = 0,
                    checked_at = NULL,
                    correct = NULL,
                    outcome_price = NULL,
                    outcome_pct = NULL,
                    outcome_direction = NULL,
                    window_first = NULL,
                    window_last = NULL,
                    window_high = NULL,
                    window_low = NULL,
                    touch_1pct = NULL,
                    error_pct = NULL,
                    eval_version = NULL
                WHERE checked = 1
                  AND window_first IS NOT NULL
                  AND window_first = window_last
                  AND window_high = window_low
            """)
            reset_count = cur.rowcount
            conn.commit()

            # Get updated accuracy stats
            cur.execute("SELECT COUNT(*) FROM ghost_predictions WHERE checked = 1 AND eval_version NOT LIKE 'skip%%'")
            remaining_checked = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM ghost_predictions WHERE checked = 1 AND correct = 1 AND eval_version NOT LIKE 'skip%%'")
            remaining_correct = cur.fetchone()[0]
            new_accuracy = round(remaining_correct / remaining_checked * 100, 1) if remaining_checked > 0 else 0

            return {
                "ok": True,
                "message": f"Reset {reset_count} single-point evaluations",
                "reset": reset_count,
                "remaining_checked": remaining_checked,
                "remaining_correct": remaining_correct,
                "new_accuracy_pct": new_accuracy,
            }
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.post("/debug/reset-all-evaluations")
async def debug_reset_all_evaluations():
    """
    Reset ALL checked predictions to unchecked so evaluator can re-run
    with the fixed correctness metric (actual direction match, flat-market skip).
    """
    try:
        from core.db_pool import get_sync_connection
        with get_sync_connection() as conn:
            cur = conn.cursor()

            cur.execute("SELECT COUNT(*) FROM ghost_predictions WHERE checked = 1")
            total_checked = cur.fetchone()[0]

            cur.execute("""
                UPDATE ghost_predictions
                SET checked = 0,
                    checked_at = NULL,
                    correct = NULL,
                    outcome_price = NULL,
                    outcome_pct = NULL,
                    outcome_direction = NULL,
                    window_first = NULL,
                    window_last = NULL,
                    window_high = NULL,
                    window_low = NULL,
                    touch_1pct = NULL,
                    touch_0_5pct = NULL,
                    correct_1pct = NULL,
                    correct_0_5pct = NULL,
                    direction_consistent = NULL,
                    error_pct = NULL,
                    eval_version = NULL
                WHERE checked = 1
            """)
            reset_count = cur.rowcount

            # Clear stale accuracy stats so they don't mislead
            cur.execute("DELETE FROM ghost_accuracy_stats")
            conn.commit()

            return {
                "ok": True,
                "message": f"Reset {reset_count} evaluations for re-evaluation with fixed metric",
                "reset": reset_count,
                "accuracy_stats_cleared": True,
            }
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.post("/debug/backfill-price-actuals")
async def debug_backfill_price_actuals():
    """
    Backfill price_actuals from SQLite → PostgreSQL.

    The prediction_evaluator reads price data from PostgreSQL, but prices
    were only written to SQLite before March 7 2026. This endpoint copies
    all SQLite price_actuals rows into PostgreSQL so that skipped predictions
    (skip-no-window-v2) can be re-evaluated with real price data.

    After running this, call POST /debug/reset-skipped-for-reevaluation
    to reset the skipped predictions so the evaluator retries them.
    """
    import sqlite3 as _sqlite3
    try:
        from core.db_pool import get_sync_connection as _bf_get_conn

        sqlite_path = os.getenv("WOLF_SQLITE_PATH", "data/wolf.db")
        if not os.path.exists(sqlite_path):
            return {"ok": False, "error": f"SQLite DB not found at {sqlite_path}"}

        # Read all price_actuals from SQLite
        with _sqlite3.connect(sqlite_path) as sq_conn:
            rows = sq_conn.execute("SELECT ts, symbol, price FROM price_actuals WHERE price IS NOT NULL").fetchall()

        if not rows:
            return {"ok": True, "message": "No price_actuals found in SQLite", "backfilled": 0}

        # Insert into PostgreSQL (skip duplicates)
        inserted = 0
        with _bf_get_conn() as pg_conn:
            pg_cur = pg_conn.cursor()
            # Ensure table + unique index exist
            pg_cur.execute("""
                CREATE TABLE IF NOT EXISTS price_actuals (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    ts BIGINT NOT NULL,
                    price DOUBLE PRECISION
                )
            """)
            try:
                pg_cur.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_price_actuals_sym_ts_uniq
                    ON price_actuals (symbol, ts)
                """)
            except Exception:
                pg_conn.rollback()

            BATCH_SIZE = 500
            for i in range(0, len(rows), BATCH_SIZE):
                batch = rows[i:i + BATCH_SIZE]
                for ts, symbol, price in batch:
                    try:
                        pg_cur.execute(
                            """INSERT INTO price_actuals (ts, symbol, price)
                               VALUES (%s, %s, %s)
                               ON CONFLICT (symbol, ts) DO NOTHING""",
                            (int(ts), str(symbol), float(price)),
                        )
                        if pg_cur.rowcount > 0:
                            inserted += 1
                    except Exception:
                        pg_conn.rollback()  # FIX: reset transaction state so next INSERT can work
                pg_conn.commit()

        return {
            "ok": True,
            "sqlite_rows": len(rows),
            "pg_inserted": inserted,
            "pg_skipped_duplicates": len(rows) - inserted,
            "message": f"Backfilled {inserted} price rows from SQLite → PostgreSQL. Now call POST /debug/reset-skipped-for-reevaluation",
        }
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.post("/debug/reset-skipped-for-reevaluation")
async def debug_reset_skipped_for_reevaluation():
    """
    Reset predictions that were skipped due to missing price data
    (eval_version = 'skip-no-window-v2') so the evaluator retries them.

    Only resets predictions that now have >= 5 price_actuals rows in PostgreSQL
    for their evaluation window, so we don't just skip them again.

    Also clears stale ghost_accuracy_stats so accuracy reflects reality.
    """
    try:
        from core.db_pool import get_sync_connection as _rs_get_conn
        with _rs_get_conn() as conn:
            cur = conn.cursor()
            now = int(time.time())

            # Find skip-no-window predictions that now have enough price data
            cur.execute("""
                SELECT p.id, p.symbol, p.predicted_at, p.check_at
                FROM ghost_predictions p
                WHERE p.checked = 1
                  AND p.eval_version = 'skip-no-window-v2'
            """)
            candidates = cur.fetchall()

            reset_ids = []
            for pred_id, symbol, pred_at, check_at in candidates:
                cur.execute("""
                    SELECT COUNT(*) FROM price_actuals
                    WHERE symbol = %s AND ts >= %s AND ts <= %s
                """, (symbol, int(pred_at), int(check_at)))
                price_count = cur.fetchone()[0]
                if price_count >= 5:
                    reset_ids.append(pred_id)

            # Reset those predictions
            reset_count = 0
            for pred_id in reset_ids:
                cur.execute("""
                    UPDATE ghost_predictions
                    SET checked = 0,
                        checked_at = NULL,
                        correct = NULL,
                        outcome_price = NULL,
                        outcome_pct = NULL,
                        outcome_direction = NULL,
                        window_first = NULL,
                        window_last = NULL,
                        window_high = NULL,
                        window_low = NULL,
                        touch_1pct = NULL,
                        touch_0_5pct = NULL,
                        correct_1pct = NULL,
                        correct_0_5pct = NULL,
                        direction_consistent = NULL,
                        error_pct = NULL,
                        eval_version = NULL
                    WHERE id = %s
                """, (pred_id,))
                reset_count += cur.rowcount

            # Clear stale ghost_accuracy_stats so it gets refreshed on next eval
            cur.execute("DELETE FROM ghost_accuracy_stats")
            conn.commit()

            # Count remaining state
            cur.execute("SELECT COUNT(*) FROM ghost_predictions WHERE checked = 0")
            new_unchecked = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM ghost_predictions WHERE checked = 1 AND eval_version = 'skip-no-window-v2'")
            still_skipped = cur.fetchone()[0]

            return {
                "ok": True,
                "candidates_checked": len(candidates),
                "had_enough_prices": len(reset_ids),
                "reset_for_reevaluation": reset_count,
                "still_skipped_no_data": still_skipped,
                "total_unchecked_now": new_unchecked,
                "accuracy_stats_cleared": True,
                "message": f"Reset {reset_count} predictions for re-evaluation. {still_skipped} still lack price data. Run evaluator to re-evaluate.",
            }
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.post("/debug/skip-pre-fix-predictions")
async def debug_skip_pre_fix_predictions():
    """
    Mark all predictions made BEFORE the price data fix (March 7 2026)
    as permanently skipped. These were generated with broken features
    (21/53 defaulting), wrong direction overrides, and corrupt target prices.
    Only post-fix predictions will be evaluated for accuracy.
    """
    FIX_TIMESTAMP = 1772896000  # March 7 2026 ~14:00 UTC
    try:
        import time as _t
        from core.db_pool import get_sync_connection
        with get_sync_connection() as conn:
            cur = conn.cursor()
            now = int(_t.time())

            # Count pre-fix predictions
            cur.execute("SELECT COUNT(*) FROM ghost_predictions WHERE predicted_at < %s", (FIX_TIMESTAMP,))
            pre_fix_count = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM ghost_predictions WHERE predicted_at < %s AND checked = 0", (FIX_TIMESTAMP,))
            unchecked_pre_fix = cur.fetchone()[0]

            # Skip all pre-fix predictions (mark as checked with skip version)
            cur.execute("""
                UPDATE ghost_predictions
                SET checked = 1,
                    checked_at = %s,
                    eval_version = 'skip-pre-fix-v1'
                WHERE predicted_at < %s AND eval_version IS DISTINCT FROM 'skip-pre-fix-v1'
            """, (now, FIX_TIMESTAMP))
            skipped_count = cur.rowcount
            conn.commit()

            # Get updated accuracy (post-fix only)
            cur.execute("SELECT COUNT(*) FROM ghost_predictions WHERE checked = 1 AND eval_version NOT LIKE 'skip%%'")
            real_checked = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM ghost_predictions WHERE checked = 1 AND correct = 1 AND eval_version NOT LIKE 'skip%%'")
            real_correct = cur.fetchone()[0]
            new_accuracy = round(real_correct / real_checked * 100, 1) if real_checked > 0 else 0

            return {
                "ok": True,
                "pre_fix_total": pre_fix_count,
                "newly_skipped": skipped_count,
                "post_fix_checked": real_checked,
                "post_fix_correct": real_correct,
                "post_fix_accuracy_pct": new_accuracy,
                "message": f"Skipped {skipped_count} pre-fix predictions. Post-fix accuracy: {new_accuracy}% ({real_correct}/{real_checked})",
            }
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/pg-accuracy-breakdown")
async def debug_pg_accuracy_breakdown():
    """
    Deep accuracy breakdown for ghost_predictions:
    - Per-symbol accuracy
    - Eval version breakdown (single-point fallback vs real window)
    - Pre-fix vs post-fix accuracy split
    """
    FIX_TIMESTAMP = 1772896000  # March 7 2026 ~14:00 UTC - price data fix deployed
    try:
        from core.db_pool import get_sync_connection
        with get_sync_connection() as conn:
            cur = conn.cursor()
            result: dict = {}

            # 1) Per-symbol accuracy
            cur.execute("""
                SELECT symbol,
                       COUNT(*) AS total,
                       SUM(CASE WHEN checked=1 THEN 1 ELSE 0 END) AS checked,
                       SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) AS correct,
                       SUM(CASE WHEN checked=0 THEN 1 ELSE 0 END) AS pending
                FROM ghost_predictions GROUP BY symbol ORDER BY total DESC
            """)
            per_symbol = []
            for sym, total, checked, correct, pending in cur.fetchall():
                acc = round(correct / checked * 100, 1) if checked > 0 else None
                per_symbol.append({"symbol": sym, "total": total, "checked": checked,
                                   "correct": correct, "pending": pending, "accuracy_pct": acc})
            result["per_symbol"] = per_symbol

            # 2) Eval version breakdown
            cur.execute("""
                SELECT eval_version,
                       COUNT(*) AS cnt,
                       SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) AS correct
                FROM ghost_predictions WHERE checked=1
                GROUP BY eval_version ORDER BY cnt DESC
            """)
            eval_versions = []
            for ver, cnt, correct in cur.fetchall():
                eval_versions.append({"version": ver, "count": cnt, "correct": correct,
                                      "accuracy_pct": round(correct / cnt * 100, 1) if cnt > 0 else 0})
            result["eval_versions"] = eval_versions

            # 3) Window data quality
            cur.execute("""
                SELECT
                  COUNT(*) AS total_checked,
                  SUM(CASE WHEN window_first IS NOT NULL AND window_high IS NOT NULL
                            AND window_first != window_last THEN 1 ELSE 0 END) AS real_window,
                  SUM(CASE WHEN window_first IS NOT NULL AND window_high IS NOT NULL
                            AND window_first = window_last AND window_high = window_low
                       THEN 1 ELSE 0 END) AS single_point,
                  SUM(CASE WHEN window_first IS NULL THEN 1 ELSE 0 END) AS no_window
                FROM ghost_predictions WHERE checked=1
            """)
            row = cur.fetchone()
            result["window_quality"] = {
                "total_checked": row[0], "real_window": row[1],
                "single_point_fallback": row[2], "no_window_data": row[3],
            }

            # 4) Accuracy by window quality
            cur.execute("""
                SELECT
                  CASE
                    WHEN window_first IS NOT NULL AND window_first = window_last
                         AND window_high = window_low THEN 'single_point'
                    WHEN window_first IS NOT NULL AND window_first != window_last
                         THEN 'real_window'
                    ELSE 'no_data'
                  END AS quality,
                  COUNT(*) AS cnt,
                  SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) AS correct
                FROM ghost_predictions WHERE checked=1 GROUP BY quality
            """)
            acc_by_quality = {}
            for quality, cnt, correct in cur.fetchall():
                acc_by_quality[quality] = {
                    "count": cnt, "correct": correct,
                    "accuracy_pct": round(correct / cnt * 100, 1) if cnt > 0 else 0}
            result["accuracy_by_window_quality"] = acc_by_quality

            # 5) Pre-fix vs post-fix (based on predicted_at timestamp)
            cur.execute("""
                SELECT
                  CASE WHEN predicted_at < %s THEN 'pre_fix' ELSE 'post_fix' END AS era,
                  COUNT(*) AS total,
                  SUM(CASE WHEN checked=1 THEN 1 ELSE 0 END) AS checked,
                  SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) AS correct
                FROM ghost_predictions GROUP BY era
            """, (FIX_TIMESTAMP,))
            era_split = {}
            for era, total, checked, correct in cur.fetchall():
                era_split[era] = {
                    "total": total, "checked": checked, "correct": correct,
                    "accuracy_pct": round(correct / checked * 100, 1) if checked > 0 else None}
            result["pre_vs_post_fix"] = era_split

            # 6) Pre-fix predictions: how many had window data?
            cur.execute("""
                SELECT
                  SUM(CASE WHEN window_first IS NOT NULL AND window_first != window_last
                       THEN 1 ELSE 0 END) AS real_window,
                  SUM(CASE WHEN window_first IS NOT NULL AND window_first = window_last
                            AND window_high = window_low THEN 1 ELSE 0 END) AS single_point,
                  SUM(CASE WHEN window_first IS NULL THEN 1 ELSE 0 END) AS no_window
                FROM ghost_predictions
                WHERE checked=1 AND predicted_at < %s
            """, (FIX_TIMESTAMP,))
            row = cur.fetchone()
            result["pre_fix_window_quality"] = {
                "real_window": row[0] or 0, "single_point_fallback": row[1] or 0,
                "no_window_data": row[2] or 0}

            # 7) Direction-consistent breakdown
            cur.execute("""
                SELECT
                  direction_consistent,
                  COUNT(*) AS cnt,
                  SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) AS correct,
                  AVG(ABS(outcome_pct)) AS avg_move
                FROM ghost_predictions WHERE checked=1 AND eval_version != 'skip-v1'
                GROUP BY direction_consistent
            """)
            dir_breakdown = {}
            for dc, cnt, correct, avg_move in cur.fetchall():
                label = "direction_correct" if dc == 1 else "direction_wrong"
                dir_breakdown[label] = {
                    "count": cnt, "correct": correct,
                    "accuracy_pct": round(correct / cnt * 100, 1) if cnt > 0 else 0,
                    "avg_abs_move_pct": round(avg_move, 2) if avg_move else 0}
            result["direction_breakdown"] = dir_breakdown

            # 8) Outcome pct distribution (why 1% threshold matters)
            cur.execute("""
                SELECT
                  CASE
                    WHEN ABS(outcome_pct) < 0.5 THEN '<0.5pct'
                    WHEN ABS(outcome_pct) < 1.0 THEN '0.5-1pct'
                    WHEN ABS(outcome_pct) < 2.0 THEN '1-2pct'
                    WHEN ABS(outcome_pct) < 5.0 THEN '2-5pct'
                    ELSE '>5pct'
                  END AS move_bucket,
                  COUNT(*) AS cnt,
                  SUM(CASE WHEN direction_consistent=1 THEN 1 ELSE 0 END) AS dir_correct,
                  SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) AS correct
                FROM ghost_predictions
                WHERE checked=1 AND eval_version != 'skip-v1'
                GROUP BY move_bucket ORDER BY move_bucket
            """)
            move_dist = []
            for bucket, cnt, dir_correct, correct in cur.fetchall():
                move_dist.append({"bucket": bucket, "count": cnt,
                                  "direction_correct": dir_correct, "correct": correct})
            result["outcome_move_distribution"] = move_dist

            result["fix_timestamp"] = FIX_TIMESTAMP
            result["fix_note"] = "Price data fix deployed at b5d837c. Predictions before this had no price_actuals window data."
            return {"ok": True, **result}
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/model-status")
async def debug_model_status(secret: str = ""):
    """
    Check if XGBoost model is loaded and its stats.
    """
    if secret != os.getenv("CRON_SECRET", "ghost-cron-2024"):
        return {"error": "Invalid secret"}
    
    try:
        from pathlib import Path
        import pickle
        
        # Check multiple possible paths
        paths_to_check = [
            Path("/app/models/trained/ghost_xgboost_v2.pkl"),
            Path("/app/models/trained/ghost_xgboost_v1.pkl"),
            Path(__file__).parent / "models" / "trained" / "ghost_xgboost_v2.pkl",
            Path(__file__).parent / "models" / "trained" / "ghost_xgboost_v1.pkl",
        ]
        
        paths_checked = {str(p): p.exists() for p in paths_to_check}
        
        # Find the first existing path
        model_path = None
        for p in paths_to_check:
            if p.exists():
                model_path = p
                break
        
        # Try to actually load the model
        model_load_result = {"success": False}
        if model_path:
            try:
                with open(model_path, "rb") as f:
                    model_data = pickle.load(f)
                
                model_load_result = {
                    "success": True,
                    "path": str(model_path),
                    "has_model": "model" in model_data,
                    "feature_count": len(model_data.get("feature_names", [])),
                    "test_accuracy": model_data.get("test_accuracy"),
                    "cv_score": model_data.get("cv_score"),
                    "version": "v2" if "v2" in str(model_path) else "v1",
                }
            except Exception as e:
                model_load_result = {"success": False, "error": str(e)}
        
        # Also try creating XGBoostModel instance
        instance_result = {"success": False}
        try:
            from core.ensemble_predictor import XGBoostModel
            xgb = XGBoostModel()
            instance_result = {
                "success": True,
                "loaded": xgb._loaded,
                "version": xgb.model_version,
                "features_count": len(xgb.feature_names) if xgb.feature_names else 0,
                "has_model": xgb.model is not None,
            }
        except Exception as e:
            instance_result = {"success": False, "error": str(e)}
        
        return {
            "ok": True,
            "paths_checked": paths_checked,
            "model_load_result": model_load_result,
            "xgboost_instance": instance_result,
            "verdict": "✅ MODEL LOADED" if instance_result.get("loaded") or model_load_result.get("success") else "⚠️ MODEL NOT LOADED"
        }
        
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/fear-greed")
async def debug_fear_greed(secret: str = ""):
    """
    Debug endpoint to check Fear & Greed Index integration.
    Shows current value, trading signal, and confidence modifier.
    """
    if secret != os.getenv("CRON_SECRET", "ghost-cron-2024"):
        return {"error": "Invalid secret"}
    
    try:
        from core.ensemble_predictor import get_fear_greed_info, get_fear_greed_index
        
        # Force a fresh fetch by getting the value
        current_value = get_fear_greed_index()
        info = get_fear_greed_info()
        
        # Add interpretation
        interpretation = ""
        if current_value < 20:
            interpretation = "🔥 EXTREME FEAR - Strong BUY signal (contrarian)"
        elif current_value < 40:
            interpretation = "😰 FEAR - Slight bullish bias"
        elif current_value > 80:
            interpretation = "🚀 EXTREME GREED - Strong SELL signal (contrarian)"
        elif current_value > 60:
            interpretation = "🤑 GREED - Slight bearish bias"
        else:
            interpretation = "😐 NEUTRAL - No signal"
        
        return {
            "ok": True,
            "fear_greed_index": current_value,
            "classification": info.get("classification", "Unknown"),
            "trading_signal": info.get("signal"),
            "confidence_modifier": info.get("confidence_modifier"),
            "interpretation": interpretation,
            "cached_at": info.get("cached_at"),
            "data_source": "https://api.alternative.me/fng/",
            "strategy": "CONTRARIAN: Fear=Buy, Greed=Sell"
        }
        
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/btc-trend")
async def debug_btc_trend(secret: str = "", symbol: str = "BTC"):
    """
    Debug endpoint to check BTC trend and correlation boost.
    Shows current BTC price, trend, and correlation boost for a symbol.
    """
    if secret != os.getenv("CRON_SECRET", "ghost-cron-2024"):
        return {"error": "Invalid secret"}
    
    try:
        from core.ensemble_predictor import (
            get_btc_trend_info, 
            get_btc_correlation_boost,
            BTC_CORRELATED_SYMBOLS
        )
        
        info = get_btc_trend_info()
        
        # Calculate boost for the given symbol
        up_boost = get_btc_correlation_boost(symbol, "UP")
        down_boost = get_btc_correlation_boost(symbol, "DOWN")
        
        # Check if symbol is crypto
        symbol_upper = symbol.upper().replace("-", "").replace("/", "").replace("USD", "")
        is_crypto = any(s in symbol_upper or symbol_upper in s for s in BTC_CORRELATED_SYMBOLS)
        
        return {
            "ok": True,
            "btc_price": f"${info.get('price', 0):,.0f}",
            "btc_trend": info.get("trend"),
            "btc_1h_change": f"{info.get('change_1h', 0):+.2f}%",
            "btc_24h_change": f"{info.get('change_24h', 0):+.2f}%",
            "symbol_tested": symbol,
            "is_crypto": is_crypto,
            "boost_if_UP": f"{(up_boost - 1) * 100:+.1f}%" if up_boost != 1.0 else "No boost",
            "boost_if_DOWN": f"{(down_boost - 1) * 100:+.1f}%" if down_boost != 1.0 else "No boost",
            "cached_at": info.get("cached_at"),
            "data_source": "https://api.coingecko.com/",
            "correlated_symbols_count": info.get("correlated_symbols", 0)
        }
        
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/movers-scanner")
async def debug_movers_scanner(secret: str = ""):
    """
    Debug endpoint to check real-time market movers scanner status.
    Shows discovered symbols today, scanner settings, and manual trigger.
    """
    if secret != os.getenv("CRON_SECRET", "ghost-cron-2024"):
        return {"error": "Invalid secret"}
    
    try:
        from core.realtime_market_movers import get_scanner_status
        
        status = get_scanner_status()
        
        return {
            "ok": True,
            "scanner_status": status,
            "description": "Scans Yahoo Finance + CoinGecko for stocks/crypto moving 3%+ today",
            "endpoints": {
                "manual_scan": "/api/movers/scan",
                "discovered_today": "/api/movers/discovered"
            }
        }
        
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/stock-status")
async def debug_stock_status(secret: str = ""):
    """
    Debug endpoint to diagnose stock prediction issues.
    Shows market hours status, timezone info, and stock counts.
    """
    if secret != os.getenv("CRON_SECRET", "ghost-cron-2024"):
        return {"error": "Invalid secret"}
    
    try:
        from datetime import datetime
        from zoneinfo import ZoneInfo
        
        # Get current times in different timezones
        utc_now = datetime.now(ZoneInfo("UTC"))
        et_now = datetime.now(ZoneInfo("America/New_York"))
        ct_now = datetime.now(ZoneInfo("America/Chicago"))
        
        # Check market hours (9:30 AM - 4:00 PM ET)
        et_hour = et_now.hour
        et_minute = et_now.minute
        is_after_open = et_hour > 9 or (et_hour == 9 and et_minute >= 30)
        is_before_close = et_hour < 16
        is_weekday = et_now.weekday() < 5
        is_market_hours = is_weekday and is_after_open and is_before_close
        
        # Get stock counts
        from core.auto_prediction_loop import HUNTER_STOCK_SYMBOLS, FORCE_STOCK_PREDICTIONS
        stock_count = len(HUNTER_STOCK_SYMBOLS)
        
        # Get latest stock predictions
        stock_predictions = [
            sym for sym, pred in _LATEST_PREDICTIONS.items()
            if sym in HUNTER_STOCK_SYMBOLS
        ]
        
        # Diagnosis
        if not is_weekday:
            diagnosis = "❌ Weekend - markets closed"
        elif not is_market_hours and not FORCE_STOCK_PREDICTIONS:
            diagnosis = "❌ Outside market hours (9:30 AM - 4 PM ET) and FORCE_STOCK_PREDICTIONS=0"
        elif not is_market_hours and FORCE_STOCK_PREDICTIONS:
            diagnosis = "✅ Outside market hours but FORCE_STOCK_PREDICTIONS=1 - stocks will process"
        elif is_market_hours:
            diagnosis = "✅ Market hours - stocks should process normally"
        else:
            diagnosis = "❓ Unknown state"
        
        return {
            "ok": True,
            "times": {
                "utc": utc_now.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "et": et_now.strftime("%Y-%m-%d %H:%M:%S %Z"),
                "ct": ct_now.strftime("%Y-%m-%d %H:%M:%S %Z"),
            },
            "market_status": {
                "is_market_hours": is_market_hours,
                "is_weekday": is_weekday,
                "et_time": et_now.strftime("%H:%M"),
                "market_open": "09:30 ET",
                "market_close": "16:00 ET",
            },
            "config": {
                "FORCE_STOCK_PREDICTIONS": FORCE_STOCK_PREDICTIONS,
                "stock_symbols_count": stock_count,
                "stock_predictions_in_memory": len(stock_predictions),
            },
            "diagnosis": diagnosis,
            "fix": "Set FORCE_STOCK_PREDICTIONS=1 in Railway env vars" if "❌" in diagnosis else "No fix needed"
        }
        
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/volatility")
async def debug_volatility(secret: str = "", symbol: str = "BTC"):
    """
    Debug endpoint to check volatility filter for a symbol.
    """
    if secret != os.getenv("CRON_SECRET", "ghost-cron-2024"):
        return {"error": "Invalid secret"}
    
    try:
        from core.ensemble_predictor import (
            get_volatility_info,
            should_skip_low_volatility,
            MIN_VOLATILITY_CRYPTO,
            MIN_VOLATILITY_STOCKS,
            LOW_CONFIDENCE_THRESHOLD
        )
        
        # Test with sample confidence values
        test_confidences = [0.35, 0.45, 0.55, 0.70, 0.85]
        skip_results = {}
        
        for conf in test_confidences:
            should_skip, reason = should_skip_low_volatility(symbol, conf, None)
            skip_results[f"{conf:.0%}"] = {"skip": should_skip, "reason": reason}
        
        return {
            "ok": True,
            "symbol": symbol,
            "thresholds": {
                "crypto_min_volatility": f"{MIN_VOLATILITY_CRYPTO}%",
                "stocks_min_volatility": f"{MIN_VOLATILITY_STOCKS}%",
                "min_confidence": f"{LOW_CONFIDENCE_THRESHOLD:.0%}",
            },
            "skip_tests": skip_results,
            "note": "Volatility filter reduces noise by skipping low-confidence or low-volatility predictions"
        }
        
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/exclusions")
async def debug_exclusions(secret: str = ""):
    """
    Debug endpoint to verify exclusion system is working.
    Shows both HARDCODED_EXCLUSIONS and GHOST_EXCLUDE_SYMBOLS env var.
    """
    if secret != os.getenv("CRON_SECRET", "ghost-cron-2024"):
        return {"error": "Invalid secret"}
    
    try:
        from core.ghost_notifications import (
            HARDCODED_EXCLUSIONS, 
            _ENV_EXCLUSIONS, 
            reload_env_exclusions,
            get_exclusion_stats,
        )
        
        # Reload to get fresh env var
        current_env = reload_env_exclusions()
        stats = get_exclusion_stats()
        all_excluded = set(HARDCODED_EXCLUSIONS.keys()) | current_env
        
        # Test specific symbols
        test_symbols = ["ALGO", "TIA", "MANA", "ENJ", "CELO", "SAND", "FLOW"]
        test_results = {s: s in all_excluded for s in test_symbols}
        
        return {
            "ok": True,
            "hardcoded_count": len(HARDCODED_EXCLUSIONS),
            "env_exclusions_count": len(current_env),
            "env_exclusions": sorted(current_env),
            "total_unique_excluded": len(all_excluded),
            "raw_env_value": os.getenv("GHOST_EXCLUDE_SYMBOLS", "")[:500],
            "test_exclusions": test_results,
            "all_excluded_symbols": sorted(all_excluded)
        }
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/watchlist-raw")
async def debug_watchlist_raw(secret: str = ""):
    """
    Get raw watchlist from database.
    """
    if secret != os.getenv("CRON_SECRET", "ghost-cron-2024"):
        return {"error": "Invalid secret"}
    
    try:
        from core.db_pool import get_sync_connection
        
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            return {"ok": False, "error": "DATABASE_URL not set"}
        
        with get_sync_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT symbol, asset_type, active
                FROM ghost_watchlist_items
                WHERE active = TRUE
                ORDER BY asset_type, symbol
            """)
            
            items = []
            for row in cursor.fetchall():
                items.append({"symbol": row[0], "type": row[1], "active": row[2]})
            
            stocks = [i["symbol"] for i in items if i["type"] == "stock"]
            cryptos = [i["symbol"] for i in items if i["type"] == "crypto"]
            
            return {
                "ok": True,
                "total": len(items),
                "stocks": stocks,
                "cryptos": cryptos,
                "stock_count": len(stocks),
                "crypto_count": len(cryptos)
            }
        
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.post("/debug/watchlist-add")
async def debug_watchlist_add(symbol: str = "", asset_type: str = "stock", secret: str = ""):
    """
    Debug endpoint to add symbols directly to watchlist.
    Bypasses API layer for testing.
    """
    if secret != os.getenv("CRON_SECRET", "ghost-cron-2024"):
        return {"error": "Invalid secret"}
    
    if not symbol:
        return {"error": "symbol required"}
    
    try:
        from core.db_pool import get_sync_connection
        
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            return {"ok": False, "error": "DATABASE_URL not set"}
        
        with get_sync_connection() as conn:
            cursor = conn.cursor()
            
            symbol = symbol.upper().strip()
            asset_type = asset_type.lower().strip()
        
            # Check if exists
            cursor.execute("""
                SELECT id, active FROM ghost_watchlist_items 
                WHERE symbol = %s AND asset_type = %s
            """, (symbol, asset_type))
            existing = cursor.fetchone()
            
            if existing:
                item_id, was_active = existing
                cursor.execute("""
                    UPDATE ghost_watchlist_items
                    SET active = TRUE, updated_at = NOW()
                    WHERE id = %s
                """, (item_id,))
                action = "re-activated" if not was_active else "updated"
            else:
                cursor.execute("""
                    INSERT INTO ghost_watchlist_items (symbol, asset_type, owns_position, notes)
                    VALUES (%s, %s, FALSE, '')
                    RETURNING id
                """, (symbol, asset_type))
                item_id = cursor.fetchone()[0]
                action = "added"
            
            return {"ok": True, "action": action, "symbol": symbol, "asset_type": asset_type, "id": item_id}
        
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.post("/debug/watchlist-bulk-add")
async def debug_watchlist_bulk_add(request: Request, secret: str = ""):
    """
    Bulk add symbols to watchlist.
    Body: {"symbols": ["AAPL", "AMD", ...], "asset_type": "stock"}
    """
    if secret != os.getenv("CRON_SECRET", "ghost-cron-2024"):
        return {"error": "Invalid secret"}
    
    try:
        import psycopg2
        
        body = await request.json()
        symbols = body.get("symbols", [])
        asset_type = body.get("asset_type", "stock").lower().strip()
        
        if not symbols:
            return {"error": "symbols list required"}
        
        from core.db_pool import get_sync_connection
        
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            return {"ok": False, "error": "DATABASE_URL not set"}
        
        with get_sync_connection() as conn:
            cursor = conn.cursor()
            
            added = []
            updated = []
            errors = []
            
            for symbol in symbols:
                try:
                    symbol = symbol.upper().strip()
                    
                    cursor.execute("""
                        SELECT id, active FROM ghost_watchlist_items 
                        WHERE symbol = %s AND asset_type = %s
                    """, (symbol, asset_type))
                    existing = cursor.fetchone()
                    
                    if existing:
                        item_id, was_active = existing
                        cursor.execute("""
                            UPDATE ghost_watchlist_items
                            SET active = TRUE, updated_at = NOW()
                            WHERE id = %s
                        """, (item_id,))
                        if was_active:
                            updated.append(symbol)
                        else:
                            added.append(symbol)
                    else:
                        cursor.execute("""
                            INSERT INTO ghost_watchlist_items (symbol, asset_type, owns_position, notes)
                            VALUES (%s, %s, FALSE, '')
                        """, (symbol, asset_type))
                        added.append(symbol)
                except Exception as e:
                    errors.append(f"{symbol}: {e}")
            
            return {
                "ok": True,
                "added": added,
                "updated": updated,
                "errors": errors,
                "total_added": len(added),
                "total_updated": len(updated)
            }
        
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/watchlist-schema")
async def debug_watchlist_schema(secret: str = ""):
    """
    Check if ghost_watchlist_items table exists and show schema.
    Also create it if it doesn't exist.
    """
    if secret != os.getenv("CRON_SECRET", "ghost-cron-2024"):
        return {"error": "Invalid secret"}
    
    try:
        from core.db_pool import get_sync_connection
        
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            return {"ok": False, "error": "DATABASE_URL not set"}
        
        with get_sync_connection() as conn:
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'ghost_watchlist_items'
                )
            """)
            exists = cursor.fetchone()[0]
            
            if not exists:
                # Create the table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ghost_watchlist_items (
                        id BIGSERIAL PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        asset_type TEXT NOT NULL CHECK (asset_type IN ('crypto', 'stock')),
                        owns_position BOOLEAN DEFAULT FALSE,
                        notes TEXT DEFAULT '',
                        added_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW(),
                        active BOOLEAN DEFAULT TRUE,
                        price_at_add REAL,
                        alert_threshold_pct REAL DEFAULT 5.0,
                        priority INTEGER DEFAULT 1,
                        CHECK (LENGTH(symbol) > 0 AND LENGTH(symbol) <= 20)
                    )
                """)
                cursor.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_watchlist_unique_active 
                    ON ghost_watchlist_items(symbol, asset_type) WHERE active = TRUE
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_watchlist_symbol 
                    ON ghost_watchlist_items(symbol) WHERE active = TRUE
                """)
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_watchlist_asset_type 
                    ON ghost_watchlist_items(asset_type) WHERE active = TRUE
                """)
                
                return {
                    "ok": True,
                    "created": True,
                    "message": "ghost_watchlist_items table created successfully!"
                }
            
            # Get current count
            cursor.execute("SELECT COUNT(*) FROM ghost_watchlist_items WHERE active = TRUE")
            active_count = cursor.fetchone()[0]
            
            # Get breakdown
            cursor.execute("""
                SELECT asset_type, COUNT(*) as cnt 
                FROM ghost_watchlist_items 
                WHERE active = TRUE 
                GROUP BY asset_type
            """)
            breakdown = dict(cursor.fetchall())
            
            return {
                "ok": True,
                "table_exists": True,
                "active_items": active_count,
                "breakdown": breakdown
            }
        
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/learning-status")
async def debug_learning_status():
    """
    Check the status of Ghost's learning system.
    
    Shows:
    - Feedback loop status (feature weights, signal performance)
    - Learning loop status (parameter adjustments)
    - Recent outcomes processed
    - PostgreSQL vs SQLite outcome counts
    """
    try:
        from core.feedback_loop import get_feedback_loop
        from core.learning_loop import get_learning_loop
        import psycopg2
        import sqlite3
        
        feedback = get_feedback_loop()
        learning = get_learning_loop()
        
        # Get feedback loop report
        feedback_report = feedback.get_performance_report(days=7)
        
        # Get PostgreSQL outcome counts
        postgres_stats = {"total": 0, "wins": 0, "losses": 0, "accuracy_pct": 0}
        try:
            database_url = os.getenv("DATABASE_URL")
            if database_url:
                from core.db_pool import get_sync_connection
                with get_sync_connection() as conn:
                    cursor = conn.cursor()
                    # Don't filter by status - just count all outcomes
                    cursor.execute("""
                        SELECT 
                            COUNT(*) as total,
                            SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END) as wins,
                            SUM(CASE WHEN hit_direction = 0 THEN 1 ELSE 0 END) as losses
                        FROM ghost_prediction_outcomes
                    """)
                    row = cursor.fetchone()
                    if row:
                        postgres_stats["total"] = row[0] or 0
                        postgres_stats["wins"] = row[1] or 0
                        postgres_stats["losses"] = row[2] or 0
                        if postgres_stats["total"] > 0:
                            decided = postgres_stats["wins"] + postgres_stats["losses"]
                            postgres_stats["accuracy_pct"] = (postgres_stats["wins"] / decided) * 100 if decided > 0 else 0
        except Exception as pg_err:
            postgres_stats["error"] = str(pg_err)
        
        # Get SQLite feedback_loop.db counts
        sqlite_stats = {"total": 0, "wins": 0, "losses": 0}
        try:
            from pathlib import Path
            sqlite_path = Path(__file__).parent / "data" / "feedback_loop.db"
            if sqlite_path.exists():
                conn = sqlite3.connect(str(sqlite_path))
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(was_correct) as wins
                    FROM prediction_outcomes
                """)
                row = cursor.fetchone()
                if row:
                    sqlite_stats["total"] = row[0] or 0
                    sqlite_stats["wins"] = row[1] or 0
                    sqlite_stats["losses"] = sqlite_stats["total"] - sqlite_stats["wins"]
                conn.close()
        except Exception as sq_err:
            sqlite_stats["error"] = str(sq_err)
        
        # Get learning status
        learning_status = {
            "enabled": True,
            "feature_weights_count": len(feedback.feature_weights),
            "signals_tracked": len(feedback.signal_performance),
            "recent_outcomes_cached": len(feedback.recent_outcomes),
        }
        
        # Top performing features
        top_features = []
        for name, weight in sorted(feedback.feature_weights.items(), key=lambda x: x[1], reverse=True)[:10]:
            top_features.append({"feature": name, "weight": weight})
        
        # Signal performance
        signal_stats = []
        for signal, stats in feedback.signal_performance.items():
            if stats["total"] >= 5:
                acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
                signal_stats.append({
                    "signal": signal,
                    "accuracy": f"{acc:.1%}",
                    "total": stats["total"],
                    "correct": stats["correct"]
                })
        signal_stats.sort(key=lambda x: x["total"], reverse=True)
        
        return {
            "ok": True,
            "learning_active": feedback_report.get("learning_status") == "active",
            "data_sources": {
                "postgres_outcomes": postgres_stats,
                "sqlite_feedback_loop": sqlite_stats,
                "memory_cache": len(feedback.recent_outcomes),
            },
            "total_outcomes_processed": feedback_report.get("total_predictions", 0),
            "accuracy_rate": f"{feedback_report.get('accuracy_rate', 0):.1%}",
            "avg_accuracy_pct": f"{feedback_report.get('avg_accuracy_pct', 0):.1f}%",
            "feature_weights": top_features[:5],
            "signal_performance": signal_stats[:5],
            "learning_status": learning_status,
            "feedback_report": feedback_report,
        }
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.post("/debug/force-send-top10")
async def force_send_top10():
    """
    FORCE SEND TOP 10 via the real pipeline (bypasses date check).
    Uses _LATEST_PREDICTIONS → V3 clean → Telegram.
    For testing V3 pipeline changes immediately.
    """
    try:
        from core.ghost_notifications import get_notification_system, format_top10_message
        from core.adapters import process_v3_from_cache
        import traceback

        notif = get_notification_system()

        # Ensure telegram function is set (may not be if notification loop hasn't started)
        if not notif.send_telegram:
            tg_token = TELEGRAM_BOT_TOKEN
            tg_chat = TELEGRAM_CHAT_ID
            if not tg_token or not tg_chat:
                return {
                    "ok": False,
                    "error": "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set",
                    "token_set": bool(tg_token),
                    "chat_set": bool(tg_chat),
                    "predictions_count": len(_LATEST_PREDICTIONS),
                }
            def _send_tg(message: str) -> bool:
                return _tg_send_chat_message(tg_chat, message)
            notif.set_telegram_func(_send_tg)

        # Reset the date guard so send_top10 doesn't skip
        notif._last_top10_date = ""

        # Call the real send_top10 pipeline
        has_tg = bool(notif.send_telegram)
        
        # Pre-check: run V3 pipeline ourselves to get diagnostic data
        try:
            from core.adapters import process_v3_from_cache as _pv3, batch_convert
            
            # Step 1: Check edge filter
            import os as _os
            _edge_set = get_edge_set()
            edge_preds = {sym: p for sym, p in _LATEST_PREDICTIONS.items() if sym.upper() in _edge_set}
            
            # Step 2: Check batch_convert
            converted = batch_convert(list(edge_preds.values()))
            
            # Step 3: Run full pipeline
            pre_stocks, pre_crypto = _pv3(_LATEST_PREDICTIONS)
            
            # Sample prediction for debugging
            sample = None
            if edge_preds:
                first_sym = list(edge_preds.keys())[0]
                first_pred = edge_preds[first_sym]
                sample = {
                    "symbol": first_pred.get("symbol"),
                    "direction": first_pred.get("direction"),
                    "confidence": first_pred.get("confidence"),
                    "current_price": first_pred.get("current_price") or first_pred.get("price_current"),
                    "ok": first_pred.get("ok"),
                }
            
            pre_info = {
                "total_predictions": len(_LATEST_PREDICTIONS),
                "edge_filtered": len(edge_preds),
                "edge_symbols": sorted(list(edge_preds.keys()))[:10],
                "batch_converted": len(converted),
                "stocks": len(pre_stocks),
                "crypto": len(pre_crypto),
                "sample_prediction": sample,
            }
        except Exception as pe:
            import traceback
            pre_info = {"error": str(pe), "trace": traceback.format_exc()[-500:]}
        
        success = notif.send_top10(_LATEST_PREDICTIONS)

        return {
            "ok": success,
            "predictions_count": len(_LATEST_PREDICTIONS),
            "telegram_func_set": has_tg,
            "v3_pre_check": pre_info,
            "message": "TOP 10 sent via real pipeline" if success else "send_top10 returned False — check deploy logs",
            "last_top10_date": notif._last_top10_date,
        }
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/top10-preview")
async def top10_preview():
    """
    Preview the TOP 10 message that would be sent.
    
    Shows the EXACT message content and individual pick directions.
    Uses the SAME V3 clean architecture pipeline as send_top10().
    """
    try:
        from core.ghost_notifications import get_notification_system, format_top10_message
        from core.adapters import process_v3_from_cache
        
        notif = get_notification_system()
        
        # Use V3 clean path (same as send_top10) with legacy fallback
        try:
            stocks, crypto = process_v3_from_cache(_LATEST_PREDICTIONS)
        except Exception as e:
            LOGGER.warning(f"[TOP10-PREVIEW] V3 clean failed, falling back to legacy: {e}")
            stocks, crypto = notif.get_top10_predictions(_LATEST_PREDICTIONS)
        
        # Build direction debug info
        stocks_debug = [
            {"symbol": s["symbol"], "direction": s.get("direction", "MISSING"), "conf": s["confidence"]}
            for s in stocks
        ]
        crypto_debug = [
            {"symbol": c["symbol"], "direction": c.get("direction", "MISSING"), "conf": c["confidence"]}
            for c in crypto
        ]
        
        # Generate the actual messages (returns list of 2: stocks + crypto)
        messages = format_top10_message(stocks, crypto)
        
        return {
            "ok": True,
            "stocks_count": len(stocks),
            "crypto_count": len(crypto),
            "stocks_with_directions": stocks_debug,
            "crypto_with_directions": crypto_debug,
            "message_preview": messages,  # Now returns list
        }
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/regime-status")
async def debug_regime_status():
    """
    Debug endpoint: Current market regime filter status.
    
    Shows BTC 24h/7d trends, SPY regime, current filter level,
    and what action would be taken on crypto/stock BUYs.
    
    Usage:
        /debug/regime-status
    """
    try:
        from core.regime_filter import get_regime_debug
        return await get_regime_debug()
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/regime-preview")
async def debug_regime_preview():
    """
    Debug endpoint: Preview what the regime filter would do to today's picks.
    
    Runs the full V3 pipeline + regime filter and shows before/after.
    Does NOT send any messages.
    
    Usage:
        /debug/regime-preview
    """
    try:
        from core.adapters import process_v3_from_cache
        from core.regime_filter import apply_regime_filter
        
        # Run V3 pipeline
        stocks_raw, crypto_raw = process_v3_from_cache(_LATEST_PREDICTIONS)
        
        # Run regime filter
        stocks_filtered, crypto_filtered, regime_info = await apply_regime_filter(
            stocks_raw, crypto_raw
        )
        
        def _pick_summary(picks):
            return [
                {
                    "symbol": p["symbol"],
                    "direction": p.get("direction", "?"),
                    "confidence": round(p.get("confidence", 0), 3),
                }
                for p in picks
            ]
        
        return {
            "ok": True,
            "regime": regime_info,
            "before": {
                "stocks": _pick_summary(stocks_raw),
                "crypto": _pick_summary(crypto_raw),
                "total": len(stocks_raw) + len(crypto_raw),
            },
            "after": {
                "stocks": _pick_summary(stocks_filtered),
                "crypto": _pick_summary(crypto_filtered),
                "total": len(stocks_filtered) + len(crypto_filtered),
            },
        }
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/tracking-status")
async def debug_tracking_status(secret: str = ""):
    """
    Debug endpoint to check pick tracking status.
    Shows all active tracked picks and their current status.
    """
    if secret != os.getenv("CRON_SECRET", "ghost-cron-2024"):
        return {"error": "Invalid secret"}
    
    try:
        from core.ghost_notifications import TRACKING_DB, get_notification_system
        
        # Get status from notification system (includes db_type)
        notif = get_notification_system()
        status = notif.get_status()
        
        picks_list = []
        all_picks = 0
        db_source = "unknown"
        
        # Try PostgreSQL first
        database_url = os.getenv("DATABASE_URL", "")
        if database_url:
            try:
                from core.db_pool import get_sync_connection
                with get_sync_connection() as conn:
                    cur = conn.cursor()
                    
                    cur.execute("""
                        SELECT symbol, asset_type, direction, entry_price, target_price, stop_price, 
                               confidence, entry_time, expires_at, status
                        FROM ghost_tracked_picks 
                        WHERE status = 'active'
                        ORDER BY entry_time DESC
                    """)
                    active_picks = cur.fetchall()
                    
                    cur.execute("SELECT COUNT(*) FROM ghost_tracked_picks")
                    all_picks = cur.fetchone()[0]
                    
                db_source = "postgresql (persistent)"
                
                for p in active_picks:
                    picks_list.append({
                        "symbol": p[0],
                        "asset_type": p[1],
                        "direction": p[2],
                        "entry_price": float(p[3]) if p[3] else 0,
                        "target_price": float(p[4]) if p[4] else 0,
                        "stop_price": float(p[5]) if p[5] else 0,
                        "confidence": float(p[6]) if p[6] else 0,
                        "entry_time": str(p[7]) if p[7] else "",
                        "expires_at": str(p[8]) if p[8] else "",
                        "status": p[9],
                    })
            except Exception as pg_err:
                db_source = f"postgresql FAILED: {pg_err}"
        
        # Fallback to SQLite
        if not picks_list and not database_url:
            import sqlite3
            conn = sqlite3.connect(TRACKING_DB)
            active_picks = conn.execute("""
                SELECT symbol, asset_type, direction, entry_price, target_price, stop_price, 
                       confidence, entry_time, expires_at, status
                FROM tracked_picks 
                WHERE status = 'active'
                ORDER BY entry_time DESC
            """).fetchall()
            
            all_picks = conn.execute("SELECT COUNT(*) FROM tracked_picks").fetchone()[0]
            conn.close()
            db_source = "sqlite (ephemeral)"
            
            for p in active_picks:
                picks_list.append({
                    "symbol": p[0],
                    "asset_type": p[1],
                    "direction": p[2],
                    "entry_price": p[3],
                    "target_price": p[4],
                    "stop_price": p[5],
                    "confidence": p[6],
                    "entry_time": p[7],
                    "expires_at": p[8],
                    "status": p[9],
                })
        
        return {
            "ok": True,
            "database": db_source,
            "persistent": "postgresql" in db_source.lower() and "FAILED" not in db_source,
            "status": status,
            "total_picks_in_db": all_picks,
            "active_picks_count": len(picks_list),
            "active_picks": picks_list,
        }
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/debug/info")
async def debug_info():
    """Lightweight diagnostics to verify deployment state.
    Returns commit hash (if available), key env flags, and a small routes summary.
    """
    try:
        commit = None
        try:
            head = os.popen("git rev-parse --short HEAD").read().strip()
            commit = head or None
        except Exception:
            commit = None

        routes = [r.path for r in getattr(APP, "routes", []) if getattr(r, "path", None)]
        return {
            "ok": True,
            "commit": commit,
            "env": {
                "AGENTS_ENABLED": os.getenv("AGENTS_ENABLED"),
                "AI_PROVIDER": os.getenv("AI_PROVIDER"),
            },
            "has_ai_chat": "/ai/chat" in routes,
            "routes_sample": sorted(
                [
                    p
                    for p in routes
                    if p
                    in ("/health", "/ai/chat", "/telegram/webhook", "/ai/agent/run", "/debug/info")
                ]
            ),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/debug/price")
async def debug_price():
    """Bypass cache and fetch provider prices for diagnosis.
    Returns tuples (price, prev_close, provider_label) per provider, plus plausibility and TTL info.
    """
    out: dict[str, object] = {}
    try:
        a = _fetch_price_alphavantage(WOLF)
        out["alphavantage"] = {
            "raw": a,
            "plausible": (_is_plausible_price(WOLF, a[0], a[1]) if isinstance(a, tuple) else False),
        }
    except Exception as e:
        out["alphavantage"] = {"error": str(e)}
    try:
        p = _fetch_price_polygon(WOLF)
        out["polygon"] = {
            "raw": p,
            "plausible": (_is_plausible_price(WOLF, p[0], p[1]) if isinstance(p, tuple) else False),
        }
    except Exception as e:
        out["polygon"] = {"error": str(e)}
    try:
        y = _fetch_price_yfinance(WOLF)
        out["yfinance"] = {
            "raw": y,
            "plausible": (_is_plausible_price(WOLF, y[0], y[1]) if isinstance(y, tuple) else False),
        }
    except Exception as e:
        out["yfinance"] = {"error": str(e)}
    try:
        yh = _fetch_price_yahoo_http(WOLF)
        out["yahoo_http"] = {
            "raw": yh,
            "plausible": (
                _is_plausible_price(WOLF, yh[0], yh[1]) if isinstance(yh, tuple) else False
            ),
        }
    except Exception as e:
        out["yahoo_http"] = {"error": str(e)}
    out["ttl_s"] = {
        "price_ttl_s": PRICE_TTL_S,
        "price_ttl_open_s": PRICE_TTL_OPEN_S,
        "news_ttl_s": NEWS_TTL_S,
        "yahoo_first": bool(PRICE_YAHOO_FIRST),
        "price_max_deviation": float(os.getenv("PRICE_MAX_DEVIATION", "0.5")),
        "price_max_deviation_open": PRICE_MAX_DEVIATION_OPEN,
    }
    return out


@router.post("/debug/price_override")
async def debug_price_override(
    body: dict | None = None,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """Set or clear a temporary manual price override.
    Payloads:
      {"symbol":"WOLF","price":1.21,"ttl_s":86400}
      {"clear":true}
    When active, provider label will be "manual" and flags should treat it as stale.
    """
    try:
        # Protected only if a token is configured; otherwise open (dev convenience)
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass
    b = body or {}
    if bool(b.get("clear")):
        PRICE_OVERRIDE.update({"symbol": None, "price": None, "until": 0.0})
        try:
            _add_event("price.override", "Cleared manual price override", {})
        except Exception:
            pass
        return {"ok": True, "cleared": True}
    sym = str(b.get("symbol") or WOLF).upper()
    if "price" not in b:
        raise HTTPException(422, "price is required unless clear=true")
    try:
        price_val = b.get("price")
        if price_val is None:
            raise HTTPException(422, "price is required unless clear=true")
        px = float(price_val)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(422, "price must be a number")
    ttl = int(b.get("ttl_s", 24 * 60 * 60))
    PRICE_OVERRIDE.update({"symbol": sym, "price": float(px), "until": time.time() + max(1, ttl)})


@router.post("/debug/prev_close")
async def debug_set_prev_close(
    body: dict | None = None,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """Test-helper: set cached prev_close for WOLF and clear live price.
    Enabled only when SNAP_TEST_MODE is active.
    """
    # Require bearer if configured
    _require_bearer(
        (f"Bearer {credentials.credentials}") if credentials and credentials.credentials else None
    )
    if os.getenv("SNAP_TEST_MODE", "0").lower() not in ("1", "true", "yes"):
        raise HTTPException(403, "forbidden")
    try:
        prev_close_val = (body or {}).get("prev_close")
        if prev_close_val is None:
            raise HTTPException(422, "prev_close is required")
        pv = float(prev_close_val)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(422, "invalid prev_close")
    _cache_put_price(WOLF, None, pv, "prev-close")
    return {"ok": True, "prev_close": pv}


@router.post("/debug/price_diag")
async def debug_set_price_diag(
    body: dict | None = None,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """Test-helper: set PRICE_DIAG fields to simulate quorum/anomaly.
    Enabled only when SNAP_TEST_MODE is active.
    """
    # Require bearer if configured
    _require_bearer(
        (f"Bearer {credentials.credentials}") if credentials and credentials.credentials else None
    )
    if os.getenv("SNAP_TEST_MODE", "0").lower() not in ("1", "true", "yes"):
        raise HTTPException(403, "forbidden")
    try:
        if isinstance(body, dict):
            for k in ("anomaly", "reason", "provider_spread", "quorum_ok"):
                if k in body:
                    PRICE_DIAG[k] = body[k]
    except Exception:
        pass
    return {"ok": True, "diag": PRICE_DIAG}


@router.post("/debug/telegram_test")
async def debug_telegram_test(
    body: dict | None = None,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """Test Telegram notifications.
    Sends a test message to configured Telegram chat(s).
    """
    try:
        # Require bearer if configured
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
        message = (body or {}).get("message", "🧪 Test notification from GHOST")

        # Format as a status card
        card = f"""<b>📡 GHOST Test Alert</b>
{message}

<i>Timestamp: {datetime.now().isoformat()}</i>
<i>Version: v0.3.0</i>"""

        success = enqueue_alert_text(card)

        if not success:
            return {"ok": False, "error": "Failed to enqueue alert"}

        # Wait a moment for the worker to process
        await asyncio.sleep(1)

        return {
            "ok": True,
            "message": "Test notification sent",
            "telegram_configured": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
            "telegram_chat_id": TELEGRAM_CHAT_ID if TELEGRAM_CHAT_ID else None,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/debug/reset_breakers")
async def debug_reset_breakers(credentials: HTTPAuthorizationCredentials | None = AUTH_DEP):
    """Emergency circuit breaker reset when all providers are stuck in backoff.
    Resets all breakers to closed state with zero failures.
    """
    # Require bearer if configured
    _require_bearer(
        (f"Bearer {credentials.credentials}") if credentials and credentials.credentials else None
    )
    global _PROVIDER_BREAKERS
    for provider_name in _PROVIDER_BREAKERS:
        _PROVIDER_BREAKERS[provider_name] = {
            "state": "closed",
            "failures": 0,
            "backoff_factor": 0,
            "open_until_ts": 0.0,
        }
    LOGGER.warning("Circuit breakers manually reset via /debug/reset_breakers")
    return {
        "ok": True,
        "breakers": _PROVIDER_BREAKERS,
        "message": "All breakers reset to closed state",
    }


@router.get("/debug/price-providers-diagnostic")
async def debug_price_providers_diagnostic():
    """
    COMPREHENSIVE PRICE PROVIDER DIAGNOSTIC
    
    Tests all price providers, checks API keys, cache state, and network connectivity.
    Use this to diagnose why price_providers show "timeout" in health check.
    """
    import aiohttp
    import subprocess
    from datetime import datetime
    
    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "api_keys": {},
        "provider_tests": {},
        "network_tests": {},
        "cache_state": {},
        "circuit_breakers": {},
        "diagnosis": [],
    }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1: CHECK API KEYS
    # ═══════════════════════════════════════════════════════════════════════════
    api_keys_to_check = [
        'BINANCE_API_KEY',
        'COINGECKO_API_KEY', 
        'COINMARKETCAP_API_KEY',
        'ALPHA_VANTAGE_API_KEY',
        'ALPHAVANTAGE_API_KEY',
        'FINNHUB_API_KEY',
        'POLYGON_API_KEY',
        'POLYGON_IO_API_KEY',
        'YAHOO_FINANCE_API_KEY',
        'SANTIMENT_API_KEY',
        'CRYPTOCOMPARE_API_KEY',
    ]
    
    for key in api_keys_to_check:
        value = os.environ.get(key, None)
        if value:
            # Mask the key (show first 4 and last 4 chars)
            if len(value) > 8:
                masked = value[:4] + '****' + value[-4:]
            else:
                masked = '****'
            results["api_keys"][key] = {"status": "SET", "masked": masked}
        else:
            results["api_keys"][key] = {"status": "NOT_SET"}
            results["diagnosis"].append(f"⚠️ {key} is not set")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2: TEST EACH PRICE PROVIDER
    # ═══════════════════════════════════════════════════════════════════════════
    async def test_provider_url(name: str, url: str, headers: dict = None, timeout_s: float = 10.0):
        """Test a single price provider endpoint"""
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout_s)) as resp:
                    elapsed = time.time() - start
                    body = await resp.text()
                    if resp.status == 200:
                        try:
                            data = await resp.json()
                            # Extract price if possible
                            price = None
                            if "price" in str(data).lower():
                                if isinstance(data, dict):
                                    price = data.get("price") or data.get("lastPrice")
                            return {
                                "status": "OK",
                                "http_code": resp.status,
                                "elapsed_s": round(elapsed, 3),
                                "sample_data": str(data)[:200],
                                "price_found": price,
                            }
                        except (ValueError, KeyError):
                            return {
                                "status": "OK_NOT_JSON",
                                "http_code": resp.status,
                                "elapsed_s": round(elapsed, 3),
                                "body_preview": body[:200],
                            }
                    elif resp.status == 429:
                        return {
                            "status": "RATE_LIMITED",
                            "http_code": 429,
                            "elapsed_s": round(elapsed, 3),
                            "body": body[:200],
                        }
                    elif resp.status == 451:
                        return {
                            "status": "GEO_BLOCKED",
                            "http_code": 451,
                            "elapsed_s": round(elapsed, 3),
                            "note": "Blocked in this region (common for Binance)",
                        }
                    elif resp.status == 403:
                        return {
                            "status": "FORBIDDEN",
                            "http_code": 403,
                            "elapsed_s": round(elapsed, 3),
                            "body": body[:200],
                        }
                    else:
                        return {
                            "status": "HTTP_ERROR",
                            "http_code": resp.status,
                            "elapsed_s": round(elapsed, 3),
                            "body": body[:200],
                        }
        except asyncio.TimeoutError:
            return {
                "status": "TIMEOUT",
                "elapsed_s": timeout_s,
                "error": f"Request timed out after {timeout_s}s",
            }
        except aiohttp.ClientConnectorError as e:
            return {
                "status": "CONNECTION_ERROR",
                "elapsed_s": round(time.time() - start, 3),
                "error": str(e)[:100],
            }
        except Exception as e:
            return {
                "status": "ERROR",
                "elapsed_s": round(time.time() - start, 3),
                "error": str(e)[:100],
            }
    
    # Test crypto providers
    crypto_tests = [
        ("binance_btc", "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", None),
        ("binance_eth", "https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT", None),
        ("binance_us_btc", "https://api.binance.us/api/v3/ticker/price?symbol=BTCUSD", None),
        ("coingecko_btc", "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd", None),
        ("coingecko_eth", "https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies=usd", None),
        ("coinbase_btc", "https://api.coinbase.com/v2/prices/BTC-USD/spot", None),
        ("coinbase_eth", "https://api.coinbase.com/v2/prices/ETH-USD/spot", None),
        ("cryptocompare_btc", "https://min-api.cryptocompare.com/data/price?fsym=BTC&tsyms=USD", None),
    ]
    
    # Test stock providers
    stock_tests = [
        ("yahoo_aapl", "https://query1.finance.yahoo.com/v8/finance/chart/AAPL?interval=1d&range=1d", None),
        ("yahoo_msft", "https://query1.finance.yahoo.com/v8/finance/chart/MSFT?interval=1d&range=1d", None),
        ("yahoo_googl", "https://query1.finance.yahoo.com/v8/finance/chart/GOOGL?interval=1d&range=1d", None),
    ]
    
    # Add Polygon test if API key is set (use FULL key, not truncated)
    polygon_key = os.environ.get("POLYGON_API_KEY") or os.environ.get("POLYGON_IO_API_KEY")
    if polygon_key:
        stock_tests.append(
            ("polygon_aapl", f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev?apiKey={polygon_key}", None)
        )
    
    # Add Alpha Vantage test if API key is set (use FULL key)
    av_key = os.environ.get("ALPHA_VANTAGE_API_KEY") or os.environ.get("ALPHAVANTAGE_API_KEY")
    if av_key:
        stock_tests.append(
            ("alphavantage_aapl", f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={av_key}", None)
        )
    
    # Run all provider tests
    results["provider_tests"]["crypto"] = {}
    for name, url, headers in crypto_tests:
        results["provider_tests"]["crypto"][name] = await test_provider_url(name, url, headers)
        await asyncio.sleep(0.3)  # Rate limit protection
    
    results["provider_tests"]["stocks"] = {}
    for name, url, headers in stock_tests:
        results["provider_tests"]["stocks"][name] = await test_provider_url(name, url, headers)
        await asyncio.sleep(0.3)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 3: CHECK NETWORK CONNECTIVITY
    # ═══════════════════════════════════════════════════════════════════════════
    async def check_dns(hostname: str):
        """Check if we can resolve a hostname"""
        import socket
        try:
            ip = socket.gethostbyname(hostname)
            return {"status": "OK", "resolved_ip": ip}
        except socket.gaierror as e:
            return {"status": "DNS_FAILED", "error": str(e)}
    
    dns_tests = [
        "api.binance.com",
        "api.coingecko.com", 
        "api.coinbase.com",
        "query1.finance.yahoo.com",
        "api.polygon.io",
    ]
    
    results["network_tests"]["dns"] = {}
    for host in dns_tests:
        results["network_tests"]["dns"][host] = await check_dns(host)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 4: CHECK CACHE STATE (TurboProvider)
    # ═══════════════════════════════════════════════════════════════════════════
    try:
        from core.providers.turbo_provider import get_turbo_provider
        turbo = get_turbo_provider()
        
        if hasattr(turbo, '_price_cache'):
            cache = turbo._price_cache
            results["cache_state"]["turbo_provider"] = {
                "cache_size": len(cache),
                "entries": [],
            }
            
            # Show cached entries with ages
            now = datetime.now()
            for symbol, cached in list(cache.items())[:20]:  # First 20
                if hasattr(cached, 'price') and hasattr(cached, 'timestamp'):
                    age_seconds = (now - cached.timestamp).total_seconds()
                    results["cache_state"]["turbo_provider"]["entries"].append({
                        "symbol": symbol,
                        "price": cached.price,
                        "provider": cached.provider,
                        "age_seconds": round(age_seconds, 1),
                        "age_human": f"{age_seconds/60:.1f} min" if age_seconds < 3600 else f"{age_seconds/3600:.1f} hr",
                        "is_stale": age_seconds > 300,  # >5 min is stale
                    })
        else:
            results["cache_state"]["turbo_provider"] = {"note": "No _price_cache attribute found"}
    except Exception as e:
        results["cache_state"]["turbo_provider"] = {"error": str(e)}
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 5: CHECK CIRCUIT BREAKERS
    # ═══════════════════════════════════════════════════════════════════════════
    try:
        results["circuit_breakers"] = dict(_PROVIDER_BREAKERS)
    except Exception:
        results["circuit_breakers"] = {"error": "Could not access _PROVIDER_BREAKERS"}
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 6: GENERATE DIAGNOSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Count working providers
    crypto_working = sum(1 for v in results["provider_tests"]["crypto"].values() if v.get("status") == "OK")
    stock_working = sum(1 for v in results["provider_tests"]["stocks"].values() if v.get("status") == "OK")
    
    results["summary"] = {
        "crypto_providers_working": f"{crypto_working}/{len(results['provider_tests']['crypto'])}",
        "stock_providers_working": f"{stock_working}/{len(results['provider_tests']['stocks'])}",
        "api_keys_set": sum(1 for v in results["api_keys"].values() if v.get("status") == "SET"),
        "api_keys_missing": sum(1 for v in results["api_keys"].values() if v.get("status") == "NOT_SET"),
    }
    
    # Add diagnosis messages
    if crypto_working == 0:
        results["diagnosis"].append("🚨 ALL CRYPTO PROVIDERS FAILING - Check network/region blocking")
    elif crypto_working < 3:
        results["diagnosis"].append("⚠️ Some crypto providers failing - May affect price reliability")
    
    if stock_working == 0:
        results["diagnosis"].append("🚨 ALL STOCK PROVIDERS FAILING - Check Yahoo/Polygon connectivity")
    
    # Check for rate limits
    for name, test in results["provider_tests"]["crypto"].items():
        if test.get("status") == "RATE_LIMITED":
            results["diagnosis"].append(f"⚠️ {name} is rate limited (429) - Need API key or backoff")
    
    # Check for geo-blocking
    for name, test in results["provider_tests"]["crypto"].items():
        if test.get("status") == "GEO_BLOCKED":
            results["diagnosis"].append(f"⚠️ {name} is geo-blocked (451) - Common in US for Binance.com")
    
    # Check cache staleness
    stale_count = 0
    if "entries" in results["cache_state"].get("turbo_provider", {}):
        for entry in results["cache_state"]["turbo_provider"]["entries"]:
            if entry.get("is_stale"):
                stale_count += 1
    
    if stale_count > 5:
        results["diagnosis"].append(f"🚨 {stale_count} cached prices are STALE (>5 min old) - Providers may be failing")
    
    if not results["diagnosis"]:
        results["diagnosis"].append("✅ All systems appear operational")
    
    return results


@router.get("/debug/test-top10")
@router.post("/debug/test-top10")
async def test_top10_endpoint():
    """
    TEST TOP 10 NOTIFICATION ENDPOINT
    
    Manually trigger a test TOP 10 notification to verify:
    1. Coinbase price fetching works
    2. Message formatting is correct
    3. Telegram bot/chat ID works
    4. End-to-end delivery succeeds
    
    Usage: GET or POST /debug/test-top10
    """
    import aiohttp
    import os
    from datetime import datetime
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "step": "init",
        "prices_fetched": {},
        "message_built": False,
        "telegram_sent": False,
        "telegram_response": None,
        "errors": [],
    }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1: FETCH REAL PRICES FROM COINBASE
    # ═══════════════════════════════════════════════════════════════════════════
    result["step"] = "fetch_prices"
    
    test_symbols = {
        "stocks": ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL"],
        "crypto": ["BTC", "ETH", "SOL", "XRP", "DOGE"],
    }
    
    prices = {}
    
    # Fetch crypto prices from Coinbase (most reliable)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
        for symbol in test_symbols["crypto"]:
            try:
                url = f"https://api.coinbase.com/v2/exchange-rates?currency={symbol}"
                async with session.get(url) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        usd_rate = float(data["data"]["rates"]["USD"])
                        prices[symbol] = {
                            "price": usd_rate,
                            "source": "coinbase",
                            "status": "OK"
                        }
                        result["prices_fetched"][symbol] = f"${usd_rate:,.2f} via Coinbase"
                    else:
                        prices[symbol] = {"price": 0, "source": "failed", "status": f"HTTP {resp.status}"}
                        result["errors"].append(f"Coinbase {symbol}: HTTP {resp.status}")
            except Exception as e:
                prices[symbol] = {"price": 0, "source": "error", "status": str(e)}
                result["errors"].append(f"Coinbase {symbol}: {str(e)}")
    
    # For stocks, use Polygon (most reliable stock provider)
    polygon_key = os.environ.get("POLYGON_API_KEY")
    if polygon_key:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            for symbol in test_symbols["stocks"]:
                try:
                    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/prev?apiKey={polygon_key}"
                    async with session.get(url) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if data.get("results"):
                                close_price = data["results"][0].get("c", 0)
                                prices[symbol] = {
                                    "price": close_price,
                                    "source": "polygon",
                                    "status": "OK"
                                }
                                result["prices_fetched"][symbol] = f"${close_price:,.2f} via Polygon"
                        else:
                            result["errors"].append(f"Polygon {symbol}: HTTP {resp.status}")
                except Exception as e:
                    result["errors"].append(f"Polygon {symbol}: {str(e)}")
    else:
        result["errors"].append("POLYGON_API_KEY not set - using mock stock prices")
        for symbol in test_symbols["stocks"]:
            mock_prices = {"AAPL": 175.50, "MSFT": 378.25, "NVDA": 495.00, "TSLA": 248.75, "GOOGL": 141.80}
            prices[symbol] = {"price": mock_prices.get(symbol, 100.00), "source": "mock", "status": "MOCK"}
            result["prices_fetched"][symbol] = f"${mock_prices.get(symbol, 100.00):,.2f} (MOCK)"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 2: BUILD THE MESSAGE (Same format as real TOP 10)
    # ═══════════════════════════════════════════════════════════════════════════
    result["step"] = "build_message"
    
    now = datetime.now()
    date_str = now.strftime("%b %d, %Y")
    
    # Build message - with INVERSE_GHOST=1, all predictions are SELL (DOWN)
    inverse_mode = os.environ.get("INVERSE_GHOST", "0") == "1"
    
    message_lines = [
        f"🔮 *Ghost Protocol TOP 10*",
        f"📅 {date_str} (TEST)",
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        "📈 *TOP 5 STOCKS*",
        "",
    ]
    
    # Stock picks - with INVERSE_GHOST, all show as SELL (🔴)
    for symbol in test_symbols["stocks"]:
        price_data = prices.get(symbol, {})
        price = price_data.get("price", 0)
        source = price_data.get("source", "unknown")
        
        # Simulate 48h targets (±5% from entry)
        target_pct = -5 if inverse_mode else 5  # INVERSE = DOWN = SELL
        target_price = price * (1 + target_pct / 100)
        
        # Direction logic: 🔴 SELL for DOWN, 🟢 BUY for UP
        direction_emoji = "🔴" if inverse_mode else "🟢"
        direction_text = "SELL" if inverse_mode else "BUY"
        
        message_lines.append(
            f"{direction_emoji} *{symbol}* - {direction_text}\n"
            f"   Entry: ${price:,.2f} ({source})\n"
            f"   Target: ${target_price:,.2f} ({target_pct:+.0f}%)"
        )
        message_lines.append("")
    
    message_lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━")
    message_lines.append("")
    message_lines.append("🪙 *TOP 5 CRYPTO*")
    message_lines.append("")
    
    # Crypto picks
    for symbol in test_symbols["crypto"]:
        price_data = prices.get(symbol, {})
        price = price_data.get("price", 0)
        source = price_data.get("source", "unknown")
        
        target_pct = -5 if inverse_mode else 5
        target_price = price * (1 + target_pct / 100)
        
        direction_emoji = "🔴" if inverse_mode else "🟢"
        direction_text = "SELL" if inverse_mode else "BUY"
        
        message_lines.append(
            f"{direction_emoji} *{symbol}* - {direction_text}\n"
            f"   Entry: ${price:,.2f} ({source})\n"
            f"   Target: ${target_price:,.2f} ({target_pct:+.0f}%)"
        )
        message_lines.append("")
    
    message_lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━")
    message_lines.append("")
    message_lines.append(f"⚙️ Mode: {'INVERSE' if inverse_mode else 'NORMAL'}")
    message_lines.append("🧪 _This is a TEST message_")
    
    full_message = "\n".join(message_lines)
    result["message_built"] = True
    result["message_preview"] = full_message[:500] + "..." if len(full_message) > 500 else full_message
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 3: SEND TO TELEGRAM
    # ═══════════════════════════════════════════════════════════════════════════
    result["step"] = "send_telegram"
    
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if not bot_token:
        result["errors"].append("TELEGRAM_BOT_TOKEN not set")
    if not chat_id:
        result["errors"].append("TELEGRAM_CHAT_ID not set")
    
    if bot_token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": full_message,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                async with session.post(url, json=payload) as resp:
                    resp_data = await resp.json()
                    
                    if resp.status == 200 and resp_data.get("ok"):
                        result["telegram_sent"] = True
                        result["telegram_response"] = {
                            "status": "SUCCESS",
                            "message_id": resp_data.get("result", {}).get("message_id"),
                            "chat_id": chat_id,
                        }
                        LOGGER.info(f"✅ Test TOP 10 sent to Telegram: message_id={resp_data.get('result', {}).get('message_id')}")
                    else:
                        result["telegram_sent"] = False
                        result["telegram_response"] = {
                            "status": "FAILED",
                            "http_status": resp.status,
                            "error": resp_data.get("description", "Unknown error"),
                        }
                        result["errors"].append(f"Telegram API error: {resp_data.get('description')}")
        
        except Exception as e:
            result["telegram_sent"] = False
            result["telegram_response"] = {"status": "EXCEPTION", "error": str(e)}
            result["errors"].append(f"Telegram exception: {str(e)}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL: GENERATE SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    result["step"] = "complete"
    
    if result["telegram_sent"]:
        result["overall_status"] = "✅ SUCCESS - Test TOP 10 sent to Telegram"
    elif result["message_built"]:
        result["overall_status"] = "⚠️ PARTIAL - Message built but Telegram send failed"
    else:
        result["overall_status"] = "❌ FAILED - Could not build message"
    
    return result


@router.get("/debug/money-game-top10")
@router.post("/debug/money-game-top10")
async def money_game_top10_endpoint():
    """
    🎯 MONEY GAME TOP 10 - FULL REAL PREDICTIONS
    
    This endpoint:
    1. Gets the Money Game TOP 10 stocks & crypto
    2. Runs REAL predictions for each symbol
    3. Sends a properly formatted Telegram message
    
    This is what the 8 AM message SHOULD look like!
    """
    import aiohttp
    import os
    from datetime import datetime
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "step": "init",
        "top10_stocks": [],
        "top10_crypto": [],
        "predictions": {},
        "telegram_sent": False,
        "errors": [],
    }
    
    # Step 1: Use RELIABLE fallback symbols (Money Game often empty)
    # These are high-quality, liquid symbols with good ML signal
    FALLBACK_STOCKS = ["NVDA", "META", "PLTR", "COIN", "MSTR", "GOOGL", "AMZN", "HOOD", "TSLA", "AMD"]
    FALLBACK_CRYPTO = ["RNDR", "TURBO", "SOL", "BTC", "SUI", "ETH", "INJ", "XRP", "AVAX", "LINK"]
    
    # START with fallback symbols (guaranteed)
    result["top10_stocks"] = FALLBACK_STOCKS.copy()
    result["top10_crypto"] = FALLBACK_CRYPTO.copy()
    
    try:
        # Try to get Money Game symbols (may be better ranked)
        from core.money_game_engine import get_money_game
        mg = get_money_game()
        mg_stocks = mg.get_best_symbols_for_top10("stock", limit=10)
        mg_crypto = mg.get_best_symbols_for_top10("crypto", limit=10)
        
        # Only use Money Game if it returned REAL data
        if mg_stocks and len(mg_stocks) >= 5:
            result["top10_stocks"] = mg_stocks[:10]
            LOGGER.info(f"[MONEY-GAME-TOP10] Using Money Game stocks: {mg_stocks}")
        else:
            LOGGER.warning("[MONEY-GAME-TOP10] Money Game stocks empty, using FALLBACK")
            
        if mg_crypto and len(mg_crypto) >= 5:
            result["top10_crypto"] = mg_crypto[:10]
            LOGGER.info(f"[MONEY-GAME-TOP10] Using Money Game crypto: {mg_crypto}")
        else:
            LOGGER.warning("[MONEY-GAME-TOP10] Money Game crypto empty, using FALLBACK")
            
    except Exception as e:
        result["errors"].append(f"Money Game error (using fallback): {e}")
        LOGGER.warning(f"[MONEY-GAME-TOP10] Money Game exception, using FALLBACK: {e}")
    
    LOGGER.info(f"[MONEY-GAME-TOP10] Final stocks: {result['top10_stocks']}")
    LOGGER.info(f"[MONEY-GAME-TOP10] Final crypto: {result['top10_crypto']}")
    
    # Step 2: Run REAL predictions for each symbol
    result["step"] = "running_predictions"
    all_symbols = result["top10_stocks"] + result["top10_crypto"]
    
    for symbol in all_symbols:
        try:
            # BYPASS stock_engine - use the REAL turbo prediction engine
            # This matches what /api/predictions/run returns
            pred_result = await _run_turbo_prediction_for_top10(symbol)
            if pred_result and pred_result.get("ok"):
                result["predictions"][symbol] = {
                    "direction": pred_result.get("direction", "FLAT"),
                    "confidence": pred_result.get("confidence", 0),
                    "current_price": pred_result.get("current_price", 0),
                    "target_price": pred_result.get("target_price", 0),
                    "stop_loss": pred_result.get("stop_loss", 0),
                    "horizon_h": pred_result.get("horizon_h", 48),
                }
            else:
                result["predictions"][symbol] = {"direction": "FLAT", "confidence": 0, "error": pred_result.get("error", "prediction_failed")}
        except Exception as e:
            result["predictions"][symbol] = {"direction": "FLAT", "confidence": 0, "error": str(e)}
            LOGGER.warning(f"[MONEY-GAME-TOP10] Prediction failed for {symbol}: {e}")
    
    # Step 3: Build the message using the CORRECT formatter
    result["step"] = "building_message"
    
    # Convert predictions to format expected by format_top10_message
    from core.ghost_notifications import format_top10_message
    
    stock_picks = []
    for symbol in result["top10_stocks"][:10]:
        pred = result["predictions"].get(symbol, {})
        if pred.get("error"):
            continue
        
        # Pull ALL data from prediction - no recalculation, mirror exactly
        direction = pred.get("direction", "FLAT")
        confidence = pred.get("confidence", 0.5)
        price = pred.get("current_price", 0)
        target = pred.get("target_price", price)
        stop = pred.get("stop_loss", price * 0.97)
        hold_days = pred.get("hold_days", 3)  # From prediction engine
        hold_reason = pred.get("hold_reason", "swing_trade")
        news_influenced = pred.get("news_influenced", False)
        expected_move = pred.get("expected_move_pct", 0.03)
        volatility = pred.get("volatility", 0.02)
        
        stock_picks.append({
            "symbol": symbol,
            "direction": "UP" if direction == "UP" else "DOWN",
            "confidence": confidence,
            "current": price,
            "target_price": target,
            "prediction_48h": target,
            "stop": stop,
            "hold_days": hold_days,  # REAL from prediction
            "hold_reason": hold_reason,
            "news_influenced": news_influenced,  # REAL from prediction
            "expected_move_pct": expected_move,  # REAL from prediction
            "volatility": volatility,  # REAL from prediction
            "sentiment_score": 0,
        })
    
    crypto_picks = []
    for symbol in result["top10_crypto"][:10]:
        pred = result["predictions"].get(symbol, {})
        if pred.get("error"):
            continue
        
        # Pull ALL data from prediction - no recalculation, mirror exactly
        direction = pred.get("direction", "FLAT")
        confidence = pred.get("confidence", 0.5)
        price = pred.get("current_price", 0)
        target = pred.get("target_price", price)
        stop = pred.get("stop_loss", price * 0.97)
        hold_days = pred.get("hold_days", 2)  # From prediction engine
        hold_reason = pred.get("hold_reason", "swing_trade")
        news_influenced = pred.get("news_influenced", False)
        expected_move = pred.get("expected_move_pct", 0.03)
        volatility = pred.get("volatility", 0.02)
        
        crypto_picks.append({
            "symbol": symbol,
            "direction": "UP" if direction == "UP" else "DOWN",
            "confidence": confidence,
            "current": price,
            "target_price": target,
            "prediction_48h": target,
            "stop": stop,
            "hold_days": hold_days,  # REAL from prediction
            "hold_reason": hold_reason,
            "news_influenced": news_influenced,  # REAL from prediction
            "expected_move_pct": expected_move,  # REAL from prediction
            "volatility": volatility,  # REAL from prediction
            "sentiment_score": 0,
        })
    
    # Use the proper formatter (returns list of 2 messages: stocks + crypto)
    messages = format_top10_message(stock_picks, crypto_picks)
    result["message_preview"] = str(messages[0][:300] + "..." if messages else "")
    
    # Step 4: Send to Telegram (send both messages)
    result["step"] = "sending_telegram"
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if bot_token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
                for msg in messages:
                    payload = {
                        "chat_id": chat_id,
                        "text": msg,
                        "disable_web_page_preview": True,
                    }
                    async with session.post(url, json=payload) as resp:
                        resp_data = await resp.json()
                        
                        if resp.status == 200 and resp_data.get("ok"):
                            result["telegram_sent"] = True
                            result["message_id"] = resp_data.get("result", {}).get("message_id")
                            LOGGER.info(f"[MONEY-GAME-TOP10] ✅ Sent to Telegram: message_id={result['message_id']}")
                        else:
                            result["errors"].append(f"Telegram error: {resp_data.get('description')}")
        except Exception as e:
            result["errors"].append(f"Telegram exception: {e}")
        except Exception as e:
            result["errors"].append(f"Telegram exception: {e}")
    else:
        result["errors"].append("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set")
    
    result["step"] = "complete"
    result["overall_status"] = "✅ SUCCESS" if result["telegram_sent"] else "❌ FAILED"
    
    return result


@router.get("/debug/send-top10-now")
@router.post("/debug/send-top10-now")
async def send_top10_now_endpoint():
    """
    🚀 SEND TOP 10 NOW - FAST, RELIABLE
    
    Simple endpoint that:
    1. Uses hardcoded QUALITY symbols (no Money Game dependency)
    2. Runs REAL ML predictions for each
    3. Sends ONE combined message to Telegram (stocks + crypto)
    
    Has 60-second cooldown to prevent accidental duplicate sends.
    """
    import aiohttp
    import os
    from datetime import datetime
    
    global _LAST_TOP10_SEND_TIME
    
    # Check cooldown to prevent duplicate sends
    now = time.time()
    seconds_since_last = now - _LAST_TOP10_SEND_TIME
    if seconds_since_last < _TOP10_COOLDOWN_SECONDS:
        return {
            "ok": False,
            "error": f"Cooldown active. Wait {int(_TOP10_COOLDOWN_SECONDS - seconds_since_last)} more seconds.",
            "telegram_sent": False,
            "cooldown_remaining": int(_TOP10_COOLDOWN_SECONDS - seconds_since_last),
        }
    
    # Hardcoded HIGH-QUALITY symbols - liquid, well-known, good ML signal
    STOCKS = ["NVDA", "META", "PLTR", "COIN", "MSTR", "GOOGL", "AMZN", "HOOD", "TSLA", "AMD"]
    CRYPTO = ["RNDR", "TURBO", "SOL", "BTC", "SUI", "ETH", "INJ", "XRP", "AVAX", "LINK"]
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "stocks": STOCKS,
        "crypto": CRYPTO,
        "predictions": {},
        "telegram_sent": False,
        "errors": [],
    }
    
    # Run predictions for all symbols
    all_symbols = STOCKS + CRYPTO
    
    for symbol in all_symbols:
        try:
            pred = await _run_turbo_prediction_for_top10(symbol)
            if pred and pred.get("ok"):
                result["predictions"][symbol] = {
                    "direction": pred.get("direction", "FLAT"),
                    "confidence": pred.get("confidence", 0.5),
                    "current_price": pred.get("current_price", 0),
                    "target_price": pred.get("target_price", 0),
                    "stop_loss": pred.get("stop_loss", 0),
                    "hold_days": pred.get("hold_days", 3),
                    "hold_reason": pred.get("hold_reason", "swing_trade"),
                    "news_influenced": pred.get("news_influenced", False),
                    "news_headline": pred.get("news_headline"),
                }
            else:
                result["predictions"][symbol] = {"error": pred.get("error", "failed")}
        except Exception as e:
            result["predictions"][symbol] = {"error": str(e)}
    
    # Build formatted message
    from core.ghost_notifications import format_top10_message
    
    stock_picks = []
    for symbol in STOCKS:
        pred = result["predictions"].get(symbol, {})
        if pred.get("error"):
            continue
        
        # Pull ALL data from prediction - mirror exactly
        direction = pred.get("direction", "FLAT")
        
        # Skip FLAT signals - no conviction = no trade
        if direction == "FLAT":
            LOGGER.info(f"[TOP10] Skipping {symbol}: FLAT signal (no conviction)")
            continue
        
        confidence = pred.get("confidence", 0.5)
        price = pred.get("current_price", 0)
        target = pred.get("target_price", price)
        stop = pred.get("stop_loss", price * 0.97)
        hold_days = pred.get("hold_days", 3)
        hold_reason = pred.get("hold_reason", "swing_trade")
        news_influenced = pred.get("news_influenced", False)
        expected_move = pred.get("expected_move_pct", 0.03)
        volatility = pred.get("volatility", 0.02)
        
        stock_picks.append({
            "symbol": symbol,
            "direction": direction,  # Keep actual direction (UP or DOWN)
            "confidence": confidence,
            "current": price,
            "target_price": target,
            "prediction_48h": target,
            "stop": stop,
            "hold_days": hold_days,
            "hold_reason": hold_reason,
            "news_influenced": news_influenced,
            "expected_move_pct": expected_move,
            "volatility": volatility,
            "asset_type": "stock",
        })
    
    crypto_picks = []
    for symbol in CRYPTO:
        pred = result["predictions"].get(symbol, {})
        if pred.get("error"):
            continue
        
        # Pull ALL data from prediction - mirror exactly
        direction = pred.get("direction", "FLAT")
        
        # Skip FLAT signals - no conviction = no trade
        if direction == "FLAT":
            LOGGER.info(f"[TOP10] Skipping {symbol}: FLAT signal (no conviction)")
            continue
        
        confidence = pred.get("confidence", 0.5)
        price = pred.get("current_price", 0)
        target = pred.get("target_price", price)
        stop = pred.get("stop_loss", price * 0.97)
        hold_days = pred.get("hold_days", 3)
        hold_reason = pred.get("hold_reason", "swing_trade")
        news_influenced = pred.get("news_influenced", False)
        expected_move = pred.get("expected_move_pct", 0.03)
        volatility = pred.get("volatility", 0.02)
        
        crypto_picks.append({
            "symbol": symbol,
            "direction": direction,  # Keep actual direction (UP or DOWN)
            "confidence": confidence,
            "current": price,
            "target_price": target,
            "prediction_48h": target,
            "stop": stop,
            "hold_days": hold_days,
            "hold_reason": hold_reason,
            "news_influenced": news_influenced,
            "expected_move_pct": expected_move,
            "volatility": volatility,
            "asset_type": "crypto",
        })
    
    # ================================================================
    # V3 FILTER: Apply historical win rate scoring and inverse logic
    # ================================================================
    try:
        from core.ghost_notifications import v3_filter_and_score, V3_ENABLED
        if V3_ENABLED:
            LOGGER.info(f"[TOP10-V3] Applying V3 quality filter...")
            stock_picks = v3_filter_and_score(stock_picks)
            crypto_picks = v3_filter_and_score(crypto_picks)
            result["v3_applied"] = True
            result["v3_stocks_after"] = len(stock_picks)
            result["v3_crypto_after"] = len(crypto_picks)
            LOGGER.info(f"[TOP10-V3] ✅ After V3: {len(stock_picks)} stocks, {len(crypto_picks)} crypto")
        else:
            result["v3_applied"] = False
    except Exception as e:
        LOGGER.error(f"[TOP10-V3] V3 filter error: {e}")
        result["v3_applied"] = False
        result["v3_error"] = str(e)
    
    # Format returns list of 2 messages (stocks + crypto)
    messages = format_top10_message(stock_picks, crypto_picks)
    result["message_preview"] = str(messages[0][:300] + "..." if messages else "")
    result["full_message"] = messages[0] if messages else ""  # Return full message for debugging
    result["stock_picks_count"] = len(stock_picks)
    result["crypto_picks_count"] = len(crypto_picks)
    
    # Send to Telegram (send both messages)
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if bot_token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                for msg in messages:
                    payload = {
                        "chat_id": chat_id,
                        "text": msg,
                        "disable_web_page_preview": True,
                    }
                    async with session.post(url, json=payload) as resp:
                        resp_data = await resp.json()
                        
                        if resp.status == 200 and resp_data.get("ok"):
                            result["telegram_sent"] = True
                            result["message_id"] = resp_data.get("result", {}).get("message_id")
                            LOGGER.info(f"[SEND-TOP10-NOW] ✅ Sent to Telegram!")
                        else:
                            result["errors"].append(f"Telegram error: {resp_data.get('description')}")
        except Exception as e:
            result["errors"].append(f"Telegram exception: {e}")
    else:
        result["errors"].append("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set")
    
    # Update cooldown on successful send
    if result["telegram_sent"]:
        _LAST_TOP10_SEND_TIME = time.time()
    
    result["status"] = "✅ SUCCESS" if result["telegram_sent"] else "❌ FAILED"
    
    return result


@router.get("/debug/force-top10")
@router.post("/debug/force-top10")
async def force_top10_endpoint(force: bool = True):
    """
    FORCE REAL TOP 10 NOTIFICATION - REDIRECTS TO CLEAN ENDPOINT
    
    Jan 30, 2026: Now uses hardcoded quality symbols to prevent wrong symbols.
    
    Usage: GET or POST /debug/force-top10?force=true
    """
    from datetime import datetime
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "action": "force_top10_via_clean_endpoint",
        "force_mode": force,
    }
    
    try:
        # Use the CLEAN hardcoded symbols endpoint
        clean_result = await send_top10_now_endpoint()
        result["telegram_sent"] = clean_result.get("telegram_sent", False)
        result["predictions"] = clean_result.get("predictions", {})
        result["overall_status"] = "✅ Sent via clean endpoint!" if result["telegram_sent"] else "❌ Failed"
    except Exception as e:
        result["error"] = str(e)
        result["overall_status"] = f"❌ Error: {e}"
    
    return result


@router.get("/debug/advisor/positions")
async def advisor_get_positions():
    """
    Get all tracked positions with current status.
    
    Returns open positions, P&L, target progress, etc.
    """
    try:
        from core.ghost_advisor import get_advisor
        advisor = get_advisor()
        
        positions = []
        for pos in advisor.get_open_positions():
            positions.append({
                "symbol": pos.symbol,
                "direction": pos.direction,
                "asset_type": pos.asset_type,
                "entry_price": pos.entry_price,
                "current_price": pos.current_price,
                "target_price": pos.target_price,
                "stop_price": pos.stop_price,
                "pnl_pct": round(pos.pnl_pct, 2),
                "target_progress": round(pos.target_progress_pct, 0),
                "hours_remaining": round(pos.hours_remaining, 1),
                "status": pos.status.value,
            })
        
        stats = advisor.get_stats()
        
        return {
            "ok": True,
            "positions": positions,
            "open_count": len(positions),
            "stats": stats
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/debug/advisor/check-prices")
async def advisor_check_prices():
    """
    Check all open positions and send alerts if needed.
    
    This is what should run every 5-15 minutes!
    Call this manually to test, or let the cron handle it.
    """
    try:
        from core.ghost_advisor import get_advisor, check_all_positions, format_advisor_alert
        from core.asset_classifier import get_asset_type
        
        advisor = get_advisor()
        open_positions = advisor.get_open_positions()
        
        if not open_positions:
            return {"ok": True, "message": "No open positions to check", "checked": 0}
        
        # Price fetch function
        async def get_price(symbol: str, asset_type: str) -> float:
            try:
                if asset_type == "crypto":
                    from core.crypto.crypto_providers import get_crypto_price_quorum
                    result = await get_crypto_price_quorum(symbol, use_cache=False)
                    return result.get("price", 0) if result else 0
                else:
                    from core.providers.turbo_provider import get_turbo_provider
                    turbo = get_turbo_provider()
                    result = turbo.turbo_stock_price(symbol, max_budget_s=2.0)
                    return result.get("price", 0) if result.get("ok") else 0
            except Exception:
                return 0
        
        # Telegram send function
        def send_telegram(msg: str) -> bool:
            try:
                chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
                if chat_id:
                    return _tg_send_chat_message(chat_id, msg)
                return False
            except Exception:
                return False
        
        # Check each position
        alerts_sent = 0
        updates = []
        
        for pos in open_positions:
            new_price = await get_price(pos.symbol, pos.asset_type)
            if not new_price:
                continue
            
            result = advisor.update_price(pos.symbol, new_price)
            
            if result:
                alert_type, updated_pos = result
                message = format_advisor_alert(alert_type, updated_pos)
                success = send_telegram(message)
                
                if success:
                    alerts_sent += 1
                    updates.append({
                        "symbol": pos.symbol,
                        "alert_type": alert_type.value,
                        "sent": True
                    })
            else:
                updates.append({
                    "symbol": pos.symbol,
                    "price": new_price,
                    "pnl": round(pos.pnl_pct, 2),
                    "no_alert_needed": True
                })
        
        return {
            "ok": True,
            "checked": len(open_positions),
            "alerts_sent": alerts_sent,
            "updates": updates
        }
        
    except Exception as e:
        LOGGER.error(f"[ADVISOR] Check prices error: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}


@router.get("/debug/advisor/stats")
async def advisor_get_stats():
    """Get advisor performance statistics"""
    try:
        from core.ghost_advisor import get_advisor
        advisor = get_advisor()
        return {"ok": True, **advisor.get_stats()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/debug/advisor/test-alert")
async def advisor_test_alert():
    """
    Send a test advisor alert to Telegram.
    This verifies the alert system is working.
    """
    try:
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        if not chat_id:
            return {"ok": False, "error": "TELEGRAM_CHAT_ID not set"}
        
        test_msg = """🧪 GHOST ADVISOR TEST

━━━━━━━━━━━━━━━━━━━━━
✅ Alert system WORKING!

This confirms:
• Telegram connection ✓
• Message formatting ✓
• Advisor ready to send alerts

Ghost is watching your positions.
━━━━━━━━━━━━━━━━━━━━━"""
        
        success = _tg_send_chat_message(chat_id, test_msg)
        return {"ok": success, "message": "Test alert sent" if success else "Failed to send"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/debug/advisor/simulate-target-hit")
async def advisor_simulate_target_hit(symbol: str = None):
    """
    Simulate a target hit to test the alert flow.
    If no symbol given, uses first open position.
    """
    try:
        from core.ghost_advisor import get_advisor, format_advisor_alert, AlertType
        
        advisor = get_advisor()
        positions = advisor.get_open_positions()
        
        if not positions:
            return {"ok": False, "error": "No open positions"}
        
        # Find position
        if symbol:
            pos = advisor.get_position(symbol)
            if not pos:
                return {"ok": False, "error": f"Position {symbol} not found"}
        else:
            pos = positions[0]
        
        # Format what the alert would look like
        # Simulate price at target
        original_price = pos.current_price
        pos.current_price = pos.target_price
        
        alert_msg = format_advisor_alert(AlertType.TARGET_HIT, pos)
        
        # Reset price
        pos.current_price = original_price
        
        # Send to telegram
        chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        if not chat_id:
            return {"ok": False, "error": "TELEGRAM_CHAT_ID not set", "preview": alert_msg}
        
        success = _tg_send_chat_message(chat_id, f"[SIMULATION]\n{alert_msg}")
        
        return {
            "ok": success,
            "symbol": pos.symbol,
            "message": "Simulated target hit alert sent" if success else "Failed to send",
            "alert_preview": alert_msg[:500]
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/debug/notification-status")
async def notification_status_endpoint():
    """
    CHECK NOTIFICATION SYSTEM STATUS
    
    Shows:
    - Current Central time
    - Whether 8 AM has passed today
    - _LATEST_PREDICTIONS count
    - Last TOP 10 send time (if tracked)
    """
    from datetime import datetime
    try:
        from zoneinfo import ZoneInfo
        central_tz = ZoneInfo("America/Chicago")
    except ImportError:
        import pytz
        central_tz = pytz.timezone("America/Chicago")
    
    now_utc = datetime.utcnow()
    now_central = datetime.now(central_tz)
    
    return {
        "utc_time": now_utc.strftime("%Y-%m-%d %H:%M:%S UTC"),
        "central_time": now_central.strftime("%Y-%m-%d %H:%M:%S Central"),
        "central_hour": now_central.hour,
        "is_top10_hour": now_central.hour == 8,
        "predictions_count": len(_LATEST_PREDICTIONS) if _LATEST_PREDICTIONS else 0,
        "sample_predictions": list(_LATEST_PREDICTIONS.keys())[:10] if _LATEST_PREDICTIONS else [],
        "telegram_configured": bool(os.environ.get("TELEGRAM_BOT_TOKEN")) and bool(os.environ.get("TELEGRAM_CHAT_ID")),
        "scheduler_info": {
            "top10_scheduled_hour": 8,
            "update_hours": [12, 16, 20],
            "timezone": "America/Chicago (Central)"
        }
    }


@router.get("/api/debug/crypto-check/{symbol}")
async def debug_crypto_check(symbol: str):
    """Debug endpoint for Fix 5 - Crypto 24h change verification."""
    sym = symbol.upper()
    in_hunter = sym in HUNTER_CRYPTO_SYMBOLS
    in_crypto = sym in CRYPTO_SYMBOLS
    classified = _classify_symbol_category(sym)
    is_crypto = in_hunter or in_crypto or classified == "crypto"
    
    result = {
        "symbol": sym,
        "is_crypto": is_crypto,
        "classified": classified,
    }
    
    # Test the price endpoint
    try:
        live_result = await fetch_price_live(sym, strict_live=True)
        result["price"] = live_result.get("price") if live_result else None
        result["provider"] = live_result.get("provider") if live_result else None
        result["change_24h_pct"] = live_result.get("change_24h_pct") if live_result else None
    except Exception as e:
        result["error"] = str(e)
    
    return result


@router.get("/api/debug/predictions")
async def api_debug_predictions():
    """
    Debug endpoint to inspect in-memory predictions store.
    Shows what /api/predict/run writes and what /api/cockpit reads.
    """
    return {
        "store": _LATEST_PREDICTIONS,
        "keys": list(_LATEST_PREDICTIONS.keys()),
        "count": len(_LATEST_PREDICTIONS),
        "sample": list(_LATEST_PREDICTIONS.values())[:3] if _LATEST_PREDICTIONS else []
    }


