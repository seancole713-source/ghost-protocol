"""Routes: predict — extracted from wolf_app.py (Step 12)"""
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

# --- 12 endpoints ---

@router.post("/api/predict/run")
async def run_single_prediction_async(symbol: str) -> dict[str, Any]:
    """
    ASYNC version of core prediction function with turbo provider architecture.
    
    This function is the ASYNC HEART OF THE GHOST TURBO SURGERY.
    - Hard 4 second budget (3s price + 1s features)
    - Hard 8 second timeout (fast-fail to prevent hanging)
    - Uses turbo_stock_price/turbo_crypto_price with fast-fail
    - Always returns dict (never raises exceptions)
    - Returns structured error on any failure
    - NON-BLOCKING: Can handle multiple symbols concurrently
    
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
    # Run synchronous prediction in DEDICATED thread pool (max 2 workers)
    # This prevents prediction batch cycles from consuming the default
    # thread pool, which would block health checks and other API endpoints.
    from concurrent.futures import ThreadPoolExecutor
    if not hasattr(run_single_prediction_async, '_executor'):
        run_single_prediction_async._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ghost-predict")
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(run_single_prediction_async._executor, run_single_prediction, symbol)


@router.get("/api/predict/run")
async def api_predict_run_get(symbol: str):
    """
    Generate a new 48h prediction (GET version - no auth required).
    Bypasses POST model validation issues.
    """
    # Reuse POST logic
    body = _PredictRunBody(symbol=symbol.upper().strip())
    return await api_predict_run(body, credentials=None)


@router.post("/api/v3/predict/enhanced")
async def api_v3_predict_enhanced(
    symbol: str,
    use_cache: bool = True
):
    """
    Data-enhanced prediction using multi-source market intelligence.
    
    Aggregates data from:
    - CoinGecko (price, volume, market cap)
    - DEXScreener (liquidity, DEX metrics)
    - Fear & Greed Index (sentiment)
    - Technical indicators (RSI, trends)
    - CryptoPanic (news sentiment) if API key configured
    
    Returns prediction with:
    - Direction (UP/DOWN/FLAT)
    - Confidence score
    - Data quality percentage
    - Signal breakdown (bullish/bearish scores)
    - Raw market features
    
    Args:
        symbol: Crypto symbol (BTC, ETH, SOL, etc.)
        use_cache: Use cached data (default: True, 5min TTL)
    
    Example:
        POST /api/v3/predict/enhanced?symbol=BTC
        
        Response:
        {
            "ok": true,
            "symbol": "BTC",
            "direction": "UP",
            "confidence": 0.70,
            "data_quality": 0.714,
            "signals": {
                "bullish_score": 2,
                "bearish_score": 0,
                "rsi": 50.0,
                "trend": "SIDEWAYS",
                "sentiment": 0.0,
                "fear_greed": 22
            },
            "features": {
                "price": 89859.0,
                "volume_24h": 45000000000,
                "fear_greed_index": 22,
                "dex_liquidity": 6500661845,
                ...
            },
            "timestamp": 1733747584.23
        }
    """
    try:
        from core.data_enhanced_predictor import DataEnhancedPredictor
        
        async with DataEnhancedPredictor() as predictor:
            result = await predictor.predict_with_data(symbol.upper())
        
        return {
            "ok": True,
            "symbol": result["symbol"],
            "direction": result["direction"],
            "confidence": result["confidence"],
            "data_quality": result["data_quality"],
            "signals": result.get("signals", {}),
            "features": result.get("features", {}),
            "timestamp": time.time()
        }
        
    except Exception as e:
        LOGGER.error(f"Enhanced prediction failed for {symbol}: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "symbol": symbol.upper(),
            "timestamp": time.time()
        }


@router.get("/api/predict/series")
async def api_predict_series(
    symbol: str,
    since_hours: int = 72,
):
    """
    Get prediction series data for chart: forecast + actual prices.
    Returns aligned time series for overlay visualization.
    (Public read-only endpoint - no auth required)
    """
    symbol = symbol.upper().strip()
    if not symbol:
        raise HTTPException(400, "symbol required")

    try:
        # Get latest prediction
        pred = predictor.get_latest_prediction(symbol)
        if not pred:
            return {
                "symbol": symbol,
                "last_prediction": None,
                "forecast": [],
                "actual": [],
            }

        # Get forecast points (convert to milliseconds for JavaScript)
        forecast_pts = predictor.get_prediction_points(pred.id, kind="forecast")
        forecast = [{"ts": int(p.ts * 1000), "price": round(p.price, 4)} for p in forecast_pts]

        # Get actual points (convert to milliseconds for JavaScript)
        actual_pts = predictor.get_prediction_points(pred.id, kind="actual")
        actual = [{"ts": int(p.ts * 1000), "price": round(p.price, 4)} for p in actual_pts]

        return {
            "symbol": symbol,
            "last_prediction": {
                "id": pred.id,
                "run_at": int(pred.run_at * 1000),  # Convert to milliseconds
                "horizon_h": pred.horizon_h,
                "confidence": pred.confidence,
                "direction": pred.direction,
            },
            "forecast": forecast,
            "actual": actual,
        }

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Prediction series fetch failed for {symbol}: {e}", exc_info=True)
        raise HTTPException(500, f"Series fetch failed: {str(e)[:200]}")


@router.get("/api/predict/history")
async def api_predict_history(
    symbol: str,
    limit: int = 20,
):
    """
    Get prediction history with outcomes for scoreboard.
    Returns list of past predictions with accuracy metrics.
    (Public read-only endpoint - no auth required)
    """
    symbol = symbol.upper().strip()
    if not symbol:
        raise HTTPException(400, "symbol required")

    try:
        history = predictor.get_prediction_history(symbol, limit=min(limit, 100))

        # Format for API response (convert timestamps to milliseconds)
        results = []
        for h in history:
            row = {
                "id": h["id"],
                "run_at": int(h["run_at"] * 1000),  # Convert to milliseconds
                "confidence": h["confidence"],
                "direction": h["direction"],
                "closed": h["closed"],
            }

            if h["closed"]:
                row["closed_at"] = int(h["closed_at"] * 1000) if h["closed_at"] else None
                row["mae"] = h["mae"]
                row["map"] = h["map"]
                row["rmse"] = h["rmse"]
                row["hit_direction"] = h["hit_direction"]

            results.append(row)

        return results

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Prediction history fetch failed for {symbol}: {e}", exc_info=True)
        raise HTTPException(500, f"History fetch failed: {str(e)[:200]}")


@router.post("/api/predict/force")
async def api_predict_force(
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    Manually trigger multi-symbol prediction generation (bypasses scheduler).
    Useful for testing or immediate prediction updates.
    Requires authentication.
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        raise HTTPException(401, "Authentication required")

    try:
        from core import scheduled_predictions

        # Trigger manual prediction run
        scheduled_predictions.force_multi_prediction()

        return {
            "status": "triggered",
            "message": "Multi-symbol prediction generation started",
            "timestamp": time.time(),
        }

    except Exception as e:
        LOGGER.error(f"Manual prediction trigger failed: {e}", exc_info=True)
        raise HTTPException(500, f"Trigger failed: {str(e)[:200]}")


@router.get("/api/predict/scoreboard")
async def api_predict_scoreboard(
    symbol: str,
    windows: str = "7,30",
):
    """
    Get aggregate accuracy scoreboard for a symbol.
    Returns overall + windowed statistics.
    (Public read-only endpoint - no auth required)
    """
    symbol = symbol.upper().strip()
    if not symbol:
        raise HTTPException(400, "symbol required")

    try:
        window_list = [int(w.strip()) for w in windows.split(",") if w.strip().isdigit()]
        if not window_list:
            window_list = [7, 30]

        scoreboard = predictor.get_scoreboard(symbol, windows=window_list)
        return scoreboard

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Scoreboard fetch failed for {symbol}: {e}", exc_info=True)
        raise HTTPException(500, f"Scoreboard fetch failed: {str(e)[:200]}")


@router.post("/api/crypto/predict/run")
async def api_crypto_predict_run(
    symbol: str,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    Generate new crypto prediction (48h forecast)

    Returns:
        {
            "prediction_id": "uuid",
            "symbol": "BTC",
            "current_price": 43251.50,
            "direction": "UP",
            "confidence": 0.75,
            "forecast_h": 48,
            "path": [...],
            "bands": {...},
            "volatility": 0.048
        }
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    symbol = symbol.upper().strip()
    if not symbol:
        raise HTTPException(400, "symbol required")

    if not os.getenv("CRYPTO_ENABLED", "0") == "1":
        raise HTTPException(503, "Crypto module not enabled. Set CRYPTO_ENABLED=1")

    try:
        engine = _get_crypto_engine()

        # Time the prediction
        start_time = time.time()
        prediction = await engine.generate_prediction(symbol)
        duration = time.time() - start_time

        # Track metrics
        if _C_CRYPTO_PREDICT_DURATION is not None:
            try:
                _C_CRYPTO_PREDICT_DURATION.labels(symbol=symbol).observe(duration)
            except Exception:
                pass

        _add_event(
            "crypto.predict.run",
            f"Generated crypto prediction for {symbol}",
            {
                "symbol": symbol,
                "prediction_id": prediction.get("prediction_id"),
                "direction": prediction.get("direction"),
                "confidence": prediction.get("confidence"),
                "duration_s": round(duration, 2),
            },
        )

        return prediction

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Crypto prediction failed for {symbol}: {e}", exc_info=True)
        raise HTTPException(500, f"Prediction failed: {str(e)[:200]}")


@router.get("/api/crypto/predict/{symbol}")
async def api_crypto_predict_get(
    symbol: str,
    h: int = 48,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    Get latest crypto prediction or generate new one

    Query params:
        h: Forecast horizon in hours (default 48)

    Returns prediction with forecast path
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    symbol = symbol.upper().strip()
    if not symbol:
        raise HTTPException(400, "symbol required")

    if not os.getenv("CRYPTO_ENABLED", "0") == "1":
        raise HTTPException(503, "Crypto module not enabled")

    try:
        engine = _get_crypto_engine()

        # Try to get recent prediction from DB
        conn = sqlite3.connect(WOLF_SQLITE_PATH)
        c = conn.cursor()

        # Get most recent prediction within last hour
        one_hour_ago = time.time() - 3600
        c.execute(
            """
            SELECT id, run_at, confidence, direction, volatility
            FROM crypto_predictions
            WHERE symbol = ? AND run_at > ?
            ORDER BY run_at DESC
            LIMIT 1
        """,
            (symbol, one_hour_ago),
        )

        row = c.fetchone()

        if row:
            # Have recent prediction - fetch full data
            pred_id, run_at, confidence, direction, volatility = row

            # Get forecast points
            c.execute(
                """
                SELECT ts, price, price_low, price_high, confidence
                FROM crypto_forecast_points
                WHERE prediction_id = ?
                ORDER BY ts
            """,
                (pred_id,),
            )

            points = c.fetchall()
            conn.close()

            # Extract entry price from path[0] for convenience
            entry_price = points[0][1] if points else 0
            
            return {
                "prediction_id": pred_id,
                "symbol": symbol,
                "forecast_h": h,
                "trend": direction,
                "direction": direction,  # Add direction alias
                "confidence": confidence * 100 if confidence < 2 else confidence,
                "volatility": volatility,
                "run_at": run_at,
                "entry_price": entry_price,  # Add entry_price at top level
                "price_at_prediction": entry_price,  # Alias
                "path": [
                    {"ts": p[0], "price": p[1], "low": p[2], "high": p[3], "confidence": p[4]}
                    for p in points
                ],
            }
        else:
            conn.close()
            # No recent prediction - generate new one
            prediction = await engine.generate_prediction(symbol)
            
            # Ensure entry_price is in response
            if isinstance(prediction, dict):
                if "path" in prediction and prediction["path"] and "entry_price" not in prediction:
                    prediction["entry_price"] = prediction["path"][0].get("price", 0)
                    prediction["price_at_prediction"] = prediction["entry_price"]
                if "direction" not in prediction and "trend" in prediction:
                    prediction["direction"] = prediction["trend"]
            
            return prediction

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Crypto predict get failed for {symbol}: {e}", exc_info=True)
        raise HTTPException(500, f"Predict get failed: {str(e)[:200]}")


@router.get("/predict/48h")
async def predict_48h():
    """Return a 48-hour price and PnL cone forecast for WOLF.
    Response schema:
    { ticker, as_of, horizon_h, step_h, points: [{t, price_mid, price_lo, price_hi, pnl_mid, pnl_lo, pnl_hi}], summary }
    """
    global PRED_CALLS_TOTAL, PRED_LAST_TS
    try:
        PRED_CALLS_TOTAL += 1
        PRED_LAST_TS = time.time()
    except Exception:
        pass
    data = _build_forecast_series(48)
    return data


@router.post("/predict/feedback")
async def predict_feedback(body: PredFeedbackBody):
    """Collect realized outcomes for lightweight calibration metrics.
    Stores in-memory ring buffer; non-persistent by design.
    """
    rec = {
        "t": int(body.t),
        "actual_price": (float(body.actual_price) if body.actual_price is not None else None),
        "actual_pnl": float(body.actual_pnl) if body.actual_pnl is not None else None,
        "horizon_h": int(body.horizon_h or 0),
        "ctx": body.ctx or {},
        "ingested_ts": int(time.time()),
    }
    try:
        PRED_FEEDBACK.append(rec)
    except Exception:
        pass
    return {"ok": True, "size": len(PRED_FEEDBACK)}


@router.get("/predict/metrics")
async def predict_metrics():
    """Simple counters and last few feedback items for visibility."""
    try:
        last_items = list(PRED_FEEDBACK)[-10:]
    except Exception:
        last_items = []
    return {
        "calls_total": PRED_CALLS_TOTAL,
        "last_call_ts": int(PRED_LAST_TS or 0),
        "feedback_count": len(PRED_FEEDBACK),
        "feedback_tail": last_items,
    }


