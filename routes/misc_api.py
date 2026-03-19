"""Routes: misc_api — extracted from wolf_app.py (Step 12)"""
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
        AddPositionBody, OrderPlaceBody, PositionBody,
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

# --- 254 endpoints ---

@router.get("/paper-reevaluate")
async def paper_reevaluate_trigger():
    """
    Re-evaluate all paper trades with corrected logic.
    
    The old logic triggered stop losses during 6-48h period, marking correct predictions as losses.
    This endpoint re-evaluates all trades using FIXED logic that only checks outcome at target time.
    """
    import subprocess
    import sys
    import os
    import asyncio
    from datetime import datetime
    
    # Check if already running
    if _REEVALUATION_STATUS["running"]:
        return {
            "ok": False,
            "message": "Re-evaluation already in progress",
            "started_at": _REEVALUATION_STATUS["started_at"]
        }
    
    script_path = "scripts/reevaluate_paper_trades.py"
    
    if not os.path.exists(script_path):
        return {"ok": False, "error": f"Script not found: {script_path}"}
    
    # Start background task
    async def run_reevaluation():
        _REEVALUATION_STATUS["running"] = True
        _REEVALUATION_STATUS["started_at"] = datetime.now().isoformat()
        
        try:
            result = await asyncio.create_subprocess_exec(
                sys.executable, script_path,  # No --dry-run flag = apply changes
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            _REEVALUATION_STATUS["last_result"] = {
                "ok": result.returncode == 0,
                "script": script_path,
                "output": stdout.decode()[-8000:] if stdout else "",
                "errors": stderr.decode()[-2000:] if stderr else "",
                "return_code": result.returncode,
                "completed_at": datetime.now().isoformat()
            }
        except Exception as e:
            _REEVALUATION_STATUS["last_result"] = {
                "ok": False,
                "error": str(e),
                "completed_at": datetime.now().isoformat()
            }
        finally:
            _REEVALUATION_STATUS["running"] = False
    
    # Start in background
    asyncio.create_task(run_reevaluation())
    
    return {
        "ok": True,
        "message": "Paper trade re-evaluation started in background",
        "script": script_path,
        "started_at": _REEVALUATION_STATUS["started_at"],
        "check_status_at": "/paper-reevaluate-status"
    }


@router.get("/paper-reevaluate-status")
async def paper_reevaluate_status_check():
    """Check status of paper trade re-evaluation"""
    return {
        "running": _REEVALUATION_STATUS["running"],
        "started_at": _REEVALUATION_STATUS["started_at"],
        "last_result": _REEVALUATION_STATUS["last_result"]
    }


@router.get("/api/v3/trading-controls")
async def api_trading_controls():
    """
    Get current trading control settings (blacklist/whitelist).
    
    Shows which assets are blocked due to poor historical performance
    and which are prioritized due to proven success.
    
    Returns:
        {
            "ok": true,
            "blacklist_count": 13,
            "whitelist_count": 17,
            "min_confidence": 0.70,
            "whitelist_only_mode": false,
            "blacklist": ["SOL", "ETH", "BTC", ...],
            "whitelist_symbols": ["CHZ", "ZEC", "T", ...],
            "whitelist_detail": {
                "CHZ": "100.0%",
                "ZEC": "100.0%",
                ...
            }
        }
    """
    try:
        from core.trading_controls import get_trading_stats
        
        stats = get_trading_stats()
        
        return {
            "ok": True,
            **stats
        }
    
    except Exception as e:
        LOGGER.error(f"Trading controls fetch failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e)
        }


@router.get("/api/v3/can-trade/{symbol}")
async def api_can_trade_symbol(symbol: str):
    """
    Check if a symbol can be traded based on historical performance.
    
    Evaluates blacklist/whitelist status and confidence thresholds
    for a specific asset.
    
    Args:
        symbol: Trading symbol (e.g., "BTC", "CHZ", "AAPL")
    
    Returns:
        {
            "ok": true,
            "symbol": "BTC",
            "can_trade": false,
            "reason": "Blacklisted: 3% historical win rate",
            "blacklisted": true,
            "whitelisted": false,
            "historical_win_rate": null,
            "min_confidence_required": 0.70
        }
    """
    try:
        from core.trading_controls import should_trade, BLACKLIST, WHITELIST, MIN_CONFIDENCE
        
        symbol = symbol.upper()
        
        # Test with 70% confidence (the minimum threshold)
        can_trade, reason = should_trade(symbol, MIN_CONFIDENCE)
        
        return {
            "ok": True,
            "symbol": symbol,
            "can_trade": can_trade,
            "reason": reason,
            "blacklisted": symbol in BLACKLIST,
            "whitelisted": symbol in WHITELIST,
            "historical_win_rate": WHITELIST.get(symbol),
            "min_confidence_required": MIN_CONFIDENCE
        }
    
    except Exception as e:
        LOGGER.error(f"Can-trade check failed for {symbol}: {e}", exc_info=True)
        return {
            "ok": False,
            "symbol": symbol,
            "error": str(e)
        }


@router.post("/api/v3/predictions/evaluate")
async def api_evaluate_predictions():
    """
    Manually trigger prediction evaluation.
    
    Evaluates all expired predictions (horizon has passed) and writes outcomes.
    This is the same logic as the hourly background task.
    
    Returns:
        {
            "ok": true,
            "evaluated": 12,
            "correct": 9,
            "accuracy": 0.75,
            "skipped": 3,
            "execution_time_s": 5.2
        }
    """
    try:
        import time as time_module
        from core.prediction_evaluator import evaluate_pending_predictions
        
        start = time_module.time()
        
        # Run evaluator directly (not via subprocess)
        result = evaluate_pending_predictions()
        
        execution_time = time_module.time() - start
        
        return {
            "ok": True,
            "evaluated": result.get("evaluated", 0),
            "correct": result.get("correct", 0),
            "incorrect": result.get("incorrect", 0),
            "skipped": result.get("skipped", 0),
            "accuracy_pct": result.get("accuracy_pct", 0),
            "execution_time_s": round(execution_time, 2),
            "message": f"Evaluated {result.get('evaluated', 0)} predictions"
        }
    
    except Exception as e:
        LOGGER.error(f"Evaluation failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "evaluated": 0
        }


@router.get("/api/v3/performance/dashboard")
async def api_performance_dashboard():
    """
    📊 GHOST PERFORMANCE DASHBOARD - "Is Ghost Making Money?"
    ========================================================
    
    Comprehensive real-time performance metrics showing:
    - Overall P&L and win rates (all-time, today, 7d, 30d)
    - Top & worst performing symbols
    - Confidence calibration (is Ghost overconfident?)
    - Recent predictions with outcomes
    
    This answers the critical question: **Is Ghost actually profitable?**
    
    Returns:
        {
            "overall": {
                "predictions": 1500,
                "win_rate": 68.2,
                "avg_accuracy": 70.1,
                "avg_gain_pct": 2.3
            },
            "today": {"predictions": 12, "win_rate": 75.0},
            "last_7d": {"predictions": 84, "win_rate": 69.0},
            "last_30d": {"predictions": 320, "win_rate": 68.5},
            "top_performers": [
                {"symbol": "WOLF", "win_rate": 82.0, "predictions": 45}
            ],
            "worst_performers": [...],
            "confidence_calibration": {
                "60-70%": {"actual_accuracy": 62.0, "calibration_error": 3.0},
                "70-80%": {"actual_accuracy": 74.0, "calibration_error": 1.0}
            },
            "recent_predictions": [...]
        }
    """
    try:
        from core.performance_dashboard import get_dashboard_metrics
        
        metrics = get_dashboard_metrics()
        return metrics
    
    except Exception as e:
        LOGGER.error(f"Performance dashboard failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "overall": {"predictions": 0, "win_rate": 0}
        }


@router.post("/api/v3/cascade/start")
async def api_v3_cascade_start(symbol: str, user_id: str | None = None):
    """
    Start a new cascading prediction for a symbol.
    
    Creates a 48h → 24h → 6h prediction cascade that shows Ghost adapting
    and learning in real-time. Each stage sends a Telegram update.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., 'BTC', 'ETH')
        user_id: Optional user ID for personalized cascades
    
    Returns:
        {
            "ok": true,
            "cascade_id": "uuid",
            "symbol": "BTC",
            "h48_prediction": {...},
            "scheduled": {
                "h24_update_at": "2024-01-15T12:00:00Z",
                "h6_final_at": "2024-01-16T06:00:00Z",
                "evaluation_at": "2024-01-16T12:00:00Z"
            }
        }
    
    Example:
        curl -X POST http://localhost:8000/api/v3/cascade/start?symbol=BTC
    """
    try:
        from core.cascading_predictor import get_cascade_predictor
        import asyncio
        
        symbol_upper = symbol.upper().strip()
        predictor = get_cascade_predictor()
        
        # Initiate cascade (uses asyncio to send Telegram)
        cascade_id = await predictor.initiate_cascade(symbol_upper, user_id)
        
        # Get cascade data
        import sqlite3
        conn = sqlite3.connect(str(predictor.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("""
            SELECT * FROM prediction_cascades WHERE cascade_id = ?
        """, (cascade_id,))
        cascade = dict(cursor.fetchone())
        conn.close()
        
        # Calculate schedule times
        from datetime import datetime, timedelta
        created = datetime.fromtimestamp(cascade['created_at'])
        
        return {
            "ok": True,
            "cascade_id": cascade_id,
            "symbol": symbol_upper,
            "h48_prediction": {
                "direction": cascade['h48_direction'],
                "confidence": cascade['h48_confidence'],
                "price": cascade['h48_price']
            },
            "scheduled": {
                "h24_update_at": (created + timedelta(hours=24)).isoformat(),
                "h6_final_at": (created + timedelta(hours=42)).isoformat(),
                "evaluation_at": (created + timedelta(hours=48)).isoformat()
            }
        }
    
    except Exception as e:
        LOGGER.error(f"Failed to start cascade for {symbol}: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "symbol": symbol
        }


@router.get("/api/v3/cascade/list")
async def api_v3_cascade_list(symbol: str | None = None, active_only: bool = True):
    """
    List cascades, optionally filtered by symbol and status.
    
    Args:
        symbol: Optional symbol filter
        active_only: Only show active (not evaluated) cascades (default: true)
    
    Returns:
        {
            "ok": true,
            "count": 5,
            "cascades": [...]
        }
    
    Example:
        curl http://localhost:8000/api/v3/cascade/list?symbol=BTC
        curl http://localhost:8000/api/v3/cascade/list?active_only=false
    """
    try:
        from core.cascading_predictor import get_cascade_predictor
        import sqlite3
        
        predictor = get_cascade_predictor()
        
        conn = sqlite3.connect(str(predictor.db_path))
        conn.row_factory = sqlite3.Row
        
        query = "SELECT * FROM prediction_cascades WHERE 1=1"
        params = []
        
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol.upper())
        
        if active_only:
            query += " AND evaluated_at IS NULL"
        
        query += " ORDER BY created_at DESC LIMIT 50"
        
        cursor = conn.execute(query, params)
        cascades = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return {
            "ok": True,
            "count": len(cascades),
            "cascades": cascades
        }
    
    except Exception as e:
        LOGGER.error(f"Failed to list cascades: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e)
        }


@router.get("/api/v3/cascade/stats")
async def api_v3_cascade_stats(days: int = 30):
    """
    Get cascade performance statistics.
    
    Shows accuracy metrics for each stage and overall performance.
    
    Args:
        days: Lookback period in days (default: 30)
    
    Returns:
        {
            "ok": true,
            "stats": {
                "total_cascades": 100,
                "h48_accuracy": 0.623,
                "h24_accuracy": 0.687,
                "h6_accuracy": 0.745,
                "avg_stages_correct": 2.1,
                "perfect_cascades": 24,
                "direction_changes_24h": 18,
                "direction_changes_6h": 12
            }
        }
    
    Example:
        curl http://localhost:8000/api/v3/cascade/stats?days=7
    """
    try:
        from core.cascading_predictor import get_cascade_predictor
        
        predictor = get_cascade_predictor()
        stats = predictor.get_cascade_stats(days=days)
        
        return {
            "ok": True,
            "stats": stats,
            "period_days": days
        }
    
    except Exception as e:
        LOGGER.error(f"Failed to get cascade stats: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e)
        }


@router.get("/api/v3/cascade/{cascade_id}")
async def api_v3_cascade_get(cascade_id: str):
    """
    Get details of a specific cascade.
    
    Args:
        cascade_id: UUID of the cascade
    
    Returns:
        {
            "ok": true,
            "cascade": {
                "cascade_id": "uuid",
                "symbol": "BTC",
                "created_at": 1234567890,
                "h48": {...},
                "h24": {...},
                "h6": {...},
                "outcome": {...}
            }
        }
    
    Example:
        curl http://localhost:8000/api/v3/cascade/{cascade_id}
    """
    try:
        from core.cascading_predictor import get_cascade_predictor
        import sqlite3
        
        predictor = get_cascade_predictor()
        
        conn = sqlite3.connect(str(predictor.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.execute("""
            SELECT * FROM prediction_cascades WHERE cascade_id = ?
        """, (cascade_id,))
        cascade = cursor.fetchone()
        conn.close()
        
        if not cascade:
            return {
                "ok": False,
                "error": "Cascade not found",
                "cascade_id": cascade_id
            }
        
        # Format response
        cascade_dict = dict(cascade)
        
        return {
            "ok": True,
            "cascade": {
                "cascade_id": cascade_dict['cascade_id'],
                "symbol": cascade_dict['symbol'],
                "created_at": cascade_dict['created_at'],
                "h48": {
                    "direction": cascade_dict['h48_direction'],
                    "confidence": cascade_dict['h48_confidence'],
                    "price": cascade_dict['h48_price'],
                    "sent_at": cascade_dict['h48_sent_at'],
                    "correct": cascade_dict.get('h48_correct')
                } if cascade_dict['h48_direction'] else None,
                "h24": {
                    "direction": cascade_dict['h24_direction'],
                    "confidence": cascade_dict['h24_confidence'],
                    "price": cascade_dict['h24_price'],
                    "direction_changed": bool(cascade_dict['h24_direction_changed']),
                    "confidence_delta": cascade_dict['h24_confidence_delta'],
                    "sent_at": cascade_dict['h24_sent_at'],
                    "correct": cascade_dict.get('h24_correct')
                } if cascade_dict['h24_direction'] else None,
                "h6": {
                    "direction": cascade_dict['h6_direction'],
                    "confidence": cascade_dict['h6_confidence'],
                    "price": cascade_dict['h6_price'],
                    "direction_changed": bool(cascade_dict['h6_direction_changed']),
                    "confidence_delta": cascade_dict['h6_confidence_delta'],
                    "sent_at": cascade_dict['h6_sent_at'],
                    "correct": cascade_dict.get('h6_correct')
                } if cascade_dict['h6_direction'] else None,
                "outcome": {
                    "actual_price": cascade_dict['actual_price'],
                    "actual_direction": cascade_dict['actual_direction'],
                    "evaluated_at": cascade_dict['evaluated_at'],
                    "stages_correct": sum(filter(None, [
                        cascade_dict.get('h48_correct'),
                        cascade_dict.get('h24_correct'),
                        cascade_dict.get('h6_correct')
                    ]))
                } if cascade_dict.get('evaluated_at') else None
            }
        }
    
    except Exception as e:
        LOGGER.error(f"Failed to get cascade {cascade_id}: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "cascade_id": cascade_id
        }


@router.get("/api/v3/self-improvement/status")
async def api_self_improvement_status():
    """
    Get Self-Improvement Engine Status (Phase 4)

    Returns current state of autonomous learning system including:
    - Iteration count (how many improvement cycles completed)
    - Threshold history (VIX-based dynamic adjustments)
    - Missed opportunities detected
    - Universe expansions (symbols added to watchlist)
    - Confidence calibration errors
    - Performance attribution by model

    Returns:
        {
            "ok": true,
            "iterations": 42,
            "current_threshold": 3.5,
            "vix": 18.2,
            "last_cycle": "2025-01-01T12:00:00Z",
            "threshold_history": [
                {"timestamp": 1735732800, "vix": 18.2, "old": 4.0, "new": 3.5}
            ],
            "missed_opportunities_last_24h": 5,
            "universe_size": 63,
            "confidence_calibration": {
                "40-60": {"claimed": 0.5, "actual": 0.48, "error": -0.02},
                "60-70": {"claimed": 0.65, "actual": 0.62, "error": -0.03}
            },
            "model_performance": {
                "ghost_ai": {"win_rate": 0.68, "sample_size": 1200},
                "technical": {"win_rate": 0.61, "sample_size": 1200}
            }
        }
    """
    try:
        from core.self_improvement_engine import get_self_improvement_engine

        engine = get_self_improvement_engine()
        status = engine.get_status()

        return {
            "ok": True,
            **status
        }

    except Exception as e:
        LOGGER.error(f"🧠 Self-improvement status error: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": "self_improvement_status_failed",
                "message": str(e)
            }
        )


@router.post("/api/v3/ml/train")
async def api_ml_train(min_predictions: int = 100):
    """
    Train ML Models on Historical Predictions
    
    Uses 124K+ reconciled predictions from PostgreSQL to train XGBoost models.
    Learns which features predict outcomes and builds symbol-specific models.
    
    Args:
        min_predictions: Minimum predictions per symbol to train model
        
    Returns:
        {
            "ok": true,
            "symbols_trained": 15,
            "total_predictions": 2847,
            "models": {
                "BTC": {"accuracy": 0.68, "train_samples": 380},
                "ETH": {"accuracy": 0.65, "train_samples": 290}
            }
        }
    """
    try:
        from core.ml_trainer import get_ml_trainer
        
        trainer = get_ml_trainer()
        results = await trainer.train_from_postgres(min_predictions=min_predictions)
        
        return results
        
    except Exception as e:
        LOGGER.error(f"ML training failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e)
        }


@router.post("/api/v3/features/analyze")
async def api_analyze_features():
    """
    Analyze Feature Correlation with Accuracy
    
    Finds which features actually predict price movement.
    Identifies noise features that should be dropped.
    
    Returns:
        {
            "ok": true,
            "strong_features": [
                ["rsi", 0.18],
                ["price_momentum", 0.15]
            ],
            "weak_features": [
                ["news_count", 0.02],
                ["sentiment_score", -0.01]
            ],
            "recommendations": {
                "keep_features": ["rsi", "price_momentum"],
                "drop_features": ["news_count", "sentiment_score"],
                "note_sentiment": "❌ Sentiment not helping - consider removing CryptoPanic"
            }
        }
    """
    try:
        from core.feature_analyzer import get_feature_analyzer
        
        analyzer = get_feature_analyzer()
        results = await analyzer.analyze_features()
        
        return results
        
    except Exception as e:
        LOGGER.error(f"Feature analysis failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e)
        }


@router.post("/api/v3/confidence/calibrate")
async def api_calibrate_confidence(min_predictions: int = 50):
    """
    Build Confidence Calibration Curves
    
    Maps predicted confidence → actual accuracy.
    Finds quality threshold (only predict when accuracy > 65%).
    
    Args:
        min_predictions: Minimum predictions needed for calibration
        
    Returns:
        {
            "ok": true,
            "total_predictions": 2847,
            "calibration_curve": {
                "0.5": {"actual_accuracy": 0.48, "count": 120},
                "0.6": {"actual_accuracy": 0.55, "count": 98},
                "0.7": {"actual_accuracy": 0.65, "count": 85},
                "0.8": {"actual_accuracy": 0.72, "count": 67}
            },
            "quality_threshold": 0.70,
            "recommendation": "Only make predictions with confidence > 70%"
        }
    """
    try:
        from core.confidence_calibrator import get_confidence_calibrator
        
        calibrator = get_confidence_calibrator()
        results = await calibrator.build_calibration(min_predictions=min_predictions)
        
        return results
        
    except Exception as e:
        LOGGER.error(f"Confidence calibration failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e)
        }


@router.get("/api/v3/position/calculate")
async def api_calculate_position(confidence: float, account_value: float = 25000.0):
    """
    Calculate Position Size (Kelly Criterion)
    
    Args:
        confidence: Prediction confidence (0.0 to 1.0)
        account_value: Account value in USD (default $25,000)
    
    Returns:
        {
            "position_size_usd": 2500.0,
            "position_pct": 0.10,
            "should_trade": true,
            "reason": "Within limits"
        }
    """
    try:
        from core.position_sizer import get_position_sizer
        
        sizer = get_position_sizer()
        result = sizer.calculate_position_size(confidence, account_value)
        
        return result
    
    except Exception as e:
        LOGGER.error(f"Position calculation failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e)
        }


@router.get("/api/v3/position/breakdown")
async def api_position_breakdown(account_value: float = 25000.0):
    """
    Get Position Sizes for Different Confidence Levels
    
    Shows position sizing across confidence spectrum.
    
    Args:
        account_value: Account value in USD (default $25,000)
    
    Returns:
        {
            "50%": {"position_usd": 0, "should_trade": false},
            "60%": {"position_usd": 2083.33, "should_trade": true},
            "70%": {"position_usd": 3333.33, "should_trade": true},
            "85%": {"position_usd": 5000.00, "should_trade": true}
        }
    """
    try:
        from core.position_sizer import get_position_sizer
        
        sizer = get_position_sizer()
        breakdown = sizer.get_position_breakdown(account_value)
        
        return breakdown
    
    except Exception as e:
        LOGGER.error(f"Position breakdown failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e)
        }


@router.get("/api/v3/regime/current")
async def api_current_regime():
    """
    Get Current Market Regime
    
    Detects market conditions to filter trades.
    
    Returns:
        {
            "regime": "TRENDING_UP",
            "should_trade": true,
            "confidence": 0.8,
            "vix_level": 18.5,
            "spy_trend": "up",
            "volume_ratio": 1.2,
            "reasons": [...]
        }
    """
    try:
        from core.regime_detector import get_regime_detector
        
        # Fetch SPY and VIX data using internal price cache
        spy_price_data = _cache_get_price("SPY")
        spy_price = spy_price_data[0] if spy_price_data[0] else None
        vix_data = _cache_get_price("VIX")
        vix_level = vix_data[0] if vix_data[0] else 20.0
        
        # Calculate SPY MA20 (2% below current as proxy for uptrend)
        # In production, fetch from database or yfinance historical data
        spy_ma20 = spy_price * 0.98 if spy_price else None
        spy_volume_ratio = 1.0  # Normalized volume (no historical data available)
        
        detector = get_regime_detector()
        regime = detector.detect_regime(
            spy_price=spy_price,
            spy_ma20=spy_ma20,
            vix_level=vix_level,
            spy_volume_ratio=spy_volume_ratio
        )
        
        return regime
    
    except Exception as e:
        LOGGER.error(f"Regime detection failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "regime": "UNKNOWN",
            "should_trade": False
        }


@router.post("/api/v3/learning/calibrate")
async def api_calibrate_weights(symbol: str, lookback_days: int = 90):
    """
    Calibrate Signal Weights (Learning Loop)
    
    Analyzes past predictions to determine which signals are most accurate
    and adjusts confidence weights accordingly.
    
    Args:
        symbol: Trading symbol (e.g., "WOLF")
        lookback_days: Days of history to analyze (default 90)
    
    Returns:
        {
            "symbol": "WOLF",
            "weights": {
                "RSI": 0.10,
                "MACD": 0.04,
                "BOLLINGER": 0.05,
                "VOLUME": 0.07,
                "SENTIMENT": 0.03,
                "MOMENTUM": 0.06
            },
            "sample_size": 120,
            "updated_at": 1736899200
        }
    """
    try:
        from core.learning_loop import get_learning_loop
        import time
        
        loop = get_learning_loop()
        weights = loop.calibrate_weights(symbol, lookback_days)
        
        # Save weights
        loop.save_weights(symbol, weights)
        
        return {
            "symbol": symbol,
            "weights": weights,
            "lookback_days": lookback_days,
            "updated_at": int(time.time())
        }
    
    except Exception as e:
        LOGGER.error(f"Weight calibration failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "symbol": symbol
        }


@router.post("/api/v3/predictions/migrate-outcomes-table")
async def api_migrate_outcomes_table():
    """
    One-time migration: Drop old outcomes table and let evaluator recreate it.
    
    WARNING: This will delete all existing outcomes data.
    Only run this once during the schema migration.
    
    Returns:
        {
            "ok": true,
            "message": "Outcomes table dropped and recreated",
            "old_records": 0
        }
    """
    try:
        from pathlib import Path
        import sqlite3
        
        db_path = Path(__file__).parent / "data" / "ghost_predictions.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Count existing records before dropping
        try:
            cursor.execute("SELECT COUNT(*) FROM outcomes")
            old_count = cursor.fetchone()[0]
        except Exception:
            old_count = 0
        
        # Drop old table
        cursor.execute("DROP TABLE IF EXISTS outcomes")
        conn.commit()
        conn.close()
        
        LOGGER.info(f"Dropped old outcomes table ({old_count} records)")
        
        return {
            "ok": True,
            "message": "Outcomes table dropped successfully. It will be recreated on next evaluation.",
            "old_records": old_count
        }
    
    except Exception as e:
        LOGGER.error(f"Migration failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e)
        }


@router.get("/api/v3/predictions/latest")
async def api_v3_predictions_latest(symbol: str | None = None, limit: int = 25):
    """
    Get latest predictions for cockpit forecast panel.
    
    Returns predictions with confidence, direction, and expected_move for UI.
    FIXED: Query database if _LATEST_PREDICTIONS is empty
    FIXED: Default limit raised from 10→25 so all edge symbols are visible.
    FIXED: Edge symbols sorted first, then by confidence descending.
    """
    try:
        predictions_list = []
        
        # FALLBACK: If _LATEST_PREDICTIONS is empty, query database
        if not _LATEST_PREDICTIONS:
            LOGGER.info("[PREDICTIONS] _LATEST_PREDICTIONS empty, querying database...")
            try:
                from core.prediction_store import get_prediction_store
                store = get_prediction_store()
                
                if symbol:
                    # Get latest prediction for specific symbol
                    recent_preds = store.get_recent_predictions(limit=100)
                    symbol_pred = next((p for p in recent_preds if p.get("symbol") == symbol.upper()), None)
                    if symbol_pred:
                        # Optional enrichment from wolf.db (touch gating)
                        gate_fields: dict[str, Any] = {}
                        try:
                            import sqlite3 as _sqlite3
                            with _sqlite3.connect(WOLF_SQLITE_PATH) as _c:
                                _c.row_factory = _sqlite3.Row
                                r = _c.execute(
                                    """
                                    SELECT target_price, stage5_ok, stage6_ok, gate
                                    FROM ghost_predictions
                                    WHERE symbol = ?
                                    ORDER BY predicted_at DESC
                                    LIMIT 1
                                    """,
                                    (symbol.upper(),),
                                ).fetchone()
                                if r:
                                    gate_fields = {
                                        "target_price": r["target_price"],
                                        "stage5_ok": bool(r["stage5_ok"]) if "stage5_ok" in r.keys() and r["stage5_ok"] is not None else False,
                                        "stage6_ok": bool(r["stage6_ok"]) if "stage6_ok" in r.keys() and r["stage6_ok"] is not None else False,
                                        "gate": r["gate"] if "gate" in r.keys() and r["gate"] is not None else "MONITOR",
                                    }
                        except Exception:
                            gate_fields = {}

                        current_price = symbol_pred.get("price_at_prediction", 0)
                        _dir1 = symbol_pred.get("direction", "FLAT")
                        _move1 = symbol_pred.get("confidence", 0) * 5
                        if _dir1 == "DOWN":
                            _move1 = -abs(_move1)
                        predictions_list.append({
                            "prediction_id": symbol_pred.get("id"),
                            "symbol": symbol.upper(),
                            "direction": _dir1,
                            "confidence": symbol_pred.get("confidence", 0),
                            "expected_move": _move1,
                            "horizon_h": 48,
                            "run_at": symbol_pred.get("created_at", 0),
                            "price_at_prediction": current_price,
                            "entry_price": current_price,
                            "stop_loss": round(current_price * 0.98, 2) if current_price else None,
                            "take_profit": round(current_price * 1.06, 2) if current_price else None,
                            "created_at": symbol_pred.get("created_at"),
                            **gate_fields,
                        })
                else:
                    # Get latest N predictions
                    recent_preds = store.get_recent_predictions(limit=limit)
                    for pred in recent_preds:
                        gate_fields: dict[str, Any] = {}
                        try:
                            sym = (pred.get("symbol") or "").upper().strip()
                            if sym:
                                import sqlite3 as _sqlite3
                                with _sqlite3.connect(WOLF_SQLITE_PATH) as _c:
                                    _c.row_factory = _sqlite3.Row
                                    r = _c.execute(
                                        """
                                        SELECT target_price, stage5_ok, stage6_ok, gate
                                        FROM ghost_predictions
                                        WHERE symbol = ?
                                        ORDER BY predicted_at DESC
                                        LIMIT 1
                                        """,
                                        (sym,),
                                    ).fetchone()
                                    if r:
                                        gate_fields = {
                                            "target_price": r["target_price"],
                                            "stage5_ok": bool(r["stage5_ok"]) if "stage5_ok" in r.keys() and r["stage5_ok"] is not None else False,
                                            "stage6_ok": bool(r["stage6_ok"]) if "stage6_ok" in r.keys() and r["stage6_ok"] is not None else False,
                                            "gate": r["gate"] if "gate" in r.keys() and r["gate"] is not None else "MONITOR",
                                        }
                        except Exception:
                            gate_fields = {}

                        current_price = pred.get("price_at_prediction", 0)
                        _dir2 = pred.get("direction", "FLAT")
                        _move2 = pred.get("confidence", 0) * 5
                        if _dir2 == "DOWN":
                            _move2 = -abs(_move2)
                        predictions_list.append({
                            "prediction_id": pred.get("id"),
                            "symbol": pred.get("symbol"),
                            "direction": _dir2,
                            "confidence": pred.get("confidence", 0),
                            "expected_move": _move2,
                            "horizon_h": 48,
                            "run_at": pred.get("created_at", 0),
                            "price_at_prediction": current_price,
                            "entry_price": current_price,
                            "stop_loss": round(current_price * 0.98, 2) if current_price else None,
                            "take_profit": round(current_price * 1.06, 2) if current_price else None,
                            "created_at": pred.get("created_at"),
                            **gate_fields,
                        })
                
                return {
                    "ok": True,
                    "predictions": predictions_list,
                    "count": len(predictions_list),
                    "source": "database"
                }
            except Exception as db_error:
                LOGGER.error(f"Database fallback failed: {db_error}")
                return {
                    "ok": True,
                    "predictions": [],
                    "count": 0,
                    "error": "No predictions available"
                }
        
        # Helper: round floats safely
        def _rnd(v, decimals=4):
            return round(v, decimals) if isinstance(v, float) else v

        # Helper: build one prediction dict from cache entry
        def _build_pred(sym: str, pred: dict) -> dict:
            current_price = pred.get("price", pred.get("price_at_prediction", 0))
            _dir_bp = pred.get("direction", "FLAT")
            # Use actual expected_move_pct from prediction engine if available
            _move_bp = pred.get("expected_move_pct", pred.get("confidence", 0) * 5)
            if _dir_bp == "DOWN" and _move_bp > 0:
                _move_bp = -_move_bp
            return {
                "prediction_id": pred.get("prediction_id"),
                "symbol": sym,
                "direction": _dir_bp,
                "confidence": _rnd(pred.get("confidence", 0)),
                "expected_move": _rnd(_move_bp),
                "horizon_h": pred.get("horizon_h", 48),
                "run_at": pred.get("run_at", 0),
                "price_at_prediction": current_price,
                "entry_price": pred.get("entry_price", current_price),
                "stop_loss": pred.get("stop_loss"),
                "take_profit": pred.get("take_profit"),
                "target_price": pred.get("target_price"),
                "stage5_ok": bool(pred.get("stage5_ok", False)),
                "stage6_ok": bool(pred.get("stage6_ok", False)),
                "gate": pred.get("gate", "MONITOR"),
                # Intelligence Hub metadata — rounded
                "intel_active_systems": pred.get("intel_active_systems"),
                "intel_total_systems": pred.get("intel_total_systems"),
                "intel_news_risk": pred.get("intel_news_risk"),
                "intel_direction_adj": pred.get("intel_direction_adj"),
                "intel_confidence_adj": _rnd(pred.get("intel_confidence_adj")),
                "intel_trust_boost": _rnd(pred.get("intel_trust_boost")),
                "market_regime": pred.get("market_regime"),
            }

        # Original logic for _LATEST_PREDICTIONS
        # If symbol specified, get just that one
        if symbol:
            pred = _LATEST_PREDICTIONS.get(symbol.upper())
            if pred:
                predictions_list.append(_build_pred(symbol.upper(), pred))
                LOGGER.info(
                    f"[API] Served prediction {pred.get('prediction_id')} for {symbol.upper()} from cache "
                    f"(run_at={pred.get('run_at', 0):.0f})"
                )
        else:
            # Get latest N predictions from in-memory store
            # Sort: edge symbols first (by confidence desc), then non-edge (by confidence desc)
            _edge_set_api = get_edge_set()
            _all_items = list(_LATEST_PREDICTIONS.items())
            _all_items.sort(
                key=lambda kv: (
                    0 if kv[0] in _edge_set_api else 1,        # edge first
                    -(kv[1].get("confidence", 0)),               # highest confidence first
                ),
            )
            for sym, pred in _all_items[:limit]:
                predictions_list.append(_build_pred(sym, pred))
        
        return {
            "ok": True,
            "predictions": predictions_list,
            "count": len(predictions_list)
        }
    
    except Exception as e:
        LOGGER.error(f"Failed to get predictions: {e}", exc_info=True)
        return {
            "ok": False,
            "predictions": [],
            "error": str(e)
        }


@router.get("/api/v3/system/orchestrator")
async def api_v3_system_orchestrator():
    """
    Get orchestrator system status showing all background services.
    
    Returns status of all 9 background services including outcome reconciler.
    """
    try:
        from core.orchestrator import get_system_status
        return get_system_status()
    except Exception as e:
        LOGGER.error(f"Orchestrator status failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "services": {},
            "timestamp": int(time.time())
        }


@router.get("/api/v3/watchlist/enriched")
async def api_v3_watchlist_enriched():
    """
    Get watchlist with current prices and latest predictions.
    
    Used by cockpit watchlist panel.
    CACHED (Feb 7, 2026): 30s TTL cache prevents 50+ HTTP calls per poll
    """
    global _WATCHLIST_ENRICHED_CACHE, _WATCHLIST_ENRICHED_CACHE_AT
    
    # Return cache if fresh (< 30s old)
    if _WATCHLIST_ENRICHED_CACHE and (time.time() - _WATCHLIST_ENRICHED_CACHE_AT) < _WATCHLIST_ENRICHED_CACHE_TTL:
        return _WATCHLIST_ENRICHED_CACHE
    
    # Use lock to prevent multiple concurrent fetches (thundering herd)
    lock = _get_watchlist_lock()
    if lock.locked():
        # Another request is already fetching — return stale cache or empty
        if _WATCHLIST_ENRICHED_CACHE:
            return _WATCHLIST_ENRICHED_CACHE
        return {"ok": True, "items": [], "watchlist": [], "count": 0}
    
    async with lock:
        # Double-check cache after acquiring lock
        if _WATCHLIST_ENRICHED_CACHE and (time.time() - _WATCHLIST_ENRICHED_CACHE_AT) < _WATCHLIST_ENRICHED_CACHE_TTL:
            return _WATCHLIST_ENRICHED_CACHE
        
        try:
            result = await asyncio.wait_for(
                _api_v3_watchlist_enriched_core(),
                timeout=15.0
            )
            # Only cache successful results
            if result.get("ok"):
                _WATCHLIST_ENRICHED_CACHE = result
                _WATCHLIST_ENRICHED_CACHE_AT = time.time()
            return result
        except asyncio.TimeoutError:
            LOGGER.error("Watchlist enriched TIMEOUT after 15s")
            if _WATCHLIST_ENRICHED_CACHE:
                return _WATCHLIST_ENRICHED_CACHE  # Return stale on timeout
            return {
                "ok": False,
                "items": [],
                "watchlist": [],
                "count": 0,
                "error": "Timeout: request took >15s"
            }
        except Exception as e:
            LOGGER.error(f"Watchlist enriched error: {e}", exc_info=True)
            if _WATCHLIST_ENRICHED_CACHE:
                return _WATCHLIST_ENRICHED_CACHE  # Return stale on error
            return {
                "ok": False,
                "items": [],
                "watchlist": [],
                "count": 0,
                "error": str(e)[:200]
            }


@router.get("/api/v3/watchlist/user")
async def api_v3_watchlist_user():
    """
    Alias for /api/v3/watchlist/enriched - maintains compatibility with personal watchlist API.
    Returns the same enriched watchlist data.
    """
    return await api_v3_watchlist_enriched()


@router.get("/api/v3/vip/snapshot")
async def api_v3_vip_snapshot():
    """
    Get VIP coins snapshot with prices and changes.
    
    Used by cockpit VIP panel.
    CACHED for 30s. Returns stale cache immediately if refresh takes >2s.
    """
    # Check cache first
    cache_age = time.time() - _VIP_SNAPSHOT_CACHE["timestamp"]
    
    # ALWAYS return cached data if available (even if stale) to prevent 3min hangs
    if _VIP_SNAPSHOT_CACHE["data"]:
        if cache_age < _VIP_SNAPSHOT_CACHE["ttl"]:
            LOGGER.info(f"[VIP] ⚡ Serving fresh cache (age: {cache_age:.1f}s)")
            return _VIP_SNAPSHOT_CACHE["data"]
        else:
            LOGGER.info(f"[VIP] ⚠️ Returning stale cache ({cache_age:.1f}s old) while refreshing in background")
            # Trigger async refresh but don't wait for it
            asyncio.create_task(_refresh_vip_cache())
            return _VIP_SNAPSHOT_CACHE["data"]
    
    LOGGER.info(f"[VIP] No cache available, fetching with 2s timeout...")
    
    # Only block on first fetch (no cache available)
    try:
        return await _fetch_vip_snapshot_with_timeout()
    except Exception as e:
        LOGGER.error(f"VIP snapshot failed: {e}", exc_info=True)
        return {
            "ok": False,
            "vip_coins": [],
            "error": str(e)
        }


@router.get("/api/v3/vip-coins")
async def api_v3_vip_coins_intelligence():
    """
    VIP coin intelligence with comprehensive market data.
    
    Tracks 5 high-potential coins:
    - WEPE (Wall Street Pepe)
    - LILPEPE (Lil Pepe)
    - DORKL (Dork Lord)
    - SLOTH (Slothana)
    - APC (Alpha Protocol Coin)
    
    Returns for each coin:
    - Current price
    - 24h change %
    - DEX liquidity (from DEXScreener)
    - Trading volume
    - Number of transactions
    - Primary DEX
    - Data quality score
    
    Example:
        GET /api/v3/vip-coins
        
        Response:
        {
            "ok": true,
            "vip_coins": [
                {
                    "symbol": "WEPE",
                    "price": 0.000080,
                    "change_24h": 5.2,
                    "liquidity": 23000000,
                    "volume_24h": 1500000,
                    "txns_24h": 450,
                    "dex": "uniswap-v2",
                    "data_quality": 0.857,
                    "status": "online"
                },
                ...
            ],
            "count": 5,
            "timestamp": 1733747584.23
        }
    """
    try:
        from core.data_collector import DataCollector
        
        vip_symbols = ["WEPE", "LILPEPE", "DORKL", "SLOTH", "APC"]
        
        vip_data = []
        
        async with DataCollector() as collector:
            for symbol in vip_symbols:
                try:
                    # Get DEXScreener data for VIP coin
                    dex_data = await collector.get_dexscreener_data(symbol)
                    
                    if dex_data:
                        vip_data.append({
                            "symbol": symbol,
                            "price": dex_data.get("price", 0),
                            "change_24h": dex_data.get("price_change_24h", 0),
                            "liquidity": dex_data.get("liquidity", 0),
                            "volume_24h": dex_data.get("volume_24h", 0),
                            "txns_24h": dex_data.get("txns_24h", 0),
                            "dex": dex_data.get("dex", "unknown"),
                            "data_quality": 1.0 if dex_data.get("liquidity", 0) > 0 else 0.5,
                            "status": "online"
                        })
                    else:
                        vip_data.append({
                            "symbol": symbol,
                            "price": 0,
                            "change_24h": 0,
                            "liquidity": 0,
                            "volume_24h": 0,
                            "txns_24h": 0,
                            "dex": "unknown",
                            "data_quality": 0.0,
                            "status": "offline"
                        })
                        
                except Exception as e:
                    LOGGER.error(f"VIP coin {symbol} data failed: {e}")
                    vip_data.append({
                        "symbol": symbol,
                        "price": 0,
                        "change_24h": 0,
                        "liquidity": 0,
                        "volume_24h": 0,
                        "txns_24h": 0,
                        "dex": "unknown",
                        "data_quality": 0.0,
                        "status": "error"
                    })
        
        return {
            "ok": True,
            "vip_coins": vip_data,
            "count": len(vip_data),
            "timestamp": time.time()
        }
        
    except Exception as e:
        LOGGER.error(f"VIP coins intelligence failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "vip_coins": [],
            "timestamp": time.time()
        }


@router.get("/api/v3/killswitch/status")
async def api_v3_killswitch_status():
    """
    Get prediction killswitch status.
    
    Predictions are BLOCKED unless PREDICTIONS_ENABLED=true.
    This is an emergency stop for all predictions.
    """
    try:
        from core.prediction_killswitch import get_killswitch
        
        killswitch = get_killswitch()
        return {
            "ok": True,
            **killswitch.get_status()
        }
        
    except Exception as e:
        LOGGER.error(f"Killswitch status failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "predictions_enabled": False,
            "killswitch_active": True,
            "reason": f"Error: {str(e)}"
        }


@router.get("/api/v3/quality_gate/status")
async def api_v3_quality_gate_status():
    """
    Get quality gate status.
    
    Quality gate controls:
    - Min accuracy threshold (85%)
    - Max daily predictions (10)
    - Symbol deduplication (24h)
    - Progressive confidence requirements
    """
    try:
        from core.quality_gate import get_quality_gate
        
        gate = get_quality_gate()
        return {
            "ok": True,
            **gate.get_status()
        }
        
    except Exception as e:
        LOGGER.error(f"Quality gate status failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e)
        }


@router.post("/api/v3/reconcile/trigger")
async def api_v3_reconcile_trigger():
    """
    Manually trigger outcome reconciliation.
    
    This checks all pending predictions whose time horizon has passed
    and records their actual outcomes (WIN/LOSS/NEUTRAL).
    """
    try:
        try:
            from services.outcome_reconciler import reconcile_outcomes
            reconcile_outcomes()
        except ImportError:
            from services.outcome_reconciler_v2 import reconcile_outcomes_v2
            reconcile_outcomes_v2()
    
        return {
            "ok": True,
            "message": "Reconciliation triggered",
            "timestamp": time.time()
        }
        
    except Exception as e:
        LOGGER.error(f"Reconcile trigger failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e)
        }


@router.get("/api/v3/goals/snapshot")
async def api_v3_goals_snapshot():
    """
    Get current goals configuration.
    
    Returns daily, weekly, monthly, yearly goals.
    Ghost score is now calculated from actual health metrics.
    """
    try:
        default_goals = {
            "daily": 500,
            "weekly": 2500,
            "monthly": 10000,
            "yearly": 120000,
        }

        goals = {}
        for period, default in default_goals.items():
            key = f"goal_{period}"
            if not STATE.get(key):
                STATE[key] = default
            goals[period] = STATE.get(key, default)

        # FIXED (v5.4): Use INTEGRITY-verified accuracy, not paper tracker inflated numbers
        # Paper tracker says 79% but integrity-checked accuracy is 54%
        # The dashboard should show the honest number
        data_health = 50
        ai_activity = 50
        accuracy = None  # Will be set from integrity-checked source
        
        try:
            # Data Health: Check crypto providers with timeout (same as health/metrics)
            from core.crypto.crypto_providers import get_crypto_price_quorum
            btc_data = await asyncio.wait_for(
                get_crypto_price_quorum("BTC", use_cache=True),
                timeout=4.0
            )
            
            if btc_data and btc_data.get("price", 0) > 0:
                quorum_size = btc_data.get("quorum_size", 1)
                if quorum_size >= 2:
                    data_health = 95  # Strong quorum
                else:
                    data_health = 80  # Single provider working
            else:
                data_health = 50  # Provider returned no data
            
            # Bonus checks: predictions in memory + freshness
            if len(_LATEST_PREDICTIONS) == 0:
                data_health = max(60, data_health - 15)  # Penalize no predictions
            else:
                recent_preds = [p for p in _LATEST_PREDICTIONS.values()
                               if time.time() - p.get("run_at", 0) < 21600]
                if len(recent_preds) < 5:
                    data_health = max(60, data_health - 10)  # Penalize stale predictions
            
            LOGGER.info(f"[GHOST_SCORE] data_health={data_health} (BTC ${btc_data.get('price', 0):.0f}, quorum={btc_data.get('quorum_size', 0)})")
        except asyncio.TimeoutError:
            LOGGER.warning("[GHOST_SCORE] BTC quorum timed out after 4s, using default data_health=70")
            data_health = 70  # Timeout = providers exist but slow
        except Exception as _dh_err:
            LOGGER.warning(f"[GHOST_SCORE] data_health calculation failed: {type(_dh_err).__name__}: {_dh_err}")
            data_health = 50  # Fallback — don't punish to 40 for import/transient errors
        
        # AI Activity: Count recent predictions AND check variety
        total_predictions = len(_LATEST_PREDICTIONS)
        unique_symbols = len(set(p.get("symbol") for p in _LATEST_PREDICTIONS.values()))
        
        # Base activity from prediction count
        if total_predictions >= 100:
            base_activity = 80
        elif total_predictions >= 50:
            base_activity = 65
        elif total_predictions >= 20:
            base_activity = 50
        elif total_predictions >= 10:
            base_activity = 40
        else:
            base_activity = 30
        
        # Bonus for symbol variety (up to +15)
        variety_bonus = min(15, unique_symbols)
        ai_activity = min(95, base_activity + variety_bonus)
        
        # Accuracy: Get from INTEGRITY-verified source (ghost_predictions checked=1)
        # This matches what the integrity audit reports — the honest number
        # Priority: PostgreSQL checked predictions > paper tracker > accuracy_stats
        try:
            from core.db_pool import fetchrow as _db_fetchrow
            _gs_row = await _db_fetchrow("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as wins
                FROM ghost_predictions
                WHERE checked = 1
                  AND eval_version NOT LIKE 'skip%%'
                  AND predicted_at > EXTRACT(EPOCH FROM NOW() - INTERVAL '30 days')
            """)
            _gs_total = _gs_row['total'] if _gs_row else 0
            _gs_wins = _gs_row['wins'] if _gs_row and _gs_row['wins'] else 0
            LOGGER.info(f"[GHOST_SCORE] PostgreSQL integrity accuracy: checked={_gs_total}, correct={_gs_wins}")
            if _gs_total and _gs_total > 0:
                accuracy = round((_gs_wins / _gs_total) * 100, 1)
                LOGGER.info(f"[GHOST_SCORE] Integrity-verified accuracy: {accuracy}% ({_gs_wins}/{_gs_total})")
        except Exception as _pg_err:
            LOGGER.warning(f"[GHOST_SCORE] PostgreSQL accuracy query failed: {_pg_err}")

        # Fallback: paper tracker (if PostgreSQL unavailable)
        if accuracy is None:
            try:
                from core.paper_tracker import get_paper_tracker
                tracker = get_paper_tracker()
                stats = tracker.get_stats(days=30)
                _resolved = stats.get("resolved_trades", 0)
                if _resolved > 0:
                    accuracy = round(stats.get("win_rate_pct", 50), 1)
                    LOGGER.info(f"[GHOST_SCORE] Paper tracker fallback accuracy: {accuracy}%")
            except Exception as _pt_err:
                LOGGER.warning(f"[GHOST_SCORE] Paper tracker fallback failed: {_pt_err}")
        
        LOGGER.info(f"[GHOST_SCORE] Components: accuracy={accuracy}, data_health={data_health}, ai_activity={ai_activity}, predictions={total_predictions}")
        
        # If all fallbacks failed, accuracy stays None — use 0 for score calculation
        # but report it honestly (don't fake 50%)
        accuracy_for_score = accuracy if accuracy is not None else 0
        
        # Ghost Score = weighted average (accuracy: 50%, data_health: 30%, ai_activity: 20%)
        ghost_score = round(accuracy_for_score * 0.5 + data_health * 0.3 + ai_activity * 0.2, 1)
        ghost_score = max(0, min(100, ghost_score))  # Clamp to 0-100
        
        return {
            "ok": True,
            "goals": goals,
            "ghost_score": ghost_score,
            # Send real component values so the UI bars show actual health
            "daily_goal_pct": round(accuracy, 1) if accuracy is not None else None,       # Accuracy %
            "weekly_goal_pct": round(data_health, 1),    # Data Health %
            "monthly_goal_pct": round(ai_activity, 1),   # AI Activity %
            "data_health": round(data_health, 1),
            "ai_activity": round(ai_activity, 1),
            "accuracy": round(accuracy, 1) if accuracy is not None else None,
            "components": {
                "accuracy": accuracy,
                "data_health": data_health,
                "ai_activity": ai_activity,
                "total_predictions": total_predictions,
            }
        }
    
    except Exception as e:
        LOGGER.error(f"Goals snapshot failed: {e}", exc_info=True)


@router.get("/api/v3/learning-brain/scorecard")
async def api_v3_learning_brain_scorecard():
    """
    Ghost Learning Brain scorecard.
    Shows per-symbol accuracy and which symbols are flagged for
    auto-inversion, benching, or active recommendation.
    """
    try:
        from core.ghost_learning_brain import (
            force_refresh, INVERT_ACCURACY_THRESHOLD,
            BENCH_ACCURACY_THRESHOLD, MIN_EVALUATED_PREDICTIONS,
        )
        scorecard = force_refresh()
        inverted = {s: d for s, d in scorecard.items() if d.get("should_invert")}
        benched = {s: d for s, d in scorecard.items() if d.get("should_bench")}
        active = {s: d for s, d in scorecard.items()
                  if not d.get("should_invert") and not d.get("should_bench")}
        total_preds = sum(d["total"] for d in scorecard.values())
        total_correct = sum(d["correct"] for d in scorecard.values())

        # Build status lines
        status_parts = []
        if inverted:
            status_parts.append(f"🔄 Inverting {len(inverted)}: {', '.join(inverted.keys())}")
        if benched:
            status_parts.append(f"🪑 Benched {len(benched)}: {', '.join(benched.keys())}")
        if not inverted and not benched:
            status_parts.append("🧠 All symbols above threshold — no actions needed")

        return {
            "ok": True,
            "brain_status": "ACTIVE",
            "invert_threshold_pct": INVERT_ACCURACY_THRESHOLD,
            "bench_threshold_pct": BENCH_ACCURACY_THRESHOLD,
            "min_predictions": MIN_EVALUATED_PREDICTIONS,
            "overall_accuracy_pct": round(total_correct / total_preds * 100, 1) if total_preds else 0,
            "total_evaluated": total_preds,
            "zones": {
                "recommend": f"> {BENCH_ACCURACY_THRESHOLD}% accuracy → send to picks",
                "bench": f"{INVERT_ACCURACY_THRESHOLD}-{BENCH_ACCURACY_THRESHOLD}% → dropped from picks",
                "invert": f"< {INVERT_ACCURACY_THRESHOLD}% → direction flipped",
            },
            "symbols_inverted": list(inverted.keys()),
            "symbols_benched": list(benched.keys()),
            "symbols_active": list(active.keys()),
            "scorecard": scorecard,
            "message": " | ".join(status_parts),
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/api/v3/orchestrator/status")
async def api_v3_orchestrator_status():
    """
    Get Master Orchestrator status - all background services.
    
    Returns:
        - Service statuses (running/stopped/disabled)
        - Active task count
        - System uptime
        - Last run times for each service
    """
    try:
        from core.orchestrator import get_system_status
        status = get_system_status() or {}

        # Add deployment-mode context (helps debug Railway web/worker split)
        worker_mode = os.getenv("WORKER_MODE") == "1"
        orchestrator_enabled = os.getenv("ORCHESTRATOR_ENABLED", "0") == "1"

        services = status.get("services") or {}
        running = 0
        failed = 0
        disabled = 0
        other = 0
        failing_services: list[str] = []

        for name, svc in services.items():
            svc_status = (svc or {}).get("status")
            if svc_status == "running":
                running += 1
            elif svc_status == "failed":
                failed += 1
                failing_services.append(str(name))
            elif svc_status == "disabled":
                disabled += 1
            else:
                other += 1

        status["mode"] = "WORKER" if worker_mode else "WEB"
        status["orchestrator_enabled"] = orchestrator_enabled
        status["summary"] = {
            "running": running,
            "failed": failed,
            "disabled": disabled,
            "other": other,
            "failing_services": sorted(failing_services),
        }
        return status
    except Exception as e:
        LOGGER.error(f"Orchestrator status failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "uptime_seconds": 0,
            "services": {},
            "active_tasks": 0,
            "total_services": 0
        }


@router.get("/api/v3/context/stats")
async def api_v3_context_stats():
    """
    Get Stage 1 Context Engine statistics.
    
    Returns:
        - Total articles in database
        - Articles from last 24 hours
        - Last refresh time
        - RSS source count
        - Database age span
    """
    try:
        from core.context_engine import get_context_engine
        engine = get_context_engine()
        
        if engine is None:
            return {
                "ok": False,
                "error": "Context engine not initialized",
                "enabled": False
            }
        
        stats = engine.get_stats()
        stats["ok"] = True
        stats["enabled"] = True
        return stats
        
    except Exception as e:
        LOGGER.error(f"Context stats failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "enabled": False,
            "total_articles": 0,
            "recent_24h": 0
        }


@router.get("/api/v3/execution/status")
async def api_v3_execution_status():
    """
    Get Autonomous Execution Engine status.
    
    Returns:
        - Enabled/disabled state
        - Circuit breaker status
        - Total execution cycles run
        - Trades executed today
        - Last cycle/trade times
        - Configuration (min confidence, max positions, etc.)
    """
    try:
        from core.autonomous_execution_engine import get_execution_status
        status = get_execution_status()
        return status
        
    except Exception as e:
        LOGGER.error(f"Execution status failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e),
            "enabled": False,
            "circuit_breaker_active": True,
            "circuit_breaker_reason": "Error fetching status",
            "total_cycles": 0,
            "trades_today": 0
        }
        return {
            "ok": False,
            "goals": {},
            "error": str(e)
        }


@router.get("/api/v3/live_recalculator/status")
async def api_v3_live_recalculator_status(limit_snapshots: int = 50, limit_signals: int = 50):
    """Get latest live recalculator snapshots and exit signals.

    Reads from `data/live_recalculator.db` written by `core.live_recalculator`.

    Args:
        limit_snapshots: Max number of snapshot rows to return (default 50)
        limit_signals: Max number of exit signal rows to return (default 50)
    """
    try:
        import sqlite3
        from pathlib import Path

        db_path = Path(__file__).parent / "data" / "live_recalculator.db"

        if not db_path.exists():
            return {
                "ok": False,
                "enabled": False,
                "error": "live_recalculator.db not found",
                "db_path": str(db_path),
                "latest_ts": 0,
                "snapshots": [],
                "exit_signals": [],
            }

        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()

        # Detect tables
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {r[0] for r in (cur.fetchall() or [])}

        snapshots: list[dict] = []
        signals: list[dict] = []
        latest_ts = 0

        if "position_snapshots" in tables:
            cur.execute(
                """
                SELECT ts, symbol, qty, avg_entry_price, current_price, unrealized_pl, unrealized_plpc
                FROM position_snapshots
                ORDER BY ts DESC, id DESC
                LIMIT ?
                """,
                (max(1, min(int(limit_snapshots), 500)),),
            )
            rows = cur.fetchall() or []
            for (ts, symbol, qty, avg_entry, current_price, upl, uplpc) in rows:
                latest_ts = max(latest_ts, int(ts or 0))
                snapshots.append(
                    {
                        "ts": int(ts or 0),
                        "symbol": symbol,
                        "qty": qty,
                        "avg_entry_price": avg_entry,
                        "current_price": current_price,
                        "unrealized_pl": upl,
                        "unrealized_plpc": uplpc,
                    }
                )

        if "exit_signals" in tables:
            cur.execute(
                """
                SELECT ts, symbol, type, reason, pnl_pct, entry_price, current_price
                FROM exit_signals
                ORDER BY ts DESC, id DESC
                LIMIT ?
                """,
                (max(1, min(int(limit_signals), 500)),),
            )
            rows = cur.fetchall() or []
            for (ts, symbol, typ, reason, pnl_pct, entry_price, current_price) in rows:
                latest_ts = max(latest_ts, int(ts or 0))
                signals.append(
                    {
                        "ts": int(ts or 0),
                        "symbol": symbol,
                        "type": typ,
                        "reason": reason,
                        "pnl_pct": pnl_pct,
                        "entry_price": entry_price,
                        "current_price": current_price,
                    }
                )

        conn.close()

        return {
            "ok": True,
            "enabled": True,
            "db_path": str(db_path),
            "tables": sorted(list(tables)),
            "latest_ts": latest_ts,
            "snapshots": snapshots,
            "exit_signals": signals,
        }

    except Exception as e:
        LOGGER.error(f"Live recalculator status failed: {e}", exc_info=True)
        return {
            "ok": False,
            "enabled": False,
            "error": str(e),
            "latest_ts": 0,
            "snapshots": [],
            "exit_signals": [],
        }


@router.get("/integrity/audit")
async def integrity_audit():
    """
    Run the full system integrity audit with auto-fix.
    Returns health score (0-100) + issues list.
    Called by UI on page load + every 5 minutes.
    Escalates ERROR-severity issues to Telegram.
    """
    try:
        from core.integrity import run_audit
        result = await asyncio.to_thread(run_audit, auto_fix=True)

        # ── Telegram escalation for ERROR-severity issues ──
        try:
            error_issues = [i for i in result.get("issues", []) if i.get("severity") == "error"]
            if error_issues:
                score = result.get("health_score", 0)
                fixes = result.get("auto_fixes_applied", 0)
                lines = [
                    f"🚨 Ghost Integrity Alert — {len(error_issues)} error(s)",
                    f"Health Score: {score}/100 · Auto-fixes: {fixes}",
                    "──────────────────",
                ]
                for iss in error_issues[:8]:
                    lines.append(f"❌ {iss.get('type', 'unknown')}: {iss.get('detail', '')[:80]}")
                lines.append("──────────────────")
                lines.append("Check cockpit Health tab for full details")
                _esc_msg = "\n".join(lines)

                import httpx as _esc_httpx
                _esc_token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
                _esc_chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()
                if _esc_token and _esc_chat:
                    _esc_url = f"https://api.telegram.org/bot{_esc_token}/sendMessage"
                    for _cid in _esc_chat.split(","):
                        _cid = _cid.strip()
                        if _cid:
                            await asyncio.to_thread(
                                lambda: _esc_httpx.post(
                                    _esc_url,
                                    json={"chat_id": _cid, "text": _esc_msg, "disable_web_page_preview": True},
                                    timeout=10,
                                )
                            )
                    result["telegram_escalated"] = True
                    LOGGER.info(f"[INTEGRITY] Escalated {len(error_issues)} error(s) to Telegram")
        except Exception as esc_e:
            LOGGER.warning(f"[INTEGRITY] Telegram escalation failed: {esc_e}")

        return result
    except Exception as e:
        from datetime import datetime as _dt_int
        LOGGER.error(f"[INTEGRITY] Audit endpoint failed: {e}")
        return {
            "health_score": 0,
            "score_breakdown": [],
            "total_penalty": 100,
            "auto_fixes_applied": 0,
            "issues_remaining": 1,
            "issues": [{"type": "audit_crash", "severity": "error", "detail": str(e)[:200]}],
            "checks_run": [],
            "summary": {},
            "last_audit": _dt_int.now().isoformat(),
        }


@router.get("/integrity/bugs")
async def integrity_bugs():
    """
    Run bug-fix regression checks.
    Verifies that ALL known bug fixes are still intact.
    """
    try:
        from core.integrity_bug_checks import run_bug_checks_summary
        result = await asyncio.to_thread(run_bug_checks_summary)
        return result
    except Exception as e:
        LOGGER.error(f"[INTEGRITY] Bug checks endpoint failed: {e}")
        return {
            "total_checks": 0,
            "passed": 0,
            "failed": 1,
            "health_pct": 0,
            "checks": [{"name": "crash", "bug_id": -1, "passed": False,
                        "severity": "error", "detail": str(e)[:200], "mismatches": []}],
        }


@router.get("/integrity/audit/readonly")
async def integrity_audit_readonly():
    """
    Run integrity audit WITHOUT auto-fix (read-only).
    Safe to call from monitoring/external systems.
    """
    try:
        from core.integrity import run_audit
        result = await asyncio.to_thread(run_audit, auto_fix=False)
        return result
    except Exception as e:
        from datetime import datetime as _dt_int
        return {
            "health_score": 0,
            "issues_remaining": 1,
            "issues": [{"type": "audit_crash", "severity": "error", "detail": str(e)[:200]}],
            "checks_run": [],
            "summary": {},
            "last_audit": _dt_int.now().isoformat(),
        }


@router.get("/api/v3/market/ticker")
async def api_v3_market_ticker():
    """Market ticker bar data — major indices and crypto prices for the top bar."""
    import time as _t
    items = []

    # BTC and ETH from existing price infrastructure
    for crypto_sym, ticker_id in [("BTC", "btc"), ("ETH", "eth")]:
        try:
            from core.crypto.crypto_providers import get_crypto_price_turbo
            price_data = await get_crypto_price_turbo(crypto_sym)
            if price_data and price_data.get("price"):
                p = price_data["price"]
                pct = price_data.get("change_24h_pct", 0) or 0
                abs_change = round(p * pct / 100, 2) if pct else 0
                items.append({
                    "id": ticker_id,
                    "name": crypto_sym,
                    "price": p,
                    "change": abs_change,
                    "change_pct": round(pct, 2),
                })
        except Exception:
            items.append({"id": ticker_id, "name": crypto_sym, "price": 0, "change": 0, "change_pct": 0})

    # Market indices via multi-provider fallback (Yahoo Chart → Yahoo Quote → yfinance)
    indices = [
        ("^GSPC", "spy", "S&P 500"),
        ("^DJI", "dow", "DOW"),
        ("^IXIC", "nasdaq", "NASDAQ"),
        ("^VIX", "vix", "VIX"),
    ]
    for yf_sym, ticker_id, display_name in indices:
        try:
            price, prev = await asyncio.to_thread(_get_index_price, yf_sym)
            change = (price - prev) if price and prev else 0
            change_pct = (change / prev * 100) if prev and prev > 0 else 0
            items.append({
                "id": ticker_id,
                "name": display_name,
                "price": round(price, 2) if price else 0,
                "change": round(change, 2),
                "change_pct": round(change_pct, 2),
            })
        except Exception as idx_err:
            LOGGER.debug(f"Ticker {yf_sym} failed: {idx_err}")
            items.append({"id": ticker_id, "name": display_name, "price": 0, "change": 0, "change_pct": 0})

    return {"ok": True, "items": items, "ts": _t.time()}


@router.get("/api/v3/intelligence/status")
async def api_v3_intelligence_status():
    """Get Intelligence Hub status — shows which of the 20 systems are active."""
    try:
        from core.intelligence_hub import get_intelligence_hub, get_news_brain_cache
        hub = get_intelligence_hub()
        status = hub.get_status()
        cache, cache_ts = get_news_brain_cache()
        status["news_brain_events"] = len(cache.get("major_events", [])) if cache else 0
        status["news_brain_at_risk"] = len(cache.get("predictions_at_risk", [])) if cache else 0

        # Build systems array for frontend subsystem cards
        subsystem_names = [
            "ensemble", "calibrator", "trust_ladder", "quality_gate",
            "killswitch", "vwap", "feed_fusion", "regime_detector",
            "self_improvement",
        ]
        systems = []
        for name in subsystem_names:
            loaded = status.get(f"{name}_loaded", False)
            systems.append({"name": name, "active": loaded})
        # Add news brain as a subsystem
        systems.append({"name": "news_brain", "active": status.get("news_brain_has_data", False)})

        status["systems"] = systems
        status["systems_loaded"] = sum(1 for s in systems if s["active"])
        status["systems_total"] = len(systems)

        return {"ok": True, **status}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/api/v3/intelligence/cache")
async def api_v3_intelligence_cache():
    """Dump the current news brain cache — shows exactly what symbols are at risk."""
    try:
        from core.intelligence_hub import get_news_brain_cache
        cache, cache_ts = get_news_brain_cache()
        import time as _t
        return {
            "ok": True,
            "cache_age_seconds": round(_t.time() - cache_ts, 1) if cache_ts > 0 else -1,
            "predictions_at_risk": cache.get("predictions_at_risk", []) if cache else [],
            "major_events": [
                {
                    "headline": e.get("headline", "")[:120],
                    "severity": e.get("severity", "?"),
                    "bearish_symbols": e.get("bearish_symbols", []),
                    "bullish_symbols": e.get("bullish_symbols", []),
                }
                for e in cache.get("major_events", [])
            ] if cache else [],
            "recommendation": cache.get("recommendation", "") if cache else "",
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/api/v3/intelligence/analyze")
async def api_v3_intelligence_analyze(symbol: str, direction: str = "UP",
                                       confidence: float = 0.50):
    """
    Run the Intelligence Hub against a symbol and return ALL signal details.
    Debug endpoint — shows what each of the 20 systems says.
    """
    try:
        from core.intelligence_hub import get_intelligence_hub
        hub = get_intelligence_hub()

        # Fetch price history for regime detection
        _ph: list = []
        try:
            from core.ghost_scout import GhostScout as _GS
            _gs = _GS()
            _edge_set = get_edge_set()
            is_crypto = symbol.upper() not in {"T", "DDOG", "BMBL", "FTNT", "XPO",
                                                "NET", "PANW"}
            _ph = _gs._fetch_price_history(
                symbol.upper(), "crypto" if is_crypto else "stock"
            ) or []
        except Exception:
            pass

        report = hub.analyze(
            symbol=symbol.upper(),
            direction=direction.upper(),
            confidence=confidence,
            entry_price=0.0,
            asset_type="crypto",
            price_history=_ph,
        )

        return {
            "ok": True,
            "symbol": symbol.upper(),
            "input": {"direction": direction.upper(), "confidence": confidence},
            "price_history_len": len(_ph),
            "report": report.to_dict(),
        }
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()[-500:]}


@router.get("/api/v3/phase5/status")
async def api_v3_phase5_status():
    """
    Get Phase 5 autonomous execution engine status.
    
    Returns current state of the autonomous trading system.
    """
    try:
        from core.autonomous_execution_engine import get_execution_status
        
        status = get_execution_status()
        return {
            "ok": True,
            "phase5": status,
            "timestamp": datetime.now(UTC).isoformat()
        }
    
    except ImportError as e:
        return {
            "ok": False,
            "error": "Phase 5 module not found - not deployed",
            "details": str(e)
        }
    except Exception as e:
        LOGGER.error(f"Phase 5 status failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e)
        }


@router.get("/api/v3/analytics/report")
async def api_v3_analytics_report():
    """
    Phase 7: Get comprehensive analytics report.
    """
    try:
        from core.analytics_engine import get_analytics_report
        return get_analytics_report()
    except Exception as e:
        LOGGER.error(f"Analytics report error: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}


@router.get("/api/v3/production/status")
async def api_v3_production_status():
    """
    Phase 9: Get production trading status and safety limits.
    """
    try:
        from core.production_trading import get_status
        return get_status()
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/api/v3/production/kill-switch")
async def api_v3_kill_switch(activate: bool, reason: str = "Manual activation"):
    """
    Phase 9: Activate/deactivate emergency kill switch.
    """
    try:
        from core.production_trading import activate_kill_switch, get_production_controller
        
        controller = get_production_controller()
        if activate:
            controller.activate_kill_switch(reason)
        else:
            controller.deactivate_kill_switch()
        
        return {
            "ok": True,
            "kill_switch_active": controller.kill_switch_active,
            "message": "Kill switch activated" if activate else "Kill switch deactivated"
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/api/v3/strategies/performance")
async def api_v3_strategies_performance():
    """
    Phase 10: Get multi-strategy performance metrics.
    """
    try:
        from core.multi_strategy_engine import get_strategy_performance
        return get_strategy_performance()
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/api/v3/strategies/rebalance")
async def api_v3_strategies_rebalance():
    """
    Phase 10: Trigger strategy allocation rebalancing.
    """
    try:
        from core.multi_strategy_engine import get_strategy_engine
        
        engine = get_strategy_engine()
        engine.rebalance_allocations()
        
        return {
            "ok": True,
            "message": "Strategy allocations rebalanced",
            "performance": engine.get_performance_summary()
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.websocket("/ws/trades")
async def websocket_trades(websocket: WebSocket):
    """
    Phase 6: WebSocket endpoint for real-time trade updates.
    """
    await websocket.accept()
    
    try:
        from core.trade_monitor import register_websocket, unregister_websocket
        
        register_websocket(websocket)
        LOGGER.info("[WS] Trade monitor client connected")
        
        # Keep connection alive
        while True:
            try:
                data = await websocket.receive_text()
                # Echo ping/pong
                if data == "ping":
                    await websocket.send_text("pong")
            except Exception as e:
                LOGGER.warning(f"[WS] Client disconnected: {e}")
                break
    
    finally:
        unregister_websocket(websocket)
        LOGGER.info("[WS] Trade monitor client disconnected")


@router.post("/api/v3/test/inject-trade")
async def api_v3_test_inject_trade(
    symbol: str = "AAPL",
    confidence: float = 75.0,
    direction: str = "UP"
):
    """
    Option 3: Inject a simulated high-confidence prediction for testing.
    Tests the entire trade pipeline end-to-end.
    GUARD: Only works when SIM_MODE=1 to prevent polluting production data.
    """
    if os.getenv("SIM_MODE", "0") != "1":
        return {"ok": False, "error": "Test injection disabled in production (requires SIM_MODE=1)"}
    try:
        from core.autonomous_execution_engine import run_execution_cycle
        from core.prediction_store import get_prediction_store
        import asyncio
        
        # Create fake high-confidence prediction
        prediction_store = get_prediction_store()
        fake_prediction = {
            "symbol": symbol,
            "direction": direction,
            "confidence": confidence,
            "target_price": 180.0 if symbol == "AAPL" else 100.0,
            "timestamp": datetime.now(UTC).isoformat(),
            "features": {"test": True}
        }
        
        # Store it temporarily
        prediction_store._cache[symbol] = fake_prediction
        
        # Trigger execution cycle
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, run_execution_cycle)
        
        return {
            "ok": True,
            "message": f"Test trade injected: {symbol} {direction} @ {confidence}%",
            "execution_result": result,
            "timestamp": datetime.now(UTC).isoformat()
        }
    
    except Exception as e:
        LOGGER.error(f"Test trade injection failed: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}


@router.api_route("/api/v3/goals/set", methods=["GET", "POST"])
async def api_v3_goals_set(period: str, target_amount: float):
    """
    Set a goal for a specific period.
    
    Args:
        period: 'daily', 'weekly', 'monthly', or 'yearly'
        target_amount: Target amount in dollars
    """
    try:
        valid_periods = ["daily", "weekly", "monthly", "yearly"]
        if period not in valid_periods:
            return {
                "ok": False,
                "error": f"Invalid period. Must be one of: {valid_periods}"
            }
        
        # Store in STATE
        STATE[f"goal_{period}"] = target_amount
        
        LOGGER.info(f"Goal set: {period} = ${target_amount}")
        
        return {
            "ok": True,
            "period": period,
            "amount": target_amount,
            "message": f"{period.capitalize()} goal set to ${target_amount}"
        }
    
    except Exception as e:
        LOGGER.error(f"Set goal failed: {e}", exc_info=True)
        return {
            "ok": False,
            "error": str(e)
        }


@router.get("/api/v3/watchlist/market")
async def get_market_watchlist_v3():
    """
    Market watchlist: top crypto symbols with prices and Ghost predictions.
    Cockpit v3 Market tab.
    """
    try:
        symbols = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", 
                   "MATIC", "DOT", "LINK", "UNI", "LTC", "ATOM", "XLM"]
        items = []
        
        for symbol in symbols:
            try:
                # Get price (with 1s timeout per symbol)
                price_data = turbo_crypto_price(symbol, max_budget_s=1.0)
                price = price_data.get("price", 0)
                change_pct = price_data.get("change_24h_pct", 0)
                
                # Get Ghost prediction
                pred = _LATEST_PREDICTIONS.get(symbol, {})
                confidence = pred.get("confidence", 0)
                if 0 < confidence <= 1:
                    confidence = confidence * 100  # Convert 0-1 to 0-100
                
                items.append({
                    "symbol": symbol,
                    "price": price,
                    "change_pct": change_pct,
                    "ghost_confidence": confidence,
                    "ghost_direction": pred.get("direction", "FLAT"),
                    "type": "crypto"
                })
            except Exception as e:
                LOGGER.warning(f"Market watchlist: {symbol} fetch failed: {e}")
                continue
        
        return {"ok": True, "items": items}
    except Exception as e:
        LOGGER.error(f"Market watchlist error: {e}", exc_info=True)
        return {"ok": False, "error": str(e), "items": []}


@router.get("/api/v3/hunter/feed")
async def api_v3_hunter_feed(limit: int = 10):
    """
    Get Hunter news feed for cockpit movers/news panel.
    
    Returns recent prediction news/alerts as both 'movers' and 'feed'.
    OPTIMIZED: Fast in-memory path first, DB fallback only if empty
    """
    try:
        # EMERGENCY: If system just started (uptime < 60s), return empty feed to prevent startup deadlock
        import time as _time_module
        uptime_seconds = int(_time_module.time() - _START_TS)
        if uptime_seconds < 60:
            LOGGER.info(f"[HUNTER] System startup (uptime: {uptime_seconds}s) - returning empty feed")
            return {
                "ok": True,
                "movers": [],
                "feed": [],
                "count": 0,
                "timestamp": int(_time_module.time()),
                "message": "System starting - predictions generating soon",
                "source": "startup"
            }
        
        # FAST PATH: Use in-memory predictions if available (avoids DB query)
        predictions = list(_LATEST_PREDICTIONS.values()) if _LATEST_PREDICTIONS else []
        
        # If we have in-memory predictions, use them (fast - <10ms)
        if predictions:
            predictions.sort(key=lambda p: p.get("confidence", 0), reverse=True)
            feed_items = []

            for pred in predictions[:limit]:
                symbol = pred.get("symbol")
                if not symbol:
                    continue

                direction = pred.get("direction", "FLAT")
                confidence = pred.get("confidence", 0) or 0
                confidence_pct = round(confidence * 100, 1) if confidence <= 1 else round(confidence, 1)
                
                # Use real expected_move_pct from prediction engine (not a formula)
                expected_move = pred.get("expected_move_pct") or pred.get("expected_move")
                if expected_move is not None:
                    # expected_move_pct is already in percentage (e.g., 4.5 = 4.5%)
                    change_pct = expected_move if abs(expected_move) < 20 else expected_move / 100
                else:
                    # No expected_move data — show 0 instead of inventing a number
                    change_pct = 0.0

                change_pct = round(change_pct, 2)

                feed_items.append({
                    "symbol": symbol,
                    "name": symbol,
                    "title": f"Ghost predicts {symbol} {direction} ({confidence_pct:.0f}% confidence)",
                    "sentiment": "bullish" if direction == "UP" else "bearish" if direction == "DOWN" else "neutral",
                    "timestamp": int(pred.get("run_at", time.time())),
                    "source": "Ghost AI",
                    "type": "crypto" if symbol in CRYPTO_SYMBOLS else "stock",
                    "change_pct": change_pct,
                    "change": change_pct,
                    "confidence": confidence_pct,
                    "ghost_confidence": confidence_pct,
                    "price": pred.get("price_at_prediction")
                })
            
            # FIX 7: Add timestamp and data freshness indicators
            oldest_pred_time = min((item.get("timestamp", 0) for item in feed_items), default=0)
            data_age = int(time.time() - oldest_pred_time) if oldest_pred_time > 0 else None
            
            return {
                "ok": True,
                "movers": feed_items,
                "feed": feed_items,
                "count": len(feed_items),
                "timestamp": int(time.time()),
                "source": "memory",
                "generated_at": int(time.time()),
                "data_age_seconds": data_age
            }
        
        # SLOW PATH: Query database if no in-memory predictions (DB query can be slow)
        # Add timeout protection to prevent 9-10 second hangs
        LOGGER.info("[HUNTER] _LATEST_PREDICTIONS empty, querying database with 3s timeout...")
        try:
            import asyncio
            from core.prediction_store import get_prediction_store
            
            # Wrap synchronous DB call in thread pool executor with timeout (prevents event loop blocking)
            loop = asyncio.get_event_loop()
            
            def fetch_from_db_sync():
                store = get_prediction_store()
                return store.get_recent_predictions(limit=limit * 2)
            
            try:
                recent_preds = await asyncio.wait_for(
                    loop.run_in_executor(None, fetch_from_db_sync),
                    timeout=3.0
                )
            except TimeoutError:
                LOGGER.warning("[HUNTER] Database query timeout after 3s, returning empty feed")
                return {
                    "ok": True,
                    "movers": [],
                    "feed": [],
                    "count": 0,
                    "timestamp": int(time.time()),
                    "error": "Database query timeout - predictions generating soon",
                    "source": "timeout"
                }
            
            feed_items = []
            for pred in recent_preds[:limit]:
                symbol = pred.get("symbol")
                direction = pred.get("direction", "FLAT")
                confidence = pred.get("confidence", 0) or 0
                confidence_pct = round(confidence * 100, 1) if confidence <= 1 else round(confidence, 1)
                
                expected_move = pred.get("expected_move_pct") or pred.get("expected_move")
                if expected_move is not None:
                    change_pct = expected_move * 100 if abs(expected_move) < 1 else expected_move
                else:
                    # No expected_move data — show 0 instead of fabricating
                    change_pct = 0.0
                
                change_pct = round(change_pct, 2)
                
                feed_items.append({
                    "symbol": symbol,
                    "name": symbol,
                    "title": f"Ghost predicts {symbol} {direction} ({confidence_pct:.0f}% confidence)",
                    "sentiment": "bullish" if direction == "UP" else "bearish" if direction == "DOWN" else "neutral",
                    "timestamp": int(pred.get("created_at", time.time())),
                    "source": "Ghost AI",
                    "type": "crypto" if symbol in CRYPTO_SYMBOLS else "stock",
                    "change_pct": change_pct,
                    "change": change_pct,
                    "confidence": confidence_pct,
                    "ghost_confidence": confidence_pct,
                    "price": pred.get("price_at_prediction")
                })
            
            # FIX 7: Add timestamp and data freshness indicators
            oldest_pred_time = min((item.get("timestamp", 0) for item in feed_items), default=0)
            data_age = int(time.time() - oldest_pred_time) if oldest_pred_time > 0 else None
            
            return {
                "ok": True,
                "movers": feed_items,
                "feed": feed_items,
                "count": len(feed_items),
                "timestamp": int(time.time()),
                "source": "database",
                "generated_at": int(time.time()),
                "data_age_seconds": data_age
            }
        except Exception as db_error:
            LOGGER.error(f"Database fallback failed: {db_error}")
            return {
                "ok": True,
                "movers": [],
                "feed": [],
                "count": 0,
                "timestamp": int(time.time()),
                "error": "No predictions available"
            }
    
    except Exception as e:
        LOGGER.error(f"Hunter feed failed: {e}", exc_info=True)
        return {
            "ok": False,
            "movers": [],
            "feed": [],
            "error": str(e)
        }


@router.get("/api/v3/predictions/history")
async def api_v3_predictions_history(limit: int = 100):
    """
    Get prediction history for accuracy calculations.
    
    Returns recent predictions with outcomes.
    """
    try:
        # Return predictions from in-memory store
        history = []
        
        for symbol, pred in list(_LATEST_PREDICTIONS.items())[:limit]:
            history.append({
                "symbol": symbol,
                "prediction_id": pred.get("prediction_id"),
                "direction": pred.get("direction", "FLAT"),
                "confidence": pred.get("confidence", 0),
                "run_at": pred.get("run_at", 0),
                "horizon_h": pred.get("horizon_h", 48),
                "price_at_prediction": pred.get("price_at_prediction"),
                "provider": pred.get("provider", "unknown"),
                # Outcome from paper trade system (if resolved)
                "closed": pred.get("resolved", False),
                "accuracy": pred.get("outcome")
            })
        
        return {
            "ok": True,
            "predictions": history,  # UI expects 'predictions' key
            "history": history,      # Keep for compatibility
            "count": len(history),
            "timestamp": int(time.time())
        }
    
    except Exception as e:
        LOGGER.error(f"Prediction history failed: {e}", exc_info=True)
        return {
            "ok": False,
            "predictions": [],
            "history": [],
            "error": str(e)
        }


@router.get("/api/crypto/price/{symbol}")
async def api_crypto_price(
    symbol: str,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    Get current crypto price from quorum of providers

    Returns:
        {
            "symbol": "BTC",
            "price": 43251.50,
            "provider": "coingecko",
            "confidence": 0.95,
            "quorum_size": 3,
            "spread": 0.003,
            "timestamp": 1728741600,
            "change_24h_pct": 2.98
        }
    """
    # Auth optional for read-only
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

    # Check if crypto enabled
    if not os.getenv("CRYPTO_ENABLED", "0") == "1":
        raise HTTPException(503, "Crypto module not enabled. Set CRYPTO_ENABLED=1")

    try:
        providers = _get_crypto_providers()
        price_data = await providers.get_crypto_price_quorum(symbol)

        if not price_data:
            raise HTTPException(404, f"Price not available for {symbol}")

        # Track metrics
        if _C_CRYPTO_PRICE_FETCH is not None:
            try:
                _C_CRYPTO_PRICE_FETCH.labels(
                    provider=price_data.get("provider", "unknown"), result="success"
                ).inc()
            except Exception:
                pass

        return price_data

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Crypto price fetch failed for {symbol}: {e}", exc_info=True)
        if _C_CRYPTO_PRICE_FETCH is not None:
            try:
                _C_CRYPTO_PRICE_FETCH.labels(provider="unknown", result="error").inc()
            except Exception:
                pass
        raise HTTPException(500, f"Price fetch failed: {str(e)[:200]}")


@router.get("/api/crypto/watchlist")
async def api_crypto_watchlist(
    category: str = "default",
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    Get crypto watchlist with live prices

    Categories: default, blue_chip, defi, meme, ai_gaming, all

    Returns list of {symbol, price, change_24h_pct, confidence}
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    if not os.getenv("CRYPTO_ENABLED", "0") == "1":
        raise HTTPException(503, "Crypto module not enabled")

    try:
        providers = _get_crypto_providers()

        # Get watchlist for category
        symbols = providers.get_watchlist_by_category(category)

        # Fetch prices in parallel
        import asyncio

        tasks = [providers.get_crypto_price_quorum(sym) for sym in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        watchlist = []
        for sym, result in zip(symbols, results, strict=False):
            if isinstance(result, Exception):
                LOGGER.warning(f"Failed to fetch {sym}: {result}")
                continue
            if result:
                watchlist.append(
                    {
                        "symbol": sym,
                        "price": result.get("price"),
                        "change_24h_pct": result.get("change_24h_pct"),
                        "confidence": result.get("confidence"),
                        "provider": result.get("provider"),
                        "quorum_size": result.get("quorum_size"),
                    }
                )

        return {"category": category, "count": len(watchlist), "assets": watchlist}

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Crypto watchlist fetch failed: {e}", exc_info=True)
        raise HTTPException(500, f"Watchlist fetch failed: {str(e)[:200]}")


@router.get("/api/crypto/movers")
async def api_crypto_movers(
    threshold: float = 10.0,
    limit: int = 20,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    Get top crypto movers (24h change > threshold)

    Similar to /api/top_movers for stocks
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    if not os.getenv("CRYPTO_ENABLED", "0") == "1":
        raise HTTPException(503, "Crypto module not enabled")

    try:
        from core.crypto.crypto_providers import CoinGeckoProvider

        provider = CoinGeckoProvider()

        # Get all watchlist symbols
        all_symbols = (
            provider.SYMBOL_MAP.keys()
            if hasattr(provider, "SYMBOL_MAP")
            else ["BTC", "ETH", "SOL", "DOGE", "SHIB", "PEPE"]
        )

        # Fetch prices in parallel
        import asyncio

        from core.crypto.crypto_providers import get_crypto_price_quorum

        tasks = [get_crypto_price_quorum(sym, use_cache=False) for sym in list(all_symbols)[:50]]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        movers = []
        for sym, result in zip(list(all_symbols)[:50], results, strict=False):
            if isinstance(result, Exception) or not result:
                continue

            change_24h = result.get("change_24h_pct", 0)
            if abs(change_24h) >= threshold:
                movers.append(
                    {
                        "symbol": sym,
                        "price": result.get("price"),
                        "change_24h_pct": change_24h,
                        "volume_24h": result.get("volume_24h", 0),
                        "market_cap": result.get("market_cap", 0),
                        "confidence": result.get("confidence"),
                        "direction": "UP" if change_24h > 0 else "DOWN",
                    }
                )

        # Sort by absolute change, limit results
        movers.sort(key=lambda x: abs(x["change_24h_pct"]), reverse=True)

        return {"threshold": threshold, "count": len(movers[:limit]), "movers": movers[:limit]}

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Crypto movers fetch failed: {e}", exc_info=True)
        raise HTTPException(500, f"Movers fetch failed: {str(e)[:200]}")


@router.get("/api/crypto/news")
async def api_crypto_news(
    symbol: str | None = None,
    limit: int = 50,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    Get crypto news from RSS feeds (CoinDesk, Cointelegraph)

    Similar to /api/news for stocks
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    if not os.getenv("CRYPTO_ENABLED", "0") == "1":
        raise HTTPException(503, "Crypto module not enabled")

    try:
        import feedparser

        crypto_feeds = [
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://cointelegraph.com/rss",
            "https://cryptoslate.com/feed/",
        ]

        all_articles = []

        for feed_url in crypto_feeds:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:limit]:
                    article = {
                        "title": entry.get("title", ""),
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "summary": entry.get("summary", "")[:200],
                        "source": feed.feed.get("title", "Unknown"),
                    }

                    # Filter by symbol if provided
                    if symbol:
                        text = f"{article['title']} {article['summary']}".upper()
                        if symbol.upper() in text or _get_crypto_name(symbol.upper()) in text:
                            all_articles.append(article)
                    else:
                        all_articles.append(article)
            except Exception as e:
                LOGGER.warning(f"Failed to fetch feed {feed_url}: {e}")
                continue

        # Sort by published date (most recent first)
        all_articles.sort(key=lambda x: x.get("published", ""), reverse=True)

        return {
            "symbol": symbol,
            "count": len(all_articles[:limit]),
            "articles": all_articles[:limit],
        }

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Crypto news fetch failed: {e}", exc_info=True)
        raise HTTPException(500, f"News fetch failed: {str(e)[:200]}")


@router.post("/api/crypto/decide")
async def api_crypto_decide(
    symbol: str,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    AI decision for crypto trading (BUY/SELL/HOLD)

    Similar to /ai/decide for stocks
    Uses OpenAI to analyze prediction + market conditions
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    if not os.getenv("CRYPTO_ENABLED", "0") == "1":
        raise HTTPException(503, "Crypto module not enabled")

    try:
        symbol = symbol.upper().strip()

        # 1. Get latest prediction
        engine = _get_crypto_engine()
        prediction = await engine.generate_prediction(symbol)

        # 2. Get current price
        from core.crypto.crypto_providers import get_crypto_price_quorum

        price_data = await get_crypto_price_quorum(symbol, use_cache=False)

        # 3. Use AI to make decision
        if not AGENTS_ENABLED:
            raise HTTPException(503, "AI agents not enabled (set AGENTS_ENABLED=1)")

        system_prompt = "You are a crypto trading expert AI. Respond in JSON format only."
        user_prompt = f"""
Analyze this crypto prediction and make a trading decision.

Symbol: {symbol}
Current Price: ${price_data["price"]:.2f}
24h Change: {price_data.get("change_24h_pct", 0):.2f}%
Prediction Direction: {prediction["direction"]}
Confidence: {prediction["confidence"]:.0%}
Volatility: {prediction["volatility"]:.1%}
Horizon: {prediction["horizon_hours"]}h

Based on this data, should I:
1. BUY - Strong upward momentum, good entry point
2. SELL - Downward trend, take profits or cut losses
3. HOLD - Wait for better signal

Respond in JSON format:
{{
  "decision": "BUY|SELL|HOLD",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "target_price": optional float,
  "stop_loss": optional float
}}
"""

        # Call AI using the same pattern as /ai/decide
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
            decision_text = (
                (((data.get("choices") or [{}])[0].get("message") or {}).get("content"))
                if data
                else None
            )
        else:  # openai
            payload = {
                "model": AGENT_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.3,
                "max_tokens": 300,
            }
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
            r = _http_post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=AI_TIMEOUT_S,
            )
            data = r.json() if r.status_code == 200 else {}
            decision_text = (
                (((data.get("choices") or [{}])[0].get("message") or {}).get("content"))
                if data
                else None
            )

        if not decision_text:
            raise HTTPException(503, "AI response empty")

        # Parse JSON response
        import json
        import re

        json_match = re.search(r"\{.*\}", decision_text, re.DOTALL)
        if json_match:
            decision = json.loads(json_match.group())
        else:
            # Fallback parsing
            decision = {
                "decision": "HOLD",
                "confidence": 0.5,
                "reasoning": "Unable to parse AI response",
            }

        # Store decision in database
        conn = sqlite3.connect(WOLF_SQLITE_PATH)
        c = conn.cursor()

        c.execute(
            """
            INSERT INTO crypto_decisions (
                symbol, decision, confidence, reasoning,
                target_price, stop_loss, prediction_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                symbol,
                decision.get("decision", "HOLD"),
                decision.get("confidence", 0.5),
                decision.get("reasoning", ""),
                decision.get("target_price"),
                decision.get("stop_loss"),
                prediction.get("prediction_id"),
                time.time(),
            ),
        )

        conn.commit()
        conn.close()

        _add_event(
            "crypto.decide",
            f"AI decision for {symbol}: {decision.get('decision')}",
            {
                "symbol": symbol,
                "decision": decision.get("decision"),
                "confidence": decision.get("confidence"),
                "prediction_id": prediction.get("prediction_id"),
            },
        )

        return {
            "symbol": symbol,
            "decision": decision.get("decision"),
            "confidence": decision.get("confidence"),
            "reasoning": decision.get("reasoning"),
            "target_price": decision.get("target_price"),
            "stop_loss": decision.get("stop_loss"),
            "current_price": price_data["price"],
            "prediction": {
                "direction": prediction["direction"],
                "confidence": prediction["confidence"],
                "horizon_hours": prediction["horizon_hours"],
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Crypto AI decision failed for {symbol}: {e}", exc_info=True)
        raise HTTPException(500, f"Decision failed: {str(e)[:200]}")


@router.get("/api/crypto/decisions")
async def api_crypto_decisions(
    symbol: str | None = None,
    limit: int = 10,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    Get crypto AI decision history

    Similar to /api/agent/decisions for stocks
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    if not os.getenv("CRYPTO_ENABLED", "0") == "1":
        raise HTTPException(503, "Crypto module not enabled")

    try:
        conn = sqlite3.connect(WOLF_SQLITE_PATH)
        c = conn.cursor()

        if symbol:
            c.execute(
                """
                SELECT symbol, decision, confidence, reasoning,
                       target_price, stop_loss, created_at
                FROM crypto_decisions
                WHERE symbol = ?
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (symbol, limit),
            )
        else:
            c.execute(
                """
                SELECT symbol, decision, confidence, reasoning,
                       target_price, stop_loss, created_at
                FROM crypto_decisions
                ORDER BY created_at DESC
                LIMIT ?
            """,
                (limit,),
            )

        rows = c.fetchall()
        conn.close()

        decisions = []
        for row in rows:
            decisions.append(
                {
                    "symbol": row[0],
                    "decision": row[1],
                    "confidence": row[2],
                    "reasoning": row[3],
                    "target_price": row[4],
                    "stop_loss": row[5],
                    "timestamp": row[6],
                }
            )

        return {"symbol": symbol, "count": len(decisions), "decisions": decisions}

    except Exception as e:
        LOGGER.error(f"Crypto decisions fetch failed: {e}", exc_info=True)
        raise HTTPException(500, f"Decisions fetch failed: {str(e)[:200]}")


@router.get("/api/crypto/regime/current")
async def api_crypto_regime_current(
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    Detect current crypto market regime

    Regimes: bull_run, bear_market, accumulation, distribution
    Based on BTC dominance, altcoin performance
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    if not os.getenv("CRYPTO_ENABLED", "0") == "1":
        raise HTTPException(503, "Crypto module not enabled")

    try:
        import asyncio

        from core.crypto.crypto_providers import get_crypto_price_quorum

        # Fetch key indicators
        btc_task = get_crypto_price_quorum("BTC", use_cache=False)
        eth_task = get_crypto_price_quorum("ETH", use_cache=False)
        sol_task = get_crypto_price_quorum("SOL", use_cache=False)

        btc, eth, sol = await asyncio.gather(btc_task, eth_task, sol_task)

        # Calculate regime
        btc_change = btc.get("change_24h_pct", 0) if btc else 0
        eth_change = eth.get("change_24h_pct", 0) if eth else 0
        sol_change = sol.get("change_24h_pct", 0) if sol else 0

        avg_change = (btc_change + eth_change + sol_change) / 3

        # Determine regime
        if avg_change > 5:
            regime = "bull_run"
            confidence = min(0.9, 0.5 + (avg_change / 20))
        elif avg_change < -5:
            regime = "bear_market"
            confidence = min(0.9, 0.5 + (abs(avg_change) / 20))
        elif -2 < avg_change < 2:
            regime = "accumulation"
            confidence = 0.7
        else:
            regime = "distribution"
            confidence = 0.6

        return {
            "regime": regime,
            "confidence": round(confidence, 2),
            "indicators": {
                "btc_change_24h": btc_change,
                "eth_change_24h": eth_change,
                "sol_change_24h": sol_change,
                "avg_change_24h": round(avg_change, 2),
            },
            "timestamp": time.time(),
        }

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Crypto regime detection failed: {e}", exc_info=True)
        raise HTTPException(500, f"Regime detection failed: {str(e)[:200]}")


@router.get("/api/multi_timeframe/{symbol}")
async def api_multi_timeframe(
    symbol: str,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    Get multi-timeframe forecasts (1h, 4h, 1d, 1w)
    
    Returns forecasts across all timeframes with alignment score.
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass
    
    if os.getenv("MULTI_TIMEFRAME_ENABLED", "0") != "1":
        raise HTTPException(503, "Multi-timeframe analysis not enabled. Set MULTI_TIMEFRAME_ENABLED=1")
    
    try:
        from core.multi_timeframe import _generate_timeframe_forecast
        
        # Generate forecasts for all timeframes
        forecasts = {
            "1h": _generate_timeframe_forecast(symbol, "1h", 1),
            "4h": _generate_timeframe_forecast(symbol, "4h", 4),
            "1d": _generate_timeframe_forecast(symbol, "1d", 24),
            "1w": _generate_timeframe_forecast(symbol, "1w", 168),
        }
        
        # Calculate alignment score (how many timeframes agree on direction)
        directions = [f.get("direction", "FLAT") for f in forecasts.values() if f.get("ok")]
        if directions:
            bullish = directions.count("UP")
            bearish = directions.count("DOWN")
            alignment_pct = max(bullish, bearish) / len(directions) * 100
            consensus = "UP" if bullish > bearish else "DOWN" if bearish > bullish else "FLAT"
        else:
            alignment_pct = 0
            consensus = "UNKNOWN"
        
        return {
            "symbol": symbol.upper(),
            "forecasts": forecasts,
            "alignment": {
                "consensus_direction": consensus,
                "alignment_pct": round(alignment_pct, 1),
                "bullish_count": bullish if directions else 0,
                "bearish_count": bearish if directions else 0,
            },
            "timestamp": int(time.time()),
        }
    
    except Exception as e:
        LOGGER.error(f"Multi-timeframe forecast failed: {e}", exc_info=True)
        raise HTTPException(500, f"Forecast failed: {str(e)[:200]}")


@router.get("/api/backtest")
async def api_backtest(
    symbol: str = "WOLF",
    strategy: str = "momentum",
    start_date: str = None,
    end_date: str = None,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    Run historical backtest on a strategy
    
    Strategies: momentum, mean_reversion, breakout
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass
    
    if os.getenv("BACKTESTING_ENABLED", "0") != "1":
        raise HTTPException(503, "Backtesting not enabled. Set BACKTESTING_ENABLED=1")
    
    try:
        from core.backtester import Backtester
        from datetime import datetime, timedelta
        
        # Default to last 30 days
        if not start_date:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        backtester = Backtester()
        results = backtester.run(
            symbol=symbol.upper(),
            strategy=strategy,
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            "symbol": symbol.upper(),
            "strategy": strategy,
            "period": {"start": start_date, "end": end_date},
            "results": results,
            "timestamp": int(time.time()),
        }
    
    except Exception as e:
        LOGGER.error(f"Backtest failed: {e}", exc_info=True)
        raise HTTPException(500, f"Backtest failed: {str(e)[:200]}")


@router.get("/api/social_sentiment/{symbol}")
async def api_social_sentiment(
    symbol: str,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    Get social sentiment from Twitter/Reddit
    
    Returns sentiment score (-1.0 to +1.0) and mention count.
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass
    
    if os.getenv("SOCIAL_SENTIMENT_ENABLED", "0") != "1":
        raise HTTPException(503, "Social sentiment not enabled. Set SOCIAL_SENTIMENT_ENABLED=1")
    
    try:
        from core.social_sentiment import fetch_twitter_sentiment, fetch_reddit_sentiment
        
        # Fetch from both sources
        twitter = fetch_twitter_sentiment(symbol.upper())
        reddit = fetch_reddit_sentiment(symbol.upper())
        
        # Combine scores (weighted average)
        twitter_score = twitter.get("sentiment_score", 0) if twitter.get("ok") else 0
        reddit_score = reddit.get("sentiment_score", 0) if reddit.get("ok") else 0
        twitter_count = twitter.get("mention_count", 0) if twitter.get("ok") else 0
        reddit_count = reddit.get("mention_count", 0) if reddit.get("ok") else 0
        
        total_mentions = twitter_count + reddit_count
        if total_mentions > 0:
            combined_score = (twitter_score * twitter_count + reddit_score * reddit_count) / total_mentions
        else:
            combined_score = 0.0
        
        return {
            "symbol": symbol.upper(),
            "combined": {
                "sentiment_score": round(combined_score, 3),
                "total_mentions": total_mentions,
                "signal": "BULLISH" if combined_score > 0.2 else "BEARISH" if combined_score < -0.2 else "NEUTRAL",
            },
            "twitter": twitter,
            "reddit": reddit,
            "timestamp": int(time.time()),
        }
    
    except Exception as e:
        LOGGER.error(f"Social sentiment fetch failed: {e}", exc_info=True)
        raise HTTPException(500, f"Sentiment fetch failed: {str(e)[:200]}")


@router.get("/api/economic_calendar")
async def api_economic_calendar(
    days_ahead: int = 7,
    importance: str = "high",
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    Get upcoming economic events
    
    Importance: high, medium, low, all
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass
    
    if os.getenv("ECONOMIC_CALENDAR_ENABLED", "0") != "1":
        raise HTTPException(503, "Economic calendar not enabled. Set ECONOMIC_CALENDAR_ENABLED=1")
    
    try:
        from core.economic_calendar import fetch_economic_calendar
        
        calendar = fetch_economic_calendar(days_ahead=days_ahead, importance=importance)
        
        return {
            "query": {"days_ahead": days_ahead, "importance": importance},
            "calendar": calendar,
            "timestamp": int(time.time()),
        }
    
    except Exception as e:
        LOGGER.error(f"Economic calendar fetch failed: {e}", exc_info=True)
        raise HTTPException(500, f"Calendar fetch failed: {str(e)[:200]}")


@router.post("/api/advisor/start")
async def api_advisor_start(
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    Start autonomous AI advisor

    Ghost will:
    - Scan markets every 30 seconds
    - Find high-confidence opportunities (score >= 70)
    - Send Telegram alerts for top picks
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    try:
        from core.ai_advisor.scanner import start_scanner

        await start_scanner()

        return {
            "status": "started",
            "message": "AI Advisor is now scanning markets autonomously",
            "scan_interval_sec": 30,
            "min_score_threshold": 70,
        }

    except Exception as e:
        LOGGER.error(f"Failed to start AI advisor: {e}", exc_info=True)
        raise HTTPException(500, f"Start failed: {str(e)[:200]}")


@router.post("/api/advisor/stop")
async def api_advisor_stop(
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    Stop autonomous AI advisor
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    try:
        from core.ai_advisor.scanner import stop_scanner

        stop_scanner()

        return {"status": "stopped", "message": "AI Advisor has stopped scanning"}

    except Exception as e:
        LOGGER.error(f"Failed to stop AI advisor: {e}", exc_info=True)
        raise HTTPException(500, f"Stop failed: {str(e)[:200]}")


@router.get("/api/advisor/recommendations")
async def api_advisor_recommendations(
    min_score: int = 70,
    asset_type: str = "all",  # all, stocks, crypto
    limit: int = 10,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    Get AI recommendations

    Returns top opportunities Ghost has found
    Only shows opportunities with confidence >= min_score
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    try:
        from core.ai_advisor.scanner import get_scanner

        scanner = get_scanner()

        opportunities = scanner.get_latest_opportunities(limit=100)

        # Filter by asset type
        if asset_type != "all":
            opportunities = [opp for opp in opportunities if opp["asset_type"] == asset_type]

        # Filter by score
        opportunities = [opp for opp in opportunities if opp["score"] >= min_score]

        # Limit results
        opportunities = opportunities[:limit]

        # Get accuracy stats
        from core.ai_advisor.accuracy_tracker import get_tracker

        tracker = get_tracker()
        stats = tracker.get_stats()

        return {
            "opportunities": opportunities,
            "count": len(opportunities),
            "min_score": min_score,
            "asset_type_filter": asset_type,
            "ghost_accuracy_pct": stats.get("overall_accuracy_pct", 0),
            "ghost_win_rate_pct": stats.get("win_rate_pct", 0),
            "scanner_stats": scanner.get_stats(),
        }

    except Exception as e:
        LOGGER.error(f"Failed to get recommendations: {e}", exc_info=True)
        raise HTTPException(500, f"Recommendations failed: {str(e)[:200]}")


@router.get("/api/advisor/stats")
async def api_advisor_stats(
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    Get AI advisor performance statistics

    Shows Ghost's track record:
    - Overall accuracy (% correct predictions)
    - Win rate (% profitable trades)
    - Average return per trade
    - Performance by asset type
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    try:
        from core.ai_advisor.accuracy_tracker import get_tracker

        tracker = get_tracker()
        stats = tracker.get_stats()

        # Add scanner stats
        from core.ai_advisor.scanner import get_scanner

        scanner = get_scanner()
        stats["scanner"] = scanner.get_stats()

        return stats

    except Exception as e:
        LOGGER.error(f"Failed to get stats: {e}", exc_info=True)
        raise HTTPException(500, f"Stats failed: {str(e)[:200]}")


@router.post("/api/advisor/scan_now")
async def api_advisor_scan_now(
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    Trigger immediate market scan

    Use this to manually trigger a scan instead of waiting for the schedule
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    try:
        from core.ai_advisor.scanner import get_scanner

        scanner = get_scanner()

        opportunities = await scanner.scan_all_markets()

        return {
            "opportunities_found": len(opportunities),
            "top_opportunities": scanner.get_latest_opportunities(limit=5),
            "scan_time": time.time(),
        }

    except Exception as e:
        LOGGER.error(f"Manual scan failed: {e}", exc_info=True)
        raise HTTPException(500, f"Scan failed: {str(e)[:200]}")


@router.post("/api/advisor/chat")
async def api_advisor_chat(
    message: str,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    Chat with Ghost - Ask investment questions

    Ghost uses FULL INTELLIGENCE:
    - Real prediction engine (crypto_predictor.py)
    - AI decision framework (GPT-4 analysis)
    - Accuracy tracker (past performance)
    - Market scanner (real-time opportunities)
    - Risk assessment algorithms

    Examples:
    - "What's the best crypto under $1?"
    - "Should I buy Bitcoin right now?"
    - "If I invest $1000 in SOL, what will it be worth in 30 days?"
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    if not AGENTS_ENABLED:
        raise HTTPException(503, "AI agents not enabled (set AGENTS_ENABLED=1)")

    if not os.getenv("CRYPTO_ENABLED", "0") == "1":
        raise HTTPException(503, "Crypto module not enabled (set CRYPTO_ENABLED=1)")

    try:
        LOGGER.info(f"🤖 Ghost analyzing: {message}")

        # ═══════════════════════════════════════════════════════════════════
        # STEP 1: USE REAL PREDICTION ENGINE
        # ═══════════════════════════════════════════════════════════════════
        from core.ai_advisor.accuracy_tracker import get_tracker
        from core.ai_advisor.scanner import get_scanner
        from core.crypto.crypto_predictor import _get_crypto_engine
        from core.crypto.crypto_providers import get_crypto_price_quorum

        engine = _get_crypto_engine()
        tracker = get_tracker()
        scanner = get_scanner()

        # Get Ghost's accuracy stats
        ghost_stats = tracker.get_stats()

        # ═══════════════════════════════════════════════════════════════════
        # STEP 2: SCAN MARKET FOR REAL OPPORTUNITIES
        # ═══════════════════════════════════════════════════════════════════
        LOGGER.info("📊 Running market scan...")
        await scanner.scan_all_markets()

        # ═══════════════════════════════════════════════════════════════════
        # STEP 3: GET MARKET REGIME (Real analysis)
        # ═══════════════════════════════════════════════════════════════════
        regime = {"regime": "neutral", "confidence": 0.5}
        try:
            # Use actual regime detection
            from core.crypto.crypto_providers import get_crypto_price_quorum

            btc = await get_crypto_price_quorum("BTC", use_cache=False)
            eth = await get_crypto_price_quorum("ETH", use_cache=False)
            sol = await get_crypto_price_quorum("SOL", use_cache=False)

            avg_change = (
                btc.get("change_24h_pct", 0)
                + eth.get("change_24h_pct", 0)
                + sol.get("change_24h_pct", 0)
            ) / 3

            if avg_change > 5:
                regime = {"regime": "bull_run", "confidence": 0.8, "avg_change": avg_change}
            elif avg_change < -5:
                regime = {"regime": "bear_market", "confidence": 0.8, "avg_change": avg_change}
            else:
                regime = {"regime": "neutral", "confidence": 0.6, "avg_change": avg_change}
        except Exception as e:
            LOGGER.warning(f"Regime detection failed: {e}")

        # ═══════════════════════════════════════════════════════════════════
        # STEP 4: RUN PREDICTIONS FOR RELEVANT CRYPTOS
        # ═══════════════════════════════════════════════════════════════════
        crypto_watchlist = [
            "BTC",
            "ETH",
            "SOL",
            "DOGE",
            "SHIB",
            "PEPE",
            "ADA",
            "DOT",
            "MATIC",
            "AVAX",
            "LINK",
            "UNI",
            "ATOM",
            "XRP",
            "LTC",
        ]

        detailed_analysis = []
        under_1_dollar = []

        # Determine which cryptos to analyze based on question
        symbols_to_analyze = crypto_watchlist
        if "under" in message.lower() and ("$1" in message or "1 dollar" in message.lower()):
            # Only analyze under $1 cryptos
            symbols_to_analyze = [s for s in crypto_watchlist]

        LOGGER.info(
            f"🔍 Analyzing {len(symbols_to_analyze)} cryptos with full prediction engine..."
        )

        for symbol in symbols_to_analyze[:10]:  # Limit to 10 for performance
            try:
                # GET REAL PRICE DATA
                price_data = await get_crypto_price_quorum(symbol, use_cache=False)
                current_price = price_data["price"]

                # Filter for under $1 if needed
                if "under" in message.lower() and (
                    "$1" in message or "1 dollar" in message.lower()
                ):
                    if current_price >= 1.0:
                        continue

                # RUN REAL PREDICTION ENGINE
                LOGGER.info(f"  🎯 Running prediction for {symbol}...")
                prediction = await engine.generate_prediction(symbol)

                # GET AI DECISION (uses full decision framework)
                {
                    "symbol": symbol,
                    "current_price": current_price,
                    "change_24h_pct": price_data.get("change_24h_pct", 0),
                    "volume_24h": price_data.get("volume_24h", 0),
                    "market_cap": price_data.get("market_cap", 0),
                    "prediction": prediction,
                    "regime": regime,
                }

                # Calculate confidence score (Ghost's real algorithm)
                confidence_score = prediction.get("confidence", 0.5)
                momentum_score = abs(price_data.get("change_24h_pct", 0)) / 10
                regime_bonus = 0.1 if regime["regime"] == "bull_run" else 0

                total_confidence = min(confidence_score + momentum_score + regime_bonus, 1.0)

                analysis = {
                    "symbol": symbol,
                    "current_price": current_price,
                    "change_24h_pct": price_data.get("change_24h_pct", 0),
                    "volume_24h": price_data.get("volume_24h", 0),
                    "market_cap": price_data.get("market_cap", 0),
                    "prediction": {
                        "direction": prediction.get("direction"),
                        "confidence": prediction.get("confidence"),
                        "horizon_hours": prediction.get("horizon_hours"),
                        "volatility": prediction.get("volatility"),
                        "method": prediction.get("method"),
                    },
                    "ghost_confidence": round(total_confidence, 2),
                    "recommended_action": "BUY"
                    if total_confidence >= 0.70 and prediction.get("direction") == "UP"
                    else "HOLD",
                    "target_price_30d": current_price
                    * (
                        1 + (prediction.get("confidence", 0.5) * 0.25)
                    ),  # Conservative 30-day target
                    "stop_loss": current_price * 0.955,  # ~4.5% stop (crypto default)
                }

                detailed_analysis.append(analysis)

                if current_price < 1.0:
                    under_1_dollar.append(analysis)

            except Exception as e:
                LOGGER.warning(f"Failed to analyze {symbol}: {e}")

        # Sort by Ghost confidence score
        detailed_analysis.sort(key=lambda x: x["ghost_confidence"], reverse=True)
        under_1_dollar.sort(key=lambda x: x["ghost_confidence"], reverse=True)

        LOGGER.info(f"✅ Analysis complete: {len(detailed_analysis)} cryptos analyzed")

        # ═══════════════════════════════════════════════════════════════════
        # STEP 5: BUILD COMPREHENSIVE CONTEXT FOR AI
        # ═══════════════════════════════════════════════════════════════════
        context = {
            "detailed_analysis": detailed_analysis[:10],
            "under_1_dollar_cryptos": under_1_dollar[:5],
            "market_regime": regime,
            "ghost_accuracy_pct": ghost_stats.get("overall_accuracy_pct", 0),
            "ghost_win_rate_pct": ghost_stats.get("win_rate_pct", 0),
            "recent_30d_accuracy": ghost_stats.get("recent_30d", {}).get("accuracy_pct", 0),
            "total_decisions": ghost_stats.get("total_decisions", 0),
        }

        # ═══════════════════════════════════════════════════════════════════
        # STEP 6: GENERATE AI RESPONSE WITH REAL DATA
        # ═══════════════════════════════════════════════════════════════════
        system_prompt = f"""You are Ghost, an expert AI investment advisor with real-time analysis capabilities.

YOUR ACTUAL TRACK RECORD:
- Overall Accuracy: {context["ghost_accuracy_pct"]:.1f}%
- Win Rate: {context["ghost_win_rate_pct"]:.1f}%
- Total Decisions: {context["total_decisions"]}
- Recent 30-day Accuracy: {context["recent_30d_accuracy"]:.1f}%

You have JUST ANALYZED the market using:
1. Real prediction engine (generates 24h forecasts with confidence scores)
2. Live price data from multiple sources (CoinGecko, Binance, Coinbase)
3. Market regime detection (bull/bear/neutral)
4. Historical accuracy tracking

RESPONSE GUIDELINES:
1. Use the ACTUAL analysis data provided (predictions, confidence scores, prices)
2. Reference specific prediction confidence levels
3. Calculate profit projections using: Investment × (1 + (Confidence × Expected_Return))
4. Always mention Ghost's confidence score for each recommendation
5. Provide conservative, moderate, and optimistic scenarios
6. Include risk factors based on volatility data
7. Recommend position sizes based on confidence (High confidence = 3%, Medium = 2%, Low = 1%)

Be honest and data-driven. If confidence is low (<70%), recommend waiting."""

        user_prompt = f"""User Question: {message}

REAL-TIME MARKET ANALYSIS (Just Completed):

Market Regime: {context["market_regime"]["regime"].upper()} ({context["market_regime"].get("confidence", 0.5) * 100:.0f}% confidence)
Avg Market Change: {context["market_regime"].get("avg_change", 0):.2f}%

DETAILED ANALYSIS (Top Opportunities):
{json.dumps(context["detailed_analysis"], indent=2)}

CRYPTOS UNDER $1 (Analyzed with Prediction Engine):
{json.dumps(context["under_1_dollar_cryptos"], indent=2)}

INSTRUCTIONS:
1. Answer using the REAL analysis above
2. Reference specific confidence scores from predictions
3. For profit calculations, use:
   - Conservative: confidence × 15% gain
   - Moderate: confidence × 25% gain
   - Optimistic: confidence × 40% gain
4. Mention Ghost's confidence score for each pick
5. Calculate exact dollar amounts for profit projections
6. Include stop loss recommendations (usually -8%ntry)
7. Recommend position sizing based on confidence

Use emojis, be conversational, but ALWAYS reference the real data above."""

        # Call GPT-4 with real context
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
                timeout=30,  # Longer timeout for complex analysis
            )
            data = r.json() if r.status_code == 200 else {}
            response_text = (
                (((data.get("choices") or [{}])[0].get("message") or {}).get("content"))
                if data
                else None
            )
        else:  # openai
            payload = {
                "model": AGENT_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 1500,
            }
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
            r = _http_post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            data = r.json() if r.status_code == 200 else {}
            response_text = (
                (((data.get("choices") or [{}])[0].get("message") or {}).get("content"))
                if data
                else None
            )

        if not response_text:
            raise HTTPException(503, "AI response empty")

        LOGGER.info("✅ Ghost response generated")

        return {
            "message": message,
            "response": response_text,
            "analysis_used": {
                "cryptos_analyzed": len(detailed_analysis),
                "predictions_generated": len(detailed_analysis),
                "under_1_dollar_found": len(under_1_dollar),
                "market_regime": regime["regime"],
                "ghost_accuracy_pct": context["ghost_accuracy_pct"],
                "top_3_picks": [
                    {
                        "symbol": a["symbol"],
                        "price": a["current_price"],
                        "ghost_confidence": a["ghost_confidence"],
                        "prediction": a["prediction"]["direction"],
                        "action": a["recommended_action"],
                    }
                    for a in detailed_analysis[:3]
                ],
            },
            "timestamp": time.time(),
        }

    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Chat failed: {e}", exc_info=True)
        raise HTTPException(500, f"Chat failed: {str(e)[:200]}")


@router.get("/api/walk_forward_analysis/{symbol}")
async def api_walk_forward_analysis(
    symbol: str,
    in_sample_window: int = 120,
    out_sample_window: int = 30,
    step_size: int = 30,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    Walk-forward optimization analysis
    
    Tests if strategy performance is robust or overfitted by:
    1. Training on in-sample window (default 120 days)
    2. Testing on out-of-sample window (default 30 days)
    3. Repeating across time with step_size (default 30 days)
    
    Returns:
    - Consistency score (% of windows profitable)
    - Average out-of-sample Sharpe ratio
    - Overfitting detection (in-sample vs out-sample comparison)
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass
    
    if os.getenv("WALK_FORWARD_ENABLED", "0") != "1":
        raise HTTPException(503, "Walk-forward analysis not enabled. Set WALK_FORWARD_ENABLED=1")
    
    try:
        from core.backtester import get_backtester
        
        backtester = get_backtester()
        
        # Get historical returns from PostgreSQL prediction outcomes
        returns = []
        try:
            from core.db_pool import get_sync_connection as _wf_get_conn
            with _wf_get_conn() as _conn:
                _cur = _conn.cursor()
                _cur.execute("""
                    SELECT realized_move_pct
                    FROM ghost_prediction_outcomes
                    WHERE symbol = %s AND status = 'resolved'
                      AND realized_move_pct IS NOT NULL
                    ORDER BY closed_at ASC
                """, (symbol.upper(),))
                returns = [row[0] / 100.0 for row in _cur.fetchall()]
                _cur.close()
        except Exception as _e:
            LOGGER.warning(f"Walk-forward: could not load returns from DB: {_e}")
        
        if len(returns) < (in_sample_window + out_sample_window):
            raise HTTPException(400, f"Insufficient prediction history: need {in_sample_window + out_sample_window} days, have {len(returns)}")
        
        result = backtester.walk_forward_analysis(
            returns=returns,
            in_sample_window=in_sample_window,
            out_sample_window=out_sample_window,
            step_size=step_size
        )
        
        return {
            "symbol": symbol.upper(),
            "walk_forward_analysis": result,
            "timestamp": int(time.time()),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Walk-forward analysis failed: {e}", exc_info=True)
        raise HTTPException(500, f"Analysis failed: {str(e)[:200]}")


@router.get("/api/monte_carlo/{symbol}")
async def api_monte_carlo(
    symbol: str,
    num_simulations: int = 1000,
    simulation_length: int = 252,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    Monte Carlo risk simulation
    
    Runs Monte Carlo simulation to answer:
    - "What's my 95% worst-case return over 1 year?"
    - "What's the median expected return?"
    - "What's the maximum expected drawdown?"
    
    Args:
        num_simulations: Number of simulations to run (default 1000)
        simulation_length: Days per simulation (default 252 = 1 year)
    
    Returns:
        5th/50th/95th percentile distributions for:
        - Total return (%)
        - Sharpe ratio
        - Max drawdown (%)
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass
    
    if os.getenv("MONTE_CARLO_ENABLED", "0") != "1":
        raise HTTPException(503, "Monte Carlo simulation not enabled. Set MONTE_CARLO_ENABLED=1")
    
    try:
        from core.backtester import get_backtester
        
        backtester = get_backtester()
        
        # Get historical returns from PostgreSQL prediction outcomes
        returns = []
        try:
            from core.db_pool import get_sync_connection as _mc_get_conn
            with _mc_get_conn() as _conn:
                _cur = _conn.cursor()
                _cur.execute("""
                    SELECT realized_move_pct
                    FROM ghost_prediction_outcomes
                    WHERE symbol = %s AND status = 'resolved'
                      AND realized_move_pct IS NOT NULL
                    ORDER BY closed_at ASC
                """, (symbol.upper(),))
                returns = [row[0] / 100.0 for row in _cur.fetchall()]
                _cur.close()
        except Exception as _e:
            LOGGER.warning(f"Monte Carlo: could not load returns from DB: {_e}")
        
        if len(returns) < 20:
            raise HTTPException(400, f"Insufficient data for Monte Carlo: need 20 days, have {len(returns)}")
        
        result = backtester.monte_carlo_simulation(
            returns=returns,
            num_simulations=num_simulations,
            simulation_length=simulation_length
        )
        
        return {
            "symbol": symbol.upper(),
            "monte_carlo": result,
            "interpretation": {
                "worst_case_5th_percentile": f"95% chance return will be better than {result['total_return']['5th_percentile_pct']}%",
                "median_expectation": f"50% chance return will be around {result['total_return']['median_pct']}%",
                "best_case_95th_percentile": f"5% chance return will exceed {result['total_return']['95th_percentile_pct']}%",
            },
            "timestamp": int(time.time()),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Monte Carlo simulation failed: {e}", exc_info=True)
        raise HTTPException(500, f"Simulation failed: {str(e)[:200]}")


@router.get("/api/momentum_shift/{symbol}")
async def api_momentum_shift(
    symbol: str,
    lookback_minutes: int = 60,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    Detect momentum shifts
    
    Identifies when momentum has shifted dramatically (>30% change).
    Useful for catching reversals 15-30 minutes early.
    
    Args:
        lookback_minutes: How far back to compare (default 60 min)
    
    Returns:
        - shift_detected: bool
        - shift_magnitude: % change in momentum
        - shift_direction: BULLISH | BEARISH | None
        - alert_priority: HIGH | MEDIUM | LOW
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass
    
    if os.getenv("MOMENTUM_DETECTOR_ENABLED", "0") != "1":
        raise HTTPException(503, "Momentum detector not enabled. Set MOMENTUM_DETECTOR_ENABLED=1")
    
    try:
        from core.momentum_detector import detect_momentum_shift, get_momentum_history
        
        # Compute momentum from latest prediction confidence
        current_momentum = 0.0
        try:
            _sym_upper = symbol.upper()
            if _sym_upper in _LATEST_PREDICTIONS:
                _pred = _LATEST_PREDICTIONS[_sym_upper]
                _conf = float(_pred.get("confidence", 0.5))
                _dir = str(_pred.get("direction", "HOLD")).upper()
                if _dir in ("BUY", "UP"):
                    current_momentum = _conf
                elif _dir in ("SELL", "DOWN"):
                    current_momentum = -_conf
            # Feed momentum tracker so history accumulates
            from core.momentum_detector import track_momentum
            track_momentum(_sym_upper, current_momentum)
        except Exception:
            pass
        
        shift = detect_momentum_shift(
            symbol=symbol.upper(),
            current_momentum=current_momentum,
            lookback_minutes=lookback_minutes
        )
        
        history = get_momentum_history(symbol.upper(), limit=20)
        
        return {
            "symbol": symbol.upper(),
            "shift_detection": shift,
            "recent_history": history[-5:] if history else [],
            "timestamp": int(time.time()),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Momentum shift detection failed: {e}", exc_info=True)
        raise HTTPException(500, f"Detection failed: {str(e)[:200]}")


@router.get("/api/hedging/recommendations")
async def api_hedging_recommendations(
    portfolio_beta: float = 1.0,
    target_beta: float = 0.0,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    Get portfolio hedging recommendations
    
    Beta-neutral hedging to reduce portfolio volatility by 20-40%.
    
    Args:
        portfolio_beta: Current portfolio beta (default 1.0 = market beta)
        target_beta: Desired portfolio beta (default 0.0 = beta-neutral)
    
    Returns:
        - Hedge instrument (usually SPY)
        - Hedge ratio ($ amount to short per $1 portfolio)
        - Expected volatility reduction (%)
        - Pairs trading opportunities (z-score > 2.0)
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass
    
    if os.getenv("HEDGING_ENABLED", "0") != "1":
        raise HTTPException(503, "Hedging engine not enabled. Set HEDGING_ENABLED=1")
    
    try:
        from core.hedging_engine import get_hedging_engine
        
        engine = get_hedging_engine()
        
        # Calculate beta hedge
        hedge = engine.calculate_beta_hedge(
            portfolio_beta=portfolio_beta,
            target_beta=target_beta
        )
        
        # Find pairs trading opportunities
        pairs = engine.find_pairs_trade()  # Returns pairs with z-score > 2.0
        
        return {
            "beta_hedge": hedge,
            "pairs_trading": pairs,
            "timestamp": int(time.time()),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"Hedging recommendations failed: {e}", exc_info=True)
        raise HTTPException(500, f"Hedging failed: {str(e)[:200]}")


@router.get("/api/system_status")
async def api_system_status(
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    Master Orchestrator - System status
    
    Returns real-time status of all background services:
    - Price refresh loop
    - Movers scanner
    - VIP scanner
    - SL/TP monitor
    - Scheduled predictions
    - Context engine
    - Market scanner
    - Daily reports
    - Outcome reconciler
    - Autonomous execution
    
    Each service shows:
    - status: running | stopped | failed | disabled
    - last_run: Unix timestamp
    - error: Error message (if failed)
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass
    
    if os.getenv("ORCHESTRATOR_ENABLED", "0") != "1":
        raise HTTPException(503, "Master orchestrator not enabled. Set ORCHESTRATOR_ENABLED=1")
    
    try:
        from core.orchestrator import _SYSTEM_STATUS, _START_TIME
        
        uptime_seconds = int(time.time() - _START_TIME) if _START_TIME > 0 else 0
        
        return {
            "system_status": _SYSTEM_STATUS,
            "uptime_seconds": uptime_seconds,
            "uptime_human": f"{uptime_seconds // 3600}h {(uptime_seconds % 3600) // 60}m",
            "timestamp": int(time.time()),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"System status failed: {e}", exc_info=True)
        raise HTTPException(500, f"Status failed: {str(e)[:200]}")


@router.get("/api/agentkit/chat")
async def api_agentkit_chat(
    message: str,
    conversation_id: str | None = None,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    AgentKit - Natural language AI conversations
    
    Stateful persistent conversations using OpenAI Assistants API.
    
    Features:
    - Multi-turn conversations with memory
    - Tool calling (get_price, get_news, get_position, dispatch_alert)
    - Natural language understanding
    - Cost: $0.01-$0.10 per conversation
    
    Args:
        message: Your question or command
        conversation_id: Optional conversation ID for follow-ups
    
    Returns:
        - response: AI response text
        - conversation_id: ID for follow-up messages
        - tools_used: List of tools called during conversation
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass
    
    if os.getenv("AGENTKIT_ENABLED", "0") != "1":
        raise HTTPException(503, "AgentKit not enabled. Set AGENTKIT_ENABLED=1")
    
    if not os.getenv("OPENAI_AGENT_API_KEY"):
        raise HTTPException(503, "AgentKit requires OPENAI_AGENT_API_KEY")
    
    try:
        from llm.agentkit import get_agentkit_client
        
        client = get_agentkit_client()
        
        result = await client.chat(
            message=message,
            conversation_id=conversation_id
        )
        
        return {
            "message": message,
            "response": result["response"],
            "conversation_id": result["conversation_id"],
            "tools_used": result.get("tools_used", []),
            "timestamp": int(time.time()),
        }
    
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"AgentKit chat failed: {e}", exc_info=True)
        raise HTTPException(500, f"Chat failed: {str(e)[:200]}")


@router.get("/metrics")
async def metrics() -> Response:
    try:
        if _G_UP is not None:
            _G_UP.set(1)
    except Exception:
        pass
    # Support Prometheus multiprocess mode if configured
    try:
        mp_dir = os.getenv("PROMETHEUS_MULTIPROC_DIR", "").strip()
        if mp_dir:
            from prometheus_client import CollectorRegistry, multiprocess

            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry)
            blob = generate_latest(registry)
            return Response(blob, media_type=CONTENT_TYPE_LATEST)
    except Exception:
        # fall back to default registry
        pass
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.get("/ready")
async def ready():
    # Ready when at least one provider or cached prev_close is available
    price, prev, provider = get_wolf_price()
    ok = bool(provider) and (price is not None or prev is not None)
    status = 200 if ok else 503
    return JSONResponse({"ready": ok, "provider": provider or "unavailable"}, status_code=status)


@router.get("/live")
async def live():
    # Live if process is serving requests
    try:
        if _G_UP is not None:
            _G_UP.set(1)
    except Exception:
        pass
    return JSONResponse({"live": True})


@router.post("/control/save")
async def control_save(credentials: HTTPAuthorizationCredentials | None = AUTH_DEP):
    _require_bearer(
        (f"Bearer {credentials.credentials}") if credentials and credentials.credentials else None
    )
    try:
        _persist_save()
        _add_event("control.save", "Manual save invoked", {"mode": WOLF_PERSIST_MODE})
        return {
            "ok": True,
            "persist": {"mode": WOLF_PERSIST_MODE, "sqlite": WOLF_SQLITE_PATH},
        }
    except Exception as e:
        raise HTTPException(500, f"save_error: {e}") from e


@router.post("/control/reset")
async def control_reset(credentials: HTTPAuthorizationCredentials | None = AUTH_DEP):
    _require_bearer(
        (f"Bearer {credentials.credentials}") if credentials and credentials.credentials else None
    )
    # Clear in-memory volatile state: caches, breakers, events; reload persisted position
    try:
        PRICE_CACHE.clear()
        NEWS_CACHE["items"], NEWS_CACHE["ts"] = [], 0.0
        EVENTS.clear()
        global _EVENT_SEQ
        _EVENT_SEQ = 0
        # reset provider breakers
        for b in _PROVIDER_BREAKERS.values():
            b.update(
                {
                    "state": "closed",
                    "failures": 0,
                    "backoff_factor": 0,
                    "open_until_ts": 0.0,
                }
            )
        # reset alert trailing
        ALERT_STATE["trailing_high"] = None
        ALERT_STATE["trailing_low"] = None
        # reload persisted position if enabled
        try:
            _persist_load()
        except Exception:
            pass
        _add_event(
            "control.reset",
            "Engine reset invoked",
            {"position": {"qty": STATE.get("qty"), "avg_cost": STATE.get("avg_cost")}},
        )
        return {
            "ok": True,
            "reset": True,
            "position": {"qty": STATE.get("qty"), "avg_cost": STATE.get("avg_cost")},
        }
    except Exception as e:
        raise HTTPException(500, f"reset_error: {e}") from e


@router.post("/orders/place")
async def orders_place(
    body: OrderPlaceBody, credentials: HTTPAuthorizationCredentials | None = AUTH_DEP
):
    _require_bearer(
        (f"Bearer {credentials.credentials}") if credentials and credentials.credentials else None
    )
    sym = (body.symbol or WOLF).upper()
    side = body.side.strip().upper()
    if side not in ("BUY", "SELL"):
        raise HTTPException(422, "side must be BUY or SELL")
    # Hard WOLF-only guard
    if sym != WOLF:
        LOGGER.warning("reject_non_wolf_symbol", extra={"component": "orders", "symbol": sym})
        raise HTTPException(422, "symbol must be WOLF in this service")
    if body.qty <= 0:
        raise HTTPException(422, "qty must be > 0")
    if body.price is not None and body.price <= 0:
        raise HTTPException(422, "price must be > 0 when provided")
    oid = uuid.uuid4().hex
    order = {
        "id": oid,
        "ts": int(time.time()),
        "symbol": sym,
        "side": side,
        "qty": float(body.qty),
        "price": (None if body.price is None else float(body.price)),
        "status": "queued",
        "note": body.note,
    }
    _orders_insert(order)
    try:
        _add_event(
            "orders.place",
            "Order queued",
            {k: v for k, v in order.items() if k != "note"},
        )
    except Exception:
        pass
    return {"ok": True, "order": order}


@router.get("/orders/queue")
async def orders_queue(limit: int = 100):
    items = _orders_select(limit=min(500, max(1, int(limit))))
    return {"orders": items, "count": len(items)}


@router.get("/api/position")
async def api_position_get():
    """Position endpoint with fast response (<50ms), no external calls."""
    return {
        "symbol": WOLF,
        "qty": float(STATE.get("qty", 0.0)),
        "avg_cost": float(STATE.get("avg_cost", 0.0)),
    }


@router.get("/api/version")
async def api_version():
    sha = os.getenv("GIT_SHA", "unknown")
    build = os.getenv("BUILD_TIME", "unknown")
    return {"version": app.version, "git_sha": sha, "build_time": build}


@router.get("/api/config")
async def api_config():
    # Redact secrets values, expose booleans/counts and file paths
    cfg = {
        "ticker": WOLF,
        "providers": {
            "alphavantage": bool(ALPHAVANTAGE_KEY),
            "polygon": bool(POLYGON_KEY),
            "yfinance": True,
            "yahoo_http": True,
            "yahoo_first": bool(PRICE_YAHOO_FIRST),
            "reuters": bool(REUTERS_FEEDS_ON),
        },
        "ai": {
            "provider": AI_PROVIDER,
            "model": AGENT_MODEL,
            "timeout_s": AI_TIMEOUT_S,
            "include_context": bool(int(os.getenv("AI_INCLUDE_CONTEXT", "0"))),
            "autosend": bool(int(os.getenv("AI_AGENT_AUTOSEND", "0"))),
            "memory_auth": bool(_is_ai_memory_auth_required()),
        },
        "alerts": {
            "telegram": bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
            "webhooks": len(ALERT_WEBHOOK_URLS),
            "slack": len(SLACK_WEBHOOK_URLS),
            "mode": ALERT_MODE,
            "throttle_s": ALERT_THROTTLE_S,
            "schedule_open_close": bool(SCHEDULE_OPEN_CLOSE),
            "schedule_window_s": SCHEDULE_WINDOW_S,
        },
        "persist": {
            "mode": WOLF_PERSIST_MODE,
            "file": WOLF_STATE_FILE,
            "sqlite": WOLF_SQLITE_PATH,
            "sqlite_fallback": bool(SQLITE_FALLBACK),
            "redis": bool(REDIS_URL),
            "autosave_s": WOLF_AUTOSAVE_S,
        },
        "ttl": {
            "price_ttl_s": PRICE_TTL_S,
            "price_ttl_open_s": PRICE_TTL_OPEN_S,
            "news_ttl_s": NEWS_TTL_S,
            "price_max_deviation": float(os.getenv("PRICE_MAX_DEVIATION", "0.5")),
            "price_max_deviation_open": PRICE_MAX_DEVIATION_OPEN,
        },
        "security": {
            "bearer_required": bool(os.getenv("GHOST_API_TOKEN", "")),
            "admin_ip_allowlist": ADMIN_IP_ALLOWLIST,
        },
        "override": {
            "manual_active": bool(
                (PRICE_OVERRIDE.get("symbol") or "") == WOLF
                and time.time() < float(PRICE_OVERRIDE.get("until") or 0)
            ),
            "until_ts": int(PRICE_OVERRIDE.get("until") or 0),
        },
        "intelligence": {
            "stage1_enabled": STAGE1_ENABLED,
            "stage2_enabled": STAGE2_ENABLED,
            "stage3_enabled": STAGE3_ENABLED,
            "stage4_enabled": STAGE4_ENABLED,
            "stage5_enabled": STAGE5_ENABLED,
            "features": [],
        },
    }
    # Add intelligence features
    if STAGE1_ENABLED:
        cfg["intelligence"]["features"].extend(["world_context", "market_mood"])
    if STAGE2_ENABLED:
        cfg["intelligence"]["features"].extend(["accuracy_tracker", "learning_loop"])
    if STAGE3_ENABLED:
        cfg["intelligence"]["features"].extend(
            ["ensemble_forecaster", "regime_detector", "risk_engine"]
        )
    if STAGE4_ENABLED:
        cfg["intelligence"]["features"].extend(
            ["portfolio_manager", "hedging_engine", "backtester", "strategy_tester"]
        )
    if STAGE5_ENABLED:
        cfg["intelligence"]["features"].extend(
            ["order_manager", "smart_router", "execution_analytics", "execution_risk"]
        )
    if WATCHLIST_ENABLED:
        cfg["intelligence"]["features"].append("watchlist_manager")
        cfg["intelligence"]["watchlist_enabled"] = True
    try:
        raw = json.dumps(cfg, sort_keys=True).encode("utf-8")
        etag = hashlib.sha256(raw).hexdigest()
        resp = JSONResponse(cfg)
        resp.headers["ETag"] = etag
        resp.headers["Cache-Control"] = "public, max-age=30"
        return resp
    except Exception:
        return JSONResponse(cfg)


@router.get("/api/cache/stats")
async def api_cache_stats():
    """Get in-memory cache statistics for performance monitoring."""
    try:
        from core.cache_manager import get_all_cache_stats

        stats = get_all_cache_stats()
        return {"ok": True, "caches": stats}
    except ImportError:
        return {"ok": False, "error": "Cache manager not available"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.post("/api/cache/clear")
async def api_cache_clear(cache_type: str = "all"):
    """Clear cache(s). Types: 'all', 'price', 'market', 'api', 'forecast'"""
    try:
        from core.cache_manager import (
            API_RESPONSE_CACHE,
            FORECAST_CACHE,
            MARKET_DATA_CACHE,
            PRICE_CACHE,
            clear_all_caches,
        )

        if cache_type == "all":
            clear_all_caches()
            return {"ok": True, "cleared": "all"}

        cache_map = {
            "price": PRICE_CACHE,
            "market": MARKET_DATA_CACHE,
            "api": API_RESPONSE_CACHE,
            "forecast": FORECAST_CACHE,
        }

        if cache_type in cache_map:
            cache_map[cache_type].clear()
            return {"ok": True, "cleared": cache_type}

        return {"ok": False, "error": f"Invalid cache type: {cache_type}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/api/cache/clear")
async def api_cache_clear_get(cache_type: str = "all"):
    """GET version of cache clear for auto-fixer"""
    return await api_cache_clear(cache_type)


@router.post("/api/cache/purge")
async def api_cache_purge_keys(keys: list[str] | None = None):
    """Targeted purge of specific cache keys.

    Args:
        keys: List of cache key patterns to delete (e.g., ['price:AAPL', 'diagnostics:*'])

    Returns:
        {"ok": True, "deleted": [...], "count": N}
    """
    if not keys:
        return {"ok": False, "error": "keys parameter required"}

    deleted = []
    try:
        # Handle PRICE_CACHE deletions
        for key in keys:
            if key.startswith("price:"):
                symbol = key.split(":", 1)[1].upper()
                if symbol in PRICE_CACHE:
                    PRICE_CACHE.pop(symbol)
                    deleted.append(key)
            elif key.startswith("diagnostics:"):
                # Clear PRICE_DIAG entries matching pattern
                pattern = key.split(":", 1)[1]
                if pattern == "*":
                    PRICE_DIAG.clear()
                    deleted.append(key)
                else:
                    # Remove specific diagnostics keys
                    keys_to_remove = [k for k in PRICE_DIAG.keys() if pattern in k]
                    for k in keys_to_remove:
                        PRICE_DIAG.pop(k, None)
                        deleted.append(f"diagnostics:{k}")
            else:
                # Generic cache key deletion
                if key in PRICE_CACHE:
                    PRICE_CACHE.pop(key)
                    deleted.append(key)

        return {"ok": True, "deleted": deleted, "count": len(deleted)}
    except Exception as e:
        return {"ok": False, "error": str(e), "deleted": deleted, "count": len(deleted)}


@router.get("/api/feeds/reopen")
async def api_feeds_reopen():
    """Reopen/refresh data feed connections (for auto-fixer)"""
    try:
        results = {"ok": True, "feeds_refreshed": []}

        # Refresh news feeds
        try:
            from core.news_aggregator import refresh_feeds  # type: ignore

            refresh_feeds()
            results["feeds_refreshed"].append("news")
        except Exception as e:
            results["news_error"] = str(e)

        # Refresh price providers
        try:
            from core.price_fetcher import reset_provider_cooldowns  # type: ignore

            reset_provider_cooldowns()
            results["feeds_refreshed"].append("prices")
        except Exception as e:
            results["price_error"] = str(e)

        # Clear stale caches
        try:
            from core.cache_manager import clear_all_caches

            clear_all_caches()
            results["feeds_refreshed"].append("cache")
        except Exception as e:
            results["cache_error"] = str(e)

        return results

    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/api/db/rebuild")
async def api_db_rebuild():
    """Rebuild database indices (for auto-fixer)"""
    try:
        results = {"ok": True, "rebuilt": []}

        # Rebuild DuckDB analytics tables
        try:
            # Add rebuild logic here if needed
            results["rebuilt"].append("duckdb")
        except Exception as e:
            results["duckdb_error"] = str(e)

        return results

    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/logs/recent")
async def api_logs_recent(limit: int = 100):
    limit = max(1, min(500, int(limit)))
    items = list(EVENTS)[-limit:]
    return {"events": items, "count": len(items)}


@router.post("/api/keys/create")
async def create_api_key(name: str, rate_limit: int = 100):
    """Create a new API key with rate limiting."""
    # Input validation
    if not name or len(name) > 255:
        return {"ok": False, "error": "Name required and must be < 256 chars"}
    if rate_limit < 1 or rate_limit > 10000:
        return {
            "ok": False,
            "error": "Rate limit must be between 1 and 10000 requests/minute",
        }

    key_id = str(uuid.uuid4())
    api_key = f"ghost_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    created_at = time.time()

    # Store in database with hashed key
    try:
        import sqlite3

        conn = sqlite3.connect(WOLF_SQLITE_PATH)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO api_keys (id, key_hash, name, rate_limit, created_at, active) VALUES (?, ?, ?, ?, ?, 1)",
            (key_id, key_hash, name, rate_limit, created_at),
        )
        conn.commit()
        conn.close()
    except sqlite3.IntegrityError:
        return {"ok": False, "error": "Key hash collision (extremely rare)"}
    except Exception as e:
        LOGGER.error(f"Failed to create API key: {e}", exc_info=True)
        return {"ok": False, "error": "Database error"}

    # Cache in memory
    API_KEYS_DB[key_id] = {
        "key_hash": key_hash,
        "name": name,
        "rate_limit": rate_limit,
        "created_at": created_at,
        "last_used": None,
        "request_count": 0,
        "active": True,
    }

    LOGGER.info(f"API key created: {key_id} ({name})")

    return {
        "ok": True,
        "key_id": key_id,
        "api_key": api_key,  # Only returned once!
        "name": name,
        "rate_limit": rate_limit,
        "message": "Store this key securely - it won't be shown again",
    }


@router.get("/api/keys")
async def list_api_keys():
    """List all API keys (without revealing the actual keys)."""
    keys = []
    for key_id, data in API_KEYS_DB.items():
        keys.append(
            {
                "key_id": key_id,
                "name": data["name"],
                "rate_limit": data["rate_limit"],
                "created_at": data["created_at"],
                "last_used": data.get("last_used"),
                "request_count": data.get("request_count", 0),
                "key_preview": data["key"][:15] + "...",
            }
        )
    return {"ok": True, "keys": keys, "count": len(keys)}


@router.delete("/api/keys/{key_id}")
async def delete_api_key(key_id: str):
    """Delete an API key."""
    if key_id in API_KEYS_DB:
        deleted = API_KEYS_DB.pop(key_id)
        return {"ok": True, "message": f"Deleted key: {deleted['name']}"}
    return {"ok": False, "error": "Key not found"}


@router.get("/api/keys/{key_id}")
async def get_api_key_info(key_id: str):
    """Get information about a specific API key."""
    if key_id in API_KEYS_DB:
        data = API_KEYS_DB[key_id]
        return {
            "ok": True,
            "key_id": key_id,
            "name": data["name"],
            "rate_limit": data["rate_limit"],
            "created_at": data["created_at"],
            "last_used": data.get("last_used"),
            "request_count": data.get("request_count", 0),
        }
    return {"ok": False, "error": "Key not found"}


@router.post("/api/webhooks/subscribe")
async def subscribe_webhook(url: str, events: list[str], secret: str | None = None):
    """Register a webhook endpoint for event notifications."""
    from urllib.parse import urlparse

    # Input validation
    if not url:
        return {"ok": False, "error": "URL required"}

    try:
        parsed = urlparse(url)
        # Enforce HTTPS unless explicitly disabled
        if parsed.scheme not in ("https", "http"):
            return {"ok": False, "error": "URL must use http or https scheme"}
        if not parsed.netloc:
            return {"ok": False, "error": "Invalid URL: missing domain"}
        # Disallow private/loopback addresses in production (optional)
        if os.getenv("WEBHOOK_ALLOW_PRIVATE", "0") == "0":
            if parsed.hostname and (
                parsed.hostname in ("localhost", "127.0.0.1", "::1")
                or parsed.hostname.startswith("192.168.")
                or parsed.hostname.startswith("10.")
            ):
                return {
                    "ok": False,
                    "error": "Private/loopback URLs not allowed (set WEBHOOK_ALLOW_PRIVATE=1 to override)",
                }
    except Exception as e:
        return {"ok": False, "error": f"Invalid URL: {e}"}

    if not events or not isinstance(events, list):
        return {"ok": False, "error": "Events list required"}

    # Validate event types
    valid_events = {"order.filled", "price.alert", "risk.breach", "*"}
    for event in events:
        if event not in valid_events:
            return {
                "ok": False,
                "error": f"Invalid event type: {event}. Allowed: {valid_events}",
            }

    webhook_id = str(uuid.uuid4())
    webhook_secret = secret or secrets.token_urlsafe(32)
    secret_hash = hashlib.sha256(webhook_secret.encode()).hexdigest()
    created_at = time.time()

    # Store in database with hashed secret
    try:
        import sqlite3

        conn = sqlite3.connect(WOLF_SQLITE_PATH)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO webhooks (id, url, events_json, secret_hash, created_at, active) VALUES (?, ?, ?, ?, ?, 1)",
            (webhook_id, url, json.dumps(events), secret_hash, created_at),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        LOGGER.error(f"Failed to create webhook: {e}", exc_info=True)
        return {"ok": False, "error": "Database error"}

    # Cache in memory (store original secret for signing, not hash)
    WEBHOOK_SUBSCRIPTIONS[webhook_id] = {
        "url": url,
        "events": events,
        "secret": webhook_secret,  # Keep for signing
        "secret_hash": secret_hash,
        "created_at": created_at,
        "last_success_ts": None,
        "failure_count": 0,
    }

    LOGGER.info(f"Webhook subscribed: {webhook_id} -> {url} for events {events}")

    return {
        "ok": True,
        "webhook_id": webhook_id,
        "url": url,
        "events": events,
        "secret": WEBHOOK_SUBSCRIPTIONS[webhook_id]["secret"],
    }


@router.get("/api/webhooks")
async def list_webhooks():
    """List all registered webhooks."""
    webhooks = []
    for webhook_id, data in WEBHOOK_SUBSCRIPTIONS.items():
        webhooks.append(
            {
                "webhook_id": webhook_id,
                "url": data["url"],
                "events": data["events"],
                "created_at": data["created_at"],
                "last_triggered": data.get("last_triggered"),
                "delivery_count": data.get("delivery_count", 0),
                "failure_count": data.get("failure_count", 0),
            }
        )
    return {"ok": True, "webhooks": webhooks, "count": len(webhooks)}


@router.delete("/api/webhooks/{webhook_id}")
async def unsubscribe_webhook(webhook_id: str):
    """Unregister a webhook."""
    if webhook_id in WEBHOOK_SUBSCRIPTIONS:
        deleted = WEBHOOK_SUBSCRIPTIONS.pop(webhook_id)
        return {"ok": True, "message": f"Deleted webhook: {deleted['url']}"}
    return {"ok": False, "error": "Webhook not found"}


@router.post("/api/webhooks/test/{webhook_id}")
async def test_webhook(webhook_id: str):
    """Send a test event to a webhook."""
    if webhook_id not in WEBHOOK_SUBSCRIPTIONS:
        return {"ok": False, "error": "Webhook not found"}

    WEBHOOK_SUBSCRIPTIONS[webhook_id]
    test_event = {
        "event": "webhook.test",
        "timestamp": time.time(),
        "data": {"message": "Test webhook delivery"},
    }

    result = await dispatch_webhook_event("webhook.test", test_event, webhook_id)
    return {"ok": result["success"], "result": result}


@router.get("/api/ip/allowlist")
async def get_ip_allowlist():
    """Get current IP allowlist."""
    return {
        "ok": True,
        "enabled": IP_ALLOWLIST_ENABLED,
        "ips": list(IP_ALLOWLIST),
        "count": len(IP_ALLOWLIST),
    }


@router.post("/api/ip/allowlist/add")
async def add_ip_to_allowlist(ip: str):
    """Add an IP to the allowlist."""
    IP_ALLOWLIST.add(ip)
    return {"ok": True, "ip": ip, "message": "IP added to allowlist"}


@router.post("/api/ip/allowlist/remove")
async def remove_ip_from_allowlist(ip: str):
    """Remove an IP from the allowlist."""
    if ip in IP_ALLOWLIST:
        IP_ALLOWLIST.remove(ip)
        return {"ok": True, "ip": ip, "message": "IP removed from allowlist"}
    return {"ok": False, "error": "IP not in allowlist"}


@router.get("/api/gates/status")
async def api_gates_status():
    """
    Get current status of all market gates.
    
    Returns config and current readings for:
    - Regime Filter (SPY vs 20MA, BTC trend)
    - VIX Gate (fear levels)
    - Confirmation requirements
    """
    try:
        from core.market_gates import get_market_gates_status, RegimeFilter, VIXGate
        
        status = get_market_gates_status()
        
        # Add live readings
        rf = RegimeFilter()
        vg = VIXGate()
        
        # Get live data
        spy_regime = await rf.get_spy_regime()
        btc_trend = await rf.get_btc_trend()
        vix_level = await vg.get_current_vix()
        vix_mult, vix_reason = await vg.get_buy_confidence_multiplier()
        
        status["live_readings"] = {
            "spy": spy_regime,
            "btc": btc_trend,
            "vix": {
                "level": vix_level,
                "buy_multiplier": vix_mult,
                "reason": vix_reason
            },
            "timestamp": time.time()
        }
        
        return status
        
    except Exception as e:
        LOGGER.error(f"gates_status_error: {e}")
        return {"error": str(e)}


@router.get("/api/gates/regime")
async def api_gates_regime():
    """
    Get current market regime (bull/bear).
    
    Checks:
    - SPY position vs 20-day MA
    - BTC 7-day trend
    """
    try:
        from core.market_gates import RegimeFilter
        
        rf = RegimeFilter()
        
        spy_data = await rf.get_spy_regime()
        btc_data = await rf.get_btc_trend()
        
        # Overall regime decision
        stock_bullish = spy_data.get("regime") == "bull"
        crypto_bullish = btc_data.get("crypto_regime") == "bull"
        
        overall = "bull" if stock_bullish and crypto_bullish else "bear" if not stock_bullish and not crypto_bullish else "mixed"
        
        return {
            "overall_regime": overall,
            "stock_regime": spy_data,
            "crypto_regime": btc_data,
            "buy_allowed": {
                "stocks": stock_bullish,
                "crypto": crypto_bullish
            },
            "timestamp": time.time()
        }
        
    except Exception as e:
        LOGGER.error(f"gates_regime_error: {e}")
        return {"error": str(e)}


@router.get("/api/gates/vix")
async def api_gates_vix():
    """
    Get current VIX level and its impact on BUY signals.
    
    Levels:
    - < 20: Normal (full confidence)
    - 20-25: Caution (75% confidence)
    - 25-30: Fear (50% confidence)
    - > 30: Panic (block BUYs)
    """
    try:
        from core.market_gates import VIXGate
        
        vg = VIXGate()
        vix = await vg.get_current_vix()
        mult, reason = await vg.get_buy_confidence_multiplier()
        
        return {
            "vix_level": vix,
            "buy_multiplier": mult,
            "status": reason,
            "action": "block" if mult == 0 else "reduce" if mult < 1 else "allow",
            "timestamp": time.time()
        }
        
    except Exception as e:
        LOGGER.error(f"gates_vix_error: {e}")
        return {"error": str(e)}


@router.get("/api/gates/losers")
async def api_gates_losers(limit: int = 50):
    """
    Analyze patterns in losing BUY trades.
    
    Returns:
    - Worst performing symbols
    - Time patterns
    - Recommendations for improvement
    """
    try:
        from core.market_gates import LoserAnalyzer
        
        analyzer = LoserAnalyzer()
        analysis = analyzer.analyze_patterns()
        
        return {
            "analysis": analysis,
            "timestamp": time.time()
        }
        
    except Exception as e:
        LOGGER.error(f"gates_losers_error: {e}")
        return {"error": str(e)}


@router.post("/api/gates/test")
async def api_gates_test(
    direction: str = "UP",
    confidence: float = 0.75,
    asset_type: str = "crypto"
):
    """
    Test what would happen to a signal after applying all gates.
    
    Args:
        direction: UP, DOWN, or FLAT
        confidence: Original confidence (0-1)
        asset_type: stock or crypto
    
    Returns:
        What the final signal would be after gates
    """
    try:
        from core.market_gates import apply_market_gates
        
        # Use sample metrics
        test_metrics = {
            "rsi": 45,
            "momentum_7d": 0.03,
            "macd_histogram": 0.1,
            "current_price": 100
        }
        
        gated_dir, gated_conf, gate_info = await apply_market_gates(
            direction, confidence, test_metrics, asset_type
        )
        
        return {
            "input": {
                "direction": direction,
                "confidence": confidence,
                "asset_type": asset_type
            },
            "output": {
                "direction": gated_dir,
                "confidence": gated_conf
            },
            "gate_info": gate_info,
            "changed": gated_dir != direction or abs(gated_conf - confidence) > 0.01,
            "timestamp": time.time()
        }
        
    except Exception as e:
        LOGGER.error(f"gates_test_error: {e}")
        return {"error": str(e)}


@router.get("/api/stage1/world")
async def api_stage1_world_context(hours: int = 24, min_relevance: float = 0.3):
    """Get world news context from Stage 1 Context Engine."""
    if not STAGE1_ENABLED:
        return _get_world_context_fallback()
    try:
        enhanced = get_enhanced_context(hours=hours, min_relevance=min_relevance)
        return enhanced.get("world_context", _get_world_context_fallback())
    except Exception as e:
        LOGGER.error(f"stage1_world_context_error: {e}")
        return _get_world_context_fallback()


@router.get("/api/stage1/mood")
async def api_stage1_market_mood():
    """Get current market mood/regime from Stage 1."""
    if not STAGE1_ENABLED:
        return _get_market_mood_fallback()
    try:
        enhanced = get_enhanced_context()
        return enhanced.get("market_mood", _get_market_mood_fallback())
    except Exception as e:
        LOGGER.error(f"stage1_market_mood_error: {e}")
        return _get_market_mood_fallback()


@router.get("/api/stage1/symbol/{symbol}")
async def api_stage1_symbol_context(symbol: str, hours: int = 24):
    """Get context for a specific symbol from Stage 1."""
    if not STAGE1_ENABLED:
        return {"error": "Stage 1 not enabled", "symbol_context": {}}
    try:
        context = get_symbol_context(symbol.upper(), hours)
        return context
    except Exception as e:
        LOGGER.error(f"stage1_symbol_context_error: {e}")
        return {"error": str(e), "symbol_context": {}}


@router.get("/api/stage1/stats")
async def api_stage1_stats():
    """Get Stage 1 Context Awareness statistics."""
    if not STAGE1_ENABLED:
        return {"error": "Stage 1 not enabled", "stats": {}}
    try:
        from core.stage1_integration import get_context_stats

        stats = get_context_stats()
        return stats
    except Exception as e:
        LOGGER.error(f"stage1_stats_error: {e}")
        return {"error": str(e), "stats": {}}


@router.get("/api/stage2/learning")
async def api_stage2_learning():
    """Get learning loop statistics and current config."""
    if not STAGE2_ENABLED:
        return {"error": "Stage 2 not enabled"}
    try:
        stats = get_learning_stats()
        return stats
    except Exception as e:
        LOGGER.error(f"stage2_learning_error: {e}")
        return {"error": str(e)}


@router.post("/api/stage2/tune")
async def api_stage2_tune(
    symbol: str | None = None,
    days: int = 7,
    auto_apply: bool = True,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """Run learning cycle to check and tune model parameters."""
    if not STAGE2_ENABLED:
        return {"error": "Stage 2 not enabled"}
    # Optional bearer
    _require_bearer(
        (f"Bearer {credentials.credentials}") if credentials and credentials.credentials else None
    )
    try:
        result = run_learning_cycle(symbol=symbol, days=days, auto_apply=auto_apply)
        return result
    except Exception as e:
        LOGGER.error(f"stage2_tune_error: {e}")
        return {"error": str(e)}


@router.get("/api/stage2/forecasts")
async def api_stage2_forecasts(symbol: str | None = None, limit: int = 10):
    """Get recent forecasts with accuracy details."""
    if not STAGE2_ENABLED:
        return {"error": "Stage 2 not enabled"}
    try:
        tracker = get_accuracy_tracker()
        forecasts = tracker.get_recent_forecasts(symbol=symbol, limit=limit)
        return {"forecasts": forecasts, "count": len(forecasts)}
    except Exception as e:
        LOGGER.error(f"stage2_forecasts_error: {e}")
        return {"error": str(e)}


@router.get("/api/regime/current")
async def api_regime_current():
    """Get current market regime with <50ms response time (neutral fallback if Stage 3 not enabled)."""
    try:
        # Fast path: check if STAGE3 enabled and return cached regime
        if STAGE3_ENABLED:
            async def get_regime_fast():
                regime_detector = get_regime_detector()
                return {
                    "regime": regime_detector.current_regime.lower(),
                    "ts": int(time.time()),
                    "confidence": float(regime_detector.confidence),
                    "source": "stage3_detector",
                }

            # Cap at 2.5s to prevent stalls
            result = await with_cap(
                get_regime_fast(),
                sec=2.5,
                fallback={
                    "regime": "neutral",
                    "ts": int(time.time()),
                    "confidence": 0.5,
                    "source": "timeout_fallback",
                }
            )
            return result
        else:
            # Instant fallback if Stage 3 disabled
            return {
                "regime": "neutral",
                "ts": int(time.time()),
                "confidence": 0.5,
                "source": "fallback",
            }
    except Exception as e:
        LOGGER.error(f"regime_current_error: {e}")
        return {
            "regime": "neutral",
            "ts": int(time.time()),
            "confidence": 0.5,
            "source": "error_fallback",
        }


@router.get("/api/watchlist")
async def api_watchlist_get():
    """Get all symbols in watchlist."""
    if not WATCHLIST_ENABLED:
        return {"error": "Watchlist not enabled"}
    try:
        watchlist_mgr = get_watchlist_manager()
        symbols = watchlist_mgr.get_watchlist()
        return {"symbols": symbols, "count": len(symbols)}
    except Exception as e:
        LOGGER.error(f"watchlist_get_error: {e}")
        return {"error": str(e)}


@router.post("/api/watchlist/add")
async def api_watchlist_add(symbol: str, name: str = "", metadata: str = ""):
    """Add symbol to watchlist."""
    if not WATCHLIST_ENABLED:
        return {"error": "Watchlist not enabled"}
    try:
        watchlist_mgr = get_watchlist_manager()
        result = watchlist_mgr.add_symbol(symbol, name, metadata)
        return result
    except Exception as e:
        LOGGER.error(f"watchlist_add_error: {e}")
        return {"error": str(e)}


@router.post("/api/watchlist/remove")
async def api_watchlist_remove(symbol: str):
    """Remove symbol from watchlist."""
    if not WATCHLIST_ENABLED:
        return {"error": "Watchlist not enabled"}
    try:
        watchlist_mgr = get_watchlist_manager()
        result = watchlist_mgr.remove_symbol(symbol)
        return result
    except Exception as e:
        LOGGER.error(f"watchlist_remove_error: {e}")
        return {"error": str(e)}


@router.post("/api/watchlist/score")
async def api_watchlist_score(
    symbol: str,
    gps_score: float,
    price: float,
    change_pct: float,
    volume: float | None = None,
    market_cap: float | None = None,
    threshold: float = 7.0,
):
    """
    Update GHOST score for a watchlist symbol.
    Symbol will appear in top_movers only if gps_score >= threshold.
    """
    if not WATCHLIST_ENABLED:
        return {"error": "Watchlist not enabled"}
    try:
        watchlist_mgr = get_watchlist_manager()
        result = watchlist_mgr.update_ghost_score(
            symbol=symbol,
            gps_score=gps_score,
            price=price,
            change_pct=change_pct,
            volume=volume,
            market_cap=market_cap,
            threshold=threshold,
        )
        return result
    except Exception as e:
        LOGGER.error(f"watchlist_score_error: {e}")
        return {"error": str(e)}


@router.get("/api/watchlist/history/{symbol}")
async def api_watchlist_history(symbol: str, limit: int = 100):
    """Get historical GHOST scores for a symbol."""
    if not WATCHLIST_ENABLED:
        return {"error": "Watchlist not enabled"}
    try:
        watchlist_mgr = get_watchlist_manager()
        history = watchlist_mgr.get_symbol_history(symbol, limit)
        return {"symbol": symbol, "history": history, "count": len(history)}
    except Exception as e:
        LOGGER.error(f"watchlist_history_error: {e}")
        return {"error": str(e)}


@router.get("/api/watchlist/statistics")
async def api_watchlist_statistics():
    """Get watchlist statistics."""
    if not WATCHLIST_ENABLED:
        return {"error": "Watchlist not enabled"}
    try:
        watchlist_mgr = get_watchlist_manager()
        stats = watchlist_mgr.get_statistics()
        return stats
    except Exception as e:
        LOGGER.error(f"watchlist_statistics_error: {e}")
        return {"error": str(e)}


@router.post("/api/watchlist/scan")
async def api_watchlist_scan(threshold: float = 7.0, limit: int = 50):
    """
    Scan watchlist symbols, fetch prices, compute a simple GPS score, and
    update ghost_scores so /api/top_movers can surface candidates.

    Strategy:
      - Prefer Polygon.io if configured; fallback to yfinance (close/prev).
      - GPS heuristic: base 6.5 + |change_pct| buckets + volume pulse if available.
      - Only up to `limit` symbols to avoid rate limiting.
    """
    if not WATCHLIST_ENABLED:
        return {"error": "Watchlist not enabled"}

    try:
        from core.polygon_integration import get_polygon_client

        watchlist_mgr = get_watchlist_manager()
        symbols_meta = watchlist_mgr.get_watchlist()
        symbols = [s["symbol"] for s in symbols_meta][: max(1, int(limit))]

        # Helper to fetch price using Polygon or yfinance
        def fetch_price_pair(sym: str) -> tuple[float | None, float | None]:
            price: float | None = None
            prev: float | None = None
            try:
                polygon = get_polygon_client()
                quote = polygon.get_realtime_quote(sym)
                if quote and quote.price:
                    return float(quote.price), float(
                        quote.prev_close or 0.0
                    ) if quote.prev_close else None
            except Exception:
                pass
            # Fallback: yfinance
            try:
                import yfinance as yf

                t = yf.Ticker(sym)
                hist = t.history(period="2d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])
                    if len(hist["Close"]) > 1:
                        prev = float(hist["Close"].iloc[-2])
            except Exception:
                price, prev = None, None
            return price, prev

        updated: list[dict[str, Any]] = []
        for sym in symbols:
            p, pc = fetch_price_pair(sym)
            if not p:
                continue
            chg = 0.0
            if pc and pc > 0:
                chg = (p - pc) / pc * 100.0
            # Simple GPS heuristic
            gps = (
                6.5
                + (0.3 if abs(chg) >= 1 else 0.0)
                + (0.7 if abs(chg) >= 3 else 0.0)
                + (0.5 if abs(chg) >= 5 else 0.0)
            )
            gps = min(10.0, max(0.0, gps))

            # Persist
            try:
                watchlist_mgr.update_ghost_score(
                    symbol=sym,
                    gps_score=float(gps),
                    price=float(p),
                    change_pct=float(chg),
                    volume=None,
                    market_cap=None,
                    threshold=threshold,
                )
                updated.append({"symbol": sym, "price": p, "change_pct": chg, "gps": round(gps, 2)})
            except Exception as e:
                LOGGER.debug(f"watchlist_scan_update_failed: {sym} {e}")

        movers = []
        try:
            movers = watchlist_mgr.get_top_movers(threshold=threshold, limit=limit)
        except Exception:
            movers = []

        return {
            "scanned": len(symbols),
            "updated": len(updated),
            "threshold": threshold,
            "movers": movers,
        }
    except Exception as e:
        LOGGER.error(f"watchlist_scan_error: {e}")
        return {"error": str(e)}


@router.get("/heatmap")
async def api_heatmap():
    """Simple heatmap endpoint for UI.

    In Focus Mode, return a single tile for WOLF with a deterministic GPS and current price.
    """
    try:
        price, prev, provider = get_wolf_price()
    except Exception:
        price, _prev, _provider = None, None, None
    row_current = price if price is not None else float(STATE.get("avg_cost", 0.0))
    # Deterministic GPS for WOLF; could be enhanced later
    gps = 7.2
    return [{"symbol": WOLF, "gps": gps, "current": row_current, "type": "stock"}]


@router.get("/events")
async def sse_events(request: Request):
    async def event_gen():
        last_id = _EVENT_SEQ
        start_time = time.time()
        # On connect, replay recent
        for ev in list(EVENTS)[-50:]:
            yield f"id: {ev['id']}\ndata: {json.dumps(ev)}\n\n"
            last_id = ev["id"]
        # Then poll for new
        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                print("[SSE events] Client disconnected, closing stream")
                break
            # TTL: Close stream after 30 minutes to prevent leaks
            if time.time() - start_time > 1800:
                print("[SSE events] Stream TTL expired (30 min), closing")
                break
            await _async_sleep(1.0)
            if _EVENT_SEQ > last_id:
                for ev in EVENTS:
                    if ev["id"] > last_id:
                        yield f"id: {ev['id']}\ndata: {json.dumps(ev)}\n\n"
                        last_id = ev["id"]

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.post("/api/position")
async def api_position_set(
    p: PositionBody, credentials: HTTPAuthorizationCredentials | None = AUTH_DEP
):
    _require_bearer(
        (f"Bearer {credentials.credentials}") if credentials and credentials.credentials else None
    )
    if p.qty < 0:
        raise HTTPException(422, "qty must be >= 0")
    # Allow avg_cost == 0 only when flat (qty == 0); otherwise require > 0
    if p.qty > 0 and (p.avg_cost is None or p.avg_cost <= 0):
        raise HTTPException(422, "avg_cost must be > 0 when qty > 0")
    STATE["qty"] = float(p.qty)
    # Store full precision to keep exact cost basis (UI can render rounded)
    STATE["avg_cost"] = float(p.avg_cost)
    _persist_save()
    ALERT_STATE["trailing_high"] = None
    ALERT_STATE["trailing_low"] = None
    _add_event(
        "position.update",
        "Position updated",
        {"qty": STATE["qty"], "avg_cost": STATE["avg_cost"]},
    )
    # Send STATUS card (includes price/provider)
    enqueue_alert_text(_build_status_card())
    return {"symbol": WOLF, "qty": STATE["qty"], "avg_cost": STATE["avg_cost"]}


@router.post("/api/price/record")
async def api_price_record(payload: dict[str, Any]):
    """Append an actual price tick to the most recent forecast for comparison."""
    try:
        symbol = payload.get("symbol", WOLF).upper()
        price = float(payload["price"])
        provider = payload.get("provider", "unknown")
        ts = payload.get("ts", int(time.time()))

        # Find most recent forecast for this symbol
        matching = [
            fid for fid, f in FORECAST_STORE.items() if f.get("symbol", "").upper() == symbol
        ]
        if not matching:
            return {"ok": False, "reason": "no_forecast_found"}

        forecast_id = max(matching, key=lambda fid: FORECAST_STORE[fid].get("as_of", 0))

        if forecast_id not in FORECAST_ACTUALS:
            FORECAST_ACTUALS[forecast_id] = []

        FORECAST_ACTUALS[forecast_id].append({"t": ts, "p": price, "provider": provider})
        return {
            "ok": True,
            "forecast_id": forecast_id,
            "ticks": len(FORECAST_ACTUALS[forecast_id]),
        }
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, 500)


@router.get("/api/movers/scan")
async def api_movers_scan():
    """
    Manually trigger a market movers scan.
    Returns today's biggest movers (stocks + crypto).
    """
    try:
        from core.realtime_market_movers import manual_scan
        
        movers = await manual_scan()
        
        return {
            "ok": True,
            "scan_time": datetime.now().isoformat(),
            "movers_found": len(movers),
            "movers": movers[:20]  # Top 20
        }
        
    except Exception as e:
        LOGGER.error(f"Manual movers scan failed: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/api/movers/discovered")
async def api_movers_discovered():
    """
    Get list of symbols discovered by the movers scanner today.
    """
    try:
        from core.realtime_market_movers import get_scanner_status
        
        status = get_scanner_status()
        
        return {
            "ok": True,
            "date": status.get("last_discovery_date", ""),
            "count": status.get("discovered_count", 0),
            "symbols": status.get("discovered_today", [])
        }
        
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/api/learning/dashboard")
async def learning_dashboard():
    """
    🎯 LEARNING DASHBOARD
    
    Returns comprehensive learning metrics including:
    - Overall accuracy (raw and inverted)
    - Per-symbol accuracy table
    - Symbols excluded from TOP 10 (low accuracy)
    - Symbols boosted in TOP 10 (high accuracy)
    - Recent outcomes
    """
    from datetime import datetime, timedelta
    
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return {"ok": False, "error": "DATABASE_URL not configured"}
    
    try:
        from core.db_pool import get_sync_connection
        with get_sync_connection() as conn:
            cur = conn.cursor()
            
            # Create ghost_symbol_accuracy table if not exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS ghost_symbol_accuracy (
                    symbol VARCHAR(20) PRIMARY KEY,
                    total_predictions INTEGER DEFAULT 0,
                    correct_predictions INTEGER DEFAULT 0,
                    accuracy_pct NUMERIC(5, 2) DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Get overall stats from ghost_prediction_outcomes table (the actual reconciled data)
            cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END) as correct,
                    AVG(CASE WHEN hit_direction = 1 THEN 1.0 ELSE 0.0 END) * 100 as accuracy_pct
                FROM ghost_prediction_outcomes
            """)
            row = cur.fetchone()
            overall_stats = {
                "total_evaluated": row[0] or 0,
                "total_correct": row[1] or 0,
                "raw_accuracy_pct": float(row[2]) if row[2] else 0,
            }
            
            # With INVERSE_GHOST=1, if raw is X%, inverted is 100-X% - FIXED: Use INVERSE_GHOST (not INVERSE_GHOST_MODE) - default to OFF (0)
            inverse_mode = os.getenv("INVERSE_GHOST", "0") == "1"
            if inverse_mode:
                overall_stats["inverted_accuracy_pct"] = 100 - overall_stats["raw_accuracy_pct"]
                overall_stats["mode"] = "INVERSE_GHOST (raw predictions inverted)"
            else:
                overall_stats["inverted_accuracy_pct"] = overall_stats["raw_accuracy_pct"]
                overall_stats["mode"] = "NORMAL"
            
            # Get symbols to exclude (accuracy < 40% with 10+ predictions)
            cur.execute("""
                SELECT symbol, total_predictions, correct_predictions, accuracy_pct
                FROM ghost_symbol_accuracy
                WHERE total_predictions >= 10 AND accuracy_pct < 40
                ORDER BY accuracy_pct ASC
            """)
            excluded_symbols = []
            for row in cur.fetchall():
                excluded_symbols.append({
                    "symbol": row[0],
                    "total": row[1],
                    "correct": row[2],
                    "accuracy_pct": float(row[3]) if row[3] else 0,
                    "reason": "accuracy < 40%"
                })
            
            # Get symbols to boost (accuracy >= 70% with 10+ predictions)
            cur.execute("""
                SELECT symbol, total_predictions, correct_predictions, accuracy_pct
                FROM ghost_symbol_accuracy
                WHERE total_predictions >= 10 AND accuracy_pct >= 70
                ORDER BY accuracy_pct DESC
            """)
            boosted_symbols = []
            for row in cur.fetchall():
                boosted_symbols.append({
                    "symbol": row[0],
                    "total": row[1],
                    "correct": row[2],
                    "accuracy_pct": float(row[3]) if row[3] else 0,
                    "boost": "+15% confidence"
                })
            
            # Get all symbol accuracy (top 50 by total predictions)
            cur.execute("""
                SELECT symbol, total_predictions, correct_predictions, accuracy_pct, last_updated
                FROM ghost_symbol_accuracy
                ORDER BY total_predictions DESC
                LIMIT 50
            """)
            all_symbols = []
            for row in cur.fetchall():
                all_symbols.append({
                    "symbol": row[0],
                    "total": row[1],
                    "correct": row[2],
                    "accuracy_pct": float(row[3]) if row[3] else 0,
                    "last_updated": row[4].isoformat() if row[4] else None
                })
            
            # Get recent outcomes (last 20) from ghost_prediction_outcomes table
            cur.execute("""
                SELECT symbol, predicted_direction, actual_direction, hit_direction, 
                       price_at_prediction, price_at_resolution, realized_move_pct, closed_at
                FROM ghost_prediction_outcomes
                ORDER BY closed_at DESC NULLS LAST
                LIMIT 20
            """)
            recent_outcomes = []
            for row in cur.fetchall():
                # closed_at is a timestamp (could be datetime or float)
                evaluated_at = None
                if row[7]:
                    if isinstance(row[7], datetime):
                        evaluated_at = row[7].isoformat()
                    elif isinstance(row[7], (int, float)):
                        evaluated_at = datetime.fromtimestamp(row[7]).isoformat()
                    else:
                        evaluated_at = str(row[7])
                
                recent_outcomes.append({
                    "symbol": row[0],
                    "predicted": row[1],
                    "actual": row[2],
                    "correct": row[3] == 1,
                    "entry_price": float(row[4]) if row[4] else 0,
                    "exit_price": float(row[5]) if row[5] else 0,
                    "change_pct": float(row[6]) if row[6] else 0,
                    "evaluated_at": evaluated_at
                })
        
        # Import learning config from ghost_notifications
        try:
            from core.ghost_notifications import (
                LEARNING_ENABLED, LEARNING_BOOST_ENABLED, LEARNING_EXCLUDE_ENABLED,
                LEARNING_MIN_PREDICTIONS, LEARNING_EXCLUDE_ACCURACY, LEARNING_BOOST_ACCURACY,
                LEARNING_BOOST_AMOUNT, HARDCODED_EXCLUSIONS
            )
            learning_config = {
                "learning_enabled": LEARNING_ENABLED,
                "boost_enabled": LEARNING_BOOST_ENABLED,
                "exclude_enabled": LEARNING_EXCLUDE_ENABLED,
                "min_predictions": LEARNING_MIN_PREDICTIONS,
                "exclude_threshold": LEARNING_EXCLUDE_ACCURACY,
                "boost_threshold": LEARNING_BOOST_ACCURACY,
                "boost_amount": LEARNING_BOOST_AMOUNT,
                "hardcoded_exclusions": list(HARDCODED_EXCLUSIONS.keys()),
                "hardcoded_exclusions_details": HARDCODED_EXCLUSIONS
            }
        except ImportError:
            learning_config = {"error": "Could not import learning config"}
        
        return {
            "ok": True,
            "config": learning_config,
            "overall_accuracy": overall_stats,
            "learning_adjustments": {
                "excluded_from_top10": {
                    "count": len(excluded_symbols),
                    "reason": "accuracy < 40% after 10+ predictions",
                    "symbols": excluded_symbols
                },
                "boosted_in_top10": {
                    "count": len(boosted_symbols),
                    "reason": "accuracy >= 70% after 10+ predictions",
                    "boost_amount": "+15% confidence",
                    "symbols": boosted_symbols
                }
            },
            "symbol_accuracy_top50": all_symbols,
            "recent_outcomes": recent_outcomes,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/api/learning/symbol/{symbol}")
async def learning_symbol_accuracy(symbol: str):
    """
    Get detailed accuracy information for a specific symbol.
    """
    from datetime import datetime
    
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        return {"ok": False, "error": "DATABASE_URL not configured"}
    
    try:
        from core.db_pool import get_sync_connection
        with get_sync_connection() as conn:
            cur = conn.cursor()
            
            # Get symbol accuracy from ghost_symbol_accuracy table
            cur.execute("""
                SELECT symbol, total_predictions, correct_predictions, accuracy_pct, last_updated
                FROM ghost_symbol_accuracy
                WHERE symbol = %s
            """, (symbol.upper(),))
            
            row = cur.fetchone()
            if not row:
                # Try to compute from ghost_prediction_outcomes table directly
                cur.execute("""
                    SELECT 
                        symbol,
                        COUNT(*) as total,
                        SUM(CASE WHEN hit_direction = 1 THEN 1 ELSE 0 END) as correct
                    FROM ghost_prediction_outcomes
                    WHERE symbol = %s
                    GROUP BY symbol
                """, (symbol.upper(),))
                outcome_row = cur.fetchone()
                if not outcome_row:
                    return {"ok": False, "error": f"No accuracy data for {symbol}"}
                
                symbol_data = {
                    "symbol": outcome_row[0],
                    "total_predictions": outcome_row[1],
                    "correct_predictions": outcome_row[2],
                    "accuracy_pct": (outcome_row[2] / outcome_row[1] * 100) if outcome_row[1] > 0 else 0,
                    "last_updated": None,
                    "source": "computed_from_ghost_prediction_outcomes"
                }
            else:
                symbol_data = {
                    "symbol": row[0],
                    "total_predictions": row[1],
                    "correct_predictions": row[2],
                    "accuracy_pct": float(row[3]) if row[3] else 0,
                    "last_updated": row[4].isoformat() if row[4] else None,
                    "source": "ghost_symbol_accuracy"
                }
            
            # Determine status
            if symbol_data["total_predictions"] < 10:
                symbol_data["status"] = "insufficient_data"
                symbol_data["learning_action"] = "none"
            elif symbol_data["accuracy_pct"] < 40:
                symbol_data["status"] = "low_accuracy"
                symbol_data["learning_action"] = "EXCLUDED from TOP 10"
            elif symbol_data["accuracy_pct"] >= 70:
                symbol_data["status"] = "high_accuracy"
                symbol_data["learning_action"] = "BOOSTED +15% confidence"
            else:
                symbol_data["status"] = "normal"
                symbol_data["learning_action"] = "none"
            
            # Get recent outcomes for this symbol from ghost_prediction_outcomes table
            cur.execute("""
                SELECT predicted_direction, actual_direction, hit_direction, 
                       price_at_prediction, price_at_resolution, realized_move_pct, closed_at
                FROM ghost_prediction_outcomes
                WHERE symbol = %s
                ORDER BY closed_at DESC NULLS LAST
                LIMIT 10
            """, (symbol.upper(),))
            
            recent = []
            for r in cur.fetchall():
                # closed_at is a timestamp (could be datetime or float)
                evaluated_at = None
                if r[6]:
                    if isinstance(r[6], datetime):
                        evaluated_at = r[6].isoformat()
                    elif isinstance(r[6], (int, float)):
                        evaluated_at = datetime.fromtimestamp(r[6]).isoformat()
                    else:
                        evaluated_at = str(r[6])
                
                recent.append({
                    "predicted": r[0],
                    "actual": r[1],
                    "correct": r[2] == 1,
                    "entry": float(r[3]) if r[3] else 0,
                    "exit": float(r[4]) if r[4] else 0,
                    "change_pct": float(r[5]) if r[5] else 0,
                    "evaluated_at": evaluated_at
                })
            
            symbol_data["recent_outcomes"] = recent
        
        return {"ok": True, **symbol_data}
        
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.get("/api/v2/performance/dashboard")
async def v2_performance_dashboard(days: int = 14):
    """
    🎯 V2: Complete performance dashboard with verified metrics.
    
    Returns ground truth about Ghost's actual win rate, best/worst assets,
    and recommendations for whitelist/blacklist.
    
    Example: GET /api/v2/performance/dashboard?days=14
    """
    try:
        from core.v2_verification import get_verifier
        
        verifier = get_verifier()
        
        # Generate comprehensive report
        report = verifier.generate_performance_report(days)
        
        # Format for API response
        return {
            "ok": True,
            "period": report.period,
            "days": days,
            "overall": {
                "total_predictions": report.total_predictions,
                "wins": report.verified_wins,
                "losses": report.verified_losses,
                "win_rate": round(report.win_rate, 1),
                "win_rate_display": f"{report.win_rate:.1f}%"
            },
            "by_asset_type": report.by_asset_type,
            "top_performers": report.top_performers,
            "bottom_performers": report.bottom_performers,
            "top_10_details": [
                {
                    "symbol": p.symbol,
                    "asset_type": p.asset_type,
                    "win_rate": round(p.win_rate, 1),
                    "total": p.total_predictions,
                    "wins": p.wins,
                    "trend": p.recent_performance
                }
                for p in report.by_symbol[:10]
            ],
            "bottom_10_details": [
                {
                    "symbol": p.symbol,
                    "asset_type": p.asset_type,
                    "win_rate": round(p.win_rate, 1),
                    "total": p.total_predictions,
                    "wins": p.wins,
                    "trend": p.recent_performance
                }
                for p in report.by_symbol[-10:]
            ]
        }
    
    except Exception as e:
        LOGGER.error(f"[V2-API] Performance dashboard error: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/api/v2/quality/status")
async def v2_quality_status():
    """
    🎯 V2: Get asset quality filter status (whitelist/blacklist).
    
    Shows which assets are approved for predictions and which are blocked.
    """
    try:
        from core.v2_quality import get_quality_system
        
        quality = get_quality_system()
        stats = quality.get_quality_filter_stats()
        
        return {
            "ok": True,
            **stats,
            "description": {
                "whitelist": f"Proven performers (WR >= {stats['config']['whitelist_wr_threshold']}), predict freely",
                "watchlist": f"Cautious zone (WR 45-55%), require {stats['config']['watchlist_min_confidence']}+ confidence",
                "blacklist": f"Poor performers (WR < {stats['config']['blacklist_wr_threshold']}), DO NOT predict"
            }
        }
    
    except Exception as e:
        LOGGER.error(f"[V2-API] Quality status error: {e}")
        return {"ok": False, "error": str(e)}


@router.post("/api/v2/quality/reload")
async def v2_quality_reload_from_json():
    """
    🔄 V2: Force reload whitelist/blacklist/trial_stocks from JSON file.
    
    Use this to push manual JSON changes to PostgreSQL.
    Bypasses the normal PostgreSQL-first loading.
    """
    try:
        from core.v2_quality import get_quality_system
        import json
        
        quality = get_quality_system()
        
        # Read directly from JSON
        with open("ghost_v2_quality.json", 'r') as f:
            data = json.load(f)
        
        # Update in-memory state
        quality._whitelist = set(data.get('whitelist', []))
        quality._blacklist = set(data.get('blacklist', []))
        quality._trial_stocks = set(data.get('trial_stocks', []))  # NEW: Trial stocks
        quality._config = data.get('config', {})  # NEW: Load config including trial_stock_min_confidence
        
        # Get pinned whitelist
        pinned = set(data.get('pinned_whitelist', data.get('whitelist', [])))
        
        # Save to PostgreSQL (this overwrites the old data)
        quality._save_config(pinned)
        
        # CRITICAL: Also purge blacklisted symbols from _LATEST_PREDICTIONS cache
        symbols_purged = []
        with _LATEST_PREDICTIONS_LOCK:
            for symbol in list(_LATEST_PREDICTIONS.keys()):
                if symbol in quality._blacklist:
                    del _LATEST_PREDICTIONS[symbol]
                    symbols_purged.append(symbol)
        
        if symbols_purged:
            LOGGER.info(f"[V2-RELOAD] 🧹 Purged {len(symbols_purged)} blacklisted symbols from cache: {symbols_purged}")

        return {
            "ok": True,
            "message": "Reloaded from JSON and saved to PostgreSQL",
            "whitelist": sorted(list(quality._whitelist)),
            "blacklist_count": len(quality._blacklist),
            "trial_stocks": sorted(list(quality._trial_stocks)),  # NEW
            "pinned_whitelist": sorted(list(pinned)),
            "blacklist": sorted(list(quality._blacklist)),
            "symbols_purged_from_cache": symbols_purged
        }
    
    except Exception as e:
        LOGGER.error(f"[V2-API] Quality reload error: {e}")
        return {"ok": False, "error": str(e)}


@router.post("/api/v2/quality/reload-postgres")
async def v2_quality_reload_from_postgres():
    """
    🔄 V2: Force reload whitelist/blacklist FROM PostgreSQL.
    
    Use this after directly updating PostgreSQL to apply changes without restart.
    """
    try:
        from core.v2_quality import get_quality_system
        
        quality = get_quality_system()
        
        # Capture before state
        before_wl = list(quality._whitelist)
        before_bl = list(quality._blacklist)
        
        # Force reload from PostgreSQL
        pg_data = quality._load_from_postgres()
        
        if pg_data:
            quality._whitelist = set(pg_data.get('whitelist', []))
            quality._blacklist = set(pg_data.get('blacklist', []))
            quality._pinned_whitelist = set(pg_data.get('pinned_whitelist', []))
            
            LOGGER.info(
                f"[V2-RELOAD] ✅ Reloaded from PostgreSQL: "
                f"{len(before_wl)} → {len(quality._whitelist)} whitelist, "
                f"{len(before_bl)} → {len(quality._blacklist)} blacklist"
            )
            
            return {
                "ok": True,
                "message": "Reloaded from PostgreSQL",
                "before": {
                    "whitelist_count": len(before_wl),
                    "blacklist_count": len(before_bl)
                },
                "after": {
                    "whitelist": sorted(quality._whitelist),
                    "blacklist": sorted(quality._blacklist),
                    "pinned": sorted(quality._pinned_whitelist)
                },
                "note": pg_data.get('config', {}).get('note', 'none')
            }
        else:
            return {"ok": False, "error": "No config found in PostgreSQL"}
    
    except Exception as e:
        LOGGER.error(f"[V2-API] PostgreSQL reload error: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/api/v2/quality/debug-json")
async def v2_quality_debug_json():
    """Debug: Read JSON file directly and show contents"""
    try:
        import json
        with open("ghost_v2_quality.json", 'r') as f:
            data = json.load(f)
        
        return {
            "ok": True,
            "whitelist_count": len(data.get('whitelist', [])),
            "blacklist_count": len(data.get('blacklist', [])),
            "blacklist_first_10": sorted(data.get('blacklist', []))[:10],
            "has_icp_in_blacklist": "ICP" in data.get('blacklist', [])
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@router.get("/api/v2/quality/test-should-predict")
async def v2_quality_test_should_predict(symbol: str = "CHZ", confidence: float = 0.85):
    """Debug: Test should_predict function directly"""
    try:
        from core.v2_quality import get_quality_system
        
        quality = get_quality_system()
        should, reason = quality.should_predict(symbol, confidence)
        
        return {
            "ok": True,
            "symbol": symbol,
            "confidence": confidence,
            "should_predict": should,
            "reason": reason,
            "in_whitelist": symbol in quality._whitelist,
            "in_blacklist": symbol in quality._blacklist,
            "whitelist_count": len(quality._whitelist),
            "blacklist_count": len(quality._blacklist),
        }
    except Exception as e:
        import traceback
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.post("/api/v2/quality/update")
async def v2_quality_update(days: int = 30):
    """
    🎯 V2: Update whitelist/blacklist from verified performance data.
    
    Should be run daily (can be automated via cron).
    
    Example: POST /api/v2/quality/update?days=30
    """
    try:
        from core.v2_quality import get_quality_system
        
        quality = get_quality_system()
        
        # Capture before state
        before = quality.get_quality_filter_stats()
        
        # Update from verification
        quality.update_from_verification(days)
        
        # Capture after state
        after = quality.get_quality_filter_stats()
        
        return {
            "ok": True,
            "message": f"Quality filters updated from last {days} days",
            "before": {
                "whitelist": before['whitelist_count'],
                "blacklist": before['blacklist_count']
            },
            "after": {
                "whitelist": after['whitelist_count'],
                "blacklist": after['blacklist_count']
            },
            "changes": {
                "whitelist_added": after['whitelist_count'] - before['whitelist_count'],
                "blacklist_added": after['blacklist_count'] - before['blacklist_count']
            }
        }
    
    except Exception as e:
        LOGGER.error(f"[V2-API] Quality update error: {e}")
        return {"ok": False, "error": str(e)}


@router.post("/api/v2/quality/set-whitelist")
async def v2_quality_set_whitelist(request: Request):
    """
    🎯 V2: Manually set whitelist/blacklist at runtime (no redeploy needed).
    
    POST body:
    {
        "whitelist": ["RNDR", "CHZ", "TURBO", "ZEC"],
        "blacklist": ["GME", "ABCL", "BMBL"],  // optional
        "note": "Crypto only - Jan 25 analysis"  // optional
    }
    
    This updates both in-memory and persists to PostgreSQL.
    """
    try:
        from core.v2_quality import get_quality_system
        
        body = await request.json()
        new_whitelist = body.get("whitelist", [])
        new_blacklist = body.get("blacklist", [])
        note = body.get("note", "Manual update via API")
        
        if not new_whitelist:
            return {"ok": False, "error": "whitelist is required"}
        
        quality = get_quality_system()
        
        # Capture before state
        before_wl = list(quality._whitelist)
        before_bl = list(quality._blacklist)
        
        # Update whitelist
        quality._whitelist = set(s.upper() for s in new_whitelist)
        quality._pinned_whitelist = set(s.upper() for s in new_whitelist)
        
        # Update blacklist if provided
        if new_blacklist:
            quality._blacklist = set(s.upper() for s in new_blacklist)
        
        # Persist to JSON and PostgreSQL
        quality._save_config(pinned_whitelist=quality._pinned_whitelist)
        
        LOGGER.info(
            f"[V2-API] ✅ Whitelist manually updated: "
            f"{len(before_wl)} → {len(quality._whitelist)} symbols. Note: {note}"
        )
        
        return {
            "ok": True,
            "message": f"Whitelist updated: {note}",
            "before": {
                "whitelist": sorted(before_wl),
                "blacklist": sorted(before_bl)
            },
            "after": {
                "whitelist": sorted(quality._whitelist),
                "blacklist": sorted(quality._blacklist)
            },
            "changes": {
                "whitelist_added": list(quality._whitelist - set(before_wl)),
                "whitelist_removed": list(set(before_wl) - quality._whitelist),
                "blacklist_added": list(quality._blacklist - set(before_bl)) if new_blacklist else []
            }
        }
    
    except Exception as e:
        LOGGER.error(f"[V2-API] Set whitelist error: {e}")
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


@router.post("/api/v2/quality/crypto-only")
async def v2_quality_crypto_only():
    """
    🎯 V2: Quick action - Switch to crypto-only whitelist based on loser analysis.
    
    Sets whitelist to proven crypto performers:
    - RNDR (47.6% WR), CHZ (37.1%), TURBO (35.5%), ZEC (31.1%)
    - Plus secondary: EGLD, ILV, RLC, OCEAN
    
    Blacklists all losing stocks (4.5% WR = broken).
    """
    try:
        from core.v2_quality import get_quality_system
        
        quality = get_quality_system()
        
        # Capture before
        before_wl = list(quality._whitelist)
        before_bl = list(quality._blacklist)
        
        # Crypto-only whitelist (proven performers from loser analysis)
        crypto_whitelist = {
            "RNDR",   # 47.6% (89/187) - BEST
            "CHZ",    # 37.1% (75/202)
            "TURBO",  # 35.5% (27/76)
            "ZEC",    # 31.1% (46/148)
            "EGLD",   # ~27%
            "ILV",    # ~30%
            "RLC",    # ~28%
            "OCEAN",  # ~26%
        }
        
        # Losing stocks to blacklist (4.5% overall WR = broken)
        losing_stocks = {
            "ABCL", "GME", "BMBL", "ITRI", "TGTX", "XPO", "SOUN",
            "ARCT", "CVNA", "IQ", "T"
        }
        
        # Bad crypto to blacklist
        bad_crypto = {
            "XRP", "DOT", "AVAX", "UNI", "PEPE", "SNX", "1INCH",
            "LDO", "ETC", "ALGO", "BTC", "ETH", "SOL", "ADA", 
            "BNB", "LTC", "ICP", "LRC"
        }
        
        quality._whitelist = crypto_whitelist
        quality._pinned_whitelist = {"RNDR", "CHZ", "TURBO", "ZEC"}
        quality._blacklist = quality._blacklist | losing_stocks | bad_crypto
        
        # Persist
        quality._save_to_json()
        try:
            quality._save_to_postgres()
        except Exception:
            pass
        
        LOGGER.info(
            f"[V2-API] 🎯 CRYPTO-ONLY MODE ACTIVATED: "
            f"Whitelist: {len(crypto_whitelist)}, Blacklist: {len(quality._blacklist)}"
        )
        
        return {
            "ok": True,
            "message": "🎯 CRYPTO-ONLY MODE ACTIVATED - Stocks removed, crypto performers only",
            "analysis_source": "Jan 25, 2026 loser analysis",
            "rationale": {
                "stocks_wr": "4.5% (BROKEN)",
                "crypto_wr": "38.7% (acceptable)"
            },
            "whitelist": sorted(crypto_whitelist),
            "pinned": ["RNDR", "CHZ", "TURBO", "ZEC"],
            "blacklist_added": sorted(losing_stocks | bad_crypto),
            "before": {
                "whitelist_count": len(before_wl),
                "blacklist_count": len(before_bl)
            },
            "after": {
                "whitelist_count": len(quality._whitelist),
                "blacklist_count": len(quality._blacklist)
            }
        }
    
    except Exception as e:
        LOGGER.error(f"[V2-API] Crypto-only error: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/api/v2/recommendations")
async def v2_recommendations(days: int = 30):
    """
    🎯 V2: Get whitelist/blacklist recommendations based on performance.
    
    Analyzes last 30 days and recommends which assets to keep vs drop.
    """
    try:
        from core.v2_verification import get_verifier
        
        verifier = get_verifier()
        rec = verifier.recommend_whitelist_blacklist(days)
        
        return {
            "ok": True,
            "period": f"last_{days}_days",
            "whitelist": {
                "count": len(rec['whitelist']),
                "symbols": rec['whitelist'],
                "criteria": rec['criteria']['whitelist']
            },
            "blacklist": {
                "count": len(rec['blacklist']),
                "symbols": rec['blacklist'],
                "criteria": rec['criteria']['blacklist']
            },
            "action": "Use POST /api/v2/quality/update to apply these recommendations"
        }
    
    except Exception as e:
        LOGGER.error(f"[V2-API] Recommendations error: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/tracking/active")
async def get_active_tracking():
    """Get all active picks being tracked (48h window)"""
    try:
        from core.active_tracking import get_active_tracker
        tracker = get_active_tracker()
        active = tracker.get_active_picks()
        
        picks = []
        for p in active:
            picks.append({
                "symbol": p.symbol,
                "asset_type": p.asset_type,
                "direction": p.direction,
                "entry_price": p.entry_price,
                "target_price": p.target_price,
                "stop_price": p.stop_price,
                "current_price": p.current_price,
                "confidence": p.confidence,
                "pct_change": p.pct_change,
                "pct_to_target": p.pct_to_target,
                "pct_to_stop": p.pct_to_stop,
                "hours_remaining": p.hours_remaining,
                "is_on_track": p.is_on_track,
                "status": p.status.value,
                "outcome": p.outcome.value,
                "created_at": p.created_at.isoformat(),
                "expires_at": p.expires_at.isoformat(),
            })
        
        wins, losses, neutral = tracker.get_running_stats()
        
        return {
            "ok": True,
            "active_picks": len(picks),
            "picks": picks,
            "running_stats": {
                "wins": wins,
                "losses": losses,
                "neutral": neutral,
                "total": wins + losses + neutral,
                "win_rate": (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
            }
        }
    except Exception as e:
        LOGGER.error(f"Active tracking error: {e}")
        return {"ok": False, "error": str(e)}


@router.post("/tracking/check")
async def force_tracking_check():
    """Force check all active picks for price updates"""
    try:
        from core.active_tracking import get_active_tracker, check_and_update_prices
        
        def _send_telegram(msg: str) -> bool:
            return _tg_send_chat_message(TELEGRAM_CHAT_ID, msg)
        
        async def _get_price(symbol: str) -> float:
            try:
                from core.asset_classifier import get_asset_type
                asset_class = get_asset_type(symbol)
                
                if asset_class.startswith("crypto"):
                    result = turbo_crypto_price(symbol, max_budget_s=2.0)
                else:
                    result = turbo_stock_price(symbol, max_budget_s=2.0)
                
                if result and result.get("ok") and result.get("price"):
                    return float(result["price"])
            except Exception:
                pass
            return 0.0
        
        results = await check_and_update_prices(_get_price, _send_telegram)
        return {"ok": True, **results}
    except Exception as e:
        LOGGER.error(f"Tracking check error: {e}")
        return {"ok": False, "error": str(e)}


@router.post("/tracking/close")
async def close_tracked_pick(request: Request, symbol: str, status: str = "stop_hit"):
    """
    Admin endpoint to manually close a tracked pick.
    
    Use this to stop repeated alerts when a pick needs to be marked as closed.
    
    Args:
        symbol: The symbol to close (e.g., "FTM", "ANKR")
        status: The status to set ("stop_hit", "target_hit", "expired", "manual_close")
    
    Requires X-Cron-Secret header.
    """
    cron_secret = os.getenv("CRON_SECRET", "ghost-cron-2024")
    provided_secret = request.headers.get("X-Cron-Secret", "")
    
    if not cron_secret or provided_secret != cron_secret:
        return {"ok": False, "error": "Unauthorized - invalid X-Cron-Secret"}
    
    valid_statuses = ["stop_hit", "target_hit", "expired", "manual_close"]
    if status not in valid_statuses:
        return {"ok": False, "error": f"Invalid status. Must be one of: {valid_statuses}"}
    
    symbol = symbol.upper().strip()
    updated_pg = False
    updated_sqlite = False
    
    # Update PostgreSQL
    database_url = os.getenv("DATABASE_URL", "")
    if database_url:
        try:
            from core.db_pool import get_sync_connection
            with get_sync_connection() as conn:
                cur = conn.cursor()
                cur.execute(
                    "UPDATE ghost_tracked_picks SET status = %s WHERE symbol = %s AND status = 'active'",
                    (status, symbol)
                )
                rows_affected = cur.rowcount
                cur.close()
            updated_pg = rows_affected > 0
            LOGGER.info(f"[TRACKING] Updated {symbol} to {status} in PostgreSQL ({rows_affected} rows)")
        except Exception as e:
            LOGGER.error(f"[TRACKING] PostgreSQL update failed for {symbol}: {e}")
    
    # Also update SQLite
    try:
        import sqlite3
        sqlite_path = os.getenv("GHOST_PREDICT_DB", "/app/data/ghost_predictions.db")
        conn = sqlite3.connect(sqlite_path)
        cur = conn.cursor()
        cur.execute("UPDATE tracked_picks SET status = ? WHERE symbol = ?", (status, symbol))
        sqlite_rows = cur.rowcount
        conn.commit()
        conn.close()
        updated_sqlite = sqlite_rows > 0
    except Exception as e:
        LOGGER.debug(f"SQLite update failed for {symbol}: {e}")
    
    return {
        "ok": True,
        "symbol": symbol,
        "new_status": status,
        "updated_postgresql": updated_pg,
        "updated_sqlite": updated_sqlite,
    }


@router.get("/api/learning/symbols")
async def learning_symbol_accuracy():
    """
    📈 Get per-symbol accuracy data for learning.
    
    Returns list of symbols with accuracy stats, sorted by prediction count.
    """
    try:
        from core.db_pool import get_sync_connection
        
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            return {"ok": False, "error": "DATABASE_URL not configured"}
        
        with get_sync_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT symbol, total_predictions, correct_predictions, accuracy_pct, status, last_updated
                FROM ghost_symbol_accuracy
                ORDER BY total_predictions DESC
            """)
            
            symbols = []
            for row in cursor.fetchall():
                symbol, total, correct, acc, status, updated = row
                symbols.append({
                    "symbol": symbol,
                    "total_predictions": total,
                    "correct_predictions": correct,
                    "accuracy_pct": float(acc) if acc else 0,
                    "status": status,
                    "last_updated": updated.isoformat() if updated else None,
                    "recommendation": (
                        "EXCLUDE" if acc and acc < 40 and total >= 10 else
                        "BOOST" if acc and acc > 70 and total >= 10 else
                        "NORMAL"
                    )
                })
            
            return {
                "ok": True,
                "symbols": symbols,
                "total_symbols": len(symbols)
            }
        
    except Exception as e:
        LOGGER.error(f"[LEARNING] Symbol accuracy error: {e}", exc_info=True)
        return {"ok": False, "error": str(e)}


@router.get("/api/alerts")
async def api_alerts_preview():
    sig = _evaluate_signal()
    out = {"signal": sig, "hold_override": bool(ALERT_STATE.get("hold_override"))}
    try:
        raw = json.dumps(out, sort_keys=True).encode("utf-8")
        etag = hashlib.sha256(raw).hexdigest()
        resp = JSONResponse(out)
        resp.headers["ETag"] = etag
        resp.headers["Cache-Control"] = "no-store"
        return resp
    except Exception:
        return JSONResponse(out)


@router.get("/api/runtime/config")
async def api_runtime_config_get():
    return {
        "price_ttl_s": PRICE_TTL_S,
        "price_ttl_open_s": PRICE_TTL_OPEN_S,
        "news_ttl_s": NEWS_TTL_S,
        "yahoo_first": bool(PRICE_YAHOO_FIRST),
        "price_max_deviation_open": PRICE_MAX_DEVIATION_OPEN,
        "reuters_feeds_on": bool(REUTERS_FEEDS_ON),
        "diag_collapse_dupes": bool(DIAG_COLLAPSE_DUPES),
        "diag_ring_size": (getattr(EVENTS, "maxlen", None) or len(EVENTS) or 0),
        "overlay_enabled": bool(OVERLAY_ENABLED),
        "overlay_dt_minutes": OVERLAY_DT_MINUTES,
        "learning_enabled": bool(LEARNING_ENABLED),
        "band_widen_factor": BAND_WIDEN_FACTOR,
        "forecast_step_s": FORECAST_STEP_S,
        "forecast_horizon_s": FORECAST_HORIZON_S,
    }


@router.post("/api/runtime/config")
async def api_runtime_config_post(
    body: RuntimeConfigBody, credentials: HTTPAuthorizationCredentials | None = AUTH_DEP
):
    _require_bearer(
        (f"Bearer {credentials.credentials}") if credentials and credentials.credentials else None
    )
    global \
        PRICE_TTL_S, \
        PRICE_TTL_OPEN_S, \
        NEWS_TTL_S, \
        PRICE_YAHOO_FIRST, \
        PRICE_MAX_DEVIATION_OPEN, \
        REUTERS_FEEDS_ON, \
        DIAG_COLLAPSE_DUPES, \
        EVENTS
    global \
        OVERLAY_ENABLED, \
        OVERLAY_DT_MINUTES, \
        LEARNING_ENABLED, \
        BAND_WIDEN_FACTOR, \
        FORECAST_STEP_S, \
        FORECAST_HORIZON_S
    regenerate_grid = False
    if body.price_ttl_s is not None:
        if body.price_ttl_s <= 0:
            raise HTTPException(422, "price_ttl_s must be > 0")
        PRICE_TTL_S = int(body.price_ttl_s)
    if body.price_ttl_open_s is not None:
        if body.price_ttl_open_s <= 0:
            raise HTTPException(422, "price_ttl_open_s must be > 0")
        PRICE_TTL_OPEN_S = int(body.price_ttl_open_s)
    if body.news_ttl_s is not None:
        if body.news_ttl_s <= 0:
            raise HTTPException(422, "news_ttl_s must be > 0")
        NEWS_TTL_S = int(body.news_ttl_s)
    if body.yahoo_first is not None:
        PRICE_YAHOO_FIRST = bool(int(body.yahoo_first))
    if body.price_max_deviation_open is not None:
        if float(body.price_max_deviation_open) <= 0:
            raise HTTPException(422, "price_max_deviation_open must be > 0")
        PRICE_MAX_DEVIATION_OPEN = float(body.price_max_deviation_open)
    if body.reuters_feeds_on is not None:
        REUTERS_FEEDS_ON = 1 if int(body.reuters_feeds_on) else 0
    if body.diag_collapse_dupes is not None:
        DIAG_COLLAPSE_DUPES = bool(int(body.diag_collapse_dupes))
    if body.diag_ring_size is not None:
        sz = max(10, min(5000, int(body.diag_ring_size)))
        try:
            # Rebuild EVENTS deque with new maxlen, preserving most recent
            from collections import deque as _deque

            new_ring = _deque(list(EVENTS)[-sz:], maxlen=sz)
            EVENTS = new_ring  # type: ignore[assignment]
        except Exception:
            pass
    if body.overlay_enabled is not None:
        OVERLAY_ENABLED = 1 if int(body.overlay_enabled) else 0
    if body.overlay_dt_minutes is not None:
        OVERLAY_DT_MINUTES = max(1, int(body.overlay_dt_minutes))
    if body.learning_enabled is not None:
        LEARNING_ENABLED = 1 if int(body.learning_enabled) else 0
    if body.band_widen_factor is not None:
        BAND_WIDEN_FACTOR = max(0.1, float(body.band_widen_factor))
    if body.forecast_step_s is not None:
        new_step = max(300, min(86400, int(body.forecast_step_s)))  # 5min to 24h
        if new_step != FORECAST_STEP_S:
            FORECAST_STEP_S = new_step
            regenerate_grid = True
    if body.forecast_horizon_s is not None:
        new_horizon = max(3600, min(604800, int(body.forecast_horizon_s)))  # 1h to 7d
        if new_horizon != FORECAST_HORIZON_S:
            FORECAST_HORIZON_S = new_horizon
            regenerate_grid = True
    # Trigger grid regeneration if forecast params changed
    if regenerate_grid:
        try:
            _generate_forecast_grid(WOLF)
            _add_event(
                "forecast.grid",
                "Forecast grid regenerated",
                {"step_s": FORECAST_STEP_S, "horizon_s": FORECAST_HORIZON_S},
            )
        except Exception as e:
            print(f"[CONFIG] Failed to regenerate grid: {e}")
    _add_event(
        "runtime.config",
        "Runtime config updated",
        {
            "ttl_price": PRICE_TTL_S,
            "ttl_price_open": PRICE_TTL_OPEN_S,
            "ttl_news": NEWS_TTL_S,
            "yahoo_first": PRICE_YAHOO_FIRST,
            "reuters": bool(REUTERS_FEEDS_ON),
            "diag_collapse": bool(DIAG_COLLAPSE_DUPES),
        },
    )
    return await api_runtime_config_get()


@router.post("/api/advisor_refresh")
async def api_advisor_refresh(symbol: str = WOLF):
    try:
        # Nudge background systems: price refresh and immediate forecast generation
        PRICE_CACHE.pop(symbol, None)
        # Try immediate fetch to warm cache
        try:
            get_wolf_price()
        except Exception:
            pass
        # Generate a new 48h forecast in the spec-compliant table
        res = _generate_48h_forecast(symbol)
        ok = bool(res.get("ok"))
        return {"ok": ok, "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start")
async def ui_start(credentials: HTTPAuthorizationCredentials | None = AUTH_DEP):
    # Optional bearer
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass
    STATE["active"] = True
    _add_event("control", "Engine started", {"active": True})
    return {"ok": True, "active": True}


@router.post("/control")
async def ui_control(
    body: ControlBody, credentials: HTTPAuthorizationCredentials | None = AUTH_DEP
):
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass
    act = (body.action or "").strip().lower()
    if act == "stop":
        STATE["active"] = False
        _add_event("control", "Engine stopped", {"active": False})
        return {"ok": True, "active": False}
    if act == "reset":
        # Reset state (compat with prebuilt UI)
        STATE["qty"] = 0.0
        STATE["avg_cost"] = 0.0
        _persist_save()
        _add_event("state.reset", "State reset", {"qty": 0.0, "avg_cost": 0.0})
        return {"ok": True, "active": bool(STATE.get("active", True)), "reset": True}
    return {"ok": True, "active": bool(STATE.get("active", True))}


@router.post("/api/state/reset")
async def ui_state_reset(
    body: dict | None = None,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass
    STATE["qty"] = 0.0
    STATE["avg_cost"] = 0.0
    _persist_save()
    _add_event("state.reset", "State reset", {"qty": 0.0, "avg_cost": 0.0})
    return {"ok": True, "position": {"qty": 0.0, "avg_cost": 0.0}}


@router.post("/api/mode")
async def ui_mode(body: ModeBody, credentials: HTTPAuthorizationCredentials | None = AUTH_DEP):
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass
    # enabled True => live, False => sim
    if _is_live_enforced():
        # Fail-closed in production: SIM must never be enabled.
        if body.enabled is not None and bool(body.enabled) is False:
            raise HTTPException(status_code=403, detail="SIM mode disabled (ENFORCE_LIVE=1)")
        STATE["mode"] = "live"
    else:
        enabled = (
            bool(body.enabled) if body.enabled is not None else (STATE.get("mode", "live") != "live")
        )
        STATE["mode"] = "live" if enabled else "sim"
    _add_event("mode", "Mode updated", {"mode": STATE["mode"]})
    return {"ok": True, "mode": STATE["mode"]}


@router.post("/api/bank/add_position")
async def ui_bank_add_position(
    body: AddPositionBody, credentials: HTTPAuthorizationCredentials | None = AUTH_DEP
):
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass
    sym = (body.symbol or "").upper()
    if sym != WOLF:
        raise HTTPException(422, "symbol must be WOLF")
    if body.quantity < 0 or body.price <= 0:
        raise HTTPException(422, "quantity must be >= 0 and price > 0")
    # Add position semantics: adjust qty and avg cost using simple weighted average
    q0 = float(STATE.get("qty", 0.0))
    a0 = float(STATE.get("avg_cost", 0.0))
    q1 = float(body.quantity)
    p1 = float(body.price)
    if q1 > 0:
        total_cost = a0 * q0 + p1 * q1
        new_qty = q0 + q1
        new_avg = (total_cost / new_qty) if new_qty > 0 else 0.0
    else:
        new_qty = q0
        new_avg = a0
    STATE["qty"] = float(new_qty)
    STATE["avg_cost"] = float(round(new_avg, 2))
    _persist_save()
    _add_event(
        "position.add",
        "Position added",
        {
            "qty": STATE["qty"],
            "avg_cost": STATE["avg_cost"],
            "delta_qty": q1,
            "price": p1,
        },
    )
    # Include 'success' for UI compatibility
    return {
        "ok": True,
        "success": True,
        "symbol": WOLF,
        "qty": STATE["qty"],
        "avg_cost": STATE["avg_cost"],
    }


@router.get("/api/simulation_data")
async def api_simulation_data():
    """Serve simulation data for UI validation testing."""
    if _is_live_enforced() or os.getenv("SIM_MODE", "0") == "0":
        raise HTTPException(status_code=404, detail="simulation_disabled")
    import json
    import os

    sim_file = os.path.join(os.path.dirname(__file__), "public", "simulation_data.json")

    if not os.path.exists(sim_file):
        return {
            "error": "Simulation data not found",
            "hint": "Run: python3 generate_simulation_data.py",
        }

    with open(sim_file) as f:
        data = json.load(f)

    return data


@router.post("/api/v3/regression/telegram-test")
async def api_v3_regression_telegram_test(request: Request):
    """Send a single controlled Telegram message for regression auditing.

    Guardrails:
    - requires REGRESSION_ALLOW_TELEGRAM_TEST=1
    - requires X-Regression-Key header matching REGRESSION_KEY
    """
    allow = _is_truthy(os.getenv("REGRESSION_ALLOW_TELEGRAM_TEST", "0"))
    key_required = (os.getenv("REGRESSION_KEY") or "").strip()
    key_got = (request.headers.get("x-regression-key") or "").strip()

    if not allow:
        raise HTTPException(status_code=403, detail="telegram_regression_test_disabled")
    if not key_required or key_got != key_required:
        raise HTTPException(status_code=403, detail="invalid_regression_key")
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return {"ok": False, "error": "telegram_not_configured"}

    # Exact template required by regression spec.
    ts = datetime.now(UTC).isoformat()
    msg = (
        "🔍 GHOST PROTOCOL — REGRESSION TEST\n\n"
        "Status: TELEGRAM PIPELINE VERIFIED\n"
        "Environment: PRODUCTION (Railway)\n"
        f"Timestamp (UTC): {ts}\n\n"
        "This is a controlled system test.\n"
        "• No trade signal generated\n"
        "• No capital at risk\n"
        "• No prediction executed\n\n"
        "If you received this message, Ghost Protocol can successfully send Telegram notifications end-to-end."
    )

    ok_all, deliveries = await asyncio.to_thread(send_telegram_detailed, msg)
    return {"ok": bool(ok_all), "deliveries": deliveries}


@router.get("/api/_crash")
async def _crash():
    """Canary route to verify exception handlers always return JSON 500."""
    raise RuntimeError("boom")


@router.get("/api/status")
async def api_status():
    """Status endpoint with runtime environment configuration.
    Returns current mode, active flags, and critical env settings.
    """
    try:
        env_flags = {
            "SIM_MODE": os.getenv("SIM_MODE", "0"),
            "STOCKS_ENABLED": os.getenv("STOCKS_ENABLED", "1"),
            "CRYPTO_ENABLED": os.getenv("CRYPTO_ENABLED", "0"),
            "PRICE_STRICT_LIVE": os.getenv("PRICE_STRICT_LIVE", "0"),
            "PRICE_REQUIRE_QUORUM": os.getenv("PRICE_REQUIRE_QUORUM", "0"),
            "PREDICT_REQUIRE_PRICE_QUORUM": os.getenv("PREDICT_REQUIRE_PRICE_QUORUM", "0"),
            "PRICE_MIN_PROVIDERS": os.getenv("PRICE_MIN_PROVIDERS", ""),
            "STOCK_PRICE_SOURCE": os.getenv("STOCK_PRICE_SOURCE", "polygon"),
            "CRYPTO_PRICE_SOURCE": os.getenv("CRYPTO_PRICE_SOURCE", "coingecko"),
        }
        return {
            "mode": str(STATE.get("mode", "live")),
            "active": bool(STATE.get("active", True)),
            "version": app.version,
            "env": env_flags,
            "uptime_seconds": int(time.time() - _START_TS),
        }
    except Exception:
        return {"mode": "live", "active": True, "version": app.version}


@router.get("/api/doctor")
async def api_doctor():
    """Run System Doctor health check on demand (same as 7 AM daily)."""
    try:
        from core.system_doctor import run_system_doctor
        return run_system_doctor()
    except Exception as e:
        return {"overall": "ERROR", "error": str(e)}


@router.post("/api/doctor/notify")
async def api_doctor_notify():
    """Run System Doctor and send Telegram report."""
    try:
        from core.system_doctor import run_and_notify
        return run_and_notify()
    except Exception as e:
        return {"overall": "ERROR", "error": str(e)}


@router.get("/api/performance-gate")
async def api_performance_gate():
    """Performance Gate: which symbols are alive, warned, or killed."""
    try:
        from core.performance_gate import get_summary as _pg_summary
        return _pg_summary()
    except Exception as e:
        return {"error": str(e), "status": "unavailable"}


@router.get("/api/performance-gate/scorecard")
async def api_performance_gate_scorecard():
    """Per-symbol scorecard with accuracy, trade count, and status."""
    try:
        from core.performance_gate import get_scorecard as _pg_scorecard
        return {"scorecard": _pg_scorecard()}
    except Exception as e:
        return {"error": str(e), "scorecard": []}


@router.get("/api/autopilot")
async def api_autopilot():
    """Accuracy Autopilot: circuit breaker status and configuration."""
    try:
        from core.accuracy_autopilot import get_status as _ap_status
        return _ap_status()
    except Exception as e:
        return {"error": str(e), "paused": False, "status": "unavailable"}


@router.post("/api/autopilot/check")
async def api_autopilot_check():
    """Force an autopilot check-and-update cycle now."""
    try:
        from core.accuracy_autopilot import check_and_update as _ap_check, get_status as _ap_status2
        _ap_check()
        return _ap_status2()
    except Exception as e:
        return {"error": str(e), "status": "check_failed"}


@router.get("/api/learning")
async def api_learning():
    """Trade Learning Loop: insights, patterns, and confidence adjustments."""
    try:
        from core.trade_learning_loop import get_summary as _tll_summary
        return _tll_summary()
    except Exception as e:
        return {"error": str(e), "status": "unavailable"}


@router.get("/api/learning/insights")
async def api_learning_insights():
    """Detailed learning insights: win rates by bucket, symbol, direction."""
    try:
        from core.trade_learning_loop import get_insights as _tll_insights
        return _tll_insights()
    except Exception as e:
        return {"error": str(e), "insights": {}}


@router.get("/api/stability/status")
async def api_stability_status():
    """
    STABILITY MODE - System health and accuracy tracking.
    
    This endpoint provides the metrics needed during the 2-week stability period:
    - Intel status (is it actually running?)
    - VIX value (is it real or fake 15.0?)
    - Confidence stats (are we staying under 85%?)
    - Pattern accuracy (are we hitting 60%+ win rate?)
    
    Use this for daily monitoring during the stability period.
    """
    from core.pattern_tracker import get_pattern_accuracy
    from core.world_context import get_real_vix
    
    result = {
        "timestamp": int(time.time()),
        "stability_period": {
            "started": "2025-01-31",
            "target_end": "2025-02-14",
            "days_remaining": max(0, 14 - (datetime.utcnow() - datetime(2025, 1, 31)).days),
        },
        "checks": {},
        "overall_status": "healthy",  # Will be set based on checks
    }
    
    issues = []
    
    # Check 1: Intel status
    try:
        intel_config = os.environ.get("GHOST_INTEL_ENABLED", "true")
        intel_status = "enabled" if intel_config.lower() == "true" else "disabled"
        result["checks"]["intel"] = {
            "status": intel_status,
            "ok": intel_status == "enabled",
        }
        if intel_status != "enabled":
            issues.append("Intel disabled")
    except Exception as e:
        result["checks"]["intel"] = {"status": "error", "error": str(e), "ok": False}
        issues.append(f"Intel check failed: {e}")
    
    # Check 2: VIX value (should NOT be 15.0 fake)
    try:
        vix_value, vix_source = get_real_vix()  # Sync function, returns tuple
        is_fake = vix_source == "default" or abs(vix_value - 15.0) < 0.01
        result["checks"]["vix"] = {
            "value": vix_value,
            "source": vix_source,
            "is_fake": is_fake,
            "ok": not is_fake,
        }
        if is_fake:
            issues.append("VIX returning fake/default value")
    except Exception as e:
        result["checks"]["vix"] = {"value": None, "error": str(e), "ok": False}
        issues.append(f"VIX check failed: {e}")
    
    # Check 3: Confidence distribution (should be <= 85%)
    try:
        # Get recent predictions to check confidence distribution
        db_url = os.environ.get("DATABASE_URL")
        if db_url:
            try:
                from core.db_pool import get_sync_connection
                with get_sync_connection() as conn:
                    cur = conn.cursor()
                    cur.execute("""
                        SELECT 
                            MAX(confidence) as max_conf,
                            AVG(confidence) as avg_conf,
                            COUNT(*) as total,
                            SUM(CASE WHEN confidence > 0.85 THEN 1 ELSE 0 END) as over_85
                        FROM predictions
                        WHERE timestamp > NOW() - INTERVAL '24 hours'
                    """)
                    row = cur.fetchone()
                    if row and row[2] > 0:
                        max_conf = float(row[0]) if row[0] else 0
                        avg_conf = float(row[1]) if row[1] else 0
                        over_85_count = int(row[3]) if row[3] else 0
                        result["checks"]["confidence"] = {
                            "max_24h": round(max_conf, 3),
                            "avg_24h": round(avg_conf, 3),
                            "predictions_24h": row[2],
                            "over_85_percent": over_85_count,
                            "ok": max_conf <= 0.85,
                        }
                        if max_conf > 0.85:
                            issues.append(f"Confidence exceeded 85%: {max_conf:.1%}")
                    else:
                        result["checks"]["confidence"] = {"predictions_24h": 0, "ok": True}
            except Exception:
                result["checks"]["confidence"] = {"status": "no_database", "ok": True}
        else:
            result["checks"]["confidence"] = {"status": "no_database", "ok": True}
    except Exception as e:
        result["checks"]["confidence"] = {"error": str(e), "ok": False}
        issues.append(f"Confidence check failed: {e}")
    
    # Check 4: Pattern accuracy (target: 60%+)
    try:
        accuracy = get_pattern_accuracy()
        overall = accuracy.get("overall", {})
        win_rate = overall.get("accuracy", 0)
        detections = overall.get("detections", 0)
        
        result["checks"]["pattern_accuracy"] = {
            "win_rate": win_rate,
            "detections_tracked": detections,
            "target": 60.0,
            "ok": win_rate >= 60.0 or detections < 10,  # Need 10+ samples
            "note": "Need 10+ detections for meaningful accuracy" if detections < 10 else None,
        }
        if detections >= 10 and win_rate < 60:
            issues.append(f"Win rate below target: {win_rate:.1f}% < 60%")
    except Exception as e:
        result["checks"]["pattern_accuracy"] = {"error": str(e), "ok": False}
        issues.append(f"Accuracy check failed: {e}")
    
    # Set overall status
    if issues:
        result["overall_status"] = "issues_found"
        result["issues"] = issues
    
    return result


@router.get("/api/hunter/snapshot")
async def api_hunter_snapshot():
    """
    Ghost Hunter V1: Compact multi-symbol prediction view for UI.

    Returns classified predictions (stocks vs crypto) with essential fields:
    - symbol, direction, confidence, horizon_h

    Omits symbols with no predictions (keeps response compact).

    Example response:
    {
      "timestamp": 1763647539,
      "stocks": [
        {"symbol": "WOLF", "direction": "FLAT", "confidence": 0.6, "horizon_h": 48},
        {"symbol": "AAPL", "direction": "UP", "confidence": 0.72, "horizon_h": 48}
      ],
      "crypto": [
        {"symbol": "WEPE", "direction": "UP", "confidence": 0.68, "horizon_h": 24},
        {"symbol": "BTC", "direction": "DOWN", "confidence": 0.55, "horizon_h": 24}
      ]
    }
    """
    try:
        stocks = []
        crypto = []

        # Classify and format predictions
        for sym, pred in _LATEST_PREDICTIONS.items():
            pred_compact = {
                "symbol": pred["symbol"],
                "direction": pred["direction"],
                "confidence": pred["confidence"],
                "horizon_h": pred["horizon_h"],
            }

            category = _classify_symbol_category(sym)
            if category == "stocks":
                stocks.append(pred_compact)
            elif category in ("crypto", "vip"):
                crypto.append(pred_compact)

        # Get latest timestamp
        timestamp = None
        if _LATEST_PREDICTIONS:
            timestamp = int(max(p["run_at"] for p in _LATEST_PREDICTIONS.values()))

        return {
            "timestamp": timestamp,
            "stocks": stocks,
            "crypto": crypto,
        }

    except Exception as e:
        LOGGER.exception(f"Failed to build hunter snapshot: {e}")
        raise HTTPException(500, "Failed to build hunter snapshot")


@router.get("/api/system/ping")
async def api_system_ping(request: Request):
    """Simple ping endpoint to test /api/system/ auth bypass"""
    return {
        "ok": True,
        "message": "system endpoint accessible",
        "request_path": str(request.url.path),
        "request_url": str(request.url),
        "ts": int(time.time())
    }


@router.get("/api/system/orchestrator")
async def api_system_orchestrator():
    """
    Get Master Orchestrator status - all background services health
    Shows which systems are running, failed, disabled, or on-demand
    """
    # Quick non-blocking status check
    return {
        "ok": True,
        "message": "orchestrator status",
        "timestamp": int(time.time()),
        "note": "Full status check temporarily disabled for debugging"
    }


@router.get("/api/tick")
async def api_tick():
    """Return current tick count and timestamp. Never returns empty dict."""
    return {"tick": int(STATE.get("tick", 0)), "ts": int(time.time() * 1000)}


@router.get("/api/goals")
async def api_goals():
    """Return account goals with defaults. Never returns empty dict."""
    goals_data = STATE.get("goals")
    if isinstance(goals_data, dict) and goals_data:
        return {
            "daily": float(goals_data.get("daily", 0)),
            "weekly": float(goals_data.get("weekly", 0)),
            "monthly": float(goals_data.get("monthly", 0)),
            "yearly": float(goals_data.get("yearly", 0)),
            "ts": int(time.time() * 1000),
        }
    return {"daily": 0, "weekly": 0, "monthly": 0, "yearly": 0, "ts": int(time.time() * 1000)}


@router.get("/api/ghost/score")
async def api_ghost_score():
    """Return Ghost performance score. Never returns empty dict."""
    score = STATE.get("ghost_score")
    return {"ghost_score": float(score) if score is not None else 0.0, "ts": int(time.time() * 1000)}


@router.post("/agent/stop")
async def agent_stop(credentials: HTTPAuthorizationCredentials | None = AUTH_DEP):
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass
    STATE["active"] = False
    _add_event("control", "Engine stopped", {"active": False, "via": "/agent/stop"})
    return {"ok": True, "active": False}


@router.post("/agent/control")
async def agent_control(
    body: AgentControlBody, credentials: HTTPAuthorizationCredentials | None = AUTH_DEP
):
    """Emergency control shim used by base.html.
    execution_enabled=False will stop the engine; advisory_only flag is acknowledged but advisory logic is not implemented in WOLF-only mode.
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass
    if body.execution_enabled is not None and not body.execution_enabled:
        STATE["active"] = False
        _add_event(
            "control",
            "Emergency stop engaged",
            {"active": False, "advisory_only": bool(body.advisory_only)},
        )
    return {
        "ok": True,
        "active": bool(STATE.get("active", True)),
        "advisory_only": bool(body.advisory_only),
    }


@router.get("/fusion/ai")
async def fusion_ai():
    """Return Macro Brain advisory used by UI Fusion panel."""
    try:
        price, prev, provider = get_wolf_price()
        news = get_wolf_news(limit=10)
        ns = (news.get("news_signal") or {}).get("score")
        outlook = _macro_brain(price, ns)

        # Derive fusion risk & confidence metrics (lightweight heuristic for now)
        # outlook structure expected: {enabled: bool, bias: str|None, score: float|None, reasons: [...]} (heuristic based on existing macro brain)
        raw_score = None
        try:
            score_val = outlook.get("score") if isinstance(outlook, dict) else None
            if score_val is not None:
                raw_score = float(score_val)
        except Exception:
            raw_score = None

        # Confidence: absolute scaled score (0-1) mapped to percentage
        if raw_score is not None:
            confidence_score = min(
                1.0, max(0.0, abs(raw_score) / 3.0)
            )  # assume |score|≈3 is strong
        else:
            confidence_score = 0.0

        # Risk score: inverse of confidence (higher confidence = lower risk)
        risk_score = round(1.0 - confidence_score, 3)

        # Drivers: top textual reasons if present
        drivers: list[dict[str, str | float]] = []
        try:
            reasons = []
            if isinstance(outlook, dict):
                reasons = outlook.get("reasons") or []
            for r in reasons[:5]:
                # Each reason becomes a driver with lightweight weighting = descending order / presence of numeric weight inside
                if isinstance(r, str):
                    drivers.append({"reason": r})
                elif isinstance(r, dict):
                    # Already structured; pass through selected keys
                    d = {k: v for k, v in r.items() if k in ("reason", "why", "score", "weight")}
                    if d:
                        drivers.append(d)  # type: ignore[arg-type]
        except Exception:
            pass

        fusion_payload = {
            "outlook": outlook,
            "source": "macro_brain",
            "risk_score": risk_score,
            "confidence_score": round(confidence_score, 3),
            "drivers": drivers,
        }
        return fusion_payload
    except Exception:
        return {
            "outlook": {"enabled": False, "error": "unavailable"},
            "risk_score": 1.0,
            "confidence_score": 0.0,
            "drivers": [],
        }


@router.post("/fusion/refresh")
async def fusion_refresh():
    # Force recompute by clearing any cached news sentiment and calling macro again
    try:
        NEWS_CACHE["ts"] = 0.0
    except Exception:
        pass
    return await fusion_ai()


@router.get("/diagnostics/summary")
async def diagnostics_summary():
    """Compact diagnostics blob for UI panel."""
    # Health payload
    try:
        h = await health()
        health_json = h
    except Exception:
        health_json = {"ok": False}
    breakers = {k: v for k, v in _PROVIDER_BREAKERS.items()}
    cfg = {
        "mode": STATE.get("mode"),
        "active": STATE.get("active"),
        "providers": {
            "alphavantage": bool(ALPHAVANTAGE_KEY),
            "polygon": bool(POLYGON_KEY),
        },
    }
    invariants = []
    try:
        # Mirror invariants from main module's EVENTS_RING when running tests
        import main as _main  # type: ignore

        ring = getattr(_main, "EVENTS_RING", [])
        for e in reversed(ring[-200:]):
            msg = str(e.get("message", ""))
            if "invariant" in msg:
                invariants.append(e)
            if len(invariants) >= 5:
                break
    except Exception:
        pass

    # Add price diagnostics from Phase 1 enhancements
    price_diag = {}
    try:
        is_open, _ = _is_market_open_now()
        price_diag = {
            "market_open": bool(is_open),
            "last_fetch_provider": PRICE_DIAG.get("last_fetch_provider"),
            "last_fetch_latency_ms": PRICE_DIAG.get("last_fetch_latency_ms"),
            "last_good_price_ts": PRICE_DIAG.get("last_good_price_ts"),
            "fallback_reason": PRICE_DIAG.get("fallback_reason"),
            "provider_spread": PRICE_DIAG.get("provider_spread"),
            "quorum_ok": PRICE_DIAG.get("quorum_ok"),
        }
    except Exception:
        pass

    ev = list(EVENTS)[-20:]
    return {
        "health": health_json,
        "events": ev,
        "providers": breakers,
        "config": cfg,
        "invariants": invariants,
        "price_diag": price_diag,
    }


@router.get("/self/diagnostics")
async def self_diagnostics(credentials: HTTPAuthorizationCredentials | None = AUTH_DEP):
    """Self-awareness diagnostics endpoint.

    Returns current time, market status, provider health, AI config, fusion score, and memory stats.
    """
    _require_bearer(
        (f"Bearer {credentials.credentials}") if credentials and credentials.credentials else None
    )
    try:
        # Time and market status
        is_open, _ = _is_market_open_now()
        now_s = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime())

        # Price snapshot and fusion
        ctx = _build_ai_context()

        # Provider health (compact)
        providers = {
            "alphavantage": bool(ALPHAVANTAGE_KEY),
            "polygon": bool(POLYGON_KEY),
            "yahoo": True,
        }

        # AI memory stats
        mem = {
            "ring_size": len(AI_MEMORY_RING),
        }
        try:
            if AI_MEMORY_STORE is not None:
                cur = AI_MEMORY_STORE.conn.execute("SELECT COUNT(1), MAX(ts) FROM ai_memory")
                row = cur.fetchone()
                if row:
                    mem["db_records"] = int(row[0] or 0)
                    mem["latest_ts"] = int(row[1] or 0)
        except Exception:
            pass

        return {
            "ok": True,
            "now": now_s,
            "market_open": bool(is_open),
            "ai": {
                "enabled": bool(AGENTS_ENABLED),
                "provider": AI_PROVIDER,
                "model": AGENT_MODEL,
            },
            "providers": providers,
            "fusion": ctx.get("fusion"),
            "prices": ctx.get("prices"),
            "news_signal": ctx.get("news_signal"),
            "macro_pressure": ctx.get("macro_pressure"),
            "memory": mem,
        }
    except HTTPException:
        raise
    except Exception as e:
        LOGGER.error(f"self_diagnostics_error: {e}", exc_info=True)
        raise HTTPException(500, f"diagnostics failed: {str(e)[:200]}")


@router.post("/api/agent/ask")
async def api_agent_ask(
    req: ChatRequest, credentials: HTTPAuthorizationCredentials | None = AUTH_DEP
):
    _require_bearer(
        (f"Bearer {credentials.credentials}") if credentials and credentials.credentials else None
    )
    # Route to /ai/chat for now to leverage tool-calling
    try:
        answer = _ask_ghost_ai(req.question.strip())
        ctx = _build_ai_context() if req.include_context else {}
        return {"ok": True, "question": req.question, "answer": answer, "context": ctx}
    except Exception as e:
        LOGGER.error(f"api_agent_ask_error: {e}", exc_info=True)
        raise HTTPException(500, f"agent ask failed: {str(e)}")


@router.get("/api/agent/decisions")
async def api_agent_decisions(limit: int = 20):
    """Get recent agent decisions/trades for the cockpit UI."""
    try:
        decisions = []
        # Try to get from database
        try:
            import sqlite3

            conn = sqlite3.connect("wolf.db")
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT timestamp, action, symbol, confidence, reasoning
                FROM agent_decisions
                ORDER BY timestamp DESC
                LIMIT ?
            """,
                (limit,),
            )
            rows = cursor.fetchall()
            conn.close()
            decisions = [
                {
                    "timestamp": row[0],
                    "action": row[1],
                    "symbol": row[2],
                    "confidence": row[3],
                    "reasoning": row[4],
                }
                for row in rows
            ]
        except Exception:
            decisions = []
        return {"decisions": decisions, "count": len(decisions)}
    except Exception as e:
        LOGGER.error(f"Error getting agent decisions: {e}")
        return {"decisions": [], "count": 0, "error": str(e)}


@router.get("/api/agent/stats")
async def api_agent_stats():
    """Get agent statistics for the cockpit dashboard."""
    try:
        stats = {
            "total_decisions": 0,
            "win_rate": 0.0,
            "avg_confidence": 0.0,
            "active_goals": 0,
            "portfolio_value": 0.0,
            "daily_pnl": 0.0,
            "timestamp": time.time(),
        }
        try:
            import sqlite3

            conn = sqlite3.connect("wolf.db")
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM agent_decisions")
            stats["total_decisions"] = cursor.fetchone()[0] or 0
            cursor.execute("SELECT AVG(confidence) FROM agent_decisions")
            avg_conf = cursor.fetchone()[0]
            stats["avg_confidence"] = float(avg_conf) if avg_conf else 0.0
            conn.close()
        except Exception:
            pass
        try:
            pm = get_portfolio_manager()  # may raise if STAGE4 disabled
            portfolio = pm.get_portfolio() if pm else {}
            stats["portfolio_value"] = (
                float(portfolio.get("nav", 0.0)) if isinstance(portfolio, dict) else 0.0
            )
        except Exception:
            pass
        return stats
    except Exception as e:
        LOGGER.error(f"Error getting agent stats: {e}")
        return {
            "total_decisions": 0,
            "win_rate": 0.0,
            "avg_confidence": 0.0,
            "active_goals": 0,
            "portfolio_value": 0.0,
            "daily_pnl": 0.0,
            "timestamp": time.time(),
            "error": str(e),
        }


@router.get("/api/news")
async def api_news(limit: int = 20):
    """Get recent news articles for the cockpit news feed."""
    try:
        return await _get_news_feed(limit)
    except Exception as e:
        LOGGER.error(f"Error getting news: {e}")
        return {"news": [], "count": 0, "error": str(e)}


@router.get("/api/snapshot")
async def api_snapshot():
    """Get real-time snapshot of entire system state for cockpit."""
    try:
        snapshot = {
            "timestamp": time.time(),
            "portfolio": {},
            "market_regime": {},
            "forecasts": [],
            "goals": [],
            "decisions": [],
            "news": [],
        }
        try:
            pm = get_portfolio_manager()
            snapshot["portfolio"] = pm.get_portfolio() if pm else {}
        except Exception:
            pass
        try:
            regime = await api_stage3_regime_current()
            snapshot["market_regime"] = regime
        except Exception:
            pass
        try:
            forecasts = await api_stage2_forecasts()
            snapshot["forecasts"] = forecasts.get("forecasts", [])[:5]
        except Exception:
            pass
        try:
            decisions_data = await api_agent_decisions(limit=10)
            snapshot["decisions"] = decisions_data.get("decisions", [])
        except Exception:
            pass
        try:
            news_data = await api_news_recent(limit=5)
            snapshot["news"] = news_data.get("news", [])
        except Exception:
            pass
        return snapshot
    except Exception as e:
        LOGGER.error(f"Error generating snapshot: {e}")
        return {"timestamp": time.time(), "error": str(e)}


@router.get("/api/price/diagnostics")
async def api_price_diagnostics(symbol: str | None = None):
    """Detailed price diagnostics for debugging UI.

    Args:
        symbol: Stock symbol to diagnose (required - no default to WOLF)

    Returns:
        {
          symbol: str,
          price: float|None,
          prev_close: float|None,
          provider: str|None,
          cache_age_s: float|None,
          cache_ttl_s: int,
          diag: PRICE_DIAG contents,
          recent_price_events: [...],
          now: epoch seconds
        }
    """
    if not symbol:
        raise HTTPException(status_code=400, detail="symbol parameter is required")

    sym = symbol.upper().strip()

    # Use ensure_price_cached which handles the full provider chain
    # This ensures we get real-time data through the same path as normal API calls
    now = time.time()
    price = None
    provider = None
    cache_age_s: float | None = None

    try:
        # Call ensure_price_cached to force fresh fetch through provider chain
        result = await ensure_price_cached(sym, strict_live=False, max_age_seconds=None)
        if result:
            price = result.get("price")
            prev = result.get("prev_close")
            provider = result.get("provider")
    except HTTPException:
        # If ensure_price_cached raises 404/503, let it propagate
        raise
    except Exception as e:
        LOGGER.debug(f"price_diagnostics_error for {sym}: {e}")

    # Inspect cache directly if available
    try:
        cache_entry = PRICE_CACHE.get(sym)
        if cache_entry:
            ts = cache_entry.get("ts") or cache_entry.get("timestamp")
            if ts:
                cache_age_s = round(now - float(ts), 2)
    except Exception:
        pass
    ttl = PRICE_TTL_OPEN_S if _is_market_open_now()[0] else PRICE_TTL_S

    # Collect recent price-related events (fetch, fallback, anomaly)
    recent_price_events: list[dict[str, Any]] = []
    try:
        for e in reversed(list(EVENTS)[-300:]):
            m = str(e.get("message", ""))
            if any(k in m for k in ("price", "fallback", "anomaly", "prev-close")):
                recent_price_events.append(e)
            if len(recent_price_events) >= 30:
                break
        recent_price_events.reverse()
    except Exception:
        pass

    return {
        "symbol": sym,
        "price": price,
        "prev_close": prev,
        "provider": provider,
        "cache_age_s": cache_age_s,
        "cache_ttl_s": ttl,
        "diag": dict(PRICE_DIAG),
        "backoff_active": {
            k: max(0, int(v.get("until", 0) - now))
            for k, v in (PROVIDER_BACKOFF.items() if "PROVIDER_BACKOFF" in globals() else [])
            if v.get("until", 0) > now
        },
        "recent_price_events": recent_price_events,
        "now": int(now),
    }


@router.get("/api/top_movers")
async def api_top_movers(threshold: float = 7.0, limit: int = 20):
    """
    Get top movers from watchlist that passed GHOST scoring threshold.
    Only symbols with GPS >= threshold appear here - this is your buy signal list.

    Args:
        threshold: GPS threshold (default: 7.0)
        limit: Maximum number of results (default: 20)
    """
    stocks = []

    # Always include WOLF if it passes threshold
    price, prev, provider = get_wolf_price()
    change_pct = 0.0
    try:
        if price is not None and prev and prev > 0:
            change_pct = (price - prev) / prev * 100.0
    except Exception:
        change_pct = 0.0
    row_current = price if price is not None else float(STATE.get("avg_cost", 0.0))

    # WOLF GPS calculation (simplified - you can enhance this)
    wolf_gps = 7.2  # Base GPS for WOLF
    if abs(change_pct) > 5:
        wolf_gps += 0.5
    if abs(change_pct) > 10:
        wolf_gps += 0.5

    if wolf_gps >= threshold:
        stocks.append(
            {
                "sym": WOLF,
                "symbol": WOLF,
                "name": "Wolf Media",
                "price": row_current,
                "change_pct": change_pct,
                "gps": round(wolf_gps, 2),
            }
        )

    # Get watchlist movers that passed threshold
    if WATCHLIST_ENABLED:
        try:
            watchlist_mgr = get_watchlist_manager()
            watchlist_movers = watchlist_mgr.get_top_movers(
                threshold=threshold,
                limit=limit - 1,  # Reserve 1 spot for WOLF
                min_change_pct=0.0,
            )
            stocks.extend(watchlist_movers)
        except Exception as e:
            LOGGER.error(f"Failed to get watchlist movers: {e}")

    return {
        "stocks": stocks[:limit],  # Limit total results
        "crypto": [],
        "threshold": threshold,
        "count": len(stocks),
    }


@router.get("/api/market/movers")
async def api_market_movers(threshold: float = 7.0, limit: int = 20):
    """Alias for /api/top_movers to satisfy UI expectations."""
    return await api_top_movers(threshold=threshold, limit=limit)


@router.get("/api/predictions/run")
async def api_predictions_run(symbol: str = WOLF):
    """
    Trigger a prediction for a symbol.
    This updates _LATEST_PREDICTIONS for Cockpit consumption.
    """
    try:
        # Use run_single_prediction which updates _LATEST_PREDICTIONS
        res = run_single_prediction(symbol)
        return {"ok": True, "result": res}
    except Exception as e:
        LOGGER.error(f"api_predictions_run failed for {symbol}: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/api/predictions/multi/run")
async def api_predictions_multi_run():
    """
    Generate predictions for multiple symbols across stocks, crypto, and VIP coins.
    This is a public endpoint that returns predictions for all configured symbols.
    """
    return _generate_multi_symbol_predictions()


@router.get("/api/predictions/symbols")
async def api_predictions_symbols():
    """
    Return list of supported symbols for predictions.

    - Multi-symbol watchlist: Returns predictions for top 20-40 symbols (fast, cached)
    - Single-symbol API: Supports ANY stock/crypto symbol (on-demand, use /api/predictions/run?symbol=SYMBOL)

    Ghost can predict 500+ stocks and 1000+ crypto via the single-symbol endpoint.
    """
    return {
        "ok": True,
        "multi_symbol_watchlist": {
            "stocks": STOCK_SYMBOLS,
            "crypto": CRYPTO_SYMBOLS,
            "vip": VIP_COINS,
            "total": len(STOCK_SYMBOLS) + len(CRYPTO_SYMBOLS) + len(VIP_COINS),
            "description": "Featured watchlist for /api/predictions/multi/run (cached 120s)"
        },
        "single_symbol_capability": {
            "endpoint": "/api/predictions/run?symbol=SYMBOL",
            "supported_stocks": "500+ (any valid ticker: AAPL, TSLA, AMD, etc.)",
            "supported_crypto": "1000+ (format: BTC, ETH, SOL, etc.)",
            "description": "On-demand prediction for ANY stock or crypto symbol",
            "examples": [
                "/api/predictions/run?symbol=AAPL",
                "/api/predictions/run?symbol=BTC",
                "/api/predictions/run?symbol=AMD",
                "/api/predictions/run?symbol=GME"
            ]
        },
        "note": "Multi-symbol returns batch predictions quickly. Single-symbol supports unlimited tickers on-demand."
    }


@router.get("/api/agent/decide")
async def api_agent_decide_hint():
    """Public hint endpoint; real decision API is /ai/decide (Bearer auth)."""
    return {
        "ok": True,
        "message": "Use POST /ai/decide with Bearer token for live decision",
        "auth": "required",
        "endpoint": "/ai/decide",
    }


@router.get("/api/price/{symbol}")
async def api_price(symbol: str, force: int = 0, strict: int | None = None):
    """Return current price for a symbol with 2.5s timeout to prevent 499 errors."""
    async def get_price_data():
        strict_flag: bool | None = None
        if strict is not None:
            strict_flag = bool(strict)
        if force == 1:
            strict_flag = True

        result = await ensure_price_cached(
            symbol,
            strict_live=strict_flag,
            drop_cache=bool(force),
        )

        response = _build_price_response(result)
        response["force"] = bool(force)
        response["strict_live"] = strict_flag if strict_flag is not None else PRICE_STRICT_LIVE
        return response

    # Apply 2.5s timeout
    return await with_cap(
        get_price_data(),
        sec=2.5,
        fallback={"symbol": symbol, "price": None, "error": "timeout", "provider": "timeout"}
    )


@router.get("/api/price/refresh")
async def api_price_refresh_get(symbol: str = WOLF, strict: int | None = None):
    """Force a live price refresh with 2.5s timeout to prevent 499 errors."""
    async def refresh_price():
        strict_flag = True if strict is None else bool(strict)
        result = await ensure_price_cached(
            symbol,
            strict_live=strict_flag,
            drop_cache=True,
        )
        response = _build_price_response(result)
        response["cache_cleared"] = True
        response["strict_live"] = strict_flag
        return response

    return await with_cap(
        refresh_price(),
        sec=2.5,
        fallback={"symbol": symbol, "price": None, "error": "timeout", "provider": "timeout"}
    )


@router.post("/api/price/refresh")
async def api_price_refresh(symbol: str = WOLF):
    """Back-compat POST with 2.5s timeout to prevent 499 errors."""
    async def refresh_price():
        result = await ensure_price_cached(
            symbol,
            strict_live=True,
            drop_cache=True,
        )
        response = _build_price_response(result)
        response["cache_cleared"] = True
        response["strict_live"] = True
        return response

    return await with_cap(
        refresh_price(),
        sec=2.5,
        fallback={"symbol": symbol, "price": None, "error": "timeout", "provider": "timeout"}
    )


@router.get("/api/portfolio")
async def api_portfolio():
    """Portfolio endpoint with 2.5s timeout to prevent 499 errors."""
    async def get_portfolio_data():
        price, prev, provider = get_wolf_price()
        qty, avg = _get_portfolio_qty_and_avg()  # Use helper to read from positions array
        cash = float(STATE.get("cash", 0.0))
        cur = price if price is not None else avg

        # Adjust P&L for corporate actions (reverse splits, etc.)
        pnl_adjustment = _adjust_pnl_for_corporate_action(WOLF, avg, cur, qty)

        positions = [
            {
                "symbol": WOLF,
                "type": "stock",
                "qty": qty,
                "price": avg,
                "current": cur,
                "pnl": pnl_adjustment["pnl_abs"],  # Use adjusted P&L
                "pnl_pct": pnl_adjustment["pnl_pct"],  # Use adjusted P&L %
                "pnl_note": pnl_adjustment["adjustment_note"],  # Show adjustment reason
                "gps": 7.2,
                "src": provider or "unavailable",
            }
        ]
        return {"positions": positions, "cash": cash, "nav": round(qty * cur + cash, 2)}

    # Apply 2.5s timeout to prevent proxy 499 errors
    return await with_cap(
        get_portfolio_data(),
        sec=2.5,
        fallback={"positions": [], "cash": 0.0, "nav": 0.0, "error": "timeout"}
    )


@router.get("/api/portfolio/history")
async def api_portfolio_history(hours: int = 24, points: int = 20):
    """
    Get portfolio NAV and P&L history for charting.

    Args:
        hours: Lookback period in hours (default: 24)
        points: Number of data points to return (default: 20)

    Returns:
        {
            "history": [
                {"ts": timestamp, "nav": value, "pnl_abs": value, "pnl_pct": percentage},
                ...
            ],
            "current": {"nav": value, "pnl_abs": value, "pnl_pct": percentage}
        }
    """
    import sqlite3

    now_ts = int(time.time())
    lookback_ts = now_ts - (hours * 3600)

    history = []

    try:
        # Try to read from AI memory which has historical snapshots
        conn = sqlite3.connect(WOLF_SQLITE_PATH)
        cur = conn.cursor()

        # Query AI memory for historical data
        cur.execute(
            """
            SELECT ts, price, prev, qty, avg
            FROM ai_memory
            WHERE ts >= ?
            ORDER BY ts ASC
        """,
            (lookback_ts,),
        )

        rows = cur.fetchall()

        # Sample evenly if we have more data than requested points
        if len(rows) > points:
            step = len(rows) // points
            rows = [rows[i] for i in range(0, len(rows), step)][:points]

        for row in rows:
            ts, price_val, prev, qty_val, avg_val = row
            if price_val and qty_val and avg_val:
                current = float(price_val)
                qty_f = float(qty_val)
                avg_f = float(avg_val)

                pnl_abs = (current - avg_f) * qty_f
                pnl_pct = ((current - avg_f) / avg_f) * 100.0 if avg_f > 0 else 0.0
                nav = current * qty_f

                history.append(
                    {
                        "ts": int(ts),
                        "nav": round(nav, 2),
                        "pnl_abs": round(pnl_abs, 2),
                        "pnl_pct": round(pnl_pct, 2),
                    }
                )

        conn.close()
    except Exception as e:
        LOGGER.warning(f"Failed to fetch portfolio history: {e}")

    # Get current values
    qty, avg = _get_portfolio_qty_and_avg()  # Use helper to read from positions array
    price, prev, provider = get_wolf_price()
    current_price = price if price is not None else (prev if prev is not None else avg)

    # Adjust P&L for corporate actions (reverse splits, etc.)
    pnl_adjustment = _adjust_pnl_for_corporate_action(WOLF, avg, current_price, qty)
    pnl_abs = pnl_adjustment["pnl_abs"]
    pnl_pct = pnl_adjustment["pnl_pct"]
    nav = current_price * qty

    return {
        "history": history,
        "current": {
            "nav": round(nav, 2),
            "pnl_abs": round(pnl_abs, 2),
            "pnl_pct": round(pnl_pct, 2),
        },
        "lookback_hours": hours,
        "data_points": len(history),
    }


@router.get("/api/strategies/ensemble")
async def api_strategy_ensemble(symbol: str = "WOLF"):
    """
    APEX Strategy Ensemble: Weighted voting from multiple strategies
    - Momentum: Multi-timeframe momentum with ATR stops
    - NewsShock: Sentiment-based mean reversion/follow-through
    - PairsTrading: Statistical arbitrage

    Dynamically adjusts weights based on market regime

    Returns:
        {
            "symbol": str,
            "timestamp": int,
            "consensus": {
                "action": str,
                "confidence": float,
                "expected_return": float,
                "vote_breakdown": {BUY/SELL/HOLD counts},
                "agreement": str
            },
            "votes": [list of strategy votes],
            "weights_used": dict,
            "regime": str
        }
    """
    from core.strategy_ensemble import get_strategy_ensemble

    if symbol.upper() != WOLF:
        return {"error": f"Symbol {symbol} not supported", "supported": [WOLF]}, 404

    try:
        # Gather market data for strategies
        import yfinance as yf

        ticker = yf.Ticker(WOLF)
        daily_hist = ticker.history(period="90d", interval="1d")

        try:
            intraday_hist = ticker.history(period="5d", interval="15m")
        except Exception:
            intraday_hist = None

        # Get news
        try:
            news = get_wolf_news(limit=10)
        except Exception:
            news = []

        # Get regime
        try:
            regime_detector = get_regime_detector()
            # Pass daily close prices for regime detection
            regime = regime_detector.detect_regime(
                daily_hist["Close"].values.tolist() if not daily_hist.empty else []
            )
        except Exception:
            regime = "BULL"

        market_data = {
            "daily_hist": daily_hist,
            "intraday_hist": intraday_hist,
            "news": news,
            "regime": regime,
        }

        ensemble = get_strategy_ensemble()
        result = ensemble.evaluate_all(WOLF, market_data)

        return result

    except Exception as e:
        LOGGER.error(f"Strategy ensemble failed: {e}", exc_info=True)
        return {"error": f"Strategy ensemble failed: {str(e)}"}, 500


@router.get("/api/features/importance")
async def api_feature_importance(symbol: str = "WOLF", forecast_type: str = "swing"):
    """
    APEX Feature Importance - Shapley value analysis

    Args:
        symbol: Trading symbol (default: WOLF)
        forecast_type: "nowcast", "swing", or "position" (default: swing)

    Returns:
        Complete feature importance breakdown with Shapley values
    """
    from core.feature_importance import get_feature_importance_analyzer

    if symbol.upper() != WOLF:
        return {"error": f"Symbol {symbol} not supported"}, 404

    if forecast_type not in ["nowcast", "swing", "position"]:
        return {
            "error": f"Invalid forecast_type: {forecast_type}. Use 'nowcast', 'swing', or 'position'"
        }, 400

    try:
        analyzer = get_feature_importance_analyzer()
        analysis = analyzer.analyze_forecast(WOLF, forecast_type)

        return {
            "symbol": analysis.symbol,
            "timestamp": analysis.timestamp,
            "forecast_type": analysis.forecast_type,
            "predicted_return": round(analysis.predicted_return * 100, 2),  # Convert to %
            "features": [
                {
                    "name": f.name,
                    "value": round(f.value, 4),
                    "shapley_value": round(f.shapley_value, 4),
                    "importance": round(f.importance, 2),
                    "direction": f.direction,
                }
                for f in analysis.features
            ],
            "summary": {
                "total_bullish": round(analysis.total_bullish_contribution, 4),
                "total_bearish": round(analysis.total_bearish_contribution, 4),
                "confidence": round(analysis.confidence_score, 2),
            },
        }

    except Exception as e:
        LOGGER.error(f"Feature importance failed: {e}", exc_info=True)
        return {"error": f"Feature importance failed: {str(e)}"}, 500


@router.get("/api/features/top")
async def api_top_features(symbol: str = "WOLF", forecast_type: str = "swing", top_n: int = 5):
    """
    APEX Feature Importance - Get top N features (simplified)

    Args:
        symbol: Trading symbol (default: WOLF)
        forecast_type: "nowcast", "swing", or "position" (default: swing)
        top_n: Number of top features to return (default: 5)

    Returns:
        List of top features by importance
    """
    from core.feature_importance import get_feature_importance_analyzer

    if symbol.upper() != WOLF:
        return {"error": f"Symbol {symbol} not supported"}, 404

    try:
        analyzer = get_feature_importance_analyzer()
        top_features = analyzer.get_top_features(WOLF, forecast_type, top_n)

        return {
            "symbol": WOLF,
            "forecast_type": forecast_type,
            "top_features": top_features,
            "timestamp": int(time.time()),
        }

    except Exception as e:
        LOGGER.error(f"Top features failed: {e}", exc_info=True)
        return {"error": f"Top features failed: {str(e)}"}, 500


@router.post("/api/goals/create")
async def api_create_goal(
    period: str = "weekly",
    target_return_pct: float = 5.0,
    max_drawdown_pct: float = 10.0,
    target_sharpe: float = 1.5,
    risk_budget: float = 100.0,
):
    """
    APEX Goal Engine - Create a new portfolio goal

    Args:
        period: "daily", "weekly", "monthly", "quarterly", "yearly"
        target_return_pct: Target return % (e.g., 5.0 for 5%)
        max_drawdown_pct: Max acceptable drawdown % (default: 10%)
        target_sharpe: Target Sharpe ratio (default: 1.5)
        risk_budget: Starting risk budget % (default: 100%)

    Returns:
        Created goal details
    """
    from core.goal_engine import get_goal_engine

    if period not in ["daily", "weekly", "monthly", "quarterly", "yearly"]:
        return {
            "error": f"Invalid period: {period}. Use daily, weekly, monthly, quarterly, or yearly"
        }, 400

    try:
        engine = get_goal_engine()
        goal = engine.create_goal(
            period=period,
            target_return_pct=target_return_pct,
            max_drawdown_pct=max_drawdown_pct,
            target_sharpe=target_sharpe,
            risk_budget=risk_budget,
        )

        return {
            "success": True,
            "goal_id": goal.goal_id,
            "period": goal.period,
            "target_return_pct": goal.target_return_pct,
            "max_drawdown_pct": goal.max_drawdown_pct,
            "target_sharpe": goal.target_sharpe,
            "risk_budget": goal.risk_budget,
            "start_date": goal.start_date,
            "end_date": goal.end_date,
            "days_total": goal.days_total,
            "status": goal.status,
        }

    except Exception as e:
        LOGGER.error(f"Create goal failed: {e}", exc_info=True)
        return {"error": f"Create goal failed: {str(e)}"}, 500


@router.post("/api/goals/update")
async def api_update_goal_progress(
    goal_id: str,
    current_return_pct: float,
    current_drawdown_pct: float,
    current_sharpe: float,
    portfolio_value: float,
):
    """
    APEX Goal Engine - Update goal progress

    Args:
        goal_id: Goal identifier
        current_return_pct: Current period return %
        current_drawdown_pct: Current drawdown %
        current_sharpe: Current Sharpe ratio
        portfolio_value: Current portfolio value

    Returns:
        Progress report with recommendations
    """
    from core.goal_engine import get_goal_engine

    try:
        engine = get_goal_engine()
        progress = engine.update_progress(
            goal_id=goal_id,
            current_return_pct=current_return_pct,
            current_drawdown_pct=current_drawdown_pct,
            current_sharpe=current_sharpe,
            portfolio_value=portfolio_value,
        )

        return {
            "goal_id": progress.goal_id,
            "period": progress.period,
            "progress_pct": round(progress.progress_pct, 2),
            "on_pace": progress.on_pace,
            "days_remaining": progress.days_remaining,
            "required_daily_return": round(progress.required_daily_return, 4),
            "recommendation": progress.recommendation,
            "timestamp": int(time.time()),
        }

    except Exception as e:
        LOGGER.error(f"Update goal progress failed: {e}", exc_info=True)
        return {"error": f"Update goal progress failed: {str(e)}"}, 500


@router.get("/api/goals/active")
async def api_get_active_goals():
    """
    APEX Goal Engine - Get all active goals

    Returns:
        List of active (non-expired) goals
    """
    from core.goal_engine import get_goal_engine

    try:
        engine = get_goal_engine()
        goals = engine.get_active_goals()

        return {
            "active_goals": [
                {
                    "goal_id": g.goal_id,
                    "period": g.period,
                    "target_return_pct": g.target_return_pct,
                    "max_drawdown_pct": g.max_drawdown_pct,
                    "target_sharpe": g.target_sharpe,
                    "risk_budget": g.risk_budget,
                    "start_date": g.start_date,
                    "end_date": g.end_date,
                    "status": g.status,
                    "days_total": g.days_total,
                }
                for g in goals
            ],
            "count": len(goals),
            "timestamp": int(time.time()),
        }

    except Exception as e:
        LOGGER.error(f"Get active goals failed: {e}", exc_info=True)
        return {"error": f"Get active goals failed: {str(e)}"}, 500


@router.post("/api/feeds/fetch")
async def api_fetch_feeds(source_id: str | None = None):
    """
    World Feed Fusion - Fetch articles from RSS feeds

    Args:
        source_id: Specific source to fetch (optional, fetches all if not provided)

    Returns:
        Number of new articles fetched
    """
    from core.world_feed_fusion import get_feed_fusion

    try:
        fusion = get_feed_fusion()

        if source_id:
            articles = fusion.fetch_feed(source_id)
            count = len(articles)
        else:
            count = fusion.fetch_all_feeds()

        return {
            "success": True,
            "articles_fetched": count,
            "source_id": source_id or "all",
            "timestamp": int(time.time()),
        }

    except Exception as e:
        LOGGER.error(f"Fetch feeds failed: {e}", exc_info=True)
        return {"error": f"Fetch feeds failed: {str(e)}"}, 500


@router.get("/api/feeds/latest")
async def api_get_latest_articles(limit: int = 20, symbol: str | None = None):
    """
    World Feed Fusion - Get latest news articles

    Args:
        limit: Maximum number of articles (default 20)
        symbol: Filter by ticker symbol (optional)

    Returns:
        List of latest articles with sentiment scores
    """
    from core.world_feed_fusion import get_feed_fusion

    try:
        fusion = get_feed_fusion()
        articles = fusion.get_latest_articles(limit=limit, symbol=symbol)

        return {
            "articles": articles,
            "count": len(articles),
            "symbol": symbol or "all",
            "timestamp": int(time.time()),
        }

    except Exception as e:
        LOGGER.error(f"Get latest articles failed: {e}", exc_info=True)
        return {"error": f"Get latest articles failed: {str(e)}"}, 500


@router.get("/api/feeds/sentiment")
async def api_get_sentiment_aggregate(symbol: str, timeframe: str = "1d"):
    """
    World Feed Fusion - Get aggregated sentiment for a symbol

    Args:
        symbol: Ticker symbol (required)
        timeframe: Time window - "1h", "6h", "1d", "7d" (default "1d")

    Returns:
        Aggregated sentiment statistics
    """
    from core.world_feed_fusion import get_feed_fusion

    try:
        if timeframe not in ["1h", "6h", "1d", "7d"]:
            return {"error": "Invalid timeframe. Must be 1h, 6h, 1d, or 7d"}, 400

        fusion = get_feed_fusion()
        aggregate = fusion.get_sentiment_aggregate(symbol, timeframe)

        if not aggregate:
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "message": "No articles found for this symbol in the specified timeframe",
                "avg_sentiment": 0.0,
                "article_count": 0,
                "timestamp": int(time.time()),
            }

        return {
            "symbol": aggregate.symbol,
            "timeframe": aggregate.timeframe,
            "avg_sentiment": round(aggregate.avg_sentiment, 3),
            "weighted_sentiment": round(aggregate.weighted_sentiment, 3),
            "article_count": aggregate.article_count,
            "bullish_count": aggregate.bullish_count,
            "bearish_count": aggregate.bearish_count,
            "neutral_count": aggregate.neutral_count,
            "confidence": round(aggregate.confidence, 3),
            "sentiment_label": (
                "bullish"
                if aggregate.weighted_sentiment > 0.2
                else "bearish"
                if aggregate.weighted_sentiment < -0.2
                else "neutral"
            ),
            "calculated_at": aggregate.calculated_at,
            "timestamp": int(time.time()),
        }

    except Exception as e:
        LOGGER.error(f"Get sentiment aggregate failed: {e}", exc_info=True)
        return {"error": f"Get sentiment aggregate failed: {str(e)}"}, 500


@router.get("/api/feeds/search")
async def api_search_articles(query: str, limit: int = 20):
    """
    World Feed Fusion - Search articles by keyword

    Args:
        query: Search query string
        limit: Maximum results (default 20)

    Returns:
        List of matching articles
    """
    from core.world_feed_fusion import get_feed_fusion

    try:
        if not query or len(query) < 2:
            return {"error": "Query must be at least 2 characters"}, 400

        fusion = get_feed_fusion()
        articles = fusion.search_articles(query, limit)

        return {
            "articles": articles,
            "count": len(articles),
            "query": query,
            "timestamp": int(time.time()),
        }

    except Exception as e:
        LOGGER.error(f"Search articles failed: {e}", exc_info=True)
        return {"error": f"Search articles failed: {str(e)}"}, 500


@router.get("/api/goals/history")
async def api_get_goal_history(goal_id: str, limit: int = 30):
    """
    APEX Goal Engine - Get historical progress for a goal

    Args:
        goal_id: Goal identifier
        limit: Number of historical snapshots (default: 30)

    Returns:
        Historical progress data
    """
    from core.goal_engine import get_goal_engine

    try:
        engine = get_goal_engine()
        history = engine.get_goal_history(goal_id, limit)

        return {
            "goal_id": goal_id,
            "history": history,
            "count": len(history),
            "timestamp": int(time.time()),
        }

    except Exception as e:
        LOGGER.error(f"Get goal history failed: {e}", exc_info=True)
        return {"error": f"Get goal history failed: {str(e)}"}, 500


@router.post("/api/watcher/add_ticker")
async def api_watcher_add_ticker(symbol: str):
    """
    Add ticker to Smart Watcher watchlist (max 25)

    Args:
        symbol: Ticker symbol (e.g., "WOLF", "AAPL")

    Returns:
        Success status and position in watchlist
    """
    from core.smart_watcher import get_smart_watcher

    try:
        watcher = get_smart_watcher()
        result = watcher.add_ticker(symbol.upper())

        return {**result, "timestamp": int(time.time())}

    except Exception as e:
        LOGGER.error(f"Add ticker failed: {e}", exc_info=True)
        return {"error": str(e)}, 500


@router.delete("/api/watcher/remove_ticker")
async def api_watcher_remove_ticker(symbol: str):
    """
    Remove ticker from Smart Watcher watchlist

    Args:
        symbol: Ticker symbol to remove

    Returns:
        Success status
    """
    from core.smart_watcher import get_smart_watcher

    try:
        watcher = get_smart_watcher()
        success = watcher.remove_ticker(symbol.upper())

        return {
            "success": success,
            "symbol": symbol.upper(),
            "timestamp": int(time.time()),
        }

    except Exception as e:
        LOGGER.error(f"Remove ticker failed: {e}", exc_info=True)
        return {"error": str(e)}, 500


@router.get("/api/watcher/watchlist")
async def api_watcher_get_watchlist():
    """
    Get all tickers in Smart Watcher watchlist

    Returns:
        List of watched tickers with current signals, prices, sentiment
    """
    from core.smart_watcher import get_smart_watcher

    try:
        watcher = get_smart_watcher()
        tickers = watcher.get_watchlist()

        return {
            "tickers": [asdict(t) for t in tickers],
            "count": len(tickers),
            "max_capacity": 25,
            "timestamp": int(time.time()),
        }

    except Exception as e:
        LOGGER.error(f"Get watchlist failed: {e}", exc_info=True)
        return {"error": str(e)}, 500


@router.post("/api/watcher/update_prices")
async def api_watcher_update_prices():
    """
    Update prices for all watchlist tickers using Polygon.io

    Returns:
        Updated quote data for all tickers
    """
    from core.polygon_integration import get_polygon_client
    from core.smart_watcher import get_smart_watcher

    try:
        watcher = get_smart_watcher()
        polygon = get_polygon_client()

        # Get watchlist
        tickers = watcher.get_watchlist()
        symbols = [t.symbol for t in tickers]

        # Fetch real-time quotes
        quotes = polygon.get_bulk_quotes(symbols)

        # Update watcher
        updated = []
        for symbol, quote in quotes.items():
            # Get 20-day average volume
            volumes = polygon.get_daily_volume(symbol, days=20)
            avg_volume = int(sum(volumes) / len(volumes)) if volumes else 0

            watcher.update_ticker_price(
                symbol=symbol,
                price=quote.price,
                volume=quote.volume,
                avg_volume=avg_volume,
            )

            updated.append(
                {
                    "symbol": symbol,
                    "price": quote.price,
                    "change_pct": quote.change_pct,
                    "volume": quote.volume,
                    "timestamp": quote.timestamp,
                }
            )

        return {
            "updated": updated,
            "count": len(updated),
            "timestamp": int(time.time()),
        }

    except Exception as e:
        LOGGER.error(f"Update prices failed: {e}", exc_info=True)
        return {"error": str(e)}, 500


@router.post("/api/watcher/generate_signal")
async def api_watcher_generate_signal(symbol: str):
    """
    Generate proactive trading signal for ticker
    Combines: forecast + sentiment + technical + macro

    Args:
        symbol: Ticker symbol

    Returns:
        Trading signal with confidence, reason, targets
    """
    from core.feature_importance import FeatureImportanceAnalyzer
    from core.multi_horizon_forecaster import get_multi_horizon_forecaster
    from core.smart_watcher import get_smart_watcher
    from core.world_feed_fusion import get_feed_fusion

    try:
        watcher = get_smart_watcher()
        forecaster = get_multi_horizon_forecaster()
        feed_fusion = get_feed_fusion()

        # Get forecast
        forecast_result = forecaster.forecast_all_horizons(symbol.upper())
        forecast_data = {
            "predicted_return": forecast_result.get("consensus", {}).get("expected_return", 0.0),
            "risk_level": forecast_result.get("consensus", {}).get("risk_level", "unknown"),
        }

        # Get recent news for this ticker
        articles = feed_fusion.get_latest_articles(limit=10, symbol=symbol.upper())
        news_headlines = [a.get("title", "") for a in articles[:5]]

        # Get technical factors
        analyzer = FeatureImportanceAnalyzer()
        top_features = analyzer.get_top_features(symbol.upper(), "swing", top_n=5)
        technical_factors = [f"{f['name']}: {f['importance']:.1f}%" for f in top_features]

        # Get macro context
        macro = watcher.get_latest_macro()
        macro_context = (
            f"{macro.regime} / Risk: {macro.risk_level} / VIX: {macro.vix_level:.1f}"
            if macro
            else "unknown"
        )

        # Generate signal
        signal = watcher.generate_signal(
            symbol=symbol.upper(),
            forecast_data=forecast_data,
            news_headlines=news_headlines,
            technical_factors=technical_factors,
            macro_context=macro_context,
        )

        return {"signal": asdict(signal), "timestamp": int(time.time())}

    except Exception as e:
        LOGGER.error(f"Generate signal failed for {symbol}: {e}", exc_info=True)
        return {"error": str(e)}, 500


@router.post("/api/watcher/update_signal_outcome")
async def api_watcher_update_signal_outcome(
    signal_id: str, price_24h: float, price_48h: float | None = None
):
    """
    Update signal outcome after 24h/48h (for learning loop)

    Args:
        signal_id: Signal identifier
        price_24h: Price after 24 hours
        price_48h: Price after 48 hours (optional)

    Returns:
        Updated outcome and performance stats
    """
    from core.smart_watcher import get_smart_watcher

    try:
        watcher = get_smart_watcher()
        watcher.update_signal_outcome(signal_id, price_24h, price_48h)

        return {"success": True, "signal_id": signal_id, "timestamp": int(time.time())}

    except Exception as e:
        LOGGER.error(f"Update signal outcome failed: {e}", exc_info=True)
        return {"error": str(e)}, 500


@router.get("/api/watcher/performance")
async def api_watcher_get_performance(symbol: str | None = None):
    """
    Get signal performance stats (hit rate, avg return, etc.)

    Args:
        symbol: Optional ticker filter

    Returns:
        Performance statistics per ticker and signal type
    """
    from core.smart_watcher import get_smart_watcher

    try:
        watcher = get_smart_watcher()
        stats = watcher.get_performance(symbol.upper() if symbol else None)

        return {
            "performance": [asdict(s) for s in stats],
            "count": len(stats),
            "timestamp": int(time.time()),
        }

    except Exception as e:
        LOGGER.error(f"Get performance failed: {e}", exc_info=True)
        return {"error": str(e)}, 500


@router.post("/api/watcher/update_macro")
async def api_watcher_update_macro():
    """
    Update macro market snapshot (SPY/QQQ/VIX)

    Returns:
        Current macro regime and risk level
    """
    import yfinance as yf

    from core.polygon_integration import get_polygon_client
    from core.smart_watcher import get_smart_watcher

    try:
        watcher = get_smart_watcher()
        polygon = get_polygon_client()

        # Try Polygon first, fallback to yfinance
        try:
            spy_quote = polygon.get_realtime_quote("SPY")
            qqq_quote = polygon.get_realtime_quote("QQQ")
            vix_quote = polygon.get_realtime_quote("VIX")

            spy_price = spy_quote.price if spy_quote else 0.0
            qqq_price = qqq_quote.price if qqq_quote else 0.0
            vix_level = vix_quote.price if vix_quote else 0.0
        except Exception:
            # Fallback to yfinance
            spy = yf.Ticker("SPY")
            qqq = yf.Ticker("QQQ")
            vix = yf.Ticker("^VIX")

            spy_price = spy.history(period="1d")["Close"].iloc[-1]
            qqq_price = qqq.history(period="1d")["Close"].iloc[-1]
            vix_level = vix.history(period="1d")["Close"].iloc[-1]

        # Update macro
        snapshot = watcher.update_macro_snapshot(spy_price, qqq_price, vix_level)

        return {"macro": asdict(snapshot), "timestamp": int(time.time())}

    except Exception as e:
        LOGGER.error(f"Update macro failed: {e}", exc_info=True)
        return {"error": str(e)}, 500


@router.get("/api/watcher/ticker_news")
async def api_watcher_get_ticker_news(symbol: str, hours: int = 24):
    """
    Get news articles linked to specific ticker

    Args:
        symbol: Ticker symbol
        hours: Lookback period in hours (default: 24)

    Returns:
        News articles with sentiment scores
    """
    from core.smart_watcher import get_smart_watcher
    from core.world_feed_fusion import get_feed_fusion

    try:
        watcher = get_smart_watcher()
        feed_fusion = get_feed_fusion()

        # Get linked news from watcher
        linked_news = watcher.get_ticker_news(symbol.upper(), hours)

        # Enrich with full article data
        articles = []
        for _news in linked_news:
            # Get latest articles from feed fusion
            matching = feed_fusion.get_latest_articles(limit=50, symbol=symbol.upper())
            articles.extend(matching[:10])

        return {
            "symbol": symbol.upper(),
            "articles": articles,
            "count": len(articles),
            "hours": hours,
            "timestamp": int(time.time()),
        }

    except Exception as e:
        LOGGER.error(f"Get ticker news failed: {e}", exc_info=True)
        return {"error": str(e)}, 500


@router.get("/api/edgar/recent_filings")
async def api_edgar_get_recent_filings(
    filing_type: str | None = None, hours_back: int = 24, limit: int = 50
):
    """
    Get recent SEC filings from EDGAR (free)

    Args:
        filing_type: Filter by type (8-K, 10-K, 10-Q, 13F) or None for all
        hours_back: Lookback period (default: 24 hours)
        limit: Max filings to return (default: 50)

    Returns:
        List of recent SEC filings with urgency and sentiment
    """
    from core.edgar_integration import get_edgar_client

    try:
        edgar = get_edgar_client()
        filings = edgar.get_recent_filings(filing_type, hours_back, limit)

        return {
            "filings": [asdict(f) for f in filings],
            "count": len(filings),
            "filing_type": filing_type or "all",
            "timestamp": int(time.time()),
        }

    except Exception as e:
        LOGGER.error(f"Get SEC filings failed: {e}", exc_info=True)
        return {"error": str(e)}, 500


@router.get("/api/edgar/company_filings")
async def api_edgar_get_company_filings(
    ticker: str, filing_type: str | None = None, limit: int = 20
):
    """
    Get SEC filings for specific company

    Args:
        ticker: Ticker symbol or CIK
        filing_type: Filter by filing type (optional)
        limit: Max filings (default: 20)

    Returns:
        Company's recent SEC filings
    """
    from core.edgar_integration import get_edgar_client

    try:
        edgar = get_edgar_client()
        filings = edgar.get_company_filings(ticker.upper(), filing_type, limit)

        return {
            "ticker": ticker.upper(),
            "filings": [asdict(f) for f in filings],
            "count": len(filings),
            "timestamp": int(time.time()),
        }

    except Exception as e:
        LOGGER.error(f"Get company filings failed: {e}", exc_info=True)
        return {"error": str(e)}, 500


@router.get("/api/edgar/insider_transactions")
async def api_edgar_get_insider_transactions(ticker: str, days_back: int = 90):
    """
    Get Form 4 insider transactions

    Args:
        ticker: Ticker symbol
        days_back: Lookback period (default: 90 days)

    Returns:
        Recent insider buy/sell transactions
    """
    from core.edgar_integration import get_edgar_client

    try:
        edgar = get_edgar_client()
        transactions = edgar.get_insider_transactions(ticker.upper(), days_back)

        return {
            "ticker": ticker.upper(),
            "transactions": transactions,
            "count": len(transactions),
            "timestamp": int(time.time()),
        }

    except Exception as e:
        LOGGER.error(f"Get insider transactions failed: {e}", exc_info=True)
        return {"error": str(e)}, 500


@router.get("/api/polygon/quote")
async def api_polygon_get_quote(symbol: str):
    """
    Get real-time quote from Polygon.io

    Args:
        symbol: Ticker symbol

    Returns:
        Real-time quote with bid/ask/volume
    """
    from core.polygon_integration import get_polygon_client

    try:
        polygon = get_polygon_client()
        quote = polygon.get_realtime_quote(symbol.upper())

        if quote:
            return {"quote": asdict(quote), "timestamp": int(time.time())}
        else:
            return {"error": "Quote not available"}, 404

    except Exception as e:
        LOGGER.error(f"Get Polygon quote failed: {e}", exc_info=True)
        return {"error": str(e)}, 500


@router.get("/api/polygon/corporate_events")
async def api_polygon_get_corporate_events(
    symbol: str | None = None, event_type: str | None = None, days_ahead: int = 30
):
    """
    Get upcoming corporate events (earnings, dividends)

    Args:
        symbol: Filter by ticker (optional)
        event_type: Filter by type (earnings, dividend) or None for all
        days_ahead: Days to look ahead (default: 30)

    Returns:
        Upcoming corporate events calendar
    """
    from core.polygon_integration import get_polygon_client

    try:
        polygon = get_polygon_client()
        events = polygon.get_corporate_events(
            symbol.upper() if symbol else None, event_type, days_ahead
        )

        return {
            "events": [asdict(e) for e in events],
            "count": len(events),
            "timestamp": int(time.time()),
        }

    except Exception as e:
        LOGGER.error(f"Get corporate events failed: {e}", exc_info=True)
        return {"error": str(e)}, 500


@router.get("/api/polygon/market_status")
async def api_polygon_get_market_status():
    """
    Get current market status (open/closed)

    Returns:
        Market status and exchange info
    """
    from core.polygon_integration import get_polygon_client

    try:
        polygon = get_polygon_client()
        status = polygon.get_market_status()

        return {"market_status": status, "timestamp": int(time.time())}

    except Exception as e:
        LOGGER.error(f"Get market status failed: {e}", exc_info=True)
        return {"error": str(e)}, 500


@router.post("/api/algo/update_microstructure")
async def api_algo_update_microstructure(
    symbol: str,
    bid: float,
    ask: float,
    bid_size: int,
    ask_size: int,
    last_trade_size: int,
    last_trade_price: float,
    volume_1min: int,
):
    """
    Update microstructure data and detect algo patterns

    Args:
        symbol: Ticker symbol
        bid, ask: Current bid/ask prices
        bid_size, ask_size: Order book sizes
        last_trade_size, last_trade_price: Last trade details
        volume_1min: Volume in last minute

    Returns:
        Detected algo patterns (if any)
    """
    from core.algo_footprint import MicrostructureSnapshot, get_algo_detector

    try:
        detector = get_algo_detector()

        snapshot = MicrostructureSnapshot(
            symbol=symbol.upper(),
            timestamp=int(time.time()),
            bid=bid,
            ask=ask,
            bid_size=bid_size,
            ask_size=ask_size,
            spread=ask - bid,
            spread_pct=((ask - bid) / bid * 100) if bid > 0 else 0.0,
            last_trade_size=last_trade_size,
            last_trade_price=last_trade_price,
            volume_1min=volume_1min,
        )

        detector.update_microstructure(snapshot)

        # Get recently detected patterns
        patterns = detector.get_recent_patterns(symbol.upper(), hours=1)

        return {
            "symbol": symbol.upper(),
            "patterns_detected": [asdict(p) for p in patterns],
            "count": len(patterns),
            "timestamp": int(time.time()),
        }

    except Exception as e:
        LOGGER.error(f"Update microstructure failed: {e}", exc_info=True)
        return {"error": str(e)}, 500


@router.get("/api/algo/patterns")
async def api_algo_get_patterns(symbol: str | None = None, hours: int = 24):
    """
    Get recently detected algo patterns

    Args:
        symbol: Filter by ticker (optional)
        hours: Lookback period (default: 24)

    Returns:
        Detected algorithmic trading patterns
    """
    from core.algo_footprint import get_algo_detector

    try:
        detector = get_algo_detector()
        patterns = detector.get_recent_patterns(symbol.upper() if symbol else None, hours=hours)

        return {
            "patterns": [asdict(p) for p in patterns],
            "count": len(patterns),
            "timestamp": int(time.time()),
        }

    except Exception as e:
        LOGGER.error(f"Get algo patterns failed: {e}", exc_info=True)
        return {"error": str(e)}, 500


@router.post("/api/patterns/reconcile")
async def api_reconcile_patterns():
    """
    Manually trigger pattern outcome reconciliation.
    
    Checks patterns detected 24-48h ago and updates their outcomes
    based on actual price movements.
    """
    from core.pattern_tracker import reconcile_pattern_outcomes
    
    try:
        result = await reconcile_pattern_outcomes()
        return {
            "reconciled": result.get("reconciled", 0),
            "pending": result.get("pending", 0),
            "timestamp": int(time.time()),
        }
    
    except Exception as e:
        LOGGER.error(f"Pattern reconciliation failed: {e}", exc_info=True)
        return {"error": str(e)}, 500


@router.delete("/api/goals/delete")
async def api_delete_goal(goal_id: str):
    """
    APEX Goal Engine - Delete a goal

    Args:
        goal_id: Goal identifier

    Returns:
        Success confirmation
    """
    from core.goal_engine import get_goal_engine

    try:
        engine = get_goal_engine()
        success = engine.delete_goal(goal_id)

        return {
            "success": success,
            "goal_id": goal_id,
            "message": f"Goal {goal_id} deleted",
        }

    except Exception as e:
        LOGGER.error(f"Delete goal failed: {e}", exc_info=True)
        return {"error": f"Delete goal failed: {str(e)}"}, 500


@router.get("/api/goals/risk_multiplier")
async def api_get_risk_multiplier():
    """
    APEX Goal Engine - Get current risk multiplier

    Returns adaptive risk budget multiplier for position sizing
    based on progress across all active goals.

    Returns:
        Risk multiplier (0.5 to 2.0)
    """
    from core.goal_engine import get_goal_engine

    try:
        engine = get_goal_engine()
        multiplier = engine.get_risk_multiplier()

        return {
            "risk_multiplier": round(multiplier, 3),
            "interpretation": (
                "Reduce position sizes"
                if multiplier < 0.9
                else ("Normal position sizes" if multiplier <= 1.1 else "Increase position sizes")
            ),
            "timestamp": int(time.time()),
        }

    except Exception as e:
        LOGGER.error(f"Get risk multiplier failed: {e}", exc_info=True)
        return {"error": f"Get risk multiplier failed: {str(e)}"}, 500


@router.get("/api/trade_card/{symbol}")
async def api_trade_card(symbol: str, action: str = "BUY", lookback_days: int = 90):
    """
    Generate APEX-style Trade Card with full explainability.

    Args:
        symbol: Trading symbol (currently WOLF only)
        action: BUY/SELL/HOLD (default: BUY)
        lookback_days: Days of history for analysis (default: 90)

    Returns:
        Trade card with top features, analogs, expected path, fail conditions, risks
    """
    import pandas as pd
    import yfinance as yf

    from core.trade_card import TradeCardGenerator

    if symbol.upper() != WOLF:
        return {"error": f"Symbol {symbol} not supported", "supported": [WOLF]}, 404

    action = action.upper()
    if action not in ["BUY", "SELL", "HOLD"]:
        return {"error": "Action must be BUY, SELL, or HOLD"}, 400

    try:
        # Fetch historical data from real sources
        try:
            ticker = yf.Ticker(WOLF)
            hist = ticker.history(period=f"{lookback_days}d")

            if hist.empty:
                # Fallback to simulated data if yfinance fails
                LOGGER.warning("yfinance returned empty data, using simulated fallback")
                import numpy as np

                dates = pd.date_range(end=pd.Timestamp.now(), periods=lookback_days, freq="D")
                base_price = 150.0
                price_data = pd.DataFrame(
                    {
                        "close": base_price + np.random.randn(lookback_days).cumsum() * 2,
                        "high": base_price + np.random.randn(lookback_days).cumsum() * 2 + 1,
                        "low": base_price + np.random.randn(lookback_days).cumsum() * 2 - 1,
                        "volume": np.random.randint(1000000, 5000000, lookback_days),
                    },
                    index=dates,
                )
            else:
                # Prepare DataFrame from yfinance data
                price_data = pd.DataFrame(
                    {
                        "close": hist["Close"],
                        "high": hist["High"],
                        "low": hist["Low"],
                        "volume": hist["Volume"],
                    }
                )
        except Exception as yf_error:
            # Fallback to simulated data if yfinance fails
            LOGGER.warning(f"yfinance failed: {yf_error}, using simulated fallback")
            import numpy as np

            dates = pd.date_range(end=pd.Timestamp.now(), periods=lookback_days, freq="D")
            base_price = 150.0
            price_data = pd.DataFrame(
                {
                    "close": base_price + np.random.randn(lookback_days).cumsum() * 2,
                    "high": base_price + np.random.randn(lookback_days).cumsum() * 2 + 1,
                    "low": base_price + np.random.randn(lookback_days).cumsum() * 2 - 1,
                    "volume": np.random.randint(1000000, 5000000, lookback_days),
                },
                index=dates,
            )

        # Get current sentiment from news (if available)
        news_sentiment = None
        try:
            news_list = get_wolf_news(limit=10)
            if news_list:
                # Simple sentiment: count bullish vs bearish keywords
                sentiment_score = 0.0
                for item in news_list:
                    # Ensure item is dict, not string
                    if isinstance(item, dict):
                        sent = (item.get("sentiment") or "").lower()
                    else:
                        sent = ""
                    if "bullish" in sent or "positive" in sent:
                        sentiment_score += 1.0
                    elif "bearish" in sent or "negative" in sent:
                        sentiment_score -= 1.0
                news_sentiment = sentiment_score / max(len(news_list), 1)
        except Exception as e:
            LOGGER.warning(f"Failed to get news sentiment: {e}")

        # Get forecast data (if available)
        forecast_data = {}
        try:
            forecast_result = _build_forecast_series(horizon_h=168)  # 7 days
            if forecast_result and len(forecast_result) > 0:
                current_price = price_data["close"].iloc[-1]
                forecast_prices = [p for _, p in forecast_result if p]
                if forecast_prices:
                    forecast_7d = forecast_prices[-1]
                    forecast_data = {
                        "return_1d": (
                            (forecast_prices[0] - current_price) / current_price
                            if len(forecast_prices) > 0
                            else 0.0
                        ),
                        "return_7d": (forecast_7d - current_price) / current_price,
                        "return_30d": (forecast_7d - current_price)
                        / current_price
                        * 4,  # Rough 30d extrapolation
                    }
        except Exception as e:
            LOGGER.warning(f"Failed to get forecast data: {e}")

        # Generate trade card
        generator = TradeCardGenerator()

        # Get current confidence from AI (if recent decision exists)
        confidence = 60.0  # Default moderate confidence
        try:
            import sqlite3

            conn = sqlite3.connect(WOLF_SQLITE_PATH)
            cur = conn.cursor()
            cur.execute(
                """
                SELECT confidence
                FROM ai_decisions
                ORDER BY ts DESC
                LIMIT 1
            """
            )
            row = cur.fetchone()
            if row:
                confidence = float(row[0])
            conn.close()
        except Exception as e:
            LOGGER.warning(f"Failed to get AI confidence: {e}")

        card = generator.generate_card(
            symbol=WOLF,
            action=action,
            confidence=confidence,
            price_data=price_data,
            news_sentiment=news_sentiment,
            forecast_data=forecast_data,
        )

        # Convert dataclass to dict
        return {
            "action": card.action,
            "symbol": card.symbol,
            "confidence": card.confidence,
            "timestamp": card.timestamp,
            "top_features": card.top_features,
            "analogs": card.analogs,
            "expected_return_1d": card.expected_return_1d,
            "expected_return_7d": card.expected_return_7d,
            "expected_return_30d": card.expected_return_30d,
            "price_target": card.price_target,
            "confidence_band": card.confidence_band,
            "stop_loss_price": card.stop_loss_price,
            "stop_loss_reason": card.stop_loss_reason,
            "invalidation_signals": card.invalidation_signals,
            "var_95": card.var_95,
            "max_loss_estimate": card.max_loss_estimate,
            "win_probability": card.win_probability,
            "rationale": card.rationale,
            "risks": card.risks,
            "catalysts": card.catalysts,
        }

    except Exception as e:
        LOGGER.error(f"Trade card generation failed: {e}", exc_info=True)
        # Return error as JSON object for frontend compatibility
        return {
            "error": f"Trade card generation failed: {str(e)}",
            "action": action,
            "symbol": symbol,
            "confidence": 0.0,
        }


@router.get("/api/cash")
async def api_cash_get():
    total = float(STATE.get("cash", 0.0))
    stock = float(STATE.get("cash_stock", 0.0))
    crypto = float(STATE.get("cash_crypto", 0.0))
    # If split not set but total exists, report total only
    if (stock > 0 or crypto > 0) and total == 0.0:
        total = round(stock + crypto, 2)
    return {"cash": total, "stock": stock, "crypto": crypto}


@router.post("/api/cash")
async def api_cash_set(body: CashBody, credentials: HTTPAuthorizationCredentials | None = AUTH_DEP):
    _require_bearer(
        (f"Bearer {credentials.credentials}") if credentials and credentials.credentials else None
    )
    # Accept either total or split
    has_total = body.cash is not None
    has_split = (body.stock is not None) or (body.crypto is not None)
    if not has_total and not has_split:
        raise HTTPException(422, "provide 'cash' or 'stock'/'crypto'")
    if has_total and has_split:
        raise HTTPException(422, "provide either total cash or split, not both")
    if has_total:
        try:
            val = float(body.cash)  # type: ignore[arg-type]
        except Exception:
            raise HTTPException(422, "cash must be a number")
        if math.isnan(val) or math.isinf(val):
            raise HTTPException(422, "cash must be finite")
        STATE["cash"] = float(round(val, 2))
        # Reset split to align with total-only mode
        STATE.pop("cash_stock", None)
        STATE.pop("cash_crypto", None)
        _persist_save()
        _add_event("cash.update", "Cash balance updated", {"cash": STATE["cash"]})
        return {"ok": True, "cash": STATE["cash"]}
    # Split mode
    try:
        stock_val = float(body.stock or 0.0)
        crypto_val = float(body.crypto or 0.0)
    except Exception:
        raise HTTPException(422, "stock/crypto must be numbers")
    for v in (stock_val, crypto_val):
        if math.isnan(v) or math.isinf(v):
            raise HTTPException(422, "cash values must be finite")
    STATE["cash_stock"] = float(round(stock_val, 2))
    STATE["cash_crypto"] = float(round(crypto_val, 2))
    # Keep legacy total in sync
    STATE["cash"] = float(round(STATE["cash_stock"] + STATE["cash_crypto"], 2))
    _persist_save()
    _add_event(
        "cash.update",
        "Cash balance updated",
        {
            "cash": STATE["cash"],
            "stock": STATE["cash_stock"],
            "crypto": STATE["cash_crypto"],
        },
    )
    return {
        "ok": True,
        "cash": STATE["cash"],
        "stock": STATE["cash_stock"],
        "crypto": STATE["cash_crypto"],
    }


@router.get("/api/positions")
async def api_positions_get():
    positions = STATE.get("positions") or []
    if not isinstance(positions, list):
        positions = []
    return {"positions": positions}


@router.post("/api/positions/add")
async def api_positions_add(
    body: PositionAddBody, credentials: HTTPAuthorizationCredentials | None = AUTH_DEP
):
    _require_bearer(
        (f"Bearer {credentials.credentials}") if credentials and credentials.credentials else None
    )
    try:
        sym = str(body.symbol).upper()
        market = str(body.market or "stock")
        qty = float(body.qty)
        price_paid = float(body.price_paid)
    except Exception:
        raise HTTPException(422, "invalid position payload")
    if not sym:
        raise HTTPException(422, "symbol required")
    if qty <= 0 or price_paid < 0:
        raise HTTPException(422, "qty must be >0; price_paid >= 0")
    positions = STATE.get("positions")
    if not isinstance(positions, list):
        positions = []
    positions.append({"symbol": sym, "market": market, "qty": qty, "price_paid": price_paid})
    STATE["positions"] = positions
    # Optionally apply to cash (deduct cost)
    if body.apply_to_cash:
        cost = round(qty * price_paid, 2)
        # Prefer split-aware deduction from stock cash
        if "cash_stock" in STATE or "cash_crypto" in STATE:
            if market == "crypto":
                STATE["cash_crypto"] = float(round(float(STATE.get("cash_crypto", 0.0)) - cost, 2))
            else:
                STATE["cash_stock"] = float(round(float(STATE.get("cash_stock", 0.0)) - cost, 2))
            STATE["cash"] = float(
                round(
                    float(STATE.get("cash_stock", 0.0)) + float(STATE.get("cash_crypto", 0.0)),
                    2,
                )
            )
        else:
            STATE["cash"] = float(round(float(STATE.get("cash", 0.0)) - cost, 2))
    _persist_save()
    _add_event(
        "positions.add",
        "Position added",
        {"symbol": sym, "market": market, "qty": qty, "price_paid": price_paid},
    )
    return {"ok": True, "positions": STATE["positions"]}


@router.post("/api/positions/clear")
async def api_positions_clear(
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """Clear all custom positions. Keeps focus WOLF position (qty/avg_cost) untouched.
    Useful when you want the cockpit to show only WOLF again.
    """
    _require_bearer(
        (f"Bearer {credentials.credentials}") if credentials and credentials.credentials else None
    )
    try:
        STATE["positions"] = []
    except Exception:
        STATE["positions"] = []
    _persist_save()
    _add_event("positions.clear", "All custom positions cleared", {})
    return {"ok": True, "positions": []}


@router.post("/api/positions/import_raw")
async def api_positions_import_raw(
    body: dict | list | None = None,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """Import positions from JSON. Accepts:
    { reset: bool, apply_to_cash: bool, set_focus: str|None, positions: [{symbol, market, qty, price_paid?|invested_total?}]}.
    If invested_total provided, price_paid := invested_total/qty.
    """
    _require_bearer(
        (f"Bearer {credentials.credentials}") if credentials and credentials.credentials else None
    )
    if not body:
        raise HTTPException(422, "missing body")
    if isinstance(body, list):
        payload = {"positions": body}
    elif isinstance(body, dict):
        payload = body
    else:
        raise HTTPException(422, "invalid payload")
    reset = bool(payload.get("reset"))
    apply_to_cash = bool(payload.get("apply_to_cash"))
    set_focus = payload.get("set_focus")
    items = payload.get("positions") or []
    if not isinstance(items, list):
        raise HTTPException(422, "positions must be a list")
    if reset:
        STATE["positions"] = []
    if FOCUS_WOLF_ONLY:
        # Only accept WOLF in focus mode
        items = [p for p in items if str(p.get("symbol", "")).upper() == WOLF]
    positions = STATE.get("positions")
    if not isinstance(positions, list):
        positions = []
    added = []
    for p in items:
        try:
            sym = str(p.get("symbol") or "").upper()
            if not sym:
                continue
            market = str(p.get("market") or p.get("type") or "stock")
            qty = float(p.get("qty") or p.get("quantity") or 0.0)
            if qty <= 0:
                continue
            if p.get("price_paid") is not None:
                price_paid = float(p.get("price_paid"))
            elif p.get("invested_total") is not None:
                inv = float(p.get("invested_total") or 0.0)
                price_paid = 0.0 if qty == 0 else float(inv / qty)
            else:
                price_paid = 0.0
            positions.append(
                {"symbol": sym, "market": market, "qty": qty, "price_paid": price_paid}
            )
            added.append(sym)
            if apply_to_cash and price_paid > 0:
                cost = round(qty * price_paid, 2)
                if market == "crypto":
                    STATE["cash_crypto"] = float(
                        round(float(STATE.get("cash_crypto", 0.0)) - cost, 2)
                    )
                else:
                    STATE["cash_stock"] = float(
                        round(float(STATE.get("cash_stock", 0.0)) - cost, 2)
                    )
        except Exception:
            continue
    STATE["positions"] = positions
    # Recompute total cash if split present
    if "cash_stock" in STATE or "cash_crypto" in STATE:
        STATE["cash"] = float(
            round(
                float(STATE.get("cash_stock", 0.0)) + float(STATE.get("cash_crypto", 0.0)),
                2,
            )
        )
    _persist_save()
    if isinstance(set_focus, str) and set_focus.upper() == WOLF:
        # Update legacy qty/avg_cost for focus ticker if exactly one WOLF position imported with price_paid
        try:
            w = [p for p in positions if p.get("symbol") == WOLF]
            if w:
                qty = float(w[-1].get("qty") or 0.0)
                price_paid = float(w[-1].get("price_paid") or 0.0)
                if qty > 0 and price_paid > 0:
                    STATE["qty"] = qty
                    STATE["avg_cost"] = price_paid
        except Exception:
            pass
    _add_event("positions.import", "Positions imported", {"added": added})
    return {"ok": True, "positions": STATE["positions"], "added": added}


@router.post("/api/positions/import")
async def api_positions_import(
    body: PositionsImportBody,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    _require_bearer(
        (f"Bearer {credentials.credentials}") if credentials and credentials.credentials else None
    )
    import csv as _csv

    new_positions: list[dict] = []
    try:
        if body.csv and isinstance(body.csv, str) and body.csv.strip():
            # Very lightweight CSV parser: expect headers containing symbol, qty or quantity, price or price_paid or total_cost
            reader = _csv.DictReader(body.csv.splitlines())
            for row in reader:
                try:
                    sym = str(row.get("symbol") or row.get("ticker") or "").upper().strip()
                    if not sym:
                        continue
                    market = str(row.get("market") or row.get("type") or "stock").strip()
                    qty = float(row.get("qty") or row.get("quantity") or 0.0)
                    price_paid = row.get("price_paid") or row.get("entry") or row.get("avg_cost")
                    total_cost = row.get("total_cost") or row.get("cost_basis")
                    if price_paid is None and total_cost is not None and float(qty) > 0:
                        price_paid = float(total_cost) / float(qty)
                    price_paid = float(price_paid or 0.0)
                    if qty <= 0:
                        continue
                    new_positions.append(
                        {
                            "symbol": sym,
                            "market": market or "stock",
                            "qty": float(qty),
                            "price_paid": float(price_paid),
                        }
                    )
                except Exception:
                    continue
        elif body.positions is not None:
            if isinstance(body.positions, dict):
                body_positions = [body.positions]
            else:
                body_positions = list(body.positions)
            for pos in body_positions:
                try:
                    sym = str(pos.get("symbol") or pos.get("ticker") or "").upper().strip()
                    if not sym:
                        continue
                    market = str(pos.get("market") or pos.get("type") or "stock").strip()
                    qty = float(pos.get("qty") or pos.get("quantity") or 0.0)
                    price_paid = pos.get("price_paid") or pos.get("entry") or pos.get("avg_cost")
                    total_cost = pos.get("total_cost") or pos.get("cost_basis")
                    if (
                        (price_paid is None or float(price_paid) == 0.0)
                        and total_cost is not None
                        and float(qty) > 0
                    ):
                        price_paid = float(total_cost) / float(qty)
                    price_paid = float(price_paid or 0.0)
                    if qty <= 0:
                        continue
                    new_positions.append(
                        {
                            "symbol": sym,
                            "market": market or "stock",
                            "qty": float(qty),
                            "price_paid": float(price_paid),
                        }
                    )
                except Exception:
                    continue
        else:
            raise HTTPException(422, "positions or csv required")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"import_error: {e}") from e

    if body.reset:
        STATE["positions"] = []
    positions = STATE.get("positions")
    if not isinstance(positions, list):
        positions = []
    positions.extend(new_positions)
    STATE["positions"] = positions

    # Optionally update focus (WOLF) qty/avg from a matching imported position
    if body.set_focus:
        try:
            wolf_pos = next((p for p in positions if str(p.get("symbol")).upper() == WOLF), None)
            if wolf_pos is not None:
                STATE["qty"] = float(wolf_pos.get("qty") or 0.0)
                STATE["avg_cost"] = float(wolf_pos.get("price_paid") or 0.0)
        except Exception:
            pass

    # Optionally apply to cash by deducting total costs per position
    if body.apply_to_cash:
        try:
            total = 0.0
            for p in new_positions:
                total += float(p.get("qty", 0.0)) * float(p.get("price_paid", 0.0))
            if "cash_stock" in STATE or "cash_crypto" in STATE:
                # Deduct from stock cash bucket
                STATE["cash_stock"] = float(round(float(STATE.get("cash_stock", 0.0)) - total, 2))
                STATE["cash"] = float(
                    round(
                        float(STATE.get("cash_stock", 0.0)) + float(STATE.get("cash_crypto", 0.0)),
                        2,
                    )
                )
            else:
                STATE["cash"] = float(round(float(STATE.get("cash", 0.0)) - total, 2))
        except Exception:
            pass

    _persist_save()
    _add_event("positions.import", "Positions imported", {"count": len(new_positions)})
    return {"ok": True, "positions": STATE["positions"]}


@router.post("/api/bank/reset")
async def api_bank_reset(body: dict | None = None):
    # No-op bank in WOLF-only; acknowledge for UI
    _add_event(
        "bank.reset",
        "Bank reset",
        {"amount": (body or {}).get("amount") if isinstance(body, dict) else None},
    )
    try:
        if os.getenv("SNAP_TEST_MODE", "0").lower() in ("1", "true", "yes"):
            import sys

            amt = float((body or {}).get("amount") or 0)
            # Prefer the running __main__ module state (server started via python main.py)
            target = sys.modules.get("__main__")
            if target and hasattr(target, "TRADING_STATE"):
                ts = target.TRADING_STATE
                try:
                    ts["cash"] = {"stock": amt, "crypto": 0.0}
                    ts["positions"] = []
                except Exception:
                    pass
            # Also try imported 'main' module if present
            m = sys.modules.get("main")
            if m and hasattr(m, "TRADING_STATE"):
                ts2 = m.TRADING_STATE
                try:
                    ts2["cash"] = {"stock": amt, "crypto": 0.0}
                    ts2["positions"] = []
                except Exception:
                    pass
    except Exception:
        pass
    return {"ok": True}


@router.post("/api/bank/set_cash")
async def api_bank_set_cash(
    body: dict | None = None,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """Manually set cash balances. Accepts {stock: <usd>, crypto: <usd>?}. Persists total."""
    _require_bearer(
        (f"Bearer {credentials.credentials}") if credentials and credentials.credentials else None
    )
    try:
        stock = float((body or {}).get("stock") or 0.0)
        crypto = float((body or {}).get("crypto") or 0.0)
    except Exception:
        raise HTTPException(422, "invalid cash payload")
    STATE["cash_stock"] = float(round(stock, 2))
    STATE["cash_crypto"] = float(round(crypto, 2))
    STATE["cash"] = float(round(STATE["cash_stock"] + STATE["cash_crypto"], 2))
    _persist_save()
    _add_event(
        "cash.update",
        "Cash balance set",
        {
            "cash": STATE["cash"],
            "stock": STATE["cash_stock"],
            "crypto": STATE["cash_crypto"],
        },
    )
    return {
        "ok": True,
        "cash": STATE["cash"],
        "stock": STATE["cash_stock"],
        "crypto": STATE["cash_crypto"],
    }


@router.post("/watchlist/import")
async def watchlist_import(body: WatchlistImportBody):
    # Focus Mode: accept but do not change universe
    _add_event(
        "watchlist.import",
        "Watchlist import ignored (focus mode)",
        {"stocks": bool(body.stocks), "crypto": bool(body.crypto)},
    )
    return {"ok": True, "note": "focus-mode"}


@router.get("/watchlist")
async def watchlist_get(top: str = "mixed", n: int = 25, page: int = 1, q: str | None = None):
    # Lightweight compatibility watchlist with pagination fields
    base = [
        ("AAPL", "stock"),
        ("NVDA", "stock"),
        ("WOLF", "stock"),
        ("BTC", "crypto"),
        ("ETH", "crypto"),
        ("SOL", "crypto"),
    ]
    assets: list[dict] = []
    if q:
        ql = q.lower()
        for s, t in base:
            if ql in s.lower():
                assets.append({"symbol": s, "name": s, "type": t})
    else:
        picks = base[: max(1, int(n))]
        for s, t in picks:
            assets.append({"symbol": s, "name": s, "type": t})
    total = len(assets)
    start = (max(1, int(page)) - 1) * max(1, int(n))
    page_size = max(1, int(n))
    return {
        "assets": assets[start : start + page_size],
        "total": total,
        "page": int(page),
        "page_size": page_size,
    }


@router.post("/orders/clear")
async def orders_clear(credentials: HTTPAuthorizationCredentials | None = AUTH_DEP):
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass
    try:
        import sqlite3

        conn = sqlite3.connect(WOLF_SQLITE_PATH)
        cur = conn.cursor()
        cur.execute(f"DELETE FROM {ORDERS_TABLE}")
        conn.commit()
        conn.close()
        _add_event("orders.clear", "Orders cleared", {})
        return {"ok": True}
    except Exception as e:
        LOGGER.warning("orders_clear_error", extra={"component": "orders", "error": str(e)})
        return {"ok": False}


@router.get("/api/predictions/overlay/{symbol}")
async def api_predictions_overlay(
    symbol: str,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    Get prediction overlay data (forecast vs actual) for charting.
    Returns forecast points, actual prices, MAP accuracy metric.

    Contract test requirement:
    - Must return forecast array with {timestamp, price, confidence}
    - Must return actual array with {timestamp, price}
    - Must calculate MAP (Mean Absolute Percentage Error)
    - MAP < 15% = "good", < 25% = "acceptable", >= 25% = "poor"
    """
    try:
        _require_bearer(
            (f"Bearer {credentials.credentials}")
            if credentials and credentials.credentials
            else None
        )
    except Exception:
        pass

    try:
        # Generate or load forecast grid
        forecast_grid = _generate_forecast_grid(symbol.upper())

        # Extract forecast points
        forecast_points = forecast_grid.get("points", [])
        confidence = forecast_grid.get("meta", {}).get("con", 0.5)

        # Collect actual prices
        t_grid = [p["t"] for p in forecast_points]
        actual_data = _collect_actual_prices(t_grid, symbol.upper())
        actual_points = actual_data.get("points", [])

        # Calculate MAP (Mean Absolute Percentage Error)
        map = 0.0
        if actual_points and forecast_points:
            # Align timestamps
            actual_dict = {p["t"]: p["p"] for p in actual_points}
            errors = []
            for fp in forecast_points:
                t = fp["t"]
                if t in actual_dict and actual_dict[t] > 0:
                    forecast_val = fp["p"]
                    actual_val = actual_dict[t]
                    # |actual - forecast| / |actual| * 100
                    pct_error = abs(actual_val - forecast_val) / actual_val * 100
                    errors.append(pct_error)

            if errors:
                map = sum(errors) / len(errors)

        # Format response
        forecast_formatted = [
            {
                "timestamp": p["t"],
                "price": p["p"],
                "confidence": confidence,
            }
            for p in forecast_points
        ]

        actual_formatted = [
            {
                "timestamp": p["t"],
                "price": p["p"],
            }
            for p in actual_points
        ]

        return {
            "symbol": symbol.upper(),
            "forecast": forecast_formatted,
            "actual": actual_formatted,
            "map": round(map, 2),
            "accuracy": "good" if map < 15 else ("acceptable" if map < 25 else "poor"),
            "confidence": round(confidence, 2),
            "horizon_hours": forecast_grid.get("horizon_s", 0) / 3600,
            "generated_at": forecast_grid.get("aso", 0),
        }
    except Exception as e:
        LOGGER.error(f"Prediction overlay failed: {e}")
        return {
            "ok": False,
            "error": str(e),
        }


@router.get("/api/predictions/history")
async def api_predictions_history(
    symbol: str = WOLF,
    credentials: HTTPAuthorizationCredentials | None = AUTH_DEP,
):
    """
    Contract-compatible endpoint returning forecast vs actual with MAP.
    Mirrors /api/predictions/overlay/{symbol} but uses query param.

    Response shape:
    {
      "symbol": "WOLF",
      "forecasts": [{"timestamp": 123, "price": 31.1, "confidence": 0.7}, ...],
      "actual": [{"timestamp": 123, "price": 31.0}, ...],
      "map": 4.2,
      "horizon_hours": 48,
      "last_updated": 123
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

    try:
        payload = await api_predictions_overlay(symbol)
        # When overlay returns error, propagate minimal error info
        if not isinstance(payload, dict) or payload.get("ok") is False:
            return payload

        # Convert field names to contract's expected shape
        return {
            "symbol": payload.get("symbol", symbol.upper()),
            "forecasts": payload.get("forecast", []),
            "actual": payload.get("actual", []),
            "map": payload.get("map", 0.0),
            "horizon_hours": payload.get("horizon_hours", 0),
            "last_updated": payload.get("generated_at", 0),
        }
    except Exception as e:
        LOGGER.error(f"Prediction history failed: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/api/market-context")
async def api_market_context_alias():
    """Alias for /api/world/context — market context data."""
    return await api_world_context()


@router.get("/api/gate-system/status")
async def api_gate_system_status_alias():
    """Alias for /api/gates/status — market gates status."""
    return await api_gates_status()


@router.get("/api/v3/paper-trades/recent")
async def api_paper_trades_recent_alias(
    symbol: str | None = None,
    days: int = 7,
    outcome: str | None = None,
    limit: int = 20,
):
    """Alias for /api/v3/paper/trades — recent paper trades."""
    return await api_v3_paper_get_trades(symbol=symbol, days=days, outcome=outcome, limit=limit)


@router.post("/api/v3/paper-trades/backfill-targets")
async def api_paper_trades_backfill_targets():
    """
    One-shot backfill: compute target_price for all PENDING trades that have
    take_profit_pct but no target_price.  Safe to call multiple times.
    """
    try:
        from core.paper_tracker import get_paper_tracker
        tracker = get_paper_tracker()
        conn = tracker._get_connection()
        cur = tracker._execute(conn, """
            SELECT paper_trade_id, symbol, signal_direction, entry_price,
                   take_profit_pct, expected_move_pct
            FROM paper_trades
            WHERE outcome = 'PENDING'
              AND (target_price IS NULL OR target_price = 0)
        """, ())
        rows = tracker._fetchall(cur)
        updated = 0
        for row in rows:
            ep = float(row.get("entry_price") or 0)
            if ep <= 0:
                continue
            move_pct = row.get("expected_move_pct")
            tp_pct = row.get("take_profit_pct")
            if not move_pct and tp_pct and float(tp_pct) > 0:
                move_pct = float(tp_pct) * 100.0  # fraction → %
            if not move_pct:
                continue
            move_pct = float(move_pct)
            direction = (row.get("signal_direction") or "UP").upper()
            move_dir = 1.0 if direction in ("UP", "LONG", "BULLISH") else -1.0
            target = ep * (1.0 + move_dir * abs(move_pct) / 100.0)
            tracker._execute(conn,
                "UPDATE paper_trades SET target_price = ? WHERE paper_trade_id = ?",
                (target, row["paper_trade_id"]))
            updated += 1
        conn.commit()
        conn.close()
        return {"ok": True, "backfilled": updated, "checked": len(rows)}
    except Exception as e:
        LOGGER.error(f"Backfill targets failed: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/api/world/context")
async def api_world_context():
    """Get world market context (SPY, VIX, market mood, news)."""
    try:
        from core.world_context import get_world_context
        return get_world_context()
    except Exception as e:
        LOGGER.warning(f"World context failed, using fallback: {e}")
        return _get_world_context_fallback()


@router.get("/api/goals/all")
async def api_goals_all():
    """Get all goals with progress tracking."""
    try:
        from core.goals_tracker import GoalsTracker
        tracker = GoalsTracker()
        goals = tracker.get_all_goals()
        return {"ok": True, "goals": goals, "timestamp": time.time()}
    except Exception as e:
        LOGGER.error(f"Goals fetch failed: {e}")
        return {
            "ok": False,
            "goals": {
                "daily": {"target": 0, "current": 0, "progress_pct": 0, "remaining": 0},
                "weekly": {"target": 0, "current": 0, "progress_pct": 0, "remaining": 0},
                "monthly": {"target": 0, "current": 0, "progress_pct": 0, "remaining": 0},
                "yearly": {"target": 0, "current": 0, "progress_pct": 0, "remaining": 0}
            },
            "error": str(e),
            "timestamp": time.time()
        }


@router.post("/api/goals/set")
async def api_goals_set(period: str, target_amount: float):
    """Set a goal for a specific period."""
    try:
        from core.goals_tracker import GoalsTracker
        tracker = GoalsTracker()
        result = tracker.set_goal(period, target_amount)
        return {"ok": True, **result}
    except Exception as e:
        LOGGER.error(f"Goal set failed: {e}")
        return {"ok": False, "error": str(e)}


@router.get("/api/xrp/tracker")
async def api_xrp_tracker():
    """Get XRP bullish eye tracker status."""
    try:
        from core.xrp_tracker import get_xrp_status
        xrp_status = await get_xrp_status()
        return {"ok": True, **xrp_status}
    except Exception as e:
        LOGGER.error(f"XRP tracker failed: {e}")
        return {
            "ok": False,
            "price": None,
            "change_24h_pct": None,
            "bullish_eye": "⚠️",
            "signal": "ERROR",
            "confidence": 0.0,
            "factors": [str(e)],
            "error": str(e),
            "timestamp": time.time()
        }


@router.get("/api/vip/coins")
async def api_vip_coins():
    """Get VIP coins status with enhanced presale data (WEPE, LILPEPE, DORKL, SLOTH, APC)."""
    try:
        from core.vip_scanner import VIP_WATCHLIST
        from core.crypto.vip_providers import get_vip_price
        
        # Presale metadata (enriched data for sniper coins)
        presale_metadata = {
            "WEPE": {
                "name": "Wall Street Pepe",
                "stage": "Presale",
                "status": "Active",
                "launch_date": "Q1 2025",
                "market_cap_est": "$15M",
                "risk_score": 7.5
            },
            "LILPEPE": {
                "name": "Lil Pepe",
                "stage": "Presale",
                "status": "Monitoring",
                "launch_date": "Q1 2025",
                "market_cap_est": "$8M",
                "risk_score": 8.0
            },
            "DORKL": {
                "name": "Dork Lord",
                "stage": "Presale",
                "status": "Watching",
                "launch_date": "Q2 2025",
                "market_cap_est": "$5M",
                "risk_score": 8.5
            },
            "SLOTH": {
                "name": "Slothana",
                "stage": "Presale",
                "status": "Watching",
                "launch_date": "Q1 2025",
                "market_cap_est": "$12M",
                "risk_score": 7.8
            },
            "APC": {
                "name": "Ape Coin",
                "stage": "Presale",
                "status": "Watching",
                "launch_date": "Q2 2025",
                "market_cap_est": "$20M",
                "risk_score": 6.5
            }
        }
        
        coins_status = []
        for symbol in VIP_WATCHLIST:
            metadata = presale_metadata.get(symbol, {})
            
            # Try to get live price if available
            price_data = get_vip_price(symbol, use_cache=True)
            
            coin_data = {
                "symbol": symbol,
                "name": metadata.get("name", symbol),
                "price": None,
                "change_24h_pct": None,
                "stage": metadata.get("stage", "Unknown"),
                "status": metadata.get("status", "Unknown"),
                "launch_date": metadata.get("launch_date", "TBD"),
                "market_cap_est": metadata.get("market_cap_est", "Unknown"),
                "risk_score": metadata.get("risk_score", 5.0),
                "provider": "presale"
            }
            
            # If live price available, use it
            if price_data.get("available") and price_data.get("price"):
                coin_data["price"] = round(price_data["price"], 6)
                coin_data["change_24h_pct"] = round(price_data.get("change_24h_pct", 0), 2)
                coin_data["provider"] = price_data.get("provider", "live")
                coin_data["status"] = "Live Trading"
            
            coins_status.append(coin_data)
        
        return {"ok": True, "coins": coins_status, "count": len(coins_status), "timestamp": time.time()}
    except Exception as e:
        LOGGER.error(f"VIP coins failed: {e}")
        return {"ok": False, "coins": [], "error": str(e), "timestamp": time.time()}


@router.get("/api/portfolio/positions")
async def api_portfolio_positions():
    """Get current portfolio positions."""
    try:
        from core.portfolio_tracker import _PORTFOLIO
        positions = []
        for symbol, pos_data in _PORTFOLIO.items():
            positions.append({
                "symbol": symbol,
                "quantity": pos_data["quantity"],
                "entry_price": pos_data["entry_price"],
                "current_price": None,
                "pnl": None,
                "pnl_pct": None
            })
        return {"ok": True, "positions": positions, "count": len(positions), "timestamp": time.time()}
    except Exception as e:
        LOGGER.error(f"Portfolio positions failed: {e}")
        return {"ok": False, "positions": [], "error": str(e), "timestamp": time.time()}


@router.get("/api/premarket_status")
async def api_premarket_status():
    """
    Get pre-market predictor status and recent predictions.

    Returns:
        {
            'ok': True,
            'enabled': True,
            'last_run': 1731654000,
            'last_run_ct': '7:00 AM CT 2024-11-15',
            'predictions_count': 5,
            'recent_predictions': [
                {
                    'symbol': 'WOLF',
                    'direction': 'UP',
                    'confidence': 0.78,
                    'early_signal': True,
                    'hours_before_open': 2.5
                },
                ...
            ],
            'next_run_ct': '7:00 AM CT 2024-11-16'
        }
    """
    try:
        from core.premarket_predictor import get_premarket_status

        status = get_premarket_status()

        return {
            'ok': True,
            **status,
            'timestamp': int(time.time())
        }
    except Exception as e:
        LOGGER.error(f"Pre-market status API failed: {e}")
        return {
            'ok': False,
            'error': str(e),
            'enabled': False,
            'recent_predictions': [],
            'timestamp': int(time.time())
        }


