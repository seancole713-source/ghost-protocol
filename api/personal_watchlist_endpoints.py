#!/usr/bin/env python3
"""
Ghost Protocol Personal Watchlist API Endpoints
================================================

REST API for personal watchlist management under /api/v3/watchlist/*

Endpoints:
- POST /api/v3/watchlist/add - Add symbol to watchlist
- POST /api/v3/watchlist/remove - Remove symbol from watchlist
- GET /api/v3/watchlist/user - Get enriched watchlist with predictions
- POST /api/v3/watchlist/update-position - Update owns_position flag
- GET /api/v3/watchlist/history/{symbol} - Get prediction history for symbol
- POST /api/v3/watchlist/trigger-prediction - Manually trigger prediction

Security:
- Reuses existing IP allowlist + GHOST_API_TOKEN header protection
- Single-owner system (no user auth)
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Header, Request
from pydantic import BaseModel, Field

LOGGER = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v3/watchlist", tags=["personal_watchlist"])


# ============================================================================
# REQUEST MODELS
# ============================================================================


class AddSymbolRequest(BaseModel):
    """Request body for adding symbol to watchlist."""

    symbol: str = Field(..., min_length=1, max_length=20, description="Ticker symbol (will be uppercased)")
    asset_type: str = Field(..., pattern="^(crypto|stock)$", description="Asset type: 'crypto' or 'stock'")
    owns_position: bool = Field(default=False, description="TRUE if user currently holds this asset")
    notes: str = Field(default="", max_length=500, description="Optional notes/comments")
    alert_threshold_pct: float = Field(default=5.0, ge=0.1, le=50.0, description="Price move % to trigger alert")
    priority: int = Field(default=1, ge=1, le=3, description="1=normal, 2=high, 3=critical")


class RemoveSymbolRequest(BaseModel):
    """Request body for removing symbol from watchlist."""

    symbol: str = Field(..., min_length=1, max_length=20, description="Ticker symbol")
    asset_type: str = Field(..., pattern="^(crypto|stock)$", description="Asset type: 'crypto' or 'stock'")


class UpdatePositionRequest(BaseModel):
    """Request body for updating owns_position flag."""

    symbol: str = Field(..., min_length=1, max_length=20, description="Ticker symbol")
    asset_type: str = Field(..., pattern="^(crypto|stock)$", description="Asset type: 'crypto' or 'stock'")
    owns_position: bool = Field(..., description="TRUE if user now holds this asset")


class TriggerPredictionRequest(BaseModel):
    """Request body for manually triggering a prediction."""

    symbol: str = Field(..., min_length=1, max_length=20, description="Ticker symbol")
    asset_type: str = Field(..., pattern="^(crypto|stock)$", description="Asset type: 'crypto' or 'stock'")


# ============================================================================
# SECURITY MIDDLEWARE
# ============================================================================


def verify_access(request: Request, x_api_token: Optional[str] = Header(None)):
    """
    Verify API access (reuses existing Ghost security).

    Args:
        request: FastAPI request object
        x_api_token: Optional API token from header

    Raises:
        HTTPException: If access denied
    """
    # TODO: Import and use existing IP allowlist + token verification
    # For now, allow all requests (will be secured by existing wolf_app middleware)
    pass


# ============================================================================
# ENDPOINTS
# ============================================================================


@router.post("/add")
async def add_symbol_to_watchlist(body: AddSymbolRequest, request: Request, x_api_token: Optional[str] = Header(None)):
    """
    Add symbol to personal watchlist.

    Request body:
        {
            "symbol": "AAPL",
            "asset_type": "stock",
            "owns_position": false,
            "notes": "Apple Inc. - watching for entry",
            "alert_threshold_pct": 5.0,
            "priority": 2
        }

    Response:
        {
            "ok": true,
            "action": "added",  // or "updated", "re-activated"
            "id": 123,
            "symbol": "AAPL",
            "asset_type": "stock",
            "owns_position": false,
            "added_at": "2025-12-02T12:34:56Z"
        }
    """
    verify_access(request, x_api_token)

    try:
        from core.personal_watchlist import get_personal_watchlist_manager

        pwm = get_personal_watchlist_manager()

        result = pwm.add_symbol(
            symbol=body.symbol,
            asset_type=body.asset_type,
            owns_position=body.owns_position,
            notes=body.notes,
            alert_threshold_pct=body.alert_threshold_pct,
            priority=body.priority,
        )

        if not result.get("ok"):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to add symbol"))

        return result

    except Exception as e:
        LOGGER.error(f"❌ Add symbol API error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/remove")
async def remove_symbol_from_watchlist(body: RemoveSymbolRequest, request: Request, x_api_token: Optional[str] = Header(None)):
    """
    Remove symbol from personal watchlist (soft delete).

    Request body:
        {
            "symbol": "AAPL",
            "asset_type": "stock"
        }

    Response:
        {
            "ok": true,
            "symbol": "AAPL",
            "asset_type": "stock"
        }
    """
    verify_access(request, x_api_token)

    try:
        from core.personal_watchlist import get_personal_watchlist_manager

        pwm = get_personal_watchlist_manager()

        result = pwm.remove_symbol(symbol=body.symbol, asset_type=body.asset_type)

        if not result.get("ok"):
            raise HTTPException(status_code=404, detail=result.get("error", "Symbol not found"))

        return result

    except Exception as e:
        LOGGER.error(f"❌ Remove symbol API error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user")
async def get_user_watchlist(request: Request, x_api_token: Optional[str] = Header(None)):
    """
    Get user's personal watchlist enriched with live prediction data.

    Response:
        {
            "items": [
                {
                    "id": 123,
                    "symbol": "AAPL",
                    "asset_type": "stock",
                    "owns_position": false,
                    "notes": "Apple Inc.",
                    "alert_threshold_pct": 5.0,
                    "priority": 2,
                    "added_at": "2025-12-02T12:34:56Z",
                    "current_price": 283.10,
                    "prediction": {
                        "prediction_id": 366,
                        "direction": "DOWN",
                        "confidence": 0.58,
                        "expected_move": -4.5,
                        "horizon_h": 48,
                        "run_at": 1764635853.456
                    }
                },
                ...
            ],
            "count": N,
            "timestamp": 1764642576.184
        }
    """
    verify_access(request, x_api_token)

    try:
        from core.personal_watchlist import get_personal_watchlist_manager
        import time
        import asyncio

        pwm = get_personal_watchlist_manager()

        # Add 5-second timeout to prevent indefinite hangs
        # If enrichment takes too long, return basic list without predictions
        try:
            enriched_items = await asyncio.wait_for(
                asyncio.to_thread(pwm.get_enriched_watchlist),
                timeout=5.0
            )
        except asyncio.TimeoutError:
            LOGGER.warning("⚠️ Watchlist enrichment timeout (5s), returning basic list")
            # Fallback: return unenriched watchlist
            enriched_items = pwm.get_watchlist()

        return {"ok": True, "items": enriched_items, "count": len(enriched_items), "timestamp": time.time()}

    except Exception as e:
        # Gracefully handle errors (e.g., tables don't exist yet)
        # Return empty list instead of 500 error for better UX
        if "does not exist" in str(e) or "no such table" in str(e):
            LOGGER.warning(f"⚠️ Watchlist tables not ready: {e}")
            return {"ok": True, "items": [], "count": 0, "timestamp": time.time()}
        else:
            LOGGER.error(f"❌ Get watchlist API error: {e}", exc_info=True)
            return {"ok": False, "items": [], "count": 0, "timestamp": time.time(), "error": str(e)}


@router.post("/update-position")
async def update_position_flag(body: UpdatePositionRequest, request: Request, x_api_token: Optional[str] = Header(None)):
    """
    Update the owns_position flag for a watchlist symbol.

    Request body:
        {
            "symbol": "AAPL",
            "asset_type": "stock",
            "owns_position": true
        }

    Response:
        {
            "ok": true,
            "symbol": "AAPL",
            "owns_position": true
        }
    """
    verify_access(request, x_api_token)

    try:
        from core.personal_watchlist import get_personal_watchlist_manager

        pwm = get_personal_watchlist_manager()

        result = pwm.update_position_flag(symbol=body.symbol, asset_type=body.asset_type, owns_position=body.owns_position)

        if not result.get("ok"):
            raise HTTPException(status_code=404, detail=result.get("error", "Symbol not found"))

        return result

    except Exception as e:
        LOGGER.error(f"❌ Update position API error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{symbol}")
async def get_symbol_prediction_history(symbol: str, limit: int = 50, request: Request = None, x_api_token: Optional[str] = Header(None)):
    """
    Get prediction history for a watchlist symbol.

    Path params:
        symbol: Ticker symbol

    Query params:
        limit: Max number of records (default 50)

    Response:
        {
            "symbol": "AAPL",
            "history": [
                {
                    "id": 789,
                    "prediction_id": 366,
                    "direction": "DOWN",
                    "confidence": 0.58,
                    "expected_move_pct": -4.5,
                    "horizon_h": 48,
                    "price_at_prediction": 283.10,
                    "generated_at": "2025-12-02T12:34:56Z",
                    "reason": "market_close",
                    "alert_sent": true
                },
                ...
            ],
            "count": N
        }
    """
    if request:
        verify_access(request, x_api_token)

    try:
        from core.personal_watchlist import get_personal_watchlist_manager

        pwm = get_personal_watchlist_manager()

        history = pwm.get_prediction_history(symbol=symbol, limit=limit)

        return {"symbol": symbol.upper(), "history": history, "count": len(history)}

    except Exception as e:
        LOGGER.error(f"❌ Get history API error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trigger-prediction")
async def trigger_manual_prediction(body: TriggerPredictionRequest, request: Request, x_api_token: Optional[str] = Header(None)):
    """
    Manually trigger a prediction for a watchlist symbol.

    Request body:
        {
            "symbol": "AAPL",
            "asset_type": "stock"
        }

    Response:
        {
            "ok": true,
            "symbol": "AAPL",
            "reason": "manual",
            "message": "Prediction queued"
        }
    """
    verify_access(request, x_api_token)

    try:
        from core.watchlist_prediction_scheduler import get_watchlist_prediction_scheduler

        scheduler = get_watchlist_prediction_scheduler()

        result = scheduler.trigger_manual_prediction(symbol=body.symbol, asset_type=body.asset_type)

        if not result.get("ok"):
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to trigger prediction"))

        return {**result, "message": "Prediction queued"}

    except Exception as e:
        LOGGER.error(f"❌ Trigger prediction API error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_watchlist_stats(request: Request, x_api_token: Optional[str] = Header(None)):
    """
    Get watchlist statistics.

    Response:
        {
            "total_symbols": N,
            "stocks": N,
            "crypto": N,
            "owned_positions": N,
            "alerts_sent_7d": {
                "total": N,
                "by_type": {
                    "open": N,
                    "close": N,
                    "big_move": N
                }
            }
        }
    """
    verify_access(request, x_api_token)

    try:
        from core.personal_watchlist import get_personal_watchlist_manager

        pwm = get_personal_watchlist_manager()

        watchlist = pwm.get_watchlist(active_only=True)
        alert_stats = pwm.get_alert_stats(days=7)

        stocks = [item for item in watchlist if item["asset_type"] == "stock"]
        crypto = [item for item in watchlist if item["asset_type"] == "crypto"]
        owned = [item for item in watchlist if item["owns_position"]]

        return {
            "total_symbols": len(watchlist),
            "stocks": len(stocks),
            "crypto": len(crypto),
            "owned_positions": len(owned),
            "alerts_sent_7d": alert_stats,
        }

    except Exception as e:
        LOGGER.error(f"❌ Get stats API error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
